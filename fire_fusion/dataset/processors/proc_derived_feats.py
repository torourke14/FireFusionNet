



import xarray as xr
import numpy as np
from typing import Tuple
from scipy.ndimage import maximum_filter

from fire_fusion.config.feature_config import CAUSAL_CLASSES, CAUSE_RAW_MAP

class DerivedProcessor:
    def __init__(self, drv_features, labels, masks, mgrid):
        self.drv_features = drv_features
        self.labels = labels
        self.masks = masks
        self.gridref = mgrid

        
    def derive_features(self, ds: xr.Dataset, 
        sp_burn_kernel = 3, sp_burn_window = 3
    ) -> Tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
        """
            label: fire_new_tplus1 : burning at t+1 AND not burning at t
                # if model is having trouble learning fire last 7 + wind direction
                # self.DS["fire_upw_neighbor"]  = self._compute_t_last_burn()
        """
        modis = (ds["modis_burn"].fillna(0) > 0)
        usfs_o = (ds["usfs_burn"].fillna(0) > 0)
        usfs_p = (ds["usfs_perimeter"].fillna(0) >= 0)
        
        
        # --- FIRE T + 1 LABEL --------------------------------------------------
        burning_t = (modis | usfs_o | usfs_p)
        burning_tp1 = burning_t.shift(time=-1, fill_value=0)
        ignition_next, burn_loss_mask = self._ignition_next(burning_t, burning_tp1)

        # --- FIRE CAUSE LABEL ----------------------------------------------------
        ign_next_cause, valid_cause_mask = self._ign_next_cause(ds['usfs_burn_cause'], ignition_next)

        ds = ds.assign(**{
            self.labels[0].name: ignition_next,    # ign_next
            self.labels[1].name: ign_next_cause,   # ign_next_cause
            self.masks[0].name: burn_loss_mask,    # act_fire_mask
            self.masks[1].name: valid_cause_mask   # v_cause_mask
            # water mask added in extraction       # water_mask
        })

        for deriv_feat in self.drv_features:
            name = deriv_feat.name

            try:
                if deriv_feat.key == "fire_spatial_roll":
                    burn_spatial_rolling = self._compute_fire_spatial_recent(burning_t, sp_burn_kernel, sp_burn_window, name)
                    ds = ds.assign({ name: burn_spatial_rolling })

                elif deriv_feat.key == "precip_2d": 
                    p5d = (ds['precip']
                        .rolling(time=2, min_periods=None, center=deriv_feat.agg_center)
                        .mean().fillna(0)
                    )
                    ds = ds.assign({ name: p5d })

                elif deriv_feat.key == "precip_4d":
                    p14d = (ds['precip']
                        .rolling(time=4, min_periods=None, center=deriv_feat.agg_center)
                        .mean().fillna(0)
                    )
                    ds = ds.assign({ name: p14d })

                elif deriv_feat.key == "ndvi_anomaly": 
                    ndvi_by_doy = ds['modis_ndvi'].groupby("time.dayofyear")
                    ndvi_anom = ndvi_by_doy.mean("time") - ndvi_by_doy
                    ndvi_anom.name = name
                    ds = ds.assign({ name: ndvi_anom })

                elif deriv_feat.key == "ffwi": 
                    fosberg_fwi = self._compute_fosberg_fwi(Tf=ds['temp_avg'], sH=ds['rh_pct'], Ws=ds['wind_mph'])
                    ds = ds.assign({ name: fosberg_fwi })

                elif deriv_feat.key == "doy_sin":
                    theta = 2 * np.pi * (ds['time'].dt.dayofyear / 365.0)
                    arr1d = xr.DataArray(theta, dims=['time'], coords={ 'time': ds.time})
                    arr2d = arr1d[:, None, None]
                    ds = ds.assign({ 'doy_sin': np.sin(arr2d), 'doy_cos': np.cos(arr2d) })
            except Exception as e:
                print(f"Feature extraction for {name} failed: ", e)
                continue

        return ds, burn_loss_mask, ds['water_mask']

    def _ignition_next(self, burning_t: xr.DataArray, burning_tp1: xr.DataArray):
        ignition_next_label = xr.DataArray(
            ((burning_t == 0) & (burning_tp1 == 1)).astype("uint8"),
            dims=burning_t.dims,
            coords=burning_t.coords,
            name=self.drv_features[0].name
        )
        active_burn_mask = (burning_t == 0).astype("uint8")
        return ignition_next_label, active_burn_mask


    def _ign_next_cause(self, burn_cause_t: xr.DataArray, ignition_next: xr.DataArray):
        # one hot >> fill -1's >> shift forward
        cause_t_oh = burn_cause_t.argmax(dim="burn_cause")
        cause_t_oh = xr.where(burn_cause_t.sum(dim="burn_cause") > 0, cause_t_oh, -1)
        cause_tp1_oh = cause_t_oh.shift(time=-1, fill_value=-1)

        # !!! next cause = cause @t+1 conditioned on ignition @t+1
        ign_next_cause = xr.where(ignition_next == 1, cause_tp1_oh, -1)
        # !!! Mask = 1 everywhere we have a "valid cause" @t+1, 0 otherwise
        valid_cause_mask = (cause_tp1_oh >= 0).astype("uint8")

        return ign_next_cause, valid_cause_mask


    def _compute_fire_spatial_recent(self,
        burning_t: xr.DataArray,
        spatial_kernel_size, 
        spatial_window_size,
        name
    ) -> xr.DataArray:
        """ 3x3 kernel max of active fires at time = T
            (ie, is there an active fire next to me?)
        """
        burn_rolling = burning_t.rolling(
            time=spatial_window_size, 
            min_periods=1
        ).max().fillna(0)
        
        burn_filter = maximum_filter(
            input=burn_rolling.values.astype(np.uint8),
            size=(1, spatial_kernel_size, spatial_kernel_size)
        )
        
        return xr.DataArray(
            data=burn_filter,
            dims=burn_rolling.dims,
            coords=burn_rolling.coords,
            name=name,
        )
    

    def _compute_fosberg_fwi(self,
        Tf: xr.DataArray,
        sH: xr.DataArray,
        Ws: xr.DataArray
    ) -> xr.DataArray:
        """
        FFWI = n sqrt(1 + U^2) / 0.3002, where
        - n = 1 - 2x + 1.5x^2 - 0.5x^3
        - x = EMC/30
        - EMC:
            if H < 10%:     EMC = 0.03229 + (0.281073 * H) - (0.000578 * H% & T(Far))
            if H in 10-50%: EMC = 2.22749 + (0.160107 * H) - (0.01478 * T(Far)) 
            if H >= 50%:    EMC = 21.0606 + (0.005565 * H^2) - (0.00035 * H * T(Far)) - (0.483199 * H%) 
        """
        EMC_p1   = 0.03229 + (0.281073 * sH) - (0.000578 * sH * Tf)
        EMC_p1p5 = 2.22749 + (0.160107 * sH) - (0.01478 * Tf) 
        EMC_p5   = 21.0606 + (0.005565 * (sH ** 2)) - (0.00035 * sH * Tf) - (0.483199 * sH)

        EMC: xr.DataArray = xr.where(sH < 0.1, EMC_p1, xr.where(sH < 0.5, EMC_p1p5, EMC_p5))
        x = EMC.clip(0.0, 30.0) / 30
        eta = 1 - (2.0 * x) + 1.5 * (x ** 2) - 0.5 * (x ** 3)
        
        ffwi = eta * np.sqrt(1.0 + Ws ** 2) / 0.3002
        return ffwi
    

    def _compute_wui(self,
        pop_norm: xr.DataArray, 
        lcov_class: xr.DataArray,
        wildland_ixs = [4, 5, 7]
    ) -> xr.DataArray:
        """
            WUI = 1 if any wildlife class
        """
        wild_ohe = lcov_class[..., list(wildland_ixs)]

        # Reduce over the ohe to get a [0, 1] wildland value on 2d grid
        wildland_mask = wild_ohe.max(dim="lc_class_index")  

        # population is HEAVILY skewed towards cities
        # add sigmoid to norm'd population to further smooth out non-linearity
        # smoothed WUI as sigmopoid of norm'd population * wildland mask

        wui_smooth = wildland_mask * (1.0 / (1.0 + np.exp(-1.0 * pop_norm)))
        wui_smooth.name = "wui_smooth"
        return wui_smooth
        
    
    