from pathlib import Path
from pyhdf.SD import SD, SDC
import json
import numpy as np

from osgeo import gdal, osr

# def _load_sds_from_gdal(hdf_path: Path) -> np.ndarray:
#     # Open HDF4 container
#     ds = gdal.Open(hdf_path)
#     if ds is None:
#         raise RuntimeError(f"Could not open {hdf_path}")
#     # List subdatasets (each MODIS SDS is exposed as a GDAL subdataset)
#     sub_ds = ds.GetSubDatasets()
#     # subdatasets is a list of (name, description) tuples
#     name, _ = sub_ds[0]
#     sds_ds = gdal.Open(name)
#     band = sds_ds.GetRasterBand(1)
#     arr = band.ReadAsArray()
#     return arr

if __name__ == "__main__":
    # sd1 = SD("MOD13Q1.A2025305.h08v04.061.2025322123302.hdf", SDC.READ)
    # sd2 = SD("MYD11A1.A2025309.h09v04.061.2025311200737.hdf", SDC.READ)
    # sd3 = SD("MCD15A2H.A2025305.h10v04.061.2025317085306.hdf", SDC.READ)
    # sd4 = SD("MCD64A1.A2008183.h11v03.006.2017017073149.hdf", SDC.READ)

    # test = sd.attributes()

    # print(sd1.datasets().keys())
    # print("\n")
    # print(sd2.datasets().keys())
    # print("\n")
    # print(sd3.datasets().keys())
    # print("\n")
    # print(sd4.datasets().keys())
    # print("\n")

    # print(sd1)

    # print("\n")
    # print(sd2.datasets())
    # print("\n")
    # print(sd3.datasets())
    # print("\n")
    # print(sd4.datasets())
    # print("\n")

    # print(sd.select(""))
    
    # sds = sd.select("250m 16 days NDVI")
    # for k in test.keys():
    #     print(test[k])
    
    # out_path = "sample_hdf_myd13q1.json"
    # with open(out_path, "w") as f:
    #     json.dump(test, f, indent=2)

    ds = gdal.Open("MOD13Q1.A2025305.h08v04.061.2025322123302.hdf")
    # Choose the NDVI SDS subdataset
    ndvi_sds_name = [s for s in ds.GetSubDatasets() 
                    if "250m 16 days NDVI" in s[0]][0][0]
    ndvi_ds = gdal.Open(ndvi_sds_name)
    band = ndvi_ds.GetRasterBand(1)
    ndvi = band.ReadAsArray().astype(np.float32)

    gt = ndvi_ds.GetGeoTransform()
    proj = ndvi_ds.GetProjection()

    # Build pixel-center coordinates in projected (sinusoidal) space
    nx = ndvi_ds.RasterXSize
    ny = ndvi_ds.RasterYSize

    xs = np.arange(nx) * gt[1] + gt[0] + gt[1] / 2.0
    ys = np.arange(ny) * gt[5] + gt[3] + gt[5] / 2.0
    X, Y = np.meshgrid(xs, ys)

    # Transform to lat/lon (WGS84)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(proj)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)

    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    lon = np.empty_like(X, dtype=np.float64)
    lat = np.empty_like(Y, dtype=np.float64)

    # Vectorize transform for speed in real code
    for i in range(ny):
        for j in range(nx):
            lon[i, j], lat[i, j], _ = ct.TransformPoint(X[i, j], Y[i, j])
            print(lon[i, j], lat[i, j])
