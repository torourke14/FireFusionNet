# FireFusion
A Spatio-Temporal ConvFormer neural model utilizing to detect ignition & cause of wildfire ignition in WA state, utilizing public datasets only.

- `python -m fire_fusion.dataset.build`: Builds model-ready feature/label sets from downloaded data (See data/README.md for reproducing)
- `python -m fire_fusion.train`: Uses built .zarr file tensor to train the model

### Data Sources
- **Climatology**: Landfire, National Landcover Database (NLCD), Gridded Population of the World (GPWv4), MCD15A2H (NASA LAADS), MOD13Q1 (NASA LAADS), gridMET (Climatology Lab)
- **Fire labeling**: MCD64A1 (NASA LAADS), USFS Fire Occurence Point and Fire Perimeter Layers
See *data/sources/READ_ME.md* and *FeatureOverview.xlsx* for data point usage

### Model
- Spatial CNN Encoder (ResNet MLP)
- Sequence of dedicated SA Transformers targeting (1) spatial (2) feature, and (3) temporal relationships.
- Upsampling decoder predicts probability *cell(i, j, t)* transitions to "ignition" at *(i, j, t + K - 1)*.
- Provides strict masking of water features, active fires, and active fire causes to focus loss on real-time priors.

### Citation
TBD!

# Download/Reproduction Instructions
1. Create a conda environment (*conda create --name <my-env> python=3.11*)
2. Install all dependencies under requirements.txt (conda install <module_name>==<version>)
   - Some requirements may require conda-forge (conda install -c conda-forge <module_name>):
   - Some requirements may require pip install
3. Download data for non-API features (all except NASA Modis)
   - *Ensure paths in* `path_config` *match*
4. Run `python -m fire_fusion.build.py`
   - Model is optimized on many fronts for WA state, feel free to adjust params
5. Run `python -m fire_fusion.train.py` (Check first for .zarr files under `data/train`, `data/eval`, `eval/test`)
