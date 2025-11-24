# FireFusion
Novel CNN/Transformer mixture neural model that computes risk of wildfire ignition in WA state. Aggregates data from multiple public datasets/APIs to build and train on a rasterized feature matrix.

- `build.py`: Builds model-ready feature set from downloaded data (Given proper download scheme)
- `train.py`: Uses built .zarr file tensor to train the model

### Data Sources
- **Climatology**: Landfire, National Landcover Database (NLCD), Gridded Population of the World (GPWv4), MCD15A2H, MOD13Q1, gridMET (Climatology Lab)
- **Fire labeling**: MCD64A1 (NASA, burndDate/severity), USFS Fire Occurence Point FL (burn date/causes), USFS Fire PL (burn perimeter)
See *data/sources/READ_ME.md* and *FeatureOverview.xlsx* for data point usage

### Model
- Spatial CNN Encoder (ResNet MLP)
- Sequence of self-attention blocks targeting (1) generalized spatial windows -> (2) feature channels -> and (3) -> temporal aspects
- Upsampling decoder to predict if *cell(i, j)* transitions to ignition (5% burn threshold) at *(t, t+K-1)*. Actively burning cells are masked to prevent model lookahead.
- Trained on 2km x 2km grid cells w/Binary-cross entropy loss

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
