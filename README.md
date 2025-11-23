# FireFusionNet
Spatial Temporal CFormer Model that computes risk of a wildfire ignition using data from a host of public climate and fire datasets. Train and optimized for a H x W grid of Washington State.
- Clipped to Washington State bounding box (lat: 40 to 50, lon: -125 to -110)

### Model
- Spatial CNN Encoder (ResNet MLP-style)
- Sequential self-attention modules targeting (1) generalized spatial windows, (2) feature channels, and (3) temporal aspects
- Upsampling decoder to predict if *cell(i, j)* transitions to ignition (5% burn threshold) at *(t, t+K-1)*. Actively burning cells are masked to prevent model lookahead.
- Trained on 2km x 2km grid cells w/Binary-cross entropy loss


### Features & Data
- **Climatology**: Landfire, National Landcover Database (NLCD), Resolve Ecoregions 2017 (RESOLVE), Fire_CCI, Gridded Population of the World (GPWv4, NASA), MCD15A2H, MOD13Q1, (MODIS), gridMET (Climatology Lab)
- **Fire labeling**: MCD64A1 Burn Dates, USFS Fire Occurence Point Feature Layer (USFS-OPFL), 
See *data/sources/READ_ME.md* and *FeatureOverview.xlsx* for data point usage

### Citation
TBD!

# Download Instructions
1. Create a conda environment (*conda create --name <my-env> python=3.11*)
2. Install all dependencies under requirements.txt (conda install <module_name>==<version>)
   - Some requirements may require conda-forge (conda install -c conda-forge <module_name>):
   - Some requirements may require pip install
