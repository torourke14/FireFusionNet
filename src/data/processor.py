from data.feature_builder import FeatureBuilder


class Processor:
    def __init__(self, cfg: FeatureBuilder, master_grid):
        self.cfg = cfg
        self.master_grid = master_grid

    def _open_data(self, feat_key: str):
        raise NotImplementedError
    
    def _extract_feature(self, feat_key: str):
        raise NotImplementedError