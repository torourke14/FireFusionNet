from processor import Processor


class UsfsFire(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        return
    
    def _open_data(self, feat_key: str):
        raise NotImplementedError
    
    def _extract_feature(self, feat_key: str):
        raise NotImplementedError
    
class MtbsFire(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        return
    
    def _open_data(self, feat_key: str):
        raise NotImplementedError
    
    def _extract_feature(self, feat_key: str):
        raise NotImplementedError
    
class FireCci(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        return
    
    def _open_data(self, feat_key: str):
        raise NotImplementedError
    
    def _extract_feature(self, feat_key: str):
        raise NotImplementedError