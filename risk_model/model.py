import torch
import torch.nn as nn
from .modules import SpatialEncoder, WindowedSpatialAttention, ChannelMixingAttention, TemporalMixingAttention, SpatialTemporalDecoder




class FireFusionModel(nn.Module):
    """
    Given feature channels (C) and timesteps (T), compute the risk of wildfire ignition across a H x W grid.

    Steps:
        - Encode (downsample) Spatial patterns with basic ResNet MLP-style CNN encoder
        - Run self-attention over larger HxW windows than Encoder (these generalize well)
        - Run self-attention over channels (features)
        - Run self-attention over time
        - Decode (upsample) into a (B, 1, H, W) grid
    """
    def __init__(self,
        in_channels, embed_dim, out_size, num_heads=4
    ):
        super().__init__()
        self.encoder = SpatialEncoder(in_channels, embed_dim)
        self.wins_attn = WindowedSpatialAttention(embed_dim, num_heads=ws_nheads, window_size=ws_window_size)
        self.chnl_attn = ChannelMixingAttention(num_heads=cm_nheads, num_channels=E, d_model=cm_d_model, mlp_ratio=cm_mlp_ratio, dropout=cm_dropout)
        self.temp_attn = TemporalMixingAttention(embed_dim, num_heads=tm_nheads, mlp_ratio=tm_mlp_ratio, dropout=tm_dropout)
        self.sp_decoder = SpatialTemporalDecoder(embed_dim, embed_dim=embed_dim, out_size=out_size)

    def forward(self, x: torch.Tensor):
        y = self.encoder(x)
        print(f"[WFM] Encoding complete...")

        y = self.wins_sattn(y)
        print(f"[WFM] Windowed Spatial Attention complete...")

        y = self.chnl_attn(y)
        print(f"[WFM] Channel Mixing Attention complete...")

        y = self.temp_attn(y)
        print(f"[WFM] Temporal Mixing Attention complete...")

        y = self.decoder(y)
        return y



