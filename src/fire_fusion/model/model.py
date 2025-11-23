import torch
import torch.nn as nn
from .modules import SpatialEncoder, WindowedSpatialAttention, ChannelMixingAttention, TemporalMixingAttention, BiHeadDecoder




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
        in_channels, embed_dim, out_size,
        ws_nheads: int, ws_win_size: int,                   ws_dropout: int,
        cm_nheads: int, cm_d_model: int, cm_mlp_ratio: int, cm_dropout: int,
        tm_nheads: int,                  tm_mlp_ratio: int, tm_dropout: int
    ):
        super().__init__()
        self.encoder = SpatialEncoder(in_channels, embed_dim)
        self.ws_attn = WindowedSpatialAttention(embed_dim, num_heads=ws_nheads, window_size=ws_win_size, dropout=ws_dropout)
        self.cm_attn = ChannelMixingAttention(num_heads=cm_nheads, num_channels=E, d_model=cm_d_model, mlp_ratio=cm_mlp_ratio, dropout=cm_dropout)
        self.tm_attn = TemporalMixingAttention(embed_dim, num_heads=tm_nheads, mlp_ratio=tm_mlp_ratio, dropout=tm_dropout)
        self.decoder = BiHeadDecoder(embed_dim, out_size, n_cause_classes=3)

    def forward(self, x: torch.Tensor):
        y = self.encoder(x)
        print(f"[WFM] Encoding complete...")

        y = self.ws_attn(y)
        print(f"[WFM] Windowed Spatial Attention complete...")

        y = self.cm_attn(y)
        print(f"[WFM] Channel Mixing Attention complete...")

        y = self.tm_attn(y)
        print(f"[WFM] Temporal Mixing Attention complete...")

        # Only decode the prediction from the last day
        y = y[:, -1]

        outputs = self.decoder(y)
        return outputs


