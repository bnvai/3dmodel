"""
UNETR++ standalone implementation for pytorch-3dunet.
No monai dependency — pure PyTorch only.

Based on: Shaker et al., "UNETR++: Delving into Efficient and Accurate
3D Medical Image Segmentation", IEEE TMI 2024.

Key change vs original: do_ds=False (single output, compatible with pytorch-3dunet trainer).
Default img_size=(64, 128, 128) matches the paper's synapse configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Union

try:
    from torch.nn.init import trunc_normal_
except ImportError:
    def trunc_normal_(tensor, std=.02, **kwargs):
        nn.init.trunc_normal_(tensor, std=std)


# ─── Low-level blocks ────────────────────────────────────────────────────────

def _pair_int(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)


class UnetResBlock(nn.Module):
    """Residual conv block (InstanceNorm + LeakyReLU)."""
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, stride,
                 norm_name="instance", dropout=None):
        super().__init__()
        ks = _pair_int(kernel_size)
        pad = tuple(k // 2 for k in ks)
        st = _pair_int(stride)

        self.conv1 = nn.Conv3d(in_channels, out_channels, ks, stride=st, padding=pad, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, ks, stride=1, padding=pad, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

        need_ds = (in_channels != out_channels) or any(s != 1 for s in st)
        if need_ds:
            self.conv3 = nn.Conv3d(in_channels, out_channels, 1, stride=st, bias=False)
            self.norm3 = nn.InstanceNorm3d(out_channels, affine=True)
        else:
            self.conv3 = None

    def forward(self, x):
        residual = self.norm3(self.conv3(x)) if self.conv3 is not None else x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class UnetOutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


# ─── EPA (Efficient Paired Attention) ────────────────────────────────────────

class EPA(nn.Module):
    """
    Efficient Paired Attention block.
    Spatial attention: linear complexity O(N·P).
    Channel attention: O(C²) per head.
    Shared query/key across both branches.
    """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature  = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        # query_shared, key_shared, value_CA, value_SA
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4)
        # E and F share weights — project spatial dimension N→P
        self.E = self.F = nn.Linear(input_size, proj_size)
        self.attn_drop   = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        self.out_proj  = nn.Linear(hidden_size, hidden_size // 2)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        head_dim = C // H

        qkvv = self.qkvv(x).reshape(B, N, 4, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        # Transpose to (B, H, head_dim, N)
        q  = q.transpose(-2, -1)
        k  = k.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_proj  = self.E(k)    # (B, H, head_dim, P)
        v_SA_p  = self.F(v_SA) # (B, H, head_dim, P)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Channel attention: (B, H, head_dim, head_dim)
        attn_CA = self.attn_drop((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        # Spatial attention: (B, H, N, P)
        attn_SA = self.attn_drop_2((q.permute(0,1,3,2) @ k_proj) * self.temperature2).softmax(dim=-1)
        x_SA = (attn_SA @ v_SA_p.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        return torch.cat([self.out_proj(x_SA), self.out_proj2(x_CA)], dim=-1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class TransformerBlock(nn.Module):
    """EPA block + conv branch with residual."""
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4,
                 dropout_rate=0.0, pos_embed=False):
        super().__init__()
        self.norm     = nn.LayerNorm(hidden_size)
        self.gamma    = nn.Parameter(1e-6 * torch.ones(hidden_size))
        self.epa      = EPA(input_size, hidden_size, proj_size, num_heads,
                            channel_attn_drop=dropout_rate, spatial_attn_drop=dropout_rate)
        self.conv51   = UnetResBlock(3, hidden_size, hidden_size, 3, 1)
        self.conv8    = nn.Sequential(nn.Dropout3d(0.1, False),
                                      nn.Conv3d(hidden_size, hidden_size, 1))
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size)) if pos_embed else None

    def forward(self, x):
        B, C, H, W, D = x.shape
        x_seq = x.reshape(B, C, H * W * D).permute(0, 2, 1)   # (B, N, C)
        if self.pos_embed is not None:
            x_seq = x_seq + self.pos_embed
        attn = x_seq + self.gamma * self.epa(self.norm(x_seq))
        attn_3d = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        return attn_3d + self.conv8(self.conv51(attn_3d))


# ─── Encoder / Decoder blocks ─────────────────────────────────────────────────

class UnetrPPEncoder(nn.Module):
    """
    Hierarchical encoder: 4 stages.
    Stage 0 stem: stride (2, 4, 4)
    Stages 1-3: stride (2, 2, 2)
    """
    def __init__(self, input_size, dims, proj_size, depths, num_heads=4,
                 in_channels=1, transformer_dropout_rate=0.15):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem: (2, 4, 4) stride; GroupNorm with groups=in_channels
        stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4), bias=False),
            nn.GroupNorm(in_channels, dims[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(dims[i], dims[i + 1]),
            ))

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                TransformerBlock(
                    input_size=input_size[i],
                    hidden_size=dims[i],
                    proj_size=proj_size[i],
                    num_heads=num_heads,
                    dropout_rate=transformer_dropout_rate,
                    pos_embed=True,
                )
                for _ in range(depths[i])
            ]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        hidden_states = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:
                B, C, H, W, D = x.shape
                x = x.reshape(B, C, H * W * D).permute(0, 2, 1)  # (B, N, C)
            hidden_states.append(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    """Decoder block: transposed conv upsample + EPA (or conv) + skip connection."""
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size,
                 upsample_kernel_size, norm_name, proj_size=64, num_heads=4,
                 out_size=0, depth=3, conv_decoder=False):
        super().__init__()
        self.transp_conv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            bias=False,
        )
        if conv_decoder:
            self.decoder_block = nn.ModuleList([
                UnetResBlock(spatial_dims, out_channels, out_channels,
                             kernel_size=kernel_size, stride=1)
            ])
        else:
            self.decoder_block = nn.ModuleList([nn.Sequential(*[
                TransformerBlock(
                    input_size=out_size,
                    hidden_size=out_channels,
                    proj_size=proj_size,
                    num_heads=num_heads,
                    dropout_rate=0.15,
                    pos_embed=True,
                )
                for _ in range(depth)
            ])])

    def forward(self, inp, skip):
        out = self.transp_conv(inp) + skip
        return self.decoder_block[0](out)


# ─── Top-level model ──────────────────────────────────────────────────────────

class UNETR_PP_SAM(nn.Module):
    """
    UNETR++ adapted for SAM chip segmentation (pytorch-3dunet compatible).

    Default img_size=(64, 128, 128) matches the paper's synapse encoder design.
    do_ds=False: single output tensor, no deep supervision.

    Patch constraint: img_size must satisfy
        D % 16 == 0  and  H % 32 == 0  and  W % 32 == 0
    because stem uses stride (2,4,4) then three (2,2,2) downsamplings.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        img_size: Tuple[int, int, int] = (64, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        depths=None,
        dims=None,
        norm_name: str = "instance",
        dropout_rate: float = 0.0,
        is_segmentation: bool = True,
        final_sigmoid: bool = False,
        **kwargs,
    ):
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        if dims is None:
            dims = [32, 64, 128, 256]

        D, H, W = img_size
        assert D % 16 == 0 and H % 32 == 0 and W % 32 == 0, \
            f"img_size {img_size} must be divisible by (16, 32, 32)"

        # Sequence lengths after each encoder stage
        # stem (2,4,4): (D/2, H/4, W/4)
        d0, h0, w0 = D // 2, H // 4, W // 4
        input_sizes = [
            d0 * h0 * w0,
            (d0 // 2) * (h0 // 2) * (w0 // 2),
            (d0 // 4) * (h0 // 4) * (w0 // 4),
            (d0 // 8) * (h0 // 8) * (w0 // 8),
        ]
        proj_sizes = [64, 64, 64, 32]

        self.feat_size = (d0 // 8, h0 // 8, w0 // 8)   # bottleneck spatial
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=input_sizes,
            dims=dims,
            proj_size=proj_sizes,
            depths=depths,
            num_heads=num_heads,
            in_channels=in_channels,
            transformer_dropout_rate=0.15,
        )

        self.encoder1 = UnetResBlock(3, in_channels, feature_size, 3, 1)

        self.decoder5 = UnetrUpBlock(3, dims[3], feature_size * 8,  3, 2, norm_name,
                                     proj_size=proj_sizes[2], num_heads=num_heads,
                                     out_size=input_sizes[2], depth=depths[2])
        self.decoder4 = UnetrUpBlock(3, feature_size * 8, feature_size * 4, 3, 2, norm_name,
                                     proj_size=proj_sizes[1], num_heads=num_heads,
                                     out_size=input_sizes[1], depth=depths[1])
        self.decoder3 = UnetrUpBlock(3, feature_size * 4, feature_size * 2, 3, 2, norm_name,
                                     proj_size=proj_sizes[0], num_heads=num_heads,
                                     out_size=input_sizes[0], depth=depths[0])
        self.decoder2 = UnetrUpBlock(3, feature_size * 2, feature_size, 3, (2, 4, 4), norm_name,
                                     out_size=D * H * W, conv_decoder=True)

        self.out1 = UnetOutBlock(feature_size, out_channels)
        if is_segmentation:
            self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def proj_feat(self, x):
        B = x.shape[0]
        fd, fh, fw = self.feat_size
        return x.view(B, fd, fh, fw, self.hidden_size).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x, return_logits=False):
        x_output, hidden_states = self.unetr_pp_encoder(x)

        enc0 = hidden_states[0]   # (B, dims[0], d0,   h0,   w0)
        enc1 = hidden_states[1]   # (B, dims[1], d0/2, h0/2, w0/2)
        enc2 = hidden_states[2]   # (B, dims[2], d0/4, h0/4, w0/4)
        enc3 = hidden_states[3]   # (B, N, hidden_size) — bottleneck

        convBlock = self.encoder1(x)

        dec4 = self.proj_feat(enc3)      # reshape bottleneck → 3-D
        dec3 = self.decoder5(dec4, enc2)
        dec2 = self.decoder4(dec3, enc1)
        dec1 = self.decoder3(dec2, enc0)
        logits = self.out1(self.decoder2(dec1, convBlock))

        if return_logits:
            out = self.final_activation(logits) if self.final_activation is not None else logits
            return out, logits

        if self.final_activation is not None:
            return self.final_activation(logits)
        return logits
