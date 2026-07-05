from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

from pytorch3dunet.unet3d.se import ChannelSELayer3D, ChannelSpatialSELayer3D, SpatialSELayer3D


def create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    order: str,
    num_groups: int,
    padding: int | tuple[int],
    dropout_prob: float,
    is3d: bool,
) -> list[tuple[str, nn.Module]]:
    """
    Create a list of modules for a given level of UNet network. It consists of a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (str): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        dropout_prob (float): dropout probability
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list modules, where each module is a tuple (name, module)
    """
    assert "c" in order, "Conv layer MUST be present"
    assert order[0] not in "rle", "Non-linearity cannot be the first operation in the layer"

    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ("g" in order or "b" in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

            modules.append(("conv", conv))
        elif char == "g":
            is_before_conv = i < order.index("c")
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, (
                f"Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}"
            )
            modules.append(("groupnorm", nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == "b":
            is_before_conv = i < order.index("c")
            if is3d:
                bn = nn.BatchNorm3d
            else:
                bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(("batchnorm", bn(in_channels)))
            else:
                modules.append(("batchnorm", bn(out_channels)))
        elif char == "d":
            modules.append(("dropout", nn.Dropout(p=dropout_prob)))
        elif char == "D":
            modules.append(("dropout2d", nn.Dropout2d(p=dropout_prob)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd', 'D']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        dropout_prob (float): dropout probability, default 0.1
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()

        for name, module in create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding, dropout_prob, is3d
        ):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers. We use 2x (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        dropout_prob (float or tuple): dropout probability for each convolution, default 0.1
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        encoder,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        upscale=2,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # check if dropout_prob is a tuple and if so
        # split it for different dropout probabilities for each convolution.
        if isinstance(dropout_prob, list) or isinstance(dropout_prob, tuple):
            dropout_prob1 = dropout_prob[0]
            dropout_prob2 = dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob

        # conv1
        self.add_module(
            "SingleConv1",
            SingleConv(
                conv1_in_channels,
                conv1_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                dropout_prob=dropout_prob1,
                is3d=is3d,
            ),
        )
        # conv2
        self.add_module(
            "SingleConv2",
            SingleConv(
                conv2_in_channels,
                conv2_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                dropout_prob=dropout_prob2,
                is3d=is3d,
            ),
        )


class ResNetBlock(nn.Module):
    """Residual block that can be used instead of standard DoubleConv in the Encoder module.

    Motivated by: https://arxiv.org/pdf/1706.00120.pdf
    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel. Default: 3.
        order: Determines the order of layers. Default: 'cge'.
        num_groups: Number of groups for the GroupNorm. Default: 8.
        is3d: If True use Conv3d, otherwise use Conv2d. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order="cge", num_groups=8, is3d=True, **kwargs):
        super().__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = SingleConv(
            out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups, is3d=is3d
        )
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in "rel":
            n_order = n_order.replace(c, "")
        self.conv3 = SingleConv(
            out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups, is3d=is3d
        )

        # create non-linearity separately
        if "l" in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif "e" in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class ResNetBlockSE(ResNetBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, order="cge", num_groups=8, se_module="scse", **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups, **kwargs
        )
        assert se_module in ["scse", "cse", "sse"]
        if se_module == "scse":
            self.se_module = ChannelSpatialSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == "cse":
            self.se_module = ChannelSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == "sse":
            self.se_module = SpatialSELayer3D(num_channels=out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se_module(out)
        return out


class HybridMambaBottleneck(nn.Module):
    """Hybrid bottleneck that combines ResNetBlockSE (local) with a Mamba SSM (global).

    Two branches run in parallel on the same input:
      - Branch A: ResNetBlockSE — local 3-D convolutional feature recalibration via scSE.
      - Branch B: Mamba SSM — linear-time selective state-space model operating on
        the spatial tokens (B, D*H*W, C) to capture long-range volumetric dependencies
        that convolutions miss at the bottleneck resolution.

    The branch outputs are channel-concatenated and fused through a 1×1×1 Conv3d +
    GroupNorm + ELU. An outer skip adds the projected input:
        output = ELU(GN(Conv1x1(cat(A, B)))) + proj(x)

    Graceful fallback: if ``mamba-ssm`` is not installed Branch B degrades to
    ``nn.Identity`` (passes the projected input unchanged) and a one-time warning is
    emitted.  The block still runs correctly with only Branch A active.

    Compatible with ``torch.compile()`` — no in-place ops on tensors that flow through
    the Mamba kernel.

    Reference:
        Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        arXiv:2312.00752, 2023.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        order: Layer-order string forwarded to ResNetBlockSE (e.g. ``'cge'``).
        num_groups: GroupNorm groups for both ResNetBlockSE and the fusion layer.
        d_state: SSM state dimension. Default 16.
        d_conv: Mamba internal local-conv width. Default 4.
        expand: Inner-dimension multiplier for Mamba. Default 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: str = "cge",
        num_groups: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Branch A — local residual + scSE recalibration
        self.branch_a = ResNetBlockSE(
            in_channels, out_channels,
            order=order, num_groups=num_groups, **kwargs,
        )

        # Branch B — global Mamba SSM
        # Channel projection so Mamba always sees out_channels features
        self.input_proj = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        try:
            from mamba_ssm import Mamba  # noqa: PLC0415
            self.mamba = Mamba(
                d_model=out_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._mamba_available = True
        except ImportError:
            import warnings
            warnings.warn(
                "mamba-ssm not installed; HybridMambaBottleneck falls back "
                "to ResNetBlockSE-only mode.",
                stacklevel=2,
            )
            self.mamba = nn.Identity()
            self._mamba_available = False

        # Fusion: cat(A, B) [2C] → C via 1×1×1 Conv + GroupNorm + ELU
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ELU(inplace=False),   # inplace=False for torch.compile safety
        )

        # Outer skip — project input to out_channels when dimensions differ
        self.residual_proj = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch A: local SE features  [B, C, D, H, W]
        a_out = self.branch_a(x)

        # Branch B: global Mamba features
        x_proj = self.input_proj(x)                          # [B, C, D, H, W]
        B, C, D, H, W = x_proj.shape
        tokens = x_proj.flatten(2).permute(0, 2, 1)         # [B, D*H*W, C]
        if self._mamba_available:
            tokens = self.mamba(tokens)                      # [B, D*H*W, C]
        b_out = tokens.permute(0, 2, 1).reshape(B, C, D, H, W)  # [B, C, D, H, W]

        # Fuse and add outer residual
        fused = self.fusion(torch.cat([a_out, b_out], dim=1))   # [B, C, D, H, W]
        return fused + self.residual_proj(x)


class AnisotropicBlock(nn.Module):
    """Anisotropic processing block v2 for SAM power module 3D volumes.

    Three complementary branches:
      - Branch A (XY): 2D convolutions applied slice-by-slice along Z.
        Captures intra-layer shape features (each horizontal layer independently).
      - Branch B (Z): Patch-token bidirectional Mamba along Z → trilinear upsample.
        Replaces the v1 Global Average Pool with P×P spatial patch tokens per slice,
        giving Mamba spatially-resolved cross-slice context instead of a single
        collapsed vector. Bidirectional scan (fwd + bwd averaged) captures both
        causal and anti-causal Z dependencies. Output is upsampled back to [D,H,W]
        so Z-context is position-aware, not a uniform slice broadcast.
      - Branch C (Local): 1×3×3 3D conv (anisotropic local).
        Captures fine-grained cross-slice local texture with no Z blurring.

    Fusion: cat(A, B, C) [3C] → 1×1×1 Conv → GroupNorm → ELU → scSE → residual.

    Fallback: if mamba-ssm is not installed, Branch B uses a 1D grouped conv along Z
    (applied independently per spatial position via batch reshape).

    Interface-compatible with ResNetBlockSE for use in create_encoders/create_decoders.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: bool = True,
        kernel_size: int = 3,
        order: str = "cge",
        num_groups: int = 8,
        padding: int = 1,
        upscale: int = 2,
        dropout_prob: float = 0.1,
        is3d: bool = True,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 4,
        patch_size: int = 4,   # P: each slice → P×P spatial tokens for Mamba
        **kwargs,
    ):
        super().__init__()

        # num_groups must divide out_channels
        ng = num_groups
        while out_channels % ng != 0 and ng > 1:
            ng -= 1

        self.patch_size = patch_size

        # Branch A: 2D conv per Z-slice (captures intra-layer features)
        self.xy_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ELU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ELU(inplace=False),
        )

        # Branch B: patch-token bidirectional Mamba for Z-ordering context
        self.z_proj = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        try:
            from mamba_ssm import Mamba  # noqa: PLC0415
            # Sequence length: D * P * P; token dim: out_channels
            self.z_mamba_fwd = Mamba(d_model=out_channels, d_state=d_state, d_conv=d_conv, expand=expand)
            self.z_mamba_bwd = Mamba(d_model=out_channels, d_state=d_state, d_conv=d_conv, expand=expand)
            self._mamba_available = True
        except ImportError:
            import warnings
            warnings.warn(
                "mamba-ssm not installed; AnisotropicBlock Z-branch uses 1D conv fallback.",
                stacklevel=2,
            )
            groups = max(1, out_channels // 4)
            while out_channels % groups != 0:
                groups -= 1
            self.z_mamba_fwd = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)
            self.z_mamba_bwd = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)
            self._mamba_available = False

        # Branch C: 1×3×3 anisotropic local conv (cross-slice local texture)
        self.local_branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ELU(inplace=False),
        )

        # Fusion: cat(branch_A, branch_B, branch_C) [3C] → [C]
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ELU(inplace=False),
        )

        # Concurrent scSE recalibration
        self.se = ChannelSpatialSELayer3D(out_channels, reduction_ratio=2)

        # Outer residual projection
        self.residual_proj = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        P = self.patch_size

        # ── Branch A: XY plane (per Z-slice 2D conv) ──────────────────────────
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)    # [B*D, C, H, W]
        xy_out = self.xy_branch(x_2d)                                # [B*D, C', H, W]
        C_out = xy_out.shape[1]
        xy_out = xy_out.reshape(B, D, C_out, H, W).permute(0, 2, 1, 3, 4)  # [B, C', D, H, W]

        # ── Branch B: patch-token bidirectional Mamba ─────────────────────────
        x_proj = self.z_proj(x)                                      # [B, C', D, H, W]
        # Downsample to P×P spatial tokens per slice
        x_pooled = F.adaptive_avg_pool3d(x_proj, (D, P, P))         # [B, C', D, P, P]
        # Flatten to sequence: [B, D*P², C']
        z_seq = x_pooled.permute(0, 2, 3, 4, 1).reshape(B, D * P * P, C_out)

        if self._mamba_available:
            z_fwd = self.z_mamba_fwd(z_seq)                         # [B, D*P², C']
            z_bwd = self.z_mamba_bwd(z_seq.flip(1)).flip(1)         # bidirectional
            z_ctx = (z_fwd + z_bwd) * 0.5                           # [B, D*P², C']
        else:
            # Fallback: 1D grouped conv per token dimension
            z_t = z_seq.permute(0, 2, 1)                            # [B, C', D*P²]
            z_fwd = self.z_mamba_fwd(z_t)
            z_bwd = self.z_mamba_bwd(z_t.flip(2)).flip(2)
            z_ctx = ((z_fwd + z_bwd) * 0.5).permute(0, 2, 1)       # [B, D*P², C']

        # Reshape back to spatial volume at patch resolution
        z_ctx_vol = z_ctx.reshape(B, D, P, P, C_out).permute(0, 4, 1, 2, 3)  # [B, C', D, P, P]
        # Upsample to full resolution — spatially-aware Z context
        z_out = F.interpolate(z_ctx_vol, size=(D, H, W), mode="trilinear", align_corners=False)

        # ── Branch C: 1×3×3 local anisotropic conv ────────────────────────────
        local_out = self.local_branch(x)                             # [B, C', D, H, W]

        # ── Fusion + SE + residual ─────────────────────────────────────────────
        fused = self.fusion(torch.cat([xy_out, z_out, local_out], dim=1))  # [B, C', D, H, W]
        fused = self.se(fused)
        fused = self.dropout(fused)
        return fused + self.residual_proj(x)


class AttentionGate3D(nn.Module):
    """Additive attention gate for 3D U-Net skip connections.

    Projects the gating signal and the skip connection into a shared intermediate
    space, adds them, applies ReLU + a 1×1×1 sigmoid to produce a soft attention
    map, and returns the recalibrated skip connection.

    Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the
    Pancreas," MIDL 2018, arXiv:1804.03999.

    Args:
        F_g: Channels in the gating signal (upsampled decoder output).
        F_l: Channels in the skip connection (encoder output).
        F_int: Intermediate projection channels (typically F_l // 2).
        num_groups: Groups for GroupNorm in projection layers.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int, num_groups: int = 8):
        super().__init__()
        # Find the largest power-of-2 divisor of F_int up to num_groups
        n_g = num_groups
        while F_int % n_g != 0 and n_g > 1:
            n_g //= 2

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=n_g, num_channels=F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=n_g, num_channels=F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply attention gate.

        Args:
            g: Gating signal (upsampled decoder output), shape (N, F_g, D, H, W).
            x: Skip connection from encoder, shape (N, F_l, D, H, W).
        Returns:
            Attention-recalibrated skip connection, shape (N, F_l, D, H, W).
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    from the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        dropout_prob (float or tuple): dropout probability, default 0.1
        is3d (bool): use 3d or 2d convolutions/pooling operation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=2,
        pool_type="max",
        basic_module=DoubleConv,
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        upscale=2,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()
        assert pool_type in ["max", "avg"]
        if apply_pooling:
            if pool_type == "max":
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            upscale=upscale,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (int or tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (str): algorithm used for upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default 0.1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        scale_factor=2,
        basic_module=DoubleConv,
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()

        # perform concat joining per default
        concat = True

        # don't adapt channels after join operation
        adapt_channels = False

        if upsample is not None and upsample != "none":
            if upsample == "default":
                if basic_module == DoubleConv:
                    upsample = "nearest"  # use nearest neighbor interpolation for upsampling
                    concat = True  # use concat joining
                    adapt_channels = False  # don't adapt channels
                elif basic_module in (ResNetBlock, ResNetBlockSE, AnisotropicBlock):
                    upsample = "deconv"  # use deconvolution upsampling
                    concat = False  # use summation joining
                    adapt_channels = True  # adapt channels after joining

            # perform deconvolution upsampling if mode is deconv
            if upsample == "deconv":
                self.upsampling = TransposeConvUpsampling(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    scale_factor=scale_factor,
                    is3d=is3d,
                )
            else:
                self.upsampling = InterpolateUpsampling(mode=upsample)
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        # perform joining operation
        self.joining = partial(self._joining, concat=concat)

        # adapt the number of in_channels for the ResNetBlock
        if adapt_channels is True:
            in_channels = out_channels

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class AttentionDecoder(Decoder):
    """Decoder with an attention gate applied to the encoder skip connection.

    Extends Decoder by inserting an AttentionGate3D between the upsampling step
    and the skip-connection join. The upsampled decoder output serves as the
    gating signal; the encoder skip is recalibrated before being added.

    This implements the architecture described in:
        Oktay et al., "Attention U-Net," MIDL 2018, arXiv:1804.03999.

    Args:
        Same as Decoder. The attention gate uses F_int = out_channels // 2.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        scale_factor=2,
        basic_module=DoubleConv,
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            conv_kernel_size=conv_kernel_size,
            scale_factor=scale_factor,
            basic_module=basic_module,
            conv_layer_order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )
        # After TransposeConvUpsampling, both gating signal and skip have out_channels channels
        F_int = max(1, out_channels // 2)
        self.attention_gate = AttentionGate3D(
            F_g=out_channels,
            F_l=out_channels,
            F_int=F_int,
            num_groups=num_groups,
        )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        # gate the skip connection using the upsampled decoder as the gating signal
        encoder_features = self.attention_gate(g=x, x=encoder_features)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x


def create_encoders(
    in_channels,
    f_maps,
    basic_module,
    conv_kernel_size,
    conv_padding,
    conv_upscale,
    dropout_prob,
    layer_order,
    num_groups,
    pool_kernel_size,
    is3d,
):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            # apply conv_coord only in the first encoder if any
            encoder = Encoder(
                in_channels,
                out_feature_num,
                apply_pooling=False,  # skip pooling in the firs encoder
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                upscale=conv_upscale,
                dropout_prob=dropout_prob,
                is3d=is3d,
            )
        else:
            encoder = Encoder(
                f_maps[i - 1],
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                pool_kernel_size=pool_kernel_size,
                padding=conv_padding,
                upscale=conv_upscale,
                dropout_prob=dropout_prob,
                is3d=is3d,
            )

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(
    f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample, dropout_prob, is3d
):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv and upsample != "deconv":
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = Decoder(
            in_feature_num,
            out_feature_num,
            basic_module=basic_module,
            conv_layer_order=layer_order,
            conv_kernel_size=conv_kernel_size,
            num_groups=num_groups,
            padding=conv_padding,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )
        decoders.append(decoder)
    return nn.ModuleList(decoders)


def create_attention_decoders(
    f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample, dropout_prob, is3d
):
    """Create a decoder path where each level has an AttentionGate3D on its skip connection.

    Intended for use with ResNetBlock / ResNetBlockSE (element-wise summation join,
    transposed convolution upsampling). Each AttentionDecoder gates the encoder skip
    with the upsampled decoder output before the residual addition.
    """
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        # For ResNetBlock/SE with deconv: in_channels = reversed_f_maps[i]
        in_feature_num = reversed_f_maps[i]
        out_feature_num = reversed_f_maps[i + 1]
        decoder = AttentionDecoder(
            in_feature_num,
            out_feature_num,
            basic_module=basic_module,
            conv_layer_order=layer_order,
            conv_kernel_size=conv_kernel_size,
            num_groups=num_groups,
            padding=conv_padding,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """Abstract class for upsampling.

    A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.

    Args:
        upsample: Upsampling function to be used.
    """

    def __init__(self, upsample):
        super().__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Non-learnable upsampling backed by interpolation.

    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode="nearest"):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Learned upsampling backed by transposed convolution layer followed by interpolation to the correct size if necessary.

    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
        is3d (bool): if True use ConvTranspose3d, otherwise use ConvTranspose2d
    """

    class Upsample(nn.Module):
        """Workaround for ValueError in transposed convolution.

        Performs transposed conv followed by interpolation to the correct size if necessary.
        This addresses the 'ValueError: requested an output size...' in the `_output_padding` method.

        Args:
            conv_transposed: Transposed convolution layer.
            is3d: If True use 3D operations, otherwise use 2D.
        """

        def __init__(self, conv_transposed, is3d):
            super().__init__()
            self.conv_transposed = conv_transposed
            self.is3d = is3d

        def forward(self, x, size):
            x = self.conv_transposed(x)
            return F.interpolate(x, size=size)

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, is3d=True):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        if is3d is True:
            conv_transposed = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1, bias=False
            )
        else:
            conv_transposed = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1, bias=False
            )
        upsample = self.Upsample(conv_transposed, is3d)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    """No upsampling, return the input as is."""

    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x
