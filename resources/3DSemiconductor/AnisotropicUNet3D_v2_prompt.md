# AnisotropicUNet3D v2 — Architecture Generation Prompt

## Task
Implement `AnisotropicUNet3D` — a 3D semantic segmentation network for power module CT/SAM scans — in PyTorch. The model exploits the strict physical layer ordering along Z (base_plate → solder → copper → chip → bond_wire) while treating XY planes as independent intra-layer slices.

---

## Overall Architecture

Standard 4-level encoder–decoder U-Net topology:

```
Input [B,1,D,H,W]
  ↓ Encoder 0 (no pool): AnisotropicBlock(1→48)
  ↓ MaxPool3d(2)
  ↓ Encoder 1:           AnisotropicBlock(48→96)
  ↓ MaxPool3d(2)
  ↓ Encoder 2:           AnisotropicBlock(96→192)
  ↓ MaxPool3d(2)
  ↓ Encoder 3 (bottleneck): AnisotropicBlock(192→384)
  ↓ AttentionDecoder 0:  upsample(384→384) + AttentionGate(skip=192) + AnisotropicBlock(384→192)
  ↓ AttentionDecoder 1:  upsample(192→192) + AttentionGate(skip=96)  + AnisotropicBlock(192→96)
  ↓ AttentionDecoder 2:  upsample(96→96)   + AttentionGate(skip=48)  + AnisotropicBlock(96→48)
  ↓ Conv3d(48→num_classes, 1×1×1)
  ↓ Softmax(dim=1)
```

Upsampling uses TransposeConv3d (`stride=2`, `kernel=2`) followed by element-wise sum join (not concat).

---

## AnisotropicBlock (core building block)

Every encoder and decoder block is an `AnisotropicBlock`. It runs three parallel branches then fuses them.

### Parameters
| param | default | meaning |
|---|---|---|
| `in_channels` | — | input feature channels |
| `out_channels` | — | output feature channels |
| `d_state` | 32 | Mamba SSM state dimension |
| `d_conv` | 4 | Mamba inner conv width |
| `expand` | 4 | Mamba hidden expansion factor |
| `patch_size` P | 4 | spatial grid size per slice for branch B |
| `num_groups` | 8 | GroupNorm groups (auto-reduced if needed) |
| `dropout_prob` | 0.1 | applied after SE, before residual add |

### Branch A — XY per-slice 2D conv
Captures intra-layer spatial morphology independently per Z slice.

```
x: [B, C, D, H, W]
→ reshape to [B*D, C, H, W]
→ Conv2d(C, C', 3, padding=1) → GroupNorm → ELU
→ Conv2d(C', C', 3, padding=1) → GroupNorm → ELU
→ reshape back to [B, C', D, H, W]
```

### Branch B — Patch-token Bidirectional Mamba (Z ordering)
Captures inter-layer ordering with spatially-resolved tokens (NOT global average pool).

```
x: [B, C, D, H, W]
→ Conv3d(C, C', 1) if C≠C' else identity          # channel projection
→ AdaptiveAvgPool3d((D, P, P))                     # [B, C', D, P, P], P=4
→ permute+reshape → [B, D*P², C']                  # flatten to sequence

→ Mamba_fwd(d_model=C', d_state=32, expand=4)      # forward scan [B, D*P², C']
→ Mamba_bwd on flipped sequence then flip back      # backward scan [B, D*P², C']
→ average fwd and bwd                              # [B, D*P², C']

→ reshape → [B, C', D, P, P]
→ F.interpolate(size=(D,H,W), mode='trilinear')    # spatially-aware Z context [B, C', D, H, W]
```

Fallback when `mamba-ssm` not installed: replace both Mambas with `Conv1d(C', C', 3, padding=1, groups=C'//4)`, apply forward and backward independently on the flattened sequence.

### Branch C — 1×3×3 anisotropic local conv
Captures fine-grained local texture across adjacent slices without Z blurring.

```
x: [B, C, D, H, W]
→ Conv3d(C, C', kernel=(1,3,3), padding=(0,1,1)) → GroupNorm → ELU   # [B, C', D, H, W]
```

### Fusion + SE + Residual

```
cat([branch_A, branch_B, branch_C], dim=1)         # [B, 3*C', D, H, W]
→ Conv3d(3*C', C', 1) → GroupNorm → ELU            # [B, C', D, H, W]
→ ChannelSpatialSELayer3D(C', reduction=2)          # concurrent scSE recalibration
→ Dropout(p=0.1)
→ + residual_proj(x)                               # outer skip (Conv3d(C,C',1) if C≠C')
```

---

## AttentionGate3D (decoder skip gates)

Additive attention gate (Oktay et al. 2018) applied to encoder skip connections before residual addition:

```
g: gating signal (upsampled decoder)  [B, F_g, D, H, W]
x: encoder skip connection             [B, F_l, D, H, W]

W_g = Conv3d(F_g, F_int, 1) + GroupNorm
W_x = Conv3d(F_l, F_int, 1) + GroupNorm
psi = sigmoid(Conv3d(F_int, 1, 1))(ReLU(W_g(g) + W_x(x)))  # [B, 1, D, H, W]
output = x * psi
```

`F_int = F_l // 2`, `F_g = F_l = out_channels` at each decoder level.

---

## ChannelSpatialSELayer3D (scSE)

Concurrent squeeze-and-excitation:

- **Channel SE**: GlobalAvgPool3d → FC(C→C//r) → ReLU → FC(C//r→C) → Sigmoid → scale channels
- **Spatial SE**: Conv3d(C, 1, 1) → Sigmoid → scale spatial locations
- Output: `channel_se(x) + spatial_se(x)` (element-wise sum, not gate)

Reduction ratio `r=2` in all blocks.

---

## Loss Function: WCEDiceLoss

```
L = 0.5 × WCE + 0.5 × SoftDice
```

- **WCE**: CrossEntropyLoss with per-class frequency-inverse weights computed from training set label histograms (applied to target labels, not predictions).
- **SoftDice**: `1 - (2·Σ p·y + ε) / (Σ p² + Σ y² + ε)` averaged over foreground classes (skip background class 0).

---

## Training Config

| setting | value |
|---|---|
| optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| grad clipping | max_norm=1.0 |
| lr scheduler | ReduceLROnPlateau(mode=max, factor=0.5, patience=15) |
| early stopping | patience=30 on val MeanIoU |
| batch size | 1 |
| patch shape | [80, 170, 170] (train), [80, 170, 170] (val, stride=full) |
| augmentation | RandomFlip, RandomRotate90, RandomRotate(±30°, XY only), ElasticDeformation, AdditiveGaussianNoise |
| normalization | per-volume Standardize (zero-mean unit-variance) |
| eval metric | MeanIoU over 5 foreground classes (skip background) |

---

## Parameter Count

With `f_maps=[48, 96, 192, 384]`, `d_state=32`, `expand=4`, `patch_size=4`:
- **Total: ~9.7M parameters**

---

## Key Design Decisions

1. **Patch tokens vs GAP**: v1 used `mean(dim=[H,W])` → 1 vector per slice, discarding all spatial info before Mamba. v2 uses `AdaptiveAvgPool3d((D,P,P))` → 16 tokens per slice → Mamba retains spatial structure within each layer.

2. **Bidirectional scan**: single-direction Mamba only captures causal Z dependencies (e.g., "what came before chip"). Bidirectional scan (fwd + bwd averaged) captures both causal and anti-causal context — critical for layers that are defined by their position relative to layers both above AND below.

3. **Trilinear upsample vs broadcast**: v1 broadcast the per-slice Z context uniformly to all (H,W) positions. v2 upsamples the P×P context map back to (H,W) via trilinear interpolation — different positions in the same Z-slice get different Z-context, letting Mamba distinguish e.g. a chip corner from chip center.

4. **1×3×3 local branch**: 3D conv with kernel (1,3,3) captures local cross-slice texture (the precise boundary where one material ends and another begins) without blurring along Z. This is complementary to both the 2D per-slice branch A and the long-range Z branch B.

5. **scSE after fusion**: recalibrates both channel importance (which features matter) and spatial saliency (which voxels to attend to) after the three branches are fused, before the outer residual add.
