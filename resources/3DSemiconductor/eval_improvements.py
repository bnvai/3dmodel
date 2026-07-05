"""
eval_improvements.py — So sanh 3 cau hinh inference tren best checkpoint v3.
Run:
  cd D:\Working\10. AI-Workspace\model\pytorch-3dunet
  python -u resources/3DSemiconductor/eval_improvements.py
"""
import sys, time, warnings, io
sys.path.insert(0, ".")
# Force UTF-8 safe stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")

import numpy as np
import torch
import h5py
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = r"resources/3DSemiconductor/checkpoints_proposed_v3_runpod/best_checkpoint.pytorch"
VAL_DIR     = r"resources/dataset/val"
PATCH       = (80, 170, 170)
NUM_CLASSES = 6
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# None = run all 3 val volumes (recommended on RunPod); "Sample5" = quick test
TEST_VOLUME = None

CLASS_NAMES = ["background", "base_cu", "circuit", "base_chip", "chip", "solder"]

# ── load model ────────────────────────────────────────────────────────────────
def load_model():
    from pytorch3dunet.unet3d.model import ResidualAttentionUNet3D
    model = ResidualAttentionUNet3D(
        in_channels=1, out_channels=6,
        f_maps=[64, 128, 256, 512],
        layer_order="cge", num_groups=8,
        final_sigmoid=False, is_segmentation=True,
        dropout_prob=0.2, bottleneck_type="mamba",
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [warn] Missing keys (mamba not loaded): {len(missing)} tensors")
    model.eval().to(DEVICE)
    step = ckpt.get("num_iterations", "?")
    score = ckpt.get("best_eval_score", "?")
    print(f"  Checkpoint step={step}  best_val_iou={score:.4f}")
    return model

# ── sliding window inference ──────────────────────────────────────────────────
def sliding_window_predict(model, volume, patch, stride, use_tta=False):
    """
    volume: np.float32 [D, H, W]  (already standardised)
    Returns: argmax label map [D, H, W] uint8
    """
    D, H, W = volume.shape
    pd, ph, pw = patch
    sd, sh, sw = stride

    # accumulate softmax probabilities (float32) and hit counts
    probs = np.zeros((NUM_CLASSES, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    # generate patch start indices with padding to cover full volume
    def starts(size, p, s):
        idx = list(range(0, max(size - p, 0) + 1, s))
        if not idx or idx[-1] + p < size:
            idx.append(max(0, size - p))
        return sorted(set(idx))

    zz = starts(D, pd, sd)
    yy = starts(H, ph, sh)
    xx = starts(W, pw, sw)

    def predict_patch(patch_np):
        """patch_np: [1, 1, pd, ph, pw] float32 tensor"""
        t = torch.from_numpy(patch_np).to(DEVICE)
        with torch.no_grad():
            # In eval mode with final_sigmoid=False, model returns softmax probs directly
            p = model(t).cpu().numpy()[0]  # [C, pd, ph, pw]
        return p

    def predict_with_tta(patch_np):
        """Average softmax over 8 flip augmentations."""
        base = predict_patch(patch_np)
        total = base.copy()
        # 7 flips: d, h, w, dh, dw, hw, dhw
        flip_axes = [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
        for axes in flip_axes:
            flipped = np.flip(patch_np, axis=axes).copy()
            pred = predict_patch(flipped)
            # un-flip the prediction along the same axes (spatial axes 1,2,3 in pred)
            pred_axes = tuple(a - 1 for a in axes)  # patch axes 2,3,4 → pred axes 1,2,3
            total += np.flip(pred, axis=pred_axes).copy()
        return total / 8.0

    pred_fn = predict_with_tta if use_tta else predict_patch

    for z in zz:
        for y in yy:
            for x in xx:
                ez = min(z + pd, D); ey = min(y + ph, H); ex = min(x + pw, W)
                # actual patch (may be smaller at volume boundary)
                patch_vol = volume[z:ez, y:ey, x:ex]
                # pad to full patch size if needed
                pz, py_, px_ = patch_vol.shape
                if (pz, py_, px_) != (pd, ph, pw):
                    pad = np.zeros((pd, ph, pw), dtype=np.float32)
                    pad[:pz, :py_, :px_] = patch_vol
                    patch_vol = pad

                patch_t = patch_vol[np.newaxis, np.newaxis]  # [1,1,pd,ph,pw]
                p = pred_fn(patch_t)                          # [C, pd, ph, pw]

                # accumulate only the valid region
                probs[:, z:ez, y:ey, x:ex] += p[:, :pz, :py_, :px_]
                count[z:ez, y:ey, x:ex]    += 1.0

    probs /= np.maximum(count[np.newaxis], 1e-7)
    return np.argmax(probs, axis=0).astype(np.uint8)

# ── IoU computation ───────────────────────────────────────────────────────────
def mean_iou(pred, gt, num_classes=NUM_CLASSES, skip_bg=True):
    """Returns per-class IoU dict and mean over foreground classes."""
    ious = {}
    start = 1 if skip_bg else 0
    for c in range(num_classes):
        p = pred == c; g = gt == c
        tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
        if not g.any():
            ious[c] = float("nan")
        else:
            ious[c] = tp / (tp + fp + fn + 1e-7)
    fg = [ious[c] for c in range(start, num_classes) if not np.isnan(ious[c])]
    return ious, float(np.mean(fg)) if fg else float("nan")

# ── load val volumes ──────────────────────────────────────────────────────────
def load_val_volumes():
    vols = []
    for h5p in sorted(Path(VAL_DIR).glob("*.h5")):
        if TEST_VOLUME is not None and TEST_VOLUME not in h5p.stem:
            continue
        with h5py.File(h5p, "r") as f:
            raw = f["image"][...].squeeze().astype(np.float32)
            lbl = f["label"][...].astype(np.uint8)
        raw = (raw - raw.mean()) / (raw.std() + 1e-8)
        vols.append((h5p.stem, raw, lbl))
        print(f"  {h5p.name}: raw={raw.shape}  label={lbl.shape}")
    return vols

# ── run one config ────────────────────────────────────────────────────────────
def eval_config(model, volumes, stride, use_tta, label):
    print(f"\n{'-'*60}")
    print(f"  Config: {label}")
    print(f"  patch={PATCH}  stride={stride}  TTA={use_tta}")
    print(f"{'-'*60}")

    all_ious = {c: [] for c in range(NUM_CLASSES)}
    total_t = 0

    for name, raw, gt in volumes:
        t0 = time.time()
        pred = sliding_window_predict(model, raw, PATCH, stride, use_tta)
        dt = time.time() - t0
        total_t += dt

        per_cls, mean = mean_iou(pred, gt)
        print(f"  {name:12s}: mean_IoU={mean:.4f}  [{dt:.1f}s]")
        for c in range(NUM_CLASSES):
            if not np.isnan(per_cls[c]):
                all_ious[c].append(per_cls[c])

    # aggregate
    print(f"\n  Per-class IoU (mean over {len(volumes)} volumes):")
    fg_means = []
    for c in range(NUM_CLASSES):
        vals = all_ious[c]
        m = float(np.mean(vals)) if vals else float("nan")
        tag = "(bg)" if c == 0 else "    "
        print(f"    [{c}] {CLASS_NAMES[c]:12s} {tag}: {m:.4f}")
        if c > 0 and not np.isnan(m):
            fg_means.append(m)
    mean_fg = float(np.mean(fg_means)) if fg_means else float("nan")
    print(f"\n  >>> Mean IoU (excl bg): {mean_fg:.4f}   total time: {total_t:.1f}s")
    return mean_fg

# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  eval_improvements.py — Inference config comparison")
    print("=" * 60)
    print(f"\n  Device: {DEVICE}")
    if DEVICE == "cpu":
        print("  [warn] CPU only — inference will be slow (esp. with TTA)")

    print("\n[1/2] Loading model...")
    model = load_model()

    print("\n[2/2] Loading val volumes...")
    volumes = load_val_volumes()

    results = {}

    # A — Baseline
    results["A_baseline"] = eval_config(
        model, volumes,
        stride=PATCH, use_tta=False,
        label="A — Baseline (no overlap, no TTA)"
    )

    # B — Overlap stride
    stride_half = tuple(p // 2 for p in PATCH)
    results["B_overlap"] = eval_config(
        model, volumes,
        stride=stride_half, use_tta=False,
        label="B — Overlap stride=patch//2"
    )

    # C — Overlap + TTA
    results["C_overlap_tta"] = eval_config(
        model, volumes,
        stride=stride_half, use_tta=True,
        label="C — Overlap + TTA (8 flips)"
    )

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY (relative improvement is the key signal)")
    print(f"{'='*60}")
    print(f"  A Baseline      : {results['A_baseline']:.4f}")
    print(f"  B Overlap       : {results['B_overlap']:.4f}  "
          f"(delta: {results['B_overlap']-results['A_baseline']:+.4f})")
    print(f"  C Overlap+TTA   : {results['C_overlap_tta']:.4f}  "
          f"(delta: {results['C_overlap_tta']-results['A_baseline']:+.4f})")
    print(f"\n  NOTE: Mamba bottleneck disabled (Windows fallback).")
    print(f"  Absolute IoU lower than RunPod 84.2%.")
    print(f"  Relative A→B→C improvement is the key signal.")
    print("=" * 60)
