"""
evaluate.py — Comprehensive post-hoc test-set evaluation for pytorch3dunet.

Computes per-class and mean ± std (across volumes):
  DSC, IoU, Precision, Recall, HD95 (voxel + physical mm), ASSD

Surface-distance library: medpy.metric.binary  (pip install medpy)
  hd95 / assd delegate to medpy so we do not hand-roll distance transforms.

NaN convention (documented in CSV footnote):
  - Class absent in GT  (no GT voxels) → all metrics = NaN; excluded from mean.
  - Class present in GT, absent in pred → Dice/IoU/Recall = 0; Precision = NaN;
    HD95/ASSD = 95th-percentile of GT surface–boundary distance (worst-case).

Usage — standalone CLI:
  python -m pytorch3dunet.unet3d.evaluate \\
      --pred-dir  eval/predictions/ \\
      --gt-dir    dataset/test/ \\
      --num-classes 5 \\
      --output-dir eval/results/

  OR provide a config + checkpoint to run inference first:
  python -m pytorch3dunet.unet3d.evaluate \\
      --config  resources/3DSemiconductor/test_config_proposed.yaml \\
      --checkpoint checkpoints/best_checkpoint.pytorch \\
      --output-dir eval/results/

Usage — post-training hook (call from trainer or predict.py):
  from pytorch3dunet.unet3d.evaluate import run_evaluation
  run_evaluation(pred_dir, gt_dir, num_classes=5, output_dir="eval/")

Small-dataset note (≤15 volumes):
  With only ~11 volumes a fixed 3-volume test split produces unreliable
  statistics (high variance).  Recommended strategy:
    • 5-fold stratified cross-validation: each fold trains on ~9, tests on ~2;
      report mean ± std of metrics across the 5 × 2 = 10 test evaluations.
    • Leave-One-Out (LOO) gives 11 individual volume scores if compute allows.
  k-fold integration outline is in run_kfold_evaluation() at the bottom of
  this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# matplotlib must use a non-interactive backend when called from a script
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from scipy.ndimage import distance_transform_edt

from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("Evaluate")

# ─── optional dependency: medpy ─────────────────────────────────────────────

try:
    from medpy.metric.binary import hd95 as _medpy_hd95, assd as _medpy_assd
    _MEDPY_OK = True
except ImportError:
    _MEDPY_OK = False
    warnings.warn(
        "medpy not installed — HD95 and ASSD will be NaN. "
        "Install with: pip install medpy",
        stacklevel=2,
    )

# ─── optional dependency: pandas (for CSV) ──────────────────────────────────

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False
    warnings.warn("pandas not installed — CSV output disabled. pip install pandas")

# ─── constants ───────────────────────────────────────────────────────────────

# RGBA colors (0–255) for each class label in overlay figures.
# Index == class id.  Background is fully transparent.
# Designed for SAM chip-assembly classes; add more rows for >6 classes.
_CLASS_RGBA = np.array([
    [0,   0,   0,   0  ],   # 0  background  — transparent
    [180, 100, 40,  200],   # 1  base_cu     — copper brown
    [255, 215, 0,   200],   # 2  circuit     — gold / yellow
    [60,  100, 220, 200],   # 3  base_chip   — blue
    [60,  180, 60,  200],   # 4  chip        — green
    [220, 50,  50,  200],   # 5  solder      — red
], dtype=np.uint8)

# Default class names for the SAM chip-assembly 6-class task.
_DEFAULT_CLASS_NAMES_6 = [
    "background", "base_cu", "circuit", "base_chip", "chip", "solder"
]


def _default_class_names(num_classes: int) -> List[str]:
    if num_classes == 6:
        return _DEFAULT_CLASS_NAMES_6
    names = ["background"] + [f"segment_{i}" for i in range(1, num_classes)]
    return names[:num_classes]


# ─── spacing helper ──────────────────────────────────────────────────────────

def read_spacing(h5_path: str) -> Tuple[float, float, float]:
    """Read voxel spacing (z, y, x) in mm from the HDF5 'meta' group.

    Tries the following locations in order:
      1. h5['meta'].attrs['spacing']   — preferred (3-element array, mm)
      2. h5['meta/spacing'][:]         — dataset form
      3. h5.attrs['spacing']           — root-level attribute fallback
    Falls back to isotropic [1.0, 1.0, 1.0] with a warning if not found.

    Args:
        h5_path: Absolute path to the HDF5 file.

    Returns:
        Tuple (sz, sy, sx) in mm, e.g. (1.0, 1.0, 1.0) for isotropic 1 mm.
    """
    fallback = (1.0, 1.0, 1.0)
    with h5py.File(h5_path, "r") as f:
        # attempt 1: meta group attribute
        if "meta" in f and "spacing" in f["meta"].attrs:
            sp = np.asarray(f["meta"].attrs["spacing"], dtype=float)
            if sp.size == 3:
                return tuple(sp.tolist())
        # attempt 2: meta/spacing dataset
        if "meta" in f and "spacing" in f["meta"]:
            sp = np.asarray(f["meta/spacing"][:], dtype=float).ravel()
            if sp.size == 3:
                return tuple(sp.tolist())
        # attempt 3: root attribute
        if "spacing" in f.attrs:
            sp = np.asarray(f.attrs["spacing"], dtype=float)
            if sp.size == 3:
                return tuple(sp.tolist())

    logger.warning(
        f"No 'spacing' found in {h5_path}; assuming isotropic [1,1,1] mm. "
        "Physical surface-distance metrics will equal voxel-unit metrics."
    )
    return fallback


# ─── per-class metric computation ────────────────────────────────────────────

def _hd95_safe(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:
    """Hausdorff Distance 95th-percentile for one binary class mask.

    Delegates to medpy.metric.binary.hd95 when available.
    Handles the degenerate cases medpy cannot:
      - GT all-zero  → NaN  (class absent; exclude from mean)
      - Pred all-zero → worst-case estimate via GT surface DT

    Args:
        pred_bin: Boolean 3-D array (D, H, W) — predicted mask for one class.
        gt_bin:   Boolean 3-D array (D, H, W) — ground-truth mask for one class.
        spacing:  Voxel size (sz, sy, sx) in mm.

    Returns:
        HD95 in mm (or voxels if spacing == [1,1,1]).  NaN if GT absent.
    """
    if not gt_bin.any():
        return float("nan")

    if not pred_bin.any():
        # Worst case: surface of GT is the reference; distance to boundary
        # of GT measures how spread out the missed structure is.
        gt_surface = gt_bin ^ _erode(gt_bin)
        dt = distance_transform_edt(~gt_surface, sampling=spacing)
        dt_on_gt = dt[gt_bin]
        return float(np.percentile(dt_on_gt, 95)) if dt_on_gt.size else float("nan")

    if not _MEDPY_OK:
        return float("nan")

    try:
        return float(_medpy_hd95(pred_bin, gt_bin, voxelspacing=spacing))
    except Exception as exc:
        logger.debug(f"medpy hd95 failed: {exc}")
        return float("nan")


def _assd_safe(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:
    """Average Symmetric Surface Distance for one binary class mask.

    Args:
        pred_bin: Boolean 3-D array — predicted mask for one class.
        gt_bin:   Boolean 3-D array — ground-truth mask for one class.
        spacing:  Voxel size (sz, sy, sx) in mm.

    Returns:
        ASSD in mm.  NaN if GT or pred is absent.
    """
    if not gt_bin.any() or not pred_bin.any():
        return float("nan")
    if not _MEDPY_OK:
        return float("nan")
    try:
        return float(_medpy_assd(pred_bin, gt_bin, voxelspacing=spacing))
    except Exception as exc:
        logger.debug(f"medpy assd failed: {exc}")
        return float("nan")


def _erode(binary: np.ndarray) -> np.ndarray:
    """Binary erosion by 1 voxel (used to extract surface voxels)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(binary)


def compute_per_class_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float],
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """Compute DSC, IoU, Precision, Recall, HD95, ASSD for every class.

    Args:
        pred:        Integer label map (D, H, W), dtype uint16 or int.
        gt:          Integer label map (D, H, W), same shape as pred.
        spacing:     Voxel size (sz, sy, sx) in mm for physical distances.
        num_classes: Total number of classes including background (class 0).

    Returns:
        Dict mapping class_id → {dsc, iou, precision, recall,
                                   hd95_vox, hd95_mm, assd_mm}.
        NaN encodes "metric undefined" (see module docstring for convention).
    """
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    eps = 1e-7
    results: Dict[int, Dict[str, float]] = {}

    for c in range(num_classes):
        p = (pred == c)
        g = (gt   == c)

        tp = float((p & g).sum())
        fp = float((p & ~g).sum())
        fn = float((~p & g).sum())
        tn = float((~p & ~g).sum())

        gt_absent = not g.any()

        dsc       = float("nan") if gt_absent else (2 * tp) / (2 * tp + fp + fn + eps)
        iou       = float("nan") if gt_absent else tp / (tp + fp + fn + eps)
        precision = float("nan") if (tp + fp) == 0 else tp / (tp + fp + eps)
        recall    = float("nan") if gt_absent else tp / (tp + fn + eps)
        # specificity = TN / (TN + FP); background has huge TN, still meaningful
        specificity = float("nan") if (tn + fp) == 0 else tn / (tn + fp + eps)

        # absolute volume difference as % of GT volume (lower = better)
        gt_vol   = float(g.sum())
        pred_vol = float(p.sum())
        volume_diff_pct = (
            float("nan") if gt_absent
            else abs(pred_vol - gt_vol) / (gt_vol + eps) * 100.0
        )

        # voxel-unit HD95: use unit spacing
        hd95_vox = _hd95_safe(p, g, (1.0, 1.0, 1.0))
        hd95_mm  = _hd95_safe(p, g, spacing)
        assd_mm  = _assd_safe(p, g, spacing)

        results[c] = {
            "dsc":             dsc,
            "iou":             iou,
            "precision":       precision,
            "recall":          recall,
            "specificity":     specificity,
            "volume_diff_pct": volume_diff_pct,
            "hd95_vox":        hd95_vox,
            "hd95_mm":         hd95_mm,
            "assd_mm":         assd_mm,
        }

    return results


# ─── per-volume evaluation ───────────────────────────────────────────────────

def evaluate_volume_pair(
    pred_h5_path: str,
    gt_h5_path: str,
    num_classes: int,
    pred_dataset: str = "predictions",
    gt_dataset: str = "label",
) -> Dict:
    """Load one prediction + GT pair and compute all metrics.

    Prediction H5 is the file produced by StandardPredictor (save_segmentation=True),
    which stores argmax uint16 under key 'predictions'.
    Spacing is read from the GT H5 'meta' group so physical units are correct.

    Args:
        pred_h5_path: Path to *_predictions.h5 file from StandardPredictor.
        gt_h5_path:   Path to original HDF5 file containing ground-truth labels.
        num_classes:  Number of classes (including background).
        pred_dataset: HDF5 internal path for predictions. Default: 'predictions'.
        gt_dataset:   HDF5 internal path for ground truth. Default: 'label'.

    Returns:
        Dict with keys:
          'volume'   : stem name of the GT file
          'spacing'  : (sz, sy, sx) tuple in mm
          'metrics'  : output of compute_per_class_metrics()
          'raw_slice': middle axial slice of raw image (for visualisation)
          'gt'       : full GT volume (D, H, W) uint16
          'pred'     : full prediction volume (D, H, W) uint16
    """
    spacing = read_spacing(gt_h5_path)

    with h5py.File(pred_h5_path, "r") as f:
        pred = f[pred_dataset][...].astype(np.uint16)

    with h5py.File(gt_h5_path, "r") as f:
        gt   = f[gt_dataset][...].astype(np.uint16)
        # load raw image for qualitative figure (squeeze channel dim if present)
        raw_key = "image" if "image" in f else ("raw" if "raw" in f else None)
        raw = f[raw_key][...].squeeze() if raw_key else np.zeros_like(gt, dtype=np.float32)

    logger.info(f"Evaluating {Path(gt_h5_path).stem}  spacing={spacing}  "
                f"pred={pred.shape} gt={gt.shape}")

    per_class = compute_per_class_metrics(pred, gt, spacing, num_classes)

    return {
        "volume":  Path(gt_h5_path).stem,
        "spacing": spacing,
        "metrics": per_class,
        "raw":     raw,
        "gt":      gt,
        "pred":    pred,
    }


# ─── aggregation ─────────────────────────────────────────────────────────────

def aggregate_results(
    volume_results: List[Dict],
    num_classes: int,
    class_names: List[str],
) -> Dict:
    """Compute mean ± std across volumes for each class and each metric.

    NaN values (absent classes) are excluded from mean/std via np.nanmean.

    Args:
        volume_results: List of dicts returned by evaluate_volume_pair().
        num_classes:    Total number of classes.
        class_names:    Human-readable name for each class index.

    Returns:
        Dict with structure:
          per_class[class_id][metric] = {'mean': float, 'std': float,
                                          'values': [float, ...]}
          summary['mean_dsc_excl_bg'] = float  (mean DSC over classes 1..K-1)
          summary['mean_dsc_incl_bg'] = float
    """
    metric_keys = ["dsc", "iou", "precision", "recall", "specificity",
                   "volume_diff_pct", "hd95_vox", "hd95_mm", "assd_mm"]

    per_class: Dict[int, Dict] = {}
    for c in range(num_classes):
        per_class[c] = {"name": class_names[c]}
        for mk in metric_keys:
            vals = [v["metrics"][c][mk] for v in volume_results]
            arr  = np.array(vals, dtype=float)
            per_class[c][mk] = {
                "mean":   float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float("nan"),
                "std":    float(np.nanstd(arr))  if not np.all(np.isnan(arr)) else float("nan"),
                "values": [float(x) for x in vals],
            }

    # summary scalars (all exclude background class 0)
    def _excl_bg_mean(key):
        vals = np.array([per_class[c][key]["mean"] for c in range(1, num_classes)], float)
        return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")

    dsc_incl = np.array([per_class[c]["dsc"]["mean"] for c in range(num_classes)], float)

    summary = {
        "mean_dsc_excl_bg":  _excl_bg_mean("dsc"),
        "mean_dsc_incl_bg":  float(np.nanmean(dsc_incl)),
        "mean_iou_excl_bg":  _excl_bg_mean("iou"),
        "mean_hd95_excl_bg": _excl_bg_mean("hd95_mm"),
        "mean_assd_excl_bg": _excl_bg_mean("assd_mm"),
        "num_volumes": len(volume_results),
        "volumes": [v["volume"] for v in volume_results],
    }

    return {"per_class": per_class, "summary": summary}


# ─── CSV output ──────────────────────────────────────────────────────────────

def save_csv(agg: Dict, output_dir: str) -> str:
    """Save aggregated per-class metrics as a publication-ready CSV table.

    Rows are class names + a final 'Mean (excl. BG)' row.
    Columns are Dice, IoU, Precision, Recall, HD95 (vox), HD95 (mm), ASSD (mm),
    each reported as 'mean ± std'.

    Args:
        agg:        Output of aggregate_results().
        output_dir: Directory where results.csv is written.

    Returns:
        Absolute path to the written CSV file.
    """
    if not _PANDAS_OK:
        logger.warning("pandas not available; skipping CSV output.")
        return ""

    pc = agg["per_class"]
    rows = []
    col_map = [
        ("Dice (DSC)",        "dsc"),
        ("IoU (Jaccard)",     "iou"),
        ("Precision",         "precision"),
        ("Recall",            "recall"),
        ("Specificity",       "specificity"),
        ("Vol.Diff (%)",      "volume_diff_pct"),
        ("HD95 (vox)",        "hd95_vox"),
        ("HD95 (mm)",         "hd95_mm"),
        ("ASSD (mm)",         "assd_mm"),
    ]

    for c, info in pc.items():
        row = {"Class": info["name"]}
        for col_name, key in col_map:
            m, s = info[key]["mean"], info[key]["std"]
            row[col_name] = (f"{m:.4f} ± {s:.4f}"
                             if not (np.isnan(m) or np.isnan(s))
                             else "NaN *")
        rows.append(row)

    # Mean (excl. BG) row
    mean_row = {"Class": "Mean (excl. BG)"}
    for col_name, key in col_map:
        vals = [pc[c][key]["mean"] for c in range(1, len(pc))]
        arr  = np.array(vals, float)
        m    = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float("nan")
        mean_row[col_name] = f"{m:.4f}" if not np.isnan(m) else "NaN *"
    rows.append(mean_row)

    df = pd.DataFrame(rows).set_index("Class")
    out = os.path.join(output_dir, "results.csv")
    df.to_csv(out)
    # append footnote as a comment line
    with open(out, "a") as fh:
        fh.write("# NaN * = class absent from GT in ≥1 volume; excluded from mean\n")
    logger.info(f"Saved CSV: {out}")
    return out


# ─── LaTeX table output ──────────────────────────────────────────────────────

def save_latex_table(agg: Dict, output_dir: str, caption: str = "", label: str = "tab:results") -> str:
    """Write a publication-ready IEEE-style booktabs LaTeX table to results.tex.

    Metrics are expressed as:
      DSC (%), IoU (%), Precision (%), Recall (%), Specificity (%) — ×100
      HD95 (mm), ASSD (mm)                                          — physical
    Volume difference is omitted from the main table (report in text).

    Requires \\usepackage{booktabs} and \\usepackage{multirow} in the preamble.

    Args:
        agg:        Output of aggregate_results().
        output_dir: Directory where results.tex is written.
        caption:    LaTeX caption string (defaults to a generic caption).
        label:      LaTeX \\label key.

    Returns:
        Absolute path to the written .tex file.
    """
    if not caption:
        caption = (
            "Per-class segmentation performance on the SAM chip-assembly dataset. "
            "Values are mean\\,\\textpm\\,std across test volumes. "
            "Background is excluded from the mean row. "
            "HD95 and ASSD are reported in physical millimetres."
        )

    pc = agg["per_class"]
    s  = agg["summary"]

    def _pct(m, sd):
        if np.isnan(m) or np.isnan(sd):
            return r"---"
        return f"{m * 100:.1f} $\\pm$ {sd * 100:.1f}"

    def _mm(m, sd):
        if np.isnan(m) or np.isnan(sd):
            return r"---"
        return f"{m:.2f} $\\pm$ {sd:.2f}"

    def _mean_pct(key):
        v = s.get(f"mean_{key}_excl_bg", float("nan"))
        return f"{v * 100:.1f}" if not np.isnan(v) else "---"

    def _mean_mm(key):
        v = s.get(f"mean_{key}_excl_bg", float("nan"))
        return f"{v:.2f}" if not np.isnan(v) else "---"

    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{multirow} in preamble",
        r"\begin{table*}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{l cc ccc cc}",
        r"    \toprule",
        r"    \multirow{2}{*}{Class}"
        r"      & \multicolumn{2}{c}{Overlap}"
        r"      & \multicolumn{3}{c}{Voxel-level}"
        r"      & \multicolumn{2}{c}{Boundary (mm)} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8}",
        r"    & DSC (\%) & IoU (\%)"
        r"    & Precision (\%) & Recall (\%) & Specificity (\%)"
        r"    & HD95 & ASSD \\",
        r"    \midrule",
    ]

    for c, info in pc.items():
        name = info["name"].replace("_", "\\_")
        dsc  = _pct(info["dsc"]["mean"],         info["dsc"]["std"])
        iou  = _pct(info["iou"]["mean"],         info["iou"]["std"])
        prec = _pct(info["precision"]["mean"],   info["precision"]["std"])
        rec  = _pct(info["recall"]["mean"],      info["recall"]["std"])
        spec = _pct(info["specificity"]["mean"], info["specificity"]["std"])
        hd   = _mm( info["hd95_mm"]["mean"],     info["hd95_mm"]["std"])
        assd = _mm( info["assd_mm"]["mean"],     info["assd_mm"]["std"])
        lines.append(
            f"    {name} & {dsc} & {iou} & {prec} & {rec} & {spec} & {hd} & {assd} \\\\"
        )

    lines += [
        r"    \midrule",
        f"    \\textbf{{Mean (excl.\\ BG)}}"
        f" & \\textbf{{{_mean_pct('dsc')}}}"
        f" & \\textbf{{{_mean_pct('iou')}}}"
        f" & --- & --- & ---"
        f" & \\textbf{{{_mean_mm('hd95')}}}"
        f" & \\textbf{{{_mean_mm('assd')}}} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"\end{table*}",
    ]

    out = os.path.join(output_dir, "results.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info(f"Saved LaTeX table: {out}")
    return out


# ─── per-volume CSV ───────────────────────────────────────────────────────────

def save_per_volume_csv(volume_results: List[Dict], class_names: List[str], output_dir: str) -> str:
    """Save individual per-volume DSC scores as a CSV for supplementary material.

    Rows = volumes, columns = per-class DSC + mean DSC (excl. BG).
    Reviewers sometimes request this table to verify consistency across subjects.

    Args:
        volume_results: List of per-volume dicts from evaluate_volume_pair().
        class_names:    Human-readable class names.
        output_dir:     Directory where per_volume_dsc.csv is written.

    Returns:
        Absolute path to the written CSV file.
    """
    if not _PANDAS_OK:
        logger.warning("pandas not available; skipping per-volume CSV.")
        return ""

    rows = []
    for vr in volume_results:
        row: Dict = {"Volume": vr["volume"]}
        dsc_vals = []
        for c, name in enumerate(class_names):
            v = vr["metrics"][c]["dsc"]
            row[f"DSC_{name}"] = f"{v:.4f}" if not np.isnan(v) else "NaN"
            if c > 0 and not np.isnan(v):
                dsc_vals.append(v)
        row["Mean_DSC_excl_BG"] = f"{np.mean(dsc_vals):.4f}" if dsc_vals else "NaN"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Volume")
    out = os.path.join(output_dir, "per_volume_dsc.csv")
    df.to_csv(out)
    logger.info(f"Saved per-volume DSC CSV: {out}")
    return out


# ─── JSON output ─────────────────────────────────────────────────────────────

def save_json(agg: Dict, volume_results: List[Dict], output_dir: str) -> str:
    """Save all raw numbers (per-volume per-class) to results.json for reproducibility.

    Args:
        agg:            Output of aggregate_results().
        volume_results: Raw per-volume dicts (without numpy arrays).
        output_dir:     Directory where results.json is written.

    Returns:
        Absolute path to the written JSON file.
    """
    # strip non-serialisable numpy arrays from volume_results before saving
    slim = []
    for vr in volume_results:
        slim.append({
            "volume":  vr["volume"],
            "spacing": list(vr["spacing"]),
            "metrics": {str(c): vr["metrics"][c] for c in vr["metrics"]},
        })

    payload = {"aggregated": agg, "per_volume": slim}
    out = os.path.join(output_dir, "results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)
    logger.info(f"Saved JSON: {out}")
    return out


# ─── confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    volume_results: List[Dict],
    class_names: List[str],
    output_dir: str,
) -> str:
    """Save a normalised voxel-level confusion matrix heatmap to PNG.

    Confusion matrix is accumulated across ALL volumes, then row-normalised
    (rows = GT class, columns = predicted class) so values are recall-style
    fractions summing to 1 per row.

    Args:
        volume_results: List of per-volume dicts with 'gt' and 'pred' arrays.
        class_names:    Human-readable name per class.
        output_dir:     Directory where confusion_matrix.png is saved.

    Returns:
        Path to saved PNG.
    """
    K = len(class_names)
    cm = np.zeros((K, K), dtype=np.int64)
    for vr in volume_results:
        gt   = vr["gt"].ravel().astype(int)
        pred = vr["pred"].ravel().astype(int)
        np.add.at(cm, (np.clip(gt, 0, K-1), np.clip(pred, 0, K-1)), 1)

    # row-normalise
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(K * 1.4 + 1, K * 1.4 + 1), dpi=300)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(K)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Ground Truth", fontsize=10)
    ax.set_title("Normalised Confusion Matrix (row = recall fraction)", fontsize=11)

    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.6 else "black",
                    fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix: {out}")
    return out


# ─── Dice box / violin plot ───────────────────────────────────────────────────

def plot_dice_boxplot(
    agg: Dict,
    class_names: List[str],
    output_dir: str,
) -> str:
    """Save a per-class Dice violin+box plot across all test volumes to PNG.

    Uses violinplot for distribution shape and overlaid boxplot for quartile
    readability.  Classes with all-NaN Dice (absent in all GT) are skipped.

    Args:
        agg:         Output of aggregate_results().
        class_names: Human-readable class names.
        output_dir:  Directory where dice_boxplot.png is saved.

    Returns:
        Path to saved PNG.
    """
    K = len(class_names)
    data, labels, colors = [], [], []

    palette = ["#888888", "#E63946", "#2A9D8F", "#457B9D", "#E9C46A", "#9B59B6"]
    for c in range(K):
        vals = [v for v in agg["per_class"][c]["dsc"]["values"] if not np.isnan(v)]
        if not vals:
            continue
        data.append(vals)
        labels.append(class_names[c])
        colors.append(palette[c % len(palette)])

    if not data:
        logger.warning("No valid Dice values found; skipping boxplot.")
        return ""

    fig, ax = plt.subplots(figsize=(max(5, len(data) * 1.5), 5), dpi=300)
    positions = list(range(1, len(data) + 1))

    parts = ax.violinplot(data, positions=positions, showmedians=False,
                          showextrema=False)
    for pc_body, col in zip(parts["bodies"], colors):
        pc_body.set_facecolor(col); pc_body.set_alpha(0.4)

    bp = ax.boxplot(data, positions=positions, widths=0.3,
                    patch_artist=True, medianprops={"color": "black", "linewidth": 2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Dice Similarity Coefficient", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Per-class DSC across test volumes", fontsize=11)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "dice_boxplot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved dice boxplot: {out}")
    return out


# ─── qualitative overlays ────────────────────────────────────────────────────

def _label_to_rgba(label_vol: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer label volume (D,H,W) → RGBA volume (D,H,W,4) uint8."""
    rgba = np.zeros(label_vol.shape + (4,), dtype=np.uint8)
    max_c = min(num_classes, len(_CLASS_RGBA))
    for c in range(max_c):
        mask = label_vol == c
        rgba[mask] = _CLASS_RGBA[c]
    return rgba


def plot_qualitative(
    volume_result: Dict,
    class_names: List[str],
    output_dir: str,
    qual_dir: str = "qualitative",
) -> List[str]:
    """Save axial / coronal / sagittal GT-vs-prediction overlay figures.

    For each of the three anatomical planes, a figure with 3 columns is saved:
      Col 1: raw image (grey)
      Col 2: raw + GT overlay (coloured)
      Col 3: raw + prediction overlay (coloured)

    Mid-slice indices are used for each plane.

    Args:
        volume_result: Single dict from evaluate_volume_pair().
        class_names:   Human-readable class names.
        output_dir:    Root output directory; figures go into output_dir/qual_dir/.
        qual_dir:      Sub-directory name for qualitative figures.

    Returns:
        List of paths to saved PNG files.
    """
    num_classes = len(class_names)
    raw  = volume_result["raw"].astype(float)
    gt   = volume_result["gt"]
    pred = volume_result["pred"]
    name = volume_result["volume"]

    # normalise raw to [0, 1]
    r_min, r_max = raw.min(), raw.max()
    raw_norm = (raw - r_min) / (r_max - r_min + 1e-8)

    gt_rgba   = _label_to_rgba(gt,   num_classes)
    pred_rgba = _label_to_rgba(pred, num_classes)

    D, H, W = raw_norm.shape
    planes = [
        ("axial",    raw_norm[D//2], gt_rgba[D//2],   pred_rgba[D//2]),
        ("coronal",  raw_norm[:,H//2,:], gt_rgba[:,H//2,:], pred_rgba[:,H//2,:]),
        ("sagittal", raw_norm[:,:,W//2], gt_rgba[:,:,W//2], pred_rgba[:,:,W//2]),
    ]

    out_dir = os.path.join(output_dir, qual_dir)
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    legend_patches = [
        mpatches.Patch(color=np.array(_CLASS_RGBA[c]) / 255.0, label=class_names[c])
        for c in range(1, min(num_classes, len(_CLASS_RGBA)))
    ]

    for plane_name, raw_slice, gt_slice, pred_slice in planes:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=300)

        for ax in axes:
            ax.axis("off")

        # col 1 — raw
        axes[0].imshow(raw_slice, cmap="gray", interpolation="nearest")
        axes[0].set_title("Input", fontsize=9)

        # col 2 — GT overlay
        axes[1].imshow(raw_slice, cmap="gray", interpolation="nearest")
        axes[1].imshow(gt_slice, interpolation="nearest")
        axes[1].set_title("Ground Truth", fontsize=9)

        # col 3 — prediction overlay
        axes[2].imshow(raw_slice, cmap="gray", interpolation="nearest")
        axes[2].imshow(pred_slice, interpolation="nearest")
        axes[2].set_title("Prediction", fontsize=9)

        if legend_patches:
            fig.legend(handles=legend_patches, loc="lower center",
                       ncol=min(num_classes, 6), fontsize=7,
                       bbox_to_anchor=(0.5, -0.05))

        fig.suptitle(f"{name} — {plane_name} mid-slice", fontsize=10)
        plt.tight_layout()

        out = os.path.join(out_dir, f"{name}_{plane_name}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        logger.info(f"Saved qualitative figure: {out}")

    return saved


# ─── console summary ─────────────────────────────────────────────────────────

def print_summary(agg: Dict, class_names: List[str]):
    """Print a compact metrics table to stdout.

    Args:
        agg:         Output of aggregate_results().
        class_names: Human-readable class names.
    """
    header = f"{'Class':<20}  {'DSC':>8}  {'IoU':>8}  {'Prec':>8}  {'Rec':>8}  {'HD95(mm)':>10}  {'ASSD(mm)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    pc = agg["per_class"]
    for c, info in pc.items():
        def _fmt(key):
            m = info[key]["mean"]
            return f"{m:.4f}" if not np.isnan(m) else "  NaN "
        print(f"{info['name']:<20}  {_fmt('dsc'):>8}  {_fmt('iou'):>8}  "
              f"{_fmt('precision'):>8}  {_fmt('recall'):>8}  "
              f"{_fmt('hd95_mm'):>10}  {_fmt('assd_mm'):>10}")

    print("-" * len(header))
    s = agg["summary"]

    def _sfmt(key, scale=1.0):
        v = s.get(key, float("nan"))
        return "  NaN " if np.isnan(v) else f"{v * scale:.4f}"

    print(f"{'Mean DSC (excl. BG)':<20}  {_sfmt('mean_dsc_excl_bg'):>8}"
          f"  {'Mean IoU (excl. BG)':<20}  {_sfmt('mean_iou_excl_bg'):>8}")
    print(f"{'Mean HD95 (mm, excl.BG)':<20}  {_sfmt('mean_hd95_excl_bg'):>8}"
          f"  {'Mean ASSD (mm, excl.BG)':<20}  {_sfmt('mean_assd_excl_bg'):>8}")
    print("=" * len(header) + "\n")


# ─── main orchestrator ───────────────────────────────────────────────────────

def run_evaluation(
    pred_dir: str,
    gt_dir: str,
    num_classes: int,
    output_dir: str,
    class_names: Optional[List[str]] = None,
    pred_suffix: str = "_predictions",
    pred_dataset: str = "predictions",
    gt_dataset: str = "label",
    num_qual_volumes: int = 3,
) -> Dict:
    """Run full evaluation: metrics + all visualisations + CSV/JSON.

    Matches prediction H5 files in pred_dir with GT H5 files in gt_dir by
    stem name (GT stem == pred stem without pred_suffix).

    Args:
        pred_dir:         Directory containing *_predictions.h5 files.
        gt_dir:           Directory containing original GT .h5 files.
        num_classes:      Total number of classes including background.
        output_dir:       Root directory for all output files.
        class_names:      Optional list of length num_classes.  Defaults to
                          ['Background', 'Segment_1', …].
        pred_suffix:      Suffix that StandardPredictor appends before '.h5'.
                          Default: '_predictions'.
        pred_dataset:     HDF5 key for predictions inside pred H5.
        gt_dataset:       HDF5 key for ground truth inside GT H5.
        num_qual_volumes: How many volumes to produce qualitative figures for.
                          Selects first N by alphabetical GT filename order.

    Returns:
        Aggregated metrics dict (same as aggregate_results() output).
    """
    if class_names is None:
        class_names = _default_class_names(num_classes)
    assert len(class_names) == num_classes, \
        f"class_names length {len(class_names)} != num_classes {num_classes}"

    os.makedirs(output_dir, exist_ok=True)

    # ── match prediction files to GT files ───────────────────────────────────
    gt_paths   = sorted(Path(gt_dir).glob("*.h5"))
    pred_paths_map: Dict[str, Path] = {
        p.stem.replace(pred_suffix, ""): p
        for p in Path(pred_dir).glob(f"*{pred_suffix}.h5")
    }

    if not gt_paths:
        raise FileNotFoundError(f"No .h5 files found in gt_dir: {gt_dir}")
    if not pred_paths_map:
        raise FileNotFoundError(
            f"No *{pred_suffix}.h5 files found in pred_dir: {pred_dir}"
        )

    volume_results: List[Dict] = []
    for gt_path in gt_paths:
        stem = gt_path.stem
        if stem not in pred_paths_map:
            logger.warning(f"No prediction found for {stem}; skipping.")
            continue
        vr = evaluate_volume_pair(
            pred_h5_path=str(pred_paths_map[stem]),
            gt_h5_path=str(gt_path),
            num_classes=num_classes,
            pred_dataset=pred_dataset,
            gt_dataset=gt_dataset,
        )
        volume_results.append(vr)

    if not volume_results:
        raise RuntimeError("No matched prediction/GT pairs found.")

    logger.info(f"Evaluated {len(volume_results)} volumes.")

    # ── aggregate ────────────────────────────────────────────────────────────
    agg = aggregate_results(volume_results, num_classes, class_names)

    # ── outputs ──────────────────────────────────────────────────────────────
    print_summary(agg, class_names)
    save_csv(agg, output_dir)
    save_latex_table(agg, output_dir)
    save_per_volume_csv(volume_results, class_names, output_dir)
    save_json(agg, volume_results, output_dir)
    plot_confusion_matrix(volume_results, class_names, output_dir)
    plot_dice_boxplot(agg, class_names, output_dir)

    for vr in volume_results[:num_qual_volumes]:
        plot_qualitative(vr, class_names, output_dir)

    logger.info(f"Evaluation complete. All outputs in: {output_dir}")
    return agg


# ─── k-fold outline ──────────────────────────────────────────────────────────

def run_kfold_evaluation(
    all_h5_paths: List[str],
    num_classes: int,
    output_dir: str,
    k: int = 5,
    class_names: Optional[List[str]] = None,
):
    """Outline for k-fold cross-validation evaluation (≤15 volumes).

    With 11 volumes:
      k=5 → 5 folds of ~9 train + ~2 test; reports mean ± std over 10 test scores.
      k=11 (LOO) → 11 individual volume evaluations, maximum reliability.

    This function assumes you have already trained k separate models and
    saved predictions for each fold's test split into fold-specific directories.

    Args:
        all_h5_paths: Sorted list of ALL h5 file paths in the dataset.
        num_classes:  Total number of classes.
        output_dir:   Root for all fold outputs.
        k:            Number of folds.
        class_names:  Optional class names.

    NOTE: This is an organisational outline, not a training loop.
    Integrate with your training script by: for each fold, train the model,
    run StandardPredictor on the test fold, then call run_evaluation() below.
    """
    if class_names is None:
        class_names = _default_class_names(num_classes)

    n = len(all_h5_paths)
    fold_size = n // k
    fold_aggs = []

    logger.info(f"k-fold outline: {n} volumes, k={k}, ~{fold_size} test per fold")

    for fold_idx in range(k):
        test_indices = list(range(fold_idx * fold_size,
                                  min((fold_idx + 1) * fold_size, n)))
        train_indices = [i for i in range(n) if i not in test_indices]

        fold_out = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_out, exist_ok=True)

        logger.info(
            f"Fold {fold_idx}: train={[all_h5_paths[i] for i in train_indices]}, "
            f"test={[all_h5_paths[i] for i in test_indices]}"
        )
        # ── INSERT HERE: ───────────────────────────────────────────────────
        # 1. Train model on train_indices → save checkpoint to fold_out/
        # 2. Run StandardPredictor on test_indices → *_predictions.h5 in fold_out/pred/
        # 3. Call run_evaluation(pred_dir=fold_out/pred, gt_dir=..., ...)
        # ──────────────────────────────────────────────────────────────────
        logger.info(f"(Fold {fold_idx} training/prediction not implemented here.)")

    if fold_aggs:
        all_dsc = [fa["summary"]["mean_dsc_excl_bg"] for fa in fold_aggs]
        logger.info(
            f"k-fold DSC (excl. BG): "
            f"{np.mean(all_dsc):.4f} ± {np.std(all_dsc):.4f}  "
            f"across {k} folds"
        )


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate pytorch3dunet segmentation predictions against GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir",     required=True,
                   help="Directory containing *_predictions.h5 files.")
    p.add_argument("--gt-dir",       required=True,
                   help="Directory containing original GT .h5 files.")
    p.add_argument("--num-classes",  type=int, default=6,
                   help="Total number of classes (including background class 0).")
    p.add_argument("--output-dir",   default="eval_results",
                   help="Directory for all outputs.")
    p.add_argument("--class-names",  nargs="+", default=None,
                   help="Optional space-separated class names, length = num-classes.")
    p.add_argument("--pred-suffix",  default="_predictions",
                   help="Suffix StandardPredictor appends before .h5.")
    p.add_argument("--pred-dataset", default="predictions",
                   help="HDF5 internal key for predictions.")
    p.add_argument("--gt-dataset",   default="label",
                   help="HDF5 internal key for ground truth.")
    p.add_argument("--num-qual",     type=int, default=3,
                   help="Number of volumes for qualitative overlay figures.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        class_names=args.class_names,
        pred_suffix=args.pred_suffix,
        pred_dataset=args.pred_dataset,
        gt_dataset=args.gt_dataset,
        num_qual_volumes=args.num_qual,
    )
