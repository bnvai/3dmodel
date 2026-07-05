"""
Prepare 5-fold cross-validation dataset structure for pytorch-3dunet,
using the exact same fold splits as nnUNet's splits_final.json.

Mapping SAMChip_XXX -> filename.h5 (alphabetical Python sort order):
  000=Sample1  001=Sample2  002=Sample3  003=Sample4  004=Sample6
  005=s1       006=s10      007=s11      008=s2       009=s3
  010=s4       011=s5

Test set (never used in training): s6.h5, s7.h5, Sample5.h5

Output structure:
  /workspace/dataset_folds/
    fold_0/train/  <- symlinks to 9 HDF5 files
    fold_0/val/    <- symlinks to 3 HDF5 files
    fold_1/train/
    fold_1/val/
    ...
    fold_4/train/
    fold_4/val/
    test/          <- symlinks to s6, s7, Sample5

Run on RunPod:
  python /workspace/prepare_folds.py
"""

import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_SRC = Path("/workspace/dataset/train")   # original 12 HDF5 files
VAL_SRC   = Path("/workspace/dataset/val")     # original 3 test HDF5 files
OUTPUT    = Path("/workspace/dataset_folds")

# ── ID → filename mapping (from alphabetical sort of 15 folders) ──────────────
ID_TO_FILE = {
    "SAMChip_000": "Sample1.h5",
    "SAMChip_001": "Sample2.h5",
    "SAMChip_002": "Sample3.h5",
    "SAMChip_003": "Sample4.h5",
    "SAMChip_004": "Sample6.h5",
    "SAMChip_005": "s1.h5",
    "SAMChip_006": "s10.h5",
    "SAMChip_007": "s11.h5",
    "SAMChip_008": "s2.h5",
    "SAMChip_009": "s3.h5",
    "SAMChip_010": "s4.h5",
    "SAMChip_011": "s5.h5",
}

TEST_FILES = ["s6.h5", "s7.h5", "Sample5.h5"]

# ── 5-fold splits (from nnUNet splits_final.json) ────────────────────────────
FOLDS = [
    {
        "train": ["SAMChip_001","SAMChip_002","SAMChip_004","SAMChip_005",
                  "SAMChip_006","SAMChip_007","SAMChip_009","SAMChip_010","SAMChip_011"],
        "val":   ["SAMChip_000","SAMChip_003","SAMChip_008"],
    },
    {
        "train": ["SAMChip_000","SAMChip_001","SAMChip_002","SAMChip_003",
                  "SAMChip_004","SAMChip_005","SAMChip_006","SAMChip_008","SAMChip_009"],
        "val":   ["SAMChip_007","SAMChip_010","SAMChip_011"],
    },
    {
        "train": ["SAMChip_000","SAMChip_001","SAMChip_002","SAMChip_003",
                  "SAMChip_004","SAMChip_005","SAMChip_007","SAMChip_008","SAMChip_010","SAMChip_011"],
        "val":   ["SAMChip_006","SAMChip_009"],
    },
    {
        "train": ["SAMChip_000","SAMChip_002","SAMChip_003","SAMChip_005",
                  "SAMChip_006","SAMChip_007","SAMChip_008","SAMChip_009","SAMChip_010","SAMChip_011"],
        "val":   ["SAMChip_001","SAMChip_004"],
    },
    {
        "train": ["SAMChip_000","SAMChip_001","SAMChip_003","SAMChip_004",
                  "SAMChip_006","SAMChip_007","SAMChip_008","SAMChip_009","SAMChip_010","SAMChip_011"],
        "val":   ["SAMChip_002","SAMChip_005"],
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def make_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main():
    # Test set
    test_dir = OUTPUT / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for fname in TEST_FILES:
        src = VAL_SRC / fname
        make_symlink(src, test_dir / fname)
        print(f"[TEST] {fname}")

    # 5 folds
    for fold_idx, fold in enumerate(FOLDS):
        print(f"\n── Fold {fold_idx} ──────────────────────")

        for split in ("train", "val"):
            split_dir = OUTPUT / f"fold_{fold_idx}" / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for sample_id in fold[split]:
                fname = ID_TO_FILE[sample_id]
                src = TRAIN_SRC / fname
                dst = split_dir / fname
                make_symlink(src, dst)
                print(f"  [{split.upper()}] {sample_id} -> {fname}")

    # Print summary
    print(f"\n{'='*55}")
    print(f"Output: {OUTPUT}")
    print(f"\nFold summary:")
    for i, fold in enumerate(FOLDS):
        print(f"  Fold {i}: {len(fold['train'])} train, {len(fold['val'])} val")
    print(f"  Test : {len(TEST_FILES)} volumes (held-out)")
    print(f"\nNext: run training for each fold:")
    for i in range(5):
        print(f"  python -m pytorch3dunet.train --config /workspace/train_config_anisotropic_fold{i}.yaml")


if __name__ == "__main__":
    main()
