"""
Prepare 5-fold dataset structure for Windows (copies instead of symlinks).
Creates dataset_folds/ next to the existing dataset/ folder.

Run once:
  python prepare_folds_windows.py
"""

import shutil
from pathlib import Path

TRAIN_SRC = Path(r"D:\Working\10. AI-Workspace\model\pytorch-3dunet\resources\dataset\train")
VAL_SRC   = Path(r"D:\Working\10. AI-Workspace\model\pytorch-3dunet\resources\dataset\val")
OUTPUT    = Path(r"D:\Working\10. AI-Workspace\model\pytorch-3dunet\resources\dataset_folds")

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


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  copied  {src.name}")
    else:
        print(f"  exists  {src.name}")


def main():
    # Test set
    test_dir = OUTPUT / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    print("── Test set ─────────────────────────────")
    for fname in TEST_FILES:
        copy_file(VAL_SRC / fname, test_dir / fname)

    # 5 folds
    for fold_idx, fold in enumerate(FOLDS):
        print(f"\n── Fold {fold_idx} ({len(fold['train'])} train / {len(fold['val'])} val) ──")
        for split in ("train", "val"):
            split_dir = OUTPUT / f"fold_{fold_idx}" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in fold[split]:
                fname = ID_TO_FILE[sample_id]
                copy_file(TRAIN_SRC / fname, split_dir / fname)

    print(f"\nDone → {OUTPUT}")
    print("Note: HDF5 files are copied (not symlinked) on Windows.")
    print("Total disk usage ≈ 5 × 12 × file_size")


if __name__ == "__main__":
    main()
