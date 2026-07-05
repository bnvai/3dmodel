"""
Split HDF5 files in `origin/` into `train/` and `val/` directories.

Usage:
    python split_dataset.py                     # default 80/20 split
    python split_dataset.py --val-ratio 0.15    # 85/15 split
    python split_dataset.py --seed 99           # different random seed
    python split_dataset.py --dry-run           # preview without copying
"""

import argparse
import random
import shutil
from pathlib import Path


DATASET_DIR = Path(__file__).parent / "dataset"
ORIGIN_DIR = DATASET_DIR / "origin"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"


def split_dataset(val_ratio: float = 0.2, seed: int = 42, dry_run: bool = False):
    h5_files = sorted(ORIGIN_DIR.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {ORIGIN_DIR}")

    rng = random.Random(seed)
    shuffled = h5_files[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_ratio))
    val_files = set(shuffled[:n_val])
    train_files = [f for f in shuffled if f not in val_files]
    val_files = list(shuffled[:n_val])

    print(f"Total: {len(h5_files)} files  |  train: {len(train_files)}  |  val: {len(val_files)}")
    print(f"Seed: {seed}  |  val_ratio: {val_ratio}")
    print()
    print("Train:")
    for f in sorted(train_files, key=lambda p: p.name):
        print(f"  {f.name}")
    print("Val:")
    for f in sorted(val_files, key=lambda p: p.name):
        print(f"  {f.name}")

    if dry_run:
        print("\n[dry-run] No files were copied.")
        return

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        dest = TRAIN_DIR / f.name
        shutil.copy2(f, dest)
        print(f"  copied → train/{f.name}")

    for f in val_files:
        dest = VAL_DIR / f.name
        shutil.copy2(f, dest)
        print(f"  copied → val/{f.name}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split origin HDF5 files into train/val.")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of files to use for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the split without copying any files")
    args = parser.parse_args()

    split_dataset(val_ratio=args.val_ratio, seed=args.seed, dry_run=args.dry_run)
