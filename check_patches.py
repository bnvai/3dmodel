import h5py, numpy as np, os, glob

train_dir = r"resources\dataset\train"
files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))

patch = (80, 170, 170)
stride = (20, 20, 20)
threshold = 0.06

total = border = interior = rejected = 0

for f in files[:3]:  # check first 3 files as sample
    with h5py.File(f, "r") as hf:
        label = hf["label"][:]
        Z, Y, X = label.shape
        fg = (label > 0).astype(np.float32)

        z_pos = range(0, max(1, Z - patch[0] + 1), stride[0])
        y_pos = range(0, max(1, Y - patch[1] + 1), stride[1])
        x_pos = range(0, max(1, X - patch[2] + 1), stride[2])

        n_y = len(list(y_pos))
        print(f"{os.path.basename(f)}: {Z}x{Y}x{X} | Y-positions: {n_y} (was 4)")

        for z in z_pos:
            for y in y_pos:
                for x in x_pos:
                    p = fg[z:z+patch[0], y:y+patch[1], x:x+patch[2]]
                    ratio = p.mean()
                    if ratio < threshold:
                        rejected += 1
                        continue
                    total += 1
                    if ratio < 0.25:
                        border += 1
                    else:
                        interior += 1

print(f"\n--- With stride [20,20,20], threshold={threshold} (sample: 3 files) ---")
print(f"Accepted patches:  {total}")
print(f"Rejected patches:  {rejected} (below threshold)")
print(f"Border (6-25% fg): {border} ({100*border/max(1,total):.1f}%)")
print(f"Interior (>25% fg): {interior} ({100*interior/max(1,total):.1f}%)")
