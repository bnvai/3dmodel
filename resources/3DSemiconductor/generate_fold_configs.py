"""
Generate 5 training configs for AnisotropicUNet3D (one per fold).
Run locally on Windows, then upload all configs to RunPod.

Usage:
  python generate_fold_configs.py
"""

from pathlib import Path

TEMPLATE = """\
# AnisotropicUNet3D — Fold {fold_idx} / 5  (nnUNet-aligned CV split)
#
# Train: {n_train} volumes   Val: {n_val} volumes   Test: s6, s7, Sample5 (held-out)
# Fold split matches nnUNet splits_final.json fold {fold_idx} exactly.

model:
  name: AnisotropicUNet3D
  in_channels: 1
  out_channels: 6
  f_maps: [ 48, 96, 192, 384 ]
  num_groups: 8
  final_sigmoid: false
  is_segmentation: true
  dropout_prob: 0.2

trainer:
  checkpoint_dir: /workspace/checkpoints_anisotropic_fold{fold_idx}
  resume: null
  pre_trained: null
  validate_after_iters: 1000
  log_after_iters: 500
  max_num_epochs: 200
  max_num_iterations: 100000
  eval_score_higher_is_better: true
  early_stopping_patience: 20
  max_val_images: 20

loss:
  name: WCEDiceLoss
  alpha: 0.5

optimizer:
  name: SGD
  learning_rate: 0.01
  momentum: 0.99
  nesterov: true
  weight_decay: 0.00003

eval_metric:
  name: MeanIoU
  skip_background: true

lr_scheduler:
  name: CosineAnnealingLR
  T_max: 100
  eta_min: 0.000001

loaders:
  dataset: LazyHDF5Dataset
  batch_size: 1
  num_workers: 4
  raw_internal_path: image
  label_internal_path: label
  global_normalization: false

  train:
    file_paths:
      - /workspace/dataset_folds/fold_{fold_idx}/train

    slice_builder:
      name: FilterSliceBuilder
      patch_shape: [ 80, 170, 170 ]
      stride_shape: [ 20, 20, 20 ]
      threshold: 0.01
      slack_acceptance: 0.33

    transformer:
      raw:
        - name: Standardize
        - name: RandomFlip
        - name: RandomRotate90
        - name: RandomRotate
          axes: [ [ 2, 1 ] ]
          angle_spectrum: 30
          mode: reflect
        - name: ElasticDeformation
          spline_order: 3
        - name: AdditiveGaussianNoise
          scale: [ 0, 0.1 ]
        - name: ToTensor
          expand_dims: true
      label:
        - name: RandomFlip
        - name: RandomRotate90
        - name: RandomRotate
          axes: [ [ 2, 1 ] ]
          angle_spectrum: 30
          mode: reflect
        - name: ElasticDeformation
          spline_order: 0
        - name: ToTensor
          expand_dims: false
          dtype: int64

  val:
    file_paths:
      - /workspace/dataset_folds/fold_{fold_idx}/val

    slice_builder:
      name: FilterSliceBuilder
      patch_shape: [ 80, 170, 170 ]
      stride_shape: [ 80, 170, 170 ]
      threshold: 0.01
      slack_acceptance: 0.01

    transformer:
      raw:
        - name: Standardize
        - name: ToTensor
          expand_dims: true
      label:
        - name: ToTensor
          expand_dims: false
          dtype: int64
"""

FOLD_SIZES = [
    (9, 3),  # fold 0
    (9, 3),  # fold 1
    (10, 2), # fold 2
    (10, 2), # fold 3
    (10, 2), # fold 4
]

OUT_DIR = Path(__file__).parent

for fold_idx, (n_train, n_val) in enumerate(FOLD_SIZES):
    content = TEMPLATE.format(fold_idx=fold_idx, n_train=n_train, n_val=n_val)
    out_path = OUT_DIR / f"train_config_anisotropic_fold{fold_idx}_runpod.yaml"
    out_path.write_text(content, encoding="utf-8")
    print(f"Created: {out_path.name}  (train={n_train}, val={n_val})")

print("\nDone. Upload all configs to RunPod:")
print("  scp -P <PORT> train_config_anisotropic_fold*.yaml root@<HOST>:/workspace/")
