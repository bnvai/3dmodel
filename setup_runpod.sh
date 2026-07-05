#!/bin/bash
# RunPod setup script for pytorch-3dunet V3 training
# Run this ONCE after pod starts:  bash /workspace/setup_runpod.sh

set -e
echo "=== pytorch-3dunet RunPod Setup ==="

# 1. Install system deps
apt-get update -qq
apt-get install -y -qq build-essential git

# 2. Install mamba-ssm (requires Linux + CUDA)
echo "--- Installing mamba-ssm ---"
pip install mamba-ssm --quiet
python -c "import mamba_ssm; print('mamba-ssm OK:', mamba_ssm.__version__)"

# 3. Install pytorch-3dunet
echo "--- Installing pytorch-3dunet ---"
cd /workspace/pytorch-3dunet
pip install -e . --quiet

# 4. Verify everything
echo "--- Verification ---"
python -c "
import torch
import mamba_ssm
from pytorch3dunet.unet3d.model import get_model

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('mamba-ssm:', mamba_ssm.__version__)

cfg = {
    'name': 'ResidualAttentionUNet3D',
    'in_channels': 1, 'out_channels': 6,
    'layer_order': 'cge', 'f_maps': [64, 128, 256, 512],
    'num_groups': 8, 'final_sigmoid': False,
    'is_segmentation': True, 'dropout_prob': 0.2,
    'bottleneck_type': 'mamba',
}
model = get_model(cfg).cuda()
total = sum(p.numel() for p in model.parameters())
print(f'Model params: {total/1e6:.2f}M')

import torch
x = torch.randn(1, 1, 80, 170, 170).cuda()
with torch.no_grad():
    y = model(x)
print(f'Forward pass OK: input {list(x.shape)} -> output {list(y.shape)}')
"

echo ""
echo "=== Setup complete! Run training with: ==="
echo "python -m pytorch3dunet.train --config /workspace/train_config_proposed_v3_runpod.yaml"
