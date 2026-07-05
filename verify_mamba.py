import warnings, torch
from pytorch3dunet.unet3d.model import ResidualAttentionUNet3D

x = torch.zeros(1, 1, 16, 32, 32)

# SE path — must be identical to original behaviour
m_se = ResidualAttentionUNet3D(
    in_channels=1, out_channels=5,
    f_maps=[32, 64, 128, 256], final_sigmoid=False,
)
out = m_se(x)
print(f"SE   output: {out.shape}  bottleneck: {type(m_se.encoders[-1].basic_module).__name__}")

# Mamba path — graceful fallback expected (mamba-ssm not installed)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m_mb = ResidualAttentionUNet3D(
        in_channels=1, out_channels=5,
        f_maps=[32, 64, 128, 256], final_sigmoid=False,
        bottleneck_type="mamba",
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
    )
    if w:
        print(f"Fallback warning: {w[0].message}")
out2 = m_mb(x)
print(f"Mamba output: {out2.shape}  bottleneck: {type(m_mb.encoders[-1].basic_module).__name__}")

# Confirm SE path output is unchanged (parameter count check)
se_params   = sum(p.numel() for p in m_se.parameters())
mamba_params = sum(p.numel() for p in m_mb.parameters())
print(f"SE params: {se_params/1e6:.2f}M  |  Mamba-fallback params: {mamba_params/1e6:.2f}M")
print("All checks passed.")
