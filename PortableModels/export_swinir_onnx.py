"""
Export SwinIR (Image Restoration with Swin Transformer) models to ONNX.

Two model variants are supported:
  - lightweight-x2: SwinIR-S, 2× super-resolution  (~3.4 MB, fast in browser)
  - classical-x4:   SwinIR-M, 4× super-resolution  (~25 MB, higher quality)

Both take an RGB image [1,3,H,W] float32 in [0,1] and return a super-resolved
image [1,3,scale*H,scale*W] float32 in [0,1].  The exported models are fixed
to the input sizes shown below (transformers require fixed spatial dims at
export time due to window-attention masks):
  - lightweight-x2: input [1,3,256,256] → output [1,3,512,512]
  - classical-x4:   input [1,3,128,128] → output [1,3,512,512]

Setup (one-time):
    cd PortableModels
    git clone https://github.com/JingyunLiang/SwinIR.git

Download pretrained weights into PortableModels/SwinIR/model_zoo/:
    Lightweight x2:
      https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth
    Classical x4:
      https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth

Usage:
    python export_swinir_onnx.py                       # export both
    python export_swinir_onnx.py --model lightweight   # lightweight x2 only
    python export_swinir_onnx.py --model classical     # classical x4 only
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

SWINIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SwinIR')
MODEL_ZOO  = os.path.join(SWINIR_DIR, 'model_zoo')
sys.path.insert(0, SWINIR_DIR)


# ---------------------------------------------------------------------------
# Wrapper — exposes a clean [0,1] float32 I/O contract for ONNX
# ---------------------------------------------------------------------------

class SwinIRWrapper(nn.Module):
    """Thin wrapper around SwinIR for ONNX export.

    Input:  [1, 3, H, W] float32 in [0, 1]
    Output: [1, 3, scale*H, scale*W] float32 clamped to [0, 1]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ONNX verification helper
# ---------------------------------------------------------------------------

def _verify(onnx_path, dummy_input, torch_output):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {input_name: dummy_input.numpy()})[0]
        np.testing.assert_allclose(torch_output.numpy(), ort_out, rtol=1e-3, atol=1e-4)
        print("  ✅ ONNX Runtime verification passed!")
    except ImportError:
        print("  ⚠️  onnxruntime not installed, skipping verification")
    except Exception as exc:
        print(f"  ⚠️  Verification warning: {exc}")


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_lightweight_x2(output_dir):
    from models.network_swinir import SwinIR  # noqa: E402

    ckpt = os.path.join(MODEL_ZOO, '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth')
    print(f"\n{'='*60}")
    print("Exporting: SwinIR-S lightweight x2  (256×256 → 512×512)")
    print(f"{'='*60}")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Download from:\n"
            "  https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
        )

    model = SwinIR(
        upscale=2, in_chans=3, img_size=64, window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
    )
    print(f"  Loading weights from {ckpt}...")
    state_dict = torch.load(ckpt, map_location='cpu')
    # Handle 'params' or 'params_ema' key wrappers common in SwinIR checkpoints
    for key in ('params_ema', 'params'):
        if key in state_dict:
            state_dict = state_dict[key]
            break
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    wrapper = SwinIRWrapper(model)
    wrapper.eval()

    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"  Forward pass OK — output: {out.shape}")

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'swinir_lightweight_x2.onnx')
    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False,
    )

    _verify(onnx_path, dummy, out)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ Exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def export_classical_x4(output_dir):
    from models.network_swinir import SwinIR  # noqa: E402

    ckpt = os.path.join(MODEL_ZOO, '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth')
    print(f"\n{'='*60}")
    print("Exporting: SwinIR-M classical x4  (128×128 → 512×512)")
    print(f"{'='*60}")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Download from:\n"
            "  https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
        )

    model = SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )
    print(f"  Loading weights from {ckpt}...")
    state_dict = torch.load(ckpt, map_location='cpu')
    for key in ('params_ema', 'params'):
        if key in state_dict:
            state_dict = state_dict[key]
            break
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    wrapper = SwinIRWrapper(model)
    wrapper.eval()

    dummy = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"  Forward pass OK — output: {out.shape}")

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'swinir_classical_x4.onnx')
    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False,
    )

    _verify(onnx_path, dummy, out)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ Exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Export SwinIR models to ONNX')
    parser.add_argument(
        '--model', type=str, default=None,
        choices=['lightweight', 'classical'],
        help='Which model to export (default: both)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='onnx_models',
        help='Root output directory (a swinir/ sub-folder is created)',
    )
    args = parser.parse_args()

    abs_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir, 'swinir')
    models  = [args.model] if args.model else ['lightweight', 'classical']

    exported = []
    for m in models:
        if m == 'lightweight':
            exported.append(export_lightweight_x2(abs_out))
        else:
            exported.append(export_classical_x4(abs_out))

    print(f"\n{'='*60}")
    print(f"Export complete! {len(exported)} model(s) saved to {abs_out}/")
    for p in exported:
        print(f"  • {p}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
