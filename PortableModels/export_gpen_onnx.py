"""
Export GPEN models to ONNX.

Supported model variants:
  - bfr-256:            FullGenerator 256x256, channel_mult=1, narrow=0.5
  - bfr-512:            FullGenerator 512x512, channel_mult=2, narrow=1.0
  - colorization-1024:  FullGenerator 1024x1024, channel_mult=2, narrow=1.0

Input:  [1, 3, size, size]  float32, RGB, normalized to [-1, 1]
Output: [1, 3, size, size]  float32, RGB, normalized to [-1, 1]

Weights are downloaded automatically if not present.

Usage:
    python export_gpen_onnx.py                          # export all
    python export_gpen_onnx.py --model bfr-512          # export BFR-512 only
    python export_gpen_onnx.py --model colorization-1024  # export colorization only
"""

import argparse
import os
import sys
import types
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GPEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GPEN')

# Add face_model and op to path (op must be on path for upfirdn2d / fused_act imports)
sys.path.insert(0, os.path.join(GPEN_DIR, 'face_model'))
sys.path.insert(0, os.path.join(GPEN_DIR, 'face_model', 'op'))

from gpen_model import FullGenerator, ConstantInput, ModulatedConv2d


WEIGHT_BASE_URL = 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/'

# model name -> (weight_name, in_size, channel_multiplier, narrow)
MODEL_CONFIGS = {
    'bfr-256': ('GPEN-BFR-256', 256, 1, 0.5),
    'bfr-512': ('GPEN-BFR-512', 512, 2, 1.0),
    'colorization-1024': ('GPEN-Colorization-1024', 1024, 2, 1.0),
}


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f'  Downloading {os.path.basename(dest)}...')

    def _progress(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f'\r  {pct}%', end='', flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


class GPENWrapper(nn.Module):
    """Wraps FullGenerator to return only the restored image (drops the latent)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        image, _latent = self.model(x)
        return image


def _verify(onnx_path, dummy_input, torch_output):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {input_name: dummy_input.numpy()})[0]
        np.testing.assert_allclose(torch_output.numpy(), ort_out, rtol=1e-3, atol=1e-3)
        print('  ✅ ONNX Runtime verification passed!')
    except ImportError:
        print('  ⚠️  onnxruntime not installed, skipping verification')
    except Exception as e:
        print(f'  ⚠️  Verification warning: {e}')


def export(variant, output_dir):
    model_name, in_size, channel_multiplier, narrow = MODEL_CONFIGS[variant]
    weight_path = os.path.join(GPEN_DIR, 'weights', model_name + '.pth')

    print(f"\n{'='*60}")
    print(f"Exporting: {model_name} ({in_size}x{in_size})")
    print(f"{'='*60}")

    if not os.path.exists(weight_path):
        url = WEIGHT_BASE_URL + model_name + '.pth'
        _download(url, weight_path)

    # Force CPU — custom CUDA ops (upfirdn2d, fused_leaky_relu) fall back
    # to pure PyTorch on CPU, which exports to standard ONNX ops cleanly.
    print(f'  Loading weights from {weight_path}...')
    model = FullGenerator(in_size, 512, 8, channel_multiplier, narrow=narrow, device='cpu')
    state = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    wrapper = GPENWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, in_size, in_size)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f'  Forward pass OK — output: {out.shape}')

    os.makedirs(output_dir, exist_ok=True)
    onnx_filename = model_name.lower().replace('-', '_') + '.onnx'
    onnx_path = os.path.join(output_dir, onnx_filename)

    print(f'  Exporting to {onnx_path}...')
    # ConstantInput: self.input.repeat(batch, 1, 1, 1) emits onnx::Tile with a symbolic
    # count, making every downstream shape dynamic. Return self.input directly (batch=1).
    def _constant_input_b1(self, input):
        return self.input

    # ModulatedConv2d: all uses of `batch` (from input.shape[0]) become aten::size nodes
    # in the ONNX graph. The reshaped weight kernel ends up with a `*` dimension, causing
    # "convolution for kernel of unknown shape". Replace every batch-scaled reshape/groups
    # with the literal integer 1.
    def _modulated_conv2d_b1(self, input, style):
        _, in_channel, height, width = input.shape

        style = self.modulation(style).view(1, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(1, self.out_channel, 1, 1, 1)

        weight = weight.view(self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, in_channel, height, width)
            weight = weight.view(1, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=1)
            _, _, height, width = out.shape
            out = out.view(1, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=1)
            _, _, height, width = out.shape
            out = out.view(1, self.out_channel, height, width)

        else:
            input = input.view(1, in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=1)
            _, _, height, width = out.shape
            out = out.view(1, self.out_channel, height, width)

        return out

    for m in wrapper.modules():
        if isinstance(m, ConstantInput):
            m.forward = types.MethodType(_constant_input_b1, m)
        elif isinstance(m, ModulatedConv2d):
            m.forward = types.MethodType(_modulated_conv2d_b1, m)

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False,
    )

    _verify(onnx_path, dummy, out)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f'  ✅ Exported: {onnx_path} ({size_mb:.1f} MB)')
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export GPEN models to ONNX')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), default=None,
                        help='Which model to export (default: all)')
    parser.add_argument('--output-dir', default='onnx_models/gpen',
                        help='Output directory (default: onnx_models/gpen/)')
    args = parser.parse_args()

    abs_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    variants = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    exported = [p for p in (export(v, abs_out) for v in variants) if p]

    print(f"\n{'='*60}")
    print(f'Export complete! {len(exported)} model(s) saved to {abs_out}/')
    for p in exported:
        print(f'  • {p}')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
