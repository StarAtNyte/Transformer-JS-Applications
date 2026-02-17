"""
Export UHDM / ESDNet (Image Demoiréing) models to ONNX.

Two model variants are supported:
  - esdnet:   ESDNet   (SAM_NUMBER=1, lightweight)
  - esdnet-l: ESDNet-L (SAM_NUMBER=2, larger / more accurate)

Both take an RGB image [1,3,H,W] (H,W must be multiples of 32) and return
a demoiréd RGB image [1,3,H,W].

Checkpoints are expected under  UHDM/pretrain_model/.
Run  `bash UHDM/scripts/download_model.sh`  first if they are missing.

Usage:
    python export_uhdm_onnx.py                     # export both variants
    python export_uhdm_onnx.py --model esdnet       # export lightweight only
    python export_uhdm_onnx.py --model esdnet-l     # export large only
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

UHDM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UHDM')
sys.path.insert(0, UHDM_DIR)

from model.nets import my_model  # noqa: E402


# ---------------------------------------------------------------------------
# Wrapper — only return the first (full-res) output
# ---------------------------------------------------------------------------

class ESDNetWrapper(nn.Module):
    """Wraps my_model so ONNX export only sees the primary output."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out_1, _out_2, _out_3 = self.net(x)
        return out_1


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

VARIANTS = {
    'esdnet': {
        'sam_number': 1,
        'checkpoints': [
            ('uhdm',    'uhdm_checkpoint.pth'),
            ('fhdmi',   'fhdmi_checkpoint.pth'),
            ('tip',     'tip_checkpoint.pth'),
            ('aim',     'aim_checkpoint.pth'),
        ],
    },
    'esdnet-l': {
        'sam_number': 2,
        'checkpoints': [
            ('uhdm',    'uhdm_large_checkpoint.pth'),
            ('fhdmi',   'fhdmi_large_checkpoint.pth'),
            ('tip',     'tip_large_checkpoint.pth'),
            ('aim',     'aim_large_checkpoint.pth'),
        ],
    },
}


def _verify(onnx_path, dummy_input, torch_output):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {input_name: dummy_input.numpy()})[0]
        np.testing.assert_allclose(torch_output.numpy(), ort_out, rtol=1e-3, atol=1e-4)
        print("  ✅ ONNX Runtime verification passed!")
    except ImportError:
        print("  ⚠️  onnxruntime not installed, skipping verification")
    except Exception as exc:
        print(f"  ⚠️  Verification warning: {exc}")


def export_variant(variant_name, output_dir):
    cfg = VARIANTS[variant_name]
    sam_number = cfg['sam_number']
    pretrain_dir = os.path.join(UHDM_DIR, 'pretrain_model')
    exported = []

    for dataset_tag, ckpt_name in cfg['checkpoints']:
        ckpt_path = os.path.join(pretrain_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            print(f"  ⚠️  Skipping {ckpt_name} (not found at {ckpt_path})")
            continue

        print(f"\n{'='*60}")
        print(f"Exporting: {variant_name} / {dataset_tag}")
        print(f"{'='*60}")

        # Build model
        net = my_model(
            en_feature_num=48,
            en_inter_num=32,
            de_feature_num=64,
            de_inter_num=32,
            sam_number=sam_number,
        )

        print(f"  Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        net.load_state_dict(state_dict)
        net.eval()

        wrapper = ESDNetWrapper(net)
        wrapper.eval()

        # Use 512×512 for export (must be multiple of 32)
        dummy = torch.randn(1, 3, 512, 512)

        with torch.no_grad():
            out = wrapper(dummy)
        print(f"  Forward pass OK — input: {dummy.shape}, output: {out.shape}")

        os.makedirs(output_dir, exist_ok=True)
        onnx_name = f'{variant_name}_{dataset_tag}.onnx'
        onnx_path = os.path.join(output_dir, onnx_name)
        print(f"  Exporting to {onnx_path}...")

        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['output'],
            dynamic_axes={
                'image':  {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 2: 'height', 3: 'width'},
            },
        )

        _verify(onnx_path, dummy, out)

        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        # Check for external data file
        data_path = onnx_path + '.data'
        if os.path.isfile(data_path):
            size_mb += os.path.getsize(data_path) / (1024 * 1024)
        print(f"  ✅ Exported: {onnx_path} ({size_mb:.1f} MB)")
        exported.append(onnx_path)

    return exported


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Export UHDM/ESDNet models to ONNX')
    parser.add_argument('--model', type=str, default=None, choices=['esdnet', 'esdnet-l'],
                        help='Which variant to export (default: both)')
    parser.add_argument('--output-dir', type=str, default='onnx_models',
                        help='Root output directory (a uhdm/ sub-folder is created)')
    args = parser.parse_args()

    abs_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir, 'uhdm')
    variants = [args.model] if args.model else ['esdnet', 'esdnet-l']

    all_exported = []
    for v in variants:
        all_exported.extend(export_variant(v, abs_out))

    print(f"\n{'='*60}")
    if all_exported:
        print(f"Export complete! {len(all_exported)} model(s) saved to {abs_out}/")
        for p in all_exported:
            print(f"  • {p}")
    else:
        print("No models exported. Download checkpoints first:")
        print(f"  cd {UHDM_DIR} && bash scripts/download_model.sh")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
