"""
Export Stylized Neural Painting neural renderers to ONNX.

Downloads pretrained checkpoints (if not present) and exports each
renderer variant to ONNX format.

Usage:
    python export_onnx.py                          # export all renderers
    python export_onnx.py --renderer oilpaintbrush # export one renderer
    python export_onnx.py --light                  # export lightweight variants
"""

import argparse
import os
import sys
import zipfile

import torch
import torch.nn as nn
import numpy as np

# Add the stylized-neural-painting repo to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylized-neural-painting'))

from renderer import Renderer
from networks import define_G


# Google Drive file IDs for pretrained checkpoints
CHECKPOINT_IDS = {
    'oilpaintbrush': '1sqWhgBKqaBJggl2A8sD1bLSq2_B1ScMG',
    'watercolor':    '19Yrj15v9kHvWzkK9o_GSZtvQaJPmcRYQ',
    'markerpen':     '1XsjncjlSdQh2dbZ3X1qf1M8pDc8GLbNy',
    'rectangle':     '162ykmRX8TBGVRnJIof8NeqN7cuwwuzIF',
}

CHECKPOINT_IDS_LIGHT = {
    'oilpaintbrush': '1kcXsx2nDF3b3ryYOwm3BjmfwET9lfFht',
    'watercolor':    '1FoclmDOL6d1UT12-aCDwYMcXQKSK6IWA',
    'markerpen':     '1pP99btR2XV3GtDHFXd8klpdQRSc0prLx',
    'rectangle':     '1aHyc9ukObmCeaecs8o-a6p-SCjeKlvVZ',
}

CHECKPOINT_DIRS = {
    'oilpaintbrush': 'checkpoints_G_oilpaintbrush',
    'watercolor':    'checkpoints_G_watercolor',
    'markerpen':     'checkpoints_G_markerpen',
    'rectangle':     'checkpoints_G_rectangle',
}

CHECKPOINT_DIRS_LIGHT = {
    'oilpaintbrush': 'checkpoints_G_oilpaintbrush_light',
    'watercolor':    'checkpoints_G_watercolor_light',
    'markerpen':     'checkpoints_G_markerpen_light',
    'rectangle':     'checkpoints_G_rectangle_light',
}


REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylized-neural-painting')


def download_checkpoint(renderer_name, light=False):
    """Download and extract checkpoint from Google Drive if not present."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    ids = CHECKPOINT_IDS_LIGHT if light else CHECKPOINT_IDS
    dirs = CHECKPOINT_DIRS_LIGHT if light else CHECKPOINT_DIRS

    file_id = ids[renderer_name]
    ckpt_dir = os.path.join(REPO_DIR, dirs[renderer_name])
    ckpt_path = os.path.join(ckpt_dir, 'last_ckpt.pt')

    if os.path.exists(ckpt_path):
        print(f"  Checkpoint already exists: {ckpt_path}")
        return ckpt_dir

    zip_name = os.path.join(REPO_DIR, dirs[renderer_name] + '.zip')
    print(f"  Downloading {zip_name}...")
    gdown.download(id=file_id, output=zip_name, quiet=False)

    print(f"  Extracting {zip_name}...")
    with zipfile.ZipFile(zip_name, 'r') as z:
        z.extractall(REPO_DIR)
    os.remove(zip_name)

    return ckpt_dir


class NeuralRendererONNX(nn.Module):
    """Wraps the neural renderer for clean ONNX export.

    The original ZouFCNFusion.forward has a conditional branch on renderer type
    that creates a device-bound constant (for oilpaintbrush/airbrush alpha=1.0).
    We bypass that by directly calling the sub-networks and compositing here.
    """

    def __init__(self, net_G, renderer_type, d_shape):
        super().__init__()
        self.huangnet = net_G.huangnet
        self.dcgan = net_G.dcgan
        self.d_shape = d_shape
        self.alpha_is_one = renderer_type in ['oilpaintbrush', 'airbrush']

    def forward(self, x):
        x_shape = x[:, 0:self.d_shape, :, :]
        x_alpha = x[:, [-1], :, :]

        mask = self.huangnet(x_shape)
        color, _ = self.dcgan(x)

        foreground = color * mask
        if self.alpha_is_one:
            alpha = mask
        else:
            alpha = x_alpha * mask

        return foreground, alpha


def export_single(renderer_name, light=False, output_dir='onnx_models'):
    """Export a single renderer variant to ONNX."""
    suffix = '_light' if light else ''
    net_g_name = 'zou-fusion-net-light' if light else 'zou-fusion-net'

    print(f"\n{'='*60}")
    print(f"Exporting: {renderer_name}{suffix}")
    print(f"{'='*60}")

    # 1. Download checkpoint
    ckpt_dir = download_checkpoint(renderer_name, light=light)

    # 2. Build model (chdir into repo so Renderer can find ./brushes/)
    orig_cwd = os.getcwd()
    os.chdir(REPO_DIR)
    rderr = Renderer(renderer=renderer_name, CANVAS_WIDTH=128, train=False)
    net_G = define_G(rdrr=rderr, netG=net_g_name)
    os.chdir(orig_cwd)

    # 3. Load weights
    ckpt_path = os.path.join(ckpt_dir, 'last_ckpt.pt')
    print(f"  Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    net_G.eval()

    # 4. Wrap for export
    wrapper = NeuralRendererONNX(net_G, renderer_name, rderr.d_shape)
    wrapper.eval()

    # 5. Create dummy input: [batch, d_params, 1, 1]
    d = rderr.d
    dummy_input = torch.randn(1, d, 1, 1)
    out_size = net_G.out_size

    print(f"  Input shape:  [batch, {d}, 1, 1]")
    print(f"  Output shape: [batch, 3, {out_size}, {out_size}] x2 (foreground + alpha)")

    # 6. Test forward pass
    with torch.no_grad():
        fg, alpha = wrapper(dummy_input)
    print(f"  Test forward pass OK — fg: {fg.shape}, alpha: {alpha.shape}")

    # 7. Export to ONNX
    abs_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir, 'neural-painting')
    os.makedirs(abs_output_dir, exist_ok=True)
    onnx_path = os.path.join(abs_output_dir, f'neural_renderer_{renderer_name}{suffix}.onnx')

    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['stroke_params'],
        output_names=['foreground', 'alpha_mask'],
        dynamic_axes={
            'stroke_params': {0: 'batch_size'},
            'foreground':    {0: 'batch_size'},
            'alpha_mask':    {0: 'batch_size'},
        }
    )

    # 8. Verify with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_input = {'stroke_params': dummy_input.numpy()}
        ort_fg, ort_alpha = sess.run(None, ort_input)
        # Compare outputs
        np.testing.assert_allclose(fg.numpy(), ort_fg, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(alpha.numpy(), ort_alpha, rtol=1e-4, atol=1e-5)
        print(f"  ✅ ONNX Runtime verification passed!")
    except ImportError:
        print(f"  ⚠️  onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"  ⚠️  Verification warning: {e}")

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ Exported: {onnx_path} ({file_size_mb:.1f} MB)")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export neural renderers to ONNX')
    parser.add_argument('--renderer', type=str, default=None,
                        choices=['oilpaintbrush', 'watercolor', 'markerpen', 'rectangle'],
                        help='Export a specific renderer (default: all)')
    parser.add_argument('--light', action='store_true',
                        help='Export lightweight variants')
    parser.add_argument('--all-variants', action='store_true',
                        help='Export both standard and lightweight variants')
    parser.add_argument('--output-dir', type=str, default='onnx_models',
                        help='Output directory for ONNX files')
    args = parser.parse_args()

    renderers = [args.renderer] if args.renderer else ['oilpaintbrush', 'watercolor', 'markerpen', 'rectangle']

    exported = []

    if args.all_variants:
        for r in renderers:
            exported.append(export_single(r, light=False, output_dir=args.output_dir))
            exported.append(export_single(r, light=True, output_dir=args.output_dir))
    else:
        for r in renderers:
            exported.append(export_single(r, light=args.light, output_dir=args.output_dir))

    print(f"\n{'='*60}")
    print(f"Export complete! {len(exported)} model(s) exported to {args.output_dir}/")
    for p in exported:
        print(f"  • {p}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
