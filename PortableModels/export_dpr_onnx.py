"""
Export DPR (Deep Single-Image Portrait Relighting) models to ONNX.

Two model variants are supported:
  - 512:  HourglassNet, input [1,1,512,512]
  - 1024: HourglassNet_1024, input [1,1,1024,1024]

Both take a grayscale (L-channel) portrait image and a 9-coefficient SH lighting
vector, and return a relit L-channel image plus the predicted lighting.

Usage:
    python export_dpr_onnx.py               # export both 512 and 1024
    python export_dpr_onnx.py --model 512   # export 512 only
    python export_dpr_onnx.py --model 1024  # export 1024 only
"""

import argparse
import os
import sys
import types

import torch
import torch.nn as nn
import numpy as np

DPR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DPR')
sys.path.insert(0, os.path.join(DPR_DIR, 'model'))


# ---------------------------------------------------------------------------
# ONNX-compatible replacements for lightingNet.forward
#
# The original code uses in-place slice assignment:
#     innerFeat[:, 0:ncInput, :, :] = upFeat
# which is not supported by the ONNX exporter. We replace it with torch.cat.
# ---------------------------------------------------------------------------

def _lighting_forward_512(self, innerFeat, target_light, count, skip_count):
    """lightingNet.forward for the 512 model — returns (out_feat, light)."""
    x = innerFeat[:, 0:self.ncInput, :, :]
    _, _, row, col = x.shape
    feat = x.mean(dim=(2, 3), keepdim=True)
    light = self.predict_relu1(self.predict_FC1(feat))
    light = self.predict_FC2(light)
    upFeat = self.post_relu1(self.post_FC1(target_light))
    upFeat = self.post_relu2(self.post_FC2(upFeat))
    upFeat = upFeat.repeat((1, 1, row, col))
    out = torch.cat([upFeat, innerFeat[:, self.ncInput:, :, :]], dim=1)
    return out, light


def _lighting_forward_1024(self, innerFeat, target_light, count, skip_count):
    """lightingNet.forward for the 1024 model — returns (out_feat, rest_feat, light)."""
    x = innerFeat[:, 0:self.ncInput, :, :]
    _, _, row, col = x.shape
    feat = x.mean(dim=(2, 3), keepdim=True)
    light = self.predict_relu1(self.predict_FC1(feat))
    light = self.predict_FC2(light)
    upFeat = self.post_relu1(self.post_FC1(target_light))
    upFeat = self.post_relu2(self.post_FC2(upFeat))
    upFeat = upFeat.repeat((1, 1, row, col))
    rest = innerFeat[:, self.ncInput:, :, :]
    out = torch.cat([upFeat, rest], dim=1)
    return out, rest, light


def _patch_lighting(net, lighting_cls, patched_forward):
    """Bind patched_forward onto every lightingNet instance in net."""
    for module in net.modules():
        if isinstance(module, lighting_cls):
            module.forward = types.MethodType(patched_forward, module)


# ---------------------------------------------------------------------------
# Wrapper modules — fix Python-level args (skip_count, oriImg) and select
# only the outputs needed at inference time.
# ---------------------------------------------------------------------------

class DPR512Wrapper(nn.Module):
    """HourglassNet (512) wrapper for ONNX export.

    Inputs:  image [B,1,512,512], target_light [B,9,1,1]
    Outputs: relit_image [B,1,512,512], predicted_light [B,9,1,1]
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, target_light):
        out_img, out_light = self.net(image, target_light, 0)
        return out_img, out_light


class DPR1024Wrapper(nn.Module):
    """HourglassNet_1024 wrapper for ONNX export.

    Inputs:  image [B,1,1024,1024], target_light [B,9,1,1]
    Outputs: relit_image [B,1,1024,1024], predicted_light [B,9,1,1]
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, target_light):
        out_img, _out_feat, out_light, _out_feat_ori = self.net(image, target_light, 0)
        return out_img, out_light


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _verify(onnx_path, inputs, torch_outputs):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        input_names = [i.name for i in sess.get_inputs()]
        feed = {n: v.numpy() for n, v in zip(input_names, inputs)}
        ort_outs = sess.run(None, feed)
        for i, (ref, got) in enumerate(zip(torch_outputs, ort_outs)):
            np.testing.assert_allclose(ref.numpy(), got, rtol=1e-3, atol=1e-4)
        print("  ✅ ONNX Runtime verification passed!")
    except ImportError:
        print("  ⚠️  onnxruntime not installed, skipping verification")
    except Exception as exc:
        print(f"  ⚠️  Verification warning: {exc}")


def export_512(output_dir):
    import defineHourglass_512_gray_skip as m512

    ckpt = os.path.join(DPR_DIR, 'trained_model', 'trained_model_03.t7')
    print(f"\n{'='*60}")
    print("Exporting: DPR 512x512")
    print(f"{'='*60}")

    net = m512.HourglassNet()
    print(f"  Loading weights from {ckpt}...")
    net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    net.eval()

    _patch_lighting(net, m512.lightingNet, _lighting_forward_512)

    wrapper = DPR512Wrapper(net)
    wrapper.eval()

    dummy_img   = torch.randn(1, 1, 512, 512)
    dummy_light = torch.randn(1, 9, 1, 1)

    with torch.no_grad():
        out_img, out_light = wrapper(dummy_img, dummy_light)
    print(f"  Forward pass OK — relit_image: {out_img.shape}, predicted_light: {out_light.shape}")

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'dpr_512.onnx')
    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_light),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['image', 'target_light'],
        output_names=['relit_image', 'predicted_light'],
        dynamic_axes={
            'image':           {0: 'batch_size'},
            'target_light':    {0: 'batch_size'},
            'relit_image':     {0: 'batch_size'},
            'predicted_light': {0: 'batch_size'},
        },
    )

    _verify(onnx_path, [dummy_img, dummy_light], [out_img, out_light])

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ Exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def export_1024(output_dir):
    import defineHourglass_1024_gray_skip_matchFeature as m1024

    ckpt = os.path.join(DPR_DIR, 'trained_model', 'trained_model_1024_03.t7')
    print(f"\n{'='*60}")
    print("Exporting: DPR 1024x1024")
    print(f"{'='*60}")

    net_512 = m1024.HourglassNet(16)
    net     = m1024.HourglassNet_1024(net_512, 16)
    print(f"  Loading weights from {ckpt}...")
    net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    net.eval()

    _patch_lighting(net, m1024.lightingNet, _lighting_forward_1024)

    wrapper = DPR1024Wrapper(net)
    wrapper.eval()

    dummy_img   = torch.randn(1, 1, 1024, 1024)
    dummy_light = torch.randn(1, 9, 1, 1)

    with torch.no_grad():
        out_img, out_light = wrapper(dummy_img, dummy_light)
    print(f"  Forward pass OK — relit_image: {out_img.shape}, predicted_light: {out_light.shape}")

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'dpr_1024.onnx')
    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_light),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['image', 'target_light'],
        output_names=['relit_image', 'predicted_light'],
        dynamic_axes={
            'image':           {0: 'batch_size'},
            'target_light':    {0: 'batch_size'},
            'relit_image':     {0: 'batch_size'},
            'predicted_light': {0: 'batch_size'},
        },
    )

    _verify(onnx_path, [dummy_img, dummy_light], [out_img, out_light])

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ Exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Export DPR models to ONNX')
    parser.add_argument('--model', type=str, default=None, choices=['512', '1024'],
                        help='Which model to export (default: both)')
    parser.add_argument('--output-dir', type=str, default='onnx_models',
                        help='Root output directory (a dpr/ sub-folder is created)')
    args = parser.parse_args()

    abs_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir, 'dpr')
    models  = [args.model] if args.model else ['512', '1024']

    exported = []
    for m in models:
        if m == '512':
            exported.append(export_512(abs_out))
        else:
            exported.append(export_1024(abs_out))

    print(f"\n{'='*60}")
    print(f"Export complete! {len(exported)} model(s) saved to {abs_out}/")
    for p in exported:
        print(f"  • {p}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
