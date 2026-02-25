"""
Export DDSP decoder to ONNX for browser-based timbre transfer.

Two-step approach:
  1. TF checkpoint → SavedModel (via AutoencoderInference)
  2. SavedModel → ONNX (via tf2onnx)

Pretrained models can be downloaded from GCS:
  gsutil -m cp -r gs://ddsp/models/timbre_transfer/solo_violin ./trained_models/

Usage:
    python export_ddsp_onnx.py --model_dir ./trained_models/solo_violin
    python export_ddsp_onnx.py --model_dir ./trained_models/solo_violin --name violin
    python export_ddsp_onnx.py --model_dir ./trained_models/solo_flute --name flute

Requirements:
    pip install tensorflow ddsp gin-config tf2onnx onnxruntime
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import tensorflow as tf

DDSP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ddsp')
sys.path.insert(0, DDSP_DIR)


# ---------------------------------------------------------------------------
# Wrapper — converts dict-based decoder I/O to plain tensor args/outputs
# so SavedModel export captures a clean signature for tf2onnx.
# ---------------------------------------------------------------------------

class DecoderWrapper(tf.Module):
    """Wraps an RnnFcDecoder for SavedModel / ONNX export.

    Inputs:  ld_scaled [1, T, 1], f0_scaled [1, T, 1]
    Outputs: amps [1, T, 1], harmonic_distribution [1, T, 60],
             noise_magnitudes [1, T, 65]
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, 1], tf.float32, name='ld_scaled'),
        tf.TensorSpec([1, None, 1], tf.float32, name='f0_scaled'),
    ])
    def __call__(self, ld_scaled, f0_scaled):
        outputs = self.decoder(ld_scaled, f0_scaled)
        return {
            'amps': outputs['amps'],
            'harmonic_distribution': outputs['harmonic_distribution'],
            'noise_magnitudes': outputs['noise_magnitudes'],
        }


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def _verify(onnx_path, n_frames):
    """Run a quick forward pass through the exported ONNX model."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]

        dummy_ld = np.random.randn(1, n_frames, 1).astype(np.float32)
        dummy_f0 = np.random.randn(1, n_frames, 1).astype(np.float32)

        feed = {}
        for name in input_names:
            if 'ld' in name:
                feed[name] = dummy_ld
            else:
                feed[name] = dummy_f0

        ort_outs = sess.run(None, feed)
        print(f"  ✅ ONNX Runtime verification passed!")
        for name, out in zip(output_names, ort_outs):
            print(f"     {name}: {out.shape}")
    except ImportError:
        print("  ⚠️  onnxruntime not installed, skipping verification")
    except Exception as exc:
        print(f"  ⚠️  Verification warning: {exc}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_ddsp(model_dir, name, output_dir, length_seconds):
    """Export the DDSP decoder from a checkpoint directory to ONNX."""
    from ddsp.training import inference

    print(f"\n{'=' * 60}")
    print(f"Exporting: DDSP decoder — {name}")
    print(f"{'=' * 60}")

    # ---- Step 1: Load the full autoencoder via DDSP inference API ----
    print(f"  Loading model from {model_dir}...")
    model = inference.AutoencoderInference(
        ckpt=model_dir,
        length_seconds=length_seconds,
        remove_reverb=True,
    )

    sample_rate = model.sample_rate
    hop_size = model.hop_size
    n_frames = model.n_frames
    n_samples = model.n_samples
    decoder = model.decoder

    output_splits = dict(decoder.output_splits)
    n_harmonics = output_splits.get('harmonic_distribution', 60)
    n_noise_bands = output_splits.get('noise_magnitudes', 65)

    print(f"  sample_rate={sample_rate}, hop_size={hop_size}, "
          f"n_frames={n_frames}, n_samples={n_samples}")
    print(f"  n_harmonics={n_harmonics}, n_noise_bands={n_noise_bands}")

    # ---- Step 2: Wrap decoder and trace with dummy data ----
    print("  Wrapping decoder for export...")
    wrapper = DecoderWrapper(decoder)

    dummy_ld = tf.random.normal([1, n_frames, 1])
    dummy_f0 = tf.random.normal([1, n_frames, 1])
    test_out = wrapper(dummy_ld, dummy_f0)
    print(f"  Forward pass OK — "
          f"amps: {test_out['amps'].shape}, "
          f"harmonic_distribution: {test_out['harmonic_distribution'].shape}, "
          f"noise_magnitudes: {test_out['noise_magnitudes'].shape}")

    # ---- Step 3: Save as SavedModel ----
    saved_model_dir = os.path.join(output_dir, f'_tmp_savedmodel_{name}')
    os.makedirs(saved_model_dir, exist_ok=True)
    print(f"  Saving TF SavedModel to {saved_model_dir}...")
    tf.saved_model.save(wrapper, saved_model_dir)

    # ---- Step 4: Convert SavedModel → ONNX via tf2onnx ----
    abs_out = os.path.join(output_dir, 'ddsp')
    os.makedirs(abs_out, exist_ok=True)
    onnx_path = os.path.join(abs_out, f'{name}.onnx')

    print(f"  Converting to ONNX at {onnx_path}...")
    cmd = [
        sys.executable, '-m', 'tf2onnx.convert',
        '--saved-model', saved_model_dir,
        '--output', onnx_path,
        '--opset', '18',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ tf2onnx conversion failed:\n{result.stderr}")
        return None
    print("  tf2onnx conversion done.")

    # ---- Step 5: Verify with ONNX Runtime ----
    _verify(onnx_path, n_frames)

    # ---- Step 6: Write metadata JSON ----
    frame_rate = sample_rate // hop_size
    metadata = {
        'model_name': name,
        'sample_rate': int(sample_rate),
        'n_harmonics': int(n_harmonics),
        'n_noise_bands': int(n_noise_bands),
        'hop_size': int(hop_size),
        'frame_rate': int(frame_rate),
        'n_frames_default': int(n_frames),
        'n_samples_default': int(n_samples),
        'f0_range': 127.0,
        'db_range': 80.0,
    }
    meta_path = os.path.join(abs_out, f'{name}_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata saved to {meta_path}")

    # ---- Step 7: Clean up temp SavedModel ----
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    print(f"  Cleaned up temp SavedModel: {saved_model_dir}")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ Exported: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Export DDSP decoder to ONNX for browser-based timbre transfer'
    )
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Path to DDSP checkpoint directory '
             '(e.g. ./trained_models/solo_violin)',
    )
    parser.add_argument(
        '--name', type=str, default=None,
        help='Model name for output files '
             '(default: inferred from model_dir basename)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='onnx_models',
        help='Root output directory (a ddsp/ sub-folder is created)',
    )
    parser.add_argument(
        '--length_seconds', type=int, default=4,
        help='Audio length in seconds for model building (default: 4)',
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        sys.exit(1)

    name = args.name or os.path.basename(model_dir.rstrip('/'))
    abs_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)

    exported = export_ddsp(model_dir, name, abs_out, args.length_seconds)

    print(f"\n{'=' * 60}")
    if exported:
        print(f"Export complete! Model saved to:")
        print(f"  • {exported}")
    else:
        print("Export failed.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
