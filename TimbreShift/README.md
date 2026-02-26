# TimbreShift — Neural Timbre Transfer

Change instrument timbre from any audio input, running entirely in the browser via ONNX Runtime Web.

## How It Works

1. Upload audio or record from mic
2. Pitch (f0) and loudness are extracted in JavaScript (YIN algorithm)
3. An ONNX decoder model predicts synthesizer parameters for the target instrument
4. Audio is resynthesized using additive harmonic synthesis + filtered noise
5. The pitch and dynamics stay the same — only the timbre changes

## Exporting ONNX Models

### Prerequisites

```bash
pip install tensorflow gin-config scipy tf2onnx onnxruntime
```

> **Note:** `ddsp` is imported directly from the cloned repo at `PortableModels/ddsp/` — no `pip install ddsp` needed. The export script mocks unavailable deps like `crepe` and `librosa` that aren't required for decoder export.

### Download Pretrained Models

```bash
# Install gsutil if needed
pip install gsutil

# Download one or more instrument models
gsutil -m cp -r gs://ddsp/models/timbre_transfer/solo_violin ./trained_models/
gsutil -m cp -r gs://ddsp/models/timbre_transfer/solo_flute ./trained_models/
gsutil -m cp -r gs://ddsp/models/timbre_transfer/solo_trumpet ./trained_models/
```

### Export to ONNX

```bash
cd PortableModels

# Export a single model
python export_ddsp_onnx.py --model_dir ./trained_models/solo_violin --name ddsp_violin

# Export with custom output directory
python export_ddsp_onnx.py --model_dir ./trained_models/solo_flute --name ddsp_flute --output_dir onnx_models

# Export all three
python export_ddsp_onnx.py --model_dir ./trained_models/solo_violin --name ddsp_violin
python export_ddsp_onnx.py --model_dir ./trained_models/solo_flute --name ddsp_flute
python export_ddsp_onnx.py --model_dir ./trained_models/solo_trumpet --name ddsp_trumpet
```

Output files are saved to `PortableModels/onnx_models/ddsp/`:

```
onnx_models/ddsp/
├── ddsp_violin.onnx
├── ddsp_violin_metadata.json
├── ddsp_flute.onnx
├── ddsp_flute_metadata.json
├── ddsp_trumpet.onnx
└── ddsp_trumpet_metadata.json
```

### Run the Web UI

Serve the project root with any static file server:

```bash
cd /path/to/TFJS
python -m http.server 8000
# Open http://localhost:8000/TimbreShift/
```
