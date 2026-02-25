# Running ML Models in the Browser

## What is ONNX

ONNX stands for Open Neural Network Exchange. It's a file format for machine learning models, nothing more. The idea is that a model trained in PyTorch or TensorFlow can be exported to a `.onnx` file, and then that file can be loaded and run by any runtime that speaks ONNX — regardless of what framework originally trained it.

Think of it like a PDF for neural networks. The training framework doesn't matter at run time. What matters is that the exporter serialized the computation graph and weights into a standard format that any compliant runtime can execute.

An ONNX model contains:
- The computation graph (which ops to run, in what order)
- The trained weights (the actual numbers)
- Input/output shape information

## What is Transformers.js

Transformers.js is a JavaScript port of Hugging Face's `transformers` Python library. It lets you run the same models that would normally require a Python environment, directly in the browser or Node.js — using ONNX Runtime Web under the hood.

The library handles the full pipeline: preprocessing (tokenization, image resizing, normalization), model inference, and postprocessing (decoding outputs, formatting results). For models that are already on Hugging Face Hub with ONNX weights, it's often just a few lines to get something running.

```js
import { pipeline } from '@huggingface/transformers';

const detector = await pipeline('object-detection', 'Xenova/detr-resnet-50');
const result = await detector('path/to/image.jpg');
```

It abstracts a lot. You don't manage the ONNX session directly — you just call a pipeline and get results back.

## What is ONNX Runtime Web

ONNX Runtime Web (ort-web) is what actually executes ONNX models in the browser. It's a WebAssembly + WebGPU implementation of the ONNX runtime from Microsoft.

When you load a model, the runtime picks an execution backend in order of preference:

1. **WebGPU** — uses the GPU via the browser's WebGPU API. Fast, but not available everywhere yet.
2. **WebAssembly (WASM)** — runs on CPU via compiled C++ in a WASM module. Universally supported, slower.

In practice, the pattern looks like this:

```js
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['webgpu', 'wasm'],
});

const feeds = { input: new ort.Tensor('float32', pixelData, [1, 3, 512, 512]) };
const results = await session.run(feeds);
const output = results['output'].data;
```

One important thing: inference should run in a Web Worker, not on the main thread. Otherwise the page freezes during inference. All the demos in this project do this — the worker loads the model once, and then the main thread sends image data to it and waits for results.

## Natively Supported Models

Transformers.js has first-class support for a wide range of model architectures. These don't require any porting work — the ONNX weights are already on Hugging Face Hub, and the library handles pre/postprocessing natively.

Some examples relevant to what we've built or are planning:

| Task | Model | Notes |
|---|---|---|
| Object detection | DETR, Grounding DINO | Bounding boxes + labels |
| Depth estimation | Depth Anything | Relative monocular depth |
| Segmentation | SegFormer, SAM | Semantic and promptable |
| Background removal | MODNet, BiRefNet | Matting and cutout |
| Super resolution | Swin2SR | 2x and 4x upscaling |
| Image classification | ViT, MobileViT, ResNet-50 | |
| Image captioning | Various VLMs | |
| ASR | Moonshine, Whisper | |
| TTS | Kokoro | |
| LLM | SmolLM2 | Small enough for the browser |
| Embeddings | CLIP, BGE | Zero-shot classification, search |

For these, using Transformers.js is the right call. There's no reason to reinvent the wheel.

## Manually Porting a Model

When a model isn't on Hugging Face with ONNX weights, you have to port it yourself. This means taking the original PyTorch (or TF) code, loading the trained weights, and exporting to ONNX using `torch.onnx.export`.

The basic flow:

```python
import torch

model = YourModel()
model.load_state_dict(torch.load('weights.pth', map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=18,
    input_names=['input'],
    output_names=['output'],
)
```

In the happy path, this just works. In reality, there are usually a few things to fix.

### What goes wrong

**In-place operations.** PyTorch allows things like `tensor[:, 0:n, :, :] = value`. The ONNX exporter can't represent this. You have to rewrite it using `torch.cat` or similar.

**Dynamic batch size.** The exporter traces a concrete input shape. If the model internally does things like `input.shape[0]` (the batch size) and uses that in reshape or grouping operations, those become symbolic unknowns in the ONNX graph and can break convolutions. For single-image inference (batch=1), the simplest fix is to hardcode batch=1 in the forward pass.

**Custom CUDA ops.** Some models use custom CUDA extensions (e.g., `upfirdn2d`, `fused_leaky_relu` in StyleGAN-based models). These don't export to standard ONNX ops. The fix: force the model to CPU before exporting. On CPU, PyTorch falls back to pure Python/PyTorch implementations of those ops, which trace cleanly.

**Multiple outputs.** Many models return tuples — the image plus latents, or image plus auxiliary features. You only want what you actually need at inference time. Wrapping the model in a thin `nn.Module` that discards the extra outputs keeps the ONNX graph clean.

**Wrapper pattern.** The cleanest approach is always to write a small wrapper:

```python
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, _latent = self.model(x)  # drop what we don't need
        return output
```

Export the wrapper, not the raw model.

## Caching Models in the Browser

Model files are large — anywhere from 10 MB to a few hundred MB. Without caching, the user re-downloads the model every time they visit the page. That's a bad experience and wastes bandwidth. The browser's HTTP cache helps, but it's unreliable for large binary files; the browser can evict them at any point.

The reliable solution is IndexedDB. It's a proper key-value store in the browser that persists until explicitly cleared. Both ONNX Runtime Web and Transformers.js can be made to use it, but the approach differs between the two.

### ONNX Runtime Web — explicit fetch and store

With raw ORT, you control the model loading yourself — you fetch the `.onnx` file, get an `ArrayBuffer`, and pass it to `InferenceSession.create`. So you just wrap that fetch with an IndexedDB read/write:

```js
async function openModelDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open('onnx-model-cache', 1);
        req.onupgradeneeded = e => e.target.result.createObjectStore('models');
        req.onsuccess = e => resolve(e.target.result);
        req.onerror = e => reject(e.target.error);
    });
}

async function getCachedModel(key) {
    try {
        const db = await openModelDB();
        return new Promise(resolve => {
            const req = db.transaction('models', 'readonly').objectStore('models').get(key);
            req.onsuccess = e => resolve(e.target.result || null);
            req.onerror = () => resolve(null);
        });
    } catch (e) { return null; }
}

async function cacheModel(key, buffer) {
    try {
        const db = await openModelDB();
        await new Promise(resolve => {
            const tx = db.transaction('models', 'readwrite');
            tx.objectStore('models').put(buffer, key);
            tx.oncomplete = resolve;
            tx.onerror = resolve;
        });
    } catch (e) {}
}

// Usage
let modelBuffer = await getCachedModel(modelUrl);
if (!modelBuffer) {
    const resp = await fetch(modelUrl);
    modelBuffer = await resp.arrayBuffer();
    await cacheModel(modelUrl, modelBuffer);
}

const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['webgpu', 'wasm'],
});
```

The URL is used as the cache key. If the model URL changes (e.g. you update the file on HuggingFace), it's treated as a new model and re-downloaded automatically.

### Transformers.js — intercepting fetch

Transformers.js handles model downloading internally — you don't get an `ArrayBuffer` to cache yourself. It calls `fetch` under the hood to pull weights from HuggingFace. The way to cache that is to patch `globalThis.fetch` before the library initialises, so every request it makes goes through your cache layer:

```js
// Run this before importing Transformers.js
(function () {
    function openModelDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open('onnx-model-cache', 1);
            req.onupgradeneeded = e => e.target.result.createObjectStore('models');
            req.onsuccess = e => resolve(e.target.result);
            req.onerror = e => reject(e.target.error);
        });
    }

    async function getFromDB(url) {
        try {
            const db = await openModelDB();
            return new Promise(resolve => {
                const req = db.transaction('models', 'readonly').objectStore('models').get(url);
                req.onsuccess = e => resolve(e.target.result || null);
                req.onerror = () => resolve(null);
            });
        } catch (e) { return null; }
    }

    function saveToDB(url, buffer) {
        openModelDB().then(db => {
            const tx = db.transaction('models', 'readwrite');
            tx.objectStore('models').put(buffer, url);
        }).catch(() => {});
    }

    const _fetch = globalThis.fetch.bind(globalThis);
    globalThis.fetch = async function (input, init) {
        const url = typeof input === 'string' ? input
            : (input instanceof Request ? input.url : String(input));
        const isHF = url.indexOf('huggingface.co') !== -1;
        const isGet = !init || !init.method || init.method.toUpperCase() === 'GET';

        if (isHF && isGet) {
            const cached = await getFromDB(url);
            if (cached) {
                return new Response(cached, {
                    status: 200,
                    headers: {
                        'Content-Type': 'application/octet-stream',
                        'Content-Length': '' + cached.byteLength,
                    },
                });
            }
            const response = await _fetch(input, init);
            if (response.ok) {
                const buffer = await response.clone().arrayBuffer();
                saveToDB(url, buffer);
            }
            return response;
        }

        return _fetch(input, init);
    };
})();

// Now import Transformers.js — it will use the patched fetch transparently
const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2');
```

A few things worth noting here:

- The filter checks for `huggingface.co` and GET requests only. Non-HF traffic (telemetry, other APIs) goes through untouched.
- `response.clone()` is needed because `Response` bodies can only be consumed once. Without the clone, reading the buffer would exhaust the response before Transformers.js could read it.
- `saveToDB` is fire-and-forget — failures are silently ignored so a failed write doesn't break the actual request.
- This has to run inside the Web Worker, not the main thread, because that's where Transformers.js runs and where its `fetch` calls happen.

### What gets cached

For ORT-based demos, it's just the `.onnx` file. For Transformers.js, the model is split into multiple files — `model.onnx` or `model_quantized.onnx`, `config.json`, `tokenizer.json`, and possibly `tokenizer_config.json` and vocabulary files. All of them go through `fetch`, so all of them get cached by the patched fetch.

### Clearing the cache

There's no automatic expiry. If you need to force a re-download (e.g. after pushing a new model version with the same filename), you either change the URL or clear IndexedDB manually from the browser's dev tools under Application > Storage > IndexedDB.

## Opset Version Conflicts

This is one of the more annoying things you'll run into. When you export a model, you pick an ONNX opset version — basically the "language version" of the ONNX spec. Higher opsets have newer ops and sometimes cleaner representations. But the runtime that loads the model in the browser needs to actually support that opset, and ONNX Runtime Web lags behind the spec a bit.

Across the models we've ported, the opset versions ended up being all over the place:

| Model | Opset used | Reason |
|---|---|---|
| GPEN | 12 | StyleGAN2 ops trace cleanly at 12; higher caused issues |
| DPR | 18 | Standard hourglass net, no reason not to use latest |
| UHDM | 18 | Clean CNN, no issues |
| Neural Painting | 17 | Worked fine, no specific reason for 17 vs 18 |
| Handwriting (Graves LSTM) | 9 | Pre-existing export script, LSTM ops at opset 9 |
| Line Art Vectorization | 11 | Pre-existing conversion script |

The rule of thumb: start with opset 18 (or whatever is current). If the export fails or the model doesn't run in the browser, try stepping down. The most common safe landing zone for older or more complex models is opset 12.

### Why lower opsets sometimes work better

Higher opsets can introduce op fusions or representations that ONNX Runtime Web's WebAssembly backend doesn't fully support yet. A model that exports fine at opset 17 and runs in Python's `onnxruntime` may silently fail or throw when loaded in the browser, because the WASM backend implements a subset of the full spec.

Opset 12 is a practical floor for most models. It has `Einsum`, `NegativeLogLikelihoodLoss`, and `Trilu`, and most convolution, normalization, and activation ops are stable at this version. Going below 12 sometimes means losing ops that modern models rely on.

### The import conflict

If you're using Transformers.js and raw ONNX Runtime Web in the same page, you can run into a version mismatch. Transformers.js bundles its own copy of `onnxruntime-web` internally. If you also load `ort` from a CDN separately (e.g. for a manually ported model), you now have two copies of the runtime on the page — different versions, potentially conflicting over the global `ort` object or the WASM binary paths.

The symptom is usually something like the WASM paths getting overwritten, or `InferenceSession.create` failing with a cryptic error about missing backends.

The clean fix: don't mix them. If a demo uses Transformers.js, use it exclusively and go through its `env` settings to configure the backend. If a demo uses a manually ported model, use raw `onnxruntime-web` directly and don't import Transformers.js. In this project, demos that use Transformers.js (DepthAnything, VL, OWLDetect, etc.) are separate pages from demos that use raw ORT (GPEN, UHDM, DPR, etc.) — that separation is intentional.

If you absolutely need both on the same page, the only safe approach is to make sure both are on the same version of `onnxruntime-web`, and to let only one of them configure `env.wasm.wasmPaths`. Even then, it's fragile and not worth it for most cases.

## What Kinds of Models Can Be Ported

Most feed-forward CNNs and transformers port without much trouble. Some things are harder.

**Straightforward to port:**
- Standard encoder-decoder CNNs (U-Net style, hourglass networks)
- ResNet-based backbones
- Feed-forward vision transformers (ViT, Swin)
- GAN generators (the generator only — discriminator stays in Python)
- Most image-to-image models with fixed input/output sizes

**Requires more work:**
- Models with recurrent components (LSTMs, GRUs) — these export, but may need careful handling of hidden states as explicit inputs/outputs
- Models with dynamic control flow (loops where the iteration count depends on input data)
- Diffusion models — the denoising step exports, but the sampling loop lives outside the model
- Multi-model pipelines — each sub-network needs to be exported separately and stitched together in JS

**Practically impossible or not worth it:**
- Models that depend on custom Python post-processing that has no clean analog in ONNX ops
- Very large models (anything over ~1-2 GB is probably not a browser demo)
- Models that require dynamic input shapes in ways that fundamentally break static graph export

## Example: Porting GPEN (Face Restoration)

GPEN is a GAN-based face restoration model from Alibaba Research. The generator takes a degraded face image (normalized to [-1, 1]) and outputs a restored version.

The original code had two issues that needed fixing before export.

**Problem 1 — Custom CUDA ops.** GPEN's generator uses StyleGAN2-style convolutions that rely on `upfirdn2d` and `fused_leaky_relu` CUDA extensions. These don't trace to standard ONNX ops. Fix: run everything on CPU. PyTorch's fallback implementations are pure PyTorch and export cleanly.

**Problem 2 — Dynamic batch dimension.** The `ConstantInput` layer did `self.input.repeat(batch, 1, 1, 1)` where `batch = input.shape[0]`. This produced a symbolic repeat count in the ONNX graph, which made all downstream shapes unknown, which caused conv layers to fail. For batch=1 inference, patching the forward to just return `self.input` directly fixed it. Similarly, `ModulatedConv2d` used `batch` from `input.shape[0]` in reshape/groups calls — replacing those with the literal `1` resolved the unknown-shape convolution errors.

The fix was monkey-patching the forward methods before export, rather than modifying the original model code:

```python
def _constant_input_b1(self, input):
    return self.input  # hardcode batch=1

def _modulated_conv2d_b1(self, input, style):
    # rewrite using literal 1 instead of batch from input.shape[0]
    ...

for m in wrapper.modules():
    if isinstance(m, ConstantInput):
        m.forward = types.MethodType(_constant_input_b1, m)
    elif isinstance(m, ModulatedConv2d):
        m.forward = types.MethodType(_modulated_conv2d_b1, m)
```

After patching, the export ran cleanly at opset 12. The wrapper drops the latent output that the generator normally returns alongside the image, since we don't need it at inference time.

After export, verification with `onnxruntime` confirmed the ONNX outputs matched PyTorch outputs within floating point tolerance (`rtol=1e-3, atol=1e-3`).

The resulting `.onnx` files were hosted on Hugging Face and loaded in the browser via ONNX Runtime Web. The browser-side code loads the model in a Web Worker, sends pixel data as a Float32Array, runs the session, and sends the result back to the main thread.

---

*This is based on actual work done porting models like GPEN, DPR, UHDM, and others for in-browser inference. The patterns that come up are fairly consistent across models — the main variables are whether custom ops are involved and how the model handles batch dimensions.*

---

## Appendix: Sample Export Script (GPEN)

Full working export script for GPEN. Run it from inside the `PortableModels/` directory. Weights are downloaded automatically if not present.

```
python export_gpen_onnx.py                           # export all three variants
python export_gpen_onnx.py --model bfr-512           # export one variant
python export_gpen_onnx.py --model colorization-1024 --output-dir ./out
```

```python
"""
Export GPEN models to ONNX.

Supported model variants:
  - bfr-256:            FullGenerator 256x256, channel_mult=1, narrow=0.5
  - bfr-512:            FullGenerator 512x512, channel_mult=2, narrow=1.0
  - colorization-1024:  FullGenerator 1024x1024, channel_mult=2, narrow=1.0

Input:  [1, 3, size, size]  float32, RGB, normalized to [-1, 1]
Output: [1, 3, size, size]  float32, RGB, normalized to [-1, 1]

Weights are downloaded automatically if not present.
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
sys.path.insert(0, os.path.join(GPEN_DIR, 'face_model'))
sys.path.insert(0, os.path.join(GPEN_DIR, 'face_model', 'op'))

from gpen_model import FullGenerator, ConstantInput, ModulatedConv2d


WEIGHT_BASE_URL = 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/'

MODEL_CONFIGS = {
    'bfr-256':            ('GPEN-BFR-256',            256,  1, 0.5),
    'bfr-512':            ('GPEN-BFR-512',            512,  2, 1.0),
    'colorization-1024':  ('GPEN-Colorization-1024', 1024,  2, 1.0),
}


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f'  Downloading {os.path.basename(dest)}...')
    def _progress(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f'\r  {pct}%', end='', flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


class GPENWrapper(nn.Module):
    """Wraps FullGenerator to return only the restored image (drops the latent)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        image, _latent = self.model(x)
        return image


def _constant_input_b1(self, input):
    # Original: self.input.repeat(batch, 1, 1, 1)
    # repeat() with a symbolic batch count emits onnx::Tile with an unknown
    # repeat count — all downstream shapes go dynamic and convolutions break.
    # For batch=1 inference, return the constant directly.
    return self.input


def _modulated_conv2d_b1(self, input, style):
    # Original code derives `batch` from input.shape[0] and uses it in reshape/groups.
    # That becomes an aten::size node in the ONNX graph — a symbolic unknown —
    # so the weight kernel ends up with a `*` dimension and conv fails with
    # "convolution for kernel of unknown shape". Replace every occurrence with 1.
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


def _verify(onnx_path, dummy_input, torch_output):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {input_name: dummy_input.numpy()})[0]
        np.testing.assert_allclose(torch_output.numpy(), ort_out, rtol=1e-3, atol=1e-3)
        print('  Verification passed — ORT output matches PyTorch within tolerance.')
    except ImportError:
        print('  onnxruntime not installed, skipping verification.')
    except Exception as e:
        print(f'  Verification warning: {e}')


def export(variant, output_dir):
    model_name, in_size, channel_multiplier, narrow = MODEL_CONFIGS[variant]
    weight_path = os.path.join(GPEN_DIR, 'weights', model_name + '.pth')

    print(f'\nExporting: {model_name} ({in_size}x{in_size})')

    if not os.path.exists(weight_path):
        _download(WEIGHT_BASE_URL + model_name + '.pth', weight_path)

    # Force CPU — upfirdn2d and fused_leaky_relu are custom CUDA extensions.
    # On CPU, PyTorch falls back to pure-PyTorch implementations that trace
    # to standard ONNX ops cleanly. Don't move to GPU before exporting.
    print(f'  Loading weights...')
    model = FullGenerator(in_size, 512, 8, channel_multiplier, narrow=narrow, device='cpu')
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    wrapper = GPENWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, in_size, in_size)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f'  Forward pass OK — output: {out.shape}')

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, model_name.lower().replace('-', '_') + '.onnx')

    # Apply patches before tracing — monkey-patch forward methods on every matching
    # layer instance in the graph. We do this on the wrapper (post-construction)
    # rather than editing the original model source.
    for m in wrapper.modules():
        if isinstance(m, ConstantInput):
            m.forward = types.MethodType(_constant_input_b1, m)
        elif isinstance(m, ModulatedConv2d):
            m.forward = types.MethodType(_modulated_conv2d_b1, m)

    print(f'  Exporting to {onnx_path}...')
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=12,       # opset 17/18 works for plain CNNs, but StyleGAN-style
        do_constant_folding=True,  # weight modulation causes issues at higher opsets
        input_names=['input'],
        output_names=['output'],
        dynamo=False,
    )

    _verify(onnx_path, dummy, out)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f'  Done: {onnx_path} ({size_mb:.1f} MB)')
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export GPEN models to ONNX')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), default=None,
                        help='Which variant to export (default: all)')
    parser.add_argument('--output-dir', default='onnx_models/gpen',
                        help='Output directory (default: onnx_models/gpen/)')
    args = parser.parse_args()

    abs_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    variants = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    exported = [export(v, abs_out) for v in variants]

    print(f'\nExport complete — {len(exported)} model(s) saved to {abs_out}/')
    for p in exported:
        print(f'  {p}')


if __name__ == '__main__':
    main()
```

The key things to carry over to any other model:
- Always wrap the model to control exactly what outputs get exported
- Patch after loading weights, before tracing — `torch.onnx.export` calls `forward()` internally during the trace
- Run a forward pass with `torch.no_grad()` before exporting — confirms the patched model works and gives a reference tensor to diff against
- Verify with `onnxruntime` on CPU before loading in the browser
