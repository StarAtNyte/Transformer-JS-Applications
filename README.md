# Browser ML Demos

Computer vision models running entirely in the browser — no server, no uploads, no cloud. Inference runs locally via Web Workers, using WebGPU where available with a WASM fallback. Built with [Transformers.js](https://huggingface.co/docs/transformers.js) and [ONNX Runtime Web](https://onnxruntime.ai/).

## Demos

**Feature Explorer** — Hover over an image to see a patch-level similarity heatmap computed from DINOv3 features.

**Cross-Image Correspondence** — Upload two images and hover on one to find the closest matching patch on the other.

**DepthLens** — Monocular depth estimation from a single photo, with Inferno, Viridis, and grayscale colormaps.

**Neural Line Art Vectorization** — Converts raster line art directly into clean, scalable vector SVG paths.

**Portrait Relighting** — Shift the lighting direction on a portrait photo without a 3D scan.

**Colorify** — Colorize black-and-white photos and video frames in real time.

**FaceParser** — Semantic face segmentation (hair, skin, eyes, lips, etc.) powered by SegFormer.

**ObjectDetect** — Real-time object detection with bounding boxes, confidence scores, and class labels.

**Vision Language** — Ask natural-language questions about any image.

**FastVLM** — Fast vision-language model with WebGPU acceleration.

**Revive** — Face restoration and colorization for degraded, blurry, or low-quality inputs, powered by GPEN.

**InkTrace** — Convert any photo into clean contour line art using the Informative Drawings model.

## Running locally

Serve the folder with any static file server:

```
npx serve .
```

Then open `http://localhost:3000` in a browser. Also deployed via GitHub Pages.
