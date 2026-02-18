# TFJS Apps

ML demos that run entirely in the browser. No server, no uploads — models execute locally via Web Workers using WebGPU where available, with a WASM fallback. Most use [Transformers.js](https://huggingface.co/docs/transformers.js); a few run ONNX models directly through [ONNX Runtime Web](https://onnxruntime.ai/).

## Demos

**Feature Explorer** — Hover over an image to see a patch-level similarity heatmap. Uses DINOv3 features.

**Cross-Image Correspondence** — Upload two images, hover on one to find the closest matching patch on the other.

**Depth Estimation** — Monocular depth from a single photo. Colorized output with Inferno, Viridis, or grayscale.

**Neural Line Art Vectorization** — Converts raster line art into clean vector SVG paths.

**Portrait Relighting** — Change the lighting direction on a portrait from a single photo.

**Colorify** — Colorize black-and-white photos and video frames.

**FaceParser** — Semantic face segmentation (hair, skin, eyes, etc.) powered by SegFormer.

**ObjectDetect** — Real-time object detection with bounding boxes, confidence scores, and labels.

**Vision Language** — Ask questions about images, powered by Transformers.js.

**FastVLM** — Vision-language model with WebGPU acceleration.

**Revive** — Face restoration and colorization for degraded, blurry, or grayscale inputs. Powered by GPEN.

## Running locally

Serve the folder with any static file server:

```
npx serve .
```

Then open `http://localhost:3000` in a browser. Also deployed via GitHub Pages.
