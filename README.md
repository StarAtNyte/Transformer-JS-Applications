# TFJS Apps

A collection of browser-based ML demos using [Transformers.js](https://huggingface.co/docs/transformers.js). Models run locally in the browser via Web Workers (WebGPU when available, WASM fallback). No server needed.

## DINOv3

Uses [onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX).

- **Feature Explorer** (`Dinov3/index.html`) — Upload an image, hover over patches to see a similarity heatmap across the image.
- **Cross-Image Correspondence** (`Dinov3/correspondence.html`) — Upload two images, hover on one to find the closest matching patch on the other.

## Depth Anything v2

Uses [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small).

- **Depth Estimation** (`DepthAnything/index.html`) — Upload an image, get a colorized monocular depth map with Inferno, Viridis, or grayscale colormaps. Features side-by-side display, downloadable results, and full-size viewer with zoom.

## Object Detection (DETR)

Uses [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) for real-time object detection.

- **Object Detection** (`ObjectDetection/index.html`) — Upload an image to detect objects with bounding boxes, confidence scores, and labels. Features:
  - Color-coded bounding boxes
  - Adjustable confidence threshold slider
  - Detection results list with scores
  - Download annotated images
  - Full-size viewer with zoom
  - Sample images for testing

## Running

Serve the folder locally:

```bash
npx serve .
```

Or just open `index.html` directly in a browser.

Deployed via GitHub Pages.
