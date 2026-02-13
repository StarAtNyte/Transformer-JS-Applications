# TFJS Apps

A collection of browser-based ML demos using [Transformers.js](https://huggingface.co/docs/transformers.js). Models run locally in the browser via Web Workers (WebGPU when available, WASM fallback). No server needed.

## DINOv3

Uses [onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX).

- **Feature Explorer** (`Dinov3/index.html`) — Upload an image, hover over patches to see a similarity heatmap across the image.
- **Cross-Image Correspondence** (`Dinov3/correspondence.html`) — Upload two images, hover on one to find the closest matching patch on the other.

## Running

Serve the folder locally:

```bash
npx serve .
```

Or just open `index.html` directly in a browser.

Deployed via GitHub Pages.
