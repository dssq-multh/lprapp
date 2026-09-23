# ALPR Sentinel — In-Browser License Plate Recognition Web App

A high-performance web app for real-time vehicle license plate recognition running entirely client-side using WebAssembly (Wasm).

## Features

- **Full-Height Plate Manager**: Starts with a `<textarea>` scaled to fit the height of the page where you can paste any number of vehicle license plates (one per line, comma, or space separated).
- **"Recognize!" Button**: Positioned right next to the textarea to start/stop live camera streaming and plate detection.
- **Forward-Facing Camera Preference**: Automatically prefers the forward-facing / environment camera (`facingMode: { ideal: "environment" }`) for road/vehicle scanning, with an interactive camera dropdown selector.
- **Wasm License Plate Recognition Engine**: Uses ONNX Runtime Web + OpenIPC LPR Wasm models (YOLOv5n-OBB spotter and TPS-STN attention recognizer) to detect and read plate text in real time.
- **1080px Resolution Fitting**: Every incoming picture/frame is proportionally scaled so that width and height do not exceed **1080px** (`Math.max(w, h) <= 1080`).
- **Dynamic Bounding Boxes with Ticks & Crosses**:
  - **Green Bounding Box with a Tick (✓)**: Displayed when an identified license plate is in the textarea list.
  - **Red Bounding Box with a Cross (✗)**: Displayed when an identified license plate is **not** in the list.
- **Interactive Live Feed & Audio Alert**: Includes a detection history stream, match counter, and audio chime on match.
- **Sample Vehicle Testing**: Includes built-in test vehicles (`CAL8942` and `B391KLT`) and an image upload option for testing without an outdoor vehicle.

## Getting Started

### 1. Install dependencies
```bash
npm install
```

### 2. Start the dev server
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 3. Build & Preview for Production
```bash
npm run build
npm run preview
```

## How It Works

1. **Plate Parsing**: When license plates are typed or pasted into the textarea, they are normalized (alphanumeric uppercase) and stored in a reactive lookup Set.
2. **Streaming & Scaling**: When **"Recognize!"** is clicked, the video stream from the preferred forward-facing camera is initiated. Each frame is downscaled to max 1080px in width or height on an offscreen canvas.
3. **Wasm Inference**: The spotter locates license plate quadrilateral bounding boxes. The crop is passed to the attention OCR recognizer.
4. **Overlay Canvas**: Coordinates are transformed back to screen space. Boxes are color-coded in green (with a ✓) or red (with a ✗).
