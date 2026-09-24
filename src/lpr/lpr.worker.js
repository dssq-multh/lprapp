// Polyfills for DOM globals required by dependencies (OpenCV.js, PaddleOCR) in a Web Worker environment
if (typeof document === 'undefined') {
  globalThis.document = {
    createElement(tagName) {
      if (tagName === 'canvas') {
        return new OffscreenCanvas(300, 150);
      }
      throw new Error(`document.createElement(${tagName}) is not supported in Worker`);
    },
    getElementById() {
      return null;
    }
  };
}

if (typeof window === 'undefined') {
  globalThis.window = globalThis;
}

if (typeof HTMLCanvasElement === 'undefined') {
  globalThis.HTMLCanvasElement = typeof OffscreenCanvas !== 'undefined' ? OffscreenCanvas : class HTMLCanvasElement {};
}

if (typeof HTMLImageElement === 'undefined') {
  globalThis.HTMLImageElement = class HTMLImageElement {};
}

if (typeof HTMLVideoElement === 'undefined') {
  globalThis.HTMLVideoElement = class HTMLVideoElement {};
}

if (typeof Image === 'undefined') {
  globalThis.Image = class Image {};
}

import { LprEngine } from './engine.js';

let engine = null;

self.onmessage = async (event) => {
  const { type, id, payload } = event.data || {};

  try {
    switch (type) {
      case 'init': {
        const { base, scoreThreshold, enableGpu } = payload || {};
        engine = new LprEngine({
          base,
          scoreThreshold,
          enableGpu
        });

        await engine.init((status) => {
          self.postMessage({ type: 'status', message: status });
        });

        self.postMessage({ type: 'initDone', id, success: true, enableGpu: engine.enableGpu });
        break;
      }

      case 'setEnableGpu': {
        const { enableGpu } = payload || {};
        if (!engine) throw new Error('Engine not initialized');

        const result = await engine.setEnableGpu(enableGpu, (status) => {
          self.postMessage({ type: 'status', message: status });
        });

        self.postMessage({ type: 'setEnableGpuDone', id, result });
        break;
      }

      case 'readAll': {
        if (!engine) throw new Error('Engine not initialized');
        const { bitmap, imageData, width, height, opts } = payload || {};

        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        if (bitmap) {
          ctx.drawImage(bitmap, 0, 0);
          try { bitmap.close(); } catch (_) {}
        } else if (imageData) {
          ctx.putImageData(imageData, 0, 0);
        } else {
          throw new Error('No valid image data or bitmap received');
        }

        const t0 = performance.now();
        const detections = await engine.readAll(canvas, opts);
        const duration = Math.round(performance.now() - t0);

        self.postMessage({
          type: 'readAllDone',
          id,
          detections,
          duration
        });
        break;
      }

      case 'paddlePredict': {
        if (!engine || !engine.paddleOcr) throw new Error('PaddleOCR engine not initialized');
        const { bitmap, imageData, width, height } = payload || {};

        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        if (bitmap) {
          ctx.drawImage(bitmap, 0, 0);
          try { bitmap.close(); } catch (_) {}
        } else if (imageData) {
          ctx.putImageData(imageData, 0, 0);
        } else {
          throw new Error('No valid image data or bitmap received');
        }

        const t0 = performance.now();
        const results = await engine.paddleOcr.predict(canvas);
        const duration = Math.round(performance.now() - t0);

        self.postMessage({
          type: 'paddlePredictDone',
          id,
          results,
          duration
        });
        break;
      }

      case 'rawDetect': {
        if (!engine) throw new Error('Engine not initialized');
        const { bitmap, imageData, width, height, thresh } = payload || {};
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (bitmap) {
          ctx.drawImage(bitmap, 0, 0);
          try { bitmap.close(); } catch (_) {}
        } else if (imageData) {
          ctx.putImageData(imageData, 0, 0);
        }
        const dets = await engine.detectIn(canvas, 0, 0, thresh || 0.05);
        self.postMessage({ type: 'rawDetectDone', id, detections: dets });
        break;
      }

      default:
        console.warn('Unknown message type in LPR worker:', type);
    }
  } catch (err) {
    console.error('Worker error processing message:', type, err);
    self.postMessage({
      type: `${type}Done`,
      id,
      error: err.message || String(err),
      detections: []
    });
  }
};
