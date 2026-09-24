import * as ort from 'onnxruntime-web';
import { PaddleOCR } from '@paddleocr/paddleocr-js';
import { letterbox, unletterboxQuad } from './preprocess.js';
import { decodeSpotter } from './decoder_spotter.js';

const SPOT_W = 416, SPOT_H = 256;
const TILE_W = 384, TILE_H = 240, OVERLAP = 0.25;

export const DEFAULT_DET_THRESH = 0.19;
const NMS_IOU = 0.30;

function boxOf(quad) {
  const xs = quad.map((p) => p[0]), ys = quad.map((p) => p[1]);
  const l = Math.min(...xs), t = Math.min(...ys);
  return { left: l, top: t, width: Math.max(...xs) - l, height: Math.max(...ys) - t };
}

function iou(a, b) {
  const x = Math.max(0, Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left));
  const y = Math.max(0, Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top));
  const i = x * y;
  return i / (a.width * a.height + b.width * b.height - i || 1);
}

function dedupe(dets) {
  const out = [];
  dets.sort((p, q) => q.score - p.score);
  for (const d of dets) {
    if (!out.some((k) => iou(d.box, k.box) > NMS_IOU)) {
      out.push(d);
    }
  }
  return out;
}

function tiles(w, h) {
  const sx = Math.round(TILE_W * (1 - OVERLAP)), sy = Math.round(TILE_H * (1 - OVERLAP));
  const out = [];
  for (let y = 0; y < Math.max(h - TILE_H, 0) + sy; y += sy) {
    for (let x = 0; x < Math.max(w - TILE_W, 0) + sx; x += sx) {
      out.push({
        x: Math.min(x, Math.max(w - TILE_W, 0)),
        y: Math.min(y, Math.max(h - TILE_H, 0)),
        w: Math.min(TILE_W, w),
        h: Math.min(TILE_H, h)
      });
    }
  }
  return out.length ? out : [{ x: 0, y: 0, w, h }];
}

function cut(img, t) {
  const c = new OffscreenCanvas(t.w, t.h);
  c.getContext('2d', { willReadFrequently: true })
    .drawImage(img, t.x, t.y, t.w, t.h, 0, 0, t.w, t.h);
  return c;
}

/**
 * Refines the raw bounding box to snap precisely to the bright plate surface.
 * Eliminates outer mounting brackets, dark bumper areas, and mounting bolts.
 */
function refinePlateBox(sourceCanvas, rawBox) {
  try {
    const marginX = Math.round(rawBox.width * 0.08);
    const marginY = Math.round(rawBox.height * 0.15);
    const cropX = Math.max(0, Math.floor(rawBox.left - marginX));
    const cropY = Math.max(0, Math.floor(rawBox.top - marginY));
    const cropW = Math.max(10, Math.min(sourceCanvas.width - cropX, Math.ceil(rawBox.width + marginX * 2)));
    const cropH = Math.max(10, Math.min(sourceCanvas.height - cropY, Math.ceil(rawBox.height + marginY * 2)));

    const c = new OffscreenCanvas(cropW, cropH);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(sourceCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

    const imgData = ctx.getImageData(0, 0, cropW, cropH);
    const data = imgData.data;

    const colBright = new Array(cropW).fill(0);
    const rowBright = new Array(cropH).fill(0);

    for (let y = 0; y < cropH; y++) {
      for (let x = 0; x < cropW; x++) {
        const idx = (y * cropW + x) * 4;
        const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        if (lum > 130) {
          colBright[x]++;
          rowBright[y]++;
        }
      }
    }

    let minX = 0, maxX = cropW - 1;
    while (minX < cropW && colBright[minX] < cropH * 0.30) minX++;
    while (maxX > minX && colBright[maxX] < cropH * 0.30) maxX--;

    let minY = 0, maxY = cropH - 1;
    while (minY < cropH && rowBright[minY] < cropW * 0.30) minY++;
    while (maxY > minY && rowBright[maxY] < cropW * 0.30) maxY--;

    if (maxX - minX >= rawBox.width * 0.5 && maxY - minY >= rawBox.height * 0.4) {
      const left = cropX + minX;
      const top = cropY + minY;
      const width = maxX - minX;
      const height = maxY - minY;
      return {
        left,
        top,
        width,
        height,
        quad: [
          [left, top],
          [left + width, top],
          [left + width, top + height],
          [left, top + height]
        ]
      };
    }
  } catch (err) {
    console.warn('refinePlateBox fallback:', err);
  }

  return rawBox;
}

/**
 * Prepares the plate crop for PaddleOCR reading.
 */
function preprocessPlateCrop(sourceCanvas, box, quad) {
  let aspect = box.width / Math.max(1, box.height);
  if (quad && quad.length === 4) {
    const edgeTop = Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]);
    const edgeSide = Math.hypot(quad[2][0] - quad[1][0], quad[2][1] - quad[1][1]);
    if (edgeSide > 0) {
      aspect = edgeTop / edgeSide;
    }
  }
  const isSquare = aspect < 2.0;
  const padX = isSquare ? 4 : Math.max(12, Math.round(box.width * 0.22));
  const padY = 4;
  const sx = Math.max(0, box.left - padX);
  const sy = Math.max(0, box.top - padY);
  const sw = Math.min(sourceCanvas.width - sx, box.width + padX * 2);
  const sh = Math.min(sourceCanvas.height - sy, box.height + padY * 2);

  // For 1-line rectangular plates, 48px height gives the single line ~40-48px.
  // For 2-line stacked plates, each line needs at least 48px plus plate margins, requiring 128px.
  const targetH = isSquare ? 128 : 48;
  const targetW = isSquare
    ? Math.max(128, Math.round(targetH * aspect))
    : Math.max(180, Math.round(targetH * aspect * (sw / Math.max(1, box.width))));

  const c = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(targetW, targetH)
    : document.createElement('canvas');
  c.width = targetW;
  c.height = targetH;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, targetW, targetH);

  return { canvas: c, sx, sy, sw, sh, targetW, targetH };
}

export class LprEngine {
  constructor(opts = {}) {
    this.base = opts.base || '/lpr/';
    this.spotter = null;
    this.paddleOcr = null;
    this.isReady = false;
    this.scoreThreshold = opts.scoreThreshold || DEFAULT_DET_THRESH;
    this.enableGpu = opts.enableGpu || false;
  }

  async createSpotterSession(wantGpu) {
    const spotterUrl = `${this.base}models/spotter_b_v12_yuv_pm.onnx`;

    if (wantGpu) {
      // First try disabled, then basic optimization to avoid Conv fusion layout bugs on WebGPU
      for (const optLevel of ['disabled', 'basic', 'all']) {
        try {
          console.log(`Validating YOLOv5n-OBB on WebGPU (graphOptimizationLevel: ${optLevel})...`);
          const session = await ort.InferenceSession.create(spotterUrl, {
            executionProviders: ['webgpu', 'wasm'],
            graphOptimizationLevel: optLevel,
          });
          // Perform a quick warmup inference to verify WebGPU kernels actually execute without JSEP runtime failures
          const dummyTensor = new ort.Tensor('float32', new Float32Array(1 * 3 * SPOT_H * SPOT_W), [1, 3, SPOT_H, SPOT_W]);
          await session.run({ [session.inputNames[0]]: dummyTensor });
          console.log(`WebGPU validation passed with graphOptimizationLevel: ${optLevel}`);
          return { session, enableGpu: true };
        } catch (err) {
          console.warn(`WebGPU validation failed with optLevel '${optLevel}':`, err);
        }
      }
      console.warn('All WebGPU initialization attempts failed. Reverting to WASM provider.');
    }

    const wasmSession = await ort.InferenceSession.create(spotterUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
    });
    console.log('YOLOv5n-OBB initialized on WASM');
    return { session: wasmSession, enableGpu: false };
  }

  async setEnableGpu(enabled, onStatus) {
    const nextGpu = Boolean(enabled);
    if (this.enableGpu === nextGpu && this.spotter) {
      return { success: true, enabled: this.enableGpu };
    }
    const providerName = nextGpu ? 'WebGPU' : 'WASM';
    if (onStatus) onStatus(`Configuring detector for ${providerName}...`);

    const oldSpotter = this.spotter;
    const { session, enableGpu } = await this.createSpotterSession(nextGpu);
    this.spotter = session;
    this.enableGpu = enableGpu;

    if (oldSpotter && typeof oldSpotter.release === 'function') {
      try { await oldSpotter.release(); } catch (_) {}
    }

    if (nextGpu && !enableGpu) {
      if (onStatus) onStatus('WebGPU unavailable; on WASM');
      return { success: false, enabled: false, reverted: true };
    }

    if (onStatus) onStatus(`YOLO running on ${this.enableGpu ? 'WebGPU' : 'WASM'}`);
    return { success: true, enabled: this.enableGpu };
  }

  async init(onStatus) {
    if (this.isReady) return;

    const baseUrl = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '') + '/';

    if (onStatus) onStatus('Configuring WebAssembly runtime...');
    ort.env.wasm.wasmPaths = `${baseUrl}ort-wasm/`;
    const isCrossOriginIsolated = typeof crossOriginIsolated !== 'undefined'
      ? crossOriginIsolated
      : (typeof self !== 'undefined' && Boolean(self.crossOriginIsolated));
    if (!isCrossOriginIsolated) {
      ort.env.wasm.numThreads = 1;
    }

    const providerName = this.enableGpu ? 'WebGPU' : 'WASM';
    if (onStatus) onStatus(`Loading license plate detector (YOLOv5n-OBB on ${providerName})...`);
    const { session, enableGpu } = await this.createSpotterSession(this.enableGpu);
    this.spotter = session;
    this.enableGpu = enableGpu;

    if (onStatus) onStatus('Initializing PaddleOCR Wasm engine...');
    try {
      this.paddleOcr = await PaddleOCR.create({
        textDetectionModelName: 'PP-OCRv6_tiny_det',
        textDetectionModelAsset: { url: `${baseUrl}models/PP-OCRv6_tiny_det_onnx_infer.tar` },
        textRecognitionModelName: 'PP-OCRv6_tiny_rec',
        textRecognitionModelAsset: { url: `${baseUrl}models/PP-OCRv6_tiny_rec_onnx_infer.tar` },
        ortOptions: {
          backend: 'wasm',
          wasmPaths: `${baseUrl}ort-wasm/`
        }
      });
      console.log('PaddleOCR Wasm engine ready!');
    } catch (err) {
      console.error('PaddleOCR initialization error:', err);
    }

    this.isReady = true;
    if (onStatus) onStatus('Wasm ALPR engine ready!');
  }

  async detectIn(canvas, ox, oy, thresh) {
    const p = letterbox(canvas, SPOT_W, SPOT_H, { norm: 'yuv' });
    let out;
    try {
      out = await this.spotter.run({
        [this.spotter.inputNames[0]]: new ort.Tensor('float32', p.tensor, [1, 3, SPOT_H, SPOT_W]),
      });
    } catch (err) {
      if (this.enableGpu) {
        console.warn('YOLO spotter run failed on WebGPU at runtime, falling back to WASM:', err);
        const { session, enableGpu } = await this.createSpotterSession(false);
        this.spotter = session;
        this.enableGpu = enableGpu;
        out = await this.spotter.run({
          [this.spotter.inputNames[0]]: new ort.Tensor('float32', p.tensor, [1, 3, SPOT_H, SPOT_W]),
        });
      } else {
        throw err;
      }
    }
    const [p3, p4] = this.spotter.outputNames;
    const rawDets = decodeSpotter(out, p3, p4, thresh, NMS_IOU);

    return rawDets
      .map((d) => {
        const q = unletterboxQuad(d.quad, p.scale, p.padX, p.padY)
          .map((pt) => [pt[0] + ox, pt[1] + oy]);
        const box = boxOf(q);
        const aspect = box.width / Math.max(1, box.height);
        return { quad: q, box, score: d.score, aspect };
      })
      .filter((d) => d.score >= thresh && d.aspect >= 1.2 && d.box.width >= 24 && d.box.height >= 10);
  }

  async detect(image, opts = {}) {
    const thresh = opts.scoreThreshold || this.scoreThreshold;
    const w = image.width, h = image.height;
    // Always single-pass full frame (tiling disabled always for now)
    const ts = [{ x: 0, y: 0, w, h }];
    /*
    if (opts.tile !== false && (w > SPOT_W * 1.5 || h > SPOT_H * 1.5)) {
      ts.push(...tiles(w, h));
    }
    */

    const all = [];
    for (let i = 0; i < ts.length; i++) {
      const t = ts[i];
      const c = (t.w === w && t.h === h && t.x === 0 && t.y === 0) ? image : cut(image, t);
      all.push(...(await this.detectIn(c, t.x, t.y, thresh)));
      if (opts.onProgress) opts.onProgress(i + 1, ts.length);
    }
    return dedupe(all);
  }

  async readWithOcr(image, box, detScore, quad) {
    if (!this.paddleOcr) {
      return { text: '', minConf: 0, confident: false, refinedQuad: null };
    }

    try {
      const crop = preprocessPlateCrop(image, box, quad);
      const input = (typeof createImageBitmap === 'function') ? await createImageBitmap(crop.canvas) : crop.canvas;
      const results = await this.paddleOcr.predict(input);
      if (input && typeof input.close === 'function') {
        try { input.close(); } catch (_) {}
      }
      const res = results && results[0];

      let text = '';
      let ocrConf = detScore;
      let refinedQuad = null;

      if (res && res.items && res.items.length > 0) {
        const getY = (pt) => (pt ? (pt.y !== undefined ? pt.y : pt[1]) : 0);
        const getX = (pt) => (pt ? (pt.x !== undefined ? pt.x : pt[0]) : 0);

        // Multi-line sorting: top line before bottom line, and within line left-to-right
        res.items.sort((a, b) => {
          const aY = a.poly && a.poly[0] && a.poly[2] ? (getY(a.poly[0]) + getY(a.poly[2])) / 2 : getY(a.poly?.[0]);
          const bY = b.poly && b.poly[0] && b.poly[2] ? (getY(b.poly[0]) + getY(b.poly[2])) / 2 : getY(b.poly?.[0]);
          const aH = a.poly && a.poly[0] && a.poly[2] ? Math.abs(getY(a.poly[2]) - getY(a.poly[0])) : 12;
          const bH = b.poly && b.poly[0] && b.poly[2] ? Math.abs(getY(b.poly[2]) - getY(b.poly[0])) : 12;
          const minH = Math.min(aH, bH);
          if (Math.abs(aY - bY) > minH * 0.45) {
            return aY - bY; // Top to bottom
          }
          const aX = getX(a.poly?.[0]);
          const bX = getX(b.poly?.[0]);
          return aX - bX; // Left to right
        });

        text = res.items
          .map((item) => item.text)
          .join('')
          .trim()
          .toUpperCase()
          .replace(/[^A-Z0-9]/g, '');

        const avgScore = res.items.reduce((s, it) => s + (it.score || 0.8), 0) / res.items.length;
        ocrConf = avgScore;

        // If PaddleOCR detected valid 4-point text polygon(s), map them back to image space
        // so the bounding quadrilateral precisely aligns with the perspective slant and orientation of all text lines
        const validPolys = res.items.map((it) => it.poly).filter((p) => p && p.length === 4);
        if (validPolys.length > 0) {
          const firstPoly = validPolys[0];
          const lastPoly = validPolys[validPolys.length - 1];

          // Top edge from first line, bottom edge from last line
          const rawQuad = [
            [firstPoly[0][0], firstPoly[0][1]], // Top-Left
            [firstPoly[1][0], firstPoly[1][1]], // Top-Right
            [lastPoly[2][0], lastPoly[2][1]],   // Bottom-Right
            [lastPoly[3][0], lastPoly[3][1]]    // Bottom-Left
          ];

          if (validPolys.length > 1) {
            // Expand horizontal bounds to encompass all lines
            const minX = Math.min(...validPolys.flatMap((p) => [p[0][0], p[3][0]]));
            const maxX = Math.max(...validPolys.flatMap((p) => [p[1][0], p[2][0]]));
            rawQuad[0][0] = Math.min(rawQuad[0][0], minX);
            rawQuad[3][0] = Math.min(rawQuad[3][0], minX);
            rawQuad[1][0] = Math.max(rawQuad[1][0], maxX);
            rawQuad[2][0] = Math.max(rawQuad[2][0], maxX);
          }

          const scaleCrop = crop.sw / crop.targetW;
          const cx = (rawQuad[0][0] + rawQuad[1][0] + rawQuad[2][0] + rawQuad[3][0]) / 4;
          const cy = (rawQuad[0][1] + rawQuad[1][1] + rawQuad[2][1] + rawQuad[3][1]) / 4;
          const expanded = rawQuad.map(([px, py]) => [
            cx + (px - cx) * 1.18,
            cy + (py - cy) * 1.25
          ]);
          refinedQuad = expanded.map(([px, py]) => [
            crop.sx + px * scaleCrop,
            crop.sy + py * scaleCrop
          ]);
        }
      }

      const combinedConf = Math.min(0.99, Math.max(0.60, (detScore * 0.4) + (ocrConf * 0.6)));

      return {
        text,
        minConf: text ? combinedConf : detScore,
        confident: text.length >= 3,
        refinedQuad
      };
    } catch (err) {
      console.warn('PaddleOCR read error:', err);
      return { text: '', minConf: detScore, confident: false, refinedQuad: null };
    }
  }

  async readAll(image, opts = {}) {
    const maxBoxes = opts.maxBoxes !== undefined ? opts.maxBoxes : 5;
    const dets = await this.detect(image, opts);

    // Limit candidate boxes sent to PaddleOCR to at most maxBoxes (default 5),
    // prioritizing the largest texts (by box area) to prevent latency spikes and
    // bounding box flooding when pointing at text-dense scenes (e.g. paper or documents).
    const prioritizedDets = dets
      .sort((a, b) => {
        const areaA = a.box.width * a.box.height;
        const areaB = b.box.width * b.box.height;
        if (Math.abs(areaB - areaA) > Math.min(areaA, areaB) * 0.15) {
          return areaB - areaA; // Largest text first
        }
        return b.score - a.score; // Higher confidence secondary
      })
      .slice(0, maxBoxes);

    const out = [];

    for (const d of prioritizedDets) {
      const ocrResult = await this.readWithOcr(image, d.box, d.score, d.quad);
      if (!ocrResult.text || ocrResult.text.length < 2) continue;

      out.push({
        score: d.score,
        box: d.box,
        quad: ocrResult.refinedQuad || d.quad,
        text: ocrResult.text || '',
        minConf: ocrResult.minConf || d.score,
        confident: ocrResult.confident !== false
      });
    }

    out.sort((a, b) => b.score - a.score);
    return out;
  }
}

/**
 * Fits pictures to max 1080px in width or height
 */
export function fitFrameToMax1080(source) {
  const origW = source.videoWidth || source.naturalWidth || source.width;
  const origH = source.videoHeight || source.naturalHeight || source.height;
  if (!origW || !origH) return null;

  let targetW = origW;
  let targetH = origH;
  const maxDim = 1080;

  if (targetW > maxDim || targetH > maxDim) {
    if (targetW >= targetH) {
      targetH = Math.round((targetH * maxDim) / targetW);
      targetW = maxDim;
    } else {
      targetW = Math.round((targetW * maxDim) / targetH);
      targetH = maxDim;
    }
  }

  const canvas = typeof document !== 'undefined'
    ? document.createElement('canvas')
    : new OffscreenCanvas(targetW, targetH);
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, targetW, targetH);

  return {
    canvas,
    width: targetW,
    height: targetH,
    origWidth: origW,
    origHeight: origH,
    scaleX: targetW / origW,
    scaleY: targetH / origH
  };
}
