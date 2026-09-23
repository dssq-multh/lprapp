import * as ort from 'onnxruntime-web';
import { letterbox, unletterboxQuad } from './preprocess.js';
import { decodeSpotter } from './decoder_spotter.js';
import { cropToTensor } from './warp.js';
import { decodeAttention } from './decoder_recognizer.js';

const SPOT_W = 416, SPOT_H = 256;
const REC_W = 130, REC_H = 70;
const TILE_W = 624, TILE_H = 384, OVERLAP = 0.25;
const DET_THRESH = 0.25;
const NMS_IOU = 0.30;
const READ_FLOOR = 0.40;
const PAD_X = 0.10, PAD_Y = 0.22;

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

export class LprEngine {
  constructor(opts = {}) {
    this.base = opts.base || '/lpr/';
    this.spotter = null;
    this.recogniser = null;
    this.isReady = false;
  }

  async init(onStatus) {
    if (this.isReady) return;

    if (onStatus) onStatus('Configuring WebAssembly runtime...');
    
    // Set wasm paths
    ort.env.wasm.wasmPaths = '/ort-wasm/';
    if (typeof window !== 'undefined' && !window.crossOriginIsolated) {
      ort.env.wasm.numThreads = 1;
    }

    if (onStatus) onStatus('Loading license plate detector model (YOLOv5n-OBB)...');
    
    const spotterUrl = `${this.base}models/spotter_b_v12_yuv_pm.onnx`;
    try {
      this.spotter = await ort.InferenceSession.create(spotterUrl, {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all'
      });
    } catch (e1) {
      console.warn('Initial spotter create failed, trying wasm provider only:', e1);
      try {
        this.spotter = await ort.InferenceSession.create(spotterUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
      } catch (e2) {
        console.warn('Local spotter load failed, trying CDN fallback:', e2);
        const cdnSpotter = 'https://cdn.jsdelivr.net/gh/OpenIPC/lpr-wasm@main/dist/models/spotter_b_v12_yuv_pm.onnx';
        this.spotter = await ort.InferenceSession.create(cdnSpotter, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
      }
    }

    if (onStatus) onStatus('Loading license plate OCR recognizer model (TPS-STN)...');
    const recogUrl = `${this.base}models/recog_f_v26_attn.onnx`;
    try {
      this.recogniser = await ort.InferenceSession.create(recogUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
    } catch (e1) {
      console.warn('Local recognizer load failed, trying CDN fallback:', e1);
      const cdnRecog = 'https://cdn.jsdelivr.net/gh/OpenIPC/lpr-wasm@main/dist/models/recog_f_v26_attn.onnx';
      this.recogniser = await ort.InferenceSession.create(cdnRecog, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
    }

    this.isReady = true;
    if (onStatus) onStatus('Wasm models ready!');
  }

  async detectIn(canvas, ox, oy) {
    const p = letterbox(canvas, SPOT_W, SPOT_H, { norm: 'yuv' });
    const out = await this.spotter.run({
      [this.spotter.inputNames[0]]: new ort.Tensor('float32', p.tensor, [1, 3, SPOT_H, SPOT_W]),
    });
    const [p3, p4] = this.spotter.outputNames;
    return decodeSpotter(out, p3, p4, DET_THRESH, NMS_IOU).map((d) => {
      const q = unletterboxQuad(d.quad, p.scale, p.padX, p.padY)
        .map((pt) => [pt[0] + ox, pt[1] + oy]);
      return { quad: q, box: boxOf(q), score: d.score };
    });
  }

  async detect(image, opts = {}) {
    const w = image.width, h = image.height;
    // For smaller images, single pass; for high res, use overlapping tiles
    const ts = opts.tile === false || (w <= SPOT_W * 1.5 && h <= SPOT_H * 1.5)
      ? [{ x: 0, y: 0, w, h }]
      : tiles(w, h);
    
    const all = [];
    for (let i = 0; i < ts.length; i++) {
      const t = ts[i];
      const c = (t.w === w && t.h === h && t.x === 0 && t.y === 0) ? image : cut(image, t);
      all.push(...(await this.detectIn(c, t.x, t.y)));
      if (opts.onProgress) opts.onProgress(i + 1, ts.length);
    }
    return dedupe(all);
  }

  async read(image, target, ro = {}) {
    const quad = Array.isArray(target) ? target : [
      [target.left, target.top],
      [target.left + target.width, target.top],
      [target.left + target.width, target.top + target.height],
      [target.left, target.top + target.height],
    ];
    const padX = ro.padX === undefined ? PAD_X : ro.padX;
    const padY = ro.padY === undefined ? PAD_Y : ro.padY;
    const xs = quad.map((q) => q[0]), ys = quad.map((q) => q[1]);
    let x0 = Math.min(...xs), y0 = Math.min(...ys);
    let x1 = Math.max(...xs), y1 = Math.max(...ys);
    const mx = (x1 - x0) * padX, my = (y1 - y0) * padY;
    x0 = Math.max(0, x0 - mx); y0 = Math.max(0, y0 - my);
    x1 = Math.min(image.width, x1 + mx); y1 = Math.min(image.height, y1 + my);

    const cc = new OffscreenCanvas(REC_W, REC_H);
    const cx = cc.getContext('2d', { willReadFrequently: true });
    cx.imageSmoothingEnabled = true;
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(image, x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0), 0, 0, REC_W, REC_H);

    const rgba = cx.getImageData(0, 0, REC_W, REC_H).data;
    const out = await this.recogniser.run({
      [this.recogniser.inputNames[0]]: new ort.Tensor('float32',
        cropToTensor(rgba, REC_W, REC_H, { norm: 'yuv' }), [1, 3, REC_H, REC_W]),
    });
    const lg = out[this.recogniser.outputNames[0]];
    const d = decodeAttention(lg.data, lg.dims[1], lg.dims[2]);
    return {
      text: d.text.trim().toUpperCase(),
      minConf: d.minConf,
      perChar: d.perPosConf,
      confident: d.minConf >= READ_FLOOR
    };
  }

  async readAll(image, opts = {}) {
    const dets = await this.detect(image, opts);
    const out = [];
    for (const d of dets) {
      try {
        const readResult = await this.read(image, d.quad, opts);
        out.push({ ...d, ...readResult });
      } catch (err) {
        console.warn('Plate read error:', err);
        out.push({ ...d, text: '', minConf: 0, perChar: [], confident: false });
      }
    }
    // Filter out completely blank or gibberish detections and sort by confidence
    return out
      .filter((item) => item.text && item.text.length >= 2)
      .sort((a, b) => b.minConf - a.minConf);
  }
}

/**
 * Fits any video or image frame to max 1080px in width or height
 * as requested: "Fit pictures to max 1080px in width or height."
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

  const canvas = document.createElement('canvas');
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
