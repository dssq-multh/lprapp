// Perspective-warp a 4-point quad from the source image to a fixed (outW, outH)
// canvas crop. Used to extract plate crops in the recognizer's canonical shape.
//
// A port of the Python inference path's OBB warp and perspective matrix, so a
// crop made here is the one the model was fed at training and evaluation time.

/** Solve 8 linear equations for the projective matrix mapping src→dst (4 point pairs each). */
function perspectiveMatrix(srcPts, dstPts) {
  // Build 8x9 matrix [A | b] then Gaussian-eliminate to get h[0..7], with h33 = 1.
  const A = new Array(8);
  for (let i = 0; i < 4; i++) {
    const [x, y]   = srcPts[i];
    const [xp, yp] = dstPts[i];
    A[2 * i]     = [x, y, 1, 0, 0, 0, -xp * x, -xp * y, xp];
    A[2 * i + 1] = [0, 0, 0, x, y, 1, -yp * x, -yp * y, yp];
  }
  // Gauss-Jordan
  for (let col = 0; col < 8; col++) {
    // Partial pivot
    let pivot = col;
    for (let r = col + 1; r < 8; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
    }
    [A[col], A[pivot]] = [A[pivot], A[col]];
    const div = A[col][col] || 1e-12;
    for (let c = 0; c < 9; c++) A[col][c] /= div;
    for (let r = 0; r < 8; r++) {
      if (r === col) continue;
      const f = A[r][col];
      if (f === 0) continue;
      for (let c = 0; c < 9; c++) A[r][c] -= f * A[col][c];
    }
  }
  const h = A.map((row) => row[8]);
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1.0],
  ];
}

/**
 * Warp the source ImageBitmap region defined by `quad` into a (outW, outH) RGBA
 * Uint8ClampedArray. Bilinear sampling, BORDER_REPLICATE for off-image pixels.
 *
 * @param {ImageBitmap|HTMLImageElement} sourceImg
 * @param {number[][]} quad   4 corners in source-image pixel coords, TL→TR→BR→BL
 * @param {number} outW
 * @param {number} outH
 * @returns {{ canvas: OffscreenCanvas, rgba: Uint8ClampedArray }}
 */
export function warpQuadToCrop(sourceImg, quad, outW = 128, outH = 32) {
  // dst → src homography: solving "given dst corners, where in src does this come from"
  const dstCorners = [[0, 0], [outW, 0], [outW, outH], [0, outH]];
  const H = perspectiveMatrix(dstCorners, quad);

  // Get source pixel data
  const srcW = sourceImg.width, srcH = sourceImg.height;
  const srcCanvas = new OffscreenCanvas(srcW, srcH);
  const srcCtx = srcCanvas.getContext("2d", { willReadFrequently: true });
  srcCtx.drawImage(sourceImg, 0, 0);
  const srcData = srcCtx.getImageData(0, 0, srcW, srcH).data;

  const out = new Uint8ClampedArray(outW * outH * 4);
  for (let v = 0; v < outH; v++) {
    for (let u = 0; u < outW; u++) {
      // dst (u, v) -> src (x, y) via H
      const w = H[2][0] * u + H[2][1] * v + H[2][2];
      const x = (H[0][0] * u + H[0][1] * v + H[0][2]) / w;
      const y = (H[1][0] * u + H[1][1] * v + H[1][2]) / w;
      // Bilinear with BORDER_REPLICATE
      const xc = Math.max(0, Math.min(srcW - 1.001, x));
      const yc = Math.max(0, Math.min(srcH - 1.001, y));
      const x0 = Math.floor(xc), y0 = Math.floor(yc);
      const x1 = x0 + 1, y1 = y0 + 1;
      const dx = xc - x0, dy = yc - y0;
      const idx = (yy, xx) => 4 * (yy * srcW + xx);
      const i00 = idx(y0, x0), i10 = idx(y0, x1);
      const i01 = idx(y1, x0), i11 = idx(y1, x1);
      const outIdx = 4 * (v * outW + u);
      for (let ch = 0; ch < 3; ch++) {
        const top = srcData[i00 + ch] * (1 - dx) + srcData[i10 + ch] * dx;
        const bot = srcData[i01 + ch] * (1 - dx) + srcData[i11 + ch] * dx;
        out[outIdx + ch] = top * (1 - dy) + bot * dy;
      }
      out[outIdx + 3] = 255;
    }
  }

  const canvas = new OffscreenCanvas(outW, outH);
  canvas.getContext("2d").putImageData(new ImageData(out, outW, outH), 0, 0);
  return { canvas, rgba: out };
}

/**
 * Axis-aligned crop of a quad's bounding box, resized to (outW, outH) — NO
 * perspective warp. Used for the Plate Recognizer OCR: its 4 quad points sit at
 * the plate's real corners regardless of how the YOLOv8-OBB head labelled w/h/θ,
 * so their bounding box is always upright — this sidesteps the OBB orientation
 * ambiguity that rotates ~90° crops into garbage. The PR model rectifies any
 * residual tilt with its own internal spatial transformer, so a plain box crop
 * (with a little padding for context) reads better than a warped one. `pad` is a
 * fraction of the box size added on every side.
 *
 * @param {ImageBitmap|HTMLImageElement} sourceImg
 * @param {number[][]} quad   4 corners in source-image pixel coords
 * @returns {Uint8ClampedArray} RGBA of size outW*outH*4
 */
export function aabbCropToRgba(sourceImg, quad, outW = 130, outH = 70, pad = 0.08) {
  const xs = quad.map((p) => p[0]), ys = quad.map((p) => p[1]);
  let x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs), y1 = Math.max(...ys);
  const mx = (x1 - x0) * pad, my = (y1 - y0) * pad;
  x0 = Math.max(0, x0 - mx); y0 = Math.max(0, y0 - my);
  x1 = Math.min(sourceImg.width, x1 + mx); y1 = Math.min(sourceImg.height, y1 + my);
  const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);
  const canvas = new OffscreenCanvas(outW, outH);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(sourceImg, x0, y0, cw, ch, 0, 0, outW, outH);   // crop + bilinear resize
  return ctx.getImageData(0, 0, outW, outH).data;
}

/** Mean width / mean height of a TL→TR→BR→BL quad (in any consistent coords). */
export function quadAspect(quad) {
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const w = (d(quad[0], quad[1]) + d(quad[3], quad[2])) / 2;   // top + bottom edges
  const h = (d(quad[0], quad[3]) + d(quad[1], quad[2])) / 2;   // left + right edges
  return w / Math.max(1e-6, h);
}

/**
 * Build the Tier-d "fake 2-line" crop for a 1-line plate: warp the plate into
 * the TOP half (outW × topH) and fill the BOTTOM half with the grey pad value
 * the recognizer saw at training time. Mirrors the evaluation path
 * (H_top = 32, PAD_VALUE = 114) so a 1-line plate decodes through the 2-line
 * model identically to training/eval.
 *
 * @returns {Uint8ClampedArray} RGBA of size outW*fullH*4
 */
export function fakeTwoLineCrop(sourceImg, quad, outW, fullH, topH = 32, pad = 114) {
  const { rgba: top } = warpQuadToCrop(sourceImg, quad, outW, topH);
  const full = new Uint8ClampedArray(outW * fullH * 4);
  full.set(top, 0);                          // rows 0..topH-1 = warped plate
  for (let i = outW * topH; i < outW * fullH; i++) {   // rows topH..fullH-1 = grey pad
    full[4 * i] = pad; full[4 * i + 1] = pad; full[4 * i + 2] = pad; full[4 * i + 3] = 255;
  }
  return full;
}

import { rgbaToYuvNormalised } from "./preprocess.js";

/**
 * Convert a perspective-warped RGBA crop into a Float32 NCHW tensor.
 *
 * @param {Uint8ClampedArray} rgba  RGBA, length w*h*4
 * @param {number} w
 * @param {number} h
 * @param {{norm?: "std"|"yuv"}} [opts]
 *   norm:
 *     "std" (default) → (x − 127.5) / 128 RGB, matching the v14 recognizer.
 *     "yuv"           → BT.601 RGB → YUV per pixel, normalised (x − 128) / 128.
 *                       Matches the v15 recognizer (NV12-native) and the v12
 *                       spotter's preprocess so the deploy pipeline keeps one
 *                       basis end-to-end.
 */
export function cropToTensor(rgba, w, h, { norm = "std" } = {}) {
  const N = w * h;
  const out = new Float32Array(3 * N);
  if (norm === "yuv") {
    rgbaToYuvNormalised(rgba, out, N);
    return out;
  }
  for (let i = 0; i < N; i++) {
    out[i]         = (rgba[4 * i]     - 127.5) / 128;
    out[N + i]     = (rgba[4 * i + 1] - 127.5) / 128;
    out[2 * N + i] = (rgba[4 * i + 2] - 127.5) / 128;
  }
  return out;
}
