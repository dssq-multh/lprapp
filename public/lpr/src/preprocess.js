// Letterbox image + convert to Float32 NCHW tensor for ONNX Runtime Web.
// Mirrors _letterbox_uint8 in src/lpr/inference/pipeline.py:59.

/**
 * BT.601 limited-range RGB → YUV per-pixel, normalised (x − 128) / 128.
 * Mirrors YUVSpotterPreprocess / YUVRecognizerPreprocess in
 * the training-time preprocessors these models were fitted with.
 *
 * @param {Uint8ClampedArray} imgData  RGBA, length N*4
 * @param {Float32Array} out           NCHW Float32, length N*3 — written into.
 * @param {number} N                   pixel count (= W*H)
 */
export function rgbaToYuvNormalised(imgData, out, N) {
  for (let i = 0; i < N; i++) {
    const r = imgData[4 * i], g = imgData[4 * i + 1], b = imgData[4 * i + 2];
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.169 * r - 0.331 * g + 0.500 * b + 128;
    let v = 0.500 * r - 0.419 * g - 0.081 * b + 128;
    if (y < 0) y = 0; else if (y > 255) y = 255;
    if (u < 0) u = 0; else if (u > 255) u = 255;
    if (v < 0) v = 0; else if (v > 255) v = 255;
    out[i] = (y - 128) / 128;
    out[N + i] = (u - 128) / 128;
    out[2 * N + i] = (v - 128) / 128;
  }
}

/**
 * Letterbox an image bitmap into a target (W, H) canvas with constant-pad fill.
 * Returns the canvas (for preview), the Float32Array NCHW tensor, and the
 * scale/pad metadata needed to project detector outputs back to the original.
 *
 * @param {ImageBitmap|HTMLImageElement} img
 * @param {number} targetW
 * @param {number} targetH
 * @param {{padValue?: number, norm?: "div255"|"std"|"yuv"}} [opts]
 *   norm: "div255" → x/255 (Ultralytics convention: YOLOv8-OBB, fast-alpr YOLOv9);
 *         "std"    → (x-127.5)/128 (our Tier B v3 spotter's RGB normalization);
 *         "yuv"    → BT.601 RGB→YUV (Y plane + Cb plane + Cr plane), each channel
 *                    normalised (x-128)/128. Our Tier B v11 spotter's NV12-native
 *                    input — first channel is the camera ISP's luma plane, 2nd/3rd
 *                    are the upsampled chroma plane. Matches the
 *                    spotter's training-time preprocessor.
 *   Using the wrong one silently wrecks detection.
 * @returns {{ tensor: Float32Array, scale, padX, padY, origW, origH, canvas }}
 */
export function letterbox(img, targetW, targetH, { padValue = 114, norm = "div255" } = {}) {
  const origW = img.width;
  const origH = img.height;
  const scale = Math.min(targetW / origW, targetH / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = Math.floor((targetW - newW) / 2);
  const padY = Math.floor((targetH - newH) / 2);

  const canvas = new OffscreenCanvas(targetW, targetH);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = `rgb(${padValue},${padValue},${padValue})`;
  ctx.fillRect(0, 0, targetW, targetH);
  // Match PIL bilinear (training-time resampler) — default canvas smoothing is
  // browser "low" quality (typically box-filter), which on a 4-5× downsample
  // shifts pixel values enough to drop a borderline detection (e.g. random_18
  // M089AH24 scores 0.54 with PIL but ~0 in the browser with default quality).
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, origW, origH, padX, padY, newW, newH);

  const imgData = ctx.getImageData(0, 0, targetW, targetH).data; // RGBA, uint8
  const N = targetW * targetH;

  // "y" → single-channel BT.601 luma, normalised (y-128)/128. Matches the Tier E
  // YuNet spotter's Y-only input (SpotterPreprocess y_only, mean/std 128) — the
  // NV12 luma plane the camera ISP delivers. Returns an [1,1,H,W] tensor.
  if (norm === "y") {
    const out1 = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let y = 0.299 * imgData[4 * i] + 0.587 * imgData[4 * i + 1] + 0.114 * imgData[4 * i + 2];
      if (y < 0) y = 0; else if (y > 255) y = 255;
      out1[i] = (y - 128) / 128;
    }
    return { tensor: out1, scale, padX, padY, origW, origH, canvas };
  }

  const out = new Float32Array(3 * N);
  // NCHW layout per branch:
  //   "std" / "div255" → channel order = R, G, B (planes 0/1/2)
  //   "yuv"            → channel order = Y, Cb (=U), Cr (=V)
  if (norm === "yuv") {
    rgbaToYuvNormalised(imgData, out, N);
  } else {
    const std = norm === "std";
    for (let i = 0; i < N; i++) {
      const r = imgData[4 * i], g = imgData[4 * i + 1], b = imgData[4 * i + 2];
      if (std) {
        out[i] = (r - 127.5) / 128; out[N + i] = (g - 127.5) / 128; out[2 * N + i] = (b - 127.5) / 128;
      } else {
        out[i] = r / 255; out[N + i] = g / 255; out[2 * N + i] = b / 255;
      }
    }
  }
  return { tensor: out, scale, padX, padY, origW, origH, canvas };
}

/**
 * Project a 4-corner quad from letterboxed-canvas coords back to original image
 * coords. Inverse of the letterbox affine.
 *
 * @param {number[][]} quad   shape [4][2]
 * @param {number} scale
 * @param {number} padX
 * @param {number} padY
 * @returns {number[][]} quad in original-image coords
 */
export function unletterboxQuad(quad, scale, padX, padY) {
  return quad.map(([x, y]) => [(x - padX) / scale, (y - padY) / scale]);
}
