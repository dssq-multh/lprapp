// Decode Tier B spotter outputs to a list of rotated detections.
// Port of decode_tier_b_outputs from src/lpr/postproc/obb.py:113.
//
// Output convention per scale: [1, 8, H_s, W_s]
//   ch0 = obj logit -> sigmoid(obj) is the score
//   ch1, ch2 = dx, dy (cell offset in stride units)
//   ch3, ch4 = log_w, log_h
//   ch5, ch6 = sin(2θ), cos(2θ)  -> θ = atan2(sin2, cos2) / 2
//   ch7 = cls (unused, single-class)

import { obbToQuad, nmsOBB } from "./nms.js";

const SIGMOID = (x) => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x))));

/**
 * @param {{data: Float32Array, dims: number[]}} pred  ORT tensor for one scale [1,8,H,W]
 * @param {number} stride
 * @param {number} scoreThresh
 * @param {number} topK
 */
function decodeOneScale(pred, stride, scoreThresh, topK) {
  const [_, C, H, W] = pred.dims;
  if (C !== 8) throw new Error(`expected 8 channels, got ${C}`);
  const d = pred.data;
  // Channel slabs: each is H*W floats.
  const plane = H * W;
  const ch = (c) => c * plane;
  // First pass: collect cells above threshold.
  const peaks = [];
  for (let i = 0; i < plane; i++) {
    const score = SIGMOID(d[ch(0) + i]);
    if (score >= scoreThresh) peaks.push({ i, score });
  }
  peaks.sort((a, b) => b.score - a.score);
  const kept = peaks.slice(0, topK);
  const dets = [];
  for (const { i, score } of kept) {
    const gy = Math.floor(i / W);
    const gx = i - gy * W;
    const dx = d[ch(1) + i], dy = d[ch(2) + i];
    const logW = d[ch(3) + i], logH = d[ch(4) + i];
    const sin2 = d[ch(5) + i], cos2 = d[ch(6) + i];
    const cx = (gx + 0.5) * stride + dx * stride;
    const cy = (gy + 0.5) * stride + dy * stride;
    const w = Math.exp(logW) * stride;
    const h = Math.exp(logH) * stride;
    const theta = Math.atan2(sin2, cos2) / 2;
    dets.push({ cx, cy, w, h, theta, score, quad: obbToQuad(cx, cy, w, h, theta) });
  }
  return dets;
}

/**
 * Decode both spotter scales and return NMS'd detections.
 * @param {object} outputs   ORT outputs map; keys are tensor names
 * @param {string} nameP3    output name for stride-8 head
 * @param {string} nameP4    output name for stride-16 head
 * @param {number} scoreThresh
 * @param {number} iouThresh
 */
export function decodeSpotter(outputs, nameP3, nameP4,
                                scoreThresh = 0.40, iouThresh = 0.4, topK = 50) {
  const p3 = outputs[nameP3];
  const p4 = outputs[nameP4];
  const dets = [
    ...decodeOneScale(p3, 8, scoreThresh, topK),
    ...decodeOneScale(p4, 16, scoreThresh, topK),
  ];
  return nmsOBB(dets, iouThresh);
}
