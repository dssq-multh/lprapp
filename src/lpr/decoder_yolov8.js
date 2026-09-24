// Decodes YOLOv8 detection output tensor [1, 5, num_anchors]
// Channel 0: cx (center X)
// Channel 1: cy (center Y)
// Channel 2: w (width)
// Channel 3: h (height)
// Channel 4: score (class confidence)

function nmsAABB(candidates, iouThresh = 0.4) {
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const cand of candidates) {
    let overlap = false;
    for (const k of kept) {
      const xx1 = Math.max(cand.left, k.left);
      const yy1 = Math.max(cand.top, k.top);
      const xx2 = Math.min(cand.right, k.right);
      const yy2 = Math.min(cand.bottom, k.bottom);
      const interW = Math.max(0, xx2 - xx1);
      const interH = Math.max(0, yy2 - yy1);
      const inter = interW * interH;
      const union = (cand.w * cand.h) + (k.w * k.h) - inter;
      if (union > 0 && inter / union > iouThresh) {
        overlap = true;
        break;
      }
    }
    if (!overlap) {
      kept.push(cand);
    }
  }
  return kept;
}

/**
 * Decodes YOLOv8 output tensor and returns NMS-filtered detections.
 * @param {object} outputTensor ORT tensor for output0 [1, 5, N]
 * @param {number} scoreThresh Minimum detection confidence
 * @param {number} iouThresh NMS IoU threshold
 * @param {number} topK Maximum detections to keep
 */
export function decodeYolov8(outputTensor, scoreThresh = 0.25, iouThresh = 0.40, topK = 50) {
  if (!outputTensor || !outputTensor.data) return [];
  const dims = outputTensor.dims || [];
  const d = outputTensor.data;
  const numAnchors = dims.length === 3 ? dims[2] : Math.floor(d.length / 5);

  const candidates = [];
  const plane = numAnchors;

  for (let i = 0; i < numAnchors; i++) {
    const score = d[4 * plane + i];
    if (score >= scoreThresh) {
      const cx = d[0 * plane + i];
      const cy = d[1 * plane + i];
      const w = d[2 * plane + i];
      const h = d[3 * plane + i];
      const left = cx - w / 2;
      const top = cy - h / 2;
      const right = cx + w / 2;
      const bottom = cy + h / 2;

      candidates.push({
        cx,
        cy,
        w,
        h,
        score,
        left,
        top,
        right,
        bottom,
        quad: [
          [left, top],
          [right, top],
          [right, bottom],
          [left, bottom]
        ]
      });
    }
  }

  const kept = nmsAABB(candidates, iouThresh);
  return kept.slice(0, topK);
}
