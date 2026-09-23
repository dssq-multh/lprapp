// Rotated-IoU NMS via Sutherland-Hodgman polygon clipping.
// Ported from src/lpr/postproc/obb.py — same math, JS-native.

function polygonArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

function ensureCCW(poly) {
  // In image coords (y down) shoelace sign is flipped; we just want consistent winding.
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return s > 0 ? poly : poly.slice().reverse();
}

function segIntersect(p1, p2, p3, p4) {
  const [x1, y1] = p1, [x2, y2] = p2, [x3, y3] = p3, [x4, y4] = p4;
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denom) < 1e-12) return p1;
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
  return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
}

function sutherlandHodgman(subject, clip) {
  let output = ensureCCW(subject);
  const clipPoly = ensureCCW(clip);
  for (let i = 0; i < clipPoly.length; i++) {
    if (output.length === 0) break;
    const input = output;
    output = [];
    const a = clipPoly[i];
    const b = clipPoly[(i + 1) % clipPoly.length];
    const edge = [b[0] - a[0], b[1] - a[1]];
    for (let j = 0; j < input.length; j++) {
      const curr = input[j];
      const prev = input[(j - 1 + input.length) % input.length];
      const sideCurr = edge[0] * (curr[1] - a[1]) - edge[1] * (curr[0] - a[0]);
      const sidePrev = edge[0] * (prev[1] - a[1]) - edge[1] * (prev[0] - a[0]);
      const insideCurr = sideCurr >= 0;
      const insidePrev = sidePrev >= 0;
      if (insideCurr) {
        if (!insidePrev) output.push(segIntersect(prev, curr, a, b));
        output.push(curr);
      } else if (insidePrev) {
        output.push(segIntersect(prev, curr, a, b));
      }
    }
  }
  return output;
}

/** Polygon IoU between two 4-corner quads. */
export function polygonIoU(quadA, quadB) {
  const inter = sutherlandHodgman(quadA, quadB);
  if (inter.length < 3) return 0;
  const interArea = polygonArea(inter);
  const a = polygonArea(quadA);
  const b = polygonArea(quadB);
  const denom = a + b - interArea;
  return denom > 0 ? interArea / denom : 0;
}

/**
 * Greedy NMS over rotated boxes.
 * @param {Array<{quad:number[][], score:number}>} dets
 * @param {number} iouThresh
 */
export function nmsOBB(dets, iouThresh = 0.4) {
  const sorted = dets.slice().sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of sorted) {
    let drop = false;
    for (const k of keep) {
      if (polygonIoU(d.quad, k.quad) >= iouThresh) { drop = true; break; }
    }
    if (!drop) keep.push(d);
  }
  return keep;
}

/** Build the 4-corner TL,TR,BR,BL quad from (cx, cy, w, h, theta). */
export function obbToQuad(cx, cy, w, h, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  const hw = w / 2, hh = h / 2;
  const local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  return local.map(([x, y]) => [c * x - s * y + cx, s * x + c * y + cy]);
}
