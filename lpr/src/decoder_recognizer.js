// CTC greedy decode for the Tier C recognizer (recog_c_v22.onnx, glyph_unified).
//
// Charset (40 classes incl. blank, blank at index 39):
//   0..9 = digits
//   10..22 = ambiguous Latin: A B C E H I K M O P T X Y
//   23..38 = Cyrillic-only: Б Г Д Ж З И Л П Ф Ц Ч Ш Щ Э Ю Я
//   39 = CTC blank
// The recogniser's own glyph set, in its class order. It has to match the
// order the model was trained against exactly: a shifted charset decodes every
// plate to confident nonsense.

export const GLYPH_UNIFIED = [
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
  "A", "B", "C", "E", "H", "I", "K", "M", "O", "P", "T", "X", "Y",
  "Б", "Г", "Д", "Ж", "З", "И", "Л", "П", "Ф", "Ц", "Ч", "Ш", "Щ", "Э", "Ю", "Я",
];
export const BLANK_ID = GLYPH_UNIFIED.length;   // 39

// Attention-decoder vocabulary (Tier f v22): PAD=0, EOS=1, SOS=2, then
// glyph_unified shifted by +3 = 42 classes total. Mirrors
// the attention decoder on the Python side.
export const ATTN_PAD = 0;
export const ATTN_EOS = 1;
export const ATTN_SOS = 2;
export const ATTN_VOCAB_OFFSET = 3;

/**
 * Greedy-decode attention-decoder logits.
 * @param {Float32Array} logits   length L*V (default V=42, L=10)
 * @param {number} L              steps
 * @param {number} V              vocabulary size
 * @returns {{ text: string, perPosConf: number[], minConf: number }}
 */
export function decodeAttention(logits, L, V) {
  const text = [];
  const perPosConf = [];
  for (let t = 0; t < L; t++) {
    const base = t * V;
    let maxV = -Infinity, maxK = 0;
    for (let k = 0; k < V; k++) {
      const v = logits[base + k];
      if (v > maxV) { maxV = v; maxK = k; }
    }
    // softmax confidence for the chosen class
    let denom = 0;
    for (let k = 0; k < V; k++) denom += Math.exp(logits[base + k] - maxV);
    const conf = 1 / denom;
    if (maxK === ATTN_EOS) break;
    if (maxK === ATTN_PAD || maxK === ATTN_SOS) continue;
    const charId = maxK - ATTN_VOCAB_OFFSET;
    if (charId >= 0 && charId < GLYPH_UNIFIED.length) {
      text.push(GLYPH_UNIFIED[charId]);
      perPosConf.push(conf);
    }
  }
  const minConf = perPosConf.length ? Math.min(...perPosConf) : 0;
  return { text: text.join(""), perPosConf, minConf };
}

// Indices 23..38 are the Cyrillic-only glyphs that never appear on
// RU / UA passenger plates (a 5k-sample bench measured ~5 A->D and 4 K->X
// confusions on v17). Masking them at decode time eliminates the failure
// mode at zero training cost; the same mask is applied on the training side.
export const DEAD_CLASS_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38];

/**
 * Greedy CTC decode + per-position softmax confidence.
 * @param {Float32Array} logits   length T*C, row-major NCHW [1, T, C]
 * @param {number} T
 * @param {number} C
 * @param {{maskDeadClasses?: boolean}} [opts]  When ``maskDeadClasses`` is true,
 *   the 16 Cyrillic-only-glyph classes are excluded from the argmax (their
 *   logits are treated as −∞) so the decoder cannot emit them.
 * @returns {{ text: string, perPosConf: number[], minConf: number }}
 */
export function greedyDecode(logits, T, C, { maskDeadClasses = true } = {}) {
  // Walk timesteps; for each pick argmax. Collapse consecutive duplicates; drop blank.
  const text = [];
  const perPosConf = [];
  let prev = -1;
  // Pre-compute the forbidden set as an array of indices to skip in the inner loop.
  const forbidden = maskDeadClasses
    ? DEAD_CLASS_IDS.filter((k) => k < C)
    : [];
  const forbidSet = new Set(forbidden);
  for (let t = 0; t < T; t++) {
    const base = t * C;
    // argmax over allowed classes only.
    let maxV = -Infinity;
    let maxK = 0;
    for (let k = 0; k < C; k++) {
      if (forbidSet.has(k)) continue;
      const v = logits[base + k];
      if (v > maxV) { maxV = v; maxK = k; }
    }
    // softmax denominator over ALL classes — the masked classes' softmax mass
    // gets redistributed only on argmax, not on the reported confidence so
    // ``minConf`` stays a meaningful "would the model have been sure here".
    let denom = 0;
    for (let k = 0; k < C; k++) denom += Math.exp(logits[base + k] - maxV);
    const conf = 1 / denom;
    if (maxK !== BLANK_ID && maxK !== prev) {
      text.push(GLYPH_UNIFIED[maxK]);
      perPosConf.push(conf);
    }
    prev = maxK;
  }
  const minConf = perPosConf.length ? Math.min(...perPosConf) : 0;
  return { text: text.join(""), perPosConf, minConf };
}
