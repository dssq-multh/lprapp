/*
 * The plate reader, packaged to the contract www/a/lpr-loader.js expects.
 *
 *     import * as lpr from '<base>/lpr.js';
 *     const s = await lpr.open({ base });
 *     const cands = await s.readAll(image);   // detect + read, best read first
 *
 * Two models, in sequence: a YOLOv5n-OBB spotter (0.24 M) that finds plate
 * quads, and a TPS-STN + attention recogniser (1.98 M) that reads one. Both
 * consume BT.601 YUV, which is what the camera's ISP already produces.
 *
 * What is HERE and not in the upstream demo is the tiling, and it is the
 * difference between finding plates and finding nothing. The spotter's input is
 * 416x256. Letterbox a 2592x1944 sensor frame into that and a 50-pixel plate
 * becomes eight pixels: measured on a lab hi3516ev300, a single full-frame pass
 * finds ZERO plates in a car park that visibly contains several. Forty-two
 * overlapping 624x384 tiles find them.
 *
 * The other measured thing worth knowing: the spotter's own scores do not
 * separate a plate from a stretch of tarmac -- 0.56 for a real plate against
 * 0.51 for a kerb on the same frame. So `readAll` ranks by how well each
 * candidate READS, not by how confidently it was found, and returns the lot so
 * the caller can show the rejects rather than pretend they were not there.
 */
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs';

const SPOT_W = 416, SPOT_H = 256;
const REC_W = 130, REC_H = 70;

/* Tile size and overlap. Big enough that a tile is worth a forward pass, small
 * enough that a 50 px plate survives the letterbox into 416x256 -- at 624 wide
 * the scale is 0.67, so fifty pixels arrive as thirty-three. The overlap is
 * what stops a plate that straddles a tile edge from being cut in half. */
const TILE_W = 624, TILE_H = 384, OVERLAP = 0.25;

/* The spotter's own floor. Deliberately below the demo's 0.40: on a camera
 * looking down at a car park the real plates sit around 0.55 and the rubbish
 * around 0.5, so a high floor loses plates without losing rubbish. The reader
 * is what sorts them afterwards. */
const DET_THRESH = 0.30;
const NMS_IOU = 0.30;

/* Below this the reader has not read anything it should be believed about. A
 * confident wrong registration is worse than none: on the lab camera the real
 * plates come back at 0.97 and 1.00, tarmac at 0.16 to 0.32. */
const READ_FLOOR = 0.5;

/* Crop padding around the detected box, as a fraction of its own size.
 * The model's contract says 8% symmetric; measured against the Python
 * reference on the same frame, 10% across and 22% down reads better on plates
 * this small -- a 14-pixel-tall plate gets three pixels of margin instead of
 * one, and the TPS-STN has something to grip. */
const PAD_X = 0.10, PAD_Y = 0.22;

let mods = null;
async function helpers(base) {
	if (mods) return mods;
	const [pre, warp, spot, rec] = await Promise.all([
		import(base + 'src/preprocess.js'),
		import(base + 'src/warp.js'),
		import(base + 'src/decoder_spotter.js'),
		import(base + 'src/decoder_recognizer.js'),
	]);
	mods = { pre, warp, spot, rec };
	return mods;
}

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

/* Across tiles, not within one: decodeSpotter already suppresses inside its own
 * pass, and the only duplicates left are the same plate seen from two
 * overlapping tiles. Axis-aligned is enough for that -- the two views of one
 * plate differ by a pixel or two, not by an angle. */
function dedupe(dets) {
	const out = [];
	dets.sort((p, q) => q.score - p.score);
	for (const d of dets) if (!out.some((k) => iou(d.box, k.box) > NMS_IOU)) out.push(d);
	return out;
}

function tiles(w, h) {
	const sx = Math.round(TILE_W * (1 - OVERLAP)), sy = Math.round(TILE_H * (1 - OVERLAP));
	const out = [];
	for (let y = 0; y < Math.max(h - TILE_H, 0) + sy; y += sy) {
		for (let x = 0; x < Math.max(w - TILE_W, 0) + sx; x += sx) {
			out.push({ x: Math.min(x, Math.max(w - TILE_W, 0)), y: Math.min(y, Math.max(h - TILE_H, 0)),
				w: Math.min(TILE_W, w), h: Math.min(TILE_H, h) });
		}
	}
	// A frame smaller than one tile is one tile.
	return out.length ? out : [{ x: 0, y: 0, w: w, h: h }];
}

function cut(img, t) {
	const c = new OffscreenCanvas(t.w, t.h);
	c.getContext('2d', { willReadFrequently: true })
		.drawImage(img, t.x, t.y, t.w, t.h, 0, 0, t.w, t.h);
	return c;
}

export async function open(opts) {
	const base = (opts && opts.base) || './';
	const h = await helpers(base);
	ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

	/* SEQUENTIALLY. ORT-web must finish initWasm() once before any other
	 * session starts; two at once throws "multiple calls to 'initWasm()'". */
	const spotter = await ort.InferenceSession.create(base + 'models/spotter_b_v12_yuv_pm.onnx',
		{ executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' });
	// WASM only: the recogniser's unrolled LSTM and GridSample-16 are opset 16,
	// which onnxruntime-web 1.20 does not run on WebGPU.
	const recogniser = await ort.InferenceSession.create(base + 'models/recog_f_v26_attn.onnx',
		{ executionProviders: ['wasm'], graphOptimizationLevel: 'all' });

	async function detectIn(canvas, ox, oy) {
		const p = h.pre.letterbox(canvas, SPOT_W, SPOT_H, { norm: 'yuv' });
		const out = await spotter.run({
			[spotter.inputNames[0]]: new ort.Tensor('float32', p.tensor, [1, 3, SPOT_H, SPOT_W]),
		});
		const [p3, p4] = spotter.outputNames;
		return h.spot.decodeSpotter(out, p3, p4, DET_THRESH, NMS_IOU).map(function (d) {
			const q = h.pre.unletterboxQuad(d.quad, p.scale, p.padX, p.padY)
				.map((pt) => [pt[0] + ox, pt[1] + oy]);
			return { quad: q, box: boxOf(q), score: d.score };
		});
	}

	/*
	 * Every plate in the frame.
	 *
	 * `onProgress(done, total)` is called per tile, because forty-two forward
	 * passes is several seconds and a spinner that says nothing is the wrong
	 * answer to that.
	 */
	async function detect(image, o) {
		o = o || {};
		const w = image.width, h2 = image.height;
		const ts = o.tile === false ? [{ x: 0, y: 0, w: w, h: h2 }] : tiles(w, h2);
		const all = [];
		for (let i = 0; i < ts.length; i++) {
			const t = ts[i];
			const c = (t.w === w && t.h === h2 && t.x === 0 && t.y === 0) ? image : cut(image, t);
			all.push.apply(all, await detectIn(c, t.x, t.y));
			if (o.onProgress) o.onProgress(i + 1, ts.length);
		}
		return dedupe(all);
	}

	/*
	 * Read one.
	 *
	 * The feeding contract is an axis-aligned bounding box of the quad plus 8%
	 * symmetric pad, stretched to 130x70 -- NOT a perspective warp. The model's
	 * own TPS-STN rectifies tilt, scale and perspective, and a warp applied
	 * first fights it: an OBB emitted rotated 90 degrees turns into a sideways
	 * crop nothing can read.
	 */
	async function read(image, target, ro) {
		ro = ro || {};
		const quad = Array.isArray(target) ? target : [
			[target.left, target.top], [target.left + target.width, target.top],
			[target.left + target.width, target.top + target.height],
			[target.left, target.top + target.height],
		];
		const padX = ro.padX === undefined ? PAD_X : ro.padX;
		const padY = ro.padY === undefined ? PAD_Y : ro.padY;
		const xs = quad.map((q) => q[0]), ys = quad.map((q) => q[1]);
		let x0 = Math.min.apply(null, xs), y0 = Math.min.apply(null, ys);
		let x1 = Math.max.apply(null, xs), y1 = Math.max.apply(null, ys);
		const mx = (x1 - x0) * padX, my = (y1 - y0) * padY;
		x0 = Math.max(0, x0 - mx); y0 = Math.max(0, y0 - my);
		x1 = Math.min(image.width, x1 + mx); y1 = Math.min(image.height, y1 + my);
		const cc = new OffscreenCanvas(REC_W, REC_H);
		const cx = cc.getContext('2d', { willReadFrequently: true });
		cx.imageSmoothingEnabled = true; cx.imageSmoothingQuality = 'high';
		cx.drawImage(image, x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0), 0, 0, REC_W, REC_H);
		const rgba = cx.getImageData(0, 0, REC_W, REC_H).data;
		const out = await recogniser.run({
			[recogniser.inputNames[0]]: new ort.Tensor('float32',
				h.warp.cropToTensor(rgba, REC_W, REC_H, { norm: 'yuv' }), [1, 3, REC_H, REC_W]),
		});
		const lg = out[recogniser.outputNames[0]];
		const d = h.rec.decodeAttention(lg.data, lg.dims[1], lg.dims[2]);
		return { text: d.text, minConf: d.minConf, perChar: d.perPosConf, confident: d.minConf >= READ_FLOOR };
	}

	/* Detect, read each, and rank by the READ rather than the detection --
	 * measured: the spotter cannot tell a plate from a kerb, the reader can. */
	async function readAll(image, o) {
		const dets = await detect(image, o);
		const out = [];
		for (const d of dets) {
			try {
				out.push(Object.assign({}, d, await read(image, d.quad, o)));
			} catch (e) {
				out.push(Object.assign({}, d, { text: '', minConf: 0, perChar: [], confident: false }));
			}
		}
		out.sort((a, b) => b.minConf - a.minConf);
		return out;
	}

	return {
		detect: detect, read: read, readAll: readAll,
		floor: READ_FLOOR,
		close: function () { try { spotter.release(); recogniser.release(); } catch (e) { /* done anyway */ } },
	};
}
