import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const groundTruthPath = path.resolve(__dirname, '../public/samples/ground_truth.json');
const testCases = JSON.parse(fs.readFileSync(groundTruthPath, 'utf-8'));

const IMAGES = testCases.map(tc => ({
  name: tc.name,
  path: tc.path,
  groundTruth: tc.allPlates
}));

const targetPlatesList = [...new Set(testCases.flatMap(tc => tc.allPlates))];

const userDataDir = `/tmp/chrome_audit_yolov8_${Date.now()}`;
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless',
  '--disable-gpu',
  '--ignore-certificate-errors',
  '--allow-insecure-localhost',
  `--user-data-dir=${userDataDir}`,
  '--remote-debugging-port=9230',
  'https://localhost:5173/lprapp/'
]);

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

(async () => {
  let ws;
  try {
    console.log('Connecting to headless Chrome on port 9230...');
    let target = null;
    for (let i = 0; i < 30; i++) {
      await sleep(500);
      try {
        const res = await fetch('http://127.0.0.1:9230/json');
        const list = await res.json();
        target = list.find(p => p.url && p.url.includes('localhost:5173'));
        if (target) break;
      } catch (e) {}
    }

    if (!target) throw new Error('Target page localhost:5173 not found.');

    ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onerror = rej;
    });

    let msgId = 1;
    const pending = new Map();

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.id && pending.has(data.id)) {
        const cb = pending.get(data.id);
        pending.delete(data.id);
        cb(data.result);
      }
    };

    function sendCommand(method, params = {}) {
      return new Promise((resolve) => {
        const id = msgId++;
        pending.set(id, resolve);
        ws.send(JSON.stringify({ id, method, params }));
      });
    }

    await sendCommand('Runtime.enable');

    console.log('Waiting for LPR Engine & PaddleOCR readiness...');
    let ready = false;
    for (let i = 0; i < 60; i++) {
      const evalRes = await sendCommand('Runtime.evaluate', {
        expression: '!!(window.__lprEngine && window.__lprEngine.isReady)',
        returnByValue: true
      });
      if (evalRes?.result?.value === true) {
        ready = true;
        break;
      }
      await sleep(500);
    }

    if (!ready) throw new Error('Timeout waiting for LPR Engine');
    console.log('Engine ready! Initializing YOLOv8 session in browser...\n');

    // Initialize YOLOv8 in browser context
    const initRes = await sendCommand('Runtime.evaluate', {
      awaitPromise: true,
      returnByValue: true,
      expression: `(async () => {
        try {
          const ort = window.ort || (await import('/lprapp/node_modules/onnxruntime-web/dist/ort.bundle.min.mjs'));
          window.__ort = ort;
          const session = await ort.InferenceSession.create('/lprapp/lpr/models/license_plate_detector_yolov8n.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'disabled',
            intraOpNumThreads: 1
          });
          window.__yolov8Session = session;
          return { success: true };
        } catch (err) {
          return { success: false, error: err.message, stack: err.stack };
        }
      })()`
    });

    if (!initRes?.result?.value?.success) {
      console.error('Failed to init YOLOv8 session:', initRes?.result?.value);
      return;
    }
    console.log('YOLOv8 ONNX session created successfully in browser!\n');

    // Populate target plates
    await sendCommand('Runtime.evaluate', {
      expression: `(() => {
        const ta = document.getElementById('platesTextarea');
        if (ta) {
          ta.value = ${JSON.stringify(targetPlatesList)}.join('\\n');
          ta.dispatchEvent(new Event('input'));
        }
      })()`
    });

    const results = [];

    for (let i = 0; i < IMAGES.length; i++) {
      const item = IMAGES[i];
      process.stdout.write(`[${i + 1}/${IMAGES.length}] Processing ${item.name}... `);

      const evalRes = await sendCommand('Runtime.evaluate', {
        awaitPromise: true,
        returnByValue: true,
        expression: `(async () => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.src = '/lprapp' + '${item.path}';
          await new Promise((res, rej) => {
            img.onload = res;
            img.onerror = rej;
          });

          const fittedRaw = window.__fitFrameToMax1080(img);
          const canvas1080 = fittedRaw.canvas;
          const origW = canvas1080.width;
          const origH = canvas1080.height;

          // Resize canvas to YOLOv8 input size [640, 384]
          const netW = 640;
          const netH = 384;
          const scaleCanvas = document.createElement('canvas');
          scaleCanvas.width = netW;
          scaleCanvas.height = netH;
          const sctx = scaleCanvas.getContext('2d');
          sctx.drawImage(canvas1080, 0, 0, netW, netH);
          const imgData = sctx.getImageData(0, 0, netW, netH).data;

          // Prepare float32 RGB planar tensor [1, 3, 384, 640] normalized to [0, 1]
          const tensorData = new Float32Array(3 * netH * netW);
          const planeSize = netH * netW;
          for (let p = 0; p < planeSize; p++) {
            tensorData[p] = imgData[p * 4] / 255.0;                      // R
            tensorData[planeSize + p] = imgData[p * 4 + 1] / 255.0;      // G
            tensorData[planeSize * 2 + p] = imgData[p * 4 + 2] / 255.0;  // B
          }

          const t0 = performance.now();
          const tensor = new window.__ort.Tensor('float32', tensorData, [1, 3, netH, netW]);
          const outMap = await window.__yolov8Session.run({ images: tensor });
          const yoloDurMs = Math.round(performance.now() - t0);

          const outData = outMap.output0.data;
          // shape: (1, 5, 5040)
          const numAnchors = 5040;
          const candidates = [];
          for (let a = 0; a < numAnchors; a++) {
            const score = outData[4 * numAnchors + a];
            if (score >= 0.20) {
              const cx = outData[0 * numAnchors + a];
              const cy = outData[1 * numAnchors + a];
              const w = outData[2 * numAnchors + a];
              const h = outData[3 * numAnchors + a];
              candidates.push({
                cx, cy, w, h, score,
                left: cx - w / 2,
                top: cy - h / 2,
                right: cx + w / 2,
                bottom: cy + h / 2
              });
            }
          }

          // NMS
          candidates.sort((a, b) => b.score - a.score);
          const kept = [];
          const iouThresh = 0.35;
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
            if (!overlap) kept.push(cand);
          }

          // Map kept boxes back to canvas1080 coordinates and run Paddle OCR
          const scaleX = origW / netW;
          const scaleY = origH / netH;

          const dets = [];
          for (const k of kept) {
            const bx = Math.max(0, k.left * scaleX);
            const by = Math.max(0, k.top * scaleY);
            const bw = Math.min(origW - bx, k.w * scaleX);
            const bh = Math.min(origH - by, k.h * scaleY);

            // Pad slightly for OCR context
            const padX = Math.round(bw * 0.08);
            const padY = Math.round(bh * 0.08);
            const sx = Math.max(0, bx - padX);
            const sy = Math.max(0, by - padY);
            const sw = Math.min(origW - sx, bw + padX * 2);
            const sh = Math.min(origH - sy, bh + padY * 2);

            // Scale crop reasonably (e.g. 2x, but capped to avoid excessive blur)
            const scaleFactor = Math.max(1, Math.min(2.0, 72 / Math.max(1, sh)));
            const tw = Math.round(sw * scaleFactor);
            const th = Math.round(sh * scaleFactor);

            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = tw;
            cropCanvas.height = th;
            const cctx = cropCanvas.getContext('2d');
            cctx.drawImage(canvas1080, sx, sy, sw, sh, 0, 0, tw, th);

            const ocrRes = await window.__lprEngine.predictDirectPaddle(cropCanvas);
            const items = ocrRes?.results?.[0]?.items || [];

            // Sort top-to-bottom then left-to-right
            items.sort((a, b) => {
              const aY = (a.poly[0][1] + a.poly[2][1]) / 2;
              const bY = (b.poly[0][1] + b.poly[2][1]) / 2;
              const aH = Math.abs(a.poly[2][1] - a.poly[0][1]);
              const bH = Math.abs(b.poly[2][1] - b.poly[0][1]);
              if (Math.abs(aY - bY) > Math.min(aH, bH) * 0.45) {
                return aY - bY;
              }
              return a.poly[0][0] - b.poly[0][0];
            });

            const rawText = items.map(it => it.text).join('').toUpperCase().replace(/[^A-Z0-9]/g, '');
            const avgScore = items.length ? items.reduce((s, it) => s + it.score, 0) / items.length : 0;

            const targets = window.__getTargetPlates ? window.__getTargetPlates() : new Set();
            const m = window.__matchPlate(rawText, targets);

            dets.push({
              text: rawText,
              yoloScore: Math.round(k.score * 100) / 100,
              ocrScore: Math.round(avgScore * 100) / 100,
              box: { left: bx, top: by, width: bw, height: bh },
              isMatch: m.isMatch,
              matchedPlate: m.matchedPlate
            });
          }

          return { yoloDurMs, dets };
        })()`
      });

      const { yoloDurMs, dets } = evalRes.result.value;
      const matched = dets.filter(d => d.isMatch).map(d => d.matchedPlate);
      console.log(`done (YOLO: ${yoloDurMs}ms, ${dets.length} dets, matches: [${matched.join(', ')}])`);

      results.push({
        name: item.name,
        path: item.path,
        groundTruth: item.groundTruth,
        yoloDurMs,
        detections: dets,
        matchedPlates: matched
      });
    }

    const outputPath = path.resolve(__dirname, '../public/samples/results_yolov8.json');
    fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
    console.log('\n======================================================');
    console.log('YOLOv8 Results saved to:', outputPath);

  } catch (err) {
    console.error('Audit failed:', err);
  } finally {
    if (ws) ws.close();
    chrome.kill();
  }
})();
