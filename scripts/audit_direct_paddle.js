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

const userDataDir = `/tmp/chrome_audit_paddle_${Date.now()}`;
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless',
  '--disable-gpu',
  '--ignore-certificate-errors',
  '--allow-insecure-localhost',
  `--user-data-dir=${userDataDir}`,
  '--remote-debugging-port=9223',
  'https://localhost:5173/lprapp/'
]);

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

(async () => {
  let ws;
  try {
    console.log('Connecting to headless Chrome on port 9223...');
    let target = null;
    for (let i = 0; i < 20; i++) {
      await sleep(500);
      try {
        const res = await fetch('http://127.0.0.1:9223/json');
        const list = await res.json();
        target = list.find(p => p.url && p.url.includes('localhost:5173'));
        if (target) break;
      } catch (e) {}
    }

    if (!target) throw new Error('Target page localhost:5173 not found. Ensure "npm run dev" is running.');

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

    console.log('Enabling CDP Runtime...');
    await sendCommand('Runtime.enable');

    console.log('Waiting for LPR Engine & PaddleOCR readiness...');
    let ready = false;
    for (let i = 0; i < 100; i++) {
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

    if (!ready) throw new Error('Timeout waiting for LPR Engine & PaddleOCR');
    console.log('ALPR Engine & PaddleOCR are READY in browser!\n');

    // Populate target plates in the page
    await sendCommand('Runtime.evaluate', {
      expression: `(() => {
        const ta = document.getElementById('platesTextarea');
        if (ta) {
          ta.value = ${JSON.stringify(targetPlatesList)}.join('\\n');
          ta.dispatchEvent(new Event('input'));
        }
      })()`
    });

    const report = [];

    for (let i = 0; i < IMAGES.length; i++) {
      const item = IMAGES[i];
      console.log(`[${i + 1}/${IMAGES.length}] Evaluating ${item.name}`);
      console.log(`Ground Truth: [${item.groundTruth.join(', ')}]`);

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
          const jpegImg = new Image();
          await new Promise((res) => {
            jpegImg.onload = res;
            jpegImg.src = fittedRaw.canvas.toDataURL('image/jpeg', 0.95);
          });
          const fitted = window.__fitFrameToMax1080(jpegImg);

          // 1. Pipeline: YOLO spotter + PaddleOCR crop
          const tYolo0 = performance.now();
          const yoloRawDets = await window.__lprEngine.readAll(fitted.canvas, { tile: false });
          const yoloDurMs = Math.round(performance.now() - tYolo0);

          const targets = window.__getTargetPlates ? window.__getTargetPlates() : new Set();
          const yoloDetections = yoloRawDets.map(d => {
            const m = window.__matchPlate(d.text, targets);
            return {
              text: d.text,
              score: Math.round(d.score * 100) / 100,
              minConf: Math.round(d.minConf * 100) / 100,
              isMatch: m.isMatch,
              matchedPlate: m.matchedPlate
            };
          });

          // 2. Direct PaddleOCR on fitted full-frame canvas (1080px)
          const paddleRes = await window.__lprEngine.predictDirectPaddle(fitted.canvas);
          const paddleDurMs = paddleRes.duration;
          const paddleItems = (paddleRes.results && paddleRes.results[0] && paddleRes.results[0].items) || [];
          const paddleDetections = paddleItems.map(item => {
            const m = window.__matchPlate(item.text, targets);
            return {
              text: item.text,
              score: Math.round(item.score * 100) / 100,
              isMatch: m.isMatch,
              matchedPlate: m.matchedPlate
            };
          });

          return {
            imgWidth: fitted.width,
            imgHeight: fitted.height,
            yoloDurMs,
            yoloDetections,
            paddleDurMs,
            paddleTotalItems: paddleItems.length,
            paddleDetections
          };
        })()`
      });

      if (evalRes.exceptionDetails) {
        console.error('Eval error:', evalRes.exceptionDetails);
        continue;
      }

      const data = evalRes.result.value;
      const groundTruth = item.groundTruth;

      const yoloMatchedPlates = [...new Set(data.yoloDetections.filter(d => d.isMatch).map(d => d.matchedPlate))];
      const yoloCorrect = groundTruth.filter(gt => yoloMatchedPlates.includes(gt));

      const paddleMatchedPlates = [...new Set(data.paddleDetections.filter(d => d.isMatch).map(d => d.matchedPlate))];
      const paddleCorrect = groundTruth.filter(gt => paddleMatchedPlates.includes(gt));

      console.log(`  YOLO Pipeline:   ${data.yoloDurMs}ms | Detected: ${data.yoloDetections.length} | Matches: [${yoloCorrect.join(', ')}]`);
      console.log(`  Direct Paddle:   ${data.paddleDurMs}ms | Extracted: ${data.paddleTotalItems} text boxes | Matches: [${paddleCorrect.join(', ')}]\n`);

      report.push({
        name: item.name,
        path: item.path,
        groundTruth,
        yolo: {
          durationMs: data.yoloDurMs,
          numPlates: data.yoloDetections.length,
          matches: yoloCorrect,
          allDetections: data.yoloDetections
        },
        directPaddle: {
          durationMs: data.paddleDurMs,
          totalTextBoxes: data.paddleTotalItems,
          matches: paddleCorrect,
          matchedPlates: paddleMatchedPlates,
          allExtracted: data.paddleDetections
        }
      });
    }

    const outPath = path.resolve(__dirname, '../public/samples/results_paddle_direct.json');
    fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log('======================================================');
    console.log(`Results written to ${outPath}`);
    console.log('======================================================\n');

    const totalYoloMs = report.reduce((sum, r) => sum + r.yolo.durationMs, 0);
    const totalPaddleMs = report.reduce((sum, r) => sum + r.directPaddle.durationMs, 0);
    const totalYoloMatches = report.reduce((sum, r) => sum + r.yolo.matches.length, 0);
    const totalPaddleMatches = report.reduce((sum, r) => sum + r.directPaddle.matches.length, 0);
    const totalGT = report.reduce((sum, r) => sum + r.groundTruth.length, 0);

    console.log(`Pipeline Comparison Summary:`);
    console.log(`  Average Latency:`);
    console.log(`    - YOLO Pipeline: ${Math.round(totalYoloMs / report.length)}ms`);
    console.log(`    - Direct Paddle: ${Math.round(totalPaddleMs / report.length)}ms (${(totalPaddleMs / totalYoloMs).toFixed(1)}x slower)`);
    console.log(`  Accuracy (Ground Truth Plates Matched):`);
    console.log(`    - YOLO Pipeline: ${totalYoloMatches} / ${totalGT} (${Math.round(totalYoloMatches / totalGT * 100)}%)`);
    console.log(`    - Direct Paddle: ${totalPaddleMatches} / ${totalGT} (${Math.round(totalPaddleMatches / totalGT * 100)}%)`);

    chrome.kill();
    process.exit(0);
  } catch (err) {
    console.error('Audit failed:', err);
    if (chrome) chrome.kill();
    process.exit(1);
  }
})();
