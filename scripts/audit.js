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

const userDataDir = `/tmp/chrome_audit_${Date.now()}`;
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless',
  '--disable-gpu',
  '--ignore-certificate-errors',
  '--allow-insecure-localhost',
  `--user-data-dir=${userDataDir}`,
  '--remote-debugging-port=9222',
  'https://localhost:5173/lprapp/'
]);

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

(async () => {
  let ws;
  try {
    console.log('Connecting to headless Chrome on port 9222...');
    let target = null;
    for (let i = 0; i < 20; i++) {
      await sleep(500);
      try {
        const res = await fetch('http://127.0.0.1:9222/json');
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

    // Populate all ground truth target plates into platesTextarea
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
          const jpegImg = new Image();
          await new Promise((res) => {
            jpegImg.onload = res;
            jpegImg.src = fittedRaw.canvas.toDataURL('image/jpeg', 0.95);
          });
          const fitted = window.__fitFrameToMax1080(jpegImg);
          const t0 = performance.now();
          const rawDets = await window.__lprEngine.readAll(fitted.canvas, { tile: false });
          const durMs = Math.round(performance.now() - t0);

          const targets = window.__getTargetPlates ? window.__getTargetPlates() : new Set();
          const detections = rawDets.map(d => {
            const m = window.__matchPlate(d.text, targets);
            return {
              text: d.text,
              score: Math.round(d.score * 100) / 100,
              minConf: Math.round(d.minConf * 100) / 100,
              box: d.box,
              isMatch: m.isMatch,
              matchedPlate: m.matchedPlate
            };
          });

          return { durMs, detections };
        })()`
      });

      const { durMs, detections } = evalRes.result.value;
      console.log(`done (${durMs}ms, ${detections.length} plates detected)`);

      results.push({
        name: item.name,
        path: item.path,
        groundTruth: item.groundTruth,
        durationMs: durMs,
        detections
      });
    }

    const outputPath = path.resolve(__dirname, '../public/samples/results.json');
    fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
    console.log('\n======================================================');
    console.log('Results successfully saved to:', outputPath);
    console.log('======================================================\n');

    let totalMatches = 0;
    let totalGroundTruth = 0;
    let totalMs = 0;

    for (const res of results) {
      totalGroundTruth += res.groundTruth.length;
      totalMs += res.durationMs;
      console.log(`\nImage: ${res.name} (${res.durationMs}ms)`);
      console.log(`  Ground Truth: [${res.groundTruth.join(', ')}]`);
      if (res.detections.length === 0) {
        console.log('  ALPR: (No plates detected)');
      } else {
        res.detections.forEach((d, idx) => {
          if (d.isMatch) totalMatches++;
          const icon = d.isMatch ? '✓' : '✗';
          console.log(`  #${idx + 1} [${icon}] "${d.text}" (conf: ${d.minConf}, detScore: ${d.score}, match: ${d.matchedPlate || 'none'})`);
        });
      }
    }

    console.log('\n------------------------------------------------------');
    console.log(`Summary:`);
    console.log(`  Total Images Evaluated: ${results.length}`);
    console.log(`  Average Latency: ${Math.round(totalMs / results.length)}ms`);
    console.log(`  Target Plates Matched: ${totalMatches} / ${totalGroundTruth}`);
    console.log('------------------------------------------------------\n');

    chrome.kill();
    process.exit(0);
  } catch (err) {
    console.error('\nError during audit:', err);
    if (chrome) chrome.kill();
    process.exit(1);
  }
})();
