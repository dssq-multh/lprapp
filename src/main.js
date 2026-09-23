import { LprEngine, fitFrameToMax1080 } from './lpr/engine.js';

// DOM Elements
const engineStatusBadge = document.getElementById('engineStatusBadge');
const engineStatusText = document.getElementById('engineStatusText');
const platesTextarea = document.getElementById('platesTextarea');
const plateCountBadge = document.getElementById('plateCountBadge');
const matchStatsBadge = document.getElementById('matchStatsBadge');
const btnRecognize = document.getElementById('btnRecognize');
const btnRecognizeText = document.getElementById('btnRecognizeText');
const btnSamplePlates = document.getElementById('btnSamplePlates');
const btnClearPlates = document.getElementById('btnClearPlates');

const cameraSelect = document.getElementById('cameraSelect');
const videoElement = document.getElementById('videoElement');
const staticImageElement = document.getElementById('staticImageElement');
const overlayCanvas = document.getElementById('overlayCanvas');
const idleOverlay = document.getElementById('idleOverlay');
const btnIdleStart = document.getElementById('btnIdleStart');
const btnIdleSample = document.getElementById('btnIdleSample');
const streamBadge = document.getElementById('streamBadge');
const scanFpsText = document.getElementById('scanFpsText');
const scanLatencyText = document.getElementById('scanLatencyText');
const scanResText = document.getElementById('scanResText');

const sampleSelect = document.getElementById('sampleSelect');
const btnPrevSample = document.getElementById('btnPrevSample');
const btnNextSample = document.getElementById('btnNextSample');
const fileInput = document.getElementById('fileInput');
const detectionsList = document.getElementById('detectionsList');
const btnClearFeed = document.getElementById('btnClearFeed');
const audioChimeToggle = document.getElementById('audioChimeToggle');

// State
let targetPlates = new Set();
const engine = new LprEngine({ base: (import.meta.env.BASE_URL ?? '/') + 'lpr/' });
if (typeof window !== 'undefined') {
  window.__lprEngine = engine;
  window.__fitFrameToMax1080 = fitFrameToMax1080;
  window.__matchPlate = matchPlate;
  window.__getTargetPlates = () => targetPlates;
}
let isStreaming = false;
let mediaStream = null;
let isProcessingFrame = false;
let animationFrameId = null;
let lastSpottedPlates = new Map(); // Plate -> { timestamp, match, count }
let lastProcessedTime = 0;
let frameCount = 0;
let fpsLastTime = performance.now();
let activeMode = 'idle'; // 'camera', 'static', 'idle'

// Web Audio API context for chime
let audioCtx = null;
function playMatchChime() {
  if (!audioChimeToggle.checked) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    // Arpeggio chime: 587Hz (D5) -> 880Hz (A5)
    osc.frequency.setValueAtTime(587.33, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(880.00, audioCtx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.18, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.36);
  } catch (e) {
    console.warn('Audio chime error:', e);
  }
}

/**
 * Normalizes a plate string for reliable matching:
 * Uppercase, stripped of spaces, dashes, dots, and non-alphanumeric chars.
 */
function normalizePlate(str) {
  if (!str) return '';
  return str.toUpperCase().replace(/[^A-Z0-9]/g, '');
}

/**
 * Canonical plate for matching:
 * The ONLY permitted substitutions:
 * I <=> 1
 * O <=> 0
 * 6 and 8 must NEVER match (6<!>8).
 */
function canonicalPlate(str) {
  return normalizePlate(str)
    .replace(/I/g, '1')
    .replace(/O/g, '0');
}

/**
 * Matches detected plate against target list:
 * 1. Exact match
 * 2. ONLY permitted substitutions (I<=>1 and O<=>0)
 * Note: 6 and 8 are strictly separate (6<!>8).
 */
function matchPlate(rawDetected, targetSet) {
  const norm = normalizePlate(rawDetected);
  if (!norm || norm.length < 2) return { isMatch: false, matchedPlate: null };

  // 1. Exact match
  if (targetSet.has(norm)) {
    return { isMatch: true, matchedPlate: norm };
  }

  // 2. The ONLY permitted substitutions: I<=>1 and O<=>0
  const canonDet = canonicalPlate(norm);
  for (const target of targetSet) {
    if (canonicalPlate(target) === canonDet) {
      return { isMatch: true, matchedPlate: target };
    }
  }

  return { isMatch: false, matchedPlate: null };
}

/**
 * Parses the textarea contents into the target plates Set.
 */
function updateTargetPlates() {
  const text = platesTextarea.value;
  const lines = text.split(/[\n,;]+/);
  targetPlates.clear();

  for (const line of lines) {
    const cleaned = normalizePlate(line);
    if (cleaned.length >= 2) {
      targetPlates.add(cleaned);
    }
  }

  const count = targetPlates.size;
  plateCountBadge.textContent = `${count} ${count === 1 ? 'plate' : 'plates'}`;
}

/**
 * Enumerates video devices and populates cameraSelect with forward-facing camera prioritized.
 */
async function setupCameraDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    cameraSelect.innerHTML = '<option value="">No Camera API available</option>';
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === 'videoinput');

    if (videoDevices.length === 0) {
      cameraSelect.innerHTML = '<option value="default">Default Camera</option>';
      return;
    }

    cameraSelect.innerHTML = '';

    // Check for forward/rear/environment cameras (dashcam/car/outward view)
    let forwardFacingId = null;
    videoDevices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      const label = device.label || `Camera ${index + 1}`;
      const lower = label.toLowerCase();

      // Check if camera label indicates forward-facing / environment
      const isForward = lower.includes('back') ||
                        lower.includes('rear') ||
                        lower.includes('environment') ||
                        lower.includes('outward') ||
                        lower.includes('world');

      if (isForward && !forwardFacingId) {
        forwardFacingId = device.deviceId;
        option.textContent = `⭐ ${label} (Forward Facing)`;
      } else {
        option.textContent = label;
      }

      cameraSelect.appendChild(option);
    });

    // Add generic environment option at top
    const envOption = document.createElement('option');
    envOption.value = 'prefer-environment';
    envOption.textContent = '🌟 Auto Forward-Facing (Recommended)';
    cameraSelect.insertBefore(envOption, cameraSelect.firstChild);

    if (forwardFacingId) {
      cameraSelect.value = forwardFacingId;
    } else {
      cameraSelect.value = 'prefer-environment';
    }
  } catch (err) {
    console.warn('Could not enumerate cameras:', err);
  }
}

/**
 * Starts camera streaming with forward-facing preference.
 */
async function startCamera() {
  if (isStreaming) return;

  try {
    engineStatusBadge.className = 'status-badge active';
    engineStatusText.textContent = 'Opening Camera...';

    const selectedDeviceId = cameraSelect.value;
    let videoConstraints = {
      width: { ideal: 1920, max: 1920 },
      height: { ideal: 1080, max: 1080 }
    };

    if (selectedDeviceId === 'prefer-environment') {
      videoConstraints.facingMode = { ideal: 'environment' };
    } else if (selectedDeviceId) {
      videoConstraints.deviceId = { exact: selectedDeviceId };
    } else {
      videoConstraints.facingMode = { ideal: 'environment' };
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false
    });

    videoElement.srcObject = mediaStream;
    staticImageElement.style.display = 'none';
    videoElement.style.display = 'block';

    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play();
        resolve();
      };
    });

    isStreaming = true;
    activeMode = 'camera';
    idleOverlay.style.display = 'none';
    streamBadge.style.display = 'flex';

    btnRecognize.classList.add('is-recognizing');
    btnRecognizeText.textContent = 'Stop Streaming';

    engineStatusBadge.className = 'status-badge active';
    engineStatusText.textContent = 'Live Streaming & Recognizing';

    // Refresh devices once permissions are granted so labels appear
    setupCameraDevices();

    // Start recognition loop
    requestRecognitionLoop();
  } catch (err) {
    console.error('Failed to open camera:', err);
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = 'Camera error';
    alert(`Could not start camera: ${err.message || err.name}. You can test with the sample car images or upload an image!`);
    stopCamera();
  }
}

/**
 * Stops camera streaming.
 */
function stopCamera() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  isStreaming = false;
  isProcessingFrame = false;

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  videoElement.srcObject = null;
  btnRecognize.classList.remove('is-recognizing');
  btnRecognizeText.textContent = 'Recognize!';
  streamBadge.style.display = 'none';

  if (activeMode === 'camera') {
    idleOverlay.style.display = 'flex';
    clearOverlay();
  }

  engineStatusBadge.className = 'status-badge ready';
  engineStatusText.textContent = 'Ready';
  scanFpsText.textContent = '-- FPS';
  scanLatencyText.textContent = '-- ms';
}

/**
 * Continuous frame recognition loop.
 * Runs non-blocking: processes one frame at a time without delaying video display.
 */
function requestRecognitionLoop() {
  if (!isStreaming || activeMode !== 'camera') return;

  if (!isProcessingFrame && videoElement.readyState >= 2) {
    processCurrentFrame();
  }

  animationFrameId = requestAnimationFrame(requestRecognitionLoop);
}

/**
 * Fits picture to max 1080px in width or height, runs LPR Wasm, and draws bounding boxes.
 */
async function processCurrentFrame() {
  if (isProcessingFrame) return;
  isProcessingFrame = true;

  const t0 = performance.now();

  try {
    const source = activeMode === 'camera' ? videoElement : staticImageElement;
    
    // Fit pictures to max 1080px in width or height
    const fitted = fitFrameToMax1080(source);
    if (!fitted) {
      isProcessingFrame = false;
      return;
    }

    scanResText.textContent = `${fitted.width}×${fitted.height} (max 1080px)`;

    // Run OpenALPR Wasm inference
    const detections = await engine.readAll(fitted.canvas, { tile: true });

    const t1 = performance.now();
    const latency = Math.round(t1 - t0);
    scanLatencyText.textContent = `${latency} ms`;

    // Calculate FPS
    frameCount++;
    if (t1 - fpsLastTime >= 1000) {
      const currentFps = Math.round((frameCount * 1000) / (t1 - fpsLastTime));
      scanFpsText.textContent = `${currentFps} FPS`;
      frameCount = 0;
      fpsLastTime = t1;
    }

    // Render bounding boxes with Green Tick or Red Cross
    renderBoundingBoxes(detections, fitted, source);
    updateDetectionsFeed(detections);
  } catch (err) {
    console.error('Frame processing error:', err);
  } finally {
    isProcessingFrame = false;
  }
}

/**
 * Clears the overlay canvas.
 */
function clearOverlay() {
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

/**
 * Draws bounding boxes around identified vehicle license plates:
 * - Green bounding boxes with a tick (✓) if the license plate is in the list.
 * - Red bounding boxes with a cross (✗) if the license plate is not in the list.
 */
function renderBoundingBoxes(detections, fitted, source) {
  const container = document.getElementById('viewportContainer');
  const containerRect = container.getBoundingClientRect();
  const cW = containerRect.width;
  const cH = containerRect.height;
  if (!cW || !cH) return;

  // Match overlay canvas size to displayed source element
  const sourceW = source.videoWidth || source.naturalWidth || source.width || 1;
  const sourceH = source.videoHeight || source.naturalHeight || source.height || 1;

  // Since #videoElement and #staticImageElement have width: 100%, height: 100%, object-fit: contain,
  // the rendered content fits within (cW, cH) centered along the unconstrained axis.
  const scale = Math.min(cW / sourceW, cH / sourceH);
  const renderW = sourceW * scale;
  const renderH = sourceH * scale;
  const renderX = (cW - renderW) / 2;
  const renderY = (cH - renderH) / 2;

  overlayCanvas.width = cW;
  overlayCanvas.height = cH;

  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!detections || detections.length === 0) return;

  // Scale factors from the 1080px fitted canvas coordinates to the displayed overlay
  const scaleX = renderW / fitted.width;
  const scaleY = renderH / fitted.height;

  let matchedCountInFrame = 0;

  for (const det of detections) {
    const rawText = det.text || '';
    const matchRes = matchPlate(rawText, targetPlates);
    const isMatch = matchRes.isMatch;
    const displayPlate = rawText || matchRes.matchedPlate || 'PLATE';

    if (isMatch) {
      matchedCountInFrame++;
    }

    // Map quad coordinates to overlay space
    const quad = det.quad.map(([qx, qy]) => [
      renderX + qx * scaleX,
      renderY + qy * scaleY
    ]);

    // Box bounds
    const xs = quad.map((p) => p[0]);
    const ys = quad.map((p) => p[1]);
    const left = Math.min(...xs);
    const top = Math.min(...ys);
    const right = Math.max(...xs);
    const bottom = Math.max(...ys);
    const width = right - left;
    const height = bottom - top;

    // Styling according to requirement:
    // Green with tick (✓) if in list, Red with cross (✗) if not in list
    const strokeColor = isMatch ? '#10b981' : '#ef4444';
    const fillColor = isMatch ? 'rgba(16, 185, 129, 0.16)' : 'rgba(239, 68, 68, 0.16)';
    const glowColor = isMatch ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)';
    const symbol = isMatch ? '✓' : '✗';
    const statusText = isMatch ? 'MATCH' : 'NOT IN LIST';

    ctx.save();

    // 1. Draw glowing oriented bounding box (quadrilateral)
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 12;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 3.5;
    ctx.fillStyle = fillColor;

    ctx.beginPath();
    ctx.moveTo(quad[0][0], quad[0][1]);
    for (let i = 1; i < quad.length; i++) {
      ctx.lineTo(quad[i][0], quad[i][1]);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // 2. Stylish oriented corner accents pointing along the true perspective edges
    drawOrientedCorners(ctx, quad, strokeColor);

    // 3. Draw Top Pill Badge with Tick/Cross, Plate Text, and Confidence
    ctx.shadowBlur = 0;
    const confPercent = Math.round((det.minConf || 0) * 100);
    const label = `${symbol} ${displayPlate} (${confPercent}%) - ${statusText}`;
    
    ctx.font = 'bold 13px "JetBrains Mono", monospace';
    const textWidth = ctx.measureText(label).width;
    const badgePaddingX = 10;
    const badgePaddingY = 6;
    const badgeW = textWidth + badgePaddingX * 2;
    const badgeH = 26;

    // Centered above the top edge of the oriented quad
    const topMidX = (quad[0][0] + quad[1][0]) / 2;
    const topMinY = Math.min(quad[0][1], quad[1][1]);
    const badgeX = Math.max(10, Math.min(overlayCanvas.width - badgeW - 10, topMidX - badgeW / 2));
    const badgeY = Math.max(10, topMinY - badgeH - 8);

    // Badge background
    ctx.fillStyle = strokeColor;
    roundRect(ctx, badgeX, badgeY, badgeW, badgeH, 6);
    ctx.fill();

    // Badge text
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, badgeX + badgePaddingX, badgeY + badgeH / 2);

    // 4. Prominent Tick or Cross Badge Icon next to the right edge of the oriented quad
    const iconRadius = 14;
    const rightMidX = (quad[1][0] + quad[2][0]) / 2;
    const rightMidY = (quad[1][1] + quad[2][1]) / 2;
    const iconX = Math.min(overlayCanvas.width - iconRadius - 8, rightMidX + iconRadius + 6);
    const iconY = rightMidY;

    ctx.fillStyle = strokeColor;
    ctx.beginPath();
    ctx.arc(iconX, iconY, iconRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = 'bold 15px "JetBrains Mono", sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(symbol, iconX, iconY);

    ctx.restore();
  }

  // Play chime once if a new match was discovered
  if (matchedCountInFrame > 0) {
    playMatchChime();
  }
}

/**
 * Draws oriented corner accents along the true polygon edges.
 * Accurately tracks the perspective tilt and rotation of the plate.
 */
function drawOrientedCorners(ctx, quad, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;

  const N = quad.length;
  for (let i = 0; i < N; i++) {
    const curr = quad[i];
    const prev = quad[(i - 1 + N) % N];
    const next = quad[(i + 1) % N];

    // Unit vector towards prev
    const dPrevX = prev[0] - curr[0];
    const dPrevY = prev[1] - curr[1];
    const lenPrev = Math.hypot(dPrevX, dPrevY) || 1;
    const arm1 = Math.min(18, lenPrev * 0.35);

    // Unit vector towards next
    const dNextX = next[0] - curr[0];
    const dNextY = next[1] - curr[1];
    const lenNext = Math.hypot(dNextX, dNextY) || 1;
    const arm2 = Math.min(18, lenNext * 0.35);

    // Draw arm towards prev
    ctx.beginPath();
    ctx.moveTo(curr[0], curr[1]);
    ctx.lineTo(curr[0] + (dPrevX / lenPrev) * arm1, curr[1] + (dPrevY / lenPrev) * arm1);
    ctx.stroke();

    // Draw arm towards next
    ctx.beginPath();
    ctx.moveTo(curr[0], curr[1]);
    ctx.lineTo(curr[0] + (dNextX / lenNext) * arm2, curr[1] + (dNextY / lenNext) * arm2);
    ctx.stroke();

    // Small vertex dot
    ctx.beginPath();
    ctx.arc(curr[0], curr[1], 2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.restore();
}

/**
 * Helper to draw rounded rectangle.
 */
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/**
 * Updates the Detections Feed at the bottom of the viewport.
 */
function updateDetectionsFeed(detections) {
  if (!detections || detections.length === 0) return;

  const now = Date.now();
  let updated = false;

  for (const det of detections) {
    const raw = det.text.trim();
    const matchRes = matchPlate(raw, targetPlates);
    const isMatch = matchRes.isMatch;
    const norm = isMatch ? matchRes.matchedPlate : normalizePlate(raw);

    const prev = lastSpottedPlates.get(norm);
    // Rate limit feed cards to once per 2.5 seconds per unique plate
    if (!prev || (now - prev.timestamp) > 2500) {
      lastSpottedPlates.set(norm, {
        raw,
        norm,
        isMatch,
        conf: det.minConf,
        timestamp: now,
        count: (prev ? prev.count : 0) + 1
      });
      updated = true;
    }
  }

  if (updated) {
    renderFeedList();
  }
}

/**
 * Renders the feed item cards.
 */
function renderFeedList() {
  const sorted = Array.from(lastSpottedPlates.values())
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 15);

  if (sorted.length === 0) {
    detectionsList.innerHTML = '<div class="empty-feed-text">No plates spotted yet. Start streaming or try a sample image!</div>';
    matchStatsBadge.textContent = '0 matches found';
    return;
  }

  const matchTotal = sorted.filter((s) => s.isMatch).length;
  matchStatsBadge.textContent = `${matchTotal} ${matchTotal === 1 ? 'match' : 'matches'} found`;

  detectionsList.innerHTML = sorted.map((item) => {
    const symbol = item.isMatch ? '✓' : '✗';
    const tagClass = item.isMatch ? 'match' : 'unlisted';
    const tagText = item.isMatch ? 'Target Matched' : 'Not In List';
    const timeStr = new Date(item.timestamp).toLocaleTimeString();
    const confStr = `${Math.round(item.conf * 100)}% conf`;

    return `
      <div class="feed-item ${tagClass}">
        <div class="feed-item-badge">${symbol}</div>
        <div class="feed-item-info">
          <span class="feed-item-plate">${item.raw}</span>
          <span class="feed-item-meta">${tagText} • ${confStr} • ${timeStr}</span>
        </div>
      </div>
    `;
  }).join('');
}

/**
 * Loads a static test image (e.g. Car 1, Car 2, or uploaded photo)
 * and renders the downscaled 1080px content so visual display matches what AI sees.
 */
async function loadStaticImage(url) {
  if (isStreaming) {
    stopCamera();
  }

  activeMode = 'static';
  idleOverlay.style.display = 'none';
  videoElement.style.display = 'none';
  staticImageElement.style.display = 'block';

  engineStatusBadge.className = 'status-badge active';
  engineStatusText.textContent = 'Downsampling to 1080px...';

  const tempImg = new Image();
  tempImg.crossOrigin = 'anonymous';
  tempImg.onload = async () => {
    // Shrink full-resolution image to max 1080px
    const fitted = fitFrameToMax1080(tempImg);
    if (!fitted) return;

    // Render the 1080px content visually so the user sees the exact resolution AI processes
    staticImageElement.onload = async () => {
      engineStatusText.textContent = 'Analyzing 1080px Frame...';
      await processCurrentFrame();
      engineStatusBadge.className = 'status-badge ready';
      engineStatusText.textContent = 'Image Analyzed (1080px View)';
    };
    staticImageElement.src = fitted.canvas.toDataURL('image/jpeg', 0.95);
  };
  tempImg.src = import.meta.env.BASE_URL.replace(/\/$/, '') + url;
}

// Event Listeners
btnRecognize.addEventListener('click', () => {
  if (isStreaming) {
    stopCamera();
  } else {
    startCamera();
  }
});

btnIdleStart.addEventListener('click', () => {
  startCamera();
});

btnIdleSample.addEventListener('click', () => {
  loadStaticImage('/samples/car_cal8942.jpg');
});

cameraSelect.addEventListener('change', () => {
  if (isStreaming) {
    stopCamera();
    startCamera();
  }
});

platesTextarea.addEventListener('input', () => {
  updateTargetPlates();
  // Re-render current bounding boxes if image or video is active
  if (activeMode !== 'idle') {
    processCurrentFrame();
  }
});

const SAMPLE_TARGET_PLATES = [
  'CAL8942',
  'B391KLT',
  'SDN6618H',
  'SMJ6650C',
  'SBU999J',
  'SML6579R',
  'ES3960A',
  'SLX9361E',
  'SNF9945S',
  'SKG516L',
  'SJV7999M',
  'SLE5647H',
  'SNY9977A',
  'SMU5178Z',
  'SJS561D',
  'SND33T'
].join('\n');

btnSamplePlates.addEventListener('click', () => {
  platesTextarea.value = SAMPLE_TARGET_PLATES;
  updateTargetPlates();
  if (activeMode !== 'idle') {
    processCurrentFrame();
  }
});

btnClearPlates.addEventListener('click', () => {
  platesTextarea.value = '';
  updateTargetPlates();
  if (activeMode !== 'idle') {
    processCurrentFrame();
  }
});

sampleSelect.addEventListener('change', () => {
  const url = sampleSelect.value;
  if (url) {
    loadStaticImage(url);
  }
});

btnPrevSample.addEventListener('click', () => {
  const total = sampleSelect.options.length;
  let idx = sampleSelect.selectedIndex - 1;
  if (idx < 0) idx = total - 1;
  sampleSelect.selectedIndex = idx;
  loadStaticImage(sampleSelect.value);
});

btnNextSample.addEventListener('click', () => {
  const total = sampleSelect.options.length;
  let idx = (sampleSelect.selectedIndex + 1) % total;
  sampleSelect.selectedIndex = idx;
  loadStaticImage(sampleSelect.value);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      loadStaticImage(event.target.result);
    };
    reader.readAsDataURL(file);
  }
});

btnClearFeed.addEventListener('click', () => {
  lastSpottedPlates.clear();
  renderFeedList();
});

window.addEventListener('resize', () => {
  if (activeMode !== 'idle') {
    processCurrentFrame();
  }
});

// Initialization
async function initApp() {
  // Prepopulate sample plates so user gets immediate visual feedback
  platesTextarea.value = SAMPLE_TARGET_PLATES;
  updateTargetPlates();

  await setupCameraDevices();

  try {
    await engine.init((status) => {
      engineStatusText.textContent = status;
    });
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = 'Wasm Models Ready';
  } catch (err) {
    console.error('Failed to initialize engine:', err);
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = 'Wasm Ready (On-Demand)';
  }
}

initApp();
