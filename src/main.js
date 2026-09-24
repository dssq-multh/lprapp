import { LprWorkerClient as LprEngine } from './lpr/worker_client.js';
import { fitFrameToMax1080 } from './lpr/engine.js';

// DOM Elements
const engineStatusBadge = document.getElementById('engineStatusBadge');
const engineStatusText = document.getElementById('engineStatusText');
const platesTextarea = document.getElementById('platesTextarea');
const btnRecognize = document.getElementById('btnRecognize');
const btnRecognizeText = document.getElementById('btnRecognizeText');

const cameraSelect = document.getElementById('cameraSelect');
const videoElement = document.getElementById('videoElement');
const staticImageElement = document.getElementById('staticImageElement');
const overlayCanvas = document.getElementById('overlayCanvas');
const viewportContainer = document.getElementById('viewportContainer');
const zoomBadge = document.getElementById('zoomBadge');
const zoomBadgeText = document.getElementById('zoomBadgeText');
const idleOverlay = document.getElementById('idleOverlay');
const btnIdleStart = document.getElementById('btnIdleStart');
const btnIdleSample = document.getElementById('btnIdleSample');
const streamBadge = document.getElementById('streamBadge');
const streamBadgeText = document.getElementById('streamBadgeText');
const btnTapToDetect = document.getElementById('btnTapToDetect');
const btnTapToDetectText = document.getElementById('btnTapToDetectText');
const scanFpsText = document.getElementById('scanFpsText');
const scanLatencyText = document.getElementById('scanLatencyText');
const scanResText = document.getElementById('scanResText');

const sampleSelect = document.getElementById('sampleSelect');
const btnPrevSample = document.getElementById('btnPrevSample');
const btnNextSample = document.getElementById('btnNextSample');
const fileInput = document.getElementById('fileInput');
const btnCloseOverlay = document.getElementById('btnCloseOverlay');
const btnMobileSampleCar = document.getElementById('btnMobileSampleCar');
const mobileFileInput = document.getElementById('mobileFileInput');
const rightPanel = document.querySelector('.right-panel');
const gpuToggle = document.getElementById('gpuToggle');
const gpuToggleToolbar = document.getElementById('gpuToggleToolbar');

// Global Loading Overlay Elements (YOLO + PaddleOCR)
const appLoadingOverlay = document.getElementById('appLoadingOverlay');
const compItemYolo = document.getElementById('compItemYolo');
const compDescYolo = document.getElementById('compDescYolo');
const compItemPaddle = document.getElementById('compItemPaddle');
const compDescPaddle = document.getElementById('compDescPaddle');
const loadingStatusText = document.getElementById('loadingStatusText');
const loadingErrorBox = document.getElementById('loadingErrorBox');
const loadingErrorMsg = document.getElementById('loadingErrorMsg');
const btnDismissLoadingOverlay = document.getElementById('btnDismissLoadingOverlay');

// iOS detection (iPhone, iPod, iPad including iPadOS desktop UA)
export const isIOS = typeof navigator !== 'undefined' && (
  /iPad|iPhone|iPod/.test(navigator.userAgent || '') ||
  (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
);
console.log(`[Device] isIOS: ${isIOS}`, {
  userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : null,
  platform: typeof navigator !== 'undefined' ? navigator.platform : null,
  maxTouchPoints: typeof navigator !== 'undefined' ? navigator.maxTouchPoints : null
});
let mockIsIOS = null;
export function checkIsIOS() {
  if (mockIsIOS !== null) return mockIsIOS;
  return isIOS;
}

// State
const STORAGE_KEY_GPU = 'alpr_enable_gpu';
// Off by default unless explicitly saved as 'true' in localStorage
let isGpuEnabled = typeof localStorage !== 'undefined' && localStorage.getItem(STORAGE_KEY_GPU) === 'true';

let targetPlates = new Set();
const engine = new LprEngine({
  base: (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '') + '/lpr/',
  enableGpu: isGpuEnabled
});
let activeVideoTrack = null;
let zoomCapabilities = null;
let currentZoom = 1.0;
let pinchStartDistance = 0;
let pinchStartZoom = 1.0;
let zoomBadgeTimeout = null;

if (typeof window !== 'undefined') {
  window.__lprEngine = engine;
  window.__fitFrameToMax1080 = fitFrameToMax1080;
  window.__matchPlate = matchPlate;
  window.__getTargetPlates = () => targetPlates;
  window.__getZoomCapabilities = () => zoomCapabilities;
  window.__getCurrentZoom = () => currentZoom;
  window.__applyCameraZoom = (z) => applyCameraZoom(z);
  window.__showZoomBadge = (z) => showZoomBadge(z);
  window.__setMockZoom = (track, caps) => {
    activeVideoTrack = track;
    zoomCapabilities = caps;
  };
  window.__getGpuEnabled = () => isGpuEnabled;
  window.__setGpuEnabled = (v) => onGpuToggleChange(v);
  window.__isIOS = isIOS;
  window.__checkIsIOS = checkIsIOS;
  window.__setIsIOS = (v) => { mockIsIOS = v; };
  window.__triggerManualDetection = () => triggerManualDetection();
}
let isStreaming = false;
let mediaStream = null;
let isProcessingFrame = false;
let animationFrameId = null;
let lastProcessedTime = 0;
let frameCount = 0;
let fpsLastTime = performance.now();
let activeMode = 'idle'; // 'camera', 'static', 'idle'

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
 * Applies native camera zoom using MediaStreamTrack applyConstraints.
 */
async function applyCameraZoom(zoomLevel) {
  if (!activeVideoTrack || !zoomCapabilities) return;
  const min = zoomCapabilities.min ?? 1.0;
  const max = zoomCapabilities.max ?? 1.0;
  const step = zoomCapabilities.step ?? 0.1;

  const clamped = Math.max(min, Math.min(max, zoomLevel));
  const rounded = Math.round(clamped / step) * step;

  currentZoom = rounded;

  try {
    await activeVideoTrack.applyConstraints({
      advanced: [{ zoom: rounded }]
    });
  } catch (err) {
    console.warn('Native camera zoom error:', err);
  }

  showZoomBadge(rounded);
}

function showZoomBadge(val) {
  if (!zoomBadge || !zoomBadgeText) return;
  zoomBadgeText.textContent = `${Number(val).toFixed(1)}×`;
  zoomBadge.style.display = 'flex';
  zoomBadge.style.opacity = '1';

  if (zoomBadgeTimeout) {
    clearTimeout(zoomBadgeTimeout);
  }
  zoomBadgeTimeout = setTimeout(() => {
    zoomBadge.style.opacity = '0';
    setTimeout(() => {
      if (zoomBadge.style.opacity === '0') {
        zoomBadge.style.display = 'none';
      }
    }, 250);
  }, 1200);
}

function hideZoomBadge() {
  if (zoomBadgeTimeout) {
    clearTimeout(zoomBadgeTimeout);
    zoomBadgeTimeout = null;
  }
  if (zoomBadge) {
    zoomBadge.style.display = 'none';
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

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { ...videoConstraints, zoom: true },
        audio: false
      });
    } catch (e) {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false
      });
    }

    // Inspect native camera hardware zoom capabilities
    activeVideoTrack = mediaStream.getVideoTracks()[0] || null;
    if (activeVideoTrack && typeof activeVideoTrack.getCapabilities === 'function') {
      const caps = activeVideoTrack.getCapabilities();
      if (caps && 'zoom' in caps) {
        zoomCapabilities = caps.zoom;
        const settings = activeVideoTrack.getSettings ? activeVideoTrack.getSettings() : {};
        currentZoom = settings.zoom || zoomCapabilities.min || 1.0;
      } else {
        zoomCapabilities = null;
      }
    } else {
      zoomCapabilities = null;
    }

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
    document.body.classList.add('camera-active-mode');
    document.body.classList.remove('static-active-mode');
    idleOverlay.style.display = 'none';
    streamBadge.style.display = 'flex';
    if (rightPanel) rightPanel.classList.add('mobile-active');

    btnRecognize.classList.add('is-recognizing');
    btnRecognizeText.textContent = 'Stop Streaming';

    if (checkIsIOS()) {
      if (streamBadgeText) streamBadgeText.textContent = 'TAP TO DETECT';
      engineStatusBadge.className = 'status-badge ready';
      engineStatusText.textContent = 'Camera Ready • Tap to Detect';
      scanFpsText.textContent = 'Tap to Scan';
      if (btnTapToDetect) {
        btnTapToDetect.style.display = 'flex';
        if (btnTapToDetectText) btnTapToDetectText.textContent = 'Tap to Detect';
      }
    } else {
      if (streamBadgeText) streamBadgeText.textContent = 'LIVE RECOGNIZING';
      engineStatusBadge.className = 'status-badge active';
      engineStatusText.textContent = 'Live Streaming & Recognizing';
      if (btnTapToDetect) btnTapToDetect.style.display = 'none';
    }

    // Refresh devices once permissions are granted so labels appear
    setupCameraDevices();

    // Reset FPS calculation for fresh stream
    frameCount = 0;
    fpsLastTime = performance.now();
    nextAllowedFrameTime = 0;
    lastDetectionFoundTime = performance.now();

    // Start recognition loop (continuous on non-iOS; manual tap on iOS)
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

  activeVideoTrack = null;
  zoomCapabilities = null;
  currentZoom = 1.0;
  hideZoomBadge();

  if (btnTapToDetect) {
    btnTapToDetect.style.display = 'none';
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  videoElement.srcObject = null;
  btnRecognize.classList.remove('is-recognizing');
  btnRecognizeText.textContent = 'Recognize!';
  streamBadge.style.display = 'none';
  if (rightPanel) rightPanel.classList.remove('mobile-active');
  document.body.classList.remove('camera-active-mode', 'static-active-mode');

  if (activeMode === 'camera') {
    activeMode = 'idle';
    idleOverlay.style.display = 'flex';
    clearOverlay();
  }

  frameCount = 0;
  fpsLastTime = performance.now();
  engineStatusBadge.className = 'status-badge ready';
  engineStatusText.textContent = 'Ready';
  scanFpsText.textContent = '-- FPS';
  scanLatencyText.textContent = '-- ms';
}

let nextAllowedFrameTime = 0;
let lastDetectionFoundTime = performance.now();

/**
 * Continuous frame recognition loop.
 * Runs non-blocking: offloaded to Web Worker without delaying video display.
 * Dynamic throttling: Sleeps for 2x detection latency when plates are visible,
 * and sleeps 1000ms if zero bounding boxes were detected in the last 3s.
 */
function requestRecognitionLoop() {
  if (!isStreaming || activeMode !== 'camera') return;

  // On iOS, automated background loop is disabled; detection is strictly user tap-to-trigger
  if (checkIsIOS()) return;

  const now = performance.now();
  if (!isProcessingFrame && videoElement.readyState >= 2 && now >= nextAllowedFrameTime) {
    processCurrentFrame();
  }

  animationFrameId = requestAnimationFrame(requestRecognitionLoop);
}

/**
 * Fits picture to max 1080px in width or height, runs LPR Wasm in Web Worker, and draws bounding boxes.
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

    // Run OpenALPR Wasm inference via Web Worker (tiling disabled)
    const detections = await engine.readAll(fitted.canvas, { tile: false });

    if (detections && detections.length > 0) {
      lastDetectionFoundTime = performance.now();
    }

    const t1 = performance.now();
    const latency = Math.round(t1 - t0);
    scanLatencyText.textContent = `${latency} ms`;

    // Calculate FPS or s/frame if below 1 FPS
    if (checkIsIOS()) {
      scanFpsText.textContent = 'Manual Tap';
    } else if (activeMode === 'camera') {
      frameCount++;
      const elapsed = t1 - fpsLastTime;
      if (elapsed >= 1000) {
        const rawFps = (frameCount * 1000) / elapsed;
        if (rawFps < 1) {
          const secPerFrame = elapsed / (frameCount * 1000);
          scanFpsText.textContent = `${secPerFrame.toFixed(1)} s/frame`;
        } else {
          scanFpsText.textContent = `${Math.round(rawFps)} FPS`;
        }
        frameCount = 0;
        fpsLastTime = t1;
      }
    } else {
      if (latency >= 1000) {
        const secPerFrame = latency / 1000;
        scanFpsText.textContent = `${secPerFrame.toFixed(1)} s/frame`;
      } else {
        scanFpsText.textContent = '-- FPS';
      }
    }

    // Render bounding boxes with Green Tick or Red Cross
    renderBoundingBoxes(detections, fitted, source);
  } catch (err) {
    console.error('Frame processing error:', err);
  } finally {
    isProcessingFrame = false;
    const now = performance.now();
    const latency = Math.round(now - t0);

    // If zero bounding boxes have been found in the last 3s,
    // sleep 1000ms after each frame to avoid burning CPU when inactive.
    // Otherwise, sleep 2x detection latency (33% compute / 67% rest).
    const isIdle = (now - lastDetectionFoundTime) >= 3000;
    const sleepMs = isIdle ? 1000 : (latency * 2);
    nextAllowedFrameTime = now + sleepMs;
  }
}

/**
 * Triggers manual single-shot detection (used on iOS).
 */
async function triggerManualDetection() {
  if (!isStreaming || activeMode !== 'camera') {
    if (!isStreaming) {
      await startCamera();
    }
    return;
  }
  if (isProcessingFrame) {
    console.log('[iOS Tap] Already processing frame, ignoring tap');
    return;
  }
  if (videoElement.readyState < 2) {
    console.log('[iOS Tap] Video not ready yet');
    return;
  }

  console.log('[iOS Tap] Manual detection triggered by user tap');

  // Flash visual feedback
  viewportContainer.classList.remove('shutter-flash');
  void viewportContainer.offsetWidth; // force reflow
  viewportContainer.classList.add('shutter-flash');
  setTimeout(() => {
    viewportContainer.classList.remove('shutter-flash');
  }, 180);

  if (btnTapToDetect) {
    btnTapToDetect.classList.add('is-detecting');
    if (btnTapToDetectText) btnTapToDetectText.textContent = 'Detecting...';
  }
  engineStatusBadge.className = 'status-badge active';
  engineStatusText.textContent = 'Analyzing Frame...';

  try {
    await processCurrentFrame();
  } catch (err) {
    console.error('Manual detection error:', err);
  } finally {
    if (btnTapToDetect) {
      btnTapToDetect.classList.remove('is-detecting');
      if (btnTapToDetectText) btnTapToDetectText.textContent = 'Tap to Detect';
    }
    if (isStreaming && activeMode === 'camera') {
      engineStatusBadge.className = 'status-badge ready';
      engineStatusText.textContent = 'Camera Ready • Tap to Detect';
    }
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

  // Sort detections: reds (non-matches) first, greens (matches) last,
  // so that any overlapping green will overlay a red
  const sortedDetections = detections
    .map((det) => {
      const rawText = det.text || '';
      const matchRes = matchPlate(rawText, targetPlates);
      return { det, rawText, matchRes };
    })
    .sort((a, b) => (a.matchRes.isMatch ? 1 : 0) - (b.matchRes.isMatch ? 1 : 0));

  for (const { det, rawText, matchRes } of sortedDetections) {
    const isMatch = matchRes.isMatch;
    const displayPlate = rawText || matchRes.matchedPlate || 'PLATE';

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
 * Loads a static test image (e.g. Car 1, Car 2, or uploaded photo)
 * and renders the downscaled 1080px content so visual display matches what AI sees.
 */
async function loadStaticImage(url) {
  if (isStreaming) {
    stopCamera();
  }

  activeMode = 'static';
  frameCount = 0;
  fpsLastTime = performance.now();
  document.body.classList.add('static-active-mode');
  document.body.classList.remove('camera-active-mode');
  if (rightPanel) rightPanel.classList.add('mobile-active');
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
  if (url.startsWith('data:') || url.startsWith('blob:') || url.startsWith('http://') || url.startsWith('https://')) {
    tempImg.src = url;
  } else {
    tempImg.src = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '') + (url.startsWith('/') ? url : '/' + url);
  }
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

let touchStartTime = 0;
let touchStartX = 0;
let touchStartY = 0;

// Native camera pinch-to-zoom touch handlers on mobile + tap detection on iOS
viewportContainer.addEventListener('touchstart', (e) => {
  if (e.touches.length === 1) {
    touchStartTime = performance.now();
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
  }
  if (e.touches.length === 2 && activeMode === 'camera' && zoomCapabilities) {
    e.preventDefault();
    const t1 = e.touches[0];
    const t2 = e.touches[1];
    pinchStartDistance = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
    pinchStartZoom = currentZoom;
  }
}, { passive: false });

viewportContainer.addEventListener('touchmove', (e) => {
  if (e.touches.length === 2 && activeMode === 'camera' && zoomCapabilities && pinchStartDistance > 0) {
    e.preventDefault();
    const t1 = e.touches[0];
    const t2 = e.touches[1];
    const currentDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
    if (pinchStartDistance > 5) {
      const scale = currentDist / pinchStartDistance;
      const targetZoom = pinchStartZoom * scale;
      applyCameraZoom(targetZoom);
    }
  }
}, { passive: false });

viewportContainer.addEventListener('touchend', (e) => {
  if (e.touches.length === 0 && pinchStartDistance === 0) {
    const elapsed = performance.now() - touchStartTime;
    const changed = e.changedTouches[0];
    if (changed) {
      const dist = Math.hypot(changed.clientX - touchStartX, changed.clientY - touchStartY);
      if (elapsed < 350 && dist < 15) {
        // Clean single tap on screen
        if (e.target.closest('button, select, input, label')) return;
        if (activeMode === 'camera' && checkIsIOS()) {
          e.preventDefault();
          triggerManualDetection();
        }
      }
    }
  }
  if (e.touches.length < 2) {
    pinchStartDistance = 0;
  }
});

viewportContainer.addEventListener('touchcancel', () => {
  pinchStartDistance = 0;
});

// Also support desktop mouse click on viewport when running in iOS mode
viewportContainer.addEventListener('click', (e) => {
  if (e.target.closest('button, select, input, label')) return;
  if (activeMode === 'camera' && checkIsIOS()) {
    triggerManualDetection();
  }
});

// Manual shutter button for iOS
if (btnTapToDetect) {
  btnTapToDetect.addEventListener('click', (e) => {
    e.stopPropagation();
    triggerManualDetection();
  });
}

// Also support trackpad pinch gesture on desktop/laptop
viewportContainer.addEventListener('wheel', (e) => {
  if (e.ctrlKey && activeMode === 'camera' && zoomCapabilities) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.05 : 0.95;
    applyCameraZoom(currentZoom * factor);
  }
}, { passive: false });

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

btnCloseOverlay?.addEventListener('click', () => {
  if (isStreaming) {
    stopCamera();
  } else {
    activeMode = 'idle';
    document.body.classList.remove('camera-active-mode', 'static-active-mode');
    if (rightPanel) rightPanel.classList.remove('mobile-active');
    staticImageElement.style.display = 'none';
    idleOverlay.style.display = 'flex';
    clearOverlay();
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = 'Ready';
  }
});

btnMobileSampleCar?.addEventListener('click', () => {
  loadStaticImage('/samples/car_cal8942.jpg');
});

mobileFileInput?.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      loadStaticImage(event.target.result);
    };
    reader.readAsDataURL(file);
  }
});

window.addEventListener('resize', () => {
  if (activeMode !== 'idle') {
    processCurrentFrame();
  }
});

function updateGpuUi(enabled) {
  if (gpuToggle) {
    gpuToggle.checked = enabled;
    gpuToggle.closest('.gpu-checkbox-label')?.classList.toggle('is-active', enabled);
  }
  if (gpuToggleToolbar) {
    gpuToggleToolbar.checked = enabled;
    gpuToggleToolbar.closest('.gpu-checkbox-label')?.classList.toggle('is-active', enabled);
  }
}

async function onGpuToggleChange(enabled) {
  isGpuEnabled = Boolean(enabled);
  try {
    localStorage.setItem(STORAGE_KEY_GPU, String(isGpuEnabled));
  } catch (_) {}
  updateGpuUi(isGpuEnabled);

  engineStatusBadge.className = 'status-badge loading';
  engineStatusText.textContent = isGpuEnabled ? 'Switching YOLO to WebGPU...' : 'Switching YOLO to WASM...';

  try {
    const res = await engine.setEnableGpu(isGpuEnabled, (msg) => {
      engineStatusText.textContent = msg;
    });

    if (res && res.reverted) {
      isGpuEnabled = false;
      try { localStorage.setItem(STORAGE_KEY_GPU, 'false'); } catch (_) {}
      updateGpuUi(false);
      engineStatusBadge.className = 'status-badge ready';
      engineStatusText.textContent = 'WebGPU unavailable; on WASM';
      alert('WebGPU is not supported or encountered an initialization error on this browser/GPU. Reverted to WASM.');
    } else {
      engineStatusBadge.className = 'status-badge ready';
      engineStatusText.textContent = `YOLO ready on ${isGpuEnabled ? 'WebGPU' : 'WASM'}`;
    }
  } catch (err) {
    console.error('Failed to change GPU provider:', err);
    isGpuEnabled = false;
    try { localStorage.setItem(STORAGE_KEY_GPU, 'false'); } catch (_) {}
    updateGpuUi(false);
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = 'Ready (WASM)';
  }
}

// Sync initial GPU toggle state and listen for changes
updateGpuUi(isGpuEnabled);
gpuToggle?.addEventListener('change', (e) => onGpuToggleChange(e.target.checked));
gpuToggleToolbar?.addEventListener('change', (e) => onGpuToggleChange(e.target.checked));

// Global Loading Overlay Controller
function updateLoadingStatus(status) {
  if (!loadingStatusText) return;
  const text = typeof status === 'string' ? status : (status?.text || JSON.stringify(status));
  loadingStatusText.textContent = text;

  const lower = text.toLowerCase();
  if (lower.includes('yolo') || lower.includes('spotter') || lower.includes('detector') || lower.includes('webassembly') || lower.includes('runtime')) {
    if (compItemYolo) compItemYolo.className = 'comp-item loading';
    if (compDescYolo) compDescYolo.textContent = text;
  } else if (lower.includes('paddle') || lower.includes('ocr')) {
    if (compItemYolo) compItemYolo.className = 'comp-item ready';
    if (compDescYolo) compDescYolo.textContent = 'Detector Ready (1 CPU)';
    if (compItemPaddle) compItemPaddle.className = 'comp-item loading';
    if (compDescPaddle) compDescPaddle.textContent = text;
  } else if (lower.includes('ready') || lower.includes('done')) {
    if (compItemYolo) compItemYolo.className = 'comp-item ready';
    if (compDescYolo) compDescYolo.textContent = 'Detector Ready';
    if (compItemPaddle) compItemPaddle.className = 'comp-item ready';
    if (compDescPaddle) compDescPaddle.textContent = 'PP-OCRv6 Engine Ready';
  }
}

function hideLoadingOverlay() {
  if (!appLoadingOverlay) return;
  if (compItemYolo) {
    compItemYolo.className = 'comp-item ready';
    if (compDescYolo) compDescYolo.textContent = 'Detector Ready';
  }
  if (compItemPaddle) {
    compItemPaddle.className = 'comp-item ready';
    if (compDescPaddle) compDescPaddle.textContent = 'PP-OCRv6 Engine Ready';
  }
  if (loadingStatusText) {
    loadingStatusText.textContent = 'All components loaded!';
  }

  // Brief pause so user sees both green checks, then smooth fade-out
  setTimeout(() => {
    appLoadingOverlay.classList.add('hidden');
    setTimeout(() => {
      appLoadingOverlay.style.display = 'none';
    }, 450);
  }, 400);
}

function showLoadingError(errMessage) {
  if (loadingErrorBox && loadingErrorMsg) {
    loadingErrorMsg.textContent = `Initialization note: ${errMessage}`;
    loadingErrorBox.style.display = 'block';
  }
  if (loadingStatusText) {
    loadingStatusText.textContent = 'Engine initialization note';
  }
}

btnDismissLoadingOverlay?.addEventListener('click', () => {
  if (appLoadingOverlay) {
    appLoadingOverlay.classList.add('hidden');
    setTimeout(() => {
      appLoadingOverlay.style.display = 'none';
    }, 450);
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
      engineStatusText.textContent = typeof status === 'string' ? status : (status?.text || '');
      updateLoadingStatus(status);
    });
    if (typeof engine.enableGpu === 'boolean' && engine.enableGpu !== isGpuEnabled) {
      isGpuEnabled = engine.enableGpu;
      try { localStorage.setItem(STORAGE_KEY_GPU, String(isGpuEnabled)); } catch (_) {}
      updateGpuUi(isGpuEnabled);
    }
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = `Models Ready (${isGpuEnabled ? 'WebGPU' : 'WASM'})`;
    btnRecognize.disabled = false;

    hideLoadingOverlay();
  } catch (err) {
    console.error('Failed to initialize engine:', err);
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = 'Wasm Ready (On-Demand)';
    btnRecognize.disabled = false;
    showLoadingError(err.message || String(err));
  }
}

initApp();
