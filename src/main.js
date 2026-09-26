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
const btnResumeCamera = document.getElementById('btnResumeCamera');
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

// Static Image Simulation Zoom & Pan Elements
const staticZoomControls = document.getElementById('staticZoomControls');
const btnStaticPanLeft = document.getElementById('btnStaticPanLeft');
const btnStaticPanUp = document.getElementById('btnStaticPanUp');
const btnStaticPanCenter = document.getElementById('btnStaticPanCenter');
const btnStaticPanDown = document.getElementById('btnStaticPanDown');
const btnStaticPanRight = document.getElementById('btnStaticPanRight');
const btnStaticZoomOut = document.getElementById('btnStaticZoomOut');
const btnStaticZoomIn = document.getElementById('btnStaticZoomIn');
const staticZoomLevelText = document.getElementById('staticZoomLevelText');
const staticZoomPresets = document.querySelectorAll('.zoom-preset-btn');

let currentStaticSourceImg = null;
let staticZoom = 1.0;
let staticPanX = 0.0;
let staticPanY = 0.0;

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
export const isIOS = false && typeof navigator !== 'undefined' && (
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
const DEFAULT_CAMERA_ZOOM = 1.5;
let activeVideoTrack = null;
let zoomCapabilities = null;
let currentZoom = DEFAULT_CAMERA_ZOOM;
let pinchStartDistance = 0;
let pinchStartZoom = DEFAULT_CAMERA_ZOOM;
let zoomBadgeTimeout = null;

if (typeof window !== 'undefined') {
  window.__lprEngine = engine;
  window.__fitFrameToMax1080 = fitFrameToMax1080;
  window.__matchPlate = matchPlate;
  window.__getTargetPlates = () => targetPlates;
  window.__DEFAULT_CAMERA_ZOOM = DEFAULT_CAMERA_ZOOM;
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
  window.__getStaticZoom = () => staticZoom;
  window.__setStaticZoom = (z) => setStaticZoom(z);
  window.__getStaticPan = () => ({ x: staticPanX, y: staticPanY });
  window.__setStaticPan = (x, y) => setStaticPan(x, y);
  window.__isIOS = isIOS;
  window.__checkIsIOS = checkIsIOS;
  window.__setIsIOS = (v) => { mockIsIOS = v; };
  window.__triggerManualDetection = () => triggerManualDetection();
  window.__isCameraFrozen = () => isCameraFrozen;
  window.__getFrozenFittedFrame = () => frozenFittedFrame;
  window.__setFrozenFittedFrame = (f) => { frozenFittedFrame = f; };
  window.__unfreezeCameraFeed = (play) => unfreezeCameraFeed(play);
  window.__handleCameraTap = (x, y) => handleViewportTap(x, y);
  window.__handleViewportTap = (x, y) => handleViewportTap(x, y);
  window.__getLastTapDetection = () => lastTapDetection;
  window.__getLastTapCropBoxes = () => lastTapCropBoxes;
  window.__getStaticFittedFrame = () => staticFittedFrame;
  window.__setStaticFittedFrame = (f) => { staticFittedFrame = f; };
  window.__setIsStreaming = (v) => { isStreaming = v; };
  window.__setActiveMode = (v) => { activeMode = v; };
  window.__loadStaticImage = (url) => loadStaticImage(url);
}
let isStreaming = false;
let mediaStream = null;
let isProcessingFrame = false;
let animationFrameId = null;
let lastProcessedTime = 0;
let frameCount = 0;
let fpsLastTime = performance.now();
let activeMode = 'idle'; // 'camera', 'static', 'idle'

// Camera freeze & tap-to-OCR state
let isCameraFrozen = false;
let frozenFittedFrame = null;
let staticFittedFrame = null;
let isTapOcrRunning = false;
let lastTapPoint = null;
let lastTapDetection = null;
let lastTapCropBoxes = null;

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
async function applyCameraZoom(zoomLevel, showBadge = true) {
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

  if (showBadge) {
    showZoomBadge(rounded);
  }
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

  if (staticZoomControls) {
    staticZoomControls.style.display = 'none';
  }
  currentStaticSourceImg = null;

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
        video: { ...videoConstraints, zoom: DEFAULT_CAMERA_ZOOM },
        audio: false
      });
    } catch (e) {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { ...videoConstraints, zoom: true },
          audio: false
        });
      } catch (e2) {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false
        });
      }
    }

    // Inspect native camera hardware zoom capabilities
    activeVideoTrack = mediaStream.getVideoTracks()[0] || null;
    if (activeVideoTrack && typeof activeVideoTrack.getCapabilities === 'function') {
      const caps = activeVideoTrack.getCapabilities();
      if (caps && 'zoom' in caps) {
        zoomCapabilities = caps.zoom;
        await applyCameraZoom(DEFAULT_CAMERA_ZOOM, false);
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
    isCameraFrozen = false;
    frozenFittedFrame = null;
    lastTapPoint = null;
    lastTapDetection = null;
    lastTapCropBoxes = null;
    if (btnResumeCamera) btnResumeCamera.style.display = 'none';
    if (streamBadge) streamBadge.classList.remove('badge-frozen');

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
  if (isCameraFrozen) {
    unfreezeCameraFeed(false);
  }
  lastTapCropBoxes = null;

  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  isStreaming = false;
  isProcessingFrame = false;

  activeVideoTrack = null;
  zoomCapabilities = null;
  currentZoom = DEFAULT_CAMERA_ZOOM;
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
  if (isCameraFrozen) return;

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
  if (isProcessingFrame || isCameraFrozen) return;
  isProcessingFrame = true;

  const t0 = performance.now();

  try {
    const source = activeMode === 'camera' ? videoElement : staticImageElement;
    
    // Fit pictures to max 1080px in width or height
    const fitted = fitFrameToMax1080(source);
    if (!fitted || isCameraFrozen) {
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

    // Render bounding boxes with Green Tick or Red Cross (unless camera feed was frozen by a user tap)
    if (!isCameraFrozen) {
      renderBoundingBoxes(detections, fitted, source);
    }
  } catch (err) {
    console.error('Frame processing error:', err);
  } finally {
    isProcessingFrame = false;
    const now = performance.now();
    const latency = Math.round(now - t0);

    // If zero bounding boxes have been found in the last 3s,
    // sleep 1000ms after each frame to avoid burning CPU when inactive.
    // Otherwise, sleep 2x detection latency (33% compute / 67% rest),
    // with a minimum sleep of 200ms enforced for all devices.
    const isIdle = (now - lastDetectionFoundTime) >= 3000;
    const baseSleep = isIdle ? 1000 : (latency * 2);
    const sleepMs = Math.max(baseSleep, 200);
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
 * Unfreezes the camera feed and resumes normal continuous recognition.
 */
function unfreezeCameraFeed(playVideo = true) {
  if (!isCameraFrozen) return;

  isCameraFrozen = false;
  frozenFittedFrame = null;
  lastTapPoint = null;
  lastTapDetection = null;
  lastTapCropBoxes = null;

  if (btnResumeCamera) {
    btnResumeCamera.style.display = 'none';
  }
  if (checkIsIOS() && btnTapToDetect) {
    btnTapToDetect.style.display = 'flex';
  }
  if (streamBadgeText) {
    streamBadgeText.textContent = checkIsIOS() ? 'TAP TO DETECT' : 'LIVE RECOGNIZING';
  }
  if (streamBadge) {
    streamBadge.classList.remove('badge-frozen');
  }

  if (playVideo && activeMode === 'camera' && isStreaming && videoElement) {
    videoElement.play().catch((err) => {
      console.warn('Video resume play warning:', err);
    });
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = checkIsIOS() ? 'Camera Ready • Tap to Detect' : 'Live Streaming & Recognizing';
    clearOverlay();
    requestRecognitionLoop();
  }
}

/**
 * Normalizes poly point representation ({x, y} or [x, y]) to [x, y].
 */
function normalizePolyPoint(pt) {
  if (!pt) return [0, 0];
  const x = pt.x !== undefined ? pt.x : (pt[0] !== undefined ? pt[0] : 0);
  const y = pt.y !== undefined ? pt.y : (pt[1] !== undefined ? pt[1] : 0);
  return [x, y];
}

/**
 * Draws visual tap indicator ring and crosshairs at the user tap position.
 */
function drawTapMarker(ctx, screenX, screenY, success = true) {
  ctx.save();
  const color = success ? '#38bdf8' : '#f59e0b';
  const fillColor = success ? 'rgba(56, 189, 248, 0.22)' : 'rgba(245, 158, 11, 0.22)';

  ctx.strokeStyle = color;
  ctx.fillStyle = fillColor;
  ctx.lineWidth = 2.2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;

  // Outer target ring
  ctx.beginPath();
  ctx.arc(screenX, screenY, 20, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Inner center dot
  ctx.beginPath();
  ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  // Crosshairs
  ctx.beginPath();
  ctx.moveTo(screenX - 28, screenY);
  ctx.lineTo(screenX - 10, screenY);
  ctx.moveTo(screenX + 10, screenY);
  ctx.lineTo(screenX + 28, screenY);
  ctx.moveTo(screenX, screenY - 28);
  ctx.lineTo(screenX - 10, screenY);
  ctx.moveTo(screenX + 10, screenY);
  ctx.lineTo(screenX + 28, screenY);
  ctx.stroke();

  ctx.restore();
}

/**
 * Draws immediate visual feedback at tap location while OCR inference is in progress,
 * including the multi-scale candidate crop bounding boxes and the target ripple.
 */
function drawTapFeedbackOverlay(screenX, screenY, sampleBoxes, scaleX, scaleY, renderX, renderY) {
  const cW = viewportContainer.clientWidth;
  const cH = viewportContainer.clientHeight;
  if (!cW || !cH) return;
  overlayCanvas.width = cW;
  overlayCanvas.height = cH;
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, cW, cH);
  if (sampleBoxes && scaleX && scaleY) {
    drawTapCropBoxes(ctx, sampleBoxes, scaleX, scaleY, renderX, renderY);
  }
  drawTapMarker(ctx, screenX, screenY, true);
}

/**
 * Handles user tap on camera feed or static image:
 * - Camera mode:
 *   a) Freezes camera feed on initial tap (videoElement.pause(), reveals Resume button at bottom of canvas).
 *   b) Performs Paddle-only OCR around the site of the tap across multi-scale candidate crops.
 *   c) Renders output bounding box and match status.
 *   Subsequent taps do not unfreeze feed, but re-run Paddle OCR at the new tap site.
 * - Static mode:
 *   Performs step (b) and (c) WITHOUT showing the "Resume camera feed" button!
 */
async function handleViewportTap(clientX, clientY) {
  if (activeMode !== 'camera' && activeMode !== 'static') return;
  if (isTapOcrRunning) return;

  const isCamera = activeMode === 'camera';

  // 1. Camera mode: Freeze camera feed on initial tap and reveal Resume button at bottom
  if (isCamera) {
    if (!isStreaming) return;

    if (!isCameraFrozen) {
      isCameraFrozen = true;
      try {
        videoElement.pause();
      } catch (_) {}

      if (!frozenFittedFrame) {
        frozenFittedFrame = fitFrameToMax1080(videoElement);
      }

      if (btnResumeCamera) {
        btnResumeCamera.style.display = 'flex';
      }
      if (btnTapToDetect) {
        btnTapToDetect.style.display = 'none';
      }
      if (streamBadgeText) {
        streamBadgeText.textContent = 'FEED FROZEN • TAP TO OCR';
      }
      if (streamBadge) {
        streamBadge.classList.add('badge-frozen');
      }
    }

    if (!frozenFittedFrame) {
      frozenFittedFrame = fitFrameToMax1080(videoElement);
      if (!frozenFittedFrame) return;
    }
  }

  // 2. Select target fitted frame and source element
  let targetFittedFrame = null;
  let source = null;

  if (isCamera) {
    targetFittedFrame = frozenFittedFrame;
    source = videoElement;
  } else {
    // Static image mode: obtain fitted frame, do NOT show resume button
    targetFittedFrame = fitFrameToMax1080(staticImageElement);
    staticFittedFrame = targetFittedFrame;
    source = staticImageElement;
  }

  if (!targetFittedFrame || !source) return;

  isTapOcrRunning = true;

  // 3. Map screen tap to container and fitted canvas coordinates
  const containerRect = viewportContainer.getBoundingClientRect();
  const cW = containerRect.width;
  const cH = containerRect.height;
  if (!cW || !cH) {
    isTapOcrRunning = false;
    return;
  }

  const clickX = clientX - containerRect.left;
  const clickY = clientY - containerRect.top;

  const sourceW = source.videoWidth || source.naturalWidth || source.width || targetFittedFrame.origWidth || 1;
  const sourceH = source.videoHeight || source.naturalHeight || source.height || targetFittedFrame.origHeight || 1;

  const scale = Math.min(cW / sourceW, cH / sourceH);
  const renderW = sourceW * scale;
  const renderH = sourceH * scale;
  const renderX = (cW - renderW) / 2;
  const renderY = (cH - renderH) / 2;

  // Clamp screen tap to the rendered bounds
  const tapScreenX = Math.max(renderX, Math.min(renderX + renderW, clickX));
  const tapScreenY = Math.max(renderY, Math.min(renderY + renderH, clickY));

  // Map to 1080p fitted canvas coordinates
  const scaleFitX = targetFittedFrame.width / renderW;
  const scaleFitY = targetFittedFrame.height / renderH;
  const tapCanvasX = (tapScreenX - renderX) * scaleFitX;
  const tapCanvasY = (tapScreenY - renderY) * scaleFitY;

  lastTapPoint = { screenX: tapScreenX, screenY: tapScreenY, canvasX: tapCanvasX, canvasY: tapCanvasY };

  const scaleX = renderW / targetFittedFrame.width;
  const scaleY = renderH / targetFittedFrame.height;

  // 4. Multi-scale candidate bounding box crops centered around the tap site:
  // Samples single-line oblong aspects (compact, standard, large, wide) and 2-line stacked plates
  const CROP_SIZES = [
    { w: 220, h: 80, name: 'compact_1line' },
    { w: 340, h: 120, name: 'standard_1line' },
    { w: 460, h: 160, name: 'large_1line' },
    { w: 260, h: 180, name: 'standard_2line' },
    { w: 380, h: 250, name: 'large_2line' },
    { w: 560, h: 220, name: 'wide_context' }
  ];

  const sampleBoxes = CROP_SIZES.map((cropSpec) => {
    const cropW = Math.min(targetFittedFrame.width, cropSpec.w);
    const cropH = Math.min(targetFittedFrame.height, cropSpec.h);
    let cropX = Math.round(tapCanvasX - cropW / 2);
    let cropY = Math.round(tapCanvasY - cropH / 2);
    cropX = Math.max(0, Math.min(targetFittedFrame.width - cropW, cropX));
    cropY = Math.max(0, Math.min(targetFittedFrame.height - cropH, cropY));
    return {
      x: cropX,
      y: cropY,
      w: cropW,
      h: cropH,
      name: cropSpec.name
    };
  });

  lastTapCropBoxes = sampleBoxes;

  // Draw tap ripple/marker and grey crop boxes immediately for responsive user feedback
  drawTapFeedbackOverlay(tapScreenX, tapScreenY, sampleBoxes, scaleX, scaleY, renderX, renderY);

  engineStatusBadge.className = 'status-badge active';
  engineStatusText.textContent = 'Paddle-only OCR at tap site...';

  const t0 = performance.now();

  try {
    const rawCandidates = [];

    for (const box of sampleBoxes) {
      const cropW = box.w;
      const cropH = box.h;
      const cropX = box.x;
      const cropY = box.y;
      const cropSpecName = box.name;

      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = cropW;
      cropCanvas.height = cropH;
      const cropCtx = cropCanvas.getContext('2d');
      cropCtx.drawImage(targetFittedFrame.canvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

      // Perform Paddle-only OCR (Web Worker)
      const directRes = await engine.predictDirectPaddle(cropCanvas);
      const items = (directRes && directRes.results && directRes.results[0] && directRes.results[0].items) || [];
      if (!items.length) continue;

      const validItems = items.filter(it => it.text && it.text.trim().length > 0);
      if (!validItems.length) continue;

      // Sort items top-to-bottom, then left-to-right
      validItems.sort((a, b) => {
        const aPoly = (a.poly || []).map(normalizePolyPoint);
        const bPoly = (b.poly || []).map(normalizePolyPoint);
        const aY = aPoly.length >= 4 ? (aPoly[0][1] + aPoly[2][1]) / 2 : (aPoly[0] ? aPoly[0][1] : 0);
        const bY = bPoly.length >= 4 ? (bPoly[0][1] + bPoly[2][1]) / 2 : (bPoly[0] ? bPoly[0][1] : 0);
        const aH = aPoly.length >= 4 ? Math.abs(aPoly[2][1] - aPoly[0][1]) : 12;
        const bH = bPoly.length >= 4 ? Math.abs(bPoly[2][1] - bPoly[0][1]) : 12;
        const minH = Math.min(aH, bH);
        if (Math.abs(aY - bY) > minH * 0.45) {
          return aY - bY;
        }
        const aX = aPoly[0] ? aPoly[0][0] : 0;
        const bX = bPoly[0] ? bPoly[0][0] : 0;
        return aX - bX;
      });

      // Candidate 1: Multi-line combined (if multiple lines detected in the crop)
      if (validItems.length > 1) {
        const combinedText = validItems.map(it => it.text).join('').trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
        if (combinedText.length >= 3) {
          const polys = validItems.map(it => (it.poly || []).map(normalizePolyPoint));
          const firstPoly = polys[0];
          const lastPoly = polys[polys.length - 1];
          if (firstPoly.length === 4 && lastPoly.length === 4) {
            const minX = Math.min(...polys.flatMap(p => [p[0][0], p[3][0]]));
            const maxX = Math.max(...polys.flatMap(p => [p[1][0], p[2][0]]));
            const rawQuad = [
              [cropX + minX, cropY + firstPoly[0][1]],
              [cropX + maxX, cropY + firstPoly[1][1]],
              [cropX + maxX, cropY + lastPoly[2][1]],
              [cropX + minX, cropY + lastPoly[3][1]]
            ];
            const avgScore = validItems.reduce((s, it) => s + (it.score || 0.8), 0) / validItems.length;
            addCandidate(combinedText, rawQuad, avgScore, cropSpecName);
          }
        }
      }

      // Candidate 2+: Individual text items
      for (const it of validItems) {
        const text = it.text.trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
        const normPoly = (it.poly || []).map(normalizePolyPoint);
        if (text.length >= 3 && normPoly.length === 4) {
          const rawQuad = normPoly.map(([px, py]) => [cropX + px, cropY + py]);
          addCandidate(text, rawQuad, it.score || 0.8, cropSpecName);
        }
      }
    }

    function addCandidate(plateText, rawQuad, conf, cropName) {
      const qcx = (rawQuad[0][0] + rawQuad[1][0] + rawQuad[2][0] + rawQuad[3][0]) / 4;
      const qcy = (rawQuad[0][1] + rawQuad[1][1] + rawQuad[2][1] + rawQuad[3][1]) / 4;
      const distFromTap = Math.hypot(qcx - tapCanvasX, qcy - tapCanvasY);

      // Expand quad slightly by 15% for aesthetic bounding box framing
      const expandedQuad = rawQuad.map(([px, py]) => [
        qcx + (px - qcx) * 1.15,
        qcy + (py - qcy) * 1.20
      ]);

      const matchRes = matchPlate(plateText, targetPlates);
      // Pure proximity and OCR confidence ranking (no target list match bias, no plate length bias)
      const rankScore = (conf * 200) - (distFromTap * 1.5);

      rawCandidates.push({
        text: plateText,
        quad: expandedQuad,
        score: conf,
        minConf: conf,
        matchRes,
        distFromTap,
        rankScore,
        cropName,
        qcx,
        qcy
      });
    }

    const t1 = performance.now();
    const duration = Math.round(t1 - t0);
    scanLatencyText.textContent = `${duration} ms (Tap OCR)`;

    if (rawCandidates.length > 0) {
      // Group by plate text and pick candidate with best rankScore
      const candidateMap = new Map();
      for (const cand of rawCandidates) {
        const existing = candidateMap.get(cand.text);
        if (!existing || cand.rankScore > existing.rankScore) {
          candidateMap.set(cand.text, cand);
        }
      }
      const sortedCandidates = Array.from(candidateMap.values())
        .sort((a, b) => b.rankScore - a.rankScore);

      const best = sortedCandidates[0];
      lastTapDetection = best;

      // Render the output with tick (✓) or cross (✗)
      renderBoundingBoxes([best], targetFittedFrame, source);

      // Draw tap marker on overlay canvas
      const ctx = overlayCanvas.getContext('2d');
      drawTapMarker(ctx, tapScreenX, tapScreenY, true);

      const isMatch = best.matchRes.isMatch;
      engineStatusBadge.className = isMatch ? 'status-badge ready' : 'status-badge active';
      engineStatusText.textContent = `Tap OCR: ${best.text} (${Math.round(best.score * 100)}%) • ${isMatch ? 'Match ✓' : 'Not in list ✗'}`;
    } else {
      // No text detected across all crops: render grey candidate crop boxes and tap marker
      renderBoundingBoxes([], targetFittedFrame, source);
      const ctx = overlayCanvas.getContext('2d');
      drawTapMarker(ctx, tapScreenX, tapScreenY, false);
      engineStatusBadge.className = 'status-badge active';
      engineStatusText.textContent = 'No plate text found at tap site • Tap directly on license plate';
    }
  } catch (err) {
    console.error('Tap Paddle OCR error:', err);
    engineStatusBadge.className = 'status-badge active';
    engineStatusText.textContent = 'Tap OCR error, try again';
  } finally {
    isTapOcrRunning = false;
  }
}

const handleCameraTap = handleViewportTap;

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

  const hasDets = Array.isArray(detections) && detections.length > 0;
  const hasYoloDets = detections && Array.isArray(detections.yoloDets) && detections.yoloDets.length > 0;
  const hasTapCropBoxes = lastTapCropBoxes && lastTapCropBoxes.length > 0;
  if (!hasDets && !hasYoloDets && !hasTapCropBoxes) return;

  // Scale factors from the 1080px fitted canvas coordinates to the displayed overlay
  const scaleX = renderW / fitted.width;
  const scaleY = renderH / fitted.height;

  // 1. Render grey bounding boxes around YOLO-detected boundaries (underneath Paddle boundaries)
  const renderedYoloKeys = new Set();
  const yoloItems = [];

  if (hasDets) {
    for (const det of detections) {
      if (det.yoloQuad) {
        const key = `${Math.round(det.yoloQuad[0][0])}_${Math.round(det.yoloQuad[0][1])}`;
        if (!renderedYoloKeys.has(key)) {
          renderedYoloKeys.add(key);
          yoloItems.push({ quad: det.yoloQuad, score: det.yoloScore || det.score });
        }
      }
    }
  }

  if (hasYoloDets) {
    for (const yd of detections.yoloDets) {
      if (yd.quad) {
        const key = `${Math.round(yd.quad[0][0])}_${Math.round(yd.quad[0][1])}`;
        if (!renderedYoloKeys.has(key)) {
          renderedYoloKeys.add(key);
          yoloItems.push({ quad: yd.quad, score: yd.score });
        }
      }
    }
  }

  for (const item of yoloItems) {
    drawYoloBoundingBox(ctx, item.quad, item.score, scaleX, scaleY, renderX, renderY);
  }

  // 1b. Render grey candidate crop bounding boxes of various sizes (underneath Paddle boundaries)
  if (hasTapCropBoxes) {
    drawTapCropBoxes(ctx, lastTapCropBoxes, scaleX, scaleY, renderX, renderY);
  }

  // 2. Render Paddle boundaries (Green for match, Red for not in list) on top of the grey YOLO & crop boxes
  if (!hasDets) return;

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
 * Draws a subtle grey bounding box representing the raw YOLO detector candidate boundaries.
 * Drawn underneath the colored Paddle OCR boundaries for visual alignment and debugging.
 */
function drawYoloBoundingBox(ctx, rawQuad, score, scaleX, scaleY, renderX, renderY) {
  if (!rawQuad || rawQuad.length !== 4) return;
  const quad = rawQuad.map(([qx, qy]) => [
    renderX + qx * scaleX,
    renderY + qy * scaleY
  ]);

  ctx.save();
  // Sleek grey styling (slate-400) with dashed outline
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.85)';
  ctx.lineWidth = 2.0;
  ctx.setLineDash([5, 4]);
  ctx.fillStyle = 'rgba(100, 116, 139, 0.12)';
  ctx.shadowColor = 'rgba(15, 23, 42, 0.6)';
  ctx.shadowBlur = 4;

  ctx.beginPath();
  ctx.moveTo(quad[0][0], quad[0][1]);
  for (let i = 1; i < quad.length; i++) {
    ctx.lineTo(quad[i][0], quad[i][1]);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw small "YOLO xx%" label badge at the bottom-left of the box
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxY = Math.max(...ys);
  const scorePercent = score ? Math.round(score * 100) : 0;
  const tagText = scorePercent > 0 ? `YOLO ${scorePercent}%` : 'YOLO';

  ctx.font = '600 10px "JetBrains Mono", monospace';
  const tagTextW = ctx.measureText(tagText).width;
  const tagPaddingX = 6;
  const tagW = tagTextW + tagPaddingX * 2;
  const tagH = 16;
  const tagX = Math.max(4, minX);
  const tagY = maxY + 3;

  // Draw tag pill background
  ctx.shadowBlur = 0;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
  ctx.lineWidth = 1;
  roundRect(ctx, tagX, tagY, tagW, tagH, 3);
  ctx.fill();
  ctx.stroke();

  // Draw tag text
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(tagText, tagX + tagPaddingX, tagY + 11.5);
  ctx.restore();
}

/**
 * Draws grey bounding boxes representing the multi-scale candidate crops sampled around the tap site.
 * Drawn underneath the colored Paddle OCR boundaries for visual debugging.
 */
function drawTapCropBoxes(ctx, boxes, scaleX, scaleY, renderX, renderY) {
  if (!boxes || !boxes.length) return;

  ctx.save();
  // Sort from largest to smallest area so inner smaller boxes sit cleanly on top
  const sortedBoxes = [...boxes].sort((a, b) => (b.w * b.h) - (a.w * a.h));

  for (const box of sortedBoxes) {
    const boxX = renderX + box.x * scaleX;
    const boxY = renderY + box.y * scaleY;
    const boxW = box.w * scaleX;
    const boxH = box.h * scaleY;

    // Sleek grey dashed styling (slate-400)
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.75)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.fillStyle = 'rgba(148, 163, 184, 0.03)';
    ctx.shadowColor = 'rgba(15, 23, 42, 0.5)';
    ctx.shadowBlur = 3;

    ctx.beginPath();
    roundRect(ctx, boxX, boxY, boxW, boxH, 4);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    // Small dimension tag at the top-left of each box: e.g. "220×80"
    const tagText = `${box.w}×${box.h}`;
    ctx.font = '600 9px "JetBrains Mono", monospace';
    const tagTextW = ctx.measureText(tagText).width;
    const tagPaddingX = 5;
    const tagW = tagTextW + tagPaddingX * 2;
    const tagH = 15;
    const tagX = Math.max(renderX + 2, boxX + 3);
    const tagY = Math.max(renderY + 2, boxY + 3);

    ctx.shadowBlur = 0;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
    ctx.lineWidth = 1;
    roundRect(ctx, tagX, tagY, tagW, tagH, 3);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(tagText, tagX + tagPaddingX, tagY + tagH / 2);
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
 * Updates UI buttons and indicators for static zoom and pan.
 */
function updateStaticZoomUi() {
  if (staticZoomLevelText) {
    staticZoomLevelText.textContent = `${staticZoom.toFixed(1)}×`;
  }
  if (btnStaticZoomOut) {
    btnStaticZoomOut.disabled = staticZoom <= 1.0;
  }
  if (btnStaticZoomIn) {
    btnStaticZoomIn.disabled = staticZoom >= 4.0;
  }
  if (btnStaticPanLeft) {
    btnStaticPanLeft.disabled = staticZoom <= 1.0 || staticPanX <= -0.98;
  }
  if (btnStaticPanRight) {
    btnStaticPanRight.disabled = staticZoom <= 1.0 || staticPanX >= 0.98;
  }
  if (btnStaticPanUp) {
    btnStaticPanUp.disabled = staticZoom <= 1.0 || staticPanY <= -0.98;
  }
  if (btnStaticPanDown) {
    btnStaticPanDown.disabled = staticZoom <= 1.0 || staticPanY >= 0.98;
  }
  if (btnStaticPanCenter) {
    btnStaticPanCenter.disabled = staticZoom <= 1.0 || (Math.abs(staticPanX) < 0.02 && Math.abs(staticPanY) < 0.02);
  }
  staticZoomPresets.forEach((btn) => {
    const z = parseFloat(btn.dataset.zoom);
    btn.classList.toggle('active', Math.abs(z - staticZoom) < 0.05);
  });
}

/**
 * Renders the static image at the specified zoom level and pan position,
 * fitting the cropped window into a max 1080px canvas (simulating optical/sensor zoom).
 */
function renderStaticZoomedFrame() {
  if (!currentStaticSourceImg) return;
  const origW = currentStaticSourceImg.naturalWidth || currentStaticSourceImg.width;
  const origH = currentStaticSourceImg.naturalHeight || currentStaticSourceImg.height;
  if (!origW || !origH) return;

  // Window size in source coordinates at current zoom level
  const cropW = origW / staticZoom;
  const cropH = origH / staticZoom;

  // Maximum pan travel in source pixels
  const maxPanX = (origW - cropW) / 2;
  const maxPanY = (origH - cropH) / 2;
  const panPxX = staticPanX * maxPanX;
  const panPxY = staticPanY * maxPanY;

  // Source crop rectangle
  const sx = Math.max(0, Math.min(origW - cropW, (origW - cropW) / 2 + panPxX));
  const sy = Math.max(0, Math.min(origH - cropH, (origH - cropH) / 2 + panPxY));

  // Target canvas dimensions (max 1080px)
  const maxDim = 1080;
  let targetW = Math.round(cropW);
  let targetH = Math.round(cropH);
  if (targetW > maxDim || targetH > maxDim) {
    if (targetW >= targetH) {
      targetH = Math.round((targetH * maxDim) / targetW);
      targetW = maxDim;
    } else {
      targetW = Math.round((targetW * maxDim) / targetH);
      targetH = maxDim;
    }
  }

  const c = document.createElement('canvas');
  c.width = targetW;
  c.height = targetH;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(currentStaticSourceImg, sx, sy, cropW, cropH, 0, 0, targetW, targetH);

  updateStaticZoomUi();

  engineStatusBadge.className = 'status-badge active';
  engineStatusText.textContent = `Analyzing ${staticZoom > 1.0 ? staticZoom.toFixed(1) + '×' : '1080px'} Frame...`;

  staticImageElement.onload = async () => {
    await processCurrentFrame();
    engineStatusBadge.className = 'status-badge ready';
    engineStatusText.textContent = `Image Analyzed (${staticZoom.toFixed(1)}× Zoom)`;
  };
  staticImageElement.src = c.toDataURL('image/jpeg', 0.95);
}

function setStaticZoom(newZoom) {
  const clamped = Math.max(1.0, Math.min(4.0, Math.round(newZoom * 10) / 10));
  if (clamped === staticZoom) return;
  staticZoom = clamped;
  if (staticZoom <= 1.0) {
    staticPanX = 0.0;
    staticPanY = 0.0;
  }
  lastTapCropBoxes = null;
  lastTapDetection = null;
  lastTapPoint = null;
  renderStaticZoomedFrame();
}

function setStaticPan(newPanX, newPanY) {
  if (staticZoom <= 1.0) return;
  const clampedX = Math.max(-1.0, Math.min(1.0, Math.round(newPanX * 100) / 100));
  const clampedY = Math.max(-1.0, Math.min(1.0, Math.round((newPanY !== undefined ? newPanY : staticPanY) * 100) / 100));
  if (clampedX === staticPanX && clampedY === staticPanY) return;
  staticPanX = clampedX;
  staticPanY = clampedY;
  lastTapCropBoxes = null;
  lastTapDetection = null;
  lastTapPoint = null;
  renderStaticZoomedFrame();
}

/**
 * Loads a static test image (e.g. Car 1, Car 2, or uploaded photo)
 * and renders the downscaled 1080px content with zoom/pan simulation.
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

  // Reset zoom & pan to default on new image load
  staticZoom = 1.0;
  staticPanX = 0.0;
  staticPanY = 0.0;
  lastTapCropBoxes = null;
  lastTapDetection = null;
  lastTapPoint = null;
  if (staticZoomControls) staticZoomControls.style.display = 'flex';
  updateStaticZoomUi();

  engineStatusBadge.className = 'status-badge active';
  engineStatusText.textContent = 'Loading Image...';

  const tempImg = new Image();
  tempImg.crossOrigin = 'anonymous';
  tempImg.onload = () => {
    currentStaticSourceImg = tempImg;
    renderStaticZoomedFrame();
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
  if (activeMode === 'camera' && isCameraFrozen && lastTapDetection && frozenFittedFrame) {
    // If camera feed is frozen, immediately re-evaluate match status of current detection
    lastTapDetection.matchRes = matchPlate(lastTapDetection.text, targetPlates);
    renderBoundingBoxes([lastTapDetection], frozenFittedFrame, videoElement);
    if (lastTapPoint) {
      const ctx = overlayCanvas.getContext('2d');
      drawTapMarker(ctx, lastTapPoint.screenX, lastTapPoint.screenY, true);
    }
  } else if (activeMode === 'static' && lastTapDetection && staticFittedFrame) {
    // If static image has an active tap detection, re-evaluate match status
    lastTapDetection.matchRes = matchPlate(lastTapDetection.text, targetPlates);
    renderBoundingBoxes([lastTapDetection], staticFittedFrame, staticImageElement);
    if (lastTapPoint) {
      const ctx = overlayCanvas.getContext('2d');
      drawTapMarker(ctx, lastTapPoint.screenX, lastTapPoint.screenY, true);
    }
  } else if (activeMode !== 'idle') {
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
  'SNF5763B',
  'SMK8800G',
  'SLE5647H',
  'SNY9977A',
  'SMU5178Z',
  'SJS561D',
  'SND33T',
  'SLK5331L',
  'SMK8837M',
  'SMZ7912G',
  'SNX6148B',
  'SLC6318L',
  'SMZ6023D',
  'SNA4496E',
  'SNC5016R',
  'SMK4973D',
  'SBR8808R',
  'SNN5085H'
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

let lastViewportTapTime = 0;
function handleViewportTouchOrClick(clientX, clientY) {
  const now = performance.now();
  if (now - lastViewportTapTime < 350) return;
  lastViewportTapTime = now;
  handleViewportTap(clientX, clientY);
}

viewportContainer.addEventListener('touchend', (e) => {
  if (e.touches.length === 0 && pinchStartDistance === 0) {
    const elapsed = performance.now() - touchStartTime;
    const changed = e.changedTouches[0];
    if (changed) {
      const dist = Math.hypot(changed.clientX - touchStartX, changed.clientY - touchStartY);
      if (elapsed < 350 && dist < 15) {
        // Clean single tap on screen (ignoring control buttons, selects, or static zoom controls)
        if (e.target.closest('button, select, input, label, .static-zoom-controls')) return;
        if (activeMode === 'camera' || activeMode === 'static') {
          e.preventDefault();
          handleViewportTouchOrClick(changed.clientX, changed.clientY);
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

// Also support desktop mouse click on viewport for both camera and static image mode
viewportContainer.addEventListener('click', (e) => {
  if (e.target.closest('button, select, input, label, .static-zoom-controls')) return;
  if (activeMode === 'camera' || activeMode === 'static') {
    handleViewportTouchOrClick(e.clientX, e.clientY);
  }
});

// Resume Camera Feed button (visible when feed is frozen by user tap)
if (btnResumeCamera) {
  btnResumeCamera.addEventListener('click', (e) => {
    e.stopPropagation();
    unfreezeCameraFeed(true);
  });
}

// Manual shutter button for iOS (runs tap OCR at center of viewport)
if (btnTapToDetect) {
  btnTapToDetect.addEventListener('click', (e) => {
    e.stopPropagation();
    const rect = viewportContainer.getBoundingClientRect();
    handleViewportCameraTap(rect.left + rect.width / 2, rect.top + rect.height / 2);
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

// Static Zoom and Pan Button Listeners
btnStaticPanLeft?.addEventListener('click', (e) => {
  e.stopPropagation();
  setStaticPan(staticPanX - 0.25, staticPanY);
});

btnStaticPanRight?.addEventListener('click', (e) => {
  e.stopPropagation();
  setStaticPan(staticPanX + 0.25, staticPanY);
});

btnStaticPanUp?.addEventListener('click', (e) => {
  e.stopPropagation();
  setStaticPan(staticPanX, staticPanY - 0.25);
});

btnStaticPanDown?.addEventListener('click', (e) => {
  e.stopPropagation();
  setStaticPan(staticPanX, staticPanY + 0.25);
});

btnStaticPanCenter?.addEventListener('click', (e) => {
  e.stopPropagation();
  setStaticPan(0.0, 0.0);
});

btnStaticZoomIn?.addEventListener('click', (e) => {
  e.stopPropagation();
  setStaticZoom(staticZoom + 0.5);
});

btnStaticZoomOut?.addEventListener('click', (e) => {
  e.stopPropagation();
  setStaticZoom(staticZoom - 0.5);
});

staticZoomPresets.forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const z = parseFloat(btn.dataset.zoom);
    if (!isNaN(z)) {
      setStaticZoom(z);
    }
  });
});

// Keyboard shortcuts for static zoom and pan
window.addEventListener('keydown', (e) => {
  if (activeMode !== 'static') return;
  if (e.target?.closest?.('textarea, input, select')) return;

  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    setStaticPan(staticPanX - 0.20, staticPanY);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    setStaticPan(staticPanX + 0.20, staticPanY);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    setStaticPan(staticPanX, staticPanY - 0.20);
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    setStaticPan(staticPanX, staticPanY + 0.20);
  } else if (e.key === '+' || e.key === '=') {
    e.preventDefault();
    setStaticZoom(staticZoom + 0.5);
  } else if (e.key === '-' || e.key === '_') {
    e.preventDefault();
    setStaticZoom(staticZoom - 0.5);
  } else if (e.key === '0' || e.key === 'Home') {
    e.preventDefault();
    setStaticPan(0.0, 0.0);
    setStaticZoom(1.0);
  }
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
    if (staticZoomControls) staticZoomControls.style.display = 'none';
    currentStaticSourceImg = null;
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
