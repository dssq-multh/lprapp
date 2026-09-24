/**
 * Client proxy that communicates with the dedicated Web Worker (lpr.worker.js)
 * running ONNX Runtime Web and PaddleOCR.
 *
 * Running in a Web Worker ensures the browser main thread remains 100% responsive
 * (smooth 60 FPS video, zero UI stutter, and immunity to iOS WebKit watchdog kills).
 */
export class LprWorkerClient {
  constructor(opts = {}) {
    this.opts = opts;
    this.reqSeq = 0;
    this.pending = new Map();
    this.statusCallback = null;
    this.isReady = false;
    this.enableGpu = opts.enableGpu || false;
    this.scoreThreshold = opts.scoreThreshold;

    this.worker = new Worker(
      new URL('./lpr.worker.js', import.meta.url),
      { type: 'module' }
    );

    this.worker.onmessage = (event) => {
      const data = event.data;
      if (!data) return;

      if (data.type === 'status') {
        if (this.statusCallback) {
          this.statusCallback(data.message);
        }
        return;
      }

      const { id, type, error, ...rest } = data;
      if (id && this.pending.has(id)) {
        const { resolve, reject } = this.pending.get(id);
        this.pending.delete(id);
        if (error) {
          reject(new Error(error));
        } else {
          resolve(rest);
        }
      }
    };

    this.worker.onerror = (err) => {
      console.error('LprWorker error:', err);
      for (const [id, { reject }] of this.pending) {
        reject(err);
      }
      this.pending.clear();
    };
  }

  async init(onStatus) {
    this.statusCallback = onStatus;
    const id = ++this.reqSeq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        resolve: (data) => {
          this.isReady = true;
          if (data && typeof data.enableGpu === 'boolean') {
            this.enableGpu = data.enableGpu;
          }
          resolve(data);
        },
        reject
      });
      this.worker.postMessage({
        type: 'init',
        id,
        payload: {
          base: this.opts.base,
          scoreThreshold: this.opts.scoreThreshold,
          enableGpu: this.enableGpu
        }
      });
    });
  }

  async setEnableGpu(enableGpu, onStatus) {
    this.statusCallback = onStatus;
    this.enableGpu = Boolean(enableGpu);
    const id = ++this.reqSeq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        resolve: (data) => {
          if (data.result && typeof data.result.enabled === 'boolean') {
            this.enableGpu = data.result.enabled;
          }
          resolve(data.result);
        },
        reject
      });
      this.worker.postMessage({
        type: 'setEnableGpu',
        id,
        payload: { enableGpu: this.enableGpu }
      });
    });
  }

  async readAll(canvas, opts = {}) {
    if (!this.isReady) {
      throw new Error('LprWorkerClient is not ready yet');
    }

    const width = canvas.width;
    const height = canvas.height;
    const id = ++this.reqSeq;

    return new Promise(async (resolve, reject) => {
      this.pending.set(id, {
        resolve: (data) => resolve(data.detections || []),
        reject
      });

      try {
        if (typeof createImageBitmap === 'function') {
          const bitmap = await createImageBitmap(canvas);
          this.worker.postMessage(
            {
              type: 'readAll',
              id,
              payload: {
                bitmap,
                width,
                height,
                opts
              }
            },
            [bitmap]
          );
        } else {
          const ctx = canvas.getContext('2d');
          const imageData = ctx.getImageData(0, 0, width, height);
          this.worker.postMessage(
            {
              type: 'readAll',
              id,
              payload: {
                imageData,
                width,
                height,
                opts
              }
            },
            [imageData.data.buffer]
          );
        }
      } catch (err) {
        this.pending.delete(id);
        reject(err);
      }
    });
  }

  terminate() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }
}
