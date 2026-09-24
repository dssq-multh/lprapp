import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';
import basicSsl from '@vitejs/plugin-basic-ssl'


export default defineConfig({
  base: '/lprapp/',
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  plugins: [
    basicSsl(),
    {
      name: 'serve-wasm-and-models',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const rawUrl = req.url ? req.url.split('?')[0] : '';
          const url = rawUrl.replace(/^\/lprapp/, '');
          if (url.startsWith('/ort-wasm/')) {
            const filePath = path.join(process.cwd(), 'public', url);
            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              if (url.endsWith('.wasm')) {
                res.setHeader('Content-Type', 'application/wasm');
              } else if (url.endsWith('.mjs') || url.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
              }
              res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
              res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
              return fs.createReadStream(filePath).pipe(res);
            }
          }
          if (url.startsWith('/models/') && url.endsWith('.tar')) {
            const filePath = path.join(process.cwd(), 'public', url);
            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              res.setHeader('Content-Type', 'application/x-tar');
              res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
              res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
              return fs.createReadStream(filePath).pipe(res);
            }
          }
          next();
        });
      },
    },
  ],
});
