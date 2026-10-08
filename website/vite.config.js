import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The in-browser demo loads Pyodide from jsDelivr and the latent-gate wheel from PyPI
const CSP = "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self' ws://localhost:5173 https://cdn.jsdelivr.net https://pypi.org https://files.pythonhosted.org;"

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves the site under /latent-gate/; Vercel and local dev serve it at /
  base: process.env.PAGES_BASE || '/',
  server: {
    headers: {
      'Content-Security-Policy': CSP,
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
  },
  define: {
    // Expose Vercel system env vars to the frontend
    'import.meta.env.VITE_VERCEL_ENV': JSON.stringify(process.env.VERCEL_ENV || 'development'),
    'import.meta.env.VITE_VERCEL_URL': JSON.stringify(process.env.VERCEL_URL || 'localhost:5173'),
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          if (id.includes('node_modules')) {
            if (id.includes('react')) return 'vendor'
            if (id.includes('lucide')) return 'lucide'
            return 'vendor'
          }
        }
      }
    }
  }
})