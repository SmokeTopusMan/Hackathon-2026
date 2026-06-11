import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // proxy API calls to the Flask backend (api/server.py) during development
  server: {
    proxy: {
      '/api': 'http://localhost:5000',
    },
  },
})
