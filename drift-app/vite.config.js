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
    // Never full-reload the app because a generated data file changed. (The sim
    // now writes to test/output/ — outside this tree — but keep this as a guard
    // so a stray write into public/ can't wipe React state mid-run.)
    watch: {
      ignored: ['**/public/drift_data.json', '**/public/incident.json'],
    },
  },
})
