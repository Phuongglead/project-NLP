import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_PROXY_TARGET || "http://192.168.1.198:1408",
        changeOrigin: true
      }
    }
  }
});