import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.DASHBOARD_PORT ?? 5173),
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${process.env.GATEWAY_PORT ?? 4787}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/ws": {
        target: `ws://127.0.0.1:${process.env.GATEWAY_PORT ?? 4787}`,
        ws: true,
      },
    },
  },
});
