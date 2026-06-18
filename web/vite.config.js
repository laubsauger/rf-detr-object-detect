import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [
    basicSsl(),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: ".",
        },
      ],
    }),
  ],
  server: {
    host: true,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    // Proxy the Python inference bridge through the (HTTPS) dev origin so the
    // browser uses wss://<host>/bridge — avoids mixed-content blocking of a
    // raw ws:// from the secure page. ws:true makes it a passthrough tunnel, so
    // the python server's compression/NODELAY settings apply end-to-end.
    proxy: {
      "/bridge": { target: "ws://localhost:8765", ws: true, changeOrigin: true },
    },
  },
});
