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
  },
});
