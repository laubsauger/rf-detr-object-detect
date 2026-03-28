import sys
import time
import numpy as np

MODELS = {
    "nano": (384, "rf-detr-nano"),
    "seg-nano": (312, "rf-detr-seg-nano"),
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "nano"
if model_name not in MODELS:
    print(f"Unknown model: {model_name}. Available: {', '.join(MODELS.keys())}")
    sys.exit(1)

resolution, _ = MODELS[model_name]
ITERS = 50
WARMUP = 5

dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def print_stats(label, times):
    print(f"  {label:<20s} median {np.median(times):6.1f}ms | mean {np.mean(times):6.1f}ms | min {np.min(times):6.1f}ms | max {np.max(times):6.1f}ms")


def bench_section(label, fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(label, times)


# ============================================================
# 1. PyTorch benchmark
# ============================================================
print(f"=== RF-DETR {model_name} Benchmark ({resolution}x{resolution}, {ITERS} iterations) ===\n")

try:
    import torch

    if model_name == "nano":
        from rfdetr import RFDETRNano as ModelClass
    elif model_name == "seg-nano":
        from rfdetr import RFDETRSegNano as ModelClass

    device = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"
    print(f"[PyTorch] device: {device}")

    model = ModelClass()
    bench_section("predict (unopt)", lambda: model.predict(dummy_img, threshold=0.5))

    model.optimize_for_inference()
    bench_section("predict (opt)", lambda: model.predict(dummy_img, threshold=0.5))

    # Raw forward pass
    input_tensor = torch.randn(1, 3, resolution, resolution)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model.model.inference_model = model.model.inference_model.cuda()
        sync = torch.cuda.synchronize
    elif torch.backends.mps.is_available():
        input_tensor = input_tensor.to("mps")
        model.model.inference_model = model.model.inference_model.to("mps")
        sync = torch.mps.synchronize
    else:
        sync = lambda: None

    def forward_only():
        with torch.no_grad():
            model.model.inference_model(input_tensor)
        sync()

    bench_section("forward only", forward_only)
    print()
except Exception as e:
    print(f"[PyTorch] skipped: {e}\n")

# ============================================================
# 2. ONNX Runtime benchmark
# ============================================================
try:
    import onnxruntime as ort

    onnx_path = f"web/public/models/{model_name}/inference_model.onnx"
    input_np = np.random.randn(1, 3, resolution, resolution).astype(np.float32)

    # Try all available providers
    available = ort.get_available_providers()
    providers_to_try = []

    if "TensorrtExecutionProvider" in available:
        providers_to_try.append(("TensorRT", ["TensorrtExecutionProvider", "CUDAExecutionProvider"]))
    if "CUDAExecutionProvider" in available:
        providers_to_try.append(("CUDA", ["CUDAExecutionProvider"]))
    if "CoreMLExecutionProvider" in available:
        providers_to_try.append(("CoreML", ["CoreMLExecutionProvider"]))
    providers_to_try.append(("CPU", ["CPUExecutionProvider"]))

    print(f"[ONNX Runtime] version: {ort.__version__}")
    print(f"[ONNX Runtime] available providers: {', '.join(available)}")

    for label, providers in providers_to_try:
        try:
            sess = ort.InferenceSession(onnx_path, providers=providers)
            actual = sess.get_providers()
            print(f"[ONNX Runtime] {label} (active: {', '.join(actual)})")

            def run_onnx():
                sess.run(None, {"input": input_np})

            bench_section(f"ONNX {label}", run_onnx)
        except Exception as e:
            print(f"[ONNX Runtime] {label} failed: {e}")

    print()
except ImportError:
    print("[ONNX Runtime] not installed (pip install onnxruntime or onnxruntime-gpu)\n")

# ============================================================
# Summary
# ============================================================
print(f"Model: {model_name} | Resolution: {resolution}x{resolution} | Iterations: {ITERS}")
