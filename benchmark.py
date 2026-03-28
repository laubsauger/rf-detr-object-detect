import time
import numpy as np
import torch
from rfdetr import RFDETRNano

# Simulate a 640x480 RGB image
dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print("=== RF-DETR Nano Benchmark (PyTorch) ===")
print(f"Device: {('MPS' if torch.backends.mps.is_available() else 'CPU')}")
print()

# Unoptimized
model = RFDETRNano()
# Warmup
for _ in range(3):
    model.predict(dummy_img, threshold=0.5)

times = []
for _ in range(50):
    t0 = time.perf_counter()
    model.predict(dummy_img, threshold=0.5)
    times.append((time.perf_counter() - t0) * 1000)

print(f"Unoptimized:  median {np.median(times):.1f}ms | mean {np.mean(times):.1f}ms | min {np.min(times):.1f}ms | max {np.max(times):.1f}ms")

# Optimized
model.optimize_for_inference()
for _ in range(3):
    model.predict(dummy_img, threshold=0.5)

times = []
for _ in range(50):
    t0 = time.perf_counter()
    model.predict(dummy_img, threshold=0.5)
    times.append((time.perf_counter() - t0) * 1000)

print(f"Optimized:    median {np.median(times):.1f}ms | mean {np.mean(times):.1f}ms | min {np.min(times):.1f}ms | max {np.max(times):.1f}ms")

# Raw model forward pass only (no pre/post processing)
print()
print("--- Raw forward pass only (no pre/post) ---")
input_tensor = torch.randn(1, 3, 384, 384)
if torch.backends.mps.is_available():
    input_tensor = input_tensor.to("mps")
    model.model.inference_model = model.model.inference_model.to("mps")

for _ in range(5):
    with torch.no_grad():
        model.model.inference_model(input_tensor)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

times = []
for _ in range(50):
    t0 = time.perf_counter()
    with torch.no_grad():
        model.model.inference_model(input_tensor)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

print(f"Forward only: median {np.median(times):.1f}ms | mean {np.mean(times):.1f}ms | min {np.min(times):.1f}ms | max {np.max(times):.1f}ms")
print()
print(f"50 iterations each")
