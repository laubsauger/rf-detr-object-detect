import sys
from rfdetr import (
    RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase,
)
from rfdetr import (
    RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
    RFDETRSegXLarge, RFDETRSeg2XLarge,
)

MODELS = {
    # Detection
    "nano":      (RFDETRNano,      384),
    "small":     (RFDETRSmall,     512),
    "medium":    (RFDETRMedium,    576),
    "large":     (RFDETRLarge,     704),
    "base":      (RFDETRBase,      560),
    # Segmentation
    "seg-nano":   (RFDETRSegNano,   312),
    "seg-small":  (RFDETRSegSmall,  384),
    "seg-medium": (RFDETRSegMedium, 432),
    "seg-large":  (RFDETRSegLarge,  504),
    "seg-xlarge": (RFDETRSegXLarge, 624),
    "seg-2xlarge":(RFDETRSeg2XLarge,768),
}

names = sys.argv[1:] if len(sys.argv) > 1 else ["nano"]
for name in names:
    if name not in MODELS:
        print(f"Unknown model: {name}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

for name in names:
    cls, resolution = MODELS[name]
    output_dir = f"./web/public/models/{name}"
    print(f"\nExporting {name} ({cls.__name__}) at {resolution}x{resolution}")

    model = cls()
    model.export(
        output_dir=output_dir,
        shape=(resolution, resolution),
        batch_size=1,
        opset_version=17,
        verbose=True,
    )
    print(f"Export complete: {output_dir}/inference_model.onnx")
