# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Real-time webcam object detection using RF-DETR (nano variant) with COCO classes. Single-script project (`cam-detect.py`) that captures webcam frames, runs inference, and displays annotated results via OpenCV.

## Running

```bash
source venv/bin/activate
python cam-detect.py
```

Press `q` to quit the webcam window.

## Dependencies

Python 3.11 venv with: `rfdetr`, `supervision`, `opencv-python`, `torch`, `torchvision`. The model weights (`rf-detr-nano.pth`, ~350MB) are auto-downloaded on first run.

## Key Details

- RF-DETR expects RGB input; OpenCV captures BGR. The BGR-to-RGB conversion (`[:, :, ::-1]`) must use `.copy()` to avoid negative-stride errors with PyTorch tensors.
- Detection labels use `rfdetr.assets.coco_classes.COCO_CLASSES` indexed by `class_id`.
- Annotation is handled by `supervision.BoxAnnotator` and `supervision.LabelAnnotator`.
