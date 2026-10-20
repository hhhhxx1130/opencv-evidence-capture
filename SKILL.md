---
name: opencv-evidence-capture
description: Analyze local images or video with OpenCV to detect faces, measure image quality, extract ROI metrics, and export explainable JSONL evidence. Use when a workflow needs reproducible visual inspection or evidence capture without uploading media to a cloud service.
---

# OpenCV Evidence Capture

Use this skill when a user needs a small, offline, explainable visual-analysis pipeline for local images or video.

## Required behavior

1. Read `references/output_schema.md` before changing output fields.
2. Run `scripts/analyze_frames.py` instead of rewriting frame extraction, detector loading, quality checks, and JSONL writing.
3. Treat `valid=false` as a normal result. Explain the `reasons` list instead of guessing a label.
4. Never infer identity, guilt, deception, or a person's emotional state from these measurements.
5. Use synthetic, consented, or properly licensed media. Do not add private recordings or biometric galleries to a public repository.

## Workflow

- Accept a directory of images or a video file.
- Sample frames deterministically.
- Detect a face with YuNet when an ONNX model is supplied; fall back to OpenCV Haar if available.
- Measure brightness, Laplacian sharpness, face area, and face center.
- Emit one JSON object per sampled frame.
- Optionally save annotated copies of the sampled frames.
- Summarize only the returned metrics and quality flags.

## Command

```bash
python scripts/analyze_frames.py \
  --input path/to/images-or-video \
  --output evidence.jsonl \
  --model path/to/face_detection_yunet_2023mar.onnx
```

Add `--save-annotated path/to/frames` to keep visual evidence. The script never needs network access.

## Output contract

Read `references/output_schema.md`. A valid frame must have a face, acceptable exposure, and sufficient sharpness. A frame that fails one or more checks must still be written with `valid=false` and a machine-readable `reasons` list.

## Resource map

- `scripts/analyze_frames.py`: deterministic frame pipeline and CLI.
- `references/output_schema.md`: JSONL fields and interpretation.
- `references/privacy.md`: public-repository data rules.
- `tests/test_analyze_frames.py`: offline quality checks.
