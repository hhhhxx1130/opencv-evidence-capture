# OpenCV Evidence Capture

A small, offline visual-analysis skill for images and video. It samples frames, detects faces, measures image quality, and writes one JSONL record per frame.

## What it does

- Reads a directory of images or a video file.
- Samples frames deterministically.
- Uses OpenCV YuNet when a model is supplied and Haar Cascade as a fallback.
- Measures brightness, Laplacian sharpness, face area, and face center.
- Writes `valid`, `reasons`, and optional annotated frames.

It does not identify people, infer emotions, or make decisions about a person's behavior. See `references/privacy.md`.

## Install as a Codex skill

Clone the repository directly into the Codex skills directory:

```powershell
$dest = Join-Path $env:USERPROFILE ".codex\skills\opencv-evidence-capture"
git clone https://github.com/hhhhxx1130/opencv-evidence-capture.git $dest
```

Codex discovers the skill from its `SKILL.md`. Python dependencies are installed separately because the skill analyzes local files rather than calling a hosted service.

## Install

```bash
python -m pip install -r requirements.txt
```

## Quick start

```bash
python scripts/analyze_frames.py \
  --input path/to/images \
  --output evidence.jsonl \
  --model path/to/face_detection_yunet_2023mar.onnx
```

For a short video, pass the video path to `--input`. For visual inspection, add:

```bash
--save-annotated annotated_frames
```

Run the offline tests:

```bash
python -m unittest discover -s tests -v
```

## Data policy

Do not commit private recordings, biometric galleries, or data collected from non-consenting people. The included example is synthetic and no private media is required to run the tests.
