#!/usr/bin/env python3
"""Offline, explainable frame analysis for the OpenCV evidence-capture skill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "This skill requires numpy and opencv-python. "
        "Install them with: python -m pip install -r requirements.txt"
    ) from exc


SCHEMA_VERSION = "1.0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_THRESHOLDS = {
    "min_brightness": 30.0,
    "max_brightness": 225.0,
    "min_sharpness": 80.0,
    "min_face_area": 0.02,
    "min_face_score": 0.70,
}


def _image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    return sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _load_detector(model_path: Optional[Path]) -> Tuple[str, Any]:
    if model_path is not None and model_path.is_file():
        factory = getattr(cv2, "FaceDetectorYN_create", None)
        if factory is None and hasattr(cv2, "FaceDetectorYN"):
            factory = cv2.FaceDetectorYN.create
        if factory is not None:
            try:
                return "yunet", factory(
                    str(model_path), "", (320, 320), 0.7, 0.3, 5000
                )
            except Exception as exc:
                print(f"warning: YuNet could not be loaded: {exc}", file=sys.stderr)
    elif model_path is not None:
        print(f"warning: YuNet model not found: {model_path}", file=sys.stderr)

    try:
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        if Path(cascade_path).exists():
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                return "haar", cascade
    except Exception as exc:
        print(f"warning: Haar could not be loaded: {exc}", file=sys.stderr)

    return "none", None


def _clip_box(x: float, y: float, w: float, h: float, width: int, height: int) -> Optional[Dict[str, int]]:
    x1 = max(0, min(width, int(round(x))))
    y1 = max(0, min(height, int(round(y))))
    x2 = max(0, min(width, int(round(x + w))))
    y2 = max(0, min(height, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _detect_face(frame: np.ndarray, detector_kind: str, detector: Any) -> Optional[Dict[str, Any]]:
    height, width = frame.shape[:2]
    if detector_kind == "yunet" and detector is not None:
        try:
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            best = max(faces, key=lambda item: float(item[14]))
            box = _clip_box(best[0], best[1], best[2], best[3], width, height)
            if box is None:
                return None
            box["score"] = round(float(best[14]), 4)
            return box
        except Exception as exc:
            print(f"warning: YuNet inference failed: {exc}", file=sys.stderr)
            return None

    if detector_kind == "haar" and detector is not None:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            if len(boxes) == 0:
                return None
            x, y, w, h = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
            box = _clip_box(x, y, w, h, width, height)
            if box is None:
                return None
            box["score"] = None
            return box
        except Exception as exc:
            print(f"warning: Haar inference failed: {exc}", file=sys.stderr)
            return None

    return None


def _empty_record(
    *, source: str, frame_index: int, time_ms: Optional[float], detector_kind: str, reason: str
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "frame_index": frame_index,
        "time_ms": None if time_ms is None else round(float(time_ms), 3),
        "detector": detector_kind,
        "face": None,
        "metrics": {
            "mean_luma": None,
            "laplacian_var": None,
            "face_area_ratio": None,
            "face_center": None,
        },
        "valid": False,
        "reasons": [reason],
    }


def analyze_frame(
    frame: Optional[np.ndarray],
    detector_kind: str = "none",
    detector: Any = None,
    *,
    source: str = "",
    frame_index: int = 0,
    time_ms: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return one explainable evidence record for a BGR frame."""
    if frame is None or getattr(frame, "size", 0) == 0 or len(frame.shape) < 2:
        return _empty_record(
            source=source,
            frame_index=frame_index,
            time_ms=time_ms,
            detector_kind=detector_kind,
            reason="unreadable_frame",
        )

    cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    face = _detect_face(frame, detector_kind, detector)
    reasons: List[str] = []
    metrics: Dict[str, Any] = {
        "mean_luma": round(mean_luma, 3),
        "laplacian_var": round(sharpness, 3),
        "face_area_ratio": None,
        "face_center": None,
    }

    if face is None:
        reasons.append("no_face")
    else:
        height, width = frame.shape[:2]
        area_ratio = (face["w"] * face["h"]) / float(width * height)
        metrics["face_area_ratio"] = round(area_ratio, 5)
        metrics["face_center"] = {
            "x": round((face["x"] + face["w"] / 2) / width, 5),
            "y": round((face["y"] + face["h"] / 2) / height, 5),
        }
        if area_ratio < cfg["min_face_area"]:
            reasons.append("face_too_small")
        if face.get("score") is not None and face["score"] < cfg["min_face_score"]:
            reasons.append("low_detection_score")

    if mean_luma < cfg["min_brightness"]:
        reasons.append("underexposed")
    elif mean_luma > cfg["max_brightness"]:
        reasons.append("overexposed")
    if sharpness < cfg["min_sharpness"]:
        reasons.append("blurry")

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "frame_index": frame_index,
        "time_ms": None if time_ms is None else round(float(time_ms), 3),
        "detector": detector_kind,
        "face": face,
        "metrics": metrics,
        "valid": not reasons,
        "reasons": reasons,
    }


def _video_frames(path: Path, sample_every: int, max_frames: Optional[int]) -> Iterator[Tuple[int, Optional[float], np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    index = 0
    emitted = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % sample_every == 0:
                time_ms = (index / fps * 1000.0) if fps > 0 else None
                yield index, time_ms, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            index += 1
    finally:
        capture.release()


def _annotate(frame: np.ndarray, record: Dict[str, Any]) -> np.ndarray:
    output = frame.copy()
    face = record.get("face")
    if face:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 220, 0), 2)
    status = "valid" if record["valid"] else "invalid: " + ",".join(record["reasons"])
    cv2.putText(output, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
    return output


def _iter_records(
    input_path: Path,
    detector_kind: str,
    detector: Any,
    sample_every: int,
    max_frames: Optional[int],
) -> Iterator[Tuple[Dict[str, Any], Optional[np.ndarray]]]:
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_SUFFIXES:
        for index, time_ms, frame in _video_frames(input_path, sample_every, max_frames):
            record = analyze_frame(
                frame, detector_kind, detector,
                source=str(input_path), frame_index=index, time_ms=time_ms,
            )
            yield record, frame
        return

    paths = _image_paths(input_path)
    if not paths:
        raise ValueError(f"No supported images found under: {input_path}")
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            record = _empty_record(
                source=str(path), frame_index=index, time_ms=None,
                detector_kind=detector_kind, reason="unreadable_image",
            )
            yield record, None
            continue
        record = analyze_frame(
            frame, detector_kind, detector,
            source=str(path), frame_index=index, time_ms=None,
        )
        yield record, frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image directory, image file, or video file")
    parser.add_argument("--output", required=True, type=Path, help="JSONL output path")
    parser.add_argument("--model", type=Path, help="Optional YuNet ONNX model")
    parser.add_argument("--sample-every", type=int, default=1, help="For video: analyze every Nth frame")
    parser.add_argument("--max-frames", type=int, help="Maximum sampled frames")
    parser.add_argument("--save-annotated", type=Path, help="Directory for annotated sampled frames")
    args = parser.parse_args(argv)

    if args.sample_every < 1:
        parser.error("--sample-every must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")

    detector_kind, detector = _load_detector(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_annotated:
        args.save_annotated.mkdir(parents=True, exist_ok=True)

    total = 0
    valid = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record, frame in _iter_records(
            args.input, detector_kind, detector, args.sample_every, args.max_frames
        ):
            total += 1
            valid += int(record["valid"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.save_annotated and frame is not None:
                out = args.save_annotated / f"frame_{record['frame_index']:06d}.jpg"
                if not cv2.imwrite(str(out), _annotate(frame, record)):
                    print(f"warning: could not write annotated frame: {out}", file=sys.stderr)

    print(json.dumps({
        "frames": total,
        "valid": valid,
        "invalid": total - valid,
        "detector": detector_kind,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
