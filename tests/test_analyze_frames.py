from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_frames import analyze_frame, main  # noqa: E402


def checkerboard(size: int = 96, cell: int = 8) -> np.ndarray:
    grid = (np.indices((size, size)).sum(axis=0) // cell) % 2
    gray = (grid * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class FrameAnalysisTests(unittest.TestCase):
    def test_unreadable_frame_returns_contract_record(self):
        record = analyze_frame(None, source="missing.jpg", frame_index=3)
        self.assertFalse(record["valid"])
        self.assertEqual(record["reasons"], ["unreadable_frame"])
        self.assertIsNone(record["metrics"]["mean_luma"])
        self.assertEqual(record["schema_version"], "1.0")

    def test_dark_frame_reports_multiple_reasons(self):
        record = analyze_frame(np.zeros((96, 96, 3), dtype=np.uint8))
        self.assertFalse(record["valid"])
        self.assertIn("no_face", record["reasons"])
        self.assertIn("underexposed", record["reasons"])
        self.assertIn("blurry", record["reasons"])

    def test_bright_textured_frame_has_numeric_metrics(self):
        record = analyze_frame(checkerboard())
        self.assertIn("no_face", record["reasons"])
        self.assertGreater(record["metrics"]["mean_luma"], 30)
        self.assertGreater(record["metrics"]["laplacian_var"], 80)

    def test_image_directory_cli_continues_after_corrupt_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs"
            annotated = root / "annotated"
            inputs.mkdir()
            (inputs / "a_corrupt.png").write_bytes(b"not an image")
            self.assertTrue(cv2.imwrite(str(inputs / "b_valid.png"), checkerboard()))
            output = root / "evidence.jsonl"

            result = main([
                "--input", str(inputs), "--output", str(output),
                "--save-annotated", str(annotated),
            ])
            self.assertEqual(result, 0)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["reasons"], ["unreadable_image"])
            self.assertEqual(len(list(annotated.glob("*.jpg"))), 1)

    def test_video_sampling_and_annotation_use_current_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "sample.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 96)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV video writer is unavailable")
            for _ in range(4):
                writer.write(checkerboard())
            writer.release()

            output = root / "video.jsonl"
            annotated = root / "frames"
            result = main([
                "--input", str(video), "--output", str(output),
                "--sample-every", "2", "--save-annotated", str(annotated),
            ])
            self.assertEqual(result, 0)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(len(list(annotated.glob("*.jpg"))), 2)
            self.assertTrue(all(record["time_ms"] is not None for record in records))


if __name__ == "__main__":
    unittest.main()
