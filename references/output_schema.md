# JSONL Output Schema

Each sampled frame produces one JSON object on its own line. The output is append-friendly and contains no unobservable labels.

## Example

```json
{
  "schema_version": "1.0",
  "source": "samples/frame_001.png",
  "frame_index": 0,
  "time_ms": null,
  "detector": "haar",
  "face": {"x": 80, "y": 40, "w": 120, "h": 120, "score": null},
  "metrics": {
    "mean_luma": 132.41,
    "laplacian_var": 214.8,
    "face_area_ratio": 0.08789,
    "face_center": {"x": 0.5, "y": 0.8}
  },
  "valid": true,
  "reasons": []
}
```

## Fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Output format version. |
| `source` | string | Input image or video path. |
| `frame_index` | integer | Zero-based source frame index. For image folders this is the sorted image index. |
| `time_ms` | number or null | Video timestamp; null for still images. |
| `detector` | string | `yunet`, `haar`, or `none`. |
| `face` | object or null | Largest detected face and an optional detector score. |
| `metrics.mean_luma` | number or null | Mean grayscale value in `[0, 255]`. |
| `metrics.laplacian_var` | number or null | Laplacian variance; higher means more high-frequency detail, not guaranteed semantic sharpness. |
| `metrics.face_area_ratio` | number or null | Face box area divided by frame area. |
| `metrics.face_center` | object or null | Normalized face-center coordinates in `[0, 1]`. |
| `valid` | boolean | True only when a face is found and all configured quality checks pass. |
| `reasons` | array of strings | Machine-readable failure reasons. |

## Reason codes

- `unreadable_frame`: decoder returned no usable frame.
- `unreadable_image`: image path existed but could not be decoded.
- `no_face`: no face passed the active detector.
- `face_too_small`: face area is below `min_face_area`.
- `low_detection_score`: YuNet confidence is below `min_face_score`.
- `underexposed`: mean luma is below `min_brightness`.
- `overexposed`: mean luma is above `max_brightness`.
- `blurry`: Laplacian variance is below `min_sharpness`.

## Interpretation limits

The metrics describe image quality and face geometry. They do not identify a person, measure emotion, indicate deception, or establish that a recording is authentic. A high `laplacian_var` can be caused by noise, and a low value can occur on a clean but intentionally soft image. Calibrate thresholds on the target camera and scene.
