#!/usr/bin/env python
"""
Export one original YOLO crop per instrument class.

This script runs the trained YOLO model on a folder of P&ID images and saves
the best raw crop (highest confidence, then larger area) for each class.

Usage:
    python export_yolo_instrument_crops.py --input ./images

Output:
    - Folder of class-named PNG crops (default: instruments_yolo)
    - manifest.json with source image, bbox, confidence per class
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_EXCLUDED_CLASSES = {"background"}


@dataclass
class CropCandidate:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    area: int
    image_path: str
    crop: "cv2.Mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO and export one original instrument crop per class."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("images"),
        help="Input image folder (recursive search). Default: ./images",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("best.pt"),
        help="Path to YOLO model weights (default: best.pt).",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("class_names.json"),
        help="Optional class names JSON to define expected instrument classes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("instruments_yolo"),
        help="Output folder for class crops (default: instruments_yolo).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest output path (default: <output>/manifest.json).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Primary YOLO confidence threshold (default: 0.4).",
    )
    parser.add_argument(
        "--recovery-conf",
        type=float,
        default=0.05,
        help=(
            "Lower confidence threshold used only for classes still missing after "
            "primary pass (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Padding around bbox in pixels (default: 2).",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Return non-zero if any expected class is missing in exported crops.",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]+", "_", str(name).strip())
    sanitized = sanitized.replace(" ", "_")
    return sanitized or "unnamed"


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [input_path]
    if not input_path.exists() or not input_path.is_dir():
        return []
    images = [
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return images


def load_expected_classes(
    class_names_path: Path, model_names: Mapping[int, str]
) -> List[str]:
    classes: List[str] = []

    if class_names_path.exists():
        try:
            with class_names_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                classes = [str(x).strip() for x in loaded if str(x).strip()]
        except Exception:
            classes = []

    if not classes:
        classes = [str(name).strip() for _, name in sorted(model_names.items(), key=lambda kv: kv[0])]

    filtered = []
    seen = set()
    for cls in classes:
        key = cls.lower()
        if key in DEFAULT_EXCLUDED_CLASSES:
            continue
        if key in seen:
            continue
        seen.add(key)
        filtered.append(cls)
    return filtered


def clip_bbox(
    bbox: Sequence[float], image_shape: Tuple[int, int, int], padding: int
) -> Optional[Tuple[int, int, int, int]]:
    if len(bbox) != 4:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, x1 - int(padding))
    y1 = max(0, y1 - int(padding))
    x2 = min(w, x2 + int(padding))
    y2 = min(h, y2 + int(padding))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def choose_better(existing: CropCandidate, candidate: CropCandidate) -> bool:
    # Prefer higher confidence; tie-breaker by larger crop area.
    if candidate.confidence > existing.confidence:
        return True
    if abs(candidate.confidence - existing.confidence) < 1e-9 and candidate.area > existing.area:
        return True
    return False


def run_export(args: argparse.Namespace) -> int:
    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}")
        return 1

    image_paths = list_images(args.input)
    if not image_paths:
        print(f"ERROR: no input images found in {args.input}")
        return 1

    model = YOLO(str(args.model))
    expected_classes = load_expected_classes(args.class_names, model.names)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output / "manifest.json")

    best_by_class: Dict[str, CropCandidate] = {}
    total_detections = 0
    skipped_invalid = 0

    def _process_image(
        image_path: Path,
        min_conf: float,
        allowed_classes: Optional[set[str]] = None,
    ) -> None:
        nonlocal total_detections, skipped_invalid
        image = cv2.imread(str(image_path))
        if image is None:
            return

        results = model(image, verbose=False)[0]
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf <= float(min_conf):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls.item())
            class_name = str(model.names.get(cls_idx, f"class_{cls_idx}")).strip()
            if not class_name or class_name.lower() in DEFAULT_EXCLUDED_CLASSES:
                continue
            if allowed_classes is not None and class_name not in allowed_classes:
                continue

            total_detections += 1
            clipped = clip_bbox((x1, y1, x2, y2), image.shape, args.padding)
            if clipped is None:
                skipped_invalid += 1
                continue

            cx1, cy1, cx2, cy2 = clipped
            crop = image[cy1:cy2, cx1:cx2]
            if crop is None or crop.size == 0:
                skipped_invalid += 1
                continue

            area = int((cx2 - cx1) * (cy2 - cy1))
            candidate = CropCandidate(
                class_name=class_name,
                confidence=conf,
                bbox=(cx1, cy1, cx2, cy2),
                area=area,
                image_path=str(image_path),
                crop=crop,
            )

            existing = best_by_class.get(class_name)
            if existing is None or choose_better(existing, candidate):
                best_by_class[class_name] = candidate

    # Pass 1: strict confidence across all images.
    for image_path in image_paths:
        _process_image(image_path=image_path, min_conf=float(args.conf), allowed_classes=None)

    # Pass 2: recovery confidence only for classes still missing.
    expected_set = {c for c in expected_classes}
    missing_after_primary = sorted(expected_set - set(best_by_class.keys()), key=lambda x: x.lower())
    if missing_after_primary and float(args.recovery_conf) < float(args.conf):
        missing_set = set(missing_after_primary)
        for image_path in image_paths:
            if not missing_set:
                break
            _process_image(
                image_path=image_path,
                min_conf=float(args.recovery_conf),
                allowed_classes=missing_set,
            )
            missing_set = expected_set - set(best_by_class.keys())

    # Save class crops.
    exported_classes: List[str] = []
    manifest: Dict[str, object] = {
        "input": str(args.input),
        "model": str(args.model),
        "confidence_threshold": float(args.conf),
        "expected_class_count": len(expected_classes),
        "total_detections_above_threshold": total_detections,
        "skipped_invalid_detections": skipped_invalid,
        "classes": {},
    }

    # Clean stale PNGs from previous runs to avoid stale/repeated folder content.
    for old_png in args.output.glob("*.png"):
        try:
            old_png.unlink()
        except OSError:
            pass

    for class_name in sorted(best_by_class.keys(), key=lambda x: x.lower()):
        cand = best_by_class[class_name]
        file_name = f"{safe_name(class_name)}.png"
        out_path = args.output / file_name
        cv2.imwrite(str(out_path), cand.crop)
        exported_classes.append(class_name)
        manifest["classes"][class_name] = {
            "output_file": file_name,
            "confidence": round(float(cand.confidence), 6),
            "bbox": [int(v) for v in cand.bbox],
            "source_image": cand.image_path,
            "area": int(cand.area),
        }

    expected_set = {c for c in expected_classes}
    exported_set = {c for c in exported_classes}
    missing = sorted(expected_set - exported_set, key=lambda x: x.lower())

    manifest["exported_class_count"] = len(exported_classes)
    manifest["exported_classes"] = sorted(exported_classes, key=lambda x: x.lower())
    manifest["missing_classes"] = missing

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(exported_classes)} class crop(s) to: {args.output}")
    print(f"Manifest: {manifest_path}")
    if missing:
        print(f"Missing classes ({len(missing)}): {', '.join(missing)}")
    else:
        print("All expected instrument classes were exported.")

    if args.require_all and missing:
        return 2
    return 0


def main() -> int:
    args = parse_args()
    return run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
