#!/usr/bin/env python
"""
Offline symbol template extractor for P&ID drawings.

Usage:
    python symbol_extractor.py --input ./images

Outputs:
    - symbol_templates.json
    - symbol_blocks.dxf (optional, enabled by default)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import ezdxf

    EZDXF_AVAILABLE = True
except Exception:
    EZDXF_AVAILABLE = False


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
INVALID_CLASS_NAMES = {"background"}

MAX_TEMPLATE_LINES = 20
MAX_TEMPLATE_CIRCLES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reusable P&ID symbol templates from image folders."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input folder containing P&ID images.",
    )
    parser.add_argument(
        "--model",
        default=Path("best.pt"),
        type=Path,
        help="Path to YOLO model weights (default: best.pt).",
    )
    parser.add_argument(
        "--output-json",
        default=Path("symbol_templates.json"),
        type=Path,
        help="Output JSON path for symbol templates.",
    )
    parser.add_argument(
        "--output-dxf",
        default=Path("symbol_blocks.dxf"),
        type=Path,
        help="Output DXF block library path.",
    )
    parser.add_argument(
        "--conf",
        default=0.4,
        type=float,
        help="YOLO confidence threshold (default: 0.4).",
    )
    parser.add_argument(
        "--roi-size",
        default=128,
        type=int,
        help="Normalized ROI size for symbol shape extraction (default: 128).",
    )
    parser.add_argument(
        "--no-dxf",
        action="store_true",
        help="Disable DXF block generation.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def detect_symbols(
    image: np.ndarray, model: YOLO, conf_threshold: float = 0.4
) -> List[Dict[str, object]]:
    """
    Run YOLO symbol detection and return filtered detections.
    """
    results = model(image, verbose=False)[0]
    detections: List[Dict[str, object]] = []

    for box in results.boxes:
        conf = float(box.conf.item())
        if conf <= conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_idx = int(box.cls.item())
        class_name = str(model.names.get(cls_idx, f"class_{cls_idx}"))

        if class_name.strip().lower() in INVALID_CLASS_NAMES:
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "class_name": class_name,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
            }
        )

    return detections


def crop_symbol_roi(
    image: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 2
) -> Optional[np.ndarray]:
    """
    Crop symbol ROI with safety bounds and optional padding.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def _line_length(line: List[int]) -> float:
    x1, y1, x2, y2 = line
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_angle_deg(line: List[int]) -> float:
    x1, y1, x2, y2 = line
    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angle < 0:
        angle += 180.0
    return angle


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs(float(a) - float(b))
    return min(diff, 180.0 - diff)


def select_main_contour(
    contours: List[np.ndarray], image_area: float, min_area_ratio: float = 0.01
) -> Tuple[Optional[np.ndarray], float]:
    """
    Keep only the dominant contour to avoid multi-contour template pollution.
    """
    if not contours:
        return None, 0.0

    main_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main_contour))
    min_area = float(image_area) * float(min_area_ratio)
    if area < min_area:
        return None, area
    return main_contour, area


def filter_lines(
    raw_lines: Optional[np.ndarray],
    min_length: float = 12.0,
    max_lines: int = MAX_TEMPLATE_LINES,
    duplicate_dist_threshold: float = 6.0,
    duplicate_angle_threshold: float = 6.0,
) -> List[List[int]]:
    """
    Remove short/noisy lines, deduplicate near-identical segments, and cap count.
    """
    if raw_lines is None:
        return []

    candidates = []
    for line in raw_lines:
        x1, y1, x2, y2 = map(int, line[0])
        geom = [x1, y1, x2, y2]
        length = _line_length(geom)
        if length < float(min_length):
            continue
        angle = _line_angle_deg(geom)
        candidates.append((length, angle, geom))

    if not candidates:
        return []

    # Keep stronger geometry first.
    candidates.sort(key=lambda item: item[0], reverse=True)

    kept: List[Tuple[float, float, List[int]]] = []
    for length, angle, geom in candidates:
        x1, y1, x2, y2 = geom
        dup = False
        for _, kept_angle, kept_geom in kept:
            kx1, ky1, kx2, ky2 = kept_geom
            angle_close = _angle_diff_deg(angle, kept_angle) <= float(duplicate_angle_threshold)
            endpoint_gap = min(
                float(np.hypot(x1 - kx1, y1 - ky1)) + float(np.hypot(x2 - kx2, y2 - ky2)),
                float(np.hypot(x1 - kx2, y1 - ky2)) + float(np.hypot(x2 - kx1, y2 - ky1)),
            )
            if angle_close and endpoint_gap <= float(duplicate_dist_threshold) * 2.0:
                dup = True
                break
        if dup:
            continue
        kept.append((length, angle, geom))
        if len(kept) >= int(max_lines):
            break

    return [item[2] for item in kept]


def filter_circles(
    raw_circles: Optional[np.ndarray],
    main_contour: Optional[np.ndarray],
    min_radius: int,
    max_circles: int = MAX_TEMPLATE_CIRCLES,
    center_margin: float = 1.5,
) -> List[List[int]]:
    """
    Filter circle noise by radius, contour inclusion, and duplicate suppression.
    """
    if raw_circles is None:
        return []

    circles = np.round(raw_circles[0, :]).astype(int)
    candidates: List[List[int]] = []
    for c in circles:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if r < int(min_radius):
            continue
        if main_contour is not None:
            inside = cv2.pointPolygonTest(main_contour, (float(cx), float(cy)), False)
            if inside < -float(center_margin):
                continue
        candidates.append([cx, cy, r])

    if not candidates:
        return []

    # Prefer larger circles first.
    candidates.sort(key=lambda x: x[2], reverse=True)
    kept: List[List[int]] = []
    for cx, cy, r in candidates:
        dup = False
        for kcx, kcy, kr in kept:
            if float(np.hypot(cx - kcx, cy - kcy)) <= 3.0 and abs(r - kr) <= 2:
                dup = True
                break
        if dup:
            continue
        kept.append([cx, cy, r])
        if len(kept) >= int(max_circles):
            break
    return kept


def limit_primitives(
    primitives: Dict[str, list],
    max_lines: int = MAX_TEMPLATE_LINES,
    max_circles: int = MAX_TEMPLATE_CIRCLES,
) -> Dict[str, list]:
    """
    Cap primitive counts to keep templates compact and stable.
    """
    return {
        "lines": list(primitives.get("lines", []))[: int(max_lines)],
        "circles": list(primitives.get("circles", []))[: int(max_circles)],
        # Exactly one contour is retained by design.
        "contours": list(primitives.get("contours", []))[:1],
    }


def compute_template_score(normalized_primitives: Dict[str, list], contour_area_ratio: float) -> float:
    """
    Score one sample for class template selection.
    Higher score favors:
    - stronger main contour area
    - moderate primitive complexity (not sparse, not noisy)
    """
    lines_count = len(normalized_primitives.get("lines", []))
    circles_count = len(normalized_primitives.get("circles", []))
    contours_count = len(normalized_primitives.get("contours", []))

    if contours_count == 0:
        return -1.0

    # Reward line count in preferred engineering range.
    if 5 <= lines_count <= 20:
        line_quality = 1.0
    elif 2 <= lines_count <= 4:
        line_quality = 0.7
    elif lines_count == 0:
        line_quality = 0.25
    else:
        line_quality = 0.5

    # Reward moderate circles.
    if circles_count <= 5:
        circle_quality = 1.0
    else:
        circle_quality = 0.3

    score = (
        float(contour_area_ratio) * 10.0
        + line_quality * 2.0
        + circle_quality * 1.0
        + min(lines_count, 20) * 0.05
    )
    return float(score)


def extract_shape_features(roi: np.ndarray, size: int = 128) -> Optional[Dict[str, list]]:
    """
    Extract clean primitive shape features from one symbol ROI.

    Strategy:
    - Keep only dominant contour
    - Filter noisy micro-lines/circles
    - Limit primitive counts
    """
    if roi is None or roi.size == 0:
        return None

    if size < 32:
        size = 32

    roi_resized = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        3,
    )

    # Morphology to suppress isolated edge speckles before primitive extraction.
    kernel = np.ones((3, 3), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(thresh, 40, 140)

    image_area = float(size * size)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour, main_area = select_main_contour(contours, image_area=image_area, min_area_ratio=0.01)
    if main_contour is None:
        return None

    epsilon = 0.015 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    contour_points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
    if len(contour_points) < 3:
        return None

    contour_mask = np.zeros_like(edges)
    cv2.drawContours(contour_mask, [main_contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
    edges_in_contour = cv2.bitwise_and(edges, contour_mask)

    raw_lines = cv2.HoughLinesP(
        edges_in_contour,
        rho=1,
        theta=np.pi / 180.0,
        threshold=18,
        minLineLength=max(10, size // 8),
        maxLineGap=max(2, size // 18),
    )
    lines = filter_lines(
        raw_lines,
        min_length=max(10.0, float(size) * 0.12),
        max_lines=MAX_TEMPLATE_LINES,
    )

    raw_circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, size // 10),
        param1=100,
        param2=16,
        minRadius=max(3, size // 24),
        maxRadius=max(6, size // 3),
    )
    circles = filter_circles(
        raw_circles,
        main_contour=main_contour,
        min_radius=max(3, size // 24),
        max_circles=MAX_TEMPLATE_CIRCLES,
    )

    primitives = limit_primitives(
        {
            "lines": lines,
            "circles": circles,
            "contours": [contour_points],
        },
        max_lines=MAX_TEMPLATE_LINES,
        max_circles=MAX_TEMPLATE_CIRCLES,
    )

    contour_area_ratio = float(main_area / max(image_area, 1.0))
    primitives["contour_area_ratio"] = contour_area_ratio
    return primitives


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_primitives(
    primitives: Dict[str, list], width: int, height: int
) -> Dict[str, list]:
    """
    Normalize primitive coordinates into [0, 1] space relative to ROI bounds.
    """
    w = max(1, int(width) - 1)
    h = max(1, int(height) - 1)
    r_base = max(1, min(int(width), int(height)))

    normalized = {"lines": [], "circles": [], "contours": []}

    for line in primitives.get("lines", []):
        if len(line) != 4:
            continue
        x1, y1, x2, y2 = map(float, line)
        normalized["lines"].append(
            [
                _clamp01(x1 / w),
                _clamp01(y1 / h),
                _clamp01(x2 / w),
                _clamp01(y2 / h),
            ]
        )

    for circle in primitives.get("circles", []):
        if len(circle) != 3:
            continue
        cx, cy, r = map(float, circle)
        normalized["circles"].append(
            [
                _clamp01(cx / w),
                _clamp01(cy / h),
                _clamp01(r / r_base),
            ]
        )

    for contour in primitives.get("contours", []):
        if not contour:
            continue
        normalized_contour = []
        for pt in contour:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            x, y = map(float, pt)
            normalized_contour.append([_clamp01(x / w), _clamp01(y / h)])
        if len(normalized_contour) >= 3:
            normalized["contours"].append(normalized_contour)

    return normalized


def update_templates(
    templates: Dict[str, dict],
    class_name: str,
    normalized_primitives: Dict[str, list],
    quality_score: float,
) -> None:
    """
    Keep one representative template per class.
    """
    if class_name not in templates:
        templates[class_name] = {
            "count": 0,
            "best_score": float("-inf"),
            "primitives": {"lines": [], "circles": [], "contours": []},
            "selection": {
                "strategy": "best_single_sample",
                "quality_score": None,
            },
        }

    templates[class_name]["count"] += 1

    # Replace only when a better sample is found.
    if float(quality_score) <= float(templates[class_name]["best_score"]):
        return

    templates[class_name]["best_score"] = float(quality_score)
    templates[class_name]["selection"]["quality_score"] = float(quality_score)
    templates[class_name]["primitives"] = {
        "lines": [[float(v) for v in line] for line in normalized_primitives.get("lines", [])],
        "circles": [
            [float(v) for v in circle] for circle in normalized_primitives.get("circles", [])
        ],
        "contours": [
            [[float(pt[0]), float(pt[1])] for pt in contour]
            for contour in normalized_primitives.get("contours", [])
        ],
    }


def save_templates(output_path: Path, templates: Dict[str, dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, sort_keys=True)


def _safe_block_name(class_name: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", class_name.strip().upper())
    if not base:
        base = "SYMBOL"
    if base[0].isdigit():
        base = f"S_{base}"
    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _n_to_block_point(nx: float, ny: float, size: float = 100.0) -> Tuple[float, float]:
    x = (float(nx) - 0.5) * size
    y = (0.5 - float(ny)) * size
    return (x, y)


def generate_dxf_blocks(templates: Dict[str, dict], output_path: Path) -> bool:
    """
    Generate optional DXF block library from normalized primitives.9
    """
    if not EZDXF_AVAILABLE:
        logging.warning("ezdxf is not available. Skipping DXF generation.")
        return False

    doc = ezdxf.new(dxfversion="R2018")
    used_names = set()

    for class_name, data in templates.items():
        primitives = data.get("primitives", {})
        if (
            not primitives.get("lines")
            and not primitives.get("circles")
            and not primitives.get("contours")
        ):
            continue

        block_name = _safe_block_name(class_name, used_names)
        block = doc.blocks.new(name=block_name)

        for line in primitives.get("lines", []):
            if len(line) != 4:
                continue
            p1 = _n_to_block_point(line[0], line[1])
            p2 = _n_to_block_point(line[2], line[3])
            block.add_line(p1, p2)

        for circle in primitives.get("circles", []):
            if len(circle) != 3:
                continue
            center = _n_to_block_point(circle[0], circle[1])
            radius = max(0.5, float(circle[2]) * 100.0)
            block.add_circle(center, radius=radius)

        for contour in primitives.get("contours", []):
            if not contour or len(contour) < 3:
                continue
            pts = [_n_to_block_point(pt[0], pt[1]) for pt in contour if len(pt) == 2]
            if len(pts) >= 3:
                closed = pts[0] != pts[-1]
                block.add_lwpolyline(pts, close=closed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(output_path))
    return True


def run_extraction(args: argparse.Namespace) -> int:
    logging.info("Starting offline symbol extraction")
    logging.info("Input folder: %s", args.input)
    logging.info("Model: %s", args.model)
    logging.info("Confidence threshold: %.2f", args.conf)

    if not args.input.exists() or not args.input.is_dir():
        logging.error("Input directory does not exist: %s", args.input)
        return 1

    if not args.model.exists():
        logging.error("Model file not found: %s", args.model)
        return 1

    image_paths = list_images(args.input)
    if not image_paths:
        logging.error("No supported images found in: %s", args.input)
        return 1
    logging.info("Found %d image(s)", len(image_paths))

    model = YOLO(str(args.model))

    templates: Dict[str, dict] = {}
    class_detection_counter = defaultdict(int)
    skipped_counter = defaultdict(int)

    total_detections = 0
    accepted_detections = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning("Skipping unreadable image: %s", image_path)
            continue

        detections = detect_symbols(image=image, model=model, conf_threshold=args.conf)
        total_detections += len(detections)

        for det in detections:
            class_name = str(det["class_name"])
            bbox = det["bbox"]
            if class_name.strip().lower() in INVALID_CLASS_NAMES:
                skipped_counter["invalid_class"] += 1
                continue
            class_detection_counter[class_name] += 1

            roi = crop_symbol_roi(image=image, bbox=bbox, pad=2)
            if roi is None or roi.size == 0:
                skipped_counter["invalid_roi"] += 1
                continue

            features = extract_shape_features(roi=roi, size=args.roi_size)
            if features is None:
                skipped_counter["no_primitives"] += 1
                continue

            normalized = normalize_primitives(
                primitives=features, width=args.roi_size, height=args.roi_size
            )

            contour_count = len(normalized["contours"])
            structural_count = len(normalized["lines"]) + len(normalized["circles"])
            if contour_count == 0:
                skipped_counter["no_main_contour"] += 1
                continue
            if structural_count == 0:
                skipped_counter["insufficient_structure"] += 1
                continue

            contour_area_ratio = float(features.get("contour_area_ratio", 0.0))
            if contour_area_ratio < 0.01:
                skipped_counter["tiny_main_contour"] += 1
                continue

            quality_score = compute_template_score(
                normalized_primitives=normalized,
                contour_area_ratio=contour_area_ratio,
            )
            if quality_score < 0.0:
                skipped_counter["normalization_empty"] += 1
                continue

            update_templates(
                templates=templates,
                class_name=class_name,
                normalized_primitives=normalized,
                quality_score=quality_score,
            )
            accepted_detections += 1

    # Drop classes that ended with zero primitives after filtering/selection.
    templates = {
        cls: data
        for cls, data in templates.items()
        if (
            len(data["primitives"]["lines"])
            + len(data["primitives"]["circles"])
            + len(data["primitives"]["contours"])
        )
        > 0
    }

    save_templates(args.output_json, templates)
    logging.info("Saved templates JSON: %s", args.output_json)

    if not args.no_dxf:
        ok = generate_dxf_blocks(templates, args.output_dxf)
        if ok:
            logging.info("Saved DXF block library: %s", args.output_dxf)

    logging.info("Total detections: %d", total_detections)
    logging.info("Accepted detections: %d", accepted_detections)
    logging.info("Classes extracted: %d", len(templates))

    for cls in sorted(templates):
        data = templates[cls]
        p = data["primitives"]
        logging.info(
            "Class %-24s count=%4d lines=%4d circles=%4d contours=%4d",
            cls,
            data["count"],
            len(p["lines"]),
            len(p["circles"]),
            len(p["contours"]),
        )

    if skipped_counter:
        logging.info("Skipped detections:")
        for reason, count in sorted(skipped_counter.items()):
            logging.info("  - %s: %d", reason, count)

    # Validation summary
    valid_classes = 0
    for data in templates.values():
        p = data["primitives"]
        if len(p["lines"]) + len(p["circles"]) + len(p["contours"]) > 0:
            valid_classes += 1

    if valid_classes == 0:
        logging.error("Extraction finished, but no valid symbol primitives were produced.")
        return 2

    logging.info("Extraction completed successfully.")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    return run_extraction(args)


if __name__ == "__main__":
    raise SystemExit(main())
