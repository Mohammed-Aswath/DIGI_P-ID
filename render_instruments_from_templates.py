#!/usr/bin/env python
"""
Render symbol templates JSON into black-and-white preview images.

Usage:
    python render_instruments_from_templates.py \
        --templates symbol_templates.json \
        --output instruments
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import cv2
import numpy as np


Point2D = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert symbol_templates.json into black-and-white PNG previews."
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("symbol_templates.json"),
        help="Path to symbol_templates.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("instruments"),
        help="Output directory for generated instrument images",
    )
    parser.add_argument(
        "--canvas",
        type=int,
        default=512,
        help="Canvas size in pixels (default: 512)",
    )
    parser.add_argument(
        "--symbol-size",
        type=float,
        default=300.0,
        help="Symbol draw size in local units mapped from template local-space (default: 300)",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Line thickness in pixels (default: 2)",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("class_names.json"),
        help="Optional class names JSON to force-generate missing class previews (default: class_names.json)",
    )
    return parser.parse_args()


def safe_filename(name: str) -> str:
    name = str(name).strip()
    if not name:
        return "unnamed"
    sanitized = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    sanitized = sanitized.replace(" ", "_")
    return sanitized or "unnamed"


def normalize_point_to_local(x: float, y: float) -> Point2D:
    # Normalized [0..1] format
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (x - 0.5) * 100.0, (0.5 - y) * 100.0
    # Pixel-like [0..128] format
    if 0.0 <= x <= 150.0 and 0.0 <= y <= 150.0:
        return ((x / 128.0) - 0.5) * 100.0, (0.5 - (y / 128.0)) * 100.0
    # Already local
    return float(x), float(y)


def normalize_circle_to_local(x: float, y: float, r: float) -> Tuple[float, float, float]:
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= r <= 1.0:
        lx, ly = normalize_point_to_local(x, y)
        return lx, ly, r * 100.0
    if 0.0 <= x <= 150.0 and 0.0 <= y <= 150.0 and 0.0 <= r <= 80.0:
        lx, ly = normalize_point_to_local(x, y)
        return lx, ly, (r / 128.0) * 100.0
    return float(x), float(y), float(r)


def to_canvas(px: float, py: float, cx: int, cy: int, scale: float) -> Tuple[int, int]:
    x = int(round(cx + (px * scale)))
    y = int(round(cy - (py * scale)))
    return x, y


def draw_template_image(
    template: Mapping[str, object],
    canvas_size: int = 512,
    symbol_size: float = 300.0,
    thickness: int = 2,
) -> np.ndarray:
    img = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

    primitives = template.get("primitives", {})
    if not isinstance(primitives, Mapping):
        primitives = {}

    lines = primitives.get("lines", [])
    circles = primitives.get("circles", [])
    contours = primitives.get("contours", [])
    texts = template.get("texts", None)
    if not isinstance(texts, list) or len(texts) == 0:
        texts = primitives.get("texts", [])
    if not isinstance(texts, list):
        texts = []

    fill_cfg = template.get("fill", {})
    if not isinstance(fill_cfg, Mapping):
        fill_cfg = primitives.get("fill", {})
    if not isinstance(fill_cfg, Mapping):
        fill_cfg = {}

    contour_fills = fill_cfg.get("contours", [])
    circle_fills = fill_cfg.get("circles", [])

    cx = canvas_size // 2
    cy = canvas_size // 2
    scale = float(symbol_size) / 100.0

    drawn = False
    base_thickness = int(max(1, thickness))

    contour_items: List[Tuple[np.ndarray, str]] = []
    if isinstance(contours, list):
        for i, contour in enumerate(contours):
            if not isinstance(contour, list) or len(contour) < 3:
                continue
            pts: List[Tuple[int, int]] = []
            for pt in contour:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    continue
                try:
                    x, y = map(float, pt)
                except (TypeError, ValueError):
                    continue
                lx, ly = normalize_point_to_local(x, y)
                pts.append(to_canvas(lx, ly, cx, cy, scale))
            if len(pts) < 3:
                continue
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            fill_mode = "none"
            if isinstance(contour_fills, list) and i < len(contour_fills):
                fill_mode = str(contour_fills[i]).strip().lower()
            contour_items.append((arr, fill_mode))

    circle_items: List[Tuple[Tuple[int, int], int, str]] = []
    if isinstance(circles, list):
        for i, circ in enumerate(circles):
            if not isinstance(circ, (list, tuple)) or len(circ) != 3:
                continue
            try:
                x, y, r = map(float, circ)
            except (TypeError, ValueError):
                continue
            lx, ly, lr = normalize_circle_to_local(x, y, r)
            center = to_canvas(lx, ly, cx, cy, scale)
            radius_px = int(round(max(1.0, lr * scale)))
            fill_mode = "none"
            if isinstance(circle_fills, list) and i < len(circle_fills):
                fill_mode = str(circle_fills[i]).strip().lower()
            circle_items.append((center, radius_px, fill_mode))

    # 1) Filled contours
    for arr, fill_mode in contour_items:
        if fill_mode == "filled":
            cv2.fillPoly(img, [arr], color=0)
            drawn = True

    # 2) Filled circles
    for center, radius_px, fill_mode in circle_items:
        if fill_mode == "filled":
            cv2.circle(img, center, radius_px, color=0, thickness=-1, lineType=cv2.LINE_AA)
            drawn = True

    # 3) Outlines (contours/circles)
    for arr, fill_mode in contour_items:
        if fill_mode != "filled":
            cv2.polylines(img, [arr], True, color=0, thickness=base_thickness, lineType=cv2.LINE_AA)
            drawn = True

    for center, radius_px, fill_mode in circle_items:
        if fill_mode != "filled":
            cv2.circle(img, center, radius_px, color=0, thickness=base_thickness, lineType=cv2.LINE_AA)
            drawn = True

    # 4) Lines
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, line)
            except (TypeError, ValueError):
                continue
            lx1, ly1 = normalize_point_to_local(x1, y1)
            lx2, ly2 = normalize_point_to_local(x2, y2)
            p1 = to_canvas(lx1, ly1, cx, cy, scale)
            p2 = to_canvas(lx2, ly2, cx, cy, scale)
            cv2.line(img, p1, p2, color=0, thickness=base_thickness, lineType=cv2.LINE_AA)
            drawn = True

    # 5) Texts on top
    if isinstance(texts, list):
        for txt in texts:
            if not isinstance(txt, (list, tuple)) or len(txt) != 4:
                continue
            try:
                x, y, content, scale_txt = txt
                x = float(x)
                y = float(y)
                content = str(content)
                scale_txt = float(scale_txt)
            except Exception:
                continue
            if not content:
                continue

            lx, ly = normalize_point_to_local(x, y)
            px, py = to_canvas(lx, ly, cx, cy, scale)

            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness_txt = max(1, int(round(base_thickness * 0.75)))
            # JSON text scale is template-relative; map it to a visible OpenCV font scale.
            # If provided value is already large, preserve it.
            if scale_txt <= 1.0:
                font_scale = max(0.26, scale_txt * (float(symbol_size) / 42.0))
            else:
                font_scale = scale_txt * 0.9
            (w, h), _ = cv2.getTextSize(content, font, font_scale, thickness_txt)

            # Center alignment
            px = int(px - w / 2)
            py = int(py + h / 2)

            cv2.putText(
                img,
                content,
                (px, py),
                font,
                font_scale,
                0,
                thickness_txt,
                cv2.LINE_AA,
            )
            drawn = True

    if not drawn:
        # Fallback square if template has no drawable primitives.
        half = int(round(symbol_size * 0.5))
        p1 = (cx - half, cy - half)
        p2 = (cx + half, cy + half)
        cv2.rectangle(img, p1, p2, color=0, thickness=base_thickness)

    return img


def main() -> int:
    args = parse_args()

    if not args.templates.exists():
        raise FileNotFoundError(f"Template file not found: {args.templates}")

    with args.templates.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Template JSON root must be an object.")

    args.output.mkdir(parents=True, exist_ok=True)

    requested_classes = set(data.keys())
    if args.class_names.exists():
        try:
            with args.class_names.open("r", encoding="utf-8") as f:
                class_names = json.load(f)
            if isinstance(class_names, list):
                for cls in class_names:
                    cls_name = str(cls).strip()
                    if not cls_name:
                        continue
                    if cls_name.lower() == "background":
                        continue
                    requested_classes.add(cls_name)
        except Exception:
            pass

    generated = 0
    for class_name in sorted(requested_classes, key=lambda x: x.lower()):
        template = data.get(class_name, {})
        if not isinstance(template, Mapping):
            template = {}
        image = draw_template_image(
            template=template,
            canvas_size=int(max(128, args.canvas)),
            symbol_size=float(max(20.0, args.symbol_size)),
            thickness=int(max(1, args.thickness)),
        )
        out_name = f"{safe_filename(class_name)}.png"
        out_path = args.output / out_name
        cv2.imwrite(str(out_path), image)
        generated += 1

    print(f"Generated {generated} instrument image(s) in: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
