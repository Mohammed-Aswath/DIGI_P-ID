# ==============================================================================
# P&ID DIGITIZATION AI - MASTER BACKEND SERVER (v16 - Full Inconsistency Flagging)
# ==============================================================================
print("--- Initializing Backend Server ---")

import os
import re
import json
import base64
import zipfile
import subprocess
import sys
import time
import threading
import logging
from pathlib import Path
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pdf2image import convert_from_bytes
from ultralytics import YOLO
from skimage.morphology import skeletonize
from collections import defaultdict, Counter, deque
import traceback
import xml.etree.ElementTree as ET
from xml.dom import minidom
from shapely.geometry import LineString, Point, box
import networkx as nx
from doctr.models import ocr_predictor
import pytesseract
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 1. INITIALIZE FLASK APP & DEFINE PATHS ---
# Flask automatically serves static files from 'static' folder at /static URL
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit for uploads
# CORS configuration for better error handling
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# --- CONFIGURE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = Path("best.pt")
CLASS_NAMES_PATH = Path("class_names.json")
FEEDBACK_DIR = Path("feedback_data")
FEEDBACK_IMAGES_DIR = FEEDBACK_DIR / "images"
FEEDBACK_LABELS_DIR = FEEDBACK_DIR / "labels"
FEEDBACK_TAGS_DIR = FEEDBACK_DIR / "tags"
DXF_OUTPUT_DIR = Path("outputs") / "dxf"
ANALYSES_DIR = Path("analyses")
ANALYSES_IMAGES_DIR = Path("static") / "images"
RESTART_FLAG_FILE = Path("_RESTART_REQUIRED_")

FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_LABELS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_TAGS_DIR.mkdir(parents=True, exist_ok=True)
DXF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
ANALYSES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION FOR ADVANCED OCR & VALIDATION ---
class CONFIG:
    DET_ARCH = 'db_mobilenet_v3_large'
    RECO_ARCH = 'crnn_vgg16_bn'
    UPSCALE_FACTOR = 2.0
    NOTES_CROP_PERCENTAGE = 0.22
    BOX_THRESHOLD = 0.3
    WORD_CONF_THRESHOLD = 0.3
    TAG_CONFIDENCE_REVIEW_THRESHOLD = 0.8
    SYMBOL_CONFIDENCE_REVIEW_THRESHOLD = 0.5  # New: Flag ambiguous symbols
    GDRIVE_DATASET_PATH = Path("/content/drive/MyDrive/colab_data/pid_dataset")
    FUZZY_ACRONYM_MAP = {
        'GRO': 'GRP', 'DE1': 'DDL', 'ZLO': 'ZLC', '2L0': 'ZLC', 'GRI': 'GRP',
        'INS38C': 'INS(38C)', 'INS-52C': 'INS(52C)', 'ODL': 'DDL', 'SDI': 'SDL'
    }
    TEXT_CORRECTION_PATTERNS = {
        r'^([A-Z]{2})(\d{5})$': r'\1-\2', r'^(\d)([A-Z]{2})(\d{4})$': r'\1-\2-\3'
    }

# --- 2. LOAD AI MODELS & CLASS NAMES (GLOBAL STATE) ---
print("Loading AI Models... This may take a moment.")
try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Could not find 'best.pt'. Make sure your trained model file is in the project folder.")
    symbol_detector = YOLO(MODEL_PATH)
    
    if CLASS_NAMES_PATH.is_file():
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = list(symbol_detector.names.values())
        with open(CLASS_NAMES_PATH, 'w') as f:
            json.dump(class_names, f)
            
    text_detector = ocr_predictor(det_arch=CONFIG.DET_ARCH, reco_arch=CONFIG.RECO_ARCH, pretrained=True, export_as_straight_boxes=True)
    text_detector.det_predictor.model.postprocessor.box_thresh = CONFIG.BOX_THRESHOLD
    print("âœ… All AI models loaded successfully.")
except Exception as e:
    print(f"âŒ CRITICAL ERROR: Failed to load AI models: {e}")
    exit()

# --- 2.5: PADDLE OCR CLIENT (Microservice) ---
PADDLE_OCR_CLIENT_AVAILABLE = False
try:
    from paddle_ocr_client import call_paddle_ocr_service, call_paddle_text_ocr_service
    PADDLE_OCR_CLIENT_AVAILABLE = True
    print("âœ… PaddleOCR client (microservice) loaded successfully.")
except ImportError as e:
    print(f"âš ï¸ PaddleOCR client not available: {e}")
    print("   Install requests: pip install requests")
    print("   Make sure paddle_ocr_client.py is in the project directory")
    print("   Continuing with existing OCR only...")

# --- 2.6: METADATA EXTRACTION (Microservice) ---
METADATA_EXTRACTOR_AVAILABLE = False
try:
    from metadata_extractor import MetadataExtractor, integrate_paddle_client
    METADATA_EXTRACTOR_AVAILABLE = True
    print("âœ… Metadata extractor loaded successfully.")
except ImportError as e:
    print(f"âš ï¸ Metadata extractor not available: {e}")
    print("   Make sure metadata_extractor.py and table_detector.py are in the project directory")
    print("   Continuing without metadata extraction...")

# --- 2.7: DXF EXPORT ---
DXF_EXPORT_AVAILABLE = False
try:
    from pid_to_dxf import export_pid_json_to_dxf, export_geometry_to_dxf, SchemaValidationError
    DXF_EXPORT_AVAILABLE = True
    print("DXF export module loaded successfully.")
except Exception as e:
    print(f"DXF export module not available: {e}")
    print("   Install ezdxf and ensure pid_to_dxf.py is in the project directory")
    print("   Continuing without DXF export...")


# --- 3. DATA FORMATTING FUNCTIONS ---
def format_to_xml(graph_data):
    root = ET.Element("PID_Graph")
    nodes_element = ET.SubElement(root, "Nodes")
    for node in graph_data:
        node_element = ET.SubElement(nodes_element, "Node", id=str(node["node_id"]))
        ET.SubElement(node_element, "Type").text = str(node["type"])
        ET.SubElement(node_element, "Tag").text = str(node.get("tag", "N/A"))
        ET.SubElement(node_element, "TagConfidence").text = str(node.get("tag_confidence", "N/A"))
        ET.SubElement(node_element, "NeedsReview").text = str(node.get("needs_review", False)).lower()
        ET.SubElement(node_element, "SymbolConfidence").text = str(node.get("symbol_confidence", "N/A"))
        ET.SubElement(node_element, "ConnectionValid").text = str(node.get("connection_valid", True)).lower()
        connections_element = ET.SubElement(node_element, "Connections")
        for connected_id in sorted(set(node.get("connections", []))):
            ET.SubElement(connections_element, "ConnectedNode", id=str(connected_id))
    xml_str = ET.tostring(root, 'utf-8')
    return minidom.parseString(xml_str).toprettyxml(indent="  ")

def format_to_iso15926_json(graph_data):
    entities, relationships = [], []
    processed_relationships = set()
    for node in graph_data:
        entities.append({
            "id": f"Node_{node['node_id']}",
            "class": "PhysicalObject",
            "attributes": {
                "type": node['type'],
                "tag": node.get('tag', 'N/A'),
                "tag_confidence": node.get('tag_confidence', 'N/A'),
                "symbol_confidence": node.get('symbol_confidence', 'N/A'),
                "needs_review": node.get('needs_review', False),
                "connection_valid": node.get('connection_valid', True)
            }
        })
        for neighbor in node.get("connections", []):
            if int(neighbor) == int(node["node_id"]):
                continue
            relationship_key = tuple(sorted((int(node["node_id"]), int(neighbor))))
            if relationship_key in processed_relationships:
                continue
            processed_relationships.add(relationship_key)
            if relationship_key[0] < relationship_key[1]:
                relationships.append({
                    "from": f"Node_{relationship_key[0]}",
                    "to": f"Node_{relationship_key[1]}",
                    "type": "is-connected-to"
                })
    return {"entities": entities, "relationships": relationships}


def _safe_numeric_pair(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _safe_bbox_center(value):
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x1, y1, x2, y2 = float(value[0]), float(value[1]), float(value[2]), float(value[3])
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        except (TypeError, ValueError):
            return None
    return None


def _build_analysis_id():
    return f"ANA_{time.strftime('%Y%m%dT%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"


def build_line_geometry_json(line_data):
    """
    Convert raw line segments into structured geometry JSON.

    Validation rules:
    - Skip zero-length lines.
    - Deduplicate identical/reversed segments.
    - Always emit float coordinates.
    """
    if not isinstance(line_data, (list, tuple)):
        return []

    lines = []
    seen_segments = set()
    for raw_line in line_data:
        if not isinstance(raw_line, (list, tuple)) or len(raw_line) != 4:
            continue

        try:
            x1, y1, x2, y2 = (float(raw_line[0]), float(raw_line[1]), float(raw_line[2]), float(raw_line[3]))
        except (TypeError, ValueError):
            continue

        length = float(np.hypot(x2 - x1, y2 - y1))
        if length <= 1e-6:
            continue

        # Canonical key removes orientation impact (A->B same as B->A).
        p1 = (round(x1, 3), round(y1, 3))
        p2 = (round(x2, 3), round(y2, 3))
        segment_key = tuple(sorted((p1, p2)))
        if segment_key in seen_segments:
            continue
        seen_segments.add(segment_key)
        lines.append({
            "start": [x1, y1],
            "end": [x2, y2],
            "length": length
        })

    # Stable deterministic IDs after filtering.
    for idx, line in enumerate(lines):
        line["id"] = f"L{idx}"
    return lines


def attach_symbols_to_lines(symbol_nodes, line_data, lines_json=None):
    """
    Attach each symbol center to the nearest detected line segment.

    Returns:
        dict[node_id, line_id]
    """
    if not isinstance(symbol_nodes, dict):
        return {}

    normalized_lines = lines_json if isinstance(lines_json, list) else build_line_geometry_json(line_data)
    if not normalized_lines:
        return {}

    line_entries = []
    for line in normalized_lines:
        start = line.get("start")
        end = line.get("end")
        line_id = line.get("id")
        if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)) or len(start) != 2 or len(end) != 2:
            continue
        if not line_id:
            continue
        line_entries.append((
            str(line_id),
            LineString([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))])
        ))

    if not line_entries:
        return {}

    symbol_line_map = {}
    for node_id, attrs in symbol_nodes.items():
        coords = attrs.get("coords") if isinstance(attrs, dict) else None
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            continue
        try:
            point = Point(float(coords[0]), float(coords[1]))
        except (TypeError, ValueError):
            continue

        best_line_id = None
        best_dist = float("inf")
        for line_id, line_geom in line_entries:
            dist = float(line_geom.distance(point))
            if dist < best_dist:
                best_dist = dist
                best_line_id = line_id

        if best_line_id is not None:
            symbol_line_map[node_id] = best_line_id
            symbol_line_map[str(node_id)] = best_line_id

    return symbol_line_map


def build_geometry_payload(graph_data_json, line_data, symbol_nodes, image_height=None, dotted_lines=None):
    """
    Build geometry-first payload for direct CAD rendering.
    """
    lines_json = build_line_geometry_json(line_data)
    symbol_line_map = attach_symbols_to_lines(symbol_nodes, line_data, lines_json=lines_json)

    equipment = []
    seen_equipment = set()
    for node in graph_data_json if isinstance(graph_data_json, list) else []:
        if not isinstance(node, dict):
            continue

        node_id_raw = node.get("node_id", node.get("id"))
        if node_id_raw is None:
            continue
        node_id = str(node_id_raw).strip()
        if not node_id or node_id in seen_equipment:
            continue

        bbox = node.get("bbox")
        bbox_payload = None
        dynamic_symbol_size = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bbox_payload = [x1, y1, x2, y2]
                bbox_w = abs(x2 - x1)
                bbox_h = abs(y2 - y1)
                if bbox_w > 0.0 and bbox_h > 0.0:
                    # Fit symbol inside YOLO detection box with a small margin.
                    dynamic_symbol_size = max(6.0, min(bbox_w, bbox_h) * 0.9)
            except (TypeError, ValueError):
                position = _safe_numeric_pair(node.get("coords")) or _safe_numeric_pair(node.get("position")) or (0.0, 0.0)
                cx, cy = float(position[0]), float(position[1])
        else:
            position = _safe_numeric_pair(node.get("coords")) or _safe_numeric_pair(node.get("position")) or (0.0, 0.0)
            cx, cy = float(position[0]), float(position[1])

        equipment.append({
            "id": node_id,
            "type": str(node.get("type", "unknown")).strip().lower() or "unknown",
            "position": [float(cx), float(cy)],
            "label": node.get("tag"),
            "bbox": bbox_payload,
            "symbol_size": dynamic_symbol_size,
            "attached_line": symbol_line_map.get(node_id, symbol_line_map.get(int(node_id) if node_id.isdigit() else node_id))
        })
        seen_equipment.add(node_id)

    dotted_keys = set()
    for raw_line in dotted_lines if isinstance(dotted_lines, (list, tuple)) else []:
        if not isinstance(raw_line, (list, tuple)) or len(raw_line) != 4:
            continue
        try:
            dx1, dy1, dx2, dy2 = map(float, raw_line)
        except (TypeError, ValueError):
            continue
        dp1 = (round(dx1, 3), round(dy1, 3))
        dp2 = (round(dx2, 3), round(dy2, 3))
        dotted_keys.add(tuple(sorted((dp1, dp2))))

    pipes = []
    for line in lines_json:
        start = line.get("start", [0.0, 0.0])
        end = line.get("end", [0.0, 0.0])
        if not (isinstance(start, (list, tuple)) and len(start) == 2 and isinstance(end, (list, tuple)) and len(end) == 2):
            continue
        start_x, start_y = float(start[0]), float(start[1])
        end_x, end_y = float(end[0]), float(end[1])
        p1 = (round(start_x, 3), round(start_y, 3))
        p2 = (round(end_x, 3), round(end_y, 3))
        seg_key = tuple(sorted((p1, p2)))
        line_style = "dashed" if seg_key in dotted_keys else "solid"
        pipes.append({
            "id": line["id"],
            "points": [[start_x, start_y], [end_x, end_y]],
            "line_style": line_style
        })

    return {
        "equipment": equipment,
        "pipes": pipes
    }


def build_dxf_geometry_payload(geometry_payload, image_height):
    """
    Convert image-space geometry (origin top-left, +Y down) to DXF-space (+Y up).
    """
    if not isinstance(geometry_payload, dict):
        return {"equipment": [], "pipes": []}

    try:
        h = float(image_height)
    except (TypeError, ValueError):
        h = 0.0

    if h <= 0.0:
        return {
            "equipment": geometry_payload.get("equipment", []),
            "pipes": geometry_payload.get("pipes", []),
        }

    converted_equipment = []
    for eq in geometry_payload.get("equipment", []) if isinstance(geometry_payload.get("equipment"), list) else []:
        if not isinstance(eq, dict):
            continue
        eq_copy = dict(eq)
        pos = eq_copy.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            try:
                eq_copy["position"] = [float(pos[0]), h - float(pos[1])]
            except (TypeError, ValueError):
                pass
        bbox = eq_copy.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = map(float, bbox)
                eq_copy["bbox"] = [x1, h - y1, x2, h - y2]
            except (TypeError, ValueError):
                pass
        converted_equipment.append(eq_copy)

    converted_pipes = []
    for pipe in geometry_payload.get("pipes", []) if isinstance(geometry_payload.get("pipes"), list) else []:
        if not isinstance(pipe, dict):
            continue
        pipe_copy = dict(pipe)
        points = pipe_copy.get("points", [])
        new_points = []
        if isinstance(points, list):
            for pt in points:
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    try:
                        new_points.append([float(pt[0]), h - float(pt[1])])
                    except (TypeError, ValueError):
                        continue
        if len(new_points) >= 2:
            pipe_copy["points"] = new_points
            converted_pipes.append(pipe_copy)

    return {
        "equipment": converted_equipment,
        "pipes": converted_pipes
    }


def build_dxf_payload_from_graph_data(graph_data_json):
    if not isinstance(graph_data_json, list):
        raise ValueError("graph_data_json must be a list")

    equipment = []
    node_ids = set()

    for idx, node in enumerate(graph_data_json):
        if not isinstance(node, dict):
            continue

        node_id_raw = node.get("node_id", node.get("id"))
        if node_id_raw is None:
            continue
        node_id = str(node_id_raw).strip()
        if not node_id or node_id in node_ids:
            continue
            
        position = (
            _safe_numeric_pair(node.get("position"))
            or _safe_numeric_pair(node.get("coords"))
            or _safe_bbox_center(node.get("bbox"))
            or (float(idx) * 40.0, 0.0)
        )
        node_type = str(node.get("type", "unknown")).strip().lower() or "unknown"
        tag_value = node.get("tag")
        label_value = str(tag_value).strip() if tag_value is not None else ""

        equipment.append({
            "id": node_id,
            "type": node_type,
            "position": [position[0], position[1]],
            "label": label_value if label_value else None,
        })
        node_ids.add(node_id)

    pipe_edges = set()
    for node in graph_data_json:
        if not isinstance(node, dict):
            continue
        source_id_raw = node.get("node_id", node.get("id"))
        if source_id_raw is None:
            continue
        source_id = str(source_id_raw).strip()
        if source_id not in node_ids:
            continue

        raw_connections = node.get("connections", [])
        if not isinstance(raw_connections, (list, tuple)):
            raw_connections = [raw_connections]

        for neighbor in raw_connections:
            neighbor_id = str(neighbor).strip()
            if not neighbor_id or neighbor_id == source_id or neighbor_id not in node_ids:
                continue
            pipe_edges.add(tuple(sorted((source_id, neighbor_id))))

    pipes = [{"from": a, "to": b} for a, b in sorted(pipe_edges)]
    return {"equipment": equipment, "pipes": pipes}


def generate_dxf_artifact(graph_data_json, analysis_id, geometry_payload=None):
    result = {
        "available": False,
        "filename": None,
        "download_url": None,
        "error": None,
        "mode": None,
    }

    if not DXF_EXPORT_AVAILABLE:
        result["error"] = "DXF export module not available"
        return result

    try:
        safe_analysis = secure_filename(str(analysis_id)) or _build_analysis_id()
        filename = f"{safe_analysis}.dxf"
        output_path = DXF_OUTPUT_DIR / filename

        if isinstance(geometry_payload, dict) and isinstance(geometry_payload.get("pipes"), list):
            export_geometry_to_dxf(payload=geometry_payload, output_path=output_path)
            result["available"] = True
            result["filename"] = filename
            result["download_url"] = f"/download_dxf/{filename}"
            result["mode"] = "geometry_direct"
            return result

        payload = build_dxf_payload_from_graph_data(graph_data_json)
        if not payload["equipment"]:
            result["error"] = "No equipment nodes available for DXF export"
            return result

        export_pid_json_to_dxf(payload=payload, output_path=output_path)

        result["available"] = True
        result["filename"] = filename
        result["download_url"] = f"/download_dxf/{filename}"
        result["mode"] = "graph_fallback"
        return result
    except SchemaValidationError as exc:
        result["error"] = f"Invalid DXF payload: {str(exc)}"
        return result
    except NameError:
        # Handles case where DXF import failed and symbols are missing.
        result["error"] = "DXF export symbols unavailable"
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _extract_intersection_points(intersection_geom):
    if intersection_geom.is_empty:
        return []
    geom_type = intersection_geom.geom_type
    if geom_type == "Point":
        return [(float(intersection_geom.x), float(intersection_geom.y))]
    if geom_type == "MultiPoint":
        return [(float(p.x), float(p.y)) for p in intersection_geom.geoms]
    if geom_type == "LineString":
        coords = list(intersection_geom.coords)
        if not coords:
            return []
        return [(float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))]
    if geom_type == "MultiLineString":
        points = []
        for line in intersection_geom.geoms:
            coords = list(line.coords)
            if coords:
                points.append((float(coords[0][0]), float(coords[0][1])))
                points.append((float(coords[-1][0]), float(coords[-1][1])))
        return points
    return []


def _project_t_on_segment(line, point):
    x1, y1, x2, y2 = line
    px, py = point
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return 0.0
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    return max(0.0, min(1.0, float(t)))


def _segment_length(line):
    x1, y1, x2, y2 = line
    return float(np.hypot(float(x2) - float(x1), float(y2) - float(y1)))


def _segment_angle_deg(line):
    x1, y1, x2, y2 = line
    angle = float(np.degrees(np.arctan2(float(y2) - float(y1), float(x2) - float(x1))))
    if angle < 0.0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0
    return angle


def _angle_difference_deg(a, b):
    diff = abs(float(a) - float(b))
    return min(diff, 180.0 - diff)


def project_point_to_segment(point, segment):
    """
    Project point onto segment and report whether projection falls within segment bounds.

    Returns:
        (proj_x, proj_y), distance, is_within_segment, t_clamped
    """
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = map(float, segment)
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        proj = (x1, y1)
        dist = float(np.hypot(px - x1, py - y1))
        return proj, dist, False, 0.0

    t_raw = ((px - x1) * dx + (py - y1) * dy) / denom
    t_clamped = max(0.0, min(1.0, float(t_raw)))
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist = float(np.hypot(px - proj_x, py - proj_y))
    is_within = (-1e-6 <= float(t_raw) <= 1.0 + 1e-6)
    return (proj_x, proj_y), dist, is_within, t_clamped


def _get_or_create_topology_node(point, topology_graph, topology_points, snap_threshold):
    px, py = point
    snap_threshold_sq = float(snap_threshold * snap_threshold)
    for node_id, (qx, qy) in enumerate(topology_points):
        if (px - qx) * (px - qx) + (py - qy) * (py - qy) <= snap_threshold_sq:
            return node_id
    node_id = len(topology_points)
    topology_points.append((float(px), float(py)))
    topology_graph.add_node(node_id, point=(float(px), float(py)))
    return node_id


def build_line_topology_graph(line_data, intersection_snap_threshold=10):
    """Pure geometric intersection — identical to oldapp.py."""
    topology_graph = nx.Graph()
    topology_points = []

    if not line_data:
        return topology_graph

    intersection_snap_threshold = float(intersection_snap_threshold)
    micro_segment_threshold = 5.0

    # Deduplicate and filter trivial segments.
    clean_lines = []
    seen_segments = set()
    for raw in line_data:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        x1, y1, x2, y2 = map(float, raw)
        if np.hypot(x2 - x1, y2 - y1) < micro_segment_threshold:
            continue
        key = tuple(sorted(((round(x1, 1), round(y1, 1)), (round(x2, 1), round(y2, 1)))))
        if key in seen_segments:
            continue
        seen_segments.add(key)
        clean_lines.append((x1, y1, x2, y2))

    if not clean_lines:
        return topology_graph

    line_geometries = [LineString([(l[0], l[1]), (l[2], l[3])]) for l in clean_lines]
    split_points_by_line = {
        idx: [(float(l[0]), float(l[1])), (float(l[2]), float(l[3]))]
        for idx, l in enumerate(clean_lines)
    }

    for i in range(len(clean_lines)):
        for j in range(i + 1, len(clean_lines)):
            intersection_geom = line_geometries[i].intersection(line_geometries[j])
            for pt in _extract_intersection_points(intersection_geom):
                split_points_by_line[i].append(pt)
                split_points_by_line[j].append(pt)

    for line_idx, line in enumerate(clean_lines):
        points = split_points_by_line[line_idx]
        points_sorted = sorted(points, key=lambda p: _project_t_on_segment(line, p))

        line_node_ids = []
        for pt in points_sorted:
            node_id = _get_or_create_topology_node(
                point=pt,
                topology_graph=topology_graph,
                topology_points=topology_points,
                snap_threshold=intersection_snap_threshold,
            )
            if not line_node_ids or line_node_ids[-1] != node_id:
                line_node_ids.append(node_id)

        for u, v in zip(line_node_ids, line_node_ids[1:]):
            if u == v:
                continue
            p1 = topology_graph.nodes[u]["point"]
            p2 = topology_graph.nodes[v]["point"]
            segment_length = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            if segment_length < micro_segment_threshold:
                continue
            if topology_graph.has_edge(u, v):
                if segment_length < topology_graph[u][v].get("length", segment_length):
                    topology_graph[u][v]["length"] = segment_length
                topology_graph[u][v].setdefault("line_indices", set()).add(int(line_idx))
            else:
                topology_graph.add_edge(
                    u,
                    v,
                    length=segment_length,
                    line_indices={int(line_idx)},
                )
    return topology_graph


def attach_symbols_to_topology(symbol_nodes, topology_graph, attach_threshold=24, fallback_threshold=60):
    """Identical to oldapp.py: attach each symbol to the endpoints of all nearby edges."""
    symbol_to_topology_nodes = defaultdict(set)
    topology_node_to_symbols = defaultdict(set)

    if topology_graph.number_of_edges() == 0:
        return symbol_to_topology_nodes, topology_node_to_symbols

    topology_edges = []
    for u, v in topology_graph.edges():
        p1 = topology_graph.nodes[u]["point"]
        p2 = topology_graph.nodes[v]["point"]
        topology_edges.append((u, v, LineString([p1, p2])))

    for symbol_id, attrs in symbol_nodes.items():
        x1, y1, x2, y2 = attrs["bbox"]
        symbol_box = box(float(x1), float(y1), float(x2), float(y2))
        attached_nodes = set()
        nearest_edge = None
        nearest_distance = float("inf")

        for u, v, line_geom in topology_edges:
            dist_to_box = float(line_geom.distance(symbol_box))
            if dist_to_box < nearest_distance:
                nearest_distance = dist_to_box
                nearest_edge = (u, v)
            if dist_to_box <= attach_threshold:
                attached_nodes.add(u)
                attached_nodes.add(v)

        if not attached_nodes and nearest_edge and nearest_distance <= fallback_threshold:
            attached_nodes.update(nearest_edge)

        symbol_to_topology_nodes[symbol_id] = attached_nodes
        for node_id in attached_nodes:
            topology_node_to_symbols[node_id].add(symbol_id)

    return symbol_to_topology_nodes, topology_node_to_symbols


def classify_topology_nodes(topology_graph):
    for node in topology_graph.nodes():
        deg = topology_graph.degree(node)
        if deg == 1:
            topology_graph.nodes[node]["node_type"] = "endpoint"
        elif deg == 2:
            topology_graph.nodes[node]["node_type"] = "continuation"
        else:
            topology_graph.nodes[node]["node_type"] = "junction"


def derive_symbol_adjacency_from_topology(topology_graph, symbol_to_topology_nodes, topology_node_to_symbols):
    """
    DFS traversal from each symbol's anchor nodes.  No depth cap — cycle
    prevention is handled entirely by the visited directed-edge set, which
    guarantees termination on any finite graph.
    """
    symbol_adjacency = defaultdict(set)
    # --- DIAGNOSTIC 3 accumulators ---
    diag_visited_counts = []
    diag_isolated = []  # symbols that visited 0 nodes (completely disconnected)

    for symbol_id, anchor_nodes in symbol_to_topology_nodes.items():
        if not anchor_nodes:
            diag_isolated.append(symbol_id)
            continue

        # Symbols sharing the same anchor node are immediately adjacent.
        for anchor in anchor_nodes:
            colocated_symbols = topology_node_to_symbols.get(anchor, set()) - {symbol_id}
            if colocated_symbols:
                symbol_adjacency[symbol_id].update(colocated_symbols)

        visited_directed_edges = set()
        branch_stack = []
        for anchor in anchor_nodes:
            for nbr in topology_graph.neighbors(anchor):
                branch_stack.append((anchor, nbr))

        nodes_visited = 0
        while branch_stack:
            prev_node, curr_node = branch_stack.pop()
            directed_edge = (prev_node, curr_node)
            if directed_edge in visited_directed_edges:
                continue
            visited_directed_edges.add(directed_edge)
            nodes_visited += 1

            hit_symbols = topology_node_to_symbols.get(curr_node, set()) - {symbol_id}
            if hit_symbols:
                symbol_adjacency[symbol_id].update(hit_symbols)
                continue

            for nxt in topology_graph.neighbors(curr_node):
                if nxt == prev_node:
                    continue
                if (curr_node, nxt) not in visited_directed_edges:
                    branch_stack.append((curr_node, nxt))

        diag_visited_counts.append((symbol_id, nodes_visited, len(symbol_adjacency.get(symbol_id, set()))))

    # Print traversal summary
    if diag_visited_counts:
        avg_visited = sum(v for _, v, _ in diag_visited_counts) / len(diag_visited_counts)
        no_conn = [(sid, v) for sid, v, c in diag_visited_counts if c == 0]
        print(f"  DFS traversal: avg {avg_visited:.0f} nodes visited per symbol, "
              f"{len(no_conn)}/{len(diag_visited_counts)} symbols found 0 connections.")
        if no_conn:
            print(f"  Symbols with anchors but 0 connections (visited nodes): "
                  f"{[(sid, v) for sid,v in no_conn[:10]]}")
    if diag_isolated:
        print(f"  Symbols skipped (no anchors at all): {diag_isolated}")

    return symbol_adjacency

# --- ADVANCED OCR HELPER FUNCTIONS ---
def auto_crop_main_diagram(img: np.ndarray):
    print("1. Auto-cropping the main diagram area...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(gray_inv, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        img_h, img_w = img.shape[:2]
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(w * h)
            if h <= 0:
                continue
            aspect_ratio = float(w) / float(h)
            if area > 0.1 * float(img_w * img_h) and 0.5 < aspect_ratio < 3.0:
                candidates.append((area, (x, y, w, h)))

        if candidates:
            _, (x, y, w, h) = max(candidates, key=lambda item: item[0])
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 30
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)
        cropped_img = img[y:y+h, x:x+w]
        print("âœ… Cropping successful.")
        return cropped_img, x, y
    else:
        print("âš ï¸ Warning: Could not find a dominant contour. Using original image.")
        return img, 0, 0

def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if CONFIG.UPSCALE_FACTOR > 1.0:
        upscaled = cv2.resize(denoised, None, fx=CONFIG.UPSCALE_FACTOR, fy=CONFIG.UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = denoised
    thresh = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh_inv = cv2.bitwise_not(closing)
    return cv2.cvtColor(thresh_inv, cv2.COLOR_GRAY2BGR)


def merge_lsd_lines(lines, angle_threshold=6, distance_threshold=24):
    """
    Merge fragmented LSD segments into longer pipe-like segments.
    Uses axis-aware regularization for P&ID style drawings.
    """
    if not lines:
        return []

    angle_threshold = float(angle_threshold)
    distance_threshold = float(distance_threshold)

    def classify_axis(seg, tol=8.0):
        angle = _segment_angle_deg(seg)
        if angle <= tol or angle >= (180.0 - tol):
            return "h"
        if abs(angle - 90.0) <= tol:
            return "v"
        return "d"

    def snap_axis(seg, tol=8.0):
        x1, y1, x2, y2 = map(float, seg)
        axis = classify_axis((x1, y1, x2, y2), tol=tol)
        if axis == "h":
            y = (y1 + y2) / 2.0
            return (x1, y, x2, y)
        if axis == "v":
            x = (x1 + x2) / 2.0
            return (x, y1, x, y2)
        return (x1, y1, x2, y2)

    def endpoint_gap(seg_a, seg_b):
        ax1, ay1, ax2, ay2 = seg_a
        bx1, by1, bx2, by2 = seg_b
        pts_a = [(ax1, ay1), (ax2, ay2)]
        pts_b = [(bx1, by1), (bx2, by2)]
        return min(float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) for pa in pts_a for pb in pts_b)

    def interval_gap(a_min, a_max, b_min, b_max):
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap >= 0:
            return 0.0
        return abs(overlap)

    def axis_gap(seg_a, seg_b):
        ax1, ay1, ax2, ay2 = seg_a
        bx1, by1, bx2, by2 = seg_b
        axis = classify_axis(seg_a)
        if axis == "h":
            a_min, a_max = sorted((ax1, ax2))
            b_min, b_max = sorted((bx1, bx2))
            along_gap = interval_gap(a_min, a_max, b_min, b_max)
            ortho_sep = abs(((ay1 + ay2) * 0.5) - ((by1 + by2) * 0.5))
            return along_gap, ortho_sep
        if axis == "v":
            a_min, a_max = sorted((ay1, ay2))
            b_min, b_max = sorted((by1, by2))
            along_gap = interval_gap(a_min, a_max, b_min, b_max)
            ortho_sep = abs(((ax1 + ax2) * 0.5) - ((bx1 + bx2) * 0.5))
            return along_gap, ortho_sep
        return endpoint_gap(seg_a, seg_b), float(
            LineString([(ax1, ay1), (ax2, ay2)]).distance(LineString([(bx1, by1), (bx2, by2)]))
        )

    def should_merge(seg_a, seg_b):
        axis_a = classify_axis(seg_a)
        axis_b = classify_axis(seg_b)

        angle_a = _segment_angle_deg(seg_a)
        angle_b = _segment_angle_deg(seg_b)
        angle_close = _angle_difference_deg(angle_a, angle_b) <= angle_threshold
        axis_close = axis_a == axis_b and axis_a in ("h", "v")
        if not (angle_close or axis_close):
            return False

        line_a = LineString([(seg_a[0], seg_a[1]), (seg_a[2], seg_a[3])])
        line_b = LineString([(seg_b[0], seg_b[1]), (seg_b[2], seg_b[3])])
        if float(line_a.distance(line_b)) > distance_threshold * 0.85:
            return False

        if endpoint_gap(seg_a, seg_b) <= distance_threshold:
            return True

        along_gap, ortho_sep = axis_gap(seg_a, seg_b)
        return along_gap <= distance_threshold * 1.35 and ortho_sep <= distance_threshold * 0.6

    def merge_pair(seg_a, seg_b):
        pts = np.array(
            [
                [seg_a[0], seg_a[1]],
                [seg_a[2], seg_a[3]],
                [seg_b[0], seg_b[1]],
                [seg_b[2], seg_b[3]],
            ],
            dtype=float,
        )
        axis = classify_axis(seg_a)
        if axis == "h":
            x_min = float(np.min(pts[:, 0]))
            x_max = float(np.max(pts[:, 0]))
            y = float(np.mean(pts[:, 1]))
            return (x_min, y, x_max, y)
        if axis == "v":
            y_min = float(np.min(pts[:, 1]))
            y_max = float(np.max(pts[:, 1]))
            x = float(np.mean(pts[:, 0]))
            return (x, y_min, x, y_max)

        # Diagonal: preserve direction by projecting points on dominant axis.
        ref = seg_a if _segment_length(seg_a) >= _segment_length(seg_b) else seg_b
        vx = float(ref[2] - ref[0])
        vy = float(ref[3] - ref[1])
        norm = float(np.hypot(vx, vy))
        if norm <= 1e-9:
            return tuple(map(float, ref))
        ux, uy = vx / norm, vy / norm
        center = np.mean(pts, axis=0)
        proj = [float((p[0] - center[0]) * ux + (p[1] - center[1]) * uy) for p in pts]
        t_min, t_max = min(proj), max(proj)
        p1 = (center[0] + t_min * ux, center[1] + t_min * uy)
        p2 = (center[0] + t_max * ux, center[1] + t_max * uy)
        return (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))

    normalized = []
    seen = set()
    for line in lines:
        if not isinstance(line, (list, tuple)) or len(line) != 4:
            continue
        x1, y1, x2, y2 = map(float, line)
        if np.hypot(x2 - x1, y2 - y1) < 3.0:
            continue
        seg = snap_axis((x1, y1, x2, y2), tol=8.0)
        key = tuple(sorted(((round(seg[0], 1), round(seg[1], 1)), (round(seg[2], 1), round(seg[3], 1)))))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(seg)

    if len(normalized) <= 1:
        return [tuple(map(int, seg)) for seg in normalized]

    merged = list(normalized)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                if should_merge(merged[i], merged[j]):
                    merged[i] = snap_axis(merge_pair(merged[i], merged[j]), tol=7.0)
                    merged.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    final_lines = []
    seen_final = set()
    for seg in merged:
        seg = snap_axis(seg, tol=7.0)
        if _segment_length(seg) < 6.0:
            continue
        as_int = tuple(map(int, seg))
        key = tuple(sorted(((as_int[0], as_int[1]), (as_int[2], as_int[3]))))
        if key in seen_final:
            continue
        seen_final.add(key)
        final_lines.append(as_int)
    return final_lines


def detect_lines_lsd(img_gray: np.ndarray, min_length=30):
    """
    Minimal, stable OpenCV LSD line detection.
    Philosophy: simple LSD, minimal filtering, preserve original behavior.
    """
    if img_gray is None or img_gray.size == 0:
        return []

    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(img_gray)[0]

    if lines is None:
        return []

    result = []
    seen = set()

    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])

        if np.hypot(x2 - x1, y2 - y1) < float(min_length):
            continue

        # Optional light axis snapping.
        if abs(y2 - y1) < 5:
            y2 = y1
        elif abs(x2 - x1) < 5:
            x2 = x1

        key = tuple(sorted(((x1, y1), (x2, y2))))
        if key in seen:
            continue

        seen.add(key)
        result.append((x1, y1, x2, y2))

    return result


def preprocess_for_centerline_lsd(img_gray: np.ndarray) -> np.ndarray:
    """
    Prepare a centerline-oriented LSD input by thinning thick pipe strokes.
    Returns a single-channel uint8 image.
    """
    if img_gray is None or img_gray.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    _, binary_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    closed = cv2.morphologyEx(
        binary_inv,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    skel = (skeletonize(opened // 255).astype(np.uint8) * 255)
    skel = cv2.dilate(skel, np.ones((2, 2), np.uint8), iterations=1)
    return skel


def preprocess_for_dotted_lsd(img_gray: np.ndarray) -> np.ndarray:
    """
    Preserve dashed periodicity while lightly stabilizing fragments.
    """
    if img_gray is None or img_gray.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, binary_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Light cleanup only.
    cleaned = cv2.morphologyEx(
        binary_inv,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    normalized = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        np.ones((1, 3), np.uint8),
        iterations=1
    )

    return normalized


def group_dotted_segments(lines, binary_img=None, gap_min=5, gap_max=40, min_chain=3):
    """
    Identify dotted/dashed chains from short LSD segments.
    Keeps only segments that form repetitive, axis-consistent chains.
    """
    if not lines:
        return []

    def _segment_len(seg):
        x1, y1, x2, y2 = seg
        return float(np.hypot(x2 - x1, y2 - y1))

    def _segment_angle(seg):
        x1, y1, x2, y2 = seg
        ang = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if ang > 90.0:
            ang = 180.0 - ang
        return ang

    def _sample_binary(x, y):
        if binary_img is None:
            return 1
        h, w = binary_img.shape[:2]
        xi = max(0, min(w - 1, int(round(x))))
        yi = max(0, min(h - 1, int(round(y))))
        return 1 if int(binary_img[yi, xi]) > 0 else 0

    def _gap_has_dash_support(prev_seg, next_seg, axis, min_ratio=0.18):
        if binary_img is None:
            return True

        px1, py1, px2, py2 = map(float, prev_seg)
        nx1, ny1, nx2, ny2 = map(float, next_seg)
        samples = 21

        if axis == "h":
            left_end = max(px1, px2)
            right_start = min(nx1, nx2)
            if right_start <= left_end:
                return True
            y = ((py1 + py2) * 0.5 + (ny1 + ny2) * 0.5) * 0.5
            vals = [
                _sample_binary(left_end + t * (right_start - left_end), y)
                for t in np.linspace(0.0, 1.0, samples)
            ]
        else:
            top_end = max(py1, py2)
            bottom_start = min(ny1, ny2)
            if bottom_start <= top_end:
                return True
            x = ((px1 + px2) * 0.5 + (nx1 + nx2) * 0.5) * 0.5
            vals = [
                _sample_binary(x, top_end + t * (bottom_start - top_end))
                for t in np.linspace(0.0, 1.0, samples)
            ]

        support_ratio = float(sum(vals)) / float(len(vals)) if vals else 0.0
        return support_ratio >= float(min_ratio)

    def _chain_is_dotted(chain, axis):
        if not chain:
            return False
        if binary_img is None:
            return True

        xs = [seg[0] for seg in chain] + [seg[2] for seg in chain]
        ys = [seg[1] for seg in chain] + [seg[3] for seg in chain]

        if axis == "h":
            x1, x2 = float(min(xs)), float(max(xs))
            if x2 <= x1:
                return False
            y = float(np.mean(ys))
            length = x2 - x1
            sample_count = max(24, min(160, int(length)))
            values = [_sample_binary(x1 + t * (x2 - x1), y) for t in np.linspace(0.0, 1.0, sample_count)]
        else:
            y1, y2 = float(min(ys)), float(max(ys))
            if y2 <= y1:
                return False
            x = float(np.mean(xs))
            length = y2 - y1
            sample_count = max(24, min(160, int(length)))
            values = [_sample_binary(x, y1 + t * (y2 - y1)) for t in np.linspace(0.0, 1.0, sample_count)]

        if not values:
            return False
        transitions = sum(1 for i in range(1, len(values)) if values[i] != values[i - 1])
        fill_ratio = float(sum(values)) / float(len(values))
        return transitions >= 4 and 0.08 <= fill_ratio <= 0.72

    short_lines = [tuple(map(int, ln)) for ln in lines if _segment_len(ln) <= 90.0]
    if not short_lines:
        return []

    horizontal = []
    vertical = []
    for line in short_lines:
        x1, y1, x2, y2 = line
        if abs(y2 - y1) <= abs(x2 - x1):
            horizontal.append(line)
        else:
            vertical.append(line)

    dotted = []
    dotted_seen = set()

    def _append_dotted(seg):
        key = tuple(sorted(((seg[0], seg[1]), (seg[2], seg[3]))))
        if key not in dotted_seen:
            dotted_seen.add(key)
            dotted.append(seg)

    def _process_group(group, axis):
        if not group:
            return

        if axis == "h":
            group = sorted(group, key=lambda l: ((l[1] + l[3]) * 0.5, min(l[0], l[2])))
        else:
            group = sorted(group, key=lambda l: ((l[0] + l[2]) * 0.5, min(l[1], l[3])))

        used = set()
        for i in range(len(group)):
            if i in used:
                continue

            seed = group[i]
            seed_angle = _segment_angle(seed)
            chain = [seed]
            used.add(i)
            current = seed
            chain_lengths = [_segment_len(seed)]

            while True:
                best_j = None
                best_gap = None
                cx1, cy1, cx2, cy2 = current
                current_axis_val = ((cy1 + cy2) * 0.5) if axis == "h" else ((cx1 + cx2) * 0.5)
                current_end = max(cx1, cx2) if axis == "h" else max(cy1, cy2)
                avg_dash_length = float(np.mean(chain_lengths)) if chain_lengths else _segment_len(current)
                local_gap_max = min(float(gap_max), max(float(gap_min) + 1.0, avg_dash_length * 1.7))

                for j in range(len(group)):
                    if j in used:
                        continue

                    cand = group[j]
                    cand_angle = _segment_angle(cand)
                    if abs(cand_angle - seed_angle) > 8.0:
                        continue

                    nx1, ny1, nx2, ny2 = cand
                    cand_axis_val = ((ny1 + ny2) * 0.5) if axis == "h" else ((nx1 + nx2) * 0.5)
                    if abs(cand_axis_val - current_axis_val) > 8.0:
                        continue

                    cand_start = min(nx1, nx2) if axis == "h" else min(ny1, ny2)
                    gap = cand_start - current_end
                    if gap < gap_min or gap > local_gap_max:
                        continue

                    if not _gap_has_dash_support(current, cand, axis=axis):
                        continue

                    if best_j is None or gap < best_gap:
                        best_j = j
                        best_gap = gap

                if best_j is None:
                    break

                current = group[best_j]
                chain.append(current)
                chain_lengths.append(_segment_len(current))
                used.add(best_j)

            if len(chain) >= int(min_chain) and _chain_is_dotted(chain, axis=axis):
                for seg in chain:
                    _append_dotted(seg)

    _process_group(horizontal, axis="h")
    _process_group(vertical, axis="v")
    return dotted


def collapse_parallel_dotted_lines(lines, axis_tol=8, offset_tol=3):
    """
    Collapse duplicate close parallel dotted segments into one centerline.
    Prevents twin-edge dashed exports.
    """
    if not lines:
        return []

    collapsed = []
    used = set()

    for i, a in enumerate(lines):
        if i in used:
            continue

        x1, y1, x2, y2 = a
        merged = [a]

        for j in range(i + 1, len(lines)):
            if j in used:
                continue

            bx1, by1, bx2, by2 = lines[j]

            # Horizontal duplicates.
            if abs(y2 - y1) <= axis_tol and abs(by2 - by1) <= axis_tol:
                ay = (y1 + y2) / 2.0
                by = (by1 + by2) / 2.0
                if abs(ay - by) > offset_tol:
                    continue
                a_min, a_max = sorted((x1, x2))
                b_min, b_max = sorted((bx1, bx2))
                overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
                min_span = max(1.0, min(abs(a_max - a_min), abs(b_max - b_min)))
                if overlap / min_span >= 0.6:
                    merged.append(lines[j])
                    used.add(j)

            # Vertical duplicates.
            elif abs(x2 - x1) <= axis_tol and abs(bx2 - bx1) <= axis_tol:
                ax = (x1 + x2) / 2.0
                bx = (bx1 + bx2) / 2.0
                if abs(ax - bx) > offset_tol:
                    continue
                a_min, a_max = sorted((y1, y2))
                b_min, b_max = sorted((by1, by2))
                overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
                min_span = max(1.0, min(abs(a_max - a_min), abs(b_max - b_min)))
                if overlap / min_span >= 0.6:
                    merged.append(lines[j])
                    used.add(j)

        if len(merged) == 1:
            collapsed.append(a)
        else:
            xs, ys = [], []
            for seg in merged:
                xs.extend([seg[0], seg[2]])
                ys.extend([seg[1], seg[3]])

            # Horizontal.
            if abs(y2 - y1) < abs(x2 - x1):
                collapsed.append((
                    int(min(xs)),
                    int(np.mean(ys)),
                    int(max(xs)),
                    int(np.mean(ys)),
                ))
            else:
                collapsed.append((
                    int(np.mean(xs)),
                    int(min(ys)),
                    int(np.mean(xs)),
                    int(max(ys)),
                ))

    return collapsed

def is_valid_text(text: str) -> bool:
    if not text: return False
    if text in CONFIG.FUZZY_ACRONYM_MAP.values() or text in ['STA', 'DDL', 'SDL', 'GRP']: return True
    if any(char.isalpha() for char in text) and any(char.isdigit() for char in text): return True
    if len(text) < 3: return False
    return True

def post_process_results(result):
    if not result.pages: return []
    words = [w for b in result.pages[0].blocks for l in b.lines for w in l.words]
    words = [{'value': re.sub(r'[^A-Z0-9\s-]', '', w.value), 'confidence': w.confidence, 'geometry': w.geometry} for w in words]
    print("\nDEBUG: Raw OCR words before merging:")
    for word in words:
        print(f"- Text: '{word['value']}', Confidence: {word['confidence']:.2f}, Geometry: {word['geometry']}")
    while True:
        merged_in_this_pass = False
        next_pass_words, used_indices = [], set()
        words.sort(key=lambda w: (w['geometry'][0][1], w['geometry'][0][0]))
        for i in range(len(words)):
            if i in used_indices: continue
            current_word, best_merge_candidate, min_dist = words[i], None, 0.03
            for j in range(i + 1, len(words)):
                if j in used_indices: continue
                candidate_word = words[j]
                y_center_current = (current_word['geometry'][0][1] + current_word['geometry'][1][1]) / 2
                y_center_candidate = (candidate_word['geometry'][0][1] + candidate_word['geometry'][1][1]) / 2
                if abs(y_center_current - y_center_candidate) < 0.02:
                    dist = candidate_word['geometry'][0][0] - current_word['geometry'][1][0]
                    if 0 <= dist < min_dist and not current_word['value'].startswith('-') and candidate_word['value'].startswith('-'):
                        best_merge_candidate = candidate_word
                        used_indices.add(j)
                        break
            if best_merge_candidate:
                new_value = current_word['value'] + best_merge_candidate['value']
                new_confidence = (current_word['confidence'] + best_merge_candidate['confidence']) / 2
                new_geometry = (current_word['geometry'][0], best_merge_candidate['geometry'][1])
                next_pass_words.append({'value': new_value, 'confidence': new_confidence, 'geometry': new_geometry})
                used_indices.add(i); merged_in_this_pass = True
            else:
                next_pass_words.append(current_word)
        words = next_pass_words
        if not merged_in_this_pass: break
    final_results = []
    for word in words:
        text = word['value']
        if text in CONFIG.FUZZY_ACRONYM_MAP: text = CONFIG.FUZZY_ACRONYM_MAP[text]
        for pattern, replacement in CONFIG.TEXT_CORRECTION_PATTERNS.items():
            text = re.sub(pattern, replacement, text)
        if is_valid_text(text):
            word['value'] = text
            final_results.append(word)
        else:
            print(f"DEBUG: Filtered out text: '{text}', Confidence: {word['confidence']:.2f}")
    return final_results

# --- 4. THE MAIN AI PIPELINE ---
def digitize_pid_image(original_img):
    print("\n--- Starting Enhanced Digitization Pipeline ---")
    
    # --- STAGE A: SYMBOL DETECTION ---
    results_yolo = symbol_detector(original_img, verbose=False)[0]
    yolo_data = []
    for box in results_yolo.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        class_name = symbol_detector.names[int(box.cls.item())]
        if conf > 0.4:
            yolo_data.append({"bbox": (x1, y1, x2, y2), "class_name": class_name, "conf": conf})
    print(f"âœ… Detected {len(yolo_data)} high-confidence symbols.")
    
    # --- STAGE B: TEXT DETECTION (ADVANCED OCR) ---
    main_diagram_cropped, crop_offset_x, crop_offset_y = auto_crop_main_diagram(original_img)
    print("1b. Cropping out the notes area...")
    crop_width_px = int(main_diagram_cropped.shape[1] * (1 - CONFIG.NOTES_CROP_PERCENTAGE))
    img_without_notes = main_diagram_cropped[:, :crop_width_px]
    print("âœ… Notes cropped out.")

    processed_img = preprocess_image(img_without_notes)
    processed_h, processed_w = processed_img.shape[:2]

    print("\n4. Running DocTR OCR...")
    print("   Processing image through DocTR text detector...")
    result = text_detector([processed_img])
    print(f"   âœ… DocTR detection completed. Processing results...")

    final_results = post_process_results(result)
    print(f"   âœ… Post-processing completed. {len(final_results)} text regions after filtering.")

    ocr_results_list = []
    print(f"\n   Processing {len(final_results)} text regions:")
    for idx, item in enumerate(final_results, 1):
        geo = item['geometry']
        x1 = int(geo[0][0] * processed_w / CONFIG.UPSCALE_FACTOR) + crop_offset_x
        y1 = int(geo[0][1] * processed_h / CONFIG.UPSCALE_FACTOR) + crop_offset_y
        x2 = int(geo[1][0] * processed_w / CONFIG.UPSCALE_FACTOR) + crop_offset_x
        y2 = int(geo[1][1] * processed_h / CONFIG.UPSCALE_FACTOR) + crop_offset_y
        text = item['value']
        conf = item['confidence']
        ocr_results_list.append({
            "text": text,
            "bbox": (x1, y1, x2, y2),
            "confidence": conf
        })
        print(f"   [{idx}/{len(final_results)}] Text: '{text}' | Conf: {conf:.2f} | BBox: ({x1}, {y1}, {x2}, {y2})")
    
    print(f"\nâœ… Recognized {len(ocr_results_list)} text blocks.")

    print("\nDEBUG: All detected text blocks:")
    for ocr_res in ocr_results_list:
        print(f"- Text: '{ocr_res['text']}', Confidence: {ocr_res['confidence']:.2f}, BBox: {ocr_res['bbox']}")
    
    # --- STAGE B.5: SYMBOL-GUIDED OCR (PaddleOCR via Microservice) ---
    symbol_guided_results = []
    symbol_guided_timing = {}
    symbol_guided_paddle_texts = []  # Store all PaddleOCR texts for logging
    
    if PADDLE_OCR_CLIENT_AVAILABLE:
        print("\n--- Running Symbol-Guided OCR (PaddleOCR Microservice) ---")
        print(f"   Processing {len(yolo_data)} symbols...")
        start_time = time.time()
        
        for idx, symbol in enumerate(yolo_data, 1):
            try:
                print(f"   [{idx}/{len(yolo_data)}] Processing symbol: {symbol['class_name']} (conf: {symbol['conf']:.2f})")
                
                # Crop ROI from original image
                x1, y1, x2, y2 = symbol["bbox"]
                
                # Ensure bbox is within image bounds
                img_h, img_w = original_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                
                print(f"      BBox: ({x1}, {y1}, {x2}, {y2}) | Size: {x2-x1}x{y2-y1}")
                
                # Crop ROI
                roi = original_img[y1:y2, x1:x2]
                
                # Skip if ROI is too small
                if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                    print(f"      âš ï¸ ROI too small, skipping...")
                    symbol_guided_results.append({
                        "symbol_idx": idx-1,
                        "tag": None,
                        "tag_confidence": 0.0,
                        "is_valid": False,
                        "raw_ocr_text": "",
                        "error": "ROI_TOO_SMALL"
                    })
                    continue
                
                print(f"      ðŸ“¤ Sending ROI to PaddleOCR microservice...")
                
                # Call PaddleOCR microservice
                ocr_result = call_paddle_ocr_service(
                    roi,
                    symbol_bbox=None,  # ROI is already cropped
                    meta={
                        "symbol_idx": idx-1,
                        "symbol_type": symbol["class_name"],
                        "symbol_confidence": symbol["conf"]
                    }
                )
                
                # Process result
                if ocr_result.get("service_available"):
                    if ocr_result.get("text"):
                        # Success - text detected
                        text_result = ocr_result.get('text', '')
                        conf_result = ocr_result.get('confidence', 0.0)
                        valid_result = ocr_result.get('valid', False)
                        print(f"      âœ… OCR Result: '{text_result}' (conf: {conf_result:.2f}, valid: {valid_result})")
                        symbol_guided_paddle_texts.append({
                            "symbol_idx": idx-1,
                            "symbol_type": symbol["class_name"],
                            "raw_text": ocr_result.get("raw_text", ""),
                            "processed_text": ocr_result.get("text", ""),
                            "confidence": ocr_result.get("confidence", 0.0),
                            "is_valid": ocr_result.get("valid", False)
                        })
                        
                        symbol_guided_results.append({
                            "symbol_idx": idx-1,
                            "tag": ocr_result.get("text"),
                            "tag_confidence": ocr_result.get("confidence", 0.0),
                            "is_valid": ocr_result.get("valid", False),
                            "raw_ocr_text": ocr_result.get("raw_text", ""),
                            "corrections_applied": [],
                            "bbox": symbol["bbox"]
                        })
                    else:
                        # Service available but no text detected
                        print(f"      âš ï¸ No text detected (error: {ocr_result.get('error', 'NO_TEXT_DETECTED')})")
                        symbol_guided_results.append({
                            "symbol_idx": idx-1,
                            "tag": None,
                            "tag_confidence": 0.0,
                            "is_valid": False,
                            "raw_ocr_text": "",
                            "error": ocr_result.get("error", "NO_TEXT_DETECTED")
                        })
                else:
                    # Service unavailable - fallback to existing OCR
                    print(f"      âŒ Service unavailable (error: {ocr_result.get('error', 'SERVICE_UNAVAILABLE')})")
                    symbol_guided_results.append({
                        "symbol_idx": idx-1,
                        "tag": None,
                        "tag_confidence": 0.0,
                        "is_valid": False,
                        "raw_ocr_text": "",
                        "error": "SERVICE_UNAVAILABLE"
                    })
                    
            except Exception as e:
                print(f"      âŒ Symbol-guided OCR failed for symbol {idx}: {e}")
                traceback.print_exc()
                symbol_guided_results.append({
                    "symbol_idx": idx-1,
                    "tag": None,
                    "tag_confidence": 0.0,
                    "is_valid": False,
                    "raw_ocr_text": "",
                    "error": str(e)
                })
        
        elapsed_time = time.time() - start_time
        symbol_guided_timing = {
            "total_time": elapsed_time,
            "time_per_symbol": elapsed_time / len(yolo_data) if yolo_data else 0,
            "symbols_processed": len(yolo_data)
        }
        print(f"âœ… Symbol-guided OCR completed: {len(symbol_guided_results)} symbols processed in {elapsed_time:.2f}s")
        
        # Display results
        if symbol_guided_paddle_texts:
            print(f"\nDEBUG: PaddleOCR texts (Symbol-Guided) - {len(symbol_guided_paddle_texts)} detections:")
            for item in symbol_guided_paddle_texts:
                print(f"  - Symbol {item['symbol_idx']} ({item['symbol_type']}): '{item['raw_text']}' -> '{item['processed_text']}' (conf: {item['confidence']:.2f}, valid: {item['is_valid']})")
        else:
            print("DEBUG: No PaddleOCR texts detected (Symbol-Guided)")
    else:
        print("âš ï¸ Symbol-guided OCR skipped (client not available)")
        symbol_guided_results = []
        symbol_guided_timing = {}
    
    # --- STAGE B.6: DOCTR-GUIDED OCR (PaddleOCR using DocTR text bboxes) ---
    doctr_guided_results = []
    doctr_guided_timing = {}
    doctr_guided_paddle_texts = []  # Store all PaddleOCR texts for logging
    
    if PADDLE_OCR_CLIENT_AVAILABLE and ocr_results_list:
        print("\n--- Running DocTR-Guided OCR (PaddleOCR Microservice) ---")
        print(f"   Processing {len(ocr_results_list)} DocTR-detected text regions...")
        start_time = time.time()
        
        # For each DocTR-detected text, use its bbox as ROI for PaddleOCR
        for idx, ocr_res in enumerate(ocr_results_list, 1):
            try:
                print(f"   [{idx}/{len(ocr_results_list)}] DocTR Text: '{ocr_res['text']}' (conf: {ocr_res['confidence']:.2f})")
                
                # Use DocTR text bbox as ROI for PaddleOCR
                x1, y1, x2, y2 = ocr_res["bbox"]
                # Expand slightly to ensure we capture the text
                padding = 10
                expanded_bbox = (
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(original_img.shape[1], x2 + padding),
                    min(original_img.shape[0], y2 + padding)
                )
                
                print(f"      Original BBox: ({x1}, {y1}, {x2}, {y2}) | Expanded: ({expanded_bbox[0]}, {expanded_bbox[1]}, {expanded_bbox[2]}, {expanded_bbox[3]})")
                
                # Crop ROI from original image
                roi = original_img[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                
                # Skip if ROI is too small
                if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                    print(f"      âš ï¸ ROI too small, skipping...")
                    doctr_guided_results.append({
                        "doctr_idx": idx-1,
                        "doctr_text": ocr_res["text"],
                        "doctr_confidence": ocr_res["confidence"],
                        "paddle_tag": None,
                        "paddle_confidence": 0.0,
                        "is_valid": False,
                        "raw_ocr_text": "",
                        "error": "ROI_TOO_SMALL",
                        "bbox": ocr_res["bbox"]
                    })
                    continue
                
                print(f"      ðŸ“¤ Sending ROI to PaddleOCR microservice...")
                
                # Call PaddleOCR microservice
                ocr_result = call_paddle_ocr_service(
                    roi,
                    symbol_bbox=None,  # ROI is already cropped
                    meta={
                        "doctr_idx": idx-1,
                        "doctr_text": ocr_res["text"],
                        "source": "doctr_guided"
                    }
                )
                
                # Process result
                if ocr_result.get("service_available") and ocr_result.get("text"):
                    print(f"      âœ… PaddleOCR Result: '{ocr_result.get('text')}' (conf: {ocr_result.get('confidence', 0.0):.2f}, valid: {ocr_result.get('valid', False)})")
                    print(f"         DocTR: '{ocr_res['text']}' â†’ PaddleOCR: '{ocr_result.get('text')}'")
                    doctr_guided_paddle_texts.append({
                        "doctr_idx": idx-1,
                        "doctr_text": ocr_res["text"],
                        "doctr_confidence": ocr_res["confidence"],
                        "paddle_raw_text": ocr_result.get("raw_text", ""),
                        "paddle_processed_text": ocr_result.get("text", ""),
                        "paddle_confidence": ocr_result.get("confidence", 0.0),
                        "is_valid": ocr_result.get("valid", False)
                    })
                    
                    final_paddle_tag = ocr_result.get("text")
                    
                    doctr_guided_results.append({
                        "doctr_idx": idx-1,
                        "doctr_text": ocr_res["text"],
                        "doctr_confidence": ocr_res["confidence"],
                        "paddle_tag": final_paddle_tag,
                        "paddle_confidence": ocr_result.get("confidence", 0.0),
                        "is_valid": ocr_result.get("valid", False),
                        "raw_ocr_text": ocr_result.get("raw_text", ""),
                        "bbox": ocr_res["bbox"]
                    })
                else:
                    # Service unavailable or no text detected
                    print(f"      âš ï¸ No PaddleOCR text detected (error: {ocr_result.get('error', 'NO_TEXT_DETECTED')})")
                    doctr_guided_results.append({
                        "doctr_idx": idx-1,
                        "doctr_text": ocr_res["text"],
                        "doctr_confidence": ocr_res["confidence"],
                        "paddle_tag": None,
                        "paddle_confidence": 0.0,
                        "is_valid": False,
                        "raw_ocr_text": "",
                        "error": ocr_result.get("error", "NO_TEXT_DETECTED"),
                        "bbox": ocr_res["bbox"]
                    })
            except Exception as e:
                print(f"      âŒ DocTR-guided OCR failed for text {idx}: {e}")
                traceback.print_exc()
                doctr_guided_results.append({
                    "doctr_idx": idx-1,
                    "doctr_text": ocr_res["text"],
                    "doctr_confidence": ocr_res["confidence"],
                    "paddle_tag": None,
                    "paddle_confidence": 0.0,
                    "is_valid": False,
                    "error": str(e)
                })
        
        elapsed_time = time.time() - start_time
        doctr_guided_timing = {
            "total_time": elapsed_time,
            "time_per_text": elapsed_time / len(ocr_results_list) if ocr_results_list else 0,
            "texts_processed": len(ocr_results_list)
        }
        print(f"âœ… DocTR-guided OCR completed: {len(doctr_guided_results)} texts processed in {elapsed_time:.2f}s")
        
        # Display all PaddleOCR texts (DocTR-Guided)
        if doctr_guided_paddle_texts:
            print(f"\nDEBUG: PaddleOCR texts (DocTR-Guided) - {len(doctr_guided_paddle_texts)} detections:")
            for item in doctr_guided_paddle_texts:
                print(f"  - DocTR '{item['doctr_text']}' (conf: {item['doctr_confidence']:.2f}) -> PaddleOCR: '{item['paddle_raw_text']}' -> '{item['paddle_processed_text']}' (conf: {item['paddle_confidence']:.2f}, valid: {item['is_valid']})")
        else:
            print("DEBUG: No PaddleOCR texts detected (DocTR-Guided)")
    else:
        print("âš ï¸ DocTR-guided OCR skipped (not available)")
    
    # --- STAGE C: LINE AND ARROW DETECTION ---
    # Dual pipeline:
    #   PASS 1 – HoughLinesP on full original image (masks symbols + OCR)
    #            → solid pipe segments for the topology graph
    #   PASS 2 – LSD on cropped image (existing dotted-line detection)
    #            → dotted/dashed segments appended to line_data for rendering

    # --- PASS 1: SOLID PIPES via HoughLinesP (same approach as oldapp.py) ---
    gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Mask symbols and OCR so pipe lines stand alone.
    hough_mask = np.ones_like(gray_orig) * 255
    for symbol in yolo_data:
        x1, y1, x2, y2 = symbol["bbox"]
        hough_mask[y1:y2, x1:x2] = 0
    for ocr_res in ocr_results_list:
        x1, y1, x2, y2 = ocr_res["bbox"]
        hough_mask[y1:y2, x1:x2] = 0
    gray_masked = cv2.bitwise_and(gray_orig, hough_mask)

    # Half-scale → skeletonize → Canny → HoughLinesP (identical to oldapp.py).
    _hough_scale = 0.5
    img_small = cv2.resize(gray_masked, None, fx=_hough_scale, fy=_hough_scale)
    _, binary_small = cv2.threshold(img_small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton_small = skeletonize(binary_small // 255).astype(np.uint8) * 255
    edges_small = cv2.Canny(skeleton_small, 75, 200)

    raw_hough = cv2.HoughLinesP(
        edges_small, 1, np.pi / 180,
        threshold=60, minLineLength=75, maxLineGap=20
    )

    def _hough_preprocess(hough_lines, scale, extend_by=15):
        """Extend endpoints, merge collinear segments, rescale — identical to oldapp.py."""
        if hough_lines is None:
            return []
        processed = []
        for seg in hough_lines:
            x1, y1, x2, y2 = seg[0]
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                dx /= length; dy /= length
                x1 = int(x1 - dx * extend_by); y1 = int(y1 - dy * extend_by)
                x2 = int(x2 + dx * extend_by); y2 = int(y2 + dy * extend_by)
            processed.append((x1, y1, x2, y2))

        horizontal = sorted([l for l in processed if abs(l[2]-l[0]) > abs(l[3]-l[1])], key=lambda l: (l[1], l[0]))
        vertical   = sorted([l for l in processed if abs(l[2]-l[0]) <= abs(l[3]-l[1])], key=lambda l: (l[0], l[1]))
        processed  = horizontal + vertical

        def _collinear(l1, l2, angle_tol=3, dist_tol=15):
            x1, y1, x2, y2 = l1; x3, y3, x4, y4 = l2
            v1 = np.array([x2-x1, y2-y1], dtype=float)
            v2 = np.array([x4-x3, y4-y3], dtype=float)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                return False
            cos_a = np.dot(v1, v2) / (n1 * n2)
            angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            if angle > 180: angle = 360 - angle
            if angle > angle_tol:
                return False
            return min(LineString([(x1,y1),(x2,y2)]).distance(Point(p)) for p in [(x3,y3),(x4,y4)]) < dist_tol

        merged = []
        while processed:
            curr = processed.pop(0)
            i = 0
            while i < len(processed):
                if _collinear(curr, processed[i]):
                    x1, y1, x2, y2 = curr; x3, y3, x4, y4 = processed.pop(i)
                    pts = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                    xs, ys = zip(*pts)
                    curr = (min(xs), min(ys), max(xs), max(ys))
                else:
                    i += 1
            merged.append(curr)
        return [(int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)) for x1,y1,x2,y2 in merged]

    solid_lines = _hough_preprocess(raw_hough, scale=_hough_scale)
    print(f"✅ HoughLinesP detected {len(solid_lines)} solid line segments.")

    # --- PASS 1.5: SOLID PIPES via LSD (precise geometry for DXF + rendering) ---
    # lsd_solid_input is computed on the full gray_masked so it keeps the same shape as
    # final_img — it is returned as 'skeleton' and used for the overlay on line 2551.
    lsd_solid_input = preprocess_for_centerline_lsd(gray_masked)
    # Crop gray_masked to the same notes-free region used by PASS 2 (img_without_notes),
    # so the metadata/title-block area at the bottom-right is excluded from LSD detection.
    _lsd_cx, _lsd_cy = int(crop_offset_x), int(crop_offset_y)
    _lsd_h, _lsd_w = img_without_notes.shape[:2]
    gray_masked_no_notes = gray_masked[_lsd_cy:_lsd_cy + _lsd_h, _lsd_cx:_lsd_cx + _lsd_w]
    lsd_solid_input_crop = preprocess_for_centerline_lsd(gray_masked_no_notes)
    lsd_solid_raw_crop = detect_lines_lsd(lsd_solid_input_crop, min_length=20)
    # Translate coordinates back to original image space
    lsd_solid_raw = [(x1 + _lsd_cx, y1 + _lsd_cy, x2 + _lsd_cx, y2 + _lsd_cy) for (x1, y1, x2, y2) in lsd_solid_raw_crop]
    print(f"✅ LSD solid detected {len(lsd_solid_raw)} segments for rendering/DXF.")

    # --- PASS 2: DOTTED/DASHED PIPES via LSD (kept unchanged) ---
    line_crop_offset_x, line_crop_offset_y = crop_offset_x, crop_offset_y
    gray_img = cv2.cvtColor(img_without_notes, cv2.COLOR_BGR2GRAY)
    masked_img = gray_img.copy()
    crop_h, crop_w = masked_img.shape[:2]

    def _shift_bbox_to_line_crop(bbox):
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        x1 -= int(line_crop_offset_x); x2 -= int(line_crop_offset_x)
        y1 -= int(line_crop_offset_y); y2 -= int(line_crop_offset_y)
        x1 = max(0, min(crop_w-1, x1)); y1 = max(0, min(crop_h-1, y1))
        x2 = max(0, min(crop_w-1, x2)); y2 = max(0, min(crop_h-1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    for symbol in yolo_data:
        shifted = _shift_bbox_to_line_crop(symbol.get("bbox"))
        if shifted:
            sx1, sy1, sx2, sy2 = shifted
            p = 4
            mx1, my1 = max(0, sx1-p), max(0, sy1-p)
            mx2, my2 = min(crop_w-1, sx2+p), min(crop_h-1, sy2+p)
            cv2.rectangle(masked_img, (mx1, my1), (mx2, my2), 255, -1)
            bw, bh = max(1, mx2-mx1), max(1, my2-my1)
            if bw >= bh:
                bx = (mx1+mx2)//2
                cv2.line(masked_img, (bx, my1), (bx, my2), 0, thickness=2)
            else:
                by = (my1+my2)//2
                cv2.line(masked_img, (mx1, by), (mx2, by), 0, thickness=2)

    for text in ocr_results_list:
        shifted = _shift_bbox_to_line_crop(text.get("bbox"))
        if shifted:
            tx1, ty1, tx2, ty2 = shifted
            p = 4
            cv2.rectangle(masked_img,
                          (max(0, tx1-p), max(0, ty1-p)),
                          (min(crop_w-1, tx2+p), min(crop_h-1, ty2+p)), 255, -1)

    # Mask LSD solid lines (converted to crop coords) so dotted LSD ignores them.
    for line in lsd_solid_raw:
        lx1 = int(line[0] - line_crop_offset_x); ly1 = int(line[1] - line_crop_offset_y)
        lx2 = int(line[2] - line_crop_offset_x); ly2 = int(line[3] - line_crop_offset_y)
        cv2.line(masked_img, (lx1, ly1), (lx2, ly2), 255, thickness=7)
    print(f"LSD solid lines masked before dotted pass: {len(lsd_solid_raw)}")

    dotted_input = preprocess_for_dotted_lsd(masked_img)
    dotted_raw_lines = detect_lines_lsd(dotted_input, min_length=5)
    print(f"Dotted raw segments detected: {len(dotted_raw_lines)}")
    dotted_lines_cropped = group_dotted_segments(dotted_raw_lines, binary_img=dotted_input)
    dotted_lines_cropped = collapse_parallel_dotted_lines(dotted_lines_cropped)

    def _line_key(line):
        x1, y1, x2, y2 = map(int, line)
        return tuple(sorted(((x1, y1), (x2, y2))))

    solid_keys = {_line_key(l) for l in solid_lines}
    dotted_lines_cropped = [l for l in dotted_lines_cropped if _line_key(l) not in solid_keys]

    def _to_original_space(lines):
        return [(int(x1+line_crop_offset_x), int(y1+line_crop_offset_y),
                 int(x2+line_crop_offset_x), int(y2+line_crop_offset_y))
                for (x1, y1, x2, y2) in lines]

    dotted_lines = _to_original_space(dotted_lines_cropped)

    # hough_line_data     → topology graph only (HoughLinesP: connected merged segments)
    # line_data           → rendering + DXF (LSD solid: precise pixel-accurate geometry)
    seen_hough = set()
    hough_line_data = []
    for line in solid_lines:
        key = _line_key(line)
        if key not in seen_hough:
            seen_hough.add(key)
            hough_line_data.append(tuple(map(int, line)))

    seen_render = set()
    lsd_solid_data = []
    for line in lsd_solid_raw:
        key = _line_key(line)
        if key not in seen_render:
            seen_render.add(key)
            lsd_solid_data.append(tuple(map(int, line)))

    line_data = list(lsd_solid_data)
    for line in dotted_lines:
        key = _line_key(line)
        if key not in seen_render:
            seen_render.add(key)
            line_data.append(tuple(map(int, line)))

    # topology_line_data = HoughLinesP solid + dotted (preserves working graph)
    seen_topo = set(seen_hough)
    topology_line_data = list(hough_line_data)
    for line in dotted_lines:
        key = _line_key(line)
        if key not in seen_topo:
            seen_topo.add(key)
            topology_line_data.append(tuple(map(int, line)))

    print(f"✅ Detected {len(hough_line_data)} solid lines (HoughLinesP) for topology.")
    print(f"✅ Detected {len(lsd_solid_data)} solid lines (LSD) for rendering/DXF.")
    print(f"✅ Detected {len(dotted_lines)} dotted lines (LSD).")
    print(f"✅ Detected {len(line_data)} total line segments for rendering.")
    print(f"✅ Detected 0 arrows.")
    
    # --- STAGE D: GRAPH CONSTRUCTION (WITH BOTH OCR METHODS) ---
    G = nx.Graph()
    bom_counts = Counter()

    # Prefix map: YOLO class name keywords → short ID prefix
    _SYMBOL_PREFIX_MAP = {
        'gate_valve': 'GV', 'ball_valve': 'BV', 'butterfly_valve': 'BFV',
        'check_valve': 'CV', 'control_valve': 'FCV', 'relief_valve': 'RV',
        'safety_valve': 'SV', 'needle_valve': 'NV', 'globe_valve': 'GLV',
        'valve': 'VLV',
        'centrifugal_pump': 'CP', 'pump': 'PMP',
        'compressor': 'CMP', 'blower': 'BLW',
        'heat_exchanger': 'HE', 'cooler': 'CLR', 'heater': 'HTR',
        'vessel': 'VSL', 'tank': 'TK', 'drum': 'DRM', 'column': 'COL',
        'filter': 'FLT', 'strainer': 'STR', 'separator': 'SEP',
        'flow_indicator': 'FI', 'flow_transmitter': 'FT', 'flow_controller': 'FC',
        'pressure_indicator': 'PI', 'pressure_transmitter': 'PT', 'pressure_controller': 'PC',
        'temperature_indicator': 'TI', 'temperature_transmitter': 'TT',
        'level_indicator': 'LI', 'level_transmitter': 'LT', 'level_controller': 'LC',
        'instrument': 'INST', 'transmitter': 'TX', 'controller': 'CTL',
        'motor': 'MOT', 'turbine': 'TRB', 'agitator': 'AGT',
        'reducer': 'RED', 'expander': 'EXP', 'ejector': 'EJT',
    }

    def _symbol_prefix(class_name: str) -> str:
        lower = class_name.lower()
        for keyword, prefix in _SYMBOL_PREFIX_MAP.items():
            if keyword in lower:
                return prefix
        # fallback: first 3 uppercase chars of class_name
        clean = ''.join(c for c in class_name.upper() if c.isalpha())
        return clean[:4] if len(clean) >= 4 else clean or 'SYM'

    _prefix_counters: dict = {}

    node_id_counter = 1
    symbol_nodes = {}
    ocr_comparison_data = []  # NEW: Store comparison data
    
    for idx, symbol in enumerate(yolo_data):
        node_id = node_id_counter
        x1, y1, x2, y2 = symbol["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # METHOD 1: Proximity-based matching with DocTR (OLD)
        best_tag_data = min(
            [(ocr_res, np.sqrt(((x1+x2)/2 - (ocr_res["bbox"][0]+ocr_res["bbox"][2])/2)**2 + ((y1+y2)/2 - (ocr_res["bbox"][1]+ocr_res["bbox"][3])/2)**2))
             for ocr_res in ocr_results_list],
            key=lambda x: x[1],
            default=(None, float('inf'))
        )
        old_tag, old_tag_conf = (best_tag_data[0]["text"], best_tag_data[0]["confidence"]) if best_tag_data[0] and best_tag_data[1] < 300 else (None, 0.0)
        
        # METHOD 2: Symbol-guided OCR (PaddleOCR using symbol bbox)
        symbol_guided_tag = None
        symbol_guided_conf = 0.0
        symbol_guided_valid = False
        if PADDLE_OCR_CLIENT_AVAILABLE and idx < len(symbol_guided_results):
            sg_result = symbol_guided_results[idx]
            symbol_guided_tag = sg_result.get("tag")
            symbol_guided_conf = sg_result.get("tag_confidence", 0.0)
            symbol_guided_valid = sg_result.get("is_valid", False)
        
        # METHOD 3: DocTR-guided OCR (PaddleOCR using DocTR text bbox)
        doctr_guided_tag = None
        doctr_guided_conf = 0.0
        doctr_guided_valid = False
        # Find closest DocTR text and get its PaddleOCR result
        if doctr_guided_results:
            best_doctr_match = min(
                [(dg_res, np.sqrt(((x1+x2)/2 - (dg_res["bbox"][0]+dg_res["bbox"][2])/2)**2 + ((y1+y2)/2 - (dg_res["bbox"][1]+dg_res["bbox"][3])/2)**2))
                 for dg_res in doctr_guided_results],
                key=lambda x: x[1],
                default=(None, float('inf'))
            )
            if best_doctr_match[0] and best_doctr_match[1] < 300:
                dg_result = best_doctr_match[0]
                doctr_guided_tag = dg_result.get("paddle_tag")
                doctr_guided_conf = dg_result.get("paddle_confidence", 0.0)
                doctr_guided_valid = dg_result.get("is_valid", False)
        
        # COMPARISON: Choose best tag from all three methods
        candidates = [
            (symbol_guided_tag, symbol_guided_conf, "symbol_guided", symbol_guided_valid),
            (doctr_guided_tag, doctr_guided_conf, "doctr_guided", doctr_guided_valid),
            (old_tag, old_tag_conf, "proximity_matching", True)
        ]
        
        # Filter valid candidates
        valid_candidates = [c for c in candidates if c[0] and c[3] and c[1] >= CONFIG.TAG_CONFIDENCE_REVIEW_THRESHOLD]
        
        if valid_candidates:
            # Prefer valid, high-confidence tags
            best_candidate = max(valid_candidates, key=lambda x: x[1])
            best_tag, best_tag_conf, tag_source, _ = best_candidate
        else:
            # Fallback: use highest confidence regardless of validity
            all_candidates = [c for c in candidates if c[0]]
            if all_candidates:
                best_candidate = max(all_candidates, key=lambda x: x[1])
                best_tag, best_tag_conf, tag_source, _ = best_candidate
            else:
                best_tag, best_tag_conf, tag_source = None, 0.0, "none"
        
        # Store comparison data for all three methods
        ocr_comparison_data.append({
            "node_id": node_id,
            "symbol_type": symbol["class_name"],
            "method_1_proximity": {
                "tag": old_tag,
                "confidence": old_tag_conf,
                "method": "proximity_matching"
            },
            "method_2_symbol_guided": {
                "tag": symbol_guided_tag,
                "confidence": symbol_guided_conf,
                "is_valid": symbol_guided_valid,
                "method": "symbol_guided"
            },
            "method_3_doctr_guided": {
                "tag": doctr_guided_tag,
                "confidence": doctr_guided_conf,
                "is_valid": doctr_guided_valid,
                "method": "doctr_guided"
            },
            "selected_tag": best_tag,
            "selected_confidence": best_tag_conf,
            "selected_source": tag_source,
            "tags_match": len(set([t for t in [old_tag, symbol_guided_tag, doctr_guided_tag] if t])) > 1
        })
        
        needs_review = best_tag_conf < CONFIG.TAG_CONFIDENCE_REVIEW_THRESHOLD or best_tag is None
        symbol_confidence = symbol["conf"]
        if symbol_confidence < CONFIG.SYMBOL_CONFIDENCE_REVIEW_THRESHOLD:
            needs_review = True  # Flag ambiguous symbols
        
        # Generate a unique auto-ID for every symbol (used as fallback tag when OCR finds nothing)
        _prefix = _symbol_prefix(symbol["class_name"])
        _prefix_counters[_prefix] = _prefix_counters.get(_prefix, 0) + 1
        auto_id = f"{_prefix}-{_prefix_counters[_prefix]:03d}"
        effective_tag = best_tag if best_tag else auto_id
        effective_tag_source = tag_source if best_tag else "auto_generated"

        attrs = {
            "type": symbol["class_name"],
            "tag": effective_tag,
            "tag_confidence": best_tag_conf,
            "tag_source": effective_tag_source,
            "symbol_confidence": symbol_confidence,
            "needs_review": needs_review,
            "auto_id": auto_id,
            "coords": (cx, cy),
            "bbox": symbol["bbox"]
        }
        G.add_node(node_id, **attrs)
        symbol_nodes[node_id] = attrs
        bom_counts[symbol["class_name"]] += 1
        node_id_counter += 1
    
    topology_graph = build_line_topology_graph(
        line_data=topology_line_data,
        intersection_snap_threshold=10
    )
    print(
        f"Built line topology with {topology_graph.number_of_nodes()} nodes "
        f"and {topology_graph.number_of_edges()} edges."
    )
    # --- DIAGNOSTIC 1: graph connectivity ---
    components = list(nx.connected_components(topology_graph))
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"  Graph components: {len(components)} total, "
          f"largest={comp_sizes[0] if comp_sizes else 0} nodes, "
          f"sizes (top-10): {comp_sizes[:10]}")
    if len(components) > 1:
        print(f"  WARNING: {len(components)-1} isolated sub-graphs — DFS cannot cross these gaps.")

    symbol_to_topology_nodes, topology_node_to_symbols = attach_symbols_to_topology(
        symbol_nodes=symbol_nodes,
        topology_graph=topology_graph,
        attach_threshold=24,
        fallback_threshold=60,
    )
    classify_topology_nodes(topology_graph)

    # --- DIAGNOSTIC 2: attachment quality ---
    zero_anchor = [sid for sid in symbol_nodes if not symbol_to_topology_nodes.get(sid)]
    attached_symbols = len(symbol_nodes) - len(zero_anchor)
    avg_anchors = (
        sum(len(v) for v in symbol_to_topology_nodes.values()) / max(attached_symbols, 1)
    )
    print(f"Attached {attached_symbols}/{len(symbol_nodes)} symbols to line topology "
          f"(avg {avg_anchors:.1f} anchor nodes each).")
    if zero_anchor:
        print(f"  Unattached symbol IDs: {zero_anchor}")
    # Recompute components: attach_symbols_to_topology may have added new nodes
    components = list(nx.connected_components(topology_graph))
    node_to_comp = {}
    for idx, comp in enumerate(components):
        for n in comp:
            node_to_comp[n] = idx
    comp_symbol_counts = defaultdict(int)
    for sid, anchors in symbol_to_topology_nodes.items():
        if anchors:
            comp_symbol_counts[node_to_comp.get(next(iter(anchors)), -1)] += 1
    print(f"  Symbols per component: { {f'comp{k}': v for k,v in sorted(comp_symbol_counts.items())} }")

    symbol_adjacency = derive_symbol_adjacency_from_topology(
        topology_graph=topology_graph,
        symbol_to_topology_nodes=symbol_to_topology_nodes,
        topology_node_to_symbols=topology_node_to_symbols,
    )
    connected_pairs = sum(len(v) for v in symbol_adjacency.values())
    print(f"Symbol adjacency: {len(symbol_adjacency)} symbols with connections, {connected_pairs // 2} unique pairs.")
    for node_id, neighbors in symbol_adjacency.items():
        for neighbor_id in neighbors:
            if node_id != neighbor_id and not G.has_edge(node_id, neighbor_id):
                G.add_edge(node_id, neighbor_id, direction="unknown")

    def propagate_flow(G):
        potential_sources = [n for n in G if G.nodes[n]["type"] in ["motor", "silencer"]]
        if not potential_sources:
            potential_sources = [min(G.nodes)]
        for start in potential_sources:
            queue = deque([(start, "downstream")])
            visited = set()
            while queue:
                node, direction = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        G.edges[node, neighbor]["direction"] = direction
                        queue.append((neighbor, direction))
    
    propagate_flow(G)
    
    # --- Validate Connections and Build Graph Data ---
    node_ids = set(G.nodes)
    missing_tags = 0
    broken_connections = 0
    ambiguous_symbols = 0
    final_img = original_img.copy()
    for line in lsd_solid_data:
        cv2.line(final_img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
    for line in dotted_lines:
        cv2.line(final_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
    graph_data_json = []
    for node_id in G.nodes:
        data = G.nodes[node_id]
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        tag_text = f"{data['tag'] or 'NO_TAG'}[{data['tag_confidence']:.2f}]" if data['tag'] else "NO_TAG"
        cv2.putText(final_img, f"{node_id}:{tag_text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        connections = sorted(int(n) for n in G.neighbors(node_id))
        # Validate connections
        valid_connections = [n for n in connections if n in node_ids and n != node_id]
        connection_valid = len(valid_connections) == len(connections)
        if not connection_valid:
            broken_connections += 1
        if data["tag"] is None:
            missing_tags += 1
        if data["symbol_confidence"] < CONFIG.SYMBOL_CONFIDENCE_REVIEW_THRESHOLD:
            ambiguous_symbols += 1
        node_data = {
            "node_id": node_id,
            "type": data['type'],
            "tag": data['tag'],
            "auto_id": data.get('auto_id', ''),
            "tag_confidence": data['tag_confidence'],
            "symbol_confidence": data['symbol_confidence'],
            "needs_review": data['needs_review'],
            "connections": valid_connections,
            "connection_valid": connection_valid,
            "bbox": data["bbox"]
        }
        graph_data_json.append(node_data)
    for u, v in G.edges:
        if u < v:
            start_point = (int(G.nodes[u]["coords"][0]), int(G.nodes[u]["coords"][1]))
            end_point = (int(G.nodes[v]["coords"][0]), int(G.nodes[v]["coords"][1]))
            dir = G.edges[u, v].get("direction", "unknown")
            if dir == "downstream":
                cv2.arrowedLine(final_img, start_point, end_point, (0, 255, 0), 3)
            elif dir == "upstream":
                cv2.arrowedLine(final_img, end_point, start_point, (0, 255, 0), 3)
            else:
                cv2.line(final_img, start_point, end_point, (0, 255, 0), 3)
    
    review_summary = {
        "missing_tags": missing_tags,
        "broken_connections": broken_connections,
        "ambiguous_symbols": ambiguous_symbols
    }
    
    # NEW: Add OCR comparison data to return (all three methods)
    ocr_comparison = {
        "symbol_guided_available": PADDLE_OCR_CLIENT_AVAILABLE,
        "comparison_data": ocr_comparison_data,
        "timing": {
            "symbol_guided": symbol_guided_timing,
            "doctr_guided": doctr_guided_timing
        },
        "statistics": {
            "total_symbols": len(yolo_data),
            "method_1_proximity_tags": sum(1 for c in ocr_comparison_data if c["method_1_proximity"]["tag"]),
            "method_2_symbol_guided_tags": sum(1 for c in ocr_comparison_data if c["method_2_symbol_guided"]["tag"]),
            "method_3_doctr_guided_tags": sum(1 for c in ocr_comparison_data if c["method_3_doctr_guided"]["tag"]),
            "tags_match": sum(1 for c in ocr_comparison_data if c["tags_match"]),
            "method_1_selected": sum(1 for c in ocr_comparison_data if c["selected_source"] == "proximity_matching"),
            "method_2_selected": sum(1 for c in ocr_comparison_data if c["selected_source"] == "symbol_guided"),
            "method_3_selected": sum(1 for c in ocr_comparison_data if c["selected_source"] == "doctr_guided")
        },
        "paddle_texts": {
            "symbol_guided": symbol_guided_paddle_texts,
            "doctr_guided": doctr_guided_paddle_texts
        }
    }
    
    # --- STAGE E: GEOMETRY PAYLOAD (LINE-FIRST FOR DIRECT DXF) ---
    geometry_output = build_geometry_payload(
        graph_data_json=graph_data_json,
        line_data=line_data,
        symbol_nodes=symbol_nodes,
        image_height=original_img.shape[0],
        dotted_lines=dotted_lines
    )

    # --- STAGE F: METADATA EXTRACTION (TABLE & REGION-BASED DATA) ---
    metadata_output = {
        "tables": [],
        "markdown": "",
        "json_data": {},
        "available": METADATA_EXTRACTOR_AVAILABLE,
        "extraction_summary": {}
    }
    
    if METADATA_EXTRACTOR_AVAILABLE:
        try:
            logger.info("ðŸ” Starting metadata extraction...")
            
            # Initialize metadata extractor with general-text OCR endpoint.
            paddle_client = integrate_paddle_client(call_paddle_text_ocr_service)
            extractor = MetadataExtractor(paddle_ocr_client=paddle_client)
            
            # Extract all metadata from the image
            metadata_elements = extractor.extract_all_metadata(original_img)
            logger.info(f"âœ… Extracted {len(metadata_elements)} metadata elements")
            
            # Generate output formats
            metadata_output["markdown"] = extractor.get_markdown_output(metadata_elements)
            metadata_output["json_data"] = extractor.get_json_output(metadata_elements)
            metadata_output["available"] = True
            
            # Summary statistics
            json_summary = metadata_output["json_data"].get("summary", {})
            metadata_output["extraction_summary"] = {
                "total_elements": len(metadata_elements),
                "tables": sum(1 for e in metadata_elements if e.element_type.name == "TABLE"),
                "text_blocks": sum(1 for e in metadata_elements if e.element_type.name == "TEXT_BLOCK"),
                "specifications": sum(1 for e in metadata_elements if e.element_type.name == "SPECIFICATIONS"),
                "revision_tables": sum(1 for e in metadata_elements if e.element_type.name == "REVISION_TABLE"),
                "notes": sum(1 for e in metadata_elements if e.element_type.name == "NOTES"),
                "title_blocks": sum(1 for e in metadata_elements if e.element_type.name == "TITLE_BLOCK"),
                "notes_items_count": json_summary.get("notes_items_count", 0),
                "revision_rows_count": json_summary.get("revision_rows_count", 0),
                "title_fields_count": json_summary.get("title_fields_count", 0),
                "ocr_confidence_mean": json_summary.get("ocr_confidence_mean", 0.0),
            }
            
            logger.info(f"ðŸ“Š Metadata extraction complete: {metadata_output['extraction_summary']}")
            
        except Exception as e:
            logger.error(f"âŒ Metadata extraction failed: {str(e)}", exc_info=True)
            metadata_output["available"] = False
            metadata_output["extraction_summary"] = {"error": str(e)}
    else:
        logger.warning("âš ï¸ Metadata extraction not available (missing dependencies)")
        metadata_output["extraction_summary"] = {"message": "Metadata extraction module not loaded"}
    
    return (
        graph_data_json,
        final_img,
        lsd_solid_input,
        bom_counts,
        ocr_results_list,
        review_summary,
        ocr_comparison,
        metadata_output,
        geometry_output,
    )

# --- 4.5. ANALYSIS PERSISTENCE HELPERS ---

def _guess_file_type(filename):
    ext = Path(str(filename)).suffix.lower()
    if ext == '.pdf':
        return 'PDF'
    if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'):
        return 'Image'
    if ext in ('.dxf', '.dwg'):
        return 'CAD'
    return 'Image'


def _save_analysis_record(analysis_id, filename, duration_seconds, symbol_count,
                           image_width, image_height, graph_data_json, geometry_payload,
                           bom_counts, dxf_artifact, annotated_img):
    safe_id = secure_filename(str(analysis_id)) or str(analysis_id)
    annotated_url = None
    try:
        annotated_path = ANALYSES_IMAGES_DIR / f"{safe_id}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        annotated_url = f"/static/images/{safe_id}_annotated.jpg"
    except Exception as e:
        logger.warning(f"Could not save annotated image: {e}")

    record = {
        "analysis_id": analysis_id,
        "project_name": str(filename),
        "filename": str(filename),
        "timestamp": int(time.time()),
        "duration_seconds": float(duration_seconds),
        "symbol_count": int(symbol_count),
        "status": "completed",
        "method": "YOLO+DocTR",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "annotated_image_url": annotated_url,
        "graph_data": {"json": graph_data_json},
        "geometry": geometry_payload,
        "bill_of_materials": [{"symbol": k, "count": v} for k, v in bom_counts.items()],
        "dxf": dxf_artifact,
    }
    record_path = ANALYSES_DIR / f"{safe_id}.json"
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    return record


def _list_analysis_records():
    records = []
    for path in sorted(ANALYSES_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rec = json.load(f)
            records.append(rec)
        except Exception:
            continue
    return records


# --- 5. DEFINE THE WEB SERVER ENDPOINTS ---
@app.route('/digitize', methods=['POST'])
def digitize():
    print("Received request for /digitize")
    print("â³ Processing started. This may take a while...")
    try:
        _digitize_start = time.time()
        if 'file' not in request.files:
            print("âŒ No file part in request")
            return jsonify({"error": "No file part in request"}), 400
        file = request.files['file']
        if file.filename == '':
            print("âŒ No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        file_bytes = file.read()
        print(f"Received file: {file.filename}, size: {len(file_bytes)} bytes")
        poppler_path = os.getenv('POPPLER_PATH', None)
        if file.filename.lower().endswith('.pdf'):
            try:
                images = convert_from_bytes(file_bytes, first_page=1, last_page=1, poppler_path=poppler_path)
                if not images:
                    print("âŒ Failed to process PDF: No images extracted")
                    raise ValueError("Could not process PDF: No images extracted")
                original_img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"âŒ PDF processing error: {str(e)}")
                return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 400
        else:
            original_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if original_img is None:
                print("âŒ Failed to decode image")
                return jsonify({"error": "Failed to decode image: Invalid or corrupted file"}), 400

        (
            graph_data_json,
            final_img,
            skeleton,
            bom_counts,
            ocr_results_list,
            review_summary,
            ocr_comparison,
            metadata_output,
            geometry_payload,
        ) = digitize_pid_image(original_img)
        
        final_img[skeleton > 0] = [200, 200, 200]
        
        _, buffer = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        _, input_buffer = cv2.imencode('.jpg', original_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        input_img_base64 = base64.b64encode(input_buffer).decode('utf-8')
        image_height, image_width = original_img.shape[:2]
        
        ocr_results_for_response = [
            {"text": r["text"], "bbox": r["bbox"], "confidence": r["confidence"]}
            for r in ocr_results_list
        ]

        analysis_id = _build_analysis_id()
        dxf_geometry_payload = build_dxf_geometry_payload(
            geometry_payload=geometry_payload,
            image_height=original_img.shape[0]
        )
        dxf_artifact = generate_dxf_artifact(
            graph_data_json=graph_data_json,
            analysis_id=analysis_id,
            geometry_payload=dxf_geometry_payload
        )
        
        response_data = {
            "analysis_id": analysis_id,
            "input_image": f"data:image/jpeg;base64,{input_img_base64}",
            "original_image": f"data:image/jpeg;base64,{input_img_base64}",
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "graph_data": {
                "json": graph_data_json,
                "xml": format_to_xml(graph_data_json),
                "iso15926_json": format_to_iso15926_json(graph_data_json)
            },
            "geometry": geometry_payload,
            "bill_of_materials": [{"symbol": name, "count": count} for name, count in bom_counts.items()],
            "class_names": class_names,
            "ocr_results": ocr_results_for_response,
            "review_summary": review_summary,
            "ocr_comparison": ocr_comparison,
            "metadata": {
                "markdown": metadata_output["markdown"],
                "json": metadata_output["json_data"],
                "available": metadata_output["available"],
                "summary": metadata_output["extraction_summary"]
            },
            "dxf": dxf_artifact
        }
        _save_analysis_record(
            analysis_id=analysis_id,
            filename=file.filename,
            duration_seconds=time.time() - _digitize_start,
            symbol_count=sum(bom_counts.values()),
            image_width=image_width,
            image_height=image_height,
            graph_data_json=graph_data_json,
            geometry_payload=geometry_payload,
            bom_counts=bom_counts,
            dxf_artifact=dxf_artifact,
            annotated_img=final_img,
        )
        
        import json as json_module
        response_size = len(json_module.dumps(response_data).encode('utf-8'))
        print(f"âœ… Successfully digitized image. Sending response (size: {response_size / 1024 / 1024:.2f} MB).")
        resp = jsonify(response_data)
        resp.headers['Connection'] = 'keep-alive'
        resp.headers['Keep-Alive'] = 'timeout=600'
        return resp
    except Exception as e:
        print(f"âŒ An error occurred during digitization: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}. Check the backend terminal for details."}), 500


@app.route('/download_dxf/<path:filename>', methods=['GET'])
def download_dxf(filename):
    safe_filename = secure_filename(Path(filename).name)
    if not safe_filename or not safe_filename.lower().endswith('.dxf'):
        return jsonify({"error": "Invalid DXF filename"}), 400

    file_path = DXF_OUTPUT_DIR / safe_filename
    if not file_path.is_file():
        return jsonify({"error": "DXF file not found"}), 404

    return send_from_directory(
        directory=str(DXF_OUTPUT_DIR),
        path=safe_filename,
        as_attachment=True,
        mimetype='application/dxf'
    )


@app.route('/history', methods=['GET'])
def get_history():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    records = _list_analysis_records()
    total = len(records)
    start = (page - 1) * per_page
    end = start + per_page
    return jsonify({
        'items': records[start:end],
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page,
    })


@app.route('/projects', methods=['GET'])
def get_projects():
    search = request.args.get('search', '').strip().lower()
    records = _list_analysis_records()
    if search:
        records = [r for r in records if search in r.get('filename', '').lower()]
    projects = []
    seen = set()
    for rec in records:
        fname = rec.get('filename', 'Unknown')
        if fname not in seen:
            seen.add(fname)
            projects.append({
                'analysis_id': rec['analysis_id'],
                'filename': fname,
                'timestamp': rec.get('timestamp'),
                'symbol_count': rec.get('symbol_count', 0),
                'status': rec.get('status', 'completed'),
                'annotated_image_url': rec.get('annotated_image_url'),
            })
    return jsonify({'items': projects, 'total': len(projects)})


@app.route('/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    safe_id = secure_filename(str(analysis_id))
    record_path = ANALYSES_DIR / f'{safe_id}.json'
    if not record_path.is_file():
        return jsonify({'error': 'Analysis not found'}), 404
    with open(record_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    annotated_url = record.get('annotated_image_url')
    if annotated_url:
        host = request.host_url.rstrip('/')
        record['annotated_image'] = f'{host}{annotated_url}'
    return jsonify(record)


@app.route('/analysis/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    safe_id = secure_filename(str(analysis_id))
    record_path = ANALYSES_DIR / f'{safe_id}.json'
    if not record_path.is_file():
        return jsonify({'error': 'Analysis not found'}), 404
    with open(record_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    annotated_url = record.get('annotated_image_url', '')
    if annotated_url:
        img_filename = Path(annotated_url).name
        img_path = ANALYSES_IMAGES_DIR / img_filename
        try:
            if img_path.is_file():
                img_path.unlink()
        except Exception:
            pass
    record_path.unlink()
    return jsonify({'deleted': analysis_id})


@app.route('/analysis/<analysis_id>/name', methods=['PATCH'])
def update_analysis_name(analysis_id):
    safe_id = secure_filename(str(analysis_id))
    record_path = ANALYSES_DIR / f'{safe_id}.json'
    if not record_path.is_file():
        return jsonify({'error': 'Analysis not found'}), 404
    body = request.get_json(silent=True) or {}
    new_name = str(body.get('project_name', '')).strip()
    if not new_name:
        return jsonify({'error': 'project_name is required'}), 400
    with open(record_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    record['project_name'] = new_name
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    return jsonify({'analysis_id': analysis_id, 'project_name': new_name})


@app.route('/export_history_csv', methods=['GET'])
def export_history_csv():
    import io
    import csv
    import datetime
    records = _list_analysis_records()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Analysis ID', 'Filename', 'Timestamp', 'Duration (s)', 'Symbol Count', 'Status', 'Method'])
    for rec in records:
        ts = rec.get('timestamp', '')
        if ts:
            ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([
            rec.get('analysis_id', ''),
            rec.get('filename', ''),
            ts,
            round(rec.get('duration_seconds', 0), 2),
            rec.get('symbol_count', 0),
            rec.get('status', ''),
            rec.get('method', ''),
        ])
    output.seek(0)
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=analysis_history.csv'}
    )


@app.route('/export_dxf', methods=['POST'])
def export_dxf():
    if not DXF_EXPORT_AVAILABLE:
        return jsonify({"error": "DXF export module is not available on backend"}), 503

    request_data = request.get_json(silent=True)
    if not isinstance(request_data, dict):
        return jsonify({"error": "Request body must be valid JSON object"}), 400

    analysis_id = str(request_data.get("analysis_id") or _build_analysis_id())
    try:
        geometry_payload = None
        if isinstance(request_data.get("geometry"), dict):
            geometry_payload = request_data.get("geometry")
        elif isinstance(request_data.get("equipment"), list) and isinstance(request_data.get("pipes"), list):
            first_pipe = request_data["pipes"][0] if request_data["pipes"] else None
            if isinstance(first_pipe, dict) and isinstance(first_pipe.get("points"), list):
                geometry_payload = request_data

        if isinstance(geometry_payload, dict):
            safe_analysis = secure_filename(analysis_id) or _build_analysis_id()
            filename = f"{safe_analysis}.dxf"
            output_path = DXF_OUTPUT_DIR / filename
            dxf_payload = build_dxf_geometry_payload(
                geometry_payload=geometry_payload,
                image_height=request_data.get("image_height", 0)
            )
            export_geometry_to_dxf(payload=dxf_payload, output_path=output_path)
            artifact = {
                "available": True,
                "filename": filename,
                "download_url": f"/download_dxf/{filename}",
                "error": None,
                "mode": "geometry_direct"
            }
        elif isinstance(request_data.get("equipment"), list) and isinstance(request_data.get("pipes"), list):
            safe_analysis = secure_filename(analysis_id) or _build_analysis_id()
            filename = f"{safe_analysis}.dxf"
            output_path = DXF_OUTPUT_DIR / filename
            export_pid_json_to_dxf(payload=request_data, output_path=output_path)
            artifact = {
                "available": True,
                "filename": filename,
                "download_url": f"/download_dxf/{filename}",
                "error": None,
                "mode": "graph_fallback"
            }
        else:
            graph_json = None
            if isinstance(request_data.get("graph_data"), dict):
                graph_json = request_data["graph_data"].get("json")
            if graph_json is None and isinstance(request_data.get("json"), list):
                graph_json = request_data.get("json")
            if graph_json is None and isinstance(request_data.get("graph_data_json"), list):
                graph_json = request_data.get("graph_data_json")

            if not isinstance(graph_json, list):
                return jsonify({
                    "error": "Provide canonical payload (equipment/pipes) or graph_data.json list."
                }), 400

            artifact = generate_dxf_artifact(graph_json, analysis_id)

        if not artifact.get("available"):
            return jsonify({
                "status": "failed",
                "dxf": artifact,
            }), 400

        return jsonify({
            "status": "success",
            "analysis_id": analysis_id,
            "dxf": artifact,
        })
    except NameError:
        return jsonify({"error": "DXF export symbols unavailable"}), 503
    except SchemaValidationError as exc:
        return jsonify({"error": f"Invalid DXF payload: {str(exc)}"}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"DXF export failed: {str(exc)}"}), 500

@app.route('/submit_correction', methods=['POST'])
def submit_correction():
    global class_names
    try:
        data = request.get_json()
        annotation = data['annotation']
        class_name = annotation['className'].strip()
        is_new_class = class_name not in class_names
        
        if is_new_class:
            return jsonify({"status": "new_class_detected", "className": class_name})

        image_b64 = data['image_b64'].split(',')[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        timestamp = int(time.time())
        img_filename = f"correction_{timestamp}.jpg"
        label_filename = f"correction_{timestamp}.txt"
        cv2.imwrite(str(FEEDBACK_IMAGES_DIR / img_filename), img)
        class_id = class_names.index(class_name)
        x1, y1, x2, y2 = annotation['box']
        cx, cy, box_w, box_h = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        with open(FEEDBACK_LABELS_DIR / label_filename, 'w') as f: f.write(f"{class_id} {cx} {cy} {box_w} {box_h}")
        
        return jsonify({"status": "success", "message": "Correction saved."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to save correction: {str(e)}"}), 500

@app.route('/submit_tag_correction', methods=['POST'])
def submit_tag_correction():
    try:
        data = request.get_json()
        node_id = data.get('node_id')
        tag = data.get('tag')
        image_b64 = data.get('image_b64', '').split(',')[1] if data.get('image_b64') else None
        
        if not node_id or not tag:
            print(f"âŒ Missing node_id or tag: node_id={node_id}, tag={tag}")
            return jsonify({"error": "Missing node_id or tag"}), 400
        
        timestamp = int(time.time())
        tag_filename = f"tag_correction_{timestamp}_{node_id}.json"
        correction_data = {
            "node_id": node_id,
            "tag": tag,
            "timestamp": timestamp,
            "tag_confidence": 1.0,
            "needs_review": False
        }
        
        if image_b64:
            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
                img_filename = f"tag_correction_{timestamp}_{node_id}.jpg"
                cv2.imwrite(str(FEEDBACK_IMAGES_DIR / img_filename), img)
                correction_data["image_filename"] = img_filename
            except Exception as e:
                print(f"âš ï¸ Warning: Failed to save tag correction image: {str(e)}")
        
        with open(FEEDBACK_TAGS_DIR / tag_filename, 'w') as f:
            json.dump(correction_data, f)
        
        print(f"âœ… Saved tag correction for node {node_id}: {tag}")
        return jsonify({
            "status": "success",
            "message": f"Tag correction saved for node {node_id}.",
            "updated_node": {
                "node_id": node_id,
                "tag": tag,
                "tag_confidence": 1.0,
                "needs_review": False
            }
        })
    except Exception as e:
        print(f"âŒ Failed to save tag correction: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to save tag correction: {str(e)}"}), 500

@app.route('/upload_new_symbol_data', methods=['POST'])
def upload_new_symbol_data():
    global class_names
    try:
        class_name = request.form['className']
        files = request.files.getlist('files')
        
        if class_name not in class_names:
            class_names.append(class_name)
            with open(CLASS_NAMES_PATH, 'w') as f:
                json.dump(class_names, f)
        
        class_id = class_names.index(class_name)
        
        for file in files:
            timestamp = int(time.time() * 1000)
            img_bytes = file.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            img_filename = f"new_class_{timestamp}_{Path(file.filename).stem}.jpg"
            label_filename = f"new_class_{timestamp}_{Path(file.filename).stem}.txt"
            cv2.imwrite(str(FEEDBACK_IMAGES_DIR / img_filename), img)
            
            yolo_label = f"{class_id} 0.5 0.5 1.0 1.0"
            with open(FEEDBACK_LABELS_DIR / label_filename, 'w') as f: f.write(yolo_label)
            
        return jsonify({"status": "success", "message": f"Added {len(files)} new files for '{class_name}'."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to process new symbol data: {str(e)}"}), 500

@app.route('/start_finetuning', methods=['POST'])
def start_finetuning():
    try:
        python_executable = sys.executable
        print(f"[BACKEND] Received fine-tuning request. Starting fine-tuning with: {python_executable} finetune.py")
        sys.stdout.flush()
        subprocess.Popen([python_executable, "finetune.py"], cwd=os.getcwd())
        return jsonify({"status": "success", "message": "Fine-tuning started in the background."})
    except Exception as e:
        print(f"âŒ Failed to start fine-tuning: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to start fine-tuning: {str(e)}"}), 500

def watch_for_restart():
    while True:
        if RESTART_FLAG_FILE.exists():
            print("ðŸš¨ New model detected! Restarting server...")
            RESTART_FLAG_FILE.unlink()
            os.execv(sys.executable, ['python'] + sys.argv)
        time.sleep(10)


# ========== DPIE — DigiPID Interactive Editor Backend ==========

import re as _re

try:
    import google.generativeai as _genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _genai = None
    _GEMINI_AVAILABLE = False

# _GEMINI_API_KEY  = os.getenv('gemini_api_key', 'AIzaSyBZuJrj0flJlKn5KLC3lDIyQRYQ7zLCI3A')

_GEMINI_API_KEY = "AIzaSyCDUjL0VCCPhdgaPvDsGBqK704ppjyQwRg"
# _OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

_VALID_SYMBOL_TYPES: set = set()


def _load_valid_types():
    global _VALID_SYMBOL_TYPES
    try:
        cn_path = Path(__file__).parent / 'class_names.json'
        with open(cn_path, 'r', encoding='utf-8') as f:
            names = json.load(f)
        _VALID_SYMBOL_TYPES = set(names) if isinstance(names, list) else set()
    except Exception:
        pass
    try:
        st_path = Path(__file__).parent / 'symbol_templates.json'
        with open(st_path, 'r', encoding='utf-8') as f:
            tmpl = json.load(f)
        _VALID_SYMBOL_TYPES |= set(tmpl.keys())
    except Exception:
        pass


_load_valid_types()


def _find_nearest_equipment(point, equipment_list, radius=60):
    px, py = point[0], point[1]
    best, best_d = None, float('inf')
    for eq in equipment_list:
        d = ((eq['position'][0] - px) ** 2 + (eq['position'][1] - py) ** 2) ** 0.5
        if d < best_d and d <= radius:
            best, best_d = eq, d
    return best


def rebuild_graph_from_geometry(geometry_payload):
    equipment = geometry_payload.get('equipment', [])
    pipes = geometry_payload.get('pipes', [])
    connections = {str(e['id']): [] for e in equipment}
    for pipe in pipes:
        fid = str(pipe.get('from_id', ''))
        tid = str(pipe.get('to_id', ''))
        pts = pipe.get('points', [])
        if not fid and pts:
            near = _find_nearest_equipment(pts[0], equipment)
            if near:
                fid = str(near['id'])
        if not tid and pts and len(pts) >= 2:
            near = _find_nearest_equipment(pts[-1], equipment)
            if near:
                tid = str(near['id'])
        if fid and tid and fid != tid:
            if tid not in connections.get(fid, []):
                connections.setdefault(fid, []).append(tid)
            if fid not in connections.get(tid, []):
                connections.setdefault(tid, []).append(fid)
    nodes = []
    for e in equipment:
        eid = str(e['id'])
        nodes.append({
            'node_id': eid,
            'type': e.get('type', 'unknown'),
            'tag': e.get('label', ''),
            'tag_confidence': 1.0,
            'needs_review': False,
            'connections': connections.get(eid, []),
            'bbox': e.get('bbox'),
            'position': e.get('position')
        })
    return nodes


def build_agent_system_prompt(state, history=None):
    equipment = state.get('equipment', [])
    pipes = state.get('pipes', [])
    eq_lines = []
    for e in equipment:
        pos = e.get('position', [0, 0])
        eq_lines.append('  id={} type={} label={} pos=[{:.0f},{:.0f}]'.format(
            e['id'], e.get('type', '?'), e.get('label', ''), pos[0], pos[1]
        ))
    pipe_lines = []
    for p in pipes:
        pipe_lines.append('  id={} from={} to={} style={}'.format(
            p['id'], p.get('from_id', '?'), p.get('to_id', '?'), p.get('line_style', 'solid')
        ))
    hist_block = ''
    if history:
        hist_block = '\nRecent conversation:\n' + '\n'.join(
            '  User: {}\n  Assistant: {}'.format(h.get('user', ''), h.get('reply', ''))
            for h in history[-3:]
        )
    return (
        'You are a P&ID diagram editing assistant. Return ONLY a JSON object with keys '
        '"reply" (string) and "actions" (array). No markdown, no extra text.\n\n'
        'Valid action types and required fields:\n'
        '  MOVE_EQUIPMENT: {type, id, dx, dy}\n'
        '  SET_EQUIPMENT_POSITION: {type, id, x, y}\n'
        '  DELETE_EQUIPMENT: {type, id}\n'
        '  UPDATE_LABEL: {type, id, label}\n'
        '  CHANGE_TYPE: {type, id, new_type}\n'
        '  ADD_EQUIPMENT: {type, new_type, x, y, label}\n'
        '  ADD_PIPE: {type, from_id, to_id, line_style}\n'
        '  DELETE_PIPE: {type, id}\n'
        '  SET_LINE_STYLE: {type, id, style}\n'
        '  REROUTE_PIPE: {type, id, new_points}\n\n'
        'Current diagram state:\nEquipment ({} items):\n{}\nPipes ({} items):\n{}{}'
    ).format(
        len(equipment), '\n'.join(eq_lines) or '  (none)',
        len(pipes), '\n'.join(pipe_lines) or '  (none)',
        hist_block
    )


def call_gemini(system_prompt, user_prompt):
    if not _GEMINI_AVAILABLE:
        raise RuntimeError('google-generativeai package not installed')
    if not _GEMINI_API_KEY:
        raise RuntimeError('GEMINI_API_KEY environment variable not set')
    _genai.configure(api_key=_GEMINI_API_KEY)
    model = _genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        system_instruction=system_prompt,
        generation_config=_genai.types.GenerationConfig(temperature=0.1)
    )
    response = model.generate_content(user_prompt)
    return response.text


def call_llama2(system_prompt, user_prompt):
    url = _OLLAMA_BASE_URL.rstrip('/') + '/api/generate'
    prompt_text = '[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]'.format(system_prompt, user_prompt)
    payload = {'model': 'llama2', 'prompt': prompt_text, 'stream': False}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get('response', '')


def parse_agent_response(text):
    raw = str(text or '').strip()
    if not raw:
        return {'reply': 'No response from model.', 'actions': []}

    cleaned = _re.sub(r'^```(?:json)?\s*', '', raw, flags=_re.MULTILINE)
    cleaned = _re.sub(r'\s*```$', '', cleaned, flags=_re.MULTILINE)

    payload = None
    try:
        payload = json.loads(cleaned)
    except Exception:
        match = _re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None

    if not isinstance(payload, dict):
        return {'reply': 'I could not parse that command.', 'actions': []}

    reply = str(payload.get('reply', 'Done.'))
    actions = payload.get('actions', [])
    if not isinstance(actions, list):
        actions = []
    return {'reply': reply, 'actions': actions}


def validate_actions(actions, state):
    if not isinstance(actions, list):
        return []
    eq_ids = {str(e.get('id')) for e in state.get('equipment', []) if isinstance(e, dict) and e.get('id') is not None}
    pipe_ids = {str(p.get('id')) for p in state.get('pipes', []) if isinstance(p, dict) and p.get('id') is not None}
    valid_styles = {'solid', 'dashed'}

    valid = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        atype = action.get('type', '')
        if atype in ('MOVE_EQUIPMENT', 'SET_EQUIPMENT_POSITION', 'DELETE_EQUIPMENT',
                     'UPDATE_LABEL', 'CHANGE_TYPE'):
            if str(action.get('id', '')) not in eq_ids:
                continue
        elif atype in ('DELETE_PIPE', 'SET_LINE_STYLE', 'REROUTE_PIPE'):
            if str(action.get('id', '')) not in pipe_ids:
                continue
            if atype == 'SET_LINE_STYLE':
                style = str(action.get('style', action.get('line_style', 'solid'))).strip().lower()
                if style not in valid_styles:
                    continue
            if atype == 'REROUTE_PIPE':
                new_points = action.get('new_points', [])
                if not isinstance(new_points, list) or len(new_points) < 2:
                    continue
        elif atype == 'ADD_PIPE':
            if str(action.get('from_id', '')) not in eq_ids:
                continue
            if str(action.get('to_id', '')) not in eq_ids:
                continue
            line_style = str(action.get('line_style', 'solid')).strip().lower()
            if line_style not in valid_styles:
                action['line_style'] = 'solid'
        elif atype == 'ADD_EQUIPMENT':
            proposed_type = str(action.get('new_type', action.get('symbol_type', action.get('eq_type', '')))).strip().lower()
            if _VALID_SYMBOL_TYPES and proposed_type and proposed_type not in _VALID_SYMBOL_TYPES:
                continue

        valid.append(action)
    return valid


@app.route('/symbol_templates', methods=['GET'])
def symbol_templates_route():
    try:
        st_path = Path(__file__).parent / 'symbol_templates.json'
        with open(st_path, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        return jsonify(templates)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ── Agent AI Query endpoint (Gemini / Llama) ─────────────────────────────────
@app.route('/agent/query', methods=['POST'])
def agent_query():
    body = request.get_json(silent=True) or {}
    model   = str(body.get('model', 'gemini-2.5-flash'))
    prompt  = str(body.get('prompt', '')).strip()
    context = body.get('context', {})
    history = body.get('history', [])

    if not prompt:
        return jsonify({'error': 'prompt is required'}), 400

    system_prompt = (
        'You are an AI engineering assistant for P&ID (Piping and Instrumentation Diagram) digitization.\n'
        'You have access to the current diagram state below. All coordinates are in image pixel space.\n'
        'Always respond in valid JSON using EXACTLY this schema:\n'
        '{"reply": "<natural language answer>", "commands": [<optional editor commands>]}\n'
        'Available commands (only use when user explicitly asks to modify the diagram):\n'
        '- Move symbol:    {"action":"MOVE_SYMBOL","target":"<label>","parameters":{"x":<num>,"y":<num>}}\n'
        '- Add symbol:     {"action":"ADD_SYMBOL","target":null,"parameters":{"symbol_type":"<type>","x":<num>,"y":<num>,"label":"<text>"}}\n'
        '- Delete symbol:  {"action":"DELETE_SYMBOL","target":"<label>","parameters":{}}\n'
        '- Rename symbol:  {"action":"UPDATE_LABEL","target":"<label>","parameters":{"label":"<new_text>"}}\n'
        '- Replace type:   {"action":"REPLACE_SYMBOL","target":"<label>","parameters":{"symbol_type":"<new_type>"}}\n'
        '- Draw pipe:      {"action":"DRAW_PIPE","target":null,"parameters":{"points":[[x1,y1],[x2,y2],...]}}\n'
        '- Connect two symbols with a straight pipe: {"action":"CONNECT_SYMBOLS","target":null,"parameters":{"from":"<label>","to":"<label>"}}\n'
        'For DRAW_PIPE you MUST provide absolute pixel coordinates in the "points" array. '
        'If the user says "draw a line N units to the right from symbol X", compute end point as [X.position[0]+N, X.position[1]] and use DRAW_PIPE with points.\n'
        'Answer questions accurately using only the data provided — do not hallucinate symbols that are not listed.\n'
        '\n--- CURRENT DIAGRAM ---\n' + _build_agent_context_str(context)
    )

    raw_reply = ''
    try:
        if model == 'gemini-2.5-flash':
            raw_reply = _call_gemini(system_prompt, history, prompt)
        elif model == 'llama2':
            raw_reply = _call_llama(system_prompt, history, prompt)
        else:
            return jsonify({'error': f'Unknown model: {model}'}), 400
    except Exception as exc:
        logger.error(f'Agent query error: {exc}')
        return jsonify({'reply': f'Model error: {exc}', 'commands': [], 'model_used': model}), 200

    reply_text, commands = _parse_agent_reply(raw_reply)
    return jsonify({'reply': reply_text, 'commands': commands, 'model_used': model})


def _build_agent_context_str(ctx):
    parts = []
    parts.append(f"Total symbols: {ctx.get('total_symbols', 0)}, Total pipes: {ctx.get('total_pipes', 0)}")
    counts = ctx.get('symbol_counts', {})
    if counts:
        parts.append('Symbol counts: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
    syms = ctx.get('all_symbols', [])
    if syms:
        entries = '; '.join(
            f"{s.get('label', '?')} ({s.get('type', '?')}) @ {s.get('position', [])}"
            for s in syms[:40]
        )
        parts.append('Symbols: ' + entries)
    conns = ctx.get('connection_summary', [])
    if conns:
        entries = '; '.join(
            f"{c.get('from_label', '?')} -> {c.get('to_label', '?')} [{c.get('line_style', 'solid')}]"
            for c in conns[:30]
        )
        parts.append('Connections: ' + entries)
    sel = ctx.get('selected_node')
    if sel:
        parts.append(f"Selected: {sel.get('label', '?')} ({sel.get('type', '?')})")
    meta = ctx.get('metadata_summary', {})
    bom = meta.get('bill_of_materials', [])
    if bom:
        parts.append('BOM: ' + ', '.join(f"{b.get('symbol', '?')}={b.get('count', '?')}" for b in bom[:20]))
    return '\n'.join(parts)


def _call_gemini(system_prompt, history, user_prompt):
    api_key = "AIzaSyAwiuY6lYsM50YqVBlvU0y9hK9uaMj8f_A"
    if not api_key:
        return '{"reply": "GEMINI_API_KEY is not set on the server. Add it to your environment.", "commands": []}'
    # Build flat prompt: system + history + user turn
    turns = [system_prompt]
    for h in history[-8:]:
        role = 'User' if h.get('role') == 'user' else 'Assistant'
        turns.append(f"{role}: {h.get('content', '')}")
    turns.append(f'User: {user_prompt}')
    full_prompt = '\n\n'.join(turns)
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        mdl = genai.GenerativeModel('gemini-flash-latest')
        response = mdl.generate_content(full_prompt)
        return response.text
    except ImportError:
        pass
    # REST fallback
    import urllib.request
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}'
    payload = json.dumps({
        'contents': [{'role': 'user', 'parts': [{'text': full_prompt}]}]
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data['candidates'][0]['content']['parts'][0]['text']


def _call_llama(system_prompt, history, user_prompt):
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
    messages = [{'role': 'system', 'content': system_prompt}]
    for h in history[-8:]:
        messages.append({'role': h.get('role', 'user'), 'content': str(h.get('content', ''))})
    messages.append({'role': 'user', 'content': user_prompt})
    import urllib.request
    payload = json.dumps({'model': 'llama2', 'messages': messages, 'stream': False}).encode()
    req = urllib.request.Request(
        f'{ollama_url}/api/chat',
        data=payload,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get('message', {}).get('content', '')


def _parse_agent_reply(raw):
    import re
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if fence:
        try:
            parsed = json.loads(fence.group(1).strip())
            if isinstance(parsed.get('reply'), str):
                return parsed['reply'], parsed.get('commands', [])
        except Exception:
            pass
    obj_match = re.search(r'\{[\s\S]*"reply"[\s\S]*\}', raw)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            if isinstance(parsed.get('reply'), str):
                return parsed['reply'], parsed.get('commands', [])
        except Exception:
            pass
    return raw.strip() or 'No response.', []



@app.route('/agent_command', methods=['POST'])
def agent_command():
    try:
        body    = request.get_json(force=True)
        command = str(body.get('command', '')).strip()
        model   = str(body.get('model', 'gemini')).lower()
        state   = body.get('state', {})
        history = body.get('history', [])
        if not command:
            return jsonify({'success': False, 'error': 'Empty command'}), 400
        system_prompt = build_agent_system_prompt(state, history)
        if model in ('llama', 'llama2'):
            raw = call_llama2(system_prompt, command)
        else:
            if not _GEMINI_API_KEY:
                return jsonify({'success': False, 'error': 'GEMINI_API_KEY is not configured on the server.'}), 503
            raw = call_gemini(system_prompt, command)
        parsed  = parse_agent_response(raw)
        reply   = str(parsed.get('reply', 'Done.'))
        actions = validate_actions(parsed.get('actions', []), state)
        return jsonify({'success': True, 'reply': reply, 'actions': actions})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/save_diagram', methods=['POST'])
def save_diagram():
    try:
        body        = request.get_json(force=True)
        analysis_id = str(body.get('analysis_id', '')).strip() or _build_analysis_id()
        geometry    = body.get('geometry', {})
        image_height = body.get('image_height', 0)
        if not isinstance(geometry, dict):
            return jsonify({'success': False, 'error': 'Missing geometry payload'}), 400

        equipment   = geometry.get('equipment', []) if isinstance(geometry.get('equipment', []), list) else []
        pipes       = geometry.get('pipes', []) if isinstance(geometry.get('pipes', []), list) else []
        outputs     = {}
        graph_nodes = []

        safe_analysis = secure_filename(analysis_id) or _build_analysis_id()
        edited_base = f"{safe_analysis}_edited"

        try:
            DXF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            json_path = DXF_OUTPUT_DIR / f"{edited_base}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({'equipment': equipment, 'pipes': pipes}, f, indent=2)
            outputs['json'] = {
                'saved': True,
                'filename': json_path.name,
                'download_url': '/download_json/' + json_path.name,
            }
        except Exception as e:
            outputs['json'] = {'saved': False, 'error': str(e)}

        try:
            graph_nodes = rebuild_graph_from_geometry(geometry)
        except Exception as e:
            graph_nodes = []
            outputs['graph'] = {'saved': False, 'error': str(e)}

        try:
            xml_data = format_to_xml(graph_nodes)
            xml_path = DXF_OUTPUT_DIR / f"{edited_base}.xml"
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_data)
            outputs['xml'] = {
                'saved': True,
                'filename': xml_path.name,
                'download_url': '/download_xml/' + xml_path.name,
            }
        except Exception as e:
            outputs['xml'] = {'saved': False, 'error': str(e)}

        try:
            iso_data = format_to_iso15926_json(graph_nodes)
            iso_path = DXF_OUTPUT_DIR / f"{edited_base}_iso.json"
            with open(iso_path, 'w', encoding='utf-8') as f:
                json.dump(iso_data, f, indent=2)
            outputs['iso15926'] = {
                'saved': True,
                'filename': iso_path.name,
                'download_url': '/download_json/' + iso_path.name,
            }
        except Exception as e:
            outputs['iso15926'] = {'saved': False, 'error': str(e)}

        try:
            geo_payload = {'equipment': equipment, 'pipes': pipes}
            dxf_payload = build_dxf_geometry_payload(geo_payload, image_height=image_height)
            dxf_info = generate_dxf_artifact(graph_nodes, edited_base, geometry_payload=dxf_payload)
            outputs['dxf'] = dxf_info
        except Exception as e:
            outputs['dxf'] = {'available': False, 'error': str(e)}

        return jsonify({'success': True, 'analysis_id': safe_analysis, 'outputs': outputs})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download_json/<path:filename>', methods=['GET'])
def download_json_file(filename):
    safe = secure_filename(Path(filename).name)
    return send_from_directory(str(DXF_OUTPUT_DIR.resolve()), safe, as_attachment=True)


@app.route('/download_xml/<path:filename>', methods=['GET'])
def download_xml_file(filename):
    safe = secure_filename(Path(filename).name)
    return send_from_directory(str(DXF_OUTPUT_DIR.resolve()), safe, as_attachment=True)


if __name__ == '__main__':
    watcher = threading.Thread(target=watch_for_restart, daemon=True)
    watcher.start()
    print("âœ… Backend server started on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
