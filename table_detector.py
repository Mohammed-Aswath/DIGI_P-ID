"""
Table Detection and Cell Extraction Module

This module handles:
1. Detecting table structures in P&ID diagrams using contour analysis
2. Identifying rows, columns, and cells
3. Extracting text from individual cells
4. Reconstructing table format

Architecture:
- CPU-only operation (no GPU required)
- Uses contour detection for table boundaries
- Hough lines for cell boundaries
- Preserves table structure information

Author: Senior Computer Vision Engineer
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Cell:
    """Represents a single table cell"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    row: int
    col: int
    text: str = ""
    confidence: float = 0.0


@dataclass
class Table:
    """Represents a detected table"""
    bbox: Tuple[int, int, int, int]  # Table bounding box
    cells: List[Cell]
    rows: int
    cols: int
    table_type: str = "data"  # data, revision, specifications, notes


class TableDetector:
    """
    Detects and extracts table structures from images.
    
    CPU-optimized operations:
    - No deep learning inference
    - Uses classical CV techniques (contours, lines)
    - Efficient for CPU-only environments
    """
    
    def __init__(self, min_table_area=5000, line_threshold=20):
        """
        Initialize table detector.
        
        Args:
            min_table_area: Minimum table area (pixels^2) to be considered valid
            line_threshold: Minimum line length to be considered a table boundary
        """
        self.min_table_area = min_table_area
        self.line_threshold = line_threshold
        self.logger = logger
    
    def detect_tables(self, image: np.ndarray, max_tables: int = 5) -> List[Table]:
        """
        Detect all tables in the image.
        
        Args:
            image: Input image (BGR format)
            max_tables: Maximum number of tables to detect
            
        Returns:
            List of detected Table objects (empty bbox for cells, to be filled later)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary_inv = cv2.bitwise_not(binary)
        
        # Find contours
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours[:max_tables]:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area < self.min_table_area:
                continue
            
            # Check if it looks like a table (rectangular)
            aspects = w / (h + 1e-5)
            if not (0.3 < aspects < 3.0):  # Too skewed
                continue
            
            table_bbox = (x, y, x + w, y + h)
            table = Table(bbox=table_bbox, cells=[], rows=0, cols=0)
            tables.append(table)
            self.logger.info(f"Detected table at {table_bbox}, area={area}")
        
        return tables
    
    def extract_cells_from_table(
        self,
        image: np.ndarray,
        table_bbox: Tuple[int, int, int, int]
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Extract cell bounding boxes from a table.
        
        Uses line detection (Hough transform) to find table grid.
        
        Args:
            image: Input image (BGR format)
            table_bbox: (x1, y1, x2, y2) - table region
            
        Returns:
            2D list of cell bboxes: cells[row][col] = (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = table_bbox
        table_region = image[y1:y2, x1:x2]
        
        # Preprocess
        gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Edge detection
        edges = cv2.Canny(binary, 50, 150)
        
        # Detect horizontal and vertical lines
        h_lines = self._detect_horizontal_lines(edges)
        v_lines = self._detect_vertical_lines(edges)
        
        self.logger.debug(f"Found {len(h_lines)} horizontal lines, {len(v_lines)} vertical lines")
        
        # Sort lines
        h_lines = sorted(set(h_lines))
        v_lines = sorted(set(v_lines))
        
        # Extract cells
        cells = self._create_cells_from_lines(h_lines, v_lines, table_region.shape)
        
        return cells
    
    def _detect_horizontal_lines(self, edges: np.ndarray, min_length: int = 50) -> List[int]:
        """
        Detect horizontal lines in edge image.
        
        Args:
            edges: Edge-detected image
            min_length: Minimum line length
            
        Returns:
            List of y-coordinates of horizontal lines
        """
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Find the y-coordinates where horizontal intensity is strong
        h_sum = np.sum(horizontal > 0, axis=1)
        threshold = np.max(h_sum) * 0.5
        h_lines = np.where(h_sum > threshold)[0]
        
        # Cluster nearby lines
        h_lines = self._cluster_lines(h_lines, cluster_dist=5)
        
        return h_lines
    
    def _detect_vertical_lines(self, edges: np.ndarray, min_length: int = 50) -> List[int]:
        """
        Detect vertical lines in edge image.
        
        Args:
            edges: Edge-detected image
            min_length: Minimum line length
            
        Returns:
            List of x-coordinates of vertical lines
        """
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
        vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Find the x-coordinates where vertical intensity is strong
        v_sum = np.sum(vertical > 0, axis=0)
        threshold = np.max(v_sum) * 0.5
        v_lines = np.where(v_sum > threshold)[0]
        
        # Cluster nearby lines
        v_lines = self._cluster_lines(v_lines, cluster_dist=5)
        
        return v_lines
    
    def _cluster_lines(self, lines: np.ndarray, cluster_dist: int = 5) -> List[int]:
        """
        Merge nearby lines into single lines (cluster).
        
        Args:
            lines: Array of line positions
            cluster_dist: Distance threshold for clustering
            
        Returns:
            List of clustered line positions (median of each cluster)
        """
        if len(lines) == 0:
            return []
        
        lines = np.array(sorted(set(lines)))
        clusters = []
        current_cluster = [lines[0]]
        
        for line in lines[1:]:
            if line - current_cluster[-1] <= cluster_dist:
                current_cluster.append(line)
            else:
                clusters.append(int(np.median(current_cluster)))
                current_cluster = [line]
        
        if current_cluster:
            clusters.append(int(np.median(current_cluster)))
        
        return clusters
    
    def _create_cells_from_lines(
        self,
        h_lines: List[int],
        v_lines: List[int],
        region_shape: Tuple[int, int]
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Create cell bounding boxes from detected lines.
        
        Args:
            h_lines: Horizontal line y-coordinates
            v_lines: Vertical line x-coordinates
            region_shape: (height, width) of table region
            
        Returns:
            2D list of cell bboxes
        """
        if not h_lines or not v_lines:
            return []
        
        cells = []
        rows = len(h_lines) - 1
        cols = len(v_lines) - 1
        
        for row_idx in range(rows):
            row_cells = []
            y1, y2 = h_lines[row_idx], h_lines[row_idx + 1]
            
            for col_idx in range(cols):
                x1, x2 = v_lines[col_idx], v_lines[col_idx + 1]
                
                # Add padding to avoid grid lines
                x1_padded = min(x1 + 2, x2 - 2)
                y1_padded = min(y1 + 2, y2 - 2)
                x2_padded = max(x2 - 2, x1_padded + 1)
                y2_padded = max(y2 - 2, y1_padded + 1)
                
                row_cells.append((x1_padded, y1_padded, x2_padded, y2_padded))
            
            cells.append(row_cells)
        
        self.logger.info(f"Extracted {rows} rows × {cols} columns = {rows * cols} cells")
        return cells
    
    def crop_cell(self, image: np.ndarray, cell_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract cell region from image.
        
        Args:
            image: Input image
            cell_bbox: (x1, y1, x2, y2) - cell coordinates
            
        Returns:
            Cropped cell image
        """
        x1, y1, x2, y2 = cell_bbox
        h, w = image.shape[:2]
        
        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        return image[y1:y2, x1:x2]
    
    def get_table_statistics(self, table: Table) -> Dict:
        """
        Get statistics about a detected table.
        
        Args:
            table: Table object
            
        Returns:
            Dictionary with table statistics
        """
        x1, y1, x2, y2 = table.bbox
        width = x2 - x1
        height = y2 - y1
        
        avg_cell_width = width / (table.cols + 1e-5)
        avg_cell_height = height / (table.rows + 1e-5)
        
        return {
            "bbox": table.bbox,
            "dimensions": (width, height),
            "cells": table.rows * table.cols,
            "rows": table.rows,
            "cols": table.cols,
            "avg_cell_size": (avg_cell_width, avg_cell_height),
            "total_cells_with_text": sum(1 for cell in table.cells if cell.text)
        }
