"""
Symbol-Guided OCR Module for P&ID Tag Extraction

This module implements a production-ready, symbol-aware OCR pipeline that:
1. Uses YOLO symbol detections as spatial anchors
2. Extracts text tags from cropped regions around symbols
3. Applies domain-specific post-processing for P&ID tags

Architecture:
- Modular design: Can be used independently or integrated into existing pipeline
- Recognition-only OCR: Uses PaddleOCR in rec-only mode (no detection)
- Symbol-guided: Leverages YOLO detections to focus OCR on relevant regions
- Domain-aware: P&ID-specific validation and correction rules

Author: Senior Computer Vision Engineer
"""

import re
import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path

# Suppress PaddleOCR verbose output
os.environ['PADDLEOCR_LOG_LEVEL'] = 'ERROR'

try:
    from paddleocr import PaddleOCR
    import paddleocr
    # Try to get version info
    try:
        PADDLEOCR_VERSION = getattr(paddleocr, '__version__', 'unknown')
    except:
        PADDLEOCR_VERSION = 'unknown'
except ImportError:
    raise ImportError(
        "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
    )


# ============================================================================
# Configuration Constants
# ============================================================================

class OCRConfig:
    """Configuration for symbol-guided OCR pipeline"""
    
    # ROI Expansion (asymmetric - more horizontal space for tags)
    EXPANSION_LEFT_PX = 50      # Expand left of symbol
    EXPANSION_RIGHT_PX = 150    # Expand right of symbol (tags usually to the right)
    EXPANSION_TOP_PX = 30       # Expand above symbol
    EXPANSION_BOTTOM_PX = 30    # Expand below symbol
    
    # OCR Model Configuration
    OCR_REC_MODEL = 'SVTR_LCNet'  # SVTR or CRNN for recognition
    OCR_LANG = 'en'                # English only
    OCR_USE_GPU = False            # Set to True if GPU available
    
    # Preprocessing
    ADAPTIVE_THRESH_BLOCK_SIZE = 11
    ADAPTIVE_THRESH_C = 2
    MORPH_KERNEL_SIZE = 2
    
    # Post-processing
    MIN_TAG_LENGTH = 2
    MAX_TAG_LENGTH = 15
    MIN_CONFIDENCE = 0.3  # Lowered from 0.5 - scanned drawings often have lower confidence
    
    # P&ID Tag Patterns (common formats)
    TAG_PATTERNS = [
        r'^[A-Z]{1,3}-\d{1,4}[A-Z]?$',      # FT-101, P-203A
        r'^[A-Z]{2}\d{1,4}[A-Z]?$',          # FT101, P203A (no hyphen)
        r'^[A-Z]{1,3}\d{1,4}[A-Z]?$',        # F101, P203A
        r'^[A-Z]{1,3}-\d{1,4}-\d{1,2}$',     # FT-101-01
    ]
    
    # Character normalization map (common OCR errors)
    CHAR_NORMALIZATION = {
        'O': '0',  # Letter O to digit 0
        'I': '1',  # Letter I to digit 1
        'S': '5',  # Letter S to digit 5
        'Z': '2',  # Letter Z to digit 2
        'B': '8',  # Letter B to digit 8
    }


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TagResult:
    """Result of tag extraction for a single symbol"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # Expanded ROI used for OCR
    raw_ocr_text: str  # Original OCR output before post-processing
    is_valid: bool     # Whether tag matches P&ID patterns
    corrections_applied: List[str]  # List of corrections made


# ============================================================================
# ROI Expansion Utilities
# ============================================================================

def expand_bbox(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    config: OCRConfig = OCRConfig()
) -> Tuple[int, int, int, int]:
    """
    Expand bounding box asymmetrically to create ROI for tag extraction.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box of detected symbol
        img_shape: (height, width) of the image
        config: OCR configuration
        
    Returns:
        Expanded bounding box (x1, y1, x2, y2) clamped to image dimensions
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = img_shape[:2]
    
    # Expand asymmetrically (more space horizontally, especially to the right)
    expanded_x1 = max(0, x1 - config.EXPANSION_LEFT_PX)
    expanded_y1 = max(0, y1 - config.EXPANSION_TOP_PX)
    expanded_x2 = min(img_w, x2 + config.EXPANSION_RIGHT_PX)
    expanded_y2 = min(img_h, y2 + config.EXPANSION_BOTTOM_PX)
    
    return (expanded_x1, expanded_y1, expanded_x2, expanded_y2)


def crop_roi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop region of interest from image.
    
    Args:
        image: Input image (BGR or grayscale)
        bbox: (x1, y1, x2, y2) bounding box
        
    Returns:
        Cropped ROI image
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


# ============================================================================
# Image Preprocessing for OCR
# ============================================================================

def preprocess_for_ocr(roi: np.ndarray, config: OCRConfig = OCRConfig(), use_light_preprocessing: bool = True) -> np.ndarray:
    """
    Preprocess ROI image to optimize OCR accuracy.
    
    Process:
    1. Convert to grayscale
    2. Apply light denoising
    3. Optional: Apply adaptive thresholding (if use_light_preprocessing=False)
    
    Args:
        roi: Cropped region of interest (BGR or grayscale)
        config: OCR configuration
        use_light_preprocessing: If True, only apply light denoising. If False, apply full preprocessing.
        
    Returns:
        Preprocessed image (RGB format for PaddleOCR)
    """
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # Light denoising to preserve text
    denoised = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=5, searchWindowSize=11)
    
    if use_light_preprocessing:
        # Light preprocessing: just denoise and enhance contrast slightly
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Convert back to RGB (PaddleOCR works better with RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced_rgb
    else:
        # Full preprocessing: adaptive thresholding (original approach)
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # Changed from THRESH_BINARY_INV - keep normal polarity
            config.ADAPTIVE_THRESH_BLOCK_SIZE,
            config.ADAPTIVE_THRESH_C
        )
        
        # Light morphological cleanup (remove small noise, preserve characters)
        kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Convert to RGB for PaddleOCR
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return cleaned_rgb


# ============================================================================
# PaddleOCR Recognition-Only Wrapper
# ============================================================================

class PaddleOCRRecognizer:
    """
    Wrapper for PaddleOCR in recognition-only mode.
    
    This class initializes PaddleOCR once and reuses it for multiple
    recognition calls, improving performance.
    """
    
    def __init__(self, config: OCRConfig = OCRConfig()):
        """
        Initialize PaddleOCR in recognition-only mode.
        
        Args:
            config: OCR configuration
        """
        self.config = config
        self.ocr = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """
        Initialize PaddleOCR with supported arguments only.
        
        CRITICAL: This version does NOT support det/rec in constructor.
        Recognition-only mode is enforced at call-time in recognize() method.
        """
        try:
            # Initialize with only supported arguments (no det/rec here)
            init_params = {
                'use_angle_cls': False,  # No angle classification needed
                'lang': self.config.OCR_LANG,
            }
            
            # Add GPU support if configured
            if self.config.OCR_USE_GPU:
                try:
                    init_params['use_gpu'] = True
                    self.ocr = PaddleOCR(**init_params)
                except (TypeError, ValueError):
                    # GPU not available or not supported - try CPU
                    init_params.pop('use_gpu', None)
                    self.ocr = PaddleOCR(**init_params)
            else:
                self.ocr = PaddleOCR(**init_params)
            
            print(f"✅ PaddleOCR initialized (version: {PADDLEOCR_VERSION}, recognition-only enforced at call-time)")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")
    
    def recognize(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Run OCR recognition on a pre-cropped image in TRUE recognition-only mode.
        
        CRITICAL: Recognition-only is enforced at call-time using det=False, rec=True.
        This prevents detection from running and failing on small ROIs.
        
        Args:
            image: Pre-cropped ROI image (can be grayscale or BGR)
            
        Returns:
            Tuple of (recognized_text, confidence_score)
            Returns (None, 0.0) if recognition fails
        """
        if self.ocr is None:
            self._initialize_ocr()
        
        try:
            # Convert to RGB format (PaddleOCR expects RGB)
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2RGB)
            else:
                return None, 0.0
            
            # Ensure minimum size (16x16 pixels minimum for recognition)
            min_size = 16
            h, w = image_rgb.shape[:2]
            if h < min_size or w < min_size:
                scale = max(min_size / h, min_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Ensure uint8 format
            if image_rgb.dtype != np.uint8:
                image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
            
            # CRITICAL: Enforce recognition-only at call-time
            # Recognition-only API: ocr([image], det=False, rec=True, cls=False)
            # Image MUST be passed as list, det/rec MUST be at call-time
            result = self.ocr.ocr(image_rgb, det=False,  cls=False)
            
            # Parse recognition-only output format: [[(text, conf)]]
            # Result structure: [page_results] where page_results = [(text, conf), ...]
            if result is None or not isinstance(result, list) or len(result) == 0:
                return None, 0.0
            
            # Get first page result
            page_result = result[0]
            if not isinstance(page_result, list) or len(page_result) == 0:
                return None, 0.0
            
            # Recognition-only format: list of (text, confidence) tuples
            # Iterate to find best confidence result
            best_text = None
            best_confidence = 0.0
            
            for item in page_result:
                # Recognition-only output: (text, conf) tuple
                if isinstance(item, tuple) and len(item) >= 2:
                    text = item[0]
                    conf = float(item[1])
                    if conf > best_confidence:
                        best_text = text
                        best_confidence = conf
                # Handle nested format: [[(text, conf)]] (some versions)
                elif isinstance(item, list) and len(item) > 0:
                    if isinstance(item[0], tuple) and len(item[0]) >= 2:
                        text = item[0][0]
                        conf = float(item[0][1])
                        if conf > best_confidence:
                            best_text = text
                            best_confidence = conf
            
            # Return best result if valid
            if best_text and str(best_text).strip() and best_confidence > 0:
                return str(best_text).strip(), float(best_confidence)
            else:
                return None, 0.0
                
        except Exception as e:
            # Return None on any error (suppress to avoid log spam)
            return None, 0.0


# ============================================================================
# Domain-Aware Post-Processing
# ============================================================================

class PIDTagValidator:
    """
    Validates and corrects P&ID tags using domain-specific rules.
    """
    
    def __init__(self, config: OCRConfig = OCRConfig()):
        self.config = config
        self.tag_patterns = [re.compile(pattern) for pattern in config.TAG_PATTERNS]
    
    def normalize_characters(self, text: str) -> Tuple[str, List[str]]:
        """
        Normalize common OCR character errors.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Tuple of (normalized_text, list_of_corrections_applied)
        """
        corrections = []
        normalized = text
        
        # Apply character normalization
        for wrong_char, correct_char in self.config.CHAR_NORMALIZATION.items():
            if wrong_char in normalized:
                # Only replace if it makes sense in context (digit positions)
                # This is a simple heuristic - can be enhanced
                normalized = normalized.replace(wrong_char, correct_char)
                corrections.append(f"{wrong_char}→{correct_char}")
        
        return normalized, corrections
    
    def validate_tag_format(self, text: str) -> bool:
        """
        Validate if text matches P&ID tag patterns.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text matches at least one P&ID tag pattern
        """
        if not text:
            return False
        
        # Check length constraints
        if len(text) < self.config.MIN_TAG_LENGTH or len(text) > self.config.MAX_TAG_LENGTH:
            return False
        
        # Check against patterns
        for pattern in self.tag_patterns:
            if pattern.match(text):
                return True
        
        return False
    
    def post_process(self, raw_text: str) -> Tuple[str, List[str], bool]:
        """
        Post-process OCR output for P&ID tags.
        
        Process:
        1. Remove whitespace and special characters (except hyphens)
        2. Normalize common OCR errors
        3. Validate against P&ID patterns
        4. Apply format corrections (add hyphens if missing)
        
        Args:
            raw_text: Raw OCR output
            
        Returns:
            Tuple of (processed_text, corrections_applied, is_valid)
        """
        if not raw_text:
            return "", [], False
        
        corrections = []
        
        # Step 1: Clean text (remove spaces, keep alphanumeric and hyphens)
        cleaned = re.sub(r'[^A-Z0-9\-]', '', raw_text.upper())
        
        if not cleaned:
            return "", [], False
        
        # Step 2: Normalize characters
        normalized, char_corrections = self.normalize_characters(cleaned)
        corrections.extend(char_corrections)
        
        # Step 3: Try to add hyphen if missing (common pattern: FT101 -> FT-101)
        if '-' not in normalized and len(normalized) >= 3:
            # Try to insert hyphen after 1-3 letters
            for i in range(1, min(4, len(normalized))):
                if normalized[i].isdigit():
                    normalized = normalized[:i] + '-' + normalized[i:]
                    corrections.append("Added hyphen")
                    break
        
        # Step 4: Validate
        is_valid = self.validate_tag_format(normalized)
        
        return normalized, corrections, is_valid


# ============================================================================
# Main Symbol-Guided Tag Extractor
# ============================================================================

class SymbolGuidedTagExtractor:
    """
    Main class for symbol-guided tag extraction.
    
    This class orchestrates the entire pipeline:
    1. Takes symbol bounding boxes from YOLO
    2. Expands bounding boxes to create ROIs
    3. Preprocesses ROIs
    4. Runs PaddleOCR recognition
    5. Post-processes and validates tags
    """
    
    def __init__(self, config: OCRConfig = OCRConfig()):
        """
        Initialize the tag extractor.
        
        Args:
            config: OCR configuration
        """
        self.config = config
        self.ocr_recognizer = PaddleOCRRecognizer(config)
        self.validator = PIDTagValidator(config)
    
    def extract(
        self,
        image: np.ndarray,
        symbol_bbox: Tuple[int, int, int, int],
        symbol_class: Optional[str] = None
    ) -> TagResult:
        """
        Extract tag for a single symbol.
        
        Args:
            image: Full P&ID image (BGR format)
            symbol_bbox: (x1, y1, x2, y2) bounding box of detected symbol
            symbol_class: Optional symbol class name (for logging)
            
        Returns:
            TagResult object with extracted tag and metadata
        """
        # Step 1: Expand bounding box to create ROI
        expanded_bbox = expand_bbox(symbol_bbox, image.shape, self.config)
        
        # Step 2: Crop ROI
        roi = crop_roi(image, expanded_bbox)
        
        # Skip if ROI is too small
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return TagResult(
                text="",
                confidence=0.0,
                bbox=expanded_bbox,
                raw_ocr_text="",
                is_valid=False,
                corrections_applied=["ROI too small"]
            )
        
        # Step 3: Preprocess ROI for OCR (try light preprocessing first)
        # Try with light preprocessing first (preserves more text features)
        preprocessed_roi = preprocess_for_ocr(roi, self.config, use_light_preprocessing=True)
        
        # Step 4: Run OCR recognition
        raw_text, ocr_confidence = self.ocr_recognizer.recognize(preprocessed_roi)
        
        # If light preprocessing fails, try with original image (minimal processing)
        if raw_text is None or ocr_confidence < self.config.MIN_CONFIDENCE:
            # Try with original ROI (just convert to RGB)
            if len(roi.shape) == 3:
                original_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                original_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            raw_text, ocr_confidence = self.ocr_recognizer.recognize(original_rgb)
        
        # If still fails, try with full preprocessing as last resort
        if raw_text is None or ocr_confidence < self.config.MIN_CONFIDENCE:
            preprocessed_roi_full = preprocess_for_ocr(roi, self.config, use_light_preprocessing=False)
            raw_text, ocr_confidence = self.ocr_recognizer.recognize(preprocessed_roi_full)
        
        # Debug: Log what we got from OCR (always log, even if None)
        if raw_text:
            print(f"  PaddleOCR recognized: '{raw_text}' (confidence: {ocr_confidence:.2f})")
        else:
            print(f"  PaddleOCR returned None (confidence: {ocr_confidence:.2f})")
        
        if raw_text is None or ocr_confidence < self.config.MIN_CONFIDENCE:
            return TagResult(
                text="",
                confidence=ocr_confidence,
                bbox=expanded_bbox,
                raw_ocr_text=raw_text or "",
                is_valid=False,
                corrections_applied=["Low OCR confidence"]
            )
        
        # Step 5: Post-process and validate
        processed_text, corrections, is_valid = self.validator.post_process(raw_text)
        
        # Debug: Log post-processing result
        if processed_text:
            print(f"  Post-processed: '{raw_text}' -> '{processed_text}' (valid: {is_valid})")
        elif raw_text:
            print(f"  Post-processing filtered out: '{raw_text}'")
        
        return TagResult(
            text=processed_text if processed_text else "",  # Return empty string instead of None
            confidence=ocr_confidence,
            bbox=expanded_bbox,
            raw_ocr_text=raw_text or "",  # Ensure not None
            is_valid=is_valid,
            corrections_applied=corrections
        )
    
    def extract_batch(
        self,
        image: np.ndarray,
        symbol_bboxes: List[Tuple[int, int, int, int]],
        symbol_classes: Optional[List[str]] = None
    ) -> List[TagResult]:
        """
        Extract tags for multiple symbols (batch processing).
        
        Args:
            image: Full P&ID image
            symbol_bboxes: List of (x1, y1, x2, y2) bounding boxes
            symbol_classes: Optional list of symbol class names
            
        Returns:
            List of TagResult objects (one per symbol)
        """
        if symbol_classes is None:
            symbol_classes = [None] * len(symbol_bboxes)
        
        results = []
        for bbox, symbol_class in zip(symbol_bboxes, symbol_classes):
            result = self.extract(image, bbox, symbol_class)
            results.append(result)
        
        return results


# ============================================================================
# Integration Helper Functions
# ============================================================================

def create_tag_extractor(config: Optional[OCRConfig] = None) -> SymbolGuidedTagExtractor:
    """
    Factory function to create a tag extractor instance.
    
    This is the recommended way to create an extractor, as it ensures
    proper initialization and can be easily mocked for testing.
    
    Args:
        config: Optional custom configuration
        
    Returns:
        Initialized SymbolGuidedTagExtractor instance
    """
    if config is None:
        config = OCRConfig()
    return SymbolGuidedTagExtractor(config)


# For backward compatibility and easier access
def extract_tag(
    image: np.ndarray,
    symbol_bbox: Tuple[int, int, int, int],
    extractor: Optional[SymbolGuidedTagExtractor] = None
) -> TagResult:
    """
    Convenience function for single tag extraction.
    
    Args:
        image: Full P&ID image
        symbol_bbox: Symbol bounding box
        extractor: Optional pre-initialized extractor (creates new one if None)
        
    Returns:
        TagResult object
    """
    if extractor is None:
        extractor = create_tag_extractor()
    return extractor.extract(image, symbol_bbox)
