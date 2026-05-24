"""
Example Usage: Symbol-Guided OCR Integration

This file demonstrates how to use the symbol-guided OCR module
in various scenarios, from simple single-symbol extraction to
full pipeline integration.
"""

import cv2
import numpy as np
from symbol_guided_ocr import (
    SymbolGuidedTagExtractor,
    TagResult,
    OCRConfig,
    create_tag_extractor
)
from ocr_integration import HybridOCREngine, integrate_with_existing_pipeline


# ============================================================================
# Example 1: Basic Single Symbol Tag Extraction
# ============================================================================

def example_single_symbol_extraction():
    """Extract tag for a single symbol"""
    print("=" * 60)
    print("Example 1: Single Symbol Tag Extraction")
    print("=" * 60)
    
    # Load image
    image = cv2.imread('path/to/pid_image.jpg')
    if image is None:
        print("⚠️ Could not load image. Using dummy example.")
        return
    
    # Symbol bounding box from YOLO detection
    # Format: (x1, y1, x2, y2)
    symbol_bbox = (100, 200, 150, 250)  # Example coordinates
    
    # Create extractor
    extractor = create_tag_extractor()
    
    # Extract tag
    result = extractor.extract(image, symbol_bbox, symbol_class='gate_valve')
    
    # Print results
    print(f"Symbol: gate_valve at {symbol_bbox}")
    print(f"Extracted Tag: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Is Valid: {result.is_valid}")
    print(f"Raw OCR: {result.raw_ocr_text}")
    print(f"Corrections: {result.corrections_applied}")
    print()


# ============================================================================
# Example 2: Batch Processing Multiple Symbols
# ============================================================================

def example_batch_extraction():
    """Extract tags for multiple symbols"""
    print("=" * 60)
    print("Example 2: Batch Symbol Tag Extraction")
    print("=" * 60)
    
    # Load image
    image = cv2.imread('path/to/pid_image.jpg')
    if image is None:
        print("⚠️ Could not load image. Using dummy example.")
        return
    
    # Multiple symbol detections from YOLO
    symbol_detections = [
        {'bbox': (100, 200, 150, 250), 'class_name': 'gate_valve', 'conf': 0.9},
        {'bbox': (300, 150, 350, 200), 'class_name': 'pump', 'conf': 0.85},
        {'bbox': (500, 300, 550, 350), 'class_name': 'pressure_gauge', 'conf': 0.92},
    ]
    
    # Create extractor
    extractor = create_tag_extractor()
    
    # Extract tags for all symbols
    bboxes = [det['bbox'] for det in symbol_detections]
    classes = [det['class_name'] for det in symbol_detections]
    
    results = extractor.extract_batch(image, bboxes, classes)
    
    # Print results
    for i, (detection, result) in enumerate(zip(symbol_detections, results)):
        print(f"\nSymbol {i+1}: {detection['class_name']}")
        print(f"  BBox: {detection['bbox']}")
        print(f"  Tag: {result.text}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Valid: {result.is_valid}")
    print()


# ============================================================================
# Example 3: Integration with Existing YOLO Pipeline
# ============================================================================

def example_yolo_integration():
    """Integrate with existing YOLO detection pipeline"""
    print("=" * 60)
    print("Example 3: YOLO Pipeline Integration")
    print("=" * 60)
    
    # Simulate YOLO detection results (existing format)
    yolo_data = [
        {"bbox": (100, 200, 150, 250), "class_name": "gate_valve", "conf": 0.9},
        {"bbox": (300, 150, 350, 200), "class_name": "pump", "conf": 0.85},
    ]
    
    # Load image
    image = cv2.imread('path/to/pid_image.jpg')
    if image is None:
        print("⚠️ Could not load image. Using dummy example.")
        return
    
    # Option A: Use hybrid engine (recommended)
    hybrid_engine = HybridOCREngine()
    enhanced_detections = hybrid_engine.extract_tags_for_symbols(
        image,
        yolo_data,
        fallback_ocr_results=None  # Can provide existing DocTR results
    )
    
    # Print enhanced results
    for det in enhanced_detections:
        print(f"Symbol: {det['class_name']}")
        print(f"  Tag: {det.get('tag', 'N/A')}")
        print(f"  Tag Confidence: {det.get('tag_confidence', 0.0):.2f}")
        print(f"  Source: {det.get('tag_source', 'unknown')}")
        print()
    
    # Option B: Use integration function (drop-in replacement)
    enhanced = integrate_with_existing_pipeline(
        image,
        yolo_data,
        existing_ocr_results=None,
        use_hybrid=True
    )
    print()


# ============================================================================
# Example 4: Custom Configuration
# ============================================================================

def example_custom_config():
    """Use custom OCR configuration"""
    print("=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    custom_config = OCRConfig()
    custom_config.EXPANSION_RIGHT_PX = 200  # More space for tags
    custom_config.EXPANSION_LEFT_PX = 30
    custom_config.MIN_CONFIDENCE = 0.6  # Higher confidence threshold
    custom_config.OCR_USE_GPU = True  # Use GPU if available
    
    # Create extractor with custom config
    extractor = SymbolGuidedTagExtractor(custom_config)
    
    # Use extractor as normal
    image = cv2.imread('path/to/pid_image.jpg')
    if image is None:
        print("⚠️ Could not load image.")
        return
    
    symbol_bbox = (100, 200, 150, 250)
    result = extractor.extract(image, symbol_bbox)
    
    print(f"Tag: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")
    print()


# ============================================================================
# Example 5: Integration into Existing app.py (Minimal Changes)
# ============================================================================

def example_app_integration():
    """
    Example showing how to modify existing digitize_pid_image() function.
    
    This shows the minimal changes needed to integrate symbol-guided OCR
    into the existing pipeline.
    """
    print("=" * 60)
    print("Example 5: Integration into app.py")
    print("=" * 60)
    
    print("""
    # In app.py, modify the graph construction section:
    
    # OLD CODE (lines ~406-416):
    # best_tag_data = min(
    #     [(ocr_res, np.sqrt(...)) for ocr_res in ocr_results_list],
    #     key=lambda x: x[1],
    #     default=(None, float('inf'))
    # )
    # best_tag, best_tag_conf = ...
    
    # NEW CODE (with symbol-guided OCR):
    from ocr_integration import integrate_with_existing_pipeline
    
    # Extract tags using symbol-guided OCR
    enhanced_detections = integrate_with_existing_pipeline(
        original_img,
        yolo_data,
        existing_ocr_results=ocr_results_list,  # Use existing OCR as fallback
        use_hybrid=True
    )
    
    # Then use enhanced_detections instead of proximity matching
    for detection in enhanced_detections:
        node_id = node_id_counter
        x1, y1, x2, y2 = detection["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Tags are already extracted and assigned
        best_tag = detection.get('tag')
        best_tag_conf = detection.get('tag_confidence', 0.0)
        needs_review = best_tag_conf < CONFIG.TAG_CONFIDENCE_REVIEW_THRESHOLD or best_tag is None
        # ... rest of the code remains the same
    """)


# ============================================================================
# Example 6: Testing and Validation
# ============================================================================

def example_testing():
    """Example of how to test the OCR module"""
    print("=" * 60)
    print("Example 6: Testing and Validation")
    print("=" * 60)
    
    # Create extractor
    extractor = create_tag_extractor()
    
    # Test with known symbol positions
    test_cases = [
        {
            'image_path': 'test_images/pid_001.jpg',
            'symbol_bbox': (100, 200, 150, 250),
            'expected_tag': 'FT-101',  # Ground truth
            'symbol_class': 'flow_transmitter'
        },
        # Add more test cases...
    ]
    
    results = []
    for test_case in test_cases:
        image = cv2.imread(test_case['image_path'])
        if image is None:
            continue
        
        result = extractor.extract(
            image,
            test_case['symbol_bbox'],
            test_case['symbol_class']
        )
        
        is_correct = result.text == test_case['expected_tag']
        results.append({
            'expected': test_case['expected_tag'],
            'got': result.text,
            'correct': is_correct,
            'confidence': result.confidence
        })
    
    # Print test results
    print("Test Results:")
    for i, r in enumerate(results, 1):
        status = "✅" if r['correct'] else "❌"
        print(f"{status} Test {i}: Expected '{r['expected']}', Got '{r['got']}' "
              f"(Confidence: {r['confidence']:.2f})")
    
    accuracy = sum(r['correct'] for r in results) / len(results) if results else 0
    print(f"\nOverall Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Symbol-Guided OCR - Example Usage")
    print("=" * 60 + "\n")
    
    # Run examples (comment out if image files not available)
    # example_single_symbol_extraction()
    # example_batch_extraction()
    # example_yolo_integration()
    # example_custom_config()
    example_app_integration()
    # example_testing()
    
    print("\n" + "=" * 60)
    print("Examples completed. See code comments for details.")
    print("=" * 60)
