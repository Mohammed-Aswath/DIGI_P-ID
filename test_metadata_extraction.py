import re
import unittest
from pathlib import Path
import sys

import cv2

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from metadata_extractor import MetadataExtractor, integrate_paddle_client


class FakeOCRClient:
    """Deterministic fake OCR client for snapshot testing."""

    def __init__(self):
        self.calls = 0
        self.payloads = [
            {
                "text": "1. VERIFY PIPING ROUTE\n2. CHECK WELD QUALITY\nALL DIMENSIONS IN MM",
                "confidence": 0.91,
                "raw_text": "notes_raw",
            },
            {
                "text": "REV  DATE  DESC\nA  2024-01-10  INITIAL ISSUE\nB  2024-02-12  UPDATED NOTES",
                "confidence": 0.87,
                "raw_text": "revision_raw",
            },
            {
                "text": "PROJECT: SAMPLE PROJECT\nDRAWING: SAMPLE\nSCALE: NONE",
                "confidence": 0.93,
                "raw_text": "title_raw",
            },
        ]

    def __call__(self, image):
        idx = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return self.payloads[idx]


class MetadataExtractorTests(unittest.TestCase):
    def setUp(self):
        self.root = THIS_DIR
        self.fixture_path = self.root / "tests" / "fixtures" / "part1_expected_metadata.md"
        self.image_path = self.root / "uploads" / "part1 img.jpg"
        self.extractor = MetadataExtractor(
            paddle_ocr_client=FakeOCRClient(),
            enable_table_detection=False,
        )

    def _normalize_markdown(self, value: str) -> str:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[ \t]+$", "", value, flags=re.MULTILINE)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip() + "\n"

    def test_wrapper_supports_dict_result(self):
        def dict_client(_image):
            return {"text": "FT-101", "confidence": 0.88, "raw_text": "FT101"}

        wrapped = integrate_paddle_client(dict_client)
        result = wrapped(None)
        self.assertEqual(result["text"], "FT-101")
        self.assertAlmostEqual(result["confidence"], 0.88, places=3)
        self.assertEqual(result["raw_text"], "FT101")

    def test_wrapper_supports_legacy_list_result(self):
        def legacy_client(_image):
            return [[None, ("FT-201", 0.80)], [None, ("PT-300", 0.60)]]

        wrapped = integrate_paddle_client(legacy_client)
        result = wrapped(None)
        self.assertEqual(result["text"], "FT-201 PT-300")
        self.assertGreater(result["confidence"], 0.0)

    def test_right_panel_region_segmentation(self):
        regions = self.extractor._get_right_panel_regions((4561, 7168, 3))
        bbox, _ = self.extractor._crop_region(
            image=cv2.imread(str(self.image_path)),
            box_ratios=regions["notes"],
        )
        self.assertEqual(bbox, (5662, 91, 7060, 3375))

    def test_parsers(self):
        notes = self.extractor._parse_notes("1. FIRST NOTE\n2) SECOND NOTE\nGENERAL WARNING")
        self.assertEqual(notes["numbered"], ["FIRST NOTE", "SECOND NOTE"])
        self.assertEqual(notes["bullets"], ["GENERAL WARNING"])

        revision = self.extractor._parse_revision_table(["REV  DATE  DESC", "A  2024-01-10  ISSUE"])
        self.assertEqual(revision["headers"], ["REV", "DATE", "DESC"])
        self.assertEqual(revision["rows"][0], ["A", "2024-01-10", "ISSUE"])

        title = self.extractor._parse_title_block(["PROJECT: SAMPLE", "DRAWING  SAMPLE"])
        self.assertEqual(title["fields"][0]["field"], "PROJECT")
        self.assertEqual(title["fields"][0]["value"], "SAMPLE")
        self.assertEqual(title["fields"][1]["field"], "DRAWING")
        self.assertEqual(title["fields"][1]["value"], "SAMPLE")

    def test_part1_snapshot_markdown(self):
        image = cv2.imread(str(self.image_path))
        self.assertIsNotNone(image, "part1 img.jpg must exist for integration snapshot test")

        elements = self.extractor.extract_all_metadata(image)
        markdown = self.extractor.get_markdown_output(elements)
        normalized = self._normalize_markdown(markdown)
        expected = self._normalize_markdown(self.fixture_path.read_text(encoding="utf-8"))

        self.assertEqual(normalized, expected)
        self.assertIn("# Notes", normalized)
        self.assertIn("# Revision History", normalized)
        self.assertIn("# Title Block", normalized)
        self.assertTrue("1. " in normalized, "Expected numbered notes")

        json_out = self.extractor.get_json_output(elements)
        self.assertIn("sections", json_out)
        self.assertGreaterEqual(json_out["summary"]["notes_items_count"], 1)


if __name__ == "__main__":
    unittest.main()
