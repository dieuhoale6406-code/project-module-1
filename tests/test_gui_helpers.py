import os
import sys
import tempfile
import unittest
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
from core.manager import VectorEngineManager

# Direct imports from new modular structure
from ui.dataset import (
    consume_flash_message,
    load_glove_subset,
    parse_glove_line,
    parse_vector_input,
    set_flash_message,
)
from ui.search import (
    ALGORITHM_DISPLAY_NAMES,
    compute_top_k_bounds,
    format_algorithm_status_badge,
)

# Backward-compatible re-exports from app
from app import (
    ALGORITHM_DISPLAY_NAMES as APP_ALGORITHM_DISPLAY_NAMES,
    format_algorithm_status_badge as app_format_badge,
    load_glove_subset as app_load_glove,
    parse_glove_line as app_parse_line,
)


class TestGuiHelpers(unittest.TestCase):
    def test_import_compatibility(self):
        """Verify exports work from both ui package and app root module."""
        self.assertIs(app_parse_line, parse_glove_line)
        self.assertIs(app_load_glove, load_glove_subset)
        self.assertIs(app_format_badge, format_algorithm_status_badge)
        self.assertEqual(APP_ALGORITHM_DISPLAY_NAMES, ALGORITHM_DISPLAY_NAMES)
        expected_keys = {"linear_scan", "kdtree", "lsh", "ivf", "hnsw"}
        self.assertTrue(expected_keys.issubset(set(ALGORITHM_DISPLAY_NAMES.keys())))

    def test_parse_glove_line_standard(self):
        line = "the " + " ".join([f"{0.1 * i:.4f}" for i in range(50)])
        parsed = parse_glove_line(line, expected_dim=50)
        self.assertIsNotNone(parsed)
        label, values = parsed
        self.assertEqual(label, "the")
        self.assertEqual(len(values), 50)
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 0.1)

    def test_parse_glove_line_with_whitespace_label(self):
        line = "new york city " + " ".join(["0.5"] * 50)
        parsed = parse_glove_line(line, expected_dim=50)
        self.assertIsNotNone(parsed)
        label, values = parsed
        self.assertEqual(label, "new york city")
        self.assertEqual(len(values), 50)
        self.assertEqual(values[0], 0.5)

    def test_parse_glove_line_invalid_or_empty(self):
        self.assertIsNone(parse_glove_line("", expected_dim=50))
        self.assertIsNone(parse_glove_line("word 0.1 0.2", expected_dim=50))  # too short
        self.assertIsNone(parse_glove_line("word " + " ".join(["nan"] + ["0.1"] * 49), expected_dim=50))

    def test_load_glove_subset_streaming_and_limit(self):
        manager = VectorEngineManager()
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            for i in range(15):
                f.write(f"word{i} " + " ".join([f"{0.01 * j:.4f}" for j in range(50)]) + "\n")
            f.write("bad_line\n")
            temp_path = f.name

        try:
            loaded, skipped = load_glove_subset(manager, temp_path, limit=10, expected_dim=50)
            self.assertEqual(loaded, 10)
            self.assertEqual(manager.record_count, 10)
            self.assertEqual(manager.dimension, 50)
            self.assertIsNotNone(manager.get_record("word0"))
            self.assertIsNotNone(manager.get_record("word9"))
            self.assertIsNone(manager.get_record("word10"))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_glove_subset_missing_file(self):
        manager = VectorEngineManager()
        with self.assertRaises(FileNotFoundError):
            load_glove_subset(manager, "non_existent_path_xyz.txt", limit=10)

    def test_format_algorithm_status_badge(self):
        self.assertIn("Sẵn sàng", format_algorithm_status_badge("ready"))
        self.assertIn("Chưa sẵn sàng", format_algorithm_status_badge("registered"))
        self.assertIn("Chưa sẵn sàng", format_algorithm_status_badge("not_implemented"))
        self.assertIn("Không khả dụng", format_algorithm_status_badge("unavailable"))
        self.assertIn("Chưa sẵn sàng", format_algorithm_status_badge("unregistered"))

    def test_compute_top_k_bounds(self):
        """Top-K must never exceed available records, and handle 0 and 1 gracefully."""
        self.assertEqual(compute_top_k_bounds(0), (0, 0, 0))
        self.assertEqual(compute_top_k_bounds(1), (1, 1, 1))
        self.assertEqual(compute_top_k_bounds(2), (1, 2, 2))
        self.assertEqual(compute_top_k_bounds(3), (1, 3, 3))
        self.assertEqual(compute_top_k_bounds(10), (1, 10, 5))
        self.assertEqual(compute_top_k_bounds(100), (1, 50, 5))

    def test_parse_vector_input_valid(self):
        vals, err = parse_vector_input("0.1 0.2 -0.5", expected_dim=3)
        self.assertIsNone(err)
        self.assertEqual(vals, [0.1, 0.2, -0.5])

        # Comma-separated
        vals, err = parse_vector_input("1.0, 2.5, 3.0", expected_dim=3)
        self.assertIsNone(err)
        self.assertEqual(vals, [1.0, 2.5, 3.0])

    def test_parse_vector_input_invalid(self):
        # Empty
        vals, err = parse_vector_input("")
        self.assertIsNotNone(err)
        self.assertIsNone(vals)

        # Non-numeric
        vals, err = parse_vector_input("0.1 abc 0.3")
        self.assertIsNotNone(err)
        self.assertIn("không phải là số thực", err)

        # Non-finite
        vals, err = parse_vector_input("0.1 nan 0.3")
        self.assertIsNotNone(err)
        self.assertIn("NaN", err)

        # Dimension mismatch
        vals, err = parse_vector_input("0.1 0.2", expected_dim=3)
        self.assertIsNotNone(err)
        self.assertIn("không khớp", err)

    def test_full_dataset_label_lookup(self):
        """Manager lookup works across all records, with whitespace and case-insensitivity."""
        manager = VectorEngineManager()
        manager.add_record(label="new york city", values=[0.1, 0.2, 0.3])
        manager.add_record(label="tokyo", values=[0.4, 0.5, 0.6])

        # Exact match
        rec = manager.get_record("new york city")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.label, "new york city")

        # Case-insensitive match with whitespace
        rec_upper = manager.get_record("New York City")
        self.assertIsNotNone(rec_upper)
        self.assertEqual(rec_upper.id, rec.id)

        # Non-existent
        self.assertIsNone(manager.get_record("paris"))

    def test_flash_message_cycle(self):
        """Flash messages are saved in session state and cleared when consumed."""
        set_flash_message("Thao tác thành công", "success")
        flash = consume_flash_message()
        self.assertEqual(flash, ("success", "Thao tác thành công"))
        # Consuming again returns None
        self.assertIsNone(consume_flash_message())


if __name__ == "__main__":
    unittest.main()
