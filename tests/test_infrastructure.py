import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from rlsm_immtrack.data import load_gt, natural_sort_key, resolve_sequence_paths
from rlsm_immtrack.experiment import (
    DEFAULT_CONFIG, inspect_dataset, load_config, prepare_output, read_predictions,
)


class InfrastructureTests(unittest.TestCase):
    def test_natural_order(self):
        self.assertEqual(sorted(["10.jpg", "2.jpg", "1.jpg"], key=natural_sort_key),
                         ["1.jpg", "2.jpg", "10.jpg"])

    def test_polygon_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.txt"
            path.write_text("1,2,11,2,11,22,1,22\n")
            np.testing.assert_array_equal(load_gt(path), [[1, 2, 10, 20]])

    def test_both_layouts_and_explicit_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, legacy in (("legacy", True), ("a2a", False)):
                dataset = root / name
                image_dir = dataset / ("img/s1" if legacy else "video01/img")
                image_dir.mkdir(parents=True)
                annotation = dataset / ("anno/s1.txt" if legacy else "video01/groundtruth.txt")
                annotation.parent.mkdir(exist_ok=True)
                annotation.write_text("1,2,10,20\n1,2,10,20\n")
                (image_dir / "1.jpg").write_bytes(b"placeholder")
                with self.assertRaises(ValueError):
                    inspect_dataset(dataset)
                _, inventory = inspect_dataset(dataset, alignment="common")
                self.assertEqual(inventory[0]["frames"], 1)

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            prepare_output(output, {"x": 1}, False)
            with self.assertRaises(FileExistsError):
                prepare_output(output, {"x": 1}, False)
            prepare_output(output, {"x": 1}, True)
            with self.assertRaises(ValueError):
                prepare_output(output, {"x": 2}, True)

    def test_prediction_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            path.write_text("x,y,w,h\n1,2,3,4\n")
            self.assertEqual(read_predictions(path, 1).shape, (1, 4))
            path.write_text("x,y,w,h,tracker_seconds\n1,2,3,4,0.1\n")
            np.testing.assert_array_equal(read_predictions(path, 1), [[1, 2, 3, 4]])
            with self.assertRaises(ValueError):
                read_predictions(path, 2)
            path.write_text("nan,2,3,4\n")
            with self.assertRaises(ValueError):
                read_predictions(path, 1)

    def test_configuration(self):
        self.assertEqual(load_config(DEFAULT_CONFIG)["entry_streak"], 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"entry_streak": 0}))
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(json.dumps({"unknown": 1}))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
