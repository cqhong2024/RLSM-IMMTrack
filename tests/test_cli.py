import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rlsm_immtrack.cli import main


class CLITests(unittest.TestCase):
    def make_data(self, root):
        sequence = root / "data/video01"
        (sequence / "img").mkdir(parents=True)
        (sequence / "img/1.jpg").write_bytes(b"not decoded in this metric-only test")
        (sequence / "groundtruth.txt").write_text("1,2,10,10\n")
        predictions = root / "predictions"
        predictions.mkdir()
        (predictions / "video01.csv").write_text("x,y,w,h\n1,2,10,10\n")
        return root / "data", predictions

    def test_validate_and_score_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions = self.make_data(root)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["validate", "--dataset-root", str(data),
                      "--output", str(root / "validation.json")])
                main(["score", "--dataset-root", str(data), "--predictions", str(predictions),
                      "--output", str(root / "scores")])
            report = json.loads((root / "scores/metrics.json").read_text())
            self.assertEqual(report["aggregate"]["success_auc"], 1.)
            self.assertEqual(report["aggregate"]["sequences"], 1)
            self.assertEqual(len(report["curves"]["success"]["values"]), 21)
            self.assertEqual(len(report["curves"]["normalized_precision"]["values"]), 51)

    def test_score_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions = self.make_data(root)
            output = root / "scores"
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("preserve")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["score", "--dataset-root", str(data), "--predictions", str(predictions),
                      "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
