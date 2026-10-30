import json
import tempfile
import unittest
from pathlib import Path

from .collect_records import collect


class RecordTests(unittest.TestCase):
    def test_collects_small_evidence_and_skips_weights_running_and_raw_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt, runs, out = root / "ckpt", root / "runs", root / "records"
            runs.mkdir()
            for name, status in (("done", "complete"), ("bad", "failed"), ("active", "running")):
                folder = ckpt / name
                folder.mkdir(parents=True)
                (folder / "train_log.json").write_text(json.dumps({"status": status, "config": {}, "epochs": []}))
                (folder / "adapter_model.safetensors").write_text("do not copy")
            (runs / "example.summary.json").write_text('{"sample_acc": 0.5}')
            (runs / "raw.jsonl").write_text("do not copy")
            result = collect(ckpt, runs, out)
            self.assertEqual(result["training_logs"], 2)
            self.assertEqual(result["evaluation_records"], 1)
            self.assertTrue((out / "checkpoints/done/train_log.json").is_file())
            self.assertFalse((out / "checkpoints/active").exists())
            self.assertFalse(list(out.rglob("*.safetensors")))
            self.assertFalse(list(out.rglob("*.jsonl")))
            self.assertIn("training-index.csv", (out / "SHA256SUMS").read_text())

    def test_rejects_output_inside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "outside source"):
                collect(root / "ckpt", root / "runs", root / "runs/records")


if __name__ == "__main__":
    unittest.main()
