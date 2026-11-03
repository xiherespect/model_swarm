"""CPU-only checks for the pre-result question audit and subset manifest."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/audit_racer_data.py"
spec = importlib.util.spec_from_file_location("audit_racer_data", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DataAuditTest(unittest.TestCase):
    def test_normalized_stem_and_row_multiplicity(self):
        dev = [{"question": " Café\n  ABC "}]
        test = [{"question": "café abc"}, {"question": "CAFÉ  abc"},
                {"question": "different"}]
        matched = audit.overlap(dev, test, lambda r: audit.stem(r["question"], "gsm8k"))
        self.assertEqual(matched["test_overlap_rows"], 2)
        self.assertEqual(matched["shared_unique"], 1)
        self.assertEqual(audit.overlap(dev, test, audit.question)["test_overlap_rows"], 0)
        with self.assertRaises(ValueError):
            audit.stem("missing footer", audit.KCROSS)

    def test_frozen_manifest_recomputes_and_has_clean_unique_selection(self):
        frozen = json.loads(audit.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(audit.build_manifest(), frozen)
        selected = frozen["kcross_subset"]["selected"]
        self.assertEqual(len(selected), 564)
        indices = [row["index"] for row in selected]
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0 <= index < 1000 for index in indices))
        data = json.loads((audit.DATA / "knowledge_crosswords.json").read_text())
        dev = {audit.stem(row["question"], audit.KCROSS) for row in data["dev"]}
        clean = [audit.stem(data["test"][row["index"]]["question"], audit.KCROSS)
                 for row in selected]
        self.assertEqual(len(clean), len(set(clean)))
        self.assertFalse(set(clean) & dev)
        self.assertEqual(frozen["datasets"]["knowledge_crosswords"]["normalized_stem"]
                         ["test_overlap_rows"], 149)
        self.assertEqual(set(frozen["tasks"]), set(audit.TASKS))

    def test_verify_fails_closed_if_manifest_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "audit.json"
            freeze = subprocess.run([sys.executable, "-I", str(SCRIPT), "--freeze",
                                     "--manifest", str(dest)], capture_output=True, text=True)
            self.assertEqual(freeze.returncode, 0, freeze.stderr)
            verify = subprocess.run([sys.executable, "-I", str(SCRIPT), "--verify",
                                     "--manifest", str(dest)], capture_output=True, text=True)
            self.assertEqual(verify.returncode, 0, verify.stderr)
            value = json.loads(dest.read_text())
            value["sources"]["gsm8k"]["sha256"] = "0" * 64
            dest.write_text(json.dumps(value))
            verify = subprocess.run([sys.executable, "-I", str(SCRIPT), "--verify",
                                     "--manifest", str(dest)], capture_output=True, text=True)
            self.assertNotEqual(verify.returncode, 0)
            self.assertIn("changed since freeze", verify.stderr)
            freeze = subprocess.run([sys.executable, "-I", str(SCRIPT), "--freeze",
                                     "--manifest", str(dest)], capture_output=True, text=True)
            self.assertNotEqual(freeze.returncode, 0)


if __name__ == "__main__":
    unittest.main()
