"""CPU/stdlib guards; these do not claim a GPU DD sampling run."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location("sample_early", Path(__file__).with_name("sample_early.py"))
SAMPLE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SAMPLE)


class SnapshotGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="test_sampler_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.snapshot = self.root / "outputs/dd_earlystop_snapshot_fixture"
        self.snapshot.mkdir(parents=True)
        (self.root / "higgs/outputs").mkdir(parents=True)
        self.out = self.root / "higgs/outputs/new_sampling"
        self.hashes = {}

        def source(rel, content):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self.hashes[str(path)] = SAMPLE.sha(path)
            return path

        self.source = source
        protected_path = source("protected.py", "fixed protected source")
        driver = source(SAMPLE.DRIVER_REL, "fixed formal driver")
        manifest = source(SAMPLE.MANIFEST_REL, json.dumps({
            "dataset": "dd", "splits": {"train": list(range(587)), "val": list(range(147)), "test": list(range(184))},
        }))
        self.digest = SAMPLE.sha(manifest)
        for h in ("2", "1", "1.5"):
            for split in ("train", "val", "test"):
                source(f"data/dd/processed/proc_{split}_v1_h{h}_max500_{self.digest}.pt", f"{h}:{split}")
        for rel in ("higgs/sample_main.py", "higgs/interface_ablation.py"):
            source(rel, "unchanged sampler source")
        source(f"outputs/{SAMPLE.SMOKE}/input_hash_check.json", json.dumps({"unchanged": True, "sha256": dict(self.hashes)}))

        self.selection = {
            "root": str(self.root), "run": str(self.root / "outputs" / SAMPLE.RUN),
            "snapshot": str(self.snapshot), "manifest_sha256": self.digest, "checkpoints": {},
        }
        for key in ("h2", "h1", "glue"):
            best = source(f"outputs/{self.snapshot.name}/{key}/best.ckpt", key + "best")
            last = source(f"outputs/{self.snapshot.name}/{key}/last.ckpt", key + "last")
            record = {"best": str(best), "best_sha256": SAMPLE.sha(best), "saved_last": str(last),
                      "last_sha256": SAMPLE.sha(last), "epoch": 100, "global_step": 123,
                      "training_last_epoch": 300 if key == "h1" else 499,
                      "monitor": SAMPLE.MONITORS[key], "best_score": 0.5, "strict_reload": True}
            self.selection["checkpoints"][key] = record
            if key != "h1":
                source(f"outputs/{SAMPLE.RUN}/{key}/formal_result.json", json.dumps({
                    "epoch": 499, "strict_reload": True, "strict_best_reload": True,
                    "manifest_sha256": self.digest, "best_sha256": record["best_sha256"],
                    "last_sha256": record["last_sha256"], "best_epoch": 100,
                }))
        self.selection["input_sha256"] = self.hashes
        self.audit = {"status": "h1_stopped", "snapshot_verified": True}
        self.patches = patch.multiple(SAMPLE, ROOT=self.root, MANIFEST_SHA=self.digest,
                                      PROTECTED={"protected.py": SAMPLE.sha(protected_path)}, DRIVER_SHA=SAMPLE.sha(driver))
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.write_metadata()

    def write_metadata(self):
        selection_path = self.snapshot / "selection.json"
        selection_path.write_text(json.dumps(self.selection), encoding="utf-8")
        self.audit["selection_sha256"] = SAMPLE.sha(selection_path)
        (self.snapshot / "stop_audit.json").write_text(json.dumps(self.audit), encoding="utf-8")

    def validate(self, count=184):
        return SAMPLE.validate_snapshot(self.root, self.snapshot, self.out, count)

    def test_true_saved_epoch_and_provenance_included(self):
        selection, audit, _, hashes = self.validate()
        self.assertEqual(selection["checkpoints"]["h1"]["training_last_epoch"], 300)
        self.assertEqual(audit["status"], "h1_stopped")
        for path in (self.snapshot / "selection.json", self.snapshot / "stop_audit.json", Path(SAMPLE.__file__).resolve()):
            self.assertEqual(hashes[str(path)], SAMPLE.sha(path))
        self.validate(1)

    def test_reject_changed_source(self):
        (self.root / "protected.py").write_text("changed")
        with self.assertRaises(AssertionError):
            self.validate()

    def test_reject_changed_selection(self):
        (self.snapshot / "selection.json").write_text(json.dumps({**self.selection, "extra": True}))
        with self.assertRaises(AssertionError):
            self.validate()

    def test_reject_false_499_earlystop_and_missing_completed_proof(self):
        self.selection["checkpoints"]["h1"]["training_last_epoch"] = 499
        self.write_metadata()
        with self.assertRaises(AssertionError):
            self.validate()
        self.audit["status"] = "already_completed"
        self.write_metadata()
        with self.assertRaises(AssertionError):
            self.validate()

    def test_reject_incomplete_h2(self):
        self.selection["checkpoints"]["h2"]["training_last_epoch"] = 498
        self.write_metadata()
        with self.assertRaises(AssertionError):
            self.validate()

    def test_already_completed_requires_real_h1_formal_proof(self):
        record = self.selection["checkpoints"]["h1"]
        record["training_last_epoch"] = 499
        self.audit["status"] = "already_completed"
        self.source(f"outputs/{SAMPLE.RUN}/h1/formal_result.json", json.dumps({
            "epoch": 499, "strict_reload": True, "strict_best_reload": True,
            "manifest_sha256": self.digest, "best_sha256": record["best_sha256"],
            "last_sha256": record["last_sha256"], "best_epoch": record["epoch"],
        }))
        self.write_metadata()
        self.validate()

    def test_reject_unverified_snapshot(self):
        self.audit["snapshot_verified"] = False
        self.write_metadata()
        with self.assertRaises(AssertionError):
            self.validate()

    def test_reject_existing_output_and_other_counts(self):
        with self.assertRaises(AssertionError):
            self.validate(183)
        self.out.mkdir()
        with self.assertRaises(AssertionError):
            self.validate()


class LockGuards(unittest.TestCase):
    def test_inherited_fd_reuses_same_file_description_without_acquire(self):
        fake_fcntl = SimpleNamespace(flock=Mock(), LOCK_EX=2, LOCK_NB=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="test_lock_") as temp:
            root = Path(temp)
            path = root / "logs/sampling/dd_mean_hog_fullchain.lock"
            path.parent.mkdir(parents=True)
            with path.open("a+") as inherited, patch.dict(sys.modules, {"fcntl": fake_fcntl}):
                with SAMPLE.sampling_lock(root, inherited.fileno()) as held:
                    self.assertEqual(os.fstat(held.fileno()).st_ino, path.stat().st_ino)
                fake_fcntl.flock.assert_not_called()
                self.assertFalse(inherited.closed)

    def test_inherited_wrong_fd_fails(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="test_lock_") as temp:
            root = Path(temp)
            path = root / "logs/sampling/dd_mean_hog_fullchain.lock"
            path.parent.mkdir(parents=True)
            path.touch()
            with (root / "other.lock").open("a+") as other, patch.dict(sys.modules, {"fcntl": SimpleNamespace()}):
                with self.assertRaises(AssertionError):
                    SAMPLE.sampling_lock(root, other.fileno())


class MaskGuards(unittest.TestCase):
    class Values:
        def __init__(self, values, dtype="float"):
            self.values, self.dtype = values, dtype

        def __eq__(self, other):
            return MaskGuards.Values([v == other for v in self.values], "bool")

        def __or__(self, other):
            return MaskGuards.Values([x or y for x, y in zip(self.values, other.values)], "bool")

        def all(self):
            return SimpleNamespace(item=lambda: all(self.values))

        def bool(self):
            return MaskGuards.Values([bool(v) for v in self.values], "bool")

    def torch(self):
        import math
        return SimpleNamespace(bool="bool", is_tensor=lambda obj: isinstance(obj, self.Values),
                               isfinite=lambda obj: self.Values([math.isfinite(v) for v in obj.values], "bool"))

    def test_only_finite_binary_float_representation_is_restored(self):
        items = [SimpleNamespace(pos=self.Values([0.0, 1.0])), SimpleNamespace(pos=self.Values([True, False], "bool"))]
        boolean_original = items[1].pos
        self.assertEqual(SAMPLE.restore_binary_pos(items, self.torch()), 1)
        self.assertEqual(items[0].pos.values, [False, True])
        self.assertIs(items[1].pos, boolean_original)
        for invalid in ([0, 0.1], [0, float("nan")], [0, float("inf")]):
            with self.assertRaises(AssertionError):
                SAMPLE.restore_binary_pos([SimpleNamespace(pos=self.Values(invalid))], self.torch())


if __name__ == "__main__":
    unittest.main()
