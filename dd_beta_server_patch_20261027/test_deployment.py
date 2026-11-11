"""CPU-only deployment checks; no datasets, GPU jobs or production edits.

Installer/launcher helpers run only against synthetic in-memory manifests and
temporary files. The three real-data runners are inspected with AST and are
never imported, since their module bodies perform server work.
"""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
DRIVERS = HERE / "payload/tools/diagnostics"


def load_helper(name):
    spec = importlib.util.spec_from_file_location("dd_deployment_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSTALL = load_helper("install_dd_beta")
LAUNCH = load_helper("launch_dd")


def synthetic_manifest():
    """Only schema fixtures, not real DD graph records or partitions."""
    ids = list(range(1, 919))
    return {
        "dataset": "dd", "schema_version": 1, "source_sha256": INSTALL.RAW_SHA,
        "settings": {"resolution": 2.0, "split_seed": 1234, "louvain_seed": 0,
                     "weight": None, "node_labels": "dummy", "graph_condition": "constant"},
        "splits": {"train": ids[:587], "val": ids[587:734], "test": ids[734:]},
        "graphs": [{"id": i, "n": 100, "edges": [[0, 1]]} for i in ids],
        "partitions": {str(i): [list(range(50)), list(range(50, 100))] for i in ids},
    }


def parse_driver(kind):
    path = DRIVERS / ("dd_beta_" + kind + "_20261027.py")
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def string_constants(tree):
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def assigned_literal(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("Missing literal assignment: " + name)


class InstallerTests(unittest.TestCase):
    MAIN_FIXTURE = '''def main(dataset_config, cfg):
    if dataset_config["name"] in ["fb_hierarchies", "shetland", "cora", "sbm", "planar"]:
        if dataset_config["name"] == "planar":
            planar_implementation = True
        else:
            old_implementation = True
        if cfg.dataset.name == "planar":
            dataset_scoped_option = True
'''

    def test_main_patch_adds_dd_but_keeps_planar_and_existing_routes(self):
        changed = INSTALL.change_main(self.MAIN_FIXTURE)
        ast.parse(changed)
        self.assertIn('"sbm", "planar", "dd"]:', changed)
        self.assertIn('if dataset_config["name"] == "dd":', changed)
        self.assertIn('elif dataset_config["name"] == "planar":', changed)
        self.assertIn("from datasets.dd_beta_adapter import DDBetaDatasetInfos", changed)
        self.assertIn("planar_implementation = True", changed)
        self.assertIn("old_implementation = True", changed)
        self.assertIn('cfg.dataset.name in ("planar", "dd")', changed)

    def test_main_patch_rejects_missing_anchor(self):
        for anchor in (
            '["fb_hierarchies", "shetland", "cora", "sbm", "planar"]:',
            '        if dataset_config["name"] == "planar":',
            '        if cfg.dataset.name == "planar":',
        ):
            with self.subTest(anchor=anchor), self.assertRaises(AssertionError):
                INSTALL.change_main(self.MAIN_FIXTURE.replace(anchor, "wrong_anchor", 1))

    def test_main_patch_rejects_duplicate_anchor(self):
        with self.assertRaises(AssertionError):
            INSTALL.change_main(self.MAIN_FIXTURE + self.MAIN_FIXTURE)

    def test_valid_frozen_manifest_schema_passes_without_changes(self):
        obj = synthetic_manifest()
        before = copy.deepcopy(obj)
        INSTALL.validate_dd_manifest(obj)
        self.assertEqual(obj, before)

    def test_manifest_rejects_missing_graph_split_id(self):
        obj = synthetic_manifest()
        obj["splits"]["test"].pop()
        with self.assertRaises(AssertionError):
            INSTALL.validate_dd_manifest(obj)

    def test_manifest_rejects_repeated_id_even_with_correct_split_lengths(self):
        obj = synthetic_manifest()
        obj["splits"]["test"][-1] = obj["splits"]["train"][0]
        with self.assertRaises(AssertionError):
            INSTALL.validate_dd_manifest(obj)

    def test_manifest_rejects_wrong_source_or_partition_or_settings(self):
        mutations = (
            lambda x: x.update(source_sha256="0" * 64),
            lambda x: x["settings"].update(split_seed=0),
            lambda x: x["partitions"]["1"].append([0]),
            lambda x: x["partitions"].pop("1"),
        )
        for mutate in mutations:
            obj = synthetic_manifest()
            mutate(obj)
            with self.subTest(mutation=mutate), self.assertRaises(AssertionError):
                INSTALL.validate_dd_manifest(obj)


class StaticRunnerTests(unittest.TestCase):
    def test_every_payload_python_file_parses_without_importing(self):
        paths = sorted((HERE / "payload").rglob("*.py"))
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_smoke_and_formal_use_latest_models_and_original_frozen_split(self):
        expected = {
            "h2": ("2", "beta_h2", "val/beta_h2_loss"),
            "h1": ("1", "beta_h1_twohop_mean_hog_rw", "val/h1_total"),
            "glue": ("1.5", "original_glue", "val/epoch_NLL"),
        }
        for kind in ("smoke", "formal"):
            _, tree = parse_driver(kind)
            strings = string_constants(tree)
            with self.subTest(kind=kind):
                self.assertEqual(assigned_literal(tree, "spec"), expected)
                for option in ("dataset=dd", "dataset.resolution=2", "dataset.split_seed=1234",
                               "dataset.louvain_seed=0", "train.batch_size=4", "train.seed=0",
                               "general.resume=null", "++general.run_test_after_fit=false"):
                    self.assertIn(option, strings)
                self.assertNotIn("dataset=planar", strings)
                self.assertIn("rw_depth", strings)
                self.assertIn("mean", strings)
                self.assertIn("Downloading/reprocessing/repartitioning is forbidden", strings)

    def test_formal_is_full_500_epoch_training_not_smoke_or_warm_start(self):
        _, tree = parse_driver("formal")
        strings = string_constants(tree)
        for option in ("train.n_epochs=500", "++general.limit_train_batches=1.0",
                       "++general.limit_val_batches=1.0", "train.lr=0.0002"):
            self.assertIn(option, strings)
        self.assertNotIn("++general.limit_train_batches=2", strings)
        self.assertNotIn("train.n_epochs=1", strings)

    def test_sampling_has_184_formal_count_seed1234_and_correct_interface(self):
        _, tree = parse_driver("sample")
        strings = string_constants(tree)
        self.assertIn("dd_structure_only", strings)
        self.assertIn("assemble H1; no retries or dropped samples", strings)
        self.assertIn("mean", strings)
        self.assertIn("dd_hierarchy_r2_s1234_l0.json", {s.rsplit("/", 1)[-1] for s in strings})
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        seeds = [ast.literal_eval(node.args[0]) for node in calls
                 if isinstance(node.func, ast.Attribute) and node.func.attr == "seed_sampling"]
        self.assertEqual(seeds, [1234])
        counts = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
                  and isinstance(node.left, ast.Name) and node.left.id == "count"]
        self.assertTrue(any(any(isinstance(op, ast.In) for op in node.ops)
                            and ast.literal_eval(node.comparators[0]) == (1, 184) for node in counts))
        config = next(value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                      for key, value in zip(node.keys, node.values)
                      if isinstance(key, ast.Constant) and key.value == "general")
        values = {ast.literal_eval(k): v for k, v in zip(config.keys, config.values)}
        self.assertEqual(ast.literal_eval(values["sampling_seed"]), 1234)
        self.assertEqual(ast.literal_eval(values["h1_batch_size"]), 4)
        self.assertEqual(ast.literal_eval(values["glue_batch_size"]), 1)
        interface = {ast.literal_eval(k): ast.literal_eval(v) for k, v in
                     zip(values["interface"].keys, values["interface"].values)
                     if not isinstance(v, ast.Call)}
        self.assertEqual(interface["glue_present_class_index"], 1)
        self.assertEqual(interface["h2_to_h1_condition_mode"], "dd_structure_only")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        # Stay in the authorized package workspace, not a Windows app-container
        # temp directory whose child-directory permissions may be restricted.
        self.temp = tempfile.TemporaryDirectory(prefix=".test_dd_deployment_", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "outputs/synthetic_run"
        self.out.mkdir(parents=True)
        self.source = self.root / "synthetic_source.txt"
        self.source.write_text("unchanged synthetic source", encoding="utf-8")
        self.audit = {"unchanged": True, "sha256": {str(self.source): LAUNCH.sha(self.source)}}
        (self.out / "input_hash_check.json").write_text(json.dumps(self.audit), encoding="utf-8")

    def make_reports(self, mode="smoke"):
        for stage in ("h2", "h1", "glue"):
            folder = self.out / stage
            folder.mkdir(exist_ok=True)
            best, last = folder / "best.ckpt", folder / "last.ckpt"
            best.write_bytes(b"synthetic bytes: never torch-loaded")
            last.write_bytes(b"synthetic last: never torch-loaded")
            report = {"strict_reload": True, "strict_best_reload": True,
                      "epoch": 0 if mode == "smoke" else 499,
                      "best": str(best), "last": str(last),
                      "best_sha256": LAUNCH.sha(best), "last_sha256": LAUNCH.sha(last)}
            (folder / (mode + "_result.json")).write_text(json.dumps(report), encoding="utf-8")

    def test_completed_accepts_three_finished_stage_reports(self):
        self.make_reports()
        self.assertEqual(LAUNCH.completed(self.root, self.out, "smoke"), self.out)

    def test_completed_rejects_unfinished_stage(self):
        self.make_reports()
        (self.out / "glue/smoke_result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            LAUNCH.completed(self.root, self.out, "smoke")

    def test_completed_rejects_changed_source_hash(self):
        self.make_reports()
        self.source.write_text("changed", encoding="utf-8")
        with self.assertRaises(AssertionError):
            LAUNCH.completed(self.root, self.out, "smoke")

    def test_completed_rejects_changed_best_checkpoint(self):
        self.make_reports()
        (self.out / "h1/best.ckpt").write_bytes(b"tampered")
        with self.assertRaises(AssertionError):
            LAUNCH.completed(self.root, self.out, "smoke")

    def test_completed_rejects_wrong_epoch_or_reload_status(self):
        for key, value in (("epoch", 8), ("strict_reload", False), ("strict_best_reload", False)):
            self.make_reports()
            path = self.out / "h1/smoke_result.json"
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj[key] = value
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.subTest(key=key), self.assertRaises(AssertionError):
                LAUNCH.completed(self.root, self.out, "smoke")

    def test_memory_check_queries_only_free_memory_and_deduplicates_gpus(self):
        with patch.object(LAUNCH.subprocess, "check_output", return_value="9000\n") as query, \
             contextlib.redirect_stdout(io.StringIO()):
            LAUNCH.check_memory([1, 0, 1])
        self.assertEqual(query.call_count, 2)
        for call in query.call_args_list:
            args = call.args[0]
            self.assertIn("--query-gpu=memory.free", args)
            self.assertNotIn("utilization.gpu", " ".join(args))
            self.assertNotIn("compute-apps", " ".join(args))

    def test_low_memory_fails_without_launching_job(self):
        with patch.object(LAUNCH.subprocess, "check_output", return_value="8191\n"), \
             patch.object(LAUNCH.subprocess, "Popen") as popen, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            LAUNCH.check_memory([0])
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
