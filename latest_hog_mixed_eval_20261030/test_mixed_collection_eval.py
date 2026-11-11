"""CPU-only tests; all graph writes are isolated temporary fixtures."""

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mixed_collection_eval", HERE / "mixed_collection_eval.py")
mixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixed)
ROOT = Path(os.environ.get("HIGGS_MIXED_TEST_ROOT", str(HERE.parents[1])))


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend, cls.graphs, cls.tables = mixed.load_backend(ROOT)
        import networkx as nx
        import numpy as np
        cls.nx, cls.np = nx, np

    def test_explicit_kernel_routing_and_spectral_all(self):
        backend = SimpleNamespace(node_count_mmd=Mock(return_value=0.1),
            spectre_utils=SimpleNamespace(degree_stats=Mock(return_value=0.2),
                clustering_stats=Mock(return_value=0.3), spectral_stats=Mock(return_value=0.4)),
            mmd_calculators=SimpleNamespace(radius_stats=Mock(return_value=0.5)))
        with contextlib.redirect_stdout(io.StringIO()):
            values = mixed.mixed_metrics(["real"], ["generated"], backend)
        self.assertEqual(list(values), list(mixed.METRICS))
        backend.node_count_mmd.assert_called_once_with(["real"], ["generated"], "tv")
        for fn in (backend.spectre_utils.degree_stats, backend.spectre_utils.clustering_stats,
                   backend.spectre_utils.spectral_stats, backend.mmd_calculators.radius_stats):
            self.assertIs(fn.call_args.kwargs["compute_emd"], True)
        self.assertEqual(backend.spectre_utils.spectral_stats.call_args.kwargs["n_eigvals"], -1)
        self.assertEqual(backend.spectre_utils.clustering_stats.call_args.kwargs["bins"], 100)

    def test_nonfinite_metrics_are_not_hidden(self):
        with patch.object(self.backend, "node_count_mmd", return_value=float("nan")):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                mixed.mixed_metrics([], [], self.backend)

    def test_real_kernels_match_manually_built_histograms(self):
        nx, np, b = self.nx, self.np, self.backend
        real, gen = [nx.path_graph(5), nx.cycle_graph(6)], [nx.complete_graph(5), nx.star_graph(4)]
        with contextlib.redirect_stdout(io.StringIO()):
            actual = mixed.mixed_metrics(real, gen, b)
        def histograms(graphs, kind):
            result = []
            for graph in graphs:
                if kind == "Degree":
                    value = np.array(nx.degree_histogram(graph))
                elif kind == "Clustering":
                    value = np.histogram(list(nx.clustering(graph).values()), bins=100, range=(0, 1))[0]
                elif kind == "Eccentricity":
                    paths = [distance for _, row in nx.all_pairs_shortest_path_length(graph) for distance in row.values()]
                    value = np.histogram(paths, bins=50)[0]
                else:
                    eigs = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(graph).toarray())
                    value = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)[0]
                    value = value / value.sum()
                result.append(value)
            return result
        expected = {"Nodes": b.dist_helper.compute_mmd([np.bincount([5, 6])], [np.bincount([5, 5])], kernel=b.dist_helper.gaussian_tv)}
        for metric in ("Degree", "Clustering", "Spectre"):
            kwargs = {"sigma": 0.1, "distance_scaling": 100} if metric == "Clustering" else {}
            expected[metric] = b.dist_helper.compute_mmd(histograms(real, metric), histograms(gen, metric),
                                                       kernel=b.dist_helper.gaussian_emd, **kwargs)
        expected["Eccentricity"] = b.mmd_calculators.compute_mmd(
            histograms(real, "Eccentricity"), histograms(gen, "Eccentricity"), kernel=b.mmd_calculators.gaussian)
        for metric in mixed.METRICS:
            self.assertAlmostEqual(actual[metric], expected[metric], places=11, msg=metric)
        with contextlib.redirect_stdout(io.StringIO()):
            identity = mixed.mixed_metrics(real, real, b)
        self.assertTrue(all(abs(v) < 1e-10 for v in identity.values()))

    def create_fixture(self, base, dataset="tree", variant="original"):
        nx, b = self.nx, self.backend
        owner = base / variant
        directory = owner / "outputs" / f"{dataset}_mean_hog_tv"
        samples = owner / "outputs" / f"{dataset}_mean_hog_samples" / "sampling"
        directory.mkdir(parents=True)
        samples.mkdir(parents=True)
        real = [nx.path_graph(5), nx.cycle_graph(6)]
        generated = [nx.disjoint_union(nx.path_graph(5), nx.empty_graph(1)), nx.complete_graph(5)]
        hashes = {}
        for i, graph in enumerate(generated):
            path = samples / f"sampled_graph_{i}.pkl"
            with path.open("wb") as stream:
                pickle.dump(graph, stream)
            hashes[str(path.resolve())] = mixed.sha(path)
        normalized = [self.graphs.normalize_graph(g, "lcc") for g in generated]
        rows = [self.tables.graph_row(g, i + 2, "Real", 1.0, 0) for i, g in enumerate(real)]
        rows += [self.tables.graph_row(g, i, "Generated", 1.0, 0) for i, g in enumerate(normalized)]
        manifest = {"dataset": dataset, "graphs": [{"id": i, "n": len(g), "edges": list(g.edges())}
            for i, g in enumerate([nx.path_graph(3), nx.path_graph(4), *real])],
            "splits": {"train": [0], "val": [1], "test": [2, 3]}}
        manifest_path = owner / "data_sources/splits" / f"frozen_{dataset}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        mixed.write_json(manifest_path, manifest)
        digest = mixed.sha(manifest_path)
        protocol = {"dataset": dataset, "manifest_sha256": digest, "generated_count": 2, "test_graph_count": 2,
                    "table1_mode": "tv", "component_policy": "lcc", "training_partitions_modified": False,
                    "table3_evaluation_louvain": {"resolution": 1, "seed": 0, "weight": None}, "samples_sha256": hashes}
        report = {"dataset": dataset, "manifest_sha256": digest, "generated_count": 2,
                  "exploratory_count_mismatch": False, "component_policy": "lcc", "samples_sha256": hashes,
                  "results": {"tv": {"Nodes": b.node_count_mmd(real, normalized, "tv")}}}
        mixed.write_json(directory / "evaluation.json", report)
        mixed.write_json(directory / "table_protocol.json", protocol)
        mixed.write_csv(directory / "table3_summary.csv", self.tables.summary_rows(rows))
        return owner, directory / "evaluation.json", manifest_path, digest

    def test_previous_completed_samples_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, report, manifest, digest = self.create_fixture(Path(temporary))
            with patch.dict(mixed.SPECS, {"tree": (2, "frozen.json", digest, report.parent.name)}):
                job = mixed.job_from_tv("tree", "original", owner, report)
                path = next(iter(job["samples_sha256"]))
                with Path(path).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaises(ValueError):
                    mixed.job_from_tv("tree", "original", owner, report)

    def test_full_graph_policy_cannot_be_silently_switched(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, path, _, digest = self.create_fixture(Path(temporary))
            report = mixed.read_json(path)
            report["component_policy"] = "full"
            mixed.write_json(path, report, exclusive=False)
            with patch.dict(mixed.SPECS, {"tree": (2, "frozen.json", digest, path.parent.name)}), self.assertRaises(ValueError):
                mixed.job_from_tv("tree", "original", owner, path)

    def test_samples_outside_owner_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sampled_graph_0.pkl"
            sample.write_bytes(b"not unpickled in this check")
            with self.assertRaises(ValueError):
                mixed.checked_samples({str(sample): mixed.sha(sample)}, root / "other", 1)

    def test_improved_discovery_uses_completed_evaluation_not_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, path, _, digest = self.create_fixture(Path(temporary), variant="improved")
            with patch.dict(mixed.SPECS, {"tree": (2, "frozen.json", digest, path.parent.name)}):
                job = mixed.discover_improved("tree", owner)
            self.assertEqual(job["prior_tv_report"], str(path.resolve()))

    def test_pending_dd_is_not_evaluated_as_partial_or_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(mixed.discover_improved("dd", Path(temporary)))
            probe = Path(temporary) / "outputs/dd_mean_hog_probe"
            probe.mkdir(parents=True)
            mixed.write_json(probe / "input_manifest.json", {"manifest_sha256": mixed.SPECS["dd"][2], "requested_samples": 1, "real_test_graphs": 184})
            mixed.write_json(probe / "input_hash_check.json", {"unchanged": True})
            mixed.write_json(probe / "sampling_progress.json", [{"index": 0}])
            self.assertIsNone(mixed.discover_improved("dd", Path(temporary)))

    def test_worker_recalculates_without_mutating_inputs_or_table3(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            owner, report, manifest, digest = self.create_fixture(base)
            with patch.dict(mixed.SPECS, {"tree": (2, "frozen.json", digest, report.parent.name)}):
                job = mixed.job_from_tv("tree", "original", owner, report)
            job.update(manifest=str(manifest), manifest_sha256=digest, count=2, test_graph_ids=[2, 3])
            plan = {"original_root": str(owner), "sources_sha256": {}, "tool_sha256": {str(mixed.SELF): mixed.sha(mixed.SELF)},
                    "evaluation_louvain": {"resolution": 1, "seed": 0, "weight": None}, "jobs": [job]}
            before = {str(p): mixed.sha(p) for p in owner.rglob("*") if p.is_file()}
            with patch.object(mixed, "load_backend", return_value=(self.backend, self.graphs, self.tables)):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = mixed.evaluate_job(plan, 0, owner / "outputs/new_mixed_result")
            self.assertEqual(before, {p: mixed.sha(p) for p in before})
            self.assertEqual(result["kernel_mapping"], mixed.KERNELS)
            self.assertTrue(result["inputs_unchanged"])
            old = mixed.read_json(report)
            self.assertAlmostEqual(result["mmd"]["Nodes"], old["results"]["tv"]["Nodes"], places=12)
            with (owner / "outputs/new_mixed_result/table3_summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["GeneratedMean"], "5.0")

    def test_missing_completed_non_dd_is_a_real_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                mixed.discover_improved("grid", Path(temporary))

    def test_modified_output_root_keeps_same_kernels_inputs_and_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            owner, report, manifest, digest = self.create_fixture(base, variant="improved")
            with patch.dict(mixed.SPECS, {"tree": (2, "frozen.json", digest, report.parent.name)}):
                job = mixed.job_from_tv("tree", "improved", owner, report)
            job.update(manifest=str(manifest), manifest_sha256=digest, count=2, test_graph_ids=[2, 3])
            destination = owner / "higgs/outputs"
            plan = {"original_root": str(base / "original"), "improved_root": str(owner),
                    "results_root": str(destination), "sources_sha256": {},
                    "tool_sha256": {str(mixed.SELF): mixed.sha(mixed.SELF)},
                    "evaluation_louvain": {"resolution": 1, "seed": 0, "weight": None}, "jobs": [job]}
            before = {str(p): mixed.sha(p) for p in owner.rglob("*") if p.is_file()}
            with patch.object(mixed, "load_backend", return_value=(self.backend, self.graphs, self.tables)):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = mixed.evaluate_job(plan, 0, destination / "new_mixed_result")
            self.assertEqual(result["variant"], "improved")
            self.assertEqual(result["kernel_mapping"], mixed.KERNELS)
            self.assertEqual(before, {p: mixed.sha(p) for p in before})
            self.assertTrue((destination / "new_mixed_result/table1_mixed.csv").is_file())
            self.assertFalse((base / "original").exists())
            self.assertAlmostEqual(result["mmd"]["Nodes"], mixed.read_json(report)["results"]["tv"]["Nodes"], places=12)

    def test_unauthorized_output_root_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, report, manifest, digest = self.create_fixture(Path(temporary))
            with patch.dict(mixed.SPECS, {"tree": (2, "frozen.json", digest, report.parent.name)}):
                job = mixed.job_from_tv("tree", "original", owner, report)
            job.update(manifest=str(manifest), manifest_sha256=digest, count=2, test_graph_ids=[2, 3])
            outside = Path(temporary) / "unrelated"
            plan = {"original_root": str(owner), "results_root": str(outside), "sources_sha256": {},
                    "tool_sha256": {str(mixed.SELF): mixed.sha(mixed.SELF)},
                    "evaluation_louvain": {"resolution": 1, "seed": 0, "weight": None}, "jobs": [job]}
            with patch.object(mixed, "load_backend", return_value=(self.backend, self.graphs, self.tables)):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    mixed.evaluate_job(plan, 0, outside / "not_allowed")
            self.assertFalse(outside.exists())

    def test_real_subprocess_worker_uses_modified_results_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            owner, report, manifest, digest = self.create_fixture(base, variant="improved")
            with patch.dict(mixed.SPECS, {"tree": (2, "frozen.json", digest, report.parent.name)}):
                job = mixed.job_from_tv("tree", "improved", owner, report)
            job.update(manifest=str(manifest), manifest_sha256=digest, count=2, test_graph_ids=[2, 3])
            plan = {"original_root": str(ROOT), "improved_root": str(owner),
                    "results_root": str(owner / "higgs/outputs"),
                    "sources_sha256": {str(ROOT / rel): mixed.sha(ROOT / rel) for rel in mixed.SOURCE_FILES},
                    "tool_sha256": {str(mixed.SELF): mixed.sha(mixed.SELF)},
                    "evaluation_louvain": {"resolution": 1, "seed": 0, "weight": None}, "jobs": [job]}
            plan_path = base / "plan.json"
            mixed.write_json(plan_path, plan)
            output = owner / "higgs/outputs/mixed_collection_tables_subprocess/tree_improved"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", MPLBACKEND="Agg",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            child = subprocess.run([sys.executable, "-u", "-B", str(mixed.SELF), "worker",
                "--plan", str(plan_path), "--index", "0", "--output", str(output)],
                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(child.returncode, 0, child.stdout[-6000:])
            result = mixed.read_json(output / "mixed_evaluation.json")
            self.assertEqual(result["kernel_mapping"], mixed.KERNELS)
            self.assertTrue(result["inputs_unchanged"])
            mixed.verify_hashes(result["input_sha256"])


if __name__ == "__main__":
    unittest.main()
