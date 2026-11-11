"""Small unit tests; no real held-out set is evaluated by this file."""
import contextlib
import ast
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock


FILE = Path(__file__).with_name('evaluate.py')
spec = importlib.util.spec_from_file_location('tu_benchmark_eval_tested', FILE)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class TinyGraph:
    def __init__(self, nodes, edges=0):
        self.nodes, self.edges = nodes, edges

    def __len__(self):
        return self.nodes

    def number_of_edges(self):
        return self.edges


def backend():
    return types.SimpleNamespace(
        node_count_mmd=Mock(return_value=0.1),
        spectre_utils=types.SimpleNamespace(degree_stats=Mock(return_value=0.2),
            clustering_stats=Mock(return_value=0.3), spectral_stats=Mock(return_value=0.5)),
        mmd_calculators=types.SimpleNamespace(radius_stats=Mock(return_value=0.4)))


class MixedDispatch(unittest.TestCase):
    def test_backend_calls_match_original_tu_evaluator_source(self):
        original = Path('C:/path/to/HiGGs-main1/HiGGs-main/deliverables/small_benchmarks_20261101/files/higgs/evaluate_small_benchmarks.py')
        if not original.is_file():
            self.skipTest('Original evaluator fixture unavailable')
        def calls(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mixed_metrics')
            return [ast.dump(n.body, include_attributes=False) for n in ast.walk(function)
                    if isinstance(n, ast.Lambda)]
        self.assertEqual(calls(FILE), calls(original))

    def test_all_five_original_calls_and_no_spectral_truncation(self):
        real, generated = [TinyGraph(1), TinyGraph(3, 2)], [TinyGraph(1), TinyGraph(4, 3)]
        metrics = backend()
        with contextlib.redirect_stdout(io.StringIO()):
            rows = evaluation.mixed_metrics(real, generated, metrics)
        self.assertEqual([r['Metric'] for r in rows], list(evaluation.METRICS))
        self.assertEqual([r['Kernel'] for r in rows], list(evaluation.KERNELS.values()))
        metrics.node_count_mmd.assert_called_once_with(real, generated, 'tv')
        metrics.spectre_utils.degree_stats.assert_called_once_with(
            real, generated, is_parallel=True, compute_emd=True)
        metrics.spectre_utils.clustering_stats.assert_called_once_with(
            real, generated, bins=100, is_parallel=True, compute_emd=True)
        metrics.mmd_calculators.radius_stats.assert_called_once_with(
            real, generated, compute_emd=True, is_parallel=True, PRINT_TIME=False)
        metrics.spectre_utils.spectral_stats.assert_called_once_with(
            real, generated, is_parallel=True, n_eigvals=-1, compute_emd=True)
        # The same complete lists, including singleton/edgeless samples, reach
        # every backend. No graph-list sorting/selection/truncation is inserted.
        self.assertIs(metrics.spectre_utils.spectral_stats.call_args.args[0], real)
        self.assertIs(metrics.spectre_utils.spectral_stats.call_args.args[1], generated)

    def test_count_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            evaluation.mixed_metrics([TinyGraph(2)], [TinyGraph(2), TinyGraph(3)], backend())

    def test_empty_sample_is_error_not_filtered(self):
        with self.assertRaises(ValueError):
            evaluation.mixed_metrics([TinyGraph(2)], [TinyGraph(0)], backend())

    def test_nonfinite_result_is_refused(self):
        metrics = backend()
        metrics.node_count_mmd.return_value = float('nan')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            evaluation.mixed_metrics([TinyGraph(2)], [TinyGraph(2)], metrics)


class InputAndTableTests(unittest.TestCase):
    def test_canonical_contiguous_sample_indices(self):
        with tempfile.TemporaryDirectory(dir=FILE.parent) as folder:
            root = Path(folder)
            for name in ('sampled_graph_1.pkl', 'sampled_graph_0.pkl'):
                (root / name).touch()
            self.assertEqual([p.name for p in evaluation.strict_samples(root, 2)],
                             ['sampled_graph_0.pkl', 'sampled_graph_1.pkl'])
            (root / 'sampled_graph_3.pkl').touch()
            with self.assertRaises(ValueError):
                evaluation.strict_samples(root, 2)

    def test_noncanonical_or_malformed_files_are_not_ignored(self):
        for name in ('sampled_graph_00.pkl', 'sampled_graph_extra.pkl'):
            with tempfile.TemporaryDirectory(dir=FILE.parent) as folder:
                root = Path(folder)
                (root / 'sampled_graph_0.pkl').touch()
                (root / name).touch()
                with self.assertRaises(ValueError):
                    evaluation.strict_samples(root, 1)

    def test_edgeless_graph_retained_without_louvain(self):
        graph = TinyGraph(1)
        graphs = types.SimpleNamespace(statistics=Mock(return_value={
            'nodes': 1, 'diameter': 0, 'average_clustering': 0., 'density': 0., 'transitivity': 0.}))
        tables = types.SimpleNamespace(graph_row=Mock(side_effect=AssertionError('No Louvain here')))
        row = evaluation.graph_row(graph, 0, 'Generated', graphs, tables)
        self.assertEqual(row['Nodes'], 1)
        self.assertEqual(row['NC'], 1)
        self.assertEqual(row['MedianCommunitySize'], 1.)
        self.assertEqual(row['Clustering'], 0.)
        tables.graph_row.assert_not_called()

    def test_louvain_settings_are_fixed(self):
        graph = TinyGraph(3, 2)
        tables = types.SimpleNamespace(graph_row=Mock(return_value={'NC': 1}))
        evaluation.graph_row(graph, 5, 'Real', types.SimpleNamespace(), tables)
        tables.graph_row.assert_called_once_with(graph, 5, 'Real', 1, 0)

    def test_descriptor_mapping_is_not_all_emd_or_all_tv(self):
        settings = evaluation.descriptor_settings()
        self.assertEqual(evaluation.KERNELS['Nodes'], 'Gaussian-TV')
        self.assertEqual(evaluation.KERNELS['Eccentricity'], 'Gaussian')
        self.assertEqual(settings['Clustering']['bins'], 100)
        self.assertEqual(settings['Clustering']['sigma'], .1)
        self.assertEqual(settings['Spectre']['n_eigvals'], -1)
        self.assertFalse(settings['Spectre']['community_truncation'])
        self.assertEqual(settings['Eccentricity']['bins'], 50)


if __name__ == '__main__':
    unittest.main()

