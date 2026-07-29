import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "summarize_interface_ablation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_interface_ablation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
TEST_TEMP_ROOT = Path(__file__).resolve().parent


def dump_graph(path: Path, graph: nx.Graph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)


class InterfaceAblationSummaryTests(unittest.TestCase):
    def test_five_node_complete_graph_is_complete(self):
        stats = summary.graph_stats(nx.complete_graph(5))
        self.assertTrue(stats["complete_applicable"])
        self.assertTrue(stats["is_complete"])

    def test_noncomplete_graph_is_not_complete(self):
        stats = summary.graph_stats(nx.path_graph(5))
        self.assertTrue(stats["complete_applicable"])
        self.assertFalse(stats["is_complete"])

    def test_full_and_used_difference_triggers_lcc(self):
        full_graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
        used_graph = full_graph.subgraph([0, 1, 2]).copy()
        comparison = summary.compare_graphs(full_graph, used_graph)
        self.assertTrue(comparison["lcc_triggered"])
        self.assertEqual(comparison["removed_nodes"], 2)
        self.assertEqual(comparison["removed_edges"], 1)

    def test_h1_communities_pair_by_identical_filename(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            sample_dir = Path(temporary_dir) / "sample_000"
            for community_id in (1, 0):
                filename = f"community_{community_id:03d}.pkl"
                dump_graph(sample_dir / "h1_full" / filename, nx.path_graph(3))
                dump_graph(sample_dir / "h1_used" / filename, nx.path_graph(2))
            pairs = summary.pair_h1_files(sample_dir)
            self.assertEqual([item[0] for item in pairs], [0, 1])
            for _, full_path, used_path in pairs:
                self.assertEqual(full_path.name, used_path.name)

    def test_missing_required_file_raises_clear_error(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            missing = Path(temporary_dir) / "h2_full.pkl"
            with self.assertRaisesRegex(
                FileNotFoundError, "Required graph file is missing"
            ):
                summary.load_graph(missing)

    def test_unpaired_h1_filename_raises_clear_error(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            sample_dir = Path(temporary_dir) / "sample_000"
            dump_graph(
                sample_dir / "h1_full" / "community_000.pkl", nx.path_graph(3)
            )
            dump_graph(
                sample_dir / "h1_used" / "community_001.pkl", nx.path_graph(2)
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "not paired by identical filename"
            ):
                summary.pair_h1_files(sample_dir)

    def test_real_h2_preserves_isolated_community_node(self):
        adjacency = [
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        communities = [{10, 11}, {12}, {13}]
        graph = summary.build_h2_meta_graph(
            adjacency, communities, global_node_offset=10
        )
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(list(nx.isolates(graph)), [2])

    def test_multiple_original_cross_edges_make_one_h2_edge(self):
        adjacency = [
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ]
        graph = summary.build_h2_meta_graph(
            adjacency, [{0, 1}, {2, 3}], global_node_offset=0
        )
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(set(graph.edges()), {(0, 1)})

    def test_intra_community_edges_do_not_make_h2_self_loop(self):
        adjacency = [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
        graph = summary.build_h2_meta_graph(
            adjacency, [{20, 21}, {22}], global_node_offset=20
        )
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(nx.number_of_selfloops(graph), 0)

    def test_three_nodes_three_edges_is_complete(self):
        is_complete, applicable = summary.complete_status(3, 3)
        self.assertTrue(applicable)
        self.assertTrue(is_complete)

    def test_four_nodes_three_edges_is_not_complete(self):
        is_complete, applicable = summary.complete_status(4, 3)
        self.assertTrue(applicable)
        self.assertFalse(is_complete)


if __name__ == "__main__":
    unittest.main()
