import random
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import wandb


def preserve_rng_state(func):
    """Keep graph layout code from advancing sampling RNG streams."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        try:
            return func(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return wrapped


def _visualization_path(stage, graph_number, filename):
    output_dir = (Path.cwd() / "visualizations" /
                  f"sample_{graph_number:03d}" / stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


@preserve_rng_state
def vis_h1_group(h1_adjs, ax=None, graph_number=0, batch_id=0):
    G = nx.Graph()

    for adj in h1_adjs:
        graph = nx.from_numpy_array(adj.cpu().numpy())
        graph = nx.convert_node_labels_to_integers(
            graph, first_label=G.order() + 1)
        G = nx.compose(G, graph)

    pos = nx.drawing.nx_agraph.graphviz_layout(
        G, prog="sfdp", args="-Gsmoothing")

    if ax is not None:
        nx.draw_networkx_edges(G, node_size=2, pos=pos, alpha=0.5, ax=ax)
        return ax, G

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw_networkx_edges(G, node_size=2, pos=pos, alpha=0.5, ax=ax)
        ax.set_title(r"Sampled $h_1$ Graphs")
        fig.tight_layout()

        output_path = _visualization_path(
            "h1", graph_number,
            f"h1_sample_{graph_number}_batch_{batch_id}.png")
        fig.savefig(output_path, dpi=300)
        print(f"Saved H1 visualization [sample={graph_number}, batch={batch_id}]: "
              f"{output_path.resolve()}")
        wandb.log({f"Sampled H1/sample_{graph_number:03d}/batch_{batch_id:03d}": [
            wandb.Image(str(output_path), caption="Un-Coloured-SFDP, h1")]})
        return G
    finally:
        if fig is not None:
            plt.close(fig)


@preserve_rng_state
def vis_glue_group(glue_adjs, ax=None, graph_number=0, batch_id=0):
    G = nx.Graph()

    for graph in glue_adjs:
        components = [graph.subgraph(c) for c in nx.connected_components(graph)]
        components = sorted(
            components, key=lambda item: item.number_of_nodes(), reverse=True)
        graph = components[0]
        graph = nx.convert_node_labels_to_integers(
            graph, first_label=G.order() + 1)
        G = nx.compose(G, graph)

    print(f"Finding big glue positions with {G.order()} nodes")
    pos = nx.drawing.nx_agraph.graphviz_layout(
        G, prog="sfdp", args="-Gsmoothing")
    G.remove_edges_from(nx.selfloop_edges(G))

    if ax is not None:
        nx.draw_networkx_edges(G, node_size=2, pos=pos, alpha=0.5, ax=ax)
        return ax

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw_networkx_edges(G, node_size=2, pos=pos, alpha=0.5, ax=ax)
        ax.set_title(r"Sampled Edge Predictions Graphs")
        fig.tight_layout()

        output_path = _visualization_path(
            "glue", graph_number,
            f"glue_sample_{graph_number}_batch_{batch_id}.png")
        fig.savefig(output_path, dpi=300)
        print(f"Saved Glue visualization [sample={graph_number}, batch={batch_id}]: "
              f"{output_path.resolve()}")
        wandb.log({f"Sampled Glue/sample_{graph_number:03d}/batch_{batch_id:03d}": [
            wandb.Image(str(output_path), caption="Un-Coloured-SFDP, glue")]})
        return G
    finally:
        if fig is not None:
            plt.close(fig)


@preserve_rng_state
def vis_big_graphs(original_G, G, cg, graph_number=0):
    pos_real = nx.drawing.nx_agraph.graphviz_layout(
        original_G, prog="sfdp", args="-Gsmoothing")
    pos_sample = nx.drawing.nx_agraph.graphviz_layout(
        G, prog="sfdp", args="-Gsmoothing")
    pos_cg = nx.drawing.nx_agraph.graphviz_layout(
        cg, prog="sfdp", args="-Gsmoothing")

    fig = None
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 6))

        nx.draw_networkx_edges(
            original_G, node_size=2, pos=pos_real, alpha=0.5, ax=ax1)
        nx.draw_networkx_nodes(
            original_G, node_size=1, pos=pos_real, ax=ax1,
            node_color=[node[1]["target"] for node in original_G.nodes(data=True)])

        nx.draw_networkx_edges(
            G, node_size=2, pos=pos_sample, ax=ax2, alpha=0.5)
        nx.draw_networkx_nodes(
            G, node_size=2, pos=pos_sample, ax=ax2,
            node_color=[node[1]["target"] for node in G.nodes(data=True)])

        nx.draw_networkx_edges(cg, node_size=2, pos=pos_cg, alpha=0.5, ax=ax3)
        nx.draw_networkx_nodes(
            cg, node_size=2, pos=pos_cg, ax=ax3,
            node_color=[node[1]["target"] for node in cg.nodes(data=True)])

        ax1.set_title("Original")
        ax2.set_title(r"Synthetic, all CCs")
        ax3.set_title("Synthetic, Largest CC")
        fig.tight_layout()

        output_path = _visualization_path(
            "final", graph_number,
            f"sample_vs_real_graphviz_{graph_number}.png")
        fig.savefig(output_path, dpi=300)
        print(f"Saved final visualization [sample={graph_number}]: "
              f"{output_path.resolve()}")
        wandb.log({f"Final Graph/sample_{graph_number:03d}": [
            wandb.Image(
                str(output_path),
                caption=f"Un-Coloured-SFDP_graph_{graph_number}")]})
    finally:
        if fig is not None:
            plt.close(fig)
