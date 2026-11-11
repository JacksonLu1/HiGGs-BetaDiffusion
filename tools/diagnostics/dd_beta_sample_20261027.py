import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

root, train, out = (Path(v).resolve() for v in sys.argv[1:4])
manifest = root / "data_sources/splits/dd_hierarchy_r2_s1234_l0.json"
digest = json.loads((root / "data_sources/splits/dd_beta_setup_20261027.json").read_text())["manifest_sha256"]
count = int(sys.argv[4])
assert count in (1, 184), "Only a one-graph exploratory probe or the formal 184 graphs"

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert root == Path("/path/to/HiGGs-GBD/HiGGs-main")
assert out.parent == root / "higgs/outputs" and not out.exists()
assert sha(manifest) == digest

frozen = json.loads(manifest.read_text())
assert tuple(
    len(frozen["splits"][s]) for s in ("train", "val", "test")
) == (587, 147, 184)

lock_path = root / "logs/sampling/dd_mean_hog_fullchain.lock"
lock = lock_path.open("a+")
fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

audit = json.loads((train / "input_hash_check.json").read_text())
assert audit["unchanged"] is True
before = dict(audit["sha256"])

for path, expected in before.items():
    assert sha(path) == expected, f"Training input/source changed: {path}"

reports = {}
for key in ("h2", "h1", "glue"):
    report_path = train / key / "formal_result.json"
    report = json.loads(report_path.read_text())
    assert report["epoch"] == 499
    assert report["strict_reload"] and report["strict_best_reload"]
    assert report["manifest_sha256"] == digest
    assert sha(report["best"]) == report["best_sha256"]
    reports[key] = report
    before[str(report_path)] = sha(report_path)
    before[str(Path(report["best"]).resolve())] = report["best_sha256"]

os.chdir(root)
sys.path[:0] = [str(root / "dgd"), str(root / "higgs"), str(root)]

import torch
import networkx as nx
from omegaconf import OmegaConf
import main as training_entry
import sample_main as sampling_entry
import sampling
import higgs_utils
import interface_ablation as interface

assert torch.cuda.is_available(), "CUDA unavailable; CPU sampling forbidden"
torch.cuda.set_device(0)
assert torch.cuda.mem_get_info()[0] >= 8 * 1024**3, "Less than 8 GiB free"

print("Physical GPU:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("Logical GPU: cuda:0", torch.cuda.get_device_name(0), flush=True)

stage_keys = {2.0: "h2", 1.0: "h1", 1.5: "glue"}
checkpoint_cfgs = {}

for number, key in stage_keys.items():
    state = higgs_utils._load_checkpoint_cpu(reports[key]["best"])
    cfg = state["hyper_parameters"]["cfg"]
    cfg = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)

    assert cfg.dataset.name == "dd"
    assert float(cfg.dataset.h) == number
    assert float(cfg.dataset.resolution) == 2
    assert Path(cfg.dataset.manifest).resolve() == manifest

    family = {
        "h2": "beta_h2",
        "h1": "beta_h1",
        "glue": "original_glue",
    }[key]

    higgs_utils._require_checkpoint_identity(
        reports[key]["best"], family, number, key.upper(),
        allowed_families={family, "digress"} if key == "glue" else None,
    )

    if key == "h1":
        assert state["h1_two_hop"]["normalization"] == "mean"
        assert state["h1_two_hop"]["warm_start"] is None
        assert state["h1_hog_rw"]["rw_depth"] == 4
        assert state["h1_hog_rw"]["two_hop"] is True

    checkpoint_cfgs[key] = copy.deepcopy(cfg)
    del state

for rel in ("higgs/sample_main.py", "higgs/interface_ablation.py"):
    before[str(root / rel)] = sha(root / rel)

glue_cache = root / "data/dd/processed" / (
    f"proc_train_v1_h1.5_max500_{digest}.pt"
)
cache_audit = interface.audit_glue_training_cache(str(glue_cache))
assert cache_audit["verified_present_class_index"] == 1

h1_cache = root / "data/dd/processed" / (
    f"proc_train_v1_h1_max500_{digest}.pt"
)
data, slices = higgs_utils._load_checkpoint_cpu(h1_cache)
assert data.y.ndim == 2 and data.y.shape[1] == 2
assert torch.all(data.y[:, 0] == 1)
assert torch.all(data.y[:, 1] == 0)
del data, slices

# DD结构数据的H1条件固定为训练中的class0。
# 只在当前采样进程内适配，不修改任何模型源文件。
interface._CONDITION_MODES.add("dd_structure_only")

def dd_h1_conditions(node_types, num_classes, mode, *, dataset_name):
    assert dataset_name == "dd"
    assert mode == "dd_structure_only"
    assert node_types.ndim == 1 and node_types.dtype == torch.long
    assert num_classes == 2
    result = torch.zeros(
        (len(node_types), num_classes),
        dtype=torch.long,
        device=node_types.device,
    )
    result[:, 0] = 1
    return result

sampling.adapt_h2_to_h1_conditions = dd_h1_conditions

# 每个阶段使用自己的checkpoint配置，避免H2配置覆盖H1的HOG设置。
def dd_get_samplers(request):
    key = stage_keys[float(request.dataset.h)]
    model_cfg = copy.deepcopy(checkpoint_cfgs[key])

    model = training_entry.get_model_class(model_cfg).load_from_checkpoint(
        str(reports[key]["best"]),
        map_location="cpu",
        cfg=model_cfg,
        strict=True,
    )

    assert model.dataset_info.manifest_sha256 == digest
    assert list(model.dataset_info.training_graph_ids) == frozen["splits"]["train"]
    assert type(model.node_dist).__module__ == "datasets.dd_beta_adapter"
    assert model.cfg.dataset.name == "dd"

    if key == "h1":
        widths = {
            k: getattr(model.model, "mlp_in_" + k)[0].in_features
            for k in ("X", "E", "y")
        }
        assert widths == {"X": 12, "E": 6, "y": 14}
        assert model.cfg.model.two_hop.normalization == "mean"
        assert model.cfg.model.hog_rw.enabled

    model.eval()
    model.freeze()

    runtime_cfg = copy.deepcopy(model.cfg)
    OmegaConf.set_struct(runtime_cfg, False)
    runtime_cfg.general = copy.deepcopy(request.general)
    runtime_cfg.general.test_only = reports[key]["best"]
    return runtime_cfg, model

sampling.get_samplers = dd_get_samplers

# H2若没有社区间边，直接装配H1，不丢弃、不重新抽样。
original_prepare = sampling_entry.prepare_glue_dataset

def dd_prepare(
    cfg, model, adjs, nodes, edge_index, edge_types,
    batch_size, return_audit=False,
):
    assert cfg.dataset.name == "dd"

    if edge_index.numel():
        assert torch.all(edge_types == 1), "Nonbinary H2 edge category"
        return original_prepare(
            cfg, model, adjs, nodes, edge_index, edge_types,
            batch_size, return_audit=return_audit,
        )

    print(
        "H2 has no meta-edge: assembling H1 without Glue; no resampling.",
        flush=True,
    )
    route = {
        "condition_mode": "direct",
        "raw_h2_edge_type_histogram": {},
        "glue_condition_histogram_after_adapter": {},
        "verified_glue_training_y_histogram": cache_audit["one_hot_histogram"],
        "glue_training_cache": cache_audit,
        "pair_ids": [],
    }
    result = ([], [], batch_size, 0, 0, [])
    return result + (route,) if return_audit else result

sampling_entry.prepare_glue_dataset = dd_prepare

cfg = OmegaConf.create({
    "dataset": OmegaConf.to_container(
        checkpoint_cfgs["h2"].dataset, resolve=True),
    "model": OmegaConf.to_container(
        checkpoint_cfgs["h2"].model, resolve=True),
    "train": OmegaConf.to_container(
        checkpoint_cfgs["h2"].train, resolve=True),
    "general": {
        "name": "dd",
        "gpus": 1,
        "wandb": "disabled",
        "root_path": str(root / "data"),
        "h2_model": reports["h2"]["best"],
        "h2_model_family": "beta_h2",
        "h1_model": reports["h1"]["best"],
        "h1_model_family": "beta_h1",
        "glue_model": reports["glue"]["best"],
        "glue_model_family": "original_glue",
        "samples_to_generate": count,
        "sampling_seed": 1234,
        "h1_batch_size": 4,
        "glue_batch_size": 1,
        "visualize": False,
        "intermediate_figures": False,
        "interface": {
            "h2_component_policy": "upstream_lcc",
            "h1_component_policy": "upstream_lcc",
            "h2_to_h1_condition_mode": "dd_structure_only",
            "h2_to_glue_condition_mode": "direct",
            "glue_present_class_index": 1,
            "glue_training_cache": str(glue_cache),
            "force_eval": True,
            "save_intermediates": True,
            "strict_assertions": True,
        },
    },
})

interface.interface_options(cfg)

out.mkdir()
(out / ".hydra").mkdir()
OmegaConf.save(cfg, out / ".hydra/config.yaml")

(out / "input_manifest.json").write_text(json.dumps({
    "training_run": str(train),
    "manifest_sha256": digest,
    "checkpoints": reports,
    "source_sha256": before,
    "protocol": "seed1234 once before sampling; strict reload per stage per graph",
    "requested_samples": count, "real_test_graphs": 184, "exploratory": count != 184,
    "dd_h1_condition": "fixed training class0; original H2 labels audited",
    "empty_glue": "assemble H1; no retries or dropped samples",
}, indent=2))

sample_dir = out / "sampling"
sample_dir.mkdir()
os.chdir(sample_dir)
interface.seed_sampling(1234)

rows = []
for index in range(count):
    print(f"\nSTART DD GRAPH {index + 1}/{count}", flush=True)

    graph, lcc = sampling_entry.sample_large_graph(
        cfg, "dd",
        h1_batch_size=4,
        glue_batch_size=1,
        graph_number=index,
        visualize=False,
        sampling_metrics=False,
        save_tensors=False,
        save_extras=False,
        intermediate_figures=False,
    )

    assert isinstance(graph, nx.Graph) and not graph.is_directed()
    assert not graph.is_multigraph() and len(graph) > 0
    assert nx.number_of_selfloops(graph) == 0

    path = sample_dir / f"sampled_graph_{index}.pkl"
    assert path.is_file()

    rows.append({
        "index": index,
        "nodes": len(graph),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "lcc_nodes": len(lcc),
        "lcc_edges": lcc.number_of_edges(),
        "sha256": sha(path),
    })

    (out / "sampling_progress.json").write_text(
        json.dumps(rows, indent=2)
    )
    print("SAVED:", rows[-1], flush=True)

assert len(list(sample_dir.glob("sampled_graph_*.pkl"))) == count
assert before == {p: sha(p) for p in before}, "Inputs/source changed"

with (out / "sampling_structure.csv").open("x", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

(out / "input_hash_check.json").write_text(json.dumps({
    "unchanged": True,
    "sha256": before,
}, indent=2))

print(f"PASS: DD_MEAN_HOG_FULLCHAIN_{count}_COMPLETE", flush=True)
print("OUTPUT:", out, flush=True)
print("Sampling complete; generation quality has not yet been evaluated.", flush=True)
