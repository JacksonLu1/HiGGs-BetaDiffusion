"""Sample frozen DD checkpoints after an audited H1 early stop; never train.

Linux server entry point. The 20261027 sampler remains unchanged. This separate
entry point reports H1's actual saved epoch instead of claiming formal epoch 499.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path("/path/to/HiGGs-GBD/HiGGs-main")
SELF = Path(__file__).resolve()
RUN = "dd_beta_mean_hog_r2_e500_scratch_seed0_20261027_205633_003751"
SMOKE = "dd_beta_mean_hog_smoke_seed0_20261027_203843_088995_gpu0"
MANIFEST_REL = "data_sources/splits/dd_hierarchy_r2_s1234_l0.json"
MANIFEST_SHA = "3162886166b33c6fd7832996664b0c6a422824c4f7c82a6d44efc4cc9e7745a2"
DRIVER_REL = "tools/diagnostics/dd_beta_formal_20261027.py"
DRIVER_SHA = "3efdf7bdd3ee4ee024a937e48b3fa956a17c5132b854596251bff7e027d1ea7c"
PROTECTED = {
    "dgd/diffusion_model_beta_h2.py": "a0b895b21afe82d8335c6b77acafcaf18f1adc9fc734c0d83149a5b3babfabe6",
    "dgd/diffusion_model_beta_h1.py": "b4d6078601afda3fb46ea40c0b9687d949898605dd29f5949d9eb65239d426a3",
    "dgd/diffusion_model_discrete.py": "fd477d76e959fca43070b0a9a36e3aa5a0916a933cf8aecbd7a74ddac4643f70",
    "dgd/diffusion/distributions.py": "ce79c6f1addca9d72b04dde0a694ea8872690963b892c9c8c8251688d2584e00",
    "higgs/higgs_utils.py": "6934a9d05d08818119ee97e64b846a9d6b59a1afc800b0dfd3966c53a049f648",
    "higgs/sampling.py": "4f770fd6df760b9a071b7fc2402e37dae1312bb94eb0b649b54a31fc6e0bc345",
}
STAGES = {2.0: "h2", 1.0: "h1", 1.5: "glue"}
MONITORS = {"h2": "val/beta_h2_loss", "h1": "val/h1_total", "glue": "val/epoch_NLL"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_hashes(hashes):
    for path, expected in hashes.items():
        assert isinstance(expected, str) and len(expected) == 64, path
        assert sha(path) == expected, f"Frozen input/source changed: {path}"


def validate_snapshot(root, snapshot, out, count, allow_prepared=False):
    """Fail closed on budget/provenance before importing model code or torch."""
    root, snapshot, out = (Path(p).resolve() for p in (root, snapshot, out))
    assert root == ROOT, "Only the explicitly authorized modified HiGGs root"
    assert count in (1, 184), "Only the one-graph probe or the 184-graph run"
    assert snapshot.parent == root / "outputs", "Snapshot must be a direct output child"
    assert snapshot.name.startswith("dd_earlystop_snapshot_")
    assert out.parent == root / "higgs/outputs" and not out.exists(), "Output must be new"
    assert not out.is_symlink() and not snapshot.is_symlink()

    selection_path = snapshot / "selection.json"
    audit_path = snapshot / "stop_audit.json"
    selection, audit = read_json(selection_path), read_json(audit_path)
    assert Path(selection["root"]).resolve() == root
    assert Path(selection["snapshot"]).resolve() == snapshot
    assert Path(selection["run"]).resolve() == root / "outputs" / RUN
    assert selection["manifest_sha256"] == MANIFEST_SHA
    allowed_statuses = ("h1_stopped", "already_completed")
    if allow_prepared:
        allowed_statuses += ("snapshot_ready_stop_intent",)
    assert audit["status"] in allowed_statuses
    assert audit["snapshot_verified"] is True
    assert audit["selection_sha256"] == sha(selection_path)

    before = dict(selection["input_sha256"])
    assert before, "Snapshot has no immutable input map"
    assert all(Path(path).is_absolute() and str(Path(path).resolve()) == path for path in before)
    verify_hashes(before)
    manifest = root / MANIFEST_REL
    assert before[str(manifest)] == MANIFEST_SHA and sha(manifest) == MANIFEST_SHA
    frozen = read_json(manifest)
    assert frozen["dataset"] == "dd"
    assert tuple(len(frozen["splits"][s]) for s in ("train", "val", "test")) == (587, 147, 184)
    for rel, expected in {**PROTECTED, DRIVER_REL: DRIVER_SHA}.items():
        assert before[str(root / rel)] == expected, f"Unproven source: {rel}"
    for rel in ("higgs/sample_main.py", "higgs/interface_ablation.py"):
        assert str(root / rel) in before, f"Sampling source not frozen: {rel}"

    baseline_path = root / "outputs" / SMOKE / "input_hash_check.json"
    assert str(baseline_path) in before
    baseline = read_json(baseline_path)
    assert baseline["unchanged"] is True
    for path, expected in baseline["sha256"].items():
        assert before[path] == expected, f"Snapshot departed from smoke baseline: {path}"
    for h in ("2", "1", "1.5"):
        for split in ("train", "val", "test"):
            cache = root / "data/dd/processed" / f"proc_{split}_v1_h{h}_max500_{MANIFEST_SHA}.pt"
            assert str(cache) in before and str(cache) in baseline["sha256"]

    checkpoints = selection["checkpoints"]
    assert set(checkpoints) == {"h2", "h1", "glue"}
    for key, record in checkpoints.items():
        best = Path(record["best"])
        last = Path(record["saved_last"])
        assert best.is_absolute() and last.is_absolute()
        assert snapshot in best.resolve().parents and snapshot in last.resolve().parents
        assert before[str(best)] == record["best_sha256"]
        assert before[str(last)] == record["last_sha256"]
        assert isinstance(record["epoch"], int) and record["epoch"] >= 0
        assert isinstance(record["global_step"], int) and record["global_step"] >= 0
        assert isinstance(record["training_last_epoch"], int)
        assert record["epoch"] <= record["training_last_epoch"] <= 499
        assert record["monitor"] == MONITORS[key]
        assert math.isfinite(float(record["best_score"]))
        assert record["strict_reload"] is True
        if key != "h1" or audit["status"] == "already_completed":
            assert record["training_last_epoch"] == 499
            report_path = Path(selection["run"]) / key / "formal_result.json"
            assert str(report_path) in before, f"Missing completed-stage proof: {key}"
            report = read_json(report_path)
            assert report["epoch"] == 499
            assert report["strict_reload"] and report["strict_best_reload"]
            assert report["manifest_sha256"] == MANIFEST_SHA
            assert report["best_sha256"] == record["best_sha256"]
            assert report["last_sha256"] == record["last_sha256"]
            assert report["best_epoch"] == record["epoch"]
        else:
            assert record["training_last_epoch"] < 499, "Do not mislabel completed H1 as early stopped"

    # These are additional sampler inputs, not retroactive edits to selection.
    for path in (selection_path, audit_path, SELF):
        before[str(path)] = sha(path)
    return selection, audit, frozen, before


def sampling_lock(root, inherited_fd=None):
    """Keep the coordinator's sampling flock alive across the whole subprocess."""
    import fcntl

    lock_path = Path(root) / "logs/sampling/dd_mean_hog_fullchain.lock"
    if inherited_fd is not None:
        actual, expected = os.fstat(inherited_fd), lock_path.stat()
        assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), "Wrong inherited lock FD"
        # dup retains the same open-file description and therefore its flock.
        # Never open/acquire a second file description when inheriting the lock.
        return os.fdopen(os.dup(inherited_fd), "a+")
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def restore_binary_pos(datalist, torch_module):
    """Restore only the original Glue mask's Boolean representation, in memory."""
    converted = 0
    for data in datalist:
        pos = data.pos
        assert torch_module.is_tensor(pos), "Glue pos is not a tensor"
        if pos.dtype != torch_module.bool:
            assert torch_module.isfinite(pos).all().item(), "Nonfinite Glue pos mask"
            assert ((pos == 0) | (pos == 1)).all().item(), "Nonbinary Glue pos mask"
            data.pos = pos.bool()
            converted += 1
    return converted


def run_sampling(root, snapshot, out, count, inherited_fd=None):
    # Acquire/reuse the lock before any metadata or checkpoint is consumed.
    with sampling_lock(root, inherited_fd):
        selection, audit, frozen, before = validate_snapshot(root, snapshot, out, count)
        return _sample_locked(root, snapshot, out, count, selection, audit, frozen, before)


def _sample_locked(root, snapshot, out, count, selection, audit, frozen, before):
    root, snapshot, out = (Path(p).resolve() for p in (root, snapshot, out))
    manifest = root / MANIFEST_REL
    reports = selection["checkpoints"]

    # Required before both torch checkpoint loads and sample_main's cwd-sensitive
    # import. sample_main changes cwd to higgs; all subsequent paths are absolute.
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
    print("Sampler SHA256:", before[str(SELF)], flush=True)
    print("Snapshot selection SHA256:", before[str(snapshot / "selection.json")], flush=True)
    print("Stop audit SHA256:", before[str(snapshot / "stop_audit.json")], flush=True)

    checkpoint_cfgs = {}
    for number, key in STAGES.items():
        record = reports[key]
        state = higgs_utils._load_checkpoint_cpu(record["best"])
        assert int(state["epoch"]) == record["epoch"]
        assert int(state["global_step"]) == record["global_step"]
        cfg = state["hyper_parameters"]["cfg"]
        cfg = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
        assert cfg.dataset.name == "dd" and float(cfg.dataset.h) == number
        assert float(cfg.dataset.resolution) == 2
        assert Path(cfg.dataset.manifest).resolve() == manifest
        family = {"h2": "beta_h2", "h1": "beta_h1", "glue": "original_glue"}[key]
        higgs_utils._require_checkpoint_identity(
            record["best"], family, number, key.upper(),
            allowed_families={family, "digress"} if key == "glue" else None,
        )
        if key == "h1":
            assert state["h1_two_hop"]["normalization"] == "mean"
            assert state["h1_two_hop"]["warm_start"] is None
            assert state["h1_hog_rw"]["rw_depth"] == 4
            assert state["h1_hog_rw"]["two_hop"] is True
        checkpoint_cfgs[key] = copy.deepcopy(cfg)
        del state
        last_state = higgs_utils._load_checkpoint_cpu(record["saved_last"])
        assert int(last_state["epoch"]) == record["training_last_epoch"]
        del last_state

    glue_cache = root / "data/dd/processed" / f"proc_train_v1_h1.5_max500_{MANIFEST_SHA}.pt"
    cache_audit = interface.audit_glue_training_cache(str(glue_cache))
    assert cache_audit["verified_present_class_index"] == 1
    h1_cache = root / "data/dd/processed" / f"proc_train_v1_h1_max500_{MANIFEST_SHA}.pt"
    data, slices = higgs_utils._load_checkpoint_cpu(h1_cache)
    assert data.y.ndim == 2 and data.y.shape[1] == 2
    assert torch.all(data.y[:, 0] == 1) and torch.all(data.y[:, 1] == 0)
    del data, slices

    # Same DD structure-only condition as the original DD full-chain sampler.
    interface._CONDITION_MODES.add("dd_structure_only")

    def dd_h1_conditions(node_types, num_classes, mode, *, dataset_name):
        assert dataset_name == "dd" and mode == "dd_structure_only"
        assert node_types.ndim == 1 and node_types.dtype == torch.long
        assert num_classes == 2
        result = torch.zeros((len(node_types), num_classes), dtype=torch.long, device=node_types.device)
        result[:, 0] = 1
        return result

    sampling.adapt_h2_to_h1_conditions = dd_h1_conditions

    def dd_get_samplers(request):
        key = STAGES[float(request.dataset.h)]
        model_cfg = copy.deepcopy(checkpoint_cfgs[key])
        model = training_entry.get_model_class(model_cfg).load_from_checkpoint(
            str(reports[key]["best"]), map_location="cpu", cfg=model_cfg, strict=True,
        )
        assert model.dataset_info.manifest_sha256 == MANIFEST_SHA
        assert list(model.dataset_info.training_graph_ids) == frozen["splits"]["train"]
        assert type(model.node_dist).__module__ == "datasets.dd_beta_adapter"
        assert model.cfg.dataset.name == "dd" and float(model.cfg.dataset.h) == float(request.dataset.h)
        if key == "h1":
            widths = {k: getattr(model.model, "mlp_in_" + k)[0].in_features for k in ("X", "E", "y")}
            assert widths == {"X": 12, "E": 6, "y": 14}
            assert model.cfg.model.two_hop.normalization == "mean"
            assert model.cfg.model.hog_rw.enabled and model.cfg.model.hog_rw.rw_depth == 4
        model.eval()
        model.freeze()
        runtime_cfg = copy.deepcopy(model.cfg)
        OmegaConf.set_struct(runtime_cfg, False)
        runtime_cfg.general = copy.deepcopy(request.general)
        runtime_cfg.general.test_only = reports[key]["best"]
        return runtime_cfg, model

    sampling.get_samplers = dd_get_samplers
    original_prepare = sampling_entry.prepare_glue_dataset
    mask_audit = {"converted_pair_masks": 0, "prepare_calls": 0}

    def dd_prepare(cfg, model, adjs, nodes, edge_index, edge_types, batch_size, return_audit=False):
        assert cfg.dataset.name == "dd"
        if edge_index.numel():
            assert torch.all(edge_types == 1), "Nonbinary H2 edge category"
            result = original_prepare(cfg, model, adjs, nodes, edge_index, edge_types, batch_size, return_audit=return_audit)
            datalist, loader = result[:2]
            assert loader.dataset is datalist, "Cannot safely restore original Glue mask representation"
            converted = restore_binary_pos(datalist, torch)
            mask_audit["prepare_calls"] += 1
            mask_audit["converted_pair_masks"] += converted
            if return_audit:
                result[-1]["runtime_boolean_pos_restored"] = converted
            return result
        print("H2 has no meta-edge: assembling H1 without Glue; no resampling.", flush=True)
        route = {
            "condition_mode": "direct", "raw_h2_edge_type_histogram": {},
            "glue_condition_histogram_after_adapter": {},
            "verified_glue_training_y_histogram": cache_audit["one_hot_histogram"],
            "glue_training_cache": cache_audit, "pair_ids": [],
        }
        result = ([], [], batch_size, 0, 0, [])
        return result + (route,) if return_audit else result

    sampling_entry.prepare_glue_dataset = dd_prepare
    cfg = OmegaConf.create({
        "dataset": OmegaConf.to_container(checkpoint_cfgs["h2"].dataset, resolve=True),
        "model": OmegaConf.to_container(checkpoint_cfgs["h2"].model, resolve=True),
        "train": OmegaConf.to_container(checkpoint_cfgs["h2"].train, resolve=True),
        "general": {
            "name": "dd", "gpus": 1, "wandb": "disabled", "root_path": str(root / "data"),
            "h2_model": reports["h2"]["best"], "h2_model_family": "beta_h2",
            "h1_model": reports["h1"]["best"], "h1_model_family": "beta_h1",
            "glue_model": reports["glue"]["best"], "glue_model_family": "original_glue",
            "samples_to_generate": count, "sampling_seed": 1234,
            "h1_batch_size": 4, "glue_batch_size": 1,
            "visualize": False, "intermediate_figures": False,
            "interface": {
                "h2_component_policy": "upstream_lcc", "h1_component_policy": "upstream_lcc",
                "h2_to_h1_condition_mode": "dd_structure_only", "h2_to_glue_condition_mode": "direct",
                "glue_present_class_index": 1, "glue_training_cache": str(glue_cache),
                "force_eval": True, "save_intermediates": True, "strict_assertions": True,
            },
        },
    })
    interface.interface_options(cfg)

    # Strictly load every stage before creating output or consuming sampling RNG.
    # Sampling subsequently repeats strict per-stage reload per graph, unchanged.
    for number in STAGES:
        request = copy.deepcopy(cfg)
        OmegaConf.set_struct(request, False)
        request.dataset.h = number
        _, model = dd_get_samplers(request)
        del model

    verify_hashes(before)
    out.mkdir()
    (out / ".hydra").mkdir()
    OmegaConf.save(cfg, out / ".hydra/config.yaml")
    input_manifest = {
        "training_run": selection["run"], "snapshot": str(snapshot),
        "manifest_sha256": MANIFEST_SHA, "checkpoints": reports,
        "source_sha256": before,
        "sampler_sha256": before[str(SELF)],
        "selection_sha256": before[str(snapshot / "selection.json")],
        "stop_audit_sha256": before[str(snapshot / "stop_audit.json")],
        "training_budget": {
            "h1_status": audit["status"], "original_target_epochs": 500,
            "h1_saved_last_epoch": reports["h1"]["training_last_epoch"],
            "h1_selected_best_epoch": reports["h1"]["epoch"],
            "h1_early_stopped": audit["status"] == "h1_stopped",
            "h2_saved_last_epoch": reports["h2"]["training_last_epoch"],
            "glue_saved_last_epoch": reports["glue"]["training_last_epoch"],
            "epochs_are_zero_based": True,
        },
        "budget_note": "H1 budget is saved checkpoint progress, not the live log epoch or selected-best epoch",
        "equal_training_budget_comparison": False,
        "comparison_note": selection.get("comparison_note", "H1 was stopped early; do not claim an equal-budget ablation"),
        "protocol": "seed1234 once before sampling; strict reload per stage per graph",
        "requested_samples": count, "real_test_graphs": 184, "exploratory": count != 184,
        "sampling_seed": 1234, "h1_batch_size": 4, "glue_batch_size": 1,
        "h2_component_policy": "upstream_lcc", "h1_component_policy": "upstream_lcc",
        "dd_h1_condition": "fixed training class0; original H2 labels audited",
        "empty_glue": "assemble H1; no retries or dropped samples",
        "glue_prepare": "original function; finite binary pos restored to bool only when needed, runtime only",
        "generation_quality_evaluated": False,
    }
    (out / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    sample_dir = out / "sampling"
    sample_dir.mkdir()
    os.chdir(sample_dir)
    interface.seed_sampling(1234)
    rows = []
    for index in range(count):
        print(f"\nSTART DD GRAPH {index + 1}/{count}", flush=True)
        graph, lcc = sampling_entry.sample_large_graph(
            cfg, "dd", h1_batch_size=4, glue_batch_size=1, graph_number=index,
            visualize=False, sampling_metrics=False, save_tensors=False,
            save_extras=False, intermediate_figures=False,
        )
        assert isinstance(graph, nx.Graph) and not graph.is_directed()
        assert not graph.is_multigraph() and len(graph) > 0
        assert nx.number_of_selfloops(graph) == 0
        path = sample_dir / f"sampled_graph_{index}.pkl"
        assert path.is_file()
        rows.append({
            "index": index, "nodes": len(graph), "edges": graph.number_of_edges(),
            "components": nx.number_connected_components(graph), "lcc_nodes": len(lcc),
            "lcc_edges": lcc.number_of_edges(), "sha256": sha(path),
        })
        (out / "sampling_progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (out / "glue_mask_runtime_audit.json").write_text(json.dumps(mask_audit, indent=2), encoding="utf-8")
        print("SAVED:", rows[-1], flush=True)

    assert len(list(sample_dir.glob("sampled_graph_*.pkl"))) == count
    verify_hashes(before)
    with (out / "sampling_structure.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "input_hash_check.json").write_text(json.dumps({"unchanged": True, "sha256": before}, indent=2), encoding="utf-8")
    print(f"PASS: DD_EARLYSTOP_MEAN_HOG_FULLCHAIN_{count}_COMPLETE", flush=True)
    print("OUTPUT:", out, flush=True)
    print("Sampling complete; generation quality has not yet been evaluated.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=184, choices=(1, 184))
    parser.add_argument("--lock-fd", type=int, help="Existing coordinator flock FD (passed via pass_fds)")
    args = parser.parse_args(argv)
    assert args.root.resolve() == ROOT
    run_sampling(args.root.resolve(), args.snapshot.resolve(), args.output.resolve(), args.count, args.lock_fd)


if __name__ == "__main__":
    main()
