"""Read-only mixed-kernel re-evaluation of existing original/improved samples."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys

sys.dont_write_bytecode = True
SELF = Path(__file__).resolve()
PROTOCOL = "higgs_frozen_mixed_v1"
ORIGINAL = Path("/path/to/HiGGs-main1/HiGGs-main")
IMPROVED = Path("/path/to/HiGGs-GBD/HiGGs-main")
METRICS = ("Nodes", "Degree", "Clustering", "Eccentricity", "Spectre")
KERNELS = dict(zip(METRICS, ("Gaussian-TV", "Gaussian-EMD", "Gaussian-EMD", "Gaussian", "Gaussian-EMD")))
SPECS = {
    "planar": (40, "planar_hierarchy_r2_s1234_l0.json", "05b42aa49d076985096e1c0e3408a452a433a4173953c60b27b438d8b4971bcc",
               "planar_table1_table3_20261024_134609_317105069"),
    "dd": (184, "dd_hierarchy_r2_s1234_l0.json", "3162886166b33c6fd7832996664b0c6a422824c4f7c82a6d44efc4cc9e7745a2",
           "dd_table1_table3_higgs_author_184_20261028_174737_562850840"),
    "community_small": (20, "community_small_higgs_r1_s1234_l0.json", "782aa9f9c04eefb09a175784700c27a6163fc01efc41cd4298bf91ef03191839",
                        "community_small_table1_table3_higgs_author_20_20261028_195952_199466900"),
    "enzymes": (117, "enzymes_higgs_r1_s1234_l0.json", "bc885cf5538c9ad53f713336d8a50c5ea3f40194dee3a86c72ba393d08891749",
                "enzymes_table1_table3_higgs_author_117_20261028_210629_865985877"),
    "tree": (40, "tree_higgs_r1_s1234_l0.json", "59c3e8b9e7eed327098018da3caf691c780c1a1c932900c224d59ebb448a7e37",
             "tree_table1_table3_higgs_author_40_20261029_102002_371358053"),
    "grid": (20, "grid_higgs_r2_s1234_l0.json", "da59020eeea53566a3f60e480c39e481fc797ba2314161a0e6e691603de38338",
             "grid_table1_table3_higgs_author_20_20261029_174219_818856115"),
}
SOURCE_FILES = (
    "higgs/evaluate_sbm_mmd.py", "higgs/evaluate_graph_collection.py", "higgs/evaluate_collection_tables.py",
    "dgd/analysis/dist_helper.py", "dgd/analysis/spectre_utils.py", "dgd/metrics/mmd/calculators.py",
    "dgd/metrics/mmd/kernels.py", "dgd/utils.py", "dgd/datasets/collection_sources.py",
)
SKIP = {"wandb", "checkpoints", "graphs", "chains", "sampling", "frozen_best", "node_modules", "__pycache__"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def verify_hashes(expected):
    for name, digest in expected.items():
        if sha(name) != digest:
            raise ValueError(f"Input changed; evaluation stopped: {name}")


def find_files(root, filename):
    for base in (root / "outputs", root / "higgs/outputs"):
        if not base.is_dir():
            continue
        for directory, folders, files in os.walk(base):
            folders[:] = [n for n in folders if n not in SKIP and not n.startswith("mixed_collection_tables_")]
            if filename in files:
                yield Path(directory) / filename


def checked_samples(hashes, owner, count):
    pattern = re.compile(r"sampled_graph_(\d+)\.pkl$")
    numbered, parents = {}, set()
    for name, digest in hashes.items():
        path = Path(name).resolve()
        match = pattern.fullmatch(path.name)
        if not match or not inside(path, owner):
            raise ValueError(f"Not a sample inside its experiment root: {path}")
        index = int(match.group(1))
        if index in numbered or not path.is_file() or len(digest) != 64:
            raise ValueError(f"Duplicate/missing sample: {path}")
        numbered[index] = (path, digest)
        parents.add(path.parent)
    if sorted(numbered) != list(range(count)) or len(parents) != 1:
        raise ValueError(f"Expected exactly {count} contiguous samples in one directory")
    parent = next(iter(parents))
    if {p.name for p in parent.glob("sampled_graph_*.pkl")} != {f"sampled_graph_{i}.pkl" for i in range(count)}:
        raise ValueError(f"Extra sample files in {parent}")
    expected = {str(numbered[i][0]): numbered[i][1] for i in range(count)}
    verify_hashes(expected)
    return expected


def job_from_tv(dataset, variant, owner, report_path):
    count, _, digest, _ = SPECS[dataset]
    report = read_json(report_path)
    protocol_path = report_path.parent / "table_protocol.json"
    protocol = read_json(protocol_path)
    if (report.get("dataset") != dataset or report.get("manifest_sha256") != digest
            or report.get("generated_count") != count or report.get("exploratory_count_mismatch")
            or report.get("component_policy") != "lcc" or set(report.get("results", {})) != {"tv"}
            or protocol.get("table1_mode") != "tv" or protocol.get("training_partitions_modified") is not False
            or protocol.get("generated_count") != count or protocol.get("test_graph_count") != count
            or protocol.get("manifest_sha256") != digest or protocol.get("component_policy") != "lcc"
            or protocol.get("table3_evaluation_louvain") != {"resolution": 1.0, "seed": 0, "weight": None}):
        raise ValueError(f"Not the completed expected TV/LCC comparison: {report_path}")
    if not (report_path.parent / "table3_summary.csv").is_file():
        raise FileNotFoundError(f"Table 3 completion evidence missing: {report_path.parent}")
    hashes = checked_samples(report["samples_sha256"], owner, count)
    if protocol.get("samples_sha256") != report["samples_sha256"]:
        raise ValueError(f"Prior report sample hashes disagree: {report_path}")
    return {"dataset": dataset, "variant": variant, "samples_sha256": hashes,
            "prior_tv_report": str(report_path.resolve()), "prior_tv_report_sha256": sha(report_path),
            "prior_protocol": str(protocol_path.resolve()), "prior_protocol_sha256": sha(protocol_path),
            "prior_table3": str((report_path.parent / "table3_summary.csv").resolve()),
            "prior_table3_sha256": sha(report_path.parent / "table3_summary.csv"),
            "prior_nodes_mmd": float(report["results"]["tv"]["Nodes"])}


def discover_improved(dataset, root):
    count = SPECS[dataset][0]
    candidates = []
    for path in find_files(root, "evaluation.json"):
        report = read_json(path)
        if report.get("dataset") != dataset or report.get("generated_count") != count:
            continue
        # Only the existing mean-two-hop + HOG arm, never a different ablation.
        if not all(word in str(path).lower() for word in (dataset, "mean", "hog")):
            continue
        if (path.parent / "table_protocol.json").is_file():
            candidates.append(path)
    if candidates:
        newest = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
        return job_from_tv(dataset, "improved", root, newest)
    if dataset != "dd":
        raise FileNotFoundError(f"No completed {dataset} mean+HOG TV report under {root}")
    # DD may complete after packaging. Its sampler writes this final hash audit
    # only after all 184 graphs; partial/probe runs are never evaluated.
    completed = []
    for audit_path in find_files(root, "input_hash_check.json"):
        run = audit_path.parent
        if not all(word in str(run).lower() for word in ("dd", "mean", "hog")):
            continue
        manifest_path = run / "input_manifest.json"
        progress_path = run / "sampling_progress.json"
        if not manifest_path.is_file() or not progress_path.is_file():
            continue
        info = read_json(manifest_path)
        if (info.get("manifest_sha256") == SPECS["dd"][2]
                and info.get("requested_samples") == count
                and info.get("real_test_graphs") == count
                and read_json(audit_path).get("unchanged") is True):
            completed.append((audit_path, info, progress_path))
    if not completed:
        return None
    audit_path, info, progress_path = max(completed, key=lambda item: item[0].stat().st_mtime_ns)
    rows = read_json(progress_path)
    if sorted(row["index"] for row in rows) != list(range(count)):
        raise ValueError("DD completion audit does not contain exactly 184 sample IDs")
    hashes = {str(audit_path.parent / "sampling" / f"sampled_graph_{row['index']}.pkl"): row["sha256"] for row in rows}
    hashes = checked_samples(hashes, root, count)
    return {"dataset": "dd", "variant": "improved", "samples_sha256": hashes,
            "completion_evidence": {str(p): sha(p) for p in (audit_path, progress_path, audit_path.parent / "input_manifest.json")},
            "training_budget": info.get("training_budget"), "equal_training_budget_comparison": False,
            "budget_note": info.get("comparison_note", "Original Glue stopped near 250; improved training budget differs")}


def make_plan(original, improved, datasets, variants):
    original, improved = original.resolve(), improved.resolve()
    plan = {"protocol": PROTOCOL, "original_root": str(original), "improved_root": str(improved),
            "kernel_mapping": KERNELS, "component_policy": "lcc",
            "evaluation_louvain": {"resolution": 1, "seed": 0, "weight": None},
            "sources_sha256": {str(original / rel): sha(original / rel) for rel in SOURCE_FILES},
            "tool_sha256": {str(SELF): sha(SELF)}, "jobs": [], "pending": []}
    for dataset in datasets:
        count, filename, digest, previous = SPECS[dataset]
        path = original / "data_sources/splits" / filename
        if sha(path) != digest:
            raise ValueError(f"Frozen {dataset} manifest changed: {path}")
        manifest = read_json(path)
        ids = [r["id"] for r in manifest["graphs"]]
        split_ids = [i for key in ("train", "val", "test") for i in manifest["splits"][key]]
        if (manifest["dataset"] != dataset or len(manifest["splits"]["test"]) != count
                or len(ids) != len(set(ids)) or sorted(split_ids) != sorted(ids)
                or len(split_ids) != len(set(split_ids))):
            raise ValueError(f"Invalid frozen whole-graph splits: {path}")
        counterpart = improved / "data_sources/splits" / filename
        if counterpart.exists() and sha(counterpart) != digest:
            raise ValueError(f"Improved manifest differs from original: {counterpart}")
        for variant in variants:
            if variant == "original":
                job = job_from_tv(dataset, variant, original, original / "outputs" / previous / "evaluation.json")
            else:
                job = discover_improved(dataset, improved)
            if job is None:
                plan["pending"].append({"dataset": dataset, "variant": variant, "reason": "184 DD samples not yet confirmed complete; no partial evaluation"})
                continue
            job.update(manifest=str(path), manifest_sha256=digest, test_graph_ids=manifest["splits"]["test"], count=count)
            plan["jobs"].append(job)
    return plan


def load_backend(root):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path[:0] = [str(root / "higgs"), str(root / "dgd")]
    import evaluate_sbm_mmd as metrics
    import evaluate_graph_collection as graphs
    import evaluate_collection_tables as tables
    if Path(metrics.__file__).resolve() != (root / "higgs/evaluate_sbm_mmd.py").resolve():
        raise ValueError("Metric module came from the wrong checkout")
    return metrics, graphs, tables


def mixed_metrics(real, generated, backend):
    # Reuse exact repository functions and their default bandwidths. In the
    # radius helper compute_emd=True selects Gaussian, not Gaussian-EMD.
    calls = (
        ("Nodes", lambda: backend.node_count_mmd(real, generated, "tv")),
        ("Degree", lambda: backend.spectre_utils.degree_stats(real, generated, is_parallel=True, compute_emd=True)),
        ("Clustering", lambda: backend.spectre_utils.clustering_stats(real, generated, bins=100, is_parallel=True, compute_emd=True)),
        ("Eccentricity", lambda: backend.mmd_calculators.radius_stats(real, generated, compute_emd=True, is_parallel=True, PRINT_TIME=False)),
        ("Spectre", lambda: backend.spectre_utils.spectral_stats(real, generated, is_parallel=True, n_eigvals=-1, compute_emd=True)),
    )
    result = {}
    for name, function in calls:
        print(f"METRIC_START {name} KERNEL={KERNELS[name]}", flush=True)
        value = float(function())
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {name} MMD")
        result[name] = value
        print(f"METRIC_DONE {name} {value:.8f}", flush=True)
    return result


def job_hashes(plan, job):
    expected = {**plan["sources_sha256"], **plan["tool_sha256"], **job["samples_sha256"],
                job["manifest"]: job["manifest_sha256"], **job.get("completion_evidence", {})}
    for prefix in ("prior_tv_report", "prior_protocol", "prior_table3"):
        if prefix in job:
            expected[job[prefix]] = job[prefix + "_sha256"]
    return expected


def evaluate_job(plan, index, output):
    job = plan["jobs"][index]
    expected = job_hashes(plan, job)
    verify_hashes(expected)
    root = Path(plan["original_root"])
    backend, graphs, tables = load_backend(root)
    manifest = read_json(job["manifest"])
    records = {r["id"]: r for r in manifest["graphs"]}
    if manifest["splits"]["test"] != job["test_graph_ids"]:
        raise ValueError("Frozen test IDs changed")
    real = [graphs.normalize_graph(graphs.record_graph(records[i]), "lcc") for i in job["test_graph_ids"]]
    generated = []
    for name in job["samples_sha256"]:
        # Only the user's own generated graphs, already hash-matched to their
        # completed evaluation/sampling audit, are unpickled here.
        with Path(name).open("rb") as stream:
            generated.append(graphs.normalize_graph(pickle.load(stream), "lcc"))
    if len(real) != job["count"] or len(generated) != job["count"]:
        raise ValueError("Sample count changed")
    print(f"EVALUATE {job['dataset']} {job['variant']} {len(generated)}vs{len(real)} mixed LCC", flush=True)
    values = mixed_metrics(real, generated, backend)
    if "prior_nodes_mmd" in job and not math.isclose(values["Nodes"], job["prior_nodes_mmd"], rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Nodes uses the same TV kernel and inputs; it must match the previous result")
    rows = []
    for group, ids, collection in (("Real", job["test_graph_ids"], real), ("Generated", range(len(generated)), generated)):
        for n, (graph_id, graph) in enumerate(zip(ids, collection), 1):
            rows.append(tables.graph_row(graph, graph_id, group, 1.0, 0))
            if n % 10 == 0 or n == len(collection):
                print(f"TABLE3 {group} {n}/{len(collection)}", flush=True)
    summary = tables.summary_rows(rows)
    if "prior_table3" in job:
        with Path(job["prior_table3"]).open(encoding="utf-8", newline="") as stream:
            prior = {r["Metric"]: r for r in csv.DictReader(stream)}
        for row in summary:
            for field in ("RealMean", "GeneratedMean", "RealStd", "GeneratedStd"):
                if not math.isclose(row[field], float(prior[row["Metric"]][field]), rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"Table 3 changed although only MMD kernels should change: {row['Metric']}/{field}")
    verify_hashes(expected)
    # Standalone modified-arm launcher: the metric backend stays in the original
    # checkout; only new result placement is configurable. No kernel/input change.
    results_root = Path(plan.get("results_root", root / "outputs")).resolve()
    allowed_roots = {root / "outputs"}
    if "improved_root" in plan:
        allowed_roots.add(Path(plan["improved_root"]).resolve() / "higgs/outputs")
    if results_root not in allowed_roots or output.exists() or not inside(output, results_root):
        raise ValueError("Use a new result directory inside an authorized outputs root")
    output.mkdir(parents=True)
    write_csv(output / "table1_mixed.csv", [{"Metric": k, "Kernel": KERNELS[k], "Value": values[k]} for k in METRICS])
    write_csv(output / "table3_summary.csv", summary)
    write_csv(output / "table3_per_graph.csv", rows)
    report = {"protocol": PROTOCOL, "dataset": job["dataset"], "variant": job["variant"],
              "count": job["count"], "test_graph_ids": job["test_graph_ids"], "kernel_mapping": KERNELS,
              "mmd": values, "table3": summary, "component_policy": "lcc",
              "evaluation_louvain": plan["evaluation_louvain"], "inputs_unchanged": True,
              "input_sha256": expected, "previous_tv_report": job.get("prior_tv_report"),
              "training_budget": job.get("training_budget"), "budget_note": job.get("budget_note"),
              "scope": "Kernel-only re-evaluation; graph policy and test IDs unchanged; extension benchmarks have no HiGGs paper scores"}
    write_json(output / "mixed_evaluation.json", report)
    print("\nTABLE 1 MIXED MMD", flush=True)
    for name in METRICS:
        print(f"{name:<20} {values[name]:.8f}", flush=True)
    print("\nTABLE 3: Real mean | Generated mean", flush=True)
    for row in summary:
        print(f"{row['Metric']:<20} {row['RealMean']:.6f} | {row['GeneratedMean']:.6f}", flush=True)
    print(f"MIXED_JOB_COMPLETE dataset={job['dataset']} variant={job['variant']} OUTPUT={output}", flush=True)
    return report


def summarize_batch(plan, output, statuses):
    reports = {}
    for job, status in zip(plan["jobs"], statuses):
        if status.get("returncode") == 0:
            reports[(job["dataset"], job["variant"])] = read_json(Path(status["output"]) / "mixed_evaluation.json")
    table1, table3 = [], []
    datasets = list(dict.fromkeys(job["dataset"] for job in plan["jobs"]))
    for dataset in datasets:
        old, new = reports.get((dataset, "original")), reports.get((dataset, "improved"))
        if old and new and (old["test_graph_ids"] != new["test_graph_ids"] or old["component_policy"] != new["component_policy"]):
            raise ValueError("Comparison inputs differ")
        print(f"\n=== {dataset.upper()} MIXED: original | improved ===", flush=True)
        for metric in METRICS:
            a = old["mmd"][metric] if old else None
            b = new["mmd"][metric] if new else None
            row = {"Dataset": dataset, "Metric": metric, "Kernel": KERNELS[metric],
                   "Original": a, "Improved": b, "ImprovedMinusOriginal": b - a if a is not None and b is not None else None}
            table1.append(row)
            print(f"{metric:<20} {a if a is not None else 'PENDING'} | {b if b is not None else 'PENDING'}", flush=True)
        ref = old or new
        if ref:
            arows = {r["Metric"]: r for r in old["table3"]} if old else {}
            brows = {r["Metric"]: r for r in new["table3"]} if new else {}
            for row in ref["table3"]:
                metric = row["Metric"]
                if old and new and not math.isclose(arows[metric]["RealMean"], brows[metric]["RealMean"], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Real Table 3 references differ")
                table3.append({"Dataset": dataset, "Metric": metric, "Real": row["RealMean"],
                               "Original": arows.get(metric, {}).get("GeneratedMean"),
                               "Improved": brows.get(metric, {}).get("GeneratedMean")})
    write_csv(output / "comparison_table1_mixed.csv", table1)
    write_csv(output / "comparison_table3_means.csv", table3)
    write_json(output / "batch_summary.json", {"protocol": PROTOCOL, "jobs": statuses, "pending": plan["pending"]})


def run_batch(plan_path):
    plan = read_json(plan_path)
    output = plan_path.parent
    statuses = []
    for index, job in enumerate(plan["jobs"]):
        name = f"{job['dataset']}_{job['variant']}"
        destination = output / name
        log = output / f"{name}.log"
        print(f"JOB_START {index + 1}/{len(plan['jobs'])} {name} LOG={log}", flush=True)
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-u", "-B", str(SELF), "worker", "--plan", str(plan_path),
                                        "--index", str(index), "--output", str(destination)],
                                       stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            print(f"JOB_PID {name} {process.pid}", flush=True)
            code = process.wait()
        statuses.append({"dataset": job["dataset"], "variant": job["variant"], "pid": process.pid,
                         "returncode": code, "log": str(log), "output": str(destination)})
        write_json(output / "progress.json", {"finished": len(statuses), "total": len(plan["jobs"]), "jobs": statuses,
                                             "pending": plan["pending"]}, exclusive=False)
        print(f"JOB_{'DONE' if code == 0 else 'FAILED'} {name} EXIT={code}", flush=True)
    summarize_batch(plan, output, statuses)
    if any(s["returncode"] for s in statuses):
        raise RuntimeError("Some evaluations failed; completed results retained; inspect per-job logs")
    print(f"MIXED_BATCH_COMPLETE READY={len(statuses)} PENDING={len(plan['pending'])} OUTPUT={output}", flush=True)


def launch(plan):
    if sys.platform != "linux":
        raise RuntimeError("Detached launch is only for the Linux server; use tests locally")
    import fcntl

    root = Path(plan["original_root"])
    logs = root / "run_logs"
    logs.mkdir(exist_ok=True)
    lock = (logs / ".mixed_collection_evaluation.lock").open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"mixed_collection_tables_{stamp}"
        output = root / "outputs" / name
        output.mkdir(parents=True, exist_ok=False)
        path = output / "plan.json"
        write_json(path, plan)
        log = logs / f"{name}.log"
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-u", "-B", str(SELF), "run", "--plan", str(path),
                                        "--lock-fd", str(lock.fileno())], env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
        log.with_suffix(".pid").write_text(str(process.pid) + "\n", encoding="ascii")
        print(f"PID={process.pid}\nLOG={log}\nOUTPUT={output}", flush=True)
    except BlockingIOError:
        raise RuntimeError("A mixed evaluation batch is already running; do not repeat launch")
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--original-root", type=Path, default=ORIGINAL)
        command.add_argument("--improved-root", type=Path, default=IMPROVED)
        command.add_argument("--datasets", nargs="+", choices=tuple(SPECS), default=list(SPECS))
        command.add_argument("--variants", nargs="+", choices=("original", "improved"), default=["original", "improved"])
    command = commands.add_parser("run")
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--lock-fd", type=int)
    command = commands.add_parser("worker")
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--index", required=True, type=int)
    command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action in ("plan", "launch"):
        plan = make_plan(args.original_root, args.improved_root, args.datasets, args.variants)
        for job in plan["jobs"]:
            print(f"READY {job['dataset']} {job['variant']} N={job['count']} PRIOR={job.get('prior_tv_report', 'completed DD sampling')}", flush=True)
        for job in plan["pending"]:
            print(f"PENDING {job['dataset']} {job['variant']}: {job['reason']}", flush=True)
        if not plan["jobs"]:
            raise RuntimeError("No completed samples available")
        if args.action == "launch":
            launch(plan)
    elif args.action == "run":
        if args.lock_fd is not None:
            root = Path(read_json(args.plan)["original_root"])
            expected = (root / "run_logs/.mixed_collection_evaluation.lock").stat()
            actual = os.fstat(args.lock_fd)
            if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
                raise ValueError("Incorrect inherited evaluation lock")
        run_batch(args.plan.resolve())
    else:
        evaluate_job(read_json(args.plan), args.index, args.output.resolve())


if __name__ == "__main__":
    main()
