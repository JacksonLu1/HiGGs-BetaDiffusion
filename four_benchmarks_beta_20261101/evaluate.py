"""Frozen mixed-kernel evaluation for the four topology-only benchmarks.

Existing original-checkout evaluation functions are imported read-only. This
entry never repartitions, retrains, resamples, filters graphs, or rewrites a
source graph, cache, checkpoint or previous evaluation directory.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
ORIGINAL = Path('/path/to/HiGGs-main1/HiGGs-main')
NAMES = ('ego_small', 'proteins', 'msrc21', 'ego')
METRICS = ('Nodes', 'Degree', 'Clustering', 'Eccentricity', 'Spectre')
KERNELS = dict(zip(METRICS, ('Gaussian-TV', 'Gaussian-EMD', 'Gaussian-EMD',
                           'Gaussian', 'Gaussian-EMD')))
SOURCE_FILES = (
    'higgs/evaluate_graph_collection.py', 'higgs/evaluate_collection_tables.py',
    'higgs/evaluate_sbm_mmd.py', 'dgd/analysis/dist_helper.py',
    'dgd/analysis/spectre_utils.py', 'dgd/metrics/mmd/calculators.py',
    'dgd/metrics/mmd/kernels.py', 'dgd/utils.py',
    'dgd/datasets/collection_sources.py',
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def checked_hashes(values):
    for path, digest in values.items():
        if sha(path) != digest:
            raise RuntimeError('Input/source changed during evaluation: ' + str(path))


def load_helpers(original):
    """Only the original experiment's metric backend owns the implementation."""
    sys.path[:0] = [str(original / 'higgs'), str(original / 'dgd')]
    names = ('evaluate_graph_collection', 'evaluate_collection_tables', 'evaluate_sbm_mmd')
    modules = [importlib.import_module(name) for name in names]
    for name, module in zip(names, modules):
        if Path(module.__file__).resolve() != original / 'higgs' / (name + '.py'):
            raise RuntimeError('Unexpected evaluation module imported: ' + name)
    graphs, tables, metrics = modules
    nested = ((metrics.spectre_utils, original / 'dgd/analysis/spectre_utils.py'),
              (metrics.mmd_calculators, original / 'dgd/metrics/mmd/calculators.py'),
              (metrics.dist_helper, original / 'dgd/analysis/dist_helper.py'))
    for module, expected in nested:
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError('Mixed metric backend was imported from another checkout')
    return graphs, tables, metrics


def mixed_metrics(real, generated, metrics):
    """Exact dispatch used by the original four-benchmark mixed evaluator."""
    if not real or not generated or any(len(graph) == 0 for graph in real + generated):
        raise ValueError('Keep every valid graph; empty graphs are errors, never silently dropped')
    if len(real) != len(generated):
        raise ValueError('Formal evaluation requires the complete equal-count comparison')
    calls = (
        ('Nodes', lambda: metrics.node_count_mmd(real, generated, 'tv')),
        ('Degree', lambda: metrics.spectre_utils.degree_stats(
            real, generated, is_parallel=True, compute_emd=True)),
        ('Clustering', lambda: metrics.spectre_utils.clustering_stats(
            real, generated, bins=100, is_parallel=True, compute_emd=True)),
        ('Eccentricity', lambda: metrics.mmd_calculators.radius_stats(
            real, generated, compute_emd=True, is_parallel=True, PRINT_TIME=False)),
        ('Spectre', lambda: metrics.spectre_utils.spectral_stats(
            real, generated, is_parallel=True, n_eigvals=-1, compute_emd=True)),
    )
    result = []
    for name, compute in calls:
        print('METRIC_START {} KERNEL={}'.format(name, KERNELS[name]), flush=True)
        value = float(compute())
        if not math.isfinite(value):
            raise ValueError('Nonfinite metric: ' + name)
        result.append({'Metric': name, 'Kernel': KERNELS[name], 'Value': value})
        print('METRIC_DONE {} {:.12g}'.format(name, value), flush=True)
    return result


def strict_samples(directory, count):
    """No recursive mixed-run discovery, missing indices or extra graph files."""
    directory = Path(directory).resolve()
    pattern = re.compile(r'sampled_graph_(\d+)\.pkl$')
    numbered = {}
    for path in directory.glob('sampled_graph_*.pkl'):
        match = pattern.fullmatch(path.name)
        if (not match or not path.is_file() or path.is_symlink()
                or path.resolve().parent != directory):
            raise ValueError('Unexpected sample filename or symlink: ' + str(path))
        index = int(match.group(1))
        if index in numbered or path.name != 'sampled_graph_{}.pkl'.format(index):
            raise ValueError('Duplicate/noncanonical graph index: ' + str(path))
        numbered[index] = path
    if sorted(numbered) != list(range(count)):
        raise ValueError('Sampling incomplete: expected contiguous indices 0..{}'.format(count - 1))
    return [numbered[index] for index in range(count)]


def graph_row(graph, graph_id, group, graphs, tables):
    if graph.number_of_edges():
        return tables.graph_row(graph, graph_id, group, 1, 0)
    # Louvain/modularity is undefined with zero total degree. Match the original
    # four-benchmark evaluator, retaining the sample as singleton communities.
    stats = graphs.statistics(graph, graph_id)
    return {'Group': group, 'GraphID': graph_id, 'Nodes': stats['nodes'],
            'Edges': 0, 'Diameter': stats['diameter'], 'NC': len(graph),
            'MedianCommunitySize': 1., 'Clustering': stats['average_clustering'],
            'Density': stats['density'], 'Transitivity': stats['transitivity']}


def descriptor_settings():
    return {
        'Nodes': {'descriptor': 'aggregate whole-graph node-count histogram', 'sigma': 1.0},
        'Degree': {'descriptor': 'per-graph degree histogram', 'sigma': 1.0,
                   'distance_scaling': 1.0},
        'Clustering': {'descriptor': 'per-graph node clustering histogram',
                       'bins': 100, 'range': [0.0, 1.0], 'sigma': 0.1,
                       'distance_scaling': 100},
        'Eccentricity': {'descriptor': 'legacy all-ordered-pairs shortest-path lengths including self paths',
                         'bins': 50, 'range': 'numpy automatic per graph', 'sigma': 1.0,
                         'kernel_note': 'compute_emd=True selects Gaussian, NOT Gaussian-EMD'},
        'Spectre': {'descriptor': 'all normalized-Laplacian eigenvalues per graph',
                    'bins': 200, 'range': [-0.00001, 2.0], 'sigma': 1.0,
                    'distance_scaling': 1.0, 'n_eigvals': -1,
                    'community_truncation': False},
        'histogram_normalization': {
            'Nodes_Degree_Clustering_Spectre': 'analysis.dist_helper sum + 1e-6',
            'Eccentricity': 'metrics.mmd.kernels exact sum',
        },
    }


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root = Path(args.root).resolve()
    if root != ROOT or sys.platform != 'linux':
        raise ValueError('Use this entry only in the modified Linux-server experiment root')
    out, sample_dir = args.output.resolve(), args.samples_dir.resolve()
    allowed = (root / 'higgs/outputs', root / 'outputs')
    if not any(parent in out.parents for parent in allowed) or out.exists() or args.output.is_symlink():
        raise ValueError('Use a NEW evaluation directory inside the modified root outputs')
    if not (root / 'higgs/outputs' in sample_dir.parents or root / 'outputs' in sample_dir.parents):
        raise ValueError('Only the modified experiment samples may be evaluated here')
    sys.path.insert(0, str(root / 'tools'))
    import qb_beta_common as common
    item = common.get_item(root, args.dataset)
    before = common.baseline(root, args.dataset)
    before[str(Path(__file__).resolve())] = sha(__file__)
    for name in SOURCE_FILES:
        path = ORIGINAL / name
        before[str(path)] = sha(path)
    graphs, tables, metrics = load_helpers(ORIGINAL)
    manifest_path = Path(item['manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    test_ids = manifest['splits']['test']
    if len(test_ids) != item['splits']['test']:
        raise ValueError('Frozen setup test count differs from its manifest')
    paths = strict_samples(sample_dir, len(test_ids))
    sample_hashes = {str(path): sha(path) for path in paths}
    before.update(sample_hashes)
    # The new sampler creates this audit only after the requested full count.
    audit_path = sample_dir.parent / 'input_hash_check.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('unchanged') is not True:
        raise ValueError('Sampling completion hash audit did not pass')
    before[str(audit_path)] = sha(audit_path)
    checked_hashes(before)
    records = {record['id']: record for record in manifest['graphs']}
    real = [graphs.normalize_graph(graphs.record_graph(records[index]), 'lcc') for index in test_ids]
    generated = []
    for path in paths:
        # Files are trusted output from this user's local server sampler only.
        with path.open('rb') as handle:
            generated.append(graphs.normalize_graph(pickle.load(handle), 'lcc'))
    if len(real) != len(generated) or len(real) != len(test_ids):
        raise ValueError('A graph was lost while loading; no count-mismatch mode is allowed')
    values = mixed_metrics(real, generated, metrics)
    rows = []
    print('Computing Table 3 communities; frozen training partitions are not modified', flush=True)
    for group, items, ids in (('Real', real, test_ids), ('Generated', generated, range(len(generated)))):
        for index, (graph, graph_id) in enumerate(zip(items, ids), 1):
            rows.append(graph_row(graph, graph_id, group, graphs, tables))
            if index % 10 == 0 or index == len(items):
                print('TABLE3 {} {}/{}'.format(group, index, len(items)), flush=True)
    summary = tables.summary_rows(rows)
    checked_hashes(before)
    # Do not publish partial metric files until every metric and audit succeeds.
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'table1_mixed.csv', values)
    write_csv(out / 'table3_summary.csv', summary)
    write_csv(out / 'table3_per_graph.csv', rows)
    protocol = {
        'dataset': args.dataset, 'variant': 'beta_h2_beta_h1_mean_twohop_hog_original_glue',
        'protocol': 'higgs_frozen_mixed_v1', 'kernel_mapping': KERNELS,
        'manifest_sha256': item['manifest_sha256'], 'test_graph_ids': test_ids,
        'count': len(paths), 'component_policy': 'lcc',
        'component_policy_detail': 'independent LCC of every real and generated complete graph',
        'descriptor_settings': descriptor_settings(),
        'evaluation_louvain': {'resolution': 1, 'seed': 0, 'weight': None},
        'edgeless_community_policy': 'one_community_per_node',
        'diameter_policy': 'largest_component_per_graph',
        'training_partitions_modified': False, 'learning_scope': 'topology_only',
        'all_requested_samples_retained': True,
        'samples_sha256': sample_hashes, 'input_sha256': before,
        'table1': values, 'table3_means': [{key: row[key] for key in
            ('Metric', 'RealMean', 'GeneratedMean')} for row in summary],
        'published_paper_scope': 'The HiGGs paper does not publish these four extension datasets',
    }
    with (out / 'protocol.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
    print('\nTABLE 1 MIXED MMD', flush=True)
    for row in values:
        print('{:<20} {:.8f}  {}'.format(row['Metric'], row['Value'], row['Kernel']), flush=True)
    print('\nTABLE 3: Real mean | Generated mean', flush=True)
    for row in summary:
        print('{:<20} {:.6f} | {:.6f}'.format(row['Metric'], row['RealMean'], row['GeneratedMean']), flush=True)
    print('OUTPUT=' + str(out), flush=True)
    print('PASS: {}_MEAN_HOG_MIXED_TABLE1_TABLE3_COMPLETE'.format(args.dataset.upper()), flush=True)
    print('No training, sampling, source edits or input changes during evaluation.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', choices=NAMES, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
