"""Use the original server evaluator, then compare matching frozen baselines."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparable(protocol, name, digest, count):
    return (protocol.get('dataset') == name and protocol.get('manifest_sha256') == digest
            and protocol.get('generated_count') == count and protocol.get('test_graph_count') == count
            and protocol.get('component_policy') == 'lcc' and protocol.get('table1_mode') == 'tv'
            and protocol.get('table3_evaluation_louvain') == {'resolution': 1.0, 'seed': 0, 'weight': None})


def comparisons(original, name, item, out, c):
    candidates = []
    for parent in (original / 'outputs', original / 'higgs/outputs'):
        if not parent.exists():
            continue
        for directory in parent.glob(name + '_table1_table3*'):
            protocol = directory / 'table_protocol.json'
            if not protocol.is_file():
                continue
            data = json.loads(protocol.read_text())
            if comparable(data, name, item['manifest_sha256'], item['splits']['test']):
                if (directory / 'table1_mmd.csv').is_file() and (directory / 'table3_summary.csv').is_file():
                    candidates.append(directory)
    if not candidates:
        (out / 'comparison_status.json').write_text(json.dumps({
            'status': 'baseline_not_found', 'new_metrics_complete': True,
            'note': 'No matching saved original baseline; no original sampling/evaluation rerun.'}, indent=2))
        print('New metrics complete; matching original baseline CSV not found. No original rerun.', flush=True)
        return
    base = sorted(candidates, key=lambda p: p.name)[-1]
    sources = [base / 'table_protocol.json', base / 'table1_mmd.csv', base / 'table3_summary.csv']
    hashes = {str(p): c.sha(p) for p in sources}
    def mmd(directory):
        rows = read_csv(directory / 'table1_mmd.csv')
        return {r['Metric']: float(r['Value']) for r in rows if r['Mode'] == 'tv'}
    old, new = mmd(base), mmd(out)
    assert set(old) == set(new) == {'Nodes', 'Degree', 'Clustering', 'Eccentricity', 'Spectre'}
    table1 = [{'Metric': key, 'OriginalHiGGs': old[key], 'BetaMeanHOG': new[key],
               'NewMinusOriginal': new[key] - old[key], 'LowerIsBetter': True,
               'Improved': new[key] < old[key]} for key in new]
    assert all(math.isfinite(float(r[key])) for r in table1
               for key in ('OriginalHiGGs', 'BetaMeanHOG', 'NewMinusOriginal'))
    old3 = {r['Metric']: r for r in read_csv(base / 'table3_summary.csv')}
    new3 = {r['Metric']: r for r in read_csv(out / 'table3_summary.csv')}
    assert set(old3) == set(new3)
    table3 = []
    for key in new3:
        real = float(new3[key]['RealMean'])
        assert math.isclose(real, float(old3[key]['RealMean']), rel_tol=1e-10, abs_tol=1e-10)
        a, b = float(old3[key]['GeneratedMean']), float(new3[key]['GeneratedMean'])
        table3.append({'Metric': key, 'RealMean': real, 'OriginalHiGGsMean': a,
                       'BetaMeanHOGMean': b, 'OriginalAbsoluteMeanError': abs(a-real),
                       'NewAbsoluteMeanError': abs(b-real)})
    c.check_hashes(hashes)
    write_csv(out / 'table1_original_vs_beta_mean_hog.csv', table1)
    write_csv(out / 'table3_original_vs_beta_mean_hog.csv', table3)
    (out / 'comparison_status.json').write_text(json.dumps({
        'status': 'complete', 'baseline': str(base), 'baseline_input_sha256': hashes,
        'scope': 'one fixed split and seed; original-code extension baseline, not published paper values',
        'table1_improved': sum(r['Improved'] for r in table1), 'table1_metrics': len(table1)}, indent=2))
    print('\nTABLE 1: original HiGGs | Beta+mean+HOG (lower better)', flush=True)
    for row in table1:
        print('{Metric:<15} {OriginalHiGGs:.8f} | {BetaMeanHOG:.8f}'.format(**row), flush=True)
    print('\nTABLE 3: real | original HiGGs | Beta+mean+HOG (means only)', flush=True)
    for row in table3:
        print('{Metric:<20} {RealMean:.6f} | {OriginalHiGGsMean:.6f} | {BetaMeanHOGMean:.6f}'.format(**row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('tree', 'grid'), required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out, samples = args.root.resolve(), args.output.resolve(), args.samples_dir.resolve()
    sys.path.insert(0, str(root / 'tools'))
    import tg_beta_common as c
    assert root == c.ROOT and c.ROOT != c.ORIGINAL
    assert root / 'higgs/outputs' in out.parents and not out.exists()
    assert root / 'higgs/outputs' in samples.parents and samples.name == 'sampling'
    item = c.get_item(root, args.dataset)
    before = c.baseline(root, args.dataset)
    evaluator = c.ORIGINAL / 'higgs/evaluate_collection_tables.py'
    original_sources = [evaluator, c.ORIGINAL / 'higgs/evaluate_graph_collection.py',
                        c.ORIGINAL / 'higgs/evaluate_sbm_mmd.py',
                        c.ORIGINAL / 'dgd/datasets/collection_sources.py',
                        c.ORIGINAL / 'dgd/datasets/tree_grid_sources.py']
    before.update({str(p): c.sha(p) for p in original_sources})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               WANDB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable, '-u', '-B', str(evaluator), '--dataset', args.dataset,
        '--manifest', item['manifest'], '--samples-dir', str(samples), '--output-dir', str(out),
        '--mode', 'tv', '--component-policy', 'lcc', '--louvain-resolution', '1',
        '--louvain-seed', '0'], cwd=c.ORIGINAL, env=env, check=True)
    protocol = json.loads((out / 'table_protocol.json').read_text())
    assert comparable(protocol, args.dataset, item['manifest_sha256'], item['splits']['test'])
    comparisons(c.ORIGINAL, args.dataset, item, out, c)
    c.check_hashes(before)
    (out / 'input_hash_check.json').write_text(json.dumps({'unchanged': True, 'sha256': before}, indent=2))
    print('PASS: {}_TABLE1_TABLE3_AND_AVAILABLE_BASELINE_COMPARISON_COMPLETE'.format(args.dataset.upper()), flush=True)
    print('No training, sampling, source edits or input changes during evaluation.', flush=True)


if __name__ == '__main__':
    main()
