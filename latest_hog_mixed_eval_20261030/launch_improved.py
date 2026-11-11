"""CPU-only mixed re-evaluation of completed latest-HOG sample graphs."""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
SELF = Path(__file__).resolve()
BUNDLE = SELF.parent / 'mixed_collection_eval.py'
spec = importlib.util.spec_from_file_location('latest_hog_mixed_backend', BUNDLE)
mix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mix)
ROOT, ORIGINAL = mix.IMPROVED, mix.ORIGINAL
POINTER = 'logs/latest_hog_mixed_eval.json'
LOCK = 'run_logs/.mixed_collection_evaluation.lock'


def completed_match(path, job, plan):
    """Accept only completed, unchanged inputs and the exact mixed protocol."""
    try:
        report = mix.read_json(path)
        fixed = {'protocol': mix.PROTOCOL, 'dataset': job['dataset'], 'variant': 'improved',
                 'count': job['count'], 'test_graph_ids': job['test_graph_ids'],
                 'component_policy': 'lcc', 'kernel_mapping': mix.KERNELS,
                 'evaluation_louvain': plan['evaluation_louvain'], 'inputs_unchanged': True}
        if any(report.get(key) != value for key, value in fixed.items()):
            return None
        inputs = report['input_sha256']
        for name, digest in {**plan['sources_sha256'], **job['samples_sha256'],
                             job['manifest']: job['manifest_sha256']}.items():
            if inputs.get(name) != digest:
                return None
        reported_samples = {p: value for p, value in inputs.items()
                            if Path(p).name.startswith('sampled_graph_') and Path(p).suffix == '.pkl'}
        if reported_samples != job['samples_sha256']:
            return None
        mix.verify_hashes(inputs)
        if set(report['mmd']) != set(mix.METRICS):
            return None
        if ('prior_nodes_mmd' in job and not math.isclose(
                float(report['mmd']['Nodes']), job['prior_nodes_mmd'], rel_tol=1e-8, abs_tol=1e-10)):
            return None
        with (path.parent / 'table1_mixed.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(mix.METRICS) or {r['Metric'] for r in rows} != set(mix.METRICS):
            return None
        for row in rows:
            value = float(row['Value'])
            if (not math.isfinite(value) or row['Kernel'] != mix.KERNELS[row['Metric']]
                    or not math.isclose(value, report['mmd'][row['Metric']], rel_tol=1e-12, abs_tol=1e-12)):
                return None
        with (path.parent / 'table3_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        expected = {row['Metric']: row for row in report['table3']}
        if set(expected) != {'Nodes', 'Edges', 'Diameter', 'NC', 'MedianCommunitySize', 'Clustering', 'Density', 'Transitivity'}:
            return None
        if len(rows) != len(expected) or {row['Metric'] for row in rows} != set(expected):
            return None
        for row in rows:
            for field in ('RealMean', 'GeneratedMean', 'RealStd', 'GeneratedStd'):
                value = float(row[field])
                if not math.isfinite(value) or not math.isclose(value, expected[row['Metric']][field], rel_tol=1e-12, abs_tol=1e-12):
                    return None
        return {'output': str(path.parent), 'report_sha256': mix.sha(path),
                'table1_sha256': mix.sha(path.parent / 'table1_mixed.csv'),
                'table3_sha256': mix.sha(path.parent / 'table3_summary.csv')}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, csv.Error):
        return None


def find_completed(plan, job):
    paths = []
    for owner in (Path(plan['original_root']), Path(plan['improved_root'])):
        for base in (owner / 'outputs', owner / 'higgs/outputs'):
            paths.extend(base.glob('mixed_collection_tables_*/' + job['dataset'] + '_improved/mixed_evaluation.json'))
    for path in sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        matched = completed_match(path, job, plan)
        if matched:
            return matched
    return None


def prepare(datasets):
    plan = mix.make_plan(ORIGINAL, ROOT, datasets, ['improved'])
    plan['results_root'] = str(ROOT / 'higgs/outputs')
    plan['tool_sha256'][str(SELF)] = mix.sha(SELF)
    for job in plan['jobs']:
        job['reuse'] = find_completed(plan, job)
        print(('REUSE' if job['reuse'] else 'READY'), job['dataset'], 'N=' + str(job['count']),
              'RESULT=' + job['reuse']['output'] if job['reuse'] else '', flush=True)
    for job in plan['pending']:
        print('PENDING', job['dataset'], job['reason'], flush=True)
    return plan


def status(root=ROOT, original=ORIGINAL):
    pointer = root / POINTER
    if pointer.is_file():
        info = mix.read_json(pointer)
        print(json.dumps(info, indent=2), flush=True)
        progress = Path(info['output']) / 'progress.json'
        if progress.is_file():
            print(progress.read_text(), flush=True)
        log = Path(info['log'])
        if log.is_file():
            subprocess.run(['tail', '-n', '45', str(log)], check=False)
    logs = sorted((original / 'run_logs').glob('mixed_collection_tables_*.log'))
    if logs:
        log = logs[-1]
        print('ORIGINAL_SHARED_BATCH_LOG=' + str(log), flush=True)
        output = original / 'outputs' / log.stem
        path = output / 'plan.json'
        if path.is_file():
            plan = mix.read_json(path)
            print('Existing batch improved arms:', [job['dataset'] for job in plan['jobs'] if job['variant'] == 'improved'], flush=True)
        progress = output / 'progress.json'
        if progress.is_file():
            print(progress.read_text(), flush=True)
        subprocess.run(['tail', '-n', '35', str(log)], check=False)
        print('View with: tail -n 60 -F "' + str(log) + '"', flush=True)
    elif not pointer.is_file():
        print('No matching launcher pointer or original shared-batch log found.', flush=True)


def verify_plan(plan):
    if (Path(plan['original_root']) != ORIGINAL or Path(plan['improved_root']) != ROOT
            or Path(plan['results_root']) != ROOT / 'higgs/outputs'
            or any(job['variant'] != 'improved' for job in plan['jobs'])):
        raise ValueError('Unexpected experiment roots, result destination, or variant')
    mix.verify_hashes({**plan['sources_sha256'], **plan['tool_sha256']})


def summarize_latest(plan, output, statuses):
    """Improved-only exports; original arms are intentionally not recomputed."""
    table1, table3 = [], []
    for job, status_record in zip(plan['jobs'], statuses):
        if status_record['returncode']:
            continue
        report = mix.read_json(Path(status_record['output']) / 'mixed_evaluation.json')
        for metric in mix.METRICS:
            table1.append({'Dataset': job['dataset'], 'Metric': metric,
                           'Kernel': mix.KERNELS[metric], 'Value': report['mmd'][metric]})
        for row in report['table3']:
            table3.append({'Dataset': job['dataset'], 'Metric': row['Metric'],
                           'RealMean': row['RealMean'], 'GeneratedMean': row['GeneratedMean']})
    mix.write_csv(output / 'latest_hog_table1_mixed.csv', table1)
    mix.write_csv(output / 'latest_hog_table3_means.csv', table3)
    mix.write_json(output / 'batch_summary.json', {'protocol': mix.PROTOCOL,
        'variant': 'improved', 'jobs': statuses, 'pending': plan['pending'],
        'note': 'Only modified-model evaluation requested; original results are not recalculated here.'})


def run(plan_path, expected_sha, lock_fd):
    if mix.sha(plan_path) != expected_sha:
        raise ValueError('Frozen launch plan changed')
    plan = mix.read_json(plan_path)
    verify_plan(plan)
    expected, inherited = (ORIGINAL / LOCK).stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino):
        raise ValueError('Wrong inherited shared-evaluation lock')
    output, statuses = plan_path.parent, []
    for index, job in enumerate(plan['jobs']):
        name = job['dataset'] + '_improved'
        if job['reuse']:
            reused = job['reuse']
            path = Path(reused['output']) / 'mixed_evaluation.json'
            current = completed_match(path, job, plan)
            if current != reused:
                raise ValueError('Reused completed result changed: ' + str(path))
            record = {'dataset': job['dataset'], 'variant': 'improved', 'returncode': 0,
                      'output': reused['output'], 'reused': True}
            print('JOB_REUSED', name, reused['output'], flush=True)
        else:
            log = output / (name + '.log')
            destination = output / name
            print('JOB_START', name, 'LOG=' + str(log), flush=True)
            with log.open('x', encoding='utf-8') as handle:
                child = subprocess.Popen([sys.executable, '-u', '-B', str(BUNDLE), 'worker',
                    '--plan', str(plan_path), '--index', str(index), '--output', str(destination)],
                    stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, pass_fds=(lock_fd,))
                print('JOB_PID', name, child.pid, flush=True)
                code = child.wait()
            record = {'dataset': job['dataset'], 'variant': 'improved', 'returncode': code,
                      'output': str(destination), 'log': str(log), 'pid': child.pid, 'reused': False}
            print('JOB_DONE' if code == 0 else 'JOB_FAILED', name, 'EXIT=' + str(code), flush=True)
        statuses.append(record)
        mix.write_json(output / 'progress.json', {'finished': len(statuses), 'total': len(plan['jobs']),
            'jobs': statuses, 'pending': plan['pending']}, exclusive=False)
    summarize_latest(plan, output, statuses)
    if any(record['returncode'] for record in statuses):
        raise RuntimeError('Some evaluations failed; results and logs retained')
    print('PASS: LATEST_HOG_MIXED_EVALUATION_COMPLETE', flush=True)
    print('OUTPUT=' + str(output), flush=True)
    print('No training, sampling, repartitioning, model changes or old-result overwrites.', flush=True)


def launch(plan):
    if sys.platform != 'linux':
        raise RuntimeError('Detached launch is Linux-only')
    import fcntl
    if not plan['jobs']:
        print('No completed samples ready; pending entries were not evaluated.', flush=True)
        return
    if all(job['reuse'] for job in plan['jobs']):
        print('ALL_AVAILABLE_RESULTS_ALREADY_COMPLETE: reuse the RESULT directories above; no duplicate run.', flush=True)
        return
    verify_plan(plan)
    lock_path = ORIGINAL / LOCK
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('SHARED_MIXED_BATCH_ALREADY_RUNNING: no second job launched. Check whether improved arms are already queued.', flush=True)
            status()
            return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name = 'mixed_collection_tables_latest_hog_' + stamp
        output = ROOT / 'higgs/outputs' / name
        output.mkdir(parents=True, exist_ok=False)
        path = output / 'plan.json'
        mix.write_json(path, plan)
        logs = ROOT / 'logs/evaluation'
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (name + '.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', WANDB_MODE='disabled',
                   PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('x', encoding='utf-8') as handle:
            child = subprocess.Popen([sys.executable, '-u', '-B', str(SELF), 'run', '--plan', str(path),
                '--plan-sha', mix.sha(path), '--lock-fd', str(lock.fileno())], env=env, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(lock.fileno(),))
        Path(str(log) + '.pid').write_text(str(child.pid))
        Path(str(log) + '.out').write_text(str(output))
        info = {'pid': child.pid, 'log': str(log), 'output': str(output), 'plan': str(path)}
        mix.write_json(ROOT / POINTER, info, exclusive=False)
        print(json.dumps(info, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'launch', 'status', 'run'))
    parser.add_argument('--datasets', nargs='+', choices=tuple(mix.SPECS), default=list(mix.SPECS))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.action == 'status':
        status()
    elif args.action == 'run':
        run(args.plan.resolve(), args.plan_sha, args.lock_fd)
    else:
        plan = prepare(args.datasets)
        if args.action == 'launch':
            launch(plan)


if __name__ == '__main__':
    main()
