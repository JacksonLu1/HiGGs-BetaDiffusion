"""Ego-only additive recovery: replay verified prefix, finish151, evaluate.

No trained model, source, seed, component policy, failed-run artifact, or earlier
dataset result is edited. The old run has no per-graph RNG snapshots, so this is
a verified replay into a new directory, NOT a direct resume at graph32.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import runpy
import subprocess
import sys

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
ORIGINAL = Path('/path/to/HiGGs-main1/HiGGs-main')
PACKAGE = Path(__file__).resolve().parent
FAILED_NAME = 'four_benchmarks_mean_hog_20261101_233541_555705_ego_gpu2'
PREFIX_COUNT = 31
TOTAL_COUNT = 151
MANIFEST = 'bd5a14072b1cf68f9e6446f0e7e79ef85a103fbfe106bf1b36233bd277027e80'
POLICY = {
    'name': 'ego_glue_eigh_cpu_float64_on_convergence_failure_v1',
    'scope': 'only this recovery process, Ego Glue EigenFeatures',
    'primary': 'unchanged torch.linalg.eigh',
    'fallback': 'finite symmetric same matrix on CPU float64; no jitter or graph changes',
    'nonunique_eigenvectors': 'fallback basis may differ; not bit-identical to another solver',
    'seed': 1234, 'h2_component_policy': 'full', 'h1_component_policy': 'full',
    'h1_batch_size': 4, 'glue_batch_size': 1,
    'prefix_policy': 'replay all31 and require exact graph data AND SHA256 before continuation',
    'rng_snapshots': 'audit/future recovery data only; this entry does not implement snapshot resume',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(tmp, path)


def check_hashes(expected):
    for name, digest in expected.items():
        require(sha(name) == digest, 'Frozen input/source changed: ' + name)


def inside(path, parent):
    return Path(path).resolve().is_relative_to(Path(parent).resolve())


def verify_bundle():
    path = PACKAGE / 'BUNDLE.json'
    record = read_json(path)
    require(record.get('schema_version') == 1 and isinstance(record.get('sha256'), dict),
            'Invalid upload bundle inventory')
    require({'run_recovery.py', 'stable_eigen.py'} <= set(record['sha256']),
            'Runtime source files missing from bundle inventory')
    hashes = {}
    for relative, digest in record['sha256'].items():
        source = PACKAGE / relative
        require(source.is_file() and not source.is_symlink() and inside(source, PACKAGE),
                'Invalid bundle path: ' + relative)
        require(sha(source) == digest, 'Uploaded recovery package differs: ' + relative)
        hashes[str(source)] = digest
    hashes[str(path)] = sha(path)
    return hashes


def load_sampler(root):
    sys.path.insert(0, str(root / 'tools'))
    import qb_beta_common as common
    require(Path(common.__file__).resolve() == root / 'tools/qb_beta_common.py',
            'Wrong common module')
    spec = importlib.util.spec_from_file_location('ego_recovery_existing_sampler',
                                                 root / 'tools/qb_beta_sample.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return common, module


def checked_prefix(sample, expected_count=PREFIX_COUNT):
    """Only a complete contiguous trusted prefix; never infer missing graphs."""
    sample = Path(sample).resolve()
    rows = read_json(sample / 'sampling_progress.json')
    require([row.get('index') for row in rows] == list(range(expected_count)),
            'Expected exactly31 completed prefix entries in order')
    directory = sample / 'sampling'
    files = list(directory.glob('sampled_graph_*.pkl'))
    wanted = {f'sampled_graph_{i}.pkl' for i in range(expected_count)}
    require({p.name for p in files} == wanted, 'Saved files do not match the31-graph prefix')
    result = []
    for row in rows:
        path = directory / f"sampled_graph_{row['index']}.pkl"
        require(path.is_file() and not path.is_symlink() and path.resolve().parent == directory,
                'Invalid prefix file: ' + str(path))
        require(sha(path) == row['sha256'], 'Prefix file differs from its saved audit')
        result.append({'index': row['index'], 'path': str(path), 'sha256': row['sha256'],
                       'nodes': row['nodes'], 'edges': row['edges']})
    return result


def gpu_info(index):
    text = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=index,uuid,name,driver_version,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=20)
    lines = [line for line in text.splitlines() if line.strip()]
    require(len(lines) == 1, 'Cannot identify the selected GPU')
    number, uuid, name, driver, free = [field.strip() for field in lines[0].split(',')]
    require(int(number) == index and uuid.startswith('GPU-'), 'Unexpected physical GPU')
    free = float(free)
    require(math.isfinite(free) and free >= 8192, 'Selected GPU has less than8GiB free memory')
    return {'index': index, 'uuid': uuid, 'name': name, 'driver_version': driver, 'free_mib': free}


def runtime_environment():
    import torch
    return {'python': sys.version, 'pytorch': str(torch.__version__),
            'pytorch_cuda_build': torch.version.cuda,
            'cuda_note': 'PyTorch build version; not nvidia-smi maximum supported CUDA version'}


def preflight(root, failed_run, gpu):
    root, failed_run = Path(root).resolve(), Path(failed_run).resolve()
    require(root == ROOT and sys.platform == 'linux', 'Use only the modified Linux server root')
    require(failed_run == root / 'outputs' / FAILED_NAME, 'Unexpected failed Ego run')
    require(gpu == 2, 'Replay must use physical GPU2 from the failed run; no utilization gate')
    package_hashes = verify_bundle()
    common, sampler = load_sampler(root)
    before = common.baseline(root, 'ego')
    before.update(package_hashes)
    state_path = failed_run / 'pipeline_status.json'
    state = read_json(state_path)
    require(state.get('dataset') == 'ego' and state.get('status') == 'failed'
            and state.get('phase') == 'sampling',
            'Old pipeline must be confirmed failed at sampling; no active handoff is attempted')
    failed_log = failed_run / 'sample.log'
    failure_text = failed_log.read_text(encoding='utf-8', errors='replace')[-18000:]
    require('linalg.eigh' in failure_text and 'failed to converge' in failure_text,
            'The failed run does not show the diagnosed eigh convergence failure')
    training = failed_run / 'train'
    require(Path(state['training_output']).resolve() == training, 'Wrong training output')
    item = common.get_item(root, 'ego')
    require(item['manifest_sha256'] == MANIFEST and item['splits']['test'] == TOTAL_COUNT,
            'Wrong frozen Ego manifest')
    audit_path = training / 'input_hash_check.json'
    audit = read_json(audit_path)
    require(audit['unchanged'] is True, 'Training group did not complete its input audit')
    common.check_hashes(audit['sha256'])
    before.update(audit['sha256'])
    reports = sampler.read_reports(training, 'ego', MANIFEST, 'formal')
    old_sample = root / 'higgs/outputs' / FAILED_NAME / 'sample'
    require(Path(state['sampling_output']).resolve() == old_sample, 'Wrong failed sample folder')
    old_manifest_path = old_sample / 'input_manifest.json'
    old_manifest = read_json(old_manifest_path)
    require(old_manifest['dataset'] == 'ego' and old_manifest['manifest_sha256'] == MANIFEST,
            'Failed sample identity differs')
    require(old_manifest['requested_samples'] == old_manifest['real_test_graphs'] == TOTAL_COUNT
            and old_manifest['exploratory'] is False, 'Not the formal151-graph series')
    require(Path(old_manifest['training_run']).resolve() == training, 'Failed series used different training')
    require(old_manifest['checkpoints'] == reports, 'Failed series checkpoint selection differs')
    require(old_manifest['protocol'] == 'seed1234 once; strict saved-config loading; full H2/H1 retained',
            'Failed series sampling policy differs')
    common.check_hashes(old_manifest['source_sha256'])
    before.update(old_manifest['source_sha256'])
    prefix = checked_prefix(old_sample)
    for record in prefix:
        before[record['path']] = record['sha256']
    for path in (state_path, failed_log, audit_path, old_manifest_path, old_sample / 'sampling_progress.json',
                 root / 'tools/qb_beta_sample.py', root / 'four_benchmarks_beta_20261101/evaluate.py',
                 Path(__file__).resolve(), PACKAGE / 'stable_eigen.py'):
        before[str(path)] = sha(path)
    for stage, report in reports.items():
        path = training / stage / 'formal_result.json'
        before[str(path)] = sha(path)
        before[report['best']] = report['best_sha256']
        before[report['last']] = report['last_sha256']
    check_hashes(before)
    selected_gpu = gpu_info(gpu)
    require(state.get('gpu', {}).get('index') == gpu
            and state.get('gpu', {}).get('uuid') == selected_gpu['uuid'],
            'Physical GPU identity differs from the failed run; replay refused')
    return {'root': str(root), 'failed_run': str(failed_run), 'training_output': str(training),
            'failed_sample': str(old_sample), 'manifest_sha256': MANIFEST,
            'reports': reports, 'prefix': prefix, 'before_sha256': before,
            'gpu': selected_gpu, 'system': runtime_environment(), 'policy': POLICY}


def exact_graph_equal(left, right):
    import networkx as nx
    require(isinstance(left, nx.Graph) and isinstance(right, nx.Graph), 'Expected NetworkX graphs')
    return nx.utils.graphs_equal(left, right)


def verify_replayed(index, generated_file, graph, prefix):
    require(0 <= index < len(prefix), 'Unexpected prefix comparison index')
    record = prefix[index]
    require(record['index'] == index and sha(record['path']) == record['sha256'], 'Old prefix changed')
    with Path(record['path']).open('rb') as handle:
        old = pickle.load(handle)  # Trusted PKL from this same experiment only.
    semantic_equal = exact_graph_equal(old, graph)
    generated_sha = sha(generated_file)
    require(semantic_equal and generated_sha == record['sha256'],
            f'Replay differs at graph{index + 1}; stopped without replacing the old graph')
    return {'index': index, 'semantic_equal': True, 'sha256_equal': True,
            'old_sha256': record['sha256'], 'replayed_sha256': generated_sha}


def save_rng_snapshot(torch, path, index):
    import random
    import numpy as np
    data = {'schema_version': 1, 'after_graph_index': index, 'next_graph_index': index + 1,
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda_all': torch.cuda.get_rng_state_all(),
            'policy': 'audit only; no snapshot resume entry implemented'}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(data, tmp)
    os.replace(tmp, path)


def run_sampling(request, output):
    root, output = Path(request['root']), Path(output)
    sample = output / 'sample'
    require(not sample.exists(), 'Never overwrite an existing recovery sample run')
    os.chdir(root)
    sys.path[:0] = [str(root / 'tools'), str(root / 'dgd'), str(root / 'higgs'), str(root)]
    import torch
    import sample_main  # Existing module changes cwd to root/higgs during import.
    os.chdir(root)
    import four_benchmark_models as models
    from stable_eigen import make_guarded_eigenfeatures

    original_model_class = models.model_class
    original_sample = sample_main.sample_large_graph
    replay_checks, numerical_events = [], []
    context = {'graph_index': None}

    def numerical_audit(event):
        event = dict(event)
        event['graph_index'] = context['graph_index']
        numerical_events.append(event)
        write_json(output / 'numerical_events.json', numerical_events)
        print('EIGH_NUMERICAL_EVENT:', json.dumps(event, default=str), flush=True)

    class RecoveryGlue(models.FourBenchmarkGlue):
        def __init__(self, cfg, dataset_infos, train_metrics=None, sampling_metrics=None,
                     visualization_tools=None, extra_features=None, domain_features=None):
            super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics,
                             visualization_tools, extra_features, domain_features)
            require(self.cfg.dataset.name == 'ego', 'Recovery numerical scope is Ego Glue only')
            self.extra_features.eigenfeatures = make_guarded_eigenfeatures(
                self.extra_features.eigenfeatures, numerical_audit,
                diagnostic_dir=output / 'diagnostics')

    def recovery_class(stage):
        return RecoveryGlue if stage == 'glue' else original_model_class(stage)

    def guarded_sample(*args, **kwargs):
        index = int(kwargs['graph_number'])
        require(index == len(completed), 'Unexpected graph order; RNG replay cannot be skipped')
        context['graph_index'] = index
        graph, largest = original_sample(*args, **kwargs)
        generated_file = Path.cwd() / f'sampled_graph_{index}.pkl'
        if index < len(request['prefix']):
            check = verify_replayed(index, generated_file, graph, request['prefix'])
            replay_checks.append(check)
            write_json(output / 'prefix_replay_audit.json', replay_checks)
            print('PREFIX_REPLAY_MATCH:', index + 1, '/', len(request['prefix']), flush=True)
        save_rng_snapshot(torch, sample / 'rng_states' / f'after_graph_{index:04d}.pt', index)
        completed.append(index)
        return graph, largest

    # Extra code files join the unchanged existing sampler's own hash audits.
    import qb_beta_common as common
    original_baseline = common.baseline
    def recovery_baseline(root_arg, dataset):
        result = dict(original_baseline(root_arg, dataset))
        result.update(request['before_sha256'])
        check_hashes(result)
        return result
    completed = []
    models.model_class = recovery_class
    sample_main.sample_large_graph = guarded_sample
    common.baseline = recovery_baseline
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(root / 'tools/qb_beta_sample.py'), '--root', str(root), '--dataset', 'ego',
                    '--training-output', request['training_output'], '--output', str(sample),
                    '--count', str(TOTAL_COUNT), '--report-mode', 'formal']
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = previous_argv
        common.baseline = original_baseline
        models.model_class = original_model_class
        sample_main.sample_large_graph = original_sample
    require(completed == list(range(TOTAL_COUNT)) and len(replay_checks) == PREFIX_COUNT,
            'Recovery sampler did not complete all151 plus31 exact prefix checks')
    write_json(output / 'recovery_sampling_complete.json', {
        'count': TOTAL_COUNT, 'prefix_verified_count': PREFIX_COUNT,
        'fallback_events': len(numerical_events), 'sample_output': str(sample),
        'rng_resume_implemented': False, 'numerical_policy': POLICY,
        'prefix_unchanged': True})


def table_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len({r['Metric'] for r in rows}) == len(rows), 'Duplicate metric rows')
    return {row['Metric']: row for row in rows}


def compare_existing(output):
    """Compare only complete, same-definition original results; no re-evaluation."""
    new = Path(output) / 'evaluation'
    pn = read_json(new / 'protocol.json')
    require(pn.get('dataset') == 'ego' and pn.get('count') == TOTAL_COUNT
            and pn.get('manifest_sha256') == MANIFEST, 'Wrong new evaluation')
    keys = ('dataset', 'protocol', 'manifest_sha256', 'test_graph_ids', 'component_policy',
            'evaluation_louvain', 'edgeless_community_policy', 'diameter_policy',
            'training_partitions_modified', 'learning_scope')
    matches = []
    for path in ORIGINAL.glob('outputs/small_benchmarks_tables_*/ego/protocol.json'):
        old = read_json(path)
        if (all(old.get(k) == pn.get(k) for k in keys)
                and len(old.get('samples_sha256', {})) == TOTAL_COUNT
                and all((path.parent / f).is_file() for f in
                        ('table1_mixed.csv', 'table3_summary.csv', 'table3_per_graph.csv'))):
            matches.append(path)
    if not matches:
        write_json(Path(output) / 'comparison_status.json', {'status': 'original_baseline_not_found'})
        print('COMPARISON_PENDING: matching original baseline not found; new metrics retained', flush=True)
        return
    old = sorted(matches)[-1].parent
    kernels = {'Nodes': 'Gaussian-TV', 'Degree': 'Gaussian-EMD', 'Clustering': 'Gaussian-EMD',
               'Eccentricity': 'Gaussian', 'Spectre': 'Gaussian-EMD'}
    a, b = table_rows(old / 'table1_mixed.csv'), table_rows(new / 'table1_mixed.csv')
    po = read_json(old / 'protocol.json')
    for table, protocol in ((a, po), (b, pn)):
        require(set(table) == set(kernels), 'Missing mixed metrics')
        recorded = {r['Metric']: r for r in protocol['table1']}
        require(set(recorded) == set(kernels), 'Protocol metrics differ')
        for metric, kernel in kernels.items():
            require(table[metric]['Kernel'] == recorded[metric]['Kernel'] == kernel,
                    'Wrong mixed kernel')
            require(math.isfinite(float(table[metric]['Value'])) and math.isclose(
                float(table[metric]['Value']), float(recorded[metric]['Value']), abs_tol=1e-12),
                'CSV/protocol metric values differ')
    c, d = table_rows(old / 'table3_summary.csv'), table_rows(new / 'table3_summary.csv')
    require(set(c) == set(d) and len(c) == 8, 'Table3 metric rows differ')
    for metric in c:
        require(math.isclose(float(c[metric]['RealMean']), float(d[metric]['RealMean']),
                            abs_tol=1e-9, rel_tol=1e-9), 'Real references differ')
        require(all(math.isfinite(float(row[field])) for row in (c[metric], d[metric])
                    for field in ('RealMean', 'GeneratedMean')), 'Nonfinite Table3 mean')
    rows1 = [{'Metric': key, 'OriginalHiGGs': float(a[key]['Value']),
              'BetaMeanHOGRecovery': float(b[key]['Value']), 'Kernel': kernels[key]} for key in kernels]
    rows3 = [{'Metric': key, 'RealMean': float(c[key]['RealMean']),
              'OriginalHiGGs': float(c[key]['GeneratedMean']),
              'BetaMeanHOGRecovery': float(d[key]['GeneratedMean'])} for key in c]
    for filename, rows in (('table1_original_vs_recovery.csv', rows1),
                           ('table3_original_vs_recovery_means.csv', rows3)):
        with (Path(output) / filename).open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(Path(output) / 'comparison_status.json', {'status': 'complete',
        'original': str(old), 'new': str(new), 'numerical_policy': POLICY,
        'limitation': 'whole-pipeline single run; originalseed0/H1LCC vs newseed1234/H1full; numericalfallback disclosed'})
    print('\nTABLE1: original HiGGs | Beta+mean+HOG recovery (lower better)', flush=True)
    for row in rows1:
        print(f"{row['Metric']:<20} {row['OriginalHiGGs']:.8f} | {row['BetaMeanHOGRecovery']:.8f}", flush=True)
    print('\nTABLE3 MEANS: real | original HiGGs | Beta+mean+HOG recovery', flush=True)
    for row in rows3:
        print(f"{row['Metric']:<20} {row['RealMean']:.6f} | {row['OriginalHiGGs']:.6f} | {row['BetaMeanHOGRecovery']:.6f}", flush=True)


def worker(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    require(root == ROOT and sys.platform == 'linux', 'Linux target only')
    require(output.parent == root / 'higgs/outputs' and not output.exists(), 'Use a fresh output')
    lock_path = root / 'logs/.four_benchmarks_beta_ego.lock'
    held, wanted = os.fstat(args.lock_fd), lock_path.stat()
    require((held.st_dev, held.st_ino) == (wanted.st_dev, wanted.st_ino), 'Wrong inherited lock')
    request = read_json(args.request)
    check_hashes(request['before_sha256'])
    require(request['root'] == str(root) and request['output'] == str(output), 'Wrong request identity')
    os.environ['CUDA_VISIBLE_DEVICES'] = request['gpu']['uuid']
    current_gpu = gpu_info(request['gpu']['index'])
    require(current_gpu['uuid'] == request['gpu']['uuid'], 'Selected GPU identity changed')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'runtime_environment.json', {'gpu': current_gpu, **runtime_environment()})
    write_json(output / 'recovery_numerical_policy.json', POLICY)
    state = {'status': 'running', 'phase': 'replay_and_sampling', 'count': TOTAL_COUNT,
             'old_prefix_count': PREFIX_COUNT, 'source_failed_run': request['failed_run'],
             'sampling_output': str(output / 'sample'), 'evaluation_output': str(output / 'evaluation')}
    try:
        write_json(output / 'recovery_status.json', state)
        run_sampling(request, output)
        state['phase'] = 'evaluation'
        write_json(output / 'recovery_status.json', state)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', MPLBACKEND='Agg', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')
        with (output / 'evaluation.log').open('xb') as handle:
            process = subprocess.Popen([sys.executable, '-u', '-B',
                str(root / 'four_benchmarks_beta_20261101/evaluate.py'), '--root', str(root),
                '--dataset', 'ego', '--samples-dir', str(output / 'sample/sampling'),
                '--output', str(output / 'evaluation')], cwd=root, env=env, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, pass_fds=(args.lock_fd,))
            (output / 'evaluation.pid').write_text(str(process.pid))
            require(process.wait() == 0, 'Evaluation failed; inspect evaluation.log; no sampling repeated')
        write_json(output / 'evaluation/recovery_numerical_policy.json', {
            'policy': POLICY, 'events': read_json(output / 'numerical_events.json')
                if (output / 'numerical_events.json').exists() else [],
            'source_sha256': {k: v for k, v in request['before_sha256'].items()
                              if str(PACKAGE) in k}})
        compare_existing(output)
        check_hashes(request['before_sha256'])
        state.update(status='complete', phase='complete', inputs_unchanged=True)
        write_json(output / 'input_hash_check.json', {'unchanged': True, 'sha256': request['before_sha256']})
        print('PASS: EGO_EIGH_RECOVERY_SAMPLING_AND_MIXED_EVALUATION_COMPLETE', flush=True)
        print('OUTPUT:', output, flush=True)
        print('Original31 graphs and all protected sources/checkpoints unchanged; no retraining.', flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(output / 'recovery_status.json', state)


def acquire_lock(root):
    import fcntl
    path = root / 'logs/.four_benchmarks_beta_ego.lock'
    require(path.is_file(), 'Original Ego pipeline lock is missing')
    lock = path.open('a+b')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Old Ego pipeline/coordinator still holds its lock; wait, do not kill or bypass it')
    sampler_path = root / 'logs/sampling/ego_mean_hog_fullchain.lock'
    require(sampler_path.is_file(), 'Old Ego sampler lock is missing')
    with sampler_path.open('r+b') as sampler_lock:
        try:
            fcntl.flock(sampler_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('An Ego sampler is active; recovery refused, use status')
        # Deliberately RELEASE this probe; qb sampler must lock it itself.
    return lock


def launch(args):
    root = Path(args.root).resolve()
    require(root == ROOT and sys.platform == 'linux', 'Launch only on the modified Linux server')
    lock = acquire_lock(root)
    try:
        request = preflight(root, Path(args.failed_run), args.gpu)
        if args.action == 'check':
            print('PASS: EGO_EIGH_RECOVERY_READ_ONLY_PREFLIGHT')
            print('PREFIX:', len(request['prefix']), 'GPU:', request['gpu'])
            print('No worker, sampling, source edits, or signals.')
            return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name = 'ego_mean_hog_eigh_recovery_' + stamp + '_gpu2'
        output = root / 'higgs/outputs' / name
        log = root / 'logs/sampling' / (name + '.log')
        req = log.with_suffix('.request.json')
        request['output'] = str(output)
        write_json(req, request)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=request['gpu']['uuid'],
                   PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MPLBACKEND='Agg', WANDB_MODE='disabled')
        with log.open('xb') as handle:
            process = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                'worker', '--root', str(root), '--output', str(output), '--request', str(req),
                '--lock-fd', str(lock.fileno())], cwd=root, env=env, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                pass_fds=(lock.fileno(),))
        Path(str(log) + '.pid').write_text(str(process.pid))
        Path(str(log) + '.out').write_text(str(output))
        info = {'pid': process.pid, 'log': str(log), 'output': str(output),
                'request': str(req), 'gpu': request['gpu'], 'failed_run': str(args.failed_run)}
        write_json(root / 'logs/ego_eigh_recovery_latest.json', info)
        print('PID=' + str(process.pid))
        print('LOG=' + str(log))
        print('OUTPUT=' + str(output))
        print('Submitted only: replay31 -> finish151 -> unchanged mixed evaluator -> comparison.')
        print('No training, no deletion, no old-pointer/status overwrite; do not launch twice.')
    finally:
        lock.close()


def status(args):
    root = Path(args.root).resolve()
    path = root / 'logs/ego_eigh_recovery_latest.json'
    if not path.is_file():
        print('No Ego recovery launch record')
        return
    info = read_json(path)
    print(json.dumps(info, indent=2))
    subprocess.run(['ps', '-p', str(info['pid']), '-o', 'pid,stat,etime,pcpu,pmem,args'], check=False)
    output = Path(info['output'])
    if (output / 'recovery_status.json').is_file():
        print((output / 'recovery_status.json').read_text())
    log = output / 'evaluation.log' if args.stage == 'evaluation' else Path(info['log'])
    if log.is_file():
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 12000))
            print('\n'.join(handle.read().decode('utf-8', errors='replace').replace('\r', '\n').splitlines()[-50:]))
    else:
        print('Stage log not created:', log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'launch', 'status', 'worker'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--failed-run', type=Path, default=ROOT / 'outputs' / FAILED_NAME)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--stage', choices=('sampling', 'evaluation'), default='sampling')
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--request', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == 'worker':
        require(args.output is not None and args.request is not None and args.lock_fd is not None,
                'Incomplete internal worker arguments')
        worker(args)
    elif args.action == 'status':
        status(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
