"""Runtime-only Boolean Glue-mask fix; preserve all original audited files."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
LEGACY_SHA = '53f74598832d5b2f1eb88858b0a5dbc29a88a11cd9f5c949f2d42a6afc17865e'
DATASETS = ('community_small', 'enzymes')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temp, path)


def restore_boolean_masks(result, stats=None):
    """Keep all mask values/shapes/devices; only restore bool indexing dtype.

    The old loader stores this same list of Data objects, so replacing each
    object's pos attribute also fixes the later collated PyG batch.
    No RNG, graph, y, edge, weight or cache is changed.
    """
    import torch
    samples, loader = result[:2]
    for data in samples:
        mask = data.pos
        if not torch.is_tensor(mask) or mask.ndim != 2:
            raise RuntimeError('Glue pos must be a two-dimensional tensor')
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise RuntimeError('Glue mask is not finite binary data; refusing coercion')
        converted = mask.to(dtype=torch.bool)
        if not torch.equal(mask, converted.to(dtype=mask.dtype)):
            raise RuntimeError('Boolean conversion would change mask values')
        data.pos = converted
        if stats is not None:
            stats['pairs_checked'] += 1
            if mask.dtype != torch.bool:
                stats['pairs_converted'] += 1
    return result


def preflight(root, name, training):
    assert root == ROOT and name in DATASETS
    assert training.parent == root / 'outputs'
    sys.path.insert(0, str(root / 'tools'))
    import sc_beta_common as c
    item = c.get_item(root, name)
    before = c.baseline(root, name)
    audit_path = training / 'input_hash_check.json'
    audit = json.loads(audit_path.read_text())
    assert audit['unchanged'] is True, 'Formal training audit has not completed'
    c.check_hashes(audit['sha256'])
    before.update(audit['sha256'])
    before[str(audit_path)] = sha(audit_path)
    source = root / 'tools/sc_beta_sample.py'
    assert sha(source) == LEGACY_SHA, 'Unknown original sampling adapter; do not bypass this check'
    spec = importlib.util.spec_from_file_location('sc_beta_original_sampling', source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    reports = legacy.read_reports(training, name, item['manifest_sha256'])
    for stage, report in reports.items():
        assert report.get('dataset') == name
        assert report.get('batch_size') == 32 and report.get('train_seed') == 1234
        for key in ('best', 'last'):
            path = Path(report[key]).resolve()
            assert path.is_relative_to(training / stage)
            assert sha(path) == report[key + '_sha256']
            before[str(path)] = report[key + '_sha256']
        report_path = training / stage / 'formal_result.json'
        before[str(report_path)] = sha(report_path)
    return c, item, before, legacy


def worker(args):
    root = Path(args.root).resolve()
    training, out = Path(args.training_output).resolve(), Path(args.output).resolve()
    assert out.parent == root / 'higgs/outputs'
    lock_stat = (root / f'logs/.{args.dataset}_small_beta.lock').stat()
    fd_stat = os.fstat(args.lock_fd)
    assert (lock_stat.st_dev, lock_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino)
    request = json.loads((out / 'boolfix_request.json').read_text())
    assert request['wrapper_sha256'] == sha(__file__), 'Repair wrapper changed after submission'
    c, item, before, legacy = preflight(root, args.dataset, training)
    before[str(Path(__file__).resolve())] = sha(__file__)
    stats = {'pairs_checked': 0, 'pairs_converted': 0}
    original_prepare = legacy.prepare_pairs

    def fixed_prepare(*a, **kw):
        previous = stats['pairs_checked']
        result = restore_boolean_masks(original_prepare(*a, **kw), stats)
        if result[0] and previous == 0:
            print('PASS: GLUE_POS_BOOL_RUNTIME; mask values and membership unchanged', flush=True)
        return result

    legacy.prepare_pairs = fixed_prepare
    record = {
        'fix': 'runtime-only binary float pos -> bool; membership values unchanged',
        'wrapper_sha256': sha(__file__), 'original_sampler_sha256': LEGACY_SHA,
        'training_output': str(training), 'dataset': args.dataset,
        'shared_files_modified': False, 'audit_records_modified': False,
        'seed': 1234, 'requested_samples': item['splits']['test'],
        'input_sha256': before, 'stats': stats, 'status': 'running',
    }
    write_json(out / 'boolfix_audit.json', record)
    print('PASS: ORIGINAL_TRAINING_AND_SOURCE_AUDITS_UNCHANGED', flush=True)
    print('FIX: runtime Boolean Glue masks only; original sampling script is unchanged', flush=True)
    sys.argv = [str(root / 'tools/sc_beta_sample.py'), '--root', str(root),
                '--dataset', args.dataset, '--training-output', str(training),
                '--output', str(out / 'sample'), '--count', str(item['splits']['test'])]
    try:
        legacy.main()
        c.check_hashes(before)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        write_json(out / 'boolfix_audit.json', record)
        raise
    record['status'] = 'complete'
    record['inputs_unchanged'] = True
    write_json(out / 'boolfix_audit.json', record)
    print(f'PASS: {args.dataset.upper()}_BOOLMASK_FIX_FULLCHAIN_COMPLETE', flush=True)
    print('OUTPUT:', out, flush=True)
    print('No retraining, repartitioning or generation-quality evaluation.', flush=True)


def launch(args):
    import fcntl
    root = Path(args.root).resolve()
    assert sys.platform == 'linux' and root == ROOT
    training = Path(args.training_output).resolve()
    c, item, before, _ = preflight(root, args.dataset, training)
    lock = (root / f'logs/.{args.dataset}_small_beta.lock').open('a+b')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        uuid, free = row[0].strip(), float(row[1])
        assert free >= 8192, 'Less than8GiB free; no process was stopped'
        print(f'GPU {args.gpu}: {free:g} MiB free; no utilization restriction')
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name = f'{args.dataset}_beta_mean_hog_r1_sample_boolfix_{stamp}_gpu{args.gpu}'
        out = root / 'higgs/outputs' / name
        out.mkdir(parents=True, exist_ok=False)
        request = {'dataset': args.dataset, 'training_output': str(training),
                   'wrapper_sha256': sha(__file__), 'original_sampler_sha256': LEGACY_SHA,
                   'gpu': args.gpu, 'gpu_uuid': uuid, 'count': item['splits']['test']}
        write_json(out / 'boolfix_request.json', request)
        log_dir = root / 'logs/sampling'
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / (name + '.log')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=uuid, WANDB_MODE='disabled', MPLBACKEND='Agg',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='1234',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                   '--root', str(root), '--dataset', args.dataset, '--gpu', str(args.gpu),
                   '--training-output', str(training), '--worker', '--output', str(out),
                   '--lock-fd', str(lock.fileno())]
        with log.open('xb') as handle:
            p = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, pass_fds=(lock.fileno(),))
        Path(str(log) + '.pid').write_text(str(p.pid))
        Path(str(log) + '.out').write_text(str(out))
        write_json(root / f'logs/{args.dataset}_beta_latest_sample.json',
                   {'pid': p.pid, 'log': str(log), 'output': str(out), 'runtime_fix': True})
        print(f'PID={p.pid}\nLOG={log}\nOUTPUT={out}')
        print('Submitted only. Do not launch twice; inspect the sampling log.')
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--training-output', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        assert args.output and args.lock_fd is not None
        worker(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
