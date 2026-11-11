"""Detached Linux launch/status, no utilization gate and no process signals."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
DATASETS = ('community_small', 'enzymes')
STAGES = ('h2', 'h1', 'glue')


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=2)
    os.replace(temporary, path)


def memory_check(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert len(row) == 2
    uuid, free = row[0].strip(), float(row[1])
    if free < 8192:
        raise RuntimeError(f'GPU {gpu}: {free:g} MiB free, requires >=8192; no jobs stopped')
    print(f'GPU {gpu}: {free:g} MiB free; no utilization/exclusive restriction', flush=True)
    return uuid


def common(root):
    sys.path.insert(0, str(root / 'tools'))
    import sc_beta_common
    return sc_beta_common


def validate_reports(root, name, output, mode):
    c = common(root)
    item = c.get_item(root, name)
    for stage in STAGES:
        report = json.loads((output / stage / (mode + '_result.json')).read_text())
        assert report['stage'] == stage and report['epoch'] == (0 if mode == 'smoke' else 499)
        assert report['manifest_sha256'] == item['manifest_sha256']
        assert report['strict_reload'] and report['strict_best_reload']
        assert report['global_step'] > 0
        assert report['monitor'] == c.STAGES[stage][2]
        for kind in ('best', 'last'):
            assert Path(report[kind]).resolve().is_relative_to(output / stage)
            assert c.sha(report[kind]) == report[kind + '_sha256']
        if stage == 'h1':
            assert report['two_hop']['normalization'] == 'mean'
            assert report['two_hop']['warm_start'] is None
            assert report['hog']['rw_depth'] == 4 and report['hog']['two_hop']
            assert report['input_widths'] == {'X': 12, 'E': 6, 'y': 14}


def verified_training(root, name, output, mode):
    audit = json.loads((output / 'input_hash_check.json').read_text())
    assert audit['unchanged'] is True
    common(root).check_hashes(audit['sha256'])
    validate_reports(root, name, output, mode)


def worker(args):
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    c = common(root)
    item = c.get_item(root, args.dataset)
    expected_lock = (root / f'logs/.{args.dataset}_small_beta.lock').stat()
    inherited = os.fstat(args.lock_fd)
    assert (expected_lock.st_dev, expected_lock.st_ino) == (inherited.st_dev, inherited.st_ino)
    before = c.baseline(root, args.dataset)
    before[str(Path(__file__).resolve())] = c.sha(__file__)
    assert not out.exists()
    out.mkdir(parents=True)
    write_json(out / 'request.json', vars(args))
    if args.action == 'sample':
        training = Path(args.training_output).resolve()
        verified_training(root, args.dataset, training, 'formal')
        command = [sys.executable, '-u', '-B', str(root / 'tools/sc_beta_sample.py'),
                   '--root', str(root), '--dataset', args.dataset,
                   '--training-output', str(training), '--output', str(out / 'sample'),
                   '--count', str(args.count or item['splits']['test'])]
        # Parent owns the run directory; sample script owns its dedicated subdirectory.
        run_child(command, out / 'sample.log', out / 'sample.pid', root, args.lock_fd)
        c.check_hashes(before)
        write_json(out / 'input_hash_check.json', {'unchanged': True, 'sha256': before})
        print(f'PASS: {args.dataset.upper()}_BETA_MEAN_HOG_SAMPLING_COMPLETE', flush=True)
        return
    mode = 'smoke' if args.action == 'smoke' else 'train'
    if mode == 'train':
        verified_training(root, args.dataset, Path(args.smoke_output).resolve(), 'smoke')
    jobs, failures = {}, {}
    for stage in STAGES:
        command = [sys.executable, '-u', '-B', str(root / 'tools/sc_beta_train.py'),
                   '--root', str(root), '--dataset', args.dataset, '--stage', stage,
                   '--output', str(out / stage), '--mode', mode]
        log = out / f'{stage}.log'
        if mode == 'smoke':
            run_child(command, log, out / f'{stage}.pid', root, args.lock_fd)
        else:
            try:
                with log.open('xb') as handle:
                    p = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, pass_fds=(args.lock_fd,))
                (out / f'{stage}.pid').write_text(str(p.pid))
                jobs[stage] = p
                print(f'START {stage} PID={p.pid} LOG={log}', flush=True)
            except OSError as error:
                failures[stage] = repr(error)
    while jobs:
        for stage, p in list(jobs.items()):
            rc = p.poll()
            if rc is not None:
                print(f'FINISHED {stage} exit={rc}', flush=True)
                if rc:
                    failures[stage] = rc
                del jobs[stage]
        if jobs:
            time.sleep(1)
    if failures:
        raise RuntimeError(f'Training failure: {failures}; remaining jobs were not signaled; no sampling')
    validate_reports(root, args.dataset, out, 'smoke' if mode == 'smoke' else 'formal')
    c.check_hashes(before)
    write_json(out / 'input_hash_check.json', {'unchanged': True, 'sha256': before})
    print(f'PASS: {args.dataset.upper()}_BETA_MEAN_HOG_THREE_STAGE_{"SMOKE" if mode == "smoke" else "FORMAL_TRAINING_COMPLETE"}', flush=True)
    print('OUTPUT:', out, flush=True)
    print('No whole-graph sampling or quality evaluation was performed.', flush=True)


def run_child(command, log, pid_path, root, lock_fd):
    with log.open('xb') as handle:
        p = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, pass_fds=(lock_fd,))
        pid_path.write_text(str(p.pid))
        print(f'START PID={p.pid} LOG={log}', flush=True)
        if p.wait():
            raise RuntimeError('Child failed; inspect ' + str(log))


def launch(args):
    import fcntl
    root = Path(args.root).resolve()
    assert sys.platform == 'linux' and root == ROOT
    c = common(root)
    c.baseline(root, args.dataset)
    if args.action == 'train':
        assert args.smoke_output, 'Pass the successful smoke output explicitly'
        verified_training(root, args.dataset, Path(args.smoke_output).resolve(), 'smoke')
    if args.action == 'sample':
        assert args.training_output, 'Pass the successful formal training output explicitly'
        verified_training(root, args.dataset, Path(args.training_output).resolve(), 'formal')
        assert args.count in (None, 1, c.SPECS[args.dataset]['splits']['test'])
    gpu_uuid = memory_check(args.gpu)
    log_dir = root / ('logs/diagnostics' if args.action == 'smoke' else
                      'logs/training' if args.action == 'train' else 'logs/sampling')
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (root / f'logs/.{args.dataset}_small_beta.lock').open('a+b')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name = f'{args.dataset}_beta_mean_hog_r1_{args.action}_{stamp}_gpu{args.gpu}'
        out = root / ('higgs/outputs' if args.action == 'sample' else 'outputs') / name
        log = log_dir / (name + '.log')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu_uuid, WANDB_MODE='disabled', MPLBACKEND='Agg',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='1234',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), args.action,
                   '--root', str(root), '--dataset', args.dataset, '--gpu', str(args.gpu),
                   '--worker', '--output', str(out), '--lock-fd', str(lock.fileno())]
        for flag, value in (('--smoke-output', args.smoke_output), ('--training-output', args.training_output),
                            ('--count', args.count)):
            if value is not None:
                command += [flag, str(value)]
        with log.open('xb') as handle:
            p = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(lock.fileno(),))
        Path(str(log) + '.pid').write_text(str(p.pid))
        Path(str(log) + '.out').write_text(str(out))
        write_json(root / f'logs/{args.dataset}_beta_latest_{args.action}.json',
                   {'pid': p.pid, 'log': str(log), 'output': str(out)})
        print(f'DATASET={args.dataset}\nPID={p.pid}\nLOG={log}\nOUTPUT={out}')
        print('Submitted only. Check completion marker; do not launch twice.')
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'train', 'sample'))
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke-output')
    parser.add_argument('--training-output')
    parser.add_argument('--count', type=int)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        assert Path(args.root).resolve() == ROOT and args.output and args.lock_fd is not None
        parent = ROOT / ('higgs/outputs' if args.action == 'sample' else 'outputs')
        assert Path(args.output).resolve().parent == parent
        worker(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
