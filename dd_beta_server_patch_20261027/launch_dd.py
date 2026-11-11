"""Detached DD jobs. Memory-only GPU checks; no process termination or utilization gate."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_memory(gpus):
    for gpu in sorted(set(gpus)):
        free = int(subprocess.check_output([
            'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        print(f'GPU {gpu}: free memory = {free} MiB', flush=True)
        if free < 8192:
            raise RuntimeError('Less than 8 GiB free; no job launched')
    print('PASS: memory check; no utilization restriction', flush=True)


def completed(root, out, mode):
    out = Path(out).resolve()
    assert out.parent == root / 'outputs'
    audit = json.loads((out / 'input_hash_check.json').read_text())
    assert audit['unchanged'] is True
    assert all(sha(path) == value for path, value in audit['sha256'].items()), 'Inputs/source changed'
    for key in ('h2', 'h1', 'glue'):
        report = json.loads((out / key / f'{mode}_result.json').read_text())
        assert report['strict_reload'] and report['strict_best_reload']
        assert report['epoch'] == (0 if mode == 'smoke' else 499)
        assert sha(report['best']) == report['best_sha256']
        assert sha(report['last']) == report['last_sha256']
    return out


def latest_output(root, mode):
    return Path(json.loads((root / f'logs/dd_latest_{mode}.json').read_text())['output'])


def launch(root, mode, *, gpu=0, gpu_a=0, gpu_b=1, count=184, previous=None):
    assert root == ROOT, 'Only the modified Linux checkout is allowed'
    assert sys.platform == 'linux', 'Run on the Linux server, not the Windows checkout'
    assert sys.flags.optimize == 0, 'Do not use python -O: safety assertions must stay enabled'
    setup = json.loads((root / 'data_sources/splits/dd_beta_setup_20261027.json').read_text())
    assert sha(root / 'dgd/main.py') == setup['main_sha256']
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               MPLBACKEND='Agg', WANDB_MODE='disabled', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if mode == 'smoke':
        check_memory([gpu])
        name = f'dd_beta_mean_hog_smoke_seed0_{stamp}_gpu{gpu}'
        out = root / 'outputs' / name
        log_dir = root / 'logs/diagnostics'
        args = [str(root), str(out), 'all']
        driver = 'dd_beta_smoke_20261027.py'
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    elif mode == 'train':
        assert gpu_a != gpu_b, 'Choose two physical GPUs for the two parallel groups'
        prior = completed(root, previous or latest_output(root, 'smoke'), 'smoke')
        check_memory([gpu_a, gpu_b])
        name = f'dd_beta_mean_hog_r2_e500_scratch_seed0_{stamp}'
        out = root / 'outputs' / name
        log_dir = root / 'logs/training'
        args = [str(root), str(out), 'all']
        driver = 'dd_beta_formal_20261027.py'
        env.update(CUDA_VISIBLE_DEVICES='', DD_TRAIN_GPU_A=str(gpu_a), DD_TRAIN_GPU_B=str(gpu_b))
        print('SMOKE_OUTPUT:', prior, flush=True)
        print(f'GPU {gpu_a}: H2 -> Glue; GPU {gpu_b}: H1; 500 epochs, batch4, seed0', flush=True)
    elif mode == 'sample':
        prior = completed(root, previous or latest_output(root, 'train'), 'formal')
        check_memory([gpu])
        name = f'dd_mean_hog_fullchain_seed1234_n{count}_{stamp}_gpu{gpu}'
        out = root / 'higgs/outputs' / name
        log_dir = root / 'logs/sampling'
        args = [str(root), str(prior), str(out), str(count)]
        driver = 'dd_beta_sample_20261027.py'
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        print('TRAINING_OUTPUT:', prior, flush=True)
        print('EXPLORATORY:', count != 184, flush=True)
    else:
        raise ValueError(mode)
    script = root / 'tools/diagnostics' / driver
    assert script.is_file() and not out.exists()
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / (name + '.log')
    with log.open('x') as handle:
        child = subprocess.Popen([sys.executable, '-u', '-B', str(script), *args],
                                 cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    Path(str(log) + '.pid').write_text(str(child.pid) + '\n')
    Path(str(log) + '.out').write_text(str(out) + '\n')
    (root / f'logs/dd_latest_{mode}.json').write_text(json.dumps({
        'pid': child.pid, 'log': str(log), 'output': str(out), 'mode': mode,
        'manifest_sha256': setup['manifest_sha256']}, indent=2))
    print(f'PID={child.pid}\nLOG={log}\nOUTPUT={out}', flush=True)
    print('Do not launch this stage twice. Completion must be checked in its log.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    sub = parser.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('smoke')
    s.add_argument('--gpu', type=int, default=0)
    t = sub.add_parser('train')
    t.add_argument('--gpu-a', type=int, default=0)
    t.add_argument('--gpu-b', type=int, default=1)
    t.add_argument('--smoke-output')
    p = sub.add_parser('sample')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--count', type=int, choices=(1, 184), default=184)
    p.add_argument('--training-output')
    args = parser.parse_args()
    launch(Path(args.root).resolve(), args.mode, gpu=getattr(args, 'gpu', 0),
           gpu_a=getattr(args, 'gpu_a', 0), gpu_b=getattr(args, 'gpu_b', 1),
           count=getattr(args, 'count', 184),
           previous=getattr(args, 'smoke_output', None) or getattr(args, 'training_output', None))


if __name__ == '__main__':
    main()
