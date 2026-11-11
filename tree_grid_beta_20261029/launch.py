"""Linux-only Tree/Grid pipeline; rank by free VRAM, never by utilization."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
STAGES = ('h2', 'h1', 'glue')


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
    os.replace(tmp, path)


def common(root):
    sys.path.insert(0, str(root / 'tools'))
    import tg_beta_common
    return tg_beta_common


def gpu_inventory():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free = [field.strip() for field in line.split(',')]
        rows.append({'index': int(index), 'uuid': uuid, 'free_mib': float(free)})
    return sorted(rows, key=lambda row: (-row['free_mib'], row['index']))


def select_gpus(names, rows):
    available = sorted((r for r in rows if r['free_mib'] >= 8192),
                       key=lambda row: (-row['free_mib'], row['index']))
    if len(available) < len(names):
        raise RuntimeError('Need one distinct GPU per dataset with >=8192 MiB free; no jobs stopped')
    return dict(zip(names, available))


def read_reports(root, name, output, mode):
    c = common(root)
    item = c.get_item(root, name)
    reports = {}
    for stage in STAGES:
        path = output / stage / (mode + '_result.json')
        report = json.loads(path.read_text())
        assert report['dataset'] == name and report['stage'] == stage
        assert report['epoch'] == (0 if mode == 'smoke' else 499)
        assert report['manifest_sha256'] == item['manifest_sha256']
        assert report['strict_reload'] and report['strict_best_reload'] and report['inputs_unchanged']
        assert report['train_seed'] == 1234 and report['batch_size'] == 32
        assert report['monitor'] == c.STAGES[stage][2]
        assert report['diffusion_steps'] == (400 if stage == 'glue' else 100)
        for key in ('best', 'last'):
            assert Path(report[key]).resolve().is_relative_to(output / stage)
            assert c.sha(report[key]) == report[key + '_sha256']
        if stage == 'h1':
            assert report['two_hop']['normalization'] == 'mean'
            assert report['two_hop']['warm_start'] is None and report['two_hop']['channels'] == 16
            assert report['hog']['rw_depth'] == 4 and report['hog']['two_hop'] is True
            assert report['input_widths'] == {'X': 12, 'E': 6, 'y': 14}
        reports[stage] = report
    return reports


def run_child(command, log, pid_path, root, lock_fd, env=None):
    with log.open('xb') as handle:
        p = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, pass_fds=(lock_fd,))
        pid_path.write_text(str(p.pid))
        print('START PID={} LOG={}'.format(p.pid, log), flush=True)
        rc = p.wait()
    if rc:
        raise RuntimeError('Child exit {}: {}'.format(rc, log))


def train_group(args, out, mode, before):
    root = Path(args.root).resolve()
    folder = out / ('smoke' if mode == 'smoke' else 'train')
    folder.mkdir()
    jobs, failures = {}, {}
    for stage in STAGES:
        command = [sys.executable, '-u', '-B', str(root / 'tools/tg_beta_train.py'),
                   '--root', str(root), '--dataset', args.dataset, '--stage', stage,
                   '--mode', 'smoke' if mode == 'smoke' else 'train',
                   '--output', str(folder / stage)]
        log = folder / (stage + '.log')
        if mode == 'smoke':
            run_child(command, log, folder / (stage + '.pid'), root, args.lock_fd)
        else:
            try:
                with log.open('xb') as handle:
                    p = subprocess.Popen(command, cwd=root, stdout=handle,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        pass_fds=(args.lock_fd,))
                (folder / (stage + '.pid')).write_text(str(p.pid))
                jobs[stage] = p
                print('TRAIN_START {} PID={} LOG={}'.format(stage, p.pid, log), flush=True)
            except OSError as error:
                failures[stage] = repr(error)
    # Do not signal siblings if one fails: let already-running stages finish.
    while jobs:
        for stage, p in list(jobs.items()):
            rc = p.poll()
            if rc is not None:
                print('FINISHED {} exit={}'.format(stage, rc), flush=True)
                if rc:
                    failures[stage] = rc
                del jobs[stage]
        if jobs:
            time.sleep(1)
    if failures:
        raise RuntimeError('Training failed: {}; no sampling; no jobs signaled'.format(failures))
    reports = read_reports(root, args.dataset, folder, mode)
    common(root).check_hashes(before)
    write_json(folder / 'input_hash_check.json', {'unchanged': True, 'sha256': before})
    print('PASS: {}_THREE_STAGE_{}'.format(args.dataset.upper(), mode.upper()), flush=True)
    return folder, reports


def worker(args):
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    assert out.parent == root / 'outputs' and not out.exists()
    expected = (root / 'logs' / ('.tree_grid_beta_' + args.dataset + '.lock')).stat()
    inherited = os.fstat(args.lock_fd)
    assert (expected.st_dev, expected.st_ino) == (inherited.st_dev, inherited.st_ino)
    request = json.loads(Path(args.request).read_text())
    assert request['launcher_sha256'] == common(root).sha(__file__)
    before = common(root).baseline(root, args.dataset)
    before[str(Path(__file__).resolve())] = common(root).sha(__file__)
    before[str(PACKAGE / 'evaluate.py')] = common(root).sha(PACKAGE / 'evaluate.py')
    out.mkdir(parents=True)
    item = common(root).get_item(root, args.dataset)
    state = {'dataset': args.dataset, 'status': 'running', 'phase': 'smoke',
             'gpu': request['gpu'], 'manifest_sha256': item['manifest_sha256'],
             'training_output': str(out / 'train')}
    write_json(out / 'pipeline_status.json', state)
    try:
        smoke, _ = train_group(args, out, 'smoke', before)
        command = [sys.executable, '-u', '-B', str(root / 'tools/tg_beta_sample.py'),
            '--root', str(root), '--dataset', args.dataset, '--training-output', str(smoke),
            '--output', str(out / 'smoke_sample'), '--count', '1', '--report-mode', 'smoke']
        state['phase'] = 'smoke_sample'
        write_json(out / 'pipeline_status.json', state)
        run_child(command, out / 'smoke_sample.log', out / 'smoke_sample.pid', root, args.lock_fd)
        smoke_audit = json.loads((out / 'smoke_sample/input_hash_check.json').read_text())
        assert smoke_audit['unchanged'] is True
        print('PASS: {}_REAL_TRAIN_RELOAD_AND_FULLCHAIN_SMOKE'.format(args.dataset.upper()), flush=True)
        state['phase'] = 'formal_training'
        write_json(out / 'pipeline_status.json', state)
        training, reports = train_group(args, out, 'formal', before)
        state['phase'] = 'sampling'
        sample = root / 'higgs/outputs' / out.name
        assert not sample.exists()
        sample.mkdir(parents=True)
        state['sampling_output'] = str(sample / 'sample')
        state['evaluation_output'] = str(sample / 'evaluation')
        write_json(out / 'pipeline_status.json', state)
        command = [sys.executable, '-u', '-B', str(root / 'tools/tg_beta_sample.py'),
            '--root', str(root), '--dataset', args.dataset, '--training-output', str(training),
            '--output', str(sample / 'sample'), '--count', str(item['splits']['test']),
            '--report-mode', 'formal']
        run_child(command, out / 'sample.log', out / 'sample.pid', root, args.lock_fd)
        state['phase'] = 'evaluation'
        write_json(out / 'pipeline_status.json', state)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        command = [sys.executable, '-u', '-B', str(PACKAGE / 'evaluate.py'),
            '--root', str(root), '--dataset', args.dataset,
            '--samples-dir', str(sample / 'sample/sampling'),
            '--output', str(sample / 'evaluation')]
        run_child(command, out / 'evaluation.log', out / 'evaluation.pid', root, args.lock_fd, env)
        common(root).check_hashes(before)
        state.update(status='complete', phase='complete', inputs_unchanged=True)
        write_json(out / 'input_hash_check.json', {'unchanged': True, 'sha256': before})
        print('PASS: {}_BETA_MEAN_HOG_TRAIN_SAMPLE_EVALUATE_COMPLETE'.format(args.dataset.upper()), flush=True)
        print('EVALUATION:', sample / 'evaluation', flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(out / 'pipeline_status.json', state)


def launch(args):
    import fcntl
    root = Path(args.root).resolve()
    assert sys.platform == 'linux' and root == ROOT
    names = ('tree', 'grid') if args.dataset == 'both' else (args.dataset,)
    c = common(root)
    for name in names:
        c.baseline(root, name)
    log_dir = root / 'logs/training'
    log_dir.mkdir(parents=True, exist_ok=True)
    locks = {}
    try:
        # Acquire every requested dataset lock before starting any job.
        for name in names:
            lock = (root / 'logs' / ('.tree_grid_beta_' + name + '.lock')).open('a+b')
            locks[name] = lock
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(name + ' pipeline is already running; use status, do not relaunch')
        rows = gpu_inventory()
        plans = select_gpus(names, rows)
        print('GPU free-memory ranking:', rows, flush=True)
        print('No utilization/exclusivity restriction; existing jobs are untouched.', flush=True)
        for name in names:
            gpu = plans[name]
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            tag = '{}_beta_mean_hog_r{}_pipeline_{}_gpu{}'.format(
                name, c.SPECS[name]['resolution'], stamp, gpu['index'])
            out, log = root / 'outputs' / tag, log_dir / (tag + '.log')
            request_path = log_dir / (tag + '.request.json')
            write_json(request_path, {'dataset': name, 'gpu': gpu,
                'launcher_sha256': c.sha(__file__), 'root': str(root), 'output': str(out)})
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], WANDB_MODE='disabled', MPLBACKEND='Agg',
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='1234',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), 'pipeline',
                '--root', str(root), '--dataset', name, '--worker', '--output', str(out),
                '--request', str(request_path), '--lock-fd', str(locks[name].fileno())]
            with log.open('xb') as handle:
                p = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                    pass_fds=(locks[name].fileno(),))
            Path(str(log) + '.pid').write_text(str(p.pid))
            Path(str(log) + '.out').write_text(str(out))
            info = {'dataset': name, 'pid': p.pid, 'log': str(log), 'output': str(out), 'gpu': gpu}
            write_json(root / 'logs' / (name + '_tg_beta_latest_pipeline.json'), info)
            print(json.dumps(info, indent=2), flush=True)
        print('SUBMITTED ONLY: smoke -> fresh500x3 -> validation-best sample -> Table1/3.', flush=True)
        print('Do not launch again to view progress.', flush=True)
    finally:
        for lock in locks.values():
            lock.close()


def status(args):
    root = Path(args.root).resolve()
    names = ('tree', 'grid') if args.dataset == 'both' else (args.dataset,)
    for name in names:
        info_path = root / 'logs' / (name + '_tg_beta_latest_pipeline.json')
        print('\n===== {} ====='.format(name.upper()), flush=True)
        if not info_path.is_file():
            print('No launch record')
            continue
        info = json.loads(info_path.read_text())
        print(json.dumps(info, indent=2))
        state_path = Path(info['output']) / 'pipeline_status.json'
        if state_path.is_file():
            print(state_path.read_text())
        subprocess.run(['ps', '-p', str(info['pid']), '-o', 'pid,stat,etime,pcpu,pmem,args'], check=False)
        stage = args.stage
        log = Path(info['log']) if stage == 'pipeline' else Path(info['output']) / (
            'train/' + stage + '.log' if stage in STAGES else stage + '.log')
        if log.is_file():
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 8000))
                lines = handle.read().decode('utf-8', errors='replace').replace('\r', '\n').splitlines()
            print('\n'.join(lines[-35:]))
        else:
            print('Stage has not started:', log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pipeline', 'status'))
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--dataset', choices=('tree', 'grid', 'both'), default='both')
    parser.add_argument('--stage', choices=('pipeline', 'h2', 'h1', 'glue', 'smoke_sample', 'sample', 'evaluation'), default='pipeline')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    parser.add_argument('--request', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        assert Path(args.root).resolve() == ROOT and args.dataset != 'both'
        assert args.lock_fd is not None and args.output and args.request
        worker(args)
    elif args.action == 'status':
        status(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
