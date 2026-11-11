"""Linux-only three-TU-collection pipeline; rank by free VRAM, never by utilization."""
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
    import tu_beta_common
    return tu_beta_common


def gpu_inventory():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free = [field.strip() for field in line.split(',')]
        rows.append({'index': int(index), 'uuid': uuid, 'free_mib': float(free)})
    return sorted(rows, key=lambda row: (-row['free_mib'], row['index']))


def read_reports(root, name, output, mode):
    c = common(root)
    item = c.get_item(root, name)
    reports = {}
    for stage in STAGES:
        path = output / stage / (mode + '_result.json')
        report = json.loads(path.read_text())
        assert report['dataset'] == name and report['stage'] == stage
        assert report['epoch'] == (0 if mode == 'smoke' else c.SPECS[name]['epochs'] - 1)
        assert report['manifest_sha256'] == item['manifest_sha256']
        assert report['strict_reload'] and report['strict_best_reload'] and report['inputs_unchanged']
        assert report['train_seed'] == 0 and report['batch_size'] == c.SPECS[name]['batch_size'][stage]
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
        command = [sys.executable, '-u', '-B', str(root / 'tools/tu_beta_train.py'),
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
    expected = (root / 'logs' / ('.tu_benchmarks_beta_' + args.dataset + '.lock')).stat()
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
        command = [sys.executable, '-u', '-B', str(root / 'tools/tu_beta_sample.py'),
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
        command = [sys.executable, '-u', '-B', str(root / 'tools/tu_beta_sample.py'),
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


def names_for(value):
    return tuple(common(ROOT).SPECS) if value == 'all' else (value,)


def candidates(rows, reserved=(), allowed=None, minimum=8192):
    """Rank memory only; existing jobs may share the selected GPU."""
    return sorted((r for r in rows if r['index'] not in reserved
                   and (allowed is None or r['index'] in allowed)
                   and r['free_mib'] >= minimum),
                  key=lambda r: (-r['free_mib'], r['index']))


def pointer(root, name, value):
    write_json(root / 'logs' / (name + '_tu_beta_latest_pipeline.json'), value)


def coordinator(args):
    root = Path(args.root).resolve()
    request = json.loads(Path(args.request).read_text())
    if request['launcher_sha256'] != common(root).sha(__file__):
        raise RuntimeError('Launcher changed since submission')
    names = request['datasets']
    fds = dict(zip(names, args.lock_fds))
    if len(fds) != len(names):
        raise RuntimeError('Dataset locks not inherited')
    for name, fd in fds.items():
        expected = (root / 'logs' / ('.tu_benchmarks_beta_' + name + '.lock')).stat()
        held = os.fstat(fd)
        if (expected.st_dev, expected.st_ino) != (held.st_dev, held.st_ino):
            raise RuntimeError('Wrong dataset lock')
    pending, running, done = list(names), {}, {}
    state_path = Path(args.request).with_suffix('.progress.json')
    waiting_announced = False
    while pending or running:
        for name, job in list(running.items()):
            rc = job['process'].poll()
            if rc is not None:
                done[name] = {'exit_code': rc, **job['info']}
                del running[name]
                print('DATASET_FINISHED', name, 'exit=', rc, flush=True)
        if pending and len(running) < request['max_parallel']:
            available = candidates(gpu_inventory(),
                {j['info']['gpu']['index'] for j in running.values()},
                request['gpus'], request['min_free_mib'])
            while pending and available and len(running) < request['max_parallel']:
                name, gpu = pending.pop(0), available.pop(0)
                tag = request['tag'] + '_' + name + '_gpu' + str(gpu['index'])
                out = root / 'outputs' / tag
                log = root / 'logs/training' / (tag + '.log')
                req = log.with_suffix('.request.json')
                write_json(req, {'dataset': name, 'gpu': gpu,
                    'launcher_sha256': request['launcher_sha256'], 'output': str(out)})
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], WANDB_MODE='disabled',
                    MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                    PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1')
                command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                    'pipeline', '--root', str(root), '--dataset', name, '--worker',
                    '--output', str(out), '--request', str(req), '--lock-fd', str(fds[name])]
                try:
                    with log.open('xb') as handle:
                        process = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            pass_fds=(fds[name],))
                    info = {'dataset': name, 'pid': process.pid, 'log': str(log),
                            'output': str(out), 'gpu': gpu, 'status': 'started'}
                    Path(str(log) + '.pid').write_text(str(process.pid))
                    Path(str(log) + '.out').write_text(str(out))
                    pointer(root, name, info)
                    running[name] = {'process': process, 'info': info}
                    print('DATASET_START', json.dumps(info), flush=True)
                except OSError as error:
                    done[name] = {'exit_code': -1, 'error': repr(error)}
                    print('DATASET_START_FAILED', name, repr(error), flush=True)
        write_json(state_path, {'queued': pending,
            'running': {n: j['info'] for n, j in running.items()}, 'finished': done})
        if pending and not running and not waiting_announced:
            print('WAITING_FOR_FREE_MEMORY; no utilization restriction;', pending, flush=True)
            waiting_announced = True
        if pending or running:
            time.sleep(5)
    if any(item['exit_code'] != 0 for item in done.values()):
        raise RuntimeError('Some pipelines failed; inspect dataset logs: ' + repr(done))
    print('PASS: TU_BENCHMARKS_BETA_MEAN_HOG_PIPELINES_COMPLETE', flush=True)


def launch(args):
    import fcntl
    root = Path(args.root).resolve()
    if sys.platform != 'linux' or root != ROOT:
        raise RuntimeError('Launch only on the modified Linux server root')
    names = names_for(args.dataset)
    if not 1 <= args.max_parallel <= 3 or args.min_free_gib < 8:
        raise ValueError('max-parallel 1..3; at least 8 GiB free VRAM')
    for name in names:
        common(root).baseline(root, name)
    log_dir = root / 'logs/training'
    log_dir.mkdir(parents=True, exist_ok=True)
    locks = {}
    try:
        for name in names:
            lock = (root / 'logs' / ('.tu_benchmarks_beta_' + name + '.lock')).open('a+b')
            locks[name] = lock
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(name + ' already queued/running; use status, do not relaunch')
        if not candidates(gpu_inventory(), allowed=args.gpus, minimum=args.min_free_gib * 1024):
            raise RuntimeError('No selected GPU has enough free VRAM; no job launched')
        tag = 'tu_benchmarks_mean_hog_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log = log_dir / (tag + '.log')
        req = log.with_suffix('.request.json')
        request = {'datasets': list(names), 'tag': tag,
            'launcher_sha256': common(root).sha(__file__), 'gpus': args.gpus,
            'max_parallel': args.max_parallel, 'min_free_mib': args.min_free_gib * 1024}
        write_json(req, request)
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), 'pipeline',
            '--root', str(root), '--coordinator', '--request', str(req),
            '--lock-fds', *[str(lock.fileno()) for lock in locks.values()]]
        # Write queued pointers before spawning, so a fast worker cannot be
        # overwritten by an older queued state in the submitting process.
        for name in names:
            pointer(root, name, {'dataset': name, 'pid': None, 'log': str(log),
                                'status': 'queued', 'output': None})
        with log.open('xb') as handle:
            process = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                pass_fds=tuple(lock.fileno() for lock in locks.values()))
        Path(str(log) + '.pid').write_text(str(process.pid))
        write_json(root / 'logs/tu_benchmarks_beta_latest.json',
            {'pid': process.pid, 'log': str(log), 'request': str(req)})
        print('PID=' + str(process.pid))
        print('LOG=' + str(log))
        print('SUBMITTED: smoke -> fresh train -> validation-best full-chain -> mixed Table1/3')
        print('1000 epochs/stage imdb_binary; 500/stage collab,nci1; train seed0.')
        print('Batch H2/H1/Glue: imdb_binary32/32/32; collab32/16/8; nci1 32/32/32.')
        print('Sampling seed0; H2/H1 upstream_lcc; mixed evaluation independent LCC.')
        print('GPU free-memory ranking only; no utilization restriction; do not submit twice.')
    finally:
        for lock in locks.values():
            lock.close()


def status(args):
    root = Path(args.root).resolve()
    for name in names_for(args.dataset):
        info_path = root / 'logs' / (name + '_tu_beta_latest_pipeline.json')
        print('\n===== {} ====='.format(name.upper()), flush=True)
        if not info_path.is_file():
            print('No launch record')
            continue
        info = json.loads(info_path.read_text())
        print(json.dumps(info, indent=2))
        if info.get('pid'):
            subprocess.run(['ps', '-p', str(info['pid']), '-o',
                            'pid,stat,etime,pcpu,pmem,args'], check=False)
        out = Path(info['output']) if info.get('output') else None
        if out and (out / 'pipeline_status.json').is_file():
            print((out / 'pipeline_status.json').read_text())
        log = Path(info['log']) if args.stage == 'pipeline' or out is None else out / (
            'train/' + args.stage + '.log' if args.stage in STAGES else args.stage + '.log')
        print('VIEW_LOG=' + str(log))
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
    parser.add_argument('--dataset', choices=('imdb_binary', 'collab', 'nci1', 'all'), default='all')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--max-parallel', type=int, default=3)
    parser.add_argument('--min-free-gib', type=float, default=32)
    parser.add_argument('--stage', choices=('pipeline', 'h2', 'h1', 'glue',
                        'smoke_sample', 'sample', 'evaluation'), default='pipeline')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--coordinator', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    parser.add_argument('--request', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fds', type=int, nargs='+', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if Path(args.root).resolve() != ROOT or args.dataset == 'all' or args.lock_fd is None:
            raise RuntimeError('Invalid internal worker arguments')
        worker(args)
    elif args.coordinator:
        coordinator(args)
    elif args.action == 'status':
        status(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()

