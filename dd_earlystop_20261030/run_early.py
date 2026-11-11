"""Freeze owned DD checkpoints, stop only exact H1 via pidfd, then sample.

No existing model, training driver, completion report or scheduler is edited.
The launcher detaches before it pauses H1, so an SSH disconnect is harmless.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
RUN = 'dd_beta_mean_hog_r2_e500_scratch_seed0_20261027_205633_003751'
SMOKE = 'dd_beta_mean_hog_smoke_seed0_20261027_203843_088995_gpu0'
DIGEST = '3162886166b33c6fd7832996664b0c6a422824c4f7c82a6d44efc4cc9e7745a2'
DRIVER_SHA = '3efdf7bdd3ee4ee024a937e48b3fa956a17c5132b854596251bff7e027d1ea7c'
MONITORS = {'h2': 'val/beta_h2_loss', 'h1': 'val/h1_total', 'glue': 'val/epoch_NLL'}
PACKAGE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(expected):
    for path, value in expected.items():
        if sha(path) != value:
            raise RuntimeError('Frozen source/input changed: ' + str(path))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def proc(pid):
    folder = Path('/proc') / str(pid)
    try:
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'state': fields[0], 'startticks': fields[19],
                'uid': folder.stat().st_uid,
                'argv': (folder / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def verify_identity(info, root, run):
    assert info and info['uid'] == os.getuid(), 'Missing process or different owner'
    assert info['state'] not in ('Z', 'T', 't'), 'H1 is zombie or already paused; refusing to manage it'
    driver = str(root / 'tools/diagnostics/dd_beta_formal_20261027.py')
    assert info['argv'].count(driver) == 1, 'PID is not the DD formal driver'
    index = info['argv'].index(driver)
    assert info['argv'][index+1:] == [str(root), str(run / 'h1'), 'h1'], 'Wrong run or stage'
    return info


def unchanged_identity(before, after):
    return bool(after and all(before[key] == after[key] for key in ('pid', 'uid', 'startticks', 'argv')))


def pidfd_api():
    """Own-process compatibility only; no edits to the old PIDFD helper."""
    import ctypes as ct
    import errno
    native_open = getattr(os, 'pidfd_open', None)
    native_send = getattr(signal, 'pidfd_send_signal', None)
    fallback = not callable(native_open) or not callable(native_send)
    if fallback:
        if (sys.platform != 'linux' or os.uname().machine != 'x86_64'
                or ct.sizeof(ct.c_void_p) != 8 or ct.sizeof(ct.c_long) != 8):
            raise RuntimeError('PIDFD fallback requires Linux x86_64 LP64')
        syscall = ct.CDLL(None, use_errno=True).syscall
        syscall.restype = ct.c_long
        syscall.argtypes = [ct.c_long]
        def checked(number, *values):
            ct.set_errno(0)
            result = syscall(ct.c_long(number), *values)
            if result == -1:
                error = ct.get_errno() or errno.EIO
                raise OSError(error, os.strerror(error))
            return int(result)
        def fallback_open(pid, flags=0):
            if flags:
                raise ValueError('Only PIDFD flags=0 supported')
            return checked(434, ct.c_int(pid), ct.c_uint(flags))
        def fallback_send(fd, value, info=None, flags=0):
            if info is not None or flags:
                raise ValueError('Only info=None flags=0 supported')
            return checked(424, ct.c_int(fd), ct.c_int(value), ct.c_void_p(None), ct.c_uint(flags))
    open_fn = native_open if callable(native_open) else fallback_open
    send_fn = native_send if callable(native_send) else fallback_send
    fd = open_fn(os.getpid(), 0)
    try:
        send_fn(fd, 0, None, 0)
    finally:
        os.close(fd)
    print('PIDFD backend:', 'Linux x86_64 syscall' if fallback else 'Python native', flush=True)
    return open_fn, send_fn


def gpu_rows():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free = [x.strip() for x in line.split(',')]
        rows.append({'index': int(index), 'uuid': uuid, 'free_mib': float(free)})
    return sorted(rows, key=lambda value: (-value['free_mib'], value['index']))


def choose_gpu():
    rows = gpu_rows()
    assert rows and rows[0]['free_mib'] >= 8192, 'No GPU with 8 GiB free; no stop should be attempted'
    print('GPU memory ranking:', rows, '; no utilization restriction', flush=True)
    return rows[0]


def import_project(root):
    os.chdir(root)
    sys.path[:0] = [str(root / 'dgd'), str(root), str(root / 'higgs')]
    from datasets import dd_beta_adapter
    assert Path(dd_beta_adapter.__file__).resolve() == root / 'dgd/datasets/dd_beta_adapter.py'


def load_cpu(path):
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def baseline(root):
    path = root / 'outputs' / SMOKE / 'input_hash_check.json'
    audit = json.loads(path.read_text())
    assert audit['unchanged'] is True
    before = dict(audit['sha256'])
    before[str(path)] = sha(path)
    formal = root / 'tools/diagnostics/dd_beta_formal_20261027.py'
    assert sha(formal) == DRIVER_SHA, 'Unknown training driver; do not force'
    manifest = root / 'data_sources/splits/dd_hierarchy_r2_s1234_l0.json'
    assert sha(manifest) == DIGEST
    before[str(formal)] = DRIVER_SHA
    for rel in ('higgs/sample_main.py', 'higgs/interface_ablation.py'):
        before[str(root / rel)] = sha(root / rel)
    for path in (Path(__file__).resolve(), PACKAGE / 'sample_early.py'):
        before[str(path)] = sha(path)
    check_hashes(before)
    return before


def complete_report(run, stage):
    path = run / stage / 'formal_result.json'
    report = json.loads(path.read_text())
    assert report['stage'] == stage and report['epoch'] == 499
    assert report['manifest_sha256'] == DIGEST
    assert report['strict_reload'] and report['strict_best_reload']
    assert report['monitor'] == MONITORS[stage]
    folder = run / stage / 'checkpoints' / ('dd_' + stage + '_mean_hog_formal')
    for key in ('best', 'last'):
        source = Path(report[key]).resolve()
        assert source.parent == folder and sha(source) == report[key + '_sha256']
    return path, report


def copy_frozen(source, target):
    assert source.is_file() and not source.is_symlink()
    first = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as original, target.open('xb') as frozen:
        shutil.copyfileobj(original, frozen)
    assert sha(target) == first == sha(source), 'Checkpoint changed during copy'
    return first


def strict_best(path, stage, manifest):
    import torch
    from omegaconf import OmegaConf
    import main as entry
    state = load_cpu(path)
    cfg = state['hyper_parameters']['cfg']
    cfg = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
    assert cfg.dataset.name == 'dd' and float(cfg.dataset.h) == {'h2': 2., 'h1': 1., 'glue': 1.5}[stage]
    assert cfg.dataset.resolution == 2 and Path(cfg.dataset.manifest).resolve() == manifest
    assert cfg.general.resume is None and cfg.general.test_only is None
    assert cfg.train.seed == 0 and cfg.train.batch_size == 4
    assert 0 <= int(state['epoch']) <= 499
    model = entry.get_model_class(cfg).load_from_checkpoint(str(path), map_location='cpu', cfg=cfg, strict=True)
    assert model.dataset_info.manifest_sha256 == DIGEST
    assert list(model.dataset_info.training_graph_ids) == json.loads(manifest.read_text())['splits']['train']
    assert type(model.node_dist).__module__ == 'datasets.dd_beta_adapter'
    assert all(torch.isfinite(p).all() for p in model.parameters())
    if stage == 'h1':
        assert state['h1_two_hop']['normalization'] == 'mean' and state['h1_two_hop']['warm_start'] is None
        assert state['h1_two_hop']['channels'] == 16
        assert state['h1_hog_rw']['rw_depth'] == 4 and state['h1_hog_rw']['two_hop'] is True
        assert {k: getattr(model.model, 'mlp_in_' + k)[0].in_features for k in ('X', 'E', 'y')} == {'X': 12, 'E': 6, 'y': 14}
    result = {'epoch': int(state['epoch']), 'global_step': int(state['global_step']), 'strict_reload': True}
    del model, state
    return result


def freeze_checkpoints(root, run, snapshot, before):
    """Called only while verified H1 is stopped, or has formally completed."""
    reports, checkpoints = {}, {}
    manifest = root / 'data_sources/splits/dd_hierarchy_r2_s1234_l0.json'
    for stage in ('h2', 'glue'):
        report_path, report = complete_report(run, stage)
        before[str(report_path)] = sha(report_path)
        reports[stage] = report
    for stage in ('h2', 'h1', 'glue'):
        source_folder = run / stage / 'checkpoints' / ('dd_' + stage + '_mean_hog_formal')
        source_last = source_folder / 'last.ckpt'
        saved_last = snapshot / stage / 'saved_last.ckpt'
        last_sha = copy_frozen(source_last, saved_last)
        state = load_cpu(saved_last)  # Refuse partially written checkpoint; caller resumes H1.
        last_reloaded = strict_best(saved_last, stage, manifest)
        last_epoch = int(state['epoch'])
        assert last_reloaded['epoch'] == last_epoch
        assert last_reloaded['global_step'] == int(state['global_step'])
        assert 0 <= last_epoch <= 499
        if stage != 'h1':
            assert last_epoch == 499
        callbacks = [v for v in state.get('callbacks', {}).values()
                     if isinstance(v, dict) and v.get('monitor') == MONITORS[stage]]
        assert len(callbacks) == 1
        callback = callbacks[0]
        score = float(callback['best_model_score'])
        assert math.isfinite(score)
        source_best = source_folder / Path(callback['best_model_path']).name
        if stage != 'h1':
            assert source_best == Path(reports[stage]['best']).resolve()
        best = snapshot / stage / source_best.name
        best_sha = copy_frozen(source_best, best)
        selected = strict_best(best, stage, manifest)
        assert selected['epoch'] <= last_epoch
        before[str(best)] = best_sha
        before[str(saved_last)] = last_sha
        checkpoints[stage] = dict(selected, best=str(best), best_sha256=best_sha,
            saved_last=str(saved_last), last_sha256=last_sha, monitor=MONITORS[stage],
            best_score=score, training_last_epoch=last_epoch,
            training_last_global_step=int(state['global_step']), source_best=str(source_best),
            source_last=str(source_last), last_state_loaded=True)
        del state
    check_hashes(before)
    selection = {'root': str(root), 'run': str(run), 'snapshot': str(snapshot),
        'manifest_sha256': DIGEST, 'checkpoints': checkpoints, 'input_sha256': before,
        'h1_early_stopped': checkpoints['h1']['training_last_epoch'] < 499,
        'budget_note': 'saved epoch is not live log epoch; selected best epoch is not total training budget',
        'comparison_note': 'original DD500/500/about250; new500/H1saved-progress/500; not equal-budget ablation',
        'no_formal_completion_fabricated': True}
    write_json(snapshot / 'selection.json', selection)
    return selection


def wait_stopped(identity, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        current = proc(identity['pid'])
        assert unchanged_identity(identity, current), 'H1 identity changed'
        if current['state'] in ('T', 't'):
            return
        time.sleep(.05)
    raise RuntimeError('H1 did not enter stopped state')


def validate_sampler_ready(root, snapshot, output, count):
    """Same metadata/source gates as real sampler, before terminating H1."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('dd_early_snapshot_preflight', PACKAGE / 'sample_early.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_snapshot(root, snapshot, output, count, allow_prepared=True)


def freeze_and_stop(root, run, snapshot, sample_output=None, count=184):
    import_project(root)
    before = baseline(root)
    for stage in ('h2', 'glue'):
        complete_report(run, stage)
    choose_gpu()  # Refuse no-memory situation before any process signal.
    pid = int((run / 'h1.pid').read_text().strip())
    identity = proc(pid)
    stopped, terminated, fd = False, False, None
    old_handlers = {}
    audit = {'status': 'preflight', 'signals': [], 'snapshot_verified': False,
             'target_h1_pid': pid, 'other_processes_signaled': False}
    audit_path = snapshot / 'stop_audit.json'
    sample_output = sample_output or (root / 'higgs/outputs' / snapshot.name)
    write_json(audit_path, audit)
    def interrupted(signum, frame):
        raise RuntimeError('Early-stop helper interrupted by signal ' + str(signum))
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        if identity is None or identity['state'] == 'Z':
            report_path, report = complete_report(run, 'h1')  # Require proven formal499 completion.
            before[str(report_path)] = sha(report_path)
            selection = freeze_checkpoints(root, run, snapshot, before)
            audit.update(status='already_completed', snapshot_verified=True,
                         selection_sha256=sha(snapshot / 'selection.json'))
            write_json(audit_path, audit)
            return selection
        verify_identity(identity, root, run)
        open_fd, send = pidfd_api()
        fd = open_fd(pid, 0)
        assert unchanged_identity(identity, proc(pid)), 'PID changed after PIDFD open'
        verify_identity(proc(pid), root, run)
        audit['identity'] = identity
        audit['status'] = 'pause_intent'
        write_json(audit_path, audit)
        stopped = True  # Set before send, so interruption immediately after SIGSTOP still resumes H1.
        send(fd, signal.SIGSTOP, None, 0)
        audit['signals'].append('SIGSTOP')
        wait_stopped(identity)
        selection = freeze_checkpoints(root, run, snapshot, before)
        assert selection['checkpoints']['h1']['training_last_epoch'] < 499, (
            'H1 reached final epoch during preparation; resume it and wait for its formal result')
        audit.update(status='snapshot_ready_stop_intent', snapshot_verified=True,
                     selection_sha256=sha(snapshot / 'selection.json'))
        write_json(audit_path, audit)  # Must persist before sending termination.
        validate_sampler_ready(root, snapshot, sample_output, count)
        assert unchanged_identity(identity, proc(pid)), 'Identity changed before termination'
        terminated = True  # From this point cleanup must never claim training resumed.
        send(fd, signal.SIGTERM, None, 0)
        audit['signals'].append('SIGTERM')
        send(fd, signal.SIGCONT, None, 0)
        stopped = False
        audit['signals'].append('SIGCONT')
        if not select.select([fd], [], [], 60)[0]:
            raise RuntimeError('H1 exit not confirmed within60seconds; sampling NOT launched; no SIGKILL')
        audit['status'] = 'h1_stopped'
        audit['h1_exit_confirmed'] = True
        write_json(audit_path, audit)
        print('PASS: H1_ONLY_STOPPED_AFTER_VERIFIED_SNAPSHOT', flush=True)
        print(json.dumps(selection['checkpoints'], indent=2), flush=True)
        return selection
    except BaseException as error:
        # Do not let a second operator interrupt prevent the required SIGCONT cleanup.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        audit['error'] = repr(error)
        audit['status'] = 'stop_uncertain' if terminated else 'failed_before_termination'
        if stopped and fd is not None:
            try:
                # If TERM was already sent, CONT permits it to take effect;
                # otherwise this resumes training that we temporarily paused.
                send(fd, signal.SIGCONT, None, 0)
                stopped = False
                audit['signals'].append('SIGCONT_cleanup')
                audit['h1_resumed_before_termination'] = not terminated
            except ProcessLookupError:
                pass
        try:
            write_json(audit_path, audit)
        except Exception as writing:
            print('WARNING: unable to persist failure audit:', repr(writing), flush=True)
        raise
    finally:
        if stopped and fd is not None:
            try:
                send(fd, signal.SIGCONT, None, 0)
            except ProcessLookupError:
                pass
        if fd is not None:
            os.close(fd)
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)


def worker(args):
    root, snapshot = Path(args.root).resolve(), Path(args.snapshot).resolve()
    assert root == ROOT and snapshot.parent == root / 'outputs'
    lock = root / 'logs/sampling/dd_mean_hog_fullchain.lock'
    expected, inherited = lock.stat(), os.fstat(args.lock_fd)
    assert (expected.st_dev, expected.st_ino) == (inherited.st_dev, inherited.st_ino)
    request = json.loads((snapshot / 'request.json').read_text())
    assert request['root'] == str(root) and request['snapshot'] == str(snapshot)
    assert request['stop_h1_authorized'] is True and request['count'] in (1, 184)
    assert Path(request['sample_output']).resolve().parent == root / 'higgs/outputs'
    assert request['helper_sha256'] == sha(__file__)
    assert request['sampler_sha256'] == sha(PACKAGE / 'sample_early.py')
    run = root / 'outputs' / RUN
    status = {'phase': 'freeze_and_stop', 'status': 'running', 'snapshot': str(snapshot)}
    write_json(snapshot / 'status.json', status)
    try:
        selection = freeze_and_stop(root, run, snapshot, Path(request['sample_output']), request['count'])
        gpu = choose_gpu()  # Recheck after H1 released its VRAM; choose most free now.
        status.update(phase='sampling', gpu=gpu, selected_epochs={k: v['epoch'] for k, v in selection['checkpoints'].items()})
        write_json(snapshot / 'status.json', status)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        command = [sys.executable, '-u', '-B', str(PACKAGE / 'sample_early.py'),
                   '--root', str(root), '--snapshot', str(snapshot), '--output', request['sample_output'],
                   '--count', str(request['count']), '--lock-fd', str(args.lock_fd)]
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 pass_fds=(args.lock_fd,))
        (snapshot / 'sample.pid').write_text(str(child.pid))
        rc = child.wait()
        if rc:
            raise RuntimeError('Early sampling failed exit={}; frozen snapshot remains safe'.format(rc))
        status.update(status='complete', phase='sampling_complete', sample_output=request['sample_output'])
        print('PASS: DD_EARLYSTOP_AND_SAMPLING_COMPLETE', flush=True)
        print('SAMPLE_OUTPUT:', request['sample_output'], flush=True)
        print('No evaluation yet; no old-model resampling or formal500 completion fabricated.', flush=True)
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(snapshot / 'status.json', status)


def launch(args):
    import fcntl
    root = Path(args.root).resolve()
    assert root == ROOT and sys.platform == 'linux' and sys.flags.optimize == 0
    assert args.stop_h1, 'Explicit --stop-h1 is required'
    run = root / 'outputs' / RUN
    assert (run / 'h1.pid').is_file() and (PACKAGE / 'sample_early.py').is_file()
    log_dir = root / 'logs/sampling'
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (log_dir / 'dd_mean_hog_fullchain.lock').open('a+b')
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('DD sampling/earlystop already active; inspect existing job, do not relaunch')
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        snapshot = root / 'outputs' / ('dd_earlystop_snapshot_' + stamp)
        snapshot.mkdir(exist_ok=False)
        name = 'dd_mean_hog_earlystop_seed1234_n{}_{}'.format(args.count, stamp)
        sample_output = root / 'higgs/outputs' / name
        assert not sample_output.exists()
        request = {'root': str(root), 'run': str(run), 'snapshot': str(snapshot),
                   'sample_output': str(sample_output), 'count': args.count,
                   'helper_sha256': sha(__file__), 'sampler_sha256': sha(PACKAGE / 'sample_early.py'),
                   'stop_h1_authorized': True}
        write_json(snapshot / 'request.json', request)
        log = log_dir / (name + '.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONHASHSEED='0', MPLBACKEND='Agg', WANDB_MODE='disabled',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), 'launch',
                   '--root', str(root), '--snapshot', str(snapshot), '--worker', '--lock-fd', str(lock.fileno())]
        with log.open('xb') as handle:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        Path(str(log) + '.pid').write_text(str(child.pid))
        Path(str(log) + '.out').write_text(str(sample_output))
        info = {'pid': child.pid, 'log': str(log), 'snapshot': str(snapshot), 'output': str(sample_output)}
        write_json(root / 'logs/dd_earlystop_latest.json', info)
        print(json.dumps(info, indent=2), flush=True)
        print('Submitted only: freeze+strictreload -> exactH1stop -> sampling. Check log.', flush=True)
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status'))
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--stop-h1', action='store_true')
    parser.add_argument('--count', type=int, choices=(1, 184), default=184)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--snapshot', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.action == 'status':
        info = json.loads((Path(args.root) / 'logs/dd_earlystop_latest.json').read_text())
        print(json.dumps(info, indent=2))
        state = Path(info['snapshot']) / 'status.json'
        if state.is_file():
            print(state.read_text())
        subprocess.run(['tail', '-n', '45', info['log']], check=False)
    else:
        launch(args)


if __name__ == '__main__':
    main()
