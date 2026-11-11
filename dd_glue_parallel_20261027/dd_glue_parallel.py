"""Adopt a verified DD scheduler, preserving H1/H2, and start Glue on another GPU.

Linux only. No model/source edits, no signals to training children, no killpg.
"""
import argparse
from datetime import datetime
try:
    import fcntl
except ImportError:  # Allow Windows CPU tests to mock the Linux-only lock API.
    fcntl = None
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

# PIDFD_COMPAT_20261027
def _ensure_pidfd_compat():
    import ctypes as ct
    import errno
    native_open = getattr(os, 'pidfd_open', None)
    native_send = getattr(signal, 'pidfd_send_signal', None)
    fallback = not callable(native_open) or not callable(native_send)
    if fallback:
        if (sys.platform != 'linux' or os.uname().machine != 'x86_64'
                or ct.sizeof(ct.c_void_p) != 8 or ct.sizeof(ct.c_long) != 8):
            raise RuntimeError('pidfd compatibility requires Linux x86_64 LP64')
        syscall = ct.CDLL(None, use_errno=True).syscall
        syscall.restype = ct.c_long
        syscall.argtypes = [ct.c_long]
        def checked(number, *arguments):
            ct.set_errno(0)
            result = syscall(ct.c_long(number), *arguments)
            if result == -1:
                code = ct.get_errno() or errno.EIO
                raise OSError(code, os.strerror(code))
            return int(result)
        def compat_open(pid, flags=0):
            if flags != 0:
                raise ValueError('Only pidfd flags=0 is supported')
            return checked(434, ct.c_int(pid), ct.c_uint(flags))
        def compat_send(fd, sig, siginfo=None, flags=0):
            if siginfo is not None or flags != 0:
                raise ValueError('Only siginfo=None and flags=0 are supported')
            return checked(424, ct.c_int(fd), ct.c_int(sig), ct.c_void_p(None), ct.c_uint(flags))
    open_fn = native_open if callable(native_open) else compat_open
    send_fn = native_send if callable(native_send) else compat_send
    fd = open_fn(os.getpid(), 0)
    try:
        send_fn(fd, 0, None, 0)
    finally:
        os.close(fd)
    os.pidfd_open, signal.pidfd_send_signal = open_fn, send_fn
    return 'Linux x86_64 syscall' if fallback else 'Python native'

if sys.platform == 'linux':
    print('PIDFD backend:', _ensure_pidfd_compat(), flush=True)
# END_PIDFD_COMPAT_20261027
ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
RUN = 'dd_beta_mean_hog_r2_e500_scratch_seed0_20261027_205633_003751'
SMOKE = 'dd_beta_mean_hog_smoke_seed0_20261027_203843_088995_gpu0'
MANIFEST_SHA = '3162886166b33c6fd7832996664b0c6a422824c4f7c82a6d44efc4cc9e7745a2'
DRIVER_SHA = '3efdf7bdd3ee4ee024a937e48b3fa956a17c5132b854596251bff7e027d1ea7c'
MONITORS = {'h2': 'val/beta_h2_loss', 'h1': 'val/h1_total', 'glue': 'val/epoch_NLL'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    os.replace(temp, path)


def proc(pid):
    folder = Path('/proc') / str(pid)
    try:
        stat = (folder / 'stat').read_text()
        fields = stat.rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]),
                'startticks': fields[19], 'uid': folder.stat().st_uid,
                'argv': (folder / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(identity):
    current = proc(identity['pid'])
    return bool(current and current['startticks'] == identity['startticks']
                and current['argv'] == identity['argv'] and current['state'] != 'Z')


def expected_process(root, out, pid, stage, parent=None):
    info = proc(pid)
    assert info and info['state'] != 'Z', f'Process is no longer active: {pid}'
    assert info['uid'] == os.getuid(), 'Refusing to manage another user process'
    driver = str(root / 'tools/diagnostics/dd_beta_formal_20261027.py')
    assert info['argv'].count(driver) == 1, 'Unexpected process command'
    index = info['argv'].index(driver)
    target = out if stage == 'all' else out / stage
    assert info['argv'][index + 1:] == [str(root), str(target), stage], 'Wrong run or stage'
    if parent is not None:
        assert info['ppid'] == parent, 'Training process is not a child of this scheduler'
    return info


def flock_owner(path, pid):
    stat = Path(path).stat()
    for line in Path('/proc/locks').read_text().splitlines():
        fields = line.split()
        if 'FLOCK' not in fields:
            continue
        index = fields.index('FLOCK')
        if int(fields[index + 3]) != pid:
            continue
        major, minor, inode = fields[index + 4].split(':')
        if (int(major, 16), int(minor, 16), int(inode)) == (
                os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
            return True
    return False


def no_glue(root, out):
    for path in (out / 'glue', out / 'glue.log', out / 'glue.pid'):
        assert not path.exists(), 'Glue has started or has prior artifacts; do not force: ' + str(path)
    driver = str(root / 'tools/diagnostics/dd_beta_formal_20261027.py')
    for folder in Path('/proc').iterdir():
        if folder.name.isdigit():
            try:
                item = proc(int(folder.name))
            except PermissionError:
                continue
            if item and driver in item['argv']:
                index = item['argv'].index(driver)
                assert item['argv'][index + 1:] != [str(root), str(out / 'glue'), 'glue'], 'Glue process already active'


def check_memory(gpu):
    available = int(subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())
    assert available >= 8192, 'Less than 8 GiB free; scheduler unchanged'
    print(f'GPU {gpu}: {available} MiB free; no utilization restriction', flush=True)


def baseline(root):
    smoke = root / 'outputs' / SMOKE
    audit = json.loads((smoke / 'input_hash_check.json').read_text())
    assert audit['unchanged'] is True
    before = dict(audit['sha256'])
    assert all(sha(path) == value for path, value in before.items()), 'Input/source changed since successful smoke'
    driver = root / 'tools/diagnostics/dd_beta_formal_20261027.py'
    assert sha(driver) == DRIVER_SHA, 'Formal driver differs; do not force handoff'
    assert sha(root / 'data_sources/splits/dd_hierarchy_r2_s1234_l0.json') == MANIFEST_SHA
    before[str(driver)] = DRIVER_SHA
    before[str(Path(__file__).resolve())] = sha(__file__)
    return before


def preflight(request):
    root, out = Path(request['root']).resolve(), Path(request['out']).resolve()
    assert sys.platform == 'linux' and sys.flags.optimize == 0 and fcntl is not None
    assert root == ROOT and out == root / 'outputs' / RUN, 'Wrong experiment run'
    assert hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Linux/Python pidfd support required'
    assert int(request['gpu']) not in (0, 1), 'Do not add Glue to the two current training GPUs'
    parent = expected_process(root, out, request['parent_pid'], 'all')
    assert flock_owner(root / 'logs/training/dd_beta_formal.lock', parent['pid']), 'Specified scheduler does not own the DD lock'
    children = {key: expected_process(root, out, request[key + '_pid'], key, parent['pid'])
                for key in ('h1', 'h2')}
    no_glue(root, out)
    assert not (out / 'input_hash_check.json').exists(), 'Run is already marked complete'
    assert not (out / 'glue_parallel_handoff.json').exists(), 'Handoff was already attempted; inspect its log'
    check_memory(request['gpu'])
    print('Verifying frozen DD inputs and sources against the successful smoke audit...', flush=True)
    before = baseline(root)
    print('PASS: read-only handoff preflight', flush=True)
    return root, out, parent, children, before


def signal_parent(fd, value):
    signal.pidfd_send_signal(fd, value, None, 0)


def wait_stopped(identity, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        current = proc(identity['pid'])
        assert current and current['startticks'] == identity['startticks'], 'Scheduler exited or identity changed'
        if current['state'] in ('T', 't'):
            return
        time.sleep(.05)
    raise RuntimeError('Scheduler did not enter stopped state')


def preserved_or_completed(out, stage, identity):
    if same_process(identity):
        return
    validate_report(out, stage)


def transfer(root, out, parent, children):
    """Signal only the verified coordinator; never its group or training children."""
    lock_path = root / 'logs/training/dd_beta_formal.lock'
    lock = lock_path.open('a+')
    inode = os.fstat(lock.fileno()).st_ino
    assert inode == lock_path.stat().st_ino
    fd = os.pidfd_open(parent['pid'], 0)
    stopped = False
    try:
        assert same_process(parent), 'Coordinator identity changed before handoff'
        signal_parent(fd, signal.SIGSTOP)
        stopped = True
        wait_stopped(parent)
        no_glue(root, out)  # Freeze then recheck: H2 cannot race us into queued Glue.
        for key, identity in children.items():
            preserved_or_completed(out, key, identity)
        assert flock_owner(lock_path, parent['pid'])
        print('SWITCH: stopping only old scheduler PID', parent['pid'], flush=True)
        signal_parent(fd, signal.SIGTERM)
        signal_parent(fd, signal.SIGCONT)
        stopped = False
        assert select.select([fd], [], [], 10)[0], 'Old scheduler did not exit; Glue not launched'
        assert inode == lock_path.stat().st_ino, 'Lock file was replaced'
        deadline = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('DD lock was not acquired; Glue not launched')
                time.sleep(.1)
        return lock
    except BaseException:
        lock.close()
        raise
    finally:
        if stopped:
            try:
                signal_parent(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
        os.close(fd)


def validate_report(out, stage):
    report = json.loads((out / stage / 'formal_result.json').read_text())
    assert report['stage'] == stage and report['epoch'] == 499
    assert report['monitor'] == MONITORS[stage]
    assert report['strict_reload'] and report['strict_best_reload']
    assert report['manifest_sha256'] == MANIFEST_SHA
    assert 0 <= int(report['best_epoch']) <= 499
    assert math.isfinite(float(report['best_score'])) and int(report['global_step']) > 0
    folder = (out / stage / 'checkpoints' / f'dd_{stage}_mean_hog_formal').resolve()
    for name in ('best', 'last'):
        path = Path(report[name]).resolve()
        assert path.parent == folder and sha(path) == report[name + '_sha256']
    if stage == 'h1':
        assert report['two_hop']['normalization'] == 'mean'
        assert report['two_hop']['warm_start'] is None
        assert report['hog']['rw_depth'] == 4 and report['hog']['two_hop'] is True
        assert report['input_widths'] == {'X': 12, 'E': 6, 'y': 14}
    return report


def record_or_warn(path, value):
    try:
        atomic_json(path, value)
        return True
    except Exception as exc:
        print('WARNING: metadata write failed; scheduler continues holding its lock:',
              str(path), repr(exc), flush=True)
        return False


def drain_existing(children, glue):
    """Do not release the run lock around known still-running training on errors."""
    print('ERROR: scheduler will keep its DD lock until existing training processes exit.', flush=True)
    print('No H1/H2/Glue training process will be signaled or restarted.', flush=True)
    if glue is not None:
        print('Already-created Glue PID:', glue.pid, flush=True)
    while (any(same_process(identity) for identity in children.values())
           or (glue is not None and glue.poll() is None)):
        time.sleep(5)


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    root, out = Path(request['root']), Path(request['out'])
    with (root / 'logs/training/dd_glue_parallel_handoff.lock').open('a+') as migration:
        fcntl.flock(migration.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return worker_locked(request_path, request)


def worker_locked(request_path, request):
    root, out, parent, children, before = preflight(request)
    handoff = {'old_scheduler': parent, 'preserved_training': children,
               'new_scheduler_pid': os.getpid(), 'glue_gpu': request['gpu'],
               'phase': 'prepared',
               'model_and_driver_sources_unchanged': True,
               'baseline': str(root / 'outputs' / SMOKE / 'input_hash_check.json'),
               'note': 'H1/H2 adopted, not restarted; their OS exit codes are unavailable to this scheduler',
               'previous_latest_train': json.loads((root / 'logs/dd_latest_train.json').read_text())}
    # Persist intent BEFORE touching the old scheduler. Failure here sends no signals.
    atomic_json(Path(request_path), dict(request, handoff_intent=handoff))
    lock = transfer(root, out, parent, children)
    glue = None
    try:
        handoff['phase'] = 'scheduler_transferred'
        record_or_warn(out / 'glue_parallel_handoff.json', handoff)
        for stage, identity in children.items():
            preserved_or_completed(out, stage, identity)
        no_glue(root, out)
        check_memory(request['gpu'])
        log = out / 'glue.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(request['gpu']),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', MPLBACKEND='Agg',
                   WANDB_MODE='disabled', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        driver = root / 'tools/diagnostics/dd_beta_formal_20261027.py'
        with log.open('x') as handle:
            glue = subprocess.Popen([sys.executable, '-u', '-B', str(driver), str(root), str(out / 'glue'), 'glue'],
                                    cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        handoff.update(phase='glue_spawned', glue_pid=glue.pid)
        try:
            (out / 'glue.pid').write_text(str(glue.pid) + '\n')
        except Exception as exc:
            print('WARNING: Glue PID file write failed; Glue is already spawned:',
                  glue.pid, repr(exc), flush=True)
        record_or_warn(out / 'glue_parallel_handoff.json', handoff)
        current = {'pid': os.getpid(), 'log': request['log'], 'output': str(out), 'mode': 'train',
                   'manifest_sha256': MANIFEST_SHA, 'glue_parallel': True}
        record_or_warn(root / 'logs/dd_latest_train.json', current)
        print('PASS: DD_GLUE_PARALLEL_STARTED', flush=True)
        print('Glue PID:', glue.pid, 'GPU:', request['gpu'], flush=True)
        print('H1/H2 processes preserved:', {k: v['pid'] for k, v in children.items()}, flush=True)
        pending = set(MONITORS)
        errors, reports = [], {}
        while pending:
            for stage in sorted(pending.copy()):
                ended = glue.poll() is not None if stage == 'glue' else not same_process(children[stage])
                if not ended:
                    continue
                try:
                    if stage == 'glue':
                        assert glue.returncode == 0, f'Glue exited with code {glue.returncode}'
                    reports[stage] = validate_report(out, stage)
                    print(f'PASS: DD_{stage.upper()}_FORMAL_TRAINING_COMPLETE', flush=True)
                except Exception as exc:
                    errors.append(f'{stage}: {exc!r}')
                    print('FAILED STAGE:', stage, repr(exc), flush=True)
                pending.remove(stage)
            if pending:
                time.sleep(10)
        if errors:
            raise RuntimeError('; '.join(errors))
        assert all(sha(path) == value for path, value in before.items()), 'Inputs/source changed'
        atomic_json(out / 'input_hash_check.json', {'unchanged': True, 'sha256': before,
                    'schedule_handoff': str(out / 'glue_parallel_handoff.json')})
        handoff.update(completed=True, stage_best_epochs={k: v['best_epoch'] for k, v in reports.items()})
        record_or_warn(out / 'glue_parallel_handoff.json', handoff)
        print('PASS: DD_BETA_MEAN_HOG_THREE_STAGE_FORMAL_TRAINING_COMPLETE', flush=True)
        print('OUTPUT:', out, flush=True)
    except BaseException as exc:
        handoff['error'] = repr(exc)
        record_or_warn(out / 'glue_parallel_handoff.json', handoff)
        drain_existing(children, glue)
        raise
    finally:
        lock.close()


def launch(args):
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    request = {'root': str(root), 'out': str(out), 'parent_pid': args.parent_pid,
               'h1_pid': args.h1_pid, 'h2_pid': args.h2_pid, 'gpu': args.gpu}
    preflight(request)  # Read-only refusal before creating a worker or signaling anything.
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log = root / 'logs/training' / (RUN + '_glue_parallel_' + stamp + '.log')
    request['log'] = str(log)
    path = out / ('glue_parallel_request_' + stamp + '.json')
    atomic_json(path, request)
    with log.open('x') as handle:
        child = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--worker', str(path)],
                                 cwd=root, stdin=subprocess.DEVNULL, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    Path(str(log) + '.pid').write_text(str(child.pid) + '\n')
    Path(str(log) + '.out').write_text(str(out) + '\n')
    print(f'PID={child.pid}\nLOG={log}\nOUTPUT={out}')
    print('Check for PASS: DD_GLUE_PARALLEL_STARTED; do not launch twice.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--out', default=str(ROOT / 'outputs' / RUN))
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--h1-pid', type=int)
    parser.add_argument('--h2-pid', type=int)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        assert all((args.parent_pid, args.h1_pid, args.h2_pid)), 'Explicit verified PIDs are required'
        launch(args)


if __name__ == '__main__':
    main()
