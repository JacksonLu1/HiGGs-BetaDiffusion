"""Offline installer: hash-check the supplied server snapshot before changes."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe path in manifest: {relative}')
    path = root.joinpath(*rel.parts)
    path.resolve().relative_to(root.resolve())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='Read-only checks (default)')
    modes.add_argument('--apply', action='store_true', help='Back up and apply verified payload')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    entries = manifest['files']
    payloads = []
    for entry in entries:
        src = safe_path(bundle / 'payload', entry['path'])
        dst = safe_path(root, entry['path'])
        if sha256(src) != entry['sha256']:
            raise RuntimeError(f'Corrupt payload: {entry["path"]}')
        data = src.read_bytes()
        if dst.suffix == '.py':
            compile(data, str(dst), 'exec')  # No imports, datasets or pycache.
        payloads.append((entry, src, dst, data))
    if all(dst.is_file() and sha256(dst) == entry['sha256']
           for entry, _, dst, _ in payloads):
        print('Already applied: all six installed files match this bundle.')
        return
    errors = []
    for rel, expected in manifest['required_inputs'].items():
        target = safe_path(root, rel)
        if not target.is_file() or sha256(target) != expected:
            errors.append(f'Server source differs from uploaded snapshot: {rel}')
    for entry, _, dst, _ in payloads:
        if entry['before_sha256'] is None and dst.exists():
            errors.append(f'New target already exists; will not overwrite: {entry["path"]}')
        elif entry['before_sha256'] is not None:
            if not dst.is_file() or sha256(dst) != entry['before_sha256']:
                errors.append(f'Original target changed: {entry["path"]}')
    if errors:
        raise RuntimeError('\n'.join(errors) + '\nNo source file was changed.')
    for entry, _, _, _ in payloads:
        action = 'MODIFY' if entry['before_sha256'] is not None else 'ADD'
        print(action, entry['path'])
    print('PASS: payload syntax and server source hashes verified.')
    if not args.apply:
        print('Check only: no files changed. Use --apply to install.')
        return
    backup = root / 'interface_patch_backups' / (
        'h1_twohop_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    for entry, _, dst, _ in payloads:
        if entry['before_sha256'] is not None:
            saved = safe_path(backup, entry['path'])
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, saved)
            if sha256(saved) != entry['before_sha256']:
                raise RuntimeError('Backup verification failed; no source file changed.')
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    created = []
    try:
        # Add new files first. Replace the one original source only at the end.
        for entry, _, dst, data in payloads:
            if entry['before_sha256'] is None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open('xb') as handle:
                    created.append((entry, dst))
                    handle.write(data)
        for entry, _, dst, data in payloads:
            if entry['before_sha256'] is not None:
                if sha256(dst) != entry['before_sha256']:
                    raise RuntimeError(f'Source changed during install: {dst}')
                fd, temporary = tempfile.mkstemp(prefix='.twohop_', dir=dst.parent)
                try:
                    with os.fdopen(fd, 'wb') as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(temporary, dst.stat().st_mode)
                    os.replace(temporary, dst)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
    except Exception:
        for entry, dst in created:
            if dst.is_file() and sha256(dst) == entry['sha256']:
                dst.unlink()  # Only files created by this invocation.
        raise
    for entry, _, dst, _ in payloads:
        if sha256(dst) != entry['sha256']:
            raise RuntimeError(f'Post-install hash mismatch: {dst}; backup: {backup}')
    print('Installed and verified. Backup:', backup)
    print('No training or sampling was launched.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
