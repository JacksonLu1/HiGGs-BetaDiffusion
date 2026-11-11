"""Build only this additive offline upload package; never production sources."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and (p.parent == ROOT or 'payload' in p.relative_to(ROOT).parts)
                   and p.suffix in ('.py', '.md') and '__pycache__' not in p.parts)
    for p in files:
        if p.suffix == '.py':
            compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    bundle = ROOT / 'BUNDLE.json'
    metadata = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    bundle.write_text(json.dumps({'schema_version': 1, 'sha256': metadata}, indent=2) + '\n',
                      encoding='utf-8')
    target = ROOT.with_suffix('.zip')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files + [bundle]:
            archive.write(p, (Path(ROOT.name) / p.relative_to(ROOT)).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    print('ZIP:', target)
    print('BYTES:', target.stat().st_size)
    print('FILES:', len(files) + 1)
    print('SHA256:', hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
