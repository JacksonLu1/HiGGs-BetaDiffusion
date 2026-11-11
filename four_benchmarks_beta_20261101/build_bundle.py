"""Package only the new additive server bundle, never production checkout data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    files = [p for p in ROOT.rglob('*') if p.is_file() and
             (p.parent == ROOT or 'payload' in p.relative_to(ROOT).parts) and
             p.suffix in ('.py', '.md') and '__pycache__' not in p.parts]
    files = sorted(files)
    metadata = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    bundle = ROOT / 'BUNDLE.json'
    bundle.write_text(json.dumps({'schema_version': 1, 'sha256': metadata}, indent=2) + '\n')
    target = ROOT.with_suffix('.zip')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [bundle]:
            archive.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    print('ZIP:', target)
    print('BYTES:', target.stat().st_size)
    print('SHA256:', hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
