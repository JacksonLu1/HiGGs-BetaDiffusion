"""Package additive recovery files only; no production source/data edits."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    files = sorted(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.md'))
    metadata = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    bundle = ROOT / 'BUNDLE.json'
    bundle.write_text(json.dumps({'schema_version': 1, 'sha256': metadata}, indent=2) + '\n')
    archive = ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for path in files + [bundle]:
            output.write(path, (Path(ROOT.name) / path.name).as_posix())
    print('ZIP:', archive)
    print('BYTES:', archive.stat().st_size)
    print('SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
