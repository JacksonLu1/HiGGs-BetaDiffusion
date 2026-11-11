"""Linux-only additive installer. Original root is a read-only data source."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'payload/tools'))
from tg_beta_common import ROOT, ORIGINAL, SETUP, SPECS, PROTECTED, sha, validate_manifest


def plans(root, original):
    items, copies = {}, {}
    for name, spec in SPECS.items():
        manifest_rel = f'data_sources/splits/{name}_higgs_r{spec["resolution"]}_s1234_l0.json'
        raw_rel = f'data_sources/benchmark_graphs/{name}.pkl'
        config_rel = f'configs/dataset/{name}.yaml'
        assert sha(original / manifest_rel) == spec['manifest_sha256'], 'Wrong original frozen manifest'
        assert sha(original / raw_rel) == spec['raw_sha256'], 'Wrong raw graph collection'
        frozen = json.loads((original / manifest_rel).read_text(encoding='utf-8'))
        validate_manifest(name, frozen)
        cache_root = f'data/{name}_higgs_r{spec["resolution"]}_s1234'
        paths = {h: {s: f'{cache_root}/processed/proc_{s}_v1_h{h}_max{spec["max_size"]}_{spec["manifest_sha256"]}.pt'
                     for s in ('train', 'val', 'test')} for h in ('2', '1', '1.5')}
        rels = [manifest_rel, raw_rel, config_rel] + [p for group in paths.values() for p in group.values()]
        for rel in rels:
            copies[rel] = original / rel
        items[name] = {
            'name': name, 'resolution': spec['resolution'], 'max_size': spec['max_size'], 'manifest': str(root / manifest_rel),
            'manifest_sha256': spec['manifest_sha256'], 'source': str(root / raw_rel),
            'datadir': str(root / cache_root), 'splits': spec['splits'],
            'cache_paths': {h: {s: str(root / p) for s, p in group.items()} for h, group in paths.items()},
            'input_sha256': {str(root / r): sha(original / r) for r in rels},
        }
    return items, copies


def cache_counts(manifest, h, split):
    records = {r['id']: r for r in manifest['graphs']}
    count = 0
    for gid in manifest['splits'][split]:
        parts = manifest['partitions'][str(gid)]
        if h == '2':
            count += 1
        elif h == '1':
            count += len(parts)
        else:
            membership = {n: i for i, p in enumerate(parts) for n in p}
            count += len({tuple(sorted((membership[a], membership[b])))
                          for a, b in records[gid]['edges'] if membership[a] != membership[b]})
    return count


def inspect_caches(root, original, items):
    sys.path[:0] = [str(root / 'dgd'), str(root)]
    import torch
    spec = importlib.util.spec_from_file_location('sc_install_adapter', PACKAGE / 'payload/dgd/datasets/tree_grid_beta.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    rows = []
    for name, item in items.items():
        frozen = json.loads(Path(item['manifest']).read_text()) if Path(item['manifest']).is_file() else json.loads(
            (original / Path(item['manifest']).relative_to(root)).read_text())
        for h, group in item['cache_paths'].items():
            for split, target in group.items():
                path = original / Path(target).relative_to(root)
                data, slices = adapter.load_cpu(path)
                count = len(slices['x']) - 1
                assert count == cache_counts(frozen, h, split), (name, h, split, count)
                assert count == SPECS[name]['cache_counts'][h][('train', 'val', 'test').index(split)]
                assert data.x.shape[1] == (2 if h == '2' else 1)
                assert torch.isfinite(data.x).all() and torch.all(data.x[:, 0] == 1)
                assert torch.all(data.x[:, 1:] == 0)
                assert data.y.shape == (count, 2)
                condition = 1 if h == '1.5' else 0
                assert torch.all(data.y[:, condition] == 1) and torch.all(data.y[:, 1-condition] == 0)
                assert data.edge_attr.shape[1] == 2
                assert torch.all(data.edge_attr[:, 1] == 1) and torch.all(data.edge_attr[:, 0] == 0)
                ids = data.source_graph_id.reshape(-1).tolist()
                assert len(ids) == count and set(ids) <= set(frozen['splits'][split])
                if h == '2':
                    assert sorted(ids) == sorted(frozen['splits'][split])
                if h == '1.5':
                    assert data.pos.dtype == torch.bool
                rows.append({'dataset': name, 'h': h, 'split': split, 'samples': count})
                print('CACHE:', rows[-1], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--original-root', default=str(ORIGINAL))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, original = Path(args.root).resolve(), Path(args.original_root).resolve()
    assert sys.platform == 'linux' and root == ROOT and original == ORIGINAL
    assert root != original
    items, copies = plans(root, original)
    protected = {str(root / p): sha(root / p) for p in PROTECTED}
    payload = {p.relative_to(PACKAGE / 'payload').as_posix(): p
               for p in (PACKAGE / 'payload').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for rel, source in payload.items():
        assert rel.endswith('.py'), 'Unexpected payload format'
        compile(source.read_text(encoding='utf-8'), rel, 'exec')
    all_files = {**copies, **payload}
    expected = {rel: sha(source) for rel, source in all_files.items()}
    for rel in all_files:
        target = root / rel
        assert not target.is_symlink() and target.resolve().is_relative_to(root), str(target)
        if target.exists():
            assert sha(target) == expected[rel], 'Existing different file; refusing overwrite: ' + str(target)
    rows = inspect_caches(root, original, items)
    setup = {'root': str(root), 'original_root': str(original), 'items': items,
             'protected_sha256': protected,
             'package_sha256': {str(root / r): expected[r] for r in payload},
             'cache_checks': rows, 'shared_sources_modified': False,
             'protocol': 'frozen_original_Tree_r1_Grid_r2; seed1234; batch32; all stages500; H2/H1 Beta; H1 mean+HOG; discreteGlue'}
    setup_path = root / SETUP
    if setup_path.exists():
        assert json.loads(setup_path.read_text()) == setup, 'Existing setup differs; refusing overwrite'
    if not args.apply:
        print('PASS: TREE_GRID_BETA_CHECK_ONLY; no files written')
        return
    backup = root / 'interface_patch_backups' / ('tree_grid_beta_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    (backup / 'before.json').write_text(json.dumps({'protected': protected, 'expected': expected}, indent=2))
    created = []
    try:
        for rel, source in all_files.items():
            target = root / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open('rb') as src, target.open('xb') as dst:
                    created.append(target)
                    shutil.copyfileobj(src, dst)
            assert sha(target) == expected[rel]
        assert protected == {p: sha(p) for p in protected}, 'Protected source changed during install'
        for rel, source in copies.items():
            assert sha(source) == expected[rel], 'Original source changed during copy'
        if not setup_path.exists():
            setup_path.parent.mkdir(parents=True, exist_ok=True)
            with setup_path.open('x', encoding='utf-8') as handle:
                created.append(setup_path)
                json.dump(setup, handle, indent=2)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)  # Only files created by this failed install.
        raise
    print('BACKUP:', backup)
    print('PASS: TREE_GRID_BETA_INSTALLED')
    print('No shared source edits, repartitioning, training, sampling, or process signals.')


if __name__ == '__main__':
    main()


