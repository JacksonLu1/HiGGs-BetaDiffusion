"""Additive Linux installer; original HiGGs is a read-only frozen-data source."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'payload/tools'))
from qb_beta_common import ROOT, ORIGINAL, SETUP, SPECS, PROTECTED, sha, validate_manifest


def verify_bundle():
    document = json.loads((PACKAGE / 'BUNDLE.json').read_text(encoding='utf-8'))
    assert document['schema_version'] == 1 and document['sha256']
    for relative, digest in document['sha256'].items():
        path = PACKAGE / relative
        assert path.resolve().is_relative_to(PACKAGE) and not path.is_symlink()
        assert sha(path) == digest, 'Upload package file differs: ' + relative
    for path in (PACKAGE / 'payload').rglob('*.py'):
        assert path.relative_to(PACKAGE).as_posix() in document['sha256']


def expected_samples(manifest, h, split):
    records = {record['id']: record for record in manifest['graphs']}
    for graph_id in manifest['splits'][split]:
        record, parts = records[graph_id], manifest['partitions'][str(graph_id)]
        member = {n: i for i, part in enumerate(parts) for n in part}
        links = sorted({tuple(sorted((member[a], member[b]))) for a, b in record['edges']
                        if member[a] != member[b]})
        if h == '2':
            yield {'graph_id': graph_id, 'nodes': len(parts), 'edges': links, 'community_id': None, 'side': None}
            continue
        groups = ((i, part, None) for i, part in enumerate(parts)) if h == '1' else (
            (None, parts[a] + parts[b], len(parts[a])) for a, b in links)
        for community_id, nodes, side in groups:
            order = {n: i for i, n in enumerate(nodes)}
            edges = [(order[a], order[b]) for a, b in record['edges'] if a in order and b in order]
            yield {'graph_id': graph_id, 'nodes': len(nodes), 'edges': edges,
                   'community_id': community_id, 'side': side}


def cache_counts(manifest, h, split):
    return sum(1 for _ in expected_samples(manifest, h, split))


def plans(root, original):
    items, copies = {}, {}
    for name, spec in SPECS.items():
        manifest_rel = f'data_sources/splits/{name}_r1_s1234_l0.json'
        source_rel = f'data_sources/small_benchmarks/{name}.json'
        config_rel = f'configs/dataset/{name}.yaml'
        assert sha(original / manifest_rel) == spec['manifest_sha256'], 'Changed original frozen manifest'
        assert sha(original / source_rel) == spec['raw_sha256'], 'Changed source collection'
        assert sha(original / config_rel) == spec['config_sha256'], 'Changed original dataset configuration'
        frozen = json.loads((original / manifest_rel).read_text(encoding='utf-8'))
        source = json.loads((original / source_rel).read_text(encoding='utf-8'))
        validate_manifest(name, frozen)
        assert source['schema'] == 'higgs_small_benchmark_source_v1' and source['dataset'] == name
        assert source['source_graphs_dropped'] == [] and source['graphs'] == frozen['graphs']
        cache_root = f'data/{name}_r1_s1234_l0'
        paths = {h: {s: f'{cache_root}/processed/proc_{s}_v1_h{h}_max{spec["max_size"]}_{spec["manifest_sha256"]}.pt'
                     for s in ('train', 'val', 'test')} for h in ('2', '1', '1.5')}
        rels = [manifest_rel, source_rel, config_rel] + [p for group in paths.values() for p in group.values()]
        for rel in rels:
            path = original / rel
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(original), str(path)
            copies[rel] = path
        items[name] = {
            'name': name, 'resolution': spec['resolution'], 'max_size': spec['max_size'],
            'manifest': str(root / manifest_rel), 'manifest_sha256': spec['manifest_sha256'],
            'source': str(root / source_rel), 'datadir': str(root / cache_root), 'splits': spec['splits'],
            'cache_paths': {h: {s: str(root / p) for s, p in group.items()} for h, group in paths.items()},
            'input_sha256': {str(root / r): sha(original / r) for r in rels},
        }
    return items, copies


def verify_sample(data, expected, h, index, padding_width):
    import torch
    n = expected['nodes']
    assert data.num_nodes == n and data.x.shape == (n, 2 if h == '2' else 1)
    assert torch.isfinite(data.x).all() and torch.all(data.x[:, 0] == 1) and torch.all(data.x[:, 1:] == 0)
    assert data.y.shape == (1, 2)
    condition = 1 if h == '1.5' else 0
    assert data.y[0, condition] == 1 and data.y[0, 1-condition] == 0
    undirected = expected['edges']
    directed = sorted(undirected + [(v, u) for u, v in undirected])
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape == (2, len(directed))
    assert data.edge_index.t().tolist() == [list(edge) for edge in directed], 'Cached graph topology differs'
    assert data.edge_attr.shape == (len(directed), 2)
    assert torch.all(data.edge_attr[:, 1] == 1) and torch.all(data.edge_attr[:, 0] == 0)
    assert data.source_graph_id.reshape(-1).tolist() == [expected['graph_id']]
    assert int(data.idx) == index
    if h == '1':
        assert data.community_id.reshape(-1).tolist() == [expected['community_id']]
    if h == '1.5':
        assert data.pos.dtype == torch.bool and data.pos.shape == (padding_width, padding_width)
        side = torch.arange(n) < expected['side']
        wanted = torch.zeros((padding_width, padding_width), dtype=torch.bool)
        wanted[:n, :n] = side[:, None] == side[None, :]
        assert torch.equal(data.pos, wanted), 'Glue membership/padding differs from frozen communities'


def inspect_caches(root, original, items):
    sys.path[:0] = [str(root / 'dgd'), str(root)]
    spec = importlib.util.spec_from_file_location('qb_install_adapter', PACKAGE / 'payload/dgd/datasets/four_benchmark_beta.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    rows = []
    for name, item in items.items():
        manifest = original / Path(item['manifest']).relative_to(root)
        frozen = json.loads(manifest.read_text(encoding='utf-8'))
        for h, group in item['cache_paths'].items():
            for split, target in group.items():
                cache = original / Path(target).relative_to(root)
                before = sha(cache)
                dataset = adapter.FrozenHierarchyDataset(original / Path(item['datadir']).relative_to(root),
                            manifest, split, float(h), item['max_size'])
                expected = list(expected_samples(frozen, h, split))
                count = len(dataset)
                assert count == len(expected) == SPECS[name]['cache_counts'][h][('train', 'val', 'test').index(split)]
                width = max(r['nodes'] for r in expected) if h == '1.5' else None
                for index, record in enumerate(expected):
                    verify_sample(dataset[index], record, h, index, width)
                assert sha(cache) == before, 'Cache audit changed input'
                row = {'dataset': name, 'h': h, 'split': split, 'samples': count,
                       'exact_topology_and_source_ids': True, 'singletons_retained': True}
                rows.append(row)
                print('CACHE:', row, flush=True)
    return rows


def target_plan(root, all_files):
    expected = {rel: sha(source) for rel, source in all_files.items()}
    for rel in all_files:
        target = root / rel
        assert not target.is_symlink() and target.resolve().is_relative_to(root), str(target)
        if target.exists():
            assert target.is_file() and sha(target) == expected[rel], 'Existing different file; refusing overwrite: ' + str(target)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--original-root', default=str(ORIGINAL))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, original = Path(args.root).resolve(), Path(args.original_root).resolve()
    assert sys.platform == 'linux' and root == ROOT and original == ORIGINAL and root != original
    verify_bundle()
    items, copies = plans(root, original)
    protected = {str(root / p): sha(root / p) for p in PROTECTED}
    payload = {p.relative_to(PACKAGE / 'payload').as_posix(): p
               for p in (PACKAGE / 'payload').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for rel, source in payload.items():
        assert rel.endswith('.py')
        compile(source.read_text(encoding='utf-8'), rel, 'exec')
    all_files = {**copies, **payload}
    expected = target_plan(root, all_files)
    rows = inspect_caches(root, original, items)
    setup = {'root': str(root), 'original_root': str(original), 'items': items,
             'protected_sha256': protected,
             'package_sha256': {str(root / r): expected[r] for r in payload},
             'cache_checks': rows, 'shared_sources_modified': False,
             'protocol': 'original frozen r1/s1234/l0; train seed0,batch32; Beta H2/H1+mean16+HOG4; discrete Glue; mixed MMD'}
    setup_path = root / SETUP
    assert not setup_path.is_symlink() and setup_path.resolve().is_relative_to(root)
    if setup_path.exists():
        assert json.loads(setup_path.read_text(encoding='utf-8')) == setup, 'Existing setup differs; refusing overwrite'
    assert protected == {p: sha(p) for p in protected}, 'Protected source changed during preflight'
    if not args.apply:
        print('PASS: FOUR_BENCHMARKS_BETA_CHECK_ONLY; no files written')
        return
    backup = root / 'interface_patch_backups' / ('four_benchmarks_beta_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
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
            assert sha(source) == expected[rel], 'Original input changed during copy'
        if not setup_path.exists():
            setup_path.parent.mkdir(parents=True, exist_ok=True)
            with setup_path.open('x', encoding='utf-8') as handle:
                created.append(setup_path)
                json.dump(setup, handle, indent=2)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)  # Only paths this failed installer created.
        raise
    print('BACKUP:', backup)
    print('PASS: FOUR_BENCHMARKS_BETA_INSTALLED')
    print('Copied exact original source/config/manifest/36 caches; no repartitioning or shared source edits.')


if __name__ == '__main__':
    main()
