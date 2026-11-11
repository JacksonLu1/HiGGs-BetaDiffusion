"""Install DD only into modified HiGGs; reuse original frozen data, never rebuild it."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
ORIGINAL = Path('/path/to/HiGGs-main1/HiGGs-main')
POST_PLANAR_MAIN = 'dfe869fbeda2bf8e1b5907b8db43473c333f68c52d2ba6059942cc9927d3d0b5'
RAW_SHA = 'd033d7aeb1a48c4b2b47cf0390e7fc9671de70d98c8df5b11e458d2ec5515cd2'
MANIFEST = 'data_sources/splits/dd_hierarchy_r2_s1234_l0.json'
AUDIT = 'data_sources/splits/dd_beta_setup_20261027.json'
PROTECTED = {
    'dgd/diffusion_model_beta_h2.py': 'a0b895b21afe82d8335c6b77acafcaf18f1adc9fc734c0d83149a5b3babfabe6',
    'dgd/diffusion_model_beta_h1.py': 'b4d6078601afda3fb46ea40c0b9687d949898605dd29f5949d9eb65239d426a3',
    'dgd/diffusion_model_discrete.py': 'fd477d76e959fca43070b0a9a36e3aa5a0916a933cf8aecbd7a74ddac4643f70',
    'dgd/diffusion/distributions.py': 'ce79c6f1addca9d72b04dde0a694ea8872690963b892c9c8c8251688d2584e00',
    'higgs/higgs_utils.py': '6934a9d05d08818119ee97e64b846a9d6b59a1afc800b0dfd3966c53a049f648',
    'higgs/sampling.py': '4f770fd6df760b9a071b7fc2402e37dae1312bb94eb0b649b54a31fc6e0bc345',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_dd_manifest(obj):
    assert obj['dataset'] == 'dd' and obj['schema_version'] == 1
    assert obj['source_sha256'] == RAW_SHA
    assert obj['settings'] == {
        'resolution': 2.0, 'split_seed': 1234, 'louvain_seed': 0,
        'weight': None, 'node_labels': 'dummy', 'graph_condition': 'constant'}
    assert tuple(len(obj['splits'][s]) for s in ('train', 'val', 'test')) == (587, 147, 184)
    assert len(obj['graphs']) == 918
    ids = [r['id'] for r in obj['graphs']]
    split_ids = [i for s in ('train', 'val', 'test') for i in obj['splits'][s]]
    assert len(set(ids)) == 918 and sorted(ids) == sorted(split_ids)
    assert set(obj['partitions']) == {str(i) for i in ids}
    for record in obj['graphs']:
        assert 100 <= record['n'] <= 500
        parts = obj['partitions'][str(record['id'])]
        assert parts and all(parts)
        assert sorted(i for p in parts for i in p) == list(range(record['n']))


def change_main(text):
    replacements = [
        ('["fb_hierarchies", "shetland", "cora", "sbm", "planar"]:',
         '["fb_hierarchies", "shetland", "cora", "sbm", "planar", "dd"]:'),
        ('        if dataset_config["name"] == "planar":',
         '        if dataset_config["name"] == "dd":\n'
         '            from datasets.graph_collection import GraphCollectionDataModule\n'
         '            from datasets.dd_beta_adapter import DDBetaDatasetInfos\n'
         '            datamodule = GraphCollectionDataModule(cfg)\n'
         '            datamodule.prepare_data()\n'
         '            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)\n'
         '            dataset_infos = DDBetaDatasetInfos(datamodule, dataset_config)\n\n'
         '        elif dataset_config["name"] == "planar":'),
        ('        if cfg.dataset.name == "planar":',
         '        if cfg.dataset.name in ("planar", "dd"):'),
    ]
    for old, new in replacements:
        assert text.count(old) == 1, 'Unexpected main.py anchor: ' + old
        text = text.replace(old, new, 1)
    compile(text, 'dgd/main.py', 'exec')
    return text


def inspect_caches(root, digest, manifest):
    import torch
    sys.path[:0] = [str(root / 'dgd'), str(root)]
    from datasets.collection_sources import load_torch_cpu, validate_manifest
    validate_manifest(manifest)
    rows = []
    for h in ('2', '1', '1.5'):
        for split in ('train', 'val', 'test'):
            path = root / 'data/dd/processed' / f'proc_{split}_v1_h{h}_max500_{digest}.pt'
            data, slices = load_torch_cpu(path)
            n = len(slices['x']) - 1
            assert n > 0 and data.x.shape[1] == (2 if h == '2' else 1)
            assert torch.isfinite(data.x).all() and torch.all(data.x[:, 0] == 1)
            assert torch.all(data.x[:, 1:] == 0)
            assert data.y.shape == (n, 2) and torch.isfinite(data.y).all()
            condition = 1 if h == '1.5' else 0
            assert torch.all(data.y[:, condition] == 1)
            assert torch.all(data.y[:, 1 - condition] == 0)
            assert data.edge_attr.shape[1] == 2
            assert torch.all(data.edge_attr[:, 0] == 0) and torch.all(data.edge_attr[:, 1] == 1)
            assert set(data.source_graph_id.reshape(-1).tolist()) <= set(manifest['splits'][split])
            rows.append({'h': h, 'split': split, 'samples': n, 'X_width': data.x.shape[1],
                         'condition': condition, 'present_edge_class': 1, 'cache': str(path)})
            print('CACHE:', rows[-1], flush=True)
            del data, slices
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--original-root', default=str(ORIGINAL))
    args = parser.parse_args()
    root, original = Path(args.root).resolve(), Path(args.original_root).resolve()
    assert root == ROOT and original == ORIGINAL and root != original, 'Wrong experiment roots'
    payload = Path(__file__).resolve().parent / 'payload'
    manifest_path = original / MANIFEST
    frozen = json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_dd_manifest(frozen)
    digest = sha(manifest_path)
    assert sha(original / 'data_sources/dd/DD.zip') == RAW_SHA

    copy_rels = ['data_sources/dd/DD.zip', MANIFEST, 'configs/dataset/dd.yaml',
                 'dgd/datasets/collection_sources.py', 'dgd/datasets/graph_collection.py']
    copy_rels += [f'data/dd/processed/proc_{s}_v1_h{h}_max500_{digest}.pt'
                  for h in ('2', '1', '1.5') for s in ('train', 'val', 'test')]
    source_hashes = {rel: sha(original / rel) for rel in copy_rels}
    for rel in copy_rels:
        target = root / rel
        assert not target.is_symlink(), 'Refusing symlink target: ' + str(target)
        if target.exists():
            assert sha(target) == source_hashes[rel], 'Existing different file; not overwriting: ' + str(target)
    for rel, expected in PROTECTED.items():
        assert sha(root / rel) == expected, 'Protected source changed: ' + rel
    main_path = root / 'dgd/main.py'
    current = main_path.read_text(encoding='utf-8')
    main_before = sha(main_path)
    if 'from datasets.dd_beta_adapter import DDBetaDatasetInfos' in current:
        installed = json.loads((root / AUDIT).read_text())
        assert installed['manifest_sha256'] == digest and installed['main_sha256'] == main_before
        candidate = current
    else:
        assert main_before == POST_PLANAR_MAIN, 'Expected proven post-Planar main.py; no forced patch'
        candidate = change_main(current)

    additions = {p.relative_to(payload).as_posix(): p for p in payload.rglob('*') if p.is_file()}
    for rel, source in additions.items():
        compile(source.read_text(encoding='utf-8'), rel, 'exec')
        target = root / rel
        assert not target.is_symlink()
        if target.exists():
            assert sha(target) == sha(source), 'Existing different adapter/driver; not overwriting: ' + rel
    backup = root / 'interface_patch_backups' / ('dd_beta_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    shutil.copy2(main_path, backup / 'main.py.before')
    (backup / 'input_hashes.json').write_text(json.dumps({'original': source_hashes, 'protected': PROTECTED}, indent=2))
    created = []
    try:
        for rel in copy_rels:
            target = root / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with (original / rel).open('rb') as source, target.open('xb') as output:
                    created.append(target)
                    shutil.copyfileobj(source, output)
                assert sha(target) == source_hashes[rel]
                print('COPIED:', rel, flush=True)
            else:
                print('REUSED:', rel, flush=True)
        rows = inspect_caches(root, digest, frozen)
        for rel, source in additions.items():
            target = root / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open('rb') as handle, target.open('xb') as output:
                    created.append(target)
                    shutil.copyfileobj(handle, output)
        if candidate != current:
            main_path.write_text(candidate, encoding='utf-8')
        assert all(sha(root / rel) == value for rel, value in PROTECTED.items())
        assert all(sha(original / rel) == value for rel, value in source_hashes.items())
        report = {'dataset': 'dd', 'manifest_sha256': digest, 'source_sha256': RAW_SHA,
                  'splits': {s: len(frozen['splits'][s]) for s in ('train', 'val', 'test')},
                  'original_root': str(original), 'main_sha256': sha(main_path),
                  'shared_file_sha256': source_hashes, 'protected_sha256': PROTECTED,
                  'cache_checks': rows, 'backup': str(backup), 'no_repartitioning': True}
        (root / AUDIT).write_text(json.dumps(report, indent=2), encoding='utf-8')
    except BaseException:
        shutil.copy2(backup / 'main.py.before', main_path)
        for target in reversed(created):
            assert root in target.resolve().parents
            if target.exists():
                target.unlink()
        raise
    print('BACKUP:', backup)
    print('MANIFEST_SHA256:', digest)
    print('PASS: DD_BETA_MEAN_HOG_ADAPTER_INSTALLED')
    print('No training, sampling, repartitioning, original-project edits or existing-model changes.')


if __name__ == '__main__':
    main()
