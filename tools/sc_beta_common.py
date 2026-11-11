"""Frozen Community-small/ENZYMES protocol; standard-library-only helpers."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
ORIGINAL = Path('/path/to/HiGGs-main1/HiGGs-main')
SETUP = 'data_sources/splits/small_collections_beta_setup_20261028.json'
SPECS = {
    'community_small': {
        'raw_sha256': 'ee0390de8ca9855e40b3a81cf0a84bfd0114a9a42ace369eb08a925e4d988157',
        'manifest_sha256': '782aa9f9c04eefb09a175784700c27a6163fc01efc41cd4298bf91ef03191839',
        'splits': {'train': 64, 'val': 16, 'test': 20},
        'train_samples': {'2': 64, '1': 128, '1.5': 64},
    },
    'enzymes': {
        'raw_sha256': 'd1fb37bd6322be1e7ee5e324672003b2b24b9b2e496d047a42e8f5a404cde110',
        'manifest_sha256': 'bc885cf5538c9ad53f713336d8a50c5ea3f40194dee3a86c72ba393d08891749',
        'splits': {'train': 376, 'val': 94, 'test': 117},
        'train_samples': {'2': 376, '1': 1766, '1.5': 1608},
    },
}
STAGES = {'h2': ('2', 'beta_h2', 'val/beta_h2_loss'),
          'h1': ('1', 'beta_h1_twohop_mean_hog_rw', 'val/h1_total'),
          'glue': ('1.5', 'original_glue', 'val/epoch_NLL')}
PROTECTED = (
    'dgd/main.py', 'dgd/datasets/abstract_dataset.py',
    'dgd/datasets/graph_collection.py', 'dgd/datasets/collection_sources.py',
    'dgd/diffusion_model_beta_h2.py', 'dgd/diffusion_model_beta_h1.py',
    'dgd/diffusion_model_discrete.py', 'dgd/diffusion/distributions.py',
    'dgd/diffusion/beta_diffusion_utils.py', 'dgd/diffusion/extra_features.py',
    'dgd/diffusion/hog_rw_features.py', 'dgd/models/two_hop_edge.py',
    'dgd/models/h1_twohop_transformer.py', 'dgd/models/h1_twohop_warm_start.py',
    'configs/config.yaml', 'configs/general/general_default.yaml',
    'configs/train/train_default.yaml', 'configs/model/beta_h2.yaml',
    'configs/model/beta_h1.yaml', 'configs/model/beta_h1_twohop.yaml',
    'configs/model/beta_h1_twohop_mean.yaml',
    'configs/model/beta_h1_twohop_mean_hog_rw.yaml',
    'configs/model/original_glue.yaml', 'higgs/higgs_utils.py',
    'higgs/sample_main.py', 'higgs/sampling.py', 'higgs/interface_ablation.py',
)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_hashes(expected):
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen input/source changed: ' + str(path))


def validate_manifest(name, manifest):
    spec = SPECS[name]
    assert manifest['dataset'] == name and manifest['schema_version'] == 1
    assert manifest['source_sha256'] == spec['raw_sha256']
    assert manifest['settings'] == {
        'resolution': 1.0, 'split_seed': 1234, 'louvain_seed': 0, 'weight': None,
        'node_labels': 'dummy', 'graph_condition': 'constant',
        'split_protocol': 'higgs_spectre_random_64_16_20_v2',
        'hierarchy_policy': 'merge_singletons_preserve_topology_v1'}
    assert manifest['validation_role'] == 'independent_held_out'
    assert manifest['source_graphs_dropped'] == []
    ids = [r['id'] for r in manifest['graphs']]
    selected = [i for s in ('train', 'val', 'test') for i in manifest['splits'][s]]
    assert sorted(ids) == sorted(selected) and len(set(selected)) == len(selected)
    assert {s: len(v) for s, v in manifest['splits'].items()} == spec['splits']
    assert len(ids) == sum(spec['splits'].values())
    for record in manifest['graphs']:
        parts = manifest['partitions'][str(record['id'])]
        assert parts and all(2 <= len(p) <= 125 for p in parts)
        assert sorted(n for p in parts for n in p) == list(range(record['n']))
        assert manifest['singleton_merges'][str(record['id'])] == []
        assert 2 <= len(parts) <= 125


def get_item(root, name):
    root = Path(root).resolve()
    setup = json.loads((root / SETUP).read_text(encoding='utf-8'))
    assert setup['root'] == str(root)
    item = setup['items'][name]
    assert item['manifest_sha256'] == SPECS[name]['manifest_sha256']
    assert sha(item['manifest']) == item['manifest_sha256']
    manifest = json.loads(Path(item['manifest']).read_text(encoding='utf-8'))
    validate_manifest(name, manifest)
    return item


def baseline(root, name):
    get_item(root, name)
    path = Path(root) / SETUP
    setup = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(setup['protected_sha256'])
    expected.update(setup['items'][name]['input_sha256'])
    expected.update(setup['package_sha256'])
    expected[str(path.resolve())] = sha(path)
    check_hashes(expected)
    return expected
