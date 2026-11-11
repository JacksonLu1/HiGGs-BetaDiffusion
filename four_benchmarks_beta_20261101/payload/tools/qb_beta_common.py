"""Pinned four-collection inputs; no downloads or hierarchy regeneration."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
ORIGINAL = Path('/path/to/HiGGs-main1/HiGGs-main')
SETUP = 'data_sources/splits/four_benchmarks_beta_setup_20261101.json'
SPECS = {
    'ego_small': {
        'resolution': 1, 'max_size': 18, 'epochs': 1000, 'train_seed': 0,
        'raw_sha256': 'd4d606d61b79c445998f0a9e9aa46f440bc052737bf2641843f328e9e73f6f8a',
        'manifest_sha256': '374529d3bb5bbc013a163bad736d0f12a3418297eeb3e6101d9ba3981234f361',
        'config_sha256': '8cde209510bde8b4d269109583a3ac124d72de9ce79431ef7bc509258efd13ec',
        'splits': {'train': 128, 'val': 32, 'test': 40},
        'train_samples': {'2': 128, '1': 227, '1.5': 118},
        'cache_counts': {'2': [128, 32, 40], '1': [227, 49, 67], '1.5': [118, 19, 28]},
    },
    'proteins': {
        'resolution': 1, 'max_size': 620, 'epochs': 500, 'train_seed': 0,
        'raw_sha256': 'b6335356f3c0fc6f913741705a8b8f68eb81da9c0afa98354bdc97e7467c1e2e',
        'manifest_sha256': 'b48aa81179c3d1cb1c9616df1f343ad995f0659a0f20e98ee3bb0edf786f158c',
        'config_sha256': '30ff96d39a78274c5d5f7b554cf88dd2ce74b2df30cdcfdcd0911aca8bce3a54',
        'splits': {'train': 712, 'val': 178, 'test': 223},
        'train_samples': {'2': 712, '1': 3382, '1.5': 3150},
        'cache_counts': {'2': [712, 178, 223], '1': [3382, 812, 1033], '1.5': [3150, 774, 958]},
    },
    'msrc21': {
        'resolution': 1, 'max_size': 141, 'epochs': 500, 'train_seed': 0,
        'raw_sha256': '327bfa64a42e398c9bdc83e0acd88e8bf8c18ce87ef0d35156d0bd8f9097061a',
        'manifest_sha256': 'd9c3c8cd6f802492cf71bc0d94c4f3f4d43af9414da8624f9ed7a6eb2e09b2a3',
        'config_sha256': 'a13d7140ea891e4d50c473f8bd768e95c13513d3eeaf88b70303691cd1b58ef1',
        'splits': {'train': 360, 'val': 90, 'test': 113},
        'train_samples': {'2': 360, '1': 2200, '1.5': 3505},
        'cache_counts': {'2': [360, 90, 113], '1': [2200, 558, 689], '1.5': [3505, 880, 1096]},
    },
    'ego': {
        'resolution': 1, 'max_size': 399, 'epochs': 500, 'train_seed': 0,
        'raw_sha256': '05fb26e90240963d294cc8c1fa5ef43362d0f9c84bdcbe198994e79c075e0510',
        'manifest_sha256': 'bd5a14072b1cf68f9e6446f0e7e79ef85a103fbfe106bf1b36233bd277027e80',
        'config_sha256': '071eb98953b20f637a1aaa461db2654e8688c3883fc08146d75dbe266dc270d4',
        'splits': {'train': 485, 'val': 121, 'test': 151},
        'train_samples': {'2': 485, '1': 3800, '1.5': 8865},
        'cache_counts': {'2': [485, 121, 151], '1': [3800, 931, 1222], '1.5': [8865, 2013, 2861]},
    },
}
STAGES = {
    'h2': ('2', 'beta_h2', 'val/beta_h2_loss'),
    'h1': ('1', 'beta_h1_twohop_mean_hog_rw', 'val/h1_total'),
    'glue': ('1.5', 'original_glue', 'val/epoch_NLL'),
}
PROTECTED = (
    'dgd/main.py', 'dgd/datasets/abstract_dataset.py',
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
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def check_hashes(expected):
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen input/source changed: ' + str(path))


def validate_manifest(name, manifest):
    spec = SPECS[name]
    assert manifest['dataset'] == name and manifest['schema_version'] == 1
    assert manifest['source_sha256'] == spec['raw_sha256']
    assert manifest['settings'] == {
        'resolution': 1.0, 'split_seed': 1234, 'louvain_seed': 0,
        'weight': None, 'node_labels': 'dummy', 'graph_condition': 'constant',
        'split_protocol': 'whole_graph_64_16_20_raw_louvain_v1',
        'hierarchy_policy': 'raw_louvain_no_merge_no_drop',
    }
    assert manifest['validation_role'] == 'held_out_graph_ids'
    assert manifest['checkpoint_selection'] == 'validation_nll_best'
    assert manifest['source_graphs_dropped'] == []
    assert set(manifest['splits']) == set(spec['splits'])
    assert {s: len(v) for s, v in manifest['splits'].items()} == spec['splits']
    ids = [r['id'] for r in manifest['graphs']]
    selected = [i for s in ('train', 'val', 'test') for i in manifest['splits'][s]]
    assert len(set(ids)) == len(ids) == len(selected) == sum(spec['splits'].values())
    assert len(set(selected)) == len(selected) and sorted(ids) == sorted(selected)
    assert set(manifest['partitions']) == {str(i) for i in ids}
    for record in manifest['graphs']:
        n, edges = record['n'], record['edges']
        assert type(n) is int and 1 <= n <= spec['max_size']
        assert all(len(e) == 2 and all(type(v) is int for v in e)
                   and 0 <= e[0] < e[1] < n for e in edges)
        assert len({tuple(e) for e in edges}) == len(edges)
        parts = manifest['partitions'][str(record['id'])]
        # Singleton communities and singleton H2 graphs must remain unfiltered.
        assert parts and all(p and len(set(p)) == len(p) for p in parts)
        assert sorted(v for part in parts for v in part) == list(range(n))


def get_item(root, name):
    root = Path(root).resolve()
    setup = json.loads((root / SETUP).read_text(encoding='utf-8'))
    assert setup['root'] == str(root)
    item = setup['items'][name]
    spec = SPECS[name]
    assert item['manifest_sha256'] == sha(item['manifest']) == spec['manifest_sha256']
    assert sha(item['source']) == spec['raw_sha256']
    assert item['resolution'] == spec['resolution'] and item['max_size'] == spec['max_size']
    assert item['splits'] == spec['splits']
    validate_manifest(name, json.loads(Path(item['manifest']).read_text(encoding='utf-8')))
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
