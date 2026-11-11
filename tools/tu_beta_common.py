"""Pinned TU collection inputs; no downloads or hierarchy regeneration."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
ORIGINAL = Path('/path/to/HiGGs-main1/HiGGs-main')
SETUP = 'data_sources/splits/tu_benchmarks_beta_setup_20261103.json'
SPECS = {
    'imdb_binary': {
        'resolution': 1, 'max_size': 136, 'epochs': 1000, 'train_seed': 0,
        'batch_size': {'h2': 32, 'h1': 32, 'glue': 32},
        'sampling_seed': 0, 'component_policy': {'h2': 'upstream_lcc', 'h1': 'upstream_lcc'},
        'raw_sha256': 'f5d3b39fced2ba7012a79af5ef60ac2b921a97d9f0c427a77563257c6de85024',
        'manifest_sha256': '06d91b898732e8a6ac24293b4128db13061f58cfa70ada181785d1b94117f6c9',
        'config_sha256': '480b74520441362539bfeac8302c442308cded9e5b936d3d98c5e8ccc6d690ea',
        'splits': {'train': 640, 'val': 160, 'test': 200},
        'train_samples': {'2': 640, '1': 1754, '1.5': 1342},
        'cache_counts': {'2': [640, 160, 200], '1': [1754, 449, 550], '1.5': [1342, 352, 402]},
    },
    'collab': {
        'resolution': 1, 'max_size': 492, 'epochs': 500, 'train_seed': 0,
        'batch_size': {'h2': 32, 'h1': 16, 'glue': 8},
        'sampling_seed': 0, 'component_policy': {'h2': 'upstream_lcc', 'h1': 'upstream_lcc'},
        'raw_sha256': '4844dd29818ef1c8df4a6c6c2b99124a793047b64c196349a50cae154360c098',
        'manifest_sha256': '738178b6df55ebe60128c22d74e639c33d47efa17da485ed1ecbd962beb3392f',
        'config_sha256': 'ed3ac9fb1dc8620b09e185605a3e42e567d6f659155a2afa43ba926be408f1e3',
        'splits': {'train': 3200, 'val': 800, 'test': 1000},
        'train_samples': {'2': 3200, '1': 9649, '1.5': 11535},
        'cache_counts': {'2': [3200, 800, 1000], '1': [9649, 2422, 3114], '1.5': [11535, 2870, 3758]},
    },
    'nci1': {
        'resolution': 1, 'max_size': 111, 'epochs': 500, 'train_seed': 0,
        'batch_size': {'h2': 32, 'h1': 32, 'glue': 32},
        'sampling_seed': 0, 'component_policy': {'h2': 'upstream_lcc', 'h1': 'upstream_lcc'},
        'raw_sha256': 'fdd1531e61141b25025730d94dcf9ded9b48a1a0f5b2d4b28480eced4de7425b',
        'manifest_sha256': 'ecb55500dfc7f8ed94c2b7528cacf81b9cafe8d400305cd07f7b7346820939d5',
        'config_sha256': '70e107040294f9a200a69cc72822bbd4107cd641ae1a539bd5cf41c43844f368',
        'splits': {'train': 2630, 'val': 658, 'test': 822},
        'train_samples': {'2': 2630, '1': 12961, '1.5': 11423},
        'cache_counts': {'2': [2630, 658, 822], '1': [12961, 3248, 4019], '1.5': [11423, 2894, 3532]},
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
