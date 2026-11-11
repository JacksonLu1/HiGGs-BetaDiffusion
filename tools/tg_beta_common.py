"""Frozen Tree/Grid inputs; no partitioning, downloads or shared source edits."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/path/to/HiGGs-GBD/HiGGs-main')
ORIGINAL = Path('/path/to/HiGGs-main1/HiGGs-main')
SETUP = 'data_sources/splits/tree_grid_beta_setup_20261029.json'
SPECS = {
    'tree': {
        'resolution': 1, 'max_size': 64,
        'raw_sha256': 'ad27d04dd28fb21267020c3f1e7881bbaab936e33a24cdbcc62274507748f450',
        'manifest_sha256': '59c3e8b9e7eed327098018da3caf691c780c1a1c932900c224d59ebb448a7e37',
        'split_protocol': 'author_tree_128_32_40_v1',
        'splits': {'train': 128, 'val': 32, 'test': 40},
        'train_samples': {'2': 128, '1': 1070, '1.5': 942},
        'cache_counts': {'2': [128, 32, 40], '1': [1070, 275, 330], '1.5': [942, 243, 290]},
    },
    'grid': {
        'resolution': 2, 'max_size': 361,
        'raw_sha256': '1c533398435301faaa1282993e1738069cafcc5e7662b284a3dba23cee680104',
        'manifest_sha256': 'da59020eeea53566a3f60e480c39e481fc797ba2314161a0e6e691603de38338',
        'split_protocol': 'higgs_spectre_random_64_16_20_v1',
        'splits': {'train': 64, 'val': 16, 'test': 20},
        'train_samples': {'2': 64, '1': 951, '1.5': 1973},
        'cache_counts': {'2': [64, 16, 20], '1': [951, 241, 294], '1.5': [1973, 497, 606]},
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes(expected):
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen input/source changed: ' + str(path))


def validate_manifest(name, manifest):
    spec = SPECS[name]
    assert manifest['dataset'] == name and manifest['schema_version'] == 1
    assert manifest['source_sha256'] == spec['raw_sha256']
    assert manifest['settings'] == {
        'resolution': float(spec['resolution']), 'split_seed': 1234,
        'louvain_seed': 0, 'weight': None, 'node_labels': 'dummy',
        'graph_condition': 'constant', 'split_protocol': spec['split_protocol'],
        'hierarchy_policy': 'raw_louvain_no_merge_no_drop_v1'}
    assert manifest['validation_role'] == 'independent_held_out'
    assert manifest['source_graphs_dropped'] == []
    ids = [r['id'] for r in manifest['graphs']]
    selected = [i for split in ('train', 'val', 'test') for i in manifest['splits'][split]]
    assert sorted(ids) == sorted(selected) and len(set(selected)) == len(selected)
    assert {s: len(v) for s, v in manifest['splits'].items()} == spec['splits']
    assert len(ids) == sum(spec['splits'].values())
    if name == 'tree':
        assert manifest['splits'] == {'train': list(range(128)), 'val': list(range(128, 160)),
                                      'test': list(range(160, 200))}
    for record in manifest['graphs']:
        parts = manifest['partitions'][str(record['id'])]
        assert 3 <= len(parts) <= spec['max_size']
        assert all(2 <= len(part) <= spec['max_size'] for part in parts)
        assert sorted(n for part in parts for n in part) == list(range(record['n']))


def get_item(root, name):
    root = Path(root).resolve()
    setup = json.loads((root / SETUP).read_text(encoding='utf-8'))
    assert setup['root'] == str(root)
    item = setup['items'][name]
    assert item['manifest_sha256'] == SPECS[name]['manifest_sha256']
    assert sha(item['manifest']) == item['manifest_sha256']
    assert item['resolution'] == SPECS[name]['resolution']
    assert item['max_size'] == SPECS[name]['max_size']
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
