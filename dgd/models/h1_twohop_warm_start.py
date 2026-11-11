"""Read-only, weights-only warm start with a fresh optimizer for H1 two-hop."""

import hashlib
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def warm_start_original_h1(model, checkpoint_path):
    if not model.two_hop_options['enabled']:
        raise ValueError("Warm start is only available for the H1 two-hop variant")
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    # Reuse the server's centralized Config*Logger pickle compatibility.
    from higgs.higgs_utils import _load_checkpoint_cpu
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = _load_checkpoint_cpu(path)
    source_cfg = checkpoint['hyper_parameters']['cfg']
    if not OmegaConf.is_config(source_cfg):
        source_cfg = OmegaConf.create(source_cfg)
    family = checkpoint.get('model_family', OmegaConf.select(source_cfg, 'model.family'))
    if family != 'beta_h1' or float(OmegaConf.select(source_cfg, 'dataset.h')) != 1:
        raise ValueError("Warm-start checkpoint must be a Beta-H1 checkpoint")
    keys = (
        'dataset.name', 'dataset.h', 'dataset.resolution', 'dataset.regime',
        'dataset.max_size', 'train.batch_size', 'model.diffusion_steps', 'model.input_space',
        'model.pre_condition', 'model.sigmoid_start', 'model.sigmoid_end',
        'model.sigmoid_power', 'model.eta', 'model.eta_strategy',
        'model.scale_shift', 'model.edge_threshold', 'model.beta_eps',
        'model.beta_loss', 'model.lambda_train', 'model.n_layers',
        'model.hidden_mlp_dims', 'model.hidden_dims', 'model.extra_features',
    )
    def plain(cfg, key):
        value = OmegaConf.select(cfg, key)
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    differences = {key: (plain(source_cfg, key), plain(model.cfg, key))
                   for key in keys if plain(source_cfg, key) != plain(model.cfg, key)}
    if differences:
        raise ValueError(f"Warm-start config mismatch (source, requested): {differences}")
    source = checkpoint['state_dict']
    target = model.state_dict()
    adapter_keys = {key for key in target if key.startswith('model.two_hop_edge.')}
    if not adapter_keys or set(source) != set(target) - adapter_keys:
        raise ValueError("Expected an original H1 state dict; only new two-hop keys may be missing")
    for key, tensor in source.items():
        if tensor.shape != target[key].shape:
            raise ValueError(f"Warm-start tensor shape mismatch: {key}")
        if key in ('node_channel_prior', 'edge_channel_prior'):
            if not torch.equal(tensor.cpu(), target[key].cpu()):
                raise ValueError(f"Warm-start train-derived prior differs: {key}")
    result = model.load_state_dict(source, strict=False)
    if set(result.missing_keys) != adapter_keys or result.unexpected_keys:
        raise RuntimeError(f"Unexpected warm-start state mismatch: {result}")
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return dict(checkpoint=str(path), sha256=digest.hexdigest(),
                epoch=checkpoint.get('epoch'), global_step=checkpoint.get('global_step'),
                optimizer_restored=False, new_parameter_tensors=len(adapter_keys))
