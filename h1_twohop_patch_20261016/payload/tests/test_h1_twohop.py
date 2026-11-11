"""No dataset/checkpoint downloads: H1 two-hop compatibility and numerical tests."""

import copy
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dgd'))

from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
from models.h1_twohop_transformer import H1TwoHopGraphTransformer
from models.h1_twohop_warm_start import warm_start_original_h1
from models.transformer_model import GraphTransformer
from models.two_hop_edge import TwoHopEdgeResidual, two_hop_options
from test_beta_h1_semantics import _cfg, _DatasetInfo


def cases(name, values):
    def decorate(function):
        function._cases = (name, values)
        return function
    return decorate


def make_model(enabled=False):
    cfg = _cfg()
    if enabled:
        cfg.model.two_hop = dict(enabled=True, channels=4, warm_start_checkpoint=None)
    return ConditionalGraphBetaDiffusionH1(cfg, _DatasetInfo()).eval()


def graph_inputs(n=5):
    mask = torch.tensor([[True] * (n - 1) + [False], [True] * n])
    X = torch.randn(2, n, 1) * mask[..., None]
    E = torch.randn(2, n, n, 1)
    E = (E + E.transpose(1, 2)) / 2
    E *= (mask[:, :, None] & mask[:, None, :])[..., None]
    E[:, torch.arange(n), torch.arange(n)] = 0
    return X, E, torch.tensor([[1., 0.], [0., 1.]]), mask


@cases('value', [dict(enabled='false'), dict(channels=0),
                dict(channels=True), dict(unknown=3),
                dict(warm_start_checkpoint='/tmp/old.ckpt')])
def test_invalid_configuration_fails(value):
    with unittest.TestCase().assertRaises((TypeError, ValueError)):
        two_hop_options(value)


def test_disabled_model_uses_unchanged_backbone_and_strict_old_state():
    model = make_model()
    assert type(model.model) is GraphTransformer
    assert not any('two_hop' in key for key in model.state_dict())
    clone = make_model()
    clone.load_state_dict(model.state_dict(), strict=True)
    X, E, y, mask = graph_inputs()
    t = torch.full((2, 1), .5)
    eta_X, eta_E = model._eta_tensors(2, 5, X)
    for a, b in zip(model._predict_clean(X, E, y, t, model._alpha(t), eta_X, eta_E, mask),
                    clone._predict_clean(X, E, y, t, clone._alpha(t), eta_X, eta_E, mask)):
        assert torch.equal(a, b)


def test_zero_initialized_variant_matches_original_predictions():
    base, variant = make_model(), make_model(True)
    missing = variant.load_state_dict(base.state_dict(), strict=False)
    assert missing.missing_keys and not missing.unexpected_keys
    assert all(k.startswith('model.two_hop_edge.') for k in missing.missing_keys)
    assert isinstance(variant.model, H1TwoHopGraphTransformer)
    X, E, y, mask = graph_inputs()
    t = torch.full((2, 1), .5)
    eta_X, eta_E = base._eta_tensors(2, 5, X)
    base_out = base._predict_clean(X, E, y, t, base._alpha(t), eta_X, eta_E, mask)
    new_out = variant._predict_clean(X, E, y, t, variant._alpha(t), eta_X, eta_E, mask)
    assert all(torch.equal(a, b) for a, b in zip(base_out, new_out))
    with torch.no_grad():
        variant.model.two_hop_edge.output[-1].weight.fill_(.05)
    changed = variant._predict_clean(X, E, y, t, variant._alpha(t), eta_X, eta_E, mask)
    assert not torch.equal(base_out[1], changed[1])


def test_trained_residual_symmetry_mask_padding_and_permutation():
    block = TwoHopEdgeResidual(8, 5, channels=4).double()
    with torch.no_grad():
        block.output[-1].weight.normal_(std=.1)
    mask = torch.tensor([[True, True, True, False, False]])
    E = torch.randn(1, 5, 5, 8, dtype=torch.float64)
    E = (E + E.transpose(1, 2)) / 2
    E[:, torch.arange(5), torch.arange(5)] = 0
    valid = mask[:, :, None] & mask[:, None, :]
    E *= valid[..., None]
    y = torch.randn(1, 5, dtype=torch.float64)
    out = block(E, y, mask)
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-12)
    assert not out.masked_select(~valid[..., None]).any()
    assert not out[:, torch.arange(5), torch.arange(5)].any()
    permutation = torch.tensor([3, 2, 0, 4, 1])
    permuted = block(E[:, permutation][:, :, permutation], y, mask[:, permutation])
    assert torch.allclose(permuted, out[:, permutation][:, :, permutation], atol=1e-12)
    small = block(E[:, :3, :3], y, mask[:, :3])
    assert torch.allclose(small, out[:, :3, :3], atol=1e-12)


@cases('n', [1, 2, 5])
def test_finite_gradients_for_singleton_small_graphs(n):
    block = TwoHopEdgeResidual(8, 5, 4)
    mask = torch.ones(2, n, dtype=torch.bool)
    E = torch.zeros(2, n, n, 8, requires_grad=True)
    out = block(E, torch.randn(2, 5), mask)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(E.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
    if n > 1:
        assert block.output[-1].weight.grad.abs().sum() > 0


def test_enabled_sampling_keeps_schema_fixed_x_y_and_singletons():
    model = make_model(True)
    y = torch.tensor([[1., 0.], [0., 1.]])
    saved = y.clone()
    graphs, nx_graphs = model.sampling_batch_h1(
        0, 2, 0, 1, num_nodes=torch.tensor([1, 4]), y=y, figures=False)
    assert torch.equal(y, saved)
    for n, (X, E), graph in zip([1, 4], graphs, nx_graphs):
        assert X.dtype == torch.long and X.shape == (n,)
        assert E.dtype == torch.long and E.shape == (n, n)
        assert torch.equal(E, E.T) and not E.diagonal().any()
        assert set(E.unique().tolist()) <= {0, 1}
        assert graph.number_of_nodes() == n


def test_warm_start_checks_config_priors_and_original_weight_identity(tmp_path):
    original = make_model()
    cfg = copy.deepcopy(original.cfg)
    source = tmp_path / 'original.ckpt'
    torch.save(dict(state_dict=original.state_dict(), model_family='beta_h1',
                    hyper_parameters=dict(cfg=cfg), epoch=10, global_step=30), source)
    digest_before = source.read_bytes()
    variant = make_model(True)
    audit = warm_start_original_h1(variant, source)
    assert audit['epoch'] == 10 and not audit['optimizer_restored']
    assert source.read_bytes() == digest_before
    for key, value in original.state_dict().items():
        assert torch.equal(value, variant.state_dict()[key])
    variant.cfg.model.edge_threshold = .4
    with unittest.TestCase().assertRaisesRegex(ValueError, 'config mismatch'):
        warm_start_original_h1(variant, source)


@cases('classes', [4, 7])
def test_multiclass_node_denoising_loss_and_gradient(classes):
    info = _DatasetInfo()
    info.input_dims = dict(X=classes, E=2, y=classes + 1)
    info.output_dims = dict(X=classes, E=2, y=classes)
    info.node_types = torch.ones(classes) / classes
    cfg = _cfg()
    cfg.model.two_hop = dict(enabled=True, channels=4, warm_start_checkpoint=None)
    model = ConditionalGraphBetaDiffusionH1(cfg, info)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Data(x=torch.eye(classes)[[0, 1, 0]], edge_index=edges,
                 edge_attr=torch.tensor([[0., 1.]]).repeat(4, 1),
                 y=torch.eye(classes)[[0]])
    result = model._shared_step(Batch.from_data_list([graph, graph]), 'train')
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    gradient = model.model.two_hop_edge.output[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_one_epoch_synthetic_train_val_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1234)
    model = make_model(True)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Data(x=torch.ones(3, 1), edge_index=edges,
                 edge_attr=torch.tensor([[0., 1.]]).repeat(4, 1),
                 y=torch.tensor([[1., 0.]]))
    loader = DataLoader([graph] * 4, batch_size=2, collate_fn=Batch.from_data_list)
    callback = ModelCheckpoint(dirpath=tmp_path, monitor='val/h1_total',
                               mode='min', save_top_k=1, save_last=True)
    trainer = Trainer(accelerator='cpu', devices=1, max_epochs=1,
                      limit_train_batches=2, limit_val_batches=2,
                      num_sanity_val_steps=0, logger=False,
                      enable_progress_bar=False, callbacks=[callback])
    trainer.fit(model, loader, loader)
    assert torch.isfinite(trainer.callback_metrics['train/h1_total'])
    assert torch.isfinite(trainer.callback_metrics['val/h1_total'])
    print('[two-hop synthetic smoke] train/h1_total=',
          float(trainer.callback_metrics['train/h1_total']),
          'val/h1_total=', float(trainer.callback_metrics['val/h1_total']))
    assert Path(callback.best_model_path).is_file()
    assert Path(callback.last_model_path).is_file()
    saved = torch.load(callback.last_model_path, map_location='cpu')
    assert saved['model_family'] == 'beta_h1' and saved['stage'] == 'h1'
    assert saved['h1_two_hop']['channels'] == 4
    restored = ConditionalGraphBetaDiffusionH1.load_from_checkpoint(callback.last_model_path)
    assert restored.two_hop_options['enabled']
    assert restored._two_hop_checkpoint_loaded
    restored.two_hop_options['warm_start_checkpoint'] = 'missing_original.ckpt'
    restored.on_fit_start()  # Resume/inference must not need the warm-start file.
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    functions = [(name, value) for name, value in globals().items()
                 if name.startswith('test_') and inspect.isfunction(value)]
    # Also run the existing server H1 semantic regression tests unchanged.
    import test_beta_h1_semantics as original
    functions += [(name, value) for name, value in vars(original).items()
                  if name.startswith('test_') and inspect.isfunction(value)]
    for name, function in functions:
        case_name, values = getattr(function, '_cases', (None, [None]))
        for value in values:
            def run(function=function, case_name=case_name, value=value):
                kwargs = {} if case_name is None else {case_name: value}
                with tempfile.TemporaryDirectory(prefix='h1_twohop_test_') as tmp:
                    if 'tmp_path' in inspect.signature(function).parameters:
                        kwargs['tmp_path'] = Path(tmp)
                    function(**kwargs)
            description = name if case_name is None else f'{name}[{value!r}]'
            suite.addTest(unittest.FunctionTestCase(run, description=description))
    return suite


if __name__ == '__main__':
    unittest.main(verbosity=2)
