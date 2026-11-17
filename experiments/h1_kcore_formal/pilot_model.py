"""Runtime-only pilot adapter. Never patches the installed H1 source."""
import copy
import hashlib

import torch
from omegaconf import OmegaConf, open_dict

from common import require
from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
from diffusion.h1_kcore_features import masked_h1_kcore_features

ARMS = ('baseline', 'kcore_tau020')
PROTOCOL = 'sbm-h1-mean-hog-kcore-paired-pilot-v1'


class PilotH1(ConditionalGraphBetaDiffusionH1):
    def __init__(self, cfg, dataset_infos, train_metrics=None, sampling_metrics=None,
                 visualization_tools=None, extra_features=None, domain_features=None):
        raw = cfg.model.get('pilot_feature_threshold')
        enabled = bool(cfg.model.h1_kcore.enabled)
        require((enabled and type(raw) is float and raw == .2) or
                (not enabled and raw is None), 'Invalid pilot feature threshold/config')
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics,
                         visualization_tools, extra_features, domain_features)
        self.pilot_tau = raw
        require(self.edge_threshold == .5 and self.T == 100, 'Output threshold/T changed')
        require(self.kcore_options == {'enabled': enabled, 'k': 3, 'max_t': .5},
                'Unexpected k-core settings')

    def _extra_data(self, X_t, E_t, condition_y, t, node_mask):
        # Parent computes all original features with .5, including HOG. Replace
        # ONLY its appended k-core columns; no model-wide threshold mutation.
        X, E, y = super()._extra_data(X_t, E_t, condition_y, t, node_mask)
        if not self.kcore_options['enabled']:
            return X, E, y
        score = self._state_to_unit(E_t, self.edge_scale, self.edge_shift)[..., 0]
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        upper = torch.triu(valid, diagonal=1)
        selected = (score > self.pilot_tau) & upper
        adjacency = (selected | selected.transpose(1, 2)).to(score)
        cx, ce = masked_h1_kcore_features(adjacency, node_mask, t, k=3, max_t=.5)
        return torch.cat((X[..., :-2], cx.to(X)), -1), torch.cat((E[..., :-1], ce.to(E)), -1), y

    def _kcore_checkpoint_metadata(self):
        metadata = super()._kcore_checkpoint_metadata()
        # A vanilla .5-kcore loader must reject this checkpoint, not silently
        # interpret it with the wrong feature graph threshold.
        metadata['feature_edge_threshold'] = float(self.cfg.model.pilot_feature_threshold)
        return metadata

    def _pilot_metadata(self):
        return {'version': 1, 'protocol': PROTOCOL,
                'arm': 'kcore_tau020' if self.kcore_options['enabled'] else 'baseline',
                'feature_threshold': self.pilot_tau, 'output_threshold': .5,
                'hog_structural_threshold': .5}

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['h1_pilot'] = self._pilot_metadata()

    def on_load_checkpoint(self, checkpoint):
        require(checkpoint.get('h1_pilot') == self._pilot_metadata(), 'Pilot metadata mismatch')
        saved = checkpoint['hyper_parameters']['cfg']
        require(OmegaConf.select(saved, 'model.pilot_feature_threshold') == self.pilot_tau,
                'Saved pilot feature threshold mismatch')
        super().on_load_checkpoint(checkpoint)


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def build_pair(before, out, epochs=50, val_every=5):
    """Reference supplies configuration/priors ONLY, never trained weights."""
    checkpoint = torch.load(before['checkpoint'], map_location='cpu')
    require(checkpoint.get('model_family') == 'beta_h1' and checkpoint.get('epoch') == 449
            and 'h1_kcore' not in checkpoint, 'Expected frozen old HOG best449 metadata')
    hp = checkpoint['hyper_parameters']
    cfg = copy.deepcopy(hp['cfg'])
    requirements = {
        'dataset.name': 'sbm', 'dataset.h': 1, 'dataset.resolution': 1,
        'dataset.max_size': 500, 'dataset.regime': 'majority',
        'train.batch_size': 8, 'train.seed': 0, 'train.lr': .0002,
        'train.weight_decay': 1e-12, 'train.amsgrad': True, 'train.optimizer': 'adamw',
        'train.clip_grad': None, 'train.ema_decay': 0,
        'model.two_hop.enabled': True, 'model.two_hop.normalization': 'mean',
        'model.two_hop.channels': 16, 'model.two_hop.warm_start_checkpoint': None,
        'model.hog_rw.enabled': True, 'model.hog_rw.rw_depth': 4,
        'model.pre_condition': False, 'model.eta_strategy': 'fixed',
        'model.edge_threshold': .5, 'model.diffusion_steps': 100,
        'general.resume': None, 'general.test_only': None,
    }
    for key, expected in requirements.items():
        require(OmegaConf.select(cfg, key) == expected, f'Reference config differs: {key}')
    require(checkpoint['h1_two_hop']['normalization'] == 'mean' and
            checkpoint['h1_hog_rw']['rw_depth'] == 4, 'Reference metadata differs')
    models = {}
    for arm in ARMS:
        c = copy.deepcopy(cfg)
        with open_dict(c):
            c.general.name = 'sbm_' + arm + '_paired_pilot'
            c.general.wandb = 'disabled'
            c.general.run_test_after_fit = False
            c.general.check_val_every_n_epochs = val_every
            c.train.n_epochs = epochs
            c.train.num_workers = 0
            c.train.seed = 0
            c.dataset.datadir = str(__import__('pathlib').Path(before['baseline']) / 'data/sbm')
            c.model.h1_kcore = {'enabled': arm != 'baseline', 'k': 3, 'max_t': .5}
            c.model.pilot_feature_threshold = .2 if arm != 'baseline' else None
        kwargs = {k: copy.deepcopy(hp.get(k)) for k in
                  ('dataset_infos', 'extra_features', 'domain_features')}
        require(kwargs['dataset_infos'] is not None and
                getattr(kwargs['dataset_infos'], 'datamodule', None) is None,
                'Unexpected saved data module')
        models[arm] = PilotH1(cfg=c, visualization_tools=None, **kwargs)
        widths = {k: getattr(models[arm].model, 'mlp_in_' + k)[0].in_features
                  for k in ('X', 'E', 'y')}
        require(widths == ({'X': 12, 'E': 6, 'y': 14} if arm == 'baseline' else
                           {'X': 14, 'E': 7, 'y': 14}), 'Unexpected input widths')
        arm_out = out / arm
        arm_out.mkdir(exist_ok=False)
        OmegaConf.save(c, arm_out / 'effective_config.yaml')
    del checkpoint
    copied = initialize_shared(models['baseline'], models['kcore_tau020'])
    return models, copied


def initialize_shared(base, candidate):
    source, destination = base.state_dict(), candidate.state_dict()
    require(source.keys() == destination.keys(), 'Shared state keys differ')
    extras = {'model.mlp_in_X.0.weight': 2, 'model.mlp_in_E.0.weight': 1}
    with torch.no_grad():
        for key, tensor in source.items():
            target = destination[key]
            if key in extras:
                require(target.shape == (tensor.shape[0], tensor.shape[1] + extras[key]),
                        'Unexpected added input columns: ' + key)
                target.zero_()
                target[:, :tensor.shape[1]].copy_(tensor)
            else:
                require(tensor.shape == target.shape, 'Unexpected state shape: ' + key)
                target.copy_(tensor)
    candidate.load_state_dict(destination, strict=True)
    for key, tensor in source.items():
        actual = candidate.state_dict()[key]
        if key in extras:
            require(torch.equal(tensor, actual[:, :tensor.shape[1]]) and
                    int(torch.count_nonzero(actual[:, tensor.shape[1]:])) == 0,
                    'New feature initialization failed')
        else:
            require(torch.equal(tensor, actual), 'Shared initial tensor differs: ' + key)
    return {'all_shared_tensors_equal': True, 'new_feature_columns_zero': True,
            'expanded_weights': extras, 'old_weights_loaded': False,
            'baseline_initial_state_sha256': state_hash(base),
            'candidate_initial_state_sha256': state_hash(candidate)}


def check_initial_predictions(base, candidate):
    """Ragged padded current states, active/inactive gates; no clean target."""
    from diffusion.beta_diffusion_utils import unit_probability_to_shifted_state
    base.eval()
    candidate.eval()
    differences = []
    with torch.no_grad():
        mask = torch.tensor([[True]*5, [True]*3 + [False]*2])
        # Scores in (.2,.5) ensure the two structural thresholds really differ.
        score = torch.full((2, 5, 5, 1), .35)
        score = score * (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1)
        score[:, torch.arange(5), torch.arange(5), :] = 0
        E = unit_probability_to_shifted_state(score, base.edge_scale, base.edge_shift)
        X = unit_probability_to_shifted_state(torch.ones(2, 5, 1), base.node_scale, base.node_shift)
        y = torch.tensor([[1., 0.], [1., 0.]])
        for time in (.25, .45, .75):
            t = torch.full((2, 1), time)
            bx, be, by = base._extra_data(X, E, y, t, mask)
            cx, ce, cy = candidate._extra_data(X, E, y, t, mask)
            require(torch.equal(bx, cx[..., :-2]) and torch.equal(be, ce[..., :-1])
                    and torch.equal(by, cy), 'Original/HOG features changed')
            if time <= .5:
                require(bool(cx[0, :, -1].all()) and bool(ce[0, 0, 1, -1]),
                        '.20 k-core features not active on score .35')
            else:
                require(not bool(cx[..., -2:].any()) and not bool(ce[..., -1:].any()),
                        'Inactive k-core gate is not zero')
            require(not bool(cx[1, 3:, -2:].any()) and not bool(ce[1, 3:, :, -1:].any()),
                    'Padding leaked into features')
            alpha = base._alpha(t)
            eta_X, eta_E = base._eta_tensors(2, 5, X)
            fixed = torch.ones_like(X) * mask.unsqueeze(-1)
            a = base._predict_clean(X, E, y, t, alpha, eta_X, eta_E, mask, fixed_X=fixed)
            b = candidate._predict_clean(X, E, y, t, alpha, eta_X, eta_E, mask, fixed_X=fixed)
            delta = max(float((u-v).abs().max()) for u, v in zip(a, b))
            require(delta <= 1e-6, f'Initial predictions not equivalent: {delta}')
            differences.append(delta)
    return {'initial_prediction_max_abs_diff': max(differences),
            'original_hog_features_unchanged': True, 'padding_and_gate_checked': True}
