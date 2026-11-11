import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1]).resolve()
assert str(root) == "/path/to/HiGGs-GBD/HiGGs-main"
main = root / "dgd/main.py"
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert sha(main) == "f1cc8f14964b234f84845b5cb64ddb0a4a8fae56bd0e06687ac72b46d44b7dc7", \
    "main.py differs or patch already installed; stop, do not force overwrite"

protected = {
    "dgd/diffusion_model_beta_h2.py": "a0b895b21afe82d8335c6b77acafcaf18f1adc9fc734c0d83149a5b3babfabe6",
    "dgd/diffusion_model_beta_h1.py": "b4d6078601afda3fb46ea40c0b9687d949898605dd29f5949d9eb65239d426a3",
    "dgd/diffusion_model_discrete.py": "fd477d76e959fca43070b0a9a36e3aa5a0916a933cf8aecbd7a74ddac4643f70",
    "dgd/diffusion/distributions.py": "ce79c6f1addca9d72b04dde0a694ea8872690963b892c9c8c8251688d2584e00",
    "higgs/higgs_utils.py": "6934a9d05d08818119ee97e64b846a9d6b59a1afc800b0dfd3966c53a049f648",
    "higgs/sampling.py": "4f770fd6df760b9a071b7fc2402e37dae1312bb94eb0b649b54a31fc6e0bc345",
}
assert all(sha(root / p) == digest for p, digest in protected.items()), \
    "A protected source changed since the audit"

text = main.read_text(encoding="utf-8")
replacements = [
("elif dataset_config[\"name\"] in [\"fb_hierarchies\", \"shetland\", \"cora\", \"sbm\"]:", "elif dataset_config[\"name\"] in [\"fb_hierarchies\", \"shetland\", \"cora\", \"sbm\", \"planar\"]:"),
("        if dataset_config[\"name\"] == \"fb_hierarchies\":", "        if dataset_config[\"name\"] == \"planar\":\n            from datasets.graph_collection import GraphCollectionDataModule\n            from datasets.planar_beta_adapter import PlanarBetaDatasetInfos\n            datamodule = GraphCollectionDataModule(cfg)\n            datamodule.prepare_data()\n            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)\n            dataset_infos = PlanarBetaDatasetInfos(datamodule, dataset_config)\n\n        elif dataset_config[\"name\"] == \"fb_hierarchies\":"),
("        checkpoint_callback = ModelCheckpoint(", "        if cfg.dataset.name == \"planar\":\n            checkpoint_every_n_epochs = int(cfg.general.get(\n                \"checkpoint_every_n_epochs\", checkpoint_every_n_epochs))\n            if checkpoint_every_n_epochs < 1:\n                raise ValueError(\"Checkpoint cadence must be positive\")\n        checkpoint_callback = ModelCheckpoint("),
]
for old, new in replacements:
    assert text.count(old) == 1, "Non-unique source anchor: " + old
    text = text.replace(old, new)
compile(text, str(main), "exec")

payload = Path(__file__).resolve().parent / "payload"
adapter = (payload / "dgd/datasets/planar_beta_adapter.py").read_text(encoding="utf-8")
smoke = (payload / "tools/diagnostics/planar_beta_smoke_20261026.py").read_text(encoding="utf-8")
new_files = {
    root / "dgd/datasets/planar_beta_adapter.py": adapter,
    root / "tools/diagnostics/planar_beta_smoke_20261026.py": smoke,
}
for path, content in new_files.items():
    compile(content, str(path), "exec")
    assert not path.exists(), "New file already exists; stop: " + str(path)

backup = root / "interface_patch_backups" / (
    "planar_beta_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
backup.mkdir(parents=True)
shutil.copy2(main, backup / "main.py.before")
(backup / "protected_hashes.json").write_text(json.dumps(protected, indent=2))
created = []
try:
    for path, content in new_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            created.append(path)
            handle.write(content)
    main.write_text(text, encoding="utf-8")
    assert all(sha(root / p) == digest for p, digest in protected.items())

    sys.path[:0] = [str(root / "dgd"), str(root)]
    import torch
    from datasets.planar_beta_adapter import PlanarDistributionNodes
    d = PlanarDistributionNodes(torch.tensor([0., .25, .75], dtype=torch.float64))
    before = d.prob.clone()
    sampler = d.m.probs.clone()
    torch.testing.assert_close(
        d.log_prob(torch.tensor([1, 2, 3, -1])),
        torch.log(torch.tensor([.25, .75, 0., 0.], dtype=torch.float64) + 1e-30))
    torch.testing.assert_close(d.prob, before)
    torch.testing.assert_close(d.m.probs, sampler)
    assert d.log_prob(torch.empty(0, dtype=torch.long)).shape == (0,)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        a = d.sample_n(20, "cpu")
        torch.manual_seed(1234)
        b = torch.distributions.Categorical(sampler).sample((20,))
        torch.testing.assert_close(a, b)
except BaseException:
    shutil.copy2(backup / "main.py.before", main)
    for path in reversed(created):
        assert root in path.parents and path in new_files
        if path.exists():
            path.unlink()
    print("ROLLBACK: main.py restored; only files created by this install removed.")
    print("BACKUP:", backup)
    raise
print("BACKUP:", backup)
print("PASS: PLANAR_ONLY_NODE_PRIOR_CHECKS")
print("PASS: PLANAR_BETA_ADAPTER_INSTALLED")
print("Existing Beta/HOG/Glue model and common prior sources unchanged.")

