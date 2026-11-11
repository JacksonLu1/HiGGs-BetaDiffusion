"""CPU-only configuration tests; no model training or server access."""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parent
TOOLS = PACKAGE / 'payload/tools'
sys.path.insert(0, str(TOOLS))
import tg_beta_train as driver


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='config_check_', dir=PACKAGE)
        cls.root = Path(cls.temporary.name)
        checkout = PACKAGE.parents[1] / 'HiGGs-main'
        shutil.copytree(checkout / 'configs', cls.root / 'configs')
        original = Path('C:/path/to/HiGGs-main1/HiGGs-main')
        for name in ('tree', 'grid'):
            shutil.copy2(original / 'deliverables/tree_grid_partition_20261028/files/configs/dataset' / (name + '.yaml'),
                         cls.root / 'configs/dataset' / (name + '.yaml'))
        hog = PACKAGE.parent / 'h1_twohop_mean_hog_rw_20261022/payload/configs/model'
        for source in hog.glob('*.yaml'):
            shutil.copy2(source, cls.root / 'configs/model' / source.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_dataset_stage_mode_configs(self):
        for dataset in ('tree', 'grid'):
            item = {
                'manifest': str(self.root / (dataset + '.json')),
                'source': str(self.root / (dataset + '.pkl')),
                'datadir': str(self.root / 'data' / dataset),
                'splits': {'test': 40 if dataset == 'tree' else 20},
                'resolution': 1 if dataset == 'tree' else 2,
                'max_size': 64 if dataset == 'tree' else 361,
            }
            for stage in ('h2', 'h1', 'glue'):
                for mode in ('smoke', 'train'):
                    with self.subTest(dataset=dataset, stage=stage, mode=mode):
                        cfg = driver.make_config(self.root, dataset, stage, mode,
                                                 self.root / 'outputs' / dataset, item)
                        self.assertEqual(cfg.dataset.name, dataset)
                        self.assertEqual(float(cfg.dataset.h), {'h2': 2., 'h1': 1., 'glue': 1.5}[stage])
                        self.assertEqual(cfg.train.seed, 1234)
                        self.assertEqual(cfg.train.batch_size, 32)
                        self.assertEqual(cfg.train.n_epochs, 1 if mode == 'smoke' else 500)
                        self.assertEqual(cfg.model.diffusion_steps, 400 if stage == 'glue' else 100)
                        self.assertEqual(cfg.general.check_val_every_n_epochs, 1 if mode == 'smoke' else 10)
                        self.assertIsNone(cfg.general.resume)
                        self.assertIsNone(cfg.general.test_only)
                        self.assertFalse(cfg.general.run_test_after_fit)
                        if stage == 'h1':
                            self.assertEqual(cfg.model.two_hop.normalization, 'mean')
                            self.assertTrue(cfg.model.hog_rw.enabled)
                        elif stage == 'glue':
                            self.assertEqual(cfg.model.family, 'original_glue')


if __name__ == '__main__':
    unittest.main(verbosity=2)


