"""CPU tests of the new runtime correction, not a real-data/server run."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_boolfix as fix
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MaskTests(unittest.TestCase):
    def test_bool_collation_values_other_fields_and_rng_preserved(self):
        samples = [Data(x=torch.ones(4, 1), y=torch.tensor([[0., 1.]]),
                        edge_index=torch.tensor([[0, 1], [1, 0]]),
                        pos=torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.],
                                          [0., 0., 1., 1.], [0., 0., 1., 1.]]))]
        loader = DataLoader(samples, batch_size=1, shuffle=False)
        result = (samples, loader, 1, 1, 0, [(0, 1)])
        mask, x, y, edges = (samples[0].pos.clone(), samples[0].x.clone(),
                             samples[0].y.clone(), samples[0].edge_index.clone())
        state = torch.get_rng_state().clone()
        stats = {'pairs_checked': 0, 'pairs_converted': 0}
        self.assertIs(fix.restore_boolean_masks(result, stats), result)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIs(loader.dataset, samples)
        self.assertEqual(next(iter(loader)).pos.dtype, torch.bool)
        self.assertTrue(torch.equal(mask, samples[0].pos.float()))
        self.assertTrue(torch.equal(samples[0].x, x))
        self.assertTrue(torch.equal(samples[0].y, y))
        self.assertTrue(torch.equal(samples[0].edge_index, edges))
        self.assertEqual(stats, {'pairs_checked': 1, 'pairs_converted': 1})

    def test_invalid_masks_refused(self):
        for mask in (torch.tensor([[0., .5]]), torch.tensor([[float('nan')]]),
                     torch.tensor([[float('inf')]]), torch.ones(3)):
            with self.subTest(mask=mask), self.assertRaises(RuntimeError):
                fix.restore_boolean_masks(([Data(pos=mask)], []))

    def test_empty_and_already_bool(self):
        result = ([], [], 1, 0, 0, [])
        self.assertIs(fix.restore_boolean_masks(result), result)
        stats = {'pairs_checked': 0, 'pairs_converted': 0}
        data = Data(pos=torch.ones(2, 2, dtype=torch.bool))
        fix.restore_boolean_masks(([data], []), stats)
        self.assertEqual(stats, {'pairs_checked': 1, 'pairs_converted': 0})

    def test_original_sampler_not_edited(self):
        source = HERE.parent / 'community_enzymes_beta_20261028/payload/tools/sc_beta_sample.py'
        self.assertEqual(fix.sha(source), fix.LEGACY_SHA)
        self.assertEqual(source.read_text().count('pos=pos.float(),'), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
