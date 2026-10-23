"""Tiny CPU tensor fixtures exercise actual ExpertPool.materialize."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FactorMergeTest(unittest.TestCase):
    def test_actual_materialization_and_quadratic_scale(self):
        import numpy as np
        import torch
        from safetensors.torch import load_file, save_file
        from alpha_space import ExpertPool

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = ('layer.lora_A.weight', 'layer.lora_B.weight')
            states = [
                {keys[0]: torch.tensor([[1., 2.]]), keys[1]: torch.tensor([[3.], [4.]])},
                {keys[0]: torch.tensor([[2., -1.]]), keys[1]: torch.tensor([[1.], [-2.]])},
            ]
            for index, state in enumerate(states):
                expert = root / 'pool' / f'expert_{index}'
                expert.mkdir(parents=True)
                save_file(state, str(expert / 'adapter_model.safetensors'))
                (expert / 'adapter_config.json').write_text(json.dumps({'r': 1, 'lora_alpha': 1}))
            pool = ExpertPool(str(root / 'pool'))
            def materialize(alpha, tag):
                path = pool.materialize(np.array(alpha), str(root / tag))
                return load_file(str(Path(path) / 'adapter_model.safetensors'))
            onehot = materialize([1., 0.], 'onehot')
            for key in keys:
                self.assertTrue(torch.equal(onehot[key], states[0][key]))
            merged = materialize([.4, .6], 'mixed')
            scaled = materialize([.8, 1.2], 'scaled')
            delta = merged[keys[1]] @ merged[keys[0]]
            self.assertTrue(torch.allclose(scaled[keys[1]] @ scaled[keys[0]], 4 * delta))
            linear = .4 * (states[0][keys[1]] @ states[0][keys[0]]) + .6 * (states[1][keys[1]] @ states[1][keys[0]])
            self.assertFalse(torch.allclose(delta, linear))
            negative = materialize([-.4, -.6], 'negative')
            self.assertTrue(torch.allclose(negative[keys[1]] @ negative[keys[0]], delta))


if __name__ == '__main__':
    unittest.main()
