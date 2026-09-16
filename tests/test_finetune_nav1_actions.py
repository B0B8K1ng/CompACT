import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'finetune_cache', Path(__file__).resolve().parents[1] / 'scripts/precompute_finetune_nav1_actions.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class PairCoverageTests(unittest.TestCase):
    def test_matches_training_observation_bounds(self):
        for n in (0, 4, 64, 67, 68, 69, 100, 300):
            expected = [(current, target)
                        for current in range(3, n - 64)
                        for target in range(max(0, current - 8), min(n, current + 9))]
            np.testing.assert_array_equal(cache.pairs_for(n), np.array(expected, dtype=np.int64).reshape(-1, 2))

    def test_signed_direction_and_identity_pairs(self):
        pairs = set(map(tuple, cache.pairs_for(100)))
        self.assertTrue({(12, 4), (12, 12), (12, 20)}.issubset(pairs))
        self.assertNotIn((12, 21), pairs)
        self.assertNotIn((36, 36), pairs)

    def test_chunk_remapping_preserves_pair_identity(self):
        pairs = cache.pairs_for(400)
        for offset in range(0, len(pairs), 4096):
            chunk = pairs[offset:offset+4096]
            indices, inverse = np.unique(chunk, return_inverse=True)
            np.testing.assert_array_equal(indices[inverse.reshape(-1, 2)], chunk)


if __name__ == '__main__':
    unittest.main()
