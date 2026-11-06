"""Two metric contract checks, runnable with pytest or standard unittest."""

import unittest

import numpy as np

from .metrics import concordance_index, training_mean_baseline


class MetricsTests(unittest.TestCase):
    """Compare exact CI and the mean predictor with independent calculations."""

    def test_chunked_ci_matches_double_loop(self):
        """Check 300 random rows, truth/prediction ties and degenerate subsets."""
        rng = np.random.default_rng(4001)
        y = rng.integers(0, 20, size=300)
        prediction = np.round(rng.normal(size=300), 1)
        count = 0
        credit = 0.0
        for i in range(len(y)):
            for j in range(len(y)):
                if y[i] > y[j]:
                    count += 1
                    credit += float(prediction[i] > prediction[j]) + 0.5 * float(prediction[i] == prediction[j])
        for chunk_size in (1, 37, 256, 500):
            self.assertEqual(concordance_index(y, prediction, chunk_size), credit / count)
        self.assertEqual(concordance_index(y, np.ones(300)), 0.5)
        self.assertTrue(np.isnan(concordance_index([1, 1], [0, 1])))
        self.assertTrue(np.isnan(concordance_index([], [])))

    def test_training_mean_baseline(self):
        """Check mean 3, MSE (4 + 1 + 9)/3 and CI 1/2 by hand."""
        result = training_mean_baseline([1, 3, 5], [1, 4, 6])
        self.assertEqual(result["mean"], 3.0)
        self.assertAlmostEqual(result["mse"], 14 / 3)
        self.assertEqual(result["ci"], 0.5)
