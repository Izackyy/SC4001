"""Three DANN contract checks using real processed data and unittest."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from .dann import lambda_schedule, reverse_gradient
from .data import processed_csv
from .train_dann import DANNConfig, load_target_domain


class DANNTests(unittest.TestCase):
    """Verify reversal, scheduling and the actual unlabelled stream's isolation."""

    def test_gradient_reversal(self):
        """Assert identity forward plus the sign and scale of encoder gradients."""
        plain = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        reversed_input = plain.detach().clone().requires_grad_()
        weights = torch.tensor([2.0, -3.0, 4.0])
        coefficient = 0.3
        (plain * weights).sum().backward()
        output = reverse_gradient(reversed_input, coefficient)
        torch.testing.assert_close(output, plain)
        (output * weights).sum().backward()
        torch.testing.assert_close(reversed_input.grad, -coefficient * plain.grad)
        self.assertTrue(torch.all(reversed_input.grad * plain.grad < 0).item())
        torch.testing.assert_close(reversed_input.grad.abs(), coefficient * plain.grad.abs())

    def test_lambda_schedule(self):
        """Check the exact origin, monotonicity and bound at eleven points."""
        for maximum in (0.0, 0.1, 0.5, 1.0):
            values = [lambda_schedule(progress, maximum) for progress in np.linspace(0, 1, 11)]
            self.assertEqual(values[0], 0.0)
            self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
            self.assertTrue(all(0 <= value <= maximum for value in values))

    def test_leakage_guard(self):
        """Check each real stream against its held-out test and both external audits."""
        data_dir = Path(__file__).resolve().parents[1] / "data/processed"
        all_test_sequences = set()
        for dataset in ("davis", "kiba"):
            with self.subTest(dataset=dataset):
                targets = pd.read_csv(processed_csv(data_dir, f"{dataset}_targets"),
                                      usecols=["target_id", "sequence"], dtype={"target_id": str})
                pairs = pd.read_csv(processed_csv(data_dir, f"{dataset}_pairs"),
                                    usecols=["target_id", "split_cold_target"], dtype={"target_id": str})
                test_ids = set(pairs.loc[pairs.split_cold_target == "test", "target_id"])
                test_sequences = set(targets.loc[targets.target_id.isin(test_ids), "sequence"])
                all_test_sequences.update(test_sequences)
                tokens, stream = load_target_domain(DANNConfig(dataset=dataset), data_dir)
                self.assertEqual(len(tokens), len(stream))
                self.assertTrue(set(stream.sequence).isdisjoint(test_sequences))
                val_ids = set(pairs.loc[pairs.split_cold_target == "val", "target_id"])
                val_sequences = set(targets.loc[targets.target_id.isin(val_ids), "sequence"])
                self.assertEqual(set(stream.loc[stream.origin == "validation", "sequence"]), val_sequences)
                external = stream.loc[stream.origin == "external"]
                self.assertEqual(len(external), 114)
                self.assertTrue(external.max_test_jaccard.between(0, 0.90, inclusive="left").all())
        self.assertTrue(set(external.sequence).isdisjoint(all_test_sequences))


if __name__ == "__main__":
    unittest.main()
