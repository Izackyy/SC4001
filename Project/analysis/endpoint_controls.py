"""Protein-blind predictors scored against standard drug-target affinity endpoints.

Two endpoints in common use for this task can be beaten, or brought to within a rounding error of
a trained model, by a predictor carrying no information about which unseen protein binds which
drug. A third resists. Each control below is protein-blind by construction, so whatever it scores
is the floor an arm must clear before its score means anything. All figures are on the DAVIS
cold-target test partition, s4001.

  1. Pooled concordance index. A predictor that is CONSTANT WITHIN EACH DRUG cannot order targets
     at all, yet pooled CI reads 0.7633 against the trained CNN's 0.7885. It does not beat the
     CNN; it comes within 0.0252, because most test pairs compare different drugs and so are
     ordered by drug level alone. That margin is too small to carry a claim about proteins.
  2. Uncensored MSE, the view that excludes the pKd 5.0 assay floor. Here the per-drug constant,
     set to each drug's mean over its uncensored TRAINING pairs, beats the trained CNN outright,
     0.8113 against 1.1237. It does not beat the k-mer baseline's 0.6875.
  3. Within-ligand CI. This one is sound: every per-drug-constant control scores exactly 0.5,
     which is why the project reports it as the primary ordering metric.

The conclusion is not that these arms are good. It is that an endpoint a protein-blind predictor
can win, or very nearly match, cannot be used to argue that a model has learned anything about
unseen proteins.

Usage:
    uv run python -m analysis.endpoint_controls
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.data import processed_csv, resolve_split_column
from src.kmer_baseline import pairwise_ci
from src.metrics import concordance_index, mse

DATA = PROJECT / "data" / "processed"
# Published DAVIS cold-target test figures, for the comparison each control has to be read against.
PUBLISHED = {"CNN pooled CI": 0.7885, "k-mer pooled CI": 0.8385,
             "CNN uncensored MSE": 1.1237, "k-mer uncensored MSE": 0.6875,
             "CNN within-ligand CI": 0.6789, "k-mer within-ligand CI": 0.7494}


def within_ligand(frame: pd.DataFrame) -> tuple[float, int]:
    """Mean within-drug concordance over drugs whose test scores vary.

    Args:
        frame: Test pairs carrying ligand_id, score and pred.

    Returns:
        Tuple of (mean within-ligand CI, number of contributing drugs).
    """
    values = []
    for _, group in frame.groupby("ligand_id"):
        if group.score.std(ddof=1) > 0:
            value = pairwise_ci(group.score, group.pred)
            if value is not None:
                values.append(value)
    return float(np.mean(values)), len(values)


def report(name: str, test: pd.DataFrame, prediction: np.ndarray) -> None:
    """Print every endpoint for one control predictor.

    Args:
        name: Control description.
        test: Test pair frame carrying score and censored.
        prediction: Per-pair prediction aligned to test.
    """
    frame = test.copy()
    frame["pred"] = prediction
    keep = ~frame.censored.to_numpy(dtype=bool)
    ci, drugs = within_ligand(frame)
    print("%s" % name)
    print("  pooled CI          %.4f" % concordance_index(frame.score.to_numpy(),
                                                          frame.pred.to_numpy()))
    print("  full-prediction MSE %.4f" % mse(frame.score, frame.pred))
    print("  uncensored MSE     %.4f  over %d measured pairs" % (
        mse(frame.score.to_numpy()[keep], frame.pred.to_numpy()[keep]), int(keep.sum())))
    print("  within-ligand CI   %.4f  over %d drugs\n" % (ci, drugs))


def main():
    """Print each degenerate control against the published arm figures."""
    pairs = pd.read_csv(processed_csv(DATA, "davis_pairs"),
                        dtype={"ligand_id": str, "target_id": str})
    partitions = resolve_split_column(pairs, DATA, "davis", "split_cold_target")
    train = pairs.loc[partitions == "train"]
    test = pairs.loc[partitions == "test"].copy()

    print("DAVIS cold-target test, s4001: %d pairs, %d drugs, %d proteins, %.1f%% at the floor.\n"
          % (len(test), test.ligand_id.nunique(), test.target_id.nunique(),
             100 * test.censored.mean()))
    print("Published arm figures for comparison: " + ", ".join(
        "%s %.4f" % (key, value) for key, value in PUBLISHED.items()) + "\n")

    uncensored_train = train.loc[~train.censored.astype(bool)]
    controls = {
        "CONTROL 1  per-drug constant = that drug's mean over ALL its training pairs":
            test.ligand_id.map(train.groupby("ligand_id").score.mean()).fillna(
                float(train.score.mean())).to_numpy(),
        "CONTROL 2  per-drug constant = that drug's mean over its UNCENSORED training pairs":
            test.ligand_id.map(uncensored_train.groupby("ligand_id").score.mean()).fillna(
                float(uncensored_train.score.mean())).to_numpy(),
        "CONTROL 3  global constant = the training mean":
            np.full(len(test), float(train.score.mean())),
    }
    for name, prediction in controls.items():
        report(name, test, prediction)

    print("Readings:")
    print("  * Control 1 reaches pooled CI 0.7633 against the trained CNN's 0.7885, while being")
    print("    unable to order targets at all. The CNN's pooled margin over a predictor that")
    print("    knows nothing about proteins is +0.0252, so pooled CI cannot support a claim")
    print("    about unseen proteins.")
    print("  * Control 2's uncensored MSE beats the published CNN (1.1237) while ordering nothing,")
    print("    so uncensored MSE alone cannot either. It does not beat k-mer (0.6875).")
    print("  * Within-ligand CI is exactly 0.5000 for every control, which is why it is the")
    print("    primary ordering endpoint in this project.")


if __name__ == "__main__":
    main()
