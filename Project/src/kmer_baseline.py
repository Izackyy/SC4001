"""A non-neural cold-target baseline: k-mer protein similarity over a drug's own measurements.

For an unseen target t and a drug d, predict a similarity-weighted average of the
scores d was measured at against the training targets most similar to t. Protein
similarity is the cosine between amino-acid k-mer count vectors, so nothing is
learned and no GPU is used.

The estimator structurally cannot leave the range of d's observed scores and
cannot predict for a drug with no training measurements at all, which is exactly
the ceiling to report alongside its accuracy. It is included because the SC4020
session's k-mer method beat our CNN on the unseen-target views, and because a
method with no domain adaptation and no learned protein encoder is the cleanest
test of whether our cold-target deficit is representational.

Usage:
    uv run python -m src.kmer_baseline --dataset davis --split split_cold_target
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import processed_csv, resolve_split_column
from src.metrics import concordance_index, mse, training_mean_baseline

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
NEIGHBOURS = (1, 3, 5, 10, 20, 50, 100, 10**6)
SHARPENINGS = (1, 2, 4, 8, 16, 32, 64)
LEAKAGE_LIMIT = 0.95


def parse_args():
    """Parse the dataset, split and k-mer length.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("davis", "kiba"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--kmer", type=int, default=3, help="Amino-acid k-mer length.")
    parser.add_argument("--output-root", default=None,
                        help="Defaults to Project/outputs; a run writes <root>/kmer/<dataset>/<split>.")
    return parser.parse_args()


def kmer_similarity(sequences, k: int) -> np.ndarray:
    """Return the cosine similarity between L2-normalised k-mer count vectors.

    Args:
        sequences: Amino-acid strings, one per target, in table row order.
        k: K-mer length.

    Returns:
        Symmetric float array of shape (len(sequences), len(sequences)).
    """
    vocabulary = {}
    rows, columns, values = [], [], []
    for row, sequence in enumerate(sequences):
        counts = {}
        for start in range(len(sequence) - k + 1):
            counts[sequence[start:start + k]] = counts.get(sequence[start:start + k], 0) + 1
        for kmer, count in counts.items():
            rows.append(row)
            columns.append(vocabulary.setdefault(kmer, len(vocabulary)))
            values.append(count)
    matrix = np.zeros((len(sequences), len(vocabulary)), dtype=np.float32)
    matrix[rows, columns] = values
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix @ matrix.T


def predict_pairs(drug_ids, target_rows, train_by_drug, similarity, fallback,
                  neighbours: int, sharpening: int) -> np.ndarray:
    """Predict a similarity-weighted average of each drug's own training scores.

    Args:
        drug_ids: Ligand identifier per pair to predict.
        target_rows: Target table row index per pair, aligned with drug_ids.
        train_by_drug: Ligand id to (training target rows, training scores).
        similarity: Target-by-target cosine similarity.
        fallback: Prediction used for a drug with no training measurement.
        neighbours: Number of most similar training targets averaged.
        sharpening: Exponent applied to the similarity before weighting.

    Returns:
        Prediction array aligned with drug_ids.
    """
    predictions = np.full(len(drug_ids), fallback, dtype=np.float64)
    order = pd.Series(range(len(drug_ids))).groupby(pd.Series(list(drug_ids)).values).indices
    for drug, positions in order.items():
        if drug not in train_by_drug:
            continue
        train_rows, train_scores = train_by_drug[drug]
        weights = similarity[np.ix_(target_rows[positions], train_rows)].astype(np.float64)
        if neighbours < weights.shape[1]:
            cut = np.partition(weights, -neighbours, axis=1)[:, -neighbours][:, None]
            weights = np.where(weights >= cut, weights, 0.0)
        weights = np.clip(weights, 0.0, None) ** sharpening
        total = weights.sum(axis=1)
        safe = total > 0
        predictions[positions[safe]] = (weights[safe] @ train_scores) / total[safe]
        predictions[positions[~safe]] = train_scores.mean()
    return predictions


def pairwise_ci(y, prediction):
    """Return one ligand's concordance index, prediction ties counted as half.

    Args:
        y: Measured scores for a single ligand.
        prediction: Predictions aligned with y.

    Returns:
        Concordant fraction, or None when no pair has distinct truths.
    """
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    better = y[:, None] > y[None, :]
    if not better.any():
        return None
    gap = prediction[:, None] - prediction[None, :]
    return float(((gap[better] > 0).sum() + 0.5 * (gap[better] == 0).sum()) / better.sum())


def main():
    """Tune on validation, evaluate on test and save one metrics file.

    Side Effects:
        Writes <output root>/kmer/<dataset>/<split>/metrics.json and prints it.

    Raises:
        ValueError: A partition is empty.
    """
    args = parse_args()
    pairs = pd.read_csv(processed_csv(DATA_DIR, f"{args.dataset}_pairs"),
                        dtype={"ligand_id": str, "target_id": str})
    targets = pd.read_csv(processed_csv(DATA_DIR, f"{args.dataset}_targets"),
                          usecols=["target_id", "sequence"], dtype={"target_id": str})
    target_row = dict(zip(targets.target_id, range(len(targets))))
    partitions = resolve_split_column(pairs, DATA_DIR, args.dataset, args.split)
    pairs["row"] = pairs.target_id.map(target_row)
    parts = {name: pairs.loc[partitions == name].copy() for name in ("train", "val", "test")}
    for name, frame in parts.items():
        if frame.empty:
            raise ValueError(f"Empty {name} partition for {args.dataset}/{args.split}.")

    similarity = kmer_similarity(targets.sequence, args.kmer)
    train_rows = np.unique(parts["train"].row.to_numpy())
    test_rows = np.unique(parts["test"].row.to_numpy())
    cross = similarity[np.ix_(test_rows, train_rows)]
    leakage = {"kmer": args.kmer, "max_test_train_similarity": float(cross.max()),
               "test_targets_above_limit": int((cross.max(axis=1) > LEAKAGE_LIMIT).sum()),
               "limit": LEAKAGE_LIMIT, "n_test_targets": len(test_rows),
               "n_train_targets": len(train_rows)}

    train_by_drug = {drug: (group.row.to_numpy(), group.score.to_numpy(dtype=np.float64))
                     for drug, group in parts["train"].groupby("ligand_id")}
    fallback = float(parts["train"].score.mean())

    best = None
    for neighbours in NEIGHBOURS:
        for sharpening in SHARPENINGS:
            prediction = predict_pairs(parts["val"].ligand_id, parts["val"].row.to_numpy(),
                                       train_by_drug, similarity, fallback, neighbours, sharpening)
            score = mse(parts["val"].score.to_numpy(), prediction)
            if best is None or score < best[0]:
                best = (score, neighbours, sharpening)
    val_mse, neighbours, sharpening = best
    print(f"Tuned on validation: neighbours={neighbours} sharpening={sharpening} "
          f"val_mse={val_mse:.4f}", flush=True)

    test = parts["test"]
    test["pred"] = predict_pairs(test.ligand_id, test.row.to_numpy(), train_by_drug,
                                 similarity, fallback, neighbours, sharpening)
    covered = test.ligand_id.isin(train_by_drug)
    y, prediction = test.score.to_numpy(), test.pred.to_numpy()
    masks = {"all": np.ones(len(test), dtype=bool)}
    if args.dataset == "davis":
        masks["uncensored"] = ~test.censored.to_numpy(dtype=bool)
    pooled = {name: {"n": int(mask.sum()), "mse": mse(y[mask], prediction[mask]),
                     "ci": concordance_index(y[mask], prediction[mask])}
              for name, mask in masks.items()}

    # Within-ligand views, on the same definitions as the CNN runs.
    cis = [value for _, group in test.groupby("ligand_id")
           if (value := pairwise_ci(group.score, group.pred)) is not None]
    test_ligands = sorted(test.ligand_id.unique())
    grid_rows = np.tile(test_rows, len(test_ligands))
    grid = predict_pairs(np.repeat(test_ligands, len(test_rows)), grid_rows,
                         train_by_drug, similarity, fallback, neighbours, sharpening)
    grid = grid.reshape(len(test_ligands), len(test_rows))
    measured = test.groupby("ligand_id").score.std(ddof=1)
    matched = sorted(measured[(measured > 0) & measured.notna()].index)
    matched_positions = [test_ligands.index(drug) for drug in matched]
    predicted_spread = float(grid[matched_positions].std(axis=1, ddof=1).mean())
    measured_spread = float(measured.loc[matched].mean())

    result = {
        "method": "kmer-knn", "dataset": args.dataset, "split": args.split,
        "neighbours": neighbours, "sharpening": sharpening, "val_mse": val_mse,
        "leakage_check": leakage,
        "coverage": {"test_pairs": len(test), "pairs_with_trained_drug": int(covered.sum()),
                     "pairs_on_global_mean": int((~covered).sum()),
                     "test_ligands": len(test_ligands), "matched_ligands": len(matched)},
        "test": pooled,
        "training_mean_baseline": {"test": {name: {"n": int(mask.sum()),
                                                   **training_mean_baseline(parts["train"].score, y[mask])}
                                            for name, mask in masks.items()}},
        "within_ligand": {"ci": float(np.mean(cis)), "n_ligands_ranked": len(cis),
                          "predicted_spread": predicted_spread,
                          "measured_spread": measured_spread,
                          "dispersion_ratio": predicted_spread / measured_spread,
                          "basis": "matched ligand set: >=2 measured test pairs and non-zero measured spread"},
    }
    root = Path(args.output_root or Path(__file__).resolve().parents[1] / "outputs")
    directory = root / "kmer" / args.dataset / args.split
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (directory / "metrics.json").write_text(text, encoding="utf-8")
    print(f"Saved {directory / 'metrics.json'}", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
