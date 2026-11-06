"""Score every trained DAVIS cold-target arm inside homology bands.

Each cold-target test protein is placed in a quartile of its maximum amino-acid 3-mer cosine to
any training protein, and every already-trained arm is re-scored inside each band from its saved
checkpoints. The question is whether the cold-target deficit is uniform, and the answer is that
it is not: ordering runs from chance in the lowest-homology band to near 0.87 in the highest, so
a pooled figure averages two very different regimes.

Two conventions matter and are easy to get wrong, both having produced wrong numbers once:

  * The k-mer tuning grid is imported from src.kmer_baseline, never restated. The published
    optimum is 10**6 neighbours, which a shorter local grid cannot reach.
  * Metrics are computed per seed and then averaged. Averaging predictions across seeds first is
    an ensemble, which is a different and better estimator, and it flatters the learned arms.

Running this reproduces the published pooled within-ligand CIs exactly (k-mer 0.7494, frozen ESM
0.6867, hybrid 0.7460), which is the check that the rest of the table can be trusted. The CNN
column is the early-stopped run and reads 0.6818; the 0.6789 quoted elsewhere is the no-early-stop
run, as the provenance note in update.md explains.

CPU only; no training and no GPU. Expect a few minutes, dominated by checkpoint inference.

Usage:
    uv run python -m analysis.homology_bands
"""
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.config import Config
from src.data import load_data, processed_csv, resolve_split_column
from src.kmer_baseline import NEIGHBOURS, SHARPENINGS, kmer_similarity, predict_pairs
from src.metrics import mse
from src.model import DeepDTA
from src.train import predict

DATA = PROJECT / "data" / "processed"
OUTPUTS = PROJECT / "outputs"
SEEDS = (4001, 4002, 4003)
BANDS = 4
BOOTSTRAP = 2000
SEED = 4001

ARMS = {
    "CNN": OUTPUTS / "phase2/davis/split_cold_target",
    "ESM": OUTPUTS / "esm/davis/split_cold_target",
    "hybrid": (OUTPUTS / "kaggle_logs/sc4001-phase5-kmer-residual/outputs"
               / "kmer-residual-esm/davis/split_cold_target"),
}


def config_from(path: Path) -> Config:
    """Rebuild a run's Config from its config.json, dropping keys Config does not declare.

    Args:
        path: Seed directory holding config.json.

    Returns:
        Config for that run.
    """
    saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
    names = {field.name for field in fields(Config)}
    return Config(**{key: tuple(value) if isinstance(value, list) else value
                     for key, value in saved.items() if key in names})


def seed_predictions(directory: Path, partition: str) -> list[np.ndarray]:
    """Per-seed predictions for one arm on one partition, in partition row order.

    Args:
        directory: Arm directory holding seed subdirectories.
        partition: "val" or "test".

    Returns:
        One prediction array per seed that has a best.pt.

    Raises:
        FileNotFoundError: No seed of this arm has a checkpoint.
    """
    stack = []
    for seed in SEEDS:
        seed_dir = directory / f"seed{seed}"
        if not (seed_dir / "best.pt").exists():
            continue
        config = config_from(seed_dir)
        loaders, vocab = load_data(config, DATA)
        model = DeepDTA(len(vocab["smiles"]) + 1, len(vocab["protein"]) + 1, config)
        model.load_state_dict(torch.load(seed_dir / "best.pt", map_location="cpu",
                                         weights_only=True))
        _, prediction, _ = predict(model, loaders[partition], torch.device("cpu"))
        stack.append(prediction)
    if not stack:
        raise FileNotFoundError(f"No checkpoint under {directory}")
    return stack


def grid_of(frame: pd.DataFrame, values: np.ndarray, drugs, targets) -> np.ndarray:
    """Reshape a per-pair vector into a dense drug-by-target matrix.

    Both DAVIS cold-target evaluation partitions are complete grids (val 68 x 64, test 68 x 63),
    which is what makes the bootstrap over proteins a column resample and therefore cheap.

    Args:
        frame: Partition frame in the same row order as values.
        values: Per-pair quantity.
        drugs: Ordered drug ids giving the row order.
        targets: Ordered target ids giving the column order.

    Returns:
        Matrix of shape (len(drugs), len(targets)).

    Raises:
        ValueError: The partition is not a complete grid.
    """
    drug_at = {drug: position for position, drug in enumerate(drugs)}
    target_at = {target: position for position, target in enumerate(targets)}
    matrix = np.full((len(drugs), len(targets)), np.nan)
    matrix[frame.ligand_id.map(drug_at).to_numpy(),
           frame.target_id.map(target_at).to_numpy()] = values
    if np.isnan(matrix).any():
        raise ValueError("Partition is not a complete drug-by-target grid.")
    return matrix


def within_ligand_ci(truth: np.ndarray, prediction: np.ndarray, columns) -> float:
    """Mean over drugs of concordance computed inside each drug's own selected targets.

    Pooled concordance is unusable here: it reads about 0.72 even for a predictor that is
    constant within each drug, so it cannot see the failure under study.

    Args:
        truth: Measured scores, shape (drugs, targets).
        prediction: Predicted scores, same shape.
        columns: Target column indices defining the band or resample.

    Returns:
        Mean within-ligand CI over drugs whose selected truths are not all equal.
    """
    y, p = truth[:, columns], prediction[:, columns]
    better = y[:, :, None] > y[:, None, :]
    gap = p[:, :, None] - p[:, None, :]
    concordant = (gap > 0).astype(float) + 0.5 * (gap == 0)
    hits = (concordant * better).sum(axis=(1, 2))
    total = better.sum(axis=(1, 2))
    usable = total > 0
    return float(np.mean(hits[usable] / total[usable])) if usable.any() else float("nan")


def tuned_kmer(similarity, parts, train_by_drug, fallback, partition) -> np.ndarray:
    """k-mer kNN predictions for one partition, tuned on validation MSE over the real grid.

    Args:
        similarity: Target-by-target 3-mer cosine matrix.
        parts: Partition name to pair frame carrying "row".
        train_by_drug: Ligand id to (training target rows, training scores).
        fallback: Prediction for a drug with no training measurement.
        partition: Partition to predict.

    Returns:
        Predictions in that partition's row order.
    """
    best = None
    for neighbours in NEIGHBOURS:
        for sharpening in SHARPENINGS:
            guess = predict_pairs(parts["val"].ligand_id, parts["val"].row.to_numpy(),
                                  train_by_drug, similarity, fallback, neighbours, sharpening)
            score = mse(parts["val"].score.to_numpy(), guess)
            if best is None or score < best[2]:
                best = (neighbours, sharpening, score)
    print("  k-mer tuned: neighbours=%s sharpening=%d (val MSE %.4f)"
          % (best[0], best[1], best[2]), flush=True)
    return predict_pairs(parts[partition].ligand_id, parts[partition].row.to_numpy(),
                         train_by_drug, similarity, fallback, best[0], best[1])


def load_partition(parts, similarity, train_rows, train_by_drug, fallback, target_row,
                   partition):
    """Build the dense grids, band assignment and arm predictions for one partition.

    Args:
        parts: Partition name to pair frame carrying "row".
        similarity: Target-by-target 3-mer cosine matrix.
        train_rows: Training target row indices.
        train_by_drug: Ligand id to (training target rows, training scores).
        fallback: Prediction for a drug with no training measurement.
        target_row: Target id to table row index.
        partition: "val" or "test".

    Returns:
        Tuple of (truth grid, censor grid, arm name to list of grids, nearest array, band array).
    """
    frame = parts[partition]
    drugs = list(dict.fromkeys(frame.ligand_id))
    target_ids = list(dict.fromkeys(frame.target_id))
    rows = np.array([target_row[t] for t in target_ids])
    nearest = similarity[np.ix_(rows, train_rows)].max(axis=1)
    edges = np.quantile(nearest, np.linspace(0, 1, BANDS + 1))
    edges[0], edges[-1] = edges[0] - 1e-9, edges[-1] + 1e-9
    band_of = pd.cut(nearest, bins=edges, labels=False)

    print("%s: %d drugs x %d proteins" % (partition.upper(), len(drugs), len(target_ids)),
          flush=True)
    truth = grid_of(frame, frame.score.to_numpy(), drugs, target_ids)
    censored = grid_of(frame, frame.censored.to_numpy(dtype=float), drugs, target_ids)
    arms = {"k-mer": [grid_of(frame, tuned_kmer(similarity, parts, train_by_drug, fallback,
                                                partition), drugs, target_ids)]}
    for label, directory in ARMS.items():
        arms[label] = [grid_of(frame, values, drugs, target_ids)
                       for values in seed_predictions(directory, partition)]
        print("  loaded %-7s %d seeds" % (label, len(arms[label])), flush=True)
    return truth, censored, arms, nearest, band_of


def main():
    """Print the per-band table for both partitions and bootstrap the hardest band."""
    pairs = pd.read_csv(processed_csv(DATA, "davis_pairs"),
                        dtype={"ligand_id": str, "target_id": str})
    targets_table = pd.read_csv(processed_csv(DATA, "davis_targets"),
                                usecols=["target_id", "sequence"], dtype={"target_id": str})
    target_row = dict(zip(targets_table.target_id, range(len(targets_table))))
    partitions = resolve_split_column(pairs, DATA, "davis", "split_cold_target")
    pairs["row"] = pairs.target_id.map(target_row)
    parts = {name: pairs.loc[partitions == name].copy() for name in ("train", "val", "test")}

    train_rows = np.unique(parts["train"].row.to_numpy())
    similarity = kmer_similarity(targets_table.sequence, 3)
    train_by_drug = {drug: (group.row.to_numpy(), group.score.to_numpy(dtype=np.float64))
                     for drug, group in parts["train"].groupby("ligand_id")}
    fallback = float(parts["train"].score.mean())

    state = {partition: load_partition(parts, similarity, train_rows, train_by_drug, fallback,
                                       target_row, partition)
             for partition in ("val", "test")}

    for partition in ("val", "test"):
        truth, censored, arms, nearest, band_of = state[partition]
        print("\n%s: within-ligand CI by homology band (per seed, then averaged)"
              % partition.upper())
        print("%-5s %-16s %8s %7s" % ("band", "cosine range", "proteins", "cens")
              + "".join("%10s" % label for label in arms))
        for band in list(range(BANDS)) + ["all"]:
            columns = (np.flatnonzero(band_of == band) if band != "all"
                       else np.arange(truth.shape[1]))
            cells = "".join("%10.4f" % np.mean([within_ligand_ci(truth, grid, columns)
                                                for grid in grids])
                            for grids in arms.values())
            label = ("%6.4f-%-9.4f" % (nearest[columns].min(), nearest[columns].max())
                     if band != "all" else "")
            print("%-5s %-16s %8d %7.3f" % (band, label, len(columns),
                                            censored[:, columns].mean()) + cells)

    generator = np.random.default_rng(SEED)
    for partition in ("val", "test"):
        truth, _, arms, _, band_of = state[partition]
        columns = np.flatnonzero(band_of == 0)
        draws = generator.integers(0, len(columns), size=(BOOTSTRAP, len(columns)))
        samples = {label: np.array([np.mean([within_ligand_ci(truth, grid, columns[draw])
                                             for grid in grids]) for draw in draws])
                   for label, grids in arms.items()}
        print("\n%s band 0, bootstrap over its %d proteins (%d draws)"
              % (partition.upper(), len(columns), BOOTSTRAP))
        for label, values in samples.items():
            low, high = np.percentile(values, [2.5, 97.5])
            print("  %-7s %.4f  95%% [%.4f, %.4f]  above chance: %s"
                  % (label, values.mean(), low, high, bool(low > 0.5)))
        margin = samples["ESM"] - samples["k-mer"]
        low, high = np.percentile(margin, [2.5, 97.5])
        print("  ESM - k-mer: %+.4f  95%% [%+.4f, %+.4f]  excludes zero: %s"
              % (margin.mean(), low, high, bool(low > 0 or high < 0)))


if __name__ == "__main__":
    main()
