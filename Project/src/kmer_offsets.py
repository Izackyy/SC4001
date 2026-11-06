"""Per-pair k-mer baseline predictions, to be used as a learned model's offset.

The network cannot extract target similarity from 400 sequences (Phase 5: a zero-parameter
k-mer nearest-neighbour baseline beats the CNN on 7 of 8 cold-target partitions). This module
hands it over instead: the offset is the k-mer prediction, and the network only has to learn the
correction on top of it.

The offset for a *training* pair must not be computed from training data that contains that
pair. If it is, the residual the network sees is the k-mer method's own in-sample error, which
is far smaller than its real error, and the network learns to correct a mistake that will not be
there at test time. Training pairs therefore get **out-of-fold** offsets: target groups inside
the training set are split into folds, and each fold's pairs are predicted from the other folds
only, which reproduces the cold-target condition the model actually faces. Validation and test
pairs are predicted from the whole training set, because that is what a deployed model has.

Writes data/processed/{dataset}_{split}_kmer_offset.npz with one offset per row of the pairs
table, in table row order, so src/data.py can index it exactly as it indexes the token arrays.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import processed_csv, resolve_split_column
from .kmer_baseline import kmer_similarity, predict_pairs
from .metrics import mse

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
FOLDS = 5


def parse_args():
    """Parse dataset, split, k-mer length and neighbour settings.

    Returns:
        Namespace with dataset, split, kmer, neighbours, sharpening and folds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("davis", "kiba"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--kmer", type=int, default=3)
    parser.add_argument("--neighbours", type=int, default=None,
                        help="Neighbour count. Default: read the tuned value from "
                             "outputs/kmer/<dataset>/<split>/metrics.json.")
    parser.add_argument("--sharpening", type=int, default=None,
                        help="Similarity exponent. Default: the tuned value, as above.")
    parser.add_argument("--folds", type=int, default=FOLDS)
    return parser.parse_args()


def tuned_settings(dataset: str, split: str, neighbours, sharpening):
    """Return the neighbour and sharpening settings, preferring the tuned ones.

    Args:
        dataset: Dataset name.
        split: Split column name.
        neighbours: Override, or None to read the tuned value.
        sharpening: Override, or None to read the tuned value.

    Returns:
        Tuple of (neighbours, sharpening, source description).

    Raises:
        SystemExit: No override was given and no tuned metrics file exists.
    """
    if neighbours is not None and sharpening is not None:
        return neighbours, sharpening, "command line"
    path = OUTPUT_ROOT / "kmer" / dataset / split / "metrics.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run src.kmer_baseline first or pass both "
                         "--neighbours and --sharpening.")
    tuned = json.loads(path.read_text(encoding="utf-8"))
    return (neighbours or tuned["neighbours"], sharpening or tuned["sharpening"],
            f"{path} (val_mse {tuned['val_mse']:.4f})")


def fold_of_group(groups, folds: int, seed: int = 4001) -> dict:
    """Assign each target group to one fold, deterministically.

    Args:
        groups: Distinct target group labels.
        folds: Number of folds.
        seed: Shuffle seed, so the assignment is reproducible.

    Returns:
        Mapping of group label to fold index.
    """
    ordered = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    return {group: index % folds for index, group in enumerate(ordered)}


def out_of_fold_offsets(train, similarity, group_of_row, folds: int, neighbours: int,
                        sharpening: int) -> np.ndarray:
    """Predict every training pair from the other folds only.

    Args:
        train: Training pairs with ligand_id, row and score columns.
        similarity: Target-by-target cosine similarity.
        group_of_row: Target table row index to target group label.
        folds: Number of folds.
        neighbours: Neighbour count.
        sharpening: Similarity exponent.

    Returns:
        Offsets aligned with train's row order.

    Raises:
        ValueError: A pair was left without an offset.
    """
    assignment = fold_of_group({group_of_row[row] for row in train.row}, folds)
    fold = train.row.map(lambda row: assignment[group_of_row[row]]).to_numpy()
    offsets = np.full(len(train), np.nan)
    for index in range(folds):
        holdout, rest = fold == index, fold != index
        if not holdout.any() or not rest.any():
            raise ValueError(f"Fold {index} leaves one side empty; lower --folds.")
        fitted = train.loc[rest]
        by_drug = {drug: (group.row.to_numpy(), group.score.to_numpy(dtype=np.float64))
                   for drug, group in fitted.groupby("ligand_id")}
        offsets[holdout] = predict_pairs(train.loc[holdout].ligand_id,
                                         train.loc[holdout].row.to_numpy(), by_drug,
                                         similarity, float(fitted.score.mean()),
                                         neighbours, sharpening)
        # A fold's own targets must be invisible to the predictor that scores it.
        shared = set(fitted.row) & set(train.loc[holdout].row)
        if shared:
            raise ValueError(f"Fold {index} shares {len(shared)} target rows with its own "
                             "fitting set; the group assignment is wrong.")
    if np.isnan(offsets).any():
        raise ValueError("Some training pair received no offset.")
    return offsets


def main():
    """Write one offset per pair and print how the folds compare to deployment.

    Side Effects:
        Writes data/processed/{dataset}_{split}_kmer_offset.npz.

    Raises:
        ValueError: A partition is empty or an offset is missing.
    """
    args = parse_args()
    neighbours, sharpening, source = tuned_settings(args.dataset, args.split,
                                                    args.neighbours, args.sharpening)
    print(f"neighbours={neighbours} sharpening={sharpening} from {source}", flush=True)

    pairs = pd.read_csv(processed_csv(DATA_DIR, f"{args.dataset}_pairs"),
                        dtype={"ligand_id": str, "target_id": str})
    targets = pd.read_csv(processed_csv(DATA_DIR, f"{args.dataset}_targets"),
                          usecols=["target_id", "sequence", "target_group"],
                          dtype={"target_id": str})
    target_row = dict(zip(targets.target_id, range(len(targets))))
    # KIBA groups are target ids; DAVIS groups are gene bases, which keep a gene's mutants
    # together exactly as the cold-target split does.
    group_of_row = {row: (group if isinstance(group, str) and group else targets.target_id[row])
                    for row, group in enumerate(targets.target_group)}
    partitions = resolve_split_column(pairs, DATA_DIR, args.dataset, args.split)
    pairs["row"] = pairs.target_id.map(target_row)
    parts = {name: pairs.loc[partitions == name].copy() for name in ("train", "val", "test")}
    for name, frame in parts.items():
        if frame.empty:
            raise ValueError(f"Empty {name} partition for {args.dataset}/{args.split}.")

    similarity = kmer_similarity(targets.sequence, args.kmer)
    offsets = np.full(len(pairs), np.nan)

    train = parts["train"]
    offsets[train.index.to_numpy()] = out_of_fold_offsets(
        train, similarity, group_of_row, args.folds, neighbours, sharpening)

    # Validation and test see the whole training set, which is what a deployed model has.
    by_drug = {drug: (group.row.to_numpy(), group.score.to_numpy(dtype=np.float64))
               for drug, group in train.groupby("ligand_id")}
    fallback = float(train.score.mean())
    for name in ("val", "test"):
        frame = parts[name]
        offsets[frame.index.to_numpy()] = predict_pairs(frame.ligand_id, frame.row.to_numpy(),
                                                        by_drug, similarity, fallback,
                                                        neighbours, sharpening)
    if np.isnan(offsets[np.isin(np.asarray(partitions), ("train", "val", "test"))]).any():
        raise ValueError("A pair in a named partition has no offset.")

    path = DATA_DIR / f"{args.dataset}_{args.split}_kmer_offset.npz"
    np.savez_compressed(path, offsets=offsets.astype(np.float32))
    print("wrote", path, offsets.shape, flush=True)

    # The number that says whether the out-of-fold scheme worked. In-sample training error far
    # below the test error would mean the network is being handed an unrealistically easy
    # residual, which is the failure this module exists to prevent.
    in_sample = predict_pairs(train.ligand_id, train.row.to_numpy(), by_drug, similarity,
                             fallback, neighbours, sharpening)
    for name, truth, prediction in (
            ("train out-of-fold", train.score.to_numpy(), offsets[train.index.to_numpy()]),
            ("train in-sample", train.score.to_numpy(), in_sample),
            ("val", parts["val"].score.to_numpy(), offsets[parts["val"].index.to_numpy()]),
            ("test", parts["test"].score.to_numpy(), offsets[parts["test"].index.to_numpy()])):
        print("  %-18s offset MSE %.4f  residual sd %.4f"
              % (name, mse(truth, prediction), float(np.std(truth - prediction, ddof=1))),
              flush=True)
    print("  Out-of-fold training MSE should sit near the test MSE, not near the in-sample one.")


if __name__ == "__main__":
    main()
