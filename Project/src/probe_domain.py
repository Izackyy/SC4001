"""Held-out domain probe: can a linear classifier on frozen protein features tell training proteins
from proteins the encoder never trained on, *for proteins the probe itself has never seen*?

Three readings, one number each (5-fold cross-validated AUC over proteins):

* near 0.5: there is no seen/unseen signature in this representation;
* well above 0.5 on the DANN target pool but near 0.5 on the test proteins: the encoder learned a
  pool-specific invariance that did not transfer;
* well above 0.5 on both: the representation separates any unseen protein from the training ones.

A random half/half split of the training proteins is reported as the null. Input-space 3-mer
composition and a randomly initialised encoder are reported as references, so a signature can be
attributed to training rather than to the sequences themselves.

Usage:
    uv run python -m src.probe_domain --dataset davis --run outputs/nostop/baseline/phase2/davis/split_cold_target/seed4001
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data import processed_csv, resolve_split_column
from src.model import DeepDTA

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def parse_args():
    """Parse the dataset, split, run directories and checkpoint choice.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("davis", "kiba"), required=True)
    parser.add_argument("--split", default="split_cold_target")
    parser.add_argument("--run", action="append", default=[], type=Path,
                        help="Run directory holding config.json and a checkpoint; repeatable.")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--seed", type=int, default=0, help="Fold and null-split RNG seed.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON destination.")
    return parser.parse_args()


def protein_roles(dataset, split):
    """Return each processed target's partition under the split, in target-table row order.

    Args:
        dataset: "davis" or "kiba".
        split: Split column name.

    Returns:
        Object array of "train", "val" or "test" per target row.
    """
    pairs = pd.read_csv(processed_csv(DATA_DIR, f"{dataset}_pairs"), dtype={"ligand_id": str, "target_id": str})
    targets = pd.read_csv(processed_csv(DATA_DIR, f"{dataset}_targets"), usecols=["target_id"], dtype={"target_id": str})
    partitions = resolve_split_column(pairs, DATA_DIR, dataset, split)
    role = {}
    for name in ("train", "val", "test"):
        for target in pairs.loc[partitions == name, "target_id"].unique():
            role[target] = name
    return targets.target_id.map(role).to_numpy()


def encoder_features(dataset, run=None, checkpoint="best"):
    """Encode every benchmark and external protein with a run's protein encoder.

    Args:
        dataset: Selects the token archives.
        run: Run directory; None builds a randomly initialised encoder from the defaults.
        checkpoint: "best" (model state only) or "last" (full training state).

    Returns:
        Benchmark features and external-pool features, both (n, filters[-1]) float arrays.
    """
    if run is None:
        vocab = json.loads((DATA_DIR / "vocab.json").read_text(encoding="utf-8"))
        torch.manual_seed(4001)
        model = DeepDTA(len(vocab["smiles"]) + 1, len(vocab["protein"]) + 1, Config(dataset=dataset))
    else:
        saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
        config = Config(**{key: value for key, value in saved.items() if key in Config.__dataclass_fields__})
        model = DeepDTA(saved["vocab_sizes"]["smiles"], saved["vocab_sizes"]["protein"], config)
        state = torch.load(run / f"{checkpoint}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    with np.load(DATA_DIR / f"{dataset}_tokens.npz", allow_pickle=False) as arrays:
        own = torch.tensor(arrays["prot_tokens"], dtype=torch.long)
    with np.load(DATA_DIR / f"external_tokens_{dataset}.npz", allow_pickle=False) as arrays:
        external = torch.tensor(arrays["prot_tokens"], dtype=torch.long)
    with torch.no_grad():
        encode = lambda tokens: torch.cat([model.protein_encoder(tokens[i:i + 64]) for i in range(0, len(tokens), 64)]).numpy()
        return encode(own), encode(external)


def kmer_features(dataset, k=3, dims=96):
    """Return PCA-reduced L2-normalised k-mer count vectors as an input-space reference.

    Args:
        dataset: Selects the target table.
        k: K-mer length.
        dims: Principal components kept, matched to the encoder width.

    Returns:
        Benchmark features and external-pool features.
    """
    own = pd.read_csv(processed_csv(DATA_DIR, f"{dataset}_targets"), usecols=["sequence"]).sequence
    external = pd.read_csv(processed_csv(DATA_DIR, "external_kinases"), keep_default_na=False).sequence
    sequences = list(own) + list(external)
    vocabulary, counts = {}, []
    for sequence in sequences:
        row = {}
        for start in range(len(sequence) - k + 1):
            row[sequence[start:start + k]] = row.get(sequence[start:start + k], 0) + 1
        counts.append(row)
        for kmer in row:
            vocabulary.setdefault(kmer, len(vocabulary))
    matrix = np.zeros((len(sequences), len(vocabulary)), dtype=np.float32)
    for index, row in enumerate(counts):
        for kmer, value in row.items():
            matrix[index, vocabulary[kmer]] = value
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    centred = matrix - matrix.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    reduced = centred @ components[:dims].T
    return reduced[:len(own)], reduced[len(own):]


def held_out_auc(features, labels, rng, folds=5, weight_decay=1e-2):
    """Cross-validated AUC of a class-balanced L2 logistic regression.

    Args:
        features: (n, d) array.
        labels: Binary int array of length n.
        rng: NumPy generator deciding the fold assignment.
        folds: Number of folds over proteins.
        weight_decay: L2 penalty on the weights.

    Returns:
        Mean held-out AUC over folds that contain both classes.
    """
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    order = rng.permutation(len(labels))
    aucs = []
    for fold in range(folds):
        held = order[fold::folds]
        fit = np.setdiff1d(order, held)
        x, y = torch.tensor(features[fit], dtype=torch.float32), torch.tensor(labels[fit], dtype=torch.float32)
        weight = torch.zeros(features.shape[1], requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], max_iter=300)
        positive_weight = (1 - y.mean()) / y.mean()

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                x @ weight + bias, y, pos_weight=positive_weight) + weight_decay * (weight * weight).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        scores = features[held] @ weight.detach().numpy() + bias.detach().numpy()
        positive, negative = scores[labels[held] == 1], scores[labels[held] == 0]
        if len(positive) and len(negative):
            aucs.append(float((positive[:, None] > negative[None, :]).mean()))
    return float(np.mean(aucs))


def probe_table(own, external, roles, rng, null_draws=5):
    """Compute the probe AUCs for one representation.

    Args:
        own: Benchmark protein features in target-table order.
        external: External-pool features.
        roles: Partition per benchmark protein.
        rng: NumPy generator.
        null_draws: Random half/half splits of the training proteins averaged for the null.

    Returns:
        Dict of contrast name to held-out AUC.
    """
    train, val, test = own[roles == "train"], own[roles == "val"], own[roles == "test"]

    def auc(a, b):
        labels = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(int)
        return held_out_auc(np.vstack([a, b]), labels, rng)

    nulls = []
    for _ in range(null_draws):
        shuffled = train[rng.permutation(len(train))]
        half = len(train) // 2
        nulls.append(held_out_auc(shuffled, np.r_[np.zeros(half), np.ones(len(train) - half)].astype(int), rng))
    return {"train_vs_target_pool": auc(train, np.vstack([val, external])),
            "train_vs_val": auc(train, val), "train_vs_external": auc(train, external),
            "train_vs_test": auc(train, test), "null_random_halves": float(np.mean(nulls))}


def main():
    """Print the probe table for the reference representations and every run given.

    Side Effects:
        Prints one row per representation; writes JSON when --out is given.
    """
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    roles = protein_roles(args.dataset, args.split)
    rows = {"input 3-mer composition (PCA-96)": probe_table(*kmer_features(args.dataset), roles, rng),
            "random-init CNN encoder": probe_table(*encoder_features(args.dataset), roles, rng)}
    for run in args.run:
        metrics = run / "metrics.json"
        epoch = json.loads(metrics.read_text(encoding="utf-8"))["best_epoch"] if metrics.exists() else None
        name = f"{run.as_posix()} [{args.checkpoint}, best epoch {epoch}]"
        rows[name] = probe_table(*encoder_features(args.dataset, run, args.checkpoint), roles, rng)
    columns = ("train_vs_target_pool", "train_vs_val", "train_vs_external", "train_vs_test", "null_random_halves")
    print(f"{args.dataset} {args.split}: held-out probe AUC, 5-fold over proteins")
    print(f"{'representation':70s} " + " ".join(f"{column:>20s}" for column in columns))
    for name, values in rows.items():
        print(f"{name:70s} " + " ".join(f"{values[column]:20.3f}" for column in columns), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"dataset": args.dataset, "split": args.split, "checkpoint": args.checkpoint,
                                        "rows": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
