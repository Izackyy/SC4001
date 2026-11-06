"""Phase 4 mutant stress test: does the model move its prediction correctly for a point edit?

DAVIS carries real kinase point mutants with measured affinities, so each wild-type/mutant
pair measured against the same ligand is a question with ground truth: the measured affinity
shift is known, and the model's predicted shift can be compared against it.

Run it on a trained run directory holding `best.pt`:

    uv run python src/eval_mutants.py --run outputs/phase2/davis/split_cold_target/seed4001

Two cautions the output states for itself rather than leaving to the reader.

Contamination. Mutants share a gene with their wild type, so a gene-level split keeps the two
together: under `split_cold_target` every mutant pair falls in train or val, never test. A
cold-target checkpoint has therefore already seen the pairs being asked about, and the result
is a question about what the architecture can represent, not about generalisation. Only a
`split_mutant` checkpoint holds the mutant pairs out. The report names the partition each pair
came from and refuses to call a contaminated run a generalisation result.

Truncation. Protein tokens are capped at 1200 residues, so an edit past position 1200 produces
a token sequence identical to the wild type's and the model cannot possibly distinguish them.
Those mutants are detected by comparing tokens, not by the mutant table's `n_edits`, which
counts edits in the untruncated sequence and so reports them as edited. They are reported
separately as a control: the predicted shift must be exactly zero.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data import processed_csv
from src.model import DeepDTA

CENSORED_FLOOR_NOTE = ("DAVIS is left-censored at 10,000 nM, which is pKd 5.0. A pair where "
                       "both measurements sit on that floor has a measured shift of exactly "
                       "zero for reporting reasons, not biological ones.")


def parse_args():
    """Read the command line.

    Returns:
        Namespace with run directories, the data directory and the device.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="DIR",
                        help="A run directory holding best.pt and config.json. Repeat for "
                             "several seeds; metrics are reported per run and pooled.")
    parser.add_argument("--data-dir", default=None,
                        help="Processed data directory; defaults to Project/data/processed.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device. CPU is the sensible default: this is inference "
                             "over a few thousand pairs, not training.")
    parser.add_argument("--out", default=None,
                        help="Optional JSON path for the full report.")
    parser.add_argument("--min-shift", type=float, default=0.5,
                        help="Measured shift in pKd below which a sign comparison is treated "
                             "as uninformative, because the sign of a near-zero shift is "
                             "noise. Default 0.5.")
    return parser.parse_args()


def load_tables(data_dir):
    """Load the DAVIS tables and token matrices needed for the stress test.

    Args:
        data_dir: Processed data directory.

    Returns:
        Tuple of (mutants, targets, ligands, pairs, drug_tokens, prot_tokens, vocab).
    """
    mutants = pd.read_csv(processed_csv(data_dir, "davis_mutants"))
    targets = pd.read_csv(processed_csv(data_dir, "davis_targets"), dtype={"target_id": str})
    ligands = pd.read_csv(processed_csv(data_dir, "davis_ligands"), dtype={"ligand_id": str})
    pairs = pd.read_csv(processed_csv(data_dir, "davis_pairs"),
                        dtype={"ligand_id": str, "target_id": str})
    with np.load(data_dir / "davis_tokens.npz", allow_pickle=False) as arrays:
        drug_tokens = arrays["drug_tokens"]
        prot_tokens = arrays["prot_tokens"]
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    return mutants, targets, ligands, pairs, drug_tokens, prot_tokens, vocab


def build_comparisons(mutants, targets, pairs, prot_tokens):
    """Assemble every wild-type/mutant pair measured against a shared ligand.

    A mutant is usable when it is constructible, its own sequence and its wild type's are
    both present, and at least one ligand was measured against both.

    Args:
        mutants: Mutant table.
        targets: Target table, whose row order indexes prot_tokens.
        pairs: Affinity table with score, censored and the split columns.
        prot_tokens: Protein token matrix.

    Returns:
        Tuple of (frame, invisible) where frame has one row per comparable
        (ligand, wild type, mutant) triple and invisible lists the mutant ids whose tokens
        match their wild type's after truncation.
    """
    row_of = {target_id: position for position, target_id in enumerate(targets.target_id)}
    usable = mutants[mutants.constructible & (mutants.n_edits >= 1)
                     & mutants.target_id.isin(row_of) & mutants.wild_type_id.isin(row_of)]

    invisible = [row.target_id for row in usable.itertuples()
                 if np.array_equal(prot_tokens[row_of[row.target_id]],
                                   prot_tokens[row_of[row.wild_type_id]])]

    # Index the affinity table once per target rather than filtering it per mutant.
    by_target = {target_id: group.set_index("ligand_id")
                 for target_id, group in pairs.groupby("target_id")}
    records = []
    for row in usable.itertuples():
        mutant_rows = by_target.get(row.target_id)
        wild_rows = by_target.get(row.wild_type_id)
        if mutant_rows is None or wild_rows is None:
            continue
        for ligand_id in mutant_rows.index.intersection(wild_rows.index):
            mutant_pair = mutant_rows.loc[ligand_id]
            wild_pair = wild_rows.loc[ligand_id]
            records.append({
                "ligand_id": ligand_id,
                "wild_type_id": row.wild_type_id,
                "mutant_id": row.target_id,
                "tag": row.tag,
                "kind": row.kind,
                "visible": row.target_id not in invisible,
                "measured_wild": wild_pair.score,
                "measured_mutant": mutant_pair.score,
                "censored_wild": bool(wild_pair.censored),
                "censored_mutant": bool(mutant_pair.censored),
                "split_cold_target": mutant_pair.split_cold_target,
                "split_mutant": mutant_pair.split_mutant,
            })
    frame = pd.DataFrame.from_records(records)
    frame["measured_shift"] = frame.measured_mutant - frame.measured_wild
    frame["either_censored"] = frame.censored_wild | frame.censored_mutant
    return frame, invisible


def predict(model, drug_tokens, prot_tokens, ligand_rows, target_rows, device, batch=512):
    """Predict affinity for aligned ligand and target row indices.

    Args:
        model: Trained DeepDTA in eval mode.
        drug_tokens: Drug token matrix.
        prot_tokens: Protein token matrix.
        ligand_rows: Integer array of drug token rows.
        target_rows: Integer array of protein token rows, same length.
        device: Torch device.
        batch: Rows per forward pass.

    Returns:
        Float numpy array of predictions.
    """
    out = []
    with torch.no_grad():
        for start in range(0, len(ligand_rows), batch):
            stop = start + batch
            drugs = torch.from_numpy(drug_tokens[ligand_rows[start:stop]].astype(np.int64))
            proteins = torch.from_numpy(prot_tokens[target_rows[start:stop]].astype(np.int64))
            out.append(model(drugs.to(device), proteins.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def shift_metrics(measured, predicted, min_shift):
    """Score predicted affinity shifts against measured ones.

    Args:
        measured: Measured pKd shifts.
        predicted: Predicted pKd shifts.
        min_shift: Magnitude below which a sign comparison is uninformative.

    Returns:
        Dict of metrics, or a dict with n = 0 when there is nothing to score.
    """
    if len(measured) < 2:
        return {"n": int(len(measured))}
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    informative = np.abs(measured) >= min_shift
    metrics = {
        "n": int(len(measured)),
        "measured_sd": float(measured.std(ddof=1)),
        "predicted_sd": float(predicted.std(ddof=1)),
        "mae": float(np.abs(predicted - measured).mean()),
        "rmse": float(np.sqrt(((predicted - measured) ** 2).mean())),
        "mean_measured": float(measured.mean()),
        "mean_predicted": float(predicted.mean()),
        "n_informative": int(informative.sum()),
    }
    # The ratio of spreads is the question the stress test really asks: a model that cannot
    # represent point edits predicts a near-constant shift whatever the measurement says.
    metrics["sd_ratio"] = (float(metrics["predicted_sd"] / metrics["measured_sd"])
                           if metrics["measured_sd"] > 0 else None)
    if predicted.std() > 0 and measured.std() > 0:
        metrics["pearson"] = float(np.corrcoef(measured, predicted)[0, 1])
        ranks = (pd.Series(measured).rank().to_numpy(), pd.Series(predicted).rank().to_numpy())
        metrics["spearman"] = float(np.corrcoef(*ranks)[0, 1])
        # Slope of measured on predicted: 1.0 would mean correctly calibrated shifts.
        metrics["slope_measured_on_predicted"] = float(
            np.polyfit(predicted, measured, 1)[0])
    else:
        metrics["pearson"] = None
        metrics["spearman"] = None
        metrics["slope_measured_on_predicted"] = None
        metrics["predicted_is_constant"] = True
    if informative.sum():
        metrics["sign_agreement"] = float(
            (np.sign(predicted[informative]) == np.sign(measured[informative])).mean())
    else:
        metrics["sign_agreement"] = None
    return metrics


def evaluate_run(run_dir, frame, drug_tokens, prot_tokens, ligand_rows, target_rows, device):
    """Load one run's best checkpoint and predict every comparison.

    Args:
        run_dir: Directory holding best.pt and config.json.
        frame: Comparison table from build_comparisons.
        drug_tokens: Drug token matrix.
        prot_tokens: Protein token matrix.
        ligand_rows: Row lookup from ligand_id to drug token row.
        target_rows: Row lookup from target_id to protein token row.
        device: Torch device.

    Returns:
        Tuple of (config dict, predicted wild-type array, predicted mutant array).

    Raises:
        FileNotFoundError: best.pt or config.json is missing.
    """
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "best.pt"
    config_path = run_dir / "config.json"
    for path in (checkpoint_path, config_path):
        if not path.exists():
            raise FileNotFoundError("Missing %s" % path)
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    config = Config(**{key: value for key, value in saved.items()
                       if key in Config.__dataclass_fields__})
    sizes = saved["vocab_sizes"]
    model = DeepDTA(sizes["smiles"], sizes["protein"], config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    drugs = frame.ligand_id.map(ligand_rows).to_numpy()
    wild = frame.wild_type_id.map(target_rows).to_numpy()
    mutant = frame.mutant_id.map(target_rows).to_numpy()
    return (saved,
            predict(model, drug_tokens, prot_tokens, drugs, wild, device),
            predict(model, drug_tokens, prot_tokens, drugs, mutant, device))


def main():
    """Run the stress test over every requested run directory and print the report."""
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else (
        Path(__file__).resolve().parents[1] / "data/processed")
    device = torch.device(args.device)

    mutants, targets, ligands, pairs, drug_tokens, prot_tokens, _ = load_tables(data_dir)
    frame, invisible = build_comparisons(mutants, targets, pairs, prot_tokens)
    ligand_rows = {ligand_id: row for row, ligand_id in enumerate(ligands.ligand_id)}
    target_rows = {target_id: row for row, target_id in enumerate(targets.target_id)}

    print("Phase 4 mutant stress test, DAVIS")
    print("  comparable (ligand, wild type, mutant) triples: %d" % len(frame))
    print("  distinct mutants: %d, of which invisible after the 1200-residue truncation: %d"
          % (frame.mutant_id.nunique(), len(invisible)))
    for mutant_id in invisible:
        print("      %s  (tokens identical to wild type; predicted shift must be 0)" % mutant_id)
    print("  censoring: %d of %d triples have at least one measurement on the pKd 5.0 floor"
          % (int(frame.either_censored.sum()), len(frame)))
    print("  " + CENSORED_FLOOR_NOTE)

    report = {"n_triples": len(frame), "invisible_mutants": invisible, "runs": {}}
    for run_dir in args.run:
        saved, predicted_wild, predicted_mutant = evaluate_run(
            run_dir, frame, drug_tokens, prot_tokens, ligand_rows, target_rows, device)
        predicted_shift = predicted_mutant - predicted_wild
        split = saved.get("split", "unknown")
        held_out = split == "split_mutant"
        partitions = frame[split].value_counts().to_dict() if split in frame else {}

        print()
        print("=" * 78)
        print("run %s" % run_dir)
        print("  trained on split %s, seed %s" % (split, saved.get("seed")))
        print("  this run's partition for these triples: %s" % partitions)
        if held_out:
            print("  HELD OUT: the mutant pairs are this split's test set, so the numbers "
                  "below are a generalisation result.")
        else:
            print("  CONTAMINATED FOR THIS PURPOSE: these triples were in this run's "
                  "training or validation data, so the numbers below say what the "
                  "architecture can represent, NOT whether it generalises to unseen mutants.")

        visible = frame.visible.to_numpy()
        uncensored = ~frame.either_censored.to_numpy()
        strata = {
            "visible_uncensored": visible & uncensored,
            "visible_all": visible,
            "invisible_control": ~visible,
        }
        run_report = {"split": split, "seed": saved.get("seed"), "held_out": held_out,
                      "partitions": partitions, "strata": {}}
        for name, mask in strata.items():
            metrics = shift_metrics(frame.measured_shift.to_numpy()[mask],
                                   predicted_shift[mask], args.min_shift)
            run_report["strata"][name] = metrics
            print("  %-20s n=%-5d" % (name, metrics["n"]), end="")
            if metrics["n"] >= 2:
                print(" measured sd %.3f | predicted sd %.3f | sd ratio %s | r %s | MAE %.3f"
                      % (metrics["measured_sd"], metrics["predicted_sd"],
                         "n/a" if metrics["sd_ratio"] is None else "%.3f" % metrics["sd_ratio"],
                         "n/a" if metrics["pearson"] is None else "%+.3f" % metrics["pearson"],
                         metrics["mae"]))
            else:
                print()

        # The control is a pipeline check, not a result: identical tokens must give identical
        # predictions, so any nonzero shift here means the comparison was built wrongly.
        control = np.abs(predicted_shift[~visible])
        if control.size:
            worst = float(control.max())
            print("  control check: largest |predicted shift| on invisible mutants = %.2e %s"
                  % (worst, "OK" if worst < 1e-4 else "FAILED, the comparison is built wrong"))
            run_report["control_max_abs_predicted_shift"] = worst
        report["runs"][str(run_dir)] = run_report

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print()
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
