"""Build the Phase 8 Kaggle notebook: confirm the Phase 5c hybrid on three more partitions.

The hybrid was measured on one cold-target partition (s4001) only. Both reviews of the project
called a second partition mandatory before the claim can be made at all, so this kernel runs the
hybrid and its width-matched control on s4002, s4003 and s4004, three seeds each.

Writing the notebook from a script keeps the cell sources readable and diffable, as
_build_phase3_notebook.py does for Phase 3.

Usage:
    uv run python kaggle/_build_phase8_notebook.py
"""
import json
from pathlib import Path

SLUG = "sc4001-phase8-davis-hybrid-confirm"
HERE = Path(__file__).resolve().parent
TARGET = HERE / SLUG

INTRO = """\
# SC4001 Phase 8: does the hybrid hold on partitions it was not built on?

Accelerator **GPU (T4 x2 or P100)**, internet off. Code, tables, frozen ESM-2 embeddings and the
precomputed out-of-fold k-mer offsets all come from `isaacwjk21/sc4001-project`.

## The question

Phase 5c measured `prediction = kmer_offset + network(drug, protein)` on **one** cold-target
partition, s4001. Against its width-matched learned control it won decisively on DAVIS: test MSE
-0.0608 (t=-6.00), test CI +0.0195 (t=+7.80), uncensored MSE -0.3167 (t=-5.36), every seed sign
aligned. Against its other parent, the k-mer lookup, it tied: within-ligand CI 0.7460 against
0.7494, signs `- - +`.

One partition cannot separate "the method works" from "that gene split was kind". Three more can.

> **RQ.** Does supplying measurement-based target similarity as an offset beat a learned protein
> encoder on cold-target partitions the method was not developed on, and does it close the gap to
> the k-mer lookup on partitions where the lookup is weaker?

## Why these three partitions are a real test, not a repeat

s4001 is the **easiest** of the four for the lookup. Measured k-mer test figures:

| partition | k-mer test MSE | k-mer test CI | k-mer uncensored MSE | max test-train similarity |
| --- | --- | --- | --- | --- |
| s4001 | 0.3516 | 0.8385 | 0.6875 | 0.695 |
| s4002 | 0.4260 | 0.8618 | 1.0551 | 0.741 |
| s4003 | 0.3815 | 0.8355 | 0.9065 | 0.830 |
| s4004 | 0.5769 | 0.7809 | 1.4281 | 0.762 |

Uncensored MSE, the view that excludes the pKd 5.0 floor and so measures the real readings, is
**1.5x to 2.1x worse** on the three new partitions. If the hybrid only ties the lookup where the
lookup is strong, these are the partitions where it has room to separate. If it loses everywhere,
the Phase 5c result was a partition artefact and the report says so.

## Arms

Six runs per partition, eighteen in all:

| arm | command | output root |
| --- | --- | --- |
| hybrid | `--protein-encoder esm --esm-proj-dim 96 --kmer-residual` | `kmer-residual-esm/` |
| control | `--protein-encoder esm --esm-proj-dim 96` | `esm/` |

Projection width 96 equals `filters[-1]`, so `output_path` files the control under `esm/` and the
hybrid under `kmer-residual-esm/`; they cannot collide. Seeds 4001-4003, no early stop, so the
two arms see identical compute and the comparison is paired by seed.

## The leakage guard, re-checked per partition

A training pair's offset must not come from training data containing that pair. `src/kmer_offsets.py`
gives training pairs **out-of-fold** offsets: training target groups are split into 5 folds and
each fold is predicted from the other four. Measured when the offsets for these partitions were
built:

| partition | train in-sample (**not used**) | train out-of-fold (used) | val | test |
| --- | --- | --- | --- | --- |
| s4002 | 0.0151 | 0.3943 | 0.3114 | 0.4260 |
| s4003 | 0.0220 | 0.4385 | 0.2599 | 0.3815 |
| s4004 | 0.0222 | 0.3892 | 0.3089 | 0.5769 |

Out-of-fold error sits beside test error and far from in-sample error on all three, which is the
condition for this experiment to mean anything.

## What counts as confirmation, fixed before the runs

* **Primary.** Paired-by-seed hybrid minus control on within-ligand CI, per partition. The claim
  "the hybrid beats a learned protein encoder" is confirmed if the sign is positive on all three
  new partitions.
* **Secondary.** Paired MSE and uncensored MSE against the control, same pairing.
* **Reported, not a pass condition.** Hybrid minus k-mer on within-ligand CI. Phase 5c tied here;
  this records whether the tie holds where the lookup is weaker.
* Treat the **four partitions** as the replication units. Nine seeds inside one partition are not
  nine independent tests of the method.
"""

SETUP = """\
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA device. Set the accelerator to GPU T4 x2 or P100, not TPU or None.")
props = torch.cuda.get_device_properties(0)
print("torch", torch.__version__, "| device", props.name,
      "|", round(props.total_memory / 1e9, 1), "GB | count", torch.cuda.device_count())
KERNEL_STARTED = time.time()
"""

LOCATE = '''\
# Locate the project root: the directory holding both src/train.py and data/processed/vocab.json.
# Kaggle strips the archive's top-level folder, so do not assume a "Project" path segment.
candidates = [p.parents[1] for p in Path("/kaggle/input").rglob("src/train.py")]
roots = [p for p in candidates if (p / "data" / "processed" / "vocab.json").exists()]
if not roots:
    raise SystemExit("No project root found. Candidates with src/train.py but no vocab.json: "
                     + str(candidates))
source = sorted(roots)[0]

project = Path("/kaggle/working/sc4001/Project")
shutil.copytree(source, project, dirs_exist_ok=True)
env = dict(os.environ, PYTHONUNBUFFERED="1")
print("source :", source)
print("project:", project)

# This kernel cannot run without the offsets for the three new partitions. Fail now, loudly,
# rather than after burning GPU on a FileNotFoundError inside the first training run.
SPLITS = ["split_cold_target_s4002", "split_cold_target_s4003", "split_cold_target_s4004"]
processed = project / "data" / "processed"
missing = [s for s in SPLITS if not (processed / ("davis_" + s + "_kmer_offset.npz")).exists()]
if missing:
    raise SystemExit("Attached dataset predates Phase 8; no k-mer offsets for: " + str(missing)
                     + ". Run src/kmer_offsets.py locally and push a new dataset version.")
print("offsets present for", SPLITS)

import numpy as np
import pandas as pd
pairs = pd.read_csv(processed / "davis_pairs.csv.gz"
                    if (processed / "davis_pairs.csv.gz").exists()
                    else processed / "davis_pairs.csv")
for split in SPLITS:
    with np.load(processed / ("davis_" + split + "_kmer_offset.npz"), allow_pickle=False) as a:
        offsets = a["offsets"]
    if offsets.shape != (len(pairs),):
        raise SystemExit(split + " offsets are " + str(offsets.shape) + ", expected "
                         + str((len(pairs),)))
    print("%-26s offsets %s  finite %s" % (split, offsets.shape, bool(np.isfinite(offsets).all())))
'''

GRID = '''\
# Eighteen runs: three partitions x three seeds x {hybrid, width-matched control}.
ROOT = "/kaggle/working/outputs"
SEEDS = (4001, 4002, 4003)
EPOCHS = 100
WIDTH = 96
# Measured on this accelerator in Phase 5: about 1.4 min for a DAVIS proj96 run either way.
PRIOR_SECONDS = 110.0
BUDGET_SECONDS = 50 * 60

GRID = [(split, seed, residual)
        for split in SPLITS for seed in SEEDS for residual in (True, False)]

status, skipped = [], []
for split, seed, residual in GRID:
    arm = "hybrid" if residual else "control"
    elapsed = time.time() - KERNEL_STARTED
    done = [row[4] for row in status if row[3] == 0]
    projected = max(done) if done else PRIOR_SECONDS
    if elapsed + projected > BUDGET_SECONDS:
        skipped.append((split, seed, arm))
        print("SKIPPING " + split + " seed" + str(seed) + " " + arm + ": projected "
              + format(projected / 60, ".1f") + " min would exceed the "
              + format(BUDGET_SECONDS / 60, ".0f") + " min guard ("
              + format(elapsed / 60, ".1f") + " min used)", flush=True)
        continue
    command = [sys.executable, "src/train.py", "--dataset", "davis", "--split", split,
               "--seed", str(seed), "--protein-encoder", "esm", "--esm-proj-dim", str(WIDTH),
               "--max-epochs", str(EPOCHS), "--patience", str(EPOCHS), "--resume"]
    if residual:
        command.append("--kmer-residual")
    print()
    print("===== " + arm + " " + split + " seed" + str(seed) + " ("
          + str(EPOCHS) + " epochs, no early stop) =====", flush=True)
    t0 = time.time()
    run = subprocess.run(command, cwd=project, env=dict(env, SC4001_OUTPUT_ROOT=ROOT),
                         check=False)
    wall = time.time() - t0
    status.append((split, seed, arm, run.returncode, wall))
    print("----- rc=" + str(run.returncode) + " wall=" + format(wall / 60, ".1f") + " min ("
          + format(wall / EPOCHS, ".2f") + " s/epoch) -----", flush=True)

print()
print("Grid wall time", format((time.time() - KERNEL_STARTED) / 60, ".1f"), "min since start")
for split, seed, arm, rc, wall in status:
    print("  %-26s seed%d %-8s rc=%d %6.1f min" % (split, seed, arm, rc, wall / 60))
failed = [row for row in status if row[3] != 0]
print("FAILED RUNS:", failed if failed else "none")
print("SKIPPED    :", skipped if skipped else "none")
'''

ANALYSE = '''\
# The decisive view: can it order one unseen target against another, inside one drug's own pairs?
# Pooled CI cannot answer that - it reads 0.7633 for a predictor constant within each drug,
# against the trained CNN's 0.7885 - so within-ligand CI is the primary endpoint here.
sys.path.insert(0, str(project))
import statistics as st
from src.config import Config
from src.data import load_esm_features, processed_csv, resolve_split_column

DATA = project / "data" / "processed"
# Settings src/kmer_baseline.py tuned on validation per partition, and so the settings the saved
# offsets were built with. Hardcoded because outputs/kmer/ is not in the attached dataset.
TUNED = {"split_cold_target_s4002": {"neighbours": 10, "sharpening": 8},
         "split_cold_target_s4003": {"neighbours": 20, "sharpening": 8},
         "split_cold_target_s4004": {"neighbours": 10, "sharpening": 4}}
KMER_TEST = {"split_cold_target_s4002": {"mse": 0.4260, "ci": 0.8618, "unc_mse": 1.0551},
             "split_cold_target_s4003": {"mse": 0.3815, "ci": 0.8355, "unc_mse": 0.9065},
             "split_cold_target_s4004": {"mse": 0.5769, "ci": 0.7809, "unc_mse": 1.4281}}
device = torch.device("cuda")


def pairwise_ci(y, prediction):
    """Return one drug's concordance index, prediction ties counted as half.

    Args:
        y: Measured scores for one drug.
        prediction: Predicted scores for the same pairs.

    Returns:
        Concordance in [0, 1], or None when no pair is orderable.
    """
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    better = y[:, None] > y[None, :]
    if not better.any():
        return None
    gap = prediction[:, None] - prediction[None, :]
    return float(((gap[better] > 0).sum() + 0.5 * (gap[better] == 0).sum()) / better.sum())


def within_ligand(frame, column):
    """Mean within-drug concordance over drugs whose test scores vary.

    Args:
        frame: Test pairs carrying ligand_id and score.
        column: Prediction column name.

    Returns:
        Mean within-ligand CI over the matched drug set.
    """
    values = []
    for _, group in frame.groupby("ligand_id"):
        if group.score.std(ddof=1) > 0:
            value = pairwise_ci(group.score, group[column])
            if value is not None:
                values.append(value)
    return float(np.mean(values))


# k-mer within-ligand CI per partition, recomputed here because only pooled figures were saved.
from src.kmer_baseline import kmer_similarity, predict_pairs

targets_table = pd.read_csv(processed_csv(DATA, "davis_targets"),
                            usecols=["target_id", "sequence"], dtype={"target_id": str})
all_pairs = pd.read_csv(processed_csv(DATA, "davis_pairs"),
                        dtype={"ligand_id": str, "target_id": str})
target_row = dict(zip(targets_table.target_id, range(len(targets_table))))
all_pairs["row"] = all_pairs.target_id.map(target_row)
similarity = kmer_similarity(targets_table.sequence, 3)

kmer_within = {}
for split in SPLITS:
    partitions = resolve_split_column(all_pairs, DATA, "davis", split)
    train = all_pairs.loc[partitions == "train"]
    test = all_pairs.loc[partitions == "test"].copy()
    train_by_drug = {d: (g.row.to_numpy(), g.score.to_numpy(dtype=np.float64))
                     for d, g in train.groupby("ligand_id")}
    test["pred"] = predict_pairs(test.ligand_id, test.row.to_numpy(), train_by_drug, similarity,
                                 float(train.score.mean()), TUNED[split]["neighbours"],
                                 TUNED[split]["sharpening"])
    kmer_within[split] = within_ligand(test, "pred")
    print("%-26s k-mer within-ligand CI %.4f" % (split, kmer_within[split]), flush=True)
'''

SCORE = '''\
# Score every finished run, then pair hybrid against control by seed within each partition.
def predictions_for(run_dir, split, seed, residual):
    """Return the test frame with this run's predictions attached.

    Args:
        run_dir: Directory holding best.pt and config.json.
        split: Split column name.
        seed: Run seed.
        residual: Whether the run adds the k-mer offset.

    Returns:
        Test pair frame carrying a "pred" column.
    """
    from src.model import DeepDTA
    from src.data import load_data
    saved = json.loads((run_dir / "config.json").read_text())
    from dataclasses import fields as dataclass_fields
    names = {f.name for f in dataclass_fields(Config)}
    config = Config(**{k: tuple(v) if isinstance(v, list) else v
                       for k, v in saved.items() if k in names})
    loaders, vocab = load_data(config, DATA)
    model = DeepDTA(len(vocab["smiles"]) + 1, len(vocab["protein"]) + 1, config).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device,
                                     weights_only=True))
    model.eval()
    out = []
    with torch.no_grad():
        for drug, protein, scores, offsets in loaders["test"]:
            out.append((model(drug.to(device), protein.to(device)).cpu() + offsets).numpy())
    partitions = resolve_split_column(all_pairs, DATA, "davis", split)
    frame = all_pairs.loc[partitions == "test"].copy()
    frame["pred"] = np.concatenate(out)
    return frame


rows = {}
for phase, residual in (("kmer-residual-esm", True), ("esm", False)):
    for path in sorted(Path(ROOT).glob(phase + "/davis/*/seed*/metrics.json")):
        run_dir = path.parent
        split = run_dir.parent.name
        seed = int(run_dir.name.replace("seed", ""))
        payload = json.loads(path.read_text())
        frame = predictions_for(run_dir, split, seed, residual)
        keep = ~frame.censored.to_numpy(dtype=bool)
        rows.setdefault((split, "hybrid" if residual else "control"), []).append({
            "seed": seed,
            "mse": payload["test"]["all"]["mse"],
            "ci": payload["test"]["all"]["ci"],
            "unc_mse": payload["test"]["uncensored"]["mse"],
            "within": within_ligand(frame, "pred"),
        })
        print("  scored %-20s %-26s seed%d within %.4f"
              % (phase, split, seed, rows[(split, "hybrid" if residual else "control")][-1]["within"]),
              flush=True)

print()
print("%-26s %-8s %-3s %-10s %-10s %-11s %-12s"
      % ("partition", "arm", "n", "test MSE", "test CI", "uncens MSE", "within-lig CI"))
for key in sorted(rows):
    group = sorted(rows[key], key=lambda r: r["seed"])
    print("%-26s %-8s %-3d %-10.4f %-10.4f %-11.4f %-12.4f"
          % (key[0], key[1], len(group), st.mean(r["mse"] for r in group),
             st.mean(r["ci"] for r in group), st.mean(r["unc_mse"] for r in group),
             st.mean(r["within"] for r in group)))

print()
print("===== PRIMARY: hybrid minus control, paired by seed, per partition =====")
verdict = {}
for split in SPLITS:
    hybrid = sorted(rows.get((split, "hybrid"), []), key=lambda r: r["seed"])
    control = sorted(rows.get((split, "control"), []), key=lambda r: r["seed"])
    if len(hybrid) != 3 or len(control) != 3:
        print("%-26s incomplete: %d hybrid, %d control" % (split, len(hybrid), len(control)))
        continue
    print()
    print(split)
    for label, key in (("within-ligand CI", "within"), ("test MSE", "mse"),
                       ("test CI", "ci"), ("uncens MSE", "unc_mse")):
        diffs = [h[key] - c[key] for h, c in zip(hybrid, control)]
        signs = " ".join("+" if d > 0 else "-" for d in diffs)
        mean = st.mean(diffs)
        t = mean / (st.stdev(diffs) / (len(diffs) ** 0.5)) if st.stdev(diffs) else float("inf")
        print("  %-18s %+8.4f  t=%+6.2f  signs %s" % (label, mean, t, signs))
        if key == "within":
            verdict[split] = mean > 0
    gap = st.mean(h["within"] for h in hybrid) - kmer_within[split]
    print("  %-18s %+8.4f  (reported, not a pass condition)" % ("vs k-mer within", gap))

print()
print("===== VERDICT, by the rule fixed before the runs =====")
for split in SPLITS:
    print("  %-26s hybrid beats its learned control on ordering: %s"
          % (split, verdict.get(split, "incomplete")))
done = [v for v in verdict.values()]
print("  CONFIRMED on all three new partitions:", bool(done) and all(done) and len(done) == 3)
'''


def main():
    """Write the kernel directory, notebook and metadata.

    Side Effects:
        Creates kaggle/<SLUG>/ with the notebook and kernel-metadata.json.
    """
    TARGET.mkdir(parents=True, exist_ok=True)
    cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO.splitlines(keepends=True)}]
    for body in (SETUP, LOCATE, GRID, ANALYSE, SCORE):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": body.splitlines(keepends=True)})
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python"}}}
    (TARGET / f"{SLUG}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n",
                                          encoding="utf-8")
    (TARGET / "kernel-metadata.json").write_text(json.dumps({
        "id": f"isaacwjk21/{SLUG}",
        "title": SLUG,
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": ["isaacwjk21/sc4001-project"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2) + "\n", encoding="utf-8")
    print("wrote", TARGET / f"{SLUG}.ipynb")
    print("wrote", TARGET / "kernel-metadata.json")
    print("cells:", len(cells))


if __name__ == "__main__":
    main()
