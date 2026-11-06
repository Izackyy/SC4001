# SC4001 drug-target affinity project

Phase 1 builds the DAVIS and KIBA inputs used by the later CNN, DANN and mutant
stress-test phases. Python 3.12 is required. From this directory, install the
locked environment with `uv sync` (the supplied environment is already synced).

Raw data comes from the [DeepDTA data directory](https://github.com/hkmztrk/DeepDTA/tree/master/data).
Place `proteins.txt` and `ligands_can.txt` under both `data/raw/davis/` and
`data/raw/kiba/`. DAVIS additionally needs
`drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt`; KIBA needs
`kiba_binding_affinity_v2.txt`. Raw files are not tracked. No pickle is read.
Optional SC4020 EDA exports `data/raw/davis_pairs.csv.gz` and
`data/raw/kiba_pairs.csv.gz` are used only to cross-check the rebuilt ID-keyed
tables, never as pipeline inputs.

Run the notebook from `Project/`:

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_data.ipynb
```

The notebook defines every helper inline, includes assertions, and saves its
tables and token arrays under `data/processed/`. Its executed outputs show the
construction audit, truncation rates, split counts and reload checks. Figures
are saved as `figures/davis_split_scores.png` and `figures/kiba_split_scores.png`.
With the environment and raw files present, execution needs no network.

| Processed file | Contents |
| --- | --- |
| `{ds}_ligands.csv.gz` | `ligand_id, smiles, merged_ids, n_ids` |
| `{ds}_targets.csv.gz` | `target_id, sequence, merged_ids, n_ids, target_group, is_mutant, wild_type_id, mutation_tag` |
| `{ds}_pairs.csv.gz` | `ligand_id, target_id, score, censored, n_merged, spread, n_censored, split_warm, split_cold_target`; DAVIS also has `split_mutant` |
| `{ds}_tokens.npz` | `drug_tokens` and `prot_tokens`, int16; rows match the respective entity CSV |
| `vocab.json` | Sorted `smiles` and `protein` character lists and dataset-specific `max_len`; token ID = list index + 1, padding = 0 |
| `davis_mutants.csv` | `target_id, base, tag, phospho, kind, offset, constructible, reason, wild_type_id, n_edits, seq_len_before, seq_len_after` |

`{ds}` means `davis` or `kiba`. DAVIS Kd in nM becomes `9 - log10(Kd)`;
10,000 nM is the censored pKd 5 screening limit. KIBA scores are unchanged.
Identical SMILES/sequence pairs merge by mean score; `spread` is sample standard
deviation (zero for singletons), and a merged pair is censored only if all its
measurements were censored. Representatives are first IDs in file order.

DAVIS mutant construction supports substitutions, concatenated or dash-joined
substitutions, and inclusive deletions with both endpoint residues validated.
Sequence index is tag position + offset - 1: ABL1 uses +37, PIK3CA +1, and other
supported genes use 0. RET and FGFR3 are excluded because their truncated
constructs have unresolved numbering. ITD, domain/species tags, missing wild-type
IDs and residue mismatches are not constructible. Phosphorylation does not alter
the sequence. The 49 validated IDs produce 44 distinct mutant sequences;
unsupported annotations keep their original sequences. Constructed sequences
replace the shipped copies before deduplication and tokenisation. `n_edits`
counts substituted or deleted residues. Target-table `wild_type_id` references
the representative wild-type entity; manifest IDs refer to the original maps.

Token lengths (SMILES/protein) are 85/1200 for DAVIS and 100/1000 for KIBA.
Vocabularies cover both datasets; strings are truncated to prefixes and padded
with zeros. Seed 4001 makes splits reproducible. Warm splits use 70/15/15 by
pairs; cold-target splits shuffle indivisible groups and place cumulative pair
counts nearest those boundaries. DAVIS groups are gene bases, including mutants
and phosphorylated forms; KIBA groups are target IDs. The DAVIS mutant split
holds all constructed mutant targets in test and randomly splits other pairs
85/15 into train/validation. Every corresponding wild-type target occurs in
training. There is no cold-drug split.

At the required DAVIS protein length of 1200, `LRRK2(G2019S)`, `MET(M1250T)`
and `MET(Y1235D)` have the same tokens as their wild types because their edited
residues fall outside the retained prefix. Their full constructed sequences
remain distinct in the target CSV; the notebook reports this token limitation.

Gzip exports use `mtime=0`; NPZ archives also have fixed timestamps. The notebook
reloads all outputs and validates values, IDs, splits and token alignment.

## Phase 2

The baseline uses separate SMILES and protein CNN encoders and a regression
head. `src.model.ProteinEncoder` is reusable by the Phase 3 discriminator.
Vocabulary sizes and sequence lengths come from `data/processed/vocab.json`.
All training pairs are used, including censored DAVIS measurements.

From `Project/`, run one configuration:

```powershell
uv run python src/train.py --dataset davis --split split_cold_target --seed 4001
```

Flags are `--dataset {davis,kiba}`, `--split
{split_warm,split_cold_target,split_mutant}`, `--seed` (4001), `--max-epochs`
(100), `--batch-size` (256), and `--resume`. The mutant split is DAVIS-only.
`--resume` restores `last.pt`, including optimiser, random states, history and
early-stopping state. Completed runs reuse saved metrics without evaluating
test again. Only the maximum epoch count may change when resuming. Best model
selection uses validation MSE; test is evaluated once at the end.

Real experiments require a hosted accelerator. `resolve_device()` tries CUDA,
then an XLA (TPU) device through `torch_xla`, and the entry point refuses CPU
training when neither exists: measured CPU runs cost about two hours per DAVIS
seed and seven per KIBA seed. `--allow-cpu` is permitted only with `--smoke`
for correctness checks. Smoke runs use two epochs, two training batches and one
evaluation batch per epoch, with batch size capped at eight; model widths and
token lengths are unchanged.

The XLA path routes the optimiser step through `xm.optimizer_step` and saves
checkpoints with `xm.save`, and is verified only against an injected fake
`torch_xla`. It does not work on a Kaggle TPU. Measured on 2026-11-05: the
image ships `torch 2.8.0+cpu` with `torch_xla 2.8.0`, device resolution
succeeds (`torch_xla.device() -> xla:0`), and the training child then aborts
with SIGABRT after 7.7 seconds, dumping core inside the XLA runtime.

**TPU has a second, independent blocker: it does not get scheduled.** An early
TPU job sat queued for hours on 4 Nov and never started, and a deliberately
small probe pushed on 5 Nov was still `QUEUED` after 45 minutes, having never
run. This is the stronger objection of the two, because nothing in this project
can fix it: a working stack that is never scheduled is unusable, so the SIGABRT
is not even the binding constraint. There were about 19 of 20 weekly TPU hours
unused at the time, so quota was never the limit either.

The probe (`kaggle/sc4001-tpu-probe/`) is retained because it is the right
experiment if TPU is ever revisited. It runs six steps in separate subprocesses,
since signal 6 cannot be caught in-process: import and device, a trivial tensor
op, `nn.Embedding`, `Conv1d` plus `amax`, the real `DeepDTA` forward, then
backward with an Adam step. Whichever step aborts localises the fault, and it
never executes `src/train.py`, so no existing result is at risk. Use a GPU; the
XLA code is retained but unproven.

Experiments run on Kaggle, driven from this machine over the Kaggle API, with
credentials in `~/.kaggle/`. `src/` and `data/processed/` are published as the
dataset `isaacwjk21/sc4001-project`. A run is launched with `kaggle kernels
push` against a directory holding the notebook and a `kernel-metadata.json`
that sets `enable_gpu` true, `enable_tpu` false, `enable_internet` false and
lists that dataset under `dataset_sources`; progress comes from `kaggle kernels
status` and results from `kaggle kernels output`. A T4 is sufficient, and no uv
or install step is needed because Kaggle preinstalls torch, numpy, pandas and
scipy.

`notebooks/kaggle_phase2_gpu.ipynb` is the executed notebook downloaded back
from Kaggle, so its saved cell outputs are the record of what ran. It finds the
project by searching `/kaggle/input` for `src/train.py` beside
`data/processed/vocab.json`, since Kaggle strips the uploaded archive's
top-level directory, copies that tree into `/kaggle/working` so it is writable,
points `SC4001_OUTPUT_ROOT` there, runs one smoke check before spending GPU
quota, then runs the grid through subprocesses so a single failure does not end
the sweep. Kaggle also decompresses `.gz` members when it unpacks a dataset, so
`src/data.py` accepts either `{name}.csv.gz` or `{name}.csv`.

Measured results on a Tesla T4, test set, mean with sample standard deviation
over training seeds. DAVIS has three seeds (4001-4003); KIBA was extended to
eight (4001-4008) to settle whether its wide cold-target spread was one odd
seed.

| dataset | split | seeds | MSE | CI | training-mean MSE |
| --- | --- | --- | --- | --- | --- |
| DAVIS | `split_warm` | 3 | 0.213 (0.011) | 0.887 (0.004) | 0.771 |
| DAVIS | `split_cold_target` | 3 | 0.383 (0.010) | 0.792 (0.009) | 0.674 |
| KIBA | `split_warm` | 8 | 0.190 (0.005) | 0.859 (0.002) | 0.700 |
| KIBA | `split_cold_target` | 8 | 0.368 (0.034) | 0.767 (0.009) | 0.744 |

KIBA reproduces the DAVIS finding independently, which matters more than either
dataset alone: its paired gap is +0.178 MSE and -0.092 CI against DAVIS's +0.170
and -0.095.

The wide KIBA cold-target spread is real, not an outlier. On three seeds it was
0.352 (sd 0.039) and looked like one seed at 0.308 against two near 0.375; on
eight seeds it is 0.368 (sd 0.034), ranging 0.308 to 0.409. So cold-target
generalisation is genuinely seed-sensitive, with about seven times the training
-seed variance of the warm split (0.034 against 0.005), while warm stays steady
to within 0.005. Three seeds also put the cold-target mean about 0.016 too low.

Read the cold-target rows with the split-seed sweep below in mind. Every split
here is Phase 1's seed 4001, and that turned out to be the kindest gene
partition on both datasets: cold-target MSE runs to 0.533 on KIBA seed 4004 and
0.557 on DAVIS seed 4004. The figures above are therefore a best case, not a
typical one.

Paired by seed, cold-target minus warm is +0.170 MSE and -0.095 CI on DAVIS
(three seeds), and +0.178 MSE and -0.092 CI on KIBA (eight seeds). Paired
t-tests give p = 0.005 (DAVIS MSE), p = 0.005 (DAVIS CI) and, on KIBA,
t(7) = 14.74 for MSE and t(7) = -28.79 for CI, both p < 1e-5. DAVIS still rests
on three seeds, so its two p-values are suggestive rather than strong; KIBA's
eight seeds settle the question on that dataset, and the effect appears at the
same magnitude on both. That gap is the quantity Phase 3 aims to shrink.

Uncensored-only test pairs are much harder: on DAVIS, 0.580 MSE warm against
1.058 cold-target. KIBA has no censored records, so its two figures coincide.

Cost: the DAVIS grid took 67.2 minutes (8.6 to 13.2 per run) and the KIBA grid
232.5 minutes (27.8 to 48.1 per run), 298 minutes of T4 time for all twelve.

DeepDTA reports DAVIS MSE 0.261 / CI 0.878 and KIBA MSE 0.194 / CI 0.863. Our
warm figures beat three of those four and fall short on one: DAVIS MSE by
0.048, DAVIS CI by 0.009 and KIBA MSE by 0.004, while KIBA CI is 0.004 lower
than the paper's. The two KIBA margins are smaller than its eight-seed standard
deviations, so neither is a real difference. None of it is like-for-like: content-level deduplication in Phase 1 leaves 28,764 DAVIS pairs
against roughly 30,056 in the paper, and 117,657 KIBA pairs against 118,254.
Treat these as a sanity check of the same order, not as an improvement.

Outputs normally go to
`outputs/phase2/{dataset}/{split}/seed{seed}/`: `config.json`, `history.csv`,
`last.pt`, `best.pt` and `metrics.json`. `SC4001_OUTPUT_ROOT` overrides the
`outputs/` root for hosted storage. Smoke directories end in `-smoke` and must
not be used as results. Checkpoint files are ignored by Git; metrics, history
and configuration remain trackable. CI is exact and calculated in NumPy chunks.
Metrics include validation and test MSE/CI, the training-mean baseline, sample
counts, best epoch, total elapsed seconds and device. DAVIS reports all and
uncensored pairs separately; KIBA reports once. Undefined metrics are JSON null.

Copy hosted outputs back to `Project/outputs/`, then execute
`notebooks/02_baseline.ipynb`. It reads saved results, reports each seed plus
mean and sample standard deviation, calculates the warm minus cold-target gap,
and plots validation curves. With no real runs, it prints a no-runs message.

## Extra cold-target split seeds

Phase 1 froze every split at seed 4001, so the seed-to-seed standard deviations
above measure training noise only; they say nothing about how much the
warm-to-cold gap depends on which genes landed in test.

`notebooks/04_split_seeds.ipynb` regenerates `split_cold_target` under split
seeds 4002, 4003 and 4004 using Phase 1's grouping rule unchanged, and writes
`data/processed/{ds}_extra_splits.csv.gz` (`ligand_id`, `target_id`, one column
per split seed). It is a separate file so the Phase 1 pair tables stay
byte-identical; the notebook asserts train/val/test target sets are pairwise
disjoint, that whole gene groups stay together, that fractions stay near
70/15/15, and that each new split differs from the others.

Pass them like any other split:

```powershell
uv run python src/train.py --dataset kiba --split split_cold_target_s4002 --seed 4001
```

`--split` accepts `split_warm`, `split_cold_target`, `split_mutant` or
`split_cold_target_s<seed>`, and rejects anything else. `resolve_split_column`
in `src/data.py` reads the column from the pair table when present and from the
extra-splits table otherwise.

Split seeds and training seeds are independent: a split seed chooses the
partition, a training seed chooses initialisation and batch order. DAVIS pair
counts are identical across split seeds because DAVIS is a complete 68 x 423
matrix, so every target carries exactly 68 pairs; the membership still differs,
with 7 to 14 of 63 test targets shared with the original split.

### What the split seeds measure: KIBA

Nine runs, three gene partitions by three training seeds, on KIBA
`split_cold_target`. Seed 4001 is Phase 1's original split and its three runs
come from the Phase 2 grid. All four rows below use three training seeds so the
partitions are compared like for like; seed 4001 separately has eight seeds,
which put it at 0.368 (sd 0.034) rather than the 0.352 shown here.

| split seed | test MSE (sd) | test CI (sd) |
| --- | --- | --- |
| `split_cold_target` (4001) | 0.352 (0.039) | 0.770 (0.007) |
| `split_cold_target_s4002` | 0.372 (0.012) | 0.732 (0.012) |
| `split_cold_target_s4003` | 0.414 (0.028) | 0.751 (0.009) |
| `split_cold_target_s4004` | 0.533 (0.042) | 0.732 (0.005) |

Which genes land in test matters far more than which training seed is used. MSE
spans 0.352 to 0.533 across partitions, a 0.18 swing, while scatter within a
partition is 0.012 to 0.042. Pooled over all nine runs, MSE is 0.440 (sd 0.077).
Concordance index is the steadier metric at 0.732 to 0.770.

The warm-to-cold gap itself survives every partition. Against warm MSE 0.189 the
cold penalty runs +0.163 on seed 4001 and +0.344 on seed 4004, and the roughly
-0.09 CI drop holds on all four. So the Phase 2 conclusion stands; only its
magnitude is partition-dependent, and the single number quoted in Phase 2 is the
most favourable of the four.

Harder partitions also train longer before early stopping: best epochs 51 to 74
on seed 4004 against 28 to 40 on seed 4003. Run times were 21.3 to 43.5 min,
mean 29.9, inside the one-hour-per-run budget.

### The same effect on DAVIS

All four DAVIS cold-target partitions are now measured at three training seeds
each, so the effect is not a KIBA artifact and it is about the same size on both
datasets.

| partition | test MSE | test CI |
| --- | --- | --- |
| `split_cold_target` (s4001) | 0.3828 (0.0095) | 0.7919 (0.0094) |
| `split_cold_target_s4003` | 0.4533 (0.0297) | 0.8065 (0.0042) |
| `split_cold_target_s4002` | 0.5527 (0.0166) | 0.8060 (0.0052) |
| `split_cold_target_s4004` | 0.5567 (0.0266) | 0.7744 (0.0087) |

The spread is 0.174 MSE, matching KIBA's 0.18, while training-seed scatter stays
under 0.030 within every partition. **Two things in this table matter more than
the spread itself.**

First, s4001 and s4004 really are the extremes, so the decision to bracket Phase
3 with those two partitions survives contact with the data.

Second, and less comfortably, **s4001 is the easiest of the four by a wide
margin.** It sits 0.104 below the four-partition mean of 0.4864, while the other
three cluster between 0.453 and 0.557. It is not one draw among four comparable
ones; it is an outlier in the favourable direction. Every Phase 3 result so far
was developed and measured on it. Any absolute number quoted from `split_cold_target`
is therefore the most flattering of the four available, and the honest headline
figure for DAVIS cold-target difficulty is the four-partition mean near 0.49,
not 0.38. That mean is itself the censored-inclusive view; the next section gives
the uncensored one, which is 1.29.

Note also that CI does not track MSE across partitions. The two middling
partitions have the *highest* concordance (0.806), while s4004 has both the worst
MSE and the worst CI (0.774) and s4001 has the best MSE but a middling CI
(0.792). Ranking quality and calibrated error are close to decoupled here, which
is why both are reported throughout.

Note the shape of the effect. Changing the partition moves MSE a long way and
concordance index very little, on both datasets. Ranking survives a harder gene
split better than calibrated prediction does.

### Two measurement views, and why the headline number needs both

Every MSE and CI above is the `all` stratum: every test pair, including those at
the pKd 5.0 screening floor. That is the standard DeepDTA setup and it is the
right default, but on DAVIS it is also the flattering one, because 64-70% of a
cold-target test partition is censored and a censored pair is the easy case. The
runs store an `uncensored` stratum beside it, and the two tell different stories:

| partition | `all` MSE | `all` CI | uncensored MSE | uncensored CI | censored fraction |
| --- | --- | --- | --- | --- | --- |
| s4001 | 0.3828 | 0.7919 | 1.0578 | 0.6523 | 0.7024 |
| s4003 | 0.4533 | 0.8065 | 1.3146 | 0.6921 | 0.6996 |
| s4002 | 0.5527 | 0.8060 | 1.3875 | 0.6499 | 0.6608 |
| s4004 | 0.5567 | 0.7744 | 1.4126 | 0.6372 | 0.6431 |
| **mean** | **0.4864** | — | **1.2931** | — | — |

**On the measured records the model's cold-target error is 1.29, not 0.49.**
Quote whichever view the comparison calls for, but say which one it is: the two
differ by a factor of 2.7, and the censored block contributes most of the
difference. KIBA has no censoring, so its figures need no such note.

Note that this is not an irreducible floor masquerading as error. Phase 1 merges
identical sequences into one row (17 groups, 442 DAVIS target ids collapsing to
423) and averages duplicate `(ligand, target)` pairs, so there is no residual
variance from several labels sharing one protein input. All 423 processed
sequences are distinct, and no identical-sequence group straddles train and
evaluation in any of the four partitions.

### The concordance index reported here is pooled, and pooled CI hides the task

Every CI above pools all test pairs, so it rewards ordering one drug against
another as well as ordering targets within a drug. On a cold-target split those
are very different things, and a model that returns **one constant per drug**
still scores a pooled CI near 0.72 while being at chance on the question the
split is asking. Two statistics separate them:

*   **Within-ligand CI** — concordance over each test drug's own measured pairs,
    averaged over drugs with at least two distinct scores.
*   **Within-ligand prediction spread** — predict each test drug against every
    cold test target, measured or not, and take the standard deviation. It is
    exactly 0 for a model that cannot separate unseen targets.

| run | pooled CI | within-ligand CI | predicted spread | measured spread | drugs at predicted spread 0 |
| --- | --- | --- | --- | --- | --- |
| DAVIS s4001 seed 4001 | 0.7919 | 0.6925 | 0.3316 | 0.6203 | 0 of 68 |
| DAVIS s4004 seed 4001 | 0.7744 | 0.6589 | 0.2607 | 0.6894 | 0 of 68 |
| KIBA s4001 baseline seed 4001 | 0.7668 | 0.6322 | 0.2632 | 0.4436 | 0 of 2025 |

The spreads in this table are averaged over **every** test drug, while the measured
spread beside them can only be averaged over drugs that have one, so the implied
ratio is optimistic. The last column counts drugs whose *predicted* spread is zero,
which is the degeneracy check, and is unrelated to the 2 DAVIS and 429 KIBA drugs
whose *measured* spread is zero. The later section on the lambda sweep gives the matched-set
figures, which are the ones to quote.

**The good news and the bad news are both in the last two columns.** The spread
is non-zero for every single drug in every run, so the model does separate unseen
targets from sequence alone. A collaborative-filtering model cannot do this at
all: an unseen target has no learned bias, so its prediction collapses to a
per-drug constant and its within-ligand CI is exactly 0.5000. That capability is
what a sequence model buys.

But the predicted spread is roughly half the measured spread in every run, 0.53
on DAVIS s4001 and 0.59 on KIBA. **The model systematically under-commits to what
the protein is**, hedging toward a per-drug constant without ever reaching one.
That is the same pathology the Phase 4 mutant stress test found from a different
direction, where the predicted shift spread was 0.156 of the measured spread. Any
report of these numbers should pair the pooled CI with the within-ligand pair,
because pooled CI alone cannot distinguish a working model from a constant.

### What the adversary actually does to the predictions, and what it does not

The within-ligand statistics above were computed for every arm of the DAVIS lambda
sweep and for the KIBA pair, 21 runs in all, by inference on checkpoints that were
already trained. Measured within-ligand score spread is 0.6203 on DAVIS and
0.4436 on KIBA, and is the same for every arm because the test partition is.

| arm | within-ligand CI | predicted spread | spread / measured | dMSE | dspread |
| --- | --- | --- | --- | --- | --- |
| DAVIS baseline | 0.6789 | 0.3679 | 0.576 | — | — |
| DAVIS lambda 0.03 | 0.6743 | 0.3740 | 0.585 | +0.0038 | +0.0061 |
| DAVIS lambda 0.1 | 0.6484 | 0.3254 | 0.509 | +0.0176 | -0.0425 |
| DAVIS lambda 0.3 | 0.6523 | 0.2435 | 0.381 | +0.0643 | -0.1245 |
| DAVIS lambda 1.0, dead adversary | 0.6750 | 0.3441 | 0.539 | +0.0112 | -0.0238 |
| KIBA baseline | 0.6363 | 0.2960 | 0.514 | — | — |
| KIBA lambda 0.03 | 0.6112 | 0.1967 | 0.341 | +0.0728 | -0.0993 |

Spreads and ratios here are on a **matched ligand set**: the 66 of 68 DAVIS and 1436
of 2025 KIBA test drugs that have at least two measured pairs and a non-zero
measured spread, which is also the set the within-ligand CI is computed over.
Measured spread on that set is 0.6391 on DAVIS and 0.5762 on KIBA. Median
per-ligand ratios agree with the ratios of means and run slightly higher: 0.583
for the DAVIS baseline, 0.543 for the KIBA baseline, 0.380 for KIBA DANN. The mean
of per-ligand ratios is never quoted, because 2 DAVIS and 429 KIBA test drugs have
a measured spread of exactly zero -- every record censored at the pKd 5.0 floor --
which would make it undefined.

**What this establishes.** Loss of predicted spread and loss of accuracy are the
same phenomenon. Across the five non-baseline arms, over both datasets, the
correlation between an arm's MSE penalty and its loss of within-ligand spread is
**r = -0.951**. Whatever damages cold-target accuracy also flattens the model's
predictions across unseen targets, and the two move together closely enough that
they are best read as one effect rather than two.

The baseline under-dispersion stands on its own and is the more important number:
**predicted spread is 0.576 of measured on DAVIS and 0.514 on KIBA before any
domain adaptation is applied.** The model under-commits to protein identity as
trained, which is the same pathology the Phase 4 mutant stress test found from a
different direction, where the predicted shift spread was 0.156 of measured.

**What this does not establish, and a claim that failed its control.** The
tempting reading is that the adversary suppresses protein-identity features, that
being why DANN hurts: aligning training and unseen proteins in feature space
removes exactly the information the task needs. Two results in this table refuse
it.

*   **The dead-adversary arm does not behave like the baseline.** At lambda 1.0
    the discriminator sat at ln 2 for all 100 epochs, so there was no adversarial
    pressure to suppress anything, and yet the spread still fell by 0.0231 and the
    MSE still rose by 0.0112. If a live adversary were doing the suppressing, this
    arm should sit on the baseline. It sits between the lambda 0.03 and lambda 0.1
    arms, exactly where its MSE sits.
*   **Within-ligand CI is not ordered by lambda.** It moves -0.0046, -0.0305,
    -0.0266 and -0.0040 across lambda 0.03, 0.1, 0.3 and 1.0. Because a uniform
    shrinkage of predictions toward their per-drug mean is order-preserving, and
    therefore cannot change within-ligand CI at all, a CI drop is the signature
    that distinguishes reordering from mere scale compression. On DAVIS that
    signature is small and unordered.

Also, the largest spread drop in the table is partly an artifact. The lambda 0.3
arm's three seeds have spreads of **0.1161**, 0.3416 and 0.2580, and the first is
the run whose validation MSE reached 1.383 at epoch 40: its predictions range only
from 0.1154 to 0.1169 across drugs, so it has collapsed to very nearly a constant.
Dropping it leaves the arm at 0.2998, close to lambda 0.1. The arm mean of 0.2386
is one unstable run, not a dose-response.

**So the honest statement is the weaker one.** Spread loss tracks accuracy loss
tightly, the baseline already under-disperses by about half, and DANN's damage on
KIBA shows up in both measures together. Whether the adversary specifically
destroys target-ranking information, rather than distorting the output scale, is
**not** settled by this evidence: on KIBA the within-ligand CI fell on 3 of 3
seeds (-0.0252, t(2) = -4.04, p = 0.056) which is suggestive, but the DAVIS sweep
does not reproduce an ordered CI effect and its dead-adversary control points away
from an adversary-specific mechanism. Settling it needs a live-adversary arm at
matched MSE damage, which this sweep does not contain.

## Phase 3 data

`notebooks/03_external_kinases.ipynb` builds the unlabelled pool from the local
`data/raw/uniprot/human_kinases.fasta`. The UniProt REST query was
`(organism_id:9606) AND (reviewed:true) AND (xref:pfam-PF00069 OR xref:pfam-PF07714)`:
reviewed human proteins carrying either protein-kinase Pfam domain. Raw data is
gitignored; the notebook names the required file in an error if it is absent
and never downloads it. Run from `Project/`:

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/03_external_kinases.ipynb
```

| Filter | Removed | Remaining | Reason |
| --- | --- | --- | --- |
| Parse FASTA | 0 | 474 | Preserve full header ID, gene name and sequence |
| Outside the existing protein alphabet | 0 | 474 | Reject unencodable residues |
| Exact DAVIS or KIBA target sequence | 353 | 121 | Add new protein content, not renamed copies |
| Maximum test 5-mer Jaccard >= 0.90 | 3 | 118 | Exclude near-copies of either dataset's cold-target test proteins |
| Length > 3000 residues | 4 | 114 | Exclude giant proteins whose kinase domains are beyond the retained prefix |

The near-copies are P19525, P29320 and Q9BZL6. The giant proteins are titin
(34350 residues), obscurin, SPEG and TRIO. Most of the 474 proteins were already
in DAVIS or KIBA because both are kinase panels drawn from the human kinome.
The pool therefore adds 114 genuinely new sequences rather than 474.

`data/processed/external_kinases.csv.gz` contains
`kinase_id, gene, sequence, length, max_test_jaccard`. The similarity column is
the maximum full-sequence 5-mer set Jaccard against both datasets' cold-target
test proteins, used only to screen the pool. The retained maximum is 0.768987.
Lengths range from 268 to 2986 residues (median 747.5, mean 864.18).
`external_tokens_davis.npz` and `external_tokens_kiba.npz` each contain only
`prot_tokens`, an int16 array in CSV row order. They use the existing alphabet,
zero padding and lengths 1200/1000. Respectively 24/39 sequences are truncated.
Decode assertions and reload checks validate every row. Gzip uses `mtime=0`
and NPZ timestamps are fixed; two executions produced byte-identical exports.

## Phase 3

`src/train_dann.py` is a separate entry point; the verified Phase 2 model, data
and metrics modules are unchanged. It accepts the Phase 2 flags plus
`--lambda-max` (default 1.0) and `--discriminator-input {protein,joint}`
(default `protein`).

Both entry points also accept `--patience` (default 15). Setting it equal to
`--max-epochs` disables early stopping, which is required when comparing two
methods on these splits: see "Early stopping confounds the comparison" below.
The value is saved in `config.json` and is part of the resume configuration
check, so a run started under one patience refuses to resume under another. The discriminator attaches to the pooled protein
vector, 96 wide in the Phase 3 configuration, and has two width-64 hidden layers
with ReLU and dropout. Its hidden width is stored in
`DANNConfig.discriminator_hidden`, and its input width is read from
`protein_encoder.out_dim` so it follows a widened projection.

`src/train_dann.py` also accepts `--protein-encoder {cnn,esm}`, `--esm-proj-dim`
and `--kmer-residual`, which mean exactly what they mean in `src/train.py`. With
`--protein-encoder esm` the unlabelled target-domain pool is embedded the same
way as the training targets, from `{dataset}_esm.npz` and `external_esm.npz`
instead of the token arrays, and `load_target_domain` checks the embedding shapes
and finiteness in place of the token-equality check. The no-test-sequence
contract is unchanged and still raises if a test sequence ever reaches the pool.
`--protein-encoder cnn` without `--kmer-residual` is the default and writes to
`outputs/phase3/...` byte-for-byte as before; any other combination gets its own
root (`outputs/kmer-residual-phase3-esm/...`, and `proj<width>` appended when the
projection is widened), so no Phase 3 result can be overwritten by a later
variant.

**`--lambda-max` is not part of the run directory**, which is
`<root>/<phase>/<dataset>/<split>/seed<N>`. A sweep over lambda under one
`SC4001_OUTPUT_ROOT` therefore collides with itself: the second lambda lands on
the first lambda's directories and `--resume` refuses because the saved config
differs. Give each lambda its own root, as the Phase 3 scripts do
(`SC4001_OUTPUT_ROOT=.../ablations/lambda$strength`). Fixing this inside
`train_dann.py` would move every existing Phase 3 run directory, so it has not
been done. The Kaggle kernel `kaggle/sc4001-phase6-dann-on-hybrid/` uses this to
run DANN on the k-mer residual hybrid, which is the first representation in the
project that demonstrably orders unseen targets.

```powershell
uv run python src/train_dann.py --dataset davis --split split_cold_target --seed 4001 --lambda-max 1.0 --discriminator-input protein
```

Source proteins come from shuffled training pairs, so target frequency matches
the labelled regression stream. The target domain is the union of unique
validation proteins and the 114 external kinases: 178 proteins for DAVIS and
153 for KIBA. Target permutations are shuffled and cycled without discarding
partial batches. Each regression step uses an equally sized domain batch,
half source and half target; for an odd batch the extra row alternates domains.
Source features are reused from the regression batch. `joint` concatenates
drug and protein features, pairing both domains with the same sampled training
drugs; external proteins never receive affinity labels.

The no-test-protein guard applies to the dataset and split being trained.
DAVIS and KIBA were split independently: three validation sequences in each
occur in the other dataset's cold-target test set. Keeping all 64/39 validation
proteins therefore does not give disjointness against the other benchmark's
test set. The external pool is screened against both benchmarks. No test
protein from the run's own split enters regression or domain training; test
tables are consulted only for exclusion checks and final evaluation. This is
domain generalisation to unseen test targets, not transductive adaptation to
test proteins. Unlabelled validation proteins do influence representation
learning, so validation is not untouched by training. Splits with overlapping
training/test sequences (including `split_warm`) are rejected: they cannot
satisfy this no-test-protein contract. The cold-target setting is the main
generalisation experiment.

Each step uses `progress = (epoch * steps + step) / (max_epochs * steps)` with
zero-based epoch and step. Lambda follows Ganin's logistic schedule.
The loss is `MSE + BCE` with lambda carried solely by the gradient reversal
coefficient, as in Ganin et al. The encoder's adversarial gradient is therefore
proportional to lambda and the discriminator receives its full gradient from the
first step. Early stopping and best-checkpoint selection use validation
regression MSE only.

This was changed on 2026-11-05 after the first GPU run. The loss had been
`MSE + lambda * BCE` with the reversal coefficient also lambda, which scales the
discriminator's own gradient by lambda: in the first epoch it trained at about
2.5% of its nominal rate and never became a useful adversary. Measured on a T4,
`domain_loss` stayed at 0.690 (ln 2) and `domain_accuracy` at chance for all 20
epochs while test MSE came out at 0.491 against the baseline's 0.383. Those runs
are kept under `outputs/phase3-lambda-squared/` as the evidence and must not be
pooled with later ones.

Outputs use `outputs/phase3/{dataset}/{split}/seed{seed}/` with the same
`config.json`, `history.csv`, `last.pt`, `best.pt` and `metrics.json` contract
as Phase 2. `SC4001_OUTPUT_ROOT` overrides the output root. History additionally
records `domain_loss` (unweighted BCE), `domain_accuracy` and `lambda` (mean
over the epoch's steps). Metrics preserve `val`, `test`, `all`, DAVIS
`uncensored`, and `training_mean_baseline`. The results notebook's metric
reader can consume them when pointed at the Phase 3 root; its current default
still points to Phase 2. Analyse different lambda settings separately.

`--resume` restores both models, both optimisers, target-sampler and other RNG
states, history and early-stopping state. Only `max_epochs` may change.
Completed runs reuse saved metrics without another test evaluation. Smoke
outputs end in `-smoke`; CPU use requires both `--smoke` and `--allow-cpu`,
using the existing `src.train.resolve_device()` for accelerator discovery.

Domain accuracy drifting toward 0.5 is consistent with useful domain confusion,
but is not sufficient evidence of better regression. Accuracy pinned near
1.0 while validation MSE rises is the memorisation/failure signature. The
small discriminator and real external diversity address the limited number
of distinct validation proteins; the held-out test MSE/CI determine whether
the cold-target gap actually shrinks.

Read the diagnostic in both directions. Accuracy that *starts* at 0.5 and never
leaves it, with `domain_loss` flat at ln 2, is not success: it means the
discriminator learned nothing, so the encoder is being pushed by an adversarial
gradient derived from noise. Distinguish the two cases by whether `domain_loss`
ever moves. A functioning run shows it fall below 0.693 before accuracy returns
toward chance; a starved or impossible one never moves off it at all.

For the real DAVIS grid, use seeds 4001, 4002 and 4003 at lambda 1.0, then the
same seeds at 0.1 and 0.5. Separate output roots prevent ablations overwriting
the main runs. From the synced `Project/` environment in PowerShell:

```powershell
$env:SC4001_OUTPUT_ROOT = Join-Path $PWD 'outputs'
foreach ($seed in 4001,4002,4003) {
    uv run python src/train_dann.py --dataset davis --split split_cold_target --seed $seed --lambda-max 1.0 --discriminator-input protein
}
foreach ($lambda in '0.1','0.5') {
    $env:SC4001_OUTPUT_ROOT = Join-Path $PWD "outputs/ablations/lambda$lambda"
    foreach ($seed in 4001,4002,4003) {
        uv run python src/train_dann.py --dataset davis --split split_cold_target --seed $seed --lambda-max $lambda --discriminator-input protein
    }
}
Remove-Item Env:SC4001_OUTPUT_ROOT
```

On Kaggle T4 use the same arguments with its preinstalled Python (no uv or
package installation), after locating/copying the dataset as in Phase 2.
CSV loading accepts Kaggle's decompressed names. From the copied project
directory, the exact Bash grid is:

```bash
for seed in 4001 4002 4003; do
  SC4001_OUTPUT_ROOT=/kaggle/working/outputs python src/train_dann.py --dataset davis --split split_cold_target --seed "$seed" --lambda-max 1.0 --discriminator-input protein || exit $?
done
for strength in 0.1 0.5; do
  for seed in 4001 4002 4003; do
    SC4001_OUTPUT_ROOT="/kaggle/working/outputs/ablations/lambda$strength" python src/train_dann.py --dataset davis --split split_cold_target --seed "$seed" --lambda-max "$strength" --discriminator-input protein || exit $?
  done
done
```

### Which cold-target partitions Phase 3 is evaluated on

The grids above run on `split_cold_target` alone, which the split-seed sweep
shows is not enough. Cold-target MSE moved by 0.18 across four KIBA gene
partitions while training-seed scatter stayed under 0.042, so a DANN result
measured on one partition could be an artifact of a kind split.

Phase 3 therefore runs on two partitions, `split_cold_target` (seed 4001) and
`split_cold_target_s4004`, the easiest and hardest measured on KIBA. Running
all four would cost four times the quota and answer the question no better:
two endpoints bracket the range, so if DANN helps at both ends the conclusion
holds between them, and if it helps only on the kind split then that is the
lucky-split caveat made visible rather than hidden.

All four DAVIS partitions have since been measured (see "The same effect on
DAVIS"), and the choice holds: s4001 at 0.383 and s4004 at 0.557 are the
extremes. One caveat became sharper rather than weaker, though. s4001 is not
merely the easiest, it is 0.104 below the four-partition mean while the other
three cluster at 0.453 to 0.557, so a result shown only on s4001 rests on the
single most favourable partition of the four. The same two split columns
are used for both datasets so DAVIS and KIBA stay comparable.

DAVIS had no Phase 2 baseline on seed 4004, only on seed 4001 and
`split_warm`, so the DAVIS kernel runs those three baseline runs itself before
its DANN runs on that partition. KIBA needs no equivalent: the split-seed
sweep already produced baselines on all four partitions.

Kaggle kernel definitions live in `kaggle/<slug>/`, each holding a notebook and
`kernel-metadata.json`, pushed with `kaggle kernels push -p kaggle/<slug>`. The
dataset already carries `src/dann.py`, `src/train_dann.py`, the external token
archives and both `*_extra_splits` tables, so Phase 3 needs no new dataset
version. Note that the Kaggle API exposes no logs for a running kernel: timings
are readable only once a kernel reports `COMPLETE`, so size an expensive grid
from a cheap finished one rather than expecting to inspect a run in flight.

### Early stopping confounds the comparison

Measured on a T4, DAVIS `split_cold_target`, seeds 4001-4003, under the
corrected loss. Baseline test MSE is 0.383.

| `--lambda-max` | test MSE | test CI | best epoch | stopped |
| --- | --- | --- | --- | --- |
| 1.0 | 0.4949 (0.0069) | 0.7700 | 3, 3, 3 | 18 |
| 0.5 | 0.4946 (0.0069) | 0.7704 | 5, 3, 3 | 18-20 |
| 0.1 | 0.4930 (0.0093) | 0.7723 | 5, 3, 6 | 18-21 |

A tenfold range in lambda changes nothing, which is the tell. At `--lambda-max
0.1` the schedule only reaches 0.077 by epoch 20, so there is almost no
adversarial pressure, yet the result matches lambda 1.0. The adversarial term is
therefore not what produced the deficit.

Early stopping produced it. Every DANN run peaked at epoch 3 to 6 and stopped at
18 to 21, while every baseline run peaked at 51 to 93 and stopped at 66 to 100.
Validation MSE here sits on a noisy plateau: the baseline is at 0.739 after one
epoch, 0.675 by epoch 5, back up at 0.690 by epoch 10, and only reaches its best
of 0.486 at epoch 52. With `patience` 15 and `min_delta` 1e-4, escaping that
plateau depends on happening to post a better-than-`min_delta` epoch inside the
window. All three baseline runs did; none of the nine DANN runs did. So DANN's
0.49 is the model after five epochs and the baseline's 0.38 is the model after
fifty-two, and the difference is training length rather than adaptation.

The corrected loss did work on its own terms. At `--lambda-max 0.1`,
`domain_loss` finally falls from 0.692 to 0.647 and domain accuracy rises to
0.62, so the discriminator learns. At 0.5 and 1.0 the reversal suppresses it from
the start and it stays at chance, which is the intended behaviour rather than a
fault.

**Neither the lambda 1.0 result nor this ablation is a valid statement about
DANN.** Both are statements about a truncated run. The equal-compute redo is
below.

### Equal compute: the deficit was the stopping rule

Measured on a T4, DAVIS `split_cold_target`, seeds 4001-4003, both arms at
`--patience 100` so neither can stop early and each gets the full 100 epochs.
The best epoch is chosen from `history.csv` afterwards. Results are under
`outputs/nostop/<arm>/`, kept apart from the patience-15 runs because the
per-run output path carries neither the patience nor the arm.

| arm | test MSE | test CI | best val MSE | best epochs |
| --- | --- | --- | --- | --- |
| baseline | 0.3991 (0.0217) | 0.7885 (0.0014) | 0.4621 (0.0776) | 80, 93, 49 |
| DANN lambda 1.0 | 0.4103 (0.0173) | 0.7923 (0.0082) | 0.5933 (0.0391) | 31, 84, 45 |

Paired by seed, DANN is +0.0112 MSE (sd 0.0106, t(2) = +1.82) and +0.0038 CI
(sd 0.0069, t(2) = +0.96). Neither is significant at three seeds. The 0.11 MSE
deficit reported under `patience` 15 is therefore gone: it was training length,
not adaptation, and the confound is confirmed.

**This still is not a measurement of DANN.** At `--lambda-max 1.0` the
discriminator never trains. Over all 100 epochs `domain_loss` stays at 0.691 to
0.692 against ln 2 = 0.6931, and `domain_accuracy` at 0.51 to 0.53, while lambda
climbs to 0.999. The reversal suppresses the discriminator from the first step,
so the encoder's adversarial gradient comes from an adversary at chance. This arm
is the baseline plus noise and extra forward passes, which is what the matched
test MSE says. The informative setting is `--lambda-max 0.1`, the one lambda at
which the discriminator demonstrably learned (`domain_loss` 0.692 to 0.647,
accuracy 0.62), but those runs were the truncated ones. Re-running lambda 0.1 and
0.3 at `--patience 100` is the next experiment.

One asymmetry is worth recording. DANN is clearly worse on validation, +0.131
MSE (sd 0.0988, t(2) = +2.30), while matching the baseline on test. Both arms see
the same 4352 validation and 4284 test pairs. Validation proteins are part of the
unlabelled target-domain stream; test proteins are excluded from it by contract.
So the damage falls on the domain the encoder was aligned to and not on the unseen
one. At three seeds this is a hint rather than a finding, and with the
discriminator at chance it is at least as likely to be the cost of the extra
unlabelled forward passes as any alignment effect.

Both arms overfit heavily by epoch 100: training MSE reaches 0.065 for the
baseline and 0.072 for DANN while validation MSE rises again (baseline seed 4001
0.374 at epoch 80 to 0.400 at 100; DANN seed 4001 0.617 at epoch 31 to 0.956 at
100). Disabling early stopping is correct for an equal-compute comparison but is
not a good training recipe. Treat any A/B whose two arms stop at very different
epochs as confounded until shown otherwise.

### The first valid DANN measurement: a trained adversary does not help

Measured on a T4, DAVIS `split_cold_target`, seeds 4001-4003, `--patience 100`
on every run, `--lambda-max` in {0.03, 0.1, 0.3}, nine runs in 3 h 09 min.
Results under `outputs/lambda-nostop/lambda<value>/`. The baseline arm is the
patience-100 baseline from the equal-compute kernel above, not a fresh run.

**The adversary trained in all nine runs**, which is what makes this the first
result that says anything about the method. `domain_loss` departed from
ln 2 = 0.693147 in every run, and how far it departed is ordered by lambda,
exactly as suppression predicts:

| `--lambda-max` | min `domain_loss` | max `domain_accuracy` | test MSE | test CI |
| --- | --- | --- | --- | --- |
| 0 (baseline) | n/a | n/a | 0.3991 (0.0217) | 0.7885 (0.0014) |
| 0.03 | 0.497-0.505 | 0.746-0.751 | 0.4029 (0.0125) | 0.7830 (0.0036) |
| 0.1 | 0.636-0.639 | 0.624-0.625 | 0.4167 (0.0125) | 0.7862 (0.0052) |
| 0.3 | 0.673-0.675 | 0.570-0.606 | 0.4634 (0.0409) | 0.7819 (0.0108) |

The weaker the reversal, the better the discriminator does: at lambda 0.03 it
reaches 75% accuracy, at 0.3 only 57-61%. Lambda controls how hard the encoder is
pushed to defeat it, so a small lambda means a strong discriminator and a lightly
perturbed encoder.

Paired by seed against the baseline: +0.0038 MSE at lambda 0.03 (t(2) = +0.28),
+0.0176 at 0.1 (t(2) = +1.19), +0.0643 at 0.3 (t(2) = +1.89). None is
significant at three seeds. CI is slightly worse at every lambda, -0.0055,
-0.0023 and -0.0066, also not significant, but negative in all three arms.

**Conclusion: with an adversary that demonstrably trains and compute held equal,
adversarial domain alignment does not improve cold-target generalisation on
DAVIS.** It is indistinguishable from the baseline at lambda 0.03 and 0.1, and
worse at 0.3. That is a negative result about the method on this problem, and
unlike the three earlier attempts it is not explained away by a bug, by lambda
being inert, or by unequal training length.

Two honest qualifications. First, the lambda 0.3 deficit is **instability, not a
dose-response**. Two of its three seeds peaked at epoch 5 and epoch 18 and then
got worse for the rest of the run: seed 4001's validation MSE goes 0.739 at epoch
1, 0.655 at 5, 0.833 at 20, **1.383 at 40**, 0.812 at 100, while training MSE
falls to 0.075. The third seed behaved normally (best epoch 89, test MSE 0.4172).
So lambda 0.3 destabilises training in some runs rather than degrading it
smoothly. Second, the per-seed pattern is not monotone in lambda. Seed 4002 is
flat across the whole sweep (0.4208 baseline, then 0.4117, 0.4089, 0.4172) and
seed 4001 is actually *better* than baseline at lambda 0.03. Reading the arm means
as a dose-response curve would overstate what three seeds support.

For scale, the dead-adversary arm at lambda 1.0 scored 0.4103, in between the
0.1 and 0.3 live-adversary arms. Making the adversary work changed the outcome
very little, which is itself the clearest statement of the result.


### KIBA: the same method is significantly worse there

Measured on a T4, KIBA `split_cold_target` (the s4001 gene partition), seeds
4001-4003, `--lambda-max 0.03`, discriminator input `protein`, `--max-epochs 80`
and `--patience 80` on every run so neither arm stops early. Six runs,
281.6 min of grid wall time. Results under `outputs/kiba-dann/<arm>/`.

Lambda 0.03 was the only value run here: it is the arm that was closest to the
baseline on DAVIS, so it is the arm with the best chance of helping.

**The adversary trained in all three DANN runs.** `domain_loss` fell to
0.5546, 0.6047 and 0.6020 against ln 2 = 0.693147, and `domain_accuracy`
reached 0.712, 0.664 and 0.669. The gate passes, so these numbers measure the
method and not a dead gradient.

| arm | test MSE | test CI | val MSE | best epoch |
| --- | --- | --- | --- | --- |
| baseline | 0.3501 (0.0281) | 0.7715 (0.0063) | 0.4196 (0.0074) | 80, 50, 59 |
| DANN, lambda 0.03 | 0.4229 (0.0259) | 0.7489 (0.0133) | 0.4983 (0.0372) | 33, 29, 14 |

Paired by seed, DANN is **worse on every seed in both metrics**: test MSE
+0.0700, +0.0603, +0.0882 (mean +0.0728, t(2) = +8.92, p = 0.012) and test CI
-0.0314, -0.0192, -0.0173 (mean -0.0226, t(2) = -5.12, p = 0.036). Three seeds
give only two degrees of freedom, so read the p-values as supporting evidence
rather than as a result on their own; the sign agreement across 3 of 3 seeds and
both metrics is the stronger part of the claim.

**Conclusion: adversarial domain alignment does not help cold-target
generalisation on either benchmark, and on KIBA it hurts.** On DAVIS it was
indistinguishable from the baseline at this lambda (+0.0038 MSE, t(2) = +0.28);
on KIBA the same configuration costs 0.073 MSE, a 21% relative increase over a
0.350 baseline. That is the project's Phase 3 conclusion.

Two observations worth keeping, neither of which is a rescue of the method:

*   **The KIBA adversary was weaker than the DAVIS one and still did more
    damage.** Its `domain_accuracy` peaked at 0.664-0.712 against 0.746-0.751 on
    DAVIS, and its departure from ln 2 was about half as large. So the size of the
    regression penalty is not explained by how well the discriminator separated
    the domains.
*   **DANN peaks much earlier and then degrades.** Its best epochs are 14, 29
    and 33 against the baseline's 50, 80 and 59, while both arms run the full 80.
    The reported metrics come from `best.pt`, selected on validation MSE, so this
    is not an early-stopping confound: DANN is measured at its own best point and
    still loses.

Caveats on scope: one split partition (s4001), three training seeds, one lambda.
The 80-epoch cap also compresses Ganin's lambda ramp relative to the 100-epoch
DAVIS runs, because the schedule is normalised by `max_epochs` -- at seed 4003's
best epoch 14 the effective lambda is about 0.021, not 0.03.

Measured DANN overhead on KIBA: **42.08 s/epoch against 28.33 s/epoch for the
baseline, 1.49x.** This replaces the 1.53x that had been assumed from DAVIS for
budgeting.

The kernel itself reported ERROR. All six runs finished `rc=0` with
`FAILED RUNS: none` and `SKIPPED FOR BUDGET: none`; the failure was in the
notebook's own summary cell, which read `domain_loss` out of every
`history.csv` including the baseline arm's, where that column does not exist.
The results were written and pulled intact.

## Phase 4: mutant stress test

DAVIS carries real kinase point mutants with measured affinities, so a wild-type/mutant pair
measured against the same ligand is a question with ground truth: the measured affinity shift
is known, and the model's predicted shift can be scored against it. 44 usable mutants give
2992 comparable triples.

```powershell
uv run python -m src.eval_mutants --run outputs/phase2/davis/split_mutant/seed4001 --run outputs/phase2/davis/split_mutant/seed4002 --out outputs/phase4/stress_test.json
```

`src/eval_mutants.py` takes one or more run directories, each holding `best.pt` and
`config.json`, and reports per run and per stratum: the spread of measured against predicted
shifts, their ratio, Pearson and Spearman correlation, MAE, RMSE, the slope of measured on
predicted, and sign agreement among shifts of at least `--min-shift` (default 0.5 pKd, because
the sign of a near-zero shift is noise). It is CPU inference over a few thousand pairs, so
`--device cpu` is the default and is the right choice.

A mutant counts as usable when it is `constructible`, its own sequence and its wild type's are
both in the target table, and at least one ligand was measured against both.

### The split decides whether this measures anything

Mutants share a gene with their wild type, and the cold-target split is by gene, so the two
always land in the same partition. Under `split_cold_target` all 2992 mutant pairs fall in
train (2244) or val (748) and **none in test**: every DAVIS cold-target checkpoint has already
seen the pairs the stress test asks about. `split_mutant` exists for this, and puts all 2992
mutant-target pairs in test with the wild types in train (451) and val (93).

Both runs are informative, but they answer different questions, so the tool prints which it is
rather than leaving the reader to work it out. A run whose split is not `split_mutant` is
labelled `CONTAMINATED FOR THIS PURPOSE`, with the train/val/test counts for its own split, and
is explicitly not a generalisation result.

On the three `split_cold_target` seeds, visible mutants with both measurements uncensored
(n = 1127):

| seed | measured sd | predicted sd | sd ratio | Pearson r | MAE |
| --- | --- | --- | --- | --- | --- |
| 4001 | 0.723 | 0.381 | 0.527 | +0.664 | 0.383 |
| 4002 | 0.723 | 0.481 | 0.666 | +0.713 | 0.365 |
| 4003 | 0.723 | 0.629 | 0.870 | +0.803 | 0.293 |

So the architecture is not blind to point mutations, but it compresses them in every seed, and
the seed spread is wide enough that a single seed would have misled. Since these pairs were in
training, this is a statement about what the architecture can represent, and the next section
shows it does not survive being held out.

### Held out: the model predicts that point mutations do nothing

Three seeds trained on `split_mutant` (`outputs/phase2/davis/split_mutant/seed*`), where all
2992 mutant pairs are the test set. Same stratum as above, visible mutants with both
measurements uncensored, n = 1127.

| arm | sd ratio | Pearson r | MAE |
| --- | --- | --- | --- |
| held out (`split_mutant`) | 0.156 | +0.140 | 0.485 |
| contaminated (`split_cold_target`) | 0.688 | +0.727 | 0.347 |

Per seed, held out: sd ratio 0.242 / 0.096 / 0.131, r +0.166 / +0.027 / +0.229.

**The comparison that settles it.** A predictor that always answers "this mutation changes
nothing", a constant zero shift, scores MAE 0.4896 on this stratum. The held-out model scores
0.4850, an improvement of 0.0046, or 0.9% of the no-change error. On mutants it has not seen,
the model is for practical purposes indistinguishable from assuming point mutations have no
effect. The ratio of predicted to measured spread, 0.156, says the same thing: it emits a
nearly constant shift whatever the edit is.

The contaminated arm is what memorisation looks like. With the same architecture and the same
evaluation, r rises from +0.14 to +0.73 and MAE falls to 0.347 (29% better than no-change)
purely because those pairs were in training. **Had the stress test been run only on the
existing cold-target checkpoints, as originally planned, the conclusion would have been that
the model handles point mutations well. It was wrong by the width of that gap.**

### Why the standard metrics do not show this

On its own test set the `split_mutant` model looks good, better than on cold-target:

| seed | test MSE | test CI | uncensored MSE | uncensored CI |
| --- | --- | --- | --- | --- |
| 4001 | 0.3180 | 0.9124 | 0.5573 | 0.8190 |
| 4002 | 0.3969 | 0.8915 | 0.7212 | 0.7985 |
| 4003 | 0.3942 | 0.8968 | 0.7212 | 0.8058 |

A concordance index near 0.90 on the mutant test set sits beside a near-total failure to
predict any mutation effect. Both are true: each test pair is a (drug, mutant protein) pair
whose affinity is largely determined by the drug and by the kinase it belongs to, and the model
predicts that well. The point edit is a small correction on top, and that correction is what it
misses. So **aggregate MSE and CI on a mutant split cannot detect this failure**, and reporting
them alone would overstate what the model does. The paired wild-type/mutant shift is the
measurement that exposes it, which is the reason this phase exists.

Note also that the gap between `all` and `uncensored` MSE runs the other way from intuition:
0.318 against 0.557 for seed 4001. Censored pairs are the easy ones, since predicting the pKd
5.0 floor is nearly free, so including them flatters the model.

### Two measurement details that change how the output reads

**Truncation.** Protein tokens are capped at 1200 residues, so an edit past position 1200
produces a token sequence identical to the wild type's and the model cannot distinguish them.
Three mutants are affected: LRRK2(G2019S), MET(M1250T) and MET(Y1235D), whose full sequences
run to 2527, 1390 and 1390 residues with edits at 2019, 1250 and 1235. They are detected by
comparing token arrays, **not** by the mutant table's `n_edits`, which counts edits in the
untruncated sequence and reports all three as edited. They are reported as a separate
`invisible_control` stratum and double as a pipeline check: identical tokens must give
identical predictions, so a nonzero shift there means the comparison was built wrongly. It
comes out at exactly 0. Their measured shifts have a spread of 0.305 pKd, so this is real
signal the model structurally cannot reach.

**Censoring.** DAVIS is left-censored at 10,000 nM, which is pKd 5.0, and 1788 of the 2992
triples have at least one measurement on that floor. A floor-to-floor pair has a measured shift
of exactly zero for reporting reasons rather than biological ones, which would otherwise look
like a shift the model should predict. The headline stratum is therefore
`visible_uncensored`; `visible_all` is reported beside it and is the more flattering of the two
on MAE precisely because so many of its shifts are trivially zero.

## Phase 5: two non-adversarial attacks on the cold-target deficit

Phase 3 established that aligning feature marginals does not help, and that the baseline
under-disperses its predictions across unseen targets before any adaptation is applied. Phase 5
tests the reading that follows from that: the deficit is in the protein representation, so
changing the representation should help where adaptation did not. Neither method here does any
domain adaptation.

### `src/kmer_baseline.py`: a non-neural similarity baseline, no GPU

For an unseen target and a drug, predict a similarity-weighted average of the scores that drug
was measured at against the training targets most similar to the unseen one. Similarity is the
cosine between amino-acid k-mer count vectors (`--kmer`, default 3). Nothing is learned, so
there is no training seed and one number per partition. The neighbour count and a similarity
sharpening exponent are tuned on validation **independently per partition**, because the tuned
values differ across split seeds.

```bash
uv run python -m src.kmer_baseline --dataset davis --split split_cold_target
```

Writes `outputs/kmer/<dataset>/<split>/metrics.json`, which carries the same `test` block as a
training run plus three extra blocks: `leakage_check` (the largest cosine similarity between any
test target and any training target, and how many test targets exceed 0.95), `coverage` (how
many test pairs fell back to the global training mean because the drug has no training
measurement), and `within_ligand` (within-ligand CI, predicted and measured spread and their
ratio, on the matched ligand set).

Read `leakage_check` before the metrics. The threshold is 0.95 rather than 1.0 because Phase 1
reconstructs DAVIS mutants from their wild-type sequence, so a mutant and its wild type sit near
0.99; the gene-level split keeps them in the same partition, and this check is what confirms it.
Read `coverage` next: on a cold-*target* split every test drug appears in training, so the
fallback never fires and the method is never tested on the case it cannot handle. It also cannot
predict outside a drug's observed score range. It is a baseline for this split, not a model.

### `--protein-encoder esm`: a frozen protein language model branch

`--protein-encoder esm` replaces `ProteinEncoder` (embedding plus three valid convolutions over
1200 or 1000 characters, global max-pooled) with `EsmProteinEncoder`: a frozen mean-pooled
ESM-2 embedding per target, projected by one `Dropout -> Linear(esm_dim, filters[-1]) -> ReLU`.
The projection width is `config.filters[-1]`, so the fusion width and the whole regression head
are identical in shape to the CNN arm and the protein representation is the only variable. The
language model is never fine-tuned.

Embeddings are precomputed once, because a fixed feature does not belong inside a training loop:

```bash
uv run python -m src.esm_embed --sources davis kiba external   # needs a GPU in practice
uv run python -m src.train --dataset davis --split split_cold_target     --protein-encoder esm --max-epochs 100 --patience 100
```

`src/esm_embed.py` writes `data/processed/{source}_esm.npz` with a `prot_embeddings` array in
processed-table row order, so `prot_index` indexes it exactly as it indexes the token array.
`src/data.py` substitutes that array for the protein tokens and validates its shape against the
target table. Run it on a GPU: on CPU the 650M model measured 8.64 s/sequence, which is about
110 minutes for the 766 sequences, against roughly 5 minutes on a T4.

**`--max-residues` defaults to 1022, and that is not a safe default for mutant work.** On DAVIS
it makes 6 of 423 embeddings duplicates of another row: PIK3CA with H1047L, H1047Y and M1043I,
MET with M1250T and Y1235D, and LRRK2 with G2019S all carry their mutation past residue 1022, so
the kept prefix is identical to the wild type's. The cold-target splits are unaffected, because
the gene-level split keeps a gene's mutants in one partition and no collapsed group straddles a
boundary in any of the four splits. Phase 4 is a different matter: the branch cannot tell those
mutants from their wild type at all. Either raise the limit (ESM-2 uses rotary position
embeddings and runs past 1024) or centre the window on the mutation. The kernel's verification
cell prints a duplicate-row count per source; read it.

`--protein-encoder cnn` is the default and nothing about it changed. ESM runs are written to a
separate root (`outputs/esm/...` rather than `outputs/phase2/...`) so the two can never share a
run directory, and `config.json` records `protein_encoder`, `esm_dim` and `esm_proj_dim` so a
checkpoint can be reloaded without being told which branch or width it used.

### `--esm-proj-dim`: the projection width is a confound, so it is a flag

The 96-wide default is a 13x compression of a 1280-wide embedding, applied before the
regression head ever sees it. It was chosen for comparison cleanliness, not for the model, so a
flat within-ligand concordance in the first ESM grid may be that choice rather than the protein
language model. `--esm-proj-dim` varies it and nothing else:

```bash
uv run python -m src.train --dataset davis --split split_cold_target     --protein-encoder esm --esm-proj-dim 512 --max-epochs 100 --patience 100
```

`0`, the default, means `config.filters[-1]` and reproduces the earlier runs byte-for-byte. Any
other value widens `Linear(esm_dim, esm_proj_dim)` and the regression head's input with it, so
the arms are no longer parameter-matched: 96 gives 1,953,185 parameters, 256 gives 2,321,985,
512 gives 2,912,065 and 1280 gives 4,682,305. `DrugEncoder`, `ProteinEncoder` and
`EsmProteinEncoder` each expose `out_dim`, and `DeepDTA` sizes its head from
`drug_encoder.out_dim + protein_encoder.out_dim` instead of assuming both branches are
`filters[-1]` wide.

A widened run writes to `outputs/esm-proj<width>/...`, a root of its own, so the 96-wide grid's
results are never overwritten and stay comparable. The CNN arm is untouched at every width
(head input 192, 1,932,033 parameters).

The Kaggle kernel `kaggle/sc4001-phase5-davis-projwidth/` runs widths 256, 512 and 1280 over
three DAVIS seeds and **selects the width on validation MSE, never on test**. Unlike the kernel
below it needs no internet, because the embeddings now ship in the dataset.

### `src/kmer_offsets.py` and `--kmer-residual`: hand the network the similarity signal

A zero-parameter k-mer baseline beats the CNN on 7 of 8 cold-target partitions, and widening the
ESM projection buys fit but not ordering. Both say the network cannot extract target similarity
from roughly 400 sequences. `--kmer-residual` supplies it instead, as an additive offset:

```text
prediction = kmer_neighbour_prediction + network(drug, protein)
```

```bash
uv run python -m src.kmer_offsets --dataset davis --split split_cold_target
uv run python -m src.train --dataset davis --split split_cold_target     --protein-encoder esm --esm-proj-dim 1280 --kmer-residual     --max-epochs 100 --patience 100
```

`src/kmer_offsets.py` writes `data/processed/{dataset}_{split}_kmer_offset.npz`, one offset per
row of the pairs table in table row order, so `src/data.py` indexes it exactly as it indexes the
token arrays. It reads the neighbour count and sharpening that `src/kmer_baseline.py` tuned on
validation from `outputs/kmer/<dataset>/<split>/metrics.json`, or takes both on the command line.

**A training pair's offset must not come from training data containing that pair**, and by
default it does. With the tuned settings the k-mer predictor nearly reproduces its own training
scores, because a pair's own target sits at similarity 1.0:

| offset MSE | DAVIS | KIBA |
| --- | --- | --- |
| train, in-sample (**never used**) | 0.0216 | 0.0005 |
| train, out-of-fold (used) | 0.3999 | 0.3531 |
| val | 0.3088 | 0.3423 |
| test | 0.3516 | 0.2788 |

A network trained on in-sample residuals would learn to correct an error roughly a quarter the
size of the one it meets at test time, and would look excellent while meaning nothing. Training
pairs therefore get **out-of-fold** offsets: target groups inside the training set are split
into `--folds` folds (5 by default) and each fold is predicted from the others, reproducing the
cold-target condition. The module asserts that no fold shares a target row with its own fitting
set, and prints both figures above so the gap is visible rather than assumed. Validation and
test pairs are predicted from the whole training set, which is what a deployed model has.

`PairDataset` carries the offset as a fourth collated item and adds it to the model's output in
both `src/train.py` and `src/train_dann.py`. The offset is all zeros when `--kmer-residual` is
absent, so every earlier arm, Phase 3 included, is unchanged. Residual runs are written to
`outputs/kmer-residual-<phase>/...`. The Kaggle kernel is
`kaggle/sc4001-phase5-kmer-residual/`; each of its arms is paired against a width-matched
control that already exists, and its success criteria are stated in the notebook before the
grid runs.

The Kaggle kernel `kaggle/sc4001-phase5-esm-coldtarget/` embeds and trains in one session. It is
the only kernel here that needs **internet enabled**, because the ESM-2 weights are downloaded
from Hugging Face; every other kernel runs with internet off. The dataset
`isaacwjk21/sc4001-project` now carries the three `*_esm.npz` files, so later kernels attach
them instead of spending roughly five GPU minutes recomputing bytes that already exist.

### Scoring the residual directly: an endpoint that looks sound and is not

With `--kmer-residual` the prediction is `offset + network`, so it is tempting to measure the
network on its own by the within-drug concordance between its output and the **residual truth**
`score - offset`, and to call 0.5 the chance level. That endpoint is unusable, and the reason is
worth keeping because the mistake is attractive rather than careless.

Set the network output to exactly minus the offset. The full prediction is then constant, so its
within-ligand CI is **0.5000** and it orders nothing at all. Yet its residual-ordering CI is
**0.7644** across all 66 matched DAVIS drugs, above 0.5 for every one of them, and still
**0.6576** when censored pairs are dropped. A predictor carrying no protein information scores
0.76.

Two causes, both structural:

* The offset is a similarity-weighted average of the drug's own training scores, so it cannot
  leave that drug's observed range and it shrinks extremes. The residual is therefore
  anti-correlated with the offset by construction, and anything tracking `-offset` scores well.
* Censoring dominates the comparison set. 53.2% of within-drug test pairs on DAVIS cold-target
  are floor-floor, both scores at the pKd floor of 5.0. Their residual difference is exactly
  minus their offset difference, which invents an ordering where the assay reported a tie. The
  measured-score CI in `src/metrics.py` correctly drops those pairs as ties; the residual target
  silently reinstates them.

**Use instead** the paired per-drug delta on the full prediction against the offset alone,
`CI_d(y, offset + network) - CI_d(y, offset)`, averaged over the same matched drug set, with the
floor-versus-above-floor subset reported separately and the `network = -offset` control reported
beside it. The control is cheap and it is what makes the headline number interpretable.

## Version control

The project is tracked from the repository root, one level above this directory.

* `.gitattributes` pins `*.sh` to LF. A `#SBATCH` line carrying a stray CR makes a Slurm submit
  fail, and `core.autocrlf` would otherwise rewrite the job scripts on checkout.
* `outputs/**/*.pt` stays untracked: checkpoints are about 3.2 GB.
* `outputs/kaggle_logs/*/sc4001/` stays untracked. Kaggle echoes the whole staged dataset back
  inside each pulled output, so those directories are a third copy of files already tracked at
  their real paths; ignoring them takes the tracked tree from 66 MB to 10 MB. The logs, configs,
  histories and metrics beside them *are* tracked.
* `.env` is ignored and holds the cluster and Kaggle credentials. Nothing in it is ever printed,
  committed or passed on a command line.

## Cluster job scripts

`jobs/` holds scheduler scripts for the NTU clusters, named after the run they submit, as
`cluster-hygiene.md` requires. Kaggle kernels live in `kaggle/<slug>/` and are unrelated.

`jobs/sc4001-phase3-kiba-s4004-a.sh` and `-b.sh` are Slurm scripts for TC1: KIBA
`split_cold_target_s4004`, baseline against DANN at `--lambda-max 0.03`, three seeds, both arms
at `--max-epochs 80 --patience 80`, matching the Kaggle s4001 grid so the two partitions stay
comparable. Now that the s4001 grid has been measured, their remaining purpose is
narrower: whether the KIBA deficit is partition-dependent, the same question the
four-way DAVIS spread answered for the baseline. They use the shared environment
`/tc1apps/2_conda_env/CZ4042_v6`, so no conda
environment is built and nothing is installed into the cluster home directory. Results go to
`~/sc4001-data/results/<run-name>/<arm>/` and logs to `~/sc4001/logs/<job-name>/`, which must
exist before `sbatch` or the job fails immediately.

They are **split by seed rather than by arm**: job A runs seeds 4001 and 4002 for both arms, job
B runs seed 4003. If only one job ever runs, the surviving results are complete baseline/DANN
pairs, which is how the comparison is read. Splitting by arm would have risked three baselines
and no DANN.

**Neither has been submitted.** TC1 is a single shared account, `iwong008`, used concurrently by
another project, and its QoS `normal` allows two submitted jobs and one running GPU job per
user. That project's sweep resubmits its successor from inside the running job, so it needs the
second slot free at every turnover: a single *pending* job from this project would make its
`sbatch` fail and stall the sweep until someone restarted it by hand. Check what is already
queued, with `squeue -u <ntu_id>` or by asking the other session, before submitting anything.
The scripts are kept ready for when the cluster is actually free.

Aspire2A (NSCC, PBS Pro rather than Slurm) has no scripts here yet. Recorded for when it does:
the GPU queue `g1` allows one running job per user and one GPU per job with a 24 h walltime,
submission is `qsub -P <project> -q normal`, compute nodes have no git so cloning happens on the
login node, and the resource line takes the form
`#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb`. Service units are billed per CPU hour and per GPU
hour with no memory component. These figures come from the other project's session rather than
from NSCC documentation, including its note that scratch is purged after about 30 days, so
confirm them before relying on any of it.
