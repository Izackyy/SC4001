# SC4001 update: the cold-target deficit is representational

Written for: the project group and whoever reviews the Phase 3 write-up.

**Read this first: the conclusion has moved on since sections 1-6 were written.** Two further
experiments ran on 6 November. Section 7 tested whether the ESM projection width was the reason
the frozen branch did not order unseen targets -- it was not, and that experiment failed its own
pre-registered bar. Section 8 then supplied the k-mer signal directly as an additive offset, and
**that arm beats every learned-protein-encoder arm in the project by large, consistent margins
and closes most of the under-dispersion gap.** The project is no longer an all-negative study:
it is a negative diagnosis plus a fix derived from it. Sections 1-6 are unchanged so that
anything already quoted from them still reads correctly, but section 5's recommendation is
superseded -- section 8 says how.

Two methods were tried against the unseen-target problem that Phase 3's DANN work failed to
solve. Both bypass domain adaptation entirely and both beat the character-CNN baseline. That
changes what the DANN negative result means, and it is the argument this document makes.

## 1. What ran

| | method | compute | status |
| --- | --- | --- | --- |
| Path 2 | k-mer protein similarity, k-nearest-neighbour over each drug's own training scores | CPU only, no GPU, nothing learned | done, 8 partitions (2 datasets x 4 split seeds) |
| Path 1 | frozen ESM-2 650M protein embedding replacing the protein CNN | Kaggle T4, 25 min of the ~90 left | done, 6 runs (2 datasets x 3 seeds), none failed |

New code: `src/kmer_baseline.py`, `src/esm_embed.py`, and a `protein_encoder` switch threaded
through `src/config.py`, `src/data.py`, `src/model.py`, `src/train.py`. The CNN path is
unchanged and its run directories are byte-identical, so nothing already measured moves.

## 2. Path 2: a non-neural k-mer baseline beats the CNN on 7 of 8 partitions

For an unseen target `t` and drug `d`, predict a similarity-weighted average of the scores `d`
was measured at against the training targets most similar to `t`. Similarity is the cosine
between amino-acid 3-mer count vectors. Nothing is trained. Neighbour count and a similarity
sharpening exponent are tuned on validation, independently per partition.

CNN column is the mean over 3 training seeds of the **Phase 2** baseline on the same
partition, which stops early at patience 15. That is the arm with results on all four
partitions, and the k-mer method has no epoch budget to match anyway. Section 3 compares
against a different arm -- the equal-compute baseline at 100 or 80 epochs with patience
equal to max-epochs -- so the CNN numbers there are higher (DAVIS all MSE 0.3991 rather
than 0.3828). Do not read a CNN figure across the two tables.

| dataset | partition | CNN MSE | k-mer MSE | CNN CI | k-mer CI | CNN MSE uncens. | k-mer MSE uncens. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DAVIS | s4001 | 0.3828 | **0.3516** | 0.7919 | **0.8385** | 1.0578 | **0.6875** |
| DAVIS | s4002 | 0.5527 | **0.4260** | 0.8060 | **0.8618** | 1.3875 | **1.0551** |
| DAVIS | s4003 | 0.4533 | **0.3815** | 0.8065 | **0.8355** | 1.3146 | **0.9065** |
| DAVIS | s4004 | **0.5567** | 0.5769 | 0.7744 | **0.7809** | **1.4126** | 1.4281 |
| KIBA | s4001 | 0.3682 | **0.2788** | 0.7668 | **0.7871** | - | - |
| KIBA | s4002 | 0.3716 | **0.3095** | 0.7316 | **0.7568** | - | - |
| KIBA | s4003 | 0.4143 | **0.3262** | 0.7509 | **0.7993** | - | - |
| KIBA | s4004 | 0.5331 | **0.3999** | 0.7318 | **0.7715** | - | - |

The one loss is DAVIS s4004, the hardest partition for both methods. On DAVIS s4001 the
uncensored MSE drops 35% (1.0578 to 0.6875); on KIBA s4004 the all-pairs MSE drops 25%.

### Within-ligand views, which is where the CNN actually fails

Pooled CI rewards ordering one drug against another and reads about 0.72 even for a predictor
that returns a single constant per drug. The honest views are concordance *inside* each drug's
own measured test pairs, and the spread of a drug's predictions across the full cold-target
grid. Both are on the matched ligand set (drugs with at least two measured test pairs and a
non-zero measured spread).

| dataset | partition | k-mer within-ligand CI | k-mer spread / measured | max test-train similarity |
| --- | --- | --- | --- | --- |
| DAVIS | s4001 | 0.7494 | 0.771 | 0.695 |
| DAVIS | s4002 | 0.8093 | 0.758 | 0.741 |
| DAVIS | s4003 | 0.7629 | 0.675 | 0.830 |
| DAVIS | s4004 | 0.6515 | 0.488 | 0.762 |
| KIBA | s4001 | 0.6467 | 0.401 | 0.695 |
| KIBA | s4002 | 0.6685 | 0.433 | 0.762 |
| KIBA | s4003 | 0.6868 | 0.440 | 0.695 |
| KIBA | s4004 | 0.6869 | 0.347 | 0.830 |

Against the CNN baseline on s4001: within-ligand CI 0.7494 vs **0.6789** on DAVIS and 0.6467 vs
**0.6363** on KIBA; dispersion ratio 0.771 vs **0.576** on DAVIS. On KIBA the k-mer model wins
MSE and CI while dispersing *less* than the CNN (0.401 vs 0.514), which is the shrinkage point:
a compressed predictor can be both accurate and well-ordered, so dispersion is a diagnostic of
what the model can separate, not a proxy for accuracy. Report both or neither.

### Leakage check

Maximum cosine similarity between any test target and any training target is 0.695 to 0.830
across the eight partitions, with zero test targets above the 0.95 threshold. The gene-level
split keeps each reconstructed mutant with its wild type, so the near-identical pairs that would
have inflated this never straddle the boundary.

### What the method cannot do

It inherits each drug's observed score range and cannot leave it, cannot predict for a drug with
no training measurements at all, and is undefined for a genuinely new chemical series. On these
partitions every test pair involves a drug seen in training (the split is cold-*target* only),
so the fallback to the global training mean was used **0 times in all 8 partitions** — the
method is never tested on the case it cannot handle. It is a strong baseline for this split and
not a deployable model.

## 3. Path 1: a frozen ESM-2 branch buys dispersion, not ordering

`--protein-encoder esm` replaces the protein CNN with a frozen mean-pooled
`facebook/esm2_t33_650M_UR50D` embedding (1280-d), projected by one
`Dropout -> Linear(1280, 96) -> ReLU`. Everything else is identical to the CNN arm: same drug
CNN, same 96+96 fusion, same 1024/1024/512 head, same splits, same seeds, 100 epochs on DAVIS
and 80 on KIBA with patience equal to max-epochs. All 6 runs completed, none failed, and the
wall-clock guard never fired. **Cost: 25 min of the roughly 90 available** (5 min to embed 766
sequences on the T4, 23.9 min for the grid), because dropping the protein CNN makes each epoch
**9.1x faster on DAVIS and 7.9x on KIBA**.

Paired by seed against the **equal-compute** CNN baseline, the one that also ran 100 or 80
epochs with no early stopping. Those are the only CNN numbers the ESM arm can be paired
with, and they differ from the Phase 2 means quoted in section 2. Seed signs are the sign
of ESM minus CNN for seeds 4001, 4002 and 4003 in that order. 3 seeds, 2 df:

| dataset | view | metric | CNN | ESM | delta | t(2) | p | seed signs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DAVIS | all | MSE | 0.3991 | 0.4020 | +0.0029 | +0.25 | 0.828 | + - + |
| DAVIS | all | CI | 0.7885 | **0.8088** | **+0.0203** | +7.99 | **0.015** | + + + |
| DAVIS | uncensored | MSE | 1.1237 | **1.0103** | **-0.1134** | -2.90 | 0.101 | - - - |
| DAVIS | uncensored | CI | 0.6502 | 0.6663 | +0.0161 | +2.35 | 0.143 | + + + |
| KIBA | all | MSE | 0.3501 | **0.3113** | **-0.0388** | -2.35 | 0.143 | - - - |
| KIBA | all | CI | 0.7715 | 0.7663 | -0.0052 | -0.65 | 0.581 | + - - |

Only DAVIS pooled CI clears significance at three seeds. The MSE improvements on DAVIS
uncensored and on KIBA have all three seed signs in the right direction but do not reach
p < 0.05 at 2 df. DAVIS all-pairs MSE is flat, and censored pairs are 3009 of its 4284 test
pairs, so the gain sits in the real measurements and is diluted by the pKd 5.0 floor.

### This is the predicted-failure case, and it was named in advance

The kernel's own design notes said: *"ESM improves MSE but not within-ligand CI or prediction
spread. That would mean it bought calibration, not the ability to separate unseen targets."*
That is what happened, except that the spread half went the other way.

| dataset | within-ligand CI (CNN / ESM / k-mer) | spread / measured (CNN / ESM / k-mer) |
| --- | --- | --- |
| DAVIS | 0.6789 / 0.6867 / **0.7494** | 0.576 / 0.622 / **0.771** |
| KIBA | 0.6363 / 0.6447 / 0.6467 | 0.514 / **0.684** / 0.401 |

Matched ligand set throughout: 66 of 68 DAVIS and 1436 of 2025 KIBA test drugs, measured spread
0.6391 and 0.5762. No drug has a predicted spread of zero in any ESM run, so no run is in the
degenerate per-drug-constant mode.

**Provenance of the CNN baseline figure.** The DAVIS CNN within-ligand CI quoted here and
elsewhere as 0.6789 is the **no-early-stop** run, `outputs/nostop/baseline/phase2/davis/
split_cold_target`, scored from `best.pt` over seeds 4001-4003. The early-stopped run under
`outputs/phase2/davis/split_cold_target` is a different number, 0.6818, and `last.pt` gives
0.6901 and 0.6906 respectively. The k-mer (0.7494), frozen ESM (0.6867) and hybrid (0.7460)
figures reproduce from `outputs/` without this ambiguity. Any recomputation must name the run and
the checkpoint, and metrics will not match if it picks the other CNN directory.

The ESM branch **widens** its predictions substantially -- the KIBA dispersion ratio moves 0.514
to 0.684 -- but its within-ligand concordance moves only +0.0078 (DAVIS) and +0.0084 (KIBA),
which is inside seed noise (DAVIS per-seed 0.6778 / 0.6692 / 0.7132). Wider predictions that are
not better ordered. The non-neural k-mer method beats ESM on every view on DAVIS, including
uncensored MSE 0.6875 against 1.0103, and on KIBA all-pairs MSE 0.2788 against 0.3113.

### Three confounds, any of which could explain the weak result

1. **The `Linear(1280, 96)` bottleneck.** The projection width was set to `filters[-1]` so the
   fusion width and the regression head would stay identical to the CNN arm. That is the clean
   comparison and also a 13x compression of the embedding. It is the first thing to vary.
2. **Mean-pooling over up to 1022 residues.** Kinase binding is local, and averaging the whole
   chain dilutes it. Published cold-start DTA work keeps per-residue embeddings and pools them
   with a CNN or attention rather than taking a flat mean.
3. **Truncation at 1022 residues silently collapses six DAVIS targets.** The embedding has 6
   duplicate rows among 423 distinct sequences: PIK3CA with its H1047L, H1047Y and M1043I
   mutants, MET with M1250T and Y1235D, and LRRK2 with G2019S. Every one of those mutations sits
   past residue 1022, so the kept prefix is identical. **No group straddles a partition in any
   of the four cold-target splits** -- the gene-level split keeps a gene's mutants together --
   so this leaks nothing and does not affect the numbers above. But it means the ESM branch as
   configured cannot tell those mutants from their wild type, which bounds any Phase 4 mutant
   work built on it. ESM-2 uses rotary position embeddings and can run past 1024, or the window
   can be centred on the mutation instead.

## 4. What this does to the Phase 3 DANN result

The honest reading is not "our DANN implementation failed". It is that **DANN optimised the
wrong thing, and the thing it optimised is provably unable to help here.** Four conditions that
would have made this an experimental failure were checked and ruled out:

1. **The adversary trained.** Validity gate against `ln 2 = 0.693147`: domain loss departed
   chance on all three KIBA seeds (domain accuracy 0.664-0.712) and all DAVIS seeds
   (0.746-0.751). Not a dead gradient, not a broken gradient reversal.
2. **Compute was equal.** Both arms ran 100 epochs on DAVIS and 80 on KIBA with
   `--patience` set to `--max-epochs`, so early stopping could not decide the result.
3. **Lambda was swept 1.5 orders of magnitude** (0.03, 0.1, 0.3, 1.0) under Ganin's schedule.
   Damage increased monotonically with lambda; there is no window where it helped.
4. **A dead-adversary control was run.** At lambda 1.0 the adversary sat at chance and the model
   still lost 0.0238 of prediction spread, so part of the damage is the extra branch and not
   alignment at all. The adversarial component is therefore *smaller* than the raw deltas look.

What remains is theory. Zhao et al. (ICML 2019) prove an algorithm-independent lower bound: when
the label marginals differ between domains, aligning feature marginals **increases** the joint
error. That is an impossibility result about DANN's objective, not a tuning statement. Our
setting violates the assumption badly — the unlabelled target pool is validation targets plus
114 external kinases with no labels at all, so the target label marginal cannot even be
estimated, let alone matched. KIBA, with the larger target set and the larger label shift, is
hurt more than DAVIS, which is the direction the bound predicts.

And the positive finding is now triangulated three ways. The baseline under-disperses to 0.576
(DAVIS) and 0.514 (KIBA) of measured within-ligand spread *before any adaptation*; MSE damage
and spread loss move together across arms at r = -0.951; and two methods that do no domain
adaptation whatsoever both improve on the CNN. **The cold-target deficit is in the protein
representation. Feature alignment cannot add information a representation never had.** That is a
result about when DANN applies, which is more useful than a result about whether our lambda was
right.

The two methods do not support that equally, and the write-up should not pretend they do. The
k-mer baseline improves ordering as well as accuracy (within-ligand CI 0.7494 against the CNN's
0.6789 on DAVIS). The frozen ESM branch improves accuracy and dispersion but leaves
within-ligand CI flat, so it is evidence that the representation was the binding constraint and
*not* evidence that a frozen mean-pooled protein language model is the fix. Stated carefully:
the deficit is representational, a method with no learned protein encoder at all already
exploits that, and the obvious protein-language-model fix has not yet been configured well
enough to.

## 5. Ways to save DANN, cheapest decisive first

| # | change | why it addresses the actual obstruction | cost |
| --- | --- | --- | --- |
| 1 | **Widen the ESM projection, or pool per-residue embeddings with attention** | Section 3 shows the frozen branch does not yet order unseen targets better, so running DANN on top of it would align a representation that still lacks the information. Fix the representation first. Also the cheapest run in the project, at 0.87 s/epoch. | 3 DAVIS seeds, ~5 min T4 |
| 2 | **DANN on top of a frozen-ESM branch that does separate targets** | Once the branch orders unseen targets, this is the one run that distinguishes "DANN is wrong here" from "DANN had nothing to work with". `data/processed/external_esm.npz` is already computed, so the unlabelled target domain needs no new embedding work. | 3 DAVIS seeds, ~15 min T4 |
| 3 | **CDAN** (Long et al. 2018): condition the discriminator on the prediction, not the feature alone | Conditional alignment is not subject to the marginal-alignment bound. DrugBAN uses exactly this for cold-start DTA. The repo already has a `joint` discriminator-input variant as a half-step. | code change + 3 seeds |
| 4 | **IWDAN / IWCDAN** (Combes et al., arXiv 2003.04475) | Importance-weights the alignment by estimated label proportions, which provably relaxes Zhao's obstruction under generalized label shift. For regression, bin pKd. | moderate |
| 5 | **Asymmetrically-relaxed alignment** (Wu et al., ICML 2019) | Replaces exact marginal matching with a relaxed divergence, sidestepping the bound directly. | moderate |
| 6 | **Enlarge the target pool to the whole human kinome** | 114 external kinases is a thin proxy for "all unseen kinases"; DANN's theory wants a target sample from the test distribution. | data work, no new method |
| 7 | **Discriminator capacity ablation** | Hidden width 64 is the one condition never varied: too weak gives an uninformative gradient, too strong collapses features. Declare it as a limitation until run. | 3 seeds per width |

Recommendation: run 1, then 2 only if 1 produces a branch whose within-ligand CI actually
moves, and report 3 as the method the literature endorses for this task. If DANN is also neutral
or negative on a representation that does separate unseen targets, the Phase 3 conclusion is as
strong as a three-seed study can make it, and the write-up says so without hedging.

## 6. Still unmeasured

* Whether the three confounds in section 3 -- projection width, mean-pooling, 1022-residue
  truncation -- account for the ESM branch's flat within-ligand CI. Until one of them is varied,
  "protein language models do not help here" is not a supportable claim; only "this
  configuration of one does not" is.
* Why the k-mer method beats a 650M-parameter protein language model on every view on DAVIS.
  The likeliest reason is that it conditions on the drug's own measurements against similar
  targets, which the ESM branch has to learn from scratch through a 96-d bottleneck.
* Roughly 65 min of the weekly GPU allowance is unspent. Confirm the remaining figure on
  kaggle.com rather than inferring it from this session.
* Discriminator capacity, target-pool size, and CDAN-style conditional alignment (section 5).
* The k-mer baseline is untested on unseen *drugs*, which it cannot handle by construction.


## 7. The ESM projection width was not the bottleneck

`sc4001-phase5-davis-projwidth`, COMPLETE, 9 of 9 runs rc=0, 14.1 min of T4. DAVIS cold-target,
projection widths 256, 512 and 1280 against the 96 already measured, three seeds each, 100
epochs with patience equal to max-epochs. The width was selected on **validation** MSE; test
metrics are reported for every width but were never used to choose.

| width | val MSE | test MSE | test CI | uncens MSE | within-ligand CI | spread / measured |
| --- | --- | --- | --- | --- | --- | --- |
| 96 | 0.4089 | 0.4021 | 0.8088 | 1.0103 | 0.6867 | 0.622 |
| 256 | 0.4006 | 0.4104 | 0.8034 | 0.9711 | 0.6754 | 0.659 |
| 512 | 0.3919 | 0.4026 | 0.8087 | 1.0345 | **0.7128** | 0.667 |
| **1280** | **0.3804** | **0.3823** | **0.8134** | **0.9703** | 0.7002 | 0.633 |
| cnn (equal compute) | - | 0.3991 | 0.7885 | 1.1237 | 0.6789 | 0.576 |
| k-mer | - | 0.3516 | 0.8385 | 0.6875 | 0.7494 | 0.771 |

**Validation MSE falls monotonically with width across all four points**, so validation selected
1280 -- no compression at all. At that width the branch is the best *neural* arm in the project
on DAVIS cold-target, beating the equal-compute CNN by 0.0168 on MSE and 0.0249 on pooled CI,
with 4.68M parameters showing no sign of overfitting at 100 epochs. It remains behind the
k-mer baseline.

**The pre-registered test failed, and that is reported as a failure.** The bar was a
within-ligand CI more than 0.02 above the 96-wide arm's 0.6867 with all three seeds in the same
direction. The width validation chose gives **+0.0135**, under the threshold. 512 clears the
mean threshold at +0.0261 but is not the selected width, and promoting it now would be selecting
on the outcome. Neither clears the sign condition: against the 96-wide per-seed figures of
0.6778 / 0.6692 / 0.7132, both 512 and 1280 give signs `+ + -`, because seed 4003's 96-wide run
was already the high outlier. No paired comparison against the 96-wide arm is significant --
every |t(2)| is at most 1.38 across all three widths and all four metrics.

**What this settles.** Confound 1 of the three listed in section 3 is tested and struck off: the
96-wide projection was part of the constraint but not the binding one. Extra width buys *fit*,
monotonically and without exception, and does not buy *ordering*. By elimination rather than by
assumption, whole-chain mean-pooling over up to 1022 residues is now the prime suspect for the
flat ordering, which is a sharper claim than section 3's "we cannot tell". Nothing in sections
1-6 needs retracting; the recommended next run changes, and it is no longer cheap, because
per-residue pooling needs a fresh embedding pass of roughly 2 GB.

## 8. Supplying the similarity signal: the first positive result

`sc4001-phase5-kmer-residual`, COMPLETE, 9 of 9 runs rc=0, 25.9 min of T4. The model becomes

```text
prediction = kmer_neighbour_prediction + network(drug, protein)
```

so the network learns only the correction. New module `src/kmer_offsets.py`, a `kmer_residual`
field in `Config`, a per-pair offset carried by `PairDataset`, and a `--kmer-residual` flag. The
offset is all zeros when the flag is absent, so every earlier arm including Phase 3 is
unchanged, which a plain CNN smoke run passing beside a residual one confirms.

### The leakage trap, which would have faked this result

A training pair's offset must not be computed from training data containing that pair, and by
default it is. With the tuned settings a pair's own target sits at cosine similarity 1.0 and
dominates its own weighted average, so the k-mer predictor nearly reproduces its own training
scores:

| offset MSE | DAVIS | KIBA |
| --- | --- | --- |
| train, in-sample (**never used**) | 0.0216 | 0.0005 |
| train, out-of-fold (used) | 0.3999 | 0.3531 |
| val | 0.3088 | 0.3423 |
| test | 0.3516 | 0.2788 |

A network trained on in-sample residuals would learn to correct an error roughly a quarter the
size of the one it meets at test time. It would have scored extremely well and meant nothing,
which is the worst class of mistake available here. Training pairs therefore get **out-of-fold**
offsets: target groups inside the training set are split into 5 folds, each predicted from the
other four, with an assertion that no fold shares a target row with its own fitting set.
Out-of-fold error lands beside test error and is slightly harder on both datasets, since a fold
fits on 4/5 of the training targets, and harder is the safe direction to err.

### Results

| arm | test MSE | test CI | uncens MSE | within-ligand CI | spread / measured |
| --- | --- | --- | --- | --- | --- |
| **hybrid, DAVIS, proj 96** | **0.3413** | 0.8284 | 0.6935 | 0.7460 | **0.795** |
| hybrid, DAVIS, proj 1280 | 0.3434 | 0.8304 | 0.7034 | 0.7465 | 0.787 |
| k-mer alone, DAVIS | 0.3516 | 0.8385 | **0.6875** | **0.7494** | 0.771 |
| ESM proj 1280, DAVIS | 0.3823 | 0.8134 | 1.0103 | 0.7002 | 0.633 |
| CNN, DAVIS | 0.3991 | 0.7885 | 1.1237 | 0.6789 | 0.576 |
| **hybrid, KIBA, proj 96** | 0.2823 | 0.7890 | - | **0.6710** | **0.522** |
| k-mer alone, KIBA | **0.2788** | 0.7871 | - | 0.6467 | 0.401 |
| ESM proj 96, KIBA | 0.3114 | 0.7663 | - | 0.6447 | **0.684** |
| CNN, KIBA | 0.3501 | 0.7715 | - | 0.6363 | 0.514 |

Paired against its own width-matched control, with every seed sign aligned:

| arm | test MSE | test CI | uncens MSE |
| --- | --- | --- | --- |
| DAVIS proj 96 | **-0.0608** (t=-6.00) | **+0.0195** (t=+7.80) | **-0.3167** (t=-5.36) |
| DAVIS proj 1280 | -0.0389 (t=-5.36) | +0.0170 (t=+10.76) | -0.2669 (t=-8.81) |
| KIBA proj 96 | -0.0290 (t=-7.24) | +0.0227 (t=+5.92) | - |

**Against every learned-protein-encoder arm the hybrid wins decisively and consistently.**
Uncensored DAVIS MSE, the view that excludes the pKd 5.0 floor and therefore measures the real
readings, falls 38% against the CNN: 1.1237 to 0.6935.

### Where it falls short of its own pre-registered bar

The bar was to beat **both** parents. It splits by dataset:

* **DAVIS: wins on accuracy, ties on ordering.** MSE 0.3413 beats k-mer's 0.3516. Within-ligand
  CI 0.7460 against k-mer's 0.7494 is **-0.0034 with signs `- - +`**, which fails the criterion
  and is realistically a tie.
* **KIBA: wins on ordering, loses narrowly on accuracy.** Within-ligand CI 0.6710 against
  0.6467 is **+0.0243 with all three seeds agreeing**. MSE 0.2823 against 0.2788 does not clear.

So on each dataset the hybrid beats both parents on one axis and ties or slightly loses on the
other. **No arm clears the full bar, and this is not written up as a pass.** The supportable
claim is that the learned model becomes competitive with the similarity baseline once it is
handed the similarity signal, not that it surpasses it; the network's contribution on top of the
offset is real but modest, and a reviewer will check that in thirty seconds.

One correction to the kernel's own output: it prints "the offset is doing the work and the
network adds nothing" for KIBA, a verdict triggered on MSE alone. That reading is wrong. KIBA's
within-ligand CI rises +0.0243 with all three seed signs positive, so the network does add
ordering there. The canned rule was too crude and the within-ligand evidence overrides it.

### The finding to lead with

**The under-dispersion that motivated the whole investigation is now largely closed.** DAVIS
within-ligand prediction spread, as a fraction of measured spread:

CNN 0.576 -> frozen ESM 0.622 -> k-mer 0.771 -> **hybrid 0.795**

On DAVIS the hybrid is the most committed arm in the project, above the lookup table itself,
and no drug sits at a predicted spread of zero in any run. Phase 3 set out to fix
under-commitment with domain adaptation and made it worse at every strength; supplying
similarity fixed it on DAVIS, and the fix is large.

**KIBA does not reproduce that, and the claim has to be scoped.** Its spread ratios are CNN
0.514, k-mer 0.401, frozen ESM **0.684**, hybrid 0.522. The hybrid improves on the CNN and on
the lookup table, but the frozen ESM branch is the most dispersed arm there by a wide margin, so
on KIBA the hybrid trades dispersion for accuracy and for ordering (within-ligand CI 0.6710, the
best KIBA figure of any arm). Two readings are available and this study cannot separate them:
either whole-chain ESM embeddings disperse KIBA predictions for a reason unrelated to being
right, since that arm's within-ligand CI is only 0.6447, or the k-mer offset anchors predictions
near each drug's observed range and caps how far they can spread. The second is the more likely,
because the offset cannot leave the range of the scores it averages. **"The dispersion deficit
is closed" is supportable on DAVIS and not on KIBA.**

Note also that projection width stops mattering once the offset is present: 96 and 1280 are
indistinguishable at 0.3413 against 0.3434 and 0.7460 against 0.7465. That is exactly what
section 7 predicts, since width buys fit and the offset now supplies the ordering.

## 9. What section 5's ranking should now say

| was | now |
| --- | --- |
| 1. Widen the ESM projection or pool per-residue | **Done for the width half, and it failed its bar (section 7).** Per-residue or attention pooling survives as the prime suspect, but it is no longer a 5-minute run. |
| 2. DANN on a branch that separates targets | **Done, and it separates the two readings.** `sc4001-phase6-dann-on-hybrid`, lambda 0.03, 3 seeds, adversary trained hard (domain accuracy about 0.84 against Phase 3's 0.66-0.75). Pooled MSE +0.0028 (t=+0.85, mixed signs: a null, not harm), pooled CI +0.0065 (t=+5.38, 3/3), and recomputed locally, **within-ligand CI +0.0084 (t=+1.75, 3/3)** with predicted spread -0.0079 (t=-4.32, 3/3). So DANN stops hurting once the representation carries the information; Phase 3's MSE-and-spread coupling (r=-0.951) is broken. See section 11. |
| 3. CDAN | Unchanged, and worth more now that a usable representation exists. |
| 4-5. IWDAN / IWCDAN, asymmetrically-relaxed alignment | Unchanged. Both still need a target label marginal this setup cannot estimate. |
| 6-7. Larger target pool, discriminator ablation | Future work. Not worth GPU before the above. |

## 10. Still unmeasured after today

* Whether the hybrid holds on the other three cold-target partitions, or is an artefact of
  s4001. The k-mer offsets for those splits are a CPU job; the runs are about 14 min of GPU.
* Whether per-residue or attention pooling moves within-ligand CI where extra width did not.
* What the network actually contributes on top of the offset. The margins say "real but
  modest"; nothing yet says *where* the correction helps, and that analysis needs no GPU.
* Whether the hybrid helps the Phase 4 mutant stress test. The k-mer offset for a mutant would
  draw on its own wild type, which may be either the fix or a leak, and that needs thinking
  through before it is run.

## 11. Phase 6: DANN on the hybrid, and what it settles

`sc4001-phase6-dann-on-hybrid`, COMPLETE. Lambda 0.03, three DAVIS seeds, all rc=0, 5.8 min.
Lambda 0.1 was also in the grid and failed on all three seeds: `src/train_dann.py` does not put
`lambda_max` in its run directory, so those runs landed on the lambda 0.03 directories and
`--resume` correctly refused. The lambda 0.03 result is unaffected. Logged in
`.claude/mistakes.md`.

**The adversary trained harder than in any Phase 3 run**: domain loss 0.348 / 0.346 / 0.361
against ln 2 = 0.6931, domain accuracy about 0.84, where Phase 3 reached 0.66-0.75. Frozen ESM
features separate the two protein pools more easily than CNN features do, so the validity gate
passes with room to spare on every seed.

| view | DANN | hybrid | delta | t(2) | signs |
| --- | --- | --- | --- | --- | --- |
| test MSE | 0.3441 | 0.3413 | +0.0028 | +0.85 | + - + |
| test CI | 0.8349 | 0.8283 | +0.0065 | +5.38 | + + + |
| uncensored MSE | 0.6898 | 0.6935 | -0.0037 | -0.46 | + - - |
| **within-ligand CI** | **0.7544** | 0.7460 | **+0.0084** | +1.75 | + + + |
| predicted spread ratio | 0.783 | 0.795 | -0.0079 (abs) | -4.32 | - - - |

The within-ligand row was not in the kernel. Pooled CI reads about 0.72 even for a
per-drug-constant predictor, so it cannot answer this question; both arms were therefore
recomputed from the pulled checkpoints on CPU with identical code, inference only, so the paired
difference cannot be an artefact of two code paths.

**Correction to the kernel's own output.** It printed "DANN still hurts", from a
`mse_delta > 0.002` rule written into the notebook. That verdict is wrong: +0.0028 at t=0.85
with mixed signs is a null, and the rule ignored the CI row entirely. The rule was too crude and
the measurements override it.

### What it settles

DANN is **no longer harmful** on a representation that orders unseen targets. Its effect is
small and points in different directions depending on the metric: ordering up, dispersion down,
accuracy flat. The substantive observation is that **Phase 3's coupling is broken.** There, MSE
damage and spread loss moved together across five arms at r = -0.951. Here spread falls slightly
while accuracy holds and ordering improves. Zhao et al.'s bound still explains the Phase 3
damage; what changed is the precondition, and the Phase 3 conclusion is therefore a statement
about alignment applied to features that lack the information, not about alignment as such.

### The both-parents claim, and why it is not a pass

Per seed on DAVIS, against k-mer alone (MSE 0.3516, within-ligand CI 0.7494):

| seed | test MSE | within-ligand CI | uncensored MSE |
| --- | --- | --- | --- |
| 4001 | 0.3474 | 0.7540 | 0.6893 |
| 4002 | 0.3419 | 0.7566 | 0.6821 |
| 4003 | 0.3431 | 0.7525 | 0.6981 |

All three seeds beat the lookup baseline on test MSE and on within-ligand CI, which is the
criterion section 8 set and missed by 0.0034. **It is still not reported as a pass.** That bar
was pre-registered for the plain hybrid, not for this arm; Phase 6's own pre-registered question
was the MSE delta against the hybrid, and that came out null. Applying section 8's criterion
here after seeing the numbers is post-hoc. Uncensored MSE is mixed against k-mer's 0.6875 in any
case, signs + - +. Treat it as a lead to confirm on the other three cold-target partitions.

## 12. Phase 7: the geometry check that killed the pooling hypothesis, and one locked null

Two reviews rejected the attention-pooling plan before it was built, on the grounds that its
motivating diagnostic had not ruled out a cheaper explanation and its primary endpoint was
invalid. Both objections held up. No GPU was spent on any of this.

### 12.1 The endpoint was invalid

The plan's primary measure was the within-drug concordance between the network's own output and
the residual truth, `score - offset`, with an assumed chance level of 0.5. Setting the network
output to exactly `-offset` gives residual-ordering CI **0.7644** over all 66 matched drugs, 66
of 66 above 0.5, and **0.6576** on uncensored pairs only - while the full prediction is constant
and so has within-ligand CI **0.5000**. The offset shrinks extremes, so the residual is
anti-correlated with it by construction, and 53.2% of within-drug test pairs are floor-floor,
where the residual difference reinstates an order the assay reported as a tie. The replacement
endpoint and its control are documented in the README.

### 12.2 Mean pooling is not convicted

Everything below is fitted on the 296 training targets only and scored on validation; test was
not loaded.

| protein space | val MSE | val pooled CI | val within-ligand CI |
| --- | --- | --- | --- |
| 3-mer cosine (reference) | 0.3088 | 0.8564 | 0.7956 |
| ESM raw cosine | 0.4992 | 0.7801 | 0.7266 |
| ESM centred (no hyperparameters) | 0.4184 | 0.8437 | 0.7838 |
| ESM whitened (rank 256, lam 1e-3) | 0.3524 | 0.8620 | 0.7959 |
| ESM supervised profile projection | 0.3720 | 0.8415 | 0.7719 |
| 3-mer supervised profile projection | 0.3243 | 0.8565 | **0.8100** |
| 3-mer, per-query rank weights | 0.3655 | 0.8516 | 0.7672 |
| ESM raw, per-query rank weights | 0.5732 | 0.8161 | 0.7473 |

Three readings, in order of how well they are supported:

1. **The ESM ordering deficit was largely a cosine artefact.** Centring alone, parameter-free,
   closes 83% of the gap; whitening closes it, and **42 of 42** whitening cells beat raw cosine.
   So section 7's suspicion that whole-chain pooling destroys the pocket signal is not supported,
   and the attention-pooling plan lost its motivation. Note the scope: this concerns cosine in
   ESM space, the quantity the swapped-space test measured. Whether the *trained* ESM arm was
   limited by the same anisotropy is untested, and a learned linear layer could absorb centring
   on its own, so it probably was not.
2. **The weighting rule is part of the raw-cosine deficit, but it is not where the fix lives.**
   Per-query rank weighting was the other candidate explanation and it is not a clean negative.
   It *raises* raw ESM ordering from 0.7266 to 0.7473, about 30% of that arm's gap, which is
   exactly what the adversarial review suspected: a similarity range compressed near 1.0 cannot
   be sharpened usefully, so its ordering reads better through ranks than through magnitudes. But
   it *hurts* 3-mer, whose similarities are already high-contrast, taking it from 0.7956 to
   0.7672, and it degrades MSE on both arms (ESM 0.4992 to 0.5732). Centring, with no
   hyperparameters at all, reaches 0.7838 and so beats rank weighting outright. The rule
   therefore contributes to how bad raw ESM cosine looks, while the repair belongs to the space.
3. **Supervision helps 3-mer and does nothing for ESM.** 8 of 42 cells beat the reference for
   3-mer, 0 of 42 for ESM. ESM needed an unsupervised variance fix; 3-mer needed supervision.
   With 296 targets and 1280 dimensions a supervised map on ESM simply overfits.

### 12.3 The one locked test run, and it is a null

The only arm that beat the reference on validation was the 3-mer supervised profile metric. Its
configuration was locked by validation MSE, deliberately not by the metric being reported - the
grid's best cell on within-ligand CI was rank 32 / penalty 1 at 0.8171 and was not used.

| arm | test MSE | pooled CI | within-ligand CI | uncensored MSE | spread ratio | floor-vs-above CI |
| --- | --- | --- | --- | --- | --- | --- |
| 3-mer cosine | 0.3516 | 0.8385 | 0.7494 | 0.6875 | 0.771 | 0.7548 |
| 3-mer supervised profile metric | 0.3689 | 0.8342 | 0.7515 | 0.8472 | 0.850 | 0.7744 |

Primary endpoint, the paired per-drug within-ligand CI delta over 66 drugs: **+0.0021**, 95%
bootstrap CI **[-0.0272, +0.0254]**. The interval includes zero, so by the rule fixed in advance
this is **not a positive result**. Validation said +0.0144; test says +0.0021. 41 drugs improved,
23 worsened, 2 unchanged, and the median (+0.0150) sits above the mean because one drug lands at
-0.6613. Accuracy also got worse, clearly so on the measurable pairs.

Two post-hoc observations, offered as leads and not as findings, because neither was
pre-registered: floor-versus-above-floor ordering improved by +0.0196 with 42 drugs better and 21
worse, and the prediction spread ratio rose from 0.771 to 0.850. Both were bought at a real cost
in uncensored MSE, so "it separates binders from non-binders better" is a hypothesis for another
partition, not a result here.

## 13. Where the cold-target deficit lives, and which endpoints cannot see it

Nothing here spent GPU. Both results are computed from checkpoints already on disk and both
replicate on validation and test. `analysis/homology_bands.py` and `analysis/endpoint_controls.py`
reproduce every number below.

### 13.1 The deficit is not uniform: it is a homology gradient

Each cold-target test protein is placed in a quartile of its **maximum amino-acid 3-mer cosine to
any training protein**, and every trained arm is re-scored inside each band. Metrics are computed
per seed and then averaged, and the k-mer tuning grid is imported from `src/kmer_baseline.py`;
doing either the other way round produces wrong numbers, which is logged as mistakes entry 37.
The check that the table can be trusted is that the pooled column reproduces the published
figures exactly: k-mer 0.7494, frozen ESM 0.6867, hybrid 0.7460.

**Test**, within-ligand CI:

| band | max cosine to train | proteins | censored | k-mer | CNN | ESM | hybrid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.1761-0.2454 | 16 | 0.767 | 0.5650 | **0.5069** | 0.5786 | 0.5507 |
| 1 | 0.2491-0.3261 | 16 | 0.681 | 0.6109 | 0.5679 | 0.6477 | 0.6224 |
| 2 | 0.3365-0.4420 | 15 | 0.655 | **0.8242** | 0.7746 | 0.7409 | 0.8125 |
| 3 | 0.4427-0.6955 | 16 | 0.703 | **0.8736** | 0.8074 | 0.8023 | 0.8478 |
| pooled | - | 63 | 0.702 | 0.7494 | 0.6818 | 0.6867 | 0.7460 |

**Validation**, same construction, its own quartiles:

| band | max cosine to train | k-mer | CNN | ESM | hybrid |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1540-0.3057 | 0.7540 | 0.6701 | 0.6731 | 0.7540 |
| 1 | 0.3104-0.3574 | 0.7112 | 0.6712 | 0.6999 | 0.7064 |
| 2 | 0.3577-0.4931 | 0.7803 | 0.7380 | 0.7319 | 0.7785 |
| 3 | 0.5050-0.8298 | 0.8491 | 0.7672 | 0.7678 | 0.8229 |
| pooled | - | 0.7956 | 0.7287 | 0.7404 | 0.7937 |

Two readings, both supported.

1. **A single cold-target figure averages two different regimes.** The k-mer baseline runs from
   **0.5650 to 0.8736** across test bands. Its headline 0.7494 therefore describes no actual
   case: it is a blend of near-chance performance on distant proteins and near-0.87 on close
   ones. Reporting one cold-target number hides the variable the task actually depends on. The
   censoring rate is flat across bands (0.655 to 0.767), so this is not an artefact of the pKd
   5.0 floor.
2. **In the hardest band the trained network cannot order unseen proteins at all.** Bootstrapping
   over band 0's 16 proteins, 2000 draws:

   | arm | band-0 within-ligand CI | 95% interval | clears chance |
   | --- | --- | --- | --- |
   | CNN | 0.5079 | [0.4620, 0.5510] | **no** |
   | hybrid | 0.5543 | [0.4896, 0.6200] | **no** |
   | k-mer | 0.5660 | [0.5008, 0.6357] | yes, marginally |
   | frozen ESM | 0.5786 | [0.5256, 0.6430] | yes |

   The CNN's interval straddles 0.5. On the most distant quarter of unseen proteins the learned
   encoder is indistinguishable from a coin. The hybrid does not rescue it there either, which
   bounds the Phase 5c claim: supplying similarity helps where similarity exists to supply.

### A claim in this area that was tested and withdrawn

A first version of this table showed frozen ESM worst pooled but best in both low-homology bands,
which would have reversed section 3 and justified routing by homology. It did not survive its own
gates. The margin shrinks to +0.0136 and +0.0368 once the two defects in mistakes entry 37 are
fixed; a bootstrap gives ESM minus k-mer **+0.0127, 95% [-0.0706, +0.1070]**, which includes
zero; and **validation reverses the sign**, at -0.0819. Section 3's conclusion stands unchanged
and no routing method is proposed. Recorded as mistakes entry 38.

### 13.2 Three standard endpoints that a degenerate predictor wins

A predictor that is **constant within each drug** carries no information about which unseen
protein binds which drug, by construction. Measured on the DAVIS cold-target test partition:

| control | pooled CI | full MSE | uncensored MSE | within-ligand CI |
| --- | --- | --- | --- | --- |
| per-drug mean of all its training pairs | **0.7633** | 0.5089 | 1.2570 | 0.5000 |
| per-drug mean of its uncensored training pairs | 0.6121 | 1.5795 | **0.8113** | 0.5000 |
| global training mean | 0.5000 | 0.6744 | 1.7913 | 0.5000 |
| *trained CNN, for comparison* | *0.7885* | *0.3991* | *1.1237* | *0.6789* |
| *k-mer baseline, for comparison* | *0.8385* | *0.3516* | *0.6875* | *0.7494* |

* **Pooled CI cannot be used here.** The first control reaches 0.7633 against the trained CNN's
  0.7885. The CNN's entire pooled margin over a predictor that knows nothing about proteins is
  **+0.0252**. Most test pairs compare different drugs, so pooled CI is dominated by drug-level
  ordering, which is not the question.
* **Uncensored MSE alone cannot be used either.** The second control scores **0.8113**, beating
  the trained CNN's 1.1237, while ordering nothing. It does not beat the k-mer baseline's 0.6875.
* **Within-ligand CI is sound.** Every control scores exactly **0.5000**, which is why it is this
  project's primary ordering endpoint.

This is the third endpoint failure found in the project, after the residual-ordering endpoint of
section 12.1 and the spread-ratio and uncensored-MSE endpoints proposed for a censoring-aware
loss. The pattern is consistent: on a drug-by-target grid with a censored floor, any endpoint
that does not hold the drug fixed can be won without learning anything about proteins.

### 13.3 Phase 8, pre-registered before the runs

The Phase 5c hybrid was measured on **one** partition. `kaggle/sc4001-phase8-davis-hybrid-confirm/`
runs the hybrid and its width-matched control on s4002, s4003 and s4004, three seeds each,
eighteen runs. Out-of-fold offsets for the three partitions are built and their leakage guard
holds in every case (out-of-fold MSE 0.3943 / 0.4385 / 0.3892 against test 0.4260 / 0.3815 /
0.5769, far from in-sample 0.0151 / 0.0220 / 0.0222).

These are harder partitions than s4001, which matters: a confirmation that only repeated an easy
case would prove little. k-mer reference figures per partition:

| partition | k-mer test MSE | k-mer uncensored MSE | k-mer within-ligand CI |
| --- | --- | --- | --- |
| s4001 | 0.3516 | 0.6875 | 0.7494 |
| s4002 | 0.4260 | 1.0551 | 0.8093 |
| s4003 | 0.3815 | 0.9065 | 0.7629 |
| s4004 | 0.5769 | 1.4281 | **0.6515** |

**The bar, fixed before the runs.** Primary: paired-by-seed hybrid minus control on within-ligand
CI, per partition; the claim "the hybrid beats a learned protein encoder" is confirmed only if
the sign is positive on all three new partitions. Secondary: paired MSE and uncensored MSE
against the same control. Reported but **not** a pass condition: hybrid minus k-mer, which Phase
5c tied. The **four partitions** are the replication units; nine seeds inside one partition are
not nine independent tests of the method.
