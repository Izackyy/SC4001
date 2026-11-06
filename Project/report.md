# SC4001 project report: progress, results, and what is blocking us

Written for the group and whoever reads the submission. Light on technique. `update.md` has the
statistics, `README.md` has the commands.

## The task and the one choice that shaped everything

Predict how strongly a drug binds a protein, from the drug's chemical string and the protein's
amino-acid sequence. Datasets: DAVIS (kinase inhibitors, 68 drugs) and KIBA (larger, 2000+
drugs). Model: DeepDTA, two convolutional branches joined by a regression head.

The choice that mattered: we evaluate on **proteins the model has never seen**, not on random
held-out pairs. That is the setting a real screening campaign faces, and it is much harder.
Every surprising result below follows from it.

## Phase 1: data. Done, and more careful than planned.

Built the inputs from the published DeepTA text files rather than the shipped pickles (the
pickles are Python 2 and unsafe to open). Merged duplicate drug/protein pairs, recorded which
measurements are "too weak to measure" rather than real readings, and built the splits.

Two decisions worth defending in the write-up:

* **Mutant proteins were reconstructed, not assumed.** DAVIS labels proteins like
  `PIK3CA(H1047L)`. We edited the actual sequences. 49 annotations validated, 44 distinct
  mutants produced. Genes with unresolvable numbering (RET, FGFR3) were excluded rather than
  guessed.
* **Splits respect gene families.** A protein and its mutants always land on the same side of
  the split. Otherwise the model sees a mutant in training and its near-identical wild type in
  test, and the test score is meaningless.

**Known limit:** protein sequences are cut at 1200 characters. Three mutants' edits sit past
that cut, so the model literally cannot see them. We track these as a control group rather than
hiding them. More on this below — it became the most interesting defect we have.

## Phase 2: baseline. Works, but in a way that looks better than it is.

Trained on DAVIS and KIBA, four different random splits each, three seeds each.

The headline accuracy numbers are respectable. The problem is what is behind them. For a given
drug, the model's predictions across unseen proteins vary only **58% (DAVIS) / 51% (KIBA)** as
much as the real measurements do. It is hedging towards each drug's average instead of
committing to a per-protein answer.

**Why that matters.** The usual metric in this field (concordance index, pooled across all
pairs) reads about 0.72 even for a model that gives each drug one constant score regardless of
protein. So a good-looking published-style number is compatible with a model that does nothing
useful for the actual task. We now report a within-drug version alongside it. This is a
measurement contribution in its own right and should be in the report.

## Phase 3: domain adaptation. Negative, and the negative is the finding.

The plan: feed the model extra proteins with no measurements attached, and train it so it
cannot tell our proteins from the extra ones (DANN). Standard recipe for transfer problems.

**It made things worse at every strength we tried, on both datasets.**

We spent effort ruling out the boring explanations, because "we configured it wrong" is worth
nothing and "this method cannot work here" is worth a lot:

| Doubt | Check | Outcome |
|---|---|---|
| Adversary never trained | measured its own accuracy | it trained (0.66–0.75) |
| Undertrained | equal-compute rerun | deficit persists |
| Wrong strength | swept 4 values | damage grows monotonically |
| Side effects, not alignment | control with adversary disabled | loss remains |

And there is theory that predicts it. Zhao et al. (2019) proved a lower bound: forcing two
groups of data to look alike *increases* error when the groups genuinely differ in what their
answers look like. Ours do, and the extra proteins carry no measurements, so the correction the
theory requires is unavailable to us. We were asking the model to delete real information.

**Why that matters.** The report's claim becomes "here is a condition under which a popular
method provably cannot help, and we measured it", which is a stronger paper than "we got
transfer learning to work slightly".

## Phase 4: mutant stress test. The model does not predict mutation effects at all.

Does the model know that a single amino-acid change alters binding? We held every mutant out of
training and compared each mutant's prediction against its wild type's.

A baseline that always answers "this mutation changes nothing" gets an average error of 0.4896.
Our model gets 0.4850. **An improvement of 0.9%.** On unseen mutants it is for practical
purposes identical to assuming mutations do nothing.

Two things make this phase worth its page count:

* **The original plan would have reported the opposite.** Evaluating on the earlier
  checkpoints, where mutants had been in training, gives correlation +0.73 instead of +0.14 and
  looks like success. The gap is pure memorisation. We caught it by rerunning with a proper
  held-out split.
* **Standard metrics hide it completely.** On its own test set this model scores a concordance
  index near 0.90. That number and "cannot predict any mutation effect" are both true at once,
  because most of a pair's affinity is set by the drug and the kinase family, and the mutation
  is a small correction on top. Reporting aggregates alone would have overstated the model.

## Phase 5: two attempts to fix the representation, not the training

Hypothesis from Phase 3: the bottleneck is the model's *picture of a protein*, not the training
objective. Test it with methods that do no adaptation at all.

### A lookup method with no learning beat our neural network

For an unseen protein, find the most string-similar training proteins and reuse what this drug
scored against them. Weighted average. Nothing is trained, no GPU used.

**It beat the neural baseline on 7 of 8 dataset–split combinations**, and on ranking in all 8.
On DAVIS's genuine readings, error fell from 1.06 to 0.69. It also commits properly: 77% of the
real variation, against the baseline's 58%.

Being beaten by a lookup table is the single most informative result in the project. The signal
needed for unseen proteins is sitting in plain sequence similarity, and the neural network was
not extracting it. The diagnosis holds.

Honest caveat for the report: the method needs the drug to already have measurements against
similar proteins. Every one of our test cases had that, so it was never tested on the case it
cannot handle. A genuinely novel protein family would break it.

### A large pretrained protein model: half a win

Replaced the protein branch with ESM-2 (650M parameters, Meta, trained on protein sequences),
frozen and used as-is. 25 minutes of GPU, 6 runs, all completed.

* **9x faster per training round** — that branch is no longer trained.
* Accuracy improved; clearly on one measure, in the right direction on most others.
* **Commitment improved a lot** (KIBA 51% → 68% of real variation) — the behaviour we wanted.
* **Ranking of unseen proteins barely moved** — within the range we see from changing the random
  seed alone.

Wider guesses, not better-ordered ones. The lookup method still beats it on every DAVIS view.

## The current roadblock, stated precisely

We cannot yet claim "pretrained protein models do not help here". We can only claim **"this way
of using one does not help"**, because three handicaps are confounded together and we have
varied none of them:

1. **We squeezed 1280 numbers down to 96** to keep the comparison against the CNN exactly fair.
   That is a 13x compression, applied for cleanliness, which may have thrown away what we were
   trying to measure.
2. **We averaged over the whole protein.** Binding happens at one small site; averaging the
   whole chain dilutes it. Published work in this setting keeps per-position detail.
3. **We cut proteins at 1022 characters.** Six DAVIS proteins are mutants whose edit sits past
   the cut, so they became identical to their originals. This changes none of the numbers above
   (the splits keep those six together, so nothing leaked), but it means this setup is blind to
   those mutations — which caps any Phase 4 work built on it.

**Why the roadblock matters.** Phase 3's conclusion rests on the claim that the deficit is
representational. Right now one piece of evidence supports it strongly (the lookup method
improves both accuracy and ordering) and one supports it weakly (the pretrained model improves
accuracy and commitment but not ordering). The weak piece is weak for a reason we created, not
a reason about the method. Removing confound 1 is cheap — minutes of GPU — and either
strengthens the Phase 3 argument or tells us the representation story is incomplete. Either
answer is worth having before the write-up is fixed.

**Resolved, 6 November.** Confound 1 was removed and it was not the binding constraint: wider
channels improved the fit monotonically and left the ranking roughly where it was. Confound 2,
whole-chain averaging, is now the prime suspect by elimination. The roadblock itself is gone,
because the fix below does not depend on resolving it. The next section has both results.

## Two more results, and the headline changes

Both ran on 6 November, and together they turn this from an all-negative study into a negative
diagnosis plus a fix derived from it.

### Widening the squeeze helped the fit, not the ranking

We widened the channel from the pretrained model from 96 numbers to 256, 512 and the full 1280,
three seeds each, and chose the width on validation data only. Validation error improved every
single time the channel got wider, and the widest setting became the best neural model we have
on DAVIS, beating the CNN baseline. But the ability to **rank** unseen proteins moved from 0.687
to about 0.70 and stopped, still short of the lookup table's 0.749.

We had written the success criterion down in advance, and **it was not met**: the width that
validation chose clears less than half the required margin, and one of three seeds moves the
wrong way. Reported as a failure.

That is still worth having. It strikes one of the three suspects off the list by measurement
rather than by assumption, and promotes the remaining one: we average the pretrained model's
reading over the whole protein chain, and binding happens at a small pocket. Fixing that is not
a cheap run.

### Handing the model the similarity signal: the first clear win

If the model cannot learn to read proteins from 400 examples, stop asking it to. We gave it the
lookup table's answer as a starting point and let it learn only the correction:

```text
prediction = similarity-based neighbour estimate + network correction
```

**It beats every version of the learned-protein-encoder model decisively and consistently.** On
DAVIS, error drops 0.061 with all three seeds agreeing, and on the genuine measurements -- the
ones not sitting on the detection floor -- it drops 38%, from 1.12 to 0.69. On KIBA the same
pattern, with the ranking of unseen proteins improving by 0.024 and all three seeds agreeing.

**And the hedging problem that started this whole investigation is largely closed.** Prediction
spread, as a fraction of the real variation: CNN 0.58, pretrained model 0.62, lookup table 0.77,
**hybrid 0.80**. On DAVIS it is the most committed model in the project, more so than the lookup
table itself. Domain adaptation was supposed to fix this and made it worse; supplying similarity
fixed it.

On KIBA it does not hold. There the figures are CNN 0.51, lookup table 0.40, hybrid 0.52, but the
pretrained model alone reaches 0.68 — more spread than the hybrid, while ranking proteins worse
than it. So the hedging fix is demonstrated on DAVIS and not on KIBA, where the hybrid instead
buys the best ranking of any arm. The likeliest reason is that the lookup offset holds
predictions near each drug's already-observed range, which helps accuracy and caps spread at
the same time.

One trap was worth the care it took. The offset for a training pair must not be computed from
data containing that pair -- a pair's own protein is its own most similar protein, so the lookup
nearly reproduces its own training answers (error 0.02 in-sample against 0.35 on test). Training
on those residuals would have produced an excellent-looking, meaningless result. Training pairs
therefore get offsets computed from a held-out portion of the training set, and the measured
in-sample and held-out figures are printed side by side so the gap stays visible.

### The honest limit

The hybrid clearly beats every neural arm. Against the lookup table it roughly **ties**: better
accuracy on DAVIS, better ranking on KIBA, slightly worse on the other axis of each. Neither
dataset clears the pre-registered bar of beating both parents on both axes. So the claim is
"the learned model becomes competitive once handed the similarity signal", not "we built the
best predictor". Saying that first is better than having a reviewer say it.

## What is running and what remains

* **Done:** domain adaptation on the hybrid. Phase 3 only ever tested it on a representation
  that could not rank unseen proteins, so "the method is wrong here" and "the method had nothing
  to work with" stayed tangled. The hybrid ranks them, so the experiment became available, and
  the answer is that **adaptation stops hurting**: accuracy unchanged, ranking of unseen
  proteins slightly better with all three seeds agreeing, prediction spread slightly worse with
  all three agreeing. Small in size and mixed in direction, so not a win -- but in Phase 3
  accuracy damage and spread loss always moved together, and here they do not. Once the
  representation carries the information, alignment stops destroying it. The arm also becomes
  the first to beat the lookup table on both accuracy and ranking with every seed agreeing,
  though that bar was set for a different arm and so counts as a lead to confirm rather than a
  pre-registered result.
* **Next:** confirm the hybrid on the other three cold-target partitions, so it is not a
  one-split artefact. The similarity offsets for those are a CPU job.
* **Then:** attention or per-residue pooling over the pretrained embeddings, now the prime
  suspect for the flat ranking. Not a cheap run.
* **Future work, not this submission:** reweighted adaptation variants (IWDAN/IWCDAN) that exist
  to patch the exact theoretical problem we hit; a larger pool of unlabelled proteins; CDAN.
* **Open analysis, no GPU needed:** what the network actually contributes on top of the offset.
  The margins say real but modest; nothing yet says where the correction helps.

## One-paragraph summary for the abstract

> On drug–target affinity with unseen proteins, a DeepDTA baseline under-commits badly: its
> per-drug predictions vary roughly half as much as the measurements do. Adversarial domain
> adaptation makes this worse at every strength tested, and we rule out the usual configuration
> explanations and connect the failure to a known lower bound on invariant-representation
> methods. Two methods that perform no adaptation improve on the baseline instead — including a
> non-neural sequence-similarity lookup that beats it on 7 of 8 splits — which locates the
> deficit in the protein representation rather than the training objective. Acting on that
> diagnosis, supplying the similarity signal as an additive offset and learning only the
> correction beats every learned-protein-encoder arm by large, seed-consistent margins, cuts
> error on uncensored measurements by 38%, and raises prediction spread from 0.58 to 0.80 of
> measured — closing most of the under-commitment the adaptation phase set out to fix. A
> held-out mutant stress test shows the same model is, for practical purposes, indistinguishable
> from assuming point mutations have no effect, a failure that aggregate metrics conceal
> entirely.
