# Where the project stands, in plain English

Companion to `update.md`, which has the same content with the statistics in it. Nothing here
is new work; it is the same results said differently.

**Written before 6 November, and two things have changed since.** First, we widened the squeeze
on the pretrained protein model, up to no squeeze at all. It improved the accuracy every time
and left the *ranking* of unseen proteins roughly where it was, so the squeeze was a handicap
but not the main one. Second, and this is the bigger one: we stopped asking the network to read
proteins and simply handed it the lookup table's answer to correct. **That beats every version
of the neural model clearly and consistently, cuts the error on genuine measurements by 38%, and
closes most of the hedging problem on DAVIS** -- prediction spread rises from 0.58 to 0.80 of
the real variation, more committed than the lookup table itself. On KIBA that part does not
hold: the hybrid reaches 0.52 against the pretrained model's 0.68, so there it buys the best
ranking instead of the most commitment. Against the lookup table it roughly
ties rather than winning outright. `report.md` has both in plain English, and `update.md`
sections 7 and 8 have the numbers.

## The problem we are actually solving

We predict how strongly a drug sticks to a protein. The hard version, and the one we chose, is
**proteins the model has never seen**. The drug side it knows well. The protein side it has to
guess from the amino-acid sequence alone.

Two ways to be wrong, and they are different:

* **Wrong on average** — every prediction is off by a similar amount. Annoying but usable.
* **Flat** — the model gives a drug nearly the same score against every new protein. The
  average looks fine, but the model is useless, because the whole point is to tell new proteins
  apart.

Our baseline model was mostly doing the second thing. For a given drug, its predictions across
new proteins varied only about **58% as much as the real measurements do** on DAVIS and **51%**
on KIBA. It was hedging towards the average instead of committing.

## What we tried before, and why it failed

Phase 3 was a domain-adaptation method (DANN). The idea: show the model extra proteins with no
measurements attached and train it so it cannot tell "our proteins" from "the extra ones". The
hope is it learns features that transfer.

It made things worse, consistently, at every strength we tried.

The important part is **why**, because "we set it up wrong" is a weak result and "this method
cannot work here" is a strong one. We ruled out the setup explanations:

* The adversary part really did train (it was not a dead component).
* Compute was equal, so it was not undertrained.
* We swept the strength dial across four values; the damage got worse as we turned it up.
* We ran a control where the adversary was deliberately disabled, and the loss remained.

And there is published theory that predicts this. Zhao et al. (2019) proved that forcing two
groups of data to *look* alike makes error worse when the two groups genuinely differ in what
their answers look like. Our two groups do differ, and the extra proteins have no measurements
at all, so there is no way to correct for it. The method was asking the model to erase real
information.

## This session: two attempts that skip adaptation entirely

Both target the same suspicion — the problem is that the model's *picture of a protein* is too
poor, not that the training procedure is misaligned. If that is right, then methods that improve
the picture should help, and no amount of alignment will.

### Attempt 1: a dumb lookup method. It won.

No neural network at all. For a new protein, find the proteins in the training data that look
most similar as strings of amino acids, and reuse what this drug scored against those. A
weighted average of neighbours. It learns nothing.

**It beat our neural baseline on 7 of the 8 dataset–split combinations**, and on ranking quality
in all 8. On the DAVIS measurements that are genuine readings rather than "too weak to measure",
average error dropped from 1.06 to 0.69. It also commits far more: 77% of the real variation
instead of 58%.

Being beaten by a lookup table sounds embarrassing. It is the most informative result we have.
It means the signal needed for new proteins is sitting in plain sequence similarity, and the
neural network was not extracting it. That is a representation failure, exactly the diagnosis.

Caveat worth stating in the report: the method only works for drugs that already have
measurements against similar proteins. In our splits every test case had one, so it was never
tested on the case it cannot do. A real screening campaign would hit that case.

### Attempt 2: a big pretrained protein model. Partly worked.

We swapped our protein component for ESM-2, a 650-million-parameter model trained on protein
sequences by Meta, frozen (we use it as-is, no training). Six runs, all finished, 25 minutes of
GPU time.

Mixed outcome:

* Faster: about **9x fewer seconds per training round**, because we no longer train that part.
* Better on some accuracy measures, clearly so on one, and in the right direction on the rest.
* It **commits much more** than the baseline, which is the behaviour we wanted.
* But its ability to *rank* new proteins correctly barely moved — within the range we see just
  from changing the random seed.

So it widened its guesses without ordering them better. Wider but not wiser. The lookup method
beats it on every DAVIS measure.

Three reasons it may have been handicapped, and we varied none of them, so we cannot say
"pretrained protein models don't help here" — only "this way of using one didn't":

1. We squeezed its 1280-number output down to 96 numbers to keep the comparison fair.
2. We averaged over the whole protein. Drug binding happens at one small site; averaging the
   whole chain dilutes it.
3. We cut proteins off at 1022 amino acids. Six DAVIS proteins are mutants whose mutation sits
   past that cut, so those six became identical to their originals. This changes none of the
   numbers above (the splits keep those six together), but it does mean this setup is blind to
   those mutations, which limits the mutant experiment in Phase 4.

## What to say in the write-up

The honest story, in one paragraph:

> The cold-protein deficit is in how the model represents proteins, not in how it is trained.
> Two methods that do no domain adaptation at all improve on the baseline, one of them a method
> that learns nothing. Domain adaptation cannot add information a representation never had, and
> there is a published lower bound saying that under our conditions it must make things worse.
> Our negative result is a statement about when the method applies, which is more useful than a
> statement about whether we tuned it correctly.

The two attempts do not support that equally, and we should not pretend otherwise. The lookup
method improves both accuracy and ordering. The pretrained model improves accuracy and
commitment but not ordering, so it is evidence that the representation was the bottleneck, not
evidence that a frozen pretrained protein model is the cure.

## If we spend more GPU time

In order of what we learn per minute:

1. **Un-handicap the pretrained model** — widen the squeeze, or attend to parts of the protein
   instead of averaging. About 5 minutes of GPU. Cheapest run in the project.
2. **Re-run domain adaptation on a protein representation that does rank new proteins.** This is
   the run that separates "the method is wrong here" from "the method had nothing to work with".
3. **Try CDAN**, the variant the literature endorses for this task, then the reweighted variants
   (IWDAN/IWCDAN) that exist specifically to patch the theoretical problem we hit.

Two things for whoever owns the account: check the real remaining weekly GPU allowance on
kaggle.com rather than trusting our arithmetic, and rotate the API token that leaked into a
session log.
