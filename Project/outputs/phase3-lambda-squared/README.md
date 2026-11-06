# Phase 3 runs under the superseded lambda-squared loss

Three DAVIS `split_cold_target` runs (seeds 4001-4003) made on a Tesla T4 on
2026-11-05 with `total = regression_loss + coefficient * bce` and the gradient
reversal coefficient also set to lambda, so the encoder's adversarial gradient
scaled as lambda^2 and the discriminator's own gradient as lambda.

Result: test MSE 0.491 (sd 0.004) against the baseline's 0.383 (sd 0.010), and
CI 0.773 against 0.792. Best epoch was 5 or 6 against the baseline's 51 to 93.
`history.csv` shows why: `domain_loss` stayed at 0.690, about ln 2, and
`domain_accuracy` at chance for all 20 epochs, so the discriminator never
learned to separate the domains and the encoder received lambda-scaled noise.

Kept as the evidence for that diagnosis. `src/train_dann.py` now applies lambda
once, through the reversal coefficient only, as in Ganin et al. Do not mix these
numbers with runs made after that change.
