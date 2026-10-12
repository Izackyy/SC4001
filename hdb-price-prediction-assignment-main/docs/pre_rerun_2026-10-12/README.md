# Pre-rerun snapshot (2026-10-12)

This folder preserves the existing generated results before cleanup and rerunning. The archived outputs are in outputs.zip with their original outputs/ paths.

## Existing results

- Part A1 selected hidden width 256 and embedding width 12 at epoch 5; test RMSE was $101,288.84 and R2 was 0.6461.
- Part A2 five-seed mean test RMSE: baseline $99,827.03, no embeddings $106,804.28, sigmoid $104,599.10. Mean R2: 0.6548, 0.6045, and 0.6203 respectively.
- Part A3 selected wide-branch lambda 1.0; five-seed mean test RMSE was $100,724.83 and mean R2 was 0.6494.
- Part B test accuracy: B1 83.33%, B2 89.58%, B3 87.98%. Sensitivity was 99.49%, 99.49%, and 98.97%; specificity was 56.41%, 73.08%, and 69.66%.

## Current script edits before the rerun

- Removed the module-level docstring from src/part_a.py, src/part_a3.py, src/part_a4.py, and src/part_b.py.
- Updated src/part_b.py so the validation search plotting helper also saves separate validation-loss and validation-accuracy figures.

## Written-answer files

The written-answer files were not edited for this snapshot. At snapshot time, Git showed both files as deleted in the working tree; their committed versions remain in the repository history. They are intentionally excluded from the pre-rerun commit.
