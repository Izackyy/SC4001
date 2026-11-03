# HDB price baseline

## Run it

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run the baseline with:

```bash
uv run python src/baseline.py
```

Run the A1 Ray Tune search, selected-model refit, and A2 five-seed ablations with:

```bash
uv run python src/part_a.py
```

After A1/A2 finish, run the A3 wide-and-deep lambda search and five-seed refits:

```bash
uv run python src/part_a3.py
```

After A3 finishes, run the A4 test-set feature-ablation analysis:

```bash
uv run python src/part_a4.py
```

The Part B entry point runs B1 through B4 sequentially by default. To execute
one stage at a time, use `--part B1`, `--part B2`, `--part B3`, or `--part B4`;
later stages use the selected checkpoints and settings saved by earlier stages.
The supplied `chestxray_binary_64.npz` file should remain one directory above
this repository root.

```bash
uv run python src/part_b.py
```

Open `SC4001_PartA_Results.ipynb` to browse mirrored source cells and view the
saved A1-A4 tables and plots inline. See `SC4001_PartA_Written_Answers.md` for
the written A1-A4 interpretation. Training cells are disabled by default.
`SC4001_PartB_Results.ipynb` contains the complete Part B source code and
inline result summaries; `SC4001_PartB_Written_Answers.md` contains the B1-B4
interpretation. Part B training is disabled in the notebook by default.

Run commands from the repository root. The input CSV and output paths are defined in the relevant entry point (`src/baseline.py`, `src/part_a.py`, `src/part_a3.py`, or `src/part_a4.py`); data splits, features, and training hyperparameters are defined in `src/config.py`.

The A1, A2, A3, and A4 outputs are written under `outputs/part_a/`: Ray trial results
and the selected-model curves/metrics are in `a1/`, while per-seed metrics,
mean/sample-standard-deviation summaries, and comparison plots are in `a2/` and
`a3/`. A4 feature scores, masking-baseline metadata, and the comparison plot are
in `a4/`.

## Code map

```text
src/config.py       run configuration
src/data.py         split, fit preprocessing, tensor conversion
src/model.py        categorical embeddings, MLP, and the A3 linear branch
src/baseline.py     single-run training, RMSE evaluation, and model export
src/part_a.py       A1 Ray search/refit and A2 ablation experiments
src/part_a3.py      A3 wide-and-deep lambda search and five-seed comparison
src/part_a4.py      A4 Captum FeatureAblation and feature-importance plots
src/part_b.py       B1-B4 CNN training, comparison, and Grad-CAM
```
