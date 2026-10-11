# SC4001 Part A: Written Answers and Evaluation

This draft covers the written explanations and evaluations for the completed A1, A2, A3, and A4 experiments. Values are taken from the saved run outputs under `outputs/part_a/`. The Part B discussion should be added after those experiments have been run.

## A1 - Baseline search and final evaluation

The baseline embeds the four categorical inputs (`month`, `town`, `flat_model_type`, and `storey_range`) and concatenates their embeddings with six standardized continuous inputs. A hidden linear layer, LayerNorm, and ReLU produce the representation used to predict resale price. The search compared 25 combinations of hidden width and shared embedding width, selecting each configuration by its best validation RMSE.

The selected configuration used hidden width 256 and embedding width 12, with its best validation result at epoch 5. Its validation RMSE was **$60,522.35**. The next-best configuration used width 256 and embedding width 8, with validation RMSE **$60,851.21**. The selected model has **16,073 trainable parameters**, compared with **3,505** for the supplied default architecture. The larger model therefore improved validation fit in this search, at the cost of more parameters; this result does not establish that larger models will always generalize better.

After refitting the selected configuration on the combined 2017-2021 rows for five epochs, the 2022 test RMSE was **$101,288.84** and the test R² was **0.6461**. An R² of 0.6461 means the predictions account for about 64.6% of the variation in 2022 resale prices relative to predicting the test-set mean. The train/test RMSE curve is:

| Epoch | Train RMSE | Test RMSE |
|---:|---:|---:|
| 1 | $57,690.69 | $117,196.41 |
| 2 | $43,249.86 | $104,091.42 |
| 3 | $41,975.13 | $110,457.18 |
| 4 | $41,218.62 | $99,198.28 |
| 5 | $40,528.26 | $101,288.84 |

Training RMSE falls each epoch, while test RMSE fluctuates and remains substantially higher. This gap is consistent with limited generalization and may also reflect differences between the 2017-2021 fitting data and the 2022 test year. The test curve is descriptive only: the five-epoch refit length was fixed from A1 model selection, and the test values were not used to select an epoch or configuration.

## A2 - Five-seed ablations

The A2 runs compare the selected baseline with two controlled variants across seeds 11, 42, 73, 101, and 2026. All variants used the selected A1 hidden width, embedding width, and five-epoch refit length. Removing embeddings reduced the parameter count from 16,073 to 2,561. Replacing ReLU with Sigmoid kept the baseline parameter count unchanged.

| Model | Test RMSE mean (sample SD) | Test R² mean (sample SD) |
|---|---:|---:|
| Baseline | $99,827.03 ($7,145.80) | 0.6548 (0.0488) |
| No embeddings | $106,804.28 ($8,607.04) | 0.6045 (0.0635) |
| Sigmoid hidden activation | $104,599.10 ($9,007.13) | 0.6203 (0.0639) |

The baseline has the lowest mean RMSE and highest mean R². Removing the categorical embeddings raises mean RMSE by about **$6,977** and lowers mean R² by about **0.0504**. This supports the value of learning representations for the categorical inputs in this model and dataset. The Sigmoid variant raises mean RMSE by about **$4,772** and lowers mean R² by about **0.0345**, so ReLU performed better on average under the tested setup.

These are average effects, not wins on every seed. For example, the Sigmoid model has lower test RMSE than the baseline for seed 42, but its five-seed average is worse. The sample standard deviations also show meaningful run-to-run variation. The experiment therefore supports the baseline choices on average, while the five seeds are not enough to establish that the ranking would hold under every training setup.

## A3 - Wide-and-deep model

The A3 model adds a linear branch over the same concatenated inputs as the MLP. Its prediction is the MLP output plus `lambda` times the linear-branch output. Lambda was selected from `{0, 0.25, 0.5, 1, 2}` using validation RMSE while keeping the A1-selected widths and other settings fixed.

The selected value was **lambda = 1.0**, with validation RMSE **$60,681.82** at epoch 5. Lambda 0.0 achieved **$60,698.74**, only **$16.93** higher. Thus lambda 1.0 was the best candidate in this search, but its validation advantage over the lambda 0.0 candidate (whose wide-branch contribution is zero) was very small. The selected A3 model has 16,128 trainable parameters, 55 more than the A1 model.

| Comparison | Test RMSE mean (sample SD) | Test R² mean (sample SD) |
|---|---:|---:|
| A1 baseline, seed 42 | $101,288.84 (one run) | 0.6461 (one run) |
| A2 baseline, five seeds | $99,827.03 ($7,145.80) | 0.6548 (0.0488) |
| A3 wide-and-deep, five seeds | $100,724.83 ($4,859.81) | 0.6494 (0.0334) |

For a like-for-like comparison, A3 should be compared with the five-seed A2 baseline. A3's mean RMSE is about **$898 higher** and its mean R² is about **0.00545 lower**, so the wide branch did not improve mean test performance in these runs. A3 has lower seed-to-seed standard deviation, but that alone is not evidence of better predictions. In the paired seed results, A3 has lower RMSE for three of the five seeds and higher RMSE for two. Overall, the observed test results do not support claiming a consistent benefit from the added linear branch, despite lambda 1.0 being the best validation candidate.

The evaluation uses one held-out year and five random seeds. The validation search and test comparison answer different questions: validation RMSE selected lambda, while the test set estimates final performance. The small validation difference between lambda 1.0 and 0.0 and the variation across seeds counsel against treating the selected lambda as a large or universal improvement.

## A4 - FeatureAblation

FeatureAblation was applied to all **2022 test rows** for the final A1 baseline and A3 wide-and-deep models using the same seed (**42**) and the same combined 2017-2021 preprocessing. The attribution target was predicted resale price in Singapore dollars. For each categorical feature, the baseline was category ID 0, which represents an unseen category. For each standardized continuous feature, the baseline was 0, the mean value under the combined training and validation preprocessing. Each score below is the mean absolute change in the model's predicted price after replacing that feature with its baseline.

The A1 baseline's three largest scores were **floor area** ($60,191.95), **town** ($46,081.88), and **remaining lease years** ($43,646.48). The A3 wide-and-deep model's three largest were **flat model type** ($68,312.56), **floor area** ($58,483.29), and **town** ($51,946.66). Floor area and town rank highly for both models, while the A3 model assigns a larger masking impact to flat model type than the A1 baseline does. The complete ten-feature rankings are shown in the saved comparison plot and score table.

One limitation is that these scores depend on the selected masking baselines and measure model response, not causal effects. Replacing one feature independently can also create combinations that are uncommon in the data, especially when inputs are correlated.

## Relevant lecture context

- **Lecture 5, Model Selection**, slides 21-23, 31-35, and 45-49: keep training, validation, and test roles separate; use validation for model and checkpoint selection; use the held-out test set for final evaluation; interpret gaps between fitting and held-out performance cautiously.
- **Lecture 4, DNN**, slides 6-9, 35, and 40-42: input normalization, network width, and model capacity help frame the A1 search and parameter-count tradeoff.
- **Lecture 2, Regression and Classification**, slides 16-19: regression outputs and error metrics provide the context for interpreting RMSE and R².
- FeatureAblation is not covered directly in the provided lecture slides; its feature scores are interpreted as baseline-relative changes in model predictions.

## Reproducibility references

- A1 metrics and train/test curve: `outputs/part_a/a1/metrics.json`, `outputs/part_a/a1/train_test_curve.csv`
- A2 seed results and summary: `outputs/part_a/a2/seed_metrics.csv`, `outputs/part_a/a2/summary.csv`
- A3 lambda selection and comparison: `outputs/part_a/a3/lambda_search_results.csv`, `outputs/part_a/a3/comparison_summary.csv`, `outputs/part_a/a3/paired_seed_comparison.csv`
- A4 feature scores and masking baselines: `outputs/part_a/a4/feature_ablation_scores.csv`, `outputs/part_a/a4/metadata.json`
- A4 plot: `outputs/part_a/a4/feature_ablation_comparison.png`
- Full code and saved plots: [`SC4001_PartA_Results.ipynb`](SC4001_PartA_Results.ipynb)
