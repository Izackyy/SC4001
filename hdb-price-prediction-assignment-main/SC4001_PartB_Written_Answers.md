# SC4001 Part B: Written Summaries and Evaluation

This report records the validation-selected settings and held-out results for B1-B4. The supplied validation partition guided model selection; the test partition was reserved for final evaluation and the shared B4 visual examples.

## Shared data preparation

The supplied archive contains 4,708 training, 524 validation, and 624 test images, each stored as a grayscale 64x64 array. The training partition contains 1,214 normal and 3,494 pneumonia images; validation contains 135 normal and 389 pneumonia images; test contains 234 normal and 390 pneumonia images. Label 1 is treated as pneumonia, the positive class. The test class counts match the published source dataset split. The source dataset comprises normal and pneumonia X-rays, as described by [Kermany et al.](https://www.sciencedirect.com/science/article/pii/S0092867418301545).

Pixel intensities were divided by 255. The training partition alone was used to calculate the scalar pixel mean (**0.5717**) and standard deviation (**0.1741**); those same values were applied to validation and test images. The models use one output logit with `BCEWithLogitsLoss`; a sigmoid probability threshold of 0.5 defines class predictions. Sensitivity treats pneumonia as positive.

## B1 - AlexNet-like model and learning-rate search

B1 used batch size 128 and searched learning rates `{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`. Each trial used validation BCE loss for early stopping and checkpoint selection; the winning learning rate was chosen by the lowest validation classification error.

The selected learning rate was **0.001**, with its best-loss checkpoint at **epoch 6**. It achieved validation loss **0.0652**, validation error **2.48%**, validation sensitivity **97.94%**, and validation specificity **96.30%**.

| Learning rate | Best epoch | Validation loss | Validation error | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 6 | 0.0652 | 2.48% | 97.94% | 96.30% |
| 0.0001 | 20 | 0.0762 | 3.24% | 98.46% | 91.85% |
| 0.01 | 5 | 0.0792 | 3.24% | 98.20% | 92.59% |
| 0.1 | 8 | 0.1705 | 8.59% | 89.72% | 96.30% |
| 0.00001 | 20 | 0.2652 | 9.73% | 96.14% | 73.33% |

On the reserved test set, B1 achieved **83.33% accuracy** and **16.67% error**, with **99.49% sensitivity** and **56.41% specificity**. Its confusion counts were TP=388, FN=2, TN=132, and FP=102. The model caught nearly all pneumonia cases, but at the fixed threshold it also labeled many normal images as pneumonia. Validation performance was substantially stronger than test performance, so the validation score did not transfer fully to the held-out partition.

The selected checkpoint and search curves are in `outputs/part_b/b1/`; the test set was evaluated once after the validation-selected learning rate was chosen.

## B2 - VGG-like model and batch-size search

B2 used B1's selected learning rate (**0.001**) and searched batch sizes `{16, 32, 64, 128, 256}`. As in B1, validation BCE loss selected the checkpoint within each trial, and validation classification error selected the batch size.

Batch size **32** was selected at **epoch 19**, with validation loss **0.0631**, validation error **1.72%**, sensitivity **98.97%**, and specificity **96.30%**.

| Batch size | Best epoch | Validation loss | Validation error | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 0.0907 | 3.63% | 98.46% | 90.37% |
| 32 | 19 | 0.0631 | 1.72% | 98.97% | 96.30% |
| 64 | 5 | 0.1056 | 3.44% | 98.71% | 90.37% |
| 128 | 17 | 0.0858 | 3.24% | 99.23% | 89.63% |
| 256 | 13 | 0.1040 | 3.24% | 99.49% | 88.89% |

On the reserved test set, B2 achieved **89.58% accuracy** and **10.42% error**, with **99.49% sensitivity** and **73.08% specificity**. Its counts were TP=388, FN=2, TN=171, and FP=63. Relative to B1, B2 retained the same high sensitivity and improved specificity by about 16.67 percentage points, although it still produced more false positives than false negatives. The B2 validation and test results are saved under `outputs/part_b/b2/`.

## B3 - ResNet-like model and architecture comparison

B3 used the selected B1 learning rate (**0.001**) and B2 batch size (**32**). Its best validation-loss checkpoint was epoch **9**, with validation loss **0.0760**, validation error **2.67%**, sensitivity **98.20%**, and specificity **94.81%**. On the reserved test set, B3 achieved **87.98% accuracy** and **12.02% error**, with **98.97% sensitivity** and **69.66% specificity** (TP=386, FN=4, TN=163, FP=71).

| Model | Test error | Accuracy | Sensitivity | Specificity | Trainable parameters | FLOPs per 64x64 image |
|---|---:|---:|---:|---:|---:|---:|
| B1 AlexNet-like | 16.67% | 83.33% | 99.49% | 56.41% | 99,569 | 39,063,616 |
| B2 VGG-like | 10.42% | 89.58% | 99.49% | 73.08% | 18,185 | 19,464,256 |
| B3 ResNet-like | 12.02% | 87.98% | 98.97% | 69.66% | 77,169 | 25,297,024 |

The FLOPs convention counts two operations per multiply-accumulate in convolutional and linear layers and excludes BatchNorm, ReLU, pooling, and bias operations. Counts use one `(1, 64, 64)` input for all models. Under this convention, B2 has the best test error and the lowest parameter and FLOP counts. B1 has slightly higher sensitivity, tied with B2 to the reported precision, but its much lower specificity yields more false positives. B3 lies between them on test error and computation. These results favor B2 for this split and threshold; they do not establish that it will be best on another dataset or threshold.

The B3 training curve and the combined test comparison table are saved under `outputs/part_b/b3/` and `outputs/part_b/model_comparison.csv`.

## B4 - Grad-CAM comparison

Grad-CAM was applied to the pneumonia logit for the same five reproducibly sampled pneumonia examples from the held-out test set: indices **486, 279, 413, 53, and 276** (seed 42). Each model used its final convolutional feature block: B1's last 64-channel convolution block, B2's last 32-channel convolution block, and B3's last residual block's second 64-channel convolution. The figure overlays each normalized heatmap on the same original grayscale image for an architecture-by-architecture comparison.

All three models classified all five selected examples as pneumonia. Their predicted pneumonia probabilities ranged from **0.864 to 1.000** for B1, **0.929 to 1.000** for B2, and **0.995 to 1.000** for B3. The heatmaps are visibly architecture-dependent: B1 tends to produce broader central regions, B2 shows more localized activations that often touch image borders, and B3 produces broader, varying regions across the five examples. This variation means the models do not rely on identical image regions even when their class predictions agree.

These maps are coarse, gradient-based visualizations of features associated with the selected class score. Activations near image margins may reflect image framing or acquisition artifacts; they do not establish that a model used a spurious cue. Grad-CAM does not prove causal reasoning or clinical localization, and these five positive cases cannot establish behavior across the full test set. The shared-image overlays, sampled indices, and per-model probabilities are saved in `outputs/part_b/b4/`.
