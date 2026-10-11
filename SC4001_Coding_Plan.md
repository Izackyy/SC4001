# SC4001 Assignment Coding Plan

References: `assignment.pdf` (pp. 1-7) and the lecture slide decks in `../../Lect/`. Slide numbers below are 1-indexed PDF pages.

## Shared rules and data handling

- Use Adam and the assignment's early-stopping settings throughout: maximum 20 epochs, patience 4, minimum improvement `1e-4`.
- For Part A, use 2017-2020 for training, 2021 for validation/model selection, and 2022 for final testing. For Part B, keep the supplied train/validation/test partitions.
- Do not use test metrics to choose hyperparameters or checkpoints. Part A1 specifically asks for a training/test RMSE curve after the train-plus-validation refit; record that curve for reporting, but do not choose from it.
- Fit category maps and continuous-feature scaling on the training partition only during model selection. When refitting on train plus validation, refit preprocessing on those rows; never use test rows to fit it. Reserve category ID 0 for categories unseen during fitting, as the supplied baseline does.
- Save each run's configuration, seed, epoch history, selected checkpoint, and metrics under separate Part A/Part B output folders.

### Stopping rule for train-plus-validation refits

The brief asks for early stopping throughout, but A1-A3 also ask for final models trained on the combined training and validation data. To keep every combined row in training without using 2022 for stopping, use the selected A1 best epoch as the fixed training length for these refits (A2 explicitly asks for the A1 epoch count). Do not stop or select a model from the test curve. If using a new internal validation split instead, document that not every train-plus-validation row was used for fitting.

## Part A: HDB resale-price regression

### 1. Prepare and understand the supplied baseline

1. Extract `housing_baseline_data.zip` and follow its README from the extracted project root: `uv sync`, then `uv run python src/baseline.py`.
2. Extend the provided `config.py`, `data.py`, `model.py`, and `baseline.py` structure. Keep the split rule, feature lists, target scaling, and unknown-category behavior explicit.
3. Use these ten model inputs: categorical `month`, `town`, `flat_model_type`, `storey_range`; continuous `dist_to_nearest_stn`, `dist_to_dhoby`, `degree_centrality`, `eigenvector_centrality`, `remaining_lease_years`, and `floor_area_sqm`. Use `year` only to create the split; exclude `full_address` and `nearest_stn`. Predict `resale_price`.
4. Preserve the baseline block order: four embedding tables -> concatenate embeddings and six scaled continuous values -> `Linear` -> `LayerNorm` -> `ReLU` -> one-unit linear output. Count trainable parameters with `sum(p.numel() for p in model.parameters() if p.requires_grad)`; include embedding rows and LayerNorm parameters.

### 2. A1 - baseline search and final evaluation

1. Use Ray Tune for all 25 combinations: hidden width `{16, 32, 64, 128, 256}` x shared embedding width `{2, 4, 8, 12, 16}`. Keep the data split, preprocessing, seed, optimizer, and stopping rule consistent across trials.
2. Log train and validation RMSE each epoch. Select the configuration with the lowest best validation RMSE and save its parameters and curves.
3. Refit that configuration on train plus validation for the selected epoch count. Convert predictions back to Singapore dollars before calculating and plotting RMSE. Report test RMSE and R2, and produce the requested train/test RMSE curve.

### 3. A2 - five-seed ablations

1. For each of five fixed seeds, train the selected baseline and both variants on train plus validation using the A1 preprocessing choices, hyperparameters, and epoch count.
2. Variant A removes all embeddings and feeds the six continuous values to the unchanged-width MLP. Variant B keeps all inputs and replaces the hidden ReLU with Sigmoid; keep the other baseline layers unchanged.
3. Evaluate each run on the test set once. Compute RMSE and R2 over the complete test set, then report the mean and sample standard deviation across seeds. Plot the requested RMSE and R2 comparisons.

### 4. A3 - wide-and-deep model

1. Add a one-layer linear branch that receives the same concatenated inputs as the MLP. Sum the MLP output and `lambda * linear_output`.
2. Select lambda from `{0, 0.25, 0.5, 1, 2}` using validation RMSE, with A1 hyperparameters fixed.
3. Refit the selected lambda on train plus validation for five seeds. Compare against the five-seed baseline runs from A2 for a like-for-like comparison, and also report the A1 baseline requested in the brief. Include the selected model's test RMSE/R2 mean and sample standard deviation.

### 5. A4 - FeatureAblation

1. Train the final A1 baseline and selected A3 wide-and-deep model with the same seed.
2. Run Captum FeatureAblation on the test rows, masking each of the ten original features separately. Record the category and numeric baselines used; the supplied category ID 0 represents an unseen category.
3. Ensure the attribution target is the predicted price in dollars. Both models' `forward` methods return a standardized target, so wrap/de-normalize their outputs (or scale the attributions by the target standard deviation).
4. For each feature, average absolute attribution over test samples. Plot the ten scores in descending order with Singapore-dollar units; report the top three per model and one limitation.

## Part B: chest X-ray classification

### 6. Prepare the supplied image partitions

1. Load `train_images`, `val_images`, `test_images` and their labels from `chestxray_binary_64.npz`. The image arrays are grayscale `(N, 64, 64)`; add a channel dimension to get `(N, 1, 64, 64)`.
2. Convert intensities to `[0, 1]`, calculate the pixel mean and standard deviation from the training images only, and apply those values to all three partitions.
3. Confirm which label denotes pneumonia and keep that as the positive class for sensitivity and Grad-CAM comparisons. Use one-logit binary output with `BCEWithLogitsLoss`; apply sigmoid and a fixed 0.5 threshold only for probabilities/metrics.

### 7. Implement the three architectures

Apply BatchNorm and ReLU after every convolution, use `bias=False` in convolutions and biased fully connected layers, and use SAME padding as required.

- **B1 AlexNet-like:** 5x5 conv 16 -> max pool /2 -> 5x5 conv 32 -> max pool /2 -> 3x3 conv 64 -> max pool /2 -> adaptive average pool 4x4 -> FC 64 -> FC 32 -> one-logit output. Use dropout 0.5 only in the hidden fully connected layers.
- **B2 VGG-like:** two 3x3 convs with 8 channels -> pool /2 -> two 3x3 convs with 16 channels -> pool /2 -> two 3x3 convs with 32 channels -> pool /2 -> adaptive average pool 1x1 -> one-logit output.
- **B3 ResNet-like:** 3x3 conv 16 -> max pool /2 -> 16-channel residual block (two 3x3 convs, identity shortcut) -> 32-channel residual block (first conv stride 2, 1x1 stride-2 shortcut) -> 64-channel residual block (first conv stride 2, 1x1 stride-2 shortcut) -> adaptive average pool 1x1 -> one-logit output. Apply ReLU after each convolution and residual addition.

The figure caption says ReLU is used throughout, while the final one-unit classifier needs an unconstrained logit for binary cross-entropy. Interpret this as ReLU on hidden/convolutional layers and after residual additions, with no activation after the final output layer.

### 8. B1-B3 training and comparison

1. **B1:** fix batch size at 128; search learning rates `{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`. Plot validation loss and accuracy across epochs. At each trial's selected checkpoint, record best epoch, validation loss/error, sensitivity, and specificity. Select the learning rate by lowest validation classification error, then evaluate the final model once on test.
2. **B2:** use B1's selected learning rate; search batch sizes `{16, 32, 64, 128, 256}` and repeat the validation plots/table. Select by validation classification error, then evaluate the final model once on test.
3. **B3:** use the selected learning rate and batch size. Report test accuracy, sensitivity, and specificity. Compare B1-B3 in one table with error, trainable parameter count, and FLOPs. State the FLOPs convention and input size; use the same convention for all three networks.
4. Use validation loss as the early-stopping monitor for B trials, then compare trials by validation classification error as requested. Define sensitivity as `TP/(TP+FN)` and specificity as `TN/(TN+FP)` with pneumonia positive.

### 9. B4 - Grad-CAM

Select five pneumonia images reproducibly from one partition and use the exact same five images for B1, B2, and B3. Apply Grad-CAM to each model's final convolutional layer, overlay the heatmaps on the grayscale X-rays, and compare where each model places emphasis. Keep the image preprocessing consistent with training and state one limitation of heatmap interpretation.

## Run and submission checklist

- Make the code runnable from a clean environment and from the documented project root. README must list environment, dependencies, seeds, and exact run commands.
- Keep Part A and Part B code/report outputs separate. Save plots with axis labels, units, legend, and selected checkpoint/config noted.
- Before zipping, run each documented command from start to finish and confirm it recreates the required outputs. Follow the brief's separate `.pdf` reports and `.zip` code names; include the required Generative AI declaration in each report if applicable.

## Relevant lecture slides

| Lecture deck | Slides (PDF pages) | Use in this assignment |
|---|---:|---|
| [`1_foundations.pdf`](../../Lect/1_foundations.pdf) | 15-17, 18-27 | PyTorch tensors/computational graphs; training, epochs, and batches. Foundational review. |
| [`2_regression_classification.pdf`](../../Lect/2_regression_classification.pdf) | 16-19, 30-37, 46, 60-62 | Regression output; binary logistic classification, cross-entropy, classification error, and learning-rate behavior. |
| [`3_neuron_layers.pdf`](../../Lect/3_neuron_layers.pdf) | 2-5, 55-57 | Dense-layer weights/biases, batch-shaped inputs, and parameter initialization. |
| [`4_dnn.pdf`](../../Lect/4_dnn.pdf) | 6-9, 35, 40-42, 49-54, 60-64 | Feed-forward model structure, input normalization, width/depth selection, model capacity, and batch size. |
| [`5_model_selection.pdf`](../../Lect/5_model_selection.pdf) | 21-23, 31-35, 45-49 | Three-way split, validation/test separation, under/overfitting, early stopping, and dropout. |
| [`6_cnn.pdf`](../../Lect/6_cnn.pdf) | 10-47, 48-60, 67-77, 79-92 | Convolution shapes/padding, pooling, fully connected layers, ReLU, image preprocessing, epochs/batches, and Adam. |
| [`7_cnn_architectures.pdf`](../../Lect/7_cnn_architectures.pdf) | 3-14, 16-25, 44-62, 65 | AlexNet/VGG/ResNet context and residuals, convolution parameter/FLOP calculations, BatchNorm, and overfitting. |
| [`8_rnn.pdf`](../../Lect/8_rnn.pdf) | - | Sequence models; no direct match to the tabular regression or image-classification tasks here. |

The provided lecture text does not directly cover tabular categorical embeddings, Ray Tune, Captum FeatureAblation, or Grad-CAM. Use the assignment brief for those specific requirements.
