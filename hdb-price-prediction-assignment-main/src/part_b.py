import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = Path("../chestxray_binary_64.npz")
OUTPUT_DIR = Path("outputs/part_b")
SEED = 42
MAX_EPOCHS = 20
PATIENCE = 4
MIN_IMPROVEMENT = 1e-4
B1_LEARNING_RATES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
B2_BATCH_SIZES = (16, 32, 64, 128, 256)
PNEUMONIA_LABEL = 1
MODEL_NAMES = ("B1 AlexNet-like", "B2 VGG-like", "B3 ResNet-like")


class ConvUnit(nn.Sequential):
    """Convolution, batch normalization, and ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        """Build a SAME-padded convolution block without convolution bias.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Odd convolution kernel width and height.
            stride: Convolution stride.
        """
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class AlexNetLike(nn.Module):
    """B1: five-stage AlexNet-like classifier for one-channel 64x64 images."""

    def __init__(self) -> None:
        """Create the B1 convolutional stack and dropout classifier."""
        super().__init__()
        self.features = nn.Sequential(
            ConvUnit(1, 16, 5),
            nn.MaxPool2d(2),
            ConvUnit(16, 32, 5),
            nn.MaxPool2d(2),
            ConvUnit(32, 64, 3),
            nn.MaxPool2d(2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one pneumonia logit per image.

        Args:
            images: Normalized grayscale batch with shape ``(N, 1, 64, 64)``.

        Returns:
            Logits with shape ``(N,)``.
        """
        features = self.adaptive_pool(self.features(images))
        return self.classifier(features.flatten(start_dim=1)).squeeze(1)


class VGGLike(nn.Module):
    """B2: three two-convolution VGG-like stages and global average pooling."""

    def __init__(self) -> None:
        """Create the B2 convolutional stages and one-logit output layer."""
        super().__init__()
        self.stage1 = nn.Sequential(ConvUnit(1, 8, 3), ConvUnit(8, 8, 3))
        self.stage2 = nn.Sequential(ConvUnit(8, 16, 3), ConvUnit(16, 16, 3))
        self.stage3 = nn.Sequential(ConvUnit(16, 32, 3), ConvUnit(32, 32, 3))
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(32, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one pneumonia logit per image.

        Args:
            images: Normalized grayscale batch with shape ``(N, 1, 64, 64)``.

        Returns:
            Logits with shape ``(N,)``.
        """
        features = self.pool(self.stage1(images))
        features = self.pool(self.stage2(features))
        features = self.pool(self.stage3(features))
        return self.output(self.adaptive_pool(features).flatten(start_dim=1)).squeeze(1)


class ResidualBlock(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Create the residual path and shape-matching shortcut.

        Args:
            in_channels: Number of channels entering the block.
            out_channels: Number of channels leaving the block.
            stride: Stride used by the first residual convolution.
        """
        super().__init__()
        self.conv1 = ConvUnit(in_channels, out_channels, 3, stride=stride)
        self.conv2 = ConvUnit(out_channels, out_channels, 3)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else ConvUnit(in_channels, out_channels, 1, stride=stride)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual path and ReLU after the skip addition.

        Args:
            inputs: Feature batch entering the block.

        Returns:
            Feature batch after residual addition and activation.
        """
        return F.relu(self.conv2(self.conv1(inputs)) + self.shortcut(inputs))


class ResNetLike(nn.Module):
    """B3: three residual stages following a convolution and max pool."""

    def __init__(self) -> None:
        """Create the B3 stem, residual blocks, and one-logit classifier."""
        super().__init__()
        self.stem = ConvUnit(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.block16 = ResidualBlock(16, 16)
        self.block32 = ResidualBlock(16, 32, stride=2)
        self.block64 = ResidualBlock(32, 64, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(64, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one pneumonia logit per image.

        Args:
            images: Normalized grayscale batch with shape ``(N, 1, 64, 64)``.

        Returns:
            Logits with shape ``(N,)``.
        """
        features = self.block16(self.pool(self.stem(images)))
        features = self.block32(features)
        features = self.block64(features)
        return self.output(self.adaptive_pool(features).flatten(start_dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Reproducibility seed used for model initialization and shuffling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_path: Path) -> dict:
    """Load the fixed partitions and normalize images from train statistics.

    Args:
        data_path: Path to the supplied ``chestxray_binary_64.npz`` archive.
    Returns:
        Mapping with train, validation, and test datasets, raw test images,
        train-only pixel mean/std, and the original partition labels.
    """
    with np.load(data_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    train_raw = torch.as_tensor(arrays["train_images"], dtype=torch.float32) / 255.0
    train_mean = train_raw.mean().item()
    train_std = train_raw.std(unbiased=False).clamp_min(1e-8).item()
    data = {"train_mean": train_mean, "train_std": train_std}
    for split in ("train", "val", "test"):
        raw = torch.as_tensor(arrays[f"{split}_images"], dtype=torch.float32) / 255.0
        normalized = ((raw - train_mean) / train_std).unsqueeze(1).contiguous(
            memory_format=torch.channels_last
        )
        labels = torch.as_tensor(arrays[f"{split}_labels"], dtype=torch.float32)
        data[split] = TensorDataset(normalized, labels)
        data[f"{split}_raw"] = raw
        data[f"{split}_labels"] = labels.to(torch.long)
    return data


def build_model(model_name: str) -> nn.Module:
    """Construct one of the three assignment-specified CNNs.

    Args:
        model_name: One of the names in ``MODEL_NAMES``.

    Returns:
        A newly initialized classifier.

    Raises:
        ValueError: If ``model_name`` is not a supported architecture.
    """
    models = {
        MODEL_NAMES[0]: AlexNetLike,
        MODEL_NAMES[1]: VGGLike,
        MODEL_NAMES[2]: ResNetLike,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return models[model_name]()


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float | int]:
    """Calculate binary metrics with pneumonia label 1 and threshold 0.5.

    Args:
        logits: One raw binary logit per example.
        labels: Binary targets where 1 denotes pneumonia.

    Returns:
        Accuracy, classification error, sensitivity, specificity, and confusion
        counts using a fixed sigmoid probability threshold of 0.5.
    """
    predicted = torch.sigmoid(logits) >= 0.5
    positive = labels >= 0.5
    true_positive = int((predicted & positive).sum().item())
    false_negative = int((~predicted & positive).sum().item())
    true_negative = int((~predicted & ~positive).sum().item())
    false_positive = int((predicted & ~positive).sum().item())
    total = len(labels)
    accuracy = (true_positive + true_negative) / total
    return {
        "accuracy": accuracy,
        "classification_error": 1.0 - accuracy,
        "sensitivity": true_positive / max(true_positive + false_negative, 1),
        "specificity": true_negative / max(true_negative + false_positive, 1),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Evaluate BCE loss and classification metrics for one data partition.

    Args:
        model: Classifier to evaluate.
        dataset: Partition of normalized images and binary labels.
        device: Device used for model evaluation.
        batch_size: Evaluation batch size.

    Returns:
        Mean BCE loss and binary classification metrics.
    """
    model.eval()
    total_loss = 0.0
    logits_all = []
    labels_all = []
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
    metrics = classification_metrics(torch.cat(logits_all), torch.cat(labels_all))
    metrics["loss"] = total_loss / len(dataset)
    return metrics


def fit_candidate(
    model: nn.Module,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[dict], dict]:
    """Train with Adam, stop on validation loss, and select by error.

    Args:
        model: Newly initialized CNN to train.
        train_data: Training partition.
        validation_data: Validation partition used for checkpoint selection.
        learning_rate: Adam learning rate for this trial.
        batch_size: Training and validation batch size.
        device: Device used for model training.

    Returns:
        Lowest-error model state, per-epoch history, and its validation metrics.
    """
    model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED + 10_000)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )
    best_state = None
    best_loss = float("inf")
    best_error = float("inf")
    selected_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)

        validation_metrics = evaluate_model(
            model, validation_data, device, batch_size
        )
        train_loss = train_loss_sum / len(train_data)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": float(validation_metrics["loss"]),
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "validation_error": float(
                    validation_metrics["classification_error"]
                ),
                "validation_sensitivity": float(validation_metrics["sensitivity"]),
                "validation_specificity": float(validation_metrics["specificity"]),
            }
        )
        current_loss = float(validation_metrics["loss"])
        current_error = float(validation_metrics["classification_error"])
        if current_error < best_error or (
            current_error == best_error and current_loss < selected_loss
        ):
            best_error = current_error
            selected_loss = current_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if current_loss < best_loss - MIN_IMPROVEMENT:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    best_metrics = evaluate_model(model, validation_data, device, batch_size)
    best_metrics["best_epoch"] = best_epoch
    return best_state, history, best_metrics


def save_search_plot(
    histories: dict[str, list[dict]],
    title: str,
    output_path: Path,
) -> None:
    """Plot validation loss and accuracy by epoch for a hyperparameter search.

    Args:
        histories: Per-trial epoch metrics keyed by learning rate or batch size.
        title: Figure title identifying the search.
        output_path: PNG destination for the two-panel and separate figures.

    Returns:
        None.

    Side Effects:
        Writes the combined figure to ``output_path`` and separate validation
        loss and accuracy figures beside it.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, history in histories.items():
        frame = pd.DataFrame(history)
        axes[0].plot(frame["epoch"], frame["validation_loss"], marker=".", label=label)
        axes[1].plot(
            frame["epoch"], frame["validation_accuracy"], marker=".", label=label
        )
    axes[0].set(xlabel="Epoch", ylabel="Validation BCE loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy", title="Accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    for metric, ylabel, filename in (
        ("validation_loss", "Validation BCE loss", "validation_loss.png"),
        ("validation_accuracy", "Validation accuracy", "validation_accuracy.png"),
    ):
        metric_figure, axis = plt.subplots(figsize=(7, 4.5))
        for label, history in histories.items():
            frame = pd.DataFrame(history)
            axis.plot(frame["epoch"], frame[metric], marker=".", label=label)
        axis.set(xlabel="Epoch", ylabel=ylabel, title=f"{ylabel} vs epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        metric_figure.suptitle(title)
        metric_figure.tight_layout(rect=(0, 0, 1, 0.95))
        metric_figure.savefig(output_path.with_name(filename), dpi=180)
        plt.close(metric_figure)


def candidate_label(value, candidate_key: str) -> str:
    """Normalize a trial setting for consistent filenames and state lookup.

    Args:
        value: Learning-rate or batch-size candidate from Python or pandas.
        candidate_key: Name of the candidate column.

    Returns:
        Stable string label independent of integer/float dtype conversion.
    """
    if candidate_key == "batch_size":
        return str(int(float(value)))
    return format(float(value), ".12g")


def train_search(
    model_name: str,
    candidates: tuple,
    candidate_key: str,
    learning_rate: float,
    batch_size: int,
    data: dict,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, nn.Module, dict]:
    """Train candidate settings, select by validation error, and save the winner.

    Args:
        model_name: CNN architecture being tuned.
        candidates: Learning rates or batch sizes to compare.
        candidate_key: Result-column name for each candidate setting.
        learning_rate: Fixed Adam rate when batch size is searched.
        batch_size: Fixed batch size when learning rate is searched.
        data: Preprocessed image partitions.
        device: Device used for training and evaluation.
        output_dir: Destination for tables, histories, curves, and checkpoint.

    Returns:
        Candidate summary table, selected model, and selected trial metadata.

    Side Effects:
        Trains one model per candidate and writes the validation search outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    histories = {}
    states = {}
    for candidate in candidates:
        key = candidate_label(candidate, candidate_key)
        trial_learning_rate = (
            float(candidate) if candidate_key == "learning_rate" else learning_rate
        )
        trial_batch_size = (
            int(candidate) if candidate_key == "batch_size" else batch_size
        )
        history_path = output_dir / f"history_{candidate_key}_{key}.csv"
        state_path = output_dir / f"trial_{candidate_key}_{key}.pt"
        set_seed(SEED)
        model = build_model(model_name)
        state, history, validation = fit_candidate(
            model,
            data["train"],
            data["val"],
            trial_learning_rate,
            trial_batch_size,
            device,
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        torch.save(state, state_path)
        histories[key] = history
        states[key] = state
        summary_rows.append(
            {
                candidate_key: candidate,
                "learning_rate": trial_learning_rate,
                "batch_size": trial_batch_size,
                "best_epoch": int(validation["best_epoch"]),
                "validation_loss": float(validation["loss"]),
                "validation_error": float(validation["classification_error"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_sensitivity": float(validation["sensitivity"]),
                "validation_specificity": float(validation["specificity"]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["validation_error", "validation_loss", candidate_key]
    )
    selected = summary.iloc[0].to_dict()
    selected_key = candidate_label(selected[candidate_key], candidate_key)
    selected_state = states[selected_key]
    selected_model = build_model(model_name).to(
        device, memory_format=torch.channels_last
    )
    selected_model.load_state_dict(selected_state)
    selected_model.eval()
    torch.save(
        {
            "model_name": model_name,
            "state_dict": selected_state,
            "seed": SEED,
            "learning_rate": float(selected["learning_rate"]),
            "batch_size": int(selected["batch_size"]),
            "best_epoch": int(selected["best_epoch"]),
        },
        output_dir / "selected_model.pt",
    )
    summary.to_csv(output_dir / "search_summary.csv", index=False)
    pd.DataFrame(
        [row for history in histories.values() for row in history]
    ).to_csv(output_dir / "search_history.csv", index=False)
    save_search_plot(
        histories,
        f"{model_name} validation search",
        output_dir / "validation_curves.png",
    )
    return summary, selected_model, selected


def count_parameters(model: nn.Module) -> int:
    """Count all trainable parameters in a model.

    Args:
        model: PyTorch model to count.

    Returns:
        Number of trainable scalar parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_flops(model: nn.Module, device: torch.device) -> int:
    """Estimate dense FLOPs as two operations per Conv2d/Linear MAC.

    BatchNorm, activation, pooling, and bias operations are excluded from this
    convention. A single one-channel 64x64 image is used for every architecture.

    Args:
        model: CNN whose Conv2d and Linear operations are counted.
        device: Device used for the shape-propagation forward pass.

    Returns:
        Estimated number of FLOPs for one ``(1, 64, 64)`` input image.
    """
    macs = {"total": 0}
    handles = []

    def count_conv(_module, _inputs, output):
        """Accumulate convolution multiply-accumulates from output shape."""
        module = _module
        output_elements = output.numel()
        kernel_elements = module.kernel_size[0] * module.kernel_size[1]
        macs["total"] += (
            output_elements
            * (module.in_channels // module.groups)
            * kernel_elements
        )

    def count_linear(_module, _inputs, output):
        """Accumulate linear-layer multiply-accumulates from output shape."""
        macs["total"] += output.numel() * _module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(count_conv))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(count_linear))
    prior_mode = model.training
    model.eval()
    with torch.no_grad():
        model(
            torch.zeros(1, 1, 64, 64, device=device).contiguous(
                memory_format=torch.channels_last
            )
        )
    for handle in handles:
        handle.remove()
    model.train(prior_mode)
    return 2 * macs["total"]


def run_b1(data: dict, device: torch.device) -> dict:
    """Run B1's learning-rate search and evaluate its selected model once.

    Args:
        data: Preprocessed train, validation, and test partitions.
        device: Device used for model training and evaluation.

    Returns:
        Selected learning rate and validation/test metrics.

    Side Effects:
        Writes B1 trial histories, plots, summary, and selected checkpoint.
    """
    output_dir = OUTPUT_DIR / "b1"
    summary, model, selected = train_search(
        MODEL_NAMES[0],
        B1_LEARNING_RATES,
        "learning_rate",
        learning_rate=0.0,
        batch_size=128,
        data=data,
        device=device,
        output_dir=output_dir,
    )
    test_metrics = evaluate_model(model, data["test"], device, 128)
    summary.to_csv(output_dir / "search_summary.csv", index=False)
    result = {
        "model_name": MODEL_NAMES[0],
        "selected_learning_rate": float(selected["learning_rate"]),
        "batch_size": 128,
        "best_epoch": int(selected["best_epoch"]),
        "validation_loss": float(selected["validation_loss"]),
        "validation_error": float(selected["validation_error"]),
        "validation_accuracy": float(selected["validation_accuracy"]),
        "validation_sensitivity": float(selected["validation_sensitivity"]),
        "validation_specificity": float(selected["validation_specificity"]),
        "test": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("B1 selected trial:", json.dumps(result, indent=2))
    return result


def run_b2(data: dict, device: torch.device, b1_metrics: dict) -> dict:
    """Run B2's batch-size search using B1's selected learning rate.

    Args:
        data: Preprocessed train, validation, and test partitions.
        device: Device used for model training and evaluation.
        b1_metrics: B1 result containing the selected learning rate.

    Returns:
        Selected batch size and validation/test metrics.

    Side Effects:
        Writes B2 trial histories, plots, summary, and selected checkpoint.
    """
    output_dir = OUTPUT_DIR / "b2"
    learning_rate = float(b1_metrics["selected_learning_rate"])
    summary, model, selected = train_search(
        MODEL_NAMES[1],
        B2_BATCH_SIZES,
        "batch_size",
        learning_rate=learning_rate,
        batch_size=0,
        data=data,
        device=device,
        output_dir=output_dir,
    )
    test_metrics = evaluate_model(
        model, data["test"], device, int(selected["batch_size"])
    )
    summary.to_csv(output_dir / "search_summary.csv", index=False)
    result = {
        "model_name": MODEL_NAMES[1],
        "learning_rate": learning_rate,
        "selected_batch_size": int(selected["batch_size"]),
        "best_epoch": int(selected["best_epoch"]),
        "validation_loss": float(selected["validation_loss"]),
        "validation_error": float(selected["validation_error"]),
        "validation_accuracy": float(selected["validation_accuracy"]),
        "validation_sensitivity": float(selected["validation_sensitivity"]),
        "validation_specificity": float(selected["validation_specificity"]),
        "test": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("B2 selected trial:", json.dumps(result, indent=2))
    return result


def run_b3(data: dict, device: torch.device, b1_metrics: dict, b2_metrics: dict) -> dict:
    """Train and test B3 using the selected B1 learning rate and B2 batch size.

    Args:
        data: Preprocessed train, validation, and test partitions.
        device: Device used for model training and evaluation.
        b1_metrics: B1 result containing the selected learning rate.
        b2_metrics: B2 result containing the selected batch size.

    Returns:
        B3 validation and test metrics, parameter count, and FLOPs estimate.

    Side Effects:
        Writes B3 validation history, checkpoint, and test metrics.
    """
    output_dir = OUTPUT_DIR / "b3"
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_rate = float(b1_metrics["selected_learning_rate"])
    batch_size = int(b2_metrics["selected_batch_size"])
    set_seed(SEED)
    model = build_model(MODEL_NAMES[2])
    state, history, validation = fit_candidate(
        model,
        data["train"],
        data["val"],
        learning_rate,
        batch_size,
        device,
    )
    model.load_state_dict(state)
    model.eval()
    test_metrics = evaluate_model(model, data["test"], device, batch_size)
    result = {
        "model_name": MODEL_NAMES[2],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_epoch": int(validation["best_epoch"]),
        "validation_loss": float(validation["loss"]),
        "validation_error": float(validation["classification_error"]),
        "validation_accuracy": float(validation["accuracy"]),
        "validation_sensitivity": float(validation["sensitivity"]),
        "validation_specificity": float(validation["specificity"]),
        "test": test_metrics,
        "trainable_parameters": count_parameters(model),
        "flops_per_64x64_image": count_flops(model, device),
    }
    torch.save(
        {
            "model_name": MODEL_NAMES[2],
            "state_dict": state,
            "seed": SEED,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "best_epoch": int(validation["best_epoch"]),
        },
        output_dir / "selected_model.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    save_search_plot(
        {"B3": history}, "B3 validation learning curves", output_dir / "validation_curves.png"
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("B3 result:", json.dumps(result, indent=2))
    return result


def load_checkpoint_model(
    model_name: str, checkpoint_path: Path, device: torch.device
) -> nn.Module:
    """Load a selected model state from one Part B checkpoint.

    Args:
        model_name: Architecture name used to construct the model.
        checkpoint_path: Saved selected-model checkpoint.
        device: Device for inference.

    Returns:
        Evaluation-mode model loaded with the saved state dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(model_name).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def gradcam_maps(
    model: nn.Module,
    target_layer: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Compute positive-class Grad-CAM maps for a batch of images.

    Args:
        model: Evaluation-mode classifier returning pneumonia logits.
        target_layer: Final convolution block to explain.
        images: Preprocessed grayscale images with shape ``(N, 1, 64, 64)``.

    Returns:
        Per-image normalized heatmaps resized to ``64x64``.
    """
    activations = {}
    gradients = {}

    def capture_activation(_module, _inputs, output):
        """Save the selected convolution block's activation map."""
        activations["value"] = output

    def capture_gradient(_module, _grad_input, grad_output):
        """Save the gradient of the positive-class score at the target block."""
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(capture_activation)
    backward_handle = target_layer.register_full_backward_hook(capture_gradient)
    model.zero_grad(set_to_none=True)
    logits = model(images)
    logits.sum().backward()
    forward_handle.remove()
    backward_handle.remove()

    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    maps = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
    maps = F.interpolate(maps, size=(64, 64), mode="bilinear", align_corners=False)
    minimum = maps.amin(dim=(2, 3), keepdim=True)
    maximum = maps.amax(dim=(2, 3), keepdim=True)
    return (maps - minimum) / (maximum - minimum).clamp_min(1e-8)


def run_b4(data: dict, device: torch.device) -> dict:
    """Create comparable Grad-CAM overlays for five pneumonia test images.

    Args:
        data: Preprocessed partitions including raw test images and labels.
        device: Device used for model inference and gradients.

    Returns:
        Saved test indices and the selected model's predictions on those rows.

    Side Effects:
        Saves the sampled image indices and a shared-image Grad-CAM figure.
    """
    output_dir = OUTPUT_DIR / "b4"
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_indices = torch.where(data["test_labels"] == PNEUMONIA_LABEL)[0].numpy()
    selected_indices = np.random.default_rng(SEED).choice(
        positive_indices, size=5, replace=False
    )
    images = torch.stack(
        [data["test"][int(index)][0] for index in selected_indices]
    ).to(device, memory_format=torch.channels_last)
    raw_images = data["test_raw"][selected_indices].numpy()
    selected_labels = data["test_labels"][selected_indices].numpy()
    checkpoint_paths = {
        MODEL_NAMES[0]: OUTPUT_DIR / "b1" / "selected_model.pt",
        MODEL_NAMES[1]: OUTPUT_DIR / "b2" / "selected_model.pt",
        MODEL_NAMES[2]: OUTPUT_DIR / "b3" / "selected_model.pt",
    }
    figure, axes = plt.subplots(3, 5, figsize=(15, 9))
    prediction_rows = []
    for row, model_name in enumerate(MODEL_NAMES):
        model = load_checkpoint_model(model_name, checkpoint_paths[model_name], device)
        if model_name == MODEL_NAMES[0]:
            target_layer = model.features[4]
        elif model_name == MODEL_NAMES[1]:
            target_layer = model.stage3[1]
        else:
            target_layer = model.block64.conv2
        cams = gradcam_maps(model, target_layer, images)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(images)).cpu().numpy()
        for column, (heatmap, probability) in enumerate(
            zip(cams.detach().cpu().numpy(), probabilities)
        ):
            axis = axes[row, column]
            axis.imshow(raw_images[column], cmap="gray", vmin=0, vmax=1)
            axis.imshow(heatmap[0], cmap="jet", alpha=0.4, vmin=0, vmax=1)
            axis.axis("off")
            if row == 0:
                axis.set_title(f"Test index {int(selected_indices[column])}")
            prediction_rows.append(
                {
                    "model": model_name,
                    "test_index": int(selected_indices[column]),
                    "true_label": int(selected_labels[column]),
                    "pneumonia_probability": float(probability),
                }
            )
    figure.suptitle("B4 Grad-CAM: same five pneumonia test images")
    figure.tight_layout(rect=(0.04, 0, 1, 0.95))
    for row, model_name in enumerate(MODEL_NAMES):
        figure.text(
            0.015,
            0.82 - row * 0.32,
            model_name,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )
    figure.savefig(output_dir / "gradcam_comparison.png", dpi=180)
    plt.close(figure)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "sample_predictions.csv", index=False)
    pd.DataFrame(
        {
            "test_index": selected_indices.astype(int),
            "true_label": selected_labels.astype(int),
        }
    ).to_csv(output_dir / "selected_images.csv", index=False)
    result = {
        "seed": SEED,
        "partition": "test",
        "positive_class": PNEUMONIA_LABEL,
        "test_indices": selected_indices.astype(int).tolist(),
        "heatmap_target": "pneumonia logit",
        "target_layers": {
            MODEL_NAMES[0]: "last convolution block (64 channels)",
            MODEL_NAMES[1]: "last convolution block (32 channels)",
            MODEL_NAMES[2]: "last residual block second convolution (64 channels)",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("B4 selected images:", json.dumps(result, indent=2))
    return result


def save_comparison_table(
    b1: dict, b2: dict, b3: dict, device: torch.device
) -> pd.DataFrame:
    """Save the requested test error, parameter, and FLOPs comparison.

    Args:
        b1: Completed B1 metrics.
        b2: Completed B2 metrics.
        b3: Completed B3 metrics.
        device: Device for standardized FLOPs shape-propagation passes.

    Returns:
        Three-row comparison table for B1, B2, and B3.

    Side Effects:
        Writes ``outputs/part_b/model_comparison.csv``.
    """
    results = []
    metrics_by_model = {MODEL_NAMES[0]: b1, MODEL_NAMES[1]: b2, MODEL_NAMES[2]: b3}
    for model_name in MODEL_NAMES:
        if model_name in metrics_by_model:
            run = metrics_by_model[model_name]
            checkpoint_path = OUTPUT_DIR / model_name[:2].lower() / "selected_model.pt"
            model = load_checkpoint_model(model_name, checkpoint_path, device)
            if "trainable_parameters" not in run:
                run["trainable_parameters"] = count_parameters(model)
            if "flops_per_64x64_image" not in run:
                run["flops_per_64x64_image"] = count_flops(model, device)
            results.append(
                {
                    "model": model_name,
                    "test_error": run["test"]["classification_error"],
                    "test_accuracy": run["test"]["accuracy"],
                    "test_sensitivity": run["test"]["sensitivity"],
                    "test_specificity": run["test"]["specificity"],
                    "trainable_parameters": run["trainable_parameters"],
                    "flops_per_64x64_image": run["flops_per_64x64_image"],
                }
            )
    comparison = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    return comparison


def run_all(data: dict, device: torch.device) -> None:
    """Run B1 through B4 in assignment order and save the final comparison.

    Args:
        data: Preprocessed image partitions.
        device: Device used for all Part B work.

    Returns:
        None.

    Side Effects:
        Trains B1-B3, evaluates their selected checkpoints, and runs Grad-CAM.
    """
    b1 = run_b1(data, device)
    b2 = run_b2(data, device, b1)
    b3 = run_b3(data, device, b1, b2)
    run_b4(data, device)
    comparison = save_comparison_table(b1, b2, b3, device)
    print("B1-B3 test comparison:")
    print(comparison.to_string(index=False))


def main(part: str = "all") -> None:
    """Load the supplied partitions and run one or all sequential Part B stages.

    Args:
        part: ``B1``, ``B2``, ``B3``, ``B4``, or ``all``. Later stages load
            selected metrics and checkpoints from prior stages.

    Returns:
        None.

    Side Effects:
        Creates per-part output folders and writes checkpoints and metrics.
    """
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(DATA_PATH)
    split_counts = {
        split: {
            "rows": len(data[split]),
            "normal": int((data[f"{split}_labels"] == 0).sum()),
            "pneumonia": int((data[f"{split}_labels"] == 1).sum()),
        }
        for split in ("train", "val", "test")
    }
    (OUTPUT_DIR / "dataset_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "positive_label": PNEUMONIA_LABEL,
                "train_pixel_mean": data["train_mean"],
                "train_pixel_std": data["train_std"],
                "split_counts": split_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    set_seed(SEED)
    if part == "all":
        run_all(data, device)
        return
    if part == "B1":
        run_b1(data, device)
        return
    b1_metrics = json.loads((OUTPUT_DIR / "b1" / "metrics.json").read_text())
    if part == "B2":
        run_b2(data, device, b1_metrics)
        return
    b2_metrics = json.loads((OUTPUT_DIR / "b2" / "metrics.json").read_text())
    if part == "B3":
        b3 = run_b3(data, device, b1_metrics, b2_metrics)
        comparison = save_comparison_table(b1_metrics, b2_metrics, b3, device)
        print(comparison.to_string(index=False))
        return
    if part == "B4":
        run_b4(data, device)
        return
    raise ValueError("part must be one of B1, B2, B3, B4, or all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", choices=("all", "B1", "B2", "B3", "B4"), default="all"
    )
    main(parser.parse_args().part)
