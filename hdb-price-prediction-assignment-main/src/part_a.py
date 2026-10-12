import copy
import json
import os
import random
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from ray import tune
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import prepare_data
from model import PriceModel

DATA_PATH = Path("hdb_price_prediction.csv")
OUTPUT_DIR = Path("outputs/part_a")
A2_SEEDS = (11, 42, 73, 101, 2026)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible run.

    Args:
        seed: Seed value shared by the model initialization and training run.

    Returns:
        None.

    Side Effects:
        Enables deterministic PyTorch algorithms where supported.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def trainable_parameter_count(model: nn.Module) -> int:
    """Count parameters that participate in gradient-based training.

    Args:
        model: PyTorch model whose trainable parameters are counted.

    Returns:
        Total number of elements in parameters with gradients enabled.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@torch.no_grad()
def calculate_metrics(model: PriceModel, dataset, batch_size: int) -> dict[str, float]:
    """Calculate RMSE and R2 in Singapore dollars over a complete dataset.

    Args:
        model: Trained price model.
        dataset: Dataset containing category IDs, scaled features, and prices.
        batch_size: Maximum number of rows evaluated in one batch.

    Returns:
        Dictionary containing whole-dataset ``rmse`` and ``r2`` values.
    """
    model.eval()
    predictions = []
    targets = []
    for categorical, continuous, target in DataLoader(dataset, batch_size=batch_size):
        predictions.append(model.predict_price(categorical, continuous))
        targets.append(target)

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    residual_sum = (prediction - target).square().sum().item()
    total_sum = (target - target.mean()).square().sum().item()
    return {
        "rmse": (residual_sum / len(target)) ** 0.5,
        "r2": 1.0 - residual_sum / total_sum,
    }


def train_one_epoch(
    model: PriceModel,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
) -> float:
    """Train one epoch and return the epoch's RMSE in dollars.

    Args:
        model: Model updated in place.
        train_loader: Shuffled batches from the fitting partition.
        optimiser: Optimizer updated from normalized-target MSE loss.

    Returns:
        RMSE accumulated from predictions made during the epoch, in dollars.
    """
    model.train()
    loss_function = nn.MSELoss()
    squared_error = 0.0
    row_count = 0

    for categorical, continuous, target in train_loader:
        optimiser.zero_grad()
        standard_prediction = model(categorical, continuous)
        standard_target = (target - model.target_mean) / model.target_std
        loss = loss_function(standard_prediction, standard_target)
        loss.backward()
        optimiser.step()

        prediction = (
            standard_prediction.detach() * model.target_std + model.target_mean
        )
        squared_error += (prediction - target).square().sum().item()
        row_count += len(target)

    return (squared_error / row_count) ** 0.5


def fit_with_early_stopping(
    model: PriceModel,
    train_data,
    validation_data,
    config: Config,
    report_to_tune: bool = True,
) -> tuple[int, float, list[dict[str, float]]]:
    """Fit a model using validation RMSE and the assignment stopping rule.

    Args:
        model: Model to train; its best validation checkpoint is restored.
        train_data: Training partition for gradient updates.
        validation_data: Validation partition used only for model selection.
        config: Learning rate, batch size, epoch limit, and patience settings.
        report_to_tune: Whether to report epoch metrics to an active Ray trial.
    Returns:
        Selected epoch, validation RMSE in dollars, and per-epoch metrics.

    Side Effects:
        Updates ``model`` to its best validation checkpoint. When enabled,
        reports each epoch's metrics to the Ray Tune function API.
    """
    generator = torch.Generator().manual_seed(config.seed + 10_000)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_validation_rmse = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_rmse = train_one_epoch(model, train_loader, optimiser)
        validation_rmse = calculate_metrics(
            model, validation_data, config.batch_size
        )["rmse"]

        if (
            best_weights is None
            or validation_rmse
            < best_validation_rmse - config.early_stopping_threshold
        ):
            best_validation_rmse = validation_rmse
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_metrics = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "validation_rmse": validation_rmse,
            "best_validation_rmse": best_validation_rmse,
            "best_epoch": best_epoch,
        }
        history.append(epoch_metrics)
        if report_to_tune:
            tune.report(epoch_metrics)

        if epochs_without_improvement == config.early_stopping_patience:
            break

    model.load_state_dict(best_weights)
    return best_epoch, best_validation_rmse, history


def train_ray_trial(trial_config: dict) -> None:
    """Run one width/embedding-width trial for Ray Tune.

    Args:
        trial_config: Ray trial values, including model widths, seed, and CSV path.

    Returns:
        None; epoch metrics are reported to Ray Tune.

    Side Effects:
        Reads the HDB CSV, trains one model, and reports validation metrics.
    """
    base_config = Config()
    embedding_dim = int(trial_config["embedding_dim"])
    config = replace(
        base_config,
        seed=int(trial_config["seed"]),
        hidden_width=int(trial_config["hidden_width"]),
        embedding_dims=(embedding_dim,) * len(base_config.categorical_features),
    )
    data = prepare_data(trial_config["data_path"], config)
    set_seed(config.seed)
    model = PriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
    )
    fit_with_early_stopping(model, data["train"], data["validation"], config)


def run_a1_grid_search(
    base_config: Config, output_dir: Path
) -> tuple[dict[str, int], int, float, pd.DataFrame]:
    """Run the required Ray Tune grid and save its trial summary.

    Args:
        base_config: Baseline configuration for all non-searched settings.
        output_dir: Directory for grid-search outputs and Ray's local results.

    Returns:
        Selected width/embedding width, selected epoch, validation RMSE, the
        selected trial's epoch history.

    Side Effects:
        Starts or restores Ray Tune trials and writes ``grid_search_results.csv``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    ray_results = (output_dir / "ray_results").resolve()
    experiment_path = ray_results / "a1_grid_search"
    if experiment_path.exists() and any(
        experiment_path.glob("experiment_state-*.json")
    ):
        tuner = tune.Tuner.restore(
            str(experiment_path), trainable=train_ray_trial
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(train_ray_trial, {"cpu": 1}),
            param_space={
                "hidden_width": tune.grid_search([16, 32, 64, 128, 256]),
                "embedding_dim": tune.grid_search([2, 4, 8, 12, 16]),
                "seed": base_config.seed,
                "data_path": str(DATA_PATH.resolve()),
            },
            tune_config=tune.TuneConfig(
                metric="best_validation_rmse",
                mode="min",
                max_concurrent_trials=1,
            ),
            run_config=tune.RunConfig(
                name="a1_grid_search",
                storage_path=str(ray_results),
                callbacks=[CSVLoggerCallback(), JsonLoggerCallback()],
            ),
        )
    results = tuner.fit()
    best_result = results.get_best_result(
        metric="best_validation_rmse", mode="min"
    )

    trial_rows = [
        {
            "hidden_width": int(result.config["hidden_width"]),
            "embedding_dim": int(result.config["embedding_dim"]),
            "best_epoch": int(result.metrics["best_epoch"]),
            "best_validation_rmse": float(result.metrics["best_validation_rmse"]),
        }
        for result in results
    ]
    trial_table = pd.DataFrame(trial_rows).sort_values(
        ["best_validation_rmse", "hidden_width", "embedding_dim"]
    )
    trial_table.to_csv(output_dir / "grid_search_results.csv", index=False)

    history = best_result.metrics_dataframe
    history_columns = [
        "epoch",
        "train_rmse",
        "validation_rmse",
        "best_validation_rmse",
    ]
    if all(column in history.columns for column in history_columns):
        history = history[history_columns].copy()
    else:
        replay_config = replace(
            base_config,
            hidden_width=int(best_result.config["hidden_width"]),
            embedding_dims=(int(best_result.config["embedding_dim"]),)
            * len(base_config.categorical_features),
        )
        replay_data = prepare_data(DATA_PATH, replay_config)
        set_seed(replay_config.seed)
        replay_model = PriceModel(
            replay_data["cardinalities"],
            replay_data["target_mean"],
            replay_data["target_std"],
            replay_config,
        )
        _, _, replay_history = fit_with_early_stopping(
            replay_model,
            replay_data["train"],
            replay_data["validation"],
            replay_config,
            report_to_tune=False,
        )
        history = pd.DataFrame(replay_history)[history_columns]
    selected_config = {
        "hidden_width": int(best_result.config["hidden_width"]),
        "embedding_dim": int(best_result.config["embedding_dim"]),
    }
    return (
        selected_config,
        int(best_result.metrics["best_epoch"]),
        float(best_result.metrics["best_validation_rmse"]),
        history,
    )


def save_validation_plot(history: pd.DataFrame, output_path: Path) -> None:
    """Save the selected A1 trial's training and validation RMSE curves.

    Args:
        history: Per-epoch metrics for the selected Ray trial.
        output_path: Destination PNG path.

    Returns:
        None.

    Side Effects:
        Writes a Matplotlib figure to ``output_path``.
    """
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history["epoch"], history["train_rmse"], label="Training RMSE")
    axis.plot(
        history["epoch"], history["validation_rmse"], label="Validation RMSE"
    )
    axis.set(
        xlabel="Epoch",
        ylabel="RMSE (Singapore dollars)",
        title="A1 selected configuration",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_final_refit_plot(curve: pd.DataFrame, output_path: Path) -> None:
    """Save the A1 train/test RMSE curve from the combined-data refit.

    Args:
        curve: Per-epoch RMSE values from the combined-data refit.
        output_path: Destination PNG path.

    Returns:
        None.

    Side Effects:
        Writes a Matplotlib figure to ``output_path``.
    """
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(curve["epoch"], curve["train_rmse"], label="Training RMSE")
    axis.plot(curve["epoch"], curve["test_rmse"], label="Testing RMSE")
    axis.set(
        xlabel="Epoch",
        ylabel="RMSE (Singapore dollars)",
        title="A1 combined-data refit",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_architecture_diagram(
    output_path: Path,
    config: Config,
    input_width: int,
    default_parameter_count: int,
    selected_parameter_count: int,
) -> None:
    """Write an A1 Mermaid block diagram and parameter-count notes.

    Args:
        output_path: Destination Markdown path.
        config: Selected A1 model configuration.
        input_width: Concatenated input width for the selected model.
        default_parameter_count: Count for the supplied baseline defaults.
        selected_parameter_count: Count for the selected model on the
            model-selection vocabulary.

    Returns:
        None.

    Side Effects:
        Writes an architecture diagram to ``output_path``.
    """
    embedding_dim = config.embedding_dims[0]
    diagram = f"""# A1 baseline architecture

```mermaid
flowchart LR
    Cat[4 categorical features] --> Emb[4 embedding tables, width {embedding_dim} each]
    Emb --> Join[Concatenate]
    Num[6 scaled continuous features] --> Join
    Join --> Dense[Linear {input_width} to {config.hidden_width}]
    Dense --> Norm[LayerNorm {config.hidden_width}]
    Norm --> Relu[ReLU]
    Relu --> Output[Linear {config.hidden_width} to 1]
    Output --> Price[De-normalize to Singapore dollars]
```

- Supplied default baseline parameters (embedding width 8, hidden width 64): {default_parameter_count:,}
- Selected A1 model parameters on the model-selection vocabulary: {selected_parameter_count:,}
"""
    output_path.write_text(diagram, encoding="utf-8")


def fit_combined_refit(
    model: PriceModel,
    train_data,
    test_data,
    config: Config,
    epochs: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Fit for the selected A1 epoch count and record its required test curve.

    Args:
        model: Fresh model initialized with combined-data scaling statistics.
        train_data: 2017-2021 rows used for fitting.
        test_data: 2022 rows, evaluated only for the requested curve/final score.
        config: Selected A1 hyperparameters and batch/optimizer settings.
        epochs: Best epoch selected during train/validation model selection.

    Returns:
        Per-epoch training/testing RMSE curve and final test RMSE/R2.

    Side Effects:
        Updates ``model`` for exactly ``epochs``; test metrics do not control
        training, stopping, or checkpoint selection.
    """
    generator = torch.Generator().manual_seed(config.seed + 10_000)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    curve = []
    test_metrics = {}
    for epoch in range(1, epochs + 1):
        train_rmse = train_one_epoch(model, train_loader, optimiser)
        test_metrics = calculate_metrics(model, test_data, config.batch_size)
        curve.append(
            {
                "epoch": epoch,
                "train_rmse": train_rmse,
                "test_rmse": test_metrics["rmse"],
            }
        )
    return curve, test_metrics


def train_fixed_epochs(
    model: PriceModel,
    train_data,
    config: Config,
    epochs: int,
) -> float:
    """Train an A2 model for the A1-selected number of epochs.

    Args:
        model: Model updated in place.
        train_data: Combined 2017-2021 fitting rows.
        config: Selected A1 hyperparameters and run seed.
        epochs: Number of epochs selected by A1 early stopping.

    Returns:
        Training RMSE in dollars from the final epoch.
    """
    generator = torch.Generator().manual_seed(config.seed + 10_000)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_rmse = float("nan")
    for _ in range(epochs):
        train_rmse = train_one_epoch(model, train_loader, optimiser)
    return train_rmse


def run_a2_ablations(
    config: Config,
    data: dict,
    epochs: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the baseline and both A2 ablations on five paired seeds.

    Args:
        config: Selected A1 model and optimization settings.
        data: Preprocessed combined-train/validation and test datasets.
        epochs: A1-selected epoch count used for all A2 runs.
        output_dir: Directory for per-seed and summary metrics.

    Returns:
        Per-seed metric dataframe and mean/sample-standard-deviation summary.

    Side Effects:
        Trains 15 models and evaluates each once on the test set. Writes CSV
        result tables and RMSE/R2 comparison figures.
    """
    variants = (
        ("baseline", True, "relu"),
        ("no_embeddings", False, "relu"),
        ("sigmoid", True, "sigmoid"),
    )
    rows = []
    for seed in A2_SEEDS:
        for name, use_embeddings, hidden_activation in variants:
            run_config = replace(config, seed=seed)
            set_seed(seed)
            model = PriceModel(
                data["cardinalities"],
                data["target_mean"],
                data["target_std"],
                run_config,
                use_embeddings=use_embeddings,
                hidden_activation=hidden_activation,
            )
            train_rmse = train_fixed_epochs(
                model, data["train"], run_config, epochs
            )
            test_metrics = calculate_metrics(
                model, data["test"], run_config.batch_size
            )
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "epochs": epochs,
                    "trainable_parameters": trainable_parameter_count(model),
                    "train_rmse": train_rmse,
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                }
            )

    results = pd.DataFrame(rows)
    summary_rows = []
    for name, group in results.groupby("model", sort=False):
        summary_rows.append(
            {
                "model": name,
                "test_rmse_mean": statistics.mean(group["test_rmse"]),
                "test_rmse_sample_std": statistics.stdev(group["test_rmse"]),
                "test_r2_mean": statistics.mean(group["test_r2"]),
                "test_r2_sample_std": statistics.stdev(group["test_r2"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_ablation_metric(
        results,
        summary,
        "test_rmse",
        "RMSE (Singapore dollars)",
        output_dir / "rmse_comparison.png",
    )
    plot_ablation_metric(
        results,
        summary,
        "test_r2",
        "R2",
        output_dir / "r2_comparison.png",
    )
    return results, summary


def plot_ablation_metric(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot five individual A2 scores and mean/sample-SD per model.

    Args:
        results: Per-seed test results with ``model`` and ``metric`` columns.
        summary: Mean and sample-standard-deviation table.
        metric: Metric column, either ``test_rmse`` or ``test_r2``.
        ylabel: Human-readable y-axis label.
        output_path: Destination PNG path.

    Returns:
        None.

    Side Effects:
        Writes a Matplotlib figure to ``output_path``.
    """
    names = summary["model"].tolist()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index, row in summary.iterrows():
        value_key = "test_rmse_mean" if metric == "test_rmse" else "test_r2_mean"
        std_key = (
            "test_rmse_sample_std"
            if metric == "test_rmse"
            else "test_r2_sample_std"
        )
        axis.errorbar(
            index,
            row[value_key],
            yerr=row[std_key],
            fmt="o",
            capsize=5,
            color="tab:blue",
            label="Mean +/- sample SD" if index == 0 else None,
        )
        seed_values = results.loc[results["model"] == row["model"], metric].tolist()
        offsets = np.linspace(-0.12, 0.12, len(seed_values))
        axis.scatter(
            index + offsets,
            seed_values,
            color="tab:orange",
            s=28,
            label="Individual seeds" if index == 0 else None,
        )
    axis.set_xticks(range(len(names)), names)
    axis.set_ylabel(ylabel)
    axis.set_title(f"A2 test {ylabel} across five seeds")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run A1 search/refit followed by all A2 five-seed ablations.

    Returns:
        None.

    Side Effects:
        Creates ``outputs/part_a/a1`` and ``outputs/part_a/a2`` artifacts,
        including Ray trial results, model weights, metric tables, and plots.
    """
    base_config = Config()
    a1_output = OUTPUT_DIR / "a1"
    a2_output = OUTPUT_DIR / "a2"
    a1_output.mkdir(parents=True, exist_ok=True)

    model_selection_data = prepare_data(DATA_PATH, base_config)
    (
        selected,
        best_epoch,
        best_validation_rmse,
        selected_history,
    ) = run_a1_grid_search(base_config, a1_output)
    selected_config = replace(
        base_config,
        hidden_width=selected["hidden_width"],
        embedding_dims=(
            selected["embedding_dim"],
        )
        * len(base_config.categorical_features),
    )

    default_model = PriceModel(
        model_selection_data["cardinalities"],
        model_selection_data["target_mean"],
        model_selection_data["target_std"],
        base_config,
    )
    selected_search_model = PriceModel(
        model_selection_data["cardinalities"],
        model_selection_data["target_mean"],
        model_selection_data["target_std"],
        selected_config,
    )
    save_validation_plot(selected_history, a1_output / "learning_curves.png")
    save_architecture_diagram(
        a1_output / "architecture.md",
        selected_config,
        sum(selected_config.embedding_dims)
        + len(selected_config.continuous_features),
        trainable_parameter_count(default_model),
        trainable_parameter_count(selected_search_model),
    )

    combined_data = prepare_data(
        DATA_PATH, selected_config, include_validation_in_train=True
    )
    set_seed(base_config.seed)
    final_model = PriceModel(
        combined_data["cardinalities"],
        combined_data["target_mean"],
        combined_data["target_std"],
        selected_config,
    )
    final_curve, final_test_metrics = fit_combined_refit(
        final_model,
        combined_data["train"],
        combined_data["test"],
        selected_config,
        best_epoch,
    )
    final_curve_frame = pd.DataFrame(final_curve)
    final_curve_frame.to_csv(a1_output / "train_test_curve.csv", index=False)
    save_final_refit_plot(
        final_curve_frame, a1_output / "train_test_rmse.png"
    )
    torch.save(final_model.state_dict(), a1_output / "selected_model.pt")

    a1_metrics = {
        "selected_hidden_width": selected["hidden_width"],
        "selected_embedding_dim": selected["embedding_dim"],
        "best_epoch": best_epoch,
        "best_validation_rmse_dollars": best_validation_rmse,
        "seed": base_config.seed,
        "learning_rate": selected_config.learning_rate,
        "batch_size": selected_config.batch_size,
        "maximum_epochs": selected_config.max_epochs,
        "early_stopping_patience": selected_config.early_stopping_patience,
        "early_stopping_minimum_improvement": selected_config.early_stopping_threshold,
        "train_years": list(selected_config.train_years),
        "validation_year": selected_config.validation_year,
        "test_year": selected_config.test_year,
        "a2_seeds": list(A2_SEEDS),
        "default_baseline_parameters": trainable_parameter_count(default_model),
        "selected_model_parameters_train_vocabulary": trainable_parameter_count(
            selected_search_model
        ),
        "selected_model_parameters_refit_vocabulary": trainable_parameter_count(
            final_model
        ),
        "test_rmse_dollars": final_test_metrics["rmse"],
        "test_r2": final_test_metrics["r2"],
        "target_mean": combined_data["target_mean"],
        "target_std": combined_data["target_std"],
        "continuous_mean": combined_data["continuous_mean"].tolist(),
        "continuous_std": combined_data["continuous_std"].tolist(),
    }
    (a1_output / "metrics.json").write_text(
        json.dumps(a1_metrics, indent=2), encoding="utf-8"
    )

    _, a2_summary = run_a2_ablations(
        selected_config, combined_data, best_epoch, a2_output
    )
    print("A1 selected configuration:", selected)
    print("A1 selected epoch:", best_epoch)
    print("A1 test metrics:", final_test_metrics)
    print("A2 five-seed summary:")
    print(a2_summary.to_string(index=False))


if __name__ == "__main__":
    main()
