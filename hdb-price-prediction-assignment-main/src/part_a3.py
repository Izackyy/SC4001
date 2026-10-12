import json
import statistics
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import pandas as pd
import torch

from config import Config
from data import prepare_data
from model import PriceModel
from part_a import (
    A2_SEEDS,
    calculate_metrics,
    fit_with_early_stopping,
    set_seed,
    train_fixed_epochs,
    trainable_parameter_count,
)

DATA_PATH = Path("hdb_price_prediction.csv")
A1_OUTPUT = Path("outputs/part_a/a1")
A2_OUTPUT = Path("outputs/part_a/a2")
OUTPUT_DIR = Path("outputs/part_a/a3")
LAMBDA_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)


def load_selected_config(base_config: Config) -> tuple[Config, int, dict]:
    """Load A1's selected widths and epoch count from its metrics file.

    Args:
        base_config: Default assignment configuration for fixed settings.

    Returns:
        A1-selected model configuration, refit epoch count, and A1 metrics.
    """
    a1_metrics = json.loads((A1_OUTPUT / "metrics.json").read_text())
    selected_config = replace(
        base_config,
        seed=int(a1_metrics["seed"]),
        hidden_width=int(a1_metrics["selected_hidden_width"]),
        embedding_dims=(int(a1_metrics["selected_embedding_dim"]),)
        * len(base_config.categorical_features),
    )
    return selected_config, int(a1_metrics["best_epoch"]), a1_metrics


def search_lambda(
    config: Config, data: dict
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Select the wide-branch weight using the validation partition.

    Args:
        config: A1-selected widths, optimizer settings, and seed.
        data: Training-only preprocessed train, validation, and test data.

    Returns:
        Best lambda, per-lambda summary, and per-epoch search history.

    Side Effects:
        Trains one early-stopped model for each candidate lambda.
    """
    summary_rows = []
    history_rows = []
    for wide_lambda in LAMBDA_VALUES:
        set_seed(config.seed)
        model = PriceModel(
            data["cardinalities"],
            data["target_mean"],
            data["target_std"],
            config,
            wide_lambda=wide_lambda,
        )
        best_epoch, best_validation_rmse, history = fit_with_early_stopping(
            model,
            data["train"],
            data["validation"],
            config,
            report_to_tune=False,
        )
        summary_rows.append(
            {
                "wide_lambda": wide_lambda,
                "best_epoch": best_epoch,
                "best_validation_rmse": best_validation_rmse,
                "trainable_parameters": trainable_parameter_count(model),
            }
        )
        history_rows.extend(
            {"wide_lambda": wide_lambda, **epoch_metrics}
            for epoch_metrics in history
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["best_validation_rmse", "wide_lambda"]
    )
    return (
        float(summary.iloc[0]["wide_lambda"]),
        summary,
        pd.DataFrame(history_rows),
    )


def summarize_runs(
    name: str, rmse_values: list[float], r2_values: list[float]
) -> dict[str, float | int | str | None]:
    """Summarize RMSE and R2 with sample standard deviations when available.

    Args:
        name: Display name for this comparison row.
        rmse_values: Test RMSE values in Singapore dollars.
        r2_values: Test R2 values.

    Returns:
        Summary row with sample count, means, and sample standard deviations.
    """
    return {
        "model": name,
        "n_seeds": len(rmse_values),
        "test_rmse_mean": statistics.mean(rmse_values),
        "test_rmse_sample_std": (
            statistics.stdev(rmse_values) if len(rmse_values) > 1 else None
        ),
        "test_r2_mean": statistics.mean(r2_values),
        "test_r2_sample_std": (
            statistics.stdev(r2_values) if len(r2_values) > 1 else None
        ),
    }


def save_lambda_plot(history: pd.DataFrame, output_path: Path) -> None:
    """Save validation RMSE curves for all A3 lambda candidates.

    Args:
        history: Per-epoch training and validation RMSE by lambda.
        output_path: PNG destination for the plot.

    Returns:
        None.

    Side Effects:
        Writes a labeled Matplotlib figure to ``output_path``.
    """
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for wide_lambda, rows in history.groupby("wide_lambda", sort=True):
        axis.plot(
            rows["epoch"],
            rows["validation_rmse"],
            marker="o",
            label=f"lambda={wide_lambda:g}",
        )
    axis.set(
        xlabel="Epoch",
        ylabel="Validation RMSE (Singapore dollars)",
        title="A3 lambda search",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_comparison_plot(
    summary: pd.DataFrame, metric: str, ylabel: str, output_path: Path
) -> None:
    """Plot A1, A2, and A3 test-metric means with seed variation.

    Args:
        summary: Comparison summary containing means and sample deviations.
        metric: Metric column to plot, either RMSE or R2.
        ylabel: Axis label describing the metric and its units.
        output_path: PNG destination for the plot.

    Returns:
        None.

    Side Effects:
        Writes a labeled Matplotlib figure to ``output_path``.
    """
    std_column = f"{metric}_sample_std"
    errors = summary[std_column].fillna(0).tolist()
    labels = [
        "A1 baseline\n(seed 42)",
        "A2 baseline\n(five seeds)",
        "A3 wide-and-deep\n(five seeds)",
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        labels,
        summary[f"{metric}_mean"],
        yerr=errors,
        capsize=4,
        color=["tab:gray", "tab:blue", "tab:orange"],
    )
    axis.set_ylabel(ylabel)
    axis.set_title(f"Part A test {ylabel}")
    axis.tick_params(axis="x", labelsize=10)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run A3 lambda selection, train five refits, and save comparisons.

    Returns:
        None.

    Side Effects:
        Trains ten models and writes A3 search, seed, comparison, and plot files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config, refit_epochs, a1_metrics = load_selected_config(Config())
    search_data = prepare_data(DATA_PATH, config)
    selected_lambda, lambda_summary, lambda_history = search_lambda(
        config, search_data
    )
    lambda_summary.to_csv(OUTPUT_DIR / "lambda_search_results.csv", index=False)
    lambda_history.to_csv(OUTPUT_DIR / "lambda_search_history.csv", index=False)
    save_lambda_plot(
        lambda_history, OUTPUT_DIR / "lambda_validation_curves.png"
    )

    combined_data = prepare_data(
        DATA_PATH, config, include_validation_in_train=True
    )
    seed_rows = []
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed in A2_SEEDS:
        run_config = replace(config, seed=seed)
        set_seed(seed)
        model = PriceModel(
            combined_data["cardinalities"],
            combined_data["target_mean"],
            combined_data["target_std"],
            run_config,
            wide_lambda=selected_lambda,
        )
        train_rmse = train_fixed_epochs(
            model, combined_data["train"], run_config, refit_epochs
        )
        test_metrics = calculate_metrics(
            model, combined_data["test"], run_config.batch_size
        )
        torch.save(model.state_dict(), model_dir / f"seed_{seed}.pt")
        seed_rows.append(
            {
                "seed": seed,
                "wide_lambda": selected_lambda,
                "epochs": refit_epochs,
                "trainable_parameters": trainable_parameter_count(model),
                "train_rmse": train_rmse,
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        pd.DataFrame(seed_rows).to_csv(
            OUTPUT_DIR / "seed_metrics.csv", index=False
        )

    a3_runs = pd.DataFrame(seed_rows)
    a2_baseline = pd.read_csv(A2_OUTPUT / "seed_metrics.csv")
    a2_baseline = a2_baseline.loc[a2_baseline["model"] == "baseline"].copy()
    paired = a2_baseline[["seed", "test_rmse", "test_r2"]].merge(
        a3_runs[["seed", "test_rmse", "test_r2"]],
        on="seed",
        suffixes=("_a2_baseline", "_a3_wide_deep"),
        validate="one_to_one",
    )
    paired["rmse_reduction_dollars"] = (
        paired["test_rmse_a2_baseline"] - paired["test_rmse_a3_wide_deep"]
    )
    paired["r2_gain"] = paired["test_r2_a3_wide_deep"] - paired[
        "test_r2_a2_baseline"
    ]
    paired.to_csv(OUTPUT_DIR / "paired_seed_comparison.csv", index=False)

    comparison = pd.DataFrame(
        [
            summarize_runs(
                "A1 baseline (seed 42)",
                [float(a1_metrics["test_rmse_dollars"])],
                [float(a1_metrics["test_r2"])],
            ),
            summarize_runs(
                "A2 baseline (five seeds)",
                a2_baseline["test_rmse"].tolist(),
                a2_baseline["test_r2"].tolist(),
            ),
            summarize_runs(
                "A3 wide-and-deep (five seeds)",
                a3_runs["test_rmse"].tolist(),
                a3_runs["test_r2"].tolist(),
            ),
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
    save_comparison_plot(
        comparison,
        "test_rmse",
        "RMSE (Singapore dollars)",
        OUTPUT_DIR / "test_rmse_comparison.png",
    )
    save_comparison_plot(
        comparison,
        "test_r2",
        "R2",
        OUTPUT_DIR / "test_r2_comparison.png",
    )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "lambda_candidates": list(LAMBDA_VALUES),
                "selected_lambda": selected_lambda,
                "lambda_selection_best_epoch": int(
                    lambda_summary.iloc[0]["best_epoch"]
                ),
                "refit_epochs_from_a1": refit_epochs,
                "seed_values": list(A2_SEEDS),
                "a1_selected_hidden_width": config.hidden_width,
                "a1_selected_embedding_dim": config.embedding_dims[0],
                "a3_test_rmse_mean": float(a3_runs["test_rmse"].mean()),
                "a3_test_rmse_sample_std": float(
                    a3_runs["test_rmse"].std(ddof=1)
                ),
                "a3_test_r2_mean": float(a3_runs["test_r2"].mean()),
                "a3_test_r2_sample_std": float(
                    a3_runs["test_r2"].std(ddof=1)
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("A3 selected lambda:", selected_lambda)
    print("A3 comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
