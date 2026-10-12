import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from captum.attr import FeatureAblation
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from torch.utils.data import DataLoader

from config import Config
from data import prepare_data
from model import PriceModel

DATA_PATH = Path("hdb_price_prediction.csv")
A1_OUTPUT = Path("outputs/part_a/a1")
A3_OUTPUT = Path("outputs/part_a/a3")
OUTPUT_DIR = Path("outputs/part_a/a4")
ATTRIBUTION_SEED = 42


def load_final_models(
    config: Config,
) -> tuple[dict[str, PriceModel], dict, dict]:
    """Load the paired A1 and A3 checkpoints with matching preprocessing.

    Args:
        config: Default assignment configuration, updated with A1 selections.

    Returns:
        A mapping of model labels to loaded models, the combined-fit dataset
        bundle, and the saved A1/A3 metadata used to build those models.

    Side Effects:
        Reads saved metrics and model checkpoints from the A1 and A3 outputs.
    """
    a1_metrics = json.loads((A1_OUTPUT / "metrics.json").read_text())
    a3_metrics = json.loads((A3_OUTPUT / "metrics.json").read_text())
    if int(a1_metrics["seed"]) != ATTRIBUTION_SEED:
        raise ValueError("A1 checkpoint seed must match the A4 attribution seed.")
    config = replace(
        config,
        seed=ATTRIBUTION_SEED,
        hidden_width=int(a1_metrics["selected_hidden_width"]),
        embedding_dims=(int(a1_metrics["selected_embedding_dim"]),)
        * len(config.categorical_features),
    )
    data = prepare_data(DATA_PATH, config, include_validation_in_train=True)

    a1_model = PriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
    )
    a1_model.load_state_dict(
        torch.load(A1_OUTPUT / "selected_model.pt", map_location="cpu", weights_only=True)
    )

    a3_seed_path = A3_OUTPUT / "models" / f"seed_{ATTRIBUTION_SEED}.pt"
    a3_model = PriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
        wide_lambda=float(a3_metrics["selected_lambda"]),
    )
    a3_model.load_state_dict(
        torch.load(a3_seed_path, map_location="cpu", weights_only=True)
    )

    for model in (a1_model, a3_model):
        model.eval()
    models = {"A1 baseline": a1_model, "A3 wide-and-deep": a3_model}
    metadata = {
        "a1": a1_metrics,
        "a3": a3_metrics,
        "seed": ATTRIBUTION_SEED,
    }
    return models, data, metadata


def attribute_features(
    model: PriceModel,
    test_data,
    config: Config,
) -> torch.Tensor:
    """Compute mean absolute dollar attribution for each of ten inputs.

    Captum replaces each input feature with its baseline: category ID 0 for
    categorical inputs and standardized value 0 for continuous inputs.

    Args:
        model: Loaded A1 or A3 model whose prediction method returns dollars.
        test_data: TensorDataset containing the held-out 2022 rows.
        config: Batch-size and feature-name configuration.

    Returns:
        Ten mean absolute FeatureAblation scores in configured feature order.
    """
    ablator = FeatureAblation(model.predict_price)
    feature_sums = torch.zeros(
        len(config.categorical_features) + len(config.continuous_features)
    )
    row_count = 0
    categorical_feature_ids = torch.arange(
        len(config.categorical_features), dtype=torch.long
    ).unsqueeze(0)
    continuous_feature_ids = torch.arange(
        len(config.categorical_features),
        len(config.categorical_features) + len(config.continuous_features),
        dtype=torch.long,
    ).unsqueeze(0)

    for categorical, continuous, _targets in DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False
    ):
        feature_mask = (
            categorical_feature_ids.expand_as(categorical),
            continuous_feature_ids.expand_as(continuous),
        )
        baselines = (torch.zeros_like(categorical), torch.zeros_like(continuous))
        attributions = ablator.attribute(
            inputs=(categorical, continuous),
            baselines=baselines,
            feature_mask=feature_mask,
        )
        batch_attributions = torch.cat(attributions, dim=1).abs().sum(dim=0)
        feature_sums += batch_attributions
        row_count += len(categorical)

    return feature_sums / row_count


def save_attribution_plot(scores: pd.DataFrame, output_path: Path) -> None:
    """Save descending A1/A3 FeatureAblation scores as dollar bar charts.

    Args:
        scores: Long-form scores with model, feature, and attribution columns.
        output_path: PNG path for the two-panel comparison figure.

    Returns:
        None.

    Side Effects:
        Writes a Matplotlib figure to ``output_path``.
    """
    models = scores["model"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(models), figsize=(14, 6.5), sharex=True)
    for axis, model_name in zip(axes, models):
        model_scores = scores.loc[scores["model"] == model_name].sort_values(
            "mean_absolute_attribution_sgd", ascending=False
        )
        axis.barh(
            model_scores["feature"],
            model_scores["mean_absolute_attribution_sgd"],
            color="tab:blue" if model_name.startswith("A1") else "tab:orange",
        )
        axis.invert_yaxis()
        axis.set_title(model_name)
        axis.set_xlabel("Mean absolute price change (Singapore dollars)")
        axis.xaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("A4 FeatureAblation on 2022 test rows")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Compute and save paired A1/A3 feature-ablation results.

    Returns:
        None.

    Side Effects:
        Writes feature scores, run metadata, and a labeled comparison plot to
        ``outputs/part_a/a4``.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models, data, metadata = load_final_models(Config())
    rows = []
    feature_names = list(Config().categorical_features + Config().continuous_features)
    baselines = [
        "category ID 0 (unseen category)"
        for _ in Config().categorical_features
    ] + [
        "standardized value 0 (combined-train/validation mean)"
        for _ in Config().continuous_features
    ]

    for model_name, model in models.items():
        attributions = attribute_features(model, data["test"], Config())
        rows.extend(
            {
                "model": model_name,
                "seed": metadata["seed"],
                "feature": feature,
                "baseline": baseline,
                "mean_absolute_attribution_sgd": float(score),
            }
            for feature, baseline, score in zip(
                feature_names, baselines, attributions
            )
        )

    scores = pd.DataFrame(rows)
    scores.to_csv(OUTPUT_DIR / "feature_ablation_scores.csv", index=False)
    save_attribution_plot(scores, OUTPUT_DIR / "feature_ablation_comparison.png")
    run_metadata = {
        "test_year": Config().test_year,
        "test_rows": len(data["test"]),
        "seed": metadata["seed"],
        "a1_hidden_width": metadata["a1"]["selected_hidden_width"],
        "a1_embedding_dim": metadata["a1"]["selected_embedding_dim"],
        "a3_lambda": metadata["a3"]["selected_lambda"],
        "target_units": "Singapore dollars",
        "categorical_baseline": "category ID 0 (unseen category)",
        "continuous_baseline": "standardized value 0 (combined-train/validation mean)",
        "feature_order": feature_names,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(scores.sort_values(
        ["model", "mean_absolute_attribution_sgd"], ascending=[True, False]
    ).to_string(index=False))
    print("Wrote A4 FeatureAblation results to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
