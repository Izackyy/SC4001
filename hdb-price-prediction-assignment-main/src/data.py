import pandas as pd
import torch
from torch.utils.data import TensorDataset

from config import Config


def make_dataset(rows, category_maps, means, stds, config):
    """Convert transaction rows to categorical, scaled, and target tensors.

    Args:
        rows: Transaction records to convert.
        category_maps: Training-fitted mapping for each categorical feature.
        means: Per-feature means used to scale continuous inputs.
        stds: Per-feature standard deviations used to scale continuous inputs.
        config: Assignment feature configuration.

    Returns:
        A TensorDataset of categorical inputs, scaled continuous inputs, and
        resale prices in dollars.
    """
    categorical = torch.tensor(
        [
            [
                category_maps[name].get(row[name], 0)
                for name in config.categorical_features
            ]
            for row in rows
        ],
        dtype=torch.long,
    )
    continuous = torch.tensor(
        [[float(row[name]) for name in config.continuous_features] for row in rows],
        dtype=torch.float32,
    )
    continuous = (continuous - means) / stds

    target = torch.tensor([float(row["resale_price"]) for row in rows])
    return TensorDataset(categorical, continuous, target)


def prepare_data(
    csv_path, config: Config, include_validation_in_train: bool = False
):
    """Split the CSV and fit preprocessing on the allowed training rows.

    Args:
        csv_path: Path to the HDB resale-price CSV.
        config: Assignment split and feature configuration.
        include_validation_in_train: If true, refit preprocessing on 2017-2021
            and return those rows together as the training dataset.

    Returns:
        A dictionary containing train, validation, and test datasets, category
        cardinalities, and the fitted input and target scaling statistics.
        ``validation`` is ``None`` when its rows are included in ``train``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    data = pd.read_csv(csv_path)
    rows = data.to_dict(orient="records")

    train_rows = [row for row in rows if int(row["year"]) in config.train_years]
    validation_rows = [
        row for row in rows if int(row["year"]) == config.validation_year
    ]
    test_rows = [row for row in rows if int(row["year"]) == config.test_year]

    fit_rows = train_rows + validation_rows if include_validation_in_train else train_rows

    # Fit categorical encoders on training data only. ID 0 means "unseen category".
    category_maps = {}
    for name in config.categorical_features:
        values = sorted({row[name] for row in fit_rows})
        category_maps[name] = {value: index + 1 for index, value in enumerate(values)}

    # Fit scaling statistics only on rows used for fitting.
    train_continuous = torch.tensor(
        [
            [float(row[name]) for name in config.continuous_features]
            for row in fit_rows
        ]
    )
    means = train_continuous.mean(dim=0)
    stds = train_continuous.std(dim=0).clamp_min(1e-8)

    train_target = torch.tensor([float(row["resale_price"]) for row in fit_rows])
    target_mean = train_target.mean().item()
    target_std = train_target.std(correction=0).clamp_min(1e-8).item()

    return {
        "train": make_dataset(fit_rows, category_maps, means, stds, config),
        "validation": (
            None
            if include_validation_in_train
            else make_dataset(validation_rows, category_maps, means, stds, config)
        ),
        "test": make_dataset(test_rows, category_maps, means, stds, config),
        "cardinalities": [
            len(category_maps[name]) + 1 for name in config.categorical_features
        ],
        "continuous_mean": means,
        "continuous_std": stds,
        "target_mean": target_mean,
        "target_std": target_std,
    }
