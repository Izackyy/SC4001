from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    train_years: tuple[int, ...] = (2017, 2018, 2019, 2020)
    validation_year: int = 2021
    test_year: int = 2022

    categorical_features: tuple[str, ...] = (
        "month",
        "town",
        "flat_model_type",
        "storey_range",
    )
    embedding_dims: tuple[int, ...] = (8, 8, 8, 8)
    continuous_features: tuple[str, ...] = (
        "dist_to_nearest_stn",
        "dist_to_dhoby",
        "degree_centrality",
        "eigenvector_centrality",
        "remaining_lease_years",
        "floor_area_sqm",
    )

    hidden_width: int = 64

    learning_rate: float = 1e-2
    batch_size: int = 512
    max_epochs: int = 20
    early_stopping_patience: int = 4
    early_stopping_threshold: float = 1e-4
