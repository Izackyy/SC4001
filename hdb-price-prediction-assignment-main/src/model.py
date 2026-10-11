from typing import Literal

import torch
from torch import nn

from config import Config


class PriceModel(nn.Module):
    """Embedding MLP used for the HDB baseline, A2 variants, and A3 model."""

    def __init__(
        self,
        cardinalities: list[int],
        target_mean: float,
        target_std: float,
        config: Config,
        *,
        use_embeddings: bool = True,
        hidden_activation: Literal["relu", "sigmoid"] = "relu",
        wide_lambda: float | None = None,
    ) -> None:
        """Build the baseline MLP, an A2 variant, or an A3 wide-and-deep model.

        Args:
            cardinalities: Number of rows in each categorical embedding table,
                including the reserved unseen-category row.
            target_mean: Mean used to standardize resale-price targets.
            target_std: Standard deviation used to standardize resale-price
                targets.
            config: Model widths, feature lists, and training configuration.
            use_embeddings: Whether to concatenate categorical embeddings.
            hidden_activation: Either ``"relu"`` for the baseline or
                ``"sigmoid"`` for the A2 activation ablation.
            wide_lambda: If provided, add a linear branch over the same
                concatenated inputs and multiply its output by this value.

        Raises:
            ValueError: If ``hidden_activation`` is not ``"relu"`` or
                ``"sigmoid"``.
        """
        super().__init__()
        if hidden_activation not in {"relu", "sigmoid"}:
            raise ValueError("hidden_activation must be 'relu' or 'sigmoid'")

        self.use_embeddings = use_embeddings
        self.wide_lambda = wide_lambda
        self.embeddings = nn.ModuleList()
        if use_embeddings:
            self.embeddings.extend(
                nn.Embedding(cardinality, width)
                for cardinality, width in zip(cardinalities, config.embedding_dims)
            )

        input_width = len(config.continuous_features)
        if use_embeddings:
            input_width += sum(config.embedding_dims)
        activation = nn.ReLU() if hidden_activation == "relu" else nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            activation,
            nn.Linear(config.hidden_width, 1),
        )
        self.wide_branch = (
            nn.Linear(input_width, 1) if wide_lambda is not None else None
        )

        self.target_mean = target_mean
        self.target_std = target_std

    def forward(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        """Predict standardized resale prices from a batch of features.

        Args:
            categorical: Integer category IDs with shape ``(batch, 4)``.
            continuous: Scaled numeric inputs with shape ``(batch, 6)``.

        Returns:
            Standardized scalar predictions with shape ``(batch,)``.
        """
        if self.use_embeddings:
            embedded = [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.embeddings)
            ]
            features = torch.cat(embedded + [continuous], dim=1)
        else:
            features = continuous
        prediction = self.mlp(features).squeeze(1)
        if self.wide_branch is not None:
            prediction = prediction + self.wide_lambda * self.wide_branch(
                features
            ).squeeze(1)
        return prediction

    def predict_price(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        """Convert standardized model outputs back to Singapore-dollar prices.

        Args:
            categorical: Integer category IDs with shape ``(batch, 4)``.
            continuous: Scaled numeric inputs with shape ``(batch, 6)``.

        Returns:
            Predicted resale prices in Singapore dollars.
        """
        return self(categorical, continuous) * self.target_std + self.target_mean
