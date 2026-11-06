"""Defaults and paths for the Phase 2 DeepDTA baseline."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class Config:
    """One experiment's data selection and model/training hyperparameters."""

    dataset: str = "davis"
    split: str = "split_cold_target"
    seed: int = 4001
    embedding_dim: int = 128
    filters: tuple[int, ...] = (32, 64, 96)
    drug_kernel: int = 4
    protein_kernel: int = 8
    fc_dims: tuple[int, ...] = (1024, 1024, 512)
    dropout: float = 0.1
    protein_encoder: str = "cnn"
    esm_dim: int = 1280
    esm_proj_dim: int = 0
    kmer_residual: bool = False
    censored_loss: bool = False
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    smoke: bool = False


def output_path(config: Config) -> Path:
    """Return the run directory, optionally under a hosted output root.

    Args:
        config: Run configuration, including smoke isolation and encoder choice.

    Returns:
        Path under SC4001_OUTPUT_ROOT (default: Project/outputs). A CNN run keeps
        the Phase 2 layout byte-for-byte; another protein encoder gets its own root
        so the two never share a run directory. A widened ESM projection gets a
        further root of its own, so the first ESM grid's results stay untouched.
        CensoredMSE training also gets a separate prefixed root.
    """
    root = Path(os.environ.get("SC4001_OUTPUT_ROOT", Path(__file__).resolve().parents[1] / "outputs"))
    suffix = "-smoke" if config.smoke else ""
    phase = "phase2" if config.protein_encoder == "cnn" else config.protein_encoder
    if config.protein_encoder == "esm" and config.esm_proj_dim not in (0, config.filters[-1]):
        phase = f"esm-proj{config.esm_proj_dim}"
    if config.kmer_residual:
        phase = f"kmer-residual-{phase}"
    if config.censored_loss:
        phase = f"censored-loss-{phase}"
    return root / phase / config.dataset / config.split / f"seed{config.seed}{suffix}"
