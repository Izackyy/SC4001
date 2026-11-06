"""ID-aligned processed tables and token datasets, reusable in Phase 3."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


class PairDataset(Dataset):
    """Pairs indexing shared token tensors; censoring follows dataset row order."""

    def __init__(self, pairs: pd.DataFrame, drug_tokens: torch.Tensor, prot_tokens: torch.Tensor,
                 offsets: np.ndarray | None = None):
        """Store pre-mapped pairs and shared entity tokens.

        Args:
            pairs: One partition with drug_index, prot_index, score and censored.
            drug_tokens: Integer token tensor in ligand-table order.
            prot_tokens: Integer token tensor in target-table order.
            offsets: Optional per-pair baseline prediction in the full pairs table's row
                order, added to the model's output. All zeros when not supplied, which
                leaves the model predicting the score directly.
        """
        self.drug_tokens = drug_tokens
        self.prot_tokens = prot_tokens
        self.drug_indices = pairs["drug_index"].to_numpy(dtype=np.int64)
        self.prot_indices = pairs["prot_index"].to_numpy(dtype=np.int64)
        self.scores = torch.tensor(pairs["score"].to_numpy(), dtype=torch.float32)
        self.censored = pairs["censored"].to_numpy(dtype=bool)
        self.offsets = torch.zeros_like(self.scores) if offsets is None else torch.tensor(
            offsets[pairs.index.to_numpy()], dtype=torch.float32)

    def __len__(self):
        """Return the number of content pairs in this partition."""
        return len(self.scores)

    def __getitem__(self, index):
        """Return drug tokens, protein tokens, a scalar score and a scalar offset.

        Args:
            index: Pair position in this partition.

        Returns:
            Four tensors suitable for default DataLoader collation. The offset is 0
            unless a baseline offset was supplied.
        """
        return (self.drug_tokens[self.drug_indices[index]],
                self.prot_tokens[self.prot_indices[index]],
                self.scores[index], self.offsets[index])


def processed_csv(data_dir: Path, name: str) -> Path:
    """Resolve a processed table stored either gzipped or plain.

    Kaggle decompresses .gz members when it unpacks an uploaded dataset archive,
    so the same table is {name}.csv.gz locally and {name}.csv on Kaggle.

    Args:
        data_dir: Processed-table directory.
        name: Table stem, for example "davis_pairs".

    Returns:
        Path to the gzipped file when it exists, otherwise the plain CSV.

    Raises:
        FileNotFoundError: Neither variant is present.
    """
    for suffix in (".csv.gz", ".csv"):
        candidate = data_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Neither {name}.csv.gz nor {name}.csv is in {data_dir}.")


def resolve_split_column(pairs, data_dir, dataset, split):
    """Return the partition label per pair for a split column.

    Phase 1's splits live in the pairs table. Extra cold-target split seeds live
    in {dataset}_extra_splits.csv.gz so the Phase 1 export stays byte-identical.

    Args:
        pairs: Content-keyed pair table, used for row order and identifiers.
        data_dir: Processed-table directory.
        dataset: Dataset name, selecting the extra-splits file.
        split: Split column name.

    Returns:
        NumPy array of partition labels aligned to the pair table's rows.

    Raises:
        ValueError: The split exists in neither table, or does not cover every pair.
    """
    if split in pairs:
        return pairs[split].to_numpy()
    try:
        extra = pd.read_csv(processed_csv(data_dir, f"{dataset}_extra_splits"),
                            dtype={"ligand_id": str, "target_id": str})
    except FileNotFoundError:
        raise ValueError(f"{split} is unavailable for {dataset}; split_mutant is DAVIS-only "
                         f"and extra split seeds need {dataset}_extra_splits.csv[.gz].") from None
    if split not in extra:
        raise ValueError(f"{split} is in neither {dataset}_pairs nor {dataset}_extra_splits.")
    merged = pairs[["ligand_id", "target_id"]].merge(
        extra[["ligand_id", "target_id", split]], on=["ligand_id", "target_id"], how="left")
    if len(merged) != len(pairs) or merged[split].isna().any():
        raise ValueError(f"{split} does not cover every {dataset} pair exactly once.")
    return merged[split].to_numpy()


def load_esm_features(data_dir: Path, dataset: str, n_targets: int, esm_dim: int):
    """Load frozen ESM-2 target embeddings in processed-target row order.

    The array replaces the protein token tensor, so the pair dataset indexes it
    with the same prot_index and nothing else in the loader changes.

    Args:
        data_dir: Processed-table directory.
        dataset: Dataset name selecting {dataset}_esm.npz.
        n_targets: Expected row count, from the processed target table.
        esm_dim: Expected embedding width.

    Returns:
        Float32 tensor of shape (n_targets, esm_dim).

    Raises:
        FileNotFoundError: The embedding file has not been precomputed.
        ValueError: The array does not match the target table or configured width.
    """
    path = data_dir / f"{dataset}_esm.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run python -m src.esm_embed first.")
    with np.load(path, allow_pickle=False) as arrays:
        embeddings = arrays["prot_embeddings"]
    if embeddings.shape != (n_targets, esm_dim):
        raise ValueError(f"{path} has shape {embeddings.shape}, expected {(n_targets, esm_dim)}.")
    return torch.tensor(embeddings, dtype=torch.float32)


def load_data(config: Config, data_dir: Path | None = None):
    """Load aligned token tables and make train/val/test loaders.

    Args:
        config: Dataset, split, batch size and seed selection.
        data_dir: Processed-table directory; defaults to Project/data/processed.

    Returns:
        A mapping of partition names to DataLoaders, and vocabulary metadata.

    Raises:
        ValueError: A split, identifier, token shape or partition is invalid.
    """
    data_dir = data_dir or Path(__file__).resolve().parents[1] / "data" / "processed"
    with (data_dir / "vocab.json").open(encoding="utf-8") as handle:
        vocab = json.load(handle)
    ds = config.dataset
    pairs = pd.read_csv(processed_csv(data_dir, f"{ds}_pairs"), dtype={"ligand_id": str, "target_id": str})
    ligands = pd.read_csv(processed_csv(data_dir, f"{ds}_ligands"), usecols=["ligand_id"], dtype={"ligand_id": str})
    targets = pd.read_csv(processed_csv(data_dir, f"{ds}_targets"), usecols=["target_id"], dtype={"target_id": str})
    partitions = resolve_split_column(pairs, data_dir, ds, config.split)
    for id_column, entities, index_column in (("ligand_id", ligands, "drug_index"), ("target_id", targets, "prot_index")):
        if entities[id_column].duplicated().any():
            raise ValueError(f"Duplicate {id_column} in the entity table.")
        pairs[index_column] = pairs[id_column].map(dict(zip(entities[id_column], range(len(entities)))))
        if pairs[index_column].isna().any():
            raise ValueError(f"Unmapped {id_column} in pairs.")
    with np.load(data_dir / f"{ds}_tokens.npz", allow_pickle=False) as tokens:
        drug_tokens = torch.tensor(tokens["drug_tokens"], dtype=torch.long)
        prot_tokens = torch.tensor(tokens["prot_tokens"], dtype=torch.long)
    lengths = vocab["max_len"][ds]
    if tuple(drug_tokens.shape) != (len(ligands), lengths["smiles"]) or tuple(prot_tokens.shape) != (len(targets), lengths["protein"]):
        raise ValueError("Token arrays do not match entity row counts and vocab.json lengths.")
    if config.protein_encoder == "esm":
        prot_tokens = load_esm_features(data_dir, ds, len(targets), config.esm_dim)
    offsets = None
    if config.kmer_residual:
        offset_path = data_dir / f"{ds}_{config.split}_kmer_offset.npz"
        if not offset_path.exists():
            raise FileNotFoundError(f"{offset_path} is missing; run python -m src.kmer_offsets "
                                    f"--dataset {ds} --split {config.split} first.")
        with np.load(offset_path, allow_pickle=False) as arrays:
            offsets = arrays["offsets"]
        if offsets.shape != (len(pairs),):
            raise ValueError(f"{offset_path} has shape {offsets.shape}, expected {(len(pairs),)}.")
    loaders = {}
    for partition in ("train", "val", "test"):
        subset = PairDataset(pairs.loc[partitions == partition], drug_tokens, prot_tokens, offsets)
        if not len(subset):
            raise ValueError(f"Empty {partition} partition for {ds}/{config.split}.")
        loaders[partition] = DataLoader(
            subset, batch_size=config.batch_size, shuffle=partition == "train",
            generator=torch.Generator().manual_seed(config.seed), num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders, vocab
