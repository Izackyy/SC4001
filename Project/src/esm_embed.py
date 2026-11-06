"""Precompute frozen ESM-2 protein embeddings for the benchmark and external targets.

The protein language model is never fine-tuned, so its embeddings are a fixed
feature per sequence and are computed once here rather than inside every
training run. That keeps the hosted GPU budget on training and lets the
training kernel run without network access or a transformers dependency.

Row order matches the processed target table exactly, so the saved array can be
indexed with the same prot_index that the token array uses.
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import processed_csv

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MAX_RESIDUES = 1022


def parse_args():
    """Parse the model name, truncation length and an optional timing limit.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--sources", nargs="+", default=["davis", "kiba", "external"],
                        choices=("davis", "kiba", "external"))
    parser.add_argument("--max-residues", type=int, default=MAX_RESIDUES,
                        help="Residues kept per sequence; ESM-2 was trained at 1024 tokens.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Embed only the first N sequences and write nothing: a timing probe.")
    return parser.parse_args()


def load_sequences(source: str) -> pd.Series:
    """Return the sequences of one source in processed-table row order.

    Args:
        source: "davis", "kiba" or "external".

    Returns:
        Sequence strings indexed by position in the source's table.
    """
    if source == "external":
        table = pd.read_csv(processed_csv(DATA_DIR, "external_kinases"), keep_default_na=False)
    else:
        table = pd.read_csv(processed_csv(DATA_DIR, f"{source}_targets"),
                            usecols=["target_id", "sequence"], dtype={"target_id": str})
    return table["sequence"]


@torch.inference_mode()
def embed(sequences, tokenizer, model, max_residues: int, device) -> np.ndarray:
    """Mean-pool the final hidden layer over the residues of each sequence.

    Special tokens are excluded from the mean so that short and long sequences
    are pooled over residues only. Sequences are embedded one at a time, which
    costs nothing extra on CPU and avoids padding waste between a 244-residue
    and a 2549-residue kinase.

    Args:
        sequences: Iterable of amino-acid strings.
        tokenizer: ESM-2 tokenizer.
        model: ESM-2 model in evaluation mode.
        max_residues: Residues kept per sequence before tokenisation.
        device: Torch device holding the model.

    Returns:
        Float32 array of shape (len(sequences), hidden width).

    Raises:
        ValueError: An embedding is not finite.
    """
    vectors = []
    started = time.perf_counter()
    for position, sequence in enumerate(sequences):
        batch = tokenizer(sequence[:max_residues], return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state[0]
        # Drop the leading cls and trailing eos token before pooling.
        vectors.append(hidden[1:-1].mean(dim=0).float().cpu().numpy())
        if (position + 1) % 25 == 0:
            rate = (time.perf_counter() - started) / (position + 1)
            print(f"  {position + 1}/{len(sequences)} at {rate:.2f} s/sequence", flush=True)
    stacked = np.stack(vectors).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise ValueError("ESM-2 produced a nonfinite embedding.")
    return stacked


def main():
    """Embed every requested source and save one npz per source.

    Side Effects:
        Writes data/processed/{source}_esm.npz with a prot_embeddings array
        aligned to that source's processed table, and prints timing.
    """
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {args.model}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    print(f"Hidden width {model.config.hidden_size} on {device}", flush=True)
    for source in args.sources:
        sequences = load_sequences(source)
        if args.limit:
            sequences = sequences.iloc[:args.limit]
        print(f"{source}: {len(sequences)} sequences", flush=True)
        started = time.perf_counter()
        embeddings = embed(sequences, tokenizer, model, args.max_residues, device)
        elapsed = time.perf_counter() - started
        print(f"{source}: {embeddings.shape} in {elapsed / 60:.1f} min "
              f"({elapsed / len(sequences):.2f} s/sequence)", flush=True)
        if args.limit:
            print(f"{source}: timing probe only, nothing written.", flush=True)
            continue
        destination = DATA_DIR / f"{source}_esm.npz"
        np.savez_compressed(destination, prot_embeddings=embeddings)
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
