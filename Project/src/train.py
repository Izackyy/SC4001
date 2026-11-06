"""Train one GPU or TPU baseline; CPU is reserved for explicit smoke checks."""

import argparse
import re
from dataclasses import asdict
from functools import lru_cache
from itertools import islice
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, output_path
from src.data import load_data
from src.metrics import concordance_index, mse, training_mean_baseline
from src.model import DeepDTA


@lru_cache(maxsize=1)
def resolve_device():
    """Resolve and cache the preferred accelerator, trying CUDA before XLA.

    XLA imports and device discovery catch Exception because machines without
    a TPU can raise runtime errors, not just ImportError. Older XLA versions
    use xla_model.xla_device() instead of the modern torch_xla.device().

    Returns:
        The available accelerator as a torch.device, or None for CPU only.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla
    except Exception:
        return None
    try:
        return torch_xla.device()
    except Exception:
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        except Exception:
            return None


def parse_args():
    """Parse one experiment and refuse unavailable/unauthorised training devices.

    Returns:
        Parsed argparse namespace.

    Raises:
        SystemExit: Arguments are invalid or CPU training is not permitted.
    """
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("davis", "kiba"), required=True)
    parser.add_argument("--split", required=True, help="split_warm, split_cold_target, split_mutant, or split_cold_target_s<seed>.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--patience", type=int, default=defaults.patience,
                        help="Epochs without a min_delta improvement before stopping. Set it to "
                             "--max-epochs to disable early stopping, which is required when "
                             "comparing two methods whose validation curves plateau.")
    parser.add_argument("--protein-encoder", choices=("cnn", "esm"), default=defaults.protein_encoder,
                        help="cnn: the DeepDTA character CNN. esm: a frozen ESM-2 embedding per "
                             "target, precomputed by src.esm_embed, projected by one linear layer.")
    parser.add_argument("--esm-proj-dim", type=int, default=defaults.esm_proj_dim,
                        help="Width of that linear projection. 0 uses the CNN branch's output "
                             "width (96), which matches the regression head exactly but is a 13x "
                             "compression of the 1280-wide embedding. A larger value widens the "
                             "projection and the head's input, and writes to its own output root.")
    parser.add_argument("--kmer-residual", action="store_true",
                        help="Add the k-mer nearest-neighbour baseline's prediction to the "
                             "model's output, so the network learns only the correction on top "
                             "of it. Needs src.kmer_offsets to have been run for this dataset "
                             "and split; training pairs use out-of-fold offsets.")
    parser.add_argument("--censored-loss", action="store_true",
                        help="Use CensoredMSE for training; a censored score is an upper bound.")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt if present.")
    parser.add_argument("--smoke", action="store_true", help="Two epochs, two train batches and one eval batch, batch size capped at 8; isolated output.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU only for the correctness smoke run; requires --smoke. Real experiments require a hosted GPU or TPU.")
    args = parser.parse_args()
    if not re.fullmatch(r"split_warm|split_cold_target|split_mutant|split_cold_target_s\d+", args.split):
        parser.error(f"Unknown --split {args.split!r}; expected split_warm, split_cold_target, "
                     "split_mutant, or split_cold_target_s<seed>.")
    if args.dataset == "kiba" and args.split == "split_mutant":
        parser.error("split_mutant is DAVIS-only; KIBA has no split_mutant column.")
    if args.max_epochs < 1 or args.batch_size < 1:
        parser.error("--max-epochs and --batch-size must be positive.")
    if args.patience < 1:
        parser.error("--patience must be positive.")
    if args.allow_cpu and not args.smoke:
        parser.error("--allow-cpu is reserved for correctness checks with --smoke.")
    if resolve_device() is None and not args.allow_cpu:
        parser.error("No GPU or TPU was found: refusing to train on CPU. Use a Colab or Kaggle GPU or TPU for real experiments. --allow-cpu is only for the correctness smoke run with --smoke.")
    return args


def optimizer_step(optimizer, device):
    """Update parameters and dispatch the accumulated graph on XLA.

    Args:
        optimizer: Optimiser whose parameters and state are updated in place.
        device: Torch device used for training; XLA requires a step barrier.
    """
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.optimizer_step(optimizer, barrier=True)
    else:
        optimizer.step()


def censored_mse(prediction, target, censored):
    """Return CensoredMSE, a left-censored squared error averaged over pairs.

    Args:
        prediction: Final predicted scores, including any baseline offsets.
        target: Stored scores, used as upper bounds for censored pairs.
        censored: Float32 flags, 1 for left-censored and 0 for observed pairs.

    Returns:
        Scalar mean squared error with zero penalty below censored bounds.
    """
    error = target - prediction
    return torch.where(censored.bool(), error.clamp(max=0), error).square().mean()


def train_epoch(model, loader, optimizer, device, batch_limit=None, censored_loss=False):
    """Run a training epoch with optional CensoredMSE objective.

    Args:
        model: Regression model, updated in place.
        loader: Training DataLoader, including censored pairs.
        optimizer: Optimiser whose state is updated.
        device: Torch device receiving batches.
        batch_limit: Optional smoke cap on training batches.
        censored_loss: Apply CensoredMSE instead of plain MSE to the gradient.

    Returns:
        Plain MSE across the examples actually trained on this epoch.
    """
    model.train()
    squared_error = 0.0
    count = 0
    for drug, protein, scores, offsets, censored in islice(loader, batch_limit):
        drug, protein, scores = drug.to(device), protein.to(device), scores.to(device)
        optimizer.zero_grad(set_to_none=True)
        # The offset is 0 unless --kmer-residual supplied one, so this is the plain model then.
        prediction = model(drug, protein) + offsets.to(device)
        loss = (censored_mse(prediction, scores, censored.to(device)) if censored_loss
                else torch.nn.functional.mse_loss(prediction, scores))
        loss.backward()
        optimizer_step(optimizer, device)
        # Keep reported train_mse as plain MSE so epochs remain comparable across objectives.
        reported_mse = (torch.nn.functional.mse_loss(prediction.detach(), scores).item()
                        if censored_loss else loss.item())
        squared_error += reported_mse * len(scores)
        count += len(scores)
    return squared_error / count


@torch.inference_mode()
def predict(model, loader, device, batch_limit=None):
    """Evaluate a non-shuffled partition without gradients or dropout.

    Args:
        model: Regression model, switched to evaluation mode.
        loader: Validation or test loader in dataset order.
        device: Torch device receiving batches.
        batch_limit: Optional smoke cap on evaluation batches.

    Returns:
        Aligned NumPy arrays of truth, prediction and censor flags.
    """
    model.eval()
    actual, predicted = [], []
    for drug, protein, scores, offsets, _ in islice(loader, batch_limit):
        predicted.append((model(drug.to(device), protein.to(device)).cpu() + offsets).numpy())
        actual.append(scores.numpy())
    y, prediction = np.concatenate(actual), np.concatenate(predicted)
    return y, prediction, loader.dataset.censored[:len(y)]


def evaluate_subsets(y, prediction, censored, train_scores, dataset):
    """Report model and mean-predictor metrics on the requested subsets.

    Args:
        y: Evaluation truth.
        prediction: Model predictions aligned with y.
        censored: Aligned censor flags.
        train_scores: All training scores, used only to estimate the mean.
        dataset: DAVIS additionally reports uncensored metrics.

    Returns:
        Model metrics and training-mean metrics, keyed by subset. Undefined
        metrics become None so saved JSON is standard JSON.
    """
    masks = {"all": np.ones(len(y), dtype=bool)}
    if dataset == "davis":
        masks["uncensored"] = ~censored
    model_metrics, baseline_metrics = {}, {}
    for name, mask in masks.items():
        model_metrics[name] = {"n": int(mask.sum()), "mse": mse(y[mask], prediction[mask]), "ci": concordance_index(y[mask], prediction[mask])}
        baseline_metrics[name] = {"n": int(mask.sum()), **training_mean_baseline(train_scores, y[mask])}
        for metrics in (model_metrics[name], baseline_metrics[name]):
            for key, value in metrics.items():
                if isinstance(value, float) and not math.isfinite(value):
                    metrics[key] = None
    return model_metrics, baseline_metrics


def save_checkpoint(state, path, device):
    """Atomically replace a checkpoint after serialisation completes.

    Args:
        state: Torch-serialisable checkpoint or model state dictionary.
        path: Destination Path; a temporary sibling is replaced into it.
        device: Torch device used for training; XLA tensors are saved via host.
    """
    temporary = path.with_suffix(".pt.tmp")
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.save(state, temporary)
    else:
        torch.save(state, temporary)
    temporary.replace(path)


def main():
    """Train/resume one run, persist every epoch and test the best model once.

    Side Effects:
        Writes configuration, checkpoints, history and metrics to the run path;
        prints device, resume status and measured epoch wall times.

    Raises:
        SystemExit: An existing run needs --resume or its configuration differs.
        ValueError: Processed data or validation scores are invalid.
    """
    args = parse_args()
    config = Config(dataset=args.dataset, split=args.split, seed=args.seed,
                    max_epochs=2 if args.smoke else args.max_epochs,
                    batch_size=min(args.batch_size, 8) if args.smoke else args.batch_size,
                    patience=args.patience, smoke=args.smoke,
                    protein_encoder=args.protein_encoder, esm_proj_dim=args.esm_proj_dim,
                    kmer_residual=args.kmer_residual, censored_loss=args.censored_loss)
    device = resolve_device() or torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    device_note = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (TPU via XLA)" if device.type == "xla" else " (correctness smoke only)"
    print(f"Device: {device}" + device_note, flush=True)
    if config.smoke:
        print(f"SMOKE: 2 epochs; at most 2 train / 1 evaluation batches; batch_size={config.batch_size}. Not an experimental result.", flush=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    directory = output_path(config)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    run_config = asdict(config)
    if not config.censored_loss:
        run_config.pop("censored_loss")
    if config_path.exists():
        if not args.resume:
            raise SystemExit(f"Run exists at {directory}. Use --resume to continue it.")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        current = json.loads(json.dumps(run_config))
        if any(saved[key] != value for key, value in current.items() if key != "max_epochs"):
            raise SystemExit("Resume configuration differs from config.json (only max_epochs may change).")
    loaders, vocab = load_data(config)
    config_path.write_text(json.dumps({**run_config, "token_lengths": vocab["max_len"][config.dataset],
                                      "vocab_sizes": {key: len(vocab[key]) + 1 for key in ("smiles", "protein")}}, indent=2) + "\n", encoding="utf-8")
    model = DeepDTA(len(vocab["smiles"]) + 1, len(vocab["protein"]) + 1, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    epoch, best_epoch, patience_count = 0, 0, 0
    best_score = patience_score = float("inf")
    total_seconds = 0.0
    history = []
    last_path = directory / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, best_epoch = state["epoch"], state["best_epoch"]
        best_score, patience_score, patience_count = state["best_score"], state["patience_score"], state["patience_count"]
        total_seconds, history = state["total_seconds"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        numpy_rng = state["numpy_rng"]
        np.random.set_state((numpy_rng[0], np.array(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
        for name, loader in loaders.items():
            loader.generator.set_state(state["loader_rng"][name])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"Resuming from saved epoch {epoch}; next epoch would be {epoch + 1}; best epoch {best_epoch}; patience {patience_count}/{config.patience}.", flush=True)
    elif args.resume:
        print("No last.pt found; starting at epoch 1.", flush=True)

    finished = epoch >= config.max_epochs or patience_count >= config.patience
    if finished and (directory / "metrics.json").exists():
        print("Run already complete; no training or repeat test evaluation. Using saved metrics.", flush=True)
        print((directory / "metrics.json").read_text(encoding="utf-8"), flush=True)
        return
    train_limit, eval_limit = (2, 1) if config.smoke else (None, None)
    while epoch < config.max_epochs and patience_count < config.patience:
        started = time.perf_counter()
        epoch += 1
        train_mse = train_epoch(model, loaders["train"], optimizer, device, train_limit,
                                config.censored_loss)
        y, prediction, _ = predict(model, loaders["val"], device, eval_limit)
        val_mse = mse(y, prediction)
        if not math.isfinite(val_mse):
            raise ValueError("Validation MSE is not finite; stopping before checkpointing.")
        if val_mse < best_score:
            best_score, best_epoch = val_mse, epoch
            save_checkpoint(model.state_dict(), directory / "best.pt", device)
        if val_mse < patience_score - config.min_delta:
            patience_score, patience_count = val_mse, 0
        else:
            patience_count += 1
        seconds = time.perf_counter() - started
        total_seconds += seconds
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "seconds": seconds})
        numpy_rng = np.random.get_state()
        save_checkpoint({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
            "patience_score": patience_score, "patience_count": patience_count,
            "total_seconds": total_seconds, "history": history,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            "loader_rng": {name: loader.generator.get_state() for name, loader in loaders.items()},
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, last_path, device)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"Epoch {epoch}/{config.max_epochs}: train_mse={train_mse:.6f} val_mse={val_mse:.6f} seconds={seconds:.3f}", flush=True)
    if patience_count >= config.patience:
        print(f"Early stopping after {epoch} epochs.", flush=True)

    started = time.perf_counter()
    model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
    result = {"dataset": config.dataset, "split": config.split, "seed": config.seed,
              "smoke": config.smoke, "best_epoch": best_epoch, "stopped_epoch": epoch,
              "device": str(device), "training_mean_baseline": {}}
    train_scores = loaders["train"].dataset.scores.numpy()
    for partition in ("val", "test"):
        y, prediction, censored = predict(model, loaders[partition], device, eval_limit)
        result[partition], result["training_mean_baseline"][partition] = evaluate_subsets(y, prediction, censored, train_scores, config.dataset)
    result["total_seconds"] = total_seconds + time.perf_counter() - started
    metrics_text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (directory / "metrics.json").write_text(metrics_text, encoding="utf-8")
    print(f"Saved {directory}", flush=True)
    print(metrics_text, flush=True)


if __name__ == "__main__":
    main()
