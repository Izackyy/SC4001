"""Train one DANN run on labelled pairs and unlabelled validation/external proteins."""

import argparse
import re
from dataclasses import asdict, dataclass
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.dann import DomainDiscriminator, lambda_schedule, reverse_gradient
from src.data import load_data, processed_csv, resolve_split_column
from src.metrics import mse
from src.model import DeepDTA
from src.train import evaluate_subsets, optimizer_step, predict, resolve_device, save_checkpoint


@dataclass
class DANNConfig(Config):
    """Phase 2 defaults plus the adversarial branch's saved configuration."""

    lambda_max: float = 1.0
    discriminator_input: str = "protein"
    discriminator_hidden: int = 64


def parse_args():
    """Parse the baseline flags and two DANN controls, enforcing CPU refusal.

    Returns:
        Parsed argparse namespace.

    Raises:
        SystemExit: Arguments are invalid or CPU training is not permitted.
    """
    defaults = DANNConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("davis", "kiba"), required=True)
    parser.add_argument("--split", required=True, help="split_warm, split_cold_target, split_mutant, or split_cold_target_s<seed>.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--patience", type=int, default=defaults.patience,
                        help="Epochs without a min_delta improvement before stopping. Set it to "
                             "--max-epochs to disable early stopping, which is required when "
                             "comparing against the baseline: validation MSE plateaus for about "
                             "twenty epochs, so the stopping rule otherwise decides the result.")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt if present.")
    parser.add_argument("--smoke", action="store_true", help="Two epochs, two train batches and one eval batch, batch size capped at 8; isolated output.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU only for the correctness smoke run; requires --smoke.")
    parser.add_argument("--lambda-max", type=float, default=defaults.lambda_max)
    parser.add_argument("--discriminator-input", choices=("protein", "joint"), default=defaults.discriminator_input)
    parser.add_argument("--protein-encoder", choices=("cnn", "esm"), default=defaults.protein_encoder,
                        help="cnn: the character CNN Phase 3 used. esm: the frozen ESM-2 branch, "
                             "whose target-domain pool is embedded the same way.")
    parser.add_argument("--esm-proj-dim", type=int, default=defaults.esm_proj_dim,
                        help="Projection width for --protein-encoder esm; 0 uses the CNN "
                             "branch's output width.")
    parser.add_argument("--kmer-residual", action="store_true",
                        help="Add the k-mer neighbour prediction to the model's output, so the "
                             "adversary acts on a representation that already orders unseen "
                             "targets. Needs src.kmer_offsets to have been run.")
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
    if not math.isfinite(args.lambda_max) or args.lambda_max < 0:
        parser.error("--lambda-max must be finite and nonnegative.")
    if args.allow_cpu and not args.smoke:
        parser.error("--allow-cpu is reserved for correctness checks with --smoke.")
    if resolve_device() is None and not args.allow_cpu:
        parser.error("No GPU or TPU was found: refusing to train on CPU. Use a Colab or Kaggle GPU or TPU for real experiments. --allow-cpu is only for the correctness smoke run with --smoke.")
    return args


def load_target_domain(config, data_dir=None):
    """Assemble unique validation proteins followed by the external pool.

    Args:
        config: Dataset and split selection.
        data_dir: Processed directory; defaults to Project/data/processed.

    Returns:
        Aligned protein token tensor and audit table (sequence, origin, IDs).
        Affinity labels are never read by this function.

    Raises:
        ValueError: Test sequences overlap training/domain inputs, external
            similarities are unsafe, or saved tokens do not match the table.
        FileNotFoundError: Required processed exports are missing.
    """
    data_dir = data_dir or Path(__file__).resolve().parents[1] / "data/processed"
    targets = pd.read_csv(processed_csv(data_dir, f"{config.dataset}_targets"),
                          usecols=["target_id", "sequence"], dtype={"target_id": str})
    # Extra cold-target split seeds live in {dataset}_extra_splits, not in the pairs table, so
    # the column must be resolved exactly as load_data does rather than read with usecols.
    pairs_path = processed_csv(data_dir, f"{config.dataset}_pairs")
    columns = ["ligand_id", "target_id"]
    if config.split in pd.read_csv(pairs_path, nrows=0).columns:
        columns.append(config.split)
    pairs = pd.read_csv(pairs_path, usecols=columns,
                        dtype={"ligand_id": str, "target_id": str})
    partitions = resolve_split_column(pairs, data_dir, config.dataset, config.split)
    test_ids = set(pairs.loc[partitions == "test", "target_id"])
    test_sequences = set(targets.loc[targets.target_id.isin(test_ids), "sequence"])
    train_ids = set(pairs.loc[partitions == "train", "target_id"])
    train_sequences = set(targets.loc[targets.target_id.isin(train_ids), "sequence"])
    if train_sequences & test_sequences:
        raise ValueError("Training and test proteins overlap in this split. DANN's no-test-protein contract requires a sequence-disjoint split; use split_cold_target.")
    val_ids = set(pairs.loc[partitions == "val", "target_id"])
    validation = targets.loc[targets.target_id.isin(val_ids)].drop_duplicates("sequence")
    external = pd.read_csv(processed_csv(data_dir, "external_kinases"), keep_default_na=False)
    if not external.max_test_jaccard.between(0, 0.90, inclusive="left").all():
        raise ValueError("External pool contains invalid or unsafe test similarities (must be < 0.90).")
    if set(external.sequence) & set(targets.sequence):
        raise ValueError("External pool overlaps benchmark target sequences.")
    table = pd.concat([validation.assign(origin="validation"), external.assign(origin="external")],
                      ignore_index=True)
    if set(table.sequence) & test_sequences:
        raise ValueError("Target-domain stream contains held-out test sequences.")
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    length = vocab["max_len"][config.dataset]["protein"]
    if config.protein_encoder == "esm":
        # The ESM branch consumes embeddings, so the unlabelled pool is embedded the same way.
        # Both files are in processed-table row order, which is what the index below assumes.
        with np.load(data_dir / f"{config.dataset}_esm.npz", allow_pickle=False) as arrays:
            validation_features = arrays["prot_embeddings"][validation.index]
        with np.load(data_dir / "external_esm.npz", allow_pickle=False) as arrays:
            external_features = arrays["prot_embeddings"]
        if external_features.shape != (len(external), config.esm_dim):
            raise ValueError(f"external_esm.npz is {external_features.shape}, expected "
                             f"{(len(external), config.esm_dim)}.")
        features = np.concatenate([validation_features, external_features])
        if features.shape != (len(table), config.esm_dim) or not np.isfinite(features).all():
            raise ValueError("Target-domain embeddings are misaligned or nonfinite.")
        return torch.tensor(features, dtype=torch.float32), table
    with np.load(data_dir / f"{config.dataset}_tokens.npz", allow_pickle=False) as arrays:
        validation_tokens = arrays["prot_tokens"][validation.index]
    with np.load(data_dir / f"external_tokens_{config.dataset}.npz", allow_pickle=False) as arrays:
        external_tokens = arrays["prot_tokens"]
    if external_tokens.dtype != np.int16 or external_tokens.shape != (len(external), length):
        raise ValueError("External tokens must be int16 with CSV-aligned rows and vocabulary length.")
    tokens = np.concatenate([validation_tokens, external_tokens])
    lookup = {char: index + 1 for index, char in enumerate(vocab["protein"])}
    for sequence, row in zip(table.sequence, tokens):
        expected = np.zeros(length, dtype=np.int16)
        prefix = sequence[:length]
        expected[:len(prefix)] = [lookup[char] for char in prefix]
        if not np.array_equal(row, expected):
            raise ValueError("Target-domain tokens do not match their sequence row.")
    return torch.tensor(tokens, dtype=torch.long), table


def cycle_target_indices(size, generator):
    """Cycle shuffled target permutations without dropping a partial batch.

    Args:
        size: Number of unique target-domain proteins.
        generator: Dedicated RNG, advanced here and checkpointed each epoch.

    Yields:
        Row indices indefinitely; each permutation visits every target once.
    """
    while True:
        yield from torch.randperm(size, generator=generator).tolist()


def train_epoch(model, discriminator, loader, target_tokens, optimizer,
                domain_optimizer, device, config, epoch, target_generator,
                batch_limit=None):
    """Train regression and the balanced domain branch for one epoch.

    Args:
        model: DeepDTA model updated in place.
        discriminator: Small domain classifier updated in place.
        loader: Shuffled training pairs, retaining their natural target frequency.
        target_tokens: Unlabelled validation/external protein tensor on CPU.
        optimizer: Model optimiser.
        domain_optimizer: Discriminator optimiser.
        device: Training device.
        config: DANN configuration, including the full epoch horizon.
        epoch: Zero-based epoch index for the per-step schedule.
        target_generator: Checkpointed RNG for target shuffles.
        batch_limit: Optional smoke cap on training steps.

    Returns:
        Sample-weighted MSE, domain BCE, domain accuracy and mean step lambda.
        An odd final batch alternates the extra row's domain by global step.
    """
    model.train()
    discriminator.train()
    steps = min(len(loader), batch_limit) if batch_limit is not None else len(loader)
    # Restart the stream at each epoch; the generator state makes resume exact.
    target_stream = cycle_target_indices(len(target_tokens), target_generator)
    squared_error, domain_loss, correct, coefficient_sum, count = 0.0, 0.0, 0, 0.0, 0
    for step, (drug, protein, scores, offsets, _) in enumerate(islice(loader, batch_limit)):
        drug, protein, scores = drug.to(device), protein.to(device), scores.to(device)
        optimizer.zero_grad(set_to_none=True)
        domain_optimizer.zero_grad(set_to_none=True)
        drug_features = model.drug_encoder(drug)
        protein_features = model.protein_encoder(protein)
        # The offset is 0 unless a baseline offset was loaded, so Phase 3 is unchanged.
        prediction = model.regressor(
            torch.cat([drug_features, protein_features], dim=1)).squeeze(-1) + offsets.to(device)
        regression_loss = torch.nn.functional.mse_loss(prediction, scores)
        batch_size = len(scores)
        source_size = batch_size // 2 + (batch_size % 2) * ((epoch * steps + step) % 2)
        target_size = batch_size - source_size
        indices = list(islice(target_stream, target_size))
        # A prefix of the shuffled regression batch samples pairs, not unique proteins.
        source_features = protein_features[:source_size]
        if target_size:
            target_features = model.protein_encoder(target_tokens[indices].to(device))
        else:
            target_features = protein_features[:0]
        if config.discriminator_input == "joint":
            # Match the training-drug distribution across domains; no target affinities.
            source_features = torch.cat([drug_features[:source_size], source_features], dim=1)
            target_features = torch.cat([drug_features[:target_size], target_features], dim=1)
        features = torch.cat([source_features, target_features], dim=0)
        labels = torch.cat([torch.zeros(source_size, device=device), torch.ones(target_size, device=device)])
        coefficient = lambda_schedule((epoch * steps + step) / (config.max_epochs * steps), config.lambda_max)
        logits = discriminator(reverse_gradient(features, coefficient))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        # Lambda enters once, through the reversal coefficient, as in Ganin et al. Weighting the
        # BCE as well would scale the discriminator's own gradient by lambda, so it would train at
        # 2.5% of its rate in the first epoch and never become a useful adversary: measured on a
        # T4, domain loss stayed at ln 2 for every epoch of a 20-epoch run.
        total = regression_loss + bce
        total.backward()
        optimizer_step(optimizer, device)
        optimizer_step(domain_optimizer, device)
        squared_error += regression_loss.item() * batch_size
        domain_loss += bce.item() * batch_size
        correct += ((logits.detach() >= 0) == labels.bool()).sum().item()
        coefficient_sum += coefficient
        count += batch_size
    return {"train_mse": squared_error / count, "domain_loss": domain_loss / count,
            "domain_accuracy": correct / count, "lambda": coefficient_sum / steps}


def main():
    """Train/resume one DANN run, select by regression MSE and evaluate once.

    Side Effects:
        Writes Phase 3 configuration, checkpoints, history and metrics; prints
        domain composition, resume status and epoch wall times.

    Raises:
        SystemExit: An existing run needs --resume or configuration differs.
        ValueError: Data leakage, token mismatch or nonfinite validation MSE.
    """
    args = parse_args()
    config = DANNConfig(dataset=args.dataset, split=args.split, seed=args.seed,
                        max_epochs=2 if args.smoke else args.max_epochs,
                        batch_size=min(args.batch_size, 8) if args.smoke else args.batch_size,
                        patience=args.patience, smoke=args.smoke, lambda_max=args.lambda_max,
                        discriminator_input=args.discriminator_input,
                        protein_encoder=args.protein_encoder, esm_proj_dim=args.esm_proj_dim,
                        kmer_residual=args.kmer_residual)
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

    root = Path(os.environ.get("SC4001_OUTPUT_ROOT", Path(__file__).resolve().parents[1] / "outputs"))
    suffix = "-smoke" if config.smoke else ""
    # The CNN arm keeps the Phase 3 layout byte-for-byte; a different protein representation or
    # an added baseline offset is a different experiment and gets its own root.
    phase = "phase3"
    if config.protein_encoder != "cnn":
        phase += f"-{config.protein_encoder}"
        if config.esm_proj_dim not in (0, config.filters[-1]):
            phase += f"proj{config.esm_proj_dim}"
    if config.kmer_residual:
        phase = f"kmer-residual-{phase}"
    directory = root / phase / config.dataset / config.split / f"seed{config.seed}{suffix}"
    config_path = directory / "config.json"
    run_config = asdict(config)
    run_config.pop("censored_loss")
    if config_path.exists():
        if not args.resume:
            raise SystemExit(f"Run exists at {directory}. Use --resume to continue it.")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        current = json.loads(json.dumps(run_config))
        if any(saved[key] != value for key, value in current.items() if key != "max_epochs"):
            raise SystemExit("Resume configuration differs from config.json (only max_epochs may change).")
    loaders, vocab = load_data(config)
    target_tokens, target_table = load_target_domain(config)
    print(f"Target domain: {(target_table.origin == 'validation').sum()} validation + {(target_table.origin == 'external').sum()} external = {len(target_table)} proteins; no own-split test sequences.", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({**run_config, "token_lengths": vocab["max_len"][config.dataset],
                                      "vocab_sizes": {key: len(vocab[key]) + 1 for key in ("smiles", "protein")}}, indent=2) + "\n", encoding="utf-8")
    model = DeepDTA(len(vocab["smiles"]) + 1, len(vocab["protein"]) + 1, config).to(device)
    input_width = model.protein_encoder.out_dim + (
        model.drug_encoder.out_dim if config.discriminator_input == "joint" else 0)
    discriminator = DomainDiscriminator(input_width, config.discriminator_hidden, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    domain_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    target_generator = torch.Generator().manual_seed(config.seed + 1)
    epoch, best_epoch, patience_count = 0, 0, 0
    best_score = patience_score = float("inf")
    total_seconds = 0.0
    history = []
    last_path = directory / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        discriminator.load_state_dict(state["discriminator"])
        domain_optimizer.load_state_dict(state["domain_optimizer"])
        target_generator.set_state(state["target_rng"])
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
        training = train_epoch(model, discriminator, loaders["train"], target_tokens,
                               optimizer, domain_optimizer, device, config, epoch,
                               target_generator, train_limit)
        epoch += 1
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
        history.append({"epoch": epoch, **training, "val_mse": val_mse, "seconds": seconds})
        numpy_rng = np.random.get_state()
        save_checkpoint({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "discriminator": discriminator.state_dict(), "domain_optimizer": domain_optimizer.state_dict(),
            "target_rng": target_generator.get_state(),
            "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
            "patience_score": patience_score, "patience_count": patience_count,
            "total_seconds": total_seconds, "history": history,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            "loader_rng": {name: loader.generator.get_state() for name, loader in loaders.items()},
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, last_path, device)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"Epoch {epoch}/{config.max_epochs}: train_mse={training['train_mse']:.6f} val_mse={val_mse:.6f} domain_loss={training['domain_loss']:.6f} domain_accuracy={training['domain_accuracy']:.4f} lambda={training['lambda']:.6f} seconds={seconds:.3f}", flush=True)
    if patience_count >= config.patience:
        print(f"Early stopping after {epoch} epochs.", flush=True)

    started = time.perf_counter()
    model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
    result = {"dataset": config.dataset, "split": config.split, "seed": config.seed,
              "smoke": config.smoke, "best_epoch": best_epoch, "stopped_epoch": epoch,
              "lambda_max": config.lambda_max, "discriminator_input": config.discriminator_input,
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
