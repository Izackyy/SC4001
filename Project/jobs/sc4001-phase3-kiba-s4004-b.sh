#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=200
#SBATCH --job-name=sc4001-phase3-kiba-s4004-b
#SBATCH --output=logs/sc4001-phase3-kiba-s4004-b/%x_%j.out
#SBATCH --error=logs/sc4001-phase3-kiba-s4004-b/%x_%j.err

# KIBA cold-target s4004, the hard end of the bracket: baseline against DANN at lambda 0.03,
# seed 4003, both arms at 80 epochs with early stopping disabled.
#
# Job B covers the remaining seed. It queues behind job A, since QoS normal permits only one
# GPU job at a time, and completes the three-seed set.
#
# Sizing: 80 epochs, measured at 29.3 s/epoch for the KIBA baseline and a projected 1.53x for
# DANN on a T4. The V100 here is faster, so about 1.3 h is expected against the 3.3 h wall
# requested.

set -euo pipefail

module load cuda/12.2
module load anaconda
source activate /tc1apps/2_conda_env/CZ4042_v6

cd "$HOME/sc4001"
export SC4001_OUTPUT_ROOT="$HOME/sc4001-data/results/sc4001-phase3-kiba-s4004"
export PYTHONUNBUFFERED=1
mkdir -p "$SC4001_OUTPUT_ROOT"

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.version.cuda, '|', torch.cuda.get_device_name(0))"

for seed in 4003; do
    echo "===== baseline kiba split_cold_target_s4004 seed${seed} ====="
    SC4001_OUTPUT_ROOT="$SC4001_OUTPUT_ROOT/baseline" \
        python src/train.py --dataset kiba --split split_cold_target_s4004 \
        --seed "$seed" --max-epochs 80 --patience 80 --resume

    echo "===== dann-lambda0.03 kiba split_cold_target_s4004 seed${seed} ====="
    SC4001_OUTPUT_ROOT="$SC4001_OUTPUT_ROOT/dann-lambda0.03" \
        python src/train_dann.py --dataset kiba --split split_cold_target_s4004 \
        --seed "$seed" --max-epochs 80 --patience 80 \
        --lambda-max 0.03 --discriminator-input protein --resume
done

echo "Job B complete. Seed 4003 done for both arms; the three-seed set is now complete."
