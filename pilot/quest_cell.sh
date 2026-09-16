#!/bin/bash
# One LSF evaluation cell on Quest (gengpu). usage: sbatch quest_cell.sh <ckpt_dir> <tag> <dataset> <H> [extra eval args]
#SBATCH --account=p32294
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/projects/p32294/slurm_log/wm4ts_%x_%j.out
#SBATCH --error=/projects/p32294/slurm_log/wm4ts_%x_%j.err
set -eo pipefail
source ~/.bashrc
export PATH=/projects/p32294/conda_env/wm4ts_paper_foundations/bin:$PATH   # not a conda-meta env: use its python directly
set -u
W=/scratch/mzh1800/wm4ts_eval
export HF_HOME=/projects/p32294/hf_ckpt HF_HUB_CACHE=/projects/p32294/hf_ckpt/hub
export XDG_CACHE_HOME=/scratch/mzh1800/.cache TORCH_HOME=/projects/p32294/.cache/torch PYTHONNOUSERSITE=1
export VJEPA21_REPO=$W/vjepa2_repo
M=$1; T=$2; DS=$3; H=$4; shift 4
cd $W/code
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) python=$(which python) start=$(date)"
python -u pilot/eval_lsf.py --dataset "$DS" --pred-len "$H" --model "$M" --tag "$T" \
    --data-dir $W/data --out $W/lsf_results --mode auto "$@"
echo "CELL_DONE $DS $H $T $(date)"
