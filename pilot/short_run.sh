#!/bin/bash
# fresh short cosine schedule (same recipe as the 20k baseline) + the three quick H=96 cells.
# usage: short_run.sh <steps> <tag> <gpu> [seed]   (GPU pinned via CUDA_VISIBLE_DEVICES, consistent with lsf_run.sh)
STEPS=$1; TAG=$2; G=$3; SEED=${4:-0}
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate wm4ts
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers
export CUDA_VISIBLE_DEVICES=$G
cd /home/mzh1800/WorldModel4TS; mkdir -p logs/route_b
python pilot/pretrain_route_b.py --arm vmae_full --steps $STEPS --batch 32 --lr 1e-4 --warmup 500 --wd 0.05 --seed $SEED --workers 6 --save-every 2500 --val-every 1000 --gpu 0 --corpus-dir /nyx-storage1/hanliu/wm4ts/route_b/corpus --out /nyx-storage1/hanliu/wm4ts/route_b --tag $TAG > logs/route_b/vmae_full$TAG.log 2>&1
R=/nyx-storage1/hanliu/wm4ts/route_b/vmae_full$TAG
ls $R/DONE && for ds in ETTh1 ETTh2 ETTm1; do pilot/lsf_run.sh $G $ds 96 $R/step_$STEPS vmf${TAG}_final --mode auto; done
echo "SHORT_RUN_DONE $TAG seed$SEED $(date)"
