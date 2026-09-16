#!/bin/bash
# train one Route-B arm, no chained evaluation. usage: train_only.sh <tag> <steps> <seed> [trainer flags]
TAG=$1; STEPS=$2; SEED=$3; shift 3
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate wm4ts
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers
export CUDA_VISIBLE_DEVICES=0
cd /home/mzh1800/WorldModel4TS; mkdir -p logs/route_b
python pilot/pretrain_route_b.py --arm vmae_full --steps $STEPS --batch 8 --accum 4 --lr 1e-4 --warmup 500 --wd 0 \
  --seed $SEED --workers 8 --save-every 2500 --val-every 500 --gpu 0 \
  --corpus-dir /nyx-storage1/hanliu/wm4ts/route_b/corpus_v2 --out /nyx-storage1/hanliu/wm4ts/route_b --tag $TAG "$@" \
  > logs/route_b/vmae_full$TAG.log 2>&1
echo "TRAIN_ONLY_DONE $TAG $(date)"
