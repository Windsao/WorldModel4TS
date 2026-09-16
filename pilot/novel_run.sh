#!/bin/bash
# novel-design pretraining run: 5k steps (or $STEPS), batch 32 x $ACC, corpus $CORPUS, extra trainer flags, then the 5-dataset x 3-horizon grid.
# usage: novel_run.sh <tag> [trainer flags...]   env: STEPS (5000) ACC (1) CORPUS (corpus) LR (1e-4) WD (0.05) SEED (0)
TAG=$1; shift
STEPS=${STEPS:-5000}; ACC=${ACC:-1}; CORPUS=${CORPUS:-corpus}; LR=${LR:-1e-4}; WD=${WD:-0.05}; SEED=${SEED:-0}
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate wm4ts
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers
cd /home/mzh1800/WorldModel4TS; mkdir -p logs/route_b
python pilot/pretrain_route_b.py --arm ${ARM:-vmae_full} --steps $STEPS --batch ${BATCH:-32} --accum $ACC --lr $LR --warmup 500 --wd $WD --seed $SEED --workers 8 --save-every 2500 --val-every 500 --gpu 0 --corpus-dir /nyx-storage1/hanliu/wm4ts/route_b/$CORPUS --out /nyx-storage1/hanliu/wm4ts/route_b --tag $TAG "$@" > logs/route_b/${ARM:-vmae_full}$TAG.log 2>&1
R=/nyx-storage1/hanliu/wm4ts/route_b/${ARM:-vmae_full}$TAG
if [ -f $R/step_$STEPS/model.safetensors ]; then bash pilot/story_grid2.sh 0 $R/step_$STEPS st$TAG; else echo "NO_CKPT $TAG"; fi
echo "NOVEL_DONE $TAG $(date)"
