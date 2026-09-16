#!/bin/bash
# V-JEPA 2 world model with the full recipe, then the 5-dataset x 3-horizon grid. usage: jepa_run.sh <tag> [trainer flags]
# env: STEPS (20000) CORPUS (corpus_v2) LR (1e-4) WD (0) SEED (0) BATCH (32) MICRO (4)
TAG=$1; shift
STEPS=${STEPS:-20000}; CORPUS=${CORPUS:-corpus_v2}; LR=${LR:-1e-4}; WD=${WD:-0}; SEED=${SEED:-0}; BATCH=${BATCH:-32}; MICRO=${MICRO:-4}
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate wm4ts_jepa
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers
cd /home/mzh1800/WorldModel4TS; mkdir -p logs/route_j
python pilot/pretrain_route_j.py --arm ${ARM:-vjepa2_wm} --steps $STEPS --batch $BATCH --micro $MICRO --lr $LR --warmup 500 --wd $WD --seed $SEED --workers 8 --save-every 2500 --val-every 500 --gpu 0 --grad-ckpt --corpus-dir /nyx-storage1/hanliu/wm4ts/route_b/$CORPUS --out /nyx-storage1/hanliu/wm4ts/route_j --tag $TAG "$@" > logs/route_j/${ARM:-vjepa2_wm}$TAG.log 2>&1
R=/nyx-storage1/hanliu/wm4ts/route_j/${ARM:-vjepa2_wm}$TAG
if [ -f $R/step_$STEPS/world_model.pt ]; then WM4TS_ENV=wm4ts_jepa bash pilot/story_grid_env.sh 0 $R/step_$STEPS stj${ARM#vjepa2_wm}$TAG; else echo "NO_CKPT $TAG"; fi
echo "JEPA_DONE $TAG $(date)"
