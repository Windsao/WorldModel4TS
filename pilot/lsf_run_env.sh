#!/bin/bash
# usage: lsf_run.sh <gpu> <dataset> <H> <model path|visionts> <tag> [extra args]
GPU=$1; DS=$2; H=$3; MODEL=$4; TAG=$5; shift 5
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate ${WM4TS_ENV:-wm4ts}
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers VISIONTS_CKPT=/nyx-storage1/hanliu/wm4ts/ckpt
cd /home/mzh1800/WorldModel4TS; mkdir -p logs/lsf
CUDA_VISIBLE_DEVICES=$GPU python pilot/eval_lsf.py --dataset $DS --pred-len $H --model $MODEL --tag $TAG "$@" > logs/lsf/${DS}_H${H}_${TAG}.log 2>&1
