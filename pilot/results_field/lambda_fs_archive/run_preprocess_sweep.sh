#!/bin/bash
# Stage C/D/E launcher for PREPROCESSING_EXPERIMENTS.md.
# Stage C: 6 renderers x {ETTh2, electricity} x {pretrained, random} = 24 runs, xattn readout.
FS=${FS:-/lambda/nfs/wm4ts-fs}
cd $FS
export WM4TS_DATA_DIR=$FS/data HF_HOME=$FS/hf_cache TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$FS/results/preprocessing
mkdir -p $OUT/screen $OUT/logs

runpair () {   # $1=dataset $2=renderer
  for pt in 1 0; do
    tag=$([ $pt = 1 ] && echo pt || echo rand)
    python3 pilot/run_preprocess.py --dataset $1 --renderer $2 --readout xattn \
      --norm std_clip_3 --pretrained $pt --data-dir $FS/data --out-dir $OUT/screen \
      --horizon-steps 96 --max-ch 112 --stride 32 --ft-cap 5000 \
      --epochs 2 --batch 16 --lr 1e-4 --seed 0 \
      > $OUT/logs/screenC_$1_$2_$tag.log 2>&1
    echo "DONE screenC $1 $2 $tag rc=$?" >> $OUT/logs/_progress.txt
  done
}
export -f runpair
