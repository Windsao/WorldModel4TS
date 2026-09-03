FS=/lambda/nfs/wm4ts-fs
cd $FS
export WM4TS_DATA_DIR=$FS/data HF_HOME=$FS/hf_cache TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$FS/results/preprocessing
mkdir -p $OUT/validation $OUT/logs
e () {   # $1=dataset $2=horizon   -- config FROZEN from Stage C: period_matrix/xattn/std_clip_3
  for pt in 1 0; do
    tag=$([ $pt = 1 ] && echo pt || echo rand)
    python3 pilot/run_preprocess.py --dataset $1 --renderer period_matrix --readout xattn       --norm std_clip_3 --pretrained $pt --data-dir $FS/data --out-dir $OUT/validation       --horizon-steps $2 --max-ch 112 --stride 8 --ft-cap 40000       --epochs 5 --batch 16 --lr 1e-4 --seed 0 > $OUT/logs/E_$1_$tag.log 2>&1
    echo "DONE E $1 $tag rc=$?" >> $OUT/logs/_progress.txt
  done
}
