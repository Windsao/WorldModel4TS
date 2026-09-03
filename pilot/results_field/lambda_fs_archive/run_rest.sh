FS=/lambda/nfs/wm4ts-fs
cd $FS
export WM4TS_DATA_DIR=$FS/data HF_HOME=$FS/hf_cache CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fz () {
  python3 pilot/run_field.py --dataset $1 --data-dir "$WM4TS_DATA_DIR" --out-dir "$FS/results/frozen"     --mode uni --backbone video --render period --pretrained 1 --tune frozen     --horizon-steps $2 --max-ch 112 --stride 8 --ft-cap 40000     --epochs 5 --batch 16 --lr 1e-4 --seed 0 > $FS/logs/frozen_$1.log 2>&1
  echo "DONE frozen $1 rc=$?" >> $FS/logs/_progress.txt
}
rc () {
  python3 pilot/run_field.py --dataset electricity --data-dir "$WM4TS_DATA_DIR" --out-dir "$FS/results/random_control"     --mode uni --backbone video --render period --pretrained 0 --tune frozen     --context-steps 384 --horizon-steps 96 --max-ch 112 --stride 8 --ft-cap 40000     --epochs 5 --batch 16 --lr 1e-4 --seed 0 > $FS/logs/random_control.log 2>&1
  echo "DONE random_control rc=$?" >> $FS/logs/_progress.txt
}
