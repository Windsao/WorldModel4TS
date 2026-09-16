#!/bin/bash
# Parallel LSF evaluation: submit one slurm job per cell instead of one sequential job for the whole grid.
# Fast datasets (ETTh1/ETTh2) share one job per dataset; slow cells (ETTm*, weather, electricity) are split
# by --origin-range so their halves/quarters run on separate GPUs. Results merge by se/ae/cnt sums
# (pilot/final_bigtable.py, pilot/grid_table.py).
#
# usage: par_grid.sh <ckpt_dir> <result_tag> <conda_env> <grid5|full6>
#   grid5 = ETTh1 ETTh2 ETTm1 ETTm2 weather x H in {96,336,720}          (backbone comparison)
#   full6 = the same five datasets x {96,192,336,720} + electricity x 4   (paper table)
M=$1; T=$2; ENV=$3; WHAT=${4:-grid5}
cd /home/mzh1800/WorldModel4TS
[ -d "$M" ] || { echo "no checkpoint dir $M"; exit 1; }
if [ "$WHAT" = full6 ]; then HS="96 192 336 720"; else HS="96 336 720"; fi
SB="sbatch --parsable --gres=gpu:a40:1 --cpus-per-task=6 --mem=48G --time=2-00:00:00"
RUN="WM4TS_ENV=$ENV pilot/lsf_run_env.sh 0"
n=0
sub () { local name=$1; shift; $SB --job-name="pg_$name" -o "logs/lsf/pargrid_${T}_${name}.log" --wrap="$*; echo PG_DONE" > /dev/null; n=$((n+1)); }

for ds in ETTh1 ETTh2; do
  sub "${ds}" "for H in $HS; do $RUN $ds \$H $M $T --mode auto; done"
done
for ds in ETTm1 ETTm2; do
  for H in $HS; do
    sub "${ds}_H${H}_a" "$RUN $ds $H $M $T --mode auto --origin-range 0:5600"
    sub "${ds}_H${H}_b" "$RUN $ds $H $M $T --mode auto --origin-range 5600:99999"
  done
done
for H in $HS; do
  sub "weather_H${H}_a" "$RUN weather $H $M $T --mode auto --origin-range 0:5100"
  sub "weather_H${H}_b" "$RUN weather $H $M $T --mode auto --origin-range 5100:99999"
done
if [ "$WHAT" = full6 ]; then
  for H in $HS; do
    for r in 0:1300 1300:2600 2600:3900 3900:99999; do
      sub "electricity_H${H}_${r%%:*}" "$RUN electricity $H $M $T --mode auto --origin-range $r"
    done
  done
fi
echo "PAR_GRID_SUBMITTED tag=$T jobs=$n $(date)"
