#!/bin/bash
# corpus v2 = our 54 LOTSA dirs + the previously excluded energy / traffic domains (+ small leftovers); no buildings_900k / largest / Q-TRAFFIC / cmip6 / era5 / azure
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate wm4ts
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub
cd /home/mzh1800/WorldModel4TS
V2=/nyx-storage1/hanliu/wm4ts/lotsa_v2; mkdir -p $V2
NEW="residential_load_power residential_pv_power wind_farms_with_missing london_smart_meters_with_missing LOOP_SEATTLE PEMS07 PEMS_BAY PEMS04 traffic_hourly lcl PEMS03 PEMS08 solar_power wind_power LOS_LOOP bdg-2_rat bdg-2_fox bdg-2_bear bdg-2_panther gfc12_load gfc14_load gfc17_load australian_electricity_demand covid19_energy traffic_weekly elecdemand bull hog cockatoo ideal smart spain borealis sceaux elf pdb M_DENSE m4_quarterly m4_yearly SZ_TAXI tourism_yearly monash_m3_yearly monash_m3_other m1_yearly"
INC=""; for d in $NEW; do INC="$INC $d/*"; done
set -f; python -m huggingface_hub.commands.huggingface_cli download Salesforce/lotsa_data --repo-type dataset --include $INC --local-dir $V2 2>&1 | tail -n 3; set +f
for d in /nyx-storage1/hanliu/wm4ts/lotsa/*; do b=$(basename $d); [ -e $V2/$b ] || ln -s $d $V2/$b; done
echo "v2 dirs: $(ls $V2 | wc -l)  real dirs: $(find $V2 -maxdepth 1 -mindepth 1 -type d ! -name .cache | wc -l)  size: $(du -shL $V2 | cut -f1)"
[ $(find $V2 -maxdepth 1 -mindepth 1 -type d ! -name .cache | wc -l) -ge 30 ] || { echo "DOWNLOAD_INCOMPLETE"; exit 1; }
python pilot/pretrain_route_b.py --build-corpus --no-domain-exclude --lotsa-dir $V2 --corpus-dir /nyx-storage1/hanliu/wm4ts/route_b/corpus_v2 2>&1 | tail -n 15
ls -la /nyx-storage1/hanliu/wm4ts/route_b/corpus_v2/ && echo "CORPUS_V2_DONE $(date)"
