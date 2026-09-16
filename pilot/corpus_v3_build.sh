#!/bin/bash
# corpus v3 = v2 + shards of the big non-climate subsets: buildings_900k (4 shards), LargeST 2019+2021, Q-TRAFFIC, azure (2 shards)
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate wm4ts
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub
cd /home/mzh1800/WorldModel4TS
V3=/nyx-storage1/hanliu/wm4ts/lotsa_v3; mkdir -p $V3
set -f
python -m huggingface_hub.commands.huggingface_cli download Salesforce/lotsa_data --repo-type dataset --include "buildings_900k/data-0000[0-3]-of-*" "largest_2019/*" "largest_2021/*" "Q-TRAFFIC/*" "azure_vm_traces_2017/data-0000[0-1]-of-*" --local-dir $V3 2>&1 | tail -n 2
set +f
for d in /nyx-storage1/hanliu/wm4ts/lotsa_v2/*; do b=$(basename $d); [ -e $V3/$b ] || ln -s $(readlink -f $d) $V3/$b; done
echo "v3 dirs: $(ls $V3 | wc -l)  new real dirs: $(find $V3 -maxdepth 1 -mindepth 1 -type d ! -name .cache | wc -l)  size: $(du -shL $V3 | cut -f1)"
[ $(find $V3 -maxdepth 1 -mindepth 1 -type d ! -name .cache | wc -l) -ge 5 ] || { echo "DOWNLOAD_INCOMPLETE"; exit 1; }
python pilot/pretrain_route_b.py --build-corpus --no-domain-exclude --lotsa-dir $V3 --corpus-dir /nyx-storage1/hanliu/wm4ts/route_b/corpus_v3 2>&1 | grep "buildings\|largest\|Q-TRAFFIC\|azure\|total"
echo "CORPUS_V3_DONE $(date)"
