#!/usr/bin/env bash
# Reproduce the GeoGater rows of Table 1 (and the LVLM-call counts of Table 2)
# from the released meta-features in precomputed/.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-precomputed}
DEVICE=${DEVICE:-cuda:0}

echo "== G3 + GeoGater =="
for ds in im2gps3k yfcc4k; do
  python train.py --work_dir "$WORK" --backbone g3 --rank_k 15 --zs_k 5 --eval_dataset "$ds" --device "$DEVICE"
done

echo "== GeoRanker + GeoGater =="
python train.py --work_dir "$WORK" --backbone georanker --rank_k 12 --zs_k 3 --eval_dataset im2gps3k --device "$DEVICE"
python train.py --work_dir "$WORK" --backbone georanker --rank_k 14 --zs_k 5 --eval_dataset yfcc4k --device "$DEVICE"
