#!/usr/bin/env bash
# Full GeoGater pipeline on top of the GeoRanker backbone (MP16-Pro database).
# Requires OPENAI_API_KEY for step 2 (skip it if zero_shot/<dataset>.csv already exists).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
WORK=${WORK:-outputs}
DEVICE=${DEVICE:-cuda:0}
G3_CKPT=${G3_CKPT:-checkpoints/g3.pth}
LORA=${LORA:-checkpoints/georanker_lora}
DATASETS=(yfcc26k im2gps3k yfcc4k)

# 1. Retrieval index (GeoRanker's G3 retriever)
python -m pipeline.build_index --backbone georanker --database mp16 --g3_ckpt "$G3_CKPT" \
  --datasets "${DATASETS[@]}" --data_root "$DATA" --work_dir "$WORK" --device "$DEVICE"

for ds in "${DATASETS[@]}"; do
  [ -f "$WORK/zero_shot/$ds.csv" ] || \
    python -m pipeline.zero_shot --dataset "$ds" --n_samples 5 --data_root "$DATA" --work_dir "$WORK"
  python -m pipeline.build_candidates --backbone georanker --dataset "$ds" --data_root "$DATA" --work_dir "$WORK"
  # 4b. GeoRanker rewards for all 15 retrieval + 5 zero-shot candidates
  python -m pipeline.georanker_rewards --dataset "$ds" --lora_path "$LORA" --data_root "$DATA" --work_dir "$WORK"
done

# 5-6. Meta-features and targets for each (K, N) setting used in the paper
feats () {  # dataset K N
  python -m pipeline.extract_features --backbone georanker --dataset "$1" --rank_k "$2" --zs_k "$3" \
    --g3_ckpt "$G3_CKPT" --data_root "$DATA" --work_dir "$WORK" --device "$DEVICE"
  python -m pipeline.build_targets --backbone georanker --dataset "$1" --rank_k "$2" --zs_k "$3" \
    --data_root "$DATA" --work_dir "$WORK"
}
feats yfcc26k 12 3; feats im2gps3k 12 3
feats yfcc26k 14 5; feats yfcc4k 14 5

# 7. Train on YFCC26K, evaluate
python train.py --backbone georanker --rank_k 12 --zs_k 3 --eval_dataset im2gps3k --work_dir "$WORK" --device "$DEVICE"
python train.py --backbone georanker --rank_k 14 --zs_k 5 --eval_dataset yfcc4k --work_dir "$WORK" --device "$DEVICE"
