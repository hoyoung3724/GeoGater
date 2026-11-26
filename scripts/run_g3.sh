#!/usr/bin/env bash
# Full GeoGater pipeline on top of the G3 backbone (MP16-Pro database).
# Requires OPENAI_API_KEY for steps 2 and 4a.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
WORK=${WORK:-outputs}
DEVICE=${DEVICE:-cuda:0}
G3_CKPT=${G3_CKPT:-checkpoints/g3.pth}
DATASETS=(yfcc26k im2gps3k yfcc4k)
K=15; N=5

# 1. Retrieval index
python -m pipeline.build_index --backbone g3 --database mp16 --g3_ckpt "$G3_CKPT" \
  --datasets "${DATASETS[@]}" --data_root "$DATA" --work_dir "$WORK" --device "$DEVICE"

for ds in "${DATASETS[@]}"; do
  # 2. Zero-shot LVLM samples with verbalized confidence (shared by both backbones)
  python -m pipeline.zero_shot --dataset "$ds" --n_samples "$N" --data_root "$DATA" --work_dir "$WORK"
  # 3. Retrieval candidates
  python -m pipeline.build_candidates --backbone g3 --dataset "$ds" --data_root "$DATA" --work_dir "$WORK"
  # 4a. G3 retrieval-augmented generation (k = 5, 10, 15)
  python -m pipeline.g3_rag --dataset "$ds" --data_root "$DATA" --work_dir "$WORK"
  # 5. Meta-features
  python -m pipeline.extract_features --backbone g3 --dataset "$ds" --rank_k "$K" --zs_k "$N" \
    --g3_ckpt "$G3_CKPT" --data_root "$DATA" --work_dir "$WORK" --device "$DEVICE"
  # 6. Routing targets
  python -m pipeline.build_targets --backbone g3 --dataset "$ds" --rank_k "$K" --zs_k "$N" \
    --g3_ckpt "$G3_CKPT" --data_root "$DATA" --work_dir "$WORK" --device "$DEVICE"
done

# 7. Train on YFCC26K, evaluate
for ds in im2gps3k yfcc4k; do
  python train.py --backbone g3 --rank_k "$K" --zs_k "$N" --eval_dataset "$ds" --work_dir "$WORK" --device "$DEVICE"
done
