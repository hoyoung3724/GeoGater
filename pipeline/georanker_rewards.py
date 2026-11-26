"""Step 4b (GeoRanker backbone). Score every candidate with the GeoRanker reward model.

This is GeoRanker's downstream LVLM stage (one reward call per candidate),
i.e. the stage GeoGater skips for the non-selected path. Rewards are computed
once for the full pool (``--topn`` retrieval + ``--topn_zs`` zero-shot
candidates); the (K, N) used by GeoGater are applied later in
``pipeline.build_targets``. Prompts follow GeoRanker
(https://github.com/Applied-Machine-Learning-Lab/GeoRanker, Apache-2.0).

Example:
    python -m pipeline.georanker_rewards --dataset im2gps3k \
        --lora_path checkpoints/georanker_lora
"""

import argparse
import base64
import io
import os
import pickle
import tarfile

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from geogater.paths import Workspace, ensure_parent
from geogater.utils import get_database, get_query_set, read_csv_checked, recall_stats, safe_geodesic

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_to_data_url(image, max_side):
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.BICUBIC)
    if image.mode != "RGB":
        image = image.convert("RGB")
    with io.BytesIO() as buf:
        image.save(buf, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


class MessageBuilder:
    """Builds the GeoRanker prompt for (query i, candidate slot j) on demand."""

    def __init__(self, args, data):
        db = get_database(args.database)
        qs = get_query_set(args.dataset)
        with open(db.tar_index_path(args.data_root), "rb") as f:
            self.tar_index = pickle.load(f)
        self.tar = tarfile.open(db.tar_path(args.data_root))
        self.image_dir = qs.image_dir(args.data_root)
        self.data = data
        self.topn, self.topn_zs = args.topn, args.topn_zs
        self.max_side = args.max_image_side
        self.blank = pil_to_data_url(Image.new("RGB", (224, 224)), 224)
        self._cache_i, self._cache = None, None

    def _sample(self, i):
        if self._cache_i == i:
            return self._cache
        row = self.data.iloc[i]
        gps = np.array(eval(row["candidate_gps_lis"]))
        texts = np.array(eval(row["candidate_text_lis"]))
        imgs = np.array(eval(row["candidate_img_lis"]))
        zs_gps = np.array(eval(row["zs_gps_lis"]))[: self.topn_zs]
        with Image.open(os.path.join(self.image_dir, row["IMG_ID"])) as im:
            query = pil_to_data_url(im, self.max_side)
        # The 5 lowest-ranked retrieved candidates act as negative examples.
        negs = [
            f"latitude: {g[0]}, longitude: {g[1]}, {' '.join([t for t in tx if t])}"
            for g, tx in zip(gps[-5:], texts[-5:])
        ]
        self._cache_i = i
        self._cache = dict(
            query=query,
            gps=gps[: self.topn],
            texts=texts[: self.topn],
            imgs=imgs[: self.topn],
            zs_gps=zs_gps,
            negs="; ".join(negs),
        )
        return self._cache

    def _ref_image(self, img_id):
        if img_id and img_id in self.tar_index:
            f = self.tar.extractfile(self.tar_index[img_id])
            with Image.open(f) as im:
                with io.BytesIO() as buf:
                    im.save(buf, format="JPEG")
                    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        return self.blank

    def ref_gps(self, i):
        s = self._sample(i)
        return list(s["gps"]) + list(s["zs_gps"])

    def message(self, flat_idx):
        per = self.topn + self.topn_zs
        i, j = divmod(flat_idx, per)
        s = self._sample(i)
        if j < self.topn:
            g = s["gps"][j]
            ref_texts = " ".join([t for t in s["texts"][j] if t])
            content = [
                {"type": "image", "image": s["query"]},
                {
                    "type": "text",
                    "text": f"How far is this place from latitude: {g[0]}, longitude: {g[1]}, {ref_texts}?",
                },
                {"type": "image", "image": self._ref_image(s["imgs"][j])},
                {"type": "text", "text": f"Negative examples: {s['negs']}"},
            ]
        else:
            g = s["zs_gps"][j - self.topn]
            content = [
                {"type": "image", "image": s["query"]},
                {
                    "type": "text",
                    "text": f"How far is this place from latitude: {g[0]}, longitude: {g[1]}?",
                },
                {"type": "text", "text": f"Negative examples: {s['negs']}"},
            ]
        return [{"role": "user", "content": content}]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--database", default="mp16", choices=["mp16", "osv5m"])
    p.add_argument("--base_model", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--lora_path", default="checkpoints/georanker_lora")
    p.add_argument("--topn", type=int, default=15, help="Retrieval candidates scored.")
    p.add_argument("--topn_zs", type=int, default=5, help="Zero-shot candidates scored.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_image_side", type=int, default=512)
    p.add_argument("--no_flash_attn", action="store_true")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--work_dir", default="./outputs")
    args = p.parse_args()

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    from geogater.backbones.georanker import load_georanker

    ws = Workspace(args.work_dir, "georanker")
    data = read_csv_checked(ws.candidates_csv(args.dataset, args.database), "candidates")
    out_path = ensure_parent(ws.rewards_csv(args.dataset, args.database))
    ckpt_path = out_path + ".partial.pkl"

    model = load_georanker(
        args.base_model, args.lora_path, use_flash_attention_2=not args.no_flash_attn
    )
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)
    processor.tokenizer.padding_side = "left"
    builder = MessageBuilder(args, data)

    per = args.topn + args.topn_zs
    total = len(data) * per
    rewards = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            rewards = pickle.load(f)
        print(f"Resuming from {len(rewards)}/{total} rewards")

    bsz = args.batch_size
    start = (len(rewards) // bsz) * bsz
    rewards = rewards[:start]
    for s in tqdm(range(start, total, bsz), desc="GeoRanker rewards"):
        batch = [builder.message(j) for j in range(s, min(s + bsz, total))]
        text = processor.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(batch)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            r, _ = model(**inputs, return_output=True)
        rewards.extend(r.float().cpu().tolist())
        if (s // bsz) % 100 == 0:
            with open(ckpt_path, "wb") as f:
                pickle.dump(rewards, f)

    dists = []
    for i in range(len(data)):
        sample = rewards[i * per : (i + 1) * per]
        ref = builder.ref_gps(i)
        best = int(np.argmax(sample))
        for j in range(args.topn):
            data.loc[i, f"retrieval_reward_{j}"] = sample[j]
        for j in range(args.topn_zs):
            data.loc[i, f"zs_reward_{j}"] = sample[args.topn + j]
        data.loc[i, "all_rewards"] = str(sample)
        data.loc[i, "LAT_pred"], data.loc[i, "LON_pred"] = ref[best][0], ref[best][1]
        dists.append(safe_geodesic((data.loc[i, "LAT"], data.loc[i, "LON"]), tuple(ref[best])))
    data["geodesic"] = dists
    data.to_csv(out_path, index=False)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("GeoRanker (all candidates):", recall_stats(dists))
    print(f"Saved rewards to {out_path}")


if __name__ == "__main__":
    main()
