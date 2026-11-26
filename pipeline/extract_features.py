"""Step 5. Extract GeoGater meta-features for every query (Sec. 4.2).

Inputs: zero-shot samples (H_ZS) and retrieval candidates (H_Ret).
Outputs (in ``<work_dir>/<backbone>/features/<dataset>_<db>_rk<K>_zs<N>/``):
    meta_feats.pt            float tensor [num_queries, 41]
    meta_feats_mask.pt       feature mask applied for the G3 backbone
    meta_feats_names.json    column names

Example:
    python -m pipeline.extract_features --backbone georanker --dataset im2gps3k \
        --rank_k 12 --zs_k 3 --g3_ckpt checkpoints/g3.pth
"""

import argparse
import json
import os
import pickle
import tarfile

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from geogater.backbones.g3 import load_g3
from geogater.meta_features import FEATURE_NAMES, G3_MASK, build_meta_feature_vector
from geogater.paths import Workspace
from geogater.utils import (
    get_database,
    get_query_set,
    is_valid_retrieval_candidate,
    read_csv_checked,
    safe_eval,
    safe_float,
)

EMB_DIM = 768


class G3Encoder:
    """Cached G3 encoders for the similarity proxies s_m(I, .) of Eq. (3)."""

    def __init__(self, model, device, tar_path, tar_index_path):
        self.model = model
        self.device = device
        self.tar = tarfile.open(tar_path, "r")
        with open(tar_index_path, "rb") as f:
            self.tar_index = pickle.load(f)
        self.img_cache, self.txt_cache, self.coord_cache = {}, {}, {}

    @torch.no_grad()
    def _encode_image(self, pil_img):
        m = self.model
        px = m.vision_processor(images=pil_img, return_tensors="pt")["pixel_values"]
        base = m.vision_projection(m.vision_model(px.to(self.device))[1])
        img_loc = m.vision_projection_else_2(base)
        img_loc = img_loc / (img_loc.norm(dim=-1, keepdim=True) + 1e-8)
        img_txt = m.vision_projection_else_1(base)
        img_txt = img_txt / (img_txt.norm(dim=-1, keepdim=True) + 1e-8)
        return img_loc.squeeze(0).cpu().numpy(), img_txt.squeeze(0).cpu().numpy()

    def query(self, path):
        try:
            return self._encode_image(Image.open(path).convert("RGB"))
        except Exception:
            zero = np.zeros(EMB_DIM, dtype=np.float32)
            return zero, zero

    def reference_image(self, img_id):
        if img_id not in self.img_cache:
            try:
                f = self.tar.extractfile(self.tar_index[img_id])
                emb = self._encode_image(Image.open(f).convert("RGB"))[0]
            except Exception:
                emb = np.zeros(EMB_DIM, dtype=np.float32)
            self.img_cache[img_id] = emb
        return self.img_cache[img_id]

    @torch.no_grad()
    def text(self, text):
        if text not in self.txt_cache:
            if not text.strip():
                emb = np.zeros(EMB_DIM, dtype=np.float32)
            else:
                m = self.model
                tok = m.text_processor(text, return_tensors="pt", padding=True, truncation=True)
                tok = {k: v.to(self.device) for k, v in tok.items()}
                e = m.text_projection_else(m.text_projection(m.text_model(**tok)[1]))
                emb = (e / (e.norm(dim=-1, keepdim=True) + 1e-8)).squeeze(0).cpu().numpy()
            self.txt_cache[text] = emb
        return self.txt_cache[text]

    @torch.no_grad()
    def coord(self, coord):
        key = tuple(coord)
        if key not in self.coord_cache:
            loc = torch.tensor([coord], dtype=torch.float32, device=self.device)
            e = self.model.location_projection_else(self.model.location_encoder(loc))
            emb = (e / (e.norm(dim=-1, keepdim=True) + 1e-8)).squeeze(0).cpu().numpy()
            self.coord_cache[key] = emb
        return self.coord_cache[key]

    def close(self):
        self.tar.close()


def parse_zero_shot(row, zs_k):
    lats, lons = safe_eval(row.get("latitudes")), safe_eval(row.get("longitudes"))
    coords = [
        (safe_float(lat), safe_float(lon))
        for lat, lon in zip(lats, lons)
        if lat is not None and lon is not None
    ][:zs_k]
    # Verbalized confidences, normalized from [0, 100] to [0, 1]. Note: all
    # returned samples are aggregated (not truncated to zs_k), as in the paper runs.
    confs = []
    for c in safe_eval(row.get("confidence_choices")):
        v = safe_float(c)
        confs.append(v / 100.0 if v > 1.0 else v)
    return coords, confs


def parse_retrieval(row, rank_k):
    triples = [
        (c, t, i)
        for c, t, i in zip(
            safe_eval(row.get("candidate_gps_lis")),
            safe_eval(row.get("candidate_text_lis")),
            safe_eval(row.get("candidate_img_lis")),
        )
        if is_valid_retrieval_candidate(c, i)
    ]
    return triples[:rank_k]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", required=True, choices=["g3", "georanker"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--database", default="mp16", choices=["mp16", "osv5m"])
    p.add_argument("--rank_k", type=int, default=15, help="K retrieved candidates.")
    p.add_argument("--zs_k", type=int, default=5, help="N zero-shot hypotheses.")
    p.add_argument("--epsilon", type=float, default=25.0, help="DBSCAN radius (km).")
    p.add_argument("--g3_ckpt", required=True, help="G3 checkpoint used for similarities.")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--work_dir", default="./outputs")
    p.add_argument("--limit", type=int, default=None, help="Debug: first n queries only.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ws = Workspace(args.work_dir, args.backbone)
    qs = get_query_set(args.dataset)
    db = get_database(args.database)
    df_zs = read_csv_checked(ws.zero_shot_csv(args.dataset), "zero-shot predictions")
    df_ret = read_csv_checked(ws.candidates_csv(args.dataset, args.database), "candidates")
    n = len(df_ret) if args.limit is None else min(args.limit, len(df_ret))
    assert len(df_zs) >= n

    model = load_g3(args.g3_ckpt, args.device)
    enc = G3Encoder(model, args.device, db.tar_path(args.data_root), db.tar_index_path(args.data_root))
    query_dir = qs.image_dir(args.data_root)

    rows = []
    for i in tqdm(range(n), desc=f"Meta-features {args.dataset}"):
        zs_coords, confs = parse_zero_shot(df_zs.iloc[i].to_dict(), args.zs_k)
        r = df_ret.iloc[i].to_dict()
        candidates = parse_retrieval(r, args.rank_k)

        q_loc, q_txt = enc.query(os.path.join(query_dir, str(r["IMG_ID"])))
        zs_sims = []
        for c in zs_coords:
            try:
                zs_sims.append(float(np.dot(q_loc, enc.coord(c))))
            except Exception:
                zs_sims.append(0.0)
        img_sims = [float(np.dot(q_loc, enc.reference_image(img))) for _, _, img in candidates]
        text_sims = [
            float(np.dot(q_txt, enc.text(" ".join([t for t in texts if t]))))
            for _, texts, _ in candidates
        ]
        gps_sims = [float(np.dot(q_loc, enc.coord(c))) for c, _, _ in candidates]

        rows.append(
            build_meta_feature_vector(
                confs,
                zs_sims,
                img_sims,
                text_sims,
                gps_sims,
                [c for c, _, _ in candidates],
                epsilon_km=args.epsilon,
            )
        )
    enc.close()

    out_dir = ws.feature_dir(args.dataset, args.database, args.rank_k, args.zs_k)
    os.makedirs(out_dir, exist_ok=True)
    feats = torch.tensor(rows, dtype=torch.float32)
    torch.save(feats, os.path.join(out_dir, "meta_feats.pt"))
    torch.save(torch.tensor(G3_MASK, dtype=torch.float32), os.path.join(out_dir, "meta_feats_mask.pt"))
    with open(os.path.join(out_dir, "meta_feats_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)
    print(f"Saved meta-features {tuple(feats.shape)} to {out_dir}")


if __name__ == "__main__":
    main()
