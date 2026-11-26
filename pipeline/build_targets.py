"""Step 6. Build routing supervision from the frozen backbone's outputs (Sec. 4.3).

For each query we find the candidate the backbone would select on each path
and its geodesic error to the ground truth:

* G3:        the candidate with the highest G3 image-GPS similarity among
             the zero-shot samples (ZS path) or among the RAG generations plus
             the top-1 retrieved coordinate (retrieval path).
* GeoRanker: the candidate with the highest GeoRanker reward among the first
             N zero-shot (ZS path) or the first K retrieved (retrieval path).

Saved next to the meta-features:
    target_dists.pt   [num_queries, 2]  error (km) of [ZS, Ret] selections
    target_sims.pt    [num_queries, 2]  backbone score of [ZS, Ret] selections
    target_binary.pt  [num_queries]     1 if the retrieval path is better

Example:
    python -m pipeline.build_targets --backbone g3 --dataset im2gps3k \
        --rank_k 15 --zs_k 5 --g3_ckpt checkpoints/g3.pth
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from geopy.distance import geodesic
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from geogater.paths import Workspace
from geogater.utils import get_query_set, read_csv_checked, safe_eval, safe_geodesic

ImageFile.LOAD_TRUNCATED_IMAGES = True
MISSING_DIST_KM = 20037.0
PER_SOURCE = 16  # max candidates per path for the G3 scorer
G3_ZS_POOL = 5  # G3 selects among the first 5 zero-shot samples


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------


def _valid_coord(lat, lon):
    return np.isfinite(lat) and np.isfinite(lon) and abs(lat) <= 90 and abs(lon) <= 180


def parse_zs_coords(row):
    coords = []
    for lat, lon in zip(safe_eval(row.get("latitudes")), safe_eval(row.get("longitudes"))):
        try:
            lat, lon = float(lat), float(lon)
        except Exception:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        coords.append((lat, lon))
        if len(coords) >= G3_ZS_POOL:
            break
    return coords


def parse_rag_coords(row):
    """Coordinates from the flat ``rag_coordinates`` list written by ``g3_rag``."""
    flat = safe_eval(row.get("rag_coordinates"))
    coords = []
    for i in range(0, len(flat) - 1, 2):
        try:
            lat, lon = float(flat[i]), float(flat[i + 1])
        except Exception:
            continue
        if _valid_coord(lat, lon):
            coords.append((lat, lon))
    return coords


def load_g3_candidates(args, ws, n):
    """Tensor [n, 2 (ZS, Ret), PER_SOURCE, 2 (lat, lon)], zero-padded."""
    zs = read_csv_checked(ws.zero_shot_csv(args.dataset), "zero-shot predictions")
    zs_list = [parse_zs_coords(zs.iloc[i]) for i in range(n)]

    rag_list = [[] for _ in range(n)]
    for k in args.rag_ks:
        df = read_csv_checked(ws.rag_csv(args.dataset, args.database, k), f"G3 RAG k={k}")
        for i in range(n):
            rag_list[i].extend(parse_rag_coords(df.iloc[i]))

    cand = read_csv_checked(ws.candidates_csv(args.dataset, args.database), "candidates")
    for i in range(n):
        top = safe_eval(cand.iloc[i]["candidate_gps_lis"])
        if top and isinstance(top[0], (list, tuple)) and len(top[0]) == 2:
            lat, lon = float(top[0][0]), float(top[0][1])
            if abs(lat) <= 90 and abs(lon) <= 180 and not (abs(lat) < 1e-8 and abs(lon) < 1e-8):
                rag_list[i].append((lat, lon))

    master = torch.zeros((n, 2, PER_SOURCE, 2), dtype=torch.float32)
    for i in range(n):
        for m, src in enumerate((zs_list, rag_list)):
            for k, (lat, lon) in enumerate(src[i][:PER_SOURCE]):
                master[i, m, k, 0] = lat
                master[i, m, k, 1] = lon
    return master


class QueryDataset(Dataset):
    def __init__(self, candidates, df_meta, qs, image_dir, vision_processor):
        self.candidates = candidates
        self.df = df_meta
        self.qs = qs
        self.image_dir = image_dir
        self.proc = vision_processor
        size = getattr(vision_processor, "size", None)
        edge = size.get("shortest_edge", 224) if isinstance(size, dict) else 224
        self.fallback = (edge, edge)

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, self.qs.image_file(row[self.qs.id_col]))
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", self.fallback)
        px = self.proc(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        gt = torch.tensor([row[self.qs.lat_col], row[self.qs.lon_col]], dtype=torch.float32)
        return px, self.candidates[idx].view(-1, 2), gt


@torch.no_grad()
def targets_g3(args, ws):
    from geogater.backbones.g3 import load_g3

    qs = get_query_set(args.dataset)
    df_meta = pd.read_csv(qs.csv_path(args.data_root))
    n = len(df_meta)
    candidates = load_g3_candidates(args, ws, n)
    model = load_g3(args.g3_ckpt, args.device)
    loader = DataLoader(
        QueryDataset(candidates, df_meta, qs, qs.image_dir(args.data_root), model.vision_processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_d, all_s = [], []
    for imgs, gps, gt in tqdm(loader, desc="Scoring candidates (G3)"):
        imgs, gps = imgs.to(args.device), gps.to(args.device)
        bsz = imgs.size(0)
        v = model.vision_projection_else_2(model.vision_projection(model.vision_model(imgs)[1]))
        v = v / v.norm(p=2, dim=-1, keepdim=True)
        l = model.location_projection_else(model.location_encoder(gps.view(-1, 2)))
        l = (l / l.norm(p=2, dim=-1, keepdim=True)).view(bsz, 2 * PER_SOURCE, -1)
        sim = torch.bmm(l, v.unsqueeze(2)).squeeze(2).view(bsz, 2, PER_SOURCE)

        gps4 = gps.view(bsz, 2, PER_SOURCE, 2)
        is_pad = (gps4[..., 0].abs() < 1e-4) & (gps4[..., 1].abs() < 1e-4)
        best_vals, best_idx = sim.masked_fill(is_pad, -float("inf")).max(dim=2)
        best = torch.gather(gps4, 2, best_idx[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)

        best, gt = best.cpu().numpy(), gt.numpy()
        d = np.zeros((bsz, 2), dtype=np.float32)
        for i in range(bsz):
            for m in range(2):
                lat, lon = best[i, m]
                d[i, m] = MISSING_DIST_KM if abs(lat) < 1e-4 else geodesic(tuple(gt[i]), (lat, lon)).km
        all_d.append(torch.tensor(d))
        all_s.append(best_vals.cpu())
    return torch.cat(all_d), torch.cat(all_s)


# ---------------------------------------------------------------------------
# GeoRanker
# ---------------------------------------------------------------------------


def _argmax_select(rewards, dists):
    if len(rewards) == 0:
        return -np.inf, np.nan
    j = int(np.argmax(rewards))
    return rewards[j], dists[j] if j < len(dists) else np.nan


def targets_georanker(args, ws):
    data = read_csv_checked(ws.rewards_csv(args.dataset, args.database), "GeoRanker rewards")
    pool_ret = min(data.columns.str.startswith("retrieval_reward_").sum(), args.num_retrieval)
    pool_zs = min(data.columns.str.startswith("zs_reward_").sum(), args.num_zs)
    use_ret, use_zs = min(args.rank_k, pool_ret), min(args.zs_k, pool_zs)

    dists, sims = [], []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Selecting (GeoRanker)"):
        gt = (row["LAT"], row["LON"])
        ret_gps = safe_eval(row["candidate_gps_lis"])[:pool_ret][:use_ret]
        zs_gps = safe_eval(row["zs_gps_lis"])[:pool_zs][:use_zs]
        ret_d = [safe_geodesic(gt, c) if isinstance(c, (list, tuple)) else np.nan for c in ret_gps]
        zs_d = [safe_geodesic(gt, c) if isinstance(c, (list, tuple)) else np.nan for c in zs_gps]
        rewards = safe_eval(row["all_rewards"])
        ret_r = rewards[:pool_ret][:use_ret]
        zs_r = rewards[pool_ret : pool_ret + pool_zs][:use_zs]
        ret_s, ret_sel = _argmax_select(ret_r, ret_d)
        zs_s, zs_sel = _argmax_select(zs_r, zs_d)
        dists.append([zs_sel, ret_sel])
        sims.append([zs_s, ret_s])
    return torch.tensor(dists, dtype=torch.float32), torch.tensor(sims, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", required=True, choices=["g3", "georanker"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--database", default="mp16", choices=["mp16", "osv5m"])
    p.add_argument("--rank_k", type=int, default=15, help="K (feature-dir key / GeoRanker pool).")
    p.add_argument("--zs_k", type=int, default=5, help="N (feature-dir key / GeoRanker pool).")
    p.add_argument("--num_retrieval", type=int, default=15, help="GeoRanker: scored retrieval pool.")
    p.add_argument("--num_zs", type=int, default=5, help="GeoRanker: scored zero-shot pool.")
    p.add_argument("--rag_ks", type=int, nargs="+", default=[5, 10, 15], help="G3: RAG runs.")
    p.add_argument("--g3_ckpt", default=None, help="G3: checkpoint used to score candidates.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--work_dir", default="./outputs")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ws = Workspace(args.work_dir, args.backbone)
    if args.backbone == "g3":
        if not args.g3_ckpt:
            p.error("--g3_ckpt is required for the G3 backbone")
        dists, sims = targets_g3(args, ws)
    else:
        dists, sims = targets_georanker(args, ws)

    # 1 = retrieval path is better. Tie-breaking follows each backbone's
    # original convention (G3: ties -> zero-shot, GeoRanker: ties -> retrieval).
    if args.backbone == "g3":
        binary = (dists[:, 0] > dists[:, 1]).long()
    else:
        binary = (dists[:, 1] <= dists[:, 0]).long()
    out_dir = ws.feature_dir(args.dataset, args.database, args.rank_k, args.zs_k)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(binary, os.path.join(out_dir, "target_binary.pt"))
    torch.save(dists, os.path.join(out_dir, "target_dists.pt"))
    torch.save(sims, os.path.join(out_dir, "target_sims.pt"))
    print(f"Saved targets {tuple(dists.shape)} to {out_dir} (retrieval better: {binary.float().mean():.3f})")


if __name__ == "__main__":
    main()
