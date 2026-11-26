"""Step 1. Build a G3 FAISS index over the reference database and search it.

For every query image this stores the indices of the top-``topk`` most similar
database rows (``I_<db>_<dataset>.npy``) and of the ``topk`` most dissimilar
rows (``..._inverse.npy``, used as negative references by G3 RAG).
Indices refer to rows of ``<data_root>/<db>/<db>.csv``. Adapted from G3's index
search (https://github.com/Applied-Machine-Learning-Lab/G3, Apache-2.0).

Example:
    python -m pipeline.build_index --backbone g3 --database mp16 \
        --g3_ckpt checkpoints/g3.pth --datasets yfcc26k im2gps3k yfcc4k
"""

import argparse
import os
import pickle
import tarfile

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from geogater.backbones.g3 import encode_images, load_g3
from geogater.paths import Workspace, ensure_parent
from geogater.utils import get_database, get_query_set

ImageFile.LOAD_TRUNCATED_IMAGES = True
EMBED_DIM = 768 * 3  # [image, image->text, image->location]


class TarImageDataset(Dataset):
    """Reference-database images stored in a single tar file."""

    def __init__(self, data_root, database, vision_processor):
        db = get_database(database)
        df = pd.read_csv(db.csv_path(data_root))
        with open(db.tar_index_path(data_root), "rb") as f:
            self.tar_index = pickle.load(f)
        keys = df[db.id_col].map(db.image_key)
        keep = keys.isin(set(self.tar_index.keys())).to_numpy()
        if not keep.all():
            print(f"[warn] {(~keep).sum()} DB rows have no image in the tar; skipped.")
        # FAISS ids -> original CSV row index.
        self.row_ids = np.flatnonzero(keep)
        self.keys = keys[keep].tolist()
        self.tar_path = db.tar_path(data_root)
        self.tar = None
        self.vision_processor = vision_processor

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.tar is None:
            self.tar = tarfile.open(self.tar_path)
        image = Image.open(self.tar.extractfile(self.tar_index[self.keys[idx]]))
        image = image.convert("RGB")
        return self.vision_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)


class QueryImageDataset(Dataset):
    def __init__(self, data_root, dataset, vision_processor):
        qs = get_query_set(dataset)
        self.df = pd.read_csv(qs.csv_path(data_root))
        self.image_dir = qs.image_dir(data_root)
        self.qs = qs
        self.vision_processor = vision_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.qs.image_file(self.df.iloc[idx][self.qs.id_col])
        image = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        return self.vision_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)


@torch.no_grad()
def embed(model, loader, device):
    chunks = []
    for images in tqdm(loader, desc="Embedding"):
        base, img_txt, img_loc = encode_images(model, images.to(device))
        chunks.append(torch.cat([base, img_txt, img_loc], dim=1).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def build_index(args, model, ws):
    dataset = TarImageDataset(args.data_root, args.database, model.vision_processor)
    loader = DataLoader(
        dataset,
        batch_size=args.db_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    index = faiss.IndexFlatIP(EMBED_DIM)
    for images in tqdm(loader, desc=f"Indexing {args.database}"):
        base, img_txt, img_loc = encode_images(model, images.to(args.device))
        index.add(torch.cat([base, img_txt, img_loc], dim=1).cpu().numpy())
    faiss.write_index(index, ensure_parent(ws.faiss_index(args.database)))
    np.save(ws.faiss_index(args.database) + ".rows.npy", dataset.row_ids)
    return index, dataset.row_ids


def search(index, queries, topk, batch_size=64):
    all_d, all_i, all_neg = [], [], []
    for s in tqdm(range(0, len(queries), batch_size), desc="Searching"):
        q = queries[s : s + batch_size]
        d, i = index.search(q, topk)
        _, neg = index.search(-q, topk)
        all_d.append(d)
        all_i.append(i)
        all_neg.append(neg)
    return np.vstack(all_d), np.vstack(all_i), np.vstack(all_neg)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", required=True, choices=["g3", "georanker"])
    p.add_argument("--database", default="mp16", choices=["mp16", "osv5m"])
    p.add_argument("--datasets", nargs="+", default=["yfcc26k", "im2gps3k", "yfcc4k"])
    p.add_argument("--g3_ckpt", required=True, help="G3 retriever checkpoint.")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--work_dir", default="./outputs")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--db_batch_size", type=int, default=1024)
    p.add_argument("--query_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--use_gpu_index", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ws = Workspace(args.work_dir, args.backbone)
    model = load_g3(args.g3_ckpt, args.device)

    index_path = ws.faiss_index(args.database)
    if os.path.exists(index_path):
        print(f"Loading existing index: {index_path}")
        index = faiss.read_index(index_path)
        rows_path = index_path + ".rows.npy"
        if os.path.exists(rows_path):
            row_ids = np.load(rows_path)
        else:
            # e.g. the prebuilt g3.index from Jia-py/G3-checkpoint: FAISS ids are
            # the row order of <db>.csv.
            row_ids = np.arange(index.ntotal)
    else:
        index, row_ids = build_index(args, model, ws)

    if args.use_gpu_index:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)

    for ds in args.datasets:
        qset = QueryImageDataset(args.data_root, ds, model.vision_processor)
        loader = DataLoader(
            qset,
            batch_size=args.query_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        queries = embed(model, loader, args.device)
        D, I, I_neg = search(index, queries, args.topk)
        np.save(ensure_parent(ws.distances(args.database, ds)), D)
        to_rows = lambda ids: np.where(ids >= 0, row_ids[np.maximum(ids, 0)], -1)
        np.save(ws.neighbors(args.database, ds), to_rows(I))
        np.save(ws.inverse_neighbors(args.database, ds), to_rows(I_neg))
        print(f"[{ds}] saved neighbours {I.shape} to {ws.index_dir}")


if __name__ == "__main__":
    main()
