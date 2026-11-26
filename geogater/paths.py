"""On-disk layout of every intermediate artifact produced by the pipeline.

All artifacts live under a single work directory (default: ``./outputs``)::

    <work_dir>/
      zero_shot/<dataset>.csv                         # LVLM zero-shot samples (shared)
      <backbone>/index/I_<db>_<dataset>.npy           # top-100 retrieved DB rows
      <backbone>/index/I_<db>_<dataset>_inverse.npy   # top-100 most dissimilar DB rows
      <backbone>/candidates/<dataset>_<db>.csv        # retrieval candidates (+ZS coords)
      g3/rag/<dataset>_<db>_k<k>.csv                  # G3 RAG generations (k = 5, 10, 15)
      georanker/rewards/<dataset>_<db>.csv            # GeoRanker rewards per candidate
      <backbone>/features/<dataset>_<db>_rk<K>_zs<N>/ # meta-features + routing targets
      <backbone>/checkpoints/                         # trained routers

``<backbone>`` is ``g3`` or ``georanker``. The two backbones use different G3
retrievers, hence separate index / candidate folders.
"""

import os

BACKBONES = ("g3", "georanker")


class Workspace:
    def __init__(self, work_dir: str, backbone: str):
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone}")
        self.root = work_dir
        self.backbone = backbone
        self.backbone_dir = os.path.join(work_dir, backbone)

    # -- shared -----------------------------------------------------------
    def zero_shot_csv(self, dataset: str) -> str:
        return os.path.join(self.root, "zero_shot", f"{dataset}.csv")

    # -- retrieval index --------------------------------------------------
    @property
    def index_dir(self) -> str:
        return os.path.join(self.backbone_dir, "index")

    def faiss_index(self, database: str) -> str:
        return os.path.join(self.index_dir, f"{database}.index")

    def neighbors(self, database: str, dataset: str) -> str:
        return os.path.join(self.index_dir, f"I_{database}_{dataset}.npy")

    def inverse_neighbors(self, database: str, dataset: str) -> str:
        return os.path.join(self.index_dir, f"I_{database}_{dataset}_inverse.npy")

    def distances(self, database: str, dataset: str) -> str:
        return os.path.join(self.index_dir, f"D_{database}_{dataset}.npy")

    # -- candidates / backbone outputs -------------------------------------
    def candidates_csv(self, dataset: str, database: str) -> str:
        return os.path.join(self.backbone_dir, "candidates", f"{dataset}_{database}.csv")

    def rag_csv(self, dataset: str, database: str, k: int) -> str:
        return os.path.join(self.root, "g3", "rag", f"{dataset}_{database}_k{k}.csv")

    def rewards_csv(self, dataset: str, database: str) -> str:
        return os.path.join(self.root, "georanker", "rewards", f"{dataset}_{database}.csv")

    # -- GeoGater ---------------------------------------------------------
    def feature_dir(self, dataset: str, database: str, rank_k: int, zs_k: int) -> str:
        return os.path.join(
            self.backbone_dir, "features", f"{dataset}_{database}_rk{rank_k}_zs{zs_k}"
        )

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.backbone_dir, "checkpoints")


def ensure_parent(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path
