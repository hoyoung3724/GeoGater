"""Shared helpers: dataset registry, safe parsing, geodesic distance, metrics."""

import ast
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from geopy.distance import geodesic

# Distance (km) used whenever a prediction is missing or invalid.
INVALID_DIST_KM = 20000.0
RECALL_THRESHOLDS_KM = (1, 25, 200, 750, 2500)


# ---------------------------------------------------------------------------
# Dataset / database registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySet:
    """An evaluation / training query set stored as ``<root>/<name>/``."""

    name: str
    id_col: str = "IMG_ID"
    lat_col: str = "LAT"
    lon_col: str = "LON"
    # Some sets store ids without the ``.jpg`` extension.
    append_jpg: bool = False

    def csv_path(self, data_root: str) -> str:
        return os.path.join(data_root, self.name, f"{self.name}_places365.csv")

    def image_dir(self, data_root: str) -> str:
        return os.path.join(data_root, self.name, f"{self.name}test")

    def image_file(self, image_id) -> str:
        image_id = str(image_id)
        if self.append_jpg and not image_id.lower().endswith(".jpg"):
            image_id += ".jpg"
        return image_id


@dataclass(frozen=True)
class Database:
    """A geo-tagged reference database stored as ``<root>/<name>/``."""

    name: str
    id_col: str
    lat_col: str
    lon_col: str
    text_cols: List[str] = field(default_factory=list)
    append_jpg: bool = False

    def csv_path(self, data_root: str) -> str:
        return os.path.join(data_root, self.name, f"{self.name}.csv")

    def tar_path(self, data_root: str) -> str:
        return os.path.join(data_root, self.name, f"{self.name}-images.tar")

    def tar_index_path(self, data_root: str) -> str:
        return os.path.join(data_root, self.name, "tar_index.pkl")

    def image_key(self, image_id) -> str:
        image_id = str(image_id)
        if self.append_jpg and not image_id.lower().endswith(".jpg"):
            image_id += ".jpg"
        return image_id


QUERY_SETS = {
    "im2gps3k": QuerySet("im2gps3k"),
    "yfcc4k": QuerySet("yfcc4k"),
    "yfcc26k": QuerySet("yfcc26k"),
    # OSV5M-derived street-view splits (5,000 images each, Supp. B).
    # osv5k = OSV5K-Balanced (grid-stratified), train5k = OSV5K-Random.
    "osv5k": QuerySet("osv5k", "id", "latitude", "longitude", append_jpg=True),
    "train5k": QuerySet("train5k", "id", "latitude", "longitude", append_jpg=True),
}

DATABASES = {
    "mp16": Database("mp16", "IMG_ID", "LAT", "LON", ["city", "state", "country"]),
    "osv5m": Database(
        "osv5m",
        "id",
        "latitude",
        "longitude",
        ["city", "region", "country"],
        append_jpg=True,
    ),
}


def get_query_set(name: str) -> QuerySet:
    if name not in QUERY_SETS:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {sorted(QUERY_SETS)}")
    return QUERY_SETS[name]


def get_database(name: str) -> Database:
    if name not in DATABASES:
        raise ValueError(f"Unknown database '{name}'. Choices: {sorted(DATABASES)}")
    return DATABASES[name]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def safe_eval(val) -> list:
    """Parse a stringified Python list (as stored in CSVs); return [] on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def safe_geodesic(p1, p2) -> float:
    """Geodesic distance in km; (0, 0) / None coordinates count as invalid."""
    if p1 == (0.0, 0.0) or p2 == (0.0, 0.0):
        return INVALID_DIST_KM
    if any(x is None for x in tuple(p1) + tuple(p2)):
        return INVALID_DIST_KM
    try:
        return geodesic(p1, p2).km
    except Exception:
        return INVALID_DIST_KM


def is_valid_retrieval_candidate(coord, img_id) -> bool:
    """A retrieved DB entry is valid if it has an image id and a non-(0,0) coordinate."""
    if img_id is None or not str(img_id).strip():
        return False
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return False
    try:
        lat, lon = float(coord[0]), float(coord[1])
    except Exception:
        return False
    return not (abs(lat) < 1e-8 and abs(lon) < 1e-8)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


METRIC_KEYS = ["R@1", "R@25", "R@200", "R@750", "R@2500", "Med", "Mean"]


def recall_stats(errors_km) -> dict:
    """Recall@{1,25,200,750,2500} km (in %), plus median / mean error (km)."""
    arr = np.asarray(errors_km, dtype=np.float64)
    if arr.size == 0:
        return {k: 0.0 for k in METRIC_KEYS}
    stats = {f"R@{t}": float(np.mean(arr <= t) * 100) for t in RECALL_THRESHOLDS_KM}
    stats["Med"] = float(np.median(arr))
    stats["Mean"] = float(np.mean(arr))
    return stats


def read_csv_checked(path: str, what: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        hint = f" ({what})" if what else ""
        raise FileNotFoundError(f"Missing file{hint}: {path}")
    return pd.read_csv(path)
