"""Meta-feature computation (Sec. 4.2).

Given per-query scalar signals from the two inference paths, build the unified
meta-feature vector F = [F_ZS, F_Ret]:

Zero-shot proxies (F_ZS)
  * verbalized confidence statistics                     -> ``conf_*``
  * query-image / ZS-coordinate similarity statistics    -> ``zs_align*``
Retrieval proxies (F_Ret)
  * image / text / coordinate similarity statistics      -> ``img*``, ``text*``, ``gps*``
  * convergence score (sum of the three similarities)    -> ``cand_sum*``
  * cluster consensus (DBSCAN noise ratio, entropy,
    dominance ratio) and spatial compactness
    (log pairwise / nearest-neighbour distances)          -> ``cluster_*``, ``*_log``

For the G3 backbone the reference-image and metadata similarities are masked
out (``g3_mask``), since G3 does not use them.
"""

import itertools
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .utils import INVALID_DIST_KM, safe_geodesic

KMS_PER_RADIAN = 6371.0088


def _stat_names(prefix: str) -> List[str]:
    return [f"{prefix}_max", f"{prefix}_median", f"{prefix}_mean", f"{prefix}_std"]


def _z_names(prefix: str) -> List[str]:
    return [f"{prefix}_max", f"{prefix}_median"]


SPATIAL_NAMES = [
    "cluster_noise_ratio",
    "cluster_entropy",
    "cluster_max_ratio",
    "pairwise_max_log",
    "pairwise_mean_log",
    "nn_max_log",
    "nn_mean_log",
]

# (names, kept_for_g3) in the exact column order of ``meta_feats.pt``.
FEATURE_GROUPS: List[Tuple[List[str], bool]] = [
    (_stat_names("conf"), True),
    (_stat_names("zs_align"), True),
    (["zs_align_z_max", "zs_align_z_median"], True),
    (_stat_names("img"), False),
    (_z_names("img_z"), False),
    (_stat_names("text"), False),
    (_z_names("text_z"), False),
    (_stat_names("gps"), True),
    (_z_names("gps_z"), True),
    (_stat_names("cand_sum"), False),
    (_z_names("cand_sum_z"), False),
    (SPATIAL_NAMES, True),
]
FEATURE_NAMES: List[str] = [n for names, _ in FEATURE_GROUPS for n in names]
G3_MASK: List[float] = [
    1.0 if keep else 0.0 for names, keep in FEATURE_GROUPS for _ in names
]


def stats4(arr) -> List[float]:
    """[max, median, mean, std] of a score set (zeros if empty)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.max(arr)),
        float(np.median(arr)),
        float(np.mean(arr)),
        float(np.std(arr)),
    ]


def zscore(arr, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    return (arr - float(np.mean(arr))) / (float(np.std(arr)) + eps)


def z_max_median(arr) -> List[float]:
    """[max, median] of the within-query z-standardized scores."""
    z = zscore(arr)
    if len(z) == 0:
        return [0.0, 0.0]
    return [float(np.max(z)), float(np.median(z))]


def aggregate(arr) -> List[float]:
    """Statistical aggregation T(S) of Eq. (4)."""
    return stats4(arr) + z_max_median(arr)


def spatial_consistency(coords: Sequence, epsilon_km: float = 25.0) -> List[float]:
    """Cluster consensus + spatial compactness of the retrieved coordinates.

    Returns [noise_ratio, cluster_entropy, dominance_ratio,
             log1p(max pairwise), log1p(mean pairwise),
             log1p(max NN dist), log1p(mean NN dist)].
    """
    valid = [
        (float(c[0]), float(c[1]))
        for c in coords
        if c
        and c[0] is not None
        and c[1] is not None
        and not (c[0] == 0 and c[1] == 0)
    ]
    if len(valid) == 0:
        return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if len(valid) == 1:
        return [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    coords_rad = np.radians(np.array(valid))
    db = DBSCAN(
        eps=epsilon_km / KMS_PER_RADIAN,
        min_samples=2,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = db.fit_predict(coords_rad)
    noise_ratio = list(labels).count(-1) / len(labels)
    if np.any(labels >= 0):
        counts = np.bincount(labels[labels >= 0])
        dominance = np.max(counts) / len(labels)
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
    else:
        dominance = 0.0
        entropy = 1.0

    pw = np.array(
        [safe_geodesic(p1, p2) for p1, p2 in itertools.combinations(valid, 2)],
        dtype=np.float32,
    )
    nn_dists = []
    for i, p1 in enumerate(valid):
        best = INVALID_DIST_KM
        for j, p2 in enumerate(valid):
            if i != j:
                best = min(best, safe_geodesic(p1, p2))
        nn_dists.append(best if best < INVALID_DIST_KM else 0.0)
    nn_dists = np.array(nn_dists, dtype=np.float32)

    return [
        float(noise_ratio),
        float(entropy),
        float(dominance),
        float(np.log1p(float(np.max(pw)))),
        float(np.log1p(float(np.mean(pw)))),
        float(np.log1p(float(np.max(nn_dists)))),
        float(np.log1p(float(np.mean(nn_dists)))),
    ]


def build_meta_feature_vector(
    confidences,
    zs_sims,
    img_sims,
    text_sims,
    gps_sims,
    retrieved_coords,
    epsilon_km: float = 25.0,
) -> List[float]:
    """Assemble one query's meta-feature vector in ``FEATURE_NAMES`` order."""
    img_sims = np.asarray(img_sims, dtype=np.float32)
    text_sims = np.asarray(text_sims, dtype=np.float32)
    gps_sims = np.asarray(gps_sims, dtype=np.float32)
    convergence = img_sims + text_sims + gps_sims

    row = []
    row += stats4(confidences)  # verbalized confidence (normalized to [0, 1])
    row += aggregate(zs_sims)  # ZS coordinate similarity
    row += aggregate(img_sims)  # reference image similarity
    row += aggregate(text_sims)  # metadata text similarity
    row += aggregate(gps_sims)  # retrieved coordinate similarity
    row += aggregate(convergence)  # convergence score
    row += spatial_consistency(retrieved_coords, epsilon_km)
    assert len(row) == len(FEATURE_NAMES)
    return row
