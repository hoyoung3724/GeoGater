"""Step 3. Materialize retrieval candidates (H_Ret) and zero-shot coordinates (H_ZS).

Writes one CSV row per query with the top-K retrieved database entries
(coordinates, textual metadata, image ids), the K most dissimilar entries and
the zero-shot coordinates. This table is the input of the GeoRanker reward
step and of meta-feature extraction.

Example:
    python -m pipeline.build_candidates --backbone g3 --dataset im2gps3k
"""

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from geogater.paths import Workspace, ensure_parent
from geogater.utils import get_database, get_query_set, read_csv_checked, safe_eval

try:
    import pycountry
except ImportError:  # optional, only used to expand OSV5M country codes
    pycountry = None

NONSTANDARD_COUNTRY_CODES = {"UK": "United Kingdom", "XK": "Kosovo"}


def country_name(value):
    if not isinstance(value, str) or len(value.strip()) != 2:
        return value
    code = value.strip().upper()
    if code in NONSTANDARD_COUNTRY_CODES:
        return NONSTANDARD_COUNTRY_CODES[code]
    if pycountry is None:
        return value
    country = pycountry.countries.get(alpha_2=code)
    return country.name if country is not None else value


def _pad(values, n, pad):
    return values + [pad] * (n - len(values))


def collect(db, db_spec, row_ids, n, country_map):
    gps, texts, imgs = [], [], []
    for row_idx in row_ids[:n]:
        try:
            row_idx = int(row_idx)
        except Exception:
            continue
        if not 0 <= row_idx < len(db):
            continue
        r = db.iloc[row_idx]
        gps.append((float(r[db_spec.lat_col]), float(r[db_spec.lon_col])))
        t = []
        for col in db_spec.text_cols:
            value = r[col]
            if col == "country" and country_map is not None:
                value = country_map.get(row_idx, value)
            t.append(str(value) if pd.notna(value) else "")
        texts.append(t)
        imgs.append(db_spec.image_key(r[db_spec.id_col]))
    return (
        _pad(gps, n, (0.0, 0.0)),
        _pad(texts, n, [""] * len(db_spec.text_cols)),
        _pad(imgs, n, ""),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", required=True, choices=["g3", "georanker"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--database", default="mp16", choices=["mp16", "osv5m"])
    p.add_argument("--retrieval_num", type=int, default=15, help="K candidates kept.")
    p.add_argument("--zs_num", type=int, default=5, help="N zero-shot samples kept.")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--work_dir", default="./outputs")
    args = p.parse_args()

    ws = Workspace(args.work_dir, args.backbone)
    qs = get_query_set(args.dataset)
    db_spec = get_database(args.database)

    df = pd.read_csv(qs.csv_path(args.data_root))
    db = pd.read_csv(db_spec.csv_path(args.data_root))
    zs = read_csv_checked(ws.zero_shot_csv(args.dataset), "zero-shot predictions")
    I = np.load(ws.neighbors(args.database, args.dataset))
    I_inv = np.load(ws.inverse_neighbors(args.database, args.dataset))
    assert len(df) == len(zs) == len(I), "query / zero-shot / index rows differ"

    country_map = None
    if args.database == "osv5m" and "country" in db.columns:
        needed = np.unique(
            np.concatenate(
                [I[:, : args.retrieval_num].ravel(), I_inv[:, : args.retrieval_num].ravel()]
            )
        )
        needed = needed[(needed >= 0) & (needed < len(db))]
        country_map = dict(zip(needed, db.loc[needed, "country"].apply(country_name)))

    out = pd.DataFrame(
        {
            "IMG_ID": [qs.image_file(x) for x in df[qs.id_col]],
            "LAT": df[qs.lat_col].astype(float),
            "LON": df[qs.lon_col].astype(float),
        }
    )
    cols = {k: [] for k in ["candidate", "reverse"]}
    zs_col = []
    for i in tqdm(range(len(df)), desc="Building candidates"):
        cols["candidate"].append(collect(db, db_spec, I[i], args.retrieval_num, country_map))
        cols["reverse"].append(collect(db, db_spec, I_inv[i], args.retrieval_num, country_map))

        zs_gps = []
        for lat, lon in zip(
            safe_eval(zs.iloc[i]["latitudes"]), safe_eval(zs.iloc[i]["longitudes"])
        ):
            if lat is None or lon is None:
                lat, lon = 0.0, 0.0
            lat, lon = float(lat), float(lon)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                lat, lon = 0.0, 0.0
            zs_gps.append((lat, lon))
        zs_col.append(str(_pad(zs_gps, args.zs_num, (0.0, 0.0))))

    for name in ["candidate", "reverse"]:
        gps, texts, imgs = zip(*cols[name])
        out[f"{name}_gps_lis"] = [str(x) for x in gps]
        out[f"{name}_text_lis"] = [str(x) for x in texts]
        out[f"{name}_img_lis"] = [str(x) for x in imgs]
    out["zs_gps_lis"] = zs_col

    path = ensure_parent(ws.candidates_csv(args.dataset, args.database))
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
