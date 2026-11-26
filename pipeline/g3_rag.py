"""Step 4a (G3 backbone). Retrieval-augmented generation with the LVLM.

This is G3's downstream LVLM stage, i.e. the stage GeoGater skips for queries
routed to the zero-shot path. For each k in {5, 10, 15} the LVLM is prompted
with the coordinates of the k most similar and k most dissimilar database
images and sampled ``n=5`` times (15 RAG calls per query in total).

The prompt follows G3 (https://github.com/Applied-Machine-Learning-Lab/G3,
Apache-2.0) with its optional confidence field enabled, as in our runs.

Requires ``OPENAI_API_KEY``. The script is resumable.

Example:
    python -m pipeline.g3_rag --dataset im2gps3k --ks 5 10 15
"""

import argparse
import ast
import base64
import json
import os
import re
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from geogater.paths import Workspace, ensure_parent
from geogater.utils import get_database, get_query_set

DEFAULT_MODEL = "gpt-4o-2024-11-20"
DETAIL = "low"
MAX_TOKENS = 200
TEMPERATURE = 1.2
N_SAMPLES = 5
_NUM = r"[-+]?\d+(?:\.\d+)?"


def build_prompt(candidates_gps: List[str], reverse_gps: List[str], conf: bool) -> str:
    text = (
        "Suppose you are an expert in geo-localization, Please analyze this image and give me a guess of the location.\n"
        "Your answer must be to the coordinates level in (latitude, longitude) format.\n"
        f"For your reference, these are coordinates of some similar images: {candidates_gps}, "
        f"and these are coordinates of some dissimilar images: {reverse_gps}.\n"
        "Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.\n"
        'Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}.'
    )
    if conf:
        text = text.replace(
            '{"latitude": float,"longitude": float}',
            '{"latitude":37.7749,"longitude":-122.4194,"confidence":83}',
        )
        text = text.replace(
            "Remember, you must have an answer",
            "Also return a confidence score from 0 to 100, where 0 = completely uncertain, "
            "50 = equally likely correct or wrong, and 100 = absolutely certain. "
            "Remember, you must have an answer",
        )
    return text


def _safe_load(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def _extract(o: Any) -> Tuple[List[float], List[float]]:
    coords, confs = [], []
    if isinstance(o, str):
        m = re.search(r"\{.*?\}", o, flags=re.DOTALL)
        if m:
            return _extract(_safe_load(m.group(0)))
        for cm in re.finditer(r'"?confidence"?\s*:\s*(' + _NUM + r")", o, flags=re.I):
            confs.append(float(cm.group(1)))
        latm = re.search(r'"?latitude"?\s*:\s*(' + _NUM + r")", o, flags=re.I)
        lonm = re.search(r'"?longitude"?\s*:\s*(' + _NUM + r")", o, flags=re.I)
        if latm and lonm:
            coords.extend([float(latm.group(1)), float(lonm.group(1))])
    elif isinstance(o, dict):
        lat, lon = o.get("latitude"), o.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            coords.extend([float(lat), float(lon)])
        if "confidence" in o:
            try:
                confs.append(float(o["confidence"]))
            except Exception:
                pass
    elif isinstance(o, (list, tuple)):
        if len(o) >= 2 and all(isinstance(x, (int, float)) for x in o[:2]):
            return [float(o[0]), float(o[1])], confs
        for item in o:
            c, f = _extract(item)
            coords.extend(c)
            confs.extend(f)
    return coords, confs


def extract_coordinates(responses: Any) -> List[str]:
    """Flatten all responses into ["lat1", "lon1", "lat2", "lon2", ...]."""
    obj = responses
    if isinstance(responses, str):
        parsed = _safe_load(responses)
        obj = parsed if parsed is not None else responses
    coords, _ = _extract(obj)
    return [f"{x:.8f}".rstrip("0").rstrip(".") for x in coords]


def query_lvlm(client, image_path, model, candidates_gps, reverse_gps, conf):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(candidates_gps, reverse_gps, conf)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": DETAIL},
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=N_SAMPLES,
    )
    out = []
    for choice in resp.choices:
        text = choice.message.content
        if not isinstance(text, str):
            text = json.dumps(text)
        m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        js = (m.group(0) if m else text).strip()
        try:
            json.loads(js)
            out.append(js)
        except Exception:
            out.append('{"latitude": 0.0, "longitude": 0.0}')
    return out


def run_k(args, k, client, ws, qs, df, db, db_spec, I, I_inv):
    out_path = ensure_parent(ws.rag_csv(args.dataset, args.database, k))
    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        done_ids = set(done[qs.id_col])
        print(f"[k={k}] resuming from {out_path} ({len(done)} rows)")
    else:
        done, done_ids = pd.DataFrame(), set()

    fmt = lambda r: f"[{db.at[r, db_spec.lat_col]}, {db.at[r, db_spec.lon_col]}]"
    pending = []

    def flush():
        nonlocal done, pending
        if pending:
            done = pd.concat([done, pd.DataFrame(pending)], ignore_index=True)
            pending = []
        done.to_csv(out_path, index=False)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"G3 RAG k={k}"):
        if row[qs.id_col] in done_ids:
            continue
        cands = [fmt(r) for r in I[i][:k]]
        revs = [fmt(r) for r in I_inv[i][:k]]
        path = os.path.join(qs.image_dir(args.data_root), qs.image_file(row[qs.id_col]))
        try:
            resp = query_lvlm(client, path, args.model_name, cands, revs, not args.no_conf)
        except Exception as e:
            print(f"[error] {row[qs.id_col]}: {e}")
            resp = []
        pending.append({qs.id_col: row[qs.id_col], "rag_response": resp})
        if len(pending) >= args.save_every:
            flush()
    flush()

    done = done.set_index(done[qs.id_col]).loc[df[qs.id_col]].reset_index(drop=True)
    done["rag_coordinates"] = done["rag_response"].apply(extract_coordinates)
    done.to_csv(out_path, index=False)
    print(f"[k={k}] saved {len(done)} rows to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--database", default="mp16", choices=["mp16", "osv5m"])
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument(
        "--no_conf",
        action="store_true",
        help="Use the vanilla G3 prompt (the paper runs ask for a confidence too).",
    )
    p.add_argument("--data_root", default="./data")
    p.add_argument("--work_dir", default="./outputs")
    p.add_argument("--save_every", type=int, default=50)
    args = p.parse_args()

    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this script.")
    client = OpenAI()

    ws = Workspace(args.work_dir, "g3")
    qs = get_query_set(args.dataset)
    db_spec = get_database(args.database)
    df = pd.read_csv(qs.csv_path(args.data_root))
    db = pd.read_csv(db_spec.csv_path(args.data_root))
    I = np.load(ws.neighbors(args.database, args.dataset))
    I_inv = np.load(ws.inverse_neighbors(args.database, args.dataset))
    for k in args.ks:
        run_k(args, k, client, ws, qs, df, db, db_spec, I, I_inv)


if __name__ == "__main__":
    main()
