"""Step 2. Zero-shot LVLM prediction with verbalized confidence (Sec. 3.1, Supp. A.1).

For each query image we draw N i.i.d. samples (one API call with ``n=N``) of a
(latitude, longitude, confidence) triple. Unparseable samples are stored as
None and later omitted from the meta-features; we never re-query.

Requires ``OPENAI_API_KEY`` in the environment. The script is resumable.

Example:
    export OPENAI_API_KEY=...
    python -m pipeline.zero_shot --dataset im2gps3k --n_samples 5
"""

import argparse
import base64
import json
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from geogater.paths import Workspace, ensure_parent
from geogater.utils import get_query_set

DEFAULT_MODEL = "gpt-4o-2024-11-20"
DETAIL = "low"
MAX_TOKENS = 40
TEMPERATURE = 1.0

ZS_PROMPT = (
    "Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image. "
    "Please give me the location of the given image.\n"
    "Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.\n"
    "In addition, include a confidence score from 0 to 100, where 0 means completely uncertain, 50 means equally likely correct or wrong, and 100 means absolutely certain.\n"
    "Your answer should be in the following JSON format without any other information: "
    '{"latitude":37.7749,"longitude":-122.4194,"confidence":83}.'
)


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _clamp_conf(v) -> Optional[int]:
    v = _to_float(v)
    if v is None or v != v:
        return None
    return int(round(max(0.0, min(100.0, v))))


def extract_json_block(text: str) -> str:
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    return m.group(0) if m else text.strip()


def parse_sample(s: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Parse one sample: JSON first, then field-wise regex fallback."""
    if s is None:
        return None, None, None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            lat = next((_to_float(obj[k]) for k in ("latitude", "lat") if k in obj), None)
            lon = next(
                (_to_float(obj[k]) for k in ("longitude", "lng", "lon", "long") if k in obj),
                None,
            )
            return lat, lon, _clamp_conf(obj.get("confidence"))
    except Exception:
        pass
    num = r"(-?\d+(?:\.\d+)?)"
    m_lat = re.search(r'"(?:latitude|lat)"\s*:\s*' + num, s, flags=re.I)
    m_lon = re.search(r'"(?:longitude|lng|lon|long)"\s*:\s*' + num, s, flags=re.I)
    m_conf = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', s, flags=re.I)
    return (
        _to_float(m_lat.group(1)) if m_lat else None,
        _to_float(m_lon.group(1)) if m_lon else None,
        _clamp_conf(m_conf.group(1)) if m_conf else None,
    )


def query_lvlm(client, image_path: str, model: str, n: int) -> List[str]:
    b64 = encode_image(image_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ZS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": DETAIL,
                        },
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=n,
    )
    outs = []
    for choice in resp.choices:
        content = choice.message.content
        if not isinstance(content, str):
            content = json.dumps(content)
        outs.append(extract_json_block(content))
    return outs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--n_samples", type=int, default=5, help="N hypotheses per query.")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--work_dir", default="./outputs")
    p.add_argument("--out_csv", default=None, help="Override output path.")
    p.add_argument("--save_every", type=int, default=50)
    args = p.parse_args()

    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this script.")
    client = OpenAI()

    qs = get_query_set(args.dataset)
    df = pd.read_csv(qs.csv_path(args.data_root))
    out_path = ensure_parent(
        args.out_csv or Workspace(args.work_dir, "g3").zero_shot_csv(args.dataset)
    )

    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        done_ids = set(done[qs.id_col])
        print(f"Resuming from {out_path} ({len(done)} rows)")
    else:
        done, done_ids = pd.DataFrame(), set()

    pending = []

    def flush():
        nonlocal done, pending
        if pending:
            done = pd.concat([done, pd.DataFrame(pending)], ignore_index=True)
            pending = []
        done.to_csv(out_path, index=False)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot"):
        if row[qs.id_col] in done_ids:
            continue
        path = os.path.join(qs.image_dir(args.data_root), qs.image_file(row[qs.id_col]))
        try:
            responses = query_lvlm(client, path, args.model_name, args.n_samples)
        except Exception as e:  # keep going; failed queries yield no hypotheses
            print(f"[error] {row[qs.id_col]}: {e}")
            responses = []
        parsed = [parse_sample(r) for r in responses]
        row = row.copy()
        row["response"] = responses
        row["latitudes"] = [p[0] for p in parsed]
        row["longitudes"] = [p[1] for p in parsed]
        row["confidence_choices"] = [p[2] for p in parsed]
        pending.append(row)
        if len(pending) >= args.save_every:
            flush()
    flush()

    # Keep the original dataset order (downstream steps align rows by position).
    done = done.set_index(done[qs.id_col]).loc[df[qs.id_col]].reset_index(drop=True)
    done.to_csv(out_path, index=False)
    print(f"Saved {len(done)} rows to {out_path}")


if __name__ == "__main__":
    main()
