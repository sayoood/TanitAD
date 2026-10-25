"""v10 country labels: the recording country of every v9 clip, by reference to the upstream PhysicalAI-AV metadata.

Source: `nvidia/PhysicalAI-Autonomous-Vehicles` `metadata/data_collection.parquet` (clip_id -> country, 306,152 clips, 25 countries).
Join: v9 `clip__sha12` (sha12 = sha256(clip_id)[:12]) -> UUID via the corpus `index/clip_index_v7.2_{train,eval}.json`.
`country_id` indexes a FROZEN table (sorted country names of the upstream release); a later release may only APPEND names.

    python stack/scripts/build_v10_country.py --probe /tmp/claude-0/probe --v9 /tmp/claude-0/hfdata --out DataEng/v10/country
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os
import numpy as np
import pandas as pd

# frozen v1 table: the 25 upstream countries, sorted. Append-only.
COUNTRIES_V1 = ["Austria", "Belgium", "Bulgaria", "Croatia", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
                "Hungary", "Italy", "Latvia", "Lithuania", "Luxembourg", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
                "Slovenia", "Spain", "Sweden", "United States"]
ISO2 = {"Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Czechia": "CZ", "Denmark": "DK", "Estonia": "EE", "Finland": "FI",
        "France": "FR", "Germany": "DE", "Greece": "GR", "Hungary": "HU", "Italy": "IT", "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU",
        "Netherlands": "NL", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK", "Slovenia": "SI", "Spain": "ES",
        "Sweden": "SE", "United States": "US"}


def sha12(u: str) -> str: return hashlib.sha256(u.encode()).hexdigest()[:12]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def build(probe: str, v9: str, out: str) -> dict:
    dc = pd.read_parquet(os.path.join(probe, "data_collection.parquet"))
    unknown = sorted(set(dc["country"]) - set(COUNTRIES_V1))
    if unknown: raise SystemExit(f"upstream has countries outside the frozen v1 table: {unknown} (append, never re-sort)")
    cid = {c: i for i, c in enumerate(COUNTRIES_V1)}
    uu = {}
    for f in ("clip_index_v7.2_train.json", "clip_index_v7.2_eval.json"):
        for u in json.load(open(os.path.join(probe, f)))["clips"]: uu[sha12(u)] = u
    rows, missing = [], []
    for split, fn in (("train", "v9_labels_train.npz"), ("eval139", "v9_labels_eval139.npz")):
        z = np.load(os.path.join(v9, fn), allow_pickle=True)
        for s in (str(x) for x in z["clip__sha12"]):
            u = uu.get(s)
            c = dc.loc[u, "country"] if u in dc.index else None
            if c is None: missing.append((split, s))
            rows.append({"split": split, "sha12": s, "clip_id": u, "country": c, "country_iso2": ISO2.get(c),
                         "country_id": cid.get(c, -1),
                         "month": int(dc.loc[u, "month"]) if c else -1, "hour_of_day": int(dc.loc[u, "hour_of_day"]) if c else -1})
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    pq = os.path.join(out, "v10_country.parquet"); df.to_parquet(pq, index=False)
    json.dump({"version": 1, "countries": COUNTRIES_V1, "iso2": [ISO2[c] for c in COUNTRIES_V1]},
              open(os.path.join(out, "country_table_v1.json"), "w"), indent=1)
    per = {sp: dict(collections.Counter(df[df.split == sp].country).most_common()) for sp in ("train", "eval139")}
    man = {"schema": "tanitad.v10_country/1",
           "source": {"repo": "nvidia/PhysicalAI-Autonomous-Vehicles", "file": "metadata/data_collection.parquet",
                      "sha256": sha256_file(os.path.join(probe, "data_collection.parquet")), "n_upstream_clips": int(len(dc))},
           "join": "v9 clip__sha12 = sha256(clip_id)[:12] -> UUID via corpus index/clip_index_v7.2_{train,eval}.json",
           "inputs_sha256": {f: sha256_file(os.path.join(probe, f)) for f in ("clip_index_v7.2_train.json", "clip_index_v7.2_eval.json")} |
                            {f: sha256_file(os.path.join(v9, f)) for f in ("v9_labels_train.npz", "v9_labels_eval139.npz")},
           "n_clips": {sp: int((df.split == sp).sum()) for sp in ("train", "eval139")},
           "unmapped": len(missing), "n_countries": {sp: len(per[sp]) for sp in per}, "per_country": per,
           "granularity": "clip-level; broadcast to every v9 row of the clip (a 20 s clip never crosses a border in this release: UNVERIFIED, "
                          "upstream gives one country per clip)",
           "admissibility": "label for the strategic country readout; never an inference input except in the explicitly supplied (GNSS) arm",
           "output_sha256": sha256_file(pq)}
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True); ap.add_argument("--v9", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = build(a.probe, a.v9, a.out)
    print(json.dumps({k: m[k] for k in ("n_clips", "unmapped", "n_countries")}, indent=1))
