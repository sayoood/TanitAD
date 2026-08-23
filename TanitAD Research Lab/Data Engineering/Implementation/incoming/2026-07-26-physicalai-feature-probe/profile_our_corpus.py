#!/usr/bin/env python3
"""Join OUR R0 selection against PhysicalAI's own clip metadata.

Never done before: we hold `r0/r0_selection.parquet` (the canonical parity
selection) but have never joined it to `clip_index.parquet` /
`metadata/data_collection.parquet` / `metadata/feature_presence.parquet`.
That join answers, for the corpus every arm trains on:
  * country / EU-vs-US mix  -> the roundabout-density prior
  * hour_of_day             -> night fraction
  * the dataset's OWN train/val/test split assignment of our clips
  * which of the 36 features are actually AVAILABLE for our exact clips
  * platform_class / radar_config coverage on our clips

Reads local parquet + the metadata cached by `pull_pai_metadata.py`. No network
beyond what is already cached; writes derived aggregates only (no clip IDs).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get(
    "PAI_PROBE_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\pai_probe_cache"))
R0 = Path(os.environ.get(
    "TANITAD_PAI_ROOT", r"C:\Users\Admin\tanitad-data\physicalai")) / "r0" / "r0_selection.parquet"


def main():
    sel = pd.read_parquet(R0)
    idx = pd.read_parquet(CACHE / "clip_index.parquet")
    dc = pd.read_parquet(CACHE / "metadata" / "data_collection.parquet")
    fp = pd.read_parquet(CACHE / "metadata" / "feature_presence.parquet")
    for d in (idx, dc, fp):
        if d.index.names and any(n is not None for n in d.index.names):
            d.reset_index(inplace=True)

    out = {"r0_selection": {"n_rows": int(len(sel)),
                            "columns": list(map(str, sel.columns))}}
    print(f"[ours] r0_selection: {len(sel)} rows, cols={list(sel.columns)}",
          flush=True)

    sel["clip_id"] = sel["clip_id"].astype(str)
    j = (sel[["clip_id"]].drop_duplicates()
         .merge(idx, on="clip_id", how="left")
         .merge(dc, on="clip_id", how="left")
         .merge(fp, on="clip_id", how="left"))
    out["join"] = {"n_selected": int(len(sel)),
                   "n_unique_clips": int(sel["clip_id"].nunique()),
                   "n_matched_in_clip_index": int(j["split"].notna().sum()),
                   "n_unmatched": int(j["split"].isna().sum())}
    print(f"[ours] matched {out['join']['n_matched_in_clip_index']}/"
          f"{len(j)} clips in clip_index", flush=True)

    def vc(col):
        return {str(k): int(v) for k, v in
                j[col].value_counts(dropna=False).items()}

    for c in ("split", "country", "platform_class", "radar_config",
              "hour_of_day", "month"):
        if c in j.columns:
            out[f"our_{c}"] = vc(c)

    # EU vs US, and a night proxy
    if "country" in j.columns:
        us = int((j["country"] == "United States").sum())
        eu = int(j["country"].notna().sum() - us)
        out["our_geo"] = {"united_states": us, "european": eu,
                          "eu_frac": round(eu / max(1, us + eu), 4)}
        # corpus-wide comparison
        cus = int((dc["country"] == "United States").sum())
        out["corpus_geo"] = {"united_states": cus,
                             "european": int(len(dc) - cus),
                             "eu_frac": round((len(dc) - cus) / len(dc), 4)}
    if "hour_of_day" in j.columns:
        night = j["hour_of_day"].isin(list(range(0, 6)) + [22, 23])
        out["our_night_frac"] = round(float(night.mean()), 4)
        cn = dc["hour_of_day"].isin(list(range(0, 6)) + [22, 23])
        out["corpus_night_frac"] = round(float(cn.mean()), 4)

    # which of the 36 features are available for OUR exact clips
    feats = [c for c in fp.columns if c != "clip_id"]
    avail = {}
    for f in feats:
        s = j[f]
        avail[f] = {"n_true": int(s.fillna(False).sum()),
                    "frac": round(float(s.fillna(False).mean()), 4)}
    out["our_feature_availability"] = avail
    print("[ours] feature availability on our clips:", flush=True)
    for f, v in avail.items():
        print(f"   {f:34s} {v['n_true']:5d}  {100*v['frac']:6.2f}%", flush=True)

    (HERE / "our_corpus_profile.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[ours] done", flush=True)


if __name__ == "__main__":
    main()
