#!/usr/bin/env python3
"""SETTLE obstacle.offline coverage from `metadata/feature_presence.parquet` — the tool that owns
the fact — over the exact clip sets our corpora draw from.

The 97.44 % figure travels through >= 3 documents as INHERITED. A per-chunk sample measured 96.54 %
here. This reads the dataset's own presence table (11 MB) and reports coverage over: the whole
train split, our 3000-clip phase0 selection, our 500-clip R0 selection, and the canonical val40.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
REL = "metadata/feature_presence.parquet"
OUT = Path(__file__).with_name("coverage.json")


def main():
    p = ROOT / REL
    if not p.exists():
        from tanitad.keys import enable_tls, load_keys
        enable_tls()
        load_keys()
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download(HF_REPO, REL, repo_type="dataset", local_dir=str(ROOT)))
    df = pd.read_parquet(p)
    if df.index.names and any(x is not None for x in df.index.names):
        df = df.reset_index()
    res = {"_source": REL, "_evidence_class": "MEASURED (ours; dataset's own presence table)",
           "n_rows": int(len(df)), "columns": list(map(str, df.columns))[:60]}

    col = next((c for c in df.columns if str(c) == "obstacle.offline"), None)
    idcol = next((c for c in df.columns
                  if str(c).lower() in ("clip_id", "index", "level_0")), df.columns[0])
    res["obstacle_column"] = str(col)
    res["id_column"] = str(idcol)
    if col is None:
        # long form: a feature column + a boolean
        res["ERROR"] = "no 'obstacle.offline' column; dumping value counts of object columns"
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique() < 60:
                res.setdefault("enums", {})[str(c)] = {
                    str(k): int(v) for k, v in df[c].value_counts().head(50).items()}
        OUT.write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str)[:3000])
        return

    df[idcol] = df[idcol].astype(str)
    has = dict(zip(df[idcol], df[col].astype(bool)))
    res["corpus_wide"] = {"n": len(has), "with_obstacle": int(sum(has.values())),
                          "frac": round(sum(has.values()) / max(len(has), 1), 4)}

    def over(name, ids):
        ids = [str(i) for i in ids]
        known = [i for i in ids if i in has]
        n_has = sum(has[i] for i in known)
        return {"n_clips": len(ids), "n_in_presence_table": len(known),
                "with_obstacle": int(n_has),
                "frac": round(n_has / max(len(known), 1), 4),
                "missing_obstacle": int(len(known) - n_has),
                "pct_missing": round(100 * (len(known) - n_has) / max(len(known), 1), 2)}

    ph = pd.read_parquet(ROOT / "r0" / "phase0_selection.parquet")
    r0 = pd.read_parquet(ROOT / "r0" / "r0_selection.parquet")
    res["phase0_selection_3000"] = over("phase0", ph["clip_id"].astype(str))
    res["r0_selection_500"] = over("r0", r0["clip_id"].astype(str))
    vm = Path(__file__).with_name("val40_map.json")
    if vm.exists():
        res["canonical_val40"] = over("val40", [r["clip_id"] for r in json.loads(vm.read_text())])
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "columns"}, indent=1, default=str))


if __name__ == "__main__":
    main()
