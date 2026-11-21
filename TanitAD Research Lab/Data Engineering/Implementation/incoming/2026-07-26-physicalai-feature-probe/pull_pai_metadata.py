#!/usr/bin/env python3
"""Pull + inspect PhysicalAI-AV's own metadata tables (METADATA ONLY, ~34 MB).

Files (all small, all at the repo root or under metadata/ or reasoning/):
    features.csv                       the AUTHORITATIVE feature list
    clip_index.parquet                 11.1 MB — per-clip index
    metadata/data_collection.parquet   11.3 MB — per-clip collection conditions
    metadata/feature_presence.parquet  11.2 MB — per-clip x feature coverage
    reasoning/ood_reasoning.parquet     0.15 MB — reasoning annotations

No clip bytes are downloaded. Cache lands OUTSIDE the repo (scratchpad) so no
gated content is ever staged; only derived JSON summaries are written here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
RTYPE = "dataset"
CACHE = Path(os.environ.get(
    "PAI_PROBE_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\pai_probe_cache"))

TARGETS = [
    "features.csv",
    "clip_index.parquet",
    "metadata/data_collection.parquet",
    "metadata/feature_presence.parquet",
    "reasoning/ood_reasoning.parquet",
]


def _bootstrap():
    for p in HERE.parents:
        if (p / "stack" / "tanitad" / "keys.py").exists():
            sys.path.insert(0, str(p / "stack"))
            break
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()
    assert os.environ.get("HF_TOKEN"), "no HF token in Keys.txt"


def describe(df, name, max_uniq=60):
    """Schema + per-column cardinality/value sample. No raw row dumps."""
    out = {"name": name, "n_rows": int(len(df)), "columns": []}
    for c in df.columns:
        s = df[c]
        col = {"col": str(c), "dtype": str(s.dtype)}
        try:
            nu = int(s.nunique(dropna=True))
            col["n_unique"] = nu
            col["n_null"] = int(s.isna().sum())
            if s.dtype == bool or str(s.dtype).startswith(("bool", "int", "uint")):
                col["true_frac"] = (round(float(s.mean()), 6)
                                    if s.dtype == bool else None)
            if nu <= max_uniq:
                vc = s.value_counts(dropna=False).head(max_uniq)
                col["values"] = {str(k): int(v) for k, v in vc.items()}
            else:
                col["sample"] = [str(v) for v in s.dropna().unique()[:8]]
        except Exception as e:  # noqa: BLE001
            col["error"] = repr(e)
        out["columns"].append(col)
    return out


def main():
    _bootstrap()
    from huggingface_hub import hf_hub_download
    import pandas as pd
    CACHE.mkdir(parents=True, exist_ok=True)

    local = {}
    for rel in TARGETS:
        try:
            p = hf_hub_download(REPO, rel, repo_type=RTYPE,
                                local_dir=str(CACHE))
            local[rel] = p
            print(f"[pull] ok {rel} -> {Path(p).stat().st_size/1e6:.2f} MB",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[pull] FAIL {rel}: {e!r}", flush=True)

    summary = {"repo": REPO, "files": {}}

    if "features.csv" in local:
        txt = Path(local["features.csv"]).read_text(encoding="utf-8")
        (HERE / "pai_features.csv").write_text(txt, encoding="utf-8")
        fdf = pd.read_csv(local["features.csv"])
        summary["files"]["features.csv"] = {
            "n_rows": int(len(fdf)), "columns": list(map(str, fdf.columns)),
            "records": json.loads(fdf.to_json(orient="records")),
        }
        print(f"[pull] features.csv: {len(fdf)} rows, cols={list(fdf.columns)}",
              flush=True)

    for rel in TARGETS:
        if rel == "features.csv" or rel not in local:
            continue
        df = pd.read_parquet(local[rel])
        if df.index.names and any(n is not None for n in df.index.names):
            df = df.reset_index()
        summary["files"][rel] = describe(df, rel)
        print(f"[pull] {rel}: {len(df)} rows x {len(df.columns)} cols",
              flush=True)

    (HERE / "pai_metadata_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[pull] done", flush=True)


if __name__ == "__main__":
    main()
