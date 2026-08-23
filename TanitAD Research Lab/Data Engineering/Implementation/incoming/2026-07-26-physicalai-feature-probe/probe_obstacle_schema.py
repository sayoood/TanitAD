#!/usr/bin/env python3
"""MEASURE the schema + full class taxonomy of PhysicalAI-AV's unread label
features: `obstacle.offline`, `egomotion.offline`, `vehicle_dimensions`,
`camera_intrinsics.offline` and `lidar_intrinsics.offline`.

`obstacle.offline` is the ONLY one of the 36 features that could plausibly carry
a traffic-light / traffic-sign / static-infrastructure object, so its category
enum is the decisive probe for the "no traffic-light feature exists" claim.
The `.offline` egomotion is the second-guess for a route/nav/goal field.

Downloads ONE chunk zip (obstacle.offline chunk_0000 == 63.7 MB) into a
scratchpad cache OUTSIDE the repo — gated bytes are never staged. Only the
derived SCHEMA (column names, dtypes, category enum, per-class counts) is
written here; no per-clip content, no geometry.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
RTYPE = "dataset"
CACHE = Path(os.environ.get(
    "PAI_PROBE_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\pai_probe_cache"))

ZIP_TARGETS = [
    "labels/obstacle.offline/obstacle.offline.chunk_0000.zip",
    "labels/egomotion.offline/egomotion.offline.chunk_0000.zip",
]
PARQUET_TARGETS = [
    "calibration/vehicle_dimensions/vehicle_dimensions.chunk_0000.parquet",
    "calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet",
    "calibration/lidar_intrinsics.offline/lidar_intrinsics.offline.chunk_0000.parquet",
    "calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet",
]


def _bootstrap():
    for p in HERE.parents:
        if (p / "stack" / "tanitad" / "keys.py").exists():
            sys.path.insert(0, str(p / "stack"))
            break
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()
    assert os.environ.get("HF_TOKEN")


def schema_of(df, name, n_clips=None):
    d = {"name": name, "n_rows": int(len(df)),
         "n_clips_sampled": n_clips,
         "columns": []}
    for c in df.columns:
        s = df[c]
        col = {"col": str(c), "dtype": str(s.dtype)}
        try:
            nu = int(s.nunique(dropna=True))
            col["n_unique"] = nu
            col["n_null"] = int(s.isna().sum())
            if nu <= 80:
                col["values"] = {str(k): int(v) for k, v in
                                 s.value_counts(dropna=False).head(80).items()}
            else:
                col["sample"] = [str(v) for v in s.dropna().unique()[:6]]
                if str(s.dtype).startswith(("float", "int")):
                    col["min"] = float(s.min())
                    col["max"] = float(s.max())
        except Exception as e:  # noqa: BLE001
            col["error"] = repr(e)
        d["columns"].append(col)
    return d


def main():
    _bootstrap()
    from huggingface_hub import hf_hub_download
    import pandas as pd
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {"repo": REPO, "note": "schema only; no gated content reproduced",
           "features": {}}

    for rel in ZIP_TARGETS:
        try:
            p = hf_hub_download(REPO, rel, repo_type=RTYPE, local_dir=str(CACHE))
        except Exception as e:  # noqa: BLE001
            out["features"][rel] = {"ERROR": repr(e)}
            print(f"[obs] FAIL {rel}: {e!r}", flush=True)
            continue
        print(f"[obs] got {rel} {Path(p).stat().st_size/1e6:.1f} MB", flush=True)
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.endswith(".parquet")]
            print(f"[obs]   {len(names)} parquets inside", flush=True)
            frames, used = [], 0
            for n in names[:12]:                     # 12 clips is plenty for enum
                frames.append(pd.read_parquet(io.BytesIO(z.read(n))))
                used += 1
            df = pd.concat(frames, ignore_index=False)
            if df.index.names and any(x is not None for x in df.index.names):
                df = df.reset_index()
        out["features"][rel] = schema_of(df, rel, n_clips=used)
        out["features"][rel]["n_parquets_in_zip"] = len(names)
        out["features"][rel]["inner_name_example"] = (
            "<clip_uuid>." + names[0].split(".", 1)[1] if names else None)

        # explicit hunt: any column whose NAME or VALUES look like map /
        # traffic-control / lane / route semantics (2nd probe per CLAUDE.md)
        hits = {}
        import re
        pat = re.compile(
            r"lane|map|road|junction|intersect|roundab|traffic|light|signal|"
            r"sign\b|route|nav|goal|topolog|centerline|crossing|stop", re.I)
        for c in df.columns:
            if pat.search(str(c)):
                hits[f"COLUMN:{c}"] = "name matches"
            if df[c].dtype == object:
                try:
                    vals = df[c].dropna().astype(str).unique()[:5000]
                    m = sorted({v for v in vals if pat.search(v)})[:40]
                    if m:
                        hits[f"VALUES:{c}"] = m
                except Exception:  # noqa: BLE001
                    pass
        out["features"][rel]["map_semantics_hits"] = hits or "NONE"
        print(f"[obs]   map/traffic semantic hits: "
              f"{list(hits) if hits else 'NONE'}", flush=True)

    for rel in PARQUET_TARGETS:
        try:
            p = hf_hub_download(REPO, rel, repo_type=RTYPE, local_dir=str(CACHE))
            df = pd.read_parquet(p)
            if df.index.names and any(x is not None for x in df.index.names):
                df = df.reset_index()
            out["features"][rel] = schema_of(df, rel)
            print(f"[obs] {rel}: {len(df)} rows x {len(df.columns)} cols",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            out["features"][rel] = {"ERROR": repr(e)}
            print(f"[obs] FAIL {rel}: {e!r}", flush=True)

    (HERE / "pai_label_schemas.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[obs] done", flush=True)


if __name__ == "__main__":
    main()
