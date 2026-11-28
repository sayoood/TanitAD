#!/usr/bin/env python3
"""A1/A3 — THE TRUE SOURCE: the per-clip vehicle parameter the PhysicalAI release ships.

Artifact under test:
  nvidia/PhysicalAI-Autonomous-Vehicles ::
      calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet

Pulls only the chunks the 20 local eval clips live in (19 unique, ~9 KB each),
reports the FULL schema, and joins the true wheelbase to each clip.

⛔ UUID-bearing output stays in the scratchpad; only the aggregate goes to the repo.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DS = "nvidia/PhysicalAI-Autonomous-Vehicles"
VD_TMPL = "calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet"
PAI = Path(r"C:\Users\Admin\tanitad-data\physicalai")
EPS_DIR = Path(r"C:\Users\Admin\refav1_eval_slice\eps")
OUT = Path(sys.argv[1])
SCRATCH = Path(sys.argv[2])
LOCAL_VD = PAI / "calibration" / "calibration" / "vehicle_dimensions"


def main() -> None:
    sys.path.insert(0, str(Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD") / "stack"))
    os.environ.setdefault(
        "TANITAD_KEYS_FILE",
        r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt")
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not populated"
    from huggingface_hub import hf_hub_download

    ids = sorted(p.name[: -len(".v2ep.pt")] for p in EPS_DIR.glob("*.v2ep.pt"))
    idx = pd.read_parquet(PAI / "clip_index.parquet")
    chunks = sorted({int(idx.loc[i, "chunk"]) for i in ids})
    print(f"[wb] {len(ids)} clips -> {len(chunks)} chunks: {chunks}", flush=True)

    cache = SCRATCH / "vd_cache"
    cache.mkdir(parents=True, exist_ok=True)
    frames, schema = [], None
    for c in chunks:
        local = LOCAL_VD / f"vehicle_dimensions.chunk_{c:04d}.parquet"
        p = str(local) if local.exists() else hf_hub_download(
            REPO_DS, VD_TMPL.format(c=c), repo_type="dataset", local_dir=str(cache))
        df = pd.read_parquet(p).reset_index()
        if schema is None:
            schema = {k: str(v) for k, v in df.dtypes.items()}
            print(f"[wb] SCHEMA of {VD_TMPL.format(c=c)}: {schema}", flush=True)
        df["chunk"] = c
        frames.append(df)
    vd = pd.concat(frames, ignore_index=True)
    vd["clip_id"] = vd["clip_id"].astype(str)
    print(f"[wb] rows={len(vd)} unique clips={vd.clip_id.nunique()}", flush=True)

    wb_of = dict(zip(vd.clip_id, vd.wheelbase.astype(float)))
    meta = None
    mp = PAI / "metadata" / "data_collection.parquet"
    if mp.exists():
        meta = pd.read_parquet(mp).reset_index()
        meta["clip_id"] = meta["clip_id"].astype(str)
        meta = meta.set_index("clip_id")
        print(f"[wb] metadata cols: {list(meta.columns)}", flush=True)

    rows = []
    for i in ids:
        m = meta.loc[i] if (meta is not None and i in meta.index) else None
        rows.append({
            "clip_id": i,
            "chunk": int(idx.loc[i, "chunk"]),
            "split_release": str(idx.loc[i, "split"]),
            "wheelbase_true_m": wb_of.get(i),
            "platform_class": None if m is None else str(m.get("platform_class")),
            "country": None if m is None else str(m.get("country")),
        })
    pd.DataFrame(rows).to_csv(SCRATCH / "true_wheelbase_per_clip_UUID.csv", index=False)
    # repo-safe aggregate + a UUID-free per-clip table keyed by an ordinal
    safe = [{"clip_ord": k, "chunk": r["chunk"], "split_release": r["split_release"],
             "wheelbase_true_m": r["wheelbase_true_m"],
             "platform_class": r["platform_class"], "country": r["country"]}
            for k, r in enumerate(rows)]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(safe).to_csv(OUT / "true_wheelbase_per_clip.csv", index=False)
    vals = [r["wheelbase_true_m"] for r in rows if r["wheelbase_true_m"] is not None]
    agg = {
        "source": f"HF {REPO_DS} :: {VD_TMPL}",
        "schema": schema,
        "n_clips": len(rows),
        "n_resolved": len(vals),
        "distinct_values": sorted({round(v, 4) for v in vals}),
        "counts": {str(round(v, 4)): int(sum(1 for x in vals if round(x, 4) == round(v, 4)))
                   for v in sorted({round(x, 4) for x in vals})},
        "mean": float(np.mean(vals)) if vals else None,
        "L_enc_used_by_the_corpus": 2.9,
        "evidence_class": "MEASURED",
    }
    (OUT / "true_wheelbase.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
