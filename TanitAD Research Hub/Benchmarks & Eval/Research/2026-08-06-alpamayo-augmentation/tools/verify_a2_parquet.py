"""INDEPENDENT verification of the A2 augmentation record set, for the registry row.

Two probes, deliberately different in shape (rule 5):
  1. read the LOCAL cached parquet and recompute every count from scratch;
  2. ask the HF API for the far-side file's size + LFS sha256 and compare to the local
     file's own sha256 — so "the local cache is the published artifact" is MEASURED,
     not assumed.

⛔ The token is read IN PLACE out of Keys.txt and never printed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PARQUET = (r"C:\Users\Admin\AppData\Local\Temp\claude"
           r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
           r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\a2dl\records.parquet")
HF_REPO = "Sayood/tanitad-alpamayo2-augmentation"

out = {"_evidence_class": "MEASURED (ours)", "parquet_path": PARQUET}

# ---------------------------------------------------------------- probe 1: parquet
import pyarrow.parquet as pq                                          # noqa: E402

t = pq.read_table(PARQUET)
out["file_size_bytes"] = os.path.getsize(PARQUET)
out["num_rows"] = t.num_rows
out["num_columns"] = t.num_columns
out["schema"] = {f.name: str(f.type) for f in t.schema}

import pyarrow.compute as pc                                          # noqa: E402


def vc(col):
    s = pc.value_counts(t.column(col))
    return {str(x["values"]): int(x["counts"]) for x in s.to_pylist()}


out["n_unique_clip_id"] = len(pc.unique(t.column("clip_id")))
for c in ("task", "quantisation", "model_id", "seed"):
    out[f"value_counts::{c}"] = vc(c)
out["error_non_null_rows"] = t.num_rows - t.column("error").null_count
out["null_count::error"] = t.column("error").null_count

# arithmetic check the brief asked to be recorded, not smoothed
tasks = out["value_counts::task"]
out["arithmetic_check"] = {
    "n_clips": out["n_unique_clip_id"],
    "n_tasks": len(tasks),
    "clips_x_tasks": out["n_unique_clip_id"] * len(tasks),
    "actual_rows": t.num_rows,
    "shortfall": out["n_unique_clip_id"] * len(tasks) - t.num_rows,
    "short_task": [k for k, v in tasks.items() if v != max(tasks.values())],
}

# wall_s per task + total
ws, tk = t.column("wall_s").to_pylist(), t.column("task").to_pylist()
per = {}
for a, b in zip(tk, ws):
    per.setdefault(a, []).append(b)
per_task, total = {}, 0.0
for k, v in sorted(per.items()):
    v2 = sorted(x for x in v if x is not None)
    total += sum(v2)
    per_task[k] = {"n": len(v2), "mean_s": round(sum(v2) / len(v2), 2),
                   "median_s": round(v2[len(v2) // 2], 1)}
out["wall_s_per_task"] = per_task
out["wall_hours_total"] = round(total / 3600.0, 1)
out["s_per_clip_full_battery"] = round(total / out["n_unique_clip_id"], 2)

# ------------------------------------------------ probe 2: HF far side, different shape
try:
    import truststore

    truststore.inject_into_ssl()
    keys = os.path.join(REPO, "Keys.txt")
    with open(keys, encoding="utf-8", errors="ignore") as fh:
        m = re.search(r"hf_[A-Za-z0-9]+", fh.read())
    tok = m.group(0) if m else None
    import urllib.request

    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main?recursive=1")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=90) as resp:
        status, body = resp.status, json.loads(resp.read().decode("utf-8"))
    far = None
    for e in body:
        if e.get("path", "").endswith("records.parquet"):
            far = e
    out["hf"] = {"status": status, "entry": far,
                 "n_files_listed": len(body),
                 "paths": [e.get("path") for e in body][:20]}
    if far:
        lfs = far.get("lfs") or {}
        far_sha = lfs.get("oid") or lfs.get("sha256")
        far_size = lfs.get("size", far.get("size"))
        hsh = hashlib.sha256()
        with open(PARQUET, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                hsh.update(chunk)
        local_sha = hsh.hexdigest()
        out["hf_vs_local"] = {
            "far_size": far_size, "local_size": out["file_size_bytes"],
            "size_match": far_size == out["file_size_bytes"],
            "far_sha256": far_sha, "local_sha256": local_sha,
            "sha256_match": (far_sha == local_sha) if far_sha else None,
        }
except Exception as exc:                                              # pragma: no cover
    out["hf"] = {"error": f"{type(exc).__name__}: {exc}"}

json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
print()
