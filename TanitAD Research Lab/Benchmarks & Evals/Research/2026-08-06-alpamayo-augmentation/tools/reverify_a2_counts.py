"""INDEPENDENT RE-VERIFICATION of the Alpamayo-2 augmentation counts, for MODEL_REGISTRY.md.

Written fresh (not a re-run of verify_a2_parquet.py) because the brief requires the counts be
re-derived from the raw artifacts, not copied from prose or from a prior agent's JSON.

Three probes, deliberately different in shape:
  P1  records.parquet  -> COUNT RECORDS (rows), never files. Per-task, per-clip, per-key.
  P2  selection_manifest.json (HF) -> the delivered-vs-selected accounting that produces the
      "356 rows missing" number the dataset card understates as "one".
  P3  aug120 fusion raw JSONs (in repo) -> COUNT RECORDS inside the sam3 files, because a
      listing probe sees a MISSING file but never a SHORT one.

Token read in place from Keys.txt, never printed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
PARQUET = os.path.join(SP, "a2dl", "records.parquet")
HF_REPO = "Sayood/tanitad-alpamayo2-augmentation"

out: dict = {"_evidence_class": "MEASURED (ours)", "_probe_date": "2026-08-16"}

# ------------------------------------------------------------------ P1: parquet, rows
import pyarrow.parquet as pq          # noqa: E402
import pyarrow.compute as pc          # noqa: E402

t = pq.read_table(PARQUET)
p1: dict = {
    "path": PARQUET,
    "file_size_bytes": os.path.getsize(PARQUET),
    "num_rows": t.num_rows,
    "num_columns": t.num_columns,
    "schema": {f.name: str(f.type) for f in t.schema},
}
h = hashlib.sha256()
with open(PARQUET, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
p1["local_sha256"] = h.hexdigest()

clip_ids = t.column("clip_id").to_pylist()
tasks = t.column("task").to_pylist()
p1["n_unique_clip_id"] = len(set(clip_ids))
p1["rows_by_task"] = dict(Counter(tasks).most_common())
p1["rows_by_quantisation"] = dict(Counter(t.column("quantisation").to_pylist()))
p1["rows_by_model_id"] = dict(Counter(t.column("model_id").to_pylist()))
p1["rows_by_seed"] = {str(k): v for k, v in Counter(t.column("seed").to_pylist()).items()}
p1["distinct_t0_us"] = sorted({int(x) for x in t.column("t0_us").to_pylist()})[:10]
p1["n_distinct_t0_us"] = len({int(x) for x in t.column("t0_us").to_pylist()})
p1["error_non_null_rows"] = t.num_rows - t.column("error").null_count

# how many of the 5 tasks each clip actually has -> RECORD-level completeness, not file-level
per_clip = Counter()
for c in clip_ids:
    per_clip[c] += 1
p1["clips_by_n_task_rows"] = {str(k): v for k, v in sorted(Counter(per_clip.values()).items())}
p1["clips_with_fewer_than_5_rows"] = [c for c, n in per_clip.items() if n < 5]

# arithmetic reconciliation, recorded not smoothed
n_tasks = len(p1["rows_by_task"])
p1["reconcile"] = {
    "clips_x_tasks": p1["n_unique_clip_id"] * n_tasks,
    "actual_rows": t.num_rows,
    "shortfall_vs_clips_x_tasks": p1["n_unique_clip_id"] * n_tasks - t.num_rows,
    "short_tasks": {k: v for k, v in p1["rows_by_task"].items()
                    if v != max(p1["rows_by_task"].values())},
}

# wall_s
ws = t.column("wall_s").to_pylist()
per_task: dict = {}
total_s = 0.0
for task, w in zip(tasks, ws):
    if w is not None:
        per_task.setdefault(task, []).append(w)
        total_s += w
p1["wall_s_per_task"] = {}
for k in sorted(per_task):
    v = sorted(per_task[k])
    p1["wall_s_per_task"][k] = {
        "n_rows": len(v),
        "mean_s": round(sum(v) / len(v), 2),
        "median_s": round(v[len(v) // 2], 2),
        "sum_hours": round(sum(v) / 3600.0, 2),
    }
p1["wall_hours_total"] = round(total_s / 3600.0, 2)
p1["s_per_clip_full_battery"] = round(total_s / p1["n_unique_clip_id"], 2)

# raw_json key census -> RECORD counts of the metric block + the card-promised provenance keys
raw = t.column("raw_json").to_pylist()
key_hits = Counter()
traj_shape = Counter()
min_ade, min_fde = [], []
n_traj = 0
for task, rj in zip(tasks, raw):
    if not rj:
        continue
    for k in ("min_ade_m", "min_fde_m", "_contamination", "peak_gib", "pred_xyz_shape",
              "num_trajectory_samples"):
        if f'"{k}"' in rj:
            key_hits[k] += 1
    if task == "trajectory":
        n_traj += 1
        try:
            d = json.loads(rj)
        except Exception:
            continue
        if isinstance(d, dict):
            if d.get("pred_xyz_shape") is not None:
                traj_shape[json.dumps(d["pred_xyz_shape"])] += 1
            if d.get("min_ade_m") is not None:
                min_ade.append(float(d["min_ade_m"]))
            if d.get("min_fde_m") is not None:
                min_fde.append(float(d["min_fde_m"]))
p1["raw_json_key_row_counts"] = dict(key_hits)
p1["trajectory_rows"] = n_traj
p1["trajectory_pred_xyz_shapes"] = dict(traj_shape)


def stat(v):
    if not v:
        return None
    v = sorted(v)
    return {"n": len(v), "mean": round(sum(v) / len(v), 4), "median": round(v[len(v) // 2], 4)}


p1["min_ade_m"] = stat(min_ade)
p1["min_fde_m"] = stat(min_fde)
out["P1_parquet"] = p1

# ------------------------------------------------------- P2: selection manifest, far side
p2: dict = {}
try:
    import truststore
    truststore.inject_into_ssl()
except Exception as exc:
    p2["truststore"] = f"{type(exc).__name__}: {exc}"
try:
    with open(os.path.join(REPO, "Keys.txt"), encoding="utf-8", errors="ignore") as fh:
        m = re.search(r"hf_[A-Za-z0-9]+", fh.read())
    tok = m.group(0) if m else None
    import urllib.request

    def get(url):
        req = urllib.request.Request(url)
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    tree = json.loads(get(
        f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main?recursive=1").decode())
    p2["farside_files"] = [{"path": e.get("path"),
                            "size": (e.get("lfs") or {}).get("size", e.get("size")),
                            "oid": (e.get("lfs") or {}).get("oid")} for e in tree]
    far_pq = [e for e in p2["farside_files"] if str(e["path"]).endswith("records.parquet")]
    if far_pq:
        p2["parquet_size_match"] = far_pq[0]["size"] == p1["file_size_bytes"]
        p2["parquet_sha256_match"] = far_pq[0]["oid"] == p1["local_sha256"]

    man = json.loads(get(
        f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/selection_manifest.json").decode())
    p2["manifest_type"] = type(man).__name__
    if isinstance(man, dict):
        p2["manifest_top_keys"] = list(man)[:20]
        rows = None
        for k in ("clips", "selection", "records", "items", "selected"):
            if isinstance(man.get(k), list):
                rows = man[k]
                p2["manifest_list_key"] = k
                break
    else:
        rows = man
        p2["manifest_list_key"] = "<root list>"
    if rows is not None:
        p2["manifest_n_entries"] = len(rows)
        if rows and isinstance(rows[0], dict):
            p2["manifest_entry_keys"] = list(rows[0])
            sel = [r.get("clip_id") for r in rows if r.get("clip_id")]
        else:
            sel = [r for r in rows if isinstance(r, str)]
        sel_set = set(sel)
        del_set = set(clip_ids)
        p2["manifest_unique_clip_ids"] = len(sel_set)
        p2["delivered_unique_clip_ids"] = len(del_set)
        p2["selected_but_zero_rows"] = len(sel_set - del_set)
        p2["delivered_not_in_manifest"] = len(del_set - sel_set)
        p2["intersection"] = len(sel_set & del_set)
        p2["accounting"] = {
            "selected_rows_at_5_tasks": len(sel_set) * 5,
            "minus_zero_row_clips": -len(sel_set - del_set) * 5,
            "plus_unmanifested_clips": len(del_set - sel_set) * 5,
            "minus_missing_task_rows": -p1["reconcile"]["shortfall_vs_clips_x_tasks"],
            "computed_delivered_rows": (len(sel_set) * 5
                                        - len(sel_set - del_set) * 5
                                        + len(del_set - sel_set) * 5
                                        - p1["reconcile"]["shortfall_vs_clips_x_tasks"]),
            "actual_delivered_rows": t.num_rows,
        }
        p2["accounting"]["closes"] = (
            p2["accounting"]["computed_delivered_rows"] == t.num_rows)
        p2["rows_missing_vs_selected_x5"] = len(sel_set) * 5 - t.num_rows
except Exception as exc:
    p2["error"] = f"{type(exc).__name__}: {exc}"
out["P2_manifest"] = p2

# ------------------------------------------------------------- P3: aug120, records not files
p3: dict = {}
base = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering", "Implementation",
                    "incoming", "2026-08-15-aug120-fusion", "raw")
for name in ("fused_aug120_summary.json", "aug120_coverage.json",
             "fused_aug120_batch_accounting.json", "fused_aug120_label_sources.json"):
    p = os.path.join(base, name)
    if not os.path.exists(p):
        p3[name] = "MISSING"
        continue
    d = json.load(open(p, encoding="utf-8"))
    if name == "fused_aug120_summary.json":
        p3["summary"] = d
    elif name == "aug120_coverage.json":
        p3["coverage_sizes"] = {k: (len(v) if isinstance(v, (list, dict)) else v)
                                for k, v in d.items()}
    elif name == "fused_aug120_batch_accounting.json":
        rows = d if isinstance(d, list) else d.get("batches", d)
        if isinstance(rows, dict):
            rows = list(rows.values())
        agg = Counter()
        for r in rows:
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if isinstance(v, int):
                    agg[k] += v
        p3["batch_accounting_n_tags"] = len(rows)
        p3["batch_accounting_column_sums"] = dict(agg)
    elif name == "fused_aug120_label_sources.json":
        rows = d if isinstance(d, list) else list(d.values())
        p3["label_sources_n_records"] = len(rows)
        sam3 = Counter()
        for r in rows:
            if isinstance(r, dict):
                sam3[str(r.get("sam3"))] += 1
        p3["label_sources_sam3_value_counts"] = dict(sam3.most_common(8))
out["P3_aug120"] = p3

json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
print()
