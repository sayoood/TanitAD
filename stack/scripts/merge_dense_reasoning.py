"""Merge `alpamayo_dense_reasoning.py` shards into one `v10_reasoning_dense_<split>.parquet` + `..._<split>.coverage.parquet` + MANIFEST.

Reads every <in>/<split>/shard_*.parquet (finalised) and shard_*.parts/part_*.parquet (in-progress; counted and reported), dedupes on
(split, sha12, v9_row, task) preferring error-free rows then the latest write, and checks every row against the TARGET frame set
re-enumerated from the v9 clock at the run's --hz (same code as the generator): t_us must equal round(now_s * 1e6) of that v9 row, and
no row may fall outside the target set. Coverage per clip x task: target / ok / ineligible (by reason) / error / missing.

    PYTHONPATH=stack python stack/scripts/merge_dense_reasoning.py --in /workspace/reasoning_dense --split train --hz 10 \
        --v9-dir /workspace/data/v9 --clip-index-dir /workspace/data/index --out /workspace/reasoning_dense/merged
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpamayo_dense_reasoning as adr                                       # noqa: E402

SCHEMA = "tanitad.v10_reasoning_dense/1"


def coverage(df: pd.DataFrame, target: pd.DataFrame, tasks) -> pd.DataFrame:
    """Per clip x task counts. `target` = enumerate_frames output (one row per target frame)."""
    tgt = target.groupby("sha12").size().rename("n_target")
    rows = []
    e = df["error"].astype("string")
    df = df.assign(_ok=e.isna(), _inel=e.str.startswith("ineligible:").fillna(False))
    df = df.assign(_err=~df._ok & ~df._inel)
    g = df.groupby(["sha12", "task"]).agg(n_ok=("_ok", "sum"), n_ineligible=("_inel", "sum"), n_error=("_err", "sum"))
    for s, n in tgt.items():
        for t in tasks:
            r = g.loc[(s, t)] if (s, t) in g.index else None
            ok, ine, err = (int(r.n_ok), int(r.n_ineligible), int(r.n_error)) if r is not None else (0, 0, 0)
            rows.append(dict(sha12=s, task=t, n_target=int(n), n_ok=ok, n_ineligible=ine, n_error=err, n_missing=int(n) - ok - ine - err))
    return pd.DataFrame(rows)


def merge(in_root: str, split: str, hz: int, v9_dir: str, clip_index_dir: str, out: str, tasks=None, allow_mixed: bool = False,
          allow_missing: bool = False) -> dict:
    files = adr.output_files(in_root, split)
    if not files: raise SystemExit(f"no shard files under {in_root}/{split}")
    n_parts = sum("/part_" in f.replace(os.sep, "/") for f in files)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_raw = len(df)
    missing_cols = [c for c in adr.COLUMNS if c not in df.columns]
    if missing_cols: raise SystemExit(f"schema: missing columns {missing_cols}")
    df = adr.dedupe(df[adr.COLUMNS].astype(adr.DTYPES))
    tasks = list(tasks) if tasks else sorted(df.task.unique().tolist())

    npz = os.path.join(v9_dir, f"v9_labels_{split}.npz")
    clock = adr.load_v9_clock(npz, split)
    umap = adr.load_uuid_map([os.path.join(clip_index_dir, f"clip_index_v7.2_{s}.json") for s in ("train", "eval")])
    target = adr.enumerate_frames(clock, umap, hz)
    tk = target.set_index(["sha12", "v9_row"])["t_us"]

    # ---- integrity: every row is a target frame, with the target timestamp
    key = pd.MultiIndex.from_arrays([df.sha12.astype(str), df.v9_row.astype(int)])
    in_target = key.isin(tk.index)
    if (~in_target).any():
        raise SystemExit(f"{int((~in_target).sum())} rows are not target frames at --hz {hz} (e.g. "
                         f"{df[~in_target][['sha12', 'v9_row']].head(3).values.tolist()}); wrong --hz or wrong split dir?")
    t_bad = int((tk.reindex(key).to_numpy() != df.t_us.to_numpy()).sum())
    if t_bad: raise SystemExit(f"{t_bad} rows have t_us != round(now_s*1e6) of their v9 row")
    if (df.split.astype(str) != split).any(): raise SystemExit("rows of another split present")
    prov = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in ("model_id", "model_revision", "quantisation", "input_profile")}
    mixed = {c: v for c, v in prov.items() if len(v) > 1}
    if mixed and not allow_mixed: raise SystemExit(f"mixed provenance {mixed}; rerun the odd rows or pass --allow-mixed")
    cov = coverage(df, target, tasks)
    n_missing = int(cov.n_missing.sum())
    if n_missing and not allow_missing: raise SystemExit(f"{n_missing} (frame, task) pairs missing; finish the shards or pass --allow-missing")

    os.makedirs(out, exist_ok=True)
    p_df = os.path.join(out, f"v10_reasoning_dense_{split}.parquet")
    p_cov = os.path.join(out, f"v10_reasoning_dense_{split}.coverage.parquet")
    df.sort_values(["sha12", "v9_row", "task"], kind="stable").to_parquet(p_df, index=False, compression="zstd")
    cov.to_parquet(p_cov, index=False)
    e = df["error"].astype("string")
    try: head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    except Exception: head = None
    metas = [json.load(open(m)) for m in sorted(glob.glob(os.path.join(in_root, split, "*.meta.json")))]
    man = dict(
        schema=SCHEMA, split=split, hz=hz, tasks=tasks, built_unix=time.time(),
        n_rows=int(len(df)), n_rows_before_dedupe=n_raw, n_files=len(files), n_unfinalised_part_files=n_parts,
        n_clips=int(df.sha12.nunique()), n_target_frames=int(len(target)), n_target_clips=int(target.sha12.nunique()),
        by_task={t: dict(ok=int(((df.task == t) & e.isna()).sum()),
                         ineligible=int(((df.task == t) & e.str.startswith("ineligible:").fillna(False)).sum()),
                         error=int(((df.task == t) & e.notna() & ~e.str.startswith("ineligible:").fillna(False)).sum()))
                 for t in tasks},
        error_reasons={k: int(v) for k, v in e.dropna().str.replace(r":\s.*$", "", regex=True).value_counts().items()},
        n_missing=n_missing, front_only_rows=int(df.front_only.sum()), provenance=prov,
        generation=sorted({json.dumps(m.get("generation"), sort_keys=True) for m in metas}),
        dataset_revision=sorted({str(m.get("dataset_revision")) for m in metas}),
        runs=sorted({m.get("run_id") for m in metas}),
        coverage_columns=["sha12", "task", "n_target", "n_ok", "n_ineligible", "n_error", "n_missing"],
        coverage_per_clip=cov[["sha12", "task", "n_target", "n_ok", "n_ineligible", "n_error", "n_missing"]].values.tolist(),
        inputs_sha256={os.path.basename(npz): adr.sha256_file(npz)},
        outputs_sha256={os.path.basename(p_df): adr.sha256_file(p_df), os.path.basename(p_cov): adr.sha256_file(p_cov)},
        builder=dict(merge=dict(path="stack/scripts/merge_dense_reasoning.py", sha256=adr.sha256_file(__file__)),
                     generator=dict(path="stack/scripts/alpamayo_dense_reasoning.py", sha256=adr.sha256_file(adr.__file__)),
                     base_commit=head),
        note="Teacher text (nvidia/Alpamayo2-Super) is INHERITED quality, not ground truth. auto_labeling is conditioned on the ego "
             "FUTURE trajectory; both tasks on ego history: LABEL-side only, never an inference input.",
    )
    p_man = os.path.join(out, f"v10_reasoning_dense_{split}.MANIFEST.json")
    json.dump(man, open(p_man, "w"), indent=1)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_root", required=True); ap.add_argument("--split", required=True, choices=["train", "eval139"])
    ap.add_argument("--hz", type=int, default=10, choices=sorted(adr.HZ_STRIDE))
    ap.add_argument("--v9-dir", required=True); ap.add_argument("--clip-index-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None); ap.add_argument("--allow-mixed", action="store_true"); ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    man = merge(a.in_root, a.split, a.hz, a.v9_dir, a.clip_index_dir, a.out, a.tasks.split(",") if a.tasks else None, a.allow_mixed,
                a.allow_missing)
    print(json.dumps({k: v for k, v in man.items() if k != "coverage_per_clip"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
