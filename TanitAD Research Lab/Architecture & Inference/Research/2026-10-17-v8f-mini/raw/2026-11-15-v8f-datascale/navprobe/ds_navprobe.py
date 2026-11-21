#!/usr/bin/env python3
"""DATASCALE nav probe (pod-side, CPU, metadata only; read-only).
Measures what the NVIDIA repo file navigation/navigation.parquet holds and which clips it covers:
schema, rows, distinct clips, rows per clip, timestamp spacing, nav_text vocabulary, texts per clip,
and coverage of our clip lists (train4572, eval141, the 10k selection, the eligible pool) and of the
NVIDIA split labels. Writes one JSON. No address, port, pod id or token in this file."""
import json
import os
import sys

import numpy as np
import pandas as pd

META = "/root/datascale/meta"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/datascale/navprobe.json"


def read_ids(p):
    return {ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()}


def main():
    rep = {"file": f"{META}/navigation/navigation.parquet",
           "bytes": os.path.getsize(f"{META}/navigation/navigation.parquet")}
    nav = pd.read_parquet(f"{META}/navigation/navigation.parquet")
    rep["index_name"] = str(nav.index.name)
    rep["index_names"] = [str(x) for x in nav.index.names]
    nav = nav.reset_index()
    rep["columns"] = {c: str(t) for c, t in nav.dtypes.items()}
    rep["rows"] = int(len(nav))
    cid = next((c for c in nav.columns if c.lower() in ("clip_id", "clip", "clip_uuid")), None)
    tcol = next((c for c in nav.columns if "time" in c.lower()), None)
    txt = next((c for c in nav.columns if "nav" in c.lower() and not pd.api.types.is_numeric_dtype(nav[c])), None)
    rep["clip_col"], rep["time_col"], rep["text_col"] = cid, tcol, txt
    if cid is None:
        rep["error"] = "no clip id column"
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1)
        return
    nav[cid] = nav[cid].astype(str)
    clips = set(nav[cid].unique())
    rep["distinct_clips"] = len(clips)
    rpc = nav.groupby(cid).size()
    rep["rows_per_clip"] = {q: float(rpc.quantile(q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)}
    if tcol is not None:
        t = nav[[cid, tcol]].sort_values([cid, tcol])
        dt = t.groupby(cid)[tcol].diff().dropna().astype("float64")
        rep["time_dtype"] = str(nav[tcol].dtype)
        rep["time_min_max"] = [float(nav[tcol].min()), float(nav[tcol].max())]
        rep["time_step_quantiles"] = {q: float(dt.quantile(q)) for q in (0.01, 0.5, 0.99)}
        span = t.groupby(cid)[tcol].agg(lambda s: float(s.max() - s.min()))
        rep["time_span_per_clip_quantiles"] = {q: float(span.quantile(q)) for q in (0.05, 0.5, 0.95)}
    if txt is not None:
        s = nav[txt]
        rep["text_null"] = int(s.isna().sum())
        rep["text_empty"] = int((s.fillna("").astype(str).str.strip() == "").sum())
        vc = s.fillna("<NULL>").astype(str).value_counts()
        rep["text_distinct"] = int(len(vc))
        rep["text_top40"] = [[k, int(v)] for k, v in vc.head(40).items()]
        tpc = nav.groupby(cid)[txt].nunique()
        rep["distinct_texts_per_clip"] = {q: float(tpc.quantile(q)) for q in (0.05, 0.5, 0.95, 1.0)}
        rep["clips_with_one_text"] = int((tpc == 1).sum())
        words = s.dropna().astype(str).str.lower().str.extract(r"^([a-z]+(?: [a-z]+)?)")[0].value_counts()
        rep["text_first_two_words_top30"] = [[k, int(v)] for k, v in words.head(30).items()]
    lists = {"train4572": "/root/ops/train4572.txt", "eval141": "/root/ops/clips141.txt",
             "sel10k": "/root/datascale/selection/clips_10k.txt", "pilot_ok200": "/root/datascale/selection/pilot_ok.txt"}
    cov = {}
    for k, p in lists.items():
        if os.path.exists(p):
            ids = read_ids(p)
            cov[k] = {"n": len(ids), "covered": len(ids & clips)}
    o = pd.read_parquet("/root/datascale/selection/order_all_eligible.parquet")
    ocol = next((c for c in o.columns if "clip" in c.lower()), None)
    oid = set(map(str, o[ocol] if ocol else o.index))
    cov["eligible_pool"] = {"n": len(oid), "covered": len(oid & clips)}
    sp = pd.read_parquet(f"{META}/splits/default.parquet")
    scol = next((c for c in sp.columns if "split" in c.lower()), None)
    if scol is not None:
        lab = sp[scol].astype(str)
        idx = sp.index.astype(str)
        for v in sorted(lab.unique()):
            ids = set(idx[lab.values == v])
            cov[f"nvidia_default_{v}"] = {"n": len(ids), "covered": len(ids & clips)}
    ci = pd.read_parquet(f"{META}/clip_index.parquet")
    ccol = next((c for c in ci.columns if c.lower() == "clip_id"), None)
    all_ids = set(map(str, ci[ccol])) if ccol else set(ci.index.astype(str))
    cov["clip_index_all"] = {"n": len(all_ids), "covered": len(all_ids & clips), "nav_clips_not_in_index": len(clips - all_ids)}
    rep["coverage"] = cov
    ex = sorted(clips & read_ids("/root/ops/train4572.txt"))[:2] if os.path.exists("/root/ops/train4572.txt") else []
    rep["example_sequences_train_corpus"] = {}
    for c in ex:
        g = nav[nav[cid] == c].sort_values(tcol) if tcol else nav[nav[cid] == c]
        chg = g[g[txt] != g[txt].shift()] if txt else g
        rep["example_sequences_train_corpus"][c] = [[float(r[tcol]) if tcol else None, str(r[txt]) if txt else None]
                                                   for _, r in chg.head(12).iterrows()]
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print("NAVPROBE OK", OUT, rep["rows"], rep["distinct_clips"])


if __name__ == "__main__":
    main()
