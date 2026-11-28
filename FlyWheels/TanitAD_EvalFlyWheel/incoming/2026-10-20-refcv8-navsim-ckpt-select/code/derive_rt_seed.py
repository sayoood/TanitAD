#!/usr/bin/env python3
"""Derive ``R8_RT`` at a NON-ZERO inference seed from that seed's ``R8_LEGAL`` rows (TANITAD VENV, CPU, no forward).

WHY THIS EXISTS: ``run_bridge8.py --retime`` derives ``R8_RT`` ONLY from ``rows_R8_LEGAL.jsonl`` (seed 0; run_bridge8.py:281-297).
D-R8-CKPT-RULE (2) needs ``R8_RT`` at inference seeds 0 AND 1, so seed 1 needs ``R8_RT_s1`` from ``rows_R8_LEGAL_s1.jsonl``.
This script is that derivation with the bridge's OWN functions -- ``refcv8_bridge.load_retime`` / ``retime_plan8`` /
``knots_to_navsim`` and ``run_bridge8.write_seam`` / ``read_rows`` -- called exactly as run_bridge8.py:281-297 calls them,
with the source rows file and the arm name as the only differences.  NEW CODE: it needs Master-Mind acceptance before it
produces a decision row.

    python derive_rt_seed.py --pkg-code <pkg/code> --out <bridge out dir> --inputs <inputs.json(.gz)> --split navtest
        --src-arm R8_LEGAL_s1 --dst-arm R8_RT_s1 --retime-table <gains.json> --retime-table-sha256 <hex>
        --retime-stage-file <retime_stage.py> [--tokens-file <subset.json>]
"""
import argparse
import gzip
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pkg-code", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--inputs", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--src-arm", default="R8_LEGAL_s1")
ap.add_argument("--dst-arm", default="R8_RT_s1")
ap.add_argument("--retime-table", required=True)
ap.add_argument("--retime-table-sha256", required=True)
ap.add_argument("--retime-stage-file", required=True)
ap.add_argument("--tokens-file", default="")
a = ap.parse_args()
sys.path.insert(0, a.pkg_code)
import numpy as np  # noqa: E402

import refcv8_bridge as B  # noqa: E402
import run_bridge8 as RB8  # noqa: E402

if a.src_arm not in B.ARMS8 or B.ARMS8[a.src_arm]["frames"] != "ST" or B.ARMS8[a.src_arm]["speed"] != "legal":
    sys.exit(f"[derive_rt] {a.src_arm} is not an ST / LEGAL forward arm")
opener = gzip.open if a.inputs.endswith(".gz") else open
toks = json.load(opener(a.inputs, "rt", encoding="utf-8"))["tokens"]
if a.split == "navtest":
    for r in toks.values():
        r["stage"] = 1
s1 = sorted(t for t, r in toks.items() if r["stage"] == 1)
s2 = sorted(t for t, r in toks.items() if r["stage"] == 2)
if a.tokens_file:
    keep = set(json.load(open(a.tokens_file, encoding="utf-8"))["tokens"])
    s1 = [t for t in s1 if t in keep]
    s2 = [t for t in s2 if t in keep]
order = s1 + s2
RS, table, rt_rec = B.load_retime(a.retime_stage_file, a.retime_table, a.retime_table_sha256)
src = RB8.read_rows(os.path.join(a.out, f"rows_{a.src_arm}.jsonl"))
rt = {}
for t in order:
    r = src.get(t)
    if r is None:
        continue
    if r["source"] != "refcv8":
        rt[t] = r
        continue
    knots = B.retime_plan8(RS, table, np.asarray(r["knots"], np.float32), np.asarray(r["prior_emitted_path"], np.float32))
    rt[t] = {"token": t, "source": "refcv8", "knots": knots.tolist(), "poses": B.knots_to_navsim(knots).tolist()}
man = {"split": a.split, "spec": f"{a.src_arm}'s emitted plan and emitted prior re-timed (seed {B.ARMS8[a.src_arm]['seed']}); "
                                 f"no extra forward", "derived_from": f"rows_{a.src_arm}.jsonl", "device": ["derived"],
       "precision": ["derived"], "retime": rt_rec, "tool": "derive_rt_seed.py (NEW; run_bridge8.py:281-297 with the source "
                                                            "rows / arm name substituted)"}
m = RB8.write_seam(a.out, a.dst_arm, order, rt, toks, man)
print(f"[derive_rt] {a.dst_arm}: {m.get('n_model_rows')} rows, missing {m.get('n_missing')}, partial {m.get('partial')}", flush=True)
