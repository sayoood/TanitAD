#!/usr/bin/env python3
"""The selection-time TTC gate read, exactly as PREREG_O2_NEXT_TTCGATE.md (sha256 pinned below, written before any gated pick was computed).

Rule T = navsim_v1_lane x p_TTC over the sigmoid of the raw logits; the same rule on both sides of every pair.  Bar = PREREG_RL_S1.md 11.3 (B1 g5 point deltas >= -0.5,
B2 PDMS lower bound > 0) at R0-B u500 + u1000 vs C0-A and R0b-B u250 + u500 vs C0-A and vs R0b-A.  Exploratory: a PASS names a lever only.

    python ttcgate_read.py --out ttcgate_read.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_common as CC  # noqa: E402
import calib_read as CR  # noqa: E402

RV = CC.RV
PREREG = "PREREG_O2_NEXT_TTCGATE.md"
PREREG_SHA256 = "f4565ac70c0da859d51c7b2fdb7ab9c73c033f25f45718a7826f44054d953487"
PREREG_WRITTEN_UTC = "2026-10-14T23:39:16Z"


def picks_gate(z, k=1.0):
    p = RV.sigmoid(z)
    return (CC.sel_v1_lane(p) * p[..., 3] ** k).argmax(-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default=CC.DL_DEFAULT)
    ap.add_argument("--x", default=CC.X_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    got = CC.sha(os.path.join(CC.HERE, PREREG))
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG} changed since registration: {got}")
    res = {"tool": "ttcgate_read/1", "prereg": {"file": PREREG, "sha256": got, "written_utc": PREREG_WRITTEN_UTC},
           "rule_T": "NC*DAC*TTC*(5EP+5TTC+2C)/12*LANE over sigmoid(raw logits)", "tier": "RL-val (selection tier; never an endpoint)",
           "estimator": "paired log-cluster bootstrap, proxy_eval.boot, 10,000, seed 20260927 -- another draw of LOGS only", "status": "EXPLORATORY: a PASS names a lever only"}
    res["inputs_sha256"] = CC.verify(a.dl, sorted(set(CR.DUMPS.values())) + ["data/data/rlfan_pdm_val.tar.gz", "data/data/rlfan_terms_val.tar.gz"])
    keys, tg, ck, s, st = CC.load_val(a.x)
    res["rl_val_join"], res["n_sets"] = st, len(keys)
    L = {d: CC.load_scores_aligned(os.path.join(a.dl, rel), keys, ck) for d, rel in CR.DUMPS.items()}
    raw = {d: CC.picks_from_logits(z) for d, z in L.items()}
    T = {d: picks_gate(z) for d, z in L.items()}
    ctrl = {"k0_changed": {f"{d[0]}_{d[1]}": int((picks_gate(z, 0.0) != raw[d]).sum()) for d, z in L.items()},
            "T_changed": {f"{d[0]}_{d[1]}": int((T[d] != raw[d]).sum()) for d in L}}
    ctrl["ok"] = bool(all(v == 0 for v in ctrl["k0_changed"].values()) and all(v > 0 for v in ctrl["T_changed"].values()))
    res["controls"] = ctrl
    if not ctrl["ok"]:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        raise SystemExit("VOID: control failed")
    dec = CR.read_set(CR.DECIDING, T, s, keys)
    res["deciding"] = dec
    res["verdict"] = CR.verdict(dec)
    res["T_effect_on_C0"] = {t: CC.delta_table(s, T[("C0A", t)], raw[("C0A", t)], keys) for t in ("update250", "update500", "update750", "update1000")}
    res["R_under_T_minus_rawC0"] = {f"{rd}_{t}": CC.delta_table(s, T[(rd, t)], raw[("C0A", t)], keys) for _a, rd, t, _c in CR.DECIDING[:4]}
    res["runtime_s"] = time.time() - t0
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("ZZTTCGATE_OK", res["verdict"], a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
