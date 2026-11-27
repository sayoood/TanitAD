#!/usr/bin/env python3
"""EP-GEOM navtest decision read (PREREG_EPGEOM.md sections 4-6). RUN ONLY AFTER the MM landed the card AND fit_outcome == PASS-HELDOUT.

    epgeom_read.py --frozen FROZEN_PARAMS.json --frozen-sha256 <hex> --tables full_C1_s0of2.jsonl full_C1_s1of2.jsonl
                   --navtest-dir F:/safehead_cache/navtest --out EPGEOM_READ.json --picks-out epgeom_picks.json [--expect-n 12146]

Refuses (rc 2) if: the frozen sha256 differs, fit_outcome != PASS-HELDOUT, the navtest dump does not cover every table token (the dump must be COMPLETE),
or a code file differs from CODE.sha256. Control failure -> outcome INVALID. The table lookup is the T1 tracked per-proposal PDMS of the banked C1
64-proposal table; M4 (the four families) and CTRL-T(c) need the confirmation seam on the rows at `epgeom_picks.json` (HANDOFF.md).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epgeom_lib as L  # noqa: E402

C1_PDMS_X100 = 92.55477798611808          # MODEL_REGISTRY 14.6 (C1, 12,146 tokens)
BAR_P2 = 0.2316                           # 2*max(F, F_REFe), ENS-SCORE card 5
SUBS = ("nc", "dac", "ttc", "ep", "c", "ddc")
JUDGED = ("nc", "dac", "ttc", "ep", "c")  # DDC reported only


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_tables(paths):
    rows = []
    for p in paths:
        for ln in open(p):
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def load_geom(navtest_dir, tokens):
    """token -> (logits [64,6], props8 [64,8,3], pick). Only these three keys are read (never `s`)."""
    need, got = set(tokens), {}
    for f in sorted(glob.glob(os.path.join(navtest_dir, "*.npz"))):
        z = np.load(f)
        tk = z["token"]
        idx = [i for i, t in enumerate(tk) if str(t) in need]
        if not idx:
            continue
        lg, p8, pk = z["logits"], z["props8"], z["pick"]
        for i in idx:
            got[str(tk[i])] = (lg[i], p8[i], int(pk[i]))
    return got


def run(rows, geom, surr, tau, expect_n=None, roll=0):
    toks = [r["token"] for r in rows]
    logits = np.stack([geom[t][0] for t in toks])
    props8 = np.stack([geom[t][1] for t in toks])
    npick = np.array([geom[t][2] for t in toks])
    tpick = np.array([r["pick"] for r in rows])
    pdms = np.array([r["tr"]["pdms"] for r in rows], dtype=np.float64)
    new = L.select(logits, props8, tpick, surr, tau, roll=roll)
    return dict(logits=logits, props8=props8, npick=npick, tpick=tpick, new=new, tr={k: np.array([r["tr"][k] for r in rows], dtype=np.float64) for k in SUBS},
                pdms=pdms, logs=[r["log"] for r in rows], toks=toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--frozen-sha256", required=True)
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--navtest-dir", default="F:/safehead_cache/navtest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--picks-out", required=True)
    ap.add_argument("--expect-n", type=int, default=12146)
    a = ap.parse_args()
    if sha(a.frozen) != a.frozen_sha256.lower():
        print("REFUSE: FROZEN_PARAMS sha256 mismatch"); return 2
    fz = json.load(open(a.frozen))
    if fz["fit_outcome"] != "PASS-HELDOUT":
        print(f"REFUSE: fit_outcome {fz['fit_outcome']} (no navtest decision read, card section 3)"); return 2
    cs = {}
    for ln in open(os.path.join(HERE, "CODE.sha256")):
        h, n = ln.split()
        cs[n.lstrip("*")] = h
    bad = [n for n, h in cs.items() if n.endswith(".py") and os.path.exists(os.path.join(HERE, n)) and sha(os.path.join(HERE, n)) != h]
    if bad:
        print("REFUSE: CODE.sha256 mismatch", bad); return 2
    rows = load_tables(a.tables)
    geom = load_geom(a.navtest_dir, [r["token"] for r in rows])
    miss = [r["token"] for r in rows if r["token"] not in geom]
    if miss or len(rows) != a.expect_n:
        print(f"REFUSE: dump incomplete or wrong n: rows={len(rows)} expect={a.expect_n} missing_geometry={len(miss)}"); return 2
    R = run(rows, geom, fz["surrogate"], fz["tau"])
    n = len(rows)
    ar = np.arange(n)
    ctrl = {}
    # CTRL-JOIN: the dump's own pick equals the table's pick on every token (same proposal order, same model)
    ctrl["CTRL-JOIN"] = bool((R["npick"] == R["tpick"]).all())
    # CTRL-TAB: C1's pick reproduces the banked PDMS
    base = R["pdms"][ar, R["tpick"]]
    ctrl["CTRL-TAB"] = bool(abs(base.mean() * 100 - C1_PDMS_X100) < 1e-6)
    ctrl["CTRL-TAB_value"] = float(base.mean() * 100)
    ctrl["CTRL-ID"] = bool(np.array_equal(L.select(R["logits"], R["props8"], R["tpick"], fz["surrogate"], 1.5), R["tpick"]))
    red = L.select(R["logits"], R["props8"], R["tpick"], fz["surrogate"], fz["tau"], roll=1)
    ctrl["CTRL-RED_rate"] = float((red != R["new"]).mean())
    ctrl["CTRL-RED"] = ctrl["CTRL-RED_rate"] >= 0.01
    ctrl["CTRL-AGG_eq_pick_rate"] = float((L.aggregate_v1(R["logits"]).argmax(1) == R["tpick"]).mean())
    ctrl["CTRL-CODE"] = not bad
    ctrl["CTRL-SHA"] = True
    valid = all(v for k, v in ctrl.items() if isinstance(v, bool))
    B = L.Boot(R["logs"])
    new = R["pdms"][ar, R["new"]]
    d = (new - base) * 100
    m1 = B.ci(d)
    subs = {}
    for k in SUBS:
        subs[k] = B.ci((R["tr"][k][ar, R["new"]] - R["tr"][k][ar, R["tpick"]]) * 100)
    p1 = m1["lo"] > 0
    p2 = m1["point"] >= BAR_P2
    g_ok = not any(subs[k]["hi"] < 0 for k in JUDGED)
    lg = np.array(sorted(set(R["logs"])))
    half = np.isin(np.array(R["logs"]), lg[::2])
    r_ok = bool(d[half].mean() > 0 and d[~half].mean() > 0)
    if not valid:
        outcome = "INVALID"
    elif m1["hi"] < 0 or not g_ok:
        outcome = "ADVERSE"
    elif p1 and p2 and g_ok and r_ok:
        outcome = "ADOPT-PENDING-SEAM"      # final ADOPT also needs M4 (no separated-adverse family) and CTRL-T on the confirmation seam
    elif p1:
        outcome = "SEPARATED-SMALL"
    else:
        outcome = "NULL"
    rep = {"tier": "T1 (banked C1 tracked 64-proposal table, T1 lookup under CTRL-TAB)", "estimator": f"paired log-cluster bootstrap, {L.NBOOT} draws, seed {L.SEED}, 95% percentile, full_set means",
           "n": n, "frozen": fz, "M1_delta_x100": m1, "M2_sub_delta_x100": subs, "bars": {"P1": p1, "P2": p2, "G": g_ok, "R_log_parity_halves": r_ok, "P2_bar": BAR_P2},
           "M5": {"switch_rate": float((R["new"] != R["tpick"]).mean()), "c1_pdms_x100": float(base.mean() * 100), "rule_pdms_x100": float(new.mean() * 100)},
           "controls": ctrl, "outcome": outcome, "goal_status": "BELOW GOAL (94.55)" if new.mean() * 100 < 94.55 else "AT/ABOVE GOAL"}
    json.dump(rep, open(a.out, "w"), indent=1)
    json.dump({t: int(k) for t, k in zip(R["toks"], R["new"])}, open(a.picks_out, "w"))
    print(json.dumps({k: rep[k] for k in ("M1_delta_x100", "bars", "controls", "outcome")}, indent=1))
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
