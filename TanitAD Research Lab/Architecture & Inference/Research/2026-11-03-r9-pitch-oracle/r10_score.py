#!/usr/bin/env python3
"""r10_score.py -- the R10-A read: static reproduction (VOID gate) + per-band paired deltas for H and C + the committed-outcome verdict.

Inputs are the per-arm extraction stats of ``e0d_score.py extract`` (the E0D instrument, UNCHANGED), one per arm, run on each arm's raster dir:

  export E0D_CI_PATH=<tree>/taniteval/taniteval/ci.py
  for A in static H C; do python -X utf8 e0d_score.py extract --rast E:/r10_pitch_out/$A/rast --split-json <DEV_HOLD_SPLIT.json> --split dev \
        --out E:/r10_pitch_out/stats_$A.npz --workers 6; done        # ~5 min each, CPU
  python -X utf8 r10_score.py read --static stats_static.npz --H stats_H.npz --C stats_C.npz --e0d-result <E0D_RESULT.json> --out-dir raw [--e0d-score <e0d_score.py>]

Protocol = E0D's (thresholds fitted per arm on one half of DEV, read on the other, cross-fit over all DEV; episode-cluster bootstrap B 2000 seed 0, PAIRED for the deltas).
Decision cells (pre-registered in HANDOFF.md): pooled 20-60 m drivable IoU and boundary recall @0.3 m, H and C minus static; BUILT iff delta > +0.010 and the CI excludes 0.
Tier T0 (perception instrument vs SAM3 GT raster, label level); a CI answers 'another draw of EPISODES' only (same checkpoint, no training variance enters).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAR = 0.010


def load_e0d(path=None):
    cands = [path or "", os.environ.get("E0D_SCORE_PATH", ""), str(HERE.parent / "2026-10-27-perception-redesign" / "e0" / "e0d_score.py"),
             "C:/Users/Admin/scratch_r10/ref/e0d_score_tip.py"]
    for c in cands:
        if c and Path(c).is_file():
            spec = importlib.util.spec_from_file_location("e0d_score", c)
            m = importlib.util.module_from_spec(spec)
            sys.modules["e0d_score"] = m
            spec.loader.exec_module(m)
            m.__path_used__ = c
            return m
    raise SystemExit("e0d_score.py not found: pass --e0d-score or set E0D_SCORE_PATH")


METRICS = {"drv_iou": ("drv", "f_iou"), "bnd_recall_0p3m": ("br", "f_ratio"), "lane_ioudil_0p5m": ("ld2", "f_ioudil")}


def comps_for(E, bank, thr_fit, thr_swap, name):
    return E.band_components(bank, "OWN", [(bank.score, thr_fit), (bank.fit, thr_swap)], name, E.BAND4[name], "perband")


def pooled(c_a, c_b):
    return {k: c_a[k] + c_b[k] for k in c_a}


def delta_cell(E, ci, ca, cs, eid, key, fname, nb):
    f = getattr(E, fname)
    A, B = ca[key], cs[key]
    if key == "ld2":
        A, B = A[:, :3], B[:, :3]
    d = E.paired(ci, A, B, eid, f, nb)
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items() if k in ("delta", "lo", "hi", "separated", "n_episodes", "n_windows", "point_a", "point_b", "a", "b")}


def read(a):
    E = load_e0d(a.e0d_score)
    ci = E._load_ci()
    nb = int(a.n_boot)
    banks = {"static": E.Bank(a.static, quiet=True), "H": E.Bank(a.H, quiet=True), "C": E.Bank(a.C, quiet=True)}
    b0 = banks["static"]
    for k, b in banks.items():
        assert np.array_equal(b.e, b0.e) and np.array_equal(b.w, b0.w), f"arm {k}: window set differs from static"
    thr = {k: (E.thresholds_from(b, b.fit), E.thresholds_from(b, b.score)) for k, b in banks.items()}
    res = {"tool": "r10_score.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "e0d_score": getattr(E, "__path_used__", None),
           "e0d_score_sha256": hashlib.sha256(Path(E.__path_used__).read_bytes()).hexdigest(), "ci_sha256": ci.__path_sha256__, "n_boot": nb, "seed": 0,
           "tier": "T0 perception instrument (map head vs SAM3 GT raster); ORACLE pitch, not a deployable arm", "n_windows": int(len(b0.e)),
           "disjointness": b0.disjoint, "bar": BAR}
    # ---- (1) static reproduction vs E0D (primary SCORE half, thresholds from FIT)
    prim = E.score_rows(ci, b0, "OWN", [(b0.score, thr["static"][0])], "perband", nb)
    ref = json.load(open(a.e0d_result, encoding="utf-8"))["PRIMARY_SCORE_half_thresholds_from_FIT_perband"]
    rep, worst = {}, 0.0
    for band in E.BAND4:
        for m in METRICS:
            mine, theirs = prim[band][m].get("point"), ref[band][m].get("point")
            dlt = None if mine is None or theirs is None else round(abs(mine - theirs), 5)
            rep.setdefault(band, {})[m] = {"static_now": mine, "E0D": theirs, "abs_diff": dlt}
            worst = max(worst, dlt or 0.0)
    res["static_reproduction"] = {"cells": rep, "worst_abs_diff": worst, "tolerance": 0.002, "PASS": bool(worst <= 0.002),
                                  "rule": "VOID gate: if not PASS the oracle comparison is reported VOID"}
    # ---- (2) deltas, cross-fit, thresholds refit per arm (primary) and the static thresholds applied to the arm (beside)
    out = {}
    for arm in ("H", "C"):
        bk = banks[arm]
        per = {"own_thresholds": {}, "static_thresholds": {}}
        for tag, (tf, ts), (tf_s, ts_s) in (("own_thresholds", thr[arm], thr["static"]), ("static_thresholds", thr["static"], thr["static"])):
            cs_b, ca_b, eid = {}, {}, None
            for band in E.BAND4:
                cs_b[band], eid_s = comps_for(E, b0, tf_s, ts_s, band)
                ca_b[band], eid = comps_for(E, bk, tf, ts, band)
                assert np.array_equal(eid, eid_s)
            cells = {}
            for band in E.BAND4:
                for m, (key, fn) in METRICS.items():
                    cells[f"{band}|{m}"] = delta_cell(E, ci, ca_b[band], cs_b[band], eid, key, fn, nb)
            ps, pa = pooled(cs_b["20-40"], cs_b["40-60"]), pooled(ca_b["20-40"], ca_b["40-60"])
            for m, (key, fn) in METRICS.items():
                cells[f"20-60 pooled|{m}"] = delta_cell(E, ci, pa, ps, eid, key, fn, nb)
            per[tag] = cells
        out[arm] = per
    res["deltas_arm_minus_static"] = out
    # ---- (3) the committed-outcome verdict (decision cells = pooled 20-60, own thresholds)
    dec, built = {}, False
    for arm in ("H", "C"):
        for m in ("drv_iou", "bnd_recall_0p3m"):
            c = out[arm]["own_thresholds"][f"20-60 pooled|{m}"]
            ok = bool(c["delta"] > BAR and c["lo"] > 0)
            dec[f"{arm}|{m}"] = {**c, "passes_bar": ok}
            built = built or ok
    res["verdict"] = {"decision_cells": dec, "R10_head_BUILT": bool(built and res["static_reproduction"]["PASS"]),
                      "static_reproduction_PASS": res["static_reproduction"]["PASS"],
                      "text": ("VOID: the static arm does not reproduce E0D" if not res["static_reproduction"]["PASS"] else
                               "R10 head BUILT: a decision cell clears +0.010 with a separated CI" if built else
                               "R10 OUT for refcv9: no decision cell clears +0.010 with a separated CI; next pitch-free lever = the depth-distribution lift (refcv10)")}
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "R10_RESULT.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    lines = ["| cell (arm minus static, cross-fit, own thresholds) | H delta [CI] | C delta [CI] |", "|---|---|---|"]
    for k in out["H"]["own_thresholds"]:
        h, c = out["H"]["own_thresholds"][k], out["C"]["own_thresholds"][k]
        f = lambda x: "n/a" if x.get("delta") is None else f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{' *' if x.get('separated') else ''}"  # noqa: E731
        lines.append(f"| {k} | {f(h)} | {f(c)} |")
    (outdir / "R10_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"static_reproduction": {"worst_abs_diff": worst, "PASS": res["static_reproduction"]["PASS"]}, "verdict": res["verdict"]["text"]}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    for n in ("static", "H", "C"):
        r.add_argument(f"--{n}", required=True)
    r.add_argument("--e0d-result", required=True)
    r.add_argument("--e0d-score")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--out-dir", required=True)
    return read(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
