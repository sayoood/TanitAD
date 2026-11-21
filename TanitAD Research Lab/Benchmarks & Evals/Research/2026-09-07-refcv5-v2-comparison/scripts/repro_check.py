#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repro_check.py — THE CONTROL. Did the dev-box harness reproduce refcv4b's
PUBLISHED numbers, field by field, against the banked landing artifact?

⭐ WHY THIS IS THE POINT. refcv5-v2 cannot be scored — it does not exist yet.
But refcv4b @40,284 can, and its numbers are published, so re-rolling it through
this harness gives a control THAT MUST READ A KNOWN VALUE. A harness that merely
runs proves nothing; a harness that reproduces a value it could get wrong is
evidence.

Reads the PUBLISHED numbers from the primary artifact
(``…/2026-09-06-refcv4b-landing/raw/refcv4b_t1.json``) rather than from any
summary, so the comparison cannot drift with a doc. Prints BOTH sets side by
side and NEVER adjusts either.

⚠️ KNOWN OPEN ITEM, not a bug of this harness: the `os` ADE is unsettled between
rolls — the registry holds 0.2975, two independent surfaces read 0.2965, and the
model-free arms reproduce EXACTLY on all three. A third-decimal difference on
`os` is therefore expected; a difference in the PAIRED MARGINS is not.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def dump_ade(d, arms):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not fs:
        sys.exit("⛔ no ep*.npz under %r — RETRY, a control reading 0 is a mount flap" % d)
    acc, eid = {a: [] for a in arms}, []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        gt = np.asarray(z["g"], dtype=np.float64)
        for a in arms:
            if a in z.files:
                acc[a].append(np.linalg.norm(np.asarray(z[a], dtype=np.float64) - gt,
                                             axis=-1).mean(axis=1))
        eid.append(np.full(gt.shape[0], int(np.asarray(z["eid"]).ravel()[0])))
    return ({a: np.concatenate(v) for a, v in acc.items() if v},
            np.concatenate(eid))


ROWS_ARM = [
    ("ADE (dense) m", ("intervals", "metrics", "ade_dense_m", "mean"), 4),
    ("LON speed_MAE m/s", ("four_families", "longitudinal", "speed_mae_mps"), 4),
    ("LON speed_bias m/s", ("four_families", "longitudinal", "speed_bias_mps"), 4),
    ("LON tgt_speed_acc@0.5", ("four_families", "longitudinal", "target_speed_acc",
                               "within_0.5_mps"), 4),
    ("LON tgt_speed_acc@1.0", ("four_families", "longitudinal", "target_speed_acc",
                               "within_1.0_mps"), 4),
    ("LON along_MAE m", ("four_families", "longitudinal", "along_mae_m"), 4),
    ("LON dk n", ("four_families", "longitudinal", "distance_keeping", "n"), 0),
    ("LON dk mean_headway_min m", ("four_families", "longitudinal", "distance_keeping",
                              "mean_headway_min_m"), 4),
    ("LON dk mean_time_gap_min s", ("four_families", "longitudinal", "distance_keeping",
                               "mean_time_gap_min_s"), 4),
    ("LON dk mean_min_ttc s", ("four_families", "longitudinal", "distance_keeping",
                          "mean_min_ttc_s"), 4),
    ("LAT heading_MAE deg", ("four_families", "lateral", "heading_mae_deg"), 4),
    ("LAT yaw_rate_MAE d/s", ("four_families", "lateral", "yaw_rate_mae_degps"), 4),
    ("LAT curvature_MAE 1/m", ("four_families", "lateral", "curvature_mae_1pm"), 6),
    ("LAT cross_MAE m", ("four_families", "lateral", "cross_mae_m"), 4),
    ("LAT n_steps_curv", ("four_families", "lateral", "n_steps_curvature"), 0),
    ("LAT excluded<min_ds", ("four_families", "lateral", "excluded_below_min_ds"), 0),
    ("TAC lat acc", ("four_families", "tactical", "lateral_decision", "accuracy"), 4),
    ("TAC lat kappa", ("four_families", "tactical", "lateral_decision", "kappa"), 4),
    ("TAC lon acc", ("four_families", "tactical", "longitudinal_decision", "accuracy"), 4),
    ("TAC lon kappa", ("four_families", "tactical", "longitudinal_decision", "kappa"), 4),
    ("TAC goal FDE m", ("four_families", "tactical", "goal_setting",
                        "goal_point_error_m"), 4),
    ("TAC goal bearing MAE", ("four_families", "tactical", "goal_setting",
                              "goal_bearing_mae_deg"), 4),
]
PAIRED_KEYS = ["paired_os_minus_ha", "paired_os_minus_ha0ext", "paired_os_minus_ha0",
               "paired_os_minus_navshuf", "paired_os_minus_navzero",
               "paired_os_navzero_minus_ha0ext", "paired_ha_minus_ha0"]


def fmt(v, nd):
    # ⚠️ some fields are DICTS in one vintage and scalars in another
    # (`target_speed_acc` grew a {value, tol} block). Render, never crash — a
    # formatter that dies mid-table hides every row after it.
    if v is None:
        return "    --   "
    if isinstance(v, dict):
        for k in ("value", "acc", "mean"):
            if k in v:
                return fmt(v[k], nd)
        return "  {dict} "
    if isinstance(v, (list, tuple)):
        return "  [list] "
    try:
        if nd == 0:
            return "%9d" % int(v)
        return ("%9." + str(nd) + "f") % float(v)
    except (TypeError, ValueError):
        return "%9s" % str(v)[:9]


def _num(v):
    """Scalar behind a value that may be wrapped in a dict; None if there isn't one."""
    if isinstance(v, dict):
        for k in ("value", "acc", "mean"):
            if k in v:
                return _num(v[k])
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def main():
    pub_p, new_p = sys.argv[1], sys.argv[2]
    pub_dump = sys.argv[3] if len(sys.argv) > 3 else None   # refcv3 prior dump
    new_dump = sys.argv[4] if len(sys.argv) > 4 else None
    out_p = sys.argv[5] if len(sys.argv) > 5 else None
    pub = json.load(open(pub_p, encoding="utf-8"))
    new = json.load(open(new_p, encoding="utf-8"))
    rec = {"published_source": pub_p, "reproduced_source": new_p, "rows": [],
           "paired": [], "grid": {}, "verdict": None}
    W = 96
    print("=" * W)
    print("CONTROL — did the dev-box RTX 4060 harness reproduce refcv4b's PUBLISHED landing?")
    print("  PUBLISHED : %s  (A40, %s)" % (pub_p, pub.get("ckpt")))
    print("  REPRODUCED: %s  (dev-box RTX 4060, %s)" % (new_p, new.get("ckpt")))
    print("=" * W)
    for k in ("n_windows", "n_episodes", "dt_s", "horizon_steps", "gt_key"):
        a, b = pub.get(k), new.get(k)
        same = (a == b)
        rec["grid"][k] = {"published": a, "reproduced": b, "same": same}
        print("  %-16s published %-12s reproduced %-12s  %s"
              % (k, a, b, "MATCH" if same else "⛔ DIFFERS"))
    print()
    hdr = "%-24s %-9s %9s %9s %11s" % ("metric", "arm", "published", "devbox", "delta")
    print(hdr)
    print("-" * W)
    n_exact = n_close = n_off = 0
    for arm in ("os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero"):
        pa, na = g(pub, "arms", arm), g(new, "arms", arm)
        if pa is None or na is None:
            continue
        for label, path, nd in ROWS_ARM:
            a, b = g(pa, *path), g(na, *path)
            if a is None and b is None:
                continue
            na_, nb_ = _num(a), _num(b)
            d = (nb_ - na_) if (na_ is not None and nb_ is not None) else None
            tag = ("EXACT" if d == 0 else
                   ("~" if (d is not None and abs(d) <= (10 ** -max(nd - 1, 1) if nd else 0.5))
                    else "⛔"))
            if d == 0:
                n_exact += 1
            elif tag == "~":
                n_close += 1
            elif d is not None:
                n_off += 1
            rec["rows"].append({"arm": arm, "metric": label, "published": a,
                                "reproduced": b, "delta": d, "tag": tag,
                                "is_count": (nd == 0)})
            print("%-24s %-9s %s %s %11s %s"
                  % (label, arm, fmt(a, nd), fmt(b, nd),
                     ("%.6g" % d) if d is not None else "--", tag))
        print("-" * W)
    print()
    print("PAIRED MARGINS  (⛔ the ones that must NOT differ)")
    print("%-34s %10s %10s %10s | %10s %10s %10s"
          % ("cell", "pub delta", "pub lo", "pub hi", "new delta", "new lo", "new hi"))
    for k in PAIRED_KEYS:
        pa, na = g(pub, "paired_decision_grade", k), g(new, "paired_decision_grade", k)
        if not isinstance(pa, dict) or not isinstance(na, dict):
            continue
        for m in ("ade_m", "speed_mae_mps"):
            A, B = pa.get(m), na.get(m)
            if not isinstance(A, dict) or not isinstance(B, dict):
                continue
            rec["paired"].append({"cell": k, "metric": m,
                                  "published": {x: A.get(x) for x in ("delta", "lo", "hi", "separated")},
                                  "reproduced": {x: B.get(x) for x in ("delta", "lo", "hi", "separated")},
                                  "separated_agrees": A.get("separated") == B.get("separated")})
            print("%-34s %10.4f %10.4f %10.4f | %10.4f %10.4f %10.4f  sep %s->%s %s"
                  % (k + "." + m, A["delta"], A["lo"], A["hi"], B["delta"], B["lo"], B["hi"],
                     A.get("separated"), B.get("separated"),
                     "AGREE" if A.get("separated") == B.get("separated") else "⛔ DISAGREE"))
    if pub_dump and new_dump:
        print()
        print("CROSS-CHECKPOINT CONTROL — refcv4b - refcv3 @40,284 on the SAME 4,823 windows")
        print("  published: -0.1444 [-0.1647, -0.1227] separated (…/raw/paired_v4b_vs_v3_FULL.json)")
        pr, e_pr = dump_ade(pub_dump, ("os", "ha", "ha0"))
        nw, e_nw = dump_ade(new_dump, ("os", "ha", "ha0"))
        if not np.array_equal(e_pr, e_nw):
            print("  ⛔ episode ids differ — refusing to pair")
        else:
            r = paired_episode_cluster_bootstrap(nw["os"], pr["os"], e_nw, n_boot=2000, seed=0)
            print("  reproduced: %.4f [%.4f, %.4f] separated=%s   (refcv4b.os - refcv3.os)"
                  % (r["delta"], r["lo"], r["hi"], r["separated"]))
            rec["cross_ckpt"] = {"published": {"delta": -0.1444, "lo": -0.1647,
                                               "hi": -0.1227, "separated": True},
                                 "reproduced": {k: r[k] for k in ("delta", "lo", "hi", "separated")}}
            for arm in ("ha", "ha0"):
                if arm in pr and arm in nw:
                    c = paired_episode_cluster_bootstrap(nw[arm], pr[arm], e_nw,
                                                         n_boot=500, seed=0)
                    print("  MODEL-FREE CONTROL %-4s delta %.10f [%.10f, %.10f] "
                          "n_changed %d/%d  (must be 0)"
                          % (arm, c["delta"], c["lo"], c["hi"],
                             int((nw[arm] != pr[arm]).sum()), nw[arm].size))
                    rec.setdefault("model_free_controls", {})[arm] = {
                        "delta": c["delta"], "lo": c["lo"], "hi": c["hi"],
                        "n_changed": int((nw[arm] != pr[arm]).sum()),
                        "n": int(nw[arm].size)}
    print()
    # ⛔ The two populations are NOT the same claim and must not be summed.
    # MODEL-FREE arms are functions of the CORPUS alone — a single non-zero
    # delta there means the surface differs and the panel is void. MODEL arms
    # run an argmax over 117 anchors whose ties break differently per GPU
    # (MEASURED: A40 landing os 0.2975 vs Thor re-roll 0.2965), so a small
    # deviation there is expected and is bounded, not zero.
    MF = ("ha", "ha0", "ha0_ext")
    mf = [r for r in rec["rows"] if r["arm"] in MF and r["delta"] is not None]
    mo = [r for r in rec["rows"] if r["arm"] not in MF and r["delta"] is not None]
    mf_bad = [r for r in mf if r["delta"] != 0.0]
    # counts are integers and move by a window or two; they are not the same
    # kind of claim as a metre, so they are reported separately.
    mo_c = [r for r in mo if not r["is_count"]]
    mo_n = [r for r in mo if r["is_count"]]
    mo_max = max((abs(r["delta"]) for r in mo_c), default=0.0)
    mo_worst = max(mo_c, key=lambda r: abs(r["delta"])) if mo_c else None
    mo_nmax = max((abs(r["delta"]) for r in mo_n), default=0.0)
    print("SUMMARY")
    print("  MODEL-FREE arms (ha/ha0/ha0_ext) — MUST be bit-exact:")
    print("      %d of %d rows EXACT, %d non-zero   %s"
          % (len(mf) - len(mf_bad), len(mf), len(mf_bad),
             "✅ the surface is IDENTICAL" if not mf_bad else
             "⛔ THE SURFACE DIFFERS — panel VOID: " +
             ", ".join("%s.%s=%+.3g" % (r["arm"], r["metric"], r["delta"])
                       for r in mf_bad[:6])))
    print("  MODEL arms (os/os_navshuf/os_navzero) — bounded by cross-hardware argmax ties:")
    print("      %d of %d rows EXACT; largest |delta| %.6g  (%s %s: %s -> %s)"
          % (sum(1 for r in mo if r["delta"] == 0.0), len(mo), mo_max,
             mo_worst["arm"] if mo_worst else "-",
             mo_worst["metric"] if mo_worst else "-",
             mo_worst["published"] if mo_worst else "-",
             mo_worst["reproduced"] if mo_worst else "-"))
    print("      COUNT rows (window tallies) move by at most %d window(s)" % int(mo_nmax))
    n_pair = len(rec["paired"])
    n_agree = sum(1 for c in rec["paired"] if c["separated_agrees"])
    print("  PAIRED MARGINS — the verdict-bearing cells:")
    print("      %d of %d agree on `separated`%s"
          % (n_agree, n_pair,
             "" if n_agree == n_pair else "; DISAGREEING: " +
             ", ".join("%s.%s" % (c["cell"], c["metric"])
                       for c in rec["paired"] if not c["separated_agrees"])))
    print("  (%d rows EXACT, %d within display tolerance, %d beyond it, overall)"
          % (n_exact, n_close, n_off))
    rec["verdict"] = {
        "n_exact": n_exact, "n_close": n_close, "n_off": n_off,
        "model_free_rows": len(mf), "model_free_nonzero": len(mf_bad),
        "model_free_identical": not mf_bad,
        "model_rows": len(mo), "model_max_abs_delta": mo_max,
        "model_max_count_delta": mo_nmax,
        "paired_cells": n_pair, "paired_separated_agree": n_agree,
        "paired_disagreeing": [c["cell"] + "." + c["metric"]
                               for c in rec["paired"] if not c["separated_agrees"]]}
    if out_p:
        with open(out_p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1, default=float, ensure_ascii=False)
        print("wrote %s" % out_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
