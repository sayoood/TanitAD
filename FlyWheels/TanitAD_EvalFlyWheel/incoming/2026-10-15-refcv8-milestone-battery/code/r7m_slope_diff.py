"""r7m_slope_diff.py -- the PRE-REGISTERED contrasts of PREREG_R7_MATCHED_PHASE_TURN.md, with paired CIs, per inference seed AND pooled.

    python r7m_slope_diff.py --label 20000 --r7-tree C:/.../battery/r7step20000_unc --ema-tree C:/.../battery/step20000ema_unc \
        --raw-tree C:/.../battery/step20000_unc --out-json raw/r7_matched_phase/R7M_SLOPE_DIFF_20000.json --out-md ...md

WHY A SEPARATE TOOL.  `heading_drift.py` reports, per arm, the through-origin slope of the turn-aligned heading error on the GT heading and
`slope_minus_ref` = arm - the embedded refcv7-50,400 arm.  The pre-registered primary contrast is `D_ema(N) = s8e(N) - s7(N)`, a difference
between two NON-reference arms, which that tool does not emit with a CI, and the pre-registration needs the two inference seeds read SEPARATELY
(the tool pools them as a 2-seed mean of the per-window heading error).  This file imports `heading_drift` read-only (it edits nothing) and
re-uses its loaders, geometry and bins, and reproduces its slope bootstrap exactly (same resampling: seed 0 + 29, the episode-cluster draw of
`heading_drift.shrinkage`) so that DIFFERENCES between arms are taken on the SAME draws.  A built-in control re-computes the tool's own
`shrinkage()` on the same arms and requires the point slopes and `slope_minus_ref` CIs to agree to 1e-9.

QUANTITIES (PREREG section 3; S2 grid, group `sharp (>= 0.15)`, `last segment` = the banked -0.092 / -0.141 cell; the other groups and S6 are
reported too).  Negative = more under-rotation.
    s7 = slope(R7-N)    s7f = slope(refcv7-50,400, the arm embedded in every panel)    s8e = slope(EMA8-N)    s8r = slope(RAW8-N)
    P7  = s7  - s7f                    (phase term)
    D_ema = s8e - s7    D_raw = s8r - s7
    phase fraction = P7 / (s8e - s7f)  (reported with its CI only when the CI of the denominator excludes 0)
ESTIMATOR: episode-cluster bootstrap over the 139 eval episodes, B 2000, models FIXED, 95 % percentile.  It answers "another draw of EPISODES";
NOT another training run (H-ESTIM-SEED-1: one checkpoint per arm per step) and NOT another inference run (the two seeds are separate columns).
MECHANICAL READ of the pre-registered outcome table (does not replace the Master Mind's reading): per seed, then "agree" iff both seeds give the
same row.  PHASE: P7 separated negative AND D_ema not separated negative.  DESIGN: D_ema separated negative AND P7 not separated negative.  MIXED: else.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "raw" / "heading_drift"))
import heading_drift as HD  # noqa: E402  (read-only import)

GROUPS = ("curved (GT |yaw| >= 0.02)", "sharp (>= 0.15)", "curved & v0 5-10", "sharp & v0 5-10")
HEADLINE = ("S2", "sharp (>= 0.15)", "last segment")


def group_masks(B: dict) -> dict:
    curved_m = (B["turn"] >= 1) & (B["turn"] <= 3)
    return {"curved (GT |yaw| >= 0.02)": curved_m, "sharp (>= 0.15)": B["turn"] == 3,
            "curved & v0 5-10": curved_m & (B["speed"] == 2), "sharp & v0 5-10": (B["turn"] == 3) & (B["speed"] == 2)}


def boot_slopes(ff, torch, dh_by_arm: dict, G, dt: float, eid, B: dict, gmask, which: str, n_boot: int):
    """== heading_drift.shrinkage for ONE (group, segment-set), but returns the bootstrap DRAWS so arm differences share a resample."""
    Gg = ff._seq_geometry(torch.as_tensor(G).float(), dt)
    hg = Gg["heading"].numpy().astype(np.float64) * 180.0 / math.pi
    sw = np.where((B["turn"] >= 1) & (B["turn"] <= 3), np.sign(B["gt_yaw_signed"]), 0.0)
    labs = list(dh_by_arm)
    valid = np.all([np.isfinite(dh_by_arm[l]) for l in labs], axis=0) & Gg["valid"].numpy()
    K = valid.shape[1]
    ue, inv = np.unique(np.asarray([str(e) for e in eid]), return_inverse=True)
    E = len(ue)
    rng = np.random.default_rng(HD.SEED + 29)
    pick = rng.integers(0, E, size=(n_boot, E))
    W = np.stack([np.bincount(pick[b], minlength=E) for b in range(n_boot)]).astype(float)
    ks = [K - 1] if which == "last segment" else list(range(1, K))
    m = np.zeros_like(valid)
    m[:, ks] = True
    m &= valid & gmask[:, None] & (sw != 0)[:, None]
    if m.sum() < 30:
        return None
    h = hg * sw[:, None]
    ii = np.broadcast_to(inv[:, None], m.shape)[m]
    Sxx = np.bincount(ii, weights=(h ** 2)[m], minlength=E)
    out = {}
    for l in labs:
        e = dh_by_arm[l] * sw[:, None]
        Sxy = np.bincount(ii, weights=(e * h)[m], minlength=E)
        out[l] = (float(Sxy.sum() / Sxx.sum()), (W @ Sxy) / (W @ Sxx))
    return out, int(m.sum()), int(m.any(1).sum())


def ci_of(draws, point):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}


def contrasts(sl: dict, n_seg: int, n_win: int, x_lab: str) -> dict:
    """sl = {arm: (point, draws)} with arms R7, REF, x_lab."""
    s7, d7 = sl["R7"]; sf, df = sl["REF"]; sx, dx = sl[x_lab]
    r = {"n_win": n_win, "n_seg": n_seg,
         "slope": {k: ci_of(sl[k][1], sl[k][0]) | {"slope": sl[k][0]} for k in sl},
         "P7 = s7 - s7f": ci_of(d7 - df, s7 - sf),
         f"D = s({x_lab}) - s7": ci_of(dx - d7, sx - s7),
         f"gap = s({x_lab}) - s7f": ci_of(dx - df, sx - sf)}
    gap_d = dx - df
    lo, hi = np.percentile(gap_d, [2.5, 97.5])
    if lo < 0 < hi:
        r["phase_fraction"] = {"point": float((s7 - sf) / (sx - sf)) if sx != sf else None,
                               "note": "CI undefined: the CI of the denominator (gap) includes 0"}
    else:
        fr = (d7 - df) / gap_d
        flo, fhi = np.percentile(fr, [2.5, 97.5])
        r["phase_fraction"] = {"point": float((s7 - sf) / (sx - sf)), "lo": float(flo), "hi": float(fhi)}
    return r


def outcome(p7: dict, d: dict) -> str:
    neg_p = p7["separated"] and p7["point"] < 0
    neg_d = d["separated"] and d["point"] < 0
    if neg_p and not neg_d:
        return "PHASE"
    if neg_d and not neg_p:
        return "DESIGN"
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="the milestone step, e.g. 20000 (a label only)")
    ap.add_argument("--r7-tree", required=True)
    ap.add_argument("--ema-tree", required=True)
    ap.add_argument("--raw-tree", default=None)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--surfaces", default="S2,S6")
    ap.add_argument("--n-boot", type=int, default=HD.N_BOOT)
    ap.add_argument("--ref-prefix", default="b_refcv7_50400")
    ap.add_argument("--taniteval-root", default="E:/Projects/TanitAD/taniteval")
    ap.add_argument("--stack-root", default="E:/Projects/TanitAD/stack")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch, ci, ff = HD.setup(a.taniteval_root, a.stack_root)
    seeds = [int(x) for x in a.seeds.split(",")]
    xs = [("EMA8", a.ema_tree)] + ([("RAW8", a.raw_tree)] if a.raw_tree else [])
    res = {"tool": "r7m_slope_diff.py", "generated": time.strftime("%FT%T"), "label": a.label, "trees": {"R7": a.r7_tree, **dict(xs)},
           "n_boot": a.n_boot, "estimator": "episode-cluster bootstrap over the eval episodes, models FIXED (not another training run; not another inference run)",
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)", "surfaces": {}, "control": {}, "baseline_identity": {}}
    for surf in a.surfaces.split(","):
        per = {}
        for s in seeds:
            refl = f"{a.ref_prefix}_s{s}"
            P0 = HD.load_panel(a.r7_tree, surf, s, refl)
            dt = P0["dt"]
            dh = {"R7": HD.seg_signed_heading(ff, torch, P0["os"], P0["g"], dt), "REF": HD.seg_signed_heading(ff, torch, P0["ref"], P0["g"], dt)}
            for lab, root in xs:
                P1 = HD.load_panel(root, surf, s, refl)
                idx = HD.align(P0, P1)                                  # raises on a window-set or GT mismatch
                res["baseline_identity"][f"{surf}|{lab}|s{s}"] = float(np.abs(P0["ref"].astype(np.float64) - P1["ref"][idx].astype(np.float64)).max())
                dh[lab] = HD.seg_signed_heading(ff, torch, P1["os"][idx], P0["g"], dt)
            per[s] = {"dh": dh, "P0": P0, "dt": dt, "B": HD.gt_bins(ff, torch, P0["g"], P0["v0"], P0["nav_cmd"], dt)}
        P0, dt, B = per[seeds[0]]["P0"], per[seeds[0]]["dt"], per[seeds[0]]["B"]
        eid, G = P0["eid"], P0["g"]
        masks = group_masks(B)
        views = {f"seed {s}": per[s]["dh"] for s in seeds}
        if len(seeds) == 2:
            views["pooled (2-seed mean heading error)"] = {k: 0.5 * (per[seeds[0]]["dh"][k] + per[seeds[1]]["dh"][k]) for k in per[seeds[0]]["dh"]}
        sr = {"meta": {"n_windows": P0["n_win"], "n_episodes": P0["n_ep"], "instants_s": P0["instants_s"]}, "views": {}}
        for vname, dhv in views.items():
            sr["views"][vname] = {}
            for x_lab, _ in xs:
                arms = {k: dhv[k] for k in ("R7", "REF", x_lab)}
                for gname in GROUPS:
                    for which in ("last segment", "segments 2..K"):
                        got = boot_slopes(ff, torch, arms, G, dt, eid, B, masks[gname], which, a.n_boot)
                        if got is None:
                            continue
                        sl, n_seg, n_win = got
                        sr["views"][vname].setdefault(x_lab, {})[f"{gname} | {which}"] = contrasts(sl, n_seg, n_win, x_lab)
        res["surfaces"][surf] = sr
        # ---- control: the tool's own shrinkage() on the same arms (pooled view) must agree to 1e-9 --------------------------------
        pv = views.get("pooled (2-seed mean heading error)") or views[f"seed {seeds[0]}"]
        for x_lab, _ in xs:
            arms = {k: pv[k] for k in ("R7", "REF", x_lab)}
            tool = HD.shrinkage(ff, torch, arms, G, dt, eid, B, {g: masks[g] for g in GROUPS}, a.n_boot, "REF")
            worst = 0.0
            vname = "pooled (2-seed mean heading error)" if "pooled (2-seed mean heading error)" in sr["views"] else f"seed {seeds[0]}"
            for key, row in tool.items():
                mine = sr["views"][vname][x_lab][key]
                for lab in ("R7", "REF", x_lab):
                    worst = max(worst, abs(row["slope"][lab]["slope"] - mine["slope"][lab]["slope"]), abs(row["slope"][lab]["lo"] - mine["slope"][lab]["lo"]),
                                abs(row["slope"][lab]["hi"] - mine["slope"][lab]["hi"]))
                for lab in ("R7", x_lab):
                    t_ = row["slope_minus_ref"][lab]
                    m_ = mine["P7 = s7 - s7f"] if lab == "R7" else mine["gap = s(%s) - s7f" % x_lab]
                    worst = max(worst, abs(t_["delta"] - m_["point"]), abs(t_["lo"] - m_["lo"]), abs(t_["hi"] - m_["hi"]))
            res["control"][f"{surf}|{x_lab}"] = {"max_abs_difference_vs_heading_drift_shrinkage": worst, "reproduced": bool(worst < 1e-9), "n_cells": len(tool)}
        print(f"[{time.time() - t0:6.1f}s] {surf} done", flush=True)
    # ---- the mechanical read of the pre-registered table ---------------------------------------------------------------------------
    surf, g, which = HEADLINE
    mech = {}
    if surf in res["surfaces"]:
        for x_lab, _ in xs:
            rows = {}
            for vname, v in res["surfaces"][surf]["views"].items():
                c = v.get(x_lab, {}).get(f"{g} | {which}")
                if c:
                    rows[vname] = {"P7": c["P7 = s7 - s7f"], "D": c[f"D = s({x_lab}) - s7"], "outcome": outcome(c["P7 = s7 - s7f"], c[f"D = s({x_lab}) - s7"])}
            per_seed = [rows[f"seed {s}"]["outcome"] for s in seeds if f"seed {s}" in rows]
            mech[x_lab] = {"rows": rows, "per_seed_outcomes": per_seed,
                           "both_seeds_agree": len(set(per_seed)) == 1 and len(per_seed) == len(seeds),
                           "mechanical_outcome": (per_seed[0] if (len(set(per_seed)) == 1 and len(per_seed) == len(seeds)) else "MIXED (seeds disagree)")}
    res["mechanical_read"] = {"cell": "/".join(HEADLINE), "note": "mechanical application of the pre-registered table; the primary contrast is EMA8 (x_lab EMA8), RAW8 is secondary and cannot override",
                              "by_contrast": mech}
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    L = [f"# R7 matched-phase contrasts at N = {a.label} (S2/S6 heading-error slope; PREREG_R7_MATCHED_PHASE_TURN.md)\n",
         f"* generated {res['generated']}; trees: " + "; ".join(f"`{k}` = {v}" for k, v in res["trees"].items()),
         f"* estimator: {res['estimator']}; B {a.n_boot}; tier: {res['tier']}",
         "* CONTROL vs `heading_drift.shrinkage` on the same arms (must be < 1e-9): " + "; ".join(f"{k}: {v['max_abs_difference_vs_heading_drift_shrinkage']:.2e} ({'ok' if v['reproduced'] else 'NOT REPRODUCED'})" for k, v in res["control"].items()),
         "* baseline identity (embedded refcv7-50,400 arm, max abs diff vs the R7 tree; must be 0): " + ", ".join(f"{k} {v:.1e}" for k, v in res["baseline_identity"].items())]
    for surf_, sr in res["surfaces"].items():
        L.append(f"\n## {surf_} ({sr['meta']['instants_s']} s), {sr['meta']['n_windows']} windows / {sr['meta']['n_episodes']} episodes\n")
        for x_lab, _ in xs:
            for gname in GROUPS:
                for which in ("last segment", "segments 2..K"):
                    key = f"{gname} | {which}"
                    L.append(f"\n### {surf_} -- {x_lab} -- {key}\n")
                    L.append("| view | n win / seg | s7f (refcv7-50,400) | s7 (R7-N) | s(%s) | P7 = s7-s7f [CI] | D = s(%s)-s7 [CI] | gap = s(%s)-s7f [CI] | phase fraction |" % (x_lab, x_lab, x_lab))
                    L.append("|---|---|---|---|---|---|---|---|---|")
                    for vname, v in sr["views"].items():
                        c = v.get(x_lab, {}).get(key)
                        if not c:
                            continue
                        f = lambda r: f"{r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' SEP' if r['separated'] else ''}"  # noqa: E731
                        pf = c["phase_fraction"]
                        pft = (f"{pf['point']:.2f} [{pf['lo']:.2f}, {pf['hi']:.2f}]" if "lo" in pf else f"{pf['point']:.2f} ({pf['note']})") if pf.get("point") is not None else "n/a"
                        L.append(f"| {vname} | {c['n_win']} / {c['n_seg']} | {c['slope']['REF']['slope']:+.4f} | {c['slope']['R7']['slope']:+.4f} | {c['slope'][x_lab]['slope']:+.4f} | "
                                 f"{f(c['P7 = s7 - s7f'])} | {f(c['D = s(%s) - s7' % x_lab])} | {f(c['gap = s(%s) - s7f' % x_lab])} | {pft} |")
    L.append("\n## MECHANICAL READ of the pre-registered outcome table (cell " + res["mechanical_read"]["cell"] + ")\n")
    for x_lab, m in mech.items():
        L.append(f"* contrast **{x_lab}**: per-seed outcomes {m['per_seed_outcomes']}; both seeds agree = {m['both_seeds_agree']}; **{m['mechanical_outcome']}**")
    open(a.out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("WROTE", a.out_json, a.out_md, f"({time.time() - t0:.0f}s)")
    print("CONTROL", json.dumps({k: v["reproduced"] for k, v in res["control"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
