"""P2 - DID THE THIRD SEGMENT FIX THE TAIL IT WAS BUILT FOR?

The two-segment work REFUTED its own mechanism hypothesis: the flip is the
SMALLEST curvature contributor (+0.001334) and 72.04 % of the degradation sits
in the 3-6 s TAIL. The three-segment candidate returns to straight at t2, so the
PRE-STATED prediction is that the 4-6 s band comes back toward the base while
the 2-4 s band (where it still counter-steers) does not.

⛔ CONTROL AT A KNOWN VALUE: on lane-change windows whose pick did NOT change,
the two paths are the SAME OBJECT and every family must read EXACTLY equal.
Asserted, not assumed.

Also re-derives the predecessor's exploratory (t1, t2) sweep FROM SCRATCH, as
CONTEXT. ⛔ It may not move the shipped default: RULE S fixed that before any
number existed.
"""
import argparse
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401
import p1_twoseg_supply as P                    # noqa: E402
import p2_split_sweep as S                      # noqa: E402
from tanitad.refs import anchor_twoseg as ts    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

T_S = P.T_S
#: slot boundaries in seconds -> the per-segment bands of RESULT.md §9
BANDS = [(0, 1, "0.5-1.0 s"), (1, 2, "1.0-1.5 s"), (2, 3, "1.5-2.0 s"),
         (3, 4, "2.0-3.0 s"), (4, 5, "3.0-4.0 s"), (5, 6, "4.0-5.0 s"),
         (6, 7, "5.0-6.0 s")]


def seg_curv(p, g, i, j):
    """|curvature error| on the single slot-pair (i, j), mean over windows."""
    q = P.path_geom(p, P.SLOT_DT)
    r = P.path_geom(g, P.SLOT_DT)
    ok = q["kappa_ok"][:, i:j] & r["kappa_ok"][:, i:j]
    d = np.abs(q["kappa"][:, i:j] - r["kappa"][:, i:j])
    nk = ok.sum(-1)
    v = np.where(nk > 0, (d * ok).sum(-1) / np.maximum(nk, 1), np.nan)
    return float(np.nanmean(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(P.CKPT, map_location="cpu", weights_only=False)
    C0 = sd.get("model", sd)["core.decoder.anchor_controls"].float()
    n_base = int(C0.shape[0])
    A1 = ts.extend_controls(C0, T_S)
    A2 = ts.extend_controls_three(C0, T_S)

    GT, V0, TAG, _ = S.P_load(a)
    n = len(V0)
    LC = TAG[:, 0] == 1
    print(f"[grid] n={n}  lane-change={int(LC.sum())}")

    def pick(c):
        f = P.roll(c, V0, P.SLOTS)
        ade, idx = P.best(f, GT)
        p = f[np.arange(n), idx].copy()
        del f
        return ade, idx, p

    a0, i0, p0 = pick(C0)
    a1, i1, p1 = pick(A1)
    a2, i2, p2 = pick(A2)

    res = {"n": int(n), "n_lane_change": int(LC.sum()), "strata": {},
           "per_segment_curvature": {}, "exploratory_sweep_context": {}}

    # ---------------------------------------------------- the strata -------
    print("\n########## STRATA -- and the control at a KNOWN value ##########")
    print(f"  {'stratum':<34}{'n':>6}{'ADE b':>9}{'ADE e':>9}"
          f"{'curv b':>10}{'curv e':>10}{'cross b':>9}{'cross e':>9}"
          f"{'head b':>8}{'head e':>8}")
    for nm, idxs, pk, mask0 in (("A1 twoseg", i1, p1, LC),
                                ("A2 triseg", i2, p2, LC)):
        for tag, m in ((f"lane-change, REPICKED   [{nm}]",
                        mask0 & (idxs >= n_base)),
                       (f"lane-change, unchanged  [{nm}]",
                        mask0 & (idxs < n_base))):
            fb = P.families(p0[m], GT[m], P.SLOT_DT)
            fe = P.families(pk[m], GT[m], P.SLOT_DT)
            row = {"n": int(m.sum())}
            for k in ("ade_m", "curv_mae_1pm", "cross_mae_m",
                      "heading_mae_deg"):
                row[f"{k}_base"] = float(np.nanmean(fb[k]))
                row[f"{k}_ext"] = float(np.nanmean(fe[k]))
            res["strata"][tag] = row
            print(f"  {tag:<34}{row['n']:>6d}{row['ade_m_base']:>9.4f}"
                  f"{row['ade_m_ext']:>9.4f}{row['curv_mae_1pm_base']:>10.6f}"
                  f"{row['curv_mae_1pm_ext']:>10.6f}"
                  f"{row['cross_mae_m_base']:>9.4f}"
                  f"{row['cross_mae_m_ext']:>9.4f}"
                  f"{row['heading_mae_deg_base']:>8.3f}"
                  f"{row['heading_mae_deg_ext']:>8.3f}")
            if "unchanged" in tag:
                same = bool(np.array_equal(p0[m], pk[m]))
                print(f"       CONTROL: pick unchanged => same object => every "
                      f"family EXACTLY equal: {same}")
                res["strata"][tag]["control_paths_exactly_equal"] = same

    # ------------------------------------------- per-segment curvature -----
    print("\n########## PER-SEGMENT CURVATURE -- did the TAIL come back? ####")
    m1 = LC & (i1 >= n_base)
    m2 = LC & (i2 >= n_base)
    print(f"  {'segment':<14}{'base|A1':>10}{'A1':>10}{'dA1':>11}"
          f"{'base|A2':>10}{'A2':>10}{'dA2':>11}")
    tot1 = tot2 = 0.0
    tail1 = tail2 = 0.0
    for i, j, nm in BANDS:
        b1 = seg_curv(p0[m1], GT[m1], i, j)
        e1 = seg_curv(p1[m1], GT[m1], i, j)
        b2 = seg_curv(p0[m2], GT[m2], i, j)
        e2 = seg_curv(p2[m2], GT[m2], i, j)
        d1, d2 = e1 - b1, e2 - b2
        tot1 += d1
        tot2 += d2
        if i >= 4:                       # the 3-6 s tail
            tail1 += d1
            tail2 += d2
        res["per_segment_curvature"][nm] = {
            "A1_base": b1, "A1_ext": e1, "A1_delta": d1,
            "A2_base": b2, "A2_ext": e2, "A2_delta": d2}
        print(f"  {nm:<14}{b1:>10.6f}{e1:>10.6f}{d1:>+11.6f}"
              f"{b2:>10.6f}{e2:>10.6f}{d2:>+11.6f}")
    print(f"  {'TOTAL':<14}{'':>10}{'':>10}{tot1:>+11.6f}"
          f"{'':>10}{'':>10}{tot2:>+11.6f}")
    print(f"  {'3-6 s TAIL':<14}{'':>10}{'':>10}{tail1:>+11.6f}"
          f"{'':>10}{'':>10}{tail2:>+11.6f}")
    print(f"  tail share of total degradation:  A1 "
          f"{100.0*tail1/tot1 if tot1 else float('nan'):.2f} %   A2 "
          f"{100.0*tail2/tot2 if tot2 else float('nan'):.2f} %")
    res["per_segment_curvature"]["_totals"] = {
        "A1_total_delta": tot1, "A2_total_delta": tot2,
        "A1_tail_3to6s_delta": tail1, "A2_tail_3to6s_delta": tail2,
        "A1_tail_share": tail1 / tot1 if tot1 else None,
        "A2_tail_share": tail2 / tot2 if tot2 else None,
        "n_A1_repicks": int(m1.sum()), "n_A2_repicks": int(m2.sum())}

    # ------------------------------------------------- context sweep -------
    print("\n########## CONTEXT: the exploratory (t1, t2) sweep, re-derived ##")
    print("  ⛔ CONTEXT ONLY. RULE S fixed the schedule before any number "
          "existed; nothing here may move the shipped default.")
    combos = [("2-seg t=2.0 (equal-cost ref)", 2.0, 6.0),
              ("3-seg 2.0 -> 3.0", 2.0, 3.0), ("3-seg 2.0 -> 3.5", 2.0, 3.5),
              ("3-seg 2.0 -> 4.0  <- RULE S", 2.0, 4.0),
              ("3-seg 2.0 -> 4.5", 2.0, 4.5), ("3-seg 1.5 -> 3.0", 1.5, 3.0),
              ("3-seg 1.5 -> 3.5", 1.5, 3.5), ("3-seg 2.5 -> 4.0", 2.5, 4.0),
              ("3-seg 2.5 -> 5.0", 2.5, 5.0), ("3-seg 3.0 -> 5.0", 3.0, 5.0)]
    print(f"\n  {'family':<30}{'LCgain':>9}{'ALLgain':>9}{'TURNgain':>10}"
          f"{'LCpicks':>9}{'d curv':>10}{'d cross':>9}{'d head':>9}"
          f"{'2s repick':>11}")
    s2 = [P.SLOTS[i] for i in range(4)]
    f2b = P.roll(C0, V0, s2)
    a2b, _ = P.best(f2b, GT[:, :4])
    del f2b
    for nm, t1, t2 in combos:
        ext = ts.extend_controls_three(C0, T_S, schedules=[(t1, t2)])
        ae, ie, pe = pick(ext)
        m = LC & (ie >= n_base)
        fe2 = P.roll(ext, V0, s2)
        ae2, ie2 = P.best(fe2, GT[:, :4])
        del fe2
        r = {"t1": t1, "t2": t2, "n_new": 2,
             "gain_lane_change": float(a0[LC].mean() - ae[LC].mean()),
             "gain_all": float(a0.mean() - ae.mean()),
             "gain_turn": float(a0[TAG[:, 1] == 1].mean()
                                - ae[TAG[:, 1] == 1].mean()),
             "lc_picks": int(m.sum()),
             "n_2s_repicks": int((ie2 >= n_base).sum()),
             "ade_2s_change": float(np.abs(a2b - ae2).max())}
        if m.any():
            fb = P.families(p0[m], GT[m], P.SLOT_DT)
            ff = P.families(pe[m], GT[m], P.SLOT_DT)
            for k, lbl in (("curv_mae_1pm", "d_curv"),
                           ("cross_mae_m", "d_cross"),
                           ("heading_mae_deg", "d_head")):
                r[lbl] = float(np.nanmean(ff[k]) - np.nanmean(fb[k]))
        res["exploratory_sweep_context"][nm] = r
        print(f"  {nm:<30}{r['gain_lane_change']:>9.4f}{r['gain_all']:>9.4f}"
              f"{r['gain_turn']:>10.4f}{r['lc_picks']:>9d}"
              f"{r.get('d_curv', float('nan')):>10.6f}"
              f"{r.get('d_cross', float('nan')):>9.4f}"
              f"{r.get('d_head', float('nan')):>9.3f}"
              f"{r['n_2s_repicks']:>11d}")

    # is RULE S's row the argmax of anything?
    rows = {k: v for k, v in res["exploratory_sweep_context"].items()
            if k.startswith("3-seg")}
    argmax = {"gain_lane_change": max(rows, key=lambda k: rows[k]["gain_lane_change"]),
              "gain_all": max(rows, key=lambda k: rows[k]["gain_all"]),
              "d_curv_lowest": min(rows, key=lambda k: rows[k].get("d_curv", 9e9)),
              "d_cross_lowest": min(rows, key=lambda k: rows[k].get("d_cross", 9e9)),
              "d_head_lowest": min(rows, key=lambda k: rows[k].get("d_head", 9e9))}
    print("\n  ⭐ IS RULE S's ROW THE ARGMAX OF ANYTHING? (the post-hoc check)")
    for k, v in argmax.items():
        flag = "  <-- RULE S" if "RULE S" in v else ""
        print(f"     best on {k:<20} = {v}{flag}")
    res["ruleS_is_argmax_of"] = argmax

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
