"""P1 - THE THREE-SEGMENT MEASUREMENT, against the PREREG D-TRISEG bars.

⛔ Model-free and DETERMINISTIC. The fan is reconstructed from the checkpoint's
own `anchor_controls` through the programme's own integrator, and the target is
the RECORDED 6 s ego path. No training draw, no inference sampling, no episode
resampling, and EVERY scoreable window is scored -- so there is no population
being estimated and NO CI IS QUOTED.

⛔ THIS IS A SUPPLY CEILING. It says what the best possible selector could reach
with the extended fan, and may never be quoted beside an achievement
(`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`). refcv4b has 117 logits and
STRUCTURALLY CANNOT SELECT THESE CANDIDATES: what changes is the SUPERVISION.

Arms, all against the SAME windows:
  A0    117            the incumbent
  A1    123 (3 col)    the SHIPPED two-segment family      <- B0's reference
  A1p   119 (3 col)    the equal-cost two-segment pair t_split = 2.0
  A2    119 (4 col)    the RULE S three-segment family     <- THE TEST ARM
  A3    125 (4 col)    both families composed (reported, not the bar)
  R1/R2/R3/R4          the deliberate-regression arms
  floors               straight-family, constant-velocity, zero-path
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
EPS0 = 1e-12


def score(ctrl, V0, GT, slots=None, gt=None):
    """(ade [n], idx [n], picked [n, S, 2]) -- the fan is freed immediately."""
    sl = P.SLOTS if slots is None else slots
    g = GT if gt is None else gt
    fan = P.roll(ctrl, V0, sl)
    ade, idx = P.best(fan, g)
    pick = fan[np.arange(len(idx)), idx].copy()
    del fan
    return ade, idx, pick


def fam(pick, gt, mask):
    f = P.families(pick[mask], gt[mask], P.SLOT_DT)
    r = {k: float(np.nanmean(v)) for k, v in f.items() if k != "n_kappa"}
    r["n"] = int(mask.sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(P.CKPT, map_location="cpu", weights_only=False)
    C0 = sd.get("model", sd)["core.decoder.anchor_controls"].float()
    n_base = int(C0.shape[0])
    print(f"[bank] source controls {tuple(C0.shape)} sha256 {P.sha(C0)}")

    sch = ts.rule_s_schedules(horizon_s=T_S)
    print(f"[RULE S] t1 grid {list(ts.DEFAULT_T_SPLIT_S)} s, horizon {T_S} s "
          f"-> schedules {sch}  (t2 = 2*t1; empty-third-segment dropped)")
    if sch != [(2.0, 4.0)]:
        raise SystemExit(f"RULE S did not reproduce the pre-registered "
                         f"schedule [(2.0, 4.0)]; got {sch}")

    A1 = ts.extend_controls(C0, T_S)                            # 123 x 3
    A1p = ts.extend_controls(C0, T_S, t_split_s=(2.0,))         # 119 x 3
    A2 = ts.extend_controls_three(C0, T_S)                      # 119 x 4
    A3 = ts.extend_controls_three(A1, T_S)                      # 125 x 4
    R1 = ts.extend_controls_three(C0, T_S, schedules=[(6.0, 6.0)])
    R2 = ts.extend_controls_three(C0, T_S, schedules=[(0.0, 6.0)])
    R3 = ts.extend_controls_three(C0, T_S, schedules=[(2.0, 6.0)])
    R4 = ts.extend_controls_three(C0, T_S, schedules=[(0.0, 0.0)])
    STRAIGHT = C0[C0[:, 1] == 0.0]
    CV = torch.tensor([[0.0, 0.0]])
    print(f"[ext] {ts.describe_family(A2, n_base)}")

    GT, V0, TAG, n_dump = S.P_load(a)
    n = len(V0)
    LC, TN = TAG[:, 0] == 1, TAG[:, 1] == 1
    ALL = np.ones(n, bool)
    print(f"[grid] 141 B1-v7.2 EVAL clips, refcv4b stride-1 dump: {n_dump} "
          f"dumped -> {n} with a full {T_S} s recorded future")
    print(f"[grid] lane-change {int(LC.sum())} ({100.0*LC.mean():.2f} %)  "
          f"turn {int(TN.sum())}  other {int((~LC & ~TN).sum())}")
    print(f"[grid] v0 range {V0.min():.2f} - {V0.max():.2f} m/s")

    res = {"corpus": "141 B1-v7.2 EVAL clips (refcv4b T1 stride-1 dump)",
           "n_dumped_windows": int(n_dump), "n_scoreable_windows": int(n),
           "n_lane_change": int(LC.sum()), "n_turn": int(TN.sum()),
           "source_controls_sha256": P.sha(C0),
           "rule_s_schedules": [list(x) for x in sch],
           "estimator": "model-free, deterministic, every scoreable window "
                        "scored; no population is estimated so NO CI is quoted",
           "arms": {}}

    banks = {"A0_base117": C0, "A1_shipped123_twoseg": A1,
             "A1p_twoseg_pair119": A1p, "A2_ruleS_triseg119": A2,
             "A3_composed125": A3,
             "R1_t1=t2=horizon": R1, "R2_t1=0_t2=horizon": R2,
             "R3_t1=2_t2=horizon": R3, "R4_t1=t2=0": R4,
             "F_straight_family": STRAIGHT, "F_const_velocity": CV}
    ade, idx, pick = {}, {}, {}
    for k, c in banks.items():
        ade[k], idx[k], pick[k] = score(c, V0, GT)
        print(f"[roll] {k:24s} N={c.shape[0]:>3d} x{c.shape[1]}  "
              f"ALL {ade[k].mean():.4f}  LC {ade[k][LC].mean():.4f}  "
              f"TURN {ade[k][TN].mean():.4f}")
    z = np.linalg.norm(GT, axis=-1).mean(-1)
    print(f"[roll] {'F_zero_path':24s} N=  0     ALL {z.mean():.4f}  "
          f"LC {z[LC].mean():.4f}  TURN {z[TN].mean():.4f}")

    b = ade["A0_base117"]
    for k in banks:
        res["arms"][k] = {
            "n_candidates": int(banks[k].shape[0]),
            "n_control_columns": int(banks[k].shape[1]),
            "ade_all": float(ade[k].mean()),
            "ade_lane_change": float(ade[k][LC].mean()),
            "ade_turn": float(ade[k][TN].mean()),
            "gain_lane_change": float(b[LC].mean() - ade[k][LC].mean()),
            "gain_all": float(b.mean() - ade[k].mean()),
            "gain_turn": float(b[TN].mean() - ade[k][TN].mean()),
            "n_windows_improved": int((ade[k] < b - EPS0).sum()),
            "n_lc_windows_improved": int((ade[k][LC] < b[LC] - EPS0).sum())}
    res["arms"]["F_zero_path"] = {"n_candidates": 0,
                                  "ade_all": float(z.mean()),
                                  "ade_lane_change": float(z[LC].mean()),
                                  "ade_turn": float(z[TN].mean())}

    # ================================================================== B4 ===
    print("\n########## B4  DELIBERATE REGRESSION -- the gate must GRADE #####")
    b4 = {}
    for k, why in (("R1_t1=t2=horizon", "+1 at every tick = existing constant arcs"),
                   ("R2_t1=0_t2=horizon", "-1 at every tick = mirror arcs (grid is mirror-complete)"),
                   ("R4_t1=t2=0", "0 at every tick = candidate 67, the straight line")):
        g_lc = float(b[LC].mean() - ade[k][LC].mean())
        g_all = float(b.mean() - ade[k].mean())
        nimp = int((ade[k] < b - EPS0).sum())
        ok = (g_lc == 0.0) and (g_all == 0.0) and nimp == 0
        print(f"  {k:22s} LC gain {g_lc:+.10f}  ALL gain {g_all:+.10f}  "
              f"improved {nimp}/{n}  -> {'EXACT ZERO' if ok else 'NON-ZERO'}"
              f"   [{why}]")
        b4[k] = {"gain_lane_change": g_lc, "gain_all": g_all,
                 "n_improved": nimp, "structural_zero": bool(ok), "why": why}
    # R3 is the GRADING arm: identical geometry to the two-segment pair
    same_ctl = torch.equal(ts.as_four_column(A1p, T_S), R3)
    r3_lc = float(b[LC].mean() - ade["R3_t1=2_t2=horizon"][LC].mean())
    a1p_lc = float(b[LC].mean() - ade["A1p_twoseg_pair119"][LC].mean())
    r3_exact = bool(np.array_equal(ade["R3_t1=2_t2=horizon"],
                                   ade["A1p_twoseg_pair119"]))
    print(f"  R3 (grading)           LC gain {r3_lc:.10f} vs the two-segment "
          f"pair {a1p_lc:.10f}")
    print(f"  R3 controls == as_four_column(A1p): {same_ctl};  per-window ADE "
          f"array EXACTLY equal: {r3_exact}")
    b4["R3_grades"] = {"gain_lane_change": r3_lc,
                       "twoseg_pair_gain_lane_change": a1p_lc,
                       "controls_widening_identical": bool(same_ctl),
                       "per_window_ade_exactly_equal": r3_exact}
    B4 = all(v["structural_zero"] for v in b4.values()
             if "structural_zero" in v) and r3_exact
    print(f"  ==> B4 {'PASS' if B4 else 'FAIL'}")
    res["B4"] = {"pass": bool(B4), "arms": b4}

    # ================================================================== B2 ===
    print("\n########## B2  THE LIMIT IS BIT-IDENTICAL, BY COMPARISON ########")
    f2 = P.roll(C0, V0, P.SLOTS)
    f4 = P.roll(ts.as_four_column(C0, T_S), V0, P.SLOTS)
    L1 = bool(np.array_equal(f2, f4))
    print(f"  L1  117x2 vs 117x4 over {f2.size:,} floats: "
          f"{'BIT-IDENTICAL' if L1 else 'DIFFERS'}")
    MUT = ts.as_four_column(C0, T_S).clone()
    MUT[:, 2], MUT[:, 3] = 2.0, 4.0
    fm = P.roll(MUT, V0, P.SLOTS)
    l3 = float(np.abs(fm - f2).max())
    del f2, f4, fm
    g3 = P.roll(A1, V0, P.SLOTS)
    g4 = P.roll(ts.as_four_column(A1, T_S), V0, P.SLOTS)
    L2 = bool(np.array_equal(g3, g4))
    print(f"  L2  123x3 vs 123x4 over {g3.size:,} floats: "
          f"{'BIT-IDENTICAL' if L2 else 'DIFFERS'}")
    del g3, g4
    print(f"  L3  MUTATION CONTROL (t1=2, t2=4 on the SAME controls): "
          f"max |diff| = {l3:.4f} m  (must be > 1.0)")
    B2 = L1 and L2 and l3 > 1.0
    print(f"  ==> B2 {'PASS' if B2 else 'FAIL'}"
          f"{'' if l3 > 1.0 else '  <-- the equality cannot fail; it proves nothing'}")
    res["B2"] = {"pass": bool(B2), "L1_2col_vs_4col_bit_identical": L1,
                 "L2_3col_vs_4col_bit_identical": L2,
                 "L3_mutation_max_abs_diff_m": l3,
                 "n_floats_compared": int(n * n_base * len(P.SLOTS) * 2)}

    # ================================================================== B5 ===
    print("\n########## B5  STRUCTURAL ZERO ON THE 2 s GRID #################")
    s2 = [P.SLOTS[i] for i in range(4)]
    a2b, i2b, p2b = score(C0, V0, GT, slots=s2, gt=GT[:, :4])
    b5 = {}
    for k in ("A2_ruleS_triseg119", "A1_shipped123_twoseg", "A3_composed125"):
        a2e, i2e, p2e = score(banks[k], V0, GT, slots=s2, gt=GT[:, :4])
        d = float(np.abs(a2b - a2e).max())
        rp = int((i2e >= n_base).sum())
        fb_ = P.families(p2b, GT[:, :4], P.SLOT_DT[:4])
        fe_ = P.families(p2e, GT[:, :4], P.SLOT_DT[:4])
        dfam = {kk: float(np.abs(np.nan_to_num(fe_[kk]) -
                                 np.nan_to_num(fb_[kk])).max())
                for kk in ("ade_m", "along_mae_m", "cross_mae_m",
                           "speed_mae_mps", "curv_mae_1pm", "heading_mae_deg")}
        print(f"  {k:24s} max|ADE change| {d:.10e}  windows repicking NEW "
              f"{rp}/{n}  max|family change| {max(dfam.values()):.10e}")
        b5[k] = {"max_abs_ade_change_2s": d, "n_windows_repicking_new": rp,
                 "max_abs_family_change": dfam}
    B5 = all(v["max_abs_ade_change_2s"] == 0.0 and
             v["n_windows_repicking_new"] == 0 for v in b5.values())
    print(f"  ==> B5 {'PASS' if B5 else 'FAIL'}  -- and this IS the scope "
          f"limit: the capability CANNOT appear in ade_0_2s")
    res["B5"] = {"pass": bool(B5), "arms": b5}

    # ================================================================== B7 ===
    print("\n########## B7  SUPERVISION RATE (NOT execution) ################")
    b7 = {}
    for k in ("A2_ruleS_triseg119", "A1_shipped123_twoseg", "A3_composed125"):
        new = idx[k] >= n_base
        print(f"  {k:24s} LC {int(new[LC].sum()):>5d}/{int(LC.sum()):<5d} "
              f"= {100.0*new[LC].mean():6.2f} %   TURN "
              f"{100.0*new[TN].mean():5.2f} %   ALL "
              f"{100.0*new.mean():5.2f} %")
        b7[k] = {"lane_change": {"n": int(LC.sum()), "k": int(new[LC].sum()),
                                 "rate": float(new[LC].mean())},
                 "turn": {"n": int(TN.sum()), "k": int(new[TN].sum()),
                          "rate": float(new[TN].mean())},
                 "all": {"n": int(n), "k": int(new.sum()),
                         "rate": float(new.mean())}}
    B7 = b7["A2_ruleS_triseg119"]["lane_change"]["rate"] >= 0.20
    print(f"  ==> B7 {'PASS' if B7 else 'FAIL'} (threshold 20 %, INHERITED "
          f"from D-TWOSEG B7, not re-chosen)")
    # per-candidate, so a takeover would be visible
    per = {}
    ie = idx["A2_ruleS_triseg119"]
    for j in range(n_base, A2.shape[0]):
        m = ie == j
        per[f"a_lat={float(A2[j,1]):+.2f}_t1={float(A2[j,2]):.1f}_"
            f"t2={float(A2[j,3]):.1f}"] = {"n_all": int(m.sum()),
                                           "n_lane_change": int((m & LC).sum())}
    for k2, v2 in per.items():
        print(f"     {k2:34s} all {v2['n_all']:>5d}  LC {v2['n_lane_change']:>5d}")
    res["B7"] = {"pass": bool(B7), "threshold": 0.20, "rates": b7,
                 "per_candidate_A2": per}

    # ================================================================== B6 ===
    print("\n########## B6  KINEMATICS + VACUITY GATE #######################")
    speeds = list(np.linspace(0.0, 36.0, 19))
    km_new = ts.kamm_report(A2[n_base:], speeds, control_units="alat",
                            mu=0.7, alat_v_floor=P.ALAT_V_FLOOR,
                            kappa_cap=P.KAPPA_CAP)
    km_inc = ts.kamm_report(C0, speeds, control_units="alat", mu=0.7,
                            alat_v_floor=P.ALAT_V_FLOOR,
                            kappa_cap=P.KAPPA_CAP)
    print(f"  new candidates : peak {km_new['peak_total_g']:.4f} g  "
          f"violations {km_new['n_violations']}/"
          f"{km_new['n_candidate_speed_pairs']}  kappa_cap reached "
          f"{km_new['kappa_cap_reached']}")
    print(f"  incumbent bank : peak {km_inc['peak_total_g']:.4f} g  "
          f"violations {km_inc['n_violations']}/"
          f"{km_inc['n_candidate_speed_pairs']}  kappa_cap reached "
          f"{km_inc['kappa_cap_reached']}")
    man_lc = int((idx["A2_ruleS_triseg119"][LC] >= n_base).sum())
    man_all = int((idx["A2_ruleS_triseg119"] >= n_base).sum())
    print(f"  VACUITY GATE   : manoeuvre rate {man_lc}/{int(LC.sum())} "
          f"lane-change and {man_all}/{n} overall windows SELECT a new "
          f"candidate -- the safety zero is not bought by declining the "
          f"manoeuvre")
    B6 = (km_new["n_violations"] == 0 and not km_new["kappa_cap_reached"]
          and km_new["peak_total_g"] <= km_inc["peak_total_g"] and man_lc > 0)
    print(f"  ==> B6 {'PASS' if B6 else 'FAIL'}")
    res["B6"] = {"pass": bool(B6), "new_candidates": km_new,
                 "incumbent": km_inc,
                 "manoeuvre_rate": {"lane_change_k": man_lc,
                                    "lane_change_n": int(LC.sum()),
                                    "all_k": man_all, "all_n": int(n)}}

    # ================================================================== B3 ===
    print("\n########## B3  SUPPLY (INHERITED threshold 0.10 m) #############")
    g_lc = res["arms"]["A2_ruleS_triseg119"]["gain_lane_change"]
    g_tn = res["arms"]["A2_ruleS_triseg119"]["gain_turn"]
    B3 = g_lc >= 0.10 and abs(g_tn) <= 0.01
    print(f"  LC supply gain {g_lc:+.4f} m (>= 0.10)  TURN control "
          f"{g_tn:+.4f} m (|.| <= 0.01)  ==> B3 {'PASS' if B3 else 'FAIL'}")
    res["B3"] = {"pass": bool(B3), "gain_lane_change": g_lc,
                 "gain_turn": g_tn, "threshold_m": 0.10,
                 "turn_control_tolerance_m": 0.01}

    # ================================================================== B0 ===
    print("\n########## B0  THE INHERITED CAPABILITY BAR ####################")
    g1 = res["arms"]["A1_shipped123_twoseg"]["gain_lane_change"]
    g1p = res["arms"]["A1p_twoseg_pair119"]["gain_lane_change"]
    B0a = g_lc > g1
    print(f"  B0a  A2 LC gain {g_lc:.4f}  vs  the SHIPPED SIX {g1:.4f}  "
          f"-> {'PASS' if B0a else 'FAIL'}")
    print(f"       (equal-cost context, NOT the verdict: the 2-candidate "
          f"two-segment pair {g1p:.4f})")

    def dcurv(k, mask):
        rep = mask & (idx[k] >= n_base)
        if not rep.any():
            return float("nan"), float("nan"), float("nan"), 0
        cb = float(np.nanmean(P.families(pick["A0_base117"][rep], GT[rep],
                                         P.SLOT_DT)["curv_mae_1pm"]))
        ce = float(np.nanmean(P.families(pick[k][rep], GT[rep],
                                         P.SLOT_DT)["curv_mae_1pm"]))
        return cb, ce, ce - cb, int(rep.sum())

    c1b, c1e, d1, n1 = dcurv("A1_shipped123_twoseg", LC)
    c2b, c2e, d2c, n2 = dcurv("A2_ruleS_triseg119", LC)
    B0b = d2c < d1
    print(f"  B0b  d-curv on OWN repicks: A2 {d2c:+.6f} (n={n2})  vs  "
          f"A1 {d1:+.6f} (n={n1})  -> {'PASS' if B0b else 'FAIL'}")
    # the robustness read the prereg promised: the COMMON repick set
    common = LC & (idx["A1_shipped123_twoseg"] >= n_base) & \
        (idx["A2_ruleS_triseg119"] >= n_base)
    if common.any():
        cc = {k: float(np.nanmean(P.families(pick[k][common], GT[common],
                                             P.SLOT_DT)["curv_mae_1pm"]))
              for k in ("A0_base117", "A1_shipped123_twoseg",
                        "A2_ruleS_triseg119")}
        print(f"  B0b' COMMON repicks (n={int(common.sum())}): base "
              f"{cc['A0_base117']:.6f}  A1 {cc['A1_shipped123_twoseg']:.6f} "
              f"({cc['A1_shipped123_twoseg']-cc['A0_base117']:+.6f})  A2 "
              f"{cc['A2_ruleS_triseg119']:.6f} "
              f"({cc['A2_ruleS_triseg119']-cc['A0_base117']:+.6f})")
        agree = ((cc["A2_ruleS_triseg119"] - cc["A0_base117"]) <
                 (cc["A1_shipped123_twoseg"] - cc["A0_base117"])) == B0b
        print(f"  B0b' the common-set read {'AGREES' if agree else 'DISAGREES'}"
              f" with the own-repicks read")
    else:
        cc, agree = {}, None
    B0c = abs(g_tn) <= 0.01
    print(f"  B0c  turn control |{g_tn:+.4f}| <= 0.01 -> "
          f"{'PASS' if B0c else 'FAIL'}")
    print(f"  B0d  degenerate arms exact zeros -> {'PASS' if B4 else 'FAIL'}")
    B0 = bool(B0a and B0b and B0c and B4)
    print(f"  ==> B0 {'PASS' if B0 else 'FAIL'}  (all four clauses; "
          f"three-of-four is a FAIL)")
    res["B0"] = {"pass": B0, "B0a_lc_supply": {"pass": bool(B0a),
                 "A2": g_lc, "A1_shipped_six": g1, "A1p_equal_cost_pair": g1p},
                 "B0b_curvature_own_repicks": {"pass": bool(B0b),
                 "A2_delta": d2c, "A2_n": n2, "A1_delta": d1, "A1_n": n1,
                 "common_repicks": cc, "common_n": int(common.sum()),
                 "common_agrees_with_own": agree},
                 "B0c_turn_control": {"pass": bool(B0c), "gain_turn": g_tn},
                 "B0d_degenerate_zeros": {"pass": bool(B4)}}

    # ====================================================== FOUR FAMILIES ===
    print("\n########## FOUR FAMILIES, never pooled, BOTH DIRECTIONS ########")
    ff = {}
    hdr = (f"  {'arm':<24}{'n':>7}{'ADE':>9}{'along':>9}{'cross':>9}"
           f"{'speed':>9}{'curv':>10}{'head':>8}")
    for nm, mask in (("lane_change", LC), ("turn", TN), ("all", ALL)):
        print(f"  -- {nm} (n = {int(mask.sum())}) --")
        print(hdr)
        ff[nm] = {}
        for k in ("A0_base117", "A1_shipped123_twoseg", "A2_ruleS_triseg119",
                  "A3_composed125", "F_straight_family", "F_const_velocity"):
            r = fam(pick[k], GT, mask)
            ff[nm][k] = r
            print(f"  {k:<24}{r['n']:>7d}{r['ade_m']:>9.4f}"
                  f"{r['along_mae_m']:>9.4f}{r['cross_mae_m']:>9.4f}"
                  f"{r['speed_mae_mps']:>9.4f}{r['curv_mae_1pm']:>10.6f}"
                  f"{r['heading_mae_deg']:>8.3f}")
    res["four_families_6s"] = ff

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")
    print(f"BARS: B0 {'PASS' if B0 else 'FAIL'} | B2 {'PASS' if B2 else 'FAIL'}"
          f" | B3 {'PASS' if B3 else 'FAIL'} | B4 {'PASS' if B4 else 'FAIL'}"
          f" | B5 {'PASS' if B5 else 'FAIL'} | B6 {'PASS' if B6 else 'FAIL'}"
          f" | B7 {'PASS' if B7 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
