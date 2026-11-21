"""s9_lever_table.py -- the lever x stratum ceilings table for T_ATTRIB.md, recomputed from the artifacts (nothing typed by hand).
EXPLORATORY. (E) = EXACT window-stratum ceiling (MEASURED, T0 oracle, additive over disjoint windows); (D) = DOSE-MODEL estimate for a cell stratum (ESTIMATED: assumes O
responds to the SIZE of T's constraint error, not to which cell carries it). Bar 2F = 0.2415 m.
"""
import re

import numpy as np

from tcommon import *
from s3_tail import KIND

TWO_F = 0.2415
ALPHAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def curve(seed):
    from s7_ceilings import ade_curve
    A = ade_curve(seed)
    gf = (A["own"] - A["full"]).mean()
    rr = np.array([(A["own"] - A[k]).mean() / gf for k in ("own", "a25", "a50", "a75", "cons")])
    return rr, gf


def r_of_alpha(a, rr):
    if a < 0.25:                                         # below the first measured point: power law through the origin fitted to (0.25, 0.5)
        b = np.log(rr[2] / rr[1]) / np.log(2.0)
        return rr[1] * (a / 0.25) ** b
    return float(np.interp(a, ALPHAS, rr))


def main():
    T = load()
    eid, has, G = T["ci"], T["has"], T["gt_std"]
    N = len(eid)
    nav = T["nav_tok"]
    ue = np.unique(eid)
    # probe headroom from s8_probe.out (kind lines)
    probe = {}
    for ln in open(S + "s8_probe.out", encoding="utf-8"):
        m = re.match(r"\s+(\S[^\d]*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+", ln)
        if m and any(k in ln for k in ("ANGLE", "GEOMETRY", "LEAD GAP", "SPEED", "TIME-GAP", "TIME-TO-EVENT")):
            probe[m.group(1).strip()] = tuple(float(m.group(i)) for i in range(2, 7))
    print("probe table parsed (const, probeA, probeB, T s0, T s1):", {k: v for k, v in probe.items()})
    lead_gap_head = 1 - (probe["LEAD GAP distance"][2] / probe["LEAD GAP distance"][3]) ** 2          # SSE share removable if T matched the privileged-lead probe
    tg_head = max(0.0, 1 - (probe["TIME-GAP to lead (gap/closing speed)"][2] / probe["TIME-GAP to lead (gap/closing speed)"][3]) ** 2)
    print(f"privileged-lead headroom (relative SSE cut if T matched probe B): LEAD GAP {lead_gap_head:.3f}, TIME-GAP {tg_head:.3f}")
    rows = []
    for seed in (0, 1):
        arm = f"M2b_s{seed}"
        rr, gf = curve(seed)
        g = np.load(S + f"gap_{arm}.npz")
        gc = g["gc"]
        E, P = sq(T, arm)
        E[~has], P[~has] = 0.0, False
        TOT = E.sum()
        cell = lambda idx: E[:, idx].sum() / TOT
        turn = np.isfinite(G[:, 1]) & has
        sil = turn & (nav == 0)
        cols = list(range(1, 10))
        phi_sil_turn = E[sil][:, cols].sum() / TOT
        # making silent windows' turn cells as good as the route-speaking ones (same model, measured): SSE cut
        rm = lambda m: E[m][:, cols].sum() / P[m][:, cols].sum()
        cut_sil = 1 - rm(turn & (nav != 0)) / rm(sil)
        big = np.zeros_like(P)
        for c in (19, 21):
            big[:, c] = P[:, c] & (np.abs(np.where(P[:, c], G[:, c], 0)) >= 3)
        time_idx = [i for i in KIND if KIND[i].startswith("TIME")]
        speed_idx = [i for i in KIND if KIND[i].startswith("SPEED")]
        lead_idx = [i for i in KIND if KIND[i].startswith("LEAD")]
        timegap_idx = [i for i in KIND if KIND[i].startswith("TIME-GAP")]
        lead_present = has & (T["lead_valid"] > 0)
        heavy = lead_present & ((np.abs(np.where(np.isfinite(G[:, 19]), G[:, 19], 0)) >= 3) | (np.abs(np.where(np.isfinite(G[:, 21]), G[:, 21], 0)) >= 3))
        topidx = np.argsort(-np.where(has, E.sum(1), -1))[:int(0.1 * has.sum())]
        top10 = np.zeros(len(eid), bool); top10[topidx] = True
        Ew = lambda m: gc[m].sum() / N
        D = lambda phi: (r_of_alpha(1 - np.sqrt(1 - phi), rr) * gf, 1 - np.sqrt(1 - phi))
        cases = [
            ("T-error-dominant cells: TIME-TO-EVENT + TIME-GAP, ALL their error removed", cell(time_idx), D(cell(time_idx)), Ew(top10), "top-10% SSE windows (all constraints)"),
            ("heavy-tail lead_tg cells (|GT|>=3 SD), removed", E[big].sum() / TOT, D(E[big].sum() / TOT), Ew(heavy), "32 heavy-tail lead windows"),
            ("turn cells of ROUTE-SILENT turn windows, removed", phi_sil_turn, D(phi_sil_turn), Ew(sil), "210 route-silent turn windows (all constraints)"),
            ("  same, but only up to what the model does when the route speaks", phi_sil_turn * cut_sil, D(phi_sil_turn * cut_sil), Ew(sil), "same windows"),
            ("LoRA stratum (lead cells: LEAD GAP + TIME-GAP), ALL removed", cell(lead_idx) + cell(timegap_idx), D(cell(lead_idx) + cell(timegap_idx)), Ew(lead_present), "lead-present windows (all constraints)"),
            ("  same, but only the privileged-lead probe headroom", cell(lead_idx) * lead_gap_head + cell(timegap_idx) * tg_head, D(cell(lead_idx) * lead_gap_head + cell(timegap_idx) * tg_head), Ew(lead_present), "same windows"),
            ("SPEED-kind cells, ALL removed (size-agnostic dose)", cell(speed_idx), D(cell(speed_idx)), float("nan"), "n/a"),
            ("all cells, RMSE cut alpha=0.06 (DSCALE 2x, PI-quoted -6%)", 1 - 0.94 ** 2, (r_of_alpha(0.06, rr) * gf, 0.06), float("nan"), "n/a"),
            ("all cells, RMSE cut alpha=0.054 (7-model ensemble of T)", 1 - 0.946 ** 2, (r_of_alpha(0.054, rr) * gf, 0.054), float("nan"), "n/a"),
            ("all cells, GRID 8x20 / HIST 10 / PGT / WC / DF50 as measured (s5)", float("nan"), (0.0, 0.0), float("nan"), "n/a (no separated improvement)"),
        ]
        print(f"\n{arm}: gap_full {gf:.4f} m, r(alpha) at 0/.25/.5/.75/1: " + " ".join(f"{x:.3f}" for x in rr) + f";  2F = {TWO_F} m")
        print(f"   {'case':74s} {'phi':>6s} {'alpha_eq':>8s} {'(D) m':>7s} | {'(E) outer bound m':>17s}  window set")
        for nm, phi, (d, a), e, es in cases:
            print(f"   {nm:74s} {phi:6.3f} {a:8.3f} {d:7.3f} | {e:17.3f}  {es}")
        print(f"   need alpha = {np.interp(TWO_F / gf, rr, ALPHAS):.2f} uniform RMSE cut to clear 2F; route-silent turn-cell SSE cut if matched to route-speaking: {cut_sil:.2f}")


if __name__ == "__main__":
    main()
