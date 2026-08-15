"""DIR_YAW_RAD 0.15 -> 0.10 for the two `-lf19` panels the PAPER quotes in words.

⛔ WHY THIS IS NOT THE RE-RUN THAT WAS ASKED FOR. The owed action was "re-run the two
`-lf19` panels gate-swept". That is NOT executable anywhere reachable today:

  * the corpus `/root/valdata/val19_leakfree` (19 leak-free episodes) lived on the
    eval pod, which was shut down 2026-08-02 (`Project Steering/POD_SHUTDOWN_2026-08-02.md`).
    Probed for it at four names/paths — no `ep_*.pt` exists anywhere in the repo (0 hits).
  * `windows_v1-lf19.pt` / `windows_v2corpus-lf19.pt` were NOT in the pod-rescue bundle
    (`stack/experiments/pod-rescue-20260802/eval/` holds 7 JSONs and no tensors).
  * the only GPU on the fleet is the Thor, which is training a 336 M model.

⭐ WHAT IS EXECUTABLE, AND IT IS EXACT. The panel banks the COMPLETE sufficient statistic
for the coherence kappa: the manoeuvre-direction marginal, the trajectory-direction
marginal, and the agreement rate. `_kappa` is a function of exactly those three
(`hierarchy.py:248-252` computes po, pe from the two marginals), so the published kappa
is REPRODUCIBLE FROM THE BANKED JSON to the published 4 decimals — verified in
`verify_reconstruction()` as a hard gate before anything else runs.

⭐ AND THE GATE MOVE IS MONOTONE, which turns "unknowable" into "bounded". `_dir_of`
(`hierarchy.py:204-210`) thresholds a SIGNED net yaw at +-g. Lowering g 0.15 -> 0.10 can
ONLY move a window S->L (net yaw in (0.10, 0.15]) or S->R (in [-0.15, -0.10)). It can
never move L->R, L->S, R->L or R->S. So kappa@0.10 is a function of:
   m  = how many trajectory windows sit in the band, and
   which manoeuvre classes those m windows carry.
Both are unknown, but both are CONSTRAINED by the banked margins + agreement. This
module computes the EXACT envelope [kappa_min(m), kappa_max(m)] over every confusion
matrix and every crossing pattern consistent with the banked numbers, and reports the
smallest m at which each published VERDICT WORD can break.

Evidence class: MEASURED (ours) for the reconstruction and the envelope -- both are
deterministic functions of banked artifacts. ESTIMATED for any statement about which m
is realistic; the band masses quoted for that are MEASURED on three other corpora.

0 GPU, CPU only, deterministic, no network.
"""
from __future__ import annotations

import argparse

import json
import os
import random

import numpy as np


R_LEFT, R_STRAIGHT, R_RIGHT = 0, 1, 2
DIRK = ("route_left", "route_straight", "route_right")

# `four_families.py:886-889` -- the ladder the programme actually PUBLISHES.
# ⚠️ NOT the `kappa >= 0.2` ladder `hierarchy._gate_sensitivity:262` tests for
# `verdict_stable`; that mismatch is a logged instrument defect.
LADDER = ((0.1, "DECORATIVE"), (0.4, "WEAK"), (float("inf"), "SUBSTANTIAL"))


def verdict(k):
    for hi, w in LADDER:
        if k < hi:
            return w
    return "SUBSTANTIAL"


def kappa_from_margins(r, c, T, n):
    """kappa given manoeuvre marginal r, trajectory marginal c, agreeing count T.

    Identical arithmetic to `hierarchy._kappa` (`:248-252`): po is the agreement
    rate, pe the product of the two marginals, both over the SAME n windows.
    """
    po = T / n
    pe = sum(r[i] * c[i] for i in range(3)) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


# --------------------------------------------------------------------------- #
# 1. every confusion matrix consistent with the banked numbers                 #
# --------------------------------------------------------------------------- #
def enumerate_confusions(r, c, T):
    """All 3x3 non-negative integer N with row sums r, col sums c, trace T.

    The banked JSON pins the two marginals and the agreement rate but NOT the
    off-diagonal structure. Rather than guess it, enumerate every N the banked
    numbers admit and carry the whole set through -- so the envelope is a
    statement about the data, not about a chosen reconstruction.
    """
    out = []
    for a in range(0, min(r[0], c[0]) + 1):                 # N[L][L]
        for f in range(0, min(r[2], c[2]) + 1):             # N[R][R]
            e = T - a - f                                   # N[S][S]
            if not (0 <= e <= min(r[1], c[1])):
                continue
            for b in range(0, r[0] - a + 1):                # N[L][S], free param
                lr = r[0] - a - b                           # N[L][R]
                rs = (c[1] - e) - b                         # N[R][S]
                if lr < 0 or rs < 0:
                    continue
                rl = (r[2] - f) - rs                        # N[R][L]
                if rl < 0:
                    continue
                sl = (c[0] - a) - rl                        # N[S][L]
                if sl < 0:
                    continue
                sr = (r[1] - e) - sl                        # N[S][R]
                if sr < 0:
                    continue
                if lr + sr + f != c[2]:                     # col R closure
                    continue
                N = ((a, b, lr), (sl, e, sr), (rl, rs, f))
                assert all(sum(N[i]) == r[i] for i in range(3))
                assert all(sum(N[i][j] for i in range(3)) == c[j] for j in range(3))
                assert N[0][0] + N[1][1] + N[2][2] == T
                out.append(N)
    return out


# --------------------------------------------------------------------------- #
# 2. exact agreement-change range for a given crossing split                   #
# --------------------------------------------------------------------------- #
def delta_range(s, A, B):
    """Range of the agreement CHANGE when A windows cross S->L and B cross S->R.

    `s = (sL, sS, sR)` is the trajectory-straight column of N split by manoeuvre
    class -- i.e. how many of the straight-classified windows were declared left,
    straight, right. Movers are drawn from those pools.

    A window declared LEFT that moves S->L GAINS an agreement (+1). Declared RIGHT
    moving S->R gains (+1). Declared STRAIGHT that moves ANYWHERE LOSES one (-1).
    Declared LEFT moving S->R (and RIGHT moving S->L) is neutral (0) -- it
    disagreed before and disagrees after.

    Solved in closed form per side (see `_delta_min` / `_delta_max`); the constraint
    matrix is a transportation matrix, so the integer optimum is attained at a
    vertex and these greedies are EXACT, not heuristics.
    Cross-checked against
    full integer brute force in `_selftest_delta_range`.
    """
    sL, sS, sR = s
    if A < 0 or B < 0 or A + B > sL + sS + sR:
        return None
    return _delta_min(sL, sS, sR, A, B), _delta_max(sL, sS, sR, A, B)


def _delta_max(sL, sS, sR, A, B):
    """Assign the +1 moves first, then the 0 moves, then the forced -1 moves.

    The two +1 options (declared-LEFT crossing to LEFT, declared-RIGHT crossing to
    RIGHT) draw on DISJOINT pools and fill DISJOINT slots, so taking both to their
    cap can never block the other -- which is what makes the greedy exact rather
    than merely plausible.
    """
    xL = min(A, sL)
    yR = min(B, sR)
    a, b = A - xL, B - yR
    xR = min(a, sR - yR); a -= xR              # declared-RIGHT crossing LEFT: 0
    yL = min(b, sL - xL); b -= yL              # declared-LEFT crossing RIGHT: 0
    return xL + yR - a - b                     # whatever is left came from pool S


def _delta_min(sL, sS, sR, A, B):
    """Push as much as possible through the declared-STRAIGHT pool (every such
    crossing DESTROYS an agreement), then take the 0-value moves, and only then the
    +1 moves.

    ⚠️ The split of the shared straight pool between the two slots is NOT decided
    greedily -- it is the one genuine degree of freedom (a pool-S unit spent on the
    slot whose fallback is free wastes the -1), so it is enumerated exactly. That
    ambiguity is why this is not simply `_delta_max` with the signs flipped.
    """
    best = None
    for xS in range(0, min(A, sS) + 1):
        yS = min(B, sS - xS)                   # spending pool S is never worse
        a, b = A - xS, B - yS
        xR = min(a, sR); a -= xR               # 0-value fills first
        yL = min(b, sL); b -= yL
        xL, yR = a, b                          # forced +1 moves
        if xL + yL > sL or xR + yR > sR:
            continue
        d = xL + yR - xS - yS
        best = d if best is None else min(best, d)
    return best


def _brute_delta_range(s, A, B):
    """Reference implementation -- full integer enumeration. Small cases only."""
    sL, sS, sR = s
    best = None
    for xL in range(min(A, sL) + 1):
        for xS in range(min(A - xL, sS) + 1):
            xR = A - xL - xS
            if xR < 0 or xR > sR:
                continue
            for yL in range(min(B, sL - xL) + 1):
                for yS in range(min(B - yL, sS - xS) + 1):
                    yR = B - yL - yS
                    if yR < 0 or yR > sR - xR:
                        continue
                    d = xL + yR - xS - yS
                    best = (d, d) if best is None else (min(best[0], d),
                                                        max(best[1], d))
    return best


def _selftest_delta_range(seed=0, trials=300):
    """⛔ The LP is the load-bearing step. Gate it against brute force before use."""
    rnd = random.Random(seed)
    for _ in range(trials):
        s = tuple(rnd.randint(0, 6) for _ in range(3))
        tot = sum(s)
        if tot == 0:
            continue
        m = rnd.randint(0, tot)
        A = rnd.randint(0, m)
        got, want = delta_range(s, A, m - A), _brute_delta_range(s, A, m - A)
        assert got == want, f"delta_range{ (s, A, m - A) } lp={got} brute={want}"
    return trials


# --------------------------------------------------------------------------- #
# 3. the envelope                                                              #
# --------------------------------------------------------------------------- #
def _reduce_cols(cols, m):
    """Exact reduction of the column-split set for a given crossing count.

    Two facts make this lossless. (1) No pool can supply more than `m` movers, so
    capping every pool at `m` changes nothing. (2) Enlarging a pool only ADDS
    feasible mover patterns, so both the minimum and the maximum of Delta are
    attained on the componentwise-MAXIMAL capped splits -- the dominated ones can
    only reproduce values the dominating ones already reach.

    Without this the envelope is ~10^6 solver calls; with it, ~10^4.
    """
    capped = {tuple(min(si, m) for si in s) for s in cols}
    return [a for a in capped
            if not any(b != a and all(bi >= ai for bi, ai in zip(b, a))
                       for b in capped)]


def _selftest_reduce_cols(seed=1, trials=200):
    """⛔ Gate the reduction: the reduced set must give the SAME Delta extremes."""
    rnd = random.Random(seed)
    for _ in range(trials):
        cols = [tuple(rnd.randint(0, 8) for _ in range(3)) for _ in range(6)]
        m = rnd.randint(0, 12)
        A = rnd.randint(0, m)
        def ext(cs):
            vals = [delta_range(s, A, m - A) for s in cs if m <= sum(s)]
            vals = [v for v in vals if v]
            return (min(v[0] for v in vals), max(v[1] for v in vals)) if vals else None
        full = ext([tuple(min(si, m) for si in s) for s in cols])
        red = ext(_reduce_cols(cols, m))
        assert full == red, f"reduce_cols lost an extreme: {full} != {red}"
    return trials


def envelope(r, c, T, n, m_max=None):
    """[kappa_min(m), kappa_max(m)] over every admissible N and crossing pattern.

    For a FIXED (A, B) the chance-agreement pe is fixed (it depends only on the new
    marginal), so kappa is strictly increasing in the agreement change -- the
    extremes therefore sit exactly at Delta_min / Delta_max and no interior search
    is needed. The outer loops over N and A are exhaustive.
    """
    confs = enumerate_confusions(r, c, T)
    if not confs:
        raise SystemExit("no confusion matrix consistent with the banked numbers")
    cols = sorted({(N[0][1], N[1][1], N[2][1]) for N in confs})   # traj-S column
    m_max = c[1] if m_max is None else min(m_max, c[1])
    rows = []
    for m in range(0, m_max + 1):
        lo = hi = None
        arg_lo = arg_hi = None
        for s in _reduce_cols(cols, m):
            if m > sum(s):
                continue
            for A in range(0, m + 1):
                B = m - A
                dr = delta_range(s, A, B)
                if dr is None:
                    continue
                cprime = (c[0] + A, c[1] - m, c[2] + B)
                for d, side in ((dr[0], "lo"), (dr[1], "hi")):
                    k = kappa_from_margins(r, cprime, T + d, n)
                    if k is None:
                        continue
                    if lo is None or k < lo:
                        lo, arg_lo = k, {"s": list(s), "A": A, "B": B, "delta": d}
                    if hi is None or k > hi:
                        hi, arg_hi = k, {"s": list(s), "A": A, "B": B, "delta": d}
        if lo is None:
            continue
        rows.append({"m": m, "frac_windows_crossing": round(m / n, 4),
                     "kappa_min": round(lo, 4), "kappa_max": round(hi, 4),
                     "verdict_min": verdict(lo), "verdict_max": verdict(hi),
                     "argmin": arg_lo, "argmax": arg_hi})
    return rows, confs, cols


def central_estimate(r, c, T, n, m, cols):
    """ESTIMATED, not measured: crossers drawn INDEPENDENTLY of the manoeuvre class.

    The null that the band (0.10, 0.15] carries the same manoeuvre mix as the rest
    of the straight-classified windows. Reported as a central line only -- the
    envelope is the admissible statement.
    """
    out = []
    for s in cols:
        tot = sum(s)
        if tot == 0 or m > tot:
            continue
        # split the m crossers pro rata across the pools, then L/R pro rata to the
        # trajectory marginal's own left/right balance.
        share = [m * si / tot for si in s]
        pL = c[0] / max(c[0] + c[2], 1)
        A, B = m * pL, m * (1 - pL)
        delta = share[0] * pL + share[2] * (1 - pL) - share[1]
        cprime = (c[0] + A, c[1] - m, c[2] + B)
        k = kappa_from_margins(r, cprime, T + delta, n)
        if k is not None:
            out.append(k)
    return (round(float(np.mean(out)), 4), round(float(min(out)), 4),
            round(float(max(out)), 4)) if out else None


def break_points(rows, k15):
    """Smallest crossing count m at which the PUBLISHED verdict word can break."""
    w0 = verdict(k15)
    for row in rows:
        if row["verdict_min"] != w0 or row["verdict_max"] != w0:
            return {"m_star": row["m"], "frac": row["frac_windows_crossing"],
                    "published_word": w0,
                    "first_other_word": (row["verdict_min"]
                                         if row["verdict_min"] != w0
                                         else row["verdict_max"]),
                    "kappa_at_m_star": [row["kappa_min"], row["kappa_max"]]}
    return {"m_star": None, "published_word": w0,
            "_read": "word holds across the entire swept crossing range"}


# --------------------------------------------------------------------------- #
def load_panel(path):
    d = json.load(open(path, encoding="utf-8"))
    con = d["consistency"]
    mvt = con["maneuver_vs_trajectory"]
    dist = con["distributions"]
    n = int(d["n_windows"])
    r = [int(dist["maneuver_dir"][k]) for k in DIRK]
    c = [int(dist["trajectory_dir"][k]) for k in DIRK]
    g = [int(dist["gt_dir"][k]) for k in DIRK]
    po = float(mvt["agreement"]["mean"])
    return {"n": n, "n_episodes": int(d["n_episodes"]), "md": r, "traj15": c,
            "gt15": g, "agreement": po, "T": int(round(po * n)),
            "kappa_published": mvt["kappa"],
            "kappa_turn_subset_published": mvt.get("kappa_turn_subset"),
            "n_turn_active": mvt.get("n_turn_active"),
            "model": d.get("model"), "ckpt_step": d.get("ckpt_step")}


def verify_reconstruction(p):
    """⛔ HARD GATE. If the banked marginals do not reproduce the published kappa,
    the whole envelope is built on a misread field and must not run."""
    k = kappa_from_margins(p["md"], p["traj15"], p["T"], p["n"])
    got, want = round(k, 4), float(p["kappa_published"])
    if abs(got - want) > 5e-4:
        raise SystemExit(f"RECONSTRUCTION FAILED: {got} != published {want}")
    return {"kappa_reconstructed": got, "kappa_published": want,
            "abs_diff": round(abs(got - want), 6), "exact": True}


PANELS = {
    "v1-lf19": "TanitAD Research Hub/Evaluation/Implementation/incoming/"
               "2026-08-02-four-family-panel/hier_v1-lf19.json",
    "v2corpus-lf19": "TanitAD Research Hub/Evaluation/Implementation/incoming/"
                     "2026-08-02-four-family-panel/hier_v2corpus-lf19.json",
}


def measured_band_masses(repo):
    """MEASURED |net yaw| band mass in (0.10, 0.15] -- the quantity that IS `m/n`.

    Reconstructed from `head_deg` banked per window (`bench.py:399` writes
    `net_heading_change_deg(ep.poses, last)` over K_MAX = 20, the SAME poses,
    horizon and wrap `hierarchy.py:557` uses for the gate's input), plus the two
    gate-swept panels. These are GROUND-TRUTH band masses on OTHER corpora; they
    calibrate what `m` is plausible, they do not measure the lf19 trajectory.
    """
    import torch
    out = {}
    wp = os.path.join(repo, "taniteval/results/windows_flagship-30k.pt")
    if os.path.exists(wp):
        hd = torch.load(wp, map_location="cpu",
                        weights_only=False)["head_deg"].numpy() * np.pi / 180.0
        out["canonical_val_881w_40ep_GT"] = {
            "n": int(hd.size),
            "frac_above_0.15": round(float((hd > 0.15).mean()), 4),
            "frac_above_0.10": round(float((hd > 0.10).mean()), 4),
            "band_0.10_0.15": round(float(((hd > 0.10) & (hd <= 0.15)).mean()), 4),
            "median_abs_net_yaw_rad": round(float(np.median(hd)), 4),
            "_source": "taniteval/results/windows_flagship-30k.pt :: head_deg",
        }
    gp = os.path.join(repo, "TanitAD Research Hub/Architecture & Inference/"
                            "Implementation/incoming/2026-08-06-v1-defect-triage/"
                            "results/hier_v1arch_gateswept.json.xz")
    if os.path.exists(gp):
        import lzma
        gs = json.loads(lzma.open(gp).read())["consistency"]["gate_sensitivity"]
        pg = gs["per_gate"]
        out["oodval_q90_880w_40ep_GT"] = {
            "n": gs["n_windows"],
            "frac_above_0.15": pg["0.15"]["frac_gt_turning"],
            "frac_above_0.10": pg["0.10"]["frac_gt_turning"],
            "band_0.10_0.15": round(pg["0.10"]["frac_gt_turning"]
                                    - pg["0.15"]["frac_gt_turning"], 4),
            "median_abs_net_yaw_rad": gs["gt_turn_magnitude"][
                "median_abs_net_yaw_rad"],
            "_source": "hier_v1arch_gateswept.json.xz (the ONE swept panel)",
            "_measured_kappa_move": {
                "kappa_0.15": pg["0.15"]["maneuver_vs_trajectory_kappa"],
                "kappa_0.10": pg["0.10"]["maneuver_vs_trajectory_kappa"],
                "delta": round(pg["0.10"]["maneuver_vs_trajectory_kappa"]
                               - pg["0.15"]["maneuver_vs_trajectory_kappa"], 4)},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-max", type=int, default=140)
    a = ap.parse_args()

    n_self = _selftest_delta_range()
    n_self_red = _selftest_reduce_cols()
    res = {
        "_question": ("do the PAPER's two verbal tactical claims (TANITAD_PAPER.md "
                      ":1867, :3127) survive DIR_YAW_RAD 0.15 -> 0.10?"),
        "_evidence_class": "MEASURED (ours) — deterministic over banked artifacts",
        "_gpu": 0,
        "_ladder": "four_families.py:886-889 — <0.1 DECORATIVE, <0.4 WEAK, >=0.4 SUBSTANTIAL",
        "_why_not_a_rerun": (
            "the corpus /root/valdata/val19_leakfree died with the eval pod "
            "(POD_SHUTDOWN_2026-08-02.md); windows_{v1,v2corpus}-lf19.pt are not in "
            "the pod-rescue bundle; the only fleet GPU is training. The envelope "
            "below is exact over everything the banked JSON admits."),
        "_selftest_delta_range_trials": n_self,
        "_selftest_reduce_cols_trials": n_self_red,
        "measured_band_masses": measured_band_masses(a.repo),
        "panels": {},
    }
    for key, rel in PANELS.items():
        p = load_panel(os.path.join(a.repo, rel))
        p["reconstruction_gate"] = verify_reconstruction(p)
        rows, confs, cols = envelope(p["md"], p["traj15"], p["T"], p["n"], a.m_max)
        k15 = kappa_from_margins(p["md"], p["traj15"], p["T"], p["n"])
        p["n_admissible_confusions"] = len(confs)
        p["traj_straight_column_splits"] = [list(s) for s in cols]
        p["kappa_0.15"] = round(k15, 4)
        p["verdict_0.15"] = verdict(k15)
        p["envelope"] = rows
        p["verdict_break"] = break_points(rows, k15)
        p["central_estimates"] = {
            str(m): central_estimate(p["md"], p["traj15"], p["T"], p["n"], m, cols)
            for m in (0, 5, 10, 17, 22, 30, 45, 60)}
        res["panels"][key] = p
        row = next(r for r in rows if r["m"] == 0)
        assert abs(row["kappa_min"] - p["kappa_0.15"]) < 1e-9, "m=0 must be the published kappa"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)

    for key, p in res["panels"].items():
        print(f"\n######## {key} — n={p['n']} / {p['n_episodes']} episodes "
              f"· ckpt {p['ckpt_step']}")
        print(f"  reconstruction: {p['reconstruction_gate']}")
        print(f"  md marginal {p['md']} · traj@0.15 {p['traj15']} · gt@0.15 {p['gt15']}")
        print(f"  kappa@0.15 = {p['kappa_0.15']}  ({p['verdict_0.15']})   "
              f"admissible confusions: {p['n_admissible_confusions']}")
        print(f"  {'m':>4} {'frac':>7} {'kappa_min':>10} {'kappa_max':>10}  verdicts")
        for r_ in p["envelope"]:
            if r_["m"] in (0, 5, 10, 17, 22, 30, 45, 60, 90, 120, 140):
                print(f"  {r_['m']:>4} {r_['frac_windows_crossing']:>7} "
                      f"{r_['kappa_min']:>10} {r_['kappa_max']:>10}  "
                      f"{r_['verdict_min']}/{r_['verdict_max']}")
        print(f"  VERDICT BREAK: {p['verdict_break']}")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
