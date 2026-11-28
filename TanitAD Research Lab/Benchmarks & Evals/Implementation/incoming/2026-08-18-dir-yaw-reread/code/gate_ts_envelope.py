"""kappa_turn_subset @ DIR_YAW_RAD 0.10 — exact admissible-set envelope over banked panels.

THE GAP THIS CLOSES (flagged three times, never executed): `kappa_turn_subset` is
gate-dependent (`taniteval/taniteval/hierarchy.py:1045-1056` — the subset mask
`turn = (x != S) | (y != S)` moves with `traj = _dir_of(traj_net)`), and `_gate_sensitivity`
does NOT sweep it (`hierarchy.py:266-276` sweeps the full-set kappas only). Flags:
2026-08-15 `DIR_YAW_GATE_REREAD.md` §2b, 2026-08-16 `DIR_YAW_REREAD.md` §5.2,
2026-08-17 `DIRYAW_REREAD.md` §5 work-item 2 (and `MANEUVER_LABEL_MISMATCH.md:280`: still OPEN).

WHY AN ENVELOPE AND NOT A RE-RUN (MEASURED 2026-08-18): the re-run is credential-blocked —
the canonical val40 FRAMES exist only on Thor (untouchable, live 30k run) and on HF
(`…/epcache-256px-phase0/physicalai-val-0c5f7dac3b11`, ~4.7 GB, HTTP 401 anonymous, no token
on this box: Keys.txt is on the dead G:). The deployed ckpt itself IS local and md5-verified
(`b5f07d9e3dd2ca643949bc86832e6585`).

METHOD — extends the verified 2026-08-16 `gate_envelope.py` machinery with constraints that
pass did not use, which is what makes kappa_turn_subset boundable:
  1. `n_turn_active` pins the (S,S) cell EXACTLY: M[S][S] = n - n_turn.
  2. kappa_turn_subset@0.15 pins the turn-diagonal M[L][L]+M[R][R]: BOTH subset marginals
     are derivable from published numbers alone — man-side (aL, aS-M[S][S], aR), traj-side
     (tL, tS-M[S][S], tR) — so kappa_ts inverts to a diagonal count.
  3. For the one gate-swept panel, the banked kappa@0.10 = 0.5715 ANCHORS the crossing set.
At 0.15 every 3x3 confusion matrix consistent with (row sums = man marginal, col sums = traj
marginal, the (S,S) pin, kappa@0.15 and kappa_ts@0.15 at published 4-dp rounding) is
enumerated. The 0.10 move is monotone (S->L / S->R on the trajectory side only — proof in the
2026-08-16 pass). Every crossing allocation is enumerated; kappa_ts@0.10 min/max per m.

REDUCTION (what makes it tractable, and it is proved not assumed): kappa@0.10 and
kappa_ts@0.10 depend on the allocation ONLY through (alpha = diagonal gains x_L + y_R,
p_S = S-pool drain, X = total movers to L), and on the matrix ONLY through its turn-diagonal
sum and the traj-straight column split (M[L][S], M[S][S], M[R][S]) — the individual cells
M[L][L] vs M[R][R] never appear separately (both kappas use their SUM). The slow exact path
enumerates the full (x_L, y_R, x_S) box per matrix; the fast path enumerates reachable
(alpha, p_S, X) once with union caps over admissible matrices. THE SELF-TEST RUNS BOTH AND
ASSERTS THEY AGREE, alongside the brute-force containment checks. The tool refuses to emit
results if any self-test fails.

Envelope bounds are DETERMINISTIC ADMISSIBLE-SET BOUNDS, not confidence intervals.
Tier: open-loop model-internal coherence diagnostic (T0-family) — never driving performance.
Evidence class: MEASURED (deterministic function of banked artifacts) for every envelope
number; ESTIMATED for any statement about which m is realistic.

0 GPU, CPU only, deterministic, no network. Run:
  python gate_ts_envelope.py     # self-tests, then panels -> ../raw/gate_ts_envelope.json
"""
from __future__ import annotations

import bisect
import json
import lzma
import os

import numpy as np

L, S, R = 0, 1, 2
DIRK = ("route_left", "route_straight", "route_right")
TOL = 5.0e-5 + 1e-12          # half-ULP of a 4-dp rounded value
# four_families.KAPPA_VERDICT_LADDER — the PUBLISHED ladder (0.2 was retired 2026-08-16).
LADDER = ((0.1, "DECORATIVE"), (0.4, "WEAK"), (float("inf"), "SUBSTANTIAL"))

CLONE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      *[".."] * 6))


def band(k):
    if k is None:
        return None
    for hi, w in LADDER:
        if k < hi:
            return w
    return "SUBSTANTIAL"


# --------------------------------------------------------------------------- #
# kappa arithmetic — identical to hierarchy._kappa (:517-531) in closed form   #
# --------------------------------------------------------------------------- #
def kappa_full(M, n):
    a = [sum(M[i]) for i in range(3)]
    t = [sum(M[i][j] for i in range(3)) for j in range(3)]
    po = (M[0][0] + M[1][1] + M[2][2]) / n
    pe = sum(a[i] * t[i] for i in range(3)) / (n * n)
    if abs(1.0 - pe) < 1e-9:
        return None
    return (po - pe) / (1.0 - pe)


def kappa_ts(M, n):
    """kappa over the subset (man != S) | (traj != S) — hierarchy.py:1047,1055."""
    n_turn = n - M[1][1]
    if n_turn == 0:
        return None
    a = [sum(M[i]) for i in range(3)]
    t = [sum(M[i][j] for i in range(3)) for j in range(3)]
    msub = (a[0], a[1] - M[1][1], a[2])
    tsub = (t[0], t[1] - M[1][1], t[2])
    po = (M[0][0] + M[2][2]) / n_turn
    pe = sum(msub[i] * tsub[i] for i in range(3)) / (n_turn * n_turn)
    if abs(1.0 - pe) < 1e-9:
        return None
    return (po - pe) / (1.0 - pe)


# --------------------------------------------------------------------------- #
# 1. admissible matrices at 0.15                                               #
# --------------------------------------------------------------------------- #
def enumerate_admissible(n, a, t, n_turn, kappa_pub, kts_pub):
    e = n - n_turn                                # M[S][S], pinned EXACTLY
    if e < 0 or e > min(a[S], t[S]):
        return []
    out = []
    for m00 in range(0, min(a[L], t[L]) + 1):
        for m01 in range(0, min(a[L] - m00, t[S] - e) + 1):   # M[L][S]
            m02 = a[L] - m00 - m01                # M[L][R]
            m21 = (t[S] - e) - m01                # col S closure -> M[R][S]
            for m22 in range(0, min(a[R] - m21, t[R]) + 1):
                m20 = a[R] - m22 - m21            # row R -> M[R][L]
                if m20 < 0:
                    continue
                m10 = t[L] - m00 - m20            # col L -> M[S][L]
                if m10 < 0:
                    continue
                m12 = a[S] - e - m10              # row S -> M[S][R]
                if m12 < 0:
                    continue
                if m02 + m12 + m22 != t[R]:       # col R closure
                    continue
                M = ((m00, m01, m02), (m10, e, m12), (m20, m21, m22))
                k = kappa_full(M, n)
                if k is None or abs(k - kappa_pub) > TOL:
                    continue
                kt = kappa_ts(M, n)
                if kt is None or abs(kt - kts_pub) > TOL:
                    continue
                out.append(M)
    return out


# --------------------------------------------------------------------------- #
# 2a. SLOW exact move envelope — full (x_L, y_R, x_S) box per matrix           #
# --------------------------------------------------------------------------- #
def _k_pair_arrays(n, a, t, M, alpha, pS, X, m):
    """(kappa@0.10, kappa_ts@0.10) for allocation summaries alpha/pS/X (arrays)."""
    MSS, diag_t = M[1][1], M[0][0] + M[2][2]
    po = (diag_t + alpha + (MSS - pS)) / n
    cL, cS, cR = t[L] + X, t[S] - m, t[R] + (m - X)
    pe = (a[L] * cL + a[S] * cS + a[R] * cR) / (n * n)
    k10 = np.where(np.abs(1.0 - pe) < 1e-9, np.nan, (po - pe) / (1.0 - pe))
    nt = n - (MSS - pS)
    po_t = (diag_t + alpha) / nt
    msubS = a[S] - MSS + pS
    tsubS = (t[S] - MSS) - (m - pS)
    pe_t = (a[L] * cL + msubS * tsubS + a[R] * cR) / (nt * nt)
    kt10 = np.where(np.abs(1.0 - pe_t) < 1e-9, np.nan,
                    (po_t - pe_t) / (1.0 - pe_t))
    return k10, kt10


def move_envelope_slow(M, n, m, anchor=None):
    a = [sum(M[i]) for i in range(3)]
    t = [sum(M[i][j] for i in range(3)) for j in range(3)]
    PL, PS, PR = M[0][1], M[1][1], M[2][1]
    if m > PL + PS + PR:
        return None
    kmin = kmax = None
    for pL in range(0, min(m, PL) + 1):
        for pR in range(0, min(m - pL, PR) + 1):
            pS = m - pL - pR
            if pS > PS:
                continue
            xL = np.arange(pL + 1).reshape(-1, 1, 1)
            yR = np.arange(pR + 1).reshape(1, -1, 1)
            xS = np.arange(pS + 1).reshape(1, 1, -1)
            alpha = xL + yR
            X = xL + xS + (pR - yR)               # movers to L
            k10, kt10 = _k_pair_arrays(n, a, t, M, alpha, pS, X, m)
            ok = ~np.isnan(kt10)
            if anchor is not None:
                ok &= ~np.isnan(k10) & (np.abs(k10 - anchor) <= TOL)
            if not ok.any():
                continue
            v = kt10[ok]
            lo, hi = float(v.min()), float(v.max())
            kmin = lo if kmin is None else min(kmin, lo)
            kmax = hi if kmax is None else max(kmax, hi)
    return None if kmin is None else (kmin, kmax)


# --------------------------------------------------------------------------- #
# 2b. FAST exact move envelope — reachable (alpha, X) per (pL, pR), union caps #
# --------------------------------------------------------------------------- #
def move_envelope_fast(n, a, t, MSS, diag_t, m01_set, tS_sub, PS, m,
                       anchor=None):
    """Envelope over the UNION of admissible matrices sharing (MSS, diag_t) and
    marginals, whose traj-straight split is (m01, MSS, tS_sub - m01), m01 in
    m01_set. Exact: a (pL, pR) pair is feasible iff some m01 admits it."""
    m01s = sorted(m01_set)
    M_fake = ((0, 0, 0), (0, MSS, 0), (0, 0, 0))  # only MSS/diag_t used below
    kmin = kmax = None

    def upd(k10, kt10, mask):
        nonlocal kmin, kmax
        if anchor is not None:
            mask &= ~np.isnan(k10) & (np.abs(k10 - anchor) <= TOL)
        mask &= ~np.isnan(kt10)
        if not mask.any():
            return
        v = kt10[mask]
        lo, hi = float(v.min()), float(v.max())
        kmin = lo if kmin is None else min(kmin, lo)
        kmax = hi if kmax is None else max(kmax, hi)

    for pL in range(0, min(m, m01s[-1]) + 1):
        for pR in range(0, min(m - pL, tS_sub - m01s[0]) + 1):
            # pair feasible iff exists m01 with m01 >= pL and tS_sub-m01 >= pR
            i = bisect.bisect_left(m01s, pL)
            if i == len(m01s) or m01s[i] > tS_sub - pR:
                continue
            pS = m - pL - pR
            if pS > PS:
                continue
            alphas = np.arange(pL + pR + 1).reshape(-1, 1)
            xLmin = np.maximum(0, alphas - pR)
            xLmax = np.minimum(pL, alphas)
            Xlo = 2 * xLmin - alphas + pR          # + xS = 0
            Xhi = 2 * xLmax - alphas + pR + pS
            Xcols = np.arange(int(Xlo.min()), int(Xhi.max()) + 1).reshape(1, -1)
            mask = (Xcols >= Xlo) & (Xcols <= Xhi)
            if pS == 0:                            # parity: X steps by 2
                mask &= ((Xcols - Xlo) % 2) == 0
            # class-sheet fake matrix carries MSS and diag_t via closure:
            Mc = ((diag_t, 0, 0), (0, MSS, 0), (0, 0, 0))
            k10, kt10 = _k_pair_arrays(n, a, t, Mc, alphas, pS, Xcols, m)
            upd(k10, kt10, mask)
    return None if kmin is None else (kmin, kmax)


# --------------------------------------------------------------------------- #
# 3. panel-level envelope                                                      #
# --------------------------------------------------------------------------- #
def panel_envelope(n, a, t, n_turn, kappa_pub, kts_pub, m_max, anchor=None,
                   force_slow=False):
    mats = enumerate_admissible(n, a, t, n_turn, kappa_pub, kts_pub)
    if not mats:
        return {"error": "no admissible matrix — banked numbers inconsistent "
                         "under this tool's constraint model"}
    MSS = n - n_turn
    diag_ts = {M[0][0] + M[2][2] for M in mats}
    fast_ok = (len(diag_ts) == 1) and not force_slow
    m01_set = {M[0][1] for M in mats}
    tS_sub = t[S] - MSS
    per_m = {}
    for m in range(0, m_max + 1):
        if fast_ok:
            r = move_envelope_fast(n, a, t, MSS, next(iter(diag_ts)),
                                   m01_set, tS_sub, MSS, m, anchor)
        else:
            r, lo, hi = None, None, None
            for M in mats:
                s = move_envelope_slow(M, n, m, anchor)
                if s:
                    lo = s[0] if lo is None else min(lo, s[0])
                    hi = s[1] if hi is None else max(hi, s[1])
            r = None if lo is None else (lo, hi)
        if r is not None:
            per_m[m] = {"kts10_min": round(r[0], 4),
                        "kts10_max": round(r[1], 4),
                        "bands": sorted({band(r[0]), band(r[1])})}
    return {"n_admissible_matrices": len(mats),
            "diag_turn_sums": sorted(diag_ts),
            "m01_range": [min(m01_set), max(m01_set)],
            "fast_path": fast_ok,
            "example_M_at_015": [list(r) for r in mats[0]],
            "per_m": per_m}


# --------------------------------------------------------------------------- #
# 4. self-test — brute-force truth vs summary-only envelope, fast vs slow      #
# --------------------------------------------------------------------------- #
def _classify(net, g):
    return np.where(net > g, L, np.where(net < -g, R, S))


def _kappa_arrays(x, y, mask=None, raw=False):
    """Verbatim replica of hierarchy._kappa (:517-531) semantics.

    ``raw=True`` skips the final 4-dp rounding — used ONLY for the self-test's
    containment truth, because the envelope bounds are unrounded and a rounded
    truth can sit up to half an ULP outside them (measured: case 0,
    -0.0125 vs hi -0.0125427). Published/constraint inputs stay rounded,
    matching the real panels."""
    if mask is not None:
        x, y = x[mask], y[mask]
    labs = sorted(set(x.tolist()) | set(y.tolist()))
    po = float((x == y).mean())
    pe = sum(float((x == c).mean()) * float((y == c).mean()) for c in labs)
    if abs(1.0 - pe) < 1e-9:
        return None
    k = (po - pe) / (1.0 - pe)
    return k if raw else round(k, 4)


def selftest(n_cases=200, seed=0):
    ran = skipped = fast_checked = 0
    for case in range(n_cases):
        npr = np.random.RandomState(seed * 1000 + case)
        n = int(npr.randint(60, 220))
        man = npr.choice([L, S, R], size=n, p=npr.dirichlet([1.2, 6.0, 1.2]))
        pull = np.where(man == L, 0.12, np.where(man == R, -0.12, 0.0))
        net = npr.normal(0, 0.05, size=n) + pull * npr.binomial(1, 0.7, size=n)
        tail = npr.rand(n) < 0.15
        net[tail] += (npr.choice([-1, 1], size=int(tail.sum()))
                      * npr.uniform(0.05, 0.5, size=int(tail.sum())))
        traj15, traj10 = _classify(net, 0.15), _classify(net, 0.10)
        turn15 = (man != S) | (traj15 != S)
        k15 = _kappa_arrays(man, traj15)
        kt15 = _kappa_arrays(man, traj15, turn15)
        if k15 is None or kt15 is None:
            skipped += 1
            continue
        a = [int((man == c).sum()) for c in (L, S, R)]
        t = [int((traj15 == c).sum()) for c in (L, S, R)]
        n_turn = int(turn15.sum())
        turn10 = (man != S) | (traj10 != S)
        k10 = _kappa_arrays(man, traj10)                      # rounded: anchor
        kt10 = _kappa_arrays(man, traj10, turn10, raw=True)   # unrounded: truth
        m_true = int(((np.abs(net) > 0.10) & (np.abs(net) <= 0.15)).sum())
        Mtrue = tuple(tuple(int(((man == i) & (traj15 == j)).sum())
                            for j in (L, S, R)) for i in (L, S, R))
        mats = enumerate_admissible(n, a, t, n_turn, k15, kt15)
        assert Mtrue in [tuple(tuple(r) for r in M) for M in mats], \
            f"case {case}: true matrix not admissible"
        if kt10 is None:
            ran += 1
            continue
        for anchor in (None, k10):
            lo = hi = None
            for M in mats:
                s = move_envelope_slow(M, n, m_true, anchor)
                if s:
                    lo = s[0] if lo is None else min(lo, s[0])
                    hi = s[1] if hi is None else max(hi, s[1])
            if anchor is not None and lo is None:
                raise AssertionError(f"case {case}: anchored set empty at true m")
            if lo is not None:
                assert lo - 1e-9 <= kt10 <= hi + 1e-9, \
                    f"case {case}: kt10 {kt10} outside [{lo},{hi}] anchor={anchor}"
            # fast path must agree with slow path when its condition holds
            diag_ts = {M[0][0] + M[2][2] for M in mats}
            if len(diag_ts) == 1:
                f = move_envelope_fast(n, a, t, n - n_turn, next(iter(diag_ts)),
                                       {M[0][1] for M in mats}, t[S] - (n - n_turn),
                                       n - n_turn, m_true, anchor)
                assert (f is None) == (lo is None), f"case {case}: fast/slow feas"
                if f is not None:
                    assert abs(f[0] - lo) < 1e-9 and abs(f[1] - hi) < 1e-9, \
                        f"case {case}: fast {f} != slow ({lo},{hi})"
                    fast_checked += 1
        ran += 1
    return {"cases_run": ran, "cases_skipped_degenerate": skipped,
            "fast_vs_slow_checks": fast_checked, "pass": True}


# --------------------------------------------------------------------------- #
# 5. the banked panels                                                         #
# --------------------------------------------------------------------------- #
def _marg(dist):
    return [int(dist[k]) for k in DIRK]


def load_panel(path, xz=False):
    op = lzma.open if xz else open
    with op(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    c = d["consistency"]
    mvt = c["maneuver_vs_trajectory"]
    agr = mvt["agreement"]
    return {
        "n": int(d["n_windows"]),
        "man": _marg(c["distributions"]["maneuver_dir"]),
        "traj": _marg(c["distributions"]["trajectory_dir"]),
        "kappa": float(mvt["kappa"]),
        "kts": float(mvt["kappa_turn_subset"]),
        "n_turn": int(mvt["n_turn_active"]),
        "agreement_field": agr if isinstance(agr, (int, float)) else dict(agr),
        "agreement_is_legacy_heldout": not (
            isinstance(agr, dict) and agr.get("estimator")
            == "episode_cluster_bootstrap"),
        "ckpt_step": d.get("ckpt_step"),
    }


PANELS = [
    dict(key="v1arch-gateswept (OOD-val q90) — THE QUOTED 0.2005",
         path="TanitAD Research Hub/Architecture & Inference/Implementation/"
              "incoming/2026-08-06-v1-defect-triage/results/"
              "hier_v1arch_gateswept.json.xz",
         xz=True, anchor_kappa_010=0.5715, m_max=120),
    dict(key="flagship-30k (canonical val, deployed v1 FINAL)",
         path="stack/experiments/pod-rescue-20260802/pod3/root/taniteval/"
              "results/hier_flagship-30k.json", m_max=88),
    dict(key="flagship-speed (canonical val, 19k relay)",
         path="stack/experiments/pod-rescue-20260802/pod3/root/taniteval/"
              "results/hier_flagship-speed.json", m_max=88),
    dict(key="v1-lf19 (OOD-val leak-free 19)",
         path="TanitAD Research Hub/Evaluation/Implementation/incoming/"
              "2026-08-02-four-family-panel/hier_v1-lf19.json", m_max=42),
    dict(key="v2corpus-lf19 (OOD-val leak-free 19)",
         path="TanitAD Research Hub/Evaluation/Implementation/incoming/"
              "2026-08-02-four-family-panel/hier_v2corpus-lf19.json", m_max=42),
]


def main():
    print("[selftest] running — no real number is emitted unless it passes",
          flush=True)
    st = selftest()
    print(f"[selftest] PASS — {st['cases_run']} cases, "
          f"{st['fast_vs_slow_checks']} fast-vs-slow agreements "
          f"({st['cases_skipped_degenerate']} degenerate skipped)", flush=True)

    out = {"_what": ("kappa_turn_subset @ gate 0.10: exact admissible-set "
                     "envelope from banked panel summaries. Bounds are "
                     "DETERMINISTIC, not confidence intervals."),
           "_tier": ("open-loop model-internal coherence diagnostic "
                     "(T0-family); NOT driving performance"),
           "_ladder": "four_families.KAPPA_VERDICT_LADDER: <0.1 DECORATIVE, "
                      "<0.4 WEAK, >=0.4 SUBSTANTIAL (0.2 retired 2026-08-16)",
           "_prereg": "../PREREG.md — written before this ran",
           "selftest": st, "panels": {}}
    for spec in PANELS:
        p = load_panel(os.path.join(CLONE, spec["path"]), spec.get("xz", False))
        anchor = spec.get("anchor_kappa_010")
        env = panel_envelope(p["n"], p["man"], p["traj"], p["n_turn"],
                             p["kappa"], p["kts"], spec["m_max"], anchor)
        pe = sum(p["man"][i] * p["traj"][i] for i in range(3)) / p["n"] ** 2
        po_impl = p["kappa"] * (1 - pe) + pe
        agr_mean = (p["agreement_field"]
                    if isinstance(p["agreement_field"], (int, float))
                    else p["agreement_field"].get("mean"))
        rec = {"po_implied_by_kappa": round(po_impl, 6),
               "T_implied": round(po_impl * p["n"], 2),
               "agreement_field_mean": agr_mean,
               "agreement_is_legacy_heldout_shape":
                   p["agreement_is_legacy_heldout"],
               "agreement_matches_full_set_po":
                   bool(agr_mean is not None
                        and abs(agr_mean - po_impl) <= 1.5e-4)}
        blk = {"inputs": {k: p[k] for k in ("n", "man", "traj", "kappa", "kts",
                                            "n_turn", "ckpt_step")},
               "anchor_kappa_010": anchor, "reconciliation": rec,
               "envelope": env}
        out["panels"][spec["key"]] = blk
        if "error" in env:
            print(f"[panel] {spec['key']}: {env['error']}", flush=True)
            continue
        pm = env["per_m"]
        feas = sorted(int(k) for k in pm)
        glo = min(pm[m]["kts10_min"] for m in feas)
        ghi = max(pm[m]["kts10_max"] for m in feas)
        blk["summary"] = {
            "m_feasible": [feas[0], feas[-1]],
            "kts10_global": [glo, ghi],
            "bands_reachable": sorted({b for m in feas for b in pm[m]["bands"]}),
            "kts15_published": p["kts"], "band_at_015": band(p["kts"]),
        }
        print(f"[panel] {spec['key']}: {env['n_admissible_matrices']} matrices "
              f"(fast={env['fast_path']}); m {feas[0]}..{feas[-1]}; kts@0.10 in "
              f"[{glo:.4f}, {ghi:.4f}]; bands {blk['summary']['bands_reachable']}",
              flush=True)

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "raw", "gate_ts_envelope.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {os.path.normpath(dst)}", flush=True)


if __name__ == "__main__":
    main()
