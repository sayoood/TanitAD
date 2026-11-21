#!/usr/bin/env python3
"""P6-HN-S: the SCREEN READ.  (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md sections 5-11; SCREEN_NOTES.md checklist item 7.)  CPU only, a table lookup: no model runs here.

    python eval/hn_screen_read.py --scores base=base.npz A0=... A1=... A2=... B0=... B1=... B2=... [A3= A4= B3= B4=] [C0=c0.npz] --pdm <pdm dir> --rl-val-scenes rl_val_scenes.tsv \\
        --sets <rl_fan sets dir> --gates gates.json --chain-root <the chain's state dir> [--key-file <gate key>] [--scores9 base=... A0=... ...] [--bank9 <rl_val.hnbank>] [--families families.json --families-per-set families_per_set.npz] --rule navsim_v1_lane --out result.json

CACHE-BINDING GATE (HNSCREEN-FP-COST, MM ruling (b')).  BEFORE any score file is opened, `--chain-root` is checked (`gate_token.ctx_chain_refusals`): every pass named in --scores (A<s> / B<s>) must carry a signed PASS token whose --ctx-cache
is bound by CONTENT (every gate call re-read the cache), or by the chain's START ANCHOR -- and then the chain's `ctx_chain/end_verify.json` must be a signed END_VERIFY_OK bound to THE SAME anchor, with equal content and stat comparisons and every
seed covered.  No token, no record, a deleted / VOID / foreign / edited record, two anchors, or a listing-only token: the read REFUSES (rc 6, `ZZHNSCREEN_REFUSED`, nothing is read and no result is written).

It REUSES rl_val_eval's lookup (`load_scores`, `load_pdm`, `read_scenes`, `select`, the reward module's PDMS) for base / A_s / B_s, and `zt4_baseline.twin_pairs` unchanged for M2.
ESTIMATOR (section 6): paired log-cluster bootstrap over the RL-val logs, 10,000 resamples, seed 20260927, percentile 2.5 / 97.5, ratio-of-sums.  ONE log draw per resample is applied JOINTLY to every
seed and arm (`JointBoot`: the same draws proxy_eval.boot makes, generated once).  A seed-mean's interval therefore answers 'another draw of LOGS?' for the mean over seeds; the TRAINING-seed question is
answered by the sign rule (Z2) and the floor (Z3), the INFERENCE question by gate I5.  Tier: SELECTION (a lookup of the NAVSIM v1.1 PDMS on navtrain RL-val), never an endpoint.

PRIMARY ENDPOINT (section 7).  z(X) = share of RL-val sets whose pick by X has true PDMS = 0, in pp.  Delta z_s = z(B_s) - z(A_s) (paired on the same sets); Delta z-bar = mean over the seeds; F_z = max_s z(A_s) - min_s z(A_s).
  ZEROS-FALL iff Z1 UB(Delta z-bar) < 0  AND  Z2 Delta z_s < 0 for EVERY seed  AND  Z3 -Delta z-bar > F_z.   Z2 and Z3 are judged on the integer pick COUNTS (no float rounding at a boundary).
  RULE E (pre-committed, used at most once): Z1 holds at seeds 0-2 but Z2 or Z3 fails -> seeds 3 and 4 are required; over 5 seeds: Z1 on the 5-seed mean, Z2' at least 4 of 5 Delta z_s < 0, Z3 with F_z over the 5 controls.
  Holds -> ZEROS-FALL, still fails -> SEED-FRAGILE.  If Z1 fails at 3 seeds there is no extension.  Seeds 3 and 4, if supplied but not needed, are IGNORED (and said so).
GUARD: Delta PDMS-bar > 0 (point) is required for GO.  CO-GATE (section 8) FAILS iff UB(Delta NC-bar) < 0 or UB(Delta TTC-bar) < 0 or, for NC or for TTC, at least 2 of the seeds individually read UB < 0.
MECHANISM (section 9, they choose the next lever, never the GO): M1 zero split; M2 the DAC head on REAL grazes (HEAD-MOVES iff LB(Delta M2-bar) > 0); M3 the head on the BANK's geometry, held out (LEARNS-BANK iff LB(Delta M3-bar) > 0;
  NOT COMPUTABLE below 100 qualifying scenes -> rows 8 and 9 merge as 'HEAD-BLIND OR SYNTHETIC-ONLY, unresolved').
IC (section 5): every two-arm difference computed as an arm minus itself reads exactly 0 with CI [0, 0] -- through the SAME functions the real differences use.
FAMILIES (section 10 + amendment B4): the trajectory-level LONGITUDINAL / LATERAL components and the TACTICAL decisions are PAIRED like the PDM sub-scores.  `--families-per-set` is the per-set export of the families instrument
  (hn_screen_families.py -> taniteval/per_window.py).  It is aligned to the PDM rows by (log, token) -- refused loudly if the two do not cover the same sets -- and every component's seed-mean B_s - A_s difference is read on the
  SAME joint log draws (`JointBoot`, 10,000, seed 20260927) as the PDM sub-scores: means, rms and the masked components' ratios-of-sums are re-taken on each draw, the TACTICAL kappa from the draw's confusion table.  A component
  separates ADVERSELY by the landed direction table (`proxy_eval.fam_direction`) read on the paired interval: higher-is-better -> UB < 0; lower-is-better -> LB > 0; bias -> the |.| of B exceeds A's (LB of |B| - |A| > 0);
  progress ratio -> |ratio - 1| likewise.  REPORTED, NOT GATING: a LONGITUDINAL or LATERAL adverse component is a named WATCH-ITEM for the full arm's amendment; nothing here changes the outcome row.  Properties of the logged human
  future (`ego_progress.gt_progress_mean_m`, `.t0_axis_gt_self_ratio`) are excluded (identical for every arm).  The families are PUBLISHED only if the export paired AND the family IC (an arm minus itself reads 0, [0, 0]) holds.
OUTCOME (section 11): the first matching row of the table below.  STRATEGIC is PAUSED (PI 2026-09-19, PI_DECISION_QUEUE.md item 25): stamped, never a pass, never omitted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for _p in (HERE, os.path.join(PKG, "refe")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import rl_val_eval as RV  # noqa: E402  -- the lookup this read reuses
rl_reward = RV.rl_reward

TOOL = "hn_screen_read/1"
N_BOOT, SEED = 10000, 20260927
SY_MIN_SHARE = 0.10
M3_MIN_SCENES = 100
SEEDS3, SEEDS_EXT = (0, 1, 2), (3, 4)
GATE_KEYS = ("I1", "I2", "I3", "I4", "I5", "I6", "D1", "G14", "G15", "launch_gate")           # IC is computed here, not supplied
ZT4_BAR, U400_NAVTEST_M2 = 0.70, 0.392
PAUSED = {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md item 25", "note": "reported as PAUSED: not a pass, not omitted"}

OUTCOMES = {
    0: ("SCREEN-INVALID", "not decided", "fix the failed control, re-run that stage; nothing is read"),
    1: ("SCREEN-SPARSE", "NO-GO (no supply)", "the draft's SPARSE lever: pair-contrast on the fan's own discordant twins, under its own prereg, on this rig"),
    2: ("SCREEN-UNSAFE", "NO-GO as specified", "(i) 0-GPU attribution of changed picks by the sub-score that moved, from the dumps. (ii) Arm S-DAC, pre-registered in the screen prereg: the same screen with a DAC-only bank "
                                              "(`--subtypes 1`), reusing the cache and labels, 3 passes; an S-DAC GO => the full arm with a DAC-only bank, by amendment before its launch"),
    3: ("SCREEN-GO", "GO (meets the PI's 2026-10-18 condition; launch still needs the draft's other preconditions and its registration)",
        "amend the draft (dated, before any P6-HN step) with the masked form, the screen's F_z, M2 and M3 and the watch-items; then the 0-GPU selection-rule read on B's dumps"),
    4: ("NO-GO / NET-ZERO", "NO-GO as specified", "0-GPU EP-versus-safety calibration of the selection rule on B's dumps, cross-fitted over 5 log folds of RL-val; a cross-fitted net > 0 goes to the PI as a re-proposal with that rule"),
    5: ("NO-GO / SEED-FRAGILE", "NO-GO", "the head-level effect does not survive training variance; next: row 8's lever if LEARNS-BANK, else row 9's"),
    6: ("NO-GO / REVERSED", "NO-GO", "0-GPU attribution by |delta|, side and sub-type of the offsets whose labels push picks into zeros; then a label audit for tracker-made zeros on offsets"),
    7: ("NO-GO / HEAD-MOVES-NO-ZEROS", "NO-GO", "0-GPU DAC-veto read with B's head (the SELRULE V(tau) instrument) on B's dumps, cross-fitted; if it reaches zeros, re-propose"),
    8: ("NO-GO / SYNTHETIC-ONLY", "NO-GO", "pair-contrast on the fan's own discordant near-twins as the next screen on this rig and cache; it needs no new labels"),
    9: ("NO-GO / HEAD-BLIND", "NO-GO", "representational at the frozen trunk: next P7.1 (the map-supervised drivable-area aux head on the trunk); the frozen-P6 re-screen runs iff P6's ZT4 reads HEAD-MOVES against u400"),
    "8or9": ("NO-GO / HEAD-BLIND OR SYNTHETIC-ONLY, unresolved", "NO-GO", "M3 is NOT COMPUTABLE (fewer than 100 qualifying RL-val bank scenes): rows 8 and 9 merge; run the pair-contrast screen (row 8's lever) and P7.1 (row 9's)"),
}


# ================================================================================================ the joint bootstrap
class JointBoot:
    """ONE set of log resamples for everything.  Groups are the sorted unique log names; the draws are proxy_eval.boot's (`rng.integers(0, G, G)` per resample from default_rng(seed))."""

    def __init__(self, logs, n=N_BOOT, seed=SEED):
        names = sorted(set(logs))
        idx = {nm: i for i, nm in enumerate(names)}
        self.li = np.asarray([idx[x] for x in logs])
        self.G, self.n, self.seed = len(names), int(n), int(seed)
        rng = np.random.default_rng(self.seed)
        counts = np.empty((self.n, self.G))
        for b in range(self.n):
            counts[b] = np.bincount(rng.integers(0, self.G, self.G), minlength=self.G)
        self.counts = counts
        self.size = np.bincount(self.li, minlength=self.G).astype(np.float64)

    def means(self, v):
        """the resampled means of a per-set vector, [n]"""
        S = np.bincount(self.li, np.asarray(v, np.float64), self.G)
        return (self.counts @ S) / (self.counts @ self.size)

    def medians(self, v, rows):
        """the resampled medians of a per-set vector defined on `rows` only: the median of the EXPANDED resample (each log drawn k times contributes its sets k times), as the single-arm bootstrap takes it.
        NaN for a draw with no row at all."""
        v, rows = np.asarray(v, np.float64), np.asarray(rows, np.int64)
        order = np.argsort(v, kind="stable")
        vs, li = v[order], self.li[rows][order]
        out = np.full(self.n, np.nan)
        if vs.size == 0:
            return out
        for a in range(0, self.n, 500):
            cum = np.cumsum(self.counts[a:a + 500][:, li], axis=1)                  # [chunk, R] cumulative multiplicity in value order
            tot = cum[:, -1]
            i_lo = (cum <= np.floor((tot - 1) / 2)[:, None]).sum(1)                 # 0-based rank r sits at the first index whose cumulative count exceeds r
            i_hi = (cum <= np.floor(tot / 2)[:, None]).sum(1)
            m = tot > 0
            out[a:a + 500][m] = 0.5 * (vs[np.minimum(i_lo, vs.size - 1)][m] + vs[np.minimum(i_hi, vs.size - 1)][m])
        return out

    def ratios(self, num, den):
        Sn = np.bincount(self.li, np.asarray(num, np.float64), self.G)
        Sd = np.bincount(self.li, np.asarray(den, np.float64), self.G)
        d = self.counts @ Sd
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.counts @ Sn) / d

    @staticmethod
    def _ci(r):
        return [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]

    def mean_interval(self, v):
        v = np.asarray(v, np.float64)
        return {"mean": float(v.mean()), "ci95": self._ci(self.means(v))}

    def ratio_interval(self, num, den):
        num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
        return {"mean": float(num.sum() / den.sum()) if den.sum() > 0 else float("nan"), "ci95": self._ci(self.ratios(num, den)), "n_sets": int((den > 0).sum()), "n_pairs": float(den.sum())}


# ================================================================================================ the lookup (rl_val_eval's)
def pick_index(scores, rule, space="logit"):
    """the pick of one set: rl_val_eval.select, i.e. the highest selection score, TIES TO THE LOWEST INDEX"""
    return RV.select(scores, rule, space)[0]


def make_data(keys, logs, tg, arms, plans8=None, bank=None, arms9=None, space="logit"):
    """in-memory data: keys [N] (log, token, step, rank), logs [N] cluster names, tg [N, M, 6] PDM labels (nc, dac, ep, ttc, c, ddc), arms {name: scores [N, M, C]},
    plans8 [N, M, 8, 3] executed NAVSIM-grid plans (for M2), bank {'dac9': [N9, 9], 'status': [N9]} (the RL-val bank's DAC labels; slot 0 = the anchor) and arms9 {name: scores9 [N9, 9, C]} (for M3)."""
    D = {"keys": list(keys), "logs": [str(x) for x in logs], "tg": np.asarray(tg, np.float64), "arms": {n: np.asarray(s, np.float64) for n, s in arms.items()}, "space": space,
         "plans8": None if plans8 is None else np.asarray(plans8, np.float64), "bank": bank, "arms9": None if arms9 is None else {n: np.asarray(s, np.float64) for n, s in arms9.items()}}
    N = len(D["keys"])
    assert D["tg"].shape[0] == N and len(D["logs"]) == N and all(s.shape[0] == N for s in D["arms"].values()), "arrays disagree on the number of sets"
    return D


def lookup(D, name, rule):
    """per-set quantities of one scorer's pick: PDMS, the six sub-scores, zero and its split, the fan's random / best (rl_val_eval.eval_set's definitions)"""
    sc, tg = D["arms"][name], D["tg"]
    N = len(D["keys"])
    out = {k: np.zeros(N) for k in ("PDMS", "NC", "DAC", "EP", "TTC", "C", "DDC", "zero", "rand", "best", "nc0", "dac0", "zero_dac_only", "zero_nc_only", "zero_both", "zero_other")}
    pick = np.zeros(N, np.int64)
    for i in range(N):
        sp = rl_reward.split_components(tg[i])
        pdms = rl_reward.pdms_v1(sp["nc"], sp["dac"], sp["ep"], sp["ttc"], sp["c"])
        j = pick_index(sc[i], rule, D["space"])
        pick[i] = j
        for k, key in (("NC", "nc"), ("DAC", "dac"), ("EP", "ep"), ("TTC", "ttc"), ("C", "c"), ("DDC", "ddc")):
            out[k][i] = sp[key][j]
        out["PDMS"][i] = pdms[j]
        out["rand"][i], out["best"][i] = pdms.mean(), pdms.max()
        z = pdms[j] == 0.0
        nc0, dac0 = sp["nc"][j] == 0.0, sp["dac"][j] == 0.0
        out["zero"][i] = float(z)
        out["nc0"][i], out["dac0"][i] = float(nc0), float(dac0)
        out["zero_dac_only"][i] = float(z and dac0 and not nc0)
        out["zero_nc_only"][i] = float(z and nc0 and not dac0)
        out["zero_both"][i] = float(z and nc0 and dac0)
        out["zero_other"][i] = float(z and not nc0 and not dac0)
    out["pick"] = pick
    return out


def twin_stats(D, name, rule):
    """per set: (n twin pairs, n correct) of the arm's DAC head on REAL near-twins, `zt4_baseline.twin_pairs` unchanged (max over 8 poses <= 0.5 m and DAC_i != DAC_j on the executed plans)"""
    zt4 = _zt4()
    sc, tg, pl = D["arms"][name], D["tg"], D["plans8"]
    N = len(D["keys"])
    npair, ncorr = np.zeros(N), np.zeros(N)
    for i in range(N):
        sp = rl_reward.split_components(tg[i])
        pdac = sc[i][:, 1] if D["space"] == "prob" else RV.sigmoid(sc[i][:, 1])
        npair[i], ncorr[i] = zt4.twin_pairs(pl[i], np.asarray(sp["dac"]), pdac)
    return npair, ncorr


def _zt4():
    """the landed zt4_baseline module (twin_pairs is read from it, never copied)"""
    d = os.path.join(PKG, "raw", "2026-10-17-zero-tail-mechanism", "code")
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("zt4_baseline_landed", os.path.join(d, "zt4_baseline.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bank_pairs(D, name):
    """M3 per scene: pairwise DAC accuracy of the arm's head among the bank's 9 trajectories (anchor + 8 offsets, each scored as a 65th candidate): pairs between DAC = 1 members
    (anchor or offsets) and DAC = 0 OFFSETS; ties count 0.5.  Scenes lacking either side have zero pairs and are counted by the caller."""
    dac9 = np.asarray(D["bank"]["dac9"], np.float64)
    lab = np.asarray(D["bank"]["status"]) == 0
    s9 = D["arms9"][name]
    pd = s9[..., 1] if D["space"] == "prob" else RV.sigmoid(s9[..., 1])
    N9 = dac9.shape[0]
    npair, ncorr = np.zeros(N9), np.zeros(N9)
    for i in range(N9):
        if not lab[i]:
            continue
        one = [j for j in range(9) if dac9[i, j] == 1.0]
        zero = [j for j in range(1, 9) if dac9[i, j] == 0.0]
        for a in one:
            for b in zero:
                npair[i] += 1
                ncorr[i] += 1.0 if pd[i, a] > pd[i, b] else (0.5 if pd[i, a] == pd[i, b] else 0.0)
    return npair, ncorr


# ================================================================================================ the statistics (each used for the real difference AND for IC)
def seedmean_diff(JB, vb, va, scale=100.0):
    """mean over seeds of (B_s - A_s), per-set, then the joint-draw interval.  vb, va: lists (one per seed) of per-set vectors."""
    d = scale * np.mean([np.asarray(b, np.float64) - np.asarray(a, np.float64) for b, a in zip(vb, va)], axis=0)
    r = JB.mean_interval(d)
    r["per_seed_point"] = [float(scale * np.mean(np.asarray(b, np.float64) - np.asarray(a, np.float64))) for b, a in zip(vb, va)]
    r["per_seed_ci95"] = [JB.mean_interval(scale * (np.asarray(b, np.float64) - np.asarray(a, np.float64)))["ci95"] for b, a in zip(vb, va)]
    return r


def ratio_diff(JB, nb, na, den):
    """seed-mean of the ratio-of-sums difference: sum_i mean_s(ncorr_B - ncorr_A) / sum_i npairs, on the sets that have a pair"""
    num = np.mean([np.asarray(b, np.float64) - np.asarray(a, np.float64) for b, a in zip(nb, na)], axis=0)
    r = JB.ratio_interval(num, den)
    r["per_seed_point"] = [float((np.asarray(b) - np.asarray(a)).sum() / np.asarray(den).sum()) if np.asarray(den).sum() > 0 else float("nan") for b, a in zip(nb, na)]
    return r


def zeros_endpoint(JB, zA, zB):
    """zA, zB: lists (one per seed) of per-set 0/1 zero-pick indicators.  Everything exact is judged on integer counts."""
    N = len(zA[0])
    nA = [int(round(float(np.sum(z)))) for z in zA]
    nB = [int(round(float(np.sum(z)))) for z in zB]
    S = len(nA)
    dsum = sum(b - a for a, b in zip(nA, nB))                              # integer: sum_s (nB_s - nA_s)
    spread = max(nA) - min(nA)
    iv = seedmean_diff(JB, zB, zA)
    return {"seeds": S, "N": N, "zero_picks_A": nA, "zero_picks_B": nB, "z_A_pp": [100.0 * n / N for n in nA], "z_B_pp": [100.0 * n / N for n in nB],
            "delta_z_s_pp": [100.0 * (b - a) / N for a, b in zip(nA, nB)], "delta_z_bar": iv, "F_z_pp": 100.0 * spread / N,
            "Z1": bool(iv["ci95"][1] < 0), "Z2": bool(all(b - a < 0 for a, b in zip(nA, nB))), "Z2_prime_ge4of5": bool(sum(b - a < 0 for a, b in zip(nA, nB)) >= 4),
            "Z3": bool(-dsum > S * spread),                                  # -delta-z-bar > F_z, exactly: -dsum/S > spread (counts)
            "LB_positive": bool(iv["ci95"][0] > 0)}


def decide_zeros(JB, V, seeds_given):
    """-> the primary-endpoint state.  V[name] = lookup dict.  Seeds 0-2 first; rule E only when Z1 holds and Z2 or Z3 fails; seeds 3-4 ignored (and said so) otherwise."""
    s3 = [s for s in SEEDS3 if f"A{s}" in V and f"B{s}" in V]
    if len(s3) < 3:
        return {"state": "INVALID_INPUT", "reason": "seeds 0, 1, 2 need both A_s and B_s", "zeros_fall": False, "rule_e_fired": False, "rule_e_required": False}
    e3 = zeros_endpoint(JB, [V[f"A{s}"]["zero"] for s in s3], [V[f"B{s}"]["zero"] for s in s3])
    res = {"seeds_used": s3, "primary_3seed": e3, "rule_e_fired": False, "rule_e_required": False, "extension_ignored": False}
    if e3["Z1"] and e3["Z2"] and e3["Z3"]:
        res.update(zeros_fall=True, state="ZEROS-FALL", final=e3)
        res["extension_ignored"] = any(f"A{s}" in V for s in SEEDS_EXT)
        return res
    if not e3["Z1"]:
        res.update(zeros_fall=False, state="Z1-FAILS-NO-EXTENSION", final=e3)
        return res
    ext = [s for s in SEEDS_EXT if f"A{s}" in V and f"B{s}" in V]
    if len(ext) < 2:
        res.update(zeros_fall=False, state="RULE_E_REQUIRED", rule_e_required=True, final=e3, reason="Z1 holds at seeds 0-2 but Z2 or Z3 fails: run seeds 3 and 4 with the identical argv apart from --seed")
        return res
    s5 = s3 + ext
    e5 = zeros_endpoint(JB, [V[f"A{s}"]["zero"] for s in s5], [V[f"B{s}"]["zero"] for s in s5])
    holds = bool(e5["Z1"] and e5["Z2_prime_ge4of5"] and e5["Z3"])
    res.update(rule_e_fired=True, primary_5seed=e5, zeros_fall=holds, state="ZEROS-FALL" if holds else "SEED-FRAGILE", final=e5, seeds_used=s5)
    return res


def cogate_decision(mean_ub, seed_ubs, k_seeds=2):
    """fails iff UB(mean) < 0, or at least `k_seeds` of the seeds individually read UB < 0"""
    n_adverse = int(sum(u < 0 for u in seed_ubs))
    return {"fails": bool(mean_ub < 0 or n_adverse >= k_seeds), "mean_ub": mean_ub, "seeds_adverse": n_adverse}


def cogate(JB, V, seeds):
    out = {}
    for q in ("NC", "TTC"):
        r = seedmean_diff(JB, [V[f"B{s}"][q] for s in seeds], [V[f"A{s}"][q] for s in seeds])
        r["decision"] = cogate_decision(r["ci95"][1], [c[1] for c in r["per_seed_ci95"]])
        out[q] = r
    out["fails"] = bool(out["NC"]["decision"]["fails"] or out["TTC"]["decision"]["fails"])
    out["reported_beside"] = {"DDC": seedmean_diff(JB, [V[f"B{s}"]["DDC"] for s in seeds], [V[f"A{s}"]["DDC"] for s in seeds]),
                              "EP": seedmean_diff(JB, [V[f"B{s}"]["EP"] for s in seeds], [V[f"A{s}"]["EP"] for s in seeds]),
                              "nc_zero_picks_per_seed": {f"{a}{s}": int(V[f"{a}{s}"]["nc0"].sum()) for s in seeds for a in "AB"}}
    return out


def classify(inp):
    """the section-11 table, first matching row wins.  inp: gates {name: bool}, sy_sparse, cogate_fails, zeros {state, zeros_fall, rule_e_required, rule_e_fired}, dpdms_point, lb_dz_positive,
    head_moves, learns_bank (True / False / None = NOT COMPUTABLE)."""
    g = inp["gates"]
    if any(v is False for v in g.values()):
        return 0
    if inp["sy_sparse"]:
        return 1
    if any(k not in g for k in GATE_KEYS + ("IC",)):
        return 0                                                              # a gate that was not run is not a pass
    if inp["cogate_fails"]:
        return 2
    z = inp["zeros"]
    if z["rule_e_required"]:
        return None                                                           # pending: seeds 3 and 4
    if z["zeros_fall"]:
        return 3 if inp["dpdms_point"] > 0 else 4
    if z["rule_e_fired"]:
        return 5
    if inp["lb_dz_positive"]:
        return 6
    if inp["head_moves"]:
        return 7
    if inp["learns_bank"] is None:
        return "8or9"
    return 8 if inp["learns_bank"] else 9


# ================================================================================================ the trajectory-level family components, PAIRED (section 10 + amendment B4)
class PerSetError(ValueError):
    """the per-set export cannot be paired with the PDM rows (different sets, no keys, duplicate keys)"""


#: tactical goal-setting has no entry in `proxy_eval.fam_direction` (the landed table names the LONGITUDINAL / LATERAL components); the directions are stated here, once
GOAL_DIRECTION = {"goal_point_error_m": "lower", "goal_long_bias_m": "magnitude", "goal_lat_bias_m": "magnitude", "goal_bearing_mae_deg": "lower", "goal_range_ratio": "closer_to_1"}
ADVERSE_TF = {"magnitude": np.abs, "closer_to_1": lambda x: np.abs(np.asarray(x, np.float64) - 1.0)}


def comp_draws(JB, c):
    """-> (point, draws [n_boot]) of one per-set component record on the JOINT log draws.  The reduction is the instrument's own: mean; rms = sqrt(mean of squares); ratio of sums (masked components); median."""
    k = c["kind"]
    if k == "mean":
        v = np.asarray(c["value"], np.float64)
        return float(v.mean()), JB.means(v)
    if k == "rms":
        v = np.asarray(c["value"], np.float64)
        return math.sqrt(float(v.mean())), np.sqrt(JB.means(v))
    if k == "ratio":
        num, den, sc = np.asarray(c["num"], np.float64), np.asarray(c["den"], np.float64), float(c["scale"])
        return (sc * float(num.sum()) / float(den.sum()) if den.sum() > 0 else float("nan")), sc * JB.ratios(num, den)
    if k == "median":
        v = np.asarray(c["value"], np.float64)
        return (float(np.median(v)) if v.size else float("nan")), JB.medians(v, c["rows"])
    raise ValueError(f"unknown component kind {k!r}")


def paired_from_draws(JB, pb, pa, direction):
    """pb, pa: per-seed lists of (point, draws) for B_s and A_s.  -> the seed-mean paired difference B - A on the joint draws, and its adverse reading by `direction`
    (higher -> UB < 0; lower -> LB > 0; magnitude / closer_to_1 -> the same on |B| - |A| / |B - 1| - |A - 1|, the paired form of proxy_eval.fam_adverse's 'farther' rules)."""
    if any(not (math.isfinite(b[0]) and math.isfinite(a[0])) for b, a in zip(pb, pa)):
        return {"status": "NOT COMPUTABLE", "reason": "a point estimate is undefined (empty denominator or an undefined kappa) in at least one arm"}
    pts = [b[0] - a[0] for b, a in zip(pb, pa)]
    drs = [b[1] - a[1] for b, a in zip(pb, pa)]
    out = {"status": "OK", "mean": float(np.mean(pts)), "ci95": JB._ci(np.mean(drs, axis=0)), "per_seed_point": [float(x) for x in pts], "per_seed_ci95": [JB._ci(d) for d in drs],
           "A_mean": float(np.mean([a[0] for a in pa])), "B_mean": float(np.mean([b[0] for b in pb])), "direction": direction}
    tf = ADVERSE_TF.get(direction)
    if tf is None:
        adv = out
    else:
        apts = [float(tf(b[0]) - tf(a[0])) for b, a in zip(pb, pa)]
        adr = [tf(b[1]) - tf(a[1]) for b, a in zip(pb, pa)]
        adv = out["adverse_stat"] = {"what": "|B| - |A| (bias)" if direction == "magnitude" else "|B - 1| - |A - 1|", "mean": float(np.mean(apts)), "ci95": JB._ci(np.mean(adr, axis=0)),
                                     "per_seed_point": apts, "per_seed_ci95": [JB._ci(d) for d in adr]}
    out["adverse"] = bool(adv["ci95"][1] < 0) if direction == "higher" else bool(adv["ci95"][0] > 0)
    return out


def decision_draws(JB, gt, pred, k):
    """(accuracy, kappa) as ((point, draws), (point, draws)) of one decision's executed class ids on the joint draws: the per-log confusion table, weighted by the draw's log multiplicities.  kappa is NaN where
    1 - p_e vanishes (both streams constant on one class): never a fake 1.0 (four_families._kappa_k)"""
    gt, pred = np.asarray(gt, np.int64), np.asarray(pred, np.int64)
    if gt.size and (gt.min() < 0 or pred.min() < 0 or gt.max() >= k or pred.max() >= k):
        raise PerSetError(f"a class id outside 0..{k - 1}")
    S = np.zeros((JB.G, k * k))
    np.add.at(S, (JB.li, gt * k + pred), 1.0)

    def ak(T):
        M = np.asarray(T, np.float64).reshape(T.shape[:-1] + (k, k))
        with np.errstate(invalid="ignore", divide="ignore"):
            tot = M.sum((-1, -2))
            po = np.trace(M, axis1=-2, axis2=-1) / tot
            pe = ((M.sum(-1) / tot[..., None]) * (M.sum(-2) / tot[..., None])).sum(-1)
            kap = np.where(1.0 - pe > 1e-9, (po - pe) / (1.0 - pe), np.nan)
        return po, kap

    a0, k0 = ak(S.sum(0))
    ad, kd = ak(JB.counts @ S)
    return (float(a0), ad), (float(k0), kd)


def _take(c, order, N):
    """one component record re-indexed to the PDM row order: new[i] = old[order[i]]"""
    c = dict(c)
    if c["kind"] in ("mean", "rms"):
        c["value"] = np.asarray(c["value"])[order]
    elif c["kind"] == "ratio":
        c["num"], c["den"] = np.asarray(c["num"])[order], np.asarray(c["den"])[order]
    elif c["kind"] == "median":
        full, has = np.full(N, np.nan), np.zeros(N, bool)
        full[np.asarray(c["rows"], np.int64)], has[np.asarray(c["rows"], np.int64)] = np.asarray(c["value"], np.float64), True
        c["rows"], c["value"] = np.flatnonzero(has[order]), full[order][has[order]]
    return c


def align_export(D, exp):
    """one arm's per-set export re-indexed to D's rows by (log, token).  Raises PerSetError unless the two cover EXACTLY the same sets (a positional join on different sets pairs unrelated windows)."""
    if exp.get("status") != "OK":
        raise PerSetError(f"the export is {exp.get('status')}: {exp.get('reason')}")
    toks, logs = exp.get("tokens"), exp.get("eid")
    if toks is None or logs is None:
        raise PerSetError("the export carries no (log, token) per row: it cannot be paired to the PDM rows")
    idx = {}
    for i, key in enumerate(zip(logs, toks)):
        if key in idx:
            raise PerSetError(f"duplicate (log, token) {key} in the export")
        idx[key] = i
    want = [(k[0], k[1]) for k in D["keys"]]
    if len(set(want)) != len(want):
        raise PerSetError("duplicate (log, token) among the PDM rows: the pairing key is ambiguous")
    miss = [w for w in want if w not in idx]
    if miss or len(idx) != len(want):
        raise PerSetError(f"the export covers {len(idx)} sets and the PDM rows {len(want)}; {len(miss)} PDM sets are absent from the export (first {miss[:2]}): not the same sets, not paired")
    order = np.asarray([idx[w] for w in want], np.int64)
    N = len(want)
    out = dict(exp)
    for fam in ("longitudinal", "lateral"):
        out[fam] = {nm: _take(c, order, N) for nm, c in exp[fam].items()}
    t = exp.get("tactical")
    if isinstance(t, dict) and t.get("status") == "OK":
        t = dict(t)
        for d in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
            t[d] = {**t[d], "gt": np.asarray(t[d]["gt"])[order], "pred": np.asarray(t[d]["pred"])[order]}
        t["goal_setting"] = {nm: _take(c, order, N) for nm, c in t["goal_setting"].items()}
        out["tactical"] = t
    return out


def families_paired(JB, D, per_set, seeds):
    """the seed-mean PAIRED differences of every trajectory-level component on the joint draws (+ the family IC).  Reported, not gating."""
    import proxy_eval as PE                                                                        # the landed direction table, imported never copied
    need = [f"{a}{s}" for s in seeds for a in "AB"]
    absent = [a for a in need if a not in per_set]
    if absent:
        return {"status": "NOT COMPUTABLE", "reason": f"the per-set export lacks {absent}"}
    try:
        ex = {a: align_export(D, per_set[a]) for a in need}
    except PerSetError as e:
        return {"status": "NOT PAIRABLE", "reason": str(e)}
    Bs, As = [ex[f"B{s}"] for s in seeds], [ex[f"A{s}"] for s in seeds]
    res = {"status": "OK", "seeds": list(seeds), "n_sets": len(D["keys"]), "estimator": "seed-mean paired B_s - A_s on the joint log draws of the PDM sub-scores (same JointBoot: draws, seed, percentile)",
           "gating": "REPORTED, NOT GATING (section 10); a LONGITUDINAL / LATERAL adverse component is a named watch-item", "LONGITUDINAL": {}, "LATERAL": {}, "unavailable": {}, "excluded_human_only": [],
           "STRATEGIC": PAUSED}
    ic = []
    for fam in ("longitudinal", "lateral"):
        names = set.intersection(*[set(e[fam]) for e in Bs + As])
        for nm in sorted(set.union(*[set(e[fam]) for e in Bs + As]) - names):
            res["unavailable"][f"{fam}.{nm}"] = {"status": "NOT COMPUTABLE", "reason": "the component is undefined in at least one arm of the pairing (e.g. every step masked)"}
        for e in Bs[:1]:
            for k_, v_ in e.get("unavailable", {}).items():
                if k_.startswith(fam + "."):
                    res["unavailable"].setdefault(k_, v_)
        for nm in sorted(names):
            if any(e[fam][nm].get("human_only") for e in Bs + As):
                res["excluded_human_only"].append(f"{fam}.{nm}")
                continue
            direction = PE.fam_direction(nm)
            r = paired_from_draws(JB, [comp_draws(JB, e[fam][nm]) for e in Bs], [comp_draws(JB, e[fam][nm]) for e in As], direction)
            r["extra_to_the_four_family_block"] = bool(Bs[0][fam][nm].get("extra_to_the_four_family_block", False))
            res[fam.upper()][nm] = r
            if r["status"] == "OK":
                c0 = comp_draws(JB, As[0][fam][nm])
                ic.append(paired_from_draws(JB, [c0], [c0], direction))
    comps = [(f, nm, r) for f in ("LONGITUDINAL", "LATERAL") for nm, r in res[f].items()]
    res["n_components_compared"] = sum(1 for _f, _n, r in comps if r.get("status") == "OK")
    res["n_components_compared_in_the_21"] = sum(1 for _f, _n, r in comps if r.get("status") == "OK" and not r.get("extra_to_the_four_family_block"))
    res["watch_items"] = sorted(f"{f}.{nm}" for f, nm, r in comps if r.get("status") == "OK" and r["adverse"])
    # ---- TACTICAL
    tac = {"status": "OK"}
    if all(isinstance(e.get("tactical"), dict) and e["tactical"].get("status") == "OK" for e in Bs + As):
        tac_adverse = []
        for d in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
            kcls = len(Bs[0]["tactical"][d]["classes"])
            dB = [decision_draws(JB, e["tactical"][d]["gt"], e["tactical"][d]["pred"], kcls) for e in Bs]
            dA = [decision_draws(JB, e["tactical"][d]["gt"], e["tactical"][d]["pred"], kcls) for e in As]
            tac[d] = {q: paired_from_draws(JB, [x[j] for x in dB], [x[j] for x in dA], "higher") for j, q in enumerate(("accuracy", "kappa"))}
            tac[d]["classes"] = list(Bs[0]["tactical"][d]["classes"])
            tac_adverse += [f"{d}.{q}" for q in ("accuracy", "kappa") if tac[d][q].get("status") == "OK" and tac[d][q]["adverse"]]
            c0 = decision_draws(JB, As[0]["tactical"][d]["gt"], As[0]["tactical"][d]["pred"], kcls)
            ic += [paired_from_draws(JB, [c0[j]], [c0[j]], "higher") for j, q in enumerate(("accuracy", "kappa")) if tac[d][q]["status"] == "OK"]
        goal = {}
        for nm in sorted(set.intersection(*[set(e["tactical"]["goal_setting"]) for e in Bs + As])):
            dr = GOAL_DIRECTION[nm]
            goal[nm] = paired_from_draws(JB, [comp_draws(JB, e["tactical"]["goal_setting"][nm]) for e in Bs], [comp_draws(JB, e["tactical"]["goal_setting"][nm]) for e in As], dr)
            if goal[nm]["status"] == "OK":
                c0 = comp_draws(JB, As[0]["tactical"]["goal_setting"][nm])
                ic.append(paired_from_draws(JB, [c0], [c0], dr))
        tac["goal_setting"] = goal
        tac["adverse_reported"] = sorted(tac_adverse + [f"goal_setting.{n}" for n, r in goal.items() if r.get("status") == "OK" and r["adverse"]])
        tac["selection_quality"] = "random / actual / best-of-64 on the same sets: families_pdm_paired.TACTICAL_selection_quality_pdms_x100 (paired, same draws)"
    else:
        tac = {"status": "NOT COMPUTABLE", "reason": "the per-set export carries no tactical decisions for every arm of the pairing"}
    tac["command_adherence_of_pick"] = {"status": "NOT COMPUTABLE", "reason": "the RL-val lookup carries no navigation command", "n": len(D["keys"])}
    res["TACTICAL"] = tac
    res["IC"] = {"ok": bool(ic and all(r.get("status") == "OK" and r["mean"] == 0.0 and r["ci95"] == [0.0, 0.0] and ("adverse_stat" not in r or (r["adverse_stat"]["mean"] == 0.0 and r["adverse_stat"]["ci95"] == [0.0, 0.0])) for r in ic)),
                 "n_statistics": len(ic), "rule": "an arm minus itself reads exactly 0 with CI [0, 0], through the same functions the real differences use"}
    return res


# ================================================================================================ the read
def families_status(fam, paired=None):
    """the result is not published without all four families, or each stated NOT COMPUTABLE with its reason and n -- and (amendment B4) not from an UNPAIRED reading: the per-set export must have paired and its IC must hold"""
    need = ("longitudinal", "lateral", "tactical", "strategic")
    if fam is None:
        return {"published": False, "reason": "no families block supplied (hn_screen_families.py)", "missing": list(need)}
    present = fam.get("families_present", {})
    miss = [k for k in need if k not in present]
    st = (present.get("strategic") or {}).get("status")
    paired_ok = bool(paired is not None and paired.get("status") == "OK" and paired.get("IC", {}).get("ok"))
    out = {"published": bool(not miss and st == "PAUSED" and paired_ok), "missing": miss, "strategic_status": st, "paired": paired_ok}
    if not paired_ok:
        out["reason"] = ("no PAIRED trajectory-level reading: " + (paired.get("reason", "the family IC failed") if paired else "no per-set export (--families-per-set) was supplied")
                         + " (amendment B4: S4 may not read families from the unpaired aggregates)")
    return out


def read(D, rule, gates, sy=None, c0=None, families=None, n_boot=N_BOOT, seed=SEED, per_set=None):
    JB = JointBoot(D["logs"], n_boot, seed)
    N = len(D["keys"])
    names = [n for n in D["arms"] if n != "C0"]
    V = {n: lookup(D, n, rule) for n in D["arms"]}
    res = {"tool": TOOL, "rule": rule, "n_sets": N, "n_logs": JB.G, "estimator": f"paired log-cluster bootstrap over {JB.G} logs, {n_boot} resamples, seed {seed}, percentile 2.5/97.5, ratio of sums; "
           "ONE log draw per resample applied jointly to every seed and arm",
           "tier": "SELECTION (lookup of the NAVSIM v1.1 PDMS of an open-loop plan on navtrain RL-val, frozen fan); not T1; any reading as navtest / T1 / P6 performance is ESTIMATED",
           "questions": {"bootstrap": "another draw of LOGS?", "Z2_Z3": "another TRAINING RUN? (sign rule and floor)", "I5": "another INFERENCE run? (closed by dump determinism)"},
           "strategic": PAUSED}
    z = decide_zeros(JB, V, None)
    res["zeros"] = {k: v for k, v in z.items()}
    seeds = z.get("seeds_used", list(SEEDS3))
    if z["state"] == "INVALID_INPUT":
        res["outcome"] = {"row": 0, "name": OUTCOMES[0][0], "reason": z["reason"]}
        return res
    res["delta_pdms"] = seedmean_diff(JB, [V[f"B{s}"]["PDMS"] for s in seeds], [V[f"A{s}"]["PDMS"] for s in seeds])
    pd_a = [100.0 * V[f"A{s}"]["PDMS"].mean() for s in seeds]
    res["F_pdms"] = max(pd_a) - min(pd_a)
    res["vs_base"] = {n: seedmean_diff(JB, [V[n]["PDMS"]], [V["base"]["PDMS"]]) for n in names if n != "base"} if "base" in V else None
    if "C0" in V:
        d = seedmean_diff(JB, [V["A0"]["PDMS"]], [V["C0"]["PDMS"]])
        sep = bool(d["ci95"][0] > 0 or d["ci95"][1] < 0)
        res["rig_check_A0_vs_C0"] = {**d, "separates": sep, "exceeds_F_pdms": bool(abs(d["mean"]) > res["F_pdms"]), "RIG_DRIFT": bool(sep and abs(d["mean"]) > res["F_pdms"]),
                                     "note": "report-only; the outcome is unchanged"}
    res["cogate"] = cogate(JB, V, seeds)
    ds = {}
    for q, key in (("DAC-only", "zero_dac_only"), ("NC-only", "zero_nc_only"), ("both", "zero_both"), ("other", "zero_other")):
        ds[q] = seedmean_diff(JB, [V[f"B{s}"][key] for s in seeds], [V[f"A{s}"][key] for s in seeds])
    res["M1_zero_split_pp"] = ds
    # ---- M2
    head_moves = False
    if D["plans8"] is not None:
        tw = {n: twin_stats(D, n, rule) for n in D["arms"] if n[0] in "AB" and n not in ("base",)}
        den = tw[f"A{seeds[0]}"][0]
        m2 = ratio_diff(JB, [tw[f"B{s}"][1] for s in seeds], [tw[f"A{s}"][1] for s in seeds], den)
        head_moves = bool(m2["ci95"][0] > 0)
        own = {}
        for n in sorted(tw):
            zk = V[n]["zero_dac_only"]
            own[n] = {**JB.ratio_interval(tw[n][1] * zk, tw[n][0] * zk), "population": "this arm's DAC-only selection-zero sets (ZT4's own)"}
        res["M2"] = {"delta_M2_bar": m2, "HEAD_MOVES": head_moves, "population": "every RL-val set with >= 1 near-twin pair (max over 8 poses <= 0.5 m, DAC_i != DAC_j)",
                     "n_sets_with_pair": int((den > 0).sum()), "bars": {"ZT4": ZT4_BAR, "u400_navtest": U400_NAVTEST_M2}, "own_population_per_arm": own}
    else:
        res["M2"] = {"status": "NOT COMPUTABLE", "reason": "no executed plans (--sets) supplied", "n": N}
    # ---- M3
    learns = None
    if D["bank"] is not None and D["arms9"] is not None:
        bp = {n: bank_pairs(D, n) for n in D["arms9"] if n[0] in "AB"}
        den3 = bp[f"A{seeds[0]}"][0]
        nq = int((den3 > 0).sum())
        status = np.asarray(D["bank"]["status"])
        res["M3"] = {"n_scenes_qualifying": nq, "n_scenes_lacking_a_side": int(((status == 0) & (den3 == 0)).sum()), "min_scenes": M3_MIN_SCENES}
        if nq >= M3_MIN_SCENES:
            JB3 = JBank(D, n_boot, seed)
            m3 = ratio_diff(JB3, [bp[f"B{s}"][1] for s in seeds], [bp[f"A{s}"][1] for s in seeds], den3)
            learns = bool(m3["ci95"][0] > 0)
            res["M3"].update(delta_M3_bar=m3, LEARNS_BANK=learns)
        else:
            res["M3"].update(status="NOT COMPUTABLE", reason=f"fewer than {M3_MIN_SCENES} RL-val bank scenes qualify (n = {nq})")
    else:
        res["M3"] = {"status": "NOT COMPUTABLE", "reason": "no RL-val bank / offset scores supplied", "n": 0}
    # ---- the tactical selection quality and the PDM sub-scores of the other families (paired, seed-mean, joint draws)
    res["families_pdm_paired"] = {"LONGITUDINAL": {q: seedmean_diff(JB, [V[f"B{s}"][q] for s in seeds], [V[f"A{s}"][q] for s in seeds]) for q in ("EP", "TTC")},
                                  "LATERAL": {"DAC": seedmean_diff(JB, [V[f"B{s}"]["DAC"] for s in seeds], [V[f"A{s}"]["DAC"] for s in seeds])},
                                  "TACTICAL_selection_quality_pdms_x100": {k: seedmean_diff(JB, [V[f"B{s}"][k] for s in seeds], [V[f"A{s}"][k] for s in seeds])
                                                                           for k in ("PDMS", "rand", "best")},
                                  "STRATEGIC": PAUSED}
    res["watch_items_pdm"] = sorted(f"{f}.{q}" for f, qs in (("LONGITUDINAL", ("EP", "TTC")), ("LATERAL", ("DAC",))) for q in qs
                                    if res["families_pdm_paired"][f][q]["ci95"][1] < 0)
    fp = families_paired(JB, D, per_set, seeds) if per_set is not None else None
    if fp is not None:
        res["families_paired"] = fp
    res["watch_items_trajectory"] = fp["watch_items"] if fp and fp.get("status") == "OK" else None
    res["watch_items"] = sorted(res["watch_items_pdm"] + (fp["watch_items"] if fp and fp.get("status") == "OK" else []))
    res["families"] = families_status(families, fp)
    # ---- IC
    a0 = [V[f"A{seeds[0]}"]]
    ic = {"delta_z": seedmean_diff(JB, [a0[0]["zero"]], [a0[0]["zero"]]), "PDMS": seedmean_diff(JB, [a0[0]["PDMS"]], [a0[0]["PDMS"]]),
          "NC": seedmean_diff(JB, [a0[0]["NC"]], [a0[0]["NC"]]), "TTC": seedmean_diff(JB, [a0[0]["TTC"]], [a0[0]["TTC"]]),
          "M1": [seedmean_diff(JB, [a0[0][k]], [a0[0][k]]) for k in ("zero_dac_only", "zero_nc_only", "zero_both")]}
    if D["plans8"] is not None:
        t = twin_stats(D, f"A{seeds[0]}", rule)
        ic["M2"] = ratio_diff(JB, [t[1]], [t[1]], t[0])
    if D["bank"] is not None and D["arms9"] is not None and res["M3"].get("n_scenes_qualifying"):
        b = bank_pairs(D, f"A{seeds[0]}")
        ic["M3"] = ratio_diff(JBank(D, n_boot, seed), [b[1]], [b[1]], b[0])
    flat = [ic["delta_z"], ic["PDMS"], ic["NC"], ic["TTC"], *ic["M1"], *( [ic["M2"]] if "M2" in ic else []), *( [ic["M3"]] if "M3" in ic else [])]
    res["IC"] = {"ok": bool(all(x["mean"] == 0.0 and x["ci95"] == [0.0, 0.0] for x in flat)), "n_statistics": len(flat), "rule": "an arm minus itself reads exactly 0 with CI [0, 0]"}
    gts = dict(gates)
    gts["IC"] = res["IC"]["ok"]
    row = classify({"gates": gts, "sy_sparse": bool(sy is not None and sy.get("share_with_hn", 1.0) < SY_MIN_SHARE) or bool(sy is not None and sy.get("sparse", False)), "cogate_fails": res["cogate"]["fails"],
                    "zeros": z, "dpdms_point": res["delta_pdms"]["mean"], "lb_dz_positive": bool(z["final"]["LB_positive"]), "head_moves": head_moves, "learns_bank": learns})
    if row is None:
        res["outcome"] = {"row": None, "name": "PENDING (rule E)", "reason": z["reason"], "full_arm": "not decided"}
    else:
        nm, full, nxt = OUTCOMES[row]
        res["outcome"] = {"row": row, "name": nm, "full_arm": full, "next_lever": nxt, "rig_drift": bool(res.get("rig_check_A0_vs_C0", {}).get("RIG_DRIFT", False))}
    return res


def JBank(D, n_boot, seed):
    """the bootstrap over the RL-val BANK scenes, clustered by their logs (the bank carries its keys' logs)"""
    return JointBoot(D["bank"]["logs"], n_boot, seed)


# ================================================================================================ the CLI assembly (files -> make_data)
def load_plans8(dir_, keys, ckpts):
    import glob
    import select_bank_subset as SB
    import navsim_pdm_targets as PT
    want, out = set(keys), {}
    for fp in sorted(glob.glob(os.path.join(dir_, "onpolicy_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                f = SB.head_fields(ln[:2000] + ln[-400:])
                if "log_name" not in f:
                    continue
                k = (f["log_name"], f.get("token", ""), int(f.get("step", 0)), int(f.get("rank", 0)))
                if k not in want or int(f.get("ckpt_step", -1)) != ckpts[k]:
                    continue
                r = json.loads(ln)
                out[k] = PT.executed_plans8(np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1), repair=True)
    return out


def assemble(arms, pdm, scenes, plans=None, bank=None, arms9=None):
    """arms: {name: RV.load_scores dict}; pdm: RV.load_pdm dict; scenes {token: log}.  The sets every arm scored, that have labels of the same ckpt_step and are RL-val scenes."""
    common = set.intersection(*[set(a) for a in arms.values()])
    keys = []
    for k in sorted(common):
        p = pdm.get(k)
        if p is None or {a[k]["ckpt"] for a in arms.values()} != {p["ckpt"]}:
            continue
        if scenes is not None and scenes.get(k[1]) != k[0]:
            raise ValueError(f"set {RV.kstr(k)} is not an RL-val scene: refusing to read a train scene as validation")
        keys.append(k)
    tg = np.stack([pdm[k]["tg"] for k in keys])
    sc = {n: np.stack([a[k]["scores"] for k in keys]) for n, a in arms.items()}
    sp = {a[k]["space"] for a in arms.values() for k in keys}
    assert len(sp) == 1, f"mixed score spaces: {sp}"
    pl = None
    if plans is not None:
        miss = [k for k in keys if k not in plans]
        if miss:
            raise ValueError(f"{len(miss)} sets have no executed plans (first {RV.kstr(miss[0])})")
        pl = np.stack([plans[k] for k in keys])
    return make_data(keys, [k[0] for k in keys], tg, sc, pl, bank, arms9, space=sp.pop())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="name=path; names base, A0..A4, B0..B4 and optionally C0")
    ap.add_argument("--pdm", required=True)
    ap.add_argument("--rl-val-scenes", required=True)
    ap.add_argument("--sets", default=None, help="the frozen-fan dump dir (executed plans for M2)")
    ap.add_argument("--gates", required=True, help="json {I1,I2,I3,I4,I5,I6,D1,G14,G15,launch_gate: bool} written by the gate runs; a missing key is a failed gate")
    ap.add_argument("--chain-root", required=True, help="the chain's state dir (passes/s<seed>/lg/PASS_*.json, ctx_chain/anchor.json, ctx_chain/end_verify.json): the read REFUSES unless the passes' cache binding is proven (see the module doc)")
    ap.add_argument("--key-file", default=None, help="the launch-gate key that signed the tokens and the chain's anchor / END record (default: the host's gate key)")
    ap.add_argument("--sy", default=None, help="the SY yield report json (hn_bank.py yield)")
    ap.add_argument("--bank9", default=None, help="npz {dac9 [N9,9], status [N9], log [N9]} of the RL-val bank")
    ap.add_argument("--scores9", nargs="*", default=[], help="name=npz {scores9 [N9,9,C]} from hn_screen_offsets_score.py")
    ap.add_argument("--families", default=None)
    ap.add_argument("--families-per-set", default=None, help="npz of the per-set trajectory-level component values (hn_screen_families.py --per-set-out); without it the families are NOT published (amendment B4)")
    ap.add_argument("--rule", required=True, choices=RV.RULES)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    passes = sorted({int(n[1:]) for n in (spec.partition("=")[0] for spec in a.scores) if len(n) > 1 and n[0] in "AB" and n[1:].isdigit()})
    try:
        import gate_token as GT                                                    # refe/ is on sys.path (above)
        why = GT.ctx_chain_refusals(a.chain_root, passes, GT.load_key(a.key_file, create=False))
    except Exception as e:                                                         # noqa: BLE001 -- a check that cannot run is a refusal, never a pass
        why = [f"the chain's cache binding cannot be checked: {type(e).__name__}: {str(e)[:200]}"]
    if why:
        for r in why[:6]:
            print("REFUSED:", r[:400], flush=True)
        print("ZZHNSCREEN_REFUSED the chain's ctx-cache binding is not proven (no END_VERIFY_OK / content-bound token): nothing was read", flush=True)
        return 6
    arms = {}
    for spec in a.scores:
        n, _, p = spec.partition("=")
        arms[n] = RV.load_scores(p)
    pdm, _ = RV.load_pdm(a.pdm)
    scenes = RV.read_scenes(a.rl_val_scenes)
    plans = None
    if a.sets:
        common = set.intersection(*[set(x) for x in arms.values()])
        plans = load_plans8(a.sets, common, {k: next(iter(arms.values()))[k]["ckpt"] for k in common})
    bank = arms9 = None
    if a.bank9:
        z = np.load(a.bank9, allow_pickle=False)
        bank = {"dac9": z["dac9"], "status": z["status"], "logs": [str(x) for x in z["log"]]}
        arms9 = {}
        for spec in a.scores9:
            n, _, p = spec.partition("=")
            arms9[n] = np.load(p, allow_pickle=False)["scores9"]
    D = assemble(arms, pdm, scenes, plans, bank, arms9)
    gates = {k: bool(v) for k, v in json.load(open(a.gates, encoding="utf-8")).items()}
    sy = json.load(open(a.sy, encoding="utf-8")) if a.sy else None
    fam = json.load(open(a.families, encoding="utf-8")) if a.families else None
    per_set = None
    if a.families_per_set:
        import hn_screen_families as HF
        per_set = HF.read_per_set(a.families_per_set)
    res = read(D, a.rule, gates, sy=sy, families=fam, n_boot=a.n_boot, per_set=per_set)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    o = res["outcome"]
    print(f"ZZHNSCREEN_{'PENDING' if o['row'] is None else 'OUTCOME'} row {o['row']} {o['name']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
