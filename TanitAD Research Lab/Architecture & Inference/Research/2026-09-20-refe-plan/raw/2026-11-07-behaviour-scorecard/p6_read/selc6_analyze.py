#!/usr/bin/env python3
"""selc6_analyze.py -- the CPU half of the pre-registered P6 read (PREREG_P6_SCORER_SPEED.md, H-P6-SCORER-SPEED-1 / "H-P6-SPEED-1").

Reads a dump + a 64-column PDM table of the 1,058 C3 tokens, runs the gates G6-PIN .. G6-REPRO of section 3, computes the section 4 statistics
with the section 4 estimators (tie-correct AVERAGE ranks; paired log-cluster bootstrap, B = 10,000, seed 20260927, the programme's
`sel_common.log_boot_idx`), and issues the section 5 verdict MECHANICALLY. It writes p6_speed_result.json and P6_SPEED_RESULT.md.

    python selc6_analyze.py --dump DIR/dump/dump_p6.npz --table DIR/score/table_p6.npz --out DIR --run-config config.json \
        --train-logs train_logs.txt --p6-seam <full-navtest seam npz> --p6-csv <full-navtest PDMS csv>
    python selc6_analyze.py --control        # u400 as the control (see p6_control.py): never issues a verdict on P6

Nothing here reads a P6 output at import time. The prereg is NOT edited; where it is ambiguous the code applies the literal text and
lists the point in `ambiguities_applied` of the result (and in AMBIGUITIES.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "E:/Projects/TanitAD-artifacts"
BEH = ART + "/refe_behaviour_20261107"
SELP = ART + "/refe_selection_probe_20261107"
SELC = ART + "/refe_selection_confirm_20261107"
EGO = ART + "/refe_ego_probe_20261106"
PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
NAVTRAIN_YAML = PK + "/splits/navtrain.yaml"
if SELP not in sys.path:
    sys.path.insert(0, SELP)
import sel_common as sc  # noqa: E402  (pinned by sha256, G6-PIN)

# ------------------------------------------------------------------------------------------------ frozen constants (prereg sections 2-5)
PREREG_SHA = "8e8aa3cac3762ae4351dc09ef8842945b2f8d6ff82aeb849ee22811b594c0e6a"
W_ZERO, W_RANDOM = 529 / 12146, 11617 / 12146
B_DRAWS, SEED = 10000, 20260927
SLOW_RATIO = 0.85
OVERSPEED_RATIO = 1.15
EQ_BAND = 0.15                  # (A) and (B)
SHARE_BAR = 0.20                # (C)
MIN_VALID = 5
HEADS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3}
SUBN = ("NC", "DAC", "EP", "TTC", "C", "DDC")
N_TOKENS, N_CRUISE = 1058, 154
FC_TOL_M = 0.001
FIDELITY_TOL_M = 1e-3
# pins: name -> (path, accepted sha256 tuple). The FIRST sha is the one the card prints in section 9 (or the file it names); a later sha is an
# AMENDED value, accepted only with a note (never silently): see AMBIGUITIES.md items 1-3.
PINS = {
    "tokens_cruise.json": (EGO + "/tokens_cruise.json", ("57db811e644218c00020cd6d25b417b604ffc9af7b81612976d0d52f02263f78",)),
    "tokens_all.json": (EGO + "/tokens_all.json", ("421aac1d51da3beb3497ac92494ed7dacd99efe62e758e49ae5ada63359fa566",)),
    "sel_common.py": (SELP + "/sel_common.py", ("50727023525c86b2be503db5c7f84d56de9b2be98f260aab700e4c9438f27d23",)),
    "p1_diag.py": (SELP + "/p1_diag.py", ("7bd7220dffa1ae117bc1803ccdf8a4c76163d17bdfabbe3016dde560e2ca8b9d",)),
    "p1_diag_all.json": (SELP + "/p1_diag_all.json", ("1645b6572f593f94ffcd46c7e8977ba0aec7182bcb5cffb660c416b06b1d9a6c",)),
    "PREREG_SEL.md": (SELP + "/PREREG_SEL.md", ("e00fa28a7a5d6352b6f5a1782b34966bd75d88ad6b99ce7a614264581db0b7f0",)),
    "selc CODE_PIN.json": (SELC + "/CODE_PIN.json", ("86e7b9fdb5a3479cfefdddc9cee906be1e48f8134666c8346b7b92a79284f3ea",
                                                     "a4b8401194a350fdfb53764b6e0939498f051ecfdb20229b85b20c2163ce7d4d", "255c75e117803d42b4389aa3ac04da2a5ea74f82965dc40681bcb97e43498329")),
    "selc_dump.py": (SELC + "/selc_dump.py", ("cf16d85b93dd470ed620f8f646219919a9fc3b9d8570b5e82b5a99d49d07e8d0",
                                              "1de686757e10b488937d9e45cc0289ebfa9a746488cc06272271ef9f7381e17d", "a742ec952a5493107deb8b228350be5ffe77e966b45b0190ea91838d10f482d1")),
    "selc_score.py": (SELC + "/selc_score.py", ("3fd4997a05e55240376fbb71af7c0459f9ccb2fa6ddf420a737648a26fb6ae09",
                                                "0e76e98becae7d6c47a838d4ce15ddc9ece0d7269fbb5d4da78f19e0e9b1d072")),
    "C3 table.npz": (sc.C3 + "/table.npz", ("cf30519051bea03b5a950975477806ce9b1ef141ac270dbe5781ea53f6dbc7c0",)),
    "u400_reference.py": (BEH + "/u400_reference.py", ("ee124c5b3f4114ff4129a8d872ec3129b928dc7807cbbf306361ef0f5e2e3a11",)),
    # the card pins 1ad5776a (= what the pinned script writes today); the file at the card's path hashes differently, see AMBIGUITIES.md item 3.
    "u400_reference.json (pinned content)": (HERE + "/u400_reference.pinned.json", ("1ad5776a19a02179cb22a63fc9d20e9a04db34d4a2ffd6b6f14796e8121682e5",)),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ G6-PIN
def g6_pin(prereg_path=None, code_pin_path=None):
    prereg_path = prereg_path or BEH + "/PREREG_P6_SCORER_SPEED.md"
    rows, ok = [], True
    got = sha256_file(prereg_path)
    rows.append({"what": "PREREG_P6_SCORER_SPEED.md", "sha256": got, "expected": PREREG_SHA, "ok": got == PREREG_SHA, "note": ""})
    for name, (p, accepted) in PINS.items():
        if not os.path.exists(p):
            rows.append({"what": name, "sha256": None, "expected": accepted[0], "ok": False, "note": "MISSING"})
            continue
        got = sha256_file(p)
        note = "" if got == accepted[0] else ("AMENDED value accepted (AMBIGUITIES.md)" if got in accepted else "DRIFT")
        rows.append({"what": name, "sha256": got, "expected": accepted[0], "ok": got in accepted, "note": note})
    if code_pin_path:                                                       # CODE_PIN.json of THIS package: every file must still hash to its pin
        pin = json.load(open(code_pin_path))
        for f, h in pin["files"].items():
            got = sha256_file(os.path.join(HERE, f))
            rows.append({"what": "p6_read/" + f, "sha256": got, "expected": h, "ok": got == h, "note": "" if got == h else "DRIFT"})
    ok = all(r["ok"] for r in rows)
    return {"PASS": bool(ok), "rows": rows}


# ------------------------------------------------------------------------------------------------ statistics (section 4)
def probs(L):
    return 1.0 / (1.0 + np.exp(-np.asarray(L).astype(np.float64)))


def per_token_spearman(X, Y, M, method="tie"):
    """Per-token Spearman over the valid proposals. method 'tie' = AVERAGE ranks (the pre-registered estimator);
    'legacy' = argsort(argsort(.)) (p1_diag.py spearman_rows; breaks ties by proposal INDEX). 'bug_argsort' is the deliberate-regression alias of 'legacy'."""
    out = np.full(X.shape[0], np.nan)
    for i in np.where(M.any(1))[0]:
        m = M[i]
        if m.sum() < MIN_VALID:
            continue
        if method in ("legacy", "bug_argsort"):
            rx = np.argsort(np.argsort(X[i, m]))
            ry = np.argsort(np.argsort(Y[i, m]))
        else:
            rx, ry = rankdata(X[i, m]), rankdata(Y[i, m])
        if rx.std() == 0 or ry.std() == 0:
            continue
        out[i] = np.corrcoef(rx, ry)[0, 1]
    return out


class Boot:
    """One shared [B, n_logs] resample of log ids for EVERY statistic and for u400 (section 4)."""

    def __init__(self, log):
        self.ul, self.li, self.draws = sc.log_boot_idx(log, B_DRAWS, SEED)
        self.nl = len(self.ul)

    def mean(self, vals, mask):
        return sc.boot_mean(vals, mask, self.li, self.nl, self.draws)

    def parts(self, vals, mask):
        m = mask & np.isfinite(vals)
        return (np.bincount(self.li[m], weights=vals[m], minlength=self.nl), np.bincount(self.li[m], minlength=self.nl).astype(float))

    def wshare(self, v, cru, zero, w_zero=W_ZERO, w_random=W_RANDOM, order="wsc"):
        """stratum-weighted share over CRUISE tokens. Returns point, per-draw array, ok mask (draws that leave a stratum empty are dropped)."""
        sz, cz = self.parts(v, cru & zero)
        sr, cr = self.parts(v, cru & ~zero)
        pt = w_zero * sz.sum() / cz.sum() + w_random * sr.sum() / cr.sum()
        if order == "zfirst":      # u400_reference.py::wshare (R1) divides first, then weights; its R3 helper weights first. Both orders are kept so the control is bit-exact.
            b = w_zero * (sz[self.draws].sum(1) / np.maximum(cz[self.draws].sum(1), 1e-12)) + w_random * (sr[self.draws].sum(1) / np.maximum(cr[self.draws].sum(1), 1e-12))
        else:
            b = w_zero * sz[self.draws].sum(1) / np.maximum(cz[self.draws].sum(1), 1e-12) + w_random * sr[self.draws].sum(1) / np.maximum(cr[self.draws].sum(1), 1e-12)
        ok = (cz[self.draws].sum(1) > 0) & (cr[self.draws].sum(1) > 0)
        return pt, b, ok, (sz, cz, sr, cr)


def ci(b, ok):
    return float(np.percentile(b[ok], 2.5)), float(np.percentile(b[ok], 97.5))


def ratio_of(R):
    v0 = np.maximum(R["ego"][:, 6], 1e-6)
    return R["v4_all"] / v0[:, None]


# ------------------------------------------------------------------------------------------------ run adapters
def align_to_c3(arrs, tokens):
    """Reorder dump/table arrays (any token order) to the C3 table order of sel_common.load()."""
    D = sc.load()
    idx = {str(t): i for i, t in enumerate(tokens)}
    assert set(idx) == set(D["token"]), f"token set != the 1,058 C3 tokens ({len(set(idx) ^ set(D['token']))} differ)"
    ix = np.array([idx[t] for t in D["token"]])
    return D, {k: v[ix] for k, v in arrs.items()}


def run_from_files(dump_npz, table_npz):
    """A `Run` for P6: dump (GPU stage) + 64-column table (CPU stage), aligned to the C3 token order."""
    Z = np.load(dump_npz, allow_pickle=False)
    T = np.load(table_npz, allow_pickle=False)
    dtok = [str(t) for t in Z["token"]]
    assert [str(t) for t in T["token"]] == dtok, "dump and table token order differ"
    keys = ("logits", "pick", "traj", "props8", "ego", "gt_valid", "gt_pose20", "gt_t20", "human8", "frame_ctrl_m", "fingerprint")
    arrs = {k: Z[k] for k in keys}
    arrs.update(pdms=T["pdms"], sub=T["sub"], valid=T["valid"])
    D, A = align_to_c3(arrs, dtok)
    R = dict(token=D["token"], log=D["log"], stratum=D["stratum"], cruise=D["cruise"], ax0=A["ego"][:, 2], v0=A["ego"][:, 6],
             v4_all=sc.step_speed(A["traj"].astype(np.float64)), v4_gt=sc.gt_speed4(A["gt_pose20"], A["gt_t20"]), **A)
    R["pick"] = R["pick"].astype(np.int64)
    R["pdms"], R["sub"] = R["pdms"].astype(np.float64), R["sub"].astype(np.float64)
    R["n_outputs"] = int(R["logits"].shape[-1])
    R["lane"] = R["n_outputs"] == 7
    R["meta"] = {k: str(Z[k]) for k in ("rule", "ckpt", "ckpt_sha256", "nav_mode", "select") if k in Z.files}
    R["red_pick_pdms"] = T["red_pick_pdms"][[{str(t): i for i, t in enumerate(dtok)}[x] for x in D["token"]]] if "red_pick_pdms" in T.files else None
    return R


def run_control_u400():
    """u400 on the banked C3 table, shaped as a Run (the control: same code path, banked data only, no GPU, no P6 value)."""
    D = sc.load()
    Z = np.load(sc.EGO + "/full/results.npz", allow_pickle=False)
    zt = {str(t): i for i, t in enumerate(Z["token"])}
    zi = np.array([zt[t] for t in D["token"]])
    T = np.load(sc.C3 + "/table.npz", allow_pickle=False)
    a0 = [str(x) for x in Z["arms"]].index("A0")
    R = dict(token=D["token"], log=D["log"], stratum=D["stratum"], cruise=D["cruise"], ego=Z["ego_real"][zi], ax0=D["ax0"], v0=D["v0"], v4_all=D["v4_all"],
             v4_gt=D["v4_gt"], logits=D["logits"], pick=D["pick"], traj=D["traj"], props8=T["proposals"], sub=D["sub"], valid=D["valid"], pdms=D["pdms"],
             gt_valid=Z["gt_valid"][zi].astype(bool), gt_pose20=Z["gt_pose20"][zi], gt_t20=Z["gt_t20"][zi], human8=Z["human8"][zi].astype(np.float64),
             frame_ctrl_m=np.zeros(len(D["token"])), n_outputs=7, lane=True, meta={"rule": "navsim_v1_lane", "nav_mode": "goal"}, red_pick_pdms=None)
    return R


def u400_vectors(Du, B):
    """u400's slow-pick indicators: banked rule (navsim_v1_lane) and the lane-free re-pick (navsim_v1 on the same logits) -- reference R3, same code."""
    r = np.arange(len(Du["token"]))
    ratio = ratio_of(Du)
    pick_nl = sc.pick_from_logits(Du["logits"], "torch", lane=False)
    return {"pick_banked": Du["pick"], "pick_nolane": pick_nl, "slow_banked": (ratio[r, Du["pick"]] < SLOW_RATIO).astype(float),
            "slow_nolane": (ratio[r, pick_nl] < SLOW_RATIO).astype(float)}


# ------------------------------------------------------------------------------------------------ the statistics block (reference-shaped)
def stats_block(R, B, rank_method="tie", swap_pred_true=False, w_zero=W_ZERO, w_random=W_RANDOM):
    """R1 + R2 in EXACTLY the shape of u400_reference.json, so the control can compare leaf by leaf."""
    N = len(R["token"])
    r = np.arange(N)
    ratio = ratio_of(R)
    P = probs(R["logits"])
    sub, val, cru = R["sub"], R["valid"], R["cruise"]
    zero = R["stratum"] == "zero"
    slow = (ratio[r, R["pick"]] < SLOW_RATIO).astype(float)
    pt, b, ok, (sz, cz, sr, cr) = B.wshare(slow, cru, zero, w_zero, w_random, order="zfirst")
    lo, hi = ci(b, ok)
    out = {"weights": {"W_zero": w_zero, "W_random": w_random},
           "R1_weighted_cruise_slow_share": dict(point=float(pt), lo=lo, hi=hi, n_cruise_zero=int((cru & zero).sum()), n_cruise_random=int((cru & ~zero).sum()),
                                                 share_zero=float(sz.sum() / cz.sum()), share_random=float(sr.sum() / cr.sum()),
                                                 share_unweighted=float(slow[cru].mean()), n_draws_used=int(ok.sum()))}
    cm = cru[:, None] & val
    for method in ("legacy", "tie"):
        meth = rank_method if method == "tie" else "legacy"
        block = {}
        for h, j in HEADS.items():
            rp = per_token_spearman(P[..., j], ratio, cm, meth)
            rt = per_token_spearman(sub[..., j], ratio, cm, meth)
            if swap_pred_true:
                rp, rt = rt, rp
            okk = np.isfinite(rp) & np.isfinite(rt)
            gap = np.where(okk, rp - rt, np.nan)
            block[h] = {"pred": B.mean(np.where(okk, rp, np.nan), cru), "true": B.mean(np.where(okk, rt, np.nan), cru),
                        "gap_pred_minus_true": B.mean(gap, cru), "n_tokens_paired": int(okk.sum()),
                        "pred_unpaired_mean": float(np.nanmean(rp[cru])), "true_unpaired_mean": float(np.nanmean(rt[cru])),
                        "n_pred": int(np.isfinite(rp[cru]).sum()), "n_true": int(np.isfinite(rt[cru]).sum())}
            block[h]["_S_all_tokens_ci"] = B.mean(rp, cru)         # extra (not in the reference): S_h with its interval, section 4
            block[h]["_T_defined_tokens_ci"] = B.mean(rt, cru)
        out["R2_spearman_" + method] = block
    out["_slow"], out["_wshare_draws"] = slow, (pt, b, ok)
    return out


def r3_block(Du, B):
    """Reference R3: u400 without the lane factor, paired delta vs banked."""
    U = u400_vectors(Du, B)
    cru, zero = Du["cruise"], Du["stratum"] == "zero"
    p1, b1, ok1, _ = B.wshare(U["slow_banked"], cru, zero)
    p2, b2, ok2, _ = B.wshare(U["slow_nolane"], cru, zero)
    d = (b2 - b1)[ok1 & ok2]
    lo2, hi2 = ci(b2, ok2)
    return {"n_picks_changed_cruise": int((U["pick_nolane"][cru] != Du["pick"][cru]).sum()), "n_picks_changed_all": int((U["pick_nolane"] != Du["pick"]).sum()),
            "weighted_slow_share": {"point": float(p2), "lo": lo2, "hi": hi2},
            "delta_vs_banked": {"point": float(p2 - p1), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))},
            "G_banked_rule_reproduced": int((sc.pick_from_logits(Du["logits"], "torch", lane=True) == Du["pick"]).sum())}, U


# ------------------------------------------------------------------------------------------------ the verdict (section 5), mechanical
def decide(S):
    """S: dict with G_TTC, G_NC, S_TTC, S_NC, T_TTC, T_NC, share, dA_hi, dB_hi (A = vs u400 banked, B = vs u400 without lane). Returns verdict + conditions.
    Section 5 text applied literally; a non-finite input -> INSTRUMENT FAILURE (never a verdict)."""
    need = ("G_TTC", "G_NC", "S_TTC", "S_NC", "T_TTC", "T_NC", "share", "dA_hi", "dB_hi")
    bad = [k for k in need if not np.isfinite(S[k])]
    if bad:
        return {"VERDICT": "INSTRUMENT FAILURE", "why": f"statistics undefined: {bad}", "conditions": {}, "subcases": []}
    A = bool(-EQ_BAND <= S["G_TTC"] <= EQ_BAND and -EQ_BAND <= S["G_NC"] <= EQ_BAND)
    Bc = bool(abs(S["S_TTC"] - S["T_TTC"]) <= EQ_BAND and abs(S["S_NC"] - S["T_NC"]) <= EQ_BAND)
    C = bool(S["share"] <= SHARE_BAR and S["dA_hi"] < 0 and S["dB_hi"] < 0)
    persists = bool(S["G_TTC"] < -EQ_BAND or S["G_NC"] < -EQ_BAND)
    sub = []
    if A and Bc and C:
        v = "HOLDS"
    elif persists and not C:
        v = "REFUTED"
    else:
        v = "NOT PROVEN"
        if A and Bc and not C:
            sub.append("N1")
        if C and (not A or not Bc):
            sub.append("N2")
        if A and not Bc:
            sub.append("N3")
        if not sub:
            sub.append("N-unnamed (matches none of N1-N3; see AMBIGUITIES.md item 6)")
    return {"VERDICT": v, "conditions": {"A_head_calibration": A, "B_not_subset_driven": Bc, "C_behaviour": C, "pessimism_persists": persists},
            "subcases": sub, "inputs": {k: float(S[k]) for k in need}}


# ------------------------------------------------------------------------------------------------ gates (section 3)
def g6_0(R):
    """G6-0: the dump's pick == the offline torch-float32 argmax of the aggregate on 1,058/1,058. RED: argmin must disagree on EVERY token; the aggregate with
    a flipped sign on p3 must disagree on >= 1 token."""
    L = np.asarray(R["logits"], np.float32)
    lane = R["n_outputs"] == 7
    pk = sc.pick_from_logits(L, "torch", lane=lane)
    a = sc.agg_torch(L, lane=lane)
    p = 1.0 / (1.0 + np.exp(-L.astype(np.float32)))
    w = np.asarray(sc.V1_W, np.float32)
    flip = p[..., 0] * p[..., 1] * (p[..., 2] * w[0] - p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()
    if lane:
        flip = flip * p[..., 6]
    n = len(pk)
    g = {"n_equal": int((pk == R["pick"]).sum()), "n": n, "RED_argmin_disagree": int((np.argmin(a, -1) != R["pick"]).sum()),
         "RED_flip_p3_disagree": int((np.argmax(flip, -1) != R["pick"]).sum())}
    g["PASS"] = bool(g["n_equal"] == n and g["RED_argmin_disagree"] == n and g["RED_flip_p3_disagree"] >= 1)
    return g


def g6_1(R, seam_npz):
    """G6-1: dump executed pick poses == the P6 read's full-navtest seam to <= 1e-3 m on 1,058/1,058. RED: a cyclic shift of the reference must fail."""
    if not seam_npz:
        return {"PASS": False, "why": "NOT EVALUATED: no --p6-seam (the P6 read's full-navtest seam npz)"}
    S = np.load(seam_npz, allow_pickle=False)
    si = {str(t): i for i, t in enumerate(S["token"])}
    miss = [t for t in R["token"] if t not in si]
    if miss:
        return {"PASS": False, "why": f"{len(miss)} of the C3 tokens are absent from the seam"}
    ix = np.array([si[t] for t in R["token"]])
    ref = S["poses"][ix].astype(np.float64)
    pick = R["props8"][np.arange(len(ix)), R["pick"]].astype(np.float64)
    d = np.abs(pick - ref).reshape(len(ix), -1).max(1)
    dsh = np.abs(pick - np.roll(ref, 1, axis=0)).reshape(len(ix), -1).max(1)
    g = {"n_within_tol": int((d <= FIDELITY_TOL_M).sum()), "n": len(ix), "max_abs_m": float(d.max()), "RED_cyclic_shift_n_failing": int((dsh > FIDELITY_TOL_M).sum())}
    g["PASS"] = bool(g["n_within_tol"] == g["n"] and g["RED_cyclic_shift_n_failing"] >= 1)
    return g


def cruise_formula(R):
    return R["gt_valid"] & (R["v0"] > 5.0) & (np.abs(R["ax0"]) < 0.3) & (np.abs(R["v4_gt"] - R["v0"]) < 1.0)


def g6_2(R):
    """G6-2: the dump's `ego` vector == `ego_real` of the ego probe on 1,058/1,058 (bitwise) and the CRUISE formula reproduces tokens_cruise.json 1,058/1,058."""
    Z = np.load(sc.EGO + "/full/results.npz", allow_pickle=False)
    zt = {str(t): i for i, t in enumerate(Z["token"])}
    ref = Z["ego_real"][np.array([zt[t] for t in R["token"]])]
    eq = np.array([np.array_equal(R["ego"][i].astype(np.float64), ref[i]) for i in range(len(ref))])
    cf = cruise_formula(R)
    bad = R["ego"].copy()
    bad[:, 6] += 1e-3
    red = int(sum(np.array_equal(bad[i].astype(np.float64), ref[i]) for i in range(len(ref))))
    g = {"ego_bit_equal": int(eq.sum()), "n": len(ref), "cruise_formula_equal": int((cf == R["cruise"]).sum()), "n_cruise_formula": int(cf.sum()),
         "n_cruise_file": int(R["cruise"].sum()), "RED_perturbed_ego_still_equal": red}
    g["PASS"] = bool(g["ego_bit_equal"] == g["n"] and g["cruise_formula_equal"] == g["n"] and red == 0)
    return g


def read_csv_scores(p):
    import csv
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]))
    return rows


def g6_3(R, csv_path):
    """G6-3: the banked-pick column (the table's PDMS at the dump's pick) == the P6 read's csv PDMS BIT FOR BIT on 1,058/1,058. RED: the pick poses shifted +2.0 m
    in y, scored by the same harness, must differ on >= 1 token (R['red_pick_pdms'], scored by p6_score.py)."""
    if not csv_path:
        return {"PASS": False, "why": "NOT EVALUATED: no --p6-csv (the P6 read's full-navtest PDMS csv)"}
    rows = read_csv_scores(csv_path)
    miss = [t for t in R["token"] if t not in rows]
    if miss:
        return {"PASS": False, "why": f"{len(miss)} C3 tokens absent from the csv"}
    r = np.arange(len(R["token"]))
    mine = R["pdms"][r, R["pick"]]
    ref = np.array([rows[t][1] for t in R["token"]])
    g = {"n_bit_equal": int((mine == ref).sum()), "n": len(ref), "max_abs_diff": float(np.abs(mine - ref).max())}
    if R.get("red_pick_pdms") is None:
        g["RED"] = "NOT EVALUATED: no shifted-pose column in the table"
        g["PASS"] = False
    else:
        g["RED_shift_y2m_n_differ"] = int((R["red_pick_pdms"] != ref).sum())
        g["PASS"] = bool(g["n_bit_equal"] == g["n"] and g["RED_shift_y2m_n_differ"] >= 1)
    return g


def g6_4(R):
    """G6-4: 64 x 1,058 finite and valid; PDMS recomputed from the six sub-scores (NC*DAC*(5EP+5TTC+2C)/12) == the harness PDMS to 1e-9."""
    s = R["sub"]
    f = s[..., 0] * s[..., 1] * (5 * s[..., 2] + 5 * s[..., 3] + 2 * s[..., 4]) / 12
    d = np.abs(f - R["pdms"])
    bad = s.copy()
    bad[..., 2] += 0.01
    fb = bad[..., 0] * bad[..., 1] * (5 * bad[..., 2] + 5 * bad[..., 3] + 2 * bad[..., 4]) / 12
    g = {"shape": list(R["pdms"].shape), "all_finite": bool(np.isfinite(R["pdms"]).all() and np.isfinite(s).all()), "all_valid": bool(R["valid"].all()),
         "max_abs_diff": float(d.max()), "RED_perturbed_EP_max_abs_diff": float(np.abs(fb - R["pdms"]).max())}
    g["PASS"] = bool(g["shape"] == [N_TOKENS, 64] and g["all_finite"] and g["all_valid"] and g["max_abs_diff"] <= 1e-9 and g["RED_perturbed_EP_max_abs_diff"] > 1e-9)
    return g


def g6_fp32(R):
    """G6-FP32 (Master Mind addition): the dump ran in fp32. Under bf16 autocast raw plan coordinates sit on a 0.25 m grid at 32-64 m, which moves the 0.2 s chord v4
    in 1.25 m/s steps and corrupts v4/v0 < 0.85. FAIL if more than 1 % of x[19] values in [32, 64) m are exact multiples of 0.25 (u400's fp32 reference: 0 of 16,712).
    Needs >= 500 values in the band, else the check cannot see (RED). RED arm: the same trajectories snapped to the 0.25 m grid must FAIL."""
    def frac(traj):
        x = np.asarray(traj)[..., 19, 0].astype(np.float32).ravel()
        x = x[(x >= 32) & (x < 64)]
        return int(x.size), (float((x * 4 == np.round(x * 4)).mean()) if x.size else float("nan"))
    n, f = frac(R["traj"])
    nq, fq = frac(np.round(np.asarray(R["traj"], np.float32) * 4) / 4)
    g = {"n_values_in_32_64m": n, "share_on_0.25m_grid": f, "bar": 0.01, "RED_snapped_n": nq, "RED_snapped_share": fq}
    g["PASS"] = bool(n >= 500 and f <= 0.01 and nq >= 500 and fq > 0.01)
    return g


def g6_argv(config_json, R):
    """G6-ARGV: the P6 run's config.json shows labeller = online, label_backend = pdm, nav_mode = command; the dump shows rule navsim_v1 and 6 score outputs.
    Quoted into the result. A red here means H-P6-SPEED-1 is VOID (the card says so), not merely 'no verdict'."""
    if not config_json or not os.path.exists(config_json):
        return {"PASS": False, "why": "NOT EVALUATED: no --run-config (the P6 run's config.json)"}
    c = json.load(open(config_json, encoding="utf-8"))
    a = c.get("args", {})
    q = {"labeller": a.get("labeller"), "label_backend": a.get("label_backend"), "nav_mode": a.get("nav_mode"), "meta_nav_mode": c.get("meta", {}).get("nav_mode"),
         "metric_cache": a.get("metric_cache"), "yaw_wrap": a.get("yaw_wrap"), "epochs": a.get("epochs"), "argv": c.get("argv")}
    q["dump_rule"], q["dump_n_outputs"], q["dump_nav_mode"] = R["meta"].get("rule"), R["n_outputs"], R["meta"].get("nav_mode")
    g = {"quoted": q}
    g["PASS"] = bool(q["labeller"] == "online" and q["label_backend"] == "pdm" and q["nav_mode"] == "command" and q["dump_rule"] == "navsim_v1"
                     and q["dump_n_outputs"] == 6 and q["dump_nav_mode"] == "command")
    if not g["PASS"]:
        g["VOID_if_not_online"] = bool(q["labeller"] != "online")
    return g


def navtrain_logs():
    t = open(NAVTRAIN_YAML, encoding="utf-8").read()
    return {ln.strip().strip("'\" ") for ln in t.split("log_names:")[1].splitlines() if ln.strip().startswith("- ")}


def g6_disjoint(R, train_logs_file):
    """G6-DISJOINT: none of the C3 tokens' logs is in P6's training token list. Two lists: (1) splits/navtrain.yaml (the scene filter, local) and
    (2) the pod's actual training logs (`ls` of the metric cache = the logs of --targets), required."""
    logs = set(R["log"])
    nt = navtrain_logs()
    g = {"n_c3_logs": len(logs), "overlap_navtrain_yaml": sorted(logs & nt)}
    if not train_logs_file or not os.path.exists(train_logs_file):
        g["PASS"] = False
        g["why"] = "NOT EVALUATED: no --train-logs (pod: ls <metric_cache> | sort > train_logs.txt)"
        return g
    tl = {ln.strip() for ln in open(train_logs_file, encoding="utf-8") if ln.strip()}
    g.update(n_train_logs=len(tl), overlap_pod_train_logs=sorted(logs & tl))
    g["PASS"] = bool(not g["overlap_navtrain_yaml"] and not g["overlap_pod_train_logs"] and len(tl) > 100)
    return g


def g6_repro(ref_json, B=None):
    """G6-REPRO: this code on the u400 C3 table reproduces the pinned u400 reference leaf by leaf (max abs diff 0.0) and its legacy block reproduces
    p1_diag_all.json Q4 (<= 1e-12). RED arms: swapped pred/true flips the sign of every gap; argsort-of-argsort ranks in the tie block must NOT reproduce."""
    Du = run_control_u400()
    B = B or Boot(Du["log"])
    S = stats_block(Du, B)
    r3, _ = r3_block(Du, B)
    ref = json.load(open(ref_json))
    mine = {"R1_weighted_cruise_slow_share": S["R1_weighted_cruise_slow_share"], "R2_spearman_tie": S["R2_spearman_tie"], "R2_spearman_legacy": S["R2_spearman_legacy"],
            "R3_u400_without_lane_factor": r3, "weights": S["weights"]}
    skip = lambda d: {k: v for k, v in d.items() if not str(k).startswith("_")}      # noqa: E731
    mine["R2_spearman_tie"] = {h: skip(v) for h, v in mine["R2_spearman_tie"].items()}
    mine["R2_spearman_legacy"] = {h: skip(v) for h, v in mine["R2_spearman_legacy"].items()}
    diff = max_abs_diff({k: ref[k] for k in mine}, mine)
    p1 = json.load(open(SELP + "/p1_diag_all.json"))["Q4_within_token_spearman_vs_speed"]
    leg = S["R2_spearman_legacy"]
    d_leg = 0.0
    for h in HEADS:
        d_leg = max(d_leg, abs(leg[h]["pred_unpaired_mean"] - p1[h][0]), abs(leg[h]["true_unpaired_mean"] - p1["true_" + h][0]),
                    abs(leg[h]["n_pred"] - p1[h][1]), abs(leg[h]["n_true"] - p1["true_" + h][1]))
    sw = stats_block(Du, B, swap_pred_true=True)
    flips = all(np.sign(sw["R2_spearman_tie"][h]["gap_pred_minus_true"][0]) == -np.sign(S["R2_spearman_tie"][h]["gap_pred_minus_true"][0]) for h in HEADS)
    bug = stats_block(Du, B, rank_method="bug_argsort")
    bug_diff = max_abs_diff({h: skip(S["R2_spearman_tie"][h]) for h in HEADS}, {h: skip(bug["R2_spearman_tie"][h]) for h in HEADS})
    g = {"max_abs_diff_vs_pinned_reference": diff, "legacy_vs_p1_diag_all_max_abs_diff": d_leg, "RED_swap_flips_every_gap_sign": bool(flips),
         "RED_argsort_of_argsort_tie_block_max_abs_diff": bug_diff}
    g["PASS"] = bool(diff == 0.0 and d_leg <= 1e-12 and flips and bug_diff > 1e-6)
    return g, (Du, B, S, r3)


def max_abs_diff(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            return float("inf")
        return max([max_abs_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max([max_abs_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if a is None or b is None:
        return 0.0 if a is b else float("inf")
    return float(abs(float(a) - float(b)))


# ------------------------------------------------------------------------------------------------ P6 analysis (statistics + descriptive families)
def analyze_p6(R, Du=None, B=None):
    B = B or Boot(R["log"])
    Du = Du or run_control_u400()
    S = stats_block(R, B)
    r3, U = r3_block(Du, B)
    cru, zero = R["cruise"], R["stratum"] == "zero"
    pt6, b6, ok6, _ = B.wshare(S["_slow"], cru, zero)          # same weight order as the u400 vectors, for the paired deltas
    pA, bA, okA, _ = B.wshare(U["slow_banked"], cru, zero)
    pB, bB, okB, _ = B.wshare(U["slow_nolane"], cru, zero)

    def delta(bu, oku, pu):
        d = (b6 - bu)[ok6 & oku]
        return {"point": float(pt6 - pu), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    dA, dB = delta(bA, okA, pA), delta(bB, okB, pB)
    t = S["R2_spearman_tie"]
    stat = dict(G_TTC=t["TTC"]["gap_pred_minus_true"][0], G_NC=t["NC"]["gap_pred_minus_true"][0], S_TTC=t["TTC"]["pred_unpaired_mean"], S_NC=t["NC"]["pred_unpaired_mean"],
                T_TTC=t["TTC"]["true_unpaired_mean"], T_NC=t["NC"]["true_unpaired_mean"], share=S["R1_weighted_cruise_slow_share"]["point"], dA_hi=dA["hi"], dB_hi=dB["hi"])
    out = {"statistics": {k: v for k, v in S.items() if not k.startswith("_")}, "delta_vs_u400_banked": dA, "delta_vs_u400_without_lane": dB,
           "u400_without_lane_control": r3, "decision_inputs": stat, "decision": decide(stat), "descriptive": describe(R, B, S)}
    return out


def describe(R, B, S):
    """Section 6 (DESCRIPTIVE; can never change the verdict). Four families, tier T0."""
    N = len(R["token"])
    r = np.arange(N)
    ratio = ratio_of(R)
    cru, zero = R["cruise"], R["stratum"] == "zero"
    pick = R["pick"]
    rp = ratio[r, pick]
    sub, val = R["sub"], R["valid"]
    d = {"tier": "T0 (open loop, one step, logged ego history); nothing here is a driving claim"}
    over = (rp > OVERSPEED_RATIO).astype(float)
    po, bo, oko, _ = B.wshare(over, cru, zero)
    d["LONGITUDINAL"] = {"slow_share_weighted": S["R1_weighted_cruise_slow_share"], "slow_share_unweighted": float(S["_slow"][cru].mean()),
                         "overspeed_share_weighted": {"point": float(po), "lo": ci(bo, oko)[0], "hi": ci(bo, oko)[1]}, "overspeed_share_unweighted": float(over[cru].mean()),
                         "plan_v4_minus_gt_v4_mps_cruise": B.mean(R["v4_all"][r, pick] - R["v4_gt"], cru & R["gt_valid"]), "mean_plan_v4_over_v0_cruise": B.mean(rp, cru),
                         "u400_references": "plan v4 minus GT v4 -0.535 [-1.391, +0.221] m/s (refe_ego_probe_20261106/RESULT.md); mean plan v4/v0 0.934 [0.803, 1.054] on A0 (INHERITED)"}
    prop = R["props8"][r, pick].astype(np.float64)
    ok_h = np.isfinite(R["human8"]).all((1, 2)) & (np.asarray(R["frame_ctrl_m"]) <= FC_TOL_M)
    ade = np.linalg.norm(prop[..., :2] - R["human8"][..., :2], axis=-1).mean(-1)
    d["LATERAL"] = {"ade_to_human_future_m_cruise": B.mean(ade, cru & ok_h), "n_ade_masked_by_frame_control": int((cru & ~ok_h).sum()),
                    "abs_y_at_4s_m_cruise": B.mean(np.abs(R["traj"][r, pick, 19, 1].astype(np.float64)), cru),
                    "dac_failure_rate_cruise": B.mean((sub[r, pick, 1] == 0).astype(float), cru), "ddc_mean_cruise": B.mean(sub[r, pick, 5], cru),
                    "note": "what the proposals allow, not tracking; ADE masks tokens whose frame control exceeds 1 mm"}
    d["SAFETY"] = {"nc_failure_rate_pick_cruise": B.mean((sub[r, pick, 0] == 0).astype(float), cru), "ttc_failure_rate_pick_cruise": B.mean((sub[r, pick, 3] == 0).astype(float), cru),
                   "changed_picks_vs_u400": "n/a: P6 and u400 propose different sets, an index comparison is meaningless; the lane-free re-pick count is in u400_without_lane_control"}
    d["TACTICAL"] = {"status": "n/a", "n": 0, "reason": "no manoeuvre-decision label on these tokens; REFe emits no tactical output here"}
    d["STRATEGIC"] = {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md item 25; not a pass, not omitted"}
    d["p1_blocks"] = p1_blocks(R, B)
    return d


def p1_blocks(R, B):
    """Q1, Q2, Q3, Q4 of p1_diag.py, scope 'all', for 6 (P6) or 7 (control) outputs; tie-correct ranks live in stats_block (Q4 rank tables are there)."""
    N = len(R["token"])
    r = np.arange(N)
    ratio = ratio_of(R)
    pick = R["pick"]
    pdms, sub, val = R["pdms"], R["sub"], R["valid"]
    L = np.asarray(R["logits"]).astype(np.float64)
    lane = R["n_outputs"] == 7
    P = 1.0 / (1.0 + np.exp(-L))
    agg = sc.agg_np(L, np.float64, lane=lane)
    cru = R["cruise"]
    rp = ratio[r, pick]
    pop = cru & (rp < SLOW_RATIO)
    F = val & (ratio >= 0.9)
    pp = pdms[r, pick]
    res = {"n_cruise": int(cru.sum()), "n_pop_slow_picks": int(pop.sum())}
    res["Q1"] = {"share_with_any_fast": B.mean(F.any(1).astype(float), pop),
                 "share_fast_pdms_ge_pick": B.mean((F & (pdms >= pp[:, None] - 1e-12)).any(1).astype(float), pop),
                 "share_fast_pdms_gt_pick": B.mean((F & (pdms > pp[:, None] + 1e-12)).any(1).astype(float), pop),
                 "pick_pdms_zero_share": B.mean((pp == 0).astype(float), pop)}
    f_sc = np.argmax(np.where(F, agg, -1e9), 1)
    popf = pop & F.any(1)
    names = list(SUBN[:5]) + (["lane"] if lane else [])
    w = np.array(sc.V1_W)
    lp, lf = np.log(np.maximum(P[r, pick], 1e-300)), np.log(np.maximum(P[r, f_sc], 1e-300))

    def avg(k):
        return (P[r, k, 2] * w[0] + P[r, k, 3] * w[1] + P[r, k, 4] * w[2]) / w.sum()
    ap_, af = avg(pick), avg(f_sc)
    dd = ap_ - af
    safe = np.where(np.abs(dd) > 1e-12, dd, 1.0)
    d_avg = np.log(ap_) - np.log(af)
    terms = {"NC": lp[:, 0] - lf[:, 0], "DAC": lp[:, 1] - lf[:, 1]}
    for k, j in (("EP", 2), ("TTC", 3), ("C", 4)):
        terms[k] = np.where(np.abs(dd) > 1e-12, d_avg * (w[j - 2] * (P[r, pick, j] - P[r, f_sc, j]) / w.sum()) / safe, 0.0)
    if lane:
        terms["lane"] = lp[:, 6] - lf[:, 6]
    T = np.stack([terms[k] for k in names], 1)
    dom = np.argmax(T, 1)
    res["Q2_f_sc"] = {"n_popf": int(popf.sum()), "dominant_term_counts": {k: int(((dom == i) & popf).sum()) for i, k in enumerate(names)},
                      "dlog_agg_by_term": {k: B.mean(terms[k], popf) for k in names}}
    h8 = R["human8"].astype(np.float64)
    ade = np.linalg.norm(R["props8"].astype(np.float64)[..., :2] - h8[:, None, :, :2], axis=-1).mean(-1)
    hn = np.argmin(np.where(val, ade, 1e9), 1)
    res["Q3_human_nearest"] = {"ade_m": B.mean(ade[r, hn], pop), "v4_over_v0": B.mean(ratio[r, hn], pop), "true_pdms": B.mean(pdms[r, hn], pop),
                               "true_pdms_minus_pick": B.mean(pdms[r, hn] - pp, pop)}
    bins = [0, 0.5, 0.7, 0.85, 0.9, 1.0, 1.1, 1.3, 9]
    cm = cru[:, None] & val
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = cm & (ratio >= lo) & (ratio < hi)
        row = {"bin": f"[{lo},{hi})", "n": int(m.sum()), "true_pdms": float(pdms[m].mean()) if m.any() else None, "agg": float(agg[m].mean()) if m.any() else None}
        for j in range(R["n_outputs"]):
            nm = (SUBN + ("lane",))[j]
            row["pred_" + nm] = float(P[..., j][m].mean()) if m.any() else None
            if j < 6:
                row["true_" + SUBN[j]] = float(sub[..., j][m].mean()) if m.any() else None
        cal.append(row)
    res["Q4_calibration_by_speed_bin"] = cal
    return res


# ------------------------------------------------------------------------------------------------ the full read
def full_read(a):
    os.makedirs(a.out, exist_ok=True)
    R = run_from_files(a.dump, a.table)
    gates = {}
    gates["G6-PIN"] = g6_pin(code_pin_path=os.path.join(HERE, "CODE_PIN.json"))
    gates["G6-ARGV"] = g6_argv(a.run_config, R)
    gates["G6-DISJOINT"] = g6_disjoint(R, a.train_logs)
    gates["G6-0"] = g6_0(R)
    gates["G6-1"] = g6_1(R, a.p6_seam)
    gates["G6-2"] = g6_2(R)
    gates["G6-3"] = g6_3(R, a.p6_csv)
    gates["G6-4"] = g6_4(R)
    gates["G6-FP32"] = g6_fp32(R)
    gates["G6-REPRO"], (Du, B, _S, _r3) = g6_repro(a.u400_reference)
    if a.expect_ckpt_sha:
        gates["G6-CKPT"] = {"dump_ckpt_sha256": R["meta"].get("ckpt_sha256"), "expected": a.expect_ckpt_sha, "PASS": R["meta"].get("ckpt_sha256") == a.expect_ckpt_sha}
    red = [k for k, g in gates.items() if not g["PASS"]]
    res = {"prereg": "PREREG_P6_SCORER_SPEED.md", "prereg_sha256": PREREG_SHA, "tier": "T0", "ckpt": R["meta"], "gates": gates, "red_gates": red}
    void = bool(gates["G6-ARGV"].get("VOID_if_not_online"))
    an = analyze_p6(R, Du, B)
    if red:
        res["verdict"] = {"VERDICT": "INSTRUMENT FAILURE", "why": f"red gates: {red}", "H_P6_SPEED_1_VOID": void,
                          "note": "no verdict and no number quoted over a red gate (section 3); the computed values are kept under `unquotable`"}
        res["unquotable"] = an
    else:
        res["verdict"] = an["decision"]
        res.update(analysis=an)
    res["ambiguities_applied"] = AMBIG_NOTES
    json.dump(res, open(os.path.join(a.out, "p6_speed_result.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "P6_SPEED_RESULT.md"), "w", encoding="utf-8").write(render_md(res))
    print(f"VERDICT {res['verdict']['VERDICT']}  red gates: {red}")
    return 0 if not red else 2


AMBIG_NOTES = [
    "S_h is the mean of rho(pred, speed) over ALL CRUISE tokens where defined (card section 4). u400's -0.546 / -0.491 in the card's section 2 table are the PAIRED-subset values; "
    "the all-token S_h is -0.479 (TTC) and -0.443 (NC). Both are reported; (B) uses S_h as defined in section 4.",
    "Sub-cases N2 and N3 can overlap (A holds, B fails, C holds); every matching sub-case is listed. A not-proven outcome matching none of N1-N3 is labelled N-unnamed.",
    "G6-DISJOINT is evaluated against splits/navtrain.yaml (local) AND the pod's metric-cache log list (the logs of --targets); both must be disjoint.",
    "A gate that could not be evaluated (missing input) is RED, so the read ends in INSTRUMENT FAILURE rather than a partial verdict.",
]


def render_md(res):
    L = ["# P6_SPEED_RESULT (H-P6-SPEED-1) -- tier T0, one checkpoint, deterministic planner", "",
         f"- checkpoint: `{res['ckpt'].get('ckpt')}` sha256 `{res['ckpt'].get('ckpt_sha256')}`", f"- prereg sha256 `{res['prereg_sha256']}`", "",
         f"## VERDICT: **{res['verdict']['VERDICT']}**", ""]
    v = res["verdict"]
    if res["red_gates"]:
        L += [f"INSTRUMENT FAILURE. Red gates: {res['red_gates']}. No number below is quotable.", ""]
        if v.get("H_P6_SPEED_1_VOID"):
            L += ["**H-P6-SPEED-1 is VOID: P6 did not train with `--labeller online`.**", ""]
    else:
        L += [f"- conditions: {v['conditions']}", f"- sub-cases: {v['subcases']}", f"- inputs: {v['inputs']}", ""]
    L += ["## Gates", "", "| gate | PASS | detail |", "|---|---|---|"]
    for k, g in res["gates"].items():
        det = {kk: vv for kk, vv in g.items() if kk not in ("PASS", "rows", "quoted")}
        L.append(f"| {k} | {g['PASS']} | `{json.dumps(det, default=float)[:400]}` |")
    if "G6-ARGV" in res["gates"] and "quoted" in res["gates"]["G6-ARGV"]:
        L += ["", "G6-ARGV quoted: `" + json.dumps(res["gates"]["G6-ARGV"]["quoted"], default=float)[:1500] + "`"]
    an = res.get("analysis") or res.get("unquotable")
    if an and not res["red_gates"]:
        s = an["statistics"]
        L += ["", "## Statistics (paired log-cluster bootstrap, B=10,000, seed 20260927, 95 % percentile)", "", "| head | S_h (all CRUISE) | T_h | pred (paired) | true (paired) | G_h gap [CI] | n paired |", "|---|---|---|---|---|---|---|"]
        for h, b in s["R2_spearman_tie"].items():
            g = b["gap_pred_minus_true"]
            L.append(f"| {h} | {b['pred_unpaired_mean']:+.3f} (n={b['n_pred']}) | {b['true_unpaired_mean']:+.3f} (n={b['n_true']}) | {b['pred'][0]:+.3f} | {b['true'][0]:+.3f} | {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] | {b['n_tokens_paired']} |")
        r1 = s["R1_weighted_cruise_slow_share"]
        L += ["", f"Weighted CRUISE slow-pick share P6: **{r1['point']:.3f}** [{r1['lo']:.3f}, {r1['hi']:.3f}] (u400 0.412 [0.275, 0.569])",
              f"Delta vs u400 banked: {an['delta_vs_u400_banked']}", f"Delta vs u400 without lane: {an['delta_vs_u400_without_lane']}", "",
              "## Descriptive (cannot change the verdict)", "", "```", json.dumps({k: an["descriptive"][k] for k in ("LONGITUDINAL", "LATERAL", "SAFETY", "TACTICAL", "STRATEGIC")}, indent=1, default=float), "```"]
    L += ["", "## Limits (prereg section 7)", "- One checkpoint, deterministic planner: no claim about training or inference variance (H-ESTIM-SEED-1). HOLDS means this checkpoint has the property, not that online labelling produces it.",
          "- P6 has no lane factor; the lane-free u400 control bounds that part.", "- 154 CRUISE tokens over 51 logs.", "", "## Ambiguities applied", *[f"- {x}" for x in res["ambiguities_applied"]]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--table")
    ap.add_argument("--out")
    ap.add_argument("--run-config")
    ap.add_argument("--train-logs")
    ap.add_argument("--p6-seam")
    ap.add_argument("--p6-csv")
    ap.add_argument("--expect-ckpt-sha")
    ap.add_argument("--u400-reference", default=HERE + "/u400_reference.pinned.json")
    a = ap.parse_args(argv)
    if not (a.dump and a.table and a.out):
        ap.error("--dump --table --out are required (the control is p6_control.py)")
    return full_read(a)


if __name__ == "__main__":
    sys.exit(main())
