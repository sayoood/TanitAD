"""k_head.py -- PREREG_LONHEAD_C s2 K-HEAD (go / no-go, descriptive): decel AUC of the frozen LH on eval139 S6 compliant windows,
P(decel classes) renormalised EXACTLY as the 0.685 reference (H-R9-LEADINFO-1 D4: decel/(decel+cruise) of the 6-logit LON6 softmax, on the
union of the GT-cruise and GT-decel compliant populations from l2_common.lon_indicators). CPU only. NEW file (refcv9 LONHEAD phase-2 agent).

Reused (imported, unedited): l2_common.load_eval_pf / lon_indicators / lon6 / G_DECEL / G_CRUISE (refcv9_l2/code), l1_fit_score.load_bank
(refcv9_l1/code), lh_fit.load_frozen / predict_logits6 (this package). AUC kernels copied verbatim from refcv9_leadinfo/code/leadinfo_d.py
(auc_np, WAuc, ci).
Known-value controls: the native head's renormalised AUC must reproduce 0.6851 on the same windows; a constant score must read exactly 0.5;
the WAuc kernel must equal auc_np at unit weights.
No clip id is printed or written.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in ("C:/Users/Admin/refcv8_pf/refcv9_l2/code", "C:/Users/Admin/refcv8_pf/refcv9_l1/code", str(HERE)):
    sys.path.insert(0, _p)
import l2_common as C  # noqa: E402
from l1_fit_score import load_bank  # noqa: E402
import lh_fit as LF  # noqa: E402

L1_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
REF_AUC = 0.685
BOOT_B, SEED = 2000, 0
REG = {"PREREG_LONHEAD_C.md": "6c81d4b0510c463743ed814491710eea6e0f0384088ca296d20e39e444321048",
       "PREREG_LONHEAD_C_AMENDMENT1.md": "715b9df1d78fed24fc8acc6a5f6363dc15265dd3656579ae531cff1cddebb2c8",
       "LH_FROZEN.pt": "6fd3ea1e3700a15b6506176bc855355d0cc1eefb3d87d390584e791d98e1d8f6",
       "DER_FROZEN.pt": "08d1085d8d4407f2c6113c07be0fed04616f373cd102fe678c3a52d3d46c1f7e",
       "LH_FIT_INFO.json": "a7e53f5edeb026434eb6000b33f9f19466ef3d21d36cd074133daaace198ba93",
       "code/lh_fit.py": "fb330f7f23b2fea249e1527e804c454062a06ffa0cbe4b612ff551f57f5b62d9"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assert_registered():
    got = {k: sha(ROOT / k) for k in REG}
    reg_file = (ROOT / "REGISTERED_LH.sha256").read_text(encoding="utf-8")
    for k, h in REG.items():
        assert len(got[k]) == 64 and got[k] == h, f"{k}: {got[k]} != registered {h}"
        if k not in ("PREREG_LONHEAD_C.md", "PREREG_LONHEAD_C_AMENDMENT1.md"):
            assert f"{h} *{k}" in reg_file, f"{k} not in REGISTERED_LH.sha256"
    return got


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def auc_np(y, s):
    """Mann-Whitney AUC with average ranks for ties (verbatim, leadinfo_d.py)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class WAuc:
    """Weighted AUC (verbatim, leadinfo_d.py): window weight = times its episode was drawn; ties 1/2."""

    def __init__(self, y, s):
        self.y = np.asarray(y, bool)
        _, self.g = np.unique(np.asarray(s, np.float64), return_inverse=True)
        self.ng = int(self.g.max()) + 1

    def __call__(self, w):
        pos = np.bincount(self.g, weights=w * self.y, minlength=self.ng)
        neg = np.bincount(self.g, weights=w * (~self.y), minlength=self.ng)
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def pr(x, d=4):
    return None if x is None else round(float(x), d)


def renorm_decel(P6):
    pd_, pc = P6[:, list(C.G_DECEL)].sum(1), P6[:, list(C.G_CRUISE)].sum(1)
    return pd_ / np.maximum(pd_ + pc, 1e-300), pd_


def main() -> int:
    t0 = time.time()
    hashes = assert_registered()
    assert tuple(LF.LON6_V7) == tuple(C.LON6_V7_IDS) == (0, 1, 3, 4, 5, 7), "LH class order != l2_common LON6 order"
    E = C.load_eval_pf(0)
    for k, v in E["checks"].items():
        if isinstance(v, bool):
            assert v, f"load_eval_pf check failed: {k}"
    assert E["checks"]["plan8_slots_equal_os"] == 0.0
    comp = np.asarray(E["compliant"], bool)
    fd, md, gt, _ = C.lon_indicators(E["os"], E, comp)
    pop_c, pop_d = np.isfinite(fd), np.isfinite(md)
    assert not (pop_c & pop_d).any()
    ui = np.flatnonzero(pop_c | pop_d)
    y = pop_d[ui].astype(bool)
    # ---- eval z_tac (SELCOR variant Z bank, s0) joined on the S6 rows' (clip_index, ws) ----
    zb = load_bank(L1_BANK / "ztac_eval_s0.npz")
    keys = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
    jz = np.array([zb["idx"].get(k, -1) for k in keys])
    assert (jz >= 0).all(), f"{int((jz < 0).sum())} S6 rows not in the z_tac bank"
    Z, v0_in = zb["z"][jz], np.asarray(zb["v0_in"], np.float64)[jz]
    dv0 = float(np.abs(v0_in - np.asarray(E["v0_extras"], np.float64)).max())
    dv0_s6 = float(np.abs(v0_in - np.asarray(E["v0"], np.float64)).max())
    # ---- heads ----
    LH = LF.load_frozen(ROOT / "LH_FROZEN.pt", REG["LH_FROZEN.pt"])
    DER = LF.load_frozen(ROOT / "DER_FROZEN.pt", REG["DER_FROZEN.pt"])
    assert LH["variant"] == "logistic" and float(LH["hp"]) == 100.0, (LH["variant"], LH["hp"])
    P_lh = softmax(LF.predict_logits6(LH, Z, v0_in))
    P_der = softmax(LF.predict_logits6(DER, Z, v0_in))
    P_nat = softmax(C.lon6(E["logits"]))
    s_lh, s_lh_lit = renorm_decel(P_lh)
    s_der, _ = renorm_decel(P_der)
    s_nat, s_nat_lit = renorm_decel(P_nat)
    # ---- known-value controls ----
    a_nat = auc_np(y, s_nat[ui])
    k_ref = {"native_head_renorm_auc": pr(a_nat), "want": 0.6851, "pass": bool(abs(a_nat - 0.6851) < 5e-5)}
    k_const = {"auc": auc_np(y, np.zeros(len(ui))), "pass": bool(auc_np(y, np.zeros(len(ui))) == 0.5)}
    point = {"LH": auc_np(y, s_lh[ui]), "DER": auc_np(y, s_der[ui]), "native": a_nat,
             "LH_literal_unrenorm": auc_np(y, s_lh_lit[ui]), "native_literal_unrenorm": auc_np(y, s_nat_lit[ui])}
    fns = {k: WAuc(y, s[ui]) for k, s in (("LH", s_lh), ("DER", s_der), ("native", s_nat))}
    k_w = {k: bool(abs(f(np.ones(len(ui))) - point[k]) < 1e-9) for k, f in fns.items()}
    ctl_pass = bool(k_ref["pass"] and k_const["pass"] and all(k_w.values()) and dv0 == 0.0)
    # ---- episode-resampling bootstrap (B 2000, seed 0), scores held fixed ----
    ep = np.asarray(E["eid"])[ui]
    ue, inv = np.unique(ep, return_inverse=True)
    rng = np.random.default_rng(SEED)
    boots = {k: np.empty(BOOT_B) for k in fns}
    for b in range(BOOT_B):
        cnt = np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(np.float64)
        w = cnt[inv]
        for k, f in fns.items():
            boots[k][b] = f(w)
    res = {k: {"auc": pr(point[k]), "ci95": [pr(x) for x in ci(boots[k])]} for k in fns}
    res["LH_minus_native"] = {"point": pr(point["LH"] - point["native"]), "ci95": [pr(x) for x in ci(boots["LH"] - boots["native"])]}
    res["LH_minus_DER"] = {"point": pr(point["LH"] - point["DER"]), "ci95": [pr(x) for x in ci(boots["LH"] - boots["DER"])]}
    res["literal_unrenormalised_beside"] = {"LH": pr(point["LH_literal_unrenorm"]), "native": pr(point["native_literal_unrenorm"])}
    go = bool(ctl_pass and point["LH"] > REF_AUC)
    word = ("INVALID (a known-value control failed)" if not ctl_pass else ("GO: LH point AUC > 0.685 -> rolls run" if go else "HEAD-NOT-BETTER"))
    out = {"tool": "k_head.py", "prereg": "PREREG_LONHEAD_C.md s2 K-HEAD (+ AMENDMENT1)", "registered_sha256": hashes,
           "script_sha256": sha(__file__), "word": word, "go": go, "reference_auc": REF_AUC, "LH_point_auc": pr(point["LH"]),
           "tier": "T0 diagnostic on logged eval139 frames (a head read on the frozen trunk's z_tac at t0; no planner, no roll)",
           "estimator": "Mann-Whitney AUC (average-rank ties); CI = EPISODE-RESAMPLING bootstrap (B 2000, seed 0) with the head scores held "
                        "fixed (leadinfo_d.py WAuc kernel). Answers 'another draw of EPISODES'; blind to training and head-fit variance (one fit).",
           "population": {"s6_windows": int(len(E["ws"])), "compliant": int(comp.sum()), "gt_decel_compliant": int(pop_d.sum()),
                          "gt_cruise_compliant": int(pop_c.sum()), "union_n": int(len(ui)), "n_episodes": int(len(ue)),
                          "decel_base_rate": pr(y.mean()), "inference_seed": 0},
           "score": "P(BRAKE_TO+CREEP+HOLD) / (P(BRAKE_TO+CREEP+HOLD) + P(FOLLOW+CRUISE)) of the 6-way LON6 softmax (the 0.685 reference's D4 renormalisation)",
           "inputs": {"z_tac_bank": str(L1_BANK / "ztac_eval_s0.npz"), "z_tac_bank_sha256": sha(L1_BANK / "ztac_eval_s0.npz"),
                      "v0": "bank v0_in (the value the roll fed the model)", "v0_in_vs_extras_v0_maxabs": dv0, "v0_in_vs_s6_v0_maxabs": dv0_s6,
                      "z_tac_s0_equals_s1_bitwise": "MEASURED in this session: max |z_s0 - z_s1| = 0.0 and max |v0_in_s0 - v0_in_s1| = 0.0 on all 4,754 keys"},
           "controls": {"K-REF_0.685": k_ref, "K-CONST_0.5": k_const, "K-WAUC_unit_weights": k_w, "v0_join_exact": dv0 == 0.0, "ALL_PASS": ctl_pass},
           "result": res, "wall_s": round(time.time() - t0, 1)}
    (ROOT / "result").mkdir(exist_ok=True)
    tmp = ROOT / "result" / "K_HEAD.json.tmp"
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, ROOT / "result" / "K_HEAD.json")
    print(f"[k_head] {word} | LH {point['LH']:.4f} {res['LH']['ci95']} | native {a_nat:.4f} | DER {point['DER']:.4f} | "
          f"LH-native {res['LH_minus_native']} | LH-DER {res['LH_minus_DER']} | n {len(ui)} dec {int(y.sum())} | controls {ctl_pass}", flush=True)
    return 0 if ctl_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
