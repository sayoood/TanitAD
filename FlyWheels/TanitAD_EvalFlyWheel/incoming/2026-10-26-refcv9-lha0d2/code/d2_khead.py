"""d2_khead.py -- PREREG_LHA0_D2 s3, K-HEAD-D2: the frozen D2 head's decel AUC on the EXACT K-HEAD eval139 window population, against LH's 0.7305.

REFUSES to run unless PREREG_LHA0_D2.md is sealed (PREREG_LHA0_D2.sha256 matches the file), the frozen heads' sha256 and this package's code sha256 are recorded in the sealed text.
Order (the prereg's): (1) population + LH reproduction -- STOP if it does not reproduce 0.7305 / 2495 / 639 / 1856 / 136, BEFORE any D2 score exists;
(2) features (lha0_features_eval139.npz + the nominal-dt fallback for the 3 no-sidecar clips); (3) scores; (4) paired episode-cluster bootstrap (B 2000, seed 0);
(5) verdict from d2_common.clears_bar / verdict_word.  Reuses lha0_diag.prep's K-POP join verbatim (population, z_tac join, LH scoring).  CPU only.  No clip id is printed or written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_common as D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = D.ROOT
EVAL_DUMP = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0")
EVAL_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_eval_s0.npz")
EVAL_FEATS = D.LHA0 / "result" / "lha0_features_eval139.npz"
LH_SHA = "6fd3ea1e3700a15b6506176bc855355d0cc1eefb3d87d390584e791d98e1d8f6"
WANT = {"s6_valid": 3668, "compliant": 3251, "gt_decel": 639, "gt_cruise": 1856, "union_n": 2495, "n_episodes": 136}
WANT_LH_AUC, WANT_LH_CI = 0.7305, [0.6749, 0.7855]
WANT_FALLBACK = {"windows": 102, "clips": 3, "union_windows": 71}
NOMINAL_DT = 0.1


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def check_registration() -> dict:
    pre, shaf = ROOT / "PREREG_LHA0_D2.md", ROOT / "PREREG_LHA0_D2.sha256"
    if not shaf.exists():
        raise SystemExit("[d2_khead] REFUSING: PREREG_LHA0_D2.sha256 does not exist -- the prereg is not sealed")
    lines = shaf.read_text(encoding="utf-8").splitlines()
    reg = lines[0].split()[0]
    got = D.sha256_file(pre)
    if len(reg) != 64 or got != reg:
        raise SystemExit(f"[d2_khead] REFUSING: PREREG_LHA0_D2.md sha256 {got} != sealed {reg}")
    txt = pre.read_text(encoding="utf-8")
    pins = {}
    for rel in ("D2_FROZEN.json", "D2NOACC_FROZEN.json", "code/d2_common.py", "code/d2_fit.py", "code/d2_khead.py", "code/d2_selftest.py"):
        s = D.sha256_file(ROOT / rel)
        if s not in txt:
            raise SystemExit(f"[d2_khead] REFUSING: {rel} sha256 {s} is not recorded in the sealed prereg")
        pins[rel] = s
    return {"prereg_sha256": reg, "seal_line_2": lines[1] if len(lines) > 1 else "", "pins": pins}


def load_population() -> dict:
    import k_head as KH
    import l2_common as C
    import lh_fit as LF
    from l1_fit_score import load_bank
    Ee = C.load_train_dump(EVAL_DUMP, "eval")
    valid, comp = np.asarray(Ee["valid_s6"], bool), np.asarray(Ee["compliant"], bool)
    m = valid & comp
    ge = np.asarray(Ee["gt"])
    ui = np.flatnonzero(m & ((ge == "decel") | (ge == "cruise")))
    zb = load_bank(EVAL_BANK)
    jz = np.array([zb["idx"].get((int(c), int(w)), -1) for c, w in zip(Ee["clip"][ui], Ee["ws"][ui])])
    assert (jz >= 0).all(), int((jz < 0).sum())
    LH = LF.load_frozen(D.LONHEAD / "LH_FROZEN.pt", LH_SHA)
    v0 = np.asarray(zb["v0_in"], np.float64)
    lg = LF.predict_logits6(LH, zb["z"][jz], v0[jz])
    s_lh, _ = KH.renorm_decel(KH.softmax(lg))
    y = (ge[ui] == "decel")
    auc_lh = KH.auc_np(y, s_lh)
    pop = {"s6_valid": int(valid.sum()), "compliant": int(m.sum()), "gt_decel": int((m & (ge == "decel")).sum()), "gt_cruise": int((m & (ge == "cruise")).sum()),
           "union_n": int(len(ui)), "n_episodes": int(len(np.unique(Ee["eid"][ui])))}
    return {"Ee": Ee, "ui": ui, "zb": zb, "jz": jz, "v0": v0, "lg": lg, "s_lh": s_lh, "y": y, "auc_lh": float(auc_lh), "pop": pop, "eid": np.asarray(Ee["eid"])[ui]}


def eval_features(zb: dict) -> tuple:
    """a0 / psi_dot0 for every eval bank window: the LHA0 package's table (per-clip dt), plus the documented nominal-dt (0.1 s) fallback for the clips with no clock-sidecar row,
    derived from the same ep_poses with the SAME backward-difference formulas at the SAME anchor k = ws.  Returns (a0, psi, fallback_mask)."""
    ft = np.load(EVAL_FEATS)
    assert np.array_equal(ft["e_i"].astype(np.int64), zb["e_i"]) and np.array_equal(ft["ws"].astype(np.int64), zb["ws"]), "feature table order != bank order"
    a0, psi = ft["a0"].astype(np.float64).copy(), ft["psi_dot0"].astype(np.float64).copy()
    fb = ~(np.isfinite(a0) & np.isfinite(psi))
    assert np.array_equal(fb, ~np.isfinite(ft["dt"].astype(np.float64))), "fallback rows != the rows without a dt"
    poses = {int(c): np.load(EVAL_DUMP / "decisions" / f"ep{int(c):03d}.npz")["ep_poses"].astype(np.float64) for c in np.unique(zb["e_i"][fb])}
    v0 = np.asarray(zb["v0_in"], np.float64)
    for i in np.flatnonzero(fb):
        c, k = int(zb["e_i"][i]), int(zb["ws"][i])
        p = poses[c]
        assert k - 1 >= 0 and abs(p[k, 3] - v0[i]) <= 1e-9, (k, abs(p[k, 3] - v0[i]))      # the anchor rule k = ws holds exactly
        a0[i] = (p[k, 3] - p[k - 1, 3]) / NOMINAL_DT
        psi[i] = wrap(p[k, 2] - p[k - 1, 2]) / NOMINAL_DT
    assert np.isfinite(a0).all() and np.isfinite(psi).all()
    return a0, psi, fb


def main() -> int:
    t0 = time.time()
    import k_head as KH
    reg = check_registration()
    (ROOT / "result").mkdir(exist_ok=True)
    # ---- (1) population + LH reproduction: STOP before any D2 score exists ----
    P = load_population()
    pop_ok = all(P["pop"][k] == v for k, v in WANT.items())
    lh_ok = abs(P["auc_lh"] - WANT_LH_AUC) < 5e-5
    if not (pop_ok and lh_ok):
        stop = {"verdict": "STOP", "reason": "LH did not reproduce K-HEAD on the reconstructed population", "population": P["pop"], "want": WANT, "LH_auc": P["auc_lh"], "want_LH_auc": WANT_LH_AUC,
                "registration": reg}
        (ROOT / "result" / "LHA0_D2_KHEAD.json").write_text(json.dumps(stop, indent=1), encoding="utf-8")
        print("ZZKHEADZZ STOP", json.dumps(stop)[:400], flush=True)
        return 2
    zb, jz, ui, y, eid = P["zb"], P["jz"], P["ui"], P["y"], P["eid"]
    # ---- (2) features ----
    a0, psi, fb = eval_features(zb)
    fb_u = fb[jz]
    fallback = {"windows": int(fb.sum()), "clips": int(len(np.unique(zb["e_i"][fb]))), "union_windows": int(fb_u.sum())}
    fb_ok = fallback == WANT_FALLBACK
    A0, PSI, V0 = a0[jz], psi[jz], P["v0"][jz]
    # ---- (3) scores ----
    H = D.load_frozen(ROOT / "D2_FROZEN.json", reg["pins"]["D2_FROZEN.json"])
    HN = D.load_frozen(ROOT / "D2NOACC_FROZEN.json", reg["pins"]["D2NOACC_FROZEN.json"])
    a0p, psip = D.permute_within_episode(zb["e_i"], [a0, psi], seed=0)
    lh_lo = D.lh_logit(P["lg"])
    s_d2 = D.head_logit(H, A0, V0, PSI)
    scores = {"LH": P["s_lh"], "D2": s_d2, "constant": np.zeros(len(ui)), "neg_a0_raw": -A0, "D2_permuted": D.head_logit(H, a0p[jz], V0, psip[jz]),
              "fuse_D2_LH": D.zscore(s_d2) + D.zscore(lh_lo), "D2NOACC": D.head_logit(HN, A0, V0, PSI), "D2NOACC_permuted": D.head_logit(HN, a0p[jz], V0, psip[jz]),
              "LH_logit_scale": lh_lo}
    # ---- (4) bootstrap: full population, and without the no-sidecar clips ----
    R = D.paired_boot(y, eid, scores)
    keep = ~fb_u
    sub = {k: v[keep] for k, v in scores.items()}
    sub["fuse_D2_LH"] = D.zscore(sub["D2"]) + D.zscore(sub["LH_logit_scale"])
    Rs = D.paired_boot(y[keep], eid[keep], sub)
    # ---- (5) verdict ----
    def arm_rows(R_):
        return {k: {"auc": D.pr(R_["pt"][k]), "ci95": [D.pr(x) for x in KH.ci(R_["bt"][k])]} for k in R_["pt"]}

    def bar(R_, a):
        d = R_["pt"][a] - R_["pt"]["LH"]
        lo = KH.ci(R_["bt"][a] - R_["bt"]["LH"])[0]
        return {"delta_vs_LH": D.diff(R_, a, "LH"), "delta_unrounded": float(d), "ci_lo_unrounded": float(lo), "clears_bar": D.clears_bar(d, lo), "gap_to_bar": D.pr(max(D.BAR_DELTA - d, 0.0))}

    b_d2, b_perm = bar(R, "D2"), bar(R, "D2_permuted")
    const_ok = bool(R["pt"]["constant"] == 0.5)
    lh_ci_repro = [D.pr(x) for x in KH.ci(R["bt"]["LH"])]
    controls = {"K-LH-REPRO_pop": pop_ok, "K-LH-REPRO_auc": {"auc": D.pr(P["auc_lh"], 6), "want": WANT_LH_AUC, "pass": lh_ok}, "K-CONST_0.5": {"auc": R["pt"]["constant"], "pass": const_ok},
                "K-WAUC_unit_weights": R["unit_ok"], "K-FALLBACK": {"got": fallback, "want": WANT_FALLBACK, "pass": fb_ok},
                "LH_CI_vs_K_HEAD_json (beside, not gating)": {"got": lh_ci_repro, "want": WANT_LH_CI, "matches_to_1e-4": bool(np.allclose(lh_ci_repro, WANT_LH_CI, atol=1.1e-4))}}
    controls_ok = bool(pop_ok and lh_ok and const_ok and all(R["unit_ok"].values()) and fb_ok)
    word = D.verdict_word(controls_ok, b_d2["clears_bar"], b_perm["clears_bar"])
    d = {k: D.diff(R, k, "LH") for k in ("D2", "D2_permuted", "neg_a0_raw", "fuse_D2_LH", "D2NOACC", "D2NOACC_permuted", "constant")}
    out = {"tool": "d2_khead.py", "prereg": "PREREG_LHA0_D2.md s3", "registration": reg, "verdict": word, "bar": "GO iff AUC(D2) - AUC(LH) >= +0.030 and the paired episode-cluster bootstrap 95% CI excludes 0",
           "tier": "T0 label-level diagnostic on logged eval139 frames (heads read the frozen trunk's t0 z_tac / measured t0 ego state; no planner, no roll)",
           "estimator": "Mann-Whitney AUC (average-rank ties); CI = EPISODE-RESAMPLING bootstrap (B 2000, seed 0), scores held fixed, every arm on the SAME draws (paired). Answers 'another draw of "
                        "eval EPISODES' ONLY; blind to training (one refcv8 seed, one head fit) and inference variance (heads deterministic; z_tac bit-equal across inference seeds).",
           "n": {"windows": R["n"], "decel": R["n_pos"], "episodes": R["n_episodes"], "d_D2": 3, "d_LH": 513}, "population": P["pop"],
           "primary": {"D2": b_d2, "D2_permuted_control": b_perm}, "arms_full_population": arm_rows(R), "paired_vs_LH_full": d,
           "contrasts_full": {"D2_minus_neg_a0_raw": D.diff(R, "D2", "neg_a0_raw"), "D2_minus_D2_permuted": D.diff(R, "D2", "D2_permuted"), "fuse_minus_D2": D.diff(R, "fuse_D2_LH", "D2"),
                              "fuse_minus_LH": D.diff(R, "fuse_D2_LH", "LH"), "D2NOACC_minus_D2": D.diff(R, "D2NOACC", "D2"), "neg_a0_raw_minus_LH": D.diff(R, "neg_a0_raw", "LH")},
           "without_no_sidecar_clips": {"n": {"windows": Rs["n"], "decel": Rs["n_pos"], "episodes": Rs["n_episodes"]}, "arms": arm_rows(Rs), "D2_vs_LH": bar(Rs, "D2"),
                                        "D2_permuted_vs_LH": bar(Rs, "D2_permuted"), "note": "the bar reads the FULL population; this row is descriptive"},
           "nominal_dt_fallback": {**fallback, "rule": "dt = 0.1 s; a0 = (v[k]-v[k-1])/0.1, psi_dot0 = wrap(yaw[k]-yaw[k-1])/0.1 from the eval dump's ep_poses at k = ws (anchor exact); "
                                   "the fresh-split sensitivity (Bnom == B, LHA0_DIAG) is the only evidence it is immaterial"},
           "controls": controls, "controls_all_pass": controls_ok,
           "heads": {"D2": {"sha256": H["_sha256"], "lambda": H["lambda"], "w_std_units": H["w"], "b": H["b"], "n_fit_rows": H["n_fit_rows"]},
                     "D2NOACC": {"sha256": HN["_sha256"], "lambda": HN["lambda"], "w_std_units": HN["w"], "b": HN["b"], "n_fit_rows": HN["n_fit_rows"]}},
           "permutation": {"scope": "within each eval EPISODE over ALL its bank windows (4,754 total), ONE shared permutation of (a0, psi_dot0), episodes in sorted order, np.random.default_rng(0); v0 stays with its window",
                           "frac_a0_values_moved": D.pr(float((a0p != a0).mean()))},
           "wall_s": round(time.time() - t0, 1)}
    (ROOT / "result" / "LHA0_D2_KHEAD.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"ZZKHEADZZ {word} | D2 {R['pt']['D2']:.4f} LH {R['pt']['LH']:.4f} delta {b_d2['delta_vs_LH']} | perm delta {b_perm['delta_vs_LH']} | -a0 {R['pt']['neg_a0_raw']:.4f} | "
          f"n {R['n']} dec {R['n_pos']} ep {R['n_episodes']} | controls {controls_ok}", flush=True)
    return 0 if controls_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
