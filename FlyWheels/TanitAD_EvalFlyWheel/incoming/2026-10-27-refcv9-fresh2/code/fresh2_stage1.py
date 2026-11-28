"""fresh2_stage1.py -- PREREG_FRESH2 s3: the T0 label-level read on CONFIRM (run ONCE), and its known-value SELF-TEST on the FIRST pull.

Reuses, IMPORTED and UNEDITED (the prereg says "exactly"):
  refcv9_lha0d2/code/d2_khead.py   load_population (K-HEAD population: l2_common.load_train_dump -> S6-valid & compliant & GT in {decel, cruise}; z_tac join;
                                   frozen LH scored as P(BRAKE_TO+CREEP+HOLD)/(that + P(FOLLOW+CRUISE))), eval_features (a0 / psi_dot0 table + nominal-dt fallback).
                                   Its module paths EVAL_DUMP / EVAL_BANK / EVAL_FEATS are re-pointed at run time (FRESH2, or the first pull for the self-test).
  refcv9_lha0d2/code/d2_common.py  load_frozen, head_logit, lh_logit, zscore, permute_within_episode, paired_boot (K-HEAD's bootstrap loop), diff.
  refcv9_lonhead/code/lh_fit.py    load_frozen, predict_logits6 (LH and LH_G);  k_head.py softmax, renorm_decel, auc_np, ci.
  refcv9_lonhead/code/lh_klabel.py trainer_lon_targets (the LON6 labels WITH the lead pass; used for the BESIDE LON6-GT reading only).
New here: the CONFIRM restriction, the two FRESH2 bars (+0.020, CI lower bound > 0 -- unrounded), the arm table, the report.

Arms (same windows): LH (LH_FROZEN.pt 6fd3ea1e...d8f6) | D2 (D2_FROZEN.json 69135b91...) | FUSE = z(D2 logit) + z(LH logit), z over the evaluated windows, no
fitted weight | LH_G (fresh2_lhg_fit.py, sha registered before this runs) | constant | FUSE_perm = FUSE with a0 and psi_dot0 permuted within each CONFIRM
episode over all its bank windows (np.random.default_rng(0), d2_common.permute_within_episode).  Beside, non-gating: neg_a0_raw, D2_perm, and the
whole table on the LON6-label GT reading (see READING_FRESH2.md).
Estimator: Mann-Whitney AUC (average-rank ties); paired episode-cluster bootstrap over CONFIRM episodes, B 2000, seed 0, every arm on the same draws.
  python fresh2_stage1.py --selftest    first pull: known values from refcv9_lha0 / refcv9_lha0d2 / refcv9_lonhead (no CONFIRM file is opened)
  python fresh2_stage1.py --run         CONFIRM, ONCE (refuses if result/FRESH2_STAGE1.json exists)
CPU only.  No clip id is printed or written.  NEW file (FRESH2 agent, 2026-10-27).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_lha0d2/code")
import d2_common as D  # noqa: E402  (puts tree stack, lonhead code, l2 code, l1 code on sys.path)
import d2_khead as K  # noqa: E402
import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parent.parent
LHA0 = Path("C:/Users/Admin/refcv8_pf/refcv9_lha0")
LHA0D2 = Path("C:/Users/Admin/refcv8_pf/refcv9_lha0d2")
LONHEAD = Path("C:/Users/Admin/refcv8_pf/refcv9_lonhead")
BAR = 0.020                                                         # PREREG_FRESH2 s3: F-1 / G-1, unrounded
PIN = {"PREREG_FRESH2.md": "956f87cdf88e6cbcd5c1ad824ad187e55f15796f23564b3b7d674cfd917dfebe",
       "LH_FROZEN.pt": "6fd3ea1e3700a15b6506176bc855355d0cc1eefb3d87d390584e791d98e1d8f6",
       "D2_FROZEN.json": "69135b9145e38e7b8b13fb1dac85ebda20b7db9b836c85032e7c4af2d25d2fc5",
       "d2_common.py": "73b0a6107c32ca6f8b22a6b16919a2fe5829bb5977f2512fcf4b42cfa171aec7",
       "d2_khead.py": "f5ca118e44dd1dcd4e6a16bda6430d187e5114ea841dac4bf28d77feed30cba7",
       "lh_fit.py": "fb330f7f23b2fea249e1527e804c454062a06ffa0cbe4b612ff551f57f5b62d9"}
FIRST = {"dump": "F:/l1c/tree/step48852ema_fresh/dump_s0", "bank": "F:/l1c/bank/ztac_fresh.npz", "feats": str(LHA0 / "result" / "lha0_features.npz")}
FRESH2 = {"dump": "F:/l1c/tree/step48852ema_fresh2/dump_s0", "bank": "F:/l1c/bank/ztac_fresh2.npz", "feats": "F:/l1c/fresh2/features/result/lha0_features.npz",
          "release": "F:/l1c/fresh2/kit_fresh2_lead/v9/v9_labels_eval.npz", "sidecar": "F:/l1c/kit_fresh2/data/clock_sidecar.jsonl",
          "windows": "F:/l1c/kit_fresh2/windows_s0.json", "split": "F:/l1c/fresh2/split.json"}
LHG = Path("F:/l1c/fresh2/lhg/LH_G_FROZEN.pt")
D_ARM = {"LH": 513, "D2": 3, "FUSE": 516, "LH_G": 513, "constant": 0, "FUSE_perm": 516, "neg_a0_raw": 1, "D2_perm": 3}
GATING = ("LH", "D2", "FUSE", "LH_G", "constant", "FUSE_perm")


def sha256_file(p) -> str:
    return D.sha256_file(p)


def sha12(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def pins() -> dict:
    got = {"PREREG_FRESH2.md": sha256_file(PKG / "PREREG_FRESH2.md"), "LH_FROZEN.pt": sha256_file(LONHEAD / "LH_FROZEN.pt"),
           "D2_FROZEN.json": sha256_file(LHA0D2 / "D2_FROZEN.json"), "d2_common.py": sha256_file(D.__file__), "d2_khead.py": sha256_file(K.__file__),
           "lh_fit.py": sha256_file(LONHEAD / "code" / "lh_fit.py")}
    bad = {k: got[k] for k in PIN if got[k] != PIN[k] or len(got[k]) != 64}
    if bad:
        raise SystemExit(f"[fresh2] PINNED sha256 MISMATCH -- STOP: {bad}")
    sealed = (PKG / "PREREG_FRESH2.sha256").read_text(encoding="utf-8").split()[0]
    if sealed != PIN["PREREG_FRESH2.md"]:
        raise SystemExit("[fresh2] PREREG_FRESH2.sha256 does not carry the sealed hash -- STOP")
    return got


def point_at(s: dict):
    K.EVAL_DUMP, K.EVAL_BANK, K.EVAL_FEATS = Path(s["dump"]), Path(s["bank"]), Path(s["feats"])


def restrict(P: dict, keep: np.ndarray) -> dict:
    """the K-HEAD population P restricted to the rows `keep` (a mask over P's union rows)."""
    Q = dict(P)
    for k in ("ui", "jz", "s_lh", "y", "eid", "lg"):
        Q[k] = np.asarray(P[k])[keep]
    return Q


def permuted_table(zb: dict, a0: np.ndarray, psi: np.ndarray, bank_rows: np.ndarray) -> tuple:
    """a0 / psi_dot0 permuted within each episode of `bank_rows` over ALL its bank windows (d2_khead's scope, restricted to the scored episodes):
    d2_common.permute_within_episode, episodes in sorted order, default_rng(0).  Rows outside `bank_rows` are NaN."""
    a0p, psip = np.full_like(a0, np.nan), np.full_like(psi, np.nan)
    pa, pp = D.permute_within_episode(zb["e_i"][bank_rows], [a0[bank_rows], psi[bank_rows]], seed=0)
    a0p[bank_rows], psip[bank_rows] = pa, pp
    return a0p, psip


def lh6_score(blob: dict, Z: np.ndarray, v0: np.ndarray) -> tuple:
    import k_head as KH
    import lh_fit as LF
    lg = LF.predict_logits6(blob, Z, v0)
    s, _ = KH.renorm_decel(KH.softmax(lg))
    return s, D.lh_logit(lg)


def arm_scores(Q: dict, zb: dict, a0, psi, a0p, psip, H: dict, LHGb: dict | None) -> dict:
    jz = Q["jz"]
    A0, PSI, V0 = a0[jz], psi[jz], np.asarray(zb["v0_in"], np.float64)[jz]
    lh_lo = D.lh_logit(Q["lg"])
    s_d2 = D.head_logit(H, A0, V0, PSI)
    s_d2p = D.head_logit(H, a0p[jz], V0, psip[jz])
    sc = {"LH": Q["s_lh"], "D2": s_d2, "FUSE": D.zscore(s_d2) + D.zscore(lh_lo), "constant": np.zeros(len(jz)),
          "FUSE_perm": D.zscore(s_d2p) + D.zscore(lh_lo), "neg_a0_raw": -A0, "D2_perm": s_d2p}
    if LHGb is not None:
        sc["LH_G"] = lh6_score(LHGb, zb["z"][jz], V0)[0]
    for k, v in sc.items():
        assert np.isfinite(v).all(), f"non-finite scores in arm {k}"
    return sc


def bar(R: dict, a: str) -> dict:
    import k_head as KH
    d = float(R["pt"][a] - R["pt"]["LH"])
    lo, hi = KH.ci(R["bt"][a] - R["bt"]["LH"])
    clears = bool(np.isfinite(d) and np.isfinite(lo) and d >= BAR and lo > 0.0)
    return {"delta_vs_LH": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)], "delta_unrounded": d, "ci_lo_unrounded": float(lo), "clears_bar": clears,
            "gap_to_bar": round(max(BAR - d, 0.0), 4)}


def read_table(Q: dict, sc: dict) -> dict:
    import k_head as KH
    R = D.paired_boot(Q["y"], Q["eid"], sc)
    arms = {k: {"auc": round(R["pt"][k], 4), "auc_unrounded": float(R["pt"][k]), "ci95": [round(x, 4) for x in KH.ci(R["bt"][k])],
                "n_windows": R["n"], "n_decel": R["n_pos"], "n_episodes": R["n_episodes"], "d": D_ARM.get(k)} for k in sc}
    deltas = {k: D.diff(R, k, "LH") for k in sc if k != "LH"}
    return {"R": R, "arms": arms, "deltas_vs_LH": deltas}


def verdicts(T: dict, have_lhg: bool) -> dict:
    R = T["R"]
    const_ok = bool(R["pt"]["constant"] == 0.5)
    unit_ok = all(R["unit_ok"].values())
    controls_ok = const_ok and unit_ok
    f1, fp = bar(R, "FUSE"), bar(R, "FUSE_perm")
    g1 = bar(R, "LH_G") if have_lhg else None
    if not controls_ok:
        wf = wg = "INVALID (a known-value control failed: constant != 0.5000 or the weighted-AUC kernel != AUC at unit weights)"
    else:
        wf = ("VOID (FUSE_perm clears F-1's bar: the gain is not per-window a0)" if fp["clears_bar"] else ("PASS" if f1["clears_bar"] else "FAIL"))
        wg = ("PASS" if g1["clears_bar"] else "FAIL") if g1 is not None else "NOT RUN"
    return {"F-1": {"word": wf, **f1, "bar": "AUC(FUSE) - AUC(LH) >= +0.020 AND the paired CI lower bound > 0"},
            "FUSE_perm_vs_F-1_bar": {**fp, "must_not_clear": True},
            "G-1": ({"word": wg, **g1, "bar": "AUC(LH_G) - AUC(LH) >= +0.020 AND the paired CI lower bound > 0"} if g1 is not None else {"word": wg}),
            "controls": {"constant_reads_0.5000": {"auc": R["pt"]["constant"], "pass": const_ok}, "WAuc_unit_weights_eq_AUC": R["unit_ok"], "controls_all_pass": controls_ok}}


def strip(T: dict) -> dict:
    return {k: v for k, v in T.items() if k != "R"}


# ---------------------------------------------------------------------------------------------------------------- self-test (first pull)
def selftest() -> int:
    import k_head as KH
    import lh_fit as LF
    t0 = time.time()
    got = pins()
    point_at(FIRST)
    P = K.load_population()                                         # first pull, all 300 clips
    zb = P["zb"]
    a0, psi, fb = K.eval_features(zb)
    U = np.load(D.UNIVERSE)                                         # LHA0's own fresh table (sha256 beadb751..., pinned by PREREG_LHA0_D2)
    chk = {"universe_sha256": sha256_file(D.UNIVERSE) == "beadb7512d2ea323544ec5853d1114580768908d4aed0a69f90d1e15b8ec4d8f", "no_fallback_rows_on_first_pull": int(fb.sum()) == 0}
    # ---- T2: population. LHA0's P = the K-HEAD rule restricted to U (the 290 lead / fit episodes with finite features) ----
    fit_eps = np.unique(U["d_eid"])
    inU = np.isin(P["eid"], fit_eps) & np.isfinite(a0[P["jz"]]) & np.isfinite(psi[P["jz"]])
    Q = restrict(P, inU)
    keyQ = set(zip(Q["eid"].tolist(), zb["ws"][Q["jz"]].tolist()))
    pm = U["Pmask"]
    keyU = set(zip(U["eid_u"][pm].tolist(), U["ws_u"][pm].tolist()))
    decU = {(int(e), int(w)) for e, w, d in zip(U["eid_u"][pm], U["ws_u"][pm], U["is_dec"][pm]) if d}
    decQ = {(int(e), int(w)) for e, w, d in zip(Q["eid"], zb["ws"][Q["jz"]], Q["y"]) if d}
    chk["T2_population_rows_equal_LHA0_P"] = keyQ == keyU
    chk["T2_decel_labels_equal_LHA0_P"] = decQ == decU
    chk["T2_counts_5038_1468_280"] = (len(Q["y"]), int(Q["y"].sum()), len(np.unique(Q["eid"]))) == (5038, 1468, 280)
    # z / v0 joins equal LHA0's banked universe rows bitwise (the LH arm's inputs)
    u_key = {(int(e), int(w)): i for i, (e, w) in enumerate(zip(U["eid_u"], U["ws_u"]))}
    ui = np.array([u_key[(int(e), int(w))] for e, w in zip(Q["eid"], zb["ws"][Q["jz"]])])
    chk["T2_z_join_bitwise_eq_universe"] = bool(np.array_equal(zb["z"][Q["jz"]], U["Zu"][ui]))
    chk["T2_v0_join_bitwise_eq_universe"] = bool(np.array_equal(np.asarray(zb["v0_in"], np.float64)[Q["jz"]], U["v0u"][ui]))
    chk["T2_a0_bitwise_eq_universe"] = bool(np.array_equal(a0[Q["jz"]], U["f_a0"][ui])) and bool(np.array_equal(psi[Q["jz"]], U["f_psi"][ui]))
    # ---- T8: the permutation wrapper over U's bank rows reproduces LHA0's C4 permutation bit for bit ----
    bk_key = {(int(e), int(w)): j for j, (e, w) in enumerate(zip(zb["e_i"], zb["ws"]))}
    u_bank = np.array([bk_key[(int(e), int(w))] for e, w in zip(U["eid_u"], U["ws_u"])])
    assert np.all(np.diff(u_bank) > 0), "U rows are not in bank order"
    a0p, psip = permuted_table(zb, a0, psi, u_bank)
    chk["T8_permutation_eq_LHA0_C4"] = bool(np.array_equal(a0p[u_bank], U["f_a0_perm"])) and bool(np.array_equal(psip[u_bank], U["f_psi_perm"]))
    # ---- T3 / T9: the arm table on P (frozen heads in-sample: no known value; neg_a0_raw and constant ARE known) ----
    H = D.load_frozen(LHA0D2 / "D2_FROZEN.json", PIN["D2_FROZEN.json"])
    LHb = LF.load_frozen(LONHEAD / "LH_FROZEN.pt", PIN["LH_FROZEN.pt"])
    sc = arm_scores(Q, zb, a0, psi, a0p, psip, H, LHb)              # LH_G slot fed the frozen LH: the LH_G code path must reproduce K-HEAD's LH score bitwise
    t7 = float(np.abs(sc["LH_G"] - sc["LH"]).max())                 # BLAS row-blocking: the same matmul over a different row count is equal to ~1 ulp, not bitwise
    chk["T7_LHG_path_eq_LH_score_maxabs_lt_1e-12"] = t7 < 1e-12
    chk["T7_LHG_path_LH_auc_identical"] = KH.auc_np(Q["y"], sc["LH_G"]) == KH.auc_np(Q["y"], sc["LH"])
    T = read_table(Q, sc)
    ph = json.load(open(LHA0 / "result" / "LHA0_POSTHOC.json", encoding="utf-8"))
    na = T["arms"]["neg_a0_raw"]
    chk["T3_neg_a0_auc_ci_eq_LHA0_POSTHOC"] = (na["auc"], na["ci95"]) == (ph["auc"]["neg_a0"]["auc"], ph["auc"]["neg_a0"]["ci95"])
    chk["T9_constant_0.5000"] = T["arms"]["constant"]["auc_unrounded"] == 0.5
    chk["T9_WAuc_unit"] = all(T["R"]["unit_ok"].values())
    V = verdicts(T, True)
    chk["T9_G1_bar_on_identical_heads_FAILS"] = V["G-1"]["word"] == "FAIL" and V["G-1"]["delta_unrounded"] == 0.0     # LH_G := LH here -> delta exactly 0
    # ---- T4: the FUSE construction + estimator on LHA0-D2's banked FRESH OOF logits (d2_posthoc fresh_OOF_fusion) ----
    z = np.load(LHA0D2 / "result" / "d2_fresh_oof.npz")
    Pf, yf, eu = z["P"], z["y"].astype(bool), z["eid_u"]
    lg_lh = np.load(LHA0 / "result" / "oof_logits" / "A_s0_l100.npz")["logits"]
    lhf, d2f = D.lh_logit(lg_lh)[Pf], z["d2_logit_oof"][Pf]
    s_lhf = KH.renorm_decel(KH.softmax(lg_lh))[0][Pf]
    Qf = {"y": yf, "eid": eu[Pf]}
    Tf = read_table(Qf, {"LH": s_lhf, "D2": d2f, "FUSE": D.zscore(d2f) + D.zscore(lhf)})
    ref = json.load(open(LHA0D2 / "result" / "LHA0_D2_POSTHOC.json", encoding="utf-8"))["fresh_OOF_fusion"]
    chk["T4_LH_OOF_auc_0.6699"] = Tf["arms"]["LH"]["auc"] == ref["auc"]["LH_OOF"]
    chk["T4_D2_OOF_auc_0.7614"] = Tf["arms"]["D2"]["auc"] == ref["auc"]["D2"]
    chk["T4_FUSE_auc_0.7569"] = Tf["arms"]["FUSE"]["auc"] == ref["auc"]["fuse"]
    chk["T4_FUSE_minus_LH_point_ci"] = (Tf["deltas_vs_LH"]["FUSE"]["point"], Tf["deltas_vs_LH"]["FUSE"]["ci95"]) == (ref["fuse_minus_LH_OOF"]["point"], ref["fuse_minus_LH_OOF"]["ci95"])
    Vf = verdicts({"R": {**Tf["R"], "pt": {**Tf["R"]["pt"], "constant": 0.5, "FUSE_perm": Tf["R"]["pt"]["LH"]},
                         "bt": {**Tf["R"]["bt"], "FUSE_perm": Tf["R"]["bt"]["LH"]}}, "arms": Tf["arms"]}, False)
    chk["T4_F1_bar_logic_on_known_+0.087"] = Vf["F-1"]["word"] == "PASS"                   # +0.087 [+0.0598, +0.1162] clears +0.020 with CI > 0 (bar logic only)
    ok = all(bool(v) for v in chk.values())
    out = {"tool": "fresh2_stage1.py --selftest", "T7_maxabs_LHG_path_minus_LH": t7, "note_T7": "first run (11:3x) asserted BITWISE equality and failed on ~ulp BLAS differences; "
           "the check was relaxed to max|diff| < 1e-12 + identical AUC (disclosed in DEVIATIONS.md)", "data": "FIRST pull fresh split (L1C), never CONFIRM, never eval139", "pins": got, "checks": chk, "ALL_PASS": ok,
           "first_pull_table_on_LHA0_P (in-sample for the frozen heads: descriptive only)": strip(T),
           "oof_fusion_reproduction": strip(Tf), "wall_s": round(time.time() - t0, 1)}
    (PKG / "result").mkdir(exist_ok=True)
    (PKG / "result" / "SELFTEST_STAGE1.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"ZZS1SELFTEST{'PASS' if ok else 'FAIL'}ZZ " + json.dumps(chk), flush=True)
    return 0 if ok else 2


# ---------------------------------------------------------------------------------------------------------------- stage 1 (CONFIRM, once)
def lon6_reading(P: dict, Q_mask_confirm: np.ndarray, zb: dict, a0, psi, a0p, psip, H, LHb, LHGb) -> dict:
    """BESIDE (READING_FRESH2.md): GT from the LON6 labels built WITH the lead pass -- decel = BRAKE_TO/CREEP/HOLD, cruise = FOLLOW/CRUISE --
    on S6-valid & compliant CONFIRM windows of clips the lead pass covers.  Same arms, same estimator, same bars (descriptive)."""
    import k_head as KH
    import lh_fit as LF
    import lh_klabel as KL
    Ee = P["Ee"]
    rel = Path(FRESH2["release"])
    kit = {"release": str(rel), "md5": KL.md5_file(rel), "sidecar": FRESH2["sidecar"], "dumps": {0: FRESH2["dump"]}, "windows": FRESH2["windows"]}
    r = KL.trainer_lon_targets(kit, 0)
    assert np.array_equal(r["clip_index"], Ee["clip"]) and np.array_equal(r["ws"], Ee["ws"]), "label rows != dump rows"
    zrel = np.load(rel, allow_pickle=False)
    nolead = {str(s) for s, h in zip(zrel["clip__sha12"], zrel["clip__has_lead"]) if not bool(h)}
    jx = json.load(open(Path(FRESH2["dump"]) / "refcv6_extras.npz.json", encoding="utf-8"))
    i2s = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
    lead = ~np.isin(np.array([i2s[int(c)] for c in Ee["clip"]]), sorted(nolead))
    lon = r["lon"]
    dec, cru = np.isin(lon, (3, 4, 5)), np.isin(lon, (0, 1))
    m = np.asarray(Ee["valid_s6"], bool) & np.asarray(Ee["compliant"], bool) & Q_mask_confirm & lead & (dec | cru)
    ui = np.flatnonzero(m)
    jz = np.array([zb["idx"].get((int(c), int(w)), -1) for c, w in zip(Ee["clip"][ui], Ee["ws"][ui])])
    assert (jz >= 0).all()
    lg = LF.predict_logits6(LHb, zb["z"][jz], np.asarray(zb["v0_in"], np.float64)[jz])
    Q = {"ui": ui, "jz": jz, "lg": lg, "s_lh": KH.renorm_decel(KH.softmax(lg))[0], "y": dec[ui], "eid": np.asarray(Ee["eid"])[ui]}
    sc = arm_scores(Q, zb, a0, psi, a0p, psip, H, LHGb)
    T = read_table(Q, sc)
    V = verdicts(T, LHGb is not None)
    return {"status": "BESIDE, non-gating (READING_FRESH2.md)", "definition": "GT = LON6 label (lead pass): decel {BRAKE_TO, CREEP, HOLD} vs cruise {FOLLOW, CRUISE}; "
            "S6-valid & compliant CONFIRM windows on lead-pass clips", "release_md5": kit["md5"], "label_checks": r["checks"],
            "n_confirm_clips_without_lead": int(len({i2s[int(c)] for c in Ee["clip"][Q_mask_confirm]} & nolead)), **strip(T), "bars_descriptive": V}


def run() -> int:
    import lh_fit as LF
    t0 = time.time()
    out_json = PKG / "result" / "FRESH2_STAGE1.json"
    if out_json.exists():
        raise SystemExit("[fresh2] result/FRESH2_STAGE1.json exists -- stage 1 is run ONCE")
    got = pins()
    rd = PKG / "READING_FRESH2.md"
    rds = (PKG / "READING_FRESH2.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(rd) != rds:
        raise SystemExit("[fresh2] READING_FRESH2.md does not match its recorded sha256 -- STOP")
    lhg_reg = (LHG.parent / "LH_G_FROZEN.sha256").read_text(encoding="utf-8").split()[0]
    LHGb = LF.load_frozen(LHG, lhg_reg)
    split = json.load(open(FRESH2["split"], encoding="utf-8"))
    conf12, grow12 = {sha12(c) for c in split["confirm"]}, {sha12(c) for c in split["grow"]}
    point_at(FRESH2)
    P = K.load_population()                                         # all usable FRESH2 clips (the K-HEAD rule, d2_khead unedited)
    zb, Ee = P["zb"], P["Ee"]
    jx = json.load(open(Path(FRESH2["dump"]) / "refcv6_extras.npz.json", encoding="utf-8"))
    i2s = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
    assert set(i2s.values()) == conf12 | grow12, "dump clips != CONFIRM + GROW"
    conf_idx = sorted(i for i, s in i2s.items() if s in conf12)
    a0, psi, fb = K.eval_features(zb)
    assert int(fb.sum()) == 0, "FRESH2 has no clip without a clock row (refused clips were excluded)"
    in_c = np.isin(P["eid"], conf_idx)
    Q = restrict(P, in_c)
    m_dump_conf = np.isin(Ee["clip"], conf_idx)
    valid, comp, ge = np.asarray(Ee["valid_s6"], bool), np.asarray(Ee["compliant"], bool), np.asarray(Ee["gt"])
    pop = {"confirm_clips": len(conf12), "confirm_dump_windows": int(m_dump_conf.sum()), "s6_valid": int((valid & m_dump_conf).sum()),
           "compliant": int((valid & comp & m_dump_conf).sum()), "gt_decel": int((valid & comp & m_dump_conf & (ge == "decel")).sum()),
           "gt_cruise": int((valid & comp & m_dump_conf & (ge == "cruise")).sum()), "union_n": int(len(Q["y"])), "n_episodes": int(len(np.unique(Q["eid"])))}
    conf_bank = np.flatnonzero(np.isin(zb["e_i"], conf_idx))
    a0p, psip = permuted_table(zb, a0, psi, conf_bank)
    H = D.load_frozen(LHA0D2 / "D2_FROZEN.json", PIN["D2_FROZEN.json"])
    LHb = LF.load_frozen(LONHEAD / "LH_FROZEN.pt", PIN["LH_FROZEN.pt"])
    sc = arm_scores(Q, zb, a0, psi, a0p, psip, H, LHGb)
    T = read_table(Q, sc)
    V = verdicts(T, True)
    side = lon6_reading(P, m_dump_conf, zb, a0, psi, a0p, psip, H, LHb, LHGb)
    f1, g1 = V["F-1"]["word"], V["G-1"]["word"]
    if f1.startswith("INVALID"):
        outcome = "INVALID: reported; no claim"
    elif f1 == "PASS" or g1 == "PASS":
        outcome = ("F-1 and/or G-1 PASS: the passing decision (" + ("FUSE" if f1 == "PASS" else "LH_G") + ") is the decel decision for refcv9's progress profiles; "
                   "stage 2 = a T1 roll pre-registration drafted and sealed BEFORE any roll")
    elif f1.startswith("VOID"):
        outcome = "F-1 VOID (" + ("G-1 FAIL" if g1 == "FAIL" else g1) + "): reported; no claim on fusion"
    else:
        outcome = ("Both FAIL: the post-hoc decision line is closed on unseen data; the decision moves into refcv9 training (an in-model decision head, "
                   "REFCV9_INMODEL_DESIGN.md clean-split calibration) -- named as the next lever, not run here")
    out = {"tool": "fresh2_stage1.py --run", "prereg": "PREREG_FRESH2.md s3 (sha256 956f87cd...)", "pins": got, "reading": {"file": "READING_FRESH2.md", "sha256": rds},
           "LH_G": {"file": str(LHG), "sha256": LHGb["_sha256"], "registered": lhg_reg, "variant": LHGb["variant"], "hp": LHGb["hp"], "n_rows": LHGb["n_rows"]},
           "tier": "T0 label-level diagnostic (heads read the frozen refcv8 EMA-48,852 trunk's t0 z_tac / measured t0 ego state; no planner, no roll) -- never driving performance",
           "estimator": "Mann-Whitney AUC (average-rank ties); paired episode-cluster bootstrap over CONFIRM episodes (B 2000, seed 0, every arm on the same draws); CI = 2.5/97.5 "
                        "percentiles of the paired difference. Answers 'another draw of EPISODES with the scores held fixed' ONLY -- not another training run (one refcv8 trunk, "
                        "one fit per head) and not another inference run (heads deterministic).",
           "population": pop, "arms": T["arms"], "deltas_vs_LH": T["deltas_vs_LH"], "bars": V, "outcome_committed_s4": outcome,
           "four_families": {"TACTICAL": "this read (decel-vs-cruise decision quality of the lon decision head, AUC)",
                             "LONGITUDINAL": "n/a at T0 label level: no plan / speed profile is produced here (stage 2's T1 roll would carry it)",
                             "LATERAL": "n/a: the decision read is longitudinal only; no path is produced", "STRATEGIC": "n/a: no route / goal decision is read here"},
           "beside_LON6_label_reading": side, "wall_s": round(time.time() - t0, 1)}
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, out_json)
    write_md(out)
    print(f"ZZS1DONEZZ F-1 {f1} | G-1 {g1} | FUSE {T['arms']['FUSE']['auc']} LH {T['arms']['LH']['auc']} LH_G {T['arms']['LH_G']['auc']} D2 {T['arms']['D2']['auc']} | "
          f"n {pop['union_n']} ep {pop['n_episodes']}", flush=True)
    return 0


def write_md(o: dict) -> None:
    A, Dl, V, p = o["arms"], o["deltas_vs_LH"], o["bars"], o["population"]
    L = ["# FRESH2 stage 1 -- T0 label-level decel-decision read on CONFIRM (PREREG_FRESH2 s3, run once)", "",
         f"**F-1 (fusion confirmed): {V['F-1']['word']}** -- AUC(FUSE) - AUC(LH) = {V['F-1']['delta_vs_LH']:+.4f} [{V['F-1']['ci95'][0]:+.4f}, {V['F-1']['ci95'][1]:+.4f}] (bar +0.020 and CI lower bound > 0)  ",
         f"**G-1 (grown head): {V['G-1']['word']}** -- AUC(LH_G) - AUC(LH) = {V['G-1']['delta_vs_LH']:+.4f} [{V['G-1']['ci95'][0]:+.4f}, {V['G-1']['ci95'][1]:+.4f}]  ",
         f"Controls: constant = {V['controls']['constant_reads_0.5000']['auc']:.4f} (must be 0.5000); FUSE_perm - LH = {V['FUSE_perm_vs_F-1_bar']['delta_vs_LH']:+.4f} "
         f"[{V['FUSE_perm_vs_F-1_bar']['ci95'][0]:+.4f}, {V['FUSE_perm_vs_F-1_bar']['ci95'][1]:+.4f}] (must NOT clear F-1's bar: clears = {V['FUSE_perm_vs_F-1_bar']['clears_bar']}).  ",
         f"Committed outcome (s4): {o['outcome_committed_s4']}", "",
         "MEASURED; tier T0 (label level, never driving performance); estimator = Mann-Whitney AUC + paired episode-cluster bootstrap over CONFIRM episodes, B 2000, seed 0. "
         "The interval answers another draw of EPISODES only (one refcv8 trunk, one fit per head; heads deterministic).", "",
         f"Population (K-HEAD rule, PREREG_LHA0_D2 s3 code-exact): {p['confirm_clips']} CONFIRM clips, {p['confirm_dump_windows']} windows; S6-valid {p['s6_valid']}, compliant {p['compliant']}, "
         f"GT-decel {p['gt_decel']}, GT-cruise {p['gt_cruise']}; scored union {p['union_n']} windows / {p['n_episodes']} episodes.", "",
         "| arm | AUC | 95 % CI | AUC - LH | paired 95 % CI | n windows | n decel | n episodes | d |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ("LH", "D2", "FUSE", "LH_G", "constant", "FUSE_perm", "neg_a0_raw", "D2_perm"):
        a = A[k]
        dl = Dl.get(k)
        ds = "--" if dl is None else f"{dl['point']:+.4f}"
        dc = "--" if dl is None else f"[{dl['ci95'][0]:+.4f}, {dl['ci95'][1]:+.4f}]"
        tag = "" if k in GATING else " (beside)"
        L.append(f"| {k}{tag} | {a['auc']:.4f} | [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}] | {ds} | {dc} | {a['n_windows']} | {a['n_decel']} | {a['n_episodes']} | {a['d']} |")
    s = o["beside_LON6_label_reading"]
    L += ["", "## Beside (non-gating): the LON6-label GT reading (READING_FRESH2.md)", "",
          f"{s['definition']}; F-1 would read {s['bars_descriptive']['F-1']['word']} ({s['bars_descriptive']['F-1']['delta_vs_LH']:+.4f} "
          f"[{s['bars_descriptive']['F-1']['ci95'][0]:+.4f}, {s['bars_descriptive']['F-1']['ci95'][1]:+.4f}]), G-1 {s['bars_descriptive']['G-1']['word']} "
          f"({s['bars_descriptive']['G-1']['delta_vs_LH']:+.4f} [{s['bars_descriptive']['G-1']['ci95'][0]:+.4f}, {s['bars_descriptive']['G-1']['ci95'][1]:+.4f}]); "
          f"n {s['arms']['LH']['n_windows']} windows / {s['arms']['LH']['n_decel']} decel / {s['arms']['LH']['n_episodes']} episodes.", "",
          "| arm | AUC | AUC - LH [paired 95 % CI] |", "|---|---|---|"]
    for k in ("LH", "D2", "FUSE", "LH_G", "constant", "FUSE_perm"):
        dl = s["deltas_vs_LH"].get(k)
        L.append(f"| {k} | {s['arms'][k]['auc']:.4f} | " + ("--" if dl is None else f"{dl['point']:+.4f} [{dl['ci95'][0]:+.4f}, {dl['ci95'][1]:+.4f}]") + " |")
    L += ["", "Four families: TACTICAL = this read; LONGITUDINAL / LATERAL / STRATEGIC = n/a at T0 label level (no plan is produced; stage 2's T1 roll would carry them).",
          f"LH_G: {o['LH_G']['variant']} hp {o['LH_G']['hp']}, {o['LH_G']['n_rows']} fit rows, sha256 {o['LH_G']['sha256'][:16]}... (registered before this run).", ""]
    (PKG / "result" / "FRESH2_STAGE1.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else run())
