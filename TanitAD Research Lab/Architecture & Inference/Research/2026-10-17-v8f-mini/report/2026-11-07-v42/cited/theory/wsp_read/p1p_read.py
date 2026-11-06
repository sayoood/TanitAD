#!/usr/bin/env python3
"""p1p_read.py -- SPEC_P1P.md (sha256 73640c67..., SPEC_P1P.sha256): P1' read on the T1-kin eval-141 tick-0 windows.  CPU only, zero training, no weight edit.
Needs P1P_PROBE_MODELS.npz (p1p_probe_refit.py).  Stages (one process):
  1. C4 unit tests (before any eval window is read)
  2. eval features (V from the eval cache, E from the native tracks) for every tick-0 window with v0 < 8, posteriors of the five fold models, the all-train model, the GO24 models
  3. controls R2, R3, K1-K3
  4. the P1' read (primary + S1..S4), controls N1-N3
Writes P1P_PERWINDOW.npz and P1P_RESULT.json.   CUDA_VISIBLE_DEVICES=-1  python -I p1p_read.py"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TP = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe"
ART = "E:/Projects/TanitAD-artifacts"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
SCR = "F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch"
EVAL_CLIPS = f"{ART}/sfresh_inputs/cache/eval/clips"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
DUMPS = {"s0": "C:/Users/Admin/rr_scratch/out/C0C_s0/ticks/ticks_C0C_s0.npz", "s1": "C:/Users/Admin/rr_scratch/out/C0C_s1/ticks/ticks_C0C_s1.npz"}
sys.path.insert(0, HERE)
import p1p_tp_lib as L  # noqa: E402

V_LO, V_HI = 3.0, 8.0
A_MIN = 0.5
CONF_MIN, UNS_LO, UNS_HI = 0.8, 0.35, 0.65
N_MIN = 30
B = 2000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ statistics
def wmedian_rows(vals, W):
    """Median of the multiset in which vals[i] appears W[b, i] times, one row b per resample (np.median convention).  vals [n]; W [B, n] integer multiplicities.  NaN if a row is empty."""
    o = np.argsort(vals, kind="stable")
    v = vals[o]
    w = W[:, o]
    cw = np.cumsum(w, axis=1)
    tot = cw[:, -1]
    k1 = (tot + 1) // 2
    k2 = tot // 2 + 1
    i1 = (cw >= k1[:, None]).argmax(axis=1)
    i2 = (cw >= k2[:, None]).argmax(axis=1)
    m = 0.5 * (v[i1] + v[i2])
    return np.where(tot > 0, m, np.nan)


def groups(post):
    return post >= CONF_MIN, (post >= UNS_LO) & (post <= UNS_HI)


def read_one(rho, post, clip_id, cnt=None, ci_lo_q=2.5, ci_hi_q=97.5):
    """P1' statistic on one window set.  rho, post, clip_id all [n].  Returns the dict of the read (counts, medians, delta, interval, separated, dropped replicates)."""
    cm, um = groups(post)
    n_c, n_u = int(cm.sum()), int(um.sum())
    out = {"n_windows": int(len(rho)), "n_conf": n_c, "n_unsure": n_u, "n_neither": int(len(rho) - n_c - n_u), "n_episodes": int(len(np.unique(clip_id))),
           "n_episodes_conf": int(len(np.unique(clip_id[cm]))), "n_episodes_unsure": int(len(np.unique(clip_id[um])))}
    if n_c == 0 or n_u == 0:
        return {**out, "median_conf": None, "median_unsure": None, "delta": None, "lo": None, "hi": None, "separated": False, "positive": False, "dropped": None}
    cnt_, inv = cnt if cnt is not None else L.boot_counts(clip_id)
    Wm = cnt_[:, inv]                                       # [B, n] multiplicities per window
    mc = wmedian_rows(rho[cm], Wm[:, cm])
    mu_ = wmedian_rows(rho[um], Wm[:, um])
    d = mc - mu_
    ok = np.isfinite(d)
    dd = d[ok]
    med_c, med_u = float(np.median(rho[cm])), float(np.median(rho[um]))
    lo, hi = (float(np.percentile(dd, ci_lo_q)), float(np.percentile(dd, ci_hi_q))) if len(dd) else (float("nan"), float("nan"))
    delta = med_c - med_u
    out.update({"median_conf": med_c, "median_unsure": med_u, "delta": delta, "lo": lo, "hi": hi, "separated": bool(np.isfinite(lo) and lo > 0), "positive": bool(delta > 0),
                "dropped": int((~ok).sum()), "median_conf_ci": [float(np.nanpercentile(mc, 2.5)), float(np.nanpercentile(mc, 97.5))],
                "median_unsure_ci": [float(np.nanpercentile(mu_, 2.5)), float(np.nanpercentile(mu_, 97.5))],
                "mean_conf": float(rho[cm].mean()), "mean_unsure": float(rho[um].mean()),
                "q25_75_conf": [float(np.percentile(rho[cm], 25)), float(np.percentile(rho[cm], 75))], "q25_75_unsure": [float(np.percentile(rho[um], 25)), float(np.percentile(rho[um], 75))]})
    return out


def verdict(per_seed):
    """per_seed: {seed: read_one dict}.  The rule of AMEND_1 section 4, as written."""
    for s, r in per_seed.items():
        if r["n_conf"] < N_MIN or r["n_unsure"] < N_MIN:
            return "NOT EVALUABLE"
    if all(r["positive"] and r["separated"] for r in per_seed.values()):
        return "HOLDS"
    return "FAILS"


# ------------------------------------------------------------------------------------------------ C4 unit tests
def unit_tests():
    rng = np.random.default_rng(123)
    res = {}
    n = 57
    vals = rng.normal(size=n)
    Wt = rng.integers(0, 4, size=(40, n))
    ref = np.array([np.median(np.repeat(vals, w)) if w.sum() > 0 else np.nan for w in Wt])
    got = wmedian_rows(vals, Wt)
    res["weighted_median_equals_np_median_of_replicated"] = bool(np.allclose(ref, got, equal_nan=True))
    clip = rng.integers(0, 9, size=80)
    c1, inv1 = L.boot_counts(clip)
    u, inv_ref = np.unique(clip, return_inverse=True)
    rr = np.random.default_rng(0)
    c_ref = np.stack([np.bincount(rr.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(L.N_BOOT)])
    res["bootstrap_counts_equal_tp_lib_construction"] = bool(np.array_equal(c1, c_ref) and np.array_equal(inv1, inv_ref))
    cm, um = groups(np.array([0.8, 0.35, 0.65, 0.7999999, 0.34999, 0.65001, 0.5, 0.95]))
    res["group_edges"] = bool(cm.tolist() == [True, False, False, False, False, False, False, True] and um.tolist() == [False, True, True, False, False, False, True, False])
    res["all_pass"] = bool(all(v for v in res.values()))
    return res


# ------------------------------------------------------------------------------------------------ eval features
def eval_features(rows):
    """rows: list of (clip name, t0).  Returns V [n, 40960] fp16 (the cache row, fp8 -> fp16), E [n, 8] float64, kin_v0 [n]."""
    import torch
    sys.path[:0] = [f"{TREE}/stack", f"{TREE}/stack/scripts", f"{TREE}/taniteval"]
    from tanitad.data import v8f_mini_data as D
    torch.set_num_threads(4)
    tracks = D.load_native_tracks(TRACKS)
    fk = D._fk()
    by = {}
    for i, (c, t) in enumerate(rows):
        by.setdefault(c, []).append((i, t))
    V = np.zeros((len(rows), 40 * 1024), np.float16)
    E = np.zeros((len(rows), 8), np.float64)
    kv = np.zeros(len(rows))
    for c, items in by.items():
        o = torch.load(f"{EVAL_CLIPS}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
        f4 = o["pool4x10"]
        kin3, _ = D.yaw_kin_from_track(tracks[c])
        for i, t in items:
            V[i] = f4[t].to(torch.float16).reshape(-1).numpy()
            E[i] = fk.kin_features(kin3[t][None].astype(np.float64))[0]
            kv[i] = kin3[t, 0]
    return V, E, kv


def sigmoid(d):
    return 1.0 / (1.0 + np.exp(-np.clip(d, -500, 500)))


def score(models, tag, m, V, E):
    mu, comp = models["pca_mu"][m], models["pca_comp"][m]
    S = ((V.astype(np.float32) - mu) @ comp.T).astype(np.float32)
    X = np.concatenate([S.astype(np.float64), E], 1)
    d = ((X - models[f"{tag}_mean"][m]) / models[f"{tag}_scale"][m]) @ models[f"{tag}_coef"][m] + models[f"{tag}_icpt"][m]
    return d


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sec(x):
    return np.hypot(x[:, 11, 0] - x[:, 7, 0], x[:, 11, 1] - x[:, 7, 1]) / 0.8


def main():
    out = {"spec_sha256": sha256_file(f"{HERE}/SPEC_P1P.md"), "spec_sha_file": open(f"{HERE}/SPEC_P1P.sha256").read().strip().splitlines()[0],
           "code_sha256": {f: sha256_file(f"{HERE}/{f}") for f in ("p1p_tp_lib.py", "p1p_probe_refit.py", "p1p_read.py", "P1P_PROBE_MODELS.npz")},
           "amend_sha256": sha256_file(f"{ART}/dagland/motion_c/theory/THEORY_CRAWL_AMEND_1.md")}
    assert out["spec_sha256"] == out["spec_sha_file"].split()[0], "SPEC_P1P.md changed after it was hashed"
    assert out["amend_sha256"].startswith("9d39ac53"), out["amend_sha256"]
    assert out["code_sha256"]["p1p_tp_lib.py"] == sha256_file(f"{TP}/tp_lib.py"), "tp_lib copy is not byte-identical"
    out["C4_unit_tests"] = unit_tests()
    log("C4", out["C4_unit_tests"])
    assert out["C4_unit_tests"]["all_pass"], "C4 failed: not reading eval windows"

    r1 = json.load(open(f"{HERE}/P1P_REFIT_R1.json"))
    r1s = {"pca_max_abs_diff_rel": {}, "heldout_vs_banked_oof_max_abs_go": {}, "heldout_vs_banked_oof_max_abs_go24": {}, "C_go": {}, "C_go24": {}}
    for f, rec in r1["folds"].items():
        r1s["pca_max_abs_diff_rel"][f] = rec["pca_max_abs_diff_vs_banked_S"] / rec["banked_S_max_abs"]
        r1s["C_go"][f], r1s["C_go24"][f] = rec.get("go_C"), rec.get("go24_C")
        if f != "5":
            r1s["heldout_vs_banked_oof_max_abs_go"][f] = rec["go_heldout_vs_banked_oof_max_abs"]
            r1s["heldout_vs_banked_oof_max_abs_go24"][f] = rec["go24_heldout_vs_banked_oof_max_abs"]
    r1s["pass"] = bool(max(v for k, v in r1s["pca_max_abs_diff_rel"].items() if k != "5") <= 1e-3 and max(r1s["heldout_vs_banked_oof_max_abs_go"].values()) <= 1e-6
                       and max(r1s["heldout_vs_banked_oof_max_abs_go24"].values()) <= 1e-6)
    out["R1_refit_reproduces_banked"] = r1s
    log("R1", r1s)
    models = dict(np.load(f"{HERE}/P1P_PROBE_MODELS.npz"))
    # ---------------------------------------------------------------- dump tick-0 rows (identical windows at both seeds, checked)
    Z = {s: np.load(p, allow_pickle=True) for s, p in DUMPS.items()}
    m0 = {s: Z[s]["tick"] == 0 for s in Z}
    base = {k: Z["s0"][k][m0["s0"]] for k in ("ci", "t0", "win", "v0", "clip", "gt", "gt_ok")}
    for s in ("s1",):
        for k in ("ci", "t0", "win", "v0", "clip", "gt", "gt_ok"):
            a_, b_ = Z[s][k][m0[s]], base[k]
            assert np.array_equal(a_, b_, **({"equal_nan": True} if a_.dtype.kind in "fc" else {})), ("tick-0 rows differ between seeds", k)
    N = len(base["v0"])
    v0, gt, ok_gt, ci, t0s, names = base["v0"], base["gt"], base["gt_ok"].astype(bool), base["ci"], base["t0"], base["clip"]
    plan = {s: Z[s]["plan"][m0[s]] for s in Z}
    plan_raw = {s: Z[s]["plan_raw"][m0[s]] for s in Z}
    out["tick0_windows"] = int(N)
    # ---------------------------------------------------------------- features for every window with v0 < 8
    need = np.flatnonzero(v0 < V_HI)
    log("feature windows", len(need))
    V, E, kv = eval_features([(str(names[i]), int(t0s[i])) for i in need])
    log("features built")
    d_all = {m: score(models, "go", m, V, E) for m in range(6)}
    d24 = {m: score(models, "go24", m, V, E) for m in range(5)}
    p_fold = np.stack([sigmoid(d_all[m]) for m in range(5)], 1)
    post = np.zeros(N)
    post[:] = np.nan
    post_all, post24 = post.copy(), post.copy()
    post[need] = p_fold.mean(1)
    post_all[need] = sigmoid(d_all[5])
    post24[need] = np.stack([sigmoid(d24[m]) for m in range(5)], 1).mean(1)
    # ---------------------------------------------------------------- R3: eval feature construction vs banked
    ev = np.load(f"{TP}/raw/eval141_features.npz", allow_pickle=False)
    Vb = np.load(f"{SCR}/V_eval141.npy", mmap_mode="r")
    key = {(str(c), int(t)): i for i, (c, t) in enumerate(zip(ev["clip"], ev["t0"]))}
    pos_need = {int(w): j for j, w in enumerate(need)}
    ov = [(pos_need[i], key[(str(names[i]), int(t0s[i]))]) for i in need if (str(names[i]), int(t0s[i])) in key]
    a_idx = np.array([a for a, _ in ov])
    b_idx = np.array([b for _, b in ov])
    go_dump = L.go_label(v0, gt, 29)
    go24_dump = L.go_label(v0, gt, 11)
    r3 = {"banked_windows": int(len(ev["v0"])), "overlap": int(len(ov)), "windows_v0_lt_6_in_dump": int((v0 < 6).sum()),
          "E_max_abs": float(np.abs(E[a_idx] - ev["E"][b_idx]).max()), "V_bit_equal": bool(np.array_equal(V[a_idx], np.asarray(Vb[b_idx]))),
          "V_max_abs": float(np.abs(V[a_idx].astype(np.float32) - np.asarray(Vb[b_idx]).astype(np.float32)).max()),
          "go_equal": bool(np.array_equal(go_dump[need][a_idx], ev["go"][b_idx])), "v0_dump_vs_kin3_max_abs": float(np.abs(v0[need] - kv).max())}
    r3["pass"] = bool(r3["overlap"] == r3["banked_windows"] == r3["windows_v0_lt_6_in_dump"] and r3["E_max_abs"] <= 1e-9 and r3["V_bit_equal"] and r3["go_equal"] and r3["v0_dump_vs_kin3_max_abs"] <= 1e-4)
    out["R3_eval_feature_construction"] = r3
    log("R3", r3)
    # ---------------------------------------------------------------- R2: all-train model reproduces the banked eval-141 confirmation; K1, K2, K3
    pm = (v0 >= 0.5) & (v0 < 6.0)
    bank = json.load(open(f"{TP}/raw/results_eval141.json"))["pops"]["pmove"]["arms"]["linear:V+E"]
    cnt_pm = L.boot_counts(ci[pm])
    r_all = L.auc_ci(post_all[pm], go_dump[pm].astype(int), ci[pm], counts=cnt_pm)
    out["R2_all_train_vs_banked_eval141"] = {"auc": r_all["auc"], "lo": r_all["lo"], "hi": r_all["hi"], "n": r_all["n"], "banked_auc": bank["auc"], "banked_hp_C": bank["hp"],
                                             "refit_C": float(models["go_C"][5]), "pass": bool(abs(r_all["auc"] - bank["auc"]) <= 0.005 and abs(float(models["go_C"][5]) - float(bank["hp"])) < 1e-12)}
    k1 = L.auc_ci(post[pm], go_dump[pm].astype(int), ci[pm], counts=cnt_pm)
    b2 = (v0 >= 3.0) & (v0 < 6.0)
    k2 = L.auc_ci(post[b2], go_dump[b2].astype(int), ci[b2])
    k1_24 = L.auc_ci(post24[pm], go24_dump[pm].astype(int), ci[pm], counts=cnt_pm)
    b8 = (v0 >= V_LO) & (v0 < V_HI)
    ah_all = (np.hypot(gt[:, 11, 0] - gt[:, 7, 0], gt[:, 11, 1] - gt[:, 7, 1]) / 0.8 - v0) / 2
    k3_mask = b8 & ok_gt
    k3 = L.auc_ci(post[k3_mask], (ah_all[k3_mask] > A_MIN).astype(int), ci[k3_mask])
    out["K1_probe_auc_eval_pmove"] = {**k1, "covers_0p5": bool(k1["lo"] <= 0.5 <= k1["hi"]), "n_go": int(go_dump[pm].sum())}
    out["K1_go24_auc_eval_pmove"] = {**k1_24, "covers_0p5": bool(k1_24["lo"] <= 0.5 <= k1_24["hi"])}
    out["K2_probe_auc_eval_band_3_6"] = {**k2, "covers_0p5": bool(k2["lo"] <= 0.5 <= k2["hi"]), "n_go": int(go_dump[b2].sum())}
    out["K3_probe_auc_for_a_human_gt_0p5_band_3_8"] = {**k3, "covers_0p5": bool(k3["lo"] <= 0.5 <= k3["hi"])}
    log("R2", out["R2_all_train_vs_banked_eval141"], "K1", k1, "K2", k2, "K3", k3)
    out["probe_at_chance_in_band_K2"] = out["K2_probe_auc_eval_band_3_6"]["covers_0p5"]

    # ---------------------------------------------------------------- the read
    def window_set(vlo, vhi, key_plan, post_vec):
        sets = {}
        base_band = (v0 >= vlo) & (v0 < vhi)
        a_h = ah_all
        steps = {"tick0": int(N), "v0_band": int(base_band.sum()), "gt_ok": int((base_band & ok_gt).sum())}
        m = base_band & ok_gt
        steps["a_human_gt_0p5"] = int((m & (a_h > A_MIN)).sum())
        m = m & (a_h > A_MIN)
        a_pl = {}
        fin = m.copy()
        for s in ("s0", "s1"):
            a_pl[s] = (sec(key_plan[s]) - v0) / 2
            fin &= np.isfinite(a_pl[s]) & np.isfinite(a_h) & np.isfinite(post_vec)
        steps["finite"] = int(fin.sum())
        rho = {s: a_pl[s] / np.where(a_h != 0, a_h, np.nan) for s in ("s0", "s1")}
        return fin, steps, rho, a_pl

    def run_cfg(name, vlo, vhi, key_plan, post_vec, rho_override=None, post_override=None):
        fin, steps, rho, a_pl = window_set(vlo, vhi, key_plan, post_vec)
        pv = post_vec if post_override is None else post_override
        cnt = L.boot_counts(ci[fin])
        per = {}
        for s in ("s0", "s1"):
            r = rho[s] if rho_override is None else rho_override
            per[s] = read_one(r[fin], pv[fin], ci[fin], cnt=cnt)
            if rho_override is None:
                cm, um = groups(pv[fin])
                per[s]["median_a_human_conf"] = float(np.median(ah_all[fin][cm])) if cm.any() else None
                per[s]["median_a_human_unsure"] = float(np.median(ah_all[fin][um])) if um.any() else None
                per[s]["median_a_plan_conf"] = float(np.median(a_pl[s][fin][cm])) if cm.any() else None
                per[s]["median_a_plan_unsure"] = float(np.median(a_pl[s][fin][um])) if um.any() else None
        pf = pv[fin]
        res = {"name": name, "window_filter_steps": steps, "posterior_bins": {"lt_0.35": int((pf < 0.35).sum()), "0.35_0.65_UNSURE": int(((pf >= 0.35) & (pf <= 0.65)).sum()),
                                                                          "0.65_0.8": int(((pf > 0.65) & (pf < 0.8)).sum()), "ge_0.8_CONFIDENT": int((pf >= 0.8).sum())},
               "per_seed": per, "verdict": verdict(per)}
        return res, fin, rho, cnt

    primary, fin_p, rho_p, cnt_p = run_cfg("PRIMARY: v0 [3,8), plan, 5-fold GO posterior", V_LO, V_HI, plan, post)
    out["PRIMARY"] = primary
    log("PRIMARY", primary["verdict"], json.dumps({s: {k: r[k] for k in ("n_conf", "n_unsure", "median_conf", "median_unsure", "delta", "lo", "hi")} for s, r in primary["per_seed"].items()}))
    sens = {}
    sens["S1_in_domain_v0_3_6"] = run_cfg("S1: v0 [3,6)", 3.0, 6.0, plan, post)[0]
    sens["S2_raw_plan"] = run_cfg("S2: plan_raw", V_LO, V_HI, plan_raw, post)[0]
    sens["S3_all_train_probe"] = run_cfg("S3: all-train probe posterior", V_LO, V_HI, plan, post_all)[0]
    sens["S4_go24_label"] = run_cfg("S4: GO24 posterior", V_LO, V_HI, plan, post24)[0]
    out["SENSITIVITIES_not_the_verdict"] = sens
    for k, r in sens.items():
        log(k, r["verdict"], json.dumps({s: {kk: x[kk] for kk in ("n_conf", "n_unsure", "delta", "lo", "hi")} for s, x in r["per_seed"].items()}))

    # ---------------------------------------------------------------- N1 / N2 / N3 on the primary window set
    post_p = post[fin_p]
    ci_p = ci[fin_p]
    n1 = {"seeds_1_to_20": {}}
    sep_count = {"s0": 0, "s1": 0}
    for ps in range(1, 21):
        perm = np.random.default_rng(ps).permutation(len(post_p))
        for s in ("s0", "s1"):
            r = read_one(rho_p[s][fin_p], post_p[perm], ci_p, cnt=cnt_p)
            sep_count[s] += int(r["separated"])
            n1["seeds_1_to_20"].setdefault(s, []).append({"perm_seed": ps, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"]})
    n1["separated_of_20"] = sep_count
    n1["pass"] = bool(all(v <= 3 for v in sep_count.values()))
    out["N1_posterior_shuffle"] = n1
    one = np.ones(int(fin_p.sum()))
    n2 = {s: read_one(one, post_p, ci_p, cnt=cnt_p) for s in ("s0", "s1")}
    out["N2_identity_rho_equals_1"] = {"per_seed": n2, "verdict": verdict(n2),
                                      "pass": bool(all(r["median_conf"] == 1.0 and r["median_unsure"] == 1.0 and r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 for r in n2.values()) and verdict(n2) == "FAILS")}
    n3 = {s: read_one(post_p.copy(), post_p, ci_p, cnt=cnt_p) for s in ("s0", "s1")}
    out["N3_planted_rho_equals_posterior"] = {"per_seed": n3, "verdict": verdict(n3), "pass": bool(verdict(n3) == "HOLDS")}
    log("N1", n1["separated_of_20"], n1["pass"], "N2", out["N2_identity_rho_equals_1"]["pass"], "N3", out["N3_planted_rho_equals_posterior"]["pass"])

    # ---------------------------------------------------------------- bank per-window arrays
    sel = np.flatnonzero(v0 < V_HI)
    np.savez(f"{HERE}/P1P_PERWINDOW.npz", win=sel, clip=names[sel], ci=ci[sel], t0=t0s[sel], v0=v0[sel], a_human=ah_all[sel], post_primary=post[sel], post_all_train=post_all[sel], post_go24=post24[sel],
             post_fold=p_fold, go=go_dump[sel], primary_mask=fin_p[sel], a_plan_s0=((sec(plan["s0"]) - v0) / 2)[sel], a_plan_s1=((sec(plan["s1"]) - v0) / 2)[sel],
             rho_s0=rho_p["s0"][sel], rho_s1=rho_p["s1"][sel])
    out["controls_summary"] = {"C4": out["C4_unit_tests"]["all_pass"], "R1": out["R1_refit_reproduces_banked"]["pass"], "R2": out["R2_all_train_vs_banked_eval141"]["pass"], "R3": r3["pass"],
                               "N1": n1["pass"], "N2": out["N2_identity_rho_equals_1"]["pass"], "N3": out["N3_planted_rho_equals_posterior"]["pass"]}
    out["seconds"] = time.time() - T0
    json.dump(out, open(f"{HERE}/P1P_RESULT.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("done; verdict", primary["verdict"])


if __name__ == "__main__":
    main()
