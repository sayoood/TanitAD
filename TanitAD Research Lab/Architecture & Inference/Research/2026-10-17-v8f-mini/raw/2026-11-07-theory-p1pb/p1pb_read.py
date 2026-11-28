#!/usr/bin/env python3
"""p1pb_read.py -- SPEC_P1PB.md (sha256 in SPEC_P1PB.sha256): the P1' rule (THEORY_CRAWL_AMEND_1 section 4) read on the RA-1 holdout windows (THEORY_CRAWL_AMEND_2).  CPU only, zero training, no weight edit.
The P1' code is loaded unchanged (p1p_read.py: read_one, verdict, groups, eval_features, score, sigmoid, sec, unit_tests; p1p_tp_lib.py; P1P_PROBE_MODELS.npz).  Only the window source differs:
the tick-0 plans of p1pb_forward.py (P1PB_TICK0_s{0,1}.npz) on the 60 holdout clips, and the cache directory the V features are read from.
Needs: P1PB_FW_CONTROL.json (pass), P1PB_F1_CONTROL.json (pass), P1PB_TICK0_s{0,1}.npz.      CUDA_VISIBLE_DEVICES=-1  python -I p1pb_read.py"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
TRAIN_CLIPS = "F:/TanitAD-scratch/speedread/cache/train/clips"
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("p1p_read", f"{HERE}/p1p_read.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
L = P.L
P.EVAL_CLIPS = TRAIN_CLIPS                        # the one change of source: the holdout clips live in the train cache (same files the probe's V_pmove was extracted from)
T0 = time.time()
A_MIN, V_LO, V_HI = P.A_MIN, P.V_LO, P.V_HI


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def clip_counts(d, nm):
    """largest single clip's share of the windows of a group (stride-1 windows overlap within a clip: this is reported, not gated)"""
    if len(d) == 0:
        return None
    u, c = np.unique(d, return_counts=True)
    return {"n_clips": int(len(u)), "max_clip_share": float(c.max() / c.sum()), "median_windows_per_clip": float(np.median(c))}


def main():
    out = {}
    # ---------------------------------------------------------------- frozen inputs
    spec_sha = sha256_file(f"{HERE}/SPEC_P1PB.md")
    spec_file = open(f"{HERE}/SPEC_P1PB.sha256").read().splitlines()[0].split()[0]
    assert spec_sha == spec_file, "SPEC_P1PB.md changed after it was hashed"
    p1p = json.load(open(f"{HERE}/P1P_RESULT.json"))
    cs = {f: sha256_file(f"{HERE}/{f}") for f in ("p1p_tp_lib.py", "p1p_probe_refit.py", "p1p_read.py", "P1P_PROBE_MODELS.npz")}
    assert cs == p1p["code_sha256"], ("P1' code or probe artifact differs from the P1' run", cs)
    assert cs["p1p_tp_lib.py"] == sha256_file(f"{ART}/dagland/trigger_probe/tp_lib.py")
    a1, a2 = sha256_file(f"{ART}/dagland/motion_c/theory/THEORY_CRAWL_AMEND_1.md"), sha256_file(f"{ART}/dagland/motion_c/theory/THEORY_CRAWL_AMEND_2.md")
    assert a1.startswith("9d39ac53") and a2.startswith("0427c325"), (a1, a2)
    adm = json.load(open(f"{HERE}/P1PB_ADMISSIBILITY.json"))
    assert adm["admissible"] is True
    fw, f1 = json.load(open(f"{HERE}/P1PB_FW_CONTROL.json")), json.load(open(f"{HERE}/P1PB_F1_CONTROL.json"))
    assert fw["pass"] and f1["pass"], "forward controls failed: the holdout arm is not read"
    out.update({"spec_sha256": spec_sha, "spec_sha_file": open(f"{HERE}/SPEC_P1PB.sha256").read().splitlines()[0], "code_sha256": cs, "amend1_sha256": a1, "amend2_sha256": a2,
                "p1pb_code_sha256": {f: sha256_file(f"{HERE}/{f}") for f in ("p1pb_forward.py", "p1pb_read.py", "p1pb_admissibility.py")}, "admissibility_sha256": sha256_file(f"{HERE}/P1PB_ADMISSIBILITY.json"),
                "admissible": adm["admissible"], "holdout_sha256": adm["holdout_sha256"]})
    out["C4_unit_tests"] = P.unit_tests()
    assert out["C4_unit_tests"]["all_pass"], "C4 failed: no holdout window is scored"
    log("C4", out["C4_unit_tests"])
    out["FW_control"] = {"pass": fw["pass"], "per_seed": fw["per_seed"], "n_windows": fw["n_windows"]}
    out["F1_control"] = {k: v for k, v in f1.items() if k != "per_clip"}

    # ---------------------------------------------------------------- the tick-0 rows
    Z = {s: np.load(f"{HERE}/P1PB_TICK0_s{s}.npz", allow_pickle=False) for s in (0, 1)}
    for k in ("v0", "gt", "gt_ok", "ci", "t0", "clip", "kin_v0", "labelled"):
        assert np.array_equal(Z[0][k], Z[1][k], **({"equal_nan": True} if Z[0][k].dtype.kind in "fc" else {})), ("tick-0 rows differ between seeds", k)
    names, ci, t0s, v0, gt, ok_gt, lab = Z[0]["clip"], Z[0]["ci"], Z[0]["t0"], Z[0]["v0"], Z[0]["gt"], Z[0]["gt_ok"].astype(bool), Z[0]["labelled"].astype(bool)
    N = len(v0)
    plan = {s: Z[s]["plan"] for s in (0, 1)}
    plan_raw = {s: Z[s]["plan_raw"] for s in (0, 1)}
    out["holdout_windows"] = {"n_windows": int(N), "n_clips": int(len(np.unique(ci))), "n_windows_in_labelled_clips": int(lab.sum()), "n_clips_labelled": int(len(np.unique(ci[lab]))),
                              "clips_without_v9_rows": f1["clips_without_v9_rows"], "windows_in_clips_without_rows": int((~lab).sum())}
    # ---------------------------------------------------------------- probe features and posteriors (P1' code)
    need = np.flatnonzero(v0 < V_HI)
    log("feature windows", len(need), "of", N)
    V, E, kv = P.eval_features([(str(names[i]), int(t0s[i])) for i in need])
    sub = np.random.default_rng(0).choice(len(need), size=min(50, len(need)), replace=False)
    V2, E2, _ = P.eval_features([(str(names[need[j]]), int(t0s[need[j]])) for j in sub])
    out["F2_features_rebuilt_twice"] = {"n": int(len(sub)), "V_bit_equal": bool(np.array_equal(V[sub], V2)), "E_max_abs": float(np.abs(E[sub] - E2).max()), "v0_vs_kin3_max_abs": float(np.abs(v0[need] - kv).max())}
    out["F2_features_rebuilt_twice"]["pass"] = bool(out["F2_features_rebuilt_twice"]["V_bit_equal"] and out["F2_features_rebuilt_twice"]["E_max_abs"] == 0.0 and out["F2_features_rebuilt_twice"]["v0_vs_kin3_max_abs"] <= 1e-4)
    log("F2", out["F2_features_rebuilt_twice"])
    models = dict(np.load(f"{HERE}/P1P_PROBE_MODELS.npz"))
    d_all = {m: P.score(models, "go", m, V, E) for m in range(6)}
    d24 = {m: P.score(models, "go24", m, V, E) for m in range(5)}
    p_fold = np.stack([P.sigmoid(d_all[m]) for m in range(5)], 1)
    post = np.full(N, np.nan)
    post_all, post24 = post.copy(), post.copy()
    post[need] = p_fold.mean(1)
    post_all[need] = P.sigmoid(d_all[5])
    post24[need] = np.stack([P.sigmoid(d24[m]) for m in range(5)], 1).mean(1)
    go_h = L.go_label(v0, gt, 29)
    go24_h = L.go_label(v0, gt, 11)
    ah_all = (P.sec(gt) - v0) / 2

    # ---------------------------------------------------------------- K1-K3 (descriptive) on the holdout clips with rows
    def auc(mask, score, y):
        try:
            return L.auc_ci(score[mask], y[mask].astype(int), ci[mask])
        except Exception as e:                                                 # one class only
            return {"error": repr(e), "n": int(mask.sum())}
    pm = lab & (v0 >= 0.5) & (v0 < 6.0) & ok_gt
    b2 = lab & (v0 >= 3.0) & (v0 < 6.0) & ok_gt
    b8 = lab & (v0 >= V_LO) & (v0 < V_HI) & ok_gt
    k1, k2 = auc(pm, post, go_h), auc(b2, post, go_h)
    k3 = auc(b8, post, ah_all > A_MIN)
    for k in (k1, k2, k3):
        if "lo" in k:
            k["covers_0p5"] = bool(k["lo"] <= 0.5 <= k["hi"])
    out["K1_probe_auc_holdout_pmove"], out["K2_probe_auc_holdout_band_3_6"], out["K3_probe_auc_holdout_a_human_gt_0p5_band_3_8"] = k1, k2, k3
    out["probe_at_chance_in_band_K2"] = bool(k2.get("covers_0p5", True))
    log("K1", k1, "K2", k2, "K3", k3)

    # ---------------------------------------------------------------- the read (the arithmetic of p1p_read.main.run_cfg, window source replaced)
    def window_set(use, vlo, vhi, key_plan, post_vec):
        base_band = use & (v0 >= vlo) & (v0 < vhi)
        steps = {"tick0_rows": int(N), "in_scope_clips": int(use.sum()), "v0_band": int(base_band.sum()), "gt_ok": int((base_band & ok_gt).sum())}
        m = base_band & ok_gt
        m = m & (ah_all > A_MIN)
        steps["a_human_gt_0p5"] = int(m.sum())
        a_pl = {s: (P.sec(key_plan[s]) - v0) / 2 for s in (0, 1)}
        fin = m.copy()
        for s in (0, 1):
            fin &= np.isfinite(a_pl[s]) & np.isfinite(ah_all) & np.isfinite(post_vec)
        steps["finite"] = int(fin.sum())
        rho = {s: a_pl[s] / np.where(ah_all != 0, ah_all, np.nan) for s in (0, 1)}
        return fin, steps, rho, a_pl

    def run_cfg(name, use, vlo, vhi, key_plan, post_vec):
        fin, steps, rho, a_pl = window_set(use, vlo, vhi, key_plan, post_vec)
        cnt = L.boot_counts(ci[fin])
        per = {}
        for s in (0, 1):
            r = P.read_one(rho[s][fin], post_vec[fin], ci[fin], cnt=cnt)
            cm, um = P.groups(post_vec[fin])
            r["median_a_human_conf"] = float(np.median(ah_all[fin][cm])) if cm.any() else None
            r["median_a_human_unsure"] = float(np.median(ah_all[fin][um])) if um.any() else None
            r["median_a_plan_conf"] = float(np.median(a_pl[s][fin][cm])) if cm.any() else None
            r["median_a_plan_unsure"] = float(np.median(a_pl[s][fin][um])) if um.any() else None
            r["median_v0_conf"] = float(np.median(v0[fin][cm])) if cm.any() else None
            r["median_v0_unsure"] = float(np.median(v0[fin][um])) if um.any() else None
            r["conf_clip_stats"] = clip_counts(ci[fin][cm], "conf")
            r["unsure_clip_stats"] = clip_counts(ci[fin][um], "unsure")
            per[f"s{s}"] = r
        pf = post_vec[fin]
        res = {"name": name, "window_filter_steps": steps,
               "posterior_bins": {"lt_0.35": int((pf < 0.35).sum()), "0.35_0.65_UNSURE": int(((pf >= 0.35) & (pf <= 0.65)).sum()), "0.65_0.8": int(((pf > 0.65) & (pf < 0.8)).sum()), "ge_0.8_CONFIDENT": int((pf >= 0.8).sum())},
               "per_seed": per, "verdict": P.verdict(per), "n_windows_v0_ge_6": int((v0[fin] >= 6.0).sum())}
        return res, fin, rho, cnt

    grid4 = ((t0s - 3) % 4 == 0)
    primary, fin_p, rho_p, cnt_p = run_cfg("PRIMARY: holdout, every window, labelled clips, v0 [3,8), plan, 5-fold GO posterior", lab, V_LO, V_HI, plan, post)
    out["PRIMARY"] = primary
    log("PRIMARY", primary["verdict"], json.dumps({s: {k: r[k] for k in ("n_conf", "n_unsure", "median_conf", "median_unsure", "delta", "lo", "hi")} for s, r in primary["per_seed"].items()}))
    sens = {}
    sens["G4_grid_stride4_t3mod4"] = run_cfg("G4: t = 3 mod 4 (eval-141 grid construction)", lab & grid4, V_LO, V_HI, plan, post)[0]
    sens["S1_in_domain_v0_3_6"] = run_cfg("S1: v0 [3,6)", lab, 3.0, 6.0, plan, post)[0]
    sens["S2_raw_plan"] = run_cfg("S2: plan_raw", lab, V_LO, V_HI, plan_raw, post)[0]
    sens["S3_all_train_probe"] = run_cfg("S3: all-train probe posterior", lab, V_LO, V_HI, plan, post_all)[0]
    sens["S4_go24_label"] = run_cfg("S4: GO24 posterior", lab, V_LO, V_HI, plan, post24)[0]
    sens["S5_include_clips_without_v9_rows"] = run_cfg("S5: all 60 clips (zero nav where no v9 rows)", np.ones(N, bool), V_LO, V_HI, plan, post)[0]
    out["SENSITIVITIES_not_the_verdict"] = sens
    for k, r in sens.items():
        log(k, r["verdict"], json.dumps({s: {kk: x[kk] for kk in ("n_conf", "n_unsure", "delta", "lo", "hi")} for s, x in r["per_seed"].items()}))

    # ---------------------------------------------------------------- the pooled read, beside (eval-141 primary windows of P1' + holdout primary windows); decides nothing
    E141 = np.load(f"{HERE}/P1P_PERWINDOW.npz", allow_pickle=False)
    pm_e = E141["primary_mask"].astype(bool)
    assert int(pm_e.sum()) == p1p["PRIMARY"]["window_filter_steps"]["finite"] == 168
    nm_e = E141["clip"][pm_e].astype(str)
    nm_h = names[fin_p].astype(str)
    assert len(set(nm_e) & set(nm_h)) == 0, "eval-141 and holdout clip sets overlap"
    allnm = np.concatenate([nm_e, nm_h])
    _, cid = np.unique(allnm, return_inverse=True)
    post_pool = np.concatenate([E141["post_primary"][pm_e], post[fin_p]])
    pooled_per = {}
    cnt_pool = L.boot_counts(cid)
    for s in (0, 1):
        rho_pool = np.concatenate([E141[f"rho_s{s}"][pm_e], rho_p[s][fin_p]])
        pooled_per[f"s{s}"] = P.read_one(rho_pool, post_pool, cid, cnt=cnt_pool)
    out["POOLED_beside_decides_nothing"] = {"n_eval141_windows": int(pm_e.sum()), "n_holdout_windows": int(fin_p.sum()), "n_clips_pooled": int(len(np.unique(cid))), "per_seed": pooled_per, "verdict_logic_returns": P.verdict(pooled_per),
                                            "note": "eval-141's 168 windows are SEEN (P1P_RESULT.md); this table decides nothing"}
    # the eval-141 part alone, from the stored P1' arrays, for the side-by-side table
    cnt_e = L.boot_counts(np.unique(nm_e, return_inverse=True)[1])
    out["EVAL141_part_P1P_primary"] = {f"s{s}": P.read_one(E141[f"rho_s{s}"][pm_e], E141["post_primary"][pm_e], np.unique(nm_e, return_inverse=True)[1], cnt=cnt_e) for s in (0, 1)}
    log("POOLED", P.verdict(pooled_per), json.dumps({s: {k: r[k] for k in ("n_conf", "n_unsure", "median_conf", "median_unsure", "delta", "lo", "hi")} for s, r in pooled_per.items()}))

    # ---------------------------------------------------------------- N1 / N2 / N3 on the holdout primary window set
    post_p, ci_p = post[fin_p], ci[fin_p]
    n1 = {"seeds_1_to_20": {}}
    sep_count = {"s0": 0, "s1": 0}
    for ps in range(1, 21):
        perm = np.random.default_rng(ps).permutation(len(post_p))
        for s in (0, 1):
            r = P.read_one(rho_p[s][fin_p], post_p[perm], ci_p, cnt=cnt_p)
            sep_count[f"s{s}"] += int(r["separated"])
            n1["seeds_1_to_20"].setdefault(f"s{s}", []).append({"perm_seed": ps, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"]})
    n1["separated_of_20"] = sep_count
    n1["pass"] = bool(all(v <= 3 for v in sep_count.values()))
    out["N1_posterior_shuffle"] = n1
    one = np.ones(int(fin_p.sum()))
    n2 = {f"s{s}": P.read_one(one, post_p, ci_p, cnt=cnt_p) for s in (0, 1)}
    # N2 as specified by "plan := gt": rho = a_human / a_human = 1 exactly; recomputed from gt rather than assumed
    rho_gt = (P.sec(gt) - v0)[fin_p] / 2 / ah_all[fin_p]
    n2b = {f"s{s}": P.read_one(rho_gt, post_p, ci_p, cnt=cnt_p) for s in (0, 1)}
    exact = lambda r: bool(r["median_conf"] == 1.0 and r["median_unsure"] == 1.0 and r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["separated"] and not r["positive"])
    out["N2_identity_rho_equals_1"] = {"per_seed_ones": n2, "per_seed_from_plan_eq_gt": n2b, "pass": bool(all(exact(r) for r in n2.values()) and all(exact(r) for r in n2b.values()))}
    n3 = {f"s{s}": P.read_one(post_p.copy(), post_p, ci_p, cnt=cnt_p) for s in (0, 1)}
    out["N3_planted_rho_equals_posterior"] = {"per_seed": n3, "pass": bool(all(r["delta"] is not None and r["delta"] > 0 and r["separated"] for r in n3.values()))}
    log("N1", sep_count, n1["pass"], "N2", out["N2_identity_rho_equals_1"]["pass"], "N3", out["N3_planted_rho_equals_posterior"]["pass"])

    # ---------------------------------------------------------------- bank the per-window arrays
    sel = need
    np.savez(f"{HERE}/P1PB_PERWINDOW.npz", win=sel, clip=names[sel], ci=ci[sel], t0=t0s[sel], v0=v0[sel], a_human=ah_all[sel], post_primary=post[sel], post_all_train=post_all[sel], post_go24=post24[sel],
             post_fold=p_fold, go=go_h[sel], labelled=lab[sel], primary_mask=fin_p[sel], a_plan_s0=((P.sec(plan[0]) - v0) / 2)[sel], a_plan_s1=((P.sec(plan[1]) - v0) / 2)[sel],
             rho_s0=rho_p[0][sel], rho_s1=rho_p[1][sel])
    out["controls_summary"] = {"C4": out["C4_unit_tests"]["all_pass"], "FW": fw["pass"], "F1": f1["pass"], "F2": out["F2_features_rebuilt_twice"]["pass"], "N1": n1["pass"],
                               "N2": out["N2_identity_rho_equals_1"]["pass"], "N3": out["N3_planted_rho_equals_posterior"]["pass"], "probe_artifacts_unchanged": True}
    out["verdict"] = primary["verdict"]
    out["seconds"] = time.time() - T0
    json.dump(out, open(f"{HERE}/P1PB_RESULT.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("done; verdict", primary["verdict"], out["controls_summary"])


if __name__ == "__main__":
    main()
