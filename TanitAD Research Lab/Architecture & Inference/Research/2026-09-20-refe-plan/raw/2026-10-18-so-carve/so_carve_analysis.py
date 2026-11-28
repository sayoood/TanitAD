#!/usr/bin/env python3
"""PREREG_SOCARVE.md stage 2 (light, numpy): the carve draws, the paired log-cluster bootstrap, the labels, the secondary readings and the controls, on the per-set arrays of stage 1.
    PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe so_carve_analysis.py [--selftest]
--selftest runs ONLY the planted-array controls (no stage-1 data needed).  Writes socarve.json."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SEED, NBOOT, NDRAW = 20260927, 10000, 10000
M_PRIMARY, M_SENS = 13, (6, 20)
RHO_VAL, RHO_TRAIN = 0.5, 0.25
BAR = 0.5                                   # the g5 bound magnitude (points)
TOL = 1e-4
QUANT = {"P2": "pi_comp", "P1": "pick_comp"}
COLS = {"TTC": 3, "NC": 0, "EP": 2}
R0_JSON = "C:/Users/Admin/r0d_out/raw/sft.jsonl"


# ------------------------------------------------------------------------------------------------------------------ primitives
def aggregate(d, logs):
    """-> (sorted unique log names, per-log sum of d, per-log set count)"""
    ul, inv = np.unique(np.asarray(logs).astype(str), return_inverse=True)
    return ul, np.bincount(inv, weights=np.asarray(d, np.float64), minlength=len(ul)), np.bincount(inv, minlength=len(ul)).astype(np.float64)


def Lof(s, c, idx):
    """L(S) = -D(S): minus the sets-weighted mean of d over the logs in idx (idx may repeat a log: a bootstrap multiset).  idx [..., k]."""
    return -(s[idx].sum(-1)) / c[idx].sum(-1)


def pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def label_fn(L_tr, L_va, L_c, ci_va):
    """PREREG s.7, as written.  -> (label, rho)"""
    G_tv, G_tc = L_va - L_tr, L_c - L_tr
    if not (G_tv > 0):
        return "INTERMEDIATE(NO-GAP)", float("nan")
    rho = G_tc / G_tv
    if rho >= RHO_VAL and ci_va[0] <= L_c <= ci_va[1]:
        return "CARVE-TRACKS-VAL", float(rho)
    if rho < RHO_TRAIN:
        return "CARVE-TRACKS-TRAIN", float(rho)
    return ("INTERMEDIATE(OUT-OF-CI)" if rho >= RHO_VAL else "INTERMEDIATE(MID)"), float(rho)


def make_resamples(ntr, nva, n=NBOOT, seed=SEED):
    """PREREG s.5, call order exactly: ti = rng.integers(0, ntr, ntr); vi = rng.integers(0, nva, nva); pos = rng.permutation(ntr)[:13]"""
    rng = np.random.default_rng(seed)
    TI, VI, POS = np.empty((n, ntr), np.int64), np.empty((n, nva), np.int64), np.empty((n, M_PRIMARY), np.int64)
    for b in range(n):
        TI[b] = rng.integers(0, ntr, ntr)
        VI[b] = rng.integers(0, nva, nva)
        POS[b] = rng.permutation(ntr)[:M_PRIMARY]
    return TI, VI, POS


def carve_draws(ntr, m, n=NDRAW, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(ntr, size=m, replace=False) for _ in range(n)])


class Resamples:
    def __init__(self, ntr, nva):
        self.TI, self.VI, self.POS = make_resamples(ntr, nva)
        self.CARVE_B = np.take_along_axis(self.TI, self.POS, axis=1)
        self.DR = {m: carve_draws(ntr, m) for m in (M_PRIMARY,) + M_SENS}


def pipeline(d_tr, logs_tr, d_va, logs_va, RS, with_boot=True):
    """-> dict of the section-5 estimands for ONE per-set quantity d (points)."""
    ul_t, s_t, c_t = aggregate(d_tr, logs_tr)
    ul_v, s_v, c_v = aggregate(d_va, logs_va)
    ntr, nva = len(ul_t), len(ul_v)
    all_t, all_v = np.arange(ntr), np.arange(nva)
    L_tr, L_va = float(Lof(s_t, c_t, all_t)), float(Lof(s_v, c_v, all_v))
    out = {"n_logs_train": ntr, "n_logs_val": nva, "n_sets_train": int(c_t.sum()), "n_sets_val": int(c_v.sum()), "L_tr": L_tr, "L_va": L_va}
    sens = {}
    for m, DR in RS.DR.items():
        Lc_r = Lof(s_t, c_t, DR)
        Lrest_r = -((s_t.sum() - s_t[DR].sum(1)) / (c_t.sum() - c_t[DR].sum(1)))
        sens[m] = {"L_c": float(Lc_r.mean()), "sd_L_c": float(Lc_r.std(ddof=1)), "share_L_gt_0": float((Lc_r > 0).mean()), "share_L_ge_bar": float((Lc_r >= BAR).mean()),
                   "L_rest_mean": float(Lrest_r.mean()), "G_tc": float(Lc_r.mean() - L_tr), "G_tv": L_va - L_tr,
                   "rho_vs_trainpool": (float((Lc_r.mean() - L_tr) / (L_va - L_tr)) if L_va - L_tr > 0 else float("nan")),
                   "G_tc_vs_rest": float((Lc_r - Lrest_r).mean()), "G_tv_vs_rest": float(L_va - Lrest_r.mean()),
                   "rho_vs_rest": (float((Lc_r - Lrest_r).mean() / (L_va - Lrest_r.mean())) if L_va - Lrest_r.mean() > 0 else float("nan"))}
    out["carve"] = {str(m): v for m, v in sens.items()}
    L_c = sens[M_PRIMARY]["L_c"]
    out["L_c"], out["G_tv"], out["G_tc"] = L_c, L_va - L_tr, L_c - L_tr
    if with_boot:
        Ltr_b, Lva_b = Lof(s_t, c_t, RS.TI), Lof(s_v, c_v, RS.VI)
        Lc_b = Lof(s_t, c_t, RS.CARVE_B)
        out["ci"] = {"L_tr": pct(Ltr_b), "L_va": pct(Lva_b), "L_c": pct(Lc_b), "G_tv": pct(Lva_b - Ltr_b), "G_tc": pct(Lc_b - Ltr_b)}
        lab, rho = label_fn(L_tr, L_va, L_c, out["ci"]["L_va"])
        out.update({"label": lab, "rho": rho, "F1_in_sample_carve": True, "F2_unresolved_gap": bool(out["ci"]["G_tv"][0] <= 0.0 <= out["ci"]["G_tv"][1]),
                    "L_c_within_val_ci": bool(out["ci"]["L_va"][0] <= L_c <= out["ci"]["L_va"][1])})
        out["_boot"] = {"Ltr_b": Ltr_b, "Lva_b": Lva_b}                 # stripped before dumping
    return out


def per_set(z, pool, quant, comp, swap=False):
    key, col = QUANT[quant], COLS[comp]
    B, A = z[f"{pool}__own__{key}"][:, col], z[f"{pool}__A_own__{key}"][:, col]
    if swap:
        B, A = A, B
    return (B - A) * 100.0


# ------------------------------------------------------------------------------------------------------------------ controls on planted arrays
def selftest(RS_logs):
    """PREREG s.9 items 1, 2, 4 on planted arrays that carry the REAL log structure (sets per log).  -> dict of named booleans"""
    logs_tr, logs_va, RS = RS_logs
    res = {}
    n_tr, n_va = len(logs_tr), len(logs_va)
    # planted: train -0.2, val -1.0 -> L_tr 0.2, L_va 1.0, G_tv 0.8
    p = pipeline(np.full(n_tr, -0.2), logs_tr, np.full(n_va, -1.0), logs_va, RS)
    res["planted_G_tv_is_0.8"] = abs(p["G_tv"] - 0.8) < 1e-12
    res["planted_in_sample_carve_labels_TRAIN"] = (p["label"] == "CARVE-TRACKS-TRAIN") and abs(p["rho"]) < 1e-9
    ci_va = p["ci"]["L_va"]
    res["label_fn_carve_eq_val"] = label_fn(p["L_tr"], p["L_va"], p["L_va"], ci_va)[0] == "CARVE-TRACKS-VAL"
    res["label_fn_carve_eq_train"] = label_fn(p["L_tr"], p["L_va"], p["L_tr"], ci_va)[0] == "CARVE-TRACKS-TRAIN"
    res["label_fn_carve_mid_0.4"] = label_fn(p["L_tr"], p["L_va"], p["L_tr"] + 0.4 * 0.8, ci_va)[0] == "INTERMEDIATE(MID)"
    res["label_fn_carve_out_of_ci"] = label_fn(p["L_tr"], p["L_va"], p["L_tr"] + 0.6 * 0.8, [0.99, 1.01])[0] == "INTERMEDIATE(OUT-OF-CI)"
    q = pipeline(np.full(n_tr, -0.2), logs_tr, np.full(n_va, -0.2), logs_va, RS)
    res["planted_no_gap_labels_NO_GAP"] = q["label"] == "INTERMEDIATE(NO-GAP)" and abs(q["G_tv"]) < 1e-12
    # zero control: B := A -> exactly 0.0 everywhere
    zc = pipeline(np.zeros(n_tr), logs_tr, np.zeros(n_va), logs_va, RS)
    flat = [zc["L_tr"], zc["L_va"], zc["L_c"], zc["G_tv"], zc["G_tc"]] + [v for k in ("L_tr", "L_va", "L_c", "G_tv", "G_tc") for v in zc["ci"][k]] \
        + [v for m in zc["carve"].values() for k, v in m.items() if k not in ("rho_vs_trainpool", "rho_vs_rest")]
    res["zero_control_all_exactly_zero"] = bool(np.array_equal(np.asarray(flat, np.float64), np.zeros(len(flat))))
    # sign control on a planted non-trivial array
    rng = np.random.default_rng(1)
    dt, dv = rng.normal(size=n_tr), rng.normal(size=n_va)
    a, b = pipeline(dt, logs_tr, dv, logs_va, RS), pipeline(-dt, logs_tr, -dv, logs_va, RS)
    res["sign_control_flips_exactly"] = (a["L_tr"] == -b["L_tr"] and a["L_va"] == -b["L_va"] and a["L_c"] == -b["L_c"] and a["G_tv"] == -b["G_tv"]
                                         and bool(np.allclose(a["ci"]["L_va"], [-b["ci"]["L_va"][1], -b["ci"]["L_va"][0]], rtol=1e-12, atol=0.0)))
    return res


def window_context():
    """RZ-1: R0d's own in-sample log, window updates 651..750 (the four 25-update rows 675 / 700 / 725 / 750)."""
    rows = [json.loads(l) for l in open(R0_JSON, encoding="utf-8") if '"event": "train"' in l[:40]]
    keys = ("con_g_ttc", "con_p1_gap_ttc", "con_cA_ttc", "con_cB_ttc", "con_g_nc", "con_p1_gap_nc", "con_cA_nc", "con_cB_nc", "con_lam_ttc", "con_lam_nc")
    win = [r for r in rows if 651 <= r["update"] <= 750]
    out = {"window_updates": [r["update"] for r in win], "n_rows": len(win)}
    for k in keys:
        v = [r.get(k) for r in win]
        out[k] = {"rows": v, "mean": (float(np.mean(v)) if all(x is not None for x in v) else None)}
    out["series"] = {k: [(r["update"], r.get(k)) for r in rows] for k in ("con_g_ttc", "con_p1_gap_ttc", "con_lam_ttc")}
    m = out["con_p1_gap_ttc"]["mean"]
    out["p1_gap_ttc_x100_window_mean"] = None if m is None else 100 * m
    out["tag"] = None if m is None else ("P1-IN-SAMPLE-HELD" if 100 * m > -0.5 else "P1-IN-SAMPLE-BREACHED")
    return out


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    z = {}
    for run, u in (("r0", 250), ("r0", 500), ("r0", 1000), ("c0", 1000)):
        p = os.path.join(OUT, f"persets_{run}_u{u}.npz")
        z[(run, u)] = dict(np.load(p, allow_pickle=False))
    ctl = json.load(open(os.path.join(HERE, "harness_control.json"), encoding="utf-8"))
    res = {"tool": "so_carve_analysis/1", "prereg_sha256_file": open(os.path.join(HERE, "PREREG_SOCARVE.sha256")).read().split()[0],
           "harness_control_verdict": ctl.get("verdict"), "controls": {}, "states": {}}
    z1000 = z[("r0", 1000)]
    logs_tr, logs_va = z1000["train__logs"], z1000["val__logs"]
    RS = Resamples(len(np.unique(logs_tr)), len(np.unique(logs_va)))
    res["controls"]["planted_and_label_controls"] = selftest((logs_tr, logs_va, RS))
    # log structure identical across states (a precondition of the paired resamples)
    res["controls"]["same_pool_in_every_state"] = all(np.array_equal(z[k]["train__keys"], z1000["train__keys"]) and np.array_equal(z[k]["val__keys"], z1000["val__keys"]) for k in z)
    if ctl.get("verdict") != "VALID":
        res["verdict"] = "INVALID (harness control failed, PREREG s.4): no outcome is drawn"
        json.dump(res, open(os.path.join(HERE, "socarve.json"), "w"), indent=1)
        print(res["verdict"]); return 3
    # real-array controls: arrays reproduce the banked per-pool means; zero control through real arrays; oracle carve labels
    for (run, u), zz in z.items():
        if run != "r0":
            continue
        full = ctl["states"][f"r0_u{u}"]["FULL"]
        for q in ("P2", "P1"):
            for pool in ("train", "val"):
                d = per_set(zz, pool, q, "TTC")
                res["controls"][f"array_mean_vs_banked_{q}_TTC_{pool}_u{u}"] = bool(abs(float(d.mean()) - full[f"{q}_TTC"][pool]) <= TOL)
    zd_tr, zd_va = per_set(z1000, "train", "P2", "TTC", False) * 0.0, per_set(z1000, "val", "P2", "TTC", False) * 0.0
    zc = pipeline(zd_tr, logs_tr, zd_va, logs_va, RS)
    res["controls"]["zero_control_real_logs"] = bool(zc["L_tr"] == 0.0 and zc["L_va"] == 0.0 and zc["L_c"] == 0.0 and zc["G_tv"] == 0.0 and zc["ci"]["G_tv"] == [0.0, 0.0])
    # A-vs-A through the real arrays (B := A)
    aa = [(z1000[f"{pool}__A_own__pi_comp"][:, 3] - z1000[f"{pool}__A_own__pi_comp"][:, 3]) * 100 for pool in ("train", "val")]
    res["controls"]["A_minus_A_is_exactly_zero"] = bool(not aa[0].any() and not aa[1].any())
    # base arrays are state-independent (they are the frozen base's): same in every state
    res["controls"]["base_arrays_state_independent"] = all(np.array_equal(z[k][f"{pool}__base__pi_comp"], z1000[f"{pool}__base__pi_comp"]) for k in z for pool in ("train", "val"))
    res["controls"]["c0_floor_pooled_abs_P2_TTC"] = ctl["states"]["c0_u1000"]["c0_floor_abs_P2_TTC_pooled"]

    # ---- the readings
    for (run, u) in ((("r0", 1000), ("r0", 500), ("r0", 250), ("c0", 1000))):
        zz = z[(run, u)]
        st = {}
        for q in ("P2", "P1"):
            for comp in ("TTC", "NC", "EP"):
                p = pipeline(per_set(zz, "train", q, comp), logs_tr, per_set(zz, "val", q, comp), logs_va, RS)
                p.pop("_boot")
                p["D_tr"], p["D_va"], p["D_c"] = -p["L_tr"], -p["L_va"], -p["L_c"]          # D = B - A (gain positive), for the EP rows
                st[f"{q}_{comp}"] = p
        res["states"][f"{run}_u{u}"] = st

    # oracle-carve label controls on the REAL R0 u1000 arrays
    p = res["states"]["r0_u1000"]["P2_TTC"]
    res["controls"]["real_oracle_carve_eq_val_label"] = label_fn(p["L_tr"], p["L_va"], p["L_va"], p["ci"]["L_va"])[0] if p["G_tv"] > 0 else "n/a (G_tv<=0)"
    res["controls"]["real_carve_eq_trainpool_label"] = label_fn(p["L_tr"], p["L_va"], p["L_tr"], p["ci"]["L_va"])[0] if p["G_tv"] > 0 else "n/a (G_tv<=0)"

    # ---- S-DOSE
    g = {u: res["states"][f"r0_u{u}"]["P2_TTC"] for u in (250, 500, 1000)}
    dose = {"G_tv": {str(u): {"point": g[u]["G_tv"], "ci": g[u]["ci"]["G_tv"]} for u in g}}
    if g[1000]["G_tv"] > 0 and g[250]["G_tv"] > 0:
        r = g[250]["G_tv"] / g[1000]["G_tv"]
        dose["r_d"] = r
        dose["tag"] = "EXPOSURE-GROWN" if r < 0.25 else ("MIXED" if r < 0.75 else "PRESENT-AT-ONE-PASS")
    else:
        dose["r_d"], dose["tag"] = None, "undefined (G_tv(u250) or G_tv(u1000) <= 0)"
    res["S_DOSE"] = dose

    # ---- S-EXCH: base-scorer and label-only covariates, train pool minus val pool (independent log resamples)
    ex = {}
    covs = {"c0_TTC_x100": ("pi_comp", 3, 100.0), "c0_NC_x100": ("pi_comp", 0, 100.0), "c0_EP_x100": ("pi_comp", 2, 100.0)}
    arr = {}
    for pool, logs in (("train", logs_tr), ("val", logs_va)):
        a = {k: z1000[f"{pool}__base__{key}"][:, col] * sc for k, (key, col, sc) in covs.items()}
        a["base_pick_PDMS_x100"] = z1000[f"{pool}__base__pick_pdms"] * 100.0
        a["oracle_mean_yTTC_x100"] = z1000[f"{pool}__tg6mean"][:, 3] * 100.0
        a["oracle_mean_yNC_x100"] = z1000[f"{pool}__tg6mean"][:, 0] * 100.0
        arr[pool] = a
    ul_t, ul_v = np.unique(logs_tr), np.unique(logs_va)
    for k in arr["train"]:
        _, st_, ct_ = aggregate(arr["train"][k], logs_tr)
        _, sv_, cv_ = aggregate(arr["val"][k], logs_va)
        mt, mv = st_.sum() / ct_.sum(), sv_.sum() / cv_.sum()
        diff_b = st_[RS.TI].sum(1) / ct_[RS.TI].sum(1) - sv_[RS.VI].sum(1) / cv_[RS.VI].sum(1)
        ex[k] = {"train": float(mt), "val": float(mv), "diff_train_minus_val": float(mt - mv), "ci": pct(diff_b)}
        ex[k]["ci_contains_0"] = bool(ex[k]["ci"][0] <= 0 <= ex[k]["ci"][1])
    res["S_EXCH"] = {"covariates": ex, "tag": "EXCH-NOT-CONTRADICTED" if all(v["ci_contains_0"] for v in ex.values()) else "EXCH-CONTRADICTED: " + ", ".join(k for k, v in ex.items() if not v["ci_contains_0"])}

    # ---- per-log table (R0 u1000)
    pl = {}
    for pool, logs in (("train", logs_tr), ("val", logs_va)):
        rows = []
        for q in ("P2", "P1"):
            ul, s, c = aggregate(per_set(z1000, pool, q, "TTC"), logs)
            pl[(pool, q)] = (ul, s, c)
        ul = pl[(pool, "P2")][0]
        res.setdefault("per_log_r0_u1000", {})[pool] = [{"log": str(l), "n_sets": int(pl[(pool, "P2")][2][i]), "L_P2_TTC": float(-pl[(pool, "P2")][1][i] / pl[(pool, "P2")][2][i]),
                                                          "L_P1_TTC": float(-pl[(pool, "P1")][1][i] / pl[(pool, "P1")][2][i])} for i, l in enumerate(ul)]
    # ---- RZ-1
    res["RZ1_R0d_in_sample"] = window_context()
    # ---- the decision
    dec = res["states"]["r0_u1000"]["P2_TTC"]
    res["DECISION_R0_u1000_P2_TTC"] = {"label": dec["label"], "rho": dec["rho"], "F1_in_sample_carve": True, "F2_unresolved_gap": dec["F2_unresolved_gap"],
                                       "L_tr": dec["L_tr"], "L_va": dec["L_va"], "L_c": dec["L_c"], "G_tv": dec["G_tv"], "G_tc": dec["G_tc"], "ci": dec["ci"],
                                       "L_c_within_val_ci": dec["L_c_within_val_ci"]}
    ok = all(v for k, v in res["controls"]["planted_and_label_controls"].items()) and all(v is True for k, v in res["controls"].items() if k.startswith(("array_mean", "zero_control", "A_minus", "base_arrays", "same_pool")))
    if p["G_tv"] > 0:
        ok = ok and res["controls"]["real_oracle_carve_eq_val_label"] == "CARVE-TRACKS-VAL" and res["controls"]["real_carve_eq_trainpool_label"] == "CARVE-TRACKS-TRAIN"
    res["all_controls_pass"] = bool(ok)
    json.dump(res, open(os.path.join(HERE, "socarve.json"), "w"), indent=1)
    print(json.dumps({"controls": res["controls"], "all_controls_pass": ok, "DECISION": res["DECISION_R0_u1000_P2_TTC"]}, indent=1))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        # planted-array controls only, on a synthetic log structure shaped like the pools (30 train logs, 13 val logs)
        rng = np.random.default_rng(0)
        cnt_t = [1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 7, 7, 7, 7, 7, 10, 10, 12, 14, 20, 23, 23, 27, 37, 38]
        cnt_v = [1, 3, 3, 3, 3, 4, 5, 5, 5, 5, 7, 9, 9]
        lt = np.concatenate([[f"t{i:02d}"] * n for i, n in enumerate(cnt_t)])
        lv = np.concatenate([[f"v{i:02d}"] * n for i, n in enumerate(cnt_v)])
        r = selftest((lt, lv, Resamples(len(cnt_t), len(cnt_v))))
        print(json.dumps(r, indent=1))
        sys.exit(0 if all(r.values()) else 1)
    sys.exit(main())
