#!/usr/bin/env python
"""Read-out for H-HJ-COLLAPSE-1: aggregate the per-episode JSON/moments written by
hj_collapse_diag.py into rho (paper Eq. 10), action sensitivities, the controls and the PREREG
outcome.  Pure numpy; no model, no GPU.  Works on a PARTIAL run (reports n_episodes).

    python hj_collapse_readout.py --work C:/Users/Admin/hjcollapse_work --out <pkg>/raw
Estimator everywhere: episode-cluster bootstrap, B=2000, seed 0 (answers: would another draw of
EPISODES say this?  It does NOT answer training-seed or inference-seed variance -- one
checkpoint, one deterministic forward, no sampling).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.environ.get("TANITAD_REPO", "E:/Projects/TanitAD")
sys.path.insert(0, os.path.join(_REPO, "taniteval"))
from taniteval import ci as TCI  # noqa: E402

KS = (1, 5, 10)
B = 2000
SEED = 0


def load_eps(work):
    eps = []
    for jp in sorted(glob.glob(os.path.join(work, "ep", "*.json"))):
        d = json.load(open(jp))
        if d.get("done"):
            d["_npz"] = jp[:-5] + "__moments.npz"
            eps.append(d)
    eps.sort(key=lambda d: d["ei"])
    return eps


# ------------------------------------------------------------------ rho
def rho_block(eps, key, rng_seed=SEED, n_boot=B):
    """rho = mean_e within_e / total  (ddof=1 both).  Bootstrap resamples EPISODES."""
    n = np.array([e["rho_scalars"][key]["n"] for e in eps], dtype=np.float64)
    ss = np.array([e["rho_scalars"][key]["ss"] for e in eps], dtype=np.float64)
    wi = np.array([np.nan if e["rho_scalars"][key]["within"] is None
                   else e["rho_scalars"][key]["within"] for e in eps], dtype=np.float64)
    ok = np.isfinite(wi) & (n >= 2)
    S = None
    G = np.zeros((len(eps), len(eps)))
    vecs = []
    for e in eps:
        z = np.load(e["_npz"])
        vecs.append(z[f"{key}__s"].astype(np.float64))
    S = np.stack(vecs)                         # [E, D]
    G = S @ S.T
    del S, vecs

    def stat(c):                               # c = episode multiplicity vector
        cc = c * ok
        N = float((cc * n).sum())
        within = float((cc * wi).sum() / cc.sum())
        tot = (float((cc * ss).sum()) - float(cc @ G @ cc) / N) / (N - 1.0)
        return within, tot, within / tot
    base = stat(np.ones(len(eps)))
    rng = np.random.default_rng(rng_seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(eps), size=len(eps))
        c = np.bincount(pick, minlength=len(eps)).astype(np.float64)
        boots.append(stat(c)[2])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": base[2], "lo": float(lo), "hi": float(hi),
            "within_trace": base[0], "total_trace": base[1],
            "n_episodes": int(ok.sum()), "n_vectors": int(n.sum()),
            "estimator": "episode-cluster bootstrap B=2000 seed 0 (ddof=1 within & pooled)"}


# ------------------------------------------------------------------ sens
def gather(eps, level, variant, k):
    num, den, eid, rms = [], [], [], []
    for e in eps:
        for w in e["windows"]:
            if "donor" not in w:
                continue
            r = w[level].get(variant)
            if r is None:
                continue
            num.append(r[f"k{k}"][0])
            den.append(r[f"k{k}"][1])
            eid.append(e["name"])
            rms.append(w["pert"][variant][f"{'op' if level == 'op' else 'tac'}_rms"])
    return (np.array(num), np.array(den), np.array(eid), np.array(rms))


def gather_stub(eps, kind, level, variant, k):
    num, den, eid = [], [], []
    for e in eps:
        for w in e["windows"]:
            if "donor" not in w:
                continue
            r = w["stub"][kind].get(variant)
            if r is None:
                continue
            num.append(r[level][f"k{k}"][0])
            den.append(r[level][f"k{k}"][1])
            eid.append(e["name"])
    return np.array(num), np.array(den), np.array(eid)


def sens_stats(num, den, eid):
    ok = den > 0
    out = {"n_windows": int(len(num)), "n_den_zero": int((~ok).sum())}
    if ok.sum() == 0:
        out.update({"mean": None})
        return out
    r = num[ok] / den[ok]
    e = eid[ok]
    out["mean_of_ratios"] = TCI.episode_cluster_bootstrap(r, e, "mean", B, SEED, dp=6)
    out["median_of_ratios"] = TCI.episode_cluster_bootstrap(r, e, "median", B, SEED, dp=6)
    out["p90_of_ratios"] = float(np.percentile(r, 90))
    out["max_ratio"] = float(r.max())
    # ratio of means: sum(num)/sum(den) with the same episode bootstrap
    uniq = np.unique(e)
    idx = {u: np.flatnonzero(e == u) for u in uniq}
    rng = np.random.default_rng(SEED)
    rm = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[p] for p in pick])
        rm.append(num[ok][sel].sum() / den[ok][sel].sum())
    out["ratio_of_means"] = {"point": float(num[ok].sum() / den[ok].sum()),
                             "lo": float(np.percentile(rm, 2.5)),
                             "hi": float(np.percentile(rm, 97.5))}
    out["num_mean"] = float(num[ok].mean())
    out["den_mean"] = float(den[ok].mean())
    out["num_max_abs"] = float(np.abs(num).max())
    return out


def terciles(num, den, rms_ch, label):
    ok = den > 0
    if ok.sum() < 6:
        return None
    r = num[ok] / den[ok]
    x = rms_ch[ok]
    qs = np.quantile(x, [1 / 3, 2 / 3])
    bins = np.digitize(x, qs)
    rows = []
    for b in range(3):
        m = bins == b
        rows.append({"bin": ["low", "mid", "high"][b], "n": int(m.sum()),
                     "perturb_rms_mean": float(x[m].mean()) if m.any() else None,
                     "sens_mean": float(r[m].mean()) if m.any() else None,
                     "sens_median": float(np.median(r[m])) if m.any() else None})
    # slope of numerator on perturbation magnitude (through origin), relative to den
    return {"channel": label, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="C:/Users/Admin/hjcollapse_work")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    eps = load_eps(a.work)
    if not eps:
        raise SystemExit("no finished episodes")
    meta = json.load(open(os.path.join(a.work, "meta.json")))
    res = {"n_episodes": len(eps), "episodes": [e["name"] for e in eps],
           "ckpt_step": meta["ckpt_step"], "strict": meta["strict"],
           "estimator": "episode-cluster bootstrap B=2000 seed 0; answers 'another draw of "
                        "EPISODES' only (single checkpoint, deterministic forward)"}
    tp = os.path.join(a.work, "tripwire.json")
    if os.path.exists(tp):
        res["tripwire_vs_forward"] = json.load(open(tp))

    # ---------------- rho
    rho = {}
    keys = ["feat_tok", "feat_full", "adp_tok", "adp_full", "adp10_tok", "adp10_full",
            "tac0_tok", "tac0_full"] \
        + [f"op{k}_{lv}" for k in KS for lv in ("tok", "full")] \
        + [f"tacp{k}_{lv}" for k in KS for lv in ("tok", "full")]
    for key in keys:
        rho[key] = rho_block(eps, key)
        print(f"rho {key:12s} {rho[key]['rho']:.4f} [{rho[key]['lo']:.4f}, {rho[key]['hi']:.4f}]"
              f"  n_ep={rho[key]['n_episodes']} n={rho[key]['n_vectors']}", flush=True)
    res["rho"] = rho

    # ---------------- sensitivities
    sens = {"op": {}, "tac": {}}
    for level in ("op", "tac"):
        for var in ("self", "all", "a", "steer", "v", "v0", "d_steer05", "d_a1"):
            sens[level][var] = {}
            for k in KS:
                num, den, eid, rms = gather(eps, level, var, k)
                if len(num) == 0:
                    continue
                st = sens_stats(num, den, eid)
                sens[level][var][f"k{k}"] = st
    res["sens"] = sens

    # EXPLORATORY robustness: the same ratio after the model's linear decoder `to_enc`
    # (op level only; entries 2,3 of each [num, den, num_enc, den_enc] record)
    senc = {}
    for var in ("self", "all", "a", "steer", "v", "v0", "d_steer05", "d_a1"):
        senc[var] = {}
        for k in KS:
            num, den, eid = [], [], []
            for e in eps:
                for w in e["windows"]:
                    if "donor" not in w:
                        continue
                    r = w["op"].get(var)
                    if r is None or len(r[f"k{k}"]) < 4:
                        continue
                    num.append(r[f"k{k}"][2])
                    den.append(r[f"k{k}"][3])
                    eid.append(e["name"])
            if num:
                senc[var][f"k{k}"] = sens_stats(np.array(num), np.array(den), np.array(eid))
    res["sens_enc_op"] = senc

    # perturbation sizes (input units): how big is the swap?
    pert = {}
    for level, key in (("op", "op_rms"), ("tac", "tac_rms")):
        pert[level] = {}
        for var in ("all", "a", "steer", "v", "v0", "d_steer05", "d_a1"):
            arr = np.array([w["pert"][var][key] for e in eps for w in e["windows"]
                            if "donor" in w])
            pert[level][var] = {"rms_a_mps2": float(np.sqrt((arr[:, 0] ** 2).mean())),
                                "rms_steer_rad": float(np.sqrt((arr[:, 1] ** 2).mean())),
                                "rms_vchan_(v/30)": float(np.sqrt((arr[:, 2] ** 2).mean())),
                                "median_steer_rad": float(np.median(arr[:, 1])),
                                "n": int(len(arr))}
    res["perturbation_input_units"] = pert

    # step displacement relative to the field norm (how far the rollout moves at all)
    fr = {"op": {}, "tac": {}}
    for level, zk in (("op", "z0n_op"), ("tac", "z0n_tac")):
        for k in KS:
            vals, zn = [], []
            for e in eps:
                for w in e["windows"]:
                    if "donor" in w and "all" in w[level]:
                        vals.append(w[level]["all"][f"k{k}"][1])
                        zn.append(w[zk])
            fr[level][f"k{k}"] = {"mean_den": float(np.mean(vals)),
                                  "mean_z0_norm": float(np.mean(zn)),
                                  "mean_den_over_z0norm": float(np.mean(np.array(vals) / np.array(zn))),
                                  "n": len(vals)}
    res["rollout_displacement_vs_field_norm"] = fr

    # exploratory: tercile of the swap magnitude (op, k=10) for steer / v / a
    expl = {}
    for var, ch, nm in (("steer", 1, "steer"), ("v", 2, "v/30"), ("a", 0, "a")):
        num, den, eid, rms = gather(eps, "op", var, 10)
        expl[var] = terciles(num, den, rms[:, ch], nm)
    res["exploratory_terciles_op_k10"] = expl

    # ---------------- controls
    ctr = {}
    for level in ("op", "tac"):
        num, den, eid, rms = gather(eps, level, "self", 10)
        allnum = {k: gather(eps, level, "self", k)[0] for k in KS}
        bit = [w[level]["self"].get("bit_identical") for e in eps for w in e["windows"]
               if "donor" in w and "self" in w[level]]
        mx = [w[level]["self"].get("maxabs", 0.0) for e in eps for w in e["windows"]
              if "donor" in w and "self" in w[level]]
        ctr[f"self_swap_{level}"] = {
            "n_windows": int(len(num)),
            "max_numerator_all_k": float(max(np.abs(v).max() for v in allnum.values())),
            "all_bit_identical": bool(all(b is True for b in bit)) if bit else None,
            "max_abs_field_diff": float(max(mx)) if mx else None,
            "reads_exactly_zero": bool(max(np.abs(v).max() for v in allnum.values()) == 0.0)}
    for kind in ("identity", "drift", "linear"):
        ctr[f"stub_{kind}"] = {}
        for level in ("op", "tac"):
            ctr[f"stub_{kind}"][level] = {}
            for var in ("self", "all", "steer", "v", "a"):
                ctr[f"stub_{kind}"][level][var] = {}
                for k in (1, 5, 10):
                    num, den, eid = gather_stub(eps, kind, level, var, k)
                    if len(num) == 0:
                        continue
                    s = sens_stats(num, den, eid)
                    ctr[f"stub_{kind}"][level][var][f"k{k}"] = {
                        "num_max_abs": float(np.abs(num).max()),
                        "den_min": float(den.min()), "den_mean": float(den.mean()),
                        "sens_mean": (s.get("mean_of_ratios") or {}).get("mean"),
                        "sens_lo": (s.get("mean_of_ratios") or {}).get("lo"),
                        "sens_hi": (s.get("mean_of_ratios") or {}).get("hi"),
                        "n_den_zero": s["n_den_zero"], "n": s["n_windows"]}
    res["controls"] = ctr

    # ---------------- WM-vs-persistence context (T0), if recorded
    wm = {}
    for kk in ("model", "const"):
        for k in KS:
            vals = [w["wm_mse"][kk][f"k{k}"] for e in eps for w in e["windows"]
                    if "wm_mse" in w]
            if vals:
                wm[f"{kk}_k{k}"] = {"mean": float(np.mean(vals)), "n": len(vals)}
    if wm:
        res["wm_mse_T0"] = wm

    # ---------------- PREREG outcome (rule committed in hj_collapse_diag.py docstring)
    o = sens["op"]
    steer = o["steer"]["k10"]["mean_of_ratios"]
    vv = o["v"]["k10"]["mean_of_ratios"]
    s_pt, v_pt = steer["mean"], vv["mean"]
    if s_pt < 0.05 and v_pt >= 5.0 * s_pt:
        outcome = "O-COLLAPSE"
    elif s_pt >= 0.10:
        outcome = "O-NOT"
    elif 0.05 <= s_pt < 0.10:
        outcome = "INCONCLUSIVE (0.05-0.10)"
    else:
        outcome = "INCONCLUSIVE (uncovered cell: steer<0.05 but v < 5x steer)"
    straddle = []
    for thr, nm in ((0.05, "0.05"), (0.10, "0.10")):
        if steer["lo"] < thr < steer["hi"]:
            straddle.append(nm)
    res["prereg"] = {
        "steer_sens_k10_op": {"mean": s_pt, "lo": steer["lo"], "hi": steer["hi"]},
        "v_sens_k10_op": {"mean": v_pt, "lo": vv["lo"], "hi": vv["hi"]},
        "v_over_steer_ratio": (v_pt / s_pt) if s_pt > 0 else None,
        "outcome_point_estimate": outcome,
        "steer_ci_straddles_thresholds": straddle,
        "n_windows": steer["n_windows"], "n_episodes": steer["n_episodes"]}
    print("PREREG:", json.dumps(res["prereg"]), flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "readout.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    # token-mean moments, for reproducibility of the tok-level rho from the repo alone
    tok = {}
    for key in [k for k in keys if k.endswith("_tok")]:
        tok[f"{key}__s"] = np.stack([np.load(e["_npz"])[f"{key}__s"] for e in eps])
        tok[f"{key}__ss"] = np.array([e["rho_scalars"][key]["ss"] for e in eps])
        tok[f"{key}__n"] = np.array([e["rho_scalars"][key]["n"] for e in eps])
    np.savez_compressed(os.path.join(a.out, "rho_moments_tokmean.npz"), **tok)
    print("wrote readout.json", flush=True)


if __name__ == "__main__":
    main()
