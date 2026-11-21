"""B-a / B-b / B-c / B-d (SPEC.md sec. 3) from the banked packs. CPU only; runs on Thor with the launch tree
on PYTHONPATH so `detection_metrics` / `slot_presence` are the run's own.

Fits (shift, temperature, Platt, the P = R gate) on a TRAIN pack set ONLY; scored on EVAL-DIAG. The TP / FP
labels of every slot come from `detection_metrics.greedy_rows(pk, 2.0)` over ALL slots (score-ordered greedy, so
a monotone transform never changes them); `summarise` is re-run on transformed packs for every transform so AP
invariance (control C5) is MEASURED, not assumed.
"""
import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_metrics as dm  # noqa: E402
from tanitad.eval import detection_metrics as det  # noqa: E402
from tanitad.models import slot_presence as sp  # noqa: E402

PRIOR_SHIFT = float(np.log((1 - sp.PRESENCE_PRIOR_FOCAL) / sp.PRESENCE_PRIOR_FOCAL))


def rows_of(packs, dist=det.PR_DIST_M):
    """Per slot row over ALL slots: z (the slot's own logit), kind (1 TP, 0 FP), episode sha12, window i.
    DontCare rows are dropped (removed from every count, A10)."""
    Z, K, E, Wn = [], [], [], []
    for wi, pk in enumerate(packs):
        for (conf, kind, band, slot, g) in det.greedy_rows(pk, dist):
            if kind < 0:
                continue
            Z.append(float(pk["logit"][slot]))
            K.append(int(kind))
            E.append(pk.get("sha12"))
            Wn.append(wi)
    return np.asarray(Z, np.float64), np.asarray(K, np.int64), np.asarray(E, object), np.asarray(Wn)


def npos_by_ep(packs):
    out = {}
    for pk in packs:
        out[pk.get("sha12")] = out.get(pk.get("sha12"), 0) + int(pk["pos"].sum())
    return out


def transform_logit(name, z, fit):
    if name == "T0":
        return z
    if name == "T1a":
        return z + PRIOR_SHIFT
    if name == "T1b":
        return z + fit["T1b_b"]
    if name == "T1c":
        ps, pis = fit["T1c_table"]
        return dm.logit(np.interp(dm.sigmoid(z), ps, pis))
    if name == "T2":
        return z / fit["T2_T"]
    if name == "T2b":
        return fit["T2b_a"] * z + fit["T2b_b"]
    if name == "T3":
        # the gate moves instead of the score: shift so that g* maps to 0.5 (monotone, AP unchanged)
        return z - float(dm.logit(fit["T3_gate"]))
    raise ValueError(name)


TRANSFORMS = ("T0", "T1a", "T1b", "T1c", "T2", "T2b", "T3")


def fit_transforms(train_packs, table):
    z, k, _, _ = rows_of(train_packs)
    fit = {"n_rows": int(z.size), "n_tp": int(k.sum())}
    _, b = dm.fit_affine(z, k, fit_a=False, fit_b=True)
    fit["T1b_b"] = b
    a, _ = dm.fit_affine(z, k, fit_a=True, fit_b=False)
    fit["T2_T"] = 1.0 / a
    a2, b2 = dm.fit_affine(z, k, fit_a=True, fit_b=True)
    fit["T2b_a"], fit["T2b_b"] = a2, b2
    pr = det.pr_equal_gate(train_packs)
    fit["T3_gate"] = float(pr["gate"])
    fit["T3_train"] = pr
    fit["T1a_shift"] = PRIOR_SHIFT
    fit["T1c_table"] = table
    return fit


def score(packs, head, fit, gate=0.5, with_summarise=True, B=1000, seed=0):
    z, k, E, _ = rows_of(packs)
    npos = npos_by_ep(packs)
    n_pos = sum(npos.values())
    eps = sorted(e for e in npos if e is not None) + ([None] if None in npos else [])
    out = {"n_pos": n_pos, "n_windows": len(packs), "n_episodes": len(eps), "n_rows": int(z.size),
           "n_tp_rows": int(k.sum())}
    per = {}
    base_ep = None
    for t in TRANSFORMS:
        zt = transform_logit(t, z, fit)
        p = dm.sigmoid(zt)
        conf = p >= gate
        n_conf, tp = int(conf.sum()), int((conf & (k == 1)).sum())
        prec = tp / n_conf if n_conf else None
        rec = tp / n_pos if n_pos else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else (0.0 if n_conf else None)
        e_all, tab = dm.ece(p, k)
        e05, _ = dm.ece(p, k, min_p=0.05)
        r = {"conf_ratio": n_conf / n_pos if n_pos else None, "n_conf": n_conf, "tp": tp,
             "prec": prec, "rec": rec, "f1": f1, "ECE_all": e_all, "ECE_ge005": e05,
             "sum_p_over_npos": float(p.sum() / n_pos) if n_pos else None, "reliability": tab,
             "hist_tp": np.histogram(p[k == 1], bins=20, range=(0, 1))[0].tolist(),
             "hist_fp": np.histogram(p[k == 0], bins=20, range=(0, 1))[0].tolist()}
        # per-episode census for the cluster bootstrap
        ep_conf = {e: [0, 0] for e in eps}
        for e, c_, kk in zip(E, conf, k):
            if c_:
                ep_conf[e][0] += 1
                ep_conf[e][1] += int(kk == 1)
        nc = np.array([ep_conf[e][0] for e in eps], float)
        ntp = np.array([ep_conf[e][1] for e in eps], float)
        npv = np.array([npos[e] for e in eps], float)
        r["_ep"] = (nc, ntp, npv)
        if with_summarise:
            tp_packs = []
            for pk in packs:
                q = dict(pk)
                q["logit"] = transform_logit(t, pk["logit"].astype(np.float64), fit)   # float64: no re-quantisation (C5)
                tp_packs.append(q)
            s = det.summarise(tp_packs, head, gate=gate)
            r["summarise"] = {kk: (None if (isinstance(v, float) and v != v) else v) for kk, v in s.items()
                              if kk.endswith(("conf_ratio", "prec@gate", "rec@gate", "f1@gate", "ap2m",
                                              "det_ap4_all_all", "det_ap2_all_all", "det_ap1_all_all",
                                              "det_ap0p5_all_all", "n_conf", "tp@gate", "n_pos",
                                              "auroc_matched", "auroc_objectness", "det_map2_all",
                                              "det_map4_all"))}
        per[t] = r
    # bootstrap (episode clusters, B draws): conf_ratio / prec / rec / f1 per transform, and the PAIRED F1 diff
    rng = np.random.default_rng(seed)
    n = len(eps)
    draws = {t: {"conf_ratio": [], "prec": [], "rec": [], "f1": []} for t in TRANSFORMS}
    dF = {t: [] for t in TRANSFORMS}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        f0 = None
        for t in TRANSFORMS:
            nc, ntp, npv = per[t]["_ep"]
            C, T, P = nc[idx].sum(), ntp[idx].sum(), npv[idx].sum()
            pr = T / C if C else np.nan
            rc = T / P if P else np.nan
            f = 2 * pr * rc / (pr + rc) if (C and P and pr + rc > 0) else (0.0 if C else np.nan)
            draws[t]["conf_ratio"].append(C / P if P else np.nan)
            draws[t]["prec"].append(pr)
            draws[t]["rec"].append(rc)
            draws[t]["f1"].append(f)
            if t == "T0":
                f0 = f
            dF[t].append(f - f0 if (f == f and f0 == f0) else np.nan)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return None if v.size == 0 else [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for t in TRANSFORMS:
        per[t]["ci95"] = {m: ci(draws[t][m]) for m in draws[t]}
        per[t]["ci95_f1_minus_T0_paired"] = ci(dF[t])
        per[t].pop("_ep")
    out["transforms"] = per
    out["interval"] = ("episode-cluster bootstrap over eval episodes, B=%d, 95 %% percentile; answers 'another "
                       "draw of EPISODES' only -- blind to training-run variance (one seed)" % B)
    # AP once on the untransformed rows (C5 compares the summarise values per transform)
    out["AP_rows"] = {"ap2m": det._ap([(c, kk) for (c, kk) in zip(dm.sigmoid(z), k)], n_pos)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-packs", required=True)
    ap.add_argument("--eval-packs", required=True)
    ap.add_argument("--sens-packs", default=None, help="TRAIN-CALIB256 packs (sensitivity fit)")
    ap.add_argument("--inrun-json", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", default="box3d,agent")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    tr = pickle.load(open(a.train_packs, "rb"))
    ev = pickle.load(open(a.eval_packs, "rb"))
    sens = pickle.load(open(a.sens_packs, "rb")) if a.sens_packs else None
    table = [x.tolist() for x in dm.focal_inversion_table(sp.presence_optimum)]
    res = {"definition": {
        "conf_ratio": "n_conf / n_pos; n_conf = greedy 2 m rows over slots with sigmoid(presence_logit) >= 0.5, "
                      "DontCare (IGNORE-matched) excluded; n_pos = VIS-1 positives",
        "source": "stack/tanitad/eval/detection_metrics.py:243-253 (_census), :262-275 (summarise, alarm band "
                  ":59 CONF_RATIO_BAND), gate slot_presence.py:68 DETECTION_GATE = 0.5",
        "prior_shift_T1a": PRIOR_SHIFT,
        "focal": {"alpha": sp.FOCAL_ALPHA, "gamma": sp.FOCAL_GAMMA, "prior": sp.PRESENCE_PRIOR_FOCAL,
                  "gate_match_belief_at_0p5": sp.gate_match_belief(0.5, "focal")}}}
    for head in a.heads.split(","):
        trp, evp = tr[head], ev[head]
        fit = fit_transforms(trp, table)
        r = {"fit_train_diag": {k: v for k, v in fit.items() if k != "T1c_table"},
             "eval": score(evp, head, fit, B=a.B)}
        if sens is not None:
            fs = fit_transforms(sens[head], table)
            r["fit_train_calib256"] = {k: v for k, v in fs.items() if k != "T1c_table"}
            r["eval_with_calib256_fit"] = score(evp, head, fs, with_summarise=False, B=a.B)
        # TRAIN-side reference (the fit set scored on itself -- NOT a result, a sanity row)
        r["train_self"] = score(trp, head, fit, with_summarise=False, B=200)
        res[head] = r
        t0 = r["eval"]["transforms"]
        print(f"[box] {head}: " + " | ".join(
            f"{t} cr {t0[t]['conf_ratio']:.3f} F1 {t0[t]['f1'] if t0[t]['f1'] is not None else float('nan'):.3f}"
            for t in TRANSFORMS), flush=True)
    if a.inrun_json:
        rec = json.load(open(a.inrun_json))
        res["B_a_inrun_reproduction"] = {k: v for k, v in (rec.get("inrun_row") or {}).items()
                                         if any(h in k for h in ("box3d_", "agent_")) and
                                         k.split("_", 2)[-1] in ("conf_ratio", "n_conf", "tp@gate", "n_pos",
                                                                 "conf_ratio_alarm", "ap2m", "auroc_matched")}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
