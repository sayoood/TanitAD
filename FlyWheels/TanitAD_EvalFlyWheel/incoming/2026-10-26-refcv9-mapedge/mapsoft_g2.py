"""mapsoft_g2.py -- PREREG_MAPSOFT_G2.md: MAPSOFT-G with the gate grid r0 in {0, 2, 4, 6, 10, 20} m on the own-map features, cross-fitted.  CPU, 0 GPU.

    python mapsoft_g.py --out <dir> [--allow-partial --trace ... --rast ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapsoft as MSOFT  # noqa: E402
from mapedge import boundary  # noqa: E402

Q2, MS, MSDIR = MSOFT.Q2, MSOFT.MS, MSOFT.MSDIR
L = MS.L
R0 = (0, 2, 4, 6, 10, 20)
NEW = (("MAPSOFT-G-XF", "softg", 0), ("DER-MAPSOFT-G-XF", "dersoftg", 0), ("SOFTG0", "softg0", 0))
STATE = {}
_orig_picks_for = MS.picks_for


def pass1_gate(tr, rdir, has_gt, hum=None, limit=None):
    F, acc_own, acc_der, don, n_samples, hum_pmin = Q2.pass1_bnd(tr, rdir, has_gt, hum, limit)
    n = F["own_off"].shape[0]
    for r in R0:
        for k in ("pswmax", "dpswmax"):
            F[f"{k}_r{r}"] = np.zeros((n, MS.N_FAN), np.float32)
        for k in ("ownbnd", "derbnd"):
            F[f"{k}_r{r}"] = np.zeros((n, MS.N_FAN), bool)
    for j in range(n):
        g = L.Geom(torch.from_numpy(tr["fan"][j]))
        x = g.ix.float() * 0.1
        o = np.load(MS.rast_file(rdir, tr["e_i"][j], tr["ws"][j]))
        planes = {"own": (o["psw"], o["pdrv"])}
        if don[j] >= 0:
            d = np.load(MS.rast_file(rdir, tr["e_i"][don[j]], tr["ws"][don[j]]))
            planes["der"] = (d["psw"], d["pdrv"])
        for src, (psw, pdrv) in planes.items():
            ps = torch.as_tensor(psw)[g.ix, g.iy].float() / 255.0
            ed = L.dilate(boundary(pdrv >= L.PDRV_MIN_U8), L.EDGE_DIL)[g.ix, g.iy]
            for r in R0:
                m = g.inside & (x >= r)
                pk, bk = ("pswmax", "ownbnd") if src == "own" else ("dpswmax", "derbnd")
                F[f"{pk}_r{r}"][j] = torch.where(m, ps, torch.zeros_like(ps)).flatten(1).amax(1).numpy()
                F[f"{bk}_r{r}"][j] = (m & ed).flatten(1).any(1).numpy()
        if j % 500 == 0:
            print(f"[mapsoft_g2] gate pass {j + 1}/{n}", flush=True)
    STATE["sk3"] = bool(np.array_equal(F["pswmax_r0"], F["pswmax"]) and np.array_equal(F["ownbnd_r0"], F["ownbnd_edge"])
                        and np.array_equal(F["dpswmax_r0"][don >= 0], F["dpswmax"][don >= 0]) and np.array_equal(F["derbnd_r0"], F["derbnd_edge"]))
    return F, acc_own, acc_der, don, n_samples, hum_pmin


def picks_for_gate(tr, elig, F, lc, n):
    saved = MS.ARMS
    MS.ARMS = tuple(a for a in saved if a[1] not in ("softg", "dersoftg", "softg0"))
    out, fbs, kept = _orig_picks_for(tr, elig, F, lc, n)
    MS.ARMS = saved
    keys = list(zip(tr["e_i"].tolist(), tr["ws"].tolist()))[:n]
    hp, hvalid = MS.D.human_plans(keys)
    fit_ok = MSOFT.s6c_mask(tr, n) & np.asarray(tr["gt_label"], bool)[:n] & hvalid
    z = MSOFT.zscore(tr["sel_score_v3"][:n], elig)
    ar = np.arange(n)
    Hc = np.linalg.norm(tr["fan"][:n].astype(np.float64) - hp[:, None].astype(np.float64), axis=-1).mean(-1)
    Vc = (F["gt_off"] | F["gt_edge"]).astype(np.float64)
    eps = np.asarray(tr["e_i"][:n])
    ue = np.unique(eps)
    perm = np.random.default_rng(0).permutation(len(ue))
    fold_of_ep = {int(ue[i]): int(r % 5) for r, i in enumerate(perm)}
    fold = np.array([fold_of_ep[int(e)] for e in eps])
    grid = [(r, a, b) for r in R0 for a in MSOFT.GRID for b in MSOFT.GRID]
    fits = {}
    for nm, pk, bk in (("MAPSOFT-G-XF", "pswmax", "ownbnd"), ("DER-MAPSOFT-G-XF", "dpswmax", "derbnd")):
        P = {gp: MSOFT.soft_pick(z, elig, F[f"{pk}_r{gp[0]}"].astype(np.float64), F[f"{bk}_r{gp[0]}"].astype(np.float64), gp[1], gp[2]) for gp in grid}
        Vg = {gp: Vc[ar, p] for gp, p in P.items()}
        Hg = {gp: Hc[ar, p] for gp, p in P.items()}
        pick = np.zeros(n, np.int64)
        per_fold = {}
        for k in range(5):
            tm = fit_ok & (fold != k)
            h0 = Hg[(0, 0.0, 0.0)][tm].mean()
            feas = [gp for gp in grid if Hg[gp][tm].mean() <= h0]
            best = min(feas, key=lambda gp: (Vg[gp][tm].mean(), gp[1] + gp[2], gp[1], gp[0])) if feas else (0, 0.0, 0.0)
            per_fold[k] = {"r0": best[0], "lam_s": best[1], "lam_e": best[2], "fit_V": float(Vg[best][tm].mean()), "fit_V_base": float(Vg[(0, 0.0, 0.0)][tm].mean()),
                           "fit_H": float(Hg[best][tm].mean()), "fit_H_base": float(h0), "n_fit": int(tm.sum()), "n_feasible": len(feas)}
            pick[fold == k] = P[best][fold == k]
        fits[nm] = per_fold
        out[nm], fbs[nm], kept[nm] = pick, np.zeros((n, 3), np.int8), elig.sum(1).astype(np.int32)
    out["SOFTG0"] = MSOFT.soft_pick(z, elig, F["pswmax_r0"], F["ownbnd_r0"].astype(np.float64), 0.0, 0.0)
    fbs["SOFTG0"], kept["SOFTG0"] = np.zeros((n, 3), np.int8), elig.sum(1).astype(np.int32)
    STATE.update(fits=fits, sk1=bool(np.array_equal(out["SOFTG0"], out["BASE"])),
                 sk2=bool(len(fold_of_ep) == len(ue) and set(fold_of_ep.values()) <= set(range(5))), n_fit_windows=int(fit_ok.sum()))
    return out, fbs, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default="F:/refcv9_mapsel/trace_s0.npz")
    ap.add_argument("--rast", default="F:/refcv9_mapsel/rast_s0")
    ap.add_argument("--allow-partial", action="store_true", help="DRY-RUN ONLY (smoke bank)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = hashlib.sha256((HERE / "PREREG_MAPSOFT_G2.md").read_bytes()).hexdigest()
    if pre != (HERE / "PREREG_MAPSOFT_G2.sha256").read_text().split()[0]:
        raise SystemExit("[mapsoft_g2] PREREG_MAPSOFT_G2.md does not match its seal: refusing")
    MS.pass1 = pass1_gate
    MS.picks_for = picks_for_gate
    MS.ARMS = tuple(MS.ARMS) + NEW
    msout = out / "mapsel_run"
    sys.argv = ["mapsel_score.py", "--seed", "0", "--trace", a.trace, "--rast", a.rast, "--out", str(msout), "--human-json", str(MSDIR / "result" / "K_HUMAN.json")] + (["--allow-partial"] if a.allow_partial else [])
    MS.main()
    res = json.loads((msout / "MAPSEL_RESULT.json").read_text(encoding="utf-8"))
    ref = json.loads((MSDIR / "result" / "s0" / "MAPSEL_RESULT.json").read_text(encoding="utf-8"))
    mc, mr = res["map_compliance"]["S6c"]["arms"], ref["map_compliance"]["S6c"]["arms"]
    mism = [f"{nm}:{k}" for nm in mr for k in ("off_drivable", "edge_contact", "lane_cross") if nm in mc and (mc[nm][k].get("rate") != mr[nm][k].get("rate") or mc[nm][k].get("vs_BASE") != mr[nm][k].get("vs_BASE"))]
    hard_fail = sorted(k for k, v in res.get("controls", {}).items() if isinstance(v, dict) and v.get("pass") is False)
    ctl = {"S-K0_mapsel_controls_and_reproduction": {"pass": bool(not hard_fail and not mism and all(nm in mc for nm in mr)), "hard_failing": hard_fail, "mismatches": mism[:20]},
           "S-K1_soft00_equals_BASE": {"pass": STATE["sk1"]}, "S-K2_folds_partition_episodes": {"pass": STATE["sk2"]}, "S-K3_r0_equals_MAPSOFT_features": {"pass": STATE["sk3"]}}
    cl = MS.arm_clauses(res, "MAPSOFT-G-XF")
    c1, c2e = mc["MAPSOFT-G-XF"]["off_drivable"]["vs_BASE"], mc["MAPSOFT-G-XF"]["edge_contact"]["vs_BASE"]
    d1, d2 = mc["DER-MAPSOFT-G-XF"]["off_drivable"]["vs_BASE"], mc["DER-MAPSOFT-G-XF"]["edge_contact"]["vs_BASE"]
    below = lambda c: bool("hi" in c and c["hi"] < 0)  # noqa: E731
    bars = {"M1": below(c1), "M2": below(c2e), "M3": bool(cl["C2"]), "M4": bool(not below(d1) and not below(d2)), "M5": bool(cl["C4"])}
    fo = STATE["fits"]["MAPSOFT-G-XF"]
    weak = sum(1 for f in fo.values() if f["r0"] == 0 or (f["lam_s"] == 0 and f["lam_e"] == 0))
    ctl_fail = sorted(k for k, v in ctl.items() if not v["pass"])
    if ctl_fail:
        word, nxt = f"VOID (control failed: {ctl_fail})", "fix the wrapper; re-run (0 GPU)."
    elif all(bars.values()):
        word = "PASS (family chosen on eval, parameters cross-fitted)"
        nxt = "re-read on the fresh L1C split (map rasters for those clips: a GPU roll with the same eval tooling); only then an opt-in inference row."
    elif weak >= 3:
        word = "FAIL (" + ", ".join(k for k, v in bars.items() if not v) + ")"
        nxt = "the gate does not rescue the own map: map-head training for sidewalk / boundary precision (compute -> PI decision), first target the near-field false alarm."
    elif not (bars["M1"] and bars["M2"]):
        word = "FAIL (" + ", ".join(k for k, v in bars.items() if not v) + ")"
        nxt = "the 2-parameter cost is too weak: a learned listwise selector over richer own-map features."
    else:
        word = "FAIL (" + ", ".join(k for k, v in bars.items() if not v) + ")"
        nxt = "the trade-off costs driving at the fitted point: constrain ALL four families in the fit (multi-objective)."
    S6 = lambda nm: {k: mc[nm][k].get("rate") for k in ("off_drivable", "edge_contact", "lane_cross")}  # noqa: E731
    base, gt1 = mc["BASE"]["off_drivable"].get("rate"), mc["GT-T1"]["off_drivable"].get("rate")
    share = (base - mc["MAPSOFT-G-XF"]["off_drivable"]["rate"]) / (base - gt1) if (base is not None and gt1 is not None and base != gt1) else None
    r = {"tool": "mapsoft_g.py", "prereg_sha256": pre, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "controls": ctl, "bars": bars,
         "cells": {"M1_sidewalk": c1, "M2_edge": c2e, "DER_sidewalk": d1, "DER_edge": d2}, "M3_worse_rows": cl["C2_worse_rows"],
         "M5_guard": {k: res["progress_guard"]["S6c"]["MAPSOFT-G-XF"][k] for k in ("path_len_m", "stop_share")},
         "ade_S6c": res["four_families"]["S6c_stage"]["MAPSOFT-G-XF"].get("ade_m"), "fits": STATE["fits"], "n_fit_windows": STATE["n_fit_windows"], "weak_folds": weak,
         "rates_S6c": {nm: S6(nm) for nm in ("BASE", "HUMAN", "GT-T1", "OWN-T1", "MAPSOFT-G-XF", "DER-MAPSOFT-G-XF")}, "share_of_GT_T1_sidewalk_gain": share,
         "verdict": word, "next": nxt}
    MS.jdump(r, out / "MAPSOFT_G2_RESULT.json")
    f = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and "delta" in c else "-"  # noqa: E731
    g = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"  # noqa: E731
    Lm = [word, "", f"# MAPSOFT-G (mechanical; PREREG_MAPSOFT_G sha256 {pre[:12]})", "",
          "tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); estimator: paired episode-cluster bootstrap B 2000 seed 0 over EPISODES only; one training seed; one inference seed", "",
          "| arm (S6c) | sidewalk | edge contact | lane cross |", "|---|---|---|---|"] + \
         [f"| {nm} | {g(v['off_drivable'])} | {g(v['edge_contact'])} | {g(v['lane_cross'])} |" for nm, v in r["rates_S6c"].items()] + \
         ["", f"M1 sidewalk vs BASE {f(c1)} -> {bars['M1']}", f"M2 edge vs BASE {f(c2e)} -> {bars['M2']}", f"M3 no four-family row worse -> {bars['M3']} {cl['C2_worse_rows'] or ''}",
          f"M4 deranged not below BASE (sidewalk {f(d1)}, edge {f(d2)}) -> {bars['M4']}", f"M5 STOP guard -> {bars['M5']}", f"S6c ADE vs BASE {f(r['ade_S6c'])}", "",
          "fitted (r0, lam_s, lam_e) per fold (own): " + json.dumps({k: (v['r0'], v['lam_s'], v['lam_e']) for k, v in fo.items()}),
          "fitted per fold (deranged): " + json.dumps({k: (v['r0'], v['lam_s'], v['lam_e']) for k, v in STATE['fits']['DER-MAPSOFT-G-XF'].items()}),
          f"share of GT-T1's sidewalk gain: {share}", "", f"controls failed: {ctl_fail or 'none'}", "", f"NEXT: {nxt}", "", "ZZMSOFTGENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(Lm), encoding="utf-8")
    print(word, bars, ctl_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
