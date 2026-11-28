"""mapsoft.py -- PREREG_MAPSOFT.md: a soft, episode-cross-fitted own-map cost on refcv8's fan, scored through MAPSEL's pipeline.  CPU, 0 GPU.

    python mapsoft.py --out <dir> [--trace F:/refcv9_mapsel/trace_s0.npz --rast F:/refcv9_mapsel/rast_s0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapedge_q2 as Q2  # noqa: E402  (imports mapsel_score UNCHANGED and defines the boundary pass)

MS = Q2.MS
MSDIR = Q2.MSDIR
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NEW = (("MAPSOFT-XF", "soft", 0), ("DER-MAPSOFT-XF", "dersoft", 0), ("SOFT0", "soft0", 0))
STATE = {}
_orig_picks_for = MS.picks_for


def zscore(sv3, elig):
    s = np.where(elig, sv3.astype(np.float64), np.nan)
    mx = np.nanmax(s, 1, keepdims=True)
    sd = np.nanstd(s, 1, keepdims=True)
    return (sv3 - mx) / np.maximum(np.nan_to_num(sd, nan=0.0), 1e-6)


def soft_pick(z, elig, ps, bd, lam_s, lam_e):
    u = np.where(elig, z - lam_s * ps - lam_e * bd, -np.inf)
    return u.argmax(1)


def s6c_mask(tr, n):
    import l2_common as C
    E = C.load_eval_pf(0)
    j = np.array([tr["idx"].get(k, -1) for k in zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist())])
    m = np.zeros(n, bool)
    ok = (j >= 0) & np.asarray(E["compliant"], bool)
    m[j[ok]] = True
    return m


def picks_for_soft(tr, elig, F, lc, n):
    saved = MS.ARMS
    MS.ARMS = tuple(a for a in saved if a[1] not in ("soft", "dersoft", "soft0"))
    out, fbs, kept = _orig_picks_for(tr, elig, F, lc, n)
    MS.ARMS = saved
    keys = list(zip(tr["e_i"].tolist(), tr["ws"].tolist()))[:n]
    hp, hvalid = MS.D.human_plans(keys)
    has_gt = np.asarray(tr["gt_label"], bool)[:n]
    fit_ok = s6c_mask(tr, n) & has_gt & hvalid
    z = zscore(tr["sel_score_v3"][:n], elig)
    ar = np.arange(n)
    Hc = np.linalg.norm(tr["fan"][:n].astype(np.float64) - hp[:, None].astype(np.float64), axis=-1).mean(-1)      # [n, 149]
    Vc = (F["gt_off"] | F["gt_edge"]).astype(np.float64)
    eps = np.asarray(tr["e_i"][:n])
    ue = np.unique(eps)
    perm = np.random.default_rng(0).permutation(len(ue))
    fold_of_ep = {int(ue[i]): int(r % 5) for r, i in enumerate(perm)}
    fold = np.array([fold_of_ep[int(e)] for e in eps])
    grid = [(a, b) for a in GRID for b in GRID]
    fits = {}
    for nm, ps, bd in (("MAPSOFT-XF", F["pswmax"].astype(np.float64), F["ownbnd_edge"].astype(np.float64)),
                       ("DER-MAPSOFT-XF", F["dpswmax"].astype(np.float64), F["derbnd_edge"].astype(np.float64))):
        P = {g: soft_pick(z, elig, ps, bd, *g) for g in grid}
        Vg = {g: Vc[ar, p] for g, p in P.items()}
        Hg = {g: Hc[ar, p] for g, p in P.items()}
        pick = np.zeros(n, np.int64)
        per_fold = {}
        for k in range(5):
            tm = fit_ok & (fold != k)
            h0 = Hg[(0.0, 0.0)][tm].mean()
            feas = [g for g in grid if Hg[g][tm].mean() <= h0]
            best = min(feas, key=lambda g: (Vg[g][tm].mean(), g[0] + g[1], g[0])) if feas else (0.0, 0.0)
            per_fold[k] = {"lam_s": best[0], "lam_e": best[1], "fit_V": float(Vg[best][tm].mean()), "fit_V_base": float(Vg[(0.0, 0.0)][tm].mean()),
                           "fit_H": float(Hg[best][tm].mean()), "fit_H_base": float(h0), "n_fit": int(tm.sum()), "n_feasible": len(feas)}
            pick[fold == k] = P[best][fold == k]
        fits[nm] = per_fold
        out[nm] = pick
        fbs[nm] = np.zeros((n, 3), np.int8)
        kept[nm] = elig.sum(1).astype(np.int32)
    out["SOFT0"] = soft_pick(z, elig, F["pswmax"], F["ownbnd_edge"].astype(np.float64), 0.0, 0.0)
    fbs["SOFT0"], kept["SOFT0"] = np.zeros((n, 3), np.int8), elig.sum(1).astype(np.int32)
    STATE.update(fits=fits, sk1=bool(np.array_equal(out["SOFT0"], out["BASE"])),
                 sk2=bool(all(sum(1 for k in range(5) if fold_of_ep[int(e)] == k) == 1 for e in ue) and len(fold_of_ep) == len(ue)),
                 n_fit_windows=int(fit_ok.sum()), n_episodes=int(len(ue)), hp=hp, hvalid=hvalid, picks=out)
    return out, fbs, kept


def human_rel_len(res_dir, tr_path):
    """descriptive: paired |path_len - human_len| (arm minus BASE) on S6c windows with a valid human path."""
    import l2_common as C
    import l2_oracle_s2_rows as S2R
    st = C.stack()
    RS, PR = st["RS"], st["PR"]
    tr = MS.load_trace(tr_path)
    n = tr["n"]
    P = S2R.load_panel(0)
    jS2 = np.array([tr["idx"].get(k, -1) for k in zip(P["clip"].tolist(), P["ws"].tolist())])
    pres = jS2 >= 0
    prior_t = np.zeros((n, 8, 2), np.float32)
    prior_t[jS2[pres]] = np.asarray(P["prior8"])[pres]
    table = C.deploy_table()
    hp, hv = STATE["hp"], STATE["hvalid"]
    s6 = s6c_mask(tr, n) & np.asarray(tr["gt_label"], bool) & hv
    E = C.load_eval_pf(0)

    def arclen(X):
        Z = np.concatenate([np.zeros_like(X[:, :1]), X], axis=1)
        return np.linalg.norm(np.diff(Z.astype(np.float64), axis=1), axis=-1).sum(1)
    lh = arclen(hp)
    ar = np.arange(n)
    dev = {}
    for nm in ("BASE", "MAPSOFT-XF", "GT-T1"):
        full = RS.retime_plan(tr["fan"][ar, STATE["picks"][nm]].astype(np.float32), prior_t, C.ALL_T, table, cast=True)
        dev[nm] = np.where(s6, np.abs(arclen(full) - lh), np.nan)
    eid = np.asarray(tr["e_i"])
    return {nm: {"mean_abs_dev_m": float(np.nanmean(dev[nm])), "vs_BASE": MS.cell_json(PR.paired_cell(dev[nm], dev["BASE"], list(eid)))} for nm in ("MAPSOFT-XF", "GT-T1")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default="F:/refcv9_mapsel/trace_s0.npz")
    ap.add_argument("--rast", default="F:/refcv9_mapsel/rast_s0")
    ap.add_argument("--allow-partial", action="store_true", help="DRY-RUN ONLY (smoke bank); a scored reading never uses it")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = hashlib.sha256((HERE / "PREREG_MAPSOFT.md").read_bytes()).hexdigest()
    if pre != (HERE / "PREREG_MAPSOFT.sha256").read_text().split()[0]:
        raise SystemExit("[mapsoft] PREREG_MAPSOFT.md does not match its seal: refusing")
    MS.pass1 = Q2.pass1_bnd
    MS.picks_for = picks_for_soft
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
           "S-K1_soft00_equals_BASE": {"pass": STATE["sk1"]}, "S-K2_folds_partition_episodes": {"pass": STATE["sk2"]}}
    cl = MS.arm_clauses(res, "MAPSOFT-XF")
    c1, c2e = mc["MAPSOFT-XF"]["off_drivable"]["vs_BASE"], mc["MAPSOFT-XF"]["edge_contact"]["vs_BASE"]
    d1, d2 = mc["DER-MAPSOFT-XF"]["off_drivable"]["vs_BASE"], mc["DER-MAPSOFT-XF"]["edge_contact"]["vs_BASE"]
    below = lambda c: bool("hi" in c and c["hi"] < 0)  # noqa: E731
    bars = {"M1": below(c1), "M2": below(c2e), "M3": bool(cl["C2"]), "M4": bool(not below(d1) and not below(d2)), "M5": bool(cl["C4"])}
    zero_folds = sum(1 for f in STATE["fits"]["MAPSOFT-XF"].values() if f["lam_s"] == 0 and f["lam_e"] == 0)
    ctl_fail = sorted(k for k, v in ctl.items() if not v["pass"])
    if ctl_fail:
        word, nxt = f"VOID (control failed: {ctl_fail})", "fix the wrapper; re-run (0 GPU)."
    elif all(bars.values()):
        word, nxt = "PASS", "register MAPSOFT-XF as an opt-in inference row (T1 stamp); refcv9 trains a map-feature selector/critic with these labels (the soft cost initialises it); NavSim only with a PI go."
    elif not (bars["M1"] and bars["M2"]):
        word = "FAIL (" + ", ".join(k for k, v in bars.items() if not v) + ")"
        nxt = ("own-map precision binds (the fit returned (0, 0) in >= 3 folds): perception training of the map head (sidewalk / boundary precision) -- compute, a PI decision."
               if zero_folds >= 3 else "the 2-parameter cost is too weak: a learned listwise selector over richer own-map features.")
    else:
        word = "FAIL (" + ", ".join(k for k, v in bars.items() if not v) + ")"
        nxt = "the trade-off costs driving at the fitted point: constrain ALL four families in the fit (multi-objective) before anything else."
    try:
        hrel = human_rel_len(msout, a.trace)
    except Exception as exc:  # noqa: BLE001 -- descriptive line only; recorded
        hrel = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    S6 = lambda nm: {k: mc[nm][k].get("rate") for k in ("off_drivable", "edge_contact", "lane_cross")}  # noqa: E731
    gt1 = mc["GT-T1"]["off_drivable"].get("rate")
    base = mc["BASE"]["off_drivable"].get("rate")
    share = (base - mc["MAPSOFT-XF"]["off_drivable"]["rate"]) / (base - gt1) if (base is not None and gt1 is not None and base != gt1) else None
    r = {"tool": "mapsoft.py", "prereg_sha256": pre, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "controls": ctl, "bars": bars,
         "cells": {"M1_sidewalk": c1, "M2_edge": c2e, "DER_sidewalk": d1, "DER_edge": d2}, "M3_worse_rows": cl["C2_worse_rows"],
         "M5_guard": {k: res["progress_guard"]["S6c"]["MAPSOFT-XF"][k] for k in ("path_len_m", "stop_share")},
         "fits": STATE["fits"], "n_fit_windows": STATE["n_fit_windows"], "zero_lambda_folds": zero_folds,
         "rates_S6c": {nm: S6(nm) for nm in ("BASE", "HUMAN", "GT-T1", "OWN-T1", "MAPSOFT-XF", "DER-MAPSOFT-XF")},
         "share_of_GT_T1_sidewalk_gain": share, "human_relative_path_length_DESCRIPTIVE": hrel, "verdict": word, "next": nxt}
    MS.jdump(r, out / "MAPSOFT_RESULT.json")
    f = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and "delta" in c else "-"  # noqa: E731
    g = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"  # noqa: E731
    Lm = [word, "", f"# MAPSOFT (mechanical; PREREG_MAPSOFT sha256 {pre[:12]})", "",
          "tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); estimator: paired episode-cluster bootstrap B 2000 seed 0 over EPISODES only; one training seed; one inference seed", "",
          "| arm (S6c) | sidewalk | edge contact | lane cross |", "|---|---|---|---|"] + \
         [f"| {nm} | {g(v['off_drivable'])} | {g(v['edge_contact'])} | {g(v['lane_cross'])} |" for nm, v in r["rates_S6c"].items()] + \
         ["", f"M1 sidewalk vs BASE {f(c1)} -> {bars['M1']}", f"M2 edge vs BASE {f(c2e)} -> {bars['M2']}", f"M3 no four-family row worse -> {bars['M3']} {cl['C2_worse_rows'] or ''}",
          f"M4 deranged not below BASE (sidewalk {f(d1)}, edge {f(d2)}) -> {bars['M4']}", f"M5 STOP guard -> {bars['M5']} {json.dumps(r['M5_guard'])[:300]}", "",
          "fitted lambda per fold (own): " + json.dumps({k: (v['lam_s'], v['lam_e']) for k, v in STATE['fits']['MAPSOFT-XF'].items()}),
          "fitted lambda per fold (deranged): " + json.dumps({k: (v['lam_s'], v['lam_e']) for k, v in STATE['fits']['DER-MAPSOFT-XF'].items()}),
          f"share of GT-T1's sidewalk gain: {share}", f"human-relative path length (DESCRIPTIVE): {json.dumps(hrel)[:400]}", "",
          f"controls failed: {ctl_fail or 'none'}", "", f"NEXT: {nxt}", "", "ZZMSOFTENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(Lm), encoding="utf-8")
    print(word, bars, ctl_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
