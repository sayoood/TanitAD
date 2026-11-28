#!/usr/bin/env python3
"""tables_g.py -- every number of RESULTS.md is printed here (raw/tables3.txt) or sits in raw/VERDICT.json.  Applies the committed rule of PREREG 4 and the controls of PREREG 6.
Reads raw/oofg_*, raw/resg_*, probe 1's banked OOF, raw/results_eval141_g_*, raw/results_strata_g.json, raw/weightmap_*.   CUDA_VISIBLE_DEVICES=-1  python -I tables_g.py"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import g_lib as G  # noqa: E402
import numpy as np

L = G.L
FAMS = ("linear", "mlp")
LINES = []


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def f3(r, key="fold"):
    v = r[key]
    return f"{v['auc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


def fd(r, key="fold"):
    v = r[key]
    return f"{v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"


def covers(v, x):
    return v["lo"] <= x <= v["hi"]


def load_oof(pop, fam, tag=None):
    p = f"{G.RAW}/oofg_{tag or pop}_{fam}.npz"
    return dict(np.load(p)) if os.path.exists(p) else {}


def controls(pop, d, counts):
    """PREREG 6 for the linear arms of one population -> dict(per-control pass flags + numbers)."""
    y, clip, outer = d["y"], d["clip"], d["outer"]
    oof = load_oof(pop, "linear")
    out = {"ok": True, "pending": [], "detail": {}}
    ys = L.shuffle_labels(y, 1)
    for src in ("G4", "G8"):
        k1, k2, k3 = f"C1:shuffle:{src}+E", f"C2:donor:{src}+E", f"C3:leak:{src}+E"
        det = {}
        if k1.replace(":", "__") in oof:
            r = G.rec(oof[k1.replace(":", "__")], ys, clip, outer, counts)
            det["C1"] = {"fold": r["fold"], "pass": covers(r["fold"], 0.5)}
        else:
            out["pending"].append(k1)
        if k3.replace(":", "__") in oof:
            r = G.rec(oof[k3.replace(":", "__")], y, clip, outer, counts)
            det["C3"] = {"fold_auc": r["fold"]["auc"], "pooled_auc": r["pooled"]["auc"], "pass": bool(r["fold"]["auc"] >= 0.99 and r["pooled"]["auc"] >= 0.99)}
        else:
            out["pending"].append(k3)
        if k2.replace(":", "__") in oof:
            yd = L.donor_labels(y, clip, 2)
            r = G.rec(oof[k2.replace(":", "__")], yd, clip, outer, counts)
            c2 = {"seed2_fold": r["fold"], "seed2_pass": covers(r["fold"], 0.5)}
            if c2["seed2_pass"]:
                c2["pass"], c2["how"] = True, "seed 2 CI covers 0.5"
            else:
                aucs = [r["fold"]["auc"]]
                for sd in (3, 4, 5, 6):
                    kk = f"C2:donor{sd}:{src}+E".replace(":", "__")
                    if kk in oof:
                        rr = G.rec(oof[kk], L.donor_labels(y, clip, sd), clip, outer, counts)
                        aucs.append(rr["fold"]["auc"])
                c2["band_aucs"] = aucs
                if len(aucs) == 5:
                    m = float(np.mean(aucs))
                    c2["band_mean"] = m
                    c2["pass"], c2["how"] = bool(0.47 <= m <= 0.53), "seed 2 missed 0.5; 5-draw band mean %.4f (rule: in [0.47, 0.53])" % m
                else:
                    c2["pass"], c2["how"] = None, "seed 2 missed 0.5; band seeds 3-6 PENDING (%d of 5 draws)" % len(aucs)
                    out["pending"].append(f"C2 band {src}")
            det["C2"] = c2
        else:
            out["pending"].append(k2)
        out["detail"][src] = det
    k8, k4 = "C1:shuffle:G8+E".replace(":", "__"), "C1:shuffle:G4+E".replace(":", "__")
    if k8 in oof and k4 in oof:
        dd = G.paired(oof[k8], oof[k4], ys, clip, outer, counts)["fold"]
        out["detail"]["C1_paired_null"] = {"delta": dd["delta"], "lo": dd["lo"], "hi": dd["hi"], "pass": bool(dd["lo"] <= 0 <= dd["hi"])}
    # probe-1 banked controls for E
    bk = G.banked(pop, "linear")
    e1 = G.rec(bk["C1:shuffle:E"], ys, clip, outer, counts)["fold"]
    e3 = G.rec(bk["C3:leak:E"], y, clip, outer, counts)
    e2 = G.rec(bk["C2:donor:E"], L.donor_labels(y, clip, 2), clip, outer, counts)["fold"]
    out["detail"]["E_banked_probe1"] = {"C1_fold": e1, "C1_pass": covers(e1, 0.5), "C2_fold": e2, "C2_covers_0.5": covers(e2, 0.5), "C3_fold_auc": e3["fold"]["auc"], "C3_pass": bool(e3["fold"]["auc"] >= 0.99)}
    flags = []
    for src in ("G4", "G8"):
        for c in ("C1", "C2", "C3"):
            if c in out["detail"][src]:
                flags.append(out["detail"][src][c].get("pass"))
    if "C1_paired_null" in out["detail"]:
        flags.append(out["detail"]["C1_paired_null"]["pass"])
    out["ok"] = all(f is not False for f in flags)
    out["all_run"] = len(out["pending"]) == 0 and all(f is not None for f in flags)
    return out


def main():
    verdict = {"cells": {}, "controls": {}}
    pr("# tables3 -- every number of RESULTS.md (MEASURED; fold-mean AUC = the mean of per-fold AUCs, primary; pooled beside; episode-cluster bootstrap B=2000 seed 0; paired on the same windows)")
    cells = {}
    for pop in G.POPS:
        d = G.load_pop(pop)
        y, clip, outer = d["y"], d["clip"], d["outer"]
        counts = L.boot_counts(clip)
        pr(f"\n## {pop.upper()}  n={len(y)} go={int(y.sum())} clips={len(np.unique(clip))}")
        for fam in FAMS:
            oof = load_oof(pop, fam)
            bk = G.banked(pop, fam)
            S = {"E": bk["E"], "V+E": bk["V+E"]}
            for k in ("G4", "G8", "G4+E", "G8+E"):
                if k in oof:
                    S[k] = oof[k]
            pr(f"\n### {pop} {fam}: arms (fold-mean AUC [95% CI] | pooled AUC [CI])")
            for k in ("E", "V+E", "G4", "G8", "G4+E", "G8+E"):
                if k in S:
                    r = G.rec(S[k], y, clip, outer, counts)
                    pr(f"  {k:6s} {f3(r)} | {f3(r, 'pooled')}")
            pr(f"### {pop} {fam}: paired dAUC (fold-mean | pooled)")
            for a_, b_ in (("G8+E", "G4+E"), ("G4+E", "E"), ("G8+E", "E"), ("G4+E", "V+E"), ("G8+E", "V+E"), ("V+E", "E"), ("G8", "G4")):
                if a_ in S and b_ in S:
                    r = G.paired(S[a_], S[b_], y, clip, outer, counts)
                    pr(f"  {a_:5s} - {b_:5s} {fd(r)} | {fd(r, 'pooled')}")
                    if (a_, b_) == ("G8+E", "G4+E"):
                        cells[(pop, fam)] = {"delta": r["fold"]["delta"], "lo": r["fold"]["lo"], "hi": r["fold"]["hi"], "pooled_delta": r["pooled"]["delta"], "pooled_lo": r["pooled"]["lo"], "pooled_hi": r["pooled"]["hi"],
                                             "auc_G8E": r["fold"]["auc_a"], "auc_G4E": r["fold"]["auc_b"]}
        # controls
        c = controls(pop, d, counts)
        verdict["controls"][pop] = c
        pr(f"\n### {pop} linear controls: ok={c['ok']} all_run={c['all_run']} pending={c['pending']}")
        for src in ("G4", "G8"):
            for cn, v in c["detail"][src].items():
                short = {k: v[k] for k in v if k not in ("fold", "seed2_fold")}
                if "fold" in v:
                    short["fold_auc"] = f"{v['fold']['auc']:.4f} [{v['fold']['lo']:.4f}, {v['fold']['hi']:.4f}]"
                if "seed2_fold" in v:
                    short["seed2_fold_auc"] = f"{v['seed2_fold']['auc']:.4f} [{v['seed2_fold']['lo']:.4f}, {v['seed2_fold']['hi']:.4f}]"
                pr(f"  {src}+E {cn}: {short}")
        if "C1_paired_null" in c["detail"]:
            pr("  C1 paired null G8+E - G4+E under shuffled labels:", c["detail"]["C1_paired_null"])
        pr("  E (probe-1 banked):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c["detail"]["E_banked_probe1"].items() if "fold" not in k or k.endswith("auc")})
        # MLP C1
        om = load_oof(pop, "mlp")
        ys = L.shuffle_labels(y, 1)
        mc = {}
        for src in ("G4", "G8"):
            kk = f"C1:shuffle:{src}+E".replace(":", "__")
            if kk in om:
                r = G.rec(om[kk], ys, clip, outer, counts)["fold"]
                mc[src] = {"fold_auc": r["auc"], "lo": r["lo"], "hi": r["hi"], "pass": covers(r, 0.5)}
        verdict["controls"][pop]["mlp_C1"] = mc
        pr(f"  MLP C1 shuffle: {mc if mc else 'NOT RUN: UNVERIFIED'}")
    # ----- decision
    pr("\n## DECISION (PREREG 4): dAUC(G8+E - G4+E), fold-mean, per cell")
    helps, nothelp_all, missing = False, True, []
    for pop in G.POPS:
        for fam in FAMS:
            cell = cells.get((pop, fam))
            if cell is None:
                missing.append(f"{pop}/{fam}")
                nothelp_all = False
                continue
            c = verdict["controls"][pop]
            void = (not c["ok"]) or (fam == "mlp" and any(v["pass"] is False for v in c.get("mlp_C1", {}).values()))
            unver = (not c["all_run"]) or (fam == "mlp" and not c.get("mlp_C1"))
            cell.update({"void": bool(void), "controls_unverified": bool(unver)})
            cond_help = (cell["delta"] >= 0.03) and (cell["lo"] > 0)
            cell["meets_HELPS"] = bool(cond_help)
            cell["lt_0_01"] = bool(cell["delta"] < 0.01)
            if cond_help and not void:
                helps = True
            if void or not cell["lt_0_01"]:
                nothelp_all = False
            verdict["cells"][f"{pop}/{fam}"] = cell
            pr(f"  {pop:6s} {fam:6s} dAUC {cell['delta']:+.4f} [{cell['lo']:+.4f}, {cell['hi']:+.4f}]  (G8+E {cell['auc_G8E']:.3f} vs G4+E {cell['auc_G4E']:.3f}; pooled {cell['pooled_delta']:+.4f})  "
               f"HELPS-cond={cond_help}  <0.01={cell['lt_0_01']}  void={void}  controls_unverified={unver}")
    if helps:
        v = "RESOLUTION HELPS"
    elif missing:
        v = "INCOMPLETE (cells missing: %s)" % missing
    elif nothelp_all:
        v = "RESOLUTION DOES NOT HELP AT 8x20"
    else:
        v = "INCONCLUSIVE"
    verdict["verdict"] = v
    pr(f"\n## VERDICT: {v}")
    # ----- eval-141
    pr("\n## EVAL-141 no-refit (AUC [episode-bootstrap CI]; n windows / go / episodes)")
    ev = {}
    for pop in G.POPS:
        for fam in FAMS:
            p = f"{G.RAW}/results_eval141_g_{pop}_{fam}.json"
            if not os.path.exists(p):
                pr(f"  {pop} {fam}: not run")
                continue
            r = json.load(open(p))
            ev[f"{pop}/{fam}"] = r
            pr(f"  {pop} {fam}: n={r['n']} go={r['n_go']} episodes={r['n_episodes']}  " + "  ".join(f"{k} {v['auc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in r["arms"].items()))
            for k, v in r.get("paired", {}).items():
                pr(f"     {k}: {v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    verdict["eval141"] = {k: {"arms": v["arms"], "paired": v.get("paired")} for k, v in ev.items()}
    # ----- weight map
    pr("\n## SPATIAL READOUT MAP (descriptive; L2 norm of each token's 16 standardised coefficients)")
    for pop in G.POPS:
        for arm in ("G8_E", "G4_E"):
            p = f"{G.RAW}/weightmap_{pop}_{arm}.npy"
            if not os.path.exists(p):
                continue
            w = np.load(p)
            w2 = w ** 2
            tot = w2.sum()
            R, C = w.shape
            rows = w2.sum(1) / tot
            pr(f"  {pop} {arm} {R}x{C}: row share of squared weight {np.round(rows, 3).tolist()}  lower-half share {rows[R // 2:].sum():.3f}  top-1 token share {w2.max() / tot:.4f}  top-10% tokens share {np.sort(w2.ravel())[::-1][:max(1, w.size // 10)].sum() / tot:.3f}  "
               f"argmax (r,c)={np.unravel_index(int(w.argmax()), w.shape)}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
        for ax, pop in zip(axs, G.POPS):
            p = f"{G.RAW}/weightmap_{pop}_G8_E.npy"
            if os.path.exists(p):
                im = ax.imshow(np.load(p), cmap="viridis", aspect="auto")
                ax.set_title(f"{pop}: G8+E linear readout, token weight norm (8x20)", fontsize=9)
                ax.set_xlabel("column (left to right of the frame)")
                ax.set_ylabel("row (top to bottom)")
                plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(f"{G.HERE}/readout_heatmap_G8.png", dpi=130)
        pr("  wrote readout_heatmap_G8.png")
    except Exception as e:                                                                    # noqa: BLE001
        pr("  heatmap not written:", repr(e))
    # ----- strata
    sp = f"{G.RAW}/results_strata_g.json"
    if os.path.exists(sp):
        S = json.load(open(sp))
        pr("\n## STRATA (TRAIN, linear, fold-mean; dAUC vs E and G8+E - G4+E)")
        for pop in G.POPS:
            for s, r in S["pops"][pop].items():
                if isinstance(r["auc"], str):
                    pr(f"  {pop} {s}: n={r['n_sample']} clips={r['n_sample_clips']} -> {r['auc'][:14]}")
                else:
                    a = r["auc"]
                    pr(f"  {pop} {s}: n={r['n_sample']} go={r['n_sample_go']} clips={r['n_sample_clips']} full-pop GO {None if r['go_rate_full'] is None else round(r['go_rate_full']['rate'], 3)} | E {a['E']['fold']['auc']:.3f} V+E {a['V+E']['fold']['auc']:.3f} "
                       f"G4+E {a['G4+E']['fold']['auc']:.3f} G8+E {a['G8+E']['fold']['auc']:.3f} | G4+E-E {fd(r['paired']['G4+E_minus_E'])} G8+E-E {fd(r['paired']['G8+E_minus_E'])} G8+E-G4+E {fd(r['paired']['G8+E_minus_G4+E'])}")
    # ----- reachability mutation
    pr("\n## C5(b) content mutation (G8 source = NN upsample of G4)")
    for pop in G.POPS:
        om = load_oof(pop, "linear", tag=f"{pop}_mut")
        if "G8mut+E" in om:
            d = G.load_pop(pop)
            counts = L.boot_counts(d["clip"])
            real, g4 = load_oof(pop, "linear")["G8+E"], load_oof(pop, "linear")["G4+E"]
            rm = G.rec(om["G8mut+E"], d["y"], d["clip"], d["outer"], counts)
            rr = G.rec(real, d["y"], d["clip"], d["outer"], counts)
            dm = G.paired(om["G8mut+E"], g4, d["y"], d["clip"], d["outer"], counts)
            d_real = G.paired(real, om["G8mut+E"], d["y"], d["clip"], d["outer"], counts)
            changed = abs(rm["fold"]["auc"] - rr["fold"]["auc"]) > 1e-4
            fa, fb = None, None
            pr(f"  {pop}: G8mut+E {f3(rm)}  vs real G8+E {f3(rr)}; |diff|>1e-4: {changed}; G8mut+E - G4+E {fd(dm)}; G8+E - G8mut+E {fd(d_real)}")
            verdict.setdefault("C5b", {})[pop] = {"G8mut_auc": rm["fold"]["auc"], "G8_auc": rr["fold"]["auc"], "numbers_changed": bool(changed)}
    json.dump(verdict, open(f"{G.RAW}/VERDICT.json", "w"), indent=1, default=float)
    open(f"{G.RAW}/tables3.txt", "w", encoding="utf-8").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
