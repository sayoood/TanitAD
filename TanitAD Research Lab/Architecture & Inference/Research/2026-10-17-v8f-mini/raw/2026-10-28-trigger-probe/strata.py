#!/usr/bin/env python3
"""strata.py -- PREREG_FROZEN.md 6.  TRAIN strata within each population: GO-rate on the FULL population (episode-cluster bootstrap), probe AUC on the banked out-of-fold scores (linear).
Strata are CLIP-level VLM-derived (TL_*) or lead-array derived (LEAD_*); none enters the decision rule.  Writes raw/results_strata.json.   CUDA_VISIBLE_DEVICES=-1  python -I strata.py"""
from __future__ import annotations
import json, os, sys, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402
import run_probe as RP  # noqa: E402

STR = ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart")


def rate_ci(go, clip, n_boot=2000, seed=0):
    u, inv = np.unique(clip, return_inverse=True)
    s = np.bincount(inv, weights=go, minlength=len(u)); n = np.bincount(inv, minlength=len(u)).astype(float)
    cnt = np.stack([np.bincount(np.random.default_rng(seed + 0).integers(0, len(u), len(u)), minlength=len(u)) for _ in range(1)])  # placeholder to keep rng simple
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        b.append((c @ s) / (c @ n))
    return {"rate": float(s.sum() / n.sum()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}


def main():
    A = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    out = {"note": "TRAIN only; TL_* are CLIP-level and VLM-derived (vlm-cot, disputed, untimed); LEAD_DEPART reads the future (stratum, never a feature)", "pops": {}}
    for pop in ("pstop", "pmove"):
        d = RP.load_pop(pop)
        v0 = A["v0"]
        full = (v0 < 0.5) if pop == "pstop" else ((v0 >= 0.5) & (v0 < 6.0))
        y, clip, outer = d["y"], d["clip"], d["outer"]
        ov, oz = np.load(f"{RAW}/oof_{pop}_V_linear.npz"), np.load(f"{RAW}/oof_{pop}_Z_linear.npz")
        sc = {"E": ov["E"], "V+E": ov["V+E"], "Z+E": oz["Z+E"]}
        P = {"all": {"n_full": int(full.sum()), "go_rate_full": rate_ci(A["go"][full].astype(float), A["clip_i"][full])}, "strata": {}}
        for s in STR:
            mf = full & A[s]
            ms = d[s].astype(bool)
            r = {"n_full": int(mf.sum()), "n_clips_full": int(len(np.unique(A["clip_i"][mf]))), "n_sample": int(ms.sum()), "n_sample_clips": int(len(np.unique(clip[ms]))) if ms.any() else 0,
                 "n_sample_go": int(y[ms].sum()) if ms.any() else 0}
            if mf.sum() >= 30:
                r["go_rate_full"] = rate_ci(A["go"][mf].astype(float), A["clip_i"][mf])
            ok = ms.sum() >= 60 and y[ms].sum() >= 30 and (ms.sum() - y[ms].sum()) >= 30 and r["n_sample_clips"] >= 20
            if not ok:
                r["auc"] = "too small (<30 windows in a class or <20 episodes in the sample)"
            else:
                cnt = L.boot_counts(clip[ms])
                r["auc"] = {k: {**L.auc_ci(v[ms], y[ms], clip[ms], counts=cnt), "foldmean": L.auc_ci_fold(v[ms], y[ms], clip[ms], outer[ms], counts=cnt)} for k, v in sc.items()}
                r["paired"] = {"V+E_minus_E": {**L.dauc_ci(sc["V+E"][ms], sc["E"][ms], y[ms], clip[ms], counts=cnt), "foldmean": L.dauc_ci_fold(sc["V+E"][ms], sc["E"][ms], y[ms], clip[ms], outer[ms], counts=cnt)},
                               "Z+E_minus_V+E": {**L.dauc_ci(sc["Z+E"][ms], sc["V+E"][ms], y[ms], clip[ms], counts=cnt), "foldmean": L.dauc_ci_fold(sc["Z+E"][ms], sc["V+E"][ms], y[ms], clip[ms], outer[ms], counts=cnt)}}
            P["strata"][s] = r
            a = r["auc"]
            print(pop, s, "n_full", r["n_full"], "go", round(r.get("go_rate_full", {}).get("rate", float("nan")), 3) if "go_rate_full" in r else None,
                  "|", (f"E {a['E']['foldmean']['auc']:.3f} V+E {a['V+E']['foldmean']['auc']:.3f} Z+E {a['Z+E']['foldmean']['auc']:.3f} dV {r['paired']['V+E_minus_E']['foldmean']['delta']:.3f}" if isinstance(a, dict) else a[:20]))
        out["pops"][pop] = P
    json.dump(out, open(f"{RAW}/results_strata.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
