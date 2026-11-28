"""l1c_next.py -- the NEXT-LEVER arms of PREREG_L1C (Rule Zero; PREREG_L1C_AMENDMENT2 A2.8).  Run after the verdict; EXPLORATORY unless it is the lever A2.8 names for the outcome.

  N1 H2n   the L1 H2n recipe (one-hidden-layer MLP on scalars + standardised z_tac, L1 loss, wd and epochs by 5-fold episode CV) fitted on the FRESH rows only
           (A2.8 names it as the O2 lever).  Scored beside H2cs / H2c / H0 on eval139 S2 at both seeds.
  N2 POOL  cross-fit on eval139 episodes where each fold's head is fitted on the FRESH rows PLUS the other four eval folds (the Amendment-3 CF2 recipe with the
           fresh rows added to every fold's fit set).  Reads whether the clean split ADDS to the in-distribution cross-fit (domain gap vs sample size).  POST-HOC:
           it carries no bar and needs its own pre-registration before it is quoted as a result.
  N3 CURVE the learning-curve points already in L1C_RESULT.json (sensitivity block), re-tabulated.

    CUDA_VISIBLE_DEVICES=-1 python -X utf8 l1c_next.py --out <result dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l1c_fit_score as L  # noqa: E402

F, FU, AL, PH, PS, RT = L.F, L.FU, L.AL, L.PH, L.PS, L.RT


def pooled_cross_fit(E, Z, H, ks, Xf, Zf, Hf, eidf):
    f = F.folds_of(E["eid"], 0)
    P = np.zeros_like(H)
    info = []
    for j in range(F.NFOLD):
        tr, te = f != j, f == j
        X = np.concatenate([Xf, E["X"][tr]])
        Zc = np.concatenate([Zf, Z[tr]])
        Hc = np.concatenate([Hf, H[tr]])
        eid = np.concatenate([np.array(["fresh_" + str(e) for e in eidf]), E["eid"][tr]])        # fresh rows keep their own episode ids for the inner CV
        (k, lam), _cv, _ = F.cv_head(X, Zc, Hc, eid, 0, ks)
        m = F.head_fit(X, Zc, Hc, k, lam)
        P[te] = F.head_pred(m, E["X"][te], Z[te])
        info.append({"fold": j, "k": int(k), "lambda": float(lam), "n_fit": int(len(Xf) + tr.sum()), "n_scored": int(te.sum())})
    return P, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-fresh", default="F:/l1c/bank")
    ap.add_argument("--tree-root", default="F:/l1c/tree")
    ap.add_argument("--tree", default="step48852ema_fresh")
    a = ap.parse_args()
    t0 = time.time()
    import compliant_reread_r8 as CR
    import retime_trainfit_r8 as TF
    tools = RT.tools_load()
    ev, first = {}, {}
    for s in L.SEEDS:
        ev[s] = F.load_eval(tools, CR, s, first, "S2")
    TF.TRAIN_ROOT, TF.TRAIN_TREE = a.tree_root, a.tree
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    tr = L.load_fresh(tools, CR2, TF, "S2", {})
    banks = {"eval0": F.load_bank(L.L1_BANK / "ztac_eval_s0.npz"), "eval1": F.load_bank(L.L1_BANK / "ztac_eval_s1.npz"),
             "fresh": F.load_bank(Path(a.bank_fresh) / "ztac_fresh.npz")}
    gains = json.loads(F.FROZEN_PF.read_text(encoding="utf-8"))["surfaces"]["S2"]["all"]["RETIME"]
    Ztr, _ = F.join(banks["fresh"], tr["clip"], tr["ws"])
    Htr = AL.arc(tr["G"]) - AL.arc(tr["HP"])
    res = {"tool": "l1c_next.py", "status": "EXPLORATORY / next-lever (A2.8); no bar", "script_sha256": F.sha256_file(__file__), "n_fresh_rows": int(len(Htr))}
    # N1: MLP on the fresh rows
    t1 = time.time()
    pn, info = F.h2n_fit(tr["X"], Ztr, Htr, tr["eid"], 0)
    res["N1_H2n_fit"] = {**info, "wall_s": round(time.time() - t1, 1)}
    h = L.fit_heads(tr, Ztr)
    cf = None
    res["seeds"] = {}
    for s in L.SEEDS:
        E = ev[s]
        Z, _ = F.join(banks[f"eval{s}"], E["clip"], E["ws"])
        sp = AL.arc(E["HP"])
        Hev = AL.arc(E["G"]) - sp
        if cf is None:
            cf = pooled_cross_fit(E, Z, Hev, F.KS, tr["X"], Ztr, Htr, tr["eid"])
        arms = {"H0": RT.retime_apply(E["O"], E["HP"], np.asarray(gains, np.float64)),
                "H2cs": PH.at_progress(E["O"], sp + PS.ridge_pred(h["m_s"], E["X"])),
                "H2c": PH.at_progress(E["O"], sp + F.head_pred(h["m2"], E["X"], Z)),
                "H2cn": PH.at_progress(E["O"], sp + pn(E["X"], Z)),
                "POOL": PH.at_progress(E["O"], sp + cf[0])}
        C = {k: F.comp(tools, p, E["G"], E["dt"])["ade_m"] for k, p in arms.items()}
        pairs = {"H2cn-H2cs": ("H2cn", "H2cs"), "H2cn-H2c": ("H2cn", "H2c"), "H2cn-H0": ("H2cn", "H0"), "POOL-H0": ("POOL", "H0"), "POOL-H2c": ("POOL", "H2c"),
                 "POOL-H2cs": ("POOL", "H2cs")}
        res["seeds"][str(s)] = {"paired": {p: F.cell(tools, C[x], C[y], E["eid"]) for p, (x, y) in pairs.items()},
                                "eval_R2": {"H2cn": F.r2(pn(E["X"], Z), Hev), "POOL": F.r2(cf[0], Hev)}}
        print(f"[l1c-next] seed {s} done", flush=True)
    res["N2_pool_folds"] = cf[1]
    r = json.loads((Path(a.out) / "L1C_RESULT.json").read_text(encoding="utf-8"))
    res["N3_curve"] = {k: {"n_rows": v.get("n_rows"), "n_clips": v.get("n_clips"), "H2c-H2cs": [v["seeds"][s]["H2c-H2cs"]["delta"] for s in ("0", "1")],
                           "H2c-H0": [v["seeds"][s]["H2c-H0"]["delta"] for s in ("0", "1")]}
                       for k, v in r["surfaces"]["S2"].get("sensitivity", {}).items() if "seeds" in v}
    res["wall_s"] = round(time.time() - t0, 1)
    (Path(a.out) / "L1C_NEXT.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"[l1c-next] WROTE L1C_NEXT.json wall {res['wall_s']}s ZZL1CNEXTDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
