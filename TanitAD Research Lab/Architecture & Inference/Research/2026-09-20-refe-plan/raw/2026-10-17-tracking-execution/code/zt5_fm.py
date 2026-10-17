#!/usr/bin/env python3
"""zt5_fm.py -- PREREG_ZT5 Amendment 3: F_m fan-consensus margin selection, evaluated on DEV as a LOOKUP of the banked
PDM labels (executed plan unchanged, so no re-simulation).  numpy only (driverl-eval venv).

    python zt5_fm.py --dev ../data/dev --out ../data/fm_dev.json [--selftest]
Uses the dev pass's set list (dev_*of2.npz: log, token, step, rank, pick) so the population is identical to section 5.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
import zt5_lib as L  # noqa: E402
import zero_tail as zt  # noqa: E402

SETS = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets"
SCORES = {f"N-s{k}": f"D:/Projects/TanitAD-artifacts/s1_read_cache/arms/N-s{k}/dump/scores_update400_B.npz" for k in range(3)}
SWITCH = 0.10


def fm_pick(score, xy8, pick, rho):
    """F_m: score-weighted mean path of the near-tie band; pick the candidate closest to it (mean knot distance)."""
    s = np.asarray(score, np.float64)
    cand = np.flatnonzero(s >= rho * s.max())
    if cand.size < 3:
        return int(pick)
    w = s[cand] / s[cand].sum()
    ref = (w[:, None, None] * xy8[cand]).sum(0)
    d = np.linalg.norm(xy8 - ref[None], axis=-1).mean(-1)
    order = np.lexsort((cand, -s[cand], d[cand]))
    j = int(cand[order[0]])
    return j if d[pick] - d[j] >= SWITCH else int(pick)


def selftest():
    xy = np.zeros((5, 8, 2))
    for j, off in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0]):
        xy[j, :, 1] = off
        xy[j, :, 0] = np.arange(1, 9)
    s = np.array([0.99, 0.98, 0.985, 0.97, 1.0])                  # pick = 4 (the +1.0 m edge)
    ok1 = fm_pick(s, xy, 4, 0.95) == 2                              # the centre candidate
    ok2 = fm_pick(s, xy, 4, 1.0) == 4                               # rho = 1 keeps the pick
    mut = int(np.argmax(np.linalg.norm(xy - xy.mean(0)[None], axis=-1).mean(-1)))   # MUTANT: farthest instead of closest
    ok3 = mut != 2
    print(f"selftest F_m centre={ok1} rho1_keeps={ok2} mutant_farthest_RED={ok3}")
    return ok1 and ok2 and ok3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not selftest():
        sys.exit("SELFTEST FAIL")
    if a.selftest:
        return
    import navsim_dac as ND
    P = [np.load(f) for f in sorted(glob.glob(f"{a.dev}/dev_*of2.npz"))]
    D = {k: np.concatenate([p[k] for p in P]) for k in ("log", "token", "step", "rank", "pick", "banked", "dacR")}
    key = {(str(l), str(t), int(s), int(r)): i for i, (l, t, s, r) in enumerate(zip(D["log"], D["token"], D["step"], D["rank"]))}
    n = len(key)
    sc = {}
    for name, p in SCORES.items():
        z = np.load(p)
        S = np.asarray(z["scores"], np.float64)
        sc[name] = {(str(lg), str(t), int(s), int(r)): S[i]
                    for i, (lg, t, s, r) in enumerate(zip(z["log"], z["token"], z["step"], z["rank"]))}
    arms = sorted(SCORES)
    rhos = list(L.RHO_GRID)
    newpick = np.full((n, 3, len(rhos)), -1, int)
    final_deg = np.zeros((n, 64))
    for fp in sorted(glob.glob(f"{SETS}/onpolicy_*.jsonl")):
        for ln in open(fp, encoding="utf-8"):
            d = json.loads(ln)
            k = (d["log_name"], d["token"], int(d["step"]), int(d["rank"]))
            if k not in key:
                continue
            i = key[k]
            props = np.concatenate([np.asarray(d["traj"], np.float64), np.asarray(d["yaw"], np.float64)[..., None]], -1)
            props[..., -1, 2] = props[..., -2, 2]                       # A7
            p8 = np.stack([ND.to_navsim(p) for p in props])
            final_deg[i] = np.degrees(L.wrap(p8[:, -1, 2]))
            for ai, arm in enumerate(arms):
                s = L.selection_score(sc[arm][k])
                pk = int(np.argmax(s))
                assert pk == int(D["pick"][i, ai])
                for ri, rho in enumerate(rhos):
                    newpick[i, ai, ri] = fm_pick(s, p8[:, :, :2], pk, rho)
    assert (newpick >= 0).all()
    logs = sorted(set(map(str, D["log"])))
    boot = zt.Boot(np.array([logs.index(str(x)) for x in D["log"]]), len(logs))
    I = np.arange(n)
    res = {"amendment": "PREREG_ZT5 Amendment 3 (F_m)", "n_sets": n, "n_logs": len(logs), "arms": {}}
    for ai, arm in enumerate(arms):
        pk = D["pick"][:, ai]
        b6 = D["banked"][I, pk]
        A = {}
        for ri, rho in enumerate(rhos):
            j = newpick[:, ai, ri]
            f6 = D["banked"][I, j]
            r = {"dPDMS_x100": boot.ratio(f6[:, 5] - b6[:, 5], np.ones(n), 100.0), "switched_pct": float((j != pk).mean() * 100),
                 "dac_zero_rescued": int(((b6[:, 1] == 0) & (f6[:, 1] == 1)).sum()),
                 "dac_zero_broken": int(((b6[:, 1] == 1) & (f6[:, 1] == 0)).sum()),
                 "zero_rescued": int(((b6[:, 5] == 0) & (f6[:, 5] > 0)).sum()), "zero_broken": int(((b6[:, 5] > 0) & (f6[:, 5] == 0)).sum()),
                 "drawn_fail_pp": boot.ratio((1 - D["dacR"][I, j]) - (1 - D["dacR"][I, pk]), np.ones(n), 100.0),
                 "manoeuvre_agreement_pct": float(np.mean([L.manoeuvre(np.radians(final_deg[i, j[i]])) == L.manoeuvre(np.radians(final_deg[i, pk[i]])) for i in I]) * 100)}
            for si, s in enumerate(("NC", "DAC", "EP", "TTC", "C")):
                r[f"d{s}_x100"] = boot.ratio(f6[:, si] - b6[:, si], np.ones(n), 100.0)
            A[str(rho)] = r
        res["arms"][arm] = A
    prim = arms[0]
    rs = max(rhos, key=lambda x: (round(res["arms"][prim][str(x)]["dPDMS_x100"]["lo"], 10), x))
    p = res["arms"][prim][str(rs)]
    conds = {"lo_gt_0_and_point_ge_0.05": p["dPDMS_x100"]["lo"] > 0 and p["dPDMS_x100"]["point"] >= 0.05,
             "replicates_point_gt_0": all(res["arms"][a][str(rs)]["dPDMS_x100"]["point"] > 0 for a in arms[1:]),
             "no_adverse_subscore": all(p[f"d{s}_x100"]["hi"] >= 0 for s in ("NC", "DAC", "EP", "TTC", "C")),
             "drawn_plan_check": p["drawn_fail_pp"]["hi"] <= 0.5}
    res["rho_star"] = rs
    res["eligibility"] = {"eligible": all(conds.values()), "conditions": conds, "dPDMS_x100": p["dPDMS_x100"]}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"rho_star": rs, "eligibility": res["eligibility"],
                      "by_rho_N-s0": {r: (v["dPDMS_x100"]["point"], v["dPDMS_x100"]["lo"], v["dPDMS_x100"]["hi"], v["switched_pct"],
                                         v["zero_rescued"], v["zero_broken"]) for r, v in res["arms"][prim].items()},
                      "replicates": {a: res["arms"][a][str(rs)]["dPDMS_x100"] for a in arms[1:]}}, indent=1))


if __name__ == "__main__":
    main()
