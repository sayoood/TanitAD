#!/usr/bin/env python3
"""G-REWARD-ID from SIDE FILES ALONE: join the `rl_terms` records (gmax) to the `pdm_targets` records (dac) on the shared key and count the 2x2.

    identity:   gmax > 0   <=>   dac == 0          (per proposal; key = [log, token, step, rank] + ckpt_step)

Pure numpy / json -- no navsim -- so the pod chain can run it as the post-label gate on the real on-policy proposals:
    python eval/rl_terms_join_identity.py --rlt <dir-or-file...> --pdm <dir-or-file...> [--out eval/raw/rl_terms/identity_onpolicy.json]
Exit code 0 always (a disagreement is a FINDING, reported with counts, never forced); a non-empty `unmatched` is reported too.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def files_of(paths, prefix):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, prefix + "*.jsonl"))) if os.path.isdir(p) else [p]
    return out


def load(files, kind):
    recs = {}
    for fp in files:
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == kind:
                recs[(tuple(r["key"]), int(r["ckpt_step"]))] = r
    return recs


def join(rlt: dict, pdm: dict) -> dict:
    keys = sorted(set(rlt) & set(pdm))
    p_off, t_off, d_all, scene_all0, scene_all1 = [], [], [], 0, 0
    dis = []
    for k in keys:
        g, dac, d = np.asarray(rlt[k]["gmax"], float), np.asarray(pdm[k]["dac"], float), np.asarray(rlt[k]["d"], float)
        if not (len(g) == len(dac) == len(d)):
            raise ValueError(f"{k}: proposal counts differ ({len(g)}, {len(dac)}, {len(d)})")
        p_off.append(g > 0)
        t_off.append(dac == 0)
        d_all.append(d)
        scene_all0 += int((dac == 0).all())
        scene_all1 += int((dac == 1).all())
        for i in np.flatnonzero((g > 0) != (dac == 0)):
            dis.append({"key": list(k[0]), "ckpt_step": k[1], "proposal": int(i), "gmax": float(g[i]), "dac": float(dac[i])})
    p, t, d = np.concatenate(p_off), np.concatenate(t_off), np.concatenate(d_all)
    return {"scenes_joined": len(keys), "scenes_rlt_only": len(set(rlt) - set(pdm)), "scenes_pdm_only": len(set(pdm) - set(rlt)),
            "proposals": int(p.size), "confusion": {"pred_off_dac0": int((p & t).sum()), "pred_off_dac1": int((p & ~t).sum()),
                                                   "pred_ok_dac0": int((~p & t).sum()), "pred_ok_dac1": int((~p & ~t).sum())},
            "D_exactly_zero_on_dac1": bool((d[~t] == 0).all()) if (~t).any() else None,
            "D_positive_on_dac0_share": float((d[t] > 0).mean()) if t.any() else None,
            "scenes_all_dac0": scene_all0, "scenes_all_dac1": scene_all1,
            "disagreements_first_25": dis[:25], "disagreements_n": len(dis)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlt", nargs="+", required=True)
    ap.add_argument("--pdm", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = join(load(files_of(a.rlt, "rl_terms_"), "rl_terms"), load(files_of(a.pdm, "pdm_"), "pdm_targets"))
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements_first_25"}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("ZZRLTJOIN", json.dumps({"proposals": res["proposals"], "confusion": res["confusion"], "disagreements": res["disagreements_n"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
