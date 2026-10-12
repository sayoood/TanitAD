#!/usr/bin/env python3
"""RL-S1 priority 1, the CONSUMER proof: do `rl_fan_dump.py`'s lean `onpolicy_set` records go through the REAL labellers and then the REAL loaders?

    python eval/rl_fan_labeller_proof.py --sets <dir of onpolicy_rlfan_*.jsonl> --work <scratch dir> [--n 6] [--run-labellers] [--out proof.json]

Pipeline, nothing re-implemented here:
  1. the first N sets (one per log, so N scenes build N scenarios) are copied to <work>/sets
  2. `refe/onpolicy_relabel_pdm.py` (PDM labels) and `refe/rl_terms_label.py` (D / K / L) run on them, UNMODIFIED, in the NAVSIM v1.1 venv (the same environment
     `eval/rl_terms_identity.py` re-launches itself in) -- only with --run-labellers; otherwise the side-file dirs <work>/pdm and <work>/terms are read as given
  3. the trainer's loaders read the result: `train.OnPolicyBank(pdm_labels=..., pdm_only=True)`, `scorer_finetune.load_rl_terms`, and the RL-val evaluator's
     `load_pdm / load_terms / load_paths`.  Every set must come out of every loader; the placeholder targets must never be served without a PDM label.
Evidence class: MEASURED on this machine; the full-corpus run on the A40 is the production chain (not run here).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REFE = os.path.join(PKG, "refe")
sys.path.insert(0, HERE)
sys.path.insert(0, REFE)

DRV = os.environ.get("REFE_DRIVE", "E:")
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = f"{DRV}/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
MAPS = f"{DRV}/Archive/devbox-C/navsim/data/maps"


def labeller_env():
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=f"{V11};{REFE}", NUPLAN_MAPS_ROOT=MAPS, NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
                REFE_SIM_HZ="10", OMP_NUM_THREADS="1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--run-labellers", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sets_dir, pdm_dir, terms_dir = (os.path.join(a.work, x) for x in ("sets", "pdm", "terms"))
    os.makedirs(sets_dir, exist_ok=True)
    picked, seen = [], set()
    for fp in sorted(glob.glob(os.path.join(a.sets, "onpolicy_rlfan_*.jsonl"))):
        for ln in open(fp, encoding="utf-8"):
            r = json.loads(ln)
            if r["log_name"] in seen:
                continue
            seen.add(r["log_name"])
            picked.append(ln if ln.endswith("\n") else ln + "\n")
            if len(picked) >= a.n:
                break
        if len(picked) >= a.n:
            break
    with open(os.path.join(sets_dir, "onpolicy_rlfan_proof.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(picked)
    res = {"tool": "rl_fan_labeller_proof/1", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "sets": len(picked)}
    if a.run_labellers:
        for tool, out in (("onpolicy_relabel_pdm.py", pdm_dir), ("rl_terms_label.py", terms_dir)):
            t0 = time.time()
            p = subprocess.run([NV_PY, os.path.join(REFE, tool), "--sets", sets_dir, "--out", out, "--workers", str(a.workers)], env=labeller_env(), capture_output=True, text=True)
            tail = [l for l in p.stdout.splitlines() if l.startswith("ZZ")]
            res[tool] = {"rc": p.returncode, "sec": round(time.time() - t0, 1), "marker": tail[-1] if tail else None}
    import train as T
    import scorer_finetune as SF
    import rl_val_eval as RV
    n_prop, horizon = 64, 20
    bank = T.OnPolicyBank(sets_dir, n_prop, horizon)
    bank2 = T.OnPolicyBank(sets_dir, n_prop, horizon, pdm_labels=pdm_dir, pdm_only=True)
    keys = sorted(bank.by)
    res["OnPolicyBank_without_pdm"] = {"rows": bank.n_rows, "incomplete": bank.n_incomplete, "bad_lines": bank.n_bad_lines, "served": len(bank.by),
                                       "placeholder_targets_are_all_ones": bool(all((bank.by[k][2] == 1.0).all() for k in keys))}
    res["OnPolicyBank_pdm_only"] = {"served": len(bank2.by), "n_pdm": bank2.n_pdm, "n_pdm_stale": bank2.n_pdm_stale, "dropped_non_pdm": bank2.n_dropped_non_pdm}
    terms, st = SF.load_rl_terms(terms_dir, bank2.by, n_prop, True, False, False)
    res["scorer_finetune.load_rl_terms"] = dict(st, records=len(terms))
    pdm, pst = RV.load_pdm(pdm_dir)
    tms, tst = RV.load_terms(terms_dir)
    ck = {k: bank.by[k][0] for k in keys}
    paths = RV.load_paths(sets_dir, keys, ck)
    res["rl_val_eval"] = {"load_pdm": dict(pst, keys=len(pdm)), "load_terms": dict(tst, keys=len(tms)), "load_paths": len(paths)}
    n = len(picked)
    ok = (bank.n_incomplete == 0 and bank.n_bad_lines == 0 and len(bank.by) == n and len(bank2.by) == n and bank2.n_pdm == n and bank2.n_dropped_non_pdm == 0
          and len(terms) == n and len(pdm) == n and len(tms) == n and len(paths) == n and pst["bad_pdms"] == 0)
    if a.run_labellers:
        ok = ok and all(res[t]["rc"] == 0 for t in ("onpolicy_relabel_pdm.py", "rl_terms_label.py"))
    res["pass"] = bool(ok)
    out = a.out or os.path.join(a.work, "labeller_proof.json")
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(res, indent=1))
    print("ZZLABPROOF_PASS" if ok else "ZZLABPROOF_FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
