#!/usr/bin/env python3
"""Step B labelling: the TRUE PDM sub-scores of every SEARCH OUTPUT, on the instrument that labelled P2's training sets, the banked held-out table and Step A's perturbations
(`label_perturbations.label_plans` -> `navsim_pdm_targets.score_plans` on the metric cache of a scenario rebuilt from the nuPlan DB).  Nothing of the labeller is re-implemented: this file only feeds it the
composites (u, d) the search chose.  Needs the navsim / nuPlan venv, CPU only (the env of L1's step 5):

    navsim-crun label_stepB.py --bundle <bundle.npz> --search <search_out.npz | shard> --plan <heldout_tokens.json> --out <labels_stepB.npz> [--tree <PK>] [--discard-labels --timing-out <json>]
    env as `steps4to7.cmd`: PYTHONPATH=<navsim v1.1>;<tree>/refe  NUPLAN_MAPS_ROOT=...  NUPLAN_MAP_VERSION=nuplan-maps-v1.0  REFE_SIM_HZ=10  OMP_NUM_THREADS=1

THE PLANTED IDENTITY (the label-path check, run FIRST and enforced, per token): the composite (u, d) = (0, 0) of the pick's raw plan (`tts_cem.composite`, the very function the search uses) is labelled by the
same path as every search output and MUST reproduce the pick's BANKED label (`bundle['sub'][n, pick]`, `bundle['pdms'][n, pick]`) to <= 1e-6 (observed exactly 0.0 or the run is refused, exit 3).  Without it the
endpoint (search output minus the pick's banked label) could carry a label-path offset: the pick's label came from the 8-pose table path, a search output's from the 20-to-8 conversion of an executed plan.
Its control (a plan shifted 0.5 m forward) must change the label for at least one token (exit 3 otherwise), so the identity is not a probe that cannot read a difference.

`--discard-labels` (smoke / cost runs): the labels are COMPUTED for timing and identity but never written or printed.  The outcome (dPDMS) is not read by this file; `analyse_stepB.py` reads it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stepB  # noqa: E402  (puts L1's code on sys.path; numpy-only here)

A, T = stepB.A, stepB.T
SUBS = A.SUBS
ID_TOL = 1e-6


def planted_identity_plans(raw_pick) -> tuple:
    """(identity exec [20,3], shifted-control exec [20,3]) for one token: composite(0, 0) of the pick's raw plan, and the same plan 0.5 m further ahead"""
    ex = T.composite(np.asarray(raw_pick, np.float32), 0.0, 0.0)[0]
    sh = np.array(ex, copy=True)
    sh[:, 0] += 0.5
    return ex, sh


def identity_record(id_sub, id_pdms, banked_sub, banked_pdms, tol: float = ID_TOL) -> dict:
    """max |label of composite(0,0) - banked label of the pick| over the six sub-scores and the PDMS"""
    d = max(float(np.max(np.abs(np.asarray(id_sub, np.float64) - np.asarray(banked_sub, np.float64)))), abs(float(id_pdms) - float(banked_pdms)))
    return {"max_abs_diff": d, "tol": tol, "passed": bool(np.isfinite(d) and d <= tol)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--search", required=True, help="a merged search_out.npz or a single shard (arrays `exec` [n,S,20,3], `bundle_index`)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tree", default="C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
    ap.add_argument("--discard-labels", action="store_true", help="smoke/cost: compute for timing + identity, write and print NO label")
    ap.add_argument("--timing-out", default=None)
    a = ap.parse_args(argv)
    if not a.discard_labels and not a.out:
        raise SystemExit("--out is required unless --discard-labels")
    stepB.verify_l1_code()
    sys.path.insert(0, os.path.join(a.tree, "refe"))
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    import label_perturbations as LP
    PT._setup()
    b = A.load_npz(a.bundle)
    so = np.load(a.search, allow_pickle=False)
    bidx = np.asarray(so["bundle_index"], np.int64)
    n, S = so["exec"].shape[:2]
    plan = json.load(open(a.plan, encoding="utf-8"))
    toks = [str(t) for t in np.asarray(b["token"])[bidx]]
    if [str(t) for t in so["token"]] != toks:
        raise SystemExit("the search file's tokens are not the bundle's tokens at its bundle_index: refusing (a label for the wrong token is not a label)")
    sub = np.full((n, S, 6), np.nan)
    pdms = np.full((n, S), np.nan)
    id_sub, id_pdms, sh_pdms = np.full((n, 6), np.nan), np.full(n, np.nan), np.full(n, np.nan)
    ident, t_log, t_parts, t_lab = [], {}, 0.0, 0.0
    by_log = {}
    for i, t in enumerate(toks):
        by_log.setdefault(plan["token_log"][t], []).append(i)
    t0 = time.time()
    for log, rows in sorted(by_log.items()):
        ts = time.time()
        scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(plan["db"][log], sorted(toks[i] for i in rows), history_rows=40, future_rows=100)}
        t_log[log] = time.time() - ts
        for i in rows:
            sc = scs.get(toks[i])
            if sc is None:
                print(f"  {log}: scenario builder skipped {toks[i]}", flush=True)
                continue
            tp = time.time()
            parts = PT.metric_cache_parts(sc)
            t_parts += time.time() - tp
            bi = int(bidx[i])
            pick = int(b["pick"][bi])
            ex_id, ex_sh = planted_identity_plans(np.asarray(b["traj_raw"][bi], np.float32)[pick])
            if not np.array_equal(ex_id, np.asarray(so["pick_exec"][i], np.float32)):
                raise SystemExit(f"token {toks[i]}: composite(0,0) of the pick is not the shard's pick_exec: the search and the labeller disagree about what 'the pick' is")
            tl = time.time()
            s_all, p_all = LP.label_plans(PT, parts, np.concatenate([ex_id[None], np.asarray(so["exec"][i], np.float32), ex_sh[None]], 0))
            t_lab += time.time() - tl
            id_sub[i], id_pdms[i], sh_pdms[i] = s_all[0], p_all[0], p_all[-1]
            sub[i], pdms[i] = s_all[1:1 + S], p_all[1:1 + S]
            ident.append({"token": toks[i], **identity_record(s_all[0], p_all[0], np.asarray(b["sub"][bi][pick], np.float64), float(b["pdms"][bi][pick]))})
    worst = max((r["max_abs_diff"] for r in ident), default=float("nan"))
    changed = int(sum(1 for i in range(n) if np.isfinite(sh_pdms[i]) and not np.isclose(sh_pdms[i], id_pdms[i], atol=1e-9)))
    gate = {"n_checked": len(ident), "n_tokens": n, "max_abs_diff": worst, "tol": ID_TOL, "passed": bool(len(ident) == n and all(r["passed"] for r in ident)), "control_tokens_whose_label_changed": changed,
            "control_ok": bool(changed > 0)}
    secs = time.time() - t0
    timing = {"n_tokens": n, "n_seeds": S, "plans_labelled_per_token": S + 2, "seconds_total": secs, "seconds_scenario_build_total": float(sum(t_log.values())), "n_logs": len(t_log),
              "seconds_metric_cache_parts_total": t_parts, "seconds_label_plans_total": t_lab, "seconds_per_token_all_in": secs / max(n, 1)}
    print("ZZIDENTITY", json.dumps({k: gate[k] for k in gate}), flush=True)
    print("ZZTIMING", json.dumps(timing), flush=True)
    if a.timing_out:
        with open(a.timing_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"identity": gate, "timing": timing}, f, indent=1)
    if not (gate["passed"] and gate["control_ok"]):
        print("ZZLABEL_FAIL planted identity or its control (the label path is not the pick's label path)", flush=True)
        return 3
    if a.discard_labels:
        print("ZZLABEL_DISCARDED (smoke: identity + timing only, no label written)", flush=True)
        return 0
    tmp = a.out + f".tmp{os.getpid()}.npz"
    np.savez(tmp, sub=sub, pdms=pdms, id_sub=id_sub, id_pdms=id_pdms, sub_names=np.array(SUBS), token=np.asarray(toks), bundle_index=bidx, search_sha256=np.array(stepB.l2_scorer.sha256_file(a.search)),
             identity=np.array(json.dumps(gate)))
    os.replace(tmp, a.out)
    ok = bool(np.isfinite(sub).all() and np.isfinite(pdms).all())
    print(f"ZZLABEL_{'OK' if ok else 'PARTIAL'} {int(np.isfinite(pdms).all(1).sum())}/{n} {secs:.0f} s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
