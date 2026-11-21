#!/usr/bin/env python3
"""SFT-4 labels: NAVSIM v1.1 PDM targets for the on-policy sets the trainer SERVES (pod, CPU). The paper's scorer labels
(DriveZero Sec. 3: "supervised by their corresponding PDM targets [17]"; RETRACTION_LOG R30).

Per set -- the newest (ckpt_step, label_version) per (log, token, step, rank), exactly train.OnPolicyBank's rule; held-out
logs are a separate --sets dir -- the proposals' EXECUTED plans (A7 last-heading repair -> to_navsim) are scored by
refe/navsim_pdm_targets.py: NAVSIM's own metric-cache steps on the nuPlan scenario at the set's token (history 2 s,
future 5 s), NAVSIM's own simulator + scorer. Side files keyed like OnPolicyBank + ckpt_step:
    {"kind": "pdm_targets", "key": [log, token, step, rank], "ckpt_step": c, "nc": [M], "dac": [M], "ep": [M],
     "ttc": [M], "c": [M], "ddc": [M], "pdms": [M]}
Validated before use: eval/validate_pdm_targets.py (rebuilt cache vs NAVSIM's real navtest cache) and, on the pod, DDC ==
the independently computed navsim_ddc side files.
    python refe/onpolicy_relabel_pdm.py --sets <dir> --out <dir> [--workers 16] [--served-only] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
_S: dict = {}


def _init():
    import navtrain_scenarios as NS
    _S["NS"] = NS
    _S["dbs"] = NS.index_dbs()


def label_log(job):
    import navsim_pdm_targets as PT
    log, rows = job
    out, failed, skipped = [], 0, 0
    try:
        if not _S:
            _init()
        scs = {sc.scenario_name: sc for sc in _S["NS"].build_scenarios_for_log(
            _S["dbs"][log], sorted({r["token"] for r in rows}), history_rows=40, future_rows=100)}
    except Exception as e:                                                   # noqa: BLE001
        return [], len(rows), 0, f"{log}: scenario build {type(e).__name__}: {str(e)[:120]}"
    for r in rows:
        sc = scs.get(r["token"])
        if sc is None:                                  # the 5 s future guard (NAVSIM's observation horizon) excludes it
            skipped += 1
            continue
        try:
            props = np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1)
            t = PT.score_plans(PT.metric_cache_parts(sc), PT.executed_plans8(props, repair=True))
            rec = {"kind": "pdm_targets", "key": [r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))],
                   "ckpt_step": int(r["ckpt_step"]), "label_version": int(r.get("label_version", 1)),
                   "plans": "executed (A7 repair)"}
            rec.update({k: [round(float(x), 5) for x in v] for k, v in t.items()})
            out.append(json.dumps(rec) + "\n")
        except Exception:                                                    # noqa: BLE001
            failed += 1
            if failed <= 2:
                print(f"  {r.get('token')}: {traceback.format_exc()[-400:]}", flush=True)
    return out, failed, skipped, None


def served_keys(files):
    import select_bank_subset as SB
    best = {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                f = SB.head_fields(line[:2000] + line[-400:])
                if not {"log_name", "step", "ckpt_step"} <= set(f):
                    continue
                k = (f["log_name"], f.get("token", ""), int(f["step"]), int(f.get("rank", 0)))
                rk = (int(f["ckpt_step"]), int(f.get("label_version", 1)))
                if k not in best or best[k] < rk:
                    best[k] = rk
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default="onpolicy_*.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--served-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.sets, a.files)))
    best = served_keys(files) if a.served_only else None
    if best is not None:
        print(f"served sets: {len(best):,}", flush=True)
    n_sets = n_fail = n_skip = 0
    t0 = time.time()
    CH = 3000
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for fpath in files:
            dst = os.path.join(a.out, "pdm_" + os.path.basename(fpath))
            done = set()
            if os.path.exists(dst):
                for ln in open(dst, encoding="utf-8"):
                    try:
                        d = json.loads(ln)
                        done.add((tuple(d["key"]), d["ckpt_step"]))
                    except (json.JSONDecodeError, KeyError):
                        pass

            def flush(by, f):
                c = [0, 0, 0]
                for lines, failed, skipped, err in pool.imap_unordered(label_log, list(by.items())):
                    if err:
                        print("  " + err, flush=True)
                    f.writelines(lines)
                    f.flush()
                    c[0] += len(lines); c[1] += failed; c[2] += skipped
                return c

            by: dict = {}
            k = 0
            with open(dst, "a", encoding="utf-8") as f:
                for ln in open(fpath, encoding="utf-8"):
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if r.get("kind") != "onpolicy_set":
                        continue
                    key = (r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))
                    rk = (int(r["ckpt_step"]), int(r.get("label_version", 1)))
                    if best is not None and best.get(key) != rk:
                        continue
                    if (key, rk[0]) in done:
                        continue
                    by.setdefault(r["log_name"], []).append({"log_name": r["log_name"], "token": r.get("token", ""),
                                                             "step": r["step"], "rank": r.get("rank", 0),
                                                             "ckpt_step": r["ckpt_step"], "label_version": r.get("label_version", 1),
                                                             "traj": r["traj"], "yaw": r["yaw"]})
                    k += 1
                    if k % CH == 0:
                        c = flush(by, f)
                        n_sets += c[0]; n_fail += c[1]; n_skip += c[2]
                        by = {}
                        print(f"  {os.path.basename(fpath)}: {k:,} read, {n_sets:,} labelled, {n_skip} skipped (5 s guard), "
                              f"{n_fail} failed, {(time.time() - t0) / max(n_sets, 1):.3f} s/set", flush=True)
                    if a.limit and k >= a.limit:
                        break
                if by:
                    c = flush(by, f)
                    n_sets += c[0]; n_fail += c[1]; n_skip += c[2]
            print(f"{os.path.basename(fpath)}: total {n_sets:,} labelled, {n_skip} skipped, {n_fail} failed", flush=True)
    print(f"ZZRELABEL_PDM_DONE {n_sets} {n_skip} {n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
