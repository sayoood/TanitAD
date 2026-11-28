#!/usr/bin/env python3
"""A2 behaviour labeller (OFFLINE side files): the per-proposal behaviour terms (q_free, q_gap, q_osc, q_stall, q_comf; behaviour_terms.ORDER) and their applicable masks for banked on-policy sets.
Copy-and-trim of rl_terms_label.py (same CLI, sharding, resume rule, scenario build); the labels come from the SAME simulated states the PDM labels use:
    PT.score_plans(parts, PT.executed_plans8(props, repair=True), behaviour=True)["behaviour"]   -> behaviour_terms.pack -> (q [M, 5], applicable [M, 5])
NAVTEST QUARANTINE: behaviour_gate.assert_batch runs on every (log_name, token) BEFORE any scenario is built or any record is written (main(), on read; label_log(), again on the job); a navtest row
raises NavtestLeak and the run stops (fail closed: the lists missing => everything refused).

Files: <out>/behaviour_<set file name> (the glob scorer_finetune.load_beh_terms reads: `--beh-labels-train/-heldout <out>`).
Record (one per (log, token, step, rank, ckpt_step)), exactly what scorer_finetune._beh_record validates:
    {"kind": "behaviour", "key": [log, token, step, rank], "ckpt_step": c, "label_version": v, "q": [M][5], "act": [M][5] (0/1), "terms_version": "behaviour/1"}

    python refe/behaviour_label.py --sets <dir> --out <dir> [--workers 16] [--served-only] [--limit N] [--shard I/N]
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
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

BEH_PREFIX = "behaviour_"                          # == the glob scorer_finetune.BEH_GLOB ("behaviour_*.jsonl"; test_a2_wiring pins it)
TERMS_VERSION = "behaviour/1"                      # == scorer_finetune.BEH_TERMS_VERSION


def make_record(key, ckpt_step: int, label_version: int, q: np.ndarray, act: np.ndarray) -> dict:
    q, act = np.asarray(q, np.float32), np.asarray(act).astype(bool)
    if q.ndim != 2 or q.shape[1] != 5 or act.shape != q.shape or not np.isfinite(q).all() or q.min() < 0.0 or q.max() > 1.0 + 1e-6:
        raise ValueError(f"behaviour record {tuple(key)}: q {q.shape} act {act.shape} (need [M, 5], finite, in [0, 1])")
    return {"kind": "behaviour", "key": list(key), "ckpt_step": int(ckpt_step), "label_version": int(label_version),
            "q": [[round(float(x), 6) for x in row] for row in q], "act": [[int(x) for x in row] for row in act], "terms_version": TERMS_VERSION}


def side_file_name(src_path: str) -> str:
    """onpolicy_<tag>.jsonl -> behaviour_onpolicy_<tag>.jsonl"""
    return BEH_PREFIX + os.path.basename(src_path)


def shard_of(log: str, n: int) -> int:
    """Log-disjoint, deterministic sharding: one log -> one shard, on every machine (== rl_terms_label.shard_of)."""
    return zlib.crc32(log.encode("utf-8")) % n


def label_scene(sc, plans8, parts=None):
    """-> (q [M, 5] float32, act [M, 5] bool) of one scene's proposals, evaluated on the EXECUTED states"""
    import behaviour_terms as BT
    import navsim_pdm_targets as PT
    parts = parts or PT.metric_cache_parts(sc)
    return BT.pack(PT.score_plans(parts, plans8, behaviour=True)["behaviour"])


# ------------------------------------------------------------------ CLI (mirrors refe/rl_terms_label.py)
_S: dict = {}


def _init():
    import navtrain_scenarios as NS
    _S["NS"] = NS
    _S["dbs"] = NS.index_dbs()


def label_log(job):
    import behaviour_gate as BG
    import navsim_pdm_targets as PT
    log, rows = job
    BG.assert_batch({"log_name": r.get("log_name"), "token": r.get("token")} for r in rows)       # the quarantine, BEFORE any scenario is built or any line returned
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
            q, act = label_scene(sc, PT.executed_plans8(props, repair=True))
            rec = make_record([r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))], r["ckpt_step"], r.get("label_version", 1), q, act)
            out.append(json.dumps(rec) + "\n")
        except Exception:                                                    # noqa: BLE001
            failed += 1
            if failed <= 2:
                print(f"  {r.get('token')}: {traceback.format_exc()[-400:]}", flush=True)
    return out, failed, skipped, None


def main() -> int:
    import behaviour_gate as BG
    import onpolicy_relabel_pdm as OR
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default="onpolicy_*.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--served-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="I/N: label only the logs whose crc32 mod N == I (log-disjoint across machines)")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    if not (0 <= si < sn):
        raise SystemExit(f"--shard {a.shard}: need 0 <= I < N")
    BG.load()                                                                # fail closed NOW if the quarantine lists are absent or altered
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.sets, a.files)))
    best = OR.served_keys(files) if a.served_only else None
    n_sets = n_fail = n_skip = 0
    t0 = time.time()
    CH = 3000
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for fpath in files:
            dst = os.path.join(a.out, side_file_name(fpath))
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
                    if r.get("kind") != "onpolicy_set" or shard_of(r["log_name"], sn) != si:
                        continue
                    BG.assert_batch([{"log_name": r["log_name"], "token": r.get("token", "")}])
                    key = (r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))
                    rk = (int(r["ckpt_step"]), int(r.get("label_version", 1)))
                    if best is not None and best.get(key) != rk:
                        continue
                    if (key, rk[0]) in done:
                        continue
                    by.setdefault(r["log_name"], []).append({"log_name": r["log_name"], "token": r.get("token", ""), "step": r["step"], "rank": r.get("rank", 0),
                                                             "ckpt_step": r["ckpt_step"], "label_version": r.get("label_version", 1), "traj": r["traj"], "yaw": r["yaw"]})
                    k += 1
                    if k % CH == 0:
                        c = flush(by, f)
                        n_sets += c[0]; n_fail += c[1]; n_skip += c[2]
                        by = {}
                        print(f"  {os.path.basename(fpath)}: {k:,} read, {n_sets:,} labelled, {n_skip} skipped (5 s guard), {n_fail} failed, {(time.time() - t0) / max(n_sets, 1):.3f} s/set", flush=True)
                    if a.limit and k >= a.limit:
                        break
                if by:
                    c = flush(by, f)
                    n_sets += c[0]; n_fail += c[1]; n_skip += c[2]
            print(f"{os.path.basename(fpath)}: total {n_sets:,} labelled, {n_skip} skipped, {n_fail} failed", flush=True)
    print(f"ZZRELABEL_BEH_DONE {n_sets} {n_skip} {n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
