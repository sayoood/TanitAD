"""UNSEEN-STRATUM S1b worker: score the S1 candidate tokens of a list of logs with OUR PDM scorer (CV agent + human), one jsonl per log.
CPU only (CUDA_VISIBLE_DEVICES=-1, OMP 1), sqlite read-only, no frame, no model, no head.  Nothing here reads a label of any endpoint.

  --mode all   score EVERY candidate of each log (identity check)
  --mode seq   score the candidates in a SEEDED per-log permutation (the SAME permutation S1's fetch list used: SeedSequence-free
               default_rng([seed, int(sha256(log)[:8], 16)])) until `--n-accept` tokens pass the frozen filter and resolve their 4 cameras,
               or `--cap` tokens were scored.  This is "draw uniformly from the FILTERED candidates" without scoring all of them.

The frozen filter (Dauner et al. 2024 sec. 3.1; NAVSIM_PROTOCOL.md:386-391):  REMOVE if PDMS(CV) > 0.8;  REMOVE if PDMS(human) < 0.8.
    keep  =  PDMS(CV) <= 0.8  and  PDMS(human) >= 0.8        (literal reading; the other inclusivity variants are evaluated offline from the stored scores)

    python s1b_worker.py --mode all --logs a,b,c --out <dir> [--extras]
Exit 0 = every log done;  75 = RSS limit reached after a finished log (driver relaunches);  1 = hard failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1b_lib as L  # noqa: E402

L.setup_env()
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import unseen_token_rule as R  # noqa: E402  (the FROZEN S1 module, imported, not copied)
import navtrain_scenarios as NS  # noqa: E402
import navsim_pdm_targets as PT  # noqa: E402

from s1b_rules import keep_rule, perm_for  # noqa: E402  (pure, tested)
RSS_LIMIT_MB = 1500
KEYS = ("pdms", "nc", "dac", "ttc", "c", "ddc")      # NOTE: the ego-progress sub-score is not stored (it is inside PDMS by the NAVSIM formula)


def slim(d: dict) -> dict:
    return {k: round(float(d[k]), 6) for k in KEYS}


def read_done(path: str) -> dict:
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["t"]] = r
                except Exception:  # noqa: BLE001  (a torn last line from a killed worker)
                    pass
    return out


def do_log(lg: str, a, proc) -> dict:
    db = f"{a.db}/{lg}.db"
    path = f"{a.out}/{lg}.jsonl"
    rec = R.read_log(db)
    c = R.candidates(rec)
    toks, lp = c["tokens"], c["lp_idx"]
    ts = {t: int(rec["ts"][i]) for t, i in zip(toks, lp)}
    idx = {t: int(i) for t, i in zip(toks, lp)}
    order = np.arange(len(toks)) if a.mode == "all" else perm_for(a.seed, lg, len(toks))
    done = read_done(path)
    n_acc = sum(1 for r in done.values() if r.get("keep") and r.get("keep_counted", True))
    n_scored = len(done)
    t_log0, c_log0 = time.perf_counter(), time.process_time()
    f = open(path, "a", encoding="utf-8")
    pos = 0
    ordered = [toks[int(j)] for j in order]
    # tokens already scored keep their place in the order; the loop skips them
    while pos < len(ordered):
        if a.mode == "seq" and (n_acc >= a.n_accept or n_scored >= a.cap):
            break
        chunk = []
        while pos < len(ordered) and len(chunk) < a.chunk:
            t = ordered[pos]
            pos += 1
            if t not in done:
                chunk.append(t)
        if not chunk:
            continue
        gen = iter(NS.build_scenarios_for_log(db, chunk, map_root=L.MAPS, history_rows=0, future_rows=100))
        seen = set()
        while True:
            if a.mode == "seq" and (n_acc >= a.n_accept or n_scored >= a.cap):
                break
            w0, p0 = time.perf_counter(), time.process_time()
            try:
                sc = next(gen)
            except StopIteration:
                break
            tok = sc._initial_lidar_token
            seen.add(tok)
            row = {"t": tok, "i": int(idx[tok])}
            try:
                extra = None
                if a.extras:
                    v0 = L.ego_speed(sc)
                    extra = np.stack([L.cv_plan(0.0), L.cv_plan(2.0 * v0)]).astype(np.float64)
                o = L.score_token(PT, sc, extra_plans=extra)
                row.update({"v": round(o["speed"], 4), "cv": slim(o["cv"]), "h": slim(o["human"]) if o["human"] is not None else None,
                            "tc": round(o["t_cache"], 4), "ts": round(o["t_score"], 4)})
                if "extra" in o:
                    row["x0"], row["x2"] = slim(o["extra"][0]), slim(o["extra"][1])
                ok = o["human"] is not None
                row["res_ok"] = bool(ok)
                row["keep"] = bool(ok and keep_rule(o["cv"]["pdms"], o["human"]["pdms"]))
            except Exception as e:  # noqa: BLE001
                row.update({"err": f"{type(e).__name__}: {str(e)[:160]}", "keep": False, "res_ok": False})
            row["cpu"] = round(time.process_time() - p0, 4)
            row["wall"] = round(time.perf_counter() - w0, 4)
            if a.mode == "seq" and row.get("keep"):
                nj = R.nearest_jpgs(db, ts[tok])          # the 4 cameras must resolve (S1's fetch-list rule); an unresolvable token does not count
                row["resolvable"] = nj is not None
                if nj is None:
                    row["keep_counted"] = False
                else:
                    n_acc += 1
            n_scored += 1
            f.write(json.dumps(row) + "\n")
            f.flush()
        for t in chunk:
            if t not in seen and not (a.mode == "seq" and (n_acc >= a.n_accept or n_scored >= a.cap)):
                f.write(json.dumps({"t": t, "skipped_guard": True, "keep": False, "res_ok": False}) + "\n")
                n_scored += 1
        f.flush()
    f.close()
    n_kept = sum(1 for r in read_done(path).values() if r.get("keep"))
    summ = {"log": lg, "mode": a.mode, "n_candidates": len(toks), "n_scored": n_scored, "n_accepted": n_acc, "n_kept": n_kept,
            "wall_s": round(time.perf_counter() - t_log0, 2), "cpu_s": round(time.process_time() - c_log0, 2)}
    open(f"{a.out}/{lg}.done", "w").write(json.dumps(summ))
    return summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("all", "seq"), required=True)
    ap.add_argument("--logs", required=True, help="comma list or @file (one log per line)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--db", default=L.DB_TRAINVAL)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--n-accept", type=int, default=4)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--extras", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logs = [l.strip() for l in open(a.logs[1:]).read().split() if l.strip()] if a.logs.startswith("@") else a.logs.split(",")
    PT._setup()
    pr = psutil.Process()
    proc = None
    t0 = time.time()
    for lg in logs:
        if os.path.exists(f"{a.out}/{lg}.done"):
            continue
        s = do_log(lg, a, proc)
        rss = pr.memory_info().rss / 1e6
        print(f"ZZLOG {json.dumps(s)} rss_mb={rss:.0f} elapsed={time.time()-t0:.0f}s", flush=True)
        if rss > RSS_LIMIT_MB:
            print("ZZRSS_LIMIT exit 75", flush=True)
            return 75
    print("ZZWORKER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
