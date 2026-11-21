#!/usr/bin/env python3
"""L2-FRESH-STEPA step 1 (the carve): MEASURE what a 'fresh carve of navtrain held-out tokens disjoint from Step A's 24 logs' can be.  Read-only, CPU, NO label and NO score is read.

Universe facts (all MEASURED here, from the artifacts named in the output):
  U0  the literal recipe: plan tokens (1,705) minus every token whose log is one of Step A's 24 logs   -> its size is the answer to 'does the carve exist'
  B   the TOKEN-disjoint remainder: plan tokens not in Step A's 400 (same 24 logs)                       -> size, per log, banked-fan availability (G2b literal)
  C   the LOG-disjoint candidates: navtrain tokens of every log OUTSIDE the 24, whose nuPlan DB is on this box and whose four camera frames resolve with the planner's own FrameResolver
      (roots: the M6 proxy extract and the M5 extract; the held-out frames root is not used here), split by `served_train_sets` (P2's PDM-label training bank) and `in_navtest`

    driverl-eval python probe_fresh_universe.py --out <dir> [--skip-c]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

W = "D:/Projects/TanitAD-artifacts/l1_stepA_run"
PLAN = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"
TABLE = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz"
PK = "C:/Users/Admin/par18/l1prep/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TOKTAB = f"{PK}/raw/rl_pool/navtrain_token_table.tsv.gz"
HELDOUT_LOGS = f"{PK}/splits/sft_heldout_logs.txt"
TREE = "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB_ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
ROOTS = ["D:/Projects/TanitAD/data/refe_proxy/pixels", "D:/Projects/TanitAD/data/refe_m5/pixels"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def literal_and_token_disjoint(plan, sample, table_tokens, heldout_logs):
    """pure function (tested): the literal log-disjoint remainder and the token-disjoint remainder"""
    tl = plan["token_log"]
    logsA = sorted({tl[t] for t in sample["tokens"]})
    lit = [t for t in plan["tokens"] if tl[t] not in set(logsA)]
    tokB = [t for t in plan["tokens"] if t not in set(sample["tokens"])]
    perlog = collections.Counter(tl[t] for t in tokB)
    return {"stepA_tokens": len(sample["tokens"]), "stepA_logs": len(logsA), "plan_tokens": len(plan["tokens"]), "plan_logs": len(set(tl.values())),
            "plan_logs_equal_sft_heldout_logs": set(tl.values()) == set(heldout_logs), "stepA_logs_equal_plan_logs": set(logsA) == set(tl.values()),
            "U0_literal_remaining_tokens": len(lit), "U0_literal_remaining_logs": len({tl[t] for t in lit}),
            "B_token_disjoint_tokens": len(tokB), "B_token_disjoint_logs": len(perlog), "B_per_log": dict(sorted(perlog.items())),
            "B_all_have_banked_fan_in_table": all(t in table_tokens for t in tokB), "table_tokens_not_in_plan": len(set(table_tokens) - set(plan["tokens"]))}


def read_tsv(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(head, line.rstrip("\n").split("\t")))


def probe_c(tt, heldout_logs, out_dir):
    sys.path.insert(0, f"{TREE}/refe")
    import planner as PL
    dbs = {}
    for s in ("mini", "test", "trainval"):
        for p in glob.glob(os.path.join(DB_ROOT, s, "*.db")):
            dbs.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    for p in glob.glob(os.path.join(DB_ROOT, "*.db")):
        dbs.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    by_log = collections.defaultdict(list)
    for t, r in tt.items():
        if r["log"] not in heldout_logs:
            by_log[r["log"]].append(t)
    # logs with ANY frame in a root (zip name or a log directory); a log with none cannot resolve a token
    have = set()
    for r in ROOTS:
        for n in os.listdir(r):
            have.add(n.split("_CAM_")[0] if n.endswith(".zip") else n)
    logs = sorted(l for l in by_log if l in have)
    out = {"navtrain_logs_outside_heldout": len(by_log), "tokens_outside_heldout": sum(len(v) for v in by_log.values()), "logs_with_any_frame_in_roots": len(logs),
           "logs_without_db_among_those": sum(1 for l in logs if l not in dbs), "roots": ROOTS}
    fr = {r: PL.FrameResolver(db_dir="", images_root=r) for r in ROOTS}
    ok, why = [], collections.Counter()
    t0 = time.time()
    for i, lg in enumerate(logs):
        if lg not in dbs:
            why["no_db"] += len(by_log[lg])
            continue
        con = sqlite3.connect(f"file:{dbs[lg]}?mode=ro", uri=True)
        want = set(by_log[lg])
        times = {t: int(ts) for t, ts in con.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc") if t in want}
        con.close()
        for t in by_log[lg]:
            if t not in times:
                why["token_not_in_db"] += 1
                continue
            got = None
            for r, res in fr.items():
                res.db_dir = os.path.dirname(dbs[lg])
                paths = res.resolve(lg, times[t])
                if paths:
                    got = r
                    break
            if got:
                ok.append((t, lg, got, int(tt[t]["served_train_sets"] or 0) > 0, tt[t]["in_navtest"]))
            else:
                why["no_frames"] += 1
        if i % 50 == 0:
            print(f"ZZC {i}/{len(logs)} ok={len(ok)} {time.time() - t0:.0f}s", flush=True)
    out["resolvable_tokens"] = len(ok)
    out["resolvable_logs"] = len({x[1] for x in ok})
    out["resolvable_unserved"] = sum(1 for x in ok if not x[3])
    out["resolvable_unserved_logs"] = len({x[1] for x in ok if not x[3]})
    out["resolvable_served"] = sum(1 for x in ok if x[3])
    out["resolvable_in_navtest_flag"] = sum(1 for x in ok if x[4] == "1")
    out["resolvable_by_root"] = dict(collections.Counter(x[2] for x in ok))
    out["miss_reasons"] = dict(why)
    json.dump([{"token": t, "log": l, "root": r, "served": s} for t, l, r, s, _ in ok], open(os.path.join(out_dir, "c_resolvable_tokens.json"), "w"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-c", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    plan = json.load(open(PLAN))
    sample = json.load(open(f"{W}/sample.json"))
    table = np.load(TABLE, allow_pickle=True)
    ttok = {str(t) for t in table["token"]}
    held = set(open(HELDOUT_LOGS, encoding="utf-8").read().split())
    res = {"what": "L2-FRESH-STEPA carve feasibility probe (MEASURED; no label/score read)", "inputs": {k: {"path": p, "sha256": sha_file(p)} for k, p in
           (("plan", PLAN), ("stepA_sample", f"{W}/sample.json"), ("table", TABLE), ("token_table", TOKTAB), ("heldout_logs", HELDOUT_LOGS))}}
    res["U0_B"] = literal_and_token_disjoint(plan, sample, ttok, held)
    tt = {r["token"]: r for r in read_tsv(TOKTAB)}
    res["token_table"] = {"rows": len(tt), "logs": len({r["log"] for r in tt.values()}), "heldout_log_tokens": sum(1 for r in tt.values() if r["heldout_log"] == "1"),
                          "other_log_served": sum(1 for r in tt.values() if r["heldout_log"] == "0" and int(r["served_train_sets"] or 0) > 0),
                          "other_log_unserved": sum(1 for r in tt.values() if r["heldout_log"] == "0" and int(r["served_train_sets"] or 0) == 0),
                          "in_navtest_rows": sum(1 for r in tt.values() if r["in_navtest"] == "1")}
    print("ZZU0B", json.dumps({k: v for k, v in res["U0_B"].items() if k != "B_per_log"}), flush=True)
    print("ZZTT", json.dumps(res["token_table"]), flush=True)
    if not a.skip_c:
        res["C_log_disjoint_other_navtrain_logs"] = probe_c(tt, held, a.out)
        print("ZZC_DONE", json.dumps(res["C_log_disjoint_other_navtrain_logs"]), flush=True)
    json.dump(res, open(os.path.join(a.out, "fresh_universe_probe.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
