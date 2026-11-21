#!/usr/bin/env python3
"""SR-TTC S0 -- per-BATCH plan + availability audit.  Pure bookkeeping: no GPU, no model.  Derived from the held-out `p2ho_audit.py` (same plan format, same `FrameResolver`).

Input   --tokens-json  {"tokens": {token: log}}   the tokens of this batch's logs (from the master plan)
Output  <out>/batch_tokens.json  {tokens, token_log, token_root, token_ts_us, db}   the tokens that have a nuPlan DB AND four resolvable camera frames on this box
        <out>/audit.json         per-log status, blocked tokens with reasons
        <out>/needed_frames.json tokens whose DB is present but whose frames are not: the nearest-timestamp `<log>/<CAM>/<hash>.jpg` per camera (what to fetch from the pod)
        <out>/needed_dbs.json    the logs with no DB anywhere on this box
DB search order: the staging DB dir first, then the dev box's nuPlan trees.  Image roots in the order given (staging first, then the proxy extracts).
Runs in the driverl-eval interpreter (numpy + the tree's planner for FrameResolver).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys

DB_ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
TREE = os.environ.get("P2HO_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
PROXY_ROOTS = ["D:/Projects/TanitAD/data/refe_proxy/pixels",            # M6 proxy extract of the pod's navtrain_pixels (4-camera zips, a stratified sample)
               "D:/Projects/TanitAD/data/refe_m5/pixels"]               # M5 dry-run extract (loose jpgs)


def index_dbs(stage_db_dir):
    out = {}
    if stage_db_dir:
        for p in glob.glob(os.path.join(stage_db_dir, "*.db")):
            out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    for s in ("mini", "test", "trainval"):
        for p in glob.glob(os.path.join(DB_ROOT, s, "*.db")):
            out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    for p in glob.glob(os.path.join(DB_ROOT, "*.db")):
        out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    return out


def token_times(db, toks):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        want = set(toks)
        return {t: int(ts) for t, ts in con.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc") if t in want}
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-db-dir", default=None)
    ap.add_argument("--images-root", action="append", default=None, help="in order; the proxy extracts are appended automatically")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok_log = json.load(open(a.tokens_json, encoding="utf-8"))["tokens"]
    roots = list(a.images_root or []) + PROXY_ROOTS
    by_log: dict = {}
    for t, lg in tok_log.items():
        by_log.setdefault(lg, []).append(t)
    dbs = index_dbs(a.stage_db_dir)
    sys.path.insert(0, f"{TREE}/refe")
    import planner as PL
    fr = {r: PL.FrameResolver(db_dir="", images_root=r) for r in roots if os.path.isdir(r)}
    blocked, ok_tokens, token_log, token_root, token_ts, needed, needed_dbs, per_log = {}, [], {}, {}, {}, {}, [], {}
    for lg in sorted(by_log):
        toks = sorted(by_log[lg])
        rec = {"n_tokens": len(toks), "db": dbs.get(lg)}
        if lg not in dbs:
            rec["status"] = "BLOCKED_NO_DB"
            needed_dbs.append(lg)
            for t in toks:
                blocked[t] = {"log": lg, "reason": "no nuPlan DB for this log on this box"}
            per_log[lg] = rec
            continue
        times = token_times(dbs[lg], toks)
        n_img, by_root = 0, {}
        for t in toks:
            if t not in times:
                blocked[t] = {"log": lg, "reason": "token not in the DB's lidar_pc"}
                continue
            got, why = None, None
            for r, res_ in fr.items():
                res_.db_dir = os.path.dirname(dbs[lg])
                paths = res_.resolve(lg, times[t])
                if paths:
                    got = (r, paths)
                    break
                why = res_.miss_reason
            if got:
                n_img += 1
                ok_tokens.append(t)
                token_log[t] = lg
                token_ts[t] = times[t]
                token_root[t] = got[0]
                by_root[got[0]] = by_root.get(got[0], 0) + 1
            else:
                blocked[t] = {"log": lg, "reason": f"no camera frames: {why}"}
                r0 = next(iter(fr.values()))
                r0.db_dir = os.path.dirname(dbs[lg])
                per = r0._index(lg)
                nf = {}
                for ch, (ts_arr, names) in per.items():
                    if ts_arr.size:
                        nf[ch] = names[int(abs(ts_arr - times[t]).argmin())]
                needed[t] = {"log": lg, "ts_us": times[t], "frames_filename_jpg": nf}
        rec.update(status="OK" if n_img == len(toks) else ("PARTIAL" if n_img else "BLOCKED_NO_FRAMES"), n_tokens_with_frames=n_img, frames_by_root=by_root)
        per_log[lg] = rec
    res = {"n_tokens": len(tok_log), "n_logs": len(by_log), "image_roots_present": sorted(fr), "per_log": per_log, "n_tokens_ok": len(ok_tokens), "n_tokens_blocked": len(blocked),
           "blocked_reasons": {r: sum(1 for b in blocked.values() if b["reason"].split(":")[0] == r) for r in {b["reason"].split(":")[0] for b in blocked.values()}}}
    json.dump(res, open(os.path.join(a.out, "audit.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"rule": "SR-TTC S0 batch plan: tokens of the batch's logs that have a DB and four resolvable camera frames", "tokens": sorted(ok_tokens), "token_log": token_log,
               "token_root": token_root, "token_ts_us": token_ts, "db": {lg: dbs[lg] for lg in sorted(set(token_log.values()))}}, open(os.path.join(a.out, "batch_tokens.json"), "w", encoding="utf-8"), indent=0)
    json.dump({"n_tokens": len(needed), "tokens": needed}, open(os.path.join(a.out, "needed_frames.json"), "w", encoding="utf-8"), indent=0)
    json.dump({"logs": needed_dbs}, open(os.path.join(a.out, "needed_dbs.json"), "w", encoding="utf-8"), indent=0)
    json.dump(blocked, open(os.path.join(a.out, "blocked_tokens.json"), "w", encoding="utf-8"), indent=0)
    print(json.dumps({k: res[k] for k in ("n_tokens", "n_logs", "n_tokens_ok", "n_tokens_blocked", "blocked_reasons")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
