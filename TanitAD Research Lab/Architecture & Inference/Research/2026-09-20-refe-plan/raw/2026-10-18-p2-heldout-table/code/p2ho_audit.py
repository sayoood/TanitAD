#!/usr/bin/env python3
"""P2 held-out table, plan + availability audit.  Pure bookkeeping: no GPU, no model.

Builds the token plan (one row per distinct (log, token) of the 3,137 banked held-out sets; P2's fan depends on the token only, so the rank-0 and the
goal-augmented rank-1 set of one token collapse to one P2 row) and audits, per token, the two things a P2 forward + the PDM scorer need:
  * DB      the nuPlan DB of its log (`navtrain_scenarios.index_dbs`: splits/*, splits/<split>/*) -> the scenario, the route, the map, the PDM scorer inputs
  * FRAMES  all four camera images of the token's lidar_pc frame, resolved by the planner's OWN `FrameResolver` against the candidate image roots
Writes plan/heldout_tokens.json (W3 format {tokens, token_log} for the tokens that pass both), plan/audit.json (everything, per log) and
plan/blocked_tokens.json (the rest, with the reason).  Runs in the driverl-eval interpreter (needs numpy only + the tree's planner for FrameResolver).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe code/p2ho_audit.py --out plan --images-root D:/.../frames [--images-root ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

LABELS = "D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm"
DB_ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
EXTRA_DB_DIR = os.environ.get("P2HO_EXTRA_DB_DIR", "C:/Users/Admin/p2heldout_work/dbs")
TREE = "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DEFAULT_IMAGE_ROOTS = ["C:/Users/Admin/p2heldout_work/frames_heldout",          # where a targeted fetch of the missing frames would be placed (empty today)
                       "D:/Projects/TanitAD/data/refe_proxy/pixels",            # M6 proxy extract of the pod's navtrain_pixels: original 4-camera jpgs, a stratified 10k-frame sample
                       "D:/Projects/TanitAD/data/refe_m5/pixels"]               # M5 dry-run extract (loose jpgs)


def index_dbs(db_root=DB_ROOT):
    out = {}
    for s in ("mini", "test", "trainval"):
        for p in glob.glob(os.path.join(db_root, s, "*.db")):
            out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    for p in glob.glob(os.path.join(db_root, "*.db")):
        out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    for p in glob.glob(os.path.join(EXTRA_DB_DIR, "*.db")):                    # where a targeted fetch of the 11 missing DBs would be placed (empty today)
        out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    return out


def heldout_sets():
    sets = []
    for p in sorted(glob.glob(f"{LABELS}/pdm_onpolicy_*.jsonl")):
        for ln in open(p, encoding="utf-8"):
            k = json.loads(ln)["key"]
            sets.append((str(k[0]), str(k[1]), int(k[2]), int(k[3])))
    return sets


def token_times(db, toks):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        want = set(toks)
        return {t: int(ts) for t, ts in con.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc") if t in want}
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-root", action="append", default=None)
    ap.add_argument("--skip-images", action="store_true", help="DB audit only")
    a = ap.parse_args()
    roots = a.images_root or DEFAULT_IMAGE_ROOTS
    os.makedirs(a.out, exist_ok=True)
    sets = heldout_sets()
    by_tok = defaultdict(list)
    for lg, t, st, rk in sets:
        by_tok[(lg, t)].append((st, rk))
    logs = sorted({lg for lg, _ in by_tok})
    dbs = index_dbs()
    res = {"n_sets": len(sets), "n_tokens": len(by_tok), "n_logs": len(logs), "image_roots_tried": roots, "per_log": {}}
    fr = None
    if not a.skip_images:
        sys.path.insert(0, f"{TREE}/refe")
        import planner as PL
        fr = {r: PL.FrameResolver(db_dir="", images_root=r) for r in roots if os.path.isdir(r)}
        res["image_roots_present"] = sorted(fr)
    blocked, ok_tokens, token_log, token_root, token_ts, needed = {}, [], {}, {}, {}, {}
    for lg in logs:
        toks = sorted(t for (l, t) in by_tok if l == lg)
        nsets = sum(len(by_tok[(lg, t)]) for t in toks)
        rec = {"n_tokens": len(toks), "n_sets": nsets, "db": dbs.get(lg)}
        if lg not in dbs:
            rec["status"] = "BLOCKED_NO_DB"
            for t in toks:
                blocked[t] = {"log": lg, "reason": "no nuPlan DB for this log on the dev box (splits/*, splits/trainval/*)"}
            res["per_log"][lg] = rec
            continue
        times = token_times(dbs[lg], toks)
        missing_ts = [t for t in toks if t not in times]
        rec["tokens_not_in_db"] = len(missing_ts)
        n_img = 0
        by_root = {}
        for t in toks:
            if t not in times:
                blocked[t] = {"log": lg, "reason": "token not in the DB's lidar_pc"}
                continue
            got, why = None, None
            if fr is not None:
                for r, res_ in fr.items():
                    res_.db_dir = os.path.dirname(dbs[lg])
                    paths = res_.resolve(lg, times[t])
                    if paths:
                        got = (r, paths)
                        break
                    why = res_.miss_reason
            if fr is None or got:
                n_img += 1
                ok_tokens.append(t)
                token_log[t] = lg
                token_ts[t] = times[t]
                if got:
                    token_root[t] = got[0]
                    by_root[got[0]] = by_root.get(got[0], 0) + 1
            else:
                blocked[t] = {"log": lg, "reason": f"no camera frames: {why}"}
                if fr is not None and fr:
                    r0 = next(iter(fr.values()))
                    r0.db_dir = os.path.dirname(dbs[lg])
                    per = r0._index(lg)
                    nf = {}
                    for ch, (ts_arr, names) in per.items():
                        if ts_arr.size:
                            nf[ch] = names[int(abs(ts_arr - times[t]).argmin())]
                    needed[t] = {"log": lg, "ts_us": times[t], "frames_filename_jpg": nf}
        rec.update(status="OK" if (n_img == len(toks) or fr is None) else ("PARTIAL" if n_img else "BLOCKED_NO_FRAMES"), n_tokens_with_frames=n_img,
                   frames_by_root=by_root)
        res["per_log"][lg] = rec
    res.update(n_tokens_ok=len(ok_tokens), n_tokens_blocked=len(blocked),
               n_logs_with_db=sum(1 for v in res["per_log"].values() if v["db"]),
               n_sets_with_db=sum(v["n_sets"] for v in res["per_log"].values() if v["db"]),
               n_tokens_with_db=sum(v["n_tokens"] for v in res["per_log"].values() if v["db"]),
               blocked_reasons={r: sum(1 for b in blocked.values() if b["reason"].split(":")[0] == r) for r in {b["reason"].split(":")[0] for b in blocked.values()}})
    json.dump(res, open(os.path.join(a.out, "audit.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"rule": "P2 held-out plan: distinct (log, token) of the 3,137 banked held-out sets whose log has a DB and whose four camera frames resolve",
               "tokens": sorted(ok_tokens), "token_log": token_log, "token_root": token_root, "token_ts_us": token_ts,
               "db": {lg: dbs[lg] for lg in sorted(set(token_log.values()))}}, open(os.path.join(a.out, "heldout_tokens.json"), "w", encoding="utf-8"), indent=0)
    json.dump(blocked, open(os.path.join(a.out, "blocked_tokens.json"), "w", encoding="utf-8"), indent=0)
    json.dump({"what": "tokens whose DB is present but whose four camera frames are not: the nearest-timestamp filename_jpg per camera, as the planner's FrameResolver would ask for it",
               "n_tokens": len(needed), "tokens": needed}, open(os.path.join(a.out, "needed_frames.json"), "w", encoding="utf-8"), indent=0)
    print(json.dumps({k: res[k] for k in ("n_sets", "n_tokens", "n_logs", "n_logs_with_db", "n_sets_with_db", "n_tokens_with_db", "n_tokens_ok", "n_tokens_blocked",
                                          "blocked_reasons")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
