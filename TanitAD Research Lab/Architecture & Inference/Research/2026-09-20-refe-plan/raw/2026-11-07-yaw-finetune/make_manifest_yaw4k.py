#!/usr/bin/env python3
"""S1: derive manifest_yaw_4k.json from proxy_manifest_10k.json (YF FINETUNE_PLAN section 4).

SEEDED per-log STRATIFIED 4,000 frames (largest-remainder quota per log, per-log seeded shuffle, the 10k manifest's own rule), EVERY `onpolicy` entry STRIPPED (so
`proxy_cache --visual-for sets` stores NO 3.9 MB visual_ctx), logs split 80/20 TRAIN/VAL by a seeded hash. Refuses: a frame whose token is absent from the yaw table
(a missing token REFUSES), a log that is also in the A5 confirmation set (the TEST tokens come from navtest logs; disjointness is asserted, not assumed).
    python make_manifest_yaw4k.py --out manifest_yaw_4k.json [--n 4000 --seed 20261106]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaw_common as YC


def hkey(seed, log):
    return hashlib.sha256(f"{seed}|{log}".encode()).hexdigest()


def select(frames, n, seed):
    by = {}
    for f in frames:
        by.setdefault(f["log_name"], []).append(f)
    pool = len(frames)
    raw = {l: len(v) * n / pool for l, v in by.items()}
    q = {l: int(r) for l, r in raw.items()}
    rem = n - sum(q.values())
    for l in sorted(raw, key=lambda l: (-(raw[l] - q[l]), l))[:rem]:
        q[l] += 1
    out = []
    for l in sorted(by):
        fs = sorted(by[l], key=lambda f: f["token"])
        random.Random(f"{seed}:{l}").shuffle(fs)
        out += fs[:q[l]]
    return out, q


def split_logs(logs, seed, val_frac=0.2):
    order = sorted(logs, key=lambda l: hkey(seed, l))
    k = round(val_frac * len(order))
    return {l: ("VAL" if i < k else "TRAIN") for i, l in enumerate(order)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261106)
    a = ap.parse_args()
    src = json.load(open(YC.PROXY_MANIFEST, encoding="utf-8"))
    table, tmeta, tsha = YC.load_yaw_table(YC.TABLE_NAVTRAIN)
    sel, quota = select(src["frames"], a.n, a.seed)
    miss = [f["token"] for f in sel if f["token"] not in table]
    if miss:
        raise SystemExit(f"REFUSED: {len(miss)} selected token(s) absent from the yaw table, e.g. {miss[:3]}")
    a5 = json.load(open(YC.A5_TOKENS, encoding="utf-8"))
    a5_logs, a5_tok = set(a5["token_log"].values()), set(a5["tokens"])
    logs = sorted({f["log_name"] for f in sel})
    if set(logs) & a5_logs or {f["token"] for f in sel} & a5_tok:
        raise SystemExit("REFUSED: the selection overlaps the A5 confirmation (TEST) logs/tokens")
    sp = split_logs(logs, a.seed)
    frames = []
    for f in sorted(sel, key=lambda f: (f["log_name"], f["token"])):
        g = dict(f)
        g["ranks"] = {r: dict(v, onpolicy=[]) for r, v in f["ranks"].items()}
        g["yf2_split"] = sp[f["log_name"]]
        frames.append(g)
    out = {k: v for k, v in src.items() if k not in ("frames",)}
    out["status"] = "final"
    out["yf2"] = {"derived_from": str(YC.PROXY_MANIFEST), "derived_from_sha256": YC.sha256_file(YC.PROXY_MANIFEST), "n_frames": len(frames), "seed": a.seed,
                  "rule": "per-log stratified largest-remainder quota (proportional to the 10k manifest's per-log count), per-log seeded shuffle of token-sorted frames, onpolicy stripped",
                  "split_rule": "logs ordered by sha256(seed|log); first round(0.2 n_logs) = VAL, rest TRAIN",
                  "n_logs": len(logs), "n_train_logs": sum(v == "TRAIN" for v in sp.values()), "n_val_logs": sum(v == "VAL" for v in sp.values()),
                  "n_train_frames": sum(f["yf2_split"] == "TRAIN" for f in frames), "n_val_frames": sum(f["yf2_split"] == "VAL" for f in frames),
                  "yaw_table": str(YC.TABLE_NAVTRAIN), "yaw_table_sha256": tsha, "all_tokens_in_table": True,
                  "disjoint_from_a5_confirm_logs": True, "a5_logs": len(a5_logs)}
    out["frames"] = frames
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f)
    print(json.dumps(out["yf2"], indent=1))
    print("sha256", YC.sha256_file(a.out))
    print("ZZMANIFEST_OK")


if __name__ == "__main__":
    main()
