#!/usr/bin/env python3
"""RL-S1: a deterministic, LOG-disjoint RL-train / RL-val partition of a pool of navtrain scenes.

A "scene" is a navtrain token with its log (NAVSIM tokens are unique, so a token belongs to exactly one log).  The partition
  * drops every scene whose token is in navtest, whose log is in navtest, whose log is one of the SFT held-out logs, whose token the BASE's
    scorer trained on (the optional trained-keys list), and every scene with no known log (it cannot be placed on a side);
  * orders the remaining LOGS by sha256(seed | log) -- independent of the input order, so the same inputs always give the same bytes;
  * RL-val: walks that order and takes at most `cap = max(1, floor(n_val / min_val_logs))` scenes per log (hash order inside the log) until n_val
    scenes are held, so RL-val spans >= min_val_logs logs by construction (3,000 scenes, 60 logs -> cap 50);
  * RL-train: continues down the same log order with WHOLE logs (hash order inside the log) until n_train scenes are held.
A log is therefore never on both sides; scenes of a val log beyond the cap, and scenes beyond n_train, are left UNUSED (counted), never moved.
If the pool cannot fill n_val or n_train the tool says so and exits 3 without writing (--allow-short writes what exists and flags it).

    python refe/rl_partition.py --pool raw/rl_pool/navtrain_token_table.tsv.gz --navtest <navtest_full_tokens.json> \\
        --heldout-logs splits/sft_heldout_logs.txt [--trained-keys raw/rl_pool/served_keys.tsv.gz] \\
        --n-train 48000 --n-val 3000 --min-val-logs 60 --seed 20261012 --out <dir>
Writes <out>/rl_train_scenes.tsv, rl_val_scenes.tsv (token<TAB>log) and rl_partition_manifest.json (input / output sha256, counts, tallies).
`verify()` re-checks every invariant on the two lists and runs before anything is written (exit 5 on a violation).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sys

TOOL_VERSION = "rl_partition/1"


def _h(seed, kind, key):
    """The one hash behind every ordering.  Seeded, input-order free."""
    return hashlib.sha256(f"{seed}|{kind}|{key}".encode("utf-8")).hexdigest()


def partition(scenes, *, navtest_tokens=(), navtest_logs=(), heldout_logs=(), trained_tokens=(), n_train, n_val, min_val_logs, seed):
    """scenes: iterable of (token, log).  Returns a dict: train, val (lists of (token, log)), tally, unused counts, shortfall flags."""
    navtest_tokens, navtest_logs = set(navtest_tokens), set(navtest_logs)
    heldout_logs, trained_tokens = set(heldout_logs), set(trained_tokens)
    if n_val < 1 or min_val_logs < 1 or n_train < 0:
        raise ValueError("n_val and min_val_logs must be >= 1 and n_train >= 0")
    tally = collections.Counter()
    pool: dict = {}
    seen: dict = {}
    for token, log in scenes:
        if token in seen and seen[token] != log:
            raise ValueError(f"token {token} appears under two logs ({seen[token]!r}, {log!r}): a token belongs to exactly one log")
        if token in seen:
            tally["duplicate_row"] += 1
            continue
        seen[token] = log
        tally["input_scenes"] += 1
        if not log:
            tally["log_unknown"] += 1
            continue
        if token in navtest_tokens:
            tally["navtest_token"] += 1
            continue
        if log in navtest_logs:
            tally["navtest_log"] += 1
            continue
        if log in heldout_logs:
            tally["sft_heldout_log"] += 1
            continue
        if token in trained_tokens:
            tally["trained_token"] += 1
            continue
        pool.setdefault(log, []).append(token)
    tally["eligible_scenes"] = sum(len(v) for v in pool.values())
    tally["eligible_logs"] = len(pool)
    logs = sorted(pool, key=lambda lg: (_h(seed, "log", lg), lg))
    cap = max(1, n_val // min_val_logs)

    def in_scene_order(lg):
        return sorted(pool[lg], key=lambda t: (_h(seed, "scene", t), t))

    val, val_logs, i = [], [], 0
    while i < len(logs) and len(val) < n_val:
        lg = logs[i]
        i += 1
        take = in_scene_order(lg)[:cap][: n_val - len(val)]
        val += [(t, lg) for t in take]
        val_logs.append(lg)
    val_unused = sum(len(pool[lg]) for lg in val_logs) - len(val)
    train, train_logs = [], []
    for lg in logs[i:]:
        if len(train) >= n_train:
            break
        take = in_scene_order(lg)[: n_train - len(train)]
        train += [(t, lg) for t in take]
        train_logs.append(lg)
    train_unused = tally["eligible_scenes"] - len(val) - len(train) - val_unused
    return {"train": train, "val": val, "tally": dict(tally), "cap_per_val_log": cap, "val_logs": len(val_logs), "train_logs": len(train_logs),
            "val_unused_scenes_of_val_logs": val_unused, "unused_scenes_not_assigned": train_unused,
            "shortfall_val": len(val) < n_val, "shortfall_train": len(train) < n_train}


def verify(train, val, *, navtest_tokens=(), navtest_logs=(), heldout_logs=(), trained_tokens=(), min_val_logs=1, n_val=None, n_train=None):
    """Independent re-check of the invariants on the two lists.  Returns a list of violation codes (empty = clean)."""
    navtest_tokens, navtest_logs = set(navtest_tokens), set(navtest_logs)
    heldout_logs, trained_tokens = set(heldout_logs), set(trained_tokens)
    bad = []
    tl, vl = {lg for _, lg in train}, {lg for _, lg in val}
    tt, vt = [t for t, _ in train], [t for t, _ in val]
    if tl & vl:
        bad.append("log_on_both_sides")
    if set(tt) & set(vt):
        bad.append("token_on_both_sides")
    if len(set(tt)) != len(tt) or len(set(vt)) != len(vt):
        bad.append("duplicate_token")
    allsc = list(train) + list(val)
    if any(t in navtest_tokens for t, _ in allsc):
        bad.append("navtest_token_admitted")
    if any(lg in navtest_logs for _, lg in allsc):
        bad.append("navtest_log_admitted")
    if any(lg in heldout_logs for _, lg in allsc):
        bad.append("heldout_log_admitted")
    if any(t in trained_tokens for t, _ in allsc):
        bad.append("trained_token_admitted")
    if any(not lg for _, lg in allsc):
        bad.append("scene_without_log")
    if val and len(vl) < min_val_logs:
        bad.append("val_spans_too_few_logs")
    if n_val is not None and len(val) > n_val:
        bad.append("val_over_n")
    if n_train is not None and len(train) > n_train:
        bad.append("train_over_n")
    return bad


# ---------------------------------------------------------------------------------------------------------------- I/O
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else open(path, encoding="utf-8", newline="")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_columns(path, *cols):
    """TSV (optionally .gz) with a header; returns the requested columns as lists of str."""
    with _open(path) as fh:
        head = fh.readline().rstrip("\r\n").split("\t")
        ix = [head.index(c) for c in cols]
        out = [[] for _ in cols]
        for ln in fh:
            f = ln.rstrip("\r\n").split("\t")
            for o, j in zip(out, ix):
                o.append(f[j] if j < len(f) else "")
    return out


def read_pool(path):
    toks, logs = read_columns(path, "token", "log")
    return list(zip(toks, logs))


def read_trained_tokens(path):
    if path.endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        return set(d["tokens"] if isinstance(d, dict) else d)
    if path.endswith(".txt"):
        return set(open(path, encoding="utf-8").read().split())
    return set(read_columns(path, "token")[0])


def read_navtest(path):
    d = json.load(open(path, encoding="utf-8"))
    toks = set(d["tokens"])
    tl = d.get("token_log") or {}
    return toks, (set(tl.values()) if isinstance(tl, dict) else set())


def _write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("token\tlog\n")
        for t, lg in rows:
            f.write(f"{t}\t{lg}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="TSV(.gz) with token and log columns (raw/rl_pool/navtrain_token_table.tsv.gz)")
    ap.add_argument("--navtest", required=True, help="navtest tokens json: {tokens: [...], token_log: {...}}")
    ap.add_argument("--heldout-logs", required=True)
    ap.add_argument("--trained-keys", default=None, help="the BASE's trained-key list (TSV with a token column, .json, or .txt); omitted = none supplied")
    ap.add_argument("--n-train", type=int, default=48000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--min-val-logs", type=int, default=60)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-short", action="store_true")
    a = ap.parse_args(argv)
    scenes = read_pool(a.pool)
    nt_toks, nt_logs = read_navtest(a.navtest)
    ho = set(open(a.heldout_logs, encoding="utf-8").read().split())
    trained = read_trained_tokens(a.trained_keys) if a.trained_keys else set()
    kw = dict(navtest_tokens=nt_toks, navtest_logs=nt_logs, heldout_logs=ho, trained_tokens=trained)
    r = partition(scenes, n_train=a.n_train, n_val=a.n_val, min_val_logs=a.min_val_logs, seed=a.seed, **kw)
    bad = verify(r["train"], r["val"], min_val_logs=a.min_val_logs, n_val=a.n_val, n_train=a.n_train, **kw)
    if bad:
        print("ZZPARTITION_VIOLATION " + ",".join(bad), flush=True)
        return 5
    if (r["shortfall_val"] or r["shortfall_train"]) and not a.allow_short:
        print(f"ZZPARTITION_SHORT pool too small: val {len(r['val'])}/{a.n_val} ({r['val_logs']} logs), train {len(r['train'])}/{a.n_train}; "
              f"eligible {r['tally']['eligible_scenes']} scenes in {r['tally']['eligible_logs']} logs; tally {r['tally']}", flush=True)
        return 3
    os.makedirs(a.out, exist_ok=True)
    ptr, pva = os.path.join(a.out, "rl_train_scenes.tsv"), os.path.join(a.out, "rl_val_scenes.tsv")
    _write_tsv(ptr, r["train"])
    _write_tsv(pva, r["val"])
    params = {"n_train": a.n_train, "n_val": a.n_val, "min_val_logs": a.min_val_logs, "seed": a.seed, "cap_per_val_log": r["cap_per_val_log"]}
    man = {"tool": TOOL_VERSION,
           "inputs": {"pool": {"file": os.path.basename(a.pool), "sha256": sha256_file(a.pool), "scenes": len(scenes)},
                      "navtest": {"file": os.path.basename(a.navtest), "sha256": sha256_file(a.navtest), "tokens": len(nt_toks), "logs": len(nt_logs)},
                      "heldout_logs": {"file": os.path.basename(a.heldout_logs), "sha256": sha256_file(a.heldout_logs), "logs": len(ho)},
                      "trained_keys": ({"file": os.path.basename(a.trained_keys), "sha256": sha256_file(a.trained_keys), "tokens": len(trained)}
                                       if a.trained_keys else None)},
           "trained_exclusion_applied": bool(a.trained_keys), "params": params, "tally": r["tally"],
           "train": {"scenes": len(r["train"]), "logs": r["train_logs"], "file": os.path.basename(ptr), "sha256": sha256_file(ptr)},
           "val": {"scenes": len(r["val"]), "logs": r["val_logs"], "file": os.path.basename(pva), "sha256": sha256_file(pva)},
           "val_unused_scenes_of_val_logs": r["val_unused_scenes_of_val_logs"], "unused_scenes_not_assigned": r["unused_scenes_not_assigned"],
           "shortfall": {"val": r["shortfall_val"], "train": r["shortfall_train"]}, "verify": "clean"}
    man["partition_id"] = hashlib.sha256(json.dumps([man["train"]["sha256"], man["val"]["sha256"], params], sort_keys=True).encode()).hexdigest()
    with open(os.path.join(a.out, "rl_partition_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"ZZPARTITION_OK train {len(r['train'])} ({r['train_logs']} logs) val {len(r['val'])} ({r['val_logs']} logs) id {man['partition_id'][:12]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
