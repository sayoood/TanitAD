#!/usr/bin/env python3
"""P6 label job helpers (pure python + numpy; no GPU, no torch).  Subcommands:

  draw        deterministic scene draw: held-out logs, FIT-extra logs, train scenes  -> key TSVs, log lists, draw.json
  check-sets  the dumped onpolicy sets of a dir equal a key TSV exactly (one ckpt_step, no foreign key)
  pack        FIT sets: proposal geometry + P6 logits + PDM labels in ONE npz (a stacker needs no images)
  manifest    counts + sha256 of every file the launch gate's data pin reads; refuses on any inconsistency

The bank is `targets_rank*.jsonl` (TargetBank's layout).  Only RANK-0 rows are scenes: a rank-1 twin shares its frame, and its
PDM label would use the scenario route (SPEC_P6 D7), so twins are not dumped by this job.
Exit codes: 0 ok, 2 refused (message printed as `P6TOOL_REFUSED <reason>`).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

SCHEMA = "p6-labels/1"
PDM_NAMES = ["nc", "dac", "ep", "ttc", "c", "ddc", "pdms"]
HEAD_RE = re.compile(r'"log_name": "([^"]+)", "token": "([^"]*)", "step": (-?\d+), "rank": (-?\d+), "ckpt_step": (-?\d+)')


class Refused(Exception):
    pass


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(l + "\n" for l in lines))


# ------------------------------------------------------------------------------------------------------------------ draw
def read_bank_scenes(targets):
    """-> {log: sorted [token,...]} over rank-0 rows of targets_rank*.jsonl.  A (log, token) listed twice is an error (a duplicated merge)."""
    files = sorted(glob.glob(os.path.join(targets, "targets_rank*.jsonl")))
    if not files:
        raise Refused(f"no targets_rank*.jsonl under {targets}")
    by, seen = {}, set()
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if int(r.get("rank", 0)) != 0:
                    continue
                k = (r["log_name"], r["token"])
                if k in seen:
                    raise Refused(f"bank holds rank-0 scene {k} twice")
                seen.add(k)
                by.setdefault(k[0], []).append(k[1])
    return {l: sorted(t) for l, t in by.items()}


def do_draw(by_log, heldout_logs, seed, n_train, fit_target, expect_heldout, min_train):
    """Pure function.  -> dict(heldout, fit_extra, train: lists of (token, log); logs: lists)."""
    ho = sorted(set(heldout_logs))
    absent = [l for l in ho if l not in by_log]
    if absent:
        raise Refused(f"held-out logs absent from the bank: {absent[:3]}")
    heldout = [(t, l) for l in ho for t in by_log[l]]
    if expect_heldout is not None and len(heldout) != expect_heldout:
        raise Refused(f"held-out sets {len(heldout)} != expected {expect_heldout}")
    others = sorted(l for l in by_log if l not in set(ho))
    rng = random.Random(int(seed))
    order = list(others)
    rng.shuffle(order)
    fit_logs, n_fit = [], len(heldout)
    for l in order:                                    # whole logs, in draw order, until the FIT total reaches the target
        if n_fit >= fit_target:
            break
        fit_logs.append(l)
        n_fit += len(by_log[l])
    fit_set = set(fit_logs)
    fit_extra = [(t, l) for l in sorted(fit_logs) for t in by_log[l]]
    train_logs_all = sorted(l for l in others if l not in fit_set)
    pool = [(t, l) for l in train_logs_all for t in by_log[l]]
    rng2 = random.Random(int(seed) + 1)
    rng2.shuffle(pool)                                 # the draw order; the trainer re-shuffles with its own seed
    n_avail = len(pool)
    n = n_avail if n_train == "all" else int(n_train)
    if n > n_avail:
        raise Refused(f"N_TRAIN {n} > the {n_avail} scenes outside the FIT logs (rank-0 only; 189,888 is the pre-3C tuple count, not reachable here)")
    if n < min_train:
        raise Refused(f"N_TRAIN {n} < {min_train}: 400 updates x 64 = 25,600 distinct sets per arm, plus losses to skips")
    train = pool[:n]
    train_logs = sorted({l for _, l in train})
    return {"heldout": heldout, "fit_extra": fit_extra, "train": train, "heldout_logs": ho, "fit_extra_logs": sorted(fit_logs),
            "train_logs": train_logs, "n_available_train": n_avail}


def cmd_draw(a):
    by_log = read_bank_scenes(a.targets)
    ho = open(a.heldout_logs, encoding="utf-8").read().split()
    d = do_draw(by_log, ho, a.seed, a.n_train if a.n_train == "all" else int(a.n_train), a.fit_target,
                None if a.expect_heldout < 0 else a.expect_heldout, a.min_train)
    out = a.out
    files = {}
    for name in ("heldout", "fit_extra", "train"):
        p = os.path.join(out, "keys", f"{name}.tsv")
        write_lines(p, ["token\tlog"] + [f"{t}\t{l}" for t, l in d[name]])
        files[f"keys/{name}.tsv"] = p
    fit_logs = sorted(set(d["heldout_logs"]) | set(d["fit_extra_logs"]))
    for name, logs in (("heldout_logs", d["heldout_logs"]), ("fit_extra_logs", d["fit_extra_logs"]), ("fit_logs", fit_logs), ("train_logs", d["train_logs"])):
        p = os.path.join(out, "logs", f"{name}.txt")
        write_lines(p, logs)
        files[f"logs/{name}.txt"] = p
    if set(fit_logs) & set(d["train_logs"]):
        raise Refused("FIT and train logs overlap")           # unreachable by construction; asserted anyway
    rep = {"schema": SCHEMA, "seed": int(a.seed), "n_train_request": a.n_train, "fit_target": a.fit_target,
           "n_heldout": len(d["heldout"]), "n_fit_extra": len(d["fit_extra"]), "n_fit": len(d["heldout"]) + len(d["fit_extra"]),
           "n_train": len(d["train"]), "n_available_train": d["n_available_train"],
           "n_heldout_logs": len(d["heldout_logs"]), "n_fit_extra_logs": len(d["fit_extra_logs"]), "n_train_logs": len(d["train_logs"]),
           "ranks": [0], "files": {k: {"path": os.path.abspath(v), "sha256": sha256_file(v)} for k, v in files.items()},
           "heldout_logs_source_sha256": sha256_file(a.heldout_logs)}
    with open(os.path.join(out, "draw.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(f"P6TOOL_DRAW heldout {rep['n_heldout']} fit_extra {rep['n_fit_extra']} train {rep['n_train']} (of {rep['n_available_train']})")
    return 0


# ------------------------------------------------------------------------------------------------------------------ check-sets
def read_tsv_keys(path):
    out = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for ln in f:
            ln = ln.rstrip("\r\n")
            if ln:
                t, l = ln.split("\t")[:2]
                out.add((l, t))
    return out


def set_keys(dirpath, pattern="onpolicy_rlfan_*.jsonl"):
    """-> ({(log, token, step, rank)}, {ckpt_step}, n_lines).  Key fields come first in a lean set; fall back to json for any other line."""
    keys, steps, n = set(), set(), 0
    for fp in sorted(glob.glob(os.path.join(dirpath, pattern))):
        with open(fp, encoding="utf-8", newline="") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                m = HEAD_RE.search(ln[:700])
                try:
                    if m:
                        k = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
                        c = int(m.group(5))
                    else:
                        r = json.loads(ln)
                        if r.get("kind") != "onpolicy_set":
                            continue
                        k, c = (r["log_name"], r["token"], int(r["step"]), int(r.get("rank", 0))), int(r["ckpt_step"])
                except json.JSONDecodeError:
                    continue                                   # a torn tail line is not a set
                n += 1
                keys.add(k)
                steps.add(c)
    return keys, steps, n


def cmd_check_sets(a):
    want = read_tsv_keys(a.keys)
    keys, steps, n = set_keys(a.dir)
    got = {(l, t) for l, t, _, rk in keys}
    bad_rank = [k for k in keys if k[3] != 0]
    problems = []
    if got != want:
        problems.append(f"keys differ: missing {len(want - got)} foreign {len(got - want)}")
    if len(keys) != n:
        problems.append(f"duplicate lines: {n} lines for {len(keys)} keys")
    if len(steps) != 1:
        problems.append(f"ckpt_step values {sorted(steps)}")
    if bad_rank:
        problems.append("non-zero rank sets present")
    if a.expect is not None and len(keys) != a.expect:
        problems.append(f"{len(keys)} sets, expected {a.expect}")
    if problems:
        print("P6TOOL_REFUSED check-sets: " + "; ".join(problems))
        return 2
    print(f"P6TOOL_SETS_OK {len(keys)}")
    return 0


# ------------------------------------------------------------------------------------------------------------------ pack
def read_logits(dirs):
    """fit_logits_*.jsonl side files -> {key: [[..]*K]*M}.  A later line replaces an earlier one (a resumed batch is recomputed)."""
    out = {}
    for d in dirs:
        for fp in sorted(glob.glob(os.path.join(d, "fit_logits_*.jsonl"))):
            with open(fp, encoding="utf-8", newline="") as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    out[tuple(r["key"])] = r["logits"]
    return out


def read_pdm(dirs):
    out = {}
    for d in dirs:
        for fp in sorted(glob.glob(os.path.join(d, "pdm_*.jsonl"))):
            with open(fp, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                        out[tuple(r["key"])] = r
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
    return out


def cmd_pack(a):
    import numpy as np
    sets_dirs = a.sets
    sets = {}
    for i, d in enumerate(sets_dirs):
        for fp in sorted(glob.glob(os.path.join(d, "onpolicy_rlfan_*.jsonl"))):
            with open(fp, encoding="utf-8", newline="") as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if r.get("kind") != "onpolicy_set":
                        continue
                    sets[(r["log_name"], r["token"], int(r["step"]), int(r.get("rank", 0)))] = (r, i)
    if not sets:
        print("P6TOOL_REFUSED pack: no sets")
        return 2
    logits = read_logits(sets_dirs)
    pdm = read_pdm(a.pdm)
    keys = sorted(sets)
    missing = [k for k in keys if k not in logits]
    if missing:
        print(f"P6TOOL_REFUSED pack: {len(missing)} FIT sets have no saved logits, e.g. {missing[0]}")
        return 2
    r0 = sets[keys[0]][0]
    M, T = len(r0["traj"]), len(r0["traj"][0])
    K = len(logits[keys[0]][0])
    N = len(keys)
    traj = np.empty((N, M, T, 2), np.float32)
    yaw = np.empty((N, M, T), np.float32)
    lg = np.empty((N, M, K), np.float32)
    lab = np.full((N, M, len(PDM_NAMES)), np.nan, np.float32)
    has = np.zeros(N, bool)
    is_ho = np.zeros(N, bool)
    for j, k in enumerate(keys):
        r, di = sets[k]
        traj[j] = np.asarray(r["traj"], np.float32)
        yaw[j] = np.asarray(r["yaw"], np.float32)
        lj = np.asarray(logits[k], np.float32)
        if lj.shape != (M, K):
            print(f"P6TOOL_REFUSED pack: logits shape {lj.shape} for {k}, need {(M, K)}")
            return 2
        lg[j] = lj
        is_ho[j] = di == 0
        p = pdm.get(k)
        if p is not None:
            lab[j] = np.stack([np.asarray(p[n], np.float32) for n in PDM_NAMES], -1)
            has[j] = True
    # guards against the zero-filled-array family: every row must carry real values
    if not (np.isfinite(lg).all() and np.isfinite(traj).all() and np.isfinite(yaw).all()):
        print("P6TOOL_REFUSED pack: non-finite geometry or logits")
        return 2
    if (np.abs(lg).reshape(N, -1).max(1) == 0).any():
        print("P6TOOL_REFUSED pack: a set has all-zero logits")
        return 2
    if (np.abs(traj).reshape(N, -1).max(1) == 0).any():
        print("P6TOOL_REFUSED pack: a set has all-zero geometry")
        return 2
    if a.require_pdm_share is not None and has.mean() < a.require_pdm_share:
        print(f"P6TOOL_REFUSED pack: only {has.sum()} of {N} sets carry PDM labels")
        return 2
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, log=np.array([k[0] for k in keys]), token=np.array([k[1] for k in keys]), step=np.array([k[2] for k in keys]),
             rank=np.array([k[3] for k in keys]), ckpt_step=np.array([sets[k][0]["ckpt_step"] for k in keys]),
             traj_xy=traj, yaw=yaw, logits=lg, pdm=lab, pdm_names=np.array(PDM_NAMES), has_pdm=has, is_heldout=is_ho,
             logit_space=np.array("raw head output, bf16 autocast, as P6 emits it"))
    os.replace(tmp, a.out)
    print(f"P6TOOL_PACK_OK {N} sets, {int(has.sum())} with PDM labels, logits {lg.shape}")
    return 0


# ------------------------------------------------------------------------------------------------------------------ manifest
def cmd_manifest(a):
    root = a.root
    draw = json.load(open(os.path.join(root, "draw.json"), encoding="utf-8"))
    problems = []
    parts = {"heldout": ("heldout_sets", "heldout_pdm", draw["n_heldout"]), "fit": ("fit_sets", "fit_pdm", draw["n_fit_extra"]),
             "train": ("train_sets", "train_pdm", draw["n_train"])}
    rep = {}
    fit_logs = set(open(os.path.join(root, "logs", "fit_logs.txt"), encoding="utf-8").read().split())
    train_logs = set(open(os.path.join(root, "logs", "train_logs.txt"), encoding="utf-8").read().split())
    ho_logs = set(open(os.path.join(root, "logs", "heldout_logs.txt"), encoding="utf-8").read().split())
    if fit_logs & train_logs:
        problems.append("FIT logs overlap the train logs")
    files = {}
    for name, (sd, pd, n_exp) in parts.items():
        keys, steps, _ = set_keys(os.path.join(root, sd))
        r = json.load(open(os.path.join(root, "reports", f"{name}.json"), encoding="utf-8"))
        if r.get("verdict") != "COMPLETE":
            problems.append(f"{name}: label report {r.get('verdict')} ({r.get('reason')})")
        if len(keys) != n_exp:
            problems.append(f"{name}: {len(keys)} sets on disk, drawn {n_exp}")
        logs_here = {k[0] for k in keys}
        if name == "train" and logs_here & fit_logs:
            problems.append("train sets include FIT logs")
        if name == "heldout" and logs_here != ho_logs:
            problems.append("held-out sets do not cover exactly the 24 logs")
        if name == "fit" and logs_here & train_logs:
            problems.append("FIT-extra sets include train logs")
        d = json.loads(json.dumps(r))
        d.update({"sets": len(keys), "ckpt_steps": sorted(steps), "pdm_rows_credited": r["rows_credited"],
                  "skipped_5s_guard": (r.get("done_marker") or [0, 0, 0])[1], "failed": (r.get("done_marker") or [0, 0, 0])[2]})
        rep[name] = d
        for sub in (sd, pd):
            for fp in sorted(glob.glob(os.path.join(root, sub, "*.jsonl"))):
                files[os.path.relpath(fp, root).replace("\\", "/")] = {"bytes": os.path.getsize(fp), "sha256": sha256_file(fp)}
    pack = os.path.join(root, "fit_pack", "fit_pack.npz")
    if not os.path.exists(pack):
        problems.append("fit_pack.npz missing")
    else:
        files["fit_pack/fit_pack.npz"] = {"bytes": os.path.getsize(pack), "sha256": sha256_file(pack)}
    if problems:
        print("P6TOOL_REFUSED manifest: " + "; ".join(problems))
        return 2
    h, f_, t = rep["heldout"], rep["fit"], rep["train"]
    man = {"schema": SCHEMA, "model_sha256": a.model_sha256, "ckpt_step": t["ckpt_steps"][0] if t["ckpt_steps"] else None,
           "targets": a.targets, "draw": draw, "parts": rep, "files": files,
           "log_split": {"fit_logs_sha256": draw["files"]["logs/fit_logs.txt"]["sha256"], "train_logs_sha256": draw["files"]["logs/train_logs.txt"]["sha256"],
                         "heldout_logs_sha256": draw["files"]["logs/heldout_logs.txt"]["sha256"],
                         "fit_extra_logs_sha256": draw["files"]["logs/fit_extra_logs.txt"]["sha256"],
                         "n_fit_logs": len(fit_logs), "n_train_logs": len(train_logs),
                         "rule": "FIT logs (24 held-out + whole extra logs drawn at the seed) are never in the arms' training sets; fit selection rules, calibrators, stackers on FIT only"},
           # the numbers a launch-gate profile's data_expect reads (names follow SFT4R_DATA_EXPECT; the profile is new, so this mapping is UNVERIFIED until it is written)
           "data_expect": {"train_sets": t["sets"], "train_sets_in_bank": t["sets"], "heldout_sets": h["sets"], "heldout_in_bank": h["sets"],
                           "heldout_logs": len(ho_logs), "train_rows_in_heldout_logs": 0, "pdm_train": t["pdm_rows_credited"],
                           "pdm_heldout": h["pdm_rows_credited"], "pdm_fit_extra": f_["pdm_rows_credited"],
                           "train_skipped_5s_guard": t["skipped_5s_guard"], "heldout_skipped_5s_guard": h["skipped_5s_guard"]},
           "tool_sha256": {n: sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), n))
                           for n in ("p6_labels_tool.py", "p6_fan_dump.py", "p6_labels_chain.sh") if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), n))}}
    tmp = os.path.join(root, "labels_manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(root, "labels_manifest.json"))
    print(f"P6TOOL_MANIFEST_OK train {t['sets']} heldout {h['sets']} fit_extra {f_['sets']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("draw")
    d.add_argument("--targets", required=True)
    d.add_argument("--heldout-logs", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=20261109)
    d.add_argument("--n-train", default="60000", help="an integer or 'all'")
    d.add_argument("--fit-target", type=int, default=8000)
    d.add_argument("--expect-heldout", type=int, default=3137, help="-1 = no check (tests only)")
    d.add_argument("--min-train", type=int, default=30000)
    c = sp.add_parser("check-sets")
    c.add_argument("--dir", required=True)
    c.add_argument("--keys", required=True)
    c.add_argument("--expect", type=int, default=None)
    p = sp.add_parser("pack")
    p.add_argument("--sets", nargs="+", required=True, help="held-out sets dir FIRST, then the FIT-extra dir")
    p.add_argument("--pdm", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require-pdm-share", type=float, default=0.95)
    m = sp.add_parser("manifest")
    m.add_argument("--root", required=True)
    m.add_argument("--model-sha256", required=True)
    m.add_argument("--targets", default="")
    a = ap.parse_args(argv)
    try:
        return {"draw": cmd_draw, "check-sets": cmd_check_sets, "pack": cmd_pack, "manifest": cmd_manifest}[a.cmd](a)
    except Refused as e:
        print(f"P6TOOL_REFUSED {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
