#!/usr/bin/env python3
"""fix_navtrain_sample.py -- fixes the D-R8-CKPT-RULE navtrain sample from a directory of DOWNLOADED shards + metadata (any python, CPU).

Applies the REGISTERED sample definition (GOALS_AND_CLAIMS ``D-R8-CKPT-RULE`` (1) + AMENDMENT 1 (a)/(d)) and nothing else:

    * the universe is the official navtrain scene filter (v1.1 tree 3e8291b, sha256 ``c37fea56...``);
    * logs are taken SORTED by ``log_name``;
    * a token is READABLE iff its OpenScene trainval metadata row exists (and is a window the devkit keeps) AND its 4 history frames x
      {CAM_L0, CAM_F0, CAM_R0} = 12 jpgs are present in the downloaded shards (KB3; AMENDMENT 1a -- "CAM_F0 history" in the registered
      text was too narrow);
    * the sample is ALL readable tokens of the first logs in that order, stopping at the FIRST log that takes the total to >= 2,000;
    * fewer than 1,000 readable tokens in total -> status ``UNDERPOWERED`` (the selection reads INCOMPLETE and EMA-48,852 holds).

The inventory is read from the shard archives' HEADERS (``.tgz`` streamed, nothing extracted) or from extracted directories.  NO image is
decoded and NO model, scorer or score is touched: availability is fixed BEFORE any score exists and is never revisited -- an existing
FIXED / UNDERPOWERED sample file is therefore NEVER overwritten with different content (exit 2).

Output ``NAVTRAIN_SAMPLE_TOKENS.json`` (the schema ``r8ns_navtrain_select_LAUNCH.sh`` gate G1 and ``select_ckpt_r8.py`` read):
``status, count, tokens (sorted), sha256_sorted_tokens, log_names (in sample order), token_log, per_log, map_locations, shards, ...``.

    python fix_navtrain_sample.py --filter-yaml navtrain.yaml --metadata-dir <navsim_logs/trainval> --shards-dir <dir with navtrain_*_N.tgz> \
        --out NAVTRAIN_SAMPLE_TOKENS.json [--verify-sha256-from raw_hf_tree_navsim.json] [--target 2000] [--min-tokens 1000]

Marker on stdout: ``ZZSAMPLE status=<FIXED|UNDERPOWERED> count=<n> logs=<k> sha256=<hex>`` (an opaque token, not an exit code).
``R8FIX_MUTATION`` (tests only) reintroduces one real defect class; ``main`` refuses to run while it is set.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import navtrain_common as N

MUTATION = os.environ.get("R8FIX_MUTATION", "")
MUTATIONS = ("no_sort_logs", "stop_strict", "f0_only", "t0_frame_only", "truncate_to_target", "hash_unsorted", "underpowered_off",
             "no_overwrite_guard", "keep_unwindowed")
RULE = ("D-R8-CKPT-RULE (1)+AMENDMENT 1: logs sorted by log_name; all readable tokens (metadata row AND 4 history frames x "
        "{CAM_L0,CAM_F0,CAM_R0}) of the first logs; stop at the first log reaching >= 2,000; < 1,000 -> UNDERPOWERED")


def fix_sample(flt: dict, available: set, loader, *, target: int = N.TARGET_TOKENS, min_tokens: int = N.MIN_TOKENS) -> dict:
    """The registered definition, as a pure function.

    ``flt``       ``navtrain_common.parse_scene_filter`` output;
    ``available`` the set of ``<log>/<CAM>/<file>.jpg`` rels present in the downloaded shards;
    ``loader``    ``log_name -> list of frame dicts`` or ``None`` when that log's metadata pickle is not on disk."""
    h = flt["header"]
    nh, nf, has_route = h["num_history_frames"], h["num_future_frames"], h["has_route"]
    cams = ("CAM_F0",) if MUTATION == "f0_only" else N.CAMS
    logs = list(flt["log_names"]) if MUTATION == "no_sort_logs" else sorted(set(flt["log_names"]))
    navtok = set(flt["tokens"])
    chosen, token_log, per_log, maps = [], {}, [], collections.Counter()
    stubbed = set()
    for rank, log in enumerate(logs):
        frames = loader(log)
        if frames is None:
            per_log.append({"log": log, "rank": rank, "status": "NO_METADATA"})
            continue
        pos = {f["token"]: i for i, f in enumerate(frames)}
        cand = [f["token"] for f in frames if f["token"] in navtok]
        ok, no_window, missing = [], 0, 0
        for t in cand:
            win = N.scene_window(frames, pos[t], nh, nf, has_route)
            if win is None and MUTATION != "keep_unwindowed":
                no_window += 1
                continue
            if win is None:
                win = frames[max(0, pos[t] - nh + 1):pos[t] + nf + 1]
                if len(win) < nh:
                    no_window += 1
                    continue
            rels = N.readable_rels(win, nh, cams)
            if MUTATION == "t0_frame_only":
                rels = [N.history_rels(win, nh, cams)[c][nh - 1] for c in cams]
            if all(r in available for r in rels):
                ok.append(t)
            else:
                missing += 1
        for t in ok:
            token_log[t] = log
            maps[str(frames[pos[t]].get("map_location"))] += 1
        chosen += ok
        per_log.append({"log": log, "rank": rank, "status": "READ", "n_navtrain_tokens": len(cand), "n_readable": len(ok),
                        "n_unreadable_no_window": no_window, "n_unreadable_frames_missing": missing, "cumulative": len(chosen)})
        reached = len(chosen) > target if MUTATION == "stop_strict" else len(chosen) >= target
        if reached:
            break
    if MUTATION == "truncate_to_target" and len(chosen) > target:
        chosen = chosen[:target]
        token_log = {t: token_log[t] for t in chosen}
    toks = sorted(chosen)
    if len(set(toks)) != len(toks):
        raise ValueError("duplicate tokens across logs -- the filter is inconsistent")
    status = "FIXED" if (len(toks) >= min_tokens or MUTATION == "underpowered_off") else "UNDERPOWERED"
    used_logs = [p["log"] for p in per_log if p.get("n_readable")]
    return {"rule": RULE, "status": status, "target_tokens": target, "min_tokens": min_tokens, "target_reached": len(toks) >= target,
            "count": len(toks), "n_logs": len(used_logs),
            "sha256_sorted_tokens": N.sha256_tokens(chosen) if MUTATION == "hash_unsorted" else N.sha256_tokens(toks),
            "tokens": toks if MUTATION != "hash_unsorted" else chosen, "log_names": used_logs,
            "token_log": token_log, "per_log": per_log, "map_locations": dict(sorted(maps.items())),
            "readability": "metadata row (a devkit-kept window) AND 4 history frames x {CAM_L0, CAM_F0, CAM_R0} present (KB3 / AMENDMENT 1a)"}


def inventory(sources: list) -> tuple:
    """``(set of rels, [per-source stats])`` from the shard headers / directories."""
    avail, stats = set(), []
    for s in sources:
        n, nbytes = 0, 0
        for rel, size in N.iter_members(s):
            avail.add(rel)
            n += 1
            nbytes += size
        stats.append({"source": s.replace("\\", "/"), "name": os.path.basename(s), "kind": "dir" if os.path.isdir(s) else "archive",
                      "bytes_on_disk": None if os.path.isdir(s) else os.path.getsize(s), "n_cam_jpgs_L0F0R0": n, "jpg_bytes": nbytes})
    return avail, stats


def make_loader(metadata_dir: str):
    stub_names, n_loaded = set(), [0]

    def loader(log: str):
        p = os.path.join(metadata_dir, f"{log}.pkl")
        if not os.path.isfile(p):
            return None
        frames, stubbed = N.load_log_pickle(p)
        stub_names.update(stubbed)
        n_loaded[0] += 1
        return frames

    loader.stubbed, loader.n_loaded = stub_names, n_loaded
    return loader


def verify_shard_hashes(stats: list, tree_json: str) -> list:
    """Compare each archive's sha256 with the HF LFS ``oid`` listed in a ``raw_hf_tree_*.json`` (the groundwork's HF metadata dumps)."""
    tree = json.load(open(tree_json, encoding="utf-8"))
    oid = {os.path.basename(e["path"]): e["lfs"]["oid"] for e in tree if e.get("type") == "file" and e.get("lfs")}
    problems = []
    for st in stats:
        if st["kind"] != "archive":
            continue
        want = oid.get(st["name"])
        st["sha256"] = N.sha256_file(st["source"])
        st["sha256_expected_hf_lfs_oid"] = want
        if want is None or st["sha256"] != want:
            problems.append(f"{st['name']}: sha256 {st['sha256']} != HF LFS oid {want}")
    return problems


def write_guarded(out: str, doc: dict) -> int:
    """Availability is fixed ONCE.  An existing FIXED / UNDERPOWERED file is never replaced with different content."""
    if os.path.exists(out) and MUTATION != "no_overwrite_guard":
        old = json.load(open(out, encoding="utf-8"))
        if old.get("status") in ("FIXED", "UNDERPOWERED"):
            if old.get("sha256_sorted_tokens") == doc["sha256_sorted_tokens"] and old.get("status") == doc["status"]:
                return 0                                   # idempotent re-run
            print(f"ZZSAMPLE_REFUSED existing {old.get('status')} sample (sha256 {old.get('sha256_sorted_tokens')}) differs from this run "
                  f"({doc['sha256_sorted_tokens']}); data availability is never revisited", flush=True)
            return 2
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return 0


def main(argv=None) -> int:
    if MUTATION:
        sys.exit("[fix_navtrain_sample] R8FIX_MUTATION is set -- a mutation-test environment never fixes a sample")
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-yaml", required=True)
    ap.add_argument("--filter-sha256", default=N.NAVTRAIN_FILTER_SHA256, help="the registered filter file (default: the v1.1 tree's)")
    ap.add_argument("--metadata-dir", required=True, help="the extracted OpenScene trainval pickles (navsim_logs/trainval)")
    ap.add_argument("--shards-dir", default="")
    ap.add_argument("--shard", action="append", default=[], help="an explicit shard archive / extracted directory (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=N.TARGET_TOKENS)
    ap.add_argument("--min-tokens", type=int, default=N.MIN_TOKENS)
    ap.add_argument("--verify-sha256-from", default="", help="a raw_hf_tree_*.json: each archive's sha256 must equal its HF LFS oid")
    a = ap.parse_args(argv)
    flt = N.parse_scene_filter(a.filter_yaml)
    if flt["file_sha256"] != a.filter_sha256:
        print(f"ZZSAMPLE_REFUSED filter sha256 {flt['file_sha256']} != registered {a.filter_sha256}", flush=True)
        return 2
    sources = list(a.shard) + (N.discover_shards(a.shards_dir) if a.shards_dir else [])
    if not sources:
        print("ZZSAMPLE_REFUSED no shard source given (--shards-dir / --shard)", flush=True)
        return 2
    avail, stats = inventory(sources)
    problems = verify_shard_hashes(stats, a.verify_sha256_from) if a.verify_sha256_from else []
    if problems:
        print("ZZSAMPLE_REFUSED shard integrity: " + " | ".join(problems), flush=True)
        return 2
    loader = make_loader(a.metadata_dir)
    doc = fix_sample(flt, avail, loader, target=a.target, min_tokens=a.min_tokens)
    doc.update({"filter": {"file": os.path.abspath(a.filter_yaml).replace("\\", "/"), "file_sha256": flt["file_sha256"],
                           "n_logs": len(set(flt["log_names"])), "n_tokens": len(set(flt["tokens"])), "header": flt["header"]},
                "shards": stats, "metadata_dir": os.path.abspath(a.metadata_dir).replace("\\", "/"),
                "n_metadata_pickles_read": loader.n_loaded[0], "unpickler_stubbed_globals": sorted(loader.stubbed),
                "shard_sha256_verified_against_hf_lfs_oid": bool(a.verify_sha256_from),
                "no_score_read": True, "tool": "fix_navtrain_sample.py"})
    rc = write_guarded(a.out, doc)
    if rc == 0:
        print(f"ZZSAMPLE status={doc['status']} count={doc['count']} logs={doc['n_logs']} sha256={doc['sha256_sorted_tokens']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
