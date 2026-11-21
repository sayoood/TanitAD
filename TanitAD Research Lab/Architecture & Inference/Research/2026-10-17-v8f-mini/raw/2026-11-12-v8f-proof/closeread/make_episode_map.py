#!/usr/bin/env python
"""make_episode_map.py -- card PREREG_V8F_PROOF_ADD77 (CLOSEREAD): the episode of every evaluation clip, and the identity of the inputs of control c3.
Reads the episode files (ep*.npz: clip_index, eid per window) of ONE banked T1-kin dump folder and the evaluation join, and writes {clip id: episode id} for the fit tool
(--episode-map). It reads no model output and no label: only the two index columns. Refusals (exit 2, nothing written): the join order differs from the grid order, a clip
with two episode ids, a clip of the join without a window, a rule log whose recorded reader or map hash differs from the file given for that seed.

  python make_episode_map.py --dump-dir DIR --join-eval NPZ --grid JSON --out JSON [--pair RULELOG,READER,QMAP]...
Last stdout line: ZZ EPISODE-MAP rc <rc>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_map(ep_files, join_clips):
    """-> (map clip id -> 'ep<eid>', problems). Every clip of the join must have windows and exactly one episode id."""
    seen = {}
    for p in ep_files:
        z = np.load(p, allow_pickle=True)
        ci, eid = np.asarray(z["clip_index"]).astype(np.int64), np.asarray(z["eid"])
        if len(ci) != len(eid):
            return None, ["%s: clip_index and eid differ in length" % os.path.basename(p)]
        for c, e in zip(ci.tolist(), eid.tolist()):
            seen.setdefault(int(c), set()).add(str(e))
    bad = []
    two = sorted(c for c, s in seen.items() if len(s) != 1)
    if two:
        bad.append("%d clips carry more than one episode id" % len(two))
    out_of = sorted(c for c in seen if c < 0 or c >= len(join_clips))
    if out_of:
        bad.append("%d clip indices lie outside the join" % len(out_of))
    missing = [i for i in range(len(join_clips)) if i not in seen]
    if missing:
        bad.append("%d clips of the join have no window in the dump" % len(missing))
    if bad:
        return None, bad
    return {join_clips[c]: "ep" + next(iter(s)) for c, s in sorted(seen.items())}, []


def check_pair(spec):
    """RULELOG,READER,QMAP -> list of problems (hash recorded in the rule log against the file)."""
    parts = spec.split(",")
    if len(parts) != 3:
        return ["--pair needs RULELOG,READER,QMAP: " + spec]
    log, reader, qmap = parts
    z = np.load(log, allow_pickle=True)
    bad = []
    for key, path in (("reader_sha256", reader), ("qmap_sha256", qmap)):
        if key not in z.files:
            bad.append("%s lacks %s" % (os.path.basename(log), key))
            continue
        rec = str(np.asarray(z[key]).reshape(-1)[0])
        got = sha256_file(path)
        if not (got == rec or got.startswith(rec) or rec.startswith(got[:16])):
            bad.append("%s: %s recorded %s, file %s gives %s" % (os.path.basename(log), key, rec[:16], os.path.basename(path), got[:16]))
    form = str(np.asarray(z["form"]).reshape(-1)[0]) if "form" in z.files else ""
    if form != "RR":
        bad.append("%s: form %r, expected RR" % (os.path.basename(log), form))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--join-eval", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", action="append", default=[])
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("REFUSED: output exists: " + a.out)
        print("ZZ EPISODE-MAP rc 2")
        return 2
    J = np.load(a.join_eval, allow_pickle=True)
    clips = [str(c) for c in J["clip"]]
    grid = json.load(open(a.grid, encoding="utf-8"))
    bad = []
    if [str(c) for c in grid["order"]] != clips:
        bad.append("the clip order of the join differs from the order of the grid")
    files = sorted(glob.glob(os.path.join(a.dump_dir, "ep*.npz")))
    if not files:
        bad.append("no episode file in " + a.dump_dir)
    m = None
    if not bad:
        m, bad = build_map(files, clips)
    for spec in a.pair:
        bad += check_pair(spec)
    if bad:
        for b in bad:
            print("REFUSED: " + b)
        print("ZZ EPISODE-MAP rc 2")
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, indent=0, sort_keys=True)
    print("episode map: %d clips, %d episodes, %d episode files, pairs checked %d, out %s sha256 %s" % (len(m), len(set(m.values())), len(files), len(a.pair), a.out, sha256_file(a.out)[:16]))
    print("ZZ EPISODE-MAP rc 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
