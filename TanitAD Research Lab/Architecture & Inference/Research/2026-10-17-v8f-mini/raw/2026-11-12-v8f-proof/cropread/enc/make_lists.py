#!/usr/bin/env python3
"""CROPREAD lists (card PREREG_V8F_PROOF_ADD99 section 2).

Reads the two joins (fields `clip`, `off`, `n`, `lead_valid`, `lead_gap`; frame i of a clip is row off+i) and writes, under --out:
  eval141.txt        the evaluation clips in the order of join_eval141.npz
  train_order.txt    "<position> <clip>" for the first 520 of the ordered train eligible set (position is 1-based)
  train_eligible.txt every eligible train clip id in the card's order (so the order is auditable)
  LISTS_RECORD.json  counts, sha256 of each list, overlaps (must be 0)

Train eligible = in join_train, NOT held out by the present reader's rule (int(sha256(clip id)[:8], 16) % 10 != 0), and at least
10 frames with lead_valid == 1, finite lead_gap, 0 <= lead_gap <= 60. Order: ascending sha256 hex of the clip id.
Pure numpy + hashlib: runs on the dev box or on the pod. No clip id is read from anywhere but the joins.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

N_LIST = 520
MIN_VALID = 10
GAP_MAX = 60.0


def sha_hex(clip: str) -> str:
    return hashlib.sha256(clip.encode("utf-8")).hexdigest()


def held_out(clip: str) -> bool:
    """The present reader's hold-out rule (card section 2)."""
    return int(sha_hex(clip)[:8], 16) % 10 == 0


def valid_lead_mask(lead_valid, lead_gap):
    g = np.asarray(lead_gap, dtype=np.float64)
    return (np.asarray(lead_valid) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX)


def join_clips(J):
    return [str(c) for c in np.asarray(J["clip"])]


def eligible_train(J, min_valid=MIN_VALID):
    """-> (eligible clip ids in the card's order, counts dict)."""
    clips = join_clips(J)
    off = np.asarray(J["off"], dtype=np.int64)
    n = np.asarray(J["n"], dtype=np.int64)
    ok = valid_lead_mask(J["lead_valid"], J["lead_gap"])
    keep, n_held, n_few = [], 0, 0
    for c, o, k in zip(clips, off, n):
        if held_out(c):
            n_held += 1
            continue
        if int(ok[o:o + k].sum()) < min_valid:
            n_few += 1
            continue
        keep.append(c)
    keep.sort(key=sha_hex)
    return keep, {"n_join_clips": len(clips), "n_held_out": n_held, "n_not_held_out_few_valid_lead": n_few,
                  "n_eligible": len(keep)}


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(ln + "\n")


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(J_eval, J_train, out_dir, expect_eligible=1679):
    os.makedirs(out_dir, exist_ok=True)
    ev = join_clips(J_eval)
    elig, cnt = eligible_train(J_train)
    train_all = set(join_clips(J_train))
    overlap_eval_train_join = sorted(set(ev) & train_all)
    overlap_eval_eligible = sorted(set(ev) & set(elig))
    if overlap_eval_train_join:
        raise SystemExit(f"REFUSED: {len(overlap_eval_train_join)} evaluation clips are in the train join "
                         f"(first: {overlap_eval_train_join[0][:8]})")
    p_eval = os.path.join(out_dir, "eval141.txt")
    p_ord = os.path.join(out_dir, "train_order.txt")
    p_all = os.path.join(out_dir, "train_eligible.txt")
    write_lines(p_eval, ev)
    write_lines(p_ord, [f"{i + 1} {c}" for i, c in enumerate(elig[:N_LIST])])
    write_lines(p_all, elig)
    rec = {"schema": "cropread_lists/1", **cnt,
           "expect_eligible": expect_eligible, "eligible_matches_expectation": cnt["n_eligible"] == expect_eligible,
           "n_eval": len(ev), "n_eval_unique": len(set(ev)),
           "n_train_order_written": min(N_LIST, len(elig)),
           "rule": {"held_out": "int(sha256(clip)[:8],16) % 10 == 0", "min_valid_lead_frames": MIN_VALID,
                    "valid_lead": f"lead_valid==1 & finite(lead_gap) & 0<=lead_gap<={GAP_MAX:g}",
                    "order": "ascending sha256 hex of the clip id", "position_base": 1},
           "overlap_eval_vs_train_join": len(overlap_eval_train_join),
           "overlap_eval_vs_train_eligible": len(overlap_eval_eligible),
           "overlap_eval_vs_train_order_520": len(set(ev) & set(elig[:N_LIST])),
           "sha256": {os.path.basename(p): file_sha(p) for p in (p_eval, p_ord, p_all)}}
    with open(os.path.join(out_dir, "LISTS_RECORD.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--join-eval", default="/workspace/v8proof/hierrep/pins/join_eval141.npz")
    ap.add_argument("--join-train", default="/workspace/v8proof/data/inputs/join_train.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Je = dict(np.load(a.join_eval, allow_pickle=True))
    Jt = dict(np.load(a.join_train, allow_pickle=True))
    rec = build(Je, Jt, a.out)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("rule", "sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
