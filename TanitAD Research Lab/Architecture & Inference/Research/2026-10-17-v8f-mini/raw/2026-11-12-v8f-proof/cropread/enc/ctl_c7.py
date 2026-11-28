#!/usr/bin/env python3
"""Control c7 (card section 7): complete. All 141 evaluation clips and at least 490 train clips are staged and encoded for both arms with
the frame count of the join; the clips left out are listed. Reads the lists, the joins (n per clip), tok/<arm>/MANIFEST.json (recomputed
from ledger.jsonl) and, with --verify-sha, the sha256 of every token file."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enc_common import read_ids, sha256_file   # noqa: E402


def join_n(path):
    J = dict(np.load(path, allow_pickle=True))
    return {str(c): int(n) for c, n in zip(J["clip"], J["n"])}


def check_list(clips, nmap, tok_dir, verify):
    left, ok = [], []
    for c in clips:
        why = []
        if c not in nmap:
            why.append("not in the join")
        for arm in ("D", "B"):
            man = os.path.join(tok_dir, arm, "MANIFEST.json")
            m = json.load(open(man))["clips"].get(c) if os.path.exists(man) else None
            if m is None:
                why.append(f"arm {arm}: not in manifest")
                continue
            p = os.path.join(tok_dir, arm, f"{c}.npy")
            if not os.path.exists(p) or os.path.getsize(p) != m["bytes"]:
                why.append(f"arm {arm}: file missing or size differs")
            elif c in nmap and m["n_frames"] != nmap[c]:
                why.append(f"arm {arm}: n_frames {m['n_frames']} != join {nmap[c]}")
            elif verify and sha256_file(p) != m["sha256"]:
                why.append(f"arm {arm}: sha256 differs")
        (left if why else ok).append({"clip": c, "why": why} if why else c)
    return ok, left


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-list", required=True)
    ap.add_argument("--train-final", required=True)
    ap.add_argument("--join-eval", default="/workspace/v8proof/hierrep/pins/join_eval141.npz")
    ap.add_argument("--join-train", default="/workspace/v8proof/data/inputs/join_train.npz")
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--verify-sha", action="store_true")
    ap.add_argument("--n-eval", type=int, default=141)
    ap.add_argument("--n-train-min", type=int, default=490)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ev, tr = read_ids(a.eval_list), read_ids(a.train_final)
    ok_e, left_e = check_list(ev, join_n(a.join_eval), a.tok_dir, a.verify_sha)
    ok_t, left_t = check_list(tr, join_n(a.join_train), a.tok_dir, a.verify_sha)
    res = {"control": "c7", "n_eval_listed": len(ev), "n_eval_complete": len(ok_e), "n_train_listed": len(tr), "n_train_complete": len(ok_t),
           "eval_required": a.n_eval, "train_required_min": a.n_train_min, "left_out_eval": left_e, "left_out_train": left_t,
           "sha_verified": bool(a.verify_sha),
           "pass": bool(len(ok_e) >= a.n_eval and len(ok_t) >= a.n_train_min)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("left_out")}))
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
