#!/usr/bin/env python3
"""Control s8 (card ADD100 section 5): complete. Every clip of the list (default: the 141 evaluation + the train_final clips = 641) has a
steering file (float32 [n,2], finite) and an arm S token file with the frame count of the join; the token sha256 is re-hashed against
tok/S/MANIFEST.json. Also (second half of control s2): the manifest carries a steering sha256 for every clip and each equals the sha256
of the steering file on disk. Writes ctl/S8.json with a top-level boolean `pass`. Reads only; never deletes."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steer_encode as SE   # noqa: E402
from enc_common import read_ids, sha256_file   # noqa: E402
from ctl_c7 import join_n   # noqa: E402


def check_clip(c, nmap, s_root, man, rehash=True):
    why, steer_sha_ok = [], False
    n = nmap.get(c)
    if n is None:
        why.append("not in the join")
    sp, tp = SE.steer_path(s_root, c), SE.tok_path(s_root, c)
    steer_sha = None
    if not os.path.exists(sp):
        why.append("no steering file")
    else:
        try:
            SE.check_steering(np.load(sp, allow_pickle=False), n if n is not None else -1)
            steer_sha = sha256_file(sp)
        except Exception as e:                                       # noqa: BLE001
            why.append(f"steering file refused: {str(e)[:120]}")
    m = man.get("clips", {}).get(c)
    if m is None:
        why.append("not in manifest")
    else:
        if not m.get("steer_sha256"):
            why.append("manifest has no steering sha256")
        elif steer_sha is not None and m["steer_sha256"] != steer_sha:
            why.append("manifest steering sha256 differs from the file on disk")
        elif steer_sha is not None:
            steer_sha_ok = True
        if not os.path.exists(tp) or os.path.getsize(tp) != m["bytes"]:
            why.append("token file missing or size differs")
        else:
            if n is not None and m["n_frames"] != n:
                why.append(f"n_frames {m['n_frames']} != join {n}")
            if rehash and sha256_file(tp) != m["sha256"]:
                why.append("token sha256 differs from the manifest")
    return why, steer_sha_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-list", required=True)
    ap.add_argument("--train-final", required=True)
    ap.add_argument("--clips", default=None, help="smoke only: a list file that replaces eval + train-final")
    ap.add_argument("--join-eval", default="/workspace/v8proof/hierrep/pins/join_eval141.npz")
    ap.add_argument("--join-train", default="/workspace/v8proof/data/inputs/join_train.npz")
    ap.add_argument("--s-root", default=os.environ.get("S_ROOT", SE.DEFAULT_S_ROOT))
    ap.add_argument("--n-required", type=int, default=641)
    ap.add_argument("--no-rehash", action="store_true", help="smoke only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.s_root, "ctl", "S8.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    clips = read_ids(a.clips) if a.clips else read_ids(a.eval_list) + read_ids(a.train_final)
    nmap = {**join_n(a.join_eval), **join_n(a.join_train)}
    mp = os.path.join(a.s_root, "tok", "S", "MANIFEST.json")
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    left, ok, sha_ok = [], 0, 0
    for c in clips:
        why, s_ok = check_clip(c, nmap, a.s_root, man, rehash=not a.no_rehash)
        sha_ok += int(s_ok)
        if why:
            left.append({"clip": c, "why": why})
        else:
            ok += 1
    need = a.n_required if not a.clips else len(clips)
    res = {"control": "s8", "n_listed": len(clips), "n_complete": ok, "n_required": need, "left_out": left, "sha_rehashed": not a.no_rehash,
           "s2_second_half": {"manifest_steering_sha_matches_file": sha_ok, "of": len(clips), "pass": bool(sha_ok == len(clips) and len(clips) > 0)},
           "pass": bool(ok == len(clips) and len(clips) >= need)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "left_out"}))
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
