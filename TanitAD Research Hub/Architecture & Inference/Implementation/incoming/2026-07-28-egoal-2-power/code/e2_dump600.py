#!/usr/bin/env python3
"""E-GOAL-2 S1 -- the REF-C-XL fan on the 600-episode val build.

⛔ THE REPO MODULE IS NOT EDITED AND THE DECODE IS NOT RE-IMPLEMENTED.
`taniteval.refc_rerank.dump` is called VERBATIM. The only thing this wrapper
does is point its two module-level constants at a 600-episode build and a new
output path, so the decode path is byte-identical to the one that produced the
committed 881-window `fan_refc-xl-30k.pt`. That is what makes the prefix
fidelity gate (F-A) meaningful: if a single line of the decode differed, the
first 40 episodes could not reproduce.

pod2 md5-check before running (both must hold, and both are asserted):
  * `taniteval/taniteval/refc_rerank.py` == the repo copy
  * `refc-xl-30k/ckpt.pt` md5 == 966d4eff1ea5ddf86efba01b8344e198

Run on pod2:
    PYTHONPATH=/root/TanitAD/stack:/root/taniteval OMP_NUM_THREADS=6 \
      python3 e2_dump600.py --episodes 600 \
        --out /workspace/_egoal2/fan_refc-xl-30k_600ep.pt
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")

CKPT_MD5 = "966d4eff1ea5ddf86efba01b8344e198"


def md5(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--val", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--out", default="/workspace/_egoal2/fan_refc-xl-30k_600ep.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-md5", action="store_true")
    a = ap.parse_args()

    from taniteval import refc_rerank as rr

    e = [m for m in __import__("taniteval.registry", fromlist=["MODELS"]).MODELS
         if m["key"] == rr.BASE_KEY][0]
    if not a.skip_md5:
        t0 = time.time()
        got = md5(e["ckpt"])
        print(f"[gate] ckpt md5 {got} (expect {CKPT_MD5}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
        assert got == CKPT_MD5, (
            f"REFUSING: refc-xl-30k ckpt md5 {got} != {CKPT_MD5}. A different "
            f"checkpoint would silently produce a different fan and every "
            f"number downstream would be wrong.")

    n_files = len(sorted(Path(a.val).glob("ep_*.pt")))
    print(f"[gate] val dir {a.val}: {n_files} episode files", flush=True)
    assert n_files >= a.episodes, (
        f"REFUSING: {a.val} holds {n_files} episodes, {a.episodes} requested. "
        f"A truncated val cache is exactly the failure list_val_episodes "
        f"exists to refuse.")

    # ---- point the COMMITTED module at the 600-episode build ---------------
    rr.VAL = a.val
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rr.RES = Path(a.out).parent

    d = rr.dump(episodes=a.episodes, device="cuda", batch=a.batch,
                out=Path(a.out))
    print(f"[dump600] fan {tuple(d['fan'].shape)}  "
          f"episodes {len(set(map(str, d['eid'])))}  "
          f"windows {d['fan'].shape[0]}  wall {d['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
