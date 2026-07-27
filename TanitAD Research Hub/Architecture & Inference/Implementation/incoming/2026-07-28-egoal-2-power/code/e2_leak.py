#!/usr/bin/env python3
"""E-GOAL-2 S0 -- THE LEAK / OVERLAP CHECK. Priority 1: a leaked result is
worse than no result.

⚠️ WHY THIS EXISTS. A sibling stream found that **4 of 36 val episodes would
have leaked (11 %)** in an adjacent cache, and caught it ONLY by fingerprinting
episodes with a hash of their POSES. `episode_id` is NOT a usable key here --
the parity train corpus holds 2376 episodes and only 2342 unique episode_ids
(34 collisions), so an id-based check OVER-reports overlap and a
filename-based check under-reports it.

WHAT IS FINGERPRINTED
  sha256 over the raw `poses[T,4]` float32 bytes, read with torch.load(mmap=True)
  so only the poses pages are touched (an episode file is ~117 MB and we need
  ~3 kB of it).

WHAT IS CHECKED
  A. val600  x  parity-train           -> overlap MUST be 0
  B. val600[0:40] == published40       -> order-preserving prefix (parity holds:
                                          40 -> 600 ADDS, never re-selects)
  C. /root/valdata/...  ==  _epcache/...  -> the copy the fan dump READS is the
                                          same 600 episodes a prior audit saw.
                                          An audit of a different copy is not an
                                          audit of this one.

Run on pod2 (the only host with both the 600-ep val build and the parity train
corpus):
    OMP_NUM_THREADS=6 python3 e2_leak.py --out /workspace/_egoal2/e2_leak.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

VAL600_READ = "/root/valdata/physicalai-val-0c5f7dac3b11"          # dump reads THIS
VAL600_EPC = ("/workspace/data/physicalai_phase0/_epcache/"
              "physicalai-val-0c5f7dac3b11")
TRAIN = ("/workspace/data/physicalai_phase0/_epcache/"
         "physicalai-train-e438721ae894")
PUB40 = None            # set on the eval pod only; on pod2 the published 40 are
                        # val600[0:40] BY CLAIM -- which is exactly what B tests


def ep_files(d: str):
    p = Path(d)
    return sorted(p.glob("ep_*.pt"))


def pose_sha(f: Path) -> tuple[str, int]:
    """sha256 of the raw poses bytes. Returns (hex, T)."""
    ep = torch.load(str(f), map_location="cpu", mmap=True, weights_only=False)
    poses = ep["poses"] if isinstance(ep, dict) else ep.poses
    t = torch.as_tensor(poses).float().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest(), int(t.shape[0])


def fingerprint(d: str, label: str, limit=None) -> dict:
    fs = ep_files(d)
    if limit:
        fs = fs[:limit]
    out, t0 = {}, time.time()
    for i, f in enumerate(fs):
        h, T = pose_sha(f)
        out[f.name] = {"sha256": h, "T": T}
        if (i + 1) % 200 == 0:
            print(f"  [{label}] {i+1}/{len(fs)} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  [{label}] {len(fs)} episodes in {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/_egoal2/e2_leak.json")
    a = ap.parse_args()

    res = {"_stream": "2026-07-28-egoal-2-power", "_stage": "S0 leak/overlap",
           "_method": ("sha256 over raw poses[T,4] float32 bytes, "
                       "torch.load(mmap=True). episode_id is NOT a key -- it "
                       "collides (2376 train episodes, 2342 unique ids)."),
           "_sources": {"val600_read_by_dump": VAL600_READ,
                        "val600_epcache_copy": VAL600_EPC,
                        "parity_train": TRAIN},
           "_host": __import__("socket").gethostname()}

    print("[S0] fingerprinting the val build the dump READS ...", flush=True)
    v_read = fingerprint(VAL600_READ, "val600/read")
    print("[S0] fingerprinting the parity TRAIN corpus ...", flush=True)
    tr = fingerprint(TRAIN, "train2376")
    print("[S0] fingerprinting the _epcache val copy ...", flush=True)
    v_epc = fingerprint(VAL600_EPC, "val600/epcache")

    v_sha = {k: v["sha256"] for k, v in v_read.items()}
    t_sha = {k: v["sha256"] for k, v in tr.items()}
    e_sha = {k: v["sha256"] for k, v in v_epc.items()}

    # ---------------- A. THE LEAK CHECK -------------------------------------
    overlap = sorted(set(v_sha.values()) & set(t_sha.values()))
    inv_t = {}
    for k, h in t_sha.items():
        inv_t.setdefault(h, []).append(k)
    inv_v = {}
    for k, h in v_sha.items():
        inv_v.setdefault(h, []).append(k)
    res["A_TRAIN_OVERLAP"] = {
        "val600_n": len(v_sha), "val600_unique_sha": len(set(v_sha.values())),
        "train_n": len(t_sha), "train_unique_sha": len(set(t_sha.values())),
        "overlap_n": len(overlap),
        "overlap_pct_of_val": round(100.0 * len(overlap) / max(1, len(v_sha)), 4),
        "overlap_detail": [{"sha256": h, "val_files": inv_v[h],
                            "train_files": inv_t[h]} for h in overlap[:50]],
        "verdict": ("CLEAN -- 0 byte overlap" if not overlap
                    else f"LEAK -- {len(overlap)} episodes must be DROPPED"),
    }
    # id-based comparison reported ONLY to show why it is not the key
    res["A_ID_BASED_FOR_CONTRAST"] = {
        "note": ("filename overlap is meaningless across caches; reported to "
                 "show the fingerprint is doing work"),
        "filename_overlap_n": len(set(v_sha) & set(t_sha)),
    }

    # ---------------- B. ORDER-PRESERVING PREFIX ----------------------------
    vk = sorted(v_sha)                 # ep_00000.pt ... ep_00599.pt
    first40 = [v_sha[k] for k in vk[:40]]
    res["B_PREFIX"] = {
        "first40_files": vk[:40],
        "first40_sha256": first40,
        "first40_all_distinct": len(set(first40)) == 40,
        "note": ("the published 40-episode deployment is claimed to be "
                 "val600[0:40]; the eval pod's own copy is fingerprinted "
                 "separately by e2_leak.py --out on that host and compared in "
                 "EGOAL_2.md"),
    }

    # ---------------- C. CROSS-COPY IDENTITY --------------------------------
    same = sorted(set(v_sha.values()) & set(e_sha.values()))
    res["C_COPY_IDENTITY"] = {
        "read_n": len(v_sha), "epcache_n": len(e_sha),
        "shared_sha_n": len(same),
        "identical_set": set(v_sha.values()) == set(e_sha.values()),
        "identical_positionally": all(
            v_sha.get(k) == e_sha.get(k) for k in vk),
        "verdict": ("SAME BUILD" if set(v_sha.values()) == set(e_sha.values())
                    else "DIFFERENT BUILDS -- a prior audit of the epcache copy "
                         "does NOT cover the copy the dump reads"),
    }

    res["_per_episode"] = {"val600_read": v_read, "val600_epcache": v_epc}
    res["_train_sha_only"] = sorted(set(t_sha.values()))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if not k.startswith("_")}, indent=1)[:4000], flush=True)
    print(f"[S0] -> {a.out}", flush=True)
    if overlap:
        print("[S0] ⛔ LEAK DETECTED", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
