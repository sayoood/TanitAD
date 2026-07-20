"""Reconcile pod1's train cache to pod2's canonical 2376 (Sayed / coordinator).

pod1 may DECODE the 24 clips pod2 found corrupt (as pod3 REF-A did), finishing
2400/0-skip. REF-B must train on the IDENTICAL 2376 episodes as the flagship, so
drop the episodes at pod2's canonical skip train-positions and write skip_
markers there.

SAFETY:
- Acts only if pod1's current skip set is a SUBSET of pod2's (pod1 built
  everything pod2 built). If pod1 skipped a position pod2 BUILT (a non-droppable
  missing clip), exit 4 -> STOP + flag; no deletion.
- Idempotent. Deletes ONLY ep files at the 24 hardcoded canonical positions.
- The launch gate re-verifies sha256 == f09e44db AFTER this, so a wrong drop can
  never cause training on non-canonical data (it HOLDs instead).
"""
import glob
import os
import sys

ROOT = os.environ.get("TANITAD_PHYSICALAI_ROOT",
                      "/workspace/data/physicalai_phase0")
KEY = "physicalai-train-e438721ae894"
# pod2 canonical corrupt-clip skip positions in the train split (from coordinator)
P2_SKIP = [1798, 1835, 1841, 1842, 1843, 1847, 1854, 1857, 1858, 1860, 1862,
           1863, 1873, 1875, 1876, 1877, 1879, 1880, 1885, 1888, 1892, 1896,
           1898, 1941]


def main():
    assert len(P2_SKIP) == 24, len(P2_SKIP)
    d = os.path.join(ROOT, "_epcache", KEY)
    if not os.path.isdir(d):
        print("NO_CACHE_DIR")
        sys.exit(2)
    p1 = sorted(int(os.path.basename(p)[5:10])
                for p in glob.glob(f"{d}/skip_*"))
    n_built = len(glob.glob(f"{d}/ep_*.pt"))
    p2 = set(P2_SKIP)
    extra = [p for p in p1 if p not in p2]
    print(f"pre: built={n_built} pod1_skips={len(p1)} pod1_skip_pos={p1}")
    if extra:
        print(f"NONDROPPABLE pod1 skipped positions pod2 BUILT: {extra} "
              f"-> STOP + flag (missing non-droppable clips)")
        sys.exit(4)
    dropped, marked = [], []
    for p in sorted(P2_SKIP):
        ep = os.path.join(d, f"ep_{p:05d}.pt")
        sk = os.path.join(d, f"skip_{p:05d}")
        if os.path.exists(ep):
            os.remove(ep)
            dropped.append(p)
        if not os.path.exists(sk):
            with open(sk, "w") as f:
                f.write("reconciled: pod2 canonical corrupt-clip skip (Sayed)\n")
            marked.append(p)
    n_after = len(glob.glob(f"{d}/ep_*.pt"))
    n_skip_after = len(glob.glob(f"{d}/skip_*"))
    print(f"RECONCILED dropped_eps={dropped} new_markers={marked}")
    print(f"post: built={n_after} skips={n_skip_after}")
    sys.exit(0)


if __name__ == "__main__":
    main()
