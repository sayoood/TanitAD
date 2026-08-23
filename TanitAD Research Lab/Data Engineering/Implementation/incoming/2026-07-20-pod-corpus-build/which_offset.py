"""Which pose slice does a cached ep actually use? Decisive, not correlational.

For a chunk-1 ep, derive the speed track v from RAW vehicle_pose at offset 0 and
at offset 121 (the same stride/dt/derivation the loader uses), then report the
RMS difference to the cached ep's poses[:,3]. The matching offset is ~0.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/TanitAD/stack")
from tanitad.data.cosmos_drive import poses_to_signals, load_vehicle_pose  # noqa: E402
from tanitad.data.mixing import load_episode                              # noqa: E402

PAIRS = Path("/root/cosmos_data/pairs")
OLD = Path("/root/valdata/cosmos-val-a7a8527ba14e")
NEW = Path("/root/valdata/cosmos-val-e8f3cef4976b")
MAN = json.loads((NEW / "build_manifest.json").read_text())["eps"]
STRIDE, DT, K = 3, 0.1, 2


def ref_v(cid: str, off: int, n_out: int) -> np.ndarray:
    M = load_vehicle_pose(PAIRS / "vehicle_pose" / cid)
    if M.shape[0] - off < 20:
        return None
    n = min(121, M.shape[0] - off)
    _a, p = poses_to_signals(M[off:off + n][::STRIDE], DT)
    return p[K:K + n_out, 3]


def rms(a, b):
    m = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:m] - b[:m]) ** 2)))


def main():
    print(f"{'ep':>4} {'ch':>2} {'clip':8} {'npose':>5} | "
          f"{'OLD@0':>7} {'OLD@121':>7} -> verdict | "
          f"{'NEW@0':>7} {'NEW@121':>7} -> verdict")
    tally = {"old": {}, "new": {}}
    for k in sorted(MAN, key=int):
        e = MAN[k]
        cid, ch, i = e["clip"], e["chunk"], int(k)
        npose = len(list((PAIRS / "vehicle_pose" / cid).glob("*.npy")))
        row = [f"{i:4d} {ch:2d} {cid[:8]} {npose:5d} |"]
        for tag, root in (("old", OLD), ("new", NEW)):
            f = root / f"ep_{i:05d}.pt"
            if not f.exists():
                row.append(f"{'-':>7} {'-':>7} -> missing |")
                continue
            v = load_episode(str(f), mmap=True).poses[:, 3].numpy()
            r0 = ref_v(cid, 0, len(v))
            r1 = ref_v(cid, 121, len(v))
            d0 = rms(v, r0) if r0 is not None else float("inf")
            d1 = rms(v, r1) if r1 is not None else float("inf")
            verdict = "off=0" if d0 < d1 else "off=121"
            tally[tag].setdefault((ch, verdict), 0)
            tally[tag][(ch, verdict)] += 1
            row.append(f"{d0:7.3f} {d1:7.3f} -> {verdict:7} |")
        print(" ".join(row), flush=True)
    print("\nTALLY (chunk, offset-actually-used) -> count")
    for tag in ("old", "new"):
        print(f"  {tag}: {dict(sorted(tally[tag].items()))}")


if __name__ == "__main__":
    main()
