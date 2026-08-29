"""⛔ BUILD GATE — run this BEFORE any B1 epcache build; a non-zero exit means DO NOT BUILD.

WHY THIS IS A SCRIPT AND NOT A SENTENCE IN A RUNBOOK. Both calibration failures
on this corpus are SILENT FALLBACKS, not errors:

  * `intrinsics_for_clip` -> corpus-median fallback (per_clip=False). The
    downstream `cylindrical_rectify` guard catches it -- but ONLY if nobody
    passes `require_per_clip=False` to get "past the crash".
  * `extrinsics_for_clip` -> **None**, and callers then "treat the mount as
    level (optical axis == horizon)" (physicalai.py:450). NOTHING raises. A
    corpus built this way is silently wrong about where the horizon is.

MEASURED 2026-08-29 on the B1 corpus: root resolution returns None (the camera
bank has no `r0` ancestor), and even WITH an explicit root the local
`r0_selection.parquet` covers 38 of 4,719 clips -- so extrinsics would be None
for 99.2 % of the corpus. Neither failure prints an error.

⇒ This gate asserts the POSITIVE fact (real per-clip calibration resolved) on a
random sample, instead of trusting the absence of a warning.

    python preflight_epcache_build.py --root <corpus root> [--sample 40]
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as FRAME  # noqa: E402
from tanitad.data.physicalai import (_physicalai_root_of,  # noqa: E402
                                     extrinsics_for_clip, intrinsics_for_clip)

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--cam-glob", default="r0/camera_front_wide_120fov/*.mp4")
ap.add_argument("--sample", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

root = Path(a.root)
fails: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        fails.append(name)


print(f"epcache build preflight | root {root}\n"
      f"frame {FRAME.height}x{FRAME.width} {FRAME.projection} "
      f"f_ref {FRAME.f_ref:.4f} hfov {FRAME.hfov_deg:.2f}", flush=True)

mp4s = sorted(root.glob(a.cam_glob))
check("camera bank found", bool(mp4s), f"{len(mp4s)} mp4 under {a.cam_glob}")
if not mp4s:
    print("\nPREFLIGHT FAILED — no camera files; nothing else can be checked.")
    sys.exit(2)

# 1. root recovery -- the failure that makes BOTH lookups fall back
rec = _physicalai_root_of(mp4s[0])
check("root recoverable from a clip path (needs an `r0` ancestor)",
      rec is not None and Path(rec).resolve() == root.resolve(),
      f"_physicalai_root_of -> {rec}")

# 2 + 3. real per-clip calibration on a random sample
random.seed(a.seed)
sample = random.sample(mp4s, min(a.sample, len(mp4s)))
ids = [p.name.split(".")[0] for p in sample]
intr = [(c, intrinsics_for_clip(c, rec or root)) for c in ids]
per_clip = [c for c, i in intr if getattr(i, "per_clip", False)]
check(f"per-clip INTRINSICS on {len(ids)} sampled clips",
      len(per_clip) == len(ids), f"{len(per_clip)}/{len(ids)} real (rest = median fallback)")

extr = [(c, extrinsics_for_clip(c, rec or root)) for c in ids]
got = [c for c, e in extr if e is not None]
check(f"per-clip EXTRINSICS on {len(ids)} sampled clips",
      len(got) == len(ids),
      f"{len(got)}/{len(ids)} resolved (None => mount silently assumed LEVEL)")

# 4. both rigs actually present -- a sample that saw one rig proves little
cys = sorted(i.cy for _, i in intr if getattr(i, "per_clip", False))
if cys:
    nA = sum(1 for v in cys if v < 650)
    check("sample covers BOTH rigs (cy ~543 A / ~755 B)",
          nA > 0 and nA < len(cys), f"{nA} rig-A / {len(cys)-nA} rig-B")

print()
if fails:
    print(f"⛔ PREFLIGHT FAILED ({len(fails)}): " + "; ".join(fails))
    print("DO NOT BUILD. Every failure above is a SILENT fallback at build time.")
    sys.exit(1)
print("✅ PREFLIGHT PASSED — real per-clip intrinsics AND extrinsics resolve. Safe to build.")
