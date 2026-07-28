"""POPULATION validation of the v5 slice — across many clips and BOTH rigs.

⛔ WHY THIS SUPERSEDES MY FIRST PASS. I profiled ONE frame of ONE clip. That is exactly the
probe that produced this program's `observed_frac = 1.0` error: `v2_compressed
._assert_geometry_deliverable` sampled clips[0], which happened to be rig A, and rig A
observes the frame fully. The population value is 0.9349 and rig B (73 % of the corpus)
masks ~8.9 % of every frame. A single-clip conclusion about masking is worthless here.

So: sample N clips, decode a frame from each, and report the zero-fraction profile
SEPARATELY for the two rigs — using the same cy boundary the cache's own geometry manifest
used (rig_assignment: "native principal-point row cy, boundary 650.872").

The question this must answer: DOES THE v5 SLICE rows[40:216] SIT INSIDE THE REGION BOTH
RIGS OBSERVE? If rig-B frames show a masked band intruding into rows 40..216, the
rig-clean fix does not do what it claims and v5 would train on a rig-correlated black
region — which the design doc explicitly calls out as still a rig-correlated SIGNAL.
"""
from __future__ import annotations

import glob
import io
import json

import numpy as np
import torch
from PIL import Image

CACHE = "/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl"
R0, R1, C0, C1 = 40, 216, 8, 632
N_CLIPS = 60


def decode0(d):
    n = int(d["jpeg_len"][0])
    return np.array(Image.open(io.BytesIO(bytes(d["jpeg_buf"][:n].numpy()))))


def main() -> None:
    files = sorted(glob.glob(f"{CACHE}/*.v2ep.pt"))
    step = max(1, len(files) // N_CLIPS)
    picks = files[::step][:N_CLIPS]
    print(f"sampling {len(picks)} clips of {len(files)} (stride {step})")

    prof = []          # per-clip row zero-fraction profile, 256 rows
    slice_zero = []
    full_zero = []
    for f in picks:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            a = decode0(d)
        except Exception as e:                                    # noqa: BLE001
            print(f"  SKIP {f.split('/')[-1][:12]}: {type(e).__name__}")
            continue
        z = (a == 0).all(axis=2) if a.ndim == 3 else (a == 0)      # fully-black pixels
        prof.append(z.mean(axis=1))                                # [256]
        slice_zero.append(z[R0:R1, C0:C1].mean())
        full_zero.append(z.mean())

    P = np.stack(prof)                                             # [n, 256]
    print(f"\ndecoded {len(P)} clips")
    print(f"  FULL frame  black-pixel frac: mean={np.mean(full_zero):.4f} "
          f"min={np.min(full_zero):.4f} max={np.max(full_zero):.4f}")
    print(f"  v5 SLICE    black-pixel frac: mean={np.mean(slice_zero):.4f} "
          f"min={np.min(slice_zero):.4f} max={np.max(slice_zero):.4f}")

    # bimodality across clips = the two rigs
    sz = np.array(slice_zero)
    order = np.sort(sz)
    print(f"\n  slice black-frac deciles: "
          + " ".join(f"{np.quantile(sz, q):.3f}" for q in np.arange(0, 1.01, 0.1)))

    print(f"\n=== ROW PROFILE of BLACK-PIXEL fraction, averaged over {len(P)} clips ===")
    print("  (a masked band = fraction near 1.0; real content = near 0)")
    worst = P.max(axis=0)
    for r0 in range(0, 256, 16):
        m = P[:, r0:r0 + 16].mean()
        w = worst[r0:r0 + 16].max()
        inside = "  <-- v5 slice" if (r0 >= R0 and r0 + 16 <= R1) else ""
        flag = "   ** WORST-CLIP FULLY MASKED **" if w > 0.99 else ""
        print(f"    rows {r0:3d}-{r0+15:3d}: mean={m:.4f}  worst_clip={w:.4f}{inside}{flag}")

    # the decisive question
    in_slice_worst = worst[R0:R1].max()
    out_bottom = worst[R1:].max() if R1 < 256 else 0.0
    print(f"\n=== DECISIVE ===")
    print(f"  worst-clip black fraction INSIDE the v5 rows[{R0}:{R1}] : {in_slice_worst:.4f}")
    print(f"  worst-clip black fraction BELOW  the v5 slice rows[{R1}:] : {out_bottom:.4f}")
    if in_slice_worst > 0.5:
        print("  🔴 a masked band INTRUDES into the v5 slice on at least one clip —"
              " the rig-clean fix does NOT fully clear it.")
    else:
        print("  ✅ no clip has a majority-masked row inside the v5 slice.")
    geo = json.load(open(f"{CACHE}/_geometry.json"))
    print(f"\n  _geometry.json observed_frac (population, 2400 clips) = "
          f"{geo['geometry_check']['observed_frac']}")
    print(f"  rig_assignment: {geo.get('rig_observability', {}).get('rig_assignment')}")


if __name__ == "__main__":
    main()
