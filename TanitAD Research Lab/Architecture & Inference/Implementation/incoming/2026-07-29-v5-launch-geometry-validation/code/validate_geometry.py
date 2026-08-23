"""VALIDATE the v5 176x624 image interpretation, intrinsics and extrinsics before GPU-days.

PI: "accept the small regression and go with 117 degree, validate your image interpretation
and extrinsics/intrinsics" (2026-07-29).

WHY THIS IS NOT CEREMONY. This program has shipped geometry errors twice: the geometric-centre
crop was ~215 px wrong for rig B because the front-wide camera has TWO rigs with principal points
~211 px apart; and a fixed 100 deg HFOV was refuted by GeoCalib on fresh video. A wrong frame or a
wrong pose convention does not crash — it trains silently and produces a plausible wrong number.

Checks, each independently falsifiable:
  1. decoded pixels match the declared image_h/image_w
  2. the v5 slice rows[40:216], cols[8:632] is real image content, not padding/mask
  3. rig-B masking is VISIBLE in the full frame and its extent matches _geometry.json
  4. the slice AVOIDS the masked band (this is the whole point of the rig-clean fix)
  5. pose/action semantics and units are what the trainer assumes
"""
from __future__ import annotations

import glob
import io
import json

import numpy as np
import torch
from PIL import Image

CACHE = "/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl"
R0, R1, C0, C1 = 40, 216, 8, 632          # the v5 centred slice


def decode(d, i):
    off = int(d["jpeg_len"][:i].sum())
    n = int(d["jpeg_len"][i])
    raw = bytes(d["jpeg_buf"][off:off + n].numpy())
    im = Image.open(io.BytesIO(raw))
    return im, np.array(im)


def main() -> None:
    geo = json.load(open(f"{CACHE}/_geometry.json"))
    files = sorted(glob.glob(f"{CACHE}/*.v2ep.pt"))
    print(f"cache has {len(files)} episode files")

    # --- 1/2: decode + slice ------------------------------------------------
    d = torch.load(files[0], map_location="cpu", weights_only=False)
    im, a = decode(d, 0)
    print(f"\n=== 1. DECODE ===")
    print(f"  PIL format={im.format} mode={im.mode} size={im.size}")
    print(f"  declared image_h={d['image_h']} image_w={d['image_w']}  decoded={a.shape}")
    ok_decode = (a.shape[0] == d["image_h"] and a.shape[1] == d["image_w"])
    print(f"  MATCH: {ok_decode}")

    sub = a[R0:R1, C0:C1]
    print(f"\n=== 2. v5 SLICE rows[{R0}:{R1}] cols[{C0}:{C1}] ===")
    print(f"  shape {sub.shape}  expect (176, 624, 3): {sub.shape[:2] == (176, 624)}")
    print(f"  full   mean={a.mean():7.2f} std={a.std():6.2f} zerofrac={np.mean(a == 0):.4f}")
    print(f"  slice  mean={sub.mean():7.2f} std={sub.std():6.2f} zerofrac={np.mean(sub == 0):.4f}")

    # --- 3/4: rig masking, and does the slice dodge it? ---------------------
    print(f"\n=== 3. ROW-BAND PROFILE (a masked band reads as near-zero) ===")
    for r0 in range(0, 256, 16):
        band = a[r0:r0 + 16]
        mark = "  <-- inside v5 slice" if (r0 >= R0 and r0 + 16 <= R1) else ""
        print(f"    rows {r0:3d}-{r0+15:3d}: mean={band.mean():7.2f} zerofrac={np.mean(band == 0):.4f}{mark}")

    ro = geo.get("rig_observability", {})
    print(f"\n=== 4. _geometry.json rig_observability (population, 2400 clips) ===")
    for rig, v in ro.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"    rig {rig}: n={v.get('n')} mean={v.get('mean'):.6f} "
                  f"min={v.get('min'):.6f} max={v.get('max'):.6f}")
    gc = geo.get("geometry_check", {})
    print(f"    observed_frac (whole frame, 120 deg) = {gc.get('observed_frac')}")
    print(f"    => the FULL 256x640 frame masks ~{(1-float(gc.get('observed_frac',1)))*100:.2f}% on average;")
    print(f"       the v5 slice exists to sit inside the region BOTH rigs observe.")

    # --- 5: pose / action semantics ----------------------------------------
    p, ac = d["poses"], d["actions"]
    print(f"\n=== 5. POSES / ACTIONS ===")
    print(f"  poses  {tuple(p.shape)}")
    print(f"    first 3 rows: {np.array2string(p[:3].numpy(), precision=4)}")
    for i in range(p.shape[1]):
        c = p[:, i]
        print(f"    col{i}: min={float(c.min()):9.4f} max={float(c.max()):9.4f} "
              f"first={float(c[0]):9.4f} last={float(c[-1]):9.4f}")
    print(f"  actions {tuple(ac.shape)}")
    print(f"    first 3 rows: {np.array2string(ac[:3].numpy(), precision=4)}")
    for i in range(ac.shape[1]):
        c = ac[:, i]
        print(f"    col{i}: min={float(c.min()):9.4f} max={float(c.max()):9.4f} "
              f"mean={float(c.mean()):9.4f}")

    # displacement magnitude sanity: is col0/col1 metres at 10 Hz?
    if p.shape[1] >= 2:
        step = (p[1:, :2] - p[:-1, :2]).norm(dim=-1)
        print(f"\n  per-step |d(x,y)| : mean={float(step.mean()):.4f} m  max={float(step.max()):.4f} m")
        print(f"    at 10 Hz that is {float(step.mean())*10:.2f} m/s mean "
              f"({float(step.mean())*36:.1f} km/h) — plausible urban speed?")


if __name__ == "__main__":
    main()
