"""WHERE do the surviving black pixels come from? — the residual, localised.

After the rig-clean slice the RAY-MAP mask is exactly 0 on both rigs, yet real
decoded pixels still show a rig-correlated all-zero fraction. Two candidate
mechanisms, which this separates:

  M1 SENSOR-BLACK (a real geometry residual): the f-theta lens' image CIRCLE does
     not fill the 1920x1080 rectangle, so pixels the rectangle test admits are
     nonetheless black IN THE SOURCE. Signature: the same pixels are zero in
     (nearly) EVERY frame of a clip -> per-pixel always_zero_frac ~ 1.
  M2 SCENE-BLACK (not a geometry problem): night, tunnels, dark vehicles.
     Signature: zeros scattered over frames, per-pixel always_zero_frac ~ 0, and
     the zero set moves between frames.

Reports, per clip: the persistent-zero mask (zero in >= 95 % of frames), its
bounding box in the PARENT frame, the (theta, phi) of its extreme pixel, and how
much of the sliced frame it covers. That last number is the honest residual of
the fix on real pixels.

🔒 No clip UUID in the artifact.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import torch


def decode_built(path: str):
    import torchvision.io as tvio
    d = torch.load(path, map_location="cpu", weights_only=False)
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    buf = d["jpeg_buf"]
    dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
    return torch.stack([dec(buf[int(offs[i]):int(offs[i + 1])],
                            mode=tvio.ImageReadMode.RGB)
                        for i in range(len(lens))]), d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--persist", type=float, default=0.95)
    ap.add_argument("--sub-h", type=int, default=176)
    ap.add_argument("--sub-w", type=int, default=624)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tanitad.data.calib import (CanonicalFrame, centred_subframe,
                                    cylindrical_grid, subframe_slice)
    from tanitad.data.physicalai import intrinsics_for_clip

    geom = json.loads((Path(a.cache) / "_geometry.json").read_text())
    parent = CanonicalFrame.from_dict(geom["frame"])
    H, W, f = parent.height, parent.width, float(parent.f_ref)
    sub = centred_subframe(parent, a.sub_h, a.sub_w)
    rs, cs = subframe_slice(parent, sub)

    files = sorted(Path(a.cache).glob("*.v2ep.pt"))
    rig_of, intr_of = {}, {}
    for i, p in enumerate(files):
        try:
            it = intrinsics_for_clip(p.name.replace(".v2ep.pt", ""), a.root)
        except Exception:                                     # noqa: BLE001
            continue
        if it.per_clip:
            rig_of[i] = "B" if it.cy >= 650.872 else "A"
            intr_of[i] = it
    byrig = {rg: [i for i in rig_of if rig_of[i] == rg] for rg in ("A", "B")}
    pick = []
    for rg in ("A", "B"):
        lst, want = byrig[rg], min(a.n // 2, len(byrig[rg]))
        pick += [lst[round(j * (len(lst) - 1) / max(want - 1, 1))]
                 for j in range(want)]
    pick = sorted(set(pick))

    out = {"measured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cache": a.cache, "parent": parent.tag(),
           "persist_threshold": a.persist, "clips": []}
    # union of persistent-zero masks, per rig, to find the largest clean frame
    union = {"A": torch.zeros(H, W, dtype=torch.bool),
             "B": torch.zeros(H, W, dtype=torch.bool)}
    for i in pick:
        V, _ = decode_built(str(files[i]))
        z = (V == 0).all(dim=1)                              # [n,H,W]
        frac_zero = z.float().mean(0)                        # [H,W]
        persist = frac_zero >= a.persist
        _, m = cylindrical_grid(intr_of[i], int(intr_of[i].height),
                                int(intr_of[i].width), parent)
        raymap_masked = ~m
        extra = persist & ~raymap_masked                     # M1 candidate
        union[rig_of[i]] |= extra
        rec = {"i": i, "rig": rig_of[i], "frames": int(V.shape[0]),
               "zero_frac_all": float(z.float().mean()),
               "persistent_zero_frac": float(persist.float().mean()),
               "raymap_masked_frac": float(raymap_masked.float().mean()),
               "extra_persistent_frac": float(extra.float().mean()),
               "transient_zero_frac": float(
                   (z & ~persist.unsqueeze(0)).float().mean()),
               # ⭐ the SAME quantities inside the rig-clean slice — the residual
               "sub_zero_frac": float(z[:, rs, cs].float().mean()),
               "sub_raymap_masked_frac": float(
                   raymap_masked[rs, cs].float().mean()),
               "sub_persistent_zero_frac": float(
                   persist[rs, cs].float().mean()),
               "sub_extra_persistent_frac": float(extra[rs, cs].float().mean()),
               "sub_transient_zero_frac": float(
                   (z[:, rs, cs] & ~persist[rs, cs].unsqueeze(0))
                   .float().mean())}
        if extra.any():
            idx = torch.nonzero(extra)
            r0, r1 = int(idx[:, 0].min()), int(idx[:, 0].max())
            c0, c1 = int(idx[:, 1].min()), int(idx[:, 1].max())
            rec["extra_bbox_rows"] = [r0, r1]
            rec["extra_bbox_cols"] = [c0, c1]
            # the extreme pixel's ray angles
            phis = ((idx[:, 1].float() - (W - 1) / 2) / f)
            vs = ((idx[:, 0].float() - (H - 1) / 2) / f)
            rho = (phis.sin() ** 2 + vs ** 2).sqrt()
            th = torch.atan2(rho, phis.cos())
            rec["extra_theta_deg"] = [float(math.degrees(th.min())),
                                      float(math.degrees(th.max()))]
            rec["extra_abs_phi_deg_max"] = float(
                math.degrees(phis.abs().max()))
            rec["extra_row_hist_top10"] = [
                int(x) for x in torch.nonzero(extra.any(1)).flatten()[:10]]
        out["clips"].append(rec)
        print("[diag] " + json.dumps(rec)[:400], flush=True)

    def agg(rows, k):
        v = [r[k] for r in rows if k in r]
        return None if not v else {
            "n": len(v), "mean": st.mean(v), "median": st.median(v),
            "min": min(v), "max": max(v)}

    out["by_rig"] = {
        rg: {k: agg([r for r in out["clips"] if r["rig"] == rg], k)
             for k in ("zero_frac_all", "persistent_zero_frac",
                       "raymap_masked_frac", "extra_persistent_frac",
                       "transient_zero_frac", "sub_zero_frac",
                       "sub_raymap_masked_frac", "sub_persistent_zero_frac",
                       "sub_extra_persistent_frac", "sub_transient_zero_frac")}
        for rg in ("A", "B")}
    # the union mask per rig: the frame that would be clean of BOTH
    for rg in ("A", "B"):
        u = union[rg]
        out.setdefault("union", {})[rg] = {
            "frac": float(u.float().mean()),
            "rows_any": [int(x) for x in
                         torch.nonzero(u.any(1)).flatten().tolist()][:40],
            "cols_any_count": int(u.any(0).sum()),
        }
    # largest CENTRED subframe with zero persistent-black, both rigs
    both = union["A"] | union["B"]
    best = None
    for h in range(H, 15, -16):
        if (H - h) % 2:
            continue
        r0 = (H - h) // 2
        for w in range(W, 15, -16):
            if (W - w) % 2:
                continue
            c0 = (W - w) // 2
            if not bool(both[r0:r0 + h, c0:c0 + w].any()):
                if best is None or h * w > best[0] * best[1]:
                    best = (h, w)
    out["largest_centred_free_of_persistent_black"] = best
    out["sub_frame"] = {**sub.to_dict(), "tag": sub.tag(),
                        "rows": [rs.start, rs.stop], "cols": [cs.start, cs.stop]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("ZERO_DIAG_DONE " + json.dumps({
        "by_rig_extra": {rg: out["by_rig"][rg]["extra_persistent_frac"]
                         for rg in ("A", "B")},
        "by_rig_transient": {rg: out["by_rig"][rg]["transient_zero_frac"]
                             for rg in ("A", "B")},
        "largest_clean": best,
        "SUB_residual": {rg: {k: out["by_rig"][rg][k]
                              for k in ("sub_raymap_masked_frac",
                                        "sub_extra_persistent_frac",
                                        "sub_transient_zero_frac")}
                         for rg in ("A", "B")}}), flush=True)


if __name__ == "__main__":
    main()
