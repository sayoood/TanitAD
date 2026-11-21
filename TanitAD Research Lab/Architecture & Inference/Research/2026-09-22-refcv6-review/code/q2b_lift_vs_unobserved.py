"""Q2b — with REAL per-clip extrinsics: how much of the BEV grid does the lift
fill at 416x1024, and how many of those cells sample the UNOBSERVED black strip?

⛔ THE SEAM. `bev_lift.build_lift_geometry` takes an optional `observed` mask and
its own docstring names the reason (`bev_lift.py:32-35`: *"optionally AND an
observed-pixel mask of the cache frame (rig B leaves ~8.9 % of the 256x640 frame
black)"*). `refcv6_perception_branch.LiftGeometryBank.geometry`
(`refcv6_perception_branch.py:271-272`) calls it with **four** keyword arguments
and `observed` is not one of them — so in the refcv6 training path the option is
dead, and a BEV cell whose pixel lands in the black strip is marked VALID.

Consequence if non-zero: those cells are filled by `F.grid_sample` from trunk
features computed over a CONSTANT black region (which ImageNet normalisation maps
to -2.118 / -2.036 / -1.804), while the module's designed handling — the learned
`unobserved` embedding, `bev_lift.py:269-270` — fires only where NO height is
valid, i.e. never for them.

Evidence classes: the extrinsics are the dataset's own `sensor_extrinsics`; the
unobserved mask is measured FROM THE DECODED PIXELS of the cache the trainer
reads, not from a re-derivation of the builder's own grid.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch
import torchvision.io as tvio

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
EXTR = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/"
        "Research/2026-09-06-refcv4b-landing/raw/extrinsics141.json")


def main() -> int:
    from tanitad.models.bev_lift import build_lift_geometry
    from tanitad.models.trunk_shapes import FRAME_416x1024 as F416

    extr = json.load(open(EXTR))
    paths = sorted(glob.glob(CACHE + "/*.v2ep.pt"))
    rows = []
    n_missing_extr = 0
    for p in paths:
        cid = os.path.basename(p).split(".")[0]
        e = extr.get(cid)
        if e is None:
            n_missing_extr += 1
            continue
        pose = {k: float(e[k]) for k in ("qx", "qy", "qz", "qw", "x", "y", "z")}

        d = torch.load(p, map_location="cpu", weights_only=False)
        lens, buf = d["jpeg_len"], d["jpeg_buf"]
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(lens, 0)])
        dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
        f = dec(buf[0:int(offs[1])], mode=tvio.ImageReadMode.RGB)
        # ⭐ a row that is EXACTLY zero across the whole width in all 3 channels
        # is unobserved; a real scene row is never exactly that. Conservative on
        # purpose: it undercounts the black region, so the numbers below are a
        # LOWER bound on the defect.
        black_row = (f == 0).all(dim=0).all(dim=1)       # [H] bool

        g = build_lift_geometry(pose, frame=F416, stride=16)
        valid = g.valid                                   # [Z, X, Y]
        ri = g.row.round().long().clamp(0, F416.height - 1)
        in_black = black_row[ri] & valid
        cell_valid = valid.any(dim=0)
        cell_black_only = (in_black.any(dim=0)) & cell_valid
        rows.append({
            "clip_id": cid,
            "n_black_rows": int(black_row.sum()),
            "cell_valid_frac": round(float(cell_valid.float().mean()), 6),
            "n_cells_valid": int(cell_valid.sum()),
            "n_samples_valid": int(valid.sum()),
            "n_samples_valid_in_black": int(in_black.sum()),
            "n_cells_touching_black": int(cell_black_only.sum()),
        })
    if not rows:
        print(json.dumps({"INCONCLUSIVE": "no clip matched the extrinsics table",
                          "n_missing_extr": n_missing_extr}))
        return 1
    t = lambda k: torch.tensor([float(r[k]) for r in rows])   # noqa: E731
    out = {
        "n_clips": len(rows), "n_missing_extr": n_missing_extr,
        "grid_cells_total": 120 * 64,
        "samples_total_per_clip": 4 * 120 * 64,
        "cell_valid_frac": {"mean": round(float(t("cell_valid_frac").mean()), 6),
                            "min": round(float(t("cell_valid_frac").min()), 6),
                            "max": round(float(t("cell_valid_frac").max()), 6)},
        "n_cells_valid": {"mean": round(float(t("n_cells_valid").mean()), 1),
                          "min": int(t("n_cells_valid").min()),
                          "max": int(t("n_cells_valid").max())},
        "clips_with_black_rows": int((t("n_black_rows") > 0).sum()),
        "n_black_rows": {"mean": round(float(t("n_black_rows").mean()), 2),
                         "max": int(t("n_black_rows").max())},
        "n_cells_touching_black": {
            "mean": round(float(t("n_cells_touching_black").mean()), 2),
            "max": int(t("n_cells_touching_black").max()),
            "clips_with_any": int((t("n_cells_touching_black") > 0).sum())},
        "frac_of_valid_cells_touching_black_mean": round(float(
            (t("n_cells_touching_black") /
             t("n_cells_valid").clamp(min=1)).mean()), 6),
        # ⭐ CONTROL: the option IS wired in build_lift_geometry -- prove it by
        # passing the mask and watching the count fall.
    }

    # CONTROL / the fix, measured: pass `observed` for one affected clip
    aff = max(rows, key=lambda r: r["n_cells_touching_black"])
    p = os.path.join(CACHE, aff["clip_id"] + ".v2ep.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                      torch.cumsum(d["jpeg_len"], 0)])
    dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
    f = dec(d["jpeg_buf"][0:int(offs[1])], mode=tvio.ImageReadMode.RGB)
    obs = ~((f == 0).all(dim=0))                          # [H, W] bool observed
    pose = {k: float(extr[aff["clip_id"]][k])
            for k in ("qx", "qy", "qz", "qw", "x", "y", "z")}
    g_no = build_lift_geometry(pose, frame=F416, stride=16)
    g_yes = build_lift_geometry(pose, frame=F416, stride=16, observed=obs)
    out["CONTROL_observed_mask_changes_the_answer"] = {
        "clip_id": aff["clip_id"],
        "n_samples_valid_without_mask": int(g_no.valid.sum()),
        "n_samples_valid_with_mask": int(g_yes.valid.sum()),
        "n_cells_valid_without_mask": int(g_no.valid.any(dim=0).sum()),
        "n_cells_valid_with_mask": int(g_yes.valid.any(dim=0).sum()),
        "DIFFERS": int(g_no.valid.sum()) != int(g_yes.valid.sum()),
    }
    out["per_clip"] = rows
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
