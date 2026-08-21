"""E-DETECT-1 prep — BEV occupancy targets + the RAW-PATCH floor.

Two banks, both keyed to the SAME 5,617 (clip_id, frame_idx) rows as every
E-TRUNK-2 probe, so the folds and the bootstrap stay comparable.

  occ.npy     (5617, 120) uint8   vehicle-centre occupancy on a 15 x 8 BEV grid
                                  (4 m x 4 m cells; x 0..60 m fwd, y -16..+16 m)
  pixels.npy  (5617, 640, 768) uint8
                                  the 256x640 frame cut into 16x16x3 patches on
                                  the SAME 16 x 40 grid the ViT tokenises.

⭐ WHY RAW PATCHES ARE THE RIGHT FLOOR, and better than a random ViT.
A random ViT is a random projection OF THESE PATCHES. The patches themselves are
strictly more informative and cost no forward pass — so `pixel` is an UPPER bound
on what any untrained encoder of this architecture could hand the head. If a
trained trunk does not beat it, the trunk added nothing the head can use; if
`pixel` itself scores at the prior, the head cannot draw the answer from
low-level structure and any lift by a trained arm is real.

⚠️ It also directly tests the one hypothesis E-V6SHAPE left standing — that the
next-latent objective is satisfied by low-level photometric structure. If `pixel`
localises vehicles well, "low-level suffices" stops being an excuse for the null.

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\sp2")
FEAT = SP / "e_trunk2_feat"
OUT = SP / "detect"
EPS = SP / "cache/slotprobe-lead130-w120-256x640cyl"
JOIN = SP / "lead130_agents.jsonl"

#: the v6 raster's own extent, so the grid is the model's grid and not a new one
X_MAX, Y_HALF = 60.0, 16.0
X_BINS, Y_BINS = 15, 8                      # 4 m x 4 m cells -> 120
N_CELL = X_BINS * Y_BINS
#: `obstacle.offline`'s vehicle classes (the enum's 10 are all dynamic agents)
VEHICLE = frozenset({"automobile", "heavy_truck", "bus", "trailer",
                     "other_vehicle"})
PATCH, GRID_H, GRID_W = 16, 16, 40
N_TOK = GRID_H * GRID_W                     # 640, matching v6 and DINOv3


def cell_of(cx: float, cy: float) -> int | None:
    """Grid index of a vehicle CENTRE, or None if outside the raster."""
    if not (0.0 <= cx < X_MAX) or not (-Y_HALF <= cy < Y_HALF):
        return None
    xi = int(cx / (X_MAX / X_BINS))
    yi = int((cy + Y_HALF) / (2 * Y_HALF / Y_BINS))
    return xi * Y_BINS + yi


def build_occupancy(keys: list[tuple[str, int]]) -> np.ndarray:
    want = {k: i for i, k in enumerate(keys)}
    occ = np.zeros((len(keys), N_CELL), dtype=np.uint8)
    seen = np.zeros(len(keys), dtype=bool)
    with JOIN.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            i = want.get((r["clip_id"], int(r["frame_idx"])))
            if i is None:
                continue
            seen[i] = True
            for a in r.get("agents", ()):
                if a.get("cls") not in VEHICLE:
                    continue
                c = cell_of(float(a["cx"]), float(a["cy"]))
                if c is not None:
                    occ[i, c] = 1
    if not seen.all():
        raise SystemExit(f"[FATAL] {(~seen).sum()} probe rows have no join row — "
                         "the target bank would silently contain all-zero rows, "
                         "which read as 'no vehicles' rather than 'no data'")
    return occ


def build_pixels(keys: list[tuple[str, int]]) -> np.ndarray:
    """Decode each frame and cut it into the ViT's own patch grid.

    ⚠️ `jpeg_buf` IS MISNAMED — these episodes carry `codec: "png"` (magic
    0x89 0x50). Decoding it as JPEG raises, and because the output is a
    pre-allocated memmap the job still leaves a full-size file of ZEROS behind.
    Read the `codec` field; never trust the buffer's name.
    """
    from PIL import Image
    by_clip: dict[str, list[tuple[int, int]]] = {}
    for i, (cid, f) in enumerate(keys):
        by_clip.setdefault(cid, []).append((f, i))
    out = np.lib.format.open_memmap(OUT / "pixels.npy", mode="w+",
                                    dtype=np.uint8,
                                    shape=(len(keys), N_TOK, PATCH * PATCH * 3))
    for n, (cid, rows) in enumerate(sorted(by_clip.items()), 1):
        d = torch.load(EPS / f"{cid}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        codec = str(d.get("codec", "")).lower()
        if codec not in ("png", "jpeg", "jpg"):
            raise SystemExit(f"[FATAL] {cid}: unhandled codec {codec!r}")
        buf, lens = d["jpeg_buf"], d["jpeg_len"].tolist()
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        raw = buf.numpy().tobytes()
        for f, i in rows:
            im = Image.open(io.BytesIO(raw[off[f]:off[f + 1]])).convert("RGB")
            a = np.asarray(im, dtype=np.uint8)              # (H, W, 3)
            if a.shape[:2] != (GRID_H * PATCH, GRID_W * PATCH):
                raise SystemExit(f"[FATAL] {cid} f{f}: frame is {a.shape[:2]}, "
                                 f"expected {(GRID_H * PATCH, GRID_W * PATCH)}")
            # (H,W,3) -> (gh,gw,16,16,3) -> (640,768), row-major exactly as a
            # ViT flattens patches, so token j here IS token j there.
            out[i] = (a.reshape(GRID_H, PATCH, GRID_W, PATCH, 3)
                       .transpose(0, 2, 1, 3, 4)
                       .reshape(N_TOK, PATCH * PATCH * 3))
        print(f"  [{n:>3}/{len(by_clip)}] {cid[:8]} {len(rows):>3} frames",
              flush=True)
    out.flush()
    # ⛔ VERIFY BY CONTENT, NOT BY PRESENCE. The first build of this bank exited
    # 0 and left 2.76 GB of zeros because the decode raised into a
    # pre-allocated memmap. A file that exists is not a file that is right.
    dead = int((np.asarray(out[::200]).reshape(-1, N_TOK * PATCH * PATCH * 3)
                .max(1) == 0).sum())
    if dead:
        raise SystemExit(f"[FATAL] {dead} sampled rows are all-zero — the bank "
                         "is not usable")
    print(f"  content check OK: sampled rows all non-zero, "
          f"mean {float(np.asarray(out[::200]).mean()):.2f}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [tuple(k) for k in
            json.loads((FEAT / "keys.json").read_text(encoding="utf-8"))]

    occ = build_occupancy(keys)
    np.save(OUT / "occ.npy", occ)
    per = occ.sum(1)
    stats = {
        "_evidence_class": "MEASURED (ours; lead130_agents.jsonl cuboid join)",
        "n_rows": len(keys), "n_clips": len({k[0] for k in keys}),
        "grid": {"x_max_m": X_MAX, "y_half_m": Y_HALF,
                 "x_bins": X_BINS, "y_bins": Y_BINS, "n_cell": N_CELL,
                 "cell_m": [X_MAX / X_BINS, 2 * Y_HALF / Y_BINS]},
        "occupied_cells_per_frame": {"mean": round(float(per.mean()), 4),
                                     "max": int(per.max()),
                                     "min": int(per.min())},
        "base_rate": round(float(occ.mean()), 6),
        "frames_with_zero_occupied": int((per == 0).sum()),
        "frac_frames_zero": round(float((per == 0).mean()), 5),
        "cells_never_occupied": int((occ.sum(0) == 0).sum()),
    }
    (OUT / "occ_stats.json").write_text(json.dumps(stats, indent=1),
                                        encoding="utf-8")
    print(json.dumps(stats, indent=1))

    if "--pixels" in sys.argv:
        print("\nbuilding raw-patch floor ...", flush=True)
        build_pixels(keys)
        print(f"-> {OUT / 'pixels.npy'}")


if __name__ == "__main__":
    main()
