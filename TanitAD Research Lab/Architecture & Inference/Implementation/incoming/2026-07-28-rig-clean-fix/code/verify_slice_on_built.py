"""Verify the rig-clean fix ON THE FRAMES ACTUALLY BUILT — not on a config.

Three questions, on real decoded pixels of the real cache:

  Q1 SLICE-vs-REBUILD. Rebuild a clip from its own mp4 at the rig-clean frame and
     compare, pixel for pixel, with the ROW/COLUMN SLICE of the frames already in
     the cache. Bit-identical => the ~3.5 h build is not wasted.
  Q2 RESIDUAL. The masked/fabricated fraction of the sliced frames, per rig, on
     real decoded pixels. Must be 0.000000 on BOTH rigs, reported to 6 dp.
  Q3 PREDICTION. Does the ray-map mask predict the genuinely-zero pixels of the
     real frame? (The real count is an upper bound: night scenes have real black
     pixels.)

🔒 No clip UUID is written to the artifact — clips carry an index only.
"""
from __future__ import annotations
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch


def decode_built(path: str):
    """The cache's own frames, decoded — [n, 3, H, W] uint8 — plus its metadata."""
    import torchvision.io as tvio
    d = torch.load(path, map_location="cpu", weights_only=False)
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    buf = d["jpeg_buf"]
    dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
    frames = [dec(buf[int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)
              for i in range(len(lens))]
    return torch.stack(frames), d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--height", type=int, default=176)
    ap.add_argument("--width", type=int, default=624)
    ap.add_argument("--n-rebuild", type=int, default=6,
                    help="clips to REBUILD from mp4 (the expensive leg)")
    ap.add_argument("--n-residual", type=int, default=120,
                    help="clips to decode from the cache for the residual")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tanitad.data.calib import (CanonicalFrame, centred_subframe,
                                    cylindrical_grid, subframe_slice)
    from tanitad.data.physicalai import intrinsics_for_clip

    geom = json.loads((Path(a.cache) / "_geometry.json").read_text())
    parent = CanonicalFrame.from_dict(geom["frame"])
    sub = centred_subframe(parent, a.height, a.width)
    rs, cs = subframe_slice(parent, sub)

    files = sorted(p for p in Path(a.cache).glob("*.v2ep.pt"))
    out: dict = {
        "measured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache": a.cache, "codec": geom.get("codec"),
        "parent": {**parent.to_dict(), "tag": parent.tag()},
        "sub": {**sub.to_dict(), "tag": sub.tag(),
                "hfov_deg": float(sub.hfov_deg), "vfov_deg": float(sub.vfov_deg)},
        "slice": {"rows": [rs.start, rs.stop], "cols": [cs.start, cs.stop]},
        "n_files": len(files),
        "rebuild": [], "residual": [],
    }

    # ---- rig of every candidate clip, from its own intrinsics --------------
    rig_of, intr_of = {}, {}
    for i, p in enumerate(files):
        cid = p.name.replace(".v2ep.pt", "")
        try:
            it = intrinsics_for_clip(cid, a.root)
        except Exception:                                     # noqa: BLE001
            continue
        if not it.per_clip:
            continue
        rig_of[i] = "B" if it.cy >= 650.872 else "A"
        intr_of[i] = it
    byrig = {"A": [i for i in rig_of if rig_of[i] == "A"],
             "B": [i for i in rig_of if rig_of[i] == "B"]}
    out["rig_counts_in_cache"] = {k: len(v) for k, v in byrig.items()}

    def pick(n):
        got = []
        for rg in ("A", "B"):
            lst = byrig[rg]
            want = min(n // 2, len(lst))
            if want:
                got += [lst[round(j * (len(lst) - 1) / max(want - 1, 1))]
                        for j in range(want)]
        return sorted(set(got))

    # ---- Q1: rebuild from mp4, compare with the slice ----------------------
    from tanitad.data.physicalai import discover_r0_clips
    clips = {c["clip_id"]: c for c in discover_r0_clips(a.root)}
    sys.path.insert(0, "/workspace/wfov")
    import importlib.util
    v2c, spec = None, None
    for cand in ("/workspace/rigfix/stack_head/scripts/v2_compressed.py",
                 "/workspace/wfov/stack_head/scripts/v2_compressed.py",
                 "/workspace/wfov/stack_v5/scripts/v2_compressed.py"):
        if Path(cand).exists():
            spec = importlib.util.spec_from_file_location("v2c", cand)
            v2c = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(v2c)
            out["v2_compressed_from"] = cand
            break

    for i in pick(a.n_rebuild):
        p = files[i]
        cid = p.name.replace(".v2ep.pt", "")
        rec = {"i": i, "rig": rig_of[i]}
        if v2c is None or cid not in clips:
            rec["skipped"] = "no mp4 or no v2_compressed"
            out["rebuild"].append(rec)
            continue
        t0 = time.time()
        V, meta = decode_built(str(p))
        rec["built_shape"] = list(V.shape)
        vid, _, _ = v2c._resampled(clips[cid], a.height, sub, "cylindrical")
        rec["rebuild_shape"] = list(vid.shape)
        n = min(vid.shape[0], V.shape[0])
        sl = V[:n, :, rs, cs]
        rec["n_frames_compared"] = int(n)
        rec["bit_identical"] = bool(torch.equal(vid[:n], sl))
        rec["max_abs_diff"] = int((vid[:n].int() - sl.int()).abs().max())
        rec["n_pixels_differing"] = int(
            (vid[:n] != sl).any(dim=1).sum())
        rec["seconds"] = round(time.time() - t0, 2)
        out["rebuild"].append(rec)
        print("[rebuild] " + json.dumps(rec), flush=True)

    # ---- Q2/Q3: residual on real decoded pixels ----------------------------
    for i in pick(a.n_residual):
        p = files[i]
        rec = {"i": i, "rig": rig_of[i]}
        try:
            V, meta = decode_built(str(p))
        except Exception as e:                                # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            out["residual"].append(rec)
            continue
        zero_parent = (V == 0).all(dim=1)                     # [n,H,W]
        sl = V[:, :, rs, cs]
        zero_sub = (sl == 0).all(dim=1)
        rec["zero_frac_parent"] = float(zero_parent.float().mean())
        rec["zero_frac_sub"] = float(zero_sub.float().mean())
        _, mp_ = cylindrical_grid(intr_of[i], int(intr_of[i].height),
                                  int(intr_of[i].width), parent)
        rec["raymap_masked_parent"] = float(1.0 - mp_.float().mean())
        rec["raymap_masked_sub"] = float(1.0 - mp_[rs, cs].float().mean())
        # Q3: are the real zeros inside the predicted mask?
        pred = (~mp_).unsqueeze(0).expand_as(zero_parent)
        rec["zeros_outside_predicted_mask_frac"] = float(
            (zero_parent & ~pred).float().mean())
        rec["frames"] = int(V.shape[0])
        out["residual"].append(rec)

    def agg(rows, key):
        v = [r[key] for r in rows if key in r]
        if not v:
            return None
        return {"n": len(v), "mean": st.mean(v), "median": st.median(v),
                "min": min(v), "max": max(v)}

    res = [r for r in out["residual"] if "zero_frac_sub" in r]
    out["residual_by_rig"] = {
        rg: {k: agg([r for r in res if r["rig"] == rg], k)
             for k in ("zero_frac_parent", "zero_frac_sub",
                       "raymap_masked_parent", "raymap_masked_sub",
                       "zeros_outside_predicted_mask_frac")}
        for rg in ("A", "B")}
    reb = [r for r in out["rebuild"] if "bit_identical" in r]
    out["verdict"] = {
        "n_rebuild_compared": len(reb),
        "all_rebuilds_bit_identical": bool(reb) and all(
            r["bit_identical"] for r in reb),
        "residual_sub_max_A": (out["residual_by_rig"]["A"]["zero_frac_sub"]
                               or {}).get("max"),
        "residual_sub_max_B": (out["residual_by_rig"]["B"]["zero_frac_sub"]
                               or {}).get("max"),
        "raymap_sub_max_A": (out["residual_by_rig"]["A"]["raymap_masked_sub"]
                             or {}).get("max"),
        "raymap_sub_max_B": (out["residual_by_rig"]["B"]["raymap_masked_sub"]
                             or {}).get("max"),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("VERIFY_SLICE_DONE " + json.dumps(out["verdict"]), flush=True)


if __name__ == "__main__":
    main()
