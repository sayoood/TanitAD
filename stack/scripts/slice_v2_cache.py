"""Re-emit a v2 episode cache at a CENTRED sub-frame — WITHOUT rebuilding.

⭐ WHY THIS EXISTS. Retraction class C26: the deployed input carries fabricated,
rig-correlated rows. The clean fix is a field BOTH rigs fully observe. Measured
over all 3,000 clips of the canonical selection, that field is
``176x624`` — and it is a CENTRED sub-rectangle of the ``256x640`` / 120 deg /
cylindrical frame the v5 corpus was already built at.

A centred sub-frame is a pure PIXEL SLICE of its parent (see
``calib.centred_subframe`` for the exact-float argument, and
``tests/test_rig_clean_fix.py`` for the bit-exactness proof on real per-clip
intrinsics). So the fix costs a decode+re-encode of an existing cache, not a
re-download and re-decode of ~374 GB of source video:

    MEASURED on pod2, 256x640 PNG -> 176x624 PNG, single process:
      ~1.4 s/clip  vs  19.4 s/clip for a rebuild from mp4  (13.9x)
      and no HF egress at all.

⛔ THE LOSSLESS PRECONDITION. The slice is bit-exact only for ``codec="png"``.
On a JPEG cache the re-encode lands on different 8x8 block boundaries, so this
script REFUSES a lossy source unless ``--allow-lossy`` is passed, and records the
fact in the manifest either way.

⚠️ WHAT THIS DOES NOT DO. It re-emits pixels; it does not re-select episodes.
The output holds exactly the clip ids of the input (parity is preserved by
construction — there is no selection step to get wrong), and the manifest records
the parent's own provenance so the chain back to the parity split is unbroken.

usage:
  python slice_v2_cache.py --src <cache> --out <cache> --height 176 --width 624 \
      [--root <corpus root, enables the per-rig observability declaration>] \
      [--shard i/K] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torchvision.io as tvio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tanitad.data.calib import (CanonicalFrame, centred_subframe,  # noqa: E402
                                observed_report, subframe_slice)


def _decode(d) -> torch.Tensor:
    """The stored frames, decoded — [n, 3, H, W] uint8."""
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    buf = d["jpeg_buf"]
    dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
    return torch.stack([dec(buf[int(offs[i]):int(offs[i + 1])],
                            mode=tvio.ImageReadMode.RGB)
                        for i in range(len(lens))])


def slice_one(src_path: str, out_path: str, sub: CanonicalFrame,
              quality: int = 90) -> tuple[int, str]:
    """Re-emit one ``.v2ep.pt`` at ``sub``. Returns (bytes, frames digest)."""
    d = torch.load(src_path, map_location="cpu", weights_only=False)
    parent = CanonicalFrame.from_dict(d["frame"])
    rs, cs = subframe_slice(parent, sub)
    vid = _decode(d)[:, :, rs, cs].contiguous()
    enc = (tvio.encode_png if d.get("codec") == "png" else
           (lambda x: tvio.encode_jpeg(x, quality=quality)))
    parts = [enc(vid[i].contiguous()) for i in range(vid.shape[0])]
    lens = torch.tensor([int(p.numel()) for p in parts], dtype=torch.int64)
    buf = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.uint8)
    out = dict(d)
    out.update({"jpeg_buf": buf, "jpeg_len": lens,
                "image_size": int(sub.height), "image_h": int(sub.height),
                "image_w": int(sub.width), "frame": sub.to_dict(),
                # provenance: this payload is a SLICE, and of what
                "sliced_from": {"frame": parent.to_dict(),
                                "tag": parent.tag(),
                                "rows": [rs.start, rs.stop],
                                "cols": [cs.start, cs.stop]}})
    tmp = out_path + ".tmp"                # atomic: a kill must not leave a
    torch.save(out, tmp)                   # half-written .pt behind
    os.replace(tmp, out_path)
    return int(buf.numel()), hashlib.sha256(vid.numpy().tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--root", default="",
                    help="corpus root — enables the per-rig observability "
                         "declaration (needs calibration/camera_intrinsics)")
    ap.add_argument("--rig-sample", type=int, default=60,
                    help="clips to probe for the observability declaration")
    ap.add_argument("--shard", default="", help="i/K")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-lossy", action="store_true")
    ap.add_argument("--corpus-role", default="",
                    help="what the sliced corpus IS (parity.py §10c)")
    ap.add_argument("--exclude-parity-overlap", action="store_true",
                    help="drop the disqualifying clips instead of refusing")
    ap.add_argument("--sanctioned-audit", default="",
                    help="REASON for keeping a disqualifying overlap")
    ap.add_argument("--digest-n", type=int, default=16,
                    help="clips whose sliced pixels are hashed into the "
                         "manifest — the check a consumer can actually run")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.src) if f.endswith(".v2ep.pt"))
    # ⛔ PARITY INGEST GATE (parity.py §10c). A slice EMITS A NEW CORPUS — it is
    # a build that never touches the source video, so it bypasses
    # v2_compressed.build entirely and would inherit any contamination in --src
    # under a new name. The uid space here IS the clip id (`<clip>.v2ep.pt`).
    from tanitad.data import parity                          # noqa: PLC0415
    parity.require_ingest_gate("slice_v2_cache")
    _keep, _gate = parity.guard_corpus_build(
        [f[:-len(".v2ep.pt")] for f in files],
        label=f"slice_v2_cache {a.src} -> {a.out}", role=a.corpus_role,
        mode="exclude" if a.exclude_parity_overlap else "refuse",
        sanctioned_audit=a.sanctioned_audit or None)
    _keepset = set(_keep)
    files = [f for f in files if f[:-len(".v2ep.pt")] in _keepset]
    if a.shard:
        i, k = (int(x) for x in a.shard.split("/"))
        files = [f for n, f in enumerate(files) if n % k == i]
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    src_geom = {}
    gp = os.path.join(a.src, "_geometry.json")
    if os.path.exists(gp):
        src_geom = json.load(open(gp))
    parent = CanonicalFrame.from_dict(
        src_geom["frame"] if "frame" in src_geom
        else torch.load(os.path.join(a.src, files[0]), map_location="cpu",
                        weights_only=False)["frame"])
    sub = centred_subframe(parent, a.height, a.width or parent.width)
    codec = src_geom.get("codec")
    if codec is None and files:
        codec = torch.load(os.path.join(a.src, files[0]), map_location="cpu",
                           weights_only=False).get("codec")
    if codec != "png" and not a.allow_lossy:
        raise SystemExit(
            f"REFUSING: source codec is {codec!r}, not 'png'. A slice is "
            f"bit-exact only on a LOSSLESS cache — re-encoding JPEG at a new "
            f"crop offset moves the 8x8 blocks and the result is NOT the frames "
            f"a rebuild would produce. Pass --allow-lossy to accept that (and it "
            f"will be recorded in the manifest), or rebuild from source.")

    manifest = {
        "measured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frame": sub.to_dict(), "frame_tag": sub.tag(),
        "projection_mode": ("cylindrical" if sub.projection == "cylindrical"
                            else "ftheta_crop"),
        "codec": codec, "bit_exact_slice": codec == "png",
        "produced_by": "slice_v2_cache.py",
        "sliced_from": {**src_geom, "dir": os.path.abspath(a.src)},
        "geometry_check": {
            "requested_hfov_deg": round(float(sub.hfov_deg), 4),
            "vfov_deg": round(float(sub.vfov_deg), 4),
            "f_eff": round(float(sub.f_ref), 4),
            "n_tokens_at_patch16": (sub.height // 16) * (sub.width // 16)},
    }

    # ---- the DECLARATION a consumer can check: per-rig observability -------
    if a.root:
        from tanitad.data.physicalai import intrinsics_for_clip
        rows = []
        step = max(1, len(files) // max(a.rig_sample, 1))
        for f in files[::step][:a.rig_sample]:
            try:
                it = intrinsics_for_clip(f.replace(".v2ep.pt", ""), a.root)
            except Exception:                                 # noqa: BLE001
                continue
            if not it.per_clip:
                continue
            rows.append((("B" if it.cy >= 650.872 else "A"),
                         observed_report(it, sub)["masked_frac"],
                         observed_report(it, parent)["masked_frac"]))
        if rows:
            manifest["rig_observability"] = {
                rg: {"n": sum(1 for r in rows if r[0] == rg),
                     "masked_frac_max": max(
                         [r[1] for r in rows if r[0] == rg], default=None),
                     "masked_frac_max_parent": max(
                         [r[2] for r in rows if r[0] == rg], default=None)}
                for rg in ("A", "B")}
            manifest["rig_observability"]["fully_observed_by_all_sampled"] = \
                max(r[1] for r in rows) == 0.0

    t0, nb, digests = time.time(), 0, {}
    for n, f in enumerate(files):
        op = os.path.join(a.out, f)
        if os.path.exists(op):
            continue
        b, dg = slice_one(os.path.join(a.src, f), op, sub)
        nb += b
        if len(digests) < a.digest_n:
            digests[n] = dg
        if (n + 1) % 100 == 0:
            print(f"[slice] {n + 1}/{len(files)} {nb / 2**30:.2f} GB "
                  f"{time.time() - t0:.0f}s", flush=True)
    manifest["pixel_digest_sha256"] = digests       # index -> sha256 of frames
    manifest["clips_written"] = len(files)
    manifest["bytes"] = nb
    manifest["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "_geometry.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("SLICE_DONE " + json.dumps({
        "clips": len(files), "GB": round(nb / 2**30, 2),
        "seconds": manifest["seconds"], "frame": sub.tag(),
        "bit_exact_slice": manifest["bit_exact_slice"],
        "rig_observability": manifest.get("rig_observability")}), flush=True)


if __name__ == "__main__":
    main()
