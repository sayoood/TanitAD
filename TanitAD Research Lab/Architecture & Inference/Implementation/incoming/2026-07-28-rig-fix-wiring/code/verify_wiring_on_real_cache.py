"""Prove the WIRED trainer seam delivers the rig-clean frame — on REAL frames.

Run on pod2, where both v5 caches live. Nothing here reads a config for its
answer: every raster below is decoded from the bytes on disk.

Legs, each able to fail:

  L1  the payloads really are 256x640 and really are PNG (the precondition).
  L2  the trainer's own seam (``resolve_v2_frames`` -> ``build_v2_data``) hands
      back providers whose raster is the SUB-frame.
  L3  those pixels are BIT-IDENTICAL to ``load_compressed(path, frame=…)`` —
      the already-verified reference implementation of the same slice.
  L4  and BIT-IDENTICAL to the parent decoded whole and sliced by hand, which
      is the claim "a centred sub-frame is a pure pixel slice" itself.
  L5  NEG: with the frame withheld from the loader, the trainer's geometry
      binding REFUSES the launch instead of training on the parent.
  L6  timing: how much the slice costs per clip.

Usage:
  PYTHONPATH=<stack> python3 verify_wiring_on_real_cache.py \
      --train <cache dir> --val <cache dir> --n 4 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--stack", default=os.environ.get(
        "TANITAD_STACK", "/workspace/TanitAD/stack"))
    ap.add_argument("--n", type=int, default=4, help="clips per split")
    ap.add_argument("--subframe", default="176x624")
    ap.add_argument("--scratch", default="/workspace/rigfix/wireprobe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    sys.path.insert(0, os.path.join(a.stack, "scripts"))
    import train_flagship_v4 as T
    import tanitad.data.v2_dataset as V2
    from tanitad.config import flagship4b_config
    from tanitad.data import parity
    from tanitad.data.calib import subframe_slice
    from v2_compressed import load_compressed

    out: dict = {"host": os.uname().nodename, "subframe": a.subframe,
                 "stack": a.stack, "legs": {}}

    # ---- a small REAL sub-cache: symlinks, so nothing is copied or rebuilt ---
    picks: dict[str, list[str]] = {}
    for split, src in (("train", a.train), ("val", a.val)):
        files = sorted(glob.glob(os.path.join(src, "*.v2ep.pt")))[:a.n]
        assert files, f"no *.v2ep.pt under {src}"
        d = os.path.join(a.scratch, split)
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*")):
            os.unlink(f)
        for f in files:
            os.symlink(f, os.path.join(d, os.path.basename(f)))
        picks[split] = files
        out.setdefault("dirs", {})[split] = {"source": src, "probe_dir": d,
                                             "n_clips_total": len(glob.glob(
                                                 os.path.join(src, "*.v2ep.pt"))),
                                             "n_probed": len(files)}

    # ---- L1: the precondition, from the bytes -------------------------------
    l1 = []
    for split, files in picks.items():
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
            l1.append({"split": split, "codec": str(d.get("codec")),
                       "frame": d.get("frame"),
                       "image_hw": [int(d.get("image_h", d["image_size"])),
                                    int(d.get("image_w", d["image_size"]))]})
    out["legs"]["L1_precondition"] = {
        "all_png": all(r["codec"] == "png" for r in l1),
        "all_256x640": all(r["image_hw"] == [256, 640] for r in l1),
        "n": len(l1), "rows": l1[:2]}
    assert out["legs"]["L1_precondition"]["all_png"], "a LOSSY cache — stop"

    # ---- L2: the trainer's OWN seam -----------------------------------------
    argv = ["--v2-train-cache", os.path.join(a.scratch, "train"),
            "--v2-val-cache", os.path.join(a.scratch, "val"),
            "--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120",
            "--projection", "cylindrical", "--v2-subframe", a.subframe,
            "--from-scratch", "--require-parity"]
    args = T.build_parser().parse_args(argv)
    cfg = flagship4b_config()
    cache_frame, frame = T.resolve_v2_frames(args, cfg)
    prov = {"train_parity": {}, "val_parity": {}}
    t0 = time.time()
    train_eps, val_eps = T.build_v2_data(args, prov, cache_frame=cache_frame,
                                         train_frame=frame, verbose=True)
    out["legs"]["L2_seam"] = {
        "cache_frame": cache_frame.to_dict(), "train_frame": frame.to_dict(),
        "encoder_image_hw": list(cfg.encoder.image_hw()),
        "token_grid": list(cfg.encoder.token_grid()),
        "n_tokens": int(cfg.encoder.token_grid()[0] * cfg.encoder.token_grid()[1]),
        "provider_shapes": sorted({tuple(int(x) for x in p.frames.shape)
                                   for p in (*train_eps, *val_eps)}).__str__(),
        "sliced_from": prov["geometry_binding"]["sliced_from"],
        "build_s": round(time.time() - t0, 3)}
    assert all(tuple(p.frames.shape[-2:]) == frame.hw
               for p in (*train_eps, *val_eps))

    # ---- L3/L4: the pixels, against two independent references --------------
    rs, cs = subframe_slice(cache_frame, frame)
    l3, l4, timing = [], [], []
    for split, eps in (("train", train_eps), ("val", val_eps)):
        for i, ep in enumerate(eps):
            path = picks[split][i]
            t = time.time(); parent_ep = load_compressed(path); t_par = time.time() - t
            t = time.time(); ref_ep = load_compressed(path, frame=frame); t_sub = time.time() - t
            n = min(int(ep.frames.shape[0]), int(ref_ep.frames.shape[0]))
            got = ep.frames[0:n]
            l3.append({"split": split, "n_frames": n,
                       "shape": list(got.shape),
                       "bit_identical_to_load_compressed_frame":
                           bool(torch.equal(got, ref_ep.frames[:n])),
                       "max_abs_diff": int((got.int() - ref_ep.frames[:n].int())
                                           .abs().max())})
            hand = parent_ep.frames[:n][:, :, rs, cs]
            l4.append({"split": split,
                       "parent_shape": list(parent_ep.frames[:n].shape),
                       "bit_identical_to_hand_slice_of_parent":
                           bool(torch.equal(got, hand)),
                       "n_pixels_differing": int((got != hand).sum())})
            timing.append({"split": split, "load_parent_s": round(t_par, 4),
                           "load_subframe_s": round(t_sub, 4)})
    out["legs"]["L3_vs_load_compressed"] = {
        "all_bit_identical": all(r["bit_identical_to_load_compressed_frame"]
                                 for r in l3),
        "max_abs_diff_over_all": max(r["max_abs_diff"] for r in l3),
        "rows": l3}
    out["legs"]["L4_vs_hand_slice_of_parent"] = {
        "all_bit_identical": all(r["bit_identical_to_hand_slice_of_parent"]
                                 for r in l4),
        "total_pixels_differing": sum(r["n_pixels_differing"] for r in l4),
        "rows": l4}

    # ---- L5: NEG — withhold the frame from the loader ------------------------
    real = V2.build_v2_providers
    V2.build_v2_providers = lambda dirs, **kw: real(
        dirs, **{k: v for k, v in kw.items() if k != "frame"})
    try:
        T.build_v2_data(args, {"train_parity": {}, "val_parity": {}},
                        cache_frame=cache_frame, train_frame=frame,
                        verbose=False)
        out["legs"]["L5_neg_frame_withheld"] = {
            "refused": False, "ERROR": "the launch was NOT refused"}
    except parity.ParityViolation as e:
        m = str(e)
        out["legs"]["L5_neg_frame_withheld"] = {
            "refused": True,
            "names_the_cause": "SUB-FRAME WAS DECLARED BUT NEVER APPLIED" in m,
            "names_the_call": "build_v2_providers" in m,
            "message_head": [ln for ln in m.splitlines() if ln.strip()][:6]}
    finally:
        V2.build_v2_providers = real
    assert out["legs"]["L5_neg_frame_withheld"]["refused"]

    # ---- L6: cost -----------------------------------------------------------
    import statistics as st
    out["legs"]["L6_cost"] = {
        "mean_load_parent_s": round(st.mean(r["load_parent_s"] for r in timing), 4),
        "mean_load_subframe_s": round(st.mean(r["load_subframe_s"] for r in timing), 4),
        "n": len(timing), "per_clip": timing}

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: (v if k != "legs" else
                          {kk: {k3: v3 for k3, v3 in vv.items() if k3 != "rows"
                                and k3 != "per_clip"}
                           for kk, vv in v.items()})
                      for k, v in out.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
