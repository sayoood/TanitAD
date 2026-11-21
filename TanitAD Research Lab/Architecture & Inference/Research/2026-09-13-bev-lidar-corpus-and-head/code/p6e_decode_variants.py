#!/usr/bin/env python3
"""P6e - G5 failed (rebuilt-frame tokens 1.73-1.84 % from pod-frame tokens; bar 1 %). Which
stage of the rebuild differs from the pod? Cheapest discriminating experiment: vary ONLY the
video decoder's YUV->RGB conversion and keep the rectification identical, on raw frames that
the payload also holds.

If one conversion variant reproduces the pod's PNG frames (near-100 % exactly equal pixels),
the decoder was the cause and G5 can be re-run with it; if NO variant moves the match, the
difference sits in the rectification (or the pod's library versions) and a same-distribution
design (every split on rebuilt frames) is the lever instead.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_build_corpus as B  # noqa: E402
from lidar_fetch import sha12  # noqa: E402


def main() -> int:
    import av
    import torchvision.io as tvio
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
    from tanitad.data.physicalai import _physicalai_root_of, intrinsics_for_clip
    torch.set_num_threads(2)
    cid = B.join_clips()[0]
    g = B.episode_grid(cid)
    want = [int(i) for i in np.linspace(10, len(g["cam_frame_idx"]) - 10, 6).round()]
    need = {int(g["cam_frame_idx"][i]): i for i in want}
    d = torch.load(Path(B.V2EP_DIR) / f"{cid}.v2ep.pt", map_location="cpu", weights_only=False)
    lens = d["jpeg_len"].to(torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    ref = {i: tvio.decode_png(d["jpeg_buf"][int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)
           for i in want}
    mp4 = Path(B.CAM_DIR) / f"{cid}.mp4"
    intr = intrinsics_for_clip(cid, _physicalai_root_of(mp4))
    fr = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391, projection="cylindrical")

    variants = {
        "to_ndarray_rgb24(default)": lambda vf: vf.to_ndarray(format="rgb24"),
    }
    for interp in ("FAST_BILINEAR", "BILINEAR", "BICUBIC", "POINT", "AREA", "LANCZOS"):
        variants[f"reformat_rgb24_{interp}"] = (lambda i: (lambda vf: vf.reformat(format="rgb24", interpolation=i).to_ndarray()))(interp)
    for cs in ("ITU709", "ITU601"):
        variants[f"reformat_rgb24_src_{cs}"] = (lambda c: (lambda vf: vf.reformat(format="rgb24", src_colorspace=c).to_ndarray()))(cs)

    res = {"schema": "tanitad.decode_variants/1", "evidence_class": "MEASURED (ours, dev-box CPU)",
           "clip": sha12(cid), "frames_compared": want, "pyav": av.__version__, "torch": torch.__version__,
           "variants": {}}
    raw = {}
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        res["stream"] = {"codec": st.codec_context.name, "pix_fmt": st.codec_context.pix_fmt,
                         "width": st.codec_context.width, "height": st.codec_context.height}
        fi = 0
        for vf in c.decode(st):
            if fi in need:
                raw[need[fi]] = {name: fn(vf) for name, fn in variants.items() if _ok(fn, vf)}
            fi += 1
    for name in variants:
        eq, mad, mx, n = [], [], [], 0
        try:
            for i in want:
                if name not in raw[i]:
                    raise RuntimeError("variant failed on this build")
                x = torch.from_numpy(raw[i][name]).permute(2, 0, 1)[None]
                out = cylindrical_rectify(x, intr, fr)[0]
                diff = (out.to(torch.int16) - ref[i].to(torch.int16)).abs()
                eq.append(float((diff == 0).float().mean()))
                mad.append(float(diff.float().mean()))
                mx.append(int(diff.max()))
            res["variants"][name] = {"exact_equal_frac": float(np.mean(eq)), "mean_abs": float(np.mean(mad)),
                                     "max_abs": int(max(mx))}
        except Exception as e:  # noqa: BLE001
            res["variants"][name] = {"error": f"{type(e).__name__}: {e}"[:160]}
        print(name, json.dumps(res["variants"][name]), flush=True)
    (HERE.parent / "raw" / "p6e_decode_variants.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def _ok(fn, vf) -> bool:
    try:
        fn(vf)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
