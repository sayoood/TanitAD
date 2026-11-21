"""Rebuild the cosmos val epcache with TRUE per-clip camera geometry + chunk fix.

Fixes vs build_cosmos.py (cache cosmos-val-a7a8527ba14e, built 2026-07-11):
  1. GEOMETRY — the old path canonicalized every generation mp4 with the NOMINAL
     120-deg pinhole focal of the 1280-px frame (369.5 px -> crop 356) and a
     geometric-center crop. But Cosmos-Drive-Dreams generations inherit the
     SOURCE clip rig: per-clip pinhole_intrinsic.camera_front_wide_120fov
     [fx fy cx cy w h] with fx~944 NATIVE (1920x1080). Native -> generation
     (1280x704) mapping: u*2/3, v=(v-12)*2/3, so fx_gen ~629 and the old cache
     is uniformly ~1.70x zoomed (true f_eff ~452, not the claimed 266).
     Fix: crop side c = round(fx_gen*256/266) ~606 px CENTERED ON THE PER-CLIP
     PRINCIPAL POINT (all 23 val clips are rig B, cy~755 native), replicate-
     padding the ~90-px bottom overflow — mirroring
     tanitad.data.calib.ftheta_crop_resize(center="principal").
  2. CHUNK PAIRING — the old cache paired ALL videos with pose[0:121]; chunk-1
     videos render label frames [121, 242). Handled here by the CURRENT
     tanitad.data.cosmos_drive.build_episode (pose index = 121*chunk + frame),
     which had the fix by build time of this script.
  3. e4ae6dee (mp4 idx 42/43) has no pinhole_intrinsic upstream (and its mp4s
     fail decode) — excluded explicitly.

Ep file naming keeps the old convention: ep_<mp4-sorted-index>.pt over the FULL
48-mp4 sorted list (42/43 absent), so /root/taniteval/results/cosmos_calib.json
keys stay aligned. Writes build_manifest.json with the per-ep crop geometry
ACTUALLY used, so the calib rebuild reads it instead of re-deriving (no drift).
"""
import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, "/root/TanitAD/stack")

from tanitad.data.calib import F_REF                              # noqa: E402
from tanitad.data.cosmos_drive import discover_clips, build_episode  # noqa: E402
from tanitad.data.mixing import save_episode                      # noqa: E402

PAIRS = Path("/root/cosmos_data/pairs")
SIZE = 256
VCROP_TOP = 12.0          # native 1080 -> generation 704: v_gen = (v-12)*2/3
SX = 2.0 / 3.0            # native -> generation scale (1920 -> 1280)
EXCLUDE = ("e4ae6dee",)   # no pinhole_intrinsic upstream; mp4s fail decode
BUILD_KEY = "cosmos|part000|pairs|n24|v2:perclip-pp+chunkfix"


def gen_intrinsics(base: str) -> dict:
    """Per-clip native pinhole intrinsics -> generation-frame (1280x704) values."""
    f = (PAIRS / "pinhole_intrinsic" / base /
         f"{base}.pinhole_intrinsic.camera_front_wide_120fov.npy")
    fx, fy, cx, cy, w, h = [float(v) for v in np.load(f)]
    assert (w, h) == (1920.0, 1080.0), (w, h)
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h,
                fx_g=fx * SX, fy_g=fy * SX,
                cx_g=cx * SX, cy_g=(cy - VCROP_TOP) * SX)


def crop_box(gi: dict) -> tuple[int, int, int]:
    """(c, top, left) in generation px: side for f_eff==F_REF, principal-centered."""
    c = int(round(gi["fx_g"] * SIZE / F_REF))
    top = int(round(gi["cy_g"] - c / 2.0))
    left = int(round(gi["cx_g"] - c / 2.0))
    return c, top, left


def decode_perclip(mp4: Path, size: int, gi: dict, pads_out: dict) -> torch.Tensor:
    """All frames -> u8 [N,3,size,size], per-clip principal-point canonical crop."""
    import av
    frames = []
    with av.open(str(mp4)) as cont:
        st = cont.streams.video[0]
        st.thread_type = "AUTO"
        for fr in cont.decode(st):
            frames.append(torch.from_numpy(
                fr.to_ndarray(format="rgb24")).permute(2, 0, 1))
    vid = torch.stack(frames)                                # [N,3,H,W] u8
    _, _, h, w = vid.shape
    assert (h, w) == (704, 1280), (h, w)
    c, top, left = crop_box(gi)
    # clip box to frame, replicate-pad shortfall (calib.ftheta_crop_resize style)
    y0, y1 = max(0, top), min(h, top + c)
    x0, x1 = max(0, left), min(w, left + c)
    out = vid[..., y0:y1, x0:x1].float()
    pt, pb, pl, pr = y0 - top, (top + c) - y1, x0 - left, (left + c) - x1
    if pt or pb or pl or pr:
        out = Fn.pad(out, (pl, pr, pt, pb), mode="replicate")
    pads_out.update(pad_top=pt, pad_bottom=pb, pad_left=pl, pad_right=pr)
    out = Fn.interpolate(out, size=(size, size), mode="bilinear",
                         align_corners=False)
    return out.clamp(0, 255).to(torch.uint8)


def main():
    t0 = time.time()
    h = hashlib.sha1(BUILD_KEY.encode()).hexdigest()[:12]
    epdir = Path(f"/root/valdata/cosmos-val-{h}")
    epdir.mkdir(parents=True, exist_ok=True)
    print(f"[cosmos-v2] epdir={epdir} F_REF={F_REF}", flush=True)

    clips = discover_clips(str(PAIRS), camera_subdir="generation")
    assert len(clips) == 48, len(clips)
    manifest = {"build_key": BUILD_KEY, "f_ref": F_REF, "size": SIZE,
                "vcrop_top": VCROP_TOP, "sx": SX,
                "native_to_gen": "u_g=u*2/3; v_g=(v-12)*2/3",
                "crop": "c=round(fx_g*256/266) centered on (cx_g,cy_g), "
                        "replicate-pad overflow; pose_idx=121*chunk+frame",
                "eps": {}}
    ok = fail = 0
    for i, clip in enumerate(clips):
        cid = clip["clip_id"]
        if any(cid.startswith(x) for x in EXCLUDE):
            print(f"[cosmos-v2] ep{i:02d} EXCLUDED {cid[:8]} "
                  f"(no pinhole_intrinsic upstream)", flush=True)
            continue
        try:
            gi = gen_intrinsics(cid)
            c, top, left = crop_box(gi)
            f_eff = gi["fx_g"] * SIZE / c
            assert abs(f_eff - F_REF) < 2.0, f_eff
            pads = {}
            ep = build_episode(
                clip, size=SIZE,
                decode_fn=lambda mp4, size: decode_perclip(mp4, size, gi, pads))
            save_episode(ep, str(epdir / f"ep_{i:05d}.pt"))
            v = ep.poses[:, 3]
            manifest["eps"][str(i)] = dict(
                mp4=clip["mp4"].name, clip=cid, chunk=int(clip["chunk"]),
                weather=clip["weather"], T=int(ep.frames.shape[0]),
                fx=gi["fx"], fy=gi["fy"], cx=gi["cx"], cy=gi["cy"],
                crop_c=c, crop_top=top, crop_left=left,
                f_eff_256=round(f_eff, 2), **pads)
            ok += 1
            print(f"[cosmos-v2] ep{i:02d} {cid[:8]} ch{clip['chunk']} "
                  f"{clip['weather']:<11} T={ep.frames.shape[0]} c={c} "
                  f"top={top} left={left} f_eff={f_eff:.1f} "
                  f"pad_b={pads.get('pad_bottom')} "
                  f"v[{v.min():.1f},{v.max():.1f}]", flush=True)
        except Exception as e:
            fail += 1
            print(f"[cosmos-v2] ep{i:02d} FAIL {cid[:8]}: "
                  f"{type(e).__name__}: {e}", flush=True)
    (epdir / "build_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"COSMOS_V2_DONE ok={ok} fail={fail} epdir={epdir} "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
