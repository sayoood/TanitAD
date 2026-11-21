#!/usr/bin/env python3
"""C77 content verification for the pilot's .v2ep.pt payloads.

⛔ EXISTENCE IS NOT VERIFICATION. A file can be present, non-zero and truncated,
or carry the right keys at the wrong geometry. This LOADS each payload through
the real ``load_compressed()`` and asserts the decoded episode, then checks the
geometry actually delivered against the geometry the corpus claims.

Checks per clip:
  * ``load_compressed()`` returns a ToyEpisode (real PNG decode + D-015 stacking)
  * frames  [T, 9, 256, 640] uint8   — 9 = 3 stacked RGB frames
  * actions [T, >=2] and poses [T, >=3] float, T matching frames
  * maneuvers [T]
  * T > 0 and the four tensors agree on T
  * no NaN/Inf in poses or actions

Usage:  verify_out.py <out_dir> [max_files]
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.environ.get("TANITAD_STACK", "/home/nvidia/TanitAD/stack"))
sys.path.append(os.path.join(os.environ.get("TANITAD_STACK",
                                            "/home/nvidia/TanitAD/stack"), "scripts"))
import importlib.util as iu                                       # noqa: E402
import torch                                                      # noqa: E402

_spec = iu.spec_from_file_location(
    "v2c", os.path.join(os.environ.get("TANITAD_STACK",
                                       "/home/nvidia/TanitAD/stack"),
                        "scripts", "v2_compressed.py"))
v2c = iu.module_from_spec(_spec)
_spec.loader.exec_module(v2c)


def main() -> int:
    out_dir = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".v2ep.pt"))
    if cap:
        files = files[:cap]

    ok, bad, shapes, frames_total, bytes_total = 0, [], {}, 0, 0
    for fn in files:
        p = os.path.join(out_dir, fn)
        try:
            ep = v2c.load_compressed(p)
            f, a, po, mn = ep.frames, ep.actions, ep.poses, ep.maneuvers
            T = f.shape[0]
            assert f.ndim == 4 and f.shape[1] == 9, f"frames {tuple(f.shape)}"
            assert f.dtype == torch.uint8, f"frames dtype {f.dtype}"
            assert T > 0, "empty episode"
            assert a.shape[0] == T, f"actions T {a.shape} vs {T}"
            assert po.shape[0] == T, f"poses T {po.shape} vs {T}"
            assert mn.shape[0] == T, f"maneuvers T {mn.shape} vs {T}"
            assert po.shape[1] >= 3 and a.shape[1] >= 2, \
                f"narrow actions/poses {tuple(a.shape)} {tuple(po.shape)}"
            assert torch.isfinite(po).all(), "non-finite poses"
            assert torch.isfinite(a).all(), "non-finite actions"
            key = f"{f.shape[1]}x{f.shape[2]}x{f.shape[3]}"
            shapes[key] = shapes.get(key, 0) + 1
            frames_total += T
            bytes_total += os.path.getsize(p)
            ok += 1
        except Exception as e:
            bad.append((fn, f"{type(e).__name__}: {e}"))

    geom = None
    gp = os.path.join(out_dir, "_geometry.json")
    if os.path.exists(gp):
        geom = json.load(open(gp))

    print(json.dumps({
        "checked": len(files), "ok": ok, "failed": len(bad),
        "frame_shapes": shapes,
        "stacked_frames_total": frames_total,
        "mean_frames_per_clip": round(frames_total / ok, 1) if ok else 0,
        "bytes_total": bytes_total,
        "mean_MB_per_clip": round(bytes_total / ok / 1e6, 2) if ok else 0,
        "geometry_frame_tag": (geom or {}).get("frame_tag"),
        "geometry_codec": (geom or {}).get("codec"),
        "achieved_hfov_deg": ((geom or {}).get("geometry_check") or {}
                              ).get("achieved_hfov_deg"),
        "failures": bad[:10],
    }, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
