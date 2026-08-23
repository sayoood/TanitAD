"""Can the gate CO-PRIMARY be produced at an ADMISSIBLE horizon on v5's frame?

``GATE_PROTOCOL`` §0.3 refuses ``K <= 20`` (that is ``ade_0_2s``' own blind
horizon) and ``K > 190`` (structurally impossible on this corpus). The only
surface that reaches ``20 < K <= 190`` is the closed-loop rollout
(``taniteval.clhorizon.corridor_rollout``) — which until 2026-07-27 re-rendered
through a pinhole homography hard-coded to the deployed 256x256 crop and
therefore could not be pointed at v5's 176x624 cylindrical frame at all.

This driver runs that rollout **on the real pod2 v5 cache, through the
projection-aware re-render**, at ``K = 60`` (6.0 s), and emits the registered
``taniteval.corridor`` block.

⛔ WHAT THIS IS NOT. There is no trained v5 checkpoint (the 25-step smoke was
deleted to free pod2's quota, and this session may not launch training). The
planner here is therefore a REFERENCE POLICY, not v5:

  * ``cv``  — constant velocity, straight ahead at the observed ``v0``. A real,
    interpretable, published-in-program baseline. ⚠️ It is **frame-independent
    by construction**, so its corridor number is a measurement of the POLICY and
    of the loop, and it does NOT exercise the renderer.
  * ``pix`` — a deterministic policy that steers on the warped pixels. It has no
    driving merit whatsoever; its ONLY purpose is that its output CHANGES when
    the re-render changes, which is what makes "the corrected renderer is in the
    loop" a measurement rather than an assertion. Run it under both the correct
    frame and ``LEGACY_WARP`` and the two corridor numbers must differ.

⇒ **No number this file produces is a v5 result, and none may be quoted as one.**
What is under test is whether the artifact the gate consumes can be produced at
an admissible K on v5's frame at all.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tanitad.data.calib import CanonicalFrame, centred_subframe
from tanitad.data.v2_dataset import build_v2_providers
from taniteval import clhorizon as CH


def cv_planner():
    """Hold ``v0``, straight ahead. Ego-frame waypoints, 20 steps."""
    def fn(fw, v0, goal):
        v = v0.float().cpu()[:, None]
        steps = torch.arange(1, 21, dtype=torch.float32)[None]
        return torch.stack([v * steps * CH.DT, torch.zeros_like(v * steps)],
                           dim=-1)
    return CH.CallablePlanner(fn, name="constant_velocity")


def pixel_sensitive_planner(gain=0.6):
    """⚠️ NOT a driving policy. A deterministic function OF THE WARPED PIXELS,
    so that a change in the re-render provably changes the rollout."""
    def fn(fw, v0, goal):
        b = fw.shape[0]
        img = fw.float().cpu()
        w = img.shape[-1]
        left = img[..., : w // 2].mean(dim=(1, 2, 3, 4))
        right = img[..., w // 2:].mean(dim=(1, 2, 3, 4))
        bias = torch.tanh((right - left) * 8.0) * gain
        v = v0.float().cpu()[:, None]
        steps = torch.arange(1, 21, dtype=torch.float32)[None]
        x = v * steps * CH.DT
        y = bias[:, None] * steps * CH.DT
        return torch.stack([x, y], dim=-1)
    return CH.CallablePlanner(fn, name="pixel_sensitive_probe")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--subframe", default="176x624")
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--planner", default="cv", choices=["cv", "pix"])
    ap.add_argument("--legacy-warp", action="store_true",
                    help="re-render with the SHIPPED 266/128 pinhole warp "
                         "instead of the frame's own projection (the "
                         "pre-2026-07-27 behaviour) — the negative control")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cache = Path(a.cache)
    geo = json.loads((cache / "_geometry.json").read_text())
    cache_frame = CanonicalFrame.from_dict(geo["frame"])        # READ
    h, w = (int(x) for x in a.subframe.lower().split("x"))
    train_frame = centred_subframe(cache_frame, h, w)

    eps = build_v2_providers([str(cache)], lru_size=8, frame=train_frame,
                             verbose=False)[:a.episodes]
    planner = cv_planner() if a.planner == "cv" else pixel_sensitive_planner()
    frame_arg = CH.LEGACY_WARP if a.legacy_warp else train_frame

    t0 = time.time()
    pw = CH.corridor_rollout(planner, eps, None, a.device, a.K,
                             stride=a.stride, batch=a.batch, verbose=True,
                             frame=frame_arg)
    if pw is None:
        raise SystemExit(f"no episode yields a window at K={a.K}")
    res = CH.emit(pw, a.K, surface="closed_loop")
    res["_NOT_A_MODEL_RESULT"] = (
        "the planner is a REFERENCE POLICY, not a v5 checkpoint. This block "
        "demonstrates that the registered co-primary is PRODUCIBLE at an "
        "admissible K on v5's 176x624 cylindrical frame through the "
        "projection-aware re-render. Its VALUE is a property of the policy "
        "named below, and may not be quoted as a v5 number.")
    res["planner"] = planner.name
    res["re_render"] = pw["_warp"]
    res["frame"] = train_frame.to_dict()
    res["frame_tag"] = train_frame.tag()
    res["cache"] = str(cache)
    res["n_episodes_loaded"] = len(eps)
    res["stride"] = a.stride
    res["wallclock_s"] = round(time.time() - t0, 1)
    res["goal_provenance"] = "NONE (this planner consumes no goal channel)"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    pwp = str(Path(a.out).with_suffix("")) + f"_perwindow_K{a.K}.pt"
    torch.save(pw, pwp)
    print(f"[corridor] wrote {a.out} and {pwp}")
    ov = (res.get("overall") or {}).get("corridor_departure_rate")
    ju = (res.get("junction") or {}).get("corridor_departure_rate")
    print(json.dumps({"planner": planner.name, "frame": res["frame_tag"],
                      "legacy_warp": bool(a.legacy_warp), "K": a.K,
                      "n_by_stratum": res.get("n_by_stratum"),
                      "overall_corridor_departure_rate": ov,
                      "junction_corridor_departure_rate": ju,
                      "wallclock_s": res["wallclock_s"]},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
