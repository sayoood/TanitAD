#!/usr/bin/env python3
"""NAV MANIPULATION SWEEP on the banked junction rollout — why did the arms not turn?

⭐ THE IDENTIFYING MOVE, and the reason this file is not an observational read of the
banked rollouts. R-2026-08-03-l established that a contingency table CANNOT separate a
head that USES nav from a head that ECHOES it: they agree whenever the command is
correct. The only separator is a MANIPULATION — hold the observation byte-fixed and move
the nav input. This file does that for the *trajectory* (the banked retraction did it for
the *route head*, a different output).

WHAT IS HELD FIXED. The rendered frames. They were saved by
`openloop_drive.py --save-video-frames` (JPEG q90) at the LOGGED rig poses, so the
observation stack for tick k is `frames[k-9 .. k]` and is IDENTICAL across every nav value
and every arm. No re-render happens here, so the renderer's step-function-of-pose
sensitivity (0.1 px camera rotation -> 6.65 m at 2 s) is entirely out of play.

⚠️ TWO HONEST LIMITS, both measured rather than argued:
  1. The banked frames are JPEG q90, not the raw uint8 the live run fed the policy. The
     sweep is therefore run on a *slightly* different observation than the banked one. The
     script MEASURES the size of that gap (`repro`) by re-running at the BANKED nav and
     comparing against the banked plan. The manipulation itself is unaffected: every nav
     arm sees the same JPEG.
  2. `--save-video-frames` only writes frames from k >= NEED_FRAMES-1 = 9, so the window
     for tick k needs frames k-9..k and the FIRST RECONSTRUCTIBLE TICK IS k = 18. 181 of
     the 190 banked ticks are swept; the 9 dropped ticks are reported, not hidden.

WHAT IT EMITS, per tick, per nav in {follow, left, right, straight}:
  traj [4,2] ego-frame waypoints at 0.5/1.0/1.5/2.0 s, kappa_plan, steer, v_target,
  route_logits, maneuver_logits, and — for the anchored decoders — the SELECTED anchor
  index and the anchor-logit landscape. That last part is what separates mechanism (b)
  "the turn is not in the fan" from (c) "the turn is in the fan and loses selection".

⛔ NAV_STRAIGHT (index 3) is sampled but must NOT be read as a condition: `_ROUTE_TO_NAV`
never emits it, so for the flagship that embedding row is UNTRAINED (R-2026-08-03-l
correction 3). It is swept only to show what an out-of-vocabulary probe looks like.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

DT = 0.1
WINDOW = 8
STACK = 3
NEED_FRAMES = WINDOW + STACK - 1          # 10 native ticks -> [8, 9, 256, 256]
WP_STEPS = (5, 10, 15, 20)                # 0.5 / 1.0 / 1.5 / 2.0 s at 10 Hz
NAV_NAMES = ("follow", "left", "right", "straight")


# --------------------------------------------------------------------------------- #
# observation                                                                        #
# --------------------------------------------------------------------------------- #
def load_frame(path: Path) -> np.ndarray:
    """JPEG -> HxWx3 uint8 RGB.

    `openloop_drive` wrote `cv2.imwrite(p, img[:, :, ::-1])`, i.e. it handed cv2 a BGR
    array of an RGB image, so the FILE holds the true colours and PIL returns RGB
    directly. (cv2 is not installed on the dev box; PIL's baseline JPEG decode differs
    from libjpeg-turbo's by at most a rounding LSB and — decisive here — it is the SAME
    decoder for every nav value, so the manipulation is exact regardless.)"""
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def build_intr(rig_json: Path, cam: str = "camera_front_wide_120fov"):
    """The scene's own f-theta intrinsics, read from `rig_trajectories.json`.

    Re-derived from the scene rather than hard-coded, because the crop is
    principal-point-centred and a wrong cy is a ~215 px vertical error (the two-rig
    trap). The field path is the SAME one `nurec_loader.RigTrajectories.camera`
    (`stack/experiments/nurec-gsplat/nurec_loader.py:405-424`) walks; it is inlined here
    only because that module hard-imports `msgpack` at module scope for the *volume*
    reader, which this script never touches."""
    from tanitad.data.calib import FThetaIntrinsics   # noqa: E402
    d = json.loads(Path(rig_json).read_text())
    clip = d["rig_trajectories"][0]["sequence_id"]
    key = cam if "@" in cam else f"{cam}@{clip}"
    c = d["camera_calibrations"][key]
    cm = c["camera_model"]
    if cm["type"] != "ftheta":
        raise NotImplementedError(f"camera_model.type={cm['type']!r}")
    p = cm["parameters"]
    return FThetaIntrinsics(
        poly=tuple(float(x) for x in p["angle_to_pixeldist_poly"]),
        cx=float(p["principal_point"][0]), cy=float(p["principal_point"][1]),
        width=int(p["resolution"][0]), height=int(p["resolution"][1]),
        per_clip=True)


# --------------------------------------------------------------------------------- #
# arms                                                                               #
# --------------------------------------------------------------------------------- #
class _Base:
    def __init__(self, device="cuda"):
        import torch
        from tanitad.data.calib import F_REF, ftheta_crop_resize
        from tanitad.data.comma2k19 import stack_frames
        self.torch, self.F_REF = torch, F_REF
        self._crop, self._stack = ftheta_crop_resize, stack_frames
        self.device = device if torch.cuda.is_available() else "cpu"
        self.f_eff = None

    def canon(self, frames, intr):
        """IDENTICAL to `closedloop_drive._BasePolicy.canon` — deliberately copied, not
        re-derived, so the raster the sweep feeds is the raster the banked run fed."""
        torch = self.torch
        vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        canon = self._crop(vid, intr, 256, center="principal")
        if self.f_eff is None:
            self.f_eff = float(self._crop.last_f_eff)
            if abs(self.f_eff - self.F_REF) >= 8.0:
                raise RuntimeError(f"f_eff self-check FAILED ({self.f_eff})")
        st = self._stack(canon, STACK)
        fw = st[-WINDOW:][None].to(self.device).float().div_(255.0)
        if tuple(fw.shape[-3:]) != (9, 256, 256) or fw.shape[1] != WINDOW:
            raise RuntimeError(f"raster assertion failed: {tuple(fw.shape)}")
        return fw


class FlagshipSweep(_Base):
    name = "flagship-v1"

    def __init__(self, ckpt, device="cuda"):
        super().__init__(device)
        torch = self.torch
        from tanitad.config import flagship4b_config
        from tanitad.models.fourbrain import WorldModel
        cfg = flagship4b_config()
        object.__setattr__(cfg.predictor, "action_dim", 3)
        if getattr(cfg, "tactical_pred", None) is not None:
            object.__setattr__(cfg.tactical_pred, "action_dim", 3)
        self.model = WorldModel(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.model = self.model.to(self.device).eval()
        self.step = ck.get("step")
        self.horizons = list(cfg.tactical_policy.waypoint_horizons)
        dec = self.model.tactical_policy.anchor_decoder
        self.anchors = None if dec is None else dec.anchors.detach().cpu().numpy()

    def encode(self, fw):
        with self.torch.no_grad():
            return self.model.encode_window(fw)

    def heads(self, states, nav_cmd):
        torch = self.torch
        with torch.no_grad():
            nav = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
            sout = self.model.strategic_policy(states, nav)
            tout = self.model.tactical_policy(states, sout["ctx"])
            traj = np.stack([tout["waypoints"][h][0].float().cpu().numpy()
                             for h in self.horizons])
        out = {"traj": traj.astype(float).tolist(),
               "route_logits": sout["route_logits"][0].float().cpu().numpy().tolist(),
               "ctx": sout["ctx"][0].float().cpu().numpy().tolist(),
               "maneuver_logits":
                   tout["maneuver_logits"][0].float().cpu().numpy().tolist()}
        if "anchor_logits" in tout:
            al = tout["anchor_logits"][0].float().cpu().numpy()
            out["sel_idx"] = int(tout["sel_idx"][0])
            out["anchor_logits"] = al.astype(float).tolist()
        return out


class RefCSweep(_Base):
    def __init__(self, ckpt, preset="base", device="cuda"):
        super().__init__(device)
        self.name = f"refc-{preset}"
        sys.path.insert(0, str(Path(__file__).resolve().parents[5]
                               / "stack" / "scripts"))
        from refc_v12_cache import load_frozen
        self.model, self.cfg, self.step = load_frozen(ckpt, preset, None, self.device)
        # REF-C's anchor vocabulary lives on the DECODER (`model.decoder.anchors`),
        # unlike the flagship's, which lives on `tactical_policy.anchor_decoder`.
        a = getattr(getattr(self.model, "decoder", None), "anchors", None)
        self.anchors = None if a is None else a.detach().cpu().numpy()

    def encode(self, fw):
        return fw            # REF-C's forward owns its encoder; no split point

    def heads(self, fw, nav_cmd, v0=0.0):
        torch = self.torch
        with torch.no_grad():
            navt = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
            v0t = torch.tensor([float(v0)], dtype=torch.float32, device=self.device)
            o = self.model(fw, nav_cmd=navt, v0=v0t, steps=2)
        out = {"traj": o["traj"][0].float().cpu().numpy().astype(float).tolist()}
        for k in ("route_logits", "maneuver_logits"):
            if k in o:
                out[k] = o[k][0].float().cpu().numpy().tolist()
        if "anchor_logits" in o:
            out["sel_idx"] = int(o["sel_idx"][0])
            out["anchor_logits"] = o["anchor_logits"][0].float().cpu().numpy(
            ).astype(float).tolist()
        if "pooled" in o:
            out["pooled_md5"] = float(np.abs(
                o["pooled"][0].float().cpu().numpy()).sum())
        return out


# --------------------------------------------------------------------------------- #
# geometry                                                                            #
# --------------------------------------------------------------------------------- #
def gt_ego_waypoints(gt, k):
    """GT future in the ego frame of tick k, at WP_STEPS. None where the clip ends."""
    if k + max(WP_STEPS) >= len(gt):
        return None
    x0, y0, yaw = gt[k]["x"], gt[k]["y"], gt[k]["yaw"]
    c, s = math.cos(-yaw), math.sin(-yaw)
    out = []
    for h in WP_STEPS:
        dx, dy = gt[k + h]["x"] - x0, gt[k + h]["y"] - y0
        out.append([c * dx - s * dy, s * dx + c * dy])
    return out


def kappa_of(traj, lookahead=0):
    """Same pure-pursuit curvature the driver records (`wp_to_control`)."""
    x, y = float(traj[lookahead][0]), float(traj[lookahead][1])
    ld2 = max(x * x + y * y, 0.25)
    return 2.0 * y / ld2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--rollout", required=True, help="banked rollout json (for gt + nav)")
    ap.add_argument("--rig-json", required=True)
    ap.add_argument("--arm", required=True, choices=["flagship-v1", "refc-base"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--navs", default="0,1,2,3")
    ap.add_argument("--max-ticks", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "6")
    fd = Path(args.frames_dir)
    roll = json.loads(Path(args.rollout).read_text())
    gt = roll["gt"]
    steps = [s for r in roll["rollouts"] for s in r["steps"]]
    by_k = {s["k"]: s for s in steps}
    navs = [int(x) for x in args.navs.split(",")]

    intr = build_intr(Path(args.rig_json))
    arm = (FlagshipSweep(args.ckpt) if args.arm == "flagship-v1"
           else RefCSweep(args.ckpt, "base"))

    have = {int(p.stem) for p in fd.glob("*.jpg")}
    ks = sorted(k for k in by_k if all((k - j) in have for j in range(NEED_FRAMES)))
    dropped = sorted(set(by_k) - set(ks))
    if args.max_ticks:
        ks = ks[:args.max_ticks]

    cache: dict[int, np.ndarray] = {}
    ticks, t0 = [], time.time()
    for i, k in enumerate(ks):
        for j in range(k - NEED_FRAMES + 1, k + 1):
            if j not in cache:
                cache[j] = load_frame(fd / f"{j:05d}.jpg")
        for j in list(cache):
            if j < k - NEED_FRAMES + 1:
                del cache[j]
        frames = [cache[j] for j in range(k - NEED_FRAMES + 1, k + 1)]
        fw = arm.canon(frames, intr)
        v0 = float(by_k[k]["v"])
        enc = arm.encode(fw)
        rec = {"k": k, "v": v0, "nav_banked": int(by_k[k]["nav"]),
               "nav_valid": bool(by_k[k]["nav_detail"].get("nav_valid", False)),
               "plan_banked": by_k[k]["plan"],
               "kappa_banked": by_k[k]["kappa_plan"],
               "gt_ego_wp": gt_ego_waypoints(gt, k), "sweep": {}}
        for nv in navs:
            h = (arm.heads(enc, nv) if args.arm == "flagship-v1"
                 else arm.heads(enc, nv, v0))
            h["kappa"] = kappa_of(h["traj"])
            rec["sweep"][str(nv)] = h
        ticks.append(rec)
        if i % 20 == 0:
            print(f"  k={k:3d} ({i+1}/{len(ks)})  {time.time()-t0:.0f}s", flush=True)

    out = {"arm": arm.name, "ckpt": args.ckpt, "step": arm.step,
           "frames_dir": str(fd), "scene": roll["scene"],
           "condition": Path(args.frames_dir).parent.name,
           "navs": navs, "n_ticks": len(ticks),
           "dropped_ticks": dropped,
           "dropped_reason": "--save-video-frames starts at k=NEED_FRAMES-1=9, so the "
                             "10-frame window is only complete from k=18",
           "f_eff": arm.f_eff, "wall_s": time.time() - t0,
           "anchors": None if arm.anchors is None else arm.anchors.tolist(),
           "wp_steps": list(WP_STEPS), "ticks": ticks}
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}  ticks={len(ticks)}  dropped={len(dropped)}  "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
