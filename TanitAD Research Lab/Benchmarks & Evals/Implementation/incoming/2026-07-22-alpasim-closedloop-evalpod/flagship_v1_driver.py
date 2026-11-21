#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Flagship v1 (4-brain WorldModel) as an AlpaSim external driver — closed-loop.

CORRECTION (2026-07-22): pure flagship v1 (`flagship4b-speedjerk`) DOES drive from
observations alone — via its trained TACTICAL POLICY head (NOT only the operative
rollout under true future actions, which is what the trajectory-video generator +
taniteval/rollout.py use, and which misled an earlier finding). The state-only
deploy path (taniteval/closedloop.py) is:
    states = model.encode_window(frames)
    ctx    = model.strategic_policy(states, nav_cmd)["ctx"]
    wp     = model.tactical_policy(states, ctx)["waypoints"]   # {5,10,15,20}->[b,2]
No future actions needed. Reuses REF-C's f-theta canonicalization + gRPC plumbing
(`RefCDriver` is model-agnostic — it calls policy.plan()); only the model swaps.

WARNING: label closed-loop numbers "flagship v1 on NuRec reconstructions" (sim2real).
Run: PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
     /workspace/.../.venv/bin/python /workspace/flagship_v1_driver.py \
       --port 6789 --ckpt /root/models/flagship-30k/ckpt.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent import futures

import numpy as np
import torch

import grpc
from alpasim_grpc.v0.egodriver_pb2_grpc import add_EgodriverServiceServicer_to_server

sys.path.insert(0, "/workspace")
from refc_driver import (F_REF, RefCDriver, ftheta_crop_resize,  # noqa: E402
                         stack_frames)

logger = logging.getLogger("flagship_v1_driver")


class FlagshipV1Policy:
    """Flagship 4-brain WorldModel tactical-policy driver (state-only deploy path)."""

    def __init__(self, ckpt: str, device: str = "cuda"):
        from tanitad.config import flagship4b_config
        from tanitad.models.fourbrain import WorldModel
        cfg = flagship4b_config()
        object.__setattr__(cfg.predictor, "action_dim", 3)      # speedjerk: v0 3rd chan
        if getattr(cfg, "tactical_pred", None) is not None:
            object.__setattr__(cfg.tactical_pred, "action_dim", 3)
        self.model = WorldModel(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["model"])                 # STRICT (smoke: 0/0)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.step = ck.get("step")
        self.window = int(self.model.predictor.cfg.window)
        self.horizons = list(cfg.tactical_policy.waypoint_horizons)   # [5,10,15,20]
        self.n_steps = len(self.horizons)
        self._feff_checked = False
        logger.info("Flagship v1 loaded (step=%s, window=%d, horizons=%s) on %s",
                    self.step, self.window, self.horizons, self.device)

    @torch.no_grad()
    def plan(self, raw_frames: list, intr, v0: float, nav_cmd: int):
        vid = torch.from_numpy(np.stack(raw_frames)).permute(0, 3, 1, 2)  # [T,3,H,W] u8
        canon = ftheta_crop_resize(vid, intr, 256, center="principal")    # [T,3,256,256]
        if not self._feff_checked:
            fe = float(ftheta_crop_resize.last_f_eff)
            logger.info("CANON f_eff=%.1f (F_REF=%.1f) %s", fe, F_REF,
                        "OK" if abs(fe - F_REF) < 8.0 else "FAIL")
            self._feff_checked = True
        stacked = stack_frames(canon, 3)                                  # [T-2,9,256,256]
        fw = stacked[-self.window:][None].to(self.device).float().div_(255.0)
        nav = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
        states = self.model.encode_window(fw)                             # [1,W,S]
        ctx = self.model.strategic_policy(states, nav)["ctx"]             # [1,d_ctx]
        wpd = self.model.tactical_policy(states, ctx)["waypoints"]        # {h:[1,2]}
        traj = np.stack([wpd[h][0].cpu().numpy() for h in self.horizons])  # [4,2] rig
        d = np.diff(np.vstack([[0.0, 0.0], traj]), axis=0)
        headings = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
        return traj, headings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6789)
    ap.add_argument("--ckpt", default="/root/models/flagship-30k/ckpt.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-preds", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    policy = FlagshipV1Policy(ckpt=args.ckpt, device=args.device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    add_EgodriverServiceServicer_to_server(
        RefCDriver(policy, log_preds=args.log_preds), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logger.info("FlagshipV1Driver serving on %s:%d (ckpt=%s)", args.host, args.port, args.ckpt)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
