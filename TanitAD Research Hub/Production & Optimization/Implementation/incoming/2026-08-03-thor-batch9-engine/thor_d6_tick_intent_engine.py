#!/usr/bin/env python3
"""D6 — the full tick RE-MEASURED on the intent-carrying engine (the correct one).

D2's arms used an engine exported with two inputs, which silently dropped the
D-030 tactical intent token. That does not change the latency conclusion much (an
extra Linear(256 -> d) plus an add per step) but the headline must be measured on
the engine we would actually deploy, not on a cheaper one. Nothing here is
inherited from D2: same harness shape, correct engine, re-measured.

Arms (identical windows, 60 real held-out ticks, K = primitive length):
  A  fp32 eager, serialised fan            — today's code and precision
  B  bf16 encoder + engine, SERIALISED fan — engine rebuilt, CALLER not fixed
  C  bf16 encoder + engine, BATCHED fan    — the fix
  D  bf16 encoder + eager,  BATCHED fan    — the no-engine fallback
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
import time

import torch

DEV = "cuda"
HOME = os.path.expanduser("~")
K_PRIM = int(os.environ.get("K_PRIM", "20"))
OUTJ = f"{HOME}/thor_d6_tick_intent_K{K_PRIM}.json"
V1_CKPT = f"{HOME}/models/flagship-v1-speedjerk/ckpt.pt"
V1_VAL = f"{HOME}/valdata/physicalai-val-0c5f7dac3b11"
PLAN = os.environ.get("PLAN", f"{HOME}/trt_deploy/predictor_v1_intent_dyn1-9_fp16.plan")
WIN, N_CAND = 8, 9
STRIDE = int(os.environ.get("STRIDE", "40"))
N_EPS = int(os.environ.get("N_EPS", "12"))
N_TICKS = int(os.environ.get("N_TICKS", "60"))
sys.path.insert(0, f"{HOME}/TanitAD/stack")
sys.path.insert(0, f"{HOME}/TanitAD/stack/scripts")
sys.path.insert(0, f"{HOME}/TanitAD/taniteval")
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")

from build_predictor_trt import TRTPredictor  # noqa: E402
from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.models.fourbrain import (Maneuver, TacticalSelector,  # noqa: E402
                                      WorldModel)
from tanitad.models.metric_dynamics import HierarchicalGrounding  # noqa: E402
import refb_labels as rl  # noqa: E402
from taniteval import rollout as RO  # noqa: E402
from taniteval.data import load_frames  # noqa: E402

OUT = {"script": os.path.abspath(__file__), "host": os.uname().nodename,
       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "budget_ms": 100.0,
       "k_primitive": K_PRIM, "n_candidates": N_CAND, "plan": PLAN}


def stats(ms):
    s = sorted(ms)
    n = len(s)
    return {"n": n, "p50": round(st.median(s), 3),
            "p95": round(s[min(n - 1, int(0.95 * n))], 3),
            "p99": round(s[min(n - 1, int(0.99 * n))], 3),
            "mean": round(st.fmean(s), 3), "max": round(s[-1], 3)}


cfg = flagship4b_config()
object.__setattr__(cfg.predictor, "action_dim", 3)
if getattr(cfg, "tactical_pred", None) is not None:
    object.__setattr__(cfg.tactical_pred, "action_dim", 3)
object.__setattr__(cfg.encoder, "grad_checkpoint", False)
model = WorldModel(cfg)
ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
grounding = HierarchicalGrounding(model.state_dim)
grounding.load_state_dict(ck["grounding"])
model, grounding = model.to(DEV).eval(), grounding.to(DEV).eval()
step_readout = grounding.step["op"]
eng = TRTPredictor(PLAN, device=DEV)
assert eng.has_intent, f"{PLAN} has no intent input — that is the bug, not the fix"
OUT["engine"] = {"plan": PLAN, "io": eng.names,
                 "profile_states": eng.profile_shapes("states"),
                 "profile_intent": eng.profile_shapes("intent")}
sel = TacticalSelector(model, probe_imag=None)

files = sorted(glob.glob(f"{V1_VAL}/ep_*.pt"))[:N_EPS]
eps = load_frames(files)
assert torch.as_tensor(eps[0].feats[:1]).shape[-1] == cfg.encoder.image_size
wins = []
for ep in eps:
    T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
    for t in range(0, T - WIN - 20, STRIDE):
        wins.append((ep, t))
OUT["val"] = {"dir": V1_VAL, "n_episodes": len(eps), "n_windows": len(wins)}
print(OUT["val"], OUT["engine"], flush=True)

STEER, ACCEL = (-0.15, 0.0, 0.15), (-1.0, 0.0, 1.0)
_CLS = {"lane_keep": 0, "turn_left": 1, "turn_right": 2, "accelerate": 3,
        "brake_stop": 4}


def _cls(s_, a_):
    if s_ > 0.01:
        return _CLS["turn_left"]
    if s_ < -0.01:
        return _CLS["turn_right"]
    if a_ < -0.01:
        return _CLS["brake_stop"]
    if a_ > 0.01:
        return _CLS["accelerate"]
    return _CLS["lane_keep"]


def vocabulary(v0, k):
    return [Maneuver(f"s{s_}_a{a_}",
                     torch.tensor([[s_, a_, v0]] * k, device=DEV, dtype=torch.float32),
                     maneuver_class=_cls(s_, a_))
            for s_ in STEER for a_ in ACCEL]


def prep(ep, t):
    last = torch.tensor([t + WIN - 1])
    fw = torch.as_tensor(ep.feats[t:t + WIN])[None].to(DEV).float().div_(255.0)
    aw = ep.actions[t:t + WIN][None].to(DEV)
    fa = ep.actions[t + WIN:t + WIN + 20][None].to(DEV)
    aw, _ = RO.append_ego(aw, fa, ep.poses, last, True, False, False, DEV)
    nav, _ = rl.nav_command(ep.poses, t + WIN - 1)
    return fw, aw, int(nav), float(ep.poses[t + WIN - 1, 3])


@torch.no_grad()
def tick(ep, t, *, bf16, use_engine, batched):
    fw, aw, nav, v0 = prep(ep, t)
    mans = vocabulary(v0, K_PRIM)
    navt = torch.tensor([nav], device=DEV)
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = model.encode_window(fw)
        states = states.float()
    else:
        states = model.encode_window(fw)
    sout = model.strategic_policy(states, navt)
    tout = model.tactical_policy(states, sout["ctx"])
    h = max(cfg.tactical_policy.waypoint_horizons)
    sg = tout["waypoints"][h][0, :2].float()
    prev = model.predictor
    if use_engine:
        model.predictor = eng
    try:
        return sel.propose_and_score(states, aw, mans, sg, step_readout,
                                     sout["ctx"], batch_fan=batched)[0]
    finally:
        model.predictor = prev


ARMS = {
    "A_fp32_eager_serialised": dict(bf16=False, use_engine=False, batched=False),
    "B_bf16_ENGINE_serialised_CALLER_NOT_FIXED": dict(bf16=True, use_engine=True,
                                                      batched=False),
    "C_bf16_ENGINE_batched_THE_FIX": dict(bf16=True, use_engine=True, batched=True),
    "D_bf16_eager_batched_NO_ENGINE": dict(bf16=True, use_engine=False, batched=True),
}
timing = {}
for name, kw in ARMS.items():
    for ep, t in wins[:5]:
        tick(ep, t, **kw)
    torch.cuda.synchronize()
    ms = []
    for ep, t in wins[:N_TICKS]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tick(ep, t, **kw)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
    timing[name] = stats(ms)
    print(name, timing[name], flush=True)
    OUT["full_tick"] = timing
    with open(OUTJ, "w") as f:
        json.dump(OUT, f, indent=2)
OUT["speedup_B_to_C"] = round(timing["B_bf16_ENGINE_serialised_CALLER_NOT_FIXED"]["p50"]
                              / timing["C_bf16_ENGINE_batched_THE_FIX"]["p50"], 3)
with open(OUTJ, "w") as f:
    json.dump(OUT, f, indent=2)
print("=== DONE ===", OUTJ, OUT["speedup_B_to_C"], flush=True)
