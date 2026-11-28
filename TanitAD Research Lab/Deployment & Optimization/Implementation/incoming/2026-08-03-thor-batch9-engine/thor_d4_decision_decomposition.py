#!/usr/bin/env python3
"""D4 — WHY did the selected candidate disagree? Decompose it, do not report it.

D2/K=20 measured **48.3 %** selected-candidate agreement between
``A = fp32 eager + serialised fan`` and ``C = bf16 encoder + TRT-fp16 engine +
batched fan``, against a 95.3 % bar — with a max score delta of **73.4**. Three
things changed at once (encoder precision, predictor precision, fan batching) and
a 73-unit score delta is far too large to be a 3e-4 numerical effect, so the
headline number is uninterpretable as it stands. This script changes ONE thing at
a time:

  P0  fp32 encoder + fp32 eager predictor + SERIALISED fan   (reference)
  P1  fp32 encoder + fp32 eager predictor + BATCHED fan      -> batching alone
  P2  fp32 encoder + TRT-fp16 engine      + BATCHED fan      -> predictor precision
  P3  bf16 encoder + TRT-fp16 engine      + BATCHED fan      -> + encoder precision

and adds the diagnostic that decides whether "agreement" means anything at all:
the **score margin** |best - second best|. If the margin is at the scale of the
perturbation, the argmin is a coin flip and a low agreement measures the
SELECTOR's determinacy, not the engine's fidelity. Also reported: the score
magnitudes (a 20-step imagination roll under a constant primitive can leave the
training manifold, and the endpoint distance then explodes).

⛔ P1 vs P0 is the falsifier for the code change itself: batching must not move
the decision. If P0<->P1 disagrees, ``_fan_batched`` is wrong.
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
OUTJ = f"{HOME}/thor_d4_decision_decomposition.json"
V1_CKPT = f"{HOME}/models/flagship-v1-speedjerk/ckpt.pt"
V1_VAL = f"{HOME}/valdata/physicalai-val-0c5f7dac3b11"
PLAN_DYN = f"{HOME}/trt_d1/v1_dyn1-9_fp16.plan"
WIN, N_CAND = 8, 9
STRIDE = int(os.environ.get("STRIDE", "20"))
N_EPS = int(os.environ.get("N_EPS", "24"))
N_WIN = int(os.environ.get("N_WIN", "200"))
K_LIST = [int(x) for x in os.environ.get("K_LIST", "4,20").split(",")]
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
from taniteval.ci import episode_cluster_bootstrap  # noqa: E402
from taniteval.data import load_frames  # noqa: E402

OUT = {"script": os.path.abspath(__file__), "host": os.uname().nodename,
       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "k_list": K_LIST,
       "agreement_bar": 0.953}


def dump():
    OUT["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(OUTJ, "w") as f:
        json.dump(OUT, f, indent=2)


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
OUT["model"] = {"ckpt": V1_CKPT, "step": int(ck.get("step", -1)),
                "raster": f"{cfg.encoder.image_size}px SQUARE", "strict_load": True}
dyn = TRTPredictor(PLAN_DYN, device=DEV)
sel = TacticalSelector(model, probe_imag=None)

files = sorted(glob.glob(f"{V1_VAL}/ep_*.pt"))[:N_EPS]
eps = load_frames(files)
assert torch.as_tensor(eps[0].feats[:1]).shape[-1] == cfg.encoder.image_size, \
    "raster mismatch — refusing"
wins = []
for ep in eps:
    T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
    for t in range(0, T - WIN - 20, STRIDE):
        wins.append((ep, t))
wins = wins[:N_WIN]
OUT["val"] = {"dir": V1_VAL, "n_episodes": len(eps), "n_windows": len(wins),
              "stride": STRIDE}
print(OUT["val"], flush=True)

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
def score_window(ep, t, k, *, enc_bf16, use_engine, batched):
    fw, aw, nav, v0 = prep(ep, t)
    mans = vocabulary(v0, k)
    navt = torch.tensor([nav], device=DEV)
    if enc_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = model.encode_window(fw)
        states = states.float()
    else:
        states = model.encode_window(fw)
    sout = model.strategic_policy(states, navt)
    tout = model.tactical_policy(states, sout["ctx"])
    h = max(cfg.tactical_policy.waypoint_horizons)
    subgoal = tout["waypoints"][h][0, :2].float()
    prev = model.predictor
    if use_engine:
        model.predictor = dyn
    try:
        best, scores = sel.propose_and_score(states, aw, mans, subgoal,
                                             step_readout, sout["ctx"],
                                             batch_fan=batched)
    finally:
        model.predictor = prev
    s = scores.float().cpu()
    srt = torch.sort(s).values
    return int(best), s, float(srt[1] - srt[0])       # margin = 2nd best - best


CONFIGS = {
    "P0_fp32_eager_serialised": dict(enc_bf16=False, use_engine=False, batched=False),
    "P1_fp32_eager_BATCHED": dict(enc_bf16=False, use_engine=False, batched=True),
    "P2_fp32enc_ENGINE_batched": dict(enc_bf16=False, use_engine=True, batched=True),
    "P3_bf16enc_ENGINE_batched": dict(enc_bf16=True, use_engine=True, batched=True),
}

for K in K_LIST:
    print(f"=== K = {K} ===", flush=True)
    picks = {c: [] for c in CONFIGS}
    scores = {c: [] for c in CONFIGS}
    margins = {c: [] for c in CONFIGS}
    eid = []
    for ep, t in wins:
        for cname, kw in CONFIGS.items():
            b, s, m = score_window(ep, t, K, **kw)
            picks[cname].append(b)
            scores[cname].append(s)
            margins[cname].append(m)
        eid.append(int(ep.episode_id))
    res = {"n_windows": len(eid), "n_episodes": len(set(eid))}
    ref = "P0_fp32_eager_serialised"
    for cname in CONFIGS:
        if cname == ref:
            continue
        agree = [1.0 if a == b else 0.0 for a, b in zip(picks[ref], picks[cname])]
        d = torch.stack([(x - y).abs().max() for x, y in
                         zip(scores[ref], scores[cname])])
        res[f"{ref}_vs_{cname}"] = {
            "agreement": round(float(sum(agree) / len(agree)), 5),
            "n_flipped": int(len(agree) - sum(agree)),
            "ci": episode_cluster_bootstrap(agree, eid),
            "max_abs_score_delta": round(float(d.max()), 4),
            "median_abs_score_delta": round(float(d.median()), 6)}
    # step-wise: each config against the PREVIOUS one isolates one factor
    order = list(CONFIGS)
    for a, b in zip(order, order[1:]):
        ag = [1.0 if x == y else 0.0 for x, y in zip(picks[a], picks[b])]
        res[f"STEP_{a}_vs_{b}"] = {
            "agreement": round(float(sum(ag) / len(ag)), 5),
            "n_flipped": int(len(ag) - sum(ag)),
            "ci": episode_cluster_bootstrap(ag, eid)}
    # is the decision even determined? margin vs the size of the perturbation
    for cname in CONFIGS:
        mm = sorted(margins[cname])
        sc = torch.stack(scores[cname])
        res[f"margin_{cname}"] = {
            "p50": round(st.median(mm), 6), "p05": round(mm[int(0.05 * len(mm))], 6),
            "frac_below_1e-2": round(sum(1 for x in mm if x < 1e-2) / len(mm), 4),
            "frac_below_1e-1": round(sum(1 for x in mm if x < 1e-1) / len(mm), 4),
            "score_abs_p50": round(float(sc.abs().median()), 4),
            "score_abs_max": round(float(sc.abs().max()), 4),
            "n_distinct_winners": len(set(picks[cname]))}
    OUT[f"K{K}"] = res
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("STEP", "margin"))},
                     indent=1)[:2200], flush=True)
    dump()
print("=== DONE ===", OUTJ, flush=True)
