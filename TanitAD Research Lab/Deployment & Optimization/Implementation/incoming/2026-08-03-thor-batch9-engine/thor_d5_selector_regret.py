#!/usr/bin/env python3
"""D5 — a flipped decision is not automatically a WORSE decision. Measure the regret.

D4 measured, at the config's own ``TacticalConfig.horizon = 4``:

  * P0 serialised  vs P1 BATCHED   -> **100.00 %** agreement (0/200) — the code change is neutral
  * P1 fp32 eager  vs P2 TRT-fp16  -> **86.50 %** (27/200 flip) — below the 95.3 % bar
  * P2 fp32 encoder vs P3 bf16     -> **100.00 %** (0/200) — the encoder lever is neutral

Agreement alone cannot say whether those 27 flips matter: a 9-way argmin over
scores whose top two differ by 1e-3 is a coin flip, and swapping two nearly
equal-cost primitives is not a driving error. The decision-theoretic quantity is
**REGRET measured in the REFERENCE's own units**::

    regret = score_ref[choice_test] - score_ref[choice_ref]   >= 0

i.e. "how much worse, according to fp32, is the candidate fp16 picked". The
scores are metres of endpoint distance to the tactical sub-goal plus a comfort
and a policy-prior term, so regret is interpretable in metres.

Reported per K: agreement, the full regret distribution, the flip rate
conditioned on the reference's own score MARGIN, and — the control that makes it
falsifiable — the same statistics for P0 vs P1, which must be exactly zero.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import torch

DEV = "cuda"
HOME = os.path.expanduser("~")
OUTJ = f"{HOME}/thor_d5_selector_regret.json"
V1_CKPT = f"{HOME}/models/flagship-v1-speedjerk/ckpt.pt"
V1_VAL = f"{HOME}/valdata/physicalai-val-0c5f7dac3b11"
PLAN_NOINTENT = f"{HOME}/trt_d1/v1_dyn1-9_fp16.plan"          # the WIRING BUG control
PLAN_INTENT = f"{HOME}/trt_deploy/predictor_v1_intent_dyn1-9_fp16.plan"
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
       "agreement_bar": 0.953,
       "regret_units": "score units = metres of endpoint distance to the tactical "
                       "sub-goal + 0.01*comfort - 1.0*log-prior"}


def dump():
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
eng_noint = TRTPredictor(PLAN_NOINTENT, device=DEV)
eng_int = TRTPredictor(PLAN_INTENT, device=DEV)
assert not eng_noint.has_intent and eng_int.has_intent, "engines are not the pair claimed"
sel = TacticalSelector(model, probe_imag=None)
OUT["model"] = {"ckpt": V1_CKPT, "step": int(ck.get("step", -1)), "strict_load": True}
OUT["engines"] = {"no_intent": {"plan": PLAN_NOINTENT, "io": eng_noint.names},
                  "intent": {"plan": PLAN_INTENT, "io": eng_int.names}}


class _StripIntent(torch.nn.Module):
    """Reproduces the 2026-08-03 wiring bug ON PURPOSE, as the control arm.

    The intent-less engine now REFUSES an intent token (that is the fix), so the
    only way to measure what the bug cost is to drop the token explicitly here,
    where it is visible, rather than silently inside the runtime."""

    def __init__(self, eng):
        super().__init__()
        self.eng = eng

    def forward(self, states, actions, intent=None):
        return self.eng(states, actions)

files = sorted(glob.glob(f"{V1_VAL}/ep_*.pt"))[:N_EPS]
eps = load_frames(files)
wins = []
for ep in eps:
    T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
    for t in range(0, T - WIN - 20, STRIDE):
        wins.append((ep, t))
wins = wins[:N_WIN]
OUT["val"] = {"dir": V1_VAL, "n_episodes": len(eps), "n_windows": len(wins)}
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


def qs(v):
    s = sorted(v)
    n = len(s)
    return {"mean": round(sum(s) / n, 6), "p50": round(s[n // 2], 6),
            "p95": round(s[min(n - 1, int(0.95 * n))], 6), "max": round(s[-1], 6),
            "frac_gt_0.01": round(sum(1 for x in s if x > 0.01) / n, 4),
            "frac_gt_0.1": round(sum(1 for x in s if x > 0.1) / n, 4),
            "frac_gt_1.0": round(sum(1 for x in s if x > 1.0) / n, 4)}


@torch.no_grad()
def run(K):
    rec = {"n": 0}
    ref_pick, tst_pick, bat_pick, nob_pick = [], [], [], []
    regret_fp16, regret_batch, regret_noint, margin, eid = [], [], [], [], []
    for ep, t in wins:
        last = torch.tensor([t + WIN - 1])
        fw = torch.as_tensor(ep.feats[t:t + WIN])[None].to(DEV).float().div_(255.0)
        aw = ep.actions[t:t + WIN][None].to(DEV)
        fa = ep.actions[t + WIN:t + WIN + 20][None].to(DEV)
        aw, _ = RO.append_ego(aw, fa, ep.poses, last, True, False, False, DEV)
        nav, _ = rl.nav_command(ep.poses, t + WIN - 1)
        v0 = float(ep.poses[t + WIN - 1, 3])
        mans = vocabulary(v0, K)
        # ONE fp32 encoder pass, shared by every arm — the encoder is not the
        # variable here and re-encoding would add its own noise.
        states = model.encode_window(fw)
        sout = model.strategic_policy(states, torch.tensor([nav], device=DEV))
        tout = model.tactical_policy(states, sout["ctx"])
        h = max(cfg.tactical_policy.waypoint_horizons)
        sg = tout["waypoints"][h][0, :2].float()

        b_ref, s_ref = sel.propose_and_score(states, aw, mans, sg, step_readout,
                                             sout["ctx"], batch_fan=False)
        b_bat, s_bat = sel.propose_and_score(states, aw, mans, sg, step_readout,
                                             sout["ctx"], batch_fan=True)
        prev = model.predictor
        model.predictor = eng_int
        try:
            b_tst, _ = sel.propose_and_score(states, aw, mans, sg, step_readout,
                                             sout["ctx"], batch_fan=True)
        finally:
            model.predictor = prev
        model.predictor = _StripIntent(eng_noint)
        try:
            b_nob, _ = sel.propose_and_score(states, aw, mans, sg, step_readout,
                                             sout["ctx"], batch_fan=True)
        finally:
            model.predictor = prev
        s_ref = s_ref.float().cpu()
        srt = torch.sort(s_ref).values
        ref_pick.append(b_ref)
        bat_pick.append(b_bat)
        tst_pick.append(b_tst)
        nob_pick.append(b_nob)
        margin.append(float(srt[1] - srt[0]))
        regret_fp16.append(float(s_ref[b_tst] - s_ref[b_ref]))
        regret_batch.append(float(s_ref[b_bat] - s_ref[b_ref]))
        regret_noint.append(float(s_ref[b_nob] - s_ref[b_ref]))
        eid.append(int(ep.episode_id))
    n = len(eid)
    ag_fp16 = [1.0 if a == b else 0.0 for a, b in zip(ref_pick, tst_pick)]
    ag_bat = [1.0 if a == b else 0.0 for a, b in zip(ref_pick, bat_pick)]
    ag_nob = [1.0 if a == b else 0.0 for a, b in zip(ref_pick, nob_pick)]
    rec = {"n": n, "n_episodes": len(set(eid)),
           "agreement_batching_only_CONTROL": {
               "agreement": round(sum(ag_bat) / n, 5),
               "n_flipped": int(n - sum(ag_bat)),
               "ci": episode_cluster_bootstrap(ag_bat, eid),
               "regret": qs(regret_batch)},
           "agreement_fp16_engine_WITH_intent": {
               "agreement": round(sum(ag_fp16) / n, 5),
               "n_flipped": int(n - sum(ag_fp16)),
               "ci": episode_cluster_bootstrap(ag_fp16, eid),
               "regret": qs(regret_fp16),
               "regret_ci_mean": episode_cluster_bootstrap(regret_fp16, eid)},
           "agreement_fp16_engine_NO_intent_THE_BUG": {
               "agreement": round(sum(ag_nob) / n, 5),
               "n_flipped": int(n - sum(ag_nob)),
               "ci": episode_cluster_bootstrap(ag_nob, eid),
               "regret": qs(regret_noint),
               "regret_ci_mean": episode_cluster_bootstrap(regret_noint, eid)},
           "margin_p50": round(sorted(margin)[n // 2], 6)}
    # flip rate conditioned on how DETERMINED the reference decision was
    buckets = [(0, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, 1e9)]
    rec["flip_rate_by_reference_margin"] = {}
    for lo, hi in buckets:
        idx = [i for i in range(n) if lo <= margin[i] < hi]
        if not idx:
            continue
        rec["flip_rate_by_reference_margin"][f"[{lo},{hi})"] = {
            "n": len(idx),
            "flip_rate": round(1 - sum(ag_fp16[i] for i in idx) / len(idx), 4),
            "mean_regret": round(sum(regret_fp16[i] for i in idx) / len(idx), 6)}
    return rec


for K in K_LIST:
    print(f"=== K={K} ===", flush=True)
    OUT[f"K{K}"] = run(K)
    print(json.dumps(OUT[f"K{K}"], indent=1)[:1800], flush=True)
    dump()
print("=== DONE ===", OUTJ, flush=True)
