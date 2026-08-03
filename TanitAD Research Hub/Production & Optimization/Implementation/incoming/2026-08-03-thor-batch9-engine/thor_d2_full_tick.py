#!/usr/bin/env python3
"""D2/D3 — the FULL wired tick on real frames, and the batch-9 engine's numerics.

Every Thor tick published so far was **composed arithmetically** from an encoder
number plus ``K x predictor``: it excluded ``step_readout`` decode, candidate
scoring and the tactical/strategic head decode, and it rolled ONE candidate. This
script wires the real thing — ``encode_window -> strategic_policy ->
tactical_policy -> TacticalSelector.propose_and_score`` over the **9**-candidate
vocabulary — and times it end to end, p50 AND p95.

Three arms on IDENTICAL windows:
  A  fp32 eager, serialised fan                    (today's code, today's precision)
  B  bf16 encoder + TRT-fp16 **batch-1** engine, serialised fan   (the SHIPPED shape)
  C  bf16 encoder + TRT-fp16 **dynamic 1..9** engine, BATCHED fan (the FIX)

and D3, on the same real activations:
  * batch-9 row i vs batch-1 — does batching change the engine's ANSWER? (this is
    what decides whether the 2026-08-03 four-family gate, which ran a 1..8 engine
    at rollout batch 8, transfers to the 9-candidate fan);
  * 1-step and K-step rel-err vs eager fp32 on REAL activations — the compounding
    question, restated for the batched engine;
  * **selected-candidate agreement** A vs C over N windows with an
    episode-cluster bootstrap CI — a TACTICAL-family decision metric, which is
    the accuracy claim this change actually needs.

⛔ v1 is 256x256 SQUARE; the val cache ``physicalai-val-0c5f7dac3b11`` is
256x256. Asserted before any frame is fed.
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
OUTJ = f"{HOME}/thor_d2_full_tick_K{os.environ.get('K_PRIM', '4')}.json"
V1_CKPT = f"{HOME}/models/flagship-v1-speedjerk/ckpt.pt"
V1_VAL = f"{HOME}/valdata/physicalai-val-0c5f7dac3b11"
PLAN_DYN = f"{HOME}/trt_d1/v1_dyn1-9_fp16.plan"
PLAN_B1 = f"{HOME}/trt_d1/v1_static_b1_fp16.plan"
WIN, N_CAND = 8, 9
STRIDE = int(os.environ.get("STRIDE", "40"))
N_EPS = int(os.environ.get("N_EPS", "12"))
K_PRIM = int(os.environ.get("K_PRIM", "4"))        # TacticalConfig.horizon
N_TICKS = int(os.environ.get("N_TICKS", "60"))     # timed ticks per arm
N_DEC = int(os.environ.get("N_DEC", "200"))        # windows for the decision gate
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
       "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "budget_ms": 100.0, "n_candidates": N_CAND, "k_primitive": K_PRIM,
       "n_ticks": N_TICKS}


def dump():
    OUT["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(OUTJ, "w") as f:
        json.dump(OUT, f, indent=2)


def stats(ms):
    s = sorted(ms)
    n = len(s)
    return {"n": n, "p50": round(st.median(s), 3),
            "p95": round(s[min(n - 1, int(0.95 * n))], 3),
            "p99": round(s[min(n - 1, int(0.99 * n))], 3),
            "mean": round(st.fmean(s), 3), "max": round(s[-1], 3)}


def relerr(a, b):
    a = a.double().flatten().cpu()
    b = b.double().flatten().cpu()
    return float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-30))


# ===================================================================== model + data
print("=== load REAL v1 (256px, its trained raster) ===", flush=True)
cfg = flagship4b_config()
object.__setattr__(cfg.predictor, "action_dim", 3)
if getattr(cfg, "tactical_pred", None) is not None:
    object.__setattr__(cfg.tactical_pred, "action_dim", 3)
object.__setattr__(cfg.encoder, "grad_checkpoint", False)
model = WorldModel(cfg)
ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])                       # STRICT
grounding = HierarchicalGrounding(model.state_dim)
grounding.load_state_dict(ck["grounding"])               # STRICT
model, grounding = model.to(DEV).eval(), grounding.to(DEV).eval()
step_readout = grounding.step["op"]
S, A = model.state_dim, 3
OUT["model"] = {"ckpt": V1_CKPT, "step": int(ck.get("step", -1)),
                "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "raster": f"{cfg.encoder.image_size}x{cfg.encoder.image_size} SQUARE",
                "strict_load": True, "state_dim": S,
                "has_tactical_policy": model.tactical_policy is not None,
                "has_strategic_policy": model.strategic_policy is not None}
print(OUT["model"], flush=True)

files = sorted(glob.glob(f"{V1_VAL}/ep_*.pt"))[:N_EPS]
eps = load_frames(files)
_shape = tuple(torch.as_tensor(eps[0].feats[:1]).shape)
assert _shape[-2:] == (cfg.encoder.image_size, cfg.encoder.image_size), (
    f"⛔ RASTER MISMATCH: val cache is {_shape[-2:]}, v1 trained at "
    f"{cfg.encoder.image_size}px — refusing to feed an arm a raster it never saw")
OUT["val"] = {"dir": V1_VAL, "n_episodes": len(eps), "frame_shape": _shape,
              "raster_assert": "PASSED"}
print(OUT["val"], flush=True)

# ===================================================================== engines
dyn = TRTPredictor(PLAN_DYN, device=DEV)
b1 = TRTPredictor(PLAN_B1, device=DEV)
OUT["engines"] = {"dyn": {"plan": PLAN_DYN, "profile_states": dyn.profile_shapes()},
                  "b1": {"plan": PLAN_B1, "profile_states": b1.profile_shapes()}}
assert dyn.profile_shapes()[2][0] >= N_CAND, "dynamic engine cannot serve the 9-fan"
print(OUT["engines"], flush=True)

# ===================================================================== the vocabulary
# 3 steer x 3 accel = TacticalConfig.n_maneuvers (9). ⚠️ INSTRUMENT, not a trained
# artifact: the programme has no committed primitive table, so the values are
# stated here and held IDENTICAL across arms — a decision-agreement test needs
# the same candidates on both sides, not the right ones.
STEER, ACCEL = (-0.15, 0.0, 0.15), (-1.0, 0.0, 1.0)
_CLS = {"lane_keep": 0, "turn_left": 1, "turn_right": 2, "accelerate": 3,
        "brake_stop": 4}


def _cls(steer, accel):                     # refb priority: turn > brake > accel
    if steer > 0.01:
        return _CLS["turn_left"]
    if steer < -0.01:
        return _CLS["turn_right"]
    if accel < -0.01:
        return _CLS["brake_stop"]
    if accel > 0.01:
        return _CLS["accelerate"]
    return _CLS["lane_keep"]


def vocabulary(v0: float, k=K_PRIM):
    """9 primitives [K, 3] = (steer, accel, speed-channel held at v0)."""
    mans = []
    for s_ in STEER:
        for a_ in ACCEL:
            act = torch.tensor([[s_, a_, v0]] * k, device=DEV, dtype=torch.float32)
            mans.append(Maneuver(f"s{s_}_a{a_}", act, maneuver_class=_cls(s_, a_)))
    assert len(mans) == N_CAND
    return mans


OUT["vocabulary"] = {"steer": STEER, "accel": ACCEL, "k": K_PRIM, "n": N_CAND,
                     "class_map": "refb priority turn > brake > accelerate",
                     "evidence_class": "INSTRUMENT (ours) — no committed primitive "
                                       "table exists; identical across arms"}

# ===================================================================== windows
wins = []
for ep in eps:
    T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
    for t in range(0, T - WIN - 20, STRIDE):
        wins.append((ep, t))
print(f"windows available: {len(wins)}", flush=True)


def prep(ep, t):
    last = torch.tensor([t + WIN - 1])
    fw = torch.as_tensor(ep.feats[t:t + WIN])[None].to(DEV).float().div_(255.0)
    aw = ep.actions[t:t + WIN][None].to(DEV)
    fa = ep.actions[t + WIN:t + WIN + 20][None].to(DEV)
    aw, _ = RO.append_ego(aw, fa, ep.poses, last, True, False, False, DEV)
    nav, _ok = rl.nav_command(ep.poses, t + WIN - 1)
    v0 = float(ep.poses[t + WIN - 1, 3])
    return fw, aw, int(nav), v0


sel = TacticalSelector(model, probe_imag=None)


def tick(ep, t, *, bf16: bool, pred, batch_fan: bool):
    """ONE deployed tick: encode -> strategic -> tactical -> 9-candidate fan.

    ``pred`` swaps the operative predictor (eager module or TRT engine); both are
    nn.Modules, which is required — ``model.predictor = <plain object>`` raises.
    """
    fw, aw, nav, v0 = prep(ep, t)
    mans = vocabulary(v0)
    navt = torch.tensor([nav], device=DEV)
    prev = model.predictor
    model.predictor = pred
    try:
        with torch.no_grad():
            if bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    states = model.encode_window(fw)
                states = states.float()
            else:
                states = model.encode_window(fw)
            sout = model.strategic_policy(states, navt)
            tout = model.tactical_policy(states, sout["ctx"])
            h = max(cfg.tactical_policy.waypoint_horizons)
            subgoal = tout["waypoints"][h][0, :2].float()
            best, scores = sel.propose_and_score(states, aw, mans, subgoal,
                                                 step_readout, sout["ctx"],
                                                 batch_fan=batch_fan)
        return {"best": int(best), "scores": scores.float().cpu(),
                "route": int(sout["route_logits"].argmax(-1)[0]),
                "man": int(tout["maneuver_logits"].argmax(-1)[0]),
                "states": states}
    finally:
        model.predictor = prev


ARMS = {
    "A_fp32_eager_serialised": dict(bf16=False, pred=model.predictor, batch_fan=False),
    "B_bf16_trtB1_serialised_SHIPPED": dict(bf16=True, pred=b1, batch_fan=False),
    "C_bf16_trtDYN_batched_FIX": dict(bf16=True, pred=dyn, batch_fan=True),
    "D_bf16_eager_batched_NOENGINE": dict(bf16=True, pred=model.predictor, batch_fan=True),
}

print("=== D2: FULL TICK, measured end to end ===", flush=True)
timing = {}
for name, kw in ARMS.items():
    for ep, t in wins[:5]:                                   # warmup
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
    timing[name]["pct_of_budget_p50"] = round(timing[name]["p50"], 1)
    print(name, timing[name], flush=True)
    OUT["full_tick"] = timing
    dump()

# stage breakdown for arm C (where the remaining time actually is)
print("=== D2b: stage breakdown, arm C ===", flush=True)
brk = {"encode_bf16": [], "heads": [], "fan": []}
for ep, t in wins[:N_TICKS]:
    fw, aw, nav, v0 = prep(ep, t)
    mans, navt = vocabulary(v0), torch.tensor([nav], device=DEV)
    prev = model.predictor
    model.predictor = dyn
    try:
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = model.encode_window(fw)
            states = states.float()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            sout = model.strategic_policy(states, navt)
            tout = model.tactical_policy(states, sout["ctx"])
            h = max(cfg.tactical_policy.waypoint_horizons)
            subgoal = tout["waypoints"][h][0, :2].float()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            sel.propose_and_score(states, aw, mans, subgoal, step_readout,
                                  sout["ctx"], batch_fan=True)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
    finally:
        model.predictor = prev
    brk["encode_bf16"].append((t1 - t0) * 1e3)
    brk["heads"].append((t2 - t1) * 1e3)
    brk["fan"].append((t3 - t2) * 1e3)
OUT["stage_breakdown_armC"] = {k: stats(v) for k, v in brk.items()}
print(OUT["stage_breakdown_armC"], flush=True)
dump()

# ===================================================================== D3 numerics
print("=== D3: batch-9 vs batch-1 numerics on REAL activations ===", flush=True)
real_states, real_acts = [], []
for ep, t in wins[:32]:
    fw, aw, nav, v0 = prep(ep, t)
    with torch.no_grad():
        real_states.append(model.encode_window(fw))
    real_acts.append(aw)
RS = torch.cat(real_states)                      # [32, W, S] REAL encoder output
RA = torch.cat(real_acts)

# (a) does BATCHING change the engine's answer? row i of a 9-batch vs a 1-batch.
rows = []
for i in range(9):
    with torch.no_grad():
        rows.append(dyn(RS[i:i + 1], RA[i:i + 1])[1])
one_by_one = torch.cat(rows)
with torch.no_grad():
    batched9 = dyn(RS[:9], RA[:9])[1]
    eager9 = model.predictor(RS[:9], RA[:9])[1]
    b1_rows = torch.cat([b1(RS[i:i + 1], RA[i:i + 1])[1] for i in range(9)])
OUT["d3_batch_consistency"] = {
    "dyn_b9_vs_dyn_b1_rel_err": round(relerr(batched9, one_by_one), 9),
    "dyn_b9_vs_eager_rel_err": round(relerr(batched9, eager9), 8),
    "dyn_b1_vs_eager_rel_err": round(relerr(one_by_one, eager9), 8),
    "static_b1_vs_eager_rel_err": round(relerr(b1_rows, eager9), 8),
    "why": ("TRT may pick different tactics per profile shape; if b9 != b1 the "
            "2026-08-03 four-family gate (a 1..8 engine at rollout batch 8) does "
            "not automatically transfer to the 9-candidate fan")}
print(OUT["d3_batch_consistency"], flush=True)
dump()

# (b) compounding on REAL activations, through the BATCHED fan
K_LONG = 20


def roll(pred, n, k, batched):
    s = RS[:n].clone()
    a = RA[:n].clone()
    prm = torch.stack([m.actions for m in vocabulary(12.0, k)])[:n] if batched else None
    outs = []
    for j in range(k):
        a = torch.roll(a, -1, dims=1)
        if batched:
            a[:, -1] = prm[:, j]
        with torch.no_grad():
            z = pred(s, a)[1]
        outs.append(z)
        s = torch.roll(s, -1, dims=1)
        s[:, -1] = z
    return outs


eg = roll(model.predictor, 9, K_LONG, True)
tg = roll(dyn, 9, K_LONG, True)
comp = {"step1_rel_err": round(relerr(tg[0], eg[0]), 8),
        f"step{K_LONG}_rel_err": round(relerr(tg[-1], eg[-1]), 8)}
comp["growth"] = round(comp[f"step{K_LONG}_rel_err"] / max(comp["step1_rel_err"], 1e-30), 3)
comp["per_step"] = [round(relerr(t_, e_), 8) for t_, e_ in zip(tg, eg)]
OUT["d3_compounding_real_activations_batched"] = comp
print(comp, flush=True)
dump()

# (c) SELECTED-CANDIDATE agreement A (fp32 eager) vs C (bf16+engine, batched)
print("=== D3c: selected-candidate agreement ===", flush=True)
agree, eid, dscore = [], [], []
for ep, t in wins[:N_DEC]:
    ra = tick(ep, t, bf16=False, pred=model.predictor, batch_fan=False)
    rc = tick(ep, t, bf16=True, pred=dyn, batch_fan=True)
    agree.append(1.0 if ra["best"] == rc["best"] else 0.0)
    dscore.append(float((ra["scores"] - rc["scores"]).abs().max()))
    eid.append(int(ep.episode_id))
OUT["d3_decision_agreement"] = {
    "n_windows": len(agree), "n_episodes": len(set(eid)),
    "agreement": round(float(sum(agree) / len(agree)), 5),
    "n_flipped": int(len(agree) - sum(agree)),
    "max_abs_score_delta": round(max(dscore), 6),
    "bar": 0.953, "family": "TACTICAL (decision quality)"}
try:
    from taniteval.ci import episode_cluster_bootstrap
    OUT["d3_decision_agreement"]["ci_episode_cluster_bootstrap"] = \
        episode_cluster_bootstrap(agree, eid)
except Exception as e:                                  # noqa: BLE001
    OUT["d3_decision_agreement"]["ci_error"] = repr(e)
print(OUT["d3_decision_agreement"], flush=True)
dump()
print("=== DONE ===", OUTJ, flush=True)
