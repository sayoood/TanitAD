#!/usr/bin/env python3
"""D-RL-REFCV3-MIN — the minimal pre-registered *RL stage on/off* experiment on the
FROZEN refcv3 @ 40,284. ⛔ PREPARED, NOT LAUNCHED (2026-09-05, DeployFlyWheel prep
for the Master Mind; SPEC: `TanitAD Research Lab/Deployment & Optimization/Research/
2026-09-05-refc-rl-readiness/SPEC.md`).

ONE FILE, FOUR MODES — so every import the ANALYSIS needs is paid at start-up, before
any GPU second is spent (an analysis-time ModuleNotFoundError has already destroyed a
completed two-arm rollout, CLAUDE.md):

  --mode preflight   imports · loads the real ckpt · trainable-surface + gradient-flow
                     probe · reward NO-FUTURE-LEAK test · reward audit at the scored
                     geometry · step-0 trust-region divergence == 0 · N timing steps at
                     lr=0 (weights asserted UNCHANGED) → preflight.json. No checkpoint.
  --mode fitlist     the RL-fit clip list (train-split ids NOT in the eval split) → txt
  --mode humanflag   ⛔ the H-RL-THRESH-1 check: how often the reward's own scene
                     context flags the HUMAN driver's future (static vs track lead),
                     hold-v0 and frozen references, PASS/FAIL at --human-flag-max. 0 GPU.
  --mode arm         ONE arm (rl | reg_echo | ctrl0): before-readout → run_posttrain →
                     ckpt_after.pt (+ the trainer's config.json beside it) → after-readout
  --mode verdict     the committed exit, selected MECHANICALLY from SPEC §6 over the
                     paired records + readouts. Never by reading a table.

WHAT IT REUSES (nothing copied): `taniteval/tools/refcv3_arm.py` (checkpoint loader,
the trainer's exact window contract, nav join, lead-block reader), `tanitad.rl` (the
DDv2-style library: surrogate sampler, GRPO+truncated advantage, veto, trust-region
anchor, counters, done-marker), `tanitad.eval.echo_gate` (imported so the eval stage's
gate is known present), `paired_openloop.py` / `openloop_suite.py` (the binding eval).

THE REWARD CARRIES NO FUTURE. ctx = {v0 at t0, the lead's position at its FIRST sample
(t0+0.2 s) held STATIC, dt}. `reward_ctx()` reads only `pose_last` and the lead block;
the preflight PROVES it by permuting every future_* field of the batch and asserting the
ctx is bit-identical. `gt_traj` enters the ctx ONLY on the deliberate-regression arm
`reg_echo`, whose reward is `gt_similarity` inside the advantage — the fan-collapse
objective rewards.py documents — and the gate G-FAN of SPEC §6 must catch it.

THE FAN IS SCORED ON ITS 2 s PREFIX. refcv3 emits 8 slots at (0.5,1,1.5,2,3,4,5,6) s — a
NON-uniform grid — while `rewards.kinematics` assumes one dt. Slots 0-3 are uniform at
0.5 s (the same index-select the `2s` eval grid uses), the origin is prepended so the
2 s displacement is the whole prefix. The trust region constrains the FULL 8-slot fan.

Tier: everything this script prints is T0 training-side. The capability read is the
taniteval harness (T1 stamped, OPEN LOOP), four families, paired episode-cluster
bootstrap — see the launch chain.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from types import SimpleNamespace

# --------------------------------------------------------------------------- #
# paths — the namespace-shadow guard, then EVERY analysis-time import, up front  #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(os.path.dirname(_HERE))
_STACK = os.path.join(_REPO, "stack")
_TE = os.path.join(_REPO, "taniteval")
_TE_TOOLS = os.path.join(_TE, "tools")
_SCRIPTS = os.path.join(_STACK, "scripts")


def _bootstrap_paths() -> None:
    for p in (_STACK, _TE, _TE_TOOLS, _SCRIPTS):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    m = sys.modules.get("taniteval")
    if m is not None:
        want = os.path.normcase(os.path.abspath(os.path.join(_TE, "taniteval")))
        paths = [os.path.normcase(os.path.abspath(p))
                 for p in (getattr(m, "__path__", None) or [])]
        if want not in paths:
            for k in [k for k in sys.modules
                      if k == "taniteval" or k.startswith("taniteval.")]:
                del sys.modules[k]


_bootstrap_paths()

IMPORT_LOG: list[str] = []


def _load_by_path(name: str, path: str):
    if not os.path.exists(path):
        raise SystemExit(f"[rl-min] ⛔ required sibling {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    IMPORT_LOG.append(f"{name} <- {path}")
    return mod


# ⛔ the analysis chain, imported NOW. Each of these is consumed AFTER the GPU work.
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

import taniteval.ci                                                  # noqa: E402,F401
import taniteval.four_families                                       # noqa: E402,F401
import taniteval.lead_metrics                                        # noqa: E402,F401
from tanitad.eval import echo_gate as EG                             # noqa: E402,F401
from tanitad.rl import (HACKABLE_WEIGHTS, DEFAULT_WEIGHTS,           # noqa: E402
                        PostTrainConfig, RewardSpec, audit as AUD,
                        rewards as RW)
from tanitad.rl.anchor import ReferencePolicy, trajectory_divergence  # noqa: E402
from tanitad.rl.posttrain import run_posttrain, select_trainable     # noqa: E402
from tanitad.rl.refcv3_adapter import sample_offsets                 # noqa: E402

ARM = _load_by_path("refcv3_arm_for_rl", os.path.join(_TE_TOOLS, "refcv3_arm.py"))
PAIRED = _load_by_path("paired_openloop_for_rl",
                       os.path.join(_TE_TOOLS, "paired_openloop.py"))
SUITE = _load_by_path("openloop_suite_for_rl",
                      os.path.join(_TE_TOOLS, "openloop_suite.py"))
CRITERIA_PATH = os.path.join(_REPO, "tools", "criteria_check.py")
REGISTRY_PATH = os.path.join(_REPO, "products", "P7-TanitEval", "CRITERIA_REGISTRY.json")
for _p in (CRITERIA_PATH, REGISTRY_PATH):
    if not os.path.exists(_p):
        raise SystemExit(f"[rl-min] ⛔ the eval stage needs {_p} and it is absent — "
                         "refusing at start-up rather than after the arms ran")
import refb_labels                                                   # noqa: E402

# --------------------------------------------------------------------------- #
# constants — every one of them is a SPEC field, none is inferred               #
# --------------------------------------------------------------------------- #
DT_REWARD_S = 0.5            # slots 0-3 of V3_HORIZONS are uniform at 0.5 s
N_REWARD_SLOTS = 4           # (5,10,15,20) frames @10 Hz -> 0.5..2.0 s = the `2s` grid
FAR_LEAD_X_M = 1.0e6         # no-lead sentinel: CONSTANT per window -> cancels in both advantages
LEAD_LEN_DEFAULT_M = 4.5     # rewards.py default; per-row size_x is RECORDED, not fed (scalar contract)
EXPECT_BASE_STEP = 40284
#: gate 5 tolerance. A MODE MISMATCH reads m^2-scale (2.774 m^2 on the v2.1 pilot,
#: TRAIN-C5); a frozen deepcopy on CUDA reads 1e-11 m^2 (MEASURED 2026-09-05, refcv3
#: @ 40,284, live-vs-live exactly 0) — kernel-selection rounding, not policy drift.
ANCHOR_STEP0_TOL_M2 = 1e-8
#: HOW THE LEAD ENTERS THE REWARD CONTEXT (set from --lead-mode in main()).
#:   "track"  — the lead agent's OWN 10-sample track (obstacle.offline join,
#:              build_lead_block_b1.py: t0 frame, 0.2–2.0 s) resampled onto the
#:              0.5 s reward grid; contact, headway and the TTC veto are
#:              TIME-ALIGNED. DEFAULT since 2026-09-05. It is privileged (not an
#:              inference input) but it is ANOTHER agent's motion, never the
#:              ego's — the preflight proves the ctx is invariant to the ego future.
#:   "static" — the predecessor's legacy: the first sample held fixed over the
#:              horizon. Kept ONLY as the recorded comparison: `--mode humanflag`
#:              measures how often it flags the human driver's own future.
LEAD_MODE = "track"
GRID_S = (0.0, 0.5, 1.0, 1.5, 2.0)   # the 5-point reward prefix (origin + 4 slots)

#: the GENERATOR surface — everything in the anchored-diffusion decoder that
#: produces the fan …
TRAINABLE_PREFIXES = ("core.decoder",)
#: … MINUS the ranking surfaces that live INSIDE the decoder. `conf_head` is
#: the v2.1 selector surface (sel_score = conf + grafts, refc.py:1379/1589-1600);
#: the *_to_anchor grafts and the four gates add to the RANK only. Excluded and
#: RECORDED (the pilot's `exclude_prefixes` idiom, refc21 pilot), never trained.
EXCLUDE_PREFIXES = ("core.decoder.conf_head", "core.decoder.lat_to_anchor",
                    "core.decoder.lon_to_anchor", "core.decoder.maneuver_to_anchor",
                    "core.decoder.route_to_anchor", "core.decoder.cons_gate",
                    "core.decoder.goal_gate", "core.decoder.goal_dist_gate",
                    "core.decoder.lan_gate")
#: the tripwire: must never require grad, whatever else is configured.
FORBIDDEN_PREFIXES = ("scorer", "conf_head", "phi_tac", "str_goal_head", "gstr_embed",
                      "gstr_film", "tac_goal_head", "tac_latent_proj", "nav_inject",
                      "core.encoder", "core.strategic", "core.aux", "core.route",
                      "core.maneuver")
#: keys that may NEVER reach the reward ctx on the honest arm (future of ANY kind)
FORBIDDEN_FUTURE_CTX = frozenset({"gt_traj", "future_poses", "future_poses_ext",
                                  "future_actions", "future_frames", "goal_tac",
                                  "lead_future"})

ARMS = {
    # name: (reward weights, w_anchor, lr, steps)  — ONE variable across rl/base:
    # the RL stage. reg_echo is the deliberate regression; ctrl0 the reproduction control.
    "rl":       dict(weights=dict(DEFAULT_WEIGHTS), w_anchor=1.0, lr=1e-5, steps=2000),
    "reg_echo": dict(weights={"gt_similarity": 1.0}, w_anchor=0.0, lr=1e-5, steps=2000),
    "ctrl0":    dict(weights=dict(DEFAULT_WEIGHTS), w_anchor=1.0, lr=0.0, steps=200),
}


def _p(*a):
    print(*a, flush=True)


def _sha256_state(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# corpus: the trainer's OWN window contract, via refcv3_arm.build_corpus         #
# --------------------------------------------------------------------------- #
def open_corpus(episodes: str, labels: str, cfg, prov, lru: int):
    ns = SimpleNamespace(episodes=episodes, lru=lru, episodes_n=0, labels=labels,
                         nav_source="v72")
    eps, files, clip_ids, ds, lman, join, src, raw_off = ARM.build_corpus(ns, cfg, prov)
    return SimpleNamespace(eps=eps, files=files, clip_ids=clip_ids, ds=ds, lman=lman,
                           join=join, nav_src=src, raw_off=raw_off, W=int(cfg.core.window))


def load_lead_block(path: str | None):
    if not path:
        return None, {}, {}
    blk, idx, meta = ARM.ra.load_lead_block_rows(path)
    leads = np.asarray(blk["leads"], dtype=np.float64)          # [R, K, 2]
    has = np.asarray(blk["has_lead"]).astype(bool).reshape(-1)
    lens = np.asarray(blk["lead_lens"], dtype=np.float64).reshape(-1)
    raw = np.load(path, allow_pickle=True)
    ts_rel = (np.asarray(raw["ts_rel_s"], dtype=np.float64).reshape(-1)
              if "ts_rel_s" in raw.files else None)
    gt_tg = (np.asarray(raw["gt_time_gap_min_s"], dtype=np.float64).reshape(-1)
             if "gt_time_gap_min_s" in raw.files else None)
    return SimpleNamespace(leads=leads, has=has, lens=lens, idx=idx, ts_rel=ts_rel,
                           gt_time_gap=gt_tg,
                           clips={c for (c, _f) in idx}), meta, {"n_rows": int(leads.shape[0])}


def scoreable_windows(corp, lead, horizon_frames: int = 20):
    """Every (window index) whose 2 s future is inside the episode. Identical across
    arms by construction (seeded sampling happens on this list)."""
    ds, W = corp.ds, corp.W
    out = []
    for wi, (e_i, t) in enumerate(ds.index):
        T = int(ds.episodes[e_i].poses.shape[0])
        if t + W - 1 + horizon_frames > T - 1:
            continue
        out.append(wi)
    return out


def lead_row(corp, lead, wi):
    """(has_lead, lead_xy at its FIRST sample, lead_len_m) for the window's t0."""
    if lead is None:
        return False, (FAR_LEAD_X_M, 0.0), LEAD_LEN_DEFAULT_M
    e_i, t = corp.ds.index[wi]
    t0_provider = t + corp.W - 1
    raw_frame = int(t0_provider + corp.raw_off)          # v2_dataset.py:36-38
    row = lead.idx.get((corp.clip_ids[e_i], raw_frame))
    if row is None or not bool(lead.has[row]):
        return False, (FAR_LEAD_X_M, 0.0), LEAD_LEN_DEFAULT_M
    xy = lead.leads[row, 0]                              # earliest sample (t0 + 0.2 s)
    if not np.all(np.isfinite(xy)):
        return False, (FAR_LEAD_X_M, 0.0), LEAD_LEN_DEFAULT_M
    return True, (float(xy[0]), float(xy[1])), float(lead.lens[row])


def resample_track(track: np.ndarray, ts_rel: np.ndarray) -> np.ndarray:
    """[K, 2] lead samples at ``ts_rel`` (0.2..2.0 s) -> [5, 2] at GRID_S. t = 0 is
    a linear extrapolation from the first two samples (one 0.2 s step back)."""
    out = np.zeros((len(GRID_S), 2))
    for i, t in enumerate(GRID_S):
        if t < ts_rel[0]:
            slope = (track[1] - track[0]) / (ts_rel[1] - ts_rel[0])
            out[i] = track[0] + slope * (t - ts_rel[0])
        else:
            out[i, 0] = np.interp(t, ts_rel, track[:, 0])
            out[i, 1] = np.interp(t, ts_rel, track[:, 1])
    return out


def lead_track(corp, lead, wi) -> torch.Tensor:
    """The lead's track on the reward grid, [5, 2] (t0 frame). No lead -> the far
    sentinel held over the horizon (CONSTANT per window -> cancels in both advantages)."""
    has, xy, _ln = lead_row(corp, lead, wi)
    if not has or lead is None or lead.ts_rel is None:
        return torch.tensor([[FAR_LEAD_X_M, 0.0]] * len(GRID_S), dtype=torch.float32)
    e_i, t = corp.ds.index[wi]
    row = lead.idx.get((corp.clip_ids[e_i], int(t + corp.W - 1 + corp.raw_off)))
    track = np.asarray(lead.leads[row], dtype=np.float64)
    if not np.all(np.isfinite(track)):
        return torch.tensor([[xy[0], xy[1]]] * len(GRID_S), dtype=torch.float32)
    return torch.tensor(resample_track(track, lead.ts_rel), dtype=torch.float32)


def build_batch(corp, lead, wis, device, *, with_gt: bool):
    """items -> the MODEL batch + the BATCH-SHAPED reward ctx inputs."""
    tr = ARM.trainer()
    items = [corp.ds[wi] for wi in wis]
    frames = torch.stack([it["frames"] for it in items])            # [B, W, 9, H, W'] u8
    nav = torch.stack([it["nav_cmd"] for it in items]).long()
    pose_last = torch.stack([it["pose_last"] for it in items]).float()
    v0 = pose_last[:, 3]
    has, xy, ln = zip(*[lead_row(corp, lead, wi) for wi in wis])
    b = {"frames": tr.frames_to_device(frames, device),
         "nav_cmd": nav.to(device), "v0": v0.to(device),
         "has_lead": list(has),
         "lead_xy": torch.tensor(xy, dtype=torch.float32, device=device),   # [B, 2]
         "lead_track": torch.stack([lead_track(corp, lead, wi) for wi in wis]).to(device),  # [B, 5, 2]
         "lead_len_m": list(ln), "wis": list(wis)}
    if with_gt:
        horizons = list(ARM_HORIZONS)
        fut = torch.stack([it["future_poses_ext"] for it in items]).float()
        gt8 = refb_labels.waypoint_targets(pose_last, fut, horizons)          # [B, 8, 2]
        b["gt_traj"] = gt8[:, :N_REWARD_SLOTS].to(device)                     # [B, 4, 2]
    return b


ARM_HORIZONS: list[int] = []      # filled from the loaded model's cfg


def with_origin(x: torch.Tensor) -> torch.Tensor:
    """[..., 4, 2] -> [..., 5, 2] with the ego origin prepended (the 2 s prefix)."""
    z = torch.zeros(*x.shape[:-2], 1, 2, dtype=x.dtype, device=x.device)
    return torch.cat([z, x], dim=-2)


def reward_ctx(batch: dict, *, S5: int, extras: dict | None = None) -> dict:
    """⛔ THE ONLY READER OF SCENE FACTS FOR THE REWARD. Reads v0 (t0) and the lead's
    first sample. Nothing here touches a future_* field — proved by the preflight."""
    B = batch["v0"].shape[0]
    lead = batch["lead_xy"]                                             # [B, 2]
    ctx = {"dt": DT_REWARD_S,
           "v0": batch["v0"].reshape(B, 1, 1),                          # [B, 1, 1]
           "lead_len_m": LEAD_LEN_DEFAULT_M}
    if LEAD_MODE == "track":
        if S5 != len(GRID_S):
            raise ValueError(f"track mode scores the {len(GRID_S)}-point prefix, got S5={S5}")
        # MOVING lead, time-aligned; NO static `obstacles` key, so `collision`
        # takes rewards._collision's lead_path branch (per-step contact).
        ctx["lead_path"] = batch["lead_track"].reshape(B, 1, 1, S5, 2)
    else:
        ctx["obstacles"] = lead.reshape(B, 1, 1, 1, 2)                   # [B,1,1,K=1,2]
        ctx["lead_path"] = lead.reshape(B, 1, 1, 1, 2).expand(B, 1, 1, S5, 2)  # static
    if extras:
        ctx.update(extras)
    return ctx


def make_sample_fn(model, reference, *, echo_reward: bool):
    """(batch, cfg) -> (traj2 [B,N,G,5,2], logp [B,N,G], ctx, extras) — the 2 s PREFIX
    of the sampled fan goes to the reward; the FULL mean fan pair goes to the anchor."""
    def sample_fn(batch, cfg):
        out = model(batch["frames"], nav_cmd=batch["nav_cmd"], v0=batch["v0"],
                    steps=int(cfg.decoder_steps))
        anchor_traj = out["anchor_traj"]                                # [B, N, 8, 2]
        offset = out["offset"]                                          # [B, N, 8, 2]
        base = anchor_traj - offset
        off_g, logp = sample_offsets(offset, cfg)
        traj = base.unsqueeze(2) + off_g                                # [B, N, G, 8, 2]
        traj2 = with_origin(traj[..., :N_REWARD_SLOTS, :])              # [B, N, G, 5, 2]
        ctx = reward_ctx(batch, S5=traj2.shape[-2])
        if echo_reward:
            # ⛔ DELIBERATE REGRESSION ONLY: the logged ego future INSIDE the advantage.
            ctx["gt_traj"] = with_origin(batch["gt_traj"]).reshape(
                traj2.shape[0], 1, 1, traj2.shape[-2], 2)
        else:
            bad = FORBIDDEN_FUTURE_CTX & set(ctx)
            if bad:
                raise RuntimeError(f"FUTURE LEAK into the reward ctx: {sorted(bad)}")
        extras = {}
        if reference is not None:
            with torch.no_grad():
                ref = reference(batch["frames"], nav_cmd=batch["nav_cmd"],
                                v0=batch["v0"], steps=int(cfg.decoder_steps))
            extras["anchor_pair"] = (anchor_traj, ref["anchor_traj"])
        return traj2, logp, ctx, extras
    return sample_fn


# --------------------------------------------------------------------------- #
# the T0 readout (before/after, deterministic eval mode, fixed window set)       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def readout(model, corp, lead, wis, device, *, decoder_steps: int, batch: int = 4):
    """R1 composed DEFAULT reward · R2 fan collision (lead-only) · R3 sel-ADE 2 s ·
    R_FAN endpoint spread · R_REACH mean |fan - bank| · R_ORACLE oracle-in-fan ADE 2 s
    (the fan-QUALITY readout the echo gate needs: an echo arm improves it while
    R_FAN collapses) · sel_idx per window.
    Per-window values are STORED so the paired bootstrap is computable."""
    was = model.training
    model.eval()
    spec = RewardSpec(weights=dict(DEFAULT_WEIGHTS), dt=DT_REWARD_S)
    bank = model.core.decoder.anchors.detach()                          # [N, 8, 2]
    rows = []
    for i in range(0, len(wis), batch):
        b = build_batch(corp, lead, wis[i:i + batch], device, with_gt=True)
        out = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=decoder_steps)
        fan = out["anchor_traj"]                                        # [B, N, 8, 2]
        fan2 = with_origin(fan[..., :N_REWARD_SLOTS, :])                # [B, N, 5, 2]
        ctx = reward_ctx(b, S5=fan2.shape[-2])
        r1 = spec(fan2, ctx)                                            # [B, N]
        coll = RW.COMPONENTS["collision"](fan2, ctx) < 0                # [B, N]
        sel = out["sel_idx"]                                            # [B]
        gt = b["gt_traj"]                                               # [B, 4, 2]
        chosen = out["traj"][:, :N_REWARD_SLOTS]                        # [B, 4, 2]
        r3 = (chosen - gt).norm(dim=-1).mean(dim=-1)                    # [B]
        end = fan[..., N_REWARD_SLOTS - 1, :]                           # [B, N, 2] the 2 s endpoint
        spread = end.std(dim=1).norm(dim=-1)                            # [B]
        reach = (fan - bank[None]).norm(dim=-1).mean(dim=(1, 2))        # [B]
        oracle = (fan[..., :N_REWARD_SLOTS, :] - gt[:, None]).norm(dim=-1).mean(dim=-1).min(dim=1).values  # [B] oracle-in-fan 2 s
        for j, wi in enumerate(b["wis"]):
            e_i, _t = corp.ds.index[wi]
            rows.append({"wi": int(wi), "eid": int(e_i), "clip": corp.clip_ids[e_i],
                         "has_lead": bool(b["has_lead"][j]),
                         "R1": float(r1[j].mean()), "R2": float(coll[j].float().mean()),
                         "R3": float(r3[j]), "R_FAN": float(spread[j]),
                         "R_REACH": float(reach[j]), "R_ORACLE": float(oracle[j]),
                         "sel_idx": int(sel[j])})
    model.train(was)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("R1", "R2", "R3", "R_FAN", "R_REACH", "R_ORACLE")}
    return {"n_windows": len(rows), "n_episodes": len({r["eid"] for r in rows}),
            "n_with_lead": int(sum(r["has_lead"] for r in rows)),
            "aggregate": agg, "per_window": rows, "eval_mode": True,
            "_tier": "T0 training-side diagnostic; the capability read is the taniteval "
                     "harness (T1, OPEN LOOP, four families)",
            "_evidence_class": "MEASURED (ours)"}


def paired_delta(before: dict, after: dict, key: str, reps: int = 4000, seed: int = 11):
    """Paired EPISODE-cluster bootstrap of the per-episode mean delta (house rule)."""
    b = {r["wi"]: r for r in before["per_window"]}
    a = {r["wi"]: r for r in after["per_window"]}
    per_ep: dict[int, list] = {}
    for wi in sorted(set(b) & set(a)):
        per_ep.setdefault(b[wi]["eid"], []).append(a[wi][key] - b[wi][key])
    d = np.array([np.mean(v) for v in per_ep.values()])
    if d.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "sep": False}
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(reps)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(d.size),
            "sep": bool((lo > 0) == (hi > 0)), "_estimator": "paired episode-cluster bootstrap"}


# --------------------------------------------------------------------------- #
# modes                                                                          #
# --------------------------------------------------------------------------- #
def load(a, device):
    model, cfg, targs, prov = ARM.load_model(a.ckpt, a.config, device, False)
    if int(prov.get("step", -1)) != int(a.expect_step):
        raise SystemExit(f"[rl-min] ⛔ checkpoint step {prov.get('step')} != "
                         f"--expect-step {a.expect_step}; refusing (the name proves nothing)")
    ARM_HORIZONS[:] = [int(h) for h in cfg.core.trajectory.horizons]
    if ARM_HORIZONS[:N_REWARD_SLOTS] != [5, 10, 15, 20]:
        raise SystemExit(f"[rl-min] ⛔ the first {N_REWARD_SLOTS} slots are {ARM_HORIZONS[:4]}, "
                         "not (5,10,15,20): the 0.5 s reward grid does not hold")
    model.eval()
    return model, cfg, targs, prov


def make_cfg(arm: str, a, prov, out_dir: str) -> PostTrainConfig:
    spec = ARMS[arm]
    return PostTrainConfig(
        method="grpo", group_size=int(a.group), normalize="none",
        noise_mode="multiplicative", noise_scale=float(a.noise),
        steps=int(a.steps if a.steps else spec["steps"]), batch=int(a.batch),
        lr=float(spec["lr"]), seed=int(a.seed), dt=DT_REWARD_S,
        decoder_steps=int(prov["decoder_steps"]),
        reward_weights=dict(spec["weights"]), w_anchor=float(spec["w_anchor"]),
        anchor_form="l2", w_imitation=0.0, train_mode_forward=False,
        freeze_trunk=True, trainable_prefixes=TRAINABLE_PREFIXES,
        exclude_prefixes=EXCLUDE_PREFIXES, forbidden_prefixes=FORBIDDEN_PREFIXES,
        out_dir=out_dir, run_name=f"refcv3-40284-rlmin-{arm}", save_every=0)


def mode_fitlist(a) -> int:
    import gzip
    def ids(path):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return {json.loads(l)["clip_id"] for l in fh if l.strip()}
    tr, ev = ids(a.train_labels), ids(a.eval_labels)
    if tr & ev:
        raise SystemExit(f"[rl-min] ⛔ train ∩ eval = {len(tr & ev)} clips — the label "
                         "split is not disjoint; refusing")
    cand = sorted(tr - ev)
    random.Random(int(a.seed)).shuffle(cand)
    pick = cand[:int(a.n_fit)]
    # newline="\n": a CRLF list feeds `818dbc51-...\r` to scp and every copy fails
    # with "No such file" (MEASURED 2026-09-05, 7/8 names) — LF only, on purpose.
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(pick) + "\n")
    _p(f"[fitlist] {len(pick)} RL-fit clips from {len(cand)} train-only candidates "
       f"(train {len(tr)}, eval {len(ev)}, intersection 0) -> {a.out}")
    return 0


def _assert_disjoint(corp_fit_clips, eval_dir: str, eval_labels: str):
    import gzip
    have = {f.split(".")[0] for f in os.listdir(eval_dir) if f.endswith(".v2ep.pt")}
    with gzip.open(eval_labels, "rt", encoding="utf-8") as fh:
        lab = {json.loads(l)["clip_id"] for l in fh if l.strip()}
    inter = set(corp_fit_clips) & (have | lab)
    if inter:
        raise SystemExit(f"[rl-min] ⛔ GATE 3: {len(inter)} RL-fit clips are EVAL clips; refusing")
    return {"n_fit": len(set(corp_fit_clips)), "n_eval_files": len(have),
            "n_eval_labels": len(lab), "intersection": 0}


def mode_preflight(a) -> int:
    """Wrapper: the partial report is WRITTEN whenever a gate raises, so a failure
    carries its measurement instead of discarding it (2026-09-05: gate 5 fired with
    no number attached, and the cause had to be re-measured by a separate probe)."""
    rep: dict = {}
    try:
        return _preflight_body(a, rep)
    except BaseException as ex:                                      # noqa: BLE001
        rep["PASS"] = False
        rep["failure"] = f"{type(ex).__name__}: {ex}"
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=1, default=str)
            _p(f"[preflight] partial report written to {a.out} (PASS=False)")
        raise


def _preflight_body(a, rep: dict) -> int:
    t_all = time.time()
    device = a.device if torch.cuda.is_available() else "cpu"
    rep.update({"_what": "D-RL-REFCV3-MIN preflight — NOTHING trained, no checkpoint written",
                 "imports": list(IMPORT_LOG), "device": device,
                 "torch": torch.__version__, "_evidence_class": "MEASURED (ours)"})
    model, cfg, targs, prov = load(a, device)
    rep["lead_mode"] = LEAD_MODE
    rep["lead_path_provenance"] = (
        "track: the lead AGENT's own obstacle.offline positions at t0+0.2..2.0 s in the "
        "ego t0 frame (build_lead_block_b1.py), resampled to the 0.5 s grid — privileged, "
        "NOT derived from the ego future (proved below by permutation)"
        if LEAD_MODE == "track" else
        "static: the lead's FIRST sample (t0+0.2 s) held over the horizon — LEGACY")
    rep["model"] = {"ckpt": a.ckpt, "step": prov.get("step"), "arm": prov.get("arm"),
                    "decoder_steps": prov.get("decoder_steps"),
                    "n_params": sum(p.numel() for p in model.parameters()),
                    "horizons": ARM_HORIZONS}
    # 1. the trainable surface, on the REAL model ------------------------------
    pcfg = make_cfg("rl", a, prov, "")
    pcfg.validate()
    fr = select_trainable(model, pcfg)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    leak = [n for n in names if not n.startswith("core.decoder.")]
    sel_surface = [n for n in names if any(s in n for s in ("conf_head", "scorer", "_to_anchor",
                                                             "goal_gate", "cons_gate", "lan_gate"))]
    rep["surface"] = {**{k: v for k, v in fr.items() if k != "trainable_names_head"},
                      "trainable_names": names, "non_decoder_trainable": leak,
                      "selector_surface_trainable": sel_surface,
                      "all_decoder_param_names": [n for n, _ in model.named_parameters()
                                                  if n.startswith("core.decoder.")]}
    if leak or sel_surface:
        raise SystemExit(f"[preflight] ⛔ surface leak: non-decoder {leak[:4]} / selector {sel_surface[:4]}")
    _p(f"[preflight] trainable {fr['trainable_params']:,} / total {fr['total_params']:,} "
       f"({fr['trainable_fraction']:.2%}); excluded {len(fr['excluded_names'])} selector tensors")
    # 2. corpus + lead block ----------------------------------------------------
    corp = open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, lmeta, linfo = load_lead_block(a.lead_block)
    wis_all = scoreable_windows(corp, lead)
    rng = random.Random(int(a.seed))
    wis = sorted(rng.sample(wis_all, min(len(wis_all), int(a.batch) * 8)))
    with_lead = [wi for wi in wis_all if lead_row(corp, lead, wi)[0]]
    rep["corpus"] = {"episodes": a.episodes, "n_episodes": len(corp.eps),
                     "n_scoreable_windows": len(wis_all),
                     "n_windows_with_lead": len(with_lead),
                     "lead_block": a.lead_block, **linfo,
                     "labels_md5": corp.lman.md5, "nav_source": corp.nav_src,
                     "provider_to_raw_offset": corp.raw_off}
    if a.eval_episodes:
        rep["gate3_disjoint"] = _assert_disjoint(corp.clip_ids, a.eval_episodes, a.eval_labels)
    # 3. NO-FUTURE-LEAK: permute every future_* field, ctx must be bit-identical -
    b = build_batch(corp, lead, wis[:int(a.batch)], device, with_gt=True)
    ctx0 = reward_ctx(b, S5=5)
    b_perm = dict(b)
    B = b["v0"].shape[0]
    perm = torch.randperm(B)
    for k in ("gt_traj",):
        b_perm[k] = b[k][perm] * 3.7 + 11.0        # garbage future
    ctx1 = reward_ctx(b_perm, S5=5)
    same = all(torch.equal(ctx0[k], ctx1[k]) if torch.is_tensor(ctx0[k]) else ctx0[k] == ctx1[k]
               for k in ctx0)
    bad = FORBIDDEN_FUTURE_CTX & set(ctx0)
    rep["no_future_leak"] = {"ctx_keys": sorted(ctx0), "forbidden_present": sorted(bad),
                             "ctx_identical_under_future_permutation": bool(same),
                             "PASS": bool(same and not bad)}
    if not rep["no_future_leak"]["PASS"]:
        raise SystemExit("[preflight] ⛔ the reward ctx depends on the future")
    # 4. the reward audit AT THE SCORED GEOMETRY (5 steps @ 0.5 s, v0 supplied) --
    aud_ctx = {"dt": DT_REWARD_S, "v0": 10.0,
               "obstacles": torch.tensor([[15.0, 0.0]]),
               "lead_path": torch.tensor([[15.0, 0.0]] * 5)}
    hack = AUD.audit_reward(RewardSpec(weights=dict(HACKABLE_WEIGHTS), dt=DT_REWARD_S),
                            {"dt": DT_REWARD_S, "v0": 10.0}, n_steps=5)
    dflt = AUD.audit_reward(RewardSpec(weights=dict(DEFAULT_WEIGHTS), dt=DT_REWARD_S),
                            aud_ctx, n_steps=5)
    rep["reward_audit"] = {"hackable_at_scored_geometry": hack.verdict,
                           "hackable_reason": hack.reason, "hackable_scores": hack.scores,
                           "default_with_lead": dflt.verdict, "default_reason": dflt.reason,
                           "default_scores": dflt.scores, "default_reference": dflt.reference,
                           "PASS": hack.verdict == "FLAGGED"}
    if hack.verdict != "FLAGGED":
        raise SystemExit(f"[preflight] ⛔ the audit does not flag the hackable reward at the "
                         f"scored geometry ({hack.verdict}) — TRAIN-C4; fix before any arm")
    # 5. step-0 trust-region divergence must read ~0 (<= ANCHOR_STEP0_TOL_M2) -------
    reference = ReferencePolicy(model).to(device)
    n_frozen = reference.assert_frozen()
    with torch.no_grad():
        live = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
        live2 = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
        ref = reference(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=int(prov["decoder_steps"]))
    div = trajectory_divergence(live["anchor_traj"], ref["anchor_traj"])
    div_ll = trajectory_divergence(live["anchor_traj"], live2["anchor_traj"])   # the harness's own noise floor
    rep["anchor_step0"] = {"max_divergence_m2": float(div.abs().max()),
                           "live_vs_live_repeat_m2": float(div_ll.abs().max()),
                           "logits_live_vs_ref_maxabs": float((live["anchor_logits"] - ref["anchor_logits"]).abs().max()),
                           "wis": [int(w) for w in b["wis"]], "n_frozen": n_frozen,
                           "exact_zero": float(div.abs().max()) == 0.0,
                           "tolerance_m2": ANCHOR_STEP0_TOL_M2,
                           "PASS": float(div.abs().max()) <= ANCHOR_STEP0_TOL_M2}
    if not rep["anchor_step0"]["PASS"]:
        raise SystemExit(f"[preflight] ⛔ step-0 divergence != 0 — a mode mismatch (TRAIN-C5 family): "
                         f"live-vs-ref {rep['anchor_step0']['max_divergence_m2']:.6g} m², "
                         f"live-vs-live {rep['anchor_step0']['live_vs_live_repeat_m2']:.6g} m² on windows "
                         f"{rep['anchor_step0']['wis']}")
    # 6. one objective step: gradient flow + the component coverage on REAL windows
    sample_fn = make_sample_fn(model, reference, echo_reward=False)
    spec = RewardSpec(weights=dict(DEFAULT_WEIGHTS), dt=DT_REWARD_S)
    from tanitad.rl.posttrain import rl_objective
    traj2, logp, ctx, extras = sample_fn(b, pcfg)
    obj = rl_objective(traj2, logp, ctx, pcfg, spec, anchor_pair=extras["anchor_pair"])
    obj["loss"].backward()
    grads = {n: float(p.grad.norm()) for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    zero = [n for n in names if n not in grads or grads[n] == 0.0]
    cov = AUD.report_component_coverage(spec, traj2, ctx)
    rep["gradient_flow"] = {"n_trainable_with_grad": len([n for n in grads if grads[n] > 0]),
                            "n_trainable_total": len(names), "zero_or_none_grad": zero,
                            "grad_norm_head": {k: grads[k] for k in list(grads)[:12]},
                            "loss": float(obj["loss"]), "pg": float(obj["pg"]),
                            "anchor_penalty": float(obj.get("anchor_penalty", 0.0))}
    rep["component_coverage_on_probe_batch"] = {k: {kk: vv for kk, vv in v.items() if kk != "_note"}
                                                for k, v in cov.items()}
    model.zero_grad(set_to_none=True)
    # 7. timing at lr=0 — the weights MUST come back unchanged ------------------
    h0 = _sha256_state(model)
    import dataclasses
    tcfg = dataclasses.replace(make_cfg("ctrl0", a, prov, ""), steps=int(a.preflight_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    times_data, times_step = [], []
    pool = wis_all
    for s in range(int(a.preflight_steps)):
        td = time.time()
        wb = rng.sample(pool, int(a.batch))
        bb = build_batch(corp, lead, wb, device, with_gt=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
        traj2, logp, ctx, extras = sample_fn(bb, tcfg)
        obj = rl_objective(traj2, logp, ctx, tcfg, spec, anchor_pair=extras["anchor_pair"])
        opt.zero_grad(set_to_none=True)
        obj["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times_data.append(t1 - td)
        times_step.append(time.time() - t1)
    h1 = _sha256_state(model)
    warm = 2
    rep["timing"] = {
        "steps": int(a.preflight_steps), "batch": int(a.batch), "group": int(a.group),
        "data_s_per_step": float(np.mean(times_data[warm:])),
        "compute_s_per_step": float(np.mean(times_step[warm:])),
        "total_s_per_step": float(np.mean(np.array(times_data[warm:]) + np.array(times_step[warm:]))),
        "peak_gpu_gb": (float(torch.cuda.max_memory_allocated()) / 1e9
                        if device.startswith("cuda") else None),
        "_note": "lr=0 timing steps; the divisor is ONE step (already per-step, not accumulated)",
        "weights_unchanged": h0 == h1, "sha256_before": h0, "sha256_after": h1}
    if h0 != h1:
        raise SystemExit("[preflight] ⛔ lr=0 timing steps CHANGED the weights — refusing")
    for arm_name in ARMS:
        st = ARMS[arm_name]["steps"]
        rep["timing"][f"est_train_min_{arm_name}"] = round(rep["timing"]["total_s_per_step"] * st / 60, 1)
    rep["wallclock_s"] = round(time.time() - t_all, 1)
    rep["PASS"] = all(rep[k]["PASS"] for k in ("no_future_leak", "reward_audit", "anchor_step0")) \
        and rep["timing"]["weights_unchanged"] and not leak and not sel_surface
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    _p(f"[preflight] {'PASS' if rep['PASS'] else 'FAIL'} · "
       f"{rep['timing']['total_s_per_step']:.3f} s/step (data {rep['timing']['data_s_per_step']:.3f} + "
       f"compute {rep['timing']['compute_s_per_step']:.3f}) · peak {rep['timing']['peak_gpu_gb']} GB · "
       f"grad on {rep['gradient_flow']['n_trainable_with_grad']}/{len(names)} tensors -> {a.out}")
    return 0 if rep["PASS"] else 1


def mode_arm(a) -> int:
    if os.environ.get("LAUNCH_APPROVED") != "1":
        raise SystemExit("[rl-min] ⛔ NOT LAUNCHED: --mode arm requires LAUNCH_APPROVED=1 in the "
                         "environment (Master Mind / PI approval of the SPEC's cost).")
    device = a.device if torch.cuda.is_available() else "cpu"
    model, cfg, targs, prov = load(a, device)
    run = os.path.join(a.out_dir, a.arm)
    os.makedirs(os.path.join(run, "rl"), exist_ok=True)
    os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
    pcfg = make_cfg(a.arm, a, prov, os.path.join(run, "rl"))
    pcfg.validate()
    _p(f"[arm {a.arm}] RESOLVED reward_weights = {json.dumps(pcfg.reward_weights, sort_keys=True)} "
       f"w_anchor={pcfg.w_anchor} lr={pcfg.lr} steps={pcfg.steps}")
    corp = open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, _li = load_lead_block(a.lead_block)
    gate3 = _assert_disjoint(corp.clip_ids, a.eval_episodes, a.eval_labels)
    ecorp = open_corpus(a.eval_episodes, a.eval_labels, cfg, prov, a.lru)
    elead, _, _ = load_lead_block(a.eval_lead_block)
    e_all = scoreable_windows(ecorp, elead)
    e_rng = random.Random(1234)
    e_wis = sorted(e_rng.sample(e_all, min(len(e_all), int(a.readout_windows))))
    h_base = _sha256_state(model)
    before = readout(model, ecorp, elead, e_wis, device, decoder_steps=int(prov["decoder_steps"]))
    json.dump(before, open(os.path.join(run, "readout_before.json"), "w"), indent=1)
    _p(f"[arm {a.arm}] BEFORE {json.dumps(before['aggregate'])} on {before['n_windows']} eval windows")
    reference = ReferencePolicy(model).to(device) if pcfg.w_anchor > 0 else None
    sample_fn = make_sample_fn(model, reference, echo_reward=(a.arm == "reg_echo"))
    pool = scoreable_windows(corp, lead)
    rng = random.Random(int(a.seed))

    def batches():
        for _ in range(pcfg.steps):
            yield build_batch(corp, lead, rng.sample(pool, pcfg.batch), device,
                              with_gt=(a.arm == "reg_echo"))
    t0 = time.time()
    summary = run_posttrain(model, sample_fn, pcfg, batches=batches(), device=device)
    summary["train_wallclock_s"] = round(time.time() - t0, 1)
    h_after = _sha256_state(model)
    summary["weights_changed"] = h_base != h_after
    if a.arm == "ctrl0" and summary["weights_changed"]:
        raise SystemExit("[arm ctrl0] ⛔ lr=0 changed the weights — the control did not reproduce")
    ck_path = os.path.join(run, "ckpt", "ckpt_after.pt")
    torch.save({"model": model.state_dict(), "step": int(prov["step"]),
                "rl": {"arm": a.arm, "cfg": pcfg.to_dict(), "steps": pcfg.steps,
                       "base_ckpt": a.ckpt, "base_sha256_state": h_base,
                       "after_sha256_state": h_after}}, ck_path)
    import shutil
    shutil.copyfile(a.config or os.path.join(os.path.dirname(a.ckpt), "config.json"),
                    os.path.join(run, "ckpt", "config.json"))
    after = readout(model, ecorp, elead, e_wis, device, decoder_steps=int(prov["decoder_steps"]))
    json.dump(after, open(os.path.join(run, "readout_after.json"), "w"), indent=1)
    deltas = {k: paired_delta(before, after, k) for k in ("R1", "R2", "R3", "R_FAN", "R_REACH", "R_ORACLE")}
    sel_same = float(np.mean([x["sel_idx"] == y["sel_idx"] for x, y in
                              zip(before["per_window"], after["per_window"])]))
    summary.update({"arm": a.arm, "gate3": gate3, "readout_deltas_paired": deltas,
                    "sel_idx_agreement_with_base": sel_same, "ckpt_after": ck_path,
                    "_tier": "T0 training-side; the capability read is the taniteval harness"})
    json.dump({k: v for k, v in summary.items() if k != "history"},
              open(os.path.join(run, "arm_summary.json"), "w"), indent=1, default=str)
    _p(f"[arm {a.arm}] AFTER {json.dumps(after['aggregate'])} · sel agreement {sel_same:.3f} · "
       f"weights_changed={summary['weights_changed']} -> {run}")
    return 0


def _human_future(corp, wi):
    """The logged ego 2 s future in the t0 frame, [1, 1, 1, 5, 2] (fan-shaped), and v0.
    Poses only — no frame decode — via the SAME waypoint_targets the reg_echo arm uses."""
    e_i, t = corp.ds.index[wi]
    poses = corp.ds.episodes[e_i].poses
    T = int(poses.shape[0])
    t0 = t + corp.W - 1
    idx = torch.arange(t0 + 1, t0 + 1 + 60).clamp(max=T - 1)
    fut = poses[idx].float()[None]
    pose_last = poses[t0].float()[None]
    gt8 = refb_labels.waypoint_targets(pose_last, fut, list(ARM_HORIZONS))
    return with_origin(gt8[:, :N_REWARD_SLOTS]).reshape(1, 1, 1, 5, 2), float(pose_last[0, 3])


def mode_humanflag(a) -> int:
    """⛔ THE H-RL-THRESH-1 CHECK, BEFORE ANY ARM: does the reward's scene context flag
    the HUMAN DRIVER'S OWN FUTURE? Scores the logged 2 s future, the hold-v0 straight
    path (the `ha0` floor) and a frozen path under BOTH lead models on every RL-fit
    window with a lead. PASS iff, under the ACTIVE mode, collision and the TTC veto
    each fire on the human on <= --human-flag-max of windows (THRESHOLD_CALIBRATION's
    design band: ~0.05-0.15 doing its job, > 0.25 mis-calibrated). Zero GPU."""
    global LEAD_MODE
    device = "cpu"
    _model, cfg, _targs, prov = load(a, device)
    corp = open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, linfo = load_lead_block(a.lead_block)
    if lead is None or lead.ts_rel is None:
        raise SystemExit("[humanflag] ⛔ a lead block with ts_rel_s is required")
    spec = RewardSpec(weights=dict(DEFAULT_WEIGHTS), dt=DT_REWARD_S)
    wis = scoreable_windows(corp, lead)
    rows = []
    active = LEAD_MODE
    for wi in wis:
        has, xy, _ln = lead_row(corp, lead, wi)
        if not has:
            continue
        e_i, t = corp.ds.index[wi]
        row = lead.idx.get((corp.clip_ids[e_i], int(t + corp.W - 1 + corp.raw_off)))
        human, v0 = _human_future(corp, wi)
        xs = torch.tensor([v0 * s for s in GRID_S], dtype=torch.float32)
        hold = torch.stack([xs, torch.zeros_like(xs)], dim=-1).reshape(1, 1, 1, 5, 2)
        frozen = torch.zeros(1, 1, 1, 5, 2)
        batch = {"v0": torch.tensor([v0]), "lead_xy": torch.tensor([xy], dtype=torch.float32),
                 "lead_track": lead_track(corp, lead, wi)[None]}
        rec = {"wi": int(wi), "clip": corp.clip_ids[e_i], "eid": int(e_i), "v0": v0,
               "gt_time_gap_min_s": (float(lead.gt_time_gap[row])
                                     if lead.gt_time_gap is not None else float("nan"))}
        for mode in ("static", "track"):
            LEAD_MODE = mode
            ctx = reward_ctx(batch, S5=5)
            for name, traj in (("human", human), ("hold_v0", hold), ("frozen", frozen)):
                parts = spec.per_component(traj, ctx)
                r = {k: float(v.reshape(-1)[0]) for k, v in parts.items()}
                r["composed"] = float(spec(traj, ctx).reshape(-1)[0])
                r["ttc_veto"] = bool(RW.ttc_violation(traj, {**ctx, "ttc_min_s": 1.5}).reshape(-1)[0])
                r["collision_fired"] = r["collision"] < 0
                rec[f"{name}_{mode}"] = r
        rows.append(rec)
    LEAD_MODE = active
    n = len(rows)

    def rate(sub, key, pred=bool):
        v = [pred(r[sub][key]) for r in rows]
        return {"rate": float(np.mean(v)) if v else float("nan"), "n": len(v)}

    def mean(sub, key):
        v = [r[sub][key] for r in rows]
        return float(np.mean(v)) if v else float("nan")

    summary = {}
    for mode in ("static", "track"):
        h = f"human_{mode}"
        summary[mode] = {
            "collision_fires_on_human": rate(h, "collision_fired"),
            "ttc_veto_fires_on_human": rate(h, "ttc_veto"),
            "headway_below_0.5_on_human": rate(h, "headway", lambda v: v < 0.5),
            "headway_mean_on_human": mean(h, "headway"),
            "composed_mean": {"human": mean(h, "composed"),
                              "hold_v0": mean(f"hold_v0_{mode}", "composed"),
                              "frozen": mean(f"frozen_{mode}", "composed")},
            "hold_v0_scores_at_least_human": {
                "rate": float(np.mean([r[f"hold_v0_{mode}"]["composed"] >= r[h]["composed"]
                                       for r in rows])) if rows else float("nan"), "n": n},
            "frozen_scores_at_least_human": {
                "rate": float(np.mean([r[f"frozen_{mode}"]["composed"] >= r[h]["composed"]
                                       for r in rows])) if rows else float("nan"), "n": n}}
    by_tg = {}
    for lo, hi in ((0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)):
        sel = [r for r in rows if lo <= r["gt_time_gap_min_s"] < hi]
        if sel:
            by_tg[f"[{lo},{hi if hi < 1e9 else 'inf'})"] = {
                "n": len(sel),
                "static_collision": float(np.mean([r["human_static"]["collision_fired"] for r in sel])),
                "static_ttc_veto": float(np.mean([r["human_static"]["ttc_veto"] for r in sel])),
                "track_collision": float(np.mean([r["human_track"]["collision_fired"] for r in sel])),
                "track_ttc_veto": float(np.mean([r["human_track"]["ttc_veto"] for r in sel]))}
    act = summary[active]
    worst = max(act["collision_fires_on_human"]["rate"], act["ttc_veto_fires_on_human"]["rate"])
    out = {"_what": "human-future flag rates under the RL reward's scene context, static vs track lead",
           "_evidence_class": "MEASURED (ours)",
           "_tier": "instrument probe, T0, NON-PARITY RL-fit clips",
           "active_lead_mode": active, "human_flag_max": float(a.human_flag_max),
           "n_scoreable_windows": len(wis), "n_lead_windows_scored": n,
           "n_episodes": len({r["eid"] for r in rows}), "reward_weights": dict(DEFAULT_WEIGHTS),
           "dt_s": DT_REWARD_S, "grid_s": list(GRID_S),
           "summary": summary, "by_human_time_gap": by_tg,
           "PASS": bool(n > 0 and worst <= float(a.human_flag_max)),
           "per_window": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    _p(f"[humanflag] {'PASS' if out['PASS'] else 'FAIL'} under lead_mode={active}: "
       f"collision fires on the human {act['collision_fires_on_human']['rate']:.3f}, "
       f"TTC veto {act['ttc_veto_fires_on_human']['rate']:.3f} (n={n}, max {a.human_flag_max}); "
       f"static: {summary['static']['collision_fires_on_human']['rate']:.3f} / "
       f"{summary['static']['ttc_veto_fires_on_human']['rate']:.3f} -> {a.out}")
    return 0 if out["PASS"] else 1


def _fam_rows(rec: dict) -> dict:
    """paired_openloop.py's record -> {metric_key: {family, delta, lo, hi, separated,
    lower_is_better}}. The record's shape is rec['families'][FAMILY]['metrics'][KEY]
    = {delta, lo, hi, separated, verdict, ...}; direction is (B-floor)-(A-floor), so
    NEGATIVE favours B (the arm) on lower-is-better metrics."""
    lib = {mk: bool(l) for mk, _f, l, _u in PAIRED.TRAJ_METRICS}
    rows = {}
    for fam, blk in (rec.get("families") or {}).items():
        for mk, r in ((blk or {}).get("metrics") or {}).items():
            if isinstance(r, dict) and r.get("delta") is not None:
                rows[mk] = {"family": fam, "delta": float(r["delta"]), "lo": r.get("lo"),
                            "hi": r.get("hi"), "separated": bool(r.get("separated")),
                            "lower_is_better": lib.get(mk, True)}
    return rows


def mode_verdict(a) -> int:
    """SPEC §6, applied mechanically. Inputs: the paired records (rl/reg_echo/ctrl0 vs base)
    and the arm summaries. Prints the exit and writes verdict.json."""
    def J(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    out = {"_rule": "the exit is chosen by code from SPEC §6; no branch is read into a result"}
    arms = {n: J(os.path.join(a.out_dir, n, "arm_summary.json")) for n in ("rl", "reg_echo", "ctrl0")
            if os.path.exists(os.path.join(a.out_dir, n, "arm_summary.json"))}
    pairs = {n: J(os.path.join(a.out_dir, f"paired_{n}_vs_base.json"))
             for n in ("rl", "reg_echo", "ctrl0")
             if os.path.exists(os.path.join(a.out_dir, f"paired_{n}_vs_base.json"))}
    void = []
    c0 = arms.get("ctrl0")
    if c0 is None or c0.get("weights_changed"):
        void.append("V1: ctrl0 absent or its weights CHANGED under lr=0")
    elif any(abs(v["delta"]) > 0 for v in c0["readout_deltas_paired"].values()):
        void.append("V1: ctrl0 readout delta != 0 on a bitwise-identical model — harness non-determinism")
    reg = arms.get("reg_echo")
    if reg is None:
        void.append("V2: the deliberate-regression arm did not run")
    else:
        fan = reg["readout_deltas_paired"]["R_FAN"]
        base_fan = float(J(os.path.join(a.out_dir, "reg_echo", "readout_before.json"))["aggregate"]["R_FAN"])
        collapsed = fan["sep"] and fan["delta"] < 0 and abs(fan["delta"]) >= 0.30 * base_fan
        out["reg_echo_fan"] = {"delta": fan, "base_R_FAN": base_fan, "G_FAN_fired": collapsed}
        if not collapsed:
            void.append("V2: G-FAN did NOT fire on reg_echo — the readout cannot see fan collapse; "
                        "no PASS is admissible")
    # the ECHO SIGNATURE on the regression arm, recorded: oracle-in-fan IMPROVES while
    # the spread collapses — an ADE-only eval would call that arm a win.
    if reg is not None:
        out["reg_echo_fan"]["R_ORACLE_delta"] = reg["readout_deltas_paired"].get("R_ORACLE")
        out["reg_echo_fan"]["R3_delta"] = reg["readout_deltas_paired"].get("R3")
    pr = pairs.get("rl") or {}
    if pr.get("void"):
        void.append(f"V3: the paired record for `rl` is VOID: {pr.get('void_reasons')}")
    out["void"] = void
    if void:
        out["exit"] = "VOID"
    else:
        rl = arms["rl"]
        fan = rl["readout_deltas_paired"]["R_FAN"]
        base_fan = float(J(os.path.join(a.out_dir, "rl", "readout_before.json"))["aggregate"]["R_FAN"])
        rl_fan_ok = not (fan["sep"] and fan["delta"] < 0 and abs(fan["delta"]) >= 0.15 * base_fan)
        fam = _fam_rows(pr)                              # paired_openloop.py's real schema
        ade = fam.get("ade_m", {})
        ade_guard_ok = not (ade.get("lo") is not None and float(ade["lo"]) > 0.02)
        improved = [k for k, v in fam.items() if v["family"] != "ADE" and v["separated"]
                    and ((v["delta"] < 0) if v["lower_is_better"] else (v["delta"] > 0))]
        worsened = [k for k, v in fam.items() if v["family"] != "ADE" and v["separated"]
                    and ((v["delta"] > 0) if v["lower_is_better"] else (v["delta"] < 0))]
        osel = rl["readout_deltas_paired"].get("R_ORACLE", {})   # oracle-in-fan, T0 readout, paired
        fan_quality_worse = bool(osel.get("sep") and osel.get("delta", 0) > 0)
        fan_quality_better = bool(osel.get("sep") and osel.get("delta", 0) < 0)
        ade_better = bool(ade.get("separated") and ade.get("delta", 0) < 0)
        out["rl"] = {"fan_delta": fan, "G_FAN_ok": rl_fan_ok, "ade_guard_ok": ade_guard_ok,
                     "ade_m_paired": ade, "families_improved_separated": improved,
                     "families_worsened_separated": worsened,
                     "oracle_in_fan_delta": osel, "fan_quality_worse": fan_quality_worse,
                     "n_family_metrics_read": len(fam)}
        if not fam:
            out["exit"] = ("VOID — the paired record carries no family metrics "
                           "(schema mismatch); nothing is decidable")
        elif not rl_fan_ok:
            out["exit"] = "FAIL-COLLAPSE — the RL arm collapsed its own fan"
        elif not ade_guard_ok:
            out["exit"] = "FAIL-GUARD — ADE degraded beyond +0.02 m with separation"
        elif fan_quality_worse:
            out["exit"] = "FAIL-FAN — oracle-in-fan (fan quality) got worse with separation"
        elif improved and not worsened:
            out["exit"] = f"PASS — RL improved {improved} with paired separation under the guards"
        elif improved and worsened:
            out["exit"] = f"SPLIT — improved {improved} but worsened {worsened}; no verdict, report per family"
        elif ade_better and not fan_quality_better:
            out["exit"] = "REJECT-SELECTOR — the selected path improved but the fan did not (DDv2 defect)"
        else:
            out["exit"] = "NULL — no family moved with separation; RL stage inert on this base"
    with open(os.path.join(a.out_dir, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    _p(f"[verdict] {out['exit']}")
    for v in void:
        _p(f"   · {v}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=("preflight", "fitlist", "arm", "verdict", "humanflag"))
    ap.add_argument("--lead-mode", choices=("track", "static"), default="track",
                    help="how the lead enters the reward ctx (see LEAD_MODE)")
    ap.add_argument("--human-flag-max", type=float, default=0.15,
                    help="humanflag PASS ceiling on the human-flag rate under the active mode")
    ap.add_argument("--ckpt"); ap.add_argument("--config", default=None)
    ap.add_argument("--expect-step", type=int, default=EXPECT_BASE_STEP)
    ap.add_argument("--episodes", help="RL-fit v2ep dir (train-split clips only)")
    ap.add_argument("--labels", help="v7.2 TRAIN labels blob (nav tokens for the fit clips)")
    ap.add_argument("--lead-block", default=None, help="lead block for the FIT clips")
    ap.add_argument("--eval-episodes", default=None); ap.add_argument("--eval-labels", default=None)
    ap.add_argument("--eval-lead-block", default=None)
    ap.add_argument("--train-labels", default=None); ap.add_argument("--n-fit", type=int, default=120)
    ap.add_argument("--arm", choices=tuple(ARMS), default="rl")
    ap.add_argument("--out", default=None); ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lru", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="override the arm's committed step count")
    ap.add_argument("--preflight-steps", type=int, default=20)
    ap.add_argument("--readout-windows", type=int, default=120)
    a = ap.parse_args(argv)
    global LEAD_MODE
    LEAD_MODE = a.lead_mode
    torch.manual_seed(int(a.seed))
    return {"preflight": mode_preflight, "fitlist": mode_fitlist,
            "arm": mode_arm, "verdict": mode_verdict,
            "humanflag": mode_humanflag}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
