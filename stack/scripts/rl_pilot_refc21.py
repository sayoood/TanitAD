#!/usr/bin/env python3
"""P-RC21 — RL post-training of the REF-C v2.1 base planner (the pilot).

Pre-registered in ``…/2026-08-29-rl-posttrain-library/PREREG_P_RC21.md`` —
both outcomes committed BEFORE any step ran. PI directive 2026-08-29: apply the
RL training to the existing REF-C until refcv3's cold start lands.

WIRING (nothing copied — refc imported, library imported)
---------------------------------------------------------
model      ``refc.RefCModel(refc.refc_config())`` + strict ``ck["model"]`` load,
           param count asserted against the registry's 104,191,577.
trainable  ``decoder.*`` EXCEPT ``decoder.conf_head`` (the v2.1 selector surface
           lives INSIDE the decoder — training it would be TRAIN-C1's inversion
           again). Enforced by ``exclude_prefixes`` + the forbidden tripwire.
sampler    the library's Gaussian-on-offset surrogate at ``decoder_steps=2``
           (the deploy path). STATED LIMIT: not the true diffusion density.
reward     ``DEFAULT_WEIGHTS`` on real scene context from the pilot join
           (A0 PASS on this corpus: ``raw/a0_pilot.json``), dt = 0.5 s.
readout    R1 fan reward · R2 fan collision rate · R3 conf-argmax ADE — the
           selector output is used for READOUT ONLY, never in the reward.

⚠️ BATCH-SHAPED CONTEXT. The reward broadcasts ctx against ``traj [B,N,G,S,2]``;
per-window facts must therefore be ``[B,1,1,…]``. No-lead windows carry a
FAR-LEAD SENTINEL (x = 1e6): constant across the window's candidates, so it
cancels in BOTH group-relative advantages — it only distorts the absolute
logged reward, which is why R1 is reported per-component too.

Tier: T0 training-side. NON-PARITY corpus. No driving claim lives here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tanitad.refs import refc  # noqa: E402
from tanitad.rl import (PostTrainConfig, RewardSpec, rewards as RW)  # noqa: E402
from tanitad.rl.posttrain import run_posttrain, select_trainable  # noqa: E402
from tanitad.rl.refcv3_adapter import make_refcv3_sample_fn  # noqa: E402

HORIZONS = (5, 10, 15, 20)          # frame offsets @10 Hz -> 0.5/1.0/1.5/2.0 s
DT_TRAJ = 0.5                        # waypoint spacing in seconds
WINDOW = 8                           # stacked frames the encoder consumes
EXPECT_PARAMS = 104_191_577          # registry, measured at instantiation
FAR_LEAD_X = 1.0e6                   # no-lead sentinel (constant per window)
LEAD_LAT_M, LEAD_MAX_GAP_M = 2.0, 80.0


def ego_frame_np(px, py, x0, y0, yaw0):
    dx, dy = px - x0, py - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    return dx * c + dy * s, -dx * s + dy * c


def load_agents(path):
    per = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                per.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r["agents"]
    return per


class WindowSource:
    """Windows over joined episodes with a tiny episode LRU (frames are 117 MB/ep)."""

    def __init__(self, epdir, agents_path, seed=0, lru=60, k_max=32):
        # lru=60 holds the WHOLE 54-episode pilot corpus (~6.3 GB uint8) in RAM.
        # The measured risk was data-boundness: 117 MB torch.load per miss with
        # 2 random episodes/step would have dominated the 0.2-0.35 s compute
        # floor. One warm pass through the corpus, then zero disk reads.
        self.epdir, self.agents = epdir, load_agents(agents_path)
        self.stems = sorted(self.agents)
        self.rng = np.random.default_rng(seed)
        self.lru, self.cache, self.k_max = lru, {}, k_max
        if not self.stems:
            raise SystemExit("[pilot] ⛔ zero joined episodes — nothing to train on")

    def _ep(self, stem):
        if stem not in self.cache:
            if len(self.cache) >= self.lru:
                self.cache.pop(next(iter(self.cache)))
            self.cache[stem] = torch.load(os.path.join(self.epdir, stem + ".pt"),
                                          map_location="cpu", weights_only=False)
        return self.cache[stem]

    def window(self, stem=None, t0=None):
        stem = stem or self.stems[int(self.rng.integers(len(self.stems)))]
        d = self._ep(stem)
        T = int(d["poses"].shape[0])
        lo, hi = WINDOW - 1, T - HORIZONS[-1] - 1
        if hi <= lo:
            return None
        t0 = int(t0 if t0 is not None else self.rng.integers(lo, hi))
        poses = d["poses"].numpy()
        frames = d["frames_u8"][t0 - WINDOW + 1: t0 + 1].float() / 255.0  # [W,9,H,W]

        x0, y0, yaw0 = poses[t0, 0], poses[t0, 1], poses[t0, 2]
        fut = poses[[t0 + h for h in HORIZONS]]
        gx, gy = ego_frame_np(fut[:, 0], fut[:, 1], x0, y0, yaw0)
        gt = torch.tensor(np.stack([gx, gy], -1), dtype=torch.float32)  # [S,2]

        obs = [(float(a["cx"]), float(a["cy"]))
               for a in self.agents[stem].get(t0, [])
               if 0.0 < a["cx"] < LEAD_MAX_GAP_M][: self.k_max]
        inlane = [(x, y) for x, y in obs if abs(y) <= LEAD_LAT_M]
        lead_xy = min(inlane, key=lambda p: p[0]) if inlane else (FAR_LEAD_X, 0.0)

        return {"stem": stem, "t0": t0, "frames": frames,
                "v0": float(poses[t0, 3]), "gt": gt,
                "obs": obs, "lead_xy": lead_xy, "has_lead": bool(inlane)}


def collate(wins, device):
    """Windows -> a model batch + BATCH-SHAPED reward context ([B,1,1,…])."""
    b = len(wins)
    k = max(1, max(len(w["obs"]) for w in wins))
    obs = torch.full((b, k, 2), FAR_LEAD_X)
    for i, w in enumerate(wins):
        for j, (x, y) in enumerate(w["obs"]):
            obs[i, j, 0], obs[i, j, 1] = x, y
    lead = torch.stack([
        torch.tensor(w["lead_xy"], dtype=torch.float32).expand(len(HORIZONS), 2)
        for w in wins])                                          # [B,S,2] (static)
    return {
        "frames": torch.stack([w["frames"] for w in wins]).to(device),
        "v0": torch.tensor([w["v0"] for w in wins], device=device),
        "gt_traj": torch.stack([w["gt"] for w in wins]).to(device),   # [B,S,2]
        "obstacles": obs.unsqueeze(1).unsqueeze(1).to(device),        # [B,1,1,K,2]
        "lead_path": lead.unsqueeze(1).unsqueeze(1).to(device),       # [B,1,1,S,2]
        "dt": DT_TRAJ,
        "has_lead": [w["has_lead"] for w in wins],
    }


def build_ctx(batch, out=None):
    return {k: batch[k] for k in ("gt_traj", "obstacles", "lead_path", "v0", "dt")}


def load_model(ckpt_path, device):
    m = refc.RefCModel(refc.refc_config())
    total = sum(p.numel() for p in m.parameters())
    if total != EXPECT_PARAMS:
        raise SystemExit(f"[pilot] ⛔ built {total:,} params, registry says "
                         f"{EXPECT_PARAMS:,} — wrong config, refusing")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=True)
    step = ck.get("step", "?")
    print(f"[pilot] cold start loaded: {total:,} params @ step {step}", flush=True)
    return m.to(device)


@torch.no_grad()
def readout(model, src: WindowSource, spec: RewardSpec, cfg, device, n=120):
    """R1/R2/R3 on a FIXED, seed-derived window set (identical before/after)."""
    rng = np.random.default_rng(1234)
    picks = []
    for stem in src.stems:
        T = int(src._ep(stem)["poses"].shape[0])
        lo, hi = WINDOW - 1, T - HORIZONS[-1] - 1
        for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(8, hi - lo),
                                    replace=False)):
            picks.append((stem, int(t0)))
    picks = picks[:n]

    per_ep: dict[str, dict] = {}
    comp_sums: dict[str, float] = {}
    n_win = 0
    for stem, t0 in picks:
        w = src.window(stem, t0)
        if w is None:
            continue
        batch = collate([w], device)
        out = model(batch["frames"], None, batch["v0"],
                    steps=int(cfg.decoder_steps))
        fan = out["anchor_traj"]                                  # [1,N,S,2]
        ctx = build_ctx(batch)
        parts = spec.per_component(fan, ctx)
        coll = RW.COMPONENTS["collision"](fan, ctx)               # [1,N]
        sel = int(out["sel_score"].argmax(dim=1))                 # readout only
        ade = float((fan[0, sel] - batch["gt_traj"][0]).norm(dim=-1).mean())

        e = per_ep.setdefault(stem, {"r": [], "cr": [], "ade": []})
        e["r"].append(float(spec(fan, ctx).mean()))
        e["cr"].append(float((coll < 0).float().mean()))
        e["ade"].append(ade)
        for k, v in parts.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.mean())
        n_win += 1

    ep_means = {k: {m: float(np.mean(v[m])) for m in v} for k, v in per_ep.items()}
    arr = lambda m: np.array([e[m] for e in ep_means.values()])
    boot = []
    ids = list(ep_means)
    brng = np.random.default_rng(7)
    for _ in range(2000):
        pick = brng.choice(len(ids), size=len(ids), replace=True)
        boot.append([arr("r")[pick].mean(), arr("cr")[pick].mean(),
                     arr("ade")[pick].mean()])
    lo_, hi_ = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "n_windows": n_win, "n_episodes": len(ep_means),
        "R1_fan_reward": {"mean": float(arr("r").mean()),
                          "ci95": [float(lo_[0]), float(hi_[0])]},
        "R2_fan_collision_rate": {"mean": float(arr("cr").mean()),
                                  "ci95": [float(lo_[1]), float(hi_[1])]},
        "R3_sel_ade_m": {"mean": float(arr("ade").mean()),
                         "ci95": [float(lo_[2]), float(hi_[2])]},
        "R4_component_means": {k: v / max(n_win, 1) for k, v in comp_sums.items()},
        "_estimator": "episode-cluster bootstrap, 2000 reps — ⚠️ few clusters, "
                      "claims only when CIs separate",
        "_tier": "T0 training-side; NON-PARITY corpus; readout-only selector use",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-epdir", required=True)
    ap.add_argument("--train-agents", required=True)
    ap.add_argument("--val-epdir", required=True)
    ap.add_argument("--val-agents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--reward", choices=("default", "hackable"), default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lru", type=int, default=60,
                    help="episode cache size; 60 = whole pilot corpus in RAM")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = (dict(RW.HACKABLE_WEIGHTS) if a.reward == "hackable"
               else dict(RW.DEFAULT_WEIGHTS))
    cfg = PostTrainConfig(
        method="grpo", group_size=4, steps=a.steps, batch=a.batch, lr=1e-5,
        seed=a.seed, dt=DT_TRAJ, decoder_steps=2, reward_weights=weights,
        trainable_prefixes=("decoder",),
        exclude_prefixes=("decoder.conf_head",),
        forbidden_prefixes=("decoder.conf_head", "scorer"),
        w_imitation=0.0,   # ⚠️ stated: gt_similarity is OUT of the reward AND no
                           # IL loss is wired for v2.1 in this pilot — the
                           # imitation anchor is the FROZEN 96 % of the model.
                           # R3's +10 % guard is the drift alarm instead.
        out_dir=a.out, run_name=f"p-rc21-{a.reward}")
    cfg.validate()

    model = load_model(a.ckpt, device)
    spec = RewardSpec(weights=weights, dt=DT_TRAJ)
    train_src = WindowSource(a.train_epdir, a.train_agents, seed=a.seed, lru=a.lru)
    val_src = WindowSource(a.val_epdir, a.val_agents, seed=a.seed)
    print(f"[pilot] train eps {len(train_src.stems)} · val eps "
          f"{len(val_src.stems)} · device {device} · reward {a.reward}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    before = readout(model, val_src, RewardSpec(dt=DT_TRAJ), cfg, device)
    json.dump(before, open(os.path.join(a.out, "readout_before.json"), "w"),
              indent=1)
    print(f"[pilot] BEFORE  R1 {before['R1_fan_reward']['mean']:+.4f}  "
          f"R2 {before['R2_fan_collision_rate']['mean']:.3%}  "
          f"R3 {before['R3_sel_ade_m']['mean']:.3f} m", flush=True)

    def batches():
        step = 0
        while step < a.steps:
            wins = []
            while len(wins) < a.batch:
                w = train_src.window()
                if w is not None:
                    wins.append(w)
            yield collate(wins, device)
            step += 1

    sample_fn = make_refcv3_sample_fn(model, cfg, build_ctx=build_ctx)
    summary = run_posttrain(model, sample_fn, cfg, batches=batches())

    after = readout(model, val_src, RewardSpec(dt=DT_TRAJ), cfg, device)
    json.dump(after, open(os.path.join(a.out, "readout_after.json"), "w"),
              indent=1)
    print(f"[pilot] AFTER   R1 {after['R1_fan_reward']['mean']:+.4f}  "
          f"R2 {after['R2_fan_collision_rate']['mean']:.3%}  "
          f"R3 {after['R3_sel_ade_m']['mean']:.3f} m", flush=True)

    delta = {
        "R1": after["R1_fan_reward"]["mean"] - before["R1_fan_reward"]["mean"],
        "R2": (after["R2_fan_collision_rate"]["mean"]
               - before["R2_fan_collision_rate"]["mean"]),
        "R3_rel": (after["R3_sel_ade_m"]["mean"]
                   / max(before["R3_sel_ade_m"]["mean"], 1e-9) - 1.0),
    }
    summary["pilot_delta"] = delta
    json.dump({k: v for k, v in summary.items() if k != "history"},
              open(os.path.join(a.out, "pilot_summary.json"), "w"), indent=1)
    print(f"[pilot] DELTA   R1 {delta['R1']:+.4f}  R2 {delta['R2']:+.3%}  "
          f"R3 {delta['R3_rel']:+.2%}  -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
