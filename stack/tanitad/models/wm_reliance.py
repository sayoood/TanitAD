"""⛔ DOES THE DECODER ACTUALLY USE THE WORLD MODEL, OR IS IT A DRIVING-DYNAMICS PREDICTOR?

**Sayed, 2026-08-06:** *"we need to be very careful not to train a driving dynamic
predictor not using the wm. so the decoder must rely on the representation and prediction
of the wm, how can we optimize it, assure it and validate it?"*

⭐ THE RISK IS REAL AND THE HEAD'S OWN DESIGN CREATED IT. Two of the four measured
parameterisation choices open paths that bypass the world model entirely:

* **speed as an input** — on a mostly-straight, mostly-constant-speed corpus, ``v`` alone
  reconstructs most of the trajectory;
* **delta prediction with carried ``a_prev``/``yr_prev``** — MEASURED lag-1 autocorrelation
  **+0.977**, so ``a_j ~= a_{j-1}`` is an excellent predictor using **no latents at all**.

A head that took both shortcuts would score well on ADE and carry **none** of the world
model's content. It would be a driving-dynamics predictor wearing a decoder's name, and
every hierarchy claim resting on it would be void.

⛔ **"REAL BEATS NONE" IS NOT THE TEST.** ``hierarchy.py:846`` already names this confound
for the tactical seam: *"a frozen-encoder operative can co-adapt to always having the term
(real > none) WITHOUT using its per-window content (real == mean)"*. The strict test is
**real vs MEAN** — does the *per-window content* help — and this module reuses that logic
rather than inventing a second one.

## The arms

| arm | what it does | what it removes |
|---|---|---|
| ``real`` | the true latents | — |
| ``mean`` | every latent replaced by the BATCH MEAN | per-window content, distribution kept |
| ``shuffled`` | latents permuted across the batch | the pairing, marginal kept EXACTLY |
| ``frozen`` | step 0's latent pair repeated for all K steps | the WM's TEMPORAL content only |
| ``cv`` | zero controls from the true ``v0`` | everything — the analytic floor |

⭐ **``cv`` is the floor that makes the whole thing interpretable.** The head's output is the
residual over constant velocity by construction (zero output = hold v0, go straight), so
``ADE(cv) - ADE(real)`` is the *entire* value the head adds, and
``ADE(cv) - ADE(mean)`` is the part of it reachable **without any per-window latent
content**. Their ratio is the number Sayed is asking for.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from tanitad.models.metric_dynamics import accumulate_se2


def _net_yaw_err(pred: Tensor, tgt: Tensor, min_ds: float = 0.05) -> float:
    """Sampling-independent lateral error — the quantity that regressed under re-timing
    and the one 'improve the heading' is actually about."""
    def ny(P):
        p = torch.cat([torch.zeros_like(P[:, :1]), P], 1)
        d = p[:, 1:] - p[:, :-1]
        ds = torch.linalg.norm(d, dim=-1)
        h = torch.atan2(d[..., 1], d[..., 0])
        dh = (h[:, 1:] - h[:, :-1] + math.pi) % (2 * math.pi) - math.pi
        ok = (ds[:, 1:] > min_ds) & (ds[:, :-1] > min_ds)
        return (dh * ok).sum(1)
    return float((ny(pred) - ny(tgt)).abs().mean())


def _cv(v0: Tensor, k: int, dt: float) -> Tensor:
    """Constant-velocity waypoints from the true v0 — the analytic floor, and exactly
    what a zero-initialised unicycle head emits."""
    step = torch.stack([v0 * dt, torch.zeros_like(v0), torch.zeros_like(v0)], -1)
    return accumulate_se2(step.unsqueeze(1).expand(-1, k, -1).contiguous())


@torch.no_grad()
def wm_reliance(rollout_fn, states: Tensor, actions: Tensor,
                future_actions: Tensor | None, gt: Tensor, v0: Tensor,
                k: int = 20, dt: float = 0.1, seed: int = 0) -> dict:
    """Run the five arms and report how much of the decoder's value comes from the WM.

    ``rollout_fn(states, actions, future_actions, v0) -> waypoints [B, k, 2]`` — a closure
    over the frozen predictor and the trained readout. ``gt`` [B, k, 2].

    ⛔ ONLY THE LATENTS ARE ABLATED. ``v0``, the actions and the integrator are untouched
    in every arm, so the contrast isolates the world model's contribution and not the
    ego-state pathway. Ablating v0 as well would conflate "does it use the WM" with "does
    it use the ego speed", which are different questions with different answers.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    b = states.shape[0]
    perm = torch.randperm(b, generator=g).to(states.device)

    variants = {
        "real": states,
        # batch mean, broadcast: same distribution, no per-window content
        "mean": states.mean(dim=0, keepdim=True).expand_as(states).contiguous(),
        # exact marginal, destroyed pairing
        "shuffled": states[perm].contiguous(),
        # the window's content kept, its TEMPORAL evolution removed
        "frozen": states[:, :1].expand_as(states).contiguous(),
    }
    out = {}
    for name, st in variants.items():
        wp = rollout_fn(st, actions, future_actions, v0)
        out[name] = {
            "ade_m": round(float(torch.linalg.norm(wp - gt, dim=-1).mean()), 5),
            "net_yaw_err_rad": round(_net_yaw_err(wp, gt), 5),
        }
    wp_cv = _cv(v0.to(gt.dtype), k, dt)
    out["cv"] = {
        "ade_m": round(float(torch.linalg.norm(wp_cv - gt, dim=-1).mean()), 5),
        "net_yaw_err_rad": round(_net_yaw_err(wp_cv, gt), 5),
        "_what": ("constant velocity from the TRUE v0 — the analytic floor, and exactly "
                  "what a zero-initialised unicycle head emits before any training"),
    }

    real, mean, cv = out["real"]["ade_m"], out["mean"]["ade_m"], out["cv"]["ade_m"]
    total_gain = cv - real                       # everything the head adds over CV
    shortcut_gain = cv - mean                    # reachable with NO per-window latent
    frac = (shortcut_gain / total_gain) if abs(total_gain) > 1e-9 else float("nan")
    out["verdict"] = {
        "total_gain_over_cv_m": round(total_gain, 5),
        "gain_without_latent_content_m": round(shortcut_gain, 5),
        "shortcut_fraction": round(frac, 4) if frac == frac else None,
        # ⭐ THE NUMBER. 1.0 = the head is a driving-dynamics predictor; 0.0 = every bit
        # of its value comes from the world model's per-window content.
        "wm_reliance": round(1.0 - frac, 4) if frac == frac else None,
        "_above_one_means": (
            "wm_reliance > 1 is NOT clipped and is not a bug: it means the latent-free "
            "arm is WORSE than constant velocity, i.e. the shortcut pathway actively "
            "hurts without the latents to steer it. That is STRONGER evidence of "
            "reliance than 1.0, and clipping would erase the distinction between "
            "'fully reliant' and 'cannot function at all without the WM'."),
        "_read": (
            "wm_reliance = 1 - (CV->mean gain) / (CV->real gain). It answers: of "
            "everything the decoder adds over a constant-velocity baseline, what "
            "fraction REQUIRES the world model's per-window content? Near 0 means the "
            "decoder is a driving-dynamics predictor that happens to sit downstream of "
            "a world model."),
        "_not_the_test": (
            "'real beats CV' is NOT evidence of WM reliance — a head using only v and "
            "its own feedback beats CV comfortably. hierarchy.py:846 names this "
            "confound: a frozen trunk lets a head co-adapt to always HAVING an input "
            "without using its per-window CONTENT. real-vs-MEAN is the strict test."),
        "_estimator": (
            "point estimates over the supplied batch, NO interval. For a decision-grade "
            "read, resample with the episode-cluster bootstrap (taniteval/ci.py) over "
            "the paired per-episode deltas."),
    }
    return out


def wm_reliance_gate(rel: dict, min_reliance: float = 0.5) -> dict:
    """PASS/FAIL on the reliance verdict, so a run cannot quietly ship a bypassed head.

    ⚠️ ``min_reliance`` is a PRE-REGISTERED threshold, not a discovered one. 0.5 means
    "at least half of what the decoder adds over constant velocity requires the world
    model". It is deliberately not 0.9: the ego-state pathway is legitimate and a real
    decoder will use both. What must never happen is a head whose value is ~entirely
    reachable without the latents.
    """
    v = rel.get("verdict", {})
    r = v.get("wm_reliance")
    if r is None:
        return {"status": "UNAVAILABLE",
                "reason": ("the head adds ~nothing over constant velocity, so the ratio "
                           "is undefined. That is itself a finding: a decoder that does "
                           "not beat CV has nothing to attribute."),
                "n": 0}
    return {"status": "PASS" if r >= min_reliance else "FAIL",
            "wm_reliance": r, "min_reliance": min_reliance,
            "_action_on_fail": (
                "the decoder is bypassing the world model. Raise shortcut_dropout, "
                "confirm detach_feedback is on, and re-check; if it still fails, the "
                "latents do not carry the information and the defect is upstream in the "
                "predictor, not in the decoder.")}
