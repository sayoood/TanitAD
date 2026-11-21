"""⭐ IS OUR SigReg CONFIGURATION EXERTING ANTI-COLLAPSE PRESSURE AT OUR GEOMETRY?

⛔ THIS IS NOT A KEEP/DROP STUDY. PI ruling 2026-08-16: SigReg is a **MUST** (it
comes from the LeWM/LeJEPA line and the decision is made). The ``w_o6 = 0`` arm
here is the **CONTROL that makes the on-arm interpretable** — nobody is proposing
to ship it. The open questions are whether SigReg is used CORRECTLY, whether it
is FORGOTTEN, and whether its effectiveness is VALIDATED against the collapse
that is specific to predictive/JEPA architectures.

⛔ THE COLLAPSE MECHANISM BEING GUARDED AGAINST — and why the obvious synthetic
experiment CANNOT see it. In the real trainer (``train_v6_staged.py:1966-1978``)
the O1/O2/O3/O5 targets are produced by the model's OWN encoder and then
detached::

    z_flat = stack.readout(stack.encoder(ff.reshape(fb * fk, ...)))
    z_true = [z_flat[:, j].detach() for j in range(need_k)]

That is self-distillation: the encoder shapes BOTH the prediction and the
target, so it can lower the loss by making its own output trivially predictable
(constant, or confined to a few directions). Detaching stops gradient INTO the
target; it does NOT remove the incentive, because the same weights produce the
target on the next step. This is the failure mode SigReg exists to prevent.

⛔ ``synthetic_train_batch`` (``train_v6_staged.py:1512``) instead sets
``z_true_steps = torch.randn(...)`` — FIXED EXTERNAL targets, independent of the
encoder. Collapsing the encoder cannot make a fixed random target easier to hit,
so under those targets there is **NO collapse incentive at all**. An ablation run
on that fixture would report "SigReg does nothing" for a reason that has nothing
to do with SigReg. ⇒ the two target regimes are the two CONDITIONS here:

  SELF   targets from the model's own encoder, detached  -> collapse-CAPABLE
  FIXED  targets fixed external randn                    -> NO collapse incentive
         == the NULL-DETECTION CONTROL. A probe that reports an effect HERE is
            reporting an artifact, and this whole file is untrustworthy.

CONFIGURATION AXIS (the reframed question). ``--sigreg-slices`` sets how many
random 1-D projections the Epps-Pulley statistic is estimated over at
``d_op = 2048``. The live run uses **512**; ``stack/tests`` fixtures use 8. If 8
and 512 buy the same retention, the knob is not doing what it is assumed to do;
if they differ, the live value is load-bearing and must be recorded as such.

ESTIMATOR. Pooled spectrum on a FIXED PROBE SET (identical inputs for every arm,
so batch composition — the dominant noise source per SIGREG_GATE_POWER.md §2.4 —
is removed by construction, not averaged over). 32 pooled probe batches x 48 rows
= 1536 rows -> ``rank_ceiling`` = min(1535, 2048) = **1535**, i.e. ADMISSIBLE
(>= 1024), which a single 48-row call never is. Interval = leave-one-cluster-out
JACKKNIFE, block = 6 rows = the window (coverage 0.850 vs 0.250 percentile
bootstrap). Verdict = ``o6_rank_verdict``.

Run:  PYTHONUTF8=1 python w_o6_ablation.py --out ../raw/w_o6_ablation.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[6]
_STACK = _ROOT / "stack"
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    SpectrumAccumulator, V6Config, V6Stack, o6_rank_verdict)
from train_v6_staged import (  # noqa: E402
    V6LossWeights, apply_stage_freeze, v6_loss_step)

# --------------------------------------------------------------------------- #
# THE LIVE RUN'S GEOMETRY — MEASURED from the running trainer's argv, captured
# in .../2026-08-15-v6-thor-resume/code/RESTART_v6F_SW.sh. Anything that sets
# the SPECTRUM's shape or SigReg's own configuration is matched exactly; the
# encoder/predictor WIDTH and DEPTH are the small CPU fixture and are NOT.
# --------------------------------------------------------------------------- #
LIVE = {
    "readout_grid": 4, "readout_dim": 128,      # -> d_op = 4*4*128 = 2048
    "window": 6, "batch": 8, "eps_per_batch": 4,
    "w_o6": 0.1, "sigreg_slices": 512, "sigreg_free_dims": 0,
    "lr": 1e-4, "wd": 0.05, "clip": 1.0, "o3_blocks": 2, "o3_block_hw": (2, 2),
    "o3_mode": "action", "o5_mode": "uniform",
}
NOT_MATCHED = ("enc_dim 768/depth 12 -> 32/1; pred_dim 1024/depth 12 -> 32/1; "
               "frame 256x640 -> 64x64; o1_k 10 -> 2; o5_k 60 -> 2; "
               "plan_steps 60 -> 6; real PhysicalAI frames -> synthetic noise")

POOL_STEPS = 32          # -> 32*48 = 1536 rows, ceiling 1535 (>= 1024)
PROBE_BATCH = 8
BLOCK_ROWS = LIVE["window"]


def build(seed: int, slices: int) -> V6Stack:
    torch.manual_seed(seed)
    cfg = V6Config(
        d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, f_blocks=1, aux_hidden=16,
        sigreg_slices=slices, sigreg_beta=1.0,
        sigreg_free_dims=LIVE["sigreg_free_dims"],
        plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6),
        hz_op=10.0, hz_tac=2.0, hz_str=0.5, d_plan_feat=16,
        emission_hidden=16, d_goal_embed=128, n_candidates=8,
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=LIVE["readout_grid"],
                              d_readout=LIVE["readout_dim"]),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2,
                                  window=LIVE["window"], action_dim=3))
    s = V6Stack(cfg)
    s.train()
    apply_stage_freeze(s, "S-W")
    return s


def frames_for(stack: V6Stack, n: int, seed: int) -> torch.Tensor:
    c = stack.cfg.encoder.in_channels
    h, w = stack.cfg.encoder.image_hw()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, LIVE["window"], c, h, w, generator=g)


def step_batch(stack: V6Stack, step: int, seed: int, k: int,
               condition: str) -> dict:
    """One training batch. ``condition`` decides where ``z_true_steps`` comes
    from, which is the whole experiment."""
    cfg = stack.cfg
    b = LIVE["batch"]
    g = torch.Generator().manual_seed(seed * 100_000 + step)
    c = cfg.encoder.in_channels
    h, w = cfg.encoder.image_hw()
    batch = {
        "frames": torch.randn(b, LIVE["window"], c, h, w, generator=g),
        "actions2": torch.randn(b, LIVE["window"], 2, generator=g) * 0.1,
        "future_actions2": torch.randn(b, max(k, 2), 2, generator=g) * 0.1,
        "v0": torch.rand(b, generator=g) * 20.0 + 1.0,
        "gt_wp": torch.zeros(b, 2, 2),
    }
    if condition == "SELF":
        # ⭐ EXACTLY the trainer's construction (train_v6_staged.py:1974-1978):
        # the model's OWN encoder+readout on future frames, DETACHED. This is
        # what makes collapse profitable, and therefore what makes the
        # experiment able to see SigReg at all.
        ff = torch.randn(b, k, c, h, w, generator=g)
        with torch.no_grad():
            zf = stack.readout(stack.encoder(
                ff.reshape(b * k, *ff.shape[2:]))).reshape(b, k, -1)
        batch["z_true_steps"] = [zf[:, j].detach() for j in range(k)]
    elif condition == "FIXED":
        # the NULL: targets independent of the encoder, so collapsing the
        # encoder cannot lower the loss and SigReg has nothing to defend.
        batch["z_true_steps"] = [
            torch.randn(b, cfg.d_op, generator=g) for _ in range(k)]
    else:
        raise ValueError(condition)
    return batch


@torch.no_grad()
def pooled_spectrum(stack: V6Stack, probe: torch.Tensor, ci_reps: int,
                    gen_seed: int = 7) -> dict:
    """The pooled reading on the FIXED probe set — identical inputs for every
    arm, so any difference is the representation and not the batch."""
    was = stack.training
    stack.eval()
    acc = SpectrumAccumulator(capacity=POOL_STEPS, block=BLOCK_ROWS)
    for i in range(POOL_STEPS):
        fr = probe[i * PROBE_BATCH:(i + 1) * PROBE_BATCH]
        z = stack.forward(frames=fr,
                          actions=torch.zeros(fr.shape[0], LIVE["window"], 3),
                          v0=torch.full((fr.shape[0],), 10.0))["z_op_win"]
        acc.push(z)
    rep = acc.report(ci_reps=ci_reps,
                     generator=torch.Generator().manual_seed(gen_seed))
    if was:
        stack.train()
    return rep


def run_arm(seed: int, slices: int, w_o6: float, condition: str, steps: int,
            k: int, ci_reps: int, ref: dict | None) -> dict:
    """One arm. Everything except ``w_o6`` is bit-identical across the pair."""
    stack = build(seed, slices)
    probe = frames_for(stack, POOL_STEPS * PROBE_BATCH, seed + 9_000)
    weights = V6LossWeights(o6_sigreg=w_o6)
    params = [p for p in stack.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LIVE["lr"], weight_decay=LIVE["wd"])

    t0 = time.time()
    losses, o6vals = [], []
    for step in range(1, steps + 1):
        b = step_batch(stack, step, seed, k, condition)
        out = v6_loss_step(
            stack, b, stage="S-W", o1_k=2, o5_k=k, weights=weights,
            o3_mode=LIVE["o3_mode"], o3_blocks=LIVE["o3_blocks"],
            o3_block_hw=LIVE["o3_block_hw"], o5_mode=LIVE["o5_mode"],
            # ⛔ SEPARATE, FIXED generators — the whole point of 142ce34. Same
            # streams in both arms, so the ONLY difference is w_o6.
            generator=torch.Generator().manual_seed(seed * 1000 + step),
            sigreg_generator=torch.Generator().manual_seed(
                seed * 7000 + step + 11))
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, LIVE["clip"])
        opt.step()
        losses.append(float(out["loss"].detach()))
        if "o6_sigreg" in out["log"]:
            o6vals.append(float(out["log"]["o6_sigreg"]))
    train_s = time.time() - t0

    cur = pooled_spectrum(stack, probe, ci_reps)
    verdict = o6_rank_verdict(cur, ref)
    return {
        "seed": seed, "sigreg_slices": slices, "w_o6": w_o6,
        "condition": condition, "steps": steps,
        "train_s": round(train_s, 1),
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_last10_mean": st.mean(losses[-10:]),
        # ⚠️ REPORTED, NEVER READ AS THE ANSWER. "the o6 term fell" is not
        # "SigReg worked" — a regulariser's loss can fall because it is
        # satisfied OR because the representation degenerated. The rank is the
        # measurement; this column is context only.
        "o6_term_first": (o6vals[0] if o6vals else None),
        "o6_term_last": (o6vals[-1] if o6vals else None),
        "pooled_spectrum": cur,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE.parents[1] / "raw"
                                         / "w_o6_ablation.json"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ci-reps", type=int, default=32)
    a = ap.parse_args()

    t0 = time.time()
    seeds = list(range(a.seeds))
    #: (condition, slices) cells. SELF x {512, 8} is the configuration
    #: question; FIXED x 512 is the null-detection control.
    cells = [("SELF", 512), ("SELF", 8), ("FIXED", 512)]
    results, refs = [], {}

    for cond, slices in cells:
        for seed in seeds:
            # the reference is the UNTRAINED model — identical for both arms of
            # a (seed, slices) pair, so it is computed once and shared.
            key = (seed, slices)
            if key not in refs:
                s0 = build(seed, slices)
                p0 = frames_for(s0, POOL_STEPS * PROBE_BATCH, seed + 9_000)
                refs[key] = pooled_spectrum(s0, p0, a.ci_reps)
                print(f"  ref seed={seed} slices={slices}: "
                      f"ER={refs[key]['effective_rank']:.3f} "
                      f"ceiling={refs[key]['rank_ceiling']}", flush=True)
            for w in (LIVE["w_o6"], 0.0):
                r = run_arm(seed, slices, w, cond, a.steps, a.k, a.ci_reps,
                            refs[key])
                r["reference_spectrum"] = refs[key]
                results.append(r)
                print(f"  {cond} slices={slices} seed={seed} w_o6={w}: "
                      f"ER={r['pooled_spectrum']['effective_rank']:.3f} "
                      f"ret={r['verdict'].get('retention')} "
                      f"{r['verdict']['status']} ({r['train_s']}s)", flush=True)

    # ---- PAIRED per-seed deltas. ⛔ The estimator is stated: this is a
    # per-seed PAIRED difference over `--seeds` seeds, reported with its full
    # spread. It is NOT a bootstrap CI and must not be quoted as one.
    paired = {}
    for cond, slices in cells:
        key = f"{cond}_slices{slices}"
        rows = []
        for seed in seeds:
            on = next(r for r in results if r["condition"] == cond
                      and r["sigreg_slices"] == slices and r["seed"] == seed
                      and r["w_o6"] == LIVE["w_o6"])
            off = next(r for r in results if r["condition"] == cond
                       and r["sigreg_slices"] == slices and r["seed"] == seed
                       and r["w_o6"] == 0.0)
            er_on = on["pooled_spectrum"]["effective_rank"]
            er_off = off["pooled_spectrum"]["effective_rank"]
            ci_on = on["pooled_spectrum"].get("effective_rank_ci95", {})
            ci_off = off["pooled_spectrum"].get("effective_rank_ci95", {})
            sep = (isinstance(ci_on, dict) and isinstance(ci_off, dict)
                   and "lo" in ci_on and "lo" in ci_off
                   and (ci_on["lo"] > ci_off["hi"] or ci_off["lo"] > ci_on["hi"]))
            rows.append({
                "seed": seed, "er_w_o6_on": er_on, "er_w_o6_off": er_off,
                "delta_on_minus_off": er_on - er_off,
                "ratio_on_over_off": er_on / er_off if er_off else None,
                "ci_on": ci_on, "ci_off": ci_off,
                "jackknife_intervals_disjoint": bool(sep)})
        d = [r["delta_on_minus_off"] for r in rows]
        paired[key] = {
            "per_seed": rows,
            "estimator": "per-seed PAIRED difference in pooled effective_rank "
                         "(same build seed, same batch sequence, same "
                         "generator streams; ONLY w_o6 differs). n_seeds="
                         f"{len(seeds)}. NOT a bootstrap CI.",
            "delta_mean": st.mean(d), "delta_min": min(d), "delta_max": max(d),
            "delta_sd": (st.stdev(d) if len(d) > 1 else None),
            "sign_consistent": all(x > 0 for x in d) or all(x < 0 for x in d),
            "all_seeds_separated_by_jackknife":
                all(r["jackknife_intervals_disjoint"] for r in rows),
            "any_seed_separated_by_jackknife":
                any(r["jackknife_intervals_disjoint"] for r in rows),
        }

    out = {
        "meta": {
            "what": "w_o6 ablation — CONFIGURATION-AND-VALIDATION study, not "
                    "keep/drop. SigReg is a PI-level MUST; w_o6=0 is the "
                    "CONTROL that makes the on-arm interpretable.",
            "date": "2026-08-16", "stage": "S-W", "device": "cpu",
            "evidence_class": "MEASURED (ours) on a SYNTHETIC CPU build",
            "collapse_mechanism": "self-distillation: targets are the model's "
                                  "OWN encoder output, detached "
                                  "(train_v6_staged.py:1966-1978), so the "
                                  "encoder can lower the loss by making its "
                                  "own output trivially predictable",
            "conditions": {
                "SELF": "targets from the model's own encoder, detached — "
                        "collapse-CAPABLE (the real trainer's construction)",
                "FIXED": "targets fixed external randn — NO collapse "
                         "incentive. THE NULL-DETECTION CONTROL."},
            "live_geometry_matched": LIVE,
            "live_geometry_NOT_matched": NOT_MATCHED,
            "pool": {"pooled_steps": POOL_STEPS, "probe_batch": PROBE_BATCH,
                     "rows": POOL_STEPS * PROBE_BATCH * LIVE["window"],
                     "block_rows": BLOCK_ROWS,
                     "note": "FIXED probe set — identical inputs for every "
                             "arm, so batch composition (the dominant noise "
                             "source at n=48) is removed by construction"},
            "interval": "leave-one-cluster-out jackknife (coverage 0.850 vs "
                        "0.250 percentile bootstrap, SIGREG_GATE_POWER.md §4)",
            "steps": a.steps, "seeds": seeds, "k": a.k, "ci_reps": a.ci_reps,
            "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "wall_s": None,
        },
        "paired_by_cell": paired,
        "arms": results,
    }
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    for key, p in paired.items():
        print(f"{key}: delta(on-off) mean={p['delta_mean']:+.4f} "
              f"[{p['delta_min']:+.4f},{p['delta_max']:+.4f}] "
              f"sign_consistent={p['sign_consistent']} "
              f"jackknife_separated_all={p['all_seeds_separated_by_jackknife']}"
              f" any={p['any_seed_separated_by_jackknife']}")
    print(f"\nwrote {a.out}  ({out['meta']['wall_s']} s)")


if __name__ == "__main__":
    main()
