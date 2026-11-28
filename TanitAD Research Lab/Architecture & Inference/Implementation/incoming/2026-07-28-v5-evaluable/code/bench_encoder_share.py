"""⭐ THE ENCODER'S SHARE OF A FULL v5 TRAINING STEP — the number that blocked
every attempt to size the arm.

WHY. `RIG_FIX_WIRING.md` §5.2 measured the ENCODER's fwd+bwd at 0.676-0.703x for
176x624 vs 256x640, and said plainly that "the predictor, the four brains, the
grounding and the anchored planner do not shrink with the token count, so the
run-level saving is strictly smaller than 0.68x" — and that **the encoder's
share of a full training step is STILL unmeasured**. Without it, 0.70x cannot be
turned into GPU-hours and the arm cannot be sized.

TWO MEASUREMENTS, and the primary one needs no decomposition at all:

  A. ⭐ FULL-STEP RATIO (primary, direct). The REAL `v4_loss_step` + backward +
     clip + `opt.step()`, on the REAL modules, at the REAL micro-batch, at both
     frames. The ratio of the two step times IS the run-level ratio. Nothing is
     attributed to a component, so nothing can be mis-attributed.

  B. ENCODER-ONLY fwd+bwd on the identical tensors, which turns A into a SHARE:
        share ~= (1 - step_ratio) / (1 - encoder_ratio)
     ⚠️ This is an ESTIMATE of the share, not a direct read: a standalone
     encoder timing does not reproduce the kernel overlap or the memory pressure
     it sees inside the step, and inside the step the encoder's backward also
     carries gradient from everything downstream. The DIRECTION of the bias is
     not known, so the share is reported as a bracket, not a point.

⚠️ `v4_loss_step` calls `world.encode_window` TWICE — once on the observation
window and once on the `plan.needed_fut` future frames — so the encoder's real
per-step image count is larger than `batch x window`. Both encodes are timed.

CONFOUND CONTROL (the rig-fix stream published a 22.7% figure that was page
cache, so this one is built against that failure):
  * ONE batch is pulled from the real cache and REUSED for every rep, on device,
    so the data loader is outside the timing entirely;
  * arms are INTERLEAVED and ROTATED — no arm is ever always first;
  * warm-up reps are discarded; `torch.cuda.synchronize()` around every timer;
  * medians are reported with min-max, and the ranges are printed so an overlap
    is visible rather than rounded away.

🔒 No clip id is printed or stored.

usage:
  PYTHONPATH=<stack> python3 bench_encoder_share.py \
      --cache <v2 cache dir> --reps 8 --batch 16 --out <json>
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="/workspace/v5eval/stack")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=["256x640", "176x624",
                                                  "128x576"],
                    help="⚠️ ONE ARM PER PROCESS on a 44 GB A40: three real "
                         "trunks + three AdamW states + three device batches "
                         "OOM'd (MEASURED). The driver runs this script once "
                         "per arm and merges; the rotation confound is handled "
                         "by re-running the whole set in rotated ORDER.")
    ap.add_argument("--probe-capacity", action="store_true",
                    help="⭐ find the LARGEST micro-batch that completes one "
                         "real step at this frame, and stop. MEASURED first "
                         "and reported first, because the staged v5 command "
                         "uses --batch 16 and that did NOT fit at 256x640.")
    ap.add_argument("--capacity-ladder", type=int, nargs="+",
                    default=[16, 12, 8, 6, 4, 2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    sys.path.insert(0, str(Path(a.stack) / "scripts"))
    import torch
    from torch.utils.data import DataLoader

    from tanitad.config import flagship4b_config
    from tanitad.data.calib import (PHYSICALAI_RIG_CLEAN_128x576,
                                    PHYSICALAI_RIG_CLEAN_176x624,
                                    PHYSICALAI_WIDE120_256x640)
    from tanitad.data.v2_dataset import build_v2_providers
    from tanitad.geometry import apply_frame
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.flagship_v4 import FlagshipV4Head, v4_config
    from tanitad.models.strategic_goal import GoalScalarConfig, GoalScalarHead
    from tanitad.train.flagship_losses import build_grounding, horizon_plan
    from flagship_v4_data import FlagshipV4Dataset
    import dataclasses
    from train_flagship_v4 import (CurriculumPhases, V4LossWeights,
                                   v4_loss_step)

    dev = "cuda"
    ALL_ARMS = {"256x640": PHYSICALAI_WIDE120_256x640,
                "176x624": PHYSICALAI_RIG_CLEAN_176x624,
                "128x576": PHYSICALAI_RIG_CLEAN_128x576}
    ARMS = {k: ALL_ARMS[k] for k in a.arms}
    parent = PHYSICALAI_WIDE120_256x640

    out: dict = {"host": "pod2", "cache": a.cache, "batch": a.batch,
                 "reps": a.reps, "warmup_discarded": a.warmup,
                 "gpu": torch.cuda.get_device_name(0),
                 "evidence_class": "MEASURED (ours; artifact = this JSON)",
                 "arms": {}}

    built = {}
    for tag, frame in ARMS.items():
        cfg = flagship4b_config()
        cfg.speed_input = True
        cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
        if getattr(cfg, "tactical_pred", None) is not None:
            cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred,
                                                    action_dim=3)
        apply_frame(cfg, frame)
        cfg.train.rollout_k = 4
        plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)

        sl = None if frame == parent else frame
        eps = build_v2_providers(a.cache, lru_size=8, frame=sl, verbose=False)
        ds = FlagshipV4Dataset(eps[:24], window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, drop_last=True,
                        num_workers=4)
        batch = next(iter(dl))
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        del dl, eps

        world = WorldModel(cfg).to(dev)
        grounding = build_grounding(world.state_dim, device=dev)
        hcfg = v4_config()
        hcfg.state_dim = world.state_dim
        hcfg.cond_imagination = False
        hcfg.window = cfg.predictor.window
        head = FlagshipV4Head(hcfg).to(dev)
        goal_head = GoalScalarHead(
            GoalScalarConfig(in_dim=world.state_dim)).to(dev)
        hg = list(head.parameters()) + list(grounding.parameters()) \
            + list(goal_head.parameters())
        opt = torch.optim.AdamW(
            [{"params": hg, "lr": 1e-4, "name": "head"},
             {"params": list(world.parameters()), "lr": 1e-4, "name": "trunk"}],
            weight_decay=0.01)
        gh, gw = cfg.encoder.token_grid()
        n_fut = len(plan.needed_fut)
        built[tag] = dict(cfg=cfg, plan=plan, world=world, grounding=grounding,
                          head=head, goal_head=goal_head, opt=opt, batch=batch,
                          hg=hg, ds=ds)
        out["arms"][tag] = {
            "frame": frame.to_dict(), "token_grid": [gh, gw],
            "n_tokens": gh * gw,
            "images_encoded_per_step": a.batch * (cfg.predictor.window + n_fut),
            "obs_window": cfg.predictor.window, "n_future_encoded": n_fut,
            "encoder_params_M": round(
                sum(p.numel() for n, p in world.named_parameters()
                    if n.startswith("encoder")) / 1e6, 3),
            "total_params_M": round(
                (sum(p.numel() for p in world.parameters())
                 + sum(p.numel() for p in hg)) / 1e6, 3),
        }
        print(f"[built] {tag}: {gh}x{gw}={gh*gw} tokens, "
              f"{a.batch*(cfg.predictor.window+n_fut)} images/step", flush=True)

    phases = CurriculumPhases(2000, 8000)
    lw = V4LossWeights()

    def full_step(b) -> float:
        """The REAL step: v4_loss_step -> backward -> clip -> opt.step."""
        torch.cuda.synchronize(); t = time.perf_counter()
        b["opt"].zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            total, _ = v4_loss_step(b["world"], b["grounding"], b["head"],
                                    b["batch"], b["plan"], b["cfg"], 12000,
                                    phases, lw, lam_mode="sched", lam_mult=1.0,
                                    device=dev, goal_head=b["goal_head"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(b["hg"], 1.0)
        torch.nn.utils.clip_grad_norm_(list(b["world"].parameters()), 1.0)
        b["opt"].step()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    def enc_only(b) -> float:
        """The SAME two encode_window calls the step makes, fwd+bwd, alone."""
        torch.cuda.synchronize(); t = time.perf_counter()
        b["world"].zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            s = b["world"].encode_window(b["batch"]["frames"])
            f = b["world"].encode_window(
                b["batch"]["future_frames"][:, b["plan"].needed_fut])
            loss = s.float().pow(2).mean() + f.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    tags = list(ARMS)

    # ---- ⭐ CAPACITY PROBE ------------------------------------------------- #
    # The staged v5 command is `--batch 16`. Whether that FITS is a launch fact,
    # not a footnote, and it is measured before anything is timed.
    if a.probe_capacity:
        for t in tags:
            b = built[t]
            fits, oom = [], []
            for n in sorted(a.capacity_ladder, reverse=True):
                dl = DataLoader(b["ds"], batch_size=n, shuffle=False,
                                drop_last=True, num_workers=2)
                bb = {k: (v.to(dev) if torch.is_tensor(v) else v)
                      for k, v in next(iter(dl)).items()}
                del dl
                probe = dict(b); probe["batch"] = bb
                torch.cuda.reset_peak_memory_stats()
                try:
                    full_step(probe)
                    fits.append({"batch": n, "peak_MiB": round(
                        torch.cuda.max_memory_allocated() / 2**20)})
                except torch.OutOfMemoryError:
                    oom.append(n)
                del bb, probe
                b["opt"].zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            out["arms"][t]["capacity"] = {
                "gpu": out["gpu"],
                "gpu_total_MiB": round(
                    torch.cuda.get_device_properties(0).total_memory / 2**20),
                "grad_checkpoint": bool(b["cfg"].encoder.grad_checkpoint),
                "ladder": sorted(a.capacity_ladder, reverse=True),
                "OOM_at": sorted(oom, reverse=True),
                "fits": fits,
                "max_micro_batch": max([f["batch"] for f in fits], default=0)}
            print(f"[capacity] {t}: max micro-batch "
                  f"{out['arms'][t]['capacity']['max_micro_batch']}  "
                  f"OOM at {sorted(oom, reverse=True)}", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"-> {a.out}")
        return 0

    for kind, fn in (("full_step_s", full_step), ("encoder_only_s", enc_only)):
        samples = {t: [] for t in tags}
        for r in range(a.warmup + a.reps):
            order = tags[r % len(tags):] + tags[:r % len(tags)]   # ROTATED
            for t in order:
                dt = fn(built[t])
                if r >= a.warmup:
                    samples[t].append(dt)
        for t in tags:
            v = sorted(samples[t])
            out["arms"][t][kind] = {
                "median": round(st.median(v), 5), "min": round(v[0], 5),
                "max": round(v[-1], 5), "n": len(v),
                "all": [round(x, 5) for x in v]}
        print(f"[{kind}] " + "  ".join(
            f"{t}={out['arms'][t][kind]['median']:.4f}" for t in tags),
            flush=True)
        torch.cuda.empty_cache()

    for t in tags:
        A = out["arms"][t]
        A["peak_mem_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)
        A["encoder_share_of_step_standalone"] = round(
            A["encoder_only_s"]["median"] / A["full_step_s"]["median"], 4)
    out["note"] = (
        "ratios ACROSS arms are computed by merge_encoder_share.py, because one "
        "arm per process is required on a 44 GB A40.")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({t: {"full_step_s": out["arms"][t]["full_step_s"]["median"],
                          "encoder_only_s": out["arms"][t]["encoder_only_s"]["median"],
                          "share_standalone":
                              out["arms"][t]["encoder_share_of_step_standalone"]}
                      for t in tags}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
