"""Joint trainer for the FULL 4-brain flagship (D-030 recovery, no toys).

Trains ALL brains FROM SCRATCH, jointly, on the real image mix (comma2k19 +
PhysicalAI epcaches = ``--data realmix``): the from-scratch ViT encoder + readout,
the operative predictor (intent-conditioned), the tactical-predictor dynamics,
the TRAINED tactical policy (maneuver + 2 s goal + intent), the TRAINED strategic
transformer (context + route), with hierarchical metric grounding at every level
and the SIGReg position-subspace relaxation.

Loss = the SHARED 4-brain assembly (``tanitad.train.flagship_losses.flagship_loss``:
JEPA + hierarchical grounding + maneuver CE + route CE + SIGReg-position-relaxed +
inv-dyn) + H15 imagination (added here — encoder/token-grid specific). The SAME
shared assembly is what a future 4-brain ``refa_train`` calls with
``sigreg_variant="pred_only"`` and a frozen-DINO encoder, so flagship vs REF-A
differ ONLY in the encoder and the SIGReg target.

Data / cgroup-guard / rollout / logging patterns mirror ``train_worldmodel`` and
``finetune_traj``: cached-mmap episodes, a PRE-ARMED v1/v2-cgroup page-cache
sweeper, cosine LR with warmup, gradient accumulation, atomic ckpt save/resume,
JSON-line logs (every component, flush=True). The grounding heads are saved under
a SEPARATE ckpt key so a vanilla WorldModel still loads ``ckpt["model"]``.

Usage (pod1, the definitive corrected-geometry flagship run):
  python scripts/train_flagship4b.py --data realmix \
     --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
     --episodes 250 --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint \
     --out /workspace/experiments/flagship4b-30k
Smoke (CPU):
  python scripts/train_flagship4b.py --data toy --config smoke --steps 15 \
     --batch-size 4 --out <dir> --log-every 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make the local checkout authoritative when run as a script (a direct script
# run does not put cwd on sys.path, so an editable install elsewhere would win):
# prepend the stack root (for `tanitad`) and the scripts dir (for refb_*).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import refb_labels  # noqa: E402  (scripts/refb_labels.py — pseudo-labels)
from finetune_traj import start_cache_guard  # noqa: E402  (reuse the OOM guard)
from refb_train import FailLoudWindowDataset  # noqa: E402  (fail-loud + nav fields)

from tanitad.config import (flagship4b_config,  # noqa: E402
                            flagship4b_reduced_config,
                            flagship4b_smoke_config)
from tanitad.data.mixing import MixedWindowDataset, load_episode  # noqa: E402
from tanitad.data.toy_driving import generate_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.imagination import imagination_nll, sector_mask  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights, build_grounding,  # noqa: E402
                                           flagship_loss, horizon_plan)
from tanitad.train.train_worldmodel import cosine_lr  # noqa: E402

# The flagship's SIGReg target: the full latent AND predictions, with the fixed
# ego-motion subspace exempted (position-relaxed, §B.3). The frozen-DINO twin
# (scripts/refa_train4b) is the SAME trainer with this one argument flipped to
# "pred_only" — the machine-checkable difference the parity test pins.
SIGREG_VARIANT = "full_relaxed"


# --------------------------------------------------------------------------- #
# v3enc STAGED encoder-grounding schedule (diagnostic 2026-07-19)              #
# --------------------------------------------------------------------------- #
def staged_lever_schedule(step: int) -> tuple[float, int]:
    """Effective ``(decorr_weight, rollout_k)`` at global ``step`` for the v3enc
    run. The flagshipv2-6k diagnostic showed the four v2 encoder levers, fired
    SIMULTANEOUSLY from step 0, strip the operative speed readout faster than it
    re-grows (v2 never reaches v1 by 30k). This staggers the two step-conditional
    levers so grounding is established before decorrelation + long rollout bite:

        decorr weight : 0.0  for step <10000, then 0.02   (was constant 0.05)
        rollout-k     : 4 (<5000) -> 8 (<10000) -> 12 (>=10000) (was constant 12)

    The two CONSTANT levers (invdyn_gradscale 0.5, fa_dropout 0.15) are set on the
    config in ``train`` (they do not vary with step). The plan is always built at
    the schedule MAX (rollout_k=12) so ``needed_fut`` covers ``range(12)`` — the
    per-step runtime K here only ever SHRINKS the K actually rolled out, never
    exceeds the encoded futures. Pure function so the smoke can probe boundaries.
    """
    decorr_w = 0.0 if step < 10000 else 0.02
    if step < 5000:
        k = 4
    elif step < 10000:
        k = 8
    else:
        k = 12
    return decorr_w, k


# --------------------------------------------------------------------------- #
# Dataset — fail-loud windows + nav fields (REF-B) + the maneuver pseudo-label #
# --------------------------------------------------------------------------- #
class FlagshipWindowDataset(FailLoudWindowDataset):
    """REF-B fail-loud windowing (nav_cmd/nav_valid/route_target) EXTENDED with
    the maneuver pseudo-label (kinematic, ``refb_labels``) at ``maneuver_h``.

    The trainer/loss read every pseudo-label from the batch, so the shared loss
    body imports no label code (package hygiene). Waypoint targets are derived in
    the loss from odometry poses (``gt_ego_waypoints``)."""

    def __init__(self, episodes, window: int, max_horizon: int, maneuver_h: int,
                 channels: int | None = None, labels_v2: bool = False):
        super().__init__(episodes, window, max_horizon, channels=channels,
                         labels_v2=labels_v2)
        assert maneuver_h <= max_horizon, (maneuver_h, max_horizon)
        self.maneuver_h = maneuver_h

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        p_last = item["pose_last"]
        if self.labels_v2:
            # v2 curvature-gated 2 s maneuver: a gentle highway curve (large R)
            # that v1 calls a turn stays lane_keep; tight junction turns still
            # turn; accel/brake/stop is preserved. classify_maneuver_v2 reads the
            # whole sub-path arc, so feed pose_last + future_poses[:maneuver_h].
            item["maneuver_label"] = refb_labels.window_maneuver_labels_v2(
                p_last[None], item["future_poses"][None],
                horizon=self.maneuver_h)[0].long()
        else:                                  # v1 path (byte-identical)
            p1 = item["future_poses"][self.maneuver_h - 1]
            item["maneuver_label"] = refb_labels.classify_maneuver(
                p_last[2], p1[2], p_last[3], p1[3]).long()
        # v2 lever 1: pose at t-1 (OBSERVED) so the loss can build yr0 = the
        # wrapped yaw delta / dt without touching the future. window >= 2 always.
        e_i, t = self.index[i]
        item["pose_prev"] = torch.as_tensor(
            self.episodes[e_i].poses[t + self.window - 2]).float()
        return item


def _wrap(episodes, cfg, plan, channels):
    return FlagshipWindowDataset(episodes, window=cfg.predictor.window,
                                 max_horizon=plan.max_horizon,
                                 maneuver_h=plan.maneuver_h, channels=channels,
                                 labels_v2=cfg.v2_labels)


def _cache_split(cache_dir: Path, split: str, n: int):
    dirs = sorted(cache_dir.glob(f"*{split}*"))
    assert dirs, f"no *{split}* dir under {cache_dir}"
    files = sorted(dirs[-1].glob("ep_*.pt"))
    files = files[:n] if n else files
    assert files, f"no ep_*.pt in {dirs[-1]}"
    return [load_episode(str(p), mmap=True) for p in files]


def build_datasets(cfg, plan, data: str, cache_dirs, n_episodes: int,
                   sim_frac: float, seed: int):
    """Datasets for the flagship. ``realmix`` = mix the ``--cache-dirs`` train
    sets (on the pod: the comma2k19 + PhysicalAI epcache roots — the image-cache
    realmix); ``cached`` = a single cache root; ``toy`` = procedural episodes
    (CI / no-cache dry run). Val stays a concat of the corpora's val splits."""
    channels = cfg.encoder.in_channels
    if data == "toy":
        ids = list(range(max(4, n_episodes)))
        n_val = max(1, len(ids) // 5)
        steps = cfg.predictor.window + plan.max_horizon + 40
        tr = [generate_episode(i, steps=steps, size=cfg.encoder.image_size)
              for i in ids[n_val:]]
        va = [generate_episode(i, steps=steps, size=cfg.encoder.image_size)
              for i in ids[:n_val]]
        return _wrap(tr, cfg, plan, channels), _wrap(va, cfg, plan, channels)
    assert cache_dirs, f"--cache-dirs required for --data {data}"
    roots = [Path(c) for c in cache_dirs]
    if data == "cached":
        roots = roots[:1]
    train_sets, val_sets = [], []
    for r in roots:
        train_sets.append(_wrap(_cache_split(r, "train", n_episodes), cfg, plan,
                                channels))
        try:
            val_sets.append(_wrap(_cache_split(r, "val", n_episodes), cfg, plan,
                                  channels))
        except AssertionError:
            pass
    if len(train_sets) == 1:
        train = train_sets[0]
    else:
        frac = min(max(sim_frac, 0.0), 1.0)      # share of the 2nd corpus
        weights = ([1.0 - frac, frac] if len(train_sets) == 2
                   else [1.0] * len(train_sets))
        train = MixedWindowDataset(list(zip(train_sets, weights)), seed=seed)
        print(f"[data] realmix: {train.mix_report()}", flush=True)
    val = (torch.utils.data.ConcatDataset(val_sets) if len(val_sets) > 1
           else (val_sets[0] if val_sets else None))
    return train, val


# --------------------------------------------------------------------------- #
# H15 imagination — encoder/token-grid specific, added on top of the shared L  #
# --------------------------------------------------------------------------- #
def h15_loss(model, frames, fut, cfg, device):
    if model.imagination is None or not (torch.rand(()) < cfg.h15.mask_prob):
        return torch.zeros((), device=device)
    masked, vis = sector_mask(frames[:, -1], model.encoder.grid_hw)
    tok_belief = model.encode_tokens(masked)
    tok_true = model.encode_tokens(fut[:, 0])
    imag_pred, logvar = model.imagination(tok_belief, vis)
    return imagination_nll(imag_pred, tok_true, logvar, vis, cfg.h15.observed_weight)


def _health(states) -> dict:
    with torch.no_grad():
        flat = states.detach().float().reshape(-1, states.shape[-1])
        s = torch.linalg.svdvals(flat - flat.mean(0))
        p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
        erank = float(torch.exp(-(p * p.log()).sum()))
        dim_std = float(flat.std(0).mean())
    return {"erank": round(erank, 1), "dim_std": round(dim_std, 5)}


# --------------------------------------------------------------------------- #
# Train                                                                        #
# --------------------------------------------------------------------------- #
def train(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    use_amp = (not args.no_amp) and device == "cuda"

    cfg = {"smoke": flagship4b_smoke_config,
           "flagship4b_reduced": flagship4b_reduced_config,
           "flagship4b": flagship4b_config}[args.config]()
    if args.grad_checkpoint:
        cfg.encoder.grad_checkpoint = True
    if args.rollout_k is not None:
        cfg.train.rollout_k = args.rollout_k
    # v2 retrain pack (fleet directive 2026-07-17): enable all seven levers.
    # Lever 3 (rollout-k) stays an explicit arg: pass --rollout-k 12 with --v2.
    if args.v2:
        cfg.v2_ego_to_planners = True
        cfg.v2_ego_dropout = 0.25
        cfg.v2_fa_dropout = 0.3
        cfg.v2_goal_decode = True
        cfg.v2_nav_dropout = 0.5
        cfg.v2_traj_jerk = 0.02
        cfg.v2_gated_intent = True
        cfg.v2_anchor_tactical = True      # (8) TIME-anchored multi-anchor decoder
        cfg.v2_route_from_vision = True    # (9) LEVER A: always-on route-from-vision aux
        cfg.v2_encoder_ego_decorr = True   # (10) LEVER B: encoder<->ego decorrelation
        cfg.v2_labels = True               # (12) curvature-relative v2 strategic/
                                           # tactical LABELS (refb_labels v2): road-
                                           # following curves stay `follow`, ambiguous
                                           # junction windows -> nav_valid=False (out
                                           # of the route CE + route-from-vision aux),
                                           # maneuver via classify_maneuver_v2.
                                           # Override: --labels-v2 / --no-labels-v2.
        cfg.v2_invdyn_gradscale = 0.25     # (11) Part A: soft gradient-decouple the
                                           # metric-invdyn REAL-PAIR probe (a) from
                                           # the encoder (research 2026-07-18);
                                           # --invdyn-gradscale overrides for the
                                           # {1.0,0.5,0.25,0.0} 5k ablation gate
    # v0 speed-input — the PROVEN operative fix (review 2026-07-18: the v1 run
    # used a pod-side trainer never committed; this commits it). Implied by
    # --v2, standalone via --speed-input (v1 parity). action_dim 2 -> 3 on the
    # operative + tactical_pred predictors; flagship_losses appends the channel.
    if args.speed_input or args.v2:
        cfg.speed_input = True
        cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
        if getattr(cfg, "tactical_pred", None) is not None:
            cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred,
                                                    action_dim=3)
    # lever 3 pairing: --v2 without an explicit --rollout-k gets the documented
    # k=12 (fa-dropout + the drift fix only bite inside the K-step rollout).
    if args.v2 and args.rollout_k is None:
        cfg.train.rollout_k = 12
    if args.sigreg_free_dims is not None:
        cfg.loss.sigreg.free_dims = args.sigreg_free_dims
    # Part A ablation override: wins over both the default (1.0) AND --v2's 0.25,
    # so `--v2 --invdyn-gradscale 1.0` runs the CONTROL arm and 0.5/0.25/0.0 the
    # ablation sweep for the 5k mid-checkpoint gate.
    if args.invdyn_gradscale is not None:
        cfg.v2_invdyn_gradscale = args.invdyn_gradscale
    # --labels-v2 / --no-labels-v2 override the v2 LABEL gate independently of the
    # rest of the --v2 pack: `--v2 --no-labels-v2` runs the v2 model levers with
    # v1 labels (control arm); `--labels-v2` alone puts v2 labels on an otherwise-
    # v1 run. None (neither flag) = follow whatever --v2 set above.
    if args.labels_v2 is not None:
        cfg.v2_labels = args.labels_v2

    # ---- v3enc STAGED encoder-grounding levers (diagnostic 2026-07-19) --------
    # Builds ON the --v2 pack. Two levers become CONSTANTS here (softer than v2);
    # the other two (decorr weight, rollout-k) are STEP-CONDITIONAL and applied
    # per-step in the loop via staged_lever_schedule(). Default-off (no flag) is
    # byte-identical to v1/v2 — this whole block is skipped.
    if args.staged_levers:
        assert args.v2, "--staged-levers builds on the v2 pack (pass --v2 too)"
        cfg.v2_fa_dropout = 0.15           # was 0.3 (constant; gentler withhold)
        if args.invdyn_gradscale is None:  # explicit --invdyn-gradscale still wins
            cfg.v2_invdyn_gradscale = 0.5  # was 0.25 (constant; softer decouple)
        # Build the plan at the schedule MAX so needed_fut covers range(12); the
        # runtime K (4/8/12) is set per-step in the loop and only ever <= 12.
        cfg.train.rollout_k = 12
        print("[staged] v3enc encoder-grounding schedule ON: decorr 0.0<10k->0.02"
              ", rollout-k 4<5k/8<10k/12>=10k, invdyn_gradscale "
              f"{cfg.v2_invdyn_gradscale}, fa_dropout {cfg.v2_fa_dropout}",
              flush=True)

    plan = horizon_plan(cfg, op_fwd_k=args.op_fwd_k, tac_fwd_k=args.tac_fwd_k,
                        str_fwd_k=args.str_fwd_k)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PRE-ARM the OOM guard BEFORE any heavy allocation / the loop.
    if args.cache_dirs:
        start_cache_guard(args.cache_dirs, limit_gb=args.guard_limit_gb)

    ds_train, ds_val = build_datasets(cfg, plan, args.data, args.cache_dirs,
                                      args.episodes, args.sim_frac, args.seed)
    assert len(ds_train) >= args.batch_size, \
        f"only {len(ds_train)} windows for batch {args.batch_size}"
    dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                    drop_last=True, num_workers=args.workers,
                    persistent_workers=bool(args.workers))
    print(f"[data] {len(ds_train)} train windows, window {cfg.predictor.window}, "
          f"max_h {plan.max_horizon}, needed_fut {plan.needed_fut}, "
          f"batch {args.batch_size}x{args.accum}, sigreg_free_dims "
          f"{cfg.loss.sigreg.free_dims}", flush=True)

    model = WorldModel(cfg).to(device)
    grounding = build_grounding(model.state_dim, device=device)
    weights = LossWeights(
        pred=args.pred_weight, tacpred=args.tacpred_weight, roll=args.roll_weight,
        goal=args.goal_weight, wp=args.wp_weight, man=args.man_weight,
        route=args.route_weight, route_vis=args.route_vis_weight,
        invdyn=args.invdyn_weight, fwd=args.fwd_weight,
        sigreg=cfg.loss.sigreg.weight, inv=cfg.loss.inv_dyn_weight,
        decorr=args.decorr_weight)
    params = list(model.parameters()) + list(grounding.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=cfg.train.betas,
                            weight_decay=args.weight_decay)

    def param_table():
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return {
            "operative": c(model.predictor) + c(model.inv_dyn),
            "tactical_pred": c(model.tactical_pred) if model.tactical_pred else 0,
            "tactical_policy": c(model.tactical_policy),
            "strategic_policy": c(model.strategic_policy),
            "encoder": c(model.encoder) + c(model.readout),
            "h15": c(model.imagination) if model.imagination else 0,
            "grounding_heads": c(grounding),
            "total_model": c(model),
            "total_trainable": sum(p.numel() for p in params),
        }

    # Resume (grounding heads travel under their own key).
    ckpt_path = out_dir / "ckpt.pt"
    step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        grounding.load_state_dict(ck["grounding"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] resuming at step {step}", flush=True)

    ptab = param_table()
    cfg_record = {
        "arch": "flagship-4b", "config": args.config, "cfg": cfg.to_json(),
        "data": args.data, "cache_dirs": args.cache_dirs,
        "horizon_plan": {"level_cfg": {k: [list(h), fk] for k, (h, fk)
                                       in plan.level_cfg.items()},
                         "goal_h": plan.goal_h, "maneuver_h": plan.maneuver_h,
                         "needed_fut": plan.needed_fut,
                         "max_horizon": plan.max_horizon},
        "weights": vars(weights), "sigreg_free_dims": cfg.loss.sigreg.free_dims,
        "pose_scale": args.pose_scale, "param_breakdown": ptab,
    }
    if args.staged_levers:                 # v3enc: record the step-conditional plan
        cfg_record["staged_levers"] = {
            "decorr_weight": "0.0 for step<10000, then 0.02",
            "rollout_k": "4 for step<5000, 8 for step<10000, then 12",
            "invdyn_gradscale": cfg.v2_invdyn_gradscale,
            "fa_dropout": cfg.v2_fa_dropout}
    (out_dir / "config.json").write_text(
        json.dumps(cfg_record, indent=2, default=str), encoding="utf-8")
    print(f"[init] params {ptab['total_trainable']/1e6:.2f}M "
          f"(model {ptab['total_model']/1e6:.2f}M + grounding "
          f"{ptab['grounding_heads']/1e6:.3f}M) | {json.dumps(ptab)}", flush=True)

    model.train()
    grounding.train()
    data_iter = iter(dl)
    accum = max(1, args.accum)
    t_data = t_step = 0.0
    logf = (out_dir / "train_log.jsonl").open("a")
    t0 = time.time()

    _MILESTONES = (5000, 15000, 20000, 30000)   # preserved for the gate protocol

    def save_ckpt(s):
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(),
                    "grounding": grounding.state_dict(),
                    "opt": opt.state_dict(), "step": s}, tmp)
        tmp.replace(ckpt_path)
        print(f"[ckpt] saved at step {s} -> {ckpt_path}", flush=True)
        # milestone archive (Sayed 2026-07-18): keep 5k/15k/20k/30k for gating.
        for m in _MILESTONES:
            arch = ckpt_path.with_name(f"ckpt_step{m}.pt")
            if s >= m and not arch.exists():
                # ATOMIC: a bare copy2 writes the final name directly, so a kill
                # mid-copy leaves a truncated milestone that exists() then treats
                # as done forever — the gate would later load a corrupt ckpt.
                from tanitad.train.ckpt_io import atomic_archive
                atomic_archive(ckpt_path, arch)
                print(f"[ckpt] milestone archived: {arch.name}", flush=True)

    while step < args.steps:
        # v3enc step-conditional levers: mutate the SHARED weights/cfg the loss
        # reads by reference (weights.decorr -> flagship_loss total; cfg.train.
        # rollout_k -> the runtime K, always <= the plan's build-time 12). No-op
        # unless --staged-levers (byte-identical v1/v2 otherwise).
        if args.staged_levers:
            weights.decorr, cfg.train.rollout_k = staged_lever_schedule(step)
        lr = cosine_lr(step, args.steps, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        t_s0 = time.perf_counter()
        log: dict = {}
        for _micro in range(accum):
            t_d0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            t_data += time.perf_counter() - t_d0
            frames = batch["frames"].to(device)
            fut = batch["future_frames"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                states = model.encode_window(frames)
                fut_states = model.encode_window(fut[:, plan.needed_fut])
                total, log, parts = flagship_loss(
                    model, grounding, batch, states, fut_states, plan, cfg,
                    weights=weights, sigreg_variant=SIGREG_VARIANT,
                    sigreg_free_dims=cfg.loss.sigreg.free_dims,
                    pose_scale=args.pose_scale,
                    fwd_step_weight=args.fwd_step_weight, device=device)
                loss_h15 = h15_loss(model, frames, fut, cfg, device)
                total = total + cfg.h15.weight * loss_h15
            (total / accum).backward()
            log["h15"] = float(loss_h15.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % args.ckpt_every == 0:
            save_ckpt(step)

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                hs = model.encode_window(batch["frames"].to(device))
            row = {"step": step, "loss": float(total.item()), "lr": round(lr, 8),
                   "data_s": round(t_data, 1), "step_s": round(t_step, 1)}
            row.update(log)
            row.update(_health(hs))
            t_data = t_step = 0.0
            line = json.dumps(row)
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()
        step += 1

    save_ckpt(step - 1)
    logf.close()
    summary = {"done": True, "final_step": step - 1,
               "wallclock_s": round(time.time() - t0, 1),
               "param_breakdown": ptab, "out": str(ckpt_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("done", "final_step",
                                              "wallclock_s")}), flush=True)
    print("FLAGSHIP4B_DONE", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["realmix", "cached", "toy"],
                    default="realmix",
                    help="realmix = mix --cache-dirs train sets (comma2k19 + "
                         "PhysicalAI epcaches on the pod); cached = single root; "
                         "toy = procedural (CI / no-cache dry run)")
    ap.add_argument("--cache-dirs", nargs="+", default=None,
                    help="epcache roots, each with *train*/*val* dirs of ep_*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config",
                    choices=["flagship4b", "flagship4b_reduced", "smoke"],
                    default="flagship4b")
    ap.add_argument("--episodes", type=int, default=0,
                    help="max episodes per cache split (0 = all); toy: #episodes")
    ap.add_argument("--sim-frac", type=float, default=0.6,
                    help="realmix: share of the 2nd corpus (PhysicalAI) windows")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--v2", action="store_true",
                    help="enable the v2 retrain levers (fleet directive "
                         "2026-07-17): ego->planners + ego-dropout 0.25 + "
                         "fa-dropout 0.3 + goal-decode + nav-dropout 0.5 + "
                         "jerk 0.02 + gated-intent + anchor-tactical + "
                         "speed-input; defaults rollout-k to 12 if unset")
    ap.add_argument("--staged-levers", action="store_true",
                    help="v3enc STAGED encoder-grounding schedule (diagnostic "
                         "2026-07-19): builds on --v2 but stages the four encoder "
                         "levers — decorr 0.0<10k->0.02, rollout-k 4<5k/8<10k/12, "
                         "invdyn_gradscale 0.5, fa_dropout 0.15. Off == v1/v2 "
                         "byte-identical")
    ap.add_argument("--speed-input", action="store_true",
                    help="v0 (t=0 speed / 10.0) as the 3rd operative action "
                         "channel — the PROVEN speed fix (implied by --v2)")
    ap.add_argument("--labels-v2", dest="labels_v2", action="store_true",
                    default=None,
                    help="force the v2 curvature-relative strategic/tactical "
                         "labels ON, overriding the --v2 default (refb_labels v2: "
                         "road-curves stay follow; ambiguous junctions get "
                         "nav_valid=False). Default: follow --v2")
    ap.add_argument("--no-labels-v2", dest="labels_v2", action="store_false",
                    help="force the v2 labels OFF even under --v2 (v1-labels "
                         "control arm)")
    ap.add_argument("--rollout-k", type=int, default=None,
                    help="K-step recursive rollout (bake-off lever; default cfg)")
    # per-level grounding rollout horizons (op fine / tac 2 s / str long)
    ap.add_argument("--op-fwd-k", type=int, default=4)
    ap.add_argument("--tac-fwd-k", type=int, default=16)
    ap.add_argument("--str-fwd-k", type=int, default=20)
    ap.add_argument("--fwd-step-weight", type=float, default=0.5)
    ap.add_argument("--pose-scale", type=float, default=10.0)
    ap.add_argument("--sigreg-free-dims", type=int, default=None,
                    help="exempt the first N state dims from SIGReg (§B.3); "
                         "default: the config value (flagship 64)")
    # loss weights
    ap.add_argument("--pred-weight", type=float, default=1.0)
    ap.add_argument("--tacpred-weight", type=float, default=0.5)
    ap.add_argument("--roll-weight", type=float, default=0.5)
    ap.add_argument("--goal-weight", type=float, default=0.5)
    ap.add_argument("--wp-weight", type=float, default=1.0)
    ap.add_argument("--man-weight", type=float, default=0.5)
    ap.add_argument("--route-weight", type=float, default=0.5)
    ap.add_argument("--route-vis-weight", type=float, default=0.3,
                    help="LEVER A: route-from-vision aux CE weight (v2 lever 9; "
                         "only active with --v2 / v2_route_from_vision)")
    ap.add_argument("--invdyn-weight", type=float, default=2.0)
    ap.add_argument("--fwd-weight", type=float, default=1.0)
    ap.add_argument("--invdyn-gradscale", type=float, default=None,
                    help="Part A loss-rebalance (research 2026-07-18): straight-"
                         "through gradient scale on the encoder-latent inputs to "
                         "the metric-invdyn REAL-PAIR term (a) ONLY. 1.0 = today "
                         "exactly (byte-identical); 0.0 = full probe-detach of "
                         "(a); --v2 sets 0.25. Overrides --v2 for the "
                         "{1.0,0.5,0.25,0.0} 5k ablation gate. Term (b) fwd-"
                         "consistency, JEPA, SIGReg, and the invdyn heads are "
                         "untouched (loss-side only; no param change)")
    ap.add_argument("--decorr-weight", type=float, default=0.05,
                    help="LEVER B: encoder<->ego decorrelation weight (v2 lever "
                         "10; only active with --v2 / v2_encoder_ego_decorr)")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--guard-limit-gb", type=float, default=60.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
