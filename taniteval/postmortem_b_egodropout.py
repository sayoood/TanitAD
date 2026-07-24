"""Post-mortem experiment B — how much of v3enc's rollout deficit is a
MEASUREMENT ARTIFACT of the ego-dropout zero-fill?

Spec (2026-07-21-flagship-v3enc-postmortem.md §9 row B): evaluate
``ckpt_step10000.pt`` with the ego-dropout mask forced OFF vs ON, on TRAIN
batches. **No training. No optimizer step. The checkpoint is opened read-only.**

THE CLAIM UNDER TEST
--------------------
``flagship_losses.py:229-231`` multiplies the v0 speed action channel by the
planner ego keep-mask, so at p=0.25 a quarter of samples are fed ``v0 = 0.0 m/s``
on ``actions`` AND ``fut_actions``.  Because the base channels are
``(steer, accel)`` (no absolute speed), 0.0 is an in-distribution "stopped".
The training-log metric ``g_op_fwd_ade_m`` is computed INSIDE that masked
forward, so part of the measured v3enc/v1 ratio may be the mask, not the model.

WHAT VARIES — NOTHING BUT THE MASK
----------------------------------
Per batch the encoder runs ONCE; ``states``/``fut_states`` are shared by every
condition (they cannot depend on actions).  Only the 3rd action channel differs:

  off    v0 present on every row                      (mask forced OFF)
  zero   v0 = 0.0 on every row                        (mask forced ON for all)
  on25   v0 = 0.0 on a random 25 % of rows            (the training condition)
  x2     v0 doubled                                   (sensitivity readout only)

``on25`` is also available ANALYTICALLY as ``0.25*zero + 0.75*off`` (exact in
expectation, zero mask-sampling noise) — both are reported.

WHY THE ENCODER-GROUNDING TERM IS A NULL BY CONSTRUCTION
--------------------------------------------------------
``grounding_losses`` term (a) (``g_*_mid_de_m``) reads ONLY ``z_t`` and
``fut_states`` — no actions.  It is therefore EXACTLY invariant to the mask, and
this script asserts that (max |Δ| == 0).  That is a harness self-check, NOT
evidence that the encoder is undamaged; see the note in the report.

ESTIMATOR
---------
Per-window values; point estimate = full-set mean; intervals =
``taniteval.ci.paired_episode_cluster_bootstrap`` over the sampled TRAIN
episodes (B=2000).  Never a single row.

RUNS ON THE POD THAT HOLDS THE CHECKPOINT — `STACK` below is a pod path and
`taniteval` is not deployed there, so the interval estimator is imported from a
sibling `expb_ci.py` (a byte-copy of `taniteval/taniteval/ci.py`) when present
and from `taniteval.ci` otherwise. Copy this file plus `taniteval/taniteval/ci.py`
(as `expb_ci.py`) into one directory on the pod. `postmortem_b_analyze.py` is
stage 2 and turns the JSONs this writes into the decision-grade results file.

Usage (pod1, GPU lock held):
  PYTHONPATH=/workspace/TanitAD/stack python3 postmortem_b_egodropout.py \
      --run-dir /workspace/experiments/flagship4b-v3enc-30k \
      --ckpt    /workspace/experiments/flagship4b-v3enc-30k/ckpt_step10000.pt \
      --batches 400 --batch-size 16 --precision bf16 --out exp_b_bf16.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

STACK = Path("/workspace/TanitAD/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

from tanitad.config import flagship4b_config                       # noqa: E402
from tanitad.models.fourbrain import WorldModel                    # noqa: E402
from tanitad.models.metric_dynamics import (decode_transitions,    # noqa: E402
                                            grounding_losses,
                                            gt_ego_waypoints,
                                            relative_ego_pose,
                                            rollout_transitions)
from tanitad.train.flagship_losses import build_grounding, horizon_plan  # noqa: E402

from train_flagship4b import build_datasets                        # noqa: E402

CONDITIONS = ("off", "zero", "on25", "x2", "perm")


# --------------------------------------------------------------------------- #
# cfg reconstruction — the trainer's OWN flag path, then pinned to the record   #
# --------------------------------------------------------------------------- #
def build_cfg(ego_dropout: float = 0.25):
    """Replicate `train_flagship4b.train()`'s cfg block for the v3enc launch
    line: `--config flagship4b --sigreg-free-dims 64 --grad-checkpoint
    --speed-input --v2 --staged-levers` (verbatim from
    /workspace/ops/runs.d/flagship-v3enc.env). Pinned by an equality assert
    against the run's recorded config.json below — so any code drift on the pod
    fails loud instead of silently changing the arm.

    ``ego_dropout`` is the run's TRAINED-WITH value and exists ONLY so the cfg
    pin can match a run that used ``--ego-dropout`` (post-mortem experiment A's
    ``flagship4b-v3enc-expA-nodrop-2k``, which recorded 0.0). It does NOT set the
    measurement condition — the five CONDITIONS below always override the v0
    channel explicitly. Default 0.25 reproduces experiment B bit-identically."""
    cfg = flagship4b_config()
    cfg.encoder.grad_checkpoint = True                 # --grad-checkpoint
    # --v2 pack
    cfg.v2_ego_to_planners = True
    cfg.v2_ego_dropout = ego_dropout
    cfg.v2_fa_dropout = 0.3
    cfg.v2_goal_decode = True
    cfg.v2_nav_dropout = 0.5
    cfg.v2_traj_jerk = 0.02
    cfg.v2_gated_intent = True
    cfg.v2_anchor_tactical = True
    cfg.v2_route_from_vision = True
    cfg.v2_encoder_ego_decorr = True
    cfg.v2_labels = True
    cfg.v2_invdyn_gradscale = 0.25
    # --speed-input (implied by --v2)
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    cfg.train.rollout_k = 12                           # --v2, no --rollout-k
    cfg.loss.sigreg.free_dims = 64                     # --sigreg-free-dims 64
    # --staged-levers
    cfg.v2_fa_dropout = 0.15
    cfg.v2_invdyn_gradscale = 0.5
    cfg.train.rollout_k = 12
    return cfg


# --------------------------------------------------------------------------- #
# per-window metric extraction — same primitives grounding_losses uses          #
# --------------------------------------------------------------------------- #
def per_window_terms(model, grounding, plan, states, fut_states, actions,
                     fut_actions, pose_last, future_poses):
    """Per-window (a) mid_de and (b) fwd_ade for every level + the A5 `inv`.

    Mirrors `metric_dynamics.grounding_losses` exactly:
      fwd_ade = ||pred_wp - gt_wp||.mean()   over [B, fwd_k]  -> kept per row
      mid_de  = mean_k ||dpose_xy - tgt_xy||                  -> kept per row
      inv     = (a_hat - actions[:, -2])^2 .mean()            -> kept per row
    `grad_scale` is omitted from term (a): its forward is bit-exactly the
    identity (x + (a-1)*(x - x.detach()), and x - x.detach() is a real 0), which
    the first-batch cross-check against grounding_losses verifies numerically.
    """
    out: dict[str, torch.Tensor] = {}
    idx_of = plan.idx_of
    k_max = max(fk for _, fk in plan.level_cfg.values())
    trans = rollout_transitions(model.predictor, states, actions, fut_actions,
                                k_max)
    z_t = states[:, -1]
    for lvl, (inv_h, fwd_k) in plan.level_cfg.items():
        pred_wp, _ = decode_transitions(grounding.step[lvl], trans, fwd_k)
        gt_wp = gt_ego_waypoints(pose_last, future_poses, range(1, fwd_k + 1))
        out[f"g_{lvl}_fwd_ade_m"] = (pred_wp.float() - gt_wp.float()) \
            .norm(dim=-1).mean(dim=1)                                    # [B]
        mid = torch.zeros(states.shape[0], device=states.device)
        for kh in inv_h:
            dpose = grounding.invdyn[lvl](z_t, fut_states[:, idx_of[kh - 1]])
            tgt = relative_ego_pose(pose_last, future_poses[:, kh - 1])
            mid = mid + (dpose[..., :2].float() - tgt[..., :2].float()) \
                .norm(dim=-1)
        out[f"g_{lvl}_mid_de_m"] = mid / len(inv_h)                      # [B]
    a_hat = model.inv_dyn(states[:, -2], states[:, -1]).float()          # [B, 3]
    sq = (a_hat - actions[:, -2].float()).pow(2)                         # [B, 3]
    out["inv"] = sq.mean(dim=-1)
    for c in range(sq.shape[-1]):
        out[f"inv_ch{c}"] = sq[:, c]
    out["a_hat_v0"] = a_hat[:, 2]                                        # [B]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir",
                    default="/workspace/data/physicalai_phase0/_epcache")
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--p-ego-dropout", type=float, default=0.25,
                    help="p(drop) for the `on25` MEASUREMENT condition — always "
                         "the v3enc training rate, whatever the arm trained with")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--expect-step", type=int, default=10000,
                    help="FAIL-LOUD guard on the checkpoint's own `step` field. "
                         "10000 = experiment B's v3enc gate ckpt (the default, "
                         "so B reproduces bit-identically); 1999 = experiment "
                         "A's flagship4b-v3enc-expA-nodrop-2k final ckpt")
    ap.add_argument("--cfg-ego-dropout", type=float, default=0.25,
                    help="the v2_ego_dropout the RUN BEING READ trained with, "
                         "used only to make the cfg pin match its config.json "
                         "(0.0 for experiment A's arm). Never a measurement knob")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    rec = json.loads((run_dir / "config.json").read_text())

    # ---- cfg + plan, PINNED to the run record -----------------------------
    cfg = build_cfg(ego_dropout=args.cfg_ego_dropout)
    got, want = json.loads(cfg.to_json()), json.loads(rec["cfg"])
    if got != want:
        diff = {k: (want.get(k), got.get(k)) for k in set(got) | set(want)
                if got.get(k) != want.get(k)}
        raise SystemExit(f"[FAIL-LOUD] reconstructed cfg != run record: {diff}")
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    rp = rec["horizon_plan"]
    assert plan.needed_fut == rp["needed_fut"], (plan.needed_fut,
                                                 rp["needed_fut"])
    assert {k: [list(h), fk] for k, (h, fk) in plan.level_cfg.items()} \
        == rp["level_cfg"], plan.level_cfg
    print(f"[pin] cfg + horizon_plan match {run_dir/'config.json'}", flush=True)

    # ---- model + grounding, READ-ONLY from the checkpoint ------------------
    model = WorldModel(cfg).to(device)
    grounding = build_grounding(model.state_dim, device=device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    ck_step = int(ck["step"])
    if ck_step != args.expect_step:
        raise SystemExit(f"[FAIL-LOUD] ckpt step {ck_step} != "
                         f"{args.expect_step}")
    model.load_state_dict(ck["model"])          # strict: any arch drift fails
    grounding.load_state_dict(ck["grounding"])
    del ck
    model.eval()
    grounding.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[ckpt] step {ck_step}, model {n_par/1e6:.2f}M params", flush=True)

    # ---- self-check: the net has NO train/eval-dependent stochastic module --
    stoch = torch.nn.modules.dropout._DropoutNd
    bn = torch.nn.modules.batchnorm._BatchNorm
    bad = [n for n, m in (list(model.named_modules())
                          + list(grounding.named_modules()))
           if isinstance(m, (stoch, bn))]
    if bad:
        raise SystemExit(f"[FAIL-LOUD] train/eval-dependent modules: {bad[:5]}")
    print("[check] no Dropout/BatchNorm anywhere -> eval() forward is "
          "bit-identical to the trainer's train() forward", flush=True)

    # ---- dataset: the trainer's OWN builder, canonical parity corpus --------
    ds_train, _ = build_datasets(cfg, plan, "cached", [args.cache_dir],
                                 0, 0.6, seed=0)
    assert hasattr(ds_train, "index"), type(ds_train)
    cache_root = sorted(Path(args.cache_dir).glob("*train*"))[-1].name
    assert "e438721ae894" in cache_root, f"PARITY: {cache_root}"
    n_win = len(ds_train)
    g = torch.Generator().manual_seed(args.seed)
    take = args.batches * args.batch_size
    assert take <= n_win, (take, n_win)
    perm = torch.randperm(n_win, generator=g)[:take].tolist()
    epid = [str(ds_train.episodes[ds_train.index[i][0]].episode_id)
            for i in perm]
    print(f"[data] corpus {cache_root}: {len(ds_train.episodes)} episodes, "
          f"{n_win} train windows -> sampling {take} windows from "
          f"{len(set(epid))} episodes", flush=True)
    dl = DataLoader(Subset(ds_train, perm), batch_size=args.batch_size,
                    shuffle=False, drop_last=True, num_workers=args.workers)

    use_amp = args.precision == "bf16" and device == "cuda"
    acc: dict[str, dict[str, list]] = {c: {} for c in CONDITIONS}
    keep_frac: list[float] = []
    v0_true: list[np.ndarray] = []
    xcheck = None
    seen = 0

    for bi, batch in enumerate(dl):
        frames = batch["frames"].to(device)
        fut = batch["future_frames"].to(device)
        pose_last = batch["pose_last"].to(device).float()
        future_poses = batch["future_poses"].to(device).float()
        actions0 = batch["actions"].to(device)
        fut_actions0 = batch["future_actions"].to(device)
        b = frames.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=use_amp):
            # encode ONCE — states cannot depend on actions, so every condition
            # below shares byte-identical latents (the pairing is exact)
            states = model.encode_window(frames)
            fut_states = model.encode_window(fut[:, plan.needed_fut])

            v0 = pose_last[:, 3:4] / 10.0                             # [B, 1]
            v0_true.append(v0.squeeze(1).float().cpu().numpy())
            gg = torch.Generator(device="cpu").manual_seed(
                args.seed * 100003 + bi)
            keep = (torch.rand(b, 1, generator=gg) >= args.p_ego_dropout) \
                .to(device=device, dtype=v0.dtype)
            keep_frac.append(float(keep.mean()))
            pidx = torch.randperm(b, generator=gg).to(device)
            # v0 VALUES per condition (not scales — `perm` is a within-batch
            # shuffle: a wrong but perfectly IN-DISTRIBUTION speed, which
            # separates "the model needs the RIGHT speed" from "0.0 is special")
            vals = {"off": v0, "zero": torch.zeros_like(v0), "on25": v0 * keep,
                    "x2": v0 * 2.0, "perm": v0[pidx]}
            for cond, v in vals.items():
                a = torch.cat([actions0,
                               v.unsqueeze(1).expand(-1, actions0.shape[1], -1)],
                              dim=-1)
                fa = torch.cat(
                    [fut_actions0,
                     v.unsqueeze(1).expand(-1, fut_actions0.shape[1], -1)],
                    dim=-1)
                rows = per_window_terms(model, grounding, plan, states,
                                        fut_states, a, fa, pose_last,
                                        future_poses)
                for k, t in rows.items():
                    acc[cond].setdefault(k, []).append(t.float().cpu().numpy())
                # first batch: cross-check the per-window path against the REAL
                # grounding_losses (the function that wrote the training log)
                if bi == 0:
                    _, _, glog = grounding_losses(
                        grounding, model.predictor, states, fut_states, a, fa,
                        pose_last, future_poses, plan.idx_of, plan.level_cfg,
                        10.0, invdyn_weight=2.0, fwd_weight=1.0,
                        fwd_step_weight=0.5, invdyn_gradscale=0.5)
                    xcheck = xcheck or {}
                    xcheck[cond] = {
                        k: [round(float(np.mean(rows[k].float().cpu().numpy())), 5),
                            round(float(glog[k]), 5)]
                        for k in glog if k in rows}
        seen += b
        if bi % 10 == 0:
            el = time.time() - t_start
            print(f"[run] batch {bi+1}/{args.batches} ({seen} windows) "
                  f"{el:.0f}s", flush=True)

    # ---- assemble ---------------------------------------------------------
    A = {c: {k: np.concatenate(v) for k, v in acc[c].items()} for c in CONDITIONS}
    eid = np.array(epid[:seen])
    # taniteval is not deployed on the pod: expb_ci.py is a byte-copy of the
    # repo's taniteval/taniteval/ci.py, md5 recorded below.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from expb_ci import paired_episode_cluster_bootstrap  # noqa: E402
    except ModuleNotFoundError:              # running from the repo checkout
        from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402
    METRICS = [f"g_{l}_{t}" for l in ("op", "tac", "str")
               for t in ("fwd_ade_m", "mid_de_m")] + \
        ["inv", "inv_ch0", "inv_ch1", "inv_ch2"]

    # ---- harness self-checks: what CANNOT move, must not move -------------
    invariance = {}
    for k in ("g_op_mid_de_m", "g_tac_mid_de_m", "g_str_mid_de_m", "a_hat_v0"):
        d = max(float(np.abs(A[c][k] - A["off"][k]).max())
                for c in CONDITIONS if c != "off")
        invariance[k] = d
        if d != 0.0:
            raise SystemExit(
                f"[FAIL-LOUD] {k} must be EXACTLY mask-invariant (it reads no "
                f"actions) but moved by {d}")
    print(f"[check] mask-invariance exact for {sorted(invariance)}", flush=True)

    res = {
        "experiment": "postmortem-B: ego-dropout zero-fill, mask OFF vs ON",
        "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "arm": run_dir.name, "ckpt": args.ckpt, "ckpt_step": ck_step,
        "arm_trained_with_v2_ego_dropout": args.cfg_ego_dropout,
        "training": False, "optimizer_steps": 0, "checkpoint_written": False,
        "split": "TRAIN batches (canonical parity corpus)",
        "corpus": cache_root, "precision": args.precision,
        "batch_size": args.batch_size, "n_batches": args.batches,
        "n_windows": int(seen), "n_episodes": int(len(set(eid.tolist()))),
        "seed": args.seed, "p_ego_dropout": args.p_ego_dropout,
        "realised_keep_frac": round(float(np.mean(keep_frac)), 4),
        "conditions": {
            "off": "v0 present on every row (ego-dropout mask forced OFF)",
            "zero": "v0 = 0.0 on every row (mask forced ON for all rows)",
            "on25": f"v0 = 0.0 on a random {args.p_ego_dropout:.0%} of rows "
                    "(the training condition)",
            "x2": "v0 doubled (sensitivity readout, OUT of distribution)",
            "perm": "v0 replaced by another row's v0 (within-batch shuffle): a "
                    "WRONG but IN-DISTRIBUTION speed. Separates 'the model "
                    "needs the right speed' from '0.0 is a special lie'."},
        "estimator": "paired_episode_cluster_bootstrap over the sampled TRAIN "
                     f"episodes, B={args.n_boot} (taniteval/ci.py). NOTE: train "
                     "windows are drawn uniformly over the whole corpus, so most "
                     "episodes contribute ~1-2 windows and the cluster bootstrap "
                     "degenerates gracefully toward a window bootstrap here.",
        "harness_checks": {
            "mask_invariance_max_abs_delta": invariance,
            "note": "g_*_mid_de_m and the A5 prediction read NO actions, so "
                    "they are EXACTLY invariant to the mask BY CONSTRUCTION. "
                    "This is a harness check, not evidence about the encoder.",
            "no_dropout_or_batchnorm_modules": True},
        "xcheck_per_window_vs_grounding_losses": xcheck,
        "means": {c: {k: round(float(A[c][k].mean()), 5) for k in METRICS}
                  for c in CONDITIONS},
        "paired": {}, "derived": {},
    }
    import hashlib
    _cim = Path(__file__).resolve().parent / "expb_ci.py"
    res["ci_module_md5"] = (hashlib.md5(_cim.read_bytes()).hexdigest()
                            if _cim.exists() else "taniteval.ci (in-repo)")
    for c in [x for x in CONDITIONS if x != "off"]:
        res["paired"][f"{c}_minus_off"] = {
            k: paired_episode_cluster_bootstrap(A[c][k], A["off"][k], eid,
                                                n_boot=args.n_boot, seed=0)
            for k in METRICS}
    # analytic mixture: E[on] = p*zero + (1-p)*off, zero mask-sampling noise
    p = args.p_ego_dropout
    mix = {k: p * A["zero"][k] + (1 - p) * A["off"][k] for k in METRICS}
    res["means"]["on25_analytic"] = {k: round(float(mix[k].mean()), 5)
                                     for k in METRICS}
    res["paired"]["on25_analytic_minus_off"] = {
        k: paired_episode_cluster_bootstrap(mix[k], A["off"][k], eid,
                                            n_boot=args.n_boot, seed=0)
        for k in METRICS}
    # does the model USE the channel? prediction-vs-truth R2 of the A5 v0 output
    vt = np.concatenate(v0_true)
    ah = A["off"]["a_hat_v0"] if "a_hat_v0" in A["off"] else None
    if ah is not None:
        ss = float(((vt - ah) ** 2).sum())
        res["derived"]["a5_v0_pred_r2_trainbatches"] = round(
            1.0 - ss / float(((vt - vt.mean()) ** 2).sum()), 4)
        res["derived"]["v0_true_mean_scaled"] = round(float(vt.mean()), 4)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"means": res["means"],
                      "keep_frac": res["realised_keep_frac"],
                      "wallclock_s": round(time.time() - t_start, 1)}, indent=2),
          flush=True)
    print(f"EXP_B_DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
