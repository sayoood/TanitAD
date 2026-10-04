#!/usr/bin/env python
"""ddv2_rl_refcv7.py -- WP-RL: DiffusionDriveV2's RL stage (arXiv 2512.07745 + hustvl/DiffusionDriveV2
@1cd12a1) as a POST-TRAINING of refcv7-r101-s0 at step 50,400 (PI, 2026-10-04).

Pre-registration: ``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-ddv2-rl-
posttrain/SPEC_RL.md``. Every constant below is either the paper's / the release's (``ddv2_rl.DDV2``)
or a DEVIATION named there (D-1 ... D-n).

  windows   enumerate the eligible windows of a split (full 6 s future + 9 TTC ticks, every
            agent frame labelled) -> windows_<split>.json (episode index + t; sha12 only for ids)
  diagnose  ZERO training. D1 binding parity vs the DEPLOYED sampler (bitwise) and vs the
            trainer's own forward; D2 the release clamps' binding rates (OFF in our chain, D-4);
            D3 tick_states == the decoder's roll at the slot ticks; D5 reward known values on
            REAL windows (human NC / DAC / TTC / C, identity, an off-road and a collision
            candidate built from the window's own map / agents, EP saturation); D4 cost per
            micro-batch; D6 the cold-start advantage statistics.
  train     one arm, SEGMENTED: ``--arm rl | rloff | rlshuf``; checkpoints every
            ``--ckpt-every`` steps and at segment end, resumes from ``<out>/ckpt_latest.pt`` with
            the SAME data order and RNG streams; yields the GPU lock when another job waits.
  export    the full model state_dict (refcv7's own keys, the generator subset replaced) for the
            refcv7 battery's T1 roll (``roll_seed_r7.py --ckpt``); strict-load verified.
  identity  gate input: the exported checkpoint of a ``--lr 0 --rl-weight 0`` run must equal the
            cold start BITWISE (every tensor), and its forward must equal the cold start's.

(!!) Tier: every number this script prints is T0 (training-side, recorded future, PDMS-SHAPED proxy
reward). It is never PDMS and never driving performance. T1 is the battery roll of an export.
(!!) Clip ids are never written: windows are (episode index, t) + sha12.
Environment (set BEFORE the run, as the route package does): REFCV6_REPO=<tree> REFCV6_KIT=<kit>
PYTHONPATH=<tree>/stack OMP_NUM_THREADS=<n>; on a train split REFCV6_REMAP_OVERRIDES is written and
set by this script itself (see ``_split_env``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

SPLIT_REMAP_THOR = {
    "train": {"--eval-cache": "/home/nvidia/data/refcv6-b1-416x1024-train",
              "--eval-labels": "/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz",
              "--speed-max-sidecar-v6-eval": "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"},
}
EXPECT_STEP = 50400
EXPECT_CKPT_MD5 = "b418d0fc4a92a6848c246a6a7c50207b"
#: scored ego horizon in ticks (SPEC_RL D-6: our plan is 6 s, the reward scores the whole plan)
N_TICKS = 60
#: agent frames the reward reads: t0 .. t0 + N_TICKS + max(TTC offset)
TTC_MAX = 9
AGENT_FRAMES = N_TICKS + 1 + TTC_MAX          # 70


def _split_env(split: str, out_dir: str | None, remap_file: str | None) -> dict:
    """Set REFCV6_REMAP_OVERRIDES for a train-split build BEFORE the loader is imported (it reads
    the env at import). -> the record."""
    if split == "eval":
        if os.environ.get("REFCV6_REMAP_OVERRIDES") or os.environ.get("REFCV7_REMAP_OVERRIDES"):
            raise SystemExit("[rl7] an eval-split run must not carry REMAP_OVERRIDES")
        return {"split": "eval", "overrides": None}
    remap = SPLIT_REMAP_THOR["train"]
    if remap_file:
        remap = json.loads(Path(remap_file).read_text(encoding="utf-8"))
    p = Path(out_dir or ".") / "remap_train.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(remap, indent=1), encoding="utf-8")
    os.environ["REFCV6_REMAP_OVERRIDES"] = str(p)
    return {"split": "train", "overrides": remap, "overrides_file": str(p)}


def log(*a):
    print(*a, flush=True)


def md5_file(path, chunk=1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# =========================================================================== #
# the model, the dataset, the deployed forward                                 #
# =========================================================================== #
class Ctx:
    def __init__(self, a, split: str):
        import torch
        self.torch = torch
        self.a = a
        self.split = split
        _od = getattr(a, "out_dir", None) or (str(Path(a.out).parent) if getattr(a, "out", None) else ".")
        self.env = _split_env(split, _od, getattr(a, "remap_file", None))
        from tanitad.eval import refcv7_loader as L
        self.L = L
        self.tr = L.trainer()
        from tanitad.rl import ddv2_rl as D, ddv2_refcv7 as R, pdm_proxy as P, ddv2_il as IL
        from tanitad.models import kinematic_prior as KP
        import refb_labels
        self.D, self.R, self.P, self.IL, self.KP, self.rb = D, R, P, IL, KP, refb_labels
        self._assert_no_g()
        t0 = time.time()
        if not getattr(a, "skip_md5", False):
            got = md5_file(a.ckpt)
            if got != a.ckpt_md5:
                raise SystemExit(f"[rl7] cold start md5 {got} != {a.ckpt_md5}")
        self.config = L.load_config(a.config)
        self.model, self.cfg, self.args, self.mrec = L.build_model(self.config, a.ckpt,
                                                                   device=a.device, strict=True)
        sd = self.mrec["state_dict"]
        if int(sd.get("step") or -1) != EXPECT_STEP or sd["missing"] or sd["unexpected"]:
            raise SystemExit(f"[rl7] bad load: step {sd.get('step')} missing {sd['missing'][:3]} "
                             f"unexpected {sd['unexpected'][:3]}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dec = self.model.core.decoder
        self.horizons = tuple(int(h) for h in self.cfg.core.trajectory.horizons)
        self.W = int(self.cfg.core.window)
        self.build_s = round(time.time() - t0, 1)
        t1 = time.time()
        self.ds, self.eps, self.drec = L.build_eval_dataset(self.model, self.cfg, self.args,
                                                            self.config,
                                                            with_perception_targets=True)
        self.ds_build_s = round(time.time() - t1, 1)
        self.reader = self.ds.agent_join
        if self.reader is None or not getattr(self.reader, "has_track_ids", False):
            raise SystemExit("[rl7] the dataset's agent join carries no track ids; NC/TTC need tracks")
        if getattr(self.ds, "map_fine_store", None) is None:
            raise SystemExit("[rl7] the dataset has no 10 cm map store: DAC would be dead")
        self.sha_of_eid = dict(getattr(self.ds, "_vis1_sha12", {}) or {})
        self.table = D.diffusers_alphas_cumprod(a.device)
        self.pcfg = P.ProxyConfig(n_ticks=N_TICKS)
        self.dac_rule = str(getattr(a, "dac_rule", "R2") or "R2")
        log(f"[rl7] built: split {split} step {sd['step']} episodes {len(self.eps)} windows "
            f"{len(self.ds)} model {self.build_s}s dataset {self.ds_build_s}s")

    def _assert_no_g(self):
        import tanitad
        here = os.path.normcase(os.path.abspath(tanitad.__file__))
        if here.lower().startswith("g:") or "meine ablage" in here.lower():
            raise SystemExit(f"[rl7] tanitad imported from {here} (G:) -- refusing")

    def sha12_of(self, e_i: int) -> str:
        eid = int(self.ds.episodes[e_i].episode_id)
        s = self.sha_of_eid.get(eid)
        return s if s else f"eid{eid}"

    # ---- the deployed forward (refc_v3_train.compute_losses_v3's input lines, launch tree) -- #
    def forward_deployed(self, batch):
        torch = self.torch
        tr, model, cfg = self.tr, self.model, self.model.cfg
        core = cfg.core
        dev = self.a.device
        frames = tr.frames_to_device(batch["frames"], dev)
        pose_last = batch["pose_last"].to(dev)
        nav_cmd = batch["nav_cmd"].to(dev)
        nav_args = batch["nav_args"].to(dev) if "nav_args" in batch else None
        v_max_ms = v_max_valid = None
        if bool(getattr(cfg, "max_speed_input", False)) or bool(getattr(cfg, "max_speed_onehot_v6", False)):
            v_max_ms = batch["v_max_ms"].to(dev)
            v_max_valid = batch["v_max_valid"].to(dev)
        lan = batch["lan"].to(dev) if "lan" in batch else None
        v0 = pose_last[:, 3]
        steps = core.decoder.diffusion_steps if self.args.mode == "diffusion" else 0
        ego_state = None
        if getattr(cfg, "ego_state_inject", False):
            from tanitad.refs import refc_v3 as _v3
            ego_state = _v3.ego_state_from_batch(
                {"pose_last": pose_last, "actions": batch["actions"]}, device=dev)
        _ag = getattr(core, "agents", None)
        if _ag is not None and getattr(_ag, "enable", False) and getattr(_ag, "oracle", False):
            raise SystemExit("[rl7] --agents oracle is not refcv7")
        if getattr(model.core, "ego_hist", None) is not None:
            ph = batch["pose_hist"].to(dev)
            _rpm = str(getattr(core.decoder, "residual_prior", self.KP.RESIDUAL_PRIOR_OFF))
            _acts = (batch["actions"].to(dev) if (_rpm != self.KP.RESIDUAL_PRIOR_OFF
                                                   and self.KP.needs_actions(_rpm)) else None)
            model.core.set_ego_window(ph, int(ph.shape[1]), actions=_acts)
        _pgrid = _pvalid = None
        _pbr = getattr(model, "_perception", None)
        if _pbr is not None and _pbr.lift is not None:
            _pgrid, _pvalid = model._lift_bank.for_episodes(batch["map_ep"], device=dev)
        elif getattr(model, "_map_hires", None) is not None:
            _pgrid, _pvalid = model._lift_bank_hires.for_episodes(batch["map_ep"], device=dev)
        return model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan, ego_state=ego_state,
                     nav_args=nav_args, v_max_ms=v_max_ms, v_max_valid=v_max_valid, agent_gt=None,
                     perception_grid=_pgrid, perception_valid=_pvalid)

    def capture(self, batch, seed: int):
        torch = self.torch
        torch.manual_seed(int(seed))
        with self.R.capture_sampler_inputs(self.dec) as rec, torch.no_grad():
            out = self.forward_deployed(batch)
        if len(rec) != 1:
            raise RuntimeError(f"expected ONE sampler call per forward, got {len(rec)}")
        return rec[0], out

    # ---- one window's reward inputs ---------------------------------------------------- #
    def window_inputs(self, e_i: int, t: int, batch_row: dict):
        """Human states / route / agent tracks / off-road map / GT waypoints for window (e_i, t)."""
        torch, P, R = self.torch, self.P, self.R
        ep = self.ds.episodes[e_i]
        t0 = int(t) + self.W - 1
        poses = ep.poses
        if t0 + AGENT_FRAMES > int(poses.shape[0]):
            raise RuntimeError("window too close to the clip end (eligibility broken)")
        pose_last = batch_row["pose_last"].float()
        if not torch.equal(pose_last, poses[t0].float()):
            raise RuntimeError("pose_last is not poses[t0] -- the window index contract broke")
        fut = poses[t0 + 1:t0 + AGENT_FRAMES].float()                       # [69, 4]
        human = P.ego_states_from_poses(pose_last[None], fut[None], self.pcfg)[0]   # [61, 4]
        route_st = P.ego_states_from_poses(pose_last[None], fut[None],
                                           P.ProxyConfig(n_ticks=AGENT_FRAMES - 1))[0]
        rcx = route_st[:, 0] + P.PROXY.rear_axle_to_center * torch.cos(route_st[:, 2])
        rcy = route_st[:, 1] + P.PROXY.rear_axle_to_center * torch.sin(route_st[:, 2])
        tracks = R.tracks_from_join(self.reader, int(ep.episode_id), t0, poses[t0:t0 + AGENT_FRAMES],
                                    AGENT_FRAMES, self.pcfg)
        if tracks is None:
            raise RuntimeError("unlabelled agent frame inside an eligible window")
        off = None
        if bool(batch_row.get("map_fine_label", torch.tensor(False))):
            off = offroad_rule(R, batch_row["map_fine"].to(self.a.device), self.dac_rule)
        gt = self.rb.waypoint_targets(pose_last[None], batch_row["future_poses_ext"].float()[None],
                                      self.horizons)[0]
        return {"human": human, "route": torch.stack([rcx, rcy], -1), "tracks": tracks, "off": off,
                "gt": gt, "sha12": self.sha12_of(e_i), "t0": t0}


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _row(batch, i):
    return {k: (v[i] if hasattr(v, "shape") and v.dim() > 0 and v.shape[0] == batch["pose_last"].shape[0]
                else v) for k, v in batch.items()}


def eligible_windows(ctx, stride: int = 1):
    """(e_i, t) of every window with a full scored future and every agent frame labelled."""
    by_ep = {}
    for wi, (e_i, t) in enumerate(ctx.ds.index):
        by_ep.setdefault(int(e_i), []).append(int(t))
    out, drop = [], {"future": 0, "agents": 0}
    for e_i in sorted(by_ep):
        ep = ctx.ds.episodes[e_i]
        eid = int(ep.episode_id)
        n = int(ep.poses.shape[0])
        lab = [ctx.reader.lookup(eid, f) is not None for f in range(n)]
        # run[f] = number of consecutive labelled frames starting at f
        run = [0] * (n + 1)
        for f in range(n - 1, -1, -1):
            run[f] = run[f + 1] + 1 if lab[f] else 0
        for t in sorted(by_ep[e_i]):
            t0 = t + ctx.W - 1
            if t0 + AGENT_FRAMES > n:
                drop["future"] += 1
                continue
            if run[t0] < AGENT_FRAMES:
                drop["agents"] += 1
                continue
            out.append((e_i, t))
    if stride > 1:
        out = out[::stride]
    return out, drop


def window_index_map(ctx):
    return {(int(e), int(t)): i for i, (e, t) in enumerate(ctx.ds.index)}


def load_windows(path):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    w = [tuple(x) for x in rec["windows"]]
    if sha256_text(json.dumps(w)) != rec["digest"]:
        raise SystemExit(f"[rl7] windows file {path}: digest mismatch")
    return w, rec


# =========================================================================== #
# windows                                                                      #
# =========================================================================== #
def cmd_windows(a):
    ctx = Ctx(a, a.split)
    w, drop = eligible_windows(ctx, a.stride)
    rec = {"split": a.split, "n_dataset_windows": len(ctx.ds), "n_eligible": len(w), "dropped": drop,
           "agent_frames": AGENT_FRAMES, "n_ticks": N_TICKS, "stride": a.stride,
           "windows": [list(x) for x in w], "digest": sha256_text(json.dumps(w)),
           "n_episodes": len({e for e, _ in w}),
           "sha12": sorted({ctx.sha12_of(e) for e, _ in w}),
           "_evidence_class": "MEASURED (ours)", "dataset_record_keys": sorted(ctx.drec)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
    log(f"[windows] {a.split}: {len(w)} eligible of {len(ctx.ds)} over {rec['n_episodes']} episodes; "
        f"dropped {drop}; digest {rec['digest'][:16]} -> {a.out}")


# =========================================================================== #
# the RL step (exact release arithmetic over the FULL step batch, micro-batched)   #
# =========================================================================== #
class Arm:
    KINDS = ("rl", "rloff", "rlshuf")

    def __init__(self, kind, rl_weight, groups, il_form, grad_clip, clip_sample):
        if kind not in self.KINDS:
            raise ValueError(kind)
        self.kind, self.rl_weight, self.groups = kind, float(rl_weight), int(groups)
        self.il_form, self.grad_clip, self.clip_sample = il_form, grad_clip, bool(clip_sample)
        if kind == "rloff" and self.rl_weight != 0.0:
            raise SystemExit("[rl7] the rloff arm is DEFINED by rl_weight == 0")

    def to_dict(self):
        return dict(kind=self.kind, rl_weight=self.rl_weight, groups=self.groups,
                    il_form=self.il_form, grad_clip=self.grad_clip, clip_sample=self.clip_sample)


def phase_a(ctx, arm, mb_batch, mb_windows, gen, fwd_seed, consts):
    """No-grad: capture, rollout, reward for ONE micro-batch. -> record."""
    torch, D, R, IL = ctx.torch, ctx.D, ctx.R, ctx.IL
    tm = {}
    t = time.time()
    inp, out = ctx.capture(mb_batch, fwd_seed)
    _sync()
    tm["capture_s"] = time.time() - t
    m = inp.bank.shape[0]
    n = int(ctx.dec.anchors.shape[0])
    t = time.time()
    start, _ = D.truncated_start(R.anchor_state(ctx.dec, m), arm.groups, ctx.table,
                                 trunc_t=consts.trunc_t, generator=gen)
    fn = R.make_x0_fn(ctx.dec, inp)
    with torch.no_grad():
        roll = D.rollout_chain(fn, start, ctx.table, eta=consts.eta, consts=consts, generator=gen,
                               keep_x0=True)
    _sync()
    tm["rollout_s"] = time.time() - t
    t = time.time()
    with torch.no_grad():
        states = R.tick_states(ctx.dec, roll["chain"][..., -1], inp, N_TICKS)       # [m, M, 61, 4]
    rewards, fails, hum, subs, gts, n_dac_dead, n_tracks = [], [], [], [], [], 0, []
    for j, (e_i, tt) in enumerate(mb_windows):
        wi = ctx.window_inputs(e_i, tt, _row(mb_batch, j))
        if wi["off"] is None:
            n_dac_dead += 1
        dev = states.device
        tracks = wi["tracks"].to(dev)
        sc = R.score_window(states[j], wi["human"].to(dev), tracks, wi["route"].to(dev), wi["off"],
                            ctx.pcfg)
        rewards.append(sc["pdms"])
        fails.append(sc["constraint_fail"])
        hum.append(sc["human"])
        subs.append({k: sc[k] for k in ("nc", "dac", "ep", "ttc", "comfort", "spd")})
        gts.append(wi["gt"].to(dev))
        n_tracks.append(sc["n_tracks_scored"])
    _sync()
    tm["reward_s"] = time.time() - t
    gt = torch.stack(gts)
    a_star = IL.matched_anchor_index(inp.bank, gt)
    return {"inp": inp, "chain": roll["chain"], "labels": roll["labels"], "x0": roll["x0"],
            "reward": torch.stack(rewards).view(m, arm.groups, n),
            "fail": torch.stack(fails).view(m, arm.groups, n),
            "reward_gt": torch.tensor([h["pdms"] for h in hum], device=states.device),
            "human": hum, "subs": subs, "gt": gt, "a_star": a_star, "n_dac_dead": n_dac_dead,
            "n_tracks": n_tracks, "timing": tm, "deployed_traj": out["traj"].detach()}


def _group_of(name: str) -> str:
    """'layers.3.cross.in_proj_weight' -> 'layers.3'; 'cascade.control_heads.2.weight' ->
    'cascade.control_heads.2'; 'traj_proj.weight' -> 'traj_proj'."""
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p.isdigit():
            return ".".join(parts[:i + 1])
    return parts[0]


def il_paths(ctx, inp, stage_x0s):
    return [ctx.R.roll_fan(ctx.dec, x, inp) for x in stage_x0s]


def rl_step(ctx, arm, micro, windows_step, gen, shuf_gen, step, seed, consts, params,
            opt=None, apply=True):
    """One optimiser step over the whole step batch (len(windows_step) windows)."""
    torch, D, R, IL = ctx.torch, ctx.D, ctx.R, ctx.IL
    recs = []
    for k, (mb_batch, mb_w) in enumerate(micro):
        recs.append(phase_a(ctx, arm, mb_batch, mb_w, gen,
                            fwd_seed=1_000_003 * (seed + 1) + 64 * step + k, consts=consts))
    reward = torch.cat([r["reward"] for r in recs])
    fail = torch.cat([r["fail"] for r in recs])
    reward_gt = torch.cat([r["reward_gt"] for r in recs])
    b = reward.shape[0]
    perm = None
    if arm.kind == "rlshuf":
        perm = R.shuffle_permutation(b, shuf_gen).to(reward.device)
        reward, fail, reward_gt = reward[perm], fail[perm], reward_gt[perm]
    adv_out = D.intra_anchor_advantage(reward, reward_gt, fail, consts=consts)
    adv = adv_out["advantage"].reshape(b, -1)
    T = len(recs[0]["labels"])
    w = D.step_loss_weights(adv, T, consts=consts)
    coef_rl = w["coef_rl"] * arm.rl_weight
    n = int(ctx.dec.anchors.shape[0])
    for p in params:
        p.grad = None
    t = time.time()
    loss_total, il_total, rl_part = 0.0, 0.0, 0.0
    row0 = 0
    il_all = 0.0
    for r in recs:
        m = r["chain"].shape[0]
        stages: list = []
        fn = R.make_x0_fn(ctx.dec, r["inp"], stages=stages)
        cr = coef_rl[row0:row0 + m]
        frac = m / b
        for i in range(T):
            lp, _x0 = D.chain_step_logprob(fn, r["chain"], i, ctx.table, labels=r["labels"],
                                           eta=consts.eta, consts=consts)
            sx0 = R.stage_x0_list(stages)
            paths = il_paths(ctx, r["inp"], sx0)
            if arm.il_form == "matched_anchor":
                il_i = torch.stack([IL.matched_anchor_il(pth, r["gt"], r["a_star"], n)
                                    for pth in paths]).mean()
            elif arm.il_form == "release_all_modes":
                il_i = torch.stack([IL.all_modes_il(pth, r["gt"]) for pth in paths]).mean()
            else:
                raise SystemExit(f"[rl7] il_form {arm.il_form}")
            li, rl_i = R.microbatch_step_loss(lp, cr[..., i], w["il_coef"], il_i, frac)
            li.backward()
            loss_total += float(li)
            rl_part += float(rl_i)
            il_total += float(il_i) * frac / T
            with torch.no_grad():
                il_all += float(torch.stack([IL.all_modes_il(pth, r["gt"]) for pth in paths]).mean()) * frac / T
        row0 += m
    _sync()
    grad_s = time.time() - t
    gst = IL.clip_gradients(params, IL.IlSettings(form=arm.il_form if arm.il_form != "release_all_modes"
                                                  else IL.RELEASE_IL_FORM, grad_clip=arm.grad_clip))
    groups_with_grad = sorted({_group_of(nm) for nm, p in ctx.named
                               if p.grad is not None and float(p.grad.abs().sum()) > 0})
    if apply:
        opt.step()
    for p in params:
        p.grad = None
    # ---- telemetry ----------------------------------------------------------------- #
    with torch.no_grad():
        last = torch.cat([r["chain"][..., -1] for r in recs])
        spread = IL.fan_endpoint_spread(torch.cat([ctx.R.roll_fan(ctx.dec, r["chain"][..., -1], r["inp"])
                                                   for r in recs]), n)
        clamp = {}
        x0s = torch.cat([r["x0"] for r in recs])
        for c, nm in ((0, "a_lon"), (1, "a_lat")):
            clamp[f"x0_out_of_box_frac_{nm}"] = float((x0s[..., c, :].abs() > 1).float().mean())
            clamp[f"x0_out_of_box_frac_final_{nm}"] = float((x0s[..., c, -1].abs() > 1).float().mean())
    subs = {k: float(torch.stack([s[k] for r in recs for s in r["subs"]]).mean())
            for k in ("nc", "dac", "ep", "ttc", "comfort", "spd")}
    hum = [h for r in recs for h in r["human"]]
    tm = {k: round(sum(r["timing"][k] for r in recs), 3) for k in recs[0]["timing"]}
    tm["grad_s"] = round(grad_s, 3)
    return {"loss": loss_total, "rl_part": rl_part, "il_m": il_total, "il_all_modes_m": il_all,
            "rl_weight": arm.rl_weight, "coef_rl_abs_sum": float(coef_rl.abs().sum()),
            "il_coef": float(w["il_coef"]), "grad_norm": gst["grad_norm"],
            "grad_clipped": gst["grad_clipped"], "grad_groups": groups_with_grad,
            "reward_mean": float(reward.mean()), "reward_best_mean": float(reward.flatten(1).amax(1).mean()),
            "reward_group_std_pos_frac": float((reward.std(dim=1) > 0).float().mean()),
            "human_pdms_mean": float(reward_gt.mean()),
            "human_nc1_frac": sum(1 for h in hum if h["nc"] == 1.0) / len(hum),
            "human_dac1_frac": sum(1 for h in hum if h["dac"] == 1.0) / len(hum),
            "human_ttc1_frac": sum(1 for h in hum if h["ttc"] == 1.0) / len(hum),
            **{f"cand_{k}_mean": v for k, v in subs.items()},
            "frac_positive_before_bar": float(adv_out["frac_positive_before_bar"]),
            "frac_admitted_by_bar": float(adv_out["frac_admitted_by_bar"]),
            "frac_positive_after_bar": float(adv_out["frac_positive_after_bar"]),
            "frac_constraint_fail": float(adv_out["frac_constraint_fail"]),
            "rows_with_positive": int((w["il_weight_b"] < 0.5).sum()),
            "n_windows": b, "n_dac_dead": sum(r["n_dac_dead"] for r in recs),
            "n_tracks_mean": sum(sum(r["n_tracks"]) for r in recs) / b,
            "chain_endpoint_spread_m": spread, **clamp,
            "shuffle_perm": None if perm is None else perm.tolist(), "timing": tm}


# =========================================================================== #
# data stream                                                                   #
# =========================================================================== #
def micro_batches(ctx, windows, start_step, end_step, B, m, workers):
    """Yield (step, [(mb_batch, mb_windows), ...]) for steps [start_step, end_step) in the FIXED
    order ``windows`` (already permuted). The DataLoader reads exactly those windows in order."""
    torch = ctx.torch
    imap = window_index_map(ctx)
    sel = windows[start_step * B:end_step * B]
    idx = [imap[(int(e), int(t))] for e, t in sel]
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ctx.ds, idx), batch_size=m, shuffle=False,
                                     num_workers=workers, prefetch_factor=(2 if workers > 0 else None),
                                     persistent_workers=False)
    it = iter(dl)
    per = B // m
    for s in range(start_step, end_step):
        mbs = []
        for k in range(per):
            batch = next(it)
            off = (s - start_step) * B + k * m
            mbs.append((batch, sel[off:off + m]))
        yield s, mbs


# =========================================================================== #
# lock yield                                                                    #
# =========================================================================== #
def other_lock_users(lock_path: str) -> list:
    """PIDs (not in this process's SESSION) holding or waiting on ``lock_path``."""
    if not lock_path or not os.path.isdir("/proc"):
        return []
    me = os.getsid(0)
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        hit = False
        for fd in fds:
            try:
                if os.readlink(f"/proc/{pid}/fd/{fd}") == lock_path:
                    hit = True
                    break
            except OSError:
                continue
        if not hit:
            continue
        try:
            with open(f"/proc/{pid}/stat") as fh:
                sid = int(fh.read().rsplit(")", 1)[1].split()[4])
        except (OSError, ValueError, IndexError):
            continue
        if sid != me:
            out.append(int(pid))
    return out


def lock_holders(lock_path: str) -> list:
    """PIDs HOLDING (not waiting on) the flock on ``lock_path``, read-only from /proc/locks
    (FLOCK rows whose inode is the lock file's; '->' rows are blocked waiters)."""
    try:
        ino = os.stat(lock_path).st_ino
        rows = open("/proc/locks").read().splitlines()
    except OSError:
        return []
    out = []
    for r in rows:
        f = r.split()
        if len(f) < 6 or "->" in f[:2] or "FLOCK" not in f:
            continue
        i = f.index("FLOCK")
        try:
            pid, dev = int(f[i + 3]), f[i + 4]
        except (IndexError, ValueError):
            continue
        if dev.rsplit(":", 1)[-1] == str(ino):
            out.append(pid)
    return out


# =========================================================================== #
# train                                                                         #
# =========================================================================== #
def lr_at(step, total, lr, min_lr=1e-6, warmup_frac=0.10):
    """P suppl. sec. 7 'cosine learning rate schedule with a 10 % linear warmup', over STEPS (SPEC A15:
    the release steps per epoch, which makes its warmup a no-op -- flagged)."""
    wu = max(1, int(round(warmup_frac * total)))
    if step < wu:
        return lr * (step + 1) / wu
    prog = (step - wu) / max(1, total - wu)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def _run_record(a, arm, consts, n_train, named, windows_rec):
    return {"wp": "WP-RL", "arm": arm.to_dict(), "seed": a.seed, "steps": a.steps, "batch": a.batch,
            "micro": a.micro, "lr": a.lr, "weight_decay": a.weight_decay,
            "schedule": "linear warmup 10% + cosine to 1e-6 over steps (P suppl. 7)",
            "consts": consts.to_dict(), "n_ticks": N_TICKS, "agent_frames": AGENT_FRAMES,
            "proxy": dict(n_ticks=N_TICKS, w_ep=5.0, w_ttc=5.0, w_c=2.0, w_spd=0.0),
            "dac_ladder_rule": a.dac_rule,
            "trainable": {"prefixes": list(named[0]), "n_params": n_train},
            "windows_digest": windows_rec["digest"], "n_windows_eligible": windows_rec["n_eligible"],
            "ckpt": a.ckpt, "ckpt_md5": a.ckpt_md5, "config": a.config,
            "_tier": "T0 training-side", "_evidence_class": "MEASURED (ours)"}


def cmd_train(a, ctx=None):
    """One arm. ``ctx`` given (the in-process ``smoke``): the model and dataset are reused, the
    generator's parameters are first RESTORED to the cold start (``ctx.pristine``), and the GPU lock
    is the caller's (``a.inproc_lock`` must be empty)."""
    import torch
    if ctx is None:
        ctx = Ctx(a, "train")
    elif getattr(ctx, "pristine", None) is not None:
        with torch.no_grad():
            for n, p in ctx.dec.named_parameters():
                if n in ctx.pristine:
                    p.copy_(ctx.pristine[n])
    D, R, IL = ctx.D, ctx.R, ctx.IL
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.batch % a.micro:
        raise SystemExit("[rl7] --batch must be a multiple of --micro")
    windows, wrec = load_windows(a.windows)
    consts = D.DDV2Constants(clip_sample=bool(a.clip_sample))
    arm = Arm(a.arm, a.rl_weight, a.groups, a.il_form, None if a.grad_clip == 0 else a.grad_clip,
              a.clip_sample)
    named = R.trainable_named(ctx.dec)
    ctx.named = named
    for _, p in named:
        p.requires_grad_(True)
    params = [p for _, p in named]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    run = _run_record(a, arm, consts, n_train, ([n for n, _ in named],), wrec)
    run["dac_rule"] = vars(R.DAC_RULE) if hasattr(R.DAC_RULE, "__dict__") else str(R.DAC_RULE)
    run["trainable"] = {"prefixes": list(R.TRAINABLE_PREFIXES), "n_params": n_train,
                        "n_tensors": len(named)}
    run_hash = sha256_text(json.dumps({k: v for k, v in run.items() if k not in ("ckpt",)},
                                      sort_keys=True, default=str))
    # ---- order + RNG ---------------------------------------------------------------- #
    order_gen = torch.Generator().manual_seed(10_000 + a.seed)
    perm = torch.randperm(len(windows), generator=order_gen).tolist()
    if a.steps * a.batch > len(perm):
        raise SystemExit(f"[rl7] {a.steps} x {a.batch} > {len(perm)} eligible windows (no epoch wrap)")
    order = [windows[i] for i in perm]
    gen = torch.Generator(device=a.device).manual_seed(20_000 + a.seed)
    shuf_gen = torch.Generator().manual_seed(30_000 + a.seed)
    base = {n: p.detach().clone() for n, p in named}
    start = 0
    ck = out / "ckpt_latest.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if st["run_hash"] != run_hash:
            raise SystemExit(f"[rl7] resume refused: run hash {st['run_hash'][:12]} != {run_hash[:12]} "
                             f"(a flag changed between segments)")
        sd = dict(named)
        for n, v in st["trainable"].items():
            sd[n].data.copy_(v.to(sd[n].device))
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen_state"])
        shuf_gen.set_state(st["shuf_gen_state"])
        start = int(st["step"])
        base = {n: v.to(a.device) for n, v in st["base"].items()}
        log(f"[train] RESUMED at step {start} from {ck}")
    else:
        (out / "run.json").write_text(json.dumps({**run, "run_hash": run_hash}, indent=1, default=str),
                                      encoding="utf-8")
    end = a.steps if a.max_steps_this_segment <= 0 else min(a.steps, start + a.max_steps_this_segment)
    log(f"[train] arm {arm.kind} seed {a.seed} steps {start}->{end}/{a.steps} batch {a.batch} micro "
        f"{a.micro} trainable {n_train:,} run_hash {run_hash[:12]}")
    mf = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def save(step_done, tag="latest"):
        st = {"trainable": {n: p.detach().cpu() for n, p in named}, "opt": opt.state_dict(),
              "gen_state": gen.get_state(), "shuf_gen_state": shuf_gen.get_state(), "step": step_done,
              "run_hash": run_hash, "base": {n: v.cpu() for n, v in base.items()},
              "windows_digest": wrec["digest"], "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = out / f"ckpt_{tag}.pt.tmp"
        torch.save(st, tmp)
        os.replace(tmp, out / f"ckpt_{tag}.pt")

    def segment_marker(done, reason, t0):
        m = {"step_done": done, "steps": a.steps, "reason": reason, "run_hash": run_hash,
             "wall_s": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / "SEGMENT.json").write_text(json.dumps(m), encoding="utf-8")
        with open(out / "segments.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(m) + "\n")
        return m

    # The stream is created and PRIMED (DataLoader workers forked) BEFORE the lock file is opened:
    # a worker forked while the lock fd is open would inherit it and hold the lock after the
    # parent releases (the 2026-09-02 supervisor trap, in a DataLoader costume).
    stream = micro_batches(ctx, order, start, end, a.batch, a.micro, a.workers)
    first = next(stream, None)
    lock = InprocLock(a.inproc_lock) if a.inproc_lock else None
    if lock is not None:
        waited = lock.acquire()
        log(f"[train] GPU lock acquired after {waited:.0f}s")
    t_seg = time.time()
    t_run = time.time()
    stop_reason = "done"
    step = start - 1

    def batches():
        if first is not None:
            yield first
        yield from stream

    for step, micro in batches():
        lr = lr_at(step, a.steps, a.lr)
        for gp in opt.param_groups:
            gp["lr"] = lr
        stats = rl_step(ctx, arm, micro, None, gen, shuf_gen, step, a.seed, consts, params, opt=opt,
                        apply=True)
        with torch.no_grad():
            dnorm = float(torch.sqrt(sum(((p - base[n]) ** 2).sum() for n, p in named)))
        stats.update({"step": step, "lr": lr, "param_delta_norm": dnorm,
                      "wall_s": round(time.time() - t_run, 1),
                      "windows": [[ctx.sha12_of(e), int(t)] for mbw in micro for e, t in mbw[1]],
                      "finite": bool(math.isfinite(stats["loss"]) and math.isfinite(stats["grad_norm"]))})
        mf.write(json.dumps(stats) + "\n")
        mf.flush()
        if not stats["finite"]:
            save(step, "nonfinite")
            raise SystemExit(f"[train] non-finite at step {step}")
        if step % a.log_every == 0 or step == end - 1:
            log(f"[train {arm.kind} s{a.seed}] step {step} lr {lr:.2e} loss {stats['loss']:.4f} "
                f"il {stats['il_m']:.3f} m R {stats['reward_mean']:.3f} H {stats['human_pdms_mean']:.3f} "
                f"pos {stats['frac_positive_after_bar']:.4f} fail {stats['frac_constraint_fail']:.3f} "
                f"dac {stats['cand_dac_mean']:.3f} g {stats['grad_norm']:.2f} d {dnorm:.4f} "
                f"spread {stats['chain_endpoint_spread_m']:.2f} t {stats['timing']}")
        done = step + 1
        if a.ckpt_every > 0 and done % a.ckpt_every == 0:
            save(done)
        if (out / "STOP").exists():
            stop_reason = "stop-file"
            break
        if done >= end:
            break
        seg_min = (time.time() - t_seg) / 60.0
        lk = a.inproc_lock or a.yield_lock
        users = other_lock_users(lk) if lk else []
        want_yield = bool(users) and seg_min >= a.min_segment_minutes
        if lock is None:
            if a.segment_minutes > 0 and seg_min >= a.segment_minutes:
                stop_reason = "segment-time"
                break
            if want_yield:
                stop_reason = f"yield:{users}"
                break
            continue
        # ---- in-process lock: checkpoint, release, wait, re-acquire (no rebuild) ---------- #
        if want_yield or (a.segment_minutes > 0 and seg_min >= a.segment_minutes):
            save(done)
            segment_marker(done, f"yield:{users}" if want_yield else "segment-time", t_seg)
            _sync()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            lock.release()
            log(f"[train] lock RELEASED at step {done} "
                f"({'yield to ' + str(users) if want_yield else 'segment end'})")
            # Hand-over: wait until the job we yielded to HOLDS the lock (or nobody wants it), and
            # only then queue behind it -- re-queueing at once could win the very race we yielded.
            # A descriptor held open by a process that never TAKES the lock (an orphaned worker that
            # inherited it) would keep 'others' non-empty forever: after 120 s with NO holder the
            # trainer queues anyway (the blocking acquire is still a fair place in the queue).
            time.sleep(2)
            t_free = None
            while True:
                if (out / "STOP").exists():
                    break
                if not other_lock_users(a.inproc_lock):
                    break
                if lock_holders(a.inproc_lock):
                    break
                t_free = t_free or time.time()
                if time.time() - t_free > 120.0:
                    log("[train] lock free for 120 s while 'others' hold descriptors -- queueing anyway")
                    break
                time.sleep(2)
            if (out / "STOP").exists():
                stop_reason = "stop-file"
                lock = None
                break
            waited = lock.acquire()
            log(f"[train] lock RE-ACQUIRED after {waited:.0f}s at step {done}")
            t_seg = time.time()
    done = step + 1
    save(done)
    mf.close()
    if lock is not None:
        lock.release()
    marker = segment_marker(done, stop_reason, t_seg)
    if done >= a.steps:
        (out / "DONE.json").write_text(json.dumps(marker), encoding="utf-8")
    log(f"[train] end: {marker}")
    if stop_reason.startswith("yield"):
        sys.exit(75)


def cmd_smoke(a):
    """The stage-2 GPU smoke in ONE process and ONE lock acquisition (gate inputs, SPEC_RL sec. 8):
    RL segmented 2 + 2 (resumed from ckpt_latest.pt) and uninterrupted 4 steps, RLOFF 2, RL-SHUF 2
    (batch 8, micro 4), and the cost at the registered batch (32, micro 8, 3 steps). The model and
    the 4,369-clip dataset are built ONCE on the CPU side, then the lock is taken; the generator is
    restored to the cold start before every run. A cross-PROCESS resume is exercised separately
    (the CPU dry run and every real segment boundary)."""
    import copy as _copy
    import torch
    ctx = Ctx(a, "train")
    ctx.pristine = {n: p.detach().clone() for n, p in ctx.dec.named_parameters()}
    base = _copy.copy(a)
    base.inproc_lock = ""
    base.yield_lock = ""
    base.ckpt_every = 0
    base.segment_minutes = 0.0
    base.min_segment_minutes = 5.0
    base.log_every = 1
    root = Path(a.out_dir)
    runs = [("rl_seg", dict(arm="rl", rl_weight=1.0, steps=4, batch=8, micro=4, max_steps_this_segment=2)),
            ("rl_seg", dict(arm="rl", rl_weight=1.0, steps=4, batch=8, micro=4, max_steps_this_segment=0)),
            ("rl_full", dict(arm="rl", rl_weight=1.0, steps=4, batch=8, micro=4, max_steps_this_segment=0)),
            ("rloff", dict(arm="rloff", rl_weight=0.0, steps=2, batch=8, micro=4, max_steps_this_segment=0)),
            ("rlshuf", dict(arm="rlshuf", rl_weight=1.0, steps=2, batch=8, micro=4, max_steps_this_segment=0)),
            ("timing32", dict(arm="rl", rl_weight=1.0, steps=3, batch=32, micro=8, max_steps_this_segment=0))]
    lock = InprocLock(a.inproc_lock) if a.inproc_lock else None
    if lock is not None:
        log(f"[smoke] GPU lock acquired after {lock.acquire():.0f}s")
    t0 = time.time()
    try:
        for name, kw in runs:
            r = _copy.copy(base)
            for k, v in kw.items():
                setattr(r, k, v)
            r.out_dir = str(root / name)
            log(f"[smoke] run {name} {kw}")
            t1 = time.time()
            cmd_train(r, ctx=ctx)
            log(f"[smoke] run {name} done in {time.time() - t1:.1f}s")
    finally:
        if lock is not None:
            lock.release()
    with torch.no_grad():
        for n, p in ctx.dec.named_parameters():
            p.copy_(ctx.pristine[n])
    log(f"[smoke] ALL RUNS DONE in {time.time() - t0:.1f}s (lock held throughout)")


class InprocLock:
    """flock(2) on the shared Thor GPU lock, from INSIDE the trainer, so a segment boundary
    releases the GPU without rebuilding the model and the 4,369-clip dataset. Compatible with
    util-linux ``flock`` (the same BSD lock). LOCK_UN is explicit before close: a forked child
    holding a copy of the descriptor would otherwise keep the lock after ``close``."""

    def __init__(self, path: str):
        self.path = path
        self.fd = None

    def acquire(self) -> float:
        """BLOCKING ``LOCK_EX`` -- the same kernel queue every ``flock <lock> cmd`` waiter sits in.
        MEASURED 2026-10-04: a 15 s ``LOCK_NB`` poll lost every hand-over to the kernel-blocked
        ``flock`` waiters of four queued jobs (35 min, 0 acquisitions), i.e. polling is not a fair
        place in the queue, it is the back of it."""
        import fcntl
        t0 = time.time()
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"[lock] {self.path} is held -- blocking in the kernel queue")
            fcntl.flock(fd, fcntl.LOCK_EX)
        self.fd = fd
        return time.time() - t0

    def release(self):
        import fcntl
        if self.fd is None:
            return
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        self.fd = None


# =========================================================================== #
# export                                                                        #
# =========================================================================== #
def cmd_export(a):
    import torch
    run_dir = Path(a.run_dir)
    st = torch.load(run_dir / f"ckpt_{a.tag}.pt", map_location="cpu", weights_only=False)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    pref = "core.decoder."
    n_rep = 0
    for n, v in st["trainable"].items():
        k = pref + n
        if k not in sd or tuple(sd[k].shape) != tuple(v.shape):
            raise SystemExit(f"[export] {k} missing or shape mismatch")
        sd[k] = v.to(sd[k].dtype)
        n_rep += 1
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "step": EXPECT_STEP, "ddv2_rl": {**run, "rl_step": st["step"]}}, outp)
    cfgj = json.loads(Path(a.config).read_text(encoding="utf-8"))
    cfgj["ddv2_rl_posttrain"] = {**run, "rl_step": st["step"]}
    Path(str(outp) + ".config.json").write_text(json.dumps(cfgj, indent=1), encoding="utf-8")
    log(f"[export] {n_rep} tensors replaced (rl step {st['step']}) -> {outp} md5 {md5_file(outp)}")


def cmd_identity(a):
    """Every tensor of an export equals the cold start's (torch.equal), keyed and counted."""
    import torch
    x = torch.load(a.export, map_location="cpu", weights_only=False)["model"]
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = c["model"] if "model" in c else c
    keys = sorted(set(x) | set(c))
    diff = [k for k in keys if k not in x or k not in c or not torch.equal(x[k], c[k])]
    rec = {"n_tensors": len(keys), "n_equal": len(keys) - len(diff), "differ": diff[:50],
           "verdict": "IDENTICAL" if not diff else "DIFFERENT"}
    log(json.dumps(rec))
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if a.expect == "identical" and diff:
        sys.exit(2)
    if a.expect == "different" and not diff:
        sys.exit(2)


# =========================================================================== #
# diagnose                                                                      #
# =========================================================================== #
def cmd_diagnose(a):
    import torch
    ctx = Ctx(a, a.split)
    D, R, P, IL = ctx.D, ctx.R, ctx.P, ctx.IL
    windows, wrec = load_windows(a.windows)
    lock = InprocLock(a.inproc_lock) if a.inproc_lock else None
    if lock is not None:
        log(f"[diag] GPU lock acquired after {lock.acquire():.0f}s")
    g = torch.Generator().manual_seed(0)
    pick = [windows[int(i)] for i in torch.randperm(len(windows), generator=g)[:a.n_windows]]
    imap = window_index_map(ctx)
    rec = {"_tier": "T0 zero-training diagnostics", "_evidence_class": "MEASURED (ours)",
           "split": a.split, "n_windows": len(pick), "windows_digest": wrec["digest"],
           "build_s": ctx.build_s, "dataset_build_s": ctx.ds_build_s}
    d1, d1c, d2, d3, d5 = [], [], [], [], []
    consts = D.DDV2Constants(clip_sample=False)
    for k, (e_i, t) in enumerate(pick):
        batch = torch.utils.data.default_collate([ctx.ds[imap[(e_i, t)]]])
        inp, out = ctx.capture(batch, seed=1000 + k)
        fan_ref, u0_ref = inp.out[0], inp.out[1]
        eps = inp.replay_eps(ctx.dec)
        with torch.no_grad():
            fan, u0 = R.native_sample(ctx.dec, inp, eps)
            # D1c: the trainer's own forward captures the SAME sampler inputs
            torch.manual_seed(1000 + k)
            with R.capture_sampler_inputs(ctx.dec) as rec2:
                ctx.tr.compute_losses_v3(ctx.model, batch, a.device, mode=ctx.args.mode,
                                         ablate_frames=False)
        same_inputs = (len(rec2) == 1 and all(
            torch.equal(getattr(inp, f), getattr(rec2[0], f)) for f in ("kv", "cond", "bank"))
            and all((x is None and y is None) or torch.equal(x, y) for x, y in
                    ((inp.agents, rec2[0].agents), (inp.bev, rec2[0].bev), (inp.agent_pad, rec2[0].agent_pad)))
            and all(torch.equal(x, y) for x, y in zip(inp.prior, rec2[0].prior)))
        d1.append({"fan_bitwise": bool(torch.equal(fan, fan_ref)), "u0_bitwise": bool(torch.equal(u0, u0_ref)),
                   "fan_max_abs": float((fan - fan_ref).abs().max()),
                   "traj_in_fan": bool((fan_ref[0] - out["traj"][0][None]).abs().amax(dim=(-1, -2)).min() < 1e-4)})
        d1c.append(bool(same_inputs))
        # D1b: G-tiling exactness of the chain's pass
        x0n = R.anchor_state(ctx.dec, 1)
        fn = R.make_x0_fn(ctx.dec, inp)
        with torch.no_grad():
            one = fn(x0n, 10)
            four = fn(D.tile_groups(x0n, 4), 10)
        n = x0n.shape[1]
        tile_dev = float(max((four[:, gi * n:(gi + 1) * n] - one).abs().max() for gi in range(4)))
        # D2 / D3: the release chain at eta=1 from the cold start, clamps OFF (D-4), its clamp rates
        gen = torch.Generator(device=a.device).manual_seed(77 + k)
        start, _ = D.truncated_start(R.anchor_state(ctx.dec, 1), 4, ctx.table, trunc_t=8, generator=gen)
        with torch.no_grad():
            roll = D.rollout_chain(fn, start, ctx.table, eta=1.0, consts=consts, generator=gen, keep_x0=True)
            last = roll["chain"][..., -1]
            st = R.tick_states(ctx.dec, last, inp, N_TICKS)
            fanc = R.roll_fan(ctx.dec, last, inp)
            hidx = [h for h in ctx.horizons]
            d3.append({"tick_states_eq_roll_bitwise": bool(torch.equal(st[..., hidx, :2], fanc)),
                       "max_abs": float((st[..., hidx, :2] - fanc).abs().max()), "tile_max_abs": tile_dev})
            resid = (ctx.KP.residual_controls(u0_ref, *inp.prior, control_units=ctx.dec.anchor_control_units,
                                              alat_v_floor=ctx.dec.anchor_alat_v_floor)
                     / u0_ref.new_tensor(tuple(ctx.dec.cfg.control_norm)))
            row = {}
            for c, nm in ((0, "a_lon"), (1, "a_lat")):
                row[f"x0_out_of_box_{nm}"] = float((roll["x0"][..., c, :].abs() > 1).float().mean())
                row[f"x0_final_out_of_box_{nm}"] = float((roll["x0"][..., c, -1].abs() > 1).float().mean())
                row[f"input_out_of_box_{nm}"] = float((roll["chain"][..., :-1][..., c, :].abs() > 1).float().mean())
                row[f"deployed_u0_resid_out_of_box_{nm}"] = float((resid[..., c].abs() > 1).float().mean())
            d2.append(row)
            # D5: reward known values on THIS window
            wi = ctx.window_inputs(e_i, t, _row(batch, 0))
            dev = st.device
            hum = wi["human"].to(dev)
            tracks = wi["tracks"].to(dev)
            route = wi["route"].to(dev)
            sc = R.score_window(st[0], hum, tracks, route, wi["off"], ctx.pcfg)
            ident = R.score_window(hum[None], hum, tracks, route, wi["off"], ctx.pcfg)
            # EP saturation control: the human path re-timed 1.5x faster / 0.5x slower
            fast = _retime(hum, 1.5, ctx.pcfg)
            slow = _retime(hum, 0.5, ctx.pcfg)
            ep_ctrl = R.score_window(torch.stack([fast, slow]), hum, tracks, route, wi["off"], ctx.pcfg)
            # DAC control: the human path shifted 6 m left / right (D3's mutation)
            shl = hum.clone()
            shl[:, 1] += 6.0
            shr = hum.clone()
            shr[:, 1] -= 6.0
            dac_shift = (None if wi["off"] is None else
                         [float(x) for x in R.dac_fine(torch.stack([shl, shr]), wi["off"], R.DAC_RULE, ctx.pcfg)])
            # NC control: a candidate whose FRONT-EDGE MIDPOINT sits on a moving agent ahead at some tick
            nc_ctrl = _collision_control(R, P, ctx, hum, tracks, route, wi["off"])
        d5.append({"sha12": wi["sha12"], "t0": wi["t0"], "human": sc["human"],
                   "identity_equal": abs(float(ident["pdms"][0]) - sc["human"]["pdms"]) == 0.0,
                   "ep_fast": float(ep_ctrl["ep"][0]), "ep_slow": float(ep_ctrl["ep"][1]),
                   "spd_fast": float(ep_ctrl["spd"][0]), "dac_shift6m": dac_shift, "nc_control": nc_ctrl,
                   "dac_dead": wi["off"] is None, "n_tracks_scored": sc["n_tracks_scored"],
                   "chain_reward_mean": float(sc["pdms"].mean()), "chain_dac_mean": float(sc["dac"].mean()),
                   "chain_nc_mean": float(sc["nc"].mean())})
        log(f"[diag {k + 1}/{len(pick)}] {wi['sha12']}@{wi['t0']} D1 {d1[-1]} D1c {d1c[-1]} D3 {d3[-1]} "
            f"human {sc['human']} nc_ctrl {nc_ctrl} dac6 {dac_shift} ep {d5[-1]['ep_fast']:.3f}/{d5[-1]['ep_slow']:.3f}")
    n = len(pick)
    rec["D1_parity"] = {"fan_bitwise_all": all(x["fan_bitwise"] for x in d1),
                        "u0_bitwise_all": all(x["u0_bitwise"] for x in d1),
                        "fan_max_abs_max": max(x["fan_max_abs"] for x in d1),
                        "traj_in_fan_all": all(x["traj_in_fan"] for x in d1),
                        "trainer_forward_same_sampler_inputs_all": all(d1c)}
    rec["D1b_group_tiling_max_abs"] = max(x["tile_max_abs"] for x in d3)
    rec["D3_tick_states_eq_roll_all"] = all(x["tick_states_eq_roll_bitwise"] for x in d3)
    rec["D3_tick_states_max_abs"] = max(x["max_abs"] for x in d3)
    rec["D2_clamp_rates_mean"] = {k: sum(x[k] for x in d2) / n for k in d2[0]}
    hm = [x["human"] for x in d5]
    rec["D5_known_values"] = {
        "human_nc1_frac": sum(h["nc"] == 1.0 for h in hm) / n,
        "human_dac1_frac": sum(h["dac"] == 1.0 for h in hm) / n,
        "human_ttc1_frac": sum(h["ttc"] == 1.0 for h in hm) / n,
        "human_comfort1_frac": sum(h["comfort"] == 1.0 for h in hm) / n,
        "human_pdms_mean": sum(h["pdms"] for h in hm) / n,
        "identity_equal_all": all(x["identity_equal"] for x in d5),
        "ep_fast_eq_1_frac": sum(x["ep_fast"] == 1.0 for x in d5) / n,
        "ep_slow_mean": sum(x["ep_slow"] for x in d5) / n,
        "spd_fast_mean": sum(x["spd_fast"] for x in d5) / n,
        "dac_dead_windows": sum(x["dac_dead"] for x in d5),
        "dac_shift6m_zero_frac": (lambda v: (sum(v) / len(v)) if v else None)(
            [1.0 - y for x in d5 if x["dac_shift6m"] for y in x["dac_shift6m"]]),
        "nc_control": {"n_built": sum(1 for x in d5 if x["nc_control"] is not None),
                       "n_nc_below_1": sum(1 for x in d5 if x["nc_control"] is not None and x["nc_control"]["nc"] < 1.0)}}
    rec["D5_rows"] = d5
    # D4 + D6: one full RL step's cost and the cold-start advantage statistics (NO update)
    named = R.trainable_named(ctx.dec)
    ctx.named = named
    for _, p in named:
        p.requires_grad_(True)
    params = [p for _, p in named]
    d4 = []
    for mb in a.cost_micro:
        if mb > len(pick):
            continue
        ws = pick[:mb]
        batch = torch.utils.data.default_collate([ctx.ds[imap[w]] for w in ws])
        arm = Arm("rl", 1.0, 4, "matched_anchor", 100.0, False)
        gen = torch.Generator(device=a.device).manual_seed(3)
        sg = torch.Generator().manual_seed(4)
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        stx = rl_step(ctx, arm, [(batch, ws)], None, gen, sg, 0, 0, consts, params, opt=None, apply=False)
        _sync()
        d4.append({"micro": mb, "step_s": round(time.time() - t, 3), "s_per_window": round((time.time() - t) / mb, 3),
                   "peak_mem_GB": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None),
                   **{k: stx[k] for k in ("timing", "grad_norm", "grad_groups", "frac_positive_after_bar",
                                         "frac_admitted_by_bar", "frac_constraint_fail", "reward_mean",
                                         "human_pdms_mean", "cand_dac_mean", "cand_nc_mean", "cand_ep_mean",
                                         "cand_ttc_mean", "cand_comfort_mean", "reward_group_std_pos_frac",
                                         "rows_with_positive", "chain_endpoint_spread_m", "il_m")}})
        log(f"[diag D4] micro {mb}: {d4[-1]}")
    rec["D4_D6_rl_step"] = d4
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    log(json.dumps({k: v for k, v in rec.items() if k not in ("D5_rows", "D4_D6_rl_step")}, indent=1, default=str))
    log(f"[diag] wrote {a.out}")
    if lock is not None:
        lock.release()


def _retime(states, factor: float, cfg):
    """The human path re-timed by ``factor`` (1.5 = 50 % faster): the SAME geometry, sampled at
    ``factor`` x the arc length per tick (positions linearly interpolated along the polyline)."""
    import torch
    xy = states[:, :2]
    seg = (xy[1:] - xy[:-1]).norm(dim=-1)
    s = torch.cat([seg.new_zeros(1), torch.cumsum(seg, 0)])
    tgt = (s * factor).clamp(max=float(s[-1]))
    j = torch.searchsorted(s, tgt).clamp(1, len(s) - 1)
    w = ((tgt - s[j - 1]) / (s[j] - s[j - 1]).clamp_min(1e-9)).clamp(0, 1)
    p = xy[j - 1] + w[:, None] * (xy[j] - xy[j - 1])
    yaw = states[j - 1, 2] + w * (states[j, 2] - states[j - 1, 2])
    v = states[:, 3] * factor
    out = torch.stack([p[:, 0], p[:, 1], yaw, v], -1)
    out[0] = states[0]
    out[0, 3] = states[0, 3] * factor
    return out


def _collision_control(R, P, ctx, hum, tracks, route, off):
    """Build a candidate whose front-edge midpoint sits on a valid, MOVING, non-static agent at the
    first tick where one is 5-40 m ahead; -> {nc, tick, agent} or None when no such agent exists."""
    import torch
    cfg = ctx.pcfg
    T0 = hum.shape[0]
    for k in range(5, T0):
        for j in range(tracks.xy.shape[1]):
            if not bool(tracks.valid[k, j]) or bool(tracks.static[j]):
                continue
            x, y = float(tracks.xy[k, j, 0]), float(tracks.xy[k, j, 1])
            if not (5.0 < x < 40.0 and abs(y) < 3.0):
                continue
            # straight path at constant speed reaching (front-edge midpoint) == agent centre at tick k
            reach = x - (cfg.rear_axle_to_center + cfg.ego_length / 2)
            yaw = math.atan2(y, max(reach, 1e-3))
            dist = math.hypot(reach, y)
            v = dist / (k * cfg.dt)
            ts = torch.arange(T0, dtype=torch.float32, device=hum.device) * cfg.dt
            st = torch.stack([v * ts * math.cos(yaw), v * ts * math.sin(yaw),
                              torch.full_like(ts, yaw), torch.full_like(ts, v)], -1)
            sc = R.score_window(st[None], hum, tracks, route, off, cfg)
            return {"nc": float(sc["nc"][0]), "tick": k, "agent_xy": [round(x, 2), round(y, 2)],
                    "v": round(v, 2), "pdms": float(sc["pdms"][0]), "fail": bool(sc["constraint_fail"][0])}
    return None


# =========================================================================== #
# census -- the reward's KNOWN VALUES on many real windows, MODEL-FREE (no forward)   #
# =========================================================================== #
DAC_LADDER = ("R1", "R2")


def offroad_rule(R, codes, rule: str):
    """The pre-listed DAC rule ladder (SPEC_RL A-0): ``R.OFFROAD_RULES[rule]``."""
    if rule not in R.OFFROAD_RULES:
        raise ValueError(rule)
    return R.OFFROAD_RULES[rule](codes)


def cmd_census(a):
    import torch
    ctx = Ctx(a, a.split)
    R, P = ctx.R, ctx.P
    windows, wrec = load_windows(a.windows)
    g = torch.Generator().manual_seed(int(a.seed))
    pick = [windows[int(i)] for i in torch.randperm(len(windows), generator=g)[:a.n_windows]]
    imap = window_index_map(ctx)
    rows = []
    code_hist = {r: torch.zeros(256, dtype=torch.long) for r in DAC_LADDER}
    t0 = time.time()
    for k, (e_i, t) in enumerate(pick):
        item = ctx.ds[imap[(e_i, t)]]
        batch = torch.utils.data.default_collate([item])
        wi = ctx.window_inputs(e_i, t, _row(batch, 0))
        hum, route, tr = wi["human"], wi["route"], wi["tracks"]
        codes = batch["map_fine"][0]
        row = {"sha12": wi["sha12"], "t0": wi["t0"], "map": bool(batch["map_fine_label"][0]),
               "v0": float(hum[0, 3])}
        cands = torch.stack([_retime(hum, 1.5, ctx.pcfg), _retime(hum, 0.5, ctx.pcfg)])
        shl, shr = hum.clone(), hum.clone()
        shl[:, 1] += 6.0
        shr[:, 1] -= 6.0
        for rule in DAC_LADDER:
            off = offroad_rule(R, codes, rule) if row["map"] else None
            sc = R.score_window(cands, hum, tr, route, off, ctx.pcfg)
            row[rule] = {"human": sc["human"],
                         "fast": {kk: float(sc[kk][0]) for kk in ("nc", "dac", "ep", "ttc", "comfort", "spd", "pdms")},
                         "slow": {kk: float(sc[kk][1]) for kk in ("nc", "dac", "ep", "ttc", "comfort", "spd", "pdms")}}
            if off is not None:
                row[rule]["shift6_dac"] = [float(x) for x in R.dac_fine(torch.stack([shl, shr]), off, R.DAC_RULE, ctx.pcfg)]
                if sc["human"]["dac"] < 1.0:
                    corners = P._ego_boxes(hum[None], ctx.pcfg)[0]                       # [T, 4, 2]
                    ix = torch.floor(corners[..., 0] / 0.1).long().clamp(0, codes.shape[0] - 1)
                    iy = torch.floor((corners[..., 1] + 30.0) / 0.1).long().clamp(0, codes.shape[1] - 1)
                    bad = off[ix, iy]
                    code_hist[rule] += torch.bincount(codes[ix[bad], iy[bad]].long(), minlength=256)
                    row[rule]["human_first_bad_tick"] = int(bad.any(-1).float().argmax())
        # the human's own CENTRE point on the map (D3's instrument), for reference
        if row["map"]:
            cx, cy = P._centers(hum[None], ctx.pcfg)
            ix = torch.floor(cx[0] / 0.1).long()
            iy = torch.floor((cy[0] + 30.0) / 0.1).long()
            ok = (ix >= 0) & (ix < codes.shape[0]) & (iy >= 0) & (iy < codes.shape[1])
            cc = codes[ix[ok], iy[ok]].long()
            row["centre_codes"] = {str(int(c)): int(n) for c, n in zip(*torch.unique(cc, return_counts=True))}
        rows.append(row)
        if k % 25 == 0:
            log(f"[census {k + 1}/{len(pick)}] {time.time() - t0:.0f}s {row['sha12']}@{row['t0']} "
                f"R1 human {row['R1']['human']['dac']} R2 human {row['R2']['human']['dac']}")
    n = len(rows)
    summ = {}
    for rule in DAC_LADDER:
        hs = [r[rule]["human"] for r in rows]
        summ[rule] = {
            "human_nc1_frac": sum(h["nc"] == 1.0 for h in hs) / n,
            "human_dac1_frac": sum(h["dac"] == 1.0 for h in hs) / n,
            "human_ttc1_frac": sum(h["ttc"] == 1.0 for h in hs) / n,
            "human_comfort1_frac": sum(h["comfort"] == 1.0 for h in hs) / n,
            "human_pdms_mean": sum(h["pdms"] for h in hs) / n,
            "fast_ep1_frac": sum(r[rule]["fast"]["ep"] == 1.0 for r in rows) / n,
            "fast_spd_mean": sum(r[rule]["fast"]["spd"] for r in rows) / n,
            "slow_ep_mean": sum(r[rule]["slow"]["ep"] for r in rows) / n,
            "slow_nc1_frac": sum(r[rule]["slow"]["nc"] == 1.0 for r in rows) / n,
            "slow_dac1_frac": sum(r[rule]["slow"]["dac"] == 1.0 for r in rows) / n,
            "shift6_dac0_frac": (lambda v: sum(v) / len(v) if v else None)(
                [1.0 - x for r in rows if "shift6_dac" in r[rule] for x in r[rule]["shift6_dac"]]),
            "codes_under_failing_human_corners": {str(c): int(code_hist[rule][c]) for c in range(256)
                                                  if int(code_hist[rule][c])}}
    rec = {"_tier": "T0 reward known values (model-free)", "_evidence_class": "MEASURED (ours)",
           "split": a.split, "n_windows": n, "seed": a.seed, "windows_digest": wrec["digest"],
           "dac_rule_block": vars(R.DAC_RULE), "ladder": list(DAC_LADDER), "summary": summ,
           "n_map_missing": sum(1 for r in rows if not r["map"]), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, default=str), encoding="utf-8")
    log(json.dumps({"n": n, "summary": summ}, indent=1, default=str))
    log(f"[census] wrote {a.out}")


# =========================================================================== #
# heldout -- the SECONDARY read on eval139 (SPEC_RL 6.2 item 1), paired inference noise  #
# =========================================================================== #
def cmd_heldout(a):
    """One checkpoint (the cold start or an EXPORT) on the eligible eval windows: the DEPLOYED
    sampler (2 steps, eta 0) with inference noise seed ``500_000 + 1_000_000 * infer_seed + k`` per
    window, so two checkpoints read with the same ``--infer-seed`` are PAIRED window by window.
    Per window: the selected plan's PDMS-proxy and sub-scores, its off-road / collision flags and
    speed bias vs the human; the fan's proxy mean, best (ORACLE), endpoint spread, best-of-N ADE."""
    import torch
    ctx = Ctx(a, "eval")
    R = ctx.R
    windows, wrec = load_windows(a.windows)
    if a.stride > 1:
        windows = windows[::a.stride]
    imap = window_index_map(ctx)
    lock = InprocLock(a.inproc_lock) if a.inproc_lock else None
    if lock is not None:
        log(f"[heldout] GPU lock acquired after {lock.acquire():.0f}s")
    rows = []
    t0 = time.time()
    norm = None
    for k, (e_i, t) in enumerate(windows):
        batch = torch.utils.data.default_collate([ctx.ds[imap[(e_i, t)]]])
        inp, out = ctx.capture(batch, seed=500_000 + 1_000_000 * int(a.infer_seed) + k)
        fan, u0 = inp.out[0], inp.out[1]                                   # [1, N, S, 2], ABSOLUTE
        if norm is None:
            norm = u0.new_tensor(tuple(ctx.dec.cfg.control_norm))
        x_n = ctx.KP.residual_controls(u0, *inp.prior, control_units=ctx.dec.anchor_control_units,
                                       alat_v_floor=ctx.dec.anchor_alat_v_floor) / norm
        with torch.no_grad():
            st = R.tick_states(ctx.dec, x_n, inp, N_TICKS)                 # [1, N, 61, 4]
        recon = float((st[0][:, list(ctx.horizons), :2] - fan[0]).abs().max())
        traj = out["traj"][0]
        dsel = (fan[0] - traj[None]).abs().amax(dim=(-1, -2))
        sel = int(dsel.argmin())
        wi = ctx.window_inputs(e_i, t, _row(batch, 0))
        dev = st.device
        hum = wi["human"].to(dev)
        sc = R.score_window(st[0], hum, wi["tracks"].to(dev), wi["route"].to(dev), wi["off"], ctx.pcfg)
        gt = wi["gt"].to(dev)
        ade = (fan[0] - gt[None]).norm(dim=-1).mean(-1)
        ends = fan[0, :, -1]
        v_sel = float(st[0, sel, 1:, 3].mean())
        v_hum = float(hum[1:, 3].mean())
        rows.append({
            "sha12": wi["sha12"], "t0": wi["t0"], "sel_idx": sel, "sel_match_abs": float(dsel[sel]),
            "tick_recon_max_abs": recon,
            **{f"sel_{kk}": float(sc[kk][sel]) for kk in ("pdms", "nc", "dac", "ep", "ttc", "comfort", "spd")},
            "sel_offroad": float(sc["dac"][sel] < 1.0), "sel_collision": float(sc["nc"][sel] < 1.0),
            "sel_speed_bias_mps": v_sel - v_hum,
            "sel_ade_m": float(ade[sel]), "sel_fde_m": float((fan[0, sel, -1] - gt[-1]).norm()),
            "fan_pdms_mean": float(sc["pdms"].mean()), "fan_pdms_best_ORACLE": float(sc["pdms"].max()),
            "fan_nc_fail_frac": float((sc["nc"] < 1).float().mean()),
            "fan_dac_fail_frac": float((sc["dac"] < 1).float().mean()),
            "fan_endpoint_spread_m": float(torch.cdist(ends, ends).mean()),
            "fan_minade_m_ORACLE": float(ade.min()),
            "human_pdms": sc["human"]["pdms"], "human_dac": sc["human"]["dac"], "human_nc": sc["human"]["nc"]})
        if k % 100 == 0:
            log(f"[heldout {k + 1}/{len(windows)}] {time.time() - t0:.0f}s {rows[-1]}")
    if lock is not None:
        lock.release()
    keys = [kk for kk in rows[0] if kk not in ("sha12", "t0", "sel_idx")]
    rec = {"ckpt": a.ckpt, "ckpt_md5": a.ckpt_md5, "infer_seed": a.infer_seed, "stride": a.stride,
           "n_windows": len(rows), "n_episodes": len({r["sha12"] for r in rows}),
           "windows_digest": wrec["digest"], "dac_rule": ctx.dac_rule,
           "means": {kk: sum(r[kk] for r in rows) / len(rows) for kk in keys},
           "max_tick_recon_abs": max(r["tick_recon_max_abs"] for r in rows),
           "max_sel_match_abs": max(r["sel_match_abs"] for r in rows),
           "_tier": "T0 secondary (deployed sampler on logged frames, recorded future; PDMS-SHAPED proxy)",
           "_evidence_class": "MEASURED (ours)", "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
    log(json.dumps({kk: v for kk, v in rec.items() if kk != "rows"}, indent=1))


def cmd_compare(a):
    """Paired episode-cluster bootstrap of every per-window metric: A - B (same windows, same
    inference seed). (!!) Answers 'another draw of EPISODES' only."""
    from taniteval import ci
    A = json.loads(Path(a.a).read_text(encoding="utf-8"))
    B = json.loads(Path(a.b).read_text(encoding="utf-8"))
    ka = {(r["sha12"], r["t0"]): r for r in A["rows"]}
    kb = {(r["sha12"], r["t0"]): r for r in B["rows"]}
    common = sorted(set(ka) & set(kb))
    if len(common) != len(ka) or len(common) != len(kb):
        raise SystemExit(f"[compare] window sets differ: {len(ka)} vs {len(kb)}, common {len(common)}")
    if A["infer_seed"] != B["infer_seed"]:
        raise SystemExit("[compare] different inference seeds -- not paired")
    eid = [w[0] for w in common]
    out = {"a": a.a, "b": a.b, "n_windows": len(common), "n_episodes": len(set(eid)),
           "infer_seed": A["infer_seed"], "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "n_boot": a.n_boot, "variance_answered": "another draw of EPISODES only", "cells": {}}
    keys = [k for k in A["rows"][0] if k not in ("sha12", "t0", "sel_idx", "sel_match_abs",
                                                  "tick_recon_max_abs")]
    for k in keys:
        va = [ka[w][k] for w in common]
        vb = [kb[w][k] for w in common]
        out["cells"][k] = ci.paired_episode_cluster_bootstrap(va, vb, eid, n_boot=a.n_boot, seed=0)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    log(json.dumps(out, indent=1, default=str)[:4000])


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--ckpt", default="/home/nvidia/refcv7_post/rl/ckpt_50400.pt")
        s.add_argument("--ckpt-md5", default=EXPECT_CKPT_MD5)
        s.add_argument("--skip-md5", action="store_true")
        s.add_argument("--config", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/config.json")
        s.add_argument("--device", default="cuda")
        s.add_argument("--remap-file", default=None)
        s.add_argument("--dac-rule", choices=("R1", "R2"), default="R2",
                       help="SPEC_RL A-0 ladder; chosen on the TRAIN human census")

    s = sub.add_parser("windows")
    common(s)
    s.add_argument("--split", choices=("train", "eval"), required=True)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--out", required=True)

    s = sub.add_parser("census")
    common(s)
    s.add_argument("--split", choices=("train", "eval"), required=True)
    s.add_argument("--windows", required=True)
    s.add_argument("--n-windows", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)

    s = sub.add_parser("diagnose")
    common(s)
    s.add_argument("--split", choices=("train", "eval"), default="train")
    s.add_argument("--windows", required=True)
    s.add_argument("--n-windows", type=int, default=12)
    s.add_argument("--cost-micro", type=int, nargs="+", default=[1, 2, 4])
    s.add_argument("--inproc-lock", default="")
    s.add_argument("--out", required=True)

    s = sub.add_parser("train")
    common(s)
    s.add_argument("--arm", choices=Arm.KINDS, required=True)
    s.add_argument("--rl-weight", type=float, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--micro", type=int, required=True)
    s.add_argument("--groups", type=int, default=4)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--weight-decay", type=float, default=1e-4)
    s.add_argument("--il-form", choices=("matched_anchor", "release_all_modes"), default="matched_anchor")
    s.add_argument("--grad-clip", type=float, default=100.0)
    s.add_argument("--clip-sample", action="store_true", help="the release's x0 clamp (OFF: SPEC D-4)")
    s.add_argument("--windows", required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--ckpt-every", type=int, default=25)
    s.add_argument("--segment-minutes", type=float, default=45.0)
    s.add_argument("--min-segment-minutes", type=float, default=5.0)
    s.add_argument("--max-steps-this-segment", type=int, default=0)
    s.add_argument("--yield-lock", default="")
    s.add_argument("--inproc-lock", default="", help="hold/yield the Thor GPU lock IN-PROCESS")
    s.add_argument("--log-every", type=int, default=1)
    s.add_argument("--out-dir", required=True)

    s = sub.add_parser("heldout")
    common(s)
    s.add_argument("--windows", required=True)
    s.add_argument("--stride", type=int, default=4)
    s.add_argument("--infer-seed", type=int, default=0)
    s.add_argument("--inproc-lock", default="")
    s.add_argument("--out", required=True)

    s = sub.add_parser("compare")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--out", required=True)

    s = sub.add_parser("smoke")
    common(s)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--groups", type=int, default=4)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--weight-decay", type=float, default=1e-4)
    s.add_argument("--il-form", choices=("matched_anchor", "release_all_modes"), default="matched_anchor")
    s.add_argument("--grad-clip", type=float, default=100.0)
    s.add_argument("--clip-sample", action="store_true")
    s.add_argument("--windows", required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--inproc-lock", default="")
    s.add_argument("--out-dir", required=True)

    s = sub.add_parser("export")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--tag", default="latest")
    s.add_argument("--ckpt", default="/home/nvidia/refcv7_post/rl/ckpt_50400.pt")
    s.add_argument("--config", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/config.json")
    s.add_argument("--out", required=True)

    s = sub.add_parser("identity")
    s.add_argument("--export", required=True)
    s.add_argument("--ckpt", default="/home/nvidia/refcv7_post/rl/ckpt_50400.pt")
    s.add_argument("--expect", choices=("identical", "different", "report"), default="report")
    s.add_argument("--out", default="")

    a = ap.parse_args(argv)
    if a.cmd == "train":
        if a.rl_weight is None:
            a.rl_weight = 0.0 if a.arm == "rloff" else 1.0
    {"windows": cmd_windows, "diagnose": cmd_diagnose, "train": cmd_train, "export": cmd_export,
     "identity": cmd_identity, "census": cmd_census, "heldout": cmd_heldout,
     "compare": cmd_compare, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
