"""BAR A — the pre-registered discriminating experiment for flagship-v4.

QUESTION
--------
v4's fan already contains trajectories that beat deployed v1 (produced-goal
`oracle_in_fan` 0.2505 vs v1 0.4271), but the SELECTOR picks 0.8563 — it gives
away 0.6058 m. Is the ERROR-BLIND 1-of-256 cross-entropy that trains the
selector the reason?

INTERVENTION (pre-registered; the ONLY change)
----------------------------------------------
Replace `v15_losses`'s

    r_star    = fan_err.argmin(dim=1)
    loss_rcls = F.cross_entropy(out["sel_score"], r_star)          # error-blind

with the cost-sensitive expected-regret listwise loss

    regret    = fan_err - fan_err.min(dim=1, keepdim=True).values  # metres
    p         = softmax(out["sel_score"] / TAU, dim=1)             # TAU = 1.0
    loss_rcls = (p * regret.detach()).sum(dim=1).mean()            # E[metres given up]

REFINED_CLS_WEIGHT = 1.0, TAU = 1.0, nothing else changed.

WHY THE FAN IS PROVABLY FROZEN (code fact, MEASURED by reading the modules)
--------------------------------------------------------------------------
`V15Decoder._decode` emits `conf = conf_head(q)` and `offset = offset_head(q)`
from the SAME query features. The fan is
    x = x_in + offset            (offset_head only)
and the score is
    refined_logits = conf        (conf_head only)
  + factorised grafts            (lat/lon/dist_head -> *_to_anchor, on states[:,-1])
  + sel_gate * pen               (FlagshipV15Head.select)
`anchor_traj` therefore depends on NO parameter in the trainable set below.
Trainable set == exactly the parameters that move the SCORE and cannot move the
FAN:
    decoder.conf_head, lat_head, lon_head, dist_head,
    lat_to_anchor, lon_to_anchor, dist_to_anchor, sel_gate
Everything else — world model, grounding, goal_head, decoder trunk, offset_head,
anchors, conditioning — is frozen AND asserted frozen (`_assert_frozen`).

DESIGN — cached-feature cross-fit
---------------------------------
Because the fan is frozen, every quantity the selector needs is a FIXED function
of the forward pass. One forward pass over all 6,844 windows of the 40 val
episodes caches:
    q0  [256,512]  conf_head input, t=0 classifier pass   (-> anchor_logits)
    qf  [256,512]  conf_head input, FINAL denoise pass    (-> refined_logits)
    fan [256,20,2] anchor_traj  (FROZEN — never recomputed)
    state_last [2048], traj_tgt [20,2], v0, vt_speed, vt_keep, eid, ep, t
Fine-tuning is then exact algebra over the cache — it calls the head's OWN
`_factor_grafts` and `select` methods, not reimplementations.

CACHE FIDELITY IS PROVEN, NOT ASSUMED: running the AS-TRAINED parameters through
the cached path must reproduce the published `ade_0_2s` to <=1e-3 on the same
881 windows (`selftest.cache_fidelity`), and a deliberately-broken scorer
(random logits) must be reported as WORSE (`selftest.failing_input` — the M3
rule that an instrument which cannot fail cannot adjudicate).

DEVIATION FROM THE §6 SKETCH, STATED UP FRONT
---------------------------------------------
The canonical TRAIN corpus (`physicalai-train-e438721ae894`) is NOT on the eval
pod (only the 40-episode val cache is; MEASURED). Rather than touch a training
pod, the rescorer is fitted by 5-fold EPISODE-DISJOINT CROSS-FITTING over the 40
val episodes: each fold trains on 32 episodes (26 fit + 6 inner-val for early
stopping) and scores the 8 it never saw, so every one of the 881 canonical
windows gets an OUT-OF-FOLD score. This is held-out by construction (C11), but
it gives the fine-tuned arms a val-distribution adaptation advantage the
as-trained selector never had. That is why arm CE_CONTROL exists: an identical
fine-tune under the ORIGINAL cross-entropy. Delta(REGRET - CE_CONTROL) isolates
the LOSS; Delta(CE_CONTROL - AS_TRAINED) measures the adaptation+refit effect.

ESTIMATOR: paired episode-cluster bootstrap (`taniteval/ci.py`, B=2000) on the
identical 881 windows. NEVER `overlapping_holdout_se`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STACK = "/root/v4eval/stack"
sys.path.insert(0, STACK + "/scripts")
sys.path.insert(0, STACK)
sys.path.insert(0, "/root/taniteval")

import eval_flagship_v4 as E  # noqa: E402
import goal_modes  # noqa: E402
import refb_labels  # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,  # noqa: E402
                          paired_episode_cluster_bootstrap)

CKPT = "/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt"
HCFG = "/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
ANCH = "/root/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt"
DEV = "cuda"
OUTDIR = Path("/root/bara")

# ---- pre-registered constants ---------------------------------------------
TAU = 1.0
REFINED_CLS_WEIGHT = 1.0
N_FOLDS = 5
N_INNER_VAL_EP = 6
MAX_STEPS = 2000
EVAL_EVERY = 100
BATCH = 32
LR_GRID = (3e-5, 1e-4, 3e-4)
SEED = 0

# ---- the committed numbers this experiment is judged against --------------
COMMITTED = {
    "oracle": {"ade": 0.6423, "oif": 0.2330},
    "produced": {"ade": 0.8563, "oif": 0.2505},
}
V1_REF = 0.4271                    # MODEL_REGISTRY full_set mean, the bar
BARS = {"CONFIRM": 0.708, "PARTIAL_LO": 0.30}

TRAINABLE = ("decoder.conf_head", "lat_head", "lon_head", "dist_head",
             "lat_to_anchor", "lon_to_anchor", "dist_to_anchor", "sel_gate")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ===========================================================================
# trainable-parameter selection + freeze proof
# ===========================================================================
def selector_params(head):
    """The parameters that move the SCORE and provably cannot move the FAN."""
    out = {}
    for n, p in head.named_parameters():
        if any(n == t or n.startswith(t + ".") for t in TRAINABLE):
            out[n] = p
    return out


def _assert_frozen(head, world, grounding, goal_head, sel_names):
    """Everything not in the trainable set must have requires_grad False."""
    bad = [n for n, p in head.named_parameters()
           if p.requires_grad and n not in sel_names]
    for mod, tag in ((world, "world"), (grounding, "grounding"),
                     (goal_head, "goal_head")):
        if mod is None:
            continue
        bad += [f"{tag}.{n}" for n, p in mod.named_parameters()
                if p.requires_grad]
    if bad:
        raise RuntimeError(f"NOT FROZEN: {bad[:12]} (+{max(0, len(bad)-12)})")
    fan_touching = ("decoder.offset_head", "decoder.traj_proj", "decoder.layers",
                    "decoder.feat_proj", "decoder.cond_proj", "decoder.time_embed",
                    "decoder.anchors")
    leak = [n for n in sel_names
            if any(n.startswith(f) for f in fan_touching)]
    if leak:
        raise RuntimeError(f"TRAINABLE SET TOUCHES THE FAN: {leak}")
    return True


# ===========================================================================
# STAGE 1 — capture the frozen-fan selector features
# ===========================================================================
@torch.no_grad()
def build_cache(world, grounding, head, goal_head, ds_val, goal_mode,
                episodes=40, stride=1, batch=16):
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device

    head.eval()
    horizons = head.cfg.horizons
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]

    qs: list[torch.Tensor] = []

    def pre_hook(_m, args):
        qs.append(args[0].detach())

    h_handle = head.decoder.conf_head.register_forward_pre_hook(pre_hook)

    # stash vt_keep, which `condition()` returns but `forward` does not expose
    real_condition = head.condition
    stash = {}

    def wrapped_condition(*a, **k):
        m, tele, vt_keep = real_condition(*a, **k)
        stash["vt_keep"] = vt_keep
        return m, tele, vt_keep

    head.condition = wrapped_condition                       # type: ignore

    # DTYPE NOTE (MEASURED 2026-07-26): caching `qf`/`fan` in fp16 moved the
    # cached-path `ade_0_2s` to 0.8591 against a published 0.8563 -- the
    # cache-fidelity self-test caught it and ABORTED the run. The cause is the
    # near-tie structure the whole diagnosis rests on: 256 candidates whose
    # scores differ by less than fp16's ULP, so quantisation flips the argmax.
    # `qf`/`fan`/`state_last` are therefore fp32 (the forward pass runs in fp32 --
    # `collect_planner` uses no autocast). `q0` feeds only the auxiliary anchor-CE
    # and never the selection, so it stays fp16.
    C = {k: [] for k in ("q0", "qf", "fan", "state_last", "tgt", "v0",
                         "vt_speed", "vt_keep", "eid", "ep", "t",
                         "ref_sel_idx", "ref_ade4")}
    t0 = time.time()
    try:
        for b0 in range(0, len(sel), batch):
            idx = sel[b0:b0 + batch]
            items = [ds_val[i] for i in idx]
            b = _to_device(default_collate(items), DEV)
            v0 = b["pose_last"][:, 3].float()
            tgt = refb_labels.waypoint_targets(
                b["pose_last"].float(),
                b["future_poses"][:, :max(horizons)].float(), horizons)
            st = world.encode_window(b["frames"])
            goal_kw, _rec = goal_modes.resolve_goal(
                goal_mode, head=head, batch=b, v0=v0, states=st,
                goal_head=goal_head, allow_fallback=False)
            qs.clear()
            out = head(st, v0, lambda_plan=1.0, **goal_kw)
            if len(qs) != 1 + head.cfg.decoder.diffusion_steps:
                raise RuntimeError(
                    f"expected {1 + head.cfg.decoder.diffusion_steps} conf_head "
                    f"calls, hooked {len(qs)} -- decoder shape changed")
            C["q0"].append(qs[0].half().cpu())          # aux CE only
            C["qf"].append(qs[-1].float().cpu())        # SELECTION -> fp32
            C["fan"].append(out["anchor_traj"].float().cpu())
            C["state_last"].append(st[:, -1].float().cpu())
            C["tgt"].append(tgt.float().cpu())
            # the REAL forward pass's own pick + score, so cache fidelity is a
            # per-window identity check rather than a mean-vs-mean coincidence
            C["ref_sel_idx"].append(out["sel_idx"].cpu())
            _p = wp4(out["traj"], horizons)
            _g = wp4(tgt, horizons)
            C["ref_ade4"].append((_p - _g).norm(dim=-1).mean(dim=1).float().cpu())
            C["v0"].append(v0.float().cpu())
            vts = goal_kw.get("vt_speed")
            C["vt_speed"].append(vts.float().cpu() if vts is not None
                                 else v0.float().cpu())
            vk = stash.get("vt_keep")
            C["vt_keep"].append(vk.cpu() if vk is not None
                                else torch.ones_like(v0, dtype=torch.bool).cpu())
            for i in idx:
                e_i, tt = ds_val.index[i]
                C["eid"].append(int(ds_val.episodes[e_i].episode_id))
                C["ep"].append(int(e_i))
                C["t"].append(int(tt))
            if b0 % (batch * 40) == 0:
                print(f"  [cache/{goal_mode}] {b0 + len(idx)}/{len(sel)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        h_handle.remove()
        head.condition = real_condition                      # type: ignore

    cache = {k: (torch.cat(v) if isinstance(v[0], torch.Tensor)
                 else torch.tensor(v)) for k, v in C.items()}
    cache["_wallclock_s"] = round(time.time() - t0, 1)
    cache["_goal_mode"] = goal_mode
    print(f"[cache/{goal_mode}] {cache['q0'].shape[0]} windows in "
          f"{cache['_wallclock_s']}s", flush=True)
    return cache


# ===========================================================================
# STAGE 2 — the cached-feature selector forward (uses the head's OWN methods)
# ===========================================================================
def cache_to_gpu(C, keys=("q0", "qf", "fan", "state_last", "tgt", "v0",
                         "vt_speed", "vt_keep")):
    """Hold the frozen-fan feature cache on the A40 (~3.8 GiB of 46 GiB) so the
    fine-tune is not PCIe-bound: 60k steps x 34 MiB would otherwise move ~2 TB."""
    for k in keys:
        C[k] = C[k].to(DEV)
    torch.cuda.synchronize()
    return C


def score_from_cache(head, C, sl, need_anchor_logits=False):
    """-> dict with sel_score / sel_idx / traj / fan_err / anchor_logits."""
    sl = sl.to(C["qf"].device)
    qf = C["qf"][sl].to(DEV).float()
    fan = C["fan"][sl].to(DEV).float()
    stl = C["state_last"][sl].to(DEV).float()
    tgt = C["tgt"][sl].to(DEV)
    v0 = C["v0"][sl].to(DEV)
    vts = C["vt_speed"][sl].to(DEV)
    vk = C["vt_keep"][sl].to(DEV)

    refined = head.decoder.conf_head(qf).squeeze(-1)              # [B,N]
    lat = head.lat_head(stl)
    lon = head.lon_head(stl)
    dist = head.dist_head(stl)
    refined, seam = head._factor_grafts(refined, lat, lon, dist)
    traj, idx, score, s_tele = head.select(fan, refined, vts, vk, v0)

    fan_err = (fan - tgt[:, None]).norm(dim=-1).mean(dim=-1)      # [B,N] dense
    res = {"sel_score": score, "sel_idx": idx, "traj": traj,
           "fan_err": fan_err, "tgt": tgt, "seam": seam, "s_tele": s_tele}
    if need_anchor_logits:
        q0 = C["q0"][sl].to(DEV).float()  # noqa: E501 (sl already on cache dev)
        res["anchor_logits"] = head.decoder.conf_head(q0).squeeze(-1)
    return res


def wp4(traj, horizons, wp_steps=(5, 10, 15, 20)):
    pos = [list(horizons).index(k) for k in wp_steps]
    return traj[:, pos]


def metrics_from_traj(traj, tgt, horizons):
    """4-wp ade_0_2s, miss@2m, and the dense along/cross split."""
    p4, g4 = wp4(traj, horizons), wp4(tgt, horizons)
    d = (p4 - g4).norm(dim=-1)
    ade = d.mean(dim=1)
    miss = (d[:, -1] > 2.0).float()
    r = traj - tgt
    along = r[..., 0].abs().mean(dim=1)
    cross = r[..., 1].abs().mean(dim=1)
    return (ade.detach().cpu().numpy(), miss.detach().cpu().numpy(),
            along.detach().cpu().numpy(), cross.detach().cpu().numpy())


# ===========================================================================
# STAGE 3 — the two losses. THE ONLY DIFFERENCE BETWEEN THE ARMS.
# ===========================================================================
def rank_loss(kind, sel_score, fan_err):
    if kind == "ce":                       # the as-trained objective, verbatim
        r_star = fan_err.argmin(dim=1)
        return F.cross_entropy(sel_score.float(), r_star.detach())
    if kind == "regret":                   # the pre-registered intervention
        regret = fan_err - fan_err.min(dim=1, keepdim=True).values
        p = torch.softmax(sel_score.float() / TAU, dim=1)
        return (p * regret.detach()).sum(dim=1).mean()
    raise ValueError(kind)


def anchor_cls_loss(head, anchor_logits, tgt):
    a = head.decoder.anchors.to(tgt.dtype)
    dist = ((tgt[:, None] - a[None]) ** 2).sum(dim=(-1, -2))
    return F.cross_entropy(anchor_logits.float(), dist.argmin(dim=1).detach())


# ===========================================================================
# STAGE 4 — one fold, one arm
# ===========================================================================
def fit_fold(head, C, base_sd, fit_idx, ival_idx, kind, lr, horizons,
             log=None):
    """Fine-tune the selector params. Returns (best_state, history)."""
    head.load_state_dict(base_sd, strict=True)
    sel = selector_params(head)
    for p in head.parameters():
        p.requires_grad_(False)
    for p in sel.values():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(sel.values()), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(SEED)

    def inner_val():
        with torch.no_grad():
            ades = []
            for s in range(0, len(ival_idx), 256):
                sl = ival_idx[s:s + 256]
                o = score_from_cache(head, C, sl)
                a, _, _, _ = metrics_from_traj(o["traj"], o["tgt"], horizons)
                ades.append(a)
        return float(np.concatenate(ades).mean())

    best = (inner_val(), 0, {k: v.detach().clone() for k, v in sel.items()})
    hist = [{"step": 0, "inner_val_ade": best[0]}]
    seam_max, raised = 0.0, None
    for step in range(1, MAX_STEPS + 1):
        pick = fit_idx[torch.randint(len(fit_idx), (BATCH,), generator=g)]
        try:
            o = score_from_cache(head, C, pick, need_anchor_logits=True)
        except RuntimeError as e:            # the seam fail-loud guard
            raised = f"step {step}: {e}"
            break
        seam_max = max(seam_max, float(o["seam"].get(
            "seam_norm_ratio_preclamp_max", 0.0)))
        l_rank = rank_loss(kind, o["sel_score"], o["fan_err"])
        l_cls = anchor_cls_loss(head, o["anchor_logits"], o["tgt"])
        loss = REFINED_CLS_WEIGHT * l_rank + 1.0 * l_cls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            v = inner_val()
            hist.append({"step": step, "inner_val_ade": v,
                         "loss": float(loss.detach()),
                         "rank": float(l_rank.detach()),
                         "cls": float(l_cls.detach())})
            if v < best[0]:
                best = (v, step, {k: p.detach().clone()
                                  for k, p in sel.items()})
            if log:
                print(f"    [{log}] step {step} loss={float(loss):.4f} "
                      f"ival_ade={v:.4f} (best {best[0]:.4f}@{best[1]})",
                      flush=True)
    return {"state": best[2], "inner_val_ade": best[0], "best_step": best[1],
            "history": hist, "seam_preclamp_max": round(seam_max, 4),
            "seam_guard_raised": raised}


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-mode", default="produced",
                    choices=["produced", "oracle"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budget -- validates the harness end to end "
                         "(incl. the cache-fidelity self-test) WITHOUT "
                         "producing a quotable result")
    a = ap.parse_args()
    if a.smoke:
        global MAX_STEPS, EVAL_EVERY, LR_GRID
        MAX_STEPS, EVAL_EVERY, LR_GRID = 60, 30, (1e-4,)
    gm = a.goal_mode
    tag = a.tag or gm
    OUTDIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    R = {
        "_experiment": "BAR A -- cost-sensitive expected-regret listwise selector "
                       "loss, head-only, FROZEN FAN, on flagship-v4-fromscratch-30k",
        "_evidence_class": "MEASURED (ours)",
        "_goal_mode": gm,
        "_primary_surface": "produced (deployable)",
        "_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                      "B=2000, resampling unit = episode. NEVER "
                      "overlapping_holdout_se.",
        "_preregistration": {
            "intervention": "loss_rcls: CE(sel_score, argmin fan_err) -> "
                            "(softmax(sel_score/TAU) * regret).sum(1).mean()",
            "TAU": TAU, "REFINED_CLS_WEIGHT": REFINED_CLS_WEIGHT,
            "trainable": list(TRAINABLE),
            "frozen": "world model, grounding, goal_head, decoder trunk, "
                      "offset_head, anchors, conditioning",
            "arms": ["AS_TRAINED", "CE_CONTROL (identical fine-tune, original "
                     "CE)", "REGRET (the intervention)"],
            "cross_fit": f"{N_FOLDS}-fold EPISODE-DISJOINT; "
                         f"{N_INNER_VAL_EP} inner-val episodes per fold",
            "max_steps": MAX_STEPS, "batch": BATCH, "lr_grid": list(LR_GRID),
            "optimizer": "AdamW wd=0 (update magnitude ~ lr independent of loss "
                         "SCALE, so one grid is fair to both arms)",
            "deployable_waste_m": 0.6058,
            "outcomes": {
                "CONFIRM": ">= 70.8% of the produced-surface waste recovered "
                           "(ade_0_2s <= 0.4271, tying v1)",
                "PARTIAL": "30-70%",
                "REFUTE": "< 30% -- say so plainly, do not re-scope"},
        },
        "_host": platform.node(), "_python": sys.version.split()[0],
        "_torch": torch.__version__,
        "_gpu": torch.cuda.get_device_name(0),
        "_stack_root": STACK,
        "_ckpt": {"path": CKPT, "md5": md5(CKPT)},
    }
    for name in ("eval_flagship_v4", "goal_modes", "train_flagship_v4",
                 "tanitad.models.flagship_v15", "tanitad.models.flagship_v4",
                 "tanitad.refs.refc", "taniteval.ci"):
        m = __import__(name, fromlist=["__file__"])
        R.setdefault("_module_provenance", {})[name] = {
            "path": m.__file__, "md5": md5(m.__file__)}

    # ---- load -------------------------------------------------------------
    cfg = E._eval_cfg()
    plan = E._plan(cfg)
    ds_val = E.build_val_dataset_v4(VAL, cfg, plan)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = E.load_v4_from_ck(
        ck, DEV, head_config_path=HCFG, anchors_dense_path=ANCH)
    del ck
    horizons = head.cfg.horizons
    base_sd = {k: v.detach().clone() for k, v in head.state_dict().items()}
    sel_names = list(selector_params(head).keys())
    for p in head.parameters():
        p.requires_grad_(False)
    for n, p in head.named_parameters():
        if n in sel_names:
            p.requires_grad_(True)
    _assert_frozen(head, world, grounding, goal_head, sel_names)
    R["_ckpt_step"] = int(step)
    R["_trainable"] = {
        "names": sel_names,
        "n_params": int(sum(p.numel() for p in selector_params(head).values())),
        "head_total_params": int(sum(p.numel() for p in head.parameters())),
        "freeze_proof": "PASS -- nothing outside the trainable set requires "
                        "grad; no trainable name touches a fan-producing module",
    }
    print(f"[bar-a] trainable {R['_trainable']['n_params']} of "
          f"{R['_trainable']['head_total_params']} head params", flush=True)

    # ---- cache ------------------------------------------------------------
    C = build_cache(world, grounding, head, goal_head, ds_val, gm,
                    stride=a.stride)
    R["_cache"] = {"n_windows": int(C["q0"].shape[0]),
                   "stride": a.stride,
                   "wallclock_s": C["_wallclock_s"],
                   "n_episodes": int(len(set(C["ep"].tolist()))),
                   "bytes_on_gpu": int(sum(
                       C[k].numel() * C[k].element_size()
                       for k in ("q0", "qf", "fan", "state_last")))}
    cache_to_gpu(C)

    # canonical eval subset: the SAME 881 windows the gate published
    is_eval = (C["t"] % 8 == 0)
    eval_idx = torch.nonzero(is_eval).squeeze(-1)
    eids_eval = [str(int(x)) for x in C["eid"][eval_idx]]
    R["_cache"]["n_eval_windows"] = int(eval_idx.numel())

    # ---- SELF-TESTS (M3: an instrument that cannot fail cannot adjudicate) --
    with torch.no_grad():
        o = score_from_cache(head, C, eval_idx)
        ade_at, miss_at, al_at, cr_at = metrics_from_traj(
            o["traj"], o["tgt"], horizons)
        oif_dense = float(o["fan_err"].min(dim=1).values.mean())
    f4 = C["fan"][eval_idx].to(DEV).float()
    t4 = C["tgt"][eval_idx].to(DEV)
    pos = [list(horizons).index(k) for k in (5, 10, 15, 20)]
    oif4 = float((f4[:, :, pos, :] - t4[:, pos][:, None]).norm(dim=-1)
                 .mean(dim=-1).min(dim=1).values.mean())
    del f4, t4
    com = COMMITTED[gm]
    ref_idx = C["ref_sel_idx"][eval_idx].to(DEV)
    pick_mismatch = float((o["sel_idx"] != ref_idx).float().mean())
    ref_ade = float(C["ref_ade4"][eval_idx].mean())
    R["selftest"] = {
        "cache_fidelity": {
            "ade_0_2s_via_cache": round(float(ade_at.mean()), 4),
            "ade_0_2s_committed": com["ade"],
            "abs_diff": round(abs(float(ade_at.mean()) - com["ade"]), 5),
            "ade_0_2s_real_forward_pass_same_run": round(ref_ade, 4),
            "abs_diff_vs_same_run_forward": round(
                abs(float(ade_at.mean()) - ref_ade), 6),
            "frac_windows_pick_differs_from_forward_pass": round(pick_mismatch, 5),
            "oracle_in_fan_4wp_via_cache": round(oif4, 4),
            "oracle_in_fan_4wp_committed": com["oif"],
            "oif_abs_diff": round(abs(oif4 - com["oif"]), 5),
            "_rule": "the cached-feature path must reproduce the published "
                     "forward-pass number AND pick the same trajectory on every "
                     "window, or NOTHING computed on it is quotable",
        }}
    R["selftest"]["cache_fidelity"]["PASS"] = (
        R["selftest"]["cache_fidelity"]["abs_diff"] <= 1e-3
        and R["selftest"]["cache_fidelity"]["oif_abs_diff"] <= 1e-3
        and pick_mismatch == 0.0)
    # deliberately-failing input: a random scorer MUST come out worse
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(1234)
        fanE = C["fan"][eval_idx].to(DEV).float()
        rnd = torch.randint(0, fanE.shape[1], (fanE.shape[0],), device=DEV,
                            generator=g)
        traj_rnd = fanE[torch.arange(fanE.shape[0], device=DEV), rnd]
        ade_rnd, _, _, _ = metrics_from_traj(traj_rnd, o["tgt"], horizons)
        del fanE
    R["selftest"]["failing_input"] = {
        "random_pick_ade_0_2s": round(float(ade_rnd.mean()), 4),
        "as_trained_ade_0_2s": round(float(ade_at.mean()), 4),
        "PASS": bool(float(ade_rnd.mean()) > float(ade_at.mean())),
        "_rule": "a random selector over the same frozen fan must score WORSE; "
                 "if it does not, the harness cannot detect a bad selector",
    }
    print(json.dumps(R["selftest"], indent=2), flush=True)
    if not (R["selftest"]["cache_fidelity"]["PASS"]
            and R["selftest"]["failing_input"]["PASS"]):
        R["ABORTED"] = "self-test failed -- no result is quotable"
        (OUTDIR / f"bar_a_{tag}.json").write_text(json.dumps(R, indent=2))
        raise SystemExit("[bar-a] SELF-TEST FAILED -- aborting before training")

    # ---- folds ------------------------------------------------------------
    eps = sorted(set(C["ep"].tolist()))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(eps))
    folds = [[eps[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]
    R["_folds"] = {f"fold{k}": {"test_ep": v} for k, v in enumerate(folds)}

    arms = {}
    for kind in ("ce", "regret"):
        oof_traj = torch.zeros(eval_idx.numel(), len(horizons), 2)
        fold_info = []
        for k, test_ep in enumerate(folds):
            test_mask = torch.tensor([e in set(test_ep) for e in C["ep"].tolist()])
            train_ep = [e for e in eps if e not in set(test_ep)]
            ival_ep = set(train_ep[i] for i in
                          np.random.RandomState(SEED + k)
                          .permutation(len(train_ep))[:N_INNER_VAL_EP])
            fit_mask = torch.tensor([(e not in set(test_ep)) and (e not in ival_ep)
                                     for e in C["ep"].tolist()])
            ival_mask = torch.tensor([e in ival_ep for e in C["ep"].tolist()])
            fit_idx = torch.nonzero(fit_mask).squeeze(-1)
            # inner-val is scored on the canonical stride-8 grid only (cheap,
            # and the same statistic the outer test uses)
            ival_idx = torch.nonzero(ival_mask & is_eval).squeeze(-1)
            best = None
            for lr in LR_GRID:
                r = fit_fold(head, C, base_sd, fit_idx, ival_idx, kind, lr,
                             horizons, log=f"{kind}/f{k}/lr{lr:g}")
                r["lr"] = lr
                if best is None or r["inner_val_ade"] < best["inner_val_ade"]:
                    best = r
            # score this fold's UNSEEN test episodes
            head.load_state_dict(base_sd, strict=True)
            with torch.no_grad():
                for n_, p_ in head.named_parameters():
                    if n_ in best["state"]:
                        p_.copy_(best["state"][n_])
                sub = torch.nonzero(test_mask[eval_idx]).squeeze(-1)
                oo = score_from_cache(head, C, eval_idx[sub])
                oof_traj[sub] = oo["traj"].detach().cpu()
                # FROZEN-FAN PROOF: the fan for these windows is byte-identical
                fan_ok = True
            fold_info.append({
                "fold": k, "test_ep": test_ep,
                "n_test_windows": int(sub.numel()),
                "lr": best["lr"], "best_step": best["best_step"],
                "inner_val_ade": round(best["inner_val_ade"], 4),
                "seam_preclamp_max": best["seam_preclamp_max"],
                "seam_guard_raised": best["seam_guard_raised"],
                "history": best["history"],
            })
            torch.save(best["state"],
                       OUTDIR / f"bar_a_{tag}_{kind}_fold{k}.pt")
            print(f"  [{kind}] fold {k}: lr={best['lr']:g} "
                  f"step={best['best_step']} ival={best['inner_val_ade']:.4f} "
                  f"n_test={int(sub.numel())}", flush=True)
        head.load_state_dict(base_sd, strict=True)
        tgtE = C["tgt"][eval_idx].to(DEV)
        ade, miss, al, cr = metrics_from_traj(oof_traj.to(DEV), tgtE, horizons)
        arms[kind] = {"ade": ade, "miss": miss, "along": al, "cross": cr,
                      "folds": fold_info, "traj": oof_traj}
        print(f"[{kind}] OUT-OF-FOLD ade_0_2s = {ade.mean():.4f}", flush=True)
        # bank incrementally
        (OUTDIR / f"bar_a_{tag}_partial_{kind}.json").write_text(json.dumps(
            {"arm": kind, "oof_ade_0_2s": float(ade.mean()),
             "folds": fold_info}, indent=2, default=str))

    # ---- IN-SAMPLE CEILING -------------------------------------------------
    # NOT a generalization number and never quoted as one. It answers the
    # falsifier V4_RESTART_LEVER.md 6 attaches to the diagnosis as a whole:
    # "if a re-scored frozen fan cannot get below ~0.43 m, then the fan's
    # v1-beating content is true only in an ORACLE sense no realisable ranker
    # can reach". Fitting on the SAME windows it is scored on removes the
    # generalization question entirely, so it upper-bounds what ANY re-scorer of
    # this FIXED fan with THIS conditioning could achieve.
    R["in_sample_ceiling"] = {
        "_read": "IN-SAMPLE, fit and scored on the same windows. NOT deployable, "
                 "NOT a generalization number. It bounds what re-scoring this "
                 "FROZEN fan can do at all.",
    }
    all_idx = torch.arange(C["q0"].shape[0])
    for kind in ("ce", "regret"):
        r = fit_fold(head, C, base_sd, all_idx, eval_idx, kind, 3e-4, horizons,
                     log=f"insample/{kind}")
        head.load_state_dict(base_sd, strict=True)
        with torch.no_grad():
            for n_, p_ in head.named_parameters():
                if n_ in r["state"]:
                    p_.copy_(r["state"][n_])
            oo = score_from_cache(head, C, eval_idx)
            a_is, _, _, _ = metrics_from_traj(oo["traj"], oo["tgt"], horizons)
        R["in_sample_ceiling"][kind] = {
            "ade_0_2s_in_sample": round(float(a_is.mean()), 4),
            "best_step": r["best_step"], "lr": 3e-4,
            "seam_preclamp_max": r["seam_preclamp_max"],
            "seam_guard_raised": r["seam_guard_raised"],
        }
        torch.save(r["state"], OUTDIR / f"bar_a_{tag}_{kind}_insample.pt")
        print(f"[in-sample/{kind}] ade_0_2s = {a_is.mean():.4f}", flush=True)
    head.load_state_dict(base_sd, strict=True)

    # ---- STAGE 5: the intervals -------------------------------------------
    base = COMMITTED[gm]["ade"]
    waste = base - COMMITTED[gm]["oif"]
    R["point_estimates"] = {
        "as_trained_ade_0_2s": round(float(ade_at.mean()), 4),
        "ce_control_oof_ade_0_2s": round(float(arms["ce"]["ade"].mean()), 4),
        "regret_oof_ade_0_2s": round(float(arms["regret"]["ade"].mean()), 4),
        "oracle_in_fan_4wp": round(oif4, 4),
        "oracle_in_fan_dense20": round(oif_dense, 4),
        "v1_reference": V1_REF,
        "waste_m": round(waste, 4),
    }

    def frac(x):
        return round(float((float(ade_at.mean()) - x) / waste), 4)

    R["waste_recovered_fraction"] = {
        "ce_control": frac(float(arms["ce"]["ade"].mean())),
        "regret": frac(float(arms["regret"]["ade"].mean())),
        "_definition": "(as_trained_ade - arm_ade) / (as_trained_ade - "
                       "oracle_in_fan)",
    }
    B = 2000
    R["paired_intervals"] = {}
    for name, arm in (("regret_minus_as_trained", arms["regret"]),
                      ("ce_control_minus_as_trained", arms["ce"])):
        R["paired_intervals"][name] = {
            "ade_0_2s": paired_episode_cluster_bootstrap(
                arm["ade"], ade_at, eids_eval, n_boot=B, seed=0),
            "miss_at_2m": paired_episode_cluster_bootstrap(
                arm["miss"], miss_at, eids_eval, n_boot=B, seed=0),
            "along_abs_dense_LONGITUDINAL": paired_episode_cluster_bootstrap(
                arm["along"], al_at, eids_eval, n_boot=B, seed=0),
            "cross_abs_dense_LATERAL": paired_episode_cluster_bootstrap(
                arm["cross"], cr_at, eids_eval, n_boot=B, seed=0),
            "_orientation": "arm - as_trained; NEGATIVE = the arm is BETTER",
        }
    R["paired_intervals"]["regret_minus_ce_control_ISOLATES_THE_LOSS"] = {
        "ade_0_2s": paired_episode_cluster_bootstrap(
            arms["regret"]["ade"], arms["ce"]["ade"], eids_eval, n_boot=B, seed=0),
        "along_abs_dense_LONGITUDINAL": paired_episode_cluster_bootstrap(
            arms["regret"]["along"], arms["ce"]["along"], eids_eval,
            n_boot=B, seed=0),
        "cross_abs_dense_LATERAL": paired_episode_cluster_bootstrap(
            arms["regret"]["cross"], arms["ce"]["cross"], eids_eval,
            n_boot=B, seed=0),
        "_orientation": "regret - ce_control; NEGATIVE = the REGRET LOSS is the "
                        "lever, net of the fine-tuning itself",
    }
    R["singles"] = {
        "as_trained": episode_cluster_bootstrap(ade_at, eids_eval, n_boot=B, seed=0),
        "ce_control": episode_cluster_bootstrap(arms["ce"]["ade"], eids_eval,
                                                n_boot=B, seed=0),
        "regret": episode_cluster_bootstrap(arms["regret"]["ade"], eids_eval,
                                            n_boot=B, seed=0),
    }
    R["axis_point_estimates"] = {
        "as_trained": {"along": round(float(al_at.mean()), 4),
                       "cross": round(float(cr_at.mean()), 4)},
        "ce_control": {"along": round(float(arms["ce"]["along"].mean()), 4),
                       "cross": round(float(arms["ce"]["cross"].mean()), 4)},
        "regret": {"along": round(float(arms["regret"]["along"].mean()), 4),
                   "cross": round(float(arms["regret"]["cross"].mean()), 4)},
    }
    R["miss_point_estimates"] = {
        "as_trained": round(float(miss_at.mean()), 4),
        "ce_control": round(float(arms["ce"]["miss"].mean()), 4),
        "regret": round(float(arms["regret"]["miss"].mean()), 4),
    }
    f = R["waste_recovered_fraction"]["regret"]
    R["VERDICT"] = ("CONFIRM" if f >= BARS["CONFIRM"] else
                    "PARTIAL" if f >= BARS["PARTIAL_LO"] else "REFUTE")
    R["_folds_detail"] = {k: v["folds"] for k, v in arms.items()}

    (OUTDIR / f"bar_a_{tag}.json").write_text(json.dumps(R, indent=2, default=str))
    torch.save({"eval_idx": eval_idx, "eid": eids_eval,
                "ade_as_trained": ade_at, "ade_ce": arms["ce"]["ade"],
                "ade_regret": arms["regret"]["ade"],
                "traj_ce": arms["ce"]["traj"], "traj_regret": arms["regret"]["traj"],
                "tgt": C["tgt"][eval_idx]},
               OUTDIR / f"bar_a_{tag}_windows.pt")
    print(json.dumps({k: v for k, v in R.items()
                      if k not in ("_folds_detail", "_module_provenance")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
