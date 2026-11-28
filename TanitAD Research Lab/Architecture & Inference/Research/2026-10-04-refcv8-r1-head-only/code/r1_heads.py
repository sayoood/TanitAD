"""refcv8 R1 -- the head-only path: refcv7's OWN forward with the frozen trunk replaced by cached tensors.

THE CUT (DESIGN.md sec. 1, file:line at the launch tree fec3a0d):
  * trunk        `TimmResNetTrunk.forward_features_s8` (refc.py:4394)  -> pooled_seq [B, 8, 2048] and the stride-32
                 map, which the planner reads ONLY through `decoder.feat_proj` (refc.py:3165)  -> cached as
                 kv = feat_proj(fmap) [B, 416, 384]; `feat_proj` is FROZEN (it belongs to the trunk side of the cut);
  * perception   `RefCV3Model._bev_hook` (refc_v3.py:1836-1961, map_hires + PlannerBEVPool)  -> cached `bev_feats`
                 [B, 96, 120, 64]; `bev_tokens` are re-derived from it by the branch's own ZERO-parameter
                 `PerceptionBranch.bev_tokens` (refcv6_perception_branch.py:542-577);
  * agent slots  `core.agent_head(fmap tokens)` (refc.py:4717)  -> cached slot dict (presence/cls/box/yaw/rates).
Everything downstream -- measurement (nav enters, refc.py:4548-4604), the hierarchy hook (nav + phi_tac + g_tac,
refc_v3.py:1467-1713), the scene hook (tactical behaviour decoder: nav + max-speed + v0, refc_v3.py:1752-1829),
agent_embed, ego_hist + the residual prior, the anchored-diffusion decoder WITH its DDIM sampler (the 117-candidate
fan is nav-conditioned through `cond`, so it is RE-RUN, never cached), the selection grafts (tac8 priors, behaviour
term, nav-compliance, reach band, speed-ceiling filter) and E9 (refc_v3.py:2246-2307) -- runs UNCHANGED.

The stubs are instance-attribute overrides and are removed by `uninstall`; `RefCV3Model.forward`, `RefCModel.forward`
and `AnchoredDiffusionDecoder.forward` are the tree's own, untouched. Nothing here edits a tree file.
"""
from __future__ import annotations

import torch
from torch import nn

FMAP_HW = (13, 32)          # 416x1024 / 32 (asserted against the capture)
SLOT_KEYS = ("presence_logit", "cls_logits", "box", "yaw_vec", "rates", "yaw", "valid")


class ContigIdentity(nn.Module):
    """`feat_proj` stand-in on the stubbed path: kv arrives as a transposed view of the stub map, and the full model's
    `feat_proj` output is CONTIGUOUS -- a strided input could take a different matmul kernel and break bit-identity."""

    def forward(self, x):
        return x.contiguous()


class Feed:
    """The current batch's cached tensors, ON DEVICE: kv [B, P, D], pooled [B, W, F], bev [B, 96, X, Y],
    slots {key: [B, M, ...]}, bev_pad (or None)."""

    def __init__(self):
        self.cur = None


def forward_like_trainer(model, batch, device, mode, tr):
    """`compute_losses_v3`'s pre-forward block and its `model(...)` call, line for line
    (refc_v3_train.py:4357-4536 at fec3a0d), WITHOUT the losses. ⛔ Validated, not assumed: the capture's
    full-model forward through this function must equal the route package's banked eval_s0 (made through
    compute_losses_v3 itself) -- identity control I-0."""
    cfg = model.cfg
    core = cfg.core
    frames = tr.frames_to_device(batch["frames"], device)
    pose_last = batch["pose_last"].to(device)
    nav_cmd = batch["nav_cmd"].to(device)
    nav_args = batch["nav_args"].to(device) if "nav_args" in batch else None
    v_max_ms = v_max_valid = None
    if bool(getattr(cfg, "max_speed_input", False)) or bool(getattr(cfg, "max_speed_onehot_v6", False)):
        v_max_ms = batch["v_max_ms"].to(device)
        v_max_valid = batch["v_max_valid"].to(device)
    lan = batch["lan"].to(device) if "lan" in batch else None
    v0 = pose_last[:, 3]
    steps = core.decoder.diffusion_steps if mode == "diffusion" else 0
    if getattr(cfg, "ego_state_inject", False):
        raise SystemExit("[r1] ego_state_inject is not a refcv7 arm")
    _ag = getattr(core, "agents", None)
    if _ag is not None and getattr(_ag, "enable", False) and getattr(_ag, "oracle", False):
        raise SystemExit("[r1] --agents oracle is not refcv7")
    if getattr(model.core, "ego_hist", None) is not None:
        ph = batch["pose_hist"].to(device)
        _rpm = str(getattr(core.decoder, "residual_prior", tr._kp.RESIDUAL_PRIOR_OFF))
        _acts = (batch["actions"].to(device)
                 if (_rpm != tr._kp.RESIDUAL_PRIOR_OFF and tr._kp.needs_actions(_rpm)) else None)
        model.core.set_ego_window(ph, int(ph.shape[1]), actions=_acts)
    _pgrid = _pvalid = None
    _pbr = getattr(model, "_perception", None)
    if getattr(model, "_r1_stubbed", False):
        pass                                    # the BEV hook is the cache's; no lift geometry is read
    elif _pbr is not None and _pbr.lift is not None:
        _pgrid, _pvalid = model._lift_bank.for_episodes(batch["map_ep"], device=device)
    elif getattr(model, "_map_hires", None) is not None:
        _pgrid, _pvalid = model._lift_bank_hires.for_episodes(batch["map_ep"], device=device)
    return model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan, ego_state=None, nav_args=nav_args,
                 v_max_ms=v_max_ms, v_max_valid=v_max_valid, agent_gt=None,
                 perception_grid=_pgrid, perception_valid=_pvalid)


def install(model, feed: Feed):
    """Replace the three trunk-side producers by the cache. Returns the record needed by `uninstall`."""
    if getattr(model, "_r1_stubbed", False):
        raise RuntimeError("[r1] stubs already installed")
    core = model.core
    enc = core.encoder
    dec = core.decoder
    if not bool(getattr(enc, "s8_tap", False)):
        raise SystemExit("[r1] the refcv7 forward takes the stride-8 path; this encoder has no s8 tap")
    saved = {"feat_proj": dec.feat_proj}

    def ffs8(x, rows, already_normalised=False):
        c = feed.cur
        kv, pooled = c["kv"], c["pooled"]
        b, w = pooled.shape[:2]
        d = kv.shape[-1]
        h, ww = FMAP_HW
        if kv.shape[1] != h * ww:
            raise RuntimeError(f"[r1] kv has {kv.shape[1]} tokens, the stride-32 map has {h * ww}")
        fm_last = kv.transpose(1, 2).reshape(b, d, h, ww)
        fmap_all = fm_last.new_zeros(b, w, d, h, ww)
        fmap_all[:, -1] = fm_last               # refc.py:4408 reads [:, -1] only
        s16 = fm_last.new_zeros(b * w, 1, 1, 1)  # read only by the (stubbed) BEV hook
        s8 = fm_last.new_zeros(b, 1, 1, 1)
        return s8, s16, fmap_all.reshape(b * w, d, h, ww), pooled.reshape(b * w, -1)

    enc.forward_features_s8 = ffs8
    dec.feat_proj = ContigIdentity()            # kv is ALREADY feat_proj(fmap): the cut point

    def agent_fwd(*_a, **_k):
        return dict(feed.cur["slots"])
    core.agent_head.forward = agent_fwd

    br = model._perception

    def bev_hook_factory(cache, grid=None, valid=None):
        def bev_hook(fmap_s16, fmap_s8=None):
            c = feed.cur
            pout = {"bev_feats": c["bev"], "bev_tokens": br.bev_tokens(c["bev"]), "bev_tokens_fed": True}
            if c.get("bev_pad") is not None:
                pout["bev_pad"] = c["bev_pad"]
            return pout
        return bev_hook
    model._bev_hook = bev_hook_factory
    saved["map_hires"] = getattr(model, "_map_hires", None)
    model._r1_stubbed = True
    return saved


def uninstall(model, saved):
    core = model.core
    del core.encoder.forward_features_s8
    core.decoder.feat_proj = saved["feat_proj"]
    del core.agent_head.forward
    del model._bev_hook
    model._r1_stubbed = False


def slots_to_cache(slots: dict) -> dict:
    """The slot keys the planner side reads (`AgentTokenEmbed.forward`, refc_agents.py:341-361; `slot_pad_mask`;
    `agent_pos = box[..., :2]`, refc.py:4727). Lists (per-layer `aux`) are loss-only and dropped."""
    return {k: v for k, v in slots.items() if k in SLOT_KEYS and torch.is_tensor(v)}


# --------------------------------------------------------------------------------------------- #
# training-time switches (arms only; never used by the capture or the identity controls)        #
# --------------------------------------------------------------------------------------------- #
FROZEN_PREFIXES = ("core.encoder.", "core.agent_head.", "_perception.", "_map_hires.", "core.law_head.",
                   "core.strategic.", "core.route_head.", "core.decoder.feat_proj.")


def trainable_head_params(model):
    """The R1 trainable set: everything on the planner side of the cut (decoder incl. sampler and selection grafts,
    measurement, agent_embed, ego_hist, the core tactical heads, phi_tac / z_tac heads, the tactical behaviour
    decoder and its gate, E9) -- minus the frozen prefixes. Returns [(name, param)]."""
    out = []
    for n, p in model.named_parameters():
        if any(n.startswith(f) for f in FROZEN_PREFIXES):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)            # the eval loader returns every tensor frozen
            out.append((n, p))
    return out
