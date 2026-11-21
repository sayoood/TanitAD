"""refcv9 C8B -- the MODEL-LEVEL proofs for the three planner wires (W-MAP-1 / W-MAP-2 / W-BOX; REFCV9_DESIGN 2.7).

``test_plan_wires_r9.py`` proves the pure-tensor core. This file closes the class CLAUDE.md names -- "built, tested, and UNREACHABLE
FROM ITS CALLER" -- on a tiny REAL ``RefCV3Model`` (resnet18 @ 64x64, the banded 10 cm branch, the planner BEV pool, a real box3d
decoder, the agent head, the refcv6 tactical decoder, the refcv8 banded drivable critic) driven by the TRAINER'S OWN
``compute_losses_v3`` / ``build_optimizer``:

  1. the hop table (argparse -> attach -> forward -> loss -> optimiser) pinned by source needle, each needle MUTATED to go RED;
  2. step 0 == the OFF model bit-for-bit, and the RED arms (non-zero projection / gate init move the planner output);
  3. a NON-ZERO gradient on EVERY new parameter after ONE optimiser step past the zero-init (why a zero-init gate still gets a
     gradient: dL/dg = <dL/dq, attend(agents+box) - attend(agents)>, which depends on the INPUT, not on g; the token MLP behind the
     gate reads exactly 0.0 at step 0 -- that zero is the control -- and moves only after the gate did);
  4. THE DETACH CONTROLS: planner-loss gradient on the banded map head AND on the box head reads EXACTLY 0.0 with every wire's weight
     set non-zero; each has a RED arm (drop the detach -> the same probe reads non-zero) and a positive control (the map loss, when
     labelled, reaches the very same banded parameters);
  5. WIRING: the tokens the planner / tactical layers consume are the box3d decoder's, not the agent head's (RED: agent slots);
  6. ADMISSIBILITY (CLAUDE.md goal-input rule): re-randomising the whole tactical decoder leaves ``map_chan_1m`` and the consumed box
     tokens bit-identical (RED: a tactical logit fed into BoxTokens), plus a static identifier walk;
  7. G-EVAL: ``refcv7_loader.build_perception_block`` replays ``attach`` and the trainer's state_dict loads STRICT 0 / 0 (RED: a loader
     that does not attach);
  8. G-DVB: one ``built`` lever per flag, each catches a lie.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses as dc
import inspect
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _refcv8_rig as rig8  # noqa: E402
import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_V2  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import plan_wires_r9 as PW  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.refs import refc as refc_mod  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac  # noqa: E402
from tanitad.refs.refc_agents import AgentSeamConfig  # noqa: E402
from tanitad.train import declared_vs_built as dvb  # noqa: E402
from tanitad.train import map_banded_flag as MF  # noqa: E402
from tanitad.train import plan_wires_flag as PF  # noqa: E402

D_BEV = int(PB.BEVEncoderConfig().d_out)
NEW_PAT = ("map_chan_proj", "box_tok_r9", ".box_gate")


# =========================================================================== #
# the rig                                                                      #
# =========================================================================== #
def _argv(extra=()):
    return (["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat", "--n-anchors",
             str(len(rig8.A_LON) * len(rig8.A_LAT)), "--ego-history", "--residual-prior", "ha0_ext_pose", "--out", "_r8_rig_unused"]
            + rig8.F1_F6 + list(extra))


def _ns(map_on=True, box_on=True, **kw):
    d = dict(r9_plan_map_channels=map_on, r9_plan_box_tokens=box_on, r9_map_banded=True, bev_source="map_hires_pool", agents="head",
             w_box3d=1.0, wp_index="off", seed=0, r9_box_quality=False)
    d.update(kw)
    return argparse.Namespace(**d)


def _rig(perc="tiny", attach=True, ns=None, critic=True, seed=0):
    """-> (cfg, model, ns). ``perc``: 'tiny' (d_model 32, 12 queries), 'default' (the loader's / trainer's PerceptionBranchConfig
    defaults) or None (the model BEFORE ``_perception`` -- what ``refcv7_loader`` starts from)."""
    args = T.build_parser().parse_args(_argv())
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    cfg.tac_vocab_version = "v7.0"
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k", trunk_pretrained=False, in_channels=9,
                                  image_size=64, image_width=None, trunk_frozen_bn=True)
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    cfg.tac_decoder_v6 = True
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(d_model=32, n_layers=1, n_heads=4, ff_mult=2, d_agent=int(cfg.core.decoder.d),
                                                      d_bev=D_BEV, sources=("agent", "bev"))
    cfg.refcv8.enable = True
    if critic:
        cfg.refcv8.critic_drivable, cfg.refcv8.w_drivable, cfg.refcv8.critic_banded = True, 0.1, True
    torch.manual_seed(seed)
    model = v3.RefCV3Model(cfg)
    model.core.encoder.enable_s8_tap()
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(EXTENT_V2.x_max_m), y_half_m=float(EXTENT_V2.y_half_m), d_lift=16, d_model=16,
                            d_up=16, dilations=(1, 2), grad_ckpt=True, near_lift_x_m=10.0, near_refine_blocks=1, banded=True)
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = 1.0
    setattr(model, MF.MODEL_ATTR, True)
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank({R.CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d, model._w_tac_v6 = 0.0, 0.0, 1.0       # w_box3d 0: no box loss, so the planner is the box head's ONLY possible gradient source
    ns = ns or _ns()
    if perc == "tiny":
        pcfg = PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, bev_source="map_hires_pool", d_model=32, n_queries=40,
                                         enforce_param_band=False)
        model._perception = PB.build_perception_branch(model, pcfg)
    elif perc == "default":
        model._w_box3d = 1.0
        model._perception = _trainer_perception(model, T, ns)
    with torch.no_grad():
        model.core.decoder.anchor_controls.copy_(rig8.grid())
        g = torch.Generator().manual_seed(11)
        for n, p in model.named_parameters():       # a TRAINED near lift / refine (zero-initialised at build, so their grads would read 0)
            if n.startswith(("_map_hires.near.", "_map_hires.near_refine.")):
                p.add_(0.02 * torch.randn(p.shape, generator=g))
        for ly in model.core.decoder.layers:        # a TRAINED planner: agent_gate is zero-init, and box_gate's gradient is agent_gate * (...)
            if getattr(ly, "cross_agent", None) is not None:
                ly.agent_gate.fill_(0.5)
    if attach and perc is not None:
        PF.attach(model, ns)
    return cfg, model, ns


def _trainer_perception(model, tr, args):
    """train():9860-9874 replayed (the lines the loader's ``build_perception_block`` mirrors): config -> branch -> attach."""
    pcfg = PB.PerceptionBranchConfig(w_map=model._w_map, w_box3d=model._w_box3d, bev_source=args.bev_source,
                                     planner_crop_m=tuple(float(v) for v in (getattr(args, "bev_planner_crop_m", None) or (60.0, 16.0))))
    pcfg = dc.replace(pcfg, **tr._slot_refine_kwargs(args), query_select=str(getattr(args, "slot_query_select", "learned") or "learned"))
    return PB.build_perception_branch(model, pcfg)


def _batch(cfg, labelled=False):
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1, v7l.IGNORE_ID], dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    batch["tac_goal_y"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 8, (2, 600, 320), generator=g, dtype=torch.uint8)
    codes[:, :40] = 255
    # unlabelled: the banded map term is a COUNTED ZERO on the graph (map_fine_label False), so every gradient on the banded head
    # below is the PLANNER's -- the isolation the detach control needs
    batch.update({"map_fine": codes, "map_fine_label": torch.tensor([bool(labelled)] * 2),
                  "map_ep": torch.tensor([int(stable_episode_id(R.CLIP))] * 2)})
    return batch


def _new_names(model):
    return [n for n, _ in model.named_parameters() if any(p in n for p in NEW_PAT)]


def _group(model, *prefixes):
    return [n for n, _ in model.named_parameters() if n.startswith(prefixes)]


def _run(model, batch, seed=0, train=True):
    model.zero_grad(set_to_none=True)
    model.train(train)
    torch.manual_seed(seed)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    return losses


def _grads(model, batch, seed=0):
    losses = _run(model, batch, seed)
    losses["loss"].backward()
    return losses, {n: (0.0 if p.grad is None else float(p.grad.abs().sum())) for n, p in model.named_parameters()}


def _scalars(losses):
    return {k: v.detach() if torch.is_tensor(v) else torch.tensor(float(v)) for k, v in sorted(losses.items())
            if (torch.is_tensor(v) and v.numel() == 1) or isinstance(v, (int, float))}


def _opt_ns(**kw):
    d = dict(opt="dd", lr=1e-2, weight_decay=1e-4, encoder_lr_mult=0.5, r8_train_only_map=False)
    d.update(kw)
    return argparse.Namespace(**d)


# =========================================================================== #
# 0. OFF                                                                       #
# =========================================================================== #
def test_OFF_builds_nothing_new_and_ON_adds_exactly_the_declared_parameters():
    _, off, ns_off = _rig(attach=False, ns=_ns(False, False))
    _, on, ns_on = _rig()
    PF.pin_args(ns_off)
    assert PF.stamp(off, ns_off) is None and not _new_names(off) and not hasattr(off, PF.MAP_ATTR) and not hasattr(off, PF.BOX_ATTR)
    assert off._perception.bev_pool.map_chan_proj is None and getattr(off.core, "box_tok_r9", None) is None
    assert not any(k for k in off.state_dict() if any(p in k for p in NEW_PAT))
    new = set(on.state_dict()) - set(off.state_dict())
    assert new == {k for k in on.state_dict() if any(p in k for p in NEW_PAT)} and not (set(off.state_dict()) - set(on.state_dict()))
    n_gates = len([k for k in new if k.endswith("box_gate")])
    assert n_gates == 3 == len(_group(on, "core.decoder.layers.0.box_gate", "core.decoder.layers.1.box_gate", "tac_decoder_v6.layers.0.box_gate"))
    d_pool, d_tok, n_cls = int(on._perception.bev_pool.d_out), int(on.core.cfg.decoder.d), len(AGENT_CLASSES)
    assert sum(p.numel() for p in on._perception.bev_pool.map_chan_proj.parameters()) == PW.N_MAP_CHAN * d_pool + d_pool
    want_tok = d_tok * 32 + ((9 + n_cls) * d_tok + d_tok) + (d_tok * d_tok + d_tok) + 2 * d_tok
    assert sum(p.numel() for p in on.core.box_tok_r9.parameters()) == want_tok
    st = PF.stamp(on, ns_on)
    assert st["map_channels"]["detached"] and st["box_tokens"]["detached"] and st["box_tokens"]["planner_gates"] == 2
    assert st["box_tokens"]["tactical_gates"] == 1 and "plain focal presence" in st["box_tokens"]["ranking"]
    assert "quality-aware" in PF.stamp(on, _ns(r9_box_quality=True))["box_tokens"]["ranking"]


# =========================================================================== #
# 1. the hop table: argparse -> attach -> forward -> loss -> optimiser          #
# =========================================================================== #
HOPS = [   # (hop, file, needle, count)
    ("argparse --r9-plan-map-channels", "scripts/refc_v3_train.py", 'add_argument("--r9-plan-map-channels", action="store_true"', 1),
    ("argparse --r9-plan-box-tokens", "scripts/refc_v3_train.py", 'add_argument("--r9-plan-box-tokens", action="store_true"', 1),
    ("argv refusals", "scripts/refc_v3_train.py", "_pwf.pin_args(args)", 1),
    ("G-DVB levers", "scripts/refc_v3_train.py", "_pwf.register_dvb_levers(_dvb.register)", 1),
    ("build: attach", "scripts/refc_v3_train.py", "_pwf.attach(model, args)", 1),
    ("config.json stamp", "scripts/refc_v3_train.py", "_pwf.stamp(model, args)", 2),
    ("W-COST print", "scripts/refc_v3_train.py", "_pwf.wcost_ratio(", 1),
    ("W-MAP-1 critic widened", "tanitad/refs/refc_v3.py", "r8c.DrivableCritic(n_feats=5) if bool(getattr(r8cfg, \"critic_banded\", False))", 1),
    ("W-MAP-1 E9 feature", "tanitad/refs/refc_v3.py", "_dfeat = _pw.critic_feature_banded(fan, _mbd)", 1),
    ("W-MAP-1 term in the selection score", "tanitad/refs/refc_v3.py", "blended = blended + _dterm", 1),
    ("W-MAP-2 bev_hook passes map_banded", "tanitad/refs/refc_v3.py", 'br(fmap_s16, bev_hires=hi["map_hires_bev"],', 1),
    ("W-MAP-2 channels at 1 m", "tanitad/models/refcv6_perception_branch.py", 'out["map_chan_1m"] = _pw.channels_1m(map_banded)', 1),
    ("W-MAP-2 pool call", "tanitad/models/refcv6_perception_branch.py", 'bev_feats = self.bev_pool(bev_hires, map_chan=out["map_chan_1m"])', 1),
    ("W-MAP-2 projection added", "tanitad/models/refcv6_perception_branch.py", "out = out + self.map_chan_proj(map_chan)", 1),
    ("W-BOX tokens from box_slots", "tanitad/refs/refc.py", '_bs = (perception_out or {}).get("box_slots")', 1),
    ("W-BOX tokens built", "tanitad/refs/refc.py", "_bt, _bp = self.box_tok_r9(_bs)", 1),
    ("W-BOX appended to the agent set", "tanitad/refs/refc.py", "agent_tokens, agent_pad = torch.cat([agent_tokens, _bt], 1)", 1),
    ("W-BOX count side channel", "tanitad/refs/refc.py", "_pw_set_n_box(int(_bt.shape[1]))", 1),
    ("W-BOX planner gated mix", "tanitad/refs/refc.py", "_pw_gated_mix(lambda t, p: self._attend_agents(q, t, p, None)", 1),
    ("W-BOX tactical key set", "tanitad/refs/refcv6_tactical.py", "_kv_base = (kv[:, _keep], kv_pad[:, _keep])", 1),
    ("W-BOX tactical layer call", "tanitad/refs/refcv6_tactical.py", "q, attn = lyr(q, kv, kv_pad, cond, base=_kv_base)", 1),
    ("W-BOX tactical gated mix", "tanitad/refs/refcv6_tactical.py", "q = q + (a0 + self.box_gate * (a1 - a0))", 1),
    ("G-EVAL loader replays attach", "tanitad/eval/refcv7_loader.py", "tr._pwf.attach(model, args)", 1),
]


def _src(rel):
    return (ROOT / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


def _hops(src_of):
    return {h: src_of(f).count(n) == c for h, f, n, c in HOPS}


def test_the_hop_table_is_green_and_every_hop_mutation_goes_RED():
    cache = {f: _src(f) for f in {h[1] for h in HOPS}}
    green = _hops(cache.get)
    assert all(green.values()), [h for h, ok in green.items() if not ok]
    lines = {}
    for h, f, n, c in HOPS:
        i = cache[f].index(n)
        lines[h] = "%s:%d" % (f, cache[f][:i].count("\n") + 1)
    print("\n[hops file:line]\n" + "\n".join("  %-42s %s" % kv for kv in lines.items()))
    for h, f, n, c in HOPS:                                     # MUTATE: comment out the one line that makes the hop -> that hop reads RED
        mut = dict(cache)
        mut[f] = cache[f].replace(n, "pass", 1)
        assert not _hops(mut.get)[h], h


def test_argv_reaches_the_namespace_and_the_refusals_fire():
    p = T.build_parser()
    off = p.parse_args(_argv())
    on = p.parse_args(_argv(["--r9-plan-map-channels", "--r9-plan-box-tokens"]))
    assert off.r9_plan_map_channels is False and off.r9_plan_box_tokens is False
    assert on.r9_plan_map_channels is True and on.r9_plan_box_tokens is True
    PF.pin_args(off)
    PF.pin_args(_ns())
    for kw in (dict(map_on=True, box_on=False, r9_map_banded=False), dict(map_on=True, box_on=False, bev_source="s16_lift"),
               dict(map_on=False, box_on=True, agents="off"), dict(map_on=False, box_on=True, w_box3d=0.0),
               dict(map_on=False, box_on=True, wp_index="on")):
        with pytest.raises(SystemExit):
            PF.pin_args(_ns(**kw))


def test_the_trainers_optimiser_holds_every_new_parameter():
    _, model, _ = _rig()
    new = {n: p for n, p in model.named_parameters() if any(q in n for q in NEW_PAT)}
    assert len(new) == 12
    for kind in ("dd", "adam"):
        opt = T.build_optimizer(model, _opt_ns(opt=kind))
        held = {id(p) for g in opt.param_groups for p in g["params"]}
        assert all(id(p) in held and p.requires_grad for p in new.values()), kind


# =========================================================================== #
# 2. step 0 == OFF, bit for bit; the RED arms                                  #
# =========================================================================== #
def test_step0_is_the_OFF_model_bit_for_bit_and_a_nonzero_init_goes_RED():
    cfg, off, _ = _rig(attach=False, ns=_ns(False, False))
    _, on, _ = _rig()
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert sorted(missing) == sorted(_new_names(on)) == sorted(k for k in on.state_dict() if any(p in k for p in NEW_PAT)) and not unexpected
    batch = _batch(cfg)

    def ev(m):
        m.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            return _scalars(T.compute_losses_v3(m, batch, "cpu", mode="diffusion"))
    a, b = ev(off), ev(on)
    assert a.keys() == b.keys() and len(a) > 10
    diff = {k: float((a[k] - b[k]).abs()) for k in a if not torch.equal(a[k], b[k])}
    # ZERO-INIT: every loss scalar (traj, sel, tactical, critic, ...) is the OFF value, bit for bit. The ONE telemetry COUNTER that moves is
    # the tactical decoder's scene-token count -- by exactly the number of box tokens appended (32 of the 40 queries): it is a count, not a loss
    assert diff == {"tacv6_n_scene_mean": 32.0}, diff
    a.pop("tacv6_n_scene_mean"), b.pop("tacv6_n_scene_mean")
    ev0 = ev
    ev = lambda m: {k: v for k, v in ev0(m).items() if k != "tacv6_n_scene_mean"}
    # RED 1 (W-MAP-2): a non-zero projection init moves the planner's BEV and the tactical decoder that reads it
    with torch.no_grad():
        on._perception.bev_pool.map_chan_proj.proj.weight.fill_(0.5)
    r1 = ev(on)
    assert any(not torch.equal(a[k], r1[k]) for k in ("lat_tac", "lon_tac", "goal_tac", "tac_v6")), "projection init must move the tactical read"
    with torch.no_grad():
        on._perception.bev_pool.map_chan_proj.proj.weight.zero_()
    assert all(torch.equal(a[k], v) for k, v in ev(on).items())
    # RED 2 (W-BOX): a non-zero gate init moves the planner's losses (planner gates; MEASURED: cascade, cls, sel_v3 -- the diffusion `traj`
    # term itself does not read the agent set on this rig) and the tactical loss (tactical gate)
    with torch.no_grad():
        for ly in on.core.decoder.layers:
            ly.box_gate.fill_(0.5)
    r2 = ev(on)
    moved = [k for k in ("cascade", "cls", "sel_v3", "loss") if not torch.equal(a[k], r2[k])]
    assert moved, "a non-zero planner gate must move the planner's output"
    with torch.no_grad():
        for ly in on.core.decoder.layers:
            ly.box_gate.zero_()
        on.tac_decoder_v6.layers[0].box_gate.fill_(0.5)
    r3 = ev(on)
    assert not torch.equal(a["tac_v6"], r3["tac_v6"])


# =========================================================================== #
# 3. gradients: step 0, then ONE optimiser step past the zero-init              #
# =========================================================================== #
def test_every_new_parameter_has_a_nonzero_gradient_one_step_past_the_zero_init():
    cfg, model, _ = _rig()
    batch = _batch(cfg)
    new = _new_names(model)
    losses, g0 = _grads(model, batch)
    assert all(torch.isfinite(torch.as_tensor(float(v))) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1)
    # step 0: the gates and the 1x1 get a gradient because dL/dW depends on the INPUT (the attention-output difference / the map
    # channels), not on W itself
    for n in new:
        if n.endswith("box_gate") or "map_chan_proj" in n:
            assert g0[n] > 0.0, n
    # ... and the token MLP BEHIND the zero gate reads EXACTLY 0.0: the control (a probe that reads 0.0 here is working, not broken)
    for n in new:
        if "box_tok_r9" in n:
            assert g0[n] == 0.0, n
    # ONE optimiser step (the trainer's own AdamW), then every new parameter must carry a gradient
    opt = T.build_optimizer(model, _opt_ns())
    opt.zero_grad(set_to_none=True)
    losses = _run(model, batch)
    losses["loss"].backward()
    opt.step()
    gates = [float(p) for n, p in model.named_parameters() if n.endswith("box_gate")]
    assert len(gates) == 3 and all(g != 0.0 for g in gates), gates
    losses, g1 = _grads(model, batch)
    assert torch.isfinite(losses["loss"])
    dead = [n for n in new if not g1[n] > 0.0]
    assert not dead, dead
    print("\n[grads] step0:", {n.replace("core.", "").replace("_perception.bev_pool.", ""): round(g0[n], 6) for n in new})
    print("[grads] step1:", {n.replace("core.", "").replace("_perception.bev_pool.", ""): round(g1[n], 6) for n in new})


def test_W_MAP_1_the_banded_critic_is_trained_by_its_BCE_on_a_labelled_batch():
    cfg, model, _ = _rig()
    assert model.r8_drv.n_feats == 5
    losses, g = _grads(model, _batch(cfg, labelled=True))
    assert float(losses["n_drv"]) > 0 and float(losses["r8_drv"]) > 0 and torch.isfinite(losses["r8_drv"])
    assert model.r8_drv.mlp[0].weight.shape[1] == 5 + 8
    crit = [n for n in g if n.startswith("r8_drv.mlp")]
    assert crit and all(g[n] > 0.0 for n in crit), {n: g[n] for n in crit}
    assert g["r8_drv.w"] > 0.0                                   # the zero-init selection weight learns at step 0 (dL/dw = logsigmoid * dL/dscore)


# =========================================================================== #
# 4. THE DETACH CONTROLS                                                        #
# =========================================================================== #
def _arm_nonzero(model):
    """every wire's weight non-zero, so the detach is the ONLY thing between the planner loss and the heads."""
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        model._perception.bev_pool.map_chan_proj.proj.weight.copy_(0.5 * torch.randn(model._perception.bev_pool.map_chan_proj.proj.weight.shape, generator=g))
        for n, p in model.named_parameters():
            if n.endswith("box_gate"):
                p.fill_(0.3)
        model.r8_drv.w.fill_(0.5)


def _channels_nodetach(mb):                      # plan_wires_r9.channels_1m with the one line that makes it safe REMOVED
    import torch.nn.functional as F
    I = {c: i for i, c in enumerate(PW.MB.CHANNELS)}
    rows = []
    for k, (z, un) in enumerate(zip(mb["logits"], mb["unobs"])):
        obs = (~un).to(torch.float32)[:, None]
        f = int(round(1.0 / PW.MB.BAND_CELL_M[k]))
        val = torch.stack([torch.sigmoid(z[:, I["drv"]]), z[:, I["drv_sdf"]].clamp(-PW.MB.SDF_CLIP_M, PW.MB.SDF_CLIP_M) / PW.MB.SDF_CLIP_M,
                           torch.sigmoid(z[:, I["sdw"]]), torch.sigmoid(z[:, I["lane"]])], 1)
        num, den = F.avg_pool2d(val * obs, f), F.avg_pool2d(obs, f)
        rows.append(torch.cat([torch.where(den > 0, num / den.clamp_min(1e-6), torch.zeros_like(num)), 1.0 - den], 1))
    return torch.cat(rows, 2)


def _box_features_nodetach(slots, k=PW.BOX_K):  # plan_wires_r9.box_features with the detach REMOVED (indices stay: topk is not differentiable)
    pl = slots["presence_logit"]
    top = pl.topk(min(int(k), pl.shape[1]), dim=1).indices
    g = lambda t: t.gather(1, top[..., None].expand(-1, -1, t.shape[-1])) if t.dim() == 3 else t.gather(1, top)
    box, yv, rates, cls_p = g(slots["box"]), g(slots["yaw_vec"]), g(slots["rates"]), g(torch.softmax(slots["cls_logits"], -1))
    pres = torch.sigmoid(g(pl))
    feat = torch.cat([box[..., :2] / torch.tensor([60.0, 16.0]), box[..., 2:3] / 10.0, box[..., 3:4] / 4.0, yv, rates[..., :2] / 10.0, cls_p, pres[..., None]], -1)
    return feat, pres, top


def _critic_nodetach(paths, mb, y_half_m=None):  # critic_feature_banded with `torch.no_grad()` / `.detach()` REMOVED
    from tanitad.refs import refcv8_conditioning as r8c
    b, n, s, _ = paths.shape
    pts = r8c.footprint_corners(paths.detach()).reshape(b, n * s * 4, 2)
    p_all, d_all = torch.ones(b, n * s * 4), torch.ones(b, n * s * 4)
    ev_all = torch.zeros(b, n * s * 4, dtype=torch.bool)
    bidx = torch.arange(b)[:, None]
    for (x0, x1, cell, rows, cols), z, un in zip(PW._geoms(mb), mb["logits"], mb["unobs"]):
        yh = cols * cell / 2.0
        x, y = pts[..., 0], pts[..., 1]
        inside = (x >= x0) & (x < x1) & (y > -yh) & (y < yh)
        ix, iy = ((x - x0) / cell).floor().long().clamp(0, rows - 1), ((y + yh) / cell).floor().long().clamp(0, cols - 1)
        zz = z.permute(0, 2, 3, 1)[bidx, ix, iy].to(torch.float32)
        ev = PW._evidence_mask(inside, un[bidx, ix, iy])
        I = {c: i for i, c in enumerate(PW.MB.CHANNELS)}
        p_all = torch.where(ev, torch.sigmoid(zz[..., I["drv"]]), p_all)
        d_all = torch.where(ev, zz[..., I["drv_sdf"]].clamp(-PW.MB.SDF_CLIP_M, PW.MB.SDF_CLIP_M) / PW.MB.SDF_CLIP_M, d_all)
        ev_all = ev_all | ev
    ev3, p3, d3 = (t.reshape(b, n, -1) for t in (ev_all, p_all, d_all))
    cnt = ev3.to(torch.float32).sum(-1)
    one = torch.ones_like(cnt)
    pmean = torch.where(cnt > 0, (p3 * ev3).sum(-1) / cnt.clamp_min(1.0), one)
    dmean = torch.where(cnt > 0, (d3 * ev3).sum(-1) / cnt.clamp_min(1.0), one)
    return torch.stack([p3.amin(-1), pmean, cnt / float(ev3.shape[-1]), d3.amin(-1), dmean], -1)


def _control(monkeypatch=None, patch=None):
    cfg, model, _ = _rig()
    _arm_nonzero(model)
    banded, boxhead = _group(model, "_map_hires.banded."), _group(model, "_perception.box_dec.", "_perception.box_mem.")
    assert len(banded) == 18 and len(boxhead) > 20
    if patch:
        patch(monkeypatch)
    _, g = _grads(model, _batch(cfg))
    return model, g, banded, boxhead


def test_DETACH_CONTROL_planner_gradient_on_the_map_head_and_the_box_head_is_exactly_zero():
    model, g, banded, boxhead = _control()
    assert sum(g[n] for n in banded) == 0.0 and all(g[n] == 0.0 for n in banded)               # W-MAP-1 + W-MAP-2: the map head
    assert sum(g[n] for n in boxhead) == 0.0 and all(g[n] == 0.0 for n in boxhead)            # W-BOX: the box head (and its memory)
    # the probe is alive: the wires' OWN parameters carry the planner gradient in this very run ...
    live = [n for n in _new_names(model)] + ["r8_drv.w"] + _group(model, "r8_drv.mlp")
    assert all(g[n] > 0.0 for n in live), {n: g[n] for n in live if not g[n] > 0.0}
    # ... and the SAME banded parameters read non-zero the moment the map loss itself is on (labelled windows)
    cfg, m2, _ = _rig()
    _, g2 = _grads(m2, _batch(cfg, labelled=True))
    assert sum(g2[n] for n in banded) > 0.0 and sum(g2[n] for n in _group(m2, "_map_hires.banded.")) > 0.0


@pytest.mark.parametrize("which", ["channels_1m", "box_features", "critic_feature_banded"])
def test_RED_dropping_a_detach_makes_the_control_read_nonzero(monkeypatch, which):
    def patch(mp):
        if which == "channels_1m":
            mp.setattr(PW, "channels_1m", _channels_nodetach)
        elif which == "box_features":
            mp.setattr(PW, "box_features", _box_features_nodetach)
        else:
            mp.setattr(PW, "critic_feature_banded", _critic_nodetach)
    _, g, banded, boxhead = _control(monkeypatch, patch)
    leak = sum(g[n] for n in (boxhead if which == "box_features" else banded))
    other = sum(g[n] for n in (banded if which == "box_features" else boxhead))
    assert leak > 0.0, "the control must go RED when %s keeps its graph" % which
    assert other == 0.0                                                                         # and the OTHER head stays clean (the arms are independent)


# =========================================================================== #
# 5. WIRING: the consumed tokens are the box3d decoder's                        #
# =========================================================================== #
def _probe(model, batch, monkeypatch):
    """one eval forward through the trainer's loss, recording what each consumer actually received."""
    rec = {"consumed": [], "tac_kv": [], "agent_slots": None, "box_slots": None, "map_chan_1m": None, "lat": None}
    orig = refc_mod._pw_gated_mix

    def spy(attend, tok, pad, n_box, gate):
        rec["consumed"].append((tok[:, tok.shape[1] - n_box:].detach().clone(), int(n_box)))
        return orig(attend, tok, pad, n_box, gate)
    monkeypatch.setattr(refc_mod, "_pw_gated_mix", spy)
    def _perc_hook(m, i, o):                      # returns None: a hook that returned a value would REPLACE the module output
        rec["box_slots"] = {k: v.detach().clone() for k, v in o["box_slots"].items() if torch.is_tensor(v)}
        rec["map_chan_1m"] = o["map_chan_1m"].detach().clone()
    hs = [model.core.agent_head.register_forward_hook(lambda m, i, o: rec.__setitem__("agent_slots", {k: v.detach().clone() for k, v in o.items() if torch.is_tensor(v)})),
          model._perception.register_forward_hook(_perc_hook),
          model.tac_decoder_v6.layers[0].register_forward_pre_hook(
              lambda m, a, kw: rec["tac_kv"].append((int(a[1].shape[1]), None if kw.get("base") is None else int(kw["base"][0].shape[1]))), with_kwargs=True),
          model.tac_decoder_v6.register_forward_hook(lambda m, i, o: rec.__setitem__("lat", o["lat_logits"].detach().clone()))]
    try:
        model.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    finally:
        for h in hs:
            h.remove()
        monkeypatch.setattr(refc_mod, "_pw_gated_mix", orig)
    return rec


def _wiring_ok(model, rec):
    """the planner layers consumed BoxTokens(box3d slots) -- and not BoxTokens(agent head slots)."""
    ref_box = PW.BoxTokens.forward(model.core.box_tok_r9, rec["box_slots"])[0]
    ref_agent = PW.BoxTokens.forward(model.core.box_tok_r9, rec["agent_slots"])[0]
    assert ref_box.shape[1] == 32
    ok_planner = all(torch.equal(t, ref_box) and n == 32 for t, n in rec["consumed"]) and not torch.equal(ref_box, ref_agent)
    assert len(rec["consumed"]) >= 2 and len(rec["consumed"]) % 2 == 0          # 2 planner layers x the denoising passes
    ok_tactical = rec["tac_kv"] and all(base is not None and kv - base == 32 for kv, base in rec["tac_kv"])
    return bool(ok_planner), bool(ok_tactical)


def test_WIRING_box_tokens_come_from_the_box3d_decoder_and_a_token_set_from_the_agent_head_goes_RED(monkeypatch):
    cfg, model, _ = _rig()
    batch = _batch(cfg)
    rec = _probe(model, batch, monkeypatch)
    assert _wiring_ok(model, rec) == (True, True), rec["tac_kv"]
    # the agent head's slots exist, have the same format, and are a DIFFERENT set (separate parameters) -- the RED arm is not vacuous
    assert rec["agent_slots"]["presence_logit"].shape[0] == 2 and not torch.equal(rec["agent_slots"]["presence_logit"][:, :12], rec["box_slots"]["presence_logit"])
    # RED: BoxTokens fed the AGENT HEAD's slots (what K3 would do, un-asked) -> the planner consumes the wrong set -> the wiring check fails
    holder = {}
    h = model.core.agent_head.register_forward_hook(lambda m, i, o: holder.__setitem__("s", o))
    orig = PW.BoxTokens.forward
    monkeypatch.setattr(PW.BoxTokens, "forward", lambda self, slots: orig(self, holder["s"] if "s" in holder else slots))
    rec2 = _probe(model, batch, monkeypatch)
    h.remove()
    monkeypatch.setattr(PW.BoxTokens, "forward", orig)
    ref_box = orig(model.core.box_tok_r9, rec2["box_slots"])[0]
    assert not all(torch.equal(t, ref_box) for t, _ in rec2["consumed"]), "agent-head tokens must NOT equal the box3d tokens"


# =========================================================================== #
# 6. ADMISSIBILITY (no tactical class posterior / argmax / embedding reaches W-MAP-2 or W-BOX)                #
# =========================================================================== #
FORBIDDEN = re.compile(r"^(lat|lon)(_|$)|^(tac|goal|situation|maneuv)|tacv6|_prior$", re.I)


def _identifiers(src):
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant):
            doc.add(id(node.body[0].value))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc and len(node.value) < 40:
            out.add(node.value)
    return out


def test_ADMISSIBILITY_static_the_wire_code_names_no_tactical_quantity_and_takes_only_perception_arguments():
    ids = _identifiers(_src("tanitad/models/plan_wires_r9.py"))
    assert {i for i in ids if FORBIDDEN.match(i)} == set()
    sig = lambda f: list(inspect.signature(f).parameters)
    assert sig(PW.channels_1m) == ["mb"] and sig(PW.critic_feature_banded) == ["paths", "mb", "y_half_m"]
    assert sig(PW.box_features) == ["slots", "k"] and sig(PW.BoxTokens.forward) == ["self", "slots"]
    # MUTATION of the walk itself: a source that reads a tactical logit is caught
    bad = _src("tanitad/models/plan_wires_r9.py") + '\n\ndef leak(slots):\n    return slots["lat_logits"]\n'
    assert {i for i in _identifiers(bad) if FORBIDDEN.match(i)} == {"lat_logits"}


def _walk(model, batch, monkeypatch):
    p0 = _probe(model, batch, monkeypatch)
    p1 = _probe(model, batch, monkeypatch)
    g = torch.Generator().manual_seed(77)
    with torch.no_grad():                          # RE-RANDOMISE THE WHOLE TACTICAL DECODER (heads, queries, norms: everything)
        for p in model.tac_decoder_v6.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
    p2 = _probe(model, batch, monkeypatch)
    p3 = _probe(model, batch, monkeypatch)
    assert not torch.equal(p1["lat"], p3["lat"]), "the tactical posterior must actually have changed"
    return p1, p3


def test_ADMISSIBILITY_dynamic_a_new_tactical_decoder_leaves_the_map_channels_and_box_tokens_bit_identical(monkeypatch):
    cfg, model, _ = _rig()
    _arm_nonzero(model)
    p1, p3 = _walk(model, _batch(cfg), monkeypatch)
    assert torch.equal(p1["map_chan_1m"], p3["map_chan_1m"])
    assert all(torch.equal(a[0], b[0]) for a, b in zip(p1["consumed"], p3["consumed"])) and len(p1["consumed"]) >= 2
    assert all(torch.equal(p1["box_slots"][k], p3["box_slots"][k]) for k in p1["box_slots"])


def test_RED_a_tactical_logit_fed_into_BoxTokens_makes_the_walk_FAIL(monkeypatch):
    cfg, model, _ = _rig()
    _arm_nonzero(model)
    holder = {}
    model.tac_decoder_v6.register_forward_hook(lambda m, i, o: holder.__setitem__("lat", o["lat_logits"].detach().clone()))
    orig = PW.BoxTokens.forward

    def leaky(self, slots):
        tok, pad = orig(self, slots)
        return (tok + holder["lat"].mean() if "lat" in holder else tok), pad            # the previous forward's class posterior leaks in
    monkeypatch.setattr(PW.BoxTokens, "forward", leaky)
    p1, p3 = _walk(model, _batch(cfg), monkeypatch)
    assert not all(torch.equal(a[0], b[0]) for a, b in zip(p1["consumed"], p3["consumed"])), "the walk must catch a tactical leak"


# =========================================================================== #
# 7. G-EVAL: the loader builds the wired model, strict 0 / 0                    #
# =========================================================================== #
def _geval_args(on=True):
    extra = ["--bev-source", "map_hires_pool", "--r9-map-banded", "--agents", "head", "--w-box3d", "1.0"]
    if on:
        extra += ["--r9-plan-map-channels", "--r9-plan-box-tokens"]
    return T.build_parser().parse_args(_argv(extra))


def _trainer_stamp(model, args):
    pc = model._perception.cfg
    return {**pc.as_dict(), "equalize_bottom_rows": 0, "branch_params": model._perception.param_breakdown(),
            **({} if PF.stamp(model, args) is None else {"plan_wires": PF.stamp(model, args)})}


@pytest.mark.parametrize("on", [True, False])
def test_GEVAL_the_loader_replays_attach_and_the_strict_load_reads_0_missing_0_unexpected(on):
    from tanitad.eval import refcv7_loader as L
    tr = L.trainer()
    args = _geval_args(on)
    ns = _ns(on, on)
    # the TRAINER's model: train():9860-9874 (config -> branch -> attach), then the config.json stamp
    cfg, tmodel, _ = _rig(perc="default", attach=True, ns=ns)
    stamp = _trainer_stamp(tmodel, args)
    # the LOADER's model: the same pre-perception build, then the loader's own block
    _, lmodel, _ = _rig(perc=None)
    lmodel._w_box3d = 1.0
    block = L.build_perception_block(tr, lmodel, args, "cpu")
    res = lmodel.load_state_dict(tmodel.state_dict(), strict=True)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []                          # strict 0 / 0
    assert ("plan_wires" in block) == on and ("plan_wires" in stamp) == on
    assert any("box_tok_r9" in k for k in lmodel.state_dict()) == on
    out = L.stamp_checks(lmodel, {"refcv6_perception": stamp}, {"perception": block}, tr)
    assert out["n_compared"] > 5
    if on:
        assert out["refcv6_perception.plan_wires"]["equal"] is True
        # RED 1: a loader that does NOT attach (the pre-C8 loader) -> the trainer's checkpoint does not load, and the stamp contradicts it
        _, bare, _ = _rig(perc=None)
        bare._w_box3d = 1.0
        saved = (PF.attach, PF.stamp)                # the pre-C8 loader: neither the attach nor the stamp call exists
        PF.attach, PF.stamp = (lambda model, a: None), (lambda model, a: None)
        try:
            blk0 = L.build_perception_block(tr, bare, args, "cpu")
        finally:
            PF.attach, PF.stamp = saved
        r0 = bare.load_state_dict(tmodel.state_dict(), strict=False)
        assert any("box_tok_r9" in k for k in r0.unexpected_keys) and any("map_chan_proj" in k for k in r0.unexpected_keys)
        with pytest.raises(SystemExit, match="plan_wires"):
            L.stamp_checks(bare, {"refcv6_perception": stamp}, {"perception": blk0}, tr)
        # RED 2: the stamp says wires, the rebuild has none (key present on one side only) -> REFUSED by name
        blk1 = {k: v for k, v in block.items() if k != "plan_wires"}
        with pytest.raises(SystemExit, match="plan_wires"):
            L.stamp_checks(lmodel, {"refcv6_perception": stamp}, {"perception": blk1}, tr)
    else:
        with pytest.raises(SystemExit, match="plan_wires"):                                             # OFF rebuild vs an ON stamp
            on_stamp = {**stamp, "plan_wires": {"map_channels": {}}}
            L.stamp_checks(lmodel, {"refcv6_perception": on_stamp}, {"perception": block}, tr)


# =========================================================================== #
# 8. G-DVB                                                                      #
# =========================================================================== #
def test_GDVB_each_flag_has_a_built_lever_and_the_check_catches_a_lie():
    PF.register_dvb_levers(dvb.register)
    for d in (PF.MAP_DEST, PF.BOX_DEST):
        assert dvb.REGISTRY[d].kind == "built", d
    _, on, ns_on = _rig()
    _, off, ns_off = _rig(attach=False, ns=_ns(False, False))
    for d in (PF.MAP_DEST, PF.BOX_DEST):
        assert dvb.REGISTRY[d].check(on, ns_on) == [] and dvb.REGISTRY[d].check(off, ns_off) == [], d
        assert dvb.REGISTRY[d].check(off, ns_on) != [], d              # RED: declared on, not built
        assert dvb.REGISTRY[d].check(on, ns_off) != [], d              # RED: built, not declared
    # RED: a consuming layer without its gate
    on.core.decoder.layers[0].box_gate = None
    assert dvb.REGISTRY[PF.BOX_DEST].check(on, ns_on) != []
