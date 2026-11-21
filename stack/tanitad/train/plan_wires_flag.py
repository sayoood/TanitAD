"""refcv9 C8 -- the planner wires' contract with the trainer (REFCV9_DESIGN 2.7). Shape of ``map_banded_flag.py`` / ``box_b3_flag.py``.

  W-MAP-1  ``--r8-critic-drivable`` (exists) UNDER ``--r9-map-banded``: no new flag; ``map_banded_flag.dvb_check`` pins the 5-feature critic.
  W-MAP-2  ``--r9-plan-map-channels``  drv / drv_sdf / sdw / lane / unobs (DETACHED, 1 m, 100 m x +-30 m) -> zero-init 1x1 -> the planner BEV pool.
  W-BOX    ``--r9-plan-box-tokens``    top-32 box3d slots -> tokens appended to the planner's AND the tactical decoder's agent-token set,
                                       behind a zero-init gate per consuming layer. K3 (retire the agent head) is NOT built.

``pin_args`` (argv refusals, no-op when OFF) | ``attach`` (build-time: the modules + gates; OFF builds nothing) | ``dvb_check`` / ``register_dvb_levers``
(G-DVB ``built`` levers) | ``stamp`` (``config.json["perception"]["plan_wires"]``, absent when OFF) | ``wcost_ratio`` (the printed step-0 critic-BCE /
trajectory-loss ratio the MM sets ``--w-r8-drivable`` from).
"""
from __future__ import annotations

MAP_FLAG, MAP_DEST = "--r9-plan-map-channels", "r9_plan_map_channels"
BOX_FLAG, BOX_DEST = "--r9-plan-box-tokens", "r9_plan_box_tokens"
MAP_ATTR, BOX_ATTR = "_r9_plan_map_channels", "_r9_plan_box_tokens"


def map_on(args) -> bool:
    return bool(getattr(args, MAP_DEST, False))


def box_on(args) -> bool:
    return bool(getattr(args, BOX_DEST, False))


def pin_args(args) -> None:
    """argv refusals. Both OFF: nothing is checked."""
    bad = []
    if map_on(args):
        if not bool(getattr(args, "r9_map_banded", False)):
            bad.append(f"{MAP_FLAG} needs --r9-map-banded (the channels ARE the banded head's drv / drv_sdf / sdw / lane / unobs)")
        if str(getattr(args, "bev_source", "s16_lift") or "s16_lift") != "map_hires_pool":
            bad.append(f"{MAP_FLAG} needs --bev-source map_hires_pool (the channels are added to THAT pool; with no pool they would reach no consumer)")
    if box_on(args):
        if str(getattr(args, "agents", "off") or "off") != "head":
            bad.append(f"{BOX_FLAG} needs --agents head: the tokens are appended to the agent-token set (K3, retiring the agent head, is not built)")
        if not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
            bad.append(f"{BOX_FLAG} needs --w-box3d > 0: the tokens are built from the box3d slots (never from the agent head's)")
        if str(getattr(args, "wp_index", "off") or "off") == "on":
            bad.append(f"{BOX_FLAG} with --wp-index on: the WP-B waypoint address is computed over the agent slots only; the gated mix would "
                       "read an index for the wrong key set (named blocker, not built)")
    if bad:
        raise SystemExit("[plan-wires] refused:\n  - " + "\n  - ".join(bad))


def attach(model, args) -> None:
    """Build the wires' modules on the BUILT model (call after ``model._perception`` exists). Draws inside ``fork_rng`` so the global RNG
    stream after the build is what the OFF build leaves."""
    import torch
    from torch import nn
    from tanitad.models import plan_wires_r9 as PW
    if not (map_on(args) or box_on(args)):
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(getattr(args, "seed", 0) or 0) + 9_008)
        if map_on(args):
            pool = model._perception.bev_pool
            if pool is None:
                raise SystemExit("[plan-wires] no planner BEV pool is built (--bev-source map_hires_pool)")
            pool.map_chan_proj = PW.MapChanProj(int(pool.d_out), pool.dst_grid)
            setattr(model, MAP_ATTR, True)
        if box_on(args):
            from tanitad.models.agent_slots import AGENT_CLASSES
            core = model.core
            if core.agent_head is None or getattr(model._perception, "box_dec", None) is None:
                raise SystemExit("[plan-wires] W-BOX needs the agent head AND the box3d decoder to be built")
            core.box_tok_r9 = PW.BoxTokens(int(core.cfg.decoder.d), len(AGENT_CLASSES))
            for ly in core.decoder.layers:
                if getattr(ly, "cross_agent", None) is not None:
                    ly.box_gate = nn.Parameter(torch.zeros(1))
            tac = getattr(model, "tac_decoder_v6", None)
            for ly in (() if tac is None else tac.layers):
                ly.box_gate = nn.Parameter(torch.zeros(1))
            setattr(model, BOX_ATTR, True)
    dev = next(model.parameters()).device
    model.to(dev)


def stamp(model, args) -> dict | None:
    """``config.json["perception"]["plan_wires"]`` (read OFF the built model); None when both are off."""
    if not (map_on(args) or box_on(args)):
        return None
    out = {}
    if map_on(args):
        p = model._perception.bev_pool.map_chan_proj
        out["map_channels"] = {"flag": MAP_FLAG, "channels": ["drv", "drv_sdf/clip", "sdw", "lane", "unobs"], "grid": "1 m over 100 m x +-30 m",
                               "detached": True, "projection": "1x1, ZERO-INIT, added to the pool output", "pool_window": list(p.dst),
                               "beyond_pool_window": "built and exported (perception.map_chan_1m), read by no consumer until the pool window is "
                                                     "widened (E0b / --bev-planner-crop-m)", "params": sum(x.numel() for x in p.parameters())}
    if box_on(args):
        core = model.core
        n_gates = sum(1 for ly in core.decoder.layers if ly.box_gate is not None)
        tac = getattr(model, "tac_decoder_v6", None)
        out["box_tokens"] = {"flag": BOX_FLAG, "k": core.box_tok_r9.k, "source": "box3d slots (perception.box_slots), NOT the agent head",
                             "ranking": ("presence_logit trained quality-aware (--r9-box-quality ON)" if bool(getattr(args, "r9_box_quality", False))
                                         else "presence_logit, plain focal presence (--r9-box-quality OFF)"),
                             "features": "x/60, y/16, l/10, w/4, sin, cos, v_rel_x/10, v_rel_y/10, class posterior, presence + rank embedding",
                             "detached": True, "gate": "zero-init scalar per consuming layer; out = attend(agents) + g*(attend(agents+box) - attend(agents))",
                             "planner_gates": n_gates, "tactical_gates": 0 if tac is None else sum(1 for ly in tac.layers if ly.box_gate is not None),
                             "k3_agent_head_retired": False, "params": sum(x.numel() for x in core.box_tok_r9.parameters())}
    return out


def dvb_check_map(model, args) -> list:
    from tanitad.train.declared_vs_built import Mismatch
    built = getattr(getattr(getattr(model, "_perception", None), "bev_pool", None), "map_chan_proj", None) is not None
    out = []
    if map_on(args) != built:
        out.append(Mismatch(MAP_DEST, map_on(args), built, "model._perception.bev_pool.map_chan_proj", "argv and the build disagree"))
    if map_on(args) != bool(getattr(model, MAP_ATTR, False)):
        out.append(Mismatch(MAP_DEST, map_on(args), bool(getattr(model, MAP_ATTR, False)), f"model.{MAP_ATTR}", "argv and the attribute disagree"))
    return out


def dvb_check_box(model, args) -> list:
    from tanitad.train.declared_vs_built import Mismatch
    core = getattr(model, "core", None)
    built = getattr(core, "box_tok_r9", None) is not None
    out = []
    if box_on(args) != built:
        out.append(Mismatch(BOX_DEST, box_on(args), built, "model.core.box_tok_r9", "argv and the build disagree"))
    if built:
        n_cross = sum(1 for ly in core.decoder.layers if getattr(ly, "cross_agent", None) is not None)
        n_gate = sum(1 for ly in core.decoder.layers if ly.box_gate is not None)
        if n_gate != n_cross or n_gate == 0:
            out.append(Mismatch(BOX_DEST, f"{n_cross} planner gates", n_gate, "model.core.decoder.layers[*].box_gate", "a consuming layer has no gate"))
        tac = getattr(model, "tac_decoder_v6", None)
        if tac is not None and any(ly.box_gate is None for ly in tac.layers):
            out.append(Mismatch(BOX_DEST, "a gate per tactical layer", "missing", "model.tac_decoder_v6.layers[*].box_gate", "a tactical layer has no gate"))
    return out


def register_dvb_levers(register) -> None:
    register(MAP_DEST, "built", dvb_check_map)
    register(BOX_DEST, "built", dvb_check_box)


def wcost_ratio(critic_bce: float, traj_loss: float) -> float:
    """The step-0 ratio critic-BCE / trajectory-loss (REFCV9_DESIGN 2.7 W-COST: ``--w-r8-drivable`` is set so the weighted BCE is ~0.1 x the
    trajectory loss, from this smoke number -- never from eval). Refuses a non-positive denominator rather than printing inf."""
    if not traj_loss > 0.0:
        raise ValueError(f"trajectory loss {traj_loss!r} is not positive")
    return float(critic_bce) / float(traj_loss)
