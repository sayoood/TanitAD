#!/usr/bin/env python3
"""TanitAD side of the NavSim seam for refcv7 (RUN IN THE TANITAD VENV).

    NavSim export (devkit AgentInput per scorer token) ──► DECLARE (arm) ──► refcv7 inputs:
      frames      416x1024 cylindrical 3-camera stitch, 10 Hz slots <- 2 Hz frames   } the refcv6
      nav         NavSim driving_command -> the v7 token (follow / left / right)      } suite's
      v0          |velocity| at t0                                                    } functions,
      ego window  8 x (x, y, yaw, v) at 10 Hz, linear between the 2 Hz states         } IMPORTED
      max speed   the map posted limit of the ego lane at t0 (valid=0 where absent)   } unchanged
      lift        NEW: the stitch's virtual camera through the MODEL'S OWN 0.25 m bank parameters
      prior       NEW: refcv7 builds P (ha0_ext_pose) from the ego window INSIDE its forward; the
                  bridge also computes P model-free from the same window (control KPR + the
                  PRIOR_ha0p arm)
    refcv7 (DDIM, 117 residual anchors on P, reach clamp, speed-ceiling filter) ──► traj [8, 2]

⛔ THE FEED IS THE TRAINER'S. ``forward_kwargs7`` returns EXACTLY the keyword set
``refc_v3_train.compute_losses_v3`` passes to ``model(...)`` (pinned by an AST read of the trainer
in ``tests/test_forward_contract7.py``), and ``run_model7`` calls ``set_ego_window(poses, 8,
actions=None)`` the way that function does for a pose-only prior. The model is built ONLY by
``stack/tanitad/eval/refcv7_loader.build_model`` (strict load, G-DVB, stamp checks).

⛔ ENFORCEMENT (gate ``navsim.ego_enforcement``) is the refcv6 suite's ``declare6``: an R7 arm is
mapped to the R6 arm with the IDENTICAL declared set / nav source / max-speed source
(``R6_TEMPLATE``, pinned by a literal test), so the tested enforcement point is reused, not copied.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boot7  # noqa: E402  (tree + sys.path + the refcv6 suite's modules)

R6 = boot7.R6
F4 = boot7.F4
RIG6 = boot7.RIG6
RefusedInput = R6.RefusedInput

KNOT_T_S = R6.KNOT_T_S                 # (0.5, 1, 1.5, 2, 3, 4, 5, 6)
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
WINDOW = R6.WINDOW                     # 8
N_ANCHORS = 117
RESIDUAL_PRIOR = "ha0_ext_pose"
EQUALIZE_BOTTOM_ROWS = 43
#: the keyword set compute_losses_v3 passes to ``model(frames, ...)`` (refc_v3_train.py:4556-4560)
FORWARD_KWARGS = ("nav_cmd", "v0", "steps", "lan", "ego_state", "nav_args", "v_max_ms",
                  "v_max_valid", "agent_gt", "perception_grid", "perception_valid")

_HIST_DECL = R6._HIST_DECL
ARMS7 = {
    "R7_A1": {"frames": "ST", "nav": "cmd", "vmax": "map", "filter": True, "seed": 0,
              "meaning": "PRIMARY (bar arm): the model AS CONFIGURED -- ceiling filter ON"},
    "R7_A1_s1": {"frames": "ST", "nav": "cmd", "vmax": "map", "filter": True, "seed": 1,
                 "meaning": "R7_A1 at inference seed 1: the run-to-run INFERENCE floor"},
    "R7_FILTOFF": {"frames": "ST", "nav": "cmd", "vmax": "map", "filter": False, "seed": 0,
                   "meaning": "SPEC_REFCV7 A2 sensitivity reading: ceiling filter OFF (real forward)"},
    "R7_BLIND": {"frames": "BLIND", "nav": "cmd", "vmax": "map", "filter": True, "seed": 0,
                 "meaning": "frames-blind (the registered deliberate regression): one constant grey"},
    "R7_NAVOFF": {"frames": "ST", "nav": None, "vmax": "map", "filter": True, "seed": 0,
                  "meaning": "nav withheld: nav_cmd=None (the model's nav-ZERO intervention)"},
    "R7_VMAXOFF": {"frames": "ST", "nav": "cmd", "vmax": "off", "filter": True, "seed": 0,
                   "meaning": "max speed withheld: valid=0 everywhere (all-zero one-hot; no ceiling)"},
    "R7_A1NT": {"frames": "NT", "nav": "cmd", "vmax": "map", "filter": True, "seed": 0,
                "meaning": "R7_A1 with the NEAREST-TIME frame history (declared sensitivity)"},
    "R7_VMAXORACLE": {"frames": "ST", "nav": "cmd", "vmax": "oracle", "filter": True, "seed": 0,
                      "meaning": ("⛔ PRIVILEGED DIAGNOSTIC: the human's own realised max over "
                                  "[t0+2, t0+6] s (the channel's TRAINING definition) -- never a "
                                  "result, never in a bar")},
}
#: arms that are never a forward of their own -- built from R7_A1's forward or model-free
DERIVED_ARMS = {
    "R7_CEILDECL_d": ("⛔ DIAGNOSTIC (SPEC amendment A1): R7_A1's forward with the speed ceiling "
                      "applied to the EMITTED (E9) pick, as SPEC_REFCV7 A2 declares it -- the "
                      "built model does not apply it (refc_v3.py:2285-2298); never a bar arm"),
    "PRIOR_ha0p": "the model's own prior P alone (Delta = 0), model-free from the declared window",
}
#: each R7 arm -> the refcv6 arm with the IDENTICAL declared set / nav source / max-speed source
R6_TEMPLATE = {"R7_A1": "R6_A1", "R7_A1_s1": "R6_A1_s1", "R7_FILTOFF": "R6_A1",
               "R7_BLIND": "R6_BLIND", "R7_NAVOFF": "R6_NAVOFF", "R7_VMAXOFF": "R6_VMAXOFF",
               "R7_A1NT": "R6_A1NT", "R7_VMAXORACLE": "R6_VMAXORACLE"}
# ---- D6 BRIDGE_FIX (opt-in additions; every line below is NEW, no base line is edited) -------------------------
#: P2 nav-intervention arms (SPEC_P1P2 s5). Frames / max speed / ego inputs / seed identical to R7_A1; ONLY the nav
#: token differs, applied by ``apply_nav_override`` after the base ``nav_input``.
ARMS7["R7_NAVFOLLOW"] = {"frames": "ST", "nav": "cmd", "vmax": "map", "filter": True, "seed": 0,
                         "nav_override": "follow",
                         "meaning": "D6 P2: nav token forced to 'follow' on every scene (premature-turn probe)"}
ARMS7["R7_NAVFLIP"] = {"frames": "ST", "nav": "cmd", "vmax": "map", "filter": True, "seed": 0,
                       "nav_override": "flip",
                       "meaning": "D6 P2 positive control: LEFT <-> RIGHT swapped, STRAIGHT unchanged"}
R6_TEMPLATE["R7_NAVFOLLOW"] = "R6_A1"
R6_TEMPLATE["R7_NAVFLIP"] = "R6_A1"
#: set True by ``run_bridge7.py --export-fan``; while False ``run_model7`` is byte-for-byte the base behaviour
EXPORT_FAN = {"on": False}


def apply_nav_override(nav: dict, mode: str) -> dict:
    """D6 P2: the nav token after an intervention. Refuses unless the base nav is a fed token."""
    from tanitad.refs import refb
    if nav["nav_index"] is None:
        raise RefusedInput("nav_override on an arm whose nav is withheld")
    idx = {n: int(refb.NAV_COMMANDS.index(n)) for n in ("follow", "left", "right")}
    if idx[nav["name"]] != int(nav["nav_index"]):
        raise RefusedInput(f"nav name/index disagree: {nav}")
    new = dict(nav)
    if mode == "follow":
        name = "follow"
    elif mode == "flip":
        name = {"left": "right", "right": "left"}.get(nav["name"], nav["name"])
    else:
        raise RefusedInput(f"unknown nav_override {mode!r}")
    new.update({"nav_index": idx[name], "name": name, "overridden_from": nav["name"], "override": mode})
    return new


def export_fan_from_out(out: dict, ds: dict) -> dict:
    """D6 P1: the candidate universe of one forward, read from ``out`` AFTER the forward (no extra model call).
    Universe = ``out["r7_candidates"]`` (117 anchor-fan members then the 64 WTA proposals) WHEN the model was built with the refcv7 WTA heads;
    otherwise ``out["anchor_traj"]`` (the 117-member fan the E9 selection actually chooses from). refcv7-r101-s0 as launched has NO WTA decoder /
    disentangled scorer (config.json argv has no --refcv7; w_r7_wta = w_r7_scorer = 0.0), so its universe is the 117 fan (SPEC_P1P2 amendment A1).
    ``sel_idx`` / ``sel_idx_base`` index the first 117."""
    wta = out.get("r7_candidates")
    src = wta if wta is not None else out.get("anchor_traj")
    if src is None:
        raise RefusedInput("export_fan: out has neither r7_candidates nor anchor_traj -- nothing to export")
    cand = src.float()[0].cpu().numpy().astype(np.float32)                           # [117 or 181, 8, 2]
    poses = np.stack([knots_to_navsim(c.astype(np.float64)) for c in cand]).astype(np.float32)   # [N, 8, 3]

    def arr(k):
        v = out.get(k)
        return None if v is None else v.float()[0].cpu().numpy().astype(np.float32)
    n_fan = int(out["r7_n_fan"]) if out.get("r7_n_fan") is not None else int(out["anchor_traj"].shape[1])
    fan = {"cands_knots": cand, "cands_poses": poses, "n_fan": n_fan, "universe": ("FAN117+WTA64" if wta is not None else "FAN117"),
           "sel_idx": int(out["sel_idx"][0]), "sel_idx_base": ds.get("sel_idx_base"),
           "sel_score_v3": arr("sel_score_v3"),
           "reach_keep": (None if out.get("reach_keep") is None else
                          out["reach_keep"][0].float().cpu().numpy().astype(np.float32)),
           "r7_score": arr("r7_score"),
           "r7_sel_idx": (None if out.get("r7_sel_idx") is None else int(out["r7_sel_idx"][0])),
           "r7_pick_is_wta": (None if out.get("r7_pick_is_wta") is None else bool(out["r7_pick_is_wta"][0])),
           "traj_r7_knots": arr("traj_r7")}
    fan["traj_r7_poses"] = (None if fan["traj_r7_knots"] is None else
                            knots_to_navsim(fan["traj_r7_knots"].astype(np.float64)).astype(np.float32))
    return fan
# ---- end D6 BRIDGE_FIX ---------------------------------------------------------------------------------------
FIELDS = R6.FIELDS


def _template(arm: str) -> str:
    t = R6_TEMPLATE.get(arm)
    if t is None or arm not in ARMS7:
        raise RefusedInput(f"unknown refcv7 arm {arm!r}")
    s6, s7 = R6.ARMS6[t], ARMS7[arm]
    if (s6["frames"], s6["nav"], s6["vmax"]) != (s7["frames"], s7["nav"], s7["vmax"]):
        raise RefusedInput(f"{arm} -> {t}: frames/nav/vmax disagree ({s6} vs {s7})")
    return t


def declared(arm: str) -> tuple:
    return tuple(R6.ARMS6[_template(arm)]["declared"])


# --------------------------------------------------------------------------- #
# 1. DECLARE + the refcv6 input functions (imported)                            #
# --------------------------------------------------------------------------- #
def declare7(ego_statuses: list, arm: str) -> dict:
    """The refcv6 enforcement point (``declare6``) under the R7 arm's template; the returned dict
    is the COMPLETE ego/route information any input function may read."""
    d = R6.declare6(ego_statuses, _template(arm))
    d["_arm"] = arm
    return d


def nav_input(decl: dict, arm: str) -> dict:
    return R6.nav_input(decl, _template(arm))


def max_speed_input(speed_rec, arm: str) -> dict:
    return R6.max_speed_input(speed_rec, _template(arm))


ego_history_poses = R6.ego_history_poses
v0_of = R6.v0_of
speed_of = R6.speed_of
Bank416 = R6.Bank416
BankNavtest416 = R6.BankNavtest416
pack_rows = R6.pack_rows
scene_seed = R6.scene_seed
knots_to_navsim = R6.knots_to_navsim
json_dump = R6.json_dump


def slot_sources7(times_s: list, construction: str) -> list:
    return R6.slot_sources6(times_s, construction)


def ceiling_ms(vmax: dict) -> float:
    """The ceiling the model's filter will read (``refc_v3.py:1782-1790``): the fed one-hot's bin
    limit, ``+inf`` for an all-zero (unknown / withheld) row. For the manifest and the tests; the
    model computes its own and the bridge RECORDS what the filter actually received."""
    if not vmax.get("v_max_valid"):
        return float("inf")
    from tanitad.refs.refcv6_max_speed import SPEED_MAX_STEPS_MS_V6, speed_max_bin
    b, _ = speed_max_bin(float(vmax["v_max_ms"]))
    return float(SPEED_MAX_STEPS_MS_V6[b])


# --------------------------------------------------------------------------- #
# 2. THE PRIOR, model-free (the model's own functions on the declared window)   #
# --------------------------------------------------------------------------- #
def prior_model_free(hist_poses: np.ndarray, v0: float, horizons=HORIZONS) -> dict:
    """``(a0, kappa0, path [8, 2])`` of ``ha0_ext_pose`` on the window, through
    ``kinematic_prior.prior_controls`` + ``prior_path`` (float32, CPU) -- exactly what the decoder
    composes on (``refc.py:4206`` / ``:3658``)."""
    import torch
    from tanitad.models import kinematic_prior as kp
    p = torch.from_numpy(np.asarray(hist_poses, np.float32))[None]
    a0, k0 = kp.prior_controls(RESIDUAL_PRIOR, p, WINDOW, None, dt=kp.DT_TICK)
    v = torch.tensor([float(v0)], dtype=torch.float32)
    path = kp.prior_path(a0, k0, v, tuple(horizons), tick=kp.DT_TICK)[0]
    return {"a0": float(a0[0]), "kappa0": float(k0[0]),
            "path": path.numpy().astype(np.float64)}


# --------------------------------------------------------------------------- #
# 3. THE LIFT (0.25 m) for a NavSim rig, through the MODEL'S OWN bank            #
# --------------------------------------------------------------------------- #
def bank_params(bank) -> dict:
    """Every parameter ``LiftGeometryBank.geometry`` passes to ``build_lift_geometry``, READ OFF the
    built bank (never retyped)."""
    return {"frame": bank.frame, "stride": int(bank.stride),
            "heights_m": tuple(bank.heights_m), "grid": bank.grid_spec,
            "observed": bank._observed, "equalize_bottom_rows": int(bank.equalize_bottom_rows)}


def lift_geometry_for_camera(cam, bank):
    """``(grid [Z,X,Y,2], valid [Z,X,Y])`` for ONE camera -- the SAME call
    ``LiftGeometryBank.geometry`` makes for a PhysicalAI clip (``refcv6_perception_branch.py:419``),
    with the bank's own parameters."""
    from tanitad.models.bev_lift import build_lift_geometry
    bp = bank_params(bank)
    g = build_lift_geometry(cam, frame=bp["frame"], stride=bp["stride"],
                            heights_m=bp["heights_m"], grid=bp["grid"], observed=bp["observed"])
    return g.grid, g.valid


def hires_lift_for_rig(rig: dict, road_z_m: float, bank) -> tuple:
    """The NavSim stitch's virtual camera (refcv6 ``rig6.rig_camera``: rotation = the stitch's M,
    centre = CAM_F0's mount, height = F0.z - road plane) through the model's 0.25 m bank."""
    import math as _m
    cam = RIG6.rig_camera(rig, road_z_m, bank.frame)
    grid, valid = lift_geometry_for_camera(cam, bank)
    fwd = cam.R_cam_to_rig[:, 2].numpy()
    summ = {"cam_height_m": float(cam.t_cam_in_rig[2]), "cam_x_m": float(cam.t_cam_in_rig[0]),
            "cam_y_m": float(cam.t_cam_in_rig[1]),
            "pitch_down_deg": float(_m.degrees(_m.asin(-fwd[2]))),
            "yaw_deg": float(_m.degrees(_m.atan2(fwd[1], fwd[0]))),
            "valid_frac": float(valid.float().mean())}
    return grid, valid, summ


# --------------------------------------------------------------------------- #
# 4. THE MODEL                                                                   #
# --------------------------------------------------------------------------- #
def load_refcv7(ckpt: str, config: str, device: str = "cpu", precision: str = "auto"):
    """``(model, cfg, args, rec, meta)`` -- built by ``refcv7_loader.build_model`` (the trainer's
    own parser + pin, train()'s build order replayed, STRICT load, G-DVB, config stamp checks),
    then every NavSim-seam fact ASSERTED. ``precision``: ``as_trained`` keeps the trunk's recorded
    levers (bf16 autocast + NHWC); ``fp32`` turns those two off (CPU, where bf16 is emulated);
    ``auto`` = as_trained on CUDA, fp32 on CPU (the refcv6 suite's rule D3)."""
    L = boot7.loader()
    cfgd = L.load_config(config)
    model, cfg, args, rec = L.build_model(cfgd, ckpt, device=device)
    dec = model.core.decoder
    checks = {
        "strict_0_0": (not rec["state_dict"]["missing"] and not rec["state_dict"]["unexpected"]),
        "gdvb_0": not rec["declared_vs_built"]["mismatches"],
        "horizons": tuple(int(h) for h in cfg.core.trajectory.horizons) == HORIZONS,
        "window": int(cfg.core.window) == WINDOW,
        "image_hw": tuple(cfg.core.encoder.image_hw()) == (416, 1024),
        "n_anchors": int(dec.anchors.shape[0]) == N_ANCHORS if hasattr(dec, "anchors") else False,
        "anchor_units_alat": str(getattr(dec, "anchor_control_units", "")) == "alat",
        "residual_prior": str(getattr(dec, "residual_prior", "")) == RESIDUAL_PRIOR,
        "speed_ceiling_filter_on": bool(getattr(dec, "speed_ceiling_filter", False)),
        "ego_hist": model.core.ego_hist is not None,
        "max_speed_onehot_v6": bool(getattr(cfg, "max_speed_onehot_v6", False)),
        "hires_bank": getattr(model, "_lift_bank_hires", None) is not None,
        "no_stride16_bank": getattr(model, "_lift_bank", None) is None,
        "trunk_equalize_43": int(cfg.core.encoder.trunk_equalize_bottom_rows) == EQUALIZE_BOTTOM_ROWS,
        "hires_mask_43": int(model._lift_bank_hires.equalize_bottom_rows) == EQUALIZE_BOTTOM_ROWS,
        "no_ego_state_inject": not bool(getattr(cfg, "ego_state_inject", False)),
        "no_lan": not bool(getattr(args, "graft_lan", False) or getattr(args, "goal_str", False)),
        "no_nav_args": not bool(getattr(args, "nav_args", False)),
        "agents_not_oracle": not bool(getattr(getattr(cfg.core, "agents", None), "oracle", False)),
        "decoder_steps_2": int(rec["decoder_steps"]) == 2,
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise RefusedInput(f"the rebuilt refcv7 does not match the NavSim seam's premises: {bad}")
    enc = model.core.encoder
    if precision == "auto":
        precision = "as_trained" if str(device).startswith("cuda") else "fp32"
    levers0 = dict(enc.memory_levers)
    if precision == "fp32":
        enc.memory_levers["bf16"] = False
        enc.memory_levers["channels_last"] = False
    elif precision != "as_trained":
        raise RefusedInput(f"precision {precision!r}")
    meta = {"precision": precision, "trunk_levers_recorded": levers0,
            "trunk_levers_eval": dict(enc.memory_levers), "device": str(device),
            "seam_checks": checks, "loader_record": {
                k: rec.get(k) for k in ("loader", "repo", "state_dict", "declared_vs_built",
                                        "param_breakdown", "departures", "trunk_equalize_as_trained",
                                        "nav_compliance_tau_file", "map_hires", "anchors",
                                        "mode", "decoder_steps", "sampler", "build_s")},
            "lift_bank_params": {k: (str(v) if k in ("frame", "grid") else
                                     (None if v is None and k == "observed" else
                                      (list(v.shape) if k == "observed" else v)))
                                 for k, v in bank_params(model._lift_bank_hires).items()},
            "tree": boot7.tree_record()}
    return model, cfg, args, rec, meta


def exact_dedup(enc) -> None:
    """The refcv6 suite's EVAL-ONLY exact-duplicate dedup, IMPORTED (``refcv6_bridge.exact_dedup``);
    re-measured on refcv7 by control KD before any use."""
    R6.exact_dedup(enc)


def forward_kwargs7(frames_dev, nav: dict, v0: float, vmax: dict, grid, valid, steps: int,
                    device: str) -> dict:
    """EXACTLY ``compute_losses_v3``'s keyword set (``FORWARD_KWARGS``), NavSim-valued:
    ``lan`` None (no LAN graft), ``ego_state`` None (not injected), ``nav_args`` None (off),
    ``agent_gt`` None (the agent head reads the image; the oracle path is off) -- each asserted by
    ``load_refcv7``. ``nav_cmd`` None only for the nav-withheld arm."""
    import torch
    kw = {"nav_cmd": (None if nav["nav_index"] is None
                      else torch.tensor([int(nav["nav_index"])], dtype=torch.long, device=device)),
          "v0": torch.tensor([float(v0)], dtype=torch.float32, device=device),
          "steps": int(steps), "lan": None, "ego_state": None, "nav_args": None,
          "v_max_ms": torch.tensor([float(vmax["v_max_ms"])], dtype=torch.float32, device=device),
          "v_max_valid": torch.tensor([float(vmax["v_max_valid"])], dtype=torch.float32,
                                      device=device),
          "agent_gt": None,
          "perception_grid": grid[None].to(device), "perception_valid": valid[None].to(device)}
    if tuple(kw) != FORWARD_KWARGS:
        raise RefusedInput(f"forward kwargs {tuple(kw)} != {FORWARD_KWARGS}")
    return kw


@contextlib.contextmanager
def capture_selection():
    """Record the two argmax masks the decoder applies (``refc.py:3578-3610``): the reachability
    band (``refc_select.reachability_mask``) and the set-speed ceiling
    (``refcv6_selection.SpeedCeilingFilter.forward``), plus the ceiling the filter READ. Pure
    observation: each wrapper returns the wrapped function's output unchanged."""
    from tanitad.refs import refc_select as sl
    from tanitad.refs import refcv6_selection as v6sel
    st = {"reach": [], "ceil": [], "v_limit": []}
    f_reach = sl.reachability_mask
    f_ceil = v6sel.SpeedCeilingFilter.forward

    def reach(*a, **k):
        out = f_reach(*a, **k)
        st["reach"].append(out.detach().clone())
        return out

    def ceil(self, cand, v_limit_ms):
        keep, tele = f_ceil(self, cand, v_limit_ms)
        st["ceil"].append(keep.detach().clone())
        st["v_limit"].append(v_limit_ms.detach().float().cpu().clone())
        return keep, tele

    sl.reachability_mask = reach
    v6sel.SpeedCeilingFilter.forward = ceil
    try:
        yield st
    finally:
        sl.reachability_mask = f_reach
        v6sel.SpeedCeilingFilter.forward = f_ceil


def derived_selection(out: dict, st: dict, filter_on: bool) -> dict:
    """The EMITTED pick, re-derived, and the pick the ceiling filter WOULD make if it reached it.

    ⛔⛔ MEASURED 2026-09-28 (step-1,500 validation; SPEC amendment A1): refcv7's emitted plan is
    NOT the decoder's argmax. ``RefCV3Model.forward`` re-selects over the fan AFTER the decoder
    (E9 goal selection, ``refc_v3.py:2285-2298``): ``blended = apply_seam_clamp(sel_score,
    goal_gate * scorer)`` -> ``rank = blended`` masked by ``reach_keep`` ONLY -> ``argmax`` ->
    ``out["traj"]``; the decoder's own (ceiling-filtered) pick is demoted to ``sel_idx_base``. So
    the speed-ceiling mask (``refc.py:3605-3610``) never reaches the emitted plan: R7_A1 (filter
    ON) and R7_FILTOFF (filter OFF) emitted BIT-IDENTICAL plans on 204/204 warmup scenes.

    Returns (all on the model's own device/dtype, the SAME kernels, so ties resolve as the model's):
      * ``ok`` -- ``argmax(sel_score_v3 masked by reach_keep)`` reproduces ``out["sel_idx"]``
        (the replication check; a row failing it has no derived reading);
      * ``idx_decl`` / ``traj_decl`` -- the ceiling AS DECLARED (SPEC_REFCV7 A2 "filters the
        ARGMAX"): the same rank ALSO masked by the captured ceiling keep; a row whose every
        reach-kept candidate is over the ceiling keeps the emitted pick (the filter's own
        empty-row rule). DIAGNOSTIC -- not the model as built.
      * ``emitted_over_ceiling`` -- the emitted pick is among the candidates the filter masks.
    """
    sel = int(out["sel_idx"][0])
    blended = out.get("sel_score_v3")
    if blended is None:
        return {"ok": False, "why": "no sel_score_v3 in the output (not the E9 path)"}
    keep_r = out.get("reach_keep")
    rank = blended if keep_r is None else blended.masked_fill(~keep_r.to(blended.device),
                                                              float("-inf"))
    idx_emit = int(rank.argmax(dim=1)[0])
    ok = idx_emit == sel
    res = {"ok": bool(ok), "why": None if ok else "argmax(sel_score_v3 | reach_keep) != sel_idx",
           "sel_idx": sel, "sel_idx_base": (int(out["sel_idx_base"][0])
                                            if out.get("sel_idx_base") is not None else None)}
    if filter_on:
        if len(st["ceil"]) != 1:
            res.update(ok=False, why=f"{len(st['ceil'])} ceiling calls (expected 1)")
            return res
        kc = st["ceil"][0].to(rank.device)
        rd = rank.masked_fill(~kc, float("-inf"))
        empty = bool(torch_isneginf_all(rd))
        idx_decl = sel if empty else int(rd.argmax(dim=1)[0])
        res.update({"idx_decl": idx_decl, "decl_row_empty": empty,
                    "n_candidates_over_ceiling": int((~kc[0]).sum()),
                    "emitted_over_ceiling": bool(not bool(kc[0, sel])),
                    "decl_changed_pick": bool(idx_decl != sel),
                    "traj_decl": out["anchor_traj"].float()[0, idx_decl].cpu().numpy()
                    .astype(np.float64)})
    return res


def torch_isneginf_all(t) -> bool:
    import torch
    return bool(torch.isneginf(t).all())


def run_model7(model, rows_u8: np.ndarray, decl: dict, nav: dict, vmax: dict,
               hist_poses: np.ndarray, grid, valid, steps: int, seed: int, device: str,
               filter_on: bool = True) -> dict:
    """One scene through refcv7's forward, fed as ``compute_losses_v3`` feeds it; the ceiling
    filter set to ``filter_on`` for this call and restored after."""
    import torch
    L = boot7.loader()
    fr = L.trainer().frames_to_device(torch.from_numpy(rows_u8)[None], device)
    v0 = v0_of(decl)
    kw = forward_kwargs7(fr, nav, v0, vmax, grid, valid, steps, device)
    dec = model.core.decoder
    f0 = bool(dec.speed_ceiling_filter)
    dec.speed_ceiling_filter = bool(filter_on)
    try:
        model.core.set_ego_window(torch.from_numpy(np.asarray(hist_poses, np.float32))[None]
                                  .to(device), WINDOW, actions=None)
        torch.manual_seed(int(seed))
        with capture_selection() as st, torch.no_grad():
            out = model(fr, **kw)
    finally:
        dec.speed_ceiling_filter = f0
    traj = out["traj"].float()[0].cpu().numpy().astype(np.float64)
    ds = derived_selection(out, st, filter_on)
    pp = out.get("residual_prior_path")
    diag = {"sel_idx": int(out["sel_idx"][0]), "sel_idx_base": ds.get("sel_idx_base"),
            "tacv6_lat_argmax": int(out["tacv6_lat_logits"][0].argmax(-1)),
            "tacv6_lon_argmax": int(out["tacv6_lon_logits"][0].argmax(-1)),
            "max_speed_injected": bool(out.get("max_speed_injected", False)),
            "nav_injected": bool(out.get("nav_injected", False)),
            "dedup": list(getattr(model.core.encoder, "last_dedup", None) or []),
            "ceiling_read_ms": (float(st["v_limit"][0][0]) if st["v_limit"] else None),
            "filter_on": bool(filter_on),
            "derived_ok": ds["ok"], "derived_why": ds.get("why"),
            "n_candidates_over_ceiling": ds.get("n_candidates_over_ceiling"),
            "emitted_over_ceiling": ds.get("emitted_over_ceiling"),
            "decl_changed_pick": ds.get("decl_changed_pick"),
            "decl_row_empty": ds.get("decl_row_empty"),
            "goal_gate": (float(out["goal_gate_value"]) if out.get("goal_gate_value") is not None
                          else None)}
    res = {"traj": traj, "v0": v0, "diag": diag,
           "prior_emitted_path": (None if pp is None else
                                  pp.float()[0].cpu().numpy().astype(np.float64)),
           "prior_emitted_ctrl": (None if out.get("residual_prior_ctrl") is None else
                                  [float(x) for x in out["residual_prior_ctrl"][0].float().cpu()])}
    if ds["ok"] and "traj_decl" in ds:
        res["traj_ceiling_declared"] = ds["traj_decl"]
    if EXPORT_FAN["on"]:                              # D6 BRIDGE_FIX: opt-in, reads `out` only
        res["fan"] = export_fan_from_out(out, ds)
    return res


def sha16(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 5. DELIBERATE-REGRESSION MUTATIONS (tests only; the runner REFUSES when set)    #
# --------------------------------------------------------------------------- #
#: ``R7_MUTATION`` reintroduces a REAL defect class into this module so that
#: ``tests/test_mutation7.py`` can prove the literal tests go RED on it:
#:   wrong_grid -- the 2 Hz states packed as consecutive 0.1 s samples (the history/prior defect);
#:   drop_mask  -- the lift built without the trunk's 43-row unobserved mask (the C26 class);
#:   stride16   -- refcv6's stride-16 lift parameters where refcv7's 0.25 m bank is required;
#:   kw_drop    -- a forward keyword silently dropped from the trainer's contract.
MUTATION = os.environ.get("R7_MUTATION", "")
if MUTATION:
    if MUTATION == "wrong_grid":
        def ego_history_poses(decl: dict, times_s: list):          # noqa: F811
            v = [speed_of(decl[f"ego_velocity[{i}]"]) for i in (1, 2, 3)]
            xy = [decl[f"ego_pose[{i}]"][:2] for i in (1, 2, 3)]
            yaw = [decl[f"ego_pose[{i}]"][2] for i in (1, 2, 3)]
            out = np.zeros((WINDOW, 4), np.float32)
            out[:] = [xy[0][0], xy[0][1], yaw[0], v[0]]
            for k in range(3):
                out[WINDOW - 3 + k] = [xy[k][0], xy[k][1], yaw[k], v[k]]
            return out
    elif MUTATION == "drop_mask":
        _bank_params0 = bank_params

        def bank_params(bank) -> dict:                              # noqa: F811
            p = _bank_params0(bank)
            p["observed"] = None
            return p
    elif MUTATION == "stride16":
        _bank_params1 = bank_params

        def bank_params(bank) -> dict:                              # noqa: F811
            p = _bank_params1(bank)
            p["stride"] = 16
            return p
    elif MUTATION == "kw_drop":
        FORWARD_KWARGS = tuple(k for k in FORWARD_KWARGS if k != "nav_args")  # noqa: F811
    else:
        raise SystemExit(f"unknown R7_MUTATION {MUTATION!r}")
