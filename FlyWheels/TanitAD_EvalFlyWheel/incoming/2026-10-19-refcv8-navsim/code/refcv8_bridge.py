#!/usr/bin/env python3
"""TanitAD side of the NavSim seam for refcv8 (RUN IN THE TANITAD VENV).  A PORT of ``refcv7_bridge.py`` (imported, not copied:
every function that does not change for refcv8 is the refcv7 / refcv6 suite's own).

    NavSim export (devkit AgentInput per scorer token) --> DECLARE (arm) --> refcv8 inputs:
      frames     416x1024 CYLINDRICAL 3-camera re-stitch of CAM_L0/F0/R0 (f_ref 488.924 px = 305.577 * 1024/640; the column is
                 LINEAR IN AZIMUTH, hfov 120.000 deg -- never the pinhole formula, which would read 92.6 deg), 10 Hz slots <- 2 Hz
                 frames (ST: every slot <- t0)                                                       } the refcv6 / refcv7 suite's
      v0         |velocity| at t0                                                                    } functions, IMPORTED
      ego window 8 x (x, y, yaw, v) at 10 Hz, LINEAR between the 2 Hz states; a0 and yaw rate are    } unchanged (a0 / yaw rate come
                 DERIVED INSIDE the forward from this window (ha0_ext_pose); curvature is NOT fed    } from the window, never a field)
      lift       the stitch's virtual camera through the MODEL'S OWN 0.25 m bank                      }
      nav_cmd    NEW: the NavSim driving command at t0 -> the v9 per-frame token {FOLLOW 0, TURN_L 1, TURN_R 2}   (section 2)
      r8_nav     NEW: the UNKNOWN row, built by the trainer's own ``scale_nav(zeros[1,5], known=False)`` = zeros [1,6]   (section 3)
      r8_rc      NOT FED (``--r8-no-rc``: ``r8_before_forward`` never writes it; the model's default None)
      speed      NEW: R8_LEGAL = the trained UNKNOWN row (v_max_ms 0, v_max_valid 0) through the trainer's
                 ``speed_input_treatment(..., legal=True)``; R8_PRIV = the map limit snapped to its containing road-type bin  (section 4)
    refcv8 --> out["traj"] [8, 2] at (0.5, 1, 1.5, 2, 3, 4, 5, 6) s --> [R8_RT: the 8-slot re-timing stage] --> E2's ``knots_to_navsim``
      (C2 not-a-knot cubic spline through (0,0) + the 8 knots, evaluated at 0.5 ... 4.0 s; heading = atan2 of the spline tangent)
      --> 8 NavSim poses (x, y, heading).  GT round-trip < 1e-3 m (E2 K2/K3).

THE FEED IS THE TRAINER'S.  ``forward_kwargs8`` is the keyword set ``refc_v3_train.compute_losses_v3`` hands to ``model(...)``
(11 keywords, ``_fwd(frames, nav_cmd=..., ..., perception_valid=...)``) PLUS the keys ``r8_before_forward`` merges over them for a
``--r8-no-rc`` build: ``r8_nav`` and the treated ``v_max_ms`` / ``v_max_valid``.  Pinned by an AST read of the trainer in
``tests/test_inputs8.py``.  ``model.core.set_ego_window(poses, 8, actions=None)`` is called exactly as the trainer calls it for a
pose-only prior.  The model is built ONLY by ``refcv7_loader.build_model`` as re-pointed by ``r8_repoint`` (strict load, G-DVB).

R8-EVAL-NAV-FEED-1: the PhysicalAI battery fed refcv8 a CLIP-level token and no ``r8_nav``.  Here the token comes from
``driving_command[3]`` ALONE -- the only declared command field -- so a clip-level (or any earlier-frame) command is not even
readable (``test_K5_undeclared_fields_are_invisible``), and ``r8_nav`` is passed on every forward.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boot8  # noqa: E402  (tree + sys.path + R8_MUTATION forwarding)

import refcv7_bridge as R7  # noqa: E402  (the refcv7 suite, from the launch tree)

R6 = R7.R6
RIG6 = R7.RIG6
B2 = R6.B2
RefusedInput = R6.RefusedInput

KNOT_T_S = R7.KNOT_T_S                 # (0.5, 1, 1.5, 2, 3, 4, 5, 6)
HORIZONS = R7.HORIZONS                 # (5, 10, 15, 20, 30, 40, 50, 60) x 0.1 s -- refcv8 changes no horizon
WINDOW = R7.WINDOW                     # 8
N_ANCHORS = 117
RESIDUAL_PRIOR = "ha0_ext_pose"
EQUALIZE_BOTTOM_ROWS = 43
FIELDS = R7.FIELDS
#: refc_v3_train.compute_losses_v3 ``_fwd(...)`` keywords (11) + the key ``r8_before_forward`` ADDS for a ``--r8-no-rc`` build
#: (``fwd["r8_nav"]``; ``v_max_ms`` / ``v_max_valid`` are the same two names REPLACED in place)
FORWARD_KWARGS = ("nav_cmd", "v0", "steps", "lan", "ego_state", "nav_args", "v_max_ms", "v_max_valid", "agent_gt",
                  "perception_grid", "perception_valid", "r8_nav")

MUTATION = boot8.MUTATION
KNOWN_MUTATIONS = ("wrong_token", "nav_fed_known", "drop_r8_nav", "legal_off", "wrong_grid", "drop_mask", "stride16")

# --------------------------------------------------------------------------- #
# 1. ARMS                                                                       #
# --------------------------------------------------------------------------- #
ARMS8 = {
    "R8_LEGAL": {"frames": "ST", "nav": "cmd", "speed": "legal", "seed": 0,
                 "meaning": "BAR ARM (A-R8-NS1 ii): bare NavSim command -> per-frame token, r8_nav = the unknown row, the "
                            "trained UNKNOWN speed row, ST frames"},
    "R8_LEGAL_s1": {"frames": "ST", "nav": "cmd", "speed": "legal", "seed": 1,
                    "meaning": "R8_LEGAL at inference seed 1: the run-to-run INFERENCE floor"},
    "R8_PRIV": {"frames": "ST", "nav": "cmd", "speed": "map", "seed": 0,
                "meaning": "DIAGNOSTIC ONLY, never a bar (A-R8-NS1 iv): the nuPlan map limit snapped to its containing "
                           "road-type bin ('no limit' -> the unknown row)"},
    "R8_NT": {"frames": "NT", "nav": "cmd", "speed": "legal", "seed": 0,
              "meaning": "R8_LEGAL with the NEAREST-TIME frame history (declared sensitivity)"},
    "R8_BLIND": {"frames": "BLIND", "nav": "cmd", "speed": "legal", "seed": 0,
                 "meaning": "frames-blind (the registered deliberate regression, SPEC_REFCV8 6.2): one constant grey; must FAIL N3"},
    "R8_NAVOFF": {"frames": "ST", "nav": None, "speed": "legal", "seed": 0,
                  "meaning": "nav withheld: nav_cmd=None (the model's nav-zero intervention)"},
}
#: arms that are never a forward of their own
DERIVED_ARMS = {
    "R8_RT": ("R8_LEGAL's forward (same seed, same rows) with the shipped 8-slot re-timing stage applied to the 8 knots BEFORE "
              "the spline; reported beside, carries a bar only if SHIP-R8 passes (D-NS8-3)"),
    "PRIOR_ha0p": "the model's own prior P alone (Delta = 0), model-free from the declared window",
}
#: each R8 arm -> the refcv6 arm with the IDENTICAL declared set / frames / nav source (the tested enforcement point, reused)
R6_TEMPLATE = {"R8_LEGAL": "R6_A1", "R8_LEGAL_s1": "R6_A1_s1", "R8_PRIV": "R6_A1", "R8_NT": "R6_A1NT",
               "R8_BLIND": "R6_BLIND", "R8_NAVOFF": "R6_NAVOFF"}


def _template(arm: str) -> str:
    t = R6_TEMPLATE.get(arm)
    if t is None or arm not in ARMS8:
        raise RefusedInput(f"unknown refcv8 arm {arm!r}")
    s6, s8 = R6.ARMS6[t], ARMS8[arm]
    if (s6["frames"], s6["nav"]) != (s8["frames"], s8["nav"]):
        raise RefusedInput(f"{arm} -> {t}: frames/nav disagree ({s6} vs {s8})")
    return t


def declared(arm: str) -> tuple:
    return tuple(R6.ARMS6[_template(arm)]["declared"])


def declare8(ego_statuses: list, arm: str) -> dict:
    """The refcv6 enforcement point (``declare6``) under the R8 arm's template: ONLY the declared EgoStatus fields are copied; the
    returned dict is the COMPLETE ego/route information any input function may read."""
    d = R6.declare6(ego_statuses, _template(arm))
    d["_arm"] = arm
    return d


ego_history_poses = R7.ego_history_poses
v0_of = R7.v0_of
speed_of = R7.speed_of
Bank416 = R7.Bank416
BankNavtest416 = R7.BankNavtest416
pack_rows = R7.pack_rows
scene_seed = R7.scene_seed
knots_to_navsim = R7.knots_to_navsim
json_dump = R7.json_dump
slot_sources8 = R7.slot_sources7
prior_model_free = R7.prior_model_free
bank_params = R7.bank_params
lift_geometry_for_camera = R7.lift_geometry_for_camera
hires_lift_for_rig = R7.hires_lift_for_rig
exact_dedup = R7.exact_dedup
sha16 = R7.sha16


def ceiling_ms(vmax: dict) -> float:
    """The ceiling the model's filter will read: the fed one-hot's bin limit, ``+inf`` for an all-zero (unknown / withheld) row.
    Binned through the TENSOR path in float32 -- the arithmetic the model's own ``MaxSpeedOneHotEncoder`` runs
    (``speed_max_bin_tensor``: ``v <= step + 1e-9`` with both sides float32).  ⛔ NOT ``speed_max_bin(float(v))``: the float32 cast of
    ``50/3.6`` is 13.88888931274414, which the float64 scalar rule reads as ABOVE the 50 km/h step (+1e-9) and drops into the 100 bin,
    while the model (float32 tensor) keeps it in the 50 bin -- refcv7's scalar ``ceiling_ms`` would record 27.78 m/s for a snapped
    50 km/h limit (MEASURED 2026-10-20 by ``test_priv_speed_snaps_to_the_containing_bin``)."""
    import torch
    if not vmax.get("v_max_valid"):
        return float("inf")
    from tanitad.refs.refcv6_max_speed import SPEED_MAX_STEPS_MS_V6, speed_max_bin_tensor
    idx, _ = speed_max_bin_tensor(torch.tensor([float(vmax["v_max_ms"])], dtype=torch.float32))
    return float(SPEED_MAX_STEPS_MS_V6[int(idx[0])])


# --------------------------------------------------------------------------- #
# 2. THE NAV TOKEN  (time-local by construction)                                #
# --------------------------------------------------------------------------- #
#: v9 ``nav_token`` ids: ``stack/scripts/build_v9_labels.py:260``  ``NAV_FOLLOW, NAV_TURN_L, NAV_TURN_R = 0, 1, 2``
NAV_NAMES8 = ("FOLLOW", "TURN_L", "TURN_R")
#: the NavSim ``driving_command`` one-hot index order (E2's control K9 on the logs): 0 left, 1 straight, 2 right, 3 unknown
NAVSIM_ONEHOT_KEY = {0: "left", 1: "forward", 2: "right", 3: "unknown"}
#: LITERAL MIRROR of ``refcv8_train.py:71-72`` ``NAVSIM_LEGAL_NAV_CMD`` (INTEGRATION.md sec. 5, MM 2026-10-04); the trainer's
#: ``R8_V9_NAV_TO_NAV_CMD = {0: 0, 1: 1, 2: 2}`` (``refc_v3_train.py:4686``) makes the v9 token id the ``nav_cmd`` index, and
#: ``refb.NAV_COMMANDS = ("follow", "left", "right", "straight")`` gives the same indices.  Both pinned by tests.
#:   left -> TURN_L (1)   forward -> FOLLOW (0, NOT 'straight' 3: the trainer never feeds 3)   right -> TURN_R (2)
#:   unknown -> FOLLOW (0), the LEGAL fallback
NAVSIM_LEGAL_NAV_CMD = {"left": 1, "forward": 0, "right": 2, "unknown": 0}


def nav_input8(decl: dict, arm: str) -> dict:
    """``{"nav_index", "name", "navsim_argmax", "navsim_key"}`` from ``decl["driving_command"][3]`` -- the t0 command, and the ONLY
    command field an arm declares, so the token is the window's NOW-row token by construction.  A non-one-hot vector is REFUSED
    (never arg-maxed).  ``nav_index`` None only for the nav-withheld arm."""
    if ARMS8[arm]["nav"] != "cmd":
        return {"nav_index": None, "name": None, "navsim_argmax": None, "navsim_key": None}
    c = np.asarray(decl["driving_command[3]"], dtype=np.float64)
    if c.shape != (4,) or not np.all(np.isin(c, (0.0, 1.0))) or c.sum() != 1.0:
        raise RefusedInput(f"driving_command is not a 4-dim one-hot: {c.tolist()}")
    k = int(np.argmax(c))
    key = NAVSIM_ONEHOT_KEY[k]
    tok = int(NAVSIM_LEGAL_NAV_CMD[key])
    if MUTATION == "wrong_token":                       # the feed-the-wrong-token defect: the two turns swapped
        tok = {1: 2, 2: 1}.get(tok, tok)
    return {"nav_index": tok, "name": NAV_NAMES8[tok], "navsim_argmax": k, "navsim_key": key}


# --------------------------------------------------------------------------- #
# 3. r8_nav = the UNKNOWN row, exactly as training's dropout path builds it     #
# --------------------------------------------------------------------------- #
def _r8train():
    from tanitad.train import refcv8_train
    return refcv8_train


def r8_nav_row() -> "torch.Tensor":                      # noqa: F821
    """``scale_nav(raw, known)`` with ``known = False`` -- ``refcv8_train.py:560-567``: the dropout path does
    ``known &= ~(rand < p)`` and the LEGAL path ``known = zeros_like(known)``, then ``fwd["r8_nav"] = scale_nav(nav, known)``; and
    ``scale_nav`` (``:367-378``) returns ``cat([out * k, k])`` with ``k = 0``  =>  exactly zeros [1, 6] float32.  The args are
    zeros because the v9 reader writes ``fz(...) = 0.0`` for an invalid row (``v9_labels.window_inputs``).  NavSim supplies no
    route geometry (and the route is derived from the logged future: leaderboard-illegal), so EVERY arm feeds this row."""
    import torch
    tr = _r8train()
    raw = torch.zeros(1, tr.NAV_IN_DIMS - 1, dtype=torch.float32)
    known = torch.zeros(1, dtype=torch.bool)
    if MUTATION == "nav_fed_known":                      # feeding a nav ARGUMENT to the LEGAL row
        raw = torch.tensor([[40.0, 120.0, 90.0, 1.0, 150.0]], dtype=torch.float32)
        known = torch.ones(1, dtype=torch.bool)
    row = tr.scale_nav(raw, known)
    if tuple(row.shape) != (1, tr.NAV_IN_DIMS):
        raise RefusedInput(f"scale_nav returned {tuple(row.shape)}, expected (1, {tr.NAV_IN_DIMS})")
    return row


# --------------------------------------------------------------------------- #
# 4. THE SPEED INPUT                                                            #
# --------------------------------------------------------------------------- #
def speed_input8(speed_rec: dict | None, arm: str) -> dict:
    """``{"v_max_ms", "v_max_valid", "why"}`` through the trainer's ``speed_input_treatment`` (eval, no draw).
    LEGAL: ``legal=True`` -> the unknown row ``(0.0, 0.0)`` (a NavSim ``AgentInput`` carries no map).  PRIV: the map limit is
    snapped to its CONTAINING road-type bin of the 4-way ladder {30, 50, 100, 120} km/h and fed as ``bin_kmh / 3.6`` computed in
    Python float64 (the GPU-scalar-divisor trap), exactly the value the training column fed; no limit -> the unknown row."""
    import torch
    tr = _r8train()
    src = ARMS8[arm]["speed"]
    z = torch.zeros(1)
    if src == "legal" or speed_rec is None or speed_rec.get("status") != "limit":
        vm, vv = tr.speed_input_treatment(z, z, None, training=False, legal=True)
        out = {"v_max_ms": float(vm[0]), "v_max_valid": float(vv[0]),
               "why": ("LEGAL: the trained unknown row" if src == "legal" else
                       f"PRIV: no map limit ({None if speed_rec is None else speed_rec.get('status')}) -> unknown row")}
        if MUTATION == "legal_off" and src == "legal":   # the defect: a ceiling leaks into the LEGAL row
            out.update(v_max_ms=50.0 / 3.6, v_max_valid=1.0, why="MUTATION legal_off")
        return out
    v = float(speed_rec["speed_limit_mps"])
    if not (math.isfinite(v) and v > 0):
        raise RefusedInput(f"non-positive map limit {v}")
    from tanitad.refs.refcv6_max_speed import SPEED_MAX_STEPS_KMH_V6, speed_max_bin
    b, over = speed_max_bin(v)
    kmh = float(SPEED_MAX_STEPS_KMH_V6[b])
    vm, vv = tr.speed_input_treatment(torch.tensor([kmh / 3.6], dtype=torch.float32), torch.ones(1), None, training=False,
                                      legal=False)
    return {"v_max_ms": float(vm[0]), "v_max_valid": float(vv[0]), "why": "PRIV: map limit snapped to the containing bin",
            "bin_kmh": kmh, "map_limit_mps": v, "over_ceiling": bool(over)}


# --------------------------------------------------------------------------- #
# 5. THE MODEL                                                                  #
# --------------------------------------------------------------------------- #
def load_refcv8(ckpt: str, config: str, device: str = "cpu", precision: str = "auto"):
    """``(model, cfg, args, rec, meta)`` -- built by ``refcv7_loader.build_model`` as re-pointed by ``r8_repoint`` (the trainer's own
    parser + pin, train()'s build order replayed, STRICT load, G-DVB, stamp checks), then every NavSim-seam premise ASSERTED.
    ``precision``: ``as_trained`` keeps the trunk's recorded levers (bf16 + NHWC); ``fp32`` turns them off (CPU); ``auto`` =
    as_trained on CUDA, fp32 on CPU.  ``model._r8_legal_row`` is stamped True (the LEGAL row) and recorded."""
    L = boot8.loader()
    cfgd = L.load_config(config)
    model, cfg, args, rec = L.build_model(cfgd, ckpt, device=device)
    model.eval()
    dec = model.core.decoder
    r8 = getattr(cfg, "refcv8", None)
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
        "no_nav_args_flag": not bool(getattr(args, "nav_args", False)),
        "agents_not_oracle": not bool(getattr(getattr(cfg.core, "agents", None), "oracle", False)),
        # ---- the refcv8 premises (plan section 5.1 B1 b) ----
        "r8_enabled": bool(getattr(model, "r8_enabled", False)),
        "r8_no_rc": bool(getattr(args, "r8_no_rc", False)) and bool(getattr(model, "_r8_no_rc", True)),
        "r8_nav_from_v9": bool(getattr(args, "r8_nav_from_v9", False)),
        "nav_args_dropout_0p5": abs(float(getattr(args, "r8_nav_args_dropout", -1.0)) - 0.5) < 1e-12,
        "speed_input_label": str(getattr(r8, "speed_input", "")) == "label",
        "speed_unknown_p_0p45": abs(float(getattr(args, "r8_speed_unknown_p", -1.0)) - 0.45) < 1e-12,
        "strategic_off": bool(getattr(args, "no_strategic", False)),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise RefusedInput(f"the rebuilt refcv8 does not match the NavSim seam's premises: {bad}")
    model._r8_legal_row = True
    enc = model.core.encoder
    if precision == "auto":
        precision = "as_trained" if str(device).startswith("cuda") else "fp32"
    levers0 = dict(enc.memory_levers)
    if precision == "fp32":
        enc.memory_levers["bf16"] = False
        enc.memory_levers["channels_last"] = False
    elif precision != "as_trained":
        raise RefusedInput(f"precision {precision!r}")
    meta = {"precision": precision, "trunk_levers_recorded": levers0, "trunk_levers_eval": dict(enc.memory_levers),
            "device": str(device), "seam_checks": checks, "r8_legal_row_stamped": True,
            "loader_record": {k: rec.get(k) for k in ("loader", "repo", "state_dict", "declared_vs_built", "param_breakdown",
                                                       "departures", "trunk_equalize_as_trained", "map_hires", "anchors",
                                                       "mode", "decoder_steps", "sampler", "build_s")},
            "lift_bank_params": {k: (str(v) if k in ("frame", "grid") else
                                     (None if v is None and k == "observed" else
                                      (list(v.shape) if k == "observed" else v)))
                                 for k, v in bank_params(model._lift_bank_hires).items()},
            "tree": boot8.tree_record(), "repoint": boot8.r8_repoint().describe(L) if hasattr(boot8.r8_repoint(), "describe") else None}
    return model, cfg, args, rec, meta


def forward_kwargs8(frames_dev, nav: dict, v0: float, vmax: dict, grid, valid, steps: int, device: str,
                    r8_nav=None) -> dict:
    """EXACTLY the trainer's keyword set for a ``--r8-no-rc`` build (``FORWARD_KWARGS``), NavSim-valued: ``lan`` / ``ego_state`` /
    ``nav_args`` / ``agent_gt`` None (each asserted by ``load_refcv8``), ``r8_nav`` = the unknown row [1, 6] (``r8_nav_row()``),
    ``nav_cmd`` None only for the nav-withheld arm.  ``r8_rc`` is NOT passed (the trainer does not pass it under ``--r8-no-rc``)."""
    import torch
    if r8_nav is None:
        r8_nav = r8_nav_row()
    kw = {"nav_cmd": (None if nav["nav_index"] is None
                      else torch.tensor([int(nav["nav_index"])], dtype=torch.long, device=device)),
          "v0": torch.tensor([float(v0)], dtype=torch.float32, device=device),
          "steps": int(steps), "lan": None, "ego_state": None, "nav_args": None,
          "v_max_ms": torch.tensor([float(vmax["v_max_ms"])], dtype=torch.float32, device=device),
          "v_max_valid": torch.tensor([float(vmax["v_max_valid"])], dtype=torch.float32, device=device),
          "agent_gt": None,
          "perception_grid": grid[None].to(device), "perception_valid": valid[None].to(device),
          "r8_nav": r8_nav.to(device=device, dtype=torch.float32)}
    if MUTATION == "drop_r8_nav":                        # the nav-feed defect of R8-EVAL-NAV-FEED-1: r8_nav never passed
        del kw["r8_nav"]
    if tuple(kw) != FORWARD_KWARGS:
        raise RefusedInput(f"forward kwargs {tuple(kw)} != {FORWARD_KWARGS}")
    return kw


def run_model8(model, rows_u8: np.ndarray, decl: dict, nav: dict, vmax: dict, hist_poses: np.ndarray, grid, valid,
               steps: int, seed: int, device: str, r8_nav=None) -> dict:
    """One scene through refcv8's forward, fed as ``compute_losses_v3`` + ``r8_before_forward`` feed it."""
    import torch
    L = boot8.loader()
    fr = L.trainer().frames_to_device(torch.from_numpy(rows_u8)[None], device)
    v0 = v0_of(decl)
    kw = forward_kwargs8(fr, nav, v0, vmax, grid, valid, steps, device, r8_nav)
    model.core.set_ego_window(torch.from_numpy(np.asarray(hist_poses, np.float32))[None].to(device), WINDOW, actions=None)
    torch.manual_seed(int(seed))
    with torch.no_grad():
        out = model(fr, **kw)
    traj32 = out["traj"].float()[0].cpu().numpy().astype(np.float32)
    pp = out.get("residual_prior_path")
    diag = {"sel_idx": (int(out["sel_idx"][0]) if out.get("sel_idx") is not None else None),
            "max_speed_injected": bool(out.get("max_speed_injected", False)),
            "nav_injected": bool(out.get("nav_injected", False)),
            "dedup": list(getattr(model.core.encoder, "last_dedup", None) or [])}
    return {"traj": traj32.astype(np.float64), "traj_f32": traj32, "v0": v0, "diag": diag,
            "prior_emitted_path": (None if pp is None else pp.float()[0].cpu().numpy().astype(np.float64)),
            "prior_emitted_path_f32": (None if pp is None else pp.float()[0].cpu().numpy().astype(np.float32)),
            "prior_emitted_ctrl": (None if out.get("residual_prior_ctrl") is None else
                                   [float(x) for x in out["residual_prior_ctrl"][0].float().cpu()])}


# --------------------------------------------------------------------------- #
# 6. THE RE-TIMING STAGE (R8_RT) -- opt-in; NO default table                    #
# --------------------------------------------------------------------------- #
def load_retime(stage_file: str, table_path: str, expected_sha256: str | None = None):
    """``(stage_module, table, record)``.  ``table_path`` is REQUIRED (no default is read anywhere): the stage's
    ``load_gains_table`` REFUSES a table whose bytes do not hash to its ``.sha256`` sidecar (and to ``expected_sha256`` when given).
    The stage module is loaded by path; its path and sha256 are recorded."""
    if not table_path:
        raise RefusedInput("R8_RT needs --retime-table: there is no default gains table")
    RS, path, sha = boot8.load_retime_stage(stage_file)
    table = RS.load_gains_table(table_path, expected_sha256)
    return RS, table, {"stage_file": path, "stage_sha256": sha, "table_file": os.path.abspath(table_path).replace("\\", "/"),
                       "table_sha256": table["sha256"], "knots_s": [float(x) for x in table["knots_s"]],
                       "gains": [float(x) for x in table["gains"]], "call": "8-slot (SHIP-R8 AMENDMENT 2)"}


def retime_plan8(RS, table, traj8, prior8) -> np.ndarray:
    """The 8-slot call (SHIP-R8 AMENDMENT 2): ONE ``retime_plan`` on all 8 model slots (``KNOT_T_S``) of the model's own plan against
    the prior ``ha0_ext_pose`` rolled at the same 8 slots, ``cast=True`` (the plan's own float32).  Returns float64 [8, 2]."""
    plan = np.asarray(traj8, np.float32)[None]
    prior = np.asarray(prior8, np.float32)[None]
    out = RS.retime_plan(plan, prior, np.asarray(KNOT_T_S, np.float64), table, cast=True)
    return np.asarray(out[0], np.float64)


# --------------------------------------------------------------------------- #
# 7. DELIBERATE-REGRESSION MUTATIONS (tests only; the runners REFUSE when set)  #
# --------------------------------------------------------------------------- #
#: ``R8_MUTATION`` reintroduces a REAL defect class so ``tests/test_mutation8.py`` can prove the literal tests go RED on it:
#:   wrong_token   -- TURN_L / TURN_R swapped (the feed-the-wrong-token defect);
#:   nav_fed_known -- a nav ARGUMENT row (known = 1) fed on the LEGAL row;
#:   drop_r8_nav   -- ``r8_nav`` never passed (R8-EVAL-NAV-FEED-1's other half);
#:   legal_off     -- a speed ceiling fed on the LEGAL row;
#:   wrong_grid / drop_mask / stride16 -- forwarded to the refcv7 modules (boot8) where the 2 Hz packing / mask / lift live.
if MUTATION and MUTATION not in KNOWN_MUTATIONS:
    raise SystemExit(f"unknown R8_MUTATION {MUTATION!r}; known {KNOWN_MUTATIONS}")
