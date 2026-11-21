#!/usr/bin/env python3
"""refcv3_arm.py — the T0 / T1 eval adapter for REF-C v3 (``tanitad.refs.refc_v3``).

WHY THIS FILE EXISTS (register D-V7-READINESS-2026-09-02 §D, two-probe absence):
refcv3 — the supervised hierarchical reference arm (pixels -> ResNet core ->
tactical / strategic heads -> anchored-diffusion trajectory; nav injected at
every layer; trained on B1 with the v7.2 labels, run
``/workspace/experiments/refcv3-b1-v72-30k``) — has NO admissible T1 number and
no instrument that can produce one. Its in-training eval is a T0 LOSS on 160
fixed windows and must never be quoted as driving performance. This adapter
reads a checkpoint at T1 with the four families, the hold-action control and
the nav-shuffle control — exactly what ``refav1_arm.py`` does for refav1, whose
dump contract / analysis / lead-block join it REUSES by import (never copies).

⛔ TIER DOCTRINE (``Project Steering/EVAL_DOCTRINE.md``). T0 = the predictor
consumes recorded future; T1 = action-closed loop, perception fixed at t0. The
stamps below travel into ``t1_eval.analyze`` unchanged.

THE T1 DEFINITION FOR A SUPERVISED TRAJECTORY MODEL (the Master Mind reviews
this before any number is quoted — REFCV3_ARM.md §2 carries the file:line).
``RefCV3Model.forward(frames, nav_cmd, v0, steps, lan)`` (refc_v3.py:480) is a
ONE-SHOT map from the OBSERVED window to a 6 s trajectory. It has no
action-conditioned predictor: nothing in the forward pass reads a future frame,
a future pose or a future action (only ``compute_losses_v3`` does, refc_v3_train
.py:428-643, and only to build targets). Its "action" IS its output — the
selected candidate ``out["traj"]`` [S, 2] at V3_HORIZONS (0.5 .. 6.0 s), the
E-DEC-48b direction (scene -> action). The kinematic state it conditions on is
ONE scalar, ``v0`` = ``pose_last[:, 3]`` — the speed MEASURED at the last
observed frame (refc_v3_train.py:445 ``v0 = pose_last[:, 3]``), admissible
under the PI ruling of 2026-09-02 (*"velocity as initial measured state at its
cycle time"* yes; *"future dynamic information from the ground truth"* never).
It enters the measurement encoder (refc.py:2025-2026, ``v0 / 10``) and the S2
reachability band (refc.py:2132, ``v_ms``); E11 keeps it out of every goal
node (pinned by tests/test_refc_v3.py).

    cl          T1  ONE forward at t0: observed frames + measured v0 + the TRUE
                    v7.2 nav token (the input the run trained on). The plan
                    horizon (6 s) covers the scored horizon, so one tick covers
                    the scored path — the SAME rule refav1's adapter applies
                    (REFAV1_ARM.md §3: "one MPC tick covers the scored path").
                    Mirror of ``t1_eval.roll_closed`` (t1_eval.py:753): there the
                    loop is closed through the WM's IMAGINED next latent; refcv3
                    has no imagination, so the loop collapses to the single
                    tick. NOTHING recorded after the window origin enters.
    cl_navshuf  T1  the same forward with nav_cmd PERMUTED across nav-valid
                    eval windows (the E13 eval OBLIGATION, refc_v3.py:433-436).
    cl_nonav    T1  the same forward with nav_cmd=None -> index 0 ('follow'),
                    the convention EVERY published REF-C number used
                    (refc.py:343-344). The three conditionings are three rows
                    of ONE batched forward — same frames, same v0.
    ha          T1  HOLD-ACTION control: the (a, kappa) that CLOSES at t0
                    (a = (v[t0]-v[t0-1])/0.1, kappa = actions[t0-1, 0]) held
                    for the horizon through the programme's unicycle
                    (``kinematic.rollout_unicycle`` via
                    ``refa_v1_plan.unicycle_paths``). Consumes no recorded
                    future (pinned bit-identical under a GT-future perturbation).
    ol          T0  the RECORDED future (a, kappa) integrated from v0 — the
                    KINEMATIC-CONTRACT control (must reproduce GT; a known
                    value), NOT a WM diagnostic.
    cl_rh       T0  (opt-in, --with-receding-arm) a RECEDING-HORIZON re-query
                    every grid step on the RECORDED frames, with v0 AND the pose
                    the model's output is applied at taken from the arm's OWN
                    rollout (never from GT poses; mirrors
                    ``t1_eval.roll_closed_grounding``'s implied-controls
                    feedback). ⛔ Stamped T0 BY DEFAULT: the recorded frames at
                    tick j >= 1 were captured along the HUMAN's path, so the
                    camera carries GT future ego dynamics (the doctrine's "T1 =
                    perception fixed at t0" is violated; T2 = re-render is what
                    would make it admissible, and T2 is not provisioned). The
                    Master Mind may re-stamp it with --tiers cl_rh=T1 — that
                    decision is then recorded in the record, never implied.

ESTIMATOR: full-set pooled point estimates; intervals = episode-cluster
bootstrap (``taniteval.ci``), paired across arms on the same windows.
``overlapping_holdout_se`` is never used (it biases the POINT ESTIMATE).

THE GRID. The model emits 8 slots at V3_HORIZONS = (5,10,15,20,30,40,50,60)
x 0.1 s. ``t1_eval``'s dump contract is a UNIFORM ``[N, K, 2]`` grid, so the
dump is an INDEX-SELECT of the model's own slots — never an interpolation:
    --grid 2s   dt 0.5 s, K 4  -> slots 0..3     (the seam / admission grid)
    --grid 6s   dt 1.0 s, K 6  -> slots 1,3,4,5,6,7 (0.5 s and 1.5 s dropped)
GT is read at exactly those instants from the RAW 10 Hz poses
(``refb_labels.waypoint_targets``, the trainer's own target function).

DISTANCE-KEEPING (backlog R1). The banked B1 EVAL lead block
(``refav1_arm.LEAD_BLOCK_DEFAULT``) is on a 0.2 s / K=10 grid keyed by
``(clip_id, RAW frame)``. refcv3's grid is not a subset of it at 0.5 s, so the
join is done on the COMMON instants — {1.0, 2.0} s for --grid 2s — by
index-select on BOTH sides (no resampling of the lead track, no interpolation
of the path), through ``refav1_arm.join_lead_block``'s own guards (grid,
label-free speed proof, NO_LABEL never scored as free flow). The block that
reaches the record says which instants it was scored on. A finer read is a
WORK ITEM: ``taniteval/tools/build_lead_block_b1.py --dt 0.5 --k 4``.

LABEL TIMING (a finding, not a fix — the trainer is read-only). The v7.2 record
is anchored on the RAW clip timeline; a v2ep PROVIDER drops the first
n_stack-1 frames (``v2_dataset._scan_meta``: ``poses[k:]``), so raw = provider
+ (n_stack-1) (``s2_labels.py:730-734`` applies it; ``refav1_loader`` needs no
offset because it reads the RAW dict). ``refc_v3_train.V3Dataset.__getitem__``
uses the PROVIDER index ``(t + w - 1) * 0.1`` — 0.2 s early on 9-channel B1.
The heads were TRAINED under that timing, so it is the PRIMARY label timing
here (mirror the supervision); the RAW-timeline labels are banked beside it
and the disagreement is COUNTED in the manifest (``label_timing``).

DUMP SCHEMA — two files per episode, so ``t1_eval.py --analyze-only`` stays valid:
    <dump>/ep{fi:03d}.npz            the t1_eval contract (unchanged):
        g [N,K,2]  cl cl_navshuf cl_nonav ha ol [cl_rh] [N,K,2]
        ws [N] RAW frame of the window origin  eid [1]  clip_index [1]  v0 [N]
    <dump>/decisions/ep{fi:03d}.npz  the refcv3 sidecar (see ``_SIDECAR_DOC``)
    <dump>/manifest.json             provenance: model rebuild + cross-checks,
                                     grid, tiers, T1 definition, nav policy,
                                     label join, label timing, episodes.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------- #
# path bootstrap — through the SIBLING refav1 adapter, whose _bootstrap_paths  #
# evicts the taniteval namespace shadow and preflights taniteval.ci            #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                       # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                       # <repo>
_SCRIPTS = os.path.join(_REPO, "stack", "scripts")        # refc_v3_train.py lives here


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: the refav1 adapter — the dump contract, the t1_eval import, the lead-block
#: join, the per-window components and the paired families are ITS; this file
#: reuses them so there is one implementation of each in the programme.
ra = _load_by_path("refav1_arm_for_refcv3", os.path.join(_HERE, "refav1_arm.py"))
t1 = ra.t1

_TRAINER = None


def trainer():
    """``stack/scripts/refc_v3_train.py`` by path (cached). The model is REBUILT
    through its ``_pin_trainer_cfg`` / ``build_parser`` and the eval windows
    through its ``V3Dataset``, so the adapter cannot drift from the trainer."""
    global _TRAINER
    if _TRAINER is None:
        if _SCRIPTS not in sys.path:
            sys.path.insert(0, _SCRIPTS)
        _TRAINER = _load_by_path("refc_v3_train_for_refcv3_arm",
                                 os.path.join(_SCRIPTS, "refc_v3_train.py"))
    return _TRAINER


# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #
DT_FRAME = 0.1            # the 10 Hz corpus tick (refb_labels.DT_DEFAULT)
#: dump grids: name -> (dt_s, K). Slot indices are DERIVED from the model's own
#: horizons at run time (never hardcoded) and a grid the model cannot serve by
#: index-select is REFUSED.
GRIDS = {"2s": (0.5, 4), "6s": (1.0, 6)}
ARM_TIERS = {"cl": "T1", "cl_navshuf": "T1", "cl_nonav": "T1", "ha": "T1",
             "ol": "T0", "cl_rh": "T0"}
ARM_MEANING = {
    "cl": "T1 — ONE forward at t0 (observed frames + measured v0 + TRUE v7.2 "
          "nav); the 6 s plan covers the scored horizon; nothing after t0 enters",
    "cl_navshuf": "T1 — as cl with nav_cmd PERMUTED across nav-valid windows "
                  "(E13 eval obligation)",
    "cl_nonav": "T1 — as cl with nav_cmd=None -> index 0 'follow' (the "
                "published REF-C eval convention)",
    "ha": "T1 — hold the (a, kappa) that CLOSES at t0 for the horizon "
          "(programme unicycle); consumes no recorded future",
    "ol": "T0 — the RECORDED future (a, kappa) integrated from v0: the "
          "kinematic-contract control (must reproduce GT), NOT a WM diagnostic",
    "cl_rh": "T0 (default stamp) — receding-horizon re-query on RECORDED frames "
             "with v0/pose from the arm's OWN rollout; the recorded camera moved "
             "along the HUMAN path so GT future dynamics enter through "
             "perception; --tiers cl_rh=T1 is a Master-Mind decision",
}
_TIER_NOTE = dict(t1._TIER_NOTE)
NAV_SOURCES = ("auto", "v72", "none")
_UNVERIFIED_ON_REAL_CKPT = (
    "UNVERIFIED on a real checkpoint — this box may not contact the training "
    "pod (tanitad-refcv3) or Thor; validated on a random-init tiny RefCV3 + "
    "synthetic v2ep slice only (stack/tests/test_refcv3_arm.py)")
_SIDECAR_DOC = {
    "ws": "[N] RAW frame of the window origin t0 (provider t0 + n_stack-1)",
    "t0_provider": "[N] provider-view index of t0 (the trainer's window index t+w-1)",
    "v0_fed": "[N] the v0 the model was conditioned on = poses[t0, 3] (measured)",
    "nav_cmd / nav_cmd_shuf / nav_valid": "[N] legacy refb.NAV_COMMANDS index fed "
        "under nav_true / nav_shuffled; valid=False rows carry the index-0 default",
    "lat_label / lon_label": "[N] v7.2 a_tac ids (v7_labels.HEADS order) under the "
        "TRAINER's window timing (provider index, refc_v3_train.V3Dataset); -100 = "
        "out of band / no record",
    "lat_label_raw / lon_label_raw": "[N] the same under the RAW clip timeline "
        "(raw = provider + n_stack-1); -100 likewise",
    "in_band_trainer / in_band_raw": "[N] bool — the two timings' band membership",
    "route_label": "[N] refb.ROUTE_CLASSES index derived from the v7.2 nav token "
        "(refav1 convention; ECHO of the fed nav by construction); -100 = none",
    "route_label_v21 / route_valid_v21": "[N] refb_labels.route_from_future_v21 — the "
        "trainer's own future-derived route aux target; -100 where invalid",
    "lat_kin3 / lon_kin3": "[N] refc_tactical.window_factored_labels over the GT 2 s "
        "future (the core heads' kinematic label)",
    "{lat,lon}_core_pred_{cond}": "[N] argmax of the CORE's lat/lon heads (kin3, "
        "3-way; man_prior_tau-adjusted decision) under nav_true/nav_shuffled/nav_zero",
    "{lat,lon}_tac_pred_{cond}": "[N] argmax of the HIERARCHY's z_tac heads (v7.0 "
        "8-way) — -1 on a flat arm",
    "route_pred_{cond}": "[N] argmax of route_logits (3-way)",
    "sel_idx_{cond} / sel_idx_base_{cond}": "[N] selected candidate after / before "
        "the E9 goal graft (-1 on a flat arm)",
    "goal_dist_sel_{cond}": "[N] |endpoint(selected) - predicted 2 s goal| (m)",
    "g_tac_{cond}": "[N, n_tau, 4] predicted tactical goals (x, y, heading, speed)",
    "goal_tac_label / goal_tac_valid": "[N, n_tau, 4] / [N, n_tau] hindsight goals "
        "(refb_labels.goal_tac_targets — a LABEL, uses the future)",
    "g_str_{cond}": "[N, 3] predicted strategic goal (unit bearing, tanh dist_pref)",
    "gstr_label_bearing / gstr_label_dist / gstr_label_valid": "LAN-derived strategic "
        "goal label (RefCModel.goal_targets) — only when the run trained --goal-str",
    "goal_gate_value / seam_goal_sel_* / reach_clipped_frac": "[N] E9 gate + seam "
        "telemetry + S2 band clipping per window (row = nav_true)",
    "law_mse_model / law_mse_persist / law_mse_zero / law_tgt_energy": "[N] the LAW "
        "world-model aux at T0 (law_pred vs encode_pooled(frame t0+0.5 s)) with the "
        "persist-last-latent floor and the zero control",
    "ha_control": "[N, 2] the held (a, kappa)",
    "rh_v0_fed / rh_gt_v": "[N, K] (cl_rh only) v0 fed at each tick from the OWN "
        "rollout vs the GT speed at the same frames",
}


def _p(*a):
    print(*a, flush=True)


_refused = ra._refused


# --------------------------------------------------------------------------- #
# config (de)serialisation — nested dataclasses, tuples survive JSON            #
# --------------------------------------------------------------------------- #
def cfg_to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def _default_of(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


def cfg_from_dict(cls, d: dict):
    """JSON dict -> dataclass ``cls`` (recursively), refusing unknown fields by
    name: a checkpoint written by a NEWER model file must be evaluated with
    that file, never with fields silently dropped."""
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise SystemExit(f"[refcv3_arm] the config carries fields this "
                         f"{cls.__name__} does not know: {unknown} — sync the "
                         f"model file; refusing to drop them silently")
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        dflt = _default_of(f)
        if dataclasses.is_dataclass(dflt) and isinstance(v, dict):
            kw[f.name] = cfg_from_dict(type(dflt), v)
        elif isinstance(dflt, tuple) and isinstance(v, list):
            kw[f.name] = tuple(v)
        else:
            kw[f.name] = v
    return cls(**kw)


# --------------------------------------------------------------------------- #
# model loading — REBUILT THROUGH THE TRAINER, cross-checked, strict            #
# --------------------------------------------------------------------------- #
def rebuild_config(config: dict):
    """``(cfg, train_args, source)`` from the run's ``config.json``.

    ``refc_v3_train.train`` writes NO model config — the model is a function of
    ``--arm/--size/--smoke/--image-hw/--v7-labels/--goal-str/--graft-lan``, so
    it is rebuilt HERE through the trainer's own ``build_parser`` +
    ``_pin_trainer_cfg`` on the recorded ``argv``. A config.json carrying the
    adapter extension ``refcv3_arm_model_cfg`` (a full RefCV3Config dict) is
    rebuilt from that instead; ``argv`` is still parsed for the run knobs
    (mode / nav source / label flags) when present.
    """
    tr = trainer()
    from tanitad.refs import refc
    from tanitad.refs import refc_v3 as v3
    argv = config.get("argv")
    args = None
    if argv:
        try:
            args = tr.build_parser().parse_args(list(argv))
        except SystemExit as ex:
            raise SystemExit(f"[refcv3_arm] config.json argv does not parse "
                             f"with this box's refc_v3_train.build_parser "
                             f"({ex}) — trainer/checkpoint version skew") from None
    if isinstance(config.get("refcv3_arm_model_cfg"), dict):
        cfg = cfg_from_dict(v3.RefCV3Config, config["refcv3_arm_model_cfg"])
        src = "config.json[refcv3_arm_model_cfg] (explicit RefCV3Config)"
    else:
        if args is None:
            raise SystemExit("[refcv3_arm] config.json carries neither argv nor "
                             "refcv3_arm_model_cfg — the model cannot be rebuilt")
        hier = args.arm == "hier"
        base = (v3.refc_v3_smoke_config(hier) if args.smoke
                else v3.refc_v3_sized_config(args.size, hier=hier))
        cfg = tr._pin_trainer_cfg(base, args)
        if args.graft_lan or args.goal_str:
            cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))
        src = ("config.json[argv] -> refc_v3_train.build_parser + "
               "_pin_trainer_cfg (the trainer's own build path)")
    return cfg, args, src


def cross_check_config(config: dict, cfg, model) -> dict:
    """Every fact config.json states about the model must hold for the rebuilt
    one. A contradiction is a REFUSAL naming both values (the
    ``adopt_ckpt_geometry`` lesson in t1_eval): never a silent override."""
    from tanitad.refs import refc_v3 as v3
    checks = {}
    conflict = []

    def _chk(name, have, want):
        checks[name] = {"config_json": want, "rebuilt": have}
        if want is not None and have != want:
            conflict.append(f"{name}: config.json {want!r} vs rebuilt {have!r}")

    if "arm" in config:
        _chk("arm", "hier" if cfg.hier else "flat", config["arm"])
    if "image_hw" in config:
        _chk("image_hw", list(cfg.core.encoder.image_hw()), list(config["image_hw"]))
    if "tac_vocab_version" in config:
        _chk("tac_vocab_version", cfg.tac_vocab_version, config["tac_vocab_version"])
    if "horizons" in config:
        _chk("horizons", list(cfg.core.trajectory.horizons), list(config["horizons"]))
    if "goal_tau_steps" in config:
        _chk("goal_tau_steps", list(cfg.goal_tau_steps), list(config["goal_tau_steps"]))
    if isinstance(config.get("param_breakdown"), dict):
        bd = v3.param_breakdown_v3(model)
        _chk("param_breakdown", {k: int(v) for k, v in bd.items()},
             {k: int(v) for k, v in config["param_breakdown"].items()})
    if conflict:
        raise SystemExit("[refcv3_arm] ⛔ config.json CONTRADICTS the rebuilt "
                         "model — refusing rather than guessing which describes "
                         "the weights:\n  " + "\n  ".join(conflict))
    return checks


def load_model(ckpt_path: str, config_path: str | None = None,
               device: str = "cpu", allow_nonstrict: bool = False):
    """``(model, cfg, train_args, provenance)`` — rebuilt, cross-checked, STRICT."""
    import torch
    from tanitad.refs import refc_v3 as v3
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        raise SystemExit(f"[refcv3_arm] {ckpt_path} has no 'model' key — not a "
                         f"refc_v3_train.py checkpoint")
    side = config_path or os.path.join(
        os.path.dirname(os.path.abspath(ckpt_path)), "config.json")
    if not os.path.exists(side):
        raise SystemExit(f"[refcv3_arm] no config.json at {side} — the trainer "
                         f"writes one beside ckpt.pt and the model cannot be "
                         f"rebuilt without it (pass --config)")
    with open(side, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise SystemExit(f"[refcv3_arm] {side} is not a config dict")
    cfg, targs, src = rebuild_config(config)
    model = v3.RefCV3Model(cfg)
    checks = cross_check_config(config, cfg, model)
    try:
        res = model.load_state_dict(ck["model"], strict=False)
    except RuntimeError as ex:
        raise SystemExit(
            f"[refcv3_arm] ⛔ the rebuilt model and the weights DISAGREE ON "
            f"SHAPE — the config.json/argv describe a different build than the "
            f"checkpoint. Refusing; fix the config, never the weights.\n"
            f"{str(ex)[:1500]}") from None
    strict_rep = {"missing_keys": list(res.missing_keys),
                  "unexpected_keys": list(res.unexpected_keys)}
    if (res.missing_keys or res.unexpected_keys) and not allow_nonstrict:
        raise SystemExit(f"[refcv3_arm] ⛔ NON-STRICT LOAD: missing "
                         f"{list(res.missing_keys)[:8]} unexpected "
                         f"{list(res.unexpected_keys)[:8]} — the checkpoint and "
                         f"the rebuilt model disagree. Pass --allow-nonstrict "
                         f"only for a deliberate diagnostic, and say so.")
    # ---- EVAL-TIME NEUTRALISATION, recorded ------------------------------
    # refc_select.apply_seam_clamp RAISES after `seam_fail_patience`
    # CONSECUTIVE saturated calls (refc_select.py:320-332). The counter lives
    # on the model, not on a batch, so a trained gate that sits above the clamp
    # would kill a 24k-window eval at window 50 — a TRAINING-DYNAMICS report
    # firing inside an eval. The clamp itself stays (it is the emitted score);
    # only the fail-loud patience is switched off here, on BOTH surfaces the
    # v3 forward uses, and the saturation telemetry is banked per window
    # instead (seam_goal_sel_* in the sidecar). No parameter changes.
    overrides = {}
    if getattr(cfg, "seam_fail_patience", 0):
        overrides["v3.seam_fail_patience"] = {"trained": cfg.seam_fail_patience, "eval": 0}
        cfg.seam_fail_patience = 0
    sel = model.core.decoder.sel
    if getattr(sel, "seam_fail_patience", 0):
        overrides["core.decoder.sel.seam_fail_patience"] = {
            "trained": sel.seam_fail_patience, "eval": 0}
        sel.seam_fail_patience = 0
    model.core.cfg.seam_fail_patience = 0
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    steps = (cfg.core.decoder.diffusion_steps
             if (targs is None or targs.mode == "diffusion") else 0)
    prov = {"ckpt": ckpt_path, "step": ck.get("step"),
            "config_json": side, "rebuilt_from": src,
            "config_cross_checks": checks,
            "state_dict_load": strict_rep,
            "eval_time_cfg_overrides": overrides,
            "cfg": cfg_to_dict(cfg),
            "param_breakdown": v3.param_breakdown_v3(model),
            "arm": "hier" if cfg.hier else "flat",
            "tac_vocab_version": cfg.tac_vocab_version,
            "decoder_steps": int(steps),
            "decoder_mode": (targs.mode if targs is not None else "diffusion(assumed)"),
            "nav_from_v7_trained": bool(config.get("nav_from_v7", False)),
            "nav_cmd_derivation_trained": config.get("nav_cmd_derivation"),
            "train_labels_manifest": config.get("v7_labels"),
            "eval_labels_md5_at_train": (((config.get("nav_from_v7_stats") or {})
                                          .get("eval") or {}).get("label_md5")),
            "goal_str_trained": bool(targs.goal_str) if targs is not None else False,
            "graft_lan": bool(targs.graft_lan) if targs is not None else False,
            "lan_arclengths": (list(targs.lan_arclengths) if targs is not None else None),
            "lan_min_lead_m": (targs.lan_min_lead_m if targs is not None else None),
            "admission_sigma_m": config.get("admission_sigma_m", cfg.admission_sigma_m),
            "u8_batches_trained": config.get("u8_batches")}
    return model, cfg, targs, prov


# --------------------------------------------------------------------------- #
# grid + kinematics                                                            #
# --------------------------------------------------------------------------- #
def grid_slots(horizons, grid: str) -> dict:
    """``--grid`` -> the model slots that ARE that grid (index-select only)."""
    if grid not in GRIDS:
        raise SystemExit(f"[refcv3_arm] unknown --grid {grid!r}; known {sorted(GRIDS)}")
    dt, k = GRIDS[grid]
    hz = [int(h) for h in horizons]
    need = [int(round(10 * dt * j)) for j in range(1, k + 1)]
    missing = [h for h in need if h not in hz]
    if missing:
        raise SystemExit(f"[refcv3_arm] --grid {grid} needs model slots at "
                         f"{need} (0.1 s steps) but the model emits {hz}; "
                         f"missing {missing}. A grid the model cannot serve by "
                         f"index-select is refused (no interpolation).")
    return {"name": grid, "dt_s": dt, "k": k, "horizons": need,
            "slots": [hz.index(h) for h in need],
            "dropped_model_slots": [h for h in hz if h not in need],
            "n_frames": need[-1]}


def recorded_controls(v, kap, t0: int, n: int):
    """The RECORDED future (a, kappa) for frames t0 .. t0+n-1 (0.1 s each):
    a_f = (v[f+1] - v[f]) / 0.1 (pose speed, exact on the grid), kappa_f =
    actions[f, 0] — the MEASURED true-kappa channel (refav1_loader docstring:
    the stored order is (kappa, accel-like)). Frames beyond the episode clamp."""
    import torch
    T = int(v.shape[0])
    f = torch.arange(t0, t0 + n)
    fn = torch.clamp(f + 1, max=T - 1)
    fc = torch.clamp(f, max=T - 1)
    a = (v[fn] - v[fc]) / DT_FRAME
    return torch.stack([a, kap[fc]], dim=-1)                         # [n, 2]


def hold_controls(v, kap, t0: int):
    """The (a, kappa) that CLOSES at t0: reads v[t0], v[t0-1], kappa[t0-1] —
    every frame <= t0, so NOTHING recorded after t0 enters."""
    if t0 < 1:
        raise ValueError("hold-action needs t0 >= 1 (one closed step before t0)")
    import torch
    a = (v[t0] - v[t0 - 1]) / DT_FRAME
    return torch.stack([a, kap[t0 - 1]])                               # [2]


def integrate_select(controls, v0: float, sel_frames) -> "np.ndarray":
    """(a, kappa) [n, 2] at 0.1 s from v0 through the programme's ONE unicycle,
    then INDEX-SELECT the grid frames (1-based) -> [1, K, 2]."""
    import torch
    path = ra.paths_from_controls(controls, v0, DT_FRAME, int(controls.shape[0]))
    idx = torch.tensor([int(s) - 1 for s in sel_frames])
    return path[:, idx].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# the eval windows — the TRAINER's dataset, future-frame decode removed        #
# --------------------------------------------------------------------------- #
def make_eval_dataset_class():
    tr = trainer()
    import torch

    class EvalV3Windows(tr.V3Dataset):
        """``refc_v3_train.V3Dataset`` (nav / v7.2 labels / route v21 / goals /
        ext future — the trainer's exact window contract) with ONE change: the
        FUTURE FRAMES are never consumed at eval (only the loss's LAW target
        reads them, refc_v3_train.py:580), so ``_window_u8`` decodes the
        observed window only and, for the T0 LAW diagnostic, the single frame
        LAW_AHEAD steps ahead. Frames leave uint8 and become float32 [0,1]
        through the trainer's own ``frames_to_device``."""
        u8_frames = True

        def _window_u8(self, i: int) -> dict:
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            la = t + w + tr.LAW_AHEAD - 1
            return {
                "frames": ep.frames[t:t + w],
                "actions": ep.actions[t:t + w],
                "future_frames": torch.zeros(0, dtype=torch.uint8),
                "law_frame": ep.frames[la:la + 1],
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }

    return EvalV3Windows


def build_corpus(a, cfg, prov: dict):
    """The eval episodes (v2 providers), the trainer's window dataset with the
    v7.2 label join and the nav source, plus the join report."""
    tr = trainer()
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import (build_v2_providers, load_or_build_manifest,
                                         stable_episode_id)
    man = load_or_build_manifest(a.episodes, verbose=False)
    eps = build_v2_providers([a.episodes], lru_size=a.lru, verbose=False)
    files = list(man["files"])
    clip_ids = [str(c) for c in man["clip_id"]]
    if len(eps) != len(files):
        raise SystemExit(f"[refcv3_arm] {len(eps)} providers for {len(files)} "
                         f"clip files under {a.episodes} — manifest drift")
    keep = list(range(len(eps)))
    if a.episodes_n:
        keep = keep[:int(a.episodes_n)]
    eps = [eps[i] for i in keep]
    files = [files[i] for i in keep]
    clip_ids = [clip_ids[i] for i in keep]
    if not eps:
        raise SystemExit(f"[refcv3_arm] no *.v2ep.pt under {a.episodes}")
    # geometry + channels are asserted against the EPISODES (the trainer's rule)
    eh, ew = cfg.core.encoder.image_hw()
    fh, fw = int(eps[0].frames.shape[-2]), int(eps[0].frames.shape[-1])
    ch = int(eps[0].frames.shape[1])
    if (fh, fw) != (eh, ew):
        raise SystemExit(f"[refcv3_arm] ⛔ geometry mismatch: encoder built for "
                         f"{eh}x{ew}, corpus emits {fh}x{fw}")
    if ch != cfg.core.encoder.in_channels:
        raise SystemExit(f"[refcv3_arm] ⛔ channel mismatch: encoder expects "
                         f"{cfg.core.encoder.in_channels}, corpus emits {ch}")
    n_stack_off = max(ch // 3 - 1, 0)      # raw = provider + (n_stack-1)
    Ds = make_eval_dataset_class()
    ds = Ds(eps, window=cfg.core.window, max_horizon=20,
            channels=cfg.core.encoder.in_channels)
    ds.u8_frames = True
    # ---- v7.2 labels (MANDATORY for a v7.2-trained arm) ---------------------
    if not a.labels:
        raise SystemExit("[refcv3_arm] --labels (the v7.2 EVAL blob) is required: "
                         "the tactical heads and the nav token are defined by it")
    if a.nav and os.path.abspath(a.nav) != os.path.abspath(a.labels):
        raise SystemExit("[refcv3_arm] the trainer reads nav AND labels from ONE "
                         "blob (--v7-labels / --eval-labels); pass --nav equal to "
                         "--labels or omit it")
    labels, lman = v7l.load_v7_labels(a.labels, allow_oracle_nav=True)
    by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    ds.v7_by_sid = by_sid
    ds.v7_dt = DT_FRAME
    hit = sum(1 for e in eps if int(e.episode_id) in by_sid)
    if hit == 0:
        raise SystemExit(f"[refcv3_arm] ⛔ {a.labels} joined ZERO of {len(eps)} "
                         f"episodes — wrong blob for this corpus (md5={lman.md5})")
    join = {"labels": {"path": a.labels, "md5": lman.md5, "n_records": lman.n_records,
                       "n_episodes": len(eps), "n_joined": hit,
                       "n_missing": len(eps) - hit,
                       "join_key": "stable_episode_id(clip_id) (the trainer's key)",
                       **{k: v for k, v in lman.to_dict().items()
                          if k in ("schema_version", "vocab", "allow_oracle_nav",
                                   "divergences")}}}
    md5_train_eval = prov.get("eval_labels_md5_at_train")
    if md5_train_eval and md5_train_eval != lman.md5:
        _p(f"[labels] WARNING: this blob md5={lman.md5} != the eval blob the run "
           f"used in training ({md5_train_eval}) — a different eval set; the "
           f"manifest records both")
    # ---- nav source ---------------------------------------------------------
    src = a.nav_source
    if src == "auto":
        src = "v72" if prov.get("nav_from_v7_trained") else "none"
    nav_stats = None
    if src == "v72":
        nav_stats = ds.enable_nav_from_v7(lman)            # the trainer's own path
        if not prov.get("nav_from_v7_trained"):
            _p("[nav] WARNING: the run trained on the v1 nav derivation but is "
               "being fed v7.2 tokens (--nav-source v72) — an input-distribution "
               "shift; recorded in the manifest")
    join["nav"] = {"source": src, "stats": nav_stats,
                   "trained_on": prov.get("nav_cmd_derivation_trained"),
                   "rule": ("v72: the clip's v7.2 nav_command token -> refb."
                            "NAV_COMMANDS index (refc_v3_train.NAV_TOKEN_TO_LEGACY, "
                            "position-pinned); a clip without a record feeds "
                            "index 0 + nav_valid=False. none: nav_cmd=0 on every "
                            "window (the published REF-C convention), no shuffle "
                            "arm — the v1 derivation reads 15-25 s of FUTURE poses "
                            "and is not an inference input")}
    return eps, files, clip_ids, ds, by_sid, lman, join, src, n_stack_off


# --------------------------------------------------------------------------- #
# the roll — writes the dump                                                    #
# --------------------------------------------------------------------------- #
def _heads_of(out, r: int, hier: bool) -> dict:
    """The declared decisions of batch row ``r``."""
    h = {"lat_core": int(out["lat_decision"][r]) if "lat_decision" in out else -1,
         "lon_core": int(out["lon_decision"][r]) if "lon_decision" in out else -1,
         "route": int(out["route_logits"][r].argmax(-1)),
         "sel_idx": int(out["sel_idx"][r])}
    if hier:
        h.update(lat_tac=int(out["lat_logits_tac"][r].argmax(-1)),
                 lon_tac=int(out["lon_logits_tac"][r].argmax(-1)),
                 sel_idx_base=int(out["sel_idx_base"][r]),
                 goal_dist_sel=float(out["goal_dist"][r, int(out["sel_idx"][r])]))
    else:
        h.update(lat_tac=-1, lon_tac=-1, sel_idx_base=-1, goal_dist_sel=float("nan"))
    return h


def roll_receding(model, frames_proxy, t0: int, W: int, nav_t, v0: float,
                  steps: int, grid: dict, gt_speed, dev, tr) -> tuple:
    """cl_rh: re-query every grid step on the RECORDED frames; v0 and the pose
    the output is applied at come from the arm's OWN rollout.

    State in the t0 ego frame: position P, heading psi, speed v (= measured v0
    at tick 0). Tick j (time j*dt): the window ending at RAW-clip frame
    t0 + j*dt*10 (recorded), fed the OWN v; the executed segment is the FIRST
    grid slot (dt ahead) of the model's output, applied at P with rotation
    psi; the next heading is the chord tangent, the next speed the chord
    speed (``t1_eval.implied_controls``' convention: v = |d| / dt). Returns
    ``(path [K, 2], v_fed [K], gt_v [K])``.
    """
    import torch
    K, dt = grid["k"], grid["dt_s"]
    step_frames = int(round(dt * 10))
    first_slot = grid["slots"][0]
    pos = np.zeros(2)
    psi = 0.0
    v = float(v0)
    path, v_fed, gt_v = [], [], []
    for j in range(K):
        f_end = t0 + j * step_frames
        fr = tr.frames_to_device(frames_proxy[f_end - W + 1:f_end + 1][None], dev)
        with torch.no_grad():
            out = model(fr, nav_cmd=nav_t, v0=torch.tensor([v], device=dev),
                        steps=steps)
        d = out["traj"][0, first_slot].double().cpu().numpy()          # (dx, dy)
        v_fed.append(v)
        gt_v.append(float(gt_speed[f_end]))
        c, s = math.cos(psi), math.sin(psi)
        pos = pos + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        if float(np.hypot(d[0], d[1])) > 1e-6:
            psi = psi + math.atan2(d[1], d[0])
        v = float(np.hypot(d[0], d[1]) / dt)
        path.append(pos.copy())
    return np.stack(path), np.array(v_fed), np.array(gt_v)


def run_dump(a) -> dict:
    """Roll every arm over the eval grid and write the dump. Returns the manifest."""
    import torch
    from tanitad.data import v7_labels as v7l
    from tanitad.refs import refc
    from tanitad.refs import refc_tactical as tac

    tr = trainer()
    t_start = time.time()
    dev = a.device
    model, cfg, targs, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)
    hier = bool(cfg.hier)
    grid = grid_slots(cfg.core.trajectory.horizons, a.grid)
    K, dt = grid["k"], grid["dt_s"]
    steps = prov["decoder_steps"]
    eps, files, clip_ids, ds, by_sid, lman, join, nav_src, off = build_corpus(a, cfg, prov)
    W = int(cfg.core.window)
    refb_labels = tr.refb_labels
    _p(f"[model] {prov['ckpt']} step={prov['step']} arm={prov['arm']} params="
       f"{prov['param_breakdown']['total']:,} vocab={cfg.tac_vocab_version} "
       f"steps={steps} rebuilt<-{prov['rebuilt_from']}")
    _p(f"[corpus] {len(eps)} episodes, {len(ds)} trainer windows (W={W}, "
       f"max_horizon 20), n_stack offset {off}; labels joined "
       f"{join['labels']['n_joined']}/{len(eps)}; nav source={nav_src}")

    # ---- window selection ---------------------------------------------------
    stride = max(1, int(a.window_stride))
    by_ep: dict[int, list] = {}
    for i, (e_i, t) in enumerate(ds.index):
        if t % stride == 0:
            by_ep.setdefault(e_i, []).append(i)
    sel = [(i, e_i) for e_i in sorted(by_ep) for i in by_ep[e_i]]
    if not sel:
        raise SystemExit("[refcv3_arm] the stride selected zero windows")
    # nav for every selected window (the dataset's own values), then the shuffle
    nav_true = np.zeros(len(sel), dtype=np.int64)
    nav_valid = np.zeros(len(sel), dtype=bool)
    if nav_src == "v72":
        for n_, (i, e_i) in enumerate(sel):
            nid = ds._nav_by_sid.get(int(eps[e_i].episode_id))
            nav_true[n_] = 0 if nid is None else int(nid)
            nav_valid[n_] = nid is not None
    nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)
    arms = ["cl", "ha", "ol"]
    if nav_src == "v72" and not a.no_navshuf:
        arms.append("cl_navshuf")
    if not a.no_nonav_arm and nav_src == "v72":
        arms.append("cl_nonav")
    if a.with_receding_arm:
        arms.append("cl_rh")
    _p(f"[grid] {grid['name']}: dt={dt} K={K} slots={grid['slots']} "
       f"(model horizons {list(cfg.core.trajectory.horizons)}) stride={stride} "
       f"windows={len(sel)} arms={arms} nav_shuffle={shuf_stats['n_changed']}/"
       f"{shuf_stats['n_windows']} changed")
    os.makedirs(a.dump_dir, exist_ok=True)
    os.makedirs(os.path.join(a.dump_dir, "decisions"), exist_ok=True)

    lan_cfg = None
    if prov.get("goal_str_trained"):
        from tanitad.data.lan import LanConfig as DataLanConfig, lan_window_features
        lan_cfg = DataLanConfig(arclengths_m=tuple(prov["lan_arclengths"]),
                                min_lead_m=float(prov["lan_min_lead_m"]))
    conds = ("nav_true", "nav_shuffled", "nav_zero")
    tau2 = model._tau_slot_2s() if hier else None
    episodes_manifest = []
    n_done = n_dropped = 0
    t_fwd_first = None
    band_disagree = 0
    sel_frames = grid["horizons"]                       # 1-based frame offsets
    n_int = grid["n_frames"]
    n_by_ep = {}
    for fi, e_i in enumerate(sorted(by_ep)):
        ep = eps[e_i]
        poses = ep.poses.float()
        v_ep, kap_ep = poses[:, 3], ep.actions[:, 0].float()
        T = int(poses.shape[0])
        lab = by_sid.get(int(ep.episode_id))
        acc = {kk: [] for kk in ["g", "v0"] + arms}
        dec: dict[str, list] = {}
        ws = []
        for n_ in [k for k, (i, e) in enumerate(sel) if e == e_i]:
            i = sel[n_][0]
            _, t = ds.index[i]
            t0 = t + W - 1
            item = ds[i]
            fut_ext = item["future_poses_ext"].float()          # [60, 4]
            fut_ok = item["future_valid_ext"]
            if not all(bool(fut_ok[h - 1]) for h in sel_frames):
                n_dropped += 1
                continue
            pose_last = item["pose_last"].float()
            v0 = float(pose_last[3])
            if abs(v0 - float(v_ep[t0])) > 1e-6:
                raise RuntimeError("pose_last[3] != poses[t0, 3] — dataset drift")
            nav_t = torch.tensor([int(nav_true[n_]), int(nav_shuf[n_]), 0],
                                 dtype=torch.long, device=dev)
            if nav_src == "v72" and int(item["nav_cmd"]) != int(nav_true[n_]):
                raise RuntimeError("nav table / dataset disagree — loader drift")
            frames = tr.frames_to_device(item["frames"][None].repeat(3, 1, 1, 1, 1), dev)
            v0_t = torch.full((3,), v0, dtype=torch.float32, device=dev)
            tf = time.time()
            with torch.no_grad():
                out = model(frames, nav_cmd=nav_t, v0=v0_t, steps=steps)
                law_tgt = model.core.encode_pooled(
                    tr.frames_to_device(item["law_frame"], dev))          # [1, F]
            if t_fwd_first is None:
                t_fwd_first = time.time() - tf
            traj = out["traj"][:, grid["slots"]].float().cpu().numpy()   # [3, K, 2]
            # -- GT at exactly the grid instants (the trainer's own target fn) --
            g = refb_labels.waypoint_targets(pose_last[None], fut_ext[None],
                                             tuple(sel_frames)).numpy()   # [1, K, 2]
            # -- the two kinematic controls (0.1 s unicycle, then index-select) --
            ol = integrate_select(recorded_controls(v_ep, kap_ep, t0, n_int), v0, sel_frames)
            hold = hold_controls(v_ep, kap_ep, t0)
            ha = integrate_select(hold[None].expand(n_int, 2), v0, sel_frames)
            acc["g"].append(g.astype(np.float32))
            acc["cl"].append(traj[0:1])
            if "cl_navshuf" in arms:
                acc["cl_navshuf"].append(traj[1:2])
            if "cl_nonav" in arms:
                acc["cl_nonav"].append(traj[2:3])
            acc["ha"].append(ha.astype(np.float32))
            acc["ol"].append(ol.astype(np.float32))
            acc["v0"].append(np.array([v0], dtype=np.float32))
            if "cl_rh" in arms:
                rh_path, rh_v, rh_gt = roll_receding(
                    model, ep.frames, t0, W, nav_t[:1], v0, steps, grid, v_ep, dev, tr)
                acc["cl_rh"].append(rh_path[None].astype(np.float32))
                dec.setdefault("rh_v0_fed", []).append(rh_v[None].astype(np.float32))
                dec.setdefault("rh_gt_v", []).append(rh_gt[None].astype(np.float32))
            # -- the sidecar ----------------------------------------------------
            raw0 = t0 + off
            ws.append(int(raw0))
            dec.setdefault("t0_provider", []).append(int(t0))
            dec.setdefault("v0_fed", []).append(float(v0))
            dec.setdefault("nav_cmd", []).append(int(nav_true[n_]))
            dec.setdefault("nav_cmd_shuf", []).append(int(nav_shuf[n_]))
            dec.setdefault("nav_valid", []).append(bool(nav_valid[n_]))
            # labels: trainer timing (the dataset's) and RAW timing
            lat_tr = int(item["lat_v7"]) if "lat_v7" in item else v7l.IGNORE_ID
            lon_tr = int(item["lon_v7"]) if "lon_v7" in item else v7l.IGNORE_ID
            if lab is None:
                lat_raw = lon_raw = v7l.IGNORE_ID
                in_tr = in_raw = False
                route_lbl = v7l.IGNORE_ID
            else:
                lat_raw, lon_raw = v7l.tactical_class_ids(lab, raw0 * DT_FRAME)
                in_tr = v7l.window_in_band(lab, t0 * DT_FRAME)
                in_raw = v7l.window_in_band(lab, raw0 * DT_FRAME)
                tok = (v7l.oracle_nav(lab, lman) or {}).get("token")
                route_lbl = (v7l.IGNORE_ID if tok not in ra_nav_route_map()
                             else ra_nav_route_map()[tok])
            band_disagree += int(in_tr != in_raw)
            dec.setdefault("lat_label", []).append(lat_tr)
            dec.setdefault("lon_label", []).append(lon_tr)
            dec.setdefault("lat_label_raw", []).append(int(lat_raw))
            dec.setdefault("lon_label_raw", []).append(int(lon_raw))
            dec.setdefault("in_band_trainer", []).append(bool(in_tr))
            dec.setdefault("in_band_raw", []).append(bool(in_raw))
            dec.setdefault("route_label", []).append(int(route_lbl))
            rv = bool(item["route_valid"]) if "route_valid" in item else False
            dec.setdefault("route_label_v21", []).append(
                int(item["route_target"]) if rv else v7l.IGNORE_ID)
            dec.setdefault("route_valid_v21", []).append(rv)
            lk, lnk = tac.window_factored_labels(pose_last[None], fut_ext[None, :20])
            dec.setdefault("lat_kin3", []).append(int(lk[0]))
            dec.setdefault("lon_kin3", []).append(int(lnk[0]))
            for r, cname in enumerate(conds):
                h = _heads_of(out, r, hier)
                dec.setdefault(f"lat_core_pred_{cname}", []).append(h["lat_core"])
                dec.setdefault(f"lon_core_pred_{cname}", []).append(h["lon_core"])
                dec.setdefault(f"lat_tac_pred_{cname}", []).append(h["lat_tac"])
                dec.setdefault(f"lon_tac_pred_{cname}", []).append(h["lon_tac"])
                dec.setdefault(f"route_pred_{cname}", []).append(h["route"])
                dec.setdefault(f"sel_idx_{cname}", []).append(h["sel_idx"])
                dec.setdefault(f"sel_idx_base_{cname}", []).append(h["sel_idx_base"])
                dec.setdefault(f"goal_dist_sel_{cname}", []).append(h["goal_dist_sel"])
                if hier:
                    dec.setdefault(f"g_tac_{cname}", []).append(
                        out["g_tac"][r].float().cpu().numpy()[None])
                    dec.setdefault(f"g_str_{cname}", []).append(
                        out["g_str"][r].float().cpu().numpy()[None])
            if hier:
                dec.setdefault("goal_tac_label", []).append(
                    item["goal_tac"].float().numpy()[None])
                dec.setdefault("goal_tac_valid", []).append(
                    item["goal_tac_valid"].float().numpy()[None])
                dec.setdefault("goal_gate_value", []).append(float(out["goal_gate_value"]))
                dec.setdefault("seam_goal_sel_ratio_preclamp_mean", []).append(
                    float(out.get("seam_goal_sel_ratio_preclamp_mean", float("nan"))))
                dec.setdefault("seam_goal_sel_bound_frac", []).append(
                    float(out.get("seam_goal_sel_bound_frac", float("nan"))))
                if lan_cfg is not None:
                    f = torch.from_numpy(lan_window_features(poses, t0, lan_cfg))
                    b_, d_, ok_ = refc.RefCModel.goal_targets(f[None], lan_cfg.k)
                    dec.setdefault("gstr_label_bearing", []).append(
                        b_.float().numpy())
                    dec.setdefault("gstr_label_dist", []).append(float(d_[0]))
                    dec.setdefault("gstr_label_valid", []).append(bool(ok_[0]))
            dec.setdefault("reach_clipped_frac", []).append(
                float(out["sel_tele"].get("reach_frac_candidates_clipped", float("nan"))))
            # T0 LAW world-model aux: model vs persist-last-latent vs zero
            lp = out["law_pred"][0:1]
            dec.setdefault("law_mse_model", []).append(float(((lp - law_tgt) ** 2).mean()))
            dec.setdefault("law_mse_persist", []).append(
                float(((out["pooled"][0:1] - law_tgt) ** 2).mean()))
            dec.setdefault("law_mse_zero", []).append(float((law_tgt ** 2).mean()))
            dec.setdefault("law_tgt_energy", []).append(float((law_tgt ** 2).mean()))
            dec.setdefault("ha_control", []).append(hold.float().cpu().numpy()[None])
            n_done += 1
            if n_done == 1:
                est = t_fwd_first * len(sel) * (1 + (K if "cl_rh" in arms else 0))
                _p(f"[cost] first forward (3 rows) {t_fwd_first:.3f} s -> ESTIMATED "
                   f"{est / 3600:.2f} h of forward time for {len(sel)} windows on "
                   f"this device (decode/IO extra)")
        if not ws:
            _p(f"  [{fi + 1}/{len(by_ep)}] {files[e_i][:12]} 0 scoreable windows — skipped")
            continue
        np.savez_compressed(
            os.path.join(a.dump_dir, f"ep{len(episodes_manifest):03d}.npz"),
            **{kk: np.concatenate(v).astype(np.float32) for kk, v in acc.items()},
            ws=np.array(ws), eid=np.array([len(episodes_manifest)]),
            clip_index=np.array([e_i]))
        dec_np = {}
        for kk, v in dec.items():
            if isinstance(v[0], np.ndarray):
                dec_np[kk] = np.concatenate(v).astype(np.float32)
            elif isinstance(v[0], bool):
                dec_np[kk] = np.array(v, dtype=bool)
            elif isinstance(v[0], float):
                dec_np[kk] = np.array(v, dtype=np.float32)
            else:
                dec_np[kk] = np.array(v, dtype=np.int64)
        np.savez_compressed(os.path.join(a.dump_dir, "decisions",
                                         f"ep{len(episodes_manifest):03d}.npz"),
                            ws=np.array(ws), **dec_np)
        episodes_manifest.append({"file_index": len(episodes_manifest),
                                  "episode_index": int(e_i), "name": files[e_i],
                                  "clip_id": clip_ids[e_i],
                                  "stable_episode_id": int(ep.episode_id),
                                  "n_windows": len(ws)})
        n_by_ep[e_i] = len(ws)
        _p(f"  [{fi + 1}/{len(by_ep)}] {files[e_i][:12]} {len(ws)} windows "
           f"{time.time() - t_start:.0f}s")

    manifest = {
        "tool": "taniteval/tools/refcv3_arm.py",
        "model": prov,
        "t1_definition": T1_DEFINITION,
        "grid": {**grid, "window": W, "n_stack_offset": off,
                 "window_stride_frames": stride, "n_windows": n_done,
                 "n_windows_dropped_no_gt_at_horizon": n_dropped,
                 "n_episodes": len(episodes_manifest),
                 "n_episodes_available": len(eps),
                 "ws_is": "RAW clip frame of t0 = provider t0 + n_stack-1 (lead "
                          "block key; the label timeline)"},
        "arms": arms, "tiers": {x: ARM_TIERS[x] for x in arms},
        "arm_meaning": {x: ARM_MEANING[x] for x in arms},
        "nav": {**join["nav"], "shuffle": shuf_stats},
        "conditionings": list(conds),
        "hold_action_rule": hold_controls.__doc__,
        "recorded_action_rule": recorded_controls.__doc__,
        "label_timing": {
            "primary": "trainer (provider index t+w-1, refc_v3_train.V3Dataset)",
            "secondary": "raw clip timeline (provider + n_stack-1, s2_labels.py:730)",
            "offset_s": round(off * DT_FRAME, 3),
            "n_windows_band_disagree": int(band_disagree), "n_windows": n_done,
            "_finding": ("the trainer's in-band test runs on the PROVIDER index "
                         "while the record's t0/band live on the RAW timeline — "
                         "0.2 s early on a 9-channel cache. Escalated, not fixed "
                         "(trainer is read-only to this adapter)")},
        "corpus": {"episodes": a.episodes, "labels": a.labels,
                   "join_report": join["labels"]},
        "episodes": episodes_manifest,
        "sidecar_doc": _SIDECAR_DOC,
        "wallclock_s": round(time.time() - t_start, 1),
        "first_forward_s": t_fwd_first,
        "device": dev,
    }
    with open(os.path.join(a.dump_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    _p("REFCV3_DUMP_DONE")
    return manifest


_NAV_ROUTE = None


def ra_nav_route_map() -> dict:
    """v7 nav token -> refb.ROUTE_CLASSES index, from the refav1 loader's own
    table (the same derivation refav1's STRATEGIC block scores against)."""
    global _NAV_ROUTE
    if _NAV_ROUTE is None:
        from tanitad.data.refav1_loader import _NAV_TOKEN_TO_ROUTE
        from tanitad.refs.refb import ROUTE_CLASSES
        _NAV_ROUTE = {tok: ROUTE_CLASSES.index(r) for tok, r in _NAV_TOKEN_TO_ROUTE.items()}
    return _NAV_ROUTE


def nav_index_to_route_index() -> dict:
    """legacy nav index (refb.NAV_COMMANDS order) -> route index — the
    codebase's own table ``refb_labels._NAV_TO_ROUTE`` (never hand-typed)."""
    tr = trainer()
    return {int(k): int(v) for k, v in tr.refb_labels._NAV_TO_ROUTE.items()}


T1_DEFINITION = (
    "T1 for refcv3 (REFCV3_ARM.md §2): ONE forward of RefCV3Model at the window "
    "origin t0 with the OBSERVED frames, the MEASURED v0 = poses[t0, 3] (PI ruling "
    "2026-09-02) and the v7.2 nav token; the emitted 6 s path covers the scored "
    "horizon, so one tick covers the scored path (the refav1 rule). The model has "
    "no action-conditioned predictor, so t1_eval.roll_closed's imagined-latent loop "
    "collapses to that single tick; NOTHING recorded after t0 enters (frames, "
    "poses or actions). A receding-horizon re-query on RECORDED frames is NOT T1 "
    "(the camera moved along the human's path) and is offered only as the "
    "T0-stamped opt-in arm cl_rh."
)


# --------------------------------------------------------------------------- #
# distance-keeping on the COMMON grid of the dump and the banked lead block     #
# --------------------------------------------------------------------------- #
def lead_block_common_grid(path: str, dt: float, k: int):
    """Index-select the banked block onto the instants it SHARES with the dump.

    Returns ``(view, idx, meta, grid_info)`` or ``(None, None, meta, refusal)``.
    ``view`` is the block with ``ts_rel_s``/``leads`` restricted to the common
    instants and every ``gt_*`` column DROPPED (they were computed over the
    block's own 10-step grid; the GT reference is recomputed on the view from
    the dump's ``g``). The common instants must be uniformly spaced starting at
    their own spacing, or ``lead_metrics.distance_keeping``'s scalar dt would
    lie; otherwise REFUSED with the rebuild command.
    """
    blk, idx, meta = ra.load_lead_block_rows(path)
    ts = np.asarray(blk["ts_rel_s"], dtype=np.float64).reshape(-1)
    want = np.arange(1, k + 1, dtype=np.float64) * dt
    cols, rows, inst = [], [], []
    for c, w in enumerate(want):
        hit = np.flatnonzero(np.abs(ts - w) <= ra.LEAD_TS_TOL_S)
        if hit.size:
            cols.append(c)
            rows.append(int(hit[0]))
            inst.append(float(w))
    rebuild = (f"taniteval/tools/build_lead_block_b1.py --dt {dt} --k {k} (a block "
               f"on the dump's own grid makes the FULL horizon scoreable)")
    if not cols:
        return None, None, meta, _refused(
            f"the lead block grid {np.round(ts, 3).tolist()} shares NO instant with "
            f"the dump grid {np.round(want, 3).tolist()} — WORK ITEM: {rebuild}",
            "n/a", 0)
    gaps = np.diff(np.array(inst)) if len(inst) > 1 else np.array([inst[0]])
    if not (np.allclose(gaps, gaps[0]) and abs(inst[0] - gaps[0]) <= ra.LEAD_TS_TOL_S):
        return None, None, meta, _refused(
            f"the common instants {inst} are not a uniform grid starting at its own "
            f"spacing — a scalar dt would misstate the closing rate; WORK ITEM: "
            f"{rebuild}", "n/a", 0)
    view = {kk: v for kk, v in blk.items() if not str(kk).startswith("gt_")}
    view["ts_rel_s"] = np.array(inst, dtype=np.float64)
    view["leads"] = np.asarray(blk["leads"], dtype=np.float64)[:, rows]
    view["dt_s"] = np.array([float(gaps[0])])
    info = {"dump_cols": cols, "block_rows": rows, "instants_s": inst,
            "dt_s": float(gaps[0]), "k": len(inst),
            "dropped_dump_instants_s": [float(w) for c, w in enumerate(want)
                                        if c not in cols],
            "block_grid_s": np.round(ts, 3).tolist(),
            "rebuild_for_full_horizon": rebuild,
            "_is": ("distance-keeping is scored on the instants the dump and the "
                    "banked block SHARE, by index-select on both sides — no "
                    "resampling of the lead track, no interpolation of the path")}
    return view, idx, meta, info


def _distance_keeping_common(rec, files, manifest, lead_path, arms, P_cat, G_all,
                             pairs, eid_w, n_boot, seed, tiers, dt, k) -> dict:
    """``rec['refcv3']['distance_keeping']`` + the per-arm LONGITUDINAL patch.

    Same instruments as refav1's block (``lead_metrics.distance_keeping``, the
    GT reference, the ``ol`` kinematic-contract check, ``by_speed``, the paired
    deltas) on the common-grid VIEW; the per-arm result is also written into
    ``rec['arms'][arm]['four_families']['longitudinal']['distance_keeping']``
    (replacing the UNAVAILABLE block) with its grid and ``_declared_by``, and
    the family's interval coverage is recomputed.
    """
    from taniteval import four_families as ff
    from taniteval import lead_metrics as lm
    view, idx, meta, info = lead_block_common_grid(lead_path, dt, k)
    base = {"block": lead_path, "block_sha256": ra._sha256(lead_path),
            "block_version": meta.get("version"), "block_tool": meta.get("tool"),
            "block_built_utc": meta.get("built_utc"),
            "block_rows_all": meta.get("n_rows"), "block_clips": meta.get("n_clips"),
            "conventions": meta.get("conventions"), "states": meta.get("states")}
    if view is None:
        out = dict(base)
        out.update(info)
        return out
    lead = ra.join_lead_block(files, manifest, view, idx, k=info["k"], dt=info["dt_s"])
    cov = lead.pop("coverage")
    out = dict(base)
    out.update({"grid": info, "coverage": cov})
    n_lab = cov["counts"]["LEAD"] + cov["counts"]["NO_LEAD"]
    if n_lab == 0:
        out.update(_refused(
            f"the lead block covers 0 labelled windows of {cov['n_windows']} "
            f"(NO_LABEL {cov['counts']['NO_LABEL']}, NOT_STRAIGHT "
            f"{cov['counts']['NOT_STRAIGHT']}, no-row {cov['n_windows_no_row']}; "
            f"episodes OK {cov['n_episodes_ok']}/{cov['n_episodes']}) — nothing is "
            f"scoreable; NOT read as free flow", "n/a", 0))
        return out
    cols = info["dump_cols"]
    dtv = info["dt_s"]
    out["status"] = "PRESENT"
    out["n"] = int(cov["counts"]["LEAD"])
    out["tier"] = {x: tiers.get(x) for x in arms}
    out["estimator"] = ("per arm: lead_metrics.distance_keeping on the common-grid "
                        "view + episode-cluster bootstrap; by_speed: "
                        "lead_metrics.distance_keeping_by_speed; pairs: "
                        "lead_metrics.paired_distance_keeping")
    out["_binding"] = ("LONGITUDINAL distance-keeping: headway / time-gap / min-TTC "
                       "per arm with n and CI, per speed band, never pooled with "
                       "speed accuracy; a censored TTC carries n_closing; scored on "
                       f"the SHARED instants {info['instants_s']} s only")
    pw, per_arm = {}, {}
    for arm in arms:
        dk = lm.distance_keeping(P_cat[arm][:, cols], lead["leads"], lead["lead_lens"],
                                 lead["speeds"], dtv)
        pw[arm] = {kk: np.asarray(dk[kk], dtype=np.float64) for kk in ra._DK_KEYS}
        blk = {"tier": tiers.get(arm), "status": dk.get("status"),
               "reason": dk.get("reason"), "n": dk.get("n"), "n_windows": dk.get("n_windows"),
               "dt_s": dtv, "instants_s": info["instants_s"],
               "mean_headway_min_m": dk.get("mean_headway_min_m"),
               "mean_time_gap_min_s": dk.get("mean_time_gap_min_s"),
               "n_time_gap": dk.get("n_time_gap"),
               "mean_min_ttc_s": dk.get("mean_min_ttc_s"),
               "n_closing": dk.get("n_closing"),
               "censoring_note": dk.get("censoring_note"),
               "gap_convention": dk.get("gap_convention"), "ttc_cap_s": dk.get("ttc_cap_s"),
               "ci": {kk: ra._boot(pw[arm][kk], eid_w, n_boot, seed) for kk in ra._DK_KEYS}}
        if dk.get("status") == "OK":
            blk["by_speed"] = lm.distance_keeping_by_speed(
                dk, lead["speeds"], lead["eid"], states=lead["state"],
                n_boot=n_boot, seed=seed)
        blk["_grid_note"] = (f"min-over-steps is a min over {info['k']} instant(s) "
                             f"{info['instants_s']} s — coarser than the block's own "
                             f"{len(info['block_grid_s'])}-step grid; a lead that is "
                             f"closest between them is not seen. Rebuild: "
                             f"{info['rebuild_for_full_horizon']}")
        per_arm[arm] = blk
        # -- patch the canonical family block -------------------------------
        fam = rec["arms"][arm]["four_families"]
        lon = fam["longitudinal"]
        lon["distance_keeping"] = {
            **{kk: v for kk, v in blk.items() if kk not in ("tier",)},
            "status": dk.get("status"),
            "admitted_by": ("D-LEAD-1 discrimination control, 2026-08-03 — GT vs "
                            "hold-v0 CV, paired episode-cluster bootstrap"),
            "_declared_by": ("taniteval/tools/refcv3_arm.py — computed on the "
                             "common-grid view of the banked B1 EVAL lead block "
                             "(refav1_arm.join_lead_block guards: exact grid, "
                             "label-free speed proof, NO_LABEL never free flow)"),
            "_time_join": info["_is"]}
        ci = lon.setdefault("ci", {})
        comps = ci.setdefault("components", {})
        unav = ci.setdefault("unavailable", {})
        for kk, name in (("headway_min_m", "mean_headway_min_m"),
                         ("time_gap_min_s", "mean_time_gap_min_s"),
                         ("min_ttc_s", "mean_min_ttc_s")):
            key = f"distance_keeping.{name}"
            b = blk["ci"][kk]
            if b.get("status") == "NOT-APPLICABLE":
                unav[key] = {"status": "UNAVAILABLE", "reason": b.get("reason"), "n": 0}
            else:
                comps[key] = b
                unav.pop(key, None)
        fam["_ci_coverage"]["longitudinal"] = ff.ci_coverage(lon, "longitudinal")
        fam["_intervals_complete"] = bool(
            fam["_ci_coverage"]["longitudinal"].get("complete")
            and fam["_ci_coverage"]["lateral"].get("complete"))
    out["per_arm"] = per_arm
    # GT reference on the SAME view + the kinematic-contract check (ol vs GT)
    gt = lm.distance_keeping(G_all[:, cols], lead["leads"], lead["lead_lens"],
                             lead["speeds"], dtv)
    gtpw = {kk: np.asarray(gt[kk], dtype=np.float64) for kk in ra._DK_KEYS}
    out["gt_reference"] = {
        "_is": ("lead_metrics.distance_keeping of the GT ego path on the same "
                "instants: the value a perfect kinematic reproduction scores"),
        **{kk: ra._boot(gtpw[kk], eid_w, n_boot, seed) for kk in ra._DK_KEYS}}
    if "ol" in pw:
        a_, b_ = pw["ol"]["headway_min_m"], gtpw["headway_min_m"]
        both = np.isfinite(a_) & np.isfinite(b_)
        dd = np.abs(a_[both] - b_[both]) if both.any() else np.zeros(0)
        out["kinematic_contract_check"] = {
            "n_both": int(both.sum()),
            "n_ol_only": int((np.isfinite(a_) & ~np.isfinite(b_)).sum()),
            "n_gt_only": int((np.isfinite(b_) & ~np.isfinite(a_)).sum()),
            "mean_abs_headway_diff_m": round(float(dd.mean()), 4) if dd.size else None,
            "median_abs_headway_diff_m": (round(float(np.median(dd)), 4) if dd.size else None),
            "max_abs_headway_diff_m": round(float(dd.max()), 4) if dd.size else None,
            "_reading": ("ol integrates the RECORDED (a, kappa) at 0.1 s from v0, so "
                         "its headway must sit within its own ADE of the GT "
                         "reference; a LARGE mean is a join or frame error")}
    out["paired"] = {nm: lm.paired_distance_keeping(pw[b], pw[a_], eid_w, names=(b, a_),
                                                    n_boot=n_boot, seed=seed)
                     for a_, b, nm in pairs}
    return out


# --------------------------------------------------------------------------- #
# the analysis (CPU)                                                           #
# --------------------------------------------------------------------------- #
_COND_OF_ARM = {"cl": "nav_true", "cl_navshuf": "nav_shuffled", "cl_nonav": "nav_zero",
                "cl_rh": "nav_true"}


def _agree(lbl, pred, names, eid, n_boot, seed, ignore=-100, tier="T1", **extra):
    """``four_families._agreement_block`` on the labelled rows, or a refusal."""
    from taniteval import four_families as ff
    lbl = np.asarray(lbl, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if (pred < 0).all():
        return _refused("no such head on this checkpoint (prediction = -1)", tier, 0)
    m = (lbl != ignore) & (pred >= 0)
    if m.sum() == 0:
        return _refused("no window carries this label (every window out of band "
                        "or unlabelled)", tier, 0)
    e_m = [e for e, k in zip(eid, m) if k]
    blk = ff._agreement_block(lbl[m], pred[m], list(names), e_m, n_boot, seed, tier=tier)
    blk["n_excluded_no_label"] = int((~m).sum())
    blk["estimator"] = "full-set point; episode_cluster_bootstrap intervals"
    blk.update(extra)
    return blk


def _paired_acc(lbl, pa, pb, eid, n_boot, seed, ignore=-100):
    from taniteval import ci as _ci
    lbl = np.asarray(lbl, dtype=int)
    pa, pb = np.asarray(pa, dtype=int), np.asarray(pb, dtype=int)
    m = (lbl != ignore) & (pa >= 0) & (pb >= 0)
    if m.sum() == 0:
        return _refused("no labelled window with both predictions", "T1", 0)
    e_m = [e for e, k in zip(eid, m) if k]
    return _ci.paired_episode_cluster_bootstrap((pa[m] == lbl[m]).astype(float),
                                                (pb[m] == lbl[m]).astype(float),
                                                e_m, n_boot=n_boot, seed=seed)


def analyze_refcv3(dump_dir: str, *, n_boot: int = 2000, seed: int = 0,
                   tiers: dict | None = None, lead_block: str | None = None) -> dict:
    """``t1_eval.analyze`` on the trajectory dump + the refcv3 sidecar analysis."""
    from taniteval import ci as _ci
    from tanitad.data import v7_labels as v7l
    from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
    from tanitad.refs import refc_tactical as tac
    from tanitad.refs.refb import ROUTE_CLASSES
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not files:
        raise ValueError(f"no ep*.npz under {dump_dir}")
    man_path = os.path.join(dump_dir, "manifest.json")
    manifest = None
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    if not manifest or "grid" not in manifest:
        raise ValueError(f"{dump_dir} carries no manifest.json with a grid — the "
                         f"dump's dt cannot be known; refusing to assume")
    dt = float(manifest["grid"]["dt_s"])
    k = int(manifest["grid"]["k"])
    tiers = dict(ARM_TIERS, **(tiers or {}))
    with np.load(files[0]) as d0:
        arms = [x for x in d0.files if x not in ("g", "ws", *t1._META_KEYS)
                and not x.endswith(t1._FAN_SUFFIXES)]
        if int(d0["g"].shape[1]) != k:
            raise ValueError(f"dump K {d0['g'].shape[1]} != manifest K {k}")
    pairs = [(x, y, nm) for x, y, nm in (
        ("ol", "cl", "paired_closed_minus_open"),
        ("ha", "cl", "paired_cl_minus_ha"),
        ("cl_navshuf", "cl", "paired_cl_minus_navshuf"),
        ("cl_nonav", "cl", "paired_cl_minus_nonav"),
        ("cl_rh", "cl", "paired_cl_minus_rh"))
        if x in arms and y in arms]
    rec = t1.analyze(files, tiers={x: tiers[x] for x in arms if x in tiers},
                     n_boot=n_boot, seed=seed, dt=dt, paired=pairs, lead=None)
    rec["tool"] = ("taniteval/tools/refcv3_arm.py (trajectory families via "
                   "taniteval/tools/t1_eval.py::analyze, IMPORTED; lead join and "
                   "paired families via refav1_arm.py, IMPORTED)")

    # ---- per-window components + paired families -----------------------------
    G_all, P_all, eid_w = [], {x: [] for x in arms}, []
    for f in files:
        with np.load(f) as d:
            G = d["g"][..., :2].astype(np.float64)
            G_all.append(G)
            eid_w += [os.path.splitext(os.path.basename(f))[0]] * G.shape[0]
            for x in arms:
                P_all[x].append(d[x][..., :2].astype(np.float64))
    G_all = np.concatenate(G_all)
    N = int(G_all.shape[0])
    P_cat = {x: np.concatenate(P_all[x]) for x in arms}
    comps = {x: ra._components(P_cat[x], G_all, dt) for x in arms}
    ref = {"n_windows": N, "n_episodes": len(files),
           "tiers": {x: tiers[x] for x in arms},
           "arm_meaning": {x: ARM_MEANING.get(x) for x in arms},
           "t1_definition": manifest.get("t1_definition", T1_DEFINITION),
           "_tier_doctrine": rec["_tier_doctrine"],
           "grid": manifest["grid"],
           "families_paired": {nm: ra._paired_families(comps, x, y, eid_w, tiers,
                                                       n_boot, seed)
                               for x, y, nm in pairs},
           "families_note": (
               "LONGITUDINAL / LATERAL / ADE and the TRAJECTORY-DERIVED tactical rows "
               "per arm live in rec['arms'][arm]['four_families'] (t1_eval, "
               "unchanged). refcv3 HAS decision heads, so STRATEGIC (route head) and "
               "the DECLARED tactical decisions are filled into those blocks per "
               "conditioning-bearing arm AND reported below (rec['refcv3'])."),
           }
    # ---- LONGITUDINAL distance-keeping on the common grid ---------------------
    if lead_block:
        ref["distance_keeping"] = _distance_keeping_common(
            rec, files, manifest, lead_block, arms, P_cat, G_all, pairs, eid_w,
            n_boot, seed, tiers, dt, k)
        dkb = ref["distance_keeping"]
        _p(f"[lead] {os.path.basename(lead_block)}: status={dkb.get('status')} "
           f"grid={(dkb.get('grid') or {}).get('instants_s')} "
           f"coverage={(dkb.get('coverage') or {}).get('counts')}")
    else:
        ref["distance_keeping"] = _refused(
            f"no lead block passed (--lead-block); the banked B1 EVAL block is "
            f"{ra.LEAD_BLOCK_DEFAULT}", "n/a", 0)

    dec, eid_d = ra._load_decisions(dump_dir)
    if dec is None:
        ref["sidecar"] = _refused("no decisions/ep*.npz sidecar — this dump was "
                                  "not written by refcv3_arm.run_dump", "n/a")
        rec["refcv3"] = ref
        return rec
    if len(eid_d) != N:
        raise ValueError(f"decisions sidecar has {len(eid_d)} rows for {N} windows")
    hier = bool(((manifest.get("model") or {}).get("arm")) == "hier")
    conds = ("nav_true", "nav_shuffled", "nav_zero")
    nav_valid = dec["nav_valid"].astype(bool)
    n2r = nav_index_to_route_index()

    # ---- STRATEGIC: route head vs TWO labels under three conditionings ---------
    strat = {"tier": "T1 (the route head reads the OBSERVED window only)",
             "n_windows": N, "n_nav_valid": int(nav_valid.sum()),
             "nav_valid_frac": round(float(nav_valid.mean()), 4) if N else None,
             "nav_shuffle": (manifest.get("nav") or {}).get("shuffle"),
             "_echo_caveat": ("route_label (v7.2 nav-derived) and the fed nav_cmd come "
                              "from the SAME field: under nav_true route accuracy is "
                              "the nav ECHO (flagship-v1 scored 1.0000 on a bijection of "
                              "its own input). Evidence of route skill is the "
                              "nav_shuffled / nav_zero conditioning and the CHANGED "
                              "subset; route_label_v21 is the trainer's own "
                              "future-derived aux target (independent thresholds, same "
                              "future path)."),
             "labels": {}}
    for lbl_key, lbl_name, note in (
            ("route_label", "v72_nav_derived", "refav1_loader._NAV_TOKEN_TO_ROUTE of "
             "the record's nav token; scored on route-labelled AND nav-valid windows"),
            ("route_label_v21", "route_from_future_v21", "refb_labels.route_from_future_"
             "v21 at t0 (route_valid rows only) — the trainer's route aux target")):
        lbl = dec[lbl_key].astype(int)
        use = lbl != v7l.IGNORE_ID
        if lbl_key == "route_label":
            use = use & nav_valid
        blk = {"label_source": note, "n_scored": int(use.sum()),
               "n_excluded": int((~use).sum()), "conditionings": {}}
        for cname, navc in (("nav_true", dec["nav_cmd"]), ("nav_shuffled", dec["nav_cmd_shuf"]),
                            ("nav_zero", np.zeros(N, dtype=int))):
            pred = dec[f"route_pred_{cname}"].astype(int)
            lm_ = np.where(use, lbl, v7l.IGNORE_ID)
            b = _agree(lm_, pred, ROUTE_CLASSES, eid_d, n_boot, seed)
            if b.get("status") == "OK":
                implied = np.array([n2r.get(int(x), -1) for x in navc])
                b["nav_implied_route_agreement"] = round(
                    float((pred[use] == implied[use]).mean()), 4)
                maj = int(np.bincount(lbl[use], minlength=len(ROUTE_CLASSES)).argmax())
                b["majority_class"] = ROUTE_CLASSES[maj]
                b["majority_class_rate"] = round(float((lbl[use] == maj).mean()), 4)
            blk["conditionings"][cname] = b
        if use.sum():
            blk["paired_true_minus_shuffled_accuracy"] = _paired_acc(
                np.where(use, lbl, v7l.IGNORE_ID), dec["route_pred_nav_true"],
                dec["route_pred_nav_shuffled"], eid_d, n_boot, seed)
            changed = use & (dec["nav_cmd_shuf"] != dec["nav_cmd"])
            blk["n_changed_subset"] = int(changed.sum())
            if changed.sum():
                e_c = [e for e, kk in zip(eid_d, changed) if kk]
                ps_ = dec["route_pred_nav_shuffled"].astype(int)
                implied_s = np.array([n2r.get(int(x), -1) for x in dec["nav_cmd_shuf"]])
                blk["changed_subset"] = {
                    "n": int(changed.sum()), "tier": "T1",
                    "estimator": "episode_cluster_bootstrap",
                    "route_follows_LABEL_under_shuffle": _ci.episode_cluster_bootstrap(
                        (ps_ == lbl).astype(float)[changed], e_c, n_boot=n_boot, seed=seed),
                    "route_follows_SHUFFLED_NAV_under_shuffle": _ci.episode_cluster_bootstrap(
                        (ps_ == implied_s).astype(float)[changed], e_c, n_boot=n_boot, seed=seed),
                    "_reading": ("an ECHO reads follows_nav ~ 1 / follows_label ~ 0; "
                                 "route skill from vision reads follows_label high "
                                 "regardless of the token")}
            else:
                blk["changed_subset"] = _refused(
                    "the permutation changed no nav token — the shuffle control has "
                    "no power on this set", "T1")
        strat["labels"][lbl_name] = blk
    ref["strategic"] = strat

    # ---- strategic GOAL (g_str bearing vs the LAN label) ------------------------
    if hier and "gstr_label_valid" in dec:
        ok = dec["gstr_label_valid"].astype(bool)
        gs = {"tier": "T1", "label_source": "RefCModel.goal_targets(lan_window_features) "
                                             "— the LAN corridor label (train-only input, "
                                             "E12; here a LABEL)",
              "n_valid": int(ok.sum()), "n_windows": N, "conditionings": {}}
        for cname in conds:
            g = dec[f"g_str_{cname}"].astype(np.float64)
            cos = (g[:, :2] * dec["gstr_label_bearing"].astype(np.float64)).sum(-1)
            e_ok = [e for e, kk in zip(eid_d, ok) if kk]
            if ok.sum():
                gs["conditionings"][cname] = {
                    "bearing_cosine": _ci.episode_cluster_bootstrap(cos[ok], e_ok,
                                                                    n_boot=n_boot, seed=seed),
                    "dist_pref_mae": _ci.episode_cluster_bootstrap(
                        np.abs(g[:, 2] - dec["gstr_label_dist"].astype(np.float64))[ok],
                        e_ok, n_boot=n_boot, seed=seed),
                    "n": int(ok.sum())}
            else:
                gs["conditionings"][cname] = _refused("no window has a valid LAN label", "T1", 0)
        ref["strategic_goal"] = gs
    else:
        ref["strategic_goal"] = _refused(
            "the run did not train the strategic goal head (--goal-str absent) or the "
            "arm is flat — g_str is untrained/absent; a bearing agreement would score "
            "an untrained head", "T1", N)

    # ---- TACTICAL, declared: hierarchy heads (v7.2) + core heads (kin3) -------
    vv = (manifest.get("model") or {}).get("tac_vocab_version", "v7.0")
    lat_names, lon_names = list(tactical_lat_actions(vv)), list(tactical_lon_actions_v(vv))
    lt = manifest.get("label_timing") or {}
    tacd = {"tier": "T1 (decision heads read the OBSERVED window only)",
            "vocabulary": vv, "n_windows": N,
            "label_timing": {**lt, "primary_used_here": "trainer"},
            "_is": ("the DECLARED decision (argmax of the HIERARCHY's z_tac lat/lon heads) "
                    "vs the v7.2 a_tac label on IN-BAND windows (-100 excluded); the "
                    "EXECUTED manoeuvre read off the driven path is the trajectory-derived "
                    "block in rec['arms'] (refc 3-way classes — a different instrument)"),
            "conditionings": {}, "conditionings_raw_timing": {}}
    for hk, names in (("lat", lat_names), ("lon", lon_names)):
        for cname in conds:
            pred = dec[f"{hk}_tac_pred_{cname}"]
            tacd["conditionings"][f"{hk}_{cname}"] = _agree(
                dec[f"{hk}_label"], pred, names, eid_d, n_boot, seed)
            tacd["conditionings_raw_timing"][f"{hk}_{cname}"] = _agree(
                dec[f"{hk}_label_raw"], pred, names, eid_d, n_boot, seed)
        tacd[f"{hk}_paired_true_minus_shuffled_accuracy"] = _paired_acc(
            dec[f"{hk}_label"], dec[f"{hk}_tac_pred_nav_true"],
            dec[f"{hk}_tac_pred_nav_shuffled"], eid_d, n_boot, seed)
    ref["tactical_declared"] = tacd
    tack = {"tier": "T1", "vocabulary": "kin3 (refc_tactical.LAT_CLASSES x LON_CLASSES)",
            "label_source": "refc_tactical.window_factored_labels over the GT 2 s future "
                            "(the core heads' own supervision; a LABEL uses the future)",
            "n_windows": N, "conditionings": {}}
    for hk, names in (("lat", tac.LAT_CLASSES), ("lon", tac.LON_CLASSES)):
        for cname in conds:
            tack["conditionings"][f"{hk}_{cname}"] = _agree(
                dec[f"{hk}_kin3"], dec[f"{hk}_core_pred_{cname}"], names, eid_d,
                n_boot, seed)
    ref["tactical_core_kin3"] = tack

    # ---- TACTICAL goal-setting: g_tac vs hindsight goals; selection telemetry --
    if hier and "goal_tac_label" in dec:
        gl = dec["goal_tac_label"].astype(np.float64)                 # [N, n_tau, 4]
        gv = dec["goal_tac_valid"].astype(bool)
        taus = list((manifest.get("model") or {}).get("cfg", {}).get(
            "goal_tau_steps", [20, 40, 60]))
        adm = float((manifest.get("model") or {}).get("admission_sigma_m", float("nan")))
        tg = {"tier": "T1", "taus_steps": taus, "n_windows": N,
              "label_source": "refb_labels.goal_tac_targets (hindsight, uses the future)",
              "admission_sigma_m": adm,
              "_admission_rule": ("PREREG_REFC_V3 §5 / refc_v3.RefCV3Config.admission_"
                                  "sigma_m: goal-selection deltas may be READ as "
                                  "improvements only when the 2 s endpoint error's "
                                  "1-sigma (RMS) is <= admission_sigma_m"),
              "conditionings": {}}
        for cname in conds:
            gp = dec[f"g_tac_{cname}"].astype(np.float64)
            per_tau = {}
            for ti, tau in enumerate(taus):
                ok = gv[:, ti]
                if not ok.any():
                    per_tau[f"tau_{tau}"] = _refused("no valid goal at this tau", "T1", 0)
                    continue
                err = np.linalg.norm(gp[:, ti, :2] - gl[:, ti, :2], axis=-1)
                e_ok = [e for e, kk in zip(eid_d, ok) if kk]
                per_tau[f"tau_{tau}"] = {
                    "n": int(ok.sum()),
                    "endpoint_err_mean_m": _ci.episode_cluster_bootstrap(
                        err[ok], e_ok, n_boot=n_boot, seed=seed),
                    "endpoint_err_rms_m": _ci.episode_cluster_bootstrap(
                        err[ok] ** 2, e_ok, reduce="rms", n_boot=n_boot, seed=seed),
                    "speed_err_mae_mps": _ci.episode_cluster_bootstrap(
                        np.abs(gp[:, ti, 3] - gl[:, ti, 3])[ok], e_ok, n_boot=n_boot, seed=seed),
                    "heading_err_mae_rad": _ci.episode_cluster_bootstrap(
                        np.abs((gp[:, ti, 2] - gl[:, ti, 2] + np.pi) % (2 * np.pi) - np.pi)[ok],
                        e_ok, n_boot=n_boot, seed=seed)}
                if int(tau) == 20:
                    rms = per_tau[f"tau_{tau}"]["endpoint_err_rms_m"]["mean"]
                    per_tau[f"tau_{tau}"]["admission_read"] = {
                        "rms_2s_m": rms, "sigma_gate_m": adm,
                        "admitted": bool(math.isfinite(adm) and rms <= adm),
                        "_note": "a gate on READING selection deltas, not a score"}
            tg["conditionings"][cname] = per_tau
        ref["tactical_goal"] = tg
        sel_true = dec["sel_idx_nav_true"].astype(int)
        base_true = dec["sel_idx_base_nav_true"].astype(int)
        gd = dec["goal_dist_sel_nav_true"].astype(np.float64)
        ref["selection"] = {
            "tier": "T1", "n_windows": N,
            "goal_gate_value": round(float(np.nanmean(dec["goal_gate_value"])), 6),
            "frac_windows_goal_changed_selection": round(float((sel_true != base_true).mean()), 4),
            "goal_dist_of_selected_m": _ci.episode_cluster_bootstrap(
                gd[np.isfinite(gd)], [e for e, kk in zip(eid_d, np.isfinite(gd)) if kk],
                n_boot=n_boot, seed=seed) if np.isfinite(gd).any() else None,
            "seam_goal_sel_ratio_preclamp_mean": round(
                float(np.nanmean(dec["seam_goal_sel_ratio_preclamp_mean"])), 6),
            "seam_goal_sel_bound_frac_mean": round(
                float(np.nanmean(dec["seam_goal_sel_bound_frac"])), 6),
            "reach_clipped_frac_mean": round(float(np.nanmean(dec["reach_clipped_frac"])), 4),
            "_reading": ("frac_windows_goal_changed_selection = 0 with goal_gate_value "
                         "= 0 means E9 contributed nothing (CAVEAT-B); a seam bound_frac "
                         "near 1 means the graft sits at the clamp and its strength is "
                         "unreadable (refc_select.apply_seam_clamp)")}
    else:
        ref["tactical_goal"] = _refused("flat arm — no tactical goal head", "T1", N)
        ref["selection"] = _refused("flat arm — no goal selection graft", "T1", N)

    # ---- T0 WM diagnostic: the LAW aux -----------------------------------------
    mm = dec["law_mse_model"].astype(np.float64)
    mp = dec["law_mse_persist"].astype(np.float64)
    mz = dec["law_mse_zero"].astype(np.float64)
    ref["wm_diagnostic_T0"] = {
        "tier": "T0", "tier_note": _TIER_NOTE["T0"], "n": N,
        "space": ("pooled encoder latent (un-standardised): law_pred = law_head(pooled, "
                  "traj) vs encode_pooled(frame t0 + LAW_AHEAD*0.1 s) — the trainer's "
                  "loss_law (refc_v3_train.py:579-582)"),
        "_controls": {"law_mse_persist": "MSE of PERSISTING the last observed pooled "
                                         "latent — the raw-input floor",
                      "law_mse_zero": "MSE of the zero prediction = target energy "
                                      "(the constant-only control; NOT pinned to 1 in "
                                      "this un-standardised space, reported with "
                                      "law_tgt_energy)"},
        "law_tgt_energy_mean": round(float(np.mean(dec["law_tgt_energy"])), 6),
        "intervals": _ci.bootstrap_metrics(
            {"law_mse_model": (mm, "mean", 6), "law_mse_persist": (mp, "mean", 6),
             "law_mse_zero": (mz, "mean", 6)}, eid_d, n_boot=n_boot, seed=seed),
        "paired_persist_minus_model": _ci.paired_episode_cluster_bootstrap(
            mp, mm, eid_d, n_boot=n_boot, seed=seed),
        "paired_zero_minus_model": _ci.paired_episode_cluster_bootstrap(
            mz, mm, eid_d, n_boot=n_boot, seed=seed),
        "estimator": "episode_cluster_bootstrap; paired for the controls"}

    # ---- cl_rh: the own-rollout conditioning proof -------------------------------
    if "rh_v0_fed" in dec:
        vf = dec["rh_v0_fed"].astype(np.float64)
        vg = dec["rh_gt_v"].astype(np.float64)
        ref["receding_arm"] = {
            "tier": tiers.get("cl_rh"), "n_windows": N, "k_ticks": int(vf.shape[1]),
            "v0_fed_tick0_equals_measured_v0": bool(np.allclose(vf[:, 0], dec["v0_fed"], atol=1e-6)),
            "mean_abs_v0_fed_minus_gt_v_ticks_ge1": round(
                float(np.abs(vf[:, 1:] - vg[:, 1:]).mean()), 4) if vf.shape[1] > 1 else None,
            "_reading": ("ticks >= 1 are conditioned on the arm's OWN chord speed, not "
                         "the GT speed at those frames (a nonzero mean |difference| is the "
                         "proof); the frames themselves are RECORDED — hence the default "
                         "T0 stamp")}

    # ---- canonical per-arm family patches + protocol -----------------------------
    v72 = strat["labels"]["v72_nav_derived"]["conditionings"]
    v21 = strat["labels"]["route_from_future_v21"]["conditionings"]
    for arm in rec.get("arms", {}):
        fam = rec["arms"][arm].get("four_families")
        if not isinstance(fam, dict):
            continue
        cond = _COND_OF_ARM.get(arm)
        traj_only = fam.get("strategic")
        if cond is None:
            fam["strategic"] = {**_refused(
                f"{arm} is a kinematic CONTROL arm — no decision head drives it; the "
                f"route head's decisions live on the conditioning-bearing arms", tiers[arm], N),
                "_trajectory_only_block": traj_only}
        else:
            fam["strategic"] = {
                "status": v72[cond].get("status", "OK"), "tier": tiers[arm],
                "n": v72[cond].get("n", 0),
                "estimator": "full-set point; episode_cluster_bootstrap intervals",
                "conditioning": cond,
                "route_head_vs_v72_nav_label": v72[cond],
                "route_head_vs_route_from_future_v21": v21[cond],
                "_echo_caveat": strat["_echo_caveat"],
                "_declared_by": "taniteval/tools/refcv3_arm.py analyze_refcv3",
                "_trajectory_only_block": traj_only}
            t_blk = fam.get("tactical")
            if isinstance(t_blk, dict):
                t_blk["declared"] = {
                    "hierarchy_v72": {hk: tacd["conditionings"].get(f"{hk}_{cond}")
                                      for hk in ("lat", "lon")},
                    "core_kin3": {hk: tack["conditionings"].get(f"{hk}_{cond}")
                                  for hk in ("lat", "lon")},
                    "conditioning": cond,
                    "_is": "the SELECTED manoeuvre (declared heads) beside the EXECUTED one "
                           "(this block's trajectory-derived rows)"}
                tgc = ref.get("tactical_goal") or {}
                t_blk["goal"] = ((tgc.get("conditionings") or {}).get(cond)
                                 if tgc.get("status") is None else tgc)
        fam["_families_unavailable"] = [
            kk for kk in ("longitudinal", "lateral", "tactical", "strategic")
            if isinstance(fam.get(kk), dict) and fam[kk].get("status") in ("UNAVAILABLE",)]
    nav_blk = manifest.get("nav") or {}
    ref["protocol"] = {
        "inference_inputs": ("the OBSERVED W-frame stacked-RGB window (vision); the "
                             "MEASURED v0 = poses[t0, 3] (PI ruling 2026-09-02) into the "
                             "measurement encoder + the S2 reachability band; the v7.2 nav "
                             "token (route command, PI 2026-08-03) into the core "
                             "measurement encoder and the E13 tactical/strategic states; "
                             "lan=None (E12). NOTHING recorded after t0 on cl / cl_navshuf / "
                             "cl_nonav / ha"),
        "vision_only": "vision + v0(t0) + nav token — the two admitted additions; no "
                       "ego state beyond v0, no future",
        "goal_source": ("predicted: g_tac (E8, from z_tac = PhiTac(pooled_seq) + nav) and "
                        "g_str (E3, from StrategicCtx + nav) — frames + nav only (E11 "
                        "pinned by tests/test_refc_v3.py); the goal-point selection (E9) "
                        "reads g_tac's 2 s slot" if hier else
                        "none — flat arm (no goal cascade)"),
        "goal_situation_disjoint": ("True by construction: RefCV3Model.provenance_roles() "
                                    "declares no situation-classifier output in the graph; "
                                    "nav is the v7.2 nav_command token (allow_oracle_nav-"
                                    "stamped), not a classifier output"),
        "corpus": (manifest.get("corpus") or {}).get("episodes"),
        "labels": (manifest.get("corpus") or {}).get("labels"),
        "nav_source": nav_blk.get("source"),
        "parity_key": ("the eval split is the v7.2 EVAL clip set (labels md5 in "
                       "corpus.join_report) — NOT the canonical train parity key "
                       "e438721ae894; cross-arm deltas are valid only against dumps on "
                       "this SAME corpus + grid"),
    }
    for arm in rec.get("arms", {}):
        fam = rec["arms"][arm].get("four_families")
        if not isinstance(fam, dict) or "_protocol" not in fam:
            continue
        fam["_protocol"].update({kk: v for kk, v in ref["protocol"].items()
                                 if kk in fam["_protocol"]})
        fam["_protocol"]["_declared_by"] = ("taniteval/tools/refcv3_arm.py analyze_refcv3 "
                                            "(post hoc, from the roll's manifest + the "
                                            "model's own input contract)")
        fam["_protocol_undeclared"] = sorted(
            kk for kk, v in fam["_protocol"].items()
            if not kk.startswith("_") and isinstance(v, str) and v.startswith("UNDECLARED"))
    ref["manifest"] = manifest
    rec["refcv3"] = ref
    return rec


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="refcv3 T0/T1 eval adapter (writes a t1_eval-compatible dump + "
                    "a decisions sidecar; analyses both).")
    ap.add_argument("--ckpt", help="refc_v3_train.py ckpt.pt / ckpt_<step>.pt")
    ap.add_argument("--config", default=None,
                    help="the run's config.json (default: sibling of --ckpt)")
    ap.add_argument("--episodes", help="v2ep EVAL episode dir (<clip>.v2ep.pt)")
    ap.add_argument("--labels", default=None, help="s2_labels_v7.2_eval.jsonl.gz")
    ap.add_argument("--nav", default=None, help="must equal --labels (one blob)")
    ap.add_argument("--nav-source", choices=NAV_SOURCES, default="auto",
                    help="auto = as trained (config.json nav_from_v7)")
    ap.add_argument("--arm", default="refcv3", help="label for the output record")
    ap.add_argument("--out", help="JSON output FILE")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--analyze-only", default=None, metavar="DUMP_DIR")
    ap.add_argument("--dump-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="2s")
    ap.add_argument("--episodes-n", type=int, default=0,
                    help="first N episodes of the sorted dir (0 = all)")
    ap.add_argument("--window-stride", type=int, default=1,
                    help="stride in FRAMES over the trainer's window index")
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--nav-shuffle-seed", type=int, default=0)
    ap.add_argument("--no-navshuf", action="store_true",
                    help="skip the nav-shuffle T1 arm (⛔ then no nav-conditioned "
                         "claim is admissible)")
    ap.add_argument("--no-nonav-arm", action="store_true")
    ap.add_argument("--with-receding-arm", action="store_true",
                    help="ALSO roll cl_rh (T0-stamped by default; see docstring)")
    ap.add_argument("--allow-nonstrict", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", default="", help="extra/override tier stamps name=T0|T1")
    ap.add_argument("--lead-block", default=None,
                    help="per-frame B1 lead block; default = the banked block when "
                         f"it exists: {ra.LEAD_BLOCK_DEFAULT}")
    ap.add_argument("--no-lead-block", action="store_true")
    a = ap.parse_args(argv)
    lead_path = None
    if not a.no_lead_block:
        lead_path = a.lead_block or (ra.LEAD_BLOCK_DEFAULT
                                     if os.path.exists(ra.LEAD_BLOCK_DEFAULT) else None)
        if a.lead_block and not os.path.exists(a.lead_block):
            sys.exit(f"--lead-block {a.lead_block} does not exist")
        if lead_path is None:
            _p(f"[lead] no lead block (banked default absent: {ra.LEAD_BLOCK_DEFAULT}); "
               f"distance-keeping will be REFUSED with its reason")
    if not a.out:
        sys.exit("--out is required")
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")
    if a.analyze_only is None:
        for r, name in ((a.ckpt, "--ckpt"), (a.episodes, "--episodes"),
                        (a.labels, "--labels"), (a.dump_dir, "--dump-dir")):
            if not r:
                sys.exit(f"rollout mode needs {name} (or use --analyze-only)")
        run_dump(a)
        dump_dir = a.dump_dir
        if a.dump_only:
            _p(f"[dump-only] {dump_dir}; analyse later with --analyze-only")
            return
    else:
        dump_dir = a.analyze_only
    rec = analyze_refcv3(dump_dir, n_boot=a.n_boot, seed=a.seed,
                         tiers=t1._parse_tiers(a.tiers), lead_block=lead_path)
    rec.update({"arm": a.arm, "ckpt": a.ckpt, "dump_dir": dump_dir,
                "mode": "analyze-only" if a.analyze_only else "rollout+analyze",
                "lead_block": lead_path, "_unverified": _UNVERIFIED_ON_REAL_CKPT})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    _p(f"[out] {a.out}")
    r = rec.get("refcv3", {})
    dkb = r.get("distance_keeping") or {}
    for arm, blk in rec["arms"].items():
        ivl = blk["intervals"]["metrics"]
        ade = ivl.get("ade_dense_m", {})
        dk = (dkb.get("per_arm") or {}).get(arm) or {}
        st = blk["four_families"]["strategic"]
        _p(f"  {arm:11s} tier={blk['tier']}  ADE={ade.get('mean')} [{ade.get('lo')}, "
           f"{ade.get('hi')}]  unavailable={blk['four_families']['_families_unavailable']}"
           f"  strategic={st.get('status')} n={st.get('n')}  distance_keeping="
           f"{dk.get('status') or dkb.get('status')} n={dk.get('n')} "
           f"headway={dk.get('mean_headway_min_m')}")
    s = r.get("strategic", {})
    _p(f"  strategic nav_valid_frac={s.get('nav_valid_frac')}; tactical_declared "
       f"lat_nav_true={((r.get('tactical_declared') or {}).get('conditionings') or {}).get('lat_nav_true', {}).get('accuracy')}")


if __name__ == "__main__":
    main()
