#!/usr/bin/env python3
"""refav1_arm.py — the T0 / T1 eval adapter for REF-A v1 (``tanitad.refs.refa_v1``).

WHY THIS FILE EXISTS (MEASURED 2026-09-02): refav1 had NO eval path at any tier —
``t1_eval.py``, ``evaluate_checkpoint.py`` and ``driving_diagnostic.py`` each carry
0 references to it, and its trainer has no in-training eval. A 21,109-step
checkpoint lands in ~5 days with nothing able to read it. This adapter reads it.

⛔ TIER DOCTRINE (``Project Steering/EVAL_DOCTRINE.md``). Every arm here carries a
stamp, and the stamp travels into ``t1_eval.analyze`` unchanged:

    cl            T1  plan() at t0 with the TRUE nav token: the operative predictor
                      consumes the PLANNER'S OWN candidate actions inside iCEM;
                      perception fixed at t0; trajectory = unicycle(controls, v0).
    cl_navshuf    T1  the same, with nav_cmd PERMUTED across the eval windows
                      (register row D-REFAV1-NAV-DEPTH: an EVAL OBLIGATION).
    ha            T1  HOLD-ACTION control: the last OBSERVED (a, kappa) — the action
                      that CLOSES at t0 — held for K steps. Consumes no recorded
                      future (pinned by stack/tests/test_refav1_arm.py).
    ha0           T1  CONSTANT-VELOCITY control: a = 0, kappa = 0 at the measured v0 —
                      a straight line at constant speed. ⭐ THE STRONGEST TRIVIAL
                      BASELINE, and the one the echo test must actually be run
                      against (D-REFAV1-HA0-ARM, 2026-09-03). It consumes strictly
                      LESS than ``ha``: not even the last observed action.
    ol            T0  the RECORDED future (a, kappa) integrated from v0 — for refav1
                      this is the KINEMATIC-CONTRACT control (the corpus action
                      contract must reproduce the GT path), NOT a WM diagnostic.
                      The WM diagnostic proper for a feature-predicting model is
                      the teacher-forced FEATURE error, emitted in the sidecar
                      (``wm_mse_model`` vs the persist-last-field control).
    cl_nonav      T1  (opt-in) plan() with nav_cmd = None -> index 0 ("follow").
    cl_oraclegoal T0  (opt-in) plan() with the TRUE future field as goal_field —
                      a goal read from the future is future information, so this is
                      T0 by the doctrine; it exists to separate "the search is the
                      bottleneck" from "the goal is the bottleneck" (C101 family).

THE PI RULING (2026-09-02, verbatim): *"It is allowed to use the velocity as initial
measured state at its cycle time. It is not allowed to use the future dynamic
information from the ground truth."* ⇒ at T1 ``v0`` = the MEASURED speed at the
window's t0 (``poses[2t, 3]``), which is what ``plan(feats, v0=...)`` takes.
Thereafter ``v_k = v0 + sum_{j<k} a_j dt`` from the arm's OWN actions — the
programme's single unicycle integrator (``kinematic.rollout_unicycle`` via
``refa_v1_plan.unicycle_paths``) does exactly that, and NOTHING after frame 2t
reaches the T1 or hold-action arms.

⚠️ HISTORY — WHAT ``plan()`` DID WITH NO GOAL (fixed at e609a98, 2026-09-02; read
before quoting a T1 number from an OLDER dump). Its docstring promised the
"tactical brain's own imagined 6 s field" while the CODE did ``search_goal = None``
and the cost contained ONLY the comfort (jerk) and curvature terms: EVERY
zero-curvature constant-acceleration candidate scored EXACTLY 0 — cv, hold_v0 AND
decel_1.5 tied — and ``icem_plan``'s floor loop kept the LAST tie, i.e.
``baseline:decel_1.5``. ⛔ MEASURED on a random-init RefAV1 (this adapter's build):
51/51 windows returned ``baseline:decel_1.5`` — a constant −1.5 m/s² brake read as
"the T1 plan". (The first draft of this paragraph said "cv by construction"; the
test corrected it — the tie-break order was load-bearing.) SINCE e609a98 ``plan()``
imagines its goal (``goal_source == "tactical_imagined"``: vision + nav through
``_run_brains`` + measured v0, decoded by the factored heads into a (lat, lon)
token whose canonical controls seed the search), a tie resolves to ``hold_v0``,
and the goal provenance is stamped on the result — this adapter banks
``goal_source`` / ``goal_action`` per window beside ``plan_source`` and reports the
fractions, so a T1 number is never read without knowing what goal (if any) it was
planned against. Whether a floor baseline still wins is then a property of the
search / the init, reported with its source — never planning skill.

SPEED CHANNEL (``RefAV1Config.speed_channel``, landed 2026-09-02): the MODEL
derives ``v_k = v0 + sum_{j<k} a_j dt`` inside ``augment_actions`` (forward AND
plan) and REFUSES a pre-widened action tensor. This adapter therefore never
widens actions: it passes the loader's 2-wide (a, kappa) plus the measured
``v0`` to ``forward(..., v0=)`` and ``plan(v0=)``, so T0 and T1 run the model's
own convention and there is exactly ONE speed-channel rule in the programme.

DUMP SCHEMA — two files per episode, so ``t1_eval.py --analyze-only`` stays valid:

    <dump>/ep{fi:03d}.npz            the t1_eval contract (unchanged):
        g [N,K,2] cl ha ol [cl_navshuf cl_nonav cl_oraclegoal] [N,K,2]
        ws [N] window origin t  ·  eid [1]  ·  clip_index [1]  ·  v0 [N]
    <dump>/decisions/ep{fi:03d}.npz  the refav1 sidecar (this tool's analysis):
        lat_label lon_label route_label [N] (-100 = no label)
        nav_cmd nav_cmd_shuf [N]  nav_valid [N] bool
        {lat,lon,route}_pred_{nav_true,nav_shuffled,nav_zero} [N]
        wm_mse_model wm_mse_const [N,K_wm]  wm_tgt_std [N]  (T0, feature space)
        plan_{source,cost,agree,neval}_<arm> [N]  <arm>_controls [N,K,2]
        goal_{source,lat,lon}_<arm> [N]  (PlanResult.goal_source index into
            manifest.goal.source_names; the decoded goal tokens' vocab index, -1 = none)
    <dump>/manifest.json             episodes (with clip_id), grid, tiers, plan config,
                                     goal provenance rule, model provenance.

ESTIMATOR: full-set pooled point estimates; intervals = episode-cluster bootstrap
(``taniteval.ci``), paired across arms on the same windows. ``overlapping_holdout_se``
is never used (it biases the POINT ESTIMATE). Every block prints its n and tier.

DISTANCE-KEEPING (backlog R1, 2026-09-02). The LONGITUDINAL family's second half —
headway / time-gap / min-TTC — is PRESENT when a per-frame lead block is joined
(``--lead-block``, default = the banked B1 EVAL block ``LEAD_BLOCK_DEFAULT`` built by
``tools/build_lead_block_b1.py`` from ``obstacle.offline`` + egomotion + camera
timestamps for the 147 v7.2 EVAL clips). The join is by ``(clip_id, RAW frame 2t)``,
guarded by the horizon grid (``ts_rel_s == 0.2·(1..K)``) and the LABEL-FREE speed
proof (dump ``v0`` == block speed at the joined row). It runs at ANALYSIS time, so
``--analyze-only`` on an existing dump gains the family with zero GPU. Windows the
block does not cover (NO_LABEL: no obstacle.offline, horizon outside the labelled
span, no row, refused episode) are COUNTED and never scored as free flow; the
record carries coverage, per-arm rows (with intervals, per speed band), the GT
reference on the same windows, the ``ol``-vs-GT kinematic-contract check, and the
paired deltas per arm pair (``rec['refav1']['distance_keeping']``).

GOAL PROVENANCE (e609a98). ``plan()`` imagines its default goal from vision + nav +
measured v0 and stamps ``goal_source`` / ``goal_space`` / ``goal_action`` on the
result; this adapter banks them per window and reports per-arm fractions, the
selected-manoeuvre histogram (the TACTICAL family's "selected" half) and its
agreement with the declared heads. The ``_protocol.goal_source`` string is DERIVED
from those fractions, never a constant.
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
# path bootstrap — EAGER, and it evicts a wrongly-bound namespace package        #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                       # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                       # <repo>
_TE_PKG = os.path.join(_TE_PARENT, "taniteval")           # the REAL package dir


def _bootstrap_paths() -> None:
    """⛔ THE NAMESPACE-PACKAGE SHADOW. The outer ``<repo>/taniteval/`` has no
    ``__init__.py``; run from the repo root, ``import taniteval`` binds THAT
    directory as a namespace package and ``taniteval.ci`` "does not exist". Once
    bound, no sys.path edit undoes it — so the wrong binding is EVICTED here and
    the import is preflighted before anything expensive runs."""
    for p in (os.path.join(_REPO, "stack"), _TE_PARENT):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    m = sys.modules.get("taniteval")
    if m is not None:
        paths = [os.path.normcase(os.path.abspath(p))
                 for p in (getattr(m, "__path__", None) or [])]
        if os.path.normcase(os.path.abspath(_TE_PKG)) not in paths:
            for k in [k for k in sys.modules
                      if k == "taniteval" or k.startswith("taniteval.")]:
                del sys.modules[k]
    try:
        import taniteval.ci  # noqa: F401  (the preflight)
        import taniteval.four_families  # noqa: F401
    except ModuleNotFoundError as ex:                      # pragma: no cover
        sys.exit(f"[refav1_arm] taniteval preflight failed ({ex}). The real "
                 f"package is {_TE_PKG}; a namespace shadow of the outer dir "
                 f"was probably bound first. sys.path[:3]={sys.path[:3]}")


_bootstrap_paths()

# The ONE encoding constant (imported after the bootstrap puts `stack` on the
# path). NEVER spelled 2.9 locally -- a second copy is a second convention.
from tanitad.models.kinematic import STEER_WHEELBASE_M  # noqa: E402


def _load_t1():
    """Import the SIBLING ``t1_eval.py`` by file, so ``analyze`` / the tier
    machinery / the dump contract are REUSED, never copied."""
    spec = importlib.util.spec_from_file_location(
        "t1_eval_for_refav1", os.path.join(_HERE, "t1_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t1 = _load_t1()

# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #
DT = 0.2                 # refav1's operative tick (the loader REFUSES any other)
K_TRAJ_DEFAULT = 10      # 2.0 s = cfg.plan_steps at defaults = t1_eval's 20 x 0.1 s
ARM_TIERS = {"cl": "T1", "cl_navshuf": "T1", "cl_nonav": "T1",
             "cl_oraclegoal": "T0", "cl_oracleseed": "T0",
             "ha": "T1", "ha0": "T1", "ha0_ext": "T1",
             "ol": "T0"}
ARM_MEANING = {
    "cl": "T1 — plan() at t0, TRUE nav; predictor consumes the planner's own "
          "actions; trajectory = unicycle(controls, measured v0)",
    "cl_navshuf": "T1 — as cl with nav_cmd PERMUTED across eval windows "
                  "(D-REFAV1-NAV-DEPTH eval obligation)",
    "cl_nonav": "T1 — as cl with nav_cmd=None (index 0 'follow')",
    "cl_oraclegoal": "T0 — as cl with the TRUE future field as goal_field "
                     "(future information => T0; search-vs-goal attribution)",
    "cl_oracleseed": "T0 — ⭐ THE DE-CONFOUNDED ORACLE (D-REFAV1-ORACLE-DECONF): "
                     "the TRUE future field as goal_field WITH the head's own "
                     "canonical seed still in the pool (plan(goal_keeps_seed="
                     "True)). ⛔ `cl_oraclegoal` differs from `cl` in TWO "
                     "things — the goal AND the missing seed — so it bounds "
                     "nothing; THIS arm differs in the goal alone",
    "ha": "T1 — hold the last OBSERVED (a, kappa) (closes at t0) for K steps; "
          "consumes no recorded future",
    "ha0": "T1 — CONSTANT VELOCITY: a = 0, kappa = 0 at the measured v0, i.e. a "
           "straight line at constant speed. The STRONGEST TRIVIAL BASELINE and "
           "the echo test's real bar; consumes strictly less than ha (not even "
           "the last observed action)",
    "ha0_ext": "T1 — THE ECHO CONTROL (stack/tanitad/eval/echo_gate.py::ha0_ext, "
               "refav1 form): constant (a0, kappa0) from the MEASURED t0 state "
               "held for K steps. a0 = (v[2t] - v[2t-2]) / 0.2 is the SAME "
               "backward difference ha holds (the forward one reads v[2t+2], "
               "the future); kappa0 = actions[2t, 0], the recorded curvature "
               "AT t0, where ha reads it at t0-1 (echo_gate's named weakness "
               "#2, closed). Same unicycle as every other arm. ⚠️ NOT the "
               "corpus-ax form (echo_gate's weakness #1): the v2ep carries no "
               "measured ax, so the sharper variant is not constructible from "
               "these inputs — a WORK ITEM, stated, not hidden",
    "ol": "T0 — the RECORDED future (a, kappa) integrated from v0: the "
          "kinematic-contract control (must reproduce GT), NOT a WM diagnostic",
}
_TIER_NOTE = dict(t1._TIER_NOTE)
PLAN_SOURCE_NAMES = ["cem", "baseline:cv", "baseline:hold_v0",
                     "baseline:proposal", "baseline:decel_1.5"]
#: ``PlanResult.goal_source`` values (refa_v1.plan, e609a98); index 0 is what an
#: OLDER model file (no attribute) reads as — never inferred from anything else.
GOAL_SOURCE_NAMES = ["none", "supplied", "tactical_imagined", "supplied+seed"]
#: the banked B1 EVAL lead block (backlog R1): one row per (clip, RAW 10 Hz frame)
#: for the 147 v7.2 EVAL clips, built by ``tools/build_lead_block_b1.py``.
LEAD_BLOCK_DEFAULT = os.path.join(
    _REPO, "TanitAD Research Lab", "Benchmarks & Evals", "Research",
    "2026-09-02-b1-eval-lead-block", "raw", "b1_eval_lead_block.npz")
#: the dump's ``v0`` (poses[2t, 3]) and the block's ``speeds`` (egomotion at the
#: registered t0) are the SAME interpolation — a mismatch is a wrong clip/frame,
#: not noise. 1e-3 m/s absorbs float32 round-tripping only (dump_lead_join's rule).
LEAD_SPEED_TOL_MPS = 1e-3
LEAD_TS_TOL_S = 1e-6
_UNVERIFIED_ON_REAL_CKPT = (
    "UNVERIFIED on a real checkpoint — this box is forbidden from contacting "
    "Thor; validated on a random-init RefAV1 + synthetic eval slice only")


def _p(*a):
    print(*a, flush=True)


def _refused(reason, tier, n=0):
    """The binding shape for a family/metric that cannot be computed here."""
    return {"status": "REFUSED", "reason": reason, "n": int(n), "tier": tier,
            "estimator": "n/a — inputs missing (WORK ITEM, not a pass)"}


# --------------------------------------------------------------------------- #
# model loading                                                                 #
# --------------------------------------------------------------------------- #
def _sub_cfg(v, cls):
    if v is None or isinstance(v, cls):
        return v
    if isinstance(v, dict):
        return cls(**v)
    raise TypeError(f"cannot rebuild {cls.__name__} from {type(v).__name__}")


def build_config(cfg_d: dict):
    """``vars(RefAV1Config)`` (from ``ckpt['cfg']``) or its JSON form -> RefAV1Config.

    Refuses fields this box's ``RefAV1Config`` does not know: a checkpoint written
    by a NEWER trainer (e.g. one carrying ``speed_channel`` / ``ema_targets``)
    must be evaluated with the matching model file, never with those fields
    silently dropped."""
    from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
    from tanitad.refs.refa_v1 import RefAV1Config
    d = dict(cfg_d)
    d["tactical_cfg"] = _sub_cfg(d.get("tactical_cfg"), TacticalPolicyConfig)
    d["strategic_cfg"] = _sub_cfg(d.get("strategic_cfg"), StrategicPolicyConfig)
    known = {f.name for f in dataclasses.fields(RefAV1Config)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise SystemExit(
            f"[refav1_arm] the config carries fields this RefAV1Config does not "
            f"know: {unknown}. The checkpoint was written by a newer trainer than "
            f"stack/tanitad/refs/refa_v1.py on this box — sync the model file; "
            f"refusing to drop them silently (a_dim/speed_channel would change "
            f"the action contract).")
    return RefAV1Config(**d)


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if dataclasses.is_dataclass(v):
            out[k] = dataclasses.asdict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def load_model(ckpt_path: str, config_path: str | None = None,
               device: str = "cpu", allow_nonstrict: bool = False):
    """``(model, cfg, provenance)`` — STRICT load, fitted standardizer required.

    Config precedence: an explicit ``--config`` / sibling ``config.json`` is
    read first (the trainer's ``ckpt['cfg']`` is the fallback), and when BOTH
    exist their scalar fields are cross-checked — a contradiction is a refusal
    naming both values, never a silent override in either direction (the
    ``adopt_ckpt_geometry`` lesson in t1_eval).
    """
    import torch
    from tanitad.refs.refa_v1 import RefAV1
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        raise SystemExit(f"[refav1_arm] {ckpt_path} has no 'model' key — not a "
                         f"refa_v1_train.py checkpoint (keys: "
                         f"{sorted(ck)[:10] if isinstance(ck, dict) else type(ck)})")
    side = config_path or os.path.join(
        os.path.dirname(os.path.abspath(ckpt_path)), "config.json")
    cfg_json, cfg_ck = None, ck.get("cfg") if isinstance(ck.get("cfg"), dict) else None
    if os.path.exists(side):
        with open(side, encoding="utf-8") as fh:
            j = json.load(fh)
        cfg_json = j.get("cfg") if isinstance(j, dict) and isinstance(
            j.get("cfg"), dict) else j
        if not isinstance(cfg_json, dict):
            raise SystemExit(f"[refav1_arm] {side} is not a config dict")
    if cfg_json is not None and cfg_ck is not None:
        conflict = []
        for k, v in cfg_json.items():
            if k in ("tactical_cfg", "strategic_cfg"):
                continue
            cv = cfg_ck.get(k, "<absent>")
            if isinstance(v, list):
                v = tuple(v)
            if isinstance(cv, list):
                cv = tuple(cv)
            if cv != "<absent>" and cv != v:
                conflict.append(f"{k}: config.json {v!r} vs ckpt['cfg'] {cv!r}")
        if conflict:
            raise SystemExit("[refav1_arm] ⛔ config.json CONTRADICTS ckpt['cfg'] "
                             "— refusing rather than guessing which describes "
                             "the weights:\n  " + "\n  ".join(conflict))
    cfg_d = cfg_json if cfg_json is not None else cfg_ck
    src = side if cfg_json is not None else f"{ckpt_path}['cfg']"
    if cfg_d is None:
        raise SystemExit(f"[refav1_arm] no config: neither {side} nor "
                         f"ckpt['cfg'] — the model cannot be rebuilt")
    cfg = build_config(cfg_d)
    model = RefAV1(cfg)
    try:
        res = model.load_state_dict(ck["model"], strict=False)
    except RuntimeError as ex:                     # size mismatch: cfg vs weights
        raise SystemExit(
            f"[refav1_arm] ⛔ the config and the weights DISAGREE ON SHAPE — e.g. "
            f"a config claiming speed_channel/a_dim/d_state the checkpoint was "
            f"not trained with. Refusing; fix the config, never the weights.\n"
            f"{str(ex)[:1200]}") from None
    strict_rep = {"missing_keys": list(res.missing_keys),
                  "unexpected_keys": list(res.unexpected_keys)}
    if (res.missing_keys or res.unexpected_keys) and not allow_nonstrict:
        raise SystemExit(f"[refav1_arm] ⛔ NON-STRICT LOAD: missing "
                         f"{list(res.missing_keys)[:8]} unexpected "
                         f"{list(res.unexpected_keys)[:8]} — the checkpoint and "
                         f"the model file disagree. Pass --allow-nonstrict only "
                         f"for a deliberate diagnostic, and say so in the report.")
    if not bool(model.std.fitted):
        raise SystemExit("[refav1_arm] ⛔ the checkpoint's FeatureStandardizerV1 is "
                         "not fitted — this is not a trained refav1 checkpoint "
                         "(the stats are part of the parity contract; never refit "
                         "at eval)")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # ⚠️ EVAL-TIME LOSS-KNOB NEUTRALISATION, recorded: forward() with w_cf > 0
    # REFUSES batch 1 (counterfactuals need >= 2 rows) and min_participation can
    # SystemExit on a small batch. Both are loss-side knobs with no parameters;
    # the state_dict is byte-identical either way.
    overrides = {}
    if getattr(cfg, "w_cf", 0.0):
        overrides["w_cf"] = (cfg.w_cf, 0.0)
        cfg.w_cf = 0.0
    if getattr(cfg, "min_participation", 0.0):
        overrides["min_participation"] = (cfg.min_participation, 0.0)
        cfg.min_participation = 0.0
    prov = {"ckpt": ckpt_path, "step": ck.get("step"), "config_source": src,
            "state_dict_load": strict_rep,
            "eval_time_cfg_overrides": {k: {"trained": a, "eval": b}
                                        for k, (a, b) in overrides.items()},
            "cfg": _jsonable(dataclasses.asdict(cfg)),
            "trainable_parameters": int(model.trainable_parameters()),
            "a_dim": int(cfg.a_dim), "target_space": cfg.target_space,
            "tac_vocab_version": getattr(cfg, "tac_vocab_version", "v6.0")}
    return model, cfg, prov


# --------------------------------------------------------------------------- #
# actions / trajectories                                                       #
# --------------------------------------------------------------------------- #
def hold_action_controls(loader, v, kap, t: int):
    """The last OBSERVED action — the (a, kappa) that CLOSES at t0.

    ``loader._kin_actions(v, kap, t-1, 1)`` reads ``v[2t-2]``, ``v[2t]`` and
    ``kap[2t-2]``: every frame is <= 2t, so NOTHING recorded after t0 enters.
    (The action OPENING at t would need ``v[2t+2]`` — a future value — which is
    why it is not the one held.)
    """
    if t < 1:
        raise ValueError("hold-action needs t >= 1 (one closed action before t0)")
    return loader._kin_actions(v, kap, t - 1, 1)[0]              # [2]


def hold_ext_controls(loader, v, kap, t: int, *, dt: float | None = None,
                      stride: int = 2):
    """``ha0_ext`` (refav1 form): the (a0, kappa0) of the MEASURED t0 state.

    ``a0 = (v[i0] - v[i0-stride]) / dt`` — the SAME backward difference
    `hold_action_controls` holds (the forward one, ``v[i0+stride]``, is future);
    ``kappa0 = kap[i0]`` — the recorded curvature AT t0, where ``ha`` holds
    ``kap[i0-stride]`` (`echo_gate.ha_finite_diff_accel`'s "reads the steer at
    t0-1" weakness, closed). Every index is <= i0: nothing recorded after t0
    enters. ⚠️ `echo_gate.ha0_ext`'s STRONGER form is built on the corpus's own
    measured ``ax``; the v2ep carries none, so this is the sharpest ADMISSIBLE
    form on these inputs, and `ARM_MEANING` names it as such.

    ⭐ ``stride`` EXISTS SO THERE IS ONE IMPLEMENTATION, NOT TWO (Rung A1,
    2026-09-05). refav1 windows a 0.1 s pose array on its own 0.2 s operative
    tick, so its origin is ``i0 = 2t`` — the DEFAULT, and with it this function
    is bit-identical to the pre-Rung-A1 file. ``refcv3_arm.py`` indexes the same
    array at the native 0.1 s tick and calls this with ``stride=1, dt=0.1``,
    passing ``loader=None`` (the loader is read ONLY for its ``dt``). A control
    re-implemented beside the harness that uses it is a control that can drift
    away from it, and then the gate measures the drift instead of the model —
    `echo_gate.ha0_ext`'s own docstring says exactly this about its shared base.
    Pinned by ``stack/tests/test_refcv3_ha0_ext_shared.py``.
    """
    stride = int(stride)
    i0 = stride * int(t)
    if i0 < stride:
        raise ValueError("ha0_ext needs one closed step before t0 "
                         f"(stride={stride}, t={t} -> i0={i0})")
    import torch
    dt = float(dt if dt is not None else loader.dt)
    a0 = (v[i0] - v[i0 - stride]) / dt
    return torch.stack([a0, kap[i0]]).float()                    # [2]


def hold_v0_controls(k: int):
    """⭐ THE STRONGEST TRIVIAL BASELINE: ``a = 0, kappa = 0`` for K steps.

    Integrated from the measured ``v0`` this is a CONSTANT-VELOCITY STRAIGHT LINE — the
    control C101 measured our CEM planner 35.8 % worse than, and the one the echo test
    must actually be run against.

    ⛔ WHY IT EXISTS (MEASURED 2026-09-03, and it voided a read): at step 1,000 the
    closed-loop arm ``cl`` was a straight constant-speed line on **140/140** windows, so
    the whole "lateral planning beats holding" reading was a straight line beating the
    hold-action control's held, NOISY kappa (which drifts 0.12 m even where the human
    drives straight). Against ``ha`` alone that reads as lateral SKILL. Against ``ha0``
    it reads as what it is: nothing. ``ha`` is NOT the trivial floor — it is a control
    that can be WORSE than trivial, and comparing only to it manufactures a win.

    Consumes strictly less than ``hold_action_controls``: no recorded action at all,
    only the measured ``v0`` the PI ruling of 2026-09-02 admits.
    """
    import torch
    return torch.zeros(int(k), 2, dtype=torch.float32)           # [K, 2]


def paths_from_controls(controls, v0: float, dt: float, k: int, *,
                        action_units: str = "kappa"):
    """``[K',>=2]`` controls -> ``[1,K,2]`` ego-frame path via the programme's
    ONE unicycle integrator (``refa_v1_plan.unicycle_paths``): position advances
    on the speed at the START of the step, v updates last, clamped at 0.

    ⭐ ``action_units`` names the unit channel 1 ARRIVES in (PI ruling
    2026-09-03; contract on `kinematic.STEER_WHEELBASE_M`). ``"kappa"`` (default)
    integrates as supplied -- byte-identical to every pre-2026-09-03 call, and
    the correct reading for a PLANNER candidate. ``"steer"`` converts with
    ``kappa = tan(steer)/L_enc`` first -- the correct reading for a RECORDED
    v2ep action, which is a road-wheel angle."""
    import torch
    from tanitad.refs.refa_v1_plan import unicycle_paths
    c = controls[..., :2]
    if c.dim() == 2:
        c = c[None]
    if c.shape[1] < k:
        raise ValueError(f"controls carry {c.shape[1]} steps for a {k}-step "
                         f"horizon — no action exists beyond the plan; refusing "
                         f"to extrapolate")
    v0_t = torch.as_tensor([float(v0)], dtype=torch.float32, device=c.device)
    return unicycle_paths(c[:, :k].float(), v0_t, dt, action_units=action_units)


def gt_waypoints(poses, t: int, k: int):
    """GT ego-frame waypoints at cache steps t+1..t+k from the RAW 10 Hz poses
    (frame 2(t+j)), via ``metric_dynamics.gt_ego_waypoints`` — the same geometry
    every other dump uses. ``poses`` [T_ep, 4] = (x, y, yaw, v)."""
    from tanitad.models.metric_dynamics import gt_ego_waypoints
    f0 = 2 * t
    fut = poses[f0 + 2: f0 + 2 * k + 1: 2]
    if fut.shape[0] != k:
        raise ValueError(f"episode too short: window t={t} needs frames up to "
                         f"{f0 + 2 * k}, poses has {poses.shape[0]}")
    return gt_ego_waypoints(poses[f0][None], fut[None], list(range(1, k + 1)))


# --------------------------------------------------------------------------- #
# window selection                                                             #
# --------------------------------------------------------------------------- #
def select_windows(ld, stride: int, window_list: str | None):
    """-> (``[(wi, ei, t), ...]``, meta or None). The panel's own definition.

    ⭐ ``window_list = None`` is the SHIPPED path and is bit-identical to the
    pre-2026-09-05 expression ``(t - (W - 1)) % stride == 0`` — pinned by
    ``stack/tests/test_refav1_window_list.py::test_a_*``, each with a
    same-breath control that must differ. Every banked dump was produced on that
    path and none of them move.

    ⛔ **A STRIDE CANNOT ENRICH A STRATUM.** The banked stride-16 panel carries
    11 GT-left and 8 GT-right turns, and a recall on 11 trials has a resolution
    of 1/11 = 0.0909 — COARSER than the 0.0750 inference-seed floor it is being
    compared against, so it cannot represent that floor in either direction.
    ``--window-list`` supplies an explicit, externally computed, GT-only
    stratified selection so a per-direction rate can be resolved at all.

    ⚠️ The file is (episode NAME, t) pairs, never window indices: an index is
    meaningful only against one loader construction and would silently select a
    DIFFERENT window under any change of episode set or grid. A requested pair
    that the loader does not carry is REFUSED by name rather than dropped — a
    silently thinned panel is how an underpowered stratum gets read as a
    negative.
    """
    if not window_list:
        return [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
                if (t - (ld.W - 1)) % stride == 0], None
    with open(window_list, "rb") as fh:
        blob = fh.read()
    doc = json.loads(blob.decode("utf-8"))
    want = [(str(e), int(t)) for e, t in doc["windows"]]
    if len(set(want)) != len(want):
        raise SystemExit(f"[refav1_arm] ⛔ {window_list}: duplicate (episode, t) "
                         f"pairs — {len(want) - len(set(want))} of {len(want)}")
    have = {(ld.names[ei], t): wi for wi, (ei, t) in enumerate(ld.windows)}
    missing = [w for w in want if w not in have]
    if missing:
        raise SystemExit(
            f"[refav1_arm] ⛔ {window_list}: {len(missing)} of {len(want)} "
            f"requested windows are not on this loader's grid; first five "
            f"{missing[:5]} — refusing rather than silently thinning the panel")
    sel = sorted((have[w] for w in want))
    return ([(wi, ld.windows[wi][0], ld.windows[wi][1]) for wi in sel],
            {"path": window_list,
             "sha256": hashlib.sha256(blob).hexdigest(),
             "rule": str(doc.get("rule", "<no rule declared>")),
             "n_requested": len(want), "n_selected": len(sel)})


# --------------------------------------------------------------------------- #
# nav shuffle                                                                  #
# --------------------------------------------------------------------------- #
def shuffle_nav(nav: np.ndarray, valid: np.ndarray, seed: int):
    """Permute ``nav`` ACROSS the eval windows (among the nav-valid ones), the
    control D-REFAV1-NAV-DEPTH makes an obligation. Returns ``(nav_shuf, stats)``;
    ``n_changed`` matters — with a FOLLOW-heavy marginal most rows keep their
    token, so the control's power lives in the changed subset, which the
    analysis reports separately."""
    nav = np.asarray(nav, dtype=np.int64).copy()
    valid = np.asarray(valid, dtype=bool)
    out = nav.copy()
    idx = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    if idx.size:
        out[idx] = nav[idx[rng.permutation(idx.size)]]
    changed = int((out != nav).sum())
    return out, {"seed": int(seed), "n_windows": int(nav.size),
                 "n_valid": int(idx.size), "n_changed": changed,
                 "changed_frac": round(changed / max(1, nav.size), 4),
                 "rule": "permutation among nav_valid windows; invalid rows keep "
                         "nav_cmd=0 / nav_valid=False and are EXCLUDED from "
                         "route accuracy"}


def nav_id_to_route_id():
    """NavEmitter id (enumerates ``vocab_v7.NAV_COMMAND_TOKENS``) -> the
    ``refb.ROUTE_CLASSES`` index the SAME token maps to in the loader — derived
    from the loader's own table, never hand-typed."""
    from tanitad.data.refav1_loader import _NAV_TOKEN_TO_ROUTE
    from tanitad.models.vocab_v7 import NAV_COMMAND_TOKENS
    from tanitad.refs.refb import ROUTE_CLASSES
    return {i: ROUTE_CLASSES.index(_NAV_TOKEN_TO_ROUTE[tok])
            for i, tok in enumerate(NAV_COMMAND_TOKENS)
            if tok in _NAV_TOKEN_TO_ROUTE}


# --------------------------------------------------------------------------- #
# the roll (GPU or CPU) — writes the dump                                       #
# --------------------------------------------------------------------------- #
def _plan_cfg(cfg, a):
    from tanitad.refs.refa_v1_plan import PlanConfig
    kw = {"horizon": cfg.plan_steps, "dt": cfg.op_dt, "seed": int(a.plan_seed)}
    for name in ("n_samples", "n_iters", "n_elites"):
        v = getattr(a, f"plan_{name}", None)
        if v is not None:
            kw[name] = int(v)
    mu = getattr(a, "kamm_mu", None)
    if mu is not None:
        import dataclasses as _dc
        if "kamm_mu" not in {f.name for f in _dc.fields(PlanConfig)}:
            raise SystemExit(
                "[refav1_arm] --kamm-mu needs a PlanConfig with kamm_mu; "
                "this stack predates the friction-circle cap")
        kw["kamm_mu"] = float(mu)
        vf = getattr(a, "kamm_v_floor", None)
        if vf is not None:
            kw["kamm_v_floor"] = float(vf)
    pc = PlanConfig(**kw)
    pc.sanity()
    return pc


def _source_code(src: str) -> int:
    if src not in PLAN_SOURCE_NAMES:
        PLAN_SOURCE_NAMES.append(src)
    return PLAN_SOURCE_NAMES.index(src)


def _goal_code(src: str) -> int:
    if src not in GOAL_SOURCE_NAMES:
        GOAL_SOURCE_NAMES.append(src)
    return GOAL_SOURCE_NAMES.index(src)


def episode_names(cache_dir: str) -> list[str]:
    from pathlib import Path
    return sorted(p.stem for p in Path(cache_dir).glob("*.pt")
                  if p.stem != "index")


def build_loader(a, cfg, k_loader: int, names=None):
    """The arm's OWN loader (``RefAV1Windows``) on the eval cache — the grid
    refusal, the v7.2 label/nav join and the (a, kappa) channel repair are all
    its, not re-derived here. ``str_ext_steps=0``: the strategic extension is
    not evaluated, so the window grid reaches only ``k_loader`` steps."""
    from tanitad.data.refav1_loader import RefAV1Windows
    return RefAV1Windows(a.cache, a.episodes, op_window=cfg.op_window,
                         op_steps=k_loader, op_dt=cfg.op_dt, str_dt=cfg.str_dt,
                         str_ext_steps=0, lru=a.lru, seed=0, episodes=names,
                         labels_path=a.labels, nav_path=a.nav)


def run_dump(a) -> dict:
    """Roll every arm over the eval grid and write the dump. Returns provenance."""
    import torch
    from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v

    t_start = time.time()
    dev = a.device
    model, cfg, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)
    k = int(a.horizon_k)
    if k > cfg.plan_steps:
        raise SystemExit(f"[refav1_arm] --horizon-k {k} exceeds the plan horizon "
                         f"{cfg.plan_steps}: beyond it the T1 arm has no action. "
                         f"Refusing to extrapolate.")
    k_wm = int(a.wm_k) if a.wm_k else int(cfg.op_steps)
    k_loader = max(k, k_wm)
    names = episode_names(a.cache)
    if not names:
        raise SystemExit(f"[refav1_arm] no <episode>.pt under {a.cache}")
    if a.episodes_n:
        names = names[:int(a.episodes_n)]
    ld = build_loader(a, cfg, k_loader, names)
    vv = getattr(cfg, "tac_vocab_version", "v6.0")
    from tanitad.data.refav1_loader import TAC_VOCAB_VERSION
    if a.labels and vv != TAC_VOCAB_VERSION:
        raise SystemExit(f"[refav1_arm] ⛔ label vocabulary {TAC_VOCAB_VERSION} "
                         f"(loader) != model heads' {vv}: the label indices and "
                         f"the head indices would disagree. Refusing.")
    lat_names, lon_names = list(tactical_lat_actions(vv)), list(tactical_lon_actions_v(vv))
    if model.lat_head[-1].out_features != len(lat_names) or \
            model.lon_head[-1].out_features != len(lon_names):
        raise SystemExit("[refav1_arm] head width != vocabulary size — the "
                         "checkpoint's vocabulary is not the one this box knows")
    _p(f"[model] {prov['ckpt']} step={prov['step']} params="
       f"{prov['trainable_parameters']:,} a_dim={cfg.a_dim} target_space="
       f"{cfg.target_space} vocab={vv} config<-{prov['config_source']}")
    _p(f"[loader] {len(ld)} windows over {len(ld.names)} episodes "
       f"(W={ld.W}, K_loader={k_loader}, grid 0.2 s); labels="
       f"{'ON' if ld._labels_on else 'OFF'} nav={'ON' if ld._nav_on else 'OFF'}")

    # ⭐ THE ACTION-UNIT CONTRACT for this run (PI ruling 2026-09-03; see
    # `kinematic.STEER_WHEELBASE_M`). "kappa" = the LEGACY reading under which
    # every banked refav1 number was produced: recorded v2ep actions integrated
    # as if channel 1 were a curvature, and planner candidates handed to the
    # model unconverted. "steer" = the repaired contract. DEFAULT IS LEGACY so a
    # re-analysis of a banked dump is byte-identical.
    rec_units = getattr(a, "action_units", "kappa")
    if rec_units not in ("kappa", "steer"):
        raise SystemExit(f"[refav1_arm] --action-units must be kappa|steer, "
                         f"got {rec_units!r}")
    if rec_units != "kappa":
        _p(f"[units] ⚠️ action_units={rec_units}: recorded actions are converted "
           f"kappa = tan(steer)/{STEER_WHEELBASE_M} before integration and "
           f"planner candidates reach the model as arctan("
           f"{STEER_WHEELBASE_M}*kappa). NUMBERS ARE NOT COMPARABLE to a "
           f"kappa-unit run.")
    stride = max(1, int(a.window_stride))
    sel, wlist_meta = select_windows(ld, stride,
                                     getattr(a, "window_list", None))
    if not sel:
        raise SystemExit("[refav1_arm] the stride selected zero windows")
    if wlist_meta is not None:
        _p(f"[window-list] {wlist_meta['n_selected']} windows from "
           f"{wlist_meta['path']} sha256={wlist_meta['sha256'][:16]} "
           f"rule={wlist_meta['rule'][:70]!r} "
           f"(--window-stride IGNORED)")
    # ---- nav for every selected window (the loader's own table), then shuffle
    nav_true = np.zeros(len(sel), dtype=np.int64)
    nav_valid = np.zeros(len(sel), dtype=bool)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)
            nav_valid[i] = nid is not None
    nav_shuf, shuf_stats = shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)

    arms = ["cl", "ha", "ha0", "ha0_ext", "ol"]
    if ld._nav_on and not a.no_navshuf:
        arms.append("cl_navshuf")
    if a.with_nonav_arm:
        arms.append("cl_nonav")
    if a.with_oracle_goal_arm:
        arms.append("cl_oraclegoal")
    if getattr(a, "with_oracle_seed_arm", False):
        arms.append("cl_oracleseed")
    plan_arms = [x for x in arms if x.startswith("cl")]
    pc = _plan_cfg(cfg, a)
    # ⭐ THE COST FLAGS (D-REFAV1-CCOS-EVAL). `cost_metric` selects the goal
    # term's form (`refa_v1.COST_METRICS`); `cost_weights` is the declared
    # (W_JERK, W_KAPPA, W_VEND) triple, None = the shipped module constants.
    # ⛔ VERIFY-GATE: a stale stack that predates the branch under test would
    # otherwise raise deep inside plan() after the model is loaded — or worse,
    # silently score a differently-named metric. Refuse up front, by name.
    from tanitad.refs import refa_v1 as _R
    from taniteval import four_families as _ff
    # ⭐ M15's LEVEL SET, resolved once. ⛔ STALE-STACK GATE: a stack that
    # predates the level set has no `GOAL_KAPPA_TURN_LEVELS`, and `plan()` would
    # SILENTLY IGNORE the kwarg (it is keyword-only with a default) and bank an
    # L1 arm under an L3 name.
    gk_levels = getattr(a, "goal_kappa_levels", None)
    if gk_levels is not None:
        if not hasattr(_R, "GOAL_KAPPA_TURN_LEVELS"):
            raise SystemExit("[refav1_arm] ⛔ STALE STACK: --goal-kappa-levels "
                             "asked for but refa_v1 has no level set")
        gk_levels = (tuple(_R.GOAL_KAPPA_TURN_LEVELS)
                     if str(gk_levels).strip().lower() == "m15"
                     else tuple(float(x) for x in str(gk_levels).split(",")))
        if getattr(a, "goal_kappa_hint", None) is None:
            raise SystemExit(
                "[refav1_arm] ⛔ --goal-kappa-levels needs --goal-kappa-hint: "
                "the v7.0 head has ONE TURN slot per direction and cannot "
                "select a magnitude, so the chooser must be named and its TIER "
                "declared ('gt' => T0)")
    gk_vocab = _R.goal_kappa_vocab_id(gk_levels) if hasattr(
        _R, "goal_kappa_vocab_id") else "L1-0.08"
    cost_metric = str(getattr(a, "cost_metric", None) or "cos")
    if cost_metric not in getattr(_R, "COST_METRICS", ("cos",)):
        raise SystemExit(f"[refav1_arm] ⛔ STALE STACK: cost_metric={cost_metric!r} "
                         f"is not in COST_METRICS={getattr(_R, 'COST_METRICS', None)} "
                         f"of {_R.__file__}; refusing to score a metric this "
                         f"tree does not implement")
    cw_raw = getattr(a, "cost_weights", None)
    cost_weights = None
    if cw_raw:
        parts = [float(x) for x in str(cw_raw).split(",")]
        if len(parts) != 3:
            raise SystemExit("[refav1_arm] --cost-weights must be 'w_jerk,w_kappa,"
                             f"w_vend', got {cw_raw!r}")
        cost_weights = tuple(parts)
    shipped_w = {"W_JERK": float(_R.W_JERK), "W_KAPPA": float(_R.W_KAPPA),
                 "W_VEND": float(_R.W_VEND)}
    used_w = (dict(zip(("W_JERK", "W_KAPPA", "W_VEND"), cost_weights))
              if cost_weights else dict(shipped_w))
    _p(f"[cost] metric={cost_metric} weights={used_w} "
       f"({'CLI override' if cost_weights else 'shipped module constants'}); "
       f"COST_METRICS={_R.COST_METRICS} from {_R.__file__}")

    # ⛔⛔ THE JERK SEAM IS MULTIPLIED BY `W_JERK`. REFUSE AN ARM THAT IS
    # INERT BY CONSTRUCTION (D-REFAV1-LON-SEAM-INERT, 2026-09-05).
    # MEASURED, and it cost a GPU hour before this guard existed: `lonseam` was
    # launched as `--jerk-seam a0` against the `wk15` baseline triple
    # `(0.0, 15.11245, 64.29715042415070)`. The cost line is
    # `c = c + w_jerk * jerk.pow(2).mean(-1)`, so with `w_jerk = 0.0` REPAIRING
    # `jerk` cannot change `c` BY A SINGLE BIT. The arm ran to completion, the
    # reached-it guard passed (the flag DID reach `plan()`; the record carries
    # `jerk_seam: "a0"`), and it produced `+0.0000 [+0.0000, +0.0000]` on all
    # ten family metrics with emitted controls bit-identical to `wk15` --
    # a null that is pure arithmetic and says NOTHING about the seam.
    # ⚠ A null arm whose lever is multiplied by zero is not a null about the
    # lever; it is a null about a term that was switched off. Same family as
    # M23 (4) ("a NULL arm with a live optimiser is not a null") and as counting
    # on the wrong predicate: the experiment was uninformative BEFORE it ran,
    # and the only thing that catches that is a refusal at flag-parse time.
    # ⇒ The informative pair is `w_jerk > 0` with the seam OFF vs the SAME
    # `w_jerk` with the seam ON -- two arms, one variable between them.
    if getattr(a, "jerk_seam", "off") != "off" and float(used_w["W_JERK"]) == 0.0:
        raise SystemExit(
            "[refav1_arm] ⛔ --jerk-seam %s with W_JERK = 0.0 is INERT BY "
            "CONSTRUCTION: the cost adds `w_jerk * mean(jerk^2)`, so repairing "
            "`jerk` while `w_jerk` is zero cannot change the objective by a "
            "single bit, and the arm would bank a meaningless +0.0000. Pass a "
            "non-zero W_JERK in --cost-weights (and compare against a baseline "
            "carrying the SAME W_JERK with --jerk-seam off, so the seam is the "
            "one variable). Refusing before the rollout rather than after it."
            % getattr(a, "jerk_seam", "off"))

    # ⭐⭐ THE GOAL-HEAD DECISION RULE (D-REFAV1-DRIVE-GATE, 2026-09-05).
    # `--lat-logit-bias` is an additive vector on the lateral logits inside
    # `_imagine_tactical_goal`, i.e. on the ONE decision that determines whether
    # a turn is reachable by the planner at all: the decoded token's canonical
    # profile is the population's only curvature-carrying candidate, so a
    # LANE_KEEP decode forces curvature EXACTLY 0 (MEASURED 244/244 windows on
    # the trained checkpoint). The whole decision-rule family is one vector —
    # a class-prior correction is `-tau*log(prior)`, a commit threshold is a
    # negative entry in one slot, and PLAIN ARGMAX IS None/zeros.
    # ⛔ VERIFY-GATE, exactly as for cost_metric above: a stack that predates
    # the parameter would silently ignore the flag and bank an argmax arm under
    # a decision-rule name. Refuse up front, by signature.
    lb_raw = getattr(a, "lat_logit_bias", None)
    lat_logit_bias = None
    if lb_raw:
        import inspect as _inspect
        if "lat_logit_bias" not in _inspect.signature(_R.RefAV1.plan).parameters:
            raise SystemExit(
                "[refav1_arm] ⛔ STALE STACK: --lat-logit-bias was given but "
                f"{_R.__file__}'s plan() has no such parameter; refusing to "
                "bank an argmax arm under a decision-rule name")
        parts = [float(x) for x in str(lb_raw).split(",")]
        if len(parts) != len(lat_names):
            raise SystemExit(
                f"[refav1_arm] --lat-logit-bias needs {len(lat_names)} "
                f"comma-separated values for vocabulary {vv} "
                f"({lat_names}), got {len(parts)}: {lb_raw!r}")
        lat_logit_bias = torch.tensor(parts, dtype=torch.float32, device=dev)
        _p(f"[goal-rule] lat_logit_bias="
           f"{dict(zip(lat_names, parts))} "
           f"(argmax == all zeros; non-zero CHANGES which token the goal and "
           f"the iCEM seed are built from)")
    else:
        _p("[goal-rule] lat_logit_bias=None -> plain argmax (the legacy path, "
           "bit-identical to pre-2026-09-05 arms)")

    # ⭐⭐ `--goal-kappa-turn` — THE VOCABULARY KNOB (D-REFAV1-VOCAB-QUANT).
    # `canonical_controls` can command exactly TWO sustained curvatures, 0 and
    # GOAL_KAPPA_TURN = 0.08 (R 12.5 m), because NUDGE_*/LANE_CHANGE_* are
    # S-curves with ZERO net heading change. The corpus curves at R 100-1000 m,
    # so on 90.6 % of GT-turn windows LANE_KEEP is the VOCABULARY-OPTIMAL token
    # and no decision rule or head re-fit can reach the road.
    # ⛔ SAME VERIFY-GATE AS ABOVE: a stack predating the parameter would ignore
    # the flag and bank a shipped-vocabulary arm under a finer-vocabulary name.
    gk_raw = getattr(a, "goal_kappa_turn", None)
    goal_kappa_turn = None
    if gk_raw is not None:
        import inspect as _inspect2
        if "goal_kappa_turn" not in _inspect2.signature(_R.RefAV1.plan).parameters:
            raise SystemExit(
                "[refav1_arm] ⛔ STALE STACK: --goal-kappa-turn was given but "
                f"{_R.__file__}'s plan() has no such parameter; refusing to "
                "bank a shipped-vocabulary arm under a finer-vocabulary name")
        goal_kappa_turn = float(gk_raw)
        if not (0.0 < goal_kappa_turn <= float(_R.GOAL_KAPPA_MAX)):
            raise SystemExit(
                f"[refav1_arm] --goal-kappa-turn {goal_kappa_turn} outside "
                f"(0, GOAL_KAPPA_MAX={_R.GOAL_KAPPA_MAX}] — the planner clips "
                "curvature at that bound, so a larger goal is unreachable")
        _p(f"[goal-vocab] goal_kappa_turn={goal_kappa_turn} 1/m "
           f"(R {1.0 / goal_kappa_turn:.1f} m) — SHIPPED is "
           f"{_R.GOAL_KAPPA_TURN} (R {1.0 / _R.GOAL_KAPPA_TURN:.1f} m). This "
           "changes the GOAL FIELD the whole search is scored against.")
    else:
        _p(f"[goal-vocab] goal_kappa_turn=None -> shipped GOAL_KAPPA_TURN="
           f"{_R.GOAL_KAPPA_TURN} (bit-identical to pre-2026-09-05 arms)")
    _p(f"[grid] K={k} ({k * DT:.1f} s) K_wm={k_wm} stride={stride} "
       f"windows={len(sel)} arms={arms} plan={{samples {pc.n_samples}, iters "
       f"{pc.n_iters}, elites {pc.n_elites}, seed {pc.seed}}} nav_shuffle="
       f"{shuf_stats['n_changed']}/{shuf_stats['n_windows']} changed")
    os.makedirs(a.dump_dir, exist_ok=True)
    os.makedirs(os.path.join(a.dump_dir, "decisions"), exist_ok=True)

    # --- L3: the SUSTAINED-CURVATURE SEED LADDER (D-REFAV1-COST-GEOMETRY).
    # NOT a vocabulary change: `canonical_controls` and the goal field are
    # untouched, only the search's iteration-0 candidate set is widened.
    skl_raw = getattr(a, "seed_kappa_ladder", None)
    seed_kappa_ladder = None
    if skl_raw:
        # ⛔ import HERE, not inherited: `_inspect2` is bound inside the
        # `--goal-kappa-turn` branch above, so referencing it from this block
        # raises UnboundLocalError on any arm that passes the ladder WITHOUT
        # also passing --goal-kappa-turn. MEASURED 2026-09-05: it killed
        # `l3ladder` and `combined` at startup (no GPU wasted - the tool fails
        # before the rollout, which is what its preflight design is for).
        import inspect as _inspect3
        if "seed_kappa_ladder" not in _inspect3.signature(_R.RefAV1.plan).parameters:
            raise SystemExit(
                "[refav1_arm] --seed-kappa-ladder needs a refa_v1.plan() that "
                "accepts seed_kappa_ladder; this stack predates it")
        seed_kappa_ladder = tuple(float(x) for x in str(skl_raw).split(","))
        if any(not (0.0 < k <= float(_R.GOAL_KAPPA_MAX)) for k in seed_kappa_ladder):
            raise SystemExit(
                f"[refav1_arm] --seed-kappa-ladder {seed_kappa_ladder} must be "
                f"finite, > 0 and <= kappa_max {_R.GOAL_KAPPA_MAX}")
        _p(f"[seed-pool] seed_kappa_ladder={seed_kappa_ladder} 1/m -> "
           f"{2 * len(seed_kappa_ladder)} EXTRA iteration-0 candidates "
           f"(both signs). Radii "
           + ", ".join(f"{1.0 / k:.0f} m" for k in seed_kappa_ladder))
    else:
        _p("[seed-pool] seed_kappa_ladder=None -> the SHIPPED pool "
           "(proposal modes + the decoded goal's canonical controls), "
           "bit-identical to every arm banked before 2026-09-05")


    frozen = cfg.target_space == "frozen"
    ld._order = [0]
    ld._cursor = 0
    episodes_manifest = []
    n_done, t_plan_first = 0, None
    goal_space_seen: set[str] = set()
    by_ep: dict[int, list] = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))

    for fi, ei in enumerate(sorted(by_ep)):
        nm = ld.names[ei]
        o = torch.load(ld.episode_dir / f"{nm}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        poses = o["poses"].float()
        F_ep, v_ep, kap_ep = ld._episode(nm)
        acc = {kk: [] for kk in ["g", "v0"] + arms}
        dec: dict[str, list] = {}
        ws = []
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)                          # [1,W,N,d]
            fut = b["future_feats"].to(dev)                     # [1,K_l,N,d]
            act = b["actions"].to(dev)                          # [1,K_l,a_dim]
            if act.shape[-1] != cfg.a_dim:
                raise SystemExit(f"[refav1_arm] loader actions are "
                                 f"{act.shape[-1]}-wide, cfg.a_dim={cfg.a_dim}: "
                                 f"the action CONTRACT changed; refusing")
            v0 = float(v_ep[2 * t])
            if b.get("v0") is not None and abs(float(b["v0"][0]) - v0) > 1e-6:
                raise RuntimeError("loader v0 != poses[2t, 3] — loader drift")
            v0_t = torch.tensor([v0], dtype=torch.float32, device=dev)
            if b.get("nav_cmd") is not None and int(b["nav_cmd"][0]) != int(nav_true[i]):
                raise RuntimeError("nav table / batch() disagree — loader drift")
            nav_t = torch.tensor([int(nav_true[i])], device=dev) if ld._nav_on else None
            nav_s = torch.tensor([int(nav_shuf[i])], device=dev) if ld._nav_on else None
            zeros = torch.zeros(1, dtype=torch.long, device=dev)
            with torch.no_grad():
                # -- GT and the two non-planning arms --------------------------
                g = gt_waypoints(poses, t, k)                    # [1,k,2]
                # ⭐ `ol` and `ha` replay RECORDED v2ep actions, whose channel 1
                # is a road-wheel angle. `rec_units` states that; "kappa" is the
                # legacy (unconverted) reading every banked number was produced
                # under. `ha0` is exactly zero, which is 0 in either unit.
                ol = paths_from_controls(act[0], v0, DT, k,
                                         action_units=rec_units)
                hold = hold_action_controls(ld, v_ep, kap_ep, t).to(dev)
                ha = paths_from_controls(hold[None].expand(k, 2), v0, DT, k,
                                         action_units=rec_units)
                # ⭐ the constant-velocity floor: SAME integrator, SAME v0, zero
                # controls — so any difference from `ha` is the held action alone.
                ha0 = paths_from_controls(hold_v0_controls(k).to(dev), v0, DT, k)
                # ⭐ the ECHO control (echo_gate.ha0_ext, refav1 form): the
                # measured t0 state's (a0, kappa0) held — kappa read AT t0.
                ext = hold_ext_controls(ld, v_ep, kap_ep, t).to(dev)
                ha0_ext = paths_from_controls(ext[None].expand(k, 2), v0, DT, k,
                                              action_units=rec_units)
                # -- T0 WM diagnostic + true-nav decisions via forward() ------
                # 2-wide controls + measured v0: the MODEL derives any speed
                # channel (augment_actions) — never widened here.
                out = model(feats, act[:, :k_wm], future_feats=fut[:, :k_wm],
                            nav_cmd=nav_t, v0=v0_t)
                if frozen:
                    pred_e = model.to_enc(out["op_pred"])
                    tgt = model.std(fut[:, :k_wm])
                    src = model.std(feats[:, -1:])
                else:
                    pred_e = out["op_pred"]
                    tgt = model.adapter(model.std(fut[:, :k_wm]))
                    src = model.adapter(model.std(feats))[:, -1:]
                mse_m = ((pred_e - tgt) ** 2).mean(dim=(-1, -2))[0]     # [K_wm]
                mse_c = ((src - tgt) ** 2).mean(dim=(-1, -2))[0]
                # the CONSTANT-ONLY control: predict the standardised mean (0).
                # In frozen space it MUST read ~ the target's per-channel
                # variance (~1.0) — a known value, the CLAUDE.md probe rule.
                mse_z = (tgt ** 2).mean(dim=(-1, -2))[0]
                # tripwire: the per-step curve must average to forward()'s own loss
                lf = float(out["loss_feat_op"])
                if abs(float(mse_m.mean()) - lf) > 1e-3 * max(1.0, abs(lf)):
                    raise RuntimeError(f"wm_mse_model.mean {float(mse_m.mean())} "
                                       f"!= loss_feat_op {lf}: the target "
                                       f"expression drifted from forward()")
                field = model.encode(feats)
                pooled_win = field.mean(dim=-2)
                heads = {}
                for cname, nv in (("nav_true", nav_t), ("nav_shuffled", nav_s),
                                  ("nav_zero", zeros)):
                    br = model._run_brains(pooled_win, nv)
                    if br is None:
                        heads = None
                        break
                    heads[cname] = (int(model.lat_head(br["intent"]).argmax(-1)),
                                    int(model.lon_head(br["intent"]).argmax(-1)),
                                    int(br["route_logits"].argmax(-1)))
                if heads is not None and "lat_logits" in out:
                    fwd = (int(out["lat_logits"].argmax(-1)),
                           int(out["lon_logits"].argmax(-1)),
                           int(out["route_logits"].argmax(-1)))
                    if fwd != heads["nav_true"]:
                        raise RuntimeError("brains-only decode != forward() "
                                           "decode under the same nav — drift")
                # -- the planning arms ----------------------------------------
                goal_oracle = None
                if "cl_oraclegoal" in arms or "cl_oracleseed" in arms:
                    hp = cfg.plan_steps
                    goal_oracle = model.adapter(model.std(fut[:, :hp]))[:, hp - 1]
                plans = {}
                for arm in plan_arms:
                    nv = {"cl": nav_t, "cl_navshuf": nav_s, "cl_nonav": None,
                          "cl_oraclegoal": nav_t, "cl_oracleseed": nav_t}[arm]
                    gf = (goal_oracle
                          if arm in ("cl_oraclegoal", "cl_oracleseed") else None)
                    # ⭐ the ONE difference between `cl_oracleseed` and `cl`:
                    # the goal field. The seed stays (goal_keeps_seed=True).
                    gks = arm == "cl_oracleseed"
                    # ⛔ T0 WHEN IT IS `gt`: the hint is the window's own true
                    # curvature, computed from the SAME `g` the dump banks and
                    # by the SAME recipe as `tools/gt_kappa.py`
                    # (yaw_rate.mean / speed.mean.clamp_min(0.5)).
                    gkh = None
                    if gk_levels is not None:
                        _G = _ff._seq_geometry(g.float().cpu(), DT)
                        gkh = float(_G["yaw_rate"].mean(1)
                                    / _G["speed"].mean(1).clamp_min(0.5))
                    tp = time.time()
                    # ⭐⭐ THE THREE LONGITUDINAL LEVERS, all OFF by default
                    # and all fed from `ext` -- the SAME measured (a0, kappa0)
                    # `ha0_ext` holds, i.e. a backward difference of past
                    # speeds closing at t0 with NOTHING from the future
                    # (`hold_ext_controls`). Admissible at T1 under the PI
                    # ruling of 2026-09-02 on the measured state at cycle time;
                    # strictly LESS information than `ha`, which holds the last
                    # observed ACTION.
                    _a0 = float(ext[0])
                    _lm = getattr(a, "a_sustain_mode", "none")
                    a_sustain = _a0 if _lm == "a0" else None
                    a_shift = _a0 if _lm == "a0_shift" else None
                    jerk_seam = (_a0 if getattr(a, "jerk_seam", "off")
                                 == "a0" else None)
                    # ⭐ the planner->model crossing travels with the call:
                    # the candidate stays curvature, the MODEL is fed
                    # arctan(L_enc*kappa) when `rec_units == "steer"`.
                    res = model.plan(feats, v0=v0, nav_cmd=nv, plan_cfg=pc,
                                     goal_field=gf,
                                     model_action_units=rec_units,
                                     cost_metric=cost_metric,
                                     cost_weights=cost_weights,
                                     lat_logit_bias=lat_logit_bias,
                                     goal_kappa_turn=goal_kappa_turn,
                                     seed_kappa_ladder=seed_kappa_ladder,
                                     goal_kappa_levels=gk_levels,
                                     goal_kappa_hint=gkh,
                                     a_sustain=a_sustain,
                                     a_shift=a_shift,
                                     jerk_seam_a0=jerk_seam,
                                     goal_keeps_seed=gks)
                    if t_plan_first is None:
                        t_plan_first = time.time() - tp
                        # ⛔ the flag must have REACHED plan(): a result that
                        # does not carry it back was produced by an older
                        # plan() and would be banked under the wrong name.
                        if getattr(res, "cost_metric", None) != cost_metric:
                            raise RuntimeError(
                                f"plan() returned cost_metric="
                                f"{getattr(res, 'cost_metric', None)!r}, asked "
                                f"for {cost_metric!r}: the flag did not reach it")
                        got_w = tuple(float(x) for x in
                                      getattr(res, "cost_weights", ()))
                        want_w = tuple(used_w.values())
                        if got_w != want_w:
                            raise RuntimeError(f"plan() used cost_weights={got_w}, "
                                               f"asked for {want_w}")
                        # ⛔ and the DECISION RULE must have reached plan() too:
                        # a result that does not carry it back was produced by
                        # an older plan() and would be banked under the wrong
                        # arm name — the same trap as the cost flags above.
                        # ⛔ AND THE THREE LONGITUDINAL LEVERS MUST HAVE
                        # REACHED plan() TOO. Same trap as the cost flags
                        # above: a stack that silently ignores a kwarg would
                        # bank a SHIPPED-vocabulary arm under a lever's name,
                        # which is the one failure this whole package exists to
                        # avoid. `w_vend_armed` is checked explicitly because
                        # its absence is exactly what made the third weight a
                        # dead term unnoticed for the whole programme.
                        if getattr(a, "a_sustain_mode", "none") != "none":
                            got_s = getattr(
                                res, "a_shift" if _lm == "a0_shift"
                                else "a_sustain", "__absent__")
                            if got_s in ("__absent__", None):
                                raise RuntimeError(
                                    "plan() returned a_sustain=%r for "
                                    "--a-sustain-mode %s: the flag did not "
                                    "reach it (stale stack)"
                                    % (got_s, getattr(a, "a_sustain_mode",
                                                      "none")))
                        if getattr(a, "jerk_seam", "off") != "off":
                            got_j = getattr(res, "jerk_seam_a0", "__absent__")
                            if got_j in ("__absent__", None):
                                raise RuntimeError(
                                    "plan() returned jerk_seam_a0=%r for "
                                    "--jerk-seam %s: the flag did not reach it"
                                    % (got_j, getattr(a, "jerk_seam",
                                                      "off")))
                        # ⛔ W_VEND MUST STAY DEAD IN THIS TOOL. `target_speed`
                        # is not passed by design: `tests/test_steer_conversion
                        # _complete.py::test_C1_no_production_plan_call_site_
                        # passes_target_speed` pins the CALL SITE, and its own
                        # docstring reserves arming T4 as a PI decision. So the
                        # third entry of every `--cost-weights` triple this tool
                        # has ever banked contributed EXACTLY ZERO cost -- which
                        # also means the 643x W_VEND difference between the
                        # shipped and A/B triples is a difference in a number
                        # that is never read, not a confound. Asserted here so a
                        # future edit that arms it cannot pass unnoticed.
                        if getattr(res, "w_vend_armed", None) is True:
                            raise RuntimeError(
                                "plan() reports w_vend_armed=True: this tool "
                                "must never arm T4 (test_C1 pins the call "
                                "site; arming it is a PI decision)")
                        got_b = getattr(res, "lat_logit_bias", "__absent__")
                        want_b = (None if lat_logit_bias is None else
                                  [float(x) for x in lat_logit_bias])
                        if got_b == "__absent__" or got_b != want_b:
                            raise RuntimeError(
                                f"plan() returned lat_logit_bias={got_b!r}, "
                                f"asked for {want_b!r}: the decision rule did "
                                f"not reach it")
                    plans[arm] = res
            # -- bank the window ---------------------------------------------
            acc["g"].append(g.float().cpu().numpy())
            acc["ol"].append(ol.float().cpu().numpy())
            acc["ha"].append(ha.float().cpu().numpy())
            acc["ha0"].append(ha0.float().cpu().numpy())
            acc["ha0_ext"].append(ha0_ext.float().cpu().numpy())
            acc["v0"].append(np.array([v0], dtype=np.float32))
            for arm, res in plans.items():
                ctrl = res.controls.detach()
                acc[arm].append(paths_from_controls(ctrl, v0, DT, k)
                                .float().cpu().numpy())
                dec.setdefault(f"{arm}_controls", []).append(
                    ctrl[:, :2].float().cpu().numpy()[None])
                dec.setdefault(f"plan_source_{arm}", []).append(_source_code(res.source))
                dec.setdefault(f"plan_cost_{arm}", []).append(float(res.cost))
                dec.setdefault(f"plan_agree_{arm}", []).append(
                    -1 if res.coarse_fine_agree is None else int(res.coarse_fine_agree))
                dec.setdefault(f"plan_neval_{arm}", []).append(int(res.n_evaluated))
                # ⭐ THE INJECTED BASELINES' OWN COSTS and the coarse->fine re-score,
                # per window (D-REFAV1-CCOS-EVAL). Under `ccos` the cv row's cost IS
                # its goal term, and whether that reads the no-information 1.0 or an
                # arbitrary value decided by batch-composition rounding is a property
                # of the run that must be banked, not assumed (test_m of
                # test_cost_ccos.py read 1.0531 on a tiny model).
                for nm_, cv_ in (res.baseline_costs or {}).items():
                    dec.setdefault(f"basecost_{nm_}_{arm}", []).append(float(cv_))
                for nm_, cv_ in (res.fine_costs or {}).items():
                    dec.setdefault(f"finecost_{nm_}_{arm}", []).append(float(cv_))
                # GOAL PROVENANCE, read off the result (refa_v1.plan stamps
                # goal_source / goal_space / goal_action since e609a98). An
                # older model file carries no attribute and reads as "none" with
                # lat/lon -1 — recorded, never inferred. goal_action is the
                # SELECTED manoeuvre the TACTICAL family names (decoded by the
                # factored heads under this arm's nav), banked per window.
                gs = getattr(res, "goal_source", None) or "none"
                # ⛔ VERIFY-GATE, PER WINDOW: a stale stack whose `plan()`
                # predates `goal_keeps_seed` SILENTLY IGNORES the kwarg (it is
                # keyword-only with a default) and produces the CONFOUNDED
                # seed-less arm under the de-confounded arm's name — which is
                # the exact error this arm exists to correct. The stamp is the
                # only positive evidence the branch ran.
                if arm == "cl_oracleseed" and str(gs) != "supplied+seed":
                    raise RuntimeError(
                        f"[refav1_arm] ⛔ cl_oracleseed got goal_source={gs!r}, "
                        f"expected 'supplied+seed': the seed did NOT enter the "
                        f"pool and this arm is the CONFOUNDED one. Stale stack?")
                if getattr(res, "goal_kappa_vocab", None) != gk_vocab:
                    raise RuntimeError(
                        f"[refav1_arm] ⛔ plan() reports vocabulary "
                        f"{getattr(res, 'goal_kappa_vocab', None)!r}, asked for "
                        f"{gk_vocab!r}: the level set did not reach it and this "
                        f"arm would be banked under the wrong action space")
                if arm == "cl" and str(gs) != "tactical_imagined":
                    raise RuntimeError(
                        f"[refav1_arm] ⛔ control arm cl got goal_source={gs!r}: "
                        f"the paired control is not the shipped pipeline")
                ga = getattr(res, "goal_action", None)
                dec.setdefault(f"goal_source_{arm}", []).append(_goal_code(str(gs)))
                dec.setdefault(f"goal_lat_{arm}", []).append(
                    lat_names.index(ga["lat"]) if isinstance(ga, dict)
                    and ga.get("lat") in lat_names else -1)
                dec.setdefault(f"goal_lon_{arm}", []).append(
                    lon_names.index(ga["lon"]) if isinstance(ga, dict)
                    and ga.get("lon") in lon_names else -1)
                gsp = getattr(res, "goal_space", None)
                if gsp is not None:
                    goal_space_seen.add(str(gsp))
            dec.setdefault("ha_controls", []).append(
                hold[None].expand(k, 2).float().cpu().numpy()[None])
            dec.setdefault("ha0_ext_controls", []).append(
                ext[None].expand(k, 2).float().cpu().numpy()[None])
            dec.setdefault("wm_mse_model", []).append(mse_m.float().cpu().numpy()[None])
            dec.setdefault("wm_mse_const", []).append(mse_c.float().cpu().numpy()[None])
            dec.setdefault("wm_mse_zero", []).append(mse_z.float().cpu().numpy()[None])
            dec.setdefault("wm_tgt_std", []).append(float(out.get("tgt_std_op", float("nan"))))
            for key in ("lat_label", "lon_label", "route_label"):
                dec.setdefault(key, []).append(
                    int(b[key][0]) if b.get(key) is not None else -100)
            dec.setdefault("nav_cmd", []).append(int(nav_true[i]))
            dec.setdefault("nav_cmd_shuf", []).append(int(nav_shuf[i]))
            dec.setdefault("nav_valid", []).append(bool(nav_valid[i]))
            for cname in ("nav_true", "nav_shuffled", "nav_zero"):
                for j, hk in enumerate(("lat", "lon", "route")):
                    dec.setdefault(f"{hk}_pred_{cname}", []).append(
                        -1 if heads is None else int(heads[cname][j]))
            ws.append(int(t))
            n_done += 1
            if n_done == 1 and t_plan_first is not None:
                est = t_plan_first * len(plan_arms) * len(sel)
                _p(f"[cost] first plan() took {t_plan_first:.2f} s -> ESTIMATED "
                   f"{est / 3600:.2f} h for {len(sel)} windows x {len(plan_arms)} "
                   f"planning arms (plan only; forward()/IO extra)")
        np.savez_compressed(
            os.path.join(a.dump_dir, f"ep{fi:03d}.npz"),
            **{kk: np.concatenate(v).astype(np.float32) for kk, v in acc.items()},
            ws=np.array(ws), eid=np.array([fi]), clip_index=np.array([ei]))
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
        np.savez_compressed(os.path.join(a.dump_dir, "decisions", f"ep{fi:03d}.npz"),
                            ws=np.array(ws), **dec_np)
        episodes_manifest.append({"file_index": fi, "episode_index": ei,
                                  "name": nm, "clip_id": ld.clip_id.get(nm),
                                  "n_windows": len(ws)})
        _p(f"  [{fi + 1}/{len(by_ep)}] {nm[:12]} {len(ws)} windows "
           f"{time.time() - t_start:.0f}s")

    cache_index = None
    idx_path = os.path.join(a.cache, "index.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as fh:
                ci_ = json.load(fh)
            cache_index = {k: v for k, v in ci_.items() if k != "episodes"}
            cache_index["n_episodes_in_index"] = len(ci_.get("episodes", []) or [])
        except Exception as ex:                                 # noqa: BLE001
            cache_index = {"unreadable": f"{type(ex).__name__}: {ex}"}
    manifest = {
        "tool": "taniteval/tools/refav1_arm.py",
        "model": prov,
        "grid": {"dt_s": DT, "horizon_k": k, "wm_k": k_wm, "window": ld.W,
                 "window_stride": (stride if wlist_meta is None
                                   else "IGNORED (--window-list)"),
                 "window_list": wlist_meta, "n_windows": n_done,
                 "n_episodes": len(episodes_manifest),
                 "n_episodes_available": len(episode_names(a.cache))},
        "arms": arms, "tiers": {x: ARM_TIERS[x] for x in arms},
        "arm_meaning": {x: ARM_MEANING[x] for x in arms},
        "plan_cfg": dataclasses.asdict(pc),
        # ⭐ THE COST PROVENANCE (D-REFAV1-CCOS-EVAL): a dump produced under
        # `ccos` and one produced under `cos` are otherwise indistinguishable
        # after the fact. `weights` is the triple plan() ACTUALLY used.
        # ⭐ THE DECISION-RULE PROVENANCE (D-REFAV1-DRIVE-GATE): two dumps that
        # differ ONLY in the goal head's decision rule are otherwise
        # indistinguishable after the fact, and that rule IS refav1's lateral
        # policy. `null` means plain argmax.
        "goal_rule": {
            "lat_logit_bias": (None if lat_logit_bias is None
                               else [float(x) for x in lat_logit_bias]),
            "source": ("CLI override (--lat-logit-bias)" if lat_logit_bias
                       is not None else "plain argmax (legacy path)"),
            "kamm_mu": getattr(a, "kamm_mu", None),
            # the longitudinal levers, so a dump says which vocabulary and
            # which cost geometry produced it
            "a_sustain_mode": getattr(a, "a_sustain_mode", "none"),
            "jerk_seam": getattr(a, "jerk_seam", "off"),
            # W_VEND is a DEAD TERM in this tool by design (test_C1 pins the
            # call site; arming it is a PI decision). Recorded so a reader of a
            # banked cost triple knows its third entry never bound.
            "w_vend_armed": False,
            "kamm_note": (
                "speed-dependent curvature cap |kappa| <= mu*g/v^2 inside "
                "_clip; None = the shipped CONSTANT kappa_max"),
            "seed_kappa_ladder": (list(seed_kappa_ladder)
                                  if seed_kappa_ladder else None),
            "seed_kappa_ladder_note": (
                "extra SUSTAINED-curvature candidates injected into iCEM's "
                "iteration-0 pool, both signs, on the decoded goal seed's own "
                "accel profile. NOT a vocabulary change: canonical_controls "
                "and the goal field are untouched, so this IS comparable "
                "across arms that share a vocabulary"),
            "goal_kappa_turn": goal_kappa_turn,
            "goal_kappa_turn_source": ("CLI override (--goal-kappa-turn)"
                                       if goal_kappa_turn is not None
                                       else "shipped GOAL_KAPPA_TURN"),
            "site": "refa_v1._imagine_tactical_goal -> lat_head(intent).argmax",
            "why_it_matters": ("the decoded token's canonical profile is the "
                               "planner's ONLY curvature-carrying candidate; a "
                               "LANE_KEEP decode forces curvature EXACTLY 0 "
                               "(MEASURED 244/244 windows, step 21109)")},
        # ⚠️ THE PARITY STAMP (M16 (1) / PREREG_D-VOCAB-L3 §4). Changing the
        # sustained-curvature level set changes the ACTION SPACE; every refav1
        # number banked before 2026-09-05 is under `L1-0.08`, and a
        # cross-vocabulary comparison is INADMISSIBLE unless it says so. The
        # stamp is read back off `plan()` per window, not asserted here.
        "goal_vocab": {
            "kappa_vocab": gk_vocab,
            "kappa_levels": (None if gk_levels is None else list(gk_levels)),
            "kappa_level_chooser": getattr(a, "goal_kappa_hint", None),
            "tier_note": ("goal_kappa_hint='gt' reads the window's TRUE "
                          "curvature => the arm is T0, a vocabulary-adequacy "
                          "BOUND and never a driving number (EVAL_DOCTRINE)"),
            "shipped": "L1-0.08 (GOAL_KAPPA_TURN = 0.08, R 12.5 m)"},
        "cost": {"metric": cost_metric, "weights": used_w,
                 "weights_source": ("CLI override (--cost-weights)" if cost_weights
                                    else "shipped module constants"),
                 "shipped_weights": shipped_w,
                 "cost_metrics_available": list(_R.COST_METRICS),
                 "rule": ("cost_metric selects the goal term's form "
                          "(refa_v1.COST_METRICS: cos = 1-cos(z,g), the shipped "
                          "default; chord = ||z^-g^||, monotone-equivalent; ccos = "
                          "1-cos(z-z_ref, g-z_ref) centred on the window's own "
                          "zero-action terminal field). Neither non-default form "
                          "is weight-neutral, so the weight triple is banked "
                          "beside the metric, never assumed")},
        "plan_source_names": list(PLAN_SOURCE_NAMES),
        "nav_shuffle": shuf_stats,
        "speed_channel": {
            "enabled": bool(getattr(cfg, "speed_channel", False)),
            "rule": ("derived by RefAV1.augment_actions in forward() AND plan(): "
                     "v_k = v0 + sum_{j<k} a_j dt, scaled by SPEED_SCALE_MPS; "
                     "this adapter passes 2-wide (a, kappa) + measured v0 and "
                     "never widens actions (the model refuses a 3-wide input)")},
        "hold_action_rule": hold_action_controls.__doc__,
        "hold_v0_rule": hold_v0_controls.__doc__,
        # ⭐ THE UNIT PROVENANCE. Without this a repaired and an unrepaired dump
        # are indistinguishable after the fact -- which is exactly how a 2.9x
        # over-rotation survived a "r = 0.995" channel check.
        "action_units": {
            "recorded": rec_units,
            "planner_search": "kappa",
            "planner_to_model": rec_units,
            "L_enc_m": STEER_WHEELBASE_M,
            "rule": ("v2ep actions[:,0] is a road-wheel angle "
                     "(physicalai.signals_at: steer = arctan(L_enc*curvature), "
                     "L_enc MEASURED = 2.9 exactly). 'kappa' = the LEGACY "
                     "reading (integrate it unconverted, hand it to the model "
                     "unconverted); 'steer' = the repaired contract "
                     "(kappa = tan(steer)/L_enc before any integration, "
                     "steer = arctan(L_enc*kappa) at the model boundary). "
                     "The two are NOT comparable.")},
        "goal": {"source_names": list(GOAL_SOURCE_NAMES),
                 "space": sorted(goal_space_seen) or None,
                 "rule": ("per window from PlanResult.goal_source / goal_space / "
                          "goal_action (refa_v1.plan, e609a98): goal_source_<arm> "
                          "indexes source_names, goal_{lat,lon}_<arm> index the "
                          "tac vocabulary (-1 = no decoded action). An absent "
                          "attribute reads 'none' — never inferred.")},
        "corpus": {"cache": a.cache, "episodes": a.episodes,
                   "labels": a.labels, "nav": a.nav,
                   "cache_index": cache_index,
                   "join_report": ld.join_report},
        "episodes": episodes_manifest,
        "wallclock_s": round(time.time() - t_start, 1),
        "first_plan_s": t_plan_first,
    }
    with open(os.path.join(a.dump_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    _p("REFAV1_DUMP_DONE")
    return manifest


# --------------------------------------------------------------------------- #
# the lead block join (backlog R1) — distance-keeping on refav1's grid          #
#                                                                               #
# A refav1 window is (episode, cache step t) with origin RAW v2ep frame 2t and   #
# a 0.2 s horizon grid. The banked B1 block (tools/build_lead_block_b1.py) has   #
# ONE ROW PER (clip_id, RAW frame), so the join is by KEY, never by position —  #
# the positional blocks of rollout.collect's grid are refused by name.          #
# NO_LABEL (no obstacle.offline / horizon outside the labelled span / no row /  #
# refused episode) is NEVER free flow: it is counted, reported, and kept out of  #
# the denominator. Only LEAD rows carry a lead; the metric itself is            #
# four_families._distance_keeping through t1_eval.analyze's own `lead=` path.   #
# --------------------------------------------------------------------------- #
LEAD_EP_OK, LEAD_EP_NO_CLIP, LEAD_EP_NO_ROWS, LEAD_EP_SPEED = (
    "OK", "NO_CLIP_ID", "NO_ROWS", "SPEED_MISMATCH")
_LEAD_STATES = ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL")
_DK_KEYS = ("headway_min_m", "time_gap_min_s", "min_ttc_s")


def _load_eff():
    """Import the sibling ``eval_four_families.py`` by file for
    ``load_lead_block`` — the programme's ONE two-container (.npz/.pt) reader."""
    spec = importlib.util.spec_from_file_location(
        "eval_four_families_for_refav1", os.path.join(_HERE, "eval_four_families.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preflight_analysis_imports(lead_used: bool) -> dict:
    """⛔ EVERY MODULE ``analyze_refav1`` NEEDS, RESOLVED IN ~2 s BEFORE THE
    ROLLOUT — because the alternative has now happened, at full price.

    MEASURED 2026-09-04: the Thor rollout completed **all 141 episodes**
    (10,898 s = 3.0 h of GPU, ``REFAV1_DUMP_DONE`` printed) and then
    ``analyze()`` died on
    ``FileNotFoundError: .../taniteval/tools/eval_four_families.py`` — a sibling
    absent from the shipped tree. The traceback reads like a total failure; it
    was a **100 %-complete run missing its last step**, recovered later with
    ``--analyze-only`` at zero GPU.

    ``_bootstrap_paths()`` already preflights ``taniteval.ci`` and
    ``taniteval.four_families``. It does **not** preflight the siblings this
    module loads **BY FILE PATH** — and the file-loaded one is exactly what
    failed. A preflight that covers only the imports written as ``import`` is
    the same scope error as reading ``df`` on a pod: it answers a neighbouring
    question and is read as an answer.

    ⇒ This probe EXECUTES every module the analysis path will need, including
    the file-loaded siblings, and it must stay AHEAD of ``run_dump``. It returns
    what it checked so a green preflight is visible rather than silent.
    """
    checked, warned = [], []
    # (a) siblings loaded BY FILE PATH — the class that actually failed
    sibs = [("t1_eval.py", True), ("eval_four_families.py", bool(lead_used))]
    for fname, fatal in sibs:
        path = os.path.join(_HERE, fname)
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            spec = importlib.util.spec_from_file_location(
                f"_preflight_{fname[:-3]}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            checked.append(f"file:{fname}")
        except Exception as ex:                                # noqa: BLE001
            msg = (f"[refav1_arm] ⛔ PREFLIGHT FAILED on the file-loaded sibling "
                   f"{path}: {type(ex).__name__}: {ex}. "
                   f"    The analysis step needs it AFTER the rollout, so "
                   f"running would spend the whole GPU budget and then die "
                   f"with nothing but the dump. Ship the file (or pass "
                   f"--no-lead-block if it is only needed for the lead block) "
                   f"and re-launch.")
            if fatal:
                sys.exit(msg)
            warned.append(f"file:{fname} ({type(ex).__name__})")
    # (b) modules imported lazily INSIDE the analysis functions
    lazy = ["taniteval.ci", "taniteval.four_families", "taniteval.lead_metrics",
            "tanitad.refs.refc_tactical", "tanitad.refs.refb",
            "tanitad.models.v6", "tanitad.models.metric_dynamics",
            "tanitad.refs.refa_v1_plan", "tanitad.data.refav1_loader"]
    import importlib as _il
    for name in lazy:
        try:
            _il.import_module(name)
            checked.append(name)
        except Exception as ex:                                # noqa: BLE001
            sys.exit(f"[refav1_arm] ⛔ PREFLIGHT FAILED importing {name}: "
                     f"{type(ex).__name__}: {ex}. "
                     f"    analyze_refav1() imports it lazily, i.e. AFTER the "
                     f"rollout has been paid for. Refusing to start.")
    rep = {"n_checked": len(checked), "checked": checked, "warned": warned,
           "lead_block_in_play": bool(lead_used)}
    _p(f"[preflight] {len(checked)} analysis imports OK"
       + (f"  ⚠️ non-fatal: {warned}" if warned else "")
       + "  (this probe exists because a 3.0 h rollout once died in analyze())")
    return rep


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_lead_block_rows(path: str):
    """The per-frame B1 block -> ``(block, {(clip_id, frame): row}, meta)``.

    Read by ``eval_four_families.load_lead_block`` (npz or pt; plain dict; the
    required keys checked there). A block without ``clip_id``/``frame`` is a
    POSITIONAL block (rollout.collect's window grid) and is REFUSED: joining it
    by position would score every refav1 window against another clip's traffic
    and the metric would still return a plausible number.
    """
    blk = _load_eff().load_lead_block(path)
    for key in ("clip_id", "frame", "ts_rel_s"):
        if key not in blk:
            raise SystemExit(
                f"[refav1_arm] lead block {path} carries no {key!r}: not a per-frame "
                f"B1 block (tools/build_lead_block_b1.py). A positional block cannot "
                f"be joined to refav1's (clip, frame) grid — refusing.")
    cid = np.asarray(blk["clip_id"]).astype(str).reshape(-1)
    fr = np.asarray(blk["frame"]).astype(np.int64).reshape(-1)
    n = int(np.asarray(blk["leads"]).shape[0])
    if cid.shape[0] != n or fr.shape[0] != n:
        raise SystemExit(f"[refav1_arm] lead block {path}: clip_id {cid.shape[0]} / "
                         f"frame {fr.shape[0]} rows for {n} lead rows — corrupt")
    idx: dict = {}
    for i, (c, f) in enumerate(zip(cid.tolist(), fr.tolist())):
        if (c, f) in idx:
            raise SystemExit(f"[refav1_arm] lead block {path}: duplicate row for "
                             f"({c}, frame {f}) — the join would be ambiguous")
        idx[(c, f)] = i
    meta = {}
    if "meta_json" in blk:
        try:
            meta = json.loads(np.asarray(blk["meta_json"], dtype=np.uint8)
                              .tobytes().decode("utf-8"))
        except Exception as ex:                                 # noqa: BLE001
            meta = {"unreadable": f"{type(ex).__name__}: {ex}"}
    return blk, idx, meta


def join_lead_block(files, manifest, blk, idx, *, k: int, dt: float,
                    speed_tol: float = LEAD_SPEED_TOL_MPS,
                    frame_of_t=None) -> dict:
    """Lead rows for the dump's windows, IN DUMP ORDER, keyed by ``(clip_id, 2t)``.

    Per episode file ``ep{fi:03d}.npz``: the clip comes from the manifest's
    ``episodes[file_index].clip_id`` (the roll banked it from the loader), the
    frame from ``ws`` (t -> RAW frame 2t). Three guards, each a loud refusal:
      * the block's ``ts_rel_s[:k]`` must equal ``dt * (1..k)`` — a lead track at
        other instants is a time join at the wrong instants, not a shape problem;
      * the dump's ``v0`` (poses[2t, 3]) must equal the block's ``speeds`` at the
        joined rows to ``speed_tol`` — the LABEL-FREE alignment proof (both are
        egomotion interpolated at the same instant): a mismatch is a wrong clip
        or a wrong frame, and the episode is refused rather than scored;
      * a window without a row stays NO_LABEL and is counted (``n_windows_no_row``).
    Returns the ``lead=`` dict ``t1_eval.analyze`` consumes plus ``coverage`` and
    the block's GT reference arrays (``_gt``) on the joined windows.
    """
    ts = np.asarray(blk["ts_rel_s"], dtype=np.float64).reshape(-1)
    want = np.arange(1, int(k) + 1, dtype=np.float64) * float(dt)
    if ts.size < k or float(np.max(np.abs(ts[:k] - want))) > LEAD_TS_TOL_S:
        raise SystemExit(
            f"[refav1_arm] lead block horizon grid {np.round(ts[:k], 4).tolist()} != "
            f"the dump's {np.round(want, 4).tolist()} (dt {dt} s, K {k}) — a lead "
            f"track at other instants must not be truncated or resampled onto this "
            f"path; rebuild the block on this grid (build_lead_block_b1 --dt/--k).")
    eps = (manifest or {}).get("episodes") or []
    by_fi = {int(e.get("file_index", i)): e for i, e in enumerate(eps)}
    b_leads = np.asarray(blk["leads"], dtype=np.float64)
    b_lens = np.asarray(blk["lead_lens"], dtype=np.float64).reshape(-1)
    b_speed = np.asarray(blk["speeds"], dtype=np.float64).reshape(-1)
    b_state = np.asarray(blk["state"]).astype(str).reshape(-1)
    b_gap = (np.asarray(blk["gap0_m"], dtype=np.float64).reshape(-1)
             if "gap0_m" in blk else np.full(b_lens.shape, np.nan))
    b_gt = {kk: (np.asarray(blk[f"gt_{kk}"], dtype=np.float64).reshape(-1)
                 if f"gt_{kk}" in blk else None) for kk in _DK_KEYS}

    L, LN, SP, ST, G0, EID = [], [], [], [], [], []
    GT = {kk: [] for kk in _DK_KEYS}
    cov_eps: dict = {}
    n_no_row = 0
    speed_max = 0.0
    for fi, f in enumerate(files):
        eid = os.path.splitext(os.path.basename(f))[0]
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(int).reshape(-1)
            v0 = (np.asarray(d["v0"], dtype=np.float64).reshape(-1)
                  if "v0" in d.files else None)
        n = int(ws.size)
        ent = by_fi.get(fi) or {}
        clip = ent.get("clip_id")
        leads = np.full((n, k, 2), np.nan)
        lens = np.full(n, np.nan)
        speeds = v0.copy() if v0 is not None and v0.size == n else np.full(n, np.nan)
        state = np.array(["NO_LABEL"] * n, dtype=object)
        gap0 = np.full(n, np.nan)
        gt = {kk: np.full(n, np.nan) for kk in _DK_KEYS}
        cov = {"n_windows": n, "clip_id": clip, "episode_name": ent.get("name")}
        if not clip:
            cov.update(status=LEAD_EP_NO_CLIP,
                       reason="the dump manifest carries no clip_id for this episode "
                              "(v2ep without `clip_id`) — nothing to join on")
            n_no_row += n
        else:
            # frame_of_t: the provider-index -> RAW-frame map. DEFAULT 2*t
            # (refav1: 5 Hz provider over a 10 Hz block). A corpus whose
            # provider index IS the raw frame passes identity; without it
            # odd frames truncate to frame-1 and the speed proof refuses.
            _f = frame_of_t if frame_of_t is not None else (lambda t: 2 * t)
            rows = np.array([idx.get((str(clip), int(_f(int(t)))), -1)
                             for t in ws], dtype=np.int64)
            have = rows >= 0
            cov["n_windows_with_row"] = int(have.sum())
            if not have.any():
                cov.update(status=LEAD_EP_NO_ROWS,
                           reason=f"clip {clip} has no rows in the block "
                                  f"(frames {int(_f(int(ws.min())))}.."
                                  f"{int(_f(int(ws.max())))})")
                n_no_row += n
            else:
                r = rows[have]
                dmax = None
                if v0 is not None and v0.size == n:
                    dmax = float(np.max(np.abs(v0[have] - b_speed[r])))
                    cov["speed_check_max_mps"] = round(dmax, 6)
                    speed_max = max(speed_max, dmax)
                else:
                    cov["speed_check_note"] = ("dump carries no v0; the label-free "
                                               "alignment proof could not run")
                if dmax is not None and dmax > speed_tol:
                    cov.update(status=LEAD_EP_SPEED,
                               reason=f"max |dump v0 - block speed| = {dmax:.4f} m/s > "
                                      f"{speed_tol}: the (clip, frame) mapping is "
                                      f"wrong — refusing to place another clip's "
                                      f"traffic on these windows")
                    n_no_row += n
                else:
                    leads[have] = b_leads[r][:, :k]
                    lens[have] = b_lens[r]
                    if v0 is None or v0.size != n:
                        speeds[have] = b_speed[r]
                    state[have] = b_state[r]
                    gap0[have] = b_gap[r]
                    for kk in _DK_KEYS:
                        if b_gt[kk] is not None:
                            gt[kk][have] = b_gt[kk][r]
                    n_no_row += int((~have).sum())
                    cov["status"] = LEAD_EP_OK
                    if not have.all():
                        cov["n_windows_no_row"] = int((~have).sum())
        cov["counts"] = {s: int((state == s).sum()) for s in _LEAD_STATES}
        cov_eps[eid] = cov
        L.append(leads), LN.append(lens), SP.append(speeds), ST.append(state)
        G0.append(gap0), EID.extend([eid] * n)
        for kk in _DK_KEYS:
            GT[kk].append(gt[kk])
    state_all = np.concatenate(ST) if ST else np.array([], dtype=object)
    counts = {s: int((state_all == s).sum()) for s in _LEAD_STATES}
    return {
        "leads": np.concatenate(L) if L else np.zeros((0, k, 2)),
        "lead_lens": np.concatenate(LN) if LN else np.zeros(0),
        "speeds": np.concatenate(SP) if SP else np.zeros(0),
        "state": state_all, "gap0_m": np.concatenate(G0) if G0 else np.zeros(0),
        "eid": EID,
        "_gt": {kk: np.concatenate(v) if v else np.zeros(0) for kk, v in GT.items()},
        "coverage": {
            "n_windows": len(EID), "n_episodes": len(cov_eps),
            "n_episodes_ok": sum(1 for c in cov_eps.values()
                                 if c.get("status") == LEAD_EP_OK),
            "counts": counts,
            "n_windows_labelled": counts["LEAD"] + counts["NO_LEAD"],
            "n_windows_lead": counts["LEAD"],
            "n_windows_no_row": int(n_no_row),
            "speed_check": {"max_mps": round(speed_max, 6), "tol_mps": speed_tol,
                            "_is": "max |dump v0 - block speed at the joined row|: "
                                   "both are egomotion interpolated at frame 2t, so "
                                   "this is the label-free proof the join hit the "
                                   "right clip and frame"},
            "join_key": "(clip_id from the dump manifest, RAW frame 2t from ws)",
            "episodes": cov_eps,
            "note": ("NO_LABEL / no-row / refused-episode windows are counted "
                     "here and NEVER scored as free flow; NOT_STRAIGHT is its "
                     "own state (a bend hides the lead outside the corridor)"),
        },
    }


def attach_lead_block(files, manifest, path: str, *, k: int, dt: float):
    """-> ``(lead | None, info)``: the joined block for ``t1_eval.analyze(lead=)``
    and the provenance/coverage record. ``lead`` is ``None`` — the REFUSED
    branch, four_families then reports UNAVAILABLE as before — when the block
    covers no labelled window; passing an all-NaN block instead would let
    ``lead_metrics.distance_keeping`` call missing labels "free-flow"."""
    blk, idx, meta = load_lead_block_rows(path)
    lead = join_lead_block(files, manifest, blk, idx, k=k, dt=dt)
    cov = lead.pop("coverage")
    info = {"block": path, "block_sha256": _sha256(path),
            "block_version": meta.get("version"), "block_tool": meta.get("tool"),
            "block_built_utc": meta.get("built_utc"),
            "block_rows_all": meta.get("n_rows"), "block_clips": meta.get("n_clips"),
            "block_counts_all_rows": meta.get("counts"),
            "block_refusals": meta.get("refusals"),
            "horizon": {"k": int(k), "dt_s": float(dt)},
            "conventions": meta.get("conventions"), "states": meta.get("states"),
            "coverage": cov}
    n_lab = cov["counts"]["LEAD"] + cov["counts"]["NO_LEAD"]
    if n_lab == 0:
        info.update(_refused(
            f"the lead block covers 0 labelled windows of {cov['n_windows']} "
            f"(NO_LABEL {cov['counts']['NO_LABEL']}, NOT_STRAIGHT "
            f"{cov['counts']['NOT_STRAIGHT']}, no-row {cov['n_windows_no_row']}; "
            f"episodes OK {cov['n_episodes_ok']}/{cov['n_episodes']}) — nothing "
            f"is scoreable; NOT read as free flow", "n/a", 0))
        return None, info
    info["status"] = "PRESENT"
    info["n"] = int(cov["counts"]["LEAD"])
    return lead, info


def _boot(vals, eid, n_boot, seed):
    from taniteval import ci as _ci
    v = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"n": 0, "status": "NOT-APPLICABLE", "reason": "no finite values"}
    r = _ci.episode_cluster_bootstrap(v[ok], [e for e, m in zip(eid, ok) if m],
                                      reduce="mean", n_boot=n_boot, seed=seed)
    r["n"] = int(ok.sum())
    return r


def _distance_keeping_block(rec, info, lead, arms, P_cat, pairs, eid_w, dt,
                            n_boot, seed, tiers) -> dict:
    """``rec['refav1']['distance_keeping']``: coverage + provenance, the per-arm
    rows four_families already emitted (summarised, with their intervals), the
    GT reference on the same windows, the kinematic-contract check (``ol`` vs
    GT — a known value), and the PAIRED deltas per arm pair.

    The per-window values come from ``lead_metrics.distance_keeping`` on the
    same inputs four_families used (no path_steps: the grids match), and are
    asserted equal to four_families' own per-window arrays before use — one
    metric, two call sites, checked rather than assumed."""
    from taniteval import lead_metrics as lm
    out = dict(info or {})
    if info is None:
        return _refused("no lead block passed (--lead-block); the banked B1 EVAL "
                        f"block is {LEAD_BLOCK_DEFAULT}", "n/a", 0)
    if lead is None:
        return out
    out["tier"] = {a: tiers.get(a) for a in arms}
    out["estimator"] = ("per arm: lead_metrics.distance_keeping via four_families "
                        "(episode-cluster bootstrap in rec['arms'][arm]['four_"
                        "families']['longitudinal']); pairs: paired_episode_"
                        "cluster_bootstrap on jointly-finite windows")
    out["_binding"] = ("LONGITUDINAL distance-keeping: headway / time-gap / min-TTC "
                       "per arm with n and CI, per speed band (by_speed), never "
                       "pooled with speed accuracy; a censored TTC carries n_closing")
    pw = {}
    per_arm = {}
    for arm in arms:
        dk = lm.distance_keeping(P_cat[arm], lead["leads"], lead["lead_lens"],
                                 lead["speeds"], dt)
        pw[arm] = {kk: np.asarray(dk[kk], dtype=np.float64) for kk in _DK_KEYS}
        fam = rec["arms"][arm]["four_families"]["longitudinal"]
        dkf = fam.get("distance_keeping") or {}
        fpw = dkf.get("_per_window")
        if isinstance(fpw, dict):
            for kk in _DK_KEYS:
                a_, b_ = np.asarray(fpw[kk], np.float64), pw[arm][kk]
                same = (np.isfinite(a_) == np.isfinite(b_)).all() and np.allclose(
                    a_[np.isfinite(a_)], b_[np.isfinite(b_)], atol=1e-9)
                if not same:
                    raise RuntimeError(f"{arm}/{kk}: the adapter's per-window "
                                       f"distance-keeping != four_families' — drift")
            # numpy arrays do not belong in the JSON record; the same values are
            # used above and in the paired block.
            dkf["_per_window"] = ("stripped (numpy); identical values feed "
                                  "rec['refav1']['distance_keeping']['paired']")
        ci_all = ((fam.get("ci") or {}).get("components") or {})
        ci_na = ((fam.get("ci") or {}).get("unavailable") or {})
        bys = dkf.get("by_speed") or {}
        per_arm[arm] = {
            "tier": tiers.get(arm),
            "status": dkf.get("status"), "reason": dkf.get("reason"),
            "n": dkf.get("n"), "n_windows": dkf.get("n_windows"),
            "mean_headway_min_m": dkf.get("mean_headway_min_m"),
            "mean_time_gap_min_s": dkf.get("mean_time_gap_min_s"),
            "n_time_gap": dkf.get("n_time_gap"),
            "mean_min_ttc_s": dkf.get("mean_min_ttc_s"),
            "n_closing": dkf.get("n_closing"),
            "censoring_note": dkf.get("censoring_note"),
            "ci": {kk: (ci_all.get(f"distance_keeping.{kk}")
                        or ci_na.get(f"distance_keeping.{kk}"))
                   for kk in ("mean_headway_min_m", "mean_time_gap_min_s",
                              "mean_min_ttc_s")},
            "by_speed": {band: {q: b.get(q) for q in ("status", "n_windows",
                                                        "n_with_lead", "lead_rate")}
                         for band, b in (bys.get("strata") or {}).items()},
            "by_speed_window_states": bys.get("window_states_total"),
        }
    out["per_arm"] = per_arm
    # -- the GT reference (the block's own D-LEAD-1-style GT arm) on the SAME
    #    windows, and the kinematic-contract check: `ol` (recorded actions
    #    integrated from v0) must reproduce it — a KNOWN VALUE, like ol's ADE.
    gt = lead.get("_gt") or {}
    if gt:
        out["gt_reference"] = {
            "_is": ("lead_metrics.distance_keeping of the GT ego path over the "
                    "same horizon, banked in the block (gt_* columns): the value a "
                    "perfect kinematic reproduction scores on these windows"),
            **{kk: _boot(gt[kk], eid_w, n_boot, seed) for kk in _DK_KEYS}}
        if "ol" in pw and gt.get("headway_min_m") is not None:
            a_, b_ = pw["ol"]["headway_min_m"], np.asarray(gt["headway_min_m"])
            both = np.isfinite(a_) & np.isfinite(b_)
            dd = np.abs(a_[both] - b_[both]) if both.any() else np.zeros(0)
            out["kinematic_contract_check"] = {
                "n_both": int(both.sum()),
                "n_ol_only": int((np.isfinite(a_) & ~np.isfinite(b_)).sum()),
                "n_gt_only": int((np.isfinite(b_) & ~np.isfinite(a_)).sum()),
                "mean_abs_headway_diff_m": round(float(dd.mean()), 4) if dd.size else None,
                "median_abs_headway_diff_m": (round(float(np.median(dd)), 4)
                                              if dd.size else None),
                "p95_abs_headway_diff_m": (round(float(np.percentile(dd, 95)), 4)
                                           if dd.size else None),
                "max_abs_headway_diff_m": round(float(dd.max()), 4) if dd.size else None,
                "_reading": ("ol integrates the RECORDED (a, kappa) from v0 through "
                             "the programme's unicycle, so its headway must sit within "
                             "its own ADE of the GT reference (MEASURED on the real "
                             "20-clip slice: ol ADE 0.47 m over 2 s, mean |dHeadway| "
                             "0.14 m). The tail (max) is where ol's path drifts across "
                             "the corridor edge and the min-over-steps jumps; a LARGE "
                             "mean or median is a join or frame error, not a model "
                             "result")}
    out["paired"] = {nm: lm.paired_distance_keeping(pw[b], pw[a], eid_w, names=(b, a),
                                                    n_boot=n_boot, seed=seed)
                     for a, b, nm in pairs}
    return out


# --------------------------------------------------------------------------- #
# the analysis (CPU)                                                            #
# --------------------------------------------------------------------------- #
_GOAL_COND = {"cl": "nav_true", "cl_navshuf": "nav_shuffled", "cl_nonav": "nav_zero",
              "cl_oraclegoal": "nav_true", "cl_oracleseed": "nav_true"}


def _goal_provenance(dec, arm, N, names, space, lat_names, lon_names) -> dict:
    """The goal behind a planning arm's windows, from the sidecar: source
    fractions, the goal space, the SELECTED manoeuvre histogram (the decoded
    lat/lon tokens), and its agreement with the declared heads under the same
    nav (the imagined goal is decoded by the SAME heads — a rate < 1 means the
    goal came from a different decode path and must be explained)."""
    key = f"goal_source_{arm}"
    if key not in dec:
        return {"goal_source_fractions": None, "goal_space": None,
                "_goal_note": ("dump predates goal provenance (adapter < 2026-09-02) "
                               "— read as 'none': no evidence of a goal")}
    src = dec[key].astype(int)
    cnt = np.bincount(src, minlength=len(names))
    fr = {names[i]: round(float(c / N), 4) for i, c in enumerate(cnt) if c}
    lat = dec.get(f"goal_lat_{arm}", np.full(N, -1)).astype(int)
    lon = dec.get(f"goal_lon_{arm}", np.full(N, -1)).astype(int)
    n_act = int((lat >= 0).sum())
    out = {"goal_source_fractions": fr,
           "goal_space": (space[0] if isinstance(space, list) and len(space) == 1
                          else space),
           "n_goal_action": n_act,
           "goal_action_lat": {lat_names[i]: round(float(c / n_act), 4) for i, c in
                               enumerate(np.bincount(lat[lat >= 0], minlength=len(lat_names)))
                               if c} if n_act else None,
           "goal_action_lon": {lon_names[i]: round(float(c / n_act), 4) for i, c in
                               enumerate(np.bincount(lon[lon >= 0], minlength=len(lon_names)))
                               if c} if n_act else None}
    cond = _GOAL_COND.get(arm)
    if cond and f"lat_pred_{cond}" in dec and n_act:
        pl_, pn_ = dec[f"lat_pred_{cond}"].astype(int), dec[f"lon_pred_{cond}"].astype(int)
        m = (lat >= 0) & (pl_ >= 0)
        out["goal_vs_declared_head_agreement"] = {
            "conditioning": cond, "n": int(m.sum()),
            "lat": round(float((lat[m] == pl_[m]).mean()), 4) if m.any() else None,
            "lon": round(float((lon[m] == pn_[m]).mean()), 4) if m.any() else None,
            "_is": ("fraction of windows whose imagined-goal token equals the "
                    "declared head argmax under the same nav; 1.0 = one decode path")}
    return out


def _goal_source_protocol(plan: dict) -> str:
    """The `_protocol.goal_source` string, DERIVED from the observed per-arm
    fractions (never a constant): what goal each planning arm actually ran on."""
    parts = []
    for arm, blk in (plan or {}).items():
        fr = blk.get("goal_source_fractions")
        if not fr:
            parts.append(f"{arm}: none (no goal provenance in the dump)")
            continue
        top = ", ".join(f"{k} {100.0 * v:.1f} %" for k, v in
                        sorted(fr.items(), key=lambda kv: -kv[1]))
        sp = blk.get("goal_space")
        parts.append(f"{arm}: {top}" + (f" [space {sp}]" if sp else ""))
    if not parts:
        return ("none — plan() goal_field=None and no goal provenance recorded "
                "(the model has no hierarchy, or the dump predates e609a98)")
    return ("per arm — " + " | ".join(parts) + ". tactical_imagined = plan() "
            "imagined the goal from vision + nav + measured v0 (refa_v1.plan, "
            "e609a98); supplied = the TRUE future field (cl_oraclegoal, T0); "
            "none = goal-free cost (floor baseline by construction)")


def _load_decisions(dump_dir: str):
    files = sorted(glob.glob(os.path.join(dump_dir, "decisions", "ep*.npz")))
    if not files:
        return None, []
    cat: dict[str, list] = {}
    eid = []
    for f in files:
        with np.load(f) as d:
            n = int(d["ws"].shape[0])
            for kk in d.files:
                cat.setdefault(kk, []).append(d[kk])
        eid += [os.path.splitext(os.path.basename(f))[0]] * n
    return {kk: np.concatenate(v) for kk, v in cat.items()}, eid


def _components(P: np.ndarray, G: np.ndarray, dt: float) -> dict:
    """Per-window components with four_families' OWN geometry + the programme's
    canonical trajectory labeller — never a re-derivation."""
    import torch
    from taniteval import four_families as ff
    from tanitad.refs.refc_tactical import factor_from_kinematics
    pt, gt = torch.as_tensor(P).float(), torch.as_tensor(G).float()
    Pg, Gg = ff._seq_geometry(pt, dt), ff._seq_geometry(gt, dt)
    both = Pg["valid"] & Gg["valid"]
    dh = (Pg["heading"] - Gg["heading"] + math.pi) % (2 * math.pi) - math.pi
    nv = both.sum(1)
    head = torch.where(nv > 0, (dh.abs() * both).sum(1) / nv.clamp_min(1),
                       torch.nan).numpy() * 180.0 / math.pi
    dy_p, dv_p, v0p, v1p, _ = ff.maneuver_kinematics(pt, dt)
    dy_g, dv_g, v0g, v1g, _ = ff.maneuver_kinematics(gt, dt)
    lat_p, lon_p = factor_from_kinematics(dy_p, dv_p, v0p, v1p)
    lat_g, lon_g = factor_from_kinematics(dy_g, dv_g, v0g, v1g)
    return {
        "ade_m": torch.linalg.norm(pt - gt, dim=-1).mean(1).numpy(),
        "fde_m": torch.linalg.norm(pt[:, -1] - gt[:, -1], dim=-1).numpy(),
        "LON_speed_mae_mps": (Pg["speed"] - Gg["speed"]).abs().mean(1).numpy(),
        "LON_along_mae_m": (Pg["along"] - Gg["along"]).abs().mean(1).numpy(),
        "LON_accel_mae_mps2": (Pg["accel"] - Gg["accel"]).abs().mean(1).numpy(),
        "LAT_cross_mae_m": (Pg["cross"] - Gg["cross"]).abs().mean(1).numpy(),
        "LAT_heading_mae_deg": head,
        "LAT_yaw_rate_mae_radps": (Pg["yaw_rate"] - Gg["yaw_rate"]).abs().mean(1).numpy(),
        "TAC_traj_lat_correct": (lat_p == lat_g).float().numpy(),
        "TAC_traj_lon_correct": (lon_p == lon_g).float().numpy(),
    }


_FAMILY_OF = {"ade_m": "ADE", "fde_m": "ADE",
              "LON_speed_mae_mps": "longitudinal", "LON_along_mae_m": "longitudinal",
              "LON_accel_mae_mps2": "longitudinal",
              "LAT_cross_mae_m": "lateral", "LAT_heading_mae_deg": "lateral",
              "LAT_yaw_rate_mae_radps": "lateral",
              "TAC_traj_lat_correct": "tactical", "TAC_traj_lon_correct": "tactical"}


def _paired_families(comps: dict, arm_a: str, arm_b: str, eid, tiers, n_boot, seed):
    """b − a per family, paired episode-cluster bootstrap on the SAME windows."""
    from taniteval import ci as _ci
    out = {"direction": f"{arm_b} - {arm_a}",
           "tier": f"{tiers[arm_b]} minus {tiers[arm_a]}",
           "estimator": "paired_episode_cluster_bootstrap", "families": {}}
    for mk, fam in _FAMILY_OF.items():
        a_v, b_v = comps[arm_a][mk], comps[arm_b][mk]
        keep = np.isfinite(a_v) & np.isfinite(b_v)
        if keep.sum() == 0:
            out["families"].setdefault(fam, {})[mk] = _refused(
                "no window has a finite value for both arms", out["tier"])
            continue
        e = [x for x, kp in zip(eid, keep) if kp]
        r = _ci.paired_episode_cluster_bootstrap(b_v[keep], a_v[keep], e,
                                                 n_boot=n_boot, seed=seed)
        r["n_dropped_nonfinite"] = int((~keep).sum())
        out["families"].setdefault(fam, {})[mk] = r
    return out


# --------------------------------------------------------------------------- #
# THE TRIVIAL-PROFILE INSTRUMENT — printed BEFORE any family row                #
# --------------------------------------------------------------------------- #
#: a plan whose max |y| is below this is a STRAIGHT LINE (metres).
TRIVIAL_STRAIGHT_M = 1e-6
#: a plan whose chord-length spread is below this is CONSTANT SPEED (metres).
TRIVIAL_CONST_SPEED_M = 1e-4
#: two arms whose trajectories agree everywhere within this are IDENTICAL (metres).
TRIVIAL_IDENTICAL_M = 1e-9


def trivial_profile(files, arms, dt: float = DT) -> dict:
    """⛔ WHAT SHAPE IS EACH ARM'S TRAJECTORY, BEFORE ANY METRIC IS COMPUTED?

    Per arm, over every window: the fraction that are an exactly STRAIGHT line
    (``max |y| < 1e-6``), the fraction that are CONSTANT SPEED (chord-length spread
    < 1e-4 m), the fraction that are BOTH (the constant-velocity trivial profile), and
    which OTHER arms it is bit-identical to.

    ⭐ WHY IT PRINTS FIRST, AND WHY IT IS NOT OPTIONAL. MEASURED 2026-09-03: a full T1
    read shipped a paragraph reading *"lateral planning already beats holding"* —
    heading −2.0°, cross-track −18 cm, all separated — while **every one of the 140
    ``cl`` plans was a straight constant-speed line**. The "gain" was a straight line
    beating the hold-action control's noisy held kappa. `cl` was also bit-identical to
    `cl_navshuf` on 122/140 windows and to `cl_oraclegoal` on 96/140. Every family row
    was correctly computed and every one of them was read wrong, because nobody had
    asked what SHAPE the arm was. **A family table cannot answer that question; this
    instrument answers it in four lines and it costs one pass over the dump.**

    Returns per-arm rows plus ``degenerate_arms`` (constant-velocity on > 50 % of
    windows) — the flag that says *"read this arm's LATERAL rows as a control, not as
    skill"*.
    """
    prof = {a: {"n": 0, "n_straight": 0, "n_const_speed": 0, "n_trivial": 0}
            for a in arms}
    ident = {a: {b: 0 for b in arms if b != a} for a in arms}
    n_tot = 0
    for f in files:
        with np.load(f) as d:
            P = {a: d[a][..., :2].astype(np.float64) for a in arms if a in d.files}
        n = next(iter(P.values())).shape[0]
        n_tot += n
        for a, tr in P.items():
            steps = np.linalg.norm(
                np.diff(np.concatenate([np.zeros((n, 1, 2)), tr], axis=1), axis=1),
                axis=-1)
            st = np.abs(tr[..., 1]).max(axis=1) < TRIVIAL_STRAIGHT_M
            cs = np.ptp(steps, axis=1) < TRIVIAL_CONST_SPEED_M
            prof[a]["n"] += n
            prof[a]["n_straight"] += int(st.sum())
            prof[a]["n_const_speed"] += int(cs.sum())
            prof[a]["n_trivial"] += int((st & cs).sum())
        for a in P:
            for b in P:
                if a < b:
                    same = int((np.abs(P[a] - P[b]).reshape(n, -1).max(axis=1)
                                < TRIVIAL_IDENTICAL_M).sum())
                    ident[a][b] += same
                    ident[b][a] += same
    out = {"n_windows": n_tot, "rule": {
        "straight_m": TRIVIAL_STRAIGHT_M, "const_speed_m": TRIVIAL_CONST_SPEED_M,
        "identical_m": TRIVIAL_IDENTICAL_M,
        "trivial": "straight AND constant-speed = the constant-velocity profile"},
        "arms": {}}
    for a in arms:
        p = prof[a]
        if not p["n"]:
            continue
        out["arms"][a] = {
            "n": p["n"],
            "straight_frac": round(p["n_straight"] / p["n"], 4),
            "const_speed_frac": round(p["n_const_speed"] / p["n"], 4),
            "trivial_frac": round(p["n_trivial"] / p["n"], 4),
            "identical_to": {b: {"n": c, "frac": round(c / p["n"], 4)}
                             for b, c in sorted(ident[a].items()) if c},
        }
    out["degenerate_arms"] = sorted(
        a for a, r in out["arms"].items() if r["trivial_frac"] > 0.5)
    out["note"] = (
        "An arm with trivial_frac near 1.0 emits the CONSTANT-VELOCITY plan: its "
        "LATERAL rows are a property of the baseline, not of planning, and a win over "
        "`ha` (hold observed a, kappa) is NOT evidence of lateral skill — compare it "
        "to `ha0`. identical_to counts windows where two arms agree to "
        f"{TRIVIAL_IDENTICAL_M:g} m: an ablation identical to its own reference "
        "measured nothing on those windows.")
    return out


def _print_trivial_profile(tp: dict) -> None:
    _p(f"[trivial-profile] {tp['n_windows']} windows — arm SHAPE before any family "
       f"row (straight |y|<{TRIVIAL_STRAIGHT_M:g} m, const-speed spread<"
       f"{TRIVIAL_CONST_SPEED_M:g} m)")
    for a, r in tp["arms"].items():
        same = ", ".join(f"{b}={v['n']}/{r['n']}" for b, v in r["identical_to"].items())
        _p(f"  {a:14s} n={r['n']:4d} straight={r['straight_frac']:.4f} "
           f"const_speed={r['const_speed_frac']:.4f} "
           f"CONSTANT-VELOCITY={r['trivial_frac']:.4f}"
           + (f"  identical_to: {same}" if same else ""))
    if tp["degenerate_arms"]:
        _p(f"  ⚠️ CONSTANT-VELOCITY on > 50 % of windows: {tp['degenerate_arms']} — "
           f"read their LATERAL rows as a control, never as planning skill")


def analyze_refav1(dump_dir: str, *, n_boot: int = 2000, seed: int = 0,
                   dt: float = DT, tiers: dict | None = None,
                   lead_block: str | None = None) -> dict:
    """``t1_eval.analyze`` on the trajectory dump + the refav1 sidecar analysis.

    ``lead_block``: the per-frame B1 lead block (backlog R1). Joined by
    ``(clip_id, RAW frame 2t)`` BEFORE ``t1_eval.analyze`` so distance-keeping
    reaches ``four_families`` through t1_eval's own ``lead=`` path; ``None`` keeps
    the UNAVAILABLE refusal (with the reason) exactly as before."""
    from taniteval import ci as _ci
    from taniteval import four_families as ff
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not files:
        raise ValueError(f"no ep*.npz under {dump_dir}")
    man_path = os.path.join(dump_dir, "manifest.json")
    manifest = None
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    tiers = dict(ARM_TIERS, **(tiers or {}))
    with np.load(files[0]) as d0:
        arms = [k for k in d0.files if k not in ("g", "ws", *t1._META_KEYS)
                and not k.endswith(t1._FAN_SUFFIXES)]
        k_dump = int(d0["g"].shape[1])
    pairs = [(x, y, nm) for x, y, nm in (
        ("ol", "cl", "paired_closed_minus_open"),
        ("ha", "cl", "paired_cl_minus_ha"),
        # ⭐ the echo test's REAL bar: cl against the constant-velocity straight line.
        # `cl - ha` alone let a straight-line plan read as lateral skill (2026-09-03).
        ("ha0", "cl", "paired_cl_minus_ha0"),
        # ⭐ the ECHO control (echo_gate.ha0_ext, refav1 form) — the bar an arm
        # must clear to claim it read the scene, not its own ego state.
        ("ha0_ext", "cl", "paired_cl_minus_ha0ext"),
        ("cl_navshuf", "cl", "paired_cl_minus_navshuf"),
        ("cl_nonav", "cl", "paired_cl_minus_nonav"),
        ("cl_oraclegoal", "cl", "paired_cl_minus_oraclegoal"),
        ("cl_oracleseed", "cl", "paired_cl_minus_oracleseed"))
        if x in arms and y in arms]
    # ---- ⭐ THE TRIVIAL-PROFILE INSTRUMENT, BEFORE ANY FAMILY ROW -------------
    # It runs here, not later, ON PURPOSE: a reader who sees the family table
    # first has already formed the reading this instrument exists to prevent.
    triv = trivial_profile(files, arms, dt=dt)
    _print_trivial_profile(triv)
    # ---- the lead block join (backlog R1), BEFORE analyze ---------------------
    lead, dk_info = None, None
    if lead_block:
        lead, dk_info = attach_lead_block(files, manifest, lead_block, k=k_dump, dt=dt)
        cov = dk_info["coverage"]
        _p(f"[lead] {os.path.basename(lead_block)}: {cov['n_windows']} windows / "
           f"{cov['n_episodes']} eps · OK eps {cov['n_episodes_ok']} · "
           f"LEAD {cov['counts']['LEAD']} NO_LEAD {cov['counts']['NO_LEAD']} "
           f"NOT_STRAIGHT {cov['counts']['NOT_STRAIGHT']} NO_LABEL "
           f"{cov['counts']['NO_LABEL']} (no-row {cov['n_windows_no_row']}) · "
           f"speed-check max {cov['speed_check']['max_mps']} m/s · "
           f"{dk_info['status']}")
    rec = t1.analyze(files, tiers={x: tiers[x] for x in arms if x in tiers},
                     n_boot=n_boot, seed=seed, dt=dt, paired=pairs, lead=lead)
    rec["tool"] = "taniteval/tools/refav1_arm.py (trajectory families via "\
                  "taniteval/tools/t1_eval.py::analyze, IMPORTED)"

    # ---- per-window components + paired families across arms ---------------
    G_all, P_all, eid_w, v0_w = [], {x: [] for x in arms}, [], []
    for f in files:
        with np.load(f) as d:
            G = d["g"][..., :2].astype(np.float64)
            G_all.append(G)
            eid_w += [os.path.splitext(os.path.basename(f))[0]] * G.shape[0]
            for x in arms:
                P_all[x].append(d[x][..., :2].astype(np.float64))
            if "v0" in d.files:
                v0_w.append(np.asarray(d["v0"], dtype=np.float64).reshape(-1))
    G_all = np.concatenate(G_all)
    N = int(G_all.shape[0])
    P_cat = {x: np.concatenate(P_all[x]) for x in arms}
    comps = {x: _components(P_cat[x], G_all, dt) for x in arms}
    ref = {"n_windows": N, "n_episodes": len(files), "tiers": {x: tiers[x] for x in arms},
           "arm_meaning": {x: ARM_MEANING.get(x) for x in arms},
           "_tier_doctrine": rec["_tier_doctrine"],
           # ⭐ banked FIRST in the record too, for the same reason it prints first.
           "trivial_profile": triv,
           # the cost provenance the dump was produced under (None for a dump
           # written before D-REFAV1-CCOS-EVAL = the shipped `cos` at shipped weights)
           "cost": (manifest or {}).get("cost"),
           "families_paired": {nm: _paired_families(comps, x, y, eid_w, tiers, n_boot, seed)
                               for x, y, nm in pairs},
           "families_note": (
               "LONGITUDINAL / LATERAL / ADE and the TRAJECTORY-DERIVED tactical "
               "rows per arm live in rec['arms'][arm]['four_families'] (t1_eval, "
               "unchanged). STRATEGIC and the DECLARED tactical decisions come "
               "from refav1's own heads and live below (rec['refav1']); the "
               "trajectory-only STRATEGIC row in rec['arms'] stays UNAVAILABLE by "
               "design because a route class cannot be read off a 2 s path.")}
    # ---- LONGITUDINAL distance-keeping (backlog R1): coverage, per-arm rows,
    #      the GT reference, the kinematic-contract check, paired deltas -------
    ref["distance_keeping"] = _distance_keeping_block(
        rec, dk_info, lead, arms, P_cat, pairs, eid_w, dt, n_boot, seed, tiers)

    dec, eid_d = _load_decisions(dump_dir)
    if dec is None:
        ref["sidecar"] = _refused("no decisions/ep*.npz sidecar — this dump was "
                                  "not written by refav1_arm.run_dump", "n/a")
        rec["refav1"] = ref
        return rec
    if len(eid_d) != N:
        raise ValueError(f"decisions sidecar has {len(eid_d)} rows for {N} "
                         f"windows — different grids; refusing the join")

    # ---- NAV + STRATEGIC (route head vs route_label) --------------------------
    from tanitad.refs.refb import ROUTE_CLASSES
    nav_valid = dec["nav_valid"].astype(bool)
    route_lbl = dec["route_label"].astype(int)
    labeled = route_lbl != -100
    n2r = nav_id_to_route_id()
    nav_frac = float(nav_valid.mean()) if N else float("nan")
    strat = {"tier": "T1 (decision heads read the OBSERVED window only; no "
                     "rollout enters them)",
             "nav_valid_frac": round(nav_frac, 4),
             "n_windows": N, "n_nav_valid": int(nav_valid.sum()),
             "n_route_labeled": int(labeled.sum()),
             "n_excluded_no_route_label": int((~labeled).sum()),
             "n_excluded_nav_invalid_among_labeled": int((labeled & ~nav_valid).sum()),
             "nav_shuffle": (manifest or {}).get("nav_shuffle"),
             "_echo_caveat": ("route_label and nav_cmd derive from the SAME "
                              "nav_command field, so under the TRUE nav route "
                              "accuracy measures the nav echo (flagship-v1 scored "
                              "1.0000 on a bijection of its own input). Evidence "
                              "of route skill is ONLY the nav-shuffled / nav-zero "
                              "conditioning; the CHANGED-subset rows are where the "
                              "shuffle control has power."),
             "conditionings": {}}
    use = labeled & nav_valid
    eid_use = [e for e, k in zip(eid_d, use) if k]
    for cname, navc in (("nav_true", dec["nav_cmd"]), ("nav_shuffled", dec["nav_cmd_shuf"]),
                        ("nav_zero", np.zeros(N, dtype=int))):
        pred = dec[f"route_pred_{cname}"].astype(int)
        if (pred < 0).all():
            strat["conditionings"][cname] = _refused(
                "the checkpoint has no hierarchy brains (route_pred = -1)", "T1", N)
            continue
        if use.sum() == 0:
            strat["conditionings"][cname] = _refused(
                "no window is both route-labeled and nav-valid", "T1", 0)
            continue
        blk = ff._agreement_block(route_lbl[use], pred[use], list(ROUTE_CLASSES),
                                  eid_use, n_boot, seed, tier="T1")
        implied = np.array([n2r.get(int(x), -1) for x in navc])
        blk["nav_implied_route_agreement"] = round(
            float((pred[use] == implied[use]).mean()), 4)
        blk["_nav_implied_route_agreement_is"] = (
            "fraction of scored windows whose route_pred equals the route the "
            "FED nav token maps to — the echo index under this conditioning")
        maj = int(np.bincount(route_lbl[use], minlength=len(ROUTE_CLASSES)).argmax())
        blk["majority_class"] = ROUTE_CLASSES[maj]
        blk["majority_class_rate"] = round(float((route_lbl[use] == maj).mean()), 4)
        strat["conditionings"][cname] = blk
    # paired contrasts on the labeled∩valid windows, and on the CHANGED subset
    if use.sum() and not (dec["route_pred_nav_true"] < 0).all():
        pt_ = dec["route_pred_nav_true"].astype(int)
        ps_ = dec["route_pred_nav_shuffled"].astype(int)
        c_t = (pt_ == route_lbl).astype(float)
        c_s = (ps_ == route_lbl).astype(float)
        strat["paired_true_minus_shuffled_accuracy"] = _ci.paired_episode_cluster_bootstrap(
            c_t[use], c_s[use], eid_use, n_boot=n_boot, seed=seed)
        changed = use & (dec["nav_cmd_shuf"] != dec["nav_cmd"])
        strat["n_changed_subset"] = int(changed.sum())
        if changed.sum():
            e_c = [e for e, k in zip(eid_d, changed) if k]
            implied_s = np.array([n2r.get(int(x), -1) for x in dec["nav_cmd_shuf"]])
            follows_nav = (ps_ == implied_s).astype(float)
            strat["changed_subset"] = {
                "n": int(changed.sum()),
                "tier": "T1",
                "estimator": "episode_cluster_bootstrap",
                "route_follows_LABEL_under_shuffle": _ci.episode_cluster_bootstrap(
                    c_s[changed], e_c, n_boot=n_boot, seed=seed),
                "route_follows_SHUFFLED_NAV_under_shuffle": _ci.episode_cluster_bootstrap(
                    follows_nav[changed], e_c, n_boot=n_boot, seed=seed),
                "_reading": ("on windows whose nav token CHANGED, the label route "
                             "and the shuffled-nav route are different classes by "
                             "construction, so these two rates are mutually "
                             "exclusive: an ECHO reads follows_nav ~ 1 / "
                             "follows_label ~ 0; route skill from vision reads "
                             "follows_label high regardless of the token."),
            }
        else:
            strat["changed_subset"] = _refused(
                "the permutation changed no nav token (FOLLOW-dominated "
                "marginal) — the shuffle control has no power on this set", "T1")
    ref["strategic"] = strat

    # ---- TACTICAL, DECLARED heads vs v7.2 lat/lon labels -----------------------
    from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
    vv = ((manifest or {}).get("model") or {}).get("tac_vocab_version", "v7.0")
    tac = {"tier": "T1 (decision heads read the OBSERVED window only)",
           "vocabulary": vv, "n_windows": N,
           "_is": ("the DECLARED decision (argmax of refav1's factored lat/lon "
                   "heads) vs the v7.2 s2 record's a_tac label on IN-BAND windows "
                   "(-100 rows excluded). This is the 'selected' half the binding "
                   "rule names; the EXECUTED manoeuvre read off the driven path "
                   "is the trajectory-derived block in rec['arms'] (refc 3-way "
                   "classes — a DIFFERENT instrument; no mapping between the two "
                   "vocabularies is invented here)."),
           "conditionings": {}}
    for hk, names in (("lat", list(tactical_lat_actions(vv))),
                      ("lon", list(tactical_lon_actions_v(vv)))):
        lbl = dec[f"{hk}_label"].astype(int)
        m = lbl != -100
        for cname in ("nav_true", "nav_shuffled", "nav_zero"):
            pred = dec[f"{hk}_pred_{cname}"].astype(int)
            key = f"{hk}_{cname}"
            if (pred < 0).all():
                tac["conditionings"][key] = _refused("no hierarchy brains", "T1", N)
            elif m.sum() == 0:
                tac["conditionings"][key] = _refused(
                    f"no window carries a {hk} label (labels absent or every "
                    f"window out of band)", "T1", 0)
            else:
                e_m = [e for e, k in zip(eid_d, m) if k]
                blk = ff._agreement_block(lbl[m], pred[m], names, e_m, n_boot, seed,
                                          tier="T1")
                blk["n_excluded_no_label"] = int((~m).sum())
                tac["conditionings"][key] = blk
        if m.sum() and not (dec[f"{hk}_pred_nav_true"] < 0).all():
            e_m = [e for e, k in zip(eid_d, m) if k]
            ct = (dec[f"{hk}_pred_nav_true"].astype(int) == lbl).astype(float)
            cs = (dec[f"{hk}_pred_nav_shuffled"].astype(int) == lbl).astype(float)
            tac[f"{hk}_paired_true_minus_shuffled_accuracy"] = \
                _ci.paired_episode_cluster_bootstrap(ct[m], cs[m], e_m,
                                                     n_boot=n_boot, seed=seed)
    ref["tactical_declared"] = tac

    # ---- T0 WM diagnostic: feature error vs the persist-last-field control ----
    mm, mc = dec["wm_mse_model"].astype(np.float64), dec["wm_mse_const"].astype(np.float64)
    mz = (dec["wm_mse_zero"].astype(np.float64) if "wm_mse_zero" in dec
          else np.full_like(mm, np.nan))
    k_wm = int(mm.shape[1])
    steps = sorted({s for s in (1, 5, 10, 15, 20, 30, k_wm) if 1 <= s <= k_wm})
    wm = {"tier": "T0", "tier_note": _TIER_NOTE["T0"],
          "n": N, "k_wm": k_wm, "dt_s": dt,
          "space": ("standardised DINOv3 space (to_enc(op_pred) vs std(future); "
                    "the trainer's loss_feat_op definition) — target per-channel "
                    "variance pinned ~1, so the constant control is a KNOWN "
                    "VALUE" if (manifest or {}).get("model", {}).get("target_space")
                    == "frozen" else "adapter space (target variance NOT pinned)"),
          "tgt_std_mean": round(float(np.nanmean(dec["wm_tgt_std"])), 4),
          "_controls": {
              "feat_mse_const": ("MSE of PERSISTING the last observed field — "
                                 "the raw-input floor; a predictor that beats "
                                 "it, separated, has learned dynamics"),
              "feat_mse_zero": ("MSE of predicting the standardised MEAN (0) — "
                                "the CONSTANT-ONLY control. In frozen space it "
                                "must read ~ the target variance (~1.0, see "
                                "tgt_std_mean); a value far from it means the "
                                "standardizer does not describe this corpus")},
          "estimator": "episode_cluster_bootstrap; paired for const - model"}
    comps_wm = {"feat_mse_model_mean_over_steps": (mm.mean(1), "mean", 5),
                "feat_mse_const_mean_over_steps": (mc.mean(1), "mean", 5)}
    if np.isfinite(mz).all():
        comps_wm["feat_mse_zero_mean_over_steps"] = (mz.mean(1), "mean", 5)
    for s in steps:
        comps_wm[f"feat_mse_model_step{s}"] = (mm[:, s - 1], "mean", 5)
        comps_wm[f"feat_mse_const_step{s}"] = (mc[:, s - 1], "mean", 5)
    wm["intervals"] = _ci.bootstrap_metrics(comps_wm, eid_d, n_boot=n_boot, seed=seed)
    if np.isfinite(mz).all():
        wm["paired_zero_minus_model_mean_over_steps"] = \
            _ci.paired_episode_cluster_bootstrap(mz.mean(1), mm.mean(1), eid_d,
                                                 n_boot=n_boot, seed=seed)
    wm["paired_const_minus_model"] = {
        "mean_over_steps": _ci.paired_episode_cluster_bootstrap(
            mc.mean(1), mm.mean(1), eid_d, n_boot=n_boot, seed=seed),
        **{f"step{s}": _ci.paired_episode_cluster_bootstrap(
            mc[:, s - 1], mm[:, s - 1], eid_d, n_boot=n_boot, seed=seed)
           for s in steps}}
    ref["wm_diagnostic_T0"] = wm

    # ---- planner provenance per T1 planning arm ------------------------------
    names_src = (manifest or {}).get("plan_source_names", PLAN_SOURCE_NAMES)
    goal_man = (manifest or {}).get("goal") or {}
    names_goal = goal_man.get("source_names") or GOAL_SOURCE_NAMES
    lat_names, lon_names = list(tactical_lat_actions(vv)), list(tactical_lon_actions_v(vv))
    plan = {}
    for arm in arms:
        if f"plan_source_{arm}" not in dec:
            continue
        src = dec[f"plan_source_{arm}"].astype(int)
        cnt = np.bincount(src, minlength=len(names_src))
        agree = dec[f"plan_agree_{arm}"].astype(int)
        plan[arm] = {
            "tier": tiers.get(arm), "n": N,
            "source_fractions": {names_src[i]: round(float(c / N), 4)
                                 for i, c in enumerate(cnt) if c},
            "baseline_won_frac": round(float((src != names_src.index("cem")).mean()), 4)
            if "cem" in names_src else None,
            "coarse_fine_agree_rate": (round(float((agree[agree >= 0] == 1).mean()), 4)
                                       if (agree >= 0).any() else None),
            "cost_mean": round(float(dec[f"plan_cost_{arm}"].mean()), 6),
            "n_evaluated_mean": round(float(dec[f"plan_neval_{arm}"].mean()), 1),
            **_goal_provenance(dec, arm, N, names_goal, goal_man.get("space"),
                               lat_names, lon_names),
            "_reading": ("read goal_source_fractions FIRST: 'none' means the cost "
                         "had no goal term and the T1 trajectory is a floor "
                         "baseline by construction (pre-e609a98 this was a "
                         "decel_1.5 brake on every window; since e609a98 ties "
                         "resolve to hold_v0); 'tactical_imagined' means plan() "
                         "imagined its goal from vision + nav + v0 and "
                         "goal_action_{lat,lon} is the SELECTED manoeuvre behind "
                         "it. A baseline_won_frac near 1.0 is then a property of "
                         "the search / the init, reported with its source — never "
                         "planning skill; see the module docstring and "
                         "REFAV1_ARM.md"),
        }
    ref["planner"] = plan
    ref["protocol"] = {
        "inference_inputs": ("cached frozen DINOv3 patch features of the OBSERVED "
                             "window (vision); measured v0 at t0 (PI ruling "
                             "2026-09-02); the v7.2 nav token (goal input, PI "
                             "2026-08-03); NOTHING recorded after frame 2t on "
                             "the T1 / hold-action arms"),
        "vision_only": ("vision + v0(t0) + nav token — both additions admitted "
                        "by the two PI rulings above; no ego state beyond v0, no "
                        "future"),
        "goal_source": _goal_source_protocol(plan),
        "goal_situation_disjoint": ("True by construction: nav is the labels blob's "
                                    "nav_command token (allow_oracle_nav-stamped, "
                                    "Alpamayo-CoT+ego derivation), not a situation "
                                    "classifier output"),
        "corpus": ((manifest or {}).get("corpus") or {}).get("cache"),
        "labels": ((manifest or {}).get("corpus") or {}).get("labels"),
        "parity_key": ("the eval split is the v7.2 EVAL clip set, identified by "
                       "the labels blob md5 in corpus.join_report and the cache "
                       "index.json (corpus.cache_index) — NOT the canonical "
                       "train parity key e438721ae894; cross-arm deltas are "
                       "valid only against dumps on this SAME cache + grid"),
    }
    # ⭐ DECLARE THE PROTOCOL ON EVERY ARM'S BLOCK. `t1_eval.analyze` cannot know
    # these (it never sees the model); this tool does, and all_families' own
    # contract is "pass protocol= — an absent value is a WORK ITEM, never
    # compliance". Filled post hoc, and SAID so, so the provenance is visible.
    for arm in rec.get("arms", {}):
        fam = rec["arms"][arm].get("four_families")
        if not isinstance(fam, dict) or "_protocol" not in fam:
            continue
        fam["_protocol"].update({k: v for k, v in ref["protocol"].items()
                                 if k in fam["_protocol"]})
        fam["_protocol"]["_declared_by"] = ("taniteval/tools/refav1_arm.py "
                                            "analyze_refav1 (post hoc, from the "
                                            "roll's manifest + the model's own "
                                            "input contract)")
        und = "UNDECLARED"
        fam["_protocol_undeclared"] = sorted(
            k for k, v in fam["_protocol"].items()
            if not k.startswith("_") and isinstance(v, str) and v.startswith(und))
    ref["manifest"] = manifest
    rec["refav1"] = ref
    return rec


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _survive_a_narrow_console() -> str | None:
    """⛔ A HELP STRING IS CODE, AND THIS FILE'S HELP STRINGS CARRY ``⛔``/``⚠️``.

    MEASURED 2026-09-05 on the dev box: ``refav1_arm.py --help`` exits **1** with
    ``UnicodeEncodeError: 'charmap' codec can't encode character '\\u26d4'``,
    because a default Windows console is **cp1252** and argparse writes the whole
    help text in one call. **Nine** pre-existing help strings carry a marker, so
    the failure is not one typo. Every RUNNING arm is fine — the queue scripts set
    ``PYTHONIOENCODING=utf-8`` — which is exactly why this stayed invisible: it
    surfaces only when an operator asks for help or mistypes a flag, i.e. at the
    worst possible moment to be handed a traceback.

    ⚠️ Sibling of the ``%`` defect logged in `M30` §4, one layer down: that one
    broke help through argparse's FORMATTER, this one through the STREAM. Fixing
    the formatter did not fix the stream, and the same command still died.

    ⇒ Degrade the markers instead of dying. On a stream that can already encode
    them **nothing changes**, so no log and no banked record moves.
    """
    enc = getattr(sys.stdout, "encoding", None)
    try:
        "⛔⚠️⭐—".encode(enc or "ascii", errors="strict")
        return None
    except (UnicodeEncodeError, LookupError, TypeError):
        pass
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="backslashreplace")
        except Exception:                                        # noqa: BLE001
            pass
    return str(enc)


def main(argv=None):
    _narrow = _survive_a_narrow_console()
    ap = argparse.ArgumentParser(
        description="refav1 T0/T1 eval adapter (writes a t1_eval-compatible dump "
                    "+ a decisions sidecar; analyses both).")
    ap.add_argument("--ckpt", help="refa_v1_train.py ckpt.pt")
    ap.add_argument("--config", default=None,
                    help="config.json (default: sibling of --ckpt, else ckpt['cfg'])")
    ap.add_argument("--cache", help="eval feature cache dir (<episode>.pt fp8/fp16)")
    ap.add_argument("--episodes", help="v2ep episode dir (<episode>.v2ep.pt)")
    ap.add_argument("--labels", default=None, help="v7.2 s2 labels blob")
    ap.add_argument("--nav", default=None, help="v7.2 nav blob (usually = --labels)")
    ap.add_argument("--arm", default="refav1", help="label for the output record")
    ap.add_argument("--out", help="JSON output FILE")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--analyze-only", default=None, metavar="DUMP_DIR")
    ap.add_argument("--dump-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes-n", type=int, default=0,
                    help="first N episodes of the sorted cache (0 = all)")
    ap.add_argument("--window-stride", type=int, default=1)
    ap.add_argument("--window-list", default=None,
                    help="JSON {'windows': [[episode_name, t], ...], 'rule': "
                         "str} naming an explicit, externally computed window "
                         "selection. OVERRIDES --window-stride. Omitting it is "
                         "bit-identical to every banked arm. Use it when a "
                         "STRATUM must be enriched: a stride cannot, and a "
                         "recall on 11 windows resolves only to 1/11 = 0.0909, "
                         "coarser than the 0.0750 absolute inference-seed floor "
                         "it is compared against.")
    ap.add_argument("--action-units", choices=("kappa", "steer"), default="kappa",
                    help="unit of v2ep actions[:,0] as this run READS it. "
                         "'kappa' (default) = the LEGACY, unconverted reading "
                         "every banked number was produced under. 'steer' = the "
                         "repaired contract: kappa = tan(steer)/2.9 before any "
                         "integration, and arctan(2.9*kappa) at the "
                         "planner->model boundary. NOT comparable to 'kappa'.")
    ap.add_argument("--horizon-k", type=int, default=K_TRAJ_DEFAULT,
                    help=f"trajectory horizon in 0.2 s steps (default "
                         f"{K_TRAJ_DEFAULT} = 2.0 s = the plan horizon)")
    ap.add_argument("--wm-k", type=int, default=0,
                    help="teacher-forced WM diagnostic horizon (0 = cfg.op_steps)")
    ap.add_argument("--lru", type=int, default=8)
    ap.add_argument("--plan-n-samples", type=int, default=None)
    ap.add_argument("--plan-n-iters", type=int, default=None)
    ap.add_argument("--plan-n-elites", type=int, default=None)
    ap.add_argument("--plan-seed", type=int, default=0)
    ap.add_argument("--cost-metric", choices=("cos", "chord", "ccos", "ccosh"),
                    default="cos",
                    help="the goal term's form (refa_v1.COST_METRICS). 'cos' = the "
                         "shipped default every banked number was produced under; "
                         "'chord' = monotone-equivalent, cannot re-rank; 'ccos' = "
                         "centred on the window's own zero-action terminal field, "
                         "CAN re-rank; 'ccosh' = ccos WITH THE HOLD BRANCH -- the "
                         "pinned chord (distance) form on windows whose goal IS the "
                         "hold field, where the centred direction is float32 noise "
                         "and ccos charges the do-nothing candidate 1.0 EXACTLY for "
                         "obeying the goal (MEASURED 40/40 windows; median "
                         "basecost_cv = 1.0, raw/cost_scale.txt). ⛔ neither non-default form is weight-neutral: "
                         "declare --cost-weights in the same arm or the record "
                         "carries an implicit re-weighting")
    ap.add_argument("--cost-weights", default=None,
                    help="'w_jerk,w_kappa,w_vend' passed to plan(cost_weights=); "
                         "default = the shipped module constants (0.02, 0.05, 0.10). "
                         "Banked in manifest['cost'] and the record")
    ap.add_argument("--goal-kappa-turn", type=float, default=None,
                    help="the SUSTAINED curvature (1/m) a TURN_L/TURN_R goal "
                         "commands. Omit for the shipped GOAL_KAPPA_TURN=0.08 "
                         "(R 12.5 m), which is BIT-IDENTICAL to every arm "
                         "banked before 2026-09-05. MEASURED "
                         "(D-REFAV1-VOCAB-QUANT, 4786 windows / 141 episodes): "
                         "the vocabulary's only sustained curvatures are 0 and "
                         "0.08 because NUDGE_*/LANE_CHANGE_* are S-curves with "
                         "zero NET heading change, while the corpus curves at "
                         "R 100-1000 m — so LANE_KEEP is the VOCABULARY-OPTIMAL "
                         "token on 90.6 %% of GT-turn windows and the head "
                         "already turns MORE than its vocabulary justifies. "
                         "Under an ORACLE chooser 0.02 makes 100 %% of real "
                         "turns expressible against 38.7 %% at 0.08, and cuts "
                         "the MEDIAN curvature error on a turn 2.5x. "
                         "⛔ Needs --cost-metric ccos to reach the wheels: "
                         "under the shipped cos a decoded turn is refused on "
                         "38/38 windows (D-REFAV1-DRIVE-GATE2)")
    ap.add_argument("--kamm-mu", type=float, default=None,
                    help="friction coefficient for a SPEED-DEPENDENT curvature "
                         "cap |kappa| <= mu*g/v^2 inside PlanConfig._clip, on the "
                         "candidate's OWN speed profile. Omit for the shipped "
                         "constant kappa_max = 0.2 (bit-identical to every "
                         "pre-2026-09-05 arm). MEASURED reason "
                         "(raw/feas_audit.txt, v0 >= 2 m/s, n = 27, GROUND-TRUTH "
                         "control reads envelope 0.0000 / kamm_over 0.0000): "
                         "refav1's plans are ENVELOPE-feasible by construction "
                         "(envelope_rate 0.0000, max|kappa| exactly the clip) yet "
                         "29.6 %% leave the mu = 0.7 friction circle, 42.1 %% at "
                         "v0 >= 5 m/s, peak_g up to 3.262 against a ground truth "
                         "of 0.373. mu = 0.7 is the programme's own MU_KAMM.")
    ap.add_argument("--kamm-v-floor", type=float, default=None,
                    help="speed (m/s) below which the constant kappa_max governs "
                         "(default 2.0); mu*g/v^2 exceeds the clip there anyway "
                         "and the division is ill-conditioned")
    ap.add_argument("--a-sustain-mode", choices=("none", "a0", "a0_shift"),
                    default="none",
                    help="THE LONGITUDINAL VOCABULARY LEVER (D-REFAV1-LON-VOCAB). "
                         "'none' is the shipped path, BIT-IDENTICAL to every arm "
                         "banked before 2026-09-05. 'a0' gives the goal's MAINTAIN "
                         "branch (v_t == v0: CRUISE always, ADAPT_SPEED_FOR_CURVE "
                         "below GOAL_CURVE_VMAX_MPS) a CONSTANT acceleration equal "
                         "to the MEASURED a0 at t0 -- the same backward difference "
                         "of past speeds ha0_ext holds, no future. MEASURED reason "
                         "(raw/lon_branch.txt, n = 40 windows, ckpt 21109): the "
                         "maintain branch is 31/40 windows (77.5 %%) and 20/24 "
                         "(83.3 %%) of the GT-LON stratum, and on it the shipped "
                         "goal commands a == 0 EXACTLY; the vocabulary's reachable "
                         "dv over 2 s is [-2.85, +0.98] m/s against a corpus p90 of "
                         "+2.34, so 14 of the 17 accelerating windows (82.4 %%) are "
                         "outside it entirely. THIS IS A VOCABULARY CHANGE: it "
                         "moves the goal field the plan is scored against, so it is "
                         "NOT window-comparable with a shipped-vocabulary arm on "
                         "the goal term -- it is comparable on the four families. "
                         "'a0_shift' is D2 and DOMINATES 'a0' on every measured "
                         "column: instead of touching only the maintain branch it "
                         "shifts EVERY relative target by the a0 extrapolation "
                         "(v_t' = v_t + a0*GOAL_REACH_S), i.e. the tokens name a "
                         "speed change relative to WHERE YOU ARE GOING rather than "
                         "to where you are; it reduces to 'a0' at the first step on "
                         "the maintain branch. MEASURED mean paired difference "
                         "against the ha0_ext floor (raw/lon_designs.txt): shipped "
                         "canonical LON speed +0.4551 / ADE +0.1492; 'a0' +0.1752 / "
                         "+0.0059; 'a0_shift' +0.1184 / -0.0389 -- 76 %% of the "
                         "deficit closed and the only design that goes NEGATIVE on "
                         "ADE. Absolute targets (HOLD, CREEP, ADAPT above "
                         "GOAL_CURVE_VMAX_MPS) are never shifted")
    ap.add_argument("--jerk-seam", choices=("off", "a0"), default="off",
                    help="PRICE THE JERK SEAM (D-REFAV1-LON-COST). The shipped "
                         "jerk term diffs the plan's OWN actions only, so the step "
                         "from the car's measured a0 to controls[0] is FREE: "
                         "dropping instantly from a0 = -2.3 m/s^2 to a = 0 costs "
                         "the same as continuing smoothly, and the all-zero plan is "
                         "the joint minimiser of both regularisers. 'a0' prepends "
                         "the measured a0 as the (-1)-th action so the seam is "
                         "priced with the SAME w_jerk -- a repair of an incomplete "
                         "term, not a new weight. 'off' is bit-identical to every "
                         "pre-2026-09-05 arm")
    ap.add_argument("--seed-kappa-ladder", default=None,
                    help="comma-separated SUSTAINED curvature magnitudes "
                         "(1/m) to add to iCEM's iteration-0 candidate pool, "
                         "both signs, on the decoded goal seed's own accel "
                         "profile. MEASURED reason (raw/kappa_quantisation.txt, "
                         "n = 40 windows, ckpt 21109, ccos): the winning plan's "
                         "curvature series is EXACTLY CONSTANT on 31/40 windows "
                         "and reads exactly 0.000000 (x10) or 0.080000 (x21) - "
                         "i.e. the decoded token's canonical profile verbatim - "
                         "because colored_noise is zero-mean along time and no "
                         "baseline carries curvature, so the pool offers only "
                         "{0, +-0.08} while the corpus curves at |kappa| "
                         "0.001-0.01. Omit for the SHIPPED pool (bit-identical "
                         "to every pre-2026-09-05 arm). NOT a vocabulary change.")
    ap.add_argument("--lat-logit-bias", default=None,
                    help="comma-separated additive bias on the LATERAL goal "
                         "logits, one value per lat token (8 for v7.0). This "
                         "is the goal head's DECISION RULE and therefore "
                         "refav1's lateral policy: the decoded token's "
                         "canonical profile is the planner's only "
                         "curvature-carrying candidate, so a LANE_KEEP decode "
                         "forces curvature EXACTLY 0 (MEASURED 244/244 "
                         "windows). Omit for plain argmax (bit-identical to "
                         "pre-2026-09-05 arms). A class-prior correction is "
                         "'-tau*log(prior)'; a commit threshold is a negative "
                         "entry in the LANE_KEEP slot.")
    ap.add_argument("--nav-shuffle-seed", type=int, default=0)
    ap.add_argument("--no-navshuf", action="store_true",
                    help="skip the nav-shuffle T1 arm (⛔ then the record is "
                         "NOT admissible for any nav-conditioned claim)")
    ap.add_argument("--with-nonav-arm", action="store_true")
    ap.add_argument("--with-oracle-goal-arm", action="store_true",
                    help="ALSO roll cl_oraclegoal (T0: the true future field as "
                         "the planning goal)")
    ap.add_argument("--goal-kappa-levels", default=None,
                    help="M15's L=3 lateral vocabulary: 'm15' for the measured "
                         "corpus quantiles, or a comma list of magnitudes (1/m). "
                         "⚠️ THIS CHANGES THE ACTION SPACE — every banked refav1 "
                         "number is under L1-0.08 and a cross-vocabulary "
                         "comparison is inadmissible unless it says so. Requires "
                         "--goal-kappa-hint.")
    ap.add_argument("--goal-kappa-hint", default=None, choices=(None, "gt"),
                    help="the level CHOOSER. 'gt' = the window's TRUE curvature "
                         "=> ⛔ T0, a vocabulary-adequacy BOUND, never a driving "
                         "number. The v7.0 head cannot choose a magnitude, so a "
                         "level set without this is refused rather than "
                         "defaulted (which would hide the tier).")
    ap.add_argument("--with-oracle-seed-arm", action="store_true",
                    help="ALSO roll cl_oracleseed (T0: the DE-CONFOUNDED "
                         "oracle — the true future field as the goal WITH the "
                         "head's own canonical seed retained, so the goal is "
                         "the only difference from cl)")
    ap.add_argument("--allow-nonstrict", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", default="", help="extra tier stamps name=T0|T1")
    ap.add_argument("--lead-block", default=None,
                    help="per-frame B1 lead block (tools/build_lead_block_b1.py) "
                         "for LONGITUDINAL distance-keeping; default = the banked "
                         f"block when it exists: {LEAD_BLOCK_DEFAULT}")
    ap.add_argument("--no-lead-block", action="store_true",
                    help="analyse WITHOUT a lead block (distance-keeping stays "
                         "UNAVAILABLE with its reason — a WORK ITEM, not a pass)")
    a = ap.parse_args(argv)
    if _narrow:
        _p(f"[console] stdout encoding is {_narrow!r}, which cannot represent "
           f"this tool's markers; they are backslash-escaped for this run. Set "
           f"PYTHONIOENCODING=utf-8 for readable output. (No banked number "
           f"depends on this — see _survive_a_narrow_console.)")
    lead_path = None
    if not a.no_lead_block:
        lead_path = a.lead_block or (LEAD_BLOCK_DEFAULT if os.path.exists(LEAD_BLOCK_DEFAULT)
                                     else None)
        if a.lead_block and not os.path.exists(a.lead_block):
            sys.exit(f"--lead-block {a.lead_block} does not exist")
        if lead_path is None:
            _p(f"[lead] no lead block (banked default absent: {LEAD_BLOCK_DEFAULT}); "
               f"distance-keeping will be UNAVAILABLE with its reason")

    if not a.out:
        sys.exit("--out is required")
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")
    # ⛔ AHEAD OF THE ROLLOUT, ALWAYS. See preflight_analysis_imports().
    preflight_analysis_imports(lead_used=lead_path is not None)

    if a.analyze_only is None:
        for r, name in ((a.ckpt, "--ckpt"), (a.cache, "--cache"),
                        (a.episodes, "--episodes"), (a.dump_dir, "--dump-dir")):
            if not r:
                sys.exit(f"rollout mode needs {name} (or use --analyze-only)")
        manifest = run_dump(a)
        dump_dir = a.dump_dir
        if a.dump_only:
            _p(f"[dump-only] {dump_dir}; analyse later with --analyze-only")
            return
    else:
        dump_dir = a.analyze_only
    rec = analyze_refav1(dump_dir, n_boot=a.n_boot, seed=a.seed,
                         tiers=t1._parse_tiers(a.tiers), lead_block=lead_path)
    rec.update({"arm": a.arm, "ckpt": a.ckpt, "dump_dir": dump_dir,
                "mode": "analyze-only" if a.analyze_only else "rollout+analyze",
                "lead_block": lead_path,
                # the cost provenance, promoted to the top level so a reader
                # (and the suite) never has to guess which metric produced `cl`
                "cost": (rec.get("refav1") or {}).get("cost"),
                "_unverified": _UNVERIFIED_ON_REAL_CKPT})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    _p(f"[out] {a.out}")
    r = rec.get("refav1", {})
    dkb = r.get("distance_keeping") or {}
    for arm, blk in rec["arms"].items():
        ivl = blk["intervals"]["metrics"]
        ade = ivl.get("ade_dense_m", {})
        dk = (dkb.get("per_arm") or {}).get(arm) or {}
        _p(f"  {arm:14s} tier={blk['tier']}  ADE={ade.get('mean')} "
           f"[{ade.get('lo')}, {ade.get('hi')}]  families_unavailable="
           f"{blk['four_families']['_families_unavailable']}  "
           f"distance_keeping={dk.get('status') or dkb.get('status')} "
           f"n={dk.get('n')} headway={dk.get('mean_headway_min_m')} "
           f"time_gap={dk.get('mean_time_gap_min_s')} min_ttc={dk.get('mean_min_ttc_s')}")
    cov = dkb.get("coverage")
    if cov:
        _p(f"  distance_keeping coverage: {cov['counts']} of {cov['n_windows']} windows "
           f"(no-row {cov['n_windows_no_row']}; eps OK {cov['n_episodes_ok']}/"
           f"{cov['n_episodes']}) status={dkb.get('status')}")
    else:
        _p(f"  distance_keeping: {dkb.get('status')} — {str(dkb.get('reason'))[:160]}")
    for arm, pl in (r.get("planner") or {}).items():
        _p(f"  planner[{arm}] baseline_won_frac={pl['baseline_won_frac']} "
           f"sources={pl['source_fractions']} goal={pl.get('goal_source_fractions')} "
           f"goal_action_lon={pl.get('goal_action_lon')}")
    s = r.get("strategic", {})
    _p(f"  strategic nav_valid_frac={s.get('nav_valid_frac')} "
       f"n_labeled={s.get('n_route_labeled')}")


if __name__ == "__main__":
    main()
