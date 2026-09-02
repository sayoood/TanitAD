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

⚠️ WHAT ``plan()`` DOES WITH NO GOAL — READ BEFORE QUOTING A T1 NUMBER. Its
docstring says ``goal_field`` "defaults to the tactical brain's own imagined 6 s
field"; the CODE does ``search_goal = None if goal_field is None`` and the cost
then contains ONLY the comfort (jerk) and curvature terms. EVERY zero-curvature
constant-acceleration candidate has jerk 0 and scores EXACTLY 0 there — cv,
hold_v0 AND decel_1.5 tie — and the floor loop in ``icem_plan`` takes the LAST
tie (``<=`` over the baseline dict in insertion order), i.e. ``baseline:decel_1.5``.
⛔ MEASURED on a random-init RefAV1 (stack/tests/test_refav1_arm.py): 51/51
windows returned ``baseline:decel_1.5`` — the "T1 plan" is a constant −1.5 m/s²
brake on every window, not constant velocity and not planning. (My first draft
of this paragraph said "cv by construction"; the test corrected it.) refav1 has
no TRAINED inference-time goal source today (the proposal head trains only with
``w_aux_head > 0``; ``target_latent`` is consumed by no loss). ⇒ this tool records
``plan_source`` per window and reports the fraction of windows whose plan was a
baseline and WHICH, so a T1 number that is really a baseline cannot be read as
planning skill. That is a DECISION for the Master Mind (see REFAV1_ARM.md), not
something this adapter invents a goal for.

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
    <dump>/manifest.json             episodes, grid, tiers, plan config, provenance.

ESTIMATOR: full-set pooled point estimates; intervals = episode-cluster bootstrap
(``taniteval.ci``), paired across arms on the same windows. ``overlapping_holdout_se``
is never used (it biases the POINT ESTIMATE). Every block prints its n and tier.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
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
             "cl_oraclegoal": "T0", "ha": "T1", "ol": "T0"}
ARM_MEANING = {
    "cl": "T1 — plan() at t0, TRUE nav; predictor consumes the planner's own "
          "actions; trajectory = unicycle(controls, measured v0)",
    "cl_navshuf": "T1 — as cl with nav_cmd PERMUTED across eval windows "
                  "(D-REFAV1-NAV-DEPTH eval obligation)",
    "cl_nonav": "T1 — as cl with nav_cmd=None (index 0 'follow')",
    "cl_oraclegoal": "T0 — as cl with the TRUE future field as goal_field "
                     "(future information => T0; search-vs-goal attribution)",
    "ha": "T1 — hold the last OBSERVED (a, kappa) (closes at t0) for K steps; "
          "consumes no recorded future",
    "ol": "T0 — the RECORDED future (a, kappa) integrated from v0: the "
          "kinematic-contract control (must reproduce GT), NOT a WM diagnostic",
}
_TIER_NOTE = dict(t1._TIER_NOTE)
PLAN_SOURCE_NAMES = ["cem", "baseline:cv", "baseline:hold_v0",
                     "baseline:proposal", "baseline:decel_1.5"]
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


def paths_from_controls(controls, v0: float, dt: float, k: int):
    """``[K',>=2]`` controls -> ``[1,K,2]`` ego-frame path via the programme's
    ONE unicycle integrator (``refa_v1_plan.unicycle_paths``): position advances
    on the speed at the START of the step, v updates last, clamped at 0."""
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
    return unicycle_paths(c[:, :k].float(), v0_t, dt)


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
    pc = PlanConfig(**kw)
    pc.sanity()
    return pc


def _source_code(src: str) -> int:
    if src not in PLAN_SOURCE_NAMES:
        PLAN_SOURCE_NAMES.append(src)
    return PLAN_SOURCE_NAMES.index(src)


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

    stride = max(1, int(a.window_stride))
    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % stride == 0]
    if not sel:
        raise SystemExit("[refav1_arm] the stride selected zero windows")
    # ---- nav for every selected window (the loader's own table), then shuffle
    nav_true = np.zeros(len(sel), dtype=np.int64)
    nav_valid = np.zeros(len(sel), dtype=bool)
    if ld._nav_on:
        for i, (wi, ei, t) in enumerate(sel):
            nid = ld._nav_id.get(ld.clip_id[ld.names[ei]])
            nav_true[i] = 0 if nid is None else int(nid)
            nav_valid[i] = nid is not None
    nav_shuf, shuf_stats = shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)

    arms = ["cl", "ha", "ol"]
    if ld._nav_on and not a.no_navshuf:
        arms.append("cl_navshuf")
    if a.with_nonav_arm:
        arms.append("cl_nonav")
    if a.with_oracle_goal_arm:
        arms.append("cl_oraclegoal")
    plan_arms = [x for x in arms if x.startswith("cl")]
    pc = _plan_cfg(cfg, a)
    _p(f"[grid] K={k} ({k * DT:.1f} s) K_wm={k_wm} stride={stride} "
       f"windows={len(sel)} arms={arms} plan={{samples {pc.n_samples}, iters "
       f"{pc.n_iters}, elites {pc.n_elites}, seed {pc.seed}}} nav_shuffle="
       f"{shuf_stats['n_changed']}/{shuf_stats['n_windows']} changed")
    os.makedirs(a.dump_dir, exist_ok=True)
    os.makedirs(os.path.join(a.dump_dir, "decisions"), exist_ok=True)

    frozen = cfg.target_space == "frozen"
    ld._order = [0]
    ld._cursor = 0
    episodes_manifest = []
    n_done, t_plan_first = 0, None
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
                ol = paths_from_controls(act[0], v0, DT, k)
                hold = hold_action_controls(ld, v_ep, kap_ep, t).to(dev)
                ha = paths_from_controls(hold[None].expand(k, 2), v0, DT, k)
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
                if "cl_oraclegoal" in arms:
                    hp = cfg.plan_steps
                    goal_oracle = model.adapter(model.std(fut[:, :hp]))[:, hp - 1]
                plans = {}
                for arm in plan_arms:
                    nv = {"cl": nav_t, "cl_navshuf": nav_s, "cl_nonav": None,
                          "cl_oraclegoal": nav_t}[arm]
                    gf = goal_oracle if arm == "cl_oraclegoal" else None
                    tp = time.time()
                    res = model.plan(feats, v0=v0, nav_cmd=nv, plan_cfg=pc,
                                     goal_field=gf)
                    if t_plan_first is None:
                        t_plan_first = time.time() - tp
                    plans[arm] = res
            # -- bank the window ---------------------------------------------
            acc["g"].append(g.float().cpu().numpy())
            acc["ol"].append(ol.float().cpu().numpy())
            acc["ha"].append(ha.float().cpu().numpy())
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
            dec.setdefault("ha_controls", []).append(
                hold[None].expand(k, 2).float().cpu().numpy()[None])
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
                 "window_stride": stride, "n_windows": n_done,
                 "n_episodes": len(episodes_manifest),
                 "n_episodes_available": len(episode_names(a.cache))},
        "arms": arms, "tiers": {x: ARM_TIERS[x] for x in arms},
        "arm_meaning": {x: ARM_MEANING[x] for x in arms},
        "plan_cfg": dataclasses.asdict(pc),
        "plan_source_names": list(PLAN_SOURCE_NAMES),
        "nav_shuffle": shuf_stats,
        "speed_channel": {
            "enabled": bool(getattr(cfg, "speed_channel", False)),
            "rule": ("derived by RefAV1.augment_actions in forward() AND plan(): "
                     "v_k = v0 + sum_{j<k} a_j dt, scaled by SPEED_SCALE_MPS; "
                     "this adapter passes 2-wide (a, kappa) + measured v0 and "
                     "never widens actions (the model refuses a 3-wide input)")},
        "hold_action_rule": hold_action_controls.__doc__,
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
# the analysis (CPU)                                                            #
# --------------------------------------------------------------------------- #
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


def analyze_refav1(dump_dir: str, *, n_boot: int = 2000, seed: int = 0,
                   dt: float = DT, tiers: dict | None = None) -> dict:
    """``t1_eval.analyze`` on the trajectory dump + the refav1 sidecar analysis."""
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
    pairs = [(x, y, nm) for x, y, nm in (
        ("ol", "cl", "paired_closed_minus_open"),
        ("ha", "cl", "paired_cl_minus_ha"),
        ("cl_navshuf", "cl", "paired_cl_minus_navshuf"),
        ("cl_nonav", "cl", "paired_cl_minus_nonav"),
        ("cl_oraclegoal", "cl", "paired_cl_minus_oraclegoal"))
        if x in arms and y in arms]
    rec = t1.analyze(files, tiers={x: tiers[x] for x in arms if x in tiers},
                     n_boot=n_boot, seed=seed, dt=dt, paired=pairs)
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
    comps = {x: _components(np.concatenate(P_all[x]), G_all, dt) for x in arms}
    ref = {"n_windows": N, "n_episodes": len(files), "tiers": {x: tiers[x] for x in arms},
           "arm_meaning": {x: ARM_MEANING.get(x) for x in arms},
           "_tier_doctrine": rec["_tier_doctrine"],
           "families_paired": {nm: _paired_families(comps, x, y, eid_w, tiers, n_boot, seed)
                               for x, y, nm in pairs},
           "families_note": (
               "LONGITUDINAL / LATERAL / ADE and the TRAJECTORY-DERIVED tactical "
               "rows per arm live in rec['arms'][arm]['four_families'] (t1_eval, "
               "unchanged). STRATEGIC and the DECLARED tactical decisions come "
               "from refav1's own heads and live below (rec['refav1']); the "
               "trajectory-only STRATEGIC row in rec['arms'] stays UNAVAILABLE by "
               "design because a route class cannot be read off a 2 s path.")}

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
            "_reading": ("a baseline_won_frac near 1.0 with goal_source=none means "
                         "the T1 trajectory IS a floor baseline by construction "
                         "(no goal term in the cost: cv / hold_v0 / decel_1.5 all "
                         "score 0 and the LAST tie wins — MEASURED decel_1.5) — "
                         "not planning skill; see the module docstring and "
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
        "goal_source": ("none — plan() goal_field=None (no trained inference-time "
                        "goal source exists); cl_oraclegoal (T0) when present"),
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
def main(argv=None):
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
    ap.add_argument("--nav-shuffle-seed", type=int, default=0)
    ap.add_argument("--no-navshuf", action="store_true",
                    help="skip the nav-shuffle T1 arm (⛔ then the record is "
                         "NOT admissible for any nav-conditioned claim)")
    ap.add_argument("--with-nonav-arm", action="store_true")
    ap.add_argument("--with-oracle-goal-arm", action="store_true",
                    help="ALSO roll cl_oraclegoal (T0: the true future field as "
                         "the planning goal)")
    ap.add_argument("--allow-nonstrict", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", default="", help="extra tier stamps name=T0|T1")
    a = ap.parse_args(argv)

    if not a.out:
        sys.exit("--out is required")
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")
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
                         tiers=t1._parse_tiers(a.tiers))
    rec.update({"arm": a.arm, "ckpt": a.ckpt, "dump_dir": dump_dir,
                "mode": "analyze-only" if a.analyze_only else "rollout+analyze",
                "_unverified": _UNVERIFIED_ON_REAL_CKPT})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    _p(f"[out] {a.out}")
    for arm, blk in rec["arms"].items():
        ivl = blk["intervals"]["metrics"]
        ade = ivl.get("ade_dense_m", {})
        _p(f"  {arm:14s} tier={blk['tier']}  ADE={ade.get('mean')} "
           f"[{ade.get('lo')}, {ade.get('hi')}]  families_unavailable="
           f"{blk['four_families']['_families_unavailable']}")
    r = rec.get("refav1", {})
    for arm, pl in (r.get("planner") or {}).items():
        _p(f"  planner[{arm}] baseline_won_frac={pl['baseline_won_frac']} "
           f"sources={pl['source_fractions']}")
    s = r.get("strategic", {})
    _p(f"  strategic nav_valid_frac={s.get('nav_valid_frac')} "
       f"n_labeled={s.get('n_route_labeled')}")


if __name__ == "__main__":
    main()
