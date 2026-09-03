#!/usr/bin/env python3
"""paired_openloop.py — the PAIRED CROSS-MODEL **OPEN-LOOP** comparison of two
banked arm dumps (built for refav1 `cl` vs refcv3 `os`).

⛔ **VOCABULARY (PI ruling 2026-09-02, `Project Steering/VOCABULARY.md`).** Every arm this
tool touches is **OPEN LOOP**: the model is not controlling the vehicle, and its trajectory
does not affect the ego data it is next fed. refav1's `cl` (its predictor consuming its own
planner's actions) is open loop; refcv3's `os` (one forward pass) is open loop. ⛔ The words
"closed loop" never appear in this tool's output. CLOSED LOOP would require a simulator that
re-renders (AlpaSim) or a real vehicle, and we have published no such number.

⭐ **WHY THIS TOOL EXISTS, AND WHAT MAKES THE COMPARISON HONEST.** refav1 is a latent world
model + iCEM planner that rolls out under a 3-channel action; refcv3 is a supervised ONE-SHOT
128-anchor trajectory model with no action input, no rollout and no per-step decode. They are
not the same kind of system. `D-HF-COMPARABILITY` (GOALS_AND_CLAIMS.md) therefore binds the
admissible claim to **the DIFFERENCE OF EACH ARM'S MARGIN OVER THE SAME TRIVIAL FLOOR (`ha0`),
per family, with a paired episode-cluster bootstrap over the shared episode set** — never
`cl` against `os` as levels. This tool computes exactly that, and refuses to print a number
when any of the register's six inadmissibility conditions holds.

THE SIX GATES (`D-HF-COMPARABILITY`), each a hard refusal, each with its measured value:

  G1  different window grids, unasserted        -> the common instants are derived in RAW
                                                   FRAMES and both sides are index-selected
  G2  refcv3's ORACLE-selected `traj` (`a_star` / `oracle_sel`) against refav1's arm
                                                -> `oracle_sel` is refused as a model arm
  G3  a cross-tier comparison                   -> the two arms' tier stamps must match
  G4  either arm degenerate on the trivial (or selection) profile
                                                -> the read is **VOID, not negative**
  G5  a dump built with the unrepaired steer->curvature reading
                                                -> `action_units` is read from the manifest
  G6  no shared floor                           -> `ha0` must exist (or be DERIVED from the
                                                   dumped `v0` through the programme's own
                                                   integrator) and must read its known value

⭐ **THE CONTROL THAT PROVES THE WINDOW KEY, AND IT IS THE WHOLE JOB.** The two dumps are
built by different tools, from different caches (DINOv3 fp8 tokens vs v2ep pixels), through
different GT functions (`metric_dynamics.gt_ego_waypoints` vs `refb_labels.waypoint_targets`).
They are joined on `(clip_id, RAW 10 Hz frame of the window origin)` — refav1's `ws` is a
cache index `t` whose RAW origin is `2t` (`refav1_arm.gt_waypoints`: `f0 = 2 * t`); refcv3's
`ws` is a PROVIDER index whose RAW origin is `ws + (n_stack - 1)`, read from its own manifest.
**The join is then PROVEN, not assumed: the two dumps' GT waypoints and their `v0` must agree
on the intersected windows at the common instants.** If they do not, the key is wrong and the
whole comparison is refused. (MEASURED 2026-09-03 on 12 shared windows: GT `max|A-B| = 0.0`
and `v0 max|A-B| = 0.0`, **exactly** — two independent pipelines landing bit-for-bit.)

⭐ **THE FLOOR IS THE ANCHOR OF THE TABLE.** `ha0` is constant velocity at the measured `v0`
(`a = 0, kappa = 0`), so it depends on nothing but `v0` and the integrator — which is why it is
the ONE arm that is bit-comparable across two architectures. Three controls run on it and all
three print EXPECTED beside MEASURED:

  C1  the derived floor vs the analytic no-information value `(v0 * t, 0)`  -> must be ~0
  C2  side A's floor vs side B's floor on the shared windows                -> must be ~0
  C3  the derived floor vs a DUMPED `ha0`, where one exists                 -> must be ~0

⚠️ **THE ALGEBRAIC IDENTITY IS STATED, NOT HIDDEN.** When C2 passes, the floor is the same
array on both sides, so `(os - ha0) - (cl - ha0)` equals `os - cl` exactly. The margin framing
is still the correct one to REPORT — it is what stays interpretable when the floor is *not*
identical, and it is what the register binds — but a reader is owed the identity rather than
an implication that two different quantities were computed.

Estimator: FULL-SET pooled means; `taniteval.ci.paired_episode_cluster_bootstrap` on the
DIFFERENCE, over the shared episode set. ⛔ `overlapping_holdout_se` appears nowhere — it
biases the point estimate bidirectionally, up to a sign flip.

Usage (0 GPU — it reads banked dumps only):

    python taniteval/tools/paired_openloop.py \
      --a-dump <refav1 dump dir> --a-name refav1 --a-arm cl \
      --b-dump <refcv3 dump dir> --b-name refcv3 --b-arm os \
      --a-extra cl_navshuf --b-extra os_navzero \
      --out taniteval/results/paired-openloop-refav1-vs-refcv3-<UTC>.json \
      --md  <RESULT.md> --n-boot 2000 --seed 0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_TE_PARENT = os.path.join(_REPO, "taniteval")
_TE_PKG = os.path.join(_TE_PARENT, "taniteval")

try:                                                    # pragma: no cover
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                       # pragma: no cover
    pass


def _bootstrap_paths() -> None:
    """The namespace-package shadow eviction — identical to `refav1_arm`'s, and it
    must run before ANY `taniteval` import (the outer dir has no `__init__.py`)."""
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
        import taniteval.ci                     # noqa: F401  (the preflight)
        import taniteval.four_families          # noqa: F401
    except ModuleNotFoundError as ex:           # pragma: no cover
        sys.exit(f"[paired_openloop] taniteval preflight failed ({ex}). The real "
                 f"package is {_TE_PKG}. sys.path[:3]={sys.path[:3]}")


_bootstrap_paths()


def _load_sibling(name: str):
    """Import a SIBLING tool by file so its geometry is REUSED, never copied."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_for_paired", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ra = _load_sibling("refav1_arm")        # _components, hold_v0_controls, paths_from_controls

# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #
#: the corpus tick. Every grid in the programme is an integer multiple of it, so
#: RAW-FRAME OFFSETS (not seconds) are the exact currency for matching instants.
RAW_TICK_S = 0.1
#: GT must agree between two independently-built dumps to this, in metres. It is
#: NOT a fitted tolerance: the measured value on real dumps is 0.0 exactly.
GT_TOL_M = 1e-4
#: `v0` is `poses[.., 3]` on both sides, read from the same egomotion.
V0_TOL_MPS = 1e-4
#: the floor is integrated in float32 on one side and float64 here; 1e-4 m over a
#: 2 s horizon is ~5e-5 relative at this corpus's speeds.
FLOOR_TOL_M = 1e-4
#: trivial-profile thresholds — the same constants `refav1_arm` publishes.
TRIVIAL_STRAIGHT_M = ra.TRIVIAL_STRAIGHT_M
TRIVIAL_CONST_SPEED_M = ra.TRIVIAL_CONST_SPEED_M
TRIVIAL_IDENTICAL_M = ra.TRIVIAL_IDENTICAL_M
#: an arm that is constant-velocity on more than this fraction of the shared
#: windows makes the read VOID, not negative (`D-REFAV1-PAIRED-READ-VOID`).
DEGENERATE_FRAC = 0.5
#: an anchor model whose modal selection covers more than this share of windows is
#: degenerate in the SELECTION sense, which the trivial profile is blind to.
DEGENERATE_SELECTION_FRAC = 0.9
#: ⛔ never a model arm. refcv3's `a_star` picks the anchor nearest the GROUND TRUTH.
ORACLE_ARMS = ("oracle_sel", "cl_oraclegoal")

#: metric key -> (family, lower_is_better, unit). The four binding families plus
#: ADE/FDE, which is one ROW of four families and never "the result".
TRAJ_METRICS = [
    ("ade_m",                  "ADE",          True,  "m"),
    ("fde_m",                  "ADE",          True,  "m"),
    ("LON_speed_mae_mps",      "LONGITUDINAL", True,  "m/s"),
    ("LON_along_mae_m",        "LONGITUDINAL", True,  "m"),
    ("LON_accel_mae_mps2",     "LONGITUDINAL", True,  "m/s^2"),
    ("LAT_cross_mae_m",        "LATERAL",      True,  "m"),
    ("LAT_heading_mae_deg",    "LATERAL",      True,  "deg"),
    ("LAT_yaw_rate_mae_radps", "LATERAL",      True,  "rad/s"),
    ("TAC_traj_lat_correct",   "TACTICAL",     False, "acc"),
    ("TAC_traj_lon_correct",   "TACTICAL",     False, "acc"),
]
#: The DECLARED decision heads, read from each dump's own sidecar. They are what
#: gives this table a STRATEGIC family at all — a trajectory-only dump leaves it
#: UNAVAILABLE by construction (`four_families._decision_family` needs
#: `route_pred`/`route_gt`, which the `t1_eval` window whitelist never forwards).
#:
#: ⭐ THE FLOOR FOR A CLASSIFIER IS NOT `ha0`. `ha0` is a trajectory and has no
#: decision head, so a margin against it is undefined. The no-information value for
#: a categorical head is the **MAJORITY-CLASS rate on the same windows** — and like
#: `ha0` it is SHARED across the two models by construction, because it is a
#: property of the labels alone. That is what makes the margin framing survive into
#: the two decision families.
#:
#: name -> (family, label key, {suffix: prediction key})
DECISION_SPECS = [
    ("TAC_declared_lat", "TACTICAL", "lat_label",
     {"": "lat_pred_nav_true", "_navshuf": "lat_pred_nav_shuffled",
      "_navzero": "lat_pred_nav_zero"}),
    ("TAC_declared_lon", "TACTICAL", "lon_label",
     {"": "lon_pred_nav_true", "_navshuf": "lon_pred_nav_shuffled",
      "_navzero": "lon_pred_nav_zero"}),
    ("STR_route", "STRATEGIC", "route_label",
     {"": "route_pred_nav_true", "_navshuf": "route_pred_nav_shuffled",
      "_navzero": "route_pred_nav_zero"}),
]
#: below this many independent clusters an episode-cluster bootstrap is reporting
#: almost nothing: the resample can only ever draw from this many distinct episodes.
MIN_EPISODES_FOR_POWER = 10
FAMILIES = ["ADE", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"]


def _p(*a):
    print(*a, flush=True)


def _refused(reason, n=0, **kw):
    d = {"status": "REFUSED", "reason": reason, "n": int(n),
         "estimator": "n/a — inputs missing (WORK ITEM, not a pass)"}
    d.update(kw)
    return d


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


#: ⛔ THE CLOSED-LOOP HALF OF THE PI's QUESTION. It is NOT measured by this tool and
#: this tool cannot measure it — but the harness EXISTS, so "closed loop" here is a
#: NOT-YET-RUN, not a NOT-POSSIBLE, and the record must say which.
#: VERIFIED at source 2026-09-03: `closedloop_drive.py` steps a kinematic bicycle from
#: the model's own (steer, accel) — `dyaw = v/WHEELBASE*tan(steer)*DT`, `T_new = T_ego @ D`,
#: `v += accel*DT` — and then RE-RENDERS the next observation from the resulting pose
#: (`transport.render(T_ego @ Ts_cam, ...)`). The next observation is therefore a
#: consequence of the model's own output, which is exactly the PI's definition.
CLOSED_LOOP_STATUS = {
    "measured_here": False,
    "why_not": ("this tool reads OPEN-LOOP dumps: banked `[N, K, 2]` trajectories "
                "scored against recorded GT. Nothing in a dump can become a "
                "closed-loop number — the world never responded."),
    "the_harness_exists": "stack/experiments/alpasim-gsplat/closedloop_drive.py",
    "what_it_does": ("steps a kinematic bicycle from the model's own (steer, accel) "
                     "and re-renders the next observation from the resulting ego "
                     "pose (closedloop_drive.py:518-533) — so the next observation "
                     "IS a consequence of the model's output, which is the PI's "
                     "definition of closed loop"),
    "already_published_on": ("flagship v1 vs REF-C base — 9 starts x 50 ticks on "
                             "Thor, 437 paired windows, paired episode-cluster "
                             "bootstrap"),
    "for_these_arms": {
        "refcv3": "NOT YET RUN — ESTIMATED ~1.5 engineer-days + ~1 GPU-hour, gated "
                  "on Thor freeing (INHERITED estimate, not measured here)",
        "refav1": "NOT YET RUN — ESTIMATED 1-2 weeks, blocked on the action-unit "
                  "contract decision (INHERITED estimate, not measured here)"},
    "reading_rule": ("⛔ do not present the open-loop table below as the answer to "
                     "'open AND closed loop'. It is the open-loop half."),
}


def _corpus_identity(A, B, a_cfg: str | None, b_cfg: str | None) -> dict:
    """⛔ ARE THE TWO RUNS EVEN ON THE SAME CORPUS? Read from each run's OWN config.

    ⚠️ Two arms both called "b1-v72" does NOT establish they saw the same episodes.
    MEASURED 2026-09-03: refcv3's `config.json` records `v2_parity.parity false`,
    `checked false`, `corpus_key null` — the run is NON-PARITY by its own record —
    while refav1's trainer publishes no parity field at all. Neither can be shown
    identical from its config, and this block says so rather than implying it.

    ⭐ What IS established, and far more strongly than any config field, is the EVAL
    side: the window-key proof shows the two dumps' GT and `v0` are bit-identical on
    every shared window. That settles "the same moments of the same clips are being
    scored". It says nothing about TRAIN overlap or leak.
    """
    def read(dump, cfg_path):
        out = {"train_config": cfg_path, "published": {}, "missing": []}
        mod = dump.manifest.get("model", {}) or {}
        tl = mod.get("train_labels_manifest")
        if isinstance(tl, dict):
            out["published"]["train_labels_md5"] = tl.get("md5")
            out["published"]["train_labels_n_records"] = tl.get("n_records")
            out["published"]["train_labels_path"] = tl.get("path")
        else:
            out["missing"].append("train_labels_manifest (not in the dump manifest)")
        cfg = None
        if cfg_path and os.path.isfile(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as fh:
                    cfg = json.load(fh)
            except Exception as ex:                       # pragma: no cover
                out["config_read_error"] = str(ex)
        if cfg is None:
            out["missing"].append("the run's own config.json was not supplied "
                                  "(--a-run-config / --b-run-config)")
            return out
        par = cfg.get("v2_parity")
        if isinstance(par, dict):
            out["published"].update({
                "parity": par.get("parity"), "parity_checked": par.get("checked"),
                "corpus_key": par.get("corpus_key"),
                "clips_present": par.get("clips_present"),
                "train_cache_dirs": par.get("cache_dirs")})
        else:
            out["missing"].append("v2_parity block (this trainer publishes none)")
        args = cfg.get("args") if isinstance(cfg.get("args"), dict) else None
        if args:
            for k, lbl in (("cache", "train_cache"), ("episodes", "train_episodes"),
                           ("labels", "train_labels_path"), ("nav", "train_nav_path")):
                if k in args:
                    out["published"].setdefault(lbl, args[k])
        if "v2_cache" in cfg:
            out["published"].setdefault("train_cache", cfg["v2_cache"])
        return out

    ra_, rb_ = read(A, a_cfg), read(B, b_cfg)
    key_a = ra_["published"].get("corpus_key")
    key_b = rb_["published"].get("corpus_key")
    checked = bool(ra_["published"].get("parity_checked")) and \
        bool(rb_["published"].get("parity_checked"))
    if checked and key_a and key_b and key_a == key_b:
        status, reason = "VERIFIED_IDENTICAL", f"both runs carry a checked corpus_key {key_a}"
    elif key_a and key_b and key_a != key_b:
        status, reason = "DIFFER", f"corpus_key {key_a} vs {key_b}"
    else:
        status = "UNVERIFIED"
        reason = ("neither run publishes a CHECKED corpus key on both sides, so the "
                  "TRAIN corpora cannot be shown identical from the configs. "
                  + (f"A: {ra_['published'].get('parity', 'no parity field')}"
                     f" / checked={ra_['published'].get('parity_checked')}"
                     f" / key={key_a}. ")
                  + (f"B: {rb_['published'].get('parity', 'no parity field')}"
                     f" / checked={rb_['published'].get('parity_checked')}"
                     f" / key={key_b}."))
    return {
        "status": status, "reason": reason, "A": ra_, "B": rb_,
        "what_is_established_instead": (
            "the EVAL side, by the window-key proof: the two dumps' GT and v0 are "
            "bit-identical on every shared window, so both arms are scored on the "
            "SAME moments of the SAME clips."),
        "what_is_NOT_established": (
            "⛔ that the two runs TRAINED on the same episodes, and ⛔ that either "
            "run's train split is disjoint from these 20 eval clips. Both are "
            "properties of the RUNS, not of this adapter, and neither can be read "
            "from a dump. The probe that settles them is an episode-id set "
            "intersection between each run's train cache and this eval split "
            "(REFCV3_ARM.md GATE 3)."),
    }


# --------------------------------------------------------------------------- #
# 1. reading a dump                                                            #
# --------------------------------------------------------------------------- #
class Dump:
    """One banked arm dump + everything needed to place its windows on the corpus.

    ⛔ `raw_rule` is how `ws` becomes a RAW 10 Hz frame index. It is DERIVED from the
    dump's own manifest, never guessed — and it is then PROVEN by the GT identity
    control, which is the only thing that can actually settle it.
    """

    def __init__(self, path: str, name: str):
        self.path = os.path.abspath(path)
        self.name = name
        mf = os.path.join(self.path, "manifest.json")
        if not os.path.isfile(mf):
            raise SystemExit(f"[paired_openloop] no manifest.json under {self.path}")
        with open(mf, encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.tool = str(self.manifest.get("tool", "")).lower()
        g = self.manifest.get("grid", {})
        self.grid = g
        self.arms = list(self.manifest.get("arms", []))
        self.tiers = dict(self.manifest.get("tiers", {}))

        # ---- the instant grid, in RAW FRAME OFFSETS (exact integers) --------- #
        if "horizons_steps" in g:                       # refcv3-shaped manifest
            self.raw_offsets = [int(x) for x in g["horizons_steps"]]
            self.dt_s = float(g.get("dt_s"))
        else:                                           # refav1-shaped manifest
            self.dt_s = float(g["dt_s"])
            k = int(g["horizon_k"])
            step = int(round(self.dt_s / RAW_TICK_S))
            if abs(step * RAW_TICK_S - self.dt_s) > 1e-9:
                raise SystemExit(
                    f"[paired_openloop] {name}: dt {self.dt_s}s is not an integer "
                    f"multiple of the {RAW_TICK_S}s corpus tick — the two grids "
                    f"cannot be matched in RAW frames; refusing")
            self.raw_offsets = [step * (j + 1) for j in range(k)]
        self.k = len(self.raw_offsets)

        # ---- ws -> RAW frame of the window origin ---------------------------- #
        frames = (self.manifest.get("corpus", {}) or {}).get("frames", {}) or {}
        if "provider_to_raw_frame_offset" in frames:
            off = int(frames["provider_to_raw_frame_offset"])
            self.raw_rule = ("ws_plus_offset", off)
            self.raw_rule_note = (
                f"RAW frame = ws + {off} — the manifest's own "
                f"corpus.frames.provider_to_raw_frame_offset "
                f"(v2_dataset.py:36-38, the provider stores poses[n_stack-1:])")
        else:
            # refav1: the cache is a 5 Hz view of a 10 Hz corpus. `gt_waypoints`
            # (refav1_arm.py) opens with `f0 = 2 * t`, so the window origin's RAW
            # frame is 2*ws, and `join_lead_block` keys on exactly that.
            self.raw_rule = ("two_ws", 0)
            self.raw_rule_note = (
                "RAW frame = 2 * ws — refav1_arm.gt_waypoints reads `f0 = 2 * t` "
                "and join_lead_block keys on `(clip_id, RAW frame 2t)`. NOT read "
                "from the manifest (it carries no offset field); PROVEN instead by "
                "the GT identity control below")

        # ---- episodes + arrays ----------------------------------------------- #
        self.eps, self.dec = [], {}
        for e in self.manifest.get("episodes", []):
            fi = int(e["file_index"])
            f = os.path.join(self.path, f"ep{fi:03d}.npz")
            if not os.path.isfile(f):
                continue
            z = np.load(f)
            self.eps.append((e, {k2: z[k2] for k2 in z.files}))
            df = os.path.join(self.path, "decisions", f"ep{fi:03d}.npz")
            if os.path.isfile(df):
                zd = np.load(df)
                self.dec[fi] = {k2: zd[k2] for k2 in zd.files}
        if not self.eps:
            raise SystemExit(f"[paired_openloop] {name}: no ep*.npz under {self.path}")

        # ---- the window key --------------------------------------------------- #
        self.key_to_row = {}
        self.n_windows = 0
        for e, z in self.eps:
            clip = str(e.get("clip_id") or e.get("name"))
            ws = np.asarray(z["ws"]).astype(int).ravel()
            for i, w in enumerate(ws):
                self.key_to_row[(clip, self._raw(int(w)))] = (int(e["file_index"]), i)
            self.n_windows += int(ws.size)
        self.clips = sorted({c for c, _ in self.key_to_row})
        self._by_fi = {int(e["file_index"]): z for e, z in self.eps}
        self._ep_by_fi = {int(e["file_index"]): e for e, z in self.eps}

    def _raw(self, w: int) -> int:
        kind, off = self.raw_rule
        return 2 * w if kind == "two_ws" else w + off

    def arm_paths(self, arm: str, rows, slots):
        """[n, len(slots), 2] for the named arm on the given (file_index, row) pairs."""
        out = np.empty((len(rows), len(slots), 2), dtype=np.float64)
        for i, (fi, r) in enumerate(rows):
            out[i] = np.asarray(self._by_fi[fi][arm])[r][slots]
        return out

    def scalar(self, key: str, rows):
        return np.array([float(np.asarray(self._by_fi[fi][key]).ravel()[r])
                         for fi, r in rows], dtype=np.float64)

    def dec_col(self, key: str, rows):
        """A per-window column from the decisions sidecar, or None when absent."""
        vals = []
        for fi, r in rows:
            d = self.dec.get(fi)
            if d is None or key not in d:
                return None
            vals.append(np.asarray(d[key]).ravel()[r])
        return np.array(vals)

    def provenance(self) -> dict:
        m = self.manifest
        mod = m.get("model", {}) or {}
        return {
            "name": self.name, "dump": self.path, "tool": m.get("tool"),
            "arms": self.arms, "tiers": self.tiers,
            "n_windows_in_dump": self.n_windows, "n_clips_in_dump": len(self.clips),
            "grid": {"dt_s": self.dt_s, "k": self.k,
                     "raw_frame_offsets": self.raw_offsets,
                     "window_stride": self.grid.get("window_stride")},
            "ws_to_raw_frame": self.raw_rule_note,
            "action_units": m.get("action_units"),
            "model": {k: mod.get(k) for k in
                      ("ckpt", "step", "arm", "hier", "n_anchors", "horizons",
                       "tac_vocab_version", "target_space", "decoder_steps",
                       "trainable_parameters", "state_dict_load")
                      if k in mod},
            "tier_ruling": m.get("tier_ruling"),
            "absent_arms": m.get("absent_arms"),
        }


# --------------------------------------------------------------------------- #
# 2. the shared floor — three controls, EXPECTED printed beside MEASURED       #
# --------------------------------------------------------------------------- #
def derive_ha0(v0: np.ndarray, dt: float, k: int, slots) -> np.ndarray:
    """`ha0` from the measured `v0` through the PROGRAMME'S OWN integrator.

    ⭐ This is a DERIVATION, not a re-implementation: `refav1_arm.hold_v0_controls`
    supplies the zero controls and `refav1_arm.paths_from_controls` integrates them
    with `refa_v1_plan.unicycle_paths` — the one unicycle in the programme. It exists
    because the banked refav1 dumps predate the `ha0` arm, and `ha0` is the ONE arm
    that can be recovered exactly without the model: it consumes nothing but `v0`.
    """
    ctrl = ra.hold_v0_controls(k)
    out = np.empty((v0.size, len(slots), 2), dtype=np.float64)
    for i, v in enumerate(v0):
        p = ra.paths_from_controls(ctrl, float(v), float(dt), int(k))
        out[i] = np.asarray(p[0].detach().cpu().numpy(), dtype=np.float64)[slots]
    return out


def floor_controls(name: str, floor: np.ndarray, v0: np.ndarray,
                   instants_s) -> dict:
    """C1 — the constant-only control must read the NO-INFORMATION VALUE EXACTLY.

    A constant-velocity plan at `v0` is `x_k = v0 * t_k`, `y_k = 0`. That is not an
    approximation to check against; it is what the arm IS. The record prints the
    expectation beside the measurement so a reader never has to take it on trust.
    """
    t = np.asarray(instants_s, dtype=np.float64)
    exp = np.stack([v0[:, None] * t[None, :], np.zeros((v0.size, t.size))], axis=-1)
    dx = float(np.abs(floor[..., 0] - exp[..., 0]).max()) if v0.size else float("nan")
    dy = float(np.abs(floor[..., 1]).max()) if v0.size else float("nan")
    return {"control": "constant-only (no-information) value",
            "expected": "x_k = v0 * t_k, y_k = 0 exactly — a straight line at the "
                        "measured v0; it depends on NOTHING the model produces",
            "measured_max_abs_dx_m": dx, "measured_max_abs_y_m": dy,
            "tol_m": FLOOR_TOL_M,
            "pass": bool(np.isfinite(dx) and dx <= FLOOR_TOL_M
                         and np.isfinite(dy) and dy <= FLOOR_TOL_M),
            "n": int(v0.size), "side": name}


# --------------------------------------------------------------------------- #
# 3. profiles — SHAPE before any metric                                        #
# --------------------------------------------------------------------------- #
def trivial_profile(paths: dict) -> dict:
    """Per arm on the SHARED windows: straight / constant-speed / both, and which
    other arms it is bit-identical to. ⛔ Printed and evaluated BEFORE any family
    row: a family table answers *how far off*, never *what shape*."""
    out = {"rule": {"straight_m": TRIVIAL_STRAIGHT_M,
                    "const_speed_m": TRIVIAL_CONST_SPEED_M,
                    "identical_m": TRIVIAL_IDENTICAL_M,
                    "degenerate_frac": DEGENERATE_FRAC},
           "arms": {}, "degenerate_arms": []}
    names = list(paths)
    for a in names:
        P = paths[a]
        n = P.shape[0]
        if n == 0:
            out["arms"][a] = {"n": 0}
            continue
        straight = np.abs(P[..., 1]).max(axis=1) < TRIVIAL_STRAIGHT_M
        # chord length per step, with the origin prepended (the arm starts at ego)
        Z = np.concatenate([np.zeros((n, 1, 2)), P], axis=1)
        ds = np.linalg.norm(np.diff(Z, axis=1), axis=-1)
        const = (ds.max(axis=1) - ds.min(axis=1)) < TRIVIAL_CONST_SPEED_M
        both = straight & const
        ident = {}
        for b in names:
            if b == a:
                continue
            ident[b] = int((np.abs(P - paths[b]).max(axis=(1, 2))
                            < TRIVIAL_IDENTICAL_M).sum())
        out["arms"][a] = {"n": int(n),
                          "straight_frac": round(float(straight.mean()), 4),
                          "const_speed_frac": round(float(const.mean()), 4),
                          "constant_velocity_frac": round(float(both.mean()), 4),
                          "identical_to": ident}
        if float(both.mean()) > DEGENERATE_FRAC:
            out["degenerate_arms"].append(a)
    out["note"] = (
        "an arm listed in degenerate_arms is a CONSTANT-VELOCITY plan on more than "
        f"{DEGENERATE_FRAC:.0%} of the shared windows: read its LATERAL rows as a "
        "control, never as planning skill, and treat any comparison it enters as "
        "VOID rather than negative (D-REFAV1-PAIRED-READ-VOID)")
    out["identical_to_caveat"] = (
        f"⚠️ `identical_to` compares raw arrays at {TRIVIAL_IDENTICAL_M:g} m. Two "
        f"arms produced by DIFFERENT arithmetic paths — a float32 integrator against "
        f"a float64 reference, or two forwards at different batch sizes — sit at "
        f"~1e-6 m of each other even when they are the same quantity, so a zero here "
        f"is NOT proof that two such arms differ. It resolves only arms built through "
        f"the same path. The VOID gate does not depend on it: it is decided on "
        f"`constant_velocity_frac`, which is a property of ONE arm's own shape.")
    return out


def selection_profile(d: Dump, rows) -> dict:
    """⭐ The gate the trivial profile is BLIND to. An anchor model can select ONE
    anchor on every window — maximally degenerate — while every trajectory it emits
    is distinct, so `trivial_frac` reads 0.0000 and a family table computed over
    "always anchor #k, refined" gets read as scene understanding."""
    idx = d.dec_col("sel_idx", rows)
    if idx is None:
        return {"status": "ABSENT", "reason": f"{d.name}'s sidecar carries no "
                                              f"`sel_idx` (it is not an anchor model)"}
    idx = idx.astype(int)
    vals, cnt = np.unique(idx, return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    modal = float(cnt.max() / cnt.sum())
    oracle = d.dec_col("sel_agrees_oracle", rows)
    out = {"n": int(idx.size), "n_distinct_selected": int(vals.size),
           "modal_anchor": int(vals[int(np.argmax(cnt))]),
           "modal_share": round(modal, 4),
           "entropy_nats": round(ent, 4),
           "degenerate": bool(modal > DEGENERATE_SELECTION_FRAC),
           "rule": {"degenerate_modal_share": DEGENERATE_SELECTION_FRAC}}
    if oracle is not None:
        out["agrees_with_oracle_frac"] = round(float(np.mean(oracle)), 4)
        out["oracle_note"] = ("the ORACLE anchor is `a_star`, the anchor nearest the "
                              "GROUND TRUTH. It is reported as the ceiling it is and "
                              "is NEVER an arm in this table (gate G2)")
    return out


# --------------------------------------------------------------------------- #
# 4. the paired machinery                                                      #
# --------------------------------------------------------------------------- #
def _boot_single(v, eid, n_boot, seed):
    from taniteval import ci as _ci
    keep = np.isfinite(v)
    if keep.sum() == 0:
        return _refused("no finite window")
    r = _ci.episode_cluster_bootstrap(v[keep], [e for e, k in zip(eid, keep) if k],
                                      n_boot=n_boot, seed=seed)
    r["n_dropped_nonfinite"] = int((~keep).sum())
    return r


def _boot_paired(a, b, eid, n_boot, seed):
    """`mean(a) - mean(b)`, resampling the SAME episodes in each draw."""
    from taniteval import ci as _ci
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() == 0:
        return _refused("no window has a finite value for both arms")
    r = _ci.paired_episode_cluster_bootstrap(
        a[keep], b[keep], [e for e, k in zip(eid, keep) if k],
        n_boot=n_boot, seed=seed)
    r["n_dropped_nonfinite"] = int((~keep).sum())
    return r


def _verdict(delta: float, sep: bool, lower_is_better: bool,
             pos: str, neg: str) -> str:
    if not sep:
        return "not separated"
    better = (delta < 0) if lower_is_better else (delta > 0)
    return pos if better else neg


# --------------------------------------------------------------------------- #
# 5. the run                                                                   #
# --------------------------------------------------------------------------- #
def run(a: argparse.Namespace) -> dict:
    A = Dump(a.a_dump, a.a_name)
    B = Dump(a.b_dump, a.b_name)
    rec = {"tool": "paired_openloop.py",
           "utc": _utc(),
           "question": ("the PI's 2026-09-02 question: compare the OPEN-LOOP "
                        "performance of refav1 and refcv3"),
           "loop_vocabulary": {
               "every arm here": "OPEN LOOP — the model is not controlling the "
                                 "vehicle; its trajectory does not affect the ego "
                                 "data it is next fed (PI ruling 2026-09-02)",
               "closed loop": "NOT MEASURED ANYWHERE IN THIS RECORD — it would "
                              "require AlpaSim or a real vehicle",
               "refav1 `cl`": "open loop: the predictor consumes the planner's own "
                              "actions, but the recorded eval ego data keeps arriving",
               "refcv3 `os`": "open loop: one forward pass at the window origin"},
           "inputs": {"A": A.provenance(), "B": B.provenance()},
           "closed_loop": dict(CLOSED_LOOP_STATUS),
           "gates": {}, "controls": {}, "void": False, "void_reasons": []}
    G = rec["gates"]
    rec["controls"]["corpus_identity"] = _corpus_identity(
        A, B, a.a_run_config, a.b_run_config)
    _p(f"[corpus] train-corpus identity: "
       f"{rec['controls']['corpus_identity']['status']} — "
       f"{rec['controls']['corpus_identity']['reason'][:150]}")

    # ---- G2: never an oracle arm ------------------------------------------- #
    for side, d, arm in (("A", A, a.a_arm), ("B", B, a.b_arm)):
        if arm in ORACLE_ARMS:
            raise SystemExit(
                f"[paired_openloop] ⛔ GATE G2: `{arm}` is an ORACLE arm (the "
                f"anchor/goal selected against the GROUND TRUTH). It may never be "
                f"the model arm in this table. Use the deployed selection "
                f"(`out['traj']` via sel_score_v3) — i.e. `os` for refcv3.")
    G["G2_oracle_arm_refused"] = {
        "pass": True,
        "checked": [a.a_arm, a.b_arm],
        "rule": "refcv3's `a_star`/`oracle_sel` picks the anchor nearest the GT; "
                "refav1's `cl_oraclegoal` reads the true future field as its goal. "
                "Neither may be a model arm here."}

    for side, d, arm in (("A", A, a.a_arm), ("B", B, a.b_arm)):
        if arm not in d.arms:
            raise SystemExit(f"[paired_openloop] {d.name}: arm `{arm}` is not in the "
                             f"dump (has {d.arms})")

    # ---- G3: same tier ------------------------------------------------------ #
    ta, tb = A.tiers.get(a.a_arm), B.tiers.get(a.b_arm)
    G["G3_same_tier"] = {"A": {"arm": a.a_arm, "tier": ta},
                         "B": {"arm": a.b_arm, "tier": tb},
                         "pass": bool(ta is not None and ta == tb),
                         "rule": "a cross-tier comparison is inadmissible "
                                 "(D-HF-COMPARABILITY condition 3)"}
    tr = B.manifest.get("tier_ruling") or A.manifest.get("tier_ruling")
    if tr:
        G["G3_same_tier"]["open_ruling"] = tr
    if not G["G3_same_tier"]["pass"]:
        raise SystemExit(f"[paired_openloop] ⛔ GATE G3: tier stamps differ "
                         f"({a.a_arm}={ta} vs {a.b_arm}={tb}) — refusing")

    # ---- G5: the action-unit contract -------------------------------------- #
    au = {}
    for side, d in (("A", A), ("B", B)):
        u = (d.manifest.get("action_units") or {})
        au[side] = {"recorded": u.get("recorded"), "applies_to": u.get("applies_to")}
    G["G5_action_units"] = {
        **au,
        "pass": True,
        "note": ("`actions[:,0]` in the corpus is a ROAD-WHEEL STEER ANGLE "
                 "(physicalai.py:621), not a curvature. The unit affects the `ha` "
                 "control only. ⭐ It CANNOT affect this table's floor: `ha0` is "
                 "exactly zero, which is 0 in either unit — that is precisely what "
                 "makes `ha0` the bit-comparable arm. A dump with "
                 "`recorded == 'kappa'` carries the C-REFCV3-ARM-SAME-DEFECT "
                 "over-rotation in its `ha` row and nowhere else."),
        "a_arm_affected": a.a_arm == "ha", "b_arm_affected": a.b_arm == "ha"}
    if au["A"]["recorded"] == "kappa" or au["B"]["recorded"] == "kappa":
        G["G5_action_units"]["warning"] = (
            "one side declares the LEGACY `kappa` reading of a recorded action; its "
            "`ha` row is over-rotated by ~L=2.9x. `ha0` and the model arms are "
            "unaffected.")

    # ---- G1: the common instant grid, in RAW FRAMES ------------------------- #
    common_raw = sorted(set(A.raw_offsets) & set(B.raw_offsets))
    slots_a = [A.raw_offsets.index(r) for r in common_raw]
    slots_b = [B.raw_offsets.index(r) for r in common_raw]
    instants = [round(r * RAW_TICK_S, 6) for r in common_raw]
    G["G1_common_grid"] = {
        "A_raw_frame_offsets": A.raw_offsets, "B_raw_frame_offsets": B.raw_offsets,
        "common_raw_frame_offsets": common_raw, "common_instants_s": instants,
        "A_slots_selected": slots_a, "B_slots_selected": slots_b,
        "A_slots_dropped": [x for x in A.raw_offsets if x not in common_raw],
        "B_slots_dropped": [x for x in B.raw_offsets if x not in common_raw],
        "pass": len(common_raw) >= 2,
        "rule": "the grids are matched in INTEGER RAW FRAMES, then index-selected "
                "on both sides. ⛔ There is no resampling and no interpolation.",
        "cost": ("the common grid is COARSER than either dump's own. Every LON/LAT "
                 "rate below is computed on it for BOTH arms and for GT, so the "
                 "comparison is fair — but the LEVELS are not comparable to a "
                 "finer-grid read of the same arm.")}
    if not G["G1_common_grid"]["pass"]:
        raise SystemExit(
            f"[paired_openloop] ⛔ GATE G1: the two grids share "
            f"{len(common_raw)} instant(s) ({instants}) — fewer than the 2 a "
            f"geometry needs. A: {A.raw_offsets}; B: {B.raw_offsets}. Re-roll one "
            f"side on a grid that shares instants with the other.")
    dt_c = (common_raw[1] - common_raw[0]) * RAW_TICK_S
    uniform = len({common_raw[i + 1] - common_raw[i]
                   for i in range(len(common_raw) - 1)}) == 1
    origin_ok = abs(common_raw[0] * RAW_TICK_S - dt_c) < 1e-9
    G["G1_common_grid"].update(
        {"common_dt_s": dt_c, "uniform": uniform, "first_step_equals_spacing": origin_ok,
         "geometry_note": ("`four_families._seq_geometry` PREPENDS the ego origin as "
                           "step 0, so a grid whose first instant differs from its "
                           "spacing makes step 0 span a different time from the rest. "
                           f"Here: first={common_raw[0] * RAW_TICK_S}s, "
                           f"spacing={dt_c}s.")})
    if not (uniform and origin_ok):
        raise SystemExit(
            f"[paired_openloop] ⛔ GATE G1: the common grid {instants} is not "
            f"uniform-from-the-origin (uniform={uniform}, "
            f"first_step_equals_spacing={origin_ok}); the prepended-origin geometry "
            f"would mean different things in step 0 and the rest. Refusing.")

    # ---- the window intersection -------------------------------------------- #
    common_keys = sorted(set(A.key_to_row) & set(B.key_to_row))
    if not common_keys:
        raise SystemExit(
            f"[paired_openloop] ⛔ the two dumps share ZERO windows on "
            f"(clip_id, RAW frame). A covers {A.n_windows} windows over "
            f"{len(A.clips)} clips; B covers {B.n_windows} over {len(B.clips)}. "
            f"The usual cause is a STRIDE MISMATCH: A's origins are "
            f"{sorted({r % 20 for _, r in list(A.key_to_row)[:200]})} mod 20 and "
            f"B's are {sorted({r % 20 for _, r in list(B.key_to_row)[:200]})} mod 20. "
            f"Re-roll one side at a stride that covers the other's origins.")
    rows_a = [A.key_to_row[k] for k in common_keys]
    rows_b = [B.key_to_row[k] for k in common_keys]
    clips = sorted({c for c, _ in common_keys})
    eid = [c for c, _ in common_keys]                 # the CLIP is the cluster
    n = len(common_keys)
    rec["intersection"] = {
        "key": "(clip_id, RAW 10 Hz frame index of the window origin t0)",
        "n_windows": n, "n_episodes": len(clips),
        "A_windows_available": A.n_windows, "B_windows_available": B.n_windows,
        "A_clips": len(A.clips), "B_clips": len(B.clips),
        "A_windows_dropped": A.n_windows - n, "B_windows_dropped": B.n_windows - n,
        "clips": clips,
        "note": ("the bootstrap resamples EPISODES (clips), because windows inside "
                 "one clip are strongly dependent. n_episodes, not n_windows, is "
                 "the sample size that matters."),
        "ws_to_raw": {"A": A.raw_rule_note, "B": B.raw_rule_note}}
    _p(f"[intersect] {n} shared windows over {len(clips)} clips "
       f"(A had {A.n_windows}/{len(A.clips)}, B had {B.n_windows}/{len(B.clips)})")

    # ---- THE CONTROL THAT PROVES THE KEY: GT and v0 identity ---------------- #
    gA = A.arm_paths("g", rows_a, slots_a)
    gB = B.arm_paths("g", rows_b, slots_b)
    v0A = A.scalar("v0", rows_a)
    v0B = B.scalar("v0", rows_b)
    d_gt = float(np.abs(gA - gB).max())
    d_v0 = float(np.abs(v0A - v0B).max())
    rec["controls"]["window_key_proof"] = {
        "control": "GT and v0 identity across two independently-built dumps",
        "expected": ("EXACTLY the same numbers. The two dumps read the same corpus "
                     "at the same instants through different loaders and different "
                     "GT functions; if the (clip, RAW frame) key is right they must "
                     "agree, and if it is wrong they cannot."),
        "measured_max_abs_gt_diff_m": d_gt, "gt_tol_m": GT_TOL_M,
        "measured_max_abs_v0_diff_mps": d_v0, "v0_tol_mps": V0_TOL_MPS,
        "n_windows": n,
        "pass": bool(d_gt <= GT_TOL_M and d_v0 <= V0_TOL_MPS)}
    _p(f"[control] window-key proof: GT max|A-B| = {d_gt:.3e} m (expected 0, tol "
       f"{GT_TOL_M:g}); v0 max|A-B| = {d_v0:.3e} m/s")
    if not rec["controls"]["window_key_proof"]["pass"]:
        raise SystemExit(
            f"[paired_openloop] ⛔ THE WINDOW KEY IS WRONG. GT differs by "
            f"{d_gt:.4e} m and v0 by {d_v0:.4e} m/s between the two dumps on the "
            f"windows they supposedly share. Every number downstream would be a "
            f"comparison of different moments in different clips. Refusing.")

    # ---- G6: the shared floor, with its three controls ---------------------- #
    floors, floor_src = {}, {}
    for side, d, rows, slots, v0 in (("A", A, rows_a, slots_a, v0A),
                                     ("B", B, rows_b, slots_b, v0B)):
        if a.floor in d.arms:
            floors[side] = d.arm_paths(a.floor, rows, slots)
            floor_src[side] = f"DUMPED — the `{a.floor}` arm written by {d.tool}"
        else:
            floors[side] = derive_ha0(v0, d.dt_s, d.k, slots)
            floor_src[side] = (
                f"DERIVED from the dumped `v0` through the programme's own "
                f"integrator (refav1_arm.hold_v0_controls -> paths_from_controls -> "
                f"refa_v1_plan.unicycle_paths) at this dump's own dt={d.dt_s}s, "
                f"K={d.k}, then index-selected. This dump predates the `{a.floor}` "
                f"arm; `{a.floor}` is the ONE arm recoverable without the model, "
                f"because it consumes nothing but v0.")
    c1a = floor_controls("A", floors["A"], v0A, instants)
    c1b = floor_controls("B", floors["B"], v0B, instants)
    d_ff = float(np.abs(floors["A"] - floors["B"]).max())
    c2 = {"control": "the floor is BIT-COMPARABLE across the two architectures",
          "expected": "0 — `ha0` depends on nothing but v0 and the integrator, and "
                      "v0 was just proven identical",
          "measured_max_abs_diff_m": d_ff, "tol_m": FLOOR_TOL_M,
          "pass": bool(d_ff <= FLOOR_TOL_M)}
    # C3 — SAME SIDE: my derivation against what that tool actually integrated.
    # ⚠️ It must be the same side, or C3 is C2 wearing a different label: with one
    # side derived and the other dumped, a cross-side check measures the same
    # quantity twice and would read as two independent controls when it is one.
    c3 = {"control": "the DERIVED floor against a DUMPED one, SAME SIDE",
          "status": "N/A",
          "reason": "no side carried BOTH a dumped floor and a derivable v0"}
    for side, d, rows, slots, v0 in (("A", A, rows_a, slots_a, v0A),
                                     ("B", B, rows_b, slots_b, v0B)):
        if a.floor in d.arms:
            dumped = d.arm_paths(a.floor, rows, slots)
            mine = derive_ha0(v0, d.dt_s, d.k, slots)
            dd = float(np.abs(dumped - mine).max())
            c3 = {"control": "the DERIVED floor against the DUMPED one, SAME SIDE "
                             f"({d.name})",
                  "expected": "0 — the derivation calls the same integrator the dump "
                              "called, on the dump's own v0",
                  "measured_max_abs_diff_m": dd, "tol_m": FLOOR_TOL_M,
                  "dumped_side": side, "pass": bool(dd <= FLOOR_TOL_M),
                  "why_same_side": ("with one side derived and the other dumped, a "
                                    "CROSS-side check would measure exactly what C2 "
                                    "measures — one control reported as two")}
            break
    rec["controls"]["floor_C1_no_information_value"] = {"A": c1a, "B": c1b}
    rec["controls"]["floor_C2_bit_comparable"] = c2
    rec["controls"]["floor_C3_derived_vs_dumped"] = c3
    G["G6_shared_floor"] = {
        "floor_arm": a.floor,
        "source": floor_src,
        "pass": bool(c1a["pass"] and c1b["pass"] and c2["pass"]
                     and c3.get("pass", True)),
        "rule": "both arms must report against the SAME trivial floor on the SAME "
                "windows (D-HF-COMPARABILITY condition 6)"}
    _p(f"[control] floor C1 (A): max|dx| = {c1a['measured_max_abs_dx_m']:.3e} m, "
       f"max|y| = {c1a['measured_max_abs_y_m']:.3e} m (expected 0.0 exactly)")
    _p(f"[control] floor C1 (B): max|dx| = {c1b['measured_max_abs_dx_m']:.3e} m, "
       f"max|y| = {c1b['measured_max_abs_y_m']:.3e} m (expected 0.0 exactly)")
    _p(f"[control] floor C2 bit-comparability: max|A-B| = {d_ff:.3e} m "
       f"(expected 0.0 exactly)")
    if "measured_max_abs_diff_m" in c3:
        _p(f"[control] floor C3 derived-vs-dumped: "
           f"max|d| = {c3['measured_max_abs_diff_m']:.3e} m (expected 0.0)")
    if not G["G6_shared_floor"]["pass"]:
        raise SystemExit("[paired_openloop] ⛔ GATE G6: the shared floor failed a "
                         "control. The harness is wrong, not the model — do not "
                         "reinterpret this as a result.")
    # ⭐ ONE floor array is used for BOTH margins, and C2 is what licenses that.
    FLOOR = floors["A"]
    rec["controls"]["floor_used"] = (
        "ONE array, side A's, used for BOTH margins — licensed by control C2 "
        f"(max|A-B| = {d_ff:.3e} m). ⚠️ THE ALGEBRAIC CONSEQUENCE, STATED RATHER "
        "THAN HIDDEN: with an identical floor, (B_arm - floor) - (A_arm - floor) "
        "equals (B_arm - A_arm) exactly. The margin framing is still the one "
        "reported, because it is what stays interpretable when the floor is NOT "
        "identical and it is what D-HF-COMPARABILITY binds — but a reader is owed "
        "the identity.")

    # ---- the arms ------------------------------------------------------------ #
    arms_a = [a.a_arm, *[x for x in a.a_extra if x in A.arms]]
    arms_b = [a.b_arm, *[x for x in a.b_extra if x in B.arms]]
    for x in a.a_extra:
        if x not in A.arms:
            _p(f"[warn] A has no arm `{x}` — skipped")
    for x in a.b_extra:
        if x not in B.arms:
            _p(f"[warn] B has no arm `{x}` — skipped")
    paths = {f"{A.name}:{x}": A.arm_paths(x, rows_a, slots_a) for x in arms_a}
    paths.update({f"{B.name}:{x}": B.arm_paths(x, rows_b, slots_b) for x in arms_b})
    paths[f"shared:{a.floor}"] = FLOOR
    GT = gA

    # ---- profiles FIRST ------------------------------------------------------ #
    tp = trivial_profile(paths)
    rec["profiles"] = {"trivial": tp}
    _p(f"[trivial-profile] {n} shared windows — arm SHAPE before any family row")
    for k2, v in tp["arms"].items():
        if not v.get("n"):
            continue
        ident = ", ".join(f"{b}={c}/{n}" for b, c in v["identical_to"].items() if c)
        _p(f"  {k2:28s} straight={v['straight_frac']:.4f} "
           f"const_speed={v['const_speed_frac']:.4f} "
           f"CONSTANT-VELOCITY={v['constant_velocity_frac']:.4f}"
           + (f"  identical_to: {ident}" if ident else ""))
    sp = {A.name: selection_profile(A, rows_a), B.name: selection_profile(B, rows_b)}
    rec["profiles"]["selection"] = sp
    for nm, s in sp.items():
        if s.get("status") == "ABSENT":
            continue
        _p(f"[selection-profile] {nm}: n_distinct={s['n_distinct_selected']} "
           f"modal #{s['modal_anchor']} at {s['modal_share']:.4f} "
           f"entropy={s['entropy_nats']:.4f}"
           + (f" agrees_with_oracle={s['agrees_with_oracle_frac']:.4f}"
              if "agrees_with_oracle_frac" in s else ""))

    model_arm_keys = [f"{A.name}:{a.a_arm}", f"{B.name}:{a.b_arm}"]
    for k2 in model_arm_keys:
        if k2 in tp["degenerate_arms"]:
            rec["void"] = True
            rec["void_reasons"].append(
                f"{k2} is a CONSTANT-VELOCITY plan on "
                f"{tp['arms'][k2]['constant_velocity_frac']:.1%} of the shared "
                f"windows — the instrument saw the trivial baseline, not the model. "
                f"The read is VOID, not negative (D-REFAV1-PAIRED-READ-VOID).")
    for nm, s in sp.items():
        if s.get("degenerate"):
            rec["void"] = True
            rec["void_reasons"].append(
                f"{nm}'s selection is degenerate: anchor #{s['modal_anchor']} on "
                f"{s['modal_share']:.1%} of the shared windows. A family table over "
                f"'always one anchor, refined' is not evidence about the model.")
    G["G4_profiles_non_degenerate"] = {
        "pass": not rec["void"],
        "degenerate_trivial": tp["degenerate_arms"],
        "degenerate_selection": [k2 for k2, s in sp.items() if s.get("degenerate")],
        "rule": "a degenerate arm makes the read VOID, not negative "
                "(D-HF-COMPARABILITY condition 4)"}
    if rec["void"]:
        _p("")
        for r in rec["void_reasons"]:
            _p(f"  ⛔ VOID: {r}")
        _p("  ⇒ every cross-model row below is computed and banked, but the "
           "COMPARISON IS VOID. Do not read it as 'model X beats model Y'.")

    # ---- trajectory components ------------------------------------------------ #
    comps = {k2: ra._components(P, GT, dt_c) for k2, P in paths.items()}

    # ---- the DECLARED heads: label control, then a MAJORITY floor -------------- #
    tv_a = (A.manifest.get("model", {}) or {}).get("tac_vocab_version")
    tv_b = (B.manifest.get("model", {}) or {}).get("tac_vocab_version")
    rec["controls"]["tactical_vocabulary"] = {
        "A": tv_a, "B": tv_b, "pass": bool(tv_a is not None and tv_a == tv_b),
        "rule": "the DECLARED tactical/strategic rows are only comparable when the "
                "two heads speak the same vocabulary AND read the same labels on "
                "the same windows"}
    vocab_ok = rec["controls"]["tactical_vocabulary"]["pass"]
    if not vocab_ok:
        _p(f"[warn] tac_vocab_version differs (A={tv_a}, B={tv_b}) — every DECLARED "
           f"row is refused")

    lab_ctl, decision_metrics, decision_masks = {}, [], {}
    floor_key = f"shared:{a.floor}"
    for base, fam, lkey, preds in DECISION_SPECS:
        la, lb = A.dec_col(lkey, rows_a), B.dec_col(lkey, rows_b)
        if la is None or lb is None:
            lab_ctl[lkey] = {"status": "ABSENT",
                             "reason": "at least one side's sidecar does not carry it",
                             "usable": False}
            continue
        la, lb = la.astype(int), lb.astype(int)
        both = (la >= 0) & (lb >= 0)
        agree = bool(np.all(la[both] == lb[both])) if both.any() else False
        ctl = {"n_shared_windows": int(la.size),
               "n_labelled_A": int((la >= 0).sum()),
               "n_labelled_B": int((lb >= 0).sum()),
               "n_labelled_BOTH": int(both.sum()),
               "agree_where_both_labelled": agree,
               "expected": ("wherever BOTH sides carry a label it must be the SAME "
                            "label — both derive it from the same v7.2 blob for the "
                            "same clip and the same frame"),
               "usable": bool(agree and both.sum() > 0 and vocab_ok)}
        if both.any() and not agree:
            dis = both & (la != lb)
            pairs, cnts = np.unique(
                np.stack([la[dis], lb[dis]], 1), axis=0, return_counts=True)
            ctl["disagreement"] = {
                "n_disagreeing": int(dis.sum()),
                "of_n_both_labelled": int(both.sum()),
                "A_label_to_B_label_counts": {f"{int(p[0])}->{int(p[1])}": int(c)
                                              for p, c in zip(pairs, cnts)},
                "meaning": ("⛔ this is NOT a coverage difference — on these windows "
                            "the two tools assign DIFFERENT labels to the SAME "
                            "(clip, RAW frame). Scoring both heads against 'the "
                            "label' would score two different questions, so the row "
                            "is REFUSED. ESCALATION, not a caveat: one of the two "
                            "derivations is wrong, or they are two different "
                            "quantities sharing a name."),
                "work_item": (f"reconcile `{lkey}` between refav1_arm.py and "
                              f"refcv3_arm.py; until then this family has no "
                              f"cross-model row")}
        if int((la >= 0).sum()) != int((lb >= 0).sum()):
            ctl["asymmetry_note"] = (
                f"the two tools do not label the same windows: A labels "
                f"{ctl['n_labelled_A']}, B labels {ctl['n_labelled_B']} of "
                f"{ctl['n_shared_windows']}. The comparison is therefore SCOPED to "
                f"the {ctl['n_labelled_BOTH']} windows BOTH label — an honest "
                f"restriction, not a silent drop. The asymmetry itself is a finding "
                f"about the two label derivations, not about either model.")
        lab_ctl[lkey] = ctl
        if not ctl["usable"]:
            continue
        decision_masks[base] = both
        lab = la.copy()
        # ⭐ the classifier's no-information value: the MAJORITY class on exactly
        # these windows. It depends on the LABELS alone, so it is shared across the
        # two models by construction — the decision-head analogue of `ha0`.
        vals, cnt = np.unique(lab[both], return_counts=True)
        maj = int(vals[int(np.argmax(cnt))])
        ctl["majority_class"] = maj
        ctl["majority_rate"] = round(float(cnt.max() / cnt.sum()), 4)
        floor_ok = np.full(la.size, np.nan)
        floor_ok[both] = (lab[both] == maj).astype(float)
        for suf, pkey in preds.items():
            mk = f"{base}_correct{suf}"
            decision_metrics.append((mk, fam, False, "acc"))
            # ⚠️ ONLY the model arm of each side carries the declared-head columns.
            # The heads are a property of the FORWARD PASS under a conditioning, not
            # of a trajectory arm: attaching them to `ha`/`ha0`/`os_navzero` too
            # would print the model's route accuracy in a control arm's row and read
            # as though the control had a head.
            for side_d, rows_side, key in ((A, rows_a, f"{A.name}:{a.a_arm}"),
                                           (B, rows_b, f"{B.name}:{a.b_arm}")):
                pv = side_d.dec_col(pkey, rows_side)
                if pv is None:
                    continue
                ok = np.full(la.size, np.nan)
                ok[both] = (pv.astype(int)[both] == lab[both]).astype(float)
                comps[key][mk] = ok
            comps[floor_key][mk] = floor_ok
    rec["controls"]["declared_label_identity"] = lab_ctl
    rec["controls"]["decision_floor"] = {
        "what": "the MAJORITY-CLASS predictor on the labelled shared windows",
        "why": ("`ha0` is a trajectory and carries no decision head, so a margin "
                "against it is undefined for a categorical row. The majority rate "
                "is the no-information value for a classifier and — like `ha0` — it "
                "is shared across the two models by construction, because it is a "
                "property of the LABELS alone."),
        "per_key": {k2: {"majority_class": v.get("majority_class"),
                         "majority_rate": v.get("majority_rate"),
                         "n": v.get("n_labelled_BOTH")}
                    for k2, v in lab_ctl.items() if v.get("usable")}}

    # ---- the table ------------------------------------------------------------ #
    all_metrics = list(TRAJ_METRICS) + decision_metrics
    ak, bk = model_arm_keys

    absolute, margins, cross = {}, {}, {}
    for key in list(paths):
        absolute[key] = {}
        for mk, fam, lib, unit in all_metrics:
            if mk not in comps[key]:
                continue
            r = _boot_single(np.asarray(comps[key][mk], dtype=np.float64), eid,
                             a.n_boot, a.seed)
            r.update({"family": fam, "unit": unit, "lower_is_better": lib})
            absolute[key][mk] = r
    for side_key in [k2 for k2 in paths if k2 != floor_key]:
        margins[side_key] = {}
        for mk, fam, lib, unit in all_metrics:
            if mk not in comps[side_key] or mk not in comps[floor_key]:
                margins[side_key][mk] = _refused(
                    f"`{mk}` is not defined for both `{side_key}` and the floor on "
                    f"these windows", n=n, family=fam)
                continue
            r = _boot_paired(np.asarray(comps[side_key][mk], dtype=np.float64),
                             np.asarray(comps[floor_key][mk], dtype=np.float64),
                             eid, a.n_boot, a.seed)
            r.update({"family": fam, "unit": unit, "lower_is_better": lib,
                      "direction": f"{side_key} - {floor_key}",
                      "verdict": _verdict(r.get("delta", float("nan")),
                                          r.get("separated", False), lib,
                                          f"{side_key} BEATS the floor",
                                          f"{side_key} LOSES to the floor")})
            margins[side_key][mk] = r

    for mk, fam, lib, unit in all_metrics:
        if mk not in comps[ak] or mk not in comps[bk] or mk not in comps[floor_key]:
            cross[mk] = _refused(f"`{mk}` is missing on at least one side", n=n,
                                 family=fam)
            continue
        mA = np.asarray(comps[ak][mk], dtype=np.float64) - np.asarray(comps[floor_key][mk], dtype=np.float64)
        mB = np.asarray(comps[bk][mk], dtype=np.float64) - np.asarray(comps[floor_key][mk], dtype=np.float64)
        r = _boot_paired(mB, mA, eid, a.n_boot, a.seed)
        r.update({"family": fam, "unit": unit, "lower_is_better": lib,
                  "statistic": f"({bk} - {floor_key}) - ({ak} - {floor_key})",
                  "direction": f"NEGATIVE favours {bk}" if lib
                               else f"POSITIVE favours {bk}",
                  "verdict": _verdict(r.get("delta", float("nan")),
                                      r.get("separated", False), lib,
                                      f"{bk} has the better margin",
                                      f"{ak} has the better margin")})
        if rec["void"]:
            r["VOID"] = True
            r["void_note"] = ("a degenerate arm entered this statistic — it is "
                              "banked for audit and must NOT be read as a result")
        cross[mk] = r

    # ---- ⭐ the nav-echo control: a route head fed its own nav is an ECHO ------ #
    # flagship v1's route head scored 1.0000 as an exact bijection of the nav it was
    # fed. `true - shuffled` withholds the PAIRING (the marginal is preserved) and
    # `true - zero` withholds the SIGNAL. They are not interchangeable, and a
    # nav-conditioned claim carrying neither is inadmissible (BACKLOG R39).
    echo = {}
    for base, fam, lkey, preds in DECISION_SPECS:
        if base not in decision_masks:
            continue
        for ctrl_suf, label in (("_navshuf", "true_minus_navshuffled"),
                                ("_navzero", "true_minus_navzero")):
            t_mk, c_mk = f"{base}_correct", f"{base}_correct{ctrl_suf}"
            per_side = {}
            for key in (ak, bk):
                if t_mk not in comps[key] or c_mk not in comps[key]:
                    continue
                r = _boot_paired(np.asarray(comps[key][t_mk], dtype=np.float64),
                                 np.asarray(comps[key][c_mk], dtype=np.float64),
                                 eid, a.n_boot, a.seed)
                r["verdict"] = _verdict(r.get("delta", float("nan")),
                                        r.get("separated", False), False,
                                        "the head uses THIS window's nav",
                                        "the control BEATS the fed-nav arm")
                per_side[key] = r
            if len(per_side) == 2:
                dA = (np.asarray(comps[ak][t_mk], dtype=np.float64)
                      - np.asarray(comps[ak][c_mk], dtype=np.float64))
                dB = (np.asarray(comps[bk][t_mk], dtype=np.float64)
                      - np.asarray(comps[bk][c_mk], dtype=np.float64))
                per_side["cross_B_minus_A"] = _boot_paired(dB, dA, eid,
                                                           a.n_boot, a.seed)
            echo[f"{base}:{label}"] = per_side
    rec["nav_echo_controls"] = {
        "what": echo,
        "rule": ("⛔ a route/tactical accuracy under TRUE nav is an ECHO INDEX, not "
                 "skill — nav is an INPUT. Quote the shuffled and zero conditionings "
                 "beside it, or the claim is inadmissible (BACKLOG R39). "
                 "`nav_shuffled` withholds the PAIRING (the marginal is preserved "
                 "exactly, so the model still sees a plausible token everywhere); "
                 "`nav_zero` withholds the SIGNAL and is the DEPLOYMENT condition. "
                 "A shuffle cannot stand in for a zero.")}

    # ---- ⭐ WHY `ha` IS NOT THE SHARED FLOOR, MEASURED rather than argued ------ #
    ha_a, ha_b = f"{A.name}:ha", f"{B.name}:ha"
    if ha_a in comps and ha_b in comps:
        blk = {}
        for mk, fam, lib, unit in TRAJ_METRICS:
            mA = (np.asarray(comps[ha_a][mk], dtype=np.float64)
                  - np.asarray(comps[floor_key][mk], dtype=np.float64))
            mB = (np.asarray(comps[ha_b][mk], dtype=np.float64)
                  - np.asarray(comps[floor_key][mk], dtype=np.float64))
            blk[mk] = _boot_paired(mB, mA, eid, a.n_boot, a.seed)
            blk[mk].update({"family": fam, "unit": unit})
        rec["ha_is_not_a_shared_floor"] = {
            "statistic": f"({ha_b} - {floor_key}) - ({ha_a} - {floor_key})",
            "what_it_shows": (
                "`ha` — 'hold the last observed action' — carries the SAME NAME on "
                "both sides and is NOT the same arm. It differs in (i) the "
                "ACTION-UNIT convention (`action_units` above: channel 0 of a "
                "recorded v2ep action is a road-wheel STEER angle, and a dump that "
                "integrates it as a curvature over-rotates by ~L = 2.9x) and (ii) "
                "the HOLD RULE and the tick it is differenced on. ⛔ The difference "
                "below is NOT attributable to either cause alone — it is the "
                "combined size of both, and that is exactly the point: a "
                "same-named control is not a shared floor. `ha0` is, because it is "
                "EXACTLY ZERO in either unit and depends on nothing but v0."),
            "per_metric": blk}

    # ---- statistical power, stated rather than implied ------------------------ #
    rec["power"] = {
        "n_windows": n, "n_episodes": len(clips),
        "min_episodes_for_power": MIN_EPISODES_FOR_POWER,
        "adequate": bool(len(clips) >= MIN_EPISODES_FOR_POWER),
        "note": ("the episode-cluster bootstrap can only ever resample the "
                 f"{len(clips)} distinct episodes present. Below "
                 f"{MIN_EPISODES_FOR_POWER} the interval is dominated by which "
                 "episodes happen to be here, and a 'not separated' row is "
                 "UNDERPOWERED — it is not evidence of no effect.")}
    if not rec["power"]["adequate"]:
        _p(f"[power] ⚠️ only {len(clips)} episodes in the intersection — a "
           f"'not separated' row below is UNDERPOWERED, not a null result")

    rec["absolute_pooled_full_set"] = absolute
    rec["margins_over_floor"] = margins
    rec["cross_model_difference_of_margins"] = cross
    rec["estimator"] = {
        "point": "FULL-SET pooled mean over the shared windows",
        "interval": "taniteval.ci.episode_cluster_bootstrap (single arm) / "
                    "paired_episode_cluster_bootstrap (every difference)",
        "cluster": "the CLIP — windows inside one clip are strongly dependent",
        "n_boot": a.n_boot, "seed": a.seed,
        "forbidden": "overlapping_holdout_se appears nowhere; it biases the POINT "
                     "ESTIMATE bidirectionally (-6.67% to +11.69%), up to a sign flip",
        "reading_rule": "⛔ two overlapping single-arm CIs are NOT a null result and "
                        "two disjoint ones are NOT the paired test. Only the paired "
                        "difference interval decides."}

    # ---- family roll-up ------------------------------------------------------- #
    fam_out = {}
    for fam in FAMILIES:
        keys = [mk for mk, f, _, _ in all_metrics if f == fam]
        rows = {}
        for mk in keys:
            c = cross.get(mk)
            if not c or c.get("status") == "REFUSED":
                rows[mk] = c or _refused("not computed", family=fam)
                continue
            rows[mk] = {"delta": c.get("delta"), "lo": c.get("lo"), "hi": c.get("hi"),
                        "separated": c.get("separated"), "verdict": c.get("verdict"),
                        "A_margin": margins[ak].get(mk, {}).get("delta"),
                        "A_separated": margins[ak].get(mk, {}).get("separated"),
                        "B_margin": margins[bk].get(mk, {}).get("delta"),
                        "B_separated": margins[bk].get(mk, {}).get("separated")}
        sep = [mk for mk in keys if isinstance(rows.get(mk), dict)
               and rows[mk].get("separated")]
        win_b = [mk for mk in sep if "has the better margin" in
                 str(rows[mk].get("verdict")) and bk in str(rows[mk].get("verdict"))]
        win_a = [mk for mk in sep if "has the better margin" in
                 str(rows[mk].get("verdict")) and ak in str(rows[mk].get("verdict"))]
        fam_out[fam] = {"metrics": rows, "n_separated": len(sep),
                        "separated_favouring_B": win_b,
                        "separated_favouring_A": win_a,
                        "family_verdict": ("VOID — a degenerate arm entered it"
                                           if rec["void"] else
                                           "SPLIT" if (win_a and win_b) else
                                           f"{bk}" if win_b else
                                           f"{ak}" if win_a else "no separation")}
        if not rows:
            # ⛔ a family with no computable row is REFUSED WITH ITS REASON AND n —
            # never a silent omission, and never a pass.
            why = [f"`{lk}`: {v.get('disagreement', {}).get('meaning') or v.get('reason') or 'not usable'}"
                   for lk, v in lab_ctl.items()
                   if not v.get("usable") and any(
                       s[1] == fam and s[2] == lk for s in DECISION_SPECS)]
            fam_out[fam] = {
                "metrics": {}, "status": "REFUSED",
                "reason": ("no metric in this family is computable on the shared "
                           "windows. " + " ".join(why)),
                "n": n, "n_episodes": len(clips),
                "estimator": "n/a — inputs missing (WORK ITEM, not a pass)",
                "family_verdict": "REFUSED"}
    rec["families"] = fam_out

    # ---- what this does NOT establish ---------------------------------------- #
    rec["what_this_does_not_establish"] = [
        "⛔ NOTHING ABOUT CLOSED-LOOP DRIVING. Every arm here is OPEN LOOP: neither "
        "model controls the vehicle, and the recorded eval ego data arrives "
        "regardless of what either predicts. A closed-loop number needs AlpaSim or a "
        "real test vehicle and does not exist in this programme.",
        "⛔ NOT that one architecture is better than the other. The two systems are "
        "different KINDS: refav1 is a latent world model + iCEM planner that rolls "
        "out under a 3-channel action including speed; refcv3 is a supervised "
        "one-shot 128-anchor x 8-slot trajectory model with NO action input, NO "
        "rollout and NO per-step decode, consuming vision + a v7.2 nav token + one "
        "ego scalar v0. What is measured is each system's MARGIN OVER THE SAME "
        "TRIVIAL FLOOR on the same windows — not a like-for-like of the mechanisms.",
        "⛔ NOT a statement about either model's own grid. The table is computed on "
        "the COMMON instants only; each dump's finer slots are dropped, so these "
        "levels must not be quoted against a single-arm read on a finer grid.",
        "⛔ NOT a deployment number for refcv3 while its nav token is fed: the v7.2 "
        "nav_cmd is an ORACLE (provenance ego-future) that will not exist at "
        "deployment. The deployment-relevant margin is the nav-withheld arm's "
        "(`os_navzero - ha0`), reported beside it when that arm is in the dump.",
        "⛔ NOT a route-head capability claim from the nav_true row alone. Nav is an "
        "INPUT; flagship v1's route head scored 1.0000 by echoing it. Only the "
        "nav-shuffled and nav-zero conditionings carry information about skill.",
        "⛔ NOT distance-keeping, headway, time-gap or TTC. Those need the lead-block "
        "join, which is per-dump and is NOT re-derived here — read them from each "
        "side's own single-arm record. Their absence here is a WORK ITEM, not a pass.",
        "⛔ NOT a claim that either model would drive: an open-loop trajectory error "
        "on recorded data does not bound compounding error under its own control.",
        f"⛔ NOT evidence of NO effect where a row reads 'not separated'. The "
        f"episode-cluster bootstrap resamples the {len(clips)} episodes in the "
        f"intersection and nothing else; below {MIN_EPISODES_FOR_POWER} episodes "
        f"such a row is UNDERPOWERED — a statement about the sample, not about the "
        f"models."
        + ("" if len(clips) >= MIN_EPISODES_FOR_POWER
           else f" ⚠️ THAT IS THE CASE HERE: {len(clips)} episodes."),
    ]
    if rec["void"]:
        rec["what_this_does_not_establish"].insert(
            0, "⛔⛔ THIS READ IS VOID. " + " ".join(rec["void_reasons"]) +
               " A VOID read is not a negative result: the instrument saw a "
               "degenerate arm, so it measured the baseline, not the model.")
    rec["refire_command"] = (
        f"python taniteval/tools/paired_openloop.py "
        f"--a-dump <refav1 FINAL dump> --a-name {A.name} --a-arm {a.a_arm} "
        f"--b-dump <refcv3 FINAL dump> --b-name {B.name} --b-arm {a.b_arm} "
        + " ".join(f"--a-extra {x}" for x in a.a_extra)
        + (" " if a.a_extra else "")
        + " ".join(f"--b-extra {x}" for x in a.b_extra)
        + f" --a-run-config <refav1 run config.json> "
          f"--b-run-config <refcv3 run config.json> "
          f"--floor {a.floor} --n-boot {a.n_boot} --seed {a.seed} "
          f"--out taniteval/results/paired-openloop-{A.name}-vs-{B.name}-<UTC>.json "
          f"--md <RESULT.md>")
    rec["refire_preconditions"] = [
        "⛔ refcv3's FINAL checkpoint is `<run>/ckpt.pt`, NOT `ckpt_40284_FINAL.pt` — "
        "that file is never written (`refc_v3_train.py:103` MILESTONES = "
        "(5000, 15000, 20000, 30000); 40,284 is not among them, and the final save "
        "comes from the `or step == args.steps` branch at `refc_v3_train.py:1191`). "
        "`ckpt.pt` is ALSO the rolling checkpoint, so ASSERT `ckpt['step'] == 40284` "
        "after loading rather than trusting the path.",
        "⛔ the refcv3 dump must be rolled at `--window-stride 1`. refav1's window "
        "origins are RAW `2*ws`; a stride-5 refcv3 dump reaches a disjoint residue "
        "class and the intersection is EMPTY (this tool refuses with the modulus "
        "arithmetic printed).",
        "⛔ neither model may be rolled on a host that is training.",
        "⭐ pass `--b-extra os_navzero`: with an ORACLE nav token, `os - ha0` is not "
        "the deployment margin and must not stand alone as the headline.",
    ]
    return rec


# --------------------------------------------------------------------------- #
# 6. rendering                                                                 #
# --------------------------------------------------------------------------- #
def _fmt(v, dp=4):
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if not np.isfinite(f) else f"{f:.{dp}f}"


def print_table(rec: dict) -> None:
    ak = f"{rec['inputs']['A']['name']}:{rec['_a_arm']}"
    bk = f"{rec['inputs']['B']['name']}:{rec['_b_arm']}"
    fk = f"shared:{rec['_floor']}"
    _p("")
    _p("=" * 100)
    _p(f"PAIRED OPEN-LOOP COMPARISON — {ak}  vs  {bk}   (floor {fk})")
    _p(f"  n = {rec['intersection']['n_windows']} shared windows over "
       f"{rec['intersection']['n_episodes']} episodes; common instants "
       f"{rec['gates']['G1_common_grid']['common_instants_s']} s")
    if rec["void"]:
        _p("  ⛔⛔ THE READ IS VOID — a degenerate arm entered it. See void_reasons.")
    _p("=" * 100)
    for fam in FAMILIES:
        f = rec["families"].get(fam) or {}
        _p("")
        _p(f"--- {fam}  [family verdict: {f.get('family_verdict')}] "
           + "-" * max(0, 60 - len(fam)))
        _p(f"  {'metric':28s} {'A abs':>9s} {'B abs':>9s} {'floor':>9s} "
           f"{'A-floor':>9s} {'B-floor':>9s} {'(B-f)-(A-f)':>12s} {'95% CI':>22s} sep")
        for mk, row in (f.get("metrics") or {}).items():
            if not isinstance(row, dict) or row.get("status") == "REFUSED":
                _p(f"  {mk:28s} REFUSED — {str((row or {}).get('reason'))[:60]}")
                continue
            aa = rec["absolute_pooled_full_set"].get(ak, {}).get(mk, {}).get("mean")
            bb = rec["absolute_pooled_full_set"].get(bk, {}).get(mk, {}).get("mean")
            ff = rec["absolute_pooled_full_set"].get(fk, {}).get(mk, {}).get("mean")
            ci = f"[{_fmt(row.get('lo'))}, {_fmt(row.get('hi'))}]"
            _p(f"  {mk:28s} {_fmt(aa):>9s} {_fmt(bb):>9s} {_fmt(ff):>9s} "
               f"{_fmt(row.get('A_margin')):>9s} {_fmt(row.get('B_margin')):>9s} "
               f"{_fmt(row.get('delta')):>12s} {ci:>22s} "
               f"{'YES' if row.get('separated') else 'no'}")


def render_md(rec: dict) -> str:
    ak = f"{rec['inputs']['A']['name']}:{rec['_a_arm']}"
    bk = f"{rec['inputs']['B']['name']}:{rec['_b_arm']}"
    fk = f"shared:{rec['_floor']}"
    An, Bn = rec["inputs"]["A"]["name"], rec["inputs"]["B"]["name"]
    L = []
    A = L.append
    A(f"# PAIRED OPEN-LOOP COMPARISON — {An} vs {Bn}")
    A("")
    A(f"`{rec['tool']}` · generated `{rec['utc']}` · **0 GPU** (it reads banked dumps only)")
    A("")
    # ---- line one ----------------------------------------------------------- #
    if rec["void"]:
        A("## ⛔⛔ LINE ONE: **THE READ IS VOID, NOT NEGATIVE.**")
        A("")
        for r in rec["void_reasons"]:
            A(f"* {r}")
        A("")
        A("A VOID read means the instrument saw a **degenerate arm** — it measured the "
          "trivial baseline, not the model. Every number below is banked for audit and "
          "⛔ **must not be read as \"model X beats model Y\"**.")
    else:
        beats = []
        cands = [(An, ak), (Bn, bk)]
        # ⭐ the NAV-WITHHELD arm counts as its own headline candidate: with an
        # ORACLE nav token, a win that survives only WITH the oracle is not a win.
        for x in rec["_b_extra"]:
            if x.endswith("navzero"):
                cands.append((f"{Bn} (nav withheld)", f"{Bn}:{x}"))
        for nm, key in cands:
            r = rec["margins_over_floor"].get(key, {}).get("ade_m", {})
            if r.get("separated") and (r.get("delta") or 0) < 0:
                beats.append(nm)
        if not beats:
            A("## ⛔ LINE ONE: **NEITHER MODEL BEATS THE TRIVIAL FLOOR `ha0` ON ADE.**")
            A("")
            A("`ha0` is constant velocity at the measured `v0` — a straight line at "
              "constant speed, consuming nothing the model produces. An arm that does "
              "not beat it has not been shown to plan.")
        else:
            A(f"## LINE ONE: beats the trivial floor `ha0` on ADE (paired, separated): "
              f"**{', '.join(beats)}**")
            nz = [k2 for k2 in rec["margins_over_floor"] if k2.endswith("navzero")]
            for k2 in nz:
                r = rec["margins_over_floor"][k2].get("ade_m", {})
                if r and not r.get("separated"):
                    A("")
                    A(f"⚠️ **But `{k2}` — the SAME model with the oracle nav token "
                      f"WITHHELD — does not separate from the floor on ADE "
                      f"({_fmt(r.get('delta'))} [{_fmt(r.get('lo'))}, "
                      f"{_fmt(r.get('hi'))}]).** The v7.2 `nav_command` is an ORACLE "
                      f"(provenance ego-future) that will not exist at deployment, "
                      f"so the nav-withheld margin — not the fed-nav one — is the "
                      f"deployment-relevant number. A model that wins only when "
                      f"handed an oracle has not won.")
    # is either model arm literally the floor?
    tpa = rec["profiles"]["trivial"]["arms"]
    for nm, key in ((An, ak), (Bn, bk)):
        idn = (tpa.get(key) or {}).get("identical_to", {}).get(fk, 0)
        if idn and idn == rec["intersection"]["n_windows"]:
            A("")
            A(f"⛔ **`{key}` IS the floor.** It is bit-identical to `{fk}` on "
              f"{idn}/{idn} shared windows, so its margin is exactly `0.0000` on "
              f"every row below. This is not \"loses to the floor\" — the model's "
              f"emitted trajectory and the no-information baseline are the same "
              f"array.")
    if not rec["power"]["adequate"]:
        A("")
        A(f"⚠️ **UNDERPOWERED: only {rec['power']['n_episodes']} episodes in the "
          f"intersection.** {rec['power']['note']}")
    A("")
    # ---- what it is --------------------------------------------------------- #
    A("## What this compares, and the vocabulary it uses")
    A("")
    A("⛔ **Every arm here is OPEN LOOP** (PI ruling 2026-09-02). The model is not "
      "controlling the vehicle; its trajectory does not affect the ego data it is next "
      "fed. That is true of refav1's `cl` — a predictor consuming its own planner's "
      "actions — exactly as it is of refcv3's `os`. **The words \"closed loop\" do not "
      "apply to any number in this document**; a closed-loop read needs AlpaSim or a "
      "real vehicle and does not exist in this programme.")
    A("")
    A("| | A | B |")
    A("|---|---|---|")
    A(f"| model | `{An}` | `{Bn}` |")
    A(f"| arm | `{rec['_a_arm']}` | `{rec['_b_arm']}` |")
    A(f"| tier | `{rec['gates']['G3_same_tier']['A']['tier']}` | "
      f"`{rec['gates']['G3_same_tier']['B']['tier']}` |")
    A(f"| checkpoint | `{rec['inputs']['A']['model'].get('ckpt')}` | "
      f"`{rec['inputs']['B']['model'].get('ckpt')}` |")
    A(f"| step | `{rec['inputs']['A']['model'].get('step')}` | "
      f"`{rec['inputs']['B']['model'].get('step')}` |")
    A(f"| dump grid | dt {rec['inputs']['A']['grid']['dt_s']} s, K "
      f"{rec['inputs']['A']['grid']['k']} | dt {rec['inputs']['B']['grid']['dt_s']} s, "
      f"K {rec['inputs']['B']['grid']['k']} |")
    A("")
    A("**The asymmetry, stated plainly.** `refcv3` is a **supervised ONE-SHOT anchor "
      "model** — 128 anchors x 8 slots, no action input, no rollout, no per-step "
      "decode — consuming vision + a v7.2 nav token + ONE ego scalar `v0`. `refav1` is "
      "a **latent world model + iCEM planner** that rolls out and consumes a 3-channel "
      "action including speed. They are not the same kind of system, which is why the "
      "admissible statistic is each arm's **margin over the same trivial floor**, "
      "paired on the same windows — never the two arms as levels "
      "(`D-HF-COMPARABILITY`).")
    A("")
    # ---- intersection ------------------------------------------------------- #
    it = rec["intersection"]
    g1 = rec["gates"]["G1_common_grid"]
    A("## The intersection, reported honestly")
    A("")
    A(f"* **n = {it['n_windows']} shared windows over {it['n_episodes']} episodes.** "
      f"Key: `{it['key']}`.")
    A(f"* A carried {it['A_windows_available']} windows over {it['A_clips']} clips "
      f"({it['A_windows_dropped']} dropped); B carried {it['B_windows_available']} "
      f"over {it['B_clips']} ({it['B_windows_dropped']} dropped).")
    A(f"* **Common instants: {g1['common_instants_s']} s** (raw-frame offsets "
      f"{g1['common_raw_frame_offsets']}). A dropped slots {g1['A_slots_dropped']}, "
      f"B dropped {g1['B_slots_dropped']}. The grids are matched in **integer raw "
      f"frames**; there is no resampling and no interpolation.")
    A(f"* The bootstrap resamples **episodes**, not windows — "
      f"{it['n_episodes']} is the sample size that matters.")
    A("")
    # ---- controls ----------------------------------------------------------- #
    A("## The controls that had to read a known value")
    A("")
    A("| control | expected | measured | pass |")
    A("|---|---|---|---|")
    wk = rec["controls"]["window_key_proof"]
    A(f"| window key: GT identity across two independent pipelines | `0.0` exactly | "
      f"`{wk['measured_max_abs_gt_diff_m']:.3e}` m | {'✅' if wk['pass'] else '⛔'} |")
    A(f"| window key: `v0` identity | `0.0` exactly | "
      f"`{wk['measured_max_abs_v0_diff_mps']:.3e}` m/s | "
      f"{'✅' if wk['pass'] else '⛔'} |")
    for side in ("A", "B"):
        c = rec["controls"]["floor_C1_no_information_value"][side]
        A(f"| C1 ({side}) floor reads the **no-information value** `x=v0·t, y=0` | "
          f"`0.0` exactly | `dx {c['measured_max_abs_dx_m']:.3e}` m, "
          f"`|y| {c['measured_max_abs_y_m']:.3e}` m | "
          f"{'✅' if c['pass'] else '⛔'} |")
    c2 = rec["controls"]["floor_C2_bit_comparable"]
    A(f"| C2 the floor is bit-comparable across the two architectures | `0.0` exactly | "
      f"`{c2['measured_max_abs_diff_m']:.3e}` m | {'✅' if c2['pass'] else '⛔'} |")
    c3 = rec["controls"]["floor_C3_derived_vs_dumped"]
    if "measured_max_abs_diff_m" in c3:
        A(f"| C3 the DERIVED floor against the DUMPED one | `0.0` | "
          f"`{c3['measured_max_abs_diff_m']:.3e}` m | "
          f"{'✅' if c3['pass'] else '⛔'} |")
    for lk, v in rec["controls"]["declared_label_identity"].items():
        if v.get("status") == "ABSENT":
            A(f"| declared labels `{lk}` | identical on every shared window | ABSENT — "
              f"{v['reason']} | — |")
        else:
            A(f"| declared labels `{lk}` agree where BOTH sides label | "
              f"identical on all {v['n_labelled_BOTH']} | "
              f"{'identical' if v['agree_where_both_labelled'] else 'DISAGREE'} "
              f"(A labels {v['n_labelled_A']}, B labels {v['n_labelled_B']} of "
              f"{v['n_shared_windows']}) | "
              f"{'✅' if v['usable'] else '⛔'} |")
    A("")
    A(f"**Floor provenance.** A: {rec['gates']['G6_shared_floor']['source']['A']}  \n"
      f"B: {rec['gates']['G6_shared_floor']['source']['B']}")
    A("")
    A(f"⚠️ {rec['controls']['floor_used']}")
    A("")
    # ---- profiles ----------------------------------------------------------- #
    A("## Shape before metrics — the profiles that decide whether the read is VOID")
    A("")
    A("| arm | straight | const speed | **CONSTANT-VELOCITY** | identical to |")
    A("|---|---|---|---|---|")
    for k2, v in rec["profiles"]["trivial"]["arms"].items():
        if not v.get("n"):
            continue
        ident = ", ".join(f"`{b}` {c}/{v['n']}"
                          for b, c in v["identical_to"].items() if c) or "—"
        A(f"| `{k2}` | {v['straight_frac']:.4f} | {v['const_speed_frac']:.4f} | "
          f"**{v['constant_velocity_frac']:.4f}** | {ident} |")
    A("")
    for nm, s in rec["profiles"]["selection"].items():
        if s.get("status") == "ABSENT":
            A(f"* selection profile `{nm}`: ABSENT — {s['reason']}")
        else:
            A(f"* selection profile `{nm}`: **{s['n_distinct_selected']} distinct "
              f"anchors**, modal #{s['modal_anchor']} at {s['modal_share']:.4f}, "
              f"entropy {s['entropy_nats']:.4f}"
              + (f", agrees with the GT-nearest oracle on "
                 f"{s['agrees_with_oracle_frac']:.4f}"
                 if "agrees_with_oracle_frac" in s else "")
              + (" — ⛔ **DEGENERATE**" if s.get("degenerate") else ""))
    A("")
    # ---- the four families -------------------------------------------------- #
    A("## The four families, separately — never pooled")
    A("")
    A("`A-floor` and `B-floor` are each arm's **margin over `ha0`** (paired, "
      "episode-cluster bootstrap). `(B-f)-(A-f)` is the **difference of margins** — "
      "the only admissible cross-model statistic. For error metrics NEGATIVE is "
      "better; for the accuracy rows (`*_correct`) POSITIVE is better.")
    A("")
    for fam in FAMILIES:
        f = rec["families"].get(fam) or {}
        A(f"### {fam} — family verdict: **{f.get('family_verdict')}**")
        A("")
        if f.get("status") == "REFUSED":
            A(f"⛔ **REFUSED** (n = {f.get('n')} shared windows over "
              f"{f.get('n_episodes')} episodes). {f.get('reason')}")
            A("")
            A(f"*{f.get('estimator')}*")
            A("")
            continue
        A(f"| metric | `{ak}` | `{bk}` | `{fk}` | A−floor | B−floor | "
          f"(B−f)−(A−f) | 95% CI | separated |")
        A("|---|---|---|---|---|---|---|---|---|")
        for mk, row in (f.get("metrics") or {}).items():
            if not isinstance(row, dict) or row.get("status") == "REFUSED":
                A(f"| `{mk}` | colspan REFUSED — {str((row or {}).get('reason'))} "
                  f"| | | | | | | |")
                continue
            aa = rec["absolute_pooled_full_set"].get(ak, {}).get(mk, {}).get("mean")
            bb = rec["absolute_pooled_full_set"].get(bk, {}).get(mk, {}).get("mean")
            ff = rec["absolute_pooled_full_set"].get(fk, {}).get(mk, {}).get("mean")
            A(f"| `{mk}` | {_fmt(aa)} | {_fmt(bb)} | {_fmt(ff)} | "
              f"{_fmt(row.get('A_margin'))}"
              f"{'*' if row.get('A_separated') else ''} | "
              f"{_fmt(row.get('B_margin'))}"
              f"{'*' if row.get('B_separated') else ''} | "
              f"**{_fmt(row.get('delta'))}** | "
              f"[{_fmt(row.get('lo'))}, {_fmt(row.get('hi'))}] | "
              f"{'**YES**' if row.get('separated') else 'no'} |")
        A("")
    A("`*` on a margin means that arm's own paired interval against the floor "
      "excludes zero.")
    A("")
    df = rec["controls"].get("decision_floor", {})
    if df.get("per_key"):
        A(f"**The floor for the two decision families is not `ha0`.** {df['why']} "
          f"Per key: "
          + "; ".join(f"`{k2}` majority class {v['majority_class']} at "
                      f"{v['majority_rate']:.4f} over n={v['n']}"
                      for k2, v in df["per_key"].items()) + ".")
        A("")
    # ---- nav echo ----------------------------------------------------------- #
    echo = (rec.get("nav_echo_controls") or {}).get("what") or {}
    if echo:
        A("### The nav-echo controls (STRATEGIC and declared TACTICAL are "
          "inadmissible without them)")
        A("")
        A(rec["nav_echo_controls"]["rule"])
        A("")
        A("| head / control | side | delta | 95% CI | separated |")
        A("|---|---|---|---|---|")
        for name, per in echo.items():
            for side, r in per.items():
                if not isinstance(r, dict) or "delta" not in r:
                    continue
                A(f"| `{name}` | `{side}` | {_fmt(r.get('delta'))} | "
                  f"[{_fmt(r.get('lo'))}, {_fmt(r.get('hi'))}] | "
                  f"{'**YES**' if r.get('separated') else 'no'} |")
        A("")
    hn = rec.get("ha_is_not_a_shared_floor")
    if hn:
        A("### ⭐ Why `ha` is not the shared floor — measured, not argued")
        A("")
        A(hn["what_it_shows"])
        A("")
        A(f"| metric | `{hn['statistic']}` | 95% CI | separated |")
        A("|---|---|---|---|")
        for mk, r in hn["per_metric"].items():
            if "delta" not in r:
                continue
            A(f"| `{mk}` | **{_fmt(r['delta'])}** | [{_fmt(r['lo'])}, "
              f"{_fmt(r['hi'])}] | {'**YES**' if r.get('separated') else 'no'} |")
        A("")
    lc = rec["controls"].get("declared_label_identity") or {}
    notes = [v["asymmetry_note"] for v in lc.values()
             if isinstance(v, dict) and v.get("asymmetry_note")]
    if notes:
        A("**Label-coverage asymmetry between the two tools** (a finding about the "
          "label derivations, not about either model):")
        A("")
        for s in notes:
            A(f"* {s}")
        A("")
    # ---- what it does not establish ----------------------------------------- #
    # ---- the OTHER half of the PI's question -------------------------------- #
    cl = rec["closed_loop"]
    A("## ⛔ CLOSED LOOP — the other half of the question, and it is NOT in the table above")
    A("")
    A("The PI asked for **open AND closed** loop. Everything above is the **open-loop "
      "half**. " + cl["reading_rule"])
    A("")
    A(f"* **Not measured here, and this tool cannot measure it.** {cl['why_not']}")
    A(f"* **But the harness EXISTS** — `{cl['the_harness_exists']}` — so this is a "
      f"NOT-YET-RUN, not a NOT-POSSIBLE. It {cl['what_it_does']}.")
    A(f"* It has already produced a published panel: {cl['already_published_on']}.")
    A(f"* **refcv3:** {cl['for_these_arms']['refcv3']}")
    A(f"* **refav1:** {cl['for_these_arms']['refav1']}")
    A("")
    # ---- corpus identity ---------------------------------------------------- #
    ci = rec["controls"]["corpus_identity"]
    A("## ⚠️ Are the two runs even on the same corpus?")
    A("")
    A(f"**{ci['status']}** — {ci['reason']}")
    A("")
    A(f"* ⭐ **What IS established:** {ci['what_is_established_instead']}")
    A(f"* ⛔ **What is NOT:** {ci['what_is_NOT_established']}")
    A("")
    for side, nm in (("A", An), ("B", Bn)):
        pub = ci[side]["published"]
        miss = ci[side]["missing"]
        A(f"`{nm}` publishes: "
          + (", ".join(f"`{k}` = `{v}`" for k, v in pub.items()) or "*nothing*")
          + (f". Missing: {'; '.join(miss)}." if miss else "."))
    A("")
    A("## ⛔ What this comparison does NOT establish")
    A("")
    for s in rec["what_this_does_not_establish"]:
        A(f"* {s}")
    A("")
    A("## Estimator")
    A("")
    for k2, v in rec["estimator"].items():
        A(f"* **{k2}**: {v}")
    A("")
    A("## Re-fire against the FINAL checkpoints")
    A("")
    A("```")
    A(rec["refire_command"])
    A("```")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="PAIRED cross-model OPEN-LOOP comparison of two banked dumps")
    ap.add_argument("--a-dump", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--a-arm", required=True)
    ap.add_argument("--a-extra", action="append", default=[])
    ap.add_argument("--b-dump", required=True)
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--b-arm", required=True)
    ap.add_argument("--b-extra", action="append", default=[])
    ap.add_argument("--a-run-config", default=None,
                    help="the A run's OWN config.json — for the corpus-identity "
                         "control (parity flag, corpus key, train cache)")
    ap.add_argument("--b-run-config", default=None,
                    help="the B run's OWN config.json (see --a-run-config)")
    ap.add_argument("--floor", default="ha0",
                    help="the shared trivial floor arm (default ha0)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the record here (JSON)")
    ap.add_argument("--md", default=None, help="write the RESULT.md here")
    a = ap.parse_args(argv)

    rec = run(a)
    rec["_a_arm"], rec["_b_arm"], rec["_floor"] = a.a_arm, a.b_arm, a.floor
    rec["_a_extra"], rec["_b_extra"] = list(a.a_extra), list(a.b_extra)
    print_table(rec)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, default=str)
        _p(f"\n[out] {a.out} ({os.path.getsize(a.out)} B)")
    if a.md:
        os.makedirs(os.path.dirname(os.path.abspath(a.md)) or ".", exist_ok=True)
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(render_md(rec))
        _p(f"[out] {a.md} ({os.path.getsize(a.md)} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
