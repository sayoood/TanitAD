#!/usr/bin/env python3
"""refcv3_arm.py — the eval adapter for REF-C v3 (``tanitad.refs.refc_v3``).

⛔⛔ READ ``taniteval/tools/REFCV3_ARM.md`` §2 FIRST. It is the DEFINITION of what
this arm is, derived with file:line, and it is the deliverable the Master Mind
reviews BEFORE any number produced here may be quoted.

WHY THIS FILE EXISTS (register ``D-V7-READINESS-2026-09-02`` §D, BACKLOG R20):
refcv3 has NO admissible T1 number and no instrument that can produce one. Its
in-training eval is a T0 loss on 160 fixed windows and must never be quoted as
driving performance.

⛔⛔ THE ONE FACT THAT SHAPES EVERY LINE BELOW (register ``D-HF-COMPARABILITY``,
MEASURED). refcv3 is a SUPERVISED ONE-SHOT ANCHOR TRAJECTORY MODEL — 128 anchors
x 8 slots (``refc_v3.py:196``/``:197``/``:106``), NO action input, NO rollout, NO
per-step decode (``refc_v3.py:480`` — the forward signature has no action
argument). ``t1_eval.roll_closed`` (``t1_eval.py:760``) carries the FLAGSHIP,
which is also supervised, because the flagship is ADDITIONALLY autoregressive: it
feeds ``(steer = atan(L*kappa), a_j)`` back into the predictor each step. refcv3
has no action to feed back, so THERE IS NO LOOP TO CLOSE and roll_closed cannot
be ported. Consequences, all implemented here:

  * the arm is named ``os`` (one-shot) and ⛔ NEVER ``cl`` — a shared column name
    is how two different procedures end up in one table read as one quantity;
  * ⭐ only ``ha0`` (constant velocity at the measured v0) is BIT-COMPARABLE
    across refav1 and refcv3, and it is the floor both must be beaten against;
  * ``ol`` DOES NOT EXIST for refcv3 (it consumes no recorded actions) — it is
    written into the record as ABSENT WITH ITS REASON, never silently dropped;
  * the admissible cross-model claim is each arm's MARGIN OVER THE SAME ``ha0``
    FLOOR, per family, paired — ⛔ never ``os`` against ``cl`` as levels.

ARMS AND TIER STAMPS (``Project Steering/EVAL_DOCTRINE.md``; the stamps travel
into ``t1_eval.analyze`` unchanged):

    os          T1*  ONE forward pass at the window origin: the observed frames,
                     the clip's v7.2 nav token and the MEASURED v0 at t0 and
                     nothing else; the path is the model's OWN selection
                     ``out["traj"]`` — ranked by ``sel_score_v3``
                     (``refc_v3.py:509-527``) on the hierarchical arm, by the
                     core's own ``sel_score`` (``refc.py:1531-1534``) on the flat
                     arm. ⛔ NEVER ``a_star``.
    os_navshuf  T1*  the same forward with nav_cmd PERMUTED across the eval
                     windows. Breaks the PAIRING, PRESERVES the nav marginal:
                     "is the model using THIS window's nav?"
    os_navzero  T1*  ⭐ the same forward with nav WITHHELD (nav_cmd=None): the
                     E13 injection into the tactical and strategic layers is
                     SKIPPED ENTIRELY (refc_v3.py:437-441) and the core collapses
                     to the 'follow' one-hot (refc.py:2021-2024). "What does the
                     model do when the ORACLE nav is not there?" — i.e. what
                     DEPLOYMENT looks like, since nav here is an oracle input
                     (provenance ego-future) that will not exist at deployment.
                     ⛔ SHUFFLE AND ZERO ARE NOT INTERCHANGEABLE and BACKLOG R39
                     binds BOTH: a shuffle preserves the distribution and breaks
                     the pairing, a zero removes the signal. MEASURED elsewhere
                     (D-REFAV1-TAC-DECODER-PANEL): refav1's tactical head ranked
                     turns at AUC 0.873 under true nav and collapsed to 0.520
                     (chance) under nav_zero, while a NAV-ONLY predictor beat the
                     model outright (0.684 > 0.650). See ``NAV_NULL`` for what
                     each of the three conditions removes, per layer.
    ha          T1   HOLD-ACTION control: the (a, steer) that CLOSES at t0, held
                     for the horizon through the programme's ONE unicycle. Reads
                     only frames <= t0. NOT a floor — it can be WORSE than
                     trivial, because a held noisy steer drifts.
    ha0         T1   ⭐ CONSTANT VELOCITY: a = 0, kappa = 0 at the measured v0 —
                     a straight line at constant speed. THE STRONGEST TRIVIAL
                     BASELINE and the echo test's real bar; consumes strictly
                     less than ``ha``. Exactly zero in either action unit, which
                     is what makes it bit-comparable with refav1's ``ha0``.
    oracle_sel  T0   (opt-in, --with-oracle-sel) the ``a_star``-selected anchor's
                     refinement — the GROUND-TRUTH-NEAREST anchor
                     (``refc_v3_train.py:460-463``). The CEILING, reported beside
                     the deployed arm to show how much of ``traj`` was selection.
                     ⛔ NEVER compared to a T1 number.
    ol          ABSENT — refcv3 consumes no recorded actions (see above).

⚠️ ``T1*`` = STAMPED T1, RULING OPEN. EVAL_DOCTRINE's T1 row says *"the predictor
consumes the decoder/planner's own actions"*, which does not literally cover a
model that consumes NO actions. Whether the doctrine admits it is a PI /
Master-Mind ruling (BACKLOG R30); the record carries ``_tier_ruling`` with
status ``UNRULED`` on every emitted block, so the numbers exist and are correctly
labelled either way, and the MARGIN framing (``os - ha0``) survives the ruling in
both directions.

THE GRID (index-select, never interpolation). The model emits 8 slots at
``V3_HORIZONS = (5,10,15,20,30,40,50,60)`` x 0.1 s; ``t1_eval``'s dump contract is
a UNIFORM [N, K, 2] grid, so the dump is an INDEX-SELECT of the model's own slots:
    --grid 2s   dt 0.5 s, K 4 -> slots 5,10,15,20      (indices 0,1,2,3)
    --grid 6s   dt 1.0 s, K 6 -> slots 10,20,30,40,50,60 (indices 1,3,4,5,6,7)
A grid the model cannot serve by index-select is REFUSED. GT comes from the
TRAINER'S OWN target function (``refb_labels.waypoint_targets``,
``refc_v3_train.py:452``), so the arm is scored against the GT it was trained
against. ``ha``/``ha0`` integrate at the corpus's native 0.1 s tick and are then
index-selected onto the same instants.

⛔ ACTION UNITS. ``physicalai.py:621`` writes ``steer = arctan(L_enc * curvature)``
into ``actions[:, 0]`` (``:632``) — a ROAD-WHEEL ANGLE, not a curvature. The
``refav1_loader`` docstring calling it "the MEASURED true-kappa channel" is wrong,
and the rescued refcv3 draft inherited that sentence into its
``recorded_controls`` (``C-REFCV3-ARM-SAME-DEFECT``; the identical defect cost
refav1 0.716 m of curved-window lateral error against a 0.053 m floor).
``--action-units`` names the unit channel 1 ARRIVES in and the value is PRINTED
into the manifest.
⚠️ THE DEFAULT HERE IS ``steer`` — DELIBERATELY DIFFERENT FROM ``refav1_arm.py``,
whose ``kappa`` default exists only to keep re-analysis of PRE-2026-09-03 BANKED
dumps byte-identical. refcv3 has no banked dumps, so shipping the known defect as
this instrument's default would have no benefit at all. ``os`` never touches the
channel (the model emits a path, not a control), so this affects ``ha`` only —
and ``ha0`` is zero in either unit.

DUMP SCHEMA — two files per episode, so ``t1_eval.py --analyze-only`` stays valid:
    <dump>/ep{fi:03d}.npz            the t1_eval contract (unchanged):
        g [N,K,2]  os ha ha0 [os_navshuf] [os_navzero] [oracle_sel] [N,K,2]
        ws [N] PROVIDER frame index of the window origin t0
        eid [1]  clip_index [1]  v0 [N]
    <dump>/decisions/ep{fi:03d}.npz  the refcv3 sidecar (_SIDECAR_DOC)
    <dump>/manifest.json             model rebuild + cross-checks, grid, tiers,
                                     the T1 definition, action-units, nav policy,
                                     label join, label timing, episodes.

ESTIMATOR: full-set pooled point estimates; intervals = episode-cluster bootstrap
(``taniteval.ci``, ``ci.py:225``), PAIRED across arms on the same windows
(``ci.py:275``). ``overlapping_holdout_se`` is never used — it biases the POINT
ESTIMATE, not only the interval.

⛔⛔ THE TRIVIAL-PROFILE INSTRUMENT PRINTS BEFORE ANY FAMILY ROW. On 2026-09-03 a
paired refav1 read was rolled for 2.5 h and then found VOID: ``cl`` was a straight
constant-speed line on 140/140 windows and bit-identical to ``cl_navshuf`` on
122/140. This instrument (``refav1_arm.trivial_profile``, IMPORTED — one
instrument, not two) reads that in the first minute.

PROVENANCE. This tool is a RESTART from the current ``refav1_arm.py``, not a
continuation of the rescued draft
(``…/incoming/2026-09-03-refcv3-arm-UNVERIFIED/refcv3_arm.py``). The draft has
ZERO references to ``ha0``, ``trivial_profile``, ``action_units``,
``sel_score_v3``, ``a_star`` or ``oracle_sel``, and its spine is the claim
``D-HF-COMPARABILITY`` forbids. Its model-rebuild path (config from ``argv``
through the trainer's own parser), its grid index-select, its eval-dataset
subclass, its corpus join and its common-grid lead join were HARVESTED and are
credited at their definitions.
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
_SCRIPTS = os.path.join(_REPO, "stack", "scripts")        # refc_v3_train.py lives here


def _bootstrap_paths() -> None:
    """⛔ THE NAMESPACE-PACKAGE SHADOW. The outer ``<repo>/taniteval/`` has no
    ``__init__.py``; run from the repo root, ``import taniteval`` binds THAT
    directory as a namespace package and ``taniteval.ci`` "does not exist". Once
    bound, no sys.path edit undoes it — so the wrong binding is EVICTED here and
    every module the ANALYSIS needs is imported NOW, before anything expensive.
    (An analysis-time ``ModuleNotFoundError`` has already destroyed a completed
    2-arm / 40-episode rollout after the GPU was paid for.)"""
    for p in (os.path.join(_REPO, "stack"), _TE_PARENT, _SCRIPTS):
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
        import taniteval.ci            # noqa: F401  (the preflight)
        import taniteval.four_families  # noqa: F401
        import taniteval.lead_metrics   # noqa: F401
        import taniteval.selgap         # noqa: F401
    except ModuleNotFoundError as ex:                      # pragma: no cover
        sys.exit(f"[refcv3_arm] taniteval preflight failed ({ex}). The real "
                 f"package is {_TE_PKG}; a namespace shadow of the outer dir "
                 f"was probably bound first. sys.path[:3]={sys.path[:3]}")


_bootstrap_paths()


def _load_by_path(name: str, path: str):
    """Import a SIBLING module by file, so its machinery is REUSED, never copied."""
    if not os.path.exists(path):
        sys.exit(f"[refcv3_arm] required sibling {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: ⭐ THE TEMPLATE, IMPORTED. ``refav1_arm`` owns the dump contract, the
#: trivial-profile instrument, the lead-block reader/join, the unicycle call with
#: its action-unit contract, the per-window family components and the paired
#: bootstrap. Importing it is what keeps ONE convention in the programme.
ra = _load_by_path("refav1_arm_for_refcv3", os.path.join(_HERE, "refav1_arm.py"))
t1 = ra.t1                                   # the SAME t1_eval module object

_TRAINER = None


def trainer():
    """``refc_v3_train.py`` imported lazily by path — it is a SCRIPT, and it owns
    ``V3Dataset`` (the window contract the run trained on), ``frames_to_device``
    (the ONE frame-ingest point) and ``build_parser`` / ``_pin_trainer_cfg`` (the
    only way a checkpoint's config is recoverable). *(Harvested from the rescued
    draft, which derived it.)*"""
    global _TRAINER
    if _TRAINER is None:
        _TRAINER = _load_by_path("refc_v3_train_for_arm",
                                 os.path.join(_SCRIPTS, "refc_v3_train.py"))
    return _TRAINER


# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #
DT_FRAME = 0.1           # the 10 Hz corpus tick (the v2ep provider's own grid)
#: ``--grid`` -> (dt_s, K). Both are index-selects of V3_HORIZONS; nothing else is.
GRIDS = {"2s": (0.5, 4), "6s": (1.0, 6)}
ARM_TIERS = {"os": "T1", "os_navshuf": "T1", "os_navzero": "T1", "ha": "T1",
             "ha0": "T1", "ha0_ext": "T1", "oracle_sel": "T0",
             "os_navflip": "T1", "os_navpred": "T1"}
#: The route -> nav mapping is IMPORTED from the label module, never re-derived
#: here. ``stack/scripts/refb_labels.py:483-484`` is the only definition:
#: ``{ROUTE_LEFT: NAV_LEFT, ROUTE_STRAIGHT: NAV_FOLLOW, ROUTE_RIGHT: NAV_RIGHT}``
#: = ``{0: 1, 1: 0, 2: 2}``.
#: |WARN| IT IS NOT THE IDENTITY, and a comment in this very file said it was:
#: ``NAV_COMMANDS`` is FOUR wide (``follow, left, right, straight``,
#: ``refb.py:64``) while ``ROUTE_CLASSES`` is THREE (``route_left,
#: route_straight, route_right``, ``refb.py:68``). Feeding ``route_pred``
#: straight through would send `route_left` to `follow` and `route_straight` to
#: `left` -- a scrambled nav that measures nothing.
#: ⛔ THE NAV NULL, DERIVED FROM SOURCE — not invented, and not ``nav_known``.
#: The question asked was whether ``RefCV3Model.forward``'s ``nav_known``
#: argument gives a principled null. MEASURED: it does NOT.
#:   * ``RefCConfig.nav_known_channel`` defaults to **False** (``refc.py:594``),
#:     nothing in the v3 path turns it on (0 references in ``refc_v3.py``, 0 in
#:     ``refc_v3_train._pin_trainer_cfg``), and
#:   * ``refc.py:2042-2045`` RAISES if ``nav_known`` is supplied while the gate
#:     is off ("it would be silently dropped. Turn the gate on or stop passing
#:     it").
#: ⇒ passing ``nav_known`` would be REFUSED, not principled. The model's OWN
#: no-nav path is ``nav_cmd=None``, which is also the published REF-C eval
#: convention (``refc.py:343-345``: *"every published REF-C number decodes with
#: nav_cmd=None -> index 0"*) and what DEPLOYMENT looks like — nav here is an
#: ORACLE input (provenance ego-future) that will not exist at deployment.
NAV_NULL = {
    "arm": "os_navzero",
    "how": "model(frames, nav_cmd=None, v0=v0, steps=steps) — a SEPARATE forward",
    "why_not_nav_known": (
        "RefCConfig.nav_known_channel is False for every v3 build (refc.py:594 "
        "default; 0 references in refc_v3.py and in _pin_trainer_cfg), and "
        "refc.py:2042-2045 RAISES if nav_known is supplied while the gate is "
        "off. There is no companion bit to set, so nav_known is not available "
        "as a null here."),
    "why_not_a_zero_embedding": (
        "zeroing nav_inj's output would be an INVENTED null the model was never "
        "trained to see. nav_cmd=None is the model's own documented path and "
        "the published REF-C eval convention."),
    "what_each_condition_removes": {
        "nav_true": "nothing — the clip's real v7.2 token, the input the run "
                    "trained on",
        "nav_shuffled": "the PAIRING between the window and its token. The nav "
                        "MARGINAL is preserved exactly (it is a permutation), so "
                        "the model still sees a plausible token on every window. "
                        "Answers: is the model using THIS window's nav?",
        "nav_zero": "the SIGNAL. Answers: what happens when the oracle nav is "
                    "absent, i.e. at deployment. ⛔ Not a stronger shuffle — a "
                    "different intervention.",
    },
    "what_nav_zero_removes_per_layer": {
        "tactical (E13 PhiTac)": "REMOVED ENTIRELY — refc_v3.py:437-441 guards "
                                 "the injection on `nav_cmd is not None`, so "
                                 "nav_to_tac is never added to z_tac_raw",
        "strategic (E13 ctx)": "REMOVED ENTIRELY — the same guard; nav_to_str is "
                               "never added to ctx, and out['nav_injected'] "
                               "reads False (refc_v3.py:471)",
        "core (measurement encoder)": (
            "⚠️ NOT REMOVED — COLLAPSED ONTO THE MAJORITY TOKEN. "
            "refc.py:2021-2024 substitutes one_hot(0) = 'follow'. NAV_COMMANDS "
            "(refc.py:136) has no 'unknown' entry and nav_known_channel is off, "
            "so at the core's input 'no nav' and 'a genuine follow' are "
            "BYTE-IDENTICAL — the exact defect nav_known_channel was written to "
            "fix (refc.py:594-604: 62.4 % of `follow` windows are a collapsed "
            "UNKNOWN)."),
    },
    "⛔ read_it_as": (
        "`os_navzero` removes nav COMPLETELY at 2 of the 3 layers it is injected "
        "at and PINS it to the majority class at the third. It is therefore a "
        "LOWER BOUND on how much this model leans on nav: a model that read nav "
        "only through the core would look LESS nav-dependent here than it is. "
        "State that beside any nav-dependence number taken from it."),
    "consequence_for_the_flat_arm": (
        "on a hier=False build there is no E13 path at all, so nav_cmd=None "
        "differs from the true nav ONLY through the core one-hot — and therefore "
        "`os_navzero` is BIT-IDENTICAL to `os` on exactly the windows whose token "
        "is already `follow` (index 0). That is the control that isolates the "
        "mechanism, and stack/tests/test_refcv3_arm.py runs it."),
    "⚠️ cross_call_float32_floor": (
        "`os_navzero` is produced by a SEPARATE forward call (nav_cmd=None is a "
        "whole-call property), while `os` and `os_navshuf` are two ROWS of one "
        "batched call. Different batch sizes take different GEMM kernel paths, "
        "so two arms that are the SAME computation still differ by a float32 "
        "floor. MEASURED on the flat smoke build, CPU: batch-2-row-0 vs batch-1 "
        "with the SAME nav = max 5.96e-07 m; the same call at MATCHED batch size "
        "with nav_cmd=None vs nav_cmd=0 = EXACTLY 0.0; a real nav difference "
        "(follow vs left) = 2.78e-02..3.41e-02 m, i.e. ~4.7e4x the floor. "
        "⛔ CONSEQUENCE: `trivial_profile.identical_to` uses a 1e-9 m threshold "
        "and therefore CANNOT resolve a cross-call arm as identical even when it "
        "is. Read `identical_to` for os-vs-os_navshuf (same call, exact) and "
        "IGNORE it for os-vs-os_navzero; the separation above is what makes a "
        "nav-zero difference readable, and it is 4-5 orders of magnitude clear."),
    "why_it_is_not_the_shuffle": (
        "MEASURED elsewhere (D-REFAV1-TAC-DECODER-PANEL): refav1's tactical head "
        "ranked turns at AUC 0.873 under true nav but collapsed to 0.520 "
        "(chance) under nav_zero, and a NAV-ONLY predictor beat the model "
        "outright (0.684 > 0.650). BACKLOG R39 binds every nav-conditioned claim "
        "to carry the nav-ZERO arm beside the nav-shuffle one."),
}
ARM_MEANING = {
    "os_navpred": "T1 -- STAR as os, but the nav token is the MODEL'S OWN "
                  "predicted route: argmax(route_logits) taken from the "
                  "nav_cmd=None forward (so NO oracle nav enters the arm at any "
                  "point), mapped to a NAV_COMMANDS index through the imported "
                  "refb_labels._ROUTE_TO_NAV. Answers 'how much of the supplied "
                  "route's value can the model produce from vision alone?' -- "
                  "i.e. the deployment condition on a corpus that ships no map. "
                  "ASYMMETRY, stated: the oracle token is per-CLIP while the "
                  "route head predicts per-WINDOW, so this arm's token can "
                  "change within a clip. Admissible under the 2026-08-03 goal "
                  "ruling (a PREDICTED goal is admissible; it carries no "
                  "situation-classifier output).",
    "os_navflip": "T1 (RULING OPEN) — as os with the nav token FLIPPED "
                  "(left<->right) on every window: the sharpest nav "
                  "intervention for taniteval.nav_compliance. OPTIONAL "
                  "(--with-navflip); shuffle + zero stay the REQUIRED controls.",
    "os": "T1 (RULING OPEN) — ONE forward pass at t0: observed frames + the "
          "clip's v7.2 nav token + the MEASURED v0, path = the model's OWN "
          "sel_score_v3 selection out['traj']. NEVER a_star. No action input, "
          "no rollout, no loop to close — hence 'os', never 'cl'.",
    "os_navshuf": "T1 (RULING OPEN) — as os with nav_cmd PERMUTED across the "
                  "eval windows: breaks the PAIRING while preserving the nav "
                  "marginal. Answers 'is the model using THIS window\'s nav?' "
                  "(nav is an INPUT; an echo scores well and learns nothing)",
    "os_navzero": "T1 (RULING OPEN) — ⭐ as os with nav WITHHELD "
                  "(nav_cmd=None): the E13 injection into the tactical and "
                  "strategic layers is SKIPPED ENTIRELY and the core collapses "
                  "to the 'follow' one-hot. Answers 'what does the model do when "
                  "the ORACLE nav is not there' — i.e. what DEPLOYMENT looks "
                  "like. ⛔ NOT interchangeable with the shuffle; BACKLOG R39 "
                  "binds both. See NAV_NULL for the per-layer breakdown and why "
                  "it is a LOWER BOUND on nav dependence.",
    "ha": "T1 — HOLD-ACTION control: the (a, steer) that CLOSES at t0 held for "
          "the horizon. Consumes no recorded future. NOT the floor — a held "
          "noisy steer drifts, so this arm can be WORSE than trivial.",
    "ha0": "T1 — ⭐ CONSTANT VELOCITY: a = 0, kappa = 0 at the measured v0, i.e. "
           "a straight line at constant speed. The STRONGEST TRIVIAL BASELINE "
           "and the ONLY arm bit-comparable with refav1's (zero is zero in "
           "either action unit). ⛔ NOT 'the echo test's real bar' — that line "
           "stood here and stack/tanitad/eval/echo_gate.py RETRACTS it by name: "
           "`ha` is 2.2x harder, and the bar is `ha` AND `ha0_ext` TOGETHER.",
    "ha0_ext": "T1 — ⭐⭐ THE ECHO CONTROL: constant a0 AND constant curvature "
               "k0, both read at the MEASURED t0 (a0 the backward difference "
               "of v, k0 the recorded channel AT t0 — where `ha` holds the one "
               "at t0-1). The SHARPENED form of `ha`, and the OTHER HALF of the "
               "acceptance bar: beating `ha` while TYING `ha0_ext` means the "
               "gain is echo, and the gate reads it that way "
               "(stack/tanitad/eval/echo_gate.py::ha0_ext). Derived by the ONE "
               "shared implementation refav1_arm.hold_ext_controls, called here "
               "with stride=1 — never re-derived (Rung A1, 2026-09-05).",
    "oracle_sel": "T0 — the a_star-selected anchor's refinement: a_star is the "
                  "GT-NEAREST anchor (refc_v3_train.py:460), so this is the "
                  "CEILING, not a driveable arm. Never compared to a T1 number.",
}
#: ⭐⭐ THE EVAL-TIME ABLATIONS — ``PREREG_REFCV4B_HIERARCHY_EVAL.md`` §3.
#:
#: ⛔ WHY THIS DICT IS THE REGISTRY AND THE PREREG'S §7 IS NOT. §7's escalation
#: names SEVEN switches ({gstr_zero, gstr_shuffle, e7_off, e9_off, h19_off,
#: ego_zero, sel_refined}) and OMITS the eighth — the frame-blind DELIBERATE
#: REGRESSION, §3's last row, on which the whole panel's validity depends: a
#: gate that has never been shown to FAIL an image-blind arm certifies nothing
#: (H-ECHO-4, where an ADE-scored gate once passed an echoing arm). An
#: incomplete list of what is missing is the same failure class as an
#: incomplete list of what is present. ⇒ §3 is authoritative; this dict mirrors
#: §3 and ``stack/tests/test_refcv3_ablations.py`` pins the bijection.
#:
#: Every ablation is EVAL-TIME and leaves the rest of the forward bit-identical.
#: ⚠️ An eval-time knockout measures the trained model's RELIANCE on an edge —
#: a lower bound on what the edge bought in training and an upper bound on
#: nothing — so each row states the regime it creates and whether training ever
#: produced it (`seen_in_training`), exactly as the prereg's §3 column does.
ABLATIONS = {
    "gstr_zero": {
        "prereg_arm": "g_str-ZERO",
        "mechanism": ("forward hook on `str_goal_head` returning (1, 0, 0), so "
                      "the model's own normalisation yields the straight-ahead "
                      "constant g_str = (1, 0, 0) before E4/E7"),
        "seen_in_training": "at init only (zero-init FiLM)",
        "tests": "the strategic goal conditions the tactical/operative levels (E4)",
        "applies_at": "model",
        "requires": "hier build (str_goal_head exists only on the goal cascade)",
    },
    "gstr_shuffle": {
        "prereg_arm": "g_str-SHUFFLE",
        "mechanism": ("the same hook, fed the g_str another WINDOW produced — "
                      "read from a banked FULL dump (--gstr-bank) and permuted "
                      "with --gstr-shuffle-seed. ⛔ NOT a batch permutation: "
                      "this harness's batch rows are the nav CONDITIONINGS of "
                      "ONE window, so permuting them would permute nav, not "
                      "windows, and would silently answer a different question"),
        "seen_in_training": "no",
        "tests": "the goal carries WINDOW-specific information downstream",
        "applies_at": "model",
        "requires": "hier build AND --gstr-bank <FULL dump dir> on the same grid",
    },
    "e7_off": {
        "prereg_arm": "E7-OFF",
        "mechanism": ("the hierarchy hook returns `target_latent=None`, so "
                      "refc.py:2208 leaves it None and the decoder skips the "
                      "FiLM entirely (refc.py:1508 `tgt_film is not None and "
                      "target_latent is not None`)"),
        "seen_in_training": "at init only",
        "tests": "the tactical latent conditions the decoder",
        "applies_at": "model",
        "requires": "hier build",
    },
    "e9_off": {
        "prereg_arm": "E9-OFF",
        "mechanism": ("`model.goal_gate` set to 0.0, so refc_v3.py:1037's "
                      "graft = goal_gate * score is exactly zero and the "
                      "blended rank collapses to sel_score"),
        "seen_in_training": "at init (the gate is zero-init)",
        "tests": "goal-distance selection improves the pick",
        "applies_at": "model",
        "requires": "hier build",
    },
    "h19_off": {
        "prereg_arm": "H19-OFF",
        "mechanism": ("the kin3-derived anchor prior removed from the "
                      "confidence: whichever of `decoder.maneuver_to_anchor` / "
                      "`decoder.lat_to_anchor` / `decoder.lon_to_anchor` the "
                      "build carries is set to None, so refc.py:1572-1581's "
                      "`terms` loses them"),
        "seen_in_training": "no",
        "tests": "the model's own tactical prediction improves its selection",
        "applies_at": "model",
        "requires": "a decoder carrying at least one anchor-prior head",
        "⛔ prereg_correction": (
            "PREREG_REFCV4B_HIERARCHY_EVAL.md §3 registers this arm as "
            "`decoder.maneuver_to_anchor = None`. MEASURED 2026-09-05: on "
            "every REF-C v3/v4 build — including refcv4b — "
            "`core.factored_maneuver` is True (refc_v3.py:437, :615), so "
            "refc.py:1214-1221 builds the FACTORED pair "
            "(lat_to_anchor + lon_to_anchor) and leaves `maneuver_to_anchor` "
            "None. The literal registered mechanism would have found nothing "
            "to remove on the very checkpoint it is registered for. This "
            "switch removes the heads that EXIST and names them in the "
            "record; the prereg needs an erratum, not the code a workaround."),
    },
    "ego_zero": {
        "prereg_arm": "EGO-ZERO",
        "mechanism": ("ego_state[:, 4] = 0 (keep = 0, the X15 regime — the "
                      "model itself then zeroes the values beside the flag) "
                      "AND v0 withheld at the core (v0=None, which is how "
                      "refc.py derives keep=0 there). ⚠️ The model-free "
                      "controls still integrate the MEASURED v0: they are "
                      "controls, not arms, and must read bit-identically "
                      "across ablations"),
        "seen_in_training": "yes (ego_dropout 0.5)",
        "tests": "the ego channels are used (E11') — a robustness read, NOT a "
                 "thesis test",
        "applies_at": "call-site",
        "requires": "an ego_state_inject build",
    },
    "sel_refined": {
        "prereg_arm": "SEL-REFINED",
        "mechanism": ("`decoder.sel.refined = True` and "
                      "`decoder.sel.score_emitted = True` (0 params) — rank the "
                      "refined fan by the refined confidence read from the "
                      "EMITTED estimate"),
        "seen_in_training": "no",
        "tests": "the selection surface is the lever the implementation audit named",
        "applies_at": "model",
        "requires": ("diffusion steps > 0 — ⛔ at steps == 0 refc.py:1596 "
                     "leaves `refined is conf` BY CONSTRUCTION and :1630 gates "
                     "score_emitted on `steps > 0`, so the switch would parse "
                     "and change nothing. REFUSED there rather than silently "
                     "inert: a flag that does nothing is worse than a missing "
                     "one"),
    },
    "frames_blind": {
        "prereg_arm": "DELIBERATE REGRESSION",
        "mechanism": ("every observed frame replaced by the window's own "
                      "scalar mean, so the encoder sees a constant image and "
                      "the arm is an echo BY CONSTRUCTION"),
        "seen_in_training": "no",
        "tests": ("⭐⭐ THAT THE INSTRUMENTS CAN FAIL. If this arm PASSES the "
                  "echo gate, or reads FOLLOWS_NAV with a high compliance, the "
                  "PANEL IS VOID (PREREG_REFC_V4 §7 OUTCOME IV). The "
                  "model-free controls ha / ha0 / ha0_ext read no frames and "
                  "must come back BIT-IDENTICAL to the FULL run — that is the "
                  "internal control on this arm"),
        "applies_at": "call-site",
        "requires": "nothing",
    },
}
#: The four §3 rows that ALREADY had a route before Rung A1, so the completeness
#: test can assert a bijection with §3's twelve rather than only with the eight.
ABLATIONS_PREEXISTING = {
    "FULL": "no flag — the final checkpoint as trained IS the default run",
    "nav-ZERO": "arm `os_navzero`, on by default; --no-navzero removes it",
    "nav-SHUFFLE": "arm `os_navshuf`, on by default; --no-navshuf removes it",
    "nav-FLIP": "--with-navflip",
}
#: ⛔ ARMS THAT DO NOT EXIST FOR THIS MODEL. Written into the record with the
#: structural reason, so a reader never sees a silently missing column.
ABSENT_ARMS = {
    "ol": {
        "status": "ABSENT",
        "arm": "ol",
        "tier_if_it_existed": "T0",
        "reason": ("`ol` is 'the RECORDED future (a, steer) integrated from v0'. "
                   "refcv3 CONSUMES NO ACTIONS (refc_v3.py:480 — the forward "
                   "signature has no action argument), so integrating the "
                   "recorded actions is not a rollout OF THIS MODEL: it is a "
                   "property of the corpus and of the unicycle, identical for "
                   "every refcv3 checkpoint ever trained. Emitting it under this "
                   "model's name would put the same name on two different "
                   "objects across the H-vs-F table — the exact failure "
                   "D-HF-COMPARABILITY forbids."),
        "for_comparison": ("use `ha0` — the ONLY bit-comparable arm across "
                           "refav1 and refcv3 — as the shared floor; the "
                           "kinematic-contract control that `ol` provides lives "
                           "in refav1_arm.py, where the model actually consumes "
                           "actions."),
    },
}
#: ⚠️ THE OPEN RULING, stamped on every emitted block (BACKLOG R30 / RESULT §5.4 W6).
TIER_RULING = {
    "arm": "os",
    "stamped": "T1",
    "status": "UNRULED",
    "decided_by": "PI / Master Mind — NOT a FlyWheel",
    "question": ("does EVAL_DOCTRINE admit as T1 a model that consumes NO "
                 "actions at all? Its T1 row reads 'the predictor consumes the "
                 "decoder/planner's own actions', which does not literally cover "
                 "refcv3."),
    "benchmarks_recommendation": (
        "ADMIT IT, flagged as a recommendation: the doctrine's PURPOSE is to "
        "keep future information out of inference, and a correctly-gated refcv3 "
        "forward pass (frames <= t0, nav token, measured v0, selection by "
        "sel_score_v3 and never by a_star) admits none — but keep the DISTINCT "
        "arm name `os` so no reader believes two `cl` columns describe the same "
        "procedure."),
    "why_the_numbers_are_valid_either_way": (
        "every headline statistic here is a MARGIN over `ha0` measured on the "
        "same windows with the same instrument (`os - ha0`), so the ruling "
        "changes the LABEL on the row, never the arithmetic in it."),
}
_TIER_NOTE = dict(t1._TIER_NOTE)
NAV_SOURCES = ("auto", "v72", "none")
_UNVERIFIED_ON_REAL_CKPT = (
    "UNVERIFIED on a real checkpoint — this box is forbidden from contacting the "
    "training pod `tanitad-refcv3`; validated on a random-init RefCV3Model at "
    "refc_v3_smoke_config over a synthetic 3-episode slice only")
_SIDECAR_DOC = {
    "sel_idx": "the anchor the MODEL selected (out['sel_idx'])",
    "sel_idx_base": "hier only: the pre-graft core selection (out['sel_idx_base'])",
    "sel_score_max": "max of the ranking score the selection argmaxed over",
    "a_star": "the GT-NEAREST anchor (refc_v3_train.py:460) — the ORACLE, banked "
              "as a diagnostic; the deployed arm NEVER selects with it",
    "anchor_acc": "1.0 where anchor_logits.argmax == a_star (refc_v3_train.py:642); "
                  "chance = 1/128 = 0.0078",
    "sel_agrees_oracle": "1.0 where sel_idx == a_star — how much of `traj` was "
                         "selection rather than refinement",
    "goal_gate": "hier only: out['goal_gate_value'] — the zero-init E9 gate",
    "goal_score_absmean": "hier only: the SCALE the gate multiplies. The gate "
                          "alone cannot distinguish 'has not opened yet' from "
                          "'will never open'; both are required.",
    "goal_dist_sel": "hier only: the selected anchor's goal distance",
    "lat_label/lon_label": "the v7.2 tactical class ids (IGNORE_ID -> -100)",
    "route_label": "the v2.1 route target (ROUTE_UNKNOWN/invalid -> -100)",
    "nav_cmd/nav_cmd_shuf/nav_valid": "the fed token, its permutation, validity",
    "{lat,lon,route}_pred_{nav_true,nav_shuffled,nav_zero}": "the DECLARED head "
        "argmaxes under each nav conditioning (-1 = the head does not exist or "
        "the conditioning was not rolled). ⭐ nav_zero comes from the SEPARATE "
        "nav_cmd=None forward, not from a row of the fed batch — see NAV_NULL",
    "nav_injected_true/nav_injected_zero": "out['nav_injected'] (refc_v3.py:471) "
        "under each forward: the E13 edge is LIVE under the fed nav and DEAD "
        "under nav_cmd=None. A conditioning edge that silently no-ops is the "
        "advertised-but-inert defect; this makes it visible per window.",
    "ha_controls": "the held (a, channel-1) actually integrated, in the run's "
                   "declared action units",
    # ⭐ NAV-COMPLIANCE sidecar (taniteval.nav_compliance, 2026-09-05) — the
    # BEHAVIOURAL readouts per conditioning. The old strategic metric scored a
    # HEAD against a label that is a bijection of the fed token; these are what
    # the model DOES, and a path cannot be produced by copying a token.
    "plan_full_{cond}": "[S, 2] the emitted plan out['traj'] at EVERY model slot "
                        "(not the dump grid) under each conditioning",
    "gstr_{cond}": "[3] the strategic goal g_str = (cos, sin, dist_pref) (hier only)",
    "sel_idx_{cond}": "the SELECTED anchor under each conditioning",
    "sel_bank_{cond}": "[S, 2] the selected anchor's UNREFINED bank path "
                       "(out['anchor_bank'][sel_idx]) — the selection surface",
    "fan_term_heading_{cond}": "[N_anchors] terminal heading of every REFINED "
                               "candidate (out['anchor_traj']) — fan coverage",
    "reach_keep_{cond}": "[N_anchors] the S2 reach mask (1.0 = survivor)",
    "gt_future_ext": "[60, 4] the recorded future poses (WORLD frame), clamped at "
                     "the episode end; gt_future_valid_ext [60] says where",
    "pose_last": "[4] (x, y, yaw, v) at t0 (world frame)",
    "ego_t0": "[4] (v0, a_long, yaw_rate, curvature) at t0 — "
              "refc_v3.ego_state_at_t0 without the keep bit",
    "ep_poses": "[T, 4] the episode's PROVIDER poses, once per file (for the "
                "label time-base control)",
}


def _p(*a):
    print(*a, flush=True)


#: ⛔ Exceptions that mean OUR CODE IS WRONG, never "the input was absent".
#: An `except Exception` that converts these into a polite refusal is how a
#: BINDING metric family goes missing from every eval without anything failing.
#: (`ValueError` is deliberately NOT here — it is the honest way a loader says
#: "this file is not what you said it was".)
DEFECT_EXCEPTIONS = (TypeError, AttributeError, NameError, IndexError,
                     UnboundLocalError, ZeroDivisionError)


def _refused(reason, tier, n=0):
    """The binding shape for a family/metric whose INPUTS are missing here."""
    return {"status": "REFUSED", "reason": reason, "n": int(n), "tier": tier,
            "estimator": "n/a — inputs missing (WORK ITEM, not a pass)"}


# --------------------------------------------------------------------------- #
# config (de)serialisation — nested dataclasses, tuples survive JSON            #
# *(harvested from the rescued draft, which derived it)*                        #
# --------------------------------------------------------------------------- #
def _default_of(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


def cfg_from_dict(cls, d: dict):
    """JSON dict -> dataclass ``cls`` (recursively), REFUSING unknown fields by
    name: a checkpoint written by a NEWER model file must be evaluated with that
    file, never with fields silently dropped."""
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
        v, dflt = d[f.name], _default_of(f)
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

    ⚠️ ``refc_v3_train.train`` WRITES NO MODEL CONFIG (``refc_v3_train.py:1097``
    stamps arm/argv/horizons/image_hw/vocab/nav, not a ``RefCV3Config``), and the
    checkpoint carries only ``model``/``opt``/``step``. The model is a function of
    ``--arm/--size/--smoke/--image-hw/...``, so it is rebuilt HERE through the
    trainer's OWN ``build_parser`` + ``_pin_trainer_cfg`` on the recorded
    ``argv`` — never re-derived. A config.json carrying the adapter extension
    ``refcv3_arm_model_cfg`` (a full RefCV3Config dict) is rebuilt from that
    instead. *(Harvested from the rescued draft.)*"""
    tr = trainer()
    from tanitad.refs import refc
    from tanitad.refs import refc_v3 as v3
    argv = config.get("argv")
    args = None
    if argv:
        try:
            args = tr.build_parser().parse_args(list(argv))
        except SystemExit:
            raise SystemExit(
                f"[refcv3_arm] config.json argv does not parse with this box's "
                f"refc_v3_train.build_parser — trainer/checkpoint version skew. "
                f"argv={list(argv)[:12]}") from None
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
        if getattr(args, "graft_lan", False) or getattr(args, "goal_str", False):
            cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))
        src = ("config.json[argv] -> refc_v3_train.build_parser + "
               "_pin_trainer_cfg (the trainer's own build path)")
    return cfg, args, src


def cross_check_config(config: dict, cfg, model) -> dict:
    """Every fact ``config.json`` states about the model must hold for the rebuilt
    one. A contradiction is a REFUSAL naming BOTH values (the
    ``adopt_ckpt_geometry`` lesson in ``t1_eval``) — never a silent override in
    either direction. *(Harvested from the rescued draft.)*"""
    from tanitad.refs import refc_v3 as v3
    checks, conflict = {}, []

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
                         f"writes one beside ckpt.pt (refc_v3_train.py:1097) and "
                         f"the model cannot be rebuilt without it (pass --config)")
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
            f"SHAPE — config.json/argv describe a different build than the "
            f"checkpoint. Refusing; fix the config, never the weights.\n"
            f"{str(ex)[:1500]}") from None
    strict_rep = {"missing_keys": list(res.missing_keys),
                  "unexpected_keys": list(res.unexpected_keys)}
    # ⭐ INERT-BUFFER TOLERANCE (2026-09-05, REF-C RL-readiness WP). refcv4-b
    # added the PERSISTENT buffer `core.decoder.anchor_controls` (refc.py:1168),
    # which `roll_bank` reads ONLY when the decoder is `v0_conditioned`. A
    # checkpoint trained before it existed (refcv3 @ 40,284, the published HF
    # weights) is therefore MISSING a key the rebuilt model never reads, and the
    # strict refusal was a FALSE refusal — MEASURED: STAGE 0 of the RL chain died
    # on it with the md5-verified base. Tolerated only when (a) the decoder is NOT
    # v0-conditioned, (b) nothing else is missing and nothing is unexpected — and
    # RECORDED in the provenance, never silent.
    _dec = getattr(model.core, "decoder", None)
    inert = sorted(k for k in res.missing_keys
                   if k.endswith(".anchor_controls") and _dec is not None
                   and not bool(getattr(_dec, "anchor_v0_cond", False)))
    strict_rep["tolerated_inert_buffers"] = inert
    real_missing = [k for k in res.missing_keys if k not in inert]
    if (real_missing or res.unexpected_keys) and not allow_nonstrict:
        raise SystemExit(f"[refcv3_arm] ⛔ NON-STRICT LOAD: missing "
                         f"{real_missing[:8]} unexpected "
                         f"{list(res.unexpected_keys)[:8]} — the checkpoint and "
                         f"the rebuilt model disagree. Pass --allow-nonstrict "
                         f"only for a deliberate diagnostic, and say so.")
    # ---- EVAL-TIME NEUTRALISATION, recorded -------------------------------- #
    # ``refc_select.apply_seam_clamp`` RAISES after `seam_fail_patience`
    # CONSECUTIVE saturated calls. The counter lives on the MODEL, not on a
    # batch, so a trained gate sitting above the clamp would kill a 20k-window
    # eval at window 50 — a TRAINING-DYNAMICS report firing inside an eval. The
    # clamp itself stays (it IS the emitted score); only the fail-loud patience
    # is switched off, and the saturation telemetry is banked per window instead.
    # No parameter changes; the state_dict is byte-identical either way.
    # *(Harvested from the rescued draft, which found this.)*
    overrides = {}
    if getattr(cfg, "seam_fail_patience", 0):
        overrides["v3.seam_fail_patience"] = {"trained": cfg.seam_fail_patience,
                                              "eval": 0}
        cfg.seam_fail_patience = 0
    sel = getattr(getattr(model.core, "decoder", None), "sel", None)
    if sel is not None and getattr(sel, "seam_fail_patience", 0):
        overrides["core.decoder.sel.seam_fail_patience"] = {
            "trained": sel.seam_fail_patience, "eval": 0}
        sel.seam_fail_patience = 0
    if hasattr(model.core, "cfg"):
        model.core.cfg.seam_fail_patience = 0
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mode = getattr(targs, "mode", "diffusion") if targs is not None else "diffusion"
    steps = cfg.core.decoder.diffusion_steps if mode == "diffusion" else 0
    prov = {"ckpt": ckpt_path, "step": ck.get("step"),
            "config_json": side, "rebuilt_from": src,
            "config_cross_checks": checks, "state_dict_load": strict_rep,
            "eval_time_cfg_overrides": overrides,
            "cfg": dataclasses.asdict(cfg),
            "param_breakdown": v3.param_breakdown_v3(model),
            "arm": "hier" if cfg.hier else "flat",
            "hier": bool(cfg.hier),
            "tac_vocab_version": cfg.tac_vocab_version,
            "horizons": list(cfg.core.trajectory.horizons),
            "n_anchors": int(cfg.core.anchors.n_anchors),
            "window": int(cfg.core.window),
            "decoder_steps": int(steps), "decoder_mode": mode,
            "nav_from_v7_trained": bool(config.get("nav_from_v7", False)),
            "nav_cmd_derivation_trained": config.get("nav_cmd_derivation"),
            "train_labels_manifest": config.get("v7_labels"),
            "eval_labels_md5_at_train": (((config.get("nav_from_v7_stats") or {})
                                          .get("eval") or {}).get("label_md5")),
            "u8_batches_trained": config.get("u8_batches")}
    return model, cfg, targs, prov


# --------------------------------------------------------------------------- #
# the eval-time ablations (PREREG_REFCV4B_HIERARCHY_EVAL.md §3)                  #
# --------------------------------------------------------------------------- #
class AblationState:
    """What the roll loop needs to know, and the per-window channel the g_str
    hook reads. Held on ONE object so the hook and the loop cannot drift."""

    def __init__(self):
        self.names: list[str] = []
        self.frames_blind = False
        self.ego_zero = False
        self.gstr_mode: str | None = None     # None | "zero" | "shuffle"
        self.gstr_target = None               # [3] for the CURRENT window
        self.gstr_bank: dict = {}             # (clip_id, ws) -> [3]
        self.gstr_perm: dict = {}             # (clip_id, ws) -> (clip_id, ws)
        self.n_gstr_windows = 0
        self.verified = {}                    # ablation -> what was observed


def resolve_ablations(a) -> list[str]:
    """``--ablate`` + the explicitly-named ``--ablate-frames``, deduplicated.

    ``--ablate-frames`` exists as its OWN flag because the frame-blind
    deliberate regression is the arm the panel's validity rests on and it must
    be nameable without remembering an enum member."""
    names = list(getattr(a, "ablate", None) or [])
    if getattr(a, "ablate_frames", False) and "frames_blind" not in names:
        names.append("frames_blind")
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"[refcv3_arm] unknown ablation(s) {unknown}; known: "
                         f"{sorted(ABLATIONS)}")
    return names


def load_gstr_bank(bank_dir: str, seed: int):
    """``(clip_id, ws) -> g_str [3]`` from a banked FULL dump, plus the seeded
    permutation OVER WINDOWS.

    ⛔ The permutation is over windows because that is the registered mechanism.
    A batch-level permutation in THIS harness would permute the nav
    conditionings of one window — a different intervention wearing the same
    name, which is the `--with-navzero` defect this tool already carries a
    warning about."""
    man_p = os.path.join(bank_dir, "manifest.json")
    if not os.path.exists(man_p):
        raise SystemExit(f"[refcv3_arm] --gstr-bank {bank_dir} has no "
                         f"manifest.json — it is not a refcv3 dump")
    with open(man_p, encoding="utf-8") as fh:
        man = json.load(fh)
    by_fi = {int(e["file_index"]): e["clip_id"] for e in man.get("episodes", [])}
    bank: dict = {}
    for f in sorted(glob.glob(os.path.join(bank_dir, "ep*.npz"))):
        fi = int(os.path.basename(f)[2:5])
        dp = os.path.join(bank_dir, "decisions", f"ep{fi:03d}.npz")
        if not os.path.exists(dp):
            raise SystemExit(f"[refcv3_arm] --gstr-bank: {dp} is missing; the "
                             f"bank needs the decisions sidecar (gstr_nav_true)")
        with np.load(f) as d, np.load(dp) as dd:
            if "gstr_nav_true" not in dd.files:
                raise SystemExit(
                    f"[refcv3_arm] --gstr-bank: {dp} carries no "
                    f"`gstr_nav_true` — the bank was rolled on a FLAT build, "
                    f"which has no strategic goal to shuffle")
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            gs = np.asarray(dd["gstr_nav_true"]).astype(np.float32)
            cid = by_fi.get(fi)
            if cid is None:
                raise SystemExit(f"[refcv3_arm] --gstr-bank: no clip_id for "
                                 f"file_index {fi} in the bank manifest")
            if gs.shape[0] != ws.shape[0]:
                raise SystemExit(f"[refcv3_arm] --gstr-bank: {fi} has "
                                 f"{gs.shape[0]} goals for {ws.shape[0]} windows")
            for j, w in enumerate(ws.tolist()):
                bank[(str(cid), int(w))] = gs[j].reshape(-1)[:3]
    if not bank:
        raise SystemExit(f"[refcv3_arm] --gstr-bank {bank_dir} yielded no windows")
    keys = sorted(bank)
    perm = np.random.default_rng(int(seed)).permutation(len(keys))
    mapping = {keys[i]: keys[int(perm[i])] for i in range(len(keys))}
    n_same = sum(1 for k, v in mapping.items() if k == v)
    stats = {"n_windows": len(keys), "seed": int(seed),
             "n_fixed_points": int(n_same),
             "frac_changed": round(1.0 - n_same / max(1, len(keys)), 6),
             "bank_dir": bank_dir,
             "⚠️": ("a fixed point feeds a window ITS OWN goal; those windows "
                    "are not ablated and the changed fraction is the honest n")}
    return bank, mapping, stats


def _gstr_forward_hook(state: "AblationState"):
    """Replaces ``str_goal_head``'s output so the model's OWN normalisation
    (bearing = g[:2]/||g[:2]||, dist = tanh(g[2])) reproduces the intended
    ``g_str`` EXACTLY — the intervention is on the strategic goal, never on the
    normalisation."""
    import torch

    def hook(_mod, _inp, out):
        if state.gstr_mode is None:
            return None
        g = out.new_zeros(out.shape[0], 3)
        if state.gstr_mode == "zero":
            g[:, 0] = 1.0                    # bearing (1, 0); tanh(0) = 0
            return g
        tgt = state.gstr_target
        if tgt is None:
            raise RuntimeError("gstr_shuffle: no banked goal set for this "
                               "window — the loop and the hook are out of step")
        bx, by, dist = float(tgt[0]), float(tgt[1]), float(tgt[2])
        n = math.hypot(bx, by)
        if n < 1e-9:
            raise RuntimeError(f"gstr_shuffle: banked bearing has norm {n}")
        dist = max(-1.0 + 1e-6, min(1.0 - 1e-6, dist))
        g[:, 0], g[:, 1] = bx / n, by / n
        g[:, 2] = math.atanh(dist)
        return g

    return hook


def apply_ablations(model, cfg, names, *, steps: int, feed_ego: bool,
                    gstr_bank: str | None = None, gstr_seed: int = 0):
    """Apply the eval-time ablations and return ``(state, record)``.

    ⛔ EVERY precondition is checked and REFUSED loudly. A flag that parses and
    changes nothing is worse than a missing one: it produces a table that reads
    like a knockout and is a copy of the FULL arm."""
    state = AblationState()
    state.names = list(names)
    per: dict = {}
    if not names:
        return state, {"applied": [], "requested": [],
                       "prereg": "PREREG_REFCV4B_HIERARCHY_EVAL.md §3",
                       "is": "FULL — the final checkpoint as trained"}
    hier = bool(getattr(cfg, "hier", False))
    dec = getattr(getattr(model, "core", None), "decoder", None)
    for n in names:
        spec = ABLATIONS[n]
        ev: dict = {"prereg_arm": spec["prereg_arm"],
                    "mechanism": spec["mechanism"],
                    "seen_in_training": spec["seen_in_training"]}
        if n in ("gstr_zero", "gstr_shuffle", "e7_off", "e9_off") and not hier:
            raise SystemExit(f"[refcv3_arm] ablation {n!r} needs a HIER build; "
                             f"this checkpoint is flat and the edge it ablates "
                             f"does not exist. Refusing rather than reporting "
                             f"an unchanged arm as a knockout.")
        if n == "gstr_zero":
            state.gstr_mode = "zero"
            model.str_goal_head.register_forward_hook(_gstr_forward_hook(state))
            ev["injected_g_str"] = [1.0, 0.0, 0.0]
        elif n == "gstr_shuffle":
            if not gstr_bank:
                raise SystemExit(
                    "[refcv3_arm] ablation 'gstr_shuffle' needs --gstr-bank "
                    "<a FULL dump dir on the SAME grid>: the registered "
                    "mechanism permutes g_str ACROSS WINDOWS, and this "
                    "harness's batch rows are the nav conditionings of ONE "
                    "window. Permuting them would permute nav, not windows.")
            state.gstr_mode = "shuffle"
            state.gstr_bank, state.gstr_perm, ev["bank"] = \
                load_gstr_bank(gstr_bank, gstr_seed)
            model.str_goal_head.register_forward_hook(_gstr_forward_hook(state))
        elif n == "e7_off":
            _orig = model._hook

            def _hook_no_e7(cache, nav_cmd=None, ego_state=None, _o=_orig):
                inner = _o(cache, nav_cmd, ego_state)

                def wrapped(pooled_seq, ctx):
                    hk = dict(inner(pooled_seq, ctx))
                    hk["target_latent"] = None      # refc.py:2208 keeps it None
                    return hk
                return wrapped

            model._hook = _hook_no_e7
            ev["effect"] = ("the decoder's tgt_film is skipped "
                            "(refc.py:1508 gates on target_latent is not None)")
        elif n == "e9_off":
            before = float(model.goal_gate.detach().reshape(()).item())
            model.goal_gate.data.fill_(0.0)
            ev["goal_gate_before"] = before
            ev["goal_gate_after"] = 0.0
            if before == 0.0:
                ev["INERT"] = (
                    "the trained goal_gate was ALREADY exactly 0.0, so E9 "
                    "contributed nothing to this checkpoint and knocking it "
                    "out cannot change the forward. That is a FINDING about "
                    "the checkpoint (Caveat-B: the zero-init gate never "
                    "opened), NOT a defect of the switch — and it means the "
                    "E9-OFF row of the panel is uninformative and must be "
                    "reported as such, never as 'no effect, therefore the "
                    "seam is inert at eval'.")
                _p(f"  ⛔ ablation e9_off is INERT: {ev['INERT']}")
        elif n == "h19_off":
            # ⛔ NOT ONLY `maneuver_to_anchor` — see the registry's
            # `prereg_correction`. A v3/v4 build carries the FACTORED pair and
            # leaves the unfactored head None, so the prereg's literal
            # mechanism would remove nothing on refcv4b itself.
            heads = ("maneuver_to_anchor", "lat_to_anchor", "lon_to_anchor")
            present = [h for h in heads
                       if dec is not None and getattr(dec, h, None) is not None]
            if not present:
                raise SystemExit(
                    "[refcv3_arm] ablation 'h19_off': this decoder carries NO "
                    f"anchor-prior head ({', '.join(heads)} are all None), so "
                    f"the prior it removes is already absent. Refusing rather "
                    f"than reporting an unchanged arm as a knockout.")
            ev["removed"] = {h: type(getattr(dec, h)).__name__ for h in present}
            ev["factored"] = bool(getattr(cfg.core, "factored_maneuver", False))
            for h in present:
                setattr(dec, h, None)
            _p(f"  [ablation] h19_off removed {present} "
               f"(factored_maneuver={ev['factored']})")
        elif n == "sel_refined":
            sel = getattr(dec, "sel", None)
            if sel is None:
                raise SystemExit("[refcv3_arm] ablation 'sel_refined': the "
                                 "decoder carries no selection config")
            if int(steps) <= 0:
                raise SystemExit(
                    "[refcv3_arm] ablation 'sel_refined' needs diffusion "
                    "steps > 0. At steps == 0 refc.py:1596 leaves `refined is "
                    "conf` BY CONSTRUCTION and :1630 gates score_emitted on "
                    "`steps > 0`, so the switch would parse and change "
                    "NOTHING. Refusing: a flag that does nothing is worse "
                    "than a missing one.")
            ev["before"] = {"refined": bool(sel.refined),
                            "score_emitted": bool(sel.score_emitted)}
            sel.refined = True
            sel.score_emitted = True
            ev["after"] = {"refined": True, "score_emitted": True}
            ev["params_changed"] = 0
        elif n == "ego_zero":
            if not feed_ego:
                raise SystemExit(
                    "[refcv3_arm] ablation 'ego_zero' needs an "
                    "ego_state_inject (v4) build: this build is fed no ego "
                    "block, so zeroing its keep bit would change nothing.")
            state.ego_zero = True
            ev["effect"] = ("ego_state[:, 4] = 0 on every forward AND v0=None "
                            "at the core (which is how refc.py derives keep=0 "
                            "there); the model-free controls keep the MEASURED "
                            "v0 and must stay bit-identical across ablations")
        elif n == "frames_blind":
            state.frames_blind = True
            ev["effect"] = ("frames := their own scalar mean; ha / ha0 / "
                            "ha0_ext read no frames and MUST come back "
                            "bit-identical to the FULL run — that is the "
                            "internal control on the deliberate regression")
        per[n] = ev
    rec = {"applied": list(names), "requested": list(names),
           "prereg": "PREREG_REFCV4B_HIERARCHY_EVAL.md §3",
           "per_ablation": per,
           "⛔ read_with": ("EVERY number in this record was produced under the "
                           "ablation(s) above. An eval-time knockout measures "
                           "the trained model's RELIANCE on an edge — a lower "
                           "bound on what the edge bought in training and an "
                           "upper bound on nothing."),
           "tier_unchanged": ("the ablations do not change what is fed from the "
                              "future; every arm keeps its ARM_TIERS stamp")}
    _p(f"[ablation] {names} — {'; '.join(ABLATIONS[n]['prereg_arm'] for n in names)}")
    return state, rec


# --------------------------------------------------------------------------- #
# grid + kinematics                                                            #
# --------------------------------------------------------------------------- #
def grid_slots(horizons, grid: str) -> dict:
    """``--grid`` -> the model slots that ARE that grid. INDEX-SELECT ONLY: a grid
    the model cannot serve by index-select is REFUSED, never interpolated.
    *(Harvested from the rescued draft.)*"""
    if grid not in GRIDS:
        raise SystemExit(f"[refcv3_arm] unknown --grid {grid!r}; known {sorted(GRIDS)}")
    dt, k = GRIDS[grid]
    hz = [int(h) for h in horizons]
    need = [int(round(10 * dt * j)) for j in range(1, k + 1)]
    missing = [h for h in need if h not in hz]
    if missing:
        raise SystemExit(f"[refcv3_arm] --grid {grid} needs model slots at {need} "
                         f"(0.1 s steps) but the model emits {hz}; missing "
                         f"{missing}. A grid the model cannot serve by "
                         f"index-select is refused (no interpolation).")
    return {"name": grid, "dt_s": dt, "k": k, "horizons_steps": need,
            "slots": [hz.index(h) for h in need],
            "instants_s": [round(h * DT_FRAME, 3) for h in need],
            "dropped_model_slots": [h for h in hz if h not in need],
            "n_frames": need[-1]}


def hold_controls(v, kap, t0: int):
    """⭐ THE ACTION THAT **CLOSES** AT t0 — ``(a, channel-1)`` from ``v[t0]``,
    ``v[t0-1]`` and ``kap[t0-1]``.

    Every frame index is ``<= t0``, so NOTHING recorded after the window origin
    enters the hold-action arm. (The action *opening* at t0 would need
    ``v[t0+1]`` — a future value — which is exactly why it is not the one held.)
    ``kap`` is ``episode.actions[:, 0]``, which ``physicalai.py:621`` wrote as a
    road-wheel STEER ANGLE; the unit travels with the integration call, not with
    this function."""
    if t0 < 1:
        raise ValueError("hold-action needs t0 >= 1 (one closed step before t0)")
    import torch
    a = (v[t0] - v[t0 - 1]) / DT_FRAME
    return torch.stack([a.to(torch.float32), kap[t0 - 1].to(torch.float32)])


def hold_v0_controls(n: int):
    """⭐ THE STRONGEST TRIVIAL BASELINE: ``a = 0, kappa = 0`` for ``n`` ticks.

    Integrated from the measured ``v0`` this is a CONSTANT-VELOCITY STRAIGHT LINE.
    It consumes strictly LESS than ``hold_controls`` — not even the last observed
    action, only the ``v0`` the PI ruling of 2026-09-02 admits — and it is EXACTLY
    ZERO in either action unit, which is what makes it the one arm that is
    bit-comparable between refav1 and refcv3."""
    import torch
    return torch.zeros(int(n), 2, dtype=torch.float32)


def integrate_select(controls, v0: float, grid: dict, *, action_units: str):
    """``[n, 2]`` controls at 0.1 s -> ``[1, K, 2]`` on the dump grid.

    Integrated at the corpus's NATIVE tick through the programme's ONE unicycle
    (``refav1_arm.paths_from_controls`` -> ``refa_v1_plan.unicycle_paths``, which
    applies ``kappa = tan(steer)/L_enc`` when ``action_units == "steer"``), then
    INDEX-SELECTED onto the grid instants — so the controls and the model share
    one grid and one GT array, and no path is ever resampled."""
    import torch
    n = int(controls.shape[0])
    path = ra.paths_from_controls(controls, v0, DT_FRAME, n,
                                  action_units=action_units)      # [1, n, 2]
    idx = torch.tensor([h - 1 for h in grid["horizons_steps"]],
                       device=path.device)
    return path[:, idx].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# the eval windows — the TRAINER's dataset, future-frame decode removed        #
# --------------------------------------------------------------------------- #
def make_eval_dataset_class():
    """``refc_v3_train.V3Dataset`` (nav / v7.2 labels / route v2.1 / goals / the
    extended future poses — the trainer's EXACT window contract) with ONE change:
    the FUTURE FRAMES are never decoded, because only the loss's LAW target reads
    them and this adapter computes no LAW. Decoding them would be the dominant
    cost of the roll for a tensor nothing consumes.
    *(Harvested from the rescued draft.)*"""
    tr = trainer()
    import torch

    class EvalV3Windows(tr.V3Dataset):
        u8_frames = True

        def _window_u8(self, i: int) -> dict:
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {
                "frames": ep.frames[t:t + w],
                "actions": ep.actions[t:t + w],
                "future_frames": torch.zeros(0, dtype=torch.uint8),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }

    return EvalV3Windows


def build_corpus(a, cfg, prov: dict):
    """The eval episodes (v2 providers), the trainer's window dataset with the
    v7.2 label join and the nav source, and the join report.
    *(Harvested from the rescued draft; the manifest-derived ``n_stack`` offset
    and the geometry/channel refusals are kept verbatim because they are the
    trainer's own rules.)*"""
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import (build_v2_providers,
                                         load_or_build_manifest,
                                         stable_episode_id)
    man = load_or_build_manifest(a.episodes, verbose=False)
    eps = build_v2_providers([a.episodes], lru_size=a.lru, verbose=False)
    files = list(man["files"])
    clip_ids = [str(c) for c in man["clip_id"]]
    n_stack = [int(x) for x in man["n_stack"]]
    if len(eps) != len(files):
        raise SystemExit(f"[refcv3_arm] {len(eps)} providers for {len(files)} clip "
                         f"files under {a.episodes} — manifest drift")
    keep = list(range(len(eps)))
    if a.episodes_n:
        keep = keep[:int(a.episodes_n)]
    eps = [eps[i] for i in keep]
    files = [files[i] for i in keep]
    clip_ids = [clip_ids[i] for i in keep]
    n_stack = [n_stack[i] for i in keep]
    if not eps:
        raise SystemExit(f"[refcv3_arm] no *.v2ep.pt under {a.episodes}")
    eh, ew = cfg.core.encoder.image_hw()
    fh, fw = int(eps[0].frames.shape[-2]), int(eps[0].frames.shape[-1])
    ch = int(eps[0].frames.shape[1])
    if (fh, fw) != (eh, ew):
        raise SystemExit(f"[refcv3_arm] ⛔ geometry mismatch: encoder built for "
                         f"{eh}x{ew}, corpus emits {fh}x{fw}")
    if ch != cfg.core.encoder.in_channels:
        raise SystemExit(f"[refcv3_arm] ⛔ channel mismatch: encoder expects "
                         f"{cfg.core.encoder.in_channels}, corpus emits {ch}")
    if len(set(n_stack)) != 1:
        raise SystemExit(f"[refcv3_arm] the cache mixes n_stack {sorted(set(n_stack))} "
                         f"— the provider->RAW frame offset would differ per clip; "
                         f"refusing (the lead-block join is keyed on RAW frames)")
    # ⭐ PROVIDER -> RAW FRAME. ``v2_dataset.py:36-38``: the provider stores
    # ``poses[n_stack-1:]``, so provider row j IS raw frame j + (n_stack - 1).
    # Read from the manifest, never derived from the channel count.
    raw_offset = n_stack[0] - 1
    Ds = make_eval_dataset_class()
    ds = Ds(eps, window=cfg.core.window, max_horizon=20,
            channels=cfg.core.encoder.in_channels)
    ds.u8_frames = True
    if not a.labels:
        raise SystemExit("[refcv3_arm] --labels (the v7.2 EVAL blob) is required: "
                         "the tactical heads and the nav token are defined by it")
    labels, lman = v7l.load_v7_labels(a.labels, allow_oracle_nav=True)
    by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    ds.v7_by_sid = by_sid
    ds.v7_dt = DT_FRAME
    hit = sum(1 for e in eps if int(e.episode_id) in by_sid)
    if hit == 0:
        raise SystemExit(f"[refcv3_arm] ⛔ {a.labels} joined ZERO of {len(eps)} "
                         f"episodes — wrong blob for this corpus (md5={lman.md5})")
    join = {"labels": {"path": a.labels, "md5": lman.md5,
                       "n_records": lman.n_records, "n_episodes": len(eps),
                       "n_joined": hit, "n_missing": len(eps) - hit,
                       "join_key": "stable_episode_id(clip_id) (the trainer's key)",
                       **{k: v for k, v in lman.to_dict().items()
                          if k in ("schema_version", "vocab", "allow_oracle_nav",
                                   "divergences")}},
            "frames": {"n_stack": n_stack[0], "provider_to_raw_frame_offset": raw_offset,
                       "rule": "v2_dataset.py:36-38 — the provider stores "
                               "poses[n_stack-1:], so provider row j is RAW frame "
                               "j + (n_stack-1). The lead block is keyed on RAW "
                               "frames and this is the ONLY place the two meet."}}
    md5_train_eval = prov.get("eval_labels_md5_at_train")
    if md5_train_eval and md5_train_eval != lman.md5:
        _p(f"[labels] ⚠️ WARNING: this blob md5={lman.md5} != the eval blob the run "
           f"used in training ({md5_train_eval}) — a DIFFERENT eval set; both are "
           f"recorded in the manifest")
    # ---- nav source -------------------------------------------------------- #
    src = a.nav_source
    if src == "auto":
        src = "v72" if prov.get("nav_from_v7_trained") else "none"
    nav_stats = None
    if src == "v72":
        nav_stats = ds.enable_nav_from_v7(lman)        # the TRAINER's own path
        if not prov.get("nav_from_v7_trained"):
            _p("[nav] ⚠️ WARNING: the run trained on the v1 nav derivation but is "
               "being fed v7.2 tokens (--nav-source v72) — an input-distribution "
               "shift; recorded in the manifest")
    join["nav"] = {
        "source": src, "stats": nav_stats,
        "trained_on": prov.get("nav_cmd_derivation_trained"),
        "rule": ("v72: the clip's v7.2 nav_command token -> refb.NAV_COMMANDS "
                 "index (V3Dataset.enable_nav_from_v7, refc_v3_train.py:246-306, "
                 "position-pinned); a clip without a record feeds index 0 + "
                 "nav_valid=False. none: nav_cmd=0 on every window (the published "
                 "REF-C convention) and NO shuffle arm — the v1 derivation reads "
                 "15-25 s of FUTURE poses and is not an inference input.")}
    return eps, files, clip_ids, ds, lman, join, src, raw_offset


# --------------------------------------------------------------------------- #
# the roll — writes the dump                                                    #
# --------------------------------------------------------------------------- #
_ROUTE_TO_NAV = {0: 1, 1: 0, 2: 2}   # asserted == refb_labels._ROUTE_TO_NAV at run_dump entry


def _nav_to_route_map():
    """NAV_COMMANDS index -> ROUTE_CLASSES index, IMPORTED from the label module
    (``stack/scripts/refb_labels.py:77-78`` ``_NAV_TO_ROUTE``), never hand-typed.
    |STOP| The two spaces are DIFFERENT WIDTHS (4 vs 3) and are NOT aligned; the
    sibling harness ``refav1_arm.py:2308`` has always mapped before comparing and
    this file did not."""
    import refb_labels
    return dict(refb_labels._NAV_TO_ROUTE)


def _route_to_nav_map():
    """The ONLY definition, imported -- never re-derived. See ARM_TIERS above."""
    import refb_labels
    return dict(refb_labels._ROUTE_TO_NAV)


def _head_argmax(out, key, row):
    v = out.get(key)
    return -1 if v is None else int(v[row].argmax(-1))


def run_dump(a) -> dict:
    """Roll every arm over the eval grid and write the dump. Returns the manifest."""
    import torch
    import refb_labels

    t_start = time.time()
    if sorted(_ROUTE_TO_NAV.items()) != sorted(refb_labels._ROUTE_TO_NAV.items()):
        raise SystemExit(
            "[refcv3_arm] the route->nav map bound at import time is not the "
            "one refb_labels defines now -- refusing rather than feeding a "
            "mapping the label module does not agree with.")
    tr = trainer()
    from tanitad.refs import refc_v3 as v3mod
    dev = a.device
    model, cfg, targs, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)
    # ⭐⭐ THE EVAL-TIME ABLATIONS (PREREG_REFCV4B_HIERARCHY_EVAL.md §3), applied
    # to the LOADED model before a single window is rolled, so the whole dump is
    # one regime and the stamp below describes all of it.
    _abl_names = resolve_ablations(a)
    abl_state, abl_rec = apply_ablations(
        model, cfg, _abl_names, steps=int(prov["decoder_steps"]),
        feed_ego=bool(getattr(cfg, "ego_state_inject", False)),
        gstr_bank=getattr(a, "gstr_bank", None),
        gstr_seed=int(getattr(a, "gstr_shuffle_seed", 0) or 0))
    # ⭐ REF-C v4 (E11'): a build with `ego_state_inject` TRAINED with the measured
    # ego block on every row (keep drawn at ego_dropout) and the trainer's own
    # eval feeds it with keep = 1. Calling such a build WITHOUT the block hands
    # the goal path nothing while the core still sees v0 — a regime the model
    # never saw. So the block is fed here exactly as `compute_losses_v3` feeds it
    # (`v3.ego_state_from_batch`, the OBSERVED window only). A v3 build
    # (`ego_state_inject` False) is untouched: ego_state=None as before.
    feed_ego = bool(getattr(cfg, "ego_state_inject", False))
    if feed_ego:
        _p("[ego] ego_state_inject build: feeding the MEASURED ego block "
           "(v0, a_long, yaw_rate, curvature, keep=1) at t0 to every forward — "
           "refc_v3.ego_state_from_batch, the trainer's own entry")
    hier = bool(cfg.hier)
    grid = grid_slots(cfg.core.trajectory.horizons, a.grid)
    eps, files, clip_ids, ds, lman, join, nav_src, raw_off = build_corpus(a, cfg, prov)
    units = a.action_units
    if units not in ("kappa", "steer"):
        raise SystemExit(f"[refcv3_arm] --action-units must be kappa|steer, got {units!r}")
    steps = int(prov["decoder_steps"])
    W = int(cfg.core.window)

    _p(f"[model] {prov['ckpt']} step={prov['step']} arm={prov['arm']} "
       f"anchors={prov['n_anchors']} horizons={prov['horizons']} "
       f"window={W} decoder={prov['decoder_mode']}/{steps} "
       f"config<-{prov['rebuilt_from']}")
    _p(f"[grid] {grid['name']}: dt={grid['dt_s']} s K={grid['k']} instants="
       f"{grid['instants_s']} s <- model slots {grid['horizons_steps']} "
       f"(indices {grid['slots']}); dropped {grid['dropped_model_slots']}")
    _p(f"[units] action_units={units} — the RECORDED channel 1 is a road-wheel "
       f"STEER angle (physicalai.py:621). "
       + ("kappa = tan(steer)/%s is applied before integration; `ha` is NOT "
          "comparable to a kappa-unit refav1 run (`ha0` is, it is zero)."
          % ra.STEER_WHEELBASE_M if units == "steer" else
          "⛔ LEGACY UNCONVERTED READING — `ha` carries the "
          "C-REFCV3-ARM-SAME-DEFECT over-rotation. Use --action-units steer "
          "unless you are deliberately reproducing a legacy number."))

    # ---- window selection -------------------------------------------------- #
    stride = max(1, int(a.window_stride))
    sel = [(wi, e_i, t) for wi, (e_i, t) in enumerate(ds.index) if wi % stride == 0]
    if not sel:
        raise SystemExit("[refcv3_arm] the stride selected zero windows")

    # ---- nav for every selected window, then the shuffle -------------------- #
    nav_on = bool(getattr(ds, "nav_from_v7", False)) or nav_src == "v72"
    nav_true = np.zeros(len(sel), dtype=np.int64)
    nav_valid = np.zeros(len(sel), dtype=bool)
    for i, (wi, e_i, t) in enumerate(sel):
        if nav_on and getattr(ds, "_nav_by_sid", None) is not None:
            nid = ds._nav_by_sid.get(int(ds.episodes[e_i].episode_id))
            nav_true[i] = 0 if nid is None else int(nid)
            nav_valid[i] = nid is not None
    nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)

    # ⭐ `ha0_ext` is NOT optional: refcv5's acceptance bar is "beat BOTH `ha`
    # AND `ha0_ext`", and until Rung A1 this harness computed neither the arm
    # nor the number, so half the bar was unreadable on the REF-C surface.
    arms = ["os", "ha", "ha0", "ha0_ext"]
    if nav_on and not a.no_navshuf:
        arms.append("os_navshuf")
    # ⭐ THE NAV-ZERO TRAJECTORY ARM (BACKLOG R39). Shuffle and zero are NOT
    # interchangeable (NAV_NULL), and the zero arm is the DEPLOYMENT-relevant
    # one, because nav here is an ORACLE input (provenance ego-future).
    if nav_on and not a.no_navzero:
        arms.append("os_navzero")
    if a.with_oracle_sel:
        arms.append("oracle_sel")
    if nav_on and a.no_navshuf:
        _p("[nav] ⚠️ --no-navshuf: the record is NOT admissible for any "
           "nav-conditioned (STRATEGIC) claim")
    if nav_on and a.no_navzero:
        _p("[nav] ⚠️ --no-navzero: BACKLOG R39 binds every nav-conditioned claim "
           "to carry the nav-ZERO arm BESIDE the nav-shuffle one; this record "
           "carries only the shuffle")
    if not nav_on:
        _p("[nav] nav source 'none': the true-nav forward ALREADY passes "
           "nav_cmd=None, so `os` IS the nav-zero arm — os_navshuf and "
           "os_navzero would be bit-identical to it and are NOT emitted; the "
           "STRATEGIC family will be REFUSED with that reason")
    # ⛔ nav_cmd=None IS A WHOLE-CALL PROPERTY, NOT A PER-ROW VALUE: it switches
    # off the E13 injection for the entire forward (refc_v3.py:437-441). It
    # therefore CANNOT ride in the batched token tensor and gets its own call.
    # ⚠️ THIS FIXES A DEFECT IN THE FIRST VERSION OF THIS TOOL: `--with-navzero`
    # put index 0 into the batched tensor, which is nav-CONSTANT (E13 ON, fed the
    # `follow` embedding) — NOT nav-zero. Two different interventions, and the
    # first one silently answered a question nobody asked.
    conds_fed = ["nav_true"] + (["nav_shuffled"] if "os_navshuf" in arms else [])
    # ⭐ --with-navflip (2026-09-05, harvested from the predecessor's draft): the
    # SHARPEST nav intervention — left <-> right, so the fed token disagrees
    # with the truth on EVERY commanded window (the shuffle feeds `follow` on
    # most of them). OPTIONAL and default OFF: the two REQUIRED controls stay
    # shuffle + zero (BACKLOG R39) and a banked dump's arm list is unchanged.
    nav_flip = None
    if nav_on and getattr(a, "with_navflip", False):
        nav_flip = nav_true.copy()
        nav_flip[nav_true == refb_labels.NAV_LEFT] = refb_labels.NAV_RIGHT
        nav_flip[nav_true == refb_labels.NAV_RIGHT] = refb_labels.NAV_LEFT
        conds_fed.append("nav_flipped")
        arms.append("os_navflip")
        _p(f"[nav-flip] os_navflip: left<->right on {int((nav_flip != nav_true).sum())}"
           f"/{len(nav_true)} windows — the sharpest intervention (optional arm)")
    do_navzero = ("os_navzero" in arms) or (a.with_navzero and nav_on)
    # -- STAR --with-navpred: the arm this whole run exists for ---------------
    # The token is the model's own route prediction, taken from the nav-null
    # forward so that NO oracle nav reaches the arm. The nav-null call therefore
    # has to run BEFORE the batched one (it SUPPLIES a row of it), which is the
    # only ordering change in the loop -- and it is made only when the flag is on,
    # so the FULL path stays bit-identical.
    navpred_on = bool(nav_on and getattr(a, "with_navpred", False))
    if getattr(a, "with_navpred", False) and not nav_on:
        raise SystemExit("[refcv3_arm] --with-navpred needs a nav source: with "
                         "nav_source 'none' the forward already passes "
                         "nav_cmd=None and there is no token to replace.")
    if navpred_on and not do_navzero:
        raise SystemExit("[refcv3_arm] --with-navpred REQUIRES the nav-zero "
                         "forward -- it is where the predicted route is read "
                         "from. Drop --no-navzero.")
    if navpred_on:
        conds_fed.append("nav_predicted")
        arms.append("os_navpred")
        _p("[nav-pred] os_navpred: nav_cmd = refb_labels._ROUTE_TO_NAV["
           "argmax(route_logits from the nav_cmd=None forward)] -- the model's "
           "OWN route, no oracle token anywhere in the arm. Map (SOURCE, "
           f"refb_labels.py:483-484): {dict(sorted(_ROUTE_TO_NAV.items()))} "
           "-- NOT the identity.")
    conds = conds_fed + (["nav_zero"] if do_navzero else [])

    _p(f"[roll] {len(sel)} windows / {len(eps)} episodes · arms={arms} · "
       f"fed conditionings={conds_fed}"
       + (" + a SEPARATE nav_cmd=None forward (nav_zero)" if do_navzero else "")
       + f" · nav_shuffle changed {shuf_stats['n_changed']}"
         f"/{shuf_stats['n_windows']}")
    if do_navzero:
        _p("[nav-null] os_navzero = model(..., nav_cmd=None): the E13 tactical + "
           "strategic injection is SKIPPED (refc_v3.py:437-441; nav_injected "
           "reads False), and the CORE collapses to one_hot(0)='follow' "
           "(refc.py:2021-2024). ⚠️ At the core that is a COLLAPSE ONTO THE "
           "MAJORITY TOKEN, not a removal — NAV_COMMANDS has no 'unknown' and "
           "nav_known_channel is False (refc.py:594) — so this arm is a LOWER "
           "BOUND on nav dependence. ⛔ nav_known is NOT usable as the null: "
           "refc.py:2042-2045 raises when the gate is off.")
    # ⛔ A DUMP DIRECTORY IS ONE REGIME. Writing an ablated roll on top of a
    # FULL one leaves ep*.npz from both and a manifest describing only the
    # second — an unreadable mixture that looks like a complete dump. Refuse.
    _prev_man = os.path.join(a.dump_dir, "manifest.json")
    if os.path.exists(_prev_man):
        try:
            with open(_prev_man, encoding="utf-8") as _fh:
                _prev = ((json.load(_fh).get("ablation") or {}).get("applied"))
        except (OSError, ValueError):
            _prev = "__unreadable__"
        if _prev != "__unreadable__" and list(_prev or []) != list(_abl_names):
            raise SystemExit(
                f"[refcv3_arm] ⛔ {a.dump_dir} already holds a dump rolled "
                f"under ablation {_prev!r}, and this run is {_abl_names!r}. "
                f"Two regimes in one directory is an unreadable mixture that "
                f"looks like a complete dump. Use a separate --dump-dir per "
                f"ablation arm.")
    os.makedirs(a.dump_dir, exist_ok=True)
    os.makedirs(os.path.join(a.dump_dir, "decisions"), exist_ok=True)
    with open(os.path.join(a.dump_dir, "ABLATION.txt"), "w",
              encoding="utf-8") as _fh:
        _fh.write((", ".join(_abl_names) if _abl_names else "FULL") + "\n")

    by_ep: dict[int, list] = {}
    for i, (wi, e_i, t) in enumerate(sel):
        by_ep.setdefault(e_i, []).append((i, wi, t))

    horizons = list(cfg.core.trajectory.horizons)
    slots = grid["slots"]
    # .cpu(): `traj_tgt` (:994) and `sv` (:980) are CPU dataset tensors; the bank
    # is used ONLY in the a_star argmin below (:1051/:1059), so keeping it on CPU
    # matches the CPU path the fixture validated and fixes a CUDA/CPU device clash.
    anchors_bank = model.core.decoder.anchors.detach().cpu()    # [N, S, 2]
    episodes_manifest, n_done, n_skipped = [], 0, 0
    t_fwd_first = None
    skip_reasons: dict[str, int] = {}

    for fi, e_i in enumerate(sorted(by_ep)):
        ep = ds.episodes[e_i]
        v_ep = ep.poses[:, 3].float()
        kap_ep = ep.actions[:, 0].float()
        acc = {k: [] for k in ["g", "v0"] + arms}
        dec: dict[str, list] = {}
        ws = []
        for (i, wi, t) in by_ep[e_i]:
            t0 = t + W - 1                       # PROVIDER index of the origin
            item = ds[wi]
            fv = item["future_valid_ext"]
            if not bool(fv[[h - 1 for h in grid["horizons_steps"]]].all()):
                n_skipped += 1
                skip_reasons["horizon_beyond_episode"] = \
                    skip_reasons.get("horizon_beyond_episode", 0) + 1
                continue
            if t0 < 1:
                n_skipped += 1
                skip_reasons["no_closed_action_before_t0"] = \
                    skip_reasons.get("no_closed_action_before_t0", 0) + 1
                continue
            pose_last = item["pose_last"].float()
            v0 = float(pose_last[3])
            # -- GT: the TRAINER's own target function, then index-select ------
            traj_tgt = refb_labels.waypoint_targets(
                pose_last[None], item["future_poses_ext"].float()[None],
                horizons)                                        # [1, S, 2]
            g = traj_tgt[:, slots].float().cpu().numpy()          # [1, K, 2]
            # -- the two model-free controls -----------------------------------
            hold = hold_controls(v_ep, kap_ep, t0)
            n_f = grid["n_frames"]
            ha = integrate_select(hold[None].expand(n_f, 2), v0, grid,
                                  action_units=units)
            # ⭐ ha0: SAME integrator, SAME v0, zero controls — so any difference
            # from `ha` is the held action alone, and the arm is exactly zero in
            # either unit (hence bit-comparable across models).
            ha0 = integrate_select(hold_v0_controls(n_f), v0, grid,
                                   action_units=units)
            # ⭐⭐ ha0_ext: THE ECHO CONTROL, and the other half of the bar.
            # ⛔ The (a0, k0) derivation is NOT written here. It is
            # `refav1_arm.hold_ext_controls` — the SAME call the refav1 harness
            # makes — invoked at this corpus's native 0.1 s tick (stride=1);
            # `loader=None` because that argument is read only for its `dt`,
            # which is passed explicitly. A second implementation of a shared
            # kinematic is how two "independent" checks come to agree on a
            # wrong answer (echo_gate.ha0_ext's own docstring).
            ext = ra.hold_ext_controls(None, v_ep, kap_ep, t0,
                                       dt=DT_FRAME, stride=1)
            ha0_ext = integrate_select(ext[None].expand(n_f, 2), v0, grid,
                                       action_units=units)
            # -- the model, one batched forward over the nav conditionings -----
            fr = tr.frames_to_device(item["frames"][None], dev)   # [1, W, C, H, W]
            if abl_state.frames_blind:
                # ⭐⭐ THE DELIBERATE REGRESSION: a constant image. The arm is an
                # echo BY CONSTRUCTION, and if the echo gate PASSES it the panel
                # is VOID (PREREG_REFC_V4 §7 OUTCOME IV).
                fr = torch.full_like(fr, float(fr.mean()))
            b = len(conds_fed)
            fr_b = fr.expand(b, *fr.shape[1:]).contiguous() if b > 1 else fr
            nav_vals = {"nav_true": int(nav_true[i]),
                        "nav_shuffled": int(nav_shuf[i])}
            if nav_flip is not None:
                nav_vals["nav_flipped"] = int(nav_flip[i])
            # ``.get(c, 0)`` is a no-op for every pre-existing conditioning
            # (all are present); it exists so `nav_predicted`, whose value is
            # only known AFTER the nav-null forward, has a placeholder here and
            # is overwritten below before the batched call runs.
            nav_t = torch.tensor([nav_vals.get(c, 0) for c in conds_fed],
                                 dtype=torch.long, device=dev)
            v0_t = torch.full((b,), v0, dtype=torch.float32, device=dev)
            # ⭐ the g_str SHUFFLE's per-window channel: the banked goal of the
            # window this one is permuted onto. Set BEFORE the forward, read by
            # the hook on str_goal_head, so the loop and the hook cannot drift.
            if abl_state.gstr_mode == "shuffle":
                _key = (str(clip_ids[e_i]), int(t0))
                _src = abl_state.gstr_perm.get(_key)
                if _src is None:
                    raise SystemExit(
                        f"[refcv3_arm] gstr_shuffle: window {_key} is not in "
                        f"the bank. The bank must be a FULL dump over the SAME "
                        f"episodes AND the same window grid (--window-stride "
                        f"and --grid), or the permutation is meaningless.")
                abl_state.gstr_target = abl_state.gstr_bank[_src]
                abl_state.n_gstr_windows += 1
            ego_b = ego_z = None
            if feed_ego:
                _es = v3mod.ego_state_from_batch(
                    {"pose_last": item["pose_last"].float()[None],
                     "actions": item["actions"].float()[None]}, device=dev)
                if abl_state.ego_zero:
                    # X15: keep = 0 means WITHHELD, and the model re-applies the
                    # flag to the values, so zeroing the bit is the whole
                    # intervention on this side.
                    _es = _es.clone()
                    _es[:, 4] = 0.0
                ego_b = _es.expand(b, -1).contiguous() if b > 1 else _es
                ego_z = _es
            # ⛔ v0 WITHHELD AT THE CORE is the other half of EGO-ZERO: refc.py
            # derives `keep` from `v0 is not None`, so a pre-zeroed v0 would
            # arrive with keep = 1 — the file says so itself. Pass None.
            _v0_fed = None if abl_state.ego_zero else v0_t
            tf = time.time()
            with torch.no_grad():
                out_z = None
                if navpred_on:
                    # STAR the nav-null forward runs FIRST here because it
                    # SUPPLIES the predicted token. Identical call to the one
                    # below; only the position moves, and only under this flag.
                    out_z = model(fr, nav_cmd=None,
                                  v0=None if abl_state.ego_zero else v0_t[:1],
                                  steps=steps, ego_state=ego_z)
                    _rl = out_z.get("route_logits")
                    if _rl is None:
                        raise SystemExit(
                            "[refcv3_arm] --with-navpred: the forward emits no "
                            "`route_logits`, so there is no predicted route to "
                            "feed. Refusing rather than inventing a token.")
                    _rz = int(_rl[0].argmax(-1))
                    if _rz not in _ROUTE_TO_NAV:
                        raise SystemExit(
                            f"[refcv3_arm] --with-navpred: route argmax {_rz} is "
                            f"outside ROUTE_CLASSES -- the head's width and the "
                            f"label module disagree; refusing.")
                    nav_vals["nav_predicted"] = int(_ROUTE_TO_NAV[_rz])
                    nav_t = torch.tensor([nav_vals[c] for c in conds_fed],
                                         dtype=torch.long, device=dev)
                out = model(fr_b, nav_cmd=nav_t if nav_on else None,
                            v0=_v0_fed, steps=steps, ego_state=ego_b)
                # ⭐ THE NAV NULL — ITS OWN CALL. `nav_cmd=None` is a whole-call
                # property (it gates the E13 injection, refc_v3.py:437-441), so
                # it cannot be one row of the batch above. Same frames, same v0,
                # same window, same grid: the ONLY difference is the nav.
                if do_navzero and out_z is None:
                    out_z = model(fr, nav_cmd=None,
                                  v0=None if abl_state.ego_zero else v0_t[:1],
                                  steps=steps, ego_state=ego_z)
            if t_fwd_first is None:
                t_fwd_first = time.time() - tf
                n_calls = b + (1 if do_navzero else 0)
                _p(f"[cost] first forward ({b} fed row(s)"
                   f"{' + 1 nav-null call' if do_navzero else ''}, "
                   f"{n_calls} rows total) took {t_fwd_first:.2f} s -> ESTIMATED "
                   f"{t_fwd_first * len(sel) / 3600:.2f} h for {len(sel)} windows "
                   f"(forward only; IO/analysis extra)")
            # ⭐ DID THE ABLATION ACTUALLY REACH THE FORWARD? Checked against
            # the model's OWN emitted g_str, once, rather than assumed. A hook
            # registered on the wrong module raises nothing and reports nothing.
            if abl_state.gstr_mode is not None and not abl_state.verified:
                _got = out.get("g_str")
                if _got is None:
                    raise SystemExit("[refcv3_arm] the g_str ablation is armed "
                                     "but the forward emits no `g_str` — the "
                                     "hook cannot be verified, so the arm is "
                                     "not admissible")
                _got = _got.float()[0].detach().cpu().numpy().tolist()
                _want = ([1.0, 0.0, 0.0] if abl_state.gstr_mode == "zero"
                         else [float(x) for x in abl_state.gstr_target[:3]])
                _err = max(abs(_got[j] - _want[j]) for j in range(3))
                if _err > 1e-4:
                    raise SystemExit(
                        f"[refcv3_arm] the g_str ablation did NOT take: the "
                        f"model emitted {_got} where {_want} was injected "
                        f"(max abs err {_err:.3g}). Refusing to roll an arm "
                        f"whose intervention cannot be demonstrated.")
                abl_state.verified["g_str"] = {"emitted": _got,
                                               "injected": _want,
                                               "max_abs_err": float(_err)}
                _p(f"  [ablation] g_str injection VERIFIED on the first window: "
                   f"emitted {[round(x, 4) for x in _got]} (err {_err:.2g})")
            # ⭐ THE DEPLOYED SELECTION IS out["traj"] AND NOTHING ELSE
            # (refc_v3.py:520-525 on hier; refc.py:1531-1534 on flat).
            traj = out["traj"].float()                            # [b, S, 2]
            acc["os"].append(traj[0:1, slots].cpu().numpy())
            if "os_navshuf" in arms:
                r = conds_fed.index("nav_shuffled")
                acc["os_navshuf"].append(traj[r:r + 1, slots].cpu().numpy())
            if "os_navzero" in arms:
                acc["os_navzero"].append(
                    out_z["traj"].float()[0:1, slots].cpu().numpy())
            if "os_navflip" in arms:
                r = conds_fed.index("nav_flipped")
                acc["os_navflip"].append(traj[r:r + 1, slots].cpu().numpy())
            if "os_navpred" in arms:
                r = conds_fed.index("nav_predicted")
                acc["os_navpred"].append(traj[r:r + 1, slots].cpu().numpy())
            # -- the ORACLE ceiling, exactly as the trainer computes it --------
            if "oracle_sel" in arms:
                sv = torch.stack([fv[h - 1] for h in horizons]).to(traj.dtype)
                tgt = traj_tgt.to(anchors_bank.dtype)
                dist = (((tgt[:, None] - anchors_bank[None]) ** 2).sum(-1)
                        * sv[None, None]).sum(-1)                 # [1, N]
                a_star = int(dist.argmin(dim=1)[0])
                acc["oracle_sel"].append(
                    out["anchor_traj"][0:1, a_star][:, slots].float().cpu().numpy())
            else:
                sv = torch.stack([fv[h - 1] for h in horizons]).to(traj.dtype)
                tgt = traj_tgt.to(anchors_bank.dtype)
                dist = (((tgt[:, None] - anchors_bank[None]) ** 2).sum(-1)
                        * sv[None, None]).sum(-1)
                a_star = int(dist.argmin(dim=1)[0])
            acc["g"].append(g)
            acc["ha"].append(ha)
            acc["ha0"].append(ha0)
            acc["ha0_ext"].append(ha0_ext)
            acc["v0"].append(np.array([v0], dtype=np.float32))
            # -- the sidecar ---------------------------------------------------
            sel_idx = int(out["sel_idx"][0])
            rank_key = "sel_score_v3" if "sel_score_v3" in out else "sel_score"
            dec.setdefault("sel_idx", []).append(sel_idx)
            dec.setdefault("sel_idx_base", []).append(
                int(out["sel_idx_base"][0]) if "sel_idx_base" in out else -1)
            dec.setdefault("sel_score_max", []).append(
                float(out[rank_key][0].max()))
            dec.setdefault("a_star", []).append(a_star)
            dec.setdefault("sel_agrees_oracle", []).append(
                float(sel_idx == a_star))
            dec.setdefault("anchor_acc", []).append(
                float(int(out["anchor_logits"][0].argmax(-1)) == a_star))
            dec.setdefault("goal_gate", []).append(
                float(out["goal_gate_value"]) if "goal_gate_value" in out
                else float("nan"))
            dec.setdefault("goal_score_absmean", []).append(
                float(out["goal_score_absmean"]) if "goal_score_absmean" in out
                else float("nan"))
            dec.setdefault("goal_dist_sel", []).append(
                float(out["goal_dist"][0, sel_idx]) if "goal_dist" in out
                else float("nan"))
            lat_key = "lat_logits_tac" if hier else "lat_decision"
            lon_key = "lon_logits_tac" if hier else "lon_decision"
            for ci_, cname in enumerate(conds_fed):
                dec.setdefault(f"lat_pred_{cname}", []).append(
                    _head_argmax(out, lat_key, ci_))
                dec.setdefault(f"lon_pred_{cname}", []).append(
                    _head_argmax(out, lon_key, ci_))
                dec.setdefault(f"route_pred_{cname}", []).append(
                    _head_argmax(out, "route_logits", ci_))
            if do_navzero:
                dec.setdefault("lat_pred_nav_zero", []).append(
                    _head_argmax(out_z, lat_key, 0))
                dec.setdefault("lon_pred_nav_zero", []).append(
                    _head_argmax(out_z, lon_key, 0))
                dec.setdefault("route_pred_nav_zero", []).append(
                    _head_argmax(out_z, "route_logits", 0))
            for cname in ("nav_true", "nav_shuffled", "nav_zero"):
                if cname in conds:
                    continue
                for hk in ("lat", "lon", "route"):
                    dec.setdefault(f"{hk}_pred_{cname}", []).append(-1)
            # ⭐ the E13 edge, per window: LIVE under the fed nav, DEAD under the
            # null. An advertised-but-inert conditioning edge is a real defect
            # class; this is the measurement, not the declaration.
            dec.setdefault("nav_injected_true", []).append(
                float(bool(out.get("nav_injected", False))))
            dec.setdefault("nav_injected_zero", []).append(
                float(bool(out_z.get("nav_injected", False))) if do_navzero
                else float("nan"))
            lat_v7 = int(item["lat_v7"]) if "lat_v7" in item else -100
            lon_v7 = int(item["lon_v7"]) if "lon_v7" in item else -100
            from tanitad.data import v7_labels as _v7l
            dec.setdefault("lat_label", []).append(
                -100 if lat_v7 == _v7l.IGNORE_ID else lat_v7)
            dec.setdefault("lon_label", []).append(
                -100 if lon_v7 == _v7l.IGNORE_ID else lon_v7)
            rv = bool(item["route_valid"])
            dec.setdefault("route_label", []).append(
                int(item["route_target"]) if rv else -100)
            dec.setdefault("nav_cmd", []).append(int(nav_true[i]))
            dec.setdefault("nav_cmd_shuf", []).append(int(nav_shuf[i]))
            if nav_flip is not None:
                dec.setdefault("nav_cmd_flip", []).append(int(nav_flip[i]))
            if navpred_on:
                # the fed token AND the raw route argmax it came from, so the
                # mapping can be re-derived from the dump without the model.
                dec.setdefault("nav_cmd_pred", []).append(
                    int(nav_vals["nav_predicted"]))
                dec.setdefault("route_argmax_navzero_fed", []).append(int(_rz))
            dec.setdefault("nav_valid", []).append(bool(nav_valid[i]))
            dec.setdefault("ha_controls", []).append(
                hold[None].expand(n_f, 2).float().cpu().numpy()[None])
            # ⭐ NAV-COMPLIANCE sidecar (taniteval.nav_compliance): the
            # BEHAVIOURAL readouts per conditioning — the emitted plan at every
            # model slot, the strategic goal, the SELECTED vocabulary element
            # (the selection surface, decided at t=0) and the refined fan's
            # terminal headings with the reach mask. Additive keys only.

            def _navcomp_row(o, r, cname):
                dec.setdefault(f"plan_full_{cname}", []).append(
                    o["traj"].float()[r:r + 1].cpu().numpy())             # [1, S, 2]
                if "g_str" in o:
                    dec.setdefault(f"gstr_{cname}", []).append(
                        o["g_str"].float()[r:r + 1].cpu().numpy())        # [1, 3]
                si = int(o["sel_idx"][r])
                dec.setdefault(f"sel_idx_{cname}", []).append(si)
                bank = o["anchor_bank"].float()[r]                          # [N, S, 2]
                dec.setdefault(f"sel_bank_{cname}", []).append(
                    bank[si:si + 1].cpu().numpy())                          # [1, S, 2]
                fan_ = o["anchor_traj"].float()[r]                          # [N, S, 2]
                d_ = fan_[:, -1] - fan_[:, -2]
                th = torch.atan2(d_[:, 1], d_[:, 0])
                th = torch.where(torch.hypot(d_[:, 0], d_[:, 1]) < 0.05,
                                 torch.zeros_like(th), th)
                dec.setdefault(f"fan_term_heading_{cname}", []).append(
                    th[None].cpu().numpy())                                 # [1, N]
                rk = o.get("reach_keep")
                dec.setdefault(f"reach_keep_{cname}", []).append(
                    (rk[r:r + 1].float() if rk is not None
                     else torch.ones(1, fan_.shape[0])).cpu().numpy())     # [1, N]

            for ci_, cname in enumerate(conds_fed):
                _navcomp_row(out, ci_, cname)
            if do_navzero:
                _navcomp_row(out_z, 0, "nav_zero")
            dec.setdefault("gt_future_ext", []).append(
                item["future_poses_ext"].float()[None].cpu().numpy())       # [1, 60, 4]
            dec.setdefault("gt_future_valid_ext", []).append(
                fv.float()[None].cpu().numpy())                             # [1, 60]
            dec.setdefault("pose_last", []).append(
                pose_last[None].cpu().numpy())                              # [1, 4]
            dec.setdefault("ego_t0", []).append(
                v3mod.ego_state_at_t0(
                    pose_last[None, None].expand(1, item["actions"].shape[0], 4),
                    item["actions"].float()[None])[:, :4].cpu().numpy())    # [1, 4]
            ws.append(int(t0))
            n_done += 1
        if not ws:
            _p(f"  [{fi + 1}/{len(by_ep)}] {clip_ids[e_i][:12]} — 0 scoreable "
               f"windows, episode SKIPPED")
            continue
        np.savez_compressed(
            os.path.join(a.dump_dir, f"ep{len(episodes_manifest):03d}.npz"),
            **{k: np.concatenate(v).astype(np.float32) for k, v in acc.items()},
            ws=np.array(ws), eid=np.array([len(episodes_manifest)]),
            clip_index=np.array([e_i]))
        dec_np = {}
        for k, v in dec.items():
            if isinstance(v[0], np.ndarray):
                dec_np[k] = np.concatenate(v).astype(np.float32)
            elif isinstance(v[0], bool):
                dec_np[k] = np.array(v, dtype=bool)
            elif isinstance(v[0], float):
                dec_np[k] = np.array(v, dtype=np.float32)
            else:
                dec_np[k] = np.array(v, dtype=np.int64)
        np.savez_compressed(
            os.path.join(a.dump_dir, "decisions",
                         f"ep{len(episodes_manifest):03d}.npz"),
            ws=np.array(ws), ep_poses=ep.poses.float().cpu().numpy(), **dec_np)
        episodes_manifest.append(
            {"file_index": len(episodes_manifest), "episode_index": e_i,
             "clip_id": clip_ids[e_i],
             "episode_id": int(ds.episodes[e_i].episode_id),
             "n_windows": len(ws)})
        _p(f"  [{fi + 1}/{len(by_ep)}] {clip_ids[e_i][:12]} {len(ws)} windows "
           f"{time.time() - t_start:.0f}s")

    if not episodes_manifest:
        raise SystemExit("[refcv3_arm] every window was skipped — nothing dumped "
                         f"(reasons: {skip_reasons})")
    manifest = {
        "tool": "taniteval/tools/refcv3_arm.py",
        "doc": "taniteval/tools/REFCV3_ARM.md",
        "model": prov,
        "t1_definition": {
            "arm": "os",
            "_is": ("ONE forward pass of RefCV3Model at the window origin, "
                    "consuming the observed frames, the clip's v7.2 nav token and "
                    "the MEASURED v0 at t0 and nothing else, emitting the whole "
                    "6 s path as the model's OWN sel_score_v3-ranked choice among "
                    "its 128 anchors."),
            "no_closed_loop_because": (
                "refc_v3.py:480 — forward(frames, nav_cmd, v0, steps, lan, "
                "nav_known) has NO action argument; there is no per-step decode "
                "and no state that advances, so t1_eval.roll_closed "
                "(t1_eval.py:760) has no action to feed back and CANNOT be "
                "ported. The flagship is supervised AND autoregressive; that is "
                "the difference."),
            "selection": ("out['traj'] — refc_v3.py:520-525: rank = "
                          "sel_score_v3 (= apply_seam_clamp(sel_score, "
                          "goal_gate*score), banked at :527) masked by "
                          "reach_keep, then argmax — on the hier arm; the core's "
                          "own sel_score at refc.py:1531-1534 on the flat arm. "
                          "⛔ NEVER a_star (refc_v3_train.py:460), which is the "
                          "GT-nearest anchor and makes eval_traj an "
                          "ORACLE-SELECTED lower bound."),
            "inputs_admitted": ["frames <= t0", "nav_cmd (v7.2 token)",
                                "v0 = pose_last[:, 3] measured at t0"],
            "ade_note": ("ADE here is recomputed as an L2 norm from the dumped "
                         "path. refcv3's training-time `traj` is a mean L1 PER "
                         "COORDINATE (refc_v3_train.py:464-465) — ⛔ a different "
                         "statistic; never convert one into the other."),
        },
        "tier_ruling": TIER_RULING,
        "grid": {**grid, "n_windows": n_done, "n_windows_skipped": n_skipped,
                 "skip_reasons": skip_reasons,
                 "n_episodes": len(episodes_manifest),
                 "n_episodes_available": len(files),
                 "window_stride": stride, "obs_window": W,
                 "ws_is": "the PROVIDER frame index of the window origin t0 "
                          "(= t + window - 1); RAW frame = ws + "
                          f"{raw_off} (see corpus.frames)"},
        "arms": arms, "tiers": {x: ARM_TIERS[x] for x in arms},
        "arm_meaning": {x: ARM_MEANING[x] for x in arms},
        "absent_arms": ABSENT_ARMS,
        "action_units": {
            "recorded": units, "L_enc_m": ra.STEER_WHEELBASE_M,
            # `ha0_ext` holds a RECORDED channel-1 value too, so the unit
            # conversion applies to it exactly as it does to `ha`.
            "applies_to": ["ha", "ha0_ext"],
            "does_not_apply_to": ["os", "os_navshuf", "oracle_sel (the model "
                                  "emits a PATH, not a control — there is no "
                                  "channel to convert)",
                                  "ha0 (exactly zero in either unit)"],
            "rule": ("v2ep actions[:,0] is a road-wheel angle "
                     "(physicalai.py:621 steer = arctan(L_enc*curvature), "
                     ":632 the column stack). 'steer' = the repaired contract "
                     "(kappa = tan(steer)/L_enc before any integration); "
                     "'kappa' = the LEGACY unconverted reading, which carries "
                     "the C-REFCV3-ARM-SAME-DEFECT over-rotation. NOT comparable.")},
        "hold_action_rule": hold_controls.__doc__,
        "hold_v0_rule": hold_v0_controls.__doc__,
        # ⭐ the SHARED derivation's own docstring, not a paraphrase — so the
        # record names the one implementation both harnesses call.
        "ha0_ext_rule": ra.hold_ext_controls.__doc__,
        "ha0_ext_call": ("refav1_arm.hold_ext_controls(None, v_ep, kap_ep, t0, "
                         "dt=0.1, stride=1) -> integrate_select — the SAME "
                         "function refav1_arm.py calls at stride=2; pinned "
                         "bit-equal by "
                         "stack/tests/test_refcv3_ha0_ext_shared.py"),
        "nav_predicted": ({
            "arm": "os_navpred",
            "how": ("nav_cmd = refb_labels._ROUTE_TO_NAV[argmax(route_logits)] "
                    "where route_logits comes from the nav_cmd=None forward, so "
                    "no oracle nav enters the arm"),
            "map": {str(k): int(v) for k, v in
                    sorted(_route_to_nav_map().items())},
            "map_source": "stack/scripts/refb_labels.py:483-484 (IMPORTED)",
            "asymmetry": ("the oracle nav token is per-CLIP; the route head "
                          "predicts per-WINDOW, so this arm's token can change "
                          "within a clip"),
        } if navpred_on else None),
        "nav_shuffle": shuf_stats,
        "nav_null": (dict(NAV_NULL, emitted=bool(do_navzero))
                     if do_navzero else
                     {**NAV_NULL, "emitted": False,
                      "reason_absent": ("--no-navzero, or nav source 'none' (in "
                                        "which case `os` already IS the "
                                        "nav-zero arm)")}),
        "head_conditionings": conds,
        "fed_conditionings": conds_fed,
        "sidecar_schema": _SIDECAR_DOC,
        "ego_state_fed": feed_ego,
        # ⭐⭐ THE ABLATION PROVENANCE. A result must never be readable without
        # knowing which ablation produced it, so this rides in the manifest AND
        # is copied onto every per-arm block by analyze_refcv3.
        "ablation": dict(abl_rec, verified=abl_state.verified,
                         n_gstr_windows=abl_state.n_gstr_windows),
        "navcomp_sidecar": True,
        #: ⛔ `**join` carries its OWN "labels" key -- the provenance BLOCK,
        #: with the path under `.path`. A literal `"labels": a.labels` here is
        #: DEAD (the splat comes last and overwrites it) while LOOKING like it
        #: publishes the path, and reading it as one cost the STRATEGIC
        #: nav-compliance family on every arm. The path is now published
        #: unambiguously as `labels_path`; `labels` stays the provenance block.
        "corpus": {"episodes": a.episodes, "labels_path": a.labels,
                   "n_episodes_available": len(files), **join},
        "episodes": episodes_manifest,
        "wallclock_s": round(time.time() - t_start, 1),
        "first_forward_s": t_fwd_first,
        "_unverified": _UNVERIFIED_ON_REAL_CKPT,
    }
    with open(os.path.join(a.dump_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    _p(f"[dump] {n_done} windows over {len(episodes_manifest)} episodes "
       f"({n_skipped} skipped: {skip_reasons or 'none'})")
    _p("REFCV3_DUMP_DONE")
    return manifest


# --------------------------------------------------------------------------- #
# distance-keeping on the COMMON grid of the dump and the banked lead block      #
# *(harvested from the rescued draft, which derived the common-grid rule)*       #
# --------------------------------------------------------------------------- #
def lead_block_common_grid(path: str, dt: float, k: int):
    """Index-select the banked B1 block onto the instants it SHARES with the dump.

    Returns ``(view, idx, meta, info)`` or ``(None, None, meta, refusal)``. The
    block is on a 0.2 s / K=10 grid; refcv3's grid is not a subset of it, so BOTH
    sides are index-selected onto the common instants — no resampling of the lead
    track, no interpolation of the path. The common instants must be uniformly
    spaced starting at their own spacing, or ``lead_metrics.distance_keeping``'s
    scalar ``dt`` would misstate the closing rate; otherwise REFUSED with the
    rebuild command. Every ``gt_*`` column is DROPPED (they were computed over the
    block's own 10-step grid) and the GT reference is recomputed on the view."""
    blk, idx, meta = ra.load_lead_block_rows(path)
    ts = np.asarray(blk["ts_rel_s"], dtype=np.float64).reshape(-1)
    want = np.arange(1, int(k) + 1, dtype=np.float64) * float(dt)
    cols, rows, inst = [], [], []
    for c, w in enumerate(want):
        hit = np.flatnonzero(np.abs(ts - w) <= ra.LEAD_TS_TOL_S)
        if hit.size:
            cols.append(c)
            rows.append(int(hit[0]))
            inst.append(float(w))
    rebuild = (f"taniteval/tools/build_lead_block_b1.py --dt {dt} --k {k} "
               f"(a block on the dump's OWN grid makes the FULL horizon scoreable)")
    if not cols:
        return None, None, meta, _refused(
            f"the lead block grid {np.round(ts, 3).tolist()} shares NO instant "
            f"with the dump grid {np.round(want, 3).tolist()} — WORK ITEM: "
            f"{rebuild}", "n/a", 0)
    gaps = np.diff(np.array(inst)) if len(inst) > 1 else np.array([inst[0]])
    if not (np.allclose(gaps, gaps[0])
            and abs(inst[0] - gaps[0]) <= ra.LEAD_TS_TOL_S):
        return None, None, meta, _refused(
            f"the common instants {inst} s are not a uniform grid starting at "
            f"their own spacing — a scalar dt would misstate the closing rate; "
            f"WORK ITEM: {rebuild}", "n/a", 0)
    view = {k_: v for k_, v in blk.items() if not str(k_).startswith("gt_")}
    view["ts_rel_s"] = np.array(inst, dtype=np.float64)
    view["leads"] = np.asarray(blk["leads"], dtype=np.float64)[:, rows]
    info = {"dump_cols": cols, "block_rows": rows, "instants_s": inst,
            "dt_s": float(gaps[0]), "k": len(inst),
            "dropped_dump_instants_s": [float(w) for c, w in enumerate(want)
                                        if c not in cols],
            "block_grid_s": np.round(ts, 3).tolist(),
            "rebuild_for_full_horizon": rebuild,
            "_is": ("distance-keeping is scored on the instants the dump and the "
                    "banked block SHARE, by index-select on BOTH sides — no "
                    "resampling of the lead track, no interpolation of the path")}
    return view, idx, meta, info


def _distance_keeping(rec, files, manifest, lead_path, arms, P_cat, G_all,
                      pairs, eid_w, n_boot, seed, tiers, dt, k, raw_off) -> dict:
    """``rec['refcv3']['distance_keeping']`` + the per-arm LONGITUDINAL patch.

    ⚠️ ``refav1_arm.join_lead_block`` keys on ``(clip_id, RAW frame 2t)`` because
    a refav1 window origin ``t`` is a 0.2 s CACHE step. refcv3's ``ws`` is a
    PROVIDER frame index, so the dump is rewritten into a temporary
    ``ws' = (ws + raw_off) / 2`` view for the join and the guards (the exact-grid
    assertion, the LABEL-FREE speed proof, NO_LABEL never scored as free flow)
    are the SAME ones. ⛔ If the offset were wrong the speed proof FAILS LOUDLY
    per episode (SPEED_MISMATCH) rather than placing another clip's traffic on
    these windows — that is why it is the guard and not a comment."""
    from taniteval import four_families as ff
    from taniteval import lead_metrics as lm
    view, idx, meta, info = lead_block_common_grid(lead_path, dt, k)
    base = {"block": lead_path, "block_sha256": ra._sha256(lead_path),
            "block_version": meta.get("version"), "block_tool": meta.get("tool"),
            "block_built_utc": meta.get("built_utc"),
            "block_rows_all": meta.get("n_rows"),
            "block_clips": meta.get("n_clips"),
            "conventions": meta.get("conventions"), "states": meta.get("states"),
            "join_key": ("(clip_id from the dump manifest, RAW frame = ws + "
                         f"{raw_off}); refav1_arm.join_lead_block maps t -> 2t, "
                         "so a half-frame view is passed to it")}
    if view is None:
        out = dict(base)
        out.update(info)
        return out
    # -- the half-frame view: join_lead_block does frame = 2 * ws --------------
    import tempfile
    shim_dir = tempfile.mkdtemp(prefix="refcv3_leadjoin_")
    shim_files = []
    odd = 0
    for fi, f in enumerate(files):
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float32).reshape(-1) if "v0" in d.files \
                else None
        raw = ws + int(raw_off)
        odd += int((raw % 2 != 0).sum())
        sf = os.path.join(shim_dir, os.path.basename(f))
        # frame-identity: this corpus's provider index IS the raw frame, so the
        # shim carries `raw` itself and the join is told frame_of_t = identity.
        # (`raw // 2` + the default 2t map truncated every ODD frame to frame-1,
        # which the label-free speed proof correctly refused.)
        np.savez(sf, ws=raw, **({"v0": v0} if v0 is not None else {}))
        shim_files.append(sf)
    if odd:
        base["_odd_raw_frames"] = (
            f"{odd} window origins land on an ODD raw frame; the B1 block carries "
            f"a row per RAW frame. RESOLVED: join_lead_block now takes "
            f"frame_of_t and this call passes IDENTITY, so each window joins "
            f"its own raw frame exactly; this count is informational only.")
    lead = ra.join_lead_block(shim_files, manifest, view, idx,
                              k=info["k"], dt=info["dt_s"],
                              frame_of_t=lambda t: t)
    cov = lead.pop("coverage")
    out = dict(base)
    out.update({"grid": info, "coverage": cov})
    n_lab = cov["counts"]["LEAD"] + cov["counts"]["NO_LEAD"]
    if n_lab == 0:
        out.update(_refused(
            f"the lead block covers 0 labelled windows of {cov['n_windows']} "
            f"(NO_LABEL {cov['counts']['NO_LABEL']}, NOT_STRAIGHT "
            f"{cov['counts']['NOT_STRAIGHT']}, no-row {cov['n_windows_no_row']}; "
            f"episodes OK {cov['n_episodes_ok']}/{cov['n_episodes']}) — nothing "
            f"is scoreable; NOT read as free flow", "n/a", 0))
        return out
    cols, dtv = info["dump_cols"], info["dt_s"]
    out.update({
        "status": "PRESENT", "n": int(cov["counts"]["LEAD"]),
        "tier": {x: tiers.get(x) for x in arms},
        "estimator": ("per arm: lead_metrics.distance_keeping on the common-grid "
                      "view + episode-cluster bootstrap; pairs: "
                      "lead_metrics.paired_distance_keeping"),
        "_binding": ("LONGITUDINAL distance-keeping: headway / time-gap / min-TTC "
                     "per arm with n and CI, per speed band, never pooled with "
                     "speed accuracy; a censored TTC carries n_closing; scored on "
                     f"the SHARED instants {info['instants_s']} s only")})
    pw, per_arm = {}, {}
    for arm in arms:
        dk = lm.distance_keeping(P_cat[arm][:, cols], lead["leads"],
                                 lead["lead_lens"], lead["speeds"], dtv)
        pw[arm] = {kk: np.asarray(dk[kk], dtype=np.float64) for kk in ra._DK_KEYS}
        blk = {"tier": tiers.get(arm), "status": dk.get("status"),
               "reason": dk.get("reason"), "n": dk.get("n"),
               "n_windows": dk.get("n_windows"), "dt_s": dtv,
               "instants_s": info["instants_s"],
               "mean_headway_min_m": dk.get("mean_headway_min_m"),
               "mean_time_gap_min_s": dk.get("mean_time_gap_min_s"),
               "n_time_gap": dk.get("n_time_gap"),
               "mean_min_ttc_s": dk.get("mean_min_ttc_s"),
               "n_closing": dk.get("n_closing"),
               "censoring_note": dk.get("censoring_note"),
               "ci": {kk: ra._boot(pw[arm][kk], eid_w, n_boot, seed)
                      for kk in ra._DK_KEYS},
               "_grid_note": (f"min-over-steps is a min over {info['k']} "
                              f"instant(s) {info['instants_s']} s — coarser than "
                              f"the block's own {len(info['block_grid_s'])}-step "
                              f"grid; a lead closest BETWEEN them is not seen. "
                              f"Rebuild: {info['rebuild_for_full_horizon']}")}
        if dk.get("status") == "OK":
            blk["by_speed"] = lm.distance_keeping_by_speed(
                dk, lead["speeds"], lead["eid"], states=lead["state"],
                n_boot=n_boot, seed=seed)
        per_arm[arm] = blk
        # -- patch the canonical family block, and re-check its CI coverage ----
        fam = rec["arms"][arm]["four_families"]
        lon = fam["longitudinal"]
        lon["distance_keeping"] = {
            **{kk: v for kk, v in blk.items() if kk != "tier"},
            "_declared_by": ("taniteval/tools/refcv3_arm.py — computed on the "
                             "COMMON-GRID view of the banked B1 EVAL lead block "
                             "(refav1_arm.join_lead_block guards: exact grid, "
                             "label-free speed proof, NO_LABEL never free flow)"),
            "_time_join": info["_is"]}
        ci_ = lon.setdefault("ci", {})
        comps_ = ci_.setdefault("components", {})
        unav_ = ci_.setdefault("unavailable", {})
        for kk, name in (("headway_min_m", "mean_headway_min_m"),
                         ("time_gap_min_s", "mean_time_gap_min_s"),
                         ("min_ttc_s", "mean_min_ttc_s")):
            key = f"distance_keeping.{name}"
            bb = blk["ci"][kk]
            if bb.get("status") == "NOT-APPLICABLE":
                unav_[key] = {"status": "UNAVAILABLE", "reason": bb.get("reason"),
                              "n": 0}
            else:
                comps_[key] = bb
                unav_.pop(key, None)
        fam["_ci_coverage"]["longitudinal"] = ff.ci_coverage(lon, "longitudinal")
        fam["_intervals_complete"] = bool(
            fam["_ci_coverage"]["longitudinal"].get("complete")
            and fam["_ci_coverage"]["lateral"].get("complete"))
    out["per_arm"] = per_arm
    gt = lm.distance_keeping(G_all[:, cols], lead["leads"], lead["lead_lens"],
                             lead["speeds"], dtv)
    out["gt_reference"] = {
        "_is": ("lead_metrics.distance_keeping of the GT ego path on the same "
                "instants: the value a perfect reproduction scores here"),
        **{kk: ra._boot(np.asarray(gt[kk], dtype=np.float64), eid_w, n_boot, seed)
           for kk in ra._DK_KEYS}}
    out["paired"] = {nm: lm.paired_distance_keeping(
        pw[hi], pw[lo], eid_w, names=(hi, lo), n_boot=n_boot, seed=seed)
        for lo, hi, nm in pairs}
    return out


# --------------------------------------------------------------------------- #
# the analysis (CPU)                                                           #
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
            for k in d.files:
                cat.setdefault(k, []).append(d[k])
        eid += [os.path.splitext(os.path.basename(f))[0]] * n
    return {k: np.concatenate(v) for k, v in cat.items()}, eid


#: an arm whose selection carries this little scene information is DEGENERATE:
#: the fraction of windows landing on the single most-chosen anchor.
SELECTION_DEGENERATE_FRAC = 0.9


def _selection_profile(dec, manifest) -> dict:
    """⛔ HOW MUCH SCENE INFORMATION IS IN THE MODEL'S OWN SELECTION?

    An anchored one-shot model can be fully degenerate while its PATHS look
    perfectly non-trivial: if ``sel_idx`` is a constant, the arm is "always
    anchor k, refined", and every family row below is a property of that one
    anchor. The trivial profile cannot see this (the paths bend and change
    speed), so it is asked here, separately, and BEFORE any family row.

    Reports ``n_distinct``, the modal anchor and its share, the selection
    entropy in nats against ``ln(n_anchors)``, and how often the deployed
    selection coincides with the ORACLE ``a_star``."""
    if not dec or "sel_idx" not in dec:
        return _refused("no decisions sidecar — the selection cannot be "
                        "profiled (this dump was not written by "
                        "refcv3_arm.run_dump)", "n/a", 0)
    n_anchors = int((((manifest or {}).get("model") or {}).get("n_anchors")) or 128)
    sel = np.asarray(dec["sel_idx"], dtype=np.int64)
    n = int(sel.size)
    counts = np.bincount(sel, minlength=n_anchors)
    modal = int(counts.argmax())
    modal_frac = float(counts[modal] / max(1, n))
    ent = _entropy(sel, n_anchors)
    star = (np.asarray(dec["a_star"], dtype=np.int64) if "a_star" in dec
            else None)
    out = {
        "n_windows": n, "n_anchors": n_anchors,
        "n_distinct_selected": int(np.count_nonzero(counts)),
        "modal_anchor": modal, "modal_frac": round(modal_frac, 4),
        "entropy_nats": round(float(ent), 4),
        "max_entropy_nats": round(float(np.log(n_anchors)), 4),
        "entropy_ratio": round(float(ent / max(1e-12, np.log(n_anchors))), 4),
        "agrees_with_oracle_frac": (round(float((sel == star).mean()), 4)
                                    if star is not None else None),
        "degenerate": bool(modal_frac >= SELECTION_DEGENERATE_FRAC),
        "rule": {"degenerate_modal_frac_at_or_above": SELECTION_DEGENERATE_FRAC},
        "_reading": ("a DEGENERATE selection means the model emits essentially "
                     "ONE anchor for every scene: the arm is a constant map and "
                     "every family row is a property of that constant, not of "
                     "the model's scene understanding. It is invisible to the "
                     "trivial profile, because the constant anchor is neither "
                     "straight nor constant-speed."),
    }
    return out


def _print_selection_profile(sp: dict) -> None:
    if sp.get("status") in ("REFUSED", "UNAVAILABLE"):
        _p(f"[selection-profile] {sp.get('status')} — {str(sp.get('reason'))[:120]}")
        return
    _p(f"[selection-profile] {sp['n_windows']} windows over {sp['n_anchors']} "
       f"anchors — the model's OWN choice, before any family row")
    _p(f"  n_distinct={sp['n_distinct_selected']:4d}  modal=#{sp['modal_anchor']} "
       f"({sp['modal_frac']:.4f})  entropy={sp['entropy_nats']:.4f}/"
       f"{sp['max_entropy_nats']:.4f} nats (ratio {sp['entropy_ratio']:.4f})  "
       f"agrees_with_oracle={sp['agrees_with_oracle_frac']}")
    if sp["degenerate"]:
        _p(f"  ⛔ VOID-RISK: the selection is a CONSTANT on "
           f"{sp['modal_frac']:.2%} of windows — the arm is 'always anchor "
           f"#{sp['modal_anchor']}, refined'. Read every family row below as a "
           f"property of that one anchor, NEVER as scene understanding. The "
           f"trivial profile CANNOT see this.")


def analyze_refcv3(dump_dir: str, *, n_boot: int = 2000, seed: int = 0,
                   dt: float | None = None, tiers: dict | None = None,
                   lead_block: str | None = None,
                   labels: str | None = None) -> dict:
    """``t1_eval.analyze`` on the trajectory dump + the refcv3 sidecar analysis."""
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
    grid_man = (manifest or {}).get("grid") or {}
    dt = float(dt if dt is not None else grid_man.get("dt_s") or GRIDS["2s"][0])
    raw_off = 0
    fr = ((manifest or {}).get("corpus") or {}).get("frames") or {}
    if "provider_to_raw_frame_offset" in fr:
        raw_off = int(fr["provider_to_raw_frame_offset"])
    tiers = dict(ARM_TIERS, **(tiers or {}))
    with np.load(files[0]) as d0:
        arms = [k for k in d0.files if k not in ("g", "ws", *t1._META_KEYS)
                and not k.endswith(t1._FAN_SUFFIXES)]
        k_dump = int(d0["g"].shape[1])
    unstamped = [x for x in arms if tiers.get(x) not in ("T0", "T1")]
    if unstamped:
        raise SystemExit(
            f"[refcv3_arm] arms {unstamped} carry no T0/T1 tier stamp. Every "
            f"emitted number carries its tier (EVAL_DOCTRINE.md); pass "
            f"--tiers name=T0|T1. Known here: {sorted(ARM_TIERS)}")
    # ⭐ THE HEADLINE PAIRS: the echo test's REAL bar is `os - ha0`, never `- ha`
    # alone. `ha` holds a noisy observed steer and drifts, so an arm that does
    # NOTHING can beat it and read as lateral skill (MEASURED 2026-09-03).
    pairs = [(x, y, nm) for x, y, nm in (
        ("ha0", "os", "paired_os_minus_ha0"),
        ("ha", "os", "paired_os_minus_ha"),
        ("os_navshuf", "os", "paired_os_minus_navshuf"),
        # ⭐ BACKLOG R39. The nav-ZERO arm needs BOTH of these: its own MARGIN
        # over the shared floor (the deployment-relevant number — this is what
        # the model is worth without the oracle nav) and the Δ against `os`
        # (what the oracle nav is worth). The shuffle answers neither.
        ("ha0", "os_navzero", "paired_os_navzero_minus_ha0"),
        ("os_navzero", "os", "paired_os_minus_navzero"),
        ("ha0", "ha", "paired_ha_minus_ha0"),
        # ⭐⭐ THE OTHER HALF OF refcv5's ACCEPTANCE BAR. `os - ha0_ext` is the
        # margin over the ego-extrapolation echo; an arm that beats `ha` while
        # TYING this one has echoed, not driven (echo_gate.ha0_ext). The
        # nav-zero pairing is the DEPLOYMENT-relevant form of the same read.
        ("ha0_ext", "os", "paired_os_minus_ha0ext"),
        ("ha0_ext", "os_navzero", "paired_os_navzero_minus_ha0ext"),
        ("os", "oracle_sel", "paired_oraclesel_minus_os"))
        if x in arms and y in arms]

    # ---- ⭐ THE TRIVIAL-PROFILE INSTRUMENT, BEFORE ANY FAMILY ROW ------------
    # It runs HERE, not later, on purpose: a reader who sees the family table
    # first has already formed the reading this instrument exists to prevent.
    # One instrument, not two — refav1_arm's, imported.
    triv = ra.trivial_profile(files, arms, dt=dt)
    ra._print_trivial_profile(triv)
    # ⚠️ `ha0` IS the constant-velocity plan BY DEFINITION, so it is degenerate on
    # every correct run. Escalating on it would make the loudest warning in this
    # tool fire every time and be learned as noise — so the VOID-RISK line names
    # only the arms whose degeneracy is a FINDING.
    triv["degenerate_arms_excluding_floor"] = [
        x for x in triv["degenerate_arms"] if x not in ("ha0", "ha0_ext")]
    triv["_floor_note"] = (
        "`ha0` is expected in degenerate_arms — it IS the constant-velocity "
        "plan. `ha0_ext` is excluded for a DIFFERENT reason: it is a "
        "constant-(a0, k0) extrapolation, so its trivial fraction is a "
        "property of HOW STRAIGHT AND STEADY THE EGO WAS AT t0 on this corpus, "
        "not a defect of the arm — escalating on it would make the loudest "
        "warning in this tool fire on a control and be learned as noise. Both "
        "stay IN degenerate_arms and are read through `ha0_ext_vs_ha0` below. "
        "VOID-RISK is raised only on degenerate_arms_excluding_floor.")
    # ⭐ THE READ THAT MATTERS FOR THE BAR: if `ha0_ext` is bit-identical to
    # `ha0` on most windows then "beat BOTH `ha` and `ha0_ext`" has collapsed
    # into "beat `ha0`", and the echo control is adding nothing on this surface.
    # Say so as a number rather than letting a reader assume two controls.
    _ext = (triv["arms"].get("ha0_ext") or {})
    _ext_vs_ha0 = (_ext.get("identical_to") or {}).get("ha0")
    triv["ha0_ext_vs_ha0"] = {
        "identical_frac": (None if _ext_vs_ha0 is None
                           else float(_ext_vs_ha0["frac"])),
        "n_identical": None if _ext_vs_ha0 is None else int(_ext_vs_ha0["n"]),
        "n": _ext.get("n"),
        "means": ("the ECHO control and the CONSTANT-VELOCITY floor are the "
                  "same path on that fraction of windows; on those windows the "
                  "acceptance bar has ONE control in it, not two"),
    }
    if _ext_vs_ha0 is not None:
        _p(f"  ha0_ext ≡ ha0 on {_ext_vs_ha0['n']}/{_ext.get('n')} windows "
           f"({_ext_vs_ha0['frac']:.2%}) — on those the echo control adds "
           f"nothing beyond the constant-velocity floor, and 'beat BOTH' "
           f"collapses to 'beat ha0'.")
    if triv["degenerate_arms_excluding_floor"]:
        _p(f"  ⛔ VOID-RISK: {triv['degenerate_arms_excluding_floor']} are the "
           f"CONSTANT-VELOCITY plan on > 50 % of windows. A read whose arm is "
           f"the baseline, or bit-identical to another arm, is stamped VOID — "
           f"never reported as 'no difference'.")
    # ⚠️ CROSS-CALL ARMS AND THE 1e-9 THRESHOLD. `identical_to` is exact only
    # for arms produced by the SAME forward call. `os_navzero` is its own call
    # (nav_cmd=None gates E13 for the whole call), so a float32 batching floor of
    # ~6e-7 m sits under it and it can NEVER read as identical at 1e-9 — even
    # where the computation is provably the same (MEASURED at matched batch size:
    # exactly 0.0). Say so here, or the diagnostic reads as evidence it is not.
    cross_call = [x for x in arms if x == "os_navzero"]
    triv["cross_call_arms"] = cross_call
    triv["cross_call_note"] = (NAV_NULL["⚠️ cross_call_float32_floor"]
                               if cross_call else None)
    if cross_call:
        _p(f"  ⚠️ cross-call arms {cross_call}: produced by a SEPARATE forward "
           f"(nav_cmd=None gates E13 for the whole call), so a float32 batching "
           f"floor of ~6e-7 m sits under them — `identical_to` at "
           f"{ra.TRIVIAL_IDENTICAL_M:g} m cannot resolve them and must not be "
           f"read as nav evidence. A REAL nav difference measured ~4.7e4x that.")
    ident_os = ((triv["arms"].get("os") or {}).get("identical_to") or {})
    for other, r in ident_os.items():
        if other != "ha0" and r["frac"] > 0.5:
            _p(f"  ⛔ VOID-RISK: `os` is bit-identical to `{other}` on "
               f"{r['n']}/{triv['arms']['os']['n']} windows ({r['frac']:.2%}) — "
               f"on those windows the instrument saw ONE arm, not two.")

    # ---- ⭐ THE SELECTION PROFILE, ALSO BEFORE ANY FAMILY ROW ----------------
    # ⛔ WHY IT EXISTS, AND WHY THE TRIVIAL PROFILE ALONE IS NOT ENOUGH HERE
    # (MEASURED on this tool's own fixture, 2026-09-03): a random-init RefCV3
    # selected ONE anchor on 42/42 windows. Its paths are neither straight nor
    # constant-speed — the anchor bends and the refinement moves with the scene —
    # so `trivial_frac` reads 0.0000 and the trivial profile CANNOT see it. Yet
    # the model's selection carried ZERO scene information, which is refcv3's
    # characteristic degeneracy and exactly the reading a family table would
    # otherwise be given as skill.
    dec_early, _ = _load_decisions(dump_dir)
    selp = _selection_profile(dec_early, manifest)
    _print_selection_profile(selp)

    # ---- the analysis ------------------------------------------------------- #
    rec = t1.analyze(files, tiers={x: tiers[x] for x in arms},
                     n_boot=n_boot, seed=seed, dt=dt, paired=pairs)
    rec["tool"] = ("taniteval/tools/refcv3_arm.py (trajectory families via "
                   "taniteval/tools/t1_eval.py::analyze, IMPORTED)")

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
    comps = {x: ra._components(P_cat[x], G_all, dt) for x in arms}

    ref = {
        "n_windows": N, "n_episodes": len(files),
        "tiers": {x: tiers[x] for x in arms},
        "arm_meaning": {x: ARM_MEANING.get(x) for x in arms},
        "absent_arms": ABSENT_ARMS,
        "tier_ruling": TIER_RULING,
        "_tier_doctrine": rec["_tier_doctrine"],
        # ⭐ banked FIRST in the record too, for the same reason they print first.
        "trivial_profile": triv,
        "selection_profile": selp,
        "t1_definition": (manifest or {}).get("t1_definition"),
        "families_paired": {nm: ra._paired_families(comps, x, y, eid_w, tiers,
                                                    n_boot, seed)
                            for x, y, nm in pairs},
        "headline": {
            "_is": ("the admissible cross-model statistic is each arm's MARGIN "
                    "over the SAME ha0 floor, per family, paired — "
                    "(cl - ha0)_refav1 vs (os - ha0)_refcv3. ⛔ NEVER cl vs os "
                    "as levels (D-HF-COMPARABILITY)."),
            "margin_block": "families_paired.paired_os_minus_ha0",
            "floor_arm": "ha0",
            "floor_is_bit_comparable_with_refav1": True,
            # ⭐ BACKLOG R39: nav is an ORACLE input that will not exist at
            # deployment, so the margin WITHOUT it is the deployment-relevant
            # one and must be quoted beside the oracle-nav margin.
            "deployment_margin_block":
                "families_paired.paired_os_navzero_minus_ha0",
            "oracle_nav_worth_block": "families_paired.paired_os_minus_navzero",
            "_nav_controls": ("BOTH are binding and they are NOT "
                              "interchangeable: os_navshuf breaks the PAIRING "
                              "with the nav marginal preserved; os_navzero "
                              "removes the SIGNAL. See refcv3.nav_null."),
        },
        "nav_null": (manifest or {}).get("nav_null"),
        "families_note": (
            "LONGITUDINAL / LATERAL / ADE and the TRAJECTORY-DERIVED tactical "
            "rows per arm live in rec['arms'][arm]['four_families'] (t1_eval, "
            "unchanged). The DECLARED tactical decisions, the anchor selection "
            "and STRATEGIC come from refcv3's own heads and live below "
            "(rec['refcv3']); the trajectory-only STRATEGIC row in rec['arms'] "
            "stays UNAVAILABLE by design because a route class cannot be read "
            "off a short path."),
    }

    # ---- LONGITUDINAL distance-keeping -------------------------------------- #
    if lead_block:
        ref["distance_keeping"] = _distance_keeping(
            rec, files, manifest, lead_block, arms, P_cat, G_all, pairs, eid_w,
            n_boot, seed, tiers, dt, k_dump, raw_off)
        dk = ref["distance_keeping"]
        cov = dk.get("coverage")
        if cov:
            _p(f"[lead] {os.path.basename(lead_block)}: {cov['n_windows']} windows / "
               f"{cov['n_episodes']} eps · OK eps {cov['n_episodes_ok']} · "
               f"LEAD {cov['counts']['LEAD']} NO_LEAD {cov['counts']['NO_LEAD']} "
               f"NOT_STRAIGHT {cov['counts']['NOT_STRAIGHT']} NO_LABEL "
               f"{cov['counts']['NO_LABEL']} · speed-check max "
               f"{cov['speed_check']['max_mps']} m/s · {dk.get('status')}")
        else:
            _p(f"[lead] {dk.get('status')} — {str(dk.get('reason'))[:200]}")
    else:
        ref["distance_keeping"] = _refused(
            "no lead block passed (--lead-block); the banked B1 EVAL block is "
            f"{ra.LEAD_BLOCK_DEFAULT}. The LONGITUDINAL family's distance-keeping "
            "half is a WORK ITEM, not a pass.", "n/a", 0)

    dec, eid_d = _load_decisions(dump_dir)
    if dec is None:
        ref["sidecar"] = _refused("no decisions/ep*.npz sidecar — this dump was "
                                  "not written by refcv3_arm.run_dump", "n/a")
        rec["refcv3"] = ref
        return rec
    if len(eid_d) != N:
        raise ValueError(f"decisions sidecar has {len(eid_d)} rows for {N} "
                         f"windows — different grids; refusing the join")

    n_anchors = int((((manifest or {}).get("model") or {})
                     .get("n_anchors")) or 128)
    # ---- TACTICAL: the anchor selection + the declared heads ---------------- #
    tac = {"tier": "T1 (the heads read the OBSERVED window only)",
           "tier_ruling": TIER_RULING["status"],
           "n_windows": N,
           "anchor_selection": {
               "_is": ("the TACTICAL family's goal/anchor-selection half. "
                       "`anchor_acc` is 1 where anchor_logits.argmax equals "
                       "a_star, the GT-NEAREST anchor "
                       "(refc_v3_train.py:460, :642) — chance is 1/n_anchors."),
               "n_anchors": n_anchors,
               "chance": round(1.0 / max(1, n_anchors), 6),
               "anchor_acc": _ci.episode_cluster_bootstrap(
                   dec["anchor_acc"].astype(np.float64), eid_d,
                   n_boot=n_boot, seed=seed),
               "deployed_selection_agrees_oracle": _ci.episode_cluster_bootstrap(
                   dec["sel_agrees_oracle"].astype(np.float64), eid_d,
                   n_boot=n_boot, seed=seed),
               "_reading": ("deployed_selection_agrees_oracle is how often the "
                            "DEPLOYED path IS the oracle path. Where it is low, "
                            "the run's own `eval_traj` (oracle-selected) is a "
                            "LOOSE lower bound on what refcv3 would drive."),
               "n_distinct_selected": int(np.unique(dec["sel_idx"]).size),
               "selected_entropy_nats": float(_entropy(dec["sel_idx"], n_anchors)),
               "_degeneracy_note": ("n_distinct_selected == 1 means the model "
                                    "emits ONE anchor for every scene: the arm is "
                                    "a constant, and every family row below is a "
                                    "property of that constant."),
           },
           "declared_heads": {}}
    for hk in ("lat", "lon"):
        lbl = dec[f"{hk}_label"].astype(int)
        m = lbl != -100
        names = [str(i) for i in range(int(max(lbl.max(initial=0), 0)) + 1)]
        for cname in ("nav_true", "nav_shuffled", "nav_zero"):
            key = f"{hk}_{cname}"
            pred = dec.get(f"{hk}_pred_{cname}")
            if pred is None or (pred.astype(int) < 0).all():
                tac["declared_heads"][key] = _refused(
                    f"the {cname} conditioning was not rolled, or this build has "
                    f"no {hk} head (pred = -1)", "T1", N)
            elif m.sum() == 0:
                tac["declared_heads"][key] = _refused(
                    f"no window carries a {hk} v7.2 label (IGNORE_ID everywhere)",
                    "T1", 0)
            else:
                pi = pred.astype(int)
                nn = [str(i) for i in range(
                    int(max(lbl[m].max(), pi[m].max())) + 1)]
                e_m = [e for e, kk in zip(eid_d, m) if kk]
                blk = ff._agreement_block(lbl[m], pi[m], nn, e_m, n_boot, seed,
                                          tier="T1")
                blk["n_excluded_no_label"] = int((~m).sum())
                blk["chance_ce_nats_if_8_class"] = round(float(np.log(8.0)), 4)
                tac["declared_heads"][key] = blk
        if m.sum() and dec.get(f"{hk}_pred_nav_shuffled") is not None \
                and not (dec[f"{hk}_pred_nav_true"].astype(int) < 0).all() \
                and not (dec[f"{hk}_pred_nav_shuffled"].astype(int) < 0).all():
            e_m = [e for e, kk in zip(eid_d, m) if kk]
            ct = (dec[f"{hk}_pred_nav_true"].astype(int) == lbl).astype(float)
            cs = (dec[f"{hk}_pred_nav_shuffled"].astype(int) == lbl).astype(float)
            tac[f"{hk}_paired_true_minus_shuffled_accuracy"] = \
                _ci.paired_episode_cluster_bootstrap(ct[m], cs[m], e_m,
                                                     n_boot=n_boot, seed=seed)
    ref["tactical_declared"] = tac

    # ---- STRATEGIC: the route head, with the nav-echo control beside it ------ #
    from tanitad.refs.refb import ROUTE_CLASSES
    nav_valid = dec["nav_valid"].astype(bool)
    route_lbl = dec["route_label"].astype(int)
    labeled = route_lbl != -100
    strat = {"tier": "T1 (the route head reads the OBSERVED window only)",
             "tier_ruling": TIER_RULING["status"],
             "n_windows": N, "n_nav_valid": int(nav_valid.sum()),
             "nav_valid_frac": round(float(nav_valid.mean()), 4) if N else None,
             "n_route_labeled": int(labeled.sum()),
             "n_excluded_no_route_label": int((~labeled).sum()),
             "nav_shuffle": (manifest or {}).get("nav_shuffle"),
             "_echo_caveat": ("nav_cmd is an INPUT and the route label derives "
                              "from the same clip, so under the TRUE nav this "
                              "measures the nav ECHO (flagship v1 scored 1.0000 "
                              "on a bijection of its own input). Evidence of "
                              "route skill is ONLY the shuffled/zero "
                              "conditioning, and the CHANGED subset is where the "
                              "control has power at all."),
             "conditionings": {}}
    use = labeled & nav_valid
    eid_use = [e for e, kk in zip(eid_d, use) if kk]
    _strat_conds = ["nav_true", "nav_shuffled", "nav_zero"]
    for _extra in ("nav_predicted", "nav_flipped"):
        if dec.get("route_pred_%s" % _extra) is not None:
            _strat_conds.append(_extra)
    for cname in _strat_conds:
        pred = dec.get(f"route_pred_{cname}")
        if pred is None or (pred.astype(int) < 0).all():
            strat["conditionings"][cname] = _refused(
                f"the {cname} conditioning was not rolled (pred = -1)", "T1", N)
            continue
        if use.sum() == 0:
            strat["conditionings"][cname] = _refused(
                "no window is both route-labeled and nav-valid", "T1", 0)
            continue
        pi = pred.astype(int)
        blk = ff._agreement_block(route_lbl[use], pi[use], list(ROUTE_CLASSES),
                                  eid_use, n_boot, seed, tier="T1")
        # |STOP||STOP| CORRECTED 2026-09-06. The line below used to compare
        # `route_pred` (a ROUTE_CLASSES index, 3-wide) DIRECTLY against
        # `nav_cmd` (a NAV_COMMANDS index, FOUR-wide) on the claim that the two
        # are "index-aligned in refb". THEY ARE NOT: refb.py:64 is
        # ("follow","left","right","straight") and refb.py:68 is
        # ("route_left","route_straight","route_right"), and the mapping
        # refb_labels.py:483-484 states is {0:1, 1:0, 2:2}. The uncorrected form
        # published 0.1621 for refcv4b @40,284 where the true value is 0.6405.
        # Both are emitted so an old record can be reconciled against a new one.
        _r2n = _route_to_nav_map()
        _pi_nav = np.array([_r2n.get(int(x), -1) for x in pi])
        blk["nav_echo_index"] = round(
            float((_pi_nav[use] == dec["nav_cmd"][use].astype(int)).mean()), 4)
        blk["nav_echo_index_RAW_INDEX_LEGACY"] = round(
            float((pi[use] == dec["nav_cmd"][use].astype(int)).mean()), 4)
        blk["_nav_echo_index_is"] = (
            "fraction of scored windows whose route_pred, MAPPED through "
            "refb_labels._ROUTE_TO_NAV = {0:1, 1:0, 2:2}, equals the FED nav "
            "index. |STOP| NAV_COMMANDS is FOUR-wide (refb.py:64) and "
            "ROUTE_CLASSES THREE (refb.py:68); they are NOT index-aligned, and "
            "the pre-2026-09-06 form of this key compared them as if they were "
            "-- it is kept as nav_echo_index_RAW_INDEX_LEGACY for reconciliation "
            "only and is NOT the echo index.")
        maj = int(np.bincount(route_lbl[use].clip(min=0),
                              minlength=len(ROUTE_CLASSES)).argmax())
        blk["majority_class"] = ROUTE_CLASSES[maj]
        blk["majority_class_rate"] = round(float((route_lbl[use] == maj).mean()), 4)
        strat["conditionings"][cname] = blk
    if use.sum() and dec.get("route_pred_nav_shuffled") is not None \
            and not (dec["route_pred_nav_true"].astype(int) < 0).all() \
            and not (dec["route_pred_nav_shuffled"].astype(int) < 0).all():
        pt_ = dec["route_pred_nav_true"].astype(int)
        ps_ = dec["route_pred_nav_shuffled"].astype(int)
        c_t = (pt_ == route_lbl).astype(float)
        c_s = (ps_ == route_lbl).astype(float)
        strat["paired_true_minus_shuffled_accuracy"] = \
            _ci.paired_episode_cluster_bootstrap(c_t[use], c_s[use], eid_use,
                                                 n_boot=n_boot, seed=seed)
        changed = use & (dec["nav_cmd_shuf"].astype(int) != dec["nav_cmd"].astype(int))
        _n2r = _nav_to_route_map()
        _imp_s = np.array([_n2r.get(int(x), -1)
                           for x in dec["nav_cmd_shuf"].astype(int)])
        strat["n_changed_subset"] = int(changed.sum())
        if changed.sum():
            e_c = [e for e, kk in zip(eid_d, changed) if kk]
            strat["changed_subset"] = {
                "n": int(changed.sum()), "tier": "T1",
                "estimator": "episode_cluster_bootstrap",
                "route_follows_LABEL_under_shuffle": _ci.episode_cluster_bootstrap(
                    c_s[changed], e_c, n_boot=n_boot, seed=seed),
                "route_follows_SHUFFLED_NAV_under_shuffle":
                    _ci.episode_cluster_bootstrap(
                        (ps_ == _imp_s).astype(float)[changed],
                        e_c, n_boot=n_boot, seed=seed),
                "route_follows_SHUFFLED_NAV_RAW_INDEX_LEGACY":
                    _ci.episode_cluster_bootstrap(
                        (ps_ == dec["nav_cmd_shuf"].astype(int)).astype(float)[changed],
                        e_c, n_boot=n_boot, seed=seed),
                "_reading": ("an ECHO reads follows_nav ~ 1 / follows_label ~ 0; "
                             "route skill from vision reads follows_label high "
                             "regardless of the token. |STOP| CORRECTED "
                             "2026-09-06: `follows_nav` now maps the fed nav "
                             "token THROUGH refb_labels._NAV_TO_ROUTE before "
                             "comparing, because route_pred is a ROUTE_CLASSES "
                             "index (3-wide) and nav_cmd_shuf a NAV_COMMANDS "
                             "index (4-wide). The pre-correction form is kept as "
                             "..._RAW_INDEX_LEGACY for reconciliation ONLY."),
                "_exclusivity": ("|STOP| the old text claimed the two rates are "
                                 "mutually exclusive BY CONSTRUCTION. MEASURED "
                                 "FALSE: route_label is a PER-WINDOW target "
                                 "(route_from_future) while nav_cmd is a "
                                 "PER-CLIP 25 s token, so the label can equal a "
                                 "DIFFERENT token's implied route. "
                                 "`changed_exclusive_subset` below is the subset "
                                 "where they really are exclusive, and it is the "
                                 "one to quote."),
            }
            _excl = changed & (route_lbl != _imp_s)
            if _excl.sum():
                e_x = [e for e, kk in zip(eid_d, _excl) if kk]
                strat["changed_exclusive_subset"] = {
                    "n": int(_excl.sum()),
                    "n_dropped_label_equals_implied_route":
                        int((changed & (route_lbl == _imp_s)).sum()),
                    "tier": "T1", "estimator": "episode_cluster_bootstrap",
                    "route_follows_LABEL": _ci.episode_cluster_bootstrap(
                        c_s[_excl], e_x, n_boot=n_boot, seed=seed),
                    "route_follows_SHUFFLED_NAV": _ci.episode_cluster_bootstrap(
                        (ps_ == _imp_s).astype(float)[_excl], e_x,
                        n_boot=n_boot, seed=seed),
                    "_is": ("the changed subset RESTRICTED to windows where the "
                            "label and the shuffled token's implied route are "
                            "genuinely different classes. Only here are the two "
                            "rates mutually exclusive, so only here does the "
                            "comparison mean what it says."),
                }
        else:
            strat["changed_subset"] = _refused(
                "the permutation changed no nav token (FOLLOW-dominated "
                "marginal) — the shuffle control has no power on this set", "T1")
    else:
        strat["changed_subset"] = _refused(
            "no nav-shuffled conditioning in this dump (--no-navshuf or "
            "--nav-source none) — ⛔ the STRATEGIC family is INADMISSIBLE without "
            "it", "T1")
    # ---- ⭐ NAV-COMPLIANCE: the BEHAVIOURAL strategic metric (PI 2026-09-04:
    # "we are evaluating the fact that the model is following the nav command in
    # consistency to the strategic goals"). Decided by the intervention pair on
    # the same windows, never by a rate; a dump without the sidecar keys gets a
    # REFUSAL with its reason.
    _nc = None
    try:
        from taniteval import nav_compliance as _nc
        strat["nav_compliance"] = _nc.from_refcv3_dump(
            dump_dir, labels_path=labels, n_boot=n_boot, seed=seed)
    except Exception as ex:      # noqa: BLE001 — a refusal, OR a defect
        #: ⛔ THE TWO KINDS OF "IT DID NOT RUN" MUST NOT LOOK ALIKE.
        #: A missing labels blob or a pre-sidecar dump is a legitimate REFUSAL
        #: and the four-families rule allows dropping the family WITH ITS
        #: REASON. A `TypeError` from our own module is a DEFECT — and one of
        #: those (`os.path.exists(<dict>)`) silently removed the STRATEGIC
        #: nav-compliance family from EVERY refcv3 arm for its whole life while
        #: the other three families reported normally and no run ever failed.
        _defect = isinstance(ex, DEFECT_EXCEPTIONS)
        _why = f"nav_compliance did not run: {type(ex).__name__}: {str(ex)[:300]}"
        if _defect:
            _why = ("⛔ DEFECT (not a refusal) — this is a bug in our own "
                    "code, not a missing input: " + _why)
        _blk = (_nc.unavailable_block(_why, N) if _nc is not None
                else _refused(_why, "T1", N))
        if _defect:
            #: recorded ON the block AND collected at the top level, so a driver
            #: can exit non-zero instead of publishing a family-shaped hole.
            _blk["defect"] = True
            _blk["defect_type"] = type(ex).__name__
            ref.setdefault("_defects", []).append(
                {"where": "strategic.nav_compliance",
                 "type": type(ex).__name__, "detail": str(ex)[:300]})
        strat["nav_compliance"] = _blk
    ref["strategic"] = strat

    # ---- the E9 goal gate: the value AND the scale it multiplies ------------- #
    gg, gs = dec.get("goal_gate"), dec.get("goal_score_absmean")
    if gg is None or not np.isfinite(gg.astype(np.float64)).any():
        ref["goal_gate"] = _refused(
            "this build has no E9 goal cascade (flat arm) — no gate exists",
            "T1", N)
    else:
        ref["goal_gate"] = {
            "tier": "T1", "n": N,
            "gate_mean": round(float(np.nanmean(gg.astype(np.float64))), 8),
            "score_absmean_mean": (round(float(np.nanmean(gs.astype(np.float64))), 8)
                                   if gs is not None else None),
            "_binding": ("⛔ REPORT BOTH. The gate alone cannot distinguish "
                         "'has not opened yet' from 'will never open'; the score "
                         "scale it multiplies is what makes the reading "
                         "falsifiable (CAVEAT-B, PI 2026-09-02)."),
        }

    # ---- the T0 arms that DO NOT exist here, said out loud ------------------- #
    ref["law_diagnostic"] = _refused(
        "`law` consumes a FUTURE frame's pooled latent (refc_v3_train.py's LAW "
        "target). It is a T0 WM-fidelity diagnostic with NO refav1 analogue and "
        "it can never enter a T1 row (D-HF-COMPARABILITY). Not computed here on "
        "purpose: this adapter never decodes a future frame.", "T0", 0)

    ref["protocol"] = {
        "inference_inputs": ("the OBSERVED window's frames (vision); the MEASURED "
                             "v0 at t0 (PI ruling 2026-09-02); the clip's v7.2 nav "
                             "token (goal input, PI 2026-08-03); NOTHING recorded "
                             "after the window origin on any arm here"),
        "vision_only": ("vision + v0(t0) + nav token — both additions admitted by "
                        "the two PI rulings above; no ego state beyond v0, no "
                        "future"),
        "goal_source": ("the model's own tactical goal head (E4/E9), decoded from "
                        "vision + nav + v0 inside the forward pass; the selection "
                        "it grafts is banked per window (sel_score_v3)"),
        "goal_situation_disjoint": ("True by construction: nav is the v7.2 labels "
                                    "blob's nav_command token, not a situation "
                                    "classifier output"),
        "corpus": ((manifest or {}).get("corpus") or {}).get("episodes"),
        "labels": ((manifest or {}).get("corpus") or {}).get("labels"),
        "parity_key": ("the eval split is the v7.2 EVAL clip set, identified by "
                       "the labels blob md5 in corpus.labels — NOT the canonical "
                       "train parity key e438721ae894; cross-arm deltas are valid "
                       "only against dumps on this SAME corpus + grid"),
    }
    for arm in rec.get("arms", {}):
        fam = rec["arms"][arm].get("four_families")
        if not isinstance(fam, dict) or "_protocol" not in fam:
            continue
        fam["_protocol"].update({k: v for k, v in ref["protocol"].items()
                                 if k in fam["_protocol"]})
        fam["_protocol"]["_declared_by"] = ("taniteval/tools/refcv3_arm.py "
                                            "analyze_refcv3 (post hoc, from the "
                                            "roll's manifest + the model's own "
                                            "input contract)")
        fam["_protocol_undeclared"] = sorted(
            k for k, v in fam["_protocol"].items()
            if not k.startswith("_") and isinstance(v, str)
            and v.startswith("UNDECLARED"))
    ref["manifest"] = manifest
    # ⭐⭐ THE ABLATION STAMP TRAVELS TO EVERY ARM BLOCK. A per-arm number that
    # can be read without its ablation is a number that WILL be read without it.
    _abl = (manifest or {}).get("ablation") or {
        "applied": None,
        "⛔ provenance": ("UNKNOWN — this dump's manifest predates the ablation "
                         "stamp (Rung A1, 2026-09-05). That is NOT evidence "
                         "that no ablation was applied; absence of the field is "
                         "absence of the record."),
    }
    ref["ablation"] = _abl
    for _blk in (rec.get("arms") or {}).values():
        _blk["ablation"] = _abl
    rec["refcv3"] = ref
    return rec


def _entropy(idx, n_classes: int) -> float:
    c = np.bincount(np.asarray(idx, dtype=np.int64), minlength=int(n_classes))
    p = c[c > 0] / max(1, c.sum())
    # max(0.0, ...): a single-class histogram gives -(1*log 1) = -0.0, which
    # prints as "-0.0000" and reads like a broken metric rather than a degenerate
    # selection. The value is zero; say zero.
    return float(max(0.0, -(p * np.log(p)).sum()))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="refcv3 eval adapter — writes a t1_eval-compatible dump + a "
                    "decisions sidecar and analyses both. Read "
                    "taniteval/tools/REFCV3_ARM.md §2 first: the deployed arm is "
                    "`os` (one-shot), never `cl`.")
    ap.add_argument("--ckpt", help="refc_v3_train.py ckpt.pt")
    ap.add_argument("--config", default=None,
                    help="config.json (default: sibling of --ckpt)")
    ap.add_argument("--episodes", help="v2 episode cache dir (<clip>.v2ep.pt)")
    ap.add_argument("--labels", default=None,
                    help="the v7.2 EVAL labels blob (REQUIRED)")
    ap.add_argument("--nav-source", choices=NAV_SOURCES, default="auto",
                    help="auto = whatever the run trained on (config.json "
                         "nav_from_v7); v72 = the clip's v7.2 token; none = "
                         "nav_cmd 0 everywhere and NO shuffle arm")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="2s",
                    help="the dump grid — an INDEX-SELECT of V3_HORIZONS")
    ap.add_argument("--arm", default="refcv3", help="label for the output record")
    ap.add_argument("--out", help="JSON output FILE")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--analyze-only", default=None, metavar="DUMP_DIR")
    ap.add_argument("--dump-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes-n", type=int, default=0,
                    help="first N clips of the cache manifest (0 = all)")
    ap.add_argument("--window-stride", type=int, default=1)
    ap.add_argument("--lru", type=int, default=8)
    ap.add_argument("--action-units", choices=("kappa", "steer"), default="steer",
                    help="unit of the RECORDED actions[:,0] as this run READS it. "
                         "DEFAULT 'steer' = the repaired contract (kappa = "
                         "tan(steer)/2.9 before integration), because "
                         "physicalai.py:621 writes a road-wheel angle there. "
                         "'kappa' is the LEGACY unconverted reading and carries "
                         "the C-REFCV3-ARM-SAME-DEFECT over-rotation; it affects "
                         "`ha` only (`os` has no control channel, `ha0` is zero).")
    ap.add_argument("--nav-shuffle-seed", type=int, default=0)
    ap.add_argument("--no-navshuf", action="store_true",
                    help="skip the nav-shuffle arm (⛔ then the record is NOT "
                         "admissible for any nav-conditioned claim)")
    ap.add_argument("--no-navzero", action="store_true",
                    help="skip the nav-ZERO trajectory arm (⛔ BACKLOG R39 binds "
                         "every nav-conditioned claim to carry it BESIDE the "
                         "nav-shuffle arm: a shuffle preserves the nav marginal "
                         "and breaks only the pairing, a zero removes the "
                         "signal, and the zero is what DEPLOYMENT looks like)")
    ap.add_argument("--with-navzero", action="store_true",
                    help="force the nav_cmd=None head conditioning even when the "
                         "os_navzero ARM is skipped (implied by the arm)")
    ap.add_argument("--with-navflip", action="store_true",
                    help="ALSO roll the nav-FLIP arm (left<->right on every "
                         "window): the sharpest nav intervention for the "
                         "nav-compliance metric (taniteval.nav_compliance). "
                         "Optional; shuffle + zero remain the REQUIRED controls.")
    ap.add_argument("--with-navpred", action="store_true",
                    help="ALSO roll the nav-PREDICTED arm `os_navpred`: the nav "
                         "token is the model's OWN route-head argmax, read off "
                         "the nav_cmd=None forward and mapped through the "
                         "imported refb_labels._ROUTE_TO_NAV. Requires the "
                         "nav-zero forward (it supplies the token), so it is "
                         "incompatible with --no-navzero.")
    ap.add_argument("--with-oracle-sel", action="store_true",
                    help="ALSO bank the T0 `oracle_sel` ceiling arm (the "
                         "a_star / GT-nearest anchor). Never compared to T1.")
    # ---- ⭐⭐ the eval-time ablations (PREREG_REFCV4B_HIERARCHY_EVAL.md §3) --
    ap.add_argument("--ablate", nargs="*", default=[], metavar="NAME",
                    choices=sorted(ABLATIONS),
                    help="eval-time ablation(s) to apply to the loaded model "
                         "before the roll; the rest of the forward stays "
                         "bit-identical. One regime per --dump-dir. Choices: "
                         + ", ".join(f"{k} ({v['prereg_arm']})"
                                     for k, v in sorted(ABLATIONS.items())))
    ap.add_argument("--ablate-frames", action="store_true",
                    help="⭐⭐ THE DELIBERATE REGRESSION (= --ablate "
                         "frames_blind), given its own flag because the panel's "
                         "VALIDITY rests on it: a gate that has never been "
                         "shown to FAIL an image-blind arm certifies nothing. "
                         "Every frame becomes its own scalar mean; ha / ha0 / "
                         "ha0_ext must come back bit-identical.")
    ap.add_argument("--gstr-bank", default=None, metavar="FULL_DUMP_DIR",
                    help="a banked FULL dump on the SAME episodes and window "
                         "grid, read for `gstr_nav_true`; REQUIRED by --ablate "
                         "gstr_shuffle, whose registered mechanism permutes "
                         "g_str ACROSS WINDOWS (this harness's batch rows are "
                         "the nav conditionings of ONE window)")
    ap.add_argument("--gstr-shuffle-seed", type=int, default=0,
                    help="seed for the g_str window permutation")
    ap.add_argument("--allow-nonstrict", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", default="",
                    help="extra tier stamps name=T0|T1, merged over this tool's "
                         "own ARM_TIERS and t1_eval.DEFAULT_TIERS")
    ap.add_argument("--lead-block", default=None,
                    help="per-frame B1 lead block for LONGITUDINAL "
                         f"distance-keeping; default = {ra.LEAD_BLOCK_DEFAULT} "
                         f"when it exists")
    ap.add_argument("--no-lead-block", action="store_true",
                    help="analyse WITHOUT a lead block (distance-keeping stays "
                         "REFUSED with its reason — a WORK ITEM, not a pass)")
    a = ap.parse_args(argv)

    lead_path = None
    if not a.no_lead_block:
        if a.lead_block and not os.path.exists(a.lead_block):
            sys.exit(f"--lead-block {a.lead_block} does not exist")
        lead_path = a.lead_block or (ra.LEAD_BLOCK_DEFAULT
                                     if os.path.exists(ra.LEAD_BLOCK_DEFAULT)
                                     else None)
        if lead_path is None:
            _p(f"[lead] no lead block (banked default absent: "
               f"{ra.LEAD_BLOCK_DEFAULT}); distance-keeping will be REFUSED with "
               f"its reason — a WORK ITEM, not a pass")
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
                         tiers=t1._parse_tiers(a.tiers), lead_block=lead_path,
                         labels=a.labels)
    rec.update({"arm": a.arm, "ckpt": a.ckpt, "dump_dir": dump_dir,
                "mode": "analyze-only" if a.analyze_only else "rollout+analyze",
                "lead_block": lead_path,
                "_unverified": _UNVERIFIED_ON_REAL_CKPT})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    _p(f"[out] {a.out}")
    r = rec.get("refcv3", {})
    dkb = r.get("distance_keeping") or {}
    for arm, blk in rec["arms"].items():
        ivl = blk["intervals"]["metrics"]
        ade = ivl.get("ade_dense_m", {})
        dk = (dkb.get("per_arm") or {}).get(arm) or {}
        _p(f"  {arm:12s} tier={blk['tier']}"
           f"{'*' if arm.startswith('os') else ' '}  ADE={ade.get('mean')} "
           f"[{ade.get('lo')}, {ade.get('hi')}]  families_unavailable="
           f"{blk['four_families']['_families_unavailable']}  "
           f"distance_keeping={dk.get('status') or dkb.get('status')}")
    _p(f"  * = tier stamped T1 with the ruling OPEN ({TIER_RULING['question'][:60]}…)")
    for nm, blk in (r.get("families_paired") or {}).items():
        ade = ((blk.get("families") or {}).get("ADE") or {}).get("ade_m") or {}
        _p(f"  {nm:28s} ADE delta={ade.get('delta')} "
           f"[{ade.get('lo')}, {ade.get('hi')}] separated={ade.get('separated')} "
           f"n_win={ade.get('n_windows')} n_ep={ade.get('n_episodes')}"
           + ("  ⛔ DEGENERATE (float64 resolution — arms effectively IDENTICAL)"
              if ade.get("degenerate") else ""))
    nn_ = r.get("nav_null") or {}
    if nn_.get("emitted"):
        _p("  nav controls: os_navshuf (pairing broken, marginal preserved) AND "
           "os_navzero (signal removed; E13 tactical+strategic OFF, core "
           "collapsed to 'follow') — BACKLOG R39 binds both; the nav-zero margin "
           "over ha0 is the DEPLOYMENT-relevant number")
    elif nn_:
        _p(f"  ⚠️ nav-zero arm ABSENT: {nn_.get('reason_absent')} — ⛔ BACKLOG "
           f"R39 requires it beside the shuffle for any nav-conditioned claim")
    sp = r.get("selection_profile") or {}
    if sp.get("degenerate"):
        _p(f"  ⛔ the selection is CONSTANT on {sp['modal_frac']:.2%} of windows "
           f"— every family row above is a property of anchor "
           f"#{sp['modal_anchor']}, not of scene understanding")
    _p(f"  ABSENT: ol — {ABSENT_ARMS['ol']['reason'][:90]}…")
    s = r.get("strategic") or {}
    _p(f"  strategic nav_valid_frac={s.get('nav_valid_frac')} "
       f"n_labeled={s.get('n_route_labeled')} "
       f"changed_subset_n={s.get('n_changed_subset')}")
    asel = ((r.get("tactical_declared") or {}).get("anchor_selection") or {})
    _p(f"  tactical anchor_acc={(asel.get('anchor_acc') or {}).get('mean')} "
       f"(chance {asel.get('chance')}) n_distinct_selected="
       f"{asel.get('n_distinct_selected')}")


if __name__ == "__main__":
    main()
