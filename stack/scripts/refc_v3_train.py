"""REF-C v3 trainer — the goal-mediated hierarchy on the supervised arm.

Design: ``TanitAD Research Lab/Architecture & Inference/Research/
2026-08-18-refc-v3-design/REFC_V3_DESIGN.md`` (edge list E1..E12).
Experiment: ``PREREG_REFC_V3.md`` (E-V3DOM-1 — arms v3-H / v3-F, 3 seeds,
both outcomes committed). ⛔ Registered BEFORE any training step; do not launch
while Thor trains.

WHAT THIS SCRIPT IS AND IS NOT
============================================================================
A THIN COMPOSITION over ``refc_train.py``'s measured machinery — the same
fail-loud dataset stack (``RouteV21Dataset``: v2.1 labels, the measured label
choice), the same optimizer convention (Adam lr 1e-4 / warmup / cosine — the
arm's point), the same loss weights for every SHARED surface (imported, not
copied, so the two trainers cannot drift apart silently). What is NEW here and
only here:

  * ``V3Dataset`` — the PARITY-PRESERVING 6 s horizon. ``max_horizon`` stays 20
    for window ENUMERATION (``_contract.py:120`` re-selects windows otherwise —
    the canonical 406,099 must stay bit-identical), and steps 21..60 arrive by
    CLAMP + per-step validity mask. Tactical goal labels via
    ``refb_labels.goal_tac_targets`` (E4.1 layout, clamp+valid by contract).
  * ``compute_losses_v3`` — masked trajectory/assignment losses over the 8-slot
    horizon (exact-zero gradient at invalid slots, pinned by
    ``tests/test_refc_v3.py``), the factored tactical CE on BOTH decision
    surfaces (core aux — identical in both arms — and the z_tac heads on the H
    arm), the masked goal losses, and the survivor-set selection CE.
  * ``--arm hier|flat`` — the dominance pair. The config delta between the two
    is DERIVED and REFUSED if it is not exactly the registered lever set
    (C122's rule, enforced at build, not documented).
  * ``--preflight`` — builds everything, pins the delta, runs the C115
    freeze-history gate and the E11 mini intervention audit, runs one synthetic
    forward+loss, and EXITS. The launch line runs this first; a flag that
    parses and then dies mid-run is the class that cost ~3.1 GPU-days.
  * ``--synth-episodes N`` — CI-only synthetic corpus so the smoke test runs
    end-to-end with zero data. REFUSED together with ``--data-root``.
  * ``--u8-batches`` (2026-09-02, next-relaunch arm; default OFF) — the
    DataLoader ships ``frames``/``future_frames`` as uint8 and
    :func:`frames_to_device` applies the contract's ``/255`` on the device.
    In-flight batches — the ``/dev/shm`` cost that pinned refcv3's 50 GB
    cgroup at its cap: 6 workers x 3.30 GB fp32 = 19.8 GB, MEASURED — drop
    4x; the loss is bit-identical (``tests/test_refc_v3_u8_batches.py``).

Done-marker discipline: on completion this trainer writes ``summary.json`` with
``{"done": true}`` IN THE SAME RUN — the v5f supervisor resurrection (a
finished run relaunched for 2 days) is the reason.
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import inspect as _inspect
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The preflight's own banners carry ⛔/✅. On a cp1252 console (the Windows dev
# box, where a preflight is MEANT to be run before spending a GPU day) printing
# them raises UnicodeEncodeError — MEASURED: the run below died on the "✅ PASS"
# line AFTER every check had passed, so a PASSING preflight exited non-zero with
# a traceback. A gate whose success path crashes is a false negative.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # already utf-8, or not a TTY
        pass

import refb_labels  # noqa: E402
from refc_train import (  # noqa: E402  — SHARED surfaces, imported not copied
    ANCHOR_CLS_WEIGHT, LAT_WEIGHT, LAW_AHEAD, LAW_WEIGHT, LON_WEIGHT,
    ROUTE_WEIGHT, TRAJ_WEIGHT, RouteV21Dataset, lan_dataset_class,
)
from refb_train import load_cached_episodes  # noqa: E402

from tanitad.data.lan import LanConfig as DataLanConfig  # noqa: E402

from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_tactical as tac  # noqa: E402
from tanitad.data import v7_labels as v7l
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import goal_point as gpm  # noqa: E402  — E15 (GP-2)
from tanitad.refs import max_speed_input as msi  # noqa: E402  — E16
from tanitad import effective_weights as _ew  # noqa: E402
from tanitad.refs import tac_goal_head as _tac_goal_head  # noqa: E402
# refcv6 §4/§5 — the behaviour decoder and the 4-way set-speed. ⛔ Imported at
# module scope, beside the other seam modules, so a missing module fails at
# IMPORT rather than after a rollout has already been paid for (the
# `t1_eval.py` trap: both arms, 40 episodes, 6,844 windows each, then a dead
# `from taniteval import selgap` in `analyze()`).
from tanitad.refs import refcv6_tactical as v6tac  # noqa: E402
from tanitad.refs import refcv6_max_speed as v6ms  # noqa: E402
from dataclasses import replace as _dc_replace  # noqa: E402
from tanitad.models import vocab_v7  # noqa: E402
from tanitad.refs import refb  # noqa: E402
from tanitad.refs import refc_agents as _refc_agents  # noqa: E402
from tanitad.refs import refc_bev_aux as _refc_bev_aux  # noqa: E402  (WP-D)
from tanitad.data import bev_aux as _bev_aux  # noqa: E402  (WP-D target)
from tanitad.refs import refc_wp_index as _refc_wp_index  # noqa: E402  (WP-B)
from tanitad.models import kinematic as kin  # noqa: E402
from tanitad.models import refcv6_diffusion as _rv6  # noqa: E402  (refcv6 §3)
from tanitad.train import grad_conflict as _gcf  # noqa: E402  (refcv6 §6)
from tanitad.models import agent_slots as _agent_slots  # noqa: E402
# --- refcv6 §2/§6: the perception branch (map + 3-D boxes) ------------------ #
# ⛔ IMPORTED UNCONDITIONALLY, USED ONLY BEHIND A POSITIVE WEIGHT. An
# analysis-time import that fails AFTER the rollout destroys a run whose
# compute is already paid for (`t1_eval.py`, 2026-08-11: both arms, 40
# episodes, ~11 min/arm, then `ImportError` in `analyze()`); at module scope it
# fails in 2 seconds instead.
from tanitad.data import perception_targets as _perception_targets  # noqa: E402
from tanitad.data import semantic_map_gt as _sem_map  # noqa: E402
from tanitad.data import agent_cuboid_gt as _agent_cuboid  # noqa: E402
from tanitad.models import box3d_head as _box3d_head  # noqa: E402
from tanitad.models import refcv6_perception_branch as _perc  # noqa: E402
import numpy as _np  # noqa: E402

# --- v3-only loss weights (everything shared is imported above) --------------
#: tactical goal regression (E8) — sized with the route/maneuver aux family;
#: the goal is a 12-number readout, not a second trajectory head.
GOAL_TAC_WEIGHT = 0.5
#: strategic goal (E3) — the ROUTE aux family's weight, deliberately: it is the
#: same kind of signal (leak-guarded LAN label) on a different head.
GOAL_STR_WEIGHT = ROUTE_WEIGHT
#: survivor-set selection CE (E9) — parallel to ANCHOR_CLS_WEIGHT: it is the
#: same "which candidate" question asked of the blended score.
SEL_V3_WEIGHT = 1.0

# ---- refcv5 weights (all default 0.0 -> the loss is bit-unchanged) --------
# ⛔ EVERY ONE OF THESE IS A DECLARED DECISION, NOT A DEFAULT, and every one
# defaults to 0.0 so that ADDING the seams to the code cannot change a run that
# does not ask for them. `--w-agent` / `--w-u0` are the switches.
#
# ⚠️ The units differ per term BY CONSTRUCTION (metres, nats, m/s^2), which is
# why they are named and stamped rather than folded into one number -- the same
# rule `SLOT_LOSS_W` and `V6LossWeights.w_select` carry.
AGENT_WEIGHT_DEFAULT = 0.0        # WP-6: the GT-supervised detection set loss
U0_WEIGHT_DEFAULT = 0.0           # WP-4: the x0 loss, in CONTROL space

# ---- ⭐⭐ E15 (GP-2, 2026-09-06): the METRIC GOAL POINT ---------------------
#: Smooth-L1 on the NORMALISED goal point (`goal_point.goal_point_loss`).
#: 1.0 is the PRE-REGISTERED launch value (`…/2026-09-06-goal-point/PREREG.md`
#: §9, `--goal-point-w 1.0`). The default here is **0.0** for the same reason
#: every refcv5 weight is: ADDING the seam to the code must not change a run
#: that does not ask for it, and the live 40 k refcv5 resumes through this file.
#: ⛔ `--goal-point-inject` with `--goal-point-w 0` is REFUSED (see
#: `_check_goal_point_args`): it builds a head, stamps the edge, and trains it
#: on nothing — the `--w-agent 0` failure, which manufactures "the goal point
#: does not help" out of a missing loss rather than out of evidence.
GOAL_POINT_WEIGHT_DEFAULT = 0.0

# ---- ⭐ M17 (2026-09-05): the detection query budget ----------------------
#: Detection queries. **100**, ruled by the Master Mind (`Decisions/
#: 2026-09-05-mm-decisions.md` §M17) after the train-corpus density was
#: MEASURED. The previous 32 was set from **val40** (max 24) and is REFUTED on
#: train: over 433,040 frames / 12,122,129 boxes / 2,308 clips the in-field ∩
#: decode-box count has mean **4.39**, p99 **30**, **max 94**, so N = 32 drops
#: **41,362 boxes (2.18 %) on 3,250 frames (0.75 %)** and the **nearest
#: sacrificed target sits at 13.1 m**.
#:
#: ⛔ ``match_slots`` keeps the **NEAREST** N, so a drop is by construction the
#: CLOSEST thing the head failed to see — not a long tail of distant clutter.
#: 13.1 m is inside the braking envelope at any urban speed, and N = 64 does
#: not fix it either (nearest sacrifice 33.9 m, still inside ≈ 43 m of
#: comfortable braking at 15 m/s). 94 is a MAX OVER A SAMPLE, not a bound, so
#: the setting carries headroom over it; 100 is also DETR's ordinary budget
#: (~100 queries against ~7 objects/image) against our mean of 4.39, i.e. 32
#: was the anomaly and surplus "no object" queries are the design working.
#:
#: MEASURED by two independent implementations (``measure_train_agent_density
#: .py`` and a numpy-free ``indep_max.py``), agreeing box for box:
#: `…/Data Engineering/Research/2026-09-05-agent-join-into-batch/RESULT.md` §P3.
AGENT_QUERIES_DEFAULT = 100
#: the registered dominance lever set — build REFUSES any other delta (C122).
REGISTERED_DELTA_KEYS = {"hier", "core.graft_target_latent"}
#: ⭐ v4 adds its own lever set on top of the hier/flat pair. Both arms of a
#: v4 run carry the SAME ego pins, so the hier-vs-flat delta is UNCHANGED - the
#: v4 levers are checked against refcv3 separately (`REGISTERED_DELTA_KEYS_V4`,
#: `tests/test_refc_v4.py::test_registered_delta_is_pinned`), which keeps the
#: two questions ("what does the hierarchy cost?" and "what does v4 change?")
#: from being answered by one confounded diff.

MILESTONES = (5000, 15000, 20000, 30000)
MAX_H_EXT = max(v3.V3_HORIZONS)            # 60 — fetched by clamp, never enum

# --- --nav-from-v7: the nav SOURCE switch (E-ARCH-NAVSRC-1, 2026-09-02) ------
#: v7.2 nav token -> legacy ``refb.NAV_COMMANDS`` name, POSITION-pinned: the
#: model's nav one-hot / embedding rows are ordered by ``NAV_COMMANDS``
#: (refc.py:2024, refc_v3.py:164) while the v7.2 ids enumerate
#: ``vocab_v7.NAV_COMMAND_TOKENS``. This is the SAME mapping
#: ``refav1_loader._NAV_TOKEN_TO_LEGACY`` pins (equality asserted by
#: tests/test_refc_v3_nav_from_v7.py). It is a local copy ON PURPOSE: the
#: trainer must not acquire an import-time dependency on a loader that is
#: shipped to pods separately; :func:`assert_nav_token_alignment` re-checks
#: the position pin at dataset init and in the preflight, so drift is loud.
NAV_TOKEN_TO_LEGACY = {"NAV_FOLLOW_ROAD": "follow", "NAV_TURN_L": "left",
                       "NAV_TURN_R": "right"}
#: the config.json stamps — an arm must be identifiable from its own artifacts
NAV_FROM_V7_DERIVATION = ("v7.2 nav_command token (oracle, provenance "
                          "ego-future; allow_oracle_nav=True)")
NAV_V1_DERIVATION = "refb_labels.nav_command (v1, unchanged)"

#: ⭐⭐ E16 — THE MAX-SPEED CHANNEL'S PROVENANCE STAMP, and the PI's binding
#: condition on using it (2026-09-10): *"stick to the labels we created in the
#: data set with the logic of minimal speed etc..."*
#:
#: ⛔ THE REQUIREMENT IS DECLARATION, NOT REFUSAL. This channel's value IS
#: `g_tac.goals.SPEED_BAND.v_hi_ms` = the max of the ego's OWN realised speed
#: over [t0+2 s, +6 s] — confirmed at three independent source sites
#: (`v7_labels.py:325-330`, this file's `--max-speed-input` help,
#: `max_speed_input.py:367`) — and quantization does not launder it (bin + `v0`
#: recovers R^2 0.9702 of the raw ego future). The PI has decided to use it
#: anyway, and that decision is CONSISTENT with the programme's existing
#: position rather than a new exception: `NAV_FROM_V7_DERIVATION` above is the
#: same class of signal, an INPUT simulating the vehicle's nav system. Max
#: speed stands in for a speed-limit service the same way.
#:
#: ⇒ So the arm must be identifiable from its own artifacts, and
#: :func:`_assert_speed_max_stamp` REFUSES TO START a run whose config would not
#: carry this string. ⛔ NO CAPABILITY CLAIM MAY BE CREDITED TO THIS CHANNEL
#: WITHOUT THE STAMP BESIDE IT.
SPEED_MAX_DERIVATION = (
    "v7.2 g_tac.goals.SPEED_BAND.v_hi_ms (oracle, provenance ego-future: max "
    "of the ego's OWN REALISED speed over [t0+2 s, +6 s]; floor from "
    ".v_lo_ms; allow_oracle_nav=True). INPUT standing in for a speed-limit "
    "service, never a training signal.")

#: ⛔ The tokens :func:`_assert_speed_max_stamp` requires to be PRESENT in the
#: stamp. Written as LITERALS, never as an expression over the constant above —
#: a check derived from the value it checks is green forever, which is exactly
#: how a label builder verified its buckets against its own rounded ladder and
#: passed while 57.5 % of the corpus was wrong.
SPEED_MAX_STAMP_REQUIRED = ("oracle", "ego-future", "SPEED_BAND.v_hi_ms",
                            "[t0+2 s, +6 s]")


def _assert_speed_max_stamp(cfg_dict: dict, args) -> None:
    """⛔ REFUSE A ``--max-speed-input`` RUN WHOSE CONFIG DOES NOT DECLARE THE
    CHANNEL'S PROVENANCE. Called before ``config.json`` is written.

    ⭐ WHY A REFUSAL AND NOT A DEFAULT. The PI authorised an **ego-future**
    input on the axis owning 88.7 % of the oracle gap. That is defensible
    exactly as long as every artifact says so — the moment a run's record omits
    it, a later reader has an arm that looks like a clean capability result and
    no way to see what fed it. A stamp that can be silently dropped is not a
    stamp.

    ⛔ It asserts on the CONTENT of the string, not on the presence of a key: a
    stamp that said "max speed: on" would satisfy a presence check and tell a
    reader nothing. The four required tokens are written as literals in
    :data:`SPEED_MAX_STAMP_REQUIRED`.

    ⚠️ Off is off: a run without the flag must NOT carry the stamp, or every
    banked arm would read as max-speed-conditioned.
    """
    on = bool(getattr(args, "max_speed_input", False))
    stamp = cfg_dict.get("speed_max_derivation")
    if not on:
        if stamp is not None:
            raise SystemExit(
                "[v3] ⛔ config carries `speed_max_derivation` but "
                "--max-speed-input is OFF. A run that did not feed the "
                "ceiling must not be stamped as one — that is the mirror of "
                "the missing-stamp failure and it manufactures a "
                "max-speed-conditioned arm out of a control.")
        return
    if not isinstance(stamp, str) or not stamp.strip():
        raise SystemExit(
            "[v3] ⛔ --max-speed-input is ON but this run's config.json would "
            "carry no `speed_max_derivation`. The channel's value is the "
            "ego's OWN FUTURE speed over the horizon being scored; the PI "
            "authorised it as a declared oracle INPUT, and the declaration is "
            "the condition. Refusing to start rather than banking an arm "
            "whose record cannot say what fed it.")
    missing = [t for t in SPEED_MAX_STAMP_REQUIRED if t not in stamp]
    if missing:
        raise SystemExit(
            f"[v3] ⛔ --max-speed-input: `speed_max_derivation` is present but "
            f"does not declare {missing}. The stamp exists so a reader who "
            f"opens config.json in ISOLATION learns the source field, the "
            f"window and the word `oracle` without going to find the code. A "
            f"stamp missing any of those is a key, not a declaration. Got: "
            f"{stamp!r}")


def _pin_trainer_cfg(cfg: v3.RefCV3Config, args) -> v3.RefCV3Config:
    """The trainer's OWN pins on a freshly built config (both arms, all paths).

    * ``tac_vocab_version = "kin3"`` — THE documented contract this trainer
      never wired (found 2026-09-01): ``RefCV3Config``'s field comment says
      *"the kinematic trainer passes 'kin3' explicitly because its labels are
      the 3x3 kinematic classes"*, and the E6 comment says *"The trainer PINS
      its version"* — but nothing pinned it, so the v7.0 default (PI mandate
      2026-08-27) built 8-wide z_tac heads that ``compute_losses_v3``
      supervised with 3-class kinematic labels. That is EXACTLY the *"train
      silently WRONG classes"* case the width refusal names, invisible to it
      because the refusal checked only the CORE's head — and
      ``derive_man5_logprobs`` (a [B,3]x[B,3] contract) then read v7.0's
      LANE_CHANGE_L/R slots as turn_left/right into the H19 prior, an improper
      AND mis-labelled 5-way. MEASURED before the pin: preflight PASSED with
      ``tac_heads 14,364`` and ``lat_tac``/``lon_tac`` init-CE at ln(8).
      The v7.0 head is the go-forward space and NEEDS v7 labels
      (``SPEC_V7_LABEL_TRAINER_WIRING.md`` — label artifacts not yet
      deliverable), so a flag would be a dead switch today; pin, don't offer.
    * ``--image-hw`` — build the encoder at the CORPUS's geometry (e.g.
      256 640 for the B1 ``*.v2ep.pt`` cache). Param count is UNCHANGED (the
      trunk is fully convolutional; feat_dim = base_width*8 regardless), so
      the registered capacity ledger still holds; only compute changes.
      The delta gate is unaffected: both arms are pinned identically, and
      ``config_delta`` is derived from configs built through this same helper.
    """
    # ⭐ The vocabulary follows THE LABELS, not a hardcode. With --v7-labels
    # the released v7.2 tactical vocabulary (8x8) supervises the heads (PI
    # 2026-09-02, MANDATORY); otherwise the kinematic 3x3 derivation does, and
    # pinning it here is what stopped an 8-wide build training on 3 classes.
    cfg.tac_vocab_version = ("v7.0" if getattr(args, "v7_labels", None)
                             else "kin3")
    # ⭐ refcv6 §2 — the TRUNK pin goes FIRST, because the `--image-hw`
    # rebuild below reconstructs `CNNEncoderConfig` field by field and would
    # otherwise silently drop it (the same class of loss as the `rebuild_config`
    # stamp that lived only on the Namespace).
    _trunk = str(getattr(args, "trunk", "refc"))
    cfg.core.encoder.trunk = _trunk
    cfg.core.encoder.trunk_name = str(getattr(args, "trunk_name",
                                              "resnet34.a1_in1k"))
    cfg.core.encoder.trunk_mode = str(getattr(args, "trunk_mode", "shared"))
    cfg.core.encoder.trunk_fuse = str(getattr(args, "trunk_fuse", "concat1x1"))
    cfg.core.encoder.trunk_fuse_identity = not bool(
        getattr(args, "trunk_fuse_plain_init", False))
    # ⛔ `--no-trunk-pretrained` -- THE KNOCKOUT ARM'S ONLY ROUTE FROM ARGV.
    # `None` (the flag's default) leaves the dataclass field ALONE, so every
    # banked arm and every caller that builds a config without this Namespace
    # is bit-identical; only an explicit flag writes it. It is pinned onto the
    # CONFIG for the same reason the four lines above are: `rebuild_config`
    # reconstructs `CNNEncoderConfig` field by field.
    _tpre = getattr(args, "trunk_pretrained", None)
    if _tpre is not None:
        cfg.core.encoder.trunk_pretrained = bool(_tpre)
    _tic = int(getattr(args, "trunk_in_channels", 0) or 0)
    if _tic:
        cfg.core.encoder.in_channels = _tic
    # ---- refcv6 §2b: the ego-history encoder onto the CORE config -------- #
    # ⛔ Pinned onto the CONFIG, never only onto `args` — `rebuild_config`
    # rebuilds through this helper, so a block on the Namespace would be lost
    # on every roll.
    if bool(getattr(args, "ego_history", False)):
        from tanitad.models.ego_history import EgoHistoryConfig
        cfg.core.ego_history = EgoHistoryConfig(
            enable=True, steps=int(cfg.core.window),
            hidden=int(getattr(args, "ego_history_hidden", 64)),
            out_dim=int(getattr(args, "ego_history_out", 32)),
            kind=str(getattr(args, "ego_history_kind", "gru")))
    # ---- refcv6 §3: the F1..F9 block onto the CORE config ----------------- #
    # ⛔ Pinned onto the CONFIG, never only onto `args`: `rebuild_config`
    # rebuilds through this same helper, so a block living on the Namespace
    # would be LOST on every roll — the identical failure the max-speed mode
    # and `u0_absent_under_ddim` are pinned here to avoid.
    _rv6_flags = refcv6_flags_from_args(args)
    cfg.core.decoder.refcv6 = _rv6_flags
    if _rv6_flags is not None and _rv6_flags.f6_w_u0_zero:
        # F6 IS `--w-u0 0`, and it carries its own acknowledgement so the
        # record reads "refcv6 F6" rather than "someone bypassed a guard".
        args.w_u0 = 0.0
        args.ack_ddim_no_u0 = True
    if args.image_hw:
        h, w = (int(args.image_hw[0]), int(args.image_hw[1]))
        enc = cfg.core.encoder
        # ⛔⛔ D-REFCV6-IMAGEHW-DROPS-TRUNK-FIELDS (MEASURED 2026-09-17).
        # This rebuild listed `trunk`, `trunk_pretrained` and
        # `trunk_imagenet_norm` and SILENTLY DROPPED FOUR MORE, so with
        # `--image-hw` every one of them reverted to the dataclass default:
        #   --trunk-name resnet34.a1_in1k  -> built resnet101.a1_in1k
        #   --trunk-mode inflate           -> built shared
        #   --trunk-fuse last              -> built concat1x1
        #   --trunk-fuse-plain-init        -> built the IDENTITY init
        # ⚠️ The last two are CONTROL ARMS -- `last` is the single-frame
        # control and the plain init is the deliberate regression -- so a run
        # that asked for either got the primary arm while `config.json`'s
        # `seams` block recorded the control. And `--image-hw 256 1024` is the
        # PI's own 2026-09-16 geometry, so EVERY refcv6 run on the 1024 cache
        # took this path. MEASURED on the first live perception run: argv said
        # resnet34, `fmap_s16_channels` read 1024 = resnet101.
        # ⭐ The comment 30 lines up already names the mechanism -- *"the
        # --image-hw rebuild reconstructs CNNEncoderConfig field by field and
        # would otherwise silently drop it"* -- and the fix had been applied to
        # ONE field. ⇒ Replaced by `dataclasses.replace`, which carries EVERY
        # field by construction and cannot fall behind a new one.
        cfg.core.encoder = _dc.replace(
            enc, image_size=h, image_width=None if w == h else w)
    # ---- ⭐⭐ REF-C v4 pins (E11' + E14 + X15) ------------------------
    # Applied to BOTH arms identically, exactly like every other pin here, so
    # `config_delta` stays the derived instrument it is: the v4 lever set is
    # registered in REGISTERED_DELTA_KEYS_V4 and checked against the SAME
    # helper's output, never against a hand-written list of intentions.
    # ⭐⭐ STRATEGIC BYPASS. Pinned onto the CORE config so `refc.py` and
    # `refc_v3.py` read ONE field, and applied to BOTH arms identically so
    # `config_delta` (hier vs flat) stays the derived instrument it is -- the
    # flag is equal on both sides and therefore never enters the delta.
    cfg.core.no_strategic = bool(getattr(args, "no_strategic", False))
    if getattr(args, "ego_state_inject", False):
        cfg.ego_state_inject = True
        cfg.core.ego_valid_channel = True     # precondition, not an option
    if getattr(args, "echo_base", False):
        cfg.echo_base = True
    if getattr(args, "ego_valid_channel", False):
        cfg.core.ego_valid_channel = True
    if getattr(args, "ego_dropout", None) is not None:
        cfg.core.ego_dropout = float(args.ego_dropout)
    # ---- ⭐⭐ E15 (GP-2): THE PREDICTED METRIC GOAL POINT ------------------
    # Applied to BOTH arms identically, exactly like every other pin here, so
    # `config_delta` (hier vs flat) stays the derived instrument it is.
    #
    # ⛔⛔ `--goal-point-inject` TURNS `nav_inject` OFF. That is the
    # PRE-REGISTRATION, not a side effect: PREREG §2 defines `gp_cond` as
    # "vs `base`: the TYPE of the route conditioning signal — E13's
    # `Embedding(4)->Linear->add` REPLACED by `GoalPointConditioning`
    # (`nav_inject=False`)". Running both would make the arm a two-variable
    # experiment whose result is non-attributable — the `--v2` conflation
    # failure (ten levers on two axes, result non-attributable). The swap is
    # stamped in `config.json` under `goal_point.replaces_nav_inject`, so a
    # reader never has to know this line exists.
    if getattr(args, "goal_point_inject", False):
        cfg.goal_point_inject = True
        # frozen dataclass -> `replace`, never mutation: the config object is
        # also the thing `goal_point_provenance()` is derived from, and a
        # provenance derived from a half-mutated config is worse than none.
        cfg.goal_point_cfg = _dc_replace(
            cfg.goal_point_cfg, t_goal_s=float(getattr(args, "goal_point_t",
                                                       4.0)))
        cfg.nav_inject = False
        # DERIVED HERE, at CONFIG time, not as a constructor side effect.
        # `RefCV3Model.__init__` derives the same two values from the same two
        # inputs (idempotent), but the RECORD is written from the CONFIG, and a
        # record field that only exists once a model has been built stamps
        # `gp_slot: -1` for any consumer that stamps first -- a run whose
        # artifact says it compared the goal against slot -1 while the weights
        # used slot 5. MEASURED by tests/test_goal_point_trainer_flags.py.
        cfg.core.gp_slot = gpm.goal_slot_index(
            cfg.goal_point_cfg.t_goal_s, cfg.core.trajectory.horizons)
        cfg.core.gp_scale_m = float(cfg.goal_point_cfg.range_norm_m)
        if getattr(args, "goal_point_geo_prior", False):
            cfg.core.graft_gp_point = True
    # ---- ⛔ S2 REACH CLAMP, RE-DERIVED FOR THE HORIZON ACTUALLY PLANNED OVER
    # `refc.py:652` DERIVES `horizon_s` from `max(trajectory.horizons)`, so at
    # V3_HORIZONS the band is a*6.0, not a*2.0. The inherited `sel_accel_max =
    # 2.5` therefore opens it to +-15.0 m/s on a corpus whose v0 mean is
    # 5.24 m/s -- near-vacuous. MEASURED 2026-09-04 on the refcv4 anchor
    # vocabulary over 19,602 held-out eval / 635,331 train windows
    # (`refc_anchors_6s_b1train_128.pt.json`, `clamp6s.json`):
    #     a=2.5 kills 18.02 %   a=2.0 kills 26.23 %   a=1.5 kills 38.18 %
    # and the GT-deletion rate -- the criterion that actually binds, because a
    # band that removes the trajectory the ego FLEW is wrong, not conservative
    # -- is eval 0.000 % at a>=1.5 and train 0.007 % at a=2.0 vs 0.048 % at
    # a=1.5. dADE on the survivors is +0.00000 m at every a>=1.25, so the
    # "inert on ADE" property the 2 s statistic claimed does still hold here.
    # ⚠️ The 72.08 %/77.28 % figures in `refc_v3.py` are 2 s statistics and are
    # NOT reproduced at 6 s -- they must never be quoted for this band.
    # Applied to BOTH arms, so `config_delta` (hier vs flat) is unchanged.
    if getattr(args, "sel_accel_max", None) is not None:
        cfg.core.sel_accel_max = float(args.sel_accel_max)
    # ---- refcv5 P14: THE SAMPLER MUST RANK THE FAN IT EMITTED ------------ #
    # ⛔ REFUSE THE HARMFUL HALF ON ITS OWN, BEFORE THE COMPUTE. `--sel-refined`
    # without `--sel-score-emitted` ranks by a score one denoising pass STALE,
    # and it is MEASURED 0.0259 m separated WORSE while flipping 29.82 % of
    # picks. `refc_train.py` already refuses the mirror-image split; this is the
    # same guard in the trainer that actually launches refcv5.
    _sel_ref = bool(getattr(args, "sel_refined", False))
    _sel_emit = bool(getattr(args, "sel_score_emitted", False))
    if _sel_ref and not _sel_emit:
        raise SystemExit(
            "[v3] ⛔ --sel-refined WITHOUT --sel-score-emitted ranks the fan by "
            "a score one pass STALE. MEASURED: 0.0259 m separated WORSE, "
            "29.82 % of picks flipped, BOTH minority recalls lowered. Pass "
            "--sel-score-emitted too, or neither.")
    if _sel_emit and not _sel_ref:
        raise SystemExit(
            "[v3] ⛔ --sel-score-emitted WITHOUT --sel-refined is INERT: the "
            "ranked score is `refined if sel.refined else conf`, so the "
            "emitted-fan pass would be computed and then DISCARDED -- a run "
            "that costs an extra decoder pass per step and changes nothing, "
            "while `sampler_ranks_the_fan` still stamps False. Pass both.")
    cfg.core.sel_refined = _sel_ref
    cfg.core.sel_score_emitted = _sel_emit
    cfg.core.sel_score_emitted_t = int(getattr(args, "sel_score_emitted_t", -1))
    # ⚠️ ON A SAMPLER ARM THE DEFAULT t IS THE WRONG ONE, AND IT IS SILENT.
    # `-1` means "continue the refinement loop's schedule", which is an
    # `nn.Embedding` index. The sampler conditions through the CONTINUOUS
    # `time_mlp` and its emitted fan is the FULLY DENOISED state, so the pass
    # that scores it must name t = 0. Corrected here rather than refused,
    # because the operator asking for both flags on a `ddim` arm can only mean
    # this -- and the correction is STAMPED so the record says who chose it.
    if (_sel_emit and str(getattr(args, "sampler", "none")) == "ddim"
            and int(getattr(args, "sel_score_emitted_t", -1)) < 0):
        cfg.core.sel_score_emitted_t = 0
        args.sel_score_emitted_t = 0
        args.sel_score_emitted_t_source = "auto-zero-on-ddim"
        print("[v3] --sel-score-emitted on a ddim arm: sel_score_emitted_t "
              "-1 -> 0 (the emitted fan IS the denoised state; -1 would index "
              "the refinement loop's embedding table instead)")
    # ⭐ refcv4-b — the v0-CONDITIONED vocabulary. Applied to BOTH arms, so
    # `config_delta` (hier vs flat) is unchanged and C122 still passes.
    if getattr(args, "n_anchors", None):
        # ⚠️ The vocabulary SIZE is a property of the built vocabulary, not of
        # the model family: a v0-conditioned (accel, curvature) product grid is
        # odd x odd, so it can never be exactly 128. Applied to BOTH arms.
        cfg.core.anchors.n_anchors = int(args.n_anchors)
    if getattr(args, "anchor_v0_conditioned", False):
        cfg.core.anchors.v0_conditioned = True
        cfg.core.anchors.ref_speed_ms = float(
            getattr(args, "anchor_ref_speed", 10.0))
        # ⛔ `--anchor-control-units` now DEFAULTS TO None: the artifact is the
        # authority on what its own `controls` column MEANS, and the flag is
        # an explicit OVERRIDE for a legacy file that declares nothing.
        # `_read_anchor_artifact` resolves the two BEFORE this pin runs and
        # writes the resolved value back into `args`; a None here means no
        # artifact was read (a fixed-path build), where the value is inert.
        # MEASURED 2026-09-04: the live refcv4b anchors.pt declares nothing,
        # and the same bytes read as curvature give 396 g at 36 m/s (104/117
        # over mu = 0.7) against the true 0.31 g (0/117) -- tanitad.refs.
        # anchor_meta carries the incident and the rule.
        cfg.core.anchors.control_units = str(
            getattr(args, "anchor_control_units", None) or "kappa")
        # The artifact's own derivation constants travel with it: a file
        # rolled under one kappa cap / speed floor must be re-rolled under the
        # SAME ones by the decoder, or the checkpoint-visible `anchors` buffer
        # and the per-window bank are two different vocabularies under one
        # name. Adopted here (both arms, so the hier/flat delta is unchanged)
        # and stamped into config.json by `_anchor_stamp`.
        _am = getattr(args, "_anchor_artifact_meta", None) or {}
        if _am.get("kappa_cap") is not None:
            cfg.core.anchors.kappa_cap = float(_am["kappa_cap"])
        if _am.get("alat_v_floor") is not None:
            cfg.core.anchors.alat_v_floor_ms = float(_am["alat_v_floor"])
    _pin_refcv5_seams(cfg, args)
    # ---- ⭐ E13b (D-GSTR-1 P3): the nav command's range and time -----------
    # ⛔ REFUSED WITHOUT ITS SUPPLIER AND WITHOUT ITS SEAM. `--nav-args`
    # without `--nav-from-v7` has no args to read (the values live on the
    # v7.2 record); with `--goal-point-inject` the nav path is switched OFF
    # entirely by the pre-registration above, so feeding args into it would
    # silently do nothing. Both are refused here, at config time, rather than
    # discovered as a flat result.
    if getattr(args, "nav_args", False):
        if not getattr(args, "nav_from_v7", False):
            raise SystemExit(
                "[v3] ⛔ --nav-args requires --nav-from-v7: `distance_m` / "
                "`time_s` live on the v7.2 nav_command record and there is "
                "no other supplier. Refusing rather than feeding zeros.")
        if not cfg.nav_inject:
            raise SystemExit(
                "[v3] ⛔ --nav-args with the nav path OFF (nav_inject=False, "
                "e.g. under --goal-point-inject) would be a silently inert "
                "flag. Refusing.")
        cfg.nav_args_inject = True
    # ---- ⭐⭐ E16: THE MAX-SPEED (map/nav posted-limit) INPUT ---------- #
    # PI 2026-09-01, reaffirmed 2026-09-06. OPT-IN, RECORDED IN ARGV.
    # ⛔ THE MODE IS PINNED ONTO THE CONFIG, NOT ONLY ONTO `args`, because
    # `refcv3_arm.rebuild_config` rebuilds through THIS helper: a mode that
    # lived only on the Namespace would be lost on every roll.
    # ⛔ A DEAD KNOB IS REFUSED, NOT IGNORED -- the class this trainer
    # already refuses four times (`--w-agent` under `--agents off`,
    # `--nav-args` without `--nav-from-v7`, both halves of the P14
    # selection split, `--tac-goal-tok-head` under kin3).
    _msi_on = bool(getattr(args, "max_speed_input", False))
    _msi_mode = str(getattr(args, "max_speed_mode", msi.DEFAULT_MODE))
    if not _msi_on and _msi_mode != msi.DEFAULT_MODE:
        raise SystemExit(
            f"[v3] ⛔ --max-speed-mode {_msi_mode!r} WITHOUT "
            f"--max-speed-input is INERT: no ceiling is fed, so the mode "
            f"selects the encoding of a channel that does not exist, and "
            f"config.json would stamp a mode the run never used. Pass "
            f"--max-speed-input too, or drop the mode.")
    if _msi_on:
        if args.arm != "hier":
            raise SystemExit(
                "[v3] ⛔ --max-speed-input on a FLAT arm: the ceiling's two "
                "injection sites (z_tac, ctx) exist only in the hierarchy, "
                "so the conditioner would be built and never read. Pass "
                "--arm hier, or drop --max-speed-input.")
        cfg.max_speed_input = True
        cfg.max_speed_cfg = msi.MaxSpeedConfig(enabled=True, mode=_msi_mode)
    # ---- ⛔ D-TACGOAL-1 / D-ROLL-1h: the tactical-goal SET head ----- #
    # OPT-IN, RECORDED IN ARGV. Building this head on the vocabulary alone
    # made refcv4b, three refcv3 checkpoints and the LIVE refcv5 run
    # unrollable, because its 11,286 params are absent from every recorded
    # `param_breakdown` and `cross_check_config` correctly refuses.
    # ⚠ The vocabulary REMAINS NECESSARY (refc_v3.py gates on
    # `_vv != "kin3" AND cfg.tac_goal_tok_head`); this only moves the
    # DECISION to an explicit lever. Asking for the head without the
    # vocabulary that has one is REFUSED here rather than silently
    # ignored -- the dead-flag class this trainer already refuses four
    # times (`--w-agent` under `--agents off`, `--nav-args` without
    # `--nav-from-v7`, and both halves of the P14 selection split).
    if getattr(args, "tac_goal_tok_head", False):
        if str(getattr(cfg, "tac_vocab_version", "")) == "kin3":
            raise SystemExit(
                "[v3] REFUSED: --tac-goal-tok-head with the kin3 "
                "vocabulary. kin3 is the 3x3 KINEMATIC derivation and "
                "has no tactical-goal token set, so refc_v3.py would "
                "build nothing and the flag would be silently inert. "
                "Pass --v7-labels (which pins tac_vocab_version=v7.0), "
                "or drop --tac-goal-tok-head.")
        cfg.tac_goal_tok_head = True
    _pin_refcv6_tactical(cfg, args)
    return cfg


def _pin_refcv6_tactical(cfg, args) -> None:
    """refcv6 §4/§5 — the behaviour decoder and the 4-way set-speed, from argv.

    ⛔⛔ THE REFUSAL THIS FUNCTION EXISTS FOR IS ``--tac-decoder-v6`` WITH A
    ZERO WEIGHT, and it is not symmetry-for-its-own-sake. MEASURED at tip
    837c308: the decoder builds 2,262,020 parameters (``d_bev`` 256), runs in
    the forward, writes ``cache["tacv6_*"]`` — and NOTHING read them, so every
    one of those parameters sat at ``grad_abs_sum`` EXACTLY 0. A run that
    built the head, converged, wrote a checkpoint and stamped
    ``tac_decoder_v6: true`` would read as *"the tactical layer does not
    help"*. That is the ``tac_goal_tok_head`` post-mortem (11,286 params, 0
    gradient, 40,284 steps) at 200x the scale, and the ``--conflict-detector
    on`` refusal one file down is the same rule in its established shape.

    ⛔ EVERY refusal here fires BEFORE ``config.json`` is written and before a
    batch is loaded, so a misconfigured arm costs seconds, not a GPU-day.
    """
    _w6 = float(getattr(args, "w_tac_v6", 0.0) or 0.0)
    _on6 = bool(getattr(args, "tac_decoder_v6", False))
    if _on6:
        if args.arm != "hier":
            raise SystemExit(
                "[v3] ⛔ --tac-decoder-v6 on a FLAT arm. The behaviour "
                "decoder's keys are the agent slots and its feeds enter the "
                "hierarchy's selection seam, both of which exist only under "
                "--arm hier, so the decoder would be built and never read. "
                "Pass --arm hier, or drop --tac-decoder-v6.")
        if str(getattr(cfg, "tac_vocab_version", "")) == "kin3":
            raise SystemExit(
                "[v3] ⛔ --tac-decoder-v6 with the kin3 vocabulary. kin3 is "
                "the 3x3 KINEMATIC derivation and has no 22-token tactical "
                "goal set, so the decoder's 22 behaviour queries would have "
                "no label to learn from. Pass --v7-labels (which pins "
                "tac_vocab_version=v7.0), or drop --tac-decoder-v6.")
        if bool(getattr(args, "tac_goal_tok_head", False)):
            raise SystemExit(
                "[v3] ⛔ --tac-goal-tok-head AND --tac-decoder-v6 both emit a "
                "22-token tactical goal posterior from the same labels. Two "
                "heads on one target is two experiments in one arm and the "
                "result is non-attributable. Pick one.")
        # ⛔⛔ THE CORE REFUSAL. A BUILT HEAD WITH NO LIVE WEIGHT.
        if _w6 <= 0.0:
            raise SystemExit(
                "[v3] ⛔ --tac-decoder-v6 with --w-tac-v6 %.6g: the behaviour "
                "decoder would be BUILT (2,262,020 params at d_bev 256, "
                "2,229,252 at d_bev 128), forward-run, stamped into "
                "config.json — and receive NO GRADIENT, because at a "
                "non-positive weight the term never enters the graph. That is "
                "exactly the defect this flag exists to close: "
                "`tac_goal_tok_head` took grad_abs_sum EXACTLY 0.0 for all "
                "40,284 steps of refcv5-v2 and nothing in the run record said "
                "so. Pass --w-tac-v6 > 0 (1.0 reproduces the spec's weights), "
                "or drop --tac-decoder-v6." % _w6)
        # ⭐⭐⭐ THE MAP HALF, UNBLOCKED BY PI RULING 2026-09-17 (R2/R3).
        # ⛔ The old refusal here was STRUCTURAL — no BEV token existed at the
        # instant the scene hook fired, because the BEV encoder ran AFTER the
        # core forward on the trainer's wrapper. That is no longer true: the
        # encoder now runs INSIDE the forward (`refc_v3.RefCV3Model._bev_hook`
        # -> `refc.py`'s `bev_hook` seam), so the tokens exist where the hook
        # reads them. What replaces it is a PRECONDITION guard, not an absence
        # of one — the failure the old refusal protected against (a decoder
        # declaring a 'bev' source it never receives) is still reachable, just
        # by a different route: asking for BEV keys with no BEV branch built.
        _dbev = int(getattr(args, "tac_decoder_d_bev", 0) or 0)
        if _dbev > 0:
            _wmap = float(getattr(args, "w_map", 0.0) or 0.0)
            if _wmap <= 0.0:
                raise SystemExit(
                    "[v3] ⛔ --tac-decoder-d-bev %d with --w-map %.6g. The BEV "
                    "tokens ARE the supervised map branch's features "
                    "(`BEVMapBranch.bev_feats`, pooled); with no live map "
                    "weight the lift and the BEV encoder are NOT BUILT "
                    "(`PerceptionBranchConfig.use_bev` is `w_map > 0`), so the "
                    "decoder would declare a 'bev' key/value source it never "
                    "receives — the arm would read as 'the map adds nothing to "
                    "behaviours' while never having had a map. ⇒ Pass --w-map "
                    "> 0 with --map-gt-root, or run agent-only "
                    "(--tac-decoder-d-bev 0) and SAY which arm you ran."
                    % (_dbev, _wmap))
            # ⛔ THE WIDTH IS DERIVED, NEVER TYPED. `BEVEncoderConfig.d_out`
            # is what the branch emits; a hand-set width that disagreed would
            # size `Linear(d_bev, d_model)` for a tensor that never arrives and
            # raise only at the first forward — after config.json was written.
            # Same family as C-ANCHOR-UNITS: a correct number in the wrong unit.
            _want = int(_perc.BEVEncoderConfig().d_out)
            if _dbev != _want:
                raise SystemExit(
                    "[v3] ⛔ --tac-decoder-d-bev %d but this build's BEV "
                    "encoder emits %d-wide tokens "
                    "(`BEVEncoderConfig.d_out`). The width is DERIVED from the "
                    "branch, not chosen: a mismatch sizes the decoder's "
                    "`Linear(d_bev, d_model)` for a tensor that never arrives. "
                    "Pass --tac-decoder-d-bev %d, or 0 for agent-only."
                    % (_dbev, _want, _want))
        _tdc = v6tac.TacticalDecoderConfig(
            d_agent=int(cfg.core.decoder.d), d_bev=_dbev,
            sources=("agent",) if _dbev <= 0 else ("agent", "bev"))
        cfg.tac_decoder_v6 = True
        cfg.tac_decoder_cfg = _tdc
        cfg.tac_decoder_valid_threshold = float(
            getattr(args, "tac_decoder_valid_threshold", 0.5))
        # ⭐ PI RULING 2026-09-17 R3. Attached is the RULING; detach is the
        # ABLATION. ⛔ Refused outright when there is no BEV path to detach —
        # a flag that is silently inert while config.json stamps it on is the
        # dead-flag class this trainer already refuses five times.
        _bdet = bool(getattr(args, "tac_decoder_bev_detach", False))
        if _bdet and _dbev <= 0:
            raise SystemExit(
                "[v3] ⛔ --tac-decoder-bev-detach with --tac-decoder-d-bev 0: "
                "there is no BEV path to detach, so the flag would be "
                "SILENTLY INERT while config.json stamped it on. It is the "
                "ablation of PI ruling R3 (*\"you can backpropagate to the "
                "trunk\"*) and only means something on a BEV arm.")
        cfg.tac_decoder_bev_detach = _bdet
    elif _w6 > 0.0:
        # ⚠️ `REFC_WEIGHT_GATES` also refuses this, and deliberately so: that
        # audit is the EXHAUSTIVE instrument and must stay able to see the
        # term. This raise is the EARLY, NAMED one — the pin runs on both
        # launch paths and before the model is constructed.
        raise SystemExit(
            "[v3] ⛔ --w-tac-v6 %.6g without --tac-decoder-v6: no behaviour "
            "decoder is built, so `out` carries no `tacv6_goal_logits` and "
            "the loss would have nothing to read. Pass --tac-decoder-v6, or "
            "--w-tac-v6 0." % _w6)
    if bool(getattr(args, "graft_behaviour_sel", False)):
        if not _on6:
            raise SystemExit(
                "[v3] ⛔ --graft-behaviour-sel without --tac-decoder-v6. The "
                "selection term is produced by the decoder's `planner_feeds` "
                "and by nothing else, so with no decoder the flag is "
                "SILENTLY INERT while config.json stamps it on — the dead-flag "
                "class this trainer already refuses five times. Pass "
                "--tac-decoder-v6, or drop --graft-behaviour-sel.")
        cfg.core.graft_behaviour_sel = True
    # ---- §5: the 4-way one-hot set-speed -------------------------------- #
    if bool(getattr(args, "max_speed_input_v6", False)):
        if args.arm != "hier":
            raise SystemExit(
                "[v3] ⛔ --max-speed-input-v6 on a FLAT arm: the one-hot "
                "enters the behaviour decoder's condition, which exists only "
                "in the hierarchy. Pass --arm hier, or drop the flag.")
        if bool(getattr(args, "max_speed_input", False)):
            raise SystemExit(
                "[v3] ⛔ --max-speed-input (E16, CONTINUOUS 8-step ladder over "
                "the v8 `speed_max_input` block) AND --max-speed-input-v6 "
                "(refcv6, 4-way one-hot over `SPEED_BAND.v_hi_ms`) are two "
                "DIFFERENT quantizations of a ceiling, from two different "
                "source fields. Feeding both makes the channel's effect "
                "non-attributable and `RefCV3Model.__init__` refuses the pair "
                "as well. Pick one.")
        if not getattr(args, "speed_max_sidecar_v6", None):
            raise SystemExit(
                "[v3] ⛔ --max-speed-input-v6 without --speed-max-sidecar-v6. "
                "The 4-value ladder is NOT a field of the label blob: it is "
                "the CONTAINING-WINDOW quantization of "
                "`g_tac.goals.SPEED_BAND.v_hi_ms`, produced by "
                "`scripts/build_refcv6_speed_max_window.py`. Without the "
                "sidecar the condition's 4 slots would be a constant all-zero "
                "pad ('not known') on every window and the arm would measure "
                "the channel as noise. Build the sidecar and pass it.")
        if not _on6:
            raise SystemExit(
                "[v3] ⛔ --max-speed-input-v6 without --tac-decoder-v6: the "
                "one-hot's ONLY consumer is the behaviour decoder's condition "
                "(`refc_v3.py:1597`), so with no decoder the encoder is built "
                "and read by nothing. Pass --tac-decoder-v6, or drop the flag.")
        cfg.max_speed_onehot_v6 = True


def _pin_refcv5_seams(cfg, args) -> None:
    """refcv5 WP-4 / WP-6 — install the sampler and the agent seam from argv.

    ⛔ EVERY seam here is OFF by default, and OFF means NOT CONSTRUCTED. A run
    that does not pass these flags builds a model that is bit-identical to
    refcv4b, RNG draw order included, so an existing checkpoint keeps loading
    strictly. That is checked by a test, not asserted here.
    """
    core = cfg.core
    # --- WP-4: the control-space DDIM sampler ---------------------------- #
    sampler = str(getattr(args, "sampler", "none"))
    core.decoder.sampler = sampler
    core.decoder.sampler_space = str(getattr(args, "sampler_space", "control"))
    core.decoder.sampler_train_t_max = int(getattr(args, "sampler_train_t_max",
                                                   50))
    core.decoder.sampler_infer_t = int(getattr(args, "sampler_infer_t", 8))
    core.decoder.sampler_steps = int(getattr(args, "sampler_steps", 2))
    core.decoder.sampler_groups = int(getattr(args, "sampler_groups", 1))
    if sampler == "ddim" and not core.anchors.v0_conditioned:
        raise SystemExit(
            "[v3] ⛔ --sampler ddim needs a v0-CONDITIONED vocabulary. The "
            "sampler's state IS the control sequence; a fixed-path bank "
            "carries anchor_controls of all zeros, so the anchored Gaussian "
            "would be centred on 'do nothing' — a plausible-looking WRONG "
            "experiment. Pass an --anchor-file built with controls.")
    # ⛔⛔ THE REFUSAL STANDS; ITS ORIGINAL REASON DOES NOT. This used to say
    # `control_head` "would stay at exactly zero". MEASURED 2026-09-10
    # (`tests/test_u0_control_head_reachability.py`): the head reaches
    # `out["anchor_traj"]`, which is what the matched-anchor L1 at `:2245-2247`
    # gathers, so it receives grad_abs_sum 9.39e4 from that path alone with the
    # u0 loss absent from the graph entirely — against exactly 0.0 for three
    # same-forward controls. ⚠️ So the arm is NOT "no denoiser at all"; it is a
    # denoiser supervised only through the integrator, which is a weaker claim
    # and a real experiment-design question (refcv6 arm D). Kept as a refusal
    # because that question is the PI's, not this gate's — but stated honestly,
    # so the decision is made against the measurement.
    if sampler == "ddim" and float(getattr(args, "w_u0", 0.0)) <= 0.0:
        # ⭐⭐ THE PI RULED 2026-09-11: "follow your recommendation" — authorise
        #     this configuration behind an EXPLICIT ACKNOWLEDGEMENT, so the run
        #     record shows a deliberate operator choice rather than a bypass.
        #     Same shape as `control_units_source: cli-override-legacy-file`.
        # ⛔ THE REFUSAL IS NOT REMOVED. Without the acknowledgement it still
        #     raises, because the thing it protects against is REAL: the arm is
        #     a denoiser supervised only through the integrator, and every log
        #     row still says 'sampler: ddim'.
        # ⛔ AND THE ACKNOWLEDGEMENT IS NOT A FIX — it is a RECORD. It changes
        #     nothing about the training; it only makes the choice attributable.
        if not bool(getattr(args, "ack_ddim_no_u0", False)):
            raise SystemExit(
                "[v3] ⛔ --sampler ddim with --w-u0 0 trains the sampler with NO "
                "loss on its OWN prediction: `control_head` is zero-init and would "
                "then be supervised only INDIRECTLY, through the matched-anchor L1 "
                "on the integrated fan (MEASURED: it does receive gradient that "
                "way, ~9.4e4 vs ~1.07e5 through the x0 loss — it does NOT stay at "
                "zero). That is a different experiment from a trained sampler, and "
                "every log row would still say 'sampler: ddim'. Pass --w-u0 > 0, "
                "run --sampler none, or — if this is refcv6 arm D and the PI has "
                "ruled — pass --ack-ddim-no-u0, which PERMITS it and STAMPS the "
                "choice into config.json as an operator decision.")
        # ⛔ PINNED ONTO THE CONFIG, not only onto `args`: `rebuild_config`
        #     rebuilds through this helper, so a stamp living only on the
        #     Namespace would be LOST on every roll — the same reason the
        #     max-speed mode is pinned above.
        cfg.u0_absent_under_ddim = "pi-acknowledged-2026-09-11-refcv6-arm-D"
    # --- STAGE 0: the feasibility-aware decode ---------------------------- #
    # ⛔ REFUSED AT STARTUP, NOT AT THE FIRST FORWARD. `refc.py::_feasible`
    # raises on a non-uniform prefix -- correctly -- but that raise arrives
    # after the model is built, the corpus is loaded and the optimiser is
    # allocated. Same class as the analysis-time import that died AFTER both
    # arms had rolled all 40 episodes: a preflight failure must cost seconds.
    feas = bool(getattr(args, "feasible_decode", False))
    core.decoder.feasible_decode = feas
    core.decoder.feasible_mu = float(getattr(args, "feasible_mu", 0.7))
    core.decoder.feasible_entry = bool(getattr(args, "feasible_entry", False))
    core.decoder.feasible_a_max = float(getattr(args, "feasible_a_max", 4.0))
    core.decoder.feasible_kappa_max = float(
        getattr(args, "feasible_kappa_max", 0.2))
    core.decoder.feasible_prefix_slots = int(
        getattr(args, "feasible_prefix_slots", 4))
    if feas:
        _hz = tuple(core.trajectory.horizons)
        _sl = [int(h) - 1 for h in _hz]
        _k = min(int(core.decoder.feasible_prefix_slots), len(_sl))
        _steps = [_sl[0] + 1] + [_sl[i] - _sl[i - 1] for i in range(1, _k)]
        if _k < 2 or len(set(_steps)) != 1:
            raise SystemExit(
                f"[v3] ⛔ --feasible-decode needs a UNIFORM prefix grid: "
                f"horizons {_hz[:_k]} give tick spacings "
                f"{_steps}. Projecting at a dt the scorer does not "
                f"differentiate produces an APPROXIMATELY feasible fan whose "
                f"residual reads as noise. Refusing rather than guessing -- "
                f"pass --feasible-prefix-slots N for the uniform head of the "
                f"grid.")
        if core.decoder.feasible_entry and not core.anchors.v0_conditioned:
            raise SystemExit(
                "[v3] ⛔ --feasible-entry without a v0-CONDITIONED vocabulary. "
                "The entry clamp binds the first step's speed to v0 +- a*dt; "
                "with no v0 the clamp is a no-op that would still be stamped "
                "into config.json, and the arm would read as the +entry "
                "variant while being the plain one. Pass an --anchors file "
                "built with controls, or drop --feasible-entry.")
        print(f"[v3] feasible decode ON: prefix {_k} slots at dt="
              f"{_steps[0] * 0.1:.2f} s, a_max {core.decoder.feasible_a_max}, "
              f"kappa_max {core.decoder.feasible_kappa_max}, mu "
              f"{core.decoder.feasible_mu}, entry "
              f"{core.decoder.feasible_entry}", flush=True)
    # --- WP-D: the BEV auxiliary head (TRAINING-ONLY) --------------------- #
    # ⛔ Its refusals mirror the `--w-agent` family EXACTLY, because the failure
    # they prevent is the same one and it has already happened here: a seam
    # declared in config.json whose loss term is silently skipped, so the run
    # record claims a lever that never entered the gradient.
    _bev_mode = str(getattr(args, "bev_aux", "off"))
    _w_bev = float(getattr(args, "w_bev_aux", 0.0))
    if _bev_mode != "off":
        if _w_bev <= 0.0:
            raise SystemExit(
                "[v3] ⛔ --bev-aux %s with --w-bev-aux 0 builds a head, stamps "
                "it into config.json and puts ZERO gradient into the trunk. "
                "Set --w-bev-aux > 0, or --bev-aux off." % _bev_mode)
        if not getattr(args, "agent_join", None):
            raise SystemExit(
                "[v3] ⛔ --bev-aux %s without --agent-join has NO LABELS. The "
                "BEV target is built from the SAME obstacle.offline join the "
                "detection head uses; without it every window is NO_LABEL and "
                "the term would be an exact 0.0 for the whole run. Pass "
                "--agent-join <joins/train2400_agents.jsonl.xz>." % _bev_mode)
        _gh, _gw = core.encoder.grid_shape
        core.bev_aux = _refc_bev_aux.BEVAuxConfig(
            enable=True, kind=_bev_mode,
            n_rng=int(getattr(args, "bev_aux_rng", 24)),
            n_az=int(_gw),
            d_tok=int(getattr(args, "bev_aux_dtok", 64)),
            hidden=int(getattr(args, "bev_aux_hidden", 256)),
            w=_w_bev,
            pos_weight=float(getattr(args, "bev_aux_pos_weight", 30.61)),
            occlusion=str(getattr(args, "bev_aux_occlusion", "mask")),
            detach_trunk=bool(getattr(args, "bev_aux_detach", False)))
        print("[v3] WP-D BEV aux ON: kind=%s grid=%dx%d target=%dx%d w=%.4g "
              "occlusion=%s detach=%s pos_weight=%.4g"
              % (_bev_mode, _gh, _gw, core.bev_aux.n_rng, core.bev_aux.n_az,
                 _w_bev, core.bev_aux.occlusion, core.bev_aux.detach_trunk,
                 core.bev_aux.pos_weight), flush=True)
    elif _w_bev > 0.0:
        # ⛔⛔ THE SILENT ONE, and the reason this branch exists: with
        # `--bev-aux off` no head is built, so `out` carries no `bev_logits`
        # and the loss-time guard skips the term -- while `w_bev_aux` is
        # stamped into config.json. That is the `w_agent` defect verbatim.
        raise SystemExit(
            "[v3] ⛔ --bev-aux off, but --w-bev-aux %.4g > 0. No head is "
            "built, so `bev_logits` never appears in `out` and the term is "
            "SILENTLY SKIPPED while the weight is stamped into config.json. "
            "Pass --bev-aux col|xcol, or --w-bev-aux 0." % _w_bev)
    # --- WP-6: the agent seam -------------------------------------------- #
    if getattr(args, "agents", "off") != "off":
        acfg = _refc_agents.AgentSeamConfig(
            enable=True,
            oracle=(args.agents == "oracle"),
            oracle_sigma_range_m=float(getattr(args, "agent_sigma_range", 0.0)),
            oracle_miss_rate=float(getattr(args, "agent_miss_rate", 0.0)),
            queries=int(getattr(args, "agent_queries",
                                AGENT_QUERIES_DEFAULT)),
            w_project=float(getattr(args, "agent_w_project", 0.0)),
            w_ground=float(getattr(args, "agent_w_ground", 0.0)),
            presence_hard=bool(getattr(args, "agent_presence_hard", False)))
        core.agents = acfg
        core.decoder.cross_agent = True
        # ⛔⛔ THE CAMERA IS BUILT HERE, AT PIN TIME, SO A CAMERA THAT CANNOT
        # BE BUILT FAILS BEFORE THE GPU AND NOT AT THE FIRST LOSS CALL.
        # `_build_rig_camera` REFUSES every configuration in which a non-zero
        # `--agent-w-project` / `--agent-w-ground` could not be computed.
        _build_rig_camera(cfg, args)
        if args.agents == "head" and float(getattr(args, "w_agent", 0.0)) <= 0.0:
            raise SystemExit(
                "[v3] ⛔ --agents head with --w-agent 0 builds a detector that "
                "is never supervised. Its tokens would be noise, the "
                "zero-init gate would have no reason to open, and the arm "
                "would read as 'agent tokens do not help' — a REFUTATION "
                "manufactured by a missing loss. Pass --w-agent > 0, or use "
                "--agents oracle (which needs no detector loss).")
        if args.agents == "head" and not getattr(args, "agent_join", None):
            raise SystemExit(
                "[v3] ⛔ --agents head without --agent-join has NO LABELS. "
                "The detector would be shaped only by the planner loss "
                "through its token gate and the arm would read as 'the "
                "learned agent head does not help' -- a REFUTATION "
                "manufactured by missing supervision. This fires HERE, before "
                "the GPU, rather than at the first batch. Pass --agent-join "
                "(HF Sayood/tanitad-ph0-aug120 -> "
                "joins/train2400_agents.jsonl.xz), or --agents oracle.")
    else:
        # ⛔⛔ --agents off MAKES EVERY AGENT WEIGHT A NO-OP, AND ONE OF THEM
        # IS A SILENT ONE THAT THE M18 AUDIT DID NOT NAME. MEASURED at source:
        # with `--agents off` no `AgentSeamConfig` is built, so `out` carries
        # no `agent_slots`; `compute_losses_v3`'s first guard only fires when
        # `agent_box` is ABSENT from the batch, so
        # `--agents off --w-agent 1.0 --agent-join <file>` passes that guard
        # (the join DOES put `agent_box` in the batch), then falls through the
        # `"agent_slots" in out` condition and computes NOTHING -- while
        # `w_agent 1.0` is stamped into config.json. Exactly the
        # `--agent-w-project` defect, one flag over.
        # ⇒ every agent weight is refused here rather than silently dropped,
        # and the camera with it. There is no third state in this direction
        # either.
        _dead = {k: float(getattr(args, k, 0.0) or 0.0)
                 for k in ("w_agent", "agent_w_project", "agent_w_ground")}
        _dead = {k: v for k, v in _dead.items() if v > 0.0}
        _cam = str(getattr(args, "agent_rig_camera", "off"))
        # ⚠️ refcv6 §2: THE CAMERA IS NOT DEAD UNDER `--w-map > 0`, and
        # refusing it there would be a FALSE REFUSAL -- the mirror of the
        # defect this guard exists for, and just as costly: it would make
        # `--agents off --w-map 1.0` (a legitimate map-only arm) unlaunchable,
        # so an operator would switch the detector on to get past the guard
        # and quietly run a different experiment. The BEV lift reads the SAME
        # per-clip extrinsics table through `LiftGeometryBank`, so under a live
        # map weight the camera has a consumer. The three WEIGHTS above stay
        # refused: none of them has one.
        if float(getattr(args, "w_map", 0.0) or 0.0) > 0.0:
            _cam = "off"
        if _dead or _cam != "off":
            raise SystemExit(
                "[v3] ⛔ --agents off, but "
                + (f"{_dead} " if _dead else "")
                + (f"--agent-rig-camera {_cam} " if _cam != "off" else "")
                + "is set. With no agent seam the model emits no "
                "`agent_slots` and `refc_agents.agent_losses` is never "
                "called, so every one of these parses, is STAMPED INTO "
                "config.json, and trains nothing -- the run record would "
                "state a configuration that did not happen (mm-decisions "
                "M18). ⚠️ `--w-agent > 0` with a join is the silent one: "
                "the join puts `agent_box` in the batch, so the loss-time "
                "guard does not fire and the detection loss is simply "
                "skipped. Pass --agents head/oracle, or set these to their "
                "defaults.")
    # --- WP-B: the waypoint index on the agent cross-attention ------------ #
    #
    # ⛔ ORDER IS LOAD-BEARING: this runs AFTER the WP-6 block above, because
    # it reads `core.decoder.cross_agent`, which that block sets. Placed before
    # it, the `--agents head --wp-index on` combination would refuse itself.
    _wpx = str(getattr(args, "wp_index", "off"))
    if _wpx != "off":
        if str(getattr(args, "agents", "off")) == "off":
            raise SystemExit(
                "[v3] ⛔ --wp-index %s with --agents off. WP-B is a "
                "waypoint-indexed cross-attention INTO THE SPARSE AGENT "
                "TOKENS; with no agent seam there are no tokens to address, "
                "so the bias heads would be built, STAMPED INTO config.json "
                "and never called -- and the arm would read as 'the waypoint "
                "index does not help' while never having had an index. That "
                "is a REFUTATION MANUFACTURED BY A MISSING SEAM, and it "
                "refuses here, before the GPU, rather than at the first "
                "batch. Pass --agents head|oracle, or --wp-index off."
                % _wpx)
        core.decoder.wp_index = _refc_wp_index.WaypointIndexConfig(
            enable=True,
            hidden=int(getattr(args, "wp_index_hidden", 32)),
            scale_m=float(getattr(args, "wp_index_scale_m", 10.0)),
            mode=str(getattr(args, "wp_index_mode", "geom")),
            const_xy=tuple(float(v) for v in
                           getattr(args, "wp_index_const_xy", (10.0, 0.0))),
            detach=bool(getattr(args, "wp_index_detach", False)),
            radius_m=float(getattr(args, "wp_index_radius_m", 0.0)))
        print("[v3] WP-B waypoint index ON: mode=%s detach=%s radius_m=%.3g "
              "hidden=%d scale_m=%.3g -> %d decoder layers"
              % (core.decoder.wp_index.mode, core.decoder.wp_index.detach,
                 core.decoder.wp_index.radius_m, core.decoder.wp_index.hidden,
                 core.decoder.wp_index.scale_m, int(core.decoder.layers)),
              flush=True)
    else:
        # ⛔⛔ --wp-index off MAKES EVERY WP-B KNOB A NO-OP THAT STILL LANDS IN
        # config.json. Same class as the `--agent-w-project` and `--w-bev-aux`
        # defects one seam over: the flag parses, the value is stamped, and the
        # run record states a configuration that did not happen. There is no
        # third state -- the knobs are refused, not silently dropped.
        # ⚠️ EVERY knob, not just the interesting three. The first draft
        # refused only mode/detach/radius and left `--wp-index-hidden`,
        # `--wp-index-scale-m` and `--wp-index-const-xy` free -- and those
        # three PARSE, land in `agent_knobs` in config.json, and do nothing,
        # which is the M18 no-op-flag defect verbatim. It was caught by
        # `test_refc_v3_agent_provenance.py::test_P2_every_knob_is_recoverable_
        # from_the_stamp_BY_VALUE`, which derives the knob list FROM ARGPARSE
        # rather than from a hand-written list that can rot.
        _dead_wp = {}
        if str(getattr(args, "wp_index_mode", "geom")) != "geom":
            _dead_wp["--wp-index-mode"] = getattr(args, "wp_index_mode")
        if bool(getattr(args, "wp_index_detach", False)):
            _dead_wp["--wp-index-detach"] = True
        if float(getattr(args, "wp_index_radius_m", 0.0)) != 0.0:
            _dead_wp["--wp-index-radius-m"] = getattr(args, "wp_index_radius_m")
        if int(getattr(args, "wp_index_hidden", 32)) != 32:
            _dead_wp["--wp-index-hidden"] = getattr(args, "wp_index_hidden")
        if float(getattr(args, "wp_index_scale_m", 10.0)) != 10.0:
            _dead_wp["--wp-index-scale-m"] = getattr(args, "wp_index_scale_m")
        if tuple(float(v) for v in
                 getattr(args, "wp_index_const_xy", (10.0, 0.0))) \
                != (10.0, 0.0):
            _dead_wp["--wp-index-const-xy"] = list(
                getattr(args, "wp_index_const_xy"))
        if _dead_wp:
            raise SystemExit(
                "[v3] ⛔ --wp-index off, but %s is set. With the index off no "
                "`WaypointIndexConfig` is built and no bias head is attached, "
                "so every one of these parses, is STAMPED INTO config.json, "
                "and does nothing -- the run record would state a "
                "configuration that did not happen (mm-decisions M18). "
                "⚠️ --wp-index-mode is the dangerous one: it names a CONTROL "
                "ARM, so a reader would believe a control had been run. Pass "
                "--wp-index on, or leave these at their defaults." % _dead_wp)
    _pin_refcv6_perception(cfg, args)


# ============================================================================
# ⛔⛔ refcv6 §2/§6 — THE PERCEPTION BRANCH'S REFUSALS
# ============================================================================
# EVERY ONE OF THESE FIRES BEFORE `config.json` IS WRITTEN AND BEFORE A SINGLE
# BATCH IS LOADED, which is the whole point: the defect class they close is a
# head that PARSES, is STAMPED `enabled`, and reaches nothing. The programme has
# now measured that class five times (`--w-agent`, `--agent-w-project`,
# `--agent-w-ground`, `--w-bev-aux`, `--conflict-detector on`), and once with
# the object being a HEAD rather than a weight: `tac_goal_tok_head`, 11,286
# parameters with `grad_abs_sum` EXACTLY 0 for all 40,284 steps.

def _pin_refcv6_perception(cfg, args) -> None:
    """Refuse every ``--w-map`` / ``--w-box3d`` combination that cannot train.

    ⛔ The REVERSE refusal is here too — a supplied artifact with a zero weight.
    An operator who passes ``--map-gt-root`` and forgets ``--w-map`` gets a run
    whose ``config.json`` names the SAM3 corpus and whose trunk never saw one
    cell of it, and there is no artifact that would later reveal it.
    """
    w_map = float(getattr(args, "w_map", 0.0) or 0.0)
    w_b3d = float(getattr(args, "w_box3d", 0.0) or 0.0)
    root = getattr(args, "map_gt_root", None)
    j3d = getattr(args, "join3d", None)
    if w_map < 0.0 or w_b3d < 0.0:
        raise SystemExit("[v3] ⛔ --w-map / --w-box3d must be >= 0.")
    if w_map == 0.0 and w_b3d == 0.0:
        # The DEFAULT path. Nothing is built; the only thing that can be wrong
        # is an artifact supplied with no weight to consume it.
        _dead = {k: v for k, v in (("--map-gt-root", root),
                                   ("--join3d", j3d)) if v}
        if float(getattr(args, "map_min_coverage", None) or 0.0) > 0.0:
            _dead["--map-min-coverage"] = getattr(args, "map_min_coverage")
        if _dead:
            raise SystemExit(
                "[v3] ⛔ --w-map 0 and --w-box3d 0, but %s is set. No "
                "perception branch is built at zero weight, so the label "
                "corpus would be NAMED in config.json and READ BY NOTHING -- "
                "a run record stating a configuration that did not happen "
                "(mm-decisions M18). Pass a weight, or drop the artifact."
                % _dead)
        return
    # ---- from here on at least one head is live -------------------------- #
    if str(getattr(args, "trunk", "refc")) != "timm":
        raise SystemExit(
            "[v3] ⛔ --w-map/--w-box3d > 0 needs --trunk timm. refcv6 "
            "perception reads the STRIDE-16 map, and the legacy REF-C "
            "ResNetEncoder exposes only stride 32 -- `fmap_s16` would be None "
            "and the branch would have nothing to read. (An oracle on the "
            "stride-32 map caps at AP 0.3341 vs 0.4713, SPEC_REFCV6_V2 §2.)")
    if w_map > 0.0:
        if not root:
            raise SystemExit(
                "[v3] ⛔ --w-map > 0 without --map-gt-root has NO LABELS. The "
                "BEV branch would be built, stamped, and shaped only by the "
                "box head's gradient -- and the arm would read as 'the map "
                "head does not help' while never having had a map.")
        if str(getattr(args, "agent_rig_camera", "off")) != "extrinsics":
            raise SystemExit(
                "[v3] ⛔ --w-map > 0 needs --agent-rig-camera extrinsics "
                "--agent-rig-extrinsics <table>. The lift back-projects "
                "through the ROAD PLANE, so the mount pose sets which image "
                "pixel fills which BEV cell; one pose for a corpus whose "
                "MEASURED height spans 1.2131-1.6672 m over 554 distinct "
                "values in 2,400 clips teaches the head that geometry as "
                "truth. A nominal camera is not admissible here.")
    if w_b3d > 0.0 and not getattr(args, "agent_join", None):
        raise SystemExit(
            "[v3] ⛔ --w-box3d > 0 without --agent-join has NO LABELS. The "
            "3-D set loss IS the 2-D set loss plus two masked metre terms; "
            "with no 2-D targets there is nothing to Hungarian-match against.")
    if j3d and w_b3d == 0.0:
        raise SystemExit(
            "[v3] ⛔ --join3d with --w-box3d 0: the cuboid heights would be "
            "loaded, joined, emitted into every batch and multiplied by zero, "
            "while config.json names the 3-D join. Pass --w-box3d > 0.")
    if w_b3d > 0.0 and not j3d:
        # ⚠️ A WARNING, NOT A REFUSAL, and the difference is real: this is a
        # LEGAL arm (the 2-D-only rung `box3d_set_loss` itself documents, whose
        # total is EXACTLY the 2-D total). It is announced because a run that
        # believes it is training 3-D and is not would otherwise be invisible.
        print("[v3] ⚠️ --w-box3d > 0 with NO --join3d: `zh_mask` is all-False, "
              "so `loss_z`/`loss_h` are 0.0 over n=0 items and the total is "
              "EXACTLY the 2-D set loss. This is the 2-D rung, not a 3-D arm.",
              flush=True)


# ============================================================================
# ⛔⛔ THE RIG CAMERA — the flag that used to PARSE, STAMP AND DO NOTHING
# ============================================================================

#: ``--agent-rig-camera`` sources. ``off`` = no camera (and then a non-zero
#: image-plane / ground weight REFUSES); ``nominal`` = boresight-forward mount,
#: NO pitch, admissible only where the caller records that; ``extrinsics`` =
#: the corpus constructor, from a ``sensor_extrinsics`` quaternion on file.
AGENT_RIG_CAMERA_SOURCES = ("off", "nominal", "extrinsics")

#: a parameter gradient at or below this is INDISTINGUISHABLE FROM ZERO in
#: float32 and is treated as "this term trains nothing". The measured
#: tautology reads 8.73e-11; a live term reads O(1e-3) or more, so the
#: floor sits three orders below the smallest live signal and seven above
#: the dead one -- it separates them by construction, not by tuning.
_GROUND_GRAD_FLOOR: float = 1e-7

#: ``_build_rig_camera`` is called at pin time (three times, via the config
#: delta), at model setup and at stamp time — deliberately, so a camera that
#: cannot be built refuses at the earliest of them. Its human-facing lines are
#: printed ONCE: a warning repeated five times reads as five warnings.
_RIG_CAM_ANNOUNCED: set = set()

#: ``(height, width) -> CanonicalFrame`` for every geometry the corpus is
#: resampled into. ⛔ A frame is NOT its pixel count: it also carries ``f_ref``
#: and the PROJECTION, and this corpus is **cylindrical**, where the column is
#: LINEAR IN AZIMUTH. Applying the pinhole formula to it reads 92.6° for a
#: 120° camera and looks entirely plausible (`CLAUDE.md`, the cylindrical-FOV
#: trap), so an unknown geometry is REFUSED rather than given an invented
#: ``f_ref``.
def _agent_cam_frames() -> dict:
    from tanitad.data import calib as _calib
    from tanitad.models import trunk_shapes as _ts
    return {(256, 640): _calib.PHYSICALAI_WIDE120_256x640,
            (176, 624): _calib.PHYSICALAI_RIG_CLEAN_176x624,
            (128, 576): _calib.PHYSICALAI_RIG_CLEAN_128x576,
            # ⭐ The PI's 2026-09-16 geometry. ⛔ TAKEN FROM
            # `trunk_shapes.FRAME_256x1024`, NOT written out here: that module
            # derives `f_ref` as the 640 frame's scaled by 1024/640
            # (305.5774907364391 x 1.6 = 488.92398517830253), which holds the
            # field at EXACTLY 120.0000 deg. Re-typing "488.92" gives
            # 120.0010 deg, and a SECOND spelling of a camera constant is how
            # two files drift apart while both look right.
            # ⚠️ Until 2026-09-17 this table had no 256x1024 row and
            # `--agent-rig-camera extrinsics` REFUSED the PI's own cache --
            # a correct refusal (a frame is not its pixel count; this corpus
            # is CYLINDRICAL and the pinhole formula reads 92.6 deg for a
            # 120 deg camera) against a frame that was ALREADY DECLARED one
            # module over. MEASURED here: it blocked the first live refcv6
            # perception run.
            (256, 1024): _ts.FRAME_256x1024,
            # ⭐ The PI's 2026-09-17 geometry, replacing 256x1024. Same discipline
            # as the row above: TAKEN FROM `trunk_shapes.FRAME_416x1024`, which
            # derives it through `frame_for_width`, never re-typed here.
            # ⚠️ This row exists because the SAME omission bit twice: the first
            # resnet101 run on the freshly built 416x1024 cache was REFUSED by
            # this very table (MEASURED 2026-09-17, 20:39Z) exactly as the
            # 256x1024 cache had been the day before. A new geometry is not
            # finished when its cache is built -- it is finished when every
            # table that must declare it does.
            (416, 1024): _ts.FRAME_416x1024}


def _extr_from_obj(d: dict, path: str, where: str):
    """One ``FrontWideExtrinsics`` from a JSON object, or REFUSE."""
    from tanitad.data.physicalai import FrontWideExtrinsics
    miss = [k for k in ("qx", "qy", "qz", "qw") if k not in d]
    if miss:
        raise SystemExit(
            f"[v3] ⛔ --agent-rig-extrinsics {path} [{where}] declares no "
            f"{miss}. The file must carry the sensor_extrinsics quaternion "
            "(rotation_cam_to_vehicle) as qx/qy/qz/qw, plus optional x/y/z "
            "[m] in the rig frame. A camera whose MOUNT POSE is guessed "
            "biases `ground_range_prior` directly (it back-projects through "
            "the road plane), so it is read from the file or refused.")
    return FrontWideExtrinsics(
        qx=float(d["qx"]), qy=float(d["qy"]), qz=float(d["qz"]),
        qw=float(d["qw"]), x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)), z=float(d.get("z", 1.5)))


def _read_rig_extrinsics(path: str):
    """``-> (single | None, {clip_id: FrontWideExtrinsics} | None)``.

    Two shapes are accepted and they are DISTINGUISHED, never conflated:

    * a **single** ``FrontWideExtrinsics``-shaped object (``qx qy qz qw``
      + optional ``x y z``) — ONE mount pose for the whole run;
    * a **per-CLIP TABLE**, ``{clip_id: {qx, qy, qz, qw, x, y, z, ...}}`` —
      the shape ``taniteval/tools/pai_extrinsics_table.py`` already emits and
      that ``taniteval/results/videos/refcv3_five_panel_step40284/
      extrinsics_used.json`` already holds.

    ⛔⛔ **WHY BOTH, AND WHY THE DIFFERENCE IS STAMPED.** MEASURED over 40
    PhysicalAI clips from the dataset's own ``calibration/sensor_extrinsics``:
    mount height **1.245-1.607 m**, median 1.306, **37 distinct values in 40
    clips**, CV 7.4 %; forward offset ~2.0-2.1 m; pitch -1.15..+2.34 deg. The
    rig label does NOT stand in for it — the two rigs' medians differ by 1.5 %
    while the WITHIN-rig spread is 29 %. ``ground_range_prior`` back-projects
    through the road plane, so its supervised range is directly proportional
    to the mount height: a single 1.5 m camera on a 1.245 m clip biases every
    range it teaches by **+20 %**. A run may still declare one camera — it is
    a legal coarse arm — but ``config.json`` says ``mount_pose_scope:
    SINGLE-CAMERA-WHOLE-CORPUS`` so no reader can mistake it for a per-clip
    one. That is the whole M18 lesson applied to the mount pose.

    ⚠️ Read from a FILE, never guessed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict):
        raise SystemExit(f"[v3] ⛔ {path} is not a JSON object.")
    d = d.get("front_wide", d) if "front_wide" in d else d
    quat = ("qx", "qy", "qz", "qw")
    vals = [v for v in d.values() if isinstance(v, dict)]
    is_table = (bool(d) and len(vals) == len(d)
                and all(all(k in v for k in quat) for v in vals))
    if is_table:
        return None, {str(k): _extr_from_obj(v, path, str(k))
                      for k, v in d.items()}
    return _extr_from_obj(d, path, "front_wide"), None


def _build_rig_camera(cfg, args):
    """Build ``model._rig_camera`` — or REFUSE. Returns ``(cam, stamp)``.

    ⛔⛔ **WHY THIS FUNCTION EXISTS.** MEASURED 2026-09-05 (`Decisions/
    2026-09-05-mm-decisions.md` §M18, DataFlyWheel escalation #2):
    ``refc_v3_train.py`` set ``model._rig_camera = None`` and **never assigned
    it**, while ``refc_agents.agent_losses`` guards both monocular terms on
    ``cam is not None``. So a run could pass ``--agent-w-project 0.2``, have it
    written into ``config.json``, and **compute nothing** — the run record
    stating a training configuration that did not happen. That is worse than a
    missing flag: it makes every later comparison between arms unfalsifiable,
    because the record lies about the arms.

    ⛔ **There is no third state.** Either a camera is built and the terms
    compute, or the non-zero weight REFUSES here, at pin time, before the GPU.
    A weight that silently no-ops is not reachable from any argv.

    ⚠️ **The mount pose carries its provenance**, exactly as
    ``anchor_meta.control_units_source`` does for the anchor artifact: a
    ``nominal`` camera is stamped ``mount_pose = "NOMINAL-no-pitch"`` and says
    so in the log, because ``ground_range_prior`` back-projects through the
    road plane and a wrong pitch biases its range directly (the projection
    term is a DIFFERENCE of two projections and cancels a common mount error
    to first order; the ground term does not).
    """
    src = str(getattr(args, "agent_rig_camera", "off"))
    if src not in AGENT_RIG_CAMERA_SOURCES:
        raise SystemExit(f"[v3] ⛔ --agent-rig-camera {src!r} not in "
                         f"{AGENT_RIG_CAMERA_SOURCES}.")
    w_pj = float(getattr(args, "agent_w_project", 0.0))
    w_gr = float(getattr(args, "agent_w_ground", 0.0))
    if src == "off":
        if w_pj > 0.0 or w_gr > 0.0:
            raise SystemExit(
                "[v3] ⛔ --agent-w-project "
                f"{w_pj} / --agent-w-ground {w_gr} > 0 with "
                "--agent-rig-camera off. Both terms are computed by "
                "`refc_agents.agent_losses` ONLY when a RigCamera is passed, "
                "so this run would stamp the weights into config.json and "
                "train NEITHER term -- a run record that states a "
                "configuration that did not happen, which makes every later "
                "comparison between arms unfalsifiable. MEASURED as a live "
                "defect 2026-09-05 (mm-decisions M18); it now REFUSES rather "
                "than no-ops. Pass --agent-rig-camera nominal (declares a "
                "pitch-free mount) or --agent-rig-camera extrinsics "
                "--agent-rig-extrinsics <sensor_extrinsics.json> (the "
                "corpus constructor), or set both weights to 0.")
        return None, {"source": "off", "reason": "not requested",
                      "mount_pose_scope": "NONE", "n_clips": 0,
                      "w_project": w_pj, "w_ground": w_gr}
    from tanitad.data.rig_projection import CAM_HEIGHT_RANGE_M, RigCamera
    hw = tuple(int(v) for v in cfg.core.encoder.image_hw())
    frames = _agent_cam_frames()
    frame = frames.get(hw)
    if frame is None:
        raise SystemExit(
            f"[v3] ⛔ --agent-rig-camera {src} at image_hw {hw}: no canonical "
            "frame is DECLARED for that geometry, and one cannot be invented. "
            "A CanonicalFrame is not its pixel count -- it also carries f_ref "
            "and the PROJECTION, and this corpus is CYLINDRICAL (the column "
            "is linear in azimuth). Guessing an f_ref, or applying the "
            "pinhole formula, reads 92.6 deg for a 120 deg camera and looks "
            "entirely plausible. Declared geometries: "
            f"{sorted(frames)}. To train the monocular terms at another "
            "geometry, add its CanonicalFrame to `tanitad.data.calib` (with "
            "its f_ref and projection) and register it here.")
    if src == "nominal":
        h_m = float(getattr(args, "agent_cam_height", 1.5))
        lo, hi = CAM_HEIGHT_RANGE_M
        if not (0.5 <= h_m <= 3.0):
            raise SystemExit(
                f"[v3] ⛔ --agent-cam-height {h_m} m is not a plausible "
                f"front-wide mount (MEASURED corpus range {lo}-{hi} m).")
        cam = RigCamera.nominal(frame, height_m=h_m)
        stamp = {"source": "nominal", "mount_pose": "NOMINAL-no-pitch",
                 "mount_pose_scope": "SINGLE-CAMERA-WHOLE-CORPUS",
                 "n_clips": 1,
                 "height_m": h_m, "extrinsics_path": None,
                 "in_measured_height_band": bool(lo <= h_m <= hi)}
        _say = ("nominal", hw, h_m) not in _RIG_CAM_ANNOUNCED
        _RIG_CAM_ANNOUNCED.add(("nominal", hw, h_m))
        if _say and not (lo <= h_m <= hi):
            print(f"[v3] ⚠️ --agent-cam-height {h_m} m is outside the "
                  f"MEASURED corpus band {lo}-{hi} m.", flush=True)
        if _say:
            print("[v3] ⚠️ rig camera = NOMINAL (boresight forward, NO mount "
                  "pitch). `ground_range_prior` back-projects through the "
                  "road plane, so any real mount pitch biases its range "
                  "DIRECTLY; `monocular_projection_loss` is a difference of "
                  "two projections and cancels a common mount error to first "
                  "order. Use --agent-rig-camera extrinsics for corpus "
                  "work.", flush=True)
    else:
        path = getattr(args, "agent_rig_extrinsics", None)
        if not path:
            raise SystemExit(
                "[v3] ⛔ --agent-rig-camera extrinsics needs "
                "--agent-rig-extrinsics <sensor_extrinsics.json>. "
                "`RigCamera.from_extrinsics` is the ONLY admissible "
                "constructor for corpus work -- the per-clip mount pitch is "
                "what puts the horizon on the right row.")
        extr, table = _read_rig_extrinsics(path)
        if table is not None:
            # ---- PER-CLIP BANK ------------------------------------------ #
            from tanitad.data.v2_dataset import stable_episode_id
            from tanitad.refs.refc_agents import RigCameraBank
            by_ep = {}
            for cid, e in table.items():
                sid = int(stable_episode_id(str(cid)))
                if sid in by_ep:
                    # ⛔ A collision would attach one clip's mount pose to
                    # another clip's rows -- a label corruption no downstream
                    # metric could attribute. 63-bit ids make this ~4e-12, so
                    # it is asserted rather than assumed.
                    raise SystemExit(
                        f"[v3] ⛔ --agent-rig-extrinsics {path}: two clip_ids "
                        f"collide on stable_episode_id {sid}. Refusing rather "
                        "than attaching one clip's camera to another's rows.")
                by_ep[sid] = RigCamera.from_extrinsics(e, frame)
            cam = RigCameraBank(by_episode=by_ep, default=None)
            zs = sorted(float(e.z) for e in table.values())
            ps = sorted(float(e.optical_axis_pitch_rad())
                        for e in table.values())
            stamp = {"source": "extrinsics",
                     "mount_pose": "from-sensor_extrinsics",
                     "mount_pose_scope": "PER-CLIP",
                     "extrinsics_path": str(path),
                     "n_clips": len(by_ep),
                     "key": "stable_episode_id(clip_id)",
                     "height_m_min": zs[0], "height_m_max": zs[-1],
                     "height_m_median": zs[len(zs) // 2],
                     "optical_axis_pitch_rad_min": ps[0],
                     "optical_axis_pitch_rad_max": ps[-1]}
            if ("per-clip", hw, str(path)) not in _RIG_CAM_ANNOUNCED:
                _RIG_CAM_ANNOUNCED.add(("per-clip", hw, str(path)))
                print("[v3] rig camera = PER-CLIP EXTRINSICS from %s "
                      "(%d clips, z %.3f-%.3f m, pitch %+.4f..%+.4f rad)"
                      % (path, len(by_ep), zs[0], zs[-1], ps[0], ps[-1]),
                      flush=True)
        else:
            cam = RigCamera.from_extrinsics(extr, frame)
            stamp = {"source": "extrinsics",
                     "mount_pose": "from-sensor_extrinsics",
                     # ⛔ STATED, so a reader of config.json cannot mistake a
                     # ONE-camera arm for a per-clip one. The corpus mount
                     # pose is per CLIP (1.245-1.607 m over 40 clips, 37
                     # distinct values, CV 7.4 %); one camera per RUN is a
                     # legal coarse arm and this is the word that says so.
                     "mount_pose_scope": "SINGLE-CAMERA-WHOLE-CORPUS",
                     "n_clips": 1,
                     "height_m": float(extr.z), "extrinsics_path": str(path),
                     "optical_axis_pitch_rad":
                         float(extr.optical_axis_pitch_rad()),
                     "quat_cam_to_vehicle": [extr.qx, extr.qy, extr.qz,
                                             extr.qw]}
            if ("extrinsics", hw, str(path)) not in _RIG_CAM_ANNOUNCED:
                _RIG_CAM_ANNOUNCED.add(("extrinsics", hw, str(path)))
                print(f"[v3] rig camera = EXTRINSICS from {path} (pitch "
                      f"{stamp['optical_axis_pitch_rad']:+.4f} rad, z "
                      f"{stamp['height_m']:.3f} m)", flush=True)
                if w_gr > 0.0:
                    print("[v3] ⚠️ ONE camera for the whole corpus with "
                          "--agent-w-ground > 0. `ground_range_prior` "
                          "back-projects through the road plane, so its "
                          "supervised range scales with the mount height, and "
                          "the MEASURED corpus height spans 1.245-1.607 m "
                          "(37 distinct values in 40 clips). Pass a PER-CLIP "
                          "extrinsics table (the shape "
                          "taniteval/tools/pai_extrinsics_table.py emits) to "
                          "remove that bias; config.json records "
                          "mount_pose_scope so the arm is identifiable "
                          "either way.", flush=True)
    stamp.update({"frame": frame.to_dict() if hasattr(frame, "to_dict") else
                  {"height": frame.height, "width": frame.width,
                   "f_ref": float(frame.f_ref),
                   "projection": frame.projection},
                  "image_hw": list(hw), "w_project": w_pj, "w_ground": w_gr})
    if w_pj <= 0.0 and w_gr <= 0.0 and ("zero", hw) not in _RIG_CAM_ANNOUNCED:
        _RIG_CAM_ANNOUNCED.add(("zero", hw))
        print("[v3] ⚠️ a rig camera is built but --agent-w-project and "
              "--agent-w-ground are both 0: neither monocular term "
              "contributes. That is a legal ablation baseline and the stamp "
              "records it, but it is not a projection arm.", flush=True)
    return cam, stamp


# ============================================================================
# Dataset — parity-preserving 6 s
# ============================================================================

def assert_nav_token_alignment() -> None:
    """⛔ The position pin: ``vocab_v7.NAV_COMMAND_TOKENS[i]`` must name
    ``refb.NAV_COMMANDS[i]`` through :data:`NAV_TOKEN_TO_LEGACY`, and the core
    must one-hot against the same tuple (``refc.NAV_COMMANDS``). Verified, not
    assumed — a silent re-ordering would land the token on the wrong nav row
    and train on a plausible wrong signal."""
    toks = tuple(vocab_v7.NAV_COMMAND_TOKENS)
    legacy = tuple(refb.NAV_COMMANDS)
    for i, tok in enumerate(toks):
        want = NAV_TOKEN_TO_LEGACY.get(tok)
        have = legacy[i] if i < len(legacy) else None
        if want is None or want != have:
            raise AssertionError(
                f"[v3 nav-from-v7] nav index alignment broken: "
                f"NAV_COMMAND_TOKENS[{i}]={tok!r} maps to {want!r} but "
                f"refb.NAV_COMMANDS[{i}]={have!r} — the v7.2 token would land "
                f"on the wrong nav row")
    if tuple(refc.NAV_COMMANDS) != legacy:
        raise AssertionError(
            f"[v3 nav-from-v7] refc.NAV_COMMANDS {tuple(refc.NAV_COMMANDS)} != "
            f"refb.NAV_COMMANDS {legacy} — the core one-hots nav_cmd against "
            f"its own tuple (refc.py:2024); the two must stay identical")
    if legacy.index("follow") != refb_labels.NAV_FOLLOW:
        raise AssertionError(
            f"[v3 nav-from-v7] 'follow' sits at {legacy.index('follow')} in "
            f"refb.NAV_COMMANDS but refb_labels.NAV_FOLLOW is "
            f"{refb_labels.NAV_FOLLOW} — the unlabeled default would not be "
            f"'follow'")


def _check_nav_from_v7_args(args) -> None:
    """Refuse AT START a --nav-from-v7 launch that would die or mislead later.

    * without ``--v7-labels`` there is no token to read — the flag would be a
      dead switch that LOOKS switched;
    * with an in-training eval (``--eval-cache`` + ``--eval-every``) but no
      ``--eval-labels``, train and eval would be fed DIFFERENT nav sources and
      every eval row would compare a v7.2-fed model against the v1 input —
      an eval that silently measures something else.
    ``getattr`` throughout: test rigs build partial Namespaces."""
    if not getattr(args, "nav_from_v7", False):
        return
    if not getattr(args, "v7_labels", None):
        raise SystemExit("[v3] ⛔ --nav-from-v7 needs --v7-labels: the nav "
                         "token is read from the v7.2 label records, and "
                         "without them the flag would be a dead switch that "
                         "looks switched.")
    if (getattr(args, "eval_cache", None) and getattr(args, "eval_every", 0)
            and not getattr(args, "eval_labels", None)):
        raise SystemExit("[v3] ⛔ --nav-from-v7 with an in-training eval needs "
                         "--eval-labels: the eval dataset must see the SAME "
                         "nav source as training, or every eval row compares "
                         "against the v1 derivation.")


def _check_max_speed_args(args) -> None:
    """Refuse AT START a ``--max-speed-input`` launch that would mislead.

    ⛔ A DEAD FLAG IS A REFUSAL, NOT A NO-OP. ``speed_max_input`` is a v8
    field: the v7.2 release carries it on 0 of 4,572 train records and the
    v8 release on 4,572/4,572 and 147/147 (MEASURED 2026-09-06). Pointed at
    a v7.2 blob the flag would look switched, feed nothing, and the arm
    would report as +max-speed while running without a ceiling. The
    LOADER's own refusal names the split and the field; this one fires
    before any GPU work, on both launch paths.
    """
    if not getattr(args, "max_speed_input", False):
        return
    # ⛔ --preflight CANNOT EXERCISE THIS CHANNEL, and it must say so HERE rather
    # than die inside the loss step. MEASURED 2026-09-07: preflight builds its
    # corpus with `_synth_episodes` (CI-only, 2 clips) and a synthetic clip
    # carries no `speed_max_input`, so `enable_max_speed` is never called and the
    # forward correctly refuses a batch with no `v_max_ms` -- 20 lines into the
    # run, after the gates have printed PASS. An operator preflighting their real
    # launch line would read that as "the flag is broken" and drop it. ⭐ It is
    # NOT the flag failing, and the refusal says so.
    if getattr(args, "preflight", False):
        raise SystemExit(
            "[v3] ⛔ --preflight --max-speed-input is not runnable, and this is "
            "NOT the flag failing: preflight's corpus is SYNTHETIC "
            "(`_synth_episodes`, CI-only) and a synthetic clip carries no "
            "`speed_max_input` block, so the loss step would meet a batch with "
            "no `v_max_ms`. ⇒ Preflight the arm WITHOUT the flag to check the "
            "build, the delta and the gates. The channel's OWN preflight is the "
            "census `enable_max_speed` prints at train() startup, which refuses "
            "a label blob carrying the field on 0 clips -- and the seam's "
            "parameter cost is pinned by "
            "stack/tests/test_max_speed_wiring.py.")
    if not getattr(args, "v7_labels", None):
        raise SystemExit(
            "[v3] ⛔ --max-speed-input needs --v7-labels: the ceiling is "
            "read from the label record's `speed_max_input` block (v8 "
            "schema `speed_max_input/1`), and without the labels the flag "
            "would be a dead switch that looks switched.")
    if (getattr(args, "eval_cache", None) and getattr(args, "eval_every", 0)
            and not getattr(args, "eval_labels", None)):
        raise SystemExit(
            "[v3] ⛔ --max-speed-input with an in-training eval needs "
            "--eval-labels: the eval dataset must be fed the SAME ceiling "
            "channel as training, or every eval row scores a "
            "max-speed-conditioned model on windows that carry none.")


# ============================================================================
# ⭐⭐ THE EFFECTIVE-WEIGHT AUDIT — `_check_goal_point_args` MADE GENERAL
# ============================================================================
#
# `_check_goal_point_args` below refuses `--goal-point-inject --goal-point-w 0`
# because it would "build a head, stamp the edge, and train it on nothing".
# That judgement is right and it is PER-FLAG. This is the same judgement as a
# CONTRACT over every weight the parser accepts.
#
# ⚠ HONEST SCOPE, MEASURED 2026-09-06: this trainer has FIVE weight-like
# flags -- `--w-u0`, `--w-agent`, `--agent-w-project`, `--agent-w-ground`,
# `--goal-point-w` -- and ALL FIVE ARE ALREADY GATED by hand-written refusals.
# So this layer is NOT plugging a live hole in refc the way it is in
# `train_v6_staged.py`, and saying otherwise would be manufacturing a defect.
# It earns its place three other ways:
#   1. it FAILS EARLIER. `w_u0 > 0` with no `control_head` is caught today by
#      `assert_seams_are_built`, which runs AFTER the corpus mounts and the
#      model is built; this catches it in the preflight, in milliseconds -- the
#      `--gate-probes` lesson;
#   2. it makes the run record AUDITABLE. `config.json` stamps `w_agent` and
#      `w_u0` as NUMBERS today, with nothing saying whether the operator typed
#      them or whether the term builds a graph at all;
#   3. ⛔ it is EXHAUSTIVE OVER THE PARSER. `REFC_WEIGHT_GATES` must cover
#      every weight-like flag, and `tests/test_v6_effective_weights.py` fails
#      if a new one is added without a gate -- so the NEXT `--w-*` cannot ship
#      ungated. That contract is the durable half; the refusals are the cheap
#      half.
#
# ⛔ refc has NO stage layer, so the v6 `DISCARDED` verdict (an explicitly
# passed weight zeroed by `for_stage`) cannot arise here. The verdict that does
# arise is `NO_GRAPH`: a weight whose MODE GATE was never opened.

#: ⛔ EXHAUSTIVENESS CONTRACT — every weight-like flag, with the gate that
#: decides whether its loss term is ever reached, and the guard that already
#: covers it. ``gate`` returns ``(met, why-not)``.
REFC_WEIGHT_GATES: dict[str, dict] = {
    "w_u0": {
        "flag": "--w-u0", "term": "u0 control-space x0 loss (WP-4)",
        # `refc.py`: `control_head` is built IFF `sampler == "ddim"` (a
        # zero-init `nn.Linear(d, n_steps * 2)`); with `--sampler none` there
        # is no `u0_hat` in `out` and `compute_losses_v3` skips the term.
        "gate": lambda a: (str(getattr(a, "sampler", "none")) == "ddim",
                           "--sampler none: `control_head` is built only for "
                           "--sampler ddim, so `u0_hat` is absent from `out` "
                           "and the term is skipped"),
        "mask": None,
        "already": "assert_seams_are_built (w_u0>0 + no control_head) and "
                   "_pin_refcv5_seams (ddim + w_u0<=0) -- but both AFTER the "
                   "model build",
    },
    "w_agent": {
        "flag": "--w-agent", "term": "GT-supervised detection set loss (WP-6)",
        "gate": lambda a: (str(getattr(a, "agents", "off")) == "head"
                           and bool(getattr(a, "agent_join", None)),
                           "--w-agent needs `--agents head` AND `--agent-join` "
                           "(no seam => no `agent_slots` in `out`; no join => "
                           "no labels)"),
        "mask": None,
        "already": "_pin_refcv5_seams: the `--agents off` branch and the "
                   "`--agents head` w_agent/agent_join refusals",
    },
    "w_bev_aux": {
        "flag": "--w-bev-aux", "term": "WP-D BEV auxiliary loss",
        "gate": lambda a: (str(getattr(a, "bev_aux", "off")) != "off"
                           and bool(getattr(a, "agent_join", None)),
                           "--w-bev-aux needs `--bev-aux col|xcol` AND "
                           "`--agent-join` (no head => no `bev_logits` in "
                           "`out`; no join => no target)"),
        "mask": None,
        "already": "_pin_refcv5_seams: the `--bev-aux off` branch and the "
                   "`--bev-aux` w/join refusals",
    },
    # ⭐⭐ D-TACGOAL-TRAINER-SEAM-OPEN (2026-09-09). This row is HALF THE FIX.
    # MEASURED 2026-09-07 on the live refcv5-v2 argv: `tac_goal_tok_head`
    # took `p.grad is None` on BOTH tensors over a 40,284-step run -- 11,286
    # parameters built, stamped and unable to learn -- and BOTH standing
    # guards were green and NEITHER was wrong. `assert_seams_are_built` asks
    # *"is it built?"* (it was). This audit enumerates DECLARED LOSS WEIGHTS
    # and asks which build a graph -- and the head HAD NO WEIGHT FLAG, so it
    # produced NO ROW AT ALL. ⛔ An instrument that enumerates weights is
    # structurally blind to a head that has none, which is why giving the head
    # a real weight flag is not packaging around the fix: it IS part of it.
    "w_tac_goal": {
        "flag": "--w-tac-goal",
        "term": "D-TACGOAL 22-token tactical goal SET (multi-label BCE)",
        # `refc_v3.py:1015` builds the head IFF `_vv != "kin3" AND
        # cfg.tac_goal_tok_head`, and `:1320` is the only writer of
        # `cache["tac_goal_logits"]`; the target comes from the v7.2 join
        # (`V3Dataset.tac_goal_targets`), so BOTH are required.
        "gate": lambda a: (bool(getattr(a, "tac_goal_tok_head", False))
                           and bool(getattr(a, "v7_labels", None)),
                           "--w-tac-goal needs `--tac-goal-tok-head` (no head "
                           "=> no `tac_goal_logits` in `out`) AND `--v7-labels` "
                           "(no join => no `tac_goal_y`/`tac_goal_w` target)"),
        "mask": None,
        "already": "NOTHING, until 2026-09-09 -- this is the first weight this "
                   "head has ever had, and until it existed the head was "
                   "invisible to this audit by construction "
                   "(D-TACGOAL-TRAINER-SEAM-OPEN).",
    },
    "agent_w_project": {
        "flag": "--agent-w-project", "term": "agent projection consistency",
        "gate": lambda a: (str(getattr(a, "agents", "off")) != "off",
                           "--agents off: no AgentSeamConfig is built, so "
                           "`refc_agents.agent_losses` is never called"),
        "mask": None,
        "already": "_pin_refcv5_seams `--agents off` branch + _build_rig_camera",
    },
    "agent_w_ground": {
        "flag": "--agent-w-ground", "term": "agent ground-plane prior",
        "gate": lambda a: (str(getattr(a, "agents", "off")) != "off",
                           "--agents off: no AgentSeamConfig is built, so "
                           "`refc_agents.agent_losses` is never called"),
        "mask": None,
        "already": "_pin_refcv5_seams `--agents off` branch + _build_rig_camera",
    },
    "goal_point_w": {
        "flag": "--goal-point-w", "term": "E15 geometric goal point (GP-2)",
        "gate": lambda a: (bool(getattr(a, "goal_point_inject", False)),
                           "no --goal-point-inject: the head that predicts the "
                           "point is never built"),
        "mask": None,
        "already": "_check_goal_point_args (both directions)",
    },
    # ---- refcv6 §2/§6: the perception branch ---------------------------- #
    # ⛔ THESE ROWS ARE NOT BOILERPLATE. `test_v6_effective_weights.py::
    # test_REFC_WEIGHT_GATES_covers_every_weight_flag_the_parser_accepts` is
    # EXHAUSTIVE over the parser, so a `--w-*` flag added without a row here
    # FAILS THE SUITE -- which is what made me write them, and is the guard
    # working exactly as the `tac_goal_tok_head` post-mortem designed it.
    "w_map": {
        "flag": "--w-map", "term": "refcv6 SAM3 BEV map soft-CE (§2)",
        # THREE conditions, and each one has its own silent failure. No
        # `--trunk timm` => `fmap_s16` is None and the branch reads nothing.
        # No `--map-gt-root` => the BEV branch is built and never supervised.
        # No per-clip extrinsics => the lift back-projects through ONE road
        # plane for a corpus with 554 distinct mount heights.
        "gate": lambda a: (
            str(getattr(a, "trunk", "refc")) == "timm"
            and bool(getattr(a, "map_gt_root", None))
            and str(getattr(a, "agent_rig_camera", "off")) == "extrinsics",
            "--w-map needs `--trunk timm` (stride-16 map), `--map-gt-root` "
            "(the labels) AND `--agent-rig-camera extrinsics` (the per-clip "
            "lift geometry)"),
        "mask": None,
        "already": "_pin_refcv6_perception (all three, plus the REVERSE "
                   "refusal: an artifact supplied with a zero weight)",
    },
    "w_box3d": {
        "flag": "--w-box3d", "term": "refcv6 3-D cuboid set loss (§6)",
        # ⚠️ `--join3d` is deliberately NOT in this gate. Without it the arm
        # is the LEGAL 2-D rung whose total is exactly the 2-D set loss
        # (`box3d_set_loss`'s own test), so gating on it would refuse a real
        # experiment; `_pin_refcv6_perception` PRINTS the distinction instead.
        "gate": lambda a: (
            str(getattr(a, "trunk", "refc")) == "timm"
            and bool(getattr(a, "agent_join", None)),
            "--w-box3d needs `--trunk timm` (stride-16 tokens) AND "
            "`--agent-join` (the 2-D targets the Hungarian matches against)"),
        "mask": None,
        "already": "_pin_refcv6_perception",
    },
    # ---- refcv6 §4: the tactical behaviour decoder ---------------------- #
    # ⭐⭐ THE ROW THAT MAKES THE HEAD VISIBLE TO THIS AUDIT AT ALL. The
    # `tac_goal_tok_head` post-mortem is explicit that this instrument
    # ENUMERATES DECLARED LOSS WEIGHTS and is therefore *structurally blind to
    # a head that has none* — which is why `--w-tac-v6` is not packaging
    # around the fix, it IS part of it. Before this row existed, 2,262,020
    # parameters could be built and trained at zero gradient and every
    # standing guard stayed green.
    "w_tac_v6": {
        "flag": "--w-tac-v6",
        "term": "refcv6 §4 tactical behaviour decoder "
                "(22-token BCE + lat/lon CE + confidence)",
        # THREE conditions, each with its own silent failure. No decoder =>
        # no `tacv6_goal_logits` in the cache. No --v7-labels => no 22-token
        # target and no lat/lon class ids. No --agents => `agent_tokens` is
        # None and the hook's only live key/value source is absent, which
        # `refc.py:4146` refuses at the first forward rather than at launch.
        "gate": lambda a: (
            bool(getattr(a, "tac_decoder_v6", False))
            and bool(getattr(a, "v7_labels", None))
            and str(getattr(a, "agents", "off")) != "off",
            "--w-tac-v6 needs `--tac-decoder-v6` (no decoder => no "
            "`tacv6_goal_logits`), `--v7-labels` (no join => no 22-token "
            "target and no lat/lon class ids) AND `--agents` (the agent "
            "slots are the decoder's only live key/value source)"),
        "mask": None,
        "already": "_pin_refcv6_tactical — which refuses the CONVERSE too "
                   "(`--tac-decoder-v6` with a zero weight), the case this "
                   "audit cannot see because a zero weight is a legal value.",
    },
}


def refc_weight_specs(args) -> list[_ew.TermSpec]:
    """Every weight-like flag: declared -> gate -> effective."""
    specs: list[_ew.TermSpec] = []
    for dest, g in REFC_WEIGHT_GATES.items():
        w = float(getattr(args, dest, 0.0) or 0.0)
        met, why = g["gate"](args)
        specs.append(_ew.TermSpec(
            term=g["term"], flag=g["flag"], dest=dest,
            # ⛔ every refc weight defaults to 0.0 ON PURPOSE, so that adding a
            # seam to the code cannot change a run that does not ask for it.
            declared=0.0, requested=w, effective=w, layer=None,
            needs=(met, why), mask=g["mask"]))
    return specs


def effective_weight_rows_v3(args) -> tuple[list[_ew.WeightRow], str]:
    explicit = _ew.explicit_dests(getattr(args, "_ew_parser", None),
                                 getattr(args, "_ew_argv", None))
    src = _ew.SRC_ARGV if explicit is not None else _ew.SRC_UNAVAILABLE
    return _ew.classify_all(refc_weight_specs(args), explicit), src


def check_effective_weights(args) -> None:
    """⛔ REFUSE, AT START, a weight whose loss term can never be reached.

    ⚠ Called from BOTH :func:`preflight` and :func:`train`, which is this
    file's established idiom (`_check_nav_from_v7_args`, `_check_goal_point_args`
    and `_read_anchor_artifact` are all double-called) and NOT belt-and-braces:
    ``main`` runs ``preflight`` only under ``--preflight`` and otherwise goes
    straight to ``train``, so a guard called from ``preflight`` alone is wired
    into ONE launch path of two. Pinned by
    ``tests/test_v6_effective_weights.py::test_refc_guard_is_on_the_real_path``.
    """
    rows, _src = effective_weight_rows_v3(args)
    problems = _ew.refusals(rows, where="this arm")
    if problems:
        # ⚠ ASCII: a SystemExit message is written to stderr, and non-ASCII
        # is fatal on the cp1252 dev box -- a guard that crashes on its own
        # refusal text has refused nothing legible.
        raise SystemExit("[v3] REFUSED (effective-weight audit): "
                         + " | ".join(problems))


def effective_weights_stamp_v3(args, *, echo: bool = False) -> dict:
    """The ``config.json`` block; ``echo`` prints the table at launch."""
    rows, src = effective_weight_rows_v3(args)
    if echo:
        warn = _ew.unknown_explicitness_warning(rows)
        if warn:
            print(f"[v3] WARNING: {warn}", flush=True)
        print(_ew.render_table(rows, where="this arm", tag="v3"), flush=True)
    return _ew.stamp(rows, where="this arm", explicit_source=src)


def _check_goal_point_args(args) -> None:
    """⭐⭐ E15 (GP-2): refuse AT START a goal-point launch that would train,
    converge, and mean nothing.

    Four refusals, each naming the arm it would have manufactured:

    * ``--goal-point-geo-prior`` without ``--goal-point-inject`` — the S7 gate
      would be built with nothing to feed it (the head that predicts the point
      lives behind ``--goal-point-inject``), the ``r_terms`` entry would be
      skipped on every forward, and the arm would report **"the geometric goal
      prior is inert"** while never having had a goal. Same class as
      ``--agents head --w-agent 0``.
    * ``--goal-point-inject`` with ``--goal-point-w <= 0`` — a head that is
      built, stamped, and **supervised by nothing**. Its prediction would be
      whatever the zero-init projections drift to, and the pre-registered HEAD
      GATE (PREREG §5: RMS lateral <= 1.0 m, RMS range <= 2.0 m) would fail for
      a reason that is not about the goal form. A failure that is not about the
      lever is worse than no run.
    * ``--goal-point-t`` at or inside the scored horizon — this is the LEAK
      GUARD, and it is enforced STRUCTURALLY in
      ``GoalPointConfig.__post_init__``, which RAISES. It is re-checked here
      only so the message names the flag instead of surfacing as a dataclass
      traceback 400 lines later; the refusal itself is the constructor's.
    * ``--goal-point-w > 0`` without ``--goal-point-inject`` — a stamped weight
      whose loss term is silently skipped (the ``w_agent`` defect verbatim).

    ``getattr`` throughout: test rigs build partial Namespaces.
    """
    inject = bool(getattr(args, "goal_point_inject", False))
    geo = bool(getattr(args, "goal_point_geo_prior", False))
    w = float(getattr(args, "goal_point_w", GOAL_POINT_WEIGHT_DEFAULT))
    if geo and not inject:
        raise SystemExit(
            "[v3] ⛔ --goal-point-geo-prior needs --goal-point-inject: the S7 "
            "ranking term is fed by the E15 head's own prediction. Without "
            "the head the gate is built, the term is never appended, and the "
            "run would report the geometric seam as inert while it was never "
            "wired.")
    if w > 0.0 and not inject:
        raise SystemExit(
            f"[v3] ⛔ --goal-point-w {w} without --goal-point-inject: the "
            f"weight would be STAMPED in config.json while the loss term is "
            f"silently skipped, which is the `w_agent` defect verbatim.")
    if not inject:
        return
    if w <= 0.0:
        raise SystemExit(
            "[v3] ⛔ --goal-point-inject with --goal-point-w 0: this builds "
            "the E15 head, stamps the edge, and SUPERVISES IT WITH NOTHING. "
            "The head gate (PREREG §5: <= 1.0 m lateral / <= 2.0 m range RMS) "
            "would then fail for a reason that is not about the goal FORM, "
            "and PREREG §5 says a head-gate failure is NOT evidence about the "
            "form. Pass --goal-point-w 1.0 (the pre-registered value).")
    # The leak guard, surfaced with the flag's name on it. The REFUSAL lives in
    # GoalPointConfig.__post_init__ — this raises the same way, earlier.
    t_goal = float(getattr(args, "goal_point_t", 4.0))
    t_pred = float(gpm.GoalPointConfig().t_pred_s)
    if t_goal <= t_pred:
        raise SystemExit(
            f"[v3] ⛔⛔ --goal-point-t {t_goal} is at or inside the scored "
            f"horizon ({t_pred} s). A goal point inside the horizon IS THE "
            f"ANSWER, not a route signal. Refused here and refused again by "
            f"GoalPointConfig, so no arm that leaks the scored horizon can be "
            f"built at all.")
    try:
        gpm.goal_slot_index(t_goal, v3.V3_HORIZONS)
    except ValueError as e:
        raise SystemExit(
            f"[v3] ⛔ --goal-point-t {t_goal}: {e}. The goal and the anchor "
            f"must be compared AT THE SAME TIME; an approximate slot compares "
            f"two different horizons and still trains.") from e


class V3Dataset(RouteV21Dataset):
    #: clip-stable-id -> V7Label, or None for the kin3 path. Set by the trainer
    #: rather than passed through the ctor, because the base class owns the
    #: signature and widening it would touch every RouteV21 consumer.
    v7_by_sid: dict | None = None
    v7_dt: float = 0.1
    #: --w-tac-goal (D-TACGOAL-TRAINER-SEAM-OPEN): when True every item carries
    #: ``tac_goal_y``/``tac_goal_w`` ``[22]`` from
    #: :func:`v7_labels.tactical_goal_targets`. ⛔ Default False keeps the batch
    #: BYTE-IDENTICAL for a recipe that does not ask for the term -- the same
    #: discipline as ``nav_from_v7`` above, and the reason the OFF path can be
    #: proven bit-identical to the pre-wiring trainer at a fixed seed.
    tac_goal_targets: bool = False
    #: ⭐ refcv6 §2b: emit ``pose_hist``, the OBSERVED window's ego track.
    #: ⛔ Default False keeps the item's KEY SET byte-identical to the base
    #: contract — pinned by ``test_refc_v3_u8_batches.py``, which compares the
    #: emitted keys against ``_contract.py``'s. The gate is per-DATASET (set
    #: once at construction), never per-ROW: a key that appeared on only some
    #: rows would make the loss-time refusal fire at random instead of at
    #: launch, which is the ``nav_args`` lesson.
    ego_history: bool = False
    #: ⛔ NOT a literal in the loss. The negative policy is a property of THE
    #: LOADED SPLIT (`v7_labels.tactical_goal_targets.__doc__`: 3,574 of 4,572
    #: clips were never ASKED the traffic-light question, so their absence is
    #: NOT-PROBED rather than negative). Surfaced as ``--tac-goal-negatives``
    #: so the cheap arm is comparable rather than hidden.
    tac_goal_negatives: str = "measured"
    #: ⭐ PI 2026-09-16. ``None`` = the provenance policy above. Set only from
    #: ``--cot-negative-sidecar``, and the two must agree: the loader refuses a
    #: sidecar without the policy and the policy without a sidecar, so this
    #: field can never be a permission the loss quietly ignored.
    cot_negative_sidecar = None
    #: --nav-from-v7 (E-ARCH-NAVSRC-1): when True, ``nav_cmd``/``nav_valid``
    #: are OVERRIDDEN per window from the clip's v7.2 ``nav_command`` token
    #: (see :meth:`enable_nav_from_v7`). ⛔ Default False keeps the v1
    #: derivation (``refb_labels.nav_command`` via ``FailLoudWindowDataset``)
    #: byte-identical — the live run resumes through this class.
    nav_from_v7: bool = False
    v7_manifest = None
    _nav_by_sid: dict | None = None
    nav_from_v7_stats: dict | None = None
    #: --nav-args (D-GSTR-1 P3): the nav token's CONTINUOUS ARGS, which this
    #: loader has never read. ``_nav_args_by_sid[sid] = (distance_m, time_s,
    #: valid)`` in RAW UNITS (metres, seconds); ``nav_arg_stats`` is the
    #: FIT-SPLIT normaliser and MUST be handed to the eval dataset from the
    #: TRAIN one — fitting it on the scored split is the 2026-08-22 ridge
    #: failure in a new costume.
    nav_args_enabled: bool = False
    _nav_args_by_sid: dict | None = None
    nav_arg_stats = None
    nav_args_report: dict | None = None
    #: --max-speed-input (E16): the map/nav posted-limit ceiling.
    #: ``_max_speed_by_sid[sid] = (v_max_ms, valid)`` in the RAW SHIPPED
    #: value's units (m/s). ⛔ RAW, NOT PRE-QUANTIZED -- see
    #: :meth:`enable_max_speed`. Default False keeps every banked arm's
    #: recipe byte-identical.
    max_speed_enabled: bool = False
    _max_speed_by_sid: dict | None = None
    max_speed_report: dict | None = None
    #: ``obstacle.offline`` agent join (refcv5 WP-6 / ``E-AGT-HEAD``), set by
    #: :meth:`enable_agent_join`. While it is None the batch carries NO
    #: ``agent_box`` and ``--w-agent > 0`` REFUSES in ``compute_losses_v3`` --
    #: which is the guard that caught an arm training no detector at all.
    agent_join = None
    #: ``n_pad`` for the padded target block. 0 = the reader's own MEASURED
    #: ``max_agents_per_frame``, so nothing is truncated before ``match_slots``
    #: applies its own counted nearest-N policy. Fixing it here instead would
    #: hide drops the loss is supposed to report.
    agent_pad: int = 0
    agent_join_stats: dict | None = None
    #: WP-D. ``PolarBEVSpec | None`` -- while it is None the batch carries NO
    #: ``bev_occ``/``bev_mask`` and ``--w-bev-aux > 0`` REFUSES at loss time,
    #: for exactly the reason the ``agent_join`` guard above exists.
    bev_spec = None
    bev_occlusion: str = "mask"
    #: ⭐⭐ refcv6 §2/§6 -- THE SAM3 MAP GT, set by :meth:`enable_map_gt`.
    #: While it is None the batch carries NO ``map_frac``/``map_seen`` and
    #: ``--w-map > 0`` REFUSES at loss time, for exactly the reason the
    #: ``agent_join`` guard above exists. ``map_clip_of_ep`` is the
    #: ``stable_episode_id -> clip_id`` table: ``LazyV2Episode`` carries only
    #: the id, and :class:`MapGTStore` resolves files by clip id / sha12.
    map_store = None
    map_clip_of_ep: dict | None = None
    #: The cache's OWN ``n_stack``, read from the v2 manifest. ⛔ NOT a
    #: literal 3: ``semantic_map_gt.raw_frame_index`` adds ``n_stack - 1`` to
    #: convert a stacked-row index into the RAW v2ep frame the labels are
    #: indexed by, and a wrong value shifts EVERY window's map by that many
    #: frames -- 1.7 m at 30 km/h, inside the tolerance of nothing and visible
    #: in no metric (``perception_targets.assert_frame_alignment.__doc__``).
    map_n_stack: int = 0
    map_stats: dict | None = None
    #: ``agent_cuboid_gt.AgentJoin3D | None`` (refcv6 §6), set by
    #: :meth:`enable_join3d`. It WIDENS the 2-D agent block with ``cz``/``h``
    #: and a MASK; with it None the mask is all-False and ``box3d_set_loss``
    #: reports ``n["z"] == 0`` rather than training on zeros.
    join3d = None
    join3d_stats: dict | None = None

    """RouteV21Dataset + clamped/masked 6 s future + E4.1 tactical goals.

    ``max_horizon`` MUST stay at the caller's 20: enumeration parity. The
    extended future is fetched here per item from the episode's own poses."""

    def enable_nav_from_v7(self, manifest) -> dict:
        """Switch THIS dataset's ``nav_cmd`` INPUT to the clip's v7.2 token.

        Per-clip constant (the release's ``t0_constant`` semantics — token
        and args are window-independent, ``v7_labels.NavEmitter``), resolved
        once here so ``__getitem__`` is a lookup. Requires the v7.2 join
        (``v7_by_sid``) and the loaded labels' ``LabelManifest`` carrying
        ``allow_oracle_nav=True``: ``v7_labels.oracle_nav`` is the ONLY route
        to the token and it checks the MANIFEST, so the permission and the
        run's recorded config cannot disagree. Returns the per-clip stats
        that go into config.json — the token distribution over THIS
        dataset's clips and the count of clips without a record.
        """
        if self.v7_by_sid is None:
            raise ValueError(
                "[v3] --nav-from-v7 needs the v7.2 join (v7_by_sid is None) — "
                "the trainer sets it from --v7-labels / --eval-labels first")
        if not getattr(manifest, "allow_oracle_nav", False):
            raise v7l.OracleNavRefused(
                "[v3] --nav-from-v7 reads an ORACLE nav (provenance ego-future) "
                "— load_v7_labels(..., allow_oracle_nav=True) is required so "
                "the manifest carries the stamp")
        assert_nav_token_alignment()
        nav_by_sid: dict[int, int] = {}
        args_by_sid: dict[int, tuple] = {}
        for sid, lab in self.v7_by_sid.items():
            nav = v7l.oracle_nav(lab, manifest) or {}      # ⭐ the gate
            tok = nav.get("token")
            if tok not in NAV_TOKEN_TO_LEGACY:
                raise ValueError(
                    f"[v3] ⛔ clip {lab.clip_id!r}: nav token {tok!r} has no "
                    f"legacy mapping (known: {sorted(NAV_TOKEN_TO_LEGACY)}) — "
                    f"vocabulary drift is a different experiment, refused")
            nav_by_sid[sid] = refb.NAV_COMMANDS.index(NAV_TOKEN_TO_LEGACY[tok])
            # ⭐ D-GSTR-1 P3 — THE ARGS, WHICH THIS LINE USED TO THROW AWAY.
            # ⛔ VALIDITY IS READ FROM THE KEY'S PRESENCE, NEVER FROM THE
            # VALUE. `NavEmitter._args_for_window` defaults a missing slot to
            # 0.0, so `distance_m == 0.0` is AMBIGUOUS: it is either "the turn
            # is 0 m away" or "there is no turn". MEASURED: `NAV_FOLLOW_ROAD`
            # carries `args: {}` on 2,897/2,897 train and 96/96 eval records,
            # and 69.97 % of train records present distance 0.0 to a consumer.
            # Presence of the KEY is the only signal that separates the two.
            _a = nav.get("args") or {}
            _ok = ("distance_m" in _a) and ("time_s" in _a)
            args_by_sid[sid] = (float(_a.get("distance_m", 0.0)),
                                float(_a.get("time_s", 0.0)),
                                1.0 if _ok else 0.0)
        counts = {name: 0 for name in NAV_TOKEN_TO_LEGACY.values()}
        missing = 0
        for ep in self.episodes:
            idx = nav_by_sid.get(int(ep.episode_id))
            if idx is None:
                missing += 1
            else:
                counts[refb.NAV_COMMANDS[idx]] += 1
        n = len(self.episodes)
        if n and missing == n:
            raise ValueError(
                f"[v3] ⛔ --nav-from-v7 joined ZERO of {n} clips — wrong label "
                f"blob for this corpus (md5={manifest.md5})")
        self._nav_by_sid = nav_by_sid
        self._nav_args_by_sid = args_by_sid
        self.v7_manifest = manifest
        self.nav_from_v7 = True
        self.nav_from_v7_stats = {
            "n_clips": n, **counts, "missing": missing,
            "derivation": NAV_FROM_V7_DERIVATION,
            "unlabeled_default": "nav_cmd=0 ('follow') + nav_valid=False "
                                 "(refav1_loader convention)",
            "label_md5": manifest.md5,
            "allow_oracle_nav": bool(manifest.allow_oracle_nav)}
        # ASCII-only by design: this prints at dataset init.
        print(f"[v3] nav_from_v7: follow {counts['follow']} / left "
              f"{counts['left']} / right {counts['right']}, missing {missing} "
              f"(of {n} clips; md5={manifest.md5})", flush=True)
        return self.nav_from_v7_stats

    # ---- ⭐⭐ E16: the MAX-SPEED ceiling ---------------------------------

    def enable_max_speed(self, manifest, mode: str = msi.DEFAULT_MODE
                         ) -> dict:
        """Turn the map/nav posted-limit ceiling ON for this dataset.

        ⛔⛔ THE VALUE THIS SHIPS IS THE RECORD'S OWN ``v_max_ms``, RAW, AND
        THE LADDER IS APPLIED EXACTLY ONCE -- inside
        ``max_speed_input.encode_block``, on the model side, under the
        arm's ``mode``. Quantizing here as well is not a harmless
        belt-and-braces: the record's ``v_max_bucket_ms`` is rounded to
        4 dp (13.8889) while the ladder's 50 km/h step is 13.888888..., so
        the shipped bucket is strictly GREATER than the step it names and
        snaps UP to the next one. MEASURED on the v8 train blob: feeding
        the shipped bucket back through the ladder moves **2,631 of 4,572
        clips (57.5 %) one step up** (50->70 on 1,593, 20->30 on 809,
        100->120 on 229). That is the same mechanism that manufactured
        2,203 phantom violations from a re-derived ``v_hi``: a rounding
        difference in the third decimal, read as a different quantity.

        ⛔ UNITS ARE REQUIRED, NEVER ASSUMED. ``read_max_speed_field``
        raises on a payload that declares none; that raise is re-raised
        here as a ``SystemExit`` NAMING THE FIELD and the clip, because
        m/s vs km/h vs mph is a 1.61x spread and this programme published
        a 396 g anchor table from exactly this error.

        ⭐ THE CONTENT ASSERTION. The module's pinned ladder and the
        record's own ``bucket_steps_kmh`` must agree, and the module's
        snap of the shipped ``v_max_ms`` must reproduce the record's
        ``v_max_bucket_kmh`` on EVERY clip. MEASURED 2026-09-06: 4,572/4,572
        train and 147/147 eval, 0 mismatches. A disagreement means the
        DataFlyWheel re-pinned the ladder under us, and the run REFUSES
        rather than training on two different quantizations at once.

        Returns the census that goes into ``config.json``.
        """
        if mode not in msi.MODES:
            raise SystemExit(f"[v3] ⛔ --max-speed-mode must be one of "
                             f"{msi.MODES}, got {mode!r}")
        if self.v7_by_sid is None:
            raise SystemExit(
                "[v3] ⛔ --max-speed-input needs the label join "
                "(v7_by_sid is None) — the trainer sets it from "
                "--v7-labels / --eval-labels first")
        by_sid: dict[int, tuple] = {}
        n_rec = n_block = n_over = 0
        for sid, lab in self.v7_by_sid.items():
            n_rec += 1
            smi = v7l.oracle_max_speed(lab, manifest)   # ⭐ the oracle gate
            if not smi:
                by_sid[sid] = (0.0, 0.0)
                continue
            n_block += 1
            # ⛔ THE LADDER THE RECORD SHIPPED MUST BE THE LADDER WE PIN.
            steps = tuple(smi.get("bucket_steps_kmh") or ())
            if steps and steps != msi.POSTED_LIMIT_STEPS_KMH:
                raise SystemExit(
                    f"[v3] ⛔ clip {lab.clip_id!r}: the record's "
                    f"`speed_max_input.bucket_steps_kmh` is {list(steps)} "
                    f"but this build pins "
                    f"{list(msi.POSTED_LIMIT_STEPS_KMH)}. Two ladders is "
                    f"two experiments; refusing rather than quantizing "
                    f"the corpus two different ways.")
            try:
                got = msi.read_max_speed_field(smi, mode="quantized")
            except msi.MaxSpeedUnitsError as exc:
                raise SystemExit(
                    f"[v3] ⛔ --max-speed-input: clip {lab.clip_id!r}'s "
                    f"`speed_max_input` block carries a max speed and its "
                    f"UNITS cannot be established. Refusing to guess — m/s "
                    f"vs km/h vs mph is a 1.61x spread. {exc}") from None
            if not got["valid"]:
                by_sid[sid] = (0.0, 0.0)
                continue
            # ⭐ THE CONTENT ASSERTION, against the SHIPPED bucket.
            want = smi.get("v_max_bucket_kmh")
            if want is not None:
                have = round(float(got["quantized_ms"]) * 3.6)
                if have != int(want):
                    raise SystemExit(
                        f"[v3] ⛔ clip {lab.clip_id!r}: this build snaps "
                        f"v_max_ms={smi.get('v_max_ms')} to {have} km/h but "
                        f"the record shipped {int(want)} km/h. The channel "
                        f"is scored against the SHIPPED value; a "
                        f"disagreement is a re-derivation, which is what "
                        f"manufactured 2,203 phantom violations.")
            n_over += int(bool(got["over_ceiling"]))
            # ⛔ RAW m/s, in the record's declared units. The mode lives on
            # the MODEL (`cfg.max_speed_cfg.mode`); see the docstring.
            by_sid[sid] = (float(got["raw_ms"]), 1.0)
        n_win = n_win_valid = 0
        for (e_i, _t) in self.index:
            v = by_sid.get(int(self.episodes[e_i].episode_id))
            n_win += 1
            if v is not None and v[1] > 0.5:
                n_win_valid += 1
        if n_win and n_win_valid == 0:
            raise SystemExit(
                f"[v3] ⛔ --max-speed-input: NOT ONE of this split's "
                f"{n_win} windows receives a `speed_max_input` value. The "
                f"field is a v8 addition — the v7.2 release carries it on "
                f"0/4,572 records — so this is almost certainly a v7.2 "
                f"label blob (md5={manifest.md5}). The channel would be a "
                f"constant invalid pad and the arm would measure it as "
                f"noise. Refusing rather than feeding nothing.")
        self._max_speed_by_sid = by_sid
        self.max_speed_enabled = True
        self.max_speed_report = {
            **msi.artifact_meta(mode),
            "label_md5": manifest.md5,
            "allow_oracle_nav": bool(manifest.allow_oracle_nav),
            "quantized_by": "tanitad.refs.max_speed_input.encode_block "
                            "(model side, ONCE); the loader ships the RAW "
                            "shipped v_max_ms",
            "bucket_cross_check": "module snap == record v_max_bucket_kmh "
                                  "on every clip (refused otherwise)",
            "n_clips": n_rec, "n_clips_with_block": n_block,
            "n_clips_over_ceiling": n_over,
            "n_windows": n_win, "n_windows_with_ceiling": n_win_valid,
            #: ⭐ THE NUMBER THAT DECIDES THE CLAIM. Quote this before
            #: saying the channel carries information.
            "window_ceiling_frac": round(n_win_valid / max(n_win, 1), 4),
        }
        print(f"[v3] max_speed_input ({mode}): {n_block}/{n_rec} clips "
              f"carry a ceiling, {n_win_valid}/{n_win} windows fed, "
              f"{n_over} over the 130 km/h top step (md5={manifest.md5})",
              flush=True)
        return self.max_speed_report

    # ---- ⭐⭐ refcv6 §5: the PI's FOUR-VALUE set-speed, from the SIDECAR --

    def enable_max_speed_v6(self, sidecar_path: str, manifest=None) -> dict:
        """Turn the 4-way one-hot set-speed ON for this dataset.

        ⛔⛔ A DIFFERENT SOURCE FIELD FROM :meth:`enable_max_speed`, and that
        is the whole reason it is a second method. E16 reads the v8
        ``speed_max_input`` block (a POSTED LIMIT, 8-step ladder
        {20,30,50,70,80,100,120,130}); this reads
        ``g_tac.goals.SPEED_BAND.v_hi_ms`` (the ego's own realised maximum over
        ``[t0+2 s, +6 s]``, 4-step ladder {30,50,100,120}). Same-looking
        quantity, different provenance, different ladder — and
        ``RefCV3Model.__init__`` refuses the pair precisely so an arm cannot
        feed both and become non-attributable.

        ⛔ IT SHIPS THE RAW ``v_hi_ms`` AND THE LADDER IS APPLIED ONCE, on the
        model side, in ``MaxSpeedOneHotEncoder`` — the E16 rule, whose own
        docstring records the 2,631-of-4,572 (57.5 %) step shift that
        double-quantizing produced.

        ⛔ THE WINDOW COVERAGE IS THE NUMBER THAT DECIDES WHETHER THE CHANNEL
        CARRIES ANYTHING, so a split where NO window is fed REFUSES here
        rather than training a constant all-zero pad and reporting it as a
        measured channel.
        """
        by_sid, report = v6ms.read_speed_max_sidecar_v6(
            str(sidecar_path),
            label_md5=(getattr(manifest, "md5", None) if manifest else None))
        n_win = n_win_valid = 0
        for (e_i, _t) in self.index:
            v = by_sid.get(int(self.episodes[e_i].episode_id))
            n_win += 1
            if v is not None and v[1] > 0.5:
                n_win_valid += 1
        if n_win and n_win_valid == 0:
            raise SystemExit(
                f"[v3] ⛔ --max-speed-input-v6: NOT ONE of this split's "
                f"{n_win} windows joins to a sidecar row with a ceiling. The "
                f"sidecar has {report['n_valid']} valid rows, so this is a "
                f"JOIN failure, not an empty sidecar — most likely a sidecar "
                f"built over a different split. The channel would be a "
                f"constant all-zero pad and the arm would measure it as "
                f"noise. Refusing rather than feeding nothing.")
        self._max_speed_by_sid = by_sid
        self.max_speed_enabled = True
        self.max_speed_report = {
            **report,
            "channel": "refcv6 4-way one-hot (§5)",
            "label_md5": getattr(manifest, "md5", None) if manifest else None,
            "n_windows": int(n_win),
            "n_windows_fed": int(n_win_valid),
            # ⭐ THE number: the fraction of WINDOWS that receive a real
            # ceiling. A per-CLIP coverage of 100 % can still be a low window
            # coverage if the join misses, and the window figure is the one
            # the model actually experiences.
            "window_ceiling_frac": round(n_win_valid / max(n_win, 1), 6),
        }
        print(f"[v3] max_speed_input_v6 (4-way one-hot): "
              f"{report['n_valid']}/{report['n_rows']} sidecar rows carry a "
              f"ceiling, {n_win_valid}/{n_win} windows fed "
              f"({100.0 * n_win_valid / max(n_win, 1):.1f} %), "
              f"{report['n_over_ceiling']} clips ABOVE the 120 km/h top step "
              f"(clamped, and on those the fed ceiling is one the ego "
              f"demonstrably exceeded)", flush=True)
        return self.max_speed_report

    # ---- D-GSTR-1 P3: the nav command's CONTINUOUS ARGS ------------------

    def enable_nav_args(self, stats=None) -> dict:
        """Turn the ``distance_m`` / ``time_s`` channel ON for this dataset.

        ⛔ ``stats`` IS THE FIT SPLIT'S NORMALISER AND THE EVAL DATASET MUST BE
        HANDED THE TRAIN ONE. Passing ``None`` FITS on this dataset's own
        clips, which is correct exactly once — on the train split. An eval
        dataset that fits its own statistics is the 2026-08-22 ridge-probe
        failure wearing a loader costume: a statistic fitted on the data it
        will later score makes the channel look more informative than it is.
        The trainer therefore fits once and passes the object down; this
        method REFUSES to fit twice by recording which happened.

        ⛔ FIT ON THE **VALID** ROWS ONLY. `NAV_FOLLOW_ROAD` presents 0.0/0.0
        with `valid = 0`, and on our corpus that is the MAJORITY (2,897/4,572
        train records). Folding those zeros into the mean/std would compress
        the real turn distances toward the pad value and hand the model a
        normaliser fitted mostly on padding.

        Returns the census that goes into ``config.json`` — including THE
        NUMBER THAT DECIDES WHETHER THIS CHANNEL CARRIES INFORMATION: the
        fraction of this dataset's WINDOWS that receive a real distance.
        """
        from tanitad.models.nav_conditioning import NavArgStats
        if not self.nav_from_v7 or self._nav_args_by_sid is None:
            raise ValueError(
                "[v3] --nav-args needs --nav-from-v7: the args live on the "
                "v7.2 nav_command record and there is no other supplier. "
                "Refusing rather than feeding a zero channel.")
        fitted_here = stats is None
        if fitted_here:
            d = [v[0] for v in self._nav_args_by_sid.values() if v[2] > 0.5]
            t = [v[1] for v in self._nav_args_by_sid.values() if v[2] > 0.5]
            if not d:
                raise ValueError(
                    "[v3] ⛔ --nav-args: NOT ONE clip in this split carries "
                    "`distance_m`/`time_s`. The channel would be a constant "
                    "pad and the arm would measure it as noise. Refusing.")
            stats = NavArgStats.from_fit_split(d, t)
        self.nav_arg_stats = stats
        self.nav_args_enabled = True
        # ---- the census, per CLIP and per WINDOW ------------------------
        n_clip = n_clip_valid = 0
        n_win = n_win_valid = 0
        for ep in self.episodes:
            v = self._nav_args_by_sid.get(int(ep.episode_id))
            n_clip += 1
            if v is not None and v[2] > 0.5:
                n_clip_valid += 1
        for (e_i, _t) in self.index:
            v = self._nav_args_by_sid.get(int(self.episodes[e_i].episode_id))
            n_win += 1
            if v is not None and v[2] > 0.5:
                n_win_valid += 1
        self.nav_args_report = {
            "slots": ["distance_norm", "time_norm", "args_valid"],
            "raw_units": ["distance_m:metres", "time_s:seconds"],
            "semantics": "t0_constant (v7_labels.NavEmitter's default; the "
                         "clip's t0 values are constant across its windows)",
            "normaliser": stats.to_dict(),
            "normaliser_fitted_here": bool(fitted_here),
            "n_clips": n_clip, "n_clips_with_real_args": n_clip_valid,
            "clip_valid_frac": round(n_clip_valid / max(n_clip, 1), 4),
            "n_windows": n_win, "n_windows_with_real_args": n_win_valid,
            #: ⭐ THE NUMBER THAT DECIDES THE CLAIM. Quote this, not the clip
            #: fraction, before saying the channel carries information.
            "window_real_distance_frac": round(n_win_valid / max(n_win, 1), 4),
            "_reads": ("args_valid=0 rows carry pad values that the MODEL "
                       "re-zeroes (refc_v3.py E13b); the bit itself always "
                       "passes, so 'no range known' stays a distinct input."),
        }
        print(f"[v3] nav_args: {n_clip_valid}/{n_clip} clips and "
              f"{n_win_valid}/{n_win} windows carry a REAL distance "
              f"({self.nav_args_report['window_real_distance_frac']}); "
              f"normaliser fitted_here={fitted_here} "
              f"n_fit={stats.n_fit}", flush=True)
        return self.nav_args_report

    # ---- refcv5 WP-6: obstacle.offline -> the batch ---------------------
    def enable_agent_join(self, reader, pad: int = 0,
                          allow_legacy_ids: bool = False) -> dict:
        """Attach the ``obstacle.offline`` join so ``__getitem__`` emits the
        ``agent_*`` target block ``compute_losses_v3`` consumes.

        ``reader`` is a ``train_p8_occupancy.JoinFileReader`` -- the SAME
        reader ``build_obstacle_join.py``'s output was verified through. No
        second label path is built here; this method is a join and a census.

        WHAT IT REFUSES, AND WHY EACH REFUSAL EXISTS
        --------------------------------------------
        * **Zero joined clips** -- the wrong join for this corpus. The same
          shape as ``--nav-from-v7``'s "joined ZERO of N clips": a flag that
          silently supervises nothing is worse than a crash.
        * **Only LEGACY-id matches** -- ``JoinFileReader`` accepts the 63-bit
          stable id AND the legacy 16-bit (first-4-BYTES-of-the-clip-id) key.
          The legacy key COLLIDES (MEASURED on this join: 34 of 2,308 clips
          share a prefix), so an episode that is NOT in the join can still
          match one that is, and the head would then be trained on ANOTHER
          CLIP'S agents while every count looked healthy. That is a label
          corruption no downstream metric could attribute, so it must be asked
          for BY NAME (``allow_legacy_ids=True``) and it is stamped into
          config.json.

        Returns the stats that go into config.json -- including the NOW-frame
        label coverage over this dataset's OWN window index, which is the
        number that decides how much supervision the run actually gets, and is
        NOT the same as clip coverage.
        """
        eps_ids = [int(e.episode_id) for e in self.episodes]
        n_stable = sum(1 for i in eps_ids if i in reader._clip_of_uid)
        n_legacy = sum(1 for i in eps_ids
                       if i not in reader._clip_of_uid
                       and i in reader._clip_of_legacy)
        n_ambig = len(reader._ambiguous_legacy)
        if (n_stable + n_legacy) == 0:
            raise SystemExit(
                f"[v3] REFUSING: the agent join covers ZERO of "
                f"{len(eps_ids)} episodes ({reader.path}). That is the wrong "
                f"join for this corpus -- a run would stamp the flag and "
                f"train no detector.")
        if n_stable == 0 and not allow_legacy_ids:
            raise SystemExit(
                f"[v3] REFUSING: the agent join matches this corpus ONLY "
                f"through the LEGACY 16-bit episode id ({n_legacy}/"
                f"{len(eps_ids)} episodes; {n_ambig} ids are ambiguous WITHIN "
                f"the join itself). That key is the first 4 BYTES of the clip "
                f"id and collides, so an episode absent from the join can "
                f"match a different clip that is present, and the detector "
                f"would train on another clip's agents. Pass "
                f"--agent-join-allow-legacy-ids to accept that risk by name "
                f"(it is stamped into config.json), or use a v2 cache whose "
                f"providers carry stable ids.")

        self.agent_join = reader
        self.agent_pad = int(pad or reader.max_agents_per_frame)
        if self.agent_pad <= 0:
            raise SystemExit("[v3] REFUSING: the agent join carries no agents "
                             "at all (max_agents_per_frame == 0).")

        # NOW-frame coverage over THIS dataset's own window index. Clip
        # coverage is not supervision coverage: the join's label span ends
        # ~20 s in, and a window whose NOW frame is past it is NO_LABEL.
        w = self.window
        n_lab = n_clear = 0
        n_boxes = 0
        for e_i, t in self.index:
            ep = self.episodes[e_i]
            ag = reader.lookup(int(ep.episode_id), t + w - 1)
            if ag is None:
                continue
            n_lab += 1
            n_boxes += int(ag.shape[0])
            if ag.shape[0] == 0:
                n_clear += 1
        n_win = len(self.index)
        self.agent_join_stats = {
            "path": str(reader.path),
            "n_episodes": len(eps_ids),
            "n_episodes_joined_stable_id": n_stable,
            "n_episodes_joined_legacy_id": n_legacy,
            "n_ambiguous_legacy_ids_in_join": n_ambig,
            "id_space": ("stable-63bit" if n_legacy == 0 else
                         ("mixed" if n_stable else "legacy-16bit-COLLIDING")),
            "allow_legacy_ids": bool(allow_legacy_ids),
            "n_windows": n_win,
            "n_windows_labelled": n_lab,
            "frac_windows_labelled": round(n_lab / max(n_win, 1), 4),
            "n_windows_labelled_clear": n_clear,
            "n_target_boxes_prefilter": n_boxes,
            "agent_pad": int(self.agent_pad),
            "reader_max_agents_per_frame": int(reader.max_agents_per_frame),
            "reader_n_records": int(reader.n_records),
            "reader_n_clips": int(reader.n_clips),
            "reader_has_classes": bool(reader.has_classes),
            "reader_has_occlusion_flags": bool(reader.has_occlusion_flags),
            "reader_with_rates": bool(getattr(reader, "with_rates", False)),
        }
        if n_lab == 0:
            raise SystemExit(
                f"[v3] REFUSING: the agent join covers "
                f"{n_stable + n_legacy}/{len(eps_ids)} episodes but ZERO of "
                f"{n_win} window NOW-frames. Every window would be NO_LABEL "
                f"and the detection loss would never be computed -- the "
                f"silent-skip failure with the labels present.")
        print("[v3] agent join: %d/%d episodes (%s), %d/%d windows labelled "
              "(%.1f %%), %d prefilter target boxes, pad %d"
              % (n_stable + n_legacy, len(eps_ids),
                 self.agent_join_stats["id_space"], n_lab, n_win,
                 100.0 * n_lab / max(n_win, 1), n_boxes, self.agent_pad),
              flush=True)
        return self.agent_join_stats

    def _agent_item(self, ep, f: int) -> dict:
        """The padded target block for ONE window, at its NOW frame ``f``.

        NO_LABEL and LABELLED-CLEAR are DIFFERENT STATES and are emitted
        differently: both carry ``valid`` all-False, but ``agent_label`` is
        False for NO_LABEL and True for labelled-clear. Collapsing them would
        train the head that an unlabelled frame is an empty road -- the
        ``"no agents"`` defect the join doc calls out by name.

        Targets are emitted RAW (no visibility filter here). The filter lives
        in ``refc_agents.agent_losses`` where it can be turned OFF by name for
        the deliberate-regression arm; filtering at the dataset would delete
        that control.
        """
        r = self.agent_join
        eid = int(ep.episode_id)
        pad = int(self.agent_pad)
        ag = r.lookup(eid, int(f))
        has = ag is not None
        if not has:
            ag = _np.zeros((0, 6), dtype=_np.float64)
            cls = rates = rmask = None
        else:
            cls = r.lookup_classes(eid, int(f))
            rr = r.lookup_rates(eid, int(f))
            rates, rmask = (rr if rr is not None else (None, None))
        n_raw = int(ag.shape[0])
        if n_raw > pad:
            # Should be impossible when pad came from the reader's own
            # measured max; kept as a COUNTED nearest-N truncation rather than
            # an exception so a caller-supplied --agent-pad degrades the way
            # match_slots does, visibly.
            order = _np.argsort(_np.hypot(ag[:, 0], ag[:, 1]))[:pad]
            ag = ag[order]
            if cls is not None:
                cls = _np.asarray(cls)[order]
            if rates is not None:
                rates, rmask = _np.asarray(rates)[order], \
                    _np.asarray(rmask)[order]
        t = _agent_slots.targets_from_join(ag, classes=cls, rates=rates,
                                           rates_mask=rmask, n_pad=pad)
        # ⭐ THE EPISODE ID TRAVELS WITH THE TARGETS. Without it a
        # per-clip RigCamera cannot be resolved at loss time and the
        # run silently falls back to ONE mount pose for a corpus whose
        # MEASURED mount height spans 1.245-1.607 m (37 distinct
        # values in 40 clips). `agent_ep` is the join key the loss
        # uses; it is the SAME id the join itself was matched on.
        item = {"agent_ep": torch.tensor(eid, dtype=torch.long),
                "agent_box": t["box"][0], "agent_yaw": t["yaw"][0],
                "agent_cls": t["cls"][0], "agent_valid": t["valid"][0],
                "agent_occ": t["occ"][0], "agent_rates": t["rates"][0],
                "agent_rates_mask": t["rates_mask"][0],
                "agent_label": torch.tensor(bool(has)),
                "agent_n_raw": torch.tensor(int(n_raw), dtype=torch.long),
                "agent_n_truncated": torch.tensor(max(n_raw - pad, 0),
                                                  dtype=torch.long)}
        # ⭐ WP-D. Built from the RAW join rows (``ag``, pre-truncation and
        # pre-visibility-filter), not from the padded slot block: the slot
        # targets are a DETECTION parameterisation with a fixed query budget,
        # and rasterising them would silently drop whatever `--agent-pad`
        # truncated. `labelled=has` keeps NO_LABEL and labelled-clear apart --
        # both give an all-zero occupancy and they differ ONLY in the mask.
        if self.bev_spec is not None:
            _occ, _msk = _bev_aux.build_target(
                ag, labelled=bool(has), spec=self.bev_spec,
                occlusion=self.bev_occlusion)
            item["bev_occ"] = torch.from_numpy(_occ)
            item["bev_mask"] = torch.from_numpy(_msk)
        # ---- refcv6 §6: WIDEN the 2-D block to 3-D, by TRACK ID ----------- #
        # ⛔ BY TRACK ID, never by position. `zh_for_frame` is the module's own
        # designed call and it aligns on the id, so a filter that ever reorders
        # either file cannot silently attach agent k's height to agent k+1.
        # ⛔ THE INDEX SPACES DIFFER AND BOTH ARE IN THIS EXPRESSION. The 2-D
        # join's `frame_idx` is POST-n_stack-trim (the episode index `f` above);
        # the 3-D join's key `frame` is the RAW v2ep index, which is
        # `f + n_stack - 1`. MEASURED on the eval join: line 1 is
        # `{"frame": 0, "frame_idx": -2}`, i.e. exactly that offset at n_stack 3.
        if self.join3d is not None:
            tids = r.lookup_track_ids(eid, int(f))
            cid = (self.map_clip_of_ep or {}).get(eid)
            if tids is None or cid is None or not has:
                _t3 = _box3d_head.zh_targets(t)          # all-False mask
            else:
                if n_raw > pad:
                    tids = _np.asarray(tids)[order]
                _cz, _h, _m = _agent_cuboid.zh_for_frame(
                    cid, int(f) + int(self.map_n_stack) - 1, list(tids),
                    join3d=self.join3d)
                _pad_f = pad - len(tids)
                if _pad_f > 0:                # pad to the slot block's width
                    _cz = _np.concatenate([_cz, _np.zeros(_pad_f)])
                    _h = _np.concatenate([_h, _np.zeros(_pad_f)])
                    _m = _np.concatenate([_m, _np.zeros(_pad_f, dtype=bool)])
                _t3 = _box3d_head.zh_targets(
                    t, torch.from_numpy(_cz).to(t["box"].dtype)[None],
                    torch.from_numpy(_h).to(t["box"].dtype)[None],
                    mask=torch.from_numpy(_m)[None])
            item["agent_cz"] = _t3["cz"][0]
            item["agent_h"] = _t3["h"][0]
            item["agent_zh_mask"] = _t3["zh_mask"][0]
        return item

    # ---- refcv6 §2: the SAM3 map GT ------------------------------------- #
    def enable_map_gt(self, store, clip_of_ep: dict, n_stack: int,
                      min_coverage: float | None = None) -> dict:
        """Attach the SAM3 BEV labels so ``__getitem__`` emits ``map_*``.

        ⛔ **COVERAGE IS A REFUSAL, NOT A WARNING.**
        :func:`perception_targets.require_map_coverage` is run over THIS
        dataset's own window index -- the windows that will actually be drawn,
        not the clips that happen to have files -- and a run below the floor
        raises :class:`MapCoverageTooLow` HERE, before the GPU. ⚠️ Its
        comparison uses ``frac_ok``, the LOWER bound, so an inconclusive window
        counts against the run: "we could not tell" is not coverage.
        """
        if int(n_stack) < 1:
            raise SystemExit(
                "[v3] ⛔ enable_map_gt needs the cache's own n_stack; it is "
                "what converts a stacked-row index into the RAW v2ep frame "
                "the labels are indexed by, and a literal here shifts every "
                "window's map by (wrong - right) frames, silently.")
        self.map_store = store
        self.map_clip_of_ep = dict(clip_of_ep)
        self.map_n_stack = int(n_stack)
        w = self.window
        miss = [int(self.episodes[e_i].episode_id) for e_i, _ in self.index
                if int(self.episodes[e_i].episode_id) not in self.map_clip_of_ep]
        if miss:
            raise SystemExit(
                f"[v3] ⛔ {len(set(miss))} episodes in this dataset have no "
                f"clip_id in the v2 manifest table, so their SAM3 labels "
                f"cannot be resolved at all. Refusing rather than counting "
                f"them as uncovered -- that would be a MEASUREMENT of the "
                f"manifest, reported as map coverage.")
        # ⛔ (clip_id, STACKED-ROW index of the window's NOW frame). The store
        # applies `+ (n_stack - 1)` itself; passing the raw frame here would
        # label every window `n_stack - 1` steps late instead.
        windows = [(self.map_clip_of_ep[int(self.episodes[e_i].episode_id)],
                    t + w - 1) for e_i, t in self.index]
        kw = {} if min_coverage is None else {"min_frac": float(min_coverage)}
        rep = _perception_targets.require_map_coverage(
            windows, store, n_stack=self.map_n_stack, **kw)
        # ⛔ THE PER-STATE COUNTS ARE `n_ok` / `n_no_file` / ... AT THE TOP
        # LEVEL, not a nested `states` dict. MEASURED here 2026-09-17: reading
        # a `states` key that does not exist printed "0/23,772 windows covered"
        # beside `frac_ok 0.9712` -- the total right and EVERY COLUMN ZERO,
        # which is the exact diagnostic shape `require_map_coverage`'s own
        # docstring warns about, reproduced one layer up in its caller.
        _states = {k[2:]: int(v) for k, v in rep.items()
                   if k.startswith("n_") and k not in
                   ("n_windows", "n_clips", "n_stack")}
        self.map_stats = {
            "root": str(getattr(store, "root", "")),
            "n_windows": int(rep.get("n_windows", 0)),
            "n_clips": int(rep.get("n_clips", 0)),
            "frac_ok": float(rep.get("frac_ok", float("nan"))),
            "frac_ok_upper": float(rep.get("frac_ok_upper", float("nan"))),
            "verdict": str(rep.get("verdict", "")),
            "min_coverage": (float(min_coverage) if min_coverage is not None
                             else float(_perception_targets.MIN_MAP_COVERAGE)),
            "states": _states,
            "reasons": dict(rep.get("reasons", {})),
            "n_stack": self.map_n_stack,
            "layouts": sorted(set((rep.get("layout_of") or {}).values())),
        }
        print("[v3] SAM3 map GT: %d/%d windows covered (%.4f, floor %.2f, "
              "verdict %s) over %d clips; states %s; layouts %s"
              % (_states.get("ok", 0), self.map_stats["n_windows"],
                 self.map_stats["frac_ok"], self.map_stats["min_coverage"],
                 self.map_stats["verdict"], self.map_stats["n_clips"],
                 _states, self.map_stats["layouts"]), flush=True)
        return self.map_stats

    def _map_item(self, ep, win_idx: int) -> dict:
        """One window's map target. NO_LABEL and LABELLED are different states.

        A clip with no GT file emits an all-zero ``map_frac`` with an ALL-FALSE
        ``map_seen`` and ``map_label`` False -- so ``map_soft_ce`` scores zero
        cells on it and the log's ``n_map_cells`` says so, rather than the loss
        reading 0.0 as "supervised, and perfect" (the ``tac_goal`` precedent).
        """
        cid = self.map_clip_of_ep[int(ep.episode_id)]
        try:
            mf = self.map_store.frames_for_windows(
                cid, [int(win_idx)], n_stack=self.map_n_stack)
        except (FileNotFoundError, IndexError, OSError):
            return {"map_frac": torch.zeros((_sem_map.N_CHANNELS,)
                                            + _sem_map.CART_SHAPE,
                                            dtype=torch.float32),
                    "map_seen": torch.zeros(_sem_map.CART_SHAPE,
                                            dtype=torch.bool),
                    "map_label": torch.tensor(False),
                    "map_raw_frame": torch.tensor(-1, dtype=torch.long)}
        return {"map_frac": torch.from_numpy(mf.cart[0]).to(torch.float32),
                "map_seen": torch.from_numpy(mf.seen[0]),
                "map_label": torch.tensor(True),
                "map_raw_frame": torch.tensor(int(mf.frame_idx[0]),
                                              dtype=torch.long)}

    def enable_join3d(self, join3d) -> dict:
        """Attach the 3-D cuboid join. A census, not a second label path."""
        self.join3d = join3d
        self.join3d_stats = {
            "path": str(getattr(join3d, "path", "")),
            "n_lines": int(getattr(join3d, "n_lines", 0)),
            "n_agents": int(getattr(join3d, "n_agents", 0)),
            "n_clips": int(getattr(join3d, "n_clips", 0))}
        return self.join3d_stats

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        T = ep.poses.shape[0]
        idx = torch.arange(t + w, t + w + MAX_H_EXT)
        item["future_poses_ext"] = ep.poses[idx.clamp(max=T - 1)]   # [60, 4]
        item["future_valid_ext"] = idx <= (T - 1)                   # [60] bool
        # ---- refcv6 §2b: the OBSERVED window's ego track -------------------
        # ⛔ EXACTLY `[t, t + w)`. The window's NOW is `t + w - 1`, the same
        # index `pose_last` reads (`_contract.py:137`), so this slice ENDS at
        # the present and contains no future sample. `future_poses_ext` above
        # starts at `t + w`; the two do not overlap, and that non-overlap is
        # the admissibility boundary the PI's 2026-09-02 ruling draws.
        # ⛔ Gated PER DATASET, never per row. Off, the item's KEY SET is
        # byte-identical to the base contract (`test_refc_v3_u8_batches.py`
        # pins exactly that); on, EVERY row carries it, so the trainer's
        # refusal fires at launch rather than at a random batch.
        if self.ego_history:
            item["pose_hist"] = ep.poses[t:t + w]                   # [W, 4]
        g, gv = refb_labels.goal_tac_targets(ep.poses, t + w - 1,
                                             v3.GOAL_TAU_STEPS)
        item["goal_tac"] = g                                        # [K, 4]
        item["goal_tac_valid"] = gv                                 # [K] bool
        # ---- v7.2 tactical labels (PI 2026-09-02: MANDATORY) --------------
        # ⭐ Joined on `stable_episode_id(clip_id)` — the v7.2 clip index names
        # it "the ONLY admissible join key", and `LazyV2Episode` carries the
        # stable id (not the clip_id string), so this is the join the artifacts
        # were built for rather than a string match we invented.
        # The window's NOW is the last OBSERVED frame: t + w - 1 at dt.
        if self.v7_by_sid is not None:
            lab = self.v7_by_sid.get(int(ep.episode_id))
            if lab is None:
                lat_v7 = lon_v7 = v7l.IGNORE_ID       # clip has no record
            else:
                lat_v7, lon_v7 = v7l.tactical_class_ids(
                    lab, (t + w - 1) * self.v7_dt)
            item["lat_v7"] = torch.tensor(lat_v7, dtype=torch.long)
            item["lon_v7"] = torch.tensor(lon_v7, dtype=torch.long)
            # ---- --w-tac-goal: the 22-token goal SET target ---------------
            # ⛔ ALWAYS EMITTED WHEN THE CHANNEL IS ON, including for a clip
            # with NO record -- as an explicitly all-ignored row, never a
            # missing key. That is the `nav_args` convention three blocks
            # down, and for the same measured reason: a batch that sometimes
            # carries the key would make the loss-time refusal fire at random
            # instead of at launch.
            # ⚠️ The window's NOW is `(t + w - 1) * v7_dt`, the SAME expression
            # `tactical_class_ids` is called with above. Passing the record's
            # own anchor instead would make `window_in_band` a tautology.
            if self.tac_goal_targets:
                if lab is None:
                    _n_tg = len(v7l.TAC_GOAL_TOKENS)
                    _tg_y = (0.0,) * _n_tg
                    _tg_w = (v7l.IGNORE_W,) * _n_tg
                else:
                    _tg_y, _tg_w = v7l.tactical_goal_targets(
                        lab, (t + w - 1) * self.v7_dt,
                        negatives=self.tac_goal_negatives,
                        sidecar=self.cot_negative_sidecar)
                item["tac_goal_y"] = torch.tensor(_tg_y, dtype=torch.float32)
                item["tac_goal_w"] = torch.tensor(_tg_w, dtype=torch.float32)
        # ---- --nav-from-v7: the nav INPUT from the clip's v7.2 token -------
        # ⛔ OVERRIDES the v1 nav_cmd/nav_valid the base chain assigned
        # (refb_train.py:220-222, refb_labels.nav_command). `route_target` /
        # `route_valid` are the aux TARGET and stay untouched — RouteV21Dataset
        # documents that separation. A clip with NO record feeds NAV_FOLLOW +
        # nav_valid=False — refav1_loader's convention (index 0 is the
        # codebase's unlabeled default, refb.py:60) — counted at init and
        # stamped into config.json.
        if self.nav_from_v7:
            nav_idx = self._nav_by_sid.get(int(ep.episode_id))
            if nav_idx is None:
                item["nav_cmd"] = torch.tensor(refb_labels.NAV_FOLLOW,
                                               dtype=torch.long)
                item["nav_valid"] = torch.tensor(False)
            else:
                item["nav_cmd"] = torch.tensor(nav_idx, dtype=torch.long)
                item["nav_valid"] = torch.tensor(True)
            # ---- --nav-args: (distance_norm, time_norm, args_valid) -------
            # ⛔ ALWAYS EMITTED WHEN THE CHANNEL IS ON, including for a clip
            # with no record — as an EXPLICITLY INVALID row, never a missing
            # key. A batch that sometimes carries `nav_args` and sometimes
            # does not would make the model's own "supplied but no seam /
            # seam but not supplied" refusals fire at random.
            if self.nav_args_enabled:
                st = self.nav_arg_stats
                raw = (self._nav_args_by_sid or {}).get(int(ep.episode_id))
                if raw is None or raw[2] <= 0.5:
                    item["nav_args"] = torch.zeros(3, dtype=torch.float32)
                else:
                    dn = (raw[0] - st.distance_mean) / st.distance_std
                    tn = (raw[1] - st.time_mean) / st.time_std
                    item["nav_args"] = torch.tensor([dn, tn, 1.0],
                                                    dtype=torch.float32)
        # ---- --max-speed-input: the map/nav posted-limit ceiling ----------
        # ⛔ ALWAYS EMITTED WHEN THE CHANNEL IS ON, including for a clip
        # with no block — as an EXPLICITLY INVALID row, never a missing key.
        # The `nav_args` rule verbatim: a batch that sometimes carries the
        # key and sometimes does not would make the model's own
        # "supplied but no seam / seam but not supplied" refusals fire at
        # random. A silent 0.0 in the VALUE slot reads as "the limit here is
        # 0 m/s — stop", which is a LIE; the validity slot is what says
        # "no limit known".
        if self.max_speed_enabled:
            raw = (self._max_speed_by_sid or {}).get(int(ep.episode_id))
            v_ms, ok = (0.0, 0.0) if raw is None else raw
            item["v_max_ms"] = torch.tensor(v_ms, dtype=torch.float32)
            item["v_max_valid"] = torch.tensor(ok, dtype=torch.float32)
        # ---- refcv5 WP-6: the obstacle.offline target block ---------------
        # The window's NOW is the last OBSERVED frame -- the same t + w - 1
        # the v7.2 tactical labels above are read at, so the detector and the
        # tactical heads are supervised at ONE instant, not two.
        if self.agent_join is not None:
            item.update(self._agent_item(ep, t + w - 1))
        # ---- refcv6 §2: the SAM3 map target at the SAME instant ------------
        # ⛔ `t + w - 1` -- the window's NOW as a STACKED-ROW index, which is
        # what `MapGTStore.raw_frames` converts. The detector, the tactical
        # heads and the map are then supervised at ONE instant, not three.
        # ⭐ `map_ep` travels with it for the same reason `agent_ep` does: the
        # BEV lift geometry is PER CLIP (mount height spans 1.2131-1.6672 m
        # over 554 distinct values in 2,400 clips) and the loss has no episode
        # ids of its own.
        if self.map_store is not None:
            item.update(self._map_item(ep, t + w - 1))
            item["map_ep"] = torch.tensor(int(ep.episode_id), dtype=torch.long)
        return item


def _synth_episodes(n: int, cfg: refc.RefCConfig, seed: int = 0,
                    min_frames: int = 40, clip_ids=None):
    """CI-only synthetic corpus (unicycle drives, tiny frames). NEVER a
    substitute for the parity cache — refused alongside --data-root.

    ⛔⛔ ``clip_ids`` EXISTS BECAUSE THE SYNTHETIC CORPUS COULD NOT JOIN A LABEL
    BLOB AT ALL, AND THE FAILURE WAS A BARE ``ValueError``, NOT A REFUSAL.
    MEASURED 2026-09-10: ``episode_id`` was the STRING ``f"synth-{e:03d}"``
    while the v7.2/v8 join is ``by_sid = {stable_episode_id(l.clip_id): l}``
    looked up as ``int(e.episode_id)`` (``train`` :4217). So
    ``--synth-episodes --v7-labels`` died with
    ``invalid literal for int() with base 10: 'synth-000'`` — i.e. **no CI or
    local smoke of ANY v7-label channel (nav, tac-goal, nav-args, max-speed)
    was possible**, and the crash looked like a broken flag rather than a
    corpus that structurally cannot carry a label.

    ⇒ When ``clip_ids`` is given, episode ``e`` is stamped with
    ``stable_episode_id(clip_ids[e])`` — a real integer id from a real label
    record — so the join HITS and the channel under test is exercised end to
    end. ⚠️ The FRAMES stay synthetic; this makes the label join real, not the
    perception. A run using it is identifiable from ``config.json``
    (``synth_clip_ids_from_labels``), because a synthetic corpus must never
    masquerade as a training cache.

    ⛔ Default ``None`` keeps the historical string id EXACTLY, so no existing
    CI path, preflight or test moves.

    ``min_frames`` exists because the DEFAULT corpus is too SHORT to carry a
    LAN route and that is not obvious from reading it: T=40 at 2-8 m/s is
    ~8-32 m of total path, while the shortest LAN anchor sits at 20 m ARC-LENGTH
    *beyond* a leak guard of ~2 s x v + 5 m. MEASURED (D-LAN-COV, this package):
    every anchor of every window is masked on the default corpus, so a LAN
    preflight built on it would compute ``goal_str`` over an all-invalid label
    and report a green 0.0 — a loss that exists and cannot learn. The LAN arm
    therefore asks for a corpus long enough for the question to be answerable.
    """
    import types
    from tanitad.data.v2_dataset import stable_episode_id
    if clip_ids is not None and len(clip_ids) < n:
        raise SystemExit(
            f"[v3] ⛔ --synth-episodes {n} but only {len(clip_ids)} label "
            f"clip_ids were supplied to stamp them with. Refusing rather than "
            f"reusing a clip_id twice — two episodes sharing one stable id "
            f"would silently collapse into one label and the join count would "
            f"still read 100 %.")
    g = torch.Generator().manual_seed(seed)
    h, wpx = cfg.encoder.image_hw()
    eps = []
    for e in range(n):
        T = max(min_frames, 40) + 8 * e
        v = 2.0 + 6.0 * torch.rand((), generator=g)
        yr = (torch.rand((), generator=g) - 0.5) * 0.4
        yaw = torch.cumsum(torch.full((T,), float(yr)) * 0.1, dim=0)
        xy = torch.cumsum(
            torch.stack([v * torch.cos(yaw), v * torch.sin(yaw)], -1) * 0.1, 0)
        poses = torch.cat([xy, yaw[:, None],
                           torch.full((T, 1), float(v))], dim=-1)
        eps.append(types.SimpleNamespace(
            frames=(torch.rand(T, cfg.encoder.in_channels, h, wpx,
                               generator=g) * 255).to(torch.uint8),
            actions=torch.zeros(T, 2),
            poses=poses,
            episode_id=(f"synth-{e:03d}" if clip_ids is None
                        else stable_episode_id(str(clip_ids[e])))))
    return eps


# ============================================================================
# Losses — masked 6 s + hierarchy terms
# ============================================================================

def frames_to_device(x: torch.Tensor, device) -> torch.Tensor:
    """THE ONE frame-ingest point of this trainer (training AND held-out eval).

    ``--u8-batches`` ships frames as uint8; here they become the float32 [0,1]
    the encoder expects (``refc.py:994`` — a bare Conv2d stem, no
    normalisation anywhere inside the model) by ``x.float().div_(<0-dim
    tensor 255.0 on x.device>)``: the SAME map as
    ``tanitad.data._contract.to_float_frames`` (``x.float().div(255.0)``,
    ``_contract.py:56``), the division IN PLACE on the fresh fp32 tensor (one
    full-size fp32 allocation fewer on the device — 2.36 GB at the live
    future-frames shape) and by a TENSOR divisor so that CUDA performs a true
    division — bit-identical to the CPU contract over all 256 uint8 values on
    BOTH devices (pinned, with the scalar form's 1-ulp drift as the control,
    in ``tests/test_refc_v3_u8_batches.py``). A float input passes through
    untouched, exactly as ``to_float_frames`` passes it: with the flag OFF
    (the live run, the preflight, every existing dump) this is
    ``x.to(device)`` and nothing else.
    """
    x = x.to(device)
    if x.dtype == torch.uint8:
        # ⛔ A 0-dim DEVICE-TENSOR divisor, NOT the Python scalar 255.0.
        # MEASURED 2026-09-02 (torch 2.11.0+cu128, RTX 4060, exhaustive over
        # the 256 uint8 values): with a Python scalar, CUDA's div kernel takes
        # the multiply-by-reciprocal fast path and 126/256 values land 1 ulp
        # (5.96e-8) off the CPU contract — `to_float_frames` itself, run on
        # CUDA, would silently drift from HEAD's inputs. With a tensor divisor
        # the kernel is a true IEEE division: 0/256 off, on CPU and on CUDA.
        return x.float().div_(torch.full((), 255.0, device=x.device,
                                         dtype=torch.float32))
    return x


def _resolve_rig_cameras(model, ep_ids, n_rows: int):
    """``model._rig_camera`` -> what ``refc_agents.agent_losses`` takes.

    ``None`` / a single ``RigCamera`` pass straight through (every banked arm
    is untouched). A :class:`refc_agents.RigCameraBank` is resolved to a
    per-ROW list using the batch's own ``agent_ep``.

    ⛔ A bank with no ``agent_ep`` in the batch REFUSES. Falling back to the
    bank's default would train the monocular terms against ONE mount pose
    while ``config.json`` says ``mount_pose_scope: PER-CLIP`` -- the M18
    dead-flag defect with the object swapped from a weight to a camera.
    """
    cam = getattr(model, "_rig_camera", None)
    if not isinstance(cam, _refc_agents.RigCameraBank):
        return cam
    if ep_ids is None:
        raise SystemExit(
            "[v3] \u26d4 REFUSING: a PER-CLIP rig camera bank was built "
            "(--agent-rig-extrinsics carries a clip table) but the batch "
            "carries no `agent_ep`. The dataset emits it only alongside the "
            "agent join, so this arm would train the monocular terms against "
            "no camera at all while config.json records mount_pose_scope "
            "PER-CLIP. Pass --agent-join, or use a single-camera "
            "extrinsics file.")
    cams = cam.for_episodes(ep_ids)
    if len(cams) != int(n_rows):
        raise SystemExit(
            "[v3] \u26d4 REFUSING: %d per-clip cameras for %d supervised rows "
            "-- the episode ids were not selected with the same mask as the "
            "targets, so a camera would attach to the wrong clip."
            % (len(cams), int(n_rows)))
    return cams


def compute_losses_v3(model: v3.RefCV3Model, batch: dict, device: str,
                      mode: str = "diffusion",
                      ablate_frames: bool = False) -> dict:
    cfg = model.cfg
    core = cfg.core
    # --u8-batches: uint8 in flight -> float32 [0,1] HERE, on the device, by
    # the contract's own /255 (frames_to_device). Float batches pass through.
    frames = frames_to_device(batch["frames"], device)
    fut_frames = frames_to_device(batch["future_frames"], device)
    # ⛔⛔ THE DELIBERATE-REGRESSION LEVER (rig only, `--ablate-frames`).
    # Replace the OBSERVED window with a scalar constant, so every sample sees
    # the SAME information-free image. The arm keeps every ego channel and
    # therefore can ONLY solve the task by echoing its own dynamics — it is an
    # echo BY CONSTRUCTION, and `TanitAD_ValidateAIDesign` §2 requires it: if
    # the anti-echo gate does not FAIL this arm, a PASS on the real arm means
    # nothing.
    # ⚠️ CONSTANT, NOT ZERO, and not noise. Zeros push the trunk's activation
    # statistics off-distribution, so a failure could be read as "the encoder
    # broke" rather than "the scene carries nothing"; fresh noise per step
    # would make the input a random variable the model can average out, which
    # is a DIFFERENT ablation. A constant removes exactly the information and
    # nothing else. ⛔ `fut_frames` is deliberately NOT ablated: it is the
    # LAW auxiliary's TARGET, and ablating a target changes the objective
    # rather than the input — a second lever hidden inside the first.
    if ablate_frames:
        frames = torch.full_like(frames, float(frames.mean()))
    fut_ext = batch["future_poses_ext"].to(device)          # [B, 60, 4]
    fut_valid = batch["future_valid_ext"].to(device)        # [B, 60] bool
    pose_last = batch["pose_last"].to(device)
    nav_cmd = batch["nav_cmd"].to(device)
    nav_valid = batch["nav_valid"].to(device)
    # ⭐ D-GSTR-1 P3 (E13b): the nav command's range and time, if the loader
    # emitted them. Absent from the batch = the channel is off, and the model
    # REFUSES the mismatch in either direction rather than dropping it.
    nav_args = (batch["nav_args"].to(device) if "nav_args" in batch else None)
    # ⭐⭐ E16 — the map/nav posted-limit ceiling, if the loader emitted it.
    # ⛔ THE REVERSE REFUSAL LIVES HERE, where the BATCH is the fact. A build
    # that asked for the seam and is handed a batch without the key would
    # train the conditioner on nothing while `config.json` stamps the edge —
    # the false-provenance class `assert_seams_are_built` exists to close.
    # ⛔ refcv6 §5 SHARES THE `v_max_ms` BATCH KEY, and that is safe ONLY
    # because the two channels are mutually exclusive: `_pin_refcv6_tactical`
    # and `RefCV3Model.__init__` both refuse `--max-speed-input` together with
    # `--max-speed-input-v6`. What travels in the key is the RAW m/s value in
    # both cases; the LADDER that turns it into a condition differs, lives on
    # the model, and is applied exactly once there.
    v_max_ms = v_max_valid = None
    _vmax_on = (bool(getattr(cfg, "max_speed_input", False))
                or bool(getattr(cfg, "max_speed_onehot_v6", False)))
    if _vmax_on:
        if "v_max_ms" not in batch:
            raise SystemExit(
                "[v3] ⛔ this build is --max-speed-input/--max-speed-input-v6 "
                "but the batch carries no `v_max_ms`: the loader's "
                "`enable_max_speed`/`enable_max_speed_v6` was never called on "
                "this dataset. The seam would be stamped and fed nothing.")
        v_max_ms = batch["v_max_ms"].to(device)
        v_max_valid = batch["v_max_valid"].to(device)
    route_tgt = batch["route_target"].to(device)
    goal_tac = batch["goal_tac"].to(device)                 # [B, K, 4]
    goal_valid = batch["goal_tac_valid"].to(device)         # [B, K] bool
    lan = batch["lan"].to(device) if "lan" in batch else None
    v0 = pose_last[:, 3]
    b = frames.shape[0]
    steps = core.decoder.diffusion_steps if mode == "diffusion" else 0

    # ⭐⭐ REF-C v4 (E11'): the MEASURED ego state at t0. Derived from
    # `pose_last` + `actions[:, -1]`, both of which sit at the LAST OBSERVED
    # frame and both of which the window contract ALREADY returns - no new
    # dataset field and no cache rebuild. `future_poses_ext` is NOT read here
    # and must not be: that is the edge E11' pins interventionally.
    ego_state = None
    if getattr(cfg, "ego_state_inject", False):
        ego_state = v3.ego_state_from_batch(
            {"pose_last": pose_last, "actions": batch["actions"]},
            device=device)

    # ⛔⛔ refcv5 WP-6 ORACLE: THE FORWARD-SIDE AGENT PLUMBING, WHICH WAS ABSENT.
    # MEASURED 2026-09-09 on `tanitad-refcv3` by the WP-C oracle gate's 20-step
    # smoke: `--agents oracle` reached `RefCV3Model.forward` with `agent_gt=None`
    # and the model's own guard REFUSED the run (`refc_v3.py:1404`,
    # `refc.py:3355`). The token `agent_gt` did not occur ANYWHERE in this file.
    #
    # The join was never the problem and the census proves it: 4427/4572 train
    # episodes, 729,243/781,635 windows labelled (93.3 %), 24,166,608 prefilter
    # target boxes. `refc.py:3346-3347` states the design exactly — *"The
    # detection LABELS never enter here at all; they enter the LOSS, in the
    # trainer, from `batch["agent_box"]`"* — which is true of the `head` path and
    # leaves the ORACLE path with no supplier, because on that path the same
    # tensors are the TOKENS rather than the targets.
    #
    # ⭐ Why this is a wiring gap and not a design choice: `--agents head` needs
    # these tensors only in the loss (its tokens come from the image), so the
    # loss-side block at `tgt_ag` below was sufficient for every arm that has
    # ever run. The oracle rung is the first consumer of the forward side.
    #
    # ⛔ STRICTLY GATED ON THE ORACLE SEAM, in BOTH directions, because the model
    # refuses both: supplying `agent_gt` to a build with no seam raises
    # (`refc_v3.py:1398`), and withholding it from an oracle build raises
    # (`:1404`). So `--agents off` and `--agents head` are bit-unchanged: the
    # branch below cannot fire for them.
    _ag_cfg = getattr(core, "agents", None)
    agent_gt = None
    if (_ag_cfg is not None and getattr(_ag_cfg, "enable", False)
            and getattr(_ag_cfg, "oracle", False)):
        if "agent_box" not in batch:
            # A refusal, never a silent None: an oracle arm whose batch carries
            # no boxes would read as "agent tokens do not help" while never
            # having had any — the manufactured negative the model's own guard
            # exists to prevent, one layer up where the cause is legible.
            raise SystemExit(
                "[v3] ⛔ --agents oracle but the batch carries no `agent_box`. "
                "The oracle's tokens ARE the ground-truth boxes, so without "
                "them the seam emits nothing and the arm would report as "
                "+agents while running without them. Pass --agent-join <file>.")
        # Mirrors the loss-side `tgt_ag` construction below key-for-key; the
        # forward reads a strict SUBSET (`refc.py:3364-3365` takes box, yaw,
        # cls, valid and an optional rates) and no `occ`/`rates_mask`.
        agent_gt = {"box": batch["agent_box"].to(device),
                    "yaw": batch["agent_yaw"].to(device),
                    "cls": batch["agent_cls"].to(device),
                    "valid": batch["agent_valid"].to(device),
                    "rates": batch.get(
                        "agent_rates",
                        torch.zeros(*batch["agent_yaw"].shape, 3)).to(device)}

    # ---- refcv6 §2b: hand the core THIS batch's OBSERVED ego window ------- #
    # ⛔ `pose_hist` is `ep.poses[t : t + w]` — the observed window, ending at
    # the same index `pose_last` reads. `n_past` is its FULL length because
    # every step of it is past; the encoder slices again, so a future read
    # would have to defeat two independent cuts. ⚠️ `set_ego_window` is a ONE-
    # SHOT channel (`RefCModel.set_ego_window`): it exists only because
    # `refc_v3.RefCV3Model.forward` does not forward unknown kwargs to the
    # core, and that file is another agent's.
    if getattr(model.core, "ego_hist", None) is not None:
        if "pose_hist" not in batch:
            raise SystemExit(
                "[v3] ⛔ --ego-history but the batch carries no `pose_hist`: "
                "the encoder would be stamped and fed nothing. Rebuild the "
                "dataset through V3Dataset, which emits it unconditionally.")
        ph = batch["pose_hist"].to(device)
        model.core.set_ego_window(ph, int(ph.shape[1]))
    # ---- refcv6 §2/§6 + PI RULING 2026-09-17 R2: THIS BATCH'S LIFT GEOMETRY --
    # ⛔ RESOLVED BEFORE THE FORWARD, because the BEV encoder now runs INSIDE
    # it. Until tonight the branch ran after `model(...)` returned and could
    # look the geometry up itself; that ordering is precisely what kept the
    # behaviour decoder from ever seeing a map. ⚠️ An EXPLICIT argument, not a
    # one-shot setter: `set_ego_window`'s POP idiom exists only because
    # `RefCV3Model.forward` could not be edited, and it can be here.
    _pgrid = _pvalid = None
    _pbr = getattr(model, "_perception", None)
    if _pbr is not None and _pbr.lift is not None:
        _bank = getattr(model, "_lift_bank", None)
        if _bank is None:
            raise SystemExit(
                "[v3] ⛔ the BEV lift is built but no per-clip geometry bank "
                "is attached. One mount pose for a corpus whose MEASURED "
                "height spans 1.2131-1.6672 m (554 distinct values in 2,400 "
                "clips) biases every cell the lift fills.")
        if "map_ep" not in batch:
            raise SystemExit(
                "[v3] ⛔ the BEV lift needs `map_ep` and the batch carries "
                "none — a camera would attach to the wrong clip, silently, "
                "with every count still looking healthy.")
        _pgrid, _pvalid = _bank.for_episodes(batch["map_ep"], device=device)
    out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan,
                ego_state=ego_state, nav_args=nav_args,
                v_max_ms=v_max_ms, v_max_valid=v_max_valid,
                agent_gt=agent_gt,
                perception_grid=_pgrid, perception_valid=_pvalid)

    # ---- trajectory target over the 8-slot 6 s horizon, masked -------------
    traj_tgt = refb_labels.waypoint_targets(pose_last, fut_ext,
                                            core.trajectory.horizons)
    slot_valid = torch.stack([fut_valid[:, h - 1]
                              for h in core.trajectory.horizons], dim=1)
    sv = slot_valid.to(traj_tgt.dtype)                      # [B, S]
    # ⛔ THE TARGET MUST BE MEASURED AGAINST THE BANK THAT WAS ACTUALLY
    # DECODED. With a v0-conditioned vocabulary `decoder.anchors` is the family
    # rolled at the REFERENCE speed, not this window's fan, so scoring `a_star`
    # against it would supervise the anchor classifier on a geometry the model
    # never emitted — silently, and with `anchor_acc` still reading plausibly.
    # `out["anchor_bank"]` is [B, N, S, 2] and is EXACTLY `x0`; for a fixed
    # vocabulary it is `anchors[None].expand(...)`, so this line is unchanged
    # arithmetic there (verified bit-identical, 2026-09-04).
    anchors = out["anchor_bank"].to(traj_tgt.dtype)          # [B, N, S, 2]
    dist = (((traj_tgt[:, None] - anchors) ** 2).sum(-1)
            * sv[:, None]).sum(-1)                          # [B, N] valid-only
    a_star = dist.argmin(dim=1)
    ar = torch.arange(b, device=device)
    # ---- refcv6 F5: DD classifies with a SIGMOID FOCAL loss, not a softmax -- #
    # `multimodal_loss.py:146-157`, gamma 2.0 / alpha 0.25, reduction 'mean'
    # over B x N. ⛔ With `f5_focal` off this is the same `cross_entropy` call
    # it has always been -- same tensor, same reduction, same scale.
    _rv6f = _refcv6_flags_of(model)
    if _rv6f.f5_focal:
        loss_cls = _rv6.focal_cls_loss(out["anchor_logits"], a_star,
                                       gamma=float(_rv6f.f5_focal_gamma),
                                       alpha=float(_rv6f.f5_focal_alpha))
    else:
        loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
    recon = out["anchor_traj"][ar, a_star]                  # [B, S, 2]
    denom = (sv.sum() * 2).clamp_min(1.0)
    loss_traj = (((recon - traj_tgt).abs().sum(-1)) * sv).sum() / denom

    # ---- factored tactical CE (2 s labels; the shared aux surface) ---------
    # ⭐ TWO SUPERVISION SOURCES, ONE REFUSAL EACH (PI 2026-09-02 made v7.2
    # MANDATORY for this launch). `lat_v7` present => the released v7.2
    # tactical vocabulary (8x8), joined per window by stable episode id;
    # absent => the runtime kin3 derivation (3x3) from poses. The head width
    # must match ITS OWN source, and a mismatch is REFUSED, never truncated --
    # a wider head fed narrower labels trains silently wrong classes, which is
    # exactly the defect found in this trainer on 2026-09-01.
    # ⛔⛔ TWO TACTICAL SURFACES, TWO VOCABULARIES — AND THE CORE'S IS FIXED.
    # MEASURED 2026-09-02: the core's heads are `nn.Linear(aux_hidden,
    # N_LAT_MAN)` with `N_LAT_MAN = tac.N_LAT` a MODULE-LEVEL CONSTANT
    # (refc.py:146,1742) — the REF-C core is STRUCTURALLY kin3, and
    # `tac_vocab_version` only sizes v3's OWN z_tac heads. Widening the core
    # would also have to redefine `derive_man5_logprobs` (a push-forward
    # DEFINED on the 3x3 kinematic vocabulary) and the H19 `lat_to_anchor`
    # prior — a change to the shared core, not a wiring job.
    # ⇒ v7.2 supervises the surface it can actually reach: the HIERARCHY's
    # tactical decision heads (z_tac). The core's legacy aux head keeps its
    # kinematic derivation. Both are trained; they are NOT the same label set,
    # and the launch record says so rather than implying one vocabulary.
    lat_k, lon_k = tac.window_factored_labels(pose_last, fut_ext[:, :20])
    if (out["lat_logits"].shape[-1] != tac.N_LAT
            or out["lon_logits"].shape[-1] != tac.N_LON):
        raise RuntimeError(
            f"CORE head/label mismatch: heads are "
            f"{out['lat_logits'].shape[-1]}x{out['lon_logits'].shape[-1]} vs "
            f"kinematic {tac.N_LAT}x{tac.N_LON}. The core surface is kin3 by "
            f"construction (refc.py N_LAT_MAN); only the v3 z_tac heads follow "
            f"tac_vocab_version.")
    loss_lat = F.cross_entropy(out["lat_logits"], lat_k)
    loss_lon = F.cross_entropy(out["lon_logits"], lon_k)
    # C-REFCV3-EVAL-PRIOR-LEAK (2026-09-02): the in-training eval reuses this function, and an
    # unconditional update EMA'd the HELD-OUT split's label marginals into core.lat/lon_log_prior
    # (buffers that shape decodes via logit_adjust). The v1 trainer gates the same call on
    # model.training (refc_train.py:521); so does this one now.
    if model.training:
        model.core.update_tactical_prior(lat_k, lon_k)
    # the z_tac surface's labels: v7.2 when supplied, else the same kin3
    use_v7 = "lat_v7" in batch
    if use_v7:
        lat_t, lon_t = batch["lat_v7"].to(device), batch["lon_v7"].to(device)
        n_lat_expect, n_lon_expect = (len(v7l.HEADS["tac_lat"]),
                                      len(v7l.HEADS["tac_lon"]))
        src = f"v7.2 ({n_lat_expect}x{n_lon_expect})"
    else:
        lat_t, lon_t = lat_k, lon_k
        n_lat_expect = n_lon_expect = tac.N_LAT
        src = f"kin3 ({tac.N_LAT}x{tac.N_LON})"
    # ⛔ -100 marks a window this clip's single record does not describe. CE
    # ignores those rows; an ALL-ignored batch is NaN, not zero, so it is
    # SKIPPED (the same guard refav1 needed -- measured, not assumed).
    lat_ok = int((lat_t != v7l.IGNORE_ID).sum())
    lon_ok = int((lon_t != v7l.IGNORE_ID).sum())
    loss_lat_tac = torch.zeros((), device=device)
    loss_lon_tac = torch.zeros((), device=device)
    if cfg.hier:                       # the H arm's z_tac decision surface
        # the SAME refusal for the v3 heads — the core-only check above let an
        # 8-wide v7.0 build train silently against kin3 labels (2026-09-01).
        if (out["lat_logits_tac"].shape[-1] != n_lat_expect
                or out["lon_logits_tac"].shape[-1] != n_lon_expect):
            raise RuntimeError(
                f"z_tac head/label vocabulary mismatch: lat "
                f"{out['lat_logits_tac'].shape[-1]} / lon "
                f"{out['lon_logits_tac'].shape[-1]} vs {src} -- the model was "
                f"built with tac_vocab_version={model.tac_vocab_version!r}. "
                f"The z_tac heads and the core heads MUST share one source; "
                f"this check mirrors the core one because a v3 build once "
                f"passed the core check and trained 8-wide z_tac heads "
                f"against 3-class labels (2026-09-01).")
        loss_lat_tac = (F.cross_entropy(out["lat_logits_tac"], lat_t) if lat_ok
                        else torch.zeros((), device=device))
        loss_lon_tac = (F.cross_entropy(out["lon_logits_tac"], lon_t) if lon_ok
                        else torch.zeros((), device=device))

    # ---- route aux (v2.1 masked CE — refc_train's convention) --------------
    # ⛔⛔ FIXED 2026-09-02, MEASURED ON THE A40: this masked by `nav_valid`,
    # which is a DIFFERENT VALIDITY FIELD than the target it guards. v2.1's
    # `route_target` is ROUTE_UNKNOWN (= 3, deliberately OUTSIDE the 3-class CE
    # range) exactly where `route_valid` is False — and a window can carry a
    # VALID nav command with an UNKNOWN route. Those rows survived the
    # nav_valid mask and hit the CE, producing
    #   `nll_loss_forward_reduce_cuda_kernel_2d: t >= 0 && t < n_classes`
    # — an async device-side assert that surfaces at the NEXT CUDA call
    # (conv2d), so the traceback blames the wrong line entirely.
    # ⚠️ WHY IT SURVIVED EVERY SMOKE: it is window-frequency dependent. The
    # 2-step batch-2 smoke and the batch-4 run never drew an offending window;
    # batch 20 hit it inside 14 steps. A 30 k launch would have died hours in.
    # ⇒ mask on `route_valid` when the v2.1 labeler supplies it (mirroring
    # `refc_train.py`'s shared path verbatim), and keep the FAIL-LOUD check —
    # an UNKNOWN surviving the mask is a labeler contract violation, never
    # something to clamp to `straight`.
    if "route_valid" in batch:
        mask = batch["route_valid"].to(device)
        if bool(mask.any()):
            tgt_v = route_tgt[mask]
            n_route = out["route_logits"].shape[-1]
            if int(tgt_v.max()) >= n_route:
                raise ValueError(
                    f"ROUTE_UNKNOWN survived the valid mask (max target "
                    f"{int(tgt_v.max())} >= n_route {n_route}) — the v2.1 "
                    f"contract is route<3 <=> valid=True")
            loss_route = F.cross_entropy(out["route_logits"][mask], tgt_v)
        else:                        # no judgeable window in this batch
            loss_route = torch.zeros((), device=device)
    else:
        mask = nav_valid
        loss_route = (F.cross_entropy(out["route_logits"][mask],
                                      route_tgt[mask])
                      if bool(mask.any())
                      else torch.zeros((), device=device))

    # ---- LAW aux (0.5 s pooled-latent target, no_grad encode) --------------
    with torch.no_grad():
        law_tgt = model.core.encode_pooled(fut_frames[:, LAW_AHEAD - 1]
                                           .to(device))
    loss_law = F.mse_loss(out["law_pred"], law_tgt)

    loss = (TRAJ_WEIGHT * loss_traj + ANCHOR_CLS_WEIGHT * loss_cls
            + LAW_WEIGHT * loss_law + ROUTE_WEIGHT * loss_route
            + (LAT_WEIGHT / 2.0) * (loss_lat + loss_lat_tac)
            + (LON_WEIGHT / 2.0) * (loss_lon + loss_lon_tac))
    # ⛔ THE /2 IS THE FIX, NOT A TYPO. `refc_train.py:83-91` states in writing
    # that the TOTAL tactical aux pressure is held at EXACTLY MANEUVER_WEIGHT
    # (0.10) so an arm differs from the base run in STRUCTURE, not in loss
    # budget. This trainer supervises TWO tactical surfaces -- the core's pooled
    # kin3 heads (`loss_lat`/`loss_lon`) and the v7.2 z_tac heads
    # (`loss_lat_tac`/`loss_lon_tac`) -- so the unhalved form spends
    # 0.05*2 + 0.05*2 = 0.20, i.e. DOUBLE the documented budget, which refcv3
    # shipped with. Halving restores 0.025*4 = 0.10 = MANEUVER_WEIGHT.

    extra: dict = {}
    # ⭐ WHICH label set trained the tactical decision surface, and how many
    # rows survived the in-band mask, IN EVERY LOG ROW. A run that silently
    # fell back to kin3 (missing --v7-labels, or a corpus whose clips have no
    # record) would otherwise look identical to a v7.2 run in the log.
    extra["tac_label_v7"] = 1.0 if use_v7 else 0.0
    extra["tac_label_rows"] = float(lat_ok)
    if cfg.hier:
        # E8 — masked goal regression (each level trains by its OWN label).
        loss_goal = v3.masked_goal_loss(out["g_tac"], goal_tac, goal_valid)
        loss = loss + GOAL_TAC_WEIGHT * loss_goal
        extra["goal_tac"] = loss_goal
        with torch.no_grad():
            e2 = (out["g_tac"][:, model._tau_slot_2s(), :2]
                  - goal_tac[:, model._tau_slot_2s(), :2]).norm(dim=-1)
            m2 = goal_valid[:, model._tau_slot_2s()]
            extra["goal2s_err_m"] = (e2[m2].mean() if bool(m2.any())
                                     else torch.zeros((), device=device))
        # E3 — strategic goal off the leak-guarded LAN label (train-only, E12).
        # ⭐⭐ STRATEGIC BYPASS: with `--no-strategic` the E4 FiLM is skipped,
        # so `g_str` has NO in-graph consumer at all. Supervising it anyway
        # would push gradient through `str_goal_head` -> `StrategicCtx` -> the
        # SHARED ENCODER, i.e. the strategic layer would still shape the trunk
        # that produces the plan. That is a SECOND variable inside a
        # one-variable arm, so the term is dropped and `config.json` says so
        # (`goal_str_loss_applied`). `--goal-str` may still be passed -- it
        # keeps the LAN LABEL pathway, and with it the dataloader, byte-for-byte
        # identical to refcv4b's, which is what makes the comparison matched.
        if lan is not None and not bool(getattr(core, "no_strategic", False)):
            bearing_t, dist_t, valid_t = refc.RefCModel.goal_targets(
                lan, core.lan.k)
            loss_gstr = v3.strategic_goal_loss(out["g_str"], bearing_t,
                                               dist_t, valid_t)
            loss = loss + GOAL_STR_WEIGHT * loss_gstr
            extra["goal_str"] = loss_gstr
        # E9 — survivor-set CE on the blended score (the gates' ONLY gradient:
        # argmax has none; the goal head is firewalled by detach).
        fan_err = (((out["anchor_traj"] - traj_tgt[:, None]).norm(dim=-1)
                    * sv[:, None]).sum(-1)
                   / sv.sum(-1, keepdim=True).clamp_min(1.0))     # [B, N]
        loss_sel = v3.selection_ce(out["sel_score_v3"], fan_err.detach(),
                                   out.get("reach_keep"))
        loss = loss + SEL_V3_WEIGHT * loss_sel
        extra["sel_v3"] = loss_sel
        extra["goal_gate"] = model.goal_gate.detach()
        # ⭐ CAVEAT-B (PI 2026-09-02): the gate value alone cannot distinguish
        # "has not opened YET" from "will never open". Emit the score scale it
        # multiplies and its own gradient, so the 30 k read answers the
        # question with data instead of a 14-step glance.
        if "goal_score_absmean" in out:
            extra["goal_score_absmean"] = out["goal_score_absmean"]
        if model.goal_gate.grad is not None:
            extra["goal_gate_grad"] = model.goal_gate.grad.detach().abs()
        # ⭐⭐ E15 (GP-2) — THE METRIC GOAL POINT'S SUPERVISION AND ITS
        # PRE-REGISTERED HEAD READOUTS.
        #
        # ⛔ THE LABEL IS `traj_tgt` AT THE GOAL SLOT — NOT A NEW FIELD, AND
        # NOT A RE-DERIVATION. `traj_tgt = refb_labels.waypoint_targets(
        # pose_last, fut_ext, horizons)` is already the ego-frame future path
        # resampled onto exactly these horizons, and `slot_valid[:, s]` is
        # `fut_valid[:, h - 1]`. `goal_point.goal_point_label_at_time` takes
        # `k = round(t/dt) - 1`, i.e. THE SAME INDEX (t=4.0 s -> tick 40 ->
        # `fut_valid[:, 39]`). Reusing the trainer's own target is what stops
        # a guard/label re-implemented beside its consumer from drifting —
        # the defect the navpred RESULT retracted two published numbers for.
        #
        # ⛔ ADMISSIBILITY, RE-STATED WHERE THE CODE IS: this is a TRAIN-ONLY
        # LABEL built from the ego's own future path, which is sanctioned
        # ("labels may use ego; inference is vision-only"). `out["goal_point"]`
        # is predicted from the STRATEGIC CONTEXT TOKEN alone; `fut_ext` never
        # enters the forward. And the goal sits at t_goal_s = 4.0 s, STRICTLY
        # BEYOND the 2 s scored horizon — enforced by GoalPointConfig, which
        # RAISES rather than warns.
        w_gp = float(getattr(model, "_w_goal_point",
                             GOAL_POINT_WEIGHT_DEFAULT))
        if cfg.goal_point_inject and out.get("goal_point") is not None:
            gcfg = cfg.goal_point_cfg
            gslot = int(model.gp_slot)
            gscale = float(gcfg.range_norm_m)
            gp_tgt = torch.stack([
                (traj_tgt[:, gslot, 0] / gscale).clamp(-gcfg.xy_clip,
                                                       gcfg.xy_clip),
                (traj_tgt[:, gslot, 1] / gscale).clamp(-gcfg.lat_clip,
                                                       gcfg.lat_clip),
            ], dim=-1)
            gp_v = slot_valid[:, gslot]
            loss_gp = gpm.goal_point_loss(out["goal_point"],
                                          gp_tgt.to(out["goal_point"].dtype),
                                          gp_v)
            loss = loss + w_gp * loss_gp
            extra["goal_point"] = loss_gp
            # ⭐ THE HEAD GATE, IN METRES, EVERY EVAL — PREREG §5. Registered
            # bars: lateral <= 1.0 m, range <= 2.0 m. MEASURED basis: at 1.0 m
            # injected lateral error the oracle goal point's recovery of the
            # lateral selection ceiling flips +78.4 % -> -80.7 %; range
            # recovery goes 57.1 % -> 17.2 % at 2 m -> -57.9 % at 4 m
            # (separated WORSE). ⛔ If these fail, the planner comparison is
            # NOT evidence about the goal FORM — stated in advance so it
            # cannot be decided after the data.
            with torch.no_grad():
                extra["gp_lat_rmse_m"] = gpm.lateral_rmse_m(
                    out["goal_point"], gp_tgt, gp_v, gscale)
                extra["gp_range_rmse_m"] = gpm.range_rmse_m(
                    out["goal_point"], gp_tgt, gp_v, gscale)
                # the DENOMINATOR, always: a window whose future ends before
                # t_goal_s has NO label, and an RMSE over a handful of rows is
                # not the same statistic as one over the batch.
                extra["gp_label_rows"] = gp_v.sum().to(loss_gp.dtype)
            # ⭐ CAVEAT-B for the S7 gate, same discipline as `goal_gate`
            # above: the gate value alone cannot tell "not opened YET" from
            # "never will", so the score scale it multiplies rides with it.
            if "gp_point_gate_value" in out:
                extra["gp_point_gate"] = out["gp_point_gate_value"]
            if "gp_point_score_absmean" in out:
                extra["gp_point_score_absmean"] = out["gp_point_score_absmean"]
        # E13 telemetry: nav actually reached the tactical/strategic states on
        # this batch. A conditioning edge that silently no-ops (nav_cmd=None
        # everywhere) is the advertised-but-inert defect; this makes it visible.
        if "nav_injected" in out:
            extra["nav_injected"] = float(bool(out["nav_injected"]))
        # ⭐⭐ THE ANTI-ECHO READOUT (E11'/E14). The model COMPUTES these
        # every forward and the trainer was DROPPING them, so the design's own
        # promise — *"is it echoing? is read off the LOG rather than inferred
        # from an eval three days later"* — was not kept by the code that
        # writes the log. Same class as the `tac_label_v7` drop noted below:
        # a diagnostic that does not survive to the log is not a diagnostic.
        #   echo_ratio = |g_tac_delta| / |echo_base| = how much goal VISION
        #   contributes beyond the ego self-extrapolation. A run that sits at
        #   ~0 has learned nothing the ego state did not already say.
        # ⚠️ It is a RATIO OF MAGNITUDES, not a verdict: a large delta can
        # still be wrong, and a small one can still be right if `ha0_ext` is
        # already near-optimal. The verdict is `tanitad.eval.echo_gate`; this
        # is the live tell that says whether to keep paying for the run.
        for _k in ("echo_ratio", "echo_base_absmean", "g_tac_delta_absmean"):
            if _k in out:
                extra[_k] = out[_k]
        # E11' telemetry, the mirror of `nav_injected`: did the ego block
        # actually reach the goal path on this batch? An edge that silently
        # no-ops reads as "the hypothesis failed".
        if "ego_injected" in out:
            extra["ego_injected"] = float(bool(out["ego_injected"]))
        if "ego_keep_frac" in out:
            extra["ego_keep_frac"] = out["ego_keep_frac"]
    # ⭐ H-EGO-LIT-4 diagnostics: the model's OWN 2 s speed vs the GT 2 s
    # speed, split by the withholding draw. `withheld_speed_mae` is the A0
    # log the pre-registration reads the `pred` warm-up step N from ("the
    # step at which the withheld-row speed MAE first drops below 2.5 m/s").
    # A key is absent (not NaN) on a batch with no rows in that regime.
    if "bank_speed_pred" in out and "ego_keep" in out:
        _slot = model._tau_slot_2s()
        _gv = goal_valid[:, _slot]
        _err = (out["bank_speed_pred"].float()
                - goal_tac[:, _slot, 3].float()).abs()
        _keep = out["ego_keep"].float() > 0.5
        for _tag, _m in (("withheld", (~_keep) & _gv), ("kept", _keep & _gv)):
            _n = int(_m.sum())
            extra[f"{_tag}_rows"] = float(_n)
            if _n:
                extra[f"{_tag}_speed_mae"] = _err[_m].mean().detach()
        extra["bank_speed_pred_mean"] = out["bank_speed_pred"].float().mean()

    # ---- refcv5 WP-4: the x0 loss, in CONTROL space ----------------------
    # ⭐ This is the ONE term that is not expressible in metres, which is why
    # `u0_hat` has to leave the decoder at all. The target is the GT path's own
    # control sequence through the programme's inverse map -- so the sampler is
    # supervised in the space it samples in, not in the space it is read out in.
    w_u0 = float(getattr(model, "_w_u0", 0.0))
    _space = str(getattr(core.decoder, "sampler_space", "control"))
    if w_u0 > 0.0 and "u0_hat" in out and _space == "metre":
        # ⛔⛔ THE DELIBERATE-REGRESSION ARM SAMPLES IN METRES, SO ITS x0
        # TARGET IS THE PATH -- NOT THE CONTROLS. Taking the branch below for
        # this arm would compare METRES against m/s^2 divided by (4.0, 3.0):
        # numerically plausible, silently wrong, and it would corrupt the very
        # arm whose job is to FAIL the flyability gate honestly. A regression
        # that fails for the wrong reason proves nothing, and this is the same
        # units error the branch below exists to prevent, one space over.
        # A metre SCALE for the loss (errors of ~1 m read as ~1). This is the
        # raw published sigma on purpose -- unlike the SAMPLER's normaliser,
        # which must be derived (see `DecoderConfig.metre_sigma_m`).
        m_norm = torch.tensor(tuple(getattr(core.decoder, "metre_sigma_m",
                                            (0.90, 0.73))),
                              device=device, dtype=out["u0_hat"].dtype)
        a_idx = a_star[:, None, None, None].expand(b, 1, traj_tgt.shape[1], 2)
        u_sel = out["u0_hat"].gather(1, a_idx).squeeze(1)           # [B, S, 2]
        loss_u0 = (((u_sel - traj_tgt) / m_norm).abs().sum(-1) * sv).sum() / denom
        loss = loss + w_u0 * loss_u0
        extra["u0"] = loss_u0
    elif w_u0 > 0.0 and "u0_hat" in out:
        u_gt = kin.unicycle_controls_from_path_varstep(
            traj_tgt, kin.slot_dts(core.trajectory.horizons))       # [B, S, 2]
        if core.anchors.control_units == "alat":
            # ⛔ THE TARGET MUST BE IN THE VOCABULARY'S UNITS. The inverse map
            # returns CURVATURE; the bank's controls are LATERAL ACCELERATION
            # when `control_units == "alat"`. Comparing the two directly is the
            # units error `anchor_meta.py` exists to prevent -- a_lat = v^2 k
            # differs from k by a factor of ~1300 at 36 m/s, and BOTH tables
            # look plausible.
            v_ref = v0.clamp_min(core.anchors.alat_v_floor_ms) ** 2
            u_gt = torch.stack([u_gt[..., 0], u_gt[..., 1] * v_ref[:, None]],
                               dim=-1)
        a_idx = a_star[:, None, None, None].expand(b, 1, u_gt.shape[1], 2)
        u_sel = out["u0_hat"].gather(1, a_idx).squeeze(1)           # [B, S, 2]
        norm = torch.tensor(core.decoder.control_norm, device=device,
                            dtype=u_sel.dtype)
        loss_u0 = (((u_sel - u_gt) / norm).abs().sum(-1) * sv).sum() / denom
        loss = loss + w_u0 * loss_u0
        extra["u0"] = loss_u0

    # ---- refcv6 F3: EVERY cascade stage carries the loss ------------------- #
    # ⭐ DD sums `trajectory_loss` over `poses_reg_list` — one term per cascade
    # layer (`transfuser_model_v2.py:492-497`) — and the trajectory handed to
    # the next stage is DETACHED (`:379`). The detach lives in the decoder; the
    # per-stage TERMS live here, because this is where the target is.
    # ⛔ THE LAST STAGE IS EXCLUDED. It is already the emitted fan, already
    # supervised by `loss_traj` + `loss_cls` above; adding it again would
    # double-weight it and the "per-layer loss" arm would be partly a
    # loss-weight arm. Only stages 0..L-2 are new terms.
    if _rv6f.f3_per_layer and "layer_u0_hat" in out:
        stages = out["layer_u0_hat"][:-1]
        stage_logits = out["layer_logits"][:-1]
        if stages:
            # ⛔ `v0` is THIS window's measured speed — the same tensor the
            # forward rolled the bank from (`:2311`). Rolling a stage at the
            # reference speed instead would compare a 6 s path against a target
            # generated at a different initial state, and every stage term
            # would be wrong by a horizon-growing offset that still looks like
            # a loss.
            v_cas = v0.reshape(-1).to(torch.float32)
            l_cas = out["anchor_logits"].new_zeros(())
            for u_i, c_i in zip(stages, stage_logits):
                # geometry: the stage's x0 in control units, rolled by the SAME
                # integrator the fan uses, then matched-anchor L1 in metres.
                p_i = model.core.decoder._state_to_path(
                    u_i, v_cas,
                    str(core.decoder.sampler_space) == "metre")
                r_i = p_i[ar, a_star]
                l_cas = l_cas + (((r_i - traj_tgt).abs().sum(-1) * sv).sum()
                                 / denom)
                l_cas = l_cas + (
                    _rv6.focal_cls_loss(c_i, a_star,
                                        gamma=float(_rv6f.f5_focal_gamma),
                                        alpha=float(_rv6f.f5_focal_alpha))
                    if _rv6f.f5_focal else F.cross_entropy(c_i, a_star))
            loss = loss + l_cas
            extra["cascade"] = l_cas

    # ---- refcv5 WP-6: the GT-supervised detection set loss ----------------
    # ⛔ `obstacle.offline` is a TRAIN-TIME LABEL. It enters HERE, in the loss,
    # and nowhere in the forward -- which is the vision-only rule enforced by
    # where the tensor is read, not by a comment.
    w_agent = float(getattr(model, "_w_agent", 0.0))
    if w_agent > 0.0 and "agent_box" not in batch:
        # ⛔⛔ REFUSE, DO NOT SKIP. MEASURED on the tiny rig 2026-09-05: with
        # `--agents head --w-agent 1.0` on a dataset that carries no
        # `obstacle.offline` join, this branch was a silent `and "agent_box" in
        # batch` no-op -- the run trained, converged, wrote a checkpoint, and
        # stamped `w_agent: 1.0` in config.json while THE DETECTOR WAS NEVER
        # SUPERVISED. The head would have been shaped only by the planner loss
        # through its token gate, and the arm would have read as "the learned
        # agent head does not help".
        # ⭐ This is the SAME failure the --w-agent 0 guard in
        # `_pin_refcv5_seams` refuses, one level down: there the loss weight is
        # missing, here the LABELS are. A guard on the flag alone is not
        # enough, because the flag was set correctly.
        raise SystemExit(
            "[v3] ⛔ --w-agent > 0 but the batch carries no `agent_box`: this "
            "dataset has no obstacle.offline join wired, so the detection loss "
            "would be SILENTLY SKIPPED and the run would stamp w_agent > 0 "
            "while training no detector. Wire the join into the dataset (see "
            "`train_p8_occupancy.JoinFileReader` + "
            "`agent_slots.targets_from_join`), or run --agents oracle / "
            "--w-agent 0.")
    if w_agent > 0.0 and "agent_slots" in out and "agent_box" in batch:
        tgt_ag = {"box": batch["agent_box"].to(device),
                  "yaw": batch["agent_yaw"].to(device),
                  "cls": batch["agent_cls"].to(device),
                  "valid": batch["agent_valid"].to(device),
                  "occ": batch.get("agent_occ",
                                   torch.full_like(batch["agent_yaw"], -1.0)
                                   ).to(device),
                  "rates": batch.get(
                      "agent_rates",
                      torch.zeros(*batch["agent_yaw"].shape, 3)).to(device),
                  "rates_mask": batch.get(
                      "agent_rates_mask",
                      torch.zeros_like(batch["agent_valid"])).to(device)}
        # ⛔⛔ NO_LABEL IS NOT "LABELLED CLEAR", AND THE DIFFERENCE IS A
        # HALLUCINATION EITHER WAY ROUND. An absent (clip, frame) line means
        # the join says NOTHING about that frame (the label span ends ~20 s
        # in); an EMPTY agents list means the road really was clear. Scoring
        # the first as the second trains the presence head to say "empty" on
        # frames full of cars -- the `"no agents"` defect the join doc names.
        # `agent_label` carries the distinction from the dataset and the rows
        # are SELECTED here, so the DETR set loss never sees a NO_LABEL frame.
        keep_ag = batch.get("agent_label")
        ep_ag = batch.get("agent_ep")
        n_lab = (int(keep_ag.sum()) if keep_ag is not None
                 else int(batch["agent_valid"].shape[0]))
        extra["agent_n_windows"] = float(batch["agent_valid"].shape[0])
        extra["agent_n_labelled"] = float(n_lab)
        if keep_ag is not None and n_lab:
            sel = keep_ag.to(device).nonzero(as_tuple=False).flatten()
            # ⛔ THE EPISODE IDS ARE SELECTED WITH THE SAME MASK. A
            # camera list built from the UNSELECTED ids would be the
            # right length only by accident and would attach clip k's
            # mount pose to clip k+1's row -- silently, and with every
            # count still looking healthy.
            if ep_ag is not None:
                ep_ag = ep_ag.index_select(
                    0, keep_ag.nonzero(as_tuple=False).flatten())
            tgt_ag = {k: v.index_select(0, sel) for k, v in tgt_ag.items()}
            slots_ag = {k: (v.index_select(0, sel)
                            if torch.is_tensor(v) and v.shape[:1] ==
                            keep_ag.shape[:1] else v)
                        for k, v in out["agent_slots"].items()}
        else:
            slots_ag = out["agent_slots"]
        if "agent_n_raw" in batch:
            extra["agent_n_raw"] = float(batch["agent_n_raw"].sum())
        if "agent_n_truncated" in batch:
            extra["agent_n_truncated"] = float(
                batch["agent_n_truncated"].sum())
    if w_agent > 0.0 and "agent_slots" in out and "agent_box" in batch \
            and n_lab > 0:
        ag = _refc_agents.agent_losses(
            slots_ag, tgt_ag, core.agents,
            cam=_resolve_rig_cameras(model, ep_ag,
                                     int(tgt_ag["valid"].shape[0])))
        loss = loss + w_agent * ag["total"]
        for k_ag in ("presence", "cls", "centre", "size", "yaw", "project",
                     "ground"):
            if f"loss_{k_ag}" in ag:
                extra[f"agent_{k_ag}"] = ag[f"loss_{k_ag}"]
        # ⭐ n PER TERM, in the log row. A detection number without its n is
        # inadmissible, and `n_dropped` is how a too-small `n_queries` becomes
        # VISIBLE instead of silently flattering the head on crowded frames.
        extra["agent_n_target"] = float(ag["n"]["target"])
        extra["agent_n_matched"] = float(ag["n"]["matched"])
        extra["agent_n_dropped"] = float(ag["n"]["dropped"])
        # ⭐ the camera scope and the rows it could not cover, IN THE
        # LOG ROW. A monocular term computed over 12 of 32 rows is not
        # the term the config says it trained.
        extra["agent_rows_with_cam"] = float(ag["n"]["rows_with_cam"])
        extra["agent_rows_no_cam"] = float(ag["n"]["rows_no_cam"])

    # ---- WP-D: the BEV auxiliary loss -------------------------------------
    # ⛔ `obstacle.offline` enters HERE, in the loss, and nowhere in the
    # forward -- the vision-only rule enforced by WHERE the tensor is read.
    w_bev = float(getattr(model, "_w_bev_aux", 0.0))
    if w_bev > 0.0 and "bev_occ" not in batch:
        # ⛔⛔ REFUSE, DO NOT SKIP -- the `--w-agent` guard one level down, and
        # for the same measured reason: a run that trains, converges, writes a
        # checkpoint and stamps `w_bev_aux` while the head was never supervised
        # would read as "the BEV auxiliary does not help".
        raise SystemExit(
            "[v3] ⛔ --w-bev-aux > 0 but the batch carries no `bev_occ`: this "
            "dataset has no BEV target wired (`ds.bev_spec` is None), so the "
            "auxiliary loss would be SILENTLY SKIPPED while config.json "
            "stamps the weight. Pass --agent-join, or --w-bev-aux 0.")
    if w_bev > 0.0 and "bev_logits" in out and "bev_occ" in batch:
        _bo = batch["bev_occ"].to(device)
        _bm = batch["bev_mask"].to(device)
        if bool(getattr(model, "_bev_shuffle", False)):
            # ⭐ THE INFORMATION CONTROL (--bev-aux-shuffle). A DETERMINISTIC
            # roll, not a random permutation, on purpose: `randperm` would
            # consume RNG and desynchronise this arm's every subsequent draw
            # from the real arm's, so the control would differ in the seed as
            # well as in the information — the same one-variable violation the
            # head's construction order exists to avoid. The training loader
            # runs with shuffle=True, so row i-1 is an unrelated window.
            _bo = torch.roll(_bo, shifts=1, dims=0)
            _bm = torch.roll(_bm, shifts=1, dims=0)
        _bv = _refc_bev_aux.bev_aux_loss(
            out["bev_logits"], _bo, _bm,
            pos_weight=float(core.bev_aux.pos_weight))
        loss = loss + w_bev * _bv["loss"]
        extra["bev"] = _bv["loss"]
        # ⭐ n PER TERM, in the log row, always. `bev_n_supervised` is how a
        # batch of NO_LABEL windows becomes VISIBLE as an exact 0.0 with its
        # reason, instead of a term that quietly contributes nothing; and
        # `bev_n_pos` is the base rate the controls must be read against.
        extra["bev_n_supervised"] = float(_bv["n_supervised"])
        extra["bev_n_pos"] = float(_bv["n_pos"])

    # ---- D-TACGOAL: the 22-token tactical goal SET ------------------------
    # ⭐⭐ THIS IS THE SEAM THAT WAS OPEN. Until 2026-09-09 this file contained
    # ZERO occurrences of `tac_goal_loss` and ZERO of `TacGoalEmitter`, so
    # `refc_v3.py:1320` wrote `cache["tac_goal_logits"]` into a dict nothing
    # read: the head was built (`--tac-goal-tok-head`), stamped
    # (`_seams["tac_goal_tok_head"]["built"]`), forward-run, and had NO
    # GRADIENT PATH. refcv5-v2 trained it for 40,284 steps at grad_abs_sum
    # exactly 0.0.
    #
    # ⛔ THE `w <= 0` BRANCH IS AN ABSENCE, NOT A MULTIPLICATION BY ZERO. The
    # term never enters the graph, so a recipe that does not pass
    # `--w-tac-goal` is BIT-IDENTICAL to the pre-wiring trainer at a fixed
    # seed. That identity is the whole reason this may land beside a live
    # recipe without a rebalancing decision (which is the PI's, queue item 10,
    # and explicitly NOT taken here).
    #
    # ⚠️ AND THE AMBIGUITY THAT COSTS -- named, because `tac_goal_head.py`
    # warns about it in the opposite direction. A guarded term makes
    # `p.grad is None`, which reads IDENTICALLY to "never wired". Here that
    # confusion cannot survive, because the head now HAS A WEIGHT and
    # therefore a ROW in `effective_weights_stamp_v3`: `w_tac_goal = 0.0` in
    # config.json is a POSITIVE RECORD that the term was switched off, which
    # is exactly the evidence the 2026-09-07 census did not have. Inside the
    # channel nothing is guarded -- `tac_goal_loss` divides by a clamped
    # denominator on purpose, so an all-ignored batch still returns a real
    # zero WITH `n_supervised`, and the head still receives a gradient tensor.
    _w_tg = float(getattr(model, "_w_tac_goal", 0.0) or 0.0)
    if _w_tg > 0.0:
        # ⛔⛔ REFUSE, DO NOT SKIP -- the `--w-bev-aux` guard immediately above,
        # and the `--w-agent` guard below it, for the one measured reason:
        # a run that trains, converges, writes a checkpoint and STAMPS
        # `w_tac_goal` while the head was never supervised would read as
        # "the tactical goal set does not help". That is the failure this
        # whole package exists to close, re-manufactured one layer up.
        if "tac_goal_logits" not in out:
            raise SystemExit(
                "[v3] ⛔ --w-tac-goal > 0 but `out` carries no "
                "`tac_goal_logits`: the 22-token head was never built. Pass "
                "--tac-goal-tok-head (with --v7-labels, which pins the v7.0 "
                "tactical vocabulary), or --w-tac-goal 0.")
        if "tac_goal_y" not in batch or "tac_goal_w" not in batch:
            raise SystemExit(
                "[v3] ⛔ --w-tac-goal > 0 but the batch carries no "
                "`tac_goal_y`/`tac_goal_w`: this dataset has no goal-set "
                "target wired (`ds.tac_goal_targets` is False), so the loss "
                "would be SILENTLY SKIPPED while config.json stamps the "
                "weight -- the `w_agent` defect verbatim. Pass --v7-labels, "
                "or --w-tac-goal 0.")
        _tg_loss, _tg_n = _tac_goal_head.tac_goal_loss(
            out["tac_goal_logits"],
            batch["tac_goal_y"].to(device), batch["tac_goal_w"].to(device),
            pos_weight=getattr(model, "_tac_goal_pos_weight", None),
            class_mask=getattr(model, "_tac_goal_class_mask", None))
        loss = loss + _w_tg * _tg_loss
        extra["tac_goal"] = _tg_loss
        # ⭐ n PER TERM, IN THE LOG ROW, ALWAYS. `tac_goal_loss` returns
        # `n_supervised` precisely so a 0.0 can be read as "nothing was in
        # band" rather than "the head is broken" -- the route-head trap its
        # own docstring names (an apparent zero gradient that was a validity
        # MASK, 0.0 -> 0.687 when forced). A bare 0.0 without this number
        # reads as "supervised, and perfect".
        extra["tac_goal_n_supervised"] = float(_tg_n)
        extra["tac_goal_n_pos"] = float(
            ((batch["tac_goal_y"] > 0.5) & (batch["tac_goal_w"] > 0)).sum())

    # ======================================================================= #
    # refcv6 §4 — THE TACTICAL BEHAVIOUR DECODER'S OWN LOSSES                  #
    # ======================================================================= #
    # ⭐⭐ THIS IS THE SEAM THAT WAS OPEN, AND IT WAS OPEN 200x WIDER THAN THE
    # ONE ABOVE. MEASURED at tip 837c308: this file contained ZERO occurrences
    # of `tacv6`, `loss_tacv6` and `behaviour_loss`, while `refc_v3.py:1128`
    # BUILT a 2,262,020-parameter decoder (d_bev 256), `:1622` ran it, and
    # `:1627` wrote `cache["tacv6_*"]` into a dict nothing read. Every one of
    # those parameters sat at `grad_abs_sum` EXACTLY 0.
    #
    # ⛔⛔ AND THE LAYER CAN NEVER LEARN THROUGH THE PLANNER, BY DESIGN.
    # `refcv6_tactical.planner_feeds` detaches all four feeds, and that is
    # CORRECT — with the feed attached, the cheapest way to lower the
    # trajectory loss is to reshape the behaviour head into whatever
    # correlates with the trajectory, and its per-class numbers would then
    # stop measuring what the LABELS taught it. ⇒ The detachment is precisely
    # why this block has to exist: these losses are the layer's ONLY gradient.
    #
    # ⛔ THE `w <= 0` BRANCH IS AN ABSENCE, NOT A MULTIPLICATION BY ZERO — the
    # `--w-tac-goal` rule verbatim. The term never enters the graph, so a
    # recipe that does not pass `--w-tac-v6` is BIT-IDENTICAL to the pre-wiring
    # trainer at a fixed seed. INSIDE the channel nothing is guarded:
    # `tactical_behaviour_losses` computes every term unconditionally and
    # applies the weight by MULTIPLICATION, so `p.grad is None` stays a clean
    # discriminator for "never wired" (42 of 138 optimizer tensors read None
    # on 2026-09-06 because objectives were weighted 0.0 AND guarded).
    _w_t6 = float(getattr(model, "_w_tac_v6", 0.0) or 0.0)
    if _w_t6 > 0.0:
        # ⛔⛔ REFUSE, DO NOT SKIP. A run that trains, converges, writes a
        # checkpoint and STAMPS `w_tac_v6` while the decoder was never
        # supervised would read as "the tactical layer does not help" — the
        # exact refutation this package exists to make impossible.
        if "tacv6_goal_logits" not in out:
            raise SystemExit(
                "[v3] ⛔ --w-tac-v6 > 0 but `out` carries no "
                "`tacv6_goal_logits`: the behaviour decoder was never built "
                "or its scene hook never fired. Pass --tac-decoder-v6 (with "
                "--v7-labels and --agents), or --w-tac-v6 0.")
        if "tac_goal_y" not in batch or "tac_goal_w" not in batch:
            raise SystemExit(
                "[v3] ⛔ --w-tac-v6 > 0 but the batch carries no "
                "`tac_goal_y`/`tac_goal_w`: this dataset has no goal-set "
                "target wired (`ds.tac_goal_targets` is False), so the "
                "22-token BCE would be SILENTLY SKIPPED while config.json "
                "stamps the weight. Pass --v7-labels, or --w-tac-v6 0.")
        if "lat_v7" not in batch or "lon_v7" not in batch:
            raise SystemExit(
                "[v3] ⛔ --w-tac-v6 > 0 but the batch carries no "
                "`lat_v7`/`lon_v7`: the 8+8 action queries would receive no "
                "target at all. Pass --v7-labels, or --w-tac-v6 0.")
        _t6_loss, _t6_tele = v6tac.tactical_behaviour_losses(
            {"goal_logits": out["tacv6_goal_logits"],
             "goal_conf": out["tacv6_goal_conf"],
             "lat_logits": out["tacv6_lat_logits"],
             "lon_logits": out["tacv6_lon_logits"]},
            goal_y=batch["tac_goal_y"].to(device),
            goal_w=batch["tac_goal_w"].to(device),
            lat_target=batch["lat_v7"].to(device),
            lon_target=batch["lon_v7"].to(device),
            # ⛔ THE SAME pos_weight AND class_mask AS `--w-tac-goal`, fitted
            # on the TRAIN split and carried on the model. Re-deriving a
            # second copy here is the derived-constant trap; using a literal
            # would be worse.
            goal_pos_weight=getattr(model, "_tac_goal_pos_weight", None),
            goal_class_mask=getattr(model, "_tac_goal_class_mask", None),
            ignore_index=v7l.IGNORE_ID)
        loss = loss + _w_t6 * _t6_loss
        # ⭐⭐ THE TACTICAL TERM, EXPOSED TO THE GRADIENT-CONFLICT DETECTOR.
        # PI RULING 2026-09-17 R3 names the detector as the mitigation for a
        # trunk optimised FOUR ways — and MEASURED on this patch's own rig, the
        # tactical loss reaches the trunk with `grad_abs_sum` 78,146 even on the
        # AGENT-ONLY arm, while `CONFLICT_PERCEPTION_TERMS` listed only
        # bev / map / box3d. So the detector has been BLIND to this gradient
        # since the term existed; R3 is what makes that blindness load-bearing.
        # ⛔ The UNWEIGHTED tensor, keyed `tac_v6`: `_conflict_terms` multiplies
        # by the arm's own weight, so handing it a pre-weighted tensor would
        # square the weight and silently misreport the ratio.
        extra["tac_v6"] = _t6_loss
        # ⭐ n PER TERM, ALWAYS, AND PER TERM MEANS THREE NUMBERS HERE. A bare
        # 0.0 on the lat/lon heads reads as "supervised, and perfect", when
        # the normal case is that the window is OUTSIDE the record's ±2 s
        # band: MEASURED 1,157 of 4,823 eval windows are in band, so most
        # batches legitimately supervise the action heads on a minority of
        # rows and `n_supervised` is the only thing that says so.
        extra["tacv6_goal_bce"] = _t6_tele["tac_goal_bce"]
        extra["tacv6_goal_conf_bce"] = _t6_tele["tac_goal_conf_bce"]
        extra["tacv6_lat_ce"] = _t6_tele["tac_lat_ce"]
        extra["tacv6_lon_ce"] = _t6_tele["tac_lon_ce"]
        extra["tacv6_total_weighted"] = _t6_tele["tac_total_weighted"]
        extra["tacv6_n_supervised_goal_cells"] = float(
            _t6_tele["n_supervised_goal_cells"])
        extra["tacv6_n_supervised_lat"] = float(_t6_tele["n_supervised_lat"])
        extra["tacv6_n_supervised_lon"] = float(_t6_tele["n_supervised_lon"])
        # ⚠️ `tacv6_n_scene` is the decoder's ATTENDED key count. A row that
        # attends to ZERO keys is a scene that reached the decoder empty, and
        # it would otherwise be invisible: the losses are all finite and the
        # arm reads as "behaviours cannot be learned from the scene".
        if "tacv6_n_scene" in out:
            extra["tacv6_n_scene_mean"] = out["tacv6_n_scene"].to(
                torch.float32).mean()

    # ======================================================================= #
    # refcv6 §2/§6 — THE PERCEPTION BRANCH: SAM3 BEV MAP + 3-D CUBOIDS        #
    # ======================================================================= #
    # The PI's central refcv6 instruction (2026-09-16): *"train jointly the
    # resnet-trunk, the bev map (based on the sam2 maps as gt) and a head for
    # 3d bounding boxes extracted from the resnet trunk"*.
    #
    # ⛔ BOTH ARE TRAIN-TIME LABELS AND THEY ENTER HERE, IN THE LOSS, AND
    # NOWHERE IN THE FORWARD. The branch's own `forward` takes ONE argument
    # family (`fmap_s16`, and the camera geometry) and there is no parameter
    # through which a label could arrive — the vision-only rule enforced by
    # where the tensor is read, not by a comment.
    #
    # ⛔ SAM3 maps ONLY. LiDAR is NOT a training target (PI); it may be quoted
    # as an INDEPENDENT EVALUATION REFERENCE and nothing here reads it.
    #
    # ⭐ At both weights 0.0 this whole block is skipped, `_perception` was
    # never built, and the returned dict carries no `map`/`box3d` key — the
    # bit-identity condition, enforced by construction rather than asserted.
    _w_map = float(getattr(model, "_w_map", 0.0))
    _w_b3d = float(getattr(model, "_w_box3d", 0.0))
    if _w_map > 0.0 or _w_b3d > 0.0:
        _br = getattr(model, "_perception", None)
        if _br is None:
            raise SystemExit(
                "[v3] ⛔ --w-map/--w-box3d > 0 but no perception branch was "
                "attached to the model. The weights would be stamped into "
                "config.json and multiply nothing — the `--w-agent` defect "
                "verbatim. This is a trainer wiring error, not an argv one.")
        # ⛔ REFUSE, DO NOT SKIP — the same rule as `agent_box` above. A batch
        # without the targets means the loader was never given the labels, and
        # a silent skip would train the trunk on the planner alone while the
        # record claims a jointly-trained perception branch.
        if _w_map > 0.0 and "map_frac" not in batch:
            raise SystemExit(
                "[v3] ⛔ --w-map > 0 but the batch carries no `map_frac`: "
                "this dataset has no SAM3 map GT attached "
                "(`ds.enable_map_gt` was never called). Pass --map-gt-root.")
        if _w_b3d > 0.0 and "agent_box" not in batch:
            raise SystemExit(
                "[v3] ⛔ --w-box3d > 0 but the batch carries no `agent_box`: "
                "the 3-D set loss is the 2-D set loss plus two masked metre "
                "terms, and without the 2-D targets there is nothing to match "
                "against. Pass --agent-join.")
        # ---- ONE branch forward for BOTH heads, on THIS batch's graph ----- #
        # ⚠️ `out["fmap_s16"]` is the stride-16 map of the LAST OBSERVED frame
        # (`refc.py:3869/3883`) — the same instant `t + w - 1` the map target
        # and the agent target are read at.
        # ⭐⭐⭐ PI RULING 2026-09-17 R2 — THE BRANCH ALREADY RAN, INSIDE THE
        # FORWARD. It used to be called HERE, after `model(...)` returned; that
        # ordering is exactly what made §4's map half unreachable (no BEV token
        # existed when the scene hook fired). The branch now runs in
        # `RefCV3Model._bev_hook`, so this reads its outputs off `out`.
        # ⛔ ONE FORWARD, ONE GRAPH — never two. Re-running `_br(...)` here
        # would build a SECOND lift/encoder/map-head graph on the same
        # `fmap_s16`, double the perception gradient into the trunk relative to
        # the planner's, and leave the behaviour decoder reading a DIFFERENT
        # BEV tensor from the one the map loss scores. Both arms would look
        # healthy and the attribution would be silently wrong.
        _pout = out.get("perception")
        if _pout is None:
            raise SystemExit(
                "[v3] ⛔ --w-map/--w-box3d > 0 but `out['perception']` is "
                "None: the model's forward did not run the perception branch. "
                "Either this core predates the `bev_hook` seam, or the model "
                "is the FLAT arm whose forward returns early. The losses below "
                "would have nothing to read while config.json stamped a "
                "jointly-trained perception branch.")
        if _w_map > 0.0:
            _mlab = batch["map_label"].to(device)
            _msel = _mlab.nonzero(as_tuple=False).flatten()
            extra["map_n_windows"] = float(_mlab.shape[0])
            extra["map_n_labelled"] = float(int(_mlab.sum()))
            if int(_mlab.sum()):
                _mrow = _perc.map_loss_row(
                    _pout["map_logits"].index_select(0, _msel),
                    batch["map_frac"].to(device).index_select(0, _msel),
                    batch["map_seen"].to(device).index_select(0, _msel))
                loss = loss + _w_map * _mrow["loss"]
                extra["map"] = _mrow["loss"]
                extra["n_map_cells"] = _mrow["n_map_cells"]
            else:
                # ⛔ A COUNTED ZERO, never an absent key. `map = 0.0` with
                # `n_map_cells = 0` is "no window in this batch had a label";
                # a missing key would read as "the head is off".
                extra["map"] = out["fmap_s16"].sum() * 0.0
                extra["n_map_cells"] = 0.0
        if _w_b3d > 0.0:
            _b3_keep = batch.get("agent_label")
            _b3_n = (int(_b3_keep.sum()) if _b3_keep is not None
                     else int(batch["agent_valid"].shape[0]))
            extra["box3d_n_windows"] = float(batch["agent_valid"].shape[0])
            extra["box3d_n_labelled"] = float(_b3_n)
            if _b3_n:
                _sel3 = (_b3_keep.to(device).nonzero(as_tuple=False).flatten()
                         if _b3_keep is not None else
                         torch.arange(batch["agent_valid"].shape[0],
                                      device=device))
                _z = batch.get("agent_zh_mask")
                _t3 = {
                    "box": batch["agent_box"].to(device),
                    "yaw": batch["agent_yaw"].to(device),
                    "cls": batch["agent_cls"].to(device),
                    "valid": batch["agent_valid"].to(device),
                    "occ": batch.get("agent_occ",
                                     torch.full_like(batch["agent_yaw"], -1.0)
                                     ).to(device),
                    "rates": batch.get(
                        "agent_rates",
                        torch.zeros(*batch["agent_yaw"].shape, 3)).to(device),
                    "rates_mask": batch.get(
                        "agent_rates_mask",
                        torch.zeros_like(batch["agent_valid"])).to(device),
                    # ⛔ An absent 3-D label is a MASK, never a zero. With no
                    # --join3d this is all-False and `box3d_set_loss` reports
                    # `n["z"] == 0` — the 2-D total EXACTLY, by its own test.
                    "cz": batch.get(
                        "agent_cz",
                        torch.zeros_like(batch["agent_yaw"])).to(device),
                    "h": batch.get(
                        "agent_h",
                        torch.zeros_like(batch["agent_yaw"])).to(device),
                    "zh_mask": (torch.zeros_like(batch["agent_valid"])
                                if _z is None else _z).to(device)}
                _t3 = {k: v.index_select(0, _sel3) for k, v in _t3.items()}
                _s3 = {k: (v.index_select(0, _sel3)
                           if torch.is_tensor(v)
                           and v.shape[:1] == _b3_keep.shape[:1] else v)
                       for k, v in _pout["box_slots"].items()} \
                    if _b3_keep is not None else _pout["box_slots"]
                _brow = _perc.box3d_loss_row(_s3, _t3)
                loss = loss + _w_b3d * _brow["loss"]
                extra["box3d"] = _brow["loss"]
                for _k3, _v3 in _brow.items():
                    if _k3 != "loss":
                        extra[_k3] = _v3
            else:
                extra["box3d"] = out["fmap_s16"].sum() * 0.0
                extra["box3d_n_z"] = 0.0

    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,
            "route": loss_route, "lat": loss_lat, "lon": loss_lon,
            "lat_tac": loss_lat_tac, "lon_tac": loss_lon_tac,
            "anchor_acc": (out["anchor_logits"].argmax(1) == a_star)
            .float().mean(), "slot_valid_frac": sv.mean(), **extra}


# ============================================================================
# Preflight — everything that must be true BEFORE a GPU day
# ============================================================================

def _lan_arm_preflight(cfg, args) -> int:
    """Exercise the LAN → ``goal_str`` pathway BEFORE a GPU day.

    ⚠️ THIS ARM EXISTS BECAUSE THE PREFLIGHT COULD NOT SEE THE THING IT WAS
    BEING CITED FOR. MEASURED (D-LAN-PF, this package): ``preflight()`` builds
    a bare ``V3Dataset``, which emits no ``lan`` key, so ``compute_losses_v3``
    skips ``loss_gstr`` BY DESIGN and ``goal_str`` never appears in the loss
    dict at all — with or without ``--goal-str``. The quantity people read
    instead, ``route``, is the v2.1 NAV-derived CE masked by ``nav_valid``; it
    is bit-identical with and without every LAN flag and can never testify
    about LAN. Same class as C9/C13/C14: an instrument structurally unable to
    report the answer it is quoted for.

    Three checks, in the order that makes a failure diagnosable:
      1. the dataset actually emits ``lan`` at the pinned width;
      2. the LABEL is live — ``any_valid_frac`` > 0. A ``goal_str`` computed
         over an all-invalid label is 0.0 and looks like a pass;
      3. ``goal_str`` is PRESENT, FINITE and NON-ZERO.
    Then a DELIBERATE-REGRESSION control (programme 6.2): with the leak guard
    pushed to +inf the label must go dead and ``goal_str`` must collapse to
    0.0. A check that cannot fail is not evidence, so the control runs every
    time and its own failure fails the preflight.
    """
    k = len(args.lan_arclengths)
    lan_cfg = DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                            min_lead_m=args.lan_min_lead_m)
    cfg.core.lan = refc.LanConfig(k=k)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)

    # Long enough that the leak guard is a GUARD and not a wall — see
    # _synth_episodes.__doc__ for the arithmetic that sets this floor.
    # ⚠️ Derived from the ARC-LENGTHS ONLY, deliberately NOT from min_lead_m:
    # the regression control below cranks min_lead_m to +inf, and a corpus that
    # grew with it would change TWO things at once — the control would compare
    # a different corpus rather than isolate the guard. The cap keeps a
    # pathological --lan-arclengths from asking for an unallocatable corpus
    # (at 4000 frames the reachable path is ~800 m, so anchors beyond that
    # still read dead, which is the correct verdict and not an evasion).
    need_m = max(args.lan_arclengths) + 50.0
    min_frames = min(int(need_m / (2.0 * 0.1)) + 4 * cfg.core.window, 4000)
    eps = _synth_episodes(2, cfg.core, seed=0, min_frames=min_frames)

    def _batch(lc):
        ds = lan_dataset_class(V3Dataset)(
            eps, window=cfg.core.window, max_horizon=20,
            channels=cfg.core.encoder.in_channels, lan_cfg=lc)
        return (torch.utils.data.default_collate([ds[0], ds[1]]),
                ds.lan_stats(n=512, seed=0))

    batch, stats = _batch(lan_cfg)
    if "lan" not in batch:
        print("[v3-preflight] ⛔ FAIL: lan dataset emitted no `lan` key")
        return 6
    if tuple(batch["lan"].shape[1:]) != (k * 4,):
        print(f"[v3-preflight] ⛔ FAIL: lan width {tuple(batch['lan'].shape)} "
              f"!= pinned {k * 4}")
        return 6
    print(f"[v3-preflight] lan label coverage: {json.dumps(stats)}")
    if stats["any_valid_frac"] <= 0.0:
        print("[v3-preflight] ⛔ FAIL: every LAN anchor is masked — `goal_str` "
              "would be 0.0 over an ALL-INVALID label and read as a pass. The "
              "strategic goal head would train on nothing (C9 class).")
        return 6

    losses = compute_losses_v3(model, batch, "cpu", mode="diffusion")
    gs = losses.get("goal_str")
    if gs is None:
        print("[v3-preflight] ⛔ FAIL: `goal_str` absent from the loss dict — "
              "the LAN pathway did not fire even though `lan` was present.")
        return 6
    gs = float(gs.detach())
    if not (gs == gs and abs(gs) != float("inf")):
        print(f"[v3-preflight] ⛔ FAIL: goal_str non-finite ({gs})")
        return 6
    if gs == 0.0:
        print("[v3-preflight] ⛔ FAIL: goal_str is exactly 0.0 on a LIVE label "
              "— the strategic head carries no gradient.")
        return 6

    # ---- deliberate-regression control: the check must be ABLE to fail ------
    dead_cfg = DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                             min_lead_m=1e9)
    dead_batch, dead_stats = _batch(dead_cfg)
    dead_gs = float(compute_losses_v3(model, dead_batch, "cpu",
                                      mode="diffusion")["goal_str"].detach())
    if dead_stats["any_valid_frac"] != 0.0 or dead_gs != 0.0:
        print(f"[v3-preflight] ⛔ FAIL: the regression control did NOT go dead "
              f"(valid_frac={dead_stats['any_valid_frac']}, "
              f"goal_str={dead_gs}) — this preflight cannot detect a dead LAN "
              f"label, so its PASS means nothing.")
        return 6
    print(f"[v3-preflight] lan arm OK: goal_str={gs:.4f} live / "
          f"{dead_gs:.4f} under the +inf-guard control "
          f"(control able to fail: True)")
    return 0



def _read_anchor_artifact(args):
    """Read ``--anchors`` through :mod:`tanitad.refs.anchor_meta` and RESOLVE
    its control units against the explicit ``--anchor-control-units`` override.

    Returns ``None`` without ``--anchors``. Otherwise it (i) refuses -- before
    any data or GPU work -- a ``controls``-carrying file that declares no
    ``control_units`` unless the override is given (the LIVE refcv4b file is
    exactly that case and keeps loading through ``--anchor-control-units
    alat``), (ii) writes the resolved units back into
    ``args.anchor_control_units`` so ``_pin_trainer_cfg`` sees ONE value, and
    (iii) stashes the file's declared derivation constants in
    ``args._anchor_artifact_meta`` for the pin to adopt.
    """
    if not getattr(args, "anchors", None):
        return None
    from tanitad.refs import anchor_meta
    try:
        art = anchor_meta.read_anchor_artifact(
            args.anchors,
            cli_control_units=getattr(args, "anchor_control_units", None))
    except anchor_meta.AnchorUnitsError as e:
        raise SystemExit(f"[v3] ⛔ anchors: {e}")
    if art.controls is not None:
        args.anchor_control_units = art.control_units
    args._anchor_artifact_meta = dict(art.declared)
    print(f"[v3] anchors: artifact {anchor_meta.describe(art)}", flush=True)
    return art


def _check_anchor_artifact_against_cfg(art, cfg, args) -> None:
    """Refuse a file whose DECLARED horizon / dt / reference speed differ from
    what the decoder built from ``cfg`` will use. Undeclared fields (a legacy
    file) never fire; a declared one that differs is a second vocabulary
    wearing the first one's name, so it is refused rather than reconciled."""
    if art is None:
        return
    from tanitad.refs import anchor_meta
    hz = tuple(cfg.core.trajectory.horizons)
    bad = anchor_meta.mismatches(
        art, horizon_s=max(hz) * 0.1, dt=0.1,
        ref_speed_ms=(float(cfg.core.anchors.ref_speed_ms)
                      if art.controls is not None else None),
        kappa_cap=(float(cfg.core.anchors.kappa_cap)
                   if art.controls is not None else None),
        alat_v_floor=(float(cfg.core.anchors.alat_v_floor_ms)
                      if art.controls is not None else None))
    if bad:
        raise SystemExit(
            "[v3] ⛔ anchors: the artifact was built for different constants "
            "than this decoder would re-roll it under -- " + "; ".join(bad)
            + ". Rebuild the vocabulary for this trainer, or launch with the "
              "matching flags (--anchor-ref-speed for ref_speed_ms; kappa_cap "
              "and alat_v_floor are adopted from the file).")


def _seam_stamp(cfg, args) -> dict:
    """The hierarchy-seam booleans, serialised so a finished run can rebuild
    its own model config from its own record.

    MEASURED 2026-09-04 (`…/2026-09-04-refcv4b-seam-state/SEAM_STATE.md`):
    `hierarchy`, `graft_maneuver`, `factored_maneuver`, `graft_prior_center`,
    `lan_enable`, `goal_str` were absent from config.json at EVERY nesting
    level; the live arm was reconstructible only by knowing that refc_v3.py
    forces them. A run record that cannot rebuild its own model config is not
    a run record. `lan_enable` is the LAN LABEL pathway (`--goal-str` or
    `--graft-lan` builds it); `graft_lan` is the corridor as a MODEL INPUT,
    refused by E12 for every registered arm and stamped separately so the two
    can never be confused.
    """
    core = cfg.core
    return {
        "hier": bool(cfg.hier),
        "hierarchy": bool(core.hierarchy),
        # ⭐⭐ THE STRATEGIC BYPASS, AND ITS ENTAILED LOSS CHANGE, IN THE
        # RECORD. `hierarchy` alone can no longer say which arm this was: a
        # bypassed run STILL reads `hierarchy: true` (that is the whole
        # bypass-never-delete design), so without this key a finished run
        # cannot answer "did the strategic layer reach the plan?" from its own
        # record -- the exact failure SEAM_STATE.md MEASURED for six seams.
        "no_strategic": bool(getattr(core, "no_strategic", False)),
        # ...and the loss it implies, stamped separately so the two can never
        # be confused: with the bypass on, NOTHING consumes `g_str`, so
        # supervising it would train a head that cannot affect the plan while
        # still shaping the SHARED ENCODER -- a second variable hiding inside
        # a one-variable arm.
        "goal_str_loss_applied": bool(
            getattr(args, "goal_str", False)
            and not getattr(core, "no_strategic", False)),
        "graft_maneuver": bool(core.graft_maneuver),
        "factored_maneuver": bool(core.factored_maneuver),
        "graft_prior_center": bool(core.graft_prior_center),
        "graft_target_latent": bool(core.graft_target_latent),
        "grounded_selector": bool(core.grounded_selector),
        "graft_imagination": bool(core.graft_imagination),
        # ⭐⭐ WP-D: the BEV auxiliary head, STRUCTURALLY and not only inside
        # `agent_knobs`. ⛔ `bev_aux_occlusion` in particular is what
        # distinguishes the pre-registered arm from its DELIBERATE-REGRESSION
        # twin (`none` supervises occluded space as free, mislabelling a
        # MEASURED 27.958 % of occupied cells), and a run record that cannot
        # say which of the two it was makes the panel unfalsifiable — the M18
        # finding with a different knob in it.
        "bev_aux": (None if getattr(core, "bev_aux", None) is None
                    else dict(core.bev_aux.to_dict(),
                              shuffled_target=bool(
                                  getattr(args, "bev_aux_shuffle", False)))),
        "tactical_speed_input": bool(core.tactical_speed_input),
        "lan_enable": bool(getattr(args, "goal_str", False)
                           or getattr(args, "graft_lan", False)),
        "graft_lan": bool(core.graft_lan or getattr(args, "graft_lan", False)),
        "goal_str": bool(getattr(args, "goal_str", False)),
        # ⭐ H-EGO-LIT-4 (2026-09-05): the withheld-row bank policy. A run that
        # does not stamp it cannot say which of the four arms it was.
        "withheld_bank": str(getattr(args, "withheld_bank", "fixed")),
        "withheld_bank_warmup": int(getattr(args, "withheld_bank_warmup", 0)),
        "withheld_speed_max_ms": float(getattr(args, "withheld_speed_max",
                                               35.0)),
        # ⭐ refcv5 (2026-09-05). A run that does not stamp these cannot say
        # which arm it was -- the exact failure `SEAM_STATE.md` MEASURED for
        # six pre-existing seams, where the live arm was reconstructible only
        # by knowing what refc_v3.py forces.
        "agents": (core.agents.as_dict()
                   if getattr(core, "agents", None) is not None else None),
        "cross_agent": bool(getattr(core.decoder, "cross_agent", False)),
        # ⭐⭐ WP-B: the waypoint index, STRUCTURALLY. ⛔ `mode` in particular
        # is what a reader needs to tell the TREATMENT from the three CONTROLS
        # (`shuffle` / `const` / `detach`), and a panel that cannot say which
        # of them ran is unfalsifiable. `None` when the seam is off, so the two
        # states are distinguishable in the record and not merely by absence.
        "wp_index": (core.decoder.wp_index.as_dict()
                     if getattr(core.decoder, "wp_index", None) is not None
                     else None),
        "sampler": str(getattr(core.decoder, "sampler", "none")),
        "sampler_space": str(getattr(core.decoder, "sampler_space", "control")),
        "sampler_infer_t": int(getattr(core.decoder, "sampler_infer_t", 8)),
        "sampler_steps": int(getattr(core.decoder, "sampler_steps", 2)),
        "sampler_groups": int(getattr(core.decoder, "sampler_groups", 1)),
        "control_norm": list(getattr(core.decoder, "control_norm", (4.0, 3.0))),
        # ⭐⭐ refcv6 §2 + §3. A run that does not stamp these cannot say which
        # trunk it trained or which of F1..F9 were live — the same
        # unfalsifiability `SEAM_STATE.md` measured for six earlier seams.
        # `refcv6: null` is the BASELINE and is distinguishable from absence.
        "trunk": str(getattr(core.encoder, "trunk", "refc")),
        "trunk_name": str(getattr(core.encoder, "trunk_name",
                                  "resnet34.a1_in1k")),
        "trunk_mode": str(getattr(core.encoder, "trunk_mode", "shared")),
        "trunk_fuse": str(getattr(core.encoder, "trunk_fuse", "concat1x1")),
        "trunk_fuse_identity": bool(getattr(core.encoder,
                                            "trunk_fuse_identity", True)),
        "trunk_frames": int(core.encoder.in_channels) // 3,
        "trunk_pretrained": bool(getattr(core.encoder, "trunk_pretrained",
                                         True)),
        "trunk_imagenet_norm": bool(getattr(core.encoder,
                                            "trunk_imagenet_norm", True)),
        "trunk_in_channels": int(core.encoder.in_channels),
        # ⭐ refcv6 §2b. `null` is the baseline and is distinguishable from
        # absence — the `wp_index` convention, for the same reason.
        "ego_history": (core.ego_history.as_dict()
                        if getattr(core, "ego_history", None) is not None
                        else None),
        "opt": str(getattr(args, "opt", "adam")),
        "weight_decay": float(getattr(args, "weight_decay", 1e-4)),
        "encoder_lr_mult": float(getattr(args, "encoder_lr_mult", 0.5)),
        "refcv6": (_rv6.flag_stamp(core.decoder.refcv6)
                   if getattr(core.decoder, "refcv6", None) is not None
                   else None),
        # ⭐ STAGE 0. A run that does not stamp these cannot say whether its
        # fan was projected -- and a projected fan reads `envelope 0.0000` as
        # an IDENTITY, which is indistinguishable in a metrics table from a
        # model that learned to be safe. The record must carry which one it is.
        "feasible_decode": bool(getattr(core.decoder, "feasible_decode",
                                        False)),
        "feasible_mu": float(getattr(core.decoder, "feasible_mu", 0.7)),
        "feasible_entry": bool(getattr(core.decoder, "feasible_entry", False)),
        "feasible_a_max": float(getattr(core.decoder, "feasible_a_max", 4.0)),
        "feasible_kappa_max": float(getattr(core.decoder, "feasible_kappa_max",
                                            0.2)),
        "feasible_prefix_slots": int(getattr(core.decoder,
                                             "feasible_prefix_slots", 4)),
        "w_agent": float(getattr(args, "w_agent", AGENT_WEIGHT_DEFAULT)),
        "w_u0": float(getattr(args, "w_u0", U0_WEIGHT_DEFAULT)),
        # ⭐⭐ E15 (GP-2, 2026-09-06) — THE GOAL-POINT STAMP, INCLUDING THE
        # MACHINE-READABLE ADMISSIBILITY DECLARATION.
        #
        # ⛔ `goal_point_provenance()` is written into the RUN RECORD rather
        # than left in a docstring, because the PI's ruling asks a question
        # ("could this goal have been computed from the situation classifier's
        # output?") that a reader must be able to answer from the artifact
        # alone, months later, without the source tree. It is the same reason
        # the ego block, the nav source and the anchor units are stamped.
        #
        # `replaces_nav_inject` is the one a reader would otherwise have to
        # know a trainer line for: `--goal-point-inject` turns E13's
        # categorical nav OFF, by pre-registration (PREREG §2, ONE VARIABLE).
        "goal_point": {
            "goal_point_inject": bool(getattr(cfg, "goal_point_inject",
                                              False)),
            "graft_gp_point": bool(getattr(core, "graft_gp_point", False)),
            "t_goal_s": float(cfg.goal_point_cfg.t_goal_s),
            "t_pred_s": float(cfg.goal_point_cfg.t_pred_s),
            "range_norm_m": float(cfg.goal_point_cfg.range_norm_m),
            "gp_slot": int(getattr(core, "gp_slot", -1)),
            "gp_scale_m": float(getattr(core, "gp_scale_m", 0.0)),
            "w_goal_point": float(getattr(args, "goal_point_w",
                                          GOAL_POINT_WEIGHT_DEFAULT)),
            "replaces_nav_inject": bool(getattr(cfg, "goal_point_inject",
                                                False)),
            "nav_inject": bool(getattr(cfg, "nav_inject", True)),
            "label_index_note": (
                "label = waypoint_targets(pose_last, future_poses_ext, "
                "horizons)[:, gp_slot]; validity = future_valid_ext[:, h-1]. "
                "Identical index to goal_point.goal_point_label_at_time "
                "(k = round(t/dt) - 1). TRAIN ONLY."),
            "provenance": gpm.goal_point_provenance(cfg.goal_point_cfg),
        } if getattr(cfg, "goal_point_inject", False) else None,
        # ⭐⭐ D-TACGOAL-1 / D-ROLL-1h (2026-09-06) — THE TACTICAL-GOAL
        # SET HEAD, STAMPED AS THREE SEPARATE FACTS.
        #
        # ⛔ `requested` is what ARGV ASKED FOR, `cfg` is what the pin
        # PUT ON THE CONFIG, and `built` is what the MODEL HAS. They are
        # deliberately NOT one re-derived key: a second copy of the build
        # condition (`_vv != "kin3" and cfg.tac_goal_tok_head`) is exactly
        # how `refcv3_arm`'s `a_star` comment drifted from the trainer and
        # cost a contaminated metric family. `param_breakdown_v3` already
        # reports its ledger line by reading the BUILT OBJECT for the same
        # reason; this stamp follows it rather than inventing a rival
        # convention.
        #
        # `built` is filled in from the constructed model at the call site
        # (the `agent_rig_camera` idiom), and `assert_seams_are_built`
        # REFUSES if the record and the weights disagree -- the D-ROLL-1
        # regression is precisely a 11,286-parameter disagreement between
        # a recorded ledger and a rebuilt model.
        "tac_goal_tok_head": {
            "requested": bool(getattr(args, "tac_goal_tok_head", False)),
            "cfg": bool(getattr(cfg, "tac_goal_tok_head", False)),
            "tac_vocab_version": str(getattr(cfg, "tac_vocab_version",
                                             "")),
            "built": None,      # ← the MODEL fills this; see `train`
        },
        # ⭐⭐ refcv6 §4 — THE BEHAVIOUR DECODER, STAMPED AS THE SAME THREE
        # SEPARATE FACTS, plus the one that would have caught this defect in
        # the run record: the WEIGHT. `tac_goal_tok_head` was `requested`,
        # `cfg` and `built` all true for 40,284 steps and still learned
        # nothing, because nothing in the record said whether a loss could
        # reach it. `w` is that missing fact.
        # ⛔ `sources` is stamped because the PI asked for "the agent and the
        # map": an arm that ran agent-only must say so in its own artifact,
        # or a later reader will credit it with a map it never had.
        "tac_decoder_v6": {
            "requested": bool(getattr(args, "tac_decoder_v6", False)),
            "cfg": bool(getattr(cfg, "tac_decoder_v6", False)),
            "w": float(getattr(args, "w_tac_v6", 0.0) or 0.0),
            "valid_threshold": float(
                getattr(cfg, "tac_decoder_valid_threshold", 0.5)),
            "graft_behaviour_sel": bool(
                getattr(cfg.core, "graft_behaviour_sel", False)),
            "decoder_cfg": (cfg.tac_decoder_cfg.to_dict()
                            if getattr(cfg, "tac_decoder_v6", False) else None),
            "sources": (list(getattr(cfg.tac_decoder_cfg, "sources", ()))
                        if getattr(cfg, "tac_decoder_v6", False) else None),
            # ⭐⭐ PI RULING 2026-09-17 R2 — THE FACT, NOT A CONSTANT. This
            # read `False` unconditionally while the seam was structurally
            # blocked; it is now the ARM's own answer, and an agent-only arm
            # still stamps `false` and must still be reported as agent-only.
            # ⛔ Derived from the width the decoder was BUILT with, never from
            # the flag alone: `--tac-decoder-d-bev` can be refused above, and a
            # stamp that believed argv would then out-claim the weights.
            "bev_tokens_reach_decoder": bool(
                getattr(cfg, "tac_decoder_v6", False)
                and int(getattr(cfg.tac_decoder_cfg, "d_bev", 0) or 0) > 0),
            # ⛔ PI RULING R3: the tactical loss MAY shape the shared trunk, and
            # attached is the ruling. A run that took the ablation says so here
            # rather than in a report nobody re-reads.
            "bev_grad_reaches_trunk": bool(
                getattr(cfg, "tac_decoder_v6", False)
                and int(getattr(cfg.tac_decoder_cfg, "d_bev", 0) or 0) > 0
                and not bool(getattr(cfg, "tac_decoder_bev_detach", False))),
            "bev_detached": bool(getattr(cfg, "tac_decoder_bev_detach", False)),
            "bev_unblocked_by": (
                "PI RULING 2026-09-17 (SPEC_REFCV6_V2.md, items 15/18 CLOSED): "
                "the BEV encoder moved INTO the model forward "
                "(refc_v3.RefCV3Model._bev_hook -> refc.py's `bev_hook` seam), "
                "so a BEV token exists where the scene hook fires. An arm with "
                "d_bev 0 is still AGENT-ONLY and must be reported as such."),
            "built": None,      # ← the MODEL fills this; see `train`
        },
        # ⭐⭐ refcv6 §5 — the 4-way one-hot set-speed. ⛔ `derivation` is the
        # module's own constant, never retyped here, and
        # `assert_speed_max_stamp_v6` refuses a run whose config.json would
        # not carry it — AND refuses the mirror (a control stamped as
        # conditioned).
        "max_speed_onehot_v6": {
            "requested": bool(getattr(args, "max_speed_input_v6", False)),
            "cfg": bool(getattr(cfg, "max_speed_onehot_v6", False)),
            "sidecar": (str(getattr(args, "speed_max_sidecar_v6", None) or "")
                        or None),
            "ladder_kmh": list(v6ms.SPEED_MAX_STEPS_KMH_V6),
            "provenance": "ego-future (oracle INPUT, ~1.7555 bits)",
            "built": None,      # ← the MODEL fills this; see `train`
        },
        # ⭐⭐ E16 — THE MAX-SPEED CEILING, STAMPED AS INTENT + FACT.
        # Same three-fact shape as `tac_goal_tok_head` above and for the
        # same reason: `requested` is what ARGV asked for, `cfg` is what the
        # pin put on the config, `built` is what the MODEL has, and
        # `assert_seams_are_built` refuses when the record and the weights
        # disagree. ⛔ `provenance` is written into the RUN RECORD rather
        # than left in a docstring, because a reader months later must be
        # able to answer "what was this channel actually trained on?" from
        # the artifact alone — and the answer is `ego-future`, a
        # TRAIN/DEPLOY MISMATCH that any result must be read against.
        "max_speed_input": {
            "requested": bool(getattr(args, "max_speed_input", False)),
            "cfg": bool(getattr(cfg, "max_speed_input", False)),
            "mode": str(getattr(getattr(cfg, "max_speed_cfg", None),
                                "mode", msi.DEFAULT_MODE)),
            "d_speed": int(getattr(getattr(cfg, "max_speed_cfg", None),
                                   "d_speed", 0)),
            "meta": (msi.artifact_meta(
                str(getattr(getattr(cfg, "max_speed_cfg", None), "mode",
                            msi.DEFAULT_MODE)))
                if getattr(cfg, "max_speed_input", False) else None),
            "required_controls": ["shuffled", "withheld"],
            "controls_note": (
                "⛔ EVAL OBLIGATION, inseparable from this edge and "
                "identical to E13's: a max-speed-conditioned result carries "
                "a SHUFFLE control (serve another clip's ceiling) and a "
                "WITHHOLD control (valid = 0), or 'the arm improved' cannot "
                "be separated from 'the arm gained a parameter'."),
            "built": None,      # ← the MODEL fills this; see `train`
        },
        # ⛔ M18: the camera the two monocular weights are computed against —
        # or the reason there is none. Without this a reader cannot tell a run
        # that trained `loss_project` from one that stamped its weight and
        # computed nothing, which is the defect this stamp exists to close.
        "agent_rig_camera": _build_rig_camera(cfg, args)[1],
        # ⭐⭐ EVERY `--agent-*` / `--w-*` KNOB, DERIVED FROM THE PARSER
        # ITSELF. See `agent_knob_dests`: a knob added to `build_parser`
        # tomorrow is stamped tomorrow, with no list here to rot.
        "agent_knobs": agent_knob_stamp(args),
    }


# ---------------------------------------------------------------------------
# ⭐ THE PROVENANCE CLOSURE — every knob reaches the record, BY CONSTRUCTION
# ---------------------------------------------------------------------------

def agent_knob_dests(parser: argparse.ArgumentParser | None = None
                     ) -> tuple[str, ...]:
    """Every ``--agent-*`` / ``--w-*`` / ``--bev-aux*`` option's ``dest``,
    **read off the parser**, sorted.

    ⛔ **DERIVED, NEVER LISTED.** MEASURED 2026-09-05 (mm-decisions M18,
    escalation #3): ``w_agent`` and ``w_u0`` were reported absent from
    ``config.json`` — a run that could not say what weight its detector
    trained at, which makes any later comparison between arms unfalsifiable.
    A hand-written list of knobs to stamp is the same defect deferred: it is
    correct the day it is written and wrong the day a knob is added. This
    function asks argparse, so the stamp cannot fall behind the CLI.

    ⭐ **WIDENED 2026-09-07 to ``--bev-aux*`` (WP-D), and the reason is the M18
    finding with a different knob in it.** Only ``--w-bev-aux`` starts with
    ``--w-``, so the WP-D family would otherwise have reached ``config.json``
    with its WEIGHT recorded and its POLICY absent — and
    ``--bev-aux-occlusion`` is precisely what separates the pre-registered arm
    from its DELIBERATE-REGRESSION twin (``none`` supervises occluded space as
    free, mislabelling a MEASURED 27.958 % of occupied cells). A record that
    cannot say which of the two ran makes the panel unfalsifiable. Same for
    ``--bev-aux-detach`` and ``--bev-aux-shuffle``, which are the other two
    control arms.
    """
    ap = parser if parser is not None else build_parser()
    return tuple(sorted({
        a.dest for a in ap._actions
        if a.dest and a.dest != argparse.SUPPRESS
        and any(o.startswith("--agent") or o.startswith("--w-")
                or o.startswith("--bev-aux") or o.startswith("--wp-index")
                for o in a.option_strings)}))


def agent_knob_stamp(args, parser: argparse.ArgumentParser | None = None
                     ) -> dict:
    """``{dest: value}`` for every knob :func:`agent_knob_dests` names.

    A dest the namespace does not carry is recorded as ``"<UNSET>"`` rather
    than dropped — an absent knob must be visible in the record, not silently
    equal to a default.
    """
    out = {}
    for d in agent_knob_dests(parser):
        v = getattr(args, d, "<UNSET>")
        out[d] = v if isinstance(v, (int, float, str, bool, type(None))) \
            else (list(v) if isinstance(v, (list, tuple)) else repr(v))
    return out


def assert_ground_prior_is_supervised(model, args) -> dict:
    """\u26d4\u26d4 MEASURE, at startup, that ``--agent-w-ground`` trains anything.

    **MEASURED 2026-09-05 (this stream): IT DOES NOT.**
    ``refc_agents.ground_range_prior`` projects a slot's foot at rig ``z = 0``
    and back-projects the resulting pixel onto the plane
    ``z = ROAD_PLANE_Z_M``, and ``ROAD_PLANE_Z_M`` **is 0.0** --
    ``RigCamera.project`` and ``RigCamera.ground_intersection`` are exact
    inverses, so ``r_back == r_pred`` **by construction**. Over 128 random
    boxes across the decode box the term reads **2.61e-08** with a parameter
    gradient of **8.73e-11**: a constant zero. An arm passing
    ``--agent-w-ground 0.5`` therefore stamps the weight into ``config.json``,
    adds ``w_ground * 0`` to the total, and would read afterwards as *"the
    ground prior does not help"*.

    \u2b50 **This is the SAME defect M18 closed, one level in.** M18 fixed
    *"the camera is None so the term never runs"*; the term now runs and is
    identically zero. A guard on the flag cannot see that, and neither can a
    guard on the camera -- only a guard on the **gradient** can.

    \u26d4 **The refusal is MEASURED, not hardcoded**, so it lifts itself the day
    the term becomes real (the decoder would need an image-plane output --
    ``AgentSlotDecoder`` emits no pixel today, which is why the term has
    nothing independent to constrain).

    \u26a0\ufe0f **The probe carries its own POSITIVE CONTROL.** A reading of
    "gradient ~ 0" is indistinguishable from a probe that never ran, so
    ``monocular_projection_loss`` is driven through the same tensors in the
    same breath and MUST read non-zero. If the control is also flat the result
    is **INCONCLUSIVE, which is a refusal** -- never a pass.
    """
    w_gr = float(getattr(args, "agent_w_ground", 0.0))
    cam = getattr(model, "_rig_camera", None)
    if w_gr <= 0.0 or cam is None:
        return {"checked": False, "reason": "w_ground == 0 or no camera"}
    if isinstance(cam, _refc_agents.RigCameraBank):
        cam = next(iter(cam.by_episode.values()), None)
        if cam is None:
            return {"checked": False, "reason": "empty bank"}
    g = torch.Generator().manual_seed(0)
    B, N = 4, 32
    cx = torch.rand(B, N, generator=g) * 50.0 + 5.0
    cy = (torch.rand(B, N, generator=g) * 2.0 - 1.0) * 14.0
    box = torch.stack([cx, cy, torch.full_like(cx, 4.5),
                       torch.full_like(cx, 1.9)], dim=-1).requires_grad_(True)
    gp = _refc_agents.ground_range_prior(box, cam)
    grad_ground = 0.0
    if gp["n"] and gp["loss"].requires_grad:
        grad_ground = float(torch.autograd.grad(gp["loss"], box,
                                                retain_graph=False)[0]
                            .abs().max())
    # -- the same-breath POSITIVE CONTROL: a term known to be alive ---------
    box2 = box.detach().clone().requires_grad_(True)
    tgt = box2.detach() + 1.0
    idx = torch.arange(N)
    match = {"rows": [idx] * B, "cols": [idx] * B,
             "n_dropped": [0] * B, "n_target": [N] * B}
    pj = _refc_agents.monocular_projection_loss(box2, tgt, cam, match)
    grad_ctrl = 0.0
    if pj["n"] and pj["loss"].requires_grad:
        grad_ctrl = float(torch.autograd.grad(pj["loss"], box2)[0].abs().max())
    out = {"checked": True, "loss_ground": float(gp["loss"].detach()),
           "n_ground": int(gp["n"]), "grad_absmax_ground": grad_ground,
           "control_grad_absmax_project": grad_ctrl,
           "control_n_project": int(pj["n"]),
           "floor": _GROUND_GRAD_FLOOR}
    print("[v3] ground-prior probe: loss %.3e grad %.3e (n %d) | CONTROL "
          "project grad %.3e (n %d)"
          % (out["loss_ground"], grad_ground, out["n_ground"], grad_ctrl,
             out["control_n_project"]), flush=True)
    if grad_ctrl <= _GROUND_GRAD_FLOOR or pj["n"] == 0:
        raise SystemExit(
            "[v3] \u26d4 INCONCLUSIVE (= a refusal): the ground-prior probe's "
            "own POSITIVE CONTROL read a flat gradient (%.3e over n=%d). A "
            "zero reading from the ground term cannot be distinguished from a "
            "probe that never ran, so the run stops rather than reporting a "
            "pass it did not earn." % (grad_ctrl, pj["n"]))
    if grad_ground <= _GROUND_GRAD_FLOOR:
        raise SystemExit(
            "[v3] \u26d4 REFUSING --agent-w-ground %g: `ground_range_prior` is a "
            "TAUTOLOGY on this geometry and trains NOTHING. MEASURED right "
            "now on this run's own camera: loss %.3e, parameter gradient "
            "%.3e over n=%d (floor %.0e), while the same-breath control term "
            "reads gradient %.3e. The term projects the box foot at rig z=0 "
            "and back-projects that pixel onto z=ROAD_PLANE_Z_M=0.0 -- "
            "`project` and `ground_intersection` are exact inverses, so "
            "r_back == r_pred BY CONSTRUCTION and there is no independent "
            "image-plane quantity for it to constrain (AgentSlotDecoder emits "
            "no pixel). Running anyway would stamp w_ground into config.json, "
            "add exactly 0 to the total, and read afterwards as 'the ground "
            "prior does not help' -- the M18 dead-flag defect one level in. "
            "Set --agent-w-ground 0, or give the head an image-plane output "
            "first (escalated: refcv5 needs a per-slot foot-row so the prior "
            "has something to constrain)."
            % (w_gr, out["loss_ground"], grad_ground, out["n_ground"],
               _GROUND_GRAD_FLOOR, grad_ctrl))
    return out


def assert_rig_camera_covers(model, ds, args) -> dict:
    """MEASURE a per-clip camera bank against the run's OWN episodes, and
    REFUSE a gap unless it was accepted by name. Returns the stamp block.

    \u26d4\u26d4 **WHY THIS IS A REFUSAL AND NOT A WARNING.** A bank keyed by
    ``stable_episode_id(clip_id)`` and built from a table that predates the
    corpus (or names a different split) can cover a fraction of the episodes
    and nothing downstream notices: the uncovered rows simply get no camera,
    both monocular terms skip them -- COUNTED, but only in the log row -- and
    ``config.json`` still reads ``mount_pose_scope: PER-CLIP``. A later
    comparison between a "projection arm" and a "BEV-only arm" would then be
    a comparison between two fractions nobody recorded. That is exactly the
    M18 defect (a stamped knob that computes nothing) with a camera in place
    of a weight, so it is measured here, before the first step, against the
    episode list the run actually loaded.

    \u26a0\ufe0f The measurement is over EPISODES, not windows: an episode with
    no camera contributes every one of its windows to the gap.
    """
    cam = getattr(model, "_rig_camera", None)
    if not isinstance(cam, _refc_agents.RigCameraBank):
        return {}
    eps = getattr(ds, "episodes", None) or []
    ids = []
    for e in eps:
        try:
            ids.append(int(e.episode_id))
        except (TypeError, ValueError):
            continue
    cov = cam.coverage(ids)
    cov["bank_n_clips"] = len(cam)
    cov["allow_partial"] = bool(
        getattr(args, "agent_rig_extrinsics_allow_partial", False))
    print("[v3] rig camera coverage: %d/%d episodes (%.4f) from a %d-clip "
          "bank" % (cov["n_covered"], cov["n"], cov["frac"],
                    cov["bank_n_clips"]), flush=True)
    if cov["n_missing"] and not cov["allow_partial"]:
        raise SystemExit(
            "[v3] \u26d4 REFUSING: the PER-CLIP extrinsics table covers "
            "%d/%d episodes of this run (%d missing, e.g. %s). The "
            "uncovered rows would get NO camera, both monocular terms "
            "would skip them, and config.json would still read "
            "mount_pose_scope PER-CLIP -- a run record stating a "
            "configuration that did not happen. Extend the table "
            "(taniteval/tools/pai_extrinsics_table.py builds it from the "
            "dataset's own calibration/sensor_extrinsics), or pass "
            "--agent-rig-extrinsics-allow-partial to accept the gap BY "
            "NAME (it is stamped into config.json)."
            % (cov["n_covered"], cov["n"], cov["n_missing"],
               cov["missing_sample"][:4]))
    return {"coverage": cov}


def assert_seams_are_built(model, stamp: dict) -> None:
    """REFUSE to write a run record that claims a seam THE WEIGHTS DO NOT HAVE.

    ⛔⛔ **THE DEFECT THIS CLOSES, MEASURED 2026-09-06.** ``DecoderConfig`` had
    no ``sampler`` field, so ``_pin_refcv5_seams``'s ``core.decoder.sampler =
    "ddim"`` created an AD-HOC ATTRIBUTE on an unfrozen dataclass. Python
    accepted it; the two guards in ``_pin_refcv5_seams`` read it back and
    PASSED; ``_seam_stamp`` copied it into ``config.json``. The model contained
    no denoiser at all -- ``refc.py`` had zero occurrences of ``control_head``.
    A run would have trained, converged, written a checkpoint and STATED IN ITS
    OWN RECORD that it used a sampler it did not have. That is not a missing
    feature, it is FALSE PROVENANCE: every later comparison against that arm
    would have been unfalsifiable.

    ⭐ **Why the field alone is not the fix.** Declaring ``sampler`` on the
    dataclass removes THIS instance and none of the class. The stamp is built
    from the CONFIG, and a config is a statement of intent; only the MODEL is a
    statement of fact. So the record is checked against the built modules, here,
    immediately before ``config.json`` is written -- the same discipline as
    :func:`assert_knobs_stamped` one level down, and it runs at startup rather
    than becoming an unanswerable question in an audit months later.

    ⚠ **The check is BIDIRECTIONAL.** A seam that is BUILT but NOT STAMPED is
    equally unfalsifiable -- it is the `SEAM_STATE.md` failure, where six live
    seams were absent from ``config.json`` and the arm was reconstructible only
    by knowing what ``refc_v3.py`` forces.
    """
    core = getattr(model, "core", model)
    dec = core.decoder
    bad: list[str] = []

    def _mod(obj, name):
        return getattr(obj, name, None)

    # --- WP-4: the sampler ------------------------------------------------ #
    if str(stamp.get("sampler", "none")) != "none":
        for name in ("control_head", "time_mlp", "sched"):
            if _mod(dec, name) is None:
                bad.append(
                    f"stamp says sampler={stamp['sampler']!r} but "
                    f"decoder.{name} is None -- the record would claim a "
                    f"denoiser the weights do not contain")
    elif _mod(dec, "control_head") is not None:
        bad.append(
            "stamp says sampler='none' but decoder.control_head WAS BUILT -- "
            "a live sampler absent from the run record")
    if float(stamp.get("w_u0", 0.0)) > 0.0 and _mod(dec, "control_head") is None:
        bad.append(
            f"stamp says w_u0={stamp['w_u0']} but there is no control_head "
            f"for that loss to supervise: the term would be silently skipped "
            f"while the weight is stamped")

    # --- refcv6 §2/§3: the trunk and the F-flags, BOTH DIRECTIONS ---------- #
    # ⛔ A stamped `trunk: timm` on a model whose encoder is the in-repo
    # ResNet would claim an ImageNet prior the weights do not have — the same
    # false provenance as a stamped sampler with no denoiser. And an F3/F4 arm
    # whose heads were not built would train the BASELINE under refcv6's name.
    _enc = _mod(core, "encoder")
    _is_timm = type(_enc).__name__ == "TimmResNetTrunk"
    if str(stamp.get("trunk", "refc")) == "timm" and not _is_timm:
        bad.append(
            "stamp says trunk='timm' but core.encoder is a "
            f"{type(_enc).__name__} -- the record would claim an ImageNet "
            "prior the weights do not have")
    if str(stamp.get("trunk", "refc")) != "timm" and _is_timm:
        bad.append(
            "stamp says trunk='refc' but core.encoder IS the timm ImageNet "
            "trunk -- a live prior absent from the run record")
    _eh = stamp.get("ego_history")
    _eh_built = _mod(core, "ego_hist") is not None
    if _eh is not None and bool(_eh.get("enable")) and not _eh_built:
        bad.append("stamp carries an `ego_history` block but core.ego_hist is "
                   "None -- the record would claim an input the model never "
                   "reads")
    if (_eh is None or not bool(_eh.get("enable"))) and _eh_built:
        bad.append("core.ego_hist IS built but the stamp says ego_history is "
                   "off -- a live input absent from the run record")
    _r6 = stamp.get("refcv6")
    _built = _mod(dec, "rv6")
    if _r6 is not None:
        if _mod(dec, "cascade") is None and bool(_r6.get("f3_per_layer")):
            bad.append("stamp says F3 but decoder.cascade is None -- the "
                       "per-layer heads were never built")
        if _mod(dec, "adaln") is None and bool(_r6.get("f4_adaln")):
            bad.append("stamp says F4 but decoder.adaln is None -- the "
                       "AdaLN modulation was never built")
        if _built is not None and _rv6.flag_stamp(_built) != _r6:
            bad.append("the stamped refcv6 block differs from the one the "
                       "decoder was BUILT with -- the record would name a "
                       "different arm than the one that trains")
    elif _built is not None and _built.any_on:
        bad.append("decoder carries live refcv6 flags but the stamp says "
                   "refcv6=null -- a live arm absent from the run record")

    # --- WP-6: the agent seam --------------------------------------------- #
    layers = list(getattr(dec, "layers", []))
    if bool(stamp.get("cross_agent", False)):
        dead = [i for i, ly in enumerate(layers)
                if _mod(ly, "cross_agent") is None]
        if dead:
            bad.append(
                f"stamp says cross_agent=True but decoder layers {dead} have "
                f"no agent attention -- the tokens would reach nothing")
    else:
        live = [i for i, ly in enumerate(layers)
                if _mod(ly, "cross_agent") is not None]
        if live:
            bad.append(
                f"stamp says cross_agent=False but decoder layers {live} DO "
                f"cross-attend agents -- a live seam absent from the record")
    # --- WP-B: the waypoint index ----------------------------------------- #
    # ⛔ BIDIRECTIONAL, exactly like the WP-4 and WP-6 checks above. A stamped
    # index with no bias heads is FALSE PROVENANCE; attached heads absent from
    # the record are the `SEAM_STATE.md` failure. And a stamped MODE that the
    # attached config does not carry would let a CONTROL arm be reported as the
    # treatment, which is worse than either.
    _wps = stamp.get("wp_index")
    _wp_live = [i for i, ly in enumerate(layers)
                if _mod(ly, "wp_index") is not None]
    if _wps is not None and bool(_wps.get("enable", False)):
        _dead = [i for i, _ in enumerate(layers) if i not in set(_wp_live)]
        if _dead:
            bad.append(
                f"stamp says wp_index.enable=True but decoder layers {_dead} "
                f"have no bias head -- the record would claim a waypoint "
                f"index the weights do not contain")
        _cfg_mode = str(getattr(_mod(dec, "wp_index_cfg"), "mode", "<none>"))
        if _wp_live and _cfg_mode != str(_wps.get("mode")):
            bad.append(
                f"stamp says wp_index.mode={_wps.get('mode')!r} but the "
                f"attached config carries {_cfg_mode!r} -- a CONTROL arm "
                f"would be reported as the treatment, or the reverse")
    elif _wp_live:
        bad.append(
            f"stamp carries no live wp_index but decoder layers {_wp_live} DO "
            f"index the agent attention by waypoint -- a live seam absent "
            f"from the run record")
    if stamp.get("agents") is not None:
        for name in ("agent_head", "agent_embed"):
            if _mod(core, name) is None:
                bad.append(
                    f"stamp carries an `agents` block but core.{name} is "
                    f"None -- the record would claim a detector that was "
                    f"never built")
    else:
        for name in ("agent_head", "agent_embed"):
            if _mod(core, name) is not None:
                bad.append(
                    f"stamp says agents=None but core.{name} WAS BUILT -- a "
                    f"live agent seam absent from the run record")

    # --- E15 (GP-2): the goal-point edge and its ranking seam ------------- #
    # ⛔ BIDIRECTIONAL, exactly like the two above. A stamped goal point with
    # no head is FALSE PROVENANCE; a built head absent from the record is the
    # `SEAM_STATE.md` failure. And a stamped `graft_gp_point` with no gate on
    # the decoder would be a run claiming a selection seam that ranks nothing.
    gpst = stamp.get("goal_point")
    head_built = _mod(model, "gp_head") is not None
    gate_built = _mod(dec, "gp_point_gate") is not None
    if gpst is not None and bool(gpst.get("goal_point_inject", False)):
        if not head_built:
            bad.append(
                "stamp says goal_point_inject=True but model.gp_head is None "
                "-- the record would claim an E15 head the weights do not "
                "contain")
        if bool(gpst.get("graft_gp_point", False)) and not gate_built:
            bad.append(
                "stamp says graft_gp_point=True but decoder.gp_point_gate is "
                "None -- the record would claim a ranking seam that ranks "
                "nothing")
        if not bool(gpst.get("graft_gp_point", False)) and gate_built:
            bad.append(
                "stamp says graft_gp_point=False but decoder.gp_point_gate "
                "WAS BUILT -- a live selection seam absent from the record")
        if float(gpst.get("w_goal_point", 0.0)) <= 0.0:
            bad.append(
                "stamp says w_goal_point=%s with an E15 head built: the head "
                "would be supervised by nothing and the PREREG section 5 head "
                "gate would fail for a reason that is not about the goal form"
                % (gpst.get("w_goal_point"),))
        if (int(gpst.get("gp_slot", -1)) < 0
                and bool(gpst.get("graft_gp_point", False))):
            bad.append(
                "stamp carries graft_gp_point=True with gp_slot < 0 -- the "
                "goal and the anchor would be compared at different times")
    else:
        if head_built:
            bad.append(
                "stamp carries no goal_point block but model.gp_head WAS "
                "BUILT -- a live E15 edge absent from the run record")
        if gate_built:
            bad.append(
                "stamp carries no goal_point block but decoder.gp_point_gate "
                "WAS BUILT -- a live S7 selection seam absent from the record")

    # --- D-TACGOAL-1 / D-ROLL-1h: the tactical-goal SET head ------------- #
    # ⛔ BIDIRECTIONAL, like the three above, and it is the check whose
    # ABSENCE was the D-ROLL-1 regression: 11,286 params in the model that
    # no recorded `param_breakdown` names makes the checkpoint unloadable,
    # and 11,286 params in the record that the weights lack is the mirror
    # image. The stamp's `built` is read off the module; `cfg` is intent.
    tg = stamp.get("tac_goal_tok_head")
    tg_built = _mod(model, "tac_goal_tok_head") is not None
    if isinstance(tg, dict):
        if tg.get("built") is not None and bool(tg["built"]) != tg_built:
            bad.append(
                f"stamp says tac_goal_tok_head.built={tg['built']!r} but "
                f"model.tac_goal_tok_head is "
                f"{'BUILT' if tg_built else 'None'} -- the record and the "
                f"weights disagree about 11,286 parameters, which is the "
                f"D-ROLL-1 rollability defect exactly")
        if bool(tg.get("cfg", False)) and not tg_built:
            bad.append(
                "stamp says tac_goal_tok_head was pinned onto the config "
                "but model.tac_goal_tok_head is None -- the record would "
                "claim a supervised head the weights do not contain")
        if not bool(tg.get("cfg", False)) and tg_built:
            bad.append(
                "model.tac_goal_tok_head WAS BUILT but the run record does "
                "not ask for it -- 11,286 parameters absent from the "
                "record, which is what makes a checkpoint unrollable")
    elif tg_built:
        bad.append(
            "the seam stamp carries no `tac_goal_tok_head` block but the "
            "head WAS BUILT -- a live seam absent from the run record")

    # --- E16: the max-speed conditioner ---------------------------------- #
    # ⛔ BIDIRECTIONAL, exactly like the block above. A conditioner in the
    # weights that no recorded `param_breakdown` names is the D-ROLL-1
    # rollability defect; a conditioner in the record that the weights lack
    # is its mirror image, and both make the checkpoint unloadable through
    # `refcv3_arm.cross_check_config`.
    ms = stamp.get("max_speed_input")
    ms_built = _mod(model, "max_speed_cond") is not None
    if isinstance(ms, dict):
        if ms.get("built") is not None and bool(ms["built"]) != ms_built:
            bad.append(
                f"stamp says max_speed_input.built={ms['built']!r} but "
                f"model.max_speed_cond is "
                f"{'BUILT' if ms_built else 'None'} -- the record and the "
                f"weights disagree about the E16 conditioner")
        if bool(ms.get("cfg", False)) and not ms_built:
            bad.append(
                "stamp says max_speed_input was pinned onto the config but "
                "model.max_speed_cond is None -- the record would claim a "
                "ceiling channel the weights do not contain")
        if not bool(ms.get("cfg", False)) and ms_built:
            bad.append(
                "model.max_speed_cond WAS BUILT but the run record does not "
                "ask for it -- parameters absent from the record, which is "
                "what makes a checkpoint unrollable")
    elif ms_built:
        bad.append(
            "the seam stamp carries no `max_speed_input` block but the "
            "conditioner WAS BUILT -- a live seam absent from the record")

    if bad:
        raise SystemExit(
            "[v3] ⛔⛔ THE RUN RECORD DOES NOT MATCH THE MODEL. Refusing to "
            "write config.json rather than stamping a configuration that did "
            "not happen:\n  - " + "\n  - ".join(bad))


def assert_knobs_stamped(args, stamp: dict,
                         parser: argparse.ArgumentParser | None = None) -> None:
    """REFUSE to start a run whose own record cannot state its knobs.

    Checked against the VALUE, not the key name: every knob's value must be
    recoverable from ``stamp``. Called from :func:`train` immediately before
    ``config.json`` is written, so the failure is a refusal at startup rather
    than an unanswerable question months later.
    """
    want = agent_knob_stamp(args, parser)
    got = (stamp or {}).get("agent_knobs")
    if not isinstance(got, dict):
        raise SystemExit(
            "[v3] ⛔ the seam stamp carries no `agent_knobs` block, so this "
            "run's config.json could not state the weights it trained at "
            "(mm-decisions M18). This is a code defect, not an argv one.")
    missing = {k: v for k, v in want.items() if k not in got or got[k] != v}
    if missing:
        raise SystemExit(
            f"[v3] ⛔ {len(missing)} --agent-*/--w-*/--bev-aux* knob(s) do NOT "
            f"reach "
            f"config.json: {sorted(missing)}. A run that cannot state its own "
            "weights makes every later comparison between arms unfalsifiable "
            "(mm-decisions M18).")


def _clip_table_for_caches(cache_dirs) -> tuple[dict, int]:
    """``({stable_episode_id: clip_id}, n_stack)`` from the v2 MANIFESTS.

    ⛔ **The clip id cannot be recovered from the episode id.**
    ``stable_episode_id`` is a blake2b digest, so this table is the ONLY route
    from what ``LazyV2Episode`` carries to what :class:`MapGTStore` and
    :class:`AgentJoin3D` resolve by. Reading it from the manifest (rather than
    from the agent join's ``_clip_of_uid``) keeps the map path independent of
    whether a join was loaded at all.

    ⛔ ``n_stack`` is READ, never assumed. It converts a stacked-row index into
    the RAW v2ep frame the SAM3 labels and the 3-D join are indexed by, and a
    literal 3 against a cache built with another value shifts every label by
    the difference -- 1.7 m at 30 km/h, and visible in no metric. Caches with
    DISAGREEING ``n_stack`` are refused rather than silently reduced to one.
    """
    from tanitad.data.v2_dataset import load_or_build_manifest
    dirs = ([cache_dirs] if isinstance(cache_dirs, (str, Path))
            else list(cache_dirs))
    table: dict[int, str] = {}
    n_stacks: set[int] = set()
    for cd in dirs:
        man = load_or_build_manifest(cd, verbose=False)
        cids = list(man.get("clip_id") or [])
        uids = list(man.get("episode_uid") or [])
        if len(uids) != len(cids):
            raise SystemExit(
                f"[v3] ⛔ {cd}: the v2 manifest has {len(cids)} clip_ids and "
                f"{len(uids)} episode_uids. Refusing rather than zipping two "
                f"lists of different length into a label join.")
        for u, c in zip(uids, cids):
            table[int(u)] = str(c)
        n_stacks.update(int(x) for x in (man.get("n_stack") or []))
    if len(n_stacks) > 1:
        raise SystemExit(
            f"[v3] ⛔ the caches disagree on n_stack {sorted(n_stacks)}. One "
            f"stacked-row -> raw-frame conversion cannot serve both, and the "
            f"wrong one shifts every map label by the difference.")
    return table, (n_stacks.pop() if n_stacks else 0)


def _verify_agent_join(args) -> dict | None:
    """Check ``--agent-join`` against its sidecar's **declared** digest scope.

    ⛔⛔ **WHY (mm-decisions M18, MEASURED 2026-09-05).** The two joins'
    sidecars record ``summary.md5`` over DIFFERENT artifacts — train2400's
    over the compressed ``.xz`` (re-verified here:
    ``24cbdca8c3b23aafc2fb17e6bf99cf76``), val40's over the decompressed
    ``.jsonl`` — and neither says so. A checker inherited from one REFUSES the
    other's perfectly good file, and the same ambiguity in the other direction
    would **accept the wrong file and report success**.

    ⛔ **A sidecar that declares no scope REFUSES, it does not guess.** The
    guess everyone reaches for — read the extension off ``summary.out`` — is
    MEASURED wrong: val40's ``.xz``-side sidecar names the ``.jsonl`` there.
    :func:`tanitad.data.join_meta.backfill` is the migration: it MEASURES
    which artifact the recorded digest covers by hashing both.

    ⚠️ A join with NO sidecar at all warns rather than refuses — an absent
    sidecar is a missing check, not a lying one — and the run records which
    of the two it was, so no reader has to assume.
    """
    path = getattr(args, "agent_join", None)
    if not path:
        return None
    mode = str(getattr(args, "agent_join_verify", "auto"))
    if mode == "off":
        print("[v3] ⚠️ --agent-join-verify off: the join's digest is NOT "
              "checked; config.json records that the operator turned it off.",
              flush=True)
        return {"verified": False, "mode": "off",
                "reason": "operator passed --agent-join-verify off"}
    from tanitad.data import join_meta as _jm
    side = _jm.sidecar_path(path)
    if side is None:
        print(f"[v3] ⚠️ no sidecar beside {path} (probed <file>.meta.json, "
              "the .xz-stripped form and <stem>.meta.json): the join's "
              "digest is UNVERIFIED and config.json says so.", flush=True)
        return {"verified": False, "mode": mode, "sidecar": None,
                "reason": "no sidecar found beside the join"}
    try:
        with open(side, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as e:                     # a broken sidecar is not a pass
        raise SystemExit(f"[v3] ⛔ {side} could not be read as JSON ({e}). An "
                         "unreadable sidecar is not an absent one.")
    try:
        ev = _jm.verify(path, meta, where=str(side))
    except _jm.JoinDigestScopeMissing as e:
        raise SystemExit(
            f"[v3] ⛔ {e}"
            "  ||  MIGRATION (one command): python -m "
            f"tanitad.data.join_meta {path!r} --write  ||  or pass "
            "--agent-join-verify off to record that this run checked "
            "nothing.")
    except _jm.JoinDigestMismatch as e:
        raise SystemExit(f"[v3] ⛔ agent join integrity: {e}")
    print(f"[v3] agent join verified: {ev['algo']}({ev['scope']} of "
          f"{ev['filename']}) = {ev['digest']}", flush=True)
    ev.update({"mode": mode, "sidecar": str(side)})
    return ev


def _refcv6_flags_of(model) -> "_rv6.DiffusionFlags":
    """The F1..F9 block the MODEL was built with. -> ``DiffusionFlags``.

    ⛔ Read off the built decoder, never off ``args``. A flag that lives only
    on the Namespace is the refcv5 false-provenance defect: the loss would take
    the F3/F5 branch for a model whose decoder built no cascade heads, and the
    run would stamp an arm it did not train.
    """
    dec = getattr(getattr(model, "core", model), "decoder", None)
    flags = getattr(dec, "rv6", None)
    return flags if isinstance(flags, _rv6.DiffusionFlags) else _rv6.DiffusionFlags()


def refcv6_flags_from_args(args) -> "_rv6.DiffusionFlags | None":
    """``--f1-…`` .. ``--f9-…`` -> a :class:`DiffusionFlags`, or ``None``.

    ⭐ ``None`` when EVERY flag is at its default, so ``DecoderConfig.refcv6``
    stays ``None`` on a baseline run and the decoder constructs nothing. That
    is what makes "all nine off == bit-identical" true of the BUILD and not
    only of the forward.
    """
    flags = _rv6.DiffusionFlags(
        f1_random_t=bool(getattr(args, "f1_random_t", False)),
        f1_t_max=int(getattr(args, "f1_t_max", 50)),
        f2_dd_step=bool(getattr(args, "f2_dd_step", False)),
        f3_per_layer=bool(getattr(args, "f3_per_layer", False)),
        f4_adaln=bool(getattr(args, "f4_adaln", False)),
        f4_zero_init=bool(getattr(args, "f4_zero_init", False)),
        f5_emitting_conf=bool(getattr(args, "f5_emitting_conf", False)),
        f5_focal=bool(getattr(args, "f5_focal", False)),
        f6_w_u0_zero=bool(getattr(args, "f6_w_u0_zero", False)),
        f7_samples_per_anchor=int(getattr(args, "f7_samples_per_anchor", 1)),
        f7_ack_eval_join=bool(getattr(args, "f7_ack_eval_join", False)),
        f8_flat_waypoint_noise=bool(getattr(args, "f8_flat_noise", False)),
        f9_assert_vocab=bool(getattr(args, "f9_assert_vocab", False)))
    return flags if flags.any_on else None


def build_optimizer(model, args):
    """refcv6 §2 — the optimiser, behind ``--opt``. -> ``torch.optim.Optimizer``.

    ``--opt adam`` (DEFAULT) returns **exactly** ``torch.optim.Adam(
    model.parameters(), lr=args.lr)``: one parameter group, one learning rate,
    no weight decay. That is the line this function replaced, and
    ``tests/test_refcv6_trunk.py::test_default_optimiser_is_bit_identical``
    pins it against a freshly-constructed ``Adam`` group-for-group so the
    refcv6 flags cannot change a banked arm by accident.

    ``--opt dd`` returns DiffusionDrive's optimiser, from the released config
    (``…/ddv2_src/diffusiondrivev2_rl_config.py:119-131``, paper §4.2):

    * ``AdamW`` (``optimizer_type = "AdamW"``, line 122);
    * ``weight_decay = 1e-4`` (line 120) — NOT torch's 1e-2 AdamW default;
    * ``opt_paramwise_cfg`` (lines 125-131) puts the **image encoder** on
      ``lr_mult = cfg_lr_mult = 0.5`` (line 124), i.e. the encoder group runs
      at HALF the head learning rate.

    ⛔ Warm-up + cosine is unchanged and lives in the ``sched`` lambda at the
    call site. It is a MULTIPLIER on each group's own ``lr``, so the 0.5x
    survives the schedule instead of being overwritten by it — which is the
    silent way a paramwise config stops meaning anything.
    """
    kind = str(getattr(args, "opt", "adam"))
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    if kind == "dd":
        from tanitad.models.timm_trunk import param_groups_dd
        # ⛔ The prefix is READ OFF THE MODEL, never assumed. `refc_v3`'s
        # wrapper nests the trunk at `core.encoder`; a bare `RefCModel` has it
        # at `encoder`. Guessing wrong does not crash — it puts EVERY tensor in
        # the head group and the 0.5x applies to nothing, which is the DD
        # recipe in name only. `param_groups_dd` raises on an empty encoder
        # group, and this picks the prefix that exists.
        _enc_attr = ("core.encoder" if hasattr(getattr(model, "core", None),
                                               "encoder") else "encoder")
        groups = param_groups_dd(
            model, float(args.lr), encoder_attr=_enc_attr,
            encoder_lr_mult=float(getattr(args, "encoder_lr_mult", 0.5)),
            weight_decay=float(getattr(args, "weight_decay", 1e-4)))
        opt = torch.optim.AdamW(groups, lr=float(args.lr))
        print(f"[v3] opt=dd AdamW wd={getattr(args, 'weight_decay', 1e-4)} "
              f"encoder_lr={groups[0]['lr']:.3e} ({len(groups[0]['params'])} "
              f"tensors) head_lr={groups[1]['lr']:.3e} "
              f"({len(groups[1]['params'])} tensors)", flush=True)
        return opt
    raise SystemExit(f"[v3] ⛔ --opt {kind!r} not in ('adam', 'dd')")


def _apply_withheld_bank(model, args, eps, device) -> dict:
    """Install ``--withheld-bank`` on the decoder and return its config stamp.

    ``pred``/``random``/``none`` need a v0-CONDITIONED vocabulary (there is
    nothing to roll otherwise) and are refused without one; ``pred`` also
    needs the hierarchy (the goal head is the speed source). ``random`` draws
    from the TRAINING marginal of v0 — every frame's speed over the training
    episodes, sub-sampled to <= 200k values — which is stamped by n / mean /
    std so the control's distribution is on record. The mode itself goes live
    in the train loop (warm-up), never here.
    """
    mode = str(getattr(args, "withheld_bank", "fixed"))
    dec = model.core.decoder
    stamp = {"mode": mode,
             "warmup_steps": int(getattr(args, "withheld_bank_warmup", 0)),
             "speed_max_ms": float(getattr(args, "withheld_speed_max", 35.0)),
             "random_pool": None}
    if mode != "fixed" and not getattr(dec, "anchor_v0_cond", False):
        raise SystemExit(
            f"[v3] ⛔ --withheld-bank {mode} needs a v0-CONDITIONED vocabulary "
            f"(--anchor-v0-conditioned with a controls-carrying --anchors); a "
            f"fixed-path vocabulary has no per-row roll to redirect.")
    if mode == "pred" and not model.cfg.hier:
        raise SystemExit("[v3] ⛔ --withheld-bank pred needs --arm hier: the "
                         "goal head (g_tac) is the speed source.")
    dec.anchor_withheld_speed_max = float(stamp["speed_max_ms"])
    if mode == "random":
        vals = []
        for ep in eps:
            po = getattr(ep, "poses", None)
            if po is None:
                continue
            vals.append(torch.as_tensor(po)[:, 3].to(torch.float32)
                        .reshape(-1).clone())
        pool = torch.cat(vals) if vals else torch.zeros(0)
        pool = pool[torch.isfinite(pool)].clamp_min(0.0)
        if pool.numel() == 0:
            raise SystemExit("[v3] ⛔ --withheld-bank random: no v0 values in "
                             "the training episodes to draw from")
        if pool.numel() > 200_000:
            g = torch.Generator().manual_seed(int(args.seed))
            pool = pool[torch.randperm(pool.numel(), generator=g)[:200_000]]
        dec.anchor_random_speed_pool = pool.to(device)
        stamp["random_pool"] = {
            "n": int(pool.numel()), "mean_ms": round(float(pool.mean()), 4),
            "std_ms": round(float(pool.std()), 4),
            "p10_p50_p90_ms": [round(float(q), 3) for q in
                               torch.quantile(pool, torch.tensor(
                                   [0.1, 0.5, 0.9]))],
            "source": "poses[:, 3] over every frame of the TRAINING episodes "
                      "(the marginal the withheld row's bank is drawn from; "
                      "independent of the row)"}
    # the decoder starts on the FIXED roll; the train loop flips it at
    # `warmup_steps` and logs `withheld_bank_active`.
    dec.anchor_withheld_bank = "fixed" if stamp["warmup_steps"] > 0 else mode
    print(f"[v3] withheld bank: mode={mode} warmup={stamp['warmup_steps']} "
          f"speed_max={stamp['speed_max_ms']} m/s "
          f"pool={stamp['random_pool']}", flush=True)
    return stamp


def _anchor_stamp(path, anchors, controls=None, units="kappa",
                  art=None) -> dict:
    """Record the anchor vocabulary BY CONTENT, not by the presence of a path.

    A run that was launched without ``--anchors`` silently carries
    ``refc.default_anchors`` (the SYNTHETIC bootstrap set), and that is
    indistinguishable from a data-driven run in every artifact refcv3 left
    behind.  The sha256 of the tensor actually installed in the decoder is what
    settles it, so `source` is derived from the tensor, never from the flag.
    """
    import hashlib
    a = anchors.detach().to("cpu", torch.float32).contiguous()
    h = hashlib.sha256(a.numpy().tobytes()).hexdigest()
    stamp = {"path": str(path) if path else None,
             "shape": list(a.shape),
             "sha256_installed": h,
             "source": "file" if path else "refc.default_anchors (SYNTHETIC)"}
    if controls is not None:
        c = controls.detach().to("cpu", torch.float32).contiguous()
        stamp["controls_shape"] = list(c.shape)
        stamp["controls_sha256_installed"] = hashlib.sha256(
            c.numpy().tobytes()).hexdigest()
        stamp["v0_conditioned"] = True
        stamp["control_units"] = units
        stamp["straight_ahead_control_present"] = bool(
            ((c[:, 0] == 0) & (c[:, 1] == 0)).any())
    else:
        stamp["v0_conditioned"] = False
    # ⭐ where the units CAME FROM, and what the file itself declares. For the
    # live refcv4b file this reads `cli-override-legacy-file` with every
    # declared field None -- the record says the operator supplied the units.
    if art is not None:
        stamp["control_units_source"] = art.control_units_source
        stamp["artifact_schema"] = art.meta.get("schema")
        stamp["artifact_declared"] = dict(art.declared)
        stamp["artifact_provenance"] = art.meta.get("provenance")
    if path:
        try:
            stamp["file_sha256"] = hashlib.sha256(
                open(path, "rb").read()).hexdigest()
        except OSError as e:
            stamp["file_sha256"] = f"unreadable: {e}"
    return stamp


def preflight(args) -> int:
    _check_nav_from_v7_args(args)          # no-op unless --nav-from-v7
    _check_max_speed_args(args)            # no-op unless --max-speed-input
    _check_goal_point_args(args)           # no-op unless --goal-point-*
    check_effective_weights(args)          # a weight whose gate is shut
    art = _read_anchor_artifact(args)      # None without --anchors
    print("[v3-preflight] building both arms + pinning the delta …")
    cfg_h = v3.refc_v3_sized_config(args.size, hier=True)
    cfg_f = v3.refc_v3_sized_config(args.size, hier=False)
    if args.smoke:
        cfg_h, cfg_f = (v3.refc_v3_smoke_config(True),
                        v3.refc_v3_smoke_config(False))
    cfg_h, cfg_f = _pin_trainer_cfg(cfg_h, args), _pin_trainer_cfg(cfg_f, args)
    delta = v3.config_delta(cfg_h, cfg_f)
    if set(delta) != REGISTERED_DELTA_KEYS:
        print(f"[v3-preflight] ⛔ FAIL: config delta {sorted(delta)} != "
              f"registered {sorted(REGISTERED_DELTA_KEYS)} — amend "
              f"PREREG_REFC_V3.md BEFORE launch.")
        return 2
    cfg = cfg_h if args.arm == "hier" else cfg_f
    _check_anchor_artifact_against_cfg(art, cfg, args)
    model = v3.RefCV3Model(cfg)
    # E15 (GP-2): the preflight's synthetic loss step must exercise the
    # SAME terms the run will. Without this the goal-point loss reads
    # `getattr(model, '_w_goal_point', 0.0)` -> 0 and the preflight would
    # pass a term it never ran — a green light for an untested path.
    model._w_goal_point = float(getattr(args, "goal_point_w",
                                        GOAL_POINT_WEIGHT_DEFAULT))
    bd = v3.param_breakdown_v3(model)
    print(f"[v3-preflight] arm={args.arm} params={bd}")
    # ⭐ A GATE ROW CARRIES ITS ARM. `arm=hier` is no longer sufficient to
    # name the arm once the bypass exists, so the preflight prints the bypass
    # state on its own line, in the same breath as the params.
    _bp = bool(getattr(cfg.core, "no_strategic", False))
    print(f"[v3-preflight] strategic_layer="
          f"{'BYPASSED (--no-strategic)' if _bp else 'ACTIVE'} "
          f"| ctx->decoder={'OFF' if _bp else 'ON'} "
          f"| g_str->tactical_FiLM={'OFF' if _bp else 'ON'} "
          f"| route_readout->selection="
          f"{'OFF' if (_bp or not cfg.core.graft_route) else 'ON'} "
          f"| goal_str_loss="
          f"{'NOT APPLIED' if _bp else ('APPLIED' if getattr(args, 'goal_str', False) else 'n/a')} "
          f"| nav->operative=ON(measurement) "
          f"| nav->tactical={'ON(E13)' if cfg.nav_inject else 'OFF'}")
    torch.manual_seed(0)
    h, wpx = cfg.core.encoder.image_hw()
    frames = torch.rand(2, cfg.core.window, cfg.core.encoder.in_channels,
                        h, wpx)
    if cfg.hier:
        rep = v3.freeze_history_report(model, frames,
                                       v0=torch.tensor([3.0, 7.0]))
        print(f"[v3-preflight] freeze-history gate: {rep}")
        if not rep["pass"]:
            print("[v3-preflight] ⛔ FAIL: the C115 gate — the H arm is "
                  "flat-in-disguise; the experiment would be VOID (OUTCOME V).")
            return 3
        model.eval()
        if getattr(cfg, "ego_state_inject", False):
            # ================= REF-C v4 (E11') =================
            # ⭐⭐ THE REFUSED EDGE MOVED, SO THE PREFLIGHT MOVES WITH IT.
            # Under v3 this block pinned `v0 -> goal` as a REFUSED edge. E11'
            # admits it, so the v4 checks are: ego MUST reach the goal path
            # (else the arm is v3 wearing a v4 config, and a null result would
            # be about the wiring), frames MUST still reach the vision-borne
            # nodes, and the E14 base MUST be exactly the extrapolation.
            e_lo = v3.ego_state_at_t0(
                torch.zeros(2, cfg.core.window, 4),
                torch.zeros(2, cfg.core.window, 2))
            e_hi = torch.tensor([[9.0, 2.5, 0.45, 0.05, 1.0],
                                 [4.0, -1.5, -0.20, -0.05, 1.0]])
            with torch.no_grad():
                a = model(frames, v0=torch.tensor([0.0, 0.0]), ego_state=e_lo)
                bq = model(frames, v0=torch.tensor([9.0, 4.0]), ego_state=e_hi)
                c = model(torch.rand_like(frames),
                          v0=torch.tensor([0.0, 0.0]), ego_state=e_lo)
            # (1) ego REACHES the goal path. ⚠️ `z_tac`/`g_str` are
            # zero-init on the ego edge, so at step 0 only the E14 base can
            # carry it - which is exactly what `echo_base` is for. Without
            # `echo_base` this check is deferred to the trained gate, and the
            # preflight SAYS SO rather than passing silently.
            if getattr(cfg, "echo_base", False):
                if torch.equal(a["g_tac"], bq["g_tac"]):
                    print("[v3-preflight] ⛔ FAIL: ego does not move "
                          "g_tac (E11' DEAD - this is v3 wearing a v4 config)")
                    return 4
                want = v3.kinematic_goal_extrapolation(
                    e_hi[:, 0], e_hi[:, 1], e_hi[:, 3], cfg.goal_tau_seconds)
                if not torch.equal(bq["g_tac"], want):
                    print("[v3-preflight] ⛔ FAIL: E14 base is not the "
                          "kinematic extrapolation at init - the residual head "
                          "is not zero-init, so the arm does NOT start at "
                          "ha0_ext and every echo-ratio reading is off-scale")
                    return 4
            else:
                print("[v3-preflight] ⚠ ego->goal is zero-init and "
                      "UNVERIFIABLE at step 0 without --echo-base; the live "
                      "check is `echo_ratio` in the training log")
            # (2) frames STILL move the vision-borne nodes (C109: a probe that
            #     cannot fire proves nothing). ⛔ `g_tac` is deliberately
            #     EXCLUDED under `echo_base`: its residual head is zero-init, so
            #     it is vision-dead AT INIT BY DESIGN. That is the one node whose
            #     liveness the preflight genuinely cannot assert - and weakening
            #     the check to cover it would be exactly the vacuous pass C109
            #     names, so it is deferred to the logged `echo_ratio` instead.
            for k in ("g_str", "z_tac"):
                if torch.equal(a[k], c[k]):
                    print(f"[v3-preflight] ⛔ FAIL: frames do not move {k}"
                          f" - probe UNPOWERED, not clean (C109)")
                    return 4
            # (3) the ego block is WITHHELD honestly: keep=0 zeroes the base.
            e_off = e_hi.clone(); e_off[:, 4] = 0.0
            with torch.no_grad():
                d = model(frames, v0=torch.tensor([9.0, 4.0]),
                          ego_state=e_off)
            if getattr(cfg, "echo_base", False) and                     not torch.equal(d["echo_base"],
                                    torch.zeros_like(d["echo_base"])):
                print("[v3-preflight] ⛔ FAIL: keep=0 did not zero the "
                      "E14 base - a withheld ego block is leaking (X15)")
                return 4
            print("[v3-preflight] E11' OK: ego reaches the goal path, frames "
                  "still move g_str/z_tac, keep=0 withholds cleanly")
        else:
            # ================= REF-C v3 (E11, unchanged) =================
            with torch.no_grad():
                a = model(frames, v0=torch.tensor([0.0, 0.0]))
                bq = model(frames, v0=torch.tensor([9.0, 4.0]))
                c = model(torch.rand_like(frames), v0=torch.tensor([0.0, 0.0]))
            for k in ("g_str", "g_tac", "z_tac"):
                if not torch.equal(a[k], bq[k]):
                    print(f"[v3-preflight] ⛔ FAIL: v0 leaked into {k} "
                          f"(E11)")
                    return 4
                if torch.equal(a[k], c[k]):
                    print(f"[v3-preflight] ⛔ FAIL: frames do not move {k}"
                          f" - probe UNPOWERED, not clean (C109)")
                    return 4
        model.train()
    # one synthetic end-to-end loss step (fail here, not on the pod). The lan
    # LABEL pathway is exercised on the train path, not here — the hier loss
    # skips loss_gstr when `lan` is absent BY DESIGN, and the preflight's job
    # is the shared+goal surfaces.
    eps = _synth_episodes(2, cfg.core, seed=0)
    ds = V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                   channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    # ⛔ THE PREFLIGHT MUST EXERCISE THE LAUNCH CONFIG, NOT A NEIGHBOUR OF IT.
    # MEASURED 2026-09-02: with --v7-labels the model is built with 8-wide
    # z_tac heads, but the synthetic preflight corpus carries no v7 records, so
    # the loss fell back to kin3 and the vocabulary refusal fired — the gate
    # working correctly on a config nobody would launch. A preflight that can
    # only pass in a shape the real run never takes is not a preflight, so the
    # synthetic batch is given v7-SHAPED labels (one in-band row and one
    # ignored row, which also exercises the -100 path the real corpus produces
    # on ~76 % of windows).
    if getattr(args, "v7_labels", None):
        n_lat, n_lon = len(v7l.HEADS["tac_lat"]), len(v7l.HEADS["tac_lon"])
        batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
        batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID],
                                       dtype=torch.long)
        assert n_lat >= 1 and n_lon >= 1
    # --nav-from-v7: the launch feeds v7.2 tokens, so the preflight batch gets
    # v7-SHAPED nav too (one turn each way, both valid) and the position pin
    # runs HERE — a broken pin must fail the preflight, not the pod launch.
    if getattr(args, "nav_from_v7", False):
        try:
            assert_nav_token_alignment()
        except AssertionError as e:
            print(f"[v3-preflight] ⛔ FAIL: {e}")
            return 7
        batch["nav_cmd"] = torch.tensor(
            [refb.NAV_COMMANDS.index("left"), refb.NAV_COMMANDS.index("right")],
            dtype=torch.long)
        batch["nav_valid"] = torch.tensor([True, True])
    losses = compute_losses_v3(model, batch, "cpu", mode="diffusion")
    bad = [k for k, t in losses.items()
           if torch.is_tensor(t) and not bool(t.isfinite().all())]
    if bad:
        print(f"[v3-preflight] ⛔ FAIL: non-finite losses {bad}")
        return 5
    print(f"[v3-preflight] loss step OK: "
          f"{ {k: round(float(t.detach()), 4) for k, t in losses.items() if torch.is_tensor(t) and t.ndim == 0} }")
    # ⛔ `route` above is the v2.1 NAV-derived CE, NOT the LAN route. It is
    # masked by nav_valid and is bit-identical with and without every LAN flag
    # (MEASURED, D-LAN-PF) — reading it as evidence about LAN is a category
    # error the LAN arm below exists to prevent.
    if bool(getattr(args, "no_strategic", False)):
        # The LAN arm preflight FAILS the run unless `goal_str` is present,
        # finite and NON-ZERO in the loss dict. Under the bypass that term is
        # deliberately absent, so running it would refuse a correctly
        # configured arm. Say WHY, loudly, rather than skipping silently.
        print("[v3-preflight] --no-strategic: LAN arm preflight SKIPPED -- "
              "`goal_str` is deliberately not in the loss (nothing consumes "
              "g_str under the bypass). The LAN label pathway itself is "
              "unchanged.")
    elif (args.goal_str or args.graft_lan) and cfg.hier:
        rc = _lan_arm_preflight(cfg, args)
        if rc:
            return rc
    print("[v3-preflight] ✅ PASS")
    return 0


# ============================================================================
# Train loop
# ============================================================================

# ============================================================================
# SPEC_REFCV6_V2 §6 — the gradient-conflict detector's three seams
# ============================================================================
#: ⭐ The PLANNING side of the cosine, exactly as ``compute_losses_v3`` weights
#: it into the total. ⛔ The TRAJECTORY term alone, per
#: ``PREREG_BEV_CAPACITY_COMPETITION.md`` §4 (``g_traj = dL_traj/dtheta``), not
#: the whole planner objective: the pre-registered quantity is the one that gets
#: measured, and widening it afterwards would be choosing a statistic after
#: seeing a number.
CONFLICT_PLAN_TERMS = (("traj", lambda m: TRAJ_WEIGHT),)

#: ⭐ The PERCEPTION side: every aux term that reaches the shared trunk, SUMMED,
#: each at the weight it actually enters ``loss`` with. ⛔ Keyed on the loss dict
#: so a term that is absent from THIS run is absent from the cosine rather than
#: contributing a silent zero. ``map`` / ``box3d`` are refcv6 §6's two heads and
#: are listed BEFORE their loss wiring lands, deliberately: the alternative is a
#: detector that silently ignores the head it was built for.
CONFLICT_PERCEPTION_TERMS = (
    ("bev", lambda m: float(getattr(m, "_w_bev_aux", 0.0))),      # WP-D
    ("map", lambda m: float(getattr(m, "_w_map", 0.0))),          # refcv6 SAM3
    ("box3d", lambda m: float(getattr(m, "_w_box3d", 0.0))),      # refcv6 boxes
    # ⭐⭐ THE FOURTH GRADIENT (PI RULING 2026-09-17 R3). The ruling states the
    # trunk is now optimised FOUR ways — planner, map head, box head AND the
    # tactical behaviour decoder — and names this detector as the mitigation for
    # the attribution that costs. A detector whose aux sum omits the tactical
    # term cannot perform that mitigation.
    # ⛔ AND THE BLINDNESS PRE-DATES THE RULING. MEASURED 2026-09-17 on a
    # tactical-only backward: the tactical loss reaches the trunk with
    # `grad_abs_sum` 78,146 on the AGENT-ONLY arm (agent slots are decoded from
    # the trunk's own feature map), so this gradient has been arriving —
    # unmeasured — for as long as the term has existed. R3 is what makes it
    # load-bearing, not what creates it.
    # ⚠️ CONSEQUENCE, STATED RATHER THAN DISCOVERED: on an arm carrying BOTH a
    # perception weight and `--w-tac-v6`, the aux side of every `cd_*` row now
    # includes the tactical term, so those numbers are NOT comparable with a
    # pre-2026-09-17 run's. Nothing banked is known to be affected — no arm
    # could run `--tac-decoder-d-bev > 0` before tonight — but a reader
    # comparing across that date must know.
    ("tac_v6", lambda m: float(getattr(m, "_w_tac_v6", 0.0))),    # refcv6 §4
)


def _conflict_override(args):
    """``--conflict-detector auto|on|off`` -> ``None`` / ``True`` / ``False``."""
    v = str(getattr(args, "conflict_detector", "auto") or "auto").lower()
    return None if v == "auto" else (v == "on")


def _conflict_aux_weights(model) -> dict:
    """-> ``{loss key: weight}`` for every perception term with a LIVE weight.

    ⛔ A weight of 0.0 is not a perception term: it reaches the trunk with a zero
    gradient, and a cosine against the zero vector is the degenerate NaN, not a
    measurement. Empty means "this arm has no second gradient", and
    :func:`grad_conflict.enabled_for_arm` then keeps the detector off rather than
    logging NaN for the whole run.
    """
    return {k: w for k, wf in CONFLICT_PERCEPTION_TERMS
            for w in (wf(model),) if w > 0.0}


def _conflict_terms(model, losses):
    """-> ``(L_traj, L_aux_summed)`` as WEIGHTED tensors, or ``(None, None)``.

    The weights are the ones the term enters ``loss`` with, so the cosine is
    between the gradients that actually arrive at the trunk -- not between two
    unweighted losses whose real influence differs by orders of magnitude.
    ``(None, None)`` when this batch carried no supervised aux (an all-NO_LABEL
    batch), which is a genuine absence and is logged as a missing row rather
    than as a zero.
    """
    plan = None
    for key, wf in CONFLICT_PLAN_TERMS:
        t = losses.get(key)
        if t is None or not torch.is_tensor(t):
            continue
        term = float(wf(model)) * t
        plan = term if plan is None else plan + term
    aux = None
    for key, w in _conflict_aux_weights(model).items():
        t = losses.get(key)
        if t is None or not torch.is_tensor(t) or not t.requires_grad:
            continue
        term = w * t
        aux = term if aux is None else aux + term
    if plan is None or aux is None or not plan.requires_grad:
        return (None, None)
    return (plan, aux)


def _grad_probe_row(model, names, log_every_hit: bool = True) -> dict:
    """Per-module ``sum(|grad|)``, read BEFORE ``clip_grad_norm_`` rescales it.

    D-TACGOAL-2 / PI queue item 10. `tac_goal_tok_head` was 11,286 params at
    ``grad_abs_sum`` EXACTLY 0.00000 for all 40,284 steps of refcv5-v2 because
    ``--w-tac-goal`` defaults to 0.0 and was never passed. "Rollable and trained
    are different claims." The only way to answer *at what weight does this head
    learn* is to read the gradient the head actually receives, per step, per
    weight -- so this probe exists, and it is OFF unless asked for.

    MEASURED BEFORE CLIPPING ON PURPOSE. ``clip_grad_norm_`` rescales every grad
    by a GLOBAL factor, so a post-clip reading confounds "this head's own
    gradient" with "how big everything else's gradient was this step".

    THE RETURNED VALUES ARE NOT ROUNDED. The caller's log row rounds to 5 dp; a
    genuinely non-zero grad at a small weight (1e-8) would round to 0.0 and
    manufacture exactly the defect this probe was built to detect. The caller
    therefore merges this dict AFTER its own rounding comprehension.

    ``found`` is emitted per module so a typo'd module name reads as
    ``gp_<name>_found = 0.0`` rather than as a silent absence -- an unreadable
    module and a module with no gradient must never look the same.
    """
    out = {}
    if not names or not log_every_hit:
        # The OFF guarantee belongs to the HELPER, not only to its call site: a
        # run that names no module computes nothing and emits no key, so its
        # metrics.jsonl schema is identical to the pre-probe trainer's.
        return out
    for nm in names:
        try:
            mod = model.get_submodule(nm)
        except (AttributeError, TypeError):
            mod = None
        if mod is None:
            out["gp_%s_found" % nm] = 0.0
            continue
        n_t = n_none = n_p = 0
        g_sum = 0.0
        for p in mod.parameters(recurse=True):
            n_t += 1
            n_p += int(p.numel())
            if p.grad is None:
                n_none += 1
            else:
                g_sum += float(p.grad.detach().abs().sum().item())
        out["gp_%s_found" % nm] = 1.0
        out["gp_%s_grad_abs_sum" % nm] = g_sum
        out["gp_%s_n_grad_none" % nm] = float(n_none)
        out["gp_%s_n_tensors" % nm] = float(n_t)
        out["gp_%s_n_params" % nm] = float(n_p)
    if torch.cuda.is_available():
        # On Thor `mem_get_info`, `free`/`tegrastats` and `VmRSS` all lie, in
        # BOTH directions (3.4 GB "free" against 60 GB allocated and written).
        # Only the in-process allocator counter is admissible there.
        out["gp_cuda_max_mem_gb"] = (
            float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3))
    return out


def train(args) -> dict:
    # --nav-from-v7 (E-ARCH-NAVSRC-1): refuse a mis-specified switch BEFORE any
    # data or GPU work; a no-op with the flag off.
    _check_nav_from_v7_args(args)
    _check_max_speed_args(args)            # E16; no-op with the flag off
    _check_goal_point_args(args)           # E15 (GP-2); no-op with the flags off
    # ⛔ NOT REDUNDANT WITH THE `preflight` CALL. `main` runs `preflight`
    # ONLY under `--preflight` and otherwise calls `train` directly, so a
    # guard wired into `preflight` alone covers one launch path of two.
    check_effective_weights(args)
    nav_on = bool(getattr(args, "nav_from_v7", False))
    # ⭐ THE ANCHOR ARTIFACT IS READ FIRST: a units-less legacy file is refused
    # before any data or GPU work, the resolved units and the file's own
    # derivation constants reach `_pin_trainer_cfg` below, and the decoder is
    # BUILT under the constants the file was rolled with.
    art = _read_anchor_artifact(args)
    # MEASURED on the A40 pod 2026-09-02: the FIRST real launch died within
    # seconds -- `DataLoader worker killed by signal: Bus error ... out of
    # shared memory`. B1 payloads are ~34 MB/clip and torch's DEFAULT tensor
    # sharing passes them through /dev/shm, which is 24 GB in this container;
    # 4 workers x an LRU of decoded clips exhausts it almost immediately.
    # `file_system` sharing moves those handles off /dev/shm at no cost here
    # (the payloads are already files on disk) -- the same fix the throughput
    # sweep needed. MUST be set before any worker forks, hence the very top.
    if getattr(args, "workers", 0) > 0:
        import torch.multiprocessing as _tmp
        _tmp.set_sharing_strategy("file_system")
        print(f"[v3] tensor sharing = file_system (workers={args.workers}; "
              f"/dev/shm 24 GB vs ~34 MB/clip payloads)", flush=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    cfg = _pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    # ⛔ The delta is derived AT THE SAME SIZE both arms run at. Deriving it at a
    # different rung would compare a config pair neither arm uses. The pair is
    # pinned through the same helper as the built arm, so the recorded delta is
    # the delta of the configs that actually train.
    delta = v3.config_delta(
        _pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=True), args),
        _pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=False), args))
    if set(delta) != REGISTERED_DELTA_KEYS:
        raise SystemExit(f"[v3] ⛔ config delta {sorted(delta)} != registered "
                         f"{sorted(REGISTERED_DELTA_KEYS)} — amend the prereg "
                         f"first (C122).")
    _check_anchor_artifact_against_cfg(art, cfg, args)
    if args.graft_lan or args.goal_str:
        cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))

    model = v3.RefCV3Model(cfg).to(device)
    # refcv5: the two loss weights travel ON the model, because
    # `compute_losses_v3(model, batch, device, ...)` has no `args`. PLAIN
    # attributes, never buffers -- they must not enter state_dict and change
    # checkpoint compatibility (the `_seam_conf` discipline).
    model._w_agent = float(getattr(args, "w_agent", AGENT_WEIGHT_DEFAULT))
    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))   # WP-D
    model._bev_shuffle = bool(getattr(args, "bev_aux_shuffle", False))
    # ---- refcv6 §2/§6: THE PERCEPTION BRANCH -------------------------- #
    # ⛔ BUILT ONLY BEHIND A POSITIVE WEIGHT, and BEFORE `build_optimizer`,
    # which is the whole reason it is here and not later: `torch.optim.Adam(
    # model.parameters(), ...)` enumerates once. A branch attached after that
    # line would be a head with parameters, gradients, and NO OPTIMIZER STATE
    # -- it would never move, and every metric would read as "the perception
    # heads do not help".
    # ⭐ At weight 0.0 NOTHING happens here: no submodule, so `model.parameters()`
    # is the same list, `state_dict()` the same keys, and the RNG draw order
    # unchanged. That is the bit-identity condition, met by construction.
    model._w_map = float(getattr(args, "w_map", 0.0) or 0.0)
    model._w_box3d = float(getattr(args, "w_box3d", 0.0) or 0.0)
    model._perception = None
    model._lift_bank = None
    perception_stamp = None
    if model._w_map > 0.0 or model._w_box3d > 0.0:
        _pcfg = _perc.PerceptionBranchConfig(w_map=model._w_map,
                                             w_box3d=model._w_box3d)
        model._perception = _perc.build_perception_branch(model, _pcfg).to(device)
        _pframe = _perc.frame_for_model(model)
        if model._w_map > 0.0:
            _pe, _ptable = _read_rig_extrinsics(
                str(getattr(args, "agent_rig_extrinsics", "")))
            if _ptable is None:
                raise SystemExit(
                    "[v3] ⛔ --w-map > 0 needs a PER-CLIP extrinsics table "
                    "(clip_id -> pose); this file carries a single camera. "
                    "One mount pose for the whole corpus biases every BEV "
                    "cell the lift fills.")
            model._lift_bank = _perc.LiftGeometryBank(
                _ptable, frame=_pframe, stride=int(_pcfg.stride))
        perception_stamp = {
            **_pcfg.as_dict(),
            "branch_params": model._perception.param_breakdown(),
            "fmap_s16_channels": int(model.core.encoder.s16_dim),
            "fmap_s16_hw": list(model.core.encoder.s16_shape),
            "frame": {"height": int(_pframe.height), "width": int(_pframe.width),
                      "projection": str(_pframe.projection),
                      "f_ref": float(_pframe.f_ref)},
            "map_gt_root": str(getattr(args, "map_gt_root", None) or "") or None,
            "join3d": str(getattr(args, "join3d", None) or "") or None,
            "lift_bank_n_clips": (0 if model._lift_bank is None
                                  else len(model._lift_bank)),
            # ⛔ The PI's constraint, IN THE RUN RECORD: a reader who opens
            # config.json in isolation learns which corpus was the BEV target.
            "bev_map_target": "SAM3 semantic maps (tanitad.sam3_map_gt/2)",
            "lidar_as_training_target": False,
        }
        print("[v3] refcv6 perception: w_map=%.4g w_box3d=%.4g; params %s"
              % (model._w_map, model._w_box3d,
                 model._perception.param_breakdown()), flush=True)
    # ⛔⛔ refcv6 §6: REFUSE A DETECTOR THAT CAN NEVER READ ANYTHING -- HERE,
    # before `config.json` is written and before a single batch is loaded.
    # MEASURED 2026-09-17 on the smoke path: `--conflict-detector on` with no
    # live perception weight built the detector, printed "ON", stamped
    # `conflict_detector.enabled=true` -- and emitted ZERO `cd_*` rows for the
    # whole run, because there is no second gradient. That is the `--w-agent`
    # defect verbatim: a run whose record claims an instrument it never read.
    # ⭐ `auto` is already correct (it returns False with no aux); only the
    # explicit override can reach this, so only the override is refused.
    if _conflict_override(args) is True and not _conflict_aux_weights(model):
        raise SystemExit(
            "[v3] ⛔ --conflict-detector on, but NO perception term has a live "
            "weight (%s). The detector would be built and stamped while its "
            "second gradient does not exist, and metrics.jsonl would carry no "
            "cd_* key for the entire run -- an instrument in the run record "
            "that was never read. Pass a perception weight (--bev-aux col|xcol "
            "with --w-bev-aux > 0 and --agent-join), or --conflict-detector "
            "auto." % ", ".join("%s=%s" % (k, wf(model))
                                for k, wf in CONFLICT_PERCEPTION_TERMS))
    # D-TACGOAL: same carrier, same reason. ⛔ `_tac_goal_pos_weight` and
    # `_tac_goal_class_mask` are set from THE LOADED SPLIT further down (never
    # from a literal -- `goal_pos_weight.__doc__` is explicit, and it is the
    # derived-constant trap that moved HORIZON 7 -> 8). They stay None here so
    # a w>0 arm that somehow reached the loop without a join trains UNWEIGHTED
    # and UNMASKED rather than silently using another split's constants.
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    # ⭐ refcv6 §4: same carrier, same reason. The pos_weight and class_mask
    # above are SHARED with this channel and are fitted from the loaded split
    # further down, never from a literal.
    model._w_tac_v6 = float(getattr(args, "w_tac_v6", 0.0) or 0.0)
    # ⛔⛔ A BUILT DECODER WITH NO LIVE WEIGHT IS REFUSED HERE TOO — and this
    # is NOT redundant with `_pin_refcv6_tactical`. That guard reads ARGV;
    # this one reads THE MODEL THAT WAS ACTUALLY BUILT. The 2026-09-17
    # `--image-hw` post-mortem is exactly this distinction: a refusal that
    # diffs argv PASSED while the defect lived between argv and the model.
    _tac6_built = getattr(model, "tac_decoder_v6", None) is not None
    if _tac6_built != (model._w_tac_v6 > 0.0):
        raise SystemExit(
            "[v3] ⛔ refcv6 §4 disagreement between the MODEL and the WEIGHT: "
            "`model.tac_decoder_v6` is %s but --w-tac-v6 is %.6g. %s This is "
            "read off the BUILT OBJECT, not off argv, because a guard that "
            "diffs argv cannot see a defect that lives between argv and the "
            "model."
            % ("BUILT" if _tac6_built else "None", model._w_tac_v6,
               ("The decoder's parameters would receive no gradient for the "
                "whole run — the `tac_goal_tok_head` defect (11,286 params, "
                "grad_abs_sum exactly 0, 40,284 steps) at 200x the scale."
                if _tac6_built else
                "The loss would have no logits to read and would refuse at "
                "the first batch, after the model was already built.")))
    model._w_u0 = float(getattr(args, "w_u0", U0_WEIGHT_DEFAULT))
    # E15 (GP-2): same carrier, same reason. `_check_goal_point_args` has
    # already refused `--goal-point-inject` with a zero weight, so a built head
    # is always supervised.
    model._w_goal_point = float(getattr(args, "goal_point_w",
                                        GOAL_POINT_WEIGHT_DEFAULT))
    # ⛔ WIRED, not `= None`. Until 2026-09-05 this line read `= None` and the
    # attribute was never assigned, so `--agent-w-project` / `--agent-w-ground`
    # were SILENT NO-OPS while being stamped into config.json (mm-decisions
    # M18). `_build_rig_camera` REFUSES at pin time when a non-zero weight has
    # no camera, so reaching this line with a weight > 0 and `cam is None` is
    # now unreachable from any argv.
    model._rig_camera, _cam_stamp = _build_rig_camera(cfg, args)
    # ⛔ M18 ONE LEVEL IN: the camera exists and the term still trains
    # nothing. MEASURED at startup, with its own positive control.
    _ground_probe = assert_ground_prior_is_supervised(model, args)
    # ⛔ THE ANCHOR VOCABULARY IS LOAD-BEARING AND ITS ABSENCE WAS SILENT.
    # refcv3 trained without `--anchors` and nobody noticed for the whole run,
    # because this branch printed nothing and recorded nothing. MEASURED cost:
    # the synthetic fallback's oracle-in-vocabulary ADE is 0.9433 m, WORSE than a
    # single straight line (0.6780 m), against the data-driven set's 0.4369 m — so
    # the model's CEILING was below the trivial floor before training began.
    # The base trainer has always been loud here (`refc_train.py:918-923`); this
    # one was not. Both branches now speak, and the default is a WARNING, not a
    # default. See RETRACTION_LOG and `.../2026-09-04-refcv3-smoothness/`.
    if args.anchors:
        assert art is not None                       # read at the top of train()
        anc = art.anchors.to(device)
        _ctrl = None if art.controls is None else art.controls.to(device)
        # ⛔ BOTH DIRECTIONS ARE REFUSED, LOUDLY. A v0-conditioned build given a
        # controls-free file would silently fall back to fixed paths — the exact
        # vocabulary this arm exists to replace — and a fixed build given a
        # controls file would train against a bank rolled at one reference speed
        # while the file's author meant per-window. Neither failure is visible in
        # any artifact the run leaves behind, so neither is allowed to be silent.
        _want = bool(getattr(args, "anchor_v0_conditioned", False))
        if _want and _ctrl is None:
            raise SystemExit(
                "[v3] ⛔ --anchor-v0-conditioned given but "
                f"{args.anchors} carries no `controls` [N, 2]. The bank is "
                "rolled per window from those controls; there is nothing to "
                "roll.")
        if _ctrl is not None and not _want:
            raise SystemExit(
                f"[v3] ⛔ {args.anchors} carries `controls` [N, 2] (a "
                "v0-CONDITIONED vocabulary) but --anchor-v0-conditioned was NOT "
                "given. Its paths are the family rolled at the REFERENCE speed "
                "only, and training on them fixed would silently be a different "
                "experiment.")
        model.core.decoder.load_anchors(
            anc.to(device), None if _ctrl is None else _ctrl.to(device))
        try:
            import hashlib
            _sha = hashlib.sha256(
                anc.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
        except Exception:                      # provenance is best-effort, never fatal
            _sha = "unavailable"
        print(f"[v3] anchors: loaded {tuple(anc.shape)} from {args.anchors} "
              f"(sha256 {_sha})", flush=True)
        if _ctrl is not None:
            _ok = bool(((_ctrl[:, 0] == 0) & (_ctrl[:, 1] == 0)).any())
            _u = art.control_units
            print(f"[v3] anchors: v0-CONDITIONED, controls {tuple(_ctrl.shape)} "
                  f"(accel, {'lateral accel' if _u == 'alat' else 'curvature'}), "
                  f"units={_u} (source: {art.control_units_source}), rolled "
                  f"per window; withheld rows at "
                  f"{getattr(args, 'anchor_ref_speed', 10.0)} m/s. "
                  f"straight-ahead control present: {_ok}", flush=True)
            if not _ok:
                # ⚠️ RETRACTED 2026-09-05 (RETRACTION_LOG): this message used
                # to say "Rebuild with odd counts". An odd count is neither
                # necessary nor sufficient -- np.linspace(-4, 3, 13) has 13
                # nodes and NO zero (step 7/12: ..., -0.5, +0.0833, ...); only
                # a SYMMETRIC odd grid, or a re-centred one, contains 0.0.
                raise SystemExit(
                    "[v3] ⛔ the control grid does not contain {a=0, kappa=0} "
                    "EXACTLY. A linspace that does not pass through 0.0 -- an "
                    "even count on a symmetric range, or ANY count on an "
                    "asymmetric one such as np.linspace(-4, 3, 13) -- omits "
                    "it, and the set then reads 1.2768 m oracle-in-vocabulary "
                    "against 0.2610 — a 4.9x artifact that looks exactly like "
                    "a resolution finding. Rebuild so that 0.0 is a node of "
                    "BOTH axes (assert `np.any(grid == 0.0)`; re-centre if "
                    "the range is asymmetric).")
    else:
        print("[v3] ⛔ anchors: NO --anchors GIVEN — using the SYNTHETIC "
              "`default_anchors` fallback. Its oracle-in-vocabulary ADE was MEASURED "
              "at 0.9433 m, worse than a single straight line (0.6780 m) and 2.16x "
              "the data-driven vocabulary (0.4369 m). The model's ceiling is below "
              "the trivial floor. This is almost certainly NOT what you want.",
              flush=True)

    # data — raw epcache, v2 compressed cache, or CI-synthetic; exactly one
    n_src = sum(1 for s in (args.data_root, args.v2_cache,
                            args.synth_episodes) if s)
    if n_src != 1:
        raise SystemExit("[v3] pass exactly one of --data-root / --v2-cache / "
                         "--synth-episodes (the synthetic corpus is CI-only "
                         "and must never masquerade as a training cache)")
    v2_parity = None
    synth_clip_ids_from_labels = None
    if args.synth_episodes:
        # ⭐ When a label blob is supplied, stamp the synthetic episodes with
        # REAL clip ids so the v7/v8 join HITS. Without this the join dies on
        # `int('synth-000')` and no v7-label channel can be smoked at all.
        # See `_synth_episodes.__doc__`. ⛔ Frames stay synthetic; only the id
        # is real, and `config.json` records that it happened.
        if args.v7_labels:
            _pre_lab, _ = v7l.load_v7_labels(args.v7_labels,
                                             allow_oracle_nav=True)
            synth_clip_ids_from_labels = [l.clip_id for l in
                                          _pre_lab[:args.synth_episodes]]
        # ⛔⛔ THE KWARG IS PASSED ONLY WHEN IT CARRIES SOMETHING, AND THAT IS
        # NOT STYLE. MEASURED 2026-09-10: passing `clip_ids=None`
        # unconditionally broke THREE tests in
        # `test_refc_v3_save_before_eval.py` that monkeypatch
        # `_synth_episodes` with their own two-line shim -- `TypeError:
        # _train_eps() got an unexpected keyword argument 'clip_ids'`. Those
        # rigs are correct and there is no reason for a NEW optional argument
        # to reach a caller that never asked for it. ⭐ Fixing the CALL SITE
        # once beats amending every shim, and it closes the whole class rather
        # than the two instances that happened to be found.
        _synth_kw = ({"clip_ids": synth_clip_ids_from_labels}
                     if synth_clip_ids_from_labels else {})
        eps = _synth_episodes(args.synth_episodes, cfg.core, seed=args.seed,
                              **_synth_kw)
    elif args.v2_cache:
        # the flagship's --v2-cache recipe (train_flagship4b.py), verbatim in
        # structure: membership guard BEFORE any GPU work, lazy LRU-bounded
        # providers (contract-identical episode surface), and the independent
        # provider-vs-guard count cross-check. `require=False` keeps
        # deliberately-non-parity corpora usable behind ONE loud line;
        # --require-parity turns the same check into a refusal.
        from tanitad.data import parity
        from tanitad.data.v2_dataset import build_v2_providers
        v2_parity = parity.assert_v2_parity_cache(
            args.v2_cache, label="v3 v2-cache", require=args.require_parity)
        eps = build_v2_providers(args.v2_cache, lru_size=args.v2_lru)
        if v2_parity.get("parity") and len(eps) != v2_parity["episodes_loaded"]:
            raise parity.ParityViolation(
                f"PARITY VIOLATION [v3 v2-cache]: the guard verified "
                f"{v2_parity['episodes_loaded']} clip files but the loader "
                f"built {len(eps)} providers — the _v2manifest.pt sidecar "
                f"disagrees with the directory; rebuild it "
                f"(build_v2_providers(..., rebuild=True)) and re-run.")
        if args.episodes:
            eps = eps[:args.episodes]
        print(f"[v3] {len(eps)} lazy v2 providers from {args.v2_cache} "
              f"(lru {args.v2_lru})")
    else:
        eps, train_dir = load_cached_episodes(args.data_root, "*train*",
                                              args.episodes)
        print(f"[v3] {len(eps)} episodes from {train_dir}")
    # ⛔ geometry is asserted against the EPISODES, not against the flag: a
    # 256x640 corpus fed to a 256x256 build would run (conv is size-agnostic)
    # and silently train a model whose config lies about its input.
    eh, ew = cfg.core.encoder.image_hw()
    fh, fw = int(eps[0].frames.shape[-2]), int(eps[0].frames.shape[-1])
    if (fh, fw) != (eh, ew):
        raise SystemExit(
            f"[v3] ⛔ geometry mismatch: encoder built for {eh}x{ew} but the "
            f"corpus emits {fh}x{fw} — pass --image-hw {fh} {fw} (params are "
            f"unchanged; only compute scales), or point at a matching cache.")
    want_lan = bool(args.graft_lan or args.goal_str)
    dcls = lan_dataset_class(V3Dataset) if want_lan else V3Dataset
    kw = dict(window=cfg.core.window, max_horizon=20,   # ⛔ parity: NEVER 60
              channels=cfg.core.encoder.in_channels)
    if want_lan:
        kw["lan_cfg"] = DataLanConfig(
            arclengths_m=tuple(args.lan_arclengths),
            min_lead_m=args.lan_min_lead_m)
    ds = dcls(eps, **kw)
    # --u8-batches: the dataset emits frames AS STORED (uint8); the device-side
    # /255 is frames_to_device in compute_losses_v3. Instance attribute — the
    # V3Dataset idiom; see FailLoudWindowDataset.u8_frames (refb_train.py).
    u8 = bool(getattr(args, "u8_batches", False))
    ds.u8_frames = u8
    nav_stats = eval_nav_stats = v7_manifest = None
    tac_goal_stats = None                       # D-TACGOAL
    nav_args_stats = eval_nav_args_stats = None
    max_speed_stats = eval_max_speed_stats = None
    max_speed_v6_stats = eval_max_speed_v6_stats = None
    tac_v6_stats = None
    # ---- v7.2 label join (PI 2026-09-02: MANDATORY for this launch) --------
    if args.v7_labels:
        from tanitad.data.v2_dataset import stable_episode_id
        labels, manifest = v7l.load_v7_labels(args.v7_labels,
                                              allow_oracle_nav=True)
        by_sid = {stable_episode_id(l.clip_id): l for l in labels}
        ds.v7_by_sid = by_sid
        ds.v7_dt = 0.1
        # ---- ⭐ PI 2026-09-16: the absence-is-negative sidecar ------------
        # ⛔ Loaded HERE, before the manifest is stamped into the run record,
        # so a run that used the ruling cannot produce a config.json that does
        # not say so. The guard runs against the ACTUAL loaded split: a future
        # blob that makes a new token CoT-backed fails at launch rather than
        # half-applying the ruling for a whole training run.
        if getattr(args, "cot_negative_sidecar", None):
            cot_sc, manifest = v7l.load_cot_negative_sidecar(
                args.cot_negative_sidecar, manifest)
            v7l.assert_sidecar_matches_presence(labels, cot_sc)
            ds.cot_negative_sidecar = cot_sc
            print(f"[v3] cot-absence-negative: {len(cot_sc.tokens)} tokens, "
                  f"{len(cot_sc.by_digest)} clips, sidecar md5 {cot_sc.md5}",
                  flush=True)
        v7_manifest = manifest.to_dict()      # md5 + BOTH stamps
        # ⛔ COVERAGE IS REPORTED, NOT ASSUMED. MEASURED 2026-09-02: the v7.2
        # train set joins 4,572/4,713 = 97.0 % of B1 but only 190/2,400 =
        # 7.9 % of the PARITY corpus — so the same flag on the wrong cache
        # silently supervises the tactical heads on ~8 % of clips. The launch
        # record must carry the number, and a low one must be loud.
        # ⛔ A NON-INTEGER episode_id used to die here as a bare
        # `ValueError: invalid literal for int() ... 'synth-000'`, which reads
        # like a broken FLAG and is actually a corpus that structurally cannot
        # carry a label. Name it instead. (MEASURED 2026-09-10; the synthetic
        # corpus can now be stamped with real clip ids — `_synth_episodes`.)
        try:
            hit = sum(1 for e in eps if int(e.episode_id) in by_sid)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"[v3] ⛔ --v7-labels: this corpus's `episode_id` is not an "
                f"integer, so the label join ({len(by_sid)} records keyed by "
                f"`stable_episode_id(clip_id)`) can never hit. This is NOT the "
                f"label blob failing. The synthetic corpus stamps real clip "
                f"ids only when --v7-labels is passed to --synth-episodes; a "
                f"v2 cache always carries integer ids. ({exc})") from None
        frac = hit / max(len(eps), 1)
        print(f"[v3] v7.2 labels: {len(labels)} records, joined "
              f"{hit}/{len(eps)} episodes = {100 * frac:.1f} % "
              f"(release {manifest.release if hasattr(manifest, 'release') else '?'})")
        if frac < 0.5:
            raise SystemExit(
                f"[v3] ⛔ v7.2 label coverage {100 * frac:.1f} % — refusing to "
                f"launch. The PI made v7.2 supervision MANDATORY; below half "
                f"the corpus the tactical/strategic heads would train on a "
                f"minority of clips while the run LOOKED labelled. Check the "
                f"cache is B1 (97.0 %) and not the parity corpus (7.9 %).")
        # ---- ⭐⭐ E16: the map/nav posted-limit ceiling, from the SAME
        # label join. ⛔ Independent of --nav-from-v7: the ceiling is its own
        # channel and tying it to the nav source would make two levers one.
        # ---- ⭐⭐ D-TACGOAL: the goal-SET target channel, and the two
        # constants that MUST come from this split rather than a literal.
        # ⛔ `goal_pos_weight` is n_neg/n_pos capped at 50 and `mask_report`
        # switches off every class with positives but NO supervised negative
        # (5 of 22 on the v7.2 train blob) -- an unmasked logit there can only
        # ever be pushed towards 1, which degrades the shared trunk and
        # inflates any pooled score. Both are stamped into config.json, so the
        # run record says what the head was actually trained on.
        # ⭐ refcv6 §2b: the ego-history channel is a DATASET decision, taken
        # here so the trainer's `pose_hist` refusal fires at launch.
        ds.ego_history = bool(getattr(args, "ego_history", False))
        # ⭐⭐ refcv6 §4 SHARES THIS CHANNEL, AND THAT IS THE POINT. The
        # behaviour decoder's 22 validity queries are supervised by the SAME
        # `tactical_goal_targets` (y, w) pair, under the SAME
        # `goal_pos_weight` and the SAME `mask_report` class mask, as
        # `--w-tac-goal`. Re-deriving a second copy would be the
        # derived-constant trap (HORIZON 7 -> 8) with two heads reading two
        # quantizations of one label set.
        # ⚠️ THE LABEL FACTS THAT TRAVEL WITH EVERY TACTICAL NUMBER, and they
        # are computed HERE from the loaded split, never from a literal:
        # after the PI's absence-as-negative ruling 21 of 22 tokens are
        # trainable, but NINE sit ON the `goal_pos_weight` cap of 50 (so for
        # those nine THE CAP, NOT THE DATA, sets the weight) and TEN sit under
        # the n = 200 scoreability floor. `tanitad/train/panel_refusals.py`
        # enforces both; a pooled accuracy over 22 tokens is dominated by
        # FOLLOW_LANE at 79.4 % prevalence and says nothing.
        _w_tv6 = float(getattr(args, "w_tac_v6", 0.0) or 0.0)
        if (float(getattr(args, "w_tac_goal", 0.0) or 0.0) > 0.0
                or _w_tv6 > 0.0):
            ds.tac_goal_targets = True
            ds.tac_goal_negatives = str(getattr(args, "tac_goal_negatives",
                                                "measured"))
            _neg = {"negatives": ds.tac_goal_negatives,
                    "sidecar": ds.cot_negative_sidecar}
            tac_goal_census = v7l.goal_supervision_census(labels, **_neg)
            tac_goal_mask = _tac_goal_head.mask_report(tac_goal_census)
            tac_goal_pw = v7l.goal_pos_weight(labels, **_neg)
            model._tac_goal_pos_weight = torch.tensor(tac_goal_pw,
                                                      dtype=torch.float32)
            model._tac_goal_class_mask = torch.tensor(tac_goal_mask["mask"],
                                                      dtype=torch.float32)
            tac_goal_stats = {
                "negatives": ds.tac_goal_negatives,
                "cot_absence_negative": (ds.cot_negative_sidecar.to_dict()
                                         if ds.cot_negative_sidecar else None),
                "n_trainable": int(tac_goal_mask["n_trainable"]),
                "n_total": int(tac_goal_mask["n_total"]),
                "trainable": list(tac_goal_mask["trainable"]),
                "masked_why": tac_goal_mask["masked_why"],
                "pos_weight": [float(x) for x in tac_goal_pw],
                "census": tac_goal_census,
            }
            # ⛔ PER CLASS, NEVER POOLED, AND THE CAPPED/UNSCOREABLE COUNTS
            # ARE PRINTED BESIDE THE TRAINABLE ONE. A bare "21/22 trainable"
            # invites the reading that 21 classes are learnable from data;
            # for the ones ON the cap it is the CAP that sets the weight.
            # ⛔ THE CAP IS READ FROM `goal_pos_weight`'s OWN SIGNATURE, never
            # retyped. It is a DEFAULT PARAMETER (`cap: float = 50.0`), not a
            # module constant, and a literal 50.0 here would be a second copy
            # that goes stale the day the DataFlyWheel moves it — the
            # derived-constant trap this file already carries three scars from.
            _pw_cap = float(_inspect.signature(
                v7l.goal_pos_weight).parameters["cap"].default)
            _n_capped = sum(1 for x in tac_goal_pw
                            if float(x) >= _pw_cap - 1e-6)
            _n_unscoreable = sum(
                1 for _t in vocab_v7.TACTICAL_GOAL_TOKENS_V7
                if int((tac_goal_census.get(_t) or {}).get("pos", 0) or 0)
                < vocab_v7.GOAL_MIN_N_FOR_METRIC)
            tac_goal_stats["n_on_pos_weight_cap"] = int(_n_capped)
            tac_goal_stats["pos_weight_cap"] = float(_pw_cap)
            tac_goal_stats["n_under_scoreability_floor"] = int(_n_unscoreable)
            tac_goal_stats["scoreability_floor_n"] = int(
                vocab_v7.GOAL_MIN_N_FOR_METRIC)
            print(f"[v3] tac_goal: {tac_goal_stats['n_trainable']}"
                  f"/{tac_goal_stats['n_total']} classes trainable "
                  f"(negatives={ds.tac_goal_negatives}), "
                  f"w_tac_goal={float(getattr(args, 'w_tac_goal', 0.0) or 0.0)}"
                  f" w_tac_v6={_w_tv6}"
                  f" | {_n_capped} ON the pos_weight cap "
                  f"{_pw_cap:g} (the CAP, not the data, "
                  f"sets their weight), {_n_unscoreable} under the n="
                  f"{int(vocab_v7.GOAL_MIN_N_FOR_METRIC)} scoreability floor "
                  f"-- report PER CLASS, never pooled", flush=True)
            # ⛔ A CHANNEL THAT IS ON AND TEACHES NOTHING IS THE DEFECT THIS
            # PACKAGE CLOSES, ONE LAYER UP. Refuse at LAUNCH, not after a
            # GPU day.
            if tac_goal_stats["n_trainable"] == 0:
                raise SystemExit(
                    "[v3] ⛔ --w-tac-goal/--w-tac-v6 > 0 but ZERO of "
                    f"{tac_goal_stats['n_total']} goal classes are trainable "
                    "on this split (every class lacks positives or lacks a "
                    "supervised negative). The head would be built, stamped "
                    "and supervised by an all-masked target.")
        if getattr(args, "max_speed_input", False):
            max_speed_stats = ds.enable_max_speed(
                manifest, str(getattr(args, "max_speed_mode",
                                      msi.DEFAULT_MODE)))
        # ⛔ refcv6 §5 — the OTHER ceiling channel. `_pin_refcv6_tactical` has
        # already refused the two together, so this `elif` can never shadow
        # E16; it is written as a separate `if` anyway so that a future
        # loosening of the pin cannot make one channel silently win.
        if getattr(args, "max_speed_input_v6", False):
            max_speed_v6_stats = ds.enable_max_speed_v6(
                str(args.speed_max_sidecar_v6), manifest)
        # ---- --nav-from-v7 (E-ARCH-NAVSRC-1, PI 2026-09-02): the nav INPUT
        # from the record's token — the input refav1 already trains on. The
        # v1 derivation feeds `follow` (+invalid) on 94.6 % of B1 windows
        # (MEASURED, nav-source-agreement package); the two agree on 65.5 %.
        if nav_on:
            nav_stats = ds.enable_nav_from_v7(manifest)
            # ⭐ D-GSTR-1 P3 (E13b): the args, fitted HERE — on the TRAIN
            # split — and handed down to the eval dataset unchanged.
            if getattr(args, "nav_args", False):
                nav_args_stats = ds.enable_nav_args()
    # ---- refcv5 WP-6: the obstacle.offline join (E-AGT-HEAD labels) -------
    # ⛔ The reader is RESTRICTED to this corpus's episode ids. The full train
    # join is 433,040 records / 12.1 M boxes and costs ~2.1 GB RSS (MEASURED
    # 2026-09-05); a tiny rig must not pay that, and a run must not silently
    # hold a corpus it is not training on.
    agent_stats = eval_agent_stats = None
    map_stats = eval_map_stats = None
    join3d_stats = eval_join3d_stats = None
    join_digest = _verify_agent_join(args)
    if getattr(args, "agent_join", None):
        from train_p8_occupancy import JoinFileReader
        _t_join = time.time()
        _rd = JoinFileReader(
            args.agent_join,
            episode_ids={int(e.episode_id) for e in eps},
            with_rates=not bool(getattr(args, "agent_join_no_rates", False)),
            # refcv6 §6: the 3-D join is keyed BY TRACK, so the 2-D rows must
            # carry their track ids or `cz`/`h` could only be aligned by
            # position. Paid for ONLY when --join3d is actually passed.
            with_track_ids=bool(getattr(args, "join3d", None)))
        print(f"[v3] agent join loaded: {_rd.n_records} records / "
              f"{_rd.n_clips} clips (filtered out "
              f"{_rd.n_records_filtered_out}) in {time.time() - _t_join:.1f} s")
        # ⭐ WP-D: the SAME reader, the SAME join, one target more. The BEV
        # spec is attached BEFORE `enable_agent_join` runs its census so a
        # single pass over the window index covers both.
        if str(getattr(args, "bev_aux", "off")) != "off":
            ds.bev_spec = _bev_aux.PolarBEVSpec(
                n_az=int(cfg.core.encoder.grid_shape[1]),
                n_rng=int(getattr(args, "bev_aux_rng", 24)),
                r_max_m=float(getattr(args, "bev_aux_rmax", 60.0)))
            ds.bev_occlusion = str(getattr(args, "bev_aux_occlusion", "mask"))
        agent_stats = ds.enable_agent_join(
            _rd, pad=int(getattr(args, "agent_pad", 0)),
            allow_legacy_ids=bool(getattr(
                args, "agent_join_allow_legacy_ids", False)))
    # ---- refcv6 §2/§6: the SAM3 map GT and the 3-D cuboid join ------------ #
    # ⛔ AFTER `enable_agent_join`, because `_agent_item` is what widens to 3-D
    # and the coverage census below walks the same window index.
    # ⛔ THE WHOLE BLOCK IS GATED, INCLUDING THE MANIFEST READ. An unconditional
    # `_clip_table_for_caches` would read every cache's manifest on a run that
    # asked for no perception at all -- work the tip does not do, on a path
    # where `--v2-cache` may be None (`--synth-episodes`). Measured here: it
    # crashed the bit-identity control, which is exactly what that control is
    # for.
    _clip_of_ep, _cache_nstack = {}, 0
    if getattr(args, "map_gt_root", None) or getattr(args, "join3d", None):
        _clip_of_ep, _cache_nstack = _clip_table_for_caches(args.v2_cache)
    if getattr(args, "map_gt_root", None):
        map_stats = ds.enable_map_gt(
            _perception_targets.MapGTStore(
                Path(args.map_gt_root),
                max_open=int(getattr(args, "map_lru", 4) or 4)),
            _clip_of_ep, _cache_nstack,
            min_coverage=getattr(args, "map_min_coverage", None))
    if getattr(args, "join3d", None):
        if ds.agent_join is None:
            raise SystemExit(
                "[v3] ⛔ --join3d without --agent-join: the 3-D join WIDENS "
                "the 2-D target block by track id; with no 2-D block there is "
                "nothing to widen and the heights would be loaded and dropped.")
        if not getattr(ds.agent_join, "has_track_ids", False):
            raise SystemExit(
                "[v3] ⛔ --join3d, but the 2-D join carries no `track_id` on "
                "any agent. The cuboid heights are keyed BY TRACK, so every "
                "target would be masked OFF and the run would report "
                "`box3d_n_z 0` for its whole life while config.json names a "
                "3-D join -- a 3-D arm that is silently the 2-D rung.")
        # ⛔⛔ THE INDEX-SPACE TRAP, AND IT IS SILENT IN BOTH DIRECTIONS.
        # `JoinFileReader` (the 2-D seam) keys on `frame_idx`, the EPISODE
        # index; `AgentJoin3D` keys on `frame`, the RAW v2ep index. They differ
        # by `n_stack - 1` -- 2 frames = 0.2 s at 10 Hz. Join on the wrong key
        # and every `cz`/`h` belongs to a moment 0.2 s from the one the trunk
        # saw; the loss stays finite, non-zero and plausible, and NOTHING
        # RAISES. ⭐ MEASURED by a sweep over offsets -3..+3 on 8 real clips
        # (`…/2026-09-17-refcv6-perception-training/raw/join3d_offset_sweep.json`):
        # at +2 = `n_stack - 1`, 2,955/2,955 agents match the 2-D rows EXACTLY
        # (mean |dx| 0.0000 m); at EVERY other offset 0.0 % match and mean |dx|
        # >= 3.24 m -- including the naive +0, which reads 3.24 m.
        # ⇒ `n_stack` is therefore LOAD-BEARING here, and a 0 would silently
        # apply an offset of -1. It is read from the manifest, and refused.
        if int(_cache_nstack) < 1:
            raise SystemExit(
                "[v3] ⛔ --join3d but the v2 manifest reports no n_stack "
                "(%r). It is what converts the 2-D join's EPISODE index into "
                "the 3-D join's RAW v2ep frame, and a wrong value shifts every "
                "cuboid height by that many frames -- 0.2 s per frame, "
                "finite-and-plausible in the loss, visible in no metric."
                % (_cache_nstack,))
        if ds.map_clip_of_ep is None:
            ds.map_clip_of_ep, ds.map_n_stack = _clip_of_ep, _cache_nstack
        join3d_stats = ds.enable_join3d(_agent_cuboid.open_join3d(
            args.join3d, clips=set(_clip_of_ep.values())))
        join3d_stats["frame_key"] = "RAW v2ep (= episode index + n_stack - 1)"
        join3d_stats["n_stack"] = int(ds.map_n_stack)
        print("[v3] 3-D cuboid join: %s" % join3d_stats, flush=True)
    # launch-line P4: the run PRINTS its episode/window counts at start — the
    # only way a parity claim about the enumeration is checkable from the log.
    print(f"[v3] {len(eps)} episodes -> {len(ds)} windows "
          f"(window {cfg.core.window}, max_horizon 20, "
          f"image_hw {cfg.core.encoder.image_hw()}, "
          f"tac_vocab {cfg.tac_vocab_version})")
    # ---- held-out eval split (PI 2026-09-02: "an eval step at 100 step") ---
    # ⛔ THE LEAK THIS CLOSES, MEASURED: the v7.2 release splits 4,572 train /
    # 147 eval with ZERO intersection, but the raw B1 corpus is 4,713 = 4,572 +
    # 141 of those eval clips. Training on "all of B1" therefore trains on 141
    # eval clips, and every eval number off them would be optimistic and
    # inadmissible. The launch trains the 4,572 and evaluates the held-out set.
    eval_dl = None
    if args.eval_cache and args.eval_every:
        # the SAME provider call the train side uses (refc_v3_train:680) —
        # a second construction path would be a second contract.
        from tanitad.data.v2_dataset import build_v2_providers
        e_eps = build_v2_providers([args.eval_cache], lru_size=args.v2_lru)
        overlap = ({int(e.episode_id) for e in eps}
                   & {int(e.episode_id) for e in e_eps})
        if overlap:
            raise SystemExit(
                f"[v3] ⛔ REFUSING: {len(overlap)} episodes appear in BOTH the "
                f"train cache and the eval cache. A held-out split that is not "
                f"held out measures memorisation.")
        e_ds = dcls(e_eps, **kw)
        e_ds.u8_frames = u8     # the eval decodes in the MAIN process: 4x less there too
        if args.eval_labels:
            from tanitad.data.v2_dataset import stable_episode_id
            e_lab, e_man = v7l.load_v7_labels(args.eval_labels,
                                              allow_oracle_nav=True)
            e_ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}
            e_ds.v7_dt = 0.1
            # the in-training eval sees the SAME nav source as training
            # (refused above when --eval-labels is missing)
            if nav_on:
                eval_nav_stats = e_ds.enable_nav_from_v7(e_man)
                # ⛔ THE EVAL DATASET IS HANDED THE TRAIN SPLIT'S NORMALISER,
                # NEVER ITS OWN. Fitting here would be a statistic fitted on
                # the data it is about to score.
                if getattr(args, "nav_args", False):
                    eval_nav_args_stats = e_ds.enable_nav_args(
                        stats=ds.nav_arg_stats)
            # ⭐⭐ D-TACGOAL-EVAL-1 — the eval dataset needs the goal-set
            # TARGET, for precisely the reason the WP-D block below needs
            # `bev_spec`, and it is here because arm `C_w0p05` of the
            # 2026-09-11 weight sweep DIED ON IT: 400/400 training steps
            # done, checkpoint written, then the step-400 eval raised the
            # REFUSE-DO-NOT-SKIP guard and the process exited 1 with no eval
            # row. `SystemExit` derives from BaseException, so the eval
            # block's `except Exception` cannot catch it
            # (`issubclass(SystemExit, Exception)` is False).
            # ⛔ WHY IT MATTERS BEYOND A DIAGNOSTIC: refcv6 arm C is
            # refcv5-v2's argv plus `--w-tac-goal`, and that argv carries
            # `--eval-every 500`. Arm C would have died at STEP 500 OF
            # 40,284.
            # ⛔ THE pos_weight AND class_mask ARE NOT REFITTED HERE. They
            # live on the MODEL and were fitted on the TRAIN split; fitting
            # them on the eval split would be a statistic fitted on the data
            # it is about to score. Only the TARGET is wired.
            # ⚠️ On an eval split whose records carry no goal token this is
            # still correct and still not a crash: every cell is IGNORE_W, so
            # `tac_goal_loss` returns a real 0.0 WITH `n_supervised == 0`
            # saying why — the documented control, never a silent skip.
            # ⭐ refcv6 §2b: the EVAL split is fed the same channel, or the
            # in-training eval would run a model whose condition is missing an
            # input it was trained with.
            e_ds.ego_history = bool(getattr(args, "ego_history", False))
            # ⭐ refcv6 §4 rides the SAME eval wiring, and for the SAME reason
            # arm C_w0p05 died on it: the in-training eval calls
            # `compute_losses_v3`, whose refcv6 block REFUSES (SystemExit, so
            # `except Exception` cannot catch it) when `--w-tac-v6 > 0` meets
            # a batch with no `tac_goal_y`. Without this line a refcv6
            # tactical arm dies at the FIRST eval with the training compute
            # already paid for.
            if (float(getattr(args, "w_tac_goal", 0.0) or 0.0) > 0.0
                    or float(getattr(args, "w_tac_v6", 0.0) or 0.0) > 0.0):
                e_ds.tac_goal_targets = True
                e_ds.tac_goal_negatives = str(getattr(
                    args, "tac_goal_negatives", "measured"))
            # ⭐ E16 — the eval dataset is fed the SAME ceiling channel.
            # ⚠️ No normaliser is handed down and none is fitted: the ladder
            # is PINNED road law, not a statistic of the split, which is
            # exactly why quantizing to a fitted quantile was rejected.
            if getattr(args, "max_speed_input", False):
                eval_max_speed_stats = e_ds.enable_max_speed(
                    e_man, str(getattr(args, "max_speed_mode",
                                       msi.DEFAULT_MODE)))
            # ⛔ refcv6 §5 on the EVAL split, with its OWN sidecar. An eval
            # dataset fed no ceiling would run a model whose condition is
            # missing an input it was TRAINED with — a train/eval channel
            # mismatch that reads as a capability drop.
            # ⚠️ `--speed-max-sidecar-v6-eval` falls back to the train
            # sidecar ONLY when the eval labels are the same blob; otherwise
            # the reader's md5 guard refuses, which is the intended outcome.
            if getattr(args, "max_speed_input_v6", False):
                eval_max_speed_v6_stats = e_ds.enable_max_speed_v6(
                    str(getattr(args, "speed_max_sidecar_v6_eval", None)
                        or args.speed_max_sidecar_v6), e_man)
        # the eval sees the SAME label source as training, with its OWN
        # episode restriction -- and the SAME pad, so the two blocks are
        # directly comparable rather than two different paddings.
        if getattr(args, "agent_join", None):
            from train_p8_occupancy import JoinFileReader as _JFR
            _e_rd = _JFR(args.agent_join,
                         episode_ids={int(e.episode_id) for e in e_eps},
                         with_rates=not bool(getattr(
                             args, "agent_join_no_rates", False)))
            # ⭐ WP-D: the EVAL dataset is handed the SAME BEV spec as the
            # train dataset -- and NOT for symmetry.  ⛔ MEASURED 2026-09-07
            # by reading source before the first GPU-hour: without this line
            # `e_ds.bev_spec` stays None, the eval batch carries no `bev_occ`,
            # and `compute_losses_v3`'s REFUSE-DO-NOT-SKIP guard raises
            # **SystemExit** -- which derives from BaseException, so the eval
            # block's `except Exception` CANNOT catch it (verified:
            # `issubclass(SystemExit, Exception)` is False).  The run would
            # die at the FIRST eval with the compute already paid for -- the
            # `t1_eval` class where an analysis-time failure destroys a
            # completed rollout.
            # ⚠️ On an eval cache with NO join coverage this is still correct
            # and still not a crash: every frame is NO_LABEL, so the term is
            # the documented control -- loss exactly 0.0 with
            # `bev_n_supervised == 0` saying why, never a silent skip.
            if str(getattr(args, "bev_aux", "off")) != "off":
                e_ds.bev_spec = _bev_aux.PolarBEVSpec(
                    n_az=int(cfg.core.encoder.grid_shape[1]),
                    n_rng=int(getattr(args, "bev_aux_rng", 24)),
                    r_max_m=float(getattr(args, "bev_aux_rmax", 60.0)))
                e_ds.bev_occlusion = str(getattr(args, "bev_aux_occlusion",
                                                 "mask"))
            eval_agent_stats = e_ds.enable_agent_join(
                _e_rd, pad=int(ds.agent_pad),
                allow_legacy_ids=bool(getattr(
                    args, "agent_join_allow_legacy_ids", False)))
        # ---- refcv6 §2/§6 on the HELD-OUT split ------------------------- #
        # ⛔ Its own coverage census, never the train split's. The eval cache
        # is a different clip set, and quoting the train number for it is the
        # scope error this programme has measured in six other costumes.
        if getattr(args, "map_gt_root", None) \
                or getattr(args, "join3d", None):
            _e_clip, _e_ns = _clip_table_for_caches([args.eval_cache])
            if getattr(args, "map_gt_root", None):
                eval_map_stats = e_ds.enable_map_gt(
                    _perception_targets.MapGTStore(
                        Path(args.map_gt_root),
                        max_open=int(getattr(args, "map_lru", 4) or 4)),
                    _e_clip, _e_ns,
                    min_coverage=getattr(args, "map_min_coverage", None))
            if getattr(args, "join3d", None) and e_ds.agent_join is not None:
                if e_ds.map_clip_of_ep is None:
                    e_ds.map_clip_of_ep, e_ds.map_n_stack = _e_clip, _e_ns
                eval_join3d_stats = e_ds.enable_join3d(
                    _agent_cuboid.open_join3d(
                        args.join3d, clips=set(_e_clip.values())))
        # FIXED **and REPRESENTATIVE** windows.
        # ⛔ shuffle=False ALONE IS A TRAP, and it bit this eval on its first
        # run: taking the first N windows takes them from the START of the
        # first episodes (window NOW ~0.7-2 s), and every v7.2 record is
        # anchored at t0 = 8.0 s with a +-2 s band — so `eval_lat_tac` read
        # exactly 0.0000, an eval that silently never measured the tactical
        # heads while reporting a clean number. ⇒ take a DETERMINISTIC RANDOM
        # subset (seeded, computed once) and iterate it in a fixed order: still
        # the same windows at every step, but drawn from the whole corpus.
        g_ev = torch.Generator().manual_seed(12345)
        n_need = args.eval_batches * args.batch
        perm = torch.randperm(len(e_ds), generator=g_ev)[:n_need].tolist()
        eval_dl = torch.utils.data.DataLoader(
            torch.utils.data.Subset(e_ds, perm),
            batch_size=args.batch, shuffle=False,
            num_workers=0, drop_last=True)
        print(f"[v3] held-out eval: {len(e_eps)} episodes -> {len(e_ds)} "
              f"windows, {args.eval_batches} fixed batches every "
              f"{args.eval_every} steps", flush=True)

    # ⛔ IN-FLIGHT BATCHES ARE THE SHM COST, NOT THE CLIP CACHE. MEASURED
    # 2026-09-02: two launches died on `Bus error ... out of shared memory`
    # even with file_system sharing. One collated batch is
    #   20 x window 8 x 9ch x 256 x 640 x 4B ~= 940 MB
    # so workers x prefetch_factor batches are in flight at once: 4 x 2 = 8
    # ~= 8 GB against a 24 GB /dev/shm, plus the eval loader's own. The knob
    # that actually bounds this is prefetch_factor, and it had been left at
    # torch's default of 2.
    pf = {"prefetch_factor": args.prefetch_factor} if args.workers > 0 else {}
    if args.workers > 0:
        print(f"[v3] loader: workers={args.workers} prefetch={args.prefetch_factor}"
              f" -> <={args.workers * args.prefetch_factor} batches in flight",
              flush=True)
    # ⭐ The in-flight bytes are COMPUTED and PRINTED (and stamped into
    # config.json) so the shm budget is checkable from train.log: frames +
    # future_frames per collated batch = B x (window + max_horizon) x C x H x W
    # x (1 B uint8 | 4 B fp32). At the live shape (20 x 28 x 9 x 256 x 640)
    # that is 3.30 GB fp32 / 0.83 GB uint8 per batch — the two /dev/shm
    # segments MEASURED per worker on 2026-09-02 (2,359,296,064 B and
    # 943,718,464 B) are exactly these two tensors plus a 64 B header each.
    _h, _w = cfg.core.encoder.image_hw()
    frame_batch_bytes = (args.batch * (cfg.core.window + kw["max_horizon"])
                         * cfg.core.encoder.in_channels * _h * _w
                         * (1 if u8 else 4))
    n_flight = args.workers * args.prefetch_factor if args.workers > 0 else 1
    print(f"[v3] u8_batches={u8}: frames+future_frames = "
          f"{frame_batch_bytes / 1e9:.3f} GB per collated batch "
          f"({'uint8' if u8 else 'float32'}), <= {n_flight} in flight "
          f"-> ~{n_flight * frame_batch_bytes / 1e9:.2f} GB", flush=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        **pf,
        drop_last=True, persistent_workers=args.workers > 0)

    withheld_stamp = _apply_withheld_bank(model, args, eps, device)

    opt = build_optimizer(model, args)
    sched = lambda s: (s + 1) / max(1, args.warmup) if s < args.warmup else \
        0.5 * (1.0 + math.cos(math.pi * (s - args.warmup)
                              / max(1, args.steps - args.warmup)))   # noqa: E731

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    ck = out_dir / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])          # strict — resume exact
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        print(f"[v3] resumed {args.arm} at step {step}")
    # ⛔ M18: the record is CHECKED before it is written. A knob that does not
    # reach config.json refuses the run here, not in an audit months later.
    _seams = _seam_stamp(cfg, args)
    _seams["agent_rig_camera"].update(
        assert_rig_camera_covers(model, ds, args))
    _seams["agent_ground_prior_probe"] = _ground_probe
    # ⭐ D-TACGOAL-1 / D-ROLL-1h: the record states the FACT, read off the
    # constructed module -- never re-derived from the config, which is
    # intent. Same idiom as `agent_rig_camera` two lines up.
    _seams["tac_goal_tok_head"]["built"] = (
        getattr(model, "tac_goal_tok_head", None) is not None)
    _seams["max_speed_input"]["built"] = (
        getattr(model, "max_speed_cond", None) is not None)
    # ⛔ refcv6 §4/§5: read off the BUILT OBJECT, never re-derived from cfg —
    # the `param_breakdown_v3` idiom, and the reason is the `--image-hw`
    # post-mortem: a second copy of the build condition is how a record drifts
    # from the model it claims to describe.
    _seams["tac_decoder_v6"]["built"] = (
        getattr(model, "tac_decoder_v6", None) is not None)
    if getattr(model, "tac_decoder_v6", None) is not None:
        # ⭐ THE PARAMETER LEDGER, IN THE RUN RECORD. The whole defect was
        # 2,262,020 parameters nobody could see from the artifact.
        _seams["tac_decoder_v6"]["param_breakdown"] = \
            model.tac_decoder_v6.param_breakdown()
        _seams["tac_decoder_v6"]["provenance"] = \
            model.tac_decoder_v6.provenance()
        _seams["tac_decoder_v6"]["loss_weights"] = \
            v6tac.TacticalLossWeights().to_dict()
    _seams["max_speed_onehot_v6"]["built"] = (
        getattr(model, "max_speed_1h_v6", None) is not None)
    assert_knobs_stamped(args, _seams)
    # ⛔ ...and the record is checked against the MODEL, not only against
    # the config that produced it. A config is intent; only the built
    # modules are fact. See `assert_seams_are_built`.
    assert_seams_are_built(model, _seams)
    _run_config = {
        "arm": args.arm, "seed": args.seed, "argv": sys.argv[1:],
        # ⛔ `argv` records what was TYPED and the seam stamps record what
        # was BUILT; neither says whether each weight's loss term is
        # actually reached. A run record that does not carry its effective
        # weights cannot be audited afterwards.
        "effective_weights": effective_weights_stamp_v3(args, echo=True),
        "registered_delta": {k: [repr(a), repr(b)]
                             for k, (a, b) in delta.items()},
        "param_breakdown": v3.param_breakdown_v3(model),
        "goal_provenance": refc.RefCModel.goal_provenance(),
        "provenance_roles": model.provenance_roles(),   # v4: INSTANCE call —
        # the refused edge differs between v3 (v0) and v4 (future ego),
        # so a static declaration could not tell the two arms apart, which
        # is exactly the "declared provenance" failure the audit exists for.
        "horizons": list(cfg.core.trajectory.horizons),
        # ⭐⭐ THE SELECTION / VOCABULARY STAMP. MEASURED 2026-09-04: refcv3's
        # config.json recorded NEITHER, so its own artifact cannot answer
        # "which anchor vocabulary?" or "how wide was the reach band?" — and
        # the answer to the first was `default_anchors` (SYNTHETIC), whose
        # oracle-in-vocabulary ADE 0-2 s is 1.0882 m against 0.3796 m for the
        # data-driven set and 0.6843 m for a single straight line. A silent
        # fallback to the synthetic default is exactly what an unstamped run
        # cannot rule out afterwards, so it is stamped here by CONTENT
        # (sha256 + shape), not by the presence of a path.
        "anchors": _anchor_stamp(
            getattr(args, "anchors", None), model.core.decoder.anchors,
            model.core.decoder.anchor_controls
            if getattr(model.core.decoder, "anchor_v0_cond", False) else None,
            getattr(args, "anchor_control_units", None) or "kappa",
            art=art),
        # ⭐⭐ THE SEAM STAMP (2026-09-05). MEASURED 2026-09-04: none of the
        # hierarchy-seam booleans were in config.json at any nesting level,
        # so the live arm could not rebuild its own model config from its
        # own record. `_seam_stamp` names each one.
        "seams": _seams,
        # ⭐ H-EGO-LIT-4: the withheld-row bank policy, with the random
        # control's marginal on record.
        "withheld_bank": withheld_stamp,
        "selection": {
            "sel_reach_clamp": bool(cfg.core.sel_reach_clamp),
            "sel_accel_max": float(cfg.core.sel_accel_max),
            # ---- refcv5 P14 --------------------------------------------- #
            # ⛔ STAMPED BECAUSE THE ABSENCE OF THIS ROW IS HOW THE DEFECT
            # SURVIVED. refcv5's `config.json` recorded `sampler: ddim` and
            # `sampler_train_t_max: 50` and said NOTHING about what ranked the
            # fan, so a reader had no way to tell a sampled-and-ranked arm from
            # a sampled-then-ranked-by-a-stale-score arm. `sampler_ranks_the_fan`
            # is the one key that separates them and it lived only in per-batch
            # telemetry, which nobody re-reads.
            "sel_refined": bool(getattr(cfg.core, "sel_refined", False)),
            "sel_score_emitted": bool(getattr(cfg.core, "sel_score_emitted",
                                              False)),
            "sel_score_emitted_t": int(getattr(cfg.core,
                                               "sel_score_emitted_t", -1)),
            "sel_score_emitted_t_source": str(
                getattr(args, "sel_score_emitted_t_source", "operator")),
            # The single fact a reader needs, precomputed, in the CONFIG.
            "sampler_ranks_the_fan": bool(
                getattr(cfg.core, "sel_refined", False)
                and getattr(cfg.core, "sel_score_emitted", False)),
            "p14_note": (
                "sampler_ranks_the_fan is False unless BOTH sel_refined and "
                "sel_score_emitted are set. False means the emitted fan was "
                "ranked by a score computed before any sample was drawn. "
                "MEASURED ceiling on the banked fan (n=881/40 ep, paired "
                "episode-cluster bootstrap): base 0.4728 -> 0.1914 m ADE "
                "(+0.2813, CI [+0.2127,+0.3543]), a >2x-better trajectory in "
                "the fan on 41.09 % of windows; xl +0.3075 m, 45.40 %."),
            "horizon_s": float(cfg.core.selection().horizon_s),
            "band_ms": float(cfg.core.sel_accel_max
                             * cfg.core.selection().horizon_s),
            "note": ("horizon_s is DERIVED from max(horizons); the 72.08 % / "
                     "77.28 % kill statistics quoted elsewhere are 2 s numbers "
                     "and do NOT hold here (re-derived 2026-09-04)."),
        },
        "goal_tau_steps": list(cfg.goal_tau_steps),
        "admission_sigma_m": cfg.admission_sigma_m,
        # ⭐⭐ THE EGO STAMP (v4). refcv3 recorded NEITHER ego knob, so a
        # finished run cannot answer "was the speed masked?" from its own
        # artifact - and that answer changes what every longitudinal number
        # MEANS. A run's config.json is the only durable record it has.
        "ego": {
            "ego_state_inject": bool(getattr(cfg, "ego_state_inject", False)),
            "echo_base": bool(getattr(cfg, "echo_base", False)),
            "ego_valid_channel": bool(cfg.core.ego_valid_channel),
            "ego_dropout": float(cfg.core.ego_dropout),
            "tactical_speed_input": bool(cfg.core.tactical_speed_input),
            "channels": (["v0", "a_long", "yaw_rate", "curvature", "keep"]
                         if getattr(cfg, "ego_state_inject", False)
                         else ["v0"]),
            "derivation": ("t0 = last OBSERVED frame; a_long = corpus ax "
                           "(actions[:,-1,1]); curvature = tan(steer)/2.9 "
                           "(actions[:,-1,0]); yaw_rate = v0*curvature. NO "
                           "finite differencing, NO future read."),
            "wheelbase_mode": "const2p9",
        },
        # ⭐ THE RUNG, STAMPED. `--size` decides ~4x of the parameter count and
        # was recorded NOWHERE, so a finished run could not say which rung it
        # was — and with a VALIDATION-RIG rung now reachable by name, an
        # unstamped run is one whose numbers cannot be told apart from a
        # registered arm's. `rig_rung: true` is the refusal token: nothing
        # carrying it may enter MODEL_REGISTRY.md.
        "size": args.size,
        "rig_rung": bool(args.size in v3.V3_RIG_SIZES),
        # ⛔ the deliberate-regression stamp. A run whose frames were
        # ablated is a GATE CONTROL; its numbers are not the model's and the
        # artifact must say so on its own.
        "ablate_frames": bool(args.ablate_frames),
        # the 2026-09-01 B1-readiness fields — the config.json is the ONLY
        # durable record of what corpus/geometry/vocab a run actually used.
        "image_hw": list(cfg.core.encoder.image_hw()),
        "tac_vocab_version": cfg.tac_vocab_version,
        "v2_cache": args.v2_cache, "require_parity": bool(args.require_parity),
        "v2_parity": v2_parity,
        # --u8-batches (2026-09-02): the in-flight dtype and its per-batch
        # frame bytes — a relaunch arm must be identifiable from its record.
        "u8_batches": u8,
        "frame_batch_bytes_est": frame_batch_bytes,
        # --nav-from-v7 (E-ARCH-NAVSRC-1): the nav SOURCE is stamped EITHER
        # way, so an arm is identifiable from its own artifacts
        # (C-NAV-SOURCE-DIVERGENCE). `v7_labels` is the label manifest —
        # md5 + the allow_oracle_nav stamp — None without --v7-labels.
        "nav_from_v7": nav_on,
        "nav_cmd_derivation": (NAV_FROM_V7_DERIVATION if nav_on
                               else NAV_V1_DERIVATION),
        "nav_from_v7_stats": ({"train": nav_stats, "eval": eval_nav_stats}
                              if nav_on else None),
        # ⭐⭐ D-TACGOAL: WHAT THE GOAL HEAD WAS ACTUALLY TRAINED ON.
        # The per-token census, the class mask with its REASONS, and the
        # split-fitted pos_weight. ⛔ None when the channel is off, so the
        # record distinguishes "not asked for" from "asked for and empty".
        "w_tac_goal": float(getattr(args, "w_tac_goal", 0.0) or 0.0),
        "tac_goal_stats": tac_goal_stats,
        # ⭐ D-GSTR-1 P3 (E13b): the nav ARGS channel. ⛔ A tensor carries its
        # UNITS or it is inadmissible — the raw slots are METRES and SECONDS,
        # the normaliser is fit-split-only, and `window_real_distance_frac`
        # is the number that decides whether the channel carries information
        # at all (69.97 % of train RECORDS present distance 0.0).
        "nav_args": bool(getattr(args, "nav_args", False)),
        "nav_args_stats": ({"train": nav_args_stats,
                            "eval": eval_nav_args_stats}
                           if getattr(args, "nav_args", False) else None),
        # ⭐⭐ E16 — the max-speed channel's own census, so an arm is
        # identifiable from its own artifacts. ⛔ `window_ceiling_frac` is
        # THE number that decides whether the channel carried information;
        # `provenance: ego-future` inside `meta` is the caveat any result
        # must be read against.
        "max_speed_input": bool(getattr(args, "max_speed_input", False)),
        "max_speed_mode": (str(getattr(args, "max_speed_mode",
                                       msi.DEFAULT_MODE))
                           if getattr(args, "max_speed_input", False)
                           else None),
        "max_speed_stats": ({"train": max_speed_stats,
                             "eval": eval_max_speed_stats}
                            if getattr(args, "max_speed_input", False)
                            else None),
        # ⭐⭐ E16 — THE PROVENANCE STAMP, and the PI's binding condition on
        # using this channel at all (2026-09-10). It names the SOURCE FIELD,
        # the WINDOW and the word `oracle`, exactly as `nav_cmd_derivation`
        # above does for the nav token — the same class of signal, an INPUT
        # simulating a vehicle service, never a training signal.
        # ⛔ `_assert_speed_max_stamp` below REFUSES TO START a --max-speed-input
        # run that would reach this point without it, and refuses the mirror
        # case (stamp present, flag off). ⛔ NO CAPABILITY CLAIM MAY BE
        # CREDITED TO THIS CHANNEL WITHOUT THIS STRING BESIDE IT.
        "speed_max_derivation": (SPEED_MAX_DERIVATION
                                 if getattr(args, "max_speed_input", False)
                                 else None),
        # ⚠️ A synthetic corpus whose episode ids were stamped from REAL label
        # clip_ids so the v7/v8 join could hit. Frames are still synthetic —
        # recorded so such a run can never be mistaken for a trained arm.
        "synth_clip_ids_from_labels": synth_clip_ids_from_labels,
        "v7_labels": v7_manifest,
        # ⭐ refcv5 WP-6: WHICH labels the detector saw, and HOW MANY windows
        # actually carried one. A run that stamps `w_agent > 0` without this
        # cannot say whether its detector was supervised on 100 % or 3 % of
        # its windows -- and those are different experiments.
        "agent_join": str(getattr(args, "agent_join", None) or "") or None,
        # ⛔ M18: WHICH artifact the join's digest covers, and whether this run
        # actually checked it. A hash without its artifact scope is a number,
        # not a verification, so the run records the scope it verified under
        # -- or the reason it verified nothing.
        "agent_join_digest": join_digest,
        "agent_join_stats": ({"train": agent_stats, "eval": eval_agent_stats}
                             if agent_stats is not None else None),
        # ⭐⭐ refcv6 §2/§6. ⛔ `None` when both weights are 0.0 — the key is
        # absent-as-null rather than a zeroed block, so a default run's
        # config.json differs from the pre-branch trainer's by NOTHING a
        # comparison can act on, and a perception run can never be mistaken
        # for one. It carries the COVERAGE, not merely the paths: a run that
        # stamps `w_map > 0` without it cannot say whether the trunk saw the
        # map on 100 % or 3 % of its windows, and those are different
        # experiments (the `agent_join_stats` argument, one head over).
        "refcv6_perception": (
            None if perception_stamp is None else
            {**perception_stamp,
             "map_gt_stats": {"train": map_stats, "eval": eval_map_stats},
             "join3d_stats": {"train": join3d_stats,
                              "eval": eval_join3d_stats}}),
        # ⭐⭐ refcv6 §4 — the tactical layer's own block, `None` when the
        # weight is 0.0 for the same absent-as-null reason as the perception
        # block above: a default run's config.json must differ from the
        # pre-wiring trainer's by NOTHING a comparison can act on.
        # ⛔ IT CARRIES THE LABEL LIMITS, not merely the weight. A tactical
        # number quoted without them is unreadable: 9 of 22 tokens sit ON the
        # pos_weight cap (for those the CAP, not the data, sets the weight)
        # and 10 of 22 sit under the n=200 scoreability floor.
        "refcv6_tactical": (
            None if float(getattr(args, "w_tac_v6", 0.0) or 0.0) <= 0.0 else
            {"w_tac_v6": float(args.w_tac_v6),
             "loss_weights": v6tac.TacticalLossWeights().to_dict(),
             "valid_threshold": float(
                 getattr(args, "tac_decoder_valid_threshold", 0.5)),
             "graft_behaviour_sel": bool(
                 getattr(args, "graft_behaviour_sel", False)),
             "label_limits": tac_goal_stats,
             "scene_sources_live": ["agent"],
             "scene_sources_blocked": ["bev"],
             "⛔": ("AGENT-ONLY. The PI asked for 'the agent and the map'; "
                    "no BEV token reaches the behaviour decoder on this path "
                    "(refc_v3.RefCV3Model.forward does not pass "
                    "`bev_tokens=` to self.core). Any result from this arm is "
                    "about the AGENT half and must say so.")}),
        # ⭐⭐ refcv6 §5 — the 4-way one-hot set-speed census.
        "refcv6_max_speed": ({"train": max_speed_v6_stats,
                              "eval": eval_max_speed_v6_stats}
                             if max_speed_v6_stats is not None else None),
        # ⛔⛔ THE §5 STAMP IS A PRECONDITION, NOT A FIELD — and it keys on
        # `speed_max_derivation_v6`, so E16's stamp cannot satisfy this guard
        # or vice versa. Two ladders sharing one key is how an arm ends up
        # describing a ceiling it never fed.
        "speed_max_derivation_v6": (
            v6ms.SPEED_MAX_DERIVATION_V6
            if getattr(args, "max_speed_input_v6", False) else None),
    }
    # ⛔⛔ E16 — THE STAMP IS A PRECONDITION, NOT A FIELD. A --max-speed-input
    # run whose record would not declare the channel's ego-future provenance
    # REFUSES TO START here, before config.json is written and before a single
    # step is taken. Pinned with a deliberate-regression arm in
    # stack/tests/test_speed_max_derivation_stamp.py.
    _assert_speed_max_stamp(_run_config, args)
    # ⛔⛔ refcv6 §5 — the SAME precondition for the 4-way channel, and it is a
    # SEPARATE call on a SEPARATE key on purpose. The PI authorised an
    # EGO-FUTURE input on the axis that owns most of the oracle gap; that is
    # defensible exactly as long as every artifact says so. Both directions
    # refuse: a channel with no stamp, and a control that carries one.
    v6ms.assert_speed_max_stamp_v6(
        _run_config, bool(getattr(args, "max_speed_input_v6", False)))
    (out_dir / "config.json").write_text(json.dumps(_run_config, indent=1),
                                         encoding="utf-8")

    log = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")
    _gp_names = [t.strip() for t in
                 str(getattr(args, "grad_probe_modules", "") or "").split(",")
                 if t.strip()]
    _gp_row = {}
    if _gp_names:
        print("[v3] grad probe ON for %d module(s): %s"
              % (len(_gp_names), ", ".join(_gp_names)), flush=True)
    # ---- SPEC_REFCV6_V2 §6: the gradient-conflict detector ----------------- #
    # ⛔ ON BY DEFAULT FOR refcv6 ARMS, off otherwise. `for_model` returns None
    # when the flag is off, so the off path constructs nothing at all and the
    # step below is the pre-detector step, byte for byte.
    # ⛔ The refcv6 block is read off the CONFIG THE MODEL WAS BUILT FROM, and
    # `None` (the baseline) stays distinguishable from a live block -- the same
    # discipline `_refcv6_flags_of` states for the F-flags, and the reason
    # `config.json` stamps `refcv6: null` instead of omitting the key.
    _cd_aux = _conflict_aux_weights(model)
    _cd_on = _gcf.enabled_for_arm(
        getattr(getattr(cfg, "core", cfg).decoder, "refcv6", None),
        override=_conflict_override(args), aux_present=bool(_cd_aux))
    _cd_cfg = _gcf.ConflictConfig(
        enabled=_cd_on,
        # ⚠️ RESOLVED from the model, never assumed: `RefCV3Model` wraps the
        # trunk at `core.encoder.` while `RefCModel` holds it at `encoder.`, and
        # a prefix that matches nothing would log NaN for the whole run.
        trunk_prefixes=(_gcf.resolve_trunk_prefixes(model) if _cd_on
                        else _gcf.TRUNK_PREFIX_CANDIDATES),
        mode=str(getattr(args, "conflict_mode", _gcf.MODE_PROBE)),
        every=max(1, int(getattr(args, "conflict_every", 1))))
    _cd = _gcf.GradientConflictDetector.for_model(model, _cd_cfg)
    _cd_row, _cd_checked = {}, False
    if _cd is not None:
        _p = _cd.provenance()
        print("[v3] conflict detector ON (%s, plan side=%s, every=%d): "
              "theta_trunk = %d tensors / %d params in %d groups (%s); "
              "aux terms = %s"
              % (_cd_cfg.mode, _gcf.PLAN_SIDE[_cd_cfg.mode], _cd_cfg.every,
                 _p["n_trunk_tensors"], _p["n_trunk_params"], len(_p["groups"]),
                 ", ".join(sorted(_p["groups"])),
                 ", ".join(sorted(_cd_aux))), flush=True)
        # ⚠️ SAY THE COST OUT LOUD, AT START. The prereg budgets this at "one
        # extra backward over the trunk -- no extra GPU-day"; MEASURED on the
        # dev box it is +72..104 % of step time in `probe` and +26..62 % in
        # `subtract`, RISING with trunk size. A run that discovers that from its
        # wall clock at hour 6 has already paid for it.
        print("[v3] ⚠️ conflict detector MEASURED overhead (dev-box CPU, batch "
              "2, 77k-6.9M trunks): probe +72..104%%, subtract +26..62%% of "
              "step time, rising with trunk size. Divide by --conflict-every "
              "(%d) to amortise; the GPU figure is UNVERIFIED."
              % _cd_cfg.every, flush=True)
        # ⭐ the detector's own config goes into the RUN RECORD, so a
        # `cd_*`-free metrics.jsonl and a run that never enabled it are
        # distinguishable after the fact -- the `refcv6: null` discipline.
        _run_config["conflict_detector"] = _cd_cfg.as_dict()
        _run_config["conflict_detector"]["aux_terms"] = sorted(_cd_aux)
        (out_dir / "config.json").write_text(json.dumps(_run_config, indent=1),
                                             encoding="utf-8")
    t0, model = time.time(), model.train()
    it = iter(dl)
    _bev_parity_checked = False        # WP-D: the E-DEC-18b gate fires once
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        # ⭐ H-EGO-LIT-4: the withheld bank goes live after the warm-up. Set
        # EVERY step (not once) so a resumed run lands in the right regime,
        # and logged so the record says which bank each step trained on.
        _wb_active = step >= args.withheld_bank_warmup
        model.core.decoder.anchor_withheld_bank = (
            args.withheld_bank if _wb_active else "fixed")
        losses = compute_losses_v3(model, batch, device, mode=args.mode,
                                   ablate_frames=args.ablate_frames)
        losses["withheld_bank_active"] = float(_wb_active)
        # ⛔⛔ WP-D: THE LOSS-SCALE PARITY GATE, AT THE FIRST SUPERVISED STEP.
        # `E-DEC-18b` MEASURED the sibling PSG term sitting 10-30x above the
        # objective it was meant to support, at EVERY weight tested, and
        # destroying the encoder — and that was visible only AFTER the GPU-days
        # were spent. This is that check, moved to before them. It fires ONCE,
        # on the first step that actually had supervised cells (step 0 can be
        # all-NO_LABEL, and a ratio computed on a 0.0 term would pass the gate
        # while measuring nothing — the same vacuity the `n` in every log row
        # exists to expose). ⛔ It RAISES SystemExit; a run whose aux term
        # already dwarfs its objective should not spend the card.
        if (not _bev_parity_checked
                and float(losses.get("bev_n_supervised", 0.0)) > 0.0
                and "bev" in losses):
            _bev_parity = _refc_bev_aux.assert_loss_parity(
                float(getattr(model, "_w_bev_aux", 0.0)),
                float(losses["bev"].detach()), float(losses["traj"].detach()))
            _bev_parity_checked = True
            print("[v3] WP-D loss parity OK at step %d: "
                  "w*bev/traj = %.4f in %s" % (step, _bev_parity["ratio"],
                                               _bev_parity["band"]), flush=True)
        opt.zero_grad(set_to_none=True)
        # ---- SPEC_REFCV6_V2 §6: cos(g_traj, g_aux) on the shared trunk ----- #
        # ⛔ BEFORE the backward, because the two gradients must be SEPARATE;
        # ⛔ LOGGED AND REPORTED, NEVER A STOP RULE -- nothing below halts,
        # clips, re-weights or projects anything. `measure` uses
        # `torch.autograd.grad`, which does NOT accumulate into `.grad`, so the
        # `backward()` that follows writes exactly the bytes it wrote before the
        # detector existed (tests/test_refcv6_grad_conflict.py proves the step
        # is bit-identical with the flag ON as well as off).
        if _cd is not None:
            # ⛔ cleared EVERY step: with `--conflict-every > 1` a retained row
            # would re-log the previous step's reading under this step's number,
            # which is a fabricated measurement, not a stale one.
            _cd_row = {}
            _cd_lt, _cd_la = _conflict_terms(model, losses)
            if _cd_lt is not None and _cd_la is not None:
                if not _cd_checked:
                    # ⛔ THE PROBE REFUSES TO RUN IF ANY CONTROL MISSES
                    # (PREREG_BEV_CAPACITY_COMPETITION.md §4). Raising here
                    # costs one step; a run whose probe cannot read +1 costs the
                    # whole card and answers nothing.
                    _cd_ctl = _cd.self_check(_cd_lt, _cd_la)
                    _cd_checked = True
                    print("[v3] conflict controls OK: cos(g,g)=%r cos(g,-g)=%r "
                          "detached cos=%r conflict=%r |g_aux|=%r"
                          % (_cd_ctl.self_cos, _cd_ctl.negated_cos,
                             _cd_ctl.detached_cos, _cd_ctl.detached_conflict,
                             _cd_ctl.detached_norm_aux), flush=True)
                    log.write(json.dumps({"step": step,
                                          "conflict_controls":
                                              _cd_ctl.as_dict(),
                                          "conflict_provenance":
                                              _cd.provenance()}) + "\n")
                    log.flush()
                if step % _cd_cfg.every == 0 and _cd_cfg.mode != _gcf.MODE_SUBTRACT:
                    _cd_row = _cd.measure(_cd_lt, _cd_la, step=step).row()
        # ⚠️ `subtract` mode reads `.grad`, so it must run AFTER the backward
        # (and the backward must retain the graph for its one aux pass) and
        # BEFORE `clip_grad_norm_`, which rescales `.grad` in place.
        # (`_cd_la` is only reached when `_cd is not None`, where the tuple
        #  unpack above has always bound it.)
        _cd_sub = (_cd is not None and _cd_cfg.mode == _gcf.MODE_SUBTRACT
                   and _cd_la is not None and step % _cd_cfg.every == 0)
        if _cd_sub:
            losses["loss"].backward(retain_graph=True)
        else:
            losses["loss"].backward()
        if _cd_sub:
            _cd_row = _cd.measure_after_backward(_cd_la, step=step).row()
        # D-TACGOAL-2: read the gradient HERE -- after backward,
        # before the global clip rescales it, and before the next
        # zero_grad wipes it.
        _gp_row = _grad_probe_row(
            model, _gp_names,
            log_every_hit=bool(_gp_names) and (
                (step + 1) % args.log_every == 0
                or (step + 1) >= args.steps))
        # ⛔⛔ refcv6 §2/§6 — PER-HEAD GRADIENT REACH, READ, NEVER ASSUMED.
        # HERE for the same reason `_grad_probe_row` is: after `backward`,
        # before the global clip rescales it, before the next `zero_grad`
        # wipes it. ⚠️ The failure it exists for is `tac_goal_tok_head`: 11,286
        # parameters with `grad_abs_sum` EXACTLY 0 for all 40,284 steps --
        # parsed, stamped, reaching nothing, and invisible in every loss value.
        # A head whose `_ga` row is 0.0 while its loss is finite is that class.
        # ⭐⭐ WIDENED BY PI RULING 2026-09-17 R3. The trunk is now optimised
        # FOUR ways — planner, map head, box head AND the tactical behaviour
        # decoder — and the ruling's own named mitigation for the attribution
        # it costs is per-head reach plus the conflict detector. Gating this on
        # `_perception` alone would leave a tactical arm with NO reach row at
        # all, which is the `tac_goal_tok_head` blind spot one head over.
        # ⛔ Still nothing on an arm with neither seam: `_pr_row` stays empty,
        # no `ga_*` key enters metrics.jsonl, and bit-identity is untouched.
        _pr_row = {}
        _ga_on = (getattr(model, "_perception", None) is not None
                  or getattr(model, "tac_decoder_v6", None) is not None)
        if _ga_on and (
                step % max(1, args.log_every) == 0 or step + 1 >= args.steps):
            for _pk, _pv in _perc.grad_reach_report(model).items():
                _pr_row[f"ga_{_pk}"] = _pv["grad_abs_sum"]
                _pr_row[f"ga_{_pk}_n"] = _pv["n_params_with_grad"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        step += 1
        if step % args.log_every == 0 or step == args.steps:
            # ⚠️ The tensor filter silently DROPPED every plain-float
            # diagnostic — `tac_label_v7`, `tac_label_rows`, `nav_injected`
            # were emitted by the loss and never reached the log, so "did the
            # v7.2 labels actually train?" was answerable only by INFERRING it
            # from the CE magnitude (lat 1.22 ~ ln 3 vs lat_tac 2.13 ~ ln 8).
            # A diagnostic that does not survive to the log is not a
            # diagnostic. Scalars now pass through as themselves.
            row = {k: (round(float(v.detach()), 5) if torch.is_tensor(v)
                       else round(float(v), 5))
                   for k, v in losses.items()
                   if (torch.is_tensor(v) and v.ndim == 0)
                   or isinstance(v, (int, float, bool))}
            row.update(step=step, elapsed_s=round(time.time() - t0, 1),
                       lr=opt.param_groups[0]["lr"])
            # AFTER the rounding comprehension above, deliberately:
            # a real 1e-8 gradient rounded to 5 dp reads 0.0, which
            # is the exact signature of the defect this measures.
            if _gp_row:
                row.update(_gp_row)
            # AFTER the rounding comprehension, same reason: a real 1e-8
            # gradient rounded to 5 dp reads 0.0, which is EXACTLY the
            # signature of the defect this measures.
            if _pr_row:
                row.update(_pr_row)
            # ⛔ AFTER the rounding comprehension, for the same reason as
            # `_gp_row`: a conflict of -1e-8 rounded to 5 dp reads 0.0, which is
            # "no conflict" -- the exact misreading this instrument exists to
            # prevent. With the detector off `_cd_row` is `{}` and the log
            # schema is identical to the pre-detector trainer's.
            if _cd_row:
                row.update(_cd_row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[v3:{args.arm}] step {step} "
                  f"loss {row['loss']:.4f} traj {row['traj']:.4f}")
        # ---- checkpoint FIRST, eval second (C-REFCV3-EVAL-DEATH) ----------
        # ⛔ THE ORDER OF THESE TWO BLOCKS IS LOAD-BEARING. MEASURED 2026-09-02:
        # refcv3 died SILENTLY at --eval-every 500 boundaries (steps 2,000 /
        # 4,500 / 6,000 / 10,500 / 17,000; empty stderr): the pod's memory
        # cgroup (v1, 50 GB, oom_kill 24, max_usage == limit) SIGKILLs the
        # trainer while the in-training eval decodes its 20-window batches in
        # the MAIN process on top of the six workers' in-flight fp32 batches.
        # The eval used to run BEFORE the save, so a death inside it lost every
        # step since the previous boundary -- the 17,000 death left ckpt.pt at
        # 16,500. Saving first bounds the cost of an eval death to ZERO
        # training steps. Training itself is untouched (same steps, same
        # losses; the eval still runs at the same point), and the checkpoint
        # is identical either way on EVERY parameter and optimizer tensor
        # (pinned by tests/test_refc_v3_save_before_eval.py) -- with ONE
        # measured exception that is a pre-existing defect, not an effect of
        # the order: compute_losses_v3 calls core.update_tactical_prior()
        # UNCONDITIONALLY, so the eval's held-out LABELS EMA into the
        # lat/lon_log_prior buffers (2 of 192 model tensors). The old order
        # saved the post-eval buffers, this order saves the pre-eval ones;
        # the leak itself is escalated (gate that call on model.training),
        # and the test flips strict-xfail -> pass the day it lands. The only
        # other observable change is that the step-N eval row / [v3:eval]
        # line now come AFTER the ckpt.pt write; metrics.jsonl's own row
        # order is unchanged (the save writes no row) and nothing consumes
        # the cross-file order. The memory fix itself (uint8 in-flight
        # batches) is a separate change.
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(), "step": step}, ck)
            # the one line that makes the order VERIFIABLE from train.log
            print(f"[v3:{args.arm}] ckpt step {step} -> {ck.name}",
                  flush=True)
        if step in MILESTONES:
            torch.save({"model": model.state_dict(), "step": step},
                       out_dir / f"ckpt_{step}.pt")
        # ---- held-out eval (PI 2026-09-02) --------------------------------
        # ⚠️ WHAT THIS IS AND IS NOT: an in-training MONITOR at T0 on the same
        # loss surface, over a FIXED set of held-out windows. It is NOT the
        # four-metric-family result and must never be quoted as one — the
        # binding families (longitudinal / lateral / tactical / strategic with
        # paired episode-cluster CIs) are a separate T1 job. What it buys is
        # the ability to see generalisation move DURING a 25 h run instead of
        # after it, and to catch train-only improvement early.
        if eval_dl is not None and (step % args.eval_every == 0
                                    or step == args.steps):
            model.eval()
            acc, nb_e, eval_err = {}, 0, None
            # ⚠️ FAIL LOUD, SURVIVE. An in-training eval is a DIAGNOSTIC; it
            # must never take the run down. A CUDA OOM or a decode error raised
            # in here used to propagate out of train() and END THE RUN. It is
            # now logged as an `eval_error` row + a `[v3:eval] FAILED` line
            # (traceback to stderr), the model goes back to train mode, and
            # training continues from the checkpoint written just above.
            # ⛔ A kernel SIGKILL -- the cgroup OOM killer of
            # C-REFCV3-EVAL-DEATH -- CANNOT be caught by any Python handler;
            # for that case the save-before-eval order above is the whole
            # protection.
            try:
                with torch.no_grad():
                    for eb in eval_dl:
                        if nb_e >= args.eval_batches:
                            break
                        el = compute_losses_v3(
                            model, eb, device, mode=args.mode,
                            ablate_frames=args.ablate_frames)
                        for k, v in el.items():
                            if torch.is_tensor(v) and v.ndim == 0:
                                acc[k] = acc.get(k, 0.0) + float(v.detach())
                            elif isinstance(v, (int, float, bool)):
                                acc[k] = acc.get(k, 0.0) + float(v)
                        nb_e += 1
            except Exception as exc:          # noqa: BLE001 (by design)
                import traceback            # local: this block is the boundary
                traceback.print_exc()
                eval_err = f"{type(exc).__name__}: {exc}"
            model.train()
            if eval_err is not None:
                log.write(json.dumps({"step": step, "eval_error": eval_err,
                                      "eval_batches_done": nb_e}) + chr(10))
                log.flush()
                # ASCII-safe on purpose: a UnicodeEncodeError raised while
                # REPORTING the failure would defeat the survival.
                print("[v3:eval] FAILED at step %d: %s -- training continues"
                      % (step, eval_err.encode("ascii", "backslashreplace")
                         .decode("ascii")), flush=True)
            elif nb_e:
                erow = {f"eval_{k}": round(v / nb_e, 5) for k, v in acc.items()}
                erow.update(step=step, eval_batches=nb_e,
                            eval_windows=nb_e * args.batch)
                log.write(json.dumps(erow) + chr(10))
                log.flush()
                print(f"[v3:eval] step {step} "
                      f"loss {erow.get('eval_loss', float('nan')):.4f} "
                      f"traj {erow.get('eval_traj', float('nan')):.4f} "
                      f"lat_tac {erow.get('eval_lat_tac', float('nan')):.4f}",
                      flush=True)
    # ⛔ the done-marker, SAME turn as completion (the v5f supervisor lesson).
    (out_dir / "summary.json").write_text(
        json.dumps({"done": True, "step": step, "arm": args.arm,
                    "seed": args.seed,
                    "wallclock_s": round(time.time() - t0, 1)}),
        encoding="utf-8")
    log.close()
    print(f"[v3:{args.arm}] DONE at {step} — summary.json written")
    return {"step": step}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size",
                    choices=tuple(v3.V3_SIZES) + tuple(v3.V3_RIG_SIZES),
                    default="small",
                    help="encoder rung. 'small' is the AS-REGISTERED arm "
                         "(62,930,419); 'xl' is the D-008 >=250M rung "
                         "(217,760,775). ⛔ The size axis moves the ENCODER "
                         "ONLY — the H-vs-F delta is identical at every rung "
                         "(pinned by tests). ⚠️ Anything but 'small' VOIDS the "
                         "registered cost line in PREREG_REFC_V3.md; amend "
                         "BEFORE any read, never after. ⛔ 'tiny' (16,989,725) "
                         "is the VALIDATION RIG rung (V3_RIG_SIZES), not a "
                         "member of the registered ladder: it exists so a "
                         "design change can be gated on the dev box before it "
                         "earns compute, and NOTHING measured at it is a model "
                         "claim or may enter MODEL_REGISTRY.md.")
    ap.add_argument("--arm", choices=("hier", "flat"), required=True,
                    help="v3-H (goal cascade) or v3-F (flat) — the dominance "
                         "pair; delta pinned to the registered lever set")
    ap.add_argument("--data-root", default=None,
                    help="raw epcache root of ep_*.pt (parity-guarded; the "
                         "physicalai-train-e438721ae894 convention)")
    ap.add_argument("--v2-cache", nargs="+", default=None,
                    help="v2 compressed cache dir(s) of *.v2ep.pt (e.g. the B1 "
                         "corpus physicalai-b1-w120-256x640cyl). Swaps the raw "
                         "loader for the lazy LRU-bounded v2 providers; the "
                         "window contract is identical. Pair with "
                         "--image-hw 256 640 for the B1 geometry — the run "
                         "REFUSES a geometry mismatch either way.")
    ap.add_argument("--prefetch-factor", type=int, default=1,
                    help="batches prefetched PER WORKER. One collated batch is "
                         "~940 MB at batch 20, so workers x prefetch bounds "
                         "shared-memory use; torch's default of 2 put 8 "
                         "batches in flight and exhausted a 24 GB /dev/shm.")
    ap.add_argument("--u8-batches", action="store_true",
                    help="ship frames/future_frames through the DataLoader as "
                         "uint8 and convert to float32 [0,1] ON THE DEVICE "
                         "(the dataset contract's own /255). Cuts every "
                         "in-flight batch 4x: at batch 20 / window 8 / 20 "
                         "future frames / 9ch / 256x640 one collated batch is "
                         "3.30 GB fp32 -> 0.83 GB uint8, so 6 workers x "
                         "prefetch 1 hold ~4.95 GB of /dev/shm instead of "
                         "~19.8 GB (the unreclaimable share that put refcv3's "
                         "50 GB cgroup at its cap and OOM-killed it at eval "
                         "steps, MEASURED 2026-09-02). Loss bit-identical "
                         "(pinned). Default OFF: the live run resumes "
                         "byte-identical; this is a next-relaunch arm.")
    ap.add_argument("--eval-cache", default=None,
                    help="HELD-OUT v2 cache (the v7.2 eval split). ⛔ MUST be "
                         "disjoint from --v2-cache: 141 of the 147 v7.2 eval "
                         "clips sit inside the raw B1 corpus, so training on "
                         "'all of B1' trains on the eval set (MEASURED "
                         "2026-09-02).")
    ap.add_argument("--eval-labels", default=None,
                    help="s2_labels_v7.2_eval.jsonl.gz for the eval split")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="run the held-out eval every N steps (0 = off)")
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="FIXED number of eval batches — the same windows "
                         "every time, so step-to-step deltas are the model "
                         "moving and not the sample moving")
    ap.add_argument("--v7-labels", default=None,
                    help="s2_labels_v7.2_*.jsonl.gz — the RELEASED tactical "
                         "vocabulary (8x8). Joined per window on "
                         "stable_episode_id(clip_id), the v7.2 index's own "
                         "'ONLY admissible join key'. Sets tac_vocab_version "
                         "to v7.0; without it the trainer derives kin3 (3x3) "
                         "from poses. Coverage is PRINTED and a run below "
                         "50 %% is REFUSED.")
    # ---- D-TACGOAL-1 / D-ROLL-1h: THE TACTICAL-GOAL SET HEAD ------- #
    # ⛔ OPT-IN, AND THE DEFAULT IS LOAD-BEARING. Building this head on
    # the vocabulary alone (D-ROLL-1) added 11,286 params to every rebuild
    # of a checkpoint trained before it existed, so refcv4b, three refcv3
    # checkpoints and the LIVE refcv5 run all stopped loading -- refcv5 was
    # unRESUMABLE, not merely unrollable. A default-OFF switch is what lets
    # a pre-drift `argv` (which cannot mention a flag that did not exist)
    # rebuild the parameter set it was trained with.
    ap.add_argument("--tac-goal-tok-head", action="store_true",
                    help="D-TACGOAL-1: build the 22-token tactical-goal "
                         "SET head (+11,286 params at d_tac 512). "
                         "Requires --v7-labels; kin3 has no tactical goal "
                         "vocabulary and refuses it. OFF by default so a "
                         "banked checkpoint rebuilds with the parameter "
                         "set it was trained with (D-ROLL-1).")
    ap.add_argument("--w-tac-goal", type=float, default=0.0,
                    help="D-TACGOAL: weight on the 22-token tactical-goal SET "
                         "loss (multi-label BCE, tac_goal_head.tac_goal_loss). "
                         "⛔ DEFAULT 0.0 AND THE TERM IS THEN ABSENT FROM THE "
                         "GRAPH -- not multiplied by zero -- so a recipe that "
                         "does not pass it is bit-identical to the pre-wiring "
                         "trainer at a fixed seed. Needs --tac-goal-tok-head "
                         "and --v7-labels; REFC_WEIGHT_GATES refuses a weight "
                         "whose term can never be reached. ⚠️ This weight is "
                         "ADDITIVE: nothing is taken from MANEUVER_WEIGHT and "
                         "no existing term is rebalanced -- that is a PI "
                         "decision (queue item 10), not this flag's.")
    ap.add_argument("--cot-negative-sidecar", default=None,
                    help="⭐ PI 2026-09-16: the absence-is-negative SIDECAR. "
                         "Required by, and only usable with, "
                         "--tac-goal-negatives cot-absence-negative. The path "
                         "IS the opt-in (there is no boolean), the loader "
                         "refuses a sidecar built over a different blob md5, "
                         "and the policy + both md5s are stamped into "
                         "config.json so an arm that supervised caption-"
                         "absence as a negative is identifiable from its own "
                         "artifacts. ⚠️ MEASURED exposure: 692/867 = 79.8 %% of "
                         "the clips asked the traffic-light question WITH a "
                         "visible light carry no traffic-light token. ⛔ That "
                         "is a PRESENCE rate, NOT a false-negative rate -- the "
                         "token is a REACTION, and a light governing another "
                         "lane is a present light with a correctly absent "
                         "reaction. It bounds how many of these new negatives "
                         "COULD be wrong; how many ARE is unmeasured "
                         "(2026-09-16-flywheel-negatives/RESULT.md §4.5).")
    ap.add_argument("--tac-goal-negatives", default="measured",
                    choices=["measured", "geometry", "all",
                             "cot-absence-negative"],
                    help="what an ABSENT goal token means. 'measured' "
                         "(default) reads provenance PER TOKEN FROM THE "
                         "LOADED SPLIT and supervises a negative only for "
                         "geometry-emitted tokens. ⛔ 'all' supervises every "
                         "absent cell: MEASURED, 3,574 of 4,572 clips were "
                         "never ASKED the traffic-light question, so 'all' "
                         "teaches the head that ~78 %% of the corpus has no "
                         "traffic light on no evidence. 'geometry' uses the "
                         "frozen declaration instead of the blob, kept so the "
                         "declaration/data divergence stays measurable.")
    ap.add_argument("--conflict-detector", choices=("auto", "on", "off"),
                    default="auto",
                    help="SPEC_REFCV6_V2 §6 / PREREG_BEV_CAPACITY_COMPETITION "
                         "§4: log cos(g_traj, g_aux) on the SHARED TRUNK every "
                         "step, pooled and per parameter group, plus the two "
                         "scale-SENSITIVE channels (|g_aux|/|g_traj| and the "
                         "projection onto g_traj) that the cosine is blind to. "
                         "'auto' (default) is ON for refcv6 arms with a live "
                         "perception weight and OFF otherwise. LOGGED AND "
                         "REPORTED, NEVER A STOP RULE -- the only thing it "
                         "refuses is to START when its own +1 / 0 / -1 "
                         "controls miss. With it off nothing is constructed "
                         "and metrics.jsonl carries no cd_* key; with it on "
                         "the training step is still bit-identical "
                         "(tests/test_refcv6_grad_conflict.py).")
    ap.add_argument("--conflict-mode", choices=("probe", "subtract"),
                    default="probe",
                    help="'probe' (default) is the PRE-REGISTERED statistic: "
                         "cos(g_traj, g_aux), two extra partial backwards. "
                         "MEASURED overhead on the dev box (CPU, batch 2): "
                         "+72 to +104 %% of step time. 'subtract' costs ONE "
                         "extra backward (+26 to +62 %% MEASURED) by recovering "
                         "the planning gradient from `.grad` by linearity -- "
                         "but its planning side is then the WHOLE planning "
                         "objective, not L_traj. ⛔ That is a DIFFERENT "
                         "statistic; every row it writes stamps "
                         "cd_plan_side=total_minus_aux, and it is not "
                         "interchangeable with 'probe' in a B2 verdict. Both "
                         "leave the training step bit-identical.")
    ap.add_argument("--conflict-every", type=int, default=1,
                    help="log the conflict reading every N steps (default 1, "
                         "which is what the prereg asks for). Raise it only if "
                         "the MEASURED overhead is unacceptable on the real "
                         "trunk -- and then say so in the run record, because "
                         "a median over sparse steps is a different statistic.")
    ap.add_argument("--grad-probe-modules", default="",
                    help="D-TACGOAL-2 / PI queue item 10: comma-separated "
                         "dotted module paths (e.g. "
                         "'tac_goal_tok_head,core.decoder.offset_head') whose "
                         "sum(|grad|) is written into metrics.jsonl on every "
                         "log step, UNROUNDED, measured BEFORE "
                         "clip_grad_norm_. EMPTY BY DEFAULT: with this flag "
                         "absent nothing is computed and the log schema is "
                         "unchanged, so a recipe that does not pass it is "
                         "identical to the pre-probe trainer. It exists "
                         "because tac_goal_tok_head took grad_abs_sum EXACTLY "
                         "0.00000 for all 40,284 steps of refcv5-v2 and "
                         "nothing in the run record said so.")
    ap.add_argument("--v2-lru", type=int, default=6,
                    help="per-process LRU of decoded-payload clips for "
                         "--v2-cache. ⚠️ B1 payloads are ~34 MB/clip "
                         "(256x640 PNG) — the old '2-4 MB' sizing is 8-17x "
                         "low (MEASURED, V5F_SIGKILL.md), and under "
                         "shuffle the hit rate is ~lru/n_clips, so big "
                         "values buy RAM pressure, not throughput. 6 "
                         "matches the v6 chain's setting.")
    ap.add_argument("--require-parity", action="store_true",
                    help="REFUSE unless --v2-cache references a REGISTERED "
                         "corpus key (B1 is unregistered as of 2026-09-01: "
                         "expect the loud NON-PARITY line without this flag; "
                         "registering B1 is the cross-arm-comparability "
                         "instrument)")
    ap.add_argument("--image-hw", type=int, nargs=2, default=None,
                    metavar=("H", "W"),
                    help="build the encoder at this input geometry (B1: "
                         "256 640). Params unchanged (fully-conv trunk); "
                         "compute scales with area. Applies to --preflight "
                         "too, so the gate runs at the launch geometry.")
    ap.add_argument("--synth-episodes", type=int, default=0,
                    help="CI-ONLY synthetic corpus (mutually exclusive with "
                         "--data-root / --v2-cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--mode", choices=("classifier", "diffusion"),
                    default="diffusion")
    ap.add_argument("--anchors", default=None,
                    help="6 s anchor vocabulary (build_refc_anchors.py over "
                         "V3_HORIZONS; the model's synthetic default otherwise)")
    ap.add_argument("--n-anchors", type=int, default=None,
                    help="anchor vocabulary size. Needed when the built file is "
                         "not the 128 the size preset assumes; `load_anchors` "
                         "refuses a shape mismatch, so a wrong value is loud.")
    ap.add_argument("--anchor-v0-conditioned", action="store_true",
                    help="refcv4-b: the anchor file carries `controls` [N, 2] = "
                         "(accel, curvature) and the bank is ROLLED PER WINDOW "
                         "from that window's measured v0 instead of being a "
                         "fixed set of paths in absolute metres. MEASURED on the "
                         "4,823-window banked surface: the fixed-path set reads "
                         "0.3773 m oracle-in-vocabulary (+0.0777 [+0.0528, "
                         "+0.1044] SEPARATED WORSE than ha = 0.2996); the "
                         "v0-conditioned family at 117 candidates reads 0.2610 "
                         "(-0.0387 [-0.0652, -0.0104] BEATS ha). Requires a "
                         "`controls` entry in --anchors; refused without one.")
    ap.add_argument("--anchor-control-units", default=None,
                    choices=["kappa", "alat"],
                    help="⛔ EXPLICIT OVERRIDE ONLY (default None): the anchor "
                         "artifact is the authority on what its own `controls` "
                         "column MEANS, and a controls-carrying file that "
                         "declares no `control_units` is REFUSED unless this "
                         "flag names them (recorded in config.json as "
                         "control_units_source='cli-override-legacy-file'); "
                         "a file that declares a different unit than this "
                         "flag is refused too. MEASURED 2026-09-04: the live "
                         "refcv4b anchors.pt declared nothing, and its "
                         "lateral column read as curvature gives 396 g at "
                         "36 m/s against the true 0.31 g. "
                         "What channel 1 of the anchor controls MEANS. "
                         "'kappa' = curvature 1/m, integrated as supplied. "
                         "'alat' = LATERAL ACCELERATION m/s^2, with curvature "
                         "DERIVED per window as a_lat / max(v0, 4.0)^2 and "
                         "clamped. MEASURED: a constant-CURVATURE family is "
                         "unflyable at speed (104 of 117 anchors break a mu=0.7 "
                         "circle at v0 = 27 m/s, peak 3.96 g); under 'alat' the "
                         "control space IS the Kamm-circle space, and the same "
                         "117 anchors read 0.1987 m oracle-in-vocabulary "
                         "(-0.1009 [-0.1213, -0.0813] vs ha) with 0/117 over "
                         "mu = 0.7 and a 0.68 g peak.")
    ap.add_argument("--anchor-ref-speed", type=float, default=10.0,
                    help="m/s the bank is rolled at where the ego channel was "
                         "WITHHELD by --ego-dropout. Rolling a withheld row from "
                         "its true v0 would put the withheld channel into the "
                         "candidate GEOMETRY, which is a harder leak than the "
                         "ranking one S2 guards.")
    # ---- refcv5 WP-4 / WP-6 ----------------------------------------------
    g5 = ap.add_argument_group(
        "refcv5", "WP-4 (control-space sampler) and WP-6 (agent tokens). "
        "EVERY flag here defaults to OFF, and OFF means NOT CONSTRUCTED: a "
        "run that passes none of them is bit-identical to refcv4b.")
    g5.add_argument("--sampler", default="none", choices=["none", "ddim"],
                    help="E-DDA-3. 'ddim' replaces the metre-space truncated "
                         "denoise with an anchored Gaussian in CONTROL space, "
                         "so every sample re-rolls through the kinematic model "
                         "and is flyable by construction. Needs a "
                         "v0-conditioned vocabulary and --w-u0 > 0.")
    g5.add_argument("--sampler-space", default="control",
                    choices=["control", "metre"],
                    help="DELIBERATE REGRESSION. 'metre' is the DD-literal "
                         "arm that noises the WAYPOINTS; pre-registered to "
                         "FAIL the flyability gate. If it does NOT fail, the "
                         "instrument cannot see what it is cited for: VOID.")
    g5.add_argument("--sampler-train-t-max", type=int, default=50,
                    help="train timestep ~ U[0, t_max). DD uses 50.")
    g5.add_argument("--sampler-infer-t", type=int, default=8,
                    help="the truncation point at eval. DD uses 8.")
    g5.add_argument("--sampler-steps", type=int, default=2,
                    help="DDIM steps at inference. DD uses 2; "
                         "D-REFC-DDAUDIT-2 MEASURED that passes beyond 2 "
                         "collapse the fan spread 104 -> 42 m, so this is NOT "
                         "a capacity knob.")
    g5.add_argument("--sampler-groups", type=int, default=1,
                    help="G samples per anchor. Only G=1 is implemented; G>1 "
                         "refuses and names the three consumers that still "
                         "assume an N-wide fan.")
    # ---- refcv5 P14: THE SAMPLER MUST RANK THE FAN IT EMITTED ----------- #
    # ⛔⛔ THESE TWO FLAGS DID NOT EXIST IN THIS TRAINER, AND THAT IS THE WHOLE
    # DEFECT. MEASURED 2026-09-06: `sel_refined` / `sel_score_emitted` appear
    # ZERO times in `refc_v3_train.py` (control: `sel_` appears 8 times, so the
    # zero is not a broken probe), while `refc_train.py` -- the v1.2 trainer --
    # has carried both since D-SEL. So `AnchoredDiffusionDecoder` implements
    # emitted-fan ranking, `refc.py` stamps `sampler_ranks_the_fan:
    # bool(self.sel.refined)` into every telemetry row, and the refcv5 arm had
    # NO WAY TO SET IT. It shipped `False`: the fan is SAMPLED, then ranked by
    # the CLASSIFIER surface -- a score computed BEFORE any sample was drawn.
    #
    # ⭐ WHAT IT COSTS, MEASURED ON THE BANKED FAN AT ZERO GPU (n = 881 windows
    # / 40 episodes, paired episode-cluster bootstrap):
    #   refc-base-30k  ADE 0.4728 shipped vs 0.1914 fan-best -- a recoverable
    #                  +0.2813 m, 95 % CI [+0.2127, +0.3543], separated;
    #                  a >2x-better trajectory sits IN THE FAN on 41.09 % of
    #                  windows and the ranking does not pick it.
    #   refc-xl-30k    +0.3075 m, CI [+0.2397, +0.3778]; 45.40 % of windows.
    # ⚠️ That is a CEILING on the lever, not its expected value -- it is what a
    # perfect re-ranking of the SAME fan would buy.
    #
    # ⛔ THE TWO ARE A PAIR AND THE TRAINER REFUSES THEM SPLIT (mirroring
    # `refc_train.py`'s own refusal). `--sel-refined` ALONE is the MEASURED
    # HARMFUL lever: 0.0259 m separated WORSE, flipping 29.82 % of picks and
    # lowering BOTH minority recalls -- because it ranks by a score that is one
    # pass STALE. `--sel-score-emitted` is what makes the score describe the
    # EMITTED fan; together they are the fix, apart they are the defect.
    g5.add_argument("--sel-refined", action="store_true",
                    help="rank by the REFINED/SAMPLED confidence instead of "
                         "the classifier surface. [!] ALONE THIS IS THE "
                         "MEASURED-HARMFUL LEVER (0.0259 m separated WORSE): "
                         "it ranks by a 1-pass-stale score. Must be paired "
                         "with --sel-score-emitted.")
    g5.add_argument("--sel-score-emitted", action="store_true",
                    help="P14. Score the EMITTED fan with one extra pass whose "
                         "OFFSET IS DISCARDED, so `anchor_traj` is bit- "
                         "unchanged and every banked oracle-in-fan contrast "
                         "stays comparable. Requires --sel-refined.")
    g5.add_argument("--sel-score-emitted-t", type=int, default=-1,
                    help="timestep for the emitted-fan scoring pass. -1 "
                         "continues the loop's own schedule. [!] ON A SAMPLER "
                         "ARM PASS 0: the sampler conditions through the "
                         "CONTINUOUS `time_mlp`, and t must name the fully "
                         "denoised state the fan actually is.")
    g5.add_argument("--ack-ddim-no-u0", action="store_true",
                    help="⭐ PI RULING 2026-09-11. PERMIT `--sampler ddim` with "
                         "`--w-u0 0` (refcv6 arm D) and STAMP the choice into "
                         "config.json as `u0_absent_under_ddim`. ⛔ This is a "
                         "RECORD, not a fix: the arm is still a denoiser "
                         "supervised only through the integrator, and every log "
                         "row still says 'sampler: ddim'. ⛔ Without it the "
                         "refusal stands. D-DDV1-NO-DENOISING-LOSS: DD-v1 has "
                         "no epsilon-prediction and no denoising MSE at all, so "
                         "our --w-u0 is an invention rather than a port, and "
                         "this trainer's own default is already 0.0.")
    g5.add_argument("--w-u0", type=float, default=U0_WEIGHT_DEFAULT,
                    help="weight on the x0 loss, in CONTROL space -- the only "
                         "term that supervises the sampler in the space it "
                         "samples in.")
    # ---- refcv5 STAGE 0: the FEASIBILITY-AWARE DECODE ------------------- #
    # ⛔⛔ THESE FLAGS EXIST BECAUSE THE MECHANISM WAS UNREACHABLE. MEASURED
    # 2026-09-05: `refc_v3_train.py` contained the string `feasible` ZERO
    # times, while `DecoderConfig.feasible_decode` and its projection had been
    # built, validated and banked. A capability the search cannot reach is not
    # a capability -- `DESIGN_CONSTRAIN_BY_CONSTRUCTION.md` part (2), *make the
    # good representable, ALONE IT DOES NOTHING*, observed on our own asset.
    #
    # What it buys, MEASURED on the refcv3 fan: `envelope` 0.8879 -> 0.0000 and
    # `kamm_over` 0.8408 -> 0.0000 (structural zeros, not small numbers),
    # `peak_g` 4.1789 -> 0.5964 g = 96.87 % of the gap; at MATCHED `fan_peak_g`
    # it costs -0.0103 m of oracle-ADE where an isotropic shrink costs +0.7680.
    # At T1 the whole deployment cost is `ade_0_2s` +0.0012 m -- 1.2 mm -- while
    # yaw-rate error falls 80.4 %. The RL stage this replaces FAILED its
    # committed exit at both seeds at +0.0362 m while making the fan LESS safe.
    #
    # ⛔ DEFAULT OFF, and OFF RETURNS THE SAME OBJECT (`refc.py::_feasible`
    # first line), so a run that passes none of these is bit-identical to
    # today -- proven at 4,823/4,823 windows, max |delta| = 0.0.
    g5.add_argument("--feasible-decode", action="store_true",
                    help="project every emitted fan onto the friction-feasible "
                         "set INSIDE the decoder, at every pass that moves a "
                         "waypoint. An envelope- or Kamm-violating path becomes "
                         "UNREPRESENTABLE -- an identity, not a penalty. A "
                         "reward moves the ranking by 0.001 and a post-train "
                         "veto closed ~2.7 pct of the gap while regressing T1 "
                         "+0.0362 m; both are soft, because they penalise a "
                         "path the decode can still emit.")
    g5.add_argument("--feasible-mu", type=float, default=0.7,
                    help="friction circle. <= 0 disables the Kamm disc and "
                         "keeps only the box clamp.")
    g5.add_argument("--feasible-entry", action="store_true",
                    help="also bind the FIRST step's speed to v0 +- a*dt. The "
                         "pre-registered `+entry` variant drives "
                         "`fan_infeasible` 0.8916 -> 0.0000 EXACTLY across all "
                         "51,200 candidates, and is the only variant that "
                         "removes the `off_reach` +0.2311 regression. Needs a "
                         "v0-conditioned bank: without v0 there is no entry "
                         "speed to bind to, and it is REFUSED at startup.")
    g5.add_argument("--feasible-a-max", type=float, default=4.0,
                    help="longitudinal box clamp, m/s^2. Defaults to the "
                         "SCORER's own A_MAX_MPS2 -- a projection run at a "
                         "different bound is approximately feasible and "
                         "reports a residual that looks like noise.")
    g5.add_argument("--feasible-kappa-max", type=float, default=0.2,
                    help="curvature box clamp, 1/m.")
    g5.add_argument("--feasible-prefix-slots", type=int, default=4,
                    help="the UNIFORM prefix the scorer differentiates (4 = "
                         "the 2 s window at 0.5 s spacing). ⛔ The prefix dt is "
                         "DERIVED from anchor_slots and a NON-UNIFORM prefix is "
                         "REFUSED at startup, never guessed -- the `df` / "
                         "`step_s` scope family in a geometry costume.")
    g5.add_argument("--agents", default="off",
                    choices=["off", "head", "oracle"],
                    help="E-AGT-*. 'head' = the LEARNED monocular 3D head "
                         "(the deliverable arm; vision-only at inference, "
                         "obstacle.offline as TRAIN-TIME labels). 'oracle' = "
                         "GROUND-TRUTH boxes fed at inference: the CEILING, "
                         "INADMISSIBLE as a capability claim, run ONCE to "
                         "price the mechanism before any detector GPU-day. If "
                         "the oracle does not separate on LONGITUDINAL AND "
                         "TACTICAL the whole mechanism is refused.")
    g5.add_argument("--w-agent", type=float, default=AGENT_WEIGHT_DEFAULT,
                    help="weight on the GT-supervised detection set loss.")
    g5.add_argument("--agent-queries", type=int,
                    default=AGENT_QUERIES_DEFAULT,
                    help="detection queries. 100, ruled by the Master Mind "
                         "(mm-decisions M17) on the TRAIN corpus: mean 4.39, "
                         "p99 30, MAX 94 over 433,040 frames / 12,122,129 "
                         "boxes / 2,308 clips. The previous 32 came from "
                         "val40 (max 24) and is REFUTED on train -- it drops "
                         "41,362 boxes (2.18 pct) on 3,250 frames and "
                         "match_slots keeps the NEAREST N, so the nearest "
                         "SACRIFICED target sits at 13.1 m, inside the "
                         "braking envelope. N=64 does not fix it either "
                         "(33.9 m). 94 is a max over a SAMPLE, not a bound; "
                         "100 carries headroom and is DETR's ordinary budget.")
    g5.add_argument("--agent-w-project", type=float, default=0.0,
                    help="weight on the IMAGE-PLANE term. A monocular head "
                         "supervised only in BEV metres is asked to regress "
                         "the one axis it cannot directly see, with no term "
                         "in the space it can. NEEDS --agent-rig-camera: a "
                         "non-zero weight with no camera REFUSES (it used to "
                         "no-op while being stamped -- mm-decisions M18).")
    g5.add_argument("--agent-w-ground", type=float, default=0.0,
                    help="weight on the road-plane range prior. Costs NO "
                         "label (rig z=0 IS the road plane, MEASURED), so it "
                         "also trains on the NO_LABEL frames past ~20 s. "
                         "NEEDS --agent-rig-camera, and prefers "
                         "'extrinsics': it back-projects through the road "
                         "plane, so a NOMINAL (pitch-free) mount biases its "
                         "range directly.")
    g5.add_argument("--agent-rig-camera", default="off",
                    choices=list(AGENT_RIG_CAMERA_SOURCES),
                    help="the RigCamera the two monocular terms are computed "
                         "against. 'off' = none, and then a non-zero "
                         "--agent-w-project/--agent-w-ground REFUSES at "
                         "startup. 'nominal' = boresight-forward mount with "
                         "NO pitch (stamped as such). 'extrinsics' = built "
                         "from a sensor_extrinsics quaternion on file -- the "
                         "only admissible constructor for corpus work.")
    g5.add_argument("--agent-rig-extrinsics", default=None,
                    help="JSON with the front-wide sensor_extrinsics "
                         "quaternion (qx/qy/qz/qw = rotation_cam_to_vehicle) "
                         "and optional x/y/z [m] in the rig frame. Required "
                         "by --agent-rig-camera extrinsics.")
    g5.add_argument("--agent-rig-extrinsics-allow-partial",
                    action="store_true",
                    help="accept a PER-CLIP extrinsics table that does not "
                         "cover every episode of the run. Without it a gap "
                         "REFUSES: the uncovered rows get no camera, both "
                         "monocular terms skip them, and config.json would "
                         "still read mount_pose_scope PER-CLIP -- the M18 "
                         "dead-flag defect with a camera in place of a "
                         "weight. The coverage is stamped either way.")
    g5.add_argument("--agent-cam-height", type=float, default=1.5,
                    help="mount height [m] for --agent-rig-camera nominal. "
                         "The MEASURED corpus band is 1.43-1.56 m; outside "
                         "it the run warns, and outside 0.5-3.0 m it "
                         "refuses.")
    g5.add_argument("--agent-sigma-range", type=float, default=0.0,
                    help="E-AGT-BUDGET: range-noise sigma (m) on the ORACLE "
                         "boxes. The sigma where separation dies IS the "
                         "detector specification.")
    g5.add_argument("--agent-miss-rate", type=float, default=0.0,
                    help="E-AGT-BUDGET: the miss rate on the ORACLE boxes.")
    g5.add_argument("--agent-join", default=None,
                    help="obstacle.offline join file (.jsonl or .jsonl.xz) "
                         "-- the TRAIN-TIME LABELS for --agents head. Without "
                         "it the batch carries no `agent_box` and "
                         "--w-agent > 0 REFUSES, because a run that stamps "
                         "w_agent > 0 while training no detector reads as "
                         "'the agent head does not help'. Train corpus: "
                         "HF Sayood/tanitad-ph0-aug120 -> "
                         "joins/train2400_agents.jsonl.xz (2,308 clips, "
                         "433,040 frames, md5 "
                         "24cbdca8c3b23aafc2fb17e6bf99cf76).")
    g5.add_argument("--agent-pad", type=int, default=0,
                    help="targets per window in the padded block. 0 = the "
                         "join's own MEASURED max, so nothing is truncated "
                         "before match_slots applies its counted nearest-N "
                         "policy. A smaller value truncates VISIBLY (counted "
                         "in agent_n_truncated), never silently.")
    g5.add_argument("--agent-join-no-rates", action="store_true",
                    help="skip the per-track rate finite differences. The "
                         "rates term is then computed over ZERO items and "
                         "SAYS SO -- but v_rel_x is what the LONGITUDINAL "
                         "family (closing speed, TTC) is built from, so the "
                         "default computes them.")
    g5.add_argument("--agent-join-verify", default="auto",
                    choices=["auto", "off"],
                    help="check --agent-join against its sidecar's DECLARED "
                         "digest scope (mm-decisions M18). 'auto': verify if "
                         "a sidecar exists; REFUSE if it exists and declares "
                         "no scope (the two joins' md5s cover DIFFERENT "
                         "artifacts -- train's the .xz, val40's the .jsonl -- "
                         "and guessing from the filename is MEASURED wrong); "
                         "warn if there is no sidecar at all. 'off' records "
                         "in config.json that the operator checked nothing.")
    g5.add_argument("--agent-join-allow-legacy-ids", action="store_true",
                    help="accept a join that matches this corpus ONLY through "
                         "the LEGACY 16-bit episode id. That key is the first "
                         "4 BYTES of the clip id and COLLIDES (34 of 2,308 "
                         "clips on the train join), so an episode absent from "
                         "the join can match a different clip that is "
                         "present. Stamped into config.json.")
    g5.add_argument("--agent-presence-hard", action="store_true",
                    help="hard-mask sub-threshold slots instead of soft "
                         "scaling. Soft is the default BECAUSE a hard mask has "
                         "zero gradient to the presence head through the "
                         "planner loss.")
    # ---- WP-D: the BEV auxiliary loss (TRAINING-ONLY) -------------------
    g5.add_argument("--bev-aux", default="off", choices=["off", "col", "xcol"],
                    help="WP-D. Supervise the trunk's feature map with a POLAR "
                         "agent-occupancy target built from the same "
                         "obstacle.offline join --agent-join loads. TRAINING-"
                         "ONLY: the head is constructed LAST and removing it "
                         "is BIT-IDENTICAL at the planner's output (pinned by "
                         "stack/tests/test_bev_aux.py). 'col' = one ray MLP "
                         "per azimuth column, no cross-column mixing (the "
                         "CHEAP FLOOR, and the arm the cylindrical projection "
                         "justifies); 'xcol' = one self-attention block over "
                         "the column tokens, so cross-column reasoning is an "
                         "ABLATION rather than an assumption.")
    g5.add_argument("--w-bev-aux", type=float, default=0.0,
                    help="weight on the BEV auxiliary loss. ⛔ Checked against "
                         "the trajectory loss at step 0 by "
                         "refc_bev_aux.assert_loss_parity: E-DEC-18b MEASURED "
                         "the sibling PSG term sitting 10-30x above the "
                         "objective it was meant to support, at EVERY weight "
                         "tested, and destroying the encoder.")
    g5.add_argument("--bev-aux-occlusion", default="mask",
                    choices=["mask", "none"],
                    help="the THIRD STATE. 'mask' (default) marks cells behind "
                         "an occluder UNOBSERVABLE and does not supervise "
                         "them. ⛔ 'none' is the DELIBERATE REGRESSION: it "
                         "supervises occluded space as FREE, which is the "
                         "two-state merge WP-A flagged. MEASURED on B1 EVAL "
                         # ⚠️ `%%`, NOT `%`. argparse formats a help string
                         # against a dict, so a bare `% o` becomes a `%o`
                         # conversion and `--help` dies with `TypeError: %o
                         # format: an integer is required, not dict`. MEASURED
                         # here 2026-09-07: this exact defect, caught by
                         # `test_p14_trainer_help_lists_the_flags_and_mine_are_ascii`
                         # -- which is why that test asserts on the RETURN CODE
                         # of a real `--help` subprocess and not on the text.
                         "(26,394 frames): 17.66 %% of cells are occluded and "
                         "27.96 %% of OCCUPIED cells are, so 'none' mislabels "
                         "more than a quarter of the agents as empty road.")
    g5.add_argument("--bev-aux-detach", action="store_true",
                    help="⛔ DELIBERATE REGRESSION: the head reads a DETACHED "
                         "feature map, so it learns the target and teaches the "
                         "trunk NOTHING. The arm that proves the aux gradient "
                         "actually reaches the trunk.")
    g5.add_argument("--bev-aux-rng", type=int, default=24,
                    help="range bins over --bev-aux-rmax (24 x 2.5 m = 60 m, "
                         "matching bev_raster's x_fwd_m).")
    g5.add_argument("--bev-aux-rmax", type=float, default=60.0,
                    help="target range, metres.")
    g5.add_argument("--bev-aux-pos-weight", type=float, default=30.61,
                    help="BCE positive-class weight. ⭐ A pre-registered "
                         "CONSTANT from the MEASURED corpus base rate "
                         "(3.1634 %% of supervised cells over the whole B1 "
                         "EVAL join => (1-p)/p = 30.61), NEVER computed from "
                         "the batch: a batch-derived weight makes two arms "
                         "with identical flags optimise different objectives.")
    g5.add_argument("--bev-aux-shuffle", action="store_true",
                    help="⭐ THE INFORMATION CONTROL. Pair each row's features "
                         "with ANOTHER row's BEV target (a deterministic "
                         "roll-by-1 over a SHUFFLED batch, so the partner is "
                         "an unrelated window and NO RNG is consumed — the "
                         "arm stays seed-comparable with the real one). Same "
                         "head, same parameter count, same gradient "
                         "magnitude, ZERO information. ⛔ If the real arm's "
                         "gain over aux-off is matched here, the gain is a "
                         "capacity/regularisation effect and NOT agent "
                         "content — the WP-A `shuffled` control, moved from "
                         "the probe into the trainer.")
    g5.add_argument("--bev-aux-hidden", type=int, default=256)
    g5.add_argument("--bev-aux-dtok", type=int, default=64)
    # ---- refcv5 WP-B (`E-WP-INDEX-1`): DiffusionDrive coupling (1) -------- #
    g5.add_argument("--wp-index", default="off", choices=["off", "on"],
                    help="⭐⭐ WP-B — WAYPOINT-INDEXED cross-attention into "
                         "the SPARSE agent tokens. Each anchor query's current "
                         "waypoint estimate (metres, ego frame) is the ADDRESS: "
                         "the metric relation between those waypoints and each "
                         "agent slot's decoded centre becomes a per-head "
                         "additive attention bias, so the trajectory's own "
                         "geometry decides which agent it reads. ⛔ NOT a "
                         "fusion block — the query LATENT never enters the "
                         "address. ⛔ REQUIRES --agents head|oracle: with no "
                         "agent tokens there is nothing to address and the arm "
                         "would read as 'the index does not help' while never "
                         "having had one. ⛔ Indexes the SPARSE tokens and "
                         "never a dense BEV raster (WP-A MEASURED a "
                         "quantisation floor of AP 0.4713 on the raster route "
                         "that no training removes).")
    g5.add_argument("--wp-index-mode", default="geom",
                    choices=list(_refc_wp_index.WP_INDEX_MODES),
                    help="⛔ THE PRE-REGISTERED CONTROLS, as run modes. "
                         "geom = the treatment. shuffle = waypoints permuted "
                         "ACROSS THE BATCH, so each row is addressed with "
                         "another sample's geometry — if this matches the real "
                         "arm the coupling is CAPACITY, not CONTENT (refuses "
                         "batch < 2, where the permutation is the identity). "
                         "const = every waypoint at --wp-index-const-xy, the "
                         "no-information address; its known value is a bias "
                         "IDENTICAL across the anchor axis.")
    g5.add_argument("--wp-index-detach", action="store_true",
                    help="⛔ THE DETACHED CONTROL. Compute the address but "
                         "block the gradient it would send back through the "
                         "waypoints and the agent boxes into the trunk. Its "
                         "known value is a trunk gradient of EXACTLY zero "
                         "through the index path.")
    g5.add_argument("--wp-index-radius-m", type=float, default=0.0,
                    help="> 0 turns on the DEFORMABLE/LOCAL variant: an agent "
                         "further than this from every waypoint of a query is "
                         "masked out of that query's attention. 0 (default) is "
                         "the soft bias-only index, which is the PRIMARY arm; "
                         "the radius is the pre-registered NEXT LEVER if the "
                         "soft arm misses its bar.")
    g5.add_argument("--wp-index-hidden", type=int, default=32,
                    help="width of the per-layer bias MLP. Cost is "
                         "`8*h + h + h*H + H` per decoder layer — 552 params "
                         "per layer at h=32, H=8, and param_breakdown reports "
                         "it on its OWN line, carved out of `decoder`.")
    g5.add_argument("--wp-index-scale-m", type=float, default=10.0,
                    help="metres. Normaliser for the metric address features "
                         "(NOT a cut-off — that is --wp-index-radius-m).")
    g5.add_argument("--wp-index-const-xy", type=float, nargs=2,
                    default=(10.0, 0.0), metavar=("X_M", "Y_M"),
                    help="the const control's fixed address, metres, ego "
                         "frame (x forward, y left).")
    ap.add_argument("--withheld-bank", default="fixed",
                    choices=list(refc.WITHHELD_BANK_MODES),
                    help="H-EGO-LIT-4: the speed a WITHHELD row's anchor bank "
                         "is rolled at (v0-conditioned vocabularies only). "
                         "fixed = --anchor-ref-speed (the shipped roll; "
                         "bit-identical to the pre-flag trainer); pred = the "
                         "model's OWN detached vision-only g_tac 2 s speed "
                         "(SparseDrive's pattern), clamped to [0, "
                         "--withheld-speed-max]; random = a draw from the "
                         "TRAINING marginal of v0 (the blindness CONTROL); "
                         "none = every row at the reference speed (the field's "
                         "speed-blind vocabulary). Stamped in config.json "
                         "under seams.withheld_bank and withheld_bank.")
    ap.add_argument("--withheld-bank-warmup", type=int, default=0,
                    help="steps trained on the FIXED withheld bank before "
                         "--withheld-bank pred/random goes live (the goal "
                         "head's speed is noise at step 0). Logged per step "
                         "as withheld_bank_active.")
    ap.add_argument("--withheld-speed-max", type=float, default=35.0,
                    help="clamp for --withheld-bank pred, m/s.")
    ap.add_argument("--sel-accel-max", type=float, default=None,
                    help="S2 reach-clamp bound in m/s^2. RE-DERIVE IT FOR THE "
                         "HORIZON: `horizon_s` is max(horizons)*0.1, so the "
                         "band is a*horizon_s. The 2.5 default is a 2 s number "
                         "(+-5 m/s); at 6 s it opens to +-15 m/s and kills "
                         "only 18.02 pct of a data-driven vocabulary. MEASURED "
                         "pick for V3_HORIZONS: 2.0 (+-12 m/s, kills 26.23 "
                         "pct, deletes 0.000 pct of eval / 0.007 pct of train "
                         "GT, dADE +0.00000 m).")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    # ---- refcv6 §2: the trunk and the optimiser the papers actually use --- #
    # ⛔ BOTH DEFAULT TO TODAY'S BEHAVIOUR. `--trunk refc` + `--opt adam` is
    # bit-for-bit the pre-refcv6 trainer (pinned by
    # tests/test_refcv6_trunk.py::test_default_optimiser_is_bit_identical).
    ap.add_argument("--trunk", choices=("refc", "timm"), default="refc",
                    help="refcv6 §2. `timm` builds `resnet34.a1_in1k` with "
                         "ImageNet weights (DiffusionDrive's trunk and "
                         "DiffusionDrive's init, `rl_config.py:16-18`); it "
                         "REFUSES to build if the weights did not load. "
                         "`refc` (default) is the in-repo random-init 90.5 M "
                         "trunk. ⚠️ The two differ in feat_dim (512 vs "
                         "base_width*8): switching is a new arm, never a "
                         "resume.")
    ap.add_argument("--trunk-in-channels", type=int, default=0,
                    help="0 = leave the config's `in_channels` alone. It is "
                         "3*K: the stack width the corpus delivers, and K is "
                         "DERIVED from it. 9 = our 3-frame stack (the PI's "
                         "2026-09-16 default), 3 = DD's single frame.")
    ap.add_argument("--trunk-name", default="resnet101.a1_in1k",
                    help="timm backbone id. ⭐ PI 2026-09-16: PRIMARY "
                         "`resnet101.a1_in1k` (42,500,160 params; 1024 @ "
                         "stride-16 / 2048 @ stride-32), SECOND COMPARISON RUN "
                         "`resnet34.a1_in1k` (21,284,672; 256 / 512) -- the "
                         "papers' and DiffusionDrive's own. Every channel "
                         "count is read from timm's `feature_info`, so a swap "
                         "needs no model-code change.")
    ap.add_argument("--trunk-mode", choices=("shared", "inflate"),
                    default="shared",
                    help="⭐ PI 2026-09-16: FRAME HISTORY IS REQUIRED. "
                         "`shared` (DEFAULT) runs the SAME weights over each "
                         "of the K frames, 3 ImageNet-normalised channels "
                         "each, and fuses AFTER the trunk at both strides -- "
                         "which keeps the ImageNet prior EXACT. `inflate` is "
                         "the cheaper alternative arm: ONE pass with a "
                         "3K-channel stem whose weights are repeated and "
                         "divided by K.")
    ap.add_argument("--trunk-fuse", choices=("concat1x1", "attn", "last"),
                    default="concat1x1",
                    help="temporal fusion over the K per-frame maps. `last` is "
                         "the single-frame CONTROL. Identity-initialised on "
                         "the NEWEST frame, so history is a REMOVABLE graft "
                         "and step 0 is bit-identical to K = 1.")
    ap.add_argument("--trunk-fuse-plain-init", action="store_true",
                    help="⛔ DELIBERATE REGRESSION of the identity init. "
                         "Without the identity, 'K frames beat 1 frame' is "
                         "confounded with 'a random 1x1 conv was inserted'.")
    # ⛔⛔ THE KNOCKOUT ARM HAD NO SWITCH. `CNNEncoderConfig.trunk_pretrained`
    # is a real field (`refc.py:348`, *"False is the knockout arm"*), consumed
    # at `refc.py:1439` and stamped into config.json at `:3325` -- and until
    # 2026-09-17 NO CLI FLAG REACHED IT, so the ImageNet-vs-random-init
    # comparison the field exists for could not be run from argv at all. Same
    # class as the `--w-agent` defect with the direction reversed: not a flag
    # that reaches nothing, but a config that no flag reaches.
    ap.add_argument("--trunk-pretrained", dest="trunk_pretrained",
                    action="store_true", default=None,
                    help="load timm's ImageNet weights (the DEFAULT; the trunk "
                         "REFUSES to build if they did not really load).")
    ap.add_argument("--no-trunk-pretrained", dest="trunk_pretrained",
                    action="store_false",
                    help="⛔ THE KNOCKOUT ARM: build the SAME timm "
                         "architecture with a RANDOM init. Without it, 'the "
                         "ImageNet prior helps' is unfalsifiable -- the "
                         "comparison arm cannot be built. Stamped into "
                         "config.json as trunk_pretrained: false.")
    # ---- refcv6 §2/§6: THE PERCEPTION BRANCH (PI 2026-09-16) ------------- #
    # *"train jointly the resnet-trunk, the bev map (based on the sam2 maps as
    # gt) and a head for 3d bounding boxes extracted from the resnet trunk"*.
    # ⛔ BOTH WEIGHTS DEFAULT TO 0.0, and at 0.0 NOTHING is built: no module,
    # no parameters in the optimiser, no key in metrics.jsonl, no block in
    # config.json. The default path is bit-identical to the pre-branch trainer.
    g6 = ap.add_argument_group("refcv6 perception (map + 3-D boxes)")
    g6.add_argument("--w-map", type=float, default=0.0,
                    help="weight on the SAM3 BEV map loss (9-class soft CE on "
                         "SEEN cells only, `bev_encoder.map_soft_ce`). > 0 "
                         "needs --map-gt-root and --agent-rig-extrinsics (the "
                         "lift back-projects through the road plane, so the "
                         "mount pose is per clip). ⛔ SAM3 maps are the ONLY "
                         "BEV map target -- LiDAR is NOT a training target "
                         "(PI); it may be quoted as an independent evaluation "
                         "reference.")
    g6.add_argument("--w-box3d", type=float, default=0.0,
                    help="weight on the 3-D cuboid set loss "
                         "(`box3d_head.box3d_set_loss`, Hungarian, metres). "
                         "> 0 needs --agent-join AND --join3d.")
    g6.add_argument("--map-gt-root", default=None,
                    help="directory of `<sha12>.sam3mapgt.npz` (or the "
                         "canonical semantic_maps/gt/ layout). Clip ids appear "
                         "in artifacts ONLY as sha12.")
    g6.add_argument("--map-min-coverage", type=float, default=None,
                    help="floor on the fraction of TRAIN WINDOWS that have a "
                         "map frame; below it the run REFUSES rather than "
                         "training on fewer cells. Default = "
                         "`perception_targets.MIN_MAP_COVERAGE` (0.90).")
    g6.add_argument("--map-lru", type=int, default=4,
                    help="how many clips' decompressed `cart_frac` arrays to "
                         "keep resident (~14 MB each for 201 frames).")
    g6.add_argument("--join3d", default=None,
                    help="the 3-D agent join (`*_agents_3d.jsonl.xz`) whose "
                         "`cz`/`h` widen the 2-D targets. ⛔ NOTE the field "
                         "names are `cz`/`h`, NOT the parquet's "
                         "`center_z`/`size_z`. Absent 3-D labels are a MASK, "
                         "never a zero (`box3d_head.zh_targets`).")
    # ---- refcv6 §4: THE TACTICAL BEHAVIOUR DECODER (PI 2026-09-16) ------- #
    # ⭐ THE PI, verbatim: *"The tactical layer must learn to emitt the valid
    # tactical behaviors, choose the best architecture for it. It shoudl learn
    # them from the scene embeddings, for the agent and the map."*
    #
    # ⛔⛔ WHY THESE FLAGS EXIST AT ALL — the defect they close. MEASURED
    # 2026-09-17 at tip 837c308: `refcv6_tactical.TacticalBehaviourDecoder` is
    # BUILT (`refc_v3.py:1128`), FORWARD-RUN (`:1622`), and writes
    # `cache["tacv6_*"]` (`:1627`) — and this file contained **ZERO**
    # occurrences of the string `tacv6`. 2,262,020 parameters (d_bev 256) with
    # NO GRADIENT PATH and no flag that could switch them on. That is
    # `tac_goal_tok_head` — 11,286 params at `grad_abs_sum` EXACTLY 0 for all
    # 40,284 steps of refcv5-v2 — rebuilt 200x larger one layer up.
    #
    # ⛔ THE DEFAULT IS 0.0/OFF AND AT THE DEFAULT NOTHING IS BUILT: no module,
    # no optimiser tensor, no metrics key, no config block. A recipe that does
    # not pass these is BIT-IDENTICAL to the pre-wiring trainer at a fixed seed
    # (proven on artifacts, not on an exit code — see the RESULT's bit-identity
    # section), which is what lets this land beside a live recipe.
    g6t = ap.add_argument_group("refcv6 tactical behaviour decoder (§4)")
    g6t.add_argument("--tac-decoder-v6", action="store_true",
                     help="build the DETR-style behaviour decoder (2 layers, "
                          "d=256; 38 queries = 22 behaviours + 8 lat + 8 lon) "
                          "over the SCENE — agent slots (and BEV tokens when "
                          "a build supplies them). ⛔ Image tokens are "
                          "STRUCTURALLY excluded. Needs --arm hier, "
                          "--v7-labels and --agents (the agent slots are the "
                          "only live key/value source; see the RESULT's "
                          "blocked-seam section for the BEV half). ⛔ REFUSES "
                          "with --w-tac-v6 0: a built head with no live "
                          "weight is exactly the 2,262,020-parameter "
                          "zero-gradient defect this flag exists to close.")
    g6t.add_argument("--w-tac-v6", type=float, default=0.0,
                     help="weight MULTIPLIER on the tactical layer's own "
                          "losses (goal BCE 0.05 + lat CE 0.025 + lon CE "
                          "0.025, i.e. the existing MANEUVER_WEIGHT budget "
                          "0.1 re-split three ways by "
                          "`refcv6_tactical.TacticalLossWeights`; the "
                          "confidence head is a SUB-SPLIT of the goal BCE, "
                          "not a fourth term). ⛔ DEFAULT 0.0 AND THE TERM IS "
                          "THEN ABSENT FROM THE GRAPH — not multiplied by "
                          "zero — so the no-flag path is bit-identical. "
                          "⚠️ 1.0 reproduces the spec's weights exactly; this "
                          "multiplier exists so the budget can be scaled "
                          "without editing the spec's split.")
    g6t.add_argument("--tac-decoder-d-bev", type=int, default=0,
                     help="declared WIDTH of the BEV tokens fed to the "
                          "behaviour decoder; 0 (default) = agent slots only. "
                          "⛔ The token COUNT is always derived from the "
                          "tensor (`bev_feats_to_tokens`); only the CHANNEL "
                          "width is declared. ⛔ It is NOT the trunk's "
                          "stride-16 width: the tokens are the SUPERVISED MAP "
                          "branch's features, so the only admissible value is "
                          "`BEVEncoderConfig.d_out` and anything else REFUSES "
                          "with the derived number named. ⭐ > 0 is live since "
                          "PI RULING 2026-09-17 (R2) and needs --w-map > 0 "
                          "(the lift and BEV encoder are built only behind a "
                          "live map weight).")
    g6t.add_argument("--tac-decoder-bev-detach", action="store_true",
                     help="ABLATION of PI ruling R3. By default the BEV "
                          "tokens reach the decoder ATTACHED, so the tactical "
                          "loss shapes the shared trunk — the PI's explicit "
                          "instruction (*\"you can backpropagate to the "
                          "trunk\"*). This flag cuts that path and is stamped "
                          "in config.json as `bev_detached`. ⛔ REFUSES with "
                          "--tac-decoder-d-bev 0: nothing to detach.")
    g6t.add_argument("--tac-decoder-valid-threshold", type=float, default=0.5,
                     help="sigmoid threshold at which a behaviour counts as "
                          "VALID for the selection gate and for "
                          "`tacv6_valid_frac`. Reporting only; the BCE loss "
                          "is threshold-free.")
    g6t.add_argument("--graft-behaviour-sel", action="store_true",
                     help="let the valid-behaviour set gate anchor SELECTION "
                          "(`refc.py:3135`, additive log-space term, fed "
                          "DETACHED). ⛔ Needs --tac-decoder-v6: without the "
                          "decoder no `behaviour_term` is ever produced and "
                          "the flag would be silently inert while "
                          "config.json stamps it on.")
    # ---- refcv6 §5: THE 4-VALUE MAX-SPEED INPUT (PI 2026-09-16) ---------- #
    # ⭐ THE PI, verbatim: *"It should few discrete values: 30 kph, 50 kph,
    # 100 kph, 120 kph ... Let use it as input in our next experiment"*.
    # ⛔ A DIFFERENT CHANNEL FROM `--max-speed-input` (E16), which is the
    # CONTINUOUS 8-step ladder over the v8 `speed_max_input` block.
    # `RefCV3Model.__init__` refuses the two together and so does the pin.
    g6t.add_argument("--max-speed-input-v6", action="store_true",
                     help="feed the PI's FOUR-value set-speed {30, 50, 100, "
                          "120} km/h as a 4-way ONE-HOT into the behaviour "
                          "decoder's condition. ⛔ EGO-FUTURE DERIVED "
                          "(~1.7555 bits, MEASURED) and stamped as "
                          "`speed_max_derivation_v6` in config.json; "
                          "`refcv6_max_speed.assert_speed_max_stamp_v6` "
                          "refuses a run that would not carry the "
                          "declaration, and refuses the mirror case too. "
                          "Needs --speed-max-sidecar-v6.")
    g6t.add_argument("--speed-max-sidecar-v6", default=None,
                     help="the sidecar built by "
                          "`scripts/build_refcv6_speed_max_window.py`: one "
                          "JSON-lines record per clip carrying the "
                          "CONTAINING-WINDOW bin. ⛔ The loader joins on "
                          "`sid` (= `stable_episode_id(clip_id)`, the "
                          "corpus's only admissible join key) and refuses a "
                          "sidecar whose `source_md5` does not match the "
                          "label blob this run loaded — two quantizations of "
                          "one corpus is two experiments.")
    g6t.add_argument("--speed-max-sidecar-v6-eval", default=None,
                     help="the EVAL split's sidecar. Defaults to "
                          "--speed-max-sidecar-v6, which is correct only when "
                          "the eval labels are the SAME blob; otherwise the "
                          "reader's md5 guard refuses, which is the point.")
    # ---- refcv6 §2b: EGO HISTORY AS AN INPUT (PI 2026-09-16) ------------- #
    ap.add_argument("--ego-history", action="store_true",
                    help="⭐ PI 2026-09-16. Encode the OBSERVED window's ego "
                         "track (speed, longitudinal accel, yaw rate per step) "
                         "into a vector that joins the condition beside nav "
                         "and max speed. ⛔ PAST ONLY -- the encoder slices "
                         "before it computes and a mutation test asserts a "
                         "future index is never read.")
    ap.add_argument("--ego-history-kind", choices=("gru", "conv1d"),
                    default="gru")
    ap.add_argument("--ego-history-hidden", type=int, default=64)
    ap.add_argument("--ego-history-out", type=int, default=32)
    ap.add_argument("--opt", choices=("adam", "dd"), default="adam",
                    help="refcv6 §2. `dd` is DiffusionDrive's optimiser "
                         "(`diffusiondrivev2_rl_config.py:119-131`): AdamW, "
                         "weight_decay 1e-4, and the ENCODER parameter group "
                         "at `cfg_lr_mult` = 0.5x the head lr. `adam` "
                         "(default) is today's plain Adam, one lr, no weight "
                         "decay -- kept as the default so every banked run "
                         "stays reproducible.")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="`--opt dd` only. DD's value, NOT torch AdamW's 1e-2 "
                         "default.")
    ap.add_argument("--encoder-lr-mult", type=float, default=0.5,
                    help="`--opt dd` only. DD's `cfg_lr_mult`.")
    # ---- refcv6 §3: F1..F9, each its own flag, each default OFF ----------- #
    # ⛔ With none of these passed, `refcv6_flags_from_args` returns None,
    # `DecoderConfig.refcv6` stays None, and the decoder constructs NOTHING --
    # pinned by tests/test_refcv6_diffusion.py::
    # test_all_flags_off_is_bit_identical_on_64_windows.
    g6 = ap.add_argument_group("refcv6 diffusion (F1..F9)")
    g6.add_argument("--f1-random-t", action="store_true",
                    help="F1. DD's TRAINING objective: ONE decoder call at "
                         "t ~ U[0, --f1-t-max) per sample, instead of "
                         "backprop through the 2-step inference chain. DD "
                         "`transfuser_model_v2.py:463-476`.")
    g6.add_argument("--f1-t-max", type=int, default=50,
                    help="F1's draw bound, EXCLUSIVE. 50 is DD's. 1 is the "
                         "pre-registered zero-noise regression arm.")
    g6.add_argument("--f2-dd-step", action="store_true",
                    help="F2. DD's step semantics t -> t-1 (it calls "
                         "`set_timesteps(1000)`, `:507`), which keeps ~95 pct "
                         "of the residual. Ours steps 10 -> 0 and keeps ~28 "
                         "pct.")
    g6.add_argument("--f3-per-layer", action="store_true",
                    help="F3. Per-layer offset + confidence heads, a loss term "
                         "per stage, and a DETACH between stages (DD `:379`, "
                         "`:492-497`). ⚠️ ADDS PARAMETERS -- not "
                         "checkpoint-compatible.")
    g6.add_argument("--f4-adaln", action="store_true",
                    help="F4. DD's `ModulationLayer` after every layer's FFN "
                         "(`:229-268`, applied `:337`). Today the timestep is "
                         "added ONCE, to the query. ⚠️ ADDS PARAMETERS.")
    g6.add_argument("--f4-zero-init", action="store_true",
                    help="F4 variant: zero-init the scale/shift so the "
                         "modulation starts as the identity. ⛔ DD's released "
                         "`if_zeroinit_scale` is FALSE (`:233`); this is OUR "
                         "removable-graft variant, and the stamp says which "
                         "ran.")
    g6.add_argument("--f5-emitting-conf", action="store_true",
                    help="F5a. Rank by the SAMPLER's own last-pass confidence "
                         "-- the pass that emitted the fan (DD `:544-552`). "
                         "REMOVES a decoder call; refuses alongside "
                         "--sel-score-emitted.")
    g6.add_argument("--f5-focal", action="store_true",
                    help="F5b. DD's sigmoid FOCAL classification loss "
                         "(gamma 2.0, alpha 0.25, `multimodal_loss.py:"
                         "146-157`) instead of our softmax CE.")
    g6.add_argument("--f6-w-u0-zero", action="store_true",
                    help="F6. DD has ONE reconstruction loss; our --w-u0 "
                         "duplicates it in control space. Sets --w-u0 0 AND "
                         "implies --ack-ddim-no-u0, so the choice is stamped "
                         "as refcv6 F6 rather than as a bypass.")
    g6.add_argument("--f7-samples-per-anchor", type=int, default=1,
                    help="F7. G independent noise draws per anchor -> a "
                         "[B, G*N] fan (DD Tab. 6: N 20 -> 40 is +0.1 PDMS). "
                         "Requires --f7-ack-eval-join.")
    g6.add_argument("--f7-ack-eval-join", action="store_true",
                    help="F7. State that the eval consumer reads the new "
                         "`sel_anchor_id` column rather than `sel_idx`. ⛔ "
                         "Without it G > 1 still REFUSES: taniteval's dumps "
                         "join `sel_idx` as an ANCHOR id and that file is not "
                         "this agent's to change.")
    g6.add_argument("--f8-flat-noise", action="store_true",
                    help="F8. DD's FLAT waypoint-space noise: the affine "
                         "`norm_odo` box (x/56.9, y/46, `:432-441`) plus DD's "
                         "[-1, 1] clamp, giving 0.90 / 0.73 m per waypoint at "
                         "EVERY horizon. Requires --sampler-space metre.")
    g6.add_argument("--f9-assert-vocab", action="store_true",
                    help="F9. Assert the vocabulary is still v0-conditioned "
                         "and 117 anchors. F9 is a NO-CHANGE item, so it is "
                         "enforced rather than described.")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--goal-str", action="store_true",
                    help="train the strategic goal head (needs the lan LABEL "
                         "field — minted WITHOUT building the input pathway, "
                         "the refc_goal_config discipline)")
    # ⭐⭐ STRATEGIC BYPASS (PI 2026-09-06, BINDING). *"remove the strategic
    # layer in the next experiments, feed the nav command to tactical and
    # operative planning ... Include a flag to ignore the strategic layer."*
    ap.add_argument("--no-strategic", action="store_true",
                    help="BYPASS the strategic layer: ctx does not condition "
                         "the decoder (operative), the strategic goal does not "
                         "FiLM the tactical state, the route readout cannot "
                         "re-rank the fan, and the E15 goal point is off. The "
                         "nav command still reaches the OPERATIVE planner "
                         "(inside the measurement vector) and the TACTICAL "
                         "state (E13 nav_to_tac) -- directly, which is the "
                         "point. BYPASS, never delete: every strategic "
                         "parameter stays in state_dict, so an ON build and an "
                         "OFF build load the same checkpoints strictly. "
                         "Implies the strategic goal aux loss is NOT applied "
                         "(nothing consumes g_str), which the config.json "
                         "stamp records.")
    ap.add_argument("--graft-lan", action="store_true",
                    help="supplied-corridor MODEL INPUT — ⛔ NOT part of any "
                         "registered v3 arm (E12); exists for diagnostics only")
    ap.add_argument("--lan-arclengths", type=float, nargs="+",
                    default=[10.0, 20.0, 40.0, 80.0])
    ap.add_argument("--lan-min-lead-m", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU config (CI)")
    ap.add_argument("--preflight", action="store_true",
                    help="build + pin delta + C115 gate + E11 audit + one "
                         "synthetic loss step, then exit")
    # ---- ⭐⭐ REF-C v4 (E11' + E14 + X15), PI 2026-09-03 -----------------
    # ⚠️ There was NO CLI flag for EITHER ego knob before this, and refcv3's
    # config.json stamped NEITHER - so a finished run cannot answer "was the
    # speed masked?" from its own record. v4 adds the flags AND the stamp, and
    # the stamp is the half that still matters in six months.
    ap.add_argument("--ego-state-inject", action="store_true",
                    help="E11prime: feed the MEASURED t0 ego state (v0, "
                         "a_long, yaw_rate, curvature) to the goal/tactical "
                         "path. Implies --ego-valid-channel (refused "
                         "otherwise).")
    ap.add_argument("--echo-base", action="store_true",
                    help="E14: g_tac = kinematic extrapolation + zero-init "
                         "residual, so the trivial solution is free and every "
                         "learned parameter is spent on the scene.")
    ap.add_argument("--ego-valid-channel", action="store_true",
                    help="X15: an explicit 'the ego block is present' bit. "
                         "MEASURED n=781,635 windows: v0 is EXACTLY 0.0 on "
                         "4.4531 pct, so with --ego-dropout 0.5 the input "
                         "reads zero on 52.227 pct of TRAIN samples of which "
                         "only 4.263 pct are a genuinely stopped car (22.5:1) "
                         "- while at eval, dropout being training-only, 100 "
                         "pct are genuine. The same token means something "
                         "23.5x different between train and eval without it.")
    ap.add_argument("--ablate-frames", action="store_true",
                    help="⛔⛔ RIG ONLY - THE DELIBERATE-REGRESSION LEVER. "
                         "Replace the observed window with a scalar constant, "
                         "so the arm keeps every ego channel and NO scene "
                         "information: an echo BY CONSTRUCTION. Required by "
                         "TanitAD_ValidateAIDesign section 2 - if the "
                         "anti-echo gate does not FAIL this arm, a PASS on the "
                         "real arm means nothing. It is stamped in config.json "
                         "as `ablate_frames`; a run carrying it is a GATE "
                         "CONTROL and never a model.")
    ap.add_argument("--ego-dropout", type=float, default=None,
                    help="override core.ego_dropout. Under v4 this is ONE "
                         "draw per sample, shared by the goal path and the "
                         "measurement encoder (never two).")
    # ---- ⭐⭐ E16 (PI 2026-09-01, reaffirmed 2026-09-06): THE MAX-SPEED
    # (map/nav posted-limit) INPUT. Both flags default to the OFF/inert
    # value, because the live 40 k refcv5 run resumes through this file.
    ap.add_argument("--max-speed-input", action="store_true",
                    help="⭐⭐ E16 — FEED the map/nav POSTED-LIMIT CEILING "
                         "as a model INPUT, beside `v0` and the nav token "
                         "and nowhere near a loss. Read from the v8 label "
                         "record's `speed_max_input` block (`v_max_ms`, "
                         "units DECLARED on the wire as m/s; the v7.2 "
                         "release carries it on 0/4,572 records and the "
                         "flag REFUSES there rather than looking switched). "
                         "It stands in for a speed-limit service the way "
                         "`nav_command` stands in for the nav system. "
                         "⚠️ TRAIN/DEPLOY MISMATCH, STATED: the training "
                         "value's provenance is `ego-future` — max of the "
                         "ego's OWN REALISED speed over [t0+2 s, +6 s] — "
                         "while deployment supplies a limit the driver may "
                         "not reach. ⚠️ AND ITS RESIDUAL DEFECT: snapping "
                         "UP from a STOPPED ego reports the LOWEST limit "
                         "(75 %% of intersection clips get <= 30 km/h where "
                         "a map would say 50), so the channel can teach "
                         "'slow ego => low limit'. ⛔ Any result carries a "
                         "SHUFFLE and a WITHHOLD control or it is not "
                         "evidence. Requires --arm hier and --v7-labels. "
                         "Default OFF: no banked arm's recipe changes.")
    ap.add_argument("--max-speed-mode", choices=list(msi.MODES),
                    default=msi.DEFAULT_MODE,
                    help="how the ceiling is encoded for the model. "
                         "`quantized` (DEFAULT) snaps UP to the pinned "
                         "posted-limit ladder (20/30/50/70/80/100/120/130 "
                         "km/h — VALUES from road law, MEMBERSHIP from the "
                         "corpus, NO step is a quantile). `raw` feeds the "
                         "unquantized float, FOR COMPARISON ONLY. "
                         "⚠️ Quantization is COSMETIC as a leak fix — the "
                         "bin plus v0 still recovers the raw value at R^2 "
                         "0.9702 vs 0.8789 for v0 alone, i.e. 75.4 %% of "
                         "the future survives — and REAL as a SEMANTICS "
                         "fix: the raw ceiling sits BELOW the ego's own "
                         "current speed on 34.8 %% of clips (incoherent for "
                         "a ceiling), quantized 8.4 %%. ⛔ INERT without "
                         "--max-speed-input, and refused as such.")
    ap.add_argument("--nav-args", action="store_true",
                    help="⭐ E13b (D-GSTR-1 P3) — FEED the nav command's "
                         "CONTINUOUS ARGS `distance_m` (METRES) and `time_s` "
                         "(SECONDS) alongside its token, plus an explicit "
                         "VALIDITY BIT. MEASURED: the field is written by "
                         "s2_geom_emit_v7.nav_command() on every TURN token "
                         "(train TURN_L median 27.3 m / 7.2 s, TURN_R 36.6 m "
                         "/ 7.4 s), v6/v7f already FEED it through "
                         "NavConditioner.arg_proj, refav1 DISCARDS it, and "
                         "this loader NEVER READ it -- so refcv4b's nav input "
                         "is the stripped BEARING that is separated WORSE by "
                         "+2.3632 m. ⛔ The validity bit is NOT optional: "
                         "NAV_FOLLOW_ROAD carries `args: {}` on 2,897/2,897 "
                         "train records and a silent 0.0 says 'the turn is "
                         "here, now'. ⛔ Requires --nav-from-v7. Normaliser "
                         "is FIT-SPLIT ONLY and is stamped in config.json "
                         "with its units. Default OFF: no banked arm's "
                         "recipe changes.")
    ap.add_argument("--nav-from-v7", action="store_true",
                    help="feed the model's nav_cmd INPUT from the clip's v7.2 "
                         "nav_command token (oracle, provenance ego-future; "
                         "loaded with allow_oracle_nav=True and STAMPED in "
                         "config.json) instead of refb_labels.nav_command "
                         "(v1: net yaw over the next 15-25 s of future poses "
                         "— `follow`+invalid on 94.6 %% of B1 windows, "
                         "E-ARCH-NAVSRC-1 2026-09-02). The same input refav1 "
                         "trains on. Applies to the train AND the held-out "
                         "eval dataset (needs --eval-labels when the eval is "
                         "on). A clip without a record feeds follow + "
                         "nav_valid=False and is COUNTED. Default OFF = "
                         "byte-identical v1 behaviour (the live run resumes "
                         "through this file).")
    # ---- ⭐⭐ E15 (GP-2, 2026-09-06): THE PREDICTED METRIC GOAL POINT ----
    # Pre-registered in `.../Research/2026-09-06-goal-point/PREREG.md` (arms in
    # section 2, the launch command in section 9). Four flags, and every one of
    # them defaults to the OFF/inert value, because the live 40 k refcv5 run
    # resumes through this file.
    ap.add_argument("--goal-point-inject", action="store_true",
                    help="E15: predict a METRIC GOAL POINT (x, y) at "
                         "--goal-point-t seconds from the STRATEGIC CONTEXT "
                         "(vision only) and feed it back through a zero-init "
                         "MLP into the tactical and strategic states. "
                         "⛔ THIS REPLACES E13's categorical nav "
                         "(nav_inject -> False), by pre-registration: the ONE "
                         "variable vs the base arm is the TYPE of the route "
                         "conditioning signal, and running both would make "
                         "the arm a two-variable experiment. WHY: E13's nav "
                         "edge is MEASURED to be a PRESENCE-GATED BIAS "
                         "(inverting the token costs +0.0022 m, NOT "
                         "separated; removing it costs +0.0961 m separated => "
                         "content 2.3 pct), and an ORACLE 3-way command's "
                         "best deterministic decoding of the lateral anchor "
                         "index is 'go straight' for ALL THREE classes. The "
                         "label is the ego's own future path (TRAIN ONLY, "
                         "sanctioned); inference reads vision only.")
    ap.add_argument("--goal-point-geo-prior", action="store_true",
                    help="E15/S7: the goal point also reaches SELECTION, "
                         "through the PARAM-FREE geometric compatibility "
                         "-||bank[:, :, slot] - goal|| / scale at a fixed TIME "
                         "slot, behind ONE zero-init gate (+1 parameter). "
                         "This is the half with no degree of freedom to "
                         "collapse into: move the goal and the ranking moves "
                         "BY CONSTRUCTION. Needs --goal-point-inject.")
    ap.add_argument("--goal-point-t", type=float, default=4.0,
                    help="seconds at which the goal point is taken. ⛔ MUST "
                         "be strictly beyond the 2 s scored horizon -- a goal "
                         "inside it IS THE ANSWER -- and MUST be a model "
                         "horizon (V3_HORIZONS at 10 Hz => 0.5/1/1.5/2/3/4/5/"
                         "6 s). Refused twice: here, and structurally by "
                         "GoalPointConfig, which RAISES. Default 4.0 is the "
                         "MEASURED design point: at t=4 s the goal recovers "
                         "57.1 pct of the LONGITUDINAL and 78.4 pct of the "
                         "lateral selection ceiling, both separated.")
    ap.add_argument("--goal-point-w", type=float,
                    default=GOAL_POINT_WEIGHT_DEFAULT,
                    help="smooth-L1 weight on the NORMALISED goal point. The "
                         "pre-registered launch value is 1.0. ⛔ "
                         "--goal-point-inject with 0 is REFUSED: it builds a "
                         "head, stamps the edge, and trains it on nothing, so "
                         "the section 5 head gate (<= 1.0 m lateral / <= 2.0 "
                         "m range RMS) would fail for a reason that is not "
                         "about the goal FORM.")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    # ⛔ The audit needs the COMMAND LINE, not the namespace: 'did the
    # operator ask for this weight?' cannot be read off a value, because
    # an operator may legitimately pass the default. Absent these the
    # audit reports explicit_source: unavailable rather than guessing.
    args._ew_parser = ap
    args._ew_argv = list(sys.argv[1:] if argv is None else argv)
    if args.preflight:
        raise SystemExit(preflight(args))
    train(args)


if __name__ == "__main__":
    main()
