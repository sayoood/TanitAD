#!/usr/bin/env python3
"""v8F-mini T1-kin: a SELF-ACTION rollout on the refav1 trunk-probe grid (design: ``DESIGN_T1_V8F.md``).

TIER, in the vocabulary of ``Project Steering/EVAL_DOCTRINE.md`` (PI amendment 2026-09-02): **T1 = self-action OPEN loop.** The
model's kinematic state is its OWN executed motion, rebuilt at every replan; the camera stream is NOT re-rendered (that is
T2, not provisioned). So this tool never says "closed loop" and supports only open-loop trajectory-quality claims, not
"the model drives". Every output carries ``scene``, ``nav``, ``limit`` and ``replan_steps`` beside the numbers, because
those four decide how strict the T1 is (DESIGN section 4).

WHAT IS REPLACED by the model's own executed motion (everything the model reads that is built from the ego track):
``p10f`` (8 kinematic features), ``base`` (kd_x path), ``pose_now / pose_t / pose_s`` (odometry), and -- by default -- the
route countdown in ``nav``. WHAT STAYS LOGGED: ``feats`` / ``frame_ok`` (camera), ``limit`` / ``limit_ok`` (a declared
oracle, D-SPEEDBAND-EGO-ECHO-1), the sample-and-hold ages (a schedule). The model is handed ONLY the keys it reads
(``MODEL_KEYS``); a logged ego row after t0 does not exist in the per-window view (NaN), so it cannot be read.

NO MODEL CHANGE: ``V8FMini.forward_infer`` is called as is; no new input, no new weight.

The scoring is ``taniteval/tools/t1_eval.py::analyze`` (four families, S-curve, lag, paired episode-cluster bootstrap),
fed per-episode dumps in its schema -- not re-implemented.

``--dump-ticks DIR`` (PREREG_V8F_HACC_LAT, carried from the hlat build): one row per (window, tick) of the MODEL's rollout -- O's plan before the post-map (tau), the prior (P), the plan the
model issued, the measured ego speed v0 at the tick, the logged future in the tick's EXECUTED frame (GT, "rejoin the human's path") and the window / episode / clip ids.
``--dump-heads`` (needs ``--dump-ticks``; T1-HEADS) adds, to the SAME rows, what the S head and the T head emitted in the very ``forward_infer`` call that produced the plan -- exactly the
tensors ``v8f_mini_eval.py``'s T0 eval scores (``checkpoint_metrics`` / ``tactical_metrics`` / ``self_consistency``): ``s_ckpt`` (+ its NOW-frame and window-origin-frame forms),
``t_goal_prob``, ``t_lat_logits``, ``t_lon_logits``, ``t_cons_pred``, and the poses that frame them. Both flags default OFF; absent, the tool's output is unchanged bit for bit
(control C-T, and ``tests/test_v8f_t1_heads.py``). The P10 floor has no heads and is never recorded.

``--smooth {off,sg72}`` (SMOOTH-T1, default off): ``sg72`` applies PREREG_V8F_SMOOTH's operator, verbatim, to the model's plan at EVERY replan, before the executor:
``S(plan) = P + SG(plan - P)`` with ``P`` = the prior the model computed in the very ``forward_infer`` call (``out["p10"]``, the P10 at the EXECUTED state) and SG = a
Savitzky-Golay filter along time, per coordinate, window 7, order 2, ``mode="interp"`` (``scipy.signal.savgol_filter``). The smoothed plan is what the executor commits, what ``ol``
(the tick-0 plan) holds and what ``--dump-ticks`` calls ``plan``; with ``--dump-ticks`` the ticks file also carries ``plan_raw`` (the model's plan before S) and ``smooth``.
The P10 floor is never smoothed. Absent the flag (or ``off``) the output is unchanged bit for bit (``tests/test_v8f_t1_smooth.py``).

``--feasible {off,box}`` (KAPPA-T1, H-KAPPA-T1-1, default off): ``box`` projects the plan, AFTER ``--smooth``, onto the envelope of
``tanitad.refs.feasible_decode.project_feasible`` verbatim (box only: ``mu=None``, ``a_max`` = ``A_MAX_MPS2`` 4.0, ``kappa_max`` = ``KAPPA_MAX_1PM`` 0.2, the module's
default margin and stop epsilons, ``clamp_entry=False``) on the origin-prepended plan at the plan's own grid ``dt`` = ``DT`` (0.2 s), at EVERY replan, before the executor.
The projected plan is what the executor commits, what ``ol`` holds and what ``--dump-ticks`` calls ``plan``; the ticks file also carries ``plan_raw`` (the model's plan) and
``plan_s`` (the plan after ``--smooth``, before the projection). The P10 floor is never projected. Absent the flag (or ``off``) the output is unchanged bit for bit
(``tests/test_v8f_t1_feasible.py``).

``--feasible {kbox,abox,kabox}`` (FEAS2-T1, H-FEAS2-T1-1) split ``box`` into its two clamps, same place (after ``--smooth``, every replan, before the executor), same ``dt``, ``mu=None``:
``kbox`` = ``project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, clamp_entry=False)`` (the curvature clamp alone); ``abox`` = ``project_feasible(kappa_max=inf, a_max=A_MAX_MPS2,
clamp_entry=True, v0=the tick's measured ego speed batch["p10f"][:, -1, 0])`` (the accel clamp and the entry-speed bound alone); ``kabox`` = ``kbox`` then ``abox``. ``box`` is unchanged
bit for bit from KAPPA-T1; ``off`` is unchanged bit for bit (``tests/test_v8f_t1_feas2.py``).

``--histswap {off,logged}`` (HISTSWAP, H-HISTSWAP-1; this file is ``v8f_t1_eval_histswap.py``): ``logged`` hands the model, at EVERY tick, its ego-history input from the LOGGED clip arrays (rows <= the tick's j;
every key ``ds.item`` builds from the ego rows -- ``p10f``, ``base``, ``kin3``, ``pose``, ``pose_now`` / ``pose_t`` / ``pose_s`` -- and the causal speed bucket) instead of from the executed ones; the plan is
still committed to the EXECUTED state, and ``nav`` (dead-reckoned from the executed motion), the camera stream, the ages, ``--smooth`` and the P10 floor are untouched. It is an ORACLE diagnostic (the logged
history after t0 is the human's, not deployable). Under it the ticks file's ``v0`` stays the EXECUTED measured speed (the quantity ``feas2`` / ``kappa`` readers define) and ``v0_in`` is the speed the network read;
the S head's ``s_pose`` / ``t_pose`` are the logged poses it read. ``off`` (default) and the flag absent are bit-identical to the FEAS2 tool; it refuses to combine with ``--feasible`` (``tests/test_v8f_t1_histswap.py``).

``--histswap {cv_exec,cv_logged}`` (HISTSWAP-LF, H-HISTSWAP-LF-1; this file is ``v8f_t1_eval_histswap_lf.py``; ``off`` and ``logged`` are the HISTSWAP tool's behaviour bit for bit): the model's ego-history
input at EVERY tick is a CONSTANT-TWIST history -- speed ``v`` and yaw rate ``omega`` held over every earlier row (zero acceleration, no trend), anchored at the EXECUTED pose of the tick -- built through
the programme's one spelling (``yaw_kin_from_track`` / ``ego_channels_native`` / ``kin_features`` / ``kdx_base`` on a synthetic native track) for the same keys ``logged`` swaps (``p10f``, ``base``, ``kin3``,
``pose``, ``pose_now`` / ``pose_t`` / ``pose_s``, the causal speed bucket); ``nav`` (dead-reckoned), the camera stream, the ages, ``--smooth`` and the P10 floor are untouched, and the plan is committed to the
EXECUTED state. ``cv_exec`` (HCV, deployable): ``v`` = the EXECUTED measured speed at the tick. ``cv_logged`` (HLVL, ORACLE, level only): ``v`` = the LOGGED measured speed at the tick (the human's level, no
trend). ``omega`` is the EXECUTED yaw rate in both. Needs ``--limit causal``; refuses ``--feasible``. The ticks file's ``v0`` stays the EXECUTED speed, ``v0_in`` is the speed the network read.

``--scene hold_now`` (FRAMEFREEZE, the frame-freeze control of DRAFT_RECOVERY_CARD_v2.md section 8.3; this file is ``v8f_t1_eval_framefreeze.py``; ``logged`` and ``frozen`` are the HISTSWAP-LF tool's behaviour bit for
bit): in ``build_item``, right after the item is built, every row of ``it["feats"]`` is replaced by the CURRENT frame's row, repeated (``feats[k] := feats[-1]`` for all ``hmax`` rows), so the ``hist`` frames O reads
(and the frames T and S read) are the tick-j frame repeated and no frame MOTION is available to the network. ``frame_ok`` is unchanged. The camera is still logged (``strict.camera`` stays False) and the run header
gets ``camera_motion: false``, an open-channel note (the current frame still depicts the human's position at the tick) and a ``frame_freeze`` receipt (items frozen, max row deviation, unequal elements: the
every-batch check). ``--freeze-card-ref`` carries the RECOVER-FT card's frozen sha line into that header. Needs the cached per-frame ``feats`` (refuses otherwise).

``--scene reindexed`` (REINDEXED, a longitudinal-only approximate re-perception; this file is ``v8f_t1_eval_ri.py``; ``logged``, ``frozen`` and ``hold_now`` are the FRAMEFREEZE tool's behaviour bit for bit).
WHY: T1-kin integrates the model's own actions but feeds the LOGGED camera frames on the human's clock. When the model lags the human (own speed far below the logged speed) the frames show positions the
car has not reached and advance at the human's speed, so a model that reads motion from vision copies the human's speed from the camera (a leak of the evaluation target). ``reindexed`` makes the frames follow
the model's OWN progress along the logged path.
RULE (``build_item``, after the item is built; ``reindex_item`` / ``reindex_rows`` / ``project_taus``): the item's ``feats`` / ``frame_ok`` / ``hmax`` rows end at the tick's cache index j; row r has time index
i = j - hmax + 1 + r. Rows with i <= t0 (the window's measured past) stay the LOGGED rows, bit for bit (checked on every item, not asserted by construction: the receipt compares the bytes). For a row with
i > t0: p_own(i) = the model's executed (x, y) at cache step i; it is projected onto the logged polyline ``pose_grid[:, :2]`` of the clip (nearest segment, the lateral offset is ignored and reported); tau* = the
logged cache index whose along-track arc length is nearest to the projected point (ties and a dwell of less than RI_DWELL_M go to the LATEST index); tau(i) = min(i, tau*); then tau is made non-decreasing over the
item's rows (running maximum; the past rows enter with their own index). The row takes the cached frame at tau(i) (``feats`` of the logged clip) and the ``frame_ok`` the item builder gives that index
(``tau >= 0`` and a finite logged ``p10f`` row, or ``tau < 1``); a row with tau(i) == i is left exactly as the item builder built it. A frame from AFTER the row's own time is never used (tau <= i: the receipt counts
``future_frame_rows``); a model AHEAD of the human sees the logged frame at i (counted ``rows_clamped``). The static scene geometry is consistent with the own position; dynamic agents are shown at the human's time i;
the lateral offset is ignored. This is an approximation of re-perception, NOT T2.
EXECUTED TRACK vs CACHE TIME (confirmed in the source tool v8f_t1_eval_framefreeze.py sha256 0721fc01..., line numbers are that file's): the clip's native 10 Hz track has 2 rows per 0.2 s cache step, row 2*i is
cache step i (``check_track_matches_grid``, v8f_mini_data.py:309: the even rows equal the cache ``pose_grid``). ``WindowState.__init__`` copies the rows <= 2*t0 of the logged track (line 264) and the rows <= t0 of the
ego arrays; ``commit_plan`` writes the 10 Hz rows 2*j+1 .. 2*(j+m) (line 294; the even row of step j+s is the plan's knot s, ``hermite_frames``) and ``_finish`` sets ``d["pose"][j] = track[2*j]`` for every executed
step (line 316) and ``self.j = j_new`` (line 325). ``rollout`` runs ticks n = 0 .. H//m - 1 (line 633), builds the items at ``st.j = t0 + n*m`` (line 637) BEFORE committing that tick's plan (line 649), so every row
i <= j of the item has an executed pose in ``st.d["pose"][i]`` (rows <= t0: the logged pose). p_own(i) = ``st.d["pose"][i][:2]``, clip-local metres, the frame of ``pose_grid``.
RECEIPT (``header["scene_reindex"]``): items, rows (= items x hmax), rows_logged (i <= t0), rows_reindexed (tau < i), rows_on_time (tau* == i), rows_clamped (tau* > i), rows_monotone_raised, future_frame_rows (must be
0), the lag i - tau distribution in cache steps (mean, p50, p90, max) over all rows with i > t0 and over the reindexed rows, the lateral offset of the projection in metres, for the MODEL's rollout (top level) and for
the P10 floor's rollout (``floor``; it reads no frames), and ``past_rows_bit_identical`` (true iff the compared past rows are > 0 and none changed). The ticks file carries ``scene``.

``--level-floor {off,on}`` (LEVEL-FLOOR, PREREG_LEVELFLOOR_A3 sha256 977a785e..., default off; this file is the RI tool + ``apply_lf_patch.py``): ``on`` applies the floor operator F (``level_floor.py``, next to this tool)
to the plan at EVERY replan, inside ``model_planner``'s ``plan(batch)``: raw -> S (``--smooth``) -> F -> the projection (``--feasible``). F runs AFTER S in every branch (the floor branch sits before the unpatched early returns; A3_RULING_1): ``--smooth sg72`` with no ``--feasible`` (the banked bases' flags = the bare arm) is raw -> S -> F; ``--smooth sg72 --feasible kbox`` (the composite) is raw -> S -> F -> kbox; ``--smooth off --feasible off`` is raw -> F. ``--floor-kappa K`` (float in [0, 1], required with on, refused with off) is the dose;
``--floor-ckpt PATH`` (required with on, refused with off) is the head checkpoint (the LLR checkpoint: its cfg must have ``level_head`` AND ``lead_head`` on and its data interface equal to the planner's): a SECOND model instance
encodes the SAME tick batch itself (``floor_heads``: the adapter, ``lvl_head`` and ``lead_head`` only; its O / S / T are not run; the first replan is checked against that model's own ``forward_infer`` outputs
``level_v`` / ``lead_logit`` / ``lead_gap``). F reads v_hat (m/s), pi = sigmoid(presence logit), g = 40 * gap_hat (m) from the floor heads and v0 = ``batch["p10f"][:, -1, 0]`` (the measured ego speed) and nothing else;
no nav, route, limit, label or situation-classifier output enters it. With ``--dump-ticks`` the ticks file carries ``lf_bind`` / ``lf_cap`` (bool) and ``lf_vf`` (NaN = no floor) / ``lf_vhat`` / ``lf_pi`` / ``lf_gap`` (m) /
``lf_lift`` (the 0-3 s mean plan speed after F minus before, m/s) per replan, plus ``lf_bind_k10`` (instrument: F changed a knot among the first 10 = the 2 s the read executes), and ``plan_raw`` (the model's plan before S and F) even when ``--smooth off --feasible off``. The result JSON header carries ``level_floor``,
``floor_kappa``, ``floor_ckpt``, ``floor_ckpt_sha256`` and ``level_floor_detail`` (the operator, its constants, the bind / cap counts, the floor model's batch keys). ``--histswap`` != off is refused with the floor
(F's v0 would be the logged speed).

``--floor-vhat-source {heads,logged}`` (LFO, PREREG_LEVELFLOOR_A4 sha256 8bfbc386..., only with ``--level-floor on``; default ``heads`` = the LF tool bit for bit): ``logged`` replaces the floor's level input v_hat with the LOGGED route
speed v_log of the tick, a CEILING diagnostic that is NOT admissible at inference (a privileged channel). v_log = the path length of knots 1..6 of the tick's ``gt`` (the logged route in the executed ego frame, ``gt_in_frame``) / 1.0 s,
the reader's own ``logged_speed`` (``lf_read.py``); it is computed at the replan BEFORE F from the same ``gt_in_frame`` the tick dump uses (``logged_route_speed`` / ``make_logged_oracle``). A tick with no complete logged route
(``gt_ok`` False) has v_hat NaN and therefore NO floor. The presence and gap heads still come from the floor model (the lead cap is unchanged). ``logged`` is refused without ``--level-floor on``, and an explicit ``heads`` is refused with
the floor off.
``--floor-vhat-noise SIGMA`` / ``--floor-vhat-noise-seed N`` (LF5, PREREG_LEVELFLOOR_A5 sha256 2aac1dbe..., only with ``--floor-vhat-source logged``; refused otherwise): v_hat := v_log * exp(SIGMA * eps), eps ~ N(0,1) from ONE
``numpy.random.default_rng(N)`` per rollout, one vector draw per floor call (per replan of a chunk) in replan order, taken whether or not a floor applies; SIGMA > 0 needs the seed. Neither flag = the LFO tool bit for bit. With the
noise path active the ticks file gains ``lf_vlog`` (v_log) and ``lf_eps`` (the draw), and the header records ``floor_vhat_noise`` and ``floor_vhat_noise_seed``.
``--floor-lift-cap DELTA`` (LF6, PREREG_LEVELFLOOR_A6 sha256 fbf5de02..., float >= 0, needs ``--level-floor on``): the bounded-lift floor F_B, v_f(tau) = min(kappa * v_hat with the lead cap, v0 + 1.0 * tau, v_O(tau) + DELTA), v_O = O's planned speed after S (chord / 0.2 s);
s' - s <= DELTA * t at every knot. Absent = the LF5 tool bit for bit. The ticks gain ``lf_dterm``, ``lf_dterm_knots``, ``lf_s``, ``lf_sp`` and ``floor_lift_cap``, the header records ``floor_lift_cap``. LF6 also omits the ticks key ``floor_vhat_noise_seed`` when the seed is None.
The result JSON header records ``floor_vhat_source`` and, for ``logged``, ``"label": "CEILING"``; the ticks file gains ``lf_vhat_heads`` (what the heads would have said on the same tick; ``lf_vhat`` is then v_log).

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402
from tanitad.refs.refcv6_max_speed import speed_max_bin  # noqa: E402  (the sidecar's own containing-window rule)

DT = D.DT                   # 0.2 s: the cache step
H = D.HORIZON                # 30 steps = 6 s
EPS_M = 1e-3                 # a tangent shorter than 1 mm per 0.2 s interval holds the previous heading (plan_kin_t's convention)
#: the ONLY physical bound the executor applies: no executed 0.2 s step is longer than V_ENV_MS * 0.2 s (60 m/s = 216 km/h, beyond
#: any road speed in the corpus). It exists so that a diverging window stays finite instead of overflowing to NaN and corrupting a
#: whole run; it is COUNTED (``WindowState.clipped``) and reported, never silent. A sane plan never reaches it.
V_ENV_MS = 60.0
#: the allowlist: the model is handed EXACTLY the keys forward_infer reads. ``act`` and ``pose`` are in INFERENCE_INPUT_KEYS but
#: forward_infer never reads them (``act`` only inside ``loss``); dropping them makes a future read a loud KeyError.
NO_READ = frozenset({"act", "pose"})
MODEL_KEYS = frozenset(M.INFERENCE_INPUT_KEYS) - NO_READ
NAV_MODES = ("deadreckon", "logged")
SCENE_MODES = ("logged", "frozen", "hold_now", "reindexed")
#: REINDEXED: a stretch of the logged path that moves less than this (metres of arc length) between two cache steps is a dwell (a stop with sensor jitter); tau* then goes to the LATEST index of the dwell.
RI_DWELL_M = 0.05
#: REINDEXED: per-run receipt of the reindexed apply line (a fresh process per run): one block per rollout role (the model's ticks / the P10 floor's ticks, told apart by ``stats is not None`` in ``build_item``)
#: plus the past-row byte comparison over both. ``lag_hist`` = {lag in cache steps: row count} over the rows with i > t0.
RI_STATS = {}


def ri_reset():
    RI_STATS.clear()
    for role in ("model", "floor"):
        RI_STATS[role] = {"items": 0, "rows": 0, "rows_logged": 0, "rows_reindexed": 0, "rows_on_time": 0, "rows_clamped": 0, "rows_monotone_raised": 0, "future_frame_rows": 0, "lag_hist": {}, "offset_m": []}
    RI_STATS["past"] = {"rows_checked": 0, "rows_changed": 0}


ri_reset()
#: FRAMEFREEZE: per-run receipt of the hold_now apply line (a fresh process per run): items frozen, max |row - tick-j frame| over finite elements, count of elements that differ from the tick-j frame as built (NaN == NaN counts as equal).
FREEZE_STATS = {"items": 0, "max_row_dev": 0.0, "n_unequal_elements": 0}
LIMIT_MODES = ("logged", "off", "causal", "causal_all")
#: PREREG_V8F_T1KIN_CAUSALBUCKET: ``causal`` reads the max MEASURED ego speed over the last CAUSAL_STEPS + 1 rows [j - 20, j] (4 s; logged
#: up to t0, own executed after), ``causal_all`` over [0, j]; both snapped by ``speed_max_bin``. Rows after j are NaN by WindowState's
#: construction, and the slice ends at j anyway.
CAUSAL_STEPS = 20
CAUSAL_STATS = {}            # per run: ticks, ticks with no finite past, agreement with the logged bucket, 4 x 4 confusion (logged x causal)
SCHEDULES = ("periodic", "fresh")
#: SMOOTH-T1 (PREREG_V8F_SMOOTH, sha256 a2f19e48...): the operator is fixed there and is NOT tuned here -- window 7 steps (1.4 s), polynomial order 2, ``mode="interp"``, along time, per coordinate.
SMOOTH_MODES = ("off", "sg72")
#: KAPPA-T1 (PREREG H-KAPPA-T1-1, gate G-KAPPA outcome (a)): the envelope is feasible_decode's, NOT tuned here.
FEASIBLE_MODES = ("off", "box", "kbox", "abox", "kabox")
#: FEAS2-T1: the header's operator text per mode (``box`` is KAPPA-T1's, verbatim).
FEASIBLE_OPERATORS = {
    "box": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=A_MAX_MPS2, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
    "kbox": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
    "abox": "tanitad.refs.feasible_decode.project_feasible(origin + plan, v0=p10f[:, -1, 0], dt=DT, a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
    "kabox": "abox(kbox(plan)): kbox = project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False), then abox = project_feasible(v0=p10f[:, -1, 0], a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
}
SG_WINDOW, SG_ORDER = 7, 2
#: HISTSWAP (PREREG_HISTSWAP_FROZEN.md): ``logged`` = the model's ego-history input is the LOGGED history at the tick; ``off`` = the executed one (FEAS2's behaviour, bit for bit).
HISTSWAP_MODES = ("off", "logged", "cv_exec", "cv_logged")
#: HISTSWAP-LF (PREREG_HISTSWAP_LF_FROZEN.md): a constant-twist history at the executed (HCV) or the logged (HLVL) current speed.
CV_MODES = ("cv_exec", "cv_logged")
#: rows [j - CV_ROWS, j] of the item builder's view are rebuilt: ``ds.item`` reads j - (hmax - 1) .. j, j - a_s (a_s <= 6) and the causal bucket j - CAUSAL_STEPS (20) .. j; ``cv_view`` asserts the bound.
CV_ROWS = 32


def hist_mode_of(x) -> str:
    """The history mode of a ``hist_swap`` argument: False / None / "off" -> off; True / "logged" -> logged (HISTSWAP's bool contract); "cv_exec" / "cv_logged" -> itself; anything else raises."""
    if x is None or x is False or x == "off":
        return "off"
    if x is True or x == "logged":
        return "logged"
    if x in CV_MODES:
        return str(x)
    raise ValueError(f"hist_swap must be one of {HISTSWAP_MODES} (or a bool), got {x!r}")
TIER_DOCTRINE = ("EVAL_DOCTRINE.md, PI amendment 2026-09-02: T0 and T1 are both OPEN LOOP; T2 is the only closed loop and is not "
                 "provisioned. T1 here = the ego state is the model's OWN executed motion; the camera stream is logged. "
                 "It supports open-loop trajectory-quality claims only, never 'drives'.")


# ------------------------------------------------------------------------------------------------ geometry
def plan_to_world(plan, pose) -> np.ndarray:
    """``plan`` [N, 2] in the ego frame at ``pose`` = (x, y, yaw, ...) -> world [N, 2]."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    plan = np.asarray(plan, np.float64)
    return np.stack([pose[0] + c * plan[:, 0] - s * plan[:, 1], pose[1] + s * plan[:, 0] + c * plan[:, 1]], 1)


def hermite_frames(k_prev, knots, m: int, psi0: float):
    """Fill the 10 Hz raw frames of ``m`` executed cache steps from the plan's 0.2 s knots (DESIGN section 3, step 5).

    ``k_prev`` [2] = the previous executed knot (step j-1); ``knots`` [m+1.., 2] with ``knots[0]`` = the current knot (step j) and
    ``knots[i]`` = the plan's knot j+i. Cubic Hermite with Catmull-Rom tangents (a backward difference at the plan's last knot).
    Returns ``(xy [2m, 2], psi [2m], v [2m])`` for raw frames 2j+1 .. 2j+2m: the odd frame of interval i is the Hermite midpoint, the
    even frame the knot itself. Heading = tangent angle (held while the tangent is < ``EPS_M``); speed = |tangent| / 0.2 s."""
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)])      # K[i] = knot j+i-1
    nk = len(K)
    if nk < m + 2:
        raise ValueError(f"need at least m+1={m + 1} knots, got {nk - 1}")

    def tang(i):
        return 0.5 * (K[i + 1] - K[i - 1]) if i + 1 < nk else K[i] - K[i - 1]

    xy, psi, v = [], [], []
    cur = float(psi0)
    for i in range(1, m + 1):                                                    # interval K[i] -> K[i + 1]
        t0, t1 = tang(i), tang(i + 1)
        mid = 0.5 * (K[i] + K[i + 1]) + (t0 - t1) / 8.0
        dmid = 1.5 * (K[i + 1] - K[i]) - 0.25 * (t0 + t1)
        for pos, d in ((mid, dmid), (K[i + 1], t1)):
            nrm = float(np.hypot(d[0], d[1]))
            if nrm >= EPS_M:
                cur = math.atan2(d[1], d[0])
            xy.append(pos)
            psi.append(cur)
            v.append(nrm / DT)
    return np.asarray(xy), np.asarray(psi), np.asarray(v)


def envelope(plan):
    """Bound the plan's 0.2 s steps by ``V_ENV_MS * DT`` (direction kept) and zero a non-finite step. Returns ``(plan, hit)``.
    The origin is prepended, so step 0 is |plan[0]| itself. A plan inside the envelope is returned UNCHANGED (bit-identical)."""
    plan = np.asarray(plan, np.float64)
    steps = np.diff(np.vstack([np.zeros((1, 2)), plan]), axis=0)
    bad = ~np.isfinite(steps).all(1)
    nrm = np.linalg.norm(np.where(bad[:, None], 0.0, steps), axis=1)
    over = nrm > V_ENV_MS * DT
    if not (bad.any() or over.any()):
        return plan, False
    steps = np.where(bad[:, None], 0.0, steps)
    steps = steps * np.where(over, V_ENV_MS * DT / np.maximum(nrm, 1e-12), 1.0)[:, None]
    return np.cumsum(steps, axis=0), True


def exec_in_ego(track, t0: int) -> np.ndarray:
    """The executed cache-step knots t0+1 .. t0+30 in the t0 ego frame, by the SAME function that builds the GT (``gt_waypoints``)."""
    return fk.gt_waypoints(np.asarray(track, np.float64)[0::2], np.array([t0]), np.arange(1, H + 1))[0]


def ages_for(step: int, schedule: str, k_t: int = 3, k_s: int = 6):
    """Sample-and-hold ages (cache steps) at ``step`` = steps since t0. periodic: T ticks every ``k_t`` steps, S every ``k_s``
    (defaults 3 / 6: inside the trained support max_age_t=2, max_age_s=6 and nested as the trainer requires a_s >= a_t)."""
    if schedule == "fresh":
        return 0, 0
    if k_s % k_t:
        raise ValueError("k_s must be a multiple of k_t (S ticks no later than T)")
    return step % k_t, step % k_s


# ------------------------------------------------------------------------------------------------ the per-window state
def cv_motion_track(pose_j, v: float, omega: float, n_back: int) -> np.ndarray:
    """``[2 * n_back + 1, 4]`` native (0.1 s) rows (x, y, yaw, v) of the CONSTANT-TWIST history that ENDS at ``pose_j`` = (x, y, yaw): speed ``v`` and yaw rate ``omega`` (rad/s) held on every earlier
    row (the exact circular arc; a straight line when |omega| < 1e-9). The last row is ``pose_j`` with ``v``; the earlier rows are the arc travelled backwards."""
    n = 2 * int(n_back) + 1
    x0, y0, p0 = float(pose_j[0]), float(pose_j[1]), float(pose_j[2])
    tau = (np.arange(n, dtype=np.float64) - (n - 1)) * D.DT_NATIVE                  # seconds before the tick, <= 0
    psi = p0 + omega * tau
    if abs(omega) < 1e-9:
        x, y = x0 + v * tau * math.cos(p0), y0 + v * tau * math.sin(p0)
    else:
        x, y = x0 + (v / omega) * (np.sin(psi) - math.sin(p0)), y0 + (v / omega) * (math.cos(p0) - np.cos(psi))
    return np.stack([x, y, psi, np.full(n, float(v))], 1)


def channels_from_track(sl) -> dict:
    """The rows ``WindowState._finish`` writes for steps 1 .. n of a native slice ``sl`` of ``2 n + 1`` rows, by the SAME calls: ``kin3``, ``ch1`` (omega), ``p10f``, ``base``, ``pose`` (4 columns), ``a``, ``v``.
    (Re-spelled here so that ``_finish`` stays byte-identical to the HISTSWAP tool; ``tests/test_v8f_t1_histswap_lf.py`` pins the two to equal exactly on a real executed slice.)"""
    sl = np.asarray(sl, np.float64)
    kin, om = D.yaw_kin_from_track(sl)
    ch = D.ego_channels_native(sl)[0::2]
    n = len(kin) - 1
    memo = {}
    base = []
    for s in range(1, n + 1):
        key = kin[s].tobytes()
        if key not in memo:
            memo[key] = D.kdx_base(kin[s:s + 1], D.KIN_UNITS["yaw"])[0]
        base.append(memo[key])
    return {"kin3": kin[1:], "ch1": np.nan_to_num(om[1:], nan=0.0), "p10f": np.stack([fk.kin_features(kin[s:s + 1])[0].astype(np.float32) for s in range(1, n + 1)]),
            "base": np.stack(base), "pose": sl[0::2][1:], "a": ch[1:, 1], "v": sl[0::2][1:, 3]}


class WindowState:
    """One window's view of its clip, with the ego arrays logged UP TO t0 and own-executed after.

    The constructor copies ONLY rows <= t0 of the logged ego arrays and of the native track (the future rows are NaN), so this
    object cannot leak the logged ego after t0 whatever the item builder does. ``d`` is a ClipStore-shaped dict that
    ``V8FMiniWindows.item`` reads unchanged. The logged track is passed to ``commit_logged`` only by the CONTROL executor."""

    EGO = ("pose", "kin3", "p10f", "base", "a", "ch1", "v")

    def __init__(self, base: dict, track, nav_rows, ci: int, t0: int, nav_mode: str = "deadreckon", hist_swap=False):
        if nav_mode not in NAV_MODES:
            raise ValueError(f"nav_mode must be one of {NAV_MODES}")
        T = len(base["pose"])
        if not bool(base["gt_ok"][t0]):
            raise ValueError("window has no complete 6 s future (gt_ok False)")
        self.ci, self.t0, self.T, self.j, self.nav_mode = int(ci), int(t0), T, int(t0), nav_mode
        tr = np.full(np.asarray(track).shape, np.nan, np.float64)
        tr[:2 * t0 + 1] = np.asarray(track, np.float64)[:2 * t0 + 1]
        self.track = tr
        d = dict(base)                                                   # shallow: feats / gt / s_h stay shared and logged
        for k in self.EGO:
            arr = np.array(base[k], copy=True)
            arr[t0 + 1:] = np.nan
            d[k] = arr
        self.nav = np.array(nav_rows, np.float32, copy=True)
        self.nav0 = self.nav[t0].copy()
        if nav_mode == "deadreckon":
            self.nav[t0 + 1:] = np.nan
        self.arc = {t0: 0.0}
        self.d = d
        self.scene0 = None
        self.clipped = 0                  # ticks whose plan had a step beyond the envelope (or a non-finite value)
        self.hist_mode = hist_mode_of(hist_swap)
        self.hist_swap = self.hist_mode != "off"
        self.base_log = base                                              # the LOGGED clip (read-only, never written): the source of the logged speed level (cv_logged) and of ``logged``
        self.d_in = base if self.hist_mode == "logged" else d             # the dict the ITEM BUILDER reads: the executed view ``d`` (off), the LOGGED clip ``base`` (logged), or, under cv_*, the per-tick :meth:`cv_view`

    # -- committing executed motion ------------------------------------------------------------------------------
    def commit_plan(self, plan_ego, m: int):
        """Execute the first ``m`` knots of ``plan_ego`` [30, 2] (ego frame at NOW), perfect tracking."""
        j = self.j
        pose = self.d["pose"][j]
        plan_ego, hit = envelope(plan_ego)
        self.clipped += int(hit)
        W = plan_to_world(plan_ego, pose)
        knots = np.vstack([pose[None, :2], W])
        xy, psi, v = hermite_frames(self.track[2 * (j - 1), :2], knots, m, float(pose[2]))
        sl = slice(2 * j + 1, 2 * (j + m) + 1)
        self.track[sl, 0:2], self.track[sl, 2], self.track[sl, 3] = xy, psi, v
        self._finish(j, j + m)

    def commit_logged(self, logged_track, m: int):
        """CONTROL executor: the 'own' motion IS the logged motion (DESIGN C1). Never used by a result."""
        j = self.j
        sl = slice(2 * j + 1, 2 * (j + m) + 1)
        self.track[sl] = np.asarray(logged_track, np.float64)[sl]
        self._finish(j, j + m)

    def _finish(self, j_old: int, j_new: int):
        sl = self.track[2 * j_old: 2 * j_new + 1]
        kin, om = D.yaw_kin_from_track(sl)                              # the ONE spelling, on the appended slice
        ch = D.ego_channels_native(sl)[0::2]
        d = self.d
        for s in range(1, j_new - j_old + 1):
            j = j_old + s
            d["kin3"][j] = kin[s]
            d["ch1"][j] = np.nan_to_num(om[s], nan=0.0)
            d["p10f"][j] = fk.kin_features(kin[s:s + 1])[0].astype(np.float32)
            d["base"][j] = D.kdx_base(kin[s:s + 1], D.KIN_UNITS["yaw"])[0]
            d["pose"][j] = self.track[2 * j]
            d["a"][j] = ch[s, 1]
            d["v"][j] = self.track[2 * j, 3]
            self.arc[j] = self.arc[j - 1] + float(np.linalg.norm(d["pose"][j, :2] - d["pose"][j - 1, :2]))
            if self.nav_mode == "deadreckon":
                row = self.nav0.copy()
                if row[4] > 0:                                           # route progress from OWN motion; token / valid held from t0
                    row[3] = np.clip(row[3] - self.arc[j] / 100.0, 0.0, 3.0)
                self.nav[j] = row
        self.j = j_new

    def cv_view(self, limit: str = "causal") -> dict:
        """HISTSWAP-LF: the dict the item builder reads at this tick under ``cv_exec`` / ``cv_logged`` -- a copy of the EXECUTED view ``d`` whose rows [j - CV_ROWS, j] of every ego key are the CONSTANT-TWIST
        history at speed ``v`` (cv_exec: the executed ``p10f[j][0]``; cv_logged: the LOGGED ``p10f[j][0]``) and yaw rate ``omega`` = the executed ``ch1[j]``, anchored at the executed pose, built by
        :func:`channels_from_track` on :func:`cv_motion_track`. Nothing is written to ``d`` or to the logged clip."""
        if self.hist_mode not in CV_MODES:
            raise RuntimeError("cv_view is defined for cv_exec / cv_logged only")
        if limit == "causal_all":
            raise ValueError("REFUSED: cv_exec / cv_logged are defined with --limit causal (the bucket reads rows [j - 20, j])")
        j, d = self.j, self.d
        src = d if self.hist_mode == "cv_exec" else self.base_log
        v = float(src["p10f"][j][0])
        om = float(np.nan_to_num(d["ch1"][j], nan=0.0))
        pj = np.asarray(d["pose"][j], np.float64)
        if not (math.isfinite(v) and bool(np.isfinite(pj[:3]).all())):
            raise ValueError(f"{self.hist_mode}: the current speed / pose is not finite at step {j} (v={v!r})")
        lo = max(1, j - CV_ROWS)
        assert CV_ROWS >= max(CAUSAL_STEPS, 9, 6), "CV_ROWS must cover every row the item builder reads"
        ch = channels_from_track(cv_motion_track(pj, v, om, j - (lo - 1)))
        out = dict(d)
        for k in self.EGO:
            out[k] = np.array(d[k], copy=True)
        out["kin3"][lo:j + 1], out["ch1"][lo:j + 1], out["p10f"][lo:j + 1], out["base"][lo:j + 1] = ch["kin3"], ch["ch1"], ch["p10f"], ch["base"]
        out["pose"][lo:j + 1], out["a"][lo:j + 1] = ch["pose"], ch["a"]
        out["pose"][j] = d["pose"][j]                                     # the anchor IS the executed pose (the cameras, the nav and every target live in this frame)
        out["v"][max(0, j - CV_ROWS):j + 1] = v                           # the causal bucket reads d["v"][max(0, j - 20): j + 1]
        return out

    def executed_ego(self) -> np.ndarray:
        return exec_in_ego(self.track, self.t0)


class _OneClipStore:
    """A ClipStore stand-in that serves the dict of whichever window is being built."""

    def __init__(self):
        self.d = None

    def get(self, clip):
        return self.d


def hold_now_feats(it: dict) -> np.ndarray:
    """FRAMEFREEZE: the item's per-frame feature rows with EVERY row replaced by the current frame's row (the last one): ``[hmax, ...]`` in, same shape and dtype out. Refuses an item without cached ``feats``."""
    f = it.get("feats")
    if f is None:
        raise RuntimeError("REFUSED: --scene hold_now needs the cached per-frame feats (absent under --encoder lora): a flag that silently freezes nothing is worse than none")
    return np.repeat(f[-1:], f.shape[0], axis=0)


def freeze_dev(f, cur) -> tuple:
    """(max |row - cur| over finite differences, number of elements that differ from ``cur``; NaN equals NaN): ``f`` = a ``[hmax, ...]`` feature array after the freeze, ``cur`` = ``[1, ...]``, the tick-j frame as the item
    builder built it (BEFORE the freeze). Both are 0 iff every row IS the current frame (a freeze to a stale row reads > 0, not 0)."""
    f, cur = np.asarray(f), np.asarray(cur)
    eq = (f == cur) | (np.isnan(f) & np.isnan(cur))
    d = np.abs(f.astype(np.float64) - cur.astype(np.float64))
    d = d[np.isfinite(d)]
    return (float(d.max()) if d.size else 0.0), int((~eq).sum())


def logged_arc(xy_log) -> np.ndarray:
    """REINDEXED: the cumulative arc length S[k] (metres, S[0] = 0, non-decreasing) of the logged polyline ``xy_log`` [T, 2] at its cache indices k. Refuses a polyline with a non-finite point."""
    xy = np.asarray(xy_log, np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("REFUSED: --scene reindexed needs a finite logged (x, y) polyline of at least two points")
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def project_taus(xy_log, own_xy, dwell_m: float = RI_DWELL_M):
    """REINDEXED: tau* for each own point. ``own_xy`` [R, 2] -> (``tau_star`` [R] int, ``offset_m`` [R] float = the distance from the point to its projection on the polyline).
    The point is projected on the NEAREST segment of ``xy_log`` (the latest segment among exact ties), its arc length s = S[k] + u * |segment| is read, and tau* is the logged index whose arc length S is nearest
    to s (the latest among exact ties), moved to the latest index whose arc length is within ``dwell_m`` of it (a dwell: a stop with jitter)."""
    xy = np.asarray(xy_log, np.float64)
    S = logged_arc(xy)
    own = np.asarray(own_xy, np.float64).reshape(-1, 2)
    if not np.isfinite(own).all():
        raise ValueError("REFUSED: --scene reindexed got a non-finite own (executed) position")
    a, ab = xy[:-1], xy[1:] - xy[:-1]
    l2 = (ab ** 2).sum(1)
    ln = np.sqrt(l2)
    safe = np.where(l2 > 0, l2, 1.0)
    tau = np.empty(len(own), np.int64)
    off = np.empty(len(own), np.float64)
    for r, p in enumerate(own):
        u = np.where(l2 > 0, np.clip(((p - a) * ab).sum(1) / safe, 0.0, 1.0), 0.0)
        d = np.linalg.norm(p - (a + u[:, None] * ab), axis=1)
        k = int(np.flatnonzero(d <= d.min() + 1e-9)[-1])
        sk = S[k] + u[k] * ln[k]
        ds_ = np.abs(S - sk)
        k0 = int(np.flatnonzero(ds_ <= ds_.min() + 1e-9)[-1])
        tau[r] = max(k0, int(np.searchsorted(S, S[k0] + dwell_m, side="right")) - 1)
        off[r] = d[k]
    return tau, off


def reindex_rows(i_rows, t0: int, own_xy, xy_log) -> dict:
    """REINDEXED: the frame index for each row of an item. ``i_rows`` [hm] = the rows' time indices (may start below 0: padded rows), ``own_xy`` [hm, 2] = the executed (x, y) at each row's time (only the rows with
    i > t0 are read), ``xy_log`` [T, 2] = the clip's logged polyline. Returns ``tau`` [hm] int (the index of the frame each row takes; = the clipped own index on the rows with i <= t0), ``tau_star`` [hm] (-1 on those rows),
    ``past`` [hm] bool, ``offset_m`` [hm] (0 on the past rows), ``raised`` [hm] bool (the running maximum moved the row), ``idc`` [hm] (the clipped own index)."""
    i_rows = np.asarray(i_rows, np.int64)
    hm = len(i_rows)
    idc = np.clip(i_rows, 0, len(xy_log) - 1)
    past = i_rows <= t0
    tau_star = np.full(hm, -1, np.int64)
    off = np.zeros(hm, np.float64)
    if (~past).any():
        tau_star[~past], off[~past] = project_taus(xy_log, np.asarray(own_xy, np.float64)[~past])
    raw = np.where(past, idc, np.minimum(i_rows, tau_star))
    tau = np.maximum.accumulate(raw)
    return {"tau": tau, "tau_star": tau_star, "past": past, "offset_m": off, "raised": tau != raw, "idc": idc}


def reindex_item(it: dict, st: WindowState, is_model: bool = True) -> tuple:
    """REINDEXED: ``(feats, frame_ok)`` of the item with the rows i > t0 taken from the logged frame at tau(i) (see the module docstring); updates ``RI_STATS``. Refuses an item without cached ``feats``."""
    f = it.get("feats")
    if f is None:
        raise RuntimeError("REFUSED: --scene reindexed needs the cached per-frame feats (absent under --encoder lora): a flag that silently reindexes nothing is worse than none")
    hm, j, t0 = f.shape[0], int(st.j), int(st.t0)
    i_rows = np.arange(j - hm + 1, j + 1)
    base = st.base_log
    idc = np.clip(i_rows, 0, st.T - 1)
    own = np.asarray(st.d["pose"], np.float64)[idc][:, :2]
    rr = reindex_rows(i_rows, t0, own, np.asarray(base["pose"], np.float64)[:, :2])
    tau = rr["tau"]
    feats, ok = np.array(f, copy=True), np.array(it["frame_ok"], copy=True)
    ch = tau != rr["idc"]
    if ch.any():
        feats[ch] = base["feats"][tau[ch]]
        ok[ch] = (tau[ch] >= 0) & (np.isfinite(np.asarray(st.d_in["p10f"])[tau[ch]]).all(1) | (tau[ch] < 1))
    pst = RI_STATS["past"]                                              # the check, not an assertion: the rows handed on for i <= t0 are byte-equal to the rows the item builder built;
    chk = i_rows <= t0                                                  # its mask is spelled here on its own, so a wrong mask inside reindex_rows cannot hide from it
    npast = int(chk.sum())
    if npast:
        a_b, b_b = feats[chk].view(np.uint8).reshape(npast, -1), np.asarray(f)[chk].view(np.uint8).reshape(npast, -1)
        pst["rows_checked"] += npast
        pst["rows_changed"] += int((((a_b != b_b).any(1)) | (ok[chk] != np.asarray(it["frame_ok"])[chk])).sum())
    r = RI_STATS["model" if is_model else "floor"]
    fut = ~chk
    r["items"] += 1
    r["rows"] += hm
    r["rows_logged"] += npast
    r["rows_reindexed"] += int((fut & (tau < i_rows)).sum())
    r["rows_on_time"] += int((fut & (rr["tau_star"] == i_rows)).sum())
    r["rows_clamped"] += int((fut & (rr["tau_star"] > i_rows)).sum())
    r["rows_monotone_raised"] += int(rr["raised"].sum())
    r["future_frame_rows"] += int((tau > rr["idc"]).sum())
    for lg in (i_rows - tau)[fut].tolist():
        r["lag_hist"][lg] = r["lag_hist"].get(lg, 0) + 1
    r["offset_m"].append(rr["offset_m"][fut])
    return feats, ok


def ri_summary(r: dict) -> dict:
    """REINDEXED: one role's receipt block (JSON-safe): counts, the lag distribution over the rows with i > t0 and over the reindexed rows (lag > 0), the lateral offset of the projection."""
    def dist(vals):
        v = np.asarray(vals, np.float64)
        return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)), "max": float(v.max())} if v.size else {"n": 0}
    lags = np.repeat(np.array(sorted(r["lag_hist"]), np.float64), [r["lag_hist"][k] for k in sorted(r["lag_hist"])]) if r["lag_hist"] else np.zeros(0)
    off = np.concatenate(r["offset_m"]) if r["offset_m"] else np.zeros(0)
    keys = ("items", "rows", "rows_logged", "rows_reindexed", "rows_on_time", "rows_clamped", "rows_monotone_raised", "future_frame_rows")
    return {**{k: int(r[k]) for k in keys}, "lag_cache_steps_all_rows_after_t0": dist(lags), "lag_cache_steps_reindexed_rows": dist(lags[lags > 0]), "lateral_offset_m_rows_after_t0": dist(off)}


def build_item(ds, proxy, st: WindowState, tick: int, a_t: int, a_s: int, scene: str, limit: str, stats: dict | None = None) -> dict:
    if st.hist_mode in CV_MODES:                                         # HISTSWAP-LF: the view is rebuilt at every tick from the tick's executed (cv_exec) / logged (cv_logged) speed
        st.d_in = st.cv_view(limit)
    proxy.d = st.d_in
    it = ds.item(st.ci, st.j, a_t, a_s)
    idc = np.clip(np.arange(st.j - ds.hmax + 1, st.j + 1), 0, st.T - 1)
    it["nav"] = st.nav[idc].astype(np.float32)
    if scene == "frozen":
        if tick == 0:
            st.scene0 = (it["feats"], it["frame_ok"])
        it["feats"], it["frame_ok"] = st.scene0
    if scene == "hold_now":                                              # FRAMEFREEZE: no frame motion; frame_ok untouched
        cur = None if it.get("feats") is None else np.array(it["feats"][-1:])      # the tick-j frame as built, before the freeze
        it["feats"] = hold_now_feats(it)
        dv, nu = freeze_dev(it["feats"], cur)
        FREEZE_STATS["items"] += 1
        FREEZE_STATS["max_row_dev"] = max(FREEZE_STATS["max_row_dev"], dv)
        FREEZE_STATS["n_unequal_elements"] += nu
    if scene == "reindexed":                                             # REINDEXED: the frames follow the model's OWN progress along the logged path (rows i <= t0 stay logged)
        it["feats"], it["frame_ok"] = reindex_item(it, st, is_model=stats is not None)
    if limit == "off":
        it["limit"], it["limit_ok"] = np.zeros(4, np.float32), False
    elif limit in ("causal", "causal_all"):
        logged = np.asarray(it["limit"], np.float32).copy() if bool(it["limit_ok"]) else None
        it["limit"], it["limit_ok"] = causal_bucket(st.d_in["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
        if stats is not None and st.d_in is not st.d:                    # HISTSWAP only: how often the swapped bucket differs from the executed-history bucket
            ex, ex_ok = causal_bucket(st.d["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
            stats["swap_ticks"] = stats.get("swap_ticks", 0) + 1
            stats["swap_bucket_differs"] = stats.get("swap_bucket_differs", 0) + int(bool(ex_ok) != bool(it["limit_ok"]) or (bool(ex_ok) and int(ex.argmax()) != int(it["limit"].argmax())))
        if stats is not None:
            stats["ticks"] = stats.get("ticks", 0) + 1
            if not it["limit_ok"]:
                stats["no_finite_past"] = stats.get("no_finite_past", 0) + 1
            elif logged is not None:
                cm = stats.setdefault("confusion_logged_x_causal", [[0] * 4 for _ in range(4)])
                cm[int(logged.argmax())][int(it["limit"].argmax())] += 1
    return it


def causal_bucket(v, j: int, steps: int | None):
    """The PREREG_V8F_T1KIN_CAUSALBUCKET bucket at step ``j``: ``speed_max_bin`` of the max finite speed over rows [j - steps, j] (or
    [0, j] when ``steps`` is None). Reads NOTHING after row j. No finite row -> (zeros, False), the ``--limit off`` format."""
    lo = 0 if steps is None else max(0, j - steps)
    w = np.asarray(v[lo:j + 1], np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.zeros(4, np.float32), False
    i, _ = speed_max_bin(float(max(w.max(), 0.0)))
    oh = np.zeros(4, np.float32)
    oh[i] = 1.0
    return oh, True


# ------------------------------------------------------------------------------------------------ planners
def smooth_plan(plan, prior, mode: str = "off"):
    """PREREG_V8F_SMOOTH's operator ``S(plan) = P + SG(plan - P)`` on a batch of plans [B, 30, 2] and their prior ``P`` [B, 30, 2] (the SAME arithmetic as
    ``v8f_smooth_read.smooth``: ``base + savgol_filter(pred - base, 7, 2, axis=1, mode="interp")``, evaluated in float64). ``off`` returns ``plan`` itself (the same object)."""
    if mode == "off":
        return plan
    if mode != "sg72":
        raise ValueError(f"--smooth must be one of {SMOOTH_MODES}, got {mode!r}")
    from scipy.signal import savgol_filter
    plan64, p64 = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    if plan64.shape != p64.shape or plan64.ndim != 3 or plan64.shape[1] < SG_WINDOW:
        raise ValueError(f"smooth_plan: plan {plan64.shape} and prior {p64.shape} must match and be [B, T>={SG_WINDOW}, 2]")
    return (p64 + savgol_filter(plan64 - p64, SG_WINDOW, SG_ORDER, axis=1, mode="interp")).astype(np.asarray(plan).dtype)


def feasible_plan(plan, mode: str = "off", v0=None):
    """The feasible-decode projection on a batch of plans [B, T, 2] (ego frame, knots 1..T; the origin is implied): ``project_feasible`` of ``tanitad.refs.feasible_decode`` on the
    origin-prepended path, float64, ``dt=DT``, ``mu=None``, returned in the plan's dtype without the origin. ``box`` = both clamps (KAPPA-T1), ``clamp_entry=False``; ``kbox`` = the
    curvature clamp alone (``a_max=inf``); ``abox`` = the accel clamp alone (``kappa_max=inf``) WITH the entry-speed bound ``clamp_entry=True`` (needs ``v0`` [B], the tick's measured ego
    speed); ``kabox`` = ``kbox`` then ``abox``. ``off`` returns ``plan`` itself (the same object)."""
    if mode == "off":
        return plan
    if mode not in FEASIBLE_MODES:
        raise ValueError(f"--feasible must be one of {FEASIBLE_MODES}, got {mode!r}")
    import torch
    from tanitad.refs import feasible_decode as FD
    p64 = np.asarray(plan, np.float64)
    if p64.ndim != 3 or p64.shape[-1] != 2 or p64.shape[1] < 3:
        raise ValueError(f"feasible_plan: plan {p64.shape} must be [B, T>=3, 2]")
    inf = float("inf")
    full = torch.from_numpy(np.concatenate([np.zeros_like(p64[:, :1]), p64], 1))
    if mode in ("abox", "kabox"):
        if v0 is None or np.asarray(v0).shape != (p64.shape[0],):
            raise ValueError(f"feasible_plan: mode {mode!r} needs v0 of shape [B]={p64.shape[0]}, got {None if v0 is None else np.asarray(v0).shape}")
        v0t = torch.from_numpy(np.asarray(v0, np.float64))
    if mode == "box":
        out = FD.project_feasible(full, dt=DT, a_max=FD.A_MAX_MPS2, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    elif mode in ("kbox", "kabox"):
        out = FD.project_feasible(full, dt=DT, a_max=inf, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    if mode in ("abox", "kabox"):
        out = FD.project_feasible(full if mode == "abox" else out, v0=v0t, dt=DT, a_max=FD.A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)
    return out[:, 1:].numpy().astype(np.asarray(plan).dtype)


#: the S / T head outputs ``--dump-heads`` keeps from ``forward_infer`` (the names ``v8f_mini_eval.cmd_run`` collects into ``outs``); float32, as the T0 eval stores them
HEAD_KEYS = ("ckpt", "goal_prob", "lat_logits", "lon_logits", "cons_pred")


# ------------------------------------------------------------------------------------------------ LEVEL-FLOOR (PREREG_LEVELFLOOR_A3, sha256 977a785e...): the floor operator F
LF_MATCH = ("hist", "hmax", "max_age_t", "max_age_s", "kin_mode", "n_tok", "d_feat")        # the planner's batch is built for ITS cfg: the floor model must read the same interface
LF_PARITY_TOL = 1e-6                                                                           # the first replan's light-path heads vs the floor model's own forward_infer (expected: 0.0)
_LF_MOD = None


def _lf():
    """``level_floor.py`` (the pure operator, next to this tool), loaded BY PATH and only under ``--level-floor on`` (``off`` never imports it)."""
    global _LF_MOD
    if _LF_MOD is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "level_floor.py")
        if not os.path.isfile(path):
            raise SystemExit(f"REFUSED: --level-floor on needs {path} next to this tool")
        spec = importlib.util.spec_from_file_location("level_floor", path)
        _LF_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_LF_MOD)
    return _LF_MOD


def lf_check_args(a):
    """The flag refusals, before anything is loaded: ``on`` needs ``--floor-kappa`` in [0, 1] and an existing ``--floor-ckpt``; ``off`` takes neither; no ``--histswap`` with the floor."""
    if a.level_floor == "on":
        if a.floor_kappa is None:
            raise SystemExit("REFUSED: --level-floor on needs --floor-kappa K (a dose is never defaulted)")
        if not (math.isfinite(float(a.floor_kappa)) and 0.0 <= float(a.floor_kappa) <= 1.0):
            raise SystemExit(f"REFUSED: --floor-kappa must be a finite number in [0, 1], got {a.floor_kappa!r}")
        if not a.floor_ckpt:
            raise SystemExit("REFUSED: --level-floor on needs --floor-ckpt PATH (the head checkpoint)")
        if not os.path.isfile(a.floor_ckpt):
            raise SystemExit(f"REFUSED: --floor-ckpt {a.floor_ckpt} does not exist")
        if a.histswap != "off":
            raise SystemExit(f"REFUSED: --level-floor on is not defined with --histswap {a.histswap} (F's v0 would be the logged speed, an oracle channel)")
    elif a.floor_kappa is not None or a.floor_ckpt:
        raise SystemExit("REFUSED: --floor-kappa / --floor-ckpt belong to --level-floor on; with the floor off they would silently do nothing")
    src_ = getattr(a, "floor_vhat_source", None)
    if a.level_floor != "on":
        if src_ is not None:
            raise SystemExit(f"REFUSED: --floor-vhat-source {src_} belongs to --level-floor on (the logged-speed oracle exists only inside the floor; with the floor off the flag would silently do nothing)")
    elif src_ is None:
        a.floor_vhat_source = "heads"
    nz_, nzs_ = getattr(a, "floor_vhat_noise", None), getattr(a, "floor_vhat_noise_seed", None)          # LF5
    if nz_ is not None or nzs_ is not None:
        if a.level_floor != "on" or getattr(a, "floor_vhat_source", None) != "logged":
            raise SystemExit("REFUSED: --floor-vhat-noise / --floor-vhat-noise-seed are defined only with --level-floor on --floor-vhat-source logged (the noise perturbs the logged speed; on the heads it would silently do nothing)")
        if nz_ is not None and not (math.isfinite(float(nz_)) and float(nz_) >= 0.0):
            raise SystemExit(f"REFUSED: --floor-vhat-noise must be a finite number >= 0, got {nz_!r}")
        if nzs_ is not None and int(nzs_) < 0:
            raise SystemExit(f"REFUSED: --floor-vhat-noise-seed must be an integer >= 0, got {nzs_!r}")
        if nz_ is not None and float(nz_) > 0.0 and nzs_ is None:
            raise SystemExit("REFUSED: --floor-vhat-noise SIGMA > 0 needs --floor-vhat-noise-seed N (a noise level is never drawn from an unnamed generator)")
    lc_ = getattr(a, "floor_lift_cap", None)                                                              # LF6
    if lc_ is not None:
        if a.level_floor != "on":
            raise SystemExit("REFUSED: --floor-lift-cap belongs to --level-floor on; with the floor off it would silently do nothing")
        if not (math.isfinite(float(lc_)) and float(lc_) >= 0.0):
            raise SystemExit(f"REFUSED: --floor-lift-cap must be a finite number >= 0, got {lc_!r}")


LOGGED_KNOTS = 6                                                                              # LFO: knots 1..6 of the tick's logged route = five 0.2 s segments = the route's first second


def logged_route_speed(gt, gt_ok):
    """The logged route's own speed at the tick over its first second: the path length of knots 1..6 of ``gt`` [N, 30, 2] (the logged route in the executed ego frame) / 1.0 s. The same arithmetic as the reader's
    ``logged_speed`` (motion_c/lf_read/code/lf_read.py). -> v [N] float64, NaN where the tick has no complete logged route (``gt_ok`` False or a non-finite value)."""
    gt = np.asarray(gt, np.float64)
    v = np.linalg.norm(np.diff(gt[:, :LOGGED_KNOTS], axis=1), axis=-1).sum(1) / ((LOGGED_KNOTS - 1) * DT)
    ok = np.asarray(gt_ok, bool) & np.isfinite(v)
    return np.where(ok, v, np.nan)


def make_logged_oracle(store, J):
    """LFO: ``oracle(states)`` -> v_log [B] float64 (NaN = no complete logged route), from the SAME ``gt_in_frame(store.get(clip)["pose"], pose_now, st.j)`` the TickRecorder dumps (``pose_now`` = the window's executed pose at the tick).
    CEILING: this reads the human's logged route, a privileged channel; it exists only behind ``--floor-vhat-source logged``."""
    def oracle(states):
        gts, oks = [], []
        for st in states:
            clip = str(J["clip"][st.ci])
            pose_now = np.array(st.d["pose"][st.j], np.float64)
            gt, ok = gt_in_frame(store.get(clip)["pose"], pose_now, st.j)
            gts.append(gt)
            oks.append(ok)
        return logged_route_speed(np.stack(gts), np.asarray(oks, bool))
    return oracle


def lf_check_cfg(fcfg, fgrid, cfg, grid):
    """Refuse a floor checkpoint whose cfg lacks ``level_head`` on AND ``lead_head`` on, or whose data interface differs from the planner's."""
    if fcfg.level_head != "on" or fcfg.lead_head != "on":
        raise SystemExit(f"REFUSED: --floor-ckpt's cfg must have level_head on AND lead_head on (it has level_head={fcfg.level_head!r}, lead_head={fcfg.lead_head!r})")
    for k in LF_MATCH + ("window_start_effective",):
        if getattr(fcfg, k) != getattr(cfg, k):
            raise SystemExit(f"REFUSED: --floor-ckpt's {k}={getattr(fcfg, k)!r} differs from the planner's {getattr(cfg, k)!r}: the two models would not read the same batch")
    if fgrid != grid:
        raise SystemExit(f"REFUSED: --floor-ckpt's grid {fgrid!r} differs from the planner's {grid!r}")


def floor_heads(fm, batch):
    """(v_hat [B], presence logit [B], gap_hat [B]) float32 numpy and the batch keys read, from the floor model's OWN encode of ``batch``: ``_encode`` (the adapter on ``feats``), then ``_level`` and ``_lead`` at the
    current frame (the exact calls ``V8FMini._forward`` makes for ``level_v`` / ``lead_logit`` / ``lead_gap``). The model's O / S / T are not run. ``batch`` is read through the model's key guard."""
    import torch
    g = M._KeyGuard(batch)
    with torch.no_grad():
        adapt = fm._encode(g)
        last_now = torch.full((adapt.shape[0],), g["feats"].shape[1] - 1, dtype=torch.long, device=adapt.device)
        v_hat = fm._level(g, adapt, last_now)[0]
        logit, gap_hat = fm._lead(g, adapt, last_now)[:2]
    return v_hat.float().cpu().numpy(), logit.float().cpu().numpy(), gap_hat.float().cpu().numpy(), set(g.read)


# ------------------------------------------------------------------------------------------------ LF6 (PREREG_LEVELFLOOR_A6, sha256 fbf5de02...): the bounded-lift floor F_B
def bounded_progress(op, v_f, v0, S, delta):
    """s_f(t_k), k = 1..30, of F_B with the cap v_O(tau) + delta. ``v_f`` [B] the scalar first term (already 0 where no floor), ``v0`` [B], ``S`` [B, 31] the plan's cumulative arc length (S[:, 0] = 0). Returns ``(s_f [B, 30], dterm_k [B, 30] bool)``.
    v_O,k = (S[k] - S[k-1]) / DT on (t_{k-1}, t_k]. On that interval the integrand is min(c_k, v0' + tau), c_k = min(v_f, v_O,k + delta): the closed form G(t; c) = v0' * tr + 0.5 * a_f * tr^2 + c * (t - tr), tr = min(t, max(c - v0', 0) / a_f), differenced over the interval.
    ``dterm_k``: the delta term is the strict minimum of the three at the interval's end t_k (v_O,k + delta < v_f and < v0' + t_k); where no interval has it, the integrand is min(v_f, v0' + tau) throughout and F_B IS F. The result is clipped at S[k] + delta * t_k (the card's bound)."""
    v_f = np.atleast_1d(np.asarray(v_f, np.float64))
    v0p = np.maximum(np.nan_to_num(np.atleast_1d(np.asarray(v0, np.float64)), nan=0.0, posinf=0.0, neginf=0.0), 0.0)[:, None]
    t_hi = op.T_KNOTS[None, :]
    t_lo = (op.DT * np.arange(0, op.K, dtype=np.float64))[None, :]
    v_o = np.diff(np.asarray(S, np.float64), axis=1) / op.DT                                     # [B, K]: O's planned speed on (t_{k-1}, t_k]
    cap_o = v_o + float(delta)
    vf_c = v_f[:, None]
    c = np.minimum(vf_c, cap_o)

    def G(t):
        tr = np.minimum(t, np.maximum(c - v0p, 0.0) / op.A_F)
        return v0p * tr + 0.5 * op.A_F * tr * tr + c * (t - tr)
    s_f = np.cumsum(G(t_hi) - G(t_lo), axis=1)
    dterm_k = (cap_o < vf_c) & (cap_o < v0p + t_hi)
    return np.minimum(s_f, np.asarray(S, np.float64)[:, 1:] + float(delta) * t_hi), dterm_k


def apply_floor_batch_bounded(op, plans, v0, v_hat, pi, gap_hat, kappa, delta):
    """F_B on a batch: ``level_floor.apply_floor_batch`` with s_f from :func:`bounded_progress`. Same return and ``info`` keys as the operator, plus ``dterm`` (bound row with the delta term the strict minimum at a knot where F lifts), ``dterm_knots`` (how many), ``s`` (s_k) and ``sp`` (s'_k), [B, 30]."""
    delta = float(delta)
    if not (np.isfinite(delta) and delta >= 0.0):
        raise ValueError(f"delta must be a finite number >= 0, got {delta}")
    p64 = np.asarray(plans, np.float64)
    full, S = op.plan_arc(p64)
    B = p64.shape[0]
    v_hat, pi, gap_hat, v0 = (np.broadcast_to(np.atleast_1d(np.asarray(x, np.float64)), (B,)) for x in (v_hat, pi, gap_hat, v0))
    v_f, band, cap = op.floor_speed(v_hat, pi, gap_hat, kappa)
    v_fb = np.where(band, v_f, 0.0)
    s_ramp = op.ramp_progress(v_fb, v0)                                                          # A5's F (closed form)
    with np.errstate(invalid="ignore"):
        s_b, dterm_k = bounded_progress(op, v_fb, v0, S, delta)
    use_b = dterm_k.any(axis=1)
    s_k = S[:, 1:]
    with np.errstate(invalid="ignore"):
        s_f = np.minimum(np.where(use_b[:, None], s_b, s_ramp), s_k + delta * op.T_KNOTS[None, :])
        s_new = np.maximum(s_k, s_f)
        knot_bind = band[:, None] & np.isfinite(p64).all(axis=(1, 2))[:, None] & (s_new != s_k)
    row_bind = knot_bind.any(axis=1)
    kd = dterm_k & knot_bind
    info = {"bind": row_bind, "cap": cap, "vf": v_f, "vhat": v_hat.copy(), "pi": pi.copy(), "gap": op.GAP_SCALE_M * gap_hat, "lift": np.zeros(B), "band": band,
            "dir_fallback": np.zeros(B, bool), "short_last": np.zeros(B, bool), "bind_k10": knot_bind[:, :op.K_EXEC].any(axis=1),
            "dterm": row_bind & kd.any(axis=1), "dterm_knots": kd.sum(axis=1).astype(np.int64), "s": s_k.copy(), "sp": s_new.copy()}
    if not row_bind.any():
        return plans, info
    out = np.array(plans, copy=True)
    for b_ in np.nonzero(row_bind)[0]:
        pts, fb = op.resample_polyline(full[b_], S[b_], s_new[b_])
        new = p64[b_].copy()
        new[knot_bind[b_]] = pts[knot_bind[b_]]
        out[b_] = new.astype(out.dtype)
        info["dir_fallback"][b_] = fb
        info["short_last"][b_] = bool(np.linalg.norm(full[b_, -1] - full[b_, -2]) < op.SHORT_LAST_M)
        _, S_out = op.plan_arc(np.asarray(out[b_], np.float64)[None])
        info["lift"][b_] = (S_out[0, op.K_LIFT] - S[b_, op.K_LIFT]) / op.T_LIFT_S
    return out, info


LF6_OPERATOR_TEXT = (" || LF6 F_B (PREREG_LEVELFLOOR_A6): v_f(tau) = min(kappa * v_hat with the lead cap, v0 + 1.0 * tau, v_O(tau) + DELTA), v_O = the plan's chord speed on (t_{k-1}, t_k] after S (knot 0 = origin); "
                     "s_f(t) = integral_0^t v_f; s' = max(s, s_f) with s_f <= s + DELTA * t; the rest as F")


class LevelFloor:
    """The floor of one run: the floor model (a second instance, ``--floor-ckpt``), the dose and the running counts. ``__call__(batch, plan, v0)`` -> ``(plan_out, info)`` (``level_floor.apply_floor_batch``)."""

    def __init__(self, fm, kappa: float, ckpt_path: str, ckpt_sha256: str, vhat_source: str = "heads", oracle=None, noise=None, noise_seed=None, lift_cap=None):
        self.fm, self.kappa, self.ckpt_path, self.ckpt_sha256 = fm, float(kappa), ckpt_path, ckpt_sha256
        if vhat_source not in ("heads", "logged"):
            raise ValueError(f"vhat_source must be 'heads' or 'logged', got {vhat_source!r}")
        self.vhat_source, self.oracle, self.states = vhat_source, oracle, None                      # LFO: the oracle and the tick's window states (set by model_planner before each call)
        self.op = _lf()
        self.keys_read = set()
        self.parity = {"checked_replans": 0, "max_abs_diff": None}
        self.stats = {"replans": 0, "band": 0, "bind": 0, "bind_k10": 0, "cap": 0, "dir_fallback_binds": 0, "short_last_seg_binds": 0, "lift_sum_ms": 0.0}
        if self.vhat_source == "logged":
            self.stats["oracle_no_route"] = 0                                                       # ticks with no complete logged route = no floor
        self.noise = 0.0 if noise is None else float(noise)                                         # LF5: SIGMA (0.0 = none)
        self.noise_seed = None if noise_seed is None else int(noise_seed)
        if (noise is not None or noise_seed is not None) and self.vhat_source != "logged":
            raise ValueError("LF5: the v_hat noise belongs to --floor-vhat-source logged")
        if self.noise > 0.0 and self.noise_seed is None:
            raise ValueError("LF5: a noise level SIGMA > 0 needs a seed")
        self.noise_path = self.noise_seed is not None                                               # the noise path runs (draws are taken) iff a seed is given; SIGMA 0 then multiplies by exp(0) = 1.0
        self.rng = np.random.default_rng(self.noise_seed) if self.noise_path else None
        if self.noise_path:
            self.stats["noise_draws"] = 0
        self.lift_cap = None if lift_cap is None else float(lift_cap)                               # LF6: DELTA of F_B (None = A5's F)
        if self.lift_cap is not None:
            if not (math.isfinite(self.lift_cap) and self.lift_cap >= 0.0):
                raise ValueError(f"LF6: the lift cap DELTA must be a finite number >= 0, got {lift_cap!r}")
            self.stats["dterm_bound"] = 0                                                           # bound ticks where the DELTA term is the strict minimum at a knot where F lifts

    def heads(self, batch):
        v_hat, logit, gap_hat, read = floor_heads(self.fm, batch)
        self.keys_read |= read
        if self.parity["checked_replans"] == 0:                                                      # the first replan: the light path equals the model's own forward_infer outputs
            import torch
            with torch.no_grad():
                o = self.fm.forward_infer(batch)
            d = max(float(np.abs(v_hat - o["level_v"].float().cpu().numpy()).max()), float(np.abs(logit - o["lead_logit"].float().cpu().numpy()).max()),
                    float(np.abs(gap_hat - o["lead_gap"].float().cpu().numpy()).max()))
            if not d <= LF_PARITY_TOL:
                raise RuntimeError(f"level floor: the floor heads' light path differs from the floor model's forward_infer by {d:.3e} (> {LF_PARITY_TOL})")
            self.parity = {"checked_replans": 1, "max_abs_diff": d}
        return v_hat, logit, gap_hat

    def oracle_vhat(self, n):
        """LFO: v_log [n] for the ticks of the current ``states``."""
        if self.oracle is None or self.states is None or len(self.states) != n:
            raise RuntimeError(f"--floor-vhat-source logged: no oracle or no window states for this call (oracle {self.oracle is not None}, states {None if self.states is None else len(self.states)}, batch {n})")
        v_log = np.asarray(self.oracle(self.states), np.float64)
        if v_log.shape != (n,):
            raise RuntimeError(f"the logged-speed oracle returned {v_log.shape}, expected ({n},)")
        return v_log

    def __call__(self, batch, plan, v0):
        v_hat, logit, gap_hat = self.heads(batch)
        if self.vhat_source == "logged":                                                             # LFO, CEILING: the level input is the logged route speed
            v_heads = v_hat
            v_hat = self.oracle_vhat(len(v_heads))
            if self.noise_path:                                                                      # LF5: one draw per window of this replan, in replan order, whether or not a floor applies
                v_log = v_hat
                eps = self.rng.standard_normal(len(v_log))
                self.stats["noise_draws"] += int(len(eps))
                v_hat = v_log * np.exp(self.noise * eps)
        if self.lift_cap is None:
            out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
        else:                                                                                        # LF6: F_B
            out, info = apply_floor_batch_bounded(self.op, plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa, self.lift_cap)
            self.stats["dterm_bound"] += int(info["dterm"].sum())
        if self.vhat_source == "logged":
            if self.noise_path:
                info["vlog"], info["eps"] = np.asarray(v_log, np.float64), np.asarray(eps, np.float64)
            info["vhat_heads"] = np.asarray(v_heads, np.float64)
            self.stats["oracle_no_route"] += int((~np.isfinite(v_hat)).sum())
        st = self.stats
        st["replans"] += len(info["bind"])
        st["band"] += int(info["band"].sum())
        st["bind"] += int(info["bind"].sum())
        st["cap"] += int(info["cap"].sum())
        st["bind_k10"] += int(info["bind_k10"].sum())
        st["dir_fallback_binds"] += int((info["dir_fallback"] & info["bind"]).sum())
        st["short_last_seg_binds"] += int((info["short_last"] & info["bind"]).sum())
        st["lift_sum_ms"] += float(info["lift"][info["bind"]].sum())
        return out, info

    def meta(self):
        m = {"level_floor": "on", "floor_kappa": self.kappa, "floor_ckpt_sha256": self.ckpt_sha256, "floor_vhat_source": self.vhat_source}
        if self.vhat_source == "logged":
            m["label"], m["floor_vhat_noise"] = "CEILING", self.noise
            if self.noise_seed is not None:                                                         # LF6 fix: a None seed is a 0-d object array that allow_pickle=False refuses (A5_NOTE_READER_FIX); absent = no draw
                m["floor_vhat_noise_seed"] = self.noise_seed
        if self.lift_cap is not None:
            m["floor_lift_cap"] = self.lift_cap
        return m

    def detail(self):
        st = dict(self.stats)
        st["lift_mean_ms_where_bound"] = (st["lift_sum_ms"] / st["bind"]) if st["bind"] else None
        d = {"operator": self.op.OPERATOR_TEXT, "constants": self.op.CONSTANTS, "prereg": "dagland/motion_c/a3_prereg/PREREG_LEVELFLOOR_A3.md (sha256 977a785e...), H-LEVELFLOOR-1",
                "applied": "at every replan inside model_planner's plan(batch): raw -> S (--smooth) -> F -> the projection (--feasible); the P10 floor is not floored",
                "vision_only": "v_hat, presence and gap_hat come from the floor model's adapter + lvl_head + lead_head on the tick's frames; v0 = batch['p10f'][:, -1, 0] (measured ego speed); nothing else enters",
                "floor_model_batch_keys_read": sorted(self.keys_read), "heads_parity_vs_forward_infer": self.parity, "counts": st}
        d["floor_vhat_source"] = self.vhat_source
        if self.vhat_source == "logged":
            d["label"] = "CEILING"
            d["prereg"] = "dagland/motion_c/a4_prereg/PREREG_LEVELFLOOR_A4.md (sha256 8bfbc386...), H-LEVELFLOOR-1 (A4)"
            d["vision_only"] = ("NO. CEILING diagnostic, inadmissible at inference: v_hat := the LOGGED route speed of the tick (path length of knots 1..6 of the tick's logged route in the executed ego frame / 1.0 s, a privileged channel); "
                                "presence and gap_hat still come from the floor heads; a tick with no complete logged route has no floor")
            d["floor_vhat_noise"], d["floor_vhat_noise_seed"] = self.noise, self.noise_seed
            if self.noise_path:
                d["noise"] = "v_hat := v_log * exp(SIGMA * eps), eps ~ N(0,1), one numpy.random.default_rng(seed) per rollout, one vector draw per floor call, in replan order, whether or not a floor applies (PREREG_LEVELFLOOR_A5 sha256 2aac1dbe...)"
            d["level_input"] = "logged_route_speed(gt_in_frame(...)) computed at the replan before F (floor_vhat_source logged); lf_vhat_heads in the ticks file holds what the heads said on the same tick"
        if self.lift_cap is not None:
            d["floor_lift_cap"] = self.lift_cap
            d["operator"] = self.op.OPERATOR_TEXT + LF6_OPERATOR_TEXT
            d["prereg"] = "dagland/motion_c/a6_prereg/PREREG_LEVELFLOOR_A6.md (sha256 fbf5de02...), H-LEVELFLOOR-1 (A6)"
            d["bounded_lift"] = "s' - s <= DELTA * t at every knot; lf_vf is the scalar first term kappa * v_hat (lead cap), the DELTA term acts inside s_f; lf_dterm = a bound tick where the DELTA term is the strict minimum at a knot where F lifts; lf_s / lf_sp = s_k / s'_k"
            st["dterm_share_of_bound"] = (st["dterm_bound"] / st["bind"]) if st["bind"] else None
        return d


def load_level_floor(a, cfg, ck, dev):
    """The floor of ``--level-floor on``: the head checkpoint is loaded into its OWN V8FMini (refused unless its cfg has both heads on and reads the planner's interface)."""
    import hashlib
    import torch
    fk_ = torch.load(a.floor_ckpt, map_location="cpu", weights_only=False)
    fcfg = M.V8FMiniConfig(**fk_["cfg"])
    lf_check_cfg(fcfg, fk_.get("grid"), cfg, ck.get("grid"))
    fm = M.V8FMini(fcfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    fm.load_state_dict(fk_["model"])
    fm.to(dev).eval()
    probs = M.level_head_source_problems() + M.lead_head_source_problems()
    if probs:
        raise SystemExit(f"REFUSED: the floor model's heads are not vision-only by source: {probs}")
    h = hashlib.sha256()
    with open(a.floor_ckpt, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return LevelFloor(fm, a.floor_kappa, a.floor_ckpt, h.hexdigest(), vhat_source=getattr(a, "floor_vhat_source", None) or "heads",
                      noise=getattr(a, "floor_vhat_noise", None), noise_seed=getattr(a, "floor_vhat_noise_seed", None), lift_cap=getattr(a, "floor_lift_cap", None))


def model_planner(model, device, keep_aux: bool = False, keep_heads: bool = False, smooth: str = "off", feasible: str = "off", floor=None):
    """The model's plan. ``keep_aux`` (``--dump-ticks``) additionally keeps, per call, O's plan before the post-map (``tau``), the prior (``P``) and the gate flags;
    ``keep_heads`` (``--dump-heads``) keeps :data:`HEAD_KEYS` of the SAME ``forward_infer`` call (never a second call). Neither changes any output.
    ``smooth`` (``--smooth``): ``sg72`` returns :func:`smooth_plan` of the model's plan over its own prior ``out["p10"]`` (the pre-S plan is kept as ``plan.plan_raw``); ``off`` returns the plan untouched.
    ``floor`` (``--level-floor on``, a :class:`LevelFloor`): after S the floor operator F runs on the plan at every replan (raw -> S -> F -> ``feasible``); ``plan.floor_info`` holds what F did (None without a floor)."""
    import torch

    def plan(batch):
        with torch.no_grad():
            out = model.forward_infer(batch)
        plan.keys_read = set(out["_keys_read"])
        if keep_aux:
            plan.aux = {"tau": out.get("traj30_o", out["traj30"]).double().cpu().numpy(), "P": out["p10"].double().cpu().numpy(),
                        "fired": None if "hacc_gate_fired" not in out else out["hacc_gate_fired"].cpu().numpy()}
        if keep_heads:
            plan.heads = {k: out[k].float().cpu().numpy() for k in HEAD_KEYS}
        raw = out["traj30"].float().cpu().numpy()
        if floor is not None:                                            # LEVEL-FLOOR: raw -> S -> F -> projection; the floor heads read the SAME batch in the floor model's own encode
            plan.plan_raw = raw
            plan_s = smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
            v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()          # the measured ego speed (== TickRecorder's ``v0``)
            if getattr(floor, "vhat_source", "heads") == "logged":       # LFO: the oracle reads the tick's window states (set by rollout)
                floor.states = plan.states
            plan_f, plan.floor_info = floor(batch, plan_s, v0)
            if feasible != "off":
                plan.plan_s = plan_s                                     # the plan after S, before F and the projection
                return feasible_plan(plan_f, feasible, v0=v0)
            return plan_f
        if smooth == "off" and feasible == "off":
            return raw
        plan.plan_raw = raw
        if feasible != "off":                                            # KAPPA-T1: the projection AFTER S (S is the identity when --smooth off)
            plan.plan_s = smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
            v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()          # FEAS2-T1: the measured ego speed the prior reads (== TickRecorder's ``v0``); used by abox / kabox only
            return feasible_plan(plan.plan_s, feasible, v0=v0)
        return smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
    plan.keys_read = set()
    plan.aux = None
    plan.heads = None
    plan.plan_raw = None
    plan.plan_s = None
    plan.floor_info = None
    plan.states = None
    return plan


def floor_planner(model, device):
    """The P10 floor driven the same way: the prior alone, the very call ``run_o`` makes (``v8f_mini.py:579``)."""
    import torch

    def plan(batch):
        with torch.no_grad():
            return model.prior(batch["p10f"][:, -1].float(), batch["base"].float()).float().cpu().numpy()
    return plan


# ------------------------------------------------------------------------------------------------ the tick recorder (--dump-ticks, --dump-heads)
def gt_in_frame(pose_grid, pose_now, j: int):
    """The logged future ``pose_grid[j+1 .. j+30]`` in the frame of ``pose_now`` = (x, y, yaw, ...): the SAME arithmetic as ``floors_kin.gt_waypoints`` / ``v8f_mini_data.gt_future`` (rotation by
    ``-yaw``), with the origin pose replaced by the EXECUTED pose. At tick 0 the executed pose IS the logged pose, so this equals the store's ``gt[t0]`` (control C-T2). ``ok`` False (NaN GT) when the logged clip
    ends before ``j + 30`` (the same ``t <= T - 31`` rule as ``gt_future``)."""
    pg = np.asarray(pose_grid, np.float64)
    if j + H > len(pg) - 1:
        return np.full((H, 2), np.nan), False
    d = pg[j + 1:j + 1 + H, :2] - np.asarray(pose_now, np.float64)[None, :2]
    c, s = np.cos(-pose_now[2]), np.sin(-pose_now[2])
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=-1), True


class TickRecorder:
    """The ``hook`` of :func:`rollout` that keeps the tick samples. ``hook(n, batch, plan, states)`` runs BEFORE the plan is committed, so ``st.j`` and ``st.d['pose'][st.j]`` are the tick's executed state.

    ``heads=True`` (``--dump-heads``) adds the S / T head outputs of the tick's own ``forward_infer`` call (``planner.heads``) under :attr:`HEAD_FIELDS`:
      * ``s_ckpt``      [N, 3, 3] f32  S's checkpoints (x, y, psi) at the 30 / 50 / 80 m arcs, RAW, in S's TICK frame = the executed pose at step ``j - age_s`` (``s_pose``); what ``forward_infer`` emits
      * ``s_ckpt_now``  [N, 3, 3] f32  the same in the NOW frame (the executed pose at the tick, ``pose``[:3]) by ``transform_ckpt`` -- the frame ``v8f_mini_eval._ckpt_to_now`` scores in
      * ``s_ckpt_w0``   [N, 3, 3] f32  the same in the WINDOW-ORIGIN frame (the logged pose at ``t0`` = ``pose_w0``): the one frame every tick of a window shares, whatever the executed path did
      * ``s_pose``      [N, 3]    f32  the world pose (x, y, yaw) of S's tick frame, as the batch gave it to the model
      * ``t_goal_prob`` [N, 22]   f32  T's goal tokens (sigmoid), ``v8f_mini.GOAL22`` order     * ``t_lat_logits`` [N, 5] f32  * ``t_lon_logits`` [N, 7] f32  * ``t_cons_pred`` [N, C] f32 (de-standardised)
      * ``t_pose``      [N, 3]    f32  the world pose of T's tick frame (``j - age_t``)
      * ``pose_w0``     [N, 3]    f64  the window-origin pose (x, y, yaw): ``st.d['pose'][t0][:3]`` (rows <= t0 are logged, never overwritten)
    T and S are the model's SAMPLE-AND-HOLD heads: their outputs at tick ``j`` belong to the frames ``j - age_t`` / ``j - age_s`` (``age_t`` / ``age_s`` are in the dump)."""

    FIELDS = ("tau", "P", "plan", "gt", "gt_ok", "v0", "pose", "fired", "win", "ci", "t0", "tick", "j", "age_t", "age_s")
    RAW_HEAD_FIELDS = ("s_ckpt", "s_pose", "t_goal_prob", "t_lat_logits", "t_lon_logits", "t_cons_pred", "t_pose", "pose_w0")      # recorded per tick
    LF_FIELDS = ("lf_bind", "lf_cap", "lf_vf", "lf_vhat", "lf_pi", "lf_gap", "lf_lift", "lf_bind_k10")                                              # LEVEL-FLOOR: per replan, only with the floor on
    LFO_FIELDS = ("lf_vhat_heads",)                                                                                                 # LFO (floor_vhat_source logged): what the floor heads said on the same tick
    LF5_FIELDS = ("lf_vlog", "lf_eps")                                                                                               # LF5 (the noise path is active): v_log and the draw of the tick
    LF6_FIELDS = ("lf_dterm", "lf_dterm_knots", "lf_s", "lf_sp")                                                                     # LF6 (--floor-lift-cap): the DELTA term binds, s_k and s'_k of the tick
    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")                                                                      # + the two frames derived at save()

    def __init__(self, planner, store, J, heads: bool = False, smooth: str = "off", feasible: str = "off", histswap: str = "off", level_floor: bool = False, floor_oracle: bool = False, floor_noise: bool = False, floor_bound: bool = False):
        self.planner, self.store, self.J, self.heads, self.smooth, self.feasible, self.histswap = planner, store, J, bool(heads), smooth, feasible, histswap
        self.level_floor = bool(level_floor)
        self.floor_oracle = bool(floor_oracle) and self.level_floor
        self.floor_noise = bool(floor_noise) and self.floor_oracle
        self.floor_bound = bool(floor_bound) and self.level_floor
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("plan_raw",) if (smooth != "off" or feasible != "off" or level_floor) else ())
                     + (("plan_s",) if feasible != "off" else ()) + (("v0_in",) if histswap != "off" else ()) + (self.LF_FIELDS if level_floor else ()) + (self.LFO_FIELDS if (level_floor and floor_oracle) else ()) + (self.LF5_FIELDS if self.floor_noise else ()) + (self.LF6_FIELDS if self.floor_bound else ())}
        self.clip = []
        self.base = 0                                                    # the index of the chunk's first window in the run's window list

    def __call__(self, n, batch, plan, states):
        aux = self.planner.aux
        if aux is None:
            raise RuntimeError("--dump-ticks needs a planner built with keep_aux=True")
        hd = self.planner.heads
        if self.heads and hd is None:
            raise RuntimeError("--dump-heads needs a planner built with keep_heads=True")
        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()             # the measured ego speed the prior (and the gate) read: exactly p10f[:, -1, 0]
        v0_in = v0
        if self.histswap != "off":                                       # HISTSWAP: v0 stays the EXECUTED measured speed (what Y's v0 is); v0_in is the speed the network read
            v0 = np.array([float(np.nan_to_num(st.d["p10f"][st.j][0], nan=0.0)) for st in states], np.float64)
        if self.heads:
            pose_s, pose_t = batch["pose_s"].float().cpu().numpy(), batch["pose_t"].float().cpu().numpy()
        for b, st in enumerate(states):
            clip = str(self.J["clip"][st.ci])
            pose_now = np.array(st.d["pose"][st.j], np.float64)
            gt, ok = gt_in_frame(self.store.get(clip)["pose"], pose_now, st.j)
            r = self.rows
            r["tau"].append(aux["tau"][b]); r["P"].append(aux["P"][b]); r["plan"].append(np.asarray(plan[b], np.float64))
            if self.smooth != "off" or self.feasible != "off" or self.level_floor:          # only then: the pre-S plan beside the issued (smoothed) one
                r["plan_raw"].append(np.asarray(self.planner.plan_raw[b], np.float64))
            if self.feasible != "off":                                   # KAPPA-T1: the plan after S, before the projection
                r["plan_s"].append(np.asarray(self.planner.plan_s[b], np.float64))
            r["gt"].append(gt); r["gt_ok"].append(ok); r["v0"].append(v0[b]); r["pose"].append(pose_now[:4])
            if self.histswap != "off":
                r["v0_in"].append(v0_in[b])
            r["fired"].append(False if aux["fired"] is None else bool(aux["fired"][b]))
            r["win"].append(self.base + b); r["ci"].append(st.ci); r["t0"].append(st.t0); r["tick"].append(n); r["j"].append(st.j)
            r["age_t"].append(int(batch["a_t"][b])); r["age_s"].append(int(batch["a_s"][b]))
            if self.level_floor:                                         # LEVEL-FLOOR: what F did at this replan (its inputs from the floor heads, the lift, whether F / the lead cap bound)
                fi = self.planner.floor_info
                r["lf_bind"].append(bool(fi["bind"][b])); r["lf_cap"].append(bool(fi["cap"][b])); r["lf_bind_k10"].append(bool(fi["bind_k10"][b]))
                for k_ in ("vf", "vhat", "pi", "gap", "lift"):
                    r["lf_" + k_].append(np.float32(fi[k_][b]))
                if self.floor_oracle:
                    r["lf_vhat_heads"].append(np.float32(fi["vhat_heads"][b]))
                if self.floor_noise:
                    r["lf_vlog"].append(np.float64(fi["vlog"][b])); r["lf_eps"].append(np.float64(fi["eps"][b]))
                if self.floor_bound:
                    r["lf_dterm"].append(bool(fi["dterm"][b])); r["lf_dterm_knots"].append(int(fi["dterm_knots"][b]))
                    r["lf_s"].append(np.asarray(fi["s"][b], np.float64)); r["lf_sp"].append(np.asarray(fi["sp"][b], np.float64))
            if self.heads:
                r["s_ckpt"].append(hd["ckpt"][b])
                r["s_pose"].append(pose_s[b])
                r["t_goal_prob"].append(hd["goal_prob"][b])
                r["t_lat_logits"].append(hd["lat_logits"][b])
                r["t_lon_logits"].append(hd["lon_logits"][b])
                r["t_cons_pred"].append(hd["cons_pred"][b])
                r["t_pose"].append(pose_t[b])
                r["pose_w0"].append(np.array(st.d["pose"][st.t0][:3], np.float64))
            self.clip.append(clip)

    def _ckpt_frames(self, out):
        """``s_ckpt_now`` / ``s_ckpt_w0``: the raw checkpoints re-expressed by the model's own ``transform_ckpt`` (float32, as ``_forward`` does for O)."""
        import torch
        ck, ps = torch.as_tensor(out["s_ckpt"]), torch.as_tensor(out["s_pose"])
        now = torch.as_tensor(out["pose"][:, :3].astype(np.float32))
        w0 = torch.as_tensor(out["pose_w0"].astype(np.float32))
        return M.transform_ckpt(ck, ps, now).numpy(), M.transform_ckpt(ck, ps, w0).numpy()

    def save(self, path, extra: dict | None = None):
        out = {k: np.asarray(v) for k, v in self.rows.items()}
        if self.heads and len(out["s_ckpt"]):
            out["s_ckpt_now"], out["s_ckpt_w0"] = self._ckpt_frames(out)
        out["clip"] = np.asarray(self.clip)
        for k, v in (extra or {}).items():
            out[k] = np.asarray(v)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **out)
        return out


# ------------------------------------------------------------------------------------------------ the rollout
def rollout(planner, states, ds, proxy, *, m: int = 1, schedule: str = "periodic", k_t: int = 3, k_s: int = 6,
            scene: str = "logged", limit: str = "logged", device="cpu", executor: str = "plan", logged_tracks=None, hook=None):
    """Roll every window of ``states`` for ``H // m`` ticks. Returns ``plan0`` [B, 30, 2] (the tick-0 plans) and the list of
    per-window executed paths in the t0 ego frame. ``hook(n, batch, plan, states)`` is the test seam."""
    import torch
    if H % m:
        raise ValueError(f"--replan-steps {m} must divide {H}")
    plan0 = None
    for n in range(H // m):
        step = n * m
        a_t, a_s = ages_for(step, schedule, k_t, k_s)
        stats = CAUSAL_STATS if hasattr(planner, "keys_read") else None          # the MODEL's ticks only (the floor reads no bucket)
        items = [build_item(ds, proxy, st, n, a_t, a_s, scene, limit, stats) for st in states]
        full = D.collate(items)
        batch = {k: full[k].to(device) for k in MODEL_KEYS}
        if hasattr(planner, "states"):                                   # LFO: the logged-speed oracle reads these window states (CEILING; unused unless --floor-vhat-source logged)
            planner.states = states
        plan = planner(batch)
        if plan0 is None:
            plan0 = plan.copy()
        if hook is not None:
            hook(n, batch, plan, states)
        for st, pl in zip(states, plan):
            if executor == "logged":
                st.commit_logged(logged_tracks[st.ci], m)
            else:
                st.commit_plan(pl.astype(np.float64), m)
    return plan0, [st.executed_ego() for st in states]


def make_states(store, tracks, J, wins, nav_mode, hist_swap=False):
    out = []
    for ci, t0, _, _ in wins:
        clip = str(J["clip"][ci])
        base = store.get(clip)
        o, n = int(J["off"][ci]), int(J["n"][ci])
        out.append(WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode, hist_swap))
    return out


# ------------------------------------------------------------------------------------------------ scoring
def load_t1_eval():
    path = os.path.join(REPO, "taniteval", "tools", "t1_eval.py")
    spec = importlib.util.spec_from_file_location("t1_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_dumps(out_dir, wins, arms: dict, g, v0):
    """One ``ep*.npz`` per episode in ``t1_eval``'s schema; ``arms`` = {name: [N, 30, 2]}."""
    os.makedirs(out_dir, exist_ok=True)
    cis = np.array([w[0] for w in wins])
    ts = np.array([w[1] for w in wins])
    files = []
    for ci in dict.fromkeys(cis.tolist()):
        m = cis == ci
        f = os.path.join(out_dir, f"ep{ci:04d}.npz")
        np.savez(f, g=g[m], ws=ts[m], eid=cis[m], clip_index=cis[m], v0=v0[m], **{k: v[m] for k, v in arms.items()})
        files.append(f)
    return files


def analyze_dumps(files, header: dict, n_boot: int):
    t1 = load_t1_eval()
    res = t1.analyze(files, tiers={"fl": "T1", "ha0": "T1"}, n_boot=n_boot, dt=DT,
                     paired=[("ol", "cl", "paired_t1_minus_open"), ("fl", "cl", "paired_model_minus_floor_t1")])
    res["_tier_doctrine"] = TIER_DOCTRINE                                # t1_eval still prints the pre-amendment 'action-closed loop'
    res["t1v8f"] = header
    return res


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--tracks", required=True, help="native_tracks.pt (yaw mode needs the 10 Hz track)")
    p.add_argument("--eval-cache", default="C:/Users/Admin/refav1_hj_cache")
    p.add_argument("--eval-grid", required=True)
    p.add_argument("--eval-join", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--replan-steps", type=int, default=1, help="m: cache steps (0.2 s) executed per plan; must divide 30. m=30 == the one-shot open loop")
    p.add_argument("--schedule", choices=SCHEDULES, default="periodic")
    p.add_argument("--nav", choices=NAV_MODES, default="deadreckon")
    p.add_argument("--scene", choices=SCENE_MODES, default="logged", help="logged = the logged frames (default); frozen = the TICK-0 feats for the whole rollout; hold_now (FRAMEFREEZE) = every history row of the feats is the CURRENT frame's row, repeated (no frame motion); reindexed (REINDEXED) = the history rows after the window's t0 take the logged frame at the along-track index of the model's OWN executed position, min(i, tau*), rows up to t0 stay logged")
    p.add_argument("--freeze-card-ref", default="", help="FRAMEFREEZE: free text recorded in the run header (the RECOVER-FT card's frozen sha line); only with --scene hold_now")
    p.add_argument("--limit", choices=LIMIT_MODES, default="logged")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lru", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=0, help="smoke: evenly-spaced subset of the grid (0 = all)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--dump-ticks", default="", help="PREREG_V8F_HACC_LAT: a directory; writes ticks_<tag>.npz = one row per (window, tick) of the model's rollout (tau, P, plan, v0, GT in the executed frame, ids)")
    p.add_argument("--feasible", choices=FEASIBLE_MODES, default="off", help="KAPPA-T1 / FEAS2-T1: feasible_decode.project_feasible on the plan AFTER --smooth, at every replan, before the executor (dt=DT, mu=None): box = A_MAX 4.0 and KAPPA_MAX 0.2, clamp_entry=False; kbox = KAPPA_MAX 0.2 only (a_max=inf); abox = A_MAX 4.0 only (kappa_max=inf) with clamp_entry=True and v0 = the tick's measured ego speed; kabox = kbox then abox; off (default) = unchanged bit for bit")
    p.add_argument("--smooth", choices=SMOOTH_MODES, default="off", help="SMOOTH-T1: sg72 = PREREG_V8F_SMOOTH's S(plan) = P + SG(plan - P) (window 7, order 2, interp) on the model's plan at every replan, before the executor; off (default) = unchanged bit for bit")
    p.add_argument("--histswap", choices=HISTSWAP_MODES, default="off", help="HISTSWAP-LF: cv_exec = a constant-twist (constant-velocity) history at the EXECUTED current speed (deployable); cv_logged = the same at the LOGGED current speed (ORACLE, level only); both need --limit causal. HISTSWAP (H-HISTSWAP-1): logged = the model's ego-history input at every tick (p10f, base, kin3, pose*, the causal speed bucket) is read from the LOGGED clip arrays instead of the executed ones; the plan is still committed to the executed state; nav, scene, ages, --smooth and the P10 floor untouched. ORACLE diagnostic, not deployable. off (default) = unchanged bit for bit; refuses to combine with --feasible")
    p.add_argument("--level-floor", choices=("off", "on"), default="off", help="LEVEL-FLOOR (PREREG_LEVELFLOOR_A3): on = the floor operator F (level_floor.py) lifts the plan's along-track progress to kappa * v_hat from the floor heads, at every replan, after --smooth and before --feasible; off (default) = unchanged bit for bit")
    p.add_argument("--floor-kappa", type=float, default=None, help="LEVEL-FLOOR: the dose kappa in [0, 1]; required with --level-floor on, refused with off")
    p.add_argument("--floor-ckpt", default="", help="LEVEL-FLOOR: the head checkpoint (cfg: level_head on AND lead_head on); required with --level-floor on, refused with off")
    p.add_argument("--floor-vhat-source", choices=("heads", "logged"), default=None, help="LFO (PREREG_LEVELFLOOR_A4): the floor's level input v_hat. heads (the default with --level-floor on) = the floor heads' estimate; logged = the LOGGED route speed of the tick (a CEILING diagnostic, NOT admissible at inference); refused without --level-floor on")
    p.add_argument("--floor-vhat-noise", type=float, default=None, help="LF5 (PREREG_LEVELFLOOR_A5): SIGMA, v_hat := v_log * exp(SIGMA * eps), eps ~ N(0,1); only with --floor-vhat-source logged; default (absent) = no noise; SIGMA > 0 needs --floor-vhat-noise-seed")
    p.add_argument("--floor-vhat-noise-seed", type=int, default=None, help="LF5: the seed of the ONE numpy.random.default_rng of the rollout that draws eps (one vector draw per replan, in replan order); only with --floor-vhat-source logged")
    p.add_argument("--floor-lift-cap", type=float, default=None, help="LF6 (PREREG_LEVELFLOOR_A6): DELTA >= 0 in m/s, the bounded-lift floor F_B, v_f(tau) = min(kappa * v_hat, v0 + 1.0 * tau, v_O(tau) + DELTA); needs --level-floor on; default (absent) = A5's F, the LF5 tool bit for bit")
    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS (needs --dump-ticks): add the S head and T head outputs of each tick's forward_infer call to those rows (s_ckpt*, t_goal_prob, t_lat_logits, t_lon_logits, t_cons_pred + poses)")
    p = sub.add_parser("analyze")
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    return ap


def cmd_run(a):
    import torch
    if a.freeze_card_ref and a.scene != "hold_now":
        raise SystemExit("REFUSED: --freeze-card-ref is recorded only under --scene hold_now (a flag that silently writes nothing is worse than none)")
    FREEZE_STATS.update({"items": 0, "max_row_dev": 0.0, "n_unequal_elements": 0})
    ri_reset()
    if a.dump_heads and not a.dump_ticks:
        raise SystemExit("REFUSED: --dump-heads adds columns to the --dump-ticks file; it needs --dump-ticks DIR (a flag that silently writes nothing is worse than none)")
    if a.histswap != "off" and a.feasible != "off":
        raise SystemExit("REFUSED: --histswap logged is not defined together with --feasible (abox / kabox read the tick's input speed; the card has no such arm)")
    if a.histswap in CV_MODES and a.limit != "causal":
        raise SystemExit(f"REFUSED: --histswap {a.histswap} is defined with --limit causal only (got {a.limit})")
    lf_check_args(a)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    if cfg.kin_mode != "yaw":
        raise SystemExit("REFUSED: T1-kin rebuilds the ego state from the executed pose track; a --kin-mode steer checkpoint reads the "
                         "recorded steering angle (PI ruling D-PI-YAWRATE-1) and has no own-motion equivalent.")
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    dev = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    floor = load_level_floor(a, cfg, ck, dev) if a.level_floor == "on" else None
    tracks = D.load_native_tracks(a.tracks)
    J = D.load_join(a.eval_join)
    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    if floor is not None and floor.vhat_source == "logged":              # LFO: the oracle reads the same store the tick dump reads
        floor.oracle = make_logged_oracle(store, J)
    wins = D.eval_grid_windows(a.eval_grid, J)
    if a.max_windows and a.max_windows < len(wins):
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, a.max_windows).round().astype(int)]
    proxy = _OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                          window_start=cfg.window_start)     # PREREG_V8F_HIST Amendment 1: the held-age clamp origin is the CHECKPOINT's (-1 = absent = hist - 1)
    kw = dict(m=a.replan_steps, schedule=a.schedule, scene=a.scene, limit=a.limit, device=dev)
    cl, ol, fl, ha0, g, v0, keys = [], [], [], [], [], [], set()
    n_clip = {"cl": 0, "fl": 0}
    rec = None
    t_start = time.time()
    for b0 in range(0, len(wins), a.batch):
        chunk = wins[b0:b0 + a.batch]
        mp = model_planner(model, dev, keep_aux=bool(a.dump_ticks), keep_heads=bool(a.dump_heads), smooth=a.smooth, feasible=a.feasible, floor=floor)
        st_m = make_states(store, tracks, J, chunk, a.nav, hist_swap=a.histswap)     # the P10 floor's states below are never swapped
        if a.dump_ticks:
            rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth, feasible=a.feasible, histswap=a.histswap, level_floor=floor is not None, floor_oracle=floor is not None and floor.vhat_source == "logged", floor_noise=floor is not None and floor.noise_path, floor_bound=floor is not None and floor.lift_cap is not None)
            rec.planner, rec.base = mp, b0
            plan0, exe = rollout(mp, st_m, ds, proxy, hook=rec, **kw)
        else:
            plan0, exe = rollout(mp, st_m, ds, proxy, **kw)
        keys |= mp.keys_read
        st_f = make_states(store, tracks, J, chunk, a.nav)
        _, exe_f = rollout(floor_planner(model, dev), st_f, ds, proxy, **kw)
        n_clip["cl"] += sum(s.clipped > 0 for s in st_m)
        n_clip["fl"] += sum(s.clipped > 0 for s in st_f)
        cl += exe
        fl += exe_f
        ol.append(plan0)
        for ci, t0, _, _ in chunk:
            d = store.get(str(J["clip"][ci]))
            g.append(d["gt"][t0])
            v0.append(d["kin3"][t0][0])                                  # the ego speed AT t0 (logged history, allowed)
        print(f"[{b0 + len(chunk)}/{len(wins)}] {time.time() - t_start:.0f}s", flush=True)
    v0 = np.asarray(v0, np.float64)
    arms = {"cl": np.asarray(cl), "ol": np.concatenate(ol).astype(np.float64), "fl": np.asarray(fl),
            "ha0": fk.ha0_paths(v0, DT, H)}
    gt = np.asarray(g, np.float64)
    files = write_dumps(os.path.join(a.out_dir, f"dump_{a.tag}"), wins, arms, gt, v0)
    header = {"tag": a.tag, "ckpt": a.ckpt, "kin_mode": cfg.kin_mode, "tier": "T1 (self-action OPEN loop; ego state = own executed motion)",
              "replan_steps": a.replan_steps, "replan_hz": 5.0 / a.replan_steps, "schedule": a.schedule, "scene": a.scene, "nav": a.nav,
              "limit": a.limit, "n_windows": len(wins), "model_keys_read": sorted(keys),
              "windows_with_envelope_clip": n_clip, "envelope_ms": V_ENV_MS,
              "strict": {"kinematics": True, "route": a.nav == "deadreckon", "camera": a.scene == "frozen", "speed_bucket": a.limit != "logged"},
              "open_channels": ["camera stream (logged frames after t0 carry the human's ego-motion)" if a.scene == "logged" else "",
                                "route countdown from the logged human position" if a.nav == "logged" else "",
                                "speed bucket from the human's realised future max speed (D-SPEEDBAND-EGO-ECHO-1)" if a.limit == "logged" else ""],
              "acts": {k: fk.acts_check(v, DT) for k, v in arms.items() if k in ("cl", "fl", "ol")}}
    header["open_channels"] = [c for c in header["open_channels"] if c]
    if a.scene == "hold_now":                                            # only then: every other --scene value => the header (and the result JSON) is byte-identical to the HISTSWAP-LF tool's
        header["camera_motion"] = False
        header["open_channels"].append("camera stream: the CURRENT frame is logged (it depicts the human's position at the tick; a position or blur cue survives); the hist frames are that frame repeated, so no frame motion reaches the network")
        header["frame_freeze"] = {"scene": "hold_now", "applied": "every row of the item's feats := the current frame's row (feats[k] := feats[-1]), right after the item is built, at every tick of every window; frame_ok unchanged",
                                  "items_frozen": FREEZE_STATS["items"], "max_row_dev": FREEZE_STATS["max_row_dev"], "n_unequal_elements": FREEZE_STATS["n_unequal_elements"],
                                  "card_ref": a.freeze_card_ref or "UNVERIFIED: no --freeze-card-ref was passed",
                                  "prereg": "dagland/histswap/DRAFT_RECOVERY_CARD_v2.md section 8.3 (frame-freeze control); reading thresholds frozen with dagland/recover_ft"}
    if a.scene == "reindexed":                                           # only then: every other --scene value => the header (and the result JSON) is byte-identical to the FRAMEFREEZE tool's
        header["open_channels"].append("camera stream: REINDEXED, an approximate longitudinal re-perception: the history rows after t0 are the logged frames at the along-track index of the model's own executed position, "
                                       "min(i, tau*); static scene geometry consistent with the own position, dynamic agents shown at the human's time, lateral offset ignored; NOT T2")
        pr = RI_STATS["past"]
        header["scene_reindex"] = {"scene": "reindexed", **ri_summary(RI_STATS["model"]), "floor": ri_summary(RI_STATS["floor"]),
                                   "past_rows_checked": pr["rows_checked"], "past_rows_changed": pr["rows_changed"], "past_rows_bit_identical": bool(pr["rows_checked"] > 0 and pr["rows_changed"] == 0),
                                   "rule": "rows i <= t0 stay logged (byte-compared); rows i > t0: tau(i) = min(i, tau*), tau* = the logged cache index nearest in arc length to the projection of the model's executed (x, y) at i on the logged polyline "
                                           "(ties and a dwell under dwell_m -> the latest index), tau non-decreasing over the item's rows; the row takes the logged feats[tau] and the frame_ok of that index",
                                   "dwell_m": RI_DWELL_M, "role_note": "top level = the MODEL's rollout (the ticks the network reads); 'floor' = the P10 floor's rollout, which reads no frames",
                                   "limits": "longitudinal only: the lateral offset of the own position from the logged path is ignored (reported as lateral_offset_m); dynamic agents are shown at the human's time; the current frame still depicts a logged "
                                             "position (the along-track one of the own position) and T2 is not provisioned"}
    if a.limit in ("causal", "causal_all"):                              # only then: --limit logged / off outputs stay byte-identical (C-L1)
        header["causal_bucket"] = {"rule": "speed_max_bin(max finite measured ego speed over rows "
                                           + (f"[j - {CAUSAL_STEPS}, j])" if a.limit == "causal" else "[0, j])"),
                                   "stats_model_ticks": CAUSAL_STATS}
    if a.smooth != "off":                                                # only then: --smooth off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["smooth"] = {"mode": a.smooth, "operator": f"P + savgol_filter(plan - P, {SG_WINDOW}, {SG_ORDER}, axis=1, mode='interp')", "prereg": "PREREG_V8F_SMOOTH (sha256 a2f19e48...)",
                            "applied": "to the model's plan at every replan, before the executor; the P10 floor is not smoothed"}
    if a.feasible != "off":                                              # only then: --feasible off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["feasible"] = {"mode": a.feasible, "operator": FEASIBLE_OPERATORS[a.feasible],
                              "prereg": "dagland/kappa_t1/PREREG_FROZEN.md (H-KAPPA-T1-1)" if a.feasible == "box" else "dagland/feas2_t1/PREREG_FROZEN.md (H-FEAS2-T1-1)",
                              "applied": "to the plan after --smooth at every replan, before the executor; the P10 floor is not projected"}
    if a.histswap != "off":                                              # only then: --histswap off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["histswap"] = {"mode": a.histswap, "applied": "the model's ego-history input at every tick comes from the LOGGED clip arrays; the plan is committed to the EXECUTED state; the P10 floor, nav, scene, ages and --smooth are untouched",
                              "oracle": "yes: the logged history after t0 is the human's; NOT deployable", "prereg": "dagland/histswap/PREREG_HISTSWAP_FROZEN.md (H-HISTSWAP-1)",
                              "bucket_stats_model_ticks": {k: CAUSAL_STATS.get(k, 0) for k in ("swap_ticks", "swap_bucket_differs")}}
    if a.histswap in CV_MODES:                                           # LF: the header says what the history IS
        header["histswap"].update({"applied": "the model's ego-history input at every tick is a CONSTANT-TWIST history (speed and yaw rate held, zero acceleration) anchored at the EXECUTED pose; the plan is committed to the EXECUTED state; "
                                              "the P10 floor, nav, scene, ages and --smooth are untouched",
                                   "speed_source": "the EXECUTED measured speed at the tick (deployable)" if a.histswap == "cv_exec" else "the LOGGED measured speed at the tick (ORACLE, level only: the human's speed, no trend)",
                                   "oracle": "no" if a.histswap == "cv_exec" else "yes: the logged speed after t0 is the human's; NOT deployable",
                                   "prereg": "dagland/histswap_lf/PREREG_HISTSWAP_LF_FROZEN.md (H-HISTSWAP-LF-1)"})
    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path = os.path.join(a.dump_ticks, f"ticks_{a.tag}.npz")
        rec.save(tick_path, {"split": "eval", "tag": a.tag, "ckpt": a.ckpt, "replan_steps": a.replan_steps, "n_windows": len(wins), "clip_names": np.asarray(J["clip"]),
                             **({"smooth": a.smooth} if a.smooth != "off" else {}), **({"feasible": a.feasible} if a.feasible != "off" else {}),
                             **({"histswap": a.histswap} if a.histswap != "off" else {}), **(floor.meta() if floor is not None else {}),
                             **({"scene": a.scene} if a.scene in ("hold_now", "reindexed") else {})})
        header["ticks_file"] = tick_path
        header["ticks_heads"] = bool(a.dump_heads)
    if floor is not None:                                                # only then: --level-floor off / absent => the header (and the result JSON) is byte-identical to the RI tool's
        header["level_floor"], header["floor_kappa"], header["floor_ckpt"], header["floor_ckpt_sha256"] = "on", floor.kappa, floor.ckpt_path, floor.ckpt_sha256
        header["level_floor_detail"] = floor.detail()
        header["floor_vhat_source"] = floor.vhat_source
        if floor.lift_cap is not None:
            header["floor_lift_cap"] = floor.lift_cap
        if floor.vhat_source == "logged":
            header["floor_vhat_noise"], header["floor_vhat_noise_seed"] = floor.noise, floor.noise_seed
            header["label"] = "CEILING"
            header["open_channels"].append("LEVEL INPUT (CEILING, inadmissible at inference): the floor's v_hat is the LOGGED route speed of the tick (a privileged channel); no complete logged route = no floor")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.no_analyze:
        json.dump(header, open(os.path.join(a.out_dir, f"t1v8f_{a.tag}.json"), "w"), indent=1)
        return 0
    res = analyze_dumps(files, header, a.n_boot)
    out = os.path.join(a.out_dir, f"t1v8f_{a.tag}.json")
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out)
    return 0


def cmd_analyze(a):
    files = sorted(os.path.join(a.dump_dir, f) for f in os.listdir(a.dump_dir) if f.startswith("ep") and f.endswith(".npz"))
    res = analyze_dumps(files, {"note": "re-scored from existing dumps; run header not available"}, a.n_boot)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"run": cmd_run, "analyze": cmd_analyze}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
