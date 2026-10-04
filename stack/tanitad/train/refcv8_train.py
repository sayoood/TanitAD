"""refcv8 WP-B -- the TRAINER side of the tactical conditioning (DESIGN.md sec. 3; the model side is
`tanitad/refs/refcv8_conditioning.py` + the seams in `refc.py` / `refc_v3.py` / `refcv6_tactical.py`).

Four things live here so the 11k-line trainer only CALLS them:

  1. :func:`r8_before_forward` -- the per-batch inputs the model needs BEFORE its forward: the GT hypothesis and
     the constraint targets (labels may use ego, PI 2026-08-03), the scheduled-sampling coins (teacher forcing of
     the GT constraint into the allocated candidates' generation), and the route-checkpoint / nav-argument inputs
     with their TRAINING treatment (MM binding 2026-10-04 + WP-A INTEGRATION.md sec. 3.2): route-checkpoint dropout
     >= 0.3 AND the registered along / lateral noise on the kept rows (the clean point leaks, E2'); nav-argument
     dropout with the token KEPT (a leaderboard-legal NavSim agent receives the bare command only).
  2. :func:`r8_losses` -- constraint-head loss, matched L1 on the allocated candidates, L_sat, the listwise
     selection loss (X1, replacing the single-winner E9 CE when on), the Hydra-style sub-score BCE (X1h).
  3. The v9 label release (WP-A, ``tanitad/data/v9_labels.py``): :class:`R8LabelJoin` (the per-window join on the
     trainer's OWN clock), :func:`v9_integrity_census` (the release contract the trainer relies on, checked at load),
     :func:`v9_goal_census` (the goal pos_weight / class mask FROM THE v9 TRAIN WINDOWS), :func:`partial_label_loss`
     (the ``-log sum_{c in allowed} p_c`` loss the v9 partial labels need), :func:`navsim_legal_nav_cmd`.
  4. LABEL-STATE ISOLATION (D1 F2 / X4). With WP-A's ``v7_labels`` fix the goal-negative policy TRAVELS on the label
     objects (``V7Label.goal_geometry_tokens``) and :func:`v7_policy_travels` is True, so the trainer installs no
     scope at all. :class:`V7PolicyScope` is the fallback for an UNFIXED module only (snapshot at load, applied
     around each target computation).

⛔ Every random draw here uses the decoder's DEDICATED generator (`decoder.r8_gen`), never the global RNG.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.refs import refcv8_conditioning as r8c

__all__ = ["r8_before_forward", "r8_losses", "V7PolicyScope", "R8LabelJoin", "tf_ratio", "RC_DROPOUT_MIN",
           "RC_NOISE_ALONG_M", "RC_NOISE_LAT_M", "V9_RELEASE_MD5", "GEOMETRY_GOALS", "scale_nav", "scale_rc",
           "rc_training_noise", "v9_integrity_census", "v9_goal_census", "partial_label_loss",
           "navsim_legal_nav_cmd", "v7_policy_travels", "load_v9_join", "R8_SPEED_UNKNOWN_P", "R8_SPEED_DERIVATION",
           "assert_r8_speed_stamp", "speed_kmh_of", "speed_input_census", "speed_input_treatment", "tactical_rows", "I2_LITERALS", "emit_live",
           "apply_emit_schedule", "emits_at_step0", "checkpoint_r8_step", "i2_identity_row",
           "speed_derivation", "R8_SPEED_ENC8_NOTE", "speed_gen", "speed_crc", "speed_only_fwd"]

#: MM binding 2026-10-04 (i): route-checkpoint dropout >= 0.3
RC_DROPOUT_MIN = 0.3
#: WP-A INTEGRATION.md sec. 3.2 / 4 (E2' CONFIRMED at exactly these): the training noise on a KEPT route checkpoint,
#: in the route-tangent frame. ⛔ The clean point carries +0.021 more future-speed information than the road-level
#: route (FAILS E2'); the noised one -0.035 (passes). Smaller values are not certified and are refused.
RC_NOISE_ALONG_M = 2.0
RC_NOISE_LAT_M = 0.75
#: the released files (WP-A LANDING_READY_WPA.txt ## WPA-S3); another release must be named by md5 explicitly
V9_RELEASE_MD5 = {"train": "f63ece410b725febb8a5242cf2b01d3c", "eval139": "6b5c7f207cffc3b7eb3cd527fd433599"}
#: v9 SPEC sec. 4.1: the GEOMETRY goals (dense, from the ego log). A reversing window carries none of them; the VLM
#: goals (sec. 4.3) keep their own timing rule.
GEOMETRY_GOALS = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "STOP_POINT",
                  "SPEED_BAND")
#: v9 / INTEGRATION sec. 8: absence claims need the full 8-s band (V7). Frozen v7 ids.
ABSENCE_LAT_V7 = (0,)                 # LANE_KEEP
ABSENCE_LON_V7 = (1, 4, 5)            # CRUISE (= v9 KEEP), CREEP, HOLD
LC_LAT_V7 = (1, 2)                    # LANE_CHANGE_L / _R -- never an exact class in this release
H_ABS_MIN_S = 8.0
#: model input widths (`refc_v3.R8_NAV_DIMS` / `R8_RC_DIMS` must agree; pinned in the tests)
NAV_IN_DIMS = 6                       # log1p(d_next/50), log1p(d_end/50), dyaw/90, side, log1p(lookahead/50), known
RC_IN_DIMS = 4                        # x/50, y/50, psi/90, valid  (NOW frame, RC-A50 by default)
NAV_D_CLIP_M = 1000.0
#: INTEGRATION.md sec. 5 (Master Mind 2026-10-04): NavSim `driving_command` (left, forward, right, unknown) -> our
#: nav_cmd, with nav_args_known = 0 and route_cp_valid = 0 on the LEGAL row.
NAVSIM_LEGAL_NAV_CMD = {"left": 1, "forward": 0, "right": 2, "unknown": 0}

# ---- X3 (SPEC_REFCV8 sec. 8.3; MM ruling Q5): the PAST-ONLY speed input -------------------------------------------- #
#: the "unknown" row's TRAINING rate (D4 N2u; the NavSim navtest no-limit share, INHERITED). refcv7 fed the all-zero
#: row on ~45 % of navtest tokens and NEVER trained it; here it is a trained input.
R8_SPEED_UNKNOWN_P = 0.45
#: the v9 release column each source reads (WP-A V9_SCHEMA: int16 km/h / int8 code; -1 = not computable)
R8_SPEED_COLUMN = {"n2": "speed_n2_kmh", "n3": "speed_n3"}
#: N2's road-law ladder (WP-A SPEC_ADDENDUM_S3A1 item 4 = D4's), LITERAL: a joined value outside it (or -1) refuses
R8_SPEED_N2_LADDER_KMH = (20, 30, 50, 70, 80, 100, 120, 130)
#: N3 code -> the km/h it is FED as: D4's own N3 ceiling (`d4_vmax_nonoracle.py`: urban 50 / rural 100 / motorway 130)
R8_SPEED_N3_KMH = {0: 50.0, 1: 100.0, 2: 130.0}
#: ⛔ THE STAMP (config.json `r8_speed_derivation`). It names the SOURCE COLUMN, the WINDOW, the word past-only, the
#: LADDER, the channel and the unknown row, so a reader of config.json alone learns what fed the channel.
R8_SPEED_DERIVATION = {
    "n2": ("refcv8 X3 past-only speed input N2: v9 `speed_n2_kmh` = the ego's realised max speed over "
           "[NOW - 20 s, NOW] (past samples only), snapped UP to the road-law ladder {20, 30, 50, 70, 80, 100, 120, "
           "130} km/h with an urban floor 50 km/h (D4 d4_vmax_nonoracle.py; WP-A SPEC_ADDENDUM_S3A1 item 4); fed "
           "through the refcv6 4-way one-hot {30, 50, 100, 120} km/h (CONTAINING-WINDOW); a trained unknown row "
           "(all-zero) on p of the TRAINING windows; the ceiling on the EMITTED plan is the bin limit and never below "
           "the fed value. D4 LEAK 0.0346 (8-step) against the 0.05 bar. No future sample is read."),
    "n3": ("refcv8 X3 past-only speed input N3: v9 `speed_n3` = urban / rural / motorway from N2 (the ego's realised "
           "max speed over [NOW - 20 s, NOW], past samples only; N2 <= 50 urban, <= 100 rural, else motorway), fed "
           "as 50 / 100 / 130 km/h through the refcv6 4-way one-hot {30, 50, 100, 120} km/h (CONTAINING-WINDOW); a "
           "trained unknown row (all-zero) on p of the TRAINING windows; the ceiling on the EMITTED plan is the bin "
           "limit and never below the fed value. D4 LEAK 0.0193 against the 0.05 bar. No future sample is read."),
}
#: the tokens each stamp must carry -- LITERALS, never an expression over the text above (a check derived from the
#: value it checks is green forever)
R8_SPEED_STAMP_REQUIRED = {
    "n2": ("past-only", "[NOW - 20 s, NOW]", "speed_n2_kmh", "{20, 30, 50, 70, 80, 100, 120, 130} km/h",
           "urban floor 50 km/h", "4-way one-hot", "unknown row", "EMITTED plan"),
    "n3": ("past-only", "[NOW - 20 s, NOW]", "speed_n3", "urban / rural / motorway", "4-way one-hot", "unknown row",
           "EMITTED plan"),
}


#: refcv8 (B) `--r8-speed-enc8`: appended to the N2 stamp; its tokens are required when the encoding is on
R8_SPEED_ENC8_NOTE = (" + refcv8 (B): the FULL N2 ladder as an 8-step one-hot + known bit through a zero-init FiLM "
                      "seam on the behaviour decoder's queries; with it the ceiling is the fed N2 value itself.")
R8_SPEED_ENC8_REQUIRED = ("8-step one-hot", "zero-init FiLM", "the ceiling is the fed N2 value")


def speed_derivation(mode, enc8: bool = False):
    """The config.json `r8_speed_derivation` text for a source (None when off)."""
    if not mode:
        return None
    return R8_SPEED_DERIVATION[str(mode)] + (R8_SPEED_ENC8_NOTE if enc8 else "")


def assert_r8_speed_stamp(cfg_dict: dict, mode, enc8: bool = False) -> None:
    """⛔ REFUSE a run whose config.json would not DECLARE its past-only speed source -- and the MIRROR, a run stamped
    with one while the source is off (that manufactures an X3 arm out of a control). Same shape as
    ``refcv6_max_speed.assert_speed_max_stamp_v6``, on its own key so neither stamp can satisfy the other guard."""
    stamp = cfg_dict.get("r8_speed_derivation")
    mode = str(mode or "")
    if not mode:
        if stamp is not None:
            raise SystemExit("[refcv8] ⛔ config carries `r8_speed_derivation` but no --r8-speed-input: a run that "
                             "fed no past-only speed must not be stamped as one")
        return
    if mode not in R8_SPEED_STAMP_REQUIRED:
        raise SystemExit(f"[refcv8] ⛔ --r8-speed-input {mode!r} is not one of {sorted(R8_SPEED_STAMP_REQUIRED)}")
    if not isinstance(stamp, str) or not stamp.strip():
        raise SystemExit("[refcv8] ⛔ --r8-speed-input is on but config.json would carry no `r8_speed_derivation`")
    missing = [t for t in R8_SPEED_STAMP_REQUIRED[mode] + (R8_SPEED_ENC8_REQUIRED if enc8 else ()) if t not in stamp]
    if missing:
        raise SystemExit(f"[refcv8] ⛔ `r8_speed_derivation` does not declare {missing} (X3 stamp, {mode})")


def speed_kmh_of(rel, r: int, mode: str) -> tuple[float, bool]:
    """``(km/h fed, known)`` of release row ``r`` for source ``mode``. -1 (or any negative) is the release's 'not
    computable' marker and reads as UNKNOWN; any other value off the ladder raises (the census refuses it at enable)."""
    if mode == "n2":
        v = int(rel.rows["speed_n2_kmh"][r])
        if v < 0:
            return 0.0, False
        if v not in R8_SPEED_N2_LADDER_KMH:
            raise ValueError(f"[refcv8] speed_n2_kmh {v} is not on the ladder {R8_SPEED_N2_LADDER_KMH}")
        return float(v), True
    if mode == "n3":
        c = int(rel.rows["speed_n3"][r])
        if c < 0:
            return 0.0, False
        if c not in R8_SPEED_N3_KMH:
            raise ValueError(f"[refcv8] speed_n3 {c} is not one of {sorted(R8_SPEED_N3_KMH)}")
        return R8_SPEED_N3_KMH[c], True
    raise ValueError(f"[refcv8] speed source {mode!r} is not 'n2' or 'n3'")


def speed_input_census(join, rows, mode: str) -> dict:
    """The fed channel over the dataset's joined rows: the km/h shares, the 4-way bin shares (through the MODEL's own
    `speed_max_bin_tensor`), the over-ceiling share, and the known share. REFUSES a column the release lacks, a value
    off the ladder, and a split where no window is known (a JOIN failure, not a channel -- the v6 rule)."""
    from tanitad.refs import refcv6_max_speed as v6ms
    col = R8_SPEED_COLUMN.get(str(mode))
    if col is None:
        raise SystemExit(f"[refcv8] --r8-speed-input {mode!r} is not one of {sorted(R8_SPEED_COLUMN)}")
    if col not in join.rel.rows:
        raise SystemExit(f"[refcv8] ⛔ --r8-speed-input {mode}: the v9 release {join.rel.path} has no `{col}` column")
    rows = np.asarray(rows, dtype=np.int64)
    raw = np.asarray(join.rel.rows[col]).astype(np.int64)[rows]
    allowed = set(R8_SPEED_N2_LADDER_KMH) if mode == "n2" else set(R8_SPEED_N3_KMH)
    bad = sorted({int(x) for x in np.unique(raw) if x >= 0 and int(x) not in allowed})
    if bad:
        raise SystemExit(f"[refcv8] ⛔ `{col}` carries values off its ladder on the joined rows: {bad[:8]}")
    known = raw >= 0
    if len(rows) and not known.any():
        raise SystemExit(f"[refcv8] ⛔ --r8-speed-input {mode}: NOT ONE of {len(rows)} joined windows carries a "
                         f"value -- the channel would be a constant unknown row")
    kmh = np.array([float(x) if mode == "n2" else R8_SPEED_N3_KMH.get(int(x), 0.0) for x in raw], np.float64)
    ms = torch.tensor([k / 3.6 for k in kmh[known]], dtype=torch.float32)     # python float64 / 3.6, as the item
    n = max(len(rows), 1)
    out = {"mode": str(mode), "column": col, "n_windows": int(len(rows)), "n_known": int(known.sum()),
           "known_frac": round(float(known.sum()) / n, 6),
           "value_shares": {str(int(u)): round(float((raw == u).sum()) / n, 6) for u in np.unique(raw)}}
    if len(ms):
        idx, over = v6ms.speed_max_bin_tensor(ms)
        out["bin4_kmh"] = list(v6ms.SPEED_MAX_STEPS_KMH_V6)
        out["bin4_shares_of_known"] = [round(float((idx == i).sum()) / len(ms), 6)
                                       for i in range(v6ms.N_SPEED_MAX_BINS_V6)]
        out["over_ceiling_frac_of_known"] = round(float(over.sum()) / len(ms), 6)
    return out


def speed_input_treatment(v_ms: Tensor, valid: Tensor, gen, *, training: bool, unknown_p: float = R8_SPEED_UNKNOWN_P,
                          roll: bool = False, eval_intervention=None, legal: bool = False) -> tuple[Tensor, Tensor]:
    """X3's per-batch treatment of the fed speed (value m/s [B], valid [B]); returns float32 (value, valid).

    * TRAINING: the "unknown" row with probability ``unknown_p`` (the dedicated generator; refused outside (0, 1)),
      then -- the L4 deliberate-regression arm V-VSHUF only -- every row takes ANOTHER window's (value, valid)
      (roll by one row; the input's distribution is kept, its information about this window is removed).
    * EVAL: nothing is drawn. ``eval_intervention`` "off" = the unknown row everywhere (VMAX-OFF), "shuf" = another
      window's value (VMAX-SHUF; refused on a batch of one, where a roll is the identity).
    * ``legal`` (the NavSim-LEGAL row, MM ruling Q3): the unknown row.
    An unknown row carries NO value (zeros next to a 0: the X15 rule)."""
    vm = v_ms.to(torch.float32).reshape(-1).clone()
    vv = valid.to(torch.float32).reshape(-1).clone()
    b = vm.shape[0]
    if training:
        p = float(unknown_p)
        if not 0.0 < p < 1.0:
            raise SystemExit(f"[refcv8] the speed input's unknown-row rate {p} must be in (0, 1): the unknown row is a "
                             f"TRAINED input (X3) and a known row must exist")
        drop = gen.rand((b,), vm.device) < p
        vv = torch.where(drop, torch.zeros_like(vv), vv)
        if roll and b > 1:
            vm, vv = torch.roll(vm, 1, 0), torch.roll(vv, 1, 0)
    elif eval_intervention in ("off",):
        vv = torch.zeros_like(vv)
    elif eval_intervention in ("shuf",):
        if b < 2:
            raise ValueError("[refcv8] VMAX-SHUF on a batch of one: a roll is the identity -- evaluate with batch >= 2")
        vm, vv = torch.roll(vm, 1, 0), torch.roll(vv, 1, 0)
    elif eval_intervention not in (None, ""):
        raise ValueError(f"[refcv8] unknown speed eval intervention {eval_intervention!r} (None / 'off' / 'shuf')")
    if legal:
        vv = torch.zeros_like(vv)
    return vm * vv, vv


def tf_ratio(step: int, steps: int, start: float, end: float) -> float:
    """Scheduled sampling (1506.03099 p4, the LINEAR form): the teacher-forcing ratio at ``step``."""
    if steps <= 1:
        return float(end)
    a = min(max(step / float(steps - 1), 0.0), 1.0)
    return float(start + (end - start) * a)


def _gen(model):
    g = getattr(model.core.decoder, "r8_gen", None)
    if g is None:
        raise ValueError("[refcv8] the decoder carries no dedicated generator -- the seams are not built")
    return g


# ================================================================================================================= #
# input scaling and the training treatment of the two supplied inputs                                               #
# ================================================================================================================= #
def scale_nav(raw: Tensor, known: Tensor) -> Tensor:
    """v9 ``nav_args`` [.., 5] = (d_next m, d_end m, dyaw_next deg, side +-1, lookahead m) + known -> the model's
    [.., 6]. Distances: ``log1p(min(d, 1000) / 50)`` (~d/50 near, compressed far: the plan horizon is <= ~180 m while
    the release carries up to 3,162 m). Unknown rows are exactly zeros next to a 0 (the model re-applies the bit)."""
    raw = raw.to(torch.float32)
    k = known.to(torch.float32).reshape(*raw.shape[:-1], 1)

    def lg(d):
        return torch.log1p(d.clamp(0.0, NAV_D_CLIP_M) / 50.0)
    out = torch.stack([lg(raw[..., 0]), lg(raw[..., 1]), raw[..., 2] / 90.0, raw[..., 3].clamp(-1.0, 1.0),
                       lg(raw[..., 4])], -1)
    return torch.cat([out * k, k], -1)


def scale_rc(raw: Tensor, valid: Tensor) -> Tensor:
    """v9 ``route_cp`` [.., 3] = (x m, y m, psi deg) in the NOW frame + valid -> the model's [.., 4]."""
    raw = raw.to(torch.float32)
    v = valid.to(torch.float32).reshape(*raw.shape[:-1], 1)
    out = torch.stack([raw[..., 0] / 50.0, raw[..., 1] / 50.0, raw[..., 2] / 90.0], -1)
    return torch.cat([out * v, v], -1)


def rc_training_noise(raw: Tensor, valid: Tensor, gen, *, sigma_along_m: float = RC_NOISE_ALONG_M,
                      sigma_lat_m: float = RC_NOISE_LAT_M) -> Tensor:
    """The registered training noise on the KEPT checkpoints: along-track N(0, sigma_along), lateral N(0, sigma_lat)
    in the ROUTE-TANGENT frame (the tangent is the checkpoint's own heading psi). psi itself is not noised. Invalid
    rows are returned unchanged (they are zeros and stay zeros). ``gen`` is the decoder's ``R8Generator``."""
    raw = raw.to(torch.float32).clone()
    b = raw.shape[0]
    n = gen.randn((b, 2), raw.device) * torch.tensor([sigma_along_m, sigma_lat_m], device=raw.device)
    psi = torch.deg2rad(raw[:, 2])
    c, s = torch.cos(psi), torch.sin(psi)
    dx = n[:, 0] * c - n[:, 1] * s
    dy = n[:, 0] * s + n[:, 1] * c
    v = valid.to(torch.bool).reshape(b)
    raw[:, 0] = torch.where(v, raw[:, 0] + dx, raw[:, 0])
    raw[:, 1] = torch.where(v, raw[:, 1] + dy, raw[:, 1])
    return raw


def navsim_legal_nav_cmd(driving_command) -> int:
    """NavSim's 4-dim one-hot ``driving_command`` (left, forward, right, unknown) -> our ``nav_cmd`` (the LEGAL row:
    nav_args_known = 0, route_cp_valid = 0). ⛔ Refuses anything that is not exactly one-hot."""
    v = [float(x) for x in driving_command]
    if len(v) != 4 or sorted(v) != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError(f"[refcv8] NavSim driving_command must be a 4-dim one-hot, got {v}")
    return NAVSIM_LEGAL_NAV_CMD[("left", "forward", "right", "unknown")[v.index(1.0)]]


# ---- MM ruling Q1 (SPEC_REFCV8 sec. 3.1, I-2): the extra candidates' EMISSION schedule ----------------------------- #
#: the I-2 literals (accepted by the Master Mind): with emission OFF at step 0, against refcv7 on the same windows
I2_LITERALS = {"sel_idx_frac": 1.0, "max_dtraj_m": 1e-3, "max_dbase_score": 1e-4}


def emit_live(step: int, start: int) -> bool:
    """The extra candidates may be EMITTED at training step ``step`` (0-based, the step about to be taken)."""
    return int(step) >= int(start)


def apply_emit_schedule(model, step: int) -> bool:
    """Set the decoder's emission switch for ``step`` from ``cfg.refcv8.emit_start``; returns it. A no-op build
    without the refcv8 seams returns True (nothing to gate)."""
    dec = getattr(getattr(model, "core", None), "decoder", None)
    if dec is None or getattr(dec, "r8_cfg", None) is None:
        return True
    live = emit_live(step, int(getattr(model.cfg.refcv8, "emit_start", 0) or 0))
    dec.r8_emit_live = live
    return live


def emits_at_step0(r8cfg) -> bool:
    """True when the config would EMIT an extra candidate at step 0 (an emission flag with candidates to emit and
    no ``emit_start``): step-0 identity of the emitted plan then does not hold BY DESIGN (MM ruling Q1)."""
    extras = (int(getattr(r8cfg, "n_alloc", 0) or 0) > 0 and bool(getattr(r8cfg, "alloc_emit", False))) or (
        bool(getattr(r8cfg, "prior_free_group", False)) and bool(getattr(r8cfg, "prior_free_emit", False)))
    return bool(extras) and int(getattr(r8cfg, "emit_start", 0) or 0) <= 0


def checkpoint_r8_step(state_keys, ckpt_step) -> int:
    """The REFCV8 training step a loaded checkpoint stands for: its own step if it carries refcv8 seams, else 0 (a
    refcv7 checkpoint loaded into a refcv8 build is the warm start = refcv8 step 0)."""
    has_r8 = any(is_refcv8_key(k) for k in state_keys)
    return int(ckpt_step or 0) if has_r8 else 0


def i2_identity_row(rows, *, emit_at_step0: bool) -> dict:
    """The I-2 identity row over per-window comparisons (refcv7 vs the refcv8 build at step 0, same windows): each row
    carries ``sel_idx_equal``, ``max_abs_dtraj_m``, ``base_score_max_abs_diff``. ⛔ An argv that EMITS extra candidates
    at step 0 FAILS the row whatever the numbers read (MM ruling Q1): its emitted plan is not refcv7's by design."""
    rows = list(rows)
    lit = I2_LITERALS
    n = len(rows)
    frac = sum(bool(r["sel_idx_equal"]) for r in rows) / max(n, 1)
    dtraj = max((float(r["max_abs_dtraj_m"]) for r in rows), default=float("inf"))
    dscore = max((float(r["base_score_max_abs_diff"]) for r in rows), default=float("inf"))
    reasons = []
    if emit_at_step0:
        reasons.append("the argv emits extra candidates at step 0 (an emission flag without --r8-alloc-emit-start)")
    if n == 0:
        reasons.append("no windows compared")
    if frac < lit["sel_idx_frac"]:
        reasons.append(f"sel_idx identical on {frac:.4f} < {lit['sel_idx_frac']}")
    if dtraj > lit["max_dtraj_m"]:
        reasons.append(f"max |dtraj| {dtraj:.3g} m > {lit['max_dtraj_m']}")
    if dscore > lit["max_dbase_score"]:
        reasons.append(f"max |d base score| {dscore:.3g} > {lit['max_dbase_score']}")
    return {"PASS": not reasons, "reasons": reasons, "n_windows": n, "sel_idx_frac": round(frac, 6),
            "max_abs_dtraj_m": dtraj, "max_base_score_abs_diff": dscore, "literals": dict(lit),
            "emit_at_step0": bool(emit_at_step0)}


def speed_gen(model):
    """X3's OWN dedicated generator (seeded ``cfg.refcv8.seed``), on the MODEL -- used by BOTH the refcv8 path
    (r8_before_forward) and the speed-only path (V0), so two arms on the same batches draw the SAME unknown rows:
    the fed speed channel is byte-identical across them (SPEC_WPB_LADDER sec. 1; MM I-0 assertion 2). Never the
    global stream, never the decoder's r8_gen (whose other draws would desynchronise it)."""
    g = getattr(model, "_r8_speed_gen", None)
    if g is None:
        g = r8c.R8Generator(int(model.cfg.refcv8.seed))
        model._r8_speed_gen = g
    return g


def speed_crc(v_ms: Tensor, valid: Tensor) -> int:
    """CRC32 of the FED speed channel's exact bytes (float32 value, float32 valid): logged per step as
    `r8_spd_crc`, so two runs' channels are compared BYTE FOR BYTE off their own logs."""
    import zlib
    a = v_ms.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    b = valid.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    return int(zlib.crc32(b, zlib.crc32(a)))


def speed_only_fwd(model, batch: Mapping[str, Tensor], device) -> dict:
    """X3 SPEED-ONLY (a model WITHOUT the refcv8 seams; SPEC_WPB_LADDER's V0 feeds N2 too): the same
    `speed_input_treatment` as r8_before_forward, on a MODEL-level dedicated generator (created on first use from
    ``cfg.refcv8.seed``; the global stream is never consumed). Returns the forward overrides."""
    gen = speed_gen(model)
    legal = (not model.training) and bool(getattr(model, "_r8_legal_row", False))
    vm, vv = speed_input_treatment(
        batch["v_max_ms"].to(device), batch["v_max_valid"].to(device), gen, training=bool(model.training),
        unknown_p=float(getattr(model, "_r8_speed_unknown_p", R8_SPEED_UNKNOWN_P)),
        roll=bool(getattr(model, "_r8_roll_speed_train", False)),
        eval_intervention=getattr(model, "_r8_speed_eval", None), legal=legal)
    return {"v_max_ms": vm, "v_max_valid": vv}


def tactical_rows(lat: Tensor, lat_allowed: Tensor, lon: Tensor, lon_allowed: Tensor) -> float:
    """R8-1-REACH's per-batch reading (SPEC_REFCV8 sec. 4.1; MM item 4): the share of windows whose lateral AND
    longitudinal target is supervised -- an exact class (>= 0) or a partial label with at least one allowed class.
    Logged as `r8v9_tac_rows` (refcv7's `tac_label_rows` counts exact lateral rows only and is not this quantity)."""
    sl = (lat.reshape(-1) >= 0) | lat_allowed.reshape(lat.numel(), -1).to(torch.bool).any(-1)
    so = (lon.reshape(-1) >= 0) | lon_allowed.reshape(lon.numel(), -1).to(torch.bool).any(-1)
    return float((sl & so).to(torch.float32).mean()) if sl.numel() else 0.0


def r8_before_forward(model, batch: Mapping[str, Tensor], device, traj_tgt: Tensor, slot_valid: Tensor,
                      pose_last: Tensor, fut_ext: Tensor, fut_valid: Tensor, v0: Tensor) -> dict:
    """Set the model's one-shot TEACHER and return the extra forward inputs + the label tensors the losses need.

    Returns {"fwd": {"r8_nav": ..., "r8_rc": ...}, "gt_lat3", "gt_lon6", "gt_hyp", "cons_t": dict, "lat_t", ...}.
    The GT hypothesis is the TAG of the GT plan (the same tagger the candidates are tagged with), so "the candidate
    belongs to the GT hypothesis" is decided by one rule, not two.

    Inputs come from the v9 join as RAW values (``r8_nav_raw`` / ``r8_nav_known``, ``r8_rc_raw`` / ``r8_rc_valid``)
    and are scaled HERE, after the training treatment, so the noise is applied in metres:
    * TRAINING: nav args dropped with p ``model._r8_nav_args_dropout`` (known -> 0, args -> 0, the token in
      ``nav_cmd`` untouched); RC dropped with p ``model._r8_rc_dropout`` (>= 0.3, refused below), then the registered
      noise on the kept rows (sigmas on ``model._r8_rc_noise``, refused below RC_NOISE_*).
    * EVAL: no dropout, no noise; ``model._r8_legal_row`` True -> the NavSim-LEGAL row (known 0, RC invalid).
    * ``model._r8_no_rc`` True (``--r8-no-rc``): the checkpoint is never fed (the switch-off the MM asked for).
    * X3 (``cfg.speed_input`` "n2" / "n3"): the batch's ``v_max_ms`` / ``v_max_valid`` through
      :func:`speed_input_treatment` (the trained unknown row, the V-VSHUF roll, VMAX-OFF / -SHUF, the LEGAL row)."""
    cfg = model.cfg.refcv8
    gen = _gen(model)
    b = traj_tgt.shape[0]
    dev = traj_tgt.device
    slot_t = [float(h) * 0.1 for h in model.cfg.core.trajectory.horizons]
    gl, go = r8c.tag_paths(traj_tgt[:, None], v0, slot_t)
    gl, go = gl[:, 0], go[:, 0]
    ok = slot_valid[:, -1].to(torch.bool)
    gl = torch.where(ok, gl, torch.full_like(gl, -1))
    go = torch.where(ok, go, torch.full_like(go, -1))
    hyp = torch.where(ok, r8c.joint_id(gl.clamp_min(0), go.clamp_min(0)), torch.full_like(gl, -1))
    ct = r8c.constraint_targets(traj_tgt, slot_valid, pose_last, fut_ext, fut_valid)
    lat_t, lat_v, lon_t, lon_v = r8c.normalise_constraints(ct)
    cons = torch.cat([torch.nan_to_num(lat_t), torch.nan_to_num(lon_t)], -1)
    tf = gen.rand((b,), dev) < float(getattr(model, "_r8_tf_ratio", cfg.tf_start))
    # the deliberate-regression arm's derange is a TRAINING-only diagnostic: eval feeds every model its own output
    model._r8_derange_feed = bool(model.training and getattr(model, "_r8_derange_train", False))
    if model.training:
        model.set_r8_teacher(hyp, cons, tf)
    fwd: dict[str, Any] = {}
    legal = (not model.training) and bool(getattr(model, "_r8_legal_row", False))
    if "r8_nav_raw" in batch:
        nav = batch["r8_nav_raw"].to(device).to(torch.float32)
        known = batch["r8_nav_known"].to(device).to(torch.bool).reshape(b).clone()
        if model.training and float(getattr(model, "_r8_nav_args_dropout", 0.0)) > 0.0:
            known &= ~(gen.rand((b,), dev) < float(model._r8_nav_args_dropout))   # the TOKEN is untouched
        if legal:
            known = torch.zeros_like(known)
        fwd["r8_nav"] = scale_nav(nav, known)
    if "r8_rc_raw" in batch and not bool(getattr(model, "_r8_no_rc", False)):
        rc = batch["r8_rc_raw"].to(device).to(torch.float32)
        valid = batch["r8_rc_valid"].to(device).to(torch.bool).reshape(b).clone()
        if model.training:
            p = float(getattr(model, "_r8_rc_dropout", RC_DROPOUT_MIN))
            if p < RC_DROPOUT_MIN:
                raise SystemExit(f"[refcv8] route-checkpoint dropout {p} < {RC_DROPOUT_MIN} (MM binding 2026-10-04)")
            sa, sl = getattr(model, "_r8_rc_noise", (RC_NOISE_ALONG_M, RC_NOISE_LAT_M))
            if float(sa) < RC_NOISE_ALONG_M or float(sl) < RC_NOISE_LAT_M:
                raise SystemExit(f"[refcv8] route-checkpoint training noise ({sa}, {sl}) m is below the CERTIFIED "
                                 f"({RC_NOISE_ALONG_M}, {RC_NOISE_LAT_M}) m -- the clean point leaks (WP-A E2')")
            valid &= ~(gen.rand((b,), dev) < p)
            rc = rc_training_noise(rc, valid, gen, sigma_along_m=float(sa), sigma_lat_m=float(sl))
            if bool(getattr(model, "_r8_rc_roll_train", False)) and b > 1:
                rc, valid = torch.roll(rc, 1, 0), torch.roll(valid, 1, 0)       # another window's checkpoint
        elif getattr(model, "_r8_rc_eval", None) == "shuf":       # the RC-SHUFFLED eval row (SPEC_WPB_LADDER sec. 2)
            if b < 2:
                raise ValueError("[refcv8] RC-shuffled on a batch of one: a roll is the identity")
            rc, valid = torch.roll(rc, 1, 0), torch.roll(valid, 1, 0)
        if legal:
            valid = torch.zeros_like(valid)
        fwd["r8_rc"] = scale_rc(rc, valid)
    # ---- X3: the past-only speed input (the SOURCE of the 4-way set-speed channel). Drawn AFTER the RC draws, so a
    # run without it consumes the dedicated stream exactly as before. The treated value REPLACES the batch's in the
    # forward (compute_losses_v3 merges `fwd` over its own kwargs). ----
    if str(getattr(cfg, "speed_input", "") or "") and "v_max_ms" in batch:
        fwd["v_max_ms"], fwd["v_max_valid"] = speed_input_treatment(
            batch["v_max_ms"].to(device), batch["v_max_valid"].to(device), speed_gen(model),
            training=bool(model.training),
            unknown_p=float(getattr(model, "_r8_speed_unknown_p", R8_SPEED_UNKNOWN_P)),
            roll=bool(getattr(model, "_r8_roll_speed_train", False)),
            eval_intervention=getattr(model, "_r8_speed_eval", None), legal=legal)
    # ---- MM ruling Q2: the v9 constraint targets on the batch's OWN v9 classes (rolled together under the tac roll) ---
    v9c = None
    if "r8_v9_lat_c" in batch and "lat_v7" in batch and "lon_v7" in batch:
        v9c = r8c.v9_constraint_targets(batch["r8_v9_lat_c"].to(device), batch["r8_v9_lon_c"].to(device),
                                        batch["r8_v9_speed"].to(device), batch["lat_v7"].to(device),
                                        batch["lon_v7"].to(device))
    # ---- MM item 4 (Q8): R8-1-REACH's in-run key -- the share of the batch's windows whose lateral AND longitudinal
    # tactical target is supervised (an exact class or a non-empty partial mask), as the workers produced them ----
    # ---- the drivable critic's LABEL source: the batch's 10 cm SAM3 target (rolled with the map family) ----
    drv_map = None
    if bool(getattr(cfg, "critic_drivable", False)) and "map_fine" in batch:
        drv_map = (batch["map_fine"].to(device), batch["map_fine_label"].to(device) if "map_fine_label" in batch
                   else None)
    tac_rows = None
    if all(k in batch for k in ("lat_v7", "lon_v7", "lat_allowed_v7", "lon_allowed_v7")):
        tac_rows = tactical_rows(batch["lat_v7"], batch["lat_allowed_v7"], batch["lon_v7"], batch["lon_allowed_v7"])
    gl_v7 = torch.where(gl >= 0, torch.as_tensor(r8c.LAT3_V7_IDS, device=dev)[gl.clamp_min(0)],
                        torch.full_like(gl, -100))
    go_v7 = torch.where(go >= 0, torch.as_tensor(r8c.LON6_V7_IDS, device=dev)[go.clamp_min(0)],
                        torch.full_like(go, -100))
    return {"fwd": fwd, "gt_lat3": gl, "gt_lon6": go, "gt_hyp": hyp, "gt_lat_v7": gl_v7, "gt_lon_v7": go_v7,
            "cons_t": ct, "lat_t": lat_t, "lat_v": lat_v, "lon_t": lon_t, "lon_v": lon_v, "tf": tf, "v9c": v9c,
            "tac_rows": tac_rows, "drv_map": drv_map}


def r8_losses(model, out: Mapping[str, Tensor], prep: Mapping[str, Any], traj_tgt: Tensor,
              slot_valid: Tensor) -> tuple[Tensor, dict]:
    """The refcv8 loss terms (weights from `cfg.refcv8`). Returns (weighted sum, telemetry).

    ⛔ No term is guarded behind `if w > 0` for a BUILT head: a zero weight multiplies (the 42-of-138 lesson), except
    the listwise / sub-score terms, which are computed only when their head / weight exists (no head = no tensor)."""
    cfg = model.cfg.refcv8
    dev = traj_tgt.device
    total = traj_tgt.new_zeros(())
    tele: dict[str, Any] = {}
    # (1) constraint heads -- the GT-active query only
    if prep.get("tac_rows") is not None:
        tele["r8v9_tac_rows"] = float(prep["tac_rows"])
    # (1b) MM ruling Q2: the v9 constraint vectors on the GT-active query of the v9 class (masked: PARTIAL / absent /
    # undefined). A BUILT head with no target in a TRAINING batch is refused (the dead-seam class), never skipped.
    if "r8_v9_lat_c" in out:
        if prep.get("v9c") is None:
            if model.training:
                raise SystemExit("[refcv8] the v9 constraint heads are built (--w-r8-v9-cons > 0) but the batch carries "
                                 "no r8_v9_* targets: the dataset was not enabled with r8_v9_cons")
        else:
            lv, vt = r8c.v9_constraint_loss(out["r8_v9_lat_c"], out["r8_v9_lon_c"], out["r8_v9_speed"], prep["v9c"])
            total = total + float(cfg.w_v9_cons) * lv
            tele.update(r8_v9_cons=lv.detach(), **vt)
    # (1c) the drivable critic: BCE against the SAM3 10 cm label of each EMITTED-fan candidate's footprint
    if "r8_drv_logit" in out:
        if prep.get("drv_map") is None:
            if model.training:
                raise SystemExit("[refcv8] the drivable critic is built but the batch carries no 10 cm map target "
                                 "(map_fine): its BCE would train nothing")
        else:
            _codes, _has = prep["drv_map"]
            _y, _w = r8c.drivable_target(out["anchor_traj"].detach(), _codes, _has)
            ld, dtel = r8c.drivable_bce(out["r8_drv_logit"], _y, _w)
            total = total + float(cfg.w_drivable) * ld
            tele.update(r8_drv=ld.detach(), **dtel)
    if "r8_cons_lat" in out:
        lc, ltel = r8c.constraint_head_loss(out["r8_cons_lat"], out["r8_cons_lon"], prep["gt_lat_v7"],
                                            prep["gt_lon_v7"], prep["lat_t"], prep["lat_v"], prep["lon_t"],
                                            prep["lon_v"])
        total = total + float(cfg.w_cons) * lc
        tele.update(r8_cons=lc.detach(), **ltel)
    if "r8_alloc" not in out:
        return total, tele
    fan = out["anchor_traj"]
    alloc = out["r8_alloc"].to(torch.bool)
    sv = slot_valid.to(fan.dtype)
    err = ((fan - traj_tgt[:, None]).norm(dim=-1) * sv[:, None]).sum(-1) / sv.sum(-1, keepdim=True).clamp_min(1.0)
    # (2) matched L1 on the best allocated candidate OF THE GT HYPOTHESIS (WTA, MTR/TNT-style hard assignment)
    if bool(alloc.any()) and "r8_hyp_alloc" in out:
        nb = int(out["r8_n_base"])
        m = out["r8_hyp_alloc"].shape[1]
        gt_h = prep["gt_hyp"]
        is_gt = (out["r8_hyp_alloc"] == gt_h[:, None]) & (gt_h[:, None] >= 0)        # [B, M]
        e_a = err[:, nb:nb + m].masked_fill(~is_gt, float("inf"))
        has = torch.isfinite(e_a).any(-1)
        if bool(has.any()):
            j = e_a.argmin(-1)
            ar = torch.arange(fan.shape[0], device=dev)
            rec = fan[ar, nb + j]
            l1 = (((rec - traj_tgt).abs().sum(-1)) * sv).sum(-1) / (sv.sum(-1) * 2).clamp_min(1.0)
            la = l1[has].mean()
        else:
            la = fan.sum() * 0.0
        total = total + float(cfg.w_alloc_l1) * la
        tele.update(r8_alloc_l1=la.detach(), r8_alloc_gt_rows=int(has.sum().item()))
        # (3) L_sat on the allocated candidates, against the constraint each was GENERATED under (phi's fields)
        phi = out["r8_phi"]
        c_gen = r8c.denorm_cons(phi[..., 11:15])
        cv = phi[..., 15]
        ls = r8c.constraint_satisfaction_loss(fan, c_gen, cv, alloc)
        total = total + float(cfg.w_sat) * ls
        tele["r8_sat"] = ls.detach()
    # (4) X1 listwise selection over the reach survivors (incl. the allocated candidates)
    if float(cfg.w_listwise) > 0.0:
        keep = out.get("reach_keep")
        keep = torch.ones_like(err, dtype=torch.bool) if keep is None else keep.to(torch.bool)
        ll = r8c.listwise_selection_loss(out["sel_score_v3"], fan.detach(), traj_tgt, slot_valid, keep,
                                         t=float(cfg.list_t), speed_scale_m=float(cfg.list_speed_scale_m),
                                         dir_scale_deg=float(cfg.list_dir_scale_deg))
        total = total + float(cfg.w_listwise) * ll
        tele["r8_listwise"] = ll.detach()
    # (5) X1h sub-score critics
    if "r8_sub_logits" in out:
        y, w = r8c.subscore_targets(fan.detach(), traj_tgt, slot_valid, out["r8_lat3"], prep["gt_lat3"])
        bce = F.binary_cross_entropy_with_logits(out["r8_sub_logits"].float(), y, reduction="none")
        lsb = (bce * w).sum() / w.sum().clamp_min(1.0)
        total = total + float(cfg.w_subscore) * lsb
        tele["r8_subscore"] = lsb.detach()
    tele["r8_emit_frac"] = out["r8_emit_keep"].float().mean().detach()
    return total, tele


# ================================================================================================================= #
# v9: the partial-label loss                                                                                        #
# ================================================================================================================= #
#: the batch keys of each target family a regression arm rolls (SPEC_WPB_LADDER sec. 2)
#: "tac" includes the v9 CONSTRAINT targets (MM ruling Q2): rolling the classes without them would leave the constraint
#: heads supervised by this window's geometry under another window's class -- half a regression arm
ROLL_FAMILIES = {"tac": ("lat_v7", "lon_v7", "lat_allowed_v7", "lon_allowed_v7", "tac_goal_y", "tac_goal_w",
                         "r8_v9_lat_c", "r8_v9_lon_c", "r8_v9_speed"),
                 "map": ("map_fine", "map_fine_label")}


def roll_targets(batch: Mapping[str, Any], family: str) -> dict:
    """A SHALLOW copy of ``batch`` with the ``family``'s target tensors rolled by one row: every row is supervised by
    ANOTHER window's targets -- same head, same gradient magnitude, zero information (the ``--bev-aux-shuffle``
    precedent; no RNG is consumed, so the arm stays seed-comparable). Inputs are never touched. A family whose keys
    are all absent from the batch is REFUSED (the regression arm would silently be the treatment arm)."""
    keys = ROLL_FAMILIES[family]
    present = [k for k in keys if k in batch and torch.is_tensor(batch[k])]
    if not present:
        raise SystemExit(f"[refcv8] --r8-roll-targets {family}: none of {keys} is in the batch -- nothing to roll")
    out = dict(batch)
    for k in present:
        if out[k].shape[0] < 2:
            raise SystemExit(f"[refcv8] --r8-roll-targets {family}: a batch of one cannot be rolled")
        out[k] = torch.roll(out[k], 1, 0)
    return out


def partial_label_loss(logits: Tensor, allowed: Tensor) -> tuple[Tensor, int]:
    """``mean_rows -log sum_{c in allowed} softmax(z)_c`` over the rows with at least one allowed class (v9
    INTEGRATION sec. 3.1). An exact label is a ONE-bit mask, so on those rows this IS cross-entropy; an all-false
    row contributes nothing. Returns (loss, n rows). ⛔ Never guarded: no row -> an attached zero."""
    z = logits.float()
    al = allowed.to(torch.bool)
    has = al.any(-1)
    lse_all = torch.logsumexp(z, -1)
    lse_al = torch.logsumexp(z.masked_fill(~al, float("-inf")), -1)
    per = torch.where(has, lse_all - lse_al, torch.zeros_like(lse_all))
    n = int(has.sum().item())
    return per.sum() / float(max(n, 1)), n


def _exact_ce(logits: Tensor, target: Tensor, ignore_index: int) -> Tensor:
    """EXACTLY `refcv6_tactical.tactical_behaviour_losses._ce` with no class weight (the trainer passes none)."""
    t = target.reshape(-1).long()
    per = F.cross_entropy(logits, t.clamp_min(0), reduction="none")
    keep = (t != ignore_index).to(per.dtype)
    return (per * keep).sum() / keep.sum().clamp_min(1.0)


def v9_partial_correction(lat_logits: Tensor, lon_logits: Tensor, lat_t: Tensor, lon_t: Tensor, lat_allowed: Tensor,
                          lon_allowed: Tensor, weights, *, ignore_index: int = -100) -> tuple[Tensor, dict]:
    """The term that turns the tactical loss's exact-row CE into the v9 COMBINED partial-label mean, inside the SAME
    weights: ``w_lat * (PL_all - CE_exact) + w_lon * (...)``. Added to ``tactical_behaviour_losses``' total (which
    holds ``w_lat * CE_exact``), the lat term becomes ``w_lat * PL_all`` -- the mean of ``-log sum_allowed p`` over
    every row with an allowed class (an exact row's mask is its one bit, so those rows are CE). Nothing is inflated:
    the budget split of ``TacticalLossWeights`` is unchanged."""
    pl_lat, n_lat = partial_label_loss(lat_logits, lat_allowed)
    pl_lon, n_lon = partial_label_loss(lon_logits, lon_allowed)
    ce_lat = _exact_ce(lat_logits, lat_t, ignore_index)
    ce_lon = _exact_ce(lon_logits, lon_t, ignore_index)
    corr = float(weights.lat_ce) * (pl_lat - ce_lat) + float(weights.lon_ce) * (pl_lon - ce_lon)
    n_ex_lat = int((lat_t.reshape(-1) != ignore_index).sum().item())
    n_ex_lon = int((lon_t.reshape(-1) != ignore_index).sum().item())
    return corr, {"r8v9_lat_pl": pl_lat.detach(), "r8v9_lon_pl": pl_lon.detach(),
                  "r8v9_n_lat_partial": float(n_lat - n_ex_lat), "r8v9_n_lon_partial": float(n_lon - n_ex_lon),
                  "r8v9_n_lat_supervised": float(n_lat), "r8v9_n_lon_supervised": float(n_lon)}


# ================================================================================================================= #
# v9: the release contract, the join, the goal census                                                              #
# ================================================================================================================= #
def _v9():
    try:
        from tanitad.data import v9_labels as V9
    except ImportError as e:            # pragma: no cover - exercised only on a tree without WP-A's reader
        raise SystemExit("[refcv8] --r8-v9-labels needs WP-A's reader `tanitad/data/v9_labels.py`, which is not in "
                         f"this tree ({e})") from None
    return V9


def _bits(a, i: int):
    return ((np.asarray(a).astype(np.int64) >> int(i)) & 1).astype(bool)


def v9_integrity_census(rel, lat_variant: str = "a") -> dict:
    """The release properties this trainer RELIES ON (MM item 4 / INTEGRATION sec. 8), counted over every row.
    Every ``violations`` entry must be 0 -- :func:`load_v9_join` refuses otherwise. Derived from the raw fields, not
    from the reader's decoding, so it cross-checks the builder rather than restating it."""
    V9 = _v9()
    R = rel.rows
    lat = np.asarray(R[f"lat_v7id_{lat_variant}"]).astype(np.int64)
    lat_al = np.asarray(R[f"lat_allowed_v7_{lat_variant}"]).astype(np.int64)
    lon = np.asarray(R["lon_v7id"]).astype(np.int64)
    lon_al = np.asarray(R["lon_allowed_v7"]).astype(np.int64)
    gy = np.asarray(R["goal_y"]).astype(np.int64)
    gw = np.asarray(R["goal_w"]).astype(np.int64)
    rev = np.asarray(R["reversing"]).astype(np.int64) == 1
    short = np.asarray(R["h_obs_s"]).astype(np.float64) < H_ABS_MIN_S - 1e-6
    gi = {t: V9.GOAL22.index(t) for t in V9.GOAL22}
    geo_w = np.zeros(len(gw), bool)
    for t in GEOMETRY_GOALS:
        geo_w |= _bits(gw, gi[t])

    def neg(t):
        return _bits(gw, gi[t]) & ~_bits(gy, gi[t])

    def pos(t):
        return _bits(gw, gi[t]) & _bits(gy, gi[t])
    ex_lat, ex_lon = lat >= 0, lon >= 0
    one = np.left_shift(1, np.clip(lat, 0, 30))
    one_lon = np.left_shift(1, np.clip(lon, 0, 30))
    viol = {
        "lc_exact_action": int(np.isin(lat, LC_LAT_V7).sum()),
        "reversing_with_action_label": int((rev & ((lat_al > 0) | (lon_al > 0))).sum()),
        "reversing_with_geometry_goal": int((rev & geo_w).sum()),
        "absence_class_on_short_band": int((short & (np.isin(lat, ABSENCE_LAT_V7) | np.isin(lon, ABSENCE_LON_V7))
                                            ).sum()),
        # a TURN_x negative ENTAILED by the other side's positive is a presence claim, not an absence claim
        # (MEASURED on the release: every short-band TURN_L negative sits on a TURN_R positive, 11 / 11 eval139)
        "absence_goal_on_short_band": int((short & (pos("FOLLOW_LANE") | (neg("TURN_L") & ~pos("TURN_R"))
                                                    | (neg("TURN_R") & ~pos("TURN_L")) | neg("STOP_POINT"))).sum()),
        "exact_label_mask_mismatch": int((ex_lat & (lat_al != one)).sum() + (ex_lon & (lon_al != one_lon)).sum()),
        "speed_band_supervised": int(_bits(gw, gi["SPEED_BAND"]).sum()),
    }
    return {"n_rows": int(len(lat)), "violations": viol,
            "n_reversing": int(rev.sum()), "n_short_band": int(short.sum()),
            "lat_partial": int(((lat < 0) & (lat_al > 0)).sum()), "lon_partial": int(((lon < 0) & (lon_al > 0)).sum()),
            "lc_goal_positive_vlm": int((pos("LANE_CHANGE_L") | pos("LANE_CHANGE_R")).sum())}


class R8LabelJoin:
    """The v9 release joined to the trainer's windows. Key ``(sid, k)`` with ``k = t + w - 1 + raw_offset`` (= t + 9
    at w 8, n_stack 3) and the trainer's ``now_s`` CHECKED against the release clock (``row_for_now`` refuses a
    mismatch > 1e-6 s). The release is an immutable, picklable ``V9Release`` held on the INSTANCE; no module state.

    ``item`` returns RAW inputs (scaled in :func:`r8_before_forward`, after the training treatment) and the targets in
    the frozen v7 ids (D-WPA-2). ⛔ ``nav_t_next_s`` / ``nav_token_ttime`` / ``rcH_*`` are never read (the reader's
    ``window_inputs`` does not expose them)."""

    def __init__(self, release, *, rc_variant: str = "A50", lat_variant: str = "a", census: dict | None = None):
        V9 = _v9()
        if rc_variant not in V9.RC_VARIANTS:
            raise SystemExit(f"[refcv8] --r8-rc-variant {rc_variant!r} not in {V9.RC_VARIANTS}")
        if lat_variant not in ("a", "b"):
            raise SystemExit(f"[refcv8] --r8-v9-lat-variant {lat_variant!r} must be 'a' or 'b'")
        self.rel = release
        self.rc_variant = str(rc_variant)
        self.lat_variant = str(lat_variant)
        man = release.manifest()
        self.manifest = {"md5": release.md5, "split": release.split, "path": release.path,
                         "schema": man.get("schema"), "base_commit": man.get("base_commit"),
                         "builder_md5": man.get("builder_md5"), "rc_variant": self.rc_variant,
                         "lat_variant": self.lat_variant, "census": census}

    def row(self, sid: int, k: int, now_s: float) -> int:
        return _v9().row_for_now(self.rel, int(sid), int(k), float(now_s))

    def item(self, sid: int, k: int, now_s: float, speed_mode: str | None = None,
             cons: bool = False) -> dict[str, Tensor]:
        """... and with ``speed_mode`` ("n2" / "n3", X3) ``r8_speed_ms`` (the FED value in m/s, computed in Python
        float64 as km/h / 3.6 and only then cast -- so it lands on the 4-way ladder's step EXACTLY, the
        GPU-scalar-divisor trap) + ``r8_speed_known``; and with ``cons`` (MM ruling Q2) the RAW v9 constraint
        vectors ``r8_v9_lat_c`` [12] / ``r8_v9_lon_c`` [10] / ``r8_v9_speed`` [4] (NaN = undefined; masked in the loss)."""
        V9 = _v9()
        r = V9.row_for_now(self.rel, int(sid), int(k), float(now_s))
        inp = V9.window_inputs(self.rel, r, rc_variant=self.rc_variant)
        tg = V9.window_targets(self.rel, r, lat_variant=self.lat_variant, ids="v7")
        out = {}
        if speed_mode:
            kmh, known = speed_kmh_of(self.rel, r, str(speed_mode))
            out["r8_speed_ms"] = torch.tensor(float(kmh) / 3.6 if known else 0.0, dtype=torch.float32)
            out["r8_speed_known"] = torch.tensor(bool(known))
        if cons:
            out["r8_v9_lat_c"] = torch.as_tensor(np.asarray(tg["lat_constraints"], np.float32))
            out["r8_v9_lon_c"] = torch.as_tensor(np.asarray(tg["lon_constraints"], np.float32))
            out["r8_v9_speed"] = torch.as_tensor(np.asarray(tg["speed_goal"], np.float32))
        return {**out, "r8_nav_token": torch.tensor(int(inp["nav_token"]), dtype=torch.long),
                "r8_nav_raw": torch.as_tensor(np.asarray(inp["nav_args"], np.float32)),
                "r8_nav_known": torch.tensor(bool(inp["nav_args_valid"])),
                "r8_rc_raw": torch.as_tensor(np.asarray(inp["rc"], np.float32)),
                "r8_rc_valid": torch.tensor(bool(inp["rc_valid"])),
                "v9_lat": torch.tensor(int(tg["lat"]), dtype=torch.long),
                "v9_lat_allowed": torch.as_tensor(np.asarray(tg["lat_allowed"], bool)),
                "v9_lon": torch.tensor(int(tg["lon"]), dtype=torch.long),
                "v9_lon_allowed": torch.as_tensor(np.asarray(tg["lon_allowed"], bool)),
                "v9_goal_y": torch.as_tensor(np.asarray(tg["goal_y"], np.float32)),
                "v9_goal_w": torch.as_tensor(np.asarray(tg["goal_w"], np.float32))}


def is_refcv8_key(key: str) -> bool:
    """A state-dict key that belongs to a refcv8 seam: any dotted component starting with ``r8_`` (the decoder's
    ``r8_mod`` / ``r8_sel`` / ``r8_sub`` / ``r8_rc_*``, the behaviour decoder's ``r8_cons_*`` / ``r8_film``)."""
    return any(part.startswith("r8_") for part in key.split("."))


def warm_start_from(model, path: str) -> dict:
    """``--init-from``: load a refcv7 (or refcv8) checkpoint into a refcv8 build -- the WARM START of DESIGN sec. 3.7.

    STRICT where it can be: every key of the SOURCE must exist in the build with the same shape (an unexpected key or a
    shape mismatch REFUSES), and every key the source LACKS must be a declared refcv8 seam key (``is_refcv8_key``) --
    any other missing key means the build is not refcv7 + seams, and is REFUSED. The seams are zero-init / gated, so
    step 0 reproduces the source (pinned by tests/test_refcv8_warm_start.py). The optimiser is NOT loaded: fresh
    AdamW moments, the run's own warmup (the DESIGN sec. 3.7 decision, stated). Returns the config.json record."""
    import hashlib
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[refcv8] --init-from {path}: no such file")
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    state = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else None
    if not isinstance(sd, dict) or not sd:
        raise SystemExit(f"[refcv8] --init-from {path}: no model state_dict in the file")
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:                        # a SHAPE mismatch: the build is not the source's architecture
        raise SystemExit(f"[refcv8] --init-from {path}: {str(e).splitlines()[0][:400]}") from None
    if unexpected:
        raise SystemExit(f"[refcv8] --init-from {path}: {len(unexpected)} source key(s) the build does not have, "
                         f"e.g. {sorted(unexpected)[:3]} -- the build is not a superset of the source")
    undeclared = [k for k in missing if not is_refcv8_key(k)]
    if undeclared:
        raise SystemExit(f"[refcv8] --init-from {path}: {len(undeclared)} key(s) missing from the source are NOT "
                         f"refcv8 seams, e.g. {sorted(undeclared)[:3]} -- they would start from random init")
    return {"path": str(p), "md5": h.hexdigest(), "source_step": (int(state["step"]) if isinstance(state, dict)
                                                                  and "step" in state else None),
            "n_loaded": len(sd), "n_new_refcv8_keys": len(missing), "optimizer": "fresh (not loaded)"}


def load_v9_join(npz_path: str, *, expect_md5: str, rc_variant: str = "A50", lat_variant: str = "a") -> R8LabelJoin:
    """Load + md5-check + contract census; REFUSES a release that violates any property the trainer relies on."""
    V9 = _v9()
    try:
        rel = V9.load_v9_release(str(npz_path), expect_md5=str(expect_md5))
    except V9.V9LabelError as e:
        raise SystemExit(str(e)) from None
    census = v9_integrity_census(rel, lat_variant)
    bad = {k: v for k, v in census["violations"].items() if v}
    if bad:
        raise SystemExit(f"[refcv8] ⛔ the v9 release {rel.path} violates the contract the trainer relies on: "
                         f"{bad} (v9 INTEGRATION sec. 8 / MM item 4)")
    return R8LabelJoin(rel, rc_variant=rc_variant, lat_variant=lat_variant, census=census)


def v9_goal_census(join: R8LabelJoin, rows) -> dict:
    """The 22-token goal census over the release ROWS the dataset's windows join to -- the v9 analogue of
    ``v7_labels.goal_supervision_census`` (which counts per CLIP record). Same keys, so ``tac_goal_head.mask_report``
    and the pos_weight rule read it unchanged."""
    V9 = _v9()
    rows = np.asarray(rows, dtype=np.int64)
    gy = np.asarray(join.rel.rows["goal_y"]).astype(np.int64)[rows]
    gw = np.asarray(join.rel.rows["goal_w"]).astype(np.int64)[rows]
    out = {}
    for i, tok in enumerate(V9.GOAL22):
        w = _bits(gw, i)
        y = _bits(gy, i)
        p, n = int((w & y).sum()), int((w & ~y).sum())
        out[tok] = {"pos": p, "neg": n, "ignored": int((~w).sum()), "prevalence": p / max(len(rows), 1),
                    "provenance": ["v9"], "supervised_negative": n > 0, "negatives_policy": "v9-release",
                    "entailed_false_by": []}
    return out


def v9_goal_pos_weight(census: dict, tokens, cap: float) -> list:
    """``n_neg / n_pos`` capped at ``cap``; 0.0 for a class with no positive (it must be masked too)."""
    out = []
    for t in tokens:
        p, n = int(census[t]["pos"]), int(census[t]["neg"])
        out.append(0.0 if p == 0 else float(min(n / p, cap)))
    return out


# ================================================================================================================= #
# label-state isolation                                                                                             #
# ================================================================================================================= #
def v7_policy_travels(v7l_module) -> bool:
    """True when ``v7_labels`` carries the goal-negative policy ON the label objects (WP-A's fix, base 86f0c46e ->
    abb1f64c): then the trainer needs no scope, and must not install one."""
    fields = getattr(getattr(v7l_module, "V7Label", None), "__dataclass_fields__", {}) or {}
    mfields = getattr(getattr(v7l_module, "LabelManifest", None), "__dataclass_fields__", {}) or {}
    return "goal_geometry_tokens" in fields and "goal_geometry_tokens" in mfields


def v7_scope_for(v7l_module):
    """What the trainer installs on a dataset right after a label load: ``None`` when the policy travels with the
    labels (the fixed module), a :class:`V7PolicyScope` snapshot otherwise (the unfixed tip module)."""
    return None if v7_policy_travels(v7l_module) else V7PolicyScope(v7l_module)


class V7PolicyScope:
    """The v7 negative policy of ONE loaded split, snapshotted at construction and applied around each target
    computation -- the FALLBACK for an unfixed ``v7_labels`` whose policy is module state (whatever was loaded LAST:
    the eval blob, in the trainer's order). The snapshot travels with the pickled dataset into a worker."""

    _FIELDS = ("_MEASURED_GEOMETRY_TOKENS", "_MEASURED_COT_TOKENS")

    def __init__(self, v7l_module):
        self.module_name = v7l_module.__name__
        self.snapshot = {f: frozenset(getattr(v7l_module, f)) for f in self._FIELDS}

    @contextmanager
    def applied(self):
        import importlib
        mod = importlib.import_module(self.module_name)
        prev = {f: getattr(mod, f) for f in self._FIELDS}
        try:
            for f, v in self.snapshot.items():
                setattr(mod, f, v)
            yield
        finally:
            for f, v in prev.items():
                setattr(mod, f, v)

    def to_dict(self) -> dict:
        return {f: sorted(v) for f, v in self.snapshot.items()}


# ================================================================================================================= #
# G-DVB: every refcv8 trainer flag, declared against what the BUILT model holds (the SPEC_REFCV7 sec. 2 rule)       #
# ================================================================================================================= #
def _register_gdvb() -> None:
    from tanitad.train import declared_vs_built as dvb

    def _r8(m):
        return getattr(dvb._dec(m), "r8_cfg", None)

    def _c_enable(m, a):
        want = bool(dvb._a(a, "refcv8", False))
        out = dvb._eq("refcv8", want, bool(getattr(m, "r8_enabled", False)), "model.r8_enabled")
        out += dvb._eq("refcv8", want, _r8(m) is not None, "core.decoder.r8_cfg is not None")
        if want:
            td = getattr(m, "tac_decoder_v6", None)
            out += dvb._eq("refcv8", True, getattr(td, "r8_cons_lat", None) is not None,
                           "model.tac_decoder_v6.r8_cons_lat is not None")
        return out

    def _field(dest, attr, conv, default):
        def chk(m, a):
            on = bool(dvb._a(a, "refcv8", False))
            want = conv(dvb._a(a, dest, default))
            if not on:
                return dvb._eq(dest, conv(default), want, "argv (refcv8 off: must hold the default)")
            c = _r8(m)
            got = None if c is None else getattr(c, attr, None)
            if isinstance(want, float):
                return dvb._near(dest, want, got, f"core.decoder.r8_cfg.{attr}")
            return dvb._eq(dest, want, got, f"core.decoder.r8_cfg.{attr}")
        return chk

    for dest, attr, conv, dflt, kind in (
            ("r8_n_alloc", "n_alloc", int, 0, "built"),
            ("r8_alloc_top_k", "alloc_top_k", int, 4, "built"),
            ("r8_alloc_emit", "alloc_emit", bool, False, "built"),
            ("r8_prior_free_group", "prior_free_group", bool, False, "built"),
            ("r8_prior_free_emit", "prior_free_emit", bool, False, "built"),
            ("r8_lat_prior_dropout", "lat_prior_dropout", float, 0.0, "built"),
            ("r8_cond_dropout", "cond_dropout", float, 0.15, "built"),
            ("r8_tf_start", "tf_start", float, 1.0, "built"),
            ("r8_tf_end", "tf_end", float, 0.25, "built"),
            ("r8_base_constraints", "base_constraints", bool, False, "built"),
            ("r8_seed", "seed", int, 20261004, "built"),
            ("w_r8_cons", "w_cons", float, 0.0, "loss"),
            ("w_r8_sat", "w_sat", float, 0.0, "loss"),
            ("w_r8_alloc_l1", "w_alloc_l1", float, 0.0, "loss"),
            ("w_r8_listwise", "w_listwise", float, 0.0, "loss")):
        dvb.register(dest, kind, _field(dest, attr, conv, dflt))

    def _c_no_mod(m, a):
        on = bool(dvb._a(a, "refcv8", False))
        want = bool(dvb._a(a, "r8_no_modulate_base", False))
        if not on:
            return dvb._eq("r8_no_modulate_base", False, want, "argv (refcv8 off)")
        c = _r8(m)
        return dvb._eq("r8_no_modulate_base", want, None if c is None else (not bool(c.modulate_base)),
                       "not core.decoder.r8_cfg.modulate_base")

    _sub_field = _field("w_r8_subscore", "w_subscore", float, 0.0)

    def _c_sub(m, a):
        want = bool(dvb._a(a, "refcv8", False)) and float(dvb._a(a, "w_r8_subscore", 0.0) or 0.0) > 0.0
        return _sub_field(m, a) + dvb._eq("w_r8_subscore", want,
                                          getattr(dvb._dec(m), "r8_sub", None) is not None,
                                          "core.decoder.r8_sub is not None")

    dvb.register("refcv8", "built", _c_enable)
    dvb.register("r8_no_modulate_base", "built", _c_no_mod)
    dvb.register("w_r8_subscore", "loss", _c_sub)
    _v9c_field = _field("w_r8_v9_cons", "w_v9_cons", float, 0.0)

    def _c_v9c(m, a):
        """MM ruling Q2: the v9 constraint heads exist IFF --w-r8-v9-cons > 0 (and the flag agrees with the config)."""
        want = bool(dvb._a(a, "refcv8", False)) and float(dvb._a(a, "w_r8_v9_cons", 0.0) or 0.0) > 0.0
        td = getattr(m, "tac_decoder_v6", None)
        return _v9c_field(m, a) + dvb._eq("w_r8_v9_cons", want, getattr(td, "r8_v9_lat_c", None) is not None,
                                          "model.tac_decoder_v6.r8_v9_lat_c is not None")

    dvb.register("w_r8_v9_cons", "loss", _c_v9c)
    _e8_field = _field("r8_speed_enc8", "speed_enc8", bool, False)

    def _c_e8(m, a):
        """refcv8 (B): the 8-step speed FiLM exists IFF --r8-speed-enc8 (and the flag agrees with the config)."""
        want = bool(dvb._a(a, "refcv8", False)) and bool(dvb._a(a, "r8_speed_enc8", False))
        td = getattr(m, "tac_decoder_v6", None)
        return _e8_field(m, a) + dvb._eq("r8_speed_enc8", want, getattr(td, "r8_speed_film", None) is not None,
                                         "model.tac_decoder_v6.r8_speed_film is not None")

    dvb.register("r8_speed_enc8", "built", _c_e8)
    _drv_field = _field("r8_critic_drivable", "critic_drivable", bool, False)

    def _c_drv(m, a):
        """the drivable critic exists IFF --r8-critic-drivable (and the flag agrees with the config)."""
        want = bool(dvb._a(a, "refcv8", False)) and bool(dvb._a(a, "r8_critic_drivable", False))
        return _drv_field(m, a) + dvb._eq("r8_critic_drivable", want, getattr(m, "r8_drv", None) is not None,
                                          "model.r8_drv is not None")

    dvb.register("r8_critic_drivable", "built", _c_drv)
    dvb.register("w_r8_drivable", "loss", _field("w_r8_drivable", "w_drivable", float, 0.0))
    dvb.register("r8_alloc_emit_start", "built",
                 _field("r8_alloc_emit_start", "emit_start", lambda v: int(v or 0), 0))
    dvb.register("r8_rc_dropout", "runtime", reason=(
        "set on the model every step by the train loop (`model._r8_rc_dropout`) and read by "
        "refcv8_train.r8_before_forward, which REFUSES a value below 0.3 (MM binding 2026-10-04)"))
    dvb.register("r8_nav_args_dropout", "runtime", reason=(
        "set on the model every step (`model._r8_nav_args_dropout`); read by r8_before_forward: a dropped "
        "row's args are zeroed with the known bit cleared, the nav TOKEN is kept"))
    dvb.register("r8_rc_noise_along_m", "runtime", reason=(
        "set on the model (`model._r8_rc_noise`) and read by r8_before_forward on TRAINING rows only; refused "
        "below the E2'-certified 2.0 m (WP-A INTEGRATION sec. 4: the clean checkpoint leaks)"))
    dvb.register("r8_rc_noise_lat_m", "runtime", reason=(
        "as --r8-rc-noise-along-m, the lateral sigma in the route-tangent frame; refused below 0.75 m"))
    for dest, why in (("r8_derange_feed", "`model._r8_derange_train`, read by r8_before_forward in TRAINING only (the "
                                         "planner reads another window's tactical output); a regression-arm diagnostic"),
                      ("r8_rc_roll", "`model._r8_rc_roll_train`, read by r8_before_forward in TRAINING only (each row "
                                    "gets another window's route checkpoint); a regression-arm diagnostic"),
                      ("r8_roll_targets", "`model._r8_roll_targets`, read by compute_losses_v3 in TRAINING only "
                                         "(refcv8_train.roll_targets); a regression-arm information control")):
        dvb.register(dest, "runtime", reason=why)
    dvb.register("init_from", "data", reason=(
        "the WARM-START checkpoint (refcv8_train.warm_start_from): every source key strict, every missing key a "
        "declared refcv8 seam, fresh optimiser; ignored when the run resumes its own ckpt.pt; md5 + source step in "
        "config.json[init_from]"))
    dvb.register("grad_share_every", "runtime", reason=(
        "the in-run gradient-share instrument's cadence (tanitad/train/grad_share.py, P-GRAD's statistic): "
        "autograd.grad on LOGGED steps only, never .grad, no parameter, no RNG; 0 = off; refused unless a multiple "
        "of --log-every (a reading on an unlogged step would be computed and discarded)"))
    def _c_speed(m, a):
        """X3: the BUILT model's source (`model.cfg.refcv8.speed_input`) equals the argv -- also on a SPEED-ONLY arm
        without the refcv8 seams (SPEC_WPB_LADDER V0), so it reads the model config, not the decoder's r8_cfg."""
        want = str(dvb._a(a, "r8_speed_input", None) or "")
        got = str(getattr(getattr(getattr(m, "cfg", None), "refcv8", None), "speed_input", "") or "")
        return dvb._eq("r8_speed_input", want, got, "model.cfg.refcv8.speed_input")

    dvb.register("r8_speed_input", "built", _c_speed)
    dvb.register("r8_speed_unknown_p", "runtime", reason=(
        "X3's trained unknown-row rate: `model._r8_speed_unknown_p`, read by r8_before_forward -> "
        "speed_input_treatment on TRAINING rows only (the dedicated generator); refused outside (0, 1); None = 0.45"))
    dvb.register("r8_roll_speed_input", "runtime", reason=(
        "`model._r8_roll_speed_train`, read by speed_input_treatment in TRAINING only (each row takes another "
        "window's speed input); SPEC_WPB_LADDER L4's deliberate-regression arm V-VSHUF"))
    dvb.register("smoke_seed_ego_frames", "runtime", reason=(
        "SMOKE ONLY (refused above --steps 100): the first batch is built by `smoke_seed_batch` from windows whose NOW "
        "frame is a --join-defect-masks ego frame, so the VIS-1 re-key runs at full size; read as `vis1_rekeyed`"))
    dvb.register("r8_no_rc", "runtime", reason=(
        "`model._r8_no_rc`: r8_before_forward never feeds the route checkpoint (the MM's switch-off while the RC "
        "ruling awaits PI confirmation); the model then sees an explicitly invalid row"))
    for dest, why in (("r8_v9_labels", "the WP-A v9 release (train) behind refcv8_train.R8LabelJoin; md5-checked "
                                       "(--r8-v9-md5), contract-censused, stamped in config.json[refcv8][v9]"),
                      ("r8_v9_labels_eval", "the WP-A v9 release (eval139) behind R8LabelJoin, md5-checked and "
                                            "stamped"),
                      ("r8_v9_md5", "the md5 the TRAIN release must have (load_v9_release refuses others)"),
                      ("r8_v9_eval_md5", "the md5 the EVAL release must have"),
                      ("r8_v9_lat_variant", "which v9 lateral variant supervises lat (a junction = default, b heading "
                                            "= the pre-registered arm), read by R8LabelJoin.item"),
                      ("r8_rc_variant", "selects which v9 route-checkpoint variant R8LabelJoin reads "
                                        "(RC-A30/50/80 or RC-B, v9 SPEC sec. 6.2)"),
                      ("r8_nav_from_v9", "makes V3Dataset feed the v9 per-frame announced nav token as nav_cmd "
                                         "(R8-2) instead of the per-clip v7 token")):
        dvb.register(dest, "data", reason=why)


_register_gdvb()
