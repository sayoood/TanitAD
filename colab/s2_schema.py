"""S2 strategic-vocabulary schema — AUTHORITATIVE `s2-strategic-v1`.

This file replaces the PROVISIONAL v0.1 diagram schema, exactly as promised
there: *"When it lands, THIS FILE is the only thing that changes."* Both
notebooks keep importing vocabulary, argument spec, mapping and validation
from here and only here.

Authoritative spec: `TanitAD Research Hub/Data Engineering/Implementation/
incoming/2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md` §1.2, executed by
`…/2026-08-16-s2-v1-labels/`. The format is THE CODE-SIDE CONTRACT: the full
v6 vocabularies (11 `g_str` / 6 `a_str` tokens), 8 float arg slots in
physical units + an arg mask (IGNORE discipline: slot unset ⇒ zero
gradient), per-instance PROVENANCE ∈ {path, signage, vlm-fused}, a validity
band around t0, and an asserted goal/situation-disjointness stamp.

The DERIVATION (geometry-primary: Engine A decides, VLM corroborates,
ROUTE_TO gated) lives in `stack/scripts/s2_derive.py` — one home, three
consumers (ph1_fuse, this module's `from_fused`, the S2 v1 builder).
`s2_derive` is stdlib-only; this module lazily adds `stack/scripts` to the
path when `from_fused` is first called, so bare-Colab import of THIS module
still needs nothing installed.

⛔ GOAL/SITUATION DISJOINTNESS (BINDING, Sayed 2026-08-03): labels here are
built from ego geometry / VLM / SAM3 / Alpamayo, NEVER from a situation
classifier; `assert_disjoint()` scans the goal payload (and ONLY the goal
payload — the first version matched its own explanatory note, the
polling-monitor trap in miniature; MEASURED 2026-08-16) and `s2_derive`
reads Engine A through an allowlist that cannot see `situations`.

⛔ ROUTE_TO IS GATED: G1 CLOSED at 0/31 (sign text unverifiable at 448 px)
AND its `text_token_id` arg is categorical with no categorical channel on
`vocab_str` (v6.py:2265-2274). `validate()` REFUSES a ROUTE_TO label —
re-opening it is a deliberate edit here, in review, never a drive-by.

Labels MAY use ego / privileged signals (labels-may-use-ego rule) — which is
why every token carries per-instance PROVENANCE, so the S-S gate's
goal-provenance audit separates privileged-label evidence from vision-only
evidence without re-running anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "s2-strategic-v1"
AUTHORITATIVE_DOC = ("TanitAD Research Hub/Data Engineering/Implementation/"
                     "incoming/2026-08-16-s2-strategic-gap/"
                     "S2_STRATEGIC_GAP.md")

# --------------------------------------------------------------------------- #
# The vocabularies (VERBATIM v6 pins; drift-checked against the real module)    #
# --------------------------------------------------------------------------- #
G_STR_TOKENS = (
    "KEEP_CORRIDOR", "LANE_TARGET", "EXIT_RIGHT", "EXIT_LEFT",
    "TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH", "ROUTE_TO", "STOP_AT",
    "FOLLOW_MAIN_ROAD", "NONE_ABSTAIN",
)
G_STR_DEFAULT = "FOLLOW_MAIN_ROAD"      # THE DEFAULT with no route set up
A_STR_TOKENS = (
    "PREPARE_LANE_CHANGE", "HOLD_CORRIDOR", "REDUCE_TO", "PREPARE_EXIT",
    "PREPARE_STOP", "RESUME_CRUISE",
)
GOAL_ARG_NAMES = ("arg0", "arg1", "arg2", "arg3",
                  "within_m", "by_time_s", "at_arc_m", "hold_for_s")
GOAL_ARG_SLOTS = len(GOAL_ARG_NAMES)                                       # 8
PROVENANCE_CLASSES = ("path", "signage", "vlm-fused")
T0_S_DEFAULT = 8.0                       # ph0_v2 decision time (t0_idx=80)
VALID_WINDOW_S_DEFAULT = (-2.0, 2.0)     # §1.2 validity band around t0

#: Which arg slots a token MAY set (physical units; §1.2 conventions table).
#: Sign convention for ±1 direction args: +1 = left, −1 = right (declared in
#: s2_derive — matches the ego frame's +y = left).
ARG_SLOT_SPEC: dict[str, tuple[str, ...]] = {
    "KEEP_CORRIDOR": ("arg0",),                    # target_arc_m
    "LANE_TARGET": ("arg0", "arg1"),               # lane_offset_idx, deadline_m
    "EXIT_RIGHT": ("arg0",), "EXIT_LEFT": ("arg0",),          # distance_m
    "TURN_LEFT": ("arg0",), "TURN_RIGHT": ("arg0",),          # junction arc m
    "STRAIGHT_THROUGH": ("arg0",),
    "ROUTE_TO": (),                                # ⛔ unfillable (gated)
    "STOP_AT": ("arg0",),                          # distance_m
    "FOLLOW_MAIN_ROAD": (), "NONE_ABSTAIN": (),
    "PREPARE_LANE_CHANGE": ("arg0", "within_m"),   # dir ±1, envelope
    "HOLD_CORRIDOR": ("at_arc_m",),
    "REDUCE_TO": ("arg0", "within_m"),             # v_target_ms, envelope
    "PREPARE_EXIT": ("arg0", "within_m"),          # dir ±1, envelope
    "PREPARE_STOP": ("within_m",),
    "RESUME_CRUISE": ("arg0",),                    # v_target_ms
}

# v6 tactical pins kept for the drift check (the lab shows g_tac beside g_str)
V6_TACTICAL_LAT_ACTIONS = (
    "LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L",
    "NUDGE_R",
)
V6_TACTICAL_LON_ACTIONS = (
    "FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
)


def check_v6_drift() -> str:
    """Assert the pins above equal the real v6 module, when importable.

    Returns 'checked' | 'v6 not importable (pins used)'. Raises AssertionError
    on drift — a drifted pin must never silently keep emitting."""
    try:
        from tanitad.models import v6
    except Exception:                                        # noqa: BLE001
        return "v6 not importable (pins used)"
    assert tuple(v6.STRATEGIC_GOAL_TOKENS) == G_STR_TOKENS, \
        "v6 STRATEGIC_GOAL_TOKENS drifted — update s2_schema.py pins"
    assert tuple(v6.STRATEGIC_ACTION_TOKENS) == A_STR_TOKENS, \
        "v6 STRATEGIC_ACTION_TOKENS drifted — update s2_schema.py pins"
    assert tuple(v6.GOAL_ARG_NAMES) == GOAL_ARG_NAMES, \
        "v6 GOAL_ARG_NAMES drifted — update s2_schema.py pins"
    assert tuple(v6.TACTICAL_LAT_ACTIONS) == V6_TACTICAL_LAT_ACTIONS, \
        "v6 TACTICAL_LAT_ACTIONS drifted — update s2_schema.py pins"
    assert tuple(v6.TACTICAL_LON_ACTIONS) == V6_TACTICAL_LON_ACTIONS, \
        "v6 TACTICAL_LON_ACTIONS drifted — update s2_schema.py pins"
    return "checked"


def _s2_derive():
    """Lazy import of the derivation core (stack/scripts/s2_derive.py)."""
    try:
        import s2_derive
        return s2_derive
    except ImportError:
        pass
    scripts = Path(__file__).resolve().parents[1] / "stack" / "scripts"
    if scripts.is_dir():
        sys.path.insert(0, str(scripts))
        import s2_derive
        return s2_derive
    raise ImportError(
        "s2_derive not importable — the derivation core lives at "
        "stack/scripts/s2_derive.py; clone/mount the repo so it is on the "
        "path (RUNNER.md §1)")


# --------------------------------------------------------------------------- #
# record construction                                                          #
# --------------------------------------------------------------------------- #
def build_record(clip_id: str, g_str: dict, a_str: dict, *,
                 t0_s: float = T0_S_DEFAULT,
                 valid_window_s=VALID_WINDOW_S_DEFAULT,
                 provenance_notes: dict | None = None,
                 extra: dict | None = None) -> dict:
    """One `s2-strategic-v1` record from derived g_str/a_str blocks.

    ``g_str``/``a_str`` are `s2_derive.derive_*` outputs (token, token_id,
    args, arg_mask, provenance, sources, confidence, corroboration, …).
    Validates + asserts disjointness before returning — an invalid record is
    an exception, never a silently-written file."""
    rec = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "t0_s": float(t0_s),
        "g_str": g_str,
        "a_str": a_str,
        "valid_window_s": [float(valid_window_s[0]), float(valid_window_s[1])],
        "disjointness": {"situation_classifier_output_used": False},
    }
    if provenance_notes:
        rec["_provenance"] = provenance_notes
    if extra:
        rec.update(extra)
    assert_disjoint(rec)
    errs = validate(rec)
    if errs:
        raise ValueError(f"s2 record invalid for {clip_id}: {errs}")
    return rec


def from_fused(fused: dict, ego_extra: dict | None = None,
               engine_a: dict | None = None) -> dict:
    """PH1 fused record (+ optional structured Engine A) -> one v1 record.

    Engine A resolution order: the ``engine_a`` argument, then the fused
    record's banked ``ego.engine_a`` (ph1_fuse ≥ this change persists it).
    With neither present the derivation falls back to VLM-primary, tagged
    ``vlm-fused`` — stated on the record, never silent.

    ``ego_extra`` (the lab's ego-yaw vote) is recorded for audit only — the
    structured Engine A supersedes it as the geometric authority."""
    sd = _s2_derive()
    ego = fused.get("ego") or {}
    ea = engine_a or ego.get("engine_a")
    sym = ((fused.get("semantics") or {}).get("symbols")) or {}
    g_str = sd.derive_g_str(ea, sym)
    a_str = sd.derive_a_str(ea, sym)
    if ego_extra and ego_extra.get("g_str_vote"):
        g_str.setdefault("corroboration", {})["ego_yaw_vote"] = \
            ego_extra["g_str_vote"]

    prov = dict((fused.get("_provenance") or {}))
    prov.setdefault("engine_a", "privileged-labels-only (hindsight ego path)"
                    if ea else "absent — vlm-primary fallback")
    vocab = fused.get("vocab") or {}
    rec = build_record(
        fused.get("clip_id"), g_str, a_str,
        provenance_notes=prov,
        extra={"_v6_drift_check": check_v6_drift(),
               # audit passthrough for the lab's review sheet — NOT part of
               # the trainer contract (unknown keys are ignored there)
               "g_tac": {"lat": vocab.get("g_tac_lat"),
                         "lon": vocab.get("g_tac_lon")}})
    return rec


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #
def assert_disjoint(rec: dict) -> None:
    """The binding rule: no situation-classifier output may reach a
    goal/action field, in any spelling.

    ⚠️ Scans ONLY the goal-payload fields (g_str / a_str), never the record's
    prose/meta fields — the first version scanned the whole record and
    matched its OWN explanatory note, which contains the word it searches
    for. That is CLAUDE.md's polling-monitor trap in miniature (the emitted
    marker must be disjoint from the searched token); MEASURED here on the
    first smoke run, 2026-08-16."""
    payload = {k: rec.get(k) for k in ("g_str", "a_str")}
    blob = json.dumps(payload).lower()
    for needle in ("situation", "sitclf"):
        assert needle not in blob, (
            f"goal/situation disjointness violated: {needle!r} found in the "
            "goal payload — a goal input must not carry the situation "
            "classifier's output (BINDING, Sayed 2026-08-03)")


def _check_tok(t: dict, tokens: tuple, where: str) -> list[str]:
    errs: list[str] = []
    tok = t.get("token")
    if tok not in tokens:
        errs.append(f"{where}: token {tok!r} not in vocabulary")
        return errs
    if tok == "ROUTE_TO":
        errs.append(f"{where}: ROUTE_TO is GATED (G1 closed 0/31; no "
                    "categorical arg channel) — emit the geometry token or "
                    "NONE_ABSTAIN with a reason, never ROUTE_TO")
    if t.get("token_id") != tokens.index(tok):
        errs.append(f"{where}: token_id {t.get('token_id')} != "
                    f"{tokens.index(tok)} for {tok}")
    args, mask = t.get("args"), t.get("arg_mask")
    if not (isinstance(args, list) and len(args) == GOAL_ARG_SLOTS):
        errs.append(f"{where}: args must be [{GOAL_ARG_SLOTS}] floats")
        return errs
    if not (isinstance(mask, list) and len(mask) == GOAL_ARG_SLOTS
            and all(m in (0, 1) for m in mask)):
        errs.append(f"{where}: arg_mask must be [{GOAL_ARG_SLOTS}] of 0/1")
        return errs
    allowed = ARG_SLOT_SPEC.get(tok, ())
    for i, m in enumerate(mask):
        if m and GOAL_ARG_NAMES[i] not in allowed:
            errs.append(f"{where}: slot {GOAL_ARG_NAMES[i]!r} set but not "
                        f"allowed for {tok} (allowed: {allowed})")
        if not m and args[i] not in (0, 0.0):
            errs.append(f"{where}: slot {GOAL_ARG_NAMES[i]!r} unset but "
                        f"carries {args[i]} — an unset slot is 0.0 by "
                        "convention (nothing may read it)")
    if t.get("provenance") not in PROVENANCE_CLASSES:
        errs.append(f"{where}: provenance {t.get('provenance')!r} not in "
                    f"{PROVENANCE_CLASSES} (REQUIRED per instance)")
    if not t.get("sources"):
        errs.append(f"{where}: sources MISSING (auditable producers required)")
    return errs


def validate(rec: dict) -> list[str]:
    """Full record check. Returns a list of violations (empty = valid)."""
    errs: list[str] = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version {rec.get('schema_version')!r} != "
                    f"{SCHEMA_VERSION}")
    if not rec.get("clip_id"):
        errs.append("clip_id missing")
    if not isinstance(rec.get("t0_s"), (int, float)):
        errs.append("t0_s missing")
    vw = rec.get("valid_window_s")
    if not (isinstance(vw, list) and len(vw) == 2 and vw[0] <= 0.0 <= vw[1]):
        errs.append(f"valid_window_s {vw!r} must be [lo, hi] with lo<=0<=hi")
    dj = rec.get("disjointness")
    if not (isinstance(dj, dict)
            and dj.get("situation_classifier_output_used") is False):
        errs.append("disjointness stamp missing or not asserted False")
    g = rec.get("g_str")
    if not isinstance(g, dict):
        errs.append("g_str missing")
    else:
        errs += _check_tok(g, G_STR_TOKENS, "g_str")
    a = rec.get("a_str")
    if not isinstance(a, dict):
        errs.append("a_str missing")
    else:
        errs += _check_tok(a, A_STR_TOKENS, "a_str")
    return errs


if __name__ == "__main__":
    print(f"s2_schema {SCHEMA_VERSION} (authoritative; spec: "
          f"{AUTHORITATIVE_DOC})")
    print(f"g_str: {G_STR_TOKENS}")
    print(f"a_str: {A_STR_TOKENS}")
    print(f"args:  {GOAL_ARG_NAMES}")
    print(f"v6 drift check: {check_v6_drift()}")
