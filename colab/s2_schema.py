"""S2 strategic-vocabulary schema for the Colab label lab — ONE swap point.

⚠️ =========================== PROVISIONAL ================================ ⚠️
This module carries a PROVISIONAL schema matching the architecture diagram's
strategic vocabulary. The AUTHORITATIVE schema is owned by the S2-gap agent
and will land at:

    TanitAD Research Hub/Data Engineering/Implementation/incoming/
        2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md

(not yet present as of 2026-08-16 — checked). When it lands, THIS FILE is the
only thing that changes: both notebooks import their vocabulary, argument
enums, mapping and validation from here and only here. Do not inline tokens
in a notebook cell.
⚠️ ======================================================================== ⚠️

Diagram vocabulary this provisional version encodes:
  g_str : FOLLOW_MAIN_ROAD (the default) · ROUTE_TO · LANE_TARGET · TURN
  a_str : PREPARE_LANE_CHANGE · REDUCE_TO · PREPARE_EXIT · PREPARE_STOP
  args  : CATEGORICAL only (closed enums below) — no free text, no floats.

Relation to the production v6 vocabulary (`stack/tanitad/models/v6.py`):
the fuser (`stack/scripts/ph1_fuse.py`) emits v6 tokens; `from_fused()` maps
them onto the S2 diagram vocabulary. Every lossy edge of that mapping is
recorded on the record (`_mapping_lossy`), never silently collapsed. The v6
token lists are pinned VERBATIM below and drift-checked against the real
module whenever it is importable (emitter and consumer cannot drift silently
— same rule ph1_fuse.py applies).

⛔ GOAL/SITUATION DISJOINTNESS (BINDING, Sayed 2026-08-03, CLAUDE.md):
a goal input is admissible, but it must NOT carry the situation classifier's
output in any form — class posterior, argmax, embedding, or any feature
derived from them. Labels here are built from ego / VLM / SAM3 / Alpamayo,
NEVER from a situation classifier; `assert_disjoint()` enforces the textual
invariant on every record and `validate()` enforces that every provenance
source is in the closed allowed set. If a future leg wants to add a source,
it must extend ALLOWED_SOURCES here, in review, not inline.

Labels MAY use ego / privileged signals (labels-may-use-ego rule) — which is
why every token carries per-token PROVENANCE, so the S-S gate's
goal-provenance audit can separate privileged-label evidence from
vision-only evidence without re-running anything.

This module is dependency-free (stdlib json only) on purpose: it must import
in bare Colab before any pip install, and on the dev box without torch.
"""
from __future__ import annotations

import json

SCHEMA_VERSION = "s2-lab-provisional-v0.1"
AUTHORITATIVE_DOC = ("TanitAD Research Hub/Data Engineering/Implementation/"
                     "incoming/2026-08-16-s2-strategic-gap/"
                     "S2_STRATEGIC_GAP.md")

# --------------------------------------------------------------------------- #
# The S2 diagram vocabulary (PROVISIONAL)                                      #
# --------------------------------------------------------------------------- #
#: FOLLOW_MAIN_ROAD is THE DEFAULT whenever no navigation route is set up
#: (PI 2026-08-11, mirrored from v6.py). NONE_ABSTAIN survives only for
#: genuinely ambiguous geometry — same rule as v6.
G_STR_TOKENS = ("FOLLOW_MAIN_ROAD", "ROUTE_TO", "LANE_TARGET", "TURN",
                "NONE_ABSTAIN")
G_STR_DEFAULT = "FOLLOW_MAIN_ROAD"

A_STR_TOKENS = ("PREPARE_LANE_CHANGE", "REDUCE_TO", "PREPARE_EXIT",
                "PREPARE_STOP")

#: Per-token CATEGORICAL argument enums. A token may only carry the args
#: named here, each from its closed list. (v6 carries numeric constraint
#: slots in physical units; the S2 diagram asks for categorical args — the
#: numeric spine stays available in the fused record's ego layer, so nothing
#: is lost, only not duplicated into the goal token.)
G_STR_ARGS: dict[str, dict[str, tuple]] = {
    "FOLLOW_MAIN_ROAD": {},
    "ROUTE_TO": {"via": ("nav_sign", "route_input", "none")},
    "LANE_TARGET": {"lane_index": tuple(range(7))},          # 0..6, from-right
    "TURN": {"direction": ("left", "right"),
             "kind": ("turn", "exit", "uturn")},
    "NONE_ABSTAIN": {},
}
#: REDUCE_TO band cutpoints are PROVISIONAL (m/s, from the ego spine's
#: v_min_future): crawl <2 · slow <6 · urban <15 · rural <25 · highway >=25.
A_STR_ARGS: dict[str, dict[str, tuple]] = {
    "PREPARE_LANE_CHANGE": {"direction": ("left", "right")},
    "REDUCE_TO": {"band": ("crawl", "slow", "urban", "rural", "highway")},
    "PREPARE_EXIT": {"side": ("left", "right")},
    "PREPARE_STOP": {"cause": ("signal", "sign", "traffic", "hazard",
                               "unknown")},
}
REDUCE_TO_BANDS_MS = (("crawl", 2.0), ("slow", 6.0), ("urban", 15.0),
                      ("rural", 25.0), ("highway", float("inf")))

#: Closed provenance set. "map" is reserved for xodr-derived option-set labels
#: (strategic_gt.py — NuRec scenes only); "fusion_default" marks the
#: FOLLOW_MAIN_ROAD default when no source voted.
ALLOWED_SOURCES = ("ego", "vlm", "sam3", "alpamayo", "map", "fusion_default")
CONFIDENCE = ("low", "med", "high")

# --------------------------------------------------------------------------- #
# v6 pins (VERBATIM from stack/tanitad/models/v6.py @ a84a1a0) + drift check   #
# --------------------------------------------------------------------------- #
V6_STRATEGIC_GOAL_TOKENS = (
    "KEEP_CORRIDOR", "LANE_TARGET", "EXIT_RIGHT", "EXIT_LEFT",
    "TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH", "ROUTE_TO", "STOP_AT",
    "FOLLOW_MAIN_ROAD", "NONE_ABSTAIN",
)
V6_STRATEGIC_ACTION_TOKENS = (
    "PREPARE_LANE_CHANGE", "HOLD_CORRIDOR", "REDUCE_TO", "PREPARE_EXIT",
    "PREPARE_STOP", "RESUME_CRUISE",
)
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
    assert tuple(v6.STRATEGIC_GOAL_TOKENS) == V6_STRATEGIC_GOAL_TOKENS, \
        "v6 STRATEGIC_GOAL_TOKENS drifted — update s2_schema.py pins"
    assert tuple(v6.STRATEGIC_ACTION_TOKENS) == V6_STRATEGIC_ACTION_TOKENS, \
        "v6 STRATEGIC_ACTION_TOKENS drifted — update s2_schema.py pins"
    assert tuple(v6.TACTICAL_LAT_ACTIONS) == V6_TACTICAL_LAT_ACTIONS, \
        "v6 TACTICAL_LAT_ACTIONS drifted — update s2_schema.py pins"
    assert tuple(v6.TACTICAL_LON_ACTIONS) == V6_TACTICAL_LON_ACTIONS, \
        "v6 TACTICAL_LON_ACTIONS drifted — update s2_schema.py pins"
    return "checked"


#: v6 g_str token -> (s2 token, args, lossy). Lossy edges are DECISIONS the
#: authoritative schema must confirm or replace — they are flagged on every
#: record they touch, so the review sheet shows them.
V6_TO_S2_GSTR: dict[str, tuple[str, dict, bool]] = {
    "FOLLOW_MAIN_ROAD": ("FOLLOW_MAIN_ROAD", {}, False),
    "ROUTE_TO": ("ROUTE_TO", {"via": "nav_sign"}, False),
    "LANE_TARGET": ("LANE_TARGET", {}, False),
    "TURN_LEFT": ("TURN", {"direction": "left", "kind": "turn"}, False),
    "TURN_RIGHT": ("TURN", {"direction": "right", "kind": "turn"}, False),
    "EXIT_LEFT": ("TURN", {"direction": "left", "kind": "exit"}, True),
    "EXIT_RIGHT": ("TURN", {"direction": "right", "kind": "exit"}, True),
    "KEEP_CORRIDOR": ("FOLLOW_MAIN_ROAD", {}, True),
    "STRAIGHT_THROUGH": ("FOLLOW_MAIN_ROAD", {}, True),
    # STOP_AT is a goal in v6; the S2 diagram carries stopping as the
    # strategic ACTION PREPARE_STOP. from_fused() emits that action when it
    # maps this token — the goal itself degrades to the default.
    "STOP_AT": ("FOLLOW_MAIN_ROAD", {}, True),
    "NONE_ABSTAIN": ("NONE_ABSTAIN", {}, False),
}

#: ph0_v2 B4 action verbs (schema `ph0-v2.2`) are ALREADY the strategic
#: action vocabulary, lowercased — v6's STRATEGIC_ACTION_TOKENS minus none.
#: HOLD_CORRIDOR / RESUME_CRUISE have no S2 diagram slot: dropped, flagged.
VERB_TO_A_STR: dict[str, str | None] = {
    "prepare_lane_change": "PREPARE_LANE_CHANGE",
    "reduce_to": "REDUCE_TO",
    "prepare_exit": "PREPARE_EXIT",
    "prepare_stop": "PREPARE_STOP",
    "hold_corridor": None,
    "resume_cruise": None,
}


def reduce_band(v_ms: float) -> str:
    for name, hi in REDUCE_TO_BANDS_MS:
        if v_ms < hi:
            return name
    return "highway"


# --------------------------------------------------------------------------- #
# record construction                                                          #
# --------------------------------------------------------------------------- #
def _tok(token: str, args: dict, provenance: list[str], confidence: str,
         **extra) -> dict:
    return {"token": token, "args": args,
            "provenance": sorted(set(provenance)), "confidence": confidence,
            **extra}


def from_fused(fused: dict, ego_extra: dict | None = None) -> dict:
    """PH1 fused record (+ optional ego-geometry extras) -> one S2 lab record.

    ``fused`` is the `ph1-fused-v1` shape produced by ph1_fuse / the lab's
    fuse_one(). ``ego_extra`` may carry the ego-yaw g_str vote
    (s2_lab_lib.ego_leg output) so the fusion default has a second opinion.
    Per-token provenance is REQUIRED output — the S-S gate's goal-provenance
    audit reads it."""
    vocab = fused.get("vocab") or {}
    sym = ((fused.get("semantics") or {}).get("symbols")) or {}
    ego = fused.get("ego") or {}
    sp = ego.get("speed_profile") or {}
    lossy: list[str] = []

    # ---- g_str: the v6 token the fuser emitted, mapped ---------------------
    v6_tok = (vocab.get("g_str") or {}).get("token") or "NONE_ABSTAIN"
    s2_tok, args, is_lossy = V6_TO_S2_GSTR.get(
        v6_tok, ("NONE_ABSTAIN", {}, True))
    if is_lossy:
        lossy.append(f"g_str {v6_tok} -> {s2_tok}")
    prov = ["vlm"]                       # ph1_fuse g_str src is the VLM B4
    conf = str(sym.get("conf") or "low")
    # ego-yaw second opinion (labels may use ego): corroborate or record
    ego_vote = (ego_extra or {}).get("g_str_vote")
    ego_agrees = None
    if ego_vote:
        ego_agrees = (ego_vote.get("token") == s2_tok
                      and all(args.get(k) == v
                              for k, v in (ego_vote.get("args") or {}).items()
                              if k in args))
        if ego_agrees:
            prov.append("ego")
    if v6_tok == "NONE_ABSTAIN" and ego_vote and ego_vote.get("token"):
        # VLM abstained but ego geometry votes: take the ego vote, said so.
        s2_tok, args = ego_vote["token"], dict(ego_vote.get("args") or {})
        prov, conf = ["ego"], "low"
    if s2_tok == "NONE_ABSTAIN":
        # the diagram default applies when nothing voted at all
        s2_tok, args, prov, conf = G_STR_DEFAULT, {}, ["fusion_default"], "low"
    if s2_tok == "LANE_TARGET" and "lane_index" not in args:
        lane = ((fused.get("semantics") or {}).get("scene") or {}).get(
            "lane_ego")
        if isinstance(lane, int) and 0 <= lane <= 6:
            args["lane_index"] = lane
    g_str = _tok(s2_tok, args, prov, conf,
                 v6_token=v6_tok,
                 ego_vote=(ego_vote or None),
                 ego_agrees=ego_agrees)

    # ---- a_str: B4 verbs (vlm) + ego-spine corroboration -------------------
    a_str: list[dict] = []
    seen: set[tuple] = set()

    def _add(token, args_, prov_, conf_, **extra):
        key = (token, tuple(sorted(args_.items())))
        if key in seen:
            for a in a_str:                    # merge provenance, keep first
                if a["token"] == token and a["args"] == args_:
                    a["provenance"] = sorted(set(a["provenance"]) | set(prov_))
            return
        seen.add(key)
        a_str.append(_tok(token, args_, prov_, conf_, **extra))

    for act in (sym.get("actions") or []):
        verb = str(act.get("verb") or "").lower()
        tok = VERB_TO_A_STR.get(verb)
        if tok is None:
            if verb in VERB_TO_A_STR:
                lossy.append(f"a_str verb {verb} dropped (no S2 slot)")
            continue
        args_: dict = {}
        if tok == "PREPARE_LANE_CHANGE":
            d = str(act.get("direction") or "none")
            if d not in ("left", "right"):
                lossy.append("PREPARE_LANE_CHANGE without direction dropped")
                continue
            args_["direction"] = d
        elif tok == "PREPARE_EXIT":
            args_["side"] = (act.get("direction")
                             if act.get("direction") in ("left", "right")
                             else "right")
        elif tok == "REDUCE_TO":
            vmin = sp.get("v_min_future")
            args_["band"] = (reduce_band(float(vmin)) if vmin is not None
                             else "urban")
        elif tok == "PREPARE_STOP":
            reds = [s for s in (((fused.get("semantics") or {}).get("signs")
                                 or {}).get("signs") or [])
                    if s.get("state") == "red"]
            args_["cause"] = "signal" if reds else "unknown"
        _add(tok, args_, ["vlm"], str(sym.get("conf") or "low"))

    # ego corroboration / ego-only votes (labels may use ego)
    lon = (vocab.get("g_tac_lon") or {})
    if lon.get("token") == "BRAKE_TO" and (sp.get("stops") or 0) > 0:
        _add("PREPARE_STOP", {"cause": "unknown"},
             ["ego"] + (["vlm"] if "vlm" in (lon.get("voters") or []) else []),
             "med", derived_from="g_tac_lon=BRAKE_TO + ego stops>0")
    lat = (vocab.get("g_tac_lat") or {})
    if lat.get("token") in ("LANE_CHANGE_L", "LANE_CHANGE_R"):
        _add("PREPARE_LANE_CHANGE",
             {"direction": "left" if lat["token"].endswith("_L") else "right"},
             list(lat.get("voters") or ["ego"]), "med",
             derived_from=f"g_tac_lat={lat['token']}")
    if (sp.get("net_dv") is not None and float(sp["net_dv"]) < -3.0
            and (sp.get("stops") or 0) == 0):
        _add("REDUCE_TO", {"band": reduce_band(float(sp["v_min_future"]))},
             ["ego"], "med", derived_from="ego net_dv < -3 m/s, no stop")
    if v6_tok == "STOP_AT":
        _add("PREPARE_STOP", {"cause": "unknown"}, ["vlm"], conf,
             derived_from="v6 g_str STOP_AT (mapped to action)")

    rec = {
        "schema_version": SCHEMA_VERSION,
        "authoritative_doc": AUTHORITATIVE_DOC,
        "clip_id": fused.get("clip_id"),
        "g_str": g_str,
        "a_str": a_str,
        "g_tac": {"lat": vocab.get("g_tac_lat"),
                  "lon": vocab.get("g_tac_lon")},
        "_mapping_lossy": lossy,
        "_provisional": True,
    }
    assert_disjoint(rec)
    errs = validate(rec)
    if errs:
        raise ValueError(f"s2 record invalid for {rec['clip_id']}: {errs}")
    # the prose note is attached AFTER the assert — see assert_disjoint's
    # docstring for why it must never be part of the scanned payload
    rec["_disjointness"] = ("goal payload information-disjoint from the "
                            "sit. classifier: asserted on g_str/a_str/g_tac")
    return rec


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #
def assert_disjoint(rec: dict) -> None:
    """The ph1_fuse rule: no situation-classifier output may reach a
    goal/action field, in any spelling.

    ⚠️ Scans ONLY the goal-payload fields (g_str / a_str / g_tac), never the
    record's prose/meta fields — the first version scanned the whole record
    and matched its OWN `_disjointness` explanatory note, which contains the
    word it searches for. That is CLAUDE.md's polling-monitor trap in
    miniature (the emitted marker must be disjoint from the searched token);
    MEASURED here on the first smoke run, 2026-08-16."""
    payload = {k: rec.get(k) for k in ("g_str", "a_str", "g_tac")}
    blob = json.dumps(payload).lower()
    for needle in ("situation", "sitclf"):
        assert needle not in blob, (
            f"goal/situation disjointness violated: {needle!r} found in the "
            "goal payload — a goal input must not carry the situation "
            "classifier's output (BINDING, Sayed 2026-08-03)")


def _check_tok(t: dict, valid_tokens, arg_spec, where: str) -> list[str]:
    errs = []
    if t.get("token") not in valid_tokens:
        errs.append(f"{where}: token {t.get('token')!r} not in {valid_tokens}")
        return errs
    spec = arg_spec.get(t["token"], {})
    for k, v in (t.get("args") or {}).items():
        if k not in spec:
            errs.append(f"{where}: arg {k!r} not allowed for {t['token']}")
        elif v not in spec[k]:
            errs.append(f"{where}: arg {k}={v!r} not in {spec[k]}")
    prov = t.get("provenance")
    if not prov:
        errs.append(f"{where}: provenance MISSING (required per token)")
    else:
        bad = [p for p in prov if p not in ALLOWED_SOURCES]
        if bad:
            errs.append(f"{where}: provenance {bad} not in ALLOWED_SOURCES")
    if t.get("confidence") not in CONFIDENCE:
        errs.append(f"{where}: confidence {t.get('confidence')!r} invalid")
    return errs


def validate(rec: dict) -> list[str]:
    """Full record check. Returns a list of violations (empty = valid)."""
    errs: list[str] = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version {rec.get('schema_version')!r} != "
                    f"{SCHEMA_VERSION}")
    if not rec.get("clip_id"):
        errs.append("clip_id missing")
    g = rec.get("g_str")
    if not isinstance(g, dict):
        errs.append("g_str missing")
    else:
        errs += _check_tok(g, G_STR_TOKENS, G_STR_ARGS, "g_str")
    for i, a in enumerate(rec.get("a_str") or []):
        errs += _check_tok(a, A_STR_TOKENS, A_STR_ARGS, f"a_str[{i}]")
    return errs


if __name__ == "__main__":
    print(f"s2_schema {SCHEMA_VERSION} (PROVISIONAL — authoritative: "
          f"{AUTHORITATIVE_DOC})")
    print(f"g_str: {G_STR_TOKENS}")
    print(f"a_str: {A_STR_TOKENS}")
    print(f"v6 drift check: {check_v6_drift()}")
