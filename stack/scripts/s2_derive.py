"""S2 v1 strategic-label derivation — geometry-primary g_str / a_str.

THE one home of the S2 mapping (fix list `…/2026-08-16-s2-strategic-gap/
S2_STRATEGIC_GAP.md` §6 items 2+3). Three consumers, zero drift:
  * `scripts/ph1_fuse.py` (the fuser's vocab block),
  * `colab/s2_schema.py::from_fused` (the Colab label lab),
  * the S2 v1 label builder (`…/2026-08-16-s2-v1-labels/code/`).

Jurisdiction (measured, not aesthetic): on aug120 the VLM's TURN precision is
100 % (19/19) but recall is 17/33 L / **2/29 R**, and 28 of its 31 ROUTE_TO
sit on plain turn geometry — while Engine A (`route_from_future_v3` /
`latmaneuver` / `lonmode` over the hindsight ego path) covers every clip.
⇒ **geometry decides, the VLM corroborates** — the same jurisdiction split
the fusion strategy uses everywhere else.

⛔ ROUTE_TO is GATED (G1 CLOSED at 0/31 + no categorical arg channel on
`vocab_str`): a VLM `route_to` is REMAPPED to its geometry token when junction
geometry backs it, and ABSTAINED (NONE_ABSTAIN + reason) otherwise. It is
never emitted as a guess.

⛔ LANE_TARGET / PREPARE_LANE_CHANGE — see §LC below. The PI adjudicated 18 of
the 19 LANE_TARGET labels this module emitted on aug120 and called 14 wrong;
on 2026-08-16 he ruled the geometric gate out entirely. `LANE_TARGET` is no
longer emitted at all, and `PREPARE_LANE_CHANGE` only when the ROUTE requires
a lane the ego is not in. ⚠️ Both tokens REMAIN IN THE v6 VOCABULARY — the
tuples size embedding tables and the live v6F run resumes strictly against
them; zero training support is safe, a changed shape is not.

⛔ GOAL/SITUATION DISJOINTNESS (BINDING, Sayed 2026-08-03): derivation reads
Engine A through the `ENGINE_A_ALLOWED` allowlist — `situations` is
structurally unreadable here, the same load-bearing omission
`ph0_pilot._fmt_engine_a` makes for the B4 prompt.

Args discipline: physical units into the 8 v6 slots
(`arg0..arg3, within_m, by_time_s, at_arc_m, hold_for_s`), mask=1 ONLY where
a measured value exists — a null `dist_m` yields an unset slot, never a
fabricated 0.0 (the `8dc5d14d…` dyaw=0/dist=null turn row is the pinned case).

Sign convention (DECLARED, used for every ±1 direction arg):
**+1 = left, −1 = right** (matches the ego frame's +y = left in
`refb_labels.ego_frame`).

stdlib-only on purpose — importable in bare Colab and on the dev box without
torch. Vocabulary pins are drift-checked against the real
`tanitad.models.v6` wherever that module is importable.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# vocabulary pins (VERBATIM from stack/tanitad/models/v6.py; drift-checked)     #
# --------------------------------------------------------------------------- #
STRATEGIC_GOAL_TOKENS = (
    "KEEP_CORRIDOR", "LANE_TARGET", "EXIT_RIGHT", "EXIT_LEFT",
    "TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH", "ROUTE_TO", "STOP_AT",
    "FOLLOW_MAIN_ROAD", "NONE_ABSTAIN",
)
STRATEGIC_ACTION_TOKENS = (
    "PREPARE_LANE_CHANGE", "HOLD_CORRIDOR", "REDUCE_TO", "PREPARE_EXIT",
    "PREPARE_STOP", "RESUME_CRUISE",
)
GOAL_ARG_NAMES = ("arg0", "arg1", "arg2", "arg3",
                  "within_m", "by_time_s", "at_arc_m", "hold_for_s")
GOAL_ARG_SLOTS = len(GOAL_ARG_NAMES)                                       # 8
#: HIERARCHY_VOCABULARY.md §2 provenance classes (per instance, REQUIRED)
PROVENANCE_CLASSES = ("path", "signage", "vlm-fused")

#: VLM goal_kind -> g_str token (the closed identity-ish map; moved here from
#: ph1_fuse so emitter, fuser and builder share one copy).
GOAL_TO_GSTR = {
    "follow_main_road": "FOLLOW_MAIN_ROAD", "route_to": "ROUTE_TO",
    "keep_corridor": "KEEP_CORRIDOR", "lane_target": "LANE_TARGET",
    "exit_right": "EXIT_RIGHT", "exit_left": "EXIT_LEFT",
    "turn_left": "TURN_LEFT", "turn_right": "TURN_RIGHT",
    "straight_through": "STRAIGHT_THROUGH", "stop_at": "STOP_AT",
    "none": "NONE_ABSTAIN",
}

# --------------------------------------------------------------------------- #
# gates — every threshold named, none inline                                    #
# --------------------------------------------------------------------------- #
STOP_V_MS = 0.3         # refb_labels.STOP_V_MS — "stopped" threshold
START_V_MS = 0.5        # below this at t0 the clip begins stopped
#: ⛔ SUPERSEDED AS AN EMISSION GATE (PI ruling 2026-08-16, see §LC below).
#: Retained ONLY to recompute the observed lateral event for the
#: CORROBORATION block — it may never again decide a token. The old §4.2
#: reading ("3.0 m ~= a lane, and route_valid keeps junction curvature out")
#: is REFUTED: measured on aug120, 15 of the 19 clips this gate fired on have
#: a lateral offset FULLY EXPLAINED by constant-curvature road following.
LC_MIN_LAT_M = 3.0
#: REDUCE_TO primary-emission gate (stronger than the −1.0 corroboration
#: check: primary labels claim, corroboration only agrees).
REDUCE_NET_DV_MS = -3.0

_STOP_EVENT_TOKENS = ("stop_at_point", "hold_stop")
_LAUNCH_EVENT_TOKENS = ("launch",)
_COAST_EVENT_TOKENS = ("coast", "creep")
_JUNCTION_ROUTE_TOKENS = ("turn_left", "turn_right", "exit_left", "exit_right",
                          "u_turn", "roundabout", "straight")

#: ⛔ the disjointness allowlist — `situations` is NOT here, by construction.
ENGINE_A_ALLOWED = ("route", "lane_change_events", "speed_events",
                    "speed_profile", "peak_kappa_per_m", "t0_idx",
                    "duration_s")


def check_vocab_drift() -> str:
    """Assert the pins equal the real v6 module when it is importable."""
    try:
        from tanitad.models import v6
    except Exception:                                        # noqa: BLE001
        return "v6 not importable (pins used)"
    assert tuple(v6.STRATEGIC_GOAL_TOKENS) == STRATEGIC_GOAL_TOKENS, \
        "v6 STRATEGIC_GOAL_TOKENS drifted — update s2_derive.py pins"
    assert tuple(v6.STRATEGIC_ACTION_TOKENS) == STRATEGIC_ACTION_TOKENS, \
        "v6 STRATEGIC_ACTION_TOKENS drifted — update s2_derive.py pins"
    assert tuple(v6.GOAL_ARG_NAMES) == GOAL_ARG_NAMES, \
        "v6 GOAL_ARG_NAMES drifted — update s2_derive.py pins"
    return "checked"


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _view(engine_a: dict | None) -> dict:
    """Engine A through the disjointness allowlist (never the raw dict)."""
    ea = engine_a or {}
    return {k: ea.get(k) for k in ENGINE_A_ALLOWED}


def _args(**named) -> tuple[list[float], list[int]]:
    """(args[8], mask[8]) from named slots; a None value NEVER sets a slot."""
    args = [0.0] * GOAL_ARG_SLOTS
    mask = [0] * GOAL_ARG_SLOTS
    for name, val in named.items():
        i = GOAL_ARG_NAMES.index(name)
        if val is not None:
            args[i] = round(float(val), 3)
            mask[i] = 1
    return args, mask


def _first_event(events, tokens, *, min_t_start=None):
    for ev in (events or []):
        if ev.get("token") not in tokens:
            continue
        if min_t_start is not None and float(ev.get("t_start_s", 0.0)) \
                <= min_t_start:
            continue
        return ev
    return None


def _gated_lc_event(ea: dict):
    """First OBSERVED lateral-displacement event under the old §4.2 gate.

    ⛔ NOT A LABEL SOURCE ANY MORE (PI 2026-08-16). The return value feeds the
    `corroboration` block only — see `lane_change_requirement()`. Kept as a
    private helper so the observation stays recorded and auditable; a caller
    that maps this to a token is violating the ruling.
    """
    route = ea.get("route") or {}
    if not (route.get("token_valid") and route.get("token") == "follow"):
        return None
    for ev in (ea.get("lane_change_events") or []):
        if ev.get("token") in ("lc_left", "lc_right") \
                and abs(float(ev.get("lat_m") or 0.0)) >= LC_MIN_LAT_M:
            return ev
    return None


# --------------------------------------------------------------------------- #
# §LC — ⛔ BINDING PI RULING 2026-08-16: THE LANE-CHANGE LABEL IS REASON-BASED  #
# --------------------------------------------------------------------------- #
#: Sayed, verbatim (2026-08-16, after adjudicating 18 of the 19 emitted
#: LANE_TARGET labels and calling 14 of them wrong):
#:
#:   "stop emitting lane_target/Prepare Lane change from geometric gate.
#:    Prepare lanbe change should onyl be to follow route or follow the main
#:    road (no route set), it must be derived from the context"
#:
#: Consequences, in force here:
#:   1. the geometric gate emits NEITHER token — removal, not retuning;
#:   2. `PREPARE_LANE_CHANGE` is admissible ONLY in service of route
#:      following: (a) a route is set and the ego's CURRENT lane does not
#:      serve it, or (b) no route is set, FOLLOW_MAIN_ROAD applies, and the
#:      current lane does not continue as the main road (lane ends / forced
#:      merge / exit-only lane). Nothing else licenses it;
#:   3. `LANE_TARGET` leaves g_str emission ENTIRELY — under (2) a lane
#:      change is a MEANS, not a GOAL, and the v6 vocabulary split agrees
#:      (`STRATEGIC_GOAL_TOKENS` is the goal channel, `STRATEGIC_ACTION_TOKENS`
#:      the action channel, v6.py:151/157). ⚠️ The token STAYS IN THE
#:      VOCABULARY — the tuples size embedding tables (v6.py:3281) and the
#:      live v6F run resumes strictly against them. Zero support is safe; a
#:      changed shape is not.
#:   4. the ego's ACTUAL lateral displacement is CORROBORATION, never the
#:      source. Observed-but-unrequired is a real category (overtaking,
#:      courtesy) and must not silently become a positive label; equally, a
#:      route-required repositioning the driver failed to execute is still
#:      required. The two can disagree, and the record says so.
#:
#: ⚠️ This INVERTS the pipeline's philosophy for this one label: every other
#: S2 token is derived from what the ego DID; this one is derived from what
#: the route DEMANDS.
#:
#: The named context inputs. NONE of these exists in the corpus today —
#: `ph0_v2` B1 supplies `lanes_visible`/`lane_ego`, but the PI's own review
#: calls the count wrong, its `conf` is degenerate ("high" on 201/201) and it
#: never uses its `0 = unclear` escape (0/201). PhysicalAI-AV ships no map or
#: lane graph. ⇒ `required` is None (UNKNOWN) for every clip today, and the
#: honest emission is therefore the route's own token, never a guess.
LANE_CONTEXT_INPUTS = (
    "n_lanes_same_direction",   # lanes on the ego's carriageway (NOT total)
    "ego_lane_idx",             # ego's 0-based lane index from the right
    "route_lane_idx",           # which lane serves the route / main road
    "lane_continues",           # does the ego lane continue (not exit-only)?
)


def lane_change_requirement(engine_a: dict | None,
                            lane_context: dict | None = None) -> dict:
    """Does the ROUTE require the ego to be in a different lane?

    ⛔ Reads the ROUTE and the LANE CONTEXT only. It is structurally unable to
    read `lane_change_events` — observed lateral displacement is not evidence
    of a requirement, and that conflation is the defect this replaces.

    Returns ``{required: True|False|None, reason, missing, basis}``;
    ``required=None`` means UNKNOWN and must produce no positive token.
    """
    ea = _view(engine_a)
    route = ea.get("route") or {}
    ctx = lane_context or {}
    missing = [k for k in LANE_CONTEXT_INPUTS if ctx.get(k) is None]
    valid = bool(route.get("token_valid"))
    tok_r = route.get("token")
    basis = ("route" if valid and tok_r in _JUNCTION_ROUTE_TOKENS
             else "follow_main_road")
    if missing:
        return {"required": None, "basis": basis, "missing": missing,
                "reason": ("lane context unavailable — cannot establish "
                           f"whether the {basis} requires a different lane "
                           f"(missing: {', '.join(missing)})")}
    ego_i, want_i = ctx["ego_lane_idx"], ctx["route_lane_idx"]
    if ego_i == want_i and ctx["lane_continues"]:
        return {"required": False, "basis": basis, "missing": [],
                "reason": f"ego lane {ego_i} already serves the {basis}"}
    if ego_i == want_i:
        # the serving lane IS the ego lane but it does not continue (lane
        # ends / forced merge). A change is required; WHICH WAY is not
        # derivable from these inputs. ⛔ The args discipline applies — the
        # direction slot stays UNSET rather than being invented.
        return {"required": True, "basis": basis, "missing": [],
                "direction": None, "n_lanes": None,
                "reason": (f"ego lane {ego_i} does not continue; direction "
                           "not derivable from the lane context")}
    # +1 = left, −1 = right (the module's declared sign convention)
    return {"required": True, "basis": basis, "missing": [],
            "direction": 1.0 if want_i > ego_i else -1.0,
            "n_lanes": int(want_i - ego_i),
            "reason": (f"ego lane {ego_i} does not serve the {basis} "
                       f"(lane {want_i} does)")}


def _tok(token: str, args, mask, provenance: str, sources: list[str],
         confidence: float, **extra) -> dict:
    assert provenance in PROVENANCE_CLASSES
    d = {"token": token, "args": args, "arg_mask": mask,
         "provenance": provenance, "sources": sources,
         "confidence": round(confidence, 2)}
    d.update(extra)
    return d


# --------------------------------------------------------------------------- #
# g_str — the strategic goal (geometry decides)                                 #
# --------------------------------------------------------------------------- #
def derive_g_str(engine_a: dict | None, vlm_symbols: dict | None, *,
                 lane_context: dict | None = None) -> dict:
    """One g_str judgement for one clip.

    Returns {token, token_id, args, arg_mask, provenance, sources,
    confidence, corroboration, [reason]} — the §1.2 record's g_str block.

    ⛔ `LANE_TARGET` IS NEVER EMITTED HERE (§LC, PI 2026-08-16): under the
    route-serving definition a lane change is a MEANS, so it belongs to
    `a_str`. The observed lateral event is recorded in
    ``corroboration.observed_lane_change`` and decides nothing.
    """
    ea = _view(engine_a)
    sym = vlm_symbols or {}
    route = ea.get("route") or {}
    sp = ea.get("speed_profile") or {}
    tok_r = route.get("token")
    valid = bool(route.get("token_valid"))
    dist = route.get("dist_m")
    have_ea = bool(engine_a) and bool(route)

    token, args, mask, sources, conf, reason = \
        None, *_args(), [], 0.6, None
    # §LC: computed for the CORROBORATION block only — it decides no token.
    lc_ev = _gated_lc_event(ea)

    if not have_ea:
        # Engine A absent (legacy inputs): the VLM's symbol judgement is the
        # best available label — emitted VLM-PRIMARY and tagged so, never
        # silently defaulted. ROUTE_TO stays gated even here.
        vlm_goal = str(sym.get("goal_kind") or "").lower() or None
        vlm_tok = GOAL_TO_GSTR.get(vlm_goal) if vlm_goal else None
        corr = {"vlm_goal_kind": vlm_goal, "agrees": None,
                "note": "engine_a absent — vlm-primary fallback"}
        if vlm_goal == "route_to" or vlm_tok is None \
                or vlm_tok == "LANE_TARGET":
            out = _tok("NONE_ABSTAIN", *_args(), "vlm-fused",
                       ["vlm.goal_kind"], 0.3, corroboration=corr)
            out["reason"] = ("route_to gated (G1 closed) and no geometry"
                             if vlm_goal == "route_to"
                             else ("LANE_TARGET left g_str emission (§LC, PI "
                                   "2026-08-16) — a lane change is a MEANS, "
                                   "and the VLM's lane count is unreliable"
                                   if vlm_tok == "LANE_TARGET"
                                   else f"goal_kind {vlm_goal!r} unmapped"))
        else:
            out = _tok(vlm_tok, *_args(), "vlm-fused", ["vlm.goal_kind"],
                       0.5, corroboration=corr)
        out["token_id"] = STRATEGIC_GOAL_TOKENS.index(out["token"])
        return out

    if valid and tok_r in ("turn_left", "turn_right"):
        token = "TURN_LEFT" if tok_r == "turn_left" else "TURN_RIGHT"
        args, mask = _args(arg0=dist)
        sources, conf = [f"engine_a.route_v3={tok_r}"], 0.9
    elif valid and tok_r in ("exit_left", "exit_right"):
        token = "EXIT_LEFT" if tok_r == "exit_left" else "EXIT_RIGHT"
        args, mask = _args(arg0=dist)
        sources, conf = [f"engine_a.route_v3={tok_r}"], 0.9
    elif valid and tok_r == "straight":
        # route_v3 never emits `straight` today (a junction is a MAP fact,
        # refb_labels.py:1022) — the path exists so a future v4 lands here.
        token = "STRAIGHT_THROUGH"
        args, mask = _args(arg0=dist)
        sources, conf = ["engine_a.route_v3=straight"], 0.9
    elif valid and tok_r == "u_turn":
        if route.get("uturn_roundabout_confounded"):
            # the same ego track satisfies BOTH signatures — guessing between
            # them is not a label (refb_labels route_from_future_v3 docstring)
            token, conf = "NONE_ABSTAIN", 0.3
            sources = ["engine_a.route_v3=u_turn"]
            reason = "u_turn/roundabout confounded — unverified without a map"
        else:
            # a u-turn is an extreme left maneuver in right-hand traffic —
            # the same confirm-set ph0_pilot._GOAL_EXPECT uses for turn_left
            token = "TURN_LEFT"
            args, mask = _args(arg0=dist)
            sources, conf = ["engine_a.route_v3=u_turn"], 0.8
    elif valid and tok_r == "roundabout":
        token, conf = "NONE_ABSTAIN", 0.3
        sources = ["engine_a.route_v3=roundabout"]
        reason = ("roundabout traverse has no 11-token mapping "
                  "(exit choice is a map-level fact)")
    else:
        stop_ev = _first_event(ea.get("speed_events"), _STOP_EVENT_TOKENS)
        began_stopped = (sp.get("v_t0_ms") is not None
                         and float(sp["v_t0_ms"]) < START_V_MS)
        stop_ahead = _first_event(ea.get("speed_events"), _STOP_EVENT_TOKENS,
                                  min_t_start=0.0) if began_stopped else stop_ev
        if stop_ahead is not None:
            token = "STOP_AT"
            args, mask = _args(arg0=(stop_ahead.get("stop_dist_m")
                                     if stop_ahead.get("stop_dist_m")
                                     is not None
                                     else stop_ahead.get("arc_from_t0_m")))
            sources = [f"engine_a.lonmode={stop_ahead.get('token')}"]
            conf = 0.85
        elif (not began_stopped) and sp.get("stops") and have_ea:
            # profile-level stop with no lonmode event: real, distance unknown
            token, sources, conf = "STOP_AT", ["engine_a.speed_profile.stops"], 0.7
        # ⛔ the LANE_TARGET branch that used to sit HERE is REMOVED (§LC, PI
        # 2026-08-16). A clip with an observed lateral event now falls through
        # to its ROUTE token — which for these clips is the measured
        # `follow` -> FOLLOW_MAIN_ROAD, not a default-of-absence.
        elif valid and tok_r in ("follow", "merge"):
            token = "FOLLOW_MAIN_ROAD"
            sources, conf = [f"engine_a.route_v3={tok_r}"], 0.8
        else:
            token = "FOLLOW_MAIN_ROAD"          # THE DEFAULT (PI, v6.py:136)
            sources, conf = ["pi_default:no_route_established"], 0.6
            if not have_ea:
                sources = ["pi_default:engine_a_absent"]

    # ---- the VLM's judgement, recorded as corroboration — never decisive ----
    vlm_goal = str(sym.get("goal_kind") or "").lower() or None
    vlm_tok = GOAL_TO_GSTR.get(vlm_goal) if vlm_goal else None
    corr: dict = {"vlm_goal_kind": vlm_goal,
                  "agrees": (vlm_tok == token) if vlm_tok else None}
    # §LC: the OBSERVATION is recorded and decides nothing.
    if lc_ev is not None:
        corr["observed_lane_change"] = {
            "token": lc_ev["token"], "lat_m": lc_ev.get("lat_m"),
            "arc_from_t0_m": lc_ev.get("arc_from_t0_m"),
            "is_label_source": False,
            "note": ("observed lateral displacement — CORROBORATION ONLY "
                     "(§LC, PI 2026-08-16); LANE_TARGET is not emitted")}
    if vlm_tok == "LANE_TARGET":
        corr["agrees"] = False
        corr["lane_target_gate"] = "LANE_TARGET left g_str emission (§LC)"
    if vlm_goal == "route_to":
        # ⛔ ROUTE_TO stays GATED: G1 CLOSED (0/31) + no categorical arg
        # channel on vocab_str. Geometry-backed claims are REMAPPED (the
        # corroboration records it); a claim with no geometric event to
        # remap to ABSTAINS rather than guessing FOLLOW.
        corr["agrees"] = False
        # ⛔ LANE_TARGET removed from the remap set (§LC): it is no longer an
        # emittable token, so it cannot be a remap destination either.
        if token in ("TURN_LEFT", "TURN_RIGHT", "EXIT_LEFT", "EXIT_RIGHT",
                     "STRAIGHT_THROUGH", "STOP_AT"):
            corr["remapped_from_route_to"] = True
            corr["route_to_gate"] = "G1 closed 0/31; geometry-backed remap"
        else:
            token, conf = "NONE_ABSTAIN", 0.3
            args, mask = _args()
            sources = ["route_to_gate"]
            reason = ("vlm route_to unverifiable (G1 closed, evidence sign "
                      "unchecked kind) and no junction geometry to remap to")
            corr["route_to_gate"] = "G1 closed; no geometry -> abstain"

    out = _tok(token, args, mask, "path", sources, conf, corroboration=corr)
    if reason:
        out["reason"] = reason
    out["token_id"] = STRATEGIC_GOAL_TOKENS.index(token)
    return out


# --------------------------------------------------------------------------- #
# a_str — the strategic action (geometry decides, VLM verb corroborates)        #
# --------------------------------------------------------------------------- #
def derive_a_str(engine_a: dict | None, vlm_symbols: dict | None, *,
                 lane_context: dict | None = None) -> dict:
    """One a_str judgement for one clip (same shape as derive_g_str).

    ⛔ `PREPARE_LANE_CHANGE` is emitted ONLY when `lane_change_requirement()`
    returns ``required=True`` (§LC, PI 2026-08-16) — i.e. when the ROUTE (or
    the main road, with no route set) demands a lane the ego is not in. It is
    NEVER emitted from observed lateral displacement. With no lane context
    supplied — the state of every clip in the corpus today — the requirement
    is UNKNOWN and the action falls through to the route's own token.
    """
    ea = _view(engine_a)
    sym = vlm_symbols or {}
    route = ea.get("route") or {}
    sp = ea.get("speed_profile") or {}
    evs = ea.get("speed_events") or []
    valid = bool(route.get("token_valid"))
    tok_r = route.get("token")
    began_stopped = (sp.get("v_t0_ms") is not None
                     and float(sp["v_t0_ms"]) < START_V_MS)

    if not (engine_a and route):
        # Engine A absent: first mappable VLM verb, tagged vlm-fused (the B4
        # ACTION_VERBS are the a_str vocabulary lowercased — no shredding)
        lc_req0 = lane_change_requirement(engine_a, lane_context)
        for act in (sym.get("actions") or []):
            verb = str(act.get("verb") or "").upper()
            # §LC: a VLM verb is not a route-serving determination either.
            if verb == "PREPARE_LANE_CHANGE" and lc_req0["required"] is not True:
                continue
            if verb in STRATEGIC_ACTION_TOKENS:
                d = str(act.get("direction") or "").lower()
                a, m = _args(arg0={"left": 1.0, "right": -1.0}.get(d)) \
                    if verb in ("PREPARE_LANE_CHANGE", "PREPARE_EXIT") \
                    else _args()
                out = _tok(verb, a, m, "vlm-fused", ["vlm.actions"], 0.5,
                           corroboration={"vlm_verbs": [], "agrees": True,
                                          "note": "engine_a absent — "
                                                  "vlm-primary fallback"})
                out["token_id"] = STRATEGIC_ACTION_TOKENS.index(verb)
                return out
        out = _tok("HOLD_CORRIDOR", *_args(), "vlm-fused",
                   ["default:no_source"], 0.4,
                   corroboration={"vlm_verbs": [], "agrees": None,
                                  "note": "engine_a absent, no vlm verbs"})
        out["token_id"] = STRATEGIC_ACTION_TOKENS.index("HOLD_CORRIDOR")
        return out

    token, args, mask, sources, conf = None, *_args(), [], 0.6

    launch = _first_event(evs, _LAUNCH_EVENT_TOKENS)
    stop_ahead = _first_event(evs, _STOP_EVENT_TOKENS,
                              min_t_start=0.0 if began_stopped else None)
    # §LC: the REQUIREMENT decides; the OBSERVATION only gets recorded.
    lc_req = lane_change_requirement(engine_a, lane_context)
    lc_ev = _gated_lc_event(ea)
    if began_stopped and launch is not None:
        token = "RESUME_CRUISE"
        args, mask = _args(arg0=sp.get("v_max_future_ms"))
        sources, conf = ["engine_a.lonmode=launch"], 0.85
    elif began_stopped and stop_ahead is None and launch is None:
        # stopped at t0 and stays stopped — the strategic action is to hold
        # the stop; PREPARE_STOP@0 m is the honest 6-token encoding of it
        token = "PREPARE_STOP"
        args, mask = _args(within_m=0.0)
        sources, conf = ["engine_a.lonmode=hold_stop"], 0.7
    elif stop_ahead is not None:
        token = "PREPARE_STOP"
        args, mask = _args(within_m=(stop_ahead.get("arc_from_t0_m")
                                     if stop_ahead.get("arc_from_t0_m")
                                     is not None
                                     else stop_ahead.get("stop_dist_m")))
        sources = [f"engine_a.lonmode={stop_ahead.get('token')}"]
        conf = 0.85
    elif valid and tok_r in ("exit_left", "exit_right"):
        token = "PREPARE_EXIT"
        args, mask = _args(arg0=(1.0 if tok_r == "exit_left" else -1.0),
                           within_m=route.get("dist_m"))
        sources, conf = [f"engine_a.route_v3={tok_r}"], 0.85
    elif lc_req["required"] is True:
        # ⭐ §LC: the ROUTE demands a lane the ego is not in. The direction and
        # the deadline come from the ROUTE and the LANE CONTEXT — never from
        # the observed lateral displacement.
        token = "PREPARE_LANE_CHANGE"
        args, mask = _args(arg0=lc_req["direction"],
                           within_m=route.get("dist_m"))
        sources = [f"lane_context.route_serving:{lc_req['basis']}"]
        conf = 0.8
    elif (sp.get("net_dv_ms") is not None
            and float(sp["net_dv_ms"]) <= REDUCE_NET_DV_MS):
        token = "REDUCE_TO"
        coast = _first_event(evs, _COAST_EVENT_TOKENS)
        args, mask = _args(arg0=sp.get("v_min_future_ms"),
                           within_m=(coast or {}).get("arc_from_t0_m"))
        sources, conf = ["engine_a.speed_profile.net_dv"], 0.75
    else:
        token = "HOLD_CORRIDOR"
        arc = route.get("arc_m")
        args, mask = _args(at_arc_m=arc if arc else None)
        sources = ["engine_a.route_v3" if valid else "default:hold"]
        conf = 0.7 if valid else 0.6

    # ---- VLM verbs: corroboration through the geometric checkers -----------
    verbs = []
    agrees = None
    for act in (sym.get("actions") or []):
        verb = str(act.get("verb") or "").lower()
        row = {"verb": verb, "direction": act.get("direction"),
               "geometry": ("no_engine_a" if not engine_a else
                            ("ok" if not check_action_geometry(act, ea)
                             else "dispute"))}
        verbs.append(row)
        if verb == token.lower():
            agrees = True
    if agrees is None and verbs:
        agrees = False
    corr = {"vlm_verbs": verbs, "agrees": agrees,
            "lane_change_requirement": lc_req}
    # §LC: observed-but-unrequired is a REAL category and stays visible.
    if lc_ev is not None:
        corr["observed_lane_change"] = {
            "token": lc_ev["token"], "lat_m": lc_ev.get("lat_m"),
            "arc_from_t0_m": lc_ev.get("arc_from_t0_m"),
            "is_label_source": False,
            "note": ("observed lateral displacement — CORROBORATION ONLY "
                     "(§LC, PI 2026-08-16). It neither licenses nor refutes "
                     "PREPARE_LANE_CHANGE: a driver may reposition for "
                     "reasons outside the route, or fail to reposition when "
                     "the route requires it.")}

    out = _tok(token, args, mask, "path", sources, conf, corroboration=corr)
    if lc_req["required"] is None and (lc_ev is not None
                                       or token == "PREPARE_LANE_CHANGE"):
        out["reason"] = lc_req["reason"]
    out["token_id"] = STRATEGIC_ACTION_TOKENS.index(token)
    return out


# --------------------------------------------------------------------------- #
# the geometric action checker — fuser-side home of                             #
# ph0_pilot._check_action_geometry (ph0_pilot.py:484), adapted to the           #
# engine_a_summary shape; previously computed and thrown away in production     #
# --------------------------------------------------------------------------- #
_LC_TOKENS = {"left": ("lc_left",), "right": ("lc_right",)}
_EXIT_TOKENS = {"left": ("exit_left", "turn_left"),
                "right": ("exit_right", "turn_right")}
_JUNCTION_TOKENS = ("turn_left", "turn_right", "exit_left", "exit_right",
                    "roundabout", "u_turn")


def check_action_geometry(action: dict, engine_a: dict) -> list[str]:
    """Dispute reasons for one VLM strategic action vs Engine A (empty=ok)."""
    ea = _view(engine_a)
    verb = str(action.get("verb", "")).lower()
    direction = str(action.get("direction", "")).lower() or None
    reasons: list[str] = []
    lc = ea.get("lane_change_events") or []
    lon = ea.get("speed_events") or []
    route = ea.get("route") or {}
    sp = ea.get("speed_profile") or {}

    if verb == "prepare_lane_change":
        # ⚠️ §LC (PI 2026-08-16): this is a HINDSIGHT-AGREEMENT note, NOT a
        # refutation. Absence of an observed lane change does not refute a
        # route-serving requirement — the driver may simply not have
        # repositioned. It is recorded so the disagreement stays visible.
        want = _LC_TOKENS.get(direction, ("lc_left", "lc_right"))
        if not any(e.get("token") in want for e in lc):
            reasons.append(f"no {direction or 'any'}-lane-change event in "
                           "hindsight path (not observed; does not refute a "
                           "route-serving requirement)")
    elif verb == "prepare_exit":
        want = _EXIT_TOKENS.get(direction,
                                ("exit_left", "exit_right",
                                 "turn_left", "turn_right"))
        if not (route.get("token") in want and route.get("token_valid")):
            reasons.append(f"route token {route.get('token')!r} does not "
                           f"confirm exit {direction or ''}".strip())
    elif verb == "prepare_stop":
        stop_ev = any(e.get("token") in _STOP_EVENT_TOKENS for e in lon)
        if not stop_ev and not sp.get("stops"):
            reasons.append("no stop event in hindsight speed profile")
    elif verb == "reduce_to":
        decel = (any(e.get("token") in ("stop_at_point", "coast", "creep")
                     for e in lon)
                 or float(sp.get("net_dv_ms") or 0.0) <= -1.0)
        if not decel:
            reasons.append("no deceleration in hindsight speed profile")
    elif verb in ("hold_corridor", "resume_cruise"):
        if route.get("token_valid") and route.get("token") in _JUNCTION_TOKENS:
            reasons.append(f"junction-scale route event "
                           f"{route.get('token')!r} inside the hold/cruise")
        if verb == "resume_cruise" and any(
                e.get("token") in _STOP_EVENT_TOKENS for e in lon):
            reasons.append("stop event ahead contradicts resume_cruise")
    else:
        reasons.append(f"verb {verb!r} not in the strategic action vocabulary")
    return reasons
