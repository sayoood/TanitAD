#!/usr/bin/env python3
"""PH1 fusion — one aligned record per clip from ego + VLM + SAM3 (+ Alpamayo).

Strategy doc: `…/2026-08-07-hierarchical-wm-redesign/PH1_FUSION_STRATEGY.md`.
The one-line version: **jurisdiction, not averaging** — SAM3 owns pixels, the
VLM owns symbols, ego owns the metric spine, Alpamayo is an external second
opinion; every cross-source relation is a recorded corroboration or a recorded
conflict, never a silent merge.

Binding rules enforced structurally here (not by convention):
  * pixels only from SAM3 — there is no code path that promotes a VLM box to
    geometry (B3 measured 2/23 same-frame agreement → diagnostic-only);
  * every field carries provenance; `inference_admissible` whitelists the
    vision-only fields (labels may use ego; INFERENCE IS VISION-ONLY);
  * the goal/vocab fields never contain situation-classifier output — the
    goal/situation information-disjointness rule survives fusion (asserted);
  * sign OCR text carried but `pending_g1_gate` until the PI grades it;
  * vocabulary tokens are IMPORTED from `tanitad.models.v6` — the emitter and
    the consumer cannot drift.

Usage:
  PYTHONPATH=<stack> python3 scripts/ph1_fuse.py \
      --v2-json <v2/ph0_v2.json> --sam3 <sam3 dir-or-json> \
      --ego-root <ego/> [--records <records.parquet>] \
      [--missing-sam3-ok REASON] --out <fused/>

A clip present in --v2-json but absent from the SAM3 output is a NAMED
PARTIAL, never a silent zero: without --missing-sam3-ok the run refuses
loudly; with it, the clip is fused from its other layers and its perception
layer carries {"absent": REASON}, and every SAM3-dependent corroboration
degrades to not_computable instead of fabricating a verdict from an empty
detector. (The 600-clip val run predates this and silently fused 4 such
clips with an empty perception layer — MEASURED n_v2=600 vs n_sam3=596.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import s2_derive  # noqa: E402  — the ONE home of the S2 strategic mapping
from s2_derive import GOAL_TO_GSTR  # noqa: E402  (moved there; re-exported)
from tanitad.models.v6 import (STRATEGIC_GOAL_TOKENS,  # noqa: E402
                               TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS)

# the pins in s2_derive must equal the real v6 lists — loudly, at import
assert s2_derive.check_vocab_drift() == "checked"

SCHEMA = "ph1-fused-v1"
IOU_TRACK = 0.3          # greedy same-concept association across frames
SPEED_TOL = 0.15         # speed-sign corroboration margin
STOP_V = 0.5             # m/s — "stopped" threshold

# =========================================================================== #
# ⛔ `goal_evidence: grounded` IS RETIRED (2026-08-16) — THE NAME ASSERTED     #
#    FAR MORE THAN THE PREDICATE MEASURED                                     #
# =========================================================================== #
# What it used to do: emit `"grounded"` for a `route_to` goal whenever the VLM
# had cited SOME sign index AND SAM3 had ≥1 `traffic sign` track ANYWHERE in
# the clip. The predicate is KIND-blind, FRAME-blind, THRESHOLD-blind and
# applies-to-ego-blind, so `grounded` was reachable by "a sign-like object
# exists in this clip" — while the word claims the NAVIGATION sign the VLM says
# it read has been confirmed.
#
# ⛔ THE INPUTS ARE FINE; THE NAME IS NOT. This is not a detector-quality
# problem and no score threshold repairs it (MEASURED, `…/incoming/
# 2026-08-16-sam3-concept-reliability/SAM3_CONCEPT_RELIABILITY.md`):
#
#   * SAM3 `traffic sign` precision is **0.880 [0.795, 0.958]**, n=64 over 33
#     clips, episode-cluster bootstrap — i.e. ~88 % REAL SIGNS. The predicate's
#     inputs were never the defect.
#   * The dominant false-positive mode is a SIGN-SHAPED NON-SIGN — a pharmacy
#     cross at **0.807** (the HIGHEST-scoring of the six sign FPs), an
#     advertising hoarding, a green traffic light. ⇒ **no score cut separates
#     them**; a KIND check would, and there is none.
#   * The sign TEXT was never read: the G1 gate is **CLOSED at 0/31**
#     (`Project Steering/G1_RESULT.md`) — the claim `grounded` made is about a
#     text nobody can verify at 448 px.
#   * The cited sign is not even a NAV sign on **24/31** aug120 `route_to`
#     clips (speed 15 · other 6 · yield 2 · stop 1 · nav 7 — MEASURED,
#     `…/2026-08-16-s2-strategic-gap/raw/aug120_analysis.json`
#     `route_to_sign_kinds`). A give-way triangle was grounding `route_to`.
#
# ⇒ THE FIX, and why it is a RETIREMENT rather than a rename in place:
#   1. the VERDICT collapses to `not_computable` with the reason NAMED — the
#      same rule the fuser already applies to `scene_vs_situations` and to an
#      absent SAM3 leg: an instrument that cannot answer must not emit a
#      verdict, in EITHER direction. (`PI_REVIEW_FINDINGS.md` removed
#      `LANE_TARGET` for the mirror-image defect — an absent measurement
#      rendered as a confident NEGATIVE; `grounded` was a weak measurement
#      rendered as a confident POSITIVE.)
#   2. the one thing SAM3 really measures SURVIVES, under its own name:
#      `sign_like_object_present` (+ the raw `sam3_sign_tracks` count). Nothing
#      is lost — a consumer that wants sign PRESENCE still has it, and can no
#      longer mistake it for goal corroboration.
#   3. `provisional` goes too. It read as "SAM3 looked and found no sign", but
#      this study measures **NO RECALL for any concept** — zero sign tracks is
#      not evidence of no sign. It was the same overstatement with the sign
#      flipped.
#
# ⚠️ WHAT WOULD MAKE THIS COMPUTABLE AGAIN (a refusal with no mechanism teaches
# nobody): a per-detection KIND check on the cited sign, on the FRAME the VLM
# grounded on, at the ≥0.70 operating point the study recommends for
# per-detection supervision — plus an open G1 text gate. Until all of those
# exist, this check is a RECORD OF A GAP, not a corroboration.
#: ⛔ Tokens `goal_evidence` may never emit again. Pinned by
#: `tests/test_ph1_fuse.py::test_the_retired_goal_evidence_tokens_are_GONE`.
GOAL_EVIDENCE_RETIRED = ("grounded", "provisional")
#: The reason stamped into every `route_to` record whose SAM3 leg DID run.
GOAL_EVIDENCE_UNVERIFIABLE = (
    "route_to evidence is NOT verifiable: SAM3's `traffic sign` class is "
    "KIND-blind, FRAME-blind and applies-to-ego-blind (precision 0.880 "
    "[0.795, 0.958], but the dominant FP mode is sign-SHAPED non-signs that no "
    "score threshold separates), and the sign-TEXT gate is CLOSED at 0/31 "
    "(G1_RESULT.md). `sign_like_object_present` is the only measured fact here")

# =========================================================================== #
# ⛔ THE TACTICAL MAPPING — AN EXPLICIT TOTAL FUNCTION, NOT SUBSTRING MATCHING #
# =========================================================================== #
# What this replaces: two ordered tuples of (substring, token) scanned with
# `sub in text.lower()`. THE MECHANISM produced three defects at once
# (`…/incoming/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md` §1.3,
# MEASURED over 270 action emissions on the 201-clip aug120 cohort):
#
#   1. ⛔ `reduce_to` — the VLM's ONLY deceleration verb — matched NOTHING on
#      either axis: 49/270 = 18.1 % of emissions SILENTLY DROPPED. A verb
#      mapping to nothing is indistinguishable from a verb never spoken. That
#      is the C77 family (an absence rendered as a result), in a mapping.
#   2. ⛔ `hold_corridor` is a LATERAL verb, and it matched the LON_RULES
#      substring `"hold"` → the LONGITUDINAL token `HOLD` on 159/270 = 58.9 %
#      of emissions. (1)+(2) made the VLM's tactical longitudinal output a
#      CONSTANT — Cohen's κ **exactly 0.0000** against BOTH other legs, n=162.
#   3. ⛔ The Alpamayo leg ran the same substrings over
#      `json.dumps(meta_action)[:400]` — a blob that CONTAINS THE FREE-TEXT
#      `cot` RATIONALE. A reason reading *"Stop for the red light"* cast a
#      longitudinal vote regardless of what the Longitudinal AXIS said, and
#      the axis value itself could fall outside the 400-char truncation.
#
# ⇒ ROOT CAUSE, fixed at the mechanism: substring matching cannot tell a verb
# from a verb that CONTAINS it, nor a token from a token MENTIONED IN PROSE.
# Both source vocabularies are CLOSED, so each mapping below is a TOTAL dict
# over its closed key set and the lookups RAISE on an unknown key. A new verb
# is then a loud failure; it can never again be a silent `None`.

#: `None` as a VALUE in these tables means **the source makes no claim on this
#: axis** — a DECLARED entry, which is precisely what the old silent
#: fall-through was not. Totality is pinned by `tests/test_ph1_fuse.py`.
NO_CLAIM = None
#: the lateral token depends on the emitted `direction`, not the verb alone
BY_DIRECTION = "__by_direction__"
#: the axis SPOKE, but v6's longitudinal vocabulary is REASON-typed while this
#: source's axis is MAGNITUDE-typed — several tokens are compatible and the
#: REASON decides. ⚠️ Distinct from `NO_CLAIM` (the source said nothing): the
#: record must be able to tell "silent" from "spoke, but untypeable".
REASON_REQUIRED = "__reason_required__"
#: the source spoke and v6 HAS NO TOKEN for what it said. Reported, never
#: bent into a neighbour — `NUDGE_L` for a left TURN would be a fabrication.
NO_V6_TOKEN = "__no_v6_token__"

_SENTINELS = (BY_DIRECTION, REASON_REQUIRED, NO_V6_TOKEN)


class UnmappedActionVerb(KeyError):
    """A key outside the closed source vocabulary reached a tactical mapping.

    Raised, never returned as `None` — the whole point of the rewrite. The
    fuser catches it and records a NAMED CONFLICT (see `emit_vocab`), exactly
    as `goal_kind_unmapped` already did, so one rogue verb cannot destroy a
    4,729-clip fuse while still being impossible to miss.
    """


#: VLM action verb → (`a_tac` LATERAL, `a_tac` LONGITUDINAL).
#: ⚠️ The keys are `ph0_v2.ACTION_VERBS`, which are `v6.STRATEGIC_ACTION_TOKENS`
#: lowercased — **the VLM speaks at the STRATEGIC layer**, so this table is an
#: explicit CROSS-LAYER projection down to `a_tac`. The substring rules did
#: that projection by accident and nobody could see they had.
#: ⭐ The VLM's verbs are REASON-typed (`prepare_stop`, `resume_cruise`), which
#: is the same type as `TACTICAL_LON_ACTIONS` — that is why this leg maps at
#: all and the magnitude-typed Alpamayo axis below does not.
VLM_VERB_TO_A_TAC: dict[str, tuple[str | None, str | None]] = {
    # a LATERAL verb. ⛔ IT MAKES NO LONGITUDINAL CLAIM — defect (2), fixed.
    "hold_corridor":       ("LANE_KEEP",  NO_CLAIM),
    "prepare_lane_change": (BY_DIRECTION, NO_CLAIM),
    # ⛔ defect (1), fixed: the VLM's only deceleration verb now lands.
    # ⚠️ RECORDED COLLAPSE: `TACTICAL_LON_ACTIONS` carries ONE deceleration
    # token, so `reduce_to` and `prepare_stop` both land on `BRAKE_TO` and are
    # no longer distinguishable downstream. REPORTED, not fixed by editing the
    # vocabulary — the tuples size embedding tables (`v6.py:3297-3298`) and a
    # shape change breaks the live 30k v6F strict resume.
    "reduce_to":           (NO_CLAIM,     "BRAKE_TO"),
    "prepare_stop":        (NO_CLAIM,     "BRAKE_TO"),
    "resume_cruise":       (NO_CLAIM,     "CRUISE"),
    # ⚠️ an exit needs LANE CONTEXT (which lane serves the exit) before it can
    # say a lateral action is required — the same gap the §LC ruling names,
    # and `lane_context` is None on every clip today. Emitting LANE_CHANGE_R
    # here would swap an unidentifiable label for a confidently wrong one, so
    # BOTH axes abstain, BY DECLARATION rather than by falling off the end.
    "prepare_exit":        (NO_CLAIM,     NO_CLAIM),
}
#: `ph0_v2.py:272-274` already rejects a sideless lane change, so `"none"` is
#: a malformed record here: it abstains and never picks a side.
_LC_DIRECTION_TO_LAT = {"left": "LANE_CHANGE_L", "right": "LANE_CHANGE_R",
                        "none": NO_CLAIM}

#: Alpamayo declares THREE INDEPENDENT AXES on labelled lines of one
#: generation. Parsed, never substring-sniffed out of the serialised blob.
ALPAMAYO_AXES = ("Longitudinal", "Lateral", "Lane")

#: Alpamayo LANE axis → `a_tac` LATERAL. Total over the 7 observed values
#: (MEASURED n=4,729, `…/2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy.json`).
ALPAMAYO_LANE_TO_A_TAC_LAT: dict[str, str | None] = {
    "lane keep":            "LANE_KEEP",
    "left lane change":     "LANE_CHANGE_L",
    "right lane change":    "LANE_CHANGE_R",
    "slightly shift left":  "NUDGE_L",
    "slightly shift right": "NUDGE_R",
    # ⛔ 186 of 4,729 clips (3.94 %) are UNREPRESENTABLE: `TACTICAL_LAT_ACTIONS`
    # has no TURN_* member. Declared, reported, never coerced.
    "turn left":            NO_V6_TOKEN,
    "turn right":           NO_V6_TOKEN,
}

#: Alpamayo LONGITUDINAL axis → `a_tac` LONGITUDINAL. ⛔ **IT DOES NOT MAP**,
#: and that is a TYPE fact, not a coverage gap (`TACTICAL_LABEL_VALIDATION.md`
#: §4.2): `TACTICAL_LON_ACTIONS` is REASON-typed (`FOLLOW`, `YIELD_MERGE`,
#: `BRAKE_TO`, `CREEP`, `HOLD`, `CRUISE`) while this axis is MAGNITUDE-typed
#: (`Gentle Deceleration`, `Strong Acceleration`). *"Gentle Deceleration"*
#: cannot say whether the ego is FOLLOWing a lead or BRAKE_TO a stop line —
#: **the reason decides, and the reason lives in `cot`**, which this table
#: deliberately cannot see. Every value is therefore `REASON_REQUIRED`: the
#: leg casts NO longitudinal vote, and the axis value + reason are RECORDED
#: for the (separately escalated) meta-action × reason label builder.
#: ⚠️ The old substring path DID cast a vote here, from a lottery over the
#: truncated `cot` text. Removing it is the honest correction, not a regression.
ALPAMAYO_LON_TO_A_TAC_LON: dict[str, str | None] = {
    "gentle deceleration": REASON_REQUIRED,
    "maintain speed":      REASON_REQUIRED,
    "gentle acceleration": REASON_REQUIRED,
    "stop":                REASON_REQUIRED,
    "strong deceleration": REASON_REQUIRED,
    "strong acceleration": REASON_REQUIRED,
    "reverse":             NO_V6_TOKEN,
}


#: Why `g_tac_lat`/`g_tac_lon` are emitted EMPTY rather than filled or dropped.
#: ⚠️ Must not contain the word this file asserts out of the vocab block.
_G_TAC_GAP = (
    "goal-token axes are REASON-typed (TACTICAL_GOAL_TOKENS_LAT/LON, "
    "v6.py:217-223) and NOTHING in this fuse derives them — they need the "
    "Alpamayo reason field (reachability 82.77 % of 4,729, correctness "
    "UNMEASURED). The ACTION tokens live in a_tac_lat/a_tac_lon; this key "
    "used to carry them under a goal's name and any consumer joining it to a "
    "goal head silently received actions. See TACTICAL_LABEL_VALIDATION.md "
    "§1.3/§4.3.")


def parse_alpamayo_axes(raw: object) -> dict:
    """Split one Alpamayo `meta_action` generation into its labelled axes.

    Mirrors `…/2026-08-16-tactical-labels/code/tac_a1_alpamayo_taxonomy.py`
    (itself mirroring `a2_parse_meta_action.py:53`) so the readings of this
    field cannot drift. Accepts the raw JSON string the fuser holds, an
    already-parsed dict, or the bare generation text.

    ⛔ The axes are READ FROM THEIR OWN LINES. The defect this replaces
    matched substrings against the whole serialised blob, so the free-text
    `cot` could out-vote the axis it was supposed to explain.
    """
    import re

    txt = raw
    if isinstance(raw, (bytes, bytearray)):
        txt = raw.decode("utf-8", "replace")
    if isinstance(txt, str):
        try:
            txt = json.loads(txt)
        except Exception:                                        # noqa: BLE001
            pass
    cot = None
    if isinstance(txt, dict):
        def _one(k):
            v = txt.get(k)
            return (v[0] if v else None) if isinstance(v, list) else v
        cot = _one("cot")
        txt = _one("raw_outputs") or _one("meta_action") or ""
    txt = txt if isinstance(txt, str) else str(txt or "")
    out: dict = {"cot": cot, "_raw_len": len(txt)}
    for axis in ALPAMAYO_AXES:
        m = re.search(rf"{axis}:\s*([^.\n<]+)", txt)
        out[axis.lower()] = m.group(1).strip() if m else None
    return out


def map_vlm_action(verb: object, direction: object) -> tuple:
    """(verb, direction) → (lat, lon, note). RAISES on an unknown verb.

    `note` is ``None`` when both axes resolved to a real token or to a
    DECLARED no-claim; otherwise it names why an axis is empty, so the record
    can distinguish "made no claim" from "could not be typed".
    """
    key = str(verb or "").strip().lower()
    if key not in VLM_VERB_TO_A_TAC:
        raise UnmappedActionVerb(
            f"VLM action verb {verb!r} is not in VLM_VERB_TO_A_TAC "
            f"(known: {sorted(VLM_VERB_TO_A_TAC)}). Add it explicitly — a "
            "verb must never map to nothing by falling off the end.")
    lat, lon = VLM_VERB_TO_A_TAC[key]
    note = None
    if lat == BY_DIRECTION:
        d = str(direction or "none").strip().lower()
        if d not in _LC_DIRECTION_TO_LAT:
            raise UnmappedActionVerb(
                f"lane-change direction {direction!r} is outside "
                f"{sorted(_LC_DIRECTION_TO_LAT)} for verb {key!r}")
        lat = _LC_DIRECTION_TO_LAT[d]
        if lat is NO_CLAIM:
            note = "lane change emitted without a side — abstained"
    return lat, lon, note


def map_alpamayo_axes(axes: dict) -> tuple:
    """Parsed Alpamayo axes → (a_tac lat, a_tac lon, notes). RAISES on an
    unknown axis value, so a taxonomy change is loud instead of silent."""
    notes: dict[str, str] = {}
    out: list[str | None] = []
    for name, table, field in (("lat", ALPAMAYO_LANE_TO_A_TAC_LAT, "lane"),
                               ("lon", ALPAMAYO_LON_TO_A_TAC_LON,
                                "longitudinal")):
        val = axes.get(field)
        if not val:
            out.append(NO_CLAIM)
            # ⚠️ NOT a parse failure: a STOPPED vehicle emits one axis only,
            # and the 304 null lateral/lane rows are exactly the 304 `Stop`
            # rows (MEASURED, n=4,729). Not-applicable, never imputed.
            notes[name] = "axis absent in the generation"
            continue
        k = str(val).strip().lower()
        if k not in table:
            raise UnmappedActionVerb(
                f"Alpamayo {field} value {val!r} is outside the observed "
                f"taxonomy {sorted(table)} — a NEW value is a FINDING about "
                "the taxonomy, not something to coerce into a neighbour.")
        tok = table[k]
        if tok == REASON_REQUIRED:
            out.append(NO_CLAIM)
            notes[name] = ("magnitude-typed axis, reason-typed vocabulary — "
                           "the reason decides (see ALPAMAYO_LON_TO_A_TAC_LON)")
        elif tok == NO_V6_TOKEN:
            out.append(NO_CLAIM)
            notes[name] = f"no v6 token exists for {val!r}"
        else:
            out.append(tok)
    return out[0], out[1], notes


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def build_tracks(frames: dict) -> list[dict]:
    """Greedy same-concept IoU association across SAM3's processed frames.

    Dynamics are ORDINAL on purpose: box-area trend + image-x drift classify
    approaching/receding/crossing. No metric depth is invented — the strategy
    doc's §7 limit, kept honest here.
    """
    idxs = sorted(int(k) for k in frames)
    open_tracks: list[dict] = []
    done: list[dict] = []
    for fi in idxs:
        dets = list((frames[str(fi)] or {}).get("det") or [])
        used = set()
        for tr in list(open_tracks):
            best, bj = 0.0, None
            for j, d in enumerate(dets):
                if j in used or d.get("concept") != tr["concept"]:
                    continue
                v = _iou(tr["boxes"][-1][1], d["box_xyxy"])
                if v > best:
                    best, bj = v, j
            if bj is not None and best >= IOU_TRACK:
                d = dets[bj]
                used.add(bj)
                tr["boxes"].append((fi, d["box_xyxy"]))
                tr["scores"].append(float(d.get("score", 0)))
                tr["areas"].append(float(d.get("mask_area_px") or 0))
            else:
                done.append(open_tracks.pop(open_tracks.index(tr)))
        for j, d in enumerate(dets):
            if j not in used:
                open_tracks.append({
                    "concept": d.get("concept"),
                    "boxes": [(fi, d["box_xyxy"])],
                    "scores": [float(d.get("score", 0))],
                    "areas": [float(d.get("mask_area_px") or 0)],
                })
    done.extend(open_tracks)
    out = []
    for k, tr in enumerate(done):
        a0, a1 = tr["areas"][0], tr["areas"][-1]
        x0 = (tr["boxes"][0][1][0] + tr["boxes"][0][1][2]) / 2
        x1 = (tr["boxes"][-1][1][0] + tr["boxes"][-1][1][2]) / 2
        if len(tr["boxes"]) < 2 or a0 <= 0:
            dyn = "single_frame"
        elif a1 > 1.3 * a0:
            dyn = "approaching"
        elif a1 < 0.7 * a0:
            dyn = "receding"
        elif abs(x1 - x0) > 40:
            dyn = "crossing"
        else:
            dyn = "steady"
        out.append({"track_id": k, "concept": tr["concept"],
                    "n_frames": len(tr["boxes"]),
                    "frame_span": [tr["boxes"][0][0], tr["boxes"][-1][0]],
                    "mean_score": round(sum(tr["scores"]) / len(tr["scores"]), 3),
                    "boxes": tr["boxes"], "dynamics": dyn, "src": "sam3"})
    return out


def ego_from_npz(path: str) -> dict:
    """Engine-A style metric spine computed from the bridged ego npz.

    The v2 records carry only ego_state (the prompt view); route/speed_profile/
    situations are NOT in them (MEASURED on the 600-clip production output).
    speed_profile is recomputed here deterministically from poses[:, 3]; the
    frozen situation detectors are NOT re-implemented — situations stay absent
    unless the record carries them, and checks degrade to not_computable.
    """
    import numpy as np
    try:
        d = np.load(path)
        poses = d["poses"]
    except Exception:                                       # noqa: BLE001
        return {}
    v = poses[:, 3].astype(float)
    yaw = poses[:, 2].astype(float)
    stops = int(((v[:-1] >= STOP_V) & (v[1:] < STOP_V)).sum()
                + (1 if v[0] < STOP_V else 0))
    net_dyaw = float(yaw[-1] - yaw[0])
    return {"speed_profile": {
                "v_t0": float(v[0]), "v_min_future": float(v.min()),
                "v_max_future": float(v.max()),
                "net_dv": float(v[-1] - v[0]), "stops": stops,
                "src": "ego_npz"},
            "net_dyaw_rad": net_dyaw}


def evidence_sign_kind(v2: dict, idx) -> str | None:
    """The KIND the **VLM ITSELF** recorded for the sign it cited, or None.

    ⚠️ This is a VLM SELF-REPORT read back out of the same record — it is NOT
    corroboration and must never be presented as one. It exists so the
    "`route_to` cites a speed sign" gap (24/31 on aug120) is visible per clip.

    Returns None — never a fabricated kind — when the index is absent, is not
    an int, or points outside `signs[]`. A `route_to` record whose `signs`
    block was dropped is exactly that case and must read as unknown, not as a
    kind we invented.
    """
    if idx is None or isinstance(idx, bool) or not isinstance(idx, int):
        return None
    signs = ((v2.get("signs") or {}).get("signs") or [])
    if idx < 0 or idx >= len(signs):
        return None
    kind = (signs[idx] or {}).get("kind")
    return kind if isinstance(kind, str) else None


def corroborate(v2: dict, sam3: dict, tracks: list[dict],
                sam3_absent: bool = False) -> tuple[dict, list]:
    cor, conflicts = {}, []
    sp = v2.get("speed_profile") or {}
    v_now = (v2.get("ego_state") or {}).get("v_now_ms")
    # --- speed sign vs ego speed profile ---------------------------------- #
    for s in ((v2.get("signs") or {}).get("signs") or []):
        if s.get("kind") == "speed" and str(s.get("text", "")).isdigit():
            lim = float(s["text"])
            vmin = sp.get("v_min_future")
            row = {"sign_text": s["text"], "v_now_ms": v_now,
                   "v_min_future": vmin, "src": ["vlm", "ego"]}
            if vmin is None:
                row["verdict"] = "not_computable"
            else:
                # unit-honest: corroborated under EITHER km/h or mph reading
                ok = any(min(vmin, v_now or vmin) <= (1 + SPEED_TOL) * lim / f
                         for f in (3.6, 2.237))
                row["verdict"] = "corroborated" if ok else "conflict"
                if not ok:
                    conflicts.append({"check": "speed_sign_vs_ego", **row})
            cor["speed_sign_vs_ego"] = row
    # --- red light vs ego stop -------------------------------------------- #
    reds = [s for s in ((v2.get("signs") or {}).get("signs") or [])
            if s.get("state") == "red"]
    if reds:
        stopped = (sp.get("stops") or 0) > 0 or (
            sp.get("v_min_future") is not None
            and sp["v_min_future"] < STOP_V)
        cor["red_light_vs_stop"] = {
            "n_red": len(reds), "ego_stopped": bool(stopped),
            "verdict": "corroborated" if stopped else "conflict",
            "src": ["vlm", "ego"]}
        if not stopped:
            conflicts.append({"check": "red_light_vs_stop", "n_red": len(reds)})
    # --- scene vs situations ---------------------------------------------- #
    scene = v2.get("scene") or {}
    sit = v2.get("situations")
    claims_int = "intersection" in str(scene.get("domain", "")) or \
                 scene.get("road_type") == "junction"
    if claims_int:
        if sit is None:
            # ⚠️ absent data must not manufacture a conflict — the frozen
            # situation detectors did not run on this batch, and saying
            # "conflict" here would be a fabricated disagreement.
            cor["scene_vs_situations"] = {"verdict": "not_computable",
                                          "reason": "no situations source",
                                          "src": ["vlm"]}
        else:
            ok = bool(sit.get("intersection"))
            cor["scene_vs_situations"] = {
                "scene_claims_intersection": True,
                "ego_intersection_window": ok,
                "verdict": "corroborated" if ok else "conflict",
                "src": ["vlm", "ego"]}
            if not ok:
                conflicts.append({"check": "scene_vs_situations"})
    # --- goal evidence — `grounded` RETIRED, see the block at the top ------ #
    sym = v2.get("symbols") or {}
    if sym.get("goal_kind") == "route_to":
        ev = sym.get("goal_evidence_sign")
        if sam3_absent:
            # ⚠️ same rule as the situations fix: an absent detector must not
            # manufacture a verdict in EITHER direction — it saw nothing.
            # ⚠️ the cited KIND is a VLM-side fact and does NOT depend on the
            # SAM3 leg — record it here too, or the 24/31 non-nav gap would be
            # auditable only on the clips SAM3 happened to cover (15 of 31 on
            # aug120), which is a coverage artifact masquerading as a rate.
            cor["goal_evidence"] = {"evidence_sign_idx": ev,
                                    "evidence_sign_kind":
                                        evidence_sign_kind(v2, ev),
                                    "verdict": "not_computable",
                                    "reason": "sam3 absent for this clip",
                                    "src": ["vlm"]}
        else:
            n_sign_tracks = sum(1 for t in tracks
                                if t["concept"] == "traffic sign")
            cor["goal_evidence"] = {
                "evidence_sign_idx": ev,
                # ⚠️ VLM SELF-REPORT, never corroboration — recorded because it
                # is exactly the field a future KIND check needs, and because
                # it makes the 24/31 non-nav gap auditable PER CLIP instead of
                # only in a study.
                "evidence_sign_kind": evidence_sign_kind(v2, ev),
                "sam3_sign_tracks": n_sign_tracks,
                # ⭐ THE ONLY MEASURED FACT, UNDER ITS OWN NAME. It says a
                # sign-LIKE object was detected somewhere in this clip. It does
                # NOT say the cited sign exists, is a navigation sign, is on
                # the grounded frame, or applies to ego.
                "sign_like_object_present": n_sign_tracks > 0,
                "verdict": "not_computable",
                "reason": GOAL_EVIDENCE_UNVERIFIABLE,
                "src": ["vlm", "sam3"]}
    # --- census vs scene --------------------------------------------------- #
    if scene.get("road_type") == "urban":
        n_agents = sum(1 for t in tracks
                       if t["concept"] in ("car", "truck", "bus", "pedestrian",
                                           "cyclist"))
        # ⛔ C77: the same rule as the prose — an absent detector may not
        # produce a "flagged_empty" FINDING. Zero PROCESSED FRAMES is
        # unavailability too, not a measured empty scene.
        cen = census_state(tracks, absent_reason=(
            "sam3 absent for this clip" if sam3_absent else None),
            n_frames=len((sam3 or {}).get("frames") or {}))
        if cen["state"] == CENSUS_UNAVAILABLE:
            cor["census_vs_scene"] = {"verdict": "not_computable",
                                      "reason": cen["reason"],
                                      "src": ["vlm"]}
        elif n_agents == 0:
            cor["census_vs_scene"] = {"verdict": "flagged_empty_urban",
                                      "src": ["sam3", "vlm"]}
    return cor, conflicts


# ⚠️ ego_past_state emits turning ∈ {turning_left, turning_right, straight}
# (ph0_v2.py:400-401). The first vote map keyed on {left, right} and the ego
# lateral vote was DEAD on every turning clip — MEASURED: null on 36/36
# turning clips, LANE_KEEP on 165/165 straight (S2_STRATEGIC_GAP §6 item 6).
_EGO_TURN_VOTE = {"turning_left": "NUDGE_L", "turning_right": "NUDGE_R",
                  "left": "NUDGE_L", "right": "NUDGE_R",
                  "straight": "LANE_KEEP"}

# =========================================================================== #
# ⛔ THE TACTICAL VOTE — `ego` AND `vlm` ARE **NOT** INDEPENDENT VOTERS        #
# =========================================================================== #
# MEASURED (`TACTICAL_LABEL_VALIDATION.md` §1.4): `_ego_prompt_mode == 'past'`
# on **201/201** v2 records, and the ego block printed into the VLM's prompt
# contains `motion` and `turning` — **exactly and only** the two fields the ego
# voter below reads. The signature is unmistakable:
#
#     VLM ↔ ego        (LAT)  κ = 0.7608     ← the leg PRINTED IN ITS PROMPT
#     Alpamayo ↔ VLM   (LAT)  κ = 0.1717     ← the leg that saw the camera rig
#     Alpamayo ↔ ego   (LAT)  κ = 0.2089
#
# ⇒ **A 2-of-3 majority over {ego, vlm, alpamayo} can be carried by ONE SOURCE
# COUNTED TWICE.** That is the action-echo defect (`EVAL_DOCTRINE` §1.12) and
# the nav-echo defect wearing a labelling costume, and the old `majority()`
# helper — which counted voters, not sources — is DELETED, not reweighted.
#
# THE REPLACEMENT: voters are partitioned into INDEPENDENCE BLOCKS and the
# BLOCKS vote. `ego` and `vlm` are ONE block; it casts at most ONE vote, and
# only when its members AGREE (a disagreement inside the block is itself a
# finding — the VLM departing from its own prompt — so it is recorded, not
# averaged). With two blocks there is no "2 of 3" left to be satisfied.
#
# ⚠️ SCOPE, stated honestly: the κ above is LATERAL. The longitudinal κ was
# 0.0000 only because the VLM's LON leg was the CONSTANT the mapping bugs
# above created; **post-fix the LON dependence is UNMEASURED** and needs the
# re-fuse. The block grouping does not rest on κ in any case — it rests on the
# STRUCTURAL fact that the two voters share their input, MEASURED 201/201.
VOTER_BLOCKS = {"ego": "ego+vlm", "vlm": "ego+vlm", "alpamayo": "alpamayo"}
#: Which block decides when the blocks disagree. Alpamayo, because it is the
#: only leg that saw the FULL CAMERA RIG and the only one external to this
#: pipeline — `TACTICAL_LABEL_VALIDATION.md` §5.3 ("Source of truth: Alpamayo.
#: Ego = corroboration."). ⚠️ Alpamayo is a TEACHER (PhysicalAI-AV is listed as
#: its training data, overlap UNRESOLVED), never ground truth.
PRIMARY_BLOCK = "alpamayo"


def block_vote(votes: list[tuple[str, str | None]], valid: tuple) -> dict:
    """Decide over INDEPENDENCE BLOCKS, never over raw voters.

    Returns a block ready to embed, carrying enough provenance that a
    downstream label builder can require independent corroboration:

      ``token``                 the emitted token (or None)
      ``provenance``            which BLOCK it came from
      ``corroborated``          ⇒ **≥2 INDEPENDENT blocks spoke and agreed**
      ``n_blocks_speaking``     how many blocks cast a vote at all
      ``blocks``                per-block token / members / internal conflict

    ⛔ `corroborated` can NEVER be True from {ego, vlm} alone — they are one
    block. That is the property the old 2-of-3 majority could not offer and
    the reason this function exists.
    """
    per: dict[str, dict] = {}
    for src, tok in votes:
        blk = VOTER_BLOCKS.get(src, src)
        b = per.setdefault(blk, {"token": None, "members": [],
                                 "internal_disagreement": False,
                                 "member_tokens": []})
        b["members"].append(src)
        if tok in valid:
            b["member_tokens"].append(tok)
    for blk, b in per.items():
        toks = set(b["member_tokens"])
        if len(toks) == 1:
            b["token"] = b["member_tokens"][0]
        elif len(toks) > 1:
            # ⚠️ a REAL finding on the ego+vlm block: the VLM departed from
            # the ego numbers it was shown. Recorded, never averaged away.
            b["internal_disagreement"] = True
    speaking = {k: v for k, v in per.items() if v["token"] is not None}
    agreed = {v["token"] for v in speaking.values()}
    if not speaking:
        tok, prov = None, None
    elif PRIMARY_BLOCK in speaking:
        tok, prov = speaking[PRIMARY_BLOCK]["token"], PRIMARY_BLOCK
    else:
        prov = sorted(speaking)[0]
        tok = speaking[prov]["token"]
    return {"token": tok, "provenance": prov,
            "corroborated": len(speaking) >= 2 and len(agreed) == 1,
            "n_blocks_speaking": len(speaking),
            "blocks": {k: {kk: vv for kk, vv in v.items()
                           if kk != "member_tokens"} for k, v in per.items()},
            "votes": [[s, t] for s, t in votes],
            "independence": "ego+vlm share their input (MEASURED 201/201: "
                            "the VLM prompt carries the ego motion/turning "
                            "fields the ego voter reads) — they are ONE block"}


def emit_vocab(v2: dict, alp: dict | None,
               engine_a: dict | None = None) -> tuple[dict, list]:
    """g_str/a_str GEOMETRY-PRIMARY via s2_derive (VLM demoted to recorded
    corroboration; ROUTE_TO gated); factored g_tac by 2-of-3 votes.

    With ``engine_a`` absent (legacy inputs) the strategic tokens fall back
    to VLM-primary, tagged ``vlm-fused`` — stated, never silent. The old
    ``corroborated_by_route`` field (structurally dead: it read a key the
    production records never carry, False on 801/801) is DELETED — the g_str
    block's ``sources``/``corroboration`` supersede it.
    """
    conflicts = []
    sym = v2.get("symbols") or {}
    if str(sym.get("goal_kind", "none")).lower() not in GOAL_TO_GSTR:
        conflicts.append({"check": "goal_kind_unmapped",
                          "value": sym.get("goal_kind")})

    # ---- strategic layer: geometry decides, the VLM corroborates ----------
    # ⛔ §LC (PI 2026-08-16): PREPARE_LANE_CHANGE needs LANE CONTEXT, not
    # observed displacement. `lane_context=None` is the HONEST state of every
    # clip today — B1's `lanes_visible`/`lane_ego` are NOT wired in on
    # purpose: the PI's own review calls the count wrong, its `conf` is
    # degenerate ("high" on 201/201) and it never uses its 0=unclear escape
    # (0/201). Wiring an unreliable signal in would replace an unidentifiable
    # label with a confidently wrong one. See LANE_CHANGE_DEEP_REVIEW.md.
    lane_context = None
    g_str = s2_derive.derive_g_str(engine_a, sym, lane_context=lane_context)
    a_str = s2_derive.derive_a_str(engine_a, sym, lane_context=lane_context)
    for blk in (g_str, a_str):
        blk["src"] = "engine_a" if blk["provenance"] == "path" else "vlm"
    assert g_str["token"] in STRATEGIC_GOAL_TOKENS

    # ---- tactical votes: OVER INDEPENDENCE BLOCKS, never 2-of-3 -----------
    ego = v2.get("ego_state") or {}
    turning = str(ego.get("turning", ""))
    lat_votes = [("ego", _EGO_TURN_VOTE.get(turning))]
    lon_ego = None
    sp = v2.get("speed_profile") or {}
    if (sp.get("stops") or 0) > 0:
        lon_ego = "BRAKE_TO"
    elif str(ego.get("motion")) == "steady":
        lon_ego = "CRUISE"
    lon_votes = [("ego", lon_ego)]
    notes: dict = {}
    for a in (sym.get("actions") or []):
        try:
            lat_t, lon_t, note = map_vlm_action(a.get("verb"),
                                                a.get("direction"))
        except UnmappedActionVerb as e:
            # the `goal_kind_unmapped` precedent: a NAMED, COUNTED conflict.
            # The mapping itself refuses to guess; the fuse survives one bad
            # verb instead of losing a 4,729-clip run to it.
            conflicts.append({"check": "action_verb_unmapped",
                              "verb": a.get("verb"),
                              "direction": a.get("direction"),
                              "detail": str(e)})
            continue
        if note:
            notes.setdefault("vlm", []).append(note)
        lat_votes.append(("vlm", lat_t))
        lon_votes.append(("vlm", lon_t))
    if alp and alp.get("meta_action"):
        axes = parse_alpamayo_axes(alp["meta_action"])
        try:
            alp_lat, alp_lon, alp_notes = map_alpamayo_axes(axes)
        except UnmappedActionVerb as e:
            conflicts.append({"check": "alpamayo_axis_unmapped",
                              "detail": str(e)})
        else:
            if alp_notes:
                notes["alpamayo"] = alp_notes
            lat_votes.append(("alpamayo", alp_lat))
            lon_votes.append(("alpamayo", alp_lon))
            notes["alpamayo_axes"] = {k: axes.get(k) for k in
                                      ("longitudinal", "lateral", "lane")}

    a_tac_lat = block_vote(lat_votes, TACTICAL_LAT_ACTIONS)
    a_tac_lon = block_vote(lon_votes, TACTICAL_LON_ACTIONS)
    if notes:
        a_tac_lat["notes"] = notes
        a_tac_lon["notes"] = notes
    # ⛔ THE FIELD NAMED WHAT IT IS NOT. The block below used to be emitted as
    # `g_tac_lat`/`g_tac_lon` while being filled from `TACTICAL_LAT_ACTIONS` /
    # `TACTICAL_LON_ACTIONS` — the **`a_tac` ACTION** vocabulary. The two are
    # DISJOINT: `LANE_KEEP` is not a member of `TACTICAL_GOAL_TOKENS_LAT`
    # (`ANCHOR_GOAL`, `CORRIDOR_OFFSET`, `EVADE_IN_CORRIDOR`,
    # `LAT_UNCONSTRAINED`). Any consumer joining `g_tac_lat` to a goal head
    # silently received ACTION tokens — the S2 `"no agents"` class again, *a
    # shape that reads like the thing it is not*. `test_ph1_fuse.py:70-71`
    # PINNED the mismatch; that pin is inverted now.
    vocab = {"g_str": g_str, "a_str": a_str,
             "a_tac_lat": a_tac_lat, "a_tac_lon": a_tac_lon,
             # ⚠️ NOT DROPPED, DECLARED. The goal-token axes are REASON-typed
             # and NOTHING in this fuse derives them (they need the Alpamayo
             # `cot`; reachability 82.77 %, correctness UNMEASURED — see
             # TACTICAL_LABEL_VALIDATION.md §4.3). Emitting the key with a
             # null token and a reason is loud; omitting it would send the
             # next consumer looking for the action field by mistake.
             "g_tac_lat": {"token": None, "unavailable_reason": _G_TAC_GAP},
             "g_tac_lon": {"token": None, "unavailable_reason": _G_TAC_GAP}}
    # ⛔ the disjointness rule: no situation-classifier output inside vocab
    assert "situation" not in json.dumps(vocab).lower()
    return vocab, conflicts


# --------------------------------------------------------------------------- #
# ⛔ C77 FAMILY — AN ABSENT MEASUREMENT MUST NEVER RENDER AS A CONFIDENT        #
# NEGATIVE. `"no agents"` and `"perception unavailable"` are DIFFERENT CLAIMS.  #
# --------------------------------------------------------------------------- #
# The defect this replaces (PI review 2026-08-16, `03ba450b`: *"not agent said
# by the pipeline. In the picture of frame 9 there is clear a car as incoming
# traffic"*) was a one-token `or` fallback:
#     ", ".join(...) or "no agents"
# An EMPTY census is falsy, so absence of evidence was rendered as evidence of
# absence. MEASURED on the fused aug120 corpus: `"no agents"` appeared on
# 119/201 records, and 115 of those 119 (96.6 %) ALREADY carried
# `perception.absent` in the structured layer — the record knew, and the prose
# said otherwise.
#: The three states a census can be in. `UNAVAILABLE` may never be phrased as
#: a quantity, and a measured zero is phrased as a COUNT ("0 agents
#: detected" — a fact about the detector), never as a claim about the world.
CENSUS_MEASURED = "measured"
CENSUS_UNAVAILABLE = "unavailable"

# --------------------------------------------------------------------------- #
# ⚠️ AND THE MIRROR-IMAGE DEFECT, FOUND BY THE RE-FUSE THAT FIXED `"no agents"` #
# --------------------------------------------------------------------------- #
# `"no agents"` rendered an ABSENT measurement as a confident NEGATIVE. Filling
# the perception layer in fixes that — and immediately creates its opposite:
# `census_phrase` renders `{"car": 73}` as **"73 car"**, which reads as
# SEVENTY-THREE CARS. It is not. It is 73 SAM3 TRACKS, and MEASURED on this
# corpus a track is very nearly a single detection:
#
#   * single-frame tracks: **7 077 / 8 067 = 87.7 %** on the v2 leg (floor 0.25)
#     and **2 049 / 2 397 = 85.5 %** on the v1 leg (floor 0.5) — the
#     fragmentation is a property of the STRIDE, not of the floor;
#   * median tracks/clip **58** (v2) against a median PEAK CONCURRENCY of
#     **20** — i.e. the count over-states the busiest single frame ~3x;
#   * mechanism: `build_tracks` associates by `IOU_TRACK = 0.3` across frames
#     that are STRIDED (6 run frames per clip). At that spacing an object's
#     boxes rarely overlap at all, so association mostly fails and each
#     detection becomes its own track.
#
# ⇒ This is the SAME CLASS as `per_scene_hits["lane marking"] = 141` not being
# 141 lane markings, and the same class as `"no agents"` itself: **a shape that
# reads like the thing it is not.** It is NOT repaired by re-tuning the IoU —
# a lower threshold on strided frames buys false merges instead of false
# splits, and neither is an object count. The honest move is to NAME THE UNIT
# and publish the fragmentation alongside it, which is what `census_state`
# emits and `census_phrase` says out loud.
#
# ⚠️ `peak_concurrent_tracks` is a LOWER BOUND on the number of distinct
# objects (the busiest processed frame), not an object count either. Both
# numbers are reported; neither is promoted to "how many cars are there".
CENSUS_UNIT_NOTE = (
    "SAM3 TRACK counts, NOT object counts: ~87 % of tracks are single-frame "
    "because build_tracks associates by IoU across STRIDED frames, so a track "
    "is ~a detection. peak_concurrent_tracks is a lower bound on distinct "
    "objects; n_agents is an upper one. Neither answers 'how many cars'")


def census_state(tracks: list[dict], *, absent_reason: str | None = None,
                 n_frames: int | None = None) -> dict:
    """Three-state agent census: MEASURED(counts) vs UNAVAILABLE(reason).

    Structurally unable to report a measured zero when perception did not
    run: the `absent_reason` / zero-frames branches return UNAVAILABLE before
    the counts are ever consulted.

    ⚠️ `n_frames=None` means **NOT SUPPLIED**, which is NOT the same as zero —
    `colab/s2_lab_lib.py:832` calls `scenario_line(r, tracks)` positionally and
    cannot pass it. Tracks are themselves proof the detector ran, so a
    non-empty census stays MEASURED; an empty one with an unknown frame count
    is honestly UNAVAILABLE, because that is exactly the case we cannot tell
    apart. (Collapsing None into 0 would have made every legacy caller report
    "unavailable" even with agents in frame — the C77 defect inverted.)
    """
    if absent_reason:
        return {"state": CENSUS_UNAVAILABLE, "reason": absent_reason,
                "counts": None, "n_agents": None}
    if n_frames == 0:
        return {"state": CENSUS_UNAVAILABLE,
                "reason": "no frames processed by the detector",
                "counts": None, "n_agents": None}
    if n_frames is None and not tracks:
        return {"state": CENSUS_UNAVAILABLE,
                "reason": "frame count not supplied and census empty — a "
                          "measured zero and an absent detector are "
                          "indistinguishable here",
                "counts": None, "n_agents": None}
    counts: dict[str, int] = {}
    per_frame: dict[int, int] = {}
    n_single = 0
    for t in tracks:
        counts[t["concept"]] = counts.get(t["concept"], 0) + 1
        n_single += int(int(t.get("n_frames") or 1) <= 1)
        for fi, _ in (t.get("boxes") or []):
            per_frame[fi] = per_frame.get(fi, 0) + 1
    return {"state": CENSUS_MEASURED, "reason": None, "counts": counts,
            "n_agents": len(tracks),
            # ⛔ THE UNIT, DECLARED — see the note below. `n_agents` is a
            # TRACK count and a track is not an object at this frame stride.
            "unit": "sam3_tracks",
            "n_single_frame_tracks": n_single,
            "peak_concurrent_tracks": max(per_frame.values(), default=0),
            "counts_are": CENSUS_UNIT_NOTE}


# =========================================================================== #
# ⛔ A DETECTION FLOOR IS INVISIBLE IN THE PAYLOAD — IT IS ONLY THE ROWS THAT  #
#    ARE NOT THERE. THE FUSED RECORD MUST CARRY THE ENGINE THAT PRODUCED IT.  #
# =========================================================================== #
#: Fields lifted VERBATIM off the SAM3 record's own `engine` block. Never
#: defaulted: a missing key means the producing run did not stamp it.
SAM3_ENGINE_FIELDS = ("confidence_threshold", "confidence_threshold_set_via",
                      "weights", "dtype_fix_applied")
#: What `perception.engine.stamped` says when the producing run predates the
#: stamping. ⚠️ NOT the same claim as "the floor is 0.5" — we do not know it
#: from the record, we know it from the run that made it.
ENGINE_UNSTAMPED = "unstamped — the producing run did not record its engine"


def perception_engine(s3: dict, *, absent_reason: str | None = None) -> dict:
    """The SAM3 engine identity for ONE clip: schema, detection floor, frames.

    ⛔ WHY THIS IS NOT COSMETIC. The aug120 perception layer is assembled from
    TWO SAM3 RUNS AT DIFFERENT DETECTION FLOORS — the batch-pipeline leg (86
    clips, vendor default 0.5, pre-schema: it stamps NEITHER field) and the
    `sam3_backfill_v2` leg (115 clips, floor 0.25, schema 2). MEASURED, not
    inferred: the two clip sets are DISJOINT and their union is exactly the
    201-clip population.

    A floor never appears in a record as a value — it appears as DETECTIONS
    THAT ARE NOT THERE. So a mixed-floor corpus reads as a homogeneous one, and
    every per-concept rate computed across it is unattributable while looking
    like an answer. That is the `df`-reports-the-cluster trap in a detector:
    *a probe whose scope is wrong is worse than no probe*. Stamping the engine
    per record makes the mixture something a consumer can FILTER ON, and
    `_summary.json` carries the census so it is loud instead of discoverable.

    ⚠️ A `None` value here means THE PRODUCING RUN DID NOT STAMP IT. It does
    not mean "unset", and it must never be coerced to a vendor default the
    record cannot support — that would manufacture the very fact this block
    exists to expose. (`SAM3_EXTRACTION_V2.md` §6: *"a floor is visible only as
    rows that are not there"*.)
    """
    if absent_reason:
        return {"stamped": False, "reason": absent_reason,
                "schema_version": None, "n_frames_run": None}
    eng = (s3 or {}).get("engine") or {}
    out: dict = {
        "schema_version": (s3 or {}).get("schema_version"),
        "n_frames_run": (s3 or {}).get("n_frames_run"),
        **{k: eng.get(k) for k in SAM3_ENGINE_FIELDS},
    }
    out["stamped"] = out["schema_version"] is not None or bool(eng)
    if not out["stamped"]:
        out["reason"] = ENGINE_UNSTAMPED
    return out


# --------------------------------------------------------------------------- #
# ⛔ THE SCENE CHANNEL IS COUNTED, NEVER TRACKED — STUFF IS NOT THINGS          #
# --------------------------------------------------------------------------- #
#: The v2 keys carried through verbatim. ⚠️ `frames[*].scene` deliberately does
#: NOT enter `build_tracks`.
SCENE_PASSTHROUGH = ("per_scene_hits", "n_scene_det_total", "concepts_scene",
                     "n_err_scene")


def scene_channel(s3: dict) -> dict | None:
    """The schema-v2 SCENE channel, carried VERBATIM. `None` on a v1 record.

    ⛔ IT IS NOT RUN THROUGH `build_tracks`, AND THAT IS THE WHOLE DESIGN.
    `ph0_sam3` returns a dashed lane line as ONE DETECTION PER DASH and an
    extended structure chopped by occlusion, so IoU association across frames
    would mint dozens of meaningless one-frame "tracks" per clip that look
    exactly like agent tracks. `per_scene_hits["lane marking"] = 141` is **141
    painted segments separately grounded**, not 141 lane markings — which is
    why `concept_kinds` (thing / stuff_instanced / stuff_extended /
    stuff_region) travels with the counts rather than being left for a reader
    to assume. A count of STUFF is not an object count.

    ⚠️ `ego_lane` is carried as the producer emitted it and is NOT promoted
    into `lane_context`. It supplies 2 of `s2_derive.LANE_CONTEXT_INPUTS`'
    four members (`n_lanes_same_direction`, `ego_lane_idx`); `route_lane_idx`
    and `lane_continues` need lane TOPOLOGY, which no camera frame contains
    and PhysicalAI-AV does not ship. `lane_change_requirement()` therefore
    still returns `required=None` and no token moves — promoting a per-frame
    estimate (null on ~32 % of frames) into a clip-level scalar would bake in
    an unreviewed aggregation policy for zero label change. Escalated, not
    half-done. See `SAM3_EXTRACTION_V2.md` §3 and §7.3.
    """
    if not s3 or s3.get("schema_version") is None:
        return None
    out = {k: s3.get(k) for k in SCENE_PASSTHROUGH if s3.get(k) is not None}
    if not out:
        return None
    kinds = s3.get("concept_kinds") or {}
    scene_names = set(s3.get("concepts_scene") or ()) | set(
        (s3.get("per_scene_hits") or {}))
    out["concept_kinds"] = {k: v for k, v in kinds.items() if k in scene_names}
    if s3.get("ego_lane") is not None:
        out["ego_lane"] = s3["ego_lane"]
    out["src"] = "sam3"
    out["tracks_note"] = (
        "counted, never tracked: SAM3 returns a dashed line as one detection "
        "per dash, so these are SEGMENT counts (see concept_kinds), not "
        "object counts, and they are deliberately absent from perception."
        "tracks / per_concept_hits — the agent contract does not move")
    return out


def census_phrase(cen: dict) -> str:
    """The prose rendering — the ONLY place a census becomes English."""
    if cen["state"] == CENSUS_UNAVAILABLE:
        return f"agent census UNAVAILABLE ({cen['reason']})"
    if not cen["counts"]:
        # a COUNT, not a claim about the world: the detector ran and returned
        # nothing, which is a measurement of the detector's output.
        return "0 agents detected"
    body = ", ".join(f"{v} {k}" for k, v in sorted(cen["counts"].items()))
    # ⛔ the UNIT is in the sentence, not in a schema a reader may not open.
    # "73 car" reads as seventy-three cars; it is 73 strided-frame TRACKS,
    # ~88 % of them single-frame. See CENSUS_UNIT_NOTE.
    peak = cen.get("peak_concurrent_tracks")
    tail = f"; peak {peak}/frame" if peak else ""
    return f"sam3 tracks (NOT object counts): {body}{tail}"


def lane_phrase(sc: dict) -> str:
    """⚠️ `lanes_visible` is B1-DEFINED as *"lanes you can count on the ego's
    carriageway"* (`ph0_v2.py:140`) — it EXCLUDES the oncoming carriageway.
    Rendered as `"rural 1-lane"` it reads as "a one-lane road", which is what
    the PI (correctly) objected to on `03ba450b`: *"the pipeline is saying
    there is one lane, better: there are two lanes, one ego lane and one for
    oncoming traffic"*. The count was right by its own definition and the
    PHRASE was wrong. Name the scope in the prose so the two readings cannot
    be confused again."""
    n = sc.get("lanes_visible")
    road = sc.get("road_type", "?")
    if not isinstance(n, int) or n <= 0:
        # B1's documented escape (0 = unclear) — an absent count, not "0 lanes"
        return f"{road}, lane count UNCLEAR (B1 returned {n!r})"
    return f"{road} {n}-lane-ego-carriageway"


def scenario_line(v2: dict, tracks: list[dict], *,
                  absent_reason: str | None = None,
                  n_frames: int | None = None) -> str:
    sc = v2.get("scene") or {}
    ego = v2.get("ego_state") or {}
    sit = v2.get("situations") or {}
    cen = census_state(tracks, absent_reason=absent_reason, n_frames=n_frames)
    parts = [
        f"{sc.get('illumination', '?')}, {sc.get('weather', '?')}, "
        f"{lane_phrase(sc)}",
        f"ego {ego.get('v_now_ms', float('nan')):.1f} m/s {ego.get('motion', '?')}"
        f"/{ego.get('turning', '?')}",
        census_phrase(cen),
    ]
    flags = [k for k in ("lane_change", "intersection", "roundabout")
             if sit.get(k)]
    if flags:
        parts.append("situations: " + "+".join(flags))
    return "; ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-json", required=True)
    ap.add_argument("--sam3", required=True,
                    help="sam3 output: a JSON file or a directory of them")
    ap.add_argument("--ego-root", required=True)
    ap.add_argument("--records", default=None,
                    help="Alpamayo records.parquet (optional layer)")
    ap.add_argument("--engine-a", default=None,
                    help="optional Engine A sidecar: JSONL rows "
                         '{"clip_id", "engine_a"} (e.g. the S2 v1 recompute '
                         "from the bridged npz). A v2 record's own persisted "
                         "engine_a field wins over the sidecar.")
    ap.add_argument("--missing-sam3-ok", default=None, metavar="REASON",
                    help="permit clips present in --v2-json but absent from "
                         "the SAM3 output: fuse their other layers, stamp "
                         "perception {'absent': REASON}, and degrade the "
                         "SAM3-dependent checks to not_computable. Without "
                         "this flag a partial SAM3 leg refuses loudly.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    v2 = json.load(open(a.v2_json))
    v2 = v2 if isinstance(v2, list) else v2.get("clips", v2)
    v2_by = {r["clip_id"]: r for r in v2 if isinstance(r, dict)}

    sam3_by: dict[str, dict] = {}
    paths = ([a.sam3] if os.path.isfile(a.sam3)
             else sorted(glob.glob(os.path.join(a.sam3, "*.json"))))
    for p in paths:
        d = json.load(open(p))
        # MEASURED formats: the production sam3.json is a metadata wrapper with
        # the records under "clips" (596 rows); older outputs were a bare list
        # or one record. The first fuse run matched 0 of 600 because this
        # wrapper wasn't handled — n_sam3 is asserted below so an empty
        # perception layer can never again look like a successful fuse.
        if isinstance(d, dict) and isinstance(d.get("clips"), list):
            rows = d["clips"]
        elif isinstance(d, list):
            rows = d
        elif isinstance(d, dict) and "clip_id" in d:
            rows = [d]
        else:
            rows = list(d.values()) if isinstance(d, dict) else []
        for r in rows:
            if isinstance(r, dict) and r.get("clip_id"):
                sam3_by[r["clip_id"]] = r
    if not sam3_by:
        raise SystemExit("[fuse] loaded 0 SAM3 records — refusing to emit "
                         "fused records with an empty perception layer "
                         f"(looked in {a.sam3})")
    missing = sorted(c for c in v2_by if c not in sam3_by)
    if missing and not a.missing_sam3_ok:
        raise SystemExit(
            f"[fuse] {len(missing)} of {len(v2_by)} v2 clips have NO SAM3 "
            f"record (e.g. {missing[:3]}) — pass --missing-sam3-ok REASON "
            "to fuse them with the perception layer explicitly marked "
            "absent, or complete the SAM3 leg first")

    alp_by: dict[str, dict] = {}
    if a.records:
        import pandas as pd
        df = pd.read_parquet(a.records)
        for cid, g in df.groupby("clip_id"):
            alp_by[cid] = {row["task"]: row.get("raw_json")
                           for _, row in g.iterrows()}

    ea_by: dict[str, dict] = {}
    if a.engine_a:
        with open(a.engine_a, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if row.get("clip_id") and row.get("engine_a"):
                        ea_by[row["clip_id"]] = row["engine_a"]
        print(f"[fuse] engine A sidecar: {len(ea_by)} clips", flush=True)

    n, summ = 0, {"corroborated": 0, "conflicts": 0, "with_alpamayo": 0,
                  "sam3_missing": 0, "with_scene_channel": 0}
    engines: dict = {}
    for cid, r in sorted(v2_by.items()):
        dst = os.path.join(a.out, f"{cid}.json")
        if os.path.exists(dst):
            continue
        absent_reason = a.missing_sam3_ok if cid not in sam3_by else None
        s3 = sam3_by.get(cid) or {}
        frames = s3.get("frames") or {}
        tracks = build_tracks(frames)
        scene = scene_channel(s3)
        # ⛔ the mixed-floor census: keyed on (schema, floor) so a corpus
        # assembled from two SAM3 runs cannot read as a homogeneous one.
        eng_row = perception_engine(s3, absent_reason=absent_reason)
        engines[(eng_row.get("schema_version"),
                 eng_row.get("confidence_threshold"))] = \
            engines.get((eng_row.get("schema_version"),
                         eng_row.get("confidence_threshold")), 0) + 1
        summ["with_scene_channel"] += int(scene is not None)
        census = census_state(tracks, absent_reason=absent_reason,
                              n_frames=len(frames))
        # engine-A spine: recompute from the bridged npz when the v2 record
        # does not carry it (the production records do not — MEASURED)
        if "speed_profile" not in r:
            spine = ego_from_npz(os.path.join(a.ego_root, f"{cid}.npz"))
            if spine:
                r = {**r, **spine}
        cor, conf = corroborate(r, s3, tracks,
                                sam3_absent=absent_reason is not None)
        alp = alp_by.get(cid)
        engine_a = r.get("engine_a") or ea_by.get(cid)
        vocab, vconf = emit_vocab(r, alp, engine_a=engine_a)
        conf += vconf
        # ⛔ the provenance lie fix (S2_STRATEGIC_GAP §5): the v2.2 records
        # were produced with ego-past kinematics AND the Engine A hindsight
        # block in the prompt — the VLM layer is NOT pure vision, and the
        # tag must say it. `semantics` is then also removed from the
        # inference whitelist: labels may use ego, INFERENCE IS VISION-ONLY.
        ego_mode = r.get("_ego_prompt_mode") or "none"
        vlm_prov = ("vision" if ego_mode == "none" else
                    f"vision+ego-{ego_mode}-prompt+engineA-prompt")
        admissible = (["perception", "semantics"] if ego_mode == "none"
                      else ["perception"])
        fused = {
            "schema_version": SCHEMA, "clip_id": cid,
            "geometry": {"frame_wh": r.get("_frame_wh"),
                         "note": "w120 cylindrical vs 256px pinhole batches "
                                 "must not be pooled"},
            "ego": {k: v for k, v in
                    [(k, r.get(k)) for k in
                     ("ego_state", "route", "speed_profile", "speed_events",
                      "lane_change_events", "situations")] +
                    [("engine_a", engine_a)] if v is not None or k in r},
            "perception": {"tracks": tracks,
                           "per_concept_hits": s3.get("per_concept_hits"),
                           "src": "sam3",
                           # ⛔ the machine-readable census — read THIS, never
                           # parse `scenario_description` (C77 fix)
                           "census": census,
                           # ⛔ the engine that produced this clip's pixels —
                           # a detection floor is invisible in the payload
                           "engine": perception_engine(
                               s3, absent_reason=absent_reason),
                           **({"scene": scene} if scene else {}),
                           **({"absent": absent_reason}
                              if absent_reason else {})},
            "semantics": {"scene": r.get("scene"), "signs": r.get("signs"),
                          "symbols": r.get("symbols"), "src": "vlm",
                          "sign_text_status": "pending_g1_gate"},
            "alpamayo": alp,
            "corroboration": cor, "vocab": vocab,
            "scenario_description": scenario_line(
                r, tracks, absent_reason=absent_reason, n_frames=len(frames)),
            "_conflicts": conf,
            "inference_admissible": admissible,
            **({"_inference_admissible_note":
                f"semantics EXCLUDED: VLM ran with ego in the prompt "
                f"(mode={ego_mode!r}) — its outputs are ego-touched"}
               if ego_mode != "none" else {}),
            "_provenance": {"ego": "privileged-labels-only",
                            "sam3": "vision", "vlm": vlm_prov,
                            "alpamayo": "external-labels-only",
                            **({"engine_a": "privileged-labels-only "
                                            "(hindsight ego path)"}
                               if engine_a else {})},
        }
        json.dump(fused, open(dst, "w"), indent=1)
        n += 1
        summ["conflicts"] += len(conf)
        summ["corroborated"] += sum(1 for c in cor.values()
                                    if c.get("verdict") == "corroborated")
        summ["with_alpamayo"] += int(alp is not None)
        summ["sam3_missing"] += int(absent_reason is not None)
        if n % 100 == 0:
            print(f"[fuse] {n} fused", flush=True)
    summ["n_fused"] = n
    summ["n_v2"] = len(v2_by)
    summ["n_sam3"] = len(sam3_by)
    # ⛔ THE MIXED-FLOOR CENSUS, EMITTED WHETHER OR NOT IT IS MIXED. A
    # homogeneous corpus reports ONE row; a corpus assembled from two SAM3 runs
    # reports two, and `perception_engine_mixed` says so in one boolean. This
    # is the fact a per-concept rate needs and the payload cannot otherwise
    # supply — a detection floor shows up only as rows that are not there.
    # ⚠️ Scoped to the records THIS invocation wrote (the fuser resumes by
    # skipping existing outputs), exactly like every other counter here.
    summ["perception_engines"] = [
        {"schema_version": k[0], "confidence_threshold": k[1], "n_clips": v}
        for k, v in sorted(engines.items(),
                           key=lambda kv: (str(kv[0][0]), str(kv[0][1])))]
    summ["perception_engine_mixed"] = len(engines) > 1
    if a.missing_sam3_ok:
        summ["missing_sam3_reason"] = a.missing_sam3_ok
    json.dump(summ, open(os.path.join(a.out, "_summary.json"), "w"), indent=1)
    print(f"FUSE_DONE {json.dumps(summ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
