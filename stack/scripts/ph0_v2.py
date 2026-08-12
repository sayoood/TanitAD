"""PH0 v2 — the four-call solvable structure (PI: "it must be solvable").

Replaces `ph0-v0.1`'s single 5-section / 4-level / ~30-field JSON object, which
MEASURED 1/8 schema-valid on Qwen3.5-9B with 3/8 unparseable. See
`PH0_TARGET_STRUCTURE_v2.md` for the full diagnosis.

Organising principle: **the VLM chooses SYMBOLS, the algorithm supplies NUMBERS,
SAM3 supplies PIXELS.** Every metric slot (`within_m`, `by_time_s`, `at_arc_m`,
`hold_for_s`, `v_target_ms`) is REMOVED from the VLM's job — Engine A measures
them from the integrated ego path. Bboxes move to their own tiny call (and to
SAM3). What is left for the VLM is what a VLM is uniquely for: closed-vocabulary
classification and verbatim text reading.

⭐ Grammar-constrained decoding is MANDATORY here, not an optimisation: with a
JSON-schema FSM over the token stream, `no parseable JSON object` becomes
impossible by construction, which removes the entire 3/8 hard-failure class
rather than retrying it. Every field is a closed enum, a bounded int, a bool or
one short string, so the grammar stays small.

Every call's PROMPT and RAW MODEL OUTPUT are dumped to the artifact so a reader
can verify what was asked and what came back, rather than trusting a pass rate.

v2.2 adds the ego state to the prompt (PI, 2026-08-12). See the EGO STATE block
below for why that is in-contract for a LABEL pipeline, and for the one place it
would have leaked (B2 reads sign numbers; the speedometer is a number) and how
that is redacted rather than hoped away.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# --------------------------------------------------------------------------- #
# vocabulary — mirrors HIERARCHY_VOCABULARY.md / ph0_pilot.py                   #
# --------------------------------------------------------------------------- #
GOAL_KINDS = ["keep_corridor", "lane_target", "exit_left", "exit_right",
              "turn_left", "turn_right", "straight_through", "route_to",
              "stop_at", "follow_main_road", "none_abstain"]
ACTION_VERBS = ["prepare_lane_change", "hold_corridor", "reduce_to",
                "prepare_exit", "prepare_stop", "resume_cruise"]
SIGN_KINDS = ["light", "speed", "nav", "stop", "yield", "other"]
CONF = ["low", "med", "high"]

SCHEMA_VERSION = "ph0-v2.2"      # v2.1 = corrected video inference
                                 # v2.2 = ego state in the prompt (PI request)
VIDEO_SAMPLE_FPS = 2.0           # must match sample_clip_frames()

# --------------------------------------------------------------------------- #
# B1–B4 schemas. Flat, bounded, closed. additionalProperties False everywhere   #
# so the grammar cannot wander into an invented key.                            #
# --------------------------------------------------------------------------- #
S_B1 = {
    "type": "object", "additionalProperties": False,
    "required": ["illumination", "weather", "road_type", "domain",
                 "lanes_visible", "lane_ego", "conf"],
    "properties": {
        "illumination": {"enum": ["day", "dusk", "night", "dark"]},
        "weather": {"enum": ["clear", "rain", "snow", "fog", "unclear"]},
        "road_type": {"enum": ["highway", "urban", "rural", "junction",
                               "unclear"]},
        "domain": {"enum": ["highway", "urban", "roundabout", "intersection",
                            "rural", "unclear"]},
        "lanes_visible": {"type": "integer", "minimum": 0, "maximum": 6},
        "lane_ego": {"type": "integer", "minimum": 0, "maximum": 6},
        "conf": {"enum": CONF},
    },
}

S_B2 = {
    "type": "object", "additionalProperties": False,
    "required": ["n_signs", "signs"],
    "properties": {
        # n_signs FIRST so the array length is committed before the items —
        # this alone removes the commonest truncation failure.
        "n_signs": {"type": "integer", "minimum": 0, "maximum": 6},
        "signs": {
            "type": "array", "maxItems": 6,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["kind", "state", "text", "applies_to_ego"],
                "properties": {
                    "kind": {"enum": SIGN_KINDS},
                    "state": {"enum": ["red", "amber", "green", "none"]},
                    "text": {"type": "string", "maxLength": 40},
                    "applies_to_ego": {"type": "boolean"},
                },
            },
        },
    },
}

S_B3 = {
    "type": "object", "additionalProperties": False,
    "required": ["visible", "frame_idx", "bbox"],
    "properties": {
        "visible": {"type": "boolean"},
        "frame_idx": {"type": "integer", "minimum": 0, "maximum": 39},
        # ⭐ NORMALIZED 0–1000, which is Qwen-VL's OWN trained convention.
        # Asking for raw pixels fought that training and produced coordinates
        # that looked wildly out of frame (952 on a 448 px frame) but were
        # perfectly consistent in the model's native space. Work WITH it and
        # convert to pixels in post.
        "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                 "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
    },
}

S_B4 = {
    "type": "object", "additionalProperties": False,
    "required": ["goal_kind", "goal_evidence_sign", "actions", "conf"],
    "properties": {
        "goal_kind": {"enum": GOAL_KINDS},
        "goal_evidence_sign": {"type": ["integer", "null"], "minimum": 0,
                               "maximum": 5},
        "actions": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["verb", "direction"],
                "properties": {
                    "verb": {"enum": ACTION_VERBS},
                    "direction": {"enum": ["left", "right", "none"]},
                },
            },
        },
        "conf": {"enum": CONF},
    },
}

# --------------------------------------------------------------------------- #
# prompts — ONE JOB EACH, short, no prose rules to hold in working memory       #
# --------------------------------------------------------------------------- #
P_B1 = """You see a driving video: frames 0..{n_past_1} are the 8 s BEFORE the \
decision time, frames {n_past}..{n_last} are the 12 s AFTER it.

Describe the SCENE only. Do not describe behaviour or intent.
lanes_visible = lanes you can count on the ego's carriageway (0 if unclear).
lane_ego = ego's lane index from the right, 0-based (0 if unclear).
conf = your confidence in this scene description."""

P_B2 = """Same driving video.

List ONLY traffic signs, traffic lights and navigation signs that are LEGIBLE \
in the frames. Do not list road markings or vehicles.
- Give n_signs first, then exactly that many entries.
- text = the VERBATIM legible text (city name, number). If nothing is legible, \
use "". NEVER invent text.
- state applies to traffic lights; use "none" for other kinds.
- applies_to_ego = does this sign govern the ego's own lane/direction?
If you see no legible signs, answer n_signs 0 and an empty list."""

P_B3 = """Same driving video. Sign {idx} was reported as: kind={kind}, \
text="{text}".

Locate THAT sign.
- frame_idx: which frame it is clearest in. An INTEGER from 0 to {n_last} \
(there are {n_frames} frames). This is a frame NUMBER, not a coordinate.
- bbox [x0,y0,x1,y1]: its box in NORMALIZED image coordinates 0-1000, where 0 \
is the left/top edge and 1000 is the right/bottom edge. Require x0<x1 and y0<y1.
If you cannot actually see it, answer visible false with bbox [0,0,0,0]."""

P_B4 = """Same driving video. The frames AFTER the decision time are HINDSIGHT \
evidence of what the driver actually did.

A deterministic geometric analysis of the ego trajectory is given as ground \
truth. Do not contradict it:
ENGINE_A = {engine_a}

Signs read from the video: {signs}

Choose the strategic GOAL and up to 3 tactical ACTIONS.
- Use ONLY the listed vocabulary.
- goal_kind "route_to" is allowed ONLY if a navigation sign was actually read; \
set goal_evidence_sign to its index. Otherwise use "follow_main_road", a \
corridor/lane goal, or "none_abstain".
- Abstaining is better than guessing.
- direction is "none" unless the verb needs a side.
DO NOT output distances, speeds, times or durations — those are measured \
separately from the trajectory."""


# --------------------------------------------------------------------------- #
# POST-PARSE VALIDATION                                                         #
# --------------------------------------------------------------------------- #
# ⛔ MEASURED 2026-08-12: the grammar enforces STRUCTURE and TYPE but NOT
# numeric bounds. A schema saying {"type":"integer","maximum":448} still let the
# model emit bbox [952,100,975,160] on a 448 px frame. Every `minimum`/`maximum`
# in the schemas above is therefore DECORATIVE, and this layer is what actually
# holds them. Nor does `maxItems` imply distinct: B4 emitted hold_corridor three
# times. Anything the grammar cannot guarantee is checked here, by hand.
BBOX_MAX = 1000          # normalized space; see S_B3


def norm_to_px(bbox, w: int, h: int) -> list[int]:
    """[0,1000] normalized -> pixels for THIS frame's own w/h.

    ⚠️ x and y have DIFFERENT pixel ranges (frames here are 179x448), which is
    the second half of the bug: validating both against a single 448 maximum
    was wrong even in pixel space."""
    x0, y0, x1, y1 = bbox
    return [int(round(x0 / BBOX_MAX * w)), int(round(y0 / BBOX_MAX * h)),
            int(round(x1 / BBOX_MAX * w)), int(round(y1 / BBOX_MAX * h))]


def validate_v2(call: str, obj: dict, *, px: int = BBOX_MAX,
                n_frames: int = 40) -> list[str]:
    """Return a list of violations; empty means valid. Pure — CPU-testable."""
    e: list[str] = []
    if not isinstance(obj, dict):
        return [f"{call}: not an object"]

    if call == "B1_scene":
        for k in ("lanes_visible", "lane_ego"):
            v = obj.get(k)
            if not isinstance(v, int) or not 0 <= v <= 6:
                e.append(f"B1.{k}={v!r} outside 0..6")
        if isinstance(obj.get("lane_ego"), int) \
                and isinstance(obj.get("lanes_visible"), int) \
                and obj["lanes_visible"] > 0 \
                and obj["lane_ego"] >= obj["lanes_visible"]:
            e.append(f"B1.lane_ego {obj['lane_ego']} >= lanes_visible "
                     f"{obj['lanes_visible']}")

    elif call == "B2_signs":
        n, signs = obj.get("n_signs"), obj.get("signs")
        if not isinstance(signs, list):
            e.append("B2.signs not a list")
        elif not isinstance(n, int) or n != len(signs):
            # the whole point of emitting n_signs FIRST is that it must match
            e.append(f"B2.n_signs {n!r} != len(signs) {len(signs)}")
        for i, s in enumerate(signs or []):
            if s.get("kind") != "light" and s.get("state") != "none":
                e.append(f"B2.signs[{i}] state {s.get('state')!r} on a "
                         f"non-light ({s.get('kind')!r})")
            if len(str(s.get("text", ""))) > 40:
                e.append(f"B2.signs[{i}].text longer than 40")

    elif call.startswith("B3_ground"):
        bb = obj.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            e.append("B3.bbox not 4 ints")
        else:
            if any(not isinstance(v, int) or not 0 <= v <= px for v in bb):
                e.append(f"B3.bbox {bb} outside 0..{px}")
            elif bb != [0, 0, 0, 0] and not (bb[0] < bb[2] and bb[1] < bb[3]):
                e.append(f"B3.bbox {bb} not x0<x1 and y0<y1")
        fi = obj.get("frame_idx")
        if not isinstance(fi, int) or not 0 <= fi < n_frames:
            e.append(f"B3.frame_idx={fi!r} outside 0..{n_frames-1}")
        if obj.get("visible") is False and obj.get("bbox") != [0, 0, 0, 0]:
            e.append("B3.visible false but bbox non-zero")

    elif call == "B4_symbols":
        if obj.get("goal_kind") not in GOAL_KINDS:
            e.append(f"B4.goal_kind {obj.get('goal_kind')!r} off-vocabulary")
        acts = obj.get("actions")
        if not isinstance(acts, list):
            e.append("B4.actions not a list")
        else:
            # ⚠️ Duplicates are NOT a violation. MEASURED: the model pads the
            # array to maxItems, repeating the same verb — 6 of 8 clips. That
            # is a formatting artifact carrying no wrong content, and failing
            # the record for it DISCARDED a perfectly good goal_kind. They are
            # deduped by dedupe_actions() and counted as a warning instead.
            for i, a in enumerate(acts):
                if a.get("verb") not in ACTION_VERBS:
                    e.append(f"B4.actions[{i}].verb {a.get('verb')!r} "
                             f"off-vocabulary")
                if a.get("verb") == "prepare_lane_change" \
                        and a.get("direction") not in ("left", "right"):
                    e.append(f"B4.actions[{i}] lane change needs a side")
        # the prereg's own rule: route_to REQUIRES read signage
        if obj.get("goal_kind") == "route_to" \
                and obj.get("goal_evidence_sign") is None:
            e.append("B4.goal_kind route_to without goal_evidence_sign")
    return e


def dedupe_signs(sg: dict | None) -> tuple[dict | None, int]:
    """Drop repeated (kind, text, applies_to_ego) signs and re-sync n_signs.

    ⛔ MEASURED on CORRECTED inference (ph0-v2.1): the padding survives. One
    clip reported speed "100" x4, another yield "" x5, another "P" x3. So this
    is NOT the near-blind-inference artifact — it is the model filling the array
    to maxItems, exactly as it did for B4 actions. I deduped actions and left
    signs, which is why the duplicates were still here to find."""
    if not sg or not isinstance(sg.get("signs"), list):
        return sg, 0
    seen, keep = set(), []
    for s_ in sg["signs"]:
        k = (s_.get("kind"), s_.get("text"), s_.get("applies_to_ego"))
        if k in seen:
            continue
        seen.add(k)
        keep.append(s_)
    n = len(sg["signs"]) - len(keep)
    if n:
        sg = dict(sg, signs=keep, n_signs=len(keep))
    return sg, n


def dedupe_actions(sym: dict | None) -> tuple[dict | None, int]:
    """Drop repeated (verb, direction) pairs, preserving order. Returns the
    cleaned record and how many duplicates were removed."""
    if not sym or not isinstance(sym.get("actions"), list):
        return sym, 0
    seen, keep = set(), []
    for a in sym["actions"]:
        k = (a.get("verb"), a.get("direction"))
        if k in seen:
            continue
        seen.add(k)
        keep.append(a)
    n = len(sym["actions"]) - len(keep)
    if n:
        sym = dict(sym, actions=keep)
    return sym, n


# --------------------------------------------------------------------------- #
# EGO STATE IN THE PROMPT (PI, 2026-08-12: "give also the ego states as          #
# additional input for the vlm")                                                #
# --------------------------------------------------------------------------- #
# ⭐ WHY THIS IS ADMISSIBLE HERE, and where it would NOT be. PH0 is a LABEL
# DERIVATION pipeline — it mints pseudo-labels offline for v6 to train on. The
# binding rule (CLAUDE.md, Sayed 2026-08-03) is explicit that at the label stage
# "ego state, other agents, maps, future poses — anything" may be used, and that
# it is INFERENCE that is vision-only. So ego here is in-contract. What it does
# NOT license is a v6 head consuming ego at inference; that constraint lives on
# the deployed arm and is unchanged by this.
#
# ⚠️ THE ONE REAL LEAK, AND WHY B2 IS REDACTED. B2 reads sign TEXT, and a speed
# sign's text is a NUMBER. Hand the model its own speedometer and "50" can be
# transcribed from the ego state rather than read off the sign — which would
# make the sign channel unfalsifiable exactly the way the nav-echo defect and
# the REF-A I-JEPA leak were. Same test as always: does an input contain
# something the thing being measured also produces? For B2 + speed signs, yes.
# ⇒ B2 sees a SPEED-REDACTED block (motion words, no magnitudes). The redaction
# is itself measurable: `--ego-in-prompt full` shows the unredacted block to B2
# as well, and the delta in speed-sign recall between the two runs IS the leak
# magnitude. B3 gets NO block at all — it is pure spatial localisation, ego
# kinematics is irrelevant to it, and it is the call that repeats up to 6x.
EGO_WINDOW_S = 8.0               # the pre-decision window; matches t0_s=8.0
EGO_MODES = ("none", "past", "full")


def _wrap_pi(a):
    import numpy as np
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def ego_past_state(poses, t0_idx: int, *, dt: float = 0.1,
                   window_s: float = EGO_WINDOW_S) -> dict | None:
    """PAST-ONLY ego kinematics over the window ending at the decision time.

    ⛔ Deliberately carries NOTHING after `t0_idx`. Engine A already supplies the
    future (that is the hindsight channel, and B4 is the only call entitled to
    it); mixing the two here would put future evidence into B1/B2, whose whole
    job is to describe what is visible up to the decision.

    poses is [T, 4] = (x, y, yaw_rad, v_ms) at 1/dt Hz. Returns None rather than
    raising when the array is unusable — absence of ego is not an error, the
    caller just emits no block."""
    import numpy as np
    p = poses.detach().cpu() if hasattr(poses, "detach") else poses
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 4:
        return None
    T = int(p.shape[0])
    t0 = max(0, min(int(t0_idx), T - 1))
    if t0 < 1:
        return None
    i0 = max(0, t0 - int(round(window_s / dt)))
    v, yaw, xy = p[i0:t0 + 1, 3], p[i0:t0 + 1, 2], p[i0:t0 + 1, :2]

    def _accel(sec):
        k = int(round(sec / dt))
        if k <= 0 or k >= len(v):
            return None
        return round(float((v[-1] - v[-1 - k]) / (k * dt)), 2)

    dyaw = _wrap_pi(np.diff(yaw))

    def _yawrate(sec):
        k = int(round(sec / dt))
        if k <= 0 or k > len(dyaw):
            return None
        return round(float(dyaw[-k:].sum() / (k * dt)), 3)

    a1, a3, r1 = _accel(1.0), _accel(3.0), _yawrate(1.0)
    v_now = float(v[-1])
    a_ref = a3 if a3 is not None else (a1 or 0.0)
    r_ref = r1 or 0.0
    motion = ("stopped" if v_now < 0.5 else
              "braking" if a_ref <= -0.6 else
              "accelerating" if a_ref >= 0.6 else "steady")
    turning = ("turning_left" if r_ref > 0.06 else
               "turning_right" if r_ref < -0.06 else "straight")
    return {
        "window_s": round((len(v) - 1) * dt, 1),
        "v_now_ms": round(v_now, 2),
        "v_now_kmh": round(v_now * 3.6, 1),
        "v_mean_ms": round(float(v.mean()), 2),
        "v_min_ms": round(float(v.min()), 2),
        "v_max_ms": round(float(v.max()), 2),
        "accel_1s_ms2": a1, "accel_3s_ms2": a3,
        "yaw_rate_rad_s": r1,
        "net_dyaw_rad": round(float(dyaw.sum()), 3) if len(dyaw) else 0.0,
        "dist_travelled_m": round(
            float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1),
        "motion": motion, "turning": turning,
    }


# keys stripped for B2 — every one of them is a magnitude a sign could state
_EGO_SPEED_KEYS = ("v_now_ms", "v_now_kmh", "v_mean_ms", "v_min_ms", "v_max_ms")


def fmt_ego(st: dict | None, *, redact_speed: bool = False) -> str:
    if not st:
        return "{}"
    d = {k: v for k, v in st.items()
         if v is not None and not (redact_speed and k in _EGO_SPEED_KEYS)}
    return json.dumps(d)


_EGO_TAIL = {
    "B1_scene": "Use it to judge road type and lane count. It tells you nothing "
                "about what any sign says.",
    "B2_signs": "It tells you HOW THE CAR MOVED. It does NOT tell you any "
                "sign's text or number. NEVER copy a number from it into "
                "`text` — if a number is not legible in the image, use \"\".",
    "B4_symbols": "This is the state the decision was made FROM; ENGINE_A is "
                  "what actually happened after it.",
}


def ego_section(call: str, st: dict | None, mode: str = "past") -> str:
    """The prompt fragment, or "" — so `mode="none"` reproduces v2.1's prompts
    BYTE-IDENTICALLY and the ablation control is exact rather than approximate."""
    if not st or mode == "none" or call not in _EGO_TAIL:
        return ""
    body = fmt_ego(st, redact_speed=(call == "B2_signs" and mode != "full"))
    return (f"\n\nEGO_STATE — measured from the ego vehicle's own sensors over "
            f"the {st.get('window_s', EGO_WINDOW_S):.0f}s BEFORE the decision "
            f"time. It is FACT, not a guess:\n{body}\n{_EGO_TAIL[call]}")


def _fmt_engine_a(ea: dict | None) -> str:
    """Engine A summary for the B4 prompt — compact and metric, so the model
    is anchored by geometry rather than asked to invent it."""
    if not ea:
        return "{}"
    # keys are engine_a_summary's own (verified against ph0_pilot.py, not
    # guessed): route{token,dist_m,arc_m,maneuver_dyaw_rad}, speed_profile{...},
    # lane_change_events, speed_events, peak_kappa_per_m. The polyline is
    # already dropped by engine_a_for_prompt.
    r, sp = ea.get("route", {}), ea.get("speed_profile", {})
    return json.dumps({
        "route_token": r.get("token"),
        "route_valid": r.get("token_valid"),
        "route_dist_m": r.get("dist_m"),
        "route_arc_m": r.get("arc_m"),
        "maneuver_dyaw_rad": r.get("maneuver_dyaw_rad"),
        "v_min_future_ms": sp.get("v_min_future_ms"),
        "v_max_future_ms": sp.get("v_max_future_ms"),
        "net_dv_ms": sp.get("net_dv_ms"),
        "stops": sp.get("stops"),
        "peak_kappa_per_m": ea.get("peak_kappa_per_m"),
        "lane_change_events": ea.get("lane_change_events", [])[:3],
        "speed_events": ea.get("speed_events", [])[:3],
    })


class ConstrainedVLM:
    """Engine B with a JSON-schema FSM over the token stream."""

    def __init__(self, model_id: str):
        import torch
        import transformers
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id,
                                                       trust_remote_code=True)
        self.model, self.auto_class, errs = None, None, []
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(
                    model_id, torch_dtype="auto", device_map="cuda:0",
                    trust_remote_code=True)
                self.auto_class = name
                break
            except Exception as e:
                errs.append(f"{name}: {type(e).__name__}: {str(e)[:90]}")
        if self.model is None:
            raise RuntimeError(f"no usable auto-class: {errs}")
        self.model.eval()
        self.model_id = model_id
        self._torch = torch
        self.tok = getattr(self.processor, "tokenizer", None) or self.processor
        # ⛔ lm-format-enforcer's OWN transformers integration is broken against
        # transformers 5.x: it does `from transformers.tokenization_utils import
        # PreTrainedTokenizerBase`, which moved, and then re-raises the import
        # error as the very misleading "transformers is not installed".
        # Its CORE enforcer is fine, so we build the adapter ourselves rather
        # than depend on their shim. This is the standard pattern their
        # integration uses, not an invention.
        from qwen_vl_utils import process_vision_info
        self._pvi = process_vision_info
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        from lmformatenforcer.characterlevelparser import \
            CharacterLevelParserConfig
        from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
        self._JsonSchemaParser = JsonSchemaParser
        self._TokenEnforcer = TokenEnforcer
        self._CfgCls = CharacterLevelParserConfig
        self._tok_data = self._build_tokenizer_data(TokenEnforcerTokenizerData)

    def _build_tokenizer_data(self, TokenEnforcerTokenizerData):
        """Vocabulary table the enforcer needs: (id, text-after-a-digit,
        is_word_start). Built ONCE per model — it is a full vocab sweep."""
        tok = self.tok
        token_0 = tok.encode("0")[-1]
        specials = set(getattr(tok, "all_special_ids", []) or [])
        regular = []
        for tid in range(len(tok)):
            if tid in specials:
                continue
            try:
                after_0 = tok.decode([token_0, tid])[1:]
                plain = tok.decode([tid])
            except Exception:
                continue
            regular.append((tid, after_0, len(after_0) > len(plain)))
        # signature VERIFIED on pod4, not assumed:
        #   (regular_tokens, decoder, eos_token_id, use_bitmask, vocab_size)
        # use_bitmask=False makes get_allowed_tokens return a LIST of ids,
        # which is what generate()'s prefix_allowed_tokens_fn expects; True
        # would return a tensor mask and silently mis-drive generation.
        return TokenEnforcerTokenizerData(regular, tok.decode,
                                          tok.eos_token_id, False, len(tok))

    def ask(self, frames, prompt: str, schema: dict, max_new: int = 256):
        """One constrained call. Returns (raw_text, parsed_or_None, err)."""
        torch = self._torch
        from PIL import Image
        pil = [Image.fromarray(f) for f in frames]
        # ⛔ THE COOKBOOK PATH (QwenLM/Qwen3-VL). The previous call passed a PIL
        # list with `fps` and no video_metadata, so transformers fell back to
        # fps=24 and the model saw a 40-frame / 20-SECOND clip as 1.67 s.
        # MEASURED by A/B on pod4 2026-08-12, and the damage was not only
        # temporal: that path produced **256 input tokens** where this one
        # produces **3122**. The model was being shown almost none of the video.
        # Its own reasoning shows the difference — "first half (00:00 - 00:01)"
        # under the old call vs "(00:00 - 00:09)" and "last frame 00:19" here.
        msg = [{"role": "user", "content": [
            {"type": "video", "video": pil, "sample_fps": VIDEO_SAMPLE_FPS},
            {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=False)
        images, videos, video_kwargs = self._pvi(
            msg, image_patch_size=16, return_video_kwargs=True,
            return_video_metadata=True)
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        inputs = self.processor(text=text, images=images, videos=videos,
                                video_metadata=video_metadatas,
                                return_tensors="pt", do_resize=False,
                                **video_kwargs)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        # ⭐ config VERIFIED on pod4, and it fixes two measured defects:
        #  · max_consecutive_whitespaces default is 12 — the model burned its
        #    whole token budget on tabs/spaces and B1 TRUNCATED mid-object.
        #    1 is enough for readability and cannot eat a budget.
        #  · force_json_field_order makes `required` order BINDING, so B2's
        #    n_signs really is emitted before signs[] instead of by luck.
        cfg = self._CfgCls(max_consecutive_whitespaces=1,
                           force_json_field_order=True,
                           max_json_array_length=6)
        parser = self._JsonSchemaParser(schema, config=cfg)
        enforcer = self._TokenEnforcer(self._tok_data, parser)

        def prefix_fn(_batch_id, sent):
            # get_allowed_tokens returns lmformatenforcer's OWN TokenList, not a
            # list — it has no __len__/__iter__, so handing it straight to
            # generate() dies with "object of type 'TokenList' has no len()".
            # With use_bitmask=False its .allowed_tokens IS a plain list of ids
            # (verified from the class source, not guessed).
            t = enforcer.get_allowed_tokens(sent.tolist())
            return getattr(t, "allowed_tokens", t)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new,
                                      do_sample=False,
                                      prefix_allowed_tokens_fn=prefix_fn)
        raw = self.tok.decode(out[0, n_in:], skip_special_tokens=True)
        try:
            return raw, json.loads(raw), None
        except Exception as e:
            return raw, None, f"{type(e).__name__}: {e}"


def run_clip(vlm, frames, n_past, engine_a, *, px=448, dump=None,
             ego_state=None, ego_mode="past"):
    n_last = len(frames) - 1
    frame_h, frame_w = int(frames[0].shape[0]), int(frames[0].shape[1])
    calls, rec = [], {}

    def record(name, prompt, schema, max_new=256):
        """One call, retried ONCE on a validation failure with the violations
        fed back. Retry is on VALIDATION, not on parse — the grammar makes a
        parse failure a budget problem, which a retry cannot fix."""
        t = time.time()
        raw, parsed, err = vlm.ask(frames, prompt, schema, max_new)
        viol = validate_v2(name, parsed) if parsed is not None else []
        attempts, retried = 1, False
        if parsed is not None and viol:
            retried = True
            fix = (prompt + "\n\nYour previous answer violated: "
                   + "; ".join(viol[:4]) + "\nAnswer again, corrected.")
            raw2, parsed2, err2 = vlm.ask(frames, fix, schema, max_new)
            viol2 = validate_v2(name, parsed2) if parsed2 is not None else []
            attempts = 2
            if parsed2 is not None and not viol2:
                raw, parsed, err, viol = raw2, parsed2, err2, viol2
        calls.append({"call": name, "prompt": prompt, "raw_output": raw,
                      "parsed": parsed, "error": err,
                      "violations": viol, "retried": retried,
                      "attempts": attempts,
                      "parsed_ok": parsed is not None,
                      "valid": parsed is not None and not viol,
                      "wall_s": round(time.time() - t, 1)})
        return parsed if not viol else None

    # budgets have headroom now that whitespace is capped at 1 — the single
    # measured parse failure was a whitespace-driven truncation at 128.
    rec["scene"] = record("B1_scene", P_B1.format(
        n_past_1=n_past - 1, n_past=n_past, n_last=n_last)
        + ego_section("B1_scene", ego_state, ego_mode), S_B1, 192)
    sg_raw = record("B2_signs",
                    P_B2 + ego_section("B2_signs", ego_state, ego_mode),
                    S_B2, 384)
    sg_raw, n_sign_dup = dedupe_signs(sg_raw)
    rec["signs"] = sg_raw
    rec["_n_sign_dupes_dropped"] = n_sign_dup

    grounded = []
    for i, s in enumerate((rec["signs"] or {}).get("signs", [])[:6]):
        g = record(f"B3_ground_{i}", P_B3.format(
            idx=i, kind=s.get("kind"), text=s.get("text", ""),
            n_last=n_last, n_frames=len(frames)), S_B3, 96)
        grounded.append(g)
    rec["grounding"] = grounded
    # normalized -> pixels, recorded BESIDE the raw so both are auditable
    rec["grounding_px"] = [norm_to_px(g["bbox"], frame_w, frame_h)
                           if g and g.get("bbox") else None for g in grounded]

    sign_desc = json.dumps([{"i": i, "kind": s.get("kind"),
                             "text": s.get("text", "")}
                            for i, s in enumerate(
                                (rec["signs"] or {}).get("signs", [])[:6])])
    sym = record("B4_symbols", P_B4.format(
        engine_a=_fmt_engine_a(engine_a), signs=sign_desc)
        + ego_section("B4_symbols", ego_state, ego_mode), S_B4, 192)
    sym, n_dup = dedupe_actions(sym)
    rec["symbols"] = sym
    rec["_n_action_dupes_dropped"] = n_dup

    # banked so the leak audit ("did B2's speed text come from the sign or from
    # the speedometer?") is computable from the artifact alone, with no re-run
    rec["ego_state"] = ego_state
    rec["_ego_prompt_mode"] = ego_mode if ego_state else "none"
    rec["_calls"] = calls
    rec["_all_valid"] = all(c["valid"] for c in calls)
    rec["_n_parse_fail"] = sum(1 for c in calls if not c["parsed_ok"])
    rec["_n_violation_fail"] = sum(1 for c in calls
                                   if c["parsed_ok"] and c["violations"])
    rec["_n_retried"] = sum(1 for c in calls if c["retried"])
    rec["_frame_wh"] = [frame_w, frame_h]
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_v2")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--ego-root", default=None)
    ap.add_argument("--arm", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ego-in-prompt", default="past", choices=EGO_MODES,
                    help="none = v2.1 prompts byte-identically (control); "
                         "past = pre-decision ego kinematics to B1/B4, "
                         "speed-REDACTED to B2 (default); "
                         "full = unredacted everywhere — the leak-measurement "
                         "arm, NOT the production setting")
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already present and all-valid in --out; "
                         "REQUIRED for any run large enough to be interrupted")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_pilot import (POSE_HZ, engine_a_for_prompt,
                           engine_a_summary, load_ego_poses,
                           sample_clip_frames)

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "ph0_v2.json")
    clips = json.load(open(a.clips))[:a.n]

    # ---- resume: a 50-clip run at ~60 s/clip is ~50 min, long enough that a
    # pod hiccup must not cost the whole thing. Only ALL-VALID clips are
    # skipped; a partial clip is redone rather than left half-scored.
    done: dict = {}
    if a.resume and os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            # ⛔ a resume across a DIFFERENT ego mode would silently mint a
            # half-and-half artifact whose clips were asked different questions.
            # Refuse the reuse rather than produce an uncomparable file.
            pm = prev.get("ego_prompt_mode")
            if pm is not None and pm != a.ego_in_prompt:
                print(f"[v2] resume REFUSED: {out_path} was run with "
                      f"ego_in_prompt={pm!r}, this run is {a.ego_in_prompt!r} "
                      f"— starting fresh", flush=True)
            else:
                for c in prev.get("clips", []):
                    if c.get("_all_valid"):
                        done[c["clip_id"]] = c
                print(f"[v2] resume: {len(done)} clips already all-valid",
                      flush=True)
        except Exception as e:
            print(f"[v2] resume: unreadable {out_path} ({e}) — starting fresh",
                  flush=True)

    todo = [c for c in clips if c not in done]
    print(f"[v2] {len(clips)} clips ({len(todo)} to run) · arm {a.arm}",
          flush=True)
    if not todo:
        print("[v2] nothing to do", flush=True)
    vlm = ConstrainedVLM(a.arm) if todo else None
    if vlm:
        print(f"[v2] auto_class {vlm.auto_class}", flush=True)

    def _write(partial):
        """Incremental write after EVERY clip — a crash at clip 47 of 50 must
        not discard 46 finished clips."""
        ok_ = [c for c in partial if c.get("_all_valid")]
        json.dump({"schema_version": SCHEMA_VERSION, "arm": a.arm,
                   "auto_class": getattr(vlm, "auto_class", None),
                   "ego_prompt_mode": a.ego_in_prompt,
                   "n": len(partial), "n_all_calls_valid": len(ok_),
                   "n_parse_failures": sum(c.get("_n_parse_fail", 0)
                                           for c in partial),
                   "n_violation_failures": sum(c.get("_n_violation_fail", 0)
                                               for c in partial),
                   "n_retried_calls": sum(c.get("_n_retried", 0)
                                          for c in partial),
                   "constrained_decoding":
                       "lm-format-enforcer JsonSchemaParser "
                       "(ws<=1, force_field_order, max_array 6)",
                   "complete": False, "clips": partial},
                  open(out_path, "w"), indent=1)

    out_clips = list(done.values())
    for ci, cid in enumerate(todo):
        vp = os.path.join(a.video_root, f"{cid}.mp4")
        try:
            frames, times, n_past = sample_clip_frames(vp, t0_s=8.0)
            ea, est = None, None
            if a.ego_root:
                try:
                    poses = load_ego_poses(cid, a.ego_root)
                    if poses is not None:
                        t0_idx = int(round(8.0 * POSE_HZ))
                        ea = engine_a_for_prompt(
                            engine_a_summary(poses, t0_idx))
                        est = ego_past_state(poses, t0_idx,
                                             dt=1.0 / POSE_HZ)
                except Exception as e:
                    print(f"[v2] engine A failed {cid}: "
                          f"{type(e).__name__}: {e}", flush=True)
            rec = run_clip(vlm, frames, n_past, ea, ego_state=est,
                           ego_mode=a.ego_in_prompt)
            rec["clip_id"] = cid
            out_clips.append(rec)
            print(f"[v2] {ci+1}/{len(todo)} {cid[:8]} "
                  f"all_valid={rec['_all_valid']} "
                  f"parse_fail={rec['_n_parse_fail']} "
                  f"viol={rec['_n_violation_fail']} "
                  f"retried={rec['_n_retried']}", flush=True)
        except Exception as e:
            out_clips.append({"clip_id": cid, "fatal": f"{type(e).__name__}: {e}"})
            print(f"[v2] {ci+1} {cid[:8]} FATAL {e}", flush=True)
        _write(out_clips)

    ok = [c for c in out_clips if c.get("_all_valid")]
    pf = sum(c.get("_n_parse_fail", 0) for c in out_clips)
    summary = {"schema_version": SCHEMA_VERSION, "arm": a.arm,
               "auto_class": getattr(vlm, "auto_class", None),
               "ego_prompt_mode": a.ego_in_prompt,
               "n": len(out_clips),
               "n_all_calls_valid": len(ok), "n_parse_failures": pf,
               "n_violation_failures": sum(c.get("_n_violation_fail", 0)
                                           for c in out_clips),
               "n_retried_calls": sum(c.get("_n_retried", 0)
                                      for c in out_clips),
               "constrained_decoding":
                   "lm-format-enforcer JsonSchemaParser "
                   "(ws<=1, force_field_order, max_array 6)",
               "complete": True, "clips": out_clips}
    p = os.path.join(a.out, "ph0_v2.json")
    json.dump(summary, open(p, "w"), indent=1)
    print(f"[v2] all-calls-valid {len(ok)}/{len(out_clips)} · "
          f"parse failures {pf}", flush=True)
    print("PH0V2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
