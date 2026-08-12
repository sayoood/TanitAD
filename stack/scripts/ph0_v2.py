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

SCHEMA_VERSION = "ph0-v2.0"

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
        "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                 "items": {"type": "integer", "minimum": 0, "maximum": 448}},
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

Locate THAT sign. Give the frame index where it is clearest and its pixel \
bounding box [x0,y0,x1,y1] in that frame. Frames are {px} px on the long side.
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
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import \
            build_transformers_prefix_allowed_tokens_fn
        self._JsonSchemaParser = JsonSchemaParser
        self._build_fn = build_transformers_prefix_allowed_tokens_fn

    def ask(self, frames, prompt: str, schema: dict, max_new: int = 256):
        """One constrained call. Returns (raw_text, parsed_or_None, err)."""
        torch = self._torch
        from PIL import Image
        pil = [Image.fromarray(f) for f in frames]
        msg = [{"role": "user", "content": [
            {"type": "video", "video": pil, "fps": 2.0},
            {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        parser = self._JsonSchemaParser(schema)
        prefix_fn = self._build_fn(self.tok, parser)
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


def run_clip(vlm, frames, n_past, engine_a, *, px=448, dump=None):
    n_last = len(frames) - 1
    calls, rec = [], {}

    def record(name, prompt, schema, max_new=256):
        t = time.time()
        raw, parsed, err = vlm.ask(frames, prompt, schema, max_new)
        calls.append({"call": name, "prompt": prompt, "raw_output": raw,
                      "parsed": parsed, "error": err,
                      "valid": parsed is not None,
                      "wall_s": round(time.time() - t, 1)})
        return parsed

    rec["scene"] = record("B1_scene", P_B1.format(
        n_past_1=n_past - 1, n_past=n_past, n_last=n_last), S_B1, 128)
    rec["signs"] = record("B2_signs", P_B2, S_B2, 256)

    grounded = []
    for i, s in enumerate((rec["signs"] or {}).get("signs", [])[:6]):
        g = record(f"B3_ground_{i}", P_B3.format(
            idx=i, kind=s.get("kind"), text=s.get("text", ""), px=px), S_B3, 96)
        grounded.append(g)
    rec["grounding"] = grounded

    sign_desc = json.dumps([{"i": i, "kind": s.get("kind"),
                             "text": s.get("text", "")}
                            for i, s in enumerate(
                                (rec["signs"] or {}).get("signs", [])[:6])])
    rec["symbols"] = record("B4_symbols", P_B4.format(
        engine_a=_fmt_engine_a(engine_a), signs=sign_desc), S_B4, 192)

    rec["_calls"] = calls
    rec["_all_valid"] = all(c["valid"] for c in calls)
    rec["_n_parse_fail"] = sum(1 for c in calls if c["error"])
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_v2")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--ego-root", default=None)
    ap.add_argument("--arm", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_pilot import (POSE_HZ, engine_a_for_prompt,
                           engine_a_summary, load_ego_poses,
                           sample_clip_frames)

    os.makedirs(a.out, exist_ok=True)
    clips = json.load(open(a.clips))[:a.n]
    print(f"[v2] {len(clips)} clips · arm {a.arm}", flush=True)
    vlm = ConstrainedVLM(a.arm)
    print(f"[v2] auto_class {vlm.auto_class}", flush=True)

    out_clips = []
    for ci, cid in enumerate(clips):
        vp = os.path.join(a.video_root, f"{cid}.mp4")
        try:
            frames, times, n_past = sample_clip_frames(vp, t0_s=8.0)
            ea = None
            if a.ego_root:
                try:
                    poses = load_ego_poses(cid, a.ego_root)
                    if poses is not None:
                        ea = engine_a_for_prompt(
                            engine_a_summary(poses, int(round(8.0 * POSE_HZ))))
                except Exception as e:
                    print(f"[v2] engine A failed {cid}: "
                          f"{type(e).__name__}: {e}", flush=True)
            rec = run_clip(vlm, frames, n_past, ea)
            rec["clip_id"] = cid
            out_clips.append(rec)
            print(f"[v2] {ci+1}/{len(clips)} {cid[:8]} "
                  f"all_valid={rec['_all_valid']} "
                  f"parse_fail={rec['_n_parse_fail']}", flush=True)
        except Exception as e:
            out_clips.append({"clip_id": cid, "fatal": f"{type(e).__name__}: {e}"})
            print(f"[v2] {ci+1} {cid[:8]} FATAL {e}", flush=True)

    ok = [c for c in out_clips if c.get("_all_valid")]
    pf = sum(c.get("_n_parse_fail", 0) for c in out_clips)
    summary = {"schema_version": SCHEMA_VERSION, "arm": a.arm,
               "auto_class": vlm.auto_class, "n": len(out_clips),
               "n_all_calls_valid": len(ok), "n_parse_failures": pf,
               "constrained_decoding": "lm-format-enforcer JsonSchemaParser",
               "clips": out_clips}
    p = os.path.join(a.out, "ph0_v2.json")
    json.dump(summary, open(p, "w"), indent=1)
    print(f"[v2] all-calls-valid {len(ok)}/{len(out_clips)} · "
          f"parse failures {pf}", flush=True)
    print("PH0V2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
