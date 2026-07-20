"""VLM labels from FUTURE FRAMES — Cosmos-Reason2-8B, two passes (A/B).

WHY THIS EXISTS. The kinematic labeler can only ever restate the ego track: it
sees a heading change and calls it a turn. It cannot see the junction the driver
turned INTO, the sign that capped the speed, or the lead vehicle that caused the
brake. A VLM shown the FUTURE FRAMES can — the future frames literally contain
the outcome the label is supposed to name. This module asks it, in our frozen
vocabulary (V3_GOAL_VOCABULARY_V1.md), and stamps every answer with provenance.

THE TWO PASSES, AND WHY THE SPLIT IS LOAD-BEARING
  PASS A — INDEPENDENT ROUTE EVIDENCE. Inputs: history frames, an ego-motion
      block computed from the PAST ONLY, and the FUTURE FRAMES. It is NOT given
      the numeric future ego track. Its ROUTE is therefore evidence gathered
      independently of our kinematics, and is the ONLY thing that may enter the
      VLM-vs-kinematic agreement statistics. Hand the model our computed net
      heading change and it will parrot it back; the cross-validation would then
      measure nothing.
  PASS B — RICH INTERPRETATION. Same inputs PLUS a plain-language summary of the
      numeric future ego track ("turns left 78 deg beginning at t+11.0 s,
      decelerating to 3 m/s"). Used for the scenario/semantic slots where
      independence does not matter and interpretability does. Pass B ROUTE is
      recorded but MUST NOT be mixed into the agreement numbers — the writer
      tags every record with its pass and the cross-validator filters on it.

WHAT THE VLM IS NOT ASKED FOR. Metric numbers it cannot measure. The 48-clip
pilot showed it fabricating VTARGET band edges on 48 % of clips, so VTARGET and
HEADWAY stay KINEMATIC: the VLM supplies the CAP and the JUSTIFICATION (the
speed-limit sign value it actually read, and VSOURCE) while kinematics compute
the band and the seconds. Every categorical answer is a SELECTION from an
explicit enum shipped in the prompt; free text is confined to the evidence and
CoC fields and is validated away everywhere else.

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack python scripts/vlm_route_labels.py \
    --val /workspace/pai_epcache/physicalai-val-f1b378f295ae \
    --out /workspace/vlm_route --episodes 0-79 --stride 40 --pass both
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import refb_labels as R  # noqa: E402

MODEL_ID = os.environ.get("VLM_MODEL", "nvidia/Cosmos-Reason2-8B")
PROMPT_VERSION = "vlmroute-2026-07-20-a"
DT = 0.1
HIST_S = (-3.0, -1.5, 0.0)                  # history frames offered, seconds
FUT_S = (2.0, 5.0, 10.0, 15.0, 20.0)        # future frames offered, seconds

# ---- enums the VLM may SELECT from (R1: discrete, no free text) -------------
ROUTE_ENUM = ("left", "straight", "right", "u_turn", "unknown")
ENUMS_SCENARIO = {
    "road_type": ("highway", "urban_arterial", "residential", "rural",
                  "intersection", "parking", "ramp", "unknown"),
    "weather": ("clear", "cloudy", "rain", "snow", "fog", "unknown"),
    "time_of_day": ("day", "dusk", "night", "dawn", "unknown"),
    "lighting": ("bright", "normal", "low", "glare", "dark", "unknown"),
    "surface": ("dry", "wet", "snow_ice", "gravel", "unknown"),
    "traffic_density": ("empty", "light", "moderate", "heavy", "congested",
                        "unknown"),
    "road_geometry": ("straight", "curve_left", "curve_right", "junction",
                      "roundabout", "merge", "fork", "unknown"),
    "scenario_tag": ("cut_in", "lead_brake", "pedestrian_crossing", "cyclist",
                     "parked_vehicle_pass", "construction", "emergency_vehicle",
                     "blocked_lane", "oncoming_encroach", "unprotected_turn",
                     "traffic_light_stop", "yield_merge", "none"),
    "difficulty": ("routine", "notable", "hard", "anomalous"),
}
ENUMS_OBS = {
    "lead_lane": ("ego", "left", "right", "oncoming", "none"),
    "distance_bucket": ("very_close", "close", "medium", "far", "none"),
    "relative_motion": ("closing", "steady", "opening", "stopped", "none"),
    "agent_type": ("car", "truck", "bus", "motorcycle", "cyclist", "pedestrian",
                   "animal", "other"),
    "agent_position": ("ahead", "ahead_left", "ahead_right", "left", "right",
                       "behind", "oncoming", "crossing"),
    "relevance": ("critical", "relevant", "background"),
    "markings": ("solid", "dashed", "double", "none", "unknown"),
    "lane_type": ("normal", "turn_only", "bus", "bike", "shoulder", "exit",
                  "unknown"),
    "light_state": ("red", "amber", "green", "off", "none", "unknown"),
    "sign_type": ("speed_limit", "stop", "yield", "no_entry", "warning",
                  "direction", "other"),
}


def _tac(slot):
    from tanitad.lake import vocab as V
    return list(V.TACTICAL_TOKENS[slot])


def _strat(slot):
    from tanitad.lake import vocab as V
    return list(V.STRATEGIC_TOKENS[slot])


# ---------------------------------------------------------------- ego motion
def ego_block(poses: torch.Tensor, t: int, hist_s: float = 3.0) -> dict:
    """PAST-ONLY ego motion at ``t`` — the conditioning block both passes get.

    Past-only is deliberate: it is the state the driver was in when the decision
    was made, and it leaks nothing about the future the model is being asked to
    read off the frames."""
    n = int(hist_s / DT)
    a = max(0, t - n)
    seg = poses[a:t + 1]
    v = seg[:, 3]
    v0 = float(poses[t, 3])
    dv = float(v[-1] - v[0]) if v.numel() > 1 else 0.0
    secs = max((seg.shape[0] - 1) * DT, DT)
    accel = dv / secs
    yr = (float(R.wrap_to_pi(poses[t, 2] - poses[t - 1, 2]) / DT)
          if t >= 1 else 0.0)
    lon = ("accelerating" if accel > 0.4 else
           "decelerating" if accel < -0.4 else "holding speed")
    steer = ("steering left" if yr > 0.08 else
             "steering right" if yr < -0.08 else "steering neutral")
    return {
        "v0_mps": round(v0, 2),
        "speed_profile_mps": [round(float(x), 1) for x in v[::5].tolist()],
        "lon_accel_mps2": round(accel, 2),
        "yaw_rate_radps": round(yr, 3),
        "summary": f"ego at {v0:.1f} m/s, {lon}, {steer}",
    }


def future_track_summary(poses: torch.Tensor, t: int) -> dict:
    """PASS B ONLY — plain-language reading of the NUMERIC future ego track.

    Never included in a Pass A prompt. `route_event_time_s` here is derived from
    the curvature peak, so Pass B's ROUTE is downstream of our kinematics by
    construction; that is exactly why Pass B is excluded from the agreement
    statistics."""
    T = poses.shape[0]
    h = min(R.NAV_HORIZON_STEPS, T - 1 - t)
    if h < 1:
        return {"available": False, "summary": "no future track available"}
    seg = poses[t:t + h + 1]
    step_dyaw = R.wrap_to_pi(seg[1:, 2] - seg[:-1, 2])
    net_deg = math.degrees(float(step_dyaw.sum()))
    ds = (seg[1:, :2] - seg[:-1, :2]).norm(dim=-1)
    kappa = torch.where(ds >= R.MIN_ARC_M,
                        step_dyaw / ds.clamp_min(R.MIN_ARC_M),
                        torch.zeros_like(ds))
    ks = R._moving_avg(kappa, R._arc_smooth_k(ds))
    onset = int(ks.abs().argmax()) if ks.numel() else 0
    v_end, v0 = float(seg[-1, 3]), float(seg[0, 3])
    turn = ("turns left" if net_deg > 10 else
            "turns right" if net_deg < -10 else "continues roughly straight")
    spd = ("accelerating" if v_end - v0 > 1.0 else
           "decelerating" if v_end - v0 < -1.0 else "holding")
    return {
        "available": True,
        "net_heading_deg": round(net_deg, 1),
        "route_event_time_s": round(onset * DT, 1),
        "arc_m": round(float(ds.sum()), 1),
        "v_end_mps": round(v_end, 1),
        "horizon_s": round(h * DT, 1),
        "summary": (f"over the next {h * DT:.1f} s the ego {turn} "
                    f"{abs(net_deg):.0f} deg beginning at t+{onset * DT:.1f} s, "
                    f"{spd} to {v_end:.1f} m/s"),
    }


# ------------------------------------------------------------------ prompts
_RULES = """You are labeling a driving clip for an autonomous-driving dataset.
You are shown HISTORY frames (what the car had just seen) and FUTURE frames
(what actually happened next, in order). Use the FUTURE FRAMES as evidence: they
show the outcome.

HARD RULES
1. Answer with a SINGLE JSON object and nothing else. No prose outside it.
2. Every categorical field must be copied EXACTLY from the allowed list given
   for it. Never invent a token, never reword one.
3. Never invent a number you cannot see. If you did not read a speed-limit sign,
   say so; do not guess the limit. Distances are BUCKETS, not metres.
4. If the evidence does not support a field, use "unknown". "unknown" is a
   correct answer and is preferred over a plausible guess.
5. Confidences are 0.0-1.0 floats reflecting what the FRAMES support."""

PASS_A_TASK = """TASK (route intent only).
From the FUTURE FRAMES, determine where the vehicle actually goes.

ROUTE must be one of: {route_enum}
  left / right  = the vehicle leaves its current road at a junction, ramp or
                  driveway, or makes a discrete turn onto a different road
  straight      = the vehicle stays on its current road, INCLUDING following a
                  bend or curve in that road
  u_turn        = reverses direction onto the opposing carriageway
  unknown       = the future frames do not show enough to tell

Return exactly:
{{"ROUTE": "<token>",
 "route_confidence": <float>,
 "route_evidence": "<one sentence naming what you SAW in the future frames -
                    e.g. 'the road ahead ends at a T-junction and frames 4-5
                    show the view swinging to face a cross street'>",
 "route_event_time_s": <float or null: seconds from now at which the maneuver
                        begins, judged from WHICH future frame it appears in>,
 "road_geometry": "<one of: {geom}>",
 "sees_junction_ahead": <true|false>}}"""

PASS_B_TASK = """TASK (full scene interpretation).
Return exactly this JSON structure, selecting every token from its list:

{{"SCENARIO": {{
   "road_type": "<{road_type}>",
   "environment": {{"weather": "<{weather}>", "time_of_day": "<{time_of_day}>",
                    "lighting": "<{lighting}>"}},
   "surface": "<{surface}>",
   "traffic_density": "<{traffic_density}>",
   "road_geometry": "<{road_geometry}>",
   "scenario_tag": "<{scenario_tag}>",
   "odd_flags": ["<free list of ODD concerns, may be empty>"],
   "difficulty": "<{difficulty}>"}},
 "STRATEGIC": {{
   "ROUTE": "<{route_enum}>", "route_confidence": <float>,
   "route_evidence": "<one sentence citing what you saw>",
   "route_event_time_s": <float or null>,
   "MISSION": "<{MISSION}>", "LANEOBJ": "<{LANEOBJ}>",
   "SPEEDPOLICY": "<{SPEEDPOLICY}>", "STYLE": "<{STYLE}>",
   "RISK": "<{RISK}>", "ODD": "<{ODD}>"}},
 "TACTICAL": {{
   "LATMANEUVER": "<{LATMANEUVER}>", "LONMODE": "<{LONMODE}>",
   "VSOURCE": "<{VSOURCE}>", "HEADWAY": "<{HEADWAY}>", "DYN": "<{DYN}>",
   "RULECTX": "<{RULECTX}>", "SIGNAL": "<{SIGNAL}>", "INTERACT": "<{INTERACT}>",
   "TACPOINT": "<{TACPOINT}>", "LIGHTSTATE": "<{LIGHTSTATE}>"}},
 "OBSERVATIONS": {{
   "sign_reads": [{{"type": "<{sign_type}>", "value": <number or null>,
                    "unit": "<kph|mph|none>", "confidence": <float>}}],
   "lead_vehicle": {{"present": <true|false>, "lane": "<{lead_lane}>",
                     "distance_bucket": "<{distance_bucket}>",
                     "relative_motion": "<{relative_motion}>"}},
   "critical_agents": [{{"type": "<{agent_type}>",
                         "position": "<{agent_position}>",
                         "behavior": "<short phrase>",
                         "relevance": "<{relevance}>"}}],
   "lane_info": {{"ego_lane_index": <int or null>, "n_lanes": <int or null>,
                  "markings": "<{markings}>", "lane_type": "<{lane_type}>"}},
   "traffic_light": {{"present": <true|false>, "state": "<{light_state}>",
                      "applies_to_ego": <true|false>}}}},
 "COC": {{
   "observation": "<what is visible that matters, one sentence>",
   "inference": "<what it implies about the situation, one sentence>",
   "decision": "<what the ego should therefore do, one sentence>"}}}}

VSOURCE is YOURS to justify: say WHY the set-speed is what it is (a sign you
read, a lead vehicle, a curve, the road class, or the traffic flow). Do NOT
state a target speed in m/s — kinematics own that number.
HEADWAY is a qualitative bucket only; do not compute seconds."""


def build_prompt(which: str) -> str:
    from tanitad.lake import vocab as V
    fmt = {"route_enum": " | ".join(ROUTE_ENUM),
           "geom": " | ".join(ENUMS_SCENARIO["road_geometry"])}
    if which == "A":
        return PASS_A_TASK.format(**fmt)
    for k, v in ENUMS_SCENARIO.items():
        fmt[k] = " | ".join(v)
    for k, v in ENUMS_OBS.items():
        fmt[k] = " | ".join(v)
    for k in ("MISSION", "LANEOBJ", "SPEEDPOLICY", "STYLE", "RISK", "ODD"):
        fmt[k] = " | ".join(V.STRATEGIC_TOKENS[k])
    for k in ("LATMANEUVER", "LONMODE", "VSOURCE", "HEADWAY", "DYN", "RULECTX",
              "SIGNAL", "INTERACT", "TACPOINT", "LIGHTSTATE"):
        fmt[k] = " | ".join(V.TACTICAL_TOKENS[k])
    return PASS_B_TASK.format(**fmt)


# ------------------------------------------------------------------- frames
def pick_frames(ep_frames, t: int, T: int):
    """(history imgs, future imgs, caption lines). Only frames that EXIST are
    offered, and the caption says which — a late-clip window genuinely has less
    future and the model must not be told otherwise."""
    hist, fut, lines = [], [], []
    for s in HIST_S:
        i = int(round(t + s / DT))
        if 0 <= i < T:
            hist.append(i)
            lines.append(f"  history frame at t{s:+.1f} s")
    for s in FUT_S:
        i = int(round(t + s / DT))
        if 0 <= i < T:
            fut.append(i)
            lines.append(f"  FUTURE frame at t{s:+.1f} s")
    return hist, fut, lines


def to_pil(ep_frames, i: int):
    from PIL import Image
    arr = ep_frames[i, -3:].permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(arr).resize((448, 448), Image.BICUBIC)


# -------------------------------------------------------------------- model
class Cosmos:
    def __init__(self, model_id=MODEL_ID, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoProcessor, AutoModelForCausalLM
        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(model_id)
        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except ImportError:                     # loud, not a silent fallback
            Cls = AutoModelForCausalLM
        self.model = Cls.from_pretrained(model_id, dtype=dtype,
                                         device_map=device).eval()
        self.model_id = model_id
        print(f"[vlm] loaded {model_id} in {time.time() - t0:.0f}s "
              f"dtype={dtype}", flush=True)

    @torch.no_grad()
    def ask(self, images, text, max_new_tokens=1400):
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": text})
        msgs = [{"role": "user", "content": content}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.proc.decode(gen, skip_special_tokens=True)


def extract_json(s: str):
    """Last balanced {...} in the reply. Cosmos-Reason emits a reasoning
    preamble; the answer is the final object."""
    depth, start, best = 0, None, None
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                best = s[start:i + 1]
    if best is None:
        return None
    for cand in (best, re.sub(r",\s*([}\]])", r"\1", best)):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def coerce(val, allowed, default="unknown"):
    """Snap a returned token onto the enum; anything else becomes `default` and
    is COUNTED as a violation by the caller (never silently accepted)."""
    if isinstance(val, str):
        v = val.strip().lower()
        if v in allowed:
            return v, True
        for a in allowed:
            if v.replace(" ", "_") == a:
                return a, True
    return (default if default in allowed else allowed[-1]), False


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser("vlm_route_labels")
    ap.add_argument("--val", default="/workspace/pai_epcache/"
                                     "physicalai-val-f1b378f295ae")
    ap.add_argument("--out", default="/workspace/vlm_route")
    ap.add_argument("--episodes", default="0-79")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--passes", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--limit-windows", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.episodes.split("-")) \
        if "-" in args.episodes else (int(args.episodes), int(args.episodes))
    files = sorted(glob.glob(os.path.join(args.val, "ep_*.pt")))[lo:hi + 1]
    os.makedirs(args.out, exist_ok=True)
    vlm = Cosmos(args.model)
    prompt_a, prompt_b = build_prompt("A"), build_prompt("B")
    n_done = n_bad = 0
    t_start = time.time()

    for f in files:
        ep_name = os.path.basename(f).replace(".pt", "")
        d = torch.load(f, map_location="cpu", weights_only=False)
        frames, poses = d["frames_u8"], d["poses"].float()
        T = min(frames.shape[0], poses.shape[0])
        for t in range(args.t0, T, args.stride):
            dst = os.path.join(args.out, f"{ep_name}_t{t:04d}.json")
            if os.path.exists(dst):
                continue
            hist, fut, lines = pick_frames(frames, t, T)
            if not fut:
                continue                        # nothing to see -> nothing to ask
            eb = ego_block(poses, t)
            imgs = [to_pil(frames, i) for i in hist + fut]
            head = (f"{_RULES}\n\nFRAMES PROVIDED (in order):\n"
                    + "\n".join(lines)
                    + f"\n\nEGO MOTION NOW (measured, past only):\n"
                      f"{json.dumps(eb)}\n{eb['summary']}\n")
            rec = {"episode": ep_name, "t": t, "model": vlm.model_id,
                   "prompt_version": PROMPT_VERSION, "provenance": "vlm",
                   "ego_block": eb, "n_future_frames": len(fut),
                   "future_frame_times_s": [FUT_S[i] for i in range(len(fut))],
                   "clip_len": int(T)}
            try:
                if "A" in args.passes:
                    raw = vlm.ask(imgs, head + "\n" + prompt_a, 1000)
                    js = extract_json(raw) or {}
                    route, ok = coerce(js.get("ROUTE"), ROUTE_ENUM)
                    n_bad += (not ok)
                    rec["pass_A"] = {
                        "pass": "A", "future_track_given": False,
                        "ROUTE": route, "enum_ok": ok,
                        "route_confidence": js.get("route_confidence"),
                        "route_evidence": js.get("route_evidence"),
                        "route_event_time_s": js.get("route_event_time_s"),
                        "road_geometry": coerce(
                            js.get("road_geometry"),
                            ENUMS_SCENARIO["road_geometry"])[0],
                        "sees_junction_ahead": js.get("sees_junction_ahead"),
                        "raw_len": len(raw)}
                if "B" in args.passes:
                    fts = future_track_summary(poses, t)
                    raw = vlm.ask(imgs, head + "\nNUMERIC FUTURE EGO TRACK "
                                  "(measured):\n" + fts["summary"] + "\n\n"
                                  + prompt_b, 2200)
                    js = extract_json(raw) or {}
                    js["pass"] = "B"
                    js["future_track_given"] = True
                    js["future_track"] = fts
                    rec["pass_B"] = js
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"[err] {ep_name} t={t}: {rec['error']}", flush=True)
            with open(dst, "w") as fh:
                json.dump(rec, fh, indent=1)
            n_done += 1
            if n_done % 5 == 0:
                el = time.time() - t_start
                print(f"[{n_done}] {ep_name} t={t} "
                      f"A={rec.get('pass_A', {}).get('ROUTE')} "
                      f"{el / n_done:.1f}s/window enum_violations={n_bad}",
                      flush=True)
            if args.limit_windows and n_done >= args.limit_windows:
                print(f"VLM_ROUTE_LABELS_DONE windows={n_done} "
                      f"enum_violations={n_bad}", flush=True)
                return
    print(f"VLM_ROUTE_LABELS_DONE windows={n_done} enum_violations={n_bad}",
          flush=True)


if __name__ == "__main__":
    main()
