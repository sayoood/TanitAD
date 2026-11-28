#!/usr/bin/env python3
"""TanitDataSet VLM augmentation pilot — v3-vocabulary labeling pass.

Two schema-locked passes per clip against a local VLM (default nvidia/Cosmos-Reason1-7B):
  PASS A (perception)  -> scene_tags + sign_reads + lead_state
  PASS B (tactical+CoC)-> v3 TACTICAL slots + chain-of-causation trace

Every emitted token is validated against the FROZEN v3 goal vocabulary
(V3_GOAL_VOCABULARY_V1.md). Out-of-vocabulary -> recorded as a violation and
coerced to "unknown" (rule R3: honest gaps are stamped, never guessed).

Non-destructive: reads /root/vlm_pilot/manifest.json, writes /root/vlm_pilot/out/.
Checkpoints per clip so an interruption loses at most one clip.
"""
import argparse, glob, json, os, re, sys, time

OUT_DIR = "/root/vlm_pilot/out"
MANIFEST = "/root/vlm_pilot/manifest.json"
PROMPT_VERSION = "v3vocab-2026-07-20-a"

# ---------------------------------------------------------------- vocabulary
def _vtarget_tokens():
    t = ["v_stop"] + ["v(%d-%d]" % (i, i + 1) for i in range(0, 10)]
    lo = 10.0
    while lo < 40.0:
        hi = lo + 2.5
        f = lambda x: ("%g" % x)
        t.append("v(%s-%s]" % (f(lo), f(hi)))
        lo = hi
    return t

VOCAB = {
    "VTARGET": _vtarget_tokens(),
    "VSOURCE": ["sign_limit", "lead_constrained", "curve_constrained",
                "road_class_default", "traffic_flow"],
    "LONMODE": ["free_cruise", "follow_lead", "close_gap", "open_gap", "stop_at_point",
                "hold_stop", "launch", "creep", "coast"],
    "LATMANEUVER": ["lane_keep", "lc_left", "lc_right", "abort_lc", "merge_in",
                    "yield_merge", "nudge_left", "nudge_right", "pull_over"],
    "HEADWAY": ["hw_0.8s", "hw_1.2s", "hw_1.45s", "hw_1.75s", "hw_2.5s+"],
    "DYN": ["gentle", "normal", "firm", "max"],
    "RULECTX": ["conform", "justified_deviation_obstacle_avoidance",
                "justified_deviation_rescue_corridor",
                "justified_deviation_stopped_vehicle_pass",
                "justified_deviation_instructed"],
    "SIGNAL": ["none", "indicator_left", "indicator_right", "hazard",
               "headlight_flash", "horn"],
    "INTERACT": ["none", "yield_to_lead", "yield_to_merger", "assert_gap_lead",
                 "assert_gap_merger", "cooperate_merge_lead", "cooperate_merge_merger",
                 "respond_emergency"],
    "TACPOINT": ["none", "stop_line", "merge_point", "creep_point", "clear_point"],
    "LIGHTSTATE": ["proceed", "prepare_stop", "stop_at_line", "creep_check"],
    "RISK": ["nominal", "elevated_weather", "elevated_visibility", "elevated_anomaly"],
    "ODD": ["in_odd", "odd_exit_ahead", "capability_degrading"],
}
SCENE_ENUMS = {
    "weather": ["clear", "overcast", "rain", "snow", "fog"],
    "time_of_day": ["dawn", "day", "dusk", "night"],
    "road_type": ["highway", "urban", "suburban", "rural", "intersection", "parking"],
    "surface": ["dry", "wet", "snow", "unknown"],
    "traffic_density": ["empty", "light", "moderate", "heavy"],
}

# ---------------------------------------------------------------- prompts
SYS = ("You are an expert autonomous-driving scene annotator for a safety-critical "
       "dataset. You output ONLY a single JSON object, no prose, no markdown fence. "
       "You annotate ONLY what is visually evident in the given forward-camera frames. "
       "When the frames do not support a field, you output \"unknown\" - you never "
       "guess and never invent agents, signs, or conditions that are not visible.")

PROMPT_A = """These are {n} forward-camera keyframes sampled in order across one ~5 s driving clip.

Return ONE JSON object with EXACTLY this schema:
{{
 "scene_tags": {{
   "weather": one of {weather},
   "time_of_day": one of {tod},
   "road_type": one of {road},
   "surface": one of {surface},
   "traffic_density": one of {traffic},
   "vru_present": true or false,
   "notable_events": [ up to 3 strings, each <=5 words, rare/safety-relevant only ]
 }},
 "sign_reads": [ {{"type": one of ["speed_limit","stop","yield","no_entry","school_zone","other"],
                  "value_kph": integer or null,
                  "confidence": number 0-1}} ],
 "lead_state": {{
   "present": true or false,
   "gap_band": one of ["near","mid","far","unknown"],
   "closing": one of ["closing","steady","opening","unknown"],
   "lead_type": one of ["car","truck","bus","motorcycle","bicycle","none","unknown"]
 }}
}}

RULES
- "sign_reads": include ONLY signs whose text/symbol you can actually read in the frames.
  Return [] if none are legible. NEVER infer a limit from road type or country.
- "lead_state.present" is true ONLY if a vehicle is in the ego lane ahead.
- "notable_events": [] when the scene is unremarkable. Do not narrate normal driving.
- Output the JSON object only."""

PROMPT_B = """These are {n} forward-camera keyframes sampled in order across one ~5 s driving clip.

MEASURED EGO KINEMATICS (from vehicle pose, trustworthy - not your estimate):
{kin}

PERCEPTION PASS (your own earlier output for this clip):
{percep}

Assign this clip's TACTICAL driving goal using the FROZEN v3 vocabulary. Every value
MUST be copied verbatim from the allowed list for that slot, or be "unknown".

{{
 "VTARGET": one of {VTARGET},
 "VSOURCE": one of {VSOURCE},
 "LONMODE": one of {LONMODE},
 "LATMANEUVER": one of {LATMANEUVER},
 "HEADWAY": one of {HEADWAY},
 "DYN": one of {DYN},
 "TACPOINT": one of {TACPOINT},
 "LIGHTSTATE": one of {LIGHTSTATE},
 "INTERACT": one of {INTERACT},
 "RISK": one of {RISK},
 "coc_trace": {{
   "observation": "what is actually visible that constrains the ego (<=40 words)",
   "critical_agents": [ {{"agent":"short noun phrase","why_critical":"<=12 words"}} ],
   "justification": "the causal reason this goal fits the observation (<=35 words)",
   "decision": "high-level intent in plain words (<=12 words)",
   "physics_flag": "note any physically implausible element, else null"
 }}
}}

RULES
- Copy each token EXACTLY as written in its allowed list: no trailing comma, no quotes
  inside the value, no added units or punctuation.
- VTARGET is the SET-SPEED the ego should hold, as a band. Use the measured speed as the
  anchor when the ego is in free flow; use a LOWER band when a lead, curve, or sign
  constrains it. "v_stop" only when the intent is to be stopped.
- VSOURCE must state WHY that set-speed: "sign_limit" ONLY if a legible limit sign was
  read in the perception pass; "lead_constrained" if a lead vehicle governs;
  "curve_constrained" if road curvature governs; else "road_class_default" or "traffic_flow".
- HEADWAY is "unknown" when no lead vehicle is present.
- INTERACT is "none" unless a specific agent is actively being yielded to or negotiated with.
- coc_trace: do NOT restate the action as its own justification. Do NOT invent agents.
  If the scene is unremarkable, say so plainly.
- Output the JSON object only."""


# ---------------------------------------------------------------- json parse
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

def parse_json(txt):
    """Best-effort strict-ish JSON extraction. Returns (obj, how) or (None, reason)."""
    if not txt:
        return None, "empty"
    s = txt.strip()
    m = FENCE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s), "clean"
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        frag = s[i:j + 1]
        try:
            return json.loads(frag), "braces"
        except Exception:
            frag2 = re.sub(r",(\s*[}\]])", r"\1", frag)  # trailing commas
            try:
                return json.loads(frag2), "repaired"
            except Exception as e:
                return None, "unparseable:%s" % str(e)[:60]
    return None, "no-object"


def _norm(v):
    """Strip list-formatting debris the model copies from the prompt (trailing comma,
    quotes, stray period/space). The underlying token stays unambiguous."""
    return str(v).strip().strip(",").strip().strip("\"'").strip().rstrip(".").strip()


def validate(goal):
    """Check every v3 slot against the frozen vocabulary.

    Returns (clean, violations, normalized) where `normalized` lists slots that only
    matched after cosmetic cleanup - so strict vs effective adherence stay separable.
    """
    viol, clean, normd = [], {}, []
    for slot, allowed in VOCAB.items():
        if slot not in goal:
            continue
        v = goal.get(slot)
        if v is None:
            v = "unknown"
        raw = str(v).strip()
        if raw == "unknown" or raw in allowed:
            clean[slot] = raw
            continue
        n = _norm(raw)
        if n in allowed or n == "unknown":
            clean[slot] = n
            normd.append({"slot": slot, "emitted": raw[:60], "normalized_to": n})
            continue
        lower = {a.lower(): a for a in allowed}
        if n.lower() in lower:
            clean[slot] = lower[n.lower()]
            normd.append({"slot": slot, "emitted": raw[:60], "normalized_to": lower[n.lower()]})
            continue
        viol.append({"slot": slot, "emitted": raw[:60]})
        clean[slot] = "unknown"
    return clean, viol, normd


def validate_scene(tags):
    viol, clean = [], {}
    for k, allowed in SCENE_ENUMS.items():
        v = tags.get(k, "unknown")
        v = "unknown" if v is None else str(v).strip()
        n = _norm(v)
        lower = {a.lower(): a for a in allowed}
        if v == "unknown" or v in allowed:
            clean[k] = v
        elif n in allowed or n == "unknown":
            clean[k] = n
        elif n.lower() in lower:
            clean[k] = lower[n.lower()]
        else:
            viol.append({"slot": "scene." + k, "emitted": v[:40]})
            clean[k] = "unknown"
    clean["vru_present"] = bool(tags.get("vru_present", False))
    ne = tags.get("notable_events") or []
    clean["notable_events"] = [str(x)[:60] for x in ne][:3] if isinstance(ne, list) else []
    return clean, viol


# ---------------------------------------------------------------- model
class VLM:
    def __init__(self, model_id, max_new_tokens=768):
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        print("[load] %s" % model_id, flush=True)
        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(model_id)
        # NOTE: load on CPU then .to("cuda") -- device_map= would require `accelerate`,
        # and this pod is shared with live TanitEval jobs, so we install nothing.
        try:
            from transformers import AutoModelForImageTextToText as M
            self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
        except Exception as e:
            print("[load] AutoModelForImageTextToText failed (%s); trying Qwen2_5_VL" % str(e)[:160])
            from transformers import Qwen2_5_VLForConditionalGeneration as M
            self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
        self.model.eval()
        self.load_s = time.time() - t0
        mem = torch.cuda.memory_allocated() / 1e9
        print("[load] ok in %.1fs, weights on GPU ~%.2f GB" % (self.load_s, mem), flush=True)

    def ask(self, images, prompt, system=SYS):
        from PIL import Image
        pil = [Image.open(p).convert("RGB") for p in images]
        content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": prompt}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=pil, return_tensors="pt").to(self.model.device)
        n_in = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.proc.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[-1]:]
        txt = self.proc.tokenizer.decode(gen, skip_special_tokens=True)
        return txt, n_in, int(gen.shape[-1])


def kin_block(k):
    if not k or "v_mean" not in k:
        return "  (ego pose unavailable for this clip - treat speed as unknown)"
    return ("  mean speed %.1f m/s (%.0f km/h); start %.1f m/s; end %.1f m/s;\n"
            "  min %.1f / max %.1f m/s; net heading change %.1f deg over %.0f s;\n"
            "  mean accel %.2f m/s^2" % (
                k["v_mean"], k["v_mean"] * 3.6, k["v_start"], k["v_end"],
                k["v_min"], k["v_max"], k["yaw_change_deg"], k["dur_s"], k["accel_mean"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/Cosmos-Reason1-7B")
    ap.add_argument("--tag", default="reason1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", default="")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()

    outdir = os.path.join(OUT_DIR, args.tag)
    os.makedirs(outdir, exist_ok=True)
    manifest = json.load(open(MANIFEST))
    if args.clips:
        want = set(args.clips.split(","))
        manifest = [m for m in manifest if m["clip_id"] in want or
                    any(m["clip_id"].startswith(w) for w in want)]
    if args.limit:
        manifest = manifest[:args.limit]
    print("clips to label: %d -> %s" % (len(manifest), outdir), flush=True)

    vlm = VLM(args.model, args.max_new_tokens)
    fmtA = dict(weather=SCENE_ENUMS["weather"], tod=SCENE_ENUMS["time_of_day"],
                road=SCENE_ENUMS["road_type"], surface=SCENE_ENUMS["surface"],
                traffic=SCENE_ENUMS["traffic_density"])
    t_start = time.time()
    done = 0
    for idx, m in enumerate(manifest):
        cid = m["clip_id"]
        dst = os.path.join(outdir, cid + ".json")
        if os.path.exists(dst):
            print("[%2d] skip (exists) %s" % (idx, cid[:34]), flush=True)
            done += 1
            continue
        t0 = time.time()
        frames = m["frames"]
        rec = {"clip_id": cid, "source": "cosmos-drive-dreams (CC-BY-4.0)",
               "pose_key": m["pose_key"], "n_frames": len(frames),
               "render_condition_gt": m.get("weather_filename_gt"),
               "kinematics": m.get("kinematics"),
               "label_stamp": {"model": args.model, "prompt_version": PROMPT_VERSION,
                               "provenance": "vlm", "source_license": "CC-BY-4.0",
                               "labeled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
        viol_all, norm_all, tok_in, tok_out = [], [], 0, 0
        try:
            # ---- PASS A
            pa = PROMPT_A.format(n=len(frames), **fmtA)
            rawA, ni, no = vlm.ask(frames, pa)
            tok_in += ni; tok_out += no
            objA, howA = parse_json(rawA)
            rec["pass_a_parse"] = howA
            if objA is None:
                rec["pass_a_raw"] = rawA[:600]
                viol_all.append({"slot": "pass_a", "emitted": howA})
                scene, signs, lead = {}, [], {}
            else:
                scene, sv = validate_scene(objA.get("scene_tags", {}) or {})
                viol_all += sv
                signs = objA.get("sign_reads", []) or []
                if not isinstance(signs, list):
                    signs = []
                lead = objA.get("lead_state", {}) or {}
            rec["scene_tags"] = scene
            rec["sign_reads"] = signs
            rec["lead_state"] = lead

            # ---- PASS B
            percep = json.dumps({"scene_tags": scene, "sign_reads": signs,
                                 "lead_state": lead}, ensure_ascii=False)
            pb = PROMPT_B.format(n=len(frames), kin=kin_block(m.get("kinematics")),
                                 percep=percep, **{k: v for k, v in VOCAB.items()})
            rawB, ni, no = vlm.ask(frames, pb)
            tok_in += ni; tok_out += no
            objB, howB = parse_json(rawB)
            rec["pass_b_parse"] = howB
            if objB is None:
                rec["pass_b_raw"] = rawB[:600]
                viol_all.append({"slot": "pass_b", "emitted": howB})
                goal, coc = {}, {}
            else:
                goal, gv, gn = validate(objB)
                viol_all += gv
                norm_all += gn
                coc = objB.get("coc_trace", {}) or {}
            rec["goal_tactical"] = goal
            rec["coc_trace"] = coc
        except Exception as e:
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            print("  ERROR %s" % rec["error"], flush=True)
        rec["violations"] = viol_all
        rec["normalized"] = norm_all
        # strict = model copied every token verbatim; effective = valid after cosmetic cleanup
        rec["schema_ok_strict"] = (not viol_all and not norm_all and "error" not in rec)
        rec["schema_ok"] = (len(viol_all) == 0 and "error" not in rec)
        rec["timing"] = {"seconds": round(time.time() - t0, 2),
                         "tokens_in": tok_in, "tokens_out": tok_out}
        with open(dst, "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        done += 1
        print("[%2d] %s ok=%s viol=%d %.1fs | %s / %s / %s" % (
            idx, cid[:30], rec["schema_ok"], len(viol_all), rec["timing"]["seconds"],
            rec.get("goal_tactical", {}).get("LONMODE"),
            rec.get("goal_tactical", {}).get("VTARGET"),
            rec.get("scene_tags", {}).get("weather")), flush=True)

    el = time.time() - t_start
    print("\nDONE %d clips in %.1f min -> %.1f clips/GPU-hr" % (
        done, el / 60.0, done / (el / 3600.0) if el > 0 else 0), flush=True)


if __name__ == "__main__":
    main()
