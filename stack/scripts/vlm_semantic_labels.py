"""PRODUCTION semantic labeling with Cosmos-Reason2-8B — prompt v2.

WHY THIS EXISTS, AND WHY IT IS A NEW MODULE. `vlm_route_labels.py` holds prompt
`vlmroute-2026-07-20-a`, the exact bytes the Reason1-vs-Reason2 head-to-head was
measured on. `vlm_model_compare.swap_route_order` even *raises* if that text
changes. So the v1 prompt is frozen evidence and is imported, never edited; the
v2 prompt lives here and stamps its own `PROMPT_VERSION`. Old records stay
interpretable because the version string on each record says which prompt
produced it, and `--prompt-version v1` still runs the old one.

WHAT V2 CHANGED, AND WHAT EACH CHANGE ACTUALLY MEASURED (2026-07-21 campaign —
read the OUTCOMES, not just the intentions; two of the four fixes failed)
  1. TOKEN BUDGET — **THE FIX FAILED. Do not repeat it.** Pass B truncated on
     26 % of clips at 2200 (head-to-head §6b) and 32.5 % on our own re-measure.
     Raising the ceiling to 3500 took truncation to **61.5 %** and median Pass-B
     generation from 24 s to 135 s: this model spends whatever budget it is
     given. What worked instead was cutting the free text (v2b, below) plus
     `salvage_json`, because the failure is a TAIL failure — the object comes
     out correctly in schema order and one late free-text field runs away, so
     `SCENARIO` survives and was recovered from 7/7 truncated replies.
     `INTERACT` and `SIGNAL` are also DELETED outright (both ~0 % informative;
     a forward camera cannot see a blinker).
  2. CONFIDENCE — **THE FIX FAILED.** `route_confidence` was a float at 0.99 on
     200/200 windows. The discrete band, with an explicit anchor for when to say
     `low`, produced `high` on **97/100** and `low` **never**. Changing the
     response format does not create calibration, it relabels the constant.
     Treat this field as absent; delete it if production reproduces a modal
     share >= 0.90.
  3. EVIDENCE CONTAMINATION — **FIXED.** The v1 prompt illustrated
     `route_evidence` with a concrete sentence and Reason2 copied a fragment on
     13.1 % of its turn calls (8/61). v2 ships NO example, only a structural
     instruction plus "in your own words": **0 of 27 turn calls, 0 of 100
     windows**, unique-string rate 0.735 -> 0.930.
  4. ENUM ORDER — **PROBED, and the bias is the model's.** Listing `right`
     before `left` did not move it (left share of turn calls 74.5 % -> 66.7 % on
     a 48.2 %-left corpus; recall on true right turns 0.2069 in BOTH arms). But
     order is not inert: it moved turn-detection recall by 8.9 pp, CI
     [+0.035, +0.153]. `--enum-order randomized` therefore exists as variance
     insurance. Sentinels (`unknown`, `none`) are PINNED LAST — rotating them
     through first position would change abstention behaviour and confound the
     very thing the randomization protects.
  5. THE FINDING THAT OUTRANKED ALL FOUR. Pass B echoes our own kinematics:
     its event times reproduce our `future_track` onset to the decimal
     (11.9 -> 11.9, 15.4 -> 15.4), while Pass A — which never sees that block —
     is 100 % compliant with the offered frame offsets. The doctrine that
     quarantines Pass B's ROUTE extends to event times AND to `road_geometry`,
     so `vlm_labels_to_lake.py` sources the scenario strata from PASS A.

WHAT THE VLM IS AND IS NOT ASKED FOR (the provenance split, unchanged)
  Kinematics own VTARGET / LONMODE / LATMANEUVER / DYN / HEADWAY / ROUTE. The
  VLM owns the WHY: scene, geometry, signs, lead semantics, scenario tags, risk,
  evidence. ROUTE is still asked and still recorded, as a CROSS-CHECK ONLY — it
  is at chance on direction (57.1 %, CI [0.400, 0.745]) and may never be minted.
  The one class of number v2 does newly ask for is an EVENT TIME OFFSET, and it
  is safe precisely because it is not a measurement: the model must COPY one of
  the frame offsets it was shown, so it is a selection from a shipped list like
  every other categorical. It matters because an intersection or roundabout
  lasts 5-20 s while our planning window is 2 s — without the offset a window
  cannot be tagged with the event that resolves after it ends.

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack:/root/vlmprod HF_HUB_OFFLINE=1 \
  python3 vlm_semantic_labels.py --val <epcache> --out <dir> --tag val_full \
      --episodes 0-79 --stride 40 --passes AB --frames base
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import refb_labels as R                      # noqa: E402
import vlm_route_labels as VL                # noqa: E402
from vlm_model_compare import VLM, build_windows, classify_A  # noqa: E402

# v2b (2026-07-21). v2a's Pass B TRUNCATED MORE at 3500 tokens (61.5 %, n=13)
# than v1 did at 2200 (32.5 %, n=40) — raising the ceiling did not fix the defect, it fed it.
# The raw replies say why: the object is emitted correctly in schema order and
# then one free-text field near the end turns into an essay. v2b therefore cuts
# the free text instead of buying more of it:
#   1. `route_evidence` DELETED from Pass B. Pass B is shown the numeric future
#      ego track, so its ROUTE is inadmissible anyway and its justification was
#      pure cost. Pass A still carries the evidence field that matters.
#   2. `odd_flags` (a free LIST) deleted — `STRATEGIC.ODD` is the enum'd twin and
#      is already asked.
#   3. Every remaining free-text value is explicitly capped, and the cap is also
#      stated in the shared HARD RULES where the model actually reads it.
# ⚠️ (3) puts the change in `_RULES_V2`, which BOTH passes prepend — so a Pass-A
# prompt is NOT byte-identical between -a and -b, only its TASK block is. Do not
# pool Pass-A records across the two versions; the frame ablation runs entirely
# on -b and is internally consistent. The per-arm `prompt_A.txt` written into
# every run directory is the authoritative record of what was actually asked.
PROMPT_VERSION = "vlmsem-2026-07-21-b"
MODEL_ID = os.environ.get("VLM_MODEL", "nvidia/Cosmos-Reason2-8B")
DT = VL.DT

# ---------------------------------------------------------------- frame plans
# The ablation axis Sayed asked for. `base` reproduces v1 exactly so the
# comparison has a fixed point. Every plan is (history offsets, future offsets,
# image edge px) — resolution is part of the plan because it trades directly
# against frame count at fixed prompt-token cost (~200 tok/image at 448 px,
# ~70 at 256 px), and the stored frame is 256 px so 448 is pure upscaling.
FRAME_PLANS = {
    "base":        ((-3.0, -1.5, 0.0), (2.0, 5.0, 10.0, 15.0, 20.0), 448),
    "dense_hist":  ((-3.0, -2.0, -1.0, -0.5, 0.0),
                    (2.0, 5.0, 10.0, 15.0, 20.0), 448),
    "dense_early": ((-3.0, -1.5, 0.0),
                    (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0), 448),
    "native_res":  ((-3.0, -1.5, 0.0), (2.0, 5.0, 10.0, 15.0, 20.0), 256),
    "wide_cheap":  ((-3.0, -1.5, 0.0),
                    (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0), 256),
    "lean":        ((-1.5, 0.0), (2.0, 5.0, 10.0, 20.0), 448),
}

# Slots deleted from the v2 ask. Recorded as `not_asked` so a consumer can tell
# "the model declined" from "we stopped asking" — the head-to-head measured both
# models at ~0 % informative here, and a forward camera cannot see a blinker.
NOT_ASKED = ("INTERACT", "SIGNAL")

SENTINELS = ("unknown", "none")             # pinned last under randomization


# ------------------------------------------------------------ salvage parser
# The implementation lives in `vlm_labels_to_lake` — the stdlib-only module —
# so the GPU harness and the pod-free converter can never drift apart on what
# "recovered" means. See its docstring for why a truncated reply is still worth
# keeping (the failure is a TAIL failure; `SCENARIO` survives it).
from vlm_labels_to_lake import salvage_json   # noqa: E402,F401


# ------------------------------------------------------------- enum ordering
def order_tokens(tokens, seed_key: str | None, slot: str) -> list:
    """Enum token order for one slot on one window.

    `seed_key=None` -> as written. Otherwise a deterministic permutation keyed
    on (window, slot), so a record is reproducible from its stored seed and two
    slots in the same prompt do not share an order.
    """
    toks = list(tokens)
    if seed_key is None:
        return toks
    tail = [t for t in toks if t in SENTINELS]
    head = [t for t in toks if t not in SENTINELS]
    h = hashlib.blake2b(f"{seed_key}|{slot}".encode(), digest_size=8).digest()
    random.Random(int.from_bytes(h, "big")).shuffle(head)
    return head + tail


# ------------------------------------------------------------------ prompts
_RULES_V2 = """You are labeling a driving clip for an autonomous-driving dataset.
You are shown HISTORY frames (what the car had just seen) and FUTURE frames
(what actually happened next, in order). The FUTURE FRAMES are your evidence:
they show the outcome.

HARD RULES
1. Answer with a SINGLE JSON object and nothing else. No prose outside it.
2. Every categorical field must be copied EXACTLY from the allowed list given
   for it. Never invent a token, never reword one, never combine two.
3. Never state a number you cannot measure. No metres, no m/s, no seconds to
   collision. Distances are BUCKETS. The only numbers you may emit are (a) a
   value you literally READ off a sign, and (b) a time offset, which must be
   COPIED from the frame offsets listed below.
4. If the evidence does not support a field, answer "unknown". "unknown" is a
   correct answer and is preferred over a plausible guess.
5. Free text must be in YOUR OWN words about THIS clip. Do not reuse phrasing
   from these instructions.
6. HARD LENGTH LIMIT: every free-text value must be UNDER 20 WORDS. One short
   sentence, then stop and move to the next field. Do not explain, do not
   summarise, do not add commentary. A reply that runs long is CUT OFF and the
   whole answer is thrown away, so brevity is not a style preference - it is
   what makes your answer count at all.

CONFIDENCE BANDS - answer with the word, never a number:
  high   = the frames show it directly and unambiguously
  medium = the frames imply it but you had to infer
  low    = the frames do not really show it and you are mostly guessing
Answer "low" whenever the future frames do not actually show the thing you are
naming - for instance if you never see the junction itself."""

PASS_A_TASK_V2 = """TASK (route intent only).
From the FUTURE FRAMES, determine where the vehicle actually goes.

ROUTE must be one of: {route_enum}
  If the vehicle leaves the road it is on - at a junction, ramp or driveway, or
    by a discrete turn onto a different road - answer with the side it departs
    toward.
  If it stays on its current road, INCLUDING following a bend or a curve in
    that road, answer "straight".
  If it reverses direction onto the opposing carriageway, answer "u_turn".
  If the future frames do not show enough to tell, answer "unknown".

Return exactly:
{{"ROUTE": "<token>",
 "route_confidence_band": "<{band}>",
 "route_evidence": "<ONE short sentence IN YOUR OWN WORDS naming which listed
                    frame offset you read this off, and what changed in it>",
 "route_event_time_s": <copy one of the FUTURE frame offsets listed above, or null>,
 "road_geometry": "<one of: {road_geometry}>",
 "road_geometry_confidence_band": "<{band}>",
 "geometry_event_time_s": <copy one of the FUTURE frame offsets listed above, or null>,
 "sees_junction_ahead": <true|false>}}"""

PASS_B_TASK_V2 = """TASK (full scene interpretation).
Return exactly this JSON structure, selecting every token from its own list.

{{"SCENARIO": {{
   "road_type": "<{road_type}>",
   "environment": {{"weather": "<{weather}>", "time_of_day": "<{time_of_day}>",
                    "lighting": "<{lighting}>"}},
   "surface": "<{surface}>",
   "traffic_density": "<{traffic_density}>",
   "road_geometry": "<{road_geometry}>",
   "road_geometry_confidence_band": "<{band}>",
   "geometry_event_time_s": <copy a listed FUTURE frame offset, or null>,
   "geometry_event_end_time_s": <copy a listed FUTURE frame offset, or null>,
   "scenario_tag": "<{scenario_tag}>",
   "scenario_confidence_band": "<{band}>",
   "scenario_event_time_s": <copy a listed FUTURE frame offset, or null>,
   "difficulty": "<{difficulty}>"}},
 "STRATEGIC": {{
   "ROUTE": "<{route_enum}>", "route_confidence_band": "<{band}>",
   "route_event_time_s": <copy a listed FUTURE frame offset, or null>,
   "MISSION": "<{MISSION}>", "LANEOBJ": "<{LANEOBJ}>",
   "SPEEDPOLICY": "<{SPEEDPOLICY}>", "STYLE": "<{STYLE}>",
   "RISK": "<{RISK}>", "ODD": "<{ODD}>"}},
 "TACTICAL": {{
   "LATMANEUVER": "<{LATMANEUVER}>", "LONMODE": "<{LONMODE}>",
   "VSOURCE": "<{VSOURCE}>", "HEADWAY": "<{HEADWAY}>", "DYN": "<{DYN}>",
   "RULECTX": "<{RULECTX}>", "TACPOINT": "<{TACPOINT}>",
   "LIGHTSTATE": "<{LIGHTSTATE}>"}},
 "OBSERVATIONS": {{
   "sign_reads": [{{"type": "<{sign_type}>", "value": <number you READ or null>,
                    "unit": "<kph|mph|none>", "band": "<{band}>"}}],
   "lead_vehicle": {{"present": <true|false>, "lane": "<{lead_lane}>",
                     "distance_bucket": "<{distance_bucket}>",
                     "relative_motion": "<{relative_motion}>",
                     "band": "<{band}>"}},
   "critical_agents": [{{"type": "<{agent_type}>",
                         "position": "<{agent_position}>",
                         "behavior": "<AT MOST 6 WORDS>",
                         "relevance": "<{relevance}>"}}],
   "lane_info": {{"ego_lane_index": <int or null>, "n_lanes": <int or null>,
                  "markings": "<{markings}>", "lane_type": "<{lane_type}>"}},
   "traffic_light": {{"present": <true|false>, "state": "<{light_state}>",
                      "applies_to_ego": <true|false>}}}},
 "COC": {{
   "observation": "<what matters, UNDER 20 WORDS>",
   "inference": "<what it implies, UNDER 20 WORDS>",
   "decision": "<what the ego should do, UNDER 20 WORDS>"}}}}

STOP after the closing brace. At most 2 entries in `critical_agents` and at
most 2 in `sign_reads`.

EVENT TIMES MATTER. An intersection, roundabout or merge takes far longer than
the clip's first seconds. `geometry_event_time_s` is when the feature you named
first becomes visible ahead, `geometry_event_end_time_s` is when the vehicle has
finished passing through it, and `scenario_event_time_s` is when the tagged
event happens. Each must be COPIED from the FUTURE frame offsets listed above,
or be null if it is not among them.

VSOURCE is YOURS to justify: say WHY the set-speed is what it is (a sign you
read, a lead vehicle, a curve, the road class, or the traffic flow). Do NOT
state a target speed - kinematics own that number.
HEADWAY is a qualitative bucket only; never compute seconds."""

BAND_ENUM = ("high", "medium", "low")


def build_prompt_v2(which: str, seed_key: str | None = None) -> str:
    """The v2 prompt for one window. `seed_key` non-None => randomized enums."""
    from tanitad.lake import vocab as V

    def j(tokens, slot):
        return " | ".join(order_tokens(tokens, seed_key, slot))

    fmt = {"band": " | ".join(BAND_ENUM),
           "route_enum": j(VL.ROUTE_ENUM, "ROUTE"),
           "road_geometry": j(VL.ENUMS_SCENARIO["road_geometry"],
                              "road_geometry")}
    if which == "A":
        return PASS_A_TASK_V2.format(**fmt)
    for k, v in VL.ENUMS_SCENARIO.items():
        fmt[k] = j(v, k)
    for k, v in VL.ENUMS_OBS.items():
        fmt[k] = j(v, k)
    for k in ("MISSION", "LANEOBJ", "SPEEDPOLICY", "STYLE", "RISK", "ODD"):
        fmt[k] = j(V.STRATEGIC_TOKENS[k], k)
    for k in ("LATMANEUVER", "LONMODE", "VSOURCE", "HEADWAY", "DYN", "RULECTX",
              "TACPOINT", "LIGHTSTATE"):
        fmt[k] = j(V.TACTICAL_TOKENS[k], k)
    return PASS_B_TASK_V2.format(**fmt)


def build_prompt(which: str, version: str, seed_key: str | None = None) -> str:
    if version == "v1":
        return VL.build_prompt(which)
    return build_prompt_v2(which, seed_key)


def enums_snapshot() -> dict:
    """{slot: allowed tokens} EXACTLY as shipped in the prompt.

    Written into the run directory so the scorer never re-imports the
    vocabulary (and therefore can never drift from what was actually asked, and
    needs neither torch nor the pod)."""
    from tanitad.lake import vocab as V
    e = {f"SCENARIO.{k}": list(v) for k, v in VL.ENUMS_SCENARIO.items()}
    e.update({f"OBS.{k}": list(v) for k, v in VL.ENUMS_OBS.items()})
    e["ROUTE"] = list(VL.ROUTE_ENUM)
    e["BAND"] = list(BAND_ENUM)
    for k in ("MISSION", "LANEOBJ", "SPEEDPOLICY", "STYLE", "RISK", "ODD"):
        e[f"STRATEGIC.{k}"] = list(V.STRATEGIC_TOKENS[k])
    for k in ("LATMANEUVER", "LONMODE", "VSOURCE", "HEADWAY", "DYN", "RULECTX",
              "TACPOINT", "LIGHTSTATE"):
        e[f"TACTICAL.{k}"] = list(V.TACTICAL_TOKENS[k])
    return e


# ------------------------------------------------------------------- frames
def pick_frames(t: int, T: int, hist_s, fut_s):
    """(indices, caption lines, actual offsets). Only frames that EXIST are
    offered and the caption says which — a late-clip window genuinely has less
    future and must not be told otherwise."""
    hist, fut, lines, h_off, f_off = [], [], [], [], []
    for s in hist_s:
        i = int(round(t + s / DT))
        if 0 <= i < T:
            hist.append(i)
            h_off.append(s)
            lines.append(f"  history frame at t{s:+.1f} s")
    for s in fut_s:
        i = int(round(t + s / DT))
        if 0 <= i < T:
            fut.append(i)
            f_off.append(s)
            lines.append(f"  FUTURE frame at t{s:+.1f} s")
    return hist, fut, lines, h_off, f_off


def to_pil(ep_frames, i: int, px: int):
    from PIL import Image
    arr = ep_frames[i, -3:].permute(1, 2, 0).contiguous().numpy()
    im = Image.fromarray(arr)
    return im if im.size == (px, px) else im.resize((px, px), Image.BICUBIC)


# ------------------------------------------------- stratified train sampling
def _poses_only(path: str):
    """Load ONLY the pose track. `mmap=True` keeps the 117 MB frame tensor on
    disk — a stratifier that paged in every frame would be slower than the
    labeling run it is meant to feed."""
    try:
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        d = torch.load(path, map_location="cpu", weights_only=False)
    return d["poses"].float(), int(d.get("episode_id", -1)), \
        int(min(d["poses"].shape[0], d["frames_u8"].shape[0]))


def window_strata(poses: torch.Tensor, t: int, T: int) -> dict:
    """Kinematic signatures for one window — poses only, no frames, no GPU.

    These are the sampling strata. They are KINEMATIC SIGNATURES and are never
    labelled "intersection"/"roundabout": naming a geometry is the VLM's job and
    the whole reason this run exists."""
    k = R.route_from_future_v21(poses, t)
    h = min(R.NAV_HORIZON_STEPS, T - 1 - t)
    seg = poses[t:t + max(h, 1) + 1]
    v = seg[:, 3]
    v0 = float(poses[t, 3])
    v_min, v_max = float(v.min()), float(v.max())
    n2 = min(int(2.0 / DT), seg.shape[0] - 1)
    acc2 = (float(seg[n2, 3] - v0) / (n2 * DT)) if n2 >= 1 else 0.0
    dyaw = R.wrap_to_pi(seg[1:, 2] - seg[:-1, 2])
    net_deg = abs(float(dyaw.sum())) * 180.0 / 3.141592653589793
    # whole-clip cumulative heading: the only window-independent hint that an
    # episode contains a roundabout-scale event
    cd = R.wrap_to_pi(poses[1:, 2] - poses[:-1, 2])
    tags = []
    if k["valid"] and R.ROUTE_V21_NAMES[k["route"]] in ("left", "right"):
        tags.append("turn_" + R.ROUTE_V21_NAMES[k["route"]])
    if v0 < R.STOP_V_MS and v_max > R.MOVING_V_MS:
        tags.append("launch_from_stop")
    if v0 > R.MOVING_V_MS and v_min < R.STOP_V_MS:
        tags.append("stop_approach")
    if acc2 <= -0.5:
        tags.append("brake")
    elif acc2 >= 0.5:
        tags.append("accel")
    else:
        tags.append("steady")
    if net_deg >= 45.0:
        tags.append("sharp_turn")
    if v0 >= 20.0:
        tags.append("high_speed")
    if not tags:
        tags.append("plain")
    return {"tags": tags, "v0": round(v0, 2), "acc2": round(acc2, 3),
            "net_deg": round(net_deg, 1),
            "clip_cum_deg": round(abs(float(cd.sum())) * 180.0 / 3.14159265, 1),
            "kin_v21": {"route": R.ROUTE_V21_NAMES[k["route"]],
                        "valid": bool(k["valid"]),
                        "ambiguous": bool(k["ambiguous"]),
                        "reason": k["reason"],
                        "net_dyaw_deg": round(k["net_dyaw"] * 57.29577951, 2),
                        "peak_kappa": round(k["peak_kappa"], 5),
                        "concentration": round(k["concentration"], 4),
                        "arc_m": round(k["arc_m"], 1),
                        "h_steps": int(k["h_steps"])}}


def enumerate_candidates(cache: str, stride: int, cache_file: str | None) -> list:
    """Tag every candidate window of an epcache. Cached: the walk is the
    expensive half (~6 min over 2376 episodes) and re-sampling must be free."""
    if cache_file and os.path.exists(cache_file):
        cand = json.load(open(cache_file))
        print(f"[strata] reusing {cache_file}: {len(cand)} candidates",
              flush=True)
        return cand
    files = sorted(glob.glob(os.path.join(cache, "ep_*.pt")))
    cand, t0 = [], time.time()
    for n, f in enumerate(files):
        ep = os.path.basename(f).replace(".pt", "")
        try:
            poses, eid, T = _poses_only(f)
        except Exception as e:
            print(f"[strata] skip {ep}: {type(e).__name__}: {e}", flush=True)
            continue
        for t in range(0, T, stride):
            if T - 1 - t < int(2.0 / DT):
                continue                      # no future to show -> nothing to ask
            s = window_strata(poses, t, T)
            cand.append({"episode": ep, "t": int(t), "episode_id": eid,
                         "clip_len": T, **s})
        if n % 200 == 0:
            print(f"[strata] {n}/{len(files)} episodes, {len(cand)} candidates, "
                  f"{time.time() - t0:.0f}s", flush=True)
    if cache_file:
        json.dump(cand, open(cache_file, "w"))
    return cand


def sample_stratified(cache: str, n_target: int, stride: int, seed: int,
                      max_per_ep: int, cache_file: str | None = None) -> dict:
    """A QUOTA-BALANCED window manifest over a whole epcache.

    Uniform sampling of this corpus buys almost nothing: ~74 % of windows are
    straight, so a uniform draw spends three quarters of the GPU budget
    re-confirming that a straight road is straight.

    But inverse-frequency WEIGHTING over-corrects, and did: a first attempt gave
    380/600 windows to `launch_from_stop` (the rarest stratum, 905 candidates)
    and **1** to `high_speed` (5034 candidates) — because sorting by weight is
    greedy and simply drains the rarest stratum first. `high_speed` is where the
    flagship's dominant failure lives, so that sample would have been worse than
    uniform for the thing we most need labelled.

    So: an explicit EQUAL QUOTA per stratum, filled rarest-first from a shuffled
    pool, capped per episode, with unused quota redistributed. Every stratum is
    represented; none can swallow the budget.
    """
    rng = random.Random(seed)
    cand = enumerate_candidates(cache, stride, cache_file)
    by_tag = {}
    for i, c in enumerate(cand):
        for tg in c["tags"]:
            by_tag.setdefault(tg, []).append(i)
    freq = {k: len(v) for k, v in by_tag.items()}
    print(f"[strata] candidates={len(cand)} strata={freq}", flush=True)

    picked, per_ep, taken = [], {}, set()

    def _fill(tag, quota):
        got = 0
        pool = list(by_tag[tag])
        rng.shuffle(pool)
        for i in pool:
            if got >= quota or len(picked) >= n_target:
                break
            if i in taken:
                continue
            c = cand[i]
            if per_ep.get(c["episode"], 0) >= max_per_ep:
                continue
            taken.add(i)
            per_ep[c["episode"]] = per_ep.get(c["episode"], 0) + 1
            picked.append(c)
            got += 1
        return got

    order = sorted(freq, key=lambda k: freq[k])        # rarest first
    quota = max(1, n_target // max(1, len(order)))
    for tag in order:
        _fill(tag, quota)
    # redistribute whatever the small strata could not fill
    for _ in range(3):
        if len(picked) >= n_target:
            break
        for tag in order:
            if len(picked) >= n_target:
                break
            _fill(tag, max(1, (n_target - len(picked)) // max(1, len(order))))

    picked = sorted(picked, key=lambda c: (c["episode"], c["t"]))
    got = {}
    for c in picked:
        for tg in c["tags"]:
            got[tg] = got.get(tg, 0) + 1
    return {"val": cache, "mode": "stratified_quota", "seed": seed,
            "stride": stride, "max_per_episode": max_per_ep,
            "quota_per_stratum": quota,
            "n_candidates": len(cand), "candidate_strata": freq,
            "n_windows": len(picked), "sampled_strata": got,
            "n_episodes": len({c["episode"] for c in picked}),
            "windows": picked}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser("vlm_semantic_labels")
    ap.add_argument("--val", required=True, help="epcache dir with ep_*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True, help="arm/run name subdirectory")
    ap.add_argument("--episodes", default="0-79")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--passes", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prompt-version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--frames", default="base", choices=sorted(FRAME_PLANS))
    ap.add_argument("--enum-order", default="as_written",
                    choices=["as_written", "randomized"])
    ap.add_argument("--max-new-a", type=int, default=1000)
    ap.add_argument("--max-new-b", type=int, default=3500)
    ap.add_argument("--raw-chars", type=int, default=6000)
    ap.add_argument("--limit-windows", type=int, default=0)
    ap.add_argument("--window-stride", type=int, default=1)
    ap.add_argument("--windows", default=None,
                    help="reuse an existing manifest (REQUIRED for a paired "
                         "ablation: two plans must see identical windows)")
    ap.add_argument("--windows-only", action="store_true")
    ap.add_argument("--sample-strata", type=int, default=0,
                    help="build a rare-event-weighted manifest of N windows "
                         "over --val and exit (no GPU)")
    ap.add_argument("--max-per-episode", type=int, default=3)
    ap.add_argument("--cand-cache", default=None,
                    help="cache the tagged candidate walk here; re-sampling "
                         "with a different budget then costs seconds, not the "
                         "~6 min pose walk over 2376 episodes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wpath = args.windows or os.path.join(args.out, "windows.json")

    if args.sample_strata:
        W = sample_stratified(args.val, args.sample_strata, args.stride,
                              args.seed, args.max_per_episode, args.cand_cache)
        json.dump(W, open(wpath, "w"), indent=1)
        print(f"[strata] wrote {wpath}: {W['n_windows']} windows over "
              f"{W['n_episodes']} episodes -> {W['sampled_strata']}", flush=True)
        return

    if os.path.exists(wpath):
        W = json.load(open(wpath))
        print(f"[win] reusing {wpath}: {len(W['windows'])} windows", flush=True)
    else:
        lo, hi = ((int(x) for x in args.episodes.split("-"))
                  if "-" in args.episodes else (int(args.episodes),) * 2)
        rows = build_windows(args.val, lo, hi, args.stride, args.t0)
        W = {"val": args.val, "episodes": args.episodes, "stride": args.stride,
             "t0": args.t0, "n_windows": len(rows),
             "n_episodes": len({r["episode"] for r in rows}), "windows": rows}
        json.dump(W, open(wpath, "w"), indent=1)
        print(f"[win] built {wpath}: {len(rows)} windows", flush=True)
    if args.windows_only:
        return

    hist_s, fut_s, px = FRAME_PLANS[args.frames]
    dst_dir = os.path.join(args.out, args.tag)
    os.makedirs(dst_dir, exist_ok=True)
    json.dump(enums_snapshot(), open(os.path.join(args.out, "enums.json"), "w"),
              indent=1)
    # the exact prompt text, so the scorer can measure verbatim reuse without
    # re-deriving anything and without importing the vocabulary
    for w_ in ("A", "B"):
        if w_ in args.passes:
            open(os.path.join(dst_dir, f"prompt_{w_}.txt"), "w",
                 encoding="utf-8").write(
                build_prompt(w_, args.prompt_version, None))

    vlm = VLM(args.model)
    torch.cuda.reset_peak_memory_stats()
    ep_name, ep_cache = None, None
    n_done = n_err = 0
    t_start = time.time()

    for w in W["windows"][::max(1, args.window_stride)]:
        dst = os.path.join(dst_dir, f"{w['episode']}_t{w['t']:04d}.json")
        if os.path.exists(dst):
            continue
        if ep_name != w["episode"]:
            ep_cache = torch.load(os.path.join(W["val"], w["episode"] + ".pt"),
                                  map_location="cpu", weights_only=False)
            ep_name = w["episode"]
        frames, poses = ep_cache["frames_u8"], ep_cache["poses"].float()
        T = min(frames.shape[0], poses.shape[0])
        hist, fut, lines, h_off, f_off = pick_frames(w["t"], T, hist_s, fut_s)
        if not fut:
            continue
        seed_key = (f"{w['episode']}|{w['t']}"
                    if args.enum_order == "randomized" else None)
        eb = VL.ego_block(poses, w["t"])
        imgs = [to_pil(frames, i, px) for i in hist + fut]
        head = (f"{_RULES_V2 if args.prompt_version == 'v2' else VL._RULES}\n\n"
                f"FRAMES PROVIDED (in order):\n" + "\n".join(lines)
                + f"\n\nEGO MOTION NOW (measured, past only):\n"
                  f"{json.dumps(eb)}\n{eb['summary']}\n")

        rec = {k: w[k] for k in ("episode", "t", "clip_len", "kin_v21")
               if k in w}
        rec.update(
            episode_id=int(ep_cache.get("episode_id", -1)),
            val_build=os.path.basename(W["val"].rstrip("/")),
            model=vlm.model_id, arch=vlm.arch, model_tag=args.tag,
            prompt_version=(PROMPT_VERSION if args.prompt_version == "v2"
                            else VL.PROMPT_VERSION),
            provenance="vlm", frames_plan=args.frames,
            hist_offsets_s=h_off, future_offsets_s=f_off, image_px=px,
            n_images=len(imgs), n_hist_frames=len(hist),
            n_future_frames=len(fut),
            enum_order=args.enum_order, enum_seed_key=seed_key,
            route_enum_order=order_tokens(VL.ROUTE_ENUM, seed_key, "ROUTE"),
            not_asked=list(NOT_ASKED), ego_block=eb,
            strata=w.get("tags"))

        if "A" in args.passes:
            pa = build_prompt("A", args.prompt_version, seed_key)
            raw, n_in, n_gen, dt, trunc, err = "", 0, 0, 0.0, False, None
            try:
                raw, n_in, n_gen, dt, trunc = vlm.ask(imgs, head + "\n" + pa,
                                                      args.max_new_a)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_err += 1
                print(f"[err A] {w['episode']} t={w['t']}: {err}", flush=True)
            c = classify_A(raw, trunc, err)
            js = c.pop("js")
            rec["pass_A"] = {
                "pass": "A", "future_track_given": False,
                "ROUTE": c["ROUTE"], "enum_ok": c["enum_ok"],
                "raw_route": c["raw_route"], "outcome": c["outcome"],
                "truncated": bool(trunc), "error": err, "parsed": js,
                "n_prompt_tokens": n_in, "n_gen_tokens": n_gen,
                "gen_seconds": round(dt, 3), "raw_len": len(raw),
                "raw": raw[:args.raw_chars]}

        if "B" in args.passes:
            pb = build_prompt("B", args.prompt_version, seed_key)
            fts = VL.future_track_summary(poses, w["t"])
            raw, n_in, n_gen, dt, trunc, err = "", 0, 0, 0.0, False, None
            try:
                raw, n_in, n_gen, dt, trunc = vlm.ask(
                    imgs, head + "\nNUMERIC FUTURE EGO TRACK (measured):\n"
                    + fts["summary"] + "\n\n" + pb, args.max_new_b)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_err += 1
                print(f"[err B] {w['episode']} t={w['t']}: {err}", flush=True)
            strict = VL.extract_json(raw)
            salv, mode = salvage_json(raw)
            rec["pass_B"] = {
                "pass": "B", "future_track_given": True, "future_track": fts,
                # `parsed` stays STRICT — no consumer inherits a repaired object
                # by accident. The salvage lands beside it, labelled.
                "parsed": strict or {}, "parse_mode": mode,
                "parsed_salvaged": (salv if mode == "partial" else None),
                "salvaged_blocks": (sorted(salv) if mode == "partial" else None),
                "error": err,
                "truncated": bool(trunc), "n_prompt_tokens": n_in,
                "n_gen_tokens": n_gen, "gen_seconds": round(dt, 3),
                "raw_len": len(raw), "raw": raw[:args.raw_chars]}

        rec["peak_vram_gib"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t_start
            print(f"[{args.tag} {n_done}] {w['episode']} t={w['t']} "
                  f"A={rec.get('pass_A', {}).get('ROUTE')} "
                  f"{el / n_done:.2f}s/win peak={rec['peak_vram_gib']:.1f}GiB "
                  f"err={n_err}", flush=True)
        if args.limit_windows and n_done >= args.limit_windows:
            break

    # NOTE: `rec` is out of scope when every window was already on disk (a fully
    # resumed run), so the version comes from the args, never from the last record.
    summary = {"tag": args.tag, "model": vlm.model_id, "arch": vlm.arch,
               "passes": args.passes,
               "prompt_version": (PROMPT_VERSION if args.prompt_version == "v2"
                                  else VL.PROMPT_VERSION),
               "frames_plan": args.frames, "hist_s": list(hist_s),
               "fut_s": list(fut_s), "image_px": px,
               "enum_order": args.enum_order,
               "max_new_a": args.max_new_a, "max_new_b": args.max_new_b,
               "n_windows_run": n_done, "n_errors": n_err,
               "load_seconds": round(vlm.load_s, 1),
               "weights_gib": round(vlm.weights_gb, 3),
               "peak_vram_gib": round(
                   torch.cuda.max_memory_allocated() / 2 ** 30, 3),
               "wall_seconds": round(time.time() - t_start, 1),
               "s_per_window": round(
                   (time.time() - t_start) / max(1, n_done), 3)}
    json.dump(summary, open(os.path.join(args.out, f"run_{args.tag}.json"), "w"),
              indent=1)
    print("VLM_SEMANTIC_DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
