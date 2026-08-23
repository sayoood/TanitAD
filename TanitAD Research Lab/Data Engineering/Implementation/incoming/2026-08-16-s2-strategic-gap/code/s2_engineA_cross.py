"""S2 gap review — Engine A (hindsight geometry) vs VLM strategic labels.

Extracts ENGINE_A from each B4 prompt in the batch_* v2 records, joins to the
fused aug120 g_str/a_str material, and measures:
  1. route_token x g_str confusion (the hindsight-derivability audit)
  2. TURN_* claims vs maneuver_dyaw / route token
  3. a_str verbs vs geometric evidence (lane_change_events, net_dv, stops)
  4. duplicate-pass replicates: same clip, two separate VLM invocations
  5. ego-prompt-mode census (provenance audit)
"""
import glob
import json
import os
import re
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_pull")
OUT = {}

# ---- load all batch v2 records -------------------------------------------- #
v2_by = {}          # clip -> list of (tag, record)
for p in sorted(glob.glob(os.path.join(ROOT, "batch_*", "v2", "*.json"))):
    tag = p.replace("\\", "/").split("/")[-3]
    d = json.load(open(p, encoding="utf-8"))
    for r in d.get("clips", []):
        v2_by.setdefault(r["clip_id"], []).append((tag, r))
OUT["n_v2_records_total"] = sum(len(v) for v in v2_by.values())
OUT["n_v2_clips"] = len(v2_by)
OUT["n_clips_with_dup"] = sum(1 for v in v2_by.values() if len(v) > 1)


def engine_a_of(rec):
    for c in rec.get("_calls", []):
        if c["call"] == "B4_symbols":
            m = re.search(r"ENGINE_A = (\{.*?\})\n", c["prompt"], re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
    return None


OUT["ego_prompt_mode"] = dict(Counter(
    r.get("_ego_prompt_mode") for v in v2_by.values() for _, r in v))
OUT["schema_version"] = dict(Counter(
    json.load(open(p, encoding="utf-8")).get("schema_version")
    for p in sorted(glob.glob(os.path.join(ROOT, "batch_*", "v2", "*.json")))))

# ---- fused g_str ----------------------------------------------------------- #
fused = {}
for p in sorted(glob.glob(os.path.join(ROOT, "fused_aug120", "*.json"))):
    if os.path.basename(p).startswith("_"):
        continue
    d = json.load(open(p, encoding="utf-8"))
    fused[d["clip_id"]] = d

# ---- 1/2: route_token x g_str ---------------------------------------------- #
conf = {}
turn_rows = []
missed_turns = []
ea_by_clip = {}
for cid, rec in fused.items():
    variants = v2_by.get(cid, [])
    ea = None
    for _, r in variants:
        ea = engine_a_of(r) or ea
    ea_by_clip[cid] = ea
    g = rec["vocab"]["g_str"]["token"]
    rt = (ea or {}).get("route_token")
    valid = (ea or {}).get("route_valid")
    key = f"{rt}|valid={valid}"
    conf.setdefault(key, Counter())[g] += 1
    dyaw = (ea or {}).get("maneuver_dyaw_rad")
    if g in ("TURN_LEFT", "TURN_RIGHT"):
        turn_rows.append({"clip": cid[:16], "g_str": g, "route": rt,
                          "route_valid": valid, "dyaw": dyaw,
                          "dist_m": (ea or {}).get("route_dist_m")})
    if rt in ("turn_left", "turn_right", "exit_left", "exit_right",
              "roundabout", "u_turn") and valid \
            and g in ("FOLLOW_MAIN_ROAD", "ROUTE_TO", "NONE_ABSTAIN"):
        missed_turns.append({"clip": cid[:16], "route": rt, "dyaw": dyaw,
                             "g_str": g})
OUT["route_x_gstr"] = {k: dict(v) for k, v in sorted(conf.items())}
OUT["turn_claims_vs_route"] = turn_rows
OUT["geometry_turns_vlm_missed"] = missed_turns
OUT["route_token_census"] = dict(Counter(
    (e or {}).get("route_token") for e in ea_by_clip.values()))

# ---- 3: a_str verbs vs geometry ------------------------------------------- #
def check(verb, direction, ea, rec):
    """Lite version of ph0_pilot._check_action_geometry (no constraints)."""
    if ea is None:
        return "no_engine_a"
    lc = ea.get("lane_change_events") or []
    lon = ea.get("speed_events") or []
    rt, rv = ea.get("route_token"), ea.get("route_valid")
    if verb == "prepare_lane_change":
        want = {"left": ("lc_left",), "right": ("lc_right",)}.get(
            direction, ("lc_left", "lc_right"))
        return "ok" if any(e.get("token") in want for e in lc) else "dispute"
    if verb == "prepare_exit":
        want = {"left": ("exit_left", "turn_left"),
                "right": ("exit_right", "turn_right")}.get(
            direction, ("exit_left", "exit_right", "turn_left", "turn_right"))
        return "ok" if (rt in want and rv) else "dispute"
    if verb == "prepare_stop":
        stop_ev = any(e.get("token") in ("stop_at_point", "hold_stop")
                      for e in lon)
        return "ok" if (stop_ev or ea.get("stops")) else "dispute"
    if verb == "reduce_to":
        decel = (any(e.get("token") in ("stop_at_point", "coast", "creep")
                     for e in lon)
                 or (ea.get("net_dv_ms") or 0.0) <= -1.0)
        return "ok" if decel else "dispute"
    if verb in ("hold_corridor", "resume_cruise"):
        junction = rv and rt in ("turn_left", "turn_right", "exit_left",
                                 "exit_right", "roundabout", "u_turn")
        return "dispute" if junction else "ok"
    return "off_vocab"


averb = {}
for cid, rec in fused.items():
    ea = ea_by_clip.get(cid)
    acts = ((rec.get("semantics") or {}).get("symbols") or {}) \
        .get("actions") or []
    for a in acts:
        v, d = a.get("verb"), a.get("direction")
        averb.setdefault(v, Counter())[check(v, d, ea, rec)] += 1
OUT["a_str_verb_vs_geometry"] = {k: dict(v) for k, v in averb.items()}

# ---- 4: duplicate-pass replicates ------------------------------------------ #
dup_same_goal = dup_diff_goal = 0
dup_same_actions = dup_diff_actions = 0
diff_examples = []
for cid, variants in v2_by.items():
    if len(variants) < 2:
        continue
    (t1, r1), (t2, r2) = variants[0], variants[1]
    s1, s2 = r1.get("symbols") or {}, r2.get("symbols") or {}
    g1, g2 = s1.get("goal_kind"), s2.get("goal_kind")
    a1 = json.dumps(s1.get("actions"), sort_keys=True)
    a2 = json.dumps(s2.get("actions"), sort_keys=True)
    if g1 == g2:
        dup_same_goal += 1
    else:
        dup_diff_goal += 1
        diff_examples.append({"clip": cid[:16], "tags": [t1, t2],
                              "goals": [g1, g2]})
    if a1 == a2:
        dup_same_actions += 1
    else:
        dup_diff_actions += 1
OUT["dup_replicates"] = {
    "n_dup_clips": OUT["n_clips_with_dup"],
    "goal_kind_same": dup_same_goal, "goal_kind_diff": dup_diff_goal,
    "actions_same": dup_same_actions, "actions_diff": dup_diff_actions,
    "goal_diff_examples": diff_examples[:10]}

# ---- 5: STOP_AT derivability from Engine A --------------------------------- #
stopish = sum(1 for e in ea_by_clip.values()
              if e and (e.get("stops")
                        or any(ev.get("token") in ("stop_at_point", "hold_stop")
                               for ev in (e.get("speed_events") or []))))
OUT["clips_engineA_stop_evidence"] = stopish
lcish = sum(1 for e in ea_by_clip.values()
            if e and (e.get("lane_change_events") or []))
OUT["clips_engineA_lane_change_events"] = lcish
OUT["clips_engineA_missing"] = sum(1 for e in ea_by_clip.values() if not e)

json.dump(OUT, open(os.path.join(ROOT, "..", "engineA_cross.json"), "w"),
          indent=1)
print(json.dumps(OUT, indent=1)[:8000])
