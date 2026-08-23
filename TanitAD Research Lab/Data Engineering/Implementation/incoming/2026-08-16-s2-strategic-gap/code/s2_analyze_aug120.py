"""S2 gap review — analysis of the pulled fused_aug120 records (n=201)."""
import glob
import json
import os
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_pull")
OUT = {}

files = sorted(glob.glob(os.path.join(ROOT, "fused_aug120", "*.json")))
recs = []
meta = {}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    if os.path.basename(f).startswith("_"):
        meta[os.path.basename(f)] = d
    else:
        recs.append(d)
OUT["n_files"] = len(files)
OUT["n_records"] = len(recs)
OUT["meta_files"] = sorted(meta)
OUT["schema_versions"] = dict(Counter(r.get("schema_version") for r in recs))

# ---- g_str ----------------------------------------------------------------- #
gs = Counter(r["vocab"]["g_str"]["token"] for r in recs)
OUT["g_str_dist"] = dict(gs.most_common())
OUT["g_str_src"] = dict(Counter(r["vocab"]["g_str"].get("src") for r in recs))
OUT["g_str_corrob_by_route"] = dict(Counter(
    r["vocab"]["g_str"].get("corroborated_by_route") for r in recs))
# any args anywhere in the g_str block?
extra_keys = Counter()
for r in recs:
    for k in r["vocab"]["g_str"]:
        if k not in ("token", "src", "corroborated_by_route"):
            extra_keys[k] += 1
OUT["g_str_extra_keys"] = dict(extra_keys)

# ---- a_str raw material: symbols.actions ----------------------------------- #
verb_clip = Counter()      # clips with >=1 action of this verb
verb_total = Counter()
dir_by_verb = {}
n_actions_per_clip = Counter()
for r in recs:
    acts = ((r.get("semantics") or {}).get("symbols") or {}).get("actions") or []
    n_actions_per_clip[len(acts)] += 1
    seen = set()
    for a in acts:
        v = a.get("verb")
        verb_total[v] += 1
        dir_by_verb.setdefault(v, Counter())[a.get("direction")] += 1
        seen.add(v)
    for v in seen:
        verb_clip[v] += 1
OUT["a_str_verbs_clips"] = dict(verb_clip.most_common())
OUT["a_str_verbs_total"] = dict(verb_total.most_common())
OUT["a_str_dir_by_verb"] = {k: dict(v) for k, v in dir_by_verb.items()}
OUT["n_actions_per_clip"] = dict(sorted(n_actions_per_clip.items()))

# ---- ego lat vote dead-mapping check --------------------------------------- #
turn_counter = Counter()
ego_lat_vote_by_turning = {}
for r in recs:
    ego = (r.get("ego") or {}).get("ego_state") or {}
    t = ego.get("turning")
    turn_counter[t] += 1
    votes = r["vocab"]["g_tac_lat"].get("votes") or []
    egov = [v for s, v in votes if s == "ego"]
    ego_lat_vote_by_turning.setdefault(t, Counter())[
        egov[0] if egov else "(no ego vote)"] += 1
OUT["ego_turning_dist"] = dict(turn_counter)
OUT["ego_lat_vote_by_turning"] = {str(k): dict(v)
                                  for k, v in ego_lat_vote_by_turning.items()}

# ---- TURN_* g_str vs ego evidence ------------------------------------------ #
turn_rows = []
for r in recs:
    tok = r["vocab"]["g_str"]["token"]
    if tok not in ("TURN_LEFT", "TURN_RIGHT"):
        continue
    ego = (r.get("ego") or {}).get("ego_state") or {}
    sp = (r.get("ego") or {}).get("speed_profile") or {}
    turn_rows.append({
        "clip": r["clip_id"][:16], "g_str": tok,
        "ego_turning_past8s": ego.get("turning"),
        "yaw_rate": ego.get("yaw_rate_rad_s"),
        "net_dyaw_past": ego.get("net_dyaw_rad"),
        "v_t0": sp.get("v_t0"), "stops": sp.get("stops"),
        "conf": ((r.get("semantics") or {}).get("symbols") or {}).get("conf")})
OUT["turn_gstr_vs_ego"] = turn_rows

# ---- ROUTE_TO audit --------------------------------------------------------- #
rt = []
for r in recs:
    if r["vocab"]["g_str"]["token"] != "ROUTE_TO":
        continue
    sym = ((r.get("semantics") or {}).get("symbols") or {})
    ev = sym.get("goal_evidence_sign")
    signs = ((r.get("semantics") or {}).get("signs") or {}).get("signs") or []
    kind = signs[ev].get("kind") if isinstance(ev, int) and ev < len(signs) \
        else None
    text = signs[ev].get("text") if isinstance(ev, int) and ev < len(signs) \
        else None
    ge = (r.get("corroboration") or {}).get("goal_evidence") or {}
    rt.append({"clip": r["clip_id"][:16], "ev_idx": ev, "sign_kind": kind,
               "sign_text": (text or "")[:24],
               "verdict": ge.get("verdict"),
               "sign_status": ((r.get("semantics") or {})
                               .get("sign_text_status"))})
OUT["route_to_rows"] = rt
OUT["route_to_verdicts"] = dict(Counter(x["verdict"] for x in rt))
OUT["route_to_sign_kinds"] = dict(Counter(x["sign_kind"] for x in rt))

# ---- STOP evidence vs STOP_AT ---------------------------------------------- #
stops_clips = sum(1 for r in recs
                  if ((r.get("ego") or {}).get("speed_profile") or {})
                  .get("stops", 0) > 0)
OUT["clips_with_ego_stops"] = stops_clips
OUT["g_str_STOP_AT"] = gs.get("STOP_AT", 0)
vmin_low = sum(1 for r in recs
               if (((r.get("ego") or {}).get("speed_profile") or {})
                   .get("v_min_future") or 99) < 0.5)
OUT["clips_vmin_below_0p5"] = vmin_low

# ---- SAM3 absence ----------------------------------------------------------- #
OUT["sam3_absent"] = sum(1 for r in recs
                         if (r.get("perception") or {}).get("absent"))
OUT["sam3_absent_reasons"] = dict(Counter(
    (r.get("perception") or {}).get("absent") for r in recs
    if (r.get("perception") or {}).get("absent")))

# ---- provenance block ------------------------------------------------------- #
OUT["provenance_variants"] = dict(Counter(
    json.dumps(r.get("_provenance"), sort_keys=True) for r in recs))
OUT["inference_admissible"] = dict(Counter(
    json.dumps(r.get("inference_admissible")) for r in recs))

# ---- scene/road_type vs g_str cross-tab ------------------------------------ #
xt = {}
for r in recs:
    road = ((r.get("semantics") or {}).get("scene") or {}).get("road_type")
    xt.setdefault(road, Counter())[r["vocab"]["g_str"]["token"]] += 1
OUT["gstr_by_road_type"] = {str(k): dict(v) for k, v in xt.items()}

json.dump(OUT, open(os.path.join(ROOT, "..", "aug120_analysis.json"), "w"),
          indent=1)
print(json.dumps(OUT, indent=1)[:7000])
