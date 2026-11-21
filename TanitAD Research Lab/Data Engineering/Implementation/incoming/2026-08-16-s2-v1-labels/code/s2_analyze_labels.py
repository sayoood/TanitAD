"""S2 v1 label analysis — the accounting behind every reported count.

Produces raw/label_analysis.json:
  * route-token x emitted-g_str cross-tab (aug120 + w120val)
  * old fused g_str -> new S2 g_str transition table (what changed, n)
  * ROUTE_TO disposition (remapped to what / abstained, per split)
  * LANE_TARGET audit: VLM prepare_lane_change corroboration + direction
  * STOP_AT accounting vs the review's 25 stop-event / 42 stops censuses
  * a_str x VLM-verb agreement
  * NONE_ABSTAIN reasons census
  * the 4 excluded val records' geometry (what a re-fuse would recover)
"""
import json
import os
from collections import Counter, defaultdict

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(PKG, "labels")
RAW = os.path.join(PKG, "raw")
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
PULL = os.path.join(SP, "s2_pull")

OUT: dict = {}


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def fused_dir(split):
    d = os.path.join(PULL, f"fused_{split}")
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and not f.startswith("_"):
            r = json.load(open(os.path.join(d, f), encoding="utf-8"))
            out[r["clip_id"]] = r
    return out


for split in ("aug120", "w120val"):
    labels = {r["clip_id"]: r
              for r in jsonl(os.path.join(LABELS, f"s2_labels_{split}.jsonl"))}
    eas = {r["clip_id"]: r["engine_a"]
           for r in jsonl(os.path.join(LABELS, f"engine_a_{split}.jsonl"))}
    fused = fused_dir(split)

    cross = defaultdict(Counter)
    transition = Counter()
    route_to_disp = Counter()
    lt_audit = {"n": 0, "vlm_lc_verb_present": 0, "direction_agrees": 0,
                "rows": []}
    stop_acct = {"clips_with_stop_event": 0, "stop_event_on_junction": 0,
                 "began_stopped_with_launch": 0, "labeled_STOP_AT": 0,
                 "stops_profile_true": 0}
    abstain_reasons = Counter()
    a_agree = Counter()
    a_census = Counter()

    for cid, rec in labels.items():
        ea = eas[cid]
        route = ea.get("route") or {}
        rt = route.get("token") if route.get("token_valid") else \
            f"{route.get('token')}(!valid)"
        g = rec["g_str"]
        a = rec["a_str"]
        cross[rt][g["token"]] += 1
        old = (((fused.get(cid) or {}).get("vocab") or {})
               .get("g_str") or {}).get("token")
        transition[f"{old} -> {g['token']}"] += 1
        vlm_goal = g["corroboration"].get("vlm_goal_kind")
        if vlm_goal == "route_to":
            route_to_disp["abstained" if g["token"] == "NONE_ABSTAIN"
                          else f"remapped->{g['token']}"] += 1
        if g["token"] == "LANE_TARGET":
            lt_audit["n"] += 1
            sym = (((fused.get(cid) or {}).get("semantics") or {})
                   .get("symbols")) or {}
            verbs = {(x.get("verb"), x.get("direction"))
                     for x in (sym.get("actions") or [])}
            has_lc = any(v == "prepare_lane_change" for v, _ in verbs)
            lt_audit["vlm_lc_verb_present"] += int(has_lc)
            want = "left" if g["args"][0] > 0 else "right"
            dir_ok = ("prepare_lane_change", want) in verbs
            lt_audit["direction_agrees"] += int(dir_ok)
            # Alpamayo third opinion: does its tactical meta_action mention a
            # lane change on this clip? (independent of both ego and our VLM)
            alp = json.dumps((fused.get(cid) or {}).get("alpamayo")
                             or {}).lower()
            alp_lc = "lane change" in alp or "lane_change" in alp
            lt_audit["alpamayo_lc_phrase"] = \
                lt_audit.get("alpamayo_lc_phrase", 0) + int(alp_lc)
            # the GATED event (the one the label actually used)
            gated = next((e for e in (ea.get("lane_change_events") or [])
                          if e.get("token") in ("lc_left", "lc_right")
                          and abs(float(e.get("lat_m") or 0.0)) >= 3.0), None)
            if split == "aug120":
                lt_audit["rows"].append(
                    {"clip": cid[:8], "dir": want,
                     "lat_m": (gated or {}).get("lat_m"),
                     "arc_m": (gated or {}).get("arc_from_t0_m"),
                     "vlm_lc": has_lc, "vlm_dir_agrees": dir_ok,
                     "alpamayo_lc": alp_lc})
        evs = ea.get("speed_events") or []
        has_stop_ev = any(e.get("token") in ("stop_at_point", "hold_stop")
                          for e in evs)
        sp = ea.get("speed_profile") or {}
        if has_stop_ev:
            stop_acct["clips_with_stop_event"] += 1
            if route.get("token_valid") and route.get("token") in (
                    "turn_left", "turn_right", "exit_left", "exit_right",
                    "u_turn", "roundabout"):
                stop_acct["stop_event_on_junction"] += 1
            if (sp.get("v_t0_ms") is not None
                    and float(sp["v_t0_ms"]) < 0.5
                    and any(e.get("token") == "launch" for e in evs)):
                stop_acct["began_stopped_with_launch"] += 1
        stop_acct["stops_profile_true"] += int(bool(sp.get("stops")))
        stop_acct["labeled_STOP_AT"] += int(g["token"] == "STOP_AT")
        if g["token"] == "NONE_ABSTAIN":
            abstain_reasons[rec["g_str"].get("reason") or "?"] += 1
        a_census[a["token"]] += 1
        agr = a["corroboration"].get("agrees")
        a_agree["agree" if agr else
                ("disagree" if agr is False else "no_vlm_verbs")] += 1

    OUT[split] = {
        "n_records": len(labels),
        "route_x_gstr": {k: dict(v) for k, v in sorted(cross.items())},
        "fused_to_s2_transition": dict(transition.most_common()),
        "route_to_disposition": dict(route_to_disp),
        "lane_target_audit": lt_audit,
        "stop_accounting": stop_acct,
        "abstain_reasons": dict(abstain_reasons),
        "a_str_census": dict(a_census),
        "a_str_vlm_agreement": dict(a_agree),
        "non_default_share": round(
            sum(1 for r in labels.values()
                if r["g_str"]["token"] != "FOLLOW_MAIN_ROAD") / len(labels),
            4),
    }

# the 4 excluded val records: what geometry says (recoverable on re-fuse)
excl = json.load(open(os.path.join(LABELS, "s2_excluded_w120val.json")))
OUT["excluded_val"] = excl

json.dump(OUT, open(os.path.join(RAW, "label_analysis.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "excluded_val"},
                 indent=1)[:6000])
print("ANALYZE_DONE")
