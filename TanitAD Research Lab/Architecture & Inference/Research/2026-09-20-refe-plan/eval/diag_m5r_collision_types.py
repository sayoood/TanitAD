#!/usr/bin/env python3
"""Measure 5r -- DIAGNOSTIC ONLY (changes no gate, no label, no labeller): WHICH KIND of collision does the teacher's
collision label see on the 0.75x copies where it disagrees with the REPAIRED NAVSIM harness (G-R4's split)?

For every (token, copy slot) of <v4_labels>/v5rep where the label and the harness NC disagree -- and, as a control, an
equal number of slots where both say "collision" -- the copy is rolled out AGAIN through the labeller's own teacher path
(`onpolicy_label.Scorer.score`: the same scenario build, lane-graph enrichment, calculators), with the candidate exactly
as v5 labelled it (planner.repair_last_heading of the served copy). A read-only hook on the teacher's NuPlanTTC.forward
(the original runs first, unchanged) records, at every rollout step where the ego is in a current collision, nuPlan's
own collision-type quantities, computed exactly as NuPlanTTC._classify_current_at_fault_collisions_for_ttc does:
ego stopped (speed <= STOPPED_SPEED_THRESHOLD), other stopped, active rear (cos(angle to other) < BEHIND_COS_THRESHOLD),
active front (front edge overlap). The FIRST collision step of the copy's rollout is classified:
  EGO_STOPPED  ACTIVE_REAR (an agent drives into the ego from behind)  STOPPED_TRACK (ego moves into a stopped agent)
  ACTIVE_FRONT  LATERAL/OTHER
NAVSIM's NC counts only at-fault collisions (ACTIVE_FRONT, STOPPED_TRACK, and lateral ones in multi-lane/non-drivable
states), so EGO_STOPPED / ACTIVE_REAR events are the ones NAVSIM does not count.

    python eval/diag_m5r_collision_types.py [--max-tokens 200]   -> eval/raw/m5r/diag_collision_types.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")
sys.path.insert(0, HERE)
sys.path.insert(0, REFE)
import m5r_gates as G  # noqa: E402
import validate_slow_labels_v4 as V  # noqa: E402

EVENTS: list = []
CUR = {"call": -1}


def install_hook():
    import torch
    from driverl.env.engine.reward_calculator.nuplan_collision import NuPlanCollision
    from driverl.env.engine.reward_calculator.nuplan_ttc import NuPlanTTC
    orig = NuPlanTTC.forward

    def forward(self, scenario_data, log_scenario_data, rewards_and_infos, **kw):
        orig(self, scenario_data, log_scenario_data, rewards_and_infos, **kw)
        polygons_now, _, valid = scenario_data.get_recent_agent_polygons()
        controlled = scenario_data.agent_control_manager.controlled_mask
        cur = self._collision.detect_current_collisions_knn(polygons_now, valid, controlled,
                                                             scenario_data.agent_nearest_indices[:, :, -1, :])
        b, e, o = torch.where(cur)
        if b.numel() == 0:
            return
        speed = torch.linalg.norm(scenario_data.agent_velocity_all[:, :, -1, :], dim=-1)
        ego_st = speed[b, e] <= NuPlanCollision.STOPPED_SPEED_THRESHOLD
        oth_st = speed[b, o] <= NuPlanCollision.STOPPED_SPEED_THRESHOLD
        ec, oc = polygons_now[b, e].mean(dim=1), polygons_now[b, o].mean(dim=1)
        yaw = scenario_data.agent_orientation_all[b, e, -1]
        rel = oc - ec
        cosang = (rel * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)).sum(-1) / torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
        rear = (~ego_st) & (~oth_st) & (cosang < NuPlanCollision.BEHIND_COS_THRESHOLD)
        front = (~ego_st) & (~oth_st) & (~rear) & self._collision._polygon_overlap_pairs(
            polygons_now[b, e][:, [0, 1]], polygons_now[b, o])
        for k in range(b.numel()):
            if not bool(controlled[b[k], e[k]]):
                continue
            EVENTS.append({"call": CUR["call"], "ego_speed": float(speed[b[k], e[k]]),
                           "other_speed": float(speed[b[k], o[k]]), "cos_angle": float(cosang[k]),
                           "ego_stopped": bool(ego_st[k]), "other_stopped": bool(oth_st[k]),
                           "active_rear": bool(rear[k]), "active_front": bool(front[k])})
    NuPlanTTC.forward = forward


def classify(ev) -> str:
    if ev["ego_stopped"]:
        return "EGO_STOPPED"
    if ev["other_stopped"]:
        return "STOPPED_TRACK"
    if ev["active_rear"]:
        return "ACTIVE_REAR"
    if ev["active_front"]:
        return "ACTIVE_FRONT"
    return "LATERAL/OTHER"


def main() -> int:
    import onpolicy_label as OL
    from planner import repair_last_heading
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    install_hook()
    T = np.load(G.TR)
    ix = {str(t): i for i, t in enumerate(T["token"])}
    top = np.load(V.SLOW_RANKS)["top"]
    Hc = {r: G.read_csv(G.REP_CSV.format(r, r)) for r in range(8)}
    props = {}
    for c in glob.glob(f"{V.WD}/queue_v5rep/props_r0_*.jsonl*"):
        for line in open(c, encoding="utf-8"):
            r = json.loads(line)
            props[r["token"]] = r
    lines = G.lines(f"{V.WD}/v5rep", "onpolicy_set")
    seen, cases = set(), []
    for r in lines:
        if r["token"] in seen or len(seen) >= a.max_tokens:
            continue
        seen.add(r["token"])
        i = ix[r["token"]]
        sl = r["slow"]
        top_i = [int(x) for x in top[i]]
        for slot, s, f in zip(sl["slots"], sl["src"], sl["factor"]):
            if f != 0.75:
                continue
            h = Hc[top_i.index(s)].get(r["token"])
            lab_nc = V.trainer_targets(r["targets"][slot])[0]
            lab_col, har_col = lab_nc < 0.5, h[2][0] < 0.999
            if lab_col != har_col or (lab_col and har_col):
                cases.append({"token": r["token"], "log": r["log_name"], "slot": slot, "src": s,
                              "label_collision": lab_col, "harness_collision": har_col,
                              "served": np.stack([np.asarray(r["traj"][slot]), np.asarray(r["yaw"][slot])[:, None]], -1)
                              if False else np.concatenate([np.asarray(r["traj"][slot], float),
                                                            np.asarray(r["yaw"][slot], float)[:, None]], -1)})
    S = OL.Scorer(0, 2)
    orig_roll = S.SP.score_proposal_rollout

    def roll(*args, **kw):
        CUR["call"] += 1
        return orig_roll(*args, **kw)
    S.SP.score_proposal_rollout = roll
    by_log: dict = {}
    for c in cases:
        by_log.setdefault(c["log"], []).append(c)
    out = []
    for lg, cs in by_log.items():
        scs = S.scenarios(lg, sorted({c["token"] for c in cs}))
        for c in cs:
            pr = props[c["token"]]
            tr = np.asarray(pr["teacher"], float)
            cand = repair_last_heading(c["served"])                    # exactly what v5 labelled
            EVENTS.clear()
            CUR["call"] = -1
            got, _nd = S.score(scs[c["token"]], pr, [("teacher", OL._t(tr[:, :2]), OL._t(tr[:, 2])),
                                                     ("cand", OL._t(cand[:, :2]), OL._t(cand[:, 2]))])
            ev = [e for e in EVENTS if e["call"] == 1]
            first = classify(ev[0]) if ev else "NO_EVENT_IN_TTC_HOOK"
            out.append({k: c[k] for k in ("token", "slot", "src", "label_collision", "harness_collision")} |
                       {"relabel_collision": float(got["cand"].get("collision.NuPlanCollision.info", float("nan"))),
                        "first_event": first, "first_event_detail": ev[0] if ev else None,
                        "event_types": dict(Counter(classify(e) for e in ev))})
    groups = {"label_col_harness_no": [o for o in out if o["label_collision"] and not o["harness_collision"]],
              "label_no_harness_col": [o for o in out if not o["label_collision"] and o["harness_collision"]],
              "both_collision (control)": [o for o in out if o["label_collision"] and o["harness_collision"]]}
    res = {"_label": "DIAGNOSTIC ONLY (no gate, no label changed): the teacher's first collision on the 0.75x copies, "
                     "classified with nuPlan's own collision-type rule (the teacher's NuPlanTTC quantities)",
           "tokens": len(seen), "cases": len(out),
           "relabel_reproduces_label": sum(1 for o in out if (o["relabel_collision"] >= 0.5) == o["label_collision"]),
           "first_event_counts": {g: dict(Counter(o["first_event"] for o in v)) for g, v in groups.items()},
           "n": {g: len(v) for g, v in groups.items()}, "cases_detail": out}
    os.makedirs(G.OUTD, exist_ok=True)
    json.dump(res, open(os.path.join(G.OUTD, "diag_collision_types.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in ("tokens", "cases", "relabel_reproduces_label", "n", "first_event_counts")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
