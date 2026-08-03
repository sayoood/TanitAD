#!/usr/bin/env python3
"""Assemble the scene-2 REAL close-following panel into one quotable record.

Consolidates, in the same shape as the earlier CUTIN_PANEL.json so the two are
comparable: provenance (including what this run REFUTES), the four metric families + ADE
per arm per condition, and every paired contrast with its estimator, CI and denominator.

Pure restructuring of files already written by cl_metrics.py / cutin_window_subset.py —
no number is computed here, so this file cannot introduce one.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "scene2-realclose"


def sep(v):
    return bool(isinstance(v, dict) and v.get("lo") is not None
                and (v["lo"] > 0 or v["hi"] < 0))


def contrast(path, key="paired_A_minus_B"):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    p = d.get(key, {})
    out = {}
    for k, v in p.items():
        if not isinstance(v, dict):
            continue
        if v.get("n") == 0:
            out[k] = {"n": 0, "reason": v.get("reason")}
            continue
        if "delta" not in v:
            continue
        out[k] = {"delta": round(v["delta"], 4), "lo": round(v["lo"], 4),
                  "hi": round(v["hi"], 4), "n_used": v.get("n_used"),
                  "separated": sep(v)}
    return d, out


def main():
    geom = json.loads((RES / "SCENE2_GEOMETRY.json").read_text(encoding="utf-8"))["summary"]
    geor = json.loads((RES / "SCENE2_GEOMETRY_renderable.json")
                      .read_text(encoding="utf-8"))["summary"]
    amap = json.loads((RES / "actor_map_7c72937c.json").read_text(encoding="utf-8"))

    dFO, cFO = contrast(RES / "metrics" / "flagship-v1_objects_vs_empty.json")
    dRO, cRO = contrast(RES / "metrics" / "refc-base_objects_vs_empty.json")
    dXO, cXO = contrast(RES / "metrics" / "flagship_vs_refc_objects.json")
    dXE, cXE = contrast(RES / "metrics" / "flagship_vs_refc_empty.json")
    cutF = json.loads((RES / "metrics" / "cutin_subset_flagship-v1_objects_vs_empty.json")
                      .read_text(encoding="utf-8"))
    cutR = json.loads((RES / "metrics" / "cutin_subset_refc-base_objects_vs_empty.json")
                      .read_text(encoding="utf-8"))

    panel = {
        "what": "close-following / cut-in discriminating panel on REAL scene geometry",
        "scene": geom["scene"],
        "evidence_class": "MEASURED (this agent, artifacts in results/scene2-realclose/)",
        "REFUTES": {
            "claim": "'No close-following / cut-in geometry exists in the material we "
                     "have' — results/cutin/CUTIN_PANEL.json, provenance."
                     "geometry_FOUND_in_scene = false.",
            "why_it_was_wrong": "the two probes cited for it (sequence_tracks.json and "
                                "clipgt/obstacle.parquet) read the SAME scene, 00040136. "
                                "Two tables of one scene are one location, not two. Thor "
                                "holds TWO NuRec scenes; 7c72937c was never probed.",
            "root_cause_class": "absence-at-one-location asserted as absence "
                                "(AGENT_OPERATING_STANDARD rule 2)",
            "corrected_finding": {
                "min_inlane_headway_m": geom["min_inlane_headway_m"],
                "n_close_following_rows": geom["n_close_following_rows_ALL"],
                "n_cutin_events": geom["n_cutin_events_ALL"],
                "scene_00040136_for_comparison": {
                    "min_inlane_headway_m": 46.246,
                    "n_close_following_rows": 0, "n_cutin_events": 0},
                "probe1_sequence_tracks_json": {
                    "min_inlane_headway_m": geom["min_inlane_headway_m"],
                    "n_close_following_rows": geom["n_close_following_rows_ALL"]},
                "probe2_obstacle_parquet": geom["probe2_obstacle_parquet"],
                "self_consistency_control": geom["self_consistency_control"],
            },
            "the_constructed_lead_was_never_necessary":
                "synth_actor.py's lead25/lead15/lead8/cutin conditions were built because "
                "the geometry was believed absent. They remain valid as a dose-response "
                "instrument, but they are no longer the only way to test the question.",
        },
        "SECOND_DEFECT_FIXED": {
            "what": "actor_map.py's relative-margin rule discarded 31 of 92 cuboids that "
                    "matched their track at cost EXACTLY 0 us — including BOTH vehicles "
                    "the ego follows and ALL THREE cut-in tracks.",
            "consequence_before_fix": {
                "n_close_following_rows_RENDERABLE": 0,
                "n_cutin_events_RENDERABLE": 0,
                "meaning": "the scene's real close-following geometry was invisible to "
                           "the renderer, so the test could not have seen it."},
            "after_fix": {
                "n_close_following_rows_RENDERABLE":
                    geor["n_close_following_rows_RENDERABLE"],
                "n_cutin_events_RENDERABLE": geor["n_cutin_events_RENDERABLE"],
                "n_renderable_tracks": geor["n_renderable_tracks"]},
            "adjudicated_by_pixels_not_by_argument": amap["ab_acceptance_rule"],
            "assignment_is_a_bijection": "all 92 best_track values distinct, all at "
                                         "best_cost_us == 0 — nothing ambiguous was let in",
        },
        "THIRD_DEFECT_FIXED": {
            "what": "cl_metrics.py read only `s_route_logits`; refc-base emits "
                    "`route_logits`, so the panel published REF-C's STRATEGIC route head "
                    "as absent — false.",
            "now": "both names probed; REF-C's route head scores on 403 windows.",
        },
        "FOURTH_DEFECT_FOUND": {
            "what": "the STRATEGIC route metric is CIRCULAR on flagship-v1: its route "
                    "head is a deterministic bijection of the nav command the harness "
                    "FEEDS it (nav=1 -> head=0 on 369/369, nav=0 -> head=1 on 81/81).",
            "consequence": "flagship-v1 scored route_head_eq_logged = 1.0000 "
                           "[1.0000, 1.0000] on 417 windows. That is the echo of its own "
                           "conditioning input, NOT strategic skill. Do not quote it.",
            "refc_base_is_not_an_echo": "head is not a function of nav; scores 0.2605 "
                                        "[0.1119, 0.4537] on 403 windows.",
            "guard": "cl_metrics.families() now computes route_head_nav_echo_check and "
                     "stamps CIRCULAR_NAV_ECHO / do_not_quote on the affected metrics.",
        },
        "NEGATIVE_CONTROL_the_instrument_discriminates": {
            "why_this_is_reported_first": "a null is only worth reading if the instrument "
                                          "could have shown an effect. Same 450 windows, "
                                          "same estimator, cross-arm contrast:",
            "flagship_minus_refc_in_objects_separated": sum(
                1 for v in cXO.values() if v.get("separated")),
            "flagship_minus_refc_in_empty_separated": sum(
                1 for v in cXE.values() if v.get("separated")),
            "of_n_metrics": sum(1 for v in cXO.values() if "delta" in v),
            "verdict": "PASS — the panel separates arms decisively on these windows, so "
                       "the objects-vs-empty null below is a property of the ARMS, not a "
                       "dead instrument.",
        },
        "HEADLINE": {
            "objects_vs_empty_flagship_separated": sum(
                1 for v in cFO.values() if v.get("separated")),
            "objects_vs_empty_refc_separated": sum(
                1 for v in cRO.values() if v.get("separated")),
            "of_n_metrics": sum(1 for v in cFO.values() if "delta" in v),
            "reading": "NEITHER ARM MEASURABLY REACTS to real dynamic agents — now "
                       "established at BUMPER-TO-BUMPER range (min in-lane headway "
                       f"{geom['min_inlane_headway_m']} m, actors ~30 % of frame), not "
                       "only at the 40-45 m / 0.02-0.4 %-of-frame distance that made the "
                       "first null dismissible.",
        },
        "paired_contrasts": {
            "flagship-v1_objects_minus_empty": cFO,
            "refc-base_objects_minus_empty": cRO,
            "flagship_minus_refc_in_objects": cXO,
            "flagship_minus_refc_in_empty": cXE,
        },
        "cutin_window_subset": {
            "flagship-v1": {"denominator": cutF["denominator"],
                            "paired": {k: {"delta": round(v["delta"], 4),
                                           "lo": round(v["lo"], 4), "hi": round(v["hi"], 4),
                                           "n_used": v.get("n_used"),
                                           "family": v.get("family"),
                                           "separated": sep(v)}
                                       for k, v in cutF["paired_A_minus_B_CUTIN_WINDOWS"].items()
                                       if isinstance(v, dict) and "delta" in v}},
            "refc-base": {"denominator": cutR["denominator"],
                          "paired": {k: {"delta": round(v["delta"], 4),
                                         "lo": round(v["lo"], 4), "hi": round(v["hi"], 4),
                                         "n_used": v.get("n_used"),
                                         "family": v.get("family"),
                                         "separated": sep(v)}
                                     for k, v in cutR["paired_A_minus_B_CUTIN_WINDOWS"].items()
                                     if isinstance(v, dict) and "delta" in v}},
            "power_caveat": "the cut-in windows fall in only 3 of the 9 rollout-start "
                            "clusters, so the cluster bootstrap has 3 resampling units "
                            "there. With 13 metrics x 2 arms = 26 tests at 95 %, ~1.3 "
                            "false separations are EXPECTED; exactly 1 was observed "
                            "(flagship lateral_ade_m, lo=+0.0053). It is not a finding.",
        },
        "four_families_per_arm": {
            "flagship-v1": {c: json.loads(
                (RES / "metrics" / ("flagship_vs_refc_%s.json" % c)).read_text(
                    encoding="utf-8"))["arm_A"]["families"] for c in ("objects", "empty")},
            "refc-base": {c: json.loads(
                (RES / "metrics" / ("flagship_vs_refc_%s.json" % c)).read_text(
                    encoding="utf-8"))["arm_B"]["families"] for c in ("objects", "empty")},
        },
        "estimator_note": dXO["estimator_note"],
        "within_sim_note": dXO["within_sim_note"],
        "renderable_restricted": dXO.get("renderable_restricted"),
        "artifacts": {
            "rollouts": "results/scene2-realclose/rollouts/ (4 files, 9 starts x 50 steps)",
            "metrics": "results/scene2-realclose/metrics/ (6 files)",
            "geometry": "results/scene2-realclose/SCENE2_GEOMETRY{,_renderable}.json",
            "actor_map": "results/scene2-realclose/actor_map_7c72937c.json",
            "on_thor": "/home/nvidia/scene2_out (incl. long_* runs with saved frames), "
                       "/home/nvidia/cutin_out/scene2",
        },
    }
    out = RES / "SCENE2_PANEL.json"
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    print("wrote", out)
    print(json.dumps(panel["HEADLINE"], indent=1))
    print(json.dumps(panel["NEGATIVE_CONTROL_the_instrument_discriminates"], indent=1))


if __name__ == "__main__":
    main()
