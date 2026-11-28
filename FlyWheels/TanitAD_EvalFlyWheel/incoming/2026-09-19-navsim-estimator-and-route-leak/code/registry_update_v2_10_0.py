#!/usr/bin/env python3
"""W2 (E3): CRITERIA_REGISTRY.json 2.9.0 -> 2.10.0. Re-runnable ONLY on 2.9.0 (it refuses
otherwise), so it can never be applied twice or on top of a sibling's newer registry.

Preserves the file's exact serialisation (indent=1, ensure_ascii=False, CRLF, no final
newline) — asserted by a byte-exact round trip of the UNMODIFIED file before any change, so
the diff contains W2's edits and nothing else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
REG = REPO / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json"
PKG = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak"
V2_SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
V1_SHA = "3e8291bfa89ff247231e0227778840cd0a036896"


def dump(d) -> bytes:
    return json.dumps(d, indent=1, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")


def main() -> int:
    raw = REG.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    if dump(d) != raw:
        print("REFUSED: the registry does not round-trip byte-exactly; not editing blind")
        return 2
    if d.get("version") != "2.9.0":
        print(f"REFUSED: expected version 2.9.0, found {d.get('version')!r}")
        return 2
    nv = d["benchmarks"]["navsim"]

    # ---- (b) EPDMS_v2 multipliers gain TLC ---------------------------------------- #
    nv["variants"]["EPDMS_v2"]["multipliers"] = ["NC", "DAC", "DDC", "TLC"]
    nv["variants"]["EPDMS_v2"]["multipliers_source"] = (
        "PUBLISHED-CODE autonomousvision/navsim@0a380a9 pdm_enums.py:165-171 MultiMetricIndex = "
        "NO_COLLISION, DRIVABLE_AREA, TRAFFIC_LIGHT_COMPLIANCE, DRIVING_DIRECTION; weights "
        "pdm_scorer.py:44-48 (EP 5, TTC 5, LK 2, HC 2, EC 2 = 16). MEASURED by E1 control C4: "
        "per-token score = NC*DAC*DDC*TLC*(5EP+5TTC+2LK+2HC+2EC)/16 on 220 rows, max |d| 1.1e-16. "
        "⛔ Until registry 2.10.0 this list read [NC, DAC, DDC] — TLC was missing while this "
        "block's own EPDMS_submetrics multiplied by it (caught by E1 and E2, 2026-09-19).")
    nv["variants"]["PDMS_v1"]["devkit_pin_required"] = V1_SHA
    nv["variants"]["PDMS_v1"]["source"] = (
        "PUBLISHED-CODE navsim v1.1.0 @3e8291b (unpacked archive, not a git checkout: file sha256 "
        "pdm_enums.py 827e7338…, pdm_scorer.py 02ab0750…): MultiMetricIndex = NO_COLLISION, "
        "DRIVABLE_AREA (pdm_enums.py:156-160); weights progress 5, ttc 5, comfortable 2, "
        "driving_direction 0 (pdm_scorer.py:38-42) -> denominator 12.")

    # ---- (c) the estimator gate: UNRESOLVED -> SETTLED ------------------------------- #
    nv.pop("ESTIMATOR_UNRESOLVED", None)
    nv["ESTIMATOR_RULE"] = (
        "SETTLED 2026-09-19 (W2/E3, pre-registered SPEC " + PKG + "/SPEC.md): the NavSim interval "
        "is a LOG-CLUSTER bootstrap — clusters = OpenScene `log_name`, B = 2000, percentile 95 %, "
        "n_clusters >= 8 (RG-14) — of the OFFICIAL aggregate (two-stage: the devkit's per-mapping-"
        "key product then a skip-NaN mean over keys; single-stage: skip-NaN mean over tokens). "
        "Implementation taniteval/adapters/navsim_ci.py. Scene-token resampling, the episode-"
        "cluster bootstrap under any other unit and overlapping_holdout_se FAIL. warmup (7 logs) "
        "and private_test_hard (logs not observable) can NEVER carry an interval.")
    g = nv["GATE_estimator_cluster_unit"]
    g.pop("admissible_until_settled", None)
    g.update({
        "status": "SETTLED 2026-09-19 (was UNRESOLVED through registry 2.9.0)",
        "rule": ("A NavSim artifact MUST declare `estimator.cluster_unit` = \"log_name\" and an "
                 "`estimator.interval` that is EITHER an honest {status: UNAVAILABLE, reason, n} "
                 "(a WORK ITEM) OR a navsim_log_cluster_bootstrap interval with resample_unit "
                 "log_name, 8 <= n_clusters <= the split's log count, the protocol's official "
                 "aggregation and n_boot >= 2000. ⛔ Anything else FAILS: a scene-token / mapping-"
                 "key / episode-cluster / overlapping_holdout_se interval, n_clusters < 8, "
                 "n_clusters above the split's log count (a finer unit was resampled), an unnamed "
                 "aggregation, a missing cluster_unit, or no interval block at all."),
        "why": ("MEASURED (" + PKG + "/raw/split_census.json): scenes OVERLAP within a log — navtest "
                "consecutive scenes share a median 13 of 14 frames and each key frame sits in 5.55 "
                "scenes; navhard's orig/prev tokens share 11 of 12 frames and are coupled by the EC "
                "term — while logs share nothing (0 shared tokens, 0 time-overlapping segment "
                "pairs, min gap 11.5 s) and every stage-2 scene lives in its key's log (5,462/5,462 "
                "navhard, 204/204 warmup). Resampling overlapping scene tokens as independent "
                "understates the variance — the overlapping_holdout_se family on a borrowed "
                "benchmark. The CONTROL reads its known value: the OpenScene standard `test` split "
                "(frame_interval null) has frame-reuse factor 1.0000."),
        "cluster_unit": "log_name",
        "sensitivity_unit": "nuplan_drive",
        "sensitivity_rule": ("computed beside the primary when >= 8 drives; a PAIRED lever claim is "
                             "`separated` only if separated under log_name AND nuplan_drive (no "
                             "tunable threshold); with < 8 drives the claim is scoped 'log level "
                             "only'. And per H-ESTIM-SEED-1 a separated interval is NECESSARY, not "
                             "sufficient — a lever claim also needs a replicate arm."),
        "estimators_admissible": ["navsim_log_cluster_bootstrap",
                                  "paired_navsim_log_cluster_bootstrap"],
        "min_clusters": 8,
        "min_clusters_source": "RG-14 — tools/release_gate.py:85 EPISODE_CLUSTER_FLOOR = 8",
        "n_boot_min": 2000,
        "aggregation_by_protocol": {
            "EPDMS_v2_navhard_two_stage": "two_stage_mapping_key_mean",
            "EPDMS_v2_warmup_two_stage": "two_stage_mapping_key_mean",
            "EPDMS_v2_private_test_hard_two_stage": "two_stage_mapping_key_mean",
            "EPDMS_v2_navtest_single_stage": "single_stage_token_mean",
            "PDMS_v1_navtest": "single_stage_token_mean"},
        "aggregation_sources": {
            "two_stage_mapping_key_mean": "run_pdm_score.py::calculate_individual_mapping_scores "
                                          "L242-290 @0a380a9",
            "single_stage_token_mean": "run_pdm_score_one_stage.py L293 @0a380a9 (EPDMS v2); "
                                       "run_pdm_score.py L144 @3e8291b (PDMS v1)"},
        "max_clusters_by_protocol": {
            "EPDMS_v2_navhard_two_stage": 76,
            "EPDMS_v2_warmup_two_stage": 7,
            "EPDMS_v2_navtest_single_stage": 136,
            "PDMS_v1_navtest": 136,
            "EPDMS_v2_private_test_hard_two_stage": None},
        "max_clusters_note": ("the split's `log_name` count, MEASURED by two agreeing derivations "
                              "(the devkit's own filter_scenes executed verbatim + a re-derivation). "
                              "None = not observable locally (private_test_hard: 0/140 tokens in "
                              "the test metadata vs the navhard control 450/450)."),
        "can_rule_by_protocol": {
            "EPDMS_v2_navhard_two_stage": "YES — 76 logs / 34 drives",
            "EPDMS_v2_navtest_single_stage": "YES — 136 logs / 44 drives",
            "PDMS_v1_navtest": "YES — 136 logs / 44 drives",
            "EPDMS_v2_warmup_two_stage": "⛔ NEVER — 7 logs < 8 (D-BENCH-PORT, re-measured)",
            "EPDMS_v2_private_test_hard_two_stage": ("⛔ NEVER — log identities and per-scene "
                                                     "scores are not released")},
        "reproduces_official": ("MEASURED: the full-sample point reproduces the devkit's "
                                "`extended_pdm_score_combined` on E1's real CV warmup run exactly "
                                "(0.1853562745165113, |d| 0.0); the per-key aggregation equals the "
                                "devkit's own function on 4 fixtures incl. physically duplicated "
                                "logs (" + PKG + "/raw/devkit_aggregation_reference*.json)."),
        "admissible_refusal": "UNAVAILABLE with a reason and n — a WORK ITEM, never a pass",
        "spec": PKG + "/SPEC.md",
        "census": PKG + "/raw/split_census.json",
        "implementation": "taniteval/adapters/navsim_ci.py",
        "keys": ["estimator.cluster_unit", "estimator.interval"],
    })
    nv["GATE_ego_status_enforcement"]["checker_rule"] = (
        "tools/criteria_check.py: PASS = `protocol.ego_status_enforcement` is an object with a "
        "non-empty `mechanism` AND non-empty structured `evidence` (a dict/list, or a string naming "
        "an artifact file). An arm that does NOT claim vision-only may instead DECLARE the gate "
        "not applicable with a `reason` (REFUSED — a work item). ⛔ FAIL: a vision-only claim "
        "(`vision_only_claimed` or `protocol.vision_only` true) without mechanism AND evidence; a "
        "bare string; an object with neither. ABSENT: no block at all.")
    nv["GATE_modality_label"]["checker_rule"] = (
        "tools/criteria_check.py: PASS = `protocol.sensor_set` AND `protocol.setting` are both "
        "non-empty strings. ⛔ A refusal object is not a label and FAILS; a missing field is ABSENT.")
    cp = nv["GATE_no_cross_protocol_comparison"]
    cp["devkit_pins"] = {
        "EPDMS_v2_navhard_two_stage": [V2_SHA],
        "EPDMS_v2_warmup_two_stage": [V2_SHA],
        "EPDMS_v2_private_test_hard_two_stage": [V2_SHA],
        "EPDMS_v2_navtest_single_stage": [V2_SHA],
        "PDMS_v1_navtest": [V1_SHA]}
    cp["checker_rule"] = (
        "tools/criteria_check.py: PASS = `protocol.navsim_protocol` is in closed_set AND "
        "`protocol.devkit_sha` is a full 40-hex SHA registered in devkit_pins for THAT protocol AND "
        "(when present) `benchmark.navsim.variant` names the protocol's variant. ⛔ A v1 PDMS "
        "stamped with the v2 SHA FAILS — 'v1 PDMS from main silently computes EPDMS' — as does an "
        "unregistered SHA (register it first: the SHA is part of the column).")

    # ---- the criteria: route leak settled -------------------------------------------- #
    for crit in nv["criteria"]:
        if crit["id"] == "navsim.route_leak_check":
            crit["note"] = (
                "SETTLED 2026-09-19 (W2/E3) — see ROUTE_LEAK_VERDICT: PARTIAL, a ROUTE-LEVEL ORACLE. "
                "The command is a function of (pose now, route, map) only (reproduced 1,902/1,902), "
                "but the route is the expert's own path at roadblock granularity. tools/"
                "criteria_check.py: an arm that CONSUMES driving_command must carry the settled "
                "verdict (a refusal or 'UNVERIFIED' FAILS); an arm that consumes no route may "
                "declare it not applicable with a reason.")
        if crit["id"] == "navsim.score":
            crit["checker_rule"] = ("a `benchmark.navsim.score` block whose `column` is not "
                                    "`score` FAILS (the column trap).")

    # ---- (d) the route-leak verdict ------------------------------------------------ #
    nv["ROUTE_LEAK_VERDICT"] = {
        "verdict": "PARTIAL — ROUTE-LEVEL ORACLE",
        "settled": "2026-09-19 (W2/E3)",
        "evidence_class": "PUBLISHED-CODE (mechanism) + MEASURED (four empirical probes)",
        "one_line": ("NavSim's driving_command is computed from (the ego pose NOW, nuPlan's route, the "
                     "map) — not from the expert's future trajectory — but the route itself is the "
                     "expert's own driven path at ROADBLOCK granularity, so the command carries the "
                     "expert's future ROUTE CHOICE (which way at the next junction). It is not a "
                     "trajectory leak (no speed, stop or lane-change information)."),
        "mechanism_at_source": [
            "OpenDriveLab/OpenScene@7286074 DriveEngine/process_data/create_openscene_metadata.py:95-99 "
            "— roadblock_ids = lidar_pc.scene.roadblock_ids (the nuPlan DB scene row)",
            "…create_openscene_metadata.py:122-127 — driving_command = get_driving_command(ego_pose "
            "NOW, map_api, roadblock_ids)",
            "…helpers/driving_command.py:40-100 — route correction, Dijkstra centreline, target 20 m "
            "ahead, |y| >= 2 m -> left/right else forward; unknown when the route is unrecoverable",
            "nuplan-devkit@ce3c323 docs/nuplan_schema.md:181-190 — a scene 'stores a goal … that is a "
            "future ego pose from beyond that scene … a sequence of road blocks to navigate towards "
            "the goal'",
            "nuplan-devkit@ce3c323 nuplan_scenario.py:171-177 — `_route_roadblock_ids`: 'Route "
            "roadblock ids extracted from expert trajectory'; :238-242 get_route_roadblock_ids reads "
            "scene.roadblock_ids (nuplan_scenario_queries.py:307-325)",
            "autonomousvision/navsim@0a380a9 dataclasses.py:204, :429, :468, :588 — the devkit only "
            "READS driving_command from the pickles; it never computes it (git grep over the pinned "
            "tree: 0 computations; control: roadblock_ids 5 hits in the same file)"],
        "measured": {
            "reproduction": "1,902/1,902 stage-1 frames (turning 739/739) — OpenScene's own function on "
                            "(pose, stored route, map) reproduces the stored command exactly",
            "route_starts_behind_the_ego": "83.5 % of 1,921 nuPlan scenes (median ego index 1) — a "
                                           "navigation route computed at scene start would start AT the ego",
            "route_ahead_is_the_driven_path": "79.8 % of scenes: every route block ahead is driven by "
                                              "the logged ego or lies beyond the recorded segment; NULL "
                                              "control (no-future straight-ahead route) 34.6 %; visits in "
                                              "route order 96.4 %",
            "counterfactual_turning_commands": "a route built from the ego's FUTURE path reproduces "
                                               "710/739 = 96.1 % of turning commands; a NO-FUTURE "
                                               "straight-ahead route 491/739 = 66.4 %",
            "agreement_with_future_path": "best threshold rule (lateral offset of the logged path "
                                          "after 20 m of driven arc, |y| >= 2-3 m) 94.2-94.7 % on "
                                          "65,783 frames; 4 s horizon rules <= 86 % — below the 99 % "
                                          "'deterministic function of the future trajectory' bar",
            "stage_two": "every synthetic scene's command AND route are COPIED from the original frame "
                         "at the same timestamp (5,462/5,462 navhard, 204/204 warmup) — the command the "
                         "EXPERT had there, not one recomputed at the perturbed start (recomputation "
                         "would agree on 83.9 % of 778 sampled)"},
        "consequences": [
            "a driving_command-conditioned NavSim number is 'driving with an ORACLE ROUTE' — comparable "
            "within NavSim (every agent receives it) but never evidence of route/strategic skill",
            "the command may never be the LABEL of a route/strategic head (the route-echo defect: "
            "flagship v1 369/369; D-REFAV1-ROUTE-LABEL-IS-THE-NAV)",
            "report the command-withheld arm PAIRED with the command arm; their delta is the value of "
            "the oracle route to the agent (E2: A1 - A3 = +0.0124 on warmup stage 2)",
            "admissible under the goal/situation-disjoint rule (it is not the situation classifier's "
            "output) but optimistic by construction, exactly like refcv4b's 'oracle, provenance "
            "ego-future' nav token"],
        "artifacts": [PKG + "/RESULT.md", PKG + "/raw/route_leak_probe.json",
                      PKG + "/raw/stage2_command_provenance.json",
                      PKG + "/raw/source_probes/SHA256SUMS.txt"],
        "consumed_by": "taniteval/adapters/navsim.py::route_leak_check (default) — pinned equal by "
                       "taniteval/tests/test_navsim_adapter.py",
    }

    # ---- (e) the navsim_v1 benchmark block --------------------------------------------- #
    d["benchmarks"]["navsim_v1"] = {
        "pending": False,
        "protocol_tag": "PDMS_v1_navtest",
        "artifact_key": "benchmark.navsim",
        "_artifact_key_note": ("one adapter for both NavSim generations "
                               "(taniteval/adapters/navsim.py::build_artifact, variant PDMS_v1); the "
                               "protocol tag distinguishes them and every benchmarks.navsim GATE_* "
                               "applies unchanged. `benchmark.navsim_v1` is ALSO recognised as a claim."),
        "protocol_doc": "products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md",
        "primary": ["arXiv:2406.15349"],
        "devkit_pin": f"autonomousvision/navsim v1.1.0 @{V1_SHA} (pinned 2025-06-05)",
        "devkit_pin_required": V1_SHA,
        "local_tree": "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/ "
                      "(unpacked archive, not a git checkout — cite file sha256)",
        "formula": "PDMS_v1 = NC * DAC * (5*EP + 5*TTC + 2*C) / 12",
        "multipliers": ["NC", "DAC"],
        "weighted": {"EP": 5, "TTC": 5, "C": 2},
        "denominator": 12,
        "ddc_note": "DDC exists in v1 code with weight 0.0 (pdm_scorer.py:38-42 @v1.1) — a navtest DDC "
                    "column contributes NOTHING to PDMS v1",
        "score_column": "score (PDMResults.score, dataclasses.py:568 @v1.1; v1 has no pdm_score column)",
        "aggregation": "single_stage_token_mean — run_pdm_score.py:144 @v1.1: "
                       "pdm_score_df.drop(columns=['token','valid']).mean(skipna=True)",
        "split": {"name": "navtest", "scene_filter": "4 history + 10 future frames, frame_interval 1",
                  "n_tokens": 12146, "n_logs": 136, "n_drives": 44,
                  "source": PKG + "/raw/split_census.json (v2 navtest; the v1.1 scene_filter/"
                                  "navtest.yaml has the same 136 + 12,146 entries — W3 to pin the "
                                  "byte identity)"},
        "sources": {"pdm_enums.py": "827e7338b8e553684f87a0fc27a16cff3e3ee382e2a18271145e7e549a6c52bc",
                    "pdm_scorer.py": "02ab075017d4edb1c8bba6476ef1674b84d43dbca0f599df87cfe052b2dd5211",
                    "run_pdm_score.py": "7299fa4617e4d5116732e1c9fcf92a991775cc476bfd00eacd8f6394792191d1",
                    "dataclasses.py": "17d96f48101982d4a0842bb4cc1de441672207ad9fed3e460bbbdcf74440a32b",
                    "scene_filter/navtest.yaml": "61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd"},
        "gates": ("every benchmarks.navsim GATE_* applies; the cross-protocol gate REQUIRES this pin for "
                  "PDMS_v1_navtest — 'v1 PDMS from main silently computes a different metric'"),
        "comparable_ladder": "benchmarks.navsim.GATE_modality_label.comparable_ladder_perception_free_front_only "
                             "(LAW 83.8 / World4Drive 85.1 / Epona 86.1 / Drive-JEPA 89.0 — v1 PDMS)",
    }

    # ---- W1's closed set: nuScenes + internal, each in its own block ------------------ #
    plan = d["benchmarks"]["nuscenes"]["tasks"]["planning_openloop"]
    plan["protocol_tags"] = ["nuScenes_OL_L2_stp3", "nuScenes_OL_L2_uniad"]
    plan["protocol_tags_map"] = {"nuScenes_OL_L2_stp3": "stp3 (L2 averaged UP TO the timestep)",
                                 "nuScenes_OL_L2_uniad": "uniad (L2 AT the timestep)"}
    plan["protocol_tags_note"] = ("the suite's tags (taniteval/taniteval/bench/schema/"
                                  "summary.schema.json, W1); claim_bearing is false by API — the "
                                  "task stays INADMISSIBLE as a criterion.")
    d["tiers"]["internal_protocols"] = {
        "TanitAD_T1_refc_physicalai": {
            "tier": "T1",
            "what": "the internal T1 panel (taniteval/tools/refcv3_arm.py / four_families, "
                    "episode-cluster bootstrap) on the PhysicalAI eval clip set",
            "estimator": "episode_cluster_bootstrap (taniteval/ci.py) — the NavSim log-cluster rule "
                         "does not apply here",
            "note": "an internal protocol, never merged into a benchmark closed set"}}
    d["protocol_tags"] = {
        "rule": ("ONE source of truth for protocol tags: the UNION of the blocks listed in `sources`, "
                 "each tag living in its own block and never merged into another benchmark's closed "
                 "set. tools/criteria_check.py FAILS an artifact whose tag (at any `key_paths`) is "
                 "outside the union; a NavSim artifact must additionally carry a NavSim tag. The suite "
                 "schema (taniteval/taniteval/bench/schema/summary.schema.json) must equal the union "
                 "— W1 tests the equality."),
        "sources": ["benchmarks.navsim.GATE_no_cross_protocol_comparison.closed_set",
                    "benchmarks.nuscenes.tasks.planning_openloop.protocol_tags",
                    "tiers.internal_protocols"],
        "key_paths": ["protocol.navsim_protocol", "protocol.protocol_tag",
                      "protocol.nuscenes_protocol", "protocol_tag"],
        "suite_record_key_paths": ["protocol"],
        "suite_record_schema_prefix": "taniteval.bench.",
        "key_paths_note": ("A bare top-level `protocol` STRING is read as a tag ONLY on a suite record "
                           "(`schema` starting with taniteval.bench.): MEASURED 2026-09-19, 24 banked "
                           "artifacts carry a free-text `protocol` sentence (e.g. 'PSEUDO-SIMULATION. "
                           "Perturbed observation states…'), and reading those as tags would "
                           "manufacture violations — the wrong-scope probe the registry's "
                           "`applicability.why` forbids."),
        "function": "tools/criteria_check.py::registered_protocol_tags(registry)"}

    # ---- changelog + version ------------------------------------------------------------ #
    d["version"] = "2.10.0"
    d["changelog"].append({
        "version": "2.10.0", "date": "2026-09-19", "by": "EvalFlyWheel W2 (E3 resumed)",
        "change": ("(a) tools/criteria_check.py now EVALUATES benchmarks.navsim — the five criteria and "
                   "all four blocking GATE_* — and FAILS an artifact whose protocol tag is outside the "
                   "registered union; (b) EPDMS_v2 multipliers gain TLC; (c) GATE_estimator_cluster_"
                   "unit UNRESOLVED -> SETTLED (log-cluster bootstrap, SPEC + census); (d) "
                   "ROUTE_LEAK_VERDICT = PARTIAL, route-level oracle; (e) benchmarks.navsim_v1; plus "
                   "the protocol_tags union with nuScenes + internal tags in their own blocks."),
        "why": ("MEASURED 2026-09-19 by E1 and independently by E2: criteria_check.py evaluated NONE of "
                "benchmarks.navsim, so all four blocking NavSim gates were unenforced and every NavSim "
                "artifact passed; and the EPDMS_v2 multiplier list omitted TLC while this registry's "
                "own EPDMS_submetrics multiplied by it."),
        "ships_with": ("tools/tests/test_criteria_check.py — a deliberate-regression arm per gate that "
                       "must go RED; taniteval/tests/test_navsim_ci.py — literal analytic targets, the "
                       "devkit-reference literal, the real-run reproduction and a RED code mutation.")})
    out = dump(d)
    REG.write_bytes(out)
    print(f"wrote {REG} version {d['version']} ({len(raw)} -> {len(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
