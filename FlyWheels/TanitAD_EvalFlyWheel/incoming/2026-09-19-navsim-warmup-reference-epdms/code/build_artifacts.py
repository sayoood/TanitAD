#!/usr/bin/env python3
"""E1: one TanitEval artifact per reference agent + criteria_check + a NavSim-gate evaluator.

Uses ``taniteval/adapters/navsim.py`` UNMODIFIED (stream E3 owns it). Gaps it does not cover are
filled HERE and each is listed in ``_e1_adapter_gaps`` so E3 can decide whether to absorb them:
  G1 devkit CSV columns are long + stage-suffixed (``no_at_fault_collisions_stage_one``);
     ``submetrics_from_row`` looks up short keys (NC, DAC, ...) and has no stage dimension.
  G2 ``build_artifact`` emits no ``estimator.cluster_unit`` (registry GATE_estimator_cluster_unit).
  G3 no ``protocol.ego_status_enforcement`` (GATE_ego_status_enforcement).
  G4 no ``protocol.sensor_set`` / ``protocol.setting`` (GATE_modality_label).
  G5 no ``protocol.navsim_protocol`` / ``protocol.devkit_sha`` (GATE_no_cross_protocol_comparison).
  G6 no ``protocol.corpus`` / ``controls`` (hyg.parity, ctrl.floor_comparison).
And: ``tools/criteria_check.py`` evaluates families / hygiene / leak guards ONLY — never
``benchmarks.navsim`` — so the four NavSim BLOCKING gates are evaluated by ``navsim_gates`` below,
whose own mutation arms must go RED (``gate_mutations``).

Run with the TanitAD venv:  PYTHONPATH=D:/Projects/TanitAD/stack
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe code/build_artifacts.py --raw raw --arms A1 A2
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "adapters"))
sys.path.insert(0, str(REPO / "tools"))
import navsim as ad            # noqa: E402  taniteval/adapters/navsim.py (path-bootstraps taniteval)
import criteria_check as cc    # noqa: E402  tools/criteria_check.py (import only; not edited)

SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
LONG = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance",
        "TLC": "traffic_light_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
        "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
SUMMARY = {"stage_one": "extended_pdm_score_stage_one", "stage_two": "extended_pdm_score_stage_two",
           "combined": "extended_pdm_score_combined"}
NO_CI_REASON = ("⛔ warmup NEVER carries a CI — D-BENCH-PORT (PI approval 2026-08-29, Project Steering/"
                "GOALS_AND_CLAIMS.md): 7 log groups < the RG-14 floor of 8 (products/P7-TanitEval/RELEASE_GATE.md). "
                "The 7 is re-MEASURED from scene_filter/warmup_two_stage.yaml `log_names`. Independently, the NavSim "
                "cluster unit (scene token vs nuPlan log) is unsettled — " + ad.ESTIMATOR_UNSETTLED_REASON)


def read_csv_rows(path: Path) -> dict:
    with open(path, newline="") as f:
        return {r["token"]: r for r in csv.DictReader(f)}


def short_row(row: dict, stage: str) -> dict:
    """G1: long stage-suffixed devkit columns -> the adapter's short keys."""
    suf = "" if stage == "one_stage_runner" else f"_stage_{stage}"
    out = {}
    for s, l in LONG.items():
        v = row.get(l + suf)
        if v not in (None, "", "nan"):
            out[s] = float(v)
    if row.get("score") not in (None, "", "nan"):
        out["score"] = float(row["score"])
    return out


def build_win(hooks: list, stage_one_tokens: set):
    calls = [c for c in hooks if c.get("token") in stage_one_tokens and c.get("human_poses") is not None]
    calls.sort(key=lambda c: c["token"])
    pred = np.asarray([c["agent_poses"] for c in calls], dtype=np.float64)
    gt = np.asarray([c["human_poses"] for c in calls], dtype=np.float64)
    v0 = np.asarray([c["v0_mps"] for c in calls], dtype=np.float64)
    toks = [c["token"] for c in calls]
    try:
        win = ad.scenes_to_win(pred, gt, frame="ego", origin_included=False, dt_s=0.5,
                               scene_tokens=toks, ego_speed_mps=v0, verify=True)
        fv = "verified"
    except Exception as e:                                                  # noqa: BLE001
        win = ad.scenes_to_win(pred, gt, frame="ego", origin_included=False, dt_s=0.5,
                               scene_tokens=toks, ego_speed_mps=v0, verify=False)
        win["_navsim"]["frame_verification"] = {"status": "FAILED", "reason": f"{type(e).__name__}: {e}"[:600],
                                                "n": len(toks)}
        fv = "FAILED"
    return win, fv, len(toks)


def navsim_gates(art: dict, reg: dict) -> dict:
    """The four benchmarks.navsim BLOCKING gates + the five navsim criteria (E1's evaluator)."""
    nv = reg["benchmarks"]["navsim"]
    out = {"criteria": {}, "gates": {}}
    for crit in nv["criteria"]:
        st, det = cc.classify(art, crit)
        out["criteria"][crit["id"]] = {"state": st, "detail": det[:160]}
    # GATE_estimator_cluster_unit
    f1, cu = cc._dig(art, "estimator.cluster_unit")
    f2, iv = cc._dig(art, "estimator.interval")
    numeric_ci = isinstance(iv, dict) and any(k in iv for k in ("lo", "hi", "ci", "ci_lo", "ci_hi", "ci95"))
    ok = f1 and cu is not None and isinstance(iv, dict) and str(iv.get("status", "")).upper() == "UNAVAILABLE" \
        and bool(iv.get("reason")) and iv.get("n") is not None and not numeric_ci
    out["gates"]["navsim.estimator_unit"] = "PASS" if ok else "FAIL"
    # GATE_ego_status_enforcement
    f3, ee = cc._dig(art, "protocol.ego_status_enforcement")
    if not f3 or not isinstance(ee, dict):
        out["gates"]["navsim.ego_enforcement"] = "FAIL"
    elif ee.get("vision_only_claimed"):
        out["gates"]["navsim.ego_enforcement"] = "PASS" if (ee.get("mechanism") and ee.get("evidence")) else "FAIL"
    else:
        out["gates"]["navsim.ego_enforcement"] = ("NOT_APPLICABLE_DECLARED" if ee.get("reason") else "FAIL")
    # GATE_modality_label
    f4, ss = cc._dig(art, "protocol.sensor_set")
    f5, se = cc._dig(art, "protocol.setting")
    out["gates"]["navsim.modality_label"] = "PASS" if (f4 and f5 and isinstance(ss, str) and ss and isinstance(se, str) and se) else "FAIL"
    # GATE_no_cross_protocol_comparison
    closed = nv["GATE_no_cross_protocol_comparison"]["closed_set"]
    f6, npc = cc._dig(art, "protocol.navsim_protocol")
    f7, dsha = cc._dig(art, "protocol.devkit_sha")
    out["gates"]["navsim.cross_protocol"] = "PASS" if (f6 and npc in closed and f7 and isinstance(dsha, str)
                                                        and re.fullmatch(r"[0-9a-f]{40}", dsha)) else "FAIL"
    return out


def gate_mutations(art: dict, reg: dict) -> dict:
    """Each mutation reintroduces a real failure; the matching gate MUST read FAIL."""
    muts = {
        "navsim.estimator_unit": lambda a: a["estimator"].update(interval={"lo": 0.1, "hi": 0.2, "estimator": "bootstrap"}),
        "navsim.ego_enforcement": lambda a: a["protocol"].update(ego_status_enforcement={"vision_only_claimed": True, "note": "we did not use it"}),
        "navsim.modality_label": lambda a: a["protocol"].pop("sensor_set"),
        "navsim.cross_protocol": lambda a: a["protocol"].update(navsim_protocol="EPDMS_v2_navtest"),
    }
    res = {}
    for gate, mut in muts.items():
        b = copy.deepcopy(art)
        mut(b)
        res[gate] = navsim_gates(b, reg)["gates"][gate]
    b = copy.deepcopy(art)
    b["protocol"]["devkit_sha"] = "0a380a9"                     # short SHA must fail too
    res["navsim.cross_protocol(short_sha)"] = navsim_gates(b, reg)["gates"]["navsim.cross_protocol"]
    res["all_red"] = all(v == "FAIL" for k, v in res.items() if k != "all_red")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--split", default="warmup_two_stage")
    ap.add_argument("--controls", type=Path, default=None)
    a = ap.parse_args()
    reg = json.loads((REPO / "products/P7-TanitEval/CRITERIA_REGISTRY.json").read_text(encoding="utf-8"))
    import yaml
    sf = yaml.safe_load(open(f"C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"))
    S1, S2, n_logs = set(sf["tokens"]), set(sf["reactive_synthetic_initial_tokens"]), len(sf["log_names"])
    controls = json.loads(a.controls.read_text()) if a.controls and a.controls.exists() else None
    outdir = a.raw / "artifacts"
    outdir.mkdir(exist_ok=True)
    summary = {}
    SPEC = {
        "A1": dict(agent="constant_velocity_agent", runner="two_stage", one_stage_arm="A1b",
                   ii=dict(cameras=False, lidar=False, ego_velocity=True, ego_acceleration=False,
                           ego_pose_history=False, driving_command=False),
                   sensor_set="none — ego-status only (ego_statuses[-1].ego_velocity)",
                   setting="kinematic floor: constant-velocity straight-line extrapolation; perception-free, map-free, route-free",
                   goal="none — constant_velocity_agent.py:31-46 reads only ego_statuses[-1].ego_velocity; no route/goal input"),
        "A2": dict(agent="human_agent", runner="two_stage", one_stage_arm="A2b",
                   ii=dict(cameras=False, lidar=False, ego_velocity=False, ego_acceleration=False,
                           ego_pose_history=False, driving_command=False),
                   sensor_set="none — PRIVILEGED log replay (Scene.get_future_trajectory: the recorded future)",
                   setting="privileged reference, NOT a planner: replays the logged human future; human-penalty filter applies on stage 1",
                   goal="none — log replay of the recorded future (privileged; not an admissible inference input)"),
    }
    for arm in a.arms:
        sp = SPEC[arm]
        csvs = sorted((a.raw / arm).glob("devkit_*.csv"))
        rows = read_csv_rows(csvs[-1]) if csvs else {}
        hooks = json.load(open(a.raw / arm / f"{arm}_hooks.json"))["pdm_score_calls"] if (a.raw / arm / f"{arm}_hooks.json").exists() else []
        one = sorted((a.raw / sp["one_stage_arm"]).glob("devkit_*.csv"))
        one_rows = read_csv_rows(one[-1]) if one else {}
        one_hooks_p = a.raw / sp["one_stage_arm"] / f"{sp['one_stage_arm']}_hooks.json"
        one_hooks = json.load(open(one_hooks_p))["pdm_score_calls"] if one_hooks_p.exists() else []
        # stage-one poses: prefer the two-stage run's hooks; fall back to the one-stage run's
        win, fv, n_win = build_win(hooks if any(c.get("token") in S1 for c in hooks) else one_hooks, S1)
        body = {t: r for t, r in rows.items() if t not in SUMMARY.values()}
        n_valid = {st: sum(1 for t, r in body.items() if t in toks and r.get("valid") == "True")
                   for st, toks in (("stage_one", S1), ("stage_two", S2))}
        combined = rows.get(SUMMARY["combined"])
        two_stage_defined = bool(combined and combined.get("score") not in (None, "", "nan")
                                 and n_valid["stage_one"] == len(S1) and n_valid["stage_two"] == len(S2))
        if two_stage_defined:
            epdms = ad.read_epdms(short_row(combined, "one") | {"score": float(combined["score"])})
            epdms["by_stage"] = {st: float(rows[SUMMARY[st]]["score"]) for st in SUMMARY}
            sub = {st: ad.submetrics_from_row(short_row(rows[SUMMARY[st]], "one" if st == "stage_one" else "two"))
                   for st in ("stage_one", "stage_two")}
        else:
            crashed = not rows
            epdms = {"status": "UNAVAILABLE", "n": len(S2),
                     "reason": ((f"two-stage EPDMS UNDEFINED for {sp['agent']}: the OFFICIAL two-stage runner EXITED 1 and "
                                 "wrote NO CSV (MEASURED, raw/A2/A2.log): 204/204 stage-2 agent calls raised IndexError "
                                 "at dataclasses.py:371 -> AssertionError 'Invalid interval nan' in scene_aggregator.py:62 "
                                 "(caught) -> uncaught TypeError in the summary aggregation. Its 16 stage-1 rows were "
                                 "scored in-process but never written; the stage-1 human numbers come from the one-stage "
                                 "runner (A2b) below. " if crashed else
                                 f"two-stage EPDMS UNDEFINED for {sp['agent']}: stage-2 valid rows "
                                 f"{n_valid['stage_two']}/{len(S2)}, stage-1 {n_valid['stage_one']}/{len(S1)}. ") +
                                "Cause: all synthetic scenes carry 4 frames / num_future_frames=0 (MEASURED, code/probe_data.py) "
                                "and HumanAgent.get_future_trajectory(8) indexes frames[3..11] (dataclasses.py:367-371). "
                                "Never imputed.")}
            sub = {"stage_two": {"status": "UNAVAILABLE", "reason": epdms["reason"], "n": len(S2)}}
        if one_rows.get("average_all_frames"):
            avg = one_rows["average_all_frames"]
            so = ad.read_epdms({"score": float(avg["score"])})
            so["runner"] = ("run_pdm_score_one_stage.py, traffic_agents=reactive (the same IDM policy as the two-stage "
                            "runner's stage 1); mean over the 16 stage-1 tokens; ⚠️ 'prev' tokens have no adjacent "
                            "earlier frame in this runner, so their EC weight is zeroed (/14) — "
                            "run_pdm_score_one_stage.py:177-189. NOT the two-stage stage_one summary.")
            so["n"] = sum(1 for t, r in one_rows.items() if t in S1 and r.get("valid") == "True")
            epdms["stage_one_via_one_stage_runner"] = so
            sub["stage_one_via_one_stage_runner"] = ad.submetrics_from_row(short_row(avg, "one_stage_runner"))
        ii = ad.navsim_inference_inputs(**sp["ii"])
        art = ad.build_artifact(win, tier="T1", variant="EPDMS_v2", split=a.split, arm=f"{sp['agent']}@warmup",
                                epdms=epdms, submetrics=sub, inference_inputs=ii, goal_source=sp["goal"],
                                route_leak={"status": "NOT_APPLICABLE", "n": len(S1) + len(S2),
                                            "reason": "this reference arm consumes no route / driving_command input, so a route-derived leak cannot enter it"})
        art["n_windows"] = len(S1) + len(S2)
        art["n_scenes"] = len(S1) + len(S2)
        art["counts"] = {"stage_one_expected": len(S1), "stage_two_expected": len(S2), "valid": n_valid,
                         "four_families_windows": n_win, "log_groups": n_logs}
        art["four_families"]["_e1_scope"] = (f"OUR instruments on the {n_win} STAGE-1 scenes only (agent vs logged human "
                                             "future, dt 0.5 s, K=8). Stage-2 synthetic scenes have NO human future "
                                             "(0 future frames), so no geometry family exists there. frame check: " + fv)
        art["estimator"] = {"point_estimate": "devkit summary rows (two-stage aggregation) / plain mean (one-stage runner)",
                            "cluster_unit": {"status": "UNAVAILABLE", "n": n_logs,
                                             "reason": "scene token vs nuPlan log unsettled (NAVSIM_PROTOCOL.md §8.5/§9 #15); warmup has 7 log groups"},
                            "interval": {"status": "UNAVAILABLE", "reason": NO_CI_REASON, "n": len(S1) + len(S2)}}
        art["tier"] = "T1"
        art["protocol"].update({
            "navsim_protocol": "EPDMS_v2_warmup_two_stage", "devkit_sha": SHA,
            "devkit_pin_full": f"autonomousvision/navsim@{SHA} (2025-10-27; post-#151 fix, MEASURED: README changelog 2025/09/29 + pdm_score.py:194-219)",
            "harness_modifications": ["PRE-EXISTING navsim/common/dataclasses.py PosixPath unpickler (blob 596cb7d)",
                                      "PRE-EXISTING venv fcntl.py flock shim (sha256 75184a48...)",
                                      "PRE-EXISTING nuplan-devkit setup.py (packaging only)",
                                      "E1 monkeypatch: MetricCacheLoader token separator (code/navsim_win.py) — reader only"],
            "runtime": "C:/Users/Admin/navsim-crun (verified mirror; raw/devkit_copy_verify.json, raw/data_mirror_verify_*.json)",
            "sensor_set": sp["sensor_set"], "setting": sp["setting"],
            "ego_status_enforcement": {"vision_only_claimed": False, "status": "NOT_APPLICABLE",
                                       "reason": "reference arm, NOT vision-only and not claimed to be: " + sp["sensor_set"]},
            "corpus": f"NavSim {a.split} (OpenScene test logs; {n_logs} logs; {len(S1)} stage-1 + {len(S2)} stage-2 tokens) — "
                      "a different benchmark from the TanitAD parity corpus physicalai-train-e438721ae894; no parity key applies",
            "loop": {"stage_one": "OPEN (PI ruling 2026-09-02: fixed plan, ego never re-queried)",
                     "stage_two": "UNRULED under the 2026-09-02 vocabulary (one-shot re-perception of a 3DGS-rendered perturbed start)",
                     "background_traffic": "IDM-REACTIVE vehicles in BOTH stages (run_pdm_score.py:77-79,132-134 -> navsim_IDM_traffic_agents.yaml); pedestrians/static log-replay; ego non-reactive"},
            "tier_note": "T1-family: the arm's own plan is executed by LQR + kinematic bicycle; nothing recorded is fed back",
        })
        if controls is not None:
            art["controls"] = controls
        # criteria that cannot exist for these arms are REFUSED with a reason (a visible work item), never omitted
        art["refused"] = {
            "nav_compliance": ("the reference arm consumes NO route / driving_command input (constant_velocity_agent.py "
                               "reads only ego_velocity; human_agent replays the log), so 'behaviour follows the route "
                               "command' is undefined for it; NavSim itself never scores the route (NAVSIM_PROTOCOL.md:812)."),
            "nav_compliance_controls": "no route input exists to shuffle or withhold (see nav_compliance).",
            "inference_seed_replicate": ("both agents are deterministic and the scoring path contains no RNG (grep 'random' "
                                         "over evaluate/, planning/simulation/, traffic_agents_policies/, "
                                         "planning/metric_caching/: 0 files); a process replicate with a different hash "
                                         "seed (R1) reproduced every per-token value bit-for-bit (controls.C7_determinism, "
                                         "raw/C7_diagnosis.json)."),
        }
        art["_e1_adapter_gaps"] = ["G1 long/stage-suffixed devkit columns vs short keys", "G2 estimator.cluster_unit",
                                   "G3 protocol.ego_status_enforcement", "G4 protocol.sensor_set/setting",
                                   "G5 protocol.navsim_protocol/devkit_sha", "G6 protocol.corpus/controls",
                                   "criteria_check.py never evaluates benchmarks.navsim"]
        p = outdir / f"{arm}_{sp['agent']}_{a.split}.json"
        p.write_text(json.dumps(art, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                                ensure_ascii=False), encoding="utf-8")
        cj = outdir / f"{arm}_criteria_check.json"
        r = subprocess.run([sys.executable, str(REPO / "tools/criteria_check.py"), str(p), "--json", str(cj)],
                           capture_output=True, text=True, encoding="utf-8")
        (outdir / f"{arm}_criteria_check.txt").write_text(r.stdout + "\n[stderr]\n" + r.stderr + f"\n[rc={r.returncode}]\n", encoding="utf-8")
        cres = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else None
        gates = navsim_gates(json.loads(p.read_text(encoding="utf-8")), reg)
        muts = gate_mutations(json.loads(p.read_text(encoding="utf-8")), reg)
        summary[arm] = {"artifact": str(p), "criteria_check_rc": r.returncode, "criteria_check_json_written": cj.exists(),
                        "navsim_gates": gates, "gate_mutations_all_red": muts, "four_families_frame": fv,
                        "two_stage_defined": two_stage_defined}
        if cres is not None:
            res = cres["results"] if "results" in cres else cres
            summary[arm]["criteria_check_raw_keys"] = list(res)[:5] if isinstance(res, dict) else type(res).__name__
    (outdir / "artifact_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
