#!/usr/bin/env python3
"""TanitEval artifacts + the FOUR METRIC FAMILIES for the navtest v1.1 arms (TANITAD venv).

    python code/artifacts_navtest.py --arms CV,HUMAN,STOP --split-run navtest

navtest is the split where our own instruments CAN run: every token carries the logged human
future (the devkit's own ``Scene.get_future_trajectory``, exported by
``export_navtest_inputs.py``), so LONGITUDINAL / LATERAL / TACTICAL are computable against a GT
— unlike warmup stage 2 (E2 §6), where no future exists. The geometry is OUR instrument on
NavSim trajectories; ``benchmark.navsim`` is NavSim's own score. ⛔ They are never merged.

Uses ``taniteval/adapters/navsim.py`` UNMODIFIED (``scenes_to_win`` → ``build_artifact``,
variant ``PDMS_v1``) and W2's registered log-cluster interval, and writes
``raw/artifact_<arm>_<split-run>.json`` (+ ``raw/criteria_<arm>_<split-run>.txt`` when
``--criteria``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RAW = PKG / "raw"
REPO = PKG.parents[3]
EXPORT = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
CLUSTER_MAP = (REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/"
               "2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json")
sys.path.insert(0, str(REPO / "taniteval"))
from adapters import navsim as ad                                      # noqa: E402
from adapters import navsim_ci as CI                                   # noqa: E402

TERM_COL = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
            "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "C": "comfort"}
#: what each arm reads AT INFERENCE (the declared-input manifest of this package)
INPUTS = {
    "CV": dict(cameras=False, ego_velocity=True, ego_acceleration=False, ego_pose_history=False,
               driving_command=False),
    "STOP": dict(cameras=False, ego_velocity=False, ego_acceleration=False, ego_pose_history=False,
                 driving_command=False, claimed_abstention_from_ego=True),
    "HUMAN": dict(cameras=False, ego_velocity=False, ego_acceleration=False, ego_pose_history=False,
                  driving_command=False),
    # refcv4b through E2's bridge, arm A1_ego_cmd: E2's ARMS["A1_ego_cmd"]["declared"] is
    # ("ego_velocity[t0]", "ego_acceleration[t0]", "driving_command[t0]") + frames "ST".
    "A1": dict(cameras=True, ego_velocity=True, ego_acceleration=True, ego_pose_history=False,
               driving_command=True),
}
WHAT = {"CV": "devkit ConstantVelocityAgent", "STOP": "W3 StopAgent (all-zero plan)",
        "HUMAN": "devkit HumanAgent — PRIVILEGED: it returns the logged future itself",
        "A1": ("refcv4b-b1-v72-40k (MODEL_REGISTRY.md §4.6, ckpt md5 99b573e8…) through E2's "
               "bridge, arm A1_ego_cmd, seam scored by W3.SeamAgentV1")}
#: what each arm's GOAL / route input is computed FROM (binding 2026-08-03: a goal input is
#: admissible, but it must be stated, and it must not carry the situation classifier's output).
GOAL = {"CV": "none — this arm reads no route/goal signal",
        "STOP": "none — this arm reads no route/goal signal",
        "HUMAN": "PRIVILEGED: the logged human future itself (reference arm, never deployable)",
        "A1": ("NavSim driving_command[t0] — the benchmark's OWN categorical route command, part "
               "of every NAVSIM agent's input; mapped to the model's nav input by E2's bridge "
               "(ARMS['A1_ego_cmd']['nav'] = 'cmd'). It is computed from the log's route, NOT "
               "from any situation classifier — refcv4b has no classifier on this path")}
#: what each arm READS of the sensors. ⛔ A default here once said NONE for every arm.
SENSORS = {"CV": "NONE (no camera/LiDAR is read by this arm)",
           "STOP": "NONE (no camera/LiDAR is read by this arm)",
           "HUMAN": "NONE (no camera/LiDAR is read by this arm)",
           "A1": ("CAM_F0 + CAM_L0 + CAM_R0, stitched to 256x640 by E2's builder (W3 navtest frame "
                  "bank, KB1 bit-exact vs E2); no LiDAR, no back/side-rear cameras")}
#: gate ``navsim.ego_enforcement``: a MECHANISM + structured evidence per arm (none of these
#: arms makes a vision-only claim; each one's input set is fixed by the agent's own source).
ENFORCE = {
    "CV": {"mechanism": ("the devkit's OWN ConstantVelocityAgent: compute_trajectory reads "
                         "exactly agent_input.ego_statuses[-1].ego_velocity "
                         "(navsim/agents/constant_velocity_agent.py:32-45 @3e8291b) and nothing "
                         "else; requires_scene = False, so the runner never builds a Scene for it"),
           "declared_fields": ["ego_velocity[t0]"],
           "vision_only_claimed": False,
           "evidence": {"file": "raw/KX_export_vs_scorer_smoke20.json",
                        "verdict": ("the poses the scorer received equal the poses recomputed "
                                    "from the exported ego_velocity: max |Δ| 0.0 on 20/20 tokens"),
                        "source": "navsim/agents/constant_velocity_agent.py:32-45"}},
    "STOP": {"mechanism": ("w3_agents_v1.StopAgent.compute_trajectory IGNORES its argument and "
                           "returns np.zeros((8, 3)); requires_scene = False. It cannot read an "
                           "ego status because it reads nothing at all"),
             "declared_fields": [],
             "vision_only_claimed": False,
             "evidence": {"file": "tests/test_w3_agents.py",
                          "verdict": ("test_stop_agent_is_exactly_zero: |poses|max == 0.0, shape "
                                      "(8,3) float32; MEASURED in the run: every recorded "
                                      "agent_pose is exactly 0"),
                          "source": "code/w3_agents_v1.py::StopAgent"}},
    "HUMAN": {"mechanism": ("the devkit's OWN privileged HumanAgent: it returns "
                            "scene.get_future_trajectory(8) — the LOGGED FUTURE. This arm is a "
                            "reference ceiling and is never deployable"),
              "declared_fields": ["logged future trajectory (privileged)"],
              "vision_only_claimed": False,
              "reason": ("not applicable: a privileged reference arm makes no vision-only claim; "
                         "its input IS the label"),
              "evidence": {"file": "raw/KX_export_vs_scorer_smoke20.json",
                           "verdict": ("the poses the scorer received equal the devkit's own "
                                       "Scene.get_future_trajectory export: max |Δ| 0.0 on 20/20"),
                           "source": "navsim/agents/human_agent.py:35-41"}},
    "A1": {"mechanism": ("E2's tanitad_navsim_bridge.declare(ego_statuses, 'A1_ego_cmd') is the ONE "
                         "enforcement point: it copies exactly ARMS['A1_ego_cmd']['declared'] = "
                         "ego_velocity[t0], ego_acceleration[t0], driving_command[t0] into the "
                         "model's inputs and nothing else; the frames are the stitched front "
                         "cameras. Measured t0 ego state is ADMISSIBLE (PI ruling 2026-09-02: "
                         "velocity at cycle time is a legal initial state)"),
           "declared_fields": ["ego_velocity[t0]", "ego_acceleration[t0]", "driving_command[t0]"],
           "vision_only_claimed": False,
           "reason": ("not a vision-only arm by design: A1 declares measured t0 ego state + the "
                      "benchmark's route command; the vision-only arm is A2_vision_pure"),
           "evidence": {"file": "raw/bridge_navtest/seam_A1_ego_cmd.manifest.json",
                        "verdict": ("the manifest records declared_inputs and, per token, the "
                                    "declared_values the model received"),
                        "source": ("FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                   "2026-09-19-navsim-refcv4b-bridge/code/"
                                   "tanitad_navsim_bridge.py::declare")}},
}


def arm_meta(arm: str) -> dict:
    """Every per-arm fact, or a REFUSAL. ⛔ No arm inherits another arm's description."""
    miss = [n for n, t in (("INPUTS", INPUTS), ("WHAT", WHAT), ("GOAL", GOAL),
                           ("SENSORS", SENSORS), ("ENFORCE", ENFORCE)) if arm not in t]
    if miss:
        raise SystemExit(f"⛔ arm {arm!r} has no explicit {miss} entry — refusing to describe it "
                         f"with another arm's defaults")
    return {"inputs": INPUTS[arm], "what": WHAT[arm], "goal": GOAL[arm],
            "sensors": SENSORS[arm], "enforce": ENFORCE[arm]}


def refuse_nones(art: dict, n: int, arm: str) -> list:
    """A metric the harness could not define must carry a REASON + n, never a bare ``None``
    (binding four-families rule 5). The reason quotes the harness's OWN counters."""
    touched = []
    ff = art.get("four_families", {})
    for fam in ("longitudinal", "lateral", "tactical", "strategic"):
        blk = ff.get(fam)
        if not isinstance(blk, dict):
            continue
        ctx = {k: blk.get(k) for k in ("n_steps_heading", "n_steps_curvature", "n_steps_yaw_rate",
                                       "excluded_below_min_ds", "min_ds_m", "n_windows")
               if blk.get(k) is not None}
        for k, v in list(blk.items()):
            if v is None:
                blk[k] = {"status": "UNAVAILABLE", "n": int(n),
                          "reason": (f"{arm}: the harness returned no value for {k!r} — its own "
                                     f"counters say why: {ctx}. A displacement below min_ds "
                                     f"leaves heading/curvature/yaw-rate undefined (atan2 of a "
                                     f"zero tangent); this is a REFUSAL with its n, never a 0."),
                          "_added_by": "W3 code/artifacts_navtest.py::refuse_nones"}
                touched.append(f"{fam}.{k}")
    return touched


def rows_of(label: str) -> dict:
    p = RAW / label / f"{label}.csv"
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] != "average" and r["valid"] in ("True", "true", "1"):
            out[r["token"]] = r
    return out


def plans(arm: str, toks: list, doc: dict, seam: str | None) -> np.ndarray:
    if arm == "CV":
        return np.asarray([doc[t]["cv_poses"] for t in toks], np.float64)
    if arm == "HUMAN":
        return np.asarray([doc[t]["human_future_poses"] for t in toks], np.float64)
    if arm == "STOP":
        return np.zeros((len(toks), 8, 3), np.float64)
    z = np.load(seam, allow_pickle=False)
    idx = {str(t): i for i, t in enumerate(z["token"])}
    return np.asarray([z["poses"][idx[t]] for t in toks], np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="CV,HUMAN,STOP")
    ap.add_argument("--split-run", default="navtest")
    ap.add_argument("--seam", default="", help="arm=npz,arm=npz for model arms")
    ap.add_argument("--export", default=str(EXPORT))
    ap.add_argument("--criteria", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    doc = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    clusters = json.load(open(CLUSTER_MAP, encoding="utf-8"))["token_to_log_name"]
    seams = dict(x.split("=", 1) for x in a.seam.split(",") if "=" in x)
    index = {}
    for arm in [x for x in a.arms.split(",") if x]:
        label = f"{arm}_{a.split_run}"
        if not (RAW / label / f"{label}.csv").exists():
            index[arm] = {"status": "ABSENT", "why": f"{label}.csv not found"}
            continue
        meta = arm_meta(arm)
        tk = rows_of(label)
        toks = sorted(tk)
        head = float(np.mean([float(tk[t]["score"]) for t in toks]))
        means = {k: float(np.mean([float(tk[t][c]) for t in toks])) for k, c in TERM_COL.items()}
        win = ad.scenes_to_win(
            plans(arm, toks, doc, seams.get(arm)),
            np.asarray([doc[t]["human_future_poses"] for t in toks], np.float64),
            frame="ego", origin_included=False, dt_s=0.5, scene_tokens=toks,
            ego_speed_mps=[float(np.hypot(*doc[t]["ego_statuses"][-1]["ego_velocity"]))
                           for t in toks],
            log_names=[clusters[t] for t in toks])
        interval = CI.interval_from_run(protocol="PDMS_v1_navtest", clusters_by_unit=clusters,
                                        scores={t: float(tk[t]["score"]) for t in toks},
                                        official_value=head)
        score = {"value": head, "column": "score", "variant": "PDMS_v1", "n": len(toks),
                 "x100": round(100 * head, 4),
                 "statistic": ("plain mean of the official per-token `score` over the valid rows "
                               "= the devkit's own `average` row (run_pdm_score.py:144-147). "
                               "PDMS_v1 = NC·DAC·(5·EP+5·TTC+2·C)/12; DDC has weight 0.")}
        sub = ad.submetrics_from_row({**means, "score": head}, variant="PDMS_v1")
        art = ad.build_artifact(
            win, tier="T1", variant="PDMS_v1", split="navtest", arm=f"navsim_v1::{arm}",
            epdms=score, submetrics=sub, n_boot=a.n_boot, interval=interval,
            inference_inputs=ad.navsim_inference_inputs(**meta["inputs"]),
            goal_source=meta["goal"],
            navsim_protocol="PDMS_v1_navtest",
            ego_status_enforcement=meta["enforce"],
            devkit_sha="3e8291bfa89ff247231e0227778840cd0a036896",
            sensor_set=meta["sensors"],
            setting=f"{meta['what']}; CPU; non-reactive logged background; single query")
        refused_keys = refuse_nones(art, len(toks), arm)
        p = RAW / f"artifact_{arm}_{a.split_run}.json"
        json.dump(art, open(p, "w", encoding="utf-8"), indent=1, default=str)
        rec = {"status": "OK", "artifact": str(p.relative_to(PKG)).replace(os.sep, "/"),
               "n": len(toks), "PDMS_x100": round(100 * head, 4),
               "none_metrics_refused_with_reason": refused_keys,
               "families": {k: (art["four_families"].get(k) or {}).get("status", "OK")
                            for k in ("longitudinal", "lateral", "tactical", "strategic")},
               "interval": {k: interval.get(k) for k in ("status", "estimator", "cluster_unit",
                                                         "lo", "hi", "n_clusters")}}
        if a.criteria:
            txt = RAW / f"criteria_{arm}_{a.split_run}.txt"
            js = RAW / f"criteria_{arm}_{a.split_run}.json"
            r = subprocess.run([sys.executable, str(REPO / "tools" / "criteria_check.py"), str(p),
                                "--json", str(js)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", cwd=str(REPO))
            txt.write_text(f"rc={r.returncode}\n{r.stdout}\n[stderr]\n{r.stderr}", encoding="utf-8")
            rec["criteria_rc"] = r.returncode
            if js.exists():
                j = json.loads(js.read_text(encoding="utf-8"))
                arts = list((j.get("artifacts") or {}).values())
                rec["criteria"] = {"n_violations": sum(int(x.get("n_violations", 0)) for x in arts),
                                   "n_work_items": sum(int(x.get("n_work_items", 0)) for x in arts),
                                   "registry_version": j.get("registry_version")}
        index[arm] = rec
        print(json.dumps({arm: rec}, indent=1))
    json.dump(index, open(RAW / f"artifacts_index_{a.split_run}.json", "w", encoding="utf-8"),
              indent=1)
    return 0 if all(v.get("status") == "OK" for v in index.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
