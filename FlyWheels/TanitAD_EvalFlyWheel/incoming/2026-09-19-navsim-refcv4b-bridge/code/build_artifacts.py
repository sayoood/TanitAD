#!/usr/bin/env python3
"""TanitEval artifacts per arm via ``taniteval/adapters/navsim.py::build_artifact`` (UNEDITED,
imported by path) + the protocol keys the four blocking NavSim gates read, then
``tools/criteria_check.py`` over each. TANITAD VENV.

The wrapper ADDS keys and never overrides a value the adapter set, except
``benchmark.navsim.score``, which is supplied explicitly because the camera arms' honest
statistic is S2-EPDMS-u (stage 2 only) and must never be mistaken for a two-stage EPDMS.
Writes raw/artifact_<arm>.json and raw/criteria_check_<arm>.{json,txt}.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RAW = os.path.join(PKG, "raw")
sys.path.insert(0, HERE)
import tanitad_navsim_bridge as B  # noqa: E402

DEVKIT_SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
DEVKIT_PATCH = ("ONE local modification in the devkit working tree, predating this stream: "
                "navsim/common/dataclasses.py (+24/-1) — Scene.load_from_disk unpickles "
                "PosixPath as PurePosixPath (Windows; TanitAD_EvalFlyWheel 2026-08-28). "
                "`git -c safe.directory=* status` in C:/Users/Admin/navsim/devkit.")
ARM_INPUTS = {   # (cameras, ego_velocity, ego_acceleration, driving_command)
    "A1_ego_cmd": (True, True, True, True),
    "A2_vision_pure": (True, False, False, False),
    "A3_ego_nocmd": (True, True, True, False),
    "A1NT_ego_cmd_nearest": (True, True, True, True),
    "A2NT_vision_pure_nearest": (True, False, False, False),
    "A4_blind_ego_cmd": (False, True, True, True),
}


NO_GT = ("no warmup scene carries BOTH the model's camera input and a logged future: the 204 "
         "stage-2 synthetic scenes have frames but num_future_frames = 0 (no human drove them), "
         "the 16 stage-1 scenes have a future but 0/192 camera jpgs on this box. Unblock = the "
         "stage-1 original frames (then n = 16).")


def _set(d: dict, path: str, val) -> None:
    """Create ``a.b.c`` under ``d`` (never overwriting an existing leaf)."""
    ks = path.split(".")
    for k in ks[:-1]:
        nxt = d.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            d[k] = nxt
        d = nxt
    d.setdefault(ks[-1], val)


def nav_compliance(arm_true: str, arm_zero: str, doc: dict) -> dict:
    """Does the PLAN follow the NavSim command? NavSim's own rule: the route point 20 m
    ahead is >= 2 m left -> 'left'. Readout: the plan's lateral offset where its arc length
    reaches 20 m (its last pose if it never does) — left-commanded scenes comply if y >= 2 m,
    straight-commanded if |y| < 2 m. Paired against the SAME model with nav withheld."""
    def plan_y20(arm):
        z = np.load(os.path.join(RAW, f"seam_{arm}.npz"), allow_pickle=False)
        out = {}
        for t, s, p in zip(z["token"], z["source"], z["poses"]):
            if str(s) != "refcv4b":
                continue
            xy = np.vstack([[0.0, 0.0], np.asarray(p, np.float64)[:, :2]])
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
            i = int(np.searchsorted(arc, 20.0)) if arc[-1] >= 20.0 else len(arc) - 1
            out[str(t)] = (float(xy[min(i, len(xy) - 1), 1]), bool(arc[-1] >= 20.0))
        return out
    yt, yz = plan_y20(arm_true), plan_y20(arm_zero)
    cmd = {t: int(np.argmax(r["ego_statuses"][-1]["driving_command"]))
           for t, r in doc["tokens"].items() if r["stage"] == 2}
    res = {}
    for k, name in ((0, "left"), (1, "straight"), (2, "right")):
        toks = [t for t in cmd if cmd[t] == k and t in yt and t in yz]
        if not toks:
            res[name] = {"status": "UNAVAILABLE", "reason": f"no stage-2 scene carries the "
                         f"'{name}' command at t0", "n": 0}
            continue
        ok = (lambda y: y >= 2.0) if k == 0 else (lambda y: y <= -2.0) if k == 2 else (
            lambda y: abs(y) < 2.0)
        ct = float(np.mean([ok(yt[t][0]) for t in toks]))
        cz = float(np.mean([ok(yz[t][0]) for t in toks]))
        res[name] = {"n": len(toks), "compliance_nav_true": ct, "compliance_nav_zero": cz,
                     "paired_true_minus_zero": ct - cz,
                     "frac_plans_reaching_20m": float(np.mean([yt[t][1] for t in toks]))}
    return {"rule": "NavSim's own derivation: 20 m ahead, +-2 m (NAVSIM_PROTOCOL.md 5.1)",
            "arm_nav_true": arm_true, "arm_nav_zero": arm_zero, "by_command": res,
            "scope": "stage-2 scenes; readout of the EMITTED plan (no GT future needed)"}


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def navsim_gate_selfcheck(art: dict) -> dict:
    """E2's OWN reading of the four BLOCKING NavSim gates, keys taken from the registry.
    ⛔ ``tools/criteria_check.py`` (registry v2.9.0) does NOT evaluate ``benchmarks.navsim``
    gates at all (0 occurrences of 'navsim' in its report) — this is a stop-gap, not the
    instrument; stream E3 owns the checker."""
    reg = json.load(open(os.path.join(B.REPO, "products", "P7-TanitEval",
                                      "CRITERIA_REGISTRY.json"), encoding="utf-8"))
    nv = reg["benchmarks"]["navsim"]
    out = {"_registry_version": reg.get("version"),
           "_note": "self-check — criteria_check.py does not evaluate these gates"}
    g = nv["GATE_ego_status_enforcement"]
    e = _get(art, "protocol.ego_status_enforcement") or {}
    out[g["id"]] = ("PASS" if e.get("mechanism") and e.get("evidence") and
                    all((e["evidence"].get("verdict") or {}).values()) else "FAIL")
    g = nv["GATE_modality_label"]
    out[g["id"]] = ("PASS" if _get(art, "protocol.sensor_set") and _get(art, "protocol.setting")
                    else "FAIL")
    g = nv["GATE_estimator_cluster_unit"]
    cu, iv = _get(art, "estimator.cluster_unit"), _get(art, "estimator.interval")
    out[g["id"]] = ("PASS" if isinstance(cu, dict) and isinstance(iv, dict) and
                    iv.get("status") == "UNAVAILABLE" and iv.get("reason") and "n" in iv
                    else "FAIL")
    g = nv["GATE_no_cross_protocol_comparison"]
    out[g["id"]] = ("PASS" if _get(art, "protocol.navsim_protocol") in g["closed_set"] and
                    _get(art, "protocol.devkit_sha") else "FAIL")
    return out


def _adapter():
    p = os.path.join(B.REPO, "taniteval", "adapters", "navsim.py")
    spec = importlib.util.spec_from_file_location("navsim_adapter_e2art", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ad = _adapter()
    summ = json.load(open(os.path.join(RAW, "scores_summary.json"), encoding="utf-8"))
    mut = json.load(open(os.path.join(RAW, "K5_K6_ego_mutation.json"), encoding="utf-8"))
    doc = json.load(open(os.path.join(RAW, "navsim_agent_inputs.json"), encoding="utf-8"))
    rep = {}
    for arm, (cam, ev, ea, dc) in ARM_INPUTS.items():
        if arm not in summ["arms"]:
            continue
        z = np.load(os.path.join(RAW, f"seam_{arm}.npz"), allow_pickle=False)
        src = np.asarray([str(s) for s in z["source"]])
        m2 = src == "refcv4b"
        toks = [str(t) for t in z["token"][m2]]
        poses = np.asarray(z["poses"][m2], dtype=np.float64)
        win = ad.scenes_to_win(poses, None, frame="ego", origin_included=False, dt_s=0.5,
                               scene_tokens=toks)
        ii = ad.navsim_inference_inputs(cameras=cam, lidar=False, ego_velocity=ev,
                                        ego_acceleration=ea, ego_pose_history=False,
                                        driving_command=dc,
                                        claimed_abstention_from_ego=not (ev or ea))
        a = summ["arms"][arm]
        u = a["S2_EPDMS_u"]
        score = {"value": u.get("value"), "column": "score", "variant": "EPDMS_v2",
                 "statistic": "S2-EPDMS-u — STAGE-2 ONLY (204 synthetic scenes), uniform "
                              "within-group weights (the devkit's own zero-mass fallback); "
                              "SPEC §2",
                 "n": u.get("n_scenes"),
                 "two_stage_epdms": {
                     "status": "UNAVAILABLE",
                     "reason": ("the 16 stage-1 scenes' camera frames are not on this box "
                                "(0/192 original jpgs), so no camera arm has stage-1 "
                                "trajectories — they set both the stage-1 factor and the "
                                "stage-2 kernel weights. The official script's summary rows "
                                "for this arm are HYBRID (stage-1 = devkit CV stand-in)."
                                if cam else
                                "frames-blind diagnostic: stage-1 rows are this arm's own; "
                                "see scores_summary.json official_summary_rows"),
                     "n": 16}}
        sub = ad.submetrics_from_row(a["S2_submetric_means"])
        sub["_statistic"] = "mean over the stage-2 scenes of each official per-scene term"
        art = ad.build_artifact(win, tier="T1", variant="EPDMS_v2", split="warmup_two_stage",
                                arm=f"refcv4b::{arm}", epdms=score, submetrics=sub,
                                inference_inputs=ii,
                                goal_source=("NavSim driving_command[t0] mapped 0->left, "
                                             "1->follow, 2->right, 3->follow onto "
                                             "refb.NAV_COMMANDS" if dc else
                                             "none — nav withheld (nav_cmd=None)"),
                                route_leak=ad.route_leak_check("UNVERIFIED"))
        declared = B.ARMS[arm]["declared"]
        art["protocol"].update({
            "navsim_protocol": "EPDMS_v2_warmup_two_stage",
            "navsim_protocol_scope": "STAGE-2 ROWS ONLY for camera arms (SPEC §0)",
            "devkit_sha": DEVKIT_SHA, "devkit_local_patch": DEVKIT_PATCH,
            "sensor_set": ("NONE (frames-blind: one constant grey, no pixels read)" if not cam
                           else "3-camera stitch cam_l0+cam_f0+cam_r0 -> 256x640 cylindrical "
                                "(PHYSICALAI_WIDE120_256x640, 120 deg, 89.29 % observed), "
                                "DataFlyWheel bank, sha-verified; t0 frame only (ST) or "
                                "nearest-time 2 Hz history (NT)"),
            "setting": ("perception-free, ZERO-SHOT from PhysicalAI-AV (no NavSim/nuPlan "
                        "training); one-shot anchor planner; CPU inference"),
            "ego_status_enforcement": {
                "mechanism": ("declared-input seam: code/tanitad_navsim_bridge.py::declare "
                              "COPIES only the arm's declared t0 fields out of the devkit's "
                              "AgentInput export; the model call receives only that dict + "
                              "frames. The NavSim-side agent is a lookup table keyed by the "
                              "scorer token and re-checks the AgentInput fingerprint."),
                "declared_fields": list(declared),
                "evidence": {"file": "raw/K5_K6_ego_mutation.json",
                             "verdict": mut.get("verdict"),
                             "cases": mut.get("cases")},
                "manifest": f"raw/seam_{arm}.manifest.json"},
        })
        art["estimator"]["cluster_unit"] = {
            "status": "UNAVAILABLE",
            "reason": ("unsettled (stream E3); warmup has 7 log groups < the RG-14 floor of "
                       "8, so no interval is admissible here in any case"),
            "n": 7}
        # ---- per-criterion refusals (the checker reads criterion keys, not families) ----
        ref = {"status": "UNAVAILABLE", "reason": NO_GT, "n": 0}
        for key in ("four_families.longitudinal.speed_mae_mps",
                    "four_families.longitudinal.along_mae_m",
                    "four_families.lateral.cross_mae_m", "four_families.lateral.heading_mae_deg",
                    "four_families.lateral.curvature_mae_1pm",
                    "four_families.lateral.yaw_rate_mae_degps",
                    "four_families.tactical.lateral_decision",
                    "four_families.tactical.lateral_decision.confusion_gt_rows_pred_cols",
                    "four_families.tactical.goal_setting"):
            _set(art, key, dict(ref))
        _set(art, "four_families.longitudinal.ego_progress",
             dict(ref, benchmark_analogue=("NavSim's own EP (normalised by PDM-Closed, "
                                           "pdm_scorer.py:231-236) is in "
                                           "benchmark.navsim.submetrics.EP — a different "
                                           "quantity, not substituted")))
        _set(art, "longitudinal.distance_keeping",
             dict(ref, reason=NO_GT + " NavSim's TTC is a binary within-bound flag, not a "
                  "headway (benchmark.navsim.submetrics.TTC)."))
        # ---- strategic nav-compliance: needs NO future, only the plan + the command -------
        base = "four_families.strategic.nav_compliance.readouts.plan"
        if arm in ("A1_ego_cmd",):
            nc = nav_compliance("A1_ego_cmd", "A3_ego_nocmd", doc)
            by = [v for v in nc["by_command"].values() if "n" in v and v["n"]]
            n = sum(v["n"] for v in by)
            ct = sum(v["compliance_nav_true"] * v["n"] for v in by) / n
            cz = sum(v["compliance_nav_zero"] * v["n"] for v in by) / n
            _set(art, base + ".conditionings.nav_true.compliance_with_TRUE_command", ct)
            _set(art, base + ".paired_true_minus_zero", ct - cz)
            _set(art, base + ".detail", nc)
        else:
            why = ("nav withheld in this arm (nav_cmd=None)" if not dc else
                   "no nav-withheld partner arm with the SAME frames construction was run")
            _set(art, base + ".conditionings.nav_true.compliance_with_TRUE_command",
                 {"status": "UNAVAILABLE", "reason": why, "n": 0})
            _set(art, base + ".paired_true_minus_zero",
                 {"status": "UNAVAILABLE", "reason": why, "n": 0})
        _set(art, "strategic.echo_test",
             {"status": "UNAVAILABLE",
              "reason": ("no route/goal head is SCORED in this NavSim eval (refcv4b's "
                         "route_logits are not read); the only route-conditioned readout is "
                         "the plan's nav-compliance, reported with its nav-zero pair"),
              "n": 0})
        _set(art, base + ".paired_true_minus_shuffled",
             {"status": "UNAVAILABLE", "reason": ("nav-shuffle arm not run in this stream "
                                                  "(budget); nav-zero is the paired control "
                                                  "reported"), "n": 0})
        # ---- floors / trivial controls, corpus identity, inference determinism ------------
        fl = {c: summ["arms"][c]["S2_EPDMS_u"].get("value") for c in
              ("CV_official", "STOP_zero", "ECHO_ha0_ext", "A2_vision_pure")
              if c in summ["arms"]}
        art["controls"] = {"statistic": "S2-EPDMS-u (stage 2, uniform weights)",
                           "this_arm": u.get("value"), "floors": fl,
                           "this_minus_floor": {c: (None if v is None or u.get("value") is None
                                                    else u["value"] - v) for c, v in fl.items()},
                           "note": ("STOP_zero and ECHO_ha0_ext are POST-HOC diagnostics; "
                                    "CV is the pre-registered bar (BAR-E2-1)")}
        art["corpus"] = {"name": "NavSim warmup_two_stage — NON-PARITY (not "
                                 "physicalai-train-e438721ae894)",
                         "frame_bank": "DataFlyWheel corpus_id 89f26a994509c379 (204 stage-2 "
                                       "scenes), sha-verified per scene",
                         "tokens": "16 stage-1 + 204 stage-2 (devkit SceneLoader)"}
        art["inference_seeds"] = {"deterministic": True, "seed_floor_m": 0.0,
                                  "evidence": ("K0: two forwards on identical inputs "
                                               "byte-identical on 20/20 scenes "
                                               "(raw/K5_K6_ego_mutation.json)")}
        art["tier_loop"] = {"tier": "T1-family", "stage1_loop": "OPEN (PI 2026-09-02)",
                            "stage2_loop": "UNRULED (pre-rendered perturbed start)",
                            "background_agents": "IDM-REACTIVE in both stages "
                                                 "(run_pdm_score.py:77-79,132-134)",
                            "closed_loop": False}
        art["navsim_gate_selfcheck"] = navsim_gate_selfcheck(art)
        path = os.path.join(RAW, f"artifact_{arm}.json")
        B.json_dump(art, path)
        out_json = os.path.join(RAW, f"criteria_check_{arm}.json")
        proc = subprocess.run([sys.executable, os.path.join(B.REPO, "tools", "criteria_check.py"),
                               path, "--json", out_json], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=B.REPO)
        with open(os.path.join(RAW, f"criteria_check_{arm}.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        rep[arm] = {"artifact": path, "criteria_rc": proc.returncode}
        print(arm, "criteria_check rc", proc.returncode)
    # ---- four families on the ONLY scenes that carry a GT future: stage 1 (n = 16) ------
    # Camera arms have no stage-1 path (no frames); the frames-blind arm A4 and the CV
    # reference do. Both are built on the SAME 16 scenes with the SAME human futures, so the
    # pair is readable; neither is refcv4b's camera arm.
    s1 = sorted(t for t, r in doc["tokens"].items() if r["stage"] == 1)
    gt = np.asarray([doc["tokens"][t]["human_future_poses"] for t in s1], dtype=np.float64)
    fam = {}
    preds = {"CV_official": np.asarray([doc["tokens"][t]["cv_poses"] for t in s1], np.float64)}
    if os.path.exists(os.path.join(RAW, "seam_A4_blind_ego_cmd.npz")):
        z = np.load(os.path.join(RAW, "seam_A4_blind_ego_cmd.npz"), allow_pickle=False)
        idx = {str(t): i for i, t in enumerate(z["token"])}
        preds["A4_blind_ego_cmd"] = np.asarray([z["poses"][idx[t]] for t in s1], np.float64)
    for name, pp in preds.items():
        try:
            w = ad.scenes_to_win(pp, gt, frame="ego", origin_included=False, dt_s=0.5,
                                 scene_tokens=s1,
                                 ego_speed_mps=[np.hypot(*doc["tokens"][t]["ego_statuses"][-1]
                                                         ["ego_velocity"]) for t in s1])
            fam[name] = ad.four_families_block(w, tier="T1")
        except Exception as ex:                  # noqa: BLE001 — recorded, never silent
            fam[name] = {"status": "RAISED", "error": f"{type(ex).__name__}: {ex}"}
    fam["_scope"] = ("stage-1 scenes only (n = 16), the only warmup scenes with a logged "
                     "future; A4 is the frames-blind DIAGNOSTIC arm and CV the devkit "
                     "reference — neither is a refcv4b camera arm (none can run stage 1)")
    B.json_dump(fam, os.path.join(RAW, "four_families_stage1.json"))
    rep["four_families_stage1"] = os.path.join(RAW, "four_families_stage1.json")
    B.json_dump(rep, os.path.join(RAW, "artifacts_index.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
