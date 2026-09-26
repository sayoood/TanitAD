"""TanitEval artifact per arm + the four BLOCKING NavSim gates — PROMOTED from E1
``code/build_artifacts.py`` (``short_row``, ``build_win``, ``navsim_gates``, ``gate_mutations``:
VERBATIM, AST-pinned) and E2 ``code/build_artifacts.py`` (``_set``, ``_get``: VERBATIM,
AST-pinned; the ``NO_GT`` refusal text). E1's gate evaluator is the one used (E2's
``navsim_gate_selfcheck`` reads the same four registry gates and is not duplicated).

Uses ``taniteval/adapters/navsim.py`` UNMODIFIED (owner W2), imported by file path under a
private name (``navsim`` is also the DEVKIT's package name). Gaps the adapter does not fill are
added HERE and listed in ``_w1_adapter_gaps`` (E1's G1–G6), never by editing the adapter.
⛔ ``tools/criteria_check.py`` (registry v2.9.0) does not evaluate ``benchmarks.navsim`` (E1/E2,
MEASURED) — so the gates are evaluated here too (E1's evaluator + its RED mutations) until W2 lands
the checker's own evaluation.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

from ..contract import REPO


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ad = _load_by_path("taniteval_bench_navsim_adapter", REPO / "taniteval" / "adapters" / "navsim.py")
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
import criteria_check as cc  # noqa: E402  tools/criteria_check.py (import only; W2 owns it)

SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
LONG = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance",
        "TLC": "traffic_light_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
        "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
SUMMARY = {"stage_one": "extended_pdm_score_stage_one", "stage_two": "extended_pdm_score_stage_two",
           "combined": "extended_pdm_score_combined"}
NO_GT = ("no warmup scene carries BOTH the model's camera input and a logged future: the 204 "
         "stage-2 synthetic scenes have frames but num_future_frames = 0 (no human drove them), "
         "the 16 stage-1 scenes have a future but 0/192 camera jpgs on this box. Unblock = the "
         "stage-1 original frames (then n = 16).")


def no_gt_reason(split: str, n_stage1: int, n_stage2: int, n_standin: int) -> str:
    """⛔ W7 2026-09-21: the refusal text is DERIVED from the run, never the warmup constant above.
    MEASURED on the navhard A1 artifact: `NO_GT` told a navhard reader "the 204 stage-2 scenes … the
    16 stage-1 scenes … 0/192 jpgs" — every number from a DIFFERENT split. A refusal whose reason
    describes another corpus is a wrong claim wearing a refusal's shape."""
    return (f"no {split} scene carries BOTH this arm's camera input and a logged human future: the "
            f"{n_stage2} stage-2 synthetic scenes have frames but num_future_frames = 0 (no human drove "
            f"them), and the {n_stage1} stage-1 scenes have a human future but their camera frames are not "
            f"unpacked on this box — so {n_standin} stage-1 rows of this arm are the devkit CV STAND-IN and "
            f"none of this arm's plans can be compared with a human future. Unblock = extract the stage-1 "
            f"frames from the verified OpenScene test camera archives (then n = {n_stage1}).")


def _set_path(art: dict, dotted: str, value) -> None:
    cur = art
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def refuse_every_absent_criterion(art: dict, reg: dict, reason: str, n: int) -> list:
    """⛔ W7 2026-09-21. A FAMILY-level refusal does not satisfy the checker: ``criteria_check.classify``
    looks up each criterion's OWN ``keys`` (e.g. ``four_families.longitudinal.speed_mae_mps``) and never
    walks up to a parent refusal. MEASURED on the navhard A1 artifact: 15 VIOLATIONS ("no key present and
    not refused") for criteria whose family WAS refused. So every criterion the checker would read as
    ABSENT gets its own ``{status: UNAVAILABLE, reason, n}`` — via ``refused_as`` where the registry names
    one, else at a key path the criterion lists (a ``four_families.``/non-headline path preferred). It
    touches ONLY criteria the checker currently classifies ABSENT; a PRESENT or already-REFUSED criterion
    is never overwritten, so this cannot hide a real value. Returns the criterion ids it refused."""
    done = []                      # `cc` = the module-level tools/criteria_check import (line ~39)
    blocks = []
    for fam in (reg.get("families") or {}).values():
        blocks.extend(fam if isinstance(fam, list) else fam.get("criteria", []))
    lg = reg.get("leak_guards")
    blocks.extend(lg if isinstance(lg, list) else (lg or {}).get("criteria", []) if isinstance(lg, dict) else [])
    for crit in blocks:
        if not isinstance(crit, dict) or not crit.get("required", True):
            continue
        state, _ = cc.classify(art, crit)
        if state != cc.ABSENT:
            continue
        refusal = {"status": "UNAVAILABLE", "reason": reason, "n": int(n)}
        if crit.get("refused_as"):
            art.setdefault("refused", {})[crit["refused_as"][0]] = reason
        else:
            keys = crit.get("keys") or []
            if not keys:
                continue
            pick = next((k for k in keys if k.startswith("four_families.")), None) or \
                next((k for k in keys if not k.startswith("headline.")), None) or keys[0]
            _set_path(art, pick, refusal)
        done.append(crit.get("id"))
    return done


# --------------------------------------------------------------------------- #
# E1 build_artifacts.py — VERBATIM                                             #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# E2 build_artifacts.py — VERBATIM                                             #
# --------------------------------------------------------------------------- #
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


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# --------------------------------------------------------------------------- #
# W1 addition: undefined LATERAL terms are REFUSED, never left as a bare None  #
# --------------------------------------------------------------------------- #
LATERAL_TERMS = ("heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps")


def refuse_undefined_lateral(art: dict) -> list:
    """MEASURED 2026-09-19 on the STOP floor: ``four_families.lateral`` emits ``None`` for heading /
    curvature / yaw-rate when EVERY step is below ``min_ds_m`` (a stationary plan has no tangent), and
    ``tools/criteria_check.py`` reads a ``None`` as ABSENT — a silent-omission VIOLATION. The fact is
    that the terms are UNDEFINED for this arm, so they are refused WITH the harness's own counts.

    ⭐ SINCE 2026-09-20 THE SOURCE DOES IT (W2): ``four_families.lateral`` emits the refusal itself,
    so this seam normally repairs NOTHING — it stays as the regression guard for a return to null,
    and the list it returns is now "the lateral terms that ARE refused", repaired here or upstream,
    which is the fact the caller and its test care about."""
    lat = (art.get("four_families") or {}).get("lateral")
    if not isinstance(lat, dict) or str(lat.get("status", "OK")).upper() == "UNAVAILABLE":
        return []
    fixed = []
    for k in LATERAL_TERMS:
        v = lat.get(k)
        if isinstance(v, dict) and str(v.get("status", "")).upper() in ("UNAVAILABLE", "REFUSED"):
            fixed.append(k)                 # already refused at the SOURCE (W2) — nothing to repair
            continue
        if k in lat and lat[k] is None:
            lat[k] = {"status": "UNAVAILABLE", "n": 0,
                      "reason": (f"{k} is UNDEFINED for this plan: {lat.get('excluded_below_min_ds')} step(s) moved less "
                                 f"than min_ds_m={lat.get('min_ds_m')} m (n_steps_heading={lat.get('n_steps_heading')}, "
                                 f"n_steps_curvature={lat.get('n_steps_curvature')}) — a stationary path has no tangent, "
                                 "so heading/curvature/yaw-rate errors do not exist (four_families.lateral min_ds rule); "
                                 "reported as a refusal, never as zero")}
            fixed.append(k)
    return fixed


# --------------------------------------------------------------------------- #
# the suite's artifact per arm (glue restructured from E1/E2 main())          #
# --------------------------------------------------------------------------- #
def load_registry() -> dict:
    return json.loads((REPO / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json").read_text(encoding="utf-8"))


def build_arm_artifact(*, arm: str, spec: dict, split: str, protocol: str, raw_rows: dict, hooks: list,
                       S1: set, S2: set, n_logs: int, interval: dict, n_standin: int = 0,
                       runtime_note: str = "", controls: dict | None = None) -> dict:
    """E1 main() loop body + E2 gap fills, as a function of ONE arm's scored rows.

    ``spec``: {agent, ii (navsim_inference_inputs kwargs), sensor_set, setting, goal, vision_only_claimed,
    ego_enforcement (dict, for vision-only claims), route_input (bool)}.
    """
    reg = load_registry()
    body = {t: r for t, r in raw_rows.items() if t not in SUMMARY.values()}
    n_valid = {st: sum(1 for t, r in body.items() if t in toks and r.get("valid") == "True")
               for st, toks in (("stage_one", S1), ("stage_two", S2))}
    combined = raw_rows.get(SUMMARY["combined"])
    two_stage_defined = bool(combined and combined.get("score") not in (None, "", "nan")
                             and n_valid["stage_one"] == len(S1) and n_valid["stage_two"] == len(S2)
                             and n_standin == 0)
    fam_scope = (f"OUR instruments on the {len(S1)} STAGE-1 scenes only (agent vs logged human future, dt 0.5 s, "
                 "K=8); stage-2 synthetic scenes have NO human future (0 future frames)")
    if n_standin == 0:
        try:
            win, fv, n_win = build_win(hooks, S1)
        except Exception as e:                                          # noqa: BLE001 — recorded, never silent
            win, fv, n_win = None, f"RAISED {type(e).__name__}: {e}"[:300], 0
    else:
        win, fv, n_win = None, "not built: stage-1 rows are a CV stand-in, not this arm", 0
    if two_stage_defined:
        epdms = ad.read_epdms(short_row(combined, "one") | {"score": float(combined["score"])})
        epdms["by_stage"] = {st: float(raw_rows[SUMMARY[st]]["score"]) for st in SUMMARY}
        sub = {st: ad.submetrics_from_row(short_row(raw_rows[SUMMARY[st]], "one" if st == "stage_one" else "two"))
               for st in ("stage_one", "stage_two")}
    else:
        why = (f"two-stage EPDMS UNDEFINED for {arm}: {n_standin} stage-1 rows are the devkit CV STAND-IN "
               "(no stage-1 frames), so both the stage-1 factor and the stage-2 kernel weights are not this arm's"
               if n_standin else f"two-stage EPDMS UNDEFINED for {arm}: valid rows {n_valid} vs expected "
               f"{len(S1)}/{len(S2)}")
        epdms = {"status": "UNAVAILABLE", "reason": why, "n": int(n_standin or len(S1))}
        sub = {"stage_two": ad.submetrics_from_row(short_row(raw_rows[SUMMARY["stage_two"]], "two"))
               if SUMMARY["stage_two"] in raw_rows else {"status": "UNAVAILABLE", "reason": why, "n": len(S2)}}
    ii = ad.navsim_inference_inputs(**spec["ii"])
    # ⛔ W2 settled the route-leak question on 2026-09-20: the adapter's DEFAULT route_leak_check()
    # returns the SETTLED verdict (PARTIAL — ROUTE-LEVEL ORACLE). Passing "UNVERIFIED" now FAILS
    # criteria_check for any arm that consumes the driving command.
    route_leak = ({"status": "NOT_APPLICABLE", "n": len(S1) + len(S2),
                   "reason": "this arm consumes no route / driving_command input, so a route-derived leak cannot enter it"}
                  if not spec.get("route_input") else ad.route_leak_check())
    if win is not None:
        try:
            art = ad.build_artifact(win, tier="T1", variant="EPDMS_v2", split=split, arm=f"{arm}@{split}",
                                    epdms=epdms, submetrics=sub, inference_inputs=ii, goal_source=spec["goal"],
                                    route_leak=route_leak)
        except Exception as e:                                          # noqa: BLE001
            fv = f"four_families RAISED {type(e).__name__}: {e}"[:400]
            win = None
    if win is None:
        dummy = ad.scenes_to_win(np.zeros((1, 8, 3)), None, frame="ego", origin_included=False, dt_s=0.5,
                                 scene_tokens=["_no_geometry_"], verify=False)
        art = ad.build_artifact(dummy, tier="T1", variant="EPDMS_v2", split=split, arm=f"{arm}@{split}",
                                epdms=epdms, submetrics=sub, inference_inputs=ii, goal_source=spec["goal"],
                                route_leak=route_leak)
        why = (no_gt_reason(split, len(S1), len(S2), n_standin) if n_standin
               else f"our geometry families could not be built: {fv}")
        for fam in ("longitudinal", "lateral", "tactical", "strategic"):
            art["four_families"][fam] = {"status": "UNAVAILABLE", "reason": why, "n": 0, "tier": "T1"}
        art["four_families"]["_families_unavailable"] = ["longitudinal", "lateral", "tactical", "strategic"]
    art["n_windows"] = len(S1) + len(S2)
    art["n_scenes"] = len(S1) + len(S2)
    art["_w1_lateral_refusals"] = refuse_undefined_lateral(art)
    art["counts"] = {"stage_one_expected": len(S1), "stage_two_expected": len(S2), "valid": n_valid,
                     "four_families_windows": n_win, "log_groups": n_logs, "stage_one_cv_standin_rows": n_standin}
    art["four_families"]["_w1_scope"] = fam_scope + " — frame check: " + str(fv)
    # ⛔ W7 2026-09-21 — this block was written for the v2.9.0 rule ("only UNAVAILABLE is admissible").
    # That rule is DEAD since registry 2.10.0 (SETTLED 2026-09-19: a log-cluster interval over >= 8
    # log_names is admitted), and v2.10.3 now REJECTS both halves of the old shape: a `cluster_unit`
    # refusal whose reason is the single token "log_name" ("a unit name is not a reason"), and an
    # `interval` that says UNAVAILABLE while HOLDING an admissible interval ("a FALSE REFUSAL").
    # MEASURED: the identical CV/STOP artifacts read 0 violations under 2.10.2 and 1 each under 2.10.3.
    # ⇒ the unit is declared as the settled string, and an admissible interval is PROMOTED.
    # ⛔⛔ EXCEPT for a CV-stand-in arm: its only two-stage aggregate is the HYBRID (stage 1 = the devkit
    # CV agent, which also sets the stage-2 kernel weights), so an OK interval on it is an interval on
    # the hybrid, NOT on this arm. MEASURED on navhard A1: the "OK" interval's point 0.1113 == the HYBRID
    # combined row 0.111329 — promoting it would publish CV's stage 1 as refcv4b's uncertainty.
    if n_standin:
        # ⛔ a CLEAN refusal. The hybrid's interval is NOT parked inside it: registry 2.10.3's gate
        # (`_admissible_interval_hiding_in`) correctly treats an admissible interval found under a
        # refusal as a FALSE REFUSAL — MEASURED on the first version of this fix. Its intent is right
        # (never hide a real interval under a refusal), so the hybrid's interval lives OUTSIDE the
        # estimator block, under a key that says whose it is.
        est_interval = {"status": "UNAVAILABLE", "n": n_logs,
                        "reason": (f"the only two-stage aggregate for {arm} is the CV-stand-in HYBRID ({n_standin} "
                                   "stage-1 rows are the devkit CV agent, which also sets the stage-2 kernel "
                                   "weights); an interval on it is an interval on the hybrid, not on this arm")}
        art["hybrid_combined_row_NOT_this_arm"] = {
            "what": "the devkit's combined row for this arm = CV stage 1 x this arm's stage 2 (a HYBRID)",
            "interval_on_the_hybrid": interval}
    elif interval.get("status") == "OK" and isinstance(interval.get("detail"), dict):
        # ⭐ promote the ESTIMATOR'S OWN block (navsim_ci output: estimator, aggregation, cluster_unit,
        # n_clusters, point, lo, hi, …). MEASURED: the suite's wrapper keeps `aggregation` inside
        # `detail`, so the gate read `estimator.interval.aggregation` as None ("not a registered
        # official one") although the aggregation IS the official two-stage one.
        est_interval = {**interval["detail"],
                        **{k: interval[k] for k in ("pre_csv_frame", "question_answered") if k in interval}}
    else:
        est_interval = interval
    art["estimator"] = {"point_estimate": "devkit summary rows (two-stage aggregation)",
                        "cluster_unit": "log_name",
                        "interval": est_interval}
    art["tier"] = "T1"
    art["protocol"].update({
        "navsim_protocol": protocol, "devkit_sha": SHA,
        "devkit_pin_full": f"autonomousvision/navsim@{SHA} (2025-10-27; post-#151 fix, MEASURED by E1)",
        "harness_modifications": ["PRE-EXISTING navsim/common/dataclasses.py PosixPath unpickler",
                                  "PRE-EXISTING venv fcntl.py flock shim",
                                  "PRE-EXISTING nuplan-devkit setup.py (packaging only)",
                                  "E1 dataloader.py token-separator fix in the C: copy + in-process monkeypatch — reader only"],
        "runtime": runtime_note or "C:/Users/Admin/navsim-crun (E1's verified mirror)",
        "sensor_set": spec["sensor_set"], "setting": spec["setting"],
        "ego_status_enforcement": (spec["ego_enforcement"] if spec.get("vision_only_claimed") else
                                   {"vision_only_claimed": False, "status": "NOT_APPLICABLE",
                                    "reason": "arm is NOT vision-only and does not claim to be: " + spec["sensor_set"]}),
        "corpus": f"NavSim {split} (OpenScene test logs; {n_logs} logs; {len(S1)} stage-1 + {len(S2)} stage-2 tokens) — "
                  "a different benchmark from the TanitAD parity corpus physicalai-train-e438721ae894",
        "loop": {"stage_one": "OPEN (PI ruling 2026-09-02: fixed plan, ego never re-queried)",
                 "stage_two": "UNRULED under the 2026-09-02 vocabulary (one-shot re-perception of a rendered perturbed start)",
                 "background_traffic": "IDM-REACTIVE vehicles in BOTH stages (run_pdm_score.py:77-79,132-134)"},
        "tier_note": "T1-family: the arm's own plan is executed by LQR + kinematic bicycle; nothing recorded is fed back",
    })
    if controls is not None:
        art["controls"] = controls
    art["refused"] = {}
    if not spec.get("route_input"):
        art["refused"]["nav_compliance"] = (f"{arm} consumes NO route / driving_command input, so 'behaviour follows the "
                                            "route command' is undefined for it; NavSim never scores the route.")
        art["refused"]["nav_compliance_controls"] = "no route input exists to shuffle or withhold."
    art["refused"]["inference_seed_replicate"] = (
        "the floor arms are deterministic and the scoring path has no RNG (E1 C7: a hash-seed-2 replicate reproduced "
        "every per-token value bit-for-bit); a sampling planner arm would need an inference-seed replicate.")
    if n_standin:
        # every four-family / leak-guard criterion the checker would read as ABSENT gets its OWN refusal
        # (a family-level refusal is invisible to per-key lookups — see refuse_every_absent_criterion)
        art["_w7_refused_criteria"] = refuse_every_absent_criterion(
            art, reg, no_gt_reason(split, len(S1), len(S2), n_standin), 0)
    art["_w1_adapter_gaps"] = ["G1 long/stage-suffixed devkit columns vs short keys", "G2 estimator.cluster_unit",
                               "G3 protocol.ego_status_enforcement", "G4 protocol.sensor_set/setting",
                               "G5 protocol.navsim_protocol/devkit_sha", "G6 protocol.corpus/controls",
                               "criteria_check.py never evaluates benchmarks.navsim (W2 owns the fix)"]
    art["_promoted_from"] = ["FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/code/build_artifacts.py",
                             "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/code/build_artifacts.py"]
    gates = navsim_gates(json.loads(json.dumps(art, default=_jd)), reg)
    muts = gate_mutations(json.loads(json.dumps(art, default=_jd)), reg)
    art["navsim_gate_selfcheck"] = {"gates": gates["gates"], "criteria": gates["criteria"],
                                    "mutations_all_red": muts["all_red"], "mutations": muts,
                                    "_note": "E1's evaluator: criteria_check.py (registry v2.9.0) does not evaluate these gates"}
    return art


def _jd(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)
