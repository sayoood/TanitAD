"""``gate_emitters`` + ``tanitad.eval.speed_benefit`` — the three v4-gate KILL
secondaries that had no emitter, pinned so they cannot drift.

Reconciliation banked (STEP 0): all three live in ``flagship-v4.card.json``'s
``secondary`` array, so run_gate treats them as **KILL** (an unsupplied on-card
secondary -> ``pass: None`` -> verdict INCOMPLETE). ``test_run_gate_eval_metric.py``
already pins the report-only OFF-card channel; this file pins the emitters that
make the on-card KILL set complete.

Pure-Python where possible; the two integration pins read committed artifacts
(the v1 train logs and the v1 efficiency lever panel) and are json/statistics
only — no torch, no GPU.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STACK = REPO / "stack"
sys.path.insert(0, str(STACK / "scripts"))
sys.path.insert(0, str(STACK))

import gate_emitters as ge                                          # noqa: E402
from tanitad.eval import speed_benefit as sb                       # noqa: E402

TRAINLOGS = REPO / "taniteval" / "results" / "trainlogs"
EFF_LEVERS_V1 = REPO / "taniteval" / "results" / "eff_levers_flagship-30k.json"


# =========================================================================== #
# speed_benefit_recovered_frac — the pinned reducer                           #
# =========================================================================== #
def test_bucket_convention_is_left_open_right_closed():
    rows = {2000: {"step": 2000, "m": 1.0}, 3000: {"step": 3000, "m": 3.0},
            4000: {"step": 4000, "m": 9.0}}
    # (2000, 4000]: excludes 2000, includes 3000 and 4000 -> mean(3,9)=6
    assert sb.bucket_mean(rows, "m", 2000, 4000) == 6.0
    # (2000, 3000]: excludes 2000, includes 3000 -> 3
    assert sb.bucket_mean(rows, "m", 2000, 3000) == 3.0


def test_reducer_reproduces_the_published_81_8_and_18_6_at_8_10k():
    """P8 acceptance: the reducer reproduces the design's headline recovered
    fractions to the quoted precision (V4_FLAGSHIP_DESIGN §7.5: v1 81.8 %, v3enc
    18.6 % at the 8-10 k bucket)."""
    v1 = sb.emit(TRAINLOGS / "v1-speedjerk_train_log.jsonl",
                 nospeed_log=TRAINLOGS / "nospeed-phase0_train_log.jsonl",
                 repo_root=REPO)
    v3 = sb.emit(TRAINLOGS / "v3enc_train_log.jsonl",
                 nospeed_log=TRAINLOGS / "nospeed-phase0_train_log.jsonl",
                 repo_root=REPO)
    assert v1["value"] == 0.8184, v1["value"]        # design 81.8 %
    assert v3["value"] == 0.1859, v3["value"]        # design 18.6 %
    assert v1["pass"] is True and v3["pass"] is False
    assert v1["metric"] == "g_op_fwd_ade_m"
    assert v1["bucket"] == [8000, 10000]


def test_recovered_frac_formula_is_nospeed_minus_arm_over_nospeed():
    arm = {9000: {"step": 9000, "g_op_fwd_ade_m": 0.2}}
    ns = {9000: {"step": 9000, "g_op_fwd_ade_m": 1.0}}
    r = sb.recovered_frac(arm, ns, bucket=(8000, 10000))
    assert r["value"] == round((1.0 - 0.2) / 1.0, 4) == 0.8


def test_missing_metric_yields_none_not_a_fabricated_pass():
    """A v4 arm whose log lacks g_op_fwd_ade_m must read NOT SUPPLIED, never a
    silent 0/None-driven pass. (The v4 trainer now logs it — the fix — but an old
    v4.1/v4.2 log does not.)"""
    v41 = sb.emit(TRAINLOGS / "flagship-v4.1-10k_train_log.jsonl",
                  nospeed_log=TRAINLOGS / "nospeed-phase0_train_log.jsonl",
                  repo_root=REPO)
    assert v41["value"] is None and v41["pass"] is None
    assert v41["arm_bucket_mean"] is None


def test_three_consecutive_below_threshold_detector():
    below = {"0-2000": 0.1, "2000-4000": 0.2, "4000-6000": 0.3,
             "6000-8000": 0.9, "8000-10000": 0.95}
    assert sb._has_3_consecutive(below, 0.70) is True
    ok = {"0-2000": 0.9, "2000-4000": 0.1, "4000-6000": 0.2,
          "6000-8000": 0.9, "8000-10000": 0.95}
    assert sb._has_3_consecutive(ok, 0.70) is False


# =========================================================================== #
# deploy_tick_p99_ms — efficiency lever panel reader                          #
# =========================================================================== #
def test_deploy_tick_reads_the_composed_lever_p99_from_the_v1_panel():
    r = ge.deploy_tick_from_eff_json(EFF_LEVERS_V1, precision="fp32")
    assert r["deployed_lever"] == "all_levers"       # the fully-composed tick
    assert r["value"] == 18.7641                     # committed A40 p99
    assert r["pass"] is True                         # <= 50
    assert r["value"] == r["tick_ms"]["p99_ms"]      # p99, NOT mean/p50
    # the mission cross-check: composed tick ~18.75 ms
    assert abs(r["tick_ms"]["mean_ms"] - 18.75) < 0.1


def test_deploy_tick_rejects_a_fast_but_WRONG_lever():
    """A composed tick that does not decode the eager trajectory (large ADE
    delta) is a fast wrong answer and must NOT be chosen as the deploy tick."""
    ev = {"fp32": {"levers": {
        "all_levers": {"tick": {"p99_ms": 5.0},
                       "equivalence": {"ade_0_2s_delta_m": 9.9, "finite": True},
                       "meta": {"desc": "broken", "weights_dtype": "float16"}},
        "graph_rollout": {"tick": {"p99_ms": 40.0},
                          "equivalence": {"ade_0_2s_delta_m": 0.0, "finite": True},
                          "meta": {"desc": "ok", "weights_dtype": "float32"}}}}}
    r = ge.deploy_tick_from_eff_json_dict(ev, precision="fp32")
    assert r["deployed_lever"] == "graph_rollout"    # skipped the wrong-but-fast one
    assert r["value"] == 40.0


def test_deploy_tick_falls_back_through_preference_when_a_lever_did_not_build():
    ev = {"fp32": {"levers": {
        "enc_cache_graph": {"tick": {"p99_ms": 33.4},
                            "equivalence": {"ade_0_2s_delta_m": 0.0, "finite": True},
                            "meta": {"desc": "ok", "weights_dtype": "float32"}}}}}
    r = ge.deploy_tick_from_eff_json_dict(ev, precision="fp32")
    assert r["deployed_lever"] == "enc_cache_graph" and r["value"] == 33.4


def test_deploy_tick_none_when_not_a_lever_panel():
    r = ge.deploy_tick_from_eff_json_dict({"fp32": {"plan_step": {"p99_ms": 100}}})
    assert r["value"] is None and r["pass"] is None


# =========================================================================== #
# nonav_route_beats_majority — hierarchy panel reader                         #
# =========================================================================== #
def test_nonav_route_command_echo_is_zero():
    """v1's real read: route_acc_follow == majority_straight_rate (the follow head
    predicts straight always) -> vision_route_beats_majority False -> 0."""
    ev = {"seam_nav_to_strategic": {
        "vision_route_beats_majority": False,
        "route_acc_follow": 0.7083, "majority_straight_rate": 0.7083,
        "route_acc_nav": 1.0, "n_valid": 72,
        "follow_pred_distribution": {"route_left": 0, "route_straight": 72,
                                     "route_right": 0}}}
    r = ge.nonav_route_from_hierarchy_dict(ev)
    assert r["value"] == 0 and r["pass"] is False
    assert r["route_acc_nav_commanded"] == 1.0       # perfect BY CONSTRUCTION


def test_nonav_route_genuine_strategic_head_is_one():
    ev = {"seam_nav_to_strategic": {
        "vision_route_beats_majority": True,
        "route_acc_follow": 0.82, "majority_straight_rate": 0.71, "n_valid": 72}}
    r = ge.nonav_route_from_hierarchy_dict(ev)
    assert r["value"] == 1 and r["pass"] is True


def test_nonav_route_recomputes_boolean_when_key_absent():
    """If a hierarchy JSON lacks the boolean, recompute from acc_follow vs
    majority + margin (hierarchy.py's own rule)."""
    ev = {"seam_nav_to_strategic": {
        "route_acc_follow": 0.7083, "majority_straight_rate": 0.7083}}
    assert ge.nonav_route_from_hierarchy_dict(ev)["value"] == 0
    ev2 = {"seam_nav_to_strategic": {
        "route_acc_follow": 0.80, "majority_straight_rate": 0.71}}
    assert ge.nonav_route_from_hierarchy_dict(ev2)["value"] == 1


# =========================================================================== #
# gate-values assembly + the --secondary-value formatting                     #
# =========================================================================== #
def test_gate_values_formats_the_secondary_value_args(tmp_path):
    hj = tmp_path / "hier.json"
    hj.write_text(json.dumps({"seam_nav_to_strategic": {
        "vision_route_beats_majority": False, "route_acc_follow": 0.7083,
        "majority_straight_rate": 0.7083}}))
    out = ge.gate_values(eff_json=str(EFF_LEVERS_V1), hierarchy_json=str(hj),
                         arm_log="taniteval/results/trainlogs/"
                                 "v1-speedjerk_train_log.jsonl", repo_root=str(REPO))
    args = set(out["secondary_value_args"])
    assert "deploy_tick_p99_ms=18.7641" in args
    assert "speed_benefit_recovered_frac=0.8184" in args
    assert "nonav_route_beats_majority=0" in args    # bool int, not 0.0/False
    assert out["all_three_emitted"] is True and out["missing"] == []


def test_gate_values_reports_missing_when_an_input_absent():
    out = ge.gate_values(eff_json=str(EFF_LEVERS_V1))       # only deploy tick
    assert "deploy_tick_p99_ms" not in out["missing"]
    assert set(out["missing"]) == {"speed_benefit_recovered_frac",
                                   "nonav_route_beats_majority"}
    assert out["all_three_emitted"] is False


# =========================================================================== #
# end-to-end through run_gate: the 3 emitters flip INCOMPLETE -> COMPLETE     #
# =========================================================================== #
def _dryrun_card(path):
    path.write_text(json.dumps({
        "run": "t", "gate_step": 10000, "primary_metric": "ade_0_2s",
        "primary_threshold": 0.60, "primary_direction": "<=",
        "primary_source": "held-out full-set",
        "secondary": ["wm_canary_ade_2s<=0.55", "speed_benefit_recovered_frac>=0.70",
                      "oracle_in_fan<=0.30", "miss_at_2m<=0.10",
                      "seam_norm_ratio_max<=1.0", "encoder_touching_levers<=2",
                      "deploy_tick_p99_ms<=50", "nonav_route_beats_majority>=1"],
        "reference_run": None, "reference_log": None, "compare_metric": None,
        "tau": 1.5, "lever_family": "joint-planner-wm", "restarts_used": 0,
        "restart_cap": 2, "registered_utc": "2026-07-23T00:00:00+00:00", "note": ""}))


def _run_gate(tmp_path, secondary_values):
    import run_gate as rg
    card = tmp_path / "card.json"
    _dryrun_card(card)
    (tmp_path / "eval.json").write_text(json.dumps({"cluster_bootstrap": {"model": {
        "ade_0_2s": {"mean": 0.4271, "lo": 0.3675, "hi": 0.4871,
                     "estimator": "episode_cluster_bootstrap"}}}}))
    (tmp_path / "log.jsonl").write_text(
        '{"step": 0, "step_s": 0, "g_op_fwd_ade_m": 1.0}\n'
        '{"step": 10000, "step_s": 100, "g_op_fwd_ade_m": 0.5}\n')
    out = tmp_path / "gate.json"
    rg.main(["check", "--card", str(card), "--log", str(tmp_path / "log.jsonl"),
             "--eval-json", str(tmp_path / "eval.json"),
             "--secondary-value", *secondary_values, "--json", str(out)])
    return json.loads(out.read_text())


FIVE_EXISTING = ["wm_canary_ade_2s=0.452", "oracle_in_fan=0.16", "miss_at_2m=0.0602",
                 "seam_norm_ratio_max=1.0", "encoder_touching_levers=2"]
THREE_NEW = ["deploy_tick_p99_ms=18.7641", "speed_benefit_recovered_frac=0.8184",
             "nonav_route_beats_majority=1"]


def test_without_the_three_emitters_the_gate_is_INCOMPLETE(tmp_path):
    g = _run_gate(tmp_path, FIVE_EXISTING)
    assert g["verdict"] == "INCOMPLETE"
    unmet = {r["metric"] for r in g["secondary"] if r["pass"] is None}
    assert unmet == {"deploy_tick_p99_ms", "speed_benefit_recovered_frac",
                     "nonav_route_beats_majority"}


def test_with_the_three_emitters_the_gate_renders_a_COMPLETE_verdict(tmp_path):
    g = _run_gate(tmp_path, FIVE_EXISTING + THREE_NEW)
    assert g["verdict"] != "INCOMPLETE"
    assert g["verdict"] == "CONTINUE"                # all 8 pass here
    assert all(r["pass"] is not None for r in g["secondary"])


def test_a_failing_new_emitter_still_completes_as_RESTART_not_INCOMPLETE(tmp_path):
    """The honest v1 read: nonav_route=0 fails, but the verdict is a COMPLETE
    RESTART, never INCOMPLETE — the machinery renders a full verdict."""
    g = _run_gate(tmp_path, FIVE_EXISTING + [
        "deploy_tick_p99_ms=18.7641", "speed_benefit_recovered_frac=0.8184",
        "nonav_route_beats_majority=0"])
    assert g["verdict"] == "RESTART"
    assert all(r["pass"] is not None for r in g["secondary"])
