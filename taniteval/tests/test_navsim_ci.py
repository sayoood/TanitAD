"""Tests for the NavSim LOG-CLUSTER bootstrap (``taniteval/adapters/navsim_ci.py``).

Pre-registered in ``FlyWheels/TanitAD_EvalFlyWheel/incoming/
2026-09-19-navsim-estimator-and-route-leak/SPEC.md`` §8 — every expectation here is a
LITERAL written before the code under test produced it (analytic targets, a value derived
on paper, the devkit's OWN function output, and the devkit's own summary row on a real
run). None is an expression over the code under test.

⭐ Two DELIBERATE-REGRESSION arms, and why both are needed:
  * a CODE mutation that makes the estimator resample units instead of logs. Its record's
    metadata still says ``n_clusters = 10`` — the metadata is produced by the same code, so
    only an INDEPENDENT analytic target (the between-log SE) can catch it;
  * a CALLER defect (scene tokens passed as the clusters). That one the metadata CAN see
    (``n_clusters`` exceeds the split's log count), which is what the gate checks.

Data-backed tests skip ONLY when their input tree is absent (``NO_TREE``) and say so.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tests
_TE = os.path.dirname(_HERE)                             # <repo>/taniteval
_REPO = os.path.dirname(_TE)                             # <repo>
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import navsim_ci as NC                      # noqa: E402

PKG = os.path.join(_REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming",
                   "2026-09-19-navsim-estimator-and-route-leak")
E1 = os.path.join(_REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming",
                  "2026-09-19-navsim-warmup-reference-epdms")
REGISTRY = os.path.join(_REPO, "products", "P7-TanitEval", "CRITERIA_REGISTRY.json")

SS = NC.AGG_SINGLE_STAGE
TS = NC.AGG_TWO_STAGE

#: sqrt(var_pop([0.1 .. 1.0]) / 10) = sqrt(0.0825 / 10) — the bootstrap SE of the mean
#: of 10 equal-size clusters with those means (SPEC §8 T-SE), derived on paper.
ANALYTIC_SE = 0.09082951062292474
#: the devkit's own `calculate_individual_mapping_scores` on the F2 fixture — identical
#: under pandas 2.3.3 (the NavSim runtime) and 3.0.3 (raw/devkit_aggregation_reference*.json)
F2_DEVKIT_COMBINED = 0.2886492611337752
#: E1's real CV warmup run: `extended_pdm_score_combined` in
#: raw/A1/devkit_2026.09.19.12.24.20.csv (= HF warmup LB 18.5356)
E1_A1_COMBINED = 0.1853562745165113


def _logs(n_logs, per):
    return [f"L{k}" for k in range(n_logs) for _ in range(per)]


def _overlapping_fixture():
    """10 logs x 20 scenes, scores IDENTICAL within a log (the extreme of sliding-window
    overlap: 5.55 scenes per frame on navtest), log means 0.1 .. 1.0."""
    means = [0.1 * (k + 1) for k in range(10)]
    c = [m for m in means for _ in range(20)]
    cl = [f"L{k}" for k in range(10) for _ in range(20)]
    toks = [f"t{i:03d}" for i in range(200)]
    return c, cl, toks


def _se_matches_analytic(rec, tol=0.08):
    return abs(rec["se"] - ANALYTIC_SE) / ANALYTIC_SE < tol


# ========================================================================== #
# 1. Analytic targets                                                        #
# ========================================================================== #
def test_T_ZERO_identical_scores_give_an_exactly_zero_width_interval():
    r = NC.log_cluster_bootstrap([0.75] * 40, _logs(10, 4), aggregation=SS)
    assert r["status"] == "OK"
    assert (r["point"], r["lo"], r["hi"]) == (0.75, 0.75, 0.75)
    assert r["se"] == 0.0 and r["ci95"] == 0.0


def test_T_TWO_two_cluster_fixture_has_the_analytic_bootstrap_distribution():
    c = [1.0] * 5 + [0.0] * 5
    cl = ["A"] * 5 + ["B"] * 5
    b = NC.boot_statistics(c, cl, n_boot=2000, seed=0)
    # resampling 2 clusters with replacement: {AA, AB, BA, BB} -> {1.0, 0.5, 0.5, 0.0}
    assert set(np.round(b, 12).tolist()) <= {0.0, 0.5, 1.0}
    assert abs(float(b.mean()) - 0.5) < 0.03              # E[theta*] = 0.5 exactly
    assert abs(float((np.round(b, 12) == 0.5).mean()) - 0.5) < 0.04   # P(mixed) = 1/2
    r = NC.log_cluster_bootstrap(c, cl, aggregation=SS, min_clusters=2)
    assert (r["point"], r["lo"], r["hi"]) == (0.5, 0.0, 1.0)


def test_T_SE_the_log_cluster_se_is_the_between_log_analytic_value():
    c, cl, _ = _overlapping_fixture()
    r = NC.log_cluster_bootstrap(c, cl, aggregation=SS)
    assert r["status"] == "OK" and r["n_clusters"] == 10 and r["n_units"] == 200
    assert _se_matches_analytic(r), f"se {r['se']} vs analytic {ANALYTIC_SE}"


def test_DELIBERATE_REGRESSION_T_MUT_unit_resampling_inside_the_estimator_goes_RED(monkeypatch):
    """Reintroduce scene-token (non-clustered) resampling INSIDE the machine."""
    c, cl, _ = _overlapping_fixture()
    correct = NC.log_cluster_bootstrap(c, cl, aggregation=SS)
    real_index = NC._ci.episode_index

    def unit_level_index(eid):                  # the mutation: every unit its own cluster
        return real_index([f"unit{i}" for i in range(len(eid))])

    monkeypatch.setattr(NC._ci, "episode_index", unit_level_index)
    mutated = NC.log_cluster_bootstrap(c, cl, aggregation=SS)
    assert not _se_matches_analytic(mutated), "the analytic target failed to go RED"
    assert mutated["se"] < 0.5 * correct["se"], "the defect's signature is a NARROWER interval"
    # ⚠️ and the record cannot see it — its metadata comes from the same code:
    assert mutated["n_clusters"] == 10


def test_DELIBERATE_REGRESSION_scene_tokens_passed_as_clusters_fail_admissibility():
    c, _, toks = _overlapping_fixture()
    r = NC.log_cluster_bootstrap(c, toks, aggregation=SS)
    assert r["status"] == "OK" and r["n_clusters"] == 200
    assert not _se_matches_analytic(r)
    verdict, why = NC.is_admissible_interval(r, protocol="EPDMS_v2_navtest_single_stage",
                                             max_clusters=10)
    assert verdict == "FAIL" and "exceeds" in why


def test_T_FLOOR_fewer_than_eight_logs_is_refused_with_its_n():
    r = NC.log_cluster_bootstrap([0.5] * 14, _logs(7, 2), aggregation=TS)
    assert r["status"] == "UNAVAILABLE"
    assert r["n"] == 7 and r["n_clusters"] == 7 and r["min_clusters"] == 8
    assert "floor of 8" in r["reason"]
    assert NC.is_admissible_interval(r)[0] == "REFUSED"


def test_eight_logs_is_enough_to_rule():
    r = NC.log_cluster_bootstrap([0.5, 0.6] * 8, _logs(8, 2), aggregation=TS)
    assert r["status"] == "OK" and r["n_clusters"] == 8


# ========================================================================== #
# 2. The statistic is the devkit's, not a proxy                              #
# ========================================================================== #
def test_T_KEY_the_worked_example_contribution_is_0_44():
    rows = {"o": {"score": 0.8, "weight": 1.0}, "p": {"score": 0.6, "weight": 1.0},
            "n1": {"score": 1.0, "weight": 0.75}, "n2": {"score": 0.5, "weight": 0.25},
            "q1": {"score": 0.4, "weight": 0.5}, "q2": {"score": 0.2, "weight": 0.5}}
    c = NC.two_stage_key_contributions(rows, [("o", "p", [("n1", "q1"), ("n2", "q2")])])
    assert c.shape == (1,) and abs(float(c[0]) - 0.44) < 1e-12


def test_the_devkit_nan_semantics_are_reproduced():
    # skip-NaN mean (pd.DataFrame.mean) — a failed unit changes the DENOMINATOR
    assert abs(NC.official_aggregate([0.2, float("nan"), 0.6]) - 0.4) < 1e-12
    assert math.isnan(NC.official_aggregate([float("nan")] * 3))
    rows = {"a": {"score": 0.5, "weight": 0.0}, "b": {"score": float("nan"), "weight": 1.0},
            "c": {"score": 0.4, "weight": 1.0}}
    assert math.isnan(NC._wavg([], rows, "score"))          # df.empty
    assert math.isnan(NC._wavg(["x"], rows, "score"))       # listed but no row
    assert math.isnan(NC._wavg(["a"], rows, "score"))       # total weight 0
    assert math.isnan(NC._wavg(["b", "c"], rows, "score"))  # plain .sum() propagates NaN
    assert NC._wavg(["c", "c", "c"], rows, "score") == 0.4  # isin(): each row counted once


def test_T_DEVKIT_the_two_stage_aggregation_equals_the_devkits_own_function():
    p = os.path.join(PKG, "raw", "devkit_aggregation_reference.json")
    if not os.path.exists(p):
        pytest.skip(f"NO_TREE: {p} (the banked devkit reference) is not in this checkout")
    d = json.load(open(p, encoding="utf-8"))
    assert d["F2"]["devkit_combined"]["score"] == F2_DEVKIT_COMBINED     # the literal, pinned
    got = NC.official_aggregate(NC.two_stage_key_contributions(
        d["F2"]["inputs"]["rows"], d["F2"]["inputs"]["mapping"]))
    assert abs(got - F2_DEVKIT_COMBINED) < 1e-12
    assert d["F1"]["devkit_equals_literal"] is True
    # dropping whole logs (F3) and PHYSICALLY duplicated logs (F4) — the bootstrap's premise
    assert d["F3"]["n_equal"] == d["F3"]["n_trials"] == 12
    assert d["F4"]["n_equal"] == d["F4"]["n_trials"] == 12


def test_T_REPRO_the_official_warmup_EPDMS_is_reproduced_from_a_real_run():
    frame = os.path.join(E1, "raw", "A1", "A1_final_scores_frame.csv")
    mp = os.path.join(PKG, "raw", "cluster_maps", "warmup_two_stage_reactive_all_mapping.json")
    for p in (frame, mp):
        if not os.path.exists(p):
            pytest.skip(f"NO_TREE: {p} is not in this checkout")
    with open(frame, newline="") as f:
        rows = {r["token"]: {"score": float(r["score"]), "weight": float(r["weight"]),
                             "log_name": r["log_name"]} for r in csv.DictReader(f)}
    mapping = json.load(open(mp, encoding="utf-8"))["reactive_all_mapping"]
    assert len(rows) == 220 and len(mapping) == 8
    c = NC.two_stage_key_contributions(rows, mapping)
    assert abs(NC.official_aggregate(c) - E1_A1_COMBINED) <= 1e-12
    r = NC.log_cluster_bootstrap(c, [rows[k[0]]["log_name"] for k in mapping],
                                 aggregation=TS, official_value=E1_A1_COMBINED)
    # ⛔ warmup NEVER carries an interval: 7 logs < 8 (D-BENCH-PORT, re-measured)
    assert r["status"] == "UNAVAILABLE" and r["n_clusters"] == 7
    assert r["reproduces_official"]["status"] == "OK"


def test_rows_from_score_frame_reads_the_pre_CSV_frame_and_refuses_the_published_one(tmp_path):
    """The one call W1 needs — and the refusal that makes the missing `weight` loud."""
    frame = os.path.join(E1, "raw", "A1", "A1_final_scores_frame.csv")
    if not os.path.exists(frame):
        pytest.skip(f"NO_TREE: {frame} is not in this checkout")
    rows, tok2log = NC.rows_from_score_frame(frame)
    assert len(rows) == 220 and len(set(tok2log.values())) == 7   # warmup: 7 logs
    mp = os.path.join(PKG, "raw", "cluster_maps", "warmup_two_stage_reactive_all_mapping.json")
    mapping = json.load(open(mp, encoding="utf-8"))["reactive_all_mapping"]
    got = NC.official_aggregate(NC.two_stage_key_contributions(rows, mapping))
    assert abs(got - E1_A1_COMBINED) <= 1e-12
    # the PUBLISHED csv shape (no weight, no log_name) must be REFUSED, not guessed
    p = tmp_path / "published.csv"
    p.write_text("\n".join(["token,valid,score", "abc,True,0.5", ""]), encoding="utf-8")
    with pytest.raises(NC.NavSimCIError, match="weight"):
        NC.rows_from_score_frame(str(p))


def test_a_point_that_does_not_reproduce_the_official_value_is_refused():
    c, cl, _ = _overlapping_fixture()
    r = NC.log_cluster_bootstrap(c, cl, aggregation=SS, official_value=0.5)
    assert r["status"] == "UNAVAILABLE" and "does not reproduce" in r["reason"]


def test_an_unnamed_aggregation_or_protocol_is_refused():
    with pytest.raises(NC.NavSimCIError):
        NC.log_cluster_bootstrap([0.5] * 16, _logs(8, 2), aggregation="mean_of_scenes")
    with pytest.raises(NC.NavSimCIError):
        NC.interval_from_run(protocol="EPDMS_v2_navtest", clusters_by_unit={}, scores={})


# ========================================================================== #
# 3. The paired form                                                          #
# ========================================================================== #
def test_T_PAIR_a_constant_offset_gives_an_exact_paired_interval():
    rng = np.random.default_rng(3)
    b = rng.uniform(0.2, 0.8, size=40)
    a = b + 0.1
    r = NC.paired_log_cluster_bootstrap(a, b, _logs(10, 4), aggregation=SS)
    assert r["status"] == "OK" and r["estimator"] == NC.PAIRED_ESTIMATOR
    assert (r["delta"], r["lo"], r["hi"]) == (0.1, 0.1, 0.1)
    assert r["separated"] is True and "AND nuplan_drive" in r["separated_scope"]
    assert "n_episodes" not in r, "the unit is a LOG; never label it an episode"


def test_T_PAIR_arms_that_failed_on_different_units_are_refused():
    b = np.full(40, 0.5)
    a = b.copy()
    a[3] = np.nan
    r = NC.paired_log_cluster_bootstrap(a, b, _logs(10, 4), aggregation=SS)
    assert r["status"] == "UNAVAILABLE" and "DIFFERENT units" in r["reason"]


def test_the_drive_level_conjunction_says_when_it_cannot_rule():
    # 8 segments from only 4 nuPlan drives: the log level rules, the drive level cannot
    logs = [f"2021.05.25.14.16.1{d}_veh-35_0000{s}_0010{s}" for d in range(4) for s in range(2)]
    cl = [lg for lg in logs for _ in range(3)]
    b = np.linspace(0.2, 0.8, len(cl))
    r = NC.paired_log_cluster_bootstrap(b + 0.05, b, cl, aggregation=SS)
    assert r["status"] == "OK" and r["n_clusters"] == 8
    assert r["sensitivity_nuplan_drive"]["status"] == "CANNOT-RULE"
    assert "log_name ONLY" in r["separated_scope"]


def test_drive_of_strips_the_segment_suffix_only():
    assert NC.drive_of("2021.05.25.14.16.10_veh-35_00083_00485") == "2021.05.25.14.16.10_veh-35"
    assert NC.drive_of("L7") == "L7"


# ========================================================================== #
# 4. End to end, and the pins to the registry / release gate                 #
# ========================================================================== #
def test_interval_from_run_reproduces_the_devkit_value_and_is_admissible():
    p = os.path.join(PKG, "raw", "devkit_aggregation_reference.json")
    if not os.path.exists(p):
        pytest.skip(f"NO_TREE: {p}")
    d = json.load(open(p, encoding="utf-8"))
    inp = d["F2"]["inputs"]
    r = NC.interval_from_run(protocol="EPDMS_v2_navhard_two_stage", rows=inp["rows"],
                             mapping=inp["mapping"], clusters_by_unit=inp["key_log"],
                             official_value=F2_DEVKIT_COMBINED)
    assert r["status"] == "OK" and r["n_clusters"] == 10
    assert r["reproduces_official"]["status"] == "OK"
    assert r["n_units_nan"] == 3, "the F2 fixture's three NaN keys must be reported, not hidden"
    assert NC.is_admissible_interval(r, protocol="EPDMS_v2_navhard_two_stage",
                                     max_clusters=76)[0] == "PASS"
    assert NC.is_admissible_interval(r, protocol="PDMS_v1_navtest")[0] == "FAIL", \
        "a two-stage aggregate must not pass as a single-stage protocol"


def test_min_clusters_is_the_RG14_floor_everywhere():
    assert NC.MIN_CLUSTERS == 8
    sys.path.insert(0, os.path.join(_REPO, "tools"))
    import release_gate                                          # noqa: E402
    assert release_gate.EPISODE_CLUSTER_FLOOR == 8
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    g = reg["benchmarks"]["navsim"]["GATE_estimator_cluster_unit"]
    assert g["min_clusters"] == 8 and g["n_boot_min"] == NC.N_BOOT == 2000
    assert g["cluster_unit"] == NC.CLUSTER_UNIT == "log_name"


def test_the_protocol_map_is_the_registry_closed_set():
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    closed = set(reg["benchmarks"]["navsim"]["GATE_no_cross_protocol_comparison"]["closed_set"])
    assert set(NC.PROTOCOL_AGGREGATION) == closed
    agg = reg["benchmarks"]["navsim"]["GATE_estimator_cluster_unit"]["aggregation_by_protocol"]
    assert agg == NC.PROTOCOL_AGGREGATION
