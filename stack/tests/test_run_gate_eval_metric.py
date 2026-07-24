"""``run_gate`` — the 3 v4 gate bugs the G1-panel-on-v1 de-risk found
(``taniteval/results/v1_g1_dryrun_gate.json``), each pinned so it cannot drift
back.

  Fix 1 ⭐  the eval reader must PREFER the episode-cluster bootstrap and FAIL
            LOUD when only the deprecated ``overlapping_holdout_se`` interval
            exists — the old code silently adjudicated on the forbidden
            statistic (1.28-2.06x too narrow, CLAUDE.md).
  Fix 2     the 3-way miss-name drift (``miss_at_2m`` / ``miss_2m`` /
            ``miss_rate@2m``) must all resolve to one another.
  Fix 3     report-only ``--secondary-value``s (off the card) must be emitted
            into the gate JSON, read, and NOT adjudicate the verdict (§9 split
            card).

Pure-Python: constructed eval dicts, no torch / no window load — the
recomputation-matches-CI_RECOMPUTE pin lives in
``taniteval/tests/test_driving_gate_block.py`` where the estimator + windows are.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gate as rg                                              # noqa: E402

# flagship-30k's published episode-cluster-bootstrap ADE@2s CI, the value the
# gate must surface (Project Steering/CI_RECOMPUTE_2026-07-20.json: flagship-30k
# boot_lo 0.3675 / boot_hi 0.4871 around the full-set mean 0.4271).
CI_RECOMPUTE_ADE = (0.4271, 0.3675, 0.4871)


def _cb_node(mean, lo, hi):
    return {"mean": mean, "lo": lo, "hi": hi, "ci95": round((hi - lo) / 2, 4),
            "estimator": "episode_cluster_bootstrap", "n_episodes": 40,
            "n_boot": 2000}


def _dep_node(mean, ci95):
    return {"mean": mean, "ci95": ci95, "estimator": "overlapping_holdout_se",
            "deprecated": True}


# =========================================================================== #
# Fix 1 — prefer cluster_bootstrap, fail loud on deprecated-only              #
# =========================================================================== #
def test_prefers_cluster_bootstrap_over_deprecated_and_surfaces_the_ci():
    m, lo, hi = CI_RECOMPUTE_ADE
    ev = {"cluster_bootstrap": {"model": {"ade_0_2s": _cb_node(m, lo, hi)}},
          # deprecated block also present — must be ignored, not chosen
          "heldout": {"model": {"ade_0_2s": _dep_node(0.4522, 0.0312)}}}
    val, prov = rg._read_eval_metric(ev, "ade_0_2s")
    assert val == 0.4271, "must take the cluster-bootstrap (full-set) mean, not 0.4522"
    assert "episode_cluster_bootstrap" in prov
    assert f"CI [{lo}, {hi}]" in prov, prov


def test_fails_loud_when_only_the_deprecated_block_exists():
    """The ⭐ bug: the old code silently returned the heldout block."""
    ev = {"full_set": {"model": {"ade_0_2s": 0.4271}},
          "heldout": {"model": {"ade_0_2s": _dep_node(0.4522, 0.0312)}}}
    with pytest.raises(SystemExit) as ei:
        rg._read_eval_metric(ev, "ade_0_2s")
    msg = str(ei.value)
    assert "DEPRECATED" in msg and "overlapping_holdout_se" in msg
    assert "cluster_bootstrap" in msg          # tells the caller how to fix it


def test_driving_headline_counts_as_cluster_bootstrap():
    """A raw ``driving_<key>.json`` (headline only) is gate-readable, because its
    headline intervals ARE the episode-cluster bootstrap."""
    m, lo, hi = CI_RECOMPUTE_ADE
    ev = {"headline": {"ade_0_2s": _cb_node(m, lo, hi)}}
    val, prov = rg._read_eval_metric(ev, "ade_0_2s")
    assert val == 0.4271 and "headline" in prov


def test_merged_driving_block_is_gate_readable():
    """The ``<key>.json`` that ``driving.run_and_save`` merges under 'driving'."""
    m, lo, hi = CI_RECOMPUTE_ADE
    ev = {"driving": {"cluster_bootstrap": {"model": {"ade_0_2s": _cb_node(m, lo, hi)}}}}
    val, prov = rg._read_eval_metric(ev, "ade_0_2s")
    assert val == 0.4271 and "driving.cluster_bootstrap" in prov


def test_point_estimate_fallback_only_when_no_deprecated_block():
    """No interval anywhere and no forbidden estimator -> a full_set POINT is an
    admissible value, explicitly labelled as carrying no interval."""
    ev = {"full_set": {"model": {"ade_0_2s": 0.4271}}}
    val, prov = rg._read_eval_metric(ev, "ade_0_2s")
    assert val == 0.4271
    assert "point estimate" in prov and "NO interval" in prov


# =========================================================================== #
# Fix 2 — the 3-way miss-name drift                                           #
# =========================================================================== #
def test_miss_name_aliases_are_symmetric():
    grp = ("miss_at_2m", "miss_2m", "miss_rate@2m")
    for name in grp:
        assert set(rg._metric_aliases(name)) == set(grp)
    assert rg._metric_aliases("ade_0_2s") == ("ade_0_2s",)   # non-aliased is itself


def test_card_miss_at_2m_resolves_to_driving_miss_2m():
    ev = {"cluster_bootstrap": {"model": {"miss_2m": _cb_node(0.0454, 0.0239, 0.0681)}}}
    val, prov = rg._read_eval_metric(ev, "miss_at_2m")
    assert val == 0.0454 and "as 'miss_2m'" in prov


def test_card_miss_at_2m_resolves_to_bench_miss_rate_at_2m():
    ev = {"cluster_bootstrap": {"model": {"miss_rate@2m": _cb_node(0.0602, 0.0481, 0.0723)}}}
    val, prov = rg._read_eval_metric(ev, "miss_at_2m")
    assert val == 0.0602 and "as 'miss_rate@2m'" in prov


def test_secondary_value_supplied_under_any_alias_satisfies_the_card():
    val, key = rg._lookup_secondary_value({"miss_2m": 0.06}, "miss_at_2m")
    assert val == 0.06 and key == "miss_2m"
    val, key = rg._lookup_secondary_value({"miss_at_2m": 0.06}, "miss_at_2m")
    assert val == 0.06 and key == "miss_at_2m"
    assert rg._lookup_secondary_value({"other": 1.0}, "miss_at_2m") == (None, None)


# =========================================================================== #
# Fix 3 — report-only secondaries + the 8/5 split, end to end through check   #
# =========================================================================== #
def _write_card(path, secondary):
    card = {"run": "t", "gate_step": 10000, "primary_metric": "ade_0_2s",
            "primary_threshold": 0.60, "primary_direction": "<=",
            "primary_source": "held-out full-set", "secondary": secondary,
            "reference_run": None, "reference_log": None, "compare_metric": None,
            "tau": 1.5, "lever_family": "joint-planner-wm", "restarts_used": 0,
            "restart_cap": 2, "registered_utc": "2026-07-22T00:00:00+00:00",
            "note": ""}
    Path(path).write_text(json.dumps(card))


def _run_check(tmp_path, ev, secondary_values, secondary=None):
    if secondary is None:
        secondary = ["miss_at_2m<=0.10", "wm_canary_ade_2s<=0.55"]
    card = tmp_path / "card.json"
    _write_card(card, secondary)
    m, lo, hi = CI_RECOMPUTE_ADE
    (tmp_path / "eval.json").write_text(json.dumps(ev))
    (tmp_path / "log.jsonl").write_text(
        '{"step": 0, "step_s": 0, "g_op_fwd_ade_m": 1.0}\n'
        '{"step": 10000, "step_s": 100, "g_op_fwd_ade_m": 0.5}\n')
    out = tmp_path / "gate.json"
    rg.main(["check", "--card", str(card), "--log", str(tmp_path / "log.jsonl"),
             "--eval-json", str(tmp_path / "eval.json"),
             "--secondary-value", *secondary_values, "--json", str(out)])
    return json.loads(out.read_text())


def _fresh_ev():
    m, lo, hi = CI_RECOMPUTE_ADE
    return {"cluster_bootstrap": {"model": {
        "ade_0_2s": _cb_node(m, lo, hi),
        "miss_2m": _cb_node(0.0454, 0.0239, 0.0681)}}}


def test_report_only_values_are_emitted_read_and_not_adjudicated(tmp_path):
    g = _run_check(tmp_path, _fresh_ev(), [
        "miss_at_2m=0.06", "wm_canary_ade_2s=0.45",         # 2 KILL (on card)
        "imag_win_at_5s=1", "strat_subspace_sufficiency=0.92",
        "longh_5s_beats_persistence=1", "cruise_delta_vs_holdv0=-0.2122"])  # 4 REPORT
    report = {r["metric"] for r in g["report_only"]}
    assert report == {"imag_win_at_5s", "strat_subspace_sufficiency",
                      "longh_5s_beats_persistence", "cruise_delta_vs_holdv0"}
    assert all(r["adjudicated"] is False for r in g["report_only"])
    # exactly the 2 card secondaries adjudicate; report-only never enters that set
    assert {r["metric"] for r in g["secondary"]} == {"miss_at_2m", "wm_canary_ade_2s"}
    assert g["verdict"] == "CONTINUE"


def test_a_failing_report_only_value_can_never_kill_the_run(tmp_path):
    """The trapdoor §9 warns about: a REPORT-ONLY falsifier at a KILL value must
    NOT flip the verdict."""
    g = _run_check(tmp_path, _fresh_ev(), [
        "miss_at_2m=0.06", "wm_canary_ade_2s=0.45",
        "imag_win_at_5s=0",                    # 'failed' — but report-only
        "cruise_delta_vs_holdv0=-9.9"])        # catastrophic — but report-only
    assert g["verdict"] == "CONTINUE", (g["verdict"], g["reason"])
    assert {r["metric"] for r in g["report_only"]} == {"imag_win_at_5s",
                                                       "cruise_delta_vs_holdv0"}


def test_a_failing_KILL_secondary_still_restarts(tmp_path):
    """The KILL path is unchanged: a real on-card failure still yields RESTART,
    and report-only rows sitting alongside it do not mask it."""
    g = _run_check(tmp_path, _fresh_ev(), [
        "miss_at_2m=0.99",                     # KILL, FAILS (>0.10)
        "wm_canary_ade_2s=0.45",
        "imag_win_at_5s=1"])                   # report-only, passes — irrelevant
    assert g["verdict"] == "RESTART", (g["verdict"], g["reason"])


def test_miss_supplied_as_miss_2m_is_not_dropped_into_report_only(tmp_path):
    """Fix 2 x Fix 3: a card 'miss_at_2m' fed as 'miss_2m' must adjudicate, not
    vanish into the report-only channel."""
    g = _run_check(tmp_path, _fresh_ev(),
                   ["miss_2m=0.06", "wm_canary_ade_2s=0.45"])
    row = [r for r in g["secondary"] if r["metric"] == "miss_at_2m"][0]
    assert row["pass"] is True and row["supplied_as"] == "miss_2m"
    assert g["report_only"] == []          # nothing dropped


def test_check_fails_loud_on_a_stale_deprecated_only_eval_json(tmp_path):
    stale = {"full_set": {"model": {"ade_0_2s": 0.4271}},
             "heldout": {"model": {"ade_0_2s": _dep_node(0.4522, 0.0312)}}}
    with pytest.raises(SystemExit) as ei:
        _run_check(tmp_path, stale, ["miss_at_2m=0.06", "wm_canary_ade_2s=0.45"])
    assert "overlapping_holdout_se" in str(ei.value)
