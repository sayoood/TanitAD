"""``tools/t1_eval.py`` — the promoted T1 (action-closed-loop) eval, pinned on CPU.

The GPU roll cannot run here (no CUDA, no checkpoint); what CAN be pinned is the
ANALYSIS half — the exact arithmetic the §1.12 byte-close gate depends on —
on synthetic dumps that mimic ``closed_loop_dump.py``'s npz schema:

    g [N,K,2] GT, per-arm [N,K,2] paths, ws [N], optional <arm>_fan_err /
    <arm>_sel_idx / <arm>_fan_scores fan surfaces.

Every hand-computed expectation below is derived in a comment beside it, so a
failure localises to the port rather than to the test's own arithmetic.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "t1_eval.py"
_spec = importlib.util.spec_from_file_location("t1_eval", TOOL)
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)

DT = t1.DT           # 0.1
N_BOOT = 25          # tiny — the tests pin arithmetic, not interval width


# --------------------------------------------------------------------------- #
# synthetic path builders (the dump's waypoint convention: origin-prepended)   #
# --------------------------------------------------------------------------- #
def _path(headings, lengths):
    """Waypoints whose per-step displacement j has heading[j] and length[j].

    ``t1.controls`` prepends the origin, so displacement j of the recovered
    path is exactly ``lengths[j] * (cos h[j], sin h[j])`` — the constructed
    dh/ds are recovered without approximation.
    """
    h = np.asarray(headings, float)
    ls = np.asarray(lengths, float)
    return np.cumsum(np.stack([np.cos(h), np.sin(h)], -1) * ls[:, None], 0)


def s_path(sign=1.0):
    """A true S: dh = sign*(+0.01 x9, then -0.011 x10), constant speed 1 m/s.

    h1 = sign*+0.09 (> THR=0.03), h2 = sign*-0.11 -> opposite signs, both
    above THR -> a GT S-window under BOTH the masked and unmasked definitions
    (every step moves 0.1 m > the 0.05 m stationarity gate).
    """
    dh = np.array([0.01] * 9 + [-0.011] * 10) * sign
    h = np.concatenate([[0.0], np.cumsum(dh)])          # 20 step headings
    return _path(h, [0.1] * 20)


def straight_path():
    return _path([0.0] * 20, [0.1] * 20)


def half_stationary_s_path():
    """S under the UNMASKED definition only.

    First 10 steps move 0.1 m turning +0.01 rad/step -> h1 = +0.09 both defs.
    Last 10 steps move 0.01 m (< 0.05 gate) turning -0.05 rad/step ->
    unmasked h2 = -0.5 (S!), masked h2 = 0 (not S). This is exactly the
    divergence between analyze_cl.py's inline S-def and the banked
    s_curve_dump.py definition (112 vs 93 S-windows in §1.12).
    """
    h = [0.01 * j for j in range(10)] + \
        [0.09 - 0.05 * (j - 9) for j in range(10, 20)]
    return _path(h, [0.1] * 10 + [0.01] * 10)


def accel_path(a, v0=5.0):
    """Straight x-only path with constant accel ``a``: near-accel == a exactly."""
    v = v0 + a * DT * np.arange(20)
    x = np.cumsum(v * DT)
    return np.stack([x, np.zeros(20)], -1)


# --------------------------------------------------------------------------- #
# dumps                                                                        #
# --------------------------------------------------------------------------- #
def s_dump(tmp_path, with_fan=False, extra_arm=None):
    """5 windows: [S(hit), S(miss), straight, stationary, unmasked-only-S].

    GT S-windows: masked n_s = 2 (w0, w1); unmasked n_s = 3 (+ w4).
    cl hits: w0 yes (identical path), w1 no (straight), w4 no (straight)
      -> masked rate 1/2, unmasked rate 1/3.
    ol == g -> every S reproduced: masked 2/2, unmasked 3/3.
    """
    g = np.stack([s_path(), s_path(-1.0), straight_path(),
                  np.zeros((20, 2)), half_stationary_s_path()])
    cl = np.stack([s_path(), straight_path(), straight_path(),
                   np.zeros((20, 2)), straight_path()])
    d = {"g": g, "cl": cl, "ol": g.copy(), "ws": np.arange(5)}
    if with_fan:
        d["cl_fan_err"] = np.array([[0.5, 0.2], [0.3, 0.1], [0.4, 0.4],
                                    [0.2, 0.6], [0.1, 0.3]])
        d["cl_sel_idx"] = np.array([0, 1, 0, 1, 0])
    if extra_arm:
        d[extra_arm] = cl.copy()
    p = tmp_path / "dump"
    p.mkdir(exist_ok=True)
    np.savez(p / "ep000.npz", **d)
    return sorted(str(f) for f in p.glob("ep*.npz"))


def lag_dump(tmp_path):
    """One episode, 46 windows. GT accel impulse (2 m/s^2) at window 10, arm's
    at window 12 -> xcorr peak at +2 windows = +0.2 s (arm responds late)."""
    n = 46
    g = np.stack([accel_path(2.0 if w == 10 else 0.0) for w in range(n)])
    cl = np.stack([accel_path(2.0 if w == 12 else 0.0) for w in range(n)])
    p = tmp_path / "dump"
    p.mkdir(exist_ok=True)
    np.savez(p / "ep000.npz", g=g, cl=cl, ws=np.arange(n))
    return sorted(str(f) for f in p.glob("ep*.npz"))


# --------------------------------------------------------------------------- #
# S-rate: hand-computed, both definitions                                      #
# --------------------------------------------------------------------------- #
def test_s_rate_matches_hand_computed_masked_and_unmasked(tmp_path):
    res = t1.analyze(s_dump(tmp_path), n_boot=N_BOOT)
    cl = res["arms"]["cl"]["s_curve"]
    # masked (the BANKED §1.12 definition): 2 S-windows, cl hits 1 -> 0.5
    assert cl["masked"]["n_s_windows"] == 2
    assert cl["masked"]["rate"] == pytest.approx(0.5)
    # unmasked (analyze_cl inline): +1 S from the half-stationary window -> 1/3
    assert cl["unmasked"]["n_s_windows"] == 3
    assert cl["unmasked"]["rate"] == pytest.approx(round(1 / 3, 4))
    # the teacher-forced arm reproduces its own conditioning: rate 1.0
    ol = res["arms"]["ol"]["s_curve"]
    assert ol["masked"]["rate"] == pytest.approx(1.0)
    assert ol["unmasked"]["rate"] == pytest.approx(1.0)
    # the legacy §1.12 blocks carry the same numbers
    assert res["s_curve_masked_legacy"]["n_s"] == 2
    assert res["s_curve_masked_legacy"]["cl"] == pytest.approx(0.5)
    assert res["arms"]["cl"]["legacy_epmean_row"]["s_reproduction_rate"] == \
        pytest.approx(round(1 / 3, 4))          # analyze_cl's row is UNMASKED
    # the CI travels with the rate and names the decision-grade estimator
    assert cl["masked"]["ci"]["estimator"] == "episode_cluster_bootstrap"


def test_masked_and_unmasked_definitions_actually_diverge(tmp_path):
    """The half-stationary window is the §1.12 112-vs-93 divergence in
    miniature; if both defs count it, the mask port is broken."""
    res = t1.analyze(s_dump(tmp_path), n_boot=N_BOOT)
    cl = res["arms"]["cl"]["s_curve"]
    assert cl["unmasked"]["n_s_windows"] == cl["masked"]["n_s_windows"] + 1


# --------------------------------------------------------------------------- #
# lag: hand-computed                                                           #
# --------------------------------------------------------------------------- #
def test_lag_matches_hand_computed_value(tmp_path):
    res = t1.analyze(lag_dump(tmp_path), n_boot=N_BOOT)
    lag = res["arms"]["cl"]["lag"]
    assert lag["lag_accel_s_mean"] == pytest.approx(0.2)   # +2 windows * 0.1 s
    assert lag["n_episodes_with_signal"] == 1
    assert "tier" in lag and "estimator" in lag and "n" in lag


def test_xcorr_lag_unit():
    """The primitive itself, isolated: impulse shifted by +2 -> +0.2 s."""
    a_g = np.zeros(46)
    a_g[10] = 2.0
    a_p = np.zeros(46)
    a_p[12] = 2.0
    assert t1.xcorr_lag(a_p, a_g) == pytest.approx(0.2)
    # no signal -> None, never a fabricated 0-lag
    assert t1.xcorr_lag(np.zeros(46), np.zeros(46)) is None


def test_response_ratio_and_event_counts(tmp_path):
    """GT accel event at w10 (2 > 1 m/s^2); the arm's near-accel THERE is 0 ->
    ratio 0.0 on n=1 event. No decel events -> UNAVAILABLE with reason + n."""
    res = t1.analyze(lag_dump(tmp_path), n_boot=N_BOOT)
    resp = res["arms"]["cl"]["response"]
    assert resp["accel"]["response_ratio"] == pytest.approx(0.0)
    assert resp["accel"]["n_events"] == 1
    assert resp["decel"]["status"] == "UNAVAILABLE"
    assert resp["decel"]["n"] == 0 and "reason" in resp["decel"]


# --------------------------------------------------------------------------- #
# tier stamps: every emitted block                                             #
# --------------------------------------------------------------------------- #
def test_tier_stamp_present_on_every_family_and_block(tmp_path):
    res = t1.analyze(s_dump(tmp_path), n_boot=N_BOOT)
    for arm, want in (("cl", "T1"), ("ol", "T0")):
        blk = res["arms"][arm]
        assert blk["tier"] == want
        for fk in ("longitudinal", "lateral", "tactical", "strategic"):
            fam = blk["four_families"][fk]
            assert fam["tier"] == want, (arm, fk)
            assert "estimator" in fam and "n" in fam, (arm, fk)
        for bk in ("intervals", "s_curve", "lag", "response", "sel_gap"):
            assert blk[bk].get("tier") == want, (arm, bk)
    # T0 is labelled as NOT quotable as driving; T1 as the primary tier
    assert "NEVER driving performance" in res["arms"]["ol"]["tier_note"]
    assert "PRIMARY" in res["arms"]["cl"]["tier_note"]
    assert "T1" in res["_tier_doctrine"] and "T0" in res["_tier_doctrine"]


def test_an_arm_without_a_tier_is_a_hard_error(tmp_path):
    """An un-tiered number is exactly what EVAL_DOCTRINE forbids — the tool
    must refuse, not guess."""
    files = s_dump(tmp_path, extra_arm="zz")
    with pytest.raises(ValueError, match="tier"):
        t1.analyze(files, n_boot=N_BOOT)
    # ... and an explicit stamp unblocks it
    res = t1.analyze(files, tiers={"zz": "T1"}, n_boot=N_BOOT)
    assert res["arms"]["zz"]["tier"] == "T1"


# --------------------------------------------------------------------------- #
# absence reporting: reason + n, never silent                                  #
# --------------------------------------------------------------------------- #
def test_families_without_inputs_report_unavailable_with_reason_and_n(tmp_path):
    res = t1.analyze(s_dump(tmp_path), n_boot=N_BOOT)
    blk = res["arms"]["cl"]
    # a T1 dump traverses no decision heads -> TACTICAL/STRATEGIC UNAVAILABLE
    for fk in ("tactical", "strategic"):
        fam = blk["four_families"][fk]
        assert fam["status"] == "UNAVAILABLE"
        assert fam["reason"] and fam["n"] == 0
    assert set(blk["four_families"]["_families_unavailable"]) == \
        {"tactical", "strategic"}
    # no lead block -> the distance-keeping HALF of LONGITUDINAL is a work item
    dk = blk["four_families"]["longitudinal"]["distance_keeping"]
    assert dk["status"] == "UNAVAILABLE" and "WORK ITEM" in dk["reason"]
    # no fan surface in the dump -> sel_gap absent WITH the reason
    assert blk["sel_gap"]["status"] == "UNAVAILABLE"
    assert "fan" in blk["sel_gap"]["reason"] and blk["sel_gap"]["n"] == 0


def test_no_s_windows_reports_unavailable_not_a_rate(tmp_path):
    """A corpus with no S-reversals has NO denominator — reporting 0.0 (or
    1.0) would fabricate the §1.12 headline metric."""
    res = t1.analyze(lag_dump(tmp_path), n_boot=N_BOOT)   # straight paths only
    s = res["arms"]["cl"]["s_curve"]
    for name in ("masked", "unmasked"):
        assert s[name]["status"] == "UNAVAILABLE"
        assert "n_s = 0" in s[name]["reason"] and s[name]["n"] == 0


def test_no_lag_signal_reports_unavailable(tmp_path):
    res = t1.analyze(s_dump(tmp_path), n_boot=N_BOOT)     # constant speed 1 m/s
    lag = res["arms"]["cl"]["lag"]
    assert lag["status"] == "UNAVAILABLE" and "reason" in lag


# --------------------------------------------------------------------------- #
# selgap hook                                                                  #
# --------------------------------------------------------------------------- #
def test_selgap_hook_fires_when_fan_surface_present(tmp_path):
    res = t1.analyze(s_dump(tmp_path, with_fan=True), n_boot=N_BOOT)
    sg = res["arms"]["cl"]["sel_gap"]
    # produced by taniteval.selgap itself, not a local re-implementation
    assert sg["block"] == "taniteval.selgap"
    assert sg["gap_ci"]["estimator"] == "episode_cluster_bootstrap"
    # hand: selected = mean(0.5, 0.1, 0.4, 0.6, 0.1) = 0.34
    #       oracle   = mean(0.2, 0.1, 0.4, 0.2, 0.1) = 0.20 -> gap 0.14
    assert sg["selected"] == pytest.approx(0.34)
    assert sg["oracle"] == pytest.approx(0.20)
    assert sg["gap"] == pytest.approx(0.14)
    assert sg["n"] == 5 and sg["tier"] == "T1"
    # the arm WITHOUT a fan surface still reports absence, not silence
    assert res["arms"]["ol"]["sel_gap"]["status"] == "UNAVAILABLE"


# --------------------------------------------------------------------------- #
# four families + paired + legacy reproduction arithmetic                      #
# --------------------------------------------------------------------------- #
def test_four_families_come_from_taniteval_machinery(tmp_path):
    """The identical-to-GT arm scores 0 error — computed through
    four_families.all_families, not a private copy."""
    res = t1.analyze(s_dump(tmp_path), n_boot=N_BOOT)
    lon = res["arms"]["ol"]["four_families"]["longitudinal"]
    assert lon["speed_mae_mps"] == pytest.approx(0.0, abs=1e-4)
    assert res["arms"]["ol"]["legacy_epmean_row"]["ade_m"] == pytest.approx(0.0)
    ints = res["arms"]["cl"]["intervals"]["metrics"]
    for k in ("ade_dense_m", "LON_speed_mae_mps", "LAT_cross_mae_m"):
        assert ints[k]["estimator"] == "episode_cluster_bootstrap", k


def test_default_pair_ol_cl_and_both_paired_estimators_emitted(tmp_path):
    res = t1.analyze(s_dump(tmp_path), n_boot=N_BOOT)
    name = "paired_closed_minus_open"
    leg = res["paired_legacy"][name]
    dec = res["paired_decision_grade"][name]
    assert "_reproduction_of" in leg          # analyze_cl.boot port, marked
    assert dec["estimator"] == "paired_episode_cluster_bootstrap"
    assert dec["direction"] == "cl - ol"
    assert dec["ade_m"]["estimator"] == "paired_episode_cluster_bootstrap"
    # cl is worse than the GT-echo ol on these windows -> positive ade delta
    assert dec["ade_m"]["delta"] > 0


def test_legacy_row_is_mean_of_episode_means_and_says_so(tmp_path):
    """Two episodes with different window counts: the legacy row must equal the
    UNWEIGHTED mean of episode means (analyze_cl arithmetic), which differs
    from the pooled mean — the exact bias the headline row avoids."""
    p = tmp_path / "dump"
    p.mkdir()
    g1 = np.stack([straight_path()] * 3)
    cl1 = g1 + np.array([1.0, 0.0])           # ep0: ade 1.0, 3 windows
    g2 = np.stack([straight_path()] * 1)
    cl2 = g2 + np.array([3.0, 0.0])           # ep1: ade 3.0, 1 window
    np.savez(p / "ep000.npz", g=g1, cl=cl1)
    np.savez(p / "ep001.npz", g=g2, cl=cl2)
    res = t1.analyze(sorted(str(f) for f in p.glob("ep*.npz")), n_boot=N_BOOT)
    row = res["arms"]["cl"]["legacy_epmean_row"]
    assert row["ade_m"] == pytest.approx(2.0)                 # (1+3)/2
    assert "mean-of-EPISODE-means" in row["_reproduction_of"]
    pooled = res["arms"]["cl"]["intervals"]["metrics"]["ade_dense_m"]["mean"]
    assert pooled == pytest.approx(1.5)                       # (3*1 + 1*3)/4


def test_byte_check_reports_max_abs_against_a_reference_dump(tmp_path):
    files = s_dump(tmp_path)
    ref = tmp_path / "ref"
    ref.mkdir()
    with np.load(files[0]) as d:
        np.savez(ref / "ep000.npz", b=d["cl"] + 0.25)
    res = t1.analyze(files, n_boot=N_BOOT,
                     byte_check=("cl", sorted(str(f) for f in ref.glob("*.npz")),
                                 "b"))
    assert res["byte_check"]["max_abs"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# the CLI surface                                                              #
# --------------------------------------------------------------------------- #
def test_cli_help_works_without_a_gpu():
    r = subprocess.run([sys.executable, str(TOOL), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    for flag in ("--ckpt", "--corpus", "--analyze-only", "--with-t0-open-loop",
                 "--tiers", "--head", "--wheelbase"):
        assert flag in r.stdout, flag


def test_t0_arm_is_off_by_default_and_labelled():
    """The teacher-forced arm must be opt-in and its help text must say what
    T0 may NOT be quoted as."""
    src = TOOL.read_text(encoding="utf-8")
    i = src.index('"--with-t0-open-loop"')       # the add_argument, not the doc
    seg = src[i:i + 500]
    assert "OFF by " in seg and "NEVER" in seg


def test_cli_analyze_only_end_to_end(tmp_path):
    files = s_dump(tmp_path)
    out = tmp_path / "t1.json"
    r = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "synthetic",
         "--analyze-only", str(Path(files[0]).parent),
         "--out", str(out), "--n-boot", "25"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    rec = json.loads(out.read_text())
    assert rec["mode"] == "analyze-only"
    assert rec["arms"]["cl"]["tier"] == "T1"
    assert rec["arms"]["cl"]["s_curve"]["masked"]["rate"] == pytest.approx(0.5)
