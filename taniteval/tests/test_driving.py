"""Tests for ``taniteval/driving.py`` — the TanitEval v2 tier-0 driving panel.

These are the tests that keep a driving-capability claim HONEST. The failure
mode this panel invites is not a crash, it is a *plausible wrong verdict*:

  * a metric silently computed with the deprecated ``overlapping_holdout_se``
    (measured 1.28-2.06x too narrow) — which would manufacture separations;
  * a paired test rendered "separated" when the interval actually favours the
    trivial FLOOR (six of our arms are CI-separated *against themselves* on
    speed MAE — a sep/tie rendering would have printed those as wins);
  * a geometry "cleanup" that quietly moves the along/cross split, breaking the
    only thing anchoring this module to MODEL_REGISTRY §1.2.

Each of those has a test here, plus the registry sanity pin itself.

CPU-only by design, no GPU and no model load: the whole suite runs against the
committed ``results/windows_*.pt`` dumps.

pytest is NOT installed on the eval pod, so these run standalone too:
  python taniteval/tests/test_driving.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, "/root/taniteval")              # pod layout

import numpy as np  # noqa: E402
import torch  # noqa: E402

from taniteval import ci as C  # noqa: E402
from taniteval import driving as D  # noqa: E402

RESULTS = _HERE.parents[1] / "results"
PIN_ARM = RESULTS / f"windows_{D.SANITY_ARM}.pt"


# --------------------------------------------------------------------------- #
# the estimator contract — the reason this module exists at all                 #
# --------------------------------------------------------------------------- #
def test_only_two_estimators_are_decision_grade():
    assert D.DECISION_ESTIMATORS == {"episode_cluster_bootstrap",
                                     "paired_episode_cluster_bootstrap"}
    assert D.DEPRECATED_ESTIMATOR == "overlapping_holdout_se"
    assert D.DEPRECATED_ESTIMATOR not in D.DECISION_ESTIMATORS


def test_deprecated_interval_can_never_be_returned_as_a_decision_number():
    """The whole point. `overlapping_holdout_se` is 1.28-2.06x too narrow; a
    driving verdict computed with it would be a FALSE separation."""
    bad = {"headline": {"ade_0_2s": {
        "mean": 0.4522, "lo": 0.421, "hi": 0.483,
        "estimator": "overlapping_holdout_se"}}}
    try:
        D.assert_no_deprecated_estimator(bad)
    except ValueError as e:
        assert "overlapping_holdout_se" in str(e)
        assert "headline.ade_0_2s" in str(e)
    else:
        raise AssertionError("a deprecated-estimator interval was accepted")


def test_interval_without_a_named_estimator_is_refused():
    """An unlabelled interval is how the deprecated one survived for months."""
    try:
        D.assert_no_deprecated_estimator({"x": {"mean": 1.0, "lo": 0.9,
                                                "hi": 1.1}})
    except ValueError as e:
        assert "without a named estimator" in str(e)
    else:
        raise AssertionError("an interval with no estimator was accepted")


def test_every_emitted_interval_names_its_estimator():
    """Walks a REAL block, not a fixture — the guard must hold end to end."""
    out = _pin_block()
    D.assert_no_deprecated_estimator(out)          # would raise
    for k, v in out["headline"].items():
        assert v["estimator"] == "episode_cluster_bootstrap", k
        assert v["n_episodes"] == 40 and v["n_boot"] > 0, k
    for floor in D.FLOORS:
        for k, v in out["vs_floor_paired"][floor].items():
            assert v["estimator"] == "paired_episode_cluster_bootstrap", (floor, k)


# --------------------------------------------------------------------------- #
# the MODEL_REGISTRY sanity pin                                                 #
# --------------------------------------------------------------------------- #
_PIN_CACHE = {}


def _pin_block(n_boot=300):
    if not PIN_ARM.exists():
        raise AssertionError(
            f"{PIN_ARM} missing — the registry pin cannot run. This dump is "
            f"committed; if it is gone the tier-0 suite is unverifiable.")
    if n_boot not in _PIN_CACHE:
        _PIN_CACHE[n_boot] = D.from_windows(PIN_ARM, n_boot=n_boot,
                                            arm=D.SANITY_ARM)
    return _PIN_CACHE[n_boot]


def test_registry_sanity_pin_is_green():
    """Reproduces MODEL_REGISTRY §1.2 to 4 decimals on the committed dump:
    ADE 0.4271, CV 0.8377, long/lat RMSE 1.042/0.360, top-decile speed bias
    +0.659, top-decile long-RMSE 1.379. If these drift, the artifact or the
    geometry convention changed — fail loud rather than publish."""
    out = _pin_block()
    assert out["sanity_all_ok"] is True, out["sanity_vs_registry"]
    for k, v in out["sanity_vs_registry"].items():
        assert abs(v["got"] - v["expected"]) <= D.SANITY_TOL, (k, v)


def test_pin_reproduces_the_published_along_cross_split():
    """The suite's headline finding, and the reason a single ADE column is not
    admissible: flagship v1's CI-separated win over CV is ENTIRELY lateral."""
    out = _pin_block(n_boot=2000)
    v = out["verdict"]
    assert v["ade_vs_cv"]["separated"] and v["ade_vs_cv"]["favours"] == "model"
    assert abs(v["ade_vs_cv"]["delta"] - 0.4106) <= 0.002
    assert abs(v["cross_track_vs_cv"]["delta"] - 0.7720) <= 0.002
    assert v["cross_track_vs_cv"]["separated"], "cross-track must separate"
    assert abs(v["along_track_vs_cv"]["delta"] - 0.2543) <= 0.002
    assert not v["along_track_vs_cv"]["separated"], \
        "along-track must NOT separate — that is the finding"
    assert abs(v["speed_mae_vs_cv"]["delta"] - (-0.0032)) <= 0.002
    assert not v["speed_mae_vs_cv"]["separated"]
    assert v["where_the_win_lives"] == "lateral only"
    assert v["tracks_speed_better_than_cv"] is False


def test_pin_reproduces_the_cruise_inversion():
    """L1 vs L2 point in OPPOSITE directions for the same checkpoint — the
    decomposition ADE destroys. On the 639 steady windows the deployed arm is
    2.0x worse than doing nothing; on brake/accel it wins decisively."""
    reg = _pin_block(n_boot=2000)["longitudinal_regime"]
    assert reg["steady"]["n"] == 639
    assert abs(reg["steady"]["model_speed_mae"] - 0.4231) <= 0.002
    assert abs(reg["steady"]["holdv0_speed_mae"] - 0.2109) <= 0.002
    d = reg["steady"]["vs_holdv0_speed_mae_paired"]
    assert d["separated"] and d["favours"] == "floor", \
        "the cruise result is separated AGAINST the model; it must render as a loss"
    assert abs(d["delta"] - (-0.2122)) <= 0.003
    for r, exp in (("brake", 0.6433), ("accel", 0.5716)):
        dd = reg[r]["vs_holdv0_speed_mae_paired"]
        assert dd["separated"] and dd["favours"] == "model", r
        assert abs(dd["delta"] - exp) <= 0.003, r


def test_pin_reproduces_the_straight_line_heading_failure():
    """T3: 5.7x worse than a straight line at going straight, while 7.5x
    BETTER than CV on sharp curves. Invisible in ADE."""
    cb = _pin_block()["by_curvature"]
    assert cb["straight"]["n"] == 634 and cb["gentle"]["n"] == 103 \
        and cb["sharp"]["n"] == 144
    assert abs(cb["straight"]["model_heading_mae_deg"] - 7.980) <= 0.01
    assert abs(cb["straight"]["cv_heading_mae_deg"] - 1.399) <= 0.01
    assert abs(cb["sharp"]["model_heading_mae_deg"] - 3.811) <= 0.01
    assert abs(cb["sharp"]["cv_heading_mae_deg"] - 28.743) <= 0.01


# --------------------------------------------------------------------------- #
# geometry — the conventions the registry pin is a pin ON                       #
# --------------------------------------------------------------------------- #
def test_frenet_basis_is_orthonormal():
    """along^2 + cross^2 == ||pred-gt||^2 exactly. If this breaks, every
    longitudinal/lateral attribution in the program is wrong."""
    g = torch.tensor([[[1., 0.], [2., .3], [3., .9], [4., 1.8]]])
    p = g + torch.tensor([[[.1, -.2], [.3, .1], [-.2, .4], [.5, .5]]])
    al, cr = D.frenet(p, g)
    de = torch.linalg.norm(p - g, dim=-1)
    assert torch.allclose(al ** 2 + cr ** 2, de ** 2, atol=1e-5)


def test_frenet_sign_convention_ahead_is_positive_left_is_positive():
    g = torch.tensor([[[5., 0.], [10., 0.], [15., 0.], [20., 0.]]])
    ahead = g + torch.tensor([[1., 0.]])
    left = g + torch.tensor([[0., 1.]])
    assert D.frenet(ahead, g)[0][0, -1] > 0.99
    assert D.frenet(left, g)[1][0, -1] > 0.99


def test_path_geometry_is_speed_decoupled():
    """T2's whole claim: two paths tracing the SAME geometry at DIFFERENT
    speeds score ~0, so a lateral error is attributable to shape, not speed."""
    gt = torch.tensor([[[4., 0.], [8., 0.], [12., 0.], [16., 0.]]])
    slow = torch.tensor([[[2., 0.], [4., 0.], [6., 0.], [8., 0.]]])
    assert float(D.path_geometry_crosstrack(slow, gt)) < 1e-4
    # ... while a genuinely different SHAPE at the same speed does not.
    bent = torch.tensor([[[4., 1.], [8., 2.], [12., 3.], [16., 4.]]])
    assert float(D.path_geometry_crosstrack(bent, gt)) > 0.5


def test_hold_v0_floor_goes_straight_at_the_entry_speed():
    v0 = torch.tensor([10.0, 0.0])
    hv = D.hold_v0(v0)
    assert torch.allclose(hv[0, :, 0], torch.tensor([5., 10., 15., 20.]))
    assert torch.allclose(hv[:, :, 1], torch.zeros(2, 4))     # never turns
    assert torch.allclose(hv[1], torch.zeros(4, 2))           # v0=0 -> stays


def test_zero_error_scores_zero_everywhere():
    gt = torch.tensor([[[5., 0.], [10., .2], [15., .6], [20., 1.2]]])
    pw = D.per_window(gt.clone(), gt)
    for k in ("ade_0_2s", "fde_2s", "miss_2m", "long_abs_2s_m",
              "lat_abs_2s_m", "speed_mae_mps", "progress_abs_err_m",
              "heading_mae_2s_deg", "pathgeom_crosstrack_m"):
        assert float(np.asarray(pw[k]).mean()) < 1e-4, k
    assert float(pw["curv_sign_agree"].mean()) == 1.0


# --------------------------------------------------------------------------- #
# paired-test orientation — the sep/tie/LOST three-way                          #
# --------------------------------------------------------------------------- #
def test_paired_orientation_is_floor_minus_model():
    """Positive delta MUST mean the model wins, everywhere, or the whole
    leaderboard inverts."""
    eid = ["e%d" % (i // 5) for i in range(40)]
    model = np.full(40, 1.0)
    floor = np.full(40, 2.0)
    d = D._paired(floor, model, D._Draws(eid, n_boot=200))
    assert d["delta"] == 1.0 and d["favours"] == "model" and d["separated"]
    d2 = D._paired(model, floor, D._Draws(eid, n_boot=200))
    assert d2["delta"] == -1.0 and d2["favours"] == "floor" and d2["separated"]


def test_sep_tag_is_three_way_not_two_way():
    win = {"separated": True, "favours": "model"}
    lost = {"separated": True, "favours": "floor"}
    tie = {"separated": False, "favours": "tie"}
    assert D.sep_tag(win) == "win"
    assert D.sep_tag(lost) == "LOST"
    assert D.sep_tag(tie) == "tie"
    assert D.sep_tag(lost) != D.sep_tag(win), \
        "a separated LOSS must never render like a separated WIN"


def test_unseparated_win_is_a_tie_in_the_verdict():
    """Suite R2. A point estimate in our favour with a CI spanning zero is a
    tie, and `where_the_win_lives` must say so."""
    out = _pin_block(n_boot=2000)
    v = out["verdict"]
    assert v["along_track_vs_cv"]["delta"] > 0            # point estimate wins
    assert v["along_track_vs_cv"]["favours"] == "tie"     # but the CI does not
    assert "longitudinal" not in v["where_the_win_lives"]


# --------------------------------------------------------------------------- #
# constants — pinned to their primary sources                                   #
# --------------------------------------------------------------------------- #
def test_constants_match_their_primary_sources():
    """MEASURED constants must come from the code that defines them, not be
    retyped here. Skips loudly if the stack is not importable (pod layout)."""
    try:
        sys.path.insert(0, "/root/TanitAD/stack/scripts")
        sys.path.insert(0, str(_HERE.parents[2] / "stack" / "scripts"))
        import driving_diagnostic as dd
        import refb_labels as rl
    except Exception as e:                                     # noqa: BLE001
        print(f"  (constants pin UNVERIFIED here: {type(e).__name__}: {e})")
        return
    assert D.STOP_V_MS == rl.STOP_V_MS and D.MOVING_V_MS == rl.MOVING_V_MS
    assert D.CURV_STRAIGHT_DEG == dd.CURV_STRAIGHT_DEG
    assert tuple(dd.WP_STEPS) == (5, 10, 15, 20)
    assert D.DT_WP == 0.5 and D.HORIZON_S == 2.0
    # The KNOWN divergence, pinned so it cannot drift silently into agreement
    # or into a third value without this test noticing (escalation E9).
    assert dd.CURV_GENTLE_DEG == 20.0 and D.CURV_SHARP_DEG == 15.0, (
        "driving_diagnostic and the v2 suite bucket curvature differently; if "
        "one of them changed, reconcile the panels rather than the test")


def test_block_is_versioned_and_self_describing():
    out = _pin_block()
    assert out["block"] == "taniteval.driving/tier0"
    assert out["version"] == D.VERSION and out["version"].count(".") == 2
    assert "TANITEVAL_V2_METRIC_SUITE.md" in out["spec"]
    assert out["claim_strength"].startswith("open-loop")
    for k in ("brake_accel_mps2", "curv_straight_deg", "curv_sharp_deg",
              "stop_v_ms", "min_n_stratum"):
        assert k in out["thresholds"], k
    assert out["estimator"]["deprecated_and_refused"] == D.DEPRECATED_ESTIMATOR


def test_refusals_are_declared_and_not_secretly_implemented():
    """Suite §6. If someone adds a headway/TTC/VTARGET/lane metric, the block
    would carry the key AND still claim to refuse it — catch that."""
    out = _pin_block()
    for k in ("headway_ttc_distance_keeping", "vtarget_referenced_speed_at_2s",
              "intersection_roundabout_merge_capability",
              "lane_centre_deviation", "curvature_mae_at_this_resolution"):
        assert k in out["refused"], k
    emitted = set(out["headline"]) | set(D.PAIRED)
    for banned in ("ttc", "headway", "gap", "vtarget", "lane_centre",
                   "lane_keep", "curv_mae", "curvature_mae"):
        assert not any(banned in m for m in emitted), banned


def test_kinematic_strata_are_never_called_scenarios():
    """The events are 5-20 s and the horizon is 2 s. A stratum named
    'intersection' would be a capability claim the data cannot support."""
    ks = _pin_block()["kinematic_strata"]
    assert "KINEMATIC SIGNATURES" in ks["_naming_contract"]
    for name in ks:
        if name.startswith("_"):
            continue
        assert name in ("launch_from_stop", "stop_approach", "sustained_turn")
        for banned in ("intersection", "roundabout", "merge", "junction"):
            assert banned not in name


def test_small_strata_are_flagged_low_confidence():
    """Suite R3: n < 30 is not a result."""
    ks = _pin_block()["kinematic_strata"]
    for name, row in ks.items():
        if name.startswith("_"):
            continue
        assert row["low_confidence"] == (row["n"] < D.MIN_N_STRATUM), name


# --------------------------------------------------------------------------- #
# ci.py reducer extension (suite E3/P6)                                         #
# --------------------------------------------------------------------------- #
def test_ci_gained_a_median_and_a_callable_path():
    """R5 is a MEASURED finding, not a style preference: heading MAE's CI is
    [2.34, 12.02] around a mean of 6.61 — the mean is the wrong reducer."""
    assert {"mean", "rms", "median", "p90"} <= set(C.REDUCERS)
    v = [1.0, 2.0, 3.0, 100.0]
    # The heavy tail is exactly why R5 exists: mean 26.5 vs median 2.5.
    assert C.REDUCERS["median"](v) == 2.5 and C.REDUCERS["mean"](v) == 26.5
    assert C.REDUCERS["p10"](v) < C.REDUCERS["median"](v) \
        < C.resolve_reducer("p90")(v) <= max(v)
    assert C.resolve_reducer(lambda x: float(np.max(x)))(v) == 100.0
    try:
        C.resolve_reducer("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown reducer must fail loud")


def test_ci_intervals_record_their_reducer():
    eid = ["e%d" % (i // 4) for i in range(40)]
    v = np.arange(40, dtype=float)
    a = C.episode_cluster_bootstrap(v, eid, reduce="median", n_boot=100)
    assert a["reducer"] == "median" and a["estimator"] == "episode_cluster_bootstrap"
    b = C.paired_episode_cluster_bootstrap(v + 1, v, eid, reduce="median",
                                           n_boot=100)
    assert b["reducer"] == "median" and abs(b["delta"] - 1.0) < 1e-9


def test_deprecated_estimator_still_exists_for_reproduction_only():
    """It is deprecated, not deleted — every historically published interval
    must stay reproducible. It just may never be a driving verdict."""
    assert callable(C.overlapping_holdout_se)
    assert C.overlapping_holdout_se([1.0, 2.0, 3.0]) > 0


# --------------------------------------------------------------------------- #
# report panel + artifact contract                                              #
# --------------------------------------------------------------------------- #
def _fake_block(key, ade, along, cross, spd, along_sep=True, cross_sep=True):
    def iv(m):
        return {"mean": m, "lo": m * .9, "hi": m * 1.1, "ci95": m * .1,
                "se": .01, "reducer": "mean", "n_windows": 881,
                "n_episodes": 40, "n_boot": 2000,
                "estimator": "episode_cluster_bootstrap"}

    def pd(delta, sep):
        return {"delta": delta, "ci": [delta - .1, delta + .1],
                "separated": sep, "favours": "model" if sep else "tie",
                "estimator": "paired_episode_cluster_bootstrap"}
    return {
        "block": D.BLOCK, "version": D.VERSION, "arm": key,
        "n_windows": 881, "n_episodes": 40,
        "estimator": {"n_boot": 2000},
        "headline": {k: iv(v) for k, v in
                     (("ade_0_2s", ade), ("long_abs_2s_m", along),
                      ("lat_abs_2s_m", cross), ("speed_mae_mps", spd),
                      ("curv_sign_agree", 0.95))},
        "floor_values": {"holdv0": {"speed_mae_mps": {"value": 0.4818}},
                         "cv": {"speed_mae_mps": {"value": 0.4678}}},
        "by_curvature": {"straight": {"n": 634, "model_heading_mae_deg": 7.98,
                                      "cv_heading_mae_deg": 1.399}},
        "verdict": {"ade_vs_cv": pd(.41, True),
                    "along_track_vs_cv": pd(.25, along_sep),
                    "cross_track_vs_cv": pd(.77, cross_sep),
                    "speed_mae_vs_cv": pd(-.003, False),
                    "cruise_speed_vs_holdv0": {
                        "n": 639, "delta": -0.2122, "ci": [-.28, -.14],
                        "separated": True, "favours": "floor",
                        "estimator": "paired_episode_cluster_bootstrap"},
                    "where_the_win_lives": ("both axes" if along_sep
                                            else "lateral only"),
                    "tracks_speed_better_than_cv": False}}


def test_panel_rows_renders_and_shows_the_split():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "driving_a.json").write_text(json.dumps(
            _fake_block("a", 0.4271, 0.8412, 0.2369, 0.4710, along_sep=False)))
        (p / "driving_b.json").write_text(json.dumps(
            _fake_block("b", 0.4714, 0.8785, 0.2803, 0.4545)))
        html = D.panel_rows(p)
        assert html.count("<tr>") == 2
        assert "0.4271" in html and "0.4714" in html
        assert "0.841" in html and "0.237" in html, "the split must be visible"
        assert "lateral only" in html and "both axes" in html
        assert "-0.212" in html, "the cruise loss must reach the panel"
        assert "crit" in html, "a separated loss must be flagged"


def test_panel_rows_sorts_by_ade_and_ignores_foreign_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "driving_slow.json").write_text(json.dumps(
            _fake_block("slow", 0.9, 1.1, 0.5, 0.6)))
        (p / "driving_fast.json").write_text(json.dumps(
            _fake_block("fast", 0.4, 0.8, 0.2, 0.4)))
        (p / "driving_bad.json").write_text("{not json")
        (p / "eff_x.json").write_text(json.dumps({"key": "x"}))
        html = D.panel_rows(p)
        assert html.count("<tr>") == 2
        assert html.index("fast") < html.index("slow")


def test_panel_rows_rejects_a_block_of_the_wrong_contract():
    """A future v3 block must not silently render in the v2 panel."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        b = _fake_block("a", 0.4, 0.8, 0.2, 0.4)
        b["block"] = "taniteval.driving/tier9"
        (p / "driving_a.json").write_text(json.dumps(b))
        assert D.panel_rows(p) == ""


def test_panel_rows_empty_dir_is_not_an_error():
    with tempfile.TemporaryDirectory() as d:
        assert D.panel_rows(Path(d)) == ""


def test_leaderboard_md_labels_units_and_names_the_estimator():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "driving_a.json").write_text(json.dumps(
            _fake_block("a", 0.4271, 0.8412, 0.2369, 0.4710, along_sep=False)))
        md = D.leaderboard_md(p)
        head = md.splitlines()[0]
        assert "ADE@2s m" in head and "m/s" in head and "ep-cluster boot" in head
        row = md.splitlines()[2]
        assert "0.4271" in row and "**tie**" in row and "0.237" in row
        assert "lateral only" in row


def test_arms_with_windows_reads_dumps_not_the_registry():
    """8 of the 24 dumps have no registry entry; tier-0 must still cover them."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        for k in ("flagship-30k", "not-in-any-registry"):
            (p / f"windows_{k}.pt").write_bytes(b"")
        assert D.arms_with_windows(p) == ["flagship-30k", "not-in-any-registry"]


def test_run_and_save_writes_the_block_and_merges_into_the_result_json():
    if not PIN_ARM.exists():
        raise AssertionError(f"{PIN_ARM} missing")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / f"windows_{D.SANITY_ARM}.pt").write_bytes(PIN_ARM.read_bytes())
        (p / f"{D.SANITY_ARM}.json").write_text(json.dumps({"heldout": {}}))
        D.run_and_save(D.SANITY_ARM, res_dir=p, n_boot=100)
        blk = json.loads((p / f"driving_{D.SANITY_ARM}.json").read_text())
        assert blk["block"] == D.BLOCK and blk["arm"] == D.SANITY_ARM
        merged = json.loads((p / f"{D.SANITY_ARM}.json").read_text())
        assert merged["driving"]["block"] == D.BLOCK, \
            "the block must live WITH the accuracy row, not only beside it"


def test_quick_is_the_same_contract_as_the_offline_block():
    """`runner.run_one` must not emit a weaker artifact than the backfill."""
    win = torch.load(str(PIN_ARM), map_location="cpu", weights_only=False)
    q = D.quick(win, n_boot=100, arm=D.SANITY_ARM)
    assert q["inline"] is True
    assert q["block"] == D.BLOCK and q["version"] == D.VERSION
    assert set(q["headline"]) == set(D.HEADLINE)
    assert q["estimator"]["interval"] == "episode_cluster_bootstrap"
    assert D.N_BOOT_INLINE == D.N_BOOT, \
        "the inline block is decision-grade; do not weaken it silently"


def test_block_is_cpu_only_and_touches_no_gpu():
    """It runs inside every eval and beside a training pod. If it ever moved a
    tensor to CUDA it could OOM a trainer — the exact accident of 2026-07-16."""
    src = (Path(D.__file__)).read_text(encoding="utf-8")
    for banned in (".cuda(", "device='cuda'", 'device="cuda"',
                   "torch.cuda.synchronize"):
        assert banned not in src, f"driving.py must stay CPU-only: {banned}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                    # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"==== {len(fns) - bad}/{len(fns)} passed ====")
    sys.exit(1 if bad else 0)
