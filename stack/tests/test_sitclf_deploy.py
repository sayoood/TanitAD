"""The DEPLOYED situation-scoring path: that late fusion is actually wired in,
that it keeps both modalities, and that the comparison it reports can be trusted.

The load-bearing tests here are the two controls, not the win: a fusion step with
free parameters must be shown UNABLE to manufacture a gain on its own, or the
before/after it produces is not evidence of anything.
"""

import numpy as np
import pytest

from tanitad.eval.ap_ci import average_precision
from tanitad.eval.sitclf import EGO_SCALE, cluster_folds
from tanitad.eval.sitclf_deploy import (
    DEPLOYED_ARM,
    MODALITY_ARMS,
    ScoreBundle,
    anticipation_lead_s,
    four_family_report,
    fuse_modalities,
    VISION_ARMS,
    VISION_NULL_ARMS,
    is_vision_only,
    load_score_bundle,
    event_anticipation_report,
    permute_labels_by_cluster,
    precision_recall_at_budget,
    regime_strata,
    vision_only_arms,
)

SITS = ("lane_change", "intersection")


def _swamped_bundle(seed=0, n_clip=150, per=120):
    """A bundle with the REAL defect's shape: an ego arm that works, an image arm
    that carries a little signal, and a deployed 'early concat' arm that is WORSE
    than the ego arm because the wide image block swamped the narrow ego block."""
    rng = np.random.default_rng(seed)
    n = n_clip * per
    cc = np.repeat(np.arange(n_clip), per)
    y = (rng.random((n, len(SITS))) < 0.06).astype(np.int64)
    ego_s = rng.normal(size=(n, len(SITS))) + 2.0 * y          # strong
    img_s = rng.normal(size=(n, len(SITS))) + 0.6 * y          # weak but real
    img_shuf = rng.normal(size=(n, len(SITS)))                 # the camera's null
    # early concat: the image block dominates, so the ego signal is mostly lost
    deployed = 0.9 * img_s + 0.1 * ego_s
    valid = np.ones((n, len(SITS)), bool)
    ego = np.stack([rng.uniform(0, 35, n), rng.normal(0, 1.5, n),
                    rng.normal(0, 0.12, n)], 1) / EGO_SCALE
    return ScoreBundle(
        situations=SITS, arms=(DEPLOYED_ARM, "head_img", "head_ego", "head_img_shuf"),
        y=y, valid=valid, clip_cluster=cc,
        scores={DEPLOYED_ARM: deployed, "head_img": img_s, "head_ego": ego_s,
                "head_img_shuf": img_shuf},
        ego=ego, source="<test>")


# --------------------------------------------------------------------------- #
# the bundle contract                                                         #
# --------------------------------------------------------------------------- #
def test_bundle_rejects_misshapen_arms():
    b = _swamped_bundle()
    with pytest.raises(ValueError):
        ScoreBundle(situations=SITS, arms=("a",), y=b.y, valid=b.valid,
                    clip_cluster=b.clip_cluster, scores={"a": np.zeros((5, 2))})


def test_bundle_rejects_label_situation_width_mismatch():
    b = _swamped_bundle()
    with pytest.raises(ValueError):
        ScoreBundle(situations=("only_one",), arms=(), y=b.y, valid=b.valid,
                    clip_cluster=b.clip_cluster, scores={})


def test_bundle_arm_lookup_is_by_situation_name():
    b = _swamped_bundle()
    assert b.arm("head_ego", "intersection").shape == (b.n_rows,)
    with pytest.raises(KeyError):
        b.arm("head_ego", "roundabout")
    with pytest.raises(KeyError):
        b.arm("nope", "lane_change")


def test_load_score_bundle_roundtrips_and_reports_missing_keys(tmp_path):
    b = _swamped_bundle(n_clip=6, per=20)
    p = tmp_path / "scores.npz"
    np.savez_compressed(p, situations=np.array(SITS), arms=np.array(list(b.scores)),
                        y=b.y, valid=b.valid, clip_cluster=b.clip_cluster, ego=b.ego,
                        **b.scores)
    got = load_score_bundle(p)
    assert got.situations == SITS and got.n_rows == b.n_rows
    assert np.allclose(got.arm("head_ego", "lane_change"), b.arm("head_ego", "lane_change"))

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, y=b.y)
    with pytest.raises(KeyError, match="missing"):
        load_score_bundle(bad)


# --------------------------------------------------------------------------- #
# THE FIX — end to end                                                        #
# --------------------------------------------------------------------------- #
def test_fusion_repairs_the_swamped_deployed_arm():
    """The whole point: on the defect's own shape, score-level fusion of the two
    modalities beats the early-concat arm it replaces — WITHOUT dropping either
    modality (the PI ruling that closed the ego-only swap)."""
    b = _swamped_bundle()
    i = b.col("lane_change")
    fused = fuse_modalities(b, "lane_change")
    m = b.valid[:, i]
    ap_fused = average_precision(b.y[m, i], fused[m])
    ap_deployed = average_precision(b.y[m, i], b.arm(DEPLOYED_ARM, "lane_change")[m])
    assert ap_fused > ap_deployed
    # and it must not merely equal the better unimodal arm by discarding the other
    assert MODALITY_ARMS == ("head_img", "head_ego")


def test_fusion_keeps_both_modalities_and_is_not_a_passthrough():
    """A 'fusion' that reproduced one input would satisfy the AP test above while
    silently being the ego-only head the PI rejected."""
    b = _swamped_bundle(seed=2)
    fused = fuse_modalities(b, "intersection")
    m = b.valid[:, b.col("intersection")]
    for a in MODALITY_ARMS:
        r = np.corrcoef(fused[m], b.arm(a, "intersection")[m])[0, 1]
        assert abs(r) < 0.999, f"fused score is a passthrough of {a} (r={r:.5f})"


def test_fusion_is_out_of_fold_on_whole_clusters():
    """Every returned score comes from a combiner fitted without that row's clip.
    Reproduced by refitting the fold complement by hand."""
    from sklearn.linear_model import LogisticRegression

    b = _swamped_bundle(seed=4, n_clip=40, per=60)
    i = b.col("lane_change")
    F = np.stack([b.arm(a, "lane_change") for a in MODALITY_ARMS], 1)
    folds = cluster_folds(b.clip_cluster, n_folds=2, seed=0)
    got = fuse_modalities(b, "lane_change", n_folds=2, seed=0)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        mu, sd = F[tr].mean(0), np.maximum(F[tr].std(0), 1e-9)
        lr = LogisticRegression(C=1.0, max_iter=300).fit((F[tr] - mu) / sd, b.y[tr, i])
        assert np.allclose(got[te], lr.decision_function((F[te] - mu) / sd))
    # a cluster is never split across folds -> no frame of a scored clip was fitted on
    for c in np.unique(b.clip_cluster):
        assert len(np.unique(folds[b.clip_cluster == c])) == 1


def test_invalid_rows_come_back_minus_inf():
    b = _swamped_bundle(seed=6, n_clip=20, per=50)
    b.valid[:41, b.col("lane_change")] = False
    fused = fuse_modalities(b, "lane_change")
    assert np.all(np.isneginf(fused[:41]))
    assert np.all(np.isfinite(fused[b.valid[:, b.col("lane_change")]]))


# --------------------------------------------------------------------------- #
# THE CONTROLS — without these the before/after is not evidence                #
# --------------------------------------------------------------------------- #
def test_single_column_fusion_cannot_manufacture_a_gain():
    """NEGATIVE CONTROL 1. The combiner has free parameters fitted on the scored
    set. Run it on ONE column and it must not improve that column: any gain the
    two-column form shows is then attributable to the second modality, not to the
    fitting protocol."""
    b = _swamped_bundle(seed=8)
    i = b.col("lane_change")
    m = b.valid[:, i]
    raw = b.arm(DEPLOYED_ARM, "lane_change")
    solo = fuse_modalities(b, "lane_change", arms=(DEPLOYED_ARM,))
    ap_raw = average_precision(b.y[m, i], raw[m])
    ap_solo = average_precision(b.y[m, i], solo[m])
    assert ap_solo <= ap_raw * 1.10, (ap_solo, ap_raw)


def test_fusing_the_camera_null_is_worse_than_fusing_the_camera():
    """NEGATIVE CONTROL 2 — the camera's marginal value, with the parameter count
    and the fitting protocol held IDENTICAL. Here the image arm carries real
    signal by construction, so the real-image fusion must win; on the corpus this
    same contrast is what decides whether the camera earns its place."""
    b = _swamped_bundle(seed=10)
    i = b.col("lane_change")
    m = b.valid[:, i]
    real = fuse_modalities(b, "lane_change", arms=("head_img", "head_ego"))
    null = fuse_modalities(b, "lane_change", arms=("head_img_shuf", "head_ego"))
    assert average_precision(b.y[m, i], real[m]) > average_precision(b.y[m, i], null[m])


def test_paired_report_does_not_separate_an_arm_from_itself():
    """NEGATIVE CONTROL 3 — the estimator's own. If comparing a score to a copy of
    itself 'separated', every separation this module reports would be an artifact."""
    b = _swamped_bundle(seed=12, n_clip=40, per=50)
    fused = fuse_modalities(b, "lane_change")
    rep = four_family_report(b, "lane_change", fused=fused, baseline=fused.copy(),
                             n_boot=200, strata_n_boot=100)
    d = rep["families"]["TACTICAL"]["paired_delta_ap_lift"]
    assert d["separated"] is False and abs(d["delta"]) < 1e-9


def test_report_separates_a_genuinely_better_arm():
    """The estimator's POSITIVE control: it must be able to fire. A test suite that
    only proves an estimator stays silent proves nothing about its sensitivity."""
    b = _swamped_bundle(seed=14)
    fused = fuse_modalities(b, "lane_change")
    rep = four_family_report(b, "lane_change", fused=fused,
                             baseline=b.arm(DEPLOYED_ARM, "lane_change"),
                             n_boot=300, strata_n_boot=100)
    d = rep["families"]["TACTICAL"]["paired_delta_ap_lift"]
    assert d["separated"] is True and d["delta"] > 0


# --------------------------------------------------------------------------- #
# regimes and the four families                                               #
# --------------------------------------------------------------------------- #
def test_regime_strata_undo_the_ego_scale():
    """The bundle stores ego ALREADY divided by EGO_SCALE (sc_train.py:93); a
    stratum thresholded on the scaled values would cut at 10x/2x/0.5x the
    intended physical value and silently mislabel the regimes."""
    raw = np.array([[20.0, -2.0, 0.0],      # cruising, braking hard, straight
                    [3.0, 0.0, 0.4]])       # slow, steady, turning
    st = regime_strata(raw / EGO_SCALE)
    assert list(st["longitudinal"]["decelerating"]) == [True, False]
    assert list(st["longitudinal"]["cruise_ge8"]) == [True, False]
    assert list(st["lateral"]["turning"]) == [False, True]
    assert list(st["lateral"]["straight"]) == [True, False]


def test_regime_strata_partition_each_axis():
    b = _swamped_bundle(seed=16, n_clip=30, per=40)
    st = regime_strata(b.ego)
    lat = st["lateral"]
    assert np.all(lat["straight"] ^ lat["turning"])          # exact partition
    lon = st["longitudinal"]
    assert np.all(lon["low_speed_lt8"] ^ lon["cruise_ge8"])


def test_regime_strata_reject_a_bad_shape():
    with pytest.raises(ValueError):
        regime_strata(np.zeros((10, 2)))


def test_report_carries_all_four_families_with_reasons():
    """The binding rule: four families, never pooled, and a family that cannot be
    computed says so WITH its reason and its n."""
    b = _swamped_bundle(seed=18, n_clip=40, per=50)
    fused = fuse_modalities(b, "lane_change")
    rep = four_family_report(b, "lane_change", fused=fused,
                             baseline=b.arm(DEPLOYED_ARM, "lane_change"),
                             n_boot=100, strata_n_boot=60)
    fams = rep["families"]
    assert set(fams) == {"TACTICAL", "LONGITUDINAL", "LATERAL", "STRATEGIC"}
    assert fams["STRATEGIC"]["_status"] == "UNAVAILABLE"
    assert fams["STRATEGIC"]["_reason"] and fams["STRATEGIC"]["n_rows"] > 0
    for fam in ("LONGITUDINAL", "LATERAL"):
        assert fams[fam]["_not_computable"] and fams[fam]["_not_computable_reason"]
        assert fams[fam]["strata"], fam
        for s in fams[fam]["strata"].values():
            assert "n_rows" in s
            if s.get("_status") != "UNPOWERED":
                assert s["paired_delta_ap_lift"]["estimator"].startswith("paired_")


def test_report_marks_lat_lon_unavailable_without_ego():
    b = _swamped_bundle(seed=20, n_clip=20, per=40)
    b.ego = None
    fused = fuse_modalities(b, "lane_change")
    rep = four_family_report(b, "lane_change", fused=fused,
                             baseline=b.arm(DEPLOYED_ARM, "lane_change"), n_boot=60)
    for fam in ("LONGITUDINAL", "LATERAL"):
        assert rep["families"][fam]["_status"] == "UNAVAILABLE"
        assert rep["families"][fam]["_reason"]


def test_report_never_pools_the_families_into_one_score():
    b = _swamped_bundle(seed=22, n_clip=20, per=40)
    fused = fuse_modalities(b, "lane_change")
    rep = four_family_report(b, "lane_change", fused=fused,
                             baseline=b.arm(DEPLOYED_ARM, "lane_change"), n_boot=60)
    flat = str(rep).lower()
    for banned in ("composite", "overall_score", "pooled"):
        assert banned not in flat


# --------------------------------------------------------------------------- #
# anticipation lead                                                           #
# --------------------------------------------------------------------------- #
def _lead_fixture(n_clip=40, per=60, onset=40, run=30):
    """One onset per clip, preceded by a `run`-frame anticipation window."""
    cc = np.repeat(np.arange(n_clip), per)
    y = np.zeros(n_clip * per, bool)
    for c in range(n_clip):
        o = c * per + onset
        y[o - run:o] = True
    return cc, y


def test_anticipation_lead_prefers_the_arm_that_fires_earlier():
    """Both arms alarm inside every anticipation window and at the SAME alarm
    budget; only WHEN they fire differs, so the metric must rank them by that."""
    n_clip, per, onset = 40, 60, 40
    cc, y = _lead_fixture(n_clip, per, onset)
    early, late = np.zeros(y.size), np.zeros(y.size)
    for c in range(n_clip):
        o = c * per + onset
        early[o - 30:o - 26] = 5.0            # first 4 frames of the window
        late[o - 4:o] = 5.0                   # last 4 frames of the window
    valid = np.ones(y.size, bool)
    tf = 4 * n_clip / y.size
    le = anticipation_lead_s(y, early, cc, valid, top_frac=tf)
    ll = anticipation_lead_s(y, late, cc, valid, top_frac=tf)
    assert le["median_lead_s"] == pytest.approx(3.0)
    assert ll["median_lead_s"] == pytest.approx(0.4)
    assert le["n_runs"] == ll["n_runs"] == n_clip
    assert le["n_runs_no_alarm"] == 0


def test_anticipation_lead_operating_point_is_rank_based_not_a_value_threshold():
    """REGRESSION. Sigmoid scores tie heavily; `s >= quantile` then admits every
    tied row and turns a 5 % operating point into a 100 % one, which made a
    late-firing arm look as early as an early-firing one. The alarm COUNT must
    track top_frac regardless of ties."""
    cc, y = _lead_fixture(20, 60, 40)
    s = np.zeros(y.size)                       # fully tied
    out = anticipation_lead_s(y, s, cc, np.ones(y.size, bool), top_frac=0.05)
    assert out["n_alarm_rows"] == pytest.approx(0.05 * y.size, abs=1)


def test_anticipation_lead_counts_silent_runs_instead_of_scoring_them_zero():
    n_clip, per, onset, run = 10, 40, 25, 5
    cc, y = _lead_fixture(n_clip, per, onset, run)
    s = np.zeros(y.size)
    for c in range(5):                         # only half the clips ever alarm
        s[c * per + 22] = 9.0
    out = anticipation_lead_s(y, s, cc, np.ones(y.size, bool), top_frac=5 / y.size)
    assert out["n_runs"] == n_clip and out["n_runs_no_alarm"] == 5
    assert out["median_lead_s"] == pytest.approx(0.3)


def test_anticipation_lead_ignores_alarms_outside_the_anticipation_window():
    """An alarm long before the window is not anticipation of THIS onset — it is
    a false positive that happens to sit in the same clip."""
    n_clip, per, onset, run = 8, 50, 40, 10
    cc, y = _lead_fixture(n_clip, per, onset, run)
    s = np.zeros(y.size)
    for c in range(n_clip):
        s[c * per + 2] = 9.0                   # 2.8 s before the window opens
    out = anticipation_lead_s(y, s, cc, np.ones(y.size, bool), top_frac=n_clip / y.size)
    assert out["n_runs"] == n_clip and out["n_runs_no_alarm"] == n_clip
    assert out["median_lead_s"] is None


def test_anticipation_lead_is_none_when_nothing_is_scorable():
    cc = np.zeros(10, int)
    out = anticipation_lead_s(np.zeros(10, bool), np.zeros(10), cc, np.ones(10, bool))
    assert out["median_lead_s"] is None and out["n_runs"] == 0


# --------------------------------------------------------------------------- #
# THE DEPLOYMENT CONTRACT — vision only at inference (PI ruling 2026-08-03)    #
#                                                                             #
# "for ground truth data of scenario classification you can use both ego and   #
#  other label, for inference only vision."                                    #
# --------------------------------------------------------------------------- #
def _vision_bundle(seed=0, n_clip=120, per=100):
    rng = np.random.default_rng(seed)
    n = n_clip * per
    cc = np.repeat(np.arange(n_clip), per)
    y = (rng.random((n, len(SITS))) < 0.06).astype(np.int64)
    def sig(k):
        return rng.normal(size=(n, len(SITS))) + k * y
    return ScoreBundle(
        situations=SITS,
        arms=("head_img", "ridge_img", "head_img_shuf", "ridge_img_shuf",
              "head_ego", "head_img_ego"),
        y=y, valid=np.ones((n, len(SITS)), bool), clip_cluster=cc,
        scores={"head_img": sig(0.7), "ridge_img": sig(0.5),
                "head_img_shuf": sig(0.0), "ridge_img_shuf": sig(0.0),
                "head_ego": sig(2.0), "head_img_ego": sig(0.3)},
        ego=np.stack([rng.uniform(0, 35, n), rng.normal(0, 1.5, n),
                      rng.normal(0, 0.12, n)], 1) / EGO_SCALE,
        source="<test-vision>")


def test_vision_panel_contains_no_ego_reading_arm():
    """The contract, mechanically: every arm in the panel must be derivable
    without an ego channel at inference. A regression here is a ruling breach,
    not a metric change."""
    assert all(is_vision_only(a) for a in VISION_ARMS + VISION_NULL_ARMS)
    for bad in ("head_ego", "head_img_ego", "ridge_img_ego", "head_priv"):
        assert not is_vision_only(bad), bad
    assert is_vision_only("head_img") and is_vision_only("ridge_img")


def test_vision_only_arms_builds_the_full_control_panel():
    b = _vision_bundle()
    arms = vision_only_arms(b, "lane_change")
    assert set(arms) == {"PRIMARY", "FUSED", "NEG_MACHINERY", "NEG_VISION",
                         "NEG_FUSED", "NEG_LABEL"}
    i = b.col("lane_change")
    m = b.valid[:, i]
    # the discrimination control must FIRE: the camera is above its own null
    ap_p = average_precision(b.y[m, i], arms["PRIMARY"][m])
    ap_n = average_precision(b.y[m, i], arms["NEG_VISION"][m])
    assert ap_p > ap_n


def test_vision_only_arms_refuses_an_ego_reading_arm():
    b = _vision_bundle()
    import tanitad.eval.sitclf_deploy as SD
    old = SD.VISION_ARMS
    try:
        SD.VISION_ARMS = ("head_img_ego", "ridge_img")
        with pytest.raises(ValueError, match="read ego at inference"):
            SD.vision_only_arms(b, "lane_change")
    finally:
        SD.VISION_ARMS = old


def test_vision_only_arms_needs_at_least_one_vision_arm():
    b = _vision_bundle()
    del b.scores["head_img"]
    del b.scores["ridge_img"]
    with pytest.raises(KeyError, match="no vision arm"):
        vision_only_arms(b, "lane_change")


def test_vision_only_degrades_to_primary_when_only_one_vision_arm_exists():
    """A bundle with a single vision arm must still produce a scorable panel
    rather than raising — the deployable arm does not depend on having a second
    one to fuse with."""
    b = _vision_bundle()
    del b.scores["ridge_img"]
    del b.scores["ridge_img_shuf"]
    arms = vision_only_arms(b, "lane_change")
    assert "FUSED" not in arms and "NEG_FUSED" not in arms
    assert {"PRIMARY", "NEG_MACHINERY", "NEG_VISION", "NEG_LABEL"} <= set(arms)


def test_label_permutation_moves_whole_clusters_not_rows():
    """A row-wise shuffle would destroy the within-clip correlation the cluster
    estimator assumes and make the control easier than the real task."""
    cc = np.repeat(np.arange(8), 10)
    y = np.repeat(np.arange(8) % 2, 10)          # each cluster is constant
    out = permute_labels_by_cluster(y, cc, seed=3)
    for c in np.unique(cc):
        assert len(np.unique(out[cc == c])) == 1  # still constant per cluster
    assert out.sum() == y.sum()                   # a permutation, not a resample
    assert not np.array_equal(out, y)


def test_label_permutation_is_deterministic_for_a_seed():
    cc = np.repeat(np.arange(12), 5)
    y = (np.arange(60) % 3 == 0).astype(int)
    a = permute_labels_by_cluster(y, cc, seed=11)
    b_ = permute_labels_by_cluster(y, cc, seed=11)
    assert np.array_equal(a, b_)


# --------------------------------------------------------------------------- #
# precision alongside recall (binding, 2026-08-03)                            #
# --------------------------------------------------------------------------- #
def test_precision_recall_at_budget_states_both_denominators():
    """A recall number is unreadable without the count it was bought with."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.random(n) < 0.1
    s = y * 1.0 + rng.normal(0, 0.5, n)
    r = precision_recall_at_budget(y, s, np.ones(n, bool), top_frac=0.05)
    assert r["n_alarm"] == 100 and r["n_pos"] == int(y.sum())
    assert r["tp"] <= r["n_alarm"] and r["tp"] <= r["n_pos"]
    assert abs(r["precision"] - r["tp"] / r["n_alarm"]) < 1e-5
    assert abs(r["recall"] - r["tp"] / r["n_pos"]) < 1e-5
    assert r["precision"] > r["base_rate"]


def test_precision_falls_when_a_wider_budget_buys_recall():
    """THE CONTROL THE RETRACTED CLAIM LACKED: raising recall by firing more must
    show up as a precision cost. If precision did not move, the metric could not
    see what a recall gain is paying."""
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.random(n) < 0.08
    s = y * 1.0 + rng.normal(0, 0.8, n)
    tight = precision_recall_at_budget(y, s, np.ones(n, bool), top_frac=0.02)
    wide = precision_recall_at_budget(y, s, np.ones(n, bool), top_frac=0.30)
    assert wide["recall"] > tight["recall"]
    assert wide["precision"] < tight["precision"]


def test_precision_at_budget_lands_at_the_base_rate_on_a_useless_score():
    rng = np.random.default_rng(2)
    n = 20000
    y = rng.random(n) < 0.2
    s = rng.normal(size=n)
    r = precision_recall_at_budget(y, s, np.ones(n, bool), top_frac=0.10)
    assert abs(r["precision"] - r["base_rate"]) < 0.03
    assert abs(r["precision_lift"] - 1.0) < 0.16


def test_precision_recall_ignores_invalid_and_nonfinite_rows():
    y = np.array([1, 1, 0, 0, 1, 0], bool)
    s = np.array([9.0, np.inf, 8.0, 1.0, 0.5, 0.0])
    valid = np.array([1, 1, 1, 1, 0, 1], bool)
    r = precision_recall_at_budget(y, s, valid, top_frac=0.5)
    assert r["n_scorable"] == 4          # inf dropped, invalid row dropped
    assert r["n_pos"] == 1


def test_precision_recall_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        precision_recall_at_budget(np.zeros(5, bool), np.zeros(4), np.ones(5, bool))


# --------------------------------------------------------------------------- #
# the EVENT-level, HORIZON-INDEPENDENT yardstick                              #
# --------------------------------------------------------------------------- #
def _two_clip_setup():
    """Two 100-frame clips, one onset each at frame 60 (global 60 and 160)."""
    n = 200
    cc = np.concatenate([np.zeros(100, np.int64), np.ones(100, np.int64)])
    onsets = np.array([60, 160])
    return n, cc, onsets


def test_event_report_takes_NO_label_so_it_cannot_move_with_lead_s():
    """The property the whole instrument exists for, asserted on the signature.

    `precision_recall_at_budget` and `anticipation_lead_s` both take `y`, which is a
    function of `lead_s`; this one takes onsets. If a `y` parameter ever appears here
    the horizon-independence claim is dead, so the check is on the parameter list.
    """
    import inspect
    params = inspect.signature(event_anticipation_report).parameters
    assert "y" not in params and "lead_s" not in params
    assert "onsets" in params


def test_event_recall_is_1_when_every_onset_has_an_alarm_in_its_window():
    n, cc, onsets = _two_clip_setup()
    s = np.zeros(n)
    s[[55, 155]] = 10.0                       # 0.5 s before each onset
    r = event_anticipation_report(s, np.ones(n, bool), onsets, cc,
                                  top_frac=0.05, h_max_s=5.0)
    assert r["n_onsets"] == 2
    assert r["n_onsets_warned"] == 2
    assert r["event_recall"] == 1.0
    assert r["median_lead_s"] == 0.5


def test_an_onset_with_no_alarm_contributes_NO_lead_not_a_zero():
    """The recall-only defect, in the lead metric: never reward silence with 0 s."""
    n, cc, onsets = _two_clip_setup()
    s = np.zeros(n)
    s[55] = 10.0                              # clip 0 warned at 0.5 s, clip 1 silent
    s[5] = 9.0                                # clip 1's budget spent far from its onset
    r = event_anticipation_report(s, np.ones(n, bool), onsets, cc,
                                  top_frac=0.01, h_max_s=5.0)
    assert r["n_onsets_warned"] == 1
    assert r["n_onsets_no_alarm"] == 1
    assert r["event_recall"] == 0.5
    assert r["median_lead_s"] == 0.5          # the silent onset did NOT contribute 0.0


def test_the_lookback_NEVER_crosses_a_clip_boundary():
    """Without the cluster guard, clip 1's onset at frame 0+ is 'warned' by clip 0's tail."""
    n = 200
    cc = np.concatenate([np.zeros(100, np.int64), np.ones(100, np.int64)])
    onsets = np.array([102])                  # 2 frames into clip 1
    s = np.zeros(n)
    s[98] = 10.0                              # in clip 0, 0.4 s earlier in GLOBAL index
    r = event_anticipation_report(s, np.ones(n, bool), onsets, cc,
                                  top_frac=0.01, h_max_s=5.0)
    assert r["n_alarm"] == 2                  # rank budget, ties keep input order
    assert r["n_onsets_warned"] == 0, "an alarm in the previous drive must not count"
    assert r["median_lead_s"] is None


def test_the_alarm_budget_is_IDENTICAL_across_arms_so_precision_is_comparable():
    n, cc, onsets = _two_clip_setup()
    rng = np.random.default_rng(0)
    a = event_anticipation_report(rng.normal(size=n), np.ones(n, bool), onsets, cc,
                                  top_frac=0.05)
    b = event_anticipation_report(rng.normal(size=n) * 100 + 7, np.ones(n, bool),
                                  onsets, cc, top_frac=0.05)
    assert a["n_alarm"] == b["n_alarm"] == 10
    assert a["n_onsets"] == b["n_onsets"]     # both denominators fixed


def test_alarm_precision_is_reported_at_BOTH_horizons_and_the_shorter_is_no_larger():
    n, cc, onsets = _two_clip_setup()
    s = np.zeros(n)
    s[[35, 55, 135, 155]] = [1.0, 4.0, 2.0, 3.0]   # 2.5 s and 0.5 s before each onset
    r = event_anticipation_report(s, np.ones(n, bool), onsets, cc, top_frac=0.02,
                                  h_max_s=5.0, deploy_lead_s=1.0)
    assert r["n_alarm"] == 4
    assert r["alarm_precision_h_max"] == 1.0       # all four within 5 s of an onset
    assert r["alarm_precision_deploy"] == 0.5      # only the two within 1 s
    assert r["alarm_precision_deploy"] <= r["alarm_precision_h_max"]


def test_onsets_with_no_reachable_scorable_row_are_EXCLUDED_not_counted_as_missed():
    """An onset at frame 0 of a clip could never be warned; charging it to recall
    would make the metric depend on where clips happen to start."""
    n, cc, _ = _two_clip_setup()
    onsets = np.array([0, 60])
    s = np.zeros(n)
    s[55] = 10.0
    r = event_anticipation_report(s, np.ones(n, bool), onsets, cc, top_frac=0.01)
    assert r["n_onsets_total"] == 2
    assert r["n_onsets_unreachable"] == 1
    assert r["n_onsets"] == 1 and r["event_recall"] == 1.0


def test_event_report_rejects_misaligned_inputs_and_out_of_range_onsets():
    n, cc, onsets = _two_clip_setup()
    with pytest.raises(ValueError):
        event_anticipation_report(np.zeros(n), np.ones(n - 1, bool), onsets, cc)
    with pytest.raises(ValueError):
        event_anticipation_report(np.zeros(n), np.ones(n, bool), np.array([n + 5]), cc)
