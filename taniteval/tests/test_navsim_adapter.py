"""Tests for the NavSim -> TanitEval adapter (``taniteval/adapters/navsim.py``).

⛔ **NO DATASET IS DOWNLOADED.** Every fixture here is SYNTHETIC and
NavSim-SHAPED: constant-curvature arcs in the NavSim ego frame with an
``(x, y, heading)`` pose triple, plus a world-frame variant. What that buys and
what it does NOT buy is stated in ``NAVSIM_ADAPTER.md`` — these tests pin the
CONVERSION and the REFUSALS, and they cannot pin anything about real NavSim
bytes (column names in a real ``.csv``, the actual pose-array layout the devkit
emits, or whether a real ``navtest`` row carries ``score``).

⭐ Two of these are DELIBERATE REGRESSION ARMS. A guard that has never been shown
to fail proves nothing:
  * :func:`test_regression_transposed_frame_is_caught` feeds a transposed
    ``(y, x)`` tensor and asserts the harness's OWN verifier raises.
  * :func:`test_regression_y_sign_flip_is_caught` feeds a right-handed
    (``y``-right) frame and asserts the adapter's sign guard raises —
    and :func:`test_assert_axis_convention_is_blind_to_a_y_sign_flip`
    MEASURES that the harness verifier alone does **not** catch it, which is
    why the second guard exists.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tests
_TE = os.path.dirname(_HERE)                             # <repo>/taniteval
_REPO = os.path.dirname(_TE)                             # <repo>
for _p in (os.path.join(_REPO, "stack"),
           os.path.join(_REPO, "stack", "scripts"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import navsim as NS                        # noqa: E402

DT = 0.5          # NavSim agent output: interval_length=0.5 s, 8 poses over 4 s
K = 8
N = 24


# --------------------------------------------------------------------------- #
# Synthetic NavSim-shaped fixtures                                             #
# --------------------------------------------------------------------------- #
def arc_poses(n=N, k=K, dt=DT, v0=8.0, seed=0, include_origin=False,
              jitter=0.0):
    """Constant-curvature arcs in the NavSim ego frame -> ``([n,k,3], speeds)``.

    Column layout is NavSim's ``(x, y, heading)``: ``x`` forward, ``y`` LEFT,
    ``heading`` = vehicle yaw in radians, CCW positive
    (``NAVSIM_PROTOCOL.md:506-509``, ``:534``). Left turns (``omega > 0``)
    therefore have ``+y`` AND ``+`` net heading change — the relation the sign
    guard tests.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, k + (1 if include_origin else 0), 3), dtype=np.float64)
    speeds = np.zeros(n)
    for i in range(n):
        v = v0 + 0.35 * (i % 7)
        speeds[i] = v
        # alternate left / right / straight so BOTH turn signs are represented
        omega = {0: 0.09, 1: -0.09, 2: 0.0}[i % 3]
        x = y = h = 0.0
        rows = []
        if include_origin:
            rows.append((0.0, 0.0, 0.0))
        for _ in range(k):
            x += v * dt * math.cos(h)
            y += v * dt * math.sin(h)
            h += omega * dt
            rows.append((x, y, h))
        a = np.asarray(rows, dtype=np.float64)
        if jitter:
            a[:, :2] += rng.normal(0.0, jitter, size=(a.shape[0], 2))
        out[i] = a
    return out, speeds


def pred_from(gt_poses, noise=0.06, seed=1):
    rng = np.random.default_rng(seed)
    p = gt_poses.copy()
    p[..., :2] += rng.normal(0.0, noise, size=p[..., :2].shape)
    return p


def make_win(**kw):
    gt, sp = arc_poses()
    pr = pred_from(gt)
    args = dict(frame="ego", origin_included=False, dt_s=DT,
                ego_speed_mps=sp,
                scene_tokens=[f"tok{i:04d}" for i in range(gt.shape[0])])
    args.update(kw)
    return NS.scenes_to_win(pr, gt, **args)


# ========================================================================== #
# 1. The win-dict contract                                                   #
# ========================================================================== #
def test_win_carries_the_required_contract_keys():
    """ff_rescore.py:404-406 + four_families.py:1165-1181."""
    win = make_win()
    for key in ("pred_dense", "gt_dense", "pred", "gt", "wp_steps", "dt_s", "eid"):
        assert key in win, f"win is missing REQUIRED key {key!r}"
    assert win["pred_dense"].shape == (N, K, 2)
    assert win["gt_dense"].shape == (N, K, 2)
    assert win["pred_dense"].shape == win["gt_dense"].shape
    assert win["dt_s"] == DT
    assert win["pred"].shape[0] == N and win["pred"].shape[-1] == 2
    assert len(win["wp_steps"]) == win["pred"].shape[1]
    assert win["v0"] is not None and win["v0"].shape == (N,)


def test_sparse_companion_view_matches_ff_rescore_source():
    """⛔ Drift guard: the sparse-view formula is PORTED, so pin the original.

    If ``score_arm`` ever changes this expression the port is stale and the two
    views diverge silently — a sparse companion built on different columns than
    the harness's own.
    """
    src = open(os.path.join(_TE, "tools", "ff_rescore.py"),
               encoding="utf-8").read()
    literal = "sorted({max(0, int(round(K * q)) - 1) for q in (.25, .5, .75, 1.0)})"
    assert literal in src, (
        "ff_rescore.score_arm no longer contains the sparse-view expression this "
        "adapter ports. Re-sync adapters/navsim.py::_sparse_view.")
    for k in (4, 8, 20, 40):
        idx, contract = NS._sparse_view(k)
        assert idx == sorted({max(0, int(round(k * q)) - 1)
                              for q in (.25, .5, .75, 1.0)})
        assert contract == [i + 1 for i in idx]
        assert max(idx) == k - 1, "the sparse view must reach the final step"


def test_eid_is_none_on_purpose_and_tokens_are_preserved():
    win = make_win()
    assert win["eid"] is None, (
        "eid must be None so the harness's own interval guard fires "
        "(four_families.py:658-664) instead of a scene-token bootstrap")
    assert win["_navsim"]["scene_tokens"][0] == "tok0000"
    assert "cluster unit unsettled" in win["_navsim"]["eid_is_none_because"]


# ========================================================================== #
# 2. Coordinate frames — round trips with HAND-COMPUTED expectations         #
# ========================================================================== #
def test_world_to_ego_hand_computed():
    """No round trip: explicit numbers, so a self-consistent-but-wrong rotation
    cannot pass. Origin (10, 20) facing world +y (yaw = pi/2)."""
    o, yaw = np.array([10.0, 20.0]), math.pi / 2
    # 10 m further along world +y == 10 m AHEAD, 0 lateral
    got = NS.ego_frame_from_world(np.array([[10.0, 30.0]]), o, yaw)
    assert np.allclose(got, [[10.0, 0.0]], atol=1e-9), got
    # 5 m in world -x == 5 m to the LEFT when facing +y
    got = NS.ego_frame_from_world(np.array([[5.0, 20.0]]), o, yaw)
    assert np.allclose(got, [[0.0, 5.0]], atol=1e-9), got
    # 5 m in world +x == 5 m to the RIGHT
    got = NS.ego_frame_from_world(np.array([[15.0, 20.0]]), o, yaw)
    assert np.allclose(got, [[0.0, -5.0]], atol=1e-9), got


def test_world_to_ego_matches_driving_diagnostic_ego_exactly():
    """⭐ The port is checked against the harness's OWN ``_ego``, not a re-derivation."""
    import torch

    from driving_diagnostic import _ego as harness_ego
    rng = np.random.default_rng(7)
    dxy = rng.normal(0, 30, size=(5, 9, 2))
    yaw = rng.uniform(-math.pi, math.pi, size=(5,))
    ours = NS.ego_frame_from_world(dxy, np.zeros((5, 2)), yaw)
    theirs = harness_ego(torch.tensor(dxy), torch.tensor(yaw)[:, None]).numpy()
    assert np.allclose(ours, theirs, atol=1e-9), np.abs(ours - theirs).max()


def test_round_trip_navsim_world_frame_to_our_ego_frame():
    """A known trajectory expressed in a world frame comes back BIT-CLOSE to the
    ego-frame trajectory it was built from."""
    gt_ego, sp = arc_poses(n=6, seed=3)
    rng = np.random.default_rng(11)
    oxy = rng.normal(0, 500, size=(6, 2))
    oyaw = rng.uniform(-math.pi, math.pi, size=(6,))
    # ego -> world (the INVERSE rotation, by +yaw), then heading back to absolute
    c, s = np.cos(oyaw)[:, None], np.sin(oyaw)[:, None]
    wx = gt_ego[..., 0] * c - gt_ego[..., 1] * s + oxy[:, 0:1]
    wy = gt_ego[..., 0] * s + gt_ego[..., 1] * c + oxy[:, 1:2]
    world = np.stack([wx, wy, gt_ego[..., 2] + oyaw[:, None]], axis=-1)

    xy, head, meta = NS.poses_to_waypoints(world, frame="world",
                                           origin_included=False,
                                           origin_xy=oxy, origin_yaw=oyaw)
    assert np.allclose(xy, gt_ego[..., :2], atol=1e-8), np.abs(xy - gt_ego[..., :2]).max()
    dh = (head - gt_ego[..., 2] + math.pi) % (2 * math.pi) - math.pi
    assert np.allclose(dh, 0.0, atol=1e-8)
    assert meta["source_frame"] == "world"


def test_ego_frame_passthrough_drops_heading_and_keeps_xy():
    gt, _ = arc_poses(n=4)
    xy, head, meta = NS.poses_to_waypoints(gt, frame="ego", origin_included=False)
    assert xy.shape == (4, K, 2)
    assert np.allclose(xy, gt[..., :2])
    assert head is not None and head.shape == (4, K)
    assert meta["origin_row_stripped"] is False


def test_a_single_scene_is_promoted_to_one_window():
    gt, _ = arc_poses(n=1)
    xy, _, meta = NS.poses_to_waypoints(gt[0], frame="ego", origin_included=False)
    assert xy.shape == (1, K, 2) and meta["n_windows"] == 1


# ========================================================================== #
# 3. The origin row — DECLARED, then VERIFIED against the data               #
# ========================================================================== #
def test_origin_row_is_stripped_when_declared():
    withorig, _ = arc_poses(include_origin=True)
    assert withorig.shape[1] == K + 1
    xy, head, meta = NS.poses_to_waypoints(withorig, frame="ego",
                                           origin_included=True)
    assert xy.shape == (N, K, 2)
    assert meta["origin_row_stripped"] is True
    plain, _ = arc_poses(include_origin=False)
    assert np.allclose(xy, plain[..., :2])


def test_declaring_no_origin_when_the_t0_row_is_there_is_refused():
    """⛔ The failure this prevents: one extra leading zero row shifts every
    horizon by a tick and its zero-length first segment is absorbed by
    ``lateral.frenet_dense``'s degenerate-tangent fallback — a wrong number."""
    withorig, _ = arc_poses(include_origin=True)
    with pytest.raises(NS.FrameConventionError) as e:
        NS.poses_to_waypoints(withorig, frame="ego", origin_included=False)
    assert "t=0 state" in str(e.value)


def test_declaring_an_origin_that_is_not_there_is_refused():
    plain, _ = arc_poses(include_origin=False)
    with pytest.raises(NS.FrameConventionError) as e:
        NS.poses_to_waypoints(plain, frame="ego", origin_included=True)
    assert "does NOT carry the t=0 row" in str(e.value)


def test_origin_included_has_no_default_and_refuses_a_non_bool():
    plain, _ = arc_poses()
    with pytest.raises(TypeError):
        NS.poses_to_waypoints(plain, frame="ego")            # keyword-only, required
    with pytest.raises(NS.FrameConventionError) as e:
        NS.poses_to_waypoints(plain, frame="ego", origin_included=1)
    assert "no default" in str(e.value)


def test_world_frame_without_an_origin_is_refused():
    plain, _ = arc_poses()
    with pytest.raises(NS.FrameConventionError) as e:
        NS.poses_to_waypoints(plain, frame="world", origin_included=False)
    assert "origin_xy" in str(e.value)


def test_misaligned_origin_rows_are_refused():
    plain, _ = arc_poses(n=6)
    with pytest.raises(NS.FrameConventionError) as e:
        NS.poses_to_waypoints(plain, frame="world", origin_included=False,
                              origin_xy=np.zeros((3, 2)), origin_yaw=np.zeros(3))
    assert "do not align" in str(e.value)


# ========================================================================== #
# 4. ⭐ THE DELIBERATE REGRESSION ARMS                                        #
# ========================================================================== #
def test_frame_verifier_passes_on_a_correct_frame():
    """The control. Without it the two failure arms below prove only that the
    guard raises on everything."""
    gt, sp = arc_poses()
    ev = NS.verify_frame(gt[..., :2], dt_s=DT, speed=sp, heading=gt[..., 2])
    assert ev["axis_convention"]["verified"] is True
    assert ev["lateral_sign"]["status"] == "OK"
    assert ev["lateral_sign"]["agreement"] == 1.0
    assert ev["lateral_sign"]["n_turning_windows"] >= NS.SIGN_MIN_N


def test_regression_transposed_frame_is_caught():
    """⭐ MANDATORY ARM 1 — a transposed ``(y, x)`` dump.

    ``lateral.assert_axis_convention`` (lateral.py:170) must raise: axis0 no
    longer dominates axis1.
    """
    gt, sp = arc_poses()
    transposed = gt[..., :2][..., ::-1].copy()          # (x,y) -> (y,x)
    with pytest.raises(ValueError) as e:
        NS.verify_frame(transposed, dt_s=DT, speed=sp, heading=gt[..., 2])
    assert "axis convention violated" in str(e.value)


def test_regression_y_sign_flip_is_caught():
    """⭐ MANDATORY ARM 2 — a right-handed (y-RIGHT) frame.

    This is the flip that would invert every ``cross_bias_m`` in the LATERAL
    family. The adapter's sign guard must raise on it.
    """
    gt, sp = arc_poses()
    flipped = gt[..., :2].copy()
    flipped[..., 1] *= -1.0
    with pytest.raises(NS.FrameConventionError) as e:
        NS.verify_frame(flipped, dt_s=DT, speed=sp, heading=gt[..., 2])
    assert "LATERAL SIGN CONVENTION VIOLATED" in str(e.value)


def test_assert_axis_convention_is_blind_to_a_y_sign_flip():
    """⛔ MEASURED, not asserted: the harness verifier alone does NOT catch arm 2.

    This is the evidence that justifies a SECOND guard. If a future change makes
    ``assert_axis_convention`` catch a y-flip, this test fails and the adapter's
    docstring claim must be corrected — which is the point.
    """
    from taniteval import lateral as _lat
    gt, sp = arc_poses()
    clean = gt[..., :2].copy()
    flipped = clean.copy()
    flipped[..., 1] *= -1.0
    ev_clean = _lat.assert_axis_convention(clean, speed=sp, dt=DT)
    ev_flip = _lat.assert_axis_convention(flipped, speed=sp, dt=DT)
    assert ev_flip["verified"] is True, "expected the flip to pass this guard"
    assert ev_clean["mean_abs_along_final_m"] == ev_flip["mean_abs_along_final_m"]
    assert ev_clean["mean_abs_cross_final_m"] == ev_flip["mean_abs_cross_final_m"], (
        "negating y leaves |y| unchanged — that IS the blind spot")


def test_sign_guard_is_unavailable_without_an_independent_heading():
    """Honest boundary: with no heading channel the flip is UNDETECTABLE, and the
    guard says so with a reason instead of passing."""
    gt, sp = arc_poses()
    ev = NS.verify_frame(gt[..., :2], dt_s=DT, speed=sp, heading=None)
    assert ev["lateral_sign"]["status"] == "UNAVAILABLE"
    assert "circular" in ev["lateral_sign"]["reason"]
    assert ev["lateral_sign"]["n"] == 0


def test_sign_guard_refuses_a_straight_only_set_rather_than_passing_it():
    straight = np.zeros((10, K, 3))
    straight[..., 0] = np.arange(1, K + 1) * DT * 9.0
    ev = NS.assert_lateral_sign_convention(straight[..., :2], straight[..., 2])
    assert ev["status"] == "UNAVAILABLE"
    assert ev["n_turning_windows"] == 0
    assert "manufactured from an absent test" in ev["reason"]


def test_verify_frame_runs_inside_the_adapter_not_only_in_tests():
    """Audit seam 11 says 'call it in the adapter's own tests'; the adapter calls
    it itself so a caller cannot skip it."""
    gt, sp = arc_poses()
    bad = pred_from(gt)
    bad[..., 1] *= -1.0                          # flip GT's y via a flipped source
    gtf = gt.copy()
    gtf[..., 1] *= -1.0
    with pytest.raises(NS.FrameConventionError):
        NS.scenes_to_win(bad, gtf, frame="ego", origin_included=False,
                         dt_s=DT, ego_speed_mps=sp)


# ========================================================================== #
# 5. ⛔ THE COLUMN TRAP                                                       #
# ========================================================================== #
def test_reading_pdm_score_as_epdms_is_refused():
    with pytest.raises(NS.PdmScoreColumnError) as e:
        NS.read_epdms({"pdm_score": 0.812, "NC": 1.0})
    msg = str(e.value)
    assert "pdm_score` IS NOT THE EPDMS" in msg
    assert "14" in msg and "16" in msg
    assert "compute_final_scores" in msg


def test_the_score_column_is_read_and_pdm_score_is_never_relabelled():
    got = NS.read_epdms({"score": 0.5321, "pdm_score": 0.6011})
    assert got["value"] == pytest.approx(0.5321)
    assert got["column"] == "score"
    assert got["denominator"] == 16
    assert got["pdm_score_column_present"] is True
    assert got["not_the_epdms__pdm_score_value"] == pytest.approx(0.6011)
    assert not any("epdms" in k.lower() and "not_the_epdms" not in k
                   for k in got if k.startswith("pdm") or "pdm_score" in k)


def test_a_row_with_neither_column_is_refused():
    with pytest.raises(NS.NavSimAdapterError):
        NS.read_epdms({"NC": 1.0, "DAC": 1.0})


def test_variant_must_be_named():
    with pytest.raises(NS.NavSimAdapterError) as e:
        NS.read_epdms({"score": 0.5}, variant="EPDMS")
    assert "incomparable protocols" in str(e.value)


def test_v1_and_v2_denominators_are_12_and_16():
    assert NS.VARIANTS["PDMS_v1"]["denominator"] == 12
    assert NS.VARIANTS["EPDMS_v2"]["denominator"] == 16
    assert sum(NS.VARIANTS["EPDMS_v2"]["weighted"].values()) == 16
    assert sum(NS.VARIANTS["PDMS_v1"]["weighted"].values()) == 12


def test_missing_submetrics_are_absent_terms_not_zeros():
    sub = NS.submetrics_from_row({"score": 0.5, "NC": 1.0, "EP": 0.8})
    assert sub["NC"]["value"] == 1.0
    assert sub["EP"]["role"].startswith("weighted")
    assert sub["EC"]["status"] == "UNAVAILABLE"
    assert "MISSING TERM, not a zero" in sub["EC"]["reason"]
    assert set(sub["_missing_terms"]) >= {"DAC", "DDC", "TLC", "TTC", "LK", "HC", "EC"}


# ========================================================================== #
# 6. Refusals carry a reason AND an n                                        #
# ========================================================================== #
_DECLINED = {"UNAVAILABLE", "REFUSED", "N/A", "NA", "NOT_APPLICABLE", "BLOCKED"}


def _walk_status_objects(obj, path="$"):
    if isinstance(obj, dict):
        st = str(obj.get("status", "")).strip().upper()
        if st in _DECLINED:
            yield path, obj
        for k, v in obj.items():
            yield from _walk_status_objects(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_status_objects(v, f"{path}[{i}]")


def test_every_refusal_in_the_artifact_carries_a_reason_and_an_n():
    """⛔ ``tools/criteria_check.py:138-144``: a declined status WITHOUT a reason
    reads as ABSENT (a silent omission), not as REFUSED (a work item)."""
    art = NS.build_artifact(make_win(), tier="T1", split="navtest",
                            variant="EPDMS_v2", n_boot=40)
    found = list(_walk_status_objects(art))
    assert found, "expected refusals in a NavSim artifact"
    bad = []
    for path, obj in found:
        reason = str(obj.get("reason", "")).strip()
        n = obj.get("n", obj.get("n_windows_it_would_have_had"))
        if not reason or n is None:
            bad.append((path, sorted(obj)))
    assert not bad, f"refusals missing reason and/or n: {bad}"


def test_the_navsim_native_absences_are_named_per_family():
    ref = NS.navsim_native_family_refusals(450)
    for key in ("longitudinal_target_speed", "lateral_heading_curvature_yaw",
                "tactical_declared_decision",
                "strategic_decision_and_route_goal",
                "longitudinal_distance_keeping_continuous"):
        blk = ref[key]
        assert blk["status"] == "UNAVAILABLE"
        assert blk["n"] == 450
        assert len(blk["reason"]) > 60
        assert "NAVSIM_PROTOCOL.md" in blk["reason"]


def test_the_two_blocks_are_never_merged():
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    assert "four_families" in art and "navsim" in art["benchmark"]
    assert "families_absent_natively" in art["benchmark"]["navsim"]
    # our own instruments DO produce geometry numbers on NavSim trajectories,
    # which is exactly why the native-absence block must not be read as theirs
    assert art["four_families"]["lateral"]["cross_mae_m"] is not None
    assert art["benchmark"]["navsim"]["families_absent_natively"][
        "lateral_heading_curvature_yaw"]["status"] == "UNAVAILABLE"


def test_without_ground_truth_every_family_refuses_with_reason_and_n():
    gt, sp = arc_poses()
    win = NS.scenes_to_win(pred_from(gt), None, frame="ego",
                           origin_included=False, dt_s=DT, ego_speed_mps=sp)
    fam = NS.four_families_block(win, tier="T1")
    for k in ("longitudinal", "lateral", "tactical", "strategic"):
        assert fam[k]["status"] == "UNAVAILABLE"
        assert fam[k]["n"] == N and fam[k]["reason"]
    assert fam["_rule_satisfied"] is True     # clause 5
    assert fam["_complete"] is False


# ========================================================================== #
# 7. ⛔ THE ESTIMATOR IS LEFT OPEN                                            #
# ========================================================================== #
def test_no_confidence_interval_is_invented_anywhere():
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    assert art["estimator"]["interval"]["status"] == "UNAVAILABLE"
    assert "cluster unit unsettled" in art["estimator"]["interval"]["reason"]
    # and the harness's OWN interval blocks self-refuse because eid is None
    tac = art["four_families"]["tactical"]
    if tac.get("status") == "OK":
        for blk in ("lateral_decision", "longitudinal_decision",
                    "maneuver_5way_collapsed", "goal_setting"):
            ci = tac[blk]["ci"]
            assert ci["status"] == "UNAVAILABLE", f"{blk} produced an interval"


def test_no_numeric_lo_hi_pair_survives_in_the_artifact():
    """⛔ The strong version: NOTHING in the artifact carries a numeric interval.
    A single surviving [lo, hi] would be a scene-token bootstrap presented as
    decision-grade."""
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    hits = []

    def walk(o, p="$"):
        if isinstance(o, dict):
            if isinstance(o.get("lo"), (int, float)) and \
                    isinstance(o.get("hi"), (int, float)):
                hits.append(p)
            for k, v in o.items():
                walk(v, f"{p}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{p}[{i}]")
    walk(art)
    assert not hits, f"numeric intervals leaked into a NavSim artifact: {hits}"


def test_the_anti_echo_separation_is_never_claimed_without_a_cluster_unit():
    """⛔ The anti-echo block does NOT use a lo/hi shape — it reports
    ``estimator`` + ``separated`` — so the generic lo/hi sweep above would be
    BLIND to it. MEASURED: with ``eid=None`` every metric reports
    ``estimator='UNAVAILABLE'``, ``verdict='NO_INTERVAL'``, ``separated=False``,
    and ``longitudinal_claim_admissible=False``. That is the honest state: the
    point delta is real, the separation is undischarged."""
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    ae = art["four_families"]["longitudinal"]["anti_echo"]
    assert ae["status"] == "OK", "v0 was supplied, so the controls must run"
    assert ae["longitudinal_claim_admissible"] is False
    hv = ae["holdv0_baseline"]
    assert hv["verdict"] == "NO_INTERVAL"
    for name, m in hv["metrics"].items():
        assert m["estimator"] == "UNAVAILABLE", (name, m["estimator"])
        assert m["separated"] is False, name
        assert "episode-cluster bootstrap cannot be formed" in m["reason"], name


def test_the_forbidden_estimator_is_named_and_refused():
    r = NS.estimator_refusal(450)
    assert "overlapping_holdout_se" in r["_forbidden"]
    assert r["n"] == 450 and r["n_windows_it_would_have_had"] == 450


def test_to_ff_dump_is_shaped_like_load_dump_but_flags_the_open_unit():
    d = NS.to_ff_dump(make_win())
    for k in ("kind", "gt", "eid", "dt_s", "wp_steps", "source",
              "n_episodes", "arms"):
        assert k in d, f"load_dump shape is missing {k!r}"
    assert d["gt"].shape == (N, K, 2)
    assert len(d["eid"]) == N
    assert list(d["arms"])[0] == "navsim"
    assert d["_cluster_unit_unsettled"] is True
    assert d["_estimator_refusal"]["status"] == "UNAVAILABLE"


def test_ff_rescore_has_not_been_silently_given_a_navsim_branch():
    """⛔ Wiring it would enable a scene-token episode-cluster bootstrap behind a
    CLI flag. If someone adds the branch, this test must be revisited WITH the
    PI's decision on the cluster unit."""
    src = open(os.path.join(_TE, "tools", "ff_rescore.py"),
               encoding="utf-8").read()
    assert "navsim" not in src.lower(), (
        "ff_rescore.py now mentions navsim — the estimator question must be "
        "settled before that branch is admissible.")


# ========================================================================== #
# 8. Vision-only compliance and the tier stamp                               #
# ========================================================================== #
def test_inference_inputs_record_the_ego_channels_honestly():
    ii = NS.navsim_inference_inputs()
    assert ii["vision_only"] is False
    assert set(ii["ego_channels_consumed"]) == {
        "ego_velocity", "ego_acceleration", "ego_pose_history"}
    assert ii["inputs"]["driving_command"]["dim"] == 4, "devkit is 4-dim, not 3"
    assert "IMPLEMENT 4" in ii["inputs"]["driving_command"]["⚠️_cardinality"]
    assert "cannot verify" in " ".join(k for k in ii).lower() or \
        "⛔_framework_cannot_verify" in ii
    assert "no switch to remove it" in ii["⛔_framework_cannot_verify"]


def test_a_claimed_vision_only_arm_still_records_that_navsim_cannot_verify_it():
    ii = NS.navsim_inference_inputs(ego_velocity=False, ego_acceleration=False,
                                    ego_pose_history=False,
                                    claimed_abstention_from_ego=True)
    assert ii["vision_only"] is True
    assert ii["claimed_abstention_from_ego"] is True
    assert "enforced and evidenced ON OUR SIDE" in ii["⛔_framework_cannot_verify"]


def test_route_leak_check_defaults_to_an_explicit_unverified():
    r = NS.route_leak_check()
    assert r["status"] == "UNAVAILABLE"
    assert "route_roadblock_ids" in r["reason"]


def test_tier_is_required_and_has_no_default():
    win = make_win()
    with pytest.raises(TypeError):
        NS.four_families_block(win)                      # keyword-only, required
    with pytest.raises(NS.NavSimAdapterError) as e:
        NS.four_families_block(win, tier="T3")
    assert "no default" in str(e.value)


def test_tier_rationale_rejects_T0_and_flags_the_two_stage_question():
    tr = NS.tier_rationale()
    assert tr["recommended_single_stage"] == "T1"
    assert "T0 is WRONG" in tr["why"]
    assert "T2" in tr["⚠️_two_stage_is_arguably_T2"]
    assert "ONLY T0/T1" in tr["⚠️_two_stage_is_arguably_T2"]


def test_dt_has_no_default_and_must_be_positive():
    gt, sp = arc_poses()
    with pytest.raises(TypeError):
        NS.scenes_to_win(pred_from(gt), gt, frame="ego", origin_included=False)
    with pytest.raises(NS.NavSimAdapterError) as e:
        NS.scenes_to_win(pred_from(gt), gt, frame="ego", origin_included=False,
                         dt_s=0.0)
    assert "dt_s must be > 0" in str(e.value)


def test_dt_actually_reaches_the_rate_metrics():
    """⛔ Pins the x5 defect: the same array at dt 0.1 vs 0.5 must NOT give the
    same speed. four_families.py:1199-1205 records GT 12.4565 m/s read as
    62.9789 m/s from exactly this."""
    gt, sp = arc_poses(dt=0.5)
    pr = pred_from(gt)
    slow = NS.four_families_block(
        NS.scenes_to_win(pr, gt, frame="ego", origin_included=False, dt_s=0.5,
                         verify=False), tier="T1", n_boot=20)
    fast = NS.four_families_block(
        NS.scenes_to_win(pr, gt, frame="ego", origin_included=False, dt_s=0.1,
                         verify=False), tier="T1", n_boot=20)
    assert slow["longitudinal"]["dt_s"] == 0.5 and fast["longitudinal"]["dt_s"] == 0.1
    r = fast["longitudinal"]["speed_mae_mps"] / slow["longitudinal"]["speed_mae_mps"]
    # ⚠️ NOT exactly 5: four_families.py:268 emits round(..., 4), so the ratio of
    # two rounded values carries ~1e-4 of residue (MEASURED 5.000806). The pin
    # that matters is 5x vs 1x — a dt that never reached the metric gives 1.0.
    assert r == pytest.approx(5.0, rel=1e-3), r
    # positions are dt-invariant, which is what makes the defect survivable
    assert fast["longitudinal"]["along_mae_m"] == \
        slow["longitudinal"]["along_mae_m"]


def test_split_must_be_a_known_one():
    with pytest.raises(NS.NavSimAdapterError) as e:
        NS.build_artifact(make_win(), tier="T1", split="navtest_v2")
    assert "incomparable protocols" in str(e.value)


# ========================================================================== #
# 9. The artifact against the REAL criteria checker                          #
# ========================================================================== #
def _criteria_check():
    spec = importlib.util.spec_from_file_location(
        "criteria_check_under_test",
        os.path.join(_REPO, "tools", "criteria_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _registry():
    import json
    with open(os.path.join(_REPO, "products", "P7-TanitEval",
                           "CRITERIA_REGISTRY.json"), encoding="utf-8") as f:
        return json.load(f)


def test_artifact_is_in_scope_and_tier_stamped_for_criteria_check():
    cc, reg = _criteria_check(), _registry()
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    scope, why = cc.scope_of(art, reg)
    assert scope == cc.IN_SCOPE, why
    tier, raw = cc.resolve_tier(art, reg)
    assert tier == "T1", f"tier unresolved (raw={raw!r})"


def test_navsim_criteria_resolve_as_present_or_refused_never_absent():
    """⛔ The distinction criteria_check exists to make: a reasoned refusal is a
    WORK ITEM; a silent omission is a defect. None of the five NavSim criteria
    may land in ABSENT."""
    cc, reg = _criteria_check(), _registry()
    art = NS.build_artifact(
        make_win(), tier="T1", split="navtest",
        epdms=NS.read_epdms({"score": 0.4412}),
        submetrics=NS.submetrics_from_row(
            {"NC": 1.0, "DAC": 1.0, "DDC": 1.0, "TLC": 1.0, "EP": 0.83,
             "TTC": 1.0, "LK": 1.0, "HC": 1.0, "EC": 0.5}),
        n_boot=40)
    rows = {}
    for crit in reg["benchmarks"]["navsim"]["criteria"]:
        rows[crit["id"]] = cc.classify(art, crit)
    absent = {k: v for k, v in rows.items() if v[0] == cc.ABSENT}
    assert not absent, f"criteria landed in ABSENT: {absent}"
    assert rows["navsim.score"][0] == cc.PRESENT
    assert rows["navsim.route_leak_check"][0] == cc.REFUSED


def test_leak_guards_and_hygiene_resolve():
    cc, reg = _criteria_check(), _registry()
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    for guard in reg["leak_guards"]["guards"]:
        state, detail = cc.classify(art, guard)
        if guard["id"] in ("vision_only_inference", "goal_situation_disjoint"):
            assert state == cc.PRESENT, f"{guard['id']}: {state} {detail}"
    est = next(c for c in reg["artifact_hygiene"]["criteria"]
               if c["id"] == "hyg.estimator_named")
    state, detail = cc.classify(art, est)
    assert state == cc.REFUSED, (state, detail)     # named, and honestly open


def test_the_four_families_keys_resolve_for_the_geometry_families():
    cc, reg = _criteria_check(), _registry()
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    got = {}
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        for crit in reg["families"][fam]["criteria"]:
            got[crit["id"]] = cc.classify(art, crit)[0]
    # computable from trajectory geometry
    for cid in ("long.target_speed", "long.along_track_error", "long.progress",
                "lat.cross_track", "lat.heading", "lat.curvature", "lat.yaw_rate"):
        assert got[cid] == cc.PRESENT, (cid, got[cid])
    # genuinely open on NavSim, and REFUSED (a work item), never ABSENT
    for cid in ("long.distance_keeping", "strat.decision", "strat.route_goal"):
        assert got[cid] == cc.REFUSED, (cid, got[cid])
    assert cc.ABSENT not in got.values(), {k: v for k, v in got.items()
                                           if v == cc.ABSENT}


def test_strategic_carries_the_navsim_specific_reason_not_the_physicalai_one():
    art = NS.build_artifact(make_win(), tier="T1", split="navtest", n_boot=40)
    s = art["four_families"]["strategic"]
    assert s["status"] == "UNAVAILABLE" and s["n"] == N
    assert "driving_command` is an INPUT" in s["navsim_specific_reason"]
    assert "buildable WORK ITEM" in s["navsim_specific_reason"]


# ---------------------------------------------------------------------------
# The protocol-declaration contract (added 2026-08-23 after the release gate
# scored RG-07/RG-08/RG-12 FAIL on our best T1 artifacts purely because nothing
# RECORDED what the model consumed — not because anything was violated).
# ---------------------------------------------------------------------------
import numpy as _np_pc
from taniteval import four_families as _ff_pc


def _tiny_win(n=24, k=12):
    rng = _np_pc.random.default_rng(0)
    return {"pred": rng.normal(size=(n, k, 2)).cumsum(1),
            "gt": rng.normal(size=(n, k, 2)).cumsum(1),
            "eid": [f"ep{i // 4}" for i in range(n)], "dt_s": 0.1}


def test_an_undeclared_protocol_is_written_out_not_left_silent():
    """⛔ The emitter must never GUESS these. An artifact that invented
    `vision_only: true` would be manufacturing compliance, which is worse than
    the gap — so the absence is emitted explicitly."""
    fam = _ff_pc.all_families(_tiny_win(), tier="T1", n_boot=30)
    undeclared = fam["_protocol_undeclared"]
    for field in ("vision_only", "goal_situation_disjoint", "corpus", "parity_key"):
        assert field in undeclared, f"{field} must be reported UNDECLARED, not omitted"
    assert "NOT assumed compliant" in fam["_protocol"]["vision_only"]


def test_a_declared_protocol_leaves_nothing_undeclared():
    fam = _ff_pc.all_families(
        _tiny_win(), tier="T1", n_boot=30,
        protocol={"inference_inputs": "camera only", "vision_only": True,
                  "goal_source": "predicted goal point", "goal_situation_disjoint": True,
                  "corpus": "physicalai-val-0c5f7dac3b11",
                  "parity_key": "e438721ae894 / f09e44db"})
    assert fam["_protocol_undeclared"] == []
    assert fam["_protocol"]["vision_only"] is True


def test_DELIBERATE_REGRESSION_a_partial_declaration_still_reports_the_rest():
    """The arm that keeps the contract honest: declaring SOME fields must not
    silence the others."""
    fam = _ff_pc.all_families(_tiny_win(), tier="T1", n_boot=30,
                              protocol={"vision_only": True})
    assert "vision_only" not in fam["_protocol_undeclared"]
    assert "parity_key" in fam["_protocol_undeclared"]
    assert "goal_situation_disjoint" in fam["_protocol_undeclared"]
