"""test_kdraw_e1_r8.py -- pins the K-draw analysis (kdraw_e1_r8.py) and the roller helpers (kdraw_roll_r8.py).

    cd <PKG>/code && PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;.' python -X utf8 -m pytest -q test_kdraw_e1_r8.py

Groups.  A: pure functions, expectations are LITERALS worked out by hand (never an expression over the code under test).  B: the estimator --
SHUFFLED WINDOW PAIRING MUST CHANGE THE E1 CELL (deliberate-regression arm).  C: the whole analysis in a SUBPROCESS on the real banked seeds 0/1 plus six
synthetic extra seeds: K=1 reproduces the banked single-draw cells EXACTLY (to 4 dp, literals), K=2 reproduces ENS2, and two deliberate-regression arms
(an ensemble that returns the first draw; a MEDOID that returns the mean) must be CAUGHT -- by the C-K2 control (VOID) and by the medoid/ensemble identity respectively.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in ("C:/lgs/Lfb47a58376/stack", "C:/lgs/Lfb47a58376/taniteval", str(HERE)):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)
import kdraw_e1_r8 as K  # noqa: E402
import kdraw_roll_r8 as R  # noqa: E402

ARCH = Path(K.ARCHIVE) / "step30000ema_unc"
PY = sys.executable


# ======================================================================================================== A. pure functions (literals)
def _one(*pts):
    """a K=len(pts) stack for ONE window with 4 identical waypoints per draw: shape [K, 1, 4, 2]"""
    return np.array([[[list(p)] * 4] for p in pts], dtype=np.float32)


def test_ensemble_of_one_is_the_draw_bit_for_bit():
    x = np.random.default_rng(1).normal(size=(1, 7, 4, 2)).astype(np.float32)
    assert np.array_equal(K.ensemble_mean(x), x[0])


def test_symmetric_two_modes_ensemble_error_zero_medoid_error_delta():
    # draws P+(0,+1) and P-(0,+1) around GT = P = (3,0): the mean is EXACTLY the GT; a real single plan is 1.0 m away (ADE 1.0 exactly)
    st = _one((3.0, 1.0), (3.0, -1.0))
    gt = np.array([[[3.0, 0.0]] * 4], dtype=np.float32)
    ens, med = K.ensemble_mean(st), K.medoid_plan(st)
    assert float(np.linalg.norm(ens - gt, axis=-1).mean()) == 0.0
    assert float(np.linalg.norm(med - gt, axis=-1).mean()) == 1.0           # tie -> lowest index (draw 0), a REAL plan, not the blend
    assert not np.array_equal(med, ens)


def test_medoid_picks_the_draw_closest_to_the_mean():
    # mean = (1.6667, 1.7); distances 2.3806 (d0), 2.3105 (d1), 4.6906 (d2) -> index 1
    st = _one((0.0, 0.0), (0.0, 0.1), (5.0, 5.0))
    assert int(K.medoid_index(st)[0]) == 1
    assert np.array_equal(K.medoid_plan(st)[0], st[1, 0])


def test_spread_is_the_largest_pairwise_endpoint_distance():
    st = _one((0.0, 0.0), (3.0, 0.0), (0.0, 0.5))
    assert float(K.spread_2s(st)[0]) == pytest.approx(np.hypot(3.0, 0.5), abs=1e-6)      # the (3,0)-(0,0.5) pair: 3.0415
    assert float(K.spread_2s(_one((0.0, 0.0), (0.5, 0.0)))[0]) == pytest.approx(0.5, abs=1e-7)


def _cell(d, lo, hi):
    return {"delta": d, "lo": lo, "hi": hi}


def test_arm_verdict_branches():
    a, b = _cell(-0.016, -0.030, -0.002), _cell(-0.015, -0.029, -0.001)
    assert K.arm_verdict(a, b, 0.001)["verdict"] == "PASS"
    v = K.arm_verdict(a, b, 0.010)                                          # separated on both, but |delta| <= 2 x floor
    assert v["verdict"] == "PARTIAL" and v["only_miss_is_not_proven_rule"] is True and v["n_sets_separated"] == 2
    v = K.arm_verdict(a, _cell(-0.010, -0.030, 0.003), 0.001)               # delta < 0 on both, one set not separated
    assert v["verdict"] == "PARTIAL" and v["n_sets_separated"] == 1 and v["only_miss_is_not_proven_rule"] is False
    assert K.arm_verdict(a, _cell(0.001, -0.02, 0.02), 0.001)["verdict"] == "FAIL"
    assert K.arm_verdict(a, None, 0.001)["verdict"] == "PENDING"
    assert K.arm_verdict(a, b, None)["verdict"] == "PENDING"
    assert K.arm_verdict(_cell(-0.01, -0.02, 0.0), b, 0.001)["verdict"] == "PARTIAL"      # hi == 0 is NOT separated (strict)


def test_overall_verdict_and_blend_downgrade():
    P, Q, F, N = ({"verdict": x} for x in ("PASS", "PARTIAL", "FAIL", "PENDING"))
    assert K.overall_verdict(P, F, False)["overall"] == "PASS"
    assert K.overall_verdict(P, F, True)["overall"] == "PASS WITH BLEND FLAG"           # ENS-only pass + flag
    assert K.overall_verdict(F, P, True)["overall"] == "PASS"                           # a MEDOID pass cannot be a blend artefact
    assert K.overall_verdict(Q, F, False)["overall"] == "PARTIAL"
    assert K.overall_verdict(F, F, False)["overall"] == "FAIL"
    assert K.overall_verdict(N, F, False)["overall"] == "PENDING"


def test_blend_flag_needs_a_strictly_positive_lower_bound():
    assert K.blend_flag({"delta": 0.001, "lo": 0.0001, "hi": 0.002}) is True
    assert K.blend_flag({"delta": 0.001, "lo": 0.0, "hi": 0.002}) is False
    assert K.blend_flag({"status": "UNDERPOWERED n=3"}) is False


def test_overlapping_sets_are_refused():
    K.assert_disjoint(K.SET_A, K.SET_B)
    with pytest.raises(SystemExit):
        K.assert_disjoint((0, 1, 2, 3), (3, 4, 5, 6))


def test_literal_comparison_tolerance_is_four_dp():
    assert K.lit_equal(_cell(-0.01384, -0.02794, 0.00121), (-0.0138, -0.0279, 0.0012))
    assert not K.lit_equal(_cell(-0.0150, -0.0279, 0.0012), (-0.0138, -0.0279, 0.0012))


def test_select_prefix_takes_whole_clips_in_file_order():
    man = {"episodes": [{"file_index": 1, "clip_id": "b"}, {"file_index": 0, "clip_id": "a"}, {"file_index": 2, "clip_id": "c"}]}
    sub, order = R.select_prefix(man, {"a": [1] * 35, "b": [1] * 34, "c": [1] * 10}, 50)
    assert order == ["a", "b"] and sum(len(v) for v in sub.values()) == 69
    with pytest.raises(ValueError):
        R.select_prefix(man, {"a": [1], "b": [1], "c": [1]}, 50)


def test_compare_rows_reports_bit_identity_and_the_first_difference():
    ref = {(0, 7): {"os": np.zeros((4, 2), np.float32)}, (0, 12): {"os": np.zeros((4, 2), np.float32)}}
    same = {(0, 7): {"os": np.zeros((4, 2), np.float32)}}
    c = R.compare_rows(same, ref)
    assert c["os_bit_identical_all"] is True and c["n_common"] == 1 and c["n_new_only"] == 0
    diff = {(0, 7): {"os": np.zeros((4, 2), np.float32)}, (0, 12): {"os": np.full((4, 2), 0.25, np.float32)}}
    c = R.compare_rows(diff, ref)
    assert c["os_bit_identical_all"] is False and c["arms"]["os"]["max_abs_diff"] == 0.25 and c["arms"]["os"]["first_differing_window"] == [0, 12]


# ======================================================================================================== B. the estimator
def test_shuffled_window_pairing_changes_the_e1_cell():
    """DELIBERATE-REGRESSION ARM.  Per-window ADE is dominated by window difficulty shared by the arm and the echo; the paired cell exploits that.  Break the
    pairing (permute the echo across windows) and the cell MUST change -- the point estimate is permutation-invariant, so it is the interval that must widen."""
    import pair_read_r8 as PR
    rng = np.random.default_rng(7)
    n_ep, per = 40, 50
    eid = np.repeat(np.arange(n_ep), per)
    diff = rng.gamma(2.0, 0.15, size=n_ep * per)                              # shared window difficulty
    arm = 0.30 + diff + 0.01 * rng.normal(size=diff.size)
    echo = 0.31 + diff + 0.01 * rng.normal(size=diff.size)
    true = PR.paired_cell(arm, echo, eid, 500, 0)
    shuf = PR.paired_cell(arm, echo[rng.permutation(echo.size)], eid, 500, 0)
    w_true, w_shuf = true["hi"] - true["lo"], shuf["hi"] - shuf["lo"]
    assert true["hi"] < 0 < shuf["hi"] and true["delta"] == pytest.approx(-0.01, abs=2e-3)   # separated when paired, NOT when the pairing is destroyed
    assert w_shuf > 5.0 * w_true
    assert (true["lo"], true["hi"]) != (shuf["lo"], shuf["hi"])
    assert true["delta"] == pytest.approx(shuf["delta"], abs=1e-9)             # the mean of the difference is permutation-invariant: the CELL changed through its interval


# ======================================================================================================== C. the whole analysis, in a subprocess
def _env(mut: str = "") -> dict:
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "KDRAW_TEST_MUTATION": mut,
              "PYTHONPATH": f"C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;{HERE}"})
    return e


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    if not (ARCH / "dump_s1" / "manifest.json").exists():
        pytest.skip(f"SKIP-REASON: banked archive absent ({ARCH}); the integration group cannot run on this machine")
    root = tmp_path_factory.mktemp("kd") / "step30000ema_unc"
    src = ARCH / "dump_s1"
    for s in range(2, 8):
        d = root / f"dump_s{s}"
        d.mkdir(parents=True)
        rng = np.random.default_rng(1000 + s)
        for f in sorted(src.glob("ep*.npz")):
            z = dict(np.load(f))
            z["os"] = (z["os"] + rng.normal(scale=0.2, size=z["os"].shape)).astype(np.float32)
            np.savez(d / f.name, **z)
        (d / "refcv6_extras.npz").write_bytes((src / "refcv6_extras.npz").read_bytes())
        (d / "KDRAW_VERIFY.json").write_text(json.dumps({"ok": True, "synthetic": "test fixture"}), encoding="utf-8")
    return root


def _run(root, out, mut=""):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "kdraw_e1_r8.py"), "run", "--tree", "step30000ema_unc", "--kd-root", str(root), "--out-dir", str(out)],
                       env=_env(mut), capture_output=True, text=True, timeout=900)
    j = out / "KDRAW_E1.json"
    assert j.exists(), f"analysis wrote no JSON (judge by the artifact): {r.stdout[-400:]} {r.stderr[-800:]}"
    return json.loads(j.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base_run(fake_root, tmp_path_factory):
    return _run(fake_root, tmp_path_factory.mktemp("out_base"))


def test_k1_reproduces_the_banked_single_draw_cells_exactly(base_run):
    for s, lit in K.LIT_K1.items():
        c = base_run["singles"][str(s)]["E1"]
        assert (c["n_windows"], c["n_episodes"]) == (4159, 138)
        assert (round(c["delta"], 4), round(c["lo"], 4), round(c["hi"], 4)) == lit
    assert base_run["controls"]["C-K1"]["PASS"] is True and base_run["controls"]["C-K2"]["PASS"] is True
    assert (round(base_run["arms"]["ENS-2"]["E1"]["delta"], 4), round(base_run["arms"]["ENS-2"]["E1"]["hi"], 4)) == (-0.0138, 0.0012)


def test_whole_analysis_structure_and_controls(base_run):
    assert base_run["void_controls_failed"] == [] and base_run["controls"]["C-MASK"]["PASS"] is True
    assert base_run["controls"]["C-MASK"]["sha"] == K.MASK_SHA
    for n in ("ENS-4A", "ENS-4B", "MEDOID-4A", "MEDOID-4B", "ENS-8", "MEDOID-8"):
        a = base_run["arms"][n]
        assert a["E1"]["n_windows"] == 4159 and "NR_family_rollup" in a and set(a["NR_family_rollup"]) >= {"ADE", "LONGITUDINAL", "LATERAL"}
    assert base_run["bars"]["ENS-4"]["verdict"] in ("PASS", "PARTIAL", "FAIL") and base_run["overall"]["overall"] in ("PASS", "PASS WITH BLEND FLAG", "PARTIAL", "FAIL")
    # 8 seeds, independent noise around the same centre: the 8-mean must be strictly better than the average single draw (triangle inequality, strict on noise)
    singles = np.mean([base_run["singles"][str(s)]["mean_ade_compliant"] for s in range(8)])
    assert base_run["arms"]["ENS-8"]["mean_ade_compliant"] < singles
    assert base_run["arms"]["ENS-4A"]["triangle_ok_every_window"] is True
    md = base_run["mode_disagreement"]["4A"]
    assert 0.0 <= md["frac_spread_gt_1m_compliant"] <= 1.0 and md["seeds"] == [0, 1, 2, 3]
    assert base_run["arms"]["ENS-4A"]["E1"] != base_run["arms"]["MEDOID-4A"]["E1"]


def test_deliberate_regression_ensemble_returning_the_first_draw_is_void(fake_root, tmp_path):
    r = _run(fake_root, tmp_path / "o1", "ens_first")
    assert r["test_mutation"] == "ens_first"
    assert "C-K2" in r["void_controls_failed"] and r["overall"]["overall"].startswith("VOID")


def test_deliberate_regression_medoid_returning_the_mean_is_visible(fake_root, tmp_path, base_run):
    r = _run(fake_root, tmp_path / "o2", "medoid_is_mean")
    assert r["test_mutation"] == "medoid_is_mean"
    assert r["arms"]["MEDOID-4A"]["E1"]["delta"] == r["arms"]["ENS-4A"]["E1"]["delta"]            # the mutation collapses the two arms ...
    assert base_run["arms"]["MEDOID-4A"]["E1"]["delta"] != base_run["arms"]["ENS-4A"]["E1"]["delta"]  # ... which the honest run keeps apart
