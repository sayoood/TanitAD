"""SPEC AMENDMENT A7, items A7.3 (the detection packs are banked) and A7.4 (the inference-seed draw is reported as ONE
draw), on LITERAL inputs and on the BANKED refcv7 G0 artifacts. No GPU.

    REFCV6_REPO=D:/Projects/TanitAD PYTHONPATH=D:/Projects/TanitAD/stack python -m pytest -q test_g0_a7_reports.py

A7.3 and A7.4 are REPORTED, NEVER GATING. The tests pin what they must do instead:
  * A7.3: a present pack ROUND-TRIPS bit-exactly; a MISSING pack is REPORTED (never silent); no raw episode id
    reaches the file;
  * A7.4: the seed-group F test reproduces the numbers `raw/PREREG_SEED_GROUP_CHECK_50400.md` QUOTES for step 5,000
    (sd 0.00212 / 0.01308, F = 38.1, one-sided p = 3.1e-5, Levene p = 0.018) -- written below as LITERALS from that
    document, not read back from the code -- and the step-50,400 read (B), F(15, 7) = 6.61, p = 0.0088, quoted in
    SPEC A7; the cross-checkpoint correlation reproduces r = 0.719 / 0.604 / 0.866 (n = 192) of the diagnosis.
Each carries a DELIBERATE-REGRESSION arm that must go RED.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RAW = HERE.parent / "raw"


def G():
    import g0_refcv7
    return g0_refcv7


def _load(rel):
    p = RAW / rel
    assert p.exists(), f"banked artifact missing: {p}"          # a missing artifact is a FAILURE, never a skip
    return json.load(open(p, encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------- #
# A7.3 -- the packs                                                                                               #
# ------------------------------------------------------------------------------------------------------------- #
EP0 = 900000000000                    # a stand-in for the trainer's stable episode id (an int); its sha12 below
SHA12_EP0, SHA12_EP3 = "897e956ac828", "61772d42cedf"          # sha256(str(ep))[:12], computed independently


def _pack(i, n_slot=4, n_gt=None):
    rng = np.random.default_rng(100 + i)
    n_gt = i % 3 if n_gt is None else n_gt                   # 0, 1 or 2 GT rows: a ragged GT block, incl. empty
    return {"ep": EP0 + i, "_batch": i // 2,
            "logit": rng.normal(size=n_slot).astype(np.float32), "xy": rng.normal(size=(n_slot, 2)).astype(np.float32),
            "cls": rng.integers(0, 10, n_slot).astype(np.int16), "cls_corr": rng.integers(0, 10, n_slot).astype(np.int16),
            "matched": np.zeros(n_slot, bool), "exempt": np.zeros(n_slot, bool),
            "gt_xy": rng.normal(size=(n_gt, 2)).astype(np.float32), "gt_cls": rng.integers(0, 10, n_gt).astype(np.int16),
            "pos": rng.integers(0, 2, n_gt).astype(bool), "ign": rng.integers(0, 2, n_gt).astype(bool)}


def _arms(n=6, drop=()):
    arms = {}
    for arm in ("s0", "fp32_s0"):
        arms[arm] = {hd: [_pack(i) for i in range(n)] for hd in ("agent", "box3d") if (arm, hd) not in drop}
    return arms


INRUN = {"eval_agent_n_windows": 6.0, "eval_box3d_n_windows": 6.0}


def test_a_present_pack_round_trips_exactly_and_no_raw_id_is_banked(tmp_path):
    g = G()
    arms = _arms()
    rep = g.bank_detection_packs(arms, INRUN, ("agent", "box3d"), tmp_path / "g0.packs.npz")
    assert rep["status"] == "ALL BANKED" and rep["gaps"] == [] and rep["gating"] is False
    assert sorted(rep["arms"]) == ["fp32_s0.agent", "fp32_s0.box3d", "s0.agent", "s0.box3d"]
    assert all(c["status"] == "BANKED" and c["n_packs"] == 6 for c in rep["arms"].values())
    z = tmp_path / "g0.packs.npz"
    assert rep["npz"]["path"] == str(z) and z.exists() and rep["npz"]["bytes"] == z.stat().st_size
    for arm in ("s0", "fp32_s0"):
        for hd in ("agent", "box3d"):
            back = g.unpack_detection_packs(z, arm, hd)
            assert len(back) == 6
            for i, (b, o) in enumerate(zip(back, arms[arm][hd])):
                assert np.array_equal(b["logit"], o["logit"]) and np.array_equal(b["xy"], o["xy"])
                assert np.array_equal(b["cls"], o["cls"])
                assert np.array_equal(b["gt_xy"], o["gt_xy"]) and np.array_equal(b["gt_cls"], o["gt_cls"])
                assert np.array_equal(b["gt_pos"], o["pos"]) and np.array_equal(b["gt_ign"], o["ign"])
                assert b["batch"] == i // 2
    # the window keys are sha12 only: ep EP0+0 and EP0+3 hash to the literals; the raw ints appear nowhere
    back = g.unpack_detection_packs(z, "s0", "agent")
    assert back[0]["ep_sha12"] == SHA12_EP0 and back[3]["ep_sha12"] == SHA12_EP3
    with np.load(str(z), allow_pickle=False) as zz:
        for name in zz.files:
            a = zz[name]
            if a.dtype.kind == "U":
                assert all(str(EP0 + i) not in str(v) for v in a.ravel() for i in range(6)), name
            elif a.dtype.kind in "iu":
                assert not np.isin(a, [EP0 + i for i in range(6)]).any(), name
        meta = json.loads(str(zz["meta"]))
    assert meta["schema"] == "g0-a7-packs/1" and meta["cells"]["s0.agent"] == 6
    assert "no raw clip id" in meta["ep_sha12"]


def test_a_missing_arm_is_REPORTED_not_silent(tmp_path):
    g = G()
    arms = _arms()
    arms["fp32_s0"] = {}                                       # the numerics arm never ran (--no-a6)
    rep = g.bank_detection_packs(arms, INRUN, ("agent", "box3d"), tmp_path / "p.npz",
                                 not_run={"fp32_s0": "--no-a6: the fp32_s0 arm did not run"})
    assert rep["status"] == "GAPS: fp32_s0.agent=MISSING, fp32_s0.box3d=MISSING"
    assert [x["cell"] for x in rep["gaps"]] == ["fp32_s0.agent", "fp32_s0.box3d"]
    assert rep["arms"]["fp32_s0.agent"] == {"n_packs": 0, "expected_n_windows_in_run_row": 6.0, "status": "MISSING",
                                            "why": "--no-a6: the fp32_s0 arm did not run"}
    assert rep["arms"]["s0.agent"]["status"] == "BANKED"
    # what IS banked is readable; what is missing raises -- it never reads back as an empty list
    assert len(g.unpack_detection_packs(tmp_path / "p.npz", "s0", "box3d")) == 6
    with pytest.raises(KeyError):
        g.unpack_detection_packs(tmp_path / "p.npz", "fp32_s0", "agent")


def test_an_arm_that_ran_but_produced_no_pack_for_one_head_is_REPORTED(tmp_path):
    rep = G().bank_detection_packs(_arms(drop=[("s0", "box3d")]), INRUN, ("agent", "box3d"), tmp_path / "p.npz")
    assert rep["arms"]["s0.box3d"]["status"] == "MISSING"
    assert rep["arms"]["s0.box3d"]["why"] == "the arm ran but produced no detection pack for this head"
    assert rep["status"] == "GAPS: s0.box3d=MISSING"


def test_a_count_that_disagrees_with_the_in_run_row_is_REPORTED_but_still_banked(tmp_path):
    rep = G().bank_detection_packs(_arms(n=6), {"eval_agent_n_windows": 7.0, "eval_box3d_n_windows": 6.0},
                                   ("agent", "box3d"), tmp_path / "p.npz")
    assert rep["arms"]["s0.agent"]["status"] == "COUNT_MISMATCH"
    assert rep["arms"]["s0.agent"]["why"] == "6 packs banked, the in-run row counts 7.0 windows"
    assert rep["arms"]["s0.box3d"]["status"] == "BANKED"
    assert rep["status"] == "GAPS: s0.agent=COUNT_MISMATCH, fp32_s0.agent=COUNT_MISMATCH"
    assert rep["npz"]["bytes"] > 0


def test_nothing_to_bank_is_reported_and_writes_no_file(tmp_path):
    g = G()
    rep = g.bank_detection_packs({"s0": {}, "fp32_s0": {}}, INRUN, ("agent", "box3d"), tmp_path / "p.npz")
    assert rep["npz"] is None and not (tmp_path / "p.npz").exists()
    assert rep["status"].startswith("GAPS: s0.agent=MISSING") and rep["status"].endswith("; npz NOT WRITTEN")
    none = g.bank_detection_packs({}, INRUN, (), tmp_path / "q.npz")
    assert none["status"] == "NO DETECTION HEADS (nothing to bank)" and not (tmp_path / "q.npz").exists()


def test_a_pack_that_cannot_be_built_is_an_ERROR_cell_not_a_crash_and_not_silent(tmp_path):
    arms = _arms()
    del arms["s0"]["agent"][2]["gt_xy"]                       # a malformed pack
    rep = G().bank_detection_packs(arms, INRUN, ("agent", "box3d"), tmp_path / "p.npz")
    assert rep["arms"]["s0.agent"]["status"] == "ERROR" and "gt_xy" in rep["arms"]["s0.agent"]["why"]
    assert rep["status"] == "GAPS: s0.agent=ERROR"


def _assert_reports_the_missing_arm(rep):
    assert rep["gaps"] and rep["gaps"][0]["cell"] == "fp32_s0.agent" and rep["status"].startswith("GAPS")


def test_REGRESSION_a_silent_banker_fails_the_missing_pack_assertion(tmp_path):
    arms = _arms()
    arms["fp32_s0"] = {}
    honest = G().bank_detection_packs(arms, INRUN, ("agent", "box3d"), tmp_path / "p.npz",
                                      not_run={"fp32_s0": "arm did not run"})
    _assert_reports_the_missing_arm(honest)                    # GREEN for the shipped banker
    silent = {"status": "ALL BANKED", "gaps": [], "arms": {"s0.agent": {"status": "BANKED"}}}   # what silence looks like
    with pytest.raises(AssertionError):
        _assert_reports_the_missing_arm(silent)


# ------------------------------------------------------------------------------------------------------------- #
# A7.4 -- the seed-group F test                                                                                   #
# ------------------------------------------------------------------------------------------------------------- #
#: the numbers PREREG_SEED_GROUP_CHECK_50400.md quotes for step 5,000, at the precision it prints them
PREREG = {"sd_0_7": (0.00212, 5e-6), "sd_8_23": (0.01308, 5e-6), "F": (38.1, 0.05), "p_one_sided": (3.1e-5, 5e-7),
          "levene_p": (0.018, 5e-4)}


def _by_seed_5000():
    """Step 5,000 as the prereg's table is built: seeds 0-7 from the G0, seeds 8-23 from the diagnostic."""
    g5, d5 = _load("step5000/g0.json"), _load("g0diag_step5000/diag.json")
    by = {int(s): v for s, v in g5["by_seed"].items() if int(s) < 8}
    for s in range(8, 24):
        a = d5["arms"][f"seed{s}"]
        by[s] = {"row": a["row"], "per_batch": [{"traj": x} for x in a["per_batch_traj"]]}
    return by


def _assert_prereg_numbers(t):
    for k, (want, tol) in PREREG.items():
        assert abs(t[k] - want) <= tol, (k, t[k], want)


def test_the_F_test_reproduces_the_step_5000_numbers_the_prereg_quotes():
    rep = G().seed_group_report(_by_seed_5000())
    assert rep["status"] == "COMPUTED" and rep["gating"] is False and rep["term"] == "eval_traj"
    t = rep["test_5dp"]                                # the form the prereg's table is built from (5-dp row values)
    _assert_prereg_numbers(t)
    assert (t["n_0_7"], t["n_8_23"], t["df"]) == (8, 16, [15, 7])
    assert rep["VERDICT"] == "B"                       # p = 3.1e-5 < 0.01, same direction
    f = rep["test_full"]                               # full precision agrees with the 5-dp form to well under 1 %
    assert abs(f["F"] - t["F"]) / t["F"] < 0.01 and abs(f["sd_8_23"] - t["sd_8_23"]) < 1e-5
    assert rep["rounding_control_max_abs"] < 5e-6


def test_REGRESSION_swapped_groups_or_a_two_sided_p_do_not_reproduce_the_prereg_numbers():
    import seed_group_check as SGC
    by = _by_seed_5000()
    lo = [float(by[s]["row"]["eval_traj"]) for s in range(8)]
    hi = [float(by[s]["row"]["eval_traj"]) for s in range(8, 24)]
    _assert_prereg_numbers(SGC.test(lo, hi))           # GREEN: the registered form var(8-23) / var(0-7)
    with pytest.raises(AssertionError):
        _assert_prereg_numbers(SGC.test(hi, lo))       # the groups the wrong way round: F = 0.026
    two_sided = dict(SGC.test(lo, hi))
    two_sided["p_one_sided"] *= 2
    with pytest.raises(AssertionError):
        _assert_prereg_numbers(two_sided)


def test_the_step_50400_read_quoted_in_SPEC_A7_is_B_with_F_6_61_p_0_0088():
    g = _load("step50400/g0.json")
    rep = G().seed_group_report({int(s): v for s, v in g["by_seed"].items()})
    assert rep["status"] == "COMPUTED" and rep["VERDICT"] == "B"
    t = rep["test_full"]
    assert abs(t["F"] - 6.61) <= 0.005 and abs(t["p_one_sided"] - 0.0088) <= 5e-5
    assert t["df"] == [15, 7]


def test_the_seed_group_report_needs_24_seeds_and_survives_zero_variance():
    g = G()
    short = g.seed_group_report({s: {"per_batch": [{"traj": 1.0}], "row": {"eval_traj": 1.0}} for s in range(8)})
    assert short["status"] == "NOT COMPUTED" and "needs inference seeds 0..23" in short["why"]
    flat = g.seed_group_report({s: {"per_batch": [{"traj": 0.5}] * 2, "row": {"eval_traj": 0.5}} for s in range(24)})
    assert flat["status"] == "NOT COMPUTABLE" and flat["why"] == "zero variance among the seeds 0-7 row means"
    broken = g.seed_group_report({s: {"per_batch": [{"traj": 0.5}], "row": {}} for s in range(24)})
    assert broken["status"] == "ERROR" and broken["gating"] is False


# ------------------------------------------------------------------------------------------------------------- #
# A7.4 -- the per-(seed, batch) correlation against an earlier checkpoint's G0                                     #
# ------------------------------------------------------------------------------------------------------------- #
def test_the_correlation_reproduces_the_diagnosis_numbers_r_0_719_0_604_0_866_n_192():
    g = G()
    c5 = g.traj_cells(_by_seed_5000())
    c30 = g.traj_cells({int(s): v for s, v in _load("step30000/g0.json")["by_seed"].items()})
    c50 = g.traj_cells({int(s): v for s, v in _load("step50400/g0.json")["by_seed"].items()})
    for (a, b, r, rs) in ((c5, c30, 0.719, 0.80), (c5, c50, 0.604, 0.68), (c30, c50, 0.866, 0.82)):
        out = g.seed_draw_correlation(a, b)
        assert out["status"] == "COMPUTED" and out["n_cells"] == 192 and out["n_seeds_common"] == 24
        assert round(out["r_cell"], 3) == r
        assert out["p_cell"] <= 1.9e-20                               # the diagnosis: p <= 1.8e-20
        assert round(out["r_seed_row_mean"], 2) == rs


# a hand-checkable case: the earlier artifact's deviations are -2 x the current ones -> r = -1 EXACTLY
D = {0: [0.10, -0.20, 0.05], 1: [-0.10, 0.30, 0.00], 2: [0.20, 0.10, -0.15], 3: [-0.05, -0.10, 0.25],
     4: [0.00, 0.20, -0.10], 5: [-0.15, -0.30, -0.05]}


def _cells(sign, base):
    return {s: [base[b] + sign * 2.0 * v[b] if sign else base[b] + v[b] for b in range(3)] for s, v in D.items()}


def _assert_perfect_anticorrelation(out):
    assert out["status"] == "COMPUTED" and out["n_cells"] == 18 and out["n_seeds_common"] == 6
    assert out["r_cell"] == pytest.approx(-1.0, abs=1e-9)


def test_the_correlation_on_a_hand_checkable_case():
    g = G()
    cur = {s: [0.5 + v[b] for b in range(3)] for s, v in D.items()}
    earlier = {s: [0.9 - 2.0 * v[b] for b in range(3)] for s, v in D.items()}
    out = g.seed_draw_correlation(cur, earlier)
    _assert_perfect_anticorrelation(out)
    assert out["deviation"].startswith("traj(seed, batch) - mean over the artifact's own seeds")
    sub = g.seed_draw_correlation(cur, {s: earlier[s] for s in (0, 1, 2)})
    assert sub["status"] == "COMPUTED" and sub["n_cells"] == 9 and sub["n_seeds_common"] == 3      # common cells only
    few = g.seed_draw_correlation(cur, {s: earlier[s] for s in (0, 1)})
    assert few["status"] == "NOT COMPUTABLE" and few["why"] == "only 2 common inference seeds"
    ragged = g.seed_draw_correlation(cur, {s: v[:2] for s, v in earlier.items()})
    assert ragged["status"] == "NOT COMPARABLE" and ragged["why"] == "3 vs 2 batches per seed"


def test_REGRESSION_a_scrambled_batch_order_destroys_the_correlation_and_is_caught():
    g = G()
    cur = {s: [0.5 + v[b] for b in range(3)] for s, v in D.items()}
    earlier = {s: [0.9 - 2.0 * v[b] for b in range(3)] for s, v in D.items()}
    _assert_perfect_anticorrelation(g.seed_draw_correlation(cur, earlier))               # GREEN
    scrambled = {s: list(reversed(v)) for s, v in earlier.items()}                       # batch b <-> batch 2-b
    with pytest.raises(AssertionError):
        _assert_perfect_anticorrelation(g.seed_draw_correlation(cur, scrambled))


def test_a_constant_artifact_gives_a_null_correlation_not_NaN():
    flat = {s: [0.5, 0.5] for s in range(5)}
    out = G().seed_draw_correlation(flat, flat)
    assert out["status"] == "COMPUTED" and out["r_cell"] is None and out["n_cells"] == 10


def test_earlier_g0_discovery_uses_comparable_artifacts_only_and_lists_every_skip(tmp_path):
    g = G()
    cells = {str(s): {"per_batch": [{"traj": 0.5 + 0.01 * ((s * 7 + b * 3) % 5)} for b in range(2)]} for s in range(6)}
    rec = {"ckpt_md5": "cur", "perm_sha256": "P"}
    cur = {s: {"per_batch": v["per_batch"]} for s, v in cells.items()}
    for name, md5, perm in (("step100", "old", "P"), ("step200", "old2", "OTHER"), ("step300", "cur", "P")):
        d = tmp_path / name
        d.mkdir()
        (d / "g0.json").write_text(json.dumps({"step": int(name[4:]), "ckpt_md5": md5, "perm_sha256": perm,
                                               "by_seed": cells}), encoding="utf-8")
    out = g.seed_draw_reports(rec, cur, root=tmp_path)
    assert out["status"] == "COMPUTED" and out["gating"] is False
    assert [(p["earlier_step"], p["n_cells"]) for p in out["pairs"]] == [(100, 12)]
    assert sorted((Path(s["path"]).parent.name, s["why"]) for s in out["skipped"]) == [
        ("step200", "not the same 128 windows (perm_sha256 differs)"), ("step300", "the same checkpoint")]
    none = g.seed_draw_reports(rec, cur, root=tmp_path / "nowhere")
    assert none["status"] == "NO EARLIER G0 (comparable) FOUND" and none["pairs"] == [] and none["skipped"] == []
    explicit = g.seed_draw_reports(rec, cur, earlier_arg=str(tmp_path / "step100" / "g0.json"))
    assert [p["earlier_step"] for p in explicit["pairs"]] == [100]
    unreadable = g.seed_draw_reports(rec, cur, earlier_arg=str(tmp_path / "missing.json"))
    assert unreadable["pairs"] == [] and unreadable["skipped"][0]["why"].startswith("FileNotFoundError")
