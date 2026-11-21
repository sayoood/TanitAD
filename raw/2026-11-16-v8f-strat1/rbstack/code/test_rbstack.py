"""test_rbstack.py -- run the WHOLE file.  CPU only.  Synthetic data.  Mutants (plain non-nested OOF probabilities, a drifted reference) must turn the checks RED;
the result is written to out/MUTATIONS_RBSTACK.json."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rbstack as R  # noqa: E402

MUT = {}


def synth(n_clip=40, per=30, ncls=3, seed=0):
    rng = np.random.RandomState(seed)
    clip = np.repeat(np.arange(n_clip), per)
    t = np.tile(np.arange(per), n_clip) + 3
    N = len(clip)
    y = rng.randint(0, ncls, N)
    X = rng.rand(N, 5)
    X[:, 0] = (y + rng.rand(N)) // 1  # weak signal in the "state" column
    M = {"tr_clip": clip, "tr_t": t, "tr_y": y, "tr_X": X, "sidx": np.arange(0, N, 2), "tr_clip_ids": np.array([f"c{i}" for i in range(n_clip)])}
    return M, y, np.ones(N, bool)


def stub_factory(M, y_all, ncls):
    """a probe stub that returns class-prior-plus-noise probabilities and is honest about the rows it was given"""
    def f(seed):
        def fit(fit_idx, hold_idx, pred_idx):
            yv = y_all[M["sidx"]]
            prior = np.bincount(yv[fit_idx], minlength=ncls) / len(fit_idx)
            rng = np.random.RandomState(seed + len(fit_idx))
            return R.softmax(np.log(prior + 1e-6) + rng.randn(len(pred_idx), ncls)), {"E": 3, "hold": {"3": 0.5}}
        return fit
    return f


def run(plan, tmp):
    M, y, ok = synth()
    return R.run_target("t", (0,), 3, y, ok, M, None, tmp, lambda *a: None, fit_factory=stub_factory(M, y, 3), write_stage=False, plan=plan)


@pytest.fixture(scope="module")
def tmp():
    d = tempfile.mkdtemp(dir=HERE + "/smoke", prefix="t_")
    yield d
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(HERE + "/out", exist_ok=True)
    json.dump(MUT, open(HERE + "/out/MUTATIONS_RBSTACK.json", "w"), indent=1)


def test_plan_counts():
    p = R.plan_fits()
    assert sum(x["kind"] == "outer" for x in p) == 5 and sum(x["kind"] == "inner" for x in p) == 10
    assert all(len(x["fit_folds"]) == 4 for x in p if x["kind"] == "outer") and all(len(x["fit_folds"]) == 3 for x in p if x["kind"] == "inner")


def test_nesting_passes_on_nested():
    assert R.assert_nested(R.provenance(R.plan_fits()))
    MUT["nested_plan"] = "PASS (as required)"


def test_nesting_red_on_plain_oof_mutant():
    with pytest.raises(R.NestingError):
        R.assert_nested(R.provenance(R.plan_plain_oof()))
    MUT["plain_oof_probabilities_mutant"] = "RED (NestingError raised, as required)"


def test_nesting_red_when_test_probe_saw_test_fold():
    prov = R.provenance(R.plan_fits())
    prov[2][2] = ((0, 1, 2, 3), "outerX")
    with pytest.raises(R.NestingError):
        R.assert_nested(prov)
    MUT["test_probe_saw_test_fold_mutant"] = "RED (NestingError raised, as required)"


def test_pipeline_nested_end_to_end(tmp):
    res, refs, fits = run(None, tmp + "/a")
    r = res[0]
    assert r["controls_seed"]["K-NEST"] == "PASS"
    # every probe that feeds a stack for outer fold k saw neither k nor the row's own fold, from the REAL index sets given to the probe
    for k, d in r["provenance_seen_folds"].items():
        for j, seen in d.items():
            assert int(k) not in seen if j == k else (int(j) not in seen and int(k) not in seen)
    assert r["arms"]["EDV"]["accuracy"]["n_windows"] == r["n_windows"]


def test_pipeline_red_on_plain_oof_mutant(tmp):
    with pytest.raises(R.NestingError):
        run(R.plan_plain_oof(), tmp + "/b")
    MUT["pipeline_with_plain_oof_plan"] = "RED (stack_arms raised NestingError, as required)"


def test_shuffle_keeps_class_marginals():
    P = R.softmax(np.random.RandomState(1).randn(200, 3))
    S = R.shuffle_rows(P, 0)
    assert not np.array_equal(P, S)
    assert np.array_equal(np.sort(P, axis=0), np.sort(S, axis=0))
    assert np.array_equal(np.bincount(P.argmax(1), minlength=3), np.bincount(S.argmax(1), minlength=3))
    MUT["shuffle_marginals"] = "PASS (rows permuted, multiset and argmax counts equal)"


def test_repro_comparator_flags_1e6_drift():
    assert R.repro_ok(0.6406453009961022, 0.6406453009961022)
    assert not R.repro_ok(0.6406453009961022 + 1e-6, 0.6406453009961022)
    MUT["repro_comparator_1e-6_drift"] = "RED (flagged, as required)"


def test_reading_rules():
    d = lambda delta, lo, sep: {"delta": delta, "lo": lo, "separated": sep}  # noqa: E731
    assert R.reading_of(d(0.05, 0.02, True)) == "STACK-CARRIES"
    assert R.reading_of(d(0.02, 0.01, True)) == "STACK-WEAK"
    assert R.reading_of(d(0.05, -0.01, False)) == "STACK-SILENT"
    assert R.combine_readings(["STACK-CARRIES", "STACK-SILENT"]) == "STACK-SILENT"
    assert R.combine_readings(["STACK-CARRIES", "STACK-WEAK"]) == "STACK-WEAK"


def test_softmax_rows_sum_to_one():
    assert np.allclose(R.softmax(np.random.RandomState(0).randn(10, 3)).sum(1), 1, atol=1e-6)
