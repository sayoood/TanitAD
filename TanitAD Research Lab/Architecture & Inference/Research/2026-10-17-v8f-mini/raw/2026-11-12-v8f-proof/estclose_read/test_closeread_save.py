"""tests of closeread_save.py (card PREREG_V8F_PROOF_ADD79). The module under test is taken from the folder in ECR_SRC_DIR (default: this folder); the mutation runner points it at a mutated copy."""
import json
import os
import sys

import numpy as np
import pytest

SRC = os.environ.get("ECR_SRC_DIR") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import closeread_save as S   # noqa: E402

CF, CL = S.CF, S.CL
NE = 10


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w")
    fits, scores, _ = CF.make_synth(seed=5, n_fit=36, n_score=36)
    lists = CF.synth_lists(fits)
    dd = tmp / "dump"
    dd.mkdir()
    for k in fits:
        np.savez(str(dd / ("reader_s%d_fit.npz" % k)), **fits[k])
        np.savez(str(dd / ("reader_s%d_score.npz" % k)), **scores[k])
        json.dump(lists[k], open(str(dd / ("reader_s%d_lists.json" % k)), "w", encoding="utf-8"))
    cells = {}
    for name, (arm, H, ck) in S.ESTS.items():
        rs = []
        for k in (0, 1):
            sr = CF.build_rows(scores[k], H)
            e, _ = CF.run_cell(arm, CF.build_rows(fits[k], H), sr, NE, train_ids=lists[k]["train_ids"])
            rs.append(CF.pearson(e, sr["y"]))
        cells[ck] = {"r_per_seed": rs}
    add78 = {"cells": cells}
    out = str(tmp / "out")
    lines = []
    rc, rec = S.run_save(str(dd), out, add78, [0, 1], NE, log=lines.append)
    return {"tmp": tmp, "dd": str(dd), "scores": scores, "add78": add78, "out": out, "rc": rc, "rec": rec, "lines": lines}


def test_good_run_passes_and_writes(world):
    assert world["rc"] == 0 and world["rec"]["e0_all_pass"] is True
    for n in S.ESTS:
        for k in (0, 1):
            assert os.path.isfile(os.path.join(world["out"], "closest_%s_s%d.npz" % (n, k)))
            assert os.path.isfile(os.path.join(world["out"], "est_score_%s_s%d.npz" % (n, k)))


def test_saved_forward_equals_fit_tool_estimate(world):
    sc = world["scores"][0]
    sr = CF.build_rows(sc, 25)
    fr = None
    E = CL.ClosingEstimator.load(os.path.join(world["out"], "closest_L25_s0.npz"))
    got = E.apply(sr["X"])
    fits, _, _ = CF.make_synth(seed=5, n_fit=36, n_score=36)
    ref, _ = CF.run_cell("L", CF.build_rows(fits[0], 25), sr, NE, train_ids=CF.synth_lists(fits)[0]["train_ids"])
    assert np.max(np.abs(got - ref)) <= 1e-6


def test_net_file_matches_h_and_arm(world):
    E = CL.ClosingEstimator.load(os.path.join(world["out"], "closest_N10_s1.npz"))
    assert E.arm == "N" and E.H == 10 and len(E.nets) == 3 and E.seed == 1


def test_table_nan_where_history_is_not_valid(world):
    T = np.load(os.path.join(world["out"], "est_score_L25_s0.npz"))
    fr = np.asarray(world["scores"][0]["frame"])
    assert len(T["est"]) == len(fr)
    assert np.isnan(T["est"][fr < 4 + 24]).all()
    assert np.isfinite(T["est"][fr >= 4 + 24 + 1]).all()


def test_table_is_the_estimate(world):
    T = np.load(os.path.join(world["out"], "est_score_N10_s0.npz"))
    E = CL.ClosingEstimator.load(os.path.join(world["out"], "closest_N10_s0.npz"))
    sel, X = S.window_rows(world["scores"][0], 10)
    assert np.allclose(T["est"][sel], E.apply(X), atol=1e-12)
    assert np.abs(T["est"][sel]).max() > 1e-3


def test_planted_files(world):
    X = np.random.default_rng(0).normal(0, 1, (4, 10, 3))
    assert np.allclose(CL.ClosingEstimator.load(os.path.join(world["out"], "planted_zero_H10.npz")).apply(X), 0.0)
    assert np.allclose(CL.ClosingEstimator.load(os.path.join(world["out"], "planted_const2_H10.npz")).apply(X), 2.0)
    assert np.allclose(CL.ClosingEstimator.load(os.path.join(world["out"], "planted_readerclos_H10.npz")).apply(X), X[:, -1, 2])


def test_existing_out_dir_refused(world):
    assert S.run_save(world["dd"], world["out"], world["add78"], [0, 1], NE, log=lambda s: None)[0] == 2


def test_planted_r_difference_fails_e0_and_writes_nothing(world):
    bad = {"cells": {ck: {"r_per_seed": [r + 0.01 for r in v["r_per_seed"]]} for ck, v in world["add78"]["cells"].items()}}
    out = str(world["tmp"] / "bad_r")
    lines = []
    rc, rec = S.run_save(world["dd"], out, bad, [0, 1], NE, log=lines.append)
    assert rc == 1 and not os.path.exists(out) and any(l.startswith("FAIL e0c") for l in lines)


def test_r_inside_tolerance_passes(world):
    ok = {"cells": {ck: {"r_per_seed": [r + 0.004 for r in v["r_per_seed"]]} for ck, v in world["add78"]["cells"].items()}}
    out = str(world["tmp"] / "ok_r")
    rc, _ = S.run_save(world["dd"], out, ok, [0, 1], NE, log=lambda s: None)
    assert rc == 0


def test_planted_normalisation_fault_fails_e0b(world):
    out = str(world["tmp"] / "bad_mu")
    lines = []
    rc, _ = S.run_save(world["dd"], out, world["add78"], [0, 1], NE, fault="mu", log=lines.append)
    assert rc == 1 and not os.path.exists(out) and any(l.startswith("FAIL e0b") for l in lines)


def test_output_prints_no_r_value(world):
    assert all(l.startswith(("PASS", "FAIL", "ZZ")) for l in world["lines"])


def test_record_has_hashes(world):
    rec = world["rec"]
    assert len(rec["inputs_sha256"]) == 6 and all(len(v) == 64 for v in rec["outputs_sha256"].values()) and len(rec["outputs_sha256"]) >= 11
