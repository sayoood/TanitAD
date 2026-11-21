"""Tests of the SLOWWM read package (PREREG_V8F_SLOWWM). CPU. Run:
    PYTHONPATH="<tree>/stack;<tree>/stack/scripts;<tree>/taniteval" CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 python -m pytest tests/test_read_slowwm.py -q --basetemp <dir that exists>
The heavy tests (R35H identity, planted basis arm) read the banked R35H s0 checkpoint and frames; they skip when free RAM is under 8 GB."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
READ = os.path.normpath(os.path.join(HERE, ".."))
THEORY = os.path.normpath(os.path.join(READ, "..", ".."))
TREE = "F:/TanitAD-scratch/slowwm/tree"
os.environ.setdefault("LADDER_READ_CODE_ROOT", TREE)
os.environ.setdefault("WMACT_CODE_ROOT", TREE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
for p in (os.path.join(TREE, "stack"), os.path.join(TREE, "stack", "scripts"), os.path.join(TREE, "taniteval"), os.path.join(THEORY, "slowwm"), os.path.join(THEORY, "ladder_plantest"), READ):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import lp_slowwm as S  # noqa: E402
import report_slowwm as R  # noqa: E402
import controls_read_slowwm as C  # noqa: E402
import run_lp_slowwm as RUN  # noqa: E402
import ia0_slowwm as IA  # noqa: E402

PREREG = os.path.join(THEORY, "slowwm", "PREREG_V8F_SLOWWM.md")


def test_tanitad_comes_from_the_f_tree():
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(TREE.lower()), tanitad.__file__


def test_prereg_hash_is_the_frozen_one():
    assert hashlib.sha256(open(PREREG, "rb").read()).hexdigest() == "7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca"


# ------------------------------------------------------------------------------------------------------------- naming, jobs, dispatch
def test_arm_names_and_kinds():
    assert [S.kind_of(n) for n in ("SLOW_s0", "PCA-B_s1", "SLOW_rep_s2", "R35H_s1")] == ["slow", "pca", "slow", "adapter"]
    assert S.seed_of("SLOW_rep_s2") == 2 and S.FRAMES_SEED[2] == 0
    with pytest.raises(ValueError):
        S.kind_of("X_s0")


def test_runner_jobs_and_paths():
    a = RUN.parse_arm("SLOW_s0|C:/r|C:/b.npz|C:/u.npz|all")
    m = RUN.parse_arm("R35H_s0|C:/r|||meta")
    J = RUN.jobs_a([a, m])
    kinds = [j[0][0] for j in J]
    assert kinds.count("meta") == 2 and kinds.count("p1") == 4 and kinds.count("tune") == 12      # 2 forms x (2 goals, 6 lambdas); the meta-only arm plans nothing
    assert RUN.path(("p1", "SLOW_s0", "c1", "true")).endswith("p1_SLOW_s0_c1_flat1_true.npz")
    assert RUN.path(("score", "SLOW_s0", "c3", "rand", 0.1)).endswith("score_SLOW_s0_c3_flat1_rand_L0.1.npz")
    with pytest.raises(AssertionError):
        RUN.parse_arm("SLOW_s0|C:/r|C:/b.npz")


def test_clip_z1_dispatch_is_undone():
    before = S.LP.clip_z1
    with S.basis_clip_z1():
        assert S.LP.clip_z1 is not before
    assert S.LP.clip_z1 is before


def test_ia0_dispatch_is_passthrough_for_an_adapter_model_and_undone():
    o_z, o_e = IA.L.z_in, IA.L.encode_clip

    class W:
        wm_basis = "adapter"
        wm_pool = "mean"

    class M:
        wm = W()
    fr = {"zmean": ["clip0-mean"], "clips": ["c"]}
    with IA.basis_read():
        assert IA.L.z_in(M(), fr, 0) == "clip0-mean"            # an adapter model reads the token mean, as lr_lib does
        W.wm_basis = "slow"
        with pytest.raises(S.SH.BasisMismatch):                 # a basis model on frames without ubasis refuses
            IA.L.z_in(M(), fr, 0)
    assert IA.L.z_in is o_z and IA.L.encode_clip is o_e


# ------------------------------------------------------------------------------------------------------------- outcome logic and the report on planted runs
def test_control_d_outcome_logic_and_mutants():
    r = C.control_d()
    assert r["pass_"], r
    assert all(v["red"] for v in r["red_arms"].values())


def test_decide_needs_both_seeds_and_a_floor():
    T = C.planted_tables()
    assert R.decide(*T["one-seed-only"])["outcomes"] == ["NO-EFFECT"]
    m, pl, fl = T["SLOW-PLANS"]
    big = dict(fl, LONGITUDINAL=0.6)                                              # a floor above the 0.5 gap: nothing beats R35H beyond it
    assert R.decide(m, pl, big)["outcomes"] == ["NO-EFFECT"]                     # LATERAL margins are all 0 in the planted table, so no family has a winner


def _write_runs(d, w, n_ep=12, per=6):
    """Planted units, noise-free: plan = HOLD + w * (logged - HOLD), G-RAND = HOLD. ``w`` maps arm name -> quality in [0, 1]."""
    rng = np.random.default_rng(0)
    n = n_ep * per
    eid = np.array([f"e{i // per:02d}" for i in range(n)])
    A = rng.standard_normal((n, 15, 2)) * np.array([1.0, 0.1]) + np.array([0.0, 0.0])
    v0 = np.abs(rng.standard_normal(n)) * 5 + 5
    tune = np.nonzero(np.isin(eid, [f"e{i:02d}" for i in range(n_ep // 2)]))[0]
    score = np.setdiff1d(np.arange(n), tune)
    hist_ok = (np.arange(n) % per) != 0
    hold = np.repeat(A[:, :1], 12, axis=1)
    for arm, q in w.items():
        np.savez(f"{d}/meta_{arm}.npz", A=A, v0=v0, eid=eid, t=np.arange(n), tune_idx=tune, score_idx=score, hist_ok=hist_ok, std_raw=np.array([1.0, 0.1]), ch1=np.array(1.0), basis_kind=np.array("x"), basis_sha=np.array(""))
        plan = (hold + q * (A[:, :12] - hold)).astype(np.float32)
        for form in R.CTXS:
            ok = np.ones(n, bool) if form == "c1" else hist_ok
            for goal, P_ in (("true", plan), ("rand", hold.astype(np.float32))):
                idx = np.nonzero(ok)[0]
                np.savez(f"{d}/p1_{arm}_{form}_flat1_{goal}.npz", idx=idx, plan_raw=P_[idx])
                sidx = np.array([i for i in score if ok[i]])
                np.savez(f"{d}/score_{arm}_{form}_flat1_{goal}_L0.npz", idx=sidx, plan_raw=P_[sidx])
            json.dump({"choice": {"LONGITUDINAL": 0.0, "LATERAL": 0.0}, "tune_table": {}}, open(f"{d}/choice_{arm}_{form}_flat1.json", "w"))


def _quality(slow, pca, r35h, rep=None):
    w = {"SLOW_s0": slow, "SLOW_s1": slow, "SLOW_rep_s2": slow if rep is None else rep, "PCA-B_s0": pca, "PCA-B_s1": pca, "R35H_s0": r35h, "R35H_s1": r35h}
    return w


@pytest.mark.parametrize("w,expected", [
    (_quality(0.9, 0.3, 0.3), ["SLOW-PLANS"]),
    (_quality(0.6, 0.6, 0.1), ["BASIS-ONLY"]),
    (_quality(0.3, 0.3, 0.3), ["NO-EFFECT"]),
    (_quality(0.1, 0.6, 0.3), ["SLOW-COSTS"]),
])
def test_report_on_planted_runs_reads_each_outcome(tmp_path, w, expected):
    _write_runs(str(tmp_path), w)
    res = R.build(str(tmp_path), str(tmp_path), "traj")
    assert res["combined"]["status"] == "CELLS-AGREE", res["combined"]
    assert res["combined"]["agree"] == expected, res["combined"]
    cell = res["cells"]["PLANTEST|c1"]["SLOW_s0"]["LONGITUDINAL"]
    assert cell["tier"] == "T0" and cell["n_windows"] == 72 and cell["n_episodes"] == 12 and "EC-BOOT" in cell["estimator"]
    assert res["tables"]["PLANTEST|c1"]["floor"]["LONGITUDINAL"] == 0.0               # the replicate is the s0 arm itself
    if expected == ["SLOW-PLANS"]:
        assert cell["rows"]["traj"]["label"] == "PLANNABLE"
    md = R.markdown(dict(res, card="x", deciding_space="traj", deciding_metrics=R.SPACES["traj"], margin_definition="m", controls={"status": "PASS", "usable": True}, outcome=expected,
                         families_not_read=R.FAMILY_STATUS))
    assert "PAUSED (PI item 25)" in md and "n = 0" in md


def test_report_floor_is_the_replicate_gap_and_a_gap_inside_it_does_not_count(tmp_path):
    w = _quality(0.9, 0.3, 0.3, rep=0.0)                                      # a replicate with no effect at all: the floor equals m(SLOW_s0), larger than SLOW's 0.6 gap to R35H
    _write_runs(str(tmp_path), w)
    res = R.build(str(tmp_path), str(tmp_path), "traj")
    fl = res["tables"]["PLANTEST|c1"]["floor"]
    assert fl["LONGITUDINAL"] > 0.0
    m = res["tables"]["PLANTEST|c1"]["m"]
    assert abs(m["SLOW"][0]["LONGITUDINAL"] - res["cells"]["PLANTEST|c1"]["SLOW_rep_s2"]["LONGITUDINAL"]["rows"]["traj"]["m"]) == pytest.approx(fl["LONGITUDINAL"])
    assert "SLOW-PLANS" not in res["combined"]["cells"]["PLANTEST|c1"]


def test_report_missing_arms_read_incomplete(tmp_path):
    w = _quality(0.9, 0.3, 0.3)
    w.pop("PCA-B_s1")
    _write_runs(str(tmp_path), w)
    res = R.build(str(tmp_path), str(tmp_path), "traj")
    assert res["combined"]["status"] == "INCOMPLETE" and res["decided"]["PLANTEST|c1"]["outcomes"] == []


def test_final_outcome_is_unusable_when_a_control_failed_or_did_not_run(tmp_path):
    comb = {"status": "CELLS-AGREE", "agree": ["SLOW-PLANS"]}
    assert R.final_outcome({"usable": True}, comb) == ["SLOW-PLANS"]
    assert R.final_outcome(R.controls_status(str(tmp_path / "absent.json")), comb).startswith("UNUSABLE")
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"all_pass": False, "a": {"pass": False}}))
    assert R.final_outcome(R.controls_status(str(bad)), comb).startswith("UNUSABLE")


def _comb(cells):
    keys = [f"{r}|{c}" for r in R.READS for c in R.CTXS]
    return {"status": "CELLS-AGREE" if len({tuple(v) for v in cells}) == 1 else "CELLS-DISAGREE", "cells": dict(zip(keys, cells))}


def test_amend_slowwm_1_frozen_hash_matches_the_code():
    p = os.path.join(THEORY, "slowwm", "AMEND_SLOWWM_1.md")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h == "5d2270d40e5ae86f0b2b552bf5e9f1cc2b244e97e16f4ccbe400e77920df1549"
    assert h[:8] in open(os.path.join(READ, "report_slowwm.py"), encoding="utf-8").read()


def test_committed_all_four_cells_one_outcome():
    for o in R.ORDER:
        got = R.committed(_comb([[o]] * 4))
        assert got["outcome"] == o and got["next_step_of"] == o


def test_committed_disagreeing_cells_take_the_least_favourable_next_step():
    got = R.committed(_comb([["SLOW-PLANS"], ["SLOW-PLANS"], ["SLOW-PLANS"], ["NO-EFFECT"]]))
    assert got["outcome"] == "CELLS-DISAGREE" and got["next_step_of"] == "NO-EFFECT"
    got = R.committed(_comb([["SLOW-PLANS"], ["BASIS-ONLY"], ["SLOW-PLANS"], ["SLOW-PLANS"]]))
    assert got["outcome"] == "CELLS-DISAGREE" and got["next_step_of"] == "BASIS-ONLY"


def test_committed_mixed_and_none_of_four_cells():
    got = R.committed(_comb([["SLOW-PLANS", "SLOW-COSTS"]] * 4))      # MIXED in every cell is never an agreed single outcome
    assert got["outcome"] == "CELLS-DISAGREE" and got["next_step_of"] == "SLOW-COSTS"
    assert set(got["cell_labels"].values()) == {"MIXED"}
    got = R.committed(_comb([[]] * 2 + [["SLOW-PLANS"]] * 2))          # NONE-OF-FOUR counts as NO-EFFECT
    assert got["outcome"] == "CELLS-DISAGREE" and got["next_step_of"] == "NO-EFFECT"
    assert R.committed({"status": "INCOMPLETE", "cells": {}}) is None


def test_committed_red_arm_majority_rule_would_differ():
    # a majority rule (3 of 4) would commit SLOW-PLANS here; the amendment commits NO-EFFECT's step. Guards against a silent relaxation of A4.
    got = R.committed(_comb([["SLOW-PLANS"]] * 3 + [["NO-EFFECT"]]))
    assert got["next_step_of"] != "SLOW-PLANS"


# ------------------------------------------------------------------------------------------------------------- controls main(): merge of an earlier error
def _planted_controls(monkeypatch, tmp_path, stale, raise_in=None):
    out = tmp_path / "CONTROLS_READ_SLOWWM.json"
    json.dump(stale, open(out, "w", encoding="utf-8"))
    monkeypatch.setattr(C, "OUT", str(out))

    def ok(tag):
        def f():
            if tag == raise_in:
                raise MemoryError("planted refusal in this run")
            return dict(pass_=True)
        return f
    monkeypatch.setattr(C, "control_a", ok("a"))
    monkeypatch.setattr(C, "control_b_c", lambda: (ok("b")(), dict(pass_=True)))
    monkeypatch.setattr(C, "control_d", ok("d"))
    rc = C.main([])
    return rc, json.load(open(out, encoding="utf-8"))


def test_controls_main_a_stale_error_from_an_earlier_run_does_not_survive_a_passing_run(monkeypatch, tmp_path):
    # MEASURED 2026-11-13: four fresh runs read a, b, c, d pass True and all_pass False, because main() merged the "error" of an earlier refused run.
    rc, got = _planted_controls(monkeypatch, tmp_path, {"tier": "T0", "error": "MemoryError: free RAM 2.0 GB < 8.0 GB", "all_pass": False})
    assert rc == 0 and got["all_pass"] is True and "error" not in got
    assert "MemoryError" in got["error_previous_run"]


def test_controls_main_red_arm_an_error_in_this_run_still_fails(monkeypatch, tmp_path):
    rc, got = _planted_controls(monkeypatch, tmp_path, {"tier": "T0"}, raise_in="a")
    assert rc == 1 and got["all_pass"] is False and "planted refusal" in got["error"]


# ------------------------------------------------------------------------------------------------------------- heavy: real checkpoint, planted basis arm
def _ram_ok():
    return S.LR.free_gb() >= 8.0 and os.path.exists(C.R35H_DIR)


@pytest.mark.skipif(not _ram_ok(), reason="needs >= 8 GB free RAM and the pulled R35H s0 run")
def test_control_b_and_c_planted_basis_arm():
    b, c = C.control_b_c()
    assert b["pass_"], b
    assert c["pass_"], c
    assert all(v["refused"] for v in c["refusals"].values())


@pytest.mark.skipif(not _ram_ok(), reason="needs >= 8 GB free RAM and the pulled R35H s0 run")
def test_control_a_adapter_identity_against_the_banked_units():
    a = C.control_a()
    assert a["pass_"], a
    assert a["a3_banked_first_batch"]["bit_equal"] and a["a3_banked_first_batch"]["max_abs_diff"] == 0.0
