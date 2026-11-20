"""test_tokclose.py -- tests of tokclose_fit.py (card PREREG_V8F_PROOF_ADD81). Synthetic data only, CPU only, small.
The tool file can be swapped by the environment (the mutation runner does): TOKCLOSE_FILE.
Run: PYCAP_MB=3000 bash ops/pycap.sh -m pytest -x -q -p no:cacheprovider --basetemp <NEW folder> tokclose/test_tokclose.py
The cheap tests come first; the pipeline tests share one synthetic world with its nine fits (module fixture).
"""
import copy
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TOKCLOSE_BUILD_DIR", os.path.join(HERE, "..", "integ10k"))
for _p in (HERE, os.path.join(HERE, "..", "closeread"), os.path.join(HERE, "..", "integ10k")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, env, default):
    path = os.environ.get(env) or os.path.join(HERE, default)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


TC = _load("tokclose_fit_t", "TOKCLOSE_FILE", "tokclose_fit.py")
CF, CD, CL = TC.CF, TC.CD, TC.CL
import torch  # noqa: E402


# ============================================================================================ cheap tests (no fixture)
def test_hyperparameters_are_the_fixed_ones():
    assert TC.HP["E"] == {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 100, "patience": 10}
    assert TC.HP["T"] == {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 8, "patience": 2}
    assert TC.HP["huber_delta"] == 1.0 and TC.HP["stop_frac"] == 0.2 and TC.HP["shuffle_seed"] == 20261120
    assert TC.DIMS == {"W": 16, "emb": 64, "hid": 128, "mid": 64, "tok_hidden": 128}
    assert TC.ARMS == ("E10", "E25", "T10", "T25") and TC.HS == (10, 25)
    assert TC.T3_TOL == 1e-5 and TC.T1_BAND == 0.10 and TC.T0_TOL_R == 0.005 and TC.T0_TOL_LABEL == 1e-6


def test_no_banned_word_in_the_files():
    bad = []
    for f in sorted(os.listdir(HERE)):
        if f.endswith((".py", ".md", ".sh", ".json")) and not f.startswith("tmp"):
            txt = open(os.path.join(HERE, f), encoding="utf-8", errors="replace").read().lower()
            bad += [(f, w) for w in TC.BANNED if w in txt]
    assert not bad


def test_eligible_frames_rule():
    T = 40
    pres = np.ones(T, bool)
    run = np.full(T, 5)
    clo = np.zeros(T)
    f = TC.eligible_frames(pres, run, clo, T, 10)
    assert f.min() == 13 and f.max() == T - 1 and len(f) == T - 13
    assert TC.eligible_frames(pres, run, clo, T, 25).min() == 28
    run2 = run.copy()
    run2[20] = 2
    pres2 = pres.copy()
    pres2[21] = False
    clo2 = clo.copy()
    clo2[22] = np.nan
    f2 = TC.eligible_frames(pres2, run2, clo2, T, 10)
    assert not {20, 21, 22} & set(f2.tolist()) and {19, 23} <= set(f2.tolist())


def test_permute_labels_keeps_mask_and_multiset_and_moves_values():
    rows = {"a": (np.arange(5), np.arange(5, dtype=np.float32)), "b": (np.arange(3), np.arange(5, 8, dtype=np.float32)), "c": (np.zeros(0, np.int64), np.zeros(0, np.float32))}
    out = TC.permute_labels(rows, ["a", "b", "c"], 7)
    assert [len(out[c][1]) for c in "abc"] == [5, 3, 0]
    assert np.array_equal(np.sort(np.concatenate([out["a"][1], out["b"][1]])), np.arange(8, dtype=np.float32))
    assert not np.array_equal(out["a"][1], rows["a"][1])
    assert np.array_equal(out["a"][0], rows["a"][0])
    again = TC.permute_labels(rows, ["a", "b", "c"], 7)
    assert np.array_equal(again["a"][1], out["a"][1])
    assert np.array_equal(rows["a"][1], np.arange(5, dtype=np.float32))


def _model(H=6, emb=8, mode="E"):
    torch.manual_seed(0)
    return TC.make_model(mode, H, {"d_feat": 8, "n_tok": 2, "W": 3, "emb": emb, "hid": 12, "mid": 6, "tok_hidden": 10})


def test_window_is_exactly_the_last_H_embeddings():
    m = _model(6).eval()
    E = torch.randn(30, 8)
    f = torch.tensor([10, 20])
    base = m.rows(E, f).detach()
    E2 = E.clone()
    E2[21:] = torch.randn(9, 8)
    assert torch.allclose(m.rows(E2, f).detach(), base, atol=0)                      # after the frame: no effect
    E3 = E.clone()
    E3[10 - 6] = torch.randn(8)                                                       # frame f-H: outside the window
    assert torch.allclose(m.rows(E3, f).detach()[0], base[0], atol=0)
    E4 = E.clone()
    E4[10 - 5] = torch.randn(8)                                                       # frame f-H+1: inside the window
    assert not torch.allclose(m.rows(E4, f).detach()[0], base[0], atol=1e-6)
    E5 = E.clone()
    E5[10] = torch.randn(8)                                                           # frame f itself
    assert not torch.allclose(m.rows(E5, f).detach()[0], base[0], atol=1e-6)


def test_predict_clip_nan_before_the_window_and_causal_by_construction():
    m = _model(6)
    E = torch.randn(30, 8)
    out = TC.predict_clip(m, E, "cpu")
    assert np.isnan(out[:5]).all() and np.isfinite(out[5:]).all() and len(out) == 30
    E2 = E.clone()
    E2[18:] = 0
    out2 = TC.predict_clip(m, E2, "cpu")
    assert np.array_equal(out[:18], out2[:18], equal_nan=True)


def test_t3_detects_a_head_that_looks_one_frame_ahead():
    m = _model(6, mode="T")
    toks = [torch.randn(30, 2, 8).to(torch.float16) for _ in range(2)]
    assert TC.t3_max_diff(m, toks, None, "cpu") <= TC.T3_TOL
    m.shift = 1
    assert TC.t3_max_diff(m, toks, None, "cpu") > 1e-3


def _summ(r_ci_lo, r_seed, r_pool=0.4):
    return {"r_pooled": r_pool, "r_ci": [r_ci_lo, r_pool + 0.1], "r_per_seed": r_seed}


def test_name_rule_cases():
    cell = _summ(0.2, [0.30, 0.30])
    (nr, nb), rd, bt = TC.name_arm("T10", _summ(0.1, [0.4, 0.4]), cell, (-0.2, -0.3, -0.1))
    assert nr == "TOKCLOSE-T10-READABLE" and nb == "TOKCLOSE-T10-BETTER-THAN-OUTPUTS" and rd and bt
    (nr, nb), rd, bt = TC.name_arm("T10", _summ(-0.01, [0.4, 0.4]), cell, (-0.2, -0.3, -0.1))
    assert nr == "TOKCLOSE-T10-NOT-READABLE" and not rd and bt                                    # readability and better are separate names
    (nr, nb), rd, bt = TC.name_arm("T10", _summ(0.1, [0.4, 0.4]), cell, (-0.2, -0.3, 0.05))
    assert nb == "TOKCLOSE-T10-NOT-BETTER"                                                        # interval reaches 0
    (nr, nb), rd, bt = TC.name_arm("T10", _summ(0.1, [0.4, 0.25]), cell, (-0.2, -0.3, -0.1))
    assert nb == "TOKCLOSE-T10-NOT-BETTER"                                                        # r above the cell in one seed only
    (nr, nb), rd, bt = TC.name_arm("T10", _summ(0.1, [0.4, 0.30]), cell, (-0.2, -0.3, -0.1))
    assert nb == "TOKCLOSE-T10-NOT-BETTER"                                                        # equal r is not above
    assert TC.verdict_name([False, False, True, False]) == ("TOKCLOSE-A-TOKEN-HEAD-READS-MORE", 1)
    assert TC.verdict_name([False] * 4) == ("TOKCLOSE-NO-GAIN-FROM-TOKENS", 0)


def test_paired_difference_is_x_minus_cell():
    rng = np.random.default_rng(0)
    ep = np.asarray(["e%02d" % e for e in np.repeat(np.arange(30), 10)])
    y = rng.normal(0, 2, 300)
    rows = [{"y": y, "ep": ep}, {"y": y, "ep": ep}]
    good = [y + rng.normal(0, 0.3, 300) for _ in range(2)]
    bad = [rng.normal(0, 3, 300) for _ in range(2)]
    d = TC.pair_vs_cell(good, bad, rows, 200, 5)
    assert d[0] < 0 and d[2] < 0
    d2 = TC.pair_vs_cell(bad, good, rows, 200, 5)
    assert d2[0] > 0 and d2[1] > 0


def test_t4_planted_names():
    r = TC.control_t4()
    assert r["pass"] and r["gain"]["better"] and not r["none"]["better"] and not r["one_seed_only"]["better"]


def _log(**kw):
    L = {"seed": 0, "mode": "T", "shuffled": False, "loss_before": 1.0, "loss_at_stop": 0.5, "n_train_clips": 10, "n_stop_clips": 4, "n_train_rows": 100, "n_stop_rows": 40, "n_eval_clips": 3,
         "overlap_fit_eval": 0, "overlap_stop_eval": 0, "overlap_fit_stop": 0, "hp": TC.HP["T"], "hp_override": False,
         "train_ids_sha256": TC.sha_ids(["t%d" % i for i in range(10)]), "stop_ids_sha256": TC.sha_ids(["h%d" % i for i in range(4)])}
    L.update(kw)
    return L


def test_t2_logic():
    lists = {0: {"train_ids": ["t%d" % i for i in range(10)], "heldout_ids": ["h%d" % i for i in range(4)]}}
    exp = {"train": 10, "heldout": 4, "eval": 3}
    assert TC.control_t2({"a": _log()}, lists, [], exp)["pass"]
    assert not TC.control_t2({"a": _log(loss_at_stop=1.0)}, lists, [], exp)["pass"]               # stopping loss not below the loss before
    assert not TC.control_t2({"a": _log(loss_at_stop=float("nan"))}, lists, [], exp)["pass"]
    assert not TC.control_t2({"a": _log(n_train_clips=9)}, lists, [], exp)["pass"]
    assert not TC.control_t2({"a": _log(overlap_fit_eval=1)}, lists, [], exp)["pass"]
    assert not TC.control_t2({"a": _log(train_ids_sha256="0" * 64)}, lists, [], exp)["pass"]
    assert not TC.control_t2({"a": _log(hp_override=True)}, lists, [], exp)["pass"]
    assert TC.control_t2({"a": _log(hp_override=True, hp={"x": 1})}, lists, [], exp, allow_override=True)["pass"]
    assert not TC.control_t2({"a": _log(hp=dict(TC.HP["T"], lr=0.01))}, lists, [], exp)["pass"]
    assert not TC.control_t2({"a": _log(n_stop_rows=0)}, lists, [], exp)["pass"]
    planted = _log(shuffled=True, loss_at_stop=1.2)
    assert TC.control_t2({"a": planted}, lists, [], exp)["pass"]                                    # the planted fit's loss pair is reported, not judged
    assert not TC.control_t2({"a": planted}, lists, [], exp, judge_planted_loss=True)["pass"]
    e = _log(mode="E", n_train_clips=3, n_stop_clips=1, hp=TC.HP["E"])
    assert TC.control_t2({"a": e}, lists, [], exp | {"heldout": 4})["pass"]
    assert not TC.control_t2({"a": _log(mode="E", n_train_clips=3, n_stop_clips=2, hp=TC.HP["E"])}, lists, [], exp | {"heldout": 4})["pass"]
    assert not TC.control_t2({}, lists, [], exp)["pass"]


def test_fit_head_early_stopping_state_and_loss_pair():
    rng = np.random.default_rng(0)
    ids = ["c%02d" % i for i in range(16)]
    f = np.arange(10, 30)
    emb, rows_tr, rows_st = {}, {}, {}
    for c in ids:
        x = rng.normal(0, 1, (30, 8)).astype(np.float32)
        emb[c] = torch.from_numpy(x)
        rows_tr[c] = (f, (1.5 * x[f - 2, 0]).astype(np.float32))
        rows_st[c] = (f, (-1.5 * x[f - 2, 0]).astype(np.float32))                          # the stopping set disagrees with the fit set: its loss rises as the fit improves
    dims = {"d_feat": 8, "n_tok": 2, "W": 3, "emb": 8, "hid": 12, "mid": 6, "tok_hidden": 10}
    hp = {"batch": 4, "lr": 3e-3, "clip_norm": 1.0, "max_evals": 40, "patience": 2}
    m, rec = TC.fit_head("E", 6, 0, ids[:10], ids[10:], lambda c: emb[c], rows_tr, rows_st, dims, "cpu", hp, log=lambda *a: None)
    assert rec["stopped_by"] == "patience" and len(rec["curve"]) < 1 + hp["max_evals"]
    assert rec["curve"][0]["step"] == 0 and rec["loss_before"] == rec["curve"][0]["loss"]
    assert abs(rec["loss_at_stop"] - min(c["loss"] for c in rec["curve"][1:])) < 1e-12
    rows = dict(rows_tr)
    rows.update(rows_st)
    got = TC.eval_loss(m, ids[10:], lambda c: emb[c], rows, "cpu", hp)
    assert abs(got - rec["loss_at_stop"]) < 1e-5                                                  # the kept state is the state of the best evaluation
    m2, rec2 = TC.fit_head("E", 6, 0, ids[:10], ids[10:], lambda c: emb[c], rows_tr, rows_st, dims, "cpu", hp, log=lambda *a: None)
    assert rec2["curve"] == rec["curve"]                                                          # repeatable per seed
    torch.manual_seed(0)
    fresh = TC.make_model("E", 6, dims)
    assert abs(TC.eval_loss(fresh, ids[10:], lambda c: emb[c], rows, "cpu", hp) - rec["loss_before"]) < 1e-6      # loss before the first step = the untrained model on the stopping set
    assert rec["n_train_rows"] == 10 * len(f) and rec["n_stop_rows"] == 6 * len(f)


def test_fit_head_learns_a_planted_signal_and_ignores_the_future():
    rng = np.random.default_rng(1)
    ids = ["c%02d" % i for i in range(24)]
    emb, rows = {}, {}
    f = np.arange(12, 40)
    for c in ids:
        x = rng.normal(0, 1, (40, 8)).astype(np.float32)
        emb[c] = torch.from_numpy(x)
        y = x[f - 1, 0] * 1.5                                                                     # the label is in frame f-1: inside a window of 6
        rows[c] = (f, y.astype(np.float32))
    dims = {"d_feat": 8, "n_tok": 2, "W": 3, "emb": 8, "hid": 16, "mid": 8, "tok_hidden": 10}
    hp = {"batch": 4, "lr": 1e-2, "clip_norm": 1.0, "max_evals": 60, "patience": 60}
    m, rec = TC.fit_head("E", 6, 0, ids[:18], ids[18:], lambda c: emb[c], rows, rows, dims, "cpu", hp, log=lambda *a: None)
    assert rec["loss_at_stop"] < 0.5 * rec["loss_before"]
    est = TC.predict_clip(m, emb[ids[0]], "cpu")
    assert CF.pearson(est[f], rows[ids[0]][1]) > 0.8


# ============================================================================================ the pipeline on a synthetic world
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("tcworld"))
    w = TC.make_world(root)
    assert TC.run_world_fits(w) == 0
    return w


def _dumps(world, ks=(0, 1)):
    return {k: CF.load_dump(os.path.join(world["root"], "dump", "reader_s%d_score.npz" % k)) for k in ks}


def _join_eval(world):
    J = np.load(os.path.join(world["root"], "join_eval.npz"))
    return {k: J[k] for k in ("off", "n", "lead_gap", "lead_valid")}, [str(c) for c in J["clip"]]


def _epmap(world):
    return json.load(open(os.path.join(world["root"], "episode_map.json")))


def test_rows_equal_build_rows(world):
    D = _dumps(world, (0,))[0]
    J, jc = _join_eval(world)
    em = _epmap(world)
    for H in TC.HS:
        mine = TC.build_clip_rows(J, jc, world["eval"], H)
        mine_set = {(c, int(f)) for c, (fr, _) in mine.items() for f in fr}
        rows = CF.build_rows(D, H, em)
        assert mine_set == {(str(c), int(f)) for c, f in zip(rows["clip"], rows["frame"])} and len(mine_set) > 100
        # the label of the tool is the label of the dump on those frames
        lab = {(str(c), int(f)): float(y) for c, f, y in zip(rows["clip"], rows["frame"], rows["y"])}
        for c, (fr, y) in mine.items():
            for f, v in zip(fr, y):
                assert abs(lab[(c, int(f))] - float(v)) < 1e-5


def test_t0_passes_and_catches_shifted_label_estimator_and_frame_set(world):
    dumps, (J, jc), em = _dumps(world), _join_eval(world), _epmap(world)
    add = json.load(open(os.path.join(world["root"], "add78_result.json")))
    est = os.path.join(world["root"], "est")
    assert TC.control_t0(dumps, J, em, est, add, [0, 1])["pass"]
    sh = {k: dict(v) for k, v in dumps.items()}
    sh[0]["lab_closing"] = np.roll(dumps[0]["lab_closing"], 1)
    r_sh = TC.control_t0(sh, J, em, est, add, [0, 1])
    assert not r_sh["pass"] and r_sh["label"]["seed0"]["pass"] is False and r_sh["label"]["seed1"]["pass"] is True
    add2 = copy.deepcopy(add)
    add2["cells"]["N_H10"]["r_per_seed"][1] += 0.01
    assert not TC.control_t0(dumps, J, em, est, add2, [0, 1])["pass"]
    add3 = copy.deepcopy(add)
    add3["cells"]["L_H25"]["r_per_seed"][0] += 0.004
    assert TC.control_t0(dumps, J, em, est, add3, [0, 1])["pass"]                                 # inside the 0.005 tolerance
    add4 = copy.deepcopy(add)
    add4["cells"]["L_H25"]["n_frames_per_seed"][0] += 1
    assert not TC.control_t0(dumps, J, em, est, add4, [0, 1])["pass"]
    fs = {k: dict(v) for k, v in dumps.items()}
    lr = np.array(dumps[0]["lab_run"], copy=True)
    lr[np.flatnonzero(lr >= 3)[:40]] = 2                                                          # a different set of labelled frames in the dump
    fs[0]["lab_run"] = lr
    assert not TC.control_t0(fs, J, em, est, add, [0, 1])["pass"]
    vs = {k: dict(v) for k, v in dumps.items()}
    vv = np.array(dumps[0]["valid"], copy=True)
    vv[np.flatnonzero(vv)[:40]] = False                                                           # the dump drops reads of some frames: labels equal, frame sets differ
    vs[0]["valid"] = vv
    r_vs = TC.control_t0(vs, J, em, est, add, [0, 1])
    assert not r_vs["pass"] and r_vs["label"]["seed0"]["pass"] is True and r_vs["frames"]["H10_seed0"]["pass"] is False
    sw = {k: dict(v) for k, v in dumps.items()}
    sw[1]["gap"] = np.array(dumps[1]["gap"], copy=True)                                           # reads changed: the estimators do not reproduce r
    sw[1]["gap"][sw[1]["valid"]] = np.random.default_rng(0).uniform(5, 55, int(sw[1]["valid"].sum()))
    assert not TC.control_t0(sw, J, em, est, add, [0, 1])["pass"]


def _table_est(world, tag, D, H):
    rows = CF.build_rows(D, H, _epmap(world))
    return rows, TC.table_lookup(TC.load_table(os.path.join(world["root"], "fits", "tokclose_%s_table.npz" % tag)), TC.row_clip_index(D, rows), rows["frame"])


def test_fits_counts_losses_and_split(world):
    held, train, ev = set(world["held"]), set(world["train"]), set(world["eval"])
    for arm in TC.ARMS:
        for k in (0, 1):
            L = json.load(open(os.path.join(world["root"], "fits", "tokclose_%s_log.json" % TC.tag_of(arm, k)), encoding="utf-8"))
            assert L["loss_at_stop"] < L["loss_before"], (arm, k)
            assert L["overlap_fit_eval"] == 0 and L["overlap_stop_eval"] == 0 and L["overlap_fit_stop"] == 0
            if arm[0] == "T":
                assert L["n_train_clips"] == len(train) and L["n_stop_clips"] == len(held)
                assert L["train_ids_sha256"] == TC.sha_ids(train) and L["stop_ids_sha256"] == TC.sha_ids(held)
            else:
                assert L["n_train_clips"] + L["n_stop_clips"] == len(held) and L["n_stop_clips"] == int(np.ceil(0.2 * len(held)))
            assert L["n_train_rows"] > 0 and L["n_stop_rows"] > 0 and L["hp_override"] is True and L["n_eval_clips"] == len(ev)
    P = json.load(open(os.path.join(world["root"], "fits", "tokclose_T10shuf_s0_log.json"), encoding="utf-8"))
    assert P["shuffled"] is True and P["n_train_clips"] == len(train)


def test_t_arm_reads_the_planted_signal_and_the_planted_arm_does_not(world):
    D = _dumps(world, (0,))[0]
    rows, e = _table_est(world, "T10_s0", D, 10)
    assert CF.pearson(e, rows["y"]) > 0.5
    rows, e = _table_est(world, "T25_s1", _dumps(world, (1,))[1], 25)
    assert CF.pearson(e, rows["y"]) > 0.3
    rows, e = _table_est(world, "E10_s0", D, 10)
    assert CF.pearson(e, rows["y"]) > 0.1
    rows, es = _table_est(world, "T10shuf_s0", D, 10)
    assert abs(CF.pearson(es, rows["y"])) < 0.2
    assert TC.control_t1(D, _epmap(world), os.path.join(world["root"], "fits", "tokclose_T10shuf_s0_table.npz"))["pass"]
    assert not TC.control_t1(D, _epmap(world), os.path.join(world["root"], "fits", "tokclose_T10_s0_table.npz"))["pass"]   # a head that reads the signal is outside the band


def test_saved_head_is_the_best_state_and_t3_holds_on_saved_heads(world):
    held = world["held"]
    J = np.load(os.path.join(world["root"], "join_train.npz"))
    Jd = {k: J[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    jc = [str(c) for c in J["clip"]]
    seen_not_last = 0
    for arm in ("T10", "T25"):
        for k in (0, 1):
            tag = TC.tag_of(arm, k)
            L = json.load(open(os.path.join(world["root"], "fits", "tokclose_%s_log.json" % tag), encoding="utf-8"))
            m, ck = TC.load_head(os.path.join(world["root"], "fits", "tokclose_%s.pt" % tag))
            rows = TC.build_clip_rows(Jd, jc, held, m.H)
            load = lambda c: TC.load_tokens(os.path.join(world["root"], "cache", "train", "clips", c + ".pt"))
            got = TC.eval_loss(m, held, load, rows, "cpu", TC.HP["T"])
            assert abs(got - L["loss_at_stop"]) < 1e-4, tag
            seen_not_last += int(L["best_step"] != L["steps"])
    assert seen_not_last >= 1                                                                    # at least one fit whose best state is not its last state
    heads = {"T10_s0": os.path.join(world["root"], "fits", "tokclose_T10_s0.pt"), "E25_s1": os.path.join(world["root"], "fits", "tokclose_E25_s1.pt")}
    r = TC.control_t3(heads, os.path.join(world["root"], "evalcache"), world["eval"], TC.default_build_dir(), 2, "cpu")
    assert r["pass"] and all(v["max_abs_diff_ms"] <= TC.T3_TOL for v in r["heads"].values())


def test_fit_refuses_existing_outputs_and_eval_overlap(world, tmp_path):
    a = TC.fit_args(world, "T10", 0)
    assert TC.run_fit(a, lambda *x: None) == 2                                                    # outputs exist
    ec = tmp_path / "evalcache" / "eval"
    ec.mkdir(parents=True)
    with open(ec / "manifest.jsonl", "w") as fh:
        fh.write(json.dumps({"clip": world["held"][0]}) + "\n")
    b = TC.fit_args(world, "T10", 1, out="fits_overlap")
    b.eval_cache = str(tmp_path / "evalcache")
    assert TC.run_fit(b, lambda *x: None) == 2
    assert not os.path.exists(os.path.join(world["root"], "fits_overlap", "tokclose_T10_s1_log.json"))


def test_score_end_to_end_names_are_consistent(world):
    out = []
    rc = TC.run_score(TC.score_args(world, out="score_e2e"), log=out.append)
    assert rc == 0
    res = json.load(open(os.path.join(world["root"], "score_e2e", "TOKCLOSE_RESULT.json"), encoding="utf-8"))
    assert res["controls_all_pass"] is True and set(res["arms"]) == set(TC.ARMS)
    nb = 0
    for arm, v in res["arms"].items():
        assert v["names"][0] == "TOKCLOSE-%s-%s" % (arm, "READABLE" if v["r_ci"][0] > 0 else "NOT-READABLE")
        better = v["paired_diff_ci"][1] < 0 and all(a > b for a, b in zip(v["r_per_seed"], v["cell_r_per_seed"]))
        assert v["names"][1] == "TOKCLOSE-%s-%s" % (arm, "BETTER-THAN-OUTPUTS" if better else "NOT-BETTER")
        nb += int(better)
        assert v["cell"] == ("N10" if v["H"] == 10 else "L25")
    assert res["n_better_of_four"] == nb and res["verdict"] == ("TOKCLOSE-A-TOKEN-HEAD-READS-MORE" if nb else "TOKCLOSE-NO-GAIN-FROM-TOKENS")
    assert res["arms"]["T10"]["names"][0] == "TOKCLOSE-T10-READABLE"
    # the cell of the card is the saved estimator applied on the same frames: its r per seed is the r of card ADD78
    add = json.load(open(os.path.join(world["root"], "add78_result.json")))
    assert np.allclose(res["arms"]["T10"]["cell_r_per_seed"], add["cells"]["N_H10"]["r_per_seed"], atol=1e-9)
    assert np.allclose(res["arms"]["E25"]["cell_r_per_seed"], add["cells"]["L_H25"]["r_per_seed"], atol=1e-9)
    # the T arm that reads the signal is better than a random estimator: the paired interval is below 0
    assert res["arms"]["T10"]["paired_diff_ci"][1] < 0
    assert res["arms"]["T10"]["n_episodes"] == len(set(_epmap(world).values())) and res["arms"]["T10"]["n_frames"] > 300
    assert os.path.exists(os.path.join(world["root"], "score_e2e", "TOKCLOSE_RESULT.txt"))
    assert TC.run_score(TC.score_args(world, out="score_e2e"), log=out.append) == 2               # the folder exists


def test_score_fails_closed_when_a_control_fails(world):
    root = world["root"]
    d2 = os.path.join(root, "dump_shifted")
    os.makedirs(d2)
    for k in (0, 1):
        z = dict(np.load(os.path.join(root, "dump", "reader_s%d_score.npz" % k)))
        if k == 0:
            z["lab_closing"] = np.roll(z["lab_closing"], 1)
        np.savez(os.path.join(d2, "reader_s%d_score.npz" % k), **z)
        shutil.copy(os.path.join(root, "dump", "reader_s%d_lists.json" % k), os.path.join(d2, "reader_s%d_lists.json" % k))
    out = []
    rc = TC.run_score(TC.score_args(world, out="score_fail", dump_dir=d2), log=out.append)
    txt = "\n".join(out)
    assert rc == 1
    assert not os.path.exists(os.path.join(root, "score_fail", "TOKCLOSE_RESULT.json")) and not os.path.exists(os.path.join(root, "score_fail", "TOKCLOSE_RESULT.txt"))
    assert "t0" in txt and "FAIL" in txt and "TOKCLOSE-" not in txt and " r +" not in txt and "== arms ==" not in txt
    c = json.load(open(os.path.join(root, "score_fail", "TOKCLOSE_CONTROLS.json"), encoding="utf-8"))
    assert c["controls_all_pass"] is False and c["controls"]["t0"]["pass"] is False
    assert "arms" not in c and "names" not in c


def test_score_fails_closed_on_a_leaking_head_and_a_failed_t1(world):
    root = world["root"]
    f2 = os.path.join(root, "fits_leak")
    shutil.copytree(os.path.join(root, "fits"), f2)
    ck = torch.load(os.path.join(f2, "tokclose_T10_s0.pt"), weights_only=False)
    m, _ = TC.load_head(os.path.join(f2, "tokclose_T10_s0.pt"))
    # the planted table is replaced by the table of the head that reads the signal: control t1 must fail
    shutil.copy(os.path.join(root, "fits", "tokclose_T10_s0_table.npz"), os.path.join(f2, "tokclose_T10shuf_s0_table.npz")) if os.path.exists(os.path.join(f2, "tokclose_T10shuf_s0_table.npz")) else None
    # (the copy above overwrites the planted table with the real one)
    out = []
    rc = TC.run_score(TC.score_args(world, out="score_t1", fits="fits_leak"), log=out.append)
    c = json.load(open(os.path.join(root, "score_t1", "TOKCLOSE_CONTROLS.json"), encoding="utf-8"))
    assert rc == 1 and c["controls"]["t1"]["pass"] is False and not os.path.exists(os.path.join(root, "score_t1", "TOKCLOSE_RESULT.json"))
    assert "TOKCLOSE-" not in "\n".join(out)
