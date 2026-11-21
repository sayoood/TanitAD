"""test_navecho3.py -- rules, mutations, controls for navecho3.py.  Run: python -m pytest test_navecho3.py -q  (CPU; TMP on F:/TanitAD-scratch/strat1/navecho3/tmp)"""
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import navecho3 as M  # noqa: E402

SRC = open(os.path.join(HERE, "navecho3.py"), encoding="utf-8").read()


def mutant(old, new):
    assert SRC.count(old) >= 1, f"anchor not found: {old!r}"
    m = types.ModuleType("navecho3_mut")
    m.__file__ = os.path.join(HERE, "navecho3.py")
    exec(compile(SRC.replace(old, new, 1), "navecho3_mut", "exec"), m.__dict__)
    return m


def cells(miss, pres):
    return {k: {"share_cue_missing": miss, "share_cue_present": pres} for k in M.RULE_ARMS}


# ---- cue states
KEY = np.array([1, 2, 0, 3, 1, 2])
LAB = np.array([1, 1, 1, 1, 2, 2])


def test_cue_state_values():
    assert M.cue_state(KEY, LAB).tolist() == [M.CORRECT, M.WRONG, M.NONE, M.NONE, M.WRONG, M.CORRECT]


def test_cue_state_mutants_go_red():
    for old, new in (("correct = key == label", "correct = key == 3 - label"), ("wrong = np.isin(key, (TURN_L, TURN_R)) & ~correct", "wrong = np.isin(key, (0, TURN_L, TURN_R)) & ~correct")):
        assert M.cue_state(KEY, LAB).tolist() != mutant(old, new).cue_state(KEY, LAB).tolist()


def test_label_as_token_is_all_correct():
    y = np.array([1, 2, 2, 1])
    assert (M.cue_state(y, y) == M.CORRECT).all()


# ---- miss split
MISSED = np.array([1, 1, 1, 1, 0, 0], bool)
TS = np.array([M.NONE, M.WRONG, M.CORRECT, M.NONE, M.NONE, M.CORRECT])
VS = np.array([M.CORRECT, M.CORRECT, M.CORRECT, M.WRONG, M.CORRECT, M.CORRECT])


def test_miss_split_partitions_missed_turns():
    s = M.miss_split(MISSED, TS, VS)
    assert s["cue_missing"].tolist() == [1, 1, 0, 0, 0, 0]
    assert s["cue_present"].tolist() == [0, 0, 1, 0, 0, 0]
    assert s["neither"].tolist() == [0, 0, 0, 1, 0, 0]
    assert (s["cue_missing"].astype(int) + s["cue_present"] + s["neither"]).tolist() == MISSED.astype(int).tolist()


def test_miss_split_mutants_go_red():
    base = M.miss_split(MISSED, TS, VS)
    m1 = mutant("(ts != CORRECT) & (vs == CORRECT), \"cue_present\"", "(ts != CORRECT), \"cue_present\"").miss_split(MISSED, TS, VS)     # drop the v9 condition
    assert m1["cue_missing"].tolist() != base["cue_missing"].tolist()
    m2 = mutant("\"cue_present\": missed & (ts == CORRECT)", "\"cue_present\": (ts == CORRECT)").miss_split(MISSED, TS, VS)             # counts a non-missed window
    assert m2["cue_present"].tolist() != base["cue_present"].tolist()
    m3 = mutant("missed, ts, vs = np.asarray(missed, bool), np.asarray(text_state)", "missed, ts, vs = ~np.asarray(missed, bool), np.asarray(text_state)").miss_split(MISSED, TS, VS)
    assert m3["cue_missing"].tolist() != base["cue_missing"].tolist()


# ---- the committed reading
def test_reading_rules():
    assert M.reading(cells(0.6, 0.2)) == "CUE-MISSING"
    assert M.reading(cells(0.2, 0.6)) == "CUE-PRESENT-MODEL"
    assert M.reading(cells(0.4, 0.4)) == "MIXED"
    c = cells(0.6, 0.2)
    c[M.RULE_ARMS[-1]]["share_cue_missing"] = 0.49
    assert M.reading(c) == "MIXED"                       # one cell below the bar breaks the rule (it must hold at both seeds)
    assert M.reading(cells(0.5, 0.2)) == "CUE-MISSING"   # the bar is inclusive


def test_reading_mutants_go_red():
    assert mutant("c[\"share_cue_missing\"] >= bar", "c[\"share_cue_missing\"] > bar").reading(cells(0.5, 0.2)) != "CUE-MISSING"
    assert mutant("c[\"share_cue_present\"] >= bar", "c[\"share_cue_present\"] > bar").reading(cells(0.2, 0.5)) != "CUE-PRESENT-MODEL"
    assert mutant("SHARE_BAR = 0.5", "SHARE_BAR = 0.4").reading(cells(0.45, 0.1)) == "CUE-MISSING"        # at the default bar this is MIXED
    assert M.reading(cells(0.45, 0.1)) == "MIXED"
    assert mutant("if all(c[\"share_cue_missing\"] >= bar for c in cells.values())", "if any(c[\"share_cue_missing\"] >= bar for c in cells.values())").reading(
        {**cells(0.6, 0.2), M.RULE_ARMS[0]: {"share_cue_missing": 0.1, "share_cue_present": 0.1}}) == "CUE-MISSING"
    assert M.reading({**cells(0.6, 0.2), M.RULE_ARMS[0]: {"share_cue_missing": 0.1, "share_cue_present": 0.1}}) == "MIXED"


# ---- distance and time
def test_distance_and_time():
    nav = np.array([[0, 1, 0, 0.5, 1], [0, 1, 0, 0.5, 0], [0, 0, 1, 3.0, 1]])
    d = M.distance_m(nav)
    assert d[0] == 50.0 and np.isnan(d[1]) and d[2] == 300.0
    t = M.time_to_turn_s(np.array([50.0, 50.0, 30.0]), np.array([10.0, 0.0, 15.0]))
    assert t[0] == 5.0 and np.isnan(t[1]) and t[2] == 2.0


def test_time_mutant_goes_red():
    m = mutant("d_m / np.where(v0 >= 0.5, v0, 1.0)", "d_m * np.where(v0 >= 0.5, v0, 1.0)")
    assert m.time_to_turn_s(np.array([50.0]), np.array([10.0]))[0] != 5.0
    m = mutant("nav[:, 3] * 100.0", "nav[:, 3] * 10.0")
    assert m.distance_m(np.array([[0, 1, 0, 0.5, 1.0]]))[0] != 50.0


# ---- grouping and importance
rng = np.random.default_rng(1)
EID = np.repeat(np.arange(40), 8)
Y = rng.integers(0, 5, len(EID))
X = np.c_[M.N2.label_onehot(Y)[:, :3], rng.normal(size=(len(Y), 2))]       # ch0-2 carry the label, ch3-4 are noise


def test_importance_finds_the_informative_channels_and_not_the_noise():
    r, _, _ = M.permutation_importance(X, Y, EID, n=4)
    g = r["groups"]
    assert g["token_group_ch0-2"]["importance_mean"] > 0.2
    assert abs(g["distance_group_ch3-4"]["importance_mean"]) < 0.1
    assert g["token_group_ch0-2"]["importance_mean"] > g["distance_group_ch3-4"]["importance_mean"] + 0.15


def test_importance_is_zero_for_an_untouched_input_and_deterministic():
    r1, _, _ = M.permutation_importance(X, Y, EID, n=3)
    r2, _, _ = M.permutation_importance(X, Y, EID, n=3)
    assert r1 == r2
    base = r1["base_macro_f1"]
    assert base == M.N.macro_f1(Y, M.N2.crossfit_hgb(X, Y, EID)[0])         # K-BASE: the unpermuted fold models are navecho2's


def test_folds_never_split_an_episode_and_a_mutant_does():
    f = M.N2.episode_folds(EID)
    for e in np.unique(EID):
        assert len(np.unique(f[EID == e])) == 1
    import navecho2
    src = open(os.path.join(os.path.dirname(navecho2.__file__), "navecho2.py"), encoding="utf-8").read()
    m = types.ModuleType("n2mut")
    m.__file__ = navecho2.__file__
    exec(compile(src.replace("return np.asarray([fold_of[int(e)] for e in eid], np.int64)", "return np.arange(len(eid)) % k"), "n2mut", "exec"), m.__dict__)
    fm = m.episode_folds(EID)
    assert any(len(np.unique(fm[EID == e])) > 1 for e in np.unique(EID))


def test_permutation_mutant_goes_red():
    # permuting nothing (identity) must give importance 0 on the signal channels; the real code must not
    m = mutant("p = rng.permutation(len(idx))", "p = np.arange(len(idx))")
    r, _, _ = m.permutation_importance(X, Y, EID, n=2)
    assert abs(r["groups"]["token_group_ch0-2"]["importance_mean"]) < 1e-12
    r0, _, _ = M.permutation_importance(X, Y, EID, n=2)
    assert r0["groups"]["token_group_ch0-2"]["importance_mean"] > 0.2


def _spy(mod):
    seen = []
    orig = mod.make_clf
    mod.make_clf = lambda params: (seen.append(dict(params)), orig(params))[1]
    return seen


def test_unweighted_shuffle_reads_near_chance_and_is_unweighted():
    seen = _spy(M)
    r = M.shuffled_unweighted(X, Y, EID, n=2)
    assert r["mean"] < 0.4 and seen and all("class_weight" not in p for p in seen)
    mm = mutant("if k != \"class_weight\"}", "}")
    seen = _spy(mm)
    mm.shuffled_unweighted(X, Y, EID, n=1)
    assert any("class_weight" in p for p in seen)


# ---- self-pair and shares
def test_self_pair_reads_exactly_zero_and_share_point():
    from taniteval import ci as CI
    a = (np.arange(len(EID)) % 3 == 0).astype(float)
    r = M.mean_pair(CI, a, a, EID, nb=50)
    assert r["delta"] == 0.0
    s = M.share_ci(CI, a > 0, np.ones(len(a)), EID, nb=50)
    assert abs(s["point"] - a.mean()) < 1e-12 and s["n_den"] == len(a)


def test_self_pair_mutant_goes_red():
    from taniteval import ci as CI
    a = (np.arange(len(EID)) % 3 == 0).astype(float)
    m = mutant("[float(x) for x in b], np.asarray(eid)", "[float(x) + 0.1 for x in b], np.asarray(eid)")
    assert m.mean_pair(CI, a, a, EID, nb=50)["delta"] != 0.0
