"""Tests of the P4 instrument (card section 8 (e)).  Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q -p no:cacheprovider test_p4_cueclass.py
Fast: synthetic windows except two tests that read the banked window flags.  CUDA_VISIBLE_DEVICES=-1."""
import inspect
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import p4_cueclass as P  # noqa: E402


def iv(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi}


def pop_case(li, le, se, dli, dle):
    return {"LIGHT": li, "LEAD": le, "SELF": se, "D_LIGHT": dli, "D_LEAD": dle}


POS, ZERO, NEG = iv(0.04, 0.01, 0.07), iv(0.002, -0.02, 0.025), iv(-0.04, -0.07, -0.01)


def verdict_cases_ok(fn_pop):
    """The constructed intervals and the word each must give (the card's rule, one line per outcome)."""
    cases = [
        (pop_case(POS, POS, ZERO, iv(0.038, 0.0, 0.08), iv(0.038, 0.0, 0.08)), True, None),             # HOLDS
        (pop_case(POS, POS, POS, iv(0.0, -0.03, 0.03), iv(0.0, -0.03, 0.03)), False, "F-c"),           # SELF lifts as much as the cue classes
        (pop_case(POS, POS, iv(0.01, 0.002, 0.02), iv(0.03, 0.0, 0.06), iv(0.03, 0.0, 0.06)), False, "F-c"),  # SELF lifts, but less: only C3 catches it
        (pop_case(ZERO, ZERO, ZERO, iv(0.0, -0.03, 0.03), iv(0.0, -0.03, 0.03)), False, "F-a"),         # nothing lifts
        (pop_case(POS, ZERO, ZERO, iv(0.03, 0.0, 0.06), iv(0.0, -0.03, 0.03)), False, "F-b"),           # one cue class
        (pop_case(POS, POS, ZERO, iv(-0.01, -0.04, 0.02), iv(0.03, 0.0, 0.06)), False, "F-d"),          # ordering reversed on one contrast
        (pop_case(NEG, POS, ZERO, iv(0.0, -0.03, 0.03), iv(0.08, 0.04, 0.12)), False, "F-e"),           # L reads worse on LIGHT
        (pop_case(POS, POS, NEG, iv(0.08, 0.04, 0.12), iv(0.08, 0.04, 0.12)), True, None),              # worse on SELF with both cue classes up: HOLDS as written
    ]
    for case, holds, sub in cases:
        r = fn_pop(case)
        assert r["holds"] is holds, (case, r)
        assert r["subcase"] == sub, (case, r)
    return True


def test_verdict_rule_each_outcome():
    assert verdict_cases_ok(P.verdict_pop)
    h = pop_case(POS, POS, ZERO, iv(0.038, 0.0, 0.08), iv(0.038, 0.0, 0.08))
    assert P.verdict_overall({"pstop": h, "pmove": h})["verdict"] == "HOLDS"
    f = pop_case(ZERO, ZERO, ZERO, iv(0.0, -0.03, 0.03), iv(0.0, -0.03, 0.03))
    assert P.verdict_overall({"pstop": h, "pmove": f}) ["verdict"] == "FAILS"
    assert P.verdict_overall({"pstop": None, "pmove": h})["verdict"] == "HOLDS"             # an unevaluable population is skipped, not padded
    assert P.verdict_overall({"pstop": None, "pmove": None})["verdict"] == "NOT EVALUABLE"


def test_mutant_without_C3_is_caught():
    """The deliberate regression: the rule with C3 (SELF must not lift) dropped.  The case table must catch it (RED)."""
    src = inspect.getsource(P.verdict_pop)
    needle = "c3 = not sep_pos(se)"
    assert src.count(needle) == 1, "the line the mutant replaces is not there: the test would pass vacuously"
    ns = {"sep_pos": P.sep_pos, "sep_neg": P.sep_neg}
    exec(src.replace(needle, "c3 = True"), ns)
    with pytest.raises(AssertionError):
        verdict_cases_ok(ns["verdict_pop"])


def test_partition_exclusive_and_complete():
    rng = np.random.default_rng(0)
    tl = rng.random(500) < 0.4
    ln = rng.random(500) < 0.4
    nl = ~ln & (rng.random(500) < 0.8)
    m = P.class_masks(tl, ln, nl)
    assert np.array_equal(sum(v.astype(int) for v in m.values()), np.ones(500, int))
    assert np.array_equal(m["LIGHT"], tl & nl) and np.array_equal(m["LEAD"], ~tl & ln) and np.array_equal(m["SELF"], ~tl & nl)
    with pytest.raises(ValueError):
        P.class_masks(tl, ln, ln)                                                              # lead_now and no_lead overlapping is refused


def test_class_counts_equal_the_card():
    card = json.load(open(f"{HERE}/P4_COUNTS_PREFREEZE.json"))["W_A"]
    for pop in ("pstop", "pmove"):
        d = P.load_pop(pop)
        for c, m in d["masks"].items():
            want = card[pop]["classes"][c]
            y = d["y"].astype(bool)
            assert (int(m.sum()), int((m & y).sum()), int((m & ~y).sum())) == (want["n"], want["pos"], want["neg"]), (pop, c)


def synth(n=3000, n_clips=300, seed=0):
    rng = np.random.default_rng(seed)
    clip = rng.integers(0, n_clips, n)
    y = (rng.random(n) < 0.5).astype(int)
    outer, inner = clip % 5, (clip // 5) % 3
    cls = rng.choice(5, size=n, p=[0.3, 0.3, 0.3, 0.05, 0.05])
    names = ("LIGHT", "LEAD", "SELF", "MIXED", "UNKNOWN")
    masks = {nm: cls == i for i, nm in enumerate(names)}
    S = rng.standard_normal((6, n, 6)).astype(np.float32)
    S[:, :, 0] += 0.15 * (2 * y - 1)                                                          # a weak shared signal, the same for both feature sets
    E = rng.standard_normal((n, 2))
    return {"y": y, "clip": clip, "outer": outer, "inner": inner, "E": E, "masks": masks}, S


def test_foldmean_auc_equals_sklearn_and_pooled_beside():
    from sklearn.metrics import roc_auc_score
    d, _ = synth(seed=1)
    rng = np.random.default_rng(5)
    sc = rng.standard_normal(len(d["y"])) + 0.5 * d["y"]
    st, boots = P.class_stats(sc, d["y"], d, B=40)
    m = d["masks"]["LEAD"]
    want = np.mean([roc_auc_score(d["y"][m & (d["outer"] == f)], sc[m & (d["outer"] == f)]) for f in range(5)])
    assert abs(st["LEAD"]["foldmean"]["auc"] - want) < 1e-12
    assert abs(st["LEAD"]["pooled"]["auc"] - roc_auc_score(d["y"][m], sc[m])) < 1e-12
    assert len(boots["LEAD"]) == 40 and st["LEAD"]["evaluable"]


def run_arm(d, S, mask, seed):
    oof, _ = P.fit_oof(S, d, extra=P.planted(d, mask, seed=seed))
    return P.class_stats(oof, d["y"], d, B=300)


def test_class_specific_plant_reads_class_specific_through_the_whole_path():
    d, S = synth(n=4500, n_clips=400, seed=2)
    oof0, _ = P.fit_oof(S, d)
    s0, b0 = P.class_stats(oof0, d["y"], d, B=300)
    base = ({f"boot_x_{c}": v for c, v in b0.items()}, {"pops": {"x": {"classes": s0}}})
    M = d["masks"]
    out = {}
    for nm, mask in (("cue", M["LIGHT"] | M["LEAD"]), ("self", M["SELF"]), ("all3", M["LIGHT"] | M["LEAD"] | M["SELF"]), ("noise", np.zeros(len(d["y"]), bool))):
        s1, b1 = run_arm(d, S, mask, 3)
        v, det = P.compare_arrays({f"boot_x_{c}": x for c, x in b1.items()}, base[0], {"pops": {"x": {"classes": s1}}}, base[1], ["x"])
        out[nm] = (v["verdict"], v["subcase"], det["x"])
    assert out["cue"][0] == "HOLDS", out["cue"]
    assert out["self"][0] == "FAILS" and P.sep_pos(out["self"][2]["SELF"]), out["self"]
    assert out["all3"][0] == "FAILS" and P.sep_pos(out["all3"][2]["SELF"]), out["all3"]
    assert out["noise"][0] == "FAILS", out["noise"]


def test_a_class_below_the_floor_is_not_evaluable():
    d, S = synth(n=3000, n_clips=300, seed=3)
    d["masks"]["LIGHT"] = np.zeros(len(d["y"]), bool)
    d["masks"]["LIGHT"][:20] = True                                                           # 20 windows: below the floor of 30 GO / 30 not-GO / 20 clips
    oof, _ = P.fit_oof(S, d)
    st, bt = P.class_stats(oof, d["y"], d, B=20)
    assert not st["LIGHT"]["evaluable"]
    arr = {f"boot_x_{c}": v for c, v in bt.items()}
    v, det = P.compare_arrays(arr, arr, {"pops": {"x": {"classes": st}}}, {"pops": {"x": {"classes": st}}}, ["x"])
    assert v["verdict"] == "NOT EVALUABLE" and isinstance(det["x"], str)


def test_card_hash_verifier_detects_an_edit_of_the_frozen_body(tmp_path_factory=None):
    r = P.verify_card()
    assert r["ok"], r
    b = open(P.CARD, "rb").read()
    flipped = b.replace(b"Evidence classes:", b"Evidence classez:", 1)
    import hashlib
    assert hashlib.sha256(flipped[:flipped.index(P.LOG_MARK)]).hexdigest() != r["frozen_sha256"]
