"""SPEC AMENDMENT A7, items A7.2 (the DISCRETE-SMALL-N population guard) and the M5 power probe, on LITERAL inputs.
No GPU, no checkpoint, no data.

    REFCV6_REPO=D:/Projects/TanitAD PYTHONPATH=D:/Projects/TanitAD/stack python -m pytest -q test_g0_a7.py

Every expectation is a LITERAL (or an independent derivation: the registered rule written out by hand), never an
expression over the code under test. Each guard carries a DELIBERATE-REGRESSION arm that must go RED -- a check that
shares the defect it checks for is green forever (CLAUDE.md):

  * the beyond-guard test is run against a judge whose guard has been DROPPED, and must fail;
  * the boundary test is run against an off-by-one guard (`>=` for `>`), and must fail;
  * M5 with the columns NOT swapped must read UNDETECTED (a probe that always says "detected" is caught);
  * the swap is checked on the REAL `AgentSlotDecoder` / `Box3DSlotDecoder` classes, and a swap at a GUESSED channel
    offset is shown to corrupt the presence channel -- so a hand-written index would be caught.
"""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ------------------------------------------------------------------------------------------------------------- #
# a synthetic low-support population: 60 per-class AP cells with support n = 3 (< 30)                              #
# ------------------------------------------------------------------------------------------------------------- #
CLASSES = ("bus", "heavy_truck", "trailer", "stroller", "animal")
BANDS = ("0_20", "20_40", "40_60")
THRS = ("0p5", "1", "2", "4")
KEYS = [f"eval_agent_det_ap{t}_{c}_{b}" for t in THRS for c in CLASSES for b in BANDS]
NPOS = {f"eval_agent_det_npos_{c}_{b}": 3.0 for c in CLASSES for b in BANDS}
BASE = 0.5
MARK = "m5: a rare-class index swap is invisible to G0"


def _row(moved=(), delta=0.05, **over):
    """The replay / fp32 row: every member at 0.5, those at index in `moved` shifted by `delta`."""
    r = {"eval_lat": 0.5, "eval_cascade": 0.8, "eval_lon_tac": 1.0, **NPOS}
    for i, k in enumerate(KEYS):
        r[k] = round(BASE + delta, 5) if i in set(moved) else BASE
    r.update(over)
    return r


def _by_seed(row, n=24):
    return {s: {"row": dict(row), "buffers_unchanged": True} for s in range(n)}


M1_ROW = _row(eval_lat=0.6)                       # M1 moves ONE term: eval_lat by 20 %


def _rec(fp32="__none__", m5=None):
    rec = {"model": {"state_dict": {"missing": [], "unexpected": []}, "param_breakdown": {"equal": True},
                     "anchor_file_vs_ckpt_buffers": {}, "declared_vs_built": {"mismatches": []}},
           "wrapper_control": {"clause": "PASS"},
           "mutations": {"m1": {"row": dict(M1_ROW)}},
           "a6": {"cells": {}}}
    if fp32 != "__none__":
        rec["a6"]["fp32_s0"] = {"row": dict(fp32)} if isinstance(fp32, dict) else fp32
    if m5 is not None:
        rec["mutations"]["m5"] = m5
    return rec


def _judge(n_in, n_num, fp32=True, amend="A7"):
    """In-run differs from the replay on n_in members; the numerics arm moves a DIFFERENT n_num members."""
    inrun = _row(moved=range(n_in))
    fp = _row(moved=range(len(KEYS) - n_num, len(KEYS))) if fp32 else "__none__"
    return G().judge(inrun, _by_seed(_row()), _rec(fp), amend=amend)


def G():
    import g0_refcv7
    return g0_refcv7


# ------------------------------------------------------------------------------------------------------------- #
# A7.2 -- the population guard                                                                                    #
# ------------------------------------------------------------------------------------------------------------- #
def test_the_population_is_60_low_support_members_and_each_is_reported_with_its_numbers():
    v = _judge(n_in=3, n_num=1)
    g = v["a7_lowsupport"]
    assert g["n_members"] == 60 and len(g["members"]) == 60
    by = {m["key"]: m for m in g["members"]}
    # index 0 moved in-run only; index 59 moved by the numerics arm only; index 30 neither
    assert by[KEYS[0]] == {"key": KEYS[0], "n": 3.0, "inrun": 0.55, "replay_mean": 0.5, "seed0": 0.5, "fp32_s0": 0.5,
                           "dev_inrun": pytest.approx(0.05), "dev_numerics": 0.0, "moved_in": True,
                           "moved_numerics": False}
    assert by[KEYS[59]]["fp32_s0"] == 0.55 and by[KEYS[59]]["moved_numerics"] is True
    assert by[KEYS[59]]["moved_in"] is False
    assert by[KEYS[30]]["moved_in"] is False and by[KEYS[30]]["moved_numerics"] is False
    # a member is REPORTED, never gated on its own (SPEC A2 / A7.2 item 1)
    t = v["terms"][KEYS[0]]
    assert t["cls"] == "DETECTION_LOWSUPPORT" and t["verdict"] == "REPORTED" and t["support_n"] == 3.0
    assert v["G0"] == "PASS", v["reasons"]


def test_beyond_the_guard_FAILS():
    v = _judge(n_in=6, n_num=0)                       # bound 2*0 + 5 = 5; 6 > 5
    g = v["a7_lowsupport"]
    assert (g["N_in"], g["N_num"], g["bound"], g["status"]) == (6, 0, 5, "FAIL")
    assert v["G0"] == "FAIL"
    assert [r for r in v["reasons"] if r.startswith("A7.2")] == [
        "A7.2 DISCRETE-SMALL-N guard: N_in=6 > 2*N_num+5=5 (N_num=0, 60 low-support members)"]


@pytest.mark.parametrize("n_in,n_num,bound,status", [
    (0, 0, 5, "PASS"),
    (5, 0, 5, "PASS"),          # AT the boundary: N_in == 2*N_num + 5  -> PASS
    (6, 0, 5, "FAIL"),
    (9, 2, 9, "PASS"),          # boundary again, with a numerics floor of 2
    (10, 2, 9, "FAIL"),
    (25, 10, 25, "PASS"),
    (26, 10, 25, "FAIL"),
])
def test_the_boundary_is_N_in_equal_2_N_num_plus_5_and_passes(n_in, n_num, bound, status):
    v = _judge(n_in, n_num)
    g = v["a7_lowsupport"]
    assert (g["N_in"], g["N_num"], g["bound"], g["status"]) == (n_in, n_num, bound, status)
    assert v["G0"] == status


def test_missing_fp32_row_FAILS_closed():
    v = _judge(n_in=0, n_num=0, fp32=False)           # the numerics arm never ran
    g = v["a7_lowsupport"]
    assert g["status"] == "NOT EVALUABLE" and g["N_num"] is None and g["bound"] is None and g["N_in"] == 0
    assert v["G0"] == "FAIL"
    assert any(r.startswith("A7.2 DISCRETE-SMALL-N guard NOT EVALUABLE (fails closed)") for r in v["reasons"])


def test_fp32_arm_that_raised_or_lacks_a_member_key_FAILS_closed():
    raised = G().judge(_row(), _by_seed(_row()), _rec({"raised": "RuntimeError: CUDA out of memory"}), amend="A7")
    assert raised["a7_lowsupport"]["status"] == "NOT EVALUABLE" and raised["G0"] == "FAIL"
    short = {k: v for k, v in _row().items() if k != KEYS[7]}          # the arm's row lacks ONE member
    v = G().judge(_row(), _by_seed(_row()), _rec(short), amend="A7")
    assert v["a7_lowsupport"]["status"] == "NOT EVALUABLE" and v["G0"] == "FAIL"
    assert v["a7_lowsupport"]["why"].startswith("the fp32_s0 row lacks 1 of the 60 members")


def test_a_null_fp32_value_where_seed0_is_defined_counts_as_a_numerics_move():
    fp = _row()
    fp[KEYS[0]] = None
    v = G().judge(_row(moved=range(11)), _by_seed(_row()), _rec(fp), amend="A7")
    g = v["a7_lowsupport"]
    assert (g["N_in"], g["N_num"], g["bound"], g["status"]) == (11, 1, 7, "FAIL")


def test_the_move_threshold_is_strictly_greater_than_002_in_written_decimal_arithmetic():
    moved = G()._a7_moved
    assert moved(0.52, 0.5) is False        # 0.52 - 0.5 is 0.020000000000000018 in floats, 0.02 as written
    assert moved(0.48, 0.5) is False
    assert moved(0.521, 0.5) is True
    assert moved(0.479, 0.5) is True


def test_guard_runs_ONLY_under_A7_the_control_arms_do_not_have_it():
    for amend in ("A5", "A6"):
        v = _judge(n_in=6, n_num=0, amend=amend)
        assert "a7_lowsupport" not in v and v["G0"] == "PASS", (amend, v["reasons"])
    assert _judge(n_in=6, n_num=0, amend="A7")["G0"] == "FAIL"


def test_A7_is_A6_plus_the_guard_on_a_population_inside_it():
    v6, v7 = _judge(2, 0, amend="A6"), _judge(2, 0, amend="A7")
    assert (v6["G0"], v6["reasons"]) == ("PASS", []) and (v7["G0"], v7["reasons"]) == ("PASS", [])
    assert v6["by_class_counts"] == v7["by_class_counts"]
    assert v6["mutation_detection"]["m1"]["n_terms_out"] == v7["mutation_detection"]["m1"]["n_terms_out"] == 1


def test_A7_carries_the_A6_floor_a_smooth_term_inside_3_phi_is_rescued_under_A6_and_A7_but_not_A5():
    """A6 items 1-6, 8 are 'unchanged' under A7 (SPEC A7): the SMOOTH floor must be in force, not silently dropped."""
    inrun = _row(eval_lon_tac=1.015)                  # rel dev 1.5 % > the registered 1 %
    fp = _row(eval_lon_tac=1.01)                      # the numerics arm moved it by phi = 0.01 -> 3*phi = 0.03
    by = _by_seed(_row())
    v5 = G().judge(inrun, by, _rec(fp), amend="A5")
    v6 = G().judge(inrun, by, _rec(fp), amend="A6")
    v7 = G().judge(inrun, by, _rec(fp), amend="A7")
    assert v5["G0"] == "FAIL" and v5["terms"]["eval_lon_tac"]["verdict"] == "OUT"
    for v in (v6, v7):
        assert v["G0"] == "PASS", v["reasons"]
        assert [r["term"] for r in v["a6_rescued"]] == ["eval_lon_tac"]


def test_the_A7_registration_gate_is_time_ordered_and_needs_A6(tmp_path, monkeypatch):
    g = G()
    a6, a7 = tmp_path / "SPEC_SHA256_AMENDMENT_A6.txt", tmp_path / "SPEC_SHA256_AMENDMENT_A7.txt"
    monkeypatch.setattr(g, "A6_REGISTRATION", a6)
    monkeypatch.setattr(g, "A7_REGISTRATION", a7)
    assert g.a7_registration("2026-10-04T12:00:00")["registered"] is False          # no file
    stamp = time.mktime(time.strptime("2026-10-04T10:00:00", "%Y-%m-%dT%H:%M:%S"))
    for p in (a7,):
        p.write_text("sha", encoding="utf-8")
        os.utime(p, (stamp, stamp))
    r = g.a7_registration("2026-10-04T12:00:00")                                    # A7 yes, A6 file absent
    assert r["registered"] is False and "A6 is not registered" in r["why"]
    a6.write_text("sha", encoding="utf-8")
    os.utime(a6, (stamp, stamp))
    r = g.a7_registration("2026-10-04T12:00:00")
    assert r["registered"] is True and r["file_mtime"] == "2026-10-04T10:00:00"
    r = g.a7_registration("2026-10-04T09:00:00")                                    # G0 started BEFORE the file
    assert r["registered"] is False and "POST HOC" in r["why"]


def test_constants_are_the_registered_ones():
    g = G()
    assert (g.A7_MOVE_TOL, g.A7_GUARD_FACTOR, g.A7_GUARD_SLACK) == (0.02, 2, 5)
    assert g.A6_AMENDS == ("A6", "A7") and "A7" in g.A2_LOWSUPPORT_AMENDS
    assert g.M5_CLASSES == ("bus", "heavy_truck")


# ---- the deliberate-regression arms of the guard ------------------------------------------------------------- #
def _assert_beyond_guard_fails():
    v = _judge(n_in=6, n_num=0)
    assert v["G0"] == "FAIL" and v["a7_lowsupport"]["status"] == "FAIL"


def _assert_boundary_passes():
    v = _judge(n_in=5, n_num=0)
    assert v["G0"] == "PASS" and v["a7_lowsupport"]["status"] == "PASS"


def test_REGRESSION_dropping_the_guard_turns_the_beyond_guard_test_RED(monkeypatch):
    _assert_beyond_guard_fails()                      # GREEN with the guard in place
    g = G()
    monkeypatch.setattr(g, "a7_lowsupport_guard",
                        lambda res, by_seed, rec: {"status": "PASS", "N_in": 0, "N_num": 0, "bound": 5,
                                                   "n_members": 0, "members": [], "why": None})
    with pytest.raises(AssertionError):               # the same test, against a judge WITHOUT the guard
        _assert_beyond_guard_fails()


def test_REGRESSION_an_off_by_one_guard_turns_the_boundary_test_RED(monkeypatch):
    _assert_boundary_passes()
    g = G()
    real = g.a7_lowsupport_guard

    def off_by_one(res, by_seed, rec):                 # '>=' where the registered text says '>'
        out = real(res, by_seed, rec)
        if out["status"] in ("PASS", "FAIL"):
            out["status"] = "FAIL" if out["N_in"] >= out["bound"] else "PASS"
        return out
    monkeypatch.setattr(g, "a7_lowsupport_guard", off_by_one)
    with pytest.raises(AssertionError):
        _assert_boundary_passes()


def test_REGRESSION_a_loosened_slack_turns_the_beyond_guard_test_RED(monkeypatch):
    _assert_beyond_guard_fails()
    monkeypatch.setattr(G(), "A7_GUARD_SLACK", 1000)
    with pytest.raises(AssertionError):
        _assert_beyond_guard_fails()


# ------------------------------------------------------------------------------------------------------------- #
# M5 -- the class vocabulary, derived and tested                                                                  #
# ------------------------------------------------------------------------------------------------------------- #
VOCAB = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person", "rider", "stroller", "animal",
         "protruding_object")


def _assert_indices(fn):
    assert fn() == {"bus": 2, "heavy_truck": 1}


def test_m5_columns_are_derived_from_the_vocabulary_the_slot_heads_and_the_GT_share():
    from tanitad.models import agent_slots as A
    assert A.AGENT_CLASSES == VOCAB                    # pins the order the literal below rests on
    assert G().m5_class_indices() == {"bus": 2, "heavy_truck": 1}
    assert G().m5_cls_slice() == slice(1, 11)          # presence is channel 0; the 10 class logits follow
    # derived, not hard-coded: a different vocabulary order moves the columns
    assert G().m5_class_indices(("bus", "automobile", "heavy_truck")) == {"bus": 0, "heavy_truck": 2}
    with pytest.raises(ValueError):
        G().m5_class_indices(("automobile", "bus"))     # heavy_truck missing -> refuse, never guess
    with pytest.raises(ValueError):
        G().m5_class_indices(("bus", "bus", "heavy_truck"))


def test_REGRESSION_a_hand_written_index_is_caught_by_the_literal():
    _assert_indices(G().m5_class_indices)               # GREEN: the derived indices
    with pytest.raises(AssertionError):                  # the plausible guess (heavy_truck listed before bus' slot)
        _assert_indices(lambda: {"bus": 1, "heavy_truck": 2})
    with pytest.raises(AssertionError):                  # the nuScenes-style guess
        _assert_indices(lambda: {"bus": 3, "heavy_truck": 4})


# ------------------------------------------------------------------------------------------------------------- #
# M5 -- the swap on the REAL head classes                                                                          #
# ------------------------------------------------------------------------------------------------------------- #
def _real_model():
    from tanitad.models import agent_slots as A
    from tanitad.models.box3d_head import Box3DSlotDecoder
    torch.manual_seed(0)
    ag = A.AgentSlotDecoder(8, 4, n_queries=3, d_model=16, depth=1, n_heads=2, enforce_band=False).eval()
    bx = Box3DSlotDecoder(8, 4, n_queries=3, d_model=16, depth=1, n_heads=2, enforce_band=False).eval()
    return SimpleNamespace(core=SimpleNamespace(agent_head=ag), _perception=SimpleNamespace(box_dec=bx)), ag, bx


def _outs(heads, mem):
    with torch.no_grad():
        return {n: h(mem) for n, h in heads.items()}


def _assert_only_the_two_class_columns_swapped(before, after, cls_start=1, a=2, b=1):
    """cls column of 'bus' is index 2, of 'heavy_truck' index 1 (literals); the presence channel is 0."""
    for n in before:
        bf, af = before[n], after[n]
        assert torch.equal(af["cls_logits"][..., 1], bf["cls_logits"][..., 2])
        assert torch.equal(af["cls_logits"][..., 2], bf["cls_logits"][..., 1])
        for c in (0, 3, 4, 5, 6, 7, 8, 9):
            assert torch.equal(af["cls_logits"][..., c], bf["cls_logits"][..., c])
        assert torch.equal(af["presence_logit"], bf["presence_logit"])           # class-agnostic: untouched
        assert torch.equal(af["box"], bf["box"])
        assert torch.equal(af["yaw_vec"], bf["yaw_vec"])
        assert torch.equal(af["rates"], bf["rates"])
        assert torch.equal(af["occ_logit"], bf["occ_logit"])


def test_swap_on_the_real_AgentSlotDecoder_and_Box3DSlotDecoder_then_bit_exact_restore():
    model, ag, bx = _real_model()
    heads = {"agent": ag, "box3d": bx}
    mem = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(1))
    before = _outs(heads, mem)
    w0 = {n: (h.head.weight.detach().clone(), h.head.bias.detach().clone()) for n, h in heads.items()}
    sw = G().ClassColumnSwap.for_model(model).apply()
    assert sw.took_effect is True
    d = sw.describe()
    assert d["classes"] == {"bus": 2, "heavy_truck": 1} and d["param_rows"] == [3, 2] and d["cls_start"] == 1
    assert sorted(d["heads"]) == ["agent", "box3d"] and d["same_module_under_two_names"] is False
    after = _outs(heads, mem)
    _assert_only_the_two_class_columns_swapped(before, after)
    assert not torch.equal(after["agent"]["cls_logits"], before["agent"]["cls_logits"])   # it DID something
    assert sw.restore() is True                                                           # swap back, bit-exact
    for n, h in heads.items():
        assert torch.equal(h.head.weight, w0[n][0]) and torch.equal(h.head.bias, w0[n][1])
    restored = _outs(heads, mem)
    for n in before:
        for k in before[n]:
            assert torch.equal(restored[n][k], before[n][k]), (n, k)


@pytest.mark.parametrize("guessed_cls_start", [0, -1])
def test_REGRESSION_a_swap_at_a_guessed_channel_offset_is_caught(guessed_cls_start):
    """The columns are `cls_start + index`. Guess cls_start = 0 and the swap exchanges the WRONG two classes
    (automobile <-> heavy_truck); guess -1 and it exchanges a class with the PRESENCE channel. Both must be caught by
    the same check that the shipped swap passes."""
    model, ag, bx = _real_model()
    heads = {"agent": ag, "box3d": bx}
    mem = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(1))
    before = _outs(heads, mem)
    g = G()
    sw = g.ClassColumnSwap(g.ClassColumnSwap.find_heads(model), g.m5_class_indices(),
                           cls_start=guessed_cls_start).apply()
    after = _outs(heads, mem)
    with pytest.raises(AssertionError):
        _assert_only_the_two_class_columns_swapped(before, after)
    assert sw.restore() is True


def test_swap_needs_BOTH_slot_heads_and_refuses_otherwise():
    model, ag, _bx = _real_model()
    with pytest.raises(ValueError, match="no \\['box3d'\\]"):
        G().ClassColumnSwap.find_heads(SimpleNamespace(core=SimpleNamespace(agent_head=ag), _perception=None))
    with pytest.raises(ValueError, match="no \\['agent'\\]"):
        G().ClassColumnSwap.find_heads(SimpleNamespace(core=SimpleNamespace(agent_head=None),
                                                       _perception=model._perception))


def test_one_module_under_two_names_is_swapped_ONCE_not_twice():
    model, ag, _bx = _real_model()
    w0 = ag.head.weight.detach().clone()
    g = G()
    sw = g.ClassColumnSwap({"agent": ag, "box3d": ag}, g.m5_class_indices(), g.m5_cls_slice().start).apply()
    assert sw.same_module is True and sw.took_effect is True
    assert not torch.equal(ag.head.weight, w0)           # swapped twice would be the identity
    assert sw.restore() is True and torch.equal(ag.head.weight, w0)


def test_a_failed_restoration_is_REPORTED_and_the_model_is_forced_back():
    model, ag, bx = _real_model()
    w0 = ag.head.weight.detach().clone()
    sw = G().ClassColumnSwap.for_model(model).apply()
    assert not torch.equal(ag.head.weight, w0)
    sw._swap = lambda: None                              # the swap-back silently does nothing
    ok = sw.restore()
    assert ok is False                                   # reported ...
    assert torch.equal(ag.head.weight, w0)               # ... and no later arm runs on a swapped model


# ------------------------------------------------------------------------------------------------------------- #
# M5 -- end to end on a SYNTHETIC head whose rare-class cells are populated, through the REAL detection metrics   #
# ------------------------------------------------------------------------------------------------------------- #
N_SLOT = 24


def _scene():
    """Windows of (class name, x, y): 40 windows of 15 automobiles (600 objects: a GATING class), then one window per
    band for a bus and one per band for a heavy_truck (n = 3 each: LOW-SUPPORT)."""
    w = [[("automobile", 2.0 + 4.0 * i, (i % 2) * 6.0 - 3.0) for i in range(15)] for _ in range(40)]
    w += [[("bus", x, 8.0)] for x in (10.0, 30.0, 50.0)]
    w += [[("heavy_truck", x, -8.0)] for x in (10.0, 30.0, 50.0)]
    return w


def _syn_head():
    """A slot head reduced to its output layer, wired to the REAL channel layout: presence from feature 10, the
    class logit of class c from feature c, the box centre from features 11 / 12."""
    from tanitad.models import agent_slots as A
    s = A.SLOT_SLICES
    m = torch.nn.Module()
    m.head = torch.nn.Linear(13, A.SLOT_WIDTH)
    with torch.no_grad():
        m.head.weight.zero_()
        m.head.bias.zero_()
        m.head.weight[s["presence"].start, 10] = 8.0
        m.head.bias[s["presence"].start] = -4.0
        for c in range(len(A.AGENT_CLASSES)):
            m.head.weight[s["cls"].start + c, c] = 8.0
        m.head.weight[s["cx"].start, 11] = 1.0 / 60.0
        m.head.weight[s["cy"].start, 12] = 1.0 / 16.0
    return m


def _features(scene):
    from tanitad.models import agent_slots as A
    idx = {c: i for i, c in enumerate(A.AGENT_CLASSES)}
    feats = torch.zeros(len(scene), N_SLOT, 13)
    feats[..., 11] = -100.0                              # empty slots sit far away, presence logit -4
    for w, objs in enumerate(scene):
        for j, (c, x, y) in enumerate(objs):
            feats[w, j, :] = 0.0
            feats[w, j, idx[c]] = 1.0
            feats[w, j, 10] = 1.0
            feats[w, j, 11], feats[w, j, 12] = x, y
    return feats, idx


def _eval_heads(model, scene, feats, idx):
    """The eval row of both heads: packs from the head outputs -> the REAL `detection_metrics.summarise` (gate 0.5)."""
    import numpy as np
    from tanitad.eval import detection_metrics as D
    from tanitad.models import agent_slots as A
    s = A.SLOT_SLICES
    row = {}
    for hd, head in (("agent", model.core.agent_head), ("box3d", model._perception.box_dec)):
        with torch.no_grad():
            raw = head.head(feats)
        logit = raw[..., s["presence"].start]
        cls = raw[..., s["cls"]].argmax(-1)
        xy = torch.stack([raw[..., s["cx"].start] * 60.0, raw[..., s["cy"].start] * 16.0], -1)
        packs = []
        for w, objs in enumerate(scene):
            n = len(objs)
            packs.append({"ep": 1000 + w, "logit": logit[w].numpy(), "xy": xy[w].numpy(),
                          "cls": cls[w].numpy().astype(np.int16), "cls_corr": cls[w].numpy().astype(np.int16),
                          "matched": np.zeros(N_SLOT, bool), "exempt": np.zeros(N_SLOT, bool),
                          "pair_err": np.zeros(0), "pair_size_err": np.zeros(0), "pair_z_err": np.zeros(0),
                          "gt_xy": np.asarray([[x, y] for _c, x, y in objs], np.float32),
                          "gt_cls": np.asarray([idx[c] for c, _x, _y in objs], np.int16),
                          "pos": np.ones(n, bool), "ign": np.zeros(n, bool), "hidden": np.zeros(n, bool)})
        for k, v in D.summarise(packs, hd, gate=0.5).items():
            row[k] = None if (isinstance(v, float) and v != v) else round(float(v), 5)
    row["eval_lat"] = 0.5
    return row


@pytest.fixture(scope="module")
def m5_scenario():
    g = G()
    scene = _scene()
    feats, idx = _features(scene)
    model = SimpleNamespace(core=SimpleNamespace(agent_head=_syn_head()),
                            _perception=SimpleNamespace(box_dec=_syn_head()))
    row0 = _eval_heads(model, scene, feats, idx)
    w0 = [(h.head.weight.detach().clone(), h.head.bias.detach().clone())
          for h in (model.core.agent_head, model._perception.box_dec)]
    sw = g.ClassColumnSwap.for_model(model).apply()
    took = sw.took_effect
    row_sw = _eval_heads(model, scene, feats, idx)
    exact = sw.restore()
    # the control: the SAME code path with the columns NOT swapped (bus <-> bus)
    ctl = g.ClassColumnSwap(g.ClassColumnSwap.find_heads(model), {"bus": 2, "heavy_truck": 2},
                            g.m5_cls_slice().start).apply()
    row_ns = _eval_heads(model, scene, feats, idx)
    ctl_took = ctl.took_effect
    ctl_exact = ctl.restore()
    row_after = _eval_heads(model, scene, feats, idx)
    same = all(torch.equal(h.head.weight, w[0]) and torch.equal(h.head.bias, w[1])
               for h, w in zip((model.core.agent_head, model._perception.box_dec), w0))
    return SimpleNamespace(row0=row0, row_sw=row_sw, row_ns=row_ns, row_after=row_after, took=took, exact=exact,
                           ctl_took=ctl_took, ctl_exact=ctl_exact, weights_restored=same)


def _m5_judge(sc, m5_row, exact=True, fp32="same", extra_gating_shift=None):
    """G0 under A7 on the synthetic scenario: in-run = replay = seed 0 = `row0`; the numerics arm is `row0` ("same"),
    a given row, or ABSENT ("__none__")."""
    fp = dict(sc.row0) if fp32 == "same" else fp32
    row = dict(m5_row)
    for k, d in (extra_gating_shift or {}).items():
        row[k] = round(row[k] + d, 5)
    m5 = {"row": row, "restoration_bit_exact": exact, "swap_took_effect": True,
          "swap": {"classes": {"bus": 2, "heavy_truck": 1}}}
    rec = _rec(fp, m5=m5)
    rec["mutations"]["m1"] = {"row": {**sc.row0, "eval_lat": 0.6}}       # M1 detected through eval_lat (SMOOTH)
    return G().judge(dict(sc.row0), _by_seed(sc.row0), rec, amend="A7")


def test_scenario_sanity_the_swap_took_effect_and_everything_is_restored_bit_exact(m5_scenario):
    sc = m5_scenario
    assert sc.took is True and sc.exact is True and sc.weights_restored is True
    assert sc.ctl_took is False and sc.ctl_exact is True
    assert sc.row_after == sc.row0                                    # the eval after restoration IS the original
    assert sc.row_ns == sc.row0                                       # swapping a column with itself changes nothing
    # the rare-class cells exist, are populated (npos = 1 per band, 3 overall) and read AP = 1.0 before the swap
    assert sc.row0["eval_agent_det_npos_bus_0_20"] == 1.0 and sc.row0["eval_agent_det_npos_bus_all"] == 3.0
    assert sc.row0["eval_agent_det_ap1_bus_0_20"] == 1.0 and sc.row0["eval_box3d_det_ap4_heavy_truck_all"] == 1.0
    # the gating class (600 automobiles) and the class-agnostic cells are untouched by the swap
    for k in ("eval_agent_det_ap1_automobile_all", "eval_agent_det_ap1_all_all", "eval_agent_ap2m",
              "eval_agent_prec@gate", "eval_box3d_rec@gate"):
        assert sc.row_sw[k] == sc.row0[k], k
    # the rare-class APs collapse: the bus detections are now the heavy_truck objects' slots, at the wrong places
    assert sc.row_sw["eval_agent_det_ap1_bus_0_20"] == 0.0
    assert sc.row_sw["eval_box3d_det_ap4_heavy_truck_all"] == 0.0


def test_M5_with_the_columns_swapped_reads_DETECTED(m5_scenario):
    v = _m5_judge(m5_scenario, m5_scenario.row_sw)
    m5 = v["m5"]
    # 2 heads x (2 classes x 4 thresholds x 4 bands = 32 AP cells + 16 class-mean mAP cells) = 96 low-support members
    assert (m5["status"], m5["detected"], m5["N_M5"], m5["N_num"], m5["bound"]) == ("DETECTED", True, 96, 0, 5)
    assert m5["clause_population"] is True and m5["clause_gating_detection"] is False
    assert m5["blind_spot"] is None and MARK not in v["blind_spots_named"]
    assert m5["restoration_bit_exact"] is True
    assert v["G0"] == "PASS", v["reasons"]                            # M5 never gates
    assert v["mutation_detection"]["m5"] == {"detected": True, "n_terms_out": 96, "how": "DETECTED"}


def test_M5_with_the_columns_NOT_swapped_reads_UNDETECTED_and_names_the_blind_spot(m5_scenario):
    v = _m5_judge(m5_scenario, m5_scenario.row_ns)
    m5 = v["m5"]
    assert (m5["status"], m5["detected"], m5["N_M5"], m5["bound"]) == ("UNDETECTED", False, 0, 5)
    assert m5["blind_spot"] == "a rare-class index swap is invisible to G0"
    assert MARK in v["blind_spots_named"]
    assert v["G0"] == "PASS", v["reasons"]                            # reported, it never VOIDs G0 and never fails it
    assert not any("M5" in r or "m5" in r for r in v["reasons"])


def test_REGRESSION_a_probe_that_always_says_detected_is_caught_by_the_unswapped_control(m5_scenario):
    def always_detected(sc, row):
        return {"detected": True}
    assert always_detected(m5_scenario, m5_scenario.row_ns)["detected"] is True      # the defect: reads detected
    assert _m5_judge(m5_scenario, m5_scenario.row_ns)["m5"]["detected"] is False     # the shipped probe does not


def test_M5_is_also_detected_by_ONE_gating_detection_term_alone(m5_scenario):
    """Clause 2: 'any gating DETECTION term moves outside its tolerance'. N_M5 = 0 here, a pooled term moves 0.05."""
    sc = m5_scenario
    v = _m5_judge(sc, sc.row_ns, extra_gating_shift={"eval_agent_prec@gate": -0.05})
    m5 = v["m5"]
    assert (m5["status"], m5["detected"], m5["N_M5"]) == ("DETECTED", True, 0)
    assert m5["clause_population"] is False and m5["clause_gating_detection"] is True
    assert [t["term"] for t in m5["gating_detection_out"]] == ["eval_agent_prec@gate"]
    inside = _m5_judge(sc, sc.row_ns, extra_gating_shift={"eval_agent_prec@gate": -0.01})   # inside abs 0.02
    assert inside["m5"]["status"] == "UNDETECTED"


def test_M5_population_clause_uses_the_numerics_floor_2_N_num_plus_5(m5_scenario):
    """96 members move under M5. With a numerics arm that moves N_num of the same cells the bound is 2*N_num + 5:
    N_num = 45 -> bound 95 (96 > 95: DETECTED); N_num = 46 -> bound 97 (96 <= 97: UNDETECTED)."""
    sc = m5_scenario
    # the members the swap moves, enumerated independently of the judge: every per-class AP / class-mean mAP cell
    # whose value differs (the class-agnostic `all` cells and the n >= 30 automobile cells do not move)
    members = [k for k in sc.row0 if sc.row0[k] != sc.row_sw[k] and ("_det_ap" in k or "_det_map" in k)]
    assert len(members) == 96
    for n_num, want, bound in ((45, "DETECTED", 95), (46, "UNDETECTED", 97)):
        fp = dict(sc.row0)
        for k in members[:n_num]:
            fp[k] = round(float(fp[k]) + 0.05, 5)
        v = _m5_judge(sc, sc.row_sw, fp32=fp)
        assert (v["m5"]["N_num"], v["m5"]["bound"], v["m5"]["status"]) == (n_num, bound, want)
        assert v["m5"]["N_M5"] == 96


def test_M5_not_restored_bit_exactly_is_NOT_EVALUABLE_never_detected_and_never_a_reason(m5_scenario):
    v = _m5_judge(m5_scenario, m5_scenario.row_sw, exact=False)
    assert (v["m5"]["status"], v["m5"]["detected"]) == ("NOT EVALUABLE", None)
    assert v["m5"]["why"] == "the restoration of the slot heads was not shown bit-exact"
    assert v["G0"] == "PASS" and v["reasons"] == []                    # M5 is reported; it never gates
    assert MARK not in v["blind_spots_named"]                          # no evidence either way -> no blind spot named


def test_M5_that_raised_is_NOT_EVALUABLE_not_a_detection(m5_scenario):
    sc = m5_scenario
    v = G().judge(dict(sc.row0), _by_seed(sc.row0), _rec(dict(sc.row0), m5={"raised": "ValueError: no heads"}),
                  amend="A7")
    assert (v["m5"]["status"], v["m5"]["detected"]) == ("NOT EVALUABLE", None)
    # the generic mutations treat 'raised' as a detection (a loud failure IS one); M5 must NOT: it is a probe of G0's
    # comparison, and a crash of the probe's own swap says nothing about whether G0 would have seen the defect
    assert v["mutation_detection"]["m5"]["detected"] is None


def test_M5_without_the_numerics_arm_cannot_read_its_population_clause(m5_scenario):
    sc = m5_scenario
    v = _m5_judge(sc, sc.row_sw, fp32="__none__")
    assert v["m5"]["N_num"] is None and v["m5"]["bound"] is None and v["m5"]["clause_population"] is None
    assert (v["m5"]["status"], v["m5"]["detected"]) == ("NOT EVALUABLE", None)
    assert v["a7_lowsupport"]["status"] == "NOT EVALUABLE" and v["G0"] == "FAIL"     # the GUARD fails closed
    assert not any("M5" in r or "m5" in r for r in v["reasons"])


def test_M5_not_run_is_reported_as_such(m5_scenario):
    sc = m5_scenario
    v = G().judge(dict(sc.row0), _by_seed(sc.row0), _rec(dict(sc.row0)), amend="A7")
    assert (v["m5"]["status"], v["m5"]["detected"]) == ("NOT RUN", None)
    assert v["G0"] == "PASS"


def test_M5_never_appears_in_the_older_verdicts(m5_scenario):
    """M5 is an A7 probe: the as-registered / A5 / A6 verdicts of the same artifact do not change shape."""
    sc = m5_scenario
    rec = _rec(dict(sc.row0), m5={"row": dict(sc.row_sw), "restoration_bit_exact": True})
    for amend in (None, "A2", "A5", "A6"):
        v = G().judge(dict(sc.row0), _by_seed(sc.row0, 8 if amend in (None, "A2") else 24), rec, amend=amend)
        assert "m5" not in v["mutation_detection"] and "m5" not in v and MARK not in v["blind_spots_named"]
