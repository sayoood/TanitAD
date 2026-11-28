"""Unit tests for the S1d read: the detach outcome (CURED / PARTIAL / REMAINS, NEW-COST, GAIN4), one mutation per rule, DETECT-LOST, the T0 view builder, and the s1d_run arm
classification and commit widening (the four S1d arms only)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1d_read as D  # noqa: E402
import s1d_run as SRUN  # noqa: E402
import s1d_view as V  # noqa: E402

SEEDS = D.SEEDS
CELLS = ("A", "B", "C")


def rows(cost=(), gain=()):
    return {c: {"cost": c in cost, "gain": c in gain} for c in CELLS}


def plant(cost_at=None, gain_at=None):
    cost_at, gain_at = cost_at or {}, gain_at or {}
    return {s: rows(cost_at.get(s, ()), gain_at.get(s, ())) for s in SEEDS}


S1COST = {0: "A", 1: "A", 10: "A"}


def test_cured():
    lab, det = D.s1d_outcome(plant(S1COST), plant(), plant({}, {0: "A", 1: "A", 11: "A"}))
    assert lab == {"A": "CURED"} and det["COST4_S1"] == {"A": [0, 1, 10]} and det["COST4_S1d"] == {} and det["GAIN4_DET"] == {"A": [0, 1, 11]}


def test_partial_no_cost_but_no_separated_detach_gain():
    lab, _ = D.s1d_outcome(plant(S1COST), plant({0: "A"}), plant())
    assert lab == {"A": "PARTIAL"}


def test_remains_even_when_detach_gain_is_separated():
    lab, _ = D.s1d_outcome(plant(S1COST), plant({0: "A", 1: "A", 11: "A"}), plant({}, {0: "A", 1: "A", 11: "A"}))
    assert lab == {"A": "REMAINS"}


def test_new_cost_listed_not_hidden_and_not_labelled():
    lab, det = D.s1d_outcome(plant(S1COST), plant({0: "B", 1: "B", 10: "B"}), plant())
    assert lab == {"A": "PARTIAL"} and det["NEW-COST"] == {"B": [0, 1, 10]}


def test_s1d_gain_vs_c0_listed():
    _, det = D.s1d_outcome(plant(), plant({}, {1: "C", 10: "C", 11: "C"}), plant())
    assert det["GAIN4_S1d_vs_C0"] == {"C": [1, 10, 11]}


def test_two_of_four_s1_cost_is_not_cost4_so_nothing_to_cure():
    lab, det = D.s1d_outcome(plant({0: "A", 1: "A"}), plant(), plant())
    assert lab == {} and det["COST4_S1"] == {}


def test_missing_seed_is_not_yet():
    s = plant()
    del s[10]
    assert D.s1d_outcome(plant(), s, plant())[1]["status"] == "NOT-YET"


# ---- mutations: each rule dropped must flip a synthetic case
def test_mutation_drop_remains_branch_flips():
    case = (plant(S1COST), plant(S1COST), plant())
    assert D.s1d_outcome(*case)[0] == {"A": "REMAINS"}
    # a rule that returns no S1d costs (the remains branch dropped) turns REMAINS into PARTIAL
    calls = {"n": 0}

    def rule(ps, k=3):
        calls["n"] += 1
        return {} if calls["n"] == 2 else D.SS.count_rule(ps, k)   # second call = COST4_S1d
    assert D.s1d_outcome(*case, rule=rule)[0] == {"A": "PARTIAL"}


def test_mutation_threshold_3_to_2_flips():
    case = (plant({0: "A", 1: "A"}), plant(), plant({}, {0: "A", 1: "A"}))
    assert D.s1d_outcome(*case)[0] == {}
    assert D.s1d_outcome(*case, rule=lambda ps, k=3: D.SS.count_rule(ps, k=2))[0] == {"A": "CURED"}


def test_mutation_cured_needs_the_detach_gain():
    assert D.s1d_outcome(plant(S1COST), plant(), plant())[0] == {"A": "PARTIAL"}
    assert D.s1d_outcome(plant(S1COST), plant(), plant({}, {0: "A", 1: "A", 10: "A"}))[0] == {"A": "CURED"}


def test_mutation_swapped_pairs_flip():
    # S1d - C0 and S1 - C0 swapped: a cost under S1d only is read as the S1 cost
    lab, det = D.s1d_outcome(plant(), plant(S1COST), plant())
    assert lab == {} and det["NEW-COST"] == {"A": [0, 1, 10]}


def test_labels_and_next_steps_are_distinct():
    assert set(D.NEXT_BY_LABEL) == {"CURED", "PARTIAL", "REMAINS"} and len(set(D.NEXT_BY_LABEL.values())) == 3


def test_detect_lost():
    s1 = {"0_1": {"weather": "NOT-DETECTED(weather)", "country": "DETECTS(country)", "surface": "DETECTS(surface)"}}
    s1d = {"0_1": {"weather": "DETECTS(weather)", "country": "NOT-DETECTED(country)", "surface": "DETECTS(surface)"}}
    assert D.detect_lost(s1, s1d) == {"0_1": ["country"]}
    assert D.detect_lost(s1, {"0_1": {}}) == {"0_1": ["country", "surface"]}


# ---- the T0 view builder
def test_view_renames_s1d_to_s1_and_copies_c0(tmp_path):
    t0, view = tmp_path / "t0", tmp_path / "view"
    t0.mkdir()
    for src, _ in V.plan(str(t0)):
        open(src, "wb").write(os.path.basename(src).encode())
    n = V.build(str(t0), str(view))
    assert n == 16 + 15
    assert (view / "eval_STRAT1-S1_s10_k4.npz").read_bytes() == b"eval_STRAT1-S1d_s10_k4.npz"
    assert (view / "eval_STRAT1-C0_rep_zero.npz").read_bytes() == b"eval_STRAT1-C0_rep_zero.npz"
    assert not any("S1d" in f for f in os.listdir(view))
    os.remove(t0 / "eval_STRAT1-S1d_s0_real.npz")
    try:
        V.build(str(t0), str(tmp_path / "v2"))
        raise AssertionError("a missing pass must refuse")
    except FileNotFoundError:
        pass


# ---- s1d_run: arm classification and commit widening
def test_s1d_arms_are_text_with_ctx_heads_like_s1():
    L = SRUN.patch_lib()
    for t in SRUN.S1D:
        assert t in L.ARMS and t in L.TEXT_ARMS and t in L.CTX_ARMS
    assert "S1_s0" in L.CTX_ARMS and "N1_s10" in L.TEXT_ARMS and "N1_s10" not in L.CTX_ARMS


def test_commit_widening_only_for_the_four_s1d_arms(tmp_path):
    RDS = SRUN.patch_rds()
    ck = tmp_path / "ckpt_last.pt"
    ck.write_bytes(b"x")
    want = RDS.RD.sha(str(ck))
    saved = dict(RDS.ARMS)
    L = SRUN.L
    seeds_commit = "a4357e7986f3aa30dab2ed83fde2351fda8d5662"
    try:
        for tag, commit, ok in (("STRAT1-S1d_s0", SRUN.S1D_COMMIT, True), ("STRAT1-S1d_s11", SRUN.S1D_COMMIT, True), ("STRAT1-S1d_s10", L.COMMIT, False), ("STRAT1-S1d_s10", seeds_commit, False),
                                ("STRAT1-S1_s0", SRUN.S1D_COMMIT, False), ("STRAT1-S1_s0", L.COMMIT, True), ("STRAT1-N1_s10", SRUN.S1D_COMMIT, False), ("STRAT1-N1_s10", seeds_commit, True)):
            man = tmp_path / f"m_{tag}_{commit[:6]}.json"
            man.write_text(json.dumps({"tag": tag[7:], "commit": commit, "files": {"ckpt_last.pt": {"sha256": want}}}), encoding="utf-8")
            RDS.ARMS[tag] = (str(ck), ("bank", str(man)))
            why = RDS.refusal(tag)
            assert (why is None) == ok, (tag, commit, why)
        assert L.COMMIT.startswith("e81bde01")
    finally:
        RDS.ARMS.clear()
        RDS.ARMS.update(saved)
