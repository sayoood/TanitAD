"""Unit tests for the N1-at-seeds-10/11 read: NAV4 / HEADS4 / COST4 and the attribution table of PREREG_ADDENDUM_N1S10.md, planted rows only, one mutation per rule,
the K-RE control (including its vacuous case), and the n1_run arm classification and commit widening (the two N1 arms only)."""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n1_read as N  # noqa: E402
import n1_run as NRUN  # noqa: E402

SEEDS = N.SEEDS
CELLS = ("A", "B", "C")


def rows(cost=()):
    return {c: {"cost": c in cost, "gain": False} for c in CELLS}


def plant(cost_at):
    return {s: rows(cost_at.get(s, ())) for s in SEEDS}


def run(s1c0, n1c0, s1n1, rule=None):
    return N.n1_outcome(plant(s1c0), plant(n1c0), plant(s1n1), rule=rule)


# ---- the four attribution labels (addendum table)
def test_nav_only():
    attr, det = run({0: "A", 1: "A", 10: "A"}, {0: "A", 1: "A", 11: "A"}, {})
    assert attr == {"A": "NAV"} and det["COST4"] == {"A": [0, 1, 10]} and det["NAV4"] == {"A": [0, 1, 11]} and det["HEADS4"] == {}


def test_heads_only_even_with_n1_costing_at_two_seeds():
    attr, _ = run({0: "A", 1: "A", 10: "A"}, {0: "A", 1: "A"}, {0: "A", 10: "A", 11: "A"})
    assert attr == {"A": "HEADS"}


def test_both():
    attr, _ = run({0: "A", 1: "A", 10: "A"}, {1: "A", 10: "A", 11: "A"}, {0: "A", 1: "A", 11: "A"})
    assert attr == {"A": "BOTH"}


def test_neither():
    attr, _ = run({0: "A", 1: "A", 10: "A"}, {0: "A", 1: "A"}, {10: "A"})
    assert attr == {"A": "NEITHER"}


def test_only_cost4_cells_are_attributed():
    attr, det = run({0: "A", 10: "A"}, {0: "A", 1: "A", 10: "A"}, {0: "A", 1: "A", 10: "A"})
    assert attr == {} and det["COST4"] == {} and det["NAV4"] == {"A": [0, 1, 10]}


def test_cells_are_independent():
    attr, _ = run({0: "AB", 1: "AB", 10: "AB"}, {0: "A", 1: "A", 11: "A"}, {0: "B", 1: "B", 11: "B"})
    assert attr == {"A": "NAV", "B": "HEADS"}


def test_missing_seed_is_not_yet():
    s = plant({})
    del s[11]
    attr, det = N.n1_outcome(s, plant({}), plant({}))
    assert attr == {} and det["status"] == "NOT-YET"


def test_nav_uses_n1_minus_c0_and_heads_uses_s1_minus_n1_not_swapped():
    a1, _ = run({0: "A", 1: "A", 10: "A"}, {0: "A", 1: "A", 10: "A"}, {})
    a2, _ = run({0: "A", 1: "A", 10: "A"}, {}, {0: "A", 1: "A", 10: "A"})
    assert a1 == {"A": "NAV"} and a2 == {"A": "HEADS"}


# ---- mutations: each rule dropped must flip a synthetic case
def rule_no_init(per_seed, k=3):
    cells = set().union(*per_seed.values())
    return {c: sorted(s for s, v in per_seed.items() if c in v) for c in sorted(cells) if sum(c in v for v in per_seed.values()) >= k}


def rule_k2(per_seed, k=3):
    return N.SS.count_rule(per_seed, k=2)


def rule_k4(per_seed, k=3):
    return N.SS.count_rule(per_seed, k=4)


def test_mutation_drop_the_init_condition_flips_at_k2_and_is_implied_at_k3():
    # at the addendum's k = 3 of 4 the init condition is implied (any 3 of {0,10} + {1,11} hold a seed of each init): exhaustive over all 16 subsets
    for r in range(5):
        for sub in itertools.combinations(SEEDS, r):
            ps = {s: ({"A"} if s in sub else set()) for s in SEEDS}
            assert N.SS.count_rule(ps) == rule_no_init(ps), sub
    # with the card's k lowered to 2 the init condition is what separates {0, 10} (one init) from {0, 1}: dropping it flips the case
    one_init = {0: {"A"}, 10: {"A"}, 1: set(), 11: set()}
    assert N.SS.count_rule(one_init, k=2) == {} and rule_no_init(one_init, k=2) == {"A": [0, 10]}
    # and through the full outcome: a mutated rule (k = 2, no init) turns an unattributed 2-seed cost into a COST4 cell
    mut = lambda ps, k=3: rule_no_init(ps, k=2)  # noqa: E731
    attr, _ = run({0: "A", 10: "A"}, {0: "A", 10: "A"}, {}, rule=mut)
    assert attr == {"A": "NAV"}
    attr, _ = run({0: "A", 10: "A"}, {0: "A", 10: "A"}, {})
    assert attr == {}


def test_mutation_threshold_3_to_2_flips():
    case = ({0: "A", 1: "A"}, {0: "A", 1: "A"}, {})
    assert run(*case)[0] == {}
    assert run(*case, rule=rule_k2)[0] == {"A": "NAV"}


def test_mutation_threshold_3_to_4_flips():
    case = ({0: "A", 1: "A", 10: "A"}, {0: "A", 1: "A", 10: "A"}, {})
    assert run(*case)[0] == {"A": "NAV"}
    assert run(*case, rule=rule_k4)[0] == {}


def test_mutation_attribution_branches_each_distinct():
    assert N.attribution({"A": [0, 1, 10]}, {"A": [0, 1, 10]}, {}) == {"A": "NAV"}
    assert N.attribution({"A": [0, 1, 10]}, {}, {"A": [0, 1, 10]}) == {"A": "HEADS"}
    assert N.attribution({"A": [0, 1, 10]}, {"A": [0, 1, 10]}, {"A": [0, 1, 10]}) == {"A": "BOTH"}
    assert N.attribution({"A": [0, 1, 10]}, {}, {}) == {"A": "NEITHER"}
    assert set(N.NEXT_BY_LABEL) == {"NAV", "HEADS", "BOTH", "NEITHER"} and len(set(N.NEXT_BY_LABEL.values())) == 4


# ---- K-RE control
def _banked(delta=0.5):
    return {"families": {"TACTICAL": {"cells": {"A": {"N1-C0_s0": {"delta": delta, "lo": 0.1, "hi": 0.9, "cost": True, "gain": False}}}}}}


def test_k_re_passes_exact_and_fails_changed_or_vacuous():
    pairs = {"N1-C0_s0": {"A": {"delta": 0.5, "lo": 0.1, "hi": 0.9, "cost": True, "gain": False}}}
    ok, bad, cnt = N.k_re(pairs, [("b", _banked())])
    assert ok and cnt == {"b": 1}
    ok, bad, _ = N.k_re(pairs, [("b", _banked(0.5000001))])
    assert not ok and len(bad) == 1
    ok, bad, _ = N.k_re({"S1-C0_s99": pairs["N1-C0_s0"]}, [("b", _banked())])
    assert not ok and "vacuous" in bad[0]


# ---- n1_run: arm classification and commit widening
def test_n1_arms_are_text_without_ctx_heads_like_n1_s0():
    L = NRUN.patch_lib()
    for t in ("N1_s10", "N1_s11"):
        assert t in L.ARMS and t in L.TEXT_ARMS and t not in L.CTX_ARMS
    assert "N1_s0" in L.TEXT_ARMS and "N1_s0" not in L.CTX_ARMS
    assert "S1_s10" in L.CTX_ARMS and "C0_s10" not in L.TEXT_ARMS


def test_commit_widening_only_for_the_two_n1_arms(tmp_path):
    RDS = NRUN.patch_rds()
    ck = tmp_path / "ckpt_last.pt"
    ck.write_bytes(b"x")
    want = RDS.RD.sha(str(ck))
    saved = dict(RDS.ARMS)
    L = NRUN.L
    try:
        for tag, commit, ok in (("STRAT1-N1_s10", NRUN.SRUN.SEEDS_COMMIT, True), ("STRAT1-N1_s11", NRUN.SRUN.SEEDS_COMMIT, True), ("STRAT1-N1_s10", L.COMMIT, False),
                                ("STRAT1-N1_s0", NRUN.SRUN.SEEDS_COMMIT, False), ("STRAT1-N1_s0", L.COMMIT, True), ("STRAT1-S1_s10", NRUN.SRUN.SEEDS_COMMIT, True)):
            man = tmp_path / f"m_{tag}_{commit[:6]}.json"
            man.write_text(json.dumps({"tag": tag[7:], "commit": commit, "files": {"ckpt_last.pt": {"sha256": want}}}), encoding="utf-8")
            RDS.ARMS[tag] = (str(ck), ("bank", str(man)))
            why = RDS.refusal(tag)
            assert (why is None) == ok, (tag, commit, why)
        assert L.COMMIT.startswith("e81bde01")
    finally:
        RDS.ARMS.clear()
        RDS.ARMS.update(saved)
