"""Unit tests for the STRAT1 seeds read: the >= 3-of-4 rule with one seed per init (r2-r4), planted rows only, and the seeds_run commit widening (four new arms only)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat1_seeds_read as SS  # noqa: E402
import seeds_run as SRUN  # noqa: E402

CELLS = ("A", "B", "C")


def rows(cost=(), gain=()):
    return {c: {"cost": c in cost, "gain": c in gain} for c in CELLS}


def plant(cost_at, gain_at=None):
    """cost_at / gain_at: {seed: (cells)} -> s1c0 at the four seeds"""
    gain_at = gain_at or {}
    return {s: rows(cost_at.get(s, ()), gain_at.get(s, ())) for s in SS.SEEDS}


def test_three_of_four_across_inits_is_a_cost_and_heads_without_n1():
    lab, det = SS.seeds_outcome(plant({0: "A", 1: "A", 10: "A"}), {0: rows(), 1: rows()})
    assert det["COST4"] == {"A": [0, 1, 10]}
    assert lab == ["COSTS-HEADS"]


def test_two_of_four_is_split_not_a_cost():
    lab, det = SS.seeds_outcome(plant({0: "A", 10: "A"}), {0: rows(), 1: rows()})
    assert det["COST4"] == {}
    assert det["SPLIT-2of4"] == ["A"]
    assert lab == ["KEEPS"]


def test_two_of_four_across_both_inits_is_still_split():
    lab, det = SS.seeds_outcome(plant({0: "A", 1: "A"}), {0: rows(), 1: rows()})
    assert det["COST4"] == {} and det["SPLIT-2of4"] == ["A"] and lab == ["KEEPS"]


def test_one_init_only_never_qualifies_even_at_lower_k():
    assert SS.count_rule({0: {"A"}, 10: {"A"}, 1: set(), 11: set()}, k=2) == {}
    assert SS.count_rule({0: {"A"}, 1: {"A"}, 10: set(), 11: set()}, k=2) == {"A": [0, 1]}


def test_cost_with_n1_cost_at_both_first_seeds_is_nav():
    lab, det = SS.seeds_outcome(plant({0: "AB", 1: "AB", 10: "AB", 11: "B"}), {0: rows(cost="A"), 1: rows(cost="A")})
    assert det["costs_nav"] == ["A"] and det["costs_heads"] == ["B"]
    assert lab == ["COSTS-NAV", "COSTS-HEADS"]


def test_gain_at_three_of_four_without_costs_is_adds():
    lab, det = SS.seeds_outcome(plant({0: "B"}, {1: "C", 10: "C", 11: "C"}), {0: rows(), 1: rows()})
    assert lab == ["ADDS"] and det["GAIN4"] == {"C": [1, 10, 11]}


def test_missing_seed_is_not_yet():
    s = plant({})
    del s[11]
    assert SS.seeds_outcome(s, {0: rows(), 1: rows()})[0] == ["NOT-YET"]


def test_k_re_flags_a_changed_banked_row():
    pairs = {"S1-C0_s0": {"A": {"delta": 0.5, "lo": 0.1, "hi": 0.9, "cost": True, "gain": False}}}
    banked = {"families": {"LONGITUDINAL": {"cells": {"A": {"S1-C0_s0": dict(pairs["S1-C0_s0"]["A"])}}}}}
    assert SS.k_re(pairs, banked)[0]
    banked["families"]["LONGITUDINAL"]["cells"]["A"]["S1-C0_s0"]["delta"] = 0.5000001
    ok, bad = SS.k_re(pairs, banked)
    assert not ok and len(bad) == 1


def test_commit_widening_only_for_the_four_new_arms(tmp_path):
    RDS = SRUN.patch_rds()
    ck = tmp_path / "ckpt_last.pt"
    ck.write_bytes(b"x")
    want = RDS.RD.sha(str(ck))
    saved = dict(RDS.ARMS)
    try:
        for tag, commit, ok in (("STRAT1-S1_s10", SRUN.SEEDS_COMMIT, True), ("STRAT1-S1_s10", SRUN.L.COMMIT, False),
                                ("STRAT1-S1_s0", SRUN.SEEDS_COMMIT, False), ("STRAT1-S1_s0", SRUN.L.COMMIT, True)):
            man = tmp_path / f"m_{tag}_{commit[:6]}.json"
            man.write_text(json.dumps({"tag": tag[7:], "commit": commit, "files": {"ckpt_last.pt": {"sha256": want}}}), encoding="utf-8")
            RDS.ARMS[tag] = (str(ck), ("bank", str(man)))
            why = RDS.refusal(tag)
            assert (why is None) == ok, (tag, commit, why)
        assert SRUN.L.COMMIT.startswith("e81bde01")
    finally:
        RDS.ARMS.clear()
        RDS.ARMS.update(saved)
