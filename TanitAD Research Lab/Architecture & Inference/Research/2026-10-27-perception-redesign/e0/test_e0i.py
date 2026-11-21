"""Known-answer test for e0i_split.py (tier 2: one known answer, no mutation suite).

The expectation is a LITERAL: sha256("a") = ca97..., sha256("b") = 3e23..., sha256("c") = 2e7d... (standard digests).  Sorted
ascending the order is c, b, a, so rank 0 = c (DEV), rank 1 = b (HOLD), rank 2 = a (DEV).
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0i_split as S  # noqa: E402

LIT = {"a": "ca978112ca1b", "b": "3e23e8160039", "c": "2e7d2c03a950"}


def test_literals_are_the_real_digests():
    for k, v in LIT.items():
        assert hashlib.sha256(k.encode()).hexdigest().startswith(v)


def test_known_rank_parity():
    dev, hold = S.split_ids(["a", "b", "c"])
    assert [h[:12] for h in dev] == [LIT["c"], LIT["a"]]
    assert [h[:12] for h in hold] == [LIT["b"]]


def test_order_does_not_depend_on_input_order():
    assert S.split_ids(["c", "a", "b"]) == S.split_ids(["b", "c", "a"]) == S.split_ids(["a", "b", "c"])


def test_build_counts_and_no_raw_ids_in_output():
    w2 = {"a": [1, 2, 3], "b": [4], "c": [5, 6]}
    w6 = {"a": [1], "b": [], "c": [5, 6]}
    o = S.build(w2, w6)
    assert o["DEV"]["n_episodes"] == 2 and o["HOLD"]["n_episodes"] == 1
    assert (o["DEV"]["n_windows_s2"], o["HOLD"]["n_windows_s2"]) == (5, 1)       # DEV = c (2) + a (3); HOLD = b (1)
    assert (o["DEV"]["n_windows_s6"], o["HOLD"]["n_windows_s6"]) == (3, 0)
    blob = repr(o)
    assert "'a'" not in blob and "'b'" not in blob and "'c'" not in blob


def test_mismatched_grids_refused():
    try:
        S.build({"a": [1]}, {"b": [1]})
    except ValueError:
        return
    raise AssertionError("expected ValueError")
