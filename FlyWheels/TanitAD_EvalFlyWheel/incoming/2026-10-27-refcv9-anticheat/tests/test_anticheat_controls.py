"""test_anticheat_controls.py -- the control checkers of PREREG_ANTICHEAT_REFCV8.md must read known values and go RED on a deliberate regression.
CPU, numpy only.  Run:  python -X utf8 -m pytest -q -p no:cacheprovider <this file>"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import anticheat_common as AC  # noqa: E402

# LITERAL pin of the registered derangement (139 episodes, seed 0): sha256 of "e:d," over the sorted episodes.
PI139_SHA256 = "48ff98cb395ef572ccbbadfcad59fd8be883666b4d485631817f3f1b0d33ba43"


def _pi139():
    return AC.sattolo_derangement(range(139), 0)


def test_derangement_properties():
    pi = _pi139()
    assert len(pi) == 139
    assert all(pi[e] != e for e in pi)                       # no fixed point
    assert sorted(pi.values()) == list(range(139))           # a permutation: each episode a donor exactly once
    cur, n = 0, 0                                            # a single 139-cycle
    while True:
        cur, n = pi[cur], n + 1
        if cur == 0:
            break
    assert n == 139


def test_derangement_is_fixed_and_seed_dependent():
    assert AC.sattolo_derangement(range(139), 0) == AC.sattolo_derangement(range(139), 0)
    assert AC.sattolo_derangement(range(139), 0) != AC.sattolo_derangement(range(139), 1)
    s = "".join(f"{e}:{d}," for e, d in sorted(_pi139().items()))
    assert hashlib.sha256(s.encode()).hexdigest() == PI139_SHA256


def test_small_literal():
    # n = 2 has exactly one derangement: the swap.  n = 1 is refused.
    assert AC.sattolo_derangement([7, 3], 0) == {3: 7, 7: 3}
    try:
        AC.sattolo_derangement([1], 0)
        assert False
    except ValueError:
        pass


def test_donor_windows_same_rank_and_clamp():
    index = [(0, 10), (0, 20), (0, 30), (1, 5), (1, 15)]
    pi = {0: 1, 1: 0}
    m = AC.donor_windows(index, pi)
    assert m[(0, 10)] == (1, 5) and m[(0, 20)] == (1, 15) and m[(0, 30)] == (1, 15)     # rank 2 clamps to the donor's last window
    assert m[(1, 5)] == (0, 10) and m[(1, 15)] == (0, 20)
    assert all(m[k][0] != k[0] for k in m)


def test_changed_fraction_known_values_and_mutation_red():
    rng = np.random.default_rng(0)
    ref = rng.integers(1, 2 ** 62, size=1000, dtype=np.uint64)
    fed = ref + np.uint64(1)                                  # every window differs
    assert AC.check_changed_fraction(fed, ref)["PASS"] is True
    assert AC.check_changed_fraction(fed, ref)["fraction_changed"] == 1.0
    half = fed.copy()
    half[::2] = ref[::2]                                      # MUTATION: half of the windows NOT swapped -> must be RED
    r = AC.check_changed_fraction(half, ref)
    assert r["PASS"] is False and r["fraction_changed"] == 0.5
    assert AC.check_changed_fraction(ref, ref)["PASS"] is False                       # identity vs identity: nothing changed
    z = fed.copy()
    z[3] = 0                                                  # an unrecorded window (sentinel 0) can never pass
    assert AC.check_changed_fraction(z, ref)["PASS"] is False


def test_ego_equal_mutation_red():
    a = np.arange(1, 501, dtype=np.uint64)
    assert AC.check_ego_equal(a, a.copy())["PASS"] is True
    b = a.copy()
    b[17] ^= np.uint64(1)                                     # ONE window's ego input digest differs
    r = AC.check_ego_equal(a, b)
    assert r["PASS"] is False and r["n_equal"] == 499
    assert AC.check_ego_equal(np.zeros(5, np.uint64), np.zeros(5, np.uint64))["PASS"] is False   # all-unrecorded never passes


def _bank(seed=0, n=50):
    rng = np.random.default_rng(seed)
    return {"e_i": np.repeat(np.arange(5), 10).astype(np.int64), "t": np.tile(np.arange(10), 5).astype(np.int64),
            "traj_own": rng.standard_normal((n, 8, 2)).astype(np.float32), "z_tac": rng.standard_normal((n, 16)).astype(np.float32),
            "sel_own": rng.integers(0, 117, n).astype(np.int64)}


def test_bank_equal_and_one_ulp_mutation_red():
    base = _bank()
    new = {k: v.copy() for k, v in base.items()}
    perm = np.random.default_rng(1).permutation(50)           # row ORDER is irrelevant: the join is on (e_i, t)
    new = {k: v[perm] for k, v in new.items()}
    assert AC.check_bank_equal(base, new)["PASS"] is True
    mut = {k: v.copy() for k, v in new.items()}
    mut["traj_own"][4, 3, 1] = np.nextafter(mut["traj_own"][4, 3, 1], np.float32(1e9))   # MUTATION: one float32 ulp
    r = AC.check_bank_equal(base, mut)
    assert r["PASS"] is False and r["arrays"]["traj_own"]["max_abs_diff"] > 0.0
    drop = {k: v[1:] for k, v in new.items()}                  # a missing window
    assert AC.check_bank_equal(base, drop)["PASS"] is False
