"""anticheat_common.py -- pure helpers shared by the camera-blind roll, the controls and the tests (PREREG_ANTICHEAT_REFCV8.md).
NEW file (EvalFlyWheel 2026-10-27).  No torch, no model: numpy + stdlib only, so the controls are unit-testable on CPU in a second.

* sattolo_derangement(episodes, seed)   -- the fixed derangement over episodes (single cycle, no fixed point), random.Random(seed)
* donor_windows(index, pi)              -- own (e, t) -> donor (e_d, t_d): the SAME RANK in the donor episode's t-sorted window list
* digest_bytes(b)                       -- blake2b-8 as an int (what the roll records per window)
* check_changed_fraction(fed, ref)      -- control (b): the fraction of windows whose fed-frame digest differs; 1.0 required
* check_ego_equal(a, b)                 -- control (c): per-window digests of every non-frame input equal
* check_bank_equal(base, new)           -- control (a): traj_own / z_tac / sel_own bit-equal on every (e_i, t) key
No clip id is ever handled here: episodes are INDICES into the roll's manifest; the only names are counts and digests.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np


def sattolo_derangement(episodes, seed: int = 0) -> dict:
    """Sattolo's algorithm on the SORTED episode list with random.Random(seed): one n-cycle, hence no fixed point.
    pi[sorted[k]] = a[k].  Asserts the two properties the prereg states."""
    eps = sorted(int(e) for e in episodes)
    if len(eps) < 2 or len(set(eps)) != len(eps):
        raise ValueError("need >= 2 distinct episodes")
    a = list(eps)
    rnd = random.Random(int(seed))
    for i in range(len(a) - 1, 0, -1):
        j = rnd.randrange(i)
        a[i], a[j] = a[j], a[i]
    pi = {e: d for e, d in zip(eps, a)}
    assert all(pi[e] != e for e in eps), "derangement has a fixed point"
    assert sorted(pi.values()) == eps, "derangement is not a permutation (every episode must be a donor exactly once)"
    # single cycle: walk from the first episode, must visit all n before returning
    seen, cur = 0, eps[0]
    while True:
        cur = pi[cur]
        seen += 1
        if cur == eps[0]:
            break
    assert seen == len(eps), "derangement is not a single cycle"
    return pi


def donor_windows(index, pi: dict) -> dict:
    """index: iterable of (e, t).  Returns {(e, t): (e_d, t_d)}: the donor episode's window of the same rank (min(rank, n_donor - 1))."""
    by_e: dict = {}
    for e, t in index:
        by_e.setdefault(int(e), []).append(int(t))
    for e in by_e:
        by_e[e].sort()
    out = {}
    for e, ts in by_e.items():
        d = pi[e]
        dts = by_e[d]
        for r, t in enumerate(ts):
            out[(e, t)] = (d, dts[min(r, len(dts) - 1)])
    return out


def digest_bytes(b) -> int:
    """blake2b with an 8-byte digest, as a (non-negative) int < 2**64, so it fits a uint64 array."""
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def check_changed_fraction(fed, ref) -> dict:
    """Control (b).  fed, ref: equal-length uint64 digest arrays of the frames fed to the model in the blind and in the identity roll.
    PASS iff every window differs (fraction exactly 1.0).  A window whose digest is the sentinel 0 on either side counts as 'not changed'
    (never as agreement): an unrecorded window cannot pass."""
    fed, ref = np.asarray(fed, dtype=np.uint64), np.asarray(ref, dtype=np.uint64)
    if fed.shape != ref.shape or fed.size == 0:
        return {"PASS": False, "n": int(fed.size), "n_changed": 0, "fraction_changed": float("nan"), "reason": "shape mismatch or empty"}
    ok = (fed != ref) & (fed != 0) & (ref != 0)
    n, c = int(fed.size), int(ok.sum())
    return {"PASS": bool(c == n), "n": n, "n_changed": c, "fraction_changed": c / n}


def check_ego_equal(a, b) -> dict:
    """Control (c).  a, b: equal-length uint64 digest arrays of every non-frame input; PASS iff all equal and non-zero (0 = unrecorded)."""
    a, b = np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)
    if a.shape != b.shape or a.size == 0:
        return {"PASS": False, "n": int(a.size), "n_equal": 0, "reason": "shape mismatch or empty"}
    eq = (a == b) & (a != 0)
    return {"PASS": bool(eq.all()), "n": int(a.size), "n_equal": int(eq.sum())}


def check_bank_equal(base: dict, new: dict, keys=("traj_own", "z_tac", "sel_own")) -> dict:
    """Control (a).  base / new: dicts with e_i, t, and the arrays in `keys`.  Joined on (e_i, t).  PASS iff the key sets are equal and every
    array is bit-equal (max abs difference reads exactly 0.0; the byte strings are compared as well, so a -0.0 / NaN difference cannot hide)."""
    kb = {(int(e), int(t)): j for j, (e, t) in enumerate(zip(base["e_i"], base["t"]))}
    kn = {(int(e), int(t)): j for j, (e, t) in enumerate(zip(new["e_i"], new["t"]))}
    rec = {"n_base": len(kb), "n_new": len(kn), "keys_equal": bool(set(kb) == set(kn)), "arrays": {}}
    if not rec["keys_equal"] or not kb:
        rec["PASS"] = False
        return rec
    ks = sorted(kb)
    ib = np.array([kb[k] for k in ks])
    inn = np.array([kn[k] for k in ks])
    allok = True
    for k in keys:
        x, y = np.asarray(base[k])[ib], np.asarray(new[k])[inn]
        same_shape = x.shape == y.shape
        mx = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if same_shape and x.size else float("nan")
        beq = bool(same_shape and x.tobytes() == y.tobytes())
        rec["arrays"][k] = {"shape": list(x.shape), "max_abs_diff": mx, "bytes_equal": beq}
        allok = allok and beq and mx == 0.0
    rec["PASS"] = bool(allok)
    return rec
