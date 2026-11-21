#!/usr/bin/env python3
"""Self-test for SPEC_NAVTEST Amendment 7's adopted repair (`refe/planner.py: repair_last_heading`, `REFePlanner.executed`).

Checks, each with a deliberate-regression arm that must FAIL:
  T1 positions and headings 0..T-2 are bit-identical; heading[T-1] equals heading[T-2]; the input is not modified
     (numpy AND torch, float32 AND float64, batched [M, T, 3] and single [T, 3])
  T2 `REFePlanner.executed(traj, k)` returns the repaired proposal k when the repair is on, and proposal k untouched
     (bit-identical, same object values) when it is off -- the switch that reproduces every pre-Amendment-7 evaluation
  T3 the real ep015 native proposals: after the repair the last-pose heading error against the path tangent falls to
     the other steps' level (measured, not asserted from the construction)
Run with the eval interpreter (it imports planner.py): driverl-eval + eval_checkpoint.env_driverl().
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "refe"))
import planner as PL  # noqa: E402


def check_repair(fn, arr) -> list[str]:
    """the invariants of a correct repair; returns the names of the checks that FAIL"""
    fails = []
    before = arr.clone() if isinstance(arr, torch.Tensor) else np.array(arr, copy=True)
    out = fn(arr)
    a = before.numpy() if isinstance(before, torch.Tensor) else before
    o = out.numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
    now = arr.numpy() if isinstance(arr, torch.Tensor) else arr
    if type(out) is not type(arr) or o.dtype != a.dtype or o.shape != a.shape:
        fails.append("type/dtype/shape")
    if not np.array_equal(now, a):
        fails.append("input modified")
    if not np.array_equal(o[..., :-1, :], a[..., :-1, :]):
        fails.append("poses 0..T-2 changed")
    if not np.array_equal(o[..., -1, :2], a[..., -1, :2]):
        fails.append("last position changed")
    if not np.array_equal(o[..., -1, 2], a[..., -2, 2]):
        fails.append("last heading != previous heading")
    return fails


def main() -> int:
    ok = True
    rng = np.random.default_rng(20260927)
    cases = []
    for dt in (np.float32, np.float64):
        x = rng.normal(size=(64, 20, 3)).astype(dt)
        x[..., 2] *= 4.0                                   # headings well outside [-pi, pi], like the defect
        cases += [("numpy", x), ("numpy-1", x[3].copy()), ("torch", torch.from_numpy(x.copy())),
                  ("torch-1", torch.from_numpy(x[5].copy()))]
    for name, arr in cases:
        f = check_repair(PL.repair_last_heading, arr)
        print(f"  [{'PASS' if not f else 'FAIL'}] T1 {name} {arr.dtype}: {f or 'all invariants hold'}")
        ok &= not f

    # deliberate regressions: each must FAIL at least one invariant
    def wrong_index(t):                                    # repairs the pose BEFORE the last
        o = np.array(t, copy=True); o[..., -2, 2] = o[..., -3, 2]; return o

    def moves_position(t):                                  # also moves the last position
        o = np.array(t, copy=True); o[..., -1, 2] = o[..., -2, 2]; o[..., -1, 0] += 1e-3; return o

    def in_place(t):                                        # modifies the caller's array
        t[..., -1, 2] = t[..., -2, 2]; return t

    def identity(t):                                        # does nothing
        return np.array(t, copy=True)

    x = rng.normal(size=(64, 20, 3))
    x[..., 2] *= 4.0
    for name, fn in (("wrong index", wrong_index), ("moves position", moves_position), ("in place", in_place),
                     ("identity", identity)):
        f = check_repair(fn, np.array(x, copy=True))
        print(f"  [{'PASS' if f else 'FAIL'}] MUTATION '{name}' is caught: {f}")
        ok &= bool(f)

    # T2: executed() with the switch on and off, without building a model
    p = PL.REFePlanner.__new__(PL.REFePlanner)
    t = torch.from_numpy(x.astype(np.float32))
    p.repair_last_heading = True
    on = p.executed(t, 7)
    p.repair_last_heading = False
    off = p.executed(t, 7)
    t2_on = torch.equal(on[:, :2], t[7, :, :2]) and torch.equal(on[:-1], t[7, :-1]) and float(on[-1, 2]) == float(t[7, -2, 2])
    t2_off = torch.equal(off, t[7])
    print(f"  [{'PASS' if t2_on else 'FAIL'}] T2 executed() ON = repaired proposal 7")
    print(f"  [{'PASS' if t2_off else 'FAIL'}] T2 executed() OFF = proposal 7 untouched")
    print(f"  [{'PASS' if PL.REFePlanner.REPAIR_LAST_HEADING is True else 'FAIL'}] T2 the adopted default is ON")
    ok &= t2_on and t2_off and PL.REFePlanner.REPAIR_LAST_HEADING is True

    # T3: the real defect, measured before and after on the ep015 native proposals
    dump = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/stop_candidate_dump.npz"
    if os.path.exists(dump):
        T = np.load(dump)["traj"].astype(np.float64)

        def last_err(P):
            xy = np.concatenate([np.zeros_like(P[..., :1, :2]), P[..., :2]], axis=-2)
            d = np.diff(xy, axis=-2)
            tang = np.arctan2(d[..., 1], d[..., 0])
            mv = np.linalg.norm(d, axis=-1) > 0.2
            e = np.abs((P[..., 2] - tang + np.pi) % (2 * np.pi) - np.pi)
            return float(np.median(e[..., -1][mv[..., -1]])), float(np.median(e[..., :-1][mv[..., :-1]]))

        b_last, b_rest = last_err(T)
        a_last, a_rest = last_err(PL.repair_last_heading(T))
        t3 = a_last < 3 * b_rest and b_last > 10 * b_rest and a_rest == b_rest
        print(f"  [{'PASS' if t3 else 'FAIL'}] T3 ep015 native: last-pose error {b_last:.3f} -> {a_last:.3f} rad "
              f"(other steps {b_rest:.3f}, unchanged)")
        ok &= t3
    else:
        print("  [SKIP] T3: the ep015 dump is not on this machine")
    print("ZZSELFTEST_OK" if ok else "ZZSELFTEST_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
