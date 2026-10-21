"""Small formatting helpers for RESULT.md tables (CI cells)."""
import math


def f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def cell(c, nd=3):
    """marginal cell -> 'mean [lo, hi]'"""
    if c is None:
        return "n/a"
    return f"{f(c['mean'], nd)} [{f(c['lo'], nd)}, {f(c['hi'], nd)}]"


def dcell(c, nd=3):
    """paired cell -> '+delta [lo, hi] *'  (* = separated)"""
    if c is None:
        return "n/a"
    s = "SEP" if c["separated"] else "ns"
    return f"{c['delta']:+.{nd}f} [{f(c['lo'], nd)}, {f(c['hi'], nd)}] {s}"
