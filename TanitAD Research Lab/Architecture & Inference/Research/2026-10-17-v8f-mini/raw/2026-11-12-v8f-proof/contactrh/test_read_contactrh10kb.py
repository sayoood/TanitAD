"""Tests of read_contactrh10kb.py (card ADD142): the derived names, Q3 of record and control c5, on synthetic values only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_contactrh10kb as T  # noqa: E402


def iv(lo, hi):
    return {"mean": (lo + hi) / 2, "interval": [lo, hi]}


def vals(q1, q2, q3, nx=19, ns=51):
    return {"Q1": q1, "Q2": q2, "Q3": q3, "n_X": nx, "n_S": ns}


def test_constants():
    assert T.CARD == "PREREG_V8F_PROOF_ADD142" and T.PRE == "CONTACTRH10KB"
    assert (T.TAG_H, T.TAG_U) == ("H_RQ50Kx", "U_RQ50Kx") and T.BOOT_SEED == 142


def test_names_each_reachable():
    z = iv(-1.0, 1.0)
    p = iv(0.2, 0.8)
    assert T.derive_name(vals(z, z, p))[0] == "CONTACTRH10KB-SAME-READ"
    assert T.derive_name(vals(p, z, z))[0] == "CONTACTRH10KB-MISREAD"
    assert T.derive_name(vals(z, p, p))[0] == "CONTACTRH10KB-BOTH"
    assert T.derive_name(vals(z, z, z))[0] == "CONTACTRH10KB-UNRESOLVED"
    assert T.derive_name(vals(z, z, p, nx=9))[0] == "CONTACTRH10KB-UNRESOLVED"
    assert T.derive_name(vals(z, z, p, ns=9))[0] == "CONTACTRH10KB-UNRESOLVED"


def _tabs(c_h, c_u, c2_h, c2_u):
    a = {"i": np.array([1, 2, 3]), "clos": np.array(c_h, float), "clos2": np.array(c2_h, float), "ep": np.array([0, 1, 2])}
    b = {"i": np.array([1, 2, 3]), "clos": np.array(c_u, float), "clos2": np.array(c2_u, float), "ep": np.array([0, 1, 2])}
    return {0: {"S": a, "S_U": b}}


def test_c5_passes_on_identity_and_fails_otherwise():
    ok = T.ctl_c5(_tabs([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [0.5, 2.0, 3.0]))
    assert ok["PASS"] and ok["n_pairs_v0_minus_vL_differ"] == 1
    bad1 = T.ctl_c5(_tabs([0.1, 0.2, 0.3], [0.1, 0.2, 0.31], [1.0, 2.0, 3.0], [0.5, 2.0, 3.0]))
    assert not bad1["PASS"]
    bad2 = T.ctl_c5(_tabs([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    assert not bad2["PASS"]


def test_q3_of_record_is_v0_minus_vl():
    idx = T.make_draws(3, 50, 1)
    tabs = _tabs([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    r = T.paired_closing(tabs, "clos2", 3, idx)
    assert abs(r["value"] - 1.0) < 1e-12
