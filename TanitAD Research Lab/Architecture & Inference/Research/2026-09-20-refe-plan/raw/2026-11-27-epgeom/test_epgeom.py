"""Synthetic tests for EP-GEOM (PREREG_EPGEOM.md section 6). CPU, numpy only:  python test_epgeom.py   (or pytest).
Includes CTRL-ID and a mutation test: three source mutations of epgeom_lib.py must each turn `core_checks` RED."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epgeom_lib as L  # noqa: E402


def synth(n=40, seed=0):
    r = np.random.default_rng(seed)
    logits = r.normal(1.5, 2.0, (n, 64, 6)).astype(np.float32)
    props8 = np.cumsum(r.normal(1.0, 0.5, (n, 64, 8, 3)), axis=2).astype(np.float32)
    pick = L.aggregate_v1(logits).argmax(1).astype(np.int64)
    return logits, props8, pick


def core_checks(lib):
    """Returns the list of FAILED check names for a lib module (so a mutated copy can be asserted RED)."""
    bad = []
    logits, props8, pick = synth()
    # CTRL-ID: tau above every q reproduces C1's pick bit for bit, for every surrogate
    for s in lib.SURROGATES:
        if not np.array_equal(lib.select(logits, props8, pick, s, 1.5), pick):
            bad.append("ctrl_id_" + s)
    # tau = 0: S = everything, so the rule is the plain argmax of g (random floats, no ties)
    for s in lib.SURROGATES:
        g = lib.surrogate(s, logits, props8)
        if not np.array_equal(lib.select(logits, props8, pick, s, 0.0), g.argmax(1)):
            bad.append("tau0_argmax_" + s)
    # the mask matters: a proposal with huge g but low q must not be picked; a safe one is
    lg = np.full((1, 64, 6), 6.0, np.float32)
    lg[0, 5, [0, 1, 3, 5]] = -6.0                                   # proposal 5 unsafe
    p8 = np.zeros((1, 64, 8, 3), np.float32)
    p8[0, 5, :, 0] = np.arange(1, 9) * 100.0                        # longest by far
    p8[0, 9, :, 0] = np.arange(1, 9) * 1.0                          # longest SAFE
    if int(lib.select(lg, p8, np.array([0]), "arc", 0.9)[0]) != 9:
        bad.append("mask_excludes_unsafe")
    # empty safe set -> C1's pick
    if int(lib.select(lg - 20.0 * (np.arange(6) != 2), p8, np.array([17]), "arc", 0.9)[0]) != 17:
        bad.append("empty_set_fallback")
    # arc length includes the origin -> pose-1 segment (hand value: pose1 at 3 m, then +1 m each => 3 + 7 = 10)
    p = np.zeros((1, 64, 8, 3), np.float32)
    p[0, 0, :, 0] = 3.0 + np.arange(8)
    if abs(float(lib.surrogate("arc", np.zeros((1, 64, 6), np.float32), p)[0, 0]) - 10.0) > 1e-5:
        bad.append("arc_includes_origin_segment")
    # tie -> C1's pick when it is among the tied; else the aggregate's argmax over the tied
    lg2 = np.full((1, 64, 6), 6.0, np.float32)
    p0 = np.zeros((1, 64, 8, 3), np.float32)                        # g identical for all -> all tied
    if int(lib.select(lg2, p0, np.array([33]), "arc", 0.5)[0]) != 33:
        bad.append("tie_keeps_c1_pick")
    lg3 = lg2.copy()
    lg3[0, 12, 2] = 9.0                                             # best aggregate (EP) inside the tied set
    pk3 = np.array([40])
    mk = np.zeros((1, 64), bool); mk[0, [12, 20]] = True
    q = np.where(mk, 0.99, 0.0).astype(np.float32)
    if int(lib.select(lg3, p0, pk3, "arc", 0.5, q=q)[0]) != 12:
        bad.append("tie_uses_aggregate_when_c1_pick_outside")
    # CTRL-RED: a one-index misalignment of g changes the pick
    if np.array_equal(lib.select(logits, props8, pick, "arc", 0.0, roll=1), lib.select(logits, props8, pick, "arc", 0.0)):
        bad.append("ctrl_red")
    return bad


def test_core_green():
    assert core_checks(L) == []


def test_mutations_go_red():
    src = open(os.path.join(HERE, "epgeom_lib.py")).read()
    muts = {
        "mask_ignored": ("mask = q >= np.float32(tau)", "mask = q >= np.float32(-1.0)"),
        "origin_segment_dropped": ("pts = np.concatenate([np.zeros(xy.shape[:-2] + (1, 2), np.float32), xy], axis=-2)", "pts = xy"),
        "empty_set_not_c1": ("return np.where(empty, pick, out).astype(np.int64)", "return out.astype(np.int64)"),
    }
    for name, (a, b) in muts.items():
        assert a in src, name
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMP")) as td:
            p = os.path.join(td, "epgeom_lib_mut.py")
            open(p, "w").write(src.replace(a, b))
            spec = importlib.util.spec_from_file_location("epgeom_lib_mut", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            assert core_checks(m), f"mutation {name} stayed GREEN"


def test_fit_recovers_planted_signal():
    r = np.random.default_rng(5)
    n = 400
    logits = r.normal(2.0, 1.0, (n, 64, 6)).astype(np.float32)
    props8 = np.cumsum(np.abs(r.normal(1.0, 0.4, (n, 64, 8, 3))), axis=2).astype(np.float32)
    pick = L.aggregate_v1(logits).argmax(1).astype(np.int64)
    q = L.safe_q(logits)
    arc = L.surrogate("arc", logits, props8)
    # label: pdms is high for the longest SAFE (q>=0.5) proposal, otherwise lower
    pdms = np.where(q >= 0.5, 0.5 + 0.5 * arc / arc.max(1, keepdims=True), 0.2).astype(np.float64)
    best, table = L.fit(logits, props8, pick, pdms)
    assert best["surrogate"] == "arc", best
    base = float(L.pdms_at(pdms, pick).mean())
    assert best["mean_pdms"] > base


def test_boot_paired_seeded():
    logs = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    v = np.arange(15.0)
    assert L.Boot(logs).ci(v) == L.Boot(logs).ci(v)
    assert L.Boot(logs).D.shape == (L.NBOOT, 3)


def test_aggregate_matches_hand():
    lg = np.zeros((1, 2, 6), np.float32)       # sigmoid 0.5 everywhere
    a = L.aggregate_v1(lg)
    assert abs(float(a[0, 0]) - 0.5 * 0.5 * (5 * .5 + 5 * .5 + 2 * .5) / 12) < 1e-7


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)}/{len(fns)} PASS")
