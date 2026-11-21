#!/usr/bin/env python3
"""Tests of a2_read.py (Amendment A2's reader).  Plain python:  python -B -I test_a2_read.py  ->  ZZTEST_PASS <n>.

Each instrument check has a deliberate-regression arm (the test passes only if the MUTATED path raises): the identity test (a perturbed readout, a different bootstrap seed, a wrong config), the zero
control (a contrast that leaks a constant), the floor-vs-text check, the config-sha refusal.  The committed outcome table is checked against an INDEPENDENT oracle over a sweep, including the
boundaries on unrounded values and the ORDER (U1 before U2).  The last test runs the identity test and zero control on the REAL banked V-s0 (2,000-update) units against the real PROBE_RESULT.json
and writes A2_IDENTITY.json -- the number the brief asks for: V seed-0 held-out IoU 0.6348906684369943 reproduced exactly.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import scene_tap_probe as S  # noqa: E402
import a2_read as R  # noqa: E402
from test_scene_tap_probe import synth_world, patched, expect_guard, many_patches  # noqa: E402

TESTS = []
_FIX = {}


def test(fn):
    TESTS.append(fn)
    return fn


def fixture():
    """A synthetic banked grid (35 units at n_updates=2, depth-1 heads) assembled into a banked-style PROBE_RESULT.json, plus A2 units: V seed 0 at n_updates=3 in a separate work dir."""
    if _FIX:
        return _FIX
    U, ctxs, lab, src = synth_world(21)
    tmp = tempfile.mkdtemp(prefix="stp_a2read_")
    banked_work, a2_work = os.path.join(tmp, "banked"), os.path.join(tmp, "a2")
    with patched(S, "DEPTH", 1):
        for tap in ("visual", "scene"):
            for seed in (0, 1, 2):
                S.run_init(U, ctxs, tap, seed, device="cpu", root=banked_work)
        for (arm, seed, fold) in S.UNIT_ORDER:
            S.run_unit(U, lab, ctxs, arm, seed, fold, n_updates=2, device="cpu", source=src, root=banked_work)
        res = S.assemble(banked_work, U, src, 2)
        S.write_result(tmp, res)
        for f in range(S.K_FOLDS):
            S.run_unit(U, lab, ctxs, "V", 0, f, n_updates=3, device="cpu", source=src, root=a2_work)
    _FIX.update(dict(tmp=tmp, U=U, ctxs=ctxs, lab=lab, src=src, banked_work=banked_work, a2_work=a2_work, banked_result=os.path.join(tmp, "PROBE_RESULT.json"), res=res))
    return _FIX


def run_synth(identity_only=False, **over):
    F = fixture()
    kw = dict(out=F["tmp"], work_a2=F["a2_work"], work_banked=F["banked_work"], banked_result=F["banked_result"], n_a2=3, n_banked=2, identity_only=identity_only, U=F["U"], source=F["src"])
    kw.update(over)
    with patched(S, "DEPTH", 1), patched(R, "FLOOR_TEXT", round(F["res"]["arms"]["V"]["seed_floor"], 4)):
        return R.run(**kw)


def expect_instrument_failed(fn, *a, **k):
    try:
        fn(*a, **k)
    except R.InstrumentFailed:
        return True
    raise AssertionError("the deliberate regression was NOT caught (the instrument check is dead)")


# ---- the committed outcome table --------------------------------------------------------------------------------------------------------------------------------------
def _d(sep, point, beyond=None):
    return {"separated_beyond_floor": sep, "point": point, "abs_point_exceeds_all_floors": sep if beyond is None else beyond, "ci95": [point - 0.01, point + 0.01]}


def oracle(h, i, sep_pos):
    """The A2 table written out independently, row by row, exactly as the contract's four lines."""
    hits = []
    if h >= 0.80:
        hits.append("U1")
    if h < 0.80 and i >= 0.80 and sep_pos:
        hits.append("U2")
    if h < 0.80 and i < 0.80:
        hits.append("U3")
    if h < 0.80 and i >= 0.80 and not sep_pos:
        hits.append("U4")
    return hits


@test
def test_decide_matches_the_independent_oracle_on_a_sweep_and_is_exhaustive():
    seen = set()
    hs = [0.0, 0.5, 0.7999999999, 0.8, 0.8000000001, 0.95, 1.0]
    is_ = [0.0, 0.7, 0.7999999999, 0.8, 0.8000000001, 0.99]
    ds = [(False, -0.1), (False, 0.0), (False, 0.05), (True, -0.05), (True, 0.05), (True, 0.2)]
    for h in hs:
        for i in is_:
            for sep, pt in ds:
                got = R.decide_a2(h, i, _d(sep, pt))["row"]
                exp = oracle(h, i, bool(sep and pt > 0))
                assert len(exp) == 1, (h, i, sep, pt, exp)                                   # the committed table is exhaustive AND exclusive
                assert [got] == exp, (h, i, sep, pt, got, exp)
                seen.add(got)
    assert seen == {"U1", "U2", "U3", "U4"}


@test
def test_decide_boundaries_and_order():
    assert R.decide_a2(0.80, 0.50, _d(False, 0.0))["row"] == "U1"                           # H >= 0.80 on the unrounded value
    assert R.decide_a2(0.7999999, 0.50, _d(False, 0.0))["row"] == "U3"
    assert R.decide_a2(0.79, 0.80, _d(True, 0.1))["row"] == "U2"                              # I >= 0.80 inclusive
    assert R.decide_a2(0.79, 0.7999999, _d(True, 0.1))["row"] == "U3"
    assert R.decide_a2(0.99, 0.99, _d(True, 0.1))["row"] == "U1"                              # U1 is applied first, before U2 (H, I and D all qualify for U2)
    assert R.decide_a2(0.79, 0.90, _d(True, -0.05))["row"] == "U4"                            # separated but ADVERSE (U10k worse than V 2k): not "> 0"
    assert R.decide_a2(0.79, 0.90, _d(False, 0.05))["row"] == "U4"                            # positive but not separated
    # the diagnostic variant: floor-only differs from the probe rule exactly when the CI clause decides
    dd = _d(False, 0.05, beyond=True)
    assert R.decide_a2(0.79, 0.90, dd)["row"] == "U4" and R.decide_a2_floor_only(0.79, 0.90, dd) == "U2"
    for r in R.ROWS_A2:
        assert R.ROWS_A2[r][0] and R.ROWS_A2[r][1]


@test
def test_loss_curve_stats_slope_and_plateau():
    n = 10000
    lin = np.stack([1.0 - 1e-4 * np.arange(n), np.zeros(n)], 1)
    st = R.loss_curve_stats(lin)
    assert abs(st["bce_last10pct_slope_per_1000_updates"] - (-0.1)) < 1e-9 and st["last10pct_updates"] == 1000
    assert abs(st["bce_last10pct_change_vs_prev10pct"] - (-0.1)) < 1e-9 and len(st["bce_decile_means"]) == 10
    flat = np.stack([np.full(n, 0.3), np.zeros(n)], 1)
    assert abs(R.loss_curve_stats(flat)["bce_last10pct_slope_per_1000_updates"]) < 1e-12
    assert "note" in R.loss_curve_stats(np.zeros((5, 2)))


# ---- the instrument on the synthetic grid -----------------------------------------------------------------------------------------------------------------------
@test
def test_identity_and_zero_control_green_on_the_synthetic_grid():
    r = run_synth(identity_only=True)
    ident, zero = r["instrument"]["identity"], r["instrument"]["zero_control"]
    assert ident["ok"] and all(v["exact"] for v in ident["checks"].values()) and len(ident["checks"]) == 9
    assert zero["ok"] and zero["contrast"]["point"] == 0.0 and zero["contrast"]["ci95"] == [0.0, 0.0] and not zero["contrast"]["separated_beyond_floor"]
    F = fixture()
    assert r["v_2k"]["heldout_iou_0_40"] == F["res"]["arms"]["V"]["seeds"]["0"]["heldout_iou_0_40"]               # exactly the assemble() number, not allclose


@test
def test_identity_red_arms():
    """Deliberate regressions: every one must raise InstrumentFailed."""
    real_p1 = S.pooled_p1

    def flipped(keys, logit, sdf, source, n_boot=S.N_BOOT):                                    # a readout that has changed (one token's logits flipped)
        lg = np.array(logit, copy=True)
        lg[0] = -lg[0]
        return real_p1(keys, lg, sdf, source, n_boot=n_boot)

    with patched(S, "pooled_p1", flipped):
        expect_instrument_failed(run_synth, identity_only=True)
    real_bw = S.ASR.boot_weights
    with patched(S.ASR, "boot_weights", lambda n, m=S.N_BOOT, seed=S.SEED_BOOT: real_bw(n, m, seed + 1)):          # a different bootstrap seed
        expect_instrument_failed(run_synth, identity_only=True)
    expect_instrument_failed(run_synth, identity_only=True, n_banked=3)                          # the banked result is not this code's config at that update count
    real_contrast = S.contrast

    def leaks(a, b, pa, pb, floors):                                                           # a contrast that is not 0 against itself
        return real_contrast(a, b, pa + 1e-9, pb, floors)

    with patched(S, "contrast", leaks):
        expect_instrument_failed(run_synth, identity_only=True)
    F = fixture()
    with patched(R, "FLOOR_TEXT", 0.0036):                                                     # the file's seed floor is not the 0.0036 A2 quotes
        try:
            with patched(S, "DEPTH", 1):
                R.run(F["tmp"], F["a2_work"], F["banked_work"], F["banked_result"], 3, 2, identity_only=True, U=F["U"], source=F["src"])
        except R.InstrumentFailed:
            pass
        else:
            raise AssertionError("a floor that disagrees with A2's text was not caught")
    # a tampered banked result (one banked number changed in the last digits) breaks the identity test
    tam = json.load(open(F["banked_result"]))
    tam["arms"]["V"]["seeds"]["0"]["heldout_iou_0_40"] += 1e-12
    p = os.path.join(F["tmp"], "tampered.json")
    json.dump(tam, open(p, "w"))
    expect_instrument_failed(run_synth, identity_only=True, banked_result=p)


@test
def test_units_of_another_config_are_refused():
    F = fixture()
    with patched(S, "DEPTH", 1):
        expect_guard(R.arm_read, F["a2_work"], "V", 0, 4, F["U"], F["src"], S.ASR.boot_weights(8, 10, 1), sorted(set(F["U"].logs.tolist())))      # these units are n_updates=3
        expect_guard(R.arm_read, F["banked_work"], "V", 0, 3, F["U"], F["src"], S.ASR.boot_weights(8, 10, 1), sorted(set(F["U"].logs.tolist())))
        expect_guard(R.arm_read, F["a2_work"], "V", 1, 3, F["U"], F["src"], S.ASR.boot_weights(8, 10, 1), sorted(set(F["U"].logs.tolist())))      # seed 1 has no a2 units


@test
def test_full_read_on_the_synthetic_grid():
    r = run_synth()
    assert r["mode"] == "full" and r["decision"]["row"] in R.ROWS_A2 and r["u10k"]["n_updates"] == 3 and r["v_2k"]["n_updates"] == 2
    d = r["delta_u10k_minus_v2k"]
    assert d["point"] == r["u10k"]["heldout_iou_0_40"] - r["v_2k"]["heldout_iou_0_40"] and d["floors_used"] == {"V_seed_floor": r["instrument"]["v_seed_floor_used"]}
    assert r["decision"]["H"] == r["u10k"]["heldout_iou_0_40"] and r["decision"]["I"] == r["u10k"]["insample_iou_0_40_mean"]
    assert len(r["u10k"]["per_fold_heldout_iou_0_40"]) == 5 and len(r["u10k"]["insample_iou_0_40_per_fold"]) == 5
    assert all(t["curve"] is not None for t in r["u10k"]["train"])                               # the A2 units carry the loss curve (the real banked 2k units do not, and read as None)
    assert r["row_depends_on_the_ci_clause"] in (True, False)
    S.dump_json_bytes(r)                                                                         # JSON-serialisable
    assert "A2 RESULT" in R.summary_md(r)
    # the zero-control twin: U10k read against ITSELF is also exactly 0 (same machinery, other arm)
    same = run_synth(work_a2=fixture()["banked_work"], n_a2=2)
    assert same["delta_u10k_minus_v2k"]["point"] == 0.0 and same["delta_u10k_minus_v2k"]["ci95"] == [0.0, 0.0] and not same["delta_u10k_minus_v2k"]["separated_beyond_floor"]


@test
def test_main_writes_result_and_markers_or_fails_loud():
    F = fixture()
    out = tempfile.mkdtemp(prefix="stp_a2main_")
    try:
        args = ["--out", out, "--work-a2", F["a2_work"], "--work-banked", F["banked_work"], "--banked-result", F["banked_result"], "--updates-a2", "3", "--updates-banked", "2"]
        with patched(S, "DEPTH", 1), patched(R, "FLOOR_TEXT", round(F["res"]["arms"]["V"]["seed_floor"], 4)), patched(S, "load_universe", lambda: F["U"]), \
                patched(S, "load_true_labels", lambda U: (F["lab"], F["src"])):
            assert R.main(args) == 0
            r = json.load(open(os.path.join(out, "A2_RESULT.json")))
            assert r["decision"]["row"] in R.ROWS_A2 and os.path.exists(os.path.join(out, "A2_RESULT.md")) and not os.path.exists(os.path.join(out, "a2_read.error"))
            assert R.main(args + ["--identity-only"]) == 0 and os.path.exists(os.path.join(out, "A2_IDENTITY.json"))
            os.remove(os.path.join(out, "A2_RESULT.json"))
            real_p1 = S.pooled_p1
            with patched(S, "pooled_p1", lambda keys, logit, sdf, source, n_boot=S.N_BOOT: real_p1(keys, np.where(np.arange(len(keys))[:, None, None] == 0, -logit, logit), sdf, source, n_boot=n_boot)):
                assert R.main(args) == 3                                                         # InstrumentFailed -> exit 3, no result
            assert not os.path.exists(os.path.join(out, "A2_RESULT.json")) and "IDENTITY TEST FAILED" in open(os.path.join(out, "a2_read.error")).read()
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ---- the REAL banked V-s0 (2,000 updates): the identity test and the zero control the brief asks for ---------------------------------------------------------------------
@test
def test_real_identity_and_zero_control_on_the_banked_v_s0_units():
    r = R.run(HERE, R.A2_WORK, R.BANKED_WORK, R.BANKED_RESULT, R.A2_UPDATES, R.BANKED_UPDATES, identity_only=True)
    ident, zero, v = r["instrument"]["identity"], r["instrument"]["zero_control"], r["v_2k"]
    assert ident["ok"], {k: x for k, x in ident["checks"].items() if not x["exact"]}
    assert v["heldout_iou_0_40"] == 0.6348906684369943 == json.load(open(R.BANKED_RESULT, encoding="utf-8"))["arms"]["V"]["seeds"]["0"]["heldout_iou_0_40"]
    assert zero["ok"] and zero["contrast"]["point"] == 0.0 and zero["contrast"]["ci95"] == [0.0, 0.0]
    assert abs(r["instrument"]["v_seed_floor_used"] - 0.003552465097100832) < 1e-18 and round(r["instrument"]["v_seed_floor_used"], 4) == 0.0036
    with open(os.path.join(HERE, "A2_IDENTITY.json"), "wb") as f:
        f.write(S.dump_json_bytes(r))


def main() -> int:
    ok, bad = 0, []
    try:
        for fn in TESTS:
            try:
                fn()
                print(f"PASS {fn.__name__}", flush=True)
                ok += 1
            except BaseException as e:                                                     # noqa: BLE001
                print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                bad.append(fn.__name__)
    finally:
        if _FIX.get("tmp"):
            shutil.rmtree(_FIX["tmp"], ignore_errors=True)
    if bad:
        print(f"ZZTEST_FAIL {len(bad)} of {len(TESTS)}: {bad}")
        return 1
    print(f"ZZTEST_PASS {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
