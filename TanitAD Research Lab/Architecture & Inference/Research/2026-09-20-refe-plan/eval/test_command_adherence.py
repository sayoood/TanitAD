#!/usr/bin/env python3
"""P6-ADHERENCE (2026-10-22) tests for eval/command_adherence.py: the TACTICAL command-adherence row, as defined by PREREG_P6 Amendment (b).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/test_command_adherence.py        # every check has a deliberate-regression (RED) arm; prints ZZADH PASS n/n

Real-data checks read the BANKED navtest seams of P2 (C2) and u400 (C1), the W3 export and P2's banked 64-proposal table: known-value controls.
NO P6 checkpoint output on navtest exists or is read. Each RED arm MUTATES the thing under test and requires the check to fail.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import command_adherence as CA  # noqa: E402

RESULTS: list = []
_C: dict = {}                                                                # cached real-data fixtures

# MEASURED 2026-10-22 by this instrument on the banked seams (our own regression pins, NOT published values): confusion rows = command LEFT/STRAIGHT/RIGHT, cols = executed-plan class
PIN_CM = {"p2": [[2278, 210, 13], [638, 6894, 538], [7, 248, 1320]],
          "u400": [[2271, 211, 19], [592, 6945, 533], [7, 239, 1329]]}


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
                RESULTS.append((name, True, ""))
            except Exception as e:                                           # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {str(e)[:300]}"))
        run.__name__ = fn.__name__
        run.check_name = name
        return run
    return deco


def must_fail(fn, *a, **k):
    """the deliberate-regression arm: `fn` must raise AssertionError/SystemExit/ValueError (the check went RED)"""
    try:
        fn(*a, **k)
    except (AssertionError, SystemExit, ValueError):
        return True
    raise AssertionError("the deliberate regression was NOT caught: the check cannot fail")


# ------------------------------------------------------------------ fixtures
def fx():
    if "exp" not in _C:
        import stage3_full as S3
        _C["S3"] = S3
        _C["exp"] = CA.load_export()
        _C["p2"] = S3.pick_baseline("p2")
        _C["u400"] = S3.pick_baseline("u400")
        assert _C["p2"]["pin_ok"] and _C["u400"]["pin_ok"], "the banked P2 / u400 files are absent or fail their md5 pins"
        _C["seam_p2"] = CA.load_seam(_C["p2"]["seam"])
        _C["seam_u4"] = CA.load_seam(_C["u400"]["seam"])
        toks = sorted(_C["seam_p2"]["tokens"])
        _C["toks"] = toks
        _C["cmd"], _C["logs"] = CA.commands_and_logs(toks, _C["exp"])
        _C["plan_p2"], _ = CA.plan_classes(CA.align(_C["seam_p2"], toks))
        _C["plan_u4"], _ = CA.plan_classes(CA.align(_C["seam_u4"], toks))
    return _C


def kappa_ref(cm):
    """independent exact reference (Fractions), no numpy: Cohen's kappa of a 3x3 count matrix"""
    n = sum(sum(r) for r in cm)
    po = Fraction(sum(cm[i][i] for i in range(3)), n)
    pe = sum(Fraction(sum(cm[i]), n) * Fraction(sum(cm[j][i] for j in range(3)), n) for i in range(3))
    return float((po - pe) / (1 - pe))


def mini_exp(tokens, cmd, logs, hum=None):
    """a minimal export-shaped dict (token -> record) for the fields the instrument reads"""
    oh = {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0, 0.0], 2: [0.0, 0.0, 1.0, 0.0], 3: [0.0, 0.0, 0.0, 1.0]}
    return {t: {"ego_statuses": [{"driving_command": oh[int(c)]}], "log_name": str(l), "human_future_poses": (hum[i] if hum is not None else np.zeros((8, 3)).tolist())}
            for i, (t, c, l) in enumerate(zip(tokens, cmd, logs))}


# ================================================================== A. the commands and the pinned class counts
@check("A1 commands: the 12,146 navtest tokens read LEFT 2,501 / STRAIGHT 8,070 / RIGHT 1,575 / UNKNOWN 0 (turn-active 4,076, 136 logs): the ruling's numbers")
def t_counts():
    c = fx()
    counts = {CA.CMD_NAMES[i]: int((c["cmd"] == i).sum()) for i in range(4)}
    assert counts == CA.EXPECT_COUNTS == {"LEFT": 2501, "STRAIGHT": 8070, "RIGHT": 1575, "UNKNOWN": 0}, counts
    assert int(((c["cmd"] == 0) | (c["cmd"] == 2)).sum()) == CA.EXPECT_TURN == 4076
    assert len(set(c["logs"].tolist())) == CA.EXPECT_LOGS == 136 and len(c["toks"]) == 12146


@check("A1 RED: a LEFT<->RIGHT index swap in the one-hot decode breaks the pinned counts, and the read's G_COUNTS gate says INVALID")
def t_counts_red():
    c = fx()
    sw = np.where(c["cmd"] == 0, 2, np.where(c["cmd"] == 2, 0, c["cmd"]))
    counts = {CA.CMD_NAMES[i]: int((sw == i).sum()) for i in range(4)}
    must_fail(lambda: _assert_eq(counts, CA.EXPECT_COUNTS))                        # 1575/8070/2501 != 2501/8070/1575
    # and through the real gate: an export whose LEFT/RIGHT one-hots are swapped
    ex = mini_exp(c["toks"], sw, c["logs"])
    r = CA.run_read({"name": "p2", "seam": c["p2"]["seam"]}, [{"role": "C1", "name": "u400", "seam": c["u400"]["seam"]}], exp=ex, verify=False, include_human=False, n_boot=50)
    assert "INVALID" in r and r["gates"]["G_COUNTS"]["ok"] is False, r.get("INVALID")


@check("A1 the gate itself: a mini export carrying the TRUE commands passes G_COUNTS (so the RED above is the swap, not the harness)")
def t_counts_gate_green():
    c = fx()
    ex = mini_exp(c["toks"], c["cmd"], c["logs"])
    r = CA.run_read({"name": "p2", "seam": c["p2"]["seam"]}, [{"role": "C1", "name": "u400", "seam": c["u400"]["seam"]}], exp=ex, verify=False, include_human=False, n_boot=50)
    assert "INVALID" not in r and r["gates"]["G_COUNTS"]["ok"], r.get("INVALID")


@check("A2 commands: a non-one-hot driving_command is REFUSED (two ones, all zeros, NaN, wrong length), never defaulted")
def t_one_hot():
    ok = {"t": {"ego_statuses": [{"driving_command": [0.0, 1.0, 0.0, 0.0]}], "log_name": "L"}}
    assert CA.commands_and_logs(["t"], ok)[0][0] == 1
    for bad in ([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [float("nan"), 1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]):
        e = {"t": {"ego_statuses": [{"driving_command": bad}], "log_name": "L"}}
        must_fail(CA.commands_and_logs, ["t"], e)
    must_fail(CA.commands_and_logs, ["t"], {"t": {"ego_statuses": [], "log_name": "L"}})


# ================================================================== B. the labeller
@check("B1 the labeller port reproduces the BANKED families block's lateral confusion + heading-held/never-moved counts for P2 AND u400, exactly (no new thresholds)")
def t_labeller_banked():
    c = fx()
    for arm in ("p2", "u400"):
        r = CA.families_lateral_control(arm, c[arm]["seam"], c["exp"])
        assert r["ok"], r


@check("B1 RED (mutation of the class mapping): a rotated LAT_TO_CMD, and a LEFT<->RIGHT swapped one, each turn the banked-confusion control RED")
def t_labeller_mapping_mutation():
    c = fx()
    orig = CA.LAT_TO_CMD.copy()
    try:
        for mut in (np.array([CA.LEFT, CA.RIGHT, CA.STRAIGHT]), np.array([CA.STRAIGHT, CA.RIGHT, CA.LEFT])):
            CA.LAT_TO_CMD = mut
            assert not CA.families_lateral_control("p2", c["p2"]["seam"], c["exp"])["ok"], f"mapping {mut.tolist()} was NOT caught"
    finally:
        CA.LAT_TO_CMD = orig
    assert CA.families_lateral_control("p2", c["p2"]["seam"], c["exp"])["ok"]       # restored: green again


@check("B2 reachability: the CANONICAL threshold is what the instrument reads (patching refc_tactical.YAW_TURN_RAD to 5 rad removes every turn; restored after)")
def t_labeller_reachable():
    c = fx()
    RT, _FF = CA.labeller()
    base, _ = CA.plan_classes(CA.align(c["seam_p2"], c["toks"]))
    assert ((base == CA.LEFT) | (base == CA.RIGHT)).sum() > 1000
    orig = RT.YAW_TURN_RAD
    try:
        RT.YAW_TURN_RAD = 5.0
        mut, _ = CA.plan_classes(CA.align(c["seam_p2"], c["toks"]))
    finally:
        RT.YAW_TURN_RAD = orig
    assert not ((mut == CA.LEFT) | (mut == CA.RIGHT)).any(), "a 5 rad turn gate still produced turns: the instrument does not read the canonical labeller"
    again, _ = CA.plan_classes(CA.align(c["seam_p2"], c["toks"]))
    assert np.array_equal(again, base)


def _threshold_assignments(src: str):
    import re
    return re.findall(r"^\s*(?:YAW_TURN_RAD|DV_ACCEL_MS|DV_BRAKE_MS|STOP_V_MS|MOVING_V_MS|MIN_ARC_M|CURV_TURN_MAN_PER_M|LABEL_HORIZON)\s*=", src, flags=re.M)


@check("B3 no new thresholds: command_adherence.py assigns none of the labeller's threshold constants, and names the canonical function")
def t_no_thresholds():
    src = (HERE / "command_adherence.py").read_text(encoding="utf-8")
    assert _threshold_assignments(src) == [], _threshold_assignments(src)
    assert CA.labeller_provenance()["factor_from_kinematics"] == "tanitad.refs.refc_tactical.factor_from_kinematics"
    assert CA.labeller_provenance()["maneuver_kinematics"] == "taniteval.four_families.maneuver_kinematics"
    must_fail(lambda: _assert_none(_threshold_assignments(src + "\nYAW_TURN_RAD = 0.2\n")))      # RED: a smuggled threshold is seen


def _assert_none(x):
    assert x == [], x


@check("B4 live: the labeller's lateral confusion equals what four_families.tactical_from_trajectory itself returns on the same plan/human arrays")
def t_tactical_from_trajectory_live():
    import torch
    c = fx()
    _RT, FF = CA.labeller()
    pred = torch.as_tensor(CA.align(c["seam_p2"], c["toks"])[..., :2]).float()
    gt = torch.as_tensor(np.asarray([c["exp"][t]["human_future_poses"] for t in c["toks"]], dtype=np.float64)[..., :2]).float()
    live = FF.tactical_from_trajectory(pred, gt, CA.DT_S, eid=None, tier="T1")["lateral_decision"]["confusion_gt_rows_pred_cols"]
    pc, _ = CA.plan_classes(CA.align(c["seam_p2"], c["toks"]))
    gc, _ = CA.plan_classes(np.asarray([c["exp"][t]["human_future_poses"] for t in c["toks"]], dtype=np.float64))
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (gc, pc), 1)
    spec = (1, 0, 2)                                                              # labeller order (lane_keep, turn_left, turn_right) -> command order, written out independently
    mapped = [[int(live[i][j]) for j in range(3)] for i in range(3)]
    live_cmd = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            live_cmd[spec[i]][spec[j]] = mapped[i][j]
    assert cm.tolist() == live_cmd, (cm.tolist(), live_cmd)
    bad = [row[::-1] for row in live_cmd]                                         # RED: a column-reversed reading must differ
    must_fail(lambda: _assert_eq(cm.tolist(), bad))


def _assert_eq(a, b):
    assert a == b, (a, b)


@check("B5 A7: the executed and the raw plan class are IDENTICAL by construction (heading column never read): zeroing it changes 0 classes, perturbing xy changes many")
def t_a7_invariance():
    c = fx()
    a = CA.align(c["seam_p2"], c["toks"])
    base, _ = CA.plan_classes(a)
    z = a.copy()
    z[..., 2:] = 0.0
    assert (CA.plan_classes(z)[0] != base).sum() == 0
    import torch
    rep = a.copy()
    rep[..., -1, 2] = rep[..., -2, 2]                                              # planner.repair_last_heading, applied by hand
    assert (CA.plan_classes(rep)[0] != base).sum() == 0
    p = a.copy()
    p[..., 1] = -p[..., 1]                                                         # RED: mirroring y MUST change classes (the detector is not blind)
    assert (CA.plan_classes(p)[0] != base).sum() > 1000


@check("B6 plan format: anything but [N, 8, >=2] finite is a STOP (PlanFormatError), never a silent re-parametrisation")
def t_plan_format():
    ok = np.zeros((3, 8, 3))
    CA.check_plan_format(ok)
    for bad in (np.zeros((3, 4, 3)), np.zeros((3, 20, 3)), np.zeros((3, 8, 1)), np.zeros((8, 3)), np.full((3, 8, 3), np.nan)):
        try:
            CA.check_plan_format(bad)
        except CA.PlanFormatError:
            continue
        raise AssertionError(f"shape {bad.shape} was accepted")


# ================================================================== C. the statistics
CM_TOY = [[40, 5, 5], [10, 100, 10], [3, 2, 45]]


def _toy_vectors(cm):
    cmd, plan = [], []
    for i in range(3):
        for j in range(3):
            cmd += [i] * cm[i][j]
            plan += [j] * cm[i][j]
    return np.array(cmd), np.array(plan)


@check("C1 accuracy / kappa / confusion on a hand-checkable toy confusion equal an independent exact (Fraction) reference, on all tokens AND the turn-active subset")
def t_toy_exact():
    cmd, plan = _toy_vectors(CM_TOY)
    logs = np.array([f"L{i % 7}" for i in range(len(cmd))])
    assert CA.confusion(cmd, plan).tolist() == CM_TOY
    st = CA.stats_from_counts(CA.confusion(cmd, plan))
    assert abs(float(st["accuracy"]) - 185 / 220) < 1e-15 and abs(float(st["kappa"]) - kappa_ref(CM_TOY)) < 1e-12
    cm_t = [CM_TOY[0], [0, 0, 0], CM_TOY[2]]
    _, _, _, Ct, cl = CA._cells(cmd, plan, logs, 200, CA.SEED, "toy")
    t = cl["turn_active"]
    assert t["n"] == 100 and t["confusion_command_rows_plan_cols"] == [[float(x) for x in r] for r in cm_t]
    assert abs(t["accuracy"]["point"] - 85 / 100) < 1e-15 and abs(t["kappa"]["point"] - kappa_ref(cm_t)) < 1e-12
    a = cl["all_tokens"]
    assert a["n"] == 220 and abs(a["kappa"]["point"] - kappa_ref(CM_TOY)) < 1e-12
    # RED: a plausible bug (STRAIGHT left out of the expected agreement) must disagree with the reference
    must_fail(lambda: _assert_close(_bad_kappa(CM_TOY), kappa_ref(CM_TOY)))


def _bad_kappa(cm):
    n = sum(sum(r) for r in cm)
    po = sum(cm[i][i] for i in range(3)) / n
    pe = sum(sum(cm[i]) / n * sum(cm[j][i] for j in range(3)) / n for i in (0, 2))      # the bug: STRAIGHT left out of pe
    return (po - pe) / (1 - pe)


def _assert_close(a, b, tol=1e-12):
    assert abs(a - b) < tol, (a, b)


@check("C2 kappa: equals the programme's own four_families._kappa_k and _class_report on random label vectors (1e-12); an undefined kappa is None/NaN, NEVER 1.0")
def t_kappa_programme():
    _RT, FF = CA.labeller()
    rng = np.random.default_rng(3)
    for _ in range(5):
        a, b = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
        r = CA.stats_from_counts(CA.confusion(a, b))
        assert abs(float(r["kappa"]) - FF._kappa_k(a, b, 3)) < 1e-12
    same = np.ones(50, dtype=np.int64)
    r = CA.stats_from_counts(CA.confusion(same, same))
    assert FF._kappa_k(same, same, 3) is None and not np.isfinite(r["kappa"]), "constant-and-equal raters must give an undefined kappa, never 1.0"
    perfect = rng.integers(0, 3, 200)
    assert float(CA.stats_from_counts(CA.confusion(perfect, perfect))["kappa"]) == 1.0
    must_fail(lambda: _assert_close(float(CA.stats_from_counts(CA.confusion(rng.integers(0, 3, 300), rng.integers(0, 3, 300)))["kappa"]), 1.0))


@check("C3 bootstrap: accuracy level and paired-difference intervals are BIT-EQUAL to proxy_eval.boot on the same tokens/logs (synthetic); a wrong seed or token-level resampling is NOT")
def t_boot_equals_PE():
    import proxy_eval as PE
    rng = np.random.default_rng(11)
    n, G = 900, 23
    logs = np.array([f"log{int(x):02d}" for x in rng.integers(0, G, n)])
    toks = [f"t{i:04d}" for i in range(n)]
    cmd = rng.integers(0, 3, n)
    pa = np.where(rng.random(n) < 0.8, cmd, rng.integers(0, 3, n))
    pb = np.where(rng.random(n) < 0.7, cmd, rng.integers(0, 3, n))
    tl = dict(zip(toks, logs.tolist()))
    _, Ca = CA.grouped(cmd, pa, logs)
    _, Cb = CA.grouped(cmd, pb, logs)
    lv = CA.cell(Ca, 500, 5)
    ia = {t: float(pa[i] == cmd[i]) for i, t in enumerate(toks)}
    ib = {t: float(pb[i] == cmd[i]) for i, t in enumerate(toks)}
    mu, lo, hi = PE.boot(ia, tl, n=500, seed=5)
    assert (mu, lo, hi) == (lv["accuracy"]["point"], *lv["accuracy"]["ci95"]), ((mu, lo, hi), lv["accuracy"])
    pr = CA.paired(Ca, Cb, 500, 5)["accuracy_delta"]
    mu, lo, hi = PE.boot({t: ia[t] - ib[t] for t in toks}, tl, n=500, seed=5)
    assert (mu, lo, hi) == (pr["point"], *pr["ci95"]), ((mu, lo, hi), pr)
    mu2, lo2, hi2 = PE.boot(ia, tl, n=500, seed=6)                                  # RED: another seed must not reproduce it
    assert (lo2, hi2) != tuple(lv["accuracy"]["ci95"])
    must_fail(lambda: _assert_eq(PE.boot(ia, {t: t for t in toks}, n=500, seed=5)[1:], tuple(lv["accuracy"]["ci95"])))   # RED: token-level clusters is another estimator


@check("C4 bootstrap on REAL data: the control read's G_PE_BOOT_EQUAL gate is True on 6 intervals (P2 and u400 levels x2 subsets, P2-u400 paired x2), kappa undefined in 0 resamples")
def t_boot_real():
    r = _control()
    g = r["gates"]["G_PE_BOOT_EQUAL"]
    assert g["ok"] is True and g["n_checked"] == 6 and not g["mismatches"], g
    for a in ("p2", "u400"):
        for sub in ("all_tokens", "turn_active"):
            assert r["arms"][a]["executed_plan_A7"][sub]["kappa"]["n_undefined_resamples"] == 0
    # RED: a verifier fed a tampered level must report a mismatch
    c = fx()
    tampered = copy.deepcopy({a: r["arms"][a]["executed_plan_A7"] for a in ("p2", "u400")})
    tampered["p2"]["turn_active"]["accuracy"]["ci95"][0] += 1e-9
    v = CA.verify_boot_equal(c["cmd"], {"p2": c["plan_p2"], "u400": c["plan_u4"]}, c["logs"], c["toks"], tampered, {}, CA.N_BOOT, CA.SEED)
    assert v["ok"] is False and v["mismatches"], v                                 # the tampered interval was CAUGHT


def _control():
    if "control" not in _C:
        _C["control"] = CA.control(None, n_boot=CA.N_BOOT, verify=True)
    return _C["control"]


# ================================================================== D. controls on the banked arms
@check("D1 the control read is VALID (no INVALID) and the banked confusions are pinned: P2 and u400 3x3 (command rows x plan cols) equal this instrument's measured pins")
def t_control_valid_and_pins():
    r = _control()
    assert "INVALID" not in r, r.get("INVALID")
    for a in ("p2", "u400"):
        assert r["arms"][a]["executed_plan_A7"]["all_tokens"]["confusion_command_rows_plan_cols"] == [[float(x) for x in row] for row in PIN_CM[a]], a
        ta = r["arms"][a]["executed_plan_A7"]["turn_active"]["confusion_command_rows_plan_cols"]
        assert ta == [[float(x) for x in PIN_CM[a][0]], [0.0, 0.0, 0.0], [float(x) for x in PIN_CM[a][2]]], a
        assert r["arms"][a]["executed_plan_A7"]["turn_active"]["n"] == 4076
    for v in r["families_lateral_control"].values():
        assert v["ok"]


@check("D1 RED (mutation): with the class mapping broken the control read's pins fail (a rotated LAT_TO_CMD changes P2's confusion)")
def t_control_mutation_red():
    c = fx()
    orig = CA.LAT_TO_CMD.copy()
    try:
        CA.LAT_TO_CMD = np.array([CA.LEFT, CA.RIGHT, CA.STRAIGHT])
        pc, _ = CA.plan_classes(CA.align(c["seam_p2"], c["toks"]))
        cm = CA.confusion(c["cmd"], pc)
    finally:
        CA.LAT_TO_CMD = orig
    must_fail(lambda: _assert_eq(cm.tolist(), PIN_CM["p2"]))                       # the broken mapping must NOT reproduce the pinned confusion


@check("D2 RED arm: LEFT<->RIGHT swapped commands drop P2's turn-active accuracy from > 0.8 to < 0.05 and make kappa strongly negative")
def t_swap_red_arm():
    c = fx()
    sw = np.where(c["cmd"] == 0, 2, np.where(c["cmd"] == 2, 0, c["cmd"]))
    ta = (c["cmd"] == 0) | (c["cmd"] == 2)

    def turn_acc(cmd):
        _, Ct = CA.grouped(cmd, c["plan_p2"], c["logs"], ta)
        return CA.stats_from_counts(Ct.sum(0))

    real, swapped = turn_acc(c["cmd"]), turn_acc(sw)
    assert float(real["accuracy"]) > 0.8 and float(real["kappa"]) > 0.7
    assert float(swapped["accuracy"]) < 0.05 and float(swapped["kappa"]) < -0.5, (float(swapped["accuracy"]), float(swapped["kappa"]))
    must_fail(lambda: _assert_lt(float(real["accuracy"]), 0.05))                       # RED: the same assertion on the REAL commands must fail
    r = _control()
    assert r["controls"]["swap_LR_commands"]["drops"] is True and r["controls"]["swap_LR_commands"]["turn_active_accuracy"]["point"] < 0.05


def _assert_lt(a, b):
    assert a < b, (a, b)


@check("D3 zero arm: a constant-STRAIGHT plan gives kappa = 0.0 EXACTLY (all tokens and turn-active), accuracy 8,070/12,146 and 0 on the turn-active subset; a perfect plan gives 1.0 (RED)")
def t_const_straight():
    c = fx()
    const = np.full(len(c["cmd"]), CA.STRAIGHT)
    cm = CA.confusion(c["cmd"], const)
    st = CA.stats_from_counts(cm)
    assert float(st["kappa"]) == 0.0 and float(st["accuracy"]) == 8070 / 12146
    ta = (c["cmd"] == 0) | (c["cmd"] == 2)
    _, Ct = CA.grouped(c["cmd"], const, c["logs"], ta)
    stt = CA.stats_from_counts(Ct.sum(0))
    assert float(stt["kappa"]) == 0.0 and float(stt["accuracy"]) == 0.0
    for k in (CA.LEFT, CA.RIGHT):                                                      # any constant plan is kappa 0
        assert float(CA.stats_from_counts(CA.confusion(c["cmd"], np.full(len(c["cmd"]), k)))["kappa"]) == 0.0
    perfect = CA.stats_from_counts(CA.confusion(c["cmd"], c["cmd"]))
    must_fail(lambda: _assert_eq(float(perfect["kappa"]), 0.0))                         # RED: a plan that IS the command is not kappa 0
    r = _control()
    assert r["controls"]["constant_straight_plan"]["kappa_is_zero"] and r["gates"]["G_KAPPA_ZERO_ID"]["ok"]


@check("D4 floor: P2's banked 64-proposal table gives a floor equal to the brute-force mean over the 64 columns (1e-12), and proposals[pick] equals the seam bit for bit after the token join")
def t_floor_identity():
    c = fx()
    d = CA.load_dump(CA.P2_TABLE)
    idx = {t: i for i, t in enumerate(d["tokens"])}
    order = np.array([idx[t] for t in c["toks"]])
    P, pk = d["proposals"][order], d["pick"][order]
    assert np.array_equal(P[np.arange(len(pk)), pk], CA.align(c["seam_p2"], c["toks"]).astype(P.dtype))
    cols = np.stack([CA.plan_classes(P[:, k])[0] for k in range(CA.M)], 1)                   # [N, 64] command-space classes, one column at a time
    ta = (c["cmd"] == 0) | (c["cmd"] == 2)
    brute_all = float((cols == c["cmd"][:, None]).mean())
    brute_turn = float((cols[ta] == c["cmd"][ta][:, None]).mean())
    F = CA.proposal_fractions(P)
    assert np.allclose(F.sum(1), 1.0, atol=1e-12)
    _, Cf = CA.grouped(c["cmd"], F, c["logs"])
    _, Cft = CA.grouped(c["cmd"], F, c["logs"], ta)
    assert abs(float(CA.stats_from_counts(Cf.sum(0))["accuracy"]) - brute_all) < 1e-12
    assert abs(float(CA.stats_from_counts(Cft.sum(0))["accuracy"]) - brute_turn) < 1e-12
    r = _control()["random_proposal_floor"]
    assert abs(r["all_tokens"]["floor"]["accuracy"]["point"] - brute_all) < 1e-12 and abs(r["turn_active"]["floor"]["accuracy"]["point"] - brute_turn) < 1e-12
    must_fail(lambda: _assert_close(brute_all, float((cols[np.random.default_rng(0).permutation(len(cols))] == c["cmd"][:, None]).mean()), 1e-6))   # RED: shuffled tokens break the join


@check("D4 RED: a dump whose `pick` is shuffled (or whose token set differs) is INVALID in the read, never a floor")
def t_floor_red():
    c = fx()
    d = CA.load_dump(CA.P2_TABLE)
    bad = dict(d)
    bad["pick"] = np.random.default_rng(1).permutation(d["pick"])
    r = CA.run_read({"name": "p2", "seam": c["p2"]["seam"]}, [{"role": "C1", "name": "u400", "seam": c["u400"]["seam"]}], dump=bad, exp=c["exp"], verify=False, include_human=False, n_boot=50)
    assert r["gates"]["G_DUMP_PICK"]["ok"] is False and "INVALID" in r
    bad2 = dict(d)
    bad2["tokens"] = d["tokens"][:-1]
    bad2["proposals"], bad2["pick"] = d["proposals"][:-1], d["pick"][:-1]
    r2 = CA.run_read({"name": "p2", "seam": c["p2"]["seam"]}, [{"role": "C1", "name": "u400", "seam": c["u400"]["seam"]}], dump=bad2, exp=c["exp"], verify=False, include_human=False, n_boot=50)
    assert "INVALID" in r2
    good = CA.run_read({"name": "p2", "seam": c["p2"]["seam"]}, [{"role": "C1", "name": "u400", "seam": c["u400"]["seam"]}], dump=d, exp=c["exp"], verify=False, include_human=False, n_boot=50)
    assert "INVALID" not in good, good.get("INVALID")


@check("D5 paired: C2 - C1 equals acc(P2) - acc(u400) exactly; C1 - C2 mirrors it (points negated, interval mirrored to 1e-12); an UNPAIRED interval is wider (RED: paired == unpaired must fail)")
def t_paired():
    c = fx()
    ta = (c["cmd"] == 0) | (c["cmd"] == 2)
    for mask in (None, ta):
        _, Ca = CA.grouped(c["cmd"], c["plan_p2"], c["logs"], mask)
        _, Cb = CA.grouped(c["cmd"], c["plan_u4"], c["logs"], mask)
        d = CA.paired(Ca, Cb)
        e = CA.paired(Cb, Ca)
        pa, pb = CA.stats_from_counts(Ca.sum(0)), CA.stats_from_counts(Cb.sum(0))
        assert abs(d["accuracy_delta"]["point"] - (float(pa["accuracy"]) - float(pb["accuracy"]))) < 1e-15
        assert abs(d["accuracy_delta"]["point"] + e["accuracy_delta"]["point"]) < 1e-15
        assert abs(d["accuracy_delta"]["ci95"][0] + e["accuracy_delta"]["ci95"][1]) < 1e-12 and abs(d["accuracy_delta"]["ci95"][1] + e["accuracy_delta"]["ci95"][0]) < 1e-12
        assert abs(d["kappa_delta"]["point"] - (float(pa["kappa"]) - float(pb["kappa"]))) < 1e-12
    _, Ca = CA.grouped(c["cmd"], c["plan_p2"], c["logs"])
    _, Cb = CA.grouped(c["cmd"], c["plan_u4"], c["logs"])
    pr = CA.paired(Ca, Cb)["accuracy_delta"]["ci95"]
    la, lb = CA.cell(Ca, CA.N_BOOT, 1)["accuracy"]["ci95"], CA.cell(Cb, CA.N_BOOT, 2)["accuracy"]["ci95"]        # two INDEPENDENT resample streams = the unpaired difference
    wa, wb = (la[1] - la[0]), (lb[1] - lb[0])
    unpaired_half_width = float(np.hypot(wa, wb) / 2)
    assert (pr[1] - pr[0]) / 2 < unpaired_half_width / 2, (pr, unpaired_half_width)          # paired is much narrower on identical tokens
    must_fail(lambda: _assert_close(pr[1] - pr[0], 2 * unpaired_half_width, 1e-3))             # RED: paired width == unpaired width must fail
    r = _control()["paired_vs_comparators"]["p2_minus_C1_u400"]
    assert r["all_tokens"]["n"] == 12146 and r["turn_active"]["n"] == 4076


@check("D6 identical tokens: a comparator whose token set differs is INVALID (pairing needs identical tokens); the CLI is INCOMPLETE (exit 2) without a dump, VALID (exit 0) with one, INVALID (exit 1) on a subset without --subset-ok, and a name collision is refused")
def t_cli_and_subset():
    c = fx()
    # a seam with one token missing -> INVALID
    with tempfile.TemporaryDirectory() as td:
        z = np.load(c["u400"]["seam"], allow_pickle=False)
        k = np.arange(len(z["token"]))[:-1]
        short = Path(td) / "short.npz"
        np.savez(short, token=z["token"][k], poses=z["poses"][k], fingerprint=z["fingerprint"][k])
        r = CA.run_read({"name": "p2", "seam": c["p2"]["seam"]}, [{"role": "C1", "name": "short", "seam": short}], exp=c["exp"], verify=False, include_human=False, n_boot=50)
        assert "INVALID" in r and "differs" in r["INVALID"][0]
        # the CLI: arm = the deployed route-fix seam (a banked third arm), C1 = u400, C2 = p2
        dep = c["S3"].pick_baseline("deployed")
        assert dep["pin_ok"]
        out = Path(td) / "cli.json"
        rc = CA.main(["read", "--name", "ctl_deployed", "--seam", str(dep["seam"]), "--base", "u400", "--also", "p2", "--out", str(out), "--n-boot", "300", "--no-verify-boot"])
        j = json.loads(out.read_text(encoding="utf-8"))
        assert "INVALID" not in j, j.get("INVALID")
        assert rc == 2 and j["INCOMPLETE"], (rc, j.get("INCOMPLETE"))                   # no dump = no floor = INCOMPLETE (exit 2), never a quiet pass
        assert set(j["paired_vs_comparators"]) == {"ctl_deployed_minus_C1_u400", "ctl_deployed_minus_C2_p2"}
        assert j["authority"].startswith("NONE") and j["strategic_family"].startswith("PAUSED")
        # the same CLI WITH P2's dump (arm = P2's seam under another name): complete and valid, exit 0
        out2 = Path(td) / "cli2.json"
        rc2 = CA.main(["read", "--name", "ctl_p2", "--seam", str(c["p2"]["seam"]), "--dump", CA.P2_TABLE, "--base", "u400", "--also", "p2", "--out", str(out2), "--n-boot", "300", "--no-verify-boot"])
        j2 = json.loads(out2.read_text(encoding="utf-8"))
        assert rc2 == 0 and "INVALID" not in j2 and "INCOMPLETE" not in j2, (rc2, j2.get("INVALID"), j2.get("INCOMPLETE"))
        # a comparator named like the arm would silently overwrite its seam: refused
        must_fail(CA.run_read, {"name": "p2", "seam": c["p2"]["seam"]}, [{"role": "C1", "name": "p2", "seam": c["u400"]["seam"]}], None, c["exp"])
        # a 200-token subset of all three: G_COUNTS fails (rc 1) unless --subset-ok
        paths = {}
        for nm, src in (("arm", dep["seam"]), ("u400", c["u400"]["seam"]), ("p2", c["p2"]["seam"])):
            zz = np.load(src, allow_pickle=False)
            toks_all = [str(t) for t in zz["token"]]
            keep = [i for i, t in enumerate(toks_all) if t in set(c["toks"][:200])]
            p = Path(td) / f"{nm}.npz"
            np.savez(p, token=zz["token"][keep], poses=zz["poses"][keep], fingerprint=zz["fingerprint"][keep])
            paths[nm] = str(p)
        base_args = ["read", "--name", "sub", "--seam", paths["arm"], "--base", paths["u400"], "--also", paths["p2"], "--n-boot", "100", "--no-verify-boot"]
        assert CA.main(base_args + ["--out", str(Path(td) / "s1.json")]) == 1                       # INVALID (counts)
        assert CA.main(base_args + ["--subset-ok", "--out", str(Path(td) / "s2.json")]) == 2          # counts waived; still INCOMPLETE (no dump), not INVALID


@check("D7 no P6 output: the control read touches no path containing 'p6' (a P6 checkpoint's navtest output does not exist and must not be produced)")
def t_no_p6():
    r = _control()
    for k, v in r["control_inputs"].items():
        assert "p6" not in str(v).lower(), (k, v)
    for a in ("p2", "u400"):
        assert "p6" not in r["arms"][a]["seam"].lower()
    must_fail(lambda: _assert_none(["p6"] if "p6" in "D:/x/stage2_p6_full.npz" else []))


def collect():
    return [v for k, v in globals().items() if k.startswith("t_") and hasattr(v, "check_name")]         # dict order = definition order


def main(argv=None) -> int:
    for t in collect():
        t()
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, why in RESULTS:
        print(("PASS " if ok else "FAIL ") + name + (f"  <-- {why}" if why else ""))
    print(f"ZZADH {'PASS' if npass == len(RESULTS) else 'FAIL'} {npass}/{len(RESULTS)}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
