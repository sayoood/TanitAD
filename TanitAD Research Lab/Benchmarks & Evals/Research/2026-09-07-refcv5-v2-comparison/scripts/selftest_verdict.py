#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest_verdict.py — prove the harness's VERDICT and INSTRUMENT checks are
REACHABLE IN BOTH DIRECTIONS, by mutation.

⛔ WHY THIS EXISTS. An AST census once read 0 suspects on BOTH the fixed and the
broken trainer: inspecting a guard proves nothing about whether its failure
branch can fire. So this file does not read ``refcv5_compare.py`` — it BUILDS
synthetic dumps that force each branch and asserts the branch actually fires.
A verdict machine that can only say PASS is a rubber stamp, and one that can
only say FAIL is a wall; both are useless and both look identical from the
outside on a run that happened to land on their one answer.

Each case writes a throwaway dump + result JSON and runs the real
``refcv5_compare.main`` on it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refcv5_compare as C  # noqa: E402

N_EP, N_WIN = 12, 20
RNG = np.random.default_rng(0)


def _traj(scale, n):
    """[n, 4, 2] paths whose per-window ADE against g is ~scale."""
    return RNG.normal(0.0, scale, size=(n, 4, 2))


def make_dump(path, scales, *, route_ok=0.8, seed=0, straight_ha0_broken=False):
    """Synthesise a dump dir. ``scales`` maps arm -> per-window error scale."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(path, "decisions"), exist_ok=True)
    for e in range(N_EP):
        g = np.zeros((N_WIN, 4, 2), dtype=np.float32)
        z = {"g": g, "eid": np.array([e]), "ws": np.arange(N_WIN) * 5,
             "v0": np.full(N_WIN, 10.0), "clip_index": np.array([e])}
        for arm, s in scales.items():
            if arm == "ha0" and not straight_ha0_broken:
                # ⭐ ha0 is CONSTANT VELOCITY: a genuine straight line, y == 0 at
                # every slot. The instrument check runs the REAL masked
                # curvature estimator over these paths and demands exactly 0
                # back, so a fixture of random noise would (correctly) fail it —
                # MEASURED here on 2026-09-07 when it did.
                v = 10.0 + rng.normal(0.0, 2.0, size=(N_WIN, 1))
                t = np.arange(1, 5)[None, :] * 0.5
                z[arm] = np.stack([v * t, np.zeros((N_WIN, 4))],
                                  axis=-1).astype(np.float32)
            else:
                z[arm] = rng.normal(0.0, s, size=(N_WIN, 4, 2)).astype(np.float32)
        np.savez(os.path.join(path, "ep%03d.npz" % e), **z)
        lab = rng.integers(0, 3, size=N_WIN)
        prd = lab.copy()
        flip = rng.random(N_WIN) > route_ok
        prd[flip] = (prd[flip] + 1) % 3
        np.savez(os.path.join(path, "decisions", "ep%03d.npz" % e),
                 route_label=lab, route_pred_nav_true=prd,
                 route_pred_nav_shuffled=prd, route_pred_nav_zero=prd,
                 nav_cmd=np.array([C.__dict__ and 1] * N_WIN), ws=np.arange(N_WIN) * 5)
    return path


def make_json(path, dump, arms, *, kappa_ha0=0.0, curv_bias_exact=True):
    """A refcv3_arm-shaped result JSON with just the fields the table reads."""
    d = {"tool": "selftest", "n_windows": N_EP * N_WIN, "n_episodes": N_EP,
         "dt_s": 0.5, "horizon_steps": 4, "ckpt": "SYNTHETIC", "arms": {}}
    for arm in arms:
        curv_mae = 0.0068 if arm == "ha0" else 0.0081
        bias = (-0.3 * curv_mae if curv_bias_exact else -3.0 * curv_mae)
        d["arms"][arm] = {
            "tier": "T1",
            "four_families": {
                "longitudinal": {"speed_mae_mps": 0.3, "speed_bias_mps": 0.01,
                                 "target_speed_acc": 0.8, "along_mae_m": 0.25,
                                 "distance_keeping": {"status": "OK", "n": 100}},
                "lateral": {"heading_mae_deg": 1.3, "yaw_rate_mae_degps": 1.7,
                            "curvature_mae_1pm": curv_mae,
                            "curvature_bias_1pm": bias,
                            "cross_mae_m": 0.1, "n_steps_curvature": 500,
                            "excluded_below_min_ds": 40, "min_ds_m": 0.25},
                "tactical": {"status": "OK", "n": N_EP * N_WIN,
                             "lateral_decision": {"accuracy": 0.9,
                                                  "kappa": (kappa_ha0 if arm == "ha0" else 0.8),
                                                  "n": N_EP * N_WIN},
                             "longitudinal_decision": {"accuracy": 0.8,
                                                       "kappa": (kappa_ha0 if arm == "ha0" else 0.5),
                                                       "n": N_EP * N_WIN},
                             "goal_setting": {"goal_point_error_m": 0.6,
                                              "goal_bearing_mae_deg": 1.5, "n": 100}},
                "strategic": {"status": "UNAVAILABLE", "n": 0, "reason": "synthetic"},
            },
            "intervals": {"estimator": "synthetic",
                          "metrics": {"ade_m": {"mean": 0.3, "lo": 0.28, "hi": 0.32}}},
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    return path


def run(tmp, new_scales, *, kappa_ha0=0.0, curv_bias_exact=True, seed=0,
        straight_ha0_broken=False):
    nd, bd = os.path.join(tmp, "new_dump"), os.path.join(tmp, "base_dump")
    make_dump(nd, new_scales, seed=seed, straight_ha0_broken=straight_ha0_broken)
    make_dump(bd, {"os": 0.30, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29}, seed=99)
    arms = list(new_scales)
    nj = make_json(os.path.join(tmp, "new.json"), nd, arms,
                   kappa_ha0=kappa_ha0, curv_bias_exact=curv_bias_exact)
    bj = make_json(os.path.join(tmp, "base.json"), bd, arms)
    pre = os.path.join(tmp, "cmp")
    argv = ["--new-json", nj, "--new-dump", nd, "--new-label", "SYN",
            "--base-json", bj, "--base-dump", bd, "--base-label", "BASE",
            "--n-boot", "300", "--out-prefix", pre]
    import io
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        C.main(argv)
    finally:
        sys.stdout = old
    return json.load(open(pre + ".json", encoding="utf-8"))


def main():
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  ⛔FAIL ") + name + ("  " + detail if detail else ""))
        if not cond:
            fails.append(name)

    tmp = tempfile.mkdtemp(prefix="ffselftest_")
    try:
        print("[1] a REAL win must read PASS  (os much better than ha0_ext)")
        r = run(os.path.join(tmp, "a"),
                {"os": 0.10, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29})
        v = r["verdicts"][0]
        check("BAR-1 fires PASS when the arm genuinely beats ha0_ext",
              v["verdict"].startswith("✅"), v["verdict"][:60])
        check("headline follows the verdict", r["headline"].startswith("HEADLINE: ✅"))

        print("[2] a TIE must read FAIL  (os identical in law to ha0_ext)")
        r = run(os.path.join(tmp, "b"),
                {"os": 0.29, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29})
        v = r["verdicts"][0]
        check("BAR-1 fires FAIL / NOT-YET-MEASURED on a tie",
              not v["verdict"].startswith("✅"), v["verdict"][:60])

        print("[3] a LOSS must read FAIL  (os worse than ha0_ext)")
        r = run(os.path.join(tmp, "c"),
                {"os": 0.60, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29})
        v = r["verdicts"][0]
        check("BAR-1 fires FAIL when the arm loses to ha0_ext",
              v["verdict"].startswith("⛔ FAIL"), v["verdict"][:60])

        print("[4] the INSTRUMENT check must be able to FAIL "
              "(ha0 tactical kappa != the no-information value)")
        r = run(os.path.join(tmp, "d"),
                {"os": 0.10, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29},
                kappa_ha0=0.42)
        cc = r["constant_arm_checks"]
        check("constant-arm check REFUSES a non-zero ha0 kappa",
              not cc["all_pass"] and cc["n_pass"] < cc["n"],
              "n_pass=%d/%d" % (cc["n_pass"], cc["n"]))

        print("[5] the INSTRUMENT check must be able to FAIL "
              "(ha0 |curvature bias| exceeds its MAE, and a non-straight ha0)")
        r = run(os.path.join(tmp, "e"),
                {"os": 0.10, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29},
                curv_bias_exact=False, straight_ha0_broken=True)
        cc = r["constant_arm_checks"]
        check("constant-arm check REFUSES a non-straight ha0 AND a bias > MAE",
              cc["n"] - cc["n_pass"] >= 2,
              "n_pass=%d/%d" % (cc["n_pass"], cc["n"]))

        print("[6] a CLEAN instrument must PASS all constant-arm checks")
        r = run(os.path.join(tmp, "f"),
                {"os": 0.10, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29})
        check("constant-arm check PASSES on a clean instrument",
              r["constant_arm_checks"]["all_pass"])

        print("[7] the MODEL-FREE control must read exactly 0 across two dumps")
        # the same synthetic ha0 in both dumps -> paired delta must be 0
        nd = os.path.join(tmp, "g", "d")
        make_dump(nd, {"os": 0.10, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29}, seed=7)
        arms = ["os", "ha", "ha0", "ha0_ext"]
        nj = make_json(os.path.join(tmp, "g", "n.json"), nd, arms)
        pre = os.path.join(tmp, "g", "cmp")
        import io
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            C.main(["--new-json", nj, "--new-dump", nd, "--new-label", "SYN",
                    "--base-json", nj, "--base-dump", nd, "--base-label", "SAME",
                    "--n-boot", "200", "--out-prefix", pre])
        finally:
            sys.stdout = old
        r = json.load(open(pre + ".json", encoding="utf-8"))
        ctrl = [v for k, v in r["paired"].items() if k.startswith("CONTROL")]
        check("every model-free CONTROL cell reads delta=0, CI [0,0]",
              bool(ctrl) and all(c["delta"] == 0.0 and c["lo"] == 0.0
                                 and c["hi"] == 0.0 for c in ctrl),
              "n_control=%d" % len(ctrl))

        print("[8] a MISMATCHED grid must be REFUSED, not aligned")
        bad = os.path.join(tmp, "h", "bad")
        os.makedirs(bad, exist_ok=True)
        make_dump(bad, {"os": 0.1, "ha": 0.3, "ha0": 0.67, "ha0_ext": 0.29}, seed=3)
        os.remove(os.path.join(bad, "ep000.npz"))          # one episode short
        bj = make_json(os.path.join(tmp, "h", "b.json"), bad, arms)
        refused = False
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            C.main(["--new-json", nj, "--new-dump", nd, "--new-label", "SYN",
                    "--base-json", bj, "--base-dump", bad, "--base-label", "SHORT",
                    "--n-boot", "100", "--out-prefix", os.path.join(tmp, "h", "c")])
        except SystemExit as e:
            refused = "same grid" in str(e) or "NOT the" in str(e)
        finally:
            sys.stdout = old
        check("pairing REFUSES a grid mismatch", refused)

        print("[9] a SMALL-MARGIN win must FAIL the pre-registered relative margin "
              "even though the CI separates")
        # a real but useless difference: ~5 % better than ha0_ext, bar is 10 %.
        r = run(os.path.join(tmp, "j"),
                {"os": 0.2755, "ha": 0.30, "ha0": 0.67, "ha0_ext": 0.29},
                seed=11)
        v = r["verdicts"][0]
        g = r["echo_gate1"]["new"]["per_reference"]["ha0_ext"]
        check("echo_gate REFUSES a separated CI on a sub-margin win",
              (not v["verdict"].startswith("✅")) or (g["all_slots_pass"] is False),
              "slots %d/%d, margins %s, verdict %s"
              % (g["n_slots_passing"], g["n_slots"],
                 g["relative_margin_by_slot"], v["verdict"][:40]))

        print("[10] echo_gate must REFUSE a panel missing ha0_ext entirely")
        r = run(os.path.join(tmp, "k"), {"os": 0.10, "ha": 0.30, "ha0": 0.67})
        check("echo_gate REFUSES a panel without ha0_ext",
              r["echo_gate1"]["new"].get("status") == "REFUSED",
              str(r["echo_gate1"]["new"].get("status")))

        print("[11] an EMPTY dump must be REFUSED as a mount flap, not believed")
        empty = os.path.join(tmp, "i", "empty")
        os.makedirs(empty, exist_ok=True)
        refused = False
        try:
            C.load_dump(empty)
        except SystemExit as e:
            refused = "flapped" in str(e)
        check("an empty dump dir is REFUSED with the mount-flap reason", refused)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if fails:
        print("⛔ %d SELF-TEST FAILURES: %s" % (len(fails), fails))
        return 1
    print("✅ all verdict/instrument branches proven REACHABLE in both directions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
