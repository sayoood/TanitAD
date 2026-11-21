"""Tests for the refcv8 STAGE-2 battery kit: r8_battery_patch.py, run_battery_r8.py, pair_read_r8.py.

Run (dev box; SEPARATE pytest invocation from test_r8_repoint.py -- both import the loader, which reads its env at import):
    PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval" OMP_NUM_THREADS=6 \
        python -m pytest -q -p no:cacheprovider test_r8_battery.py

Every expected value is a LITERAL (hand-derived or closed-form), never an expression over the code under test.  Each guard has
a `test_REGRESSION_*` arm that reintroduces the real defect and must go RED on the check that is green with the fix:
  (1) a roll child NOT rewritten (it would run without the pose-sync policy);   (2) a baseline path that exists nowhere;
  (3) a G0 stop that is not lifted / an anchor that no longer matches;           (4) an optional baseline that cannot pair;
  (5) a refcv8 arm that is WORSE than refcv7 by a lateral offset -> BAR-R8-NR must FAIL, ADE delta > 0, separated;
  (6) a model-free control that is not bit-identical.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_repoint  # noqa: E402

TREE = r8_repoint.LAUNCH_TREE
needs_tree = pytest.mark.skipif(not (TREE / "stack" / "scripts" / "refc_v3_train.py").exists(),
                                reason="launch tree C:/lgs/Lfb47a58376 absent (dev-box-only test)")
needs_kit = pytest.mark.skipif(not Path("E:/refcv8_eval_kit/ckpt/config.json").exists(),
                               reason="E:/refcv8_eval_kit absent (dev box only)")


@pytest.fixture(scope="module")
def stack():
    """BP.apply() once: (BP, loader, r7_panel, refav1_arm)."""
    import r8_battery_patch as BP
    L, P = BP.apply()
    return BP, L, P, P.RR.ra3().ra


# ------------------------------------------------------------------------------------------ run_battery_r8.prepare_argv
def test_prepare_argv_refusals_and_defaults(monkeypatch):
    import run_battery_r8 as W
    monkeypatch.setenv("R8_NO_POSE_SYNC", "1")
    with pytest.raises(SystemExit, match="g0-json"):
        W.prepare_argv(["--ckpt", "x", "--tag", "t"])
    with pytest.raises(SystemExit, match="--tag"):
        W.prepare_argv(["--ckpt", "x", "--no-g0"])
    with pytest.raises(SystemExit, match="standalone clock"):                      # --no-baselines on the paired pass
        W.prepare_argv(["--ckpt", "x", "--no-g0", "--tag", "t", "--no-baselines"])
    argv, opt = W.prepare_argv(["--ckpt", "x", "--no-g0", "--tag", "t", "--g0-nonblocking"])
    assert opt == {"smoke": False, "nonblock": True, "nobase": False, "osall": False, "help": False}
    assert "--g0-nonblocking" not in argv                                          # stripped before RB's own argparse
    assert argv[argv.index("--config") + 1] == "E:/refcv8_eval_kit/ckpt/config.json"
    assert argv[argv.index("--out-root") + 1] == "E:/refcv8_eval_kit/battery"
    monkeypatch.delenv("R8_NO_POSE_SYNC")                                          # the corrected clock
    with pytest.raises(SystemExit, match="cannot be paired"):
        W.prepare_argv(["--ckpt", "x", "--no-g0", "--tag", "t"])
    argv, opt = W.prepare_argv(["--ckpt", "x", "--no-g0", "--tag", "t", "--no-baselines"])
    assert opt["nobase"] is True


def test_prepare_argv_smoke_limits(monkeypatch):
    import run_battery_r8 as W
    monkeypatch.setenv("R8_NO_POSE_SYNC", "1")
    with pytest.raises(SystemExit, match="<=2"):
        W.prepare_argv(["--smoke-cpu", "--ckpt", "x", "--no-g0", "--tag", "t", "--max-clips", "3", "--max-windows-per-clip", "3"])
    with pytest.raises(SystemExit, match="<=2"):
        W.prepare_argv(["--smoke-cpu", "--ckpt", "x", "--no-g0", "--tag", "t"])
    argv, opt = W.prepare_argv(["--smoke-cpu", "--ckpt", "x", "--no-g0", "--tag", "t", "--max-clips", "2",
                                "--max-windows-per-clip", "3"])
    assert argv[argv.index("--device") + 1] == "cpu" and "--lock-held" in argv       # never takes the GPU lock
    assert argv[argv.index("--n-boot") + 1] == "50" and opt["smoke"] is True


# ------------------------------------------------------------------------------------------ child command rewrite
def test_rewrite_child_cmd():
    import r8_battery_patch as BP
    r7 = "C:/lgs/Lfb47a58376/FlyWheels/x/battery/code/roll_seed_r7.py"
    cmd = BP.rewrite_child_cmd(["python", r7, "--seed", "1", "--os-only"])
    assert cmd[1].replace("\\", "/").endswith("/code/roll_seed_r8.py") and "roll_seed_r7" not in " ".join(cmd)
    assert cmd[2:] == ["--seed", "1", "--os-only"]                                    # argv otherwise untouched
    assert BP.rewrite_child_cmd(["python", "C:/x/criteria_check.py", "a"]) == ["python", "C:/x/criteria_check.py", "a"]
    assert BP.rewrite_child_cmd(["python", r7], smoke_cpu=True)[-1] == "--smoke-cpu-fp32-trunk"
    c = BP.rewrite_child_cmd(["python", r7, "--seed", "0"], os_only_all=True)           # the corrected-clock pass: os-only at EVERY seed
    assert c[-1] == "--os-only" and c.count("--os-only") == 1
    assert BP.rewrite_child_cmd(["python", r7, "--seed", "1", "--os-only"], os_only_all=True).count("--os-only") == 1
    assert "--os-only" not in BP.rewrite_child_cmd(["python", r7, "--seed", "0"])


def test_REGRESSION_unrewritten_child_would_run_unpatched():
    """Without the rewrite the command still names the refcv7 child, which carries NO pose-sync policy: the assertion that is
    green with the rewrite goes red here."""
    r7 = "C:/lgs/Lfb47a58376/FlyWheels/x/battery/code/roll_seed_r7.py"
    cmd = ["python", r7, "--seed", "0"]
    assert cmd[1].endswith("roll_seed_r7.py")
    import r8_battery_patch as BP
    assert BP.rewrite_child_cmd(cmd)[1] != cmd[1]


# ------------------------------------------------------------------------------------------ dumps on disk (synthetic)
def make_dump(root: Path, windows: dict, *, with_decisions=True) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "decisions").mkdir(exist_ok=True)
    eps = []
    for i, (cid, ws) in enumerate(sorted(windows.items())):
        np.savez(root / f"ep{i:03d}.npz", ws=np.asarray(ws, np.int64))
        if with_decisions:
            np.savez(root / "decisions" / f"ep{i:03d}.npz", x=np.zeros(1))
        eps.append({"file_index": i, "clip_id": cid, "episode_index": i, "n_windows": len(ws)})
    json.dump({"episodes": eps}, open(root / "manifest.json", "w"))
    return str(root)


def test_fix_baselines_repoints_by_content(tmp_path):
    import r8_battery_patch as BP
    new = make_dump(tmp_path / "art" / "b_x", {"c1": [5, 6]})
    P = types.SimpleNamespace(BASELINES={"b_x": str(tmp_path / "gone" / "b_x")})
    rec = BP.fix_baselines(P, roots=[str(tmp_path / "nowhere"), str(tmp_path / "art")])
    assert P.BASELINES["b_x"] == new and rec["b_x"]["moved"] is True


def test_REGRESSION_baseline_in_no_root_refuses(tmp_path):
    import r8_battery_patch as BP
    P = types.SimpleNamespace(BASELINES={"b_x": str(tmp_path / "gone" / "b_x")})
    with pytest.raises(SystemExit, match="absent"):
        BP.fix_baselines(P, roots=[str(tmp_path / "nowhere")])


def test_drop_unusable_optional_and_required_refuses(tmp_path):
    import r8_battery_patch as BP
    win = {"c1": [5, 6, 7], "c2": [9]}
    good = make_dump(tmp_path / "good", win)
    short = make_dump(tmp_path / "short", {"c1": [5, 6], "c2": [9]})              # lacks (c1, 7)
    P = types.SimpleNamespace(BASELINES={"b_refcv4b": good}, REFCV6_38K={"b_opt": short})
    dropped = BP.drop_unusable_optional(P, win)
    assert list(dropped) == ["b_opt"] and "b_opt" not in P.REFCV6_38K and "1 S2 windows absent" in dropped["b_opt"]
    # REGRESSION arm: the REQUIRED window source cannot pair -> refuses (it must never be silently dropped)
    P2 = types.SimpleNamespace(BASELINES={"b_refcv4b": short}, REFCV6_38K={})
    with pytest.raises(SystemExit, match="REQUIRED"):
        BP.drop_unusable_optional(P2, win)


def test_add_refcv7_baselines_registers_tier_pairs_and_checks_dumps(tmp_path):
    """build_s6_dump does `TIERS[arm]` for every baseline arm: a label missing from TIERS is a KeyError hours into the run
    (this is the defect the 2026-10-15 CPU smoke found).  REGRESSION: an absent refcv7 dump refuses."""
    import r8_battery_patch as BP
    for s in (0, 1):
        make_dump(tmp_path / "r7" / f"dump_s{s}", {"c1": [5]})
    P = types.SimpleNamespace(REFCV6_38K={}, TIERS={"os": "T1"})
    RB = types.SimpleNamespace(PAIRS=[("ha", "os", "x")])
    added = BP.add_refcv7_baselines(P, RB, root=str(tmp_path / "r7"))
    assert sorted(added) == ["b_refcv7_50400_s0", "b_refcv7_50400_s1"]
    assert P.TIERS["b_refcv7_50400_s0"] == "T1" and P.TIERS["b_refcv7_50400_s1"] == "T1"
    assert ("b_refcv7_50400_s1", "os", "os_minus_refcv7_50400_s1") in RB.PAIRS
    assert ("ha0_ext", "b_refcv7_50400_s0", "refcv7_50400_s0_minus_ha0ext") in RB.PAIRS
    with pytest.raises(SystemExit, match="incomplete"):
        BP.add_refcv7_baselines(types.SimpleNamespace(REFCV6_38K={}, TIERS={}), None, root=str(tmp_path / "nowhere"))


# ------------------------------------------------------------------------------------------ the G0-stop edit
def test_patched_source_anchor_exactly_once():
    import r8_battery_patch as BP
    src = (r8_repoint.BATTERY_CODE / "run_battery_r7.py").read_text(encoding="utf-8")
    assert src.count(BP._ANCHOR) == 1                                                # the upstream file still matches
    out = BP.patched_source(src)
    compile(out, "run_battery_r7.py", "exec")
    assert "R8_G0_NONBLOCKING" in out and out.count('if not a.g0_text_verdict:') == src.count('if not a.g0_text_verdict:')
    with pytest.raises(SystemExit, match="0 times"):                                # REGRESSION: upstream changed
        BP.patched_source(src.replace(BP._ANCHOR, "# gone\n"))
    with pytest.raises(SystemExit, match="2 times"):                                # REGRESSION: anchor duplicated
        BP.patched_source(src + "\n" + BP._ANCHOR)


@needs_tree
def test_g0_stop_behaviour_nonblocking_vs_blocking(stack, tmp_path):
    """The REAL `_run`, driven to the point right after the G0 stage: a sentinel replaces the next call (`_clip_ids_by_index`).
    FAIL + blocking -> SystemExit 'STOP'; FAIL + non-blocking -> reaches the sentinel and stamps G0_NONBLOCKING; PASS -> both reach it."""
    BP = stack[0]

    class Sentinel(Exception):
        pass

    def drive(g0_verdict: str, nonblocking: bool):
        mod = BP.load_battery_module(g0_nonblocking=nonblocking)
        mod._clip_ids_by_index = lambda *a, **k: (_ for _ in ()).throw(Sentinel())
        g0 = {"ckpt_md5": "abc", "verdict": {"G0": g0_verdict, "reasons": ["r1"], "by_class_counts": {}, "medians": {},
                                              "mutation_detection": {}}}
        p = tmp_path / f"g0_{g0_verdict}_{nonblocking}.json"
        json.dump(g0, open(p, "w"))
        a = types.SimpleNamespace(g0_json=str(p), no_g0=False, g0_text_verdict=None, g0_text_ruling=None)
        summary = {"ckpt_md5": "abc", "stages": {}}
        try:
            mod._run(a, summary, tmp_path, lambda: None)
        except Sentinel:
            return "REACHED_ROLLS", summary
        except SystemExit as exc:
            return f"EXIT:{exc}", summary
        return "RETURNED", summary

    r, s = drive("FAIL", False)
    assert r.startswith("EXIT:") and "G0 = FAIL -- STOP" in r and "G0_NONBLOCKING" not in s     # REGRESSION arm (default behaviour)
    r, s = drive("FAIL", True)
    assert r == "REACHED_ROLLS" and s["G0_NONBLOCKING"]["G0_as_coded"] == "FAIL" and "PROVISIONAL" in s["G0_NONBLOCKING"]["stamp"]
    assert drive("PASS", False)[0] == "REACHED_ROLLS" and "G0_NONBLOCKING" not in drive("PASS", True)[1]


REAL_G0 = Path("D:/refcv7_eval_kit/battery/step50400/g0.json")


@needs_tree
@pytest.mark.skipif(not REAL_G0.exists(), reason="banked refcv7 g0.json absent (dev box only)")
def test_g0_stage_reads_a_real_g0_artifact_without_crashing(stack, tmp_path):
    """The G0 block of the real `_run` on a REAL banked artifact (refcv7-50,400: coded FAIL), non-blocking: it must reach the
    rolls (sentinel) and stamp the nonblocking record -- the schema refcv8's g0_refcv7.main() writes is this one."""
    BP = stack[0]

    class Sentinel(Exception):
        pass

    mod = BP.load_battery_module(g0_nonblocking=True)
    mod._clip_ids_by_index = lambda *a, **k: (_ for _ in ()).throw(Sentinel())
    g0 = json.load(open(REAL_G0, encoding="utf-8"))
    assert (g0["verdict"]["G0"]) == "FAIL"                                            # the banked fact this test leans on
    a = types.SimpleNamespace(g0_json=str(REAL_G0), no_g0=False, g0_text_verdict=None, g0_text_ruling=None)
    summary = {"ckpt_md5": g0["ckpt_md5"], "stages": {}}
    with pytest.raises(Sentinel):
        mod._run(a, summary, tmp_path, lambda: None)
    assert summary["G0_NONBLOCKING"]["G0_as_coded"] == "FAIL" and summary["stages"]["g0"]["G0"] == "FAIL"
    assert summary["stages"]["g0"]["by_class_counts"]                                   # real content flowed through
    mod2 = BP.load_battery_module(g0_nonblocking=False)                                 # REGRESSION: the default still stops
    mod2._clip_ids_by_index = lambda *a, **k: (_ for _ in ()).throw(Sentinel())
    with pytest.raises(SystemExit, match="STOP"):
        mod2._run(a, {"ckpt_md5": g0["ckpt_md5"], "stages": {}}, tmp_path, lambda: None)


# ------------------------------------------------------------------------------------------ pair_read: pure bars
def cell(delta, lo, hi, n=4754):
    return {"delta": delta, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "n_windows": n, "n_episodes": 139}


def test_bar_E_literals():
    import pair_read_r8 as PR
    ok = {0: cell(-0.5, -0.9, -0.1), 1: cell(-0.5, -0.9, -0.1)}
    assert PR.bar_E(ok, 0.001)["verdict"].startswith("PASS")
    assert PR.bar_E(ok, 0.3)["verdict"].startswith("NOT PROVEN (|delta| within 2x")      # 0.5 <= 2 x 0.3
    assert PR.bar_E(ok, None)["verdict"].startswith("NOT PROVEN (no inference-seed floor)")
    bad = {0: cell(-0.5, -0.9, -0.1), 1: cell(-0.01, -0.05, 0.03)}                       # seed 1 not separated
    assert PR.bar_E(bad, 0.001)["verdict"] == "FAILED"
    wrong_way = {0: cell(0.2, 0.1, 0.3), 1: cell(0.2, 0.1, 0.3)}                          # separated but os WORSE than the echo
    assert PR.bar_E(wrong_way, 0.001)["verdict"] == "FAILED"
    assert PR.bar_E({0: ok[0]}, 0.001)["verdict"].startswith("NOT EVALUABLE")


def test_bar_NR_metric_literals():
    import pair_read_r8 as PR
    assert PR.bar_NR_metric({0: cell(0.001, -0.01, 0.02), 1: cell(-0.002, -0.01, 0.005)})["verdict"].startswith("PASS")
    assert PR.bar_NR_metric({0: cell(0.01, 0.002, 0.02), 1: cell(0.0, -0.01, 0.01)})["verdict"].startswith("FAIL")   # lo > 0 at one seed
    assert PR.bar_NR_metric({0: cell(0.0, -0.1, 0.1)})["verdict"] == "NOT EVALUABLE"
    r = PR.rollup_NR({"a": {"verdict": "PASS (x)"}, "b": {"verdict": "FAIL (separated worse)"}, "c": {"verdict": "PASS (x)"}},
                     {"a": "LATERAL", "b": "LATERAL", "c": "LONGITUDINAL"})
    assert r == {"LATERAL": "FAIL: b", "LONGITUDINAL": "PASS"}


# ------------------------------------------------------------------------------------------ pair_read: statistics
@needs_tree
def test_paired_cell_identical_inputs_is_exactly_zero(stack):
    import pair_read_r8 as PR
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    eid = np.repeat(np.arange(30), 10)
    c = PR.paired_cell(x, x.copy(), eid, n_boot=200)
    assert c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0 and c["separated"] is False
    assert c["n_windows"] == 300 and c["n_episodes"] == 30 and c["estimator"] == "paired_episode_cluster_bootstrap"


@needs_tree
def test_paired_cell_literal_shift_and_nan_drop(stack):
    import pair_read_r8 as PR
    x = np.arange(100, dtype=float)
    eid = np.repeat(np.arange(10), 10)
    c = PR.paired_cell(x + 0.25, x, eid, n_boot=200)                 # every window is exactly 0.25 worse
    assert c["delta"] == 0.25 and c["lo"] == 0.25 and c["hi"] == 0.25 and c["separated"] is True
    y = x.copy()
    y[3] = np.nan
    d = PR.paired_cell(x, y, eid, n_boot=200)
    assert d["n_dropped_nonfinite"] == 1 and d["n_windows"] == 99


@needs_tree
def test_components_analytic_circle_vs_straight(stack):
    """Closed form: GT = arc of R = 50 m at 10 m/s on a 0.5 s grid (4 instants); pred = straight line at 10 m/s.
    curvature_gt = dtheta / chord = 0.1 / 4.997916927 = 0.0200083358; final along 20 - 19.4709171 = 0.5290829;
    final cross 3.9469503; speed error |10 - 9.9958339| = 0.0041661 at every step; yaw-rate error = dtheta/dt = 0.2 rad/s."""
    import pair_read_r8 as PR
    ra = stack[3]
    R_, v, dt, H = 50.0, 10.0, 0.5, 4
    th = np.arange(1, H + 1) * (v * dt / R_)
    gt = np.stack([R_ * np.sin(th), R_ * (1 - np.cos(th))], -1)[None].astype(np.float32)
    pr = np.stack([v * dt * np.arange(1, H + 1), np.zeros(H)], -1)[None].astype(np.float32)
    c = PR.components(ra, pr, gt, dt)
    assert abs(float(c["LAT_curvature_mae_1pm_valid"][0]) - 0.0200083358) < 2e-6
    assert abs(float(c["LON_along_abs_final_m"][0]) - 0.5290829) < 1e-4
    assert abs(float(c["LAT_cross_abs_final_m"][0]) - 3.9469503) < 1e-4
    assert abs(float(c["LON_speed_mae_mps"][0]) - 0.0041661) < 1e-5
    assert abs(float(c["LAT_yaw_rate_mae_radps_valid"][0]) - 0.2) < 1e-4
    assert float(c["LON_speed_within_0p5"][0]) == 1.0
    z = PR.components(ra, gt, gt, dt)                                  # identical -> every error cell exactly 0
    for k in ("ade_m", "LON_speed_mae_mps", "LON_along_abs_final_m", "LAT_cross_abs_final_m", "LAT_curvature_mae_1pm_valid",
              "LAT_yaw_rate_mae_radps_valid", "LAT_heading_mae_deg"):
        assert float(z[k][0]) == 0.0, k


@needs_tree
def test_yaw_valid_matches_the_batterys_A3_cell(stack):
    import pair_read_r8 as PR
    BP, L, P, ra = stack
    rng = np.random.default_rng(1)
    G = np.cumsum(rng.normal([1.0, 0.0], 0.2, size=(40, 4, 2)), axis=1).astype(np.float32)
    Pp = (G + rng.normal(0, 0.1, size=G.shape)).astype(np.float32)
    mine = PR.components(ra, Pp, G, 0.5)["LAT_yaw_rate_mae_radps_valid"]
    theirs = P.yaw_rate_valid(Pp, G, 0.5)
    assert np.array_equal(np.isnan(mine), np.isnan(theirs)) and np.allclose(mine, theirs, equal_nan=True, atol=0, rtol=0)


# ------------------------------------------------------------------------------------------ pair_read: synthetic panels end to end
def make_panel(root: Path, *, regress=False, ref_label="b_refcv7_50400_s0", mf_noise=0.0, n_ep=6, n_w=5, seed=0):
    """A synthetic S2-shaped panel (H = 4, dt = 0.5): GT, os, the echo ha0_ext, the model-free arms, and the refcv7 plan."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    eps = []
    for e in range(n_ep):
        n = n_w
        g = np.cumsum(np.tile([5.0, 0.0], (n, 4, 1)) + rng.normal(0, 0.05, (n, 4, 2)), axis=1).astype(np.float32)
        os_ = g + rng.normal(0, 0.1, g.shape).astype(np.float32)
        z = {"g": g, "os": os_, "ha0_ext": g + rng.normal(0, 0.3, g.shape).astype(np.float32),
             "ha": g + 0.01, "ha0": g + 0.02, "stop": np.zeros_like(g),
             ref_label: os_.copy(), "ws": (np.arange(n) + 10 * (e + 1)).astype(np.int64), "v0": np.full(n, 5.0, np.float32)}
        if regress:                                                     # refcv8 WORSE than refcv7: a +0.5 m lateral offset on `os`
            z["os"] = os_ + np.array([0.0, 0.5], np.float32)
        if mf_noise:
            z["ha0_ext"] = z["ha0_ext"] + mf_noise
        np.savez(root / f"ep{e:03d}.npz", **z)
        eps.append({"file_index": e, "clip_id": f"clip-{e}", "episode_index": e, "n_windows": n})
    json.dump({"episodes": eps, "grid": {"dt_s": 0.5, "instants_s": [0.5, 1.0, 1.5, 2.0]}}, open(root / "manifest.json", "w"))
    return str(root)


@needs_tree
def test_read_surface_identical_refcv7_is_exactly_zero_and_NR_passes(stack, tmp_path):
    import pair_read_r8 as PR
    BP, L, P, ra = stack
    pd = make_panel(tmp_path / "panel")
    bw = {f"clip-{e}": [int(w) for w in (np.arange(5) + 10 * (e + 1))] for e in range(6)}
    sha = lambda c: c                                                     # identity keys for the synthetic ids
    out = PR.read_surface(ra, sha, pd, "b_refcv7_50400_s0", 100, bw)
    assert out["windows_check"]["equal"] is True and out["windows_check"]["n_panel"] == 30
    for key, row in out["rows"].items():
        c = row["os_minus_refcv7_50400"]
        assert c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0 and c["separated"] is False, key
    nr = PR.bar_NR_metric({0: out["rows"]["ade_m"]["os_minus_refcv7_50400"], 1: out["rows"]["ade_m"]["os_minus_refcv7_50400"]})
    assert nr["verdict"].startswith("PASS")


@needs_tree
def test_REGRESSION_worse_than_refcv7_fails_NR_and_bar_E_against_a_noisy_echo(stack, tmp_path):
    import pair_read_r8 as PR
    BP, L, P, ra = stack
    pd = make_panel(tmp_path / "panel", regress=True)
    bw = {f"clip-{e}": [int(w) for w in (np.arange(5) + 10 * (e + 1))] for e in range(6)}
    out = PR.read_surface(ra, lambda c: c, pd, "b_refcv7_50400_s0", 200, bw)
    ade = out["rows"]["ade_m"]["os_minus_refcv7_50400"]
    assert ade["delta"] > 0.3 and ade["lo"] > 0 and ade["separated"] is True          # the +0.5 m offset is detected
    cross = out["rows"]["LAT_cross_abs_final_m"]["os_minus_refcv7_50400"]
    assert cross["lo"] > 0.3
    assert PR.bar_NR_metric({0: ade, 1: ade})["verdict"].startswith("FAIL")
    # the echo is NOISIER than the clean os (0.3 vs 0.1 sigma) -> E-bar reads PASS only when os is clean; the regression erases it
    clean = PR.read_surface(ra, lambda c: c, make_panel(tmp_path / "clean"), None, 200, None)
    e_clean = clean["rows"]["ade_m"]["os_minus_ha0_ext"]
    assert e_clean["delta"] < 0 and e_clean["hi"] < 0
    assert PR.bar_E({0: e_clean, 1: e_clean}, 0.0001)["verdict"].startswith("PASS")
    # the regression flips the sign of os - echo: ADE(os + 0.5 m offset) is now WORSE than the 0.3-sigma echo
    e_reg = out["rows"]["ade_m"]["os_minus_ha0_ext"]
    assert PR.bar_E({0: e_reg, 1: e_reg}, 0.0001)["verdict"] == "FAILED"


@needs_tree
def test_windows_check_catches_a_dropped_window(stack, tmp_path):
    import pair_read_r8 as PR
    pd = make_panel(tmp_path / "panel")
    bw = {f"clip-{e}": [int(w) for w in (np.arange(5) + 10 * (e + 1))] for e in range(6)}
    man, A, eid, keys = PR.load_panel_keyed(pd, lambda c: c)
    assert PR.windows_check(keys, bw, lambda c: c)["equal"] is True
    bw["clip-2"] = bw["clip-2"][:-1]                                                   # REGRESSION: the baseline lost a window
    r = PR.windows_check(keys, bw, lambda c: c)
    assert r["equal"] is False and r["only_in_panel"] == 1 and r["only_in_baseline"] == 0


@needs_tree
def test_controls_bit_identical_and_regression(stack, tmp_path):
    import pair_read_r8 as PR
    ra = stack[3]
    a = make_panel(tmp_path / "a")
    b = make_panel(tmp_path / "b")
    c = PR.controls(ra, lambda x: x, a, b, 50)
    assert c["status"] == "OK" and c["all_bit_identical"] is True
    assert c["arms"]["ha0_ext"]["paired_ade_delta"]["delta"] == 0.0
    bad = make_panel(tmp_path / "bad", mf_noise=1e-3)                                  # REGRESSION: model-free arm drifted
    c2 = PR.controls(ra, lambda x: x, bad, b, 50)
    assert c2["all_bit_identical"] is False and c2["arms"]["ha0_ext"]["bit_identical"] is False


@needs_tree
def test_controls_subset_and_foreign_windows(stack, tmp_path):
    import pair_read_r8 as PR
    ra = stack[3]
    big = make_panel(tmp_path / "big", n_ep=6, n_w=5)
    small = make_panel(tmp_path / "small", n_ep=4, n_w=5)                 # a restricted run: a strict subset of the baseline
    c = PR.controls(ra, lambda x: x, small, big, 50)
    assert c["status"].startswith("OK-SUBSET") and c["all_bit_identical"] is True
    other = make_panel(tmp_path / "other", n_ep=6, n_w=5, seed=0)
    for f in sorted((tmp_path / "other").glob("ep000.npz")):                # REGRESSION: windows the baseline does not have
        z = dict(np.load(f))
        z["ws"] = z["ws"] + 1000
        np.savez(f, **z)
    assert PR.controls(ra, lambda x: x, other, big, 50)["status"] == "WINDOWS DIFFER"


def test_strategic_stamp(tmp_path):
    import pair_read_r8 as PR
    p = tmp_path / "config.json"
    json.dump({"argv": ["--arm", "hier", "--no-strategic"]}, open(p, "w"))
    s = PR.strategic_stamp(str(p))
    assert s["status"] == "NOT APPLICABLE" and s["n"] == 0 and "OFF" in s["reason"]
    json.dump({"argv": ["--arm", "hier"]}, open(p, "w"))
    assert PR.strategic_stamp(str(p))["status"] == "NOT EVALUATED HERE"


def test_g0_stamp_provisional(tmp_path):
    import pair_read_r8 as PR
    p = tmp_path / "g0.json"
    json.dump({"ckpt_md5": "m", "verdict": {"G0": "FAIL", "amendment": "A7",
                                             "mutation_detection": {"m1": {"detected": True}}}}, open(p, "w"))
    g = PR.g0_stamp(str(p))
    assert g["provisional"] is True and "PROVISIONAL" in g["stamp"] and g["M1_detected"] is True
    json.dump({"ckpt_md5": "m", "verdict": {"G0": "PASS", "mutation_detection": {"m1": {"detected": True}}}}, open(p, "w"))
    assert PR.g0_stamp(str(p))["provisional"] is False
    json.dump({"ckpt_md5": "m", "verdict": {"G0": "PASS", "mutation_detection": {"m1": {"detected": False}}}}, open(p, "w"))
    assert PR.g0_stamp(str(p))["provisional"] is True                                  # PASS with M1 undetected is still VOID


def _g0(tmp_path, reasons, m1=True, g0="FAIL", md5="m", step=20000):
    p = tmp_path / "g0.json"
    json.dump({"ckpt_md5": md5, "step": step, "verdict": {"G0": g0, "amendment": "A7", "reasons": reasons,
                                                          "mutation_detection": {"m1": {"detected": m1}}}}, open(p, "w"))
    return str(p)


def _j(tmp_path, name, d):
    p = tmp_path / name
    json.dump(d, open(p, "w"))
    return str(p)


WRAP_ONLY = ["wrapper clause FAIL: ['merge defect: lat_tac wrapper 1.065e-05 > bar 1.000e-05']"]
FP64_OK = {"step": 20000, "criteria": {"F64_rel_le_1e-9": True}, "F64_n_terms_wrapper_gt_1e-9": 0,
           "F64_W1_W2_detected": [True, True], "F64_max_wrapper_rel": 2e-14}


def test_amended_g0_registered_reading(tmp_path):
    """W-R8-1 / W-R8-2 (PI 2026-10-17): literal expectations, each deliberate defect must read NOT PASS."""
    import pair_read_r8 as PR
    g = _g0(tmp_path, WRAP_ONLY)
    wpass = _j(tmp_path, "wp.json", {"ok": True, "ckpt_md5": "m", "CLAUSE": "PASS", "max_wrapper_rel": 3e-6})
    wfail = _j(tmp_path, "wf.json", {"ok": True, "ckpt_md5": "m", "CLAUSE": "FAIL", "max_wrapper_rel": 1.06e-5})
    f_ok = _j(tmp_path, "f.json", FP64_OK)
    s = PR.g0_stamp(g, wpass)
    assert (s["G0"], s["G0_amended"], s["amendments"], s["provisional"]) == ("FAIL", "PASS", ["A7", "W-R8-1"], False)
    s = PR.g0_stamp(g, wfail, f_ok)
    assert (s["G0_amended"], s["amendments"], s["provisional"]) == ("PASS", ["A7", "W-R8-1", "W-R8-2"], False)
    # defects -> NOT PASS, PROVISIONAL
    assert PR.g0_stamp(g)["provisional"] is True                                              # no probe
    assert PR.g0_stamp(g, wfail)["provisional"] is True                                       # FAIL, no fp64
    other = _j(tmp_path, "wo.json", {"ok": True, "ckpt_md5": "OTHER", "CLAUSE": "PASS"})
    assert PR.g0_stamp(g, other)["provisional"] is True                                       # another checkpoint
    notok = _j(tmp_path, "wn.json", {"ok": False, "ckpt_md5": "m", "CLAUSE": "PASS"})
    assert PR.g0_stamp(g, notok)["provisional"] is True                                       # probe not ok
    for k, v in (("F64_n_terms_wrapper_gt_1e-9", 1), ("step", 15000), ("F64_W1_W2_detected", [True, False]),
                 ("F64_W1_W2_detected", []), ("criteria", {"F64_rel_le_1e-9": False})):
        bad = dict(FP64_OK, **{k: v})
        assert PR.g0_stamp(g, wfail, _j(tmp_path, "fb.json", bad))["provisional"] is True, (k, v)
    two = _g0(tmp_path, ["eval_box3d_det_ap2_automobile_0_20 [DETECTION] moved"] + WRAP_ONLY)
    assert PR.g0_stamp(two, wpass)["provisional"] is True                                     # step-5,000 shape
    nom1 = _g0(tmp_path, WRAP_ONLY, m1=False)
    assert PR.g0_stamp(nom1, wpass)["provisional"] is True                                    # M1 blind


def test_read_label_and_stamp():
    import pair_read_r8 as PR
    lab = PR.read_label("C:/x/battery/step30000ema_unc", None)
    assert lab == {"step": 30000, "weights": "EMA", "total": 48852, "pass": "unc"}
    assert PR.read_stamp(lab) == "EARLY READ at step 30,000 of 48,852: not the pre-registered final-checkpoint bar verdict"
    lab = PR.read_label("E:/k/battery/step48852_corr/", None)
    assert (lab["weights"], lab["pass"]) == ("raw", "corr")
    assert PR.read_stamp(lab).startswith("FINAL checkpoint (step 48,852 of 48,852)")
    assert PR.read_label("E:/k/battery/smoke", None)["step"] is None


def test_bank_stamps_follow_the_tag():
    import bank_battery_r8 as B
    tier, st = B.bank_stamps("step30000ema")
    assert tier.endswith("EARLY READ at step 30,000 of 48,852, EMA weights")
    assert "5,000" not in " ".join(st) and "G0_amended" in st[2]
    assert B.bank_stamps("step48852")[1][1] == "the pre-registered bar verdict (final checkpoint)"
