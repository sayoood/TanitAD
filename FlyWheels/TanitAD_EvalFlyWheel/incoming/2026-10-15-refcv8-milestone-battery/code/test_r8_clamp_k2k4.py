"""CPU tests for the clamp-ON battery path + the K2 / K4 comparison code (AMENDMENT A-R8-CLAMP-1).  NEW file.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONPATH='E:/refcv8_eval_kit/tree_de4e15b/stack;E:/refcv8_eval_kit/tree_de4e15b/taniteval' \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 -m pytest -q test_r8_clamp_k2k4.py

Every comparison here carries its DELIBERATE-REGRESSION arm: a deliberate mismatch must read RED (FAIL), and the expectation of the clamp is an
INDEPENDENTLY AUTHORED analytic one (a straight plan: progress per interval = min(v_plan, allowed) * dt), never an expression over the code under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k2k4_compare as K  # noqa: E402
import r8clamp_common as C  # noqa: E402

TREE = Path("E:/refcv8_eval_kit/tree_de4e15b")
BANKED = Path("C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc")
S2W = Path("E:/refcv8_eval_kit/windows/s2_windows.json")
PY = sys.executable


# ------------------------------------------------------------------------------------------------ synthetic extras
def synth_banked(n_ep=6, per=7, seed=0):
    rng = np.random.default_rng(seed)
    ci, ws = [], []
    for e in range(n_ep):
        for w in range(per):
            ci.append(e)
            ws.append(7 + 5 * w)
    n = len(ci)
    ex = {"clip_index": np.array(ci, np.int64), "ws": np.array(ws, np.int64),
          "v0": rng.uniform(0, 20, n).astype(np.float32), "v_max_ms": np.full(n, 8.333333, np.float32), "v_max_valid": np.ones(n, np.float32),
          "gt_max_speed_6s": rng.uniform(0, 20, n).astype(np.float32)}
    for c in ("nav_true", "nav_shuffled", "nav_flipped", "nav_zero"):
        ex[f"{c}.traj"] = rng.normal(size=(n, 8, 2)).astype(np.float32) * 30
        ex[f"{c}.sel_idx"] = rng.integers(0, 117, n).astype(np.float32)
    return ex


def take_prefix(ex, n_ep):
    m = ex["clip_index"] < n_ep
    return {k: v[m].copy() for k, v in ex.items()}


# ------------------------------------------------------------------------------------------------ provenance
def _fake_tree(tmp_path, body=b"line\ncap_plan_to_limit\n"):
    t = tmp_path / "tree"
    (t).mkdir()
    (t / "PROVENANCE.txt").write_text(f"tree: git archive {C.COMMIT_FULL} -- stack taniteval\n", encoding="utf-8")
    (t / "f.py").write_bytes(body.replace(b"\n", b"\r\n"))          # the export arrives CRLF
    return t


def test_provenance_real_tree_passes_and_names_markers():
    if not TREE.exists():
        pytest.skip("tree_de4e15b absent")
    r = C.check_provenance()
    assert r["ok"] and len(r["files"]) == len(C.LOAD_BEARING)
    assert r["markers"]["stack/tanitad/refs/refc_v3.py"]["traj_precap"] >= 1
    assert all(len(v["git_blob"]) == 40 and v["lf_equal"] for v in r["files"].values())


def test_provenance_fake_tree_match_and_each_refusal(tmp_path):
    t = _fake_tree(tmp_path)
    rd = lambda p: b"line\ncap_plan_to_limit\n"                      # noqa: E731  (LF blob vs CRLF tree -> equal after normalisation)
    sh = lambda p: "a" * 40                                          # noqa: E731
    ok = C.check_provenance(t, paths=("f.py",), markers={"f.py": ("cap_plan_to_limit",)}, blob_reader=rd, sha_reader=sh)
    assert ok["ok"]
    with pytest.raises(C.ProvenanceError):                           # a DIFFERENT blob
        C.check_provenance(t, paths=("f.py",), markers={}, blob_reader=lambda p: b"other\n", sha_reader=sh)
    with pytest.raises(C.ProvenanceError):                           # EMPTY git read: INCONCLUSIVE, never a match
        C.check_provenance(t, paths=("f.py",), markers={}, blob_reader=lambda p: b"", sha_reader=sh)
    with pytest.raises(C.ProvenanceError):                           # short object name: INCONCLUSIVE
        C.check_provenance(t, paths=("f.py",), markers={}, blob_reader=rd, sha_reader=lambda p: "")
    with pytest.raises(C.ProvenanceError):                           # clamp marker absent although the hash matches
        C.check_provenance(t, paths=("f.py",), markers={"f.py": ("traj_precap",)}, blob_reader=rd, sha_reader=sh)
    with pytest.raises(C.ProvenanceError):                           # file missing
        C.check_provenance(t, paths=("g.py",), markers={}, blob_reader=rd, sha_reader=sh)
    (t / "PROVENANCE.txt").write_text("tree: git archive 0000000000000000000000000000000000000000\n", encoding="utf-8")
    with pytest.raises(C.ProvenanceError):                           # PROVENANCE.txt names another commit
        C.check_provenance(t, paths=("f.py",), markers={}, blob_reader=rd, sha_reader=sh)
    (t / "PROVENANCE.txt").unlink()
    with pytest.raises(C.ProvenanceError):
        C.check_provenance(t, paths=("f.py",), markers={}, blob_reader=rd, sha_reader=sh)


# ------------------------------------------------------------------------------------------------ K2
def test_k2_bit_equal_prefix_passes_and_ignores_window_list_dependent_keys():
    b = synth_banked()
    new = take_prefix(b, 4)
    new["nav_shuffled.traj"] = new["nav_shuffled.traj"] + 1.0           # a prefix roll legitimately feeds other shuffled tokens
    r = K.k2_compare(b, new)
    assert r["verdict"] == "PASS" and r["n_episodes"] == 4 and r["secondary_all_equal"]
    assert "nav_shuffled.traj" in r["window_list_dependent_excluded"] and not r["window_list_dependent_excluded"]["nav_shuffled.traj"]["equal"]


def test_k2_deliberate_mismatch_goes_red():
    b = synth_banked()
    new = take_prefix(b, 4)
    t = new["nav_true.traj"]
    t[5, 3, 0] = np.nextafter(t[5, 3, 0], np.float32(np.inf))             # ONE ULP in ONE waypoint of ONE row
    r = K.k2_compare(b, new)
    assert r["verdict"] == "FAIL" and r["primary_result"]["n_rows_unequal"] == 1


def test_k2_secondary_mismatch_is_reported_not_hidden_and_inconclusive_cases():
    b = synth_banked()
    new = take_prefix(b, 4)
    new["nav_true.sel_idx"][0] += 1
    r = K.k2_compare(b, new)
    assert r["verdict"] == "PASS" and not r["secondary_all_equal"] and r["secondary_unequal_keys"] == ["nav_true.sel_idx"]
    assert K.k2_compare(b, take_prefix(b, 2))["verdict"] == "INCONCLUSIVE"                # < 3 episodes
    extra = take_prefix(b, 4)
    extra["ws"][0] = 999                                                                  # a window the banked dump does not have
    assert K.k2_compare(b, extra)["verdict"] == "INCONCLUSIVE"
    nok = take_prefix(b, 4)
    del nok["nav_true.traj"]
    assert K.k2_compare(b, nok)["verdict"] == "INCONCLUSIVE"


def test_k2_row_order_is_by_key_not_position():
    b = synth_banked()
    new = take_prefix(b, 4)
    perm = np.random.default_rng(1).permutation(len(new["ws"]))
    new = {k: v[perm] for k, v in new.items()}
    assert K.k2_compare(b, new)["verdict"] == "PASS"


# ------------------------------------------------------------------------------------------------ K4: an independent, analytic clamp
HZ = [5, 10, 15, 20, 30, 40, 50, 60]
TICK = 0.1
BRAKE = 2.5


def analytic_clamp(plan, lim, v0, horizons=HZ, tick=TICK, brake=BRAKE):
    """Authored from the docstring: straight plans along +x only.  progress_i = min(v_plan_i, allowed_i) * dt_i,  allowed_i = max(lim, v0 - brake * t_mid_i)."""
    t = np.array([0.0] + [h * tick for h in horizons])
    dt = np.diff(t)
    tmid = t[:-1] + 0.5 * dt
    out = np.zeros_like(plan)
    binding = np.zeros(len(plan), bool)
    for r in range(len(plan)):
        x = 0.0
        prev = 0.0
        for i in range(len(dt)):
            vp = (plan[r, i, 0] - prev) / dt[i]
            allowed = max(float(lim[r]), float(v0[r]) - brake * tmid[i])
            if vp > allowed + 1e-9:
                binding[r] = True
            x += min(vp, allowed) * dt[i]
            out[r, i, 0] = x
            prev = plan[r, i, 0]
    return np.where(binding[:, None, None], out, plan).astype(np.float32), binding


def straight_extras(n_ep=4, per=6, seed=3):
    rng = np.random.default_rng(seed)
    ex = synth_banked(n_ep, per, seed)
    n = len(ex["ws"])
    t = np.array([h * TICK for h in HZ])
    vp = rng.uniform(4, 16, n).astype(np.float32)
    plan = np.zeros((n, 8, 2), np.float32)
    plan[:, :, 0] = vp[:, None] * t[None, :]
    for c in ("nav_true", "nav_shuffled", "nav_flipped", "nav_zero"):
        ex[f"{c}.traj"] = plan.copy()
    ex["v_max_ms"] = np.full(n, 8.333333, np.float32)
    ex["v0"] = rng.uniform(2, 14, n).astype(np.float32)
    return ex


CEIL = lambda vm, valid: np.where(valid > 0.5, vm, np.inf).astype(np.float32)        # noqa: E731  (independent of the bin ladder: the synthetic limit IS the fed value)


def make_on(off, clamp_fn, lim, perturb=None):
    on = {k: v.copy() for k, v in off.items()}
    v0 = off["v0"]
    for c in K.K4_CONDS:
        out, bnd = clamp_fn(off[f"{c}.traj"], lim, v0)
        on[f"{c}.traj"] = out
        on[f"{c}.traj_precap"] = off[f"{c}.traj"].copy()
        on[f"{c}.plan_cap_binding"] = bnd.astype(np.float32)
        on[f"{c}.plan_v_lim_ms"] = lim.copy()
    if perturb:
        perturb(on)
    return on


def _clamp_np(plan, lim, v0):
    return analytic_clamp(plan, lim, v0)


def test_analytic_reference_binds_some_rows_but_not_all():
    off = straight_extras()
    _, b = analytic_clamp(off["nav_true.traj"], CEIL(off["v_max_ms"], off["v_max_valid"]), off["v0"])
    assert 3 <= b.sum() < len(b)


def test_k4_equal_passes_and_every_deliberate_regression_goes_red():
    off = straight_extras()
    lim = CEIL(off["v_max_ms"], off["v_max_valid"])
    on = make_on(off, _clamp_np, lim)
    r = K.k4_compare(off, on, ceiling_fn=CEIL, clamp_fn=_clamp_np)
    assert r["verdict"] == "PASS" and r["n_binding_rows_nav_true"] >= K.MIN_BINDING and r["aux_all_true"]
    # (1) one binding waypoint off by 1e-3 m
    b0 = int(np.nonzero(analytic_clamp(off["nav_true.traj"], lim, off["v0"])[1])[0][0])

    def bump(o):
        o["nav_true.traj"][b0, 7, 0] += 1e-3
    assert K.k4_compare(off, make_on(off, _clamp_np, lim, bump), ceiling_fn=CEIL, clamp_fn=_clamp_np)["verdict"] == "FAIL"
    # (2) below the tolerance: 5e-7 m on a small waypoint must still PASS (tolerance is 1e-6, not 0)
    def tiny(o):
        o["nav_true.traj"][b0, 0, 0] += 5e-7
    assert K.k4_compare(off, make_on(off, _clamp_np, lim, tiny), ceiling_fn=CEIL, clamp_fn=_clamp_np)["verdict"] == "PASS"
    # (3) the forward did NOT clamp (ON == OFF): must go RED, the real failure the flag-wiring could have
    unclamped = {k: v.copy() for k, v in off.items()}
    assert K.k4_compare(off, unclamped, ceiling_fn=CEIL, clamp_fn=_clamp_np)["verdict"] == "FAIL"
    # (4) a wrong limit in the forward: the aux limit identity reads False while K4 itself still reads the arithmetic
    def badlim(o):
        o["nav_true.plan_v_lim_ms"][:] = 99.0
    r4 = K.k4_compare(off, make_on(off, _clamp_np, lim, badlim), ceiling_fn=CEIL, clamp_fn=_clamp_np)
    assert r4["aux"]["K4b_forward_limit_equals_fed_ceiling"]["nav_true"] is False and not r4["aux_all_true"]
    # (5) a condition's plan clamped but another not: the per-condition max catches it
    def one_cond(o):
        o["nav_zero.traj"] = off["nav_zero.traj"].copy()
    assert K.k4_compare(off, make_on(off, _clamp_np, lim, one_cond), ceiling_fn=CEIL, clamp_fn=_clamp_np)["verdict"] == "FAIL"


def test_k4_vacuous_when_nothing_binds_and_when_inputs_differ():
    off = straight_extras()
    off["v0"][:] = 0.0
    for c in K.K4_CONDS:
        off[f"{c}.traj"][:, :, 0] = np.linspace(0.5, 3.0, 8)[None, :] * 1.0   # tiny speeds: nothing can bind
    lim = CEIL(off["v_max_ms"], off["v_max_valid"])
    on = make_on(off, _clamp_np, lim)
    r = K.k4_compare(off, on, ceiling_fn=CEIL, clamp_fn=_clamp_np)
    assert r["verdict"] == "INCONCLUSIVE" and "nothing" in r["why"]
    off2 = straight_extras()
    on2 = make_on(off2, _clamp_np, CEIL(off2["v_max_ms"], off2["v_max_valid"]))
    on2["v0"] = on2["v0"] + 1.0
    assert K.k4_compare(off2, on2, ceiling_fn=CEIL, clamp_fn=_clamp_np)["verdict"] == "INCONCLUSIVE"


def _bump_ulps(n):
    def f(o, b0):
        x = o["nav_true.traj"][b0, 7, 0]
        for _ in range(n):
            x = np.nextafter(x, np.float32(np.inf))
        o["nav_true.traj"][b0, 7, 0] = x
    return f


def test_k4_ulp_band_is_the_registered_criterion():
    """Master Mind 2026-10-18 ~20:40, before any K4 data: PASS iff |d| <= 1e-6 m OR <= ULP_TOL (4) float32 ulp on all rows.
    Literal expectations, both sides of the band, plus a centimetre-scale (logic-level) mutation that must FAIL."""
    assert K.ULP_TOL == 4.0 and K.TOL_M == 1e-6
    off = straight_extras()
    lim = CEIL(off["v_max_ms"], off["v_max_valid"])
    b0 = int(np.nonzero(analytic_clamp(off["nav_true.traj"], lim, off["v0"])[1])[0][0])
    assert abs(float(off["nav_true.traj"][b0, 7, 0])) > 8.0        # beyond ~8 m one ulp exceeds 1e-6 m: the case the band exists for

    r1 = K.k4_compare(off, make_on(off, _clamp_np, lim, lambda o: _bump_ulps(1)(o, b0)), ceiling_fn=CEIL, clamp_fn=_clamp_np)
    assert r1["by_cond"]["nav_true"]["max_ulp_on_binding_rows"] == pytest.approx(1.0, abs=1e-6)
    assert r1["by_cond"]["nav_true"]["max_abs_m"] > 1e-6
    assert r1["verdict"] == "PASS" and r1["ulp_tol"] == 4.0

    r8 = K.k4_compare(off, make_on(off, _clamp_np, lim, lambda o: _bump_ulps(8)(o, b0)), ceiling_fn=CEIL, clamp_fn=_clamp_np)
    assert r8["by_cond"]["nav_true"]["max_ulp_on_binding_rows"] == pytest.approx(8.0, abs=1e-6)
    assert r8["verdict"] == "FAIL" and "not device rounding" in r8["why"]

    def one_cm(o):
        o["nav_true.traj"][b0, 7, 0] = o["nav_true.traj"][b0, 7, 0] + np.float32(0.01)
    rc = K.k4_compare(off, make_on(off, _clamp_np, lim, one_cm), ceiling_fn=CEIL, clamp_fn=_clamp_np)
    assert rc["verdict"] == "FAIL"


# ------------------------------------------------------------------------------------------------ K4 against the REAL vendored function
@pytest.fixture(scope="module")
def vendored():
    if not (TREE / "stack").exists() or not Path("C:/Users/Admin/tanitad-push/.git").exists():
        pytest.skip("clamp tree / git dir absent")
    sys.path.insert(0, str(TREE / "stack"))
    sys.path.insert(0, str(TREE / "taniteval"))
    try:
        return K.default_posthoc("cpu")
    except Exception as exc:                                              # noqa: BLE001
        pytest.skip(f"vendored function not loadable here: {type(exc).__name__}: {exc}")


def test_vendored_function_matches_the_independent_analytic_clamp(vendored):
    ceiling_fn, clamp_fn, rec = vendored
    off = straight_extras()
    lim = CEIL(off["v_max_ms"], off["v_max_valid"])
    exp, eb = analytic_clamp(off["nav_true.traj"], lim, off["v0"])
    got, gb = clamp_fn(off["nav_true.traj"], lim, off["v0"])
    assert rec["horizons"] == HZ and rec["tick_s"] == TICK                  # the config the analytic expectation assumed
    assert (eb == gb).all() and eb.sum() >= 3
    assert float(np.abs(exp - got).max()) < 2e-4                              # different arithmetic, same semantics (float32 cumsum vs python sum)
    assert float(np.abs(got[~gb] - off["nav_true.traj"][~gb]).max()) == 0.0   # non-binding rows come back BIT-identical
    # deliberate regression: an analytic clamp with the brake allowance DROPPED (strict ceiling) must DIFFER from the vendored function on rows
    # whose v0 is above the limit -- i.e. this test can tell the two semantics apart
    strict, _ = analytic_clamp(off["nav_true.traj"], lim, off["v0"], brake=0.0)
    assert float(np.abs(strict - got).max()) > 1e-2


def test_k4_end_to_end_with_the_vendored_function_as_both_sides(vendored):
    ceiling_fn, clamp_fn, _ = vendored
    off = straight_extras()
    lim = ceiling_fn(off["v_max_ms"], off["v_max_valid"])
    on = make_on(off, clamp_fn, lim)
    r = K.k4_compare(off, on, ceiling_fn=ceiling_fn, clamp_fn=clamp_fn)
    assert r["verdict"] == "PASS" and r["max_abs_m_all"] == 0.0 and r["aux_all_true"]


# ------------------------------------------------------------------------------------------------ the prefix planner on the real banked tree
def test_plan_prefix_real_banked_tree_is_a_dump_order_prefix_with_a_proven_clip_map():
    if not BANKED.exists() or not S2W.exists():
        pytest.skip("banked tree absent")
    sub, rec = K.plan_prefix(str(BANKED), str(S2W), n_episodes=3, posthoc=None)
    assert rec["n_episodes"] == 3 and len(sub) == 3
    allw = json.loads(S2W.read_text(encoding="utf-8"))
    for cid, ws in sub.items():
        assert allw[cid] == ws                                                # identical to the grid: no window dropped or invented
    assert [e["clip_index"] for e in rec["episodes"]] == [0, 1, 2]
    assert rec["episodes"][0]["sha12"] == "0dfe63214bc2"


def test_plan_prefix_refuses_an_unprovable_clip_map(tmp_path):
    if not BANKED.exists():
        pytest.skip("banked tree absent")
    import shutil
    d = tmp_path / "t" / "dump_s0"
    d.mkdir(parents=True)
    for f in ("refcv6_extras.npz", "manifest.json"):
        shutil.copy(BANKED / "dump_s0" / f, d / f)
    j = json.loads((BANKED / "dump_s0" / "refcv6_extras.npz.json").read_text(encoding="utf-8"))
    j["clip_sha12_by_index"]["1"] = "deadbeef0000"                            # tamper: index 1 no longer maps to its clip
    (d / "refcv6_extras.npz.json").write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(SystemExit):
        K.plan_prefix(str(tmp_path / "t"), str(S2W), n_episodes=3, posthoc=None)


# ------------------------------------------------------------------------------------------------ the model hooks + the wrapper's pure parts
def _fake_model(max_speed=True, hier=True, r7sel=False, clamp=False, training=False):
    torch = pytest.importorskip("torch")

    class Cfg:
        pass

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = Cfg()
            self.cfg.max_speed_onehot_v6, self.cfg.hier, self.cfg.refcv7_select, self.cfg.plan_vmax_clamp = max_speed, hier, r7sel, clamp
            self.emit = {}

        def forward(self, x):
            return dict(self.emit)

    m = M()
    m.train(training)
    return m


def test_patch_on_refuses_each_invalid_build_and_sets_the_flag():
    torch = pytest.importorskip("torch")
    import r8clamp_patch as P
    m = _fake_model()
    rec = P._enable_or_assert_off(m, m.cfg, "on")
    assert m.cfg.plan_vmax_clamp is True and rec["cfg_plan_vmax_clamp_now"]
    for kw in ({"max_speed": False}, {"hier": False}, {"r7sel": True}, {"training": True}):
        with pytest.raises(RuntimeError):
            P._enable_or_assert_off(_fake_model(**kw), None, "on")
    with pytest.raises(RuntimeError):
        P._enable_or_assert_off(_fake_model(clamp=True), None, "off")        # OFF mode with the flag already True in the BUILT cfg


def test_forward_hook_on_requires_the_clamp_keys_and_a_live_limit():
    torch = pytest.importorskip("torch")
    import r8clamp_patch as P
    P.STATS.update(n_forwards=0, n_rows=0, n_rows_finite_limit=0, n_binding=0, n_v0_over_limit=0)
    m = _fake_model()
    P._enable_or_assert_off(m, m.cfg, "on")
    x = torch.zeros(1)
    m.emit = {"traj": torch.zeros(2, 8, 2)}
    with pytest.raises(RuntimeError, match="did NOT emit"):
        m(x)                                                                    # ON but the forward emitted no clamp keys -> REFUSE
    m.emit = {"traj_precap": torch.zeros(2, 8, 2), "plan_cap_binding": torch.tensor([True, False]), "plan_v_lim_ms": torch.tensor([8.3, float("inf")])}
    m(x)
    assert P.STATS["n_binding"] == 1 and P.STATS["n_rows_finite_limit"] == 1
    P.STATS.update(n_rows=0, n_rows_finite_limit=0)
    m.emit = {"traj_precap": torch.zeros(40, 8, 2), "plan_cap_binding": torch.zeros(40, dtype=torch.bool), "plan_v_lim_ms": torch.full((40,), float("inf"))}
    with pytest.raises(RuntimeError, match="INERT"):
        m(x)                                                                    # a limit that is +inf everywhere = built and inert


def test_forward_hook_off_forbids_the_clamp_keys():
    torch = pytest.importorskip("torch")
    import r8clamp_patch as P
    m = _fake_model()
    P._enable_or_assert_off(m, m.cfg, "off")
    m.emit = {"traj": torch.zeros(1, 8, 2)}
    m(torch.zeros(1))                                                           # fine: no clamp keys
    m.emit = {"traj": torch.zeros(1, 8, 2), "traj_precap": torch.zeros(1, 8, 2), "plan_cap_binding": torch.tensor([False])}
    with pytest.raises(RuntimeError, match="OFF mode"):
        m(torch.zeros(1))


def test_split_clamp_requires_mode_and_a_matching_tag():
    import run_battery_r8clamp as R
    assert R.split_clamp(["--clamp", "on", "--tag", "s_clampon_unc", "--x", "1"]) == ("on", ["--tag", "s_clampon_unc", "--x", "1"])
    assert R.split_clamp(["--clamp=off", "--tag", "s_clampoff"])[0] == "off"
    for bad in (["--tag", "t_clampon"], ["--clamp", "on", "--tag", "t_clampoff"], ["--clamp", "off", "--tag", "t_clampon"], ["--clamp", "on"],
                ["--clamp", "maybe", "--tag", "x"], ["--clamp", "on", "--tag", "plain"]):
        with pytest.raises(SystemExit):
            R.split_clamp(bad)


# ------------------------------------------------------------------------------------------------ the install, in a FRESH process, on the real tree
def test_install_in_a_fresh_process_points_tanitad_at_the_clamp_tree_and_hooks_the_loader():
    if not TREE.exists() or not C.BATTERY_CODE.exists():
        pytest.skip("clamp tree / battery code absent")
    code = (
        "import os, sys, json\n"
        f"sys.path.insert(0, r'{HERE}')\n"
        "import r8clamp_patch as P\n"
        "P.install('on')\n"
        "import r8_repoint, r8_battery_patch as BP\n"
        "L = r8_repoint.apply()\n"
        "import tanitad\n"
        "ok = {'tanitad': tanitad.__file__, 'loader': L.__file__, 'repo': str(L.REPO),\n"
        "      'build_model_hooked': bool(getattr(L.build_model, '_r8clamp_build', False)),\n"
        "      'eval_ds_hooked': bool(getattr(L.build_eval_dataset, '_r8clamp_pose', False)),\n"
        "      'extra_keys': list(__import__('r7_roll').EXTRA_ROW_KEYS)[-7:],\n"
        "      'child': BP.rewrite_child_cmd(['python', r'C:/x/roll_seed_r8.py', '--seed', '0'])[1],\n"
        "      'battery_code': str(r8_repoint.BATTERY_CODE), 'pose_policy_env': os.environ.get('R8_NO_POSE_SYNC')}\n"
        "print('ZZOK' + json.dumps(ok) + 'OKZZ')\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8", R8_NO_POSE_SYNC="1")
    env.pop("R8_LAUNCH_TREE", None)
    env["PYTHONPATH"] = f"{C.TREE}/stack;{C.TREE}/taniteval"
    p = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True, env=env, timeout=600)
    out = p.stdout
    assert "ZZOK" in out, p.stderr[-2000:]
    ok = json.loads(out.split("ZZOK")[1].split("OKZZ")[0])
    norm = lambda s: s.replace("\\", "/").lower()                            # noqa: E731
    assert norm(ok["tanitad"]).startswith(norm(str(C.TREE))) and norm(ok["loader"]).startswith(norm(str(C.TREE)))
    assert ok["build_model_hooked"] and ok["eval_ds_hooked"]
    assert ok["extra_keys"] == list(C.CLAMP_KEYS)
    assert norm(ok["child"]).endswith("roll_seed_r8clamp.py")
    assert norm(ok["battery_code"]) == norm(str(C.BATTERY_CODE))


def test_install_refuses_when_the_tree_fails_provenance(tmp_path):
    if not C.BATTERY_CODE.exists():
        pytest.skip("battery code absent")
    t = tmp_path / "tree"
    t.mkdir()
    code = f"import sys; sys.path.insert(0, r'{HERE}'); import r8clamp_patch as P; P.install('off')\n"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", R8_CLAMP_TREE=str(t))
    p = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode != 0 and "REFUSING" in (p.stderr + p.stdout)


# ------------------------------------------------------------------------------------------------ the pose-clock policy (the de4e15b loader enables the shift ITSELF)
def _fake_loader(shift_when_sidecar: bool):
    """A stand-in for the de4e15b loader: build_eval_dataset attaches the pose shift whenever args names a sidecar (refcv7_loader.py:801-802)."""
    import types
    seen = {}

    class DS:
        pose_sync = None

    def build_eval_dataset(model, cfg, args, config, **kw):
        seen["sidecar_during_build"] = getattr(args, "pose_sync_sidecar", None)
        ds = DS()
        if shift_when_sidecar and getattr(args, "pose_sync_sidecar", None):
            ds.pose_sync = object()
        return ds, [], {}

    L = types.SimpleNamespace(__file__=str(TREE / "stack/tanitad/eval/refcv7_loader.py"), build_eval_dataset=build_eval_dataset,
                              build_model=lambda *a, **k: (_fake_model(), None, None, {}))
    return L, seen


def test_pose_clock_standalone_blanks_the_sidecar_and_restores_it(monkeypatch):
    import types
    import r8clamp_patch as P
    monkeypatch.setattr(P, "_under", lambda *a: True)
    L, seen = _fake_loader(shift_when_sidecar=True)
    P._post_loader(L, False, "off")                                      # R8_NO_POSE_SYNC=1: the UNCORRECTED clock the banked _unc trees were rolled on
    args = types.SimpleNamespace(pose_sync_sidecar="E:/x/sidecar.jsonl")
    ds, _, _ = L.build_eval_dataset(None, None, args, {})
    assert seen["sidecar_during_build"] is None and ds.pose_sync is None      # not shifted although the argv names a sidecar
    assert args.pose_sync_sidecar == "E:/x/sidecar.jsonl"                      # restored for every later reader of args
    assert P._STATE["pose_clock"]["shift_attached"] is False


def test_pose_clock_corrected_requires_the_shift_and_standalone_refuses_a_shifted_dataset(monkeypatch):
    import types
    import r8clamp_patch as P
    monkeypatch.setattr(P, "_under", lambda *a: True)
    args = types.SimpleNamespace(pose_sync_sidecar="E:/x/sidecar.jsonl")
    L, _ = _fake_loader(shift_when_sidecar=True)
    P._post_loader(L, True, "off")                                       # corrected clock: the shift must be attached
    assert L.build_eval_dataset(None, None, args, {})[0].pose_sync is not None
    L2, _ = _fake_loader(shift_when_sidecar=False)                       # a loader that FAILS to shift on the corrected clock -> refuse
    P._post_loader(L2, True, "off")
    with pytest.raises(SystemExit, match="no pose shift"):
        L2.build_eval_dataset(None, None, args, {})
    # standalone clock but a loader whose dataset is shifted EVEN with the sidecar blanked: the standalone assertion must fire
    import types as _t

    class DS:
        pose_sync = object()
    L4 = _t.SimpleNamespace(__file__=L.__file__, build_eval_dataset=lambda m, c, a, cfg, **k: (DS(), [], {}), build_model=lambda *a, **k: None)
    P._post_loader(L4, False, "off")
    with pytest.raises(SystemExit, match="HAS the pose shift"):
        L4.build_eval_dataset(None, None, args, {})
