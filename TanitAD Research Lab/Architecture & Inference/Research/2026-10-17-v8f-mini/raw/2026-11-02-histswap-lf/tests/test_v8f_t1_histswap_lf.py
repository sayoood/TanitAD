"""HISTSWAP-LF (H-HISTSWAP-LF-1): ``--histswap {off,logged,cv_exec,cv_logged}`` of ``stack/scripts/v8f_t1_eval_histswap_lf.py`` (the HISTSWAP-T1 tool 7751673d + the LF patch).

Run (from a tree that holds stack/ and taniteval/, the tool in stack/scripts/, this file in stack/tests/):
  CUDA_VISIBLE_DEVICES=-1  V8F_T1_HISTSWAP_TOOL=<the HISTSWAP-T1 tool, sha256 7751673d...>  pytest -q stack/tests/test_v8f_t1_histswap_lf.py
  * C-OFF    ``--histswap off`` and the flag absent give byte-identical dumps / ticks / header, and ``off`` and ``logged`` equal the HISTSWAP tool's (sha256 pinned; skipped without ``V8F_T1_HISTSWAP_TOOL``).
  * CV       under ``cv_exec`` / ``cv_logged`` every ego-history key the network reads is the CONSTANT-TWIST history at the stated speed: p10f speed rows == v, ACCELERATION rows == 0, curvature rows == the
             yaw rate over max(v, 2) (all spelled here independently), ``base`` == kd_x of (v, 0, kappa), pose_now == the EXECUTED pose, pose_t / pose_s lie on the arc v x age x 0.2 s behind it, the causal
             bucket == the bucket of v alone. ``v`` is the EXECUTED p10f[j][0] (cv_exec) or the LOGGED p10f[j][0] (cv_logged); the yaw rate is the EXECUTED one in both. Non-vacuous: the executed history has
             acceleration; the logged speed differs from the executed one after tick 0.
  * SPELL    ``channels_from_track`` equals, exactly, the rows ``WindowState._finish`` writes on a real executed slice; ``cv_motion_track`` has the stated twist.
  * EXEC     the plan is committed to the EXECUTED state (as HISTSWAP's EXEC).
  * TICKS    ``v0`` is the executed speed; ``v0_in`` == ``v0`` at EVERY tick under cv_exec, and at tick 0 only under cv_logged (differs after); the P10 floor is unchanged; refusals (--feasible, --limit).
  * MUTATION cv_exec reading the LOGGED speed, a ramp injected into the constant-speed track, and the item builder on the executed dict each turn CV RED.
"""

import dataclasses
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_t1_eval_histswap_lf as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
HS_TOOL_SHA256 = "7751673d17fcff763a940c1e49ba50038d216d0049482582cca2aec7c28af63f"
M_STEPS = 15
SWAP_KEYS = ("p10f", "base", "pose_now", "pose_t", "pose_s")
CV = ("cv_exec", "cv_logged")



@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_histswap")
    w = D.make_synthetic_world(root, n_train=2, n_eval=3, T=90, seed=5)
    return {"root": root, **w}


def tiny_model(J, seed=0):
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():
        for q in prior.parameters():
            q.mul_(0.02)
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.head_o[2].weight, std=0.3)               # a rough O head: the plan wanders, the prior does not
    return model, cfg


def tool_from_source(src: str, name: str):
    mod = types.ModuleType(name)
    mod.__file__ = str(Path(T1.__file__).resolve())
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def mutated(old: str, new: str):
    src = Path(T1.__file__).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert src.count(old) == 1, (old, src.count(old))
    return tool_from_source(src.replace(old, new), "v8f_t1_eval_histswap_mut")



def load_hs_tool():
    p = os.environ.get("V8F_T1_HISTSWAP_TOOL", "")
    if not p or not Path(p).is_file():
        pytest.skip("V8F_T1_HISTSWAP_TOOL=<the HISTSWAP-T1 tool 7751673d> not given")
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == HS_TOOL_SHA256, "REFUSED: that file is not the HISTSWAP-T1 tool (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_histswap_hs")


def run_cli(world, tmp, label, *extra, tool=None, model=None, tag="syn", m=M_STEPS):
    root = Path(world["root"])
    if model is not None:
        torch.save({"cfg": dataclasses.asdict(model[1]), "model": model[0].state_dict(), "grid": "pool4x10"}, tmp / "ck.pt")
    t = tool or T1
    t.CAUSAL_STATS.clear()
    rc = t.main(["run", "--ckpt", str(tmp / "ck.pt"), "--tag", tag, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root),
                 "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(tmp / label),
                 "--replan-steps", str(m), "--batch", "6", "--max-windows", "12", "--n-boot", "20", "--device", "cpu",
                 "--dump-ticks", str(tmp / label / "ticks"), *extra])
    assert rc == 0
    return tmp / label


def dumps(out, key=None, tag="syn"):
    fs = sorted((out / f"dump_{tag}").iterdir())
    if key is not None:
        return np.concatenate([np.load(f)[key] for f in fs])
    return {f.name: {k: np.load(f)[k] for k in np.load(f).files} for f in fs}


def ticks(out, tag="syn"):
    z = np.load(out / "ticks" / f"ticks_{tag}.npz")
    return {k: z[k] for k in z.files}


def header(out, tag="syn"):
    return json.load(open(out / f"t1v8f_{tag}.json"))["t1v8f"]


def assert_same_run(a, b):
    da, db = dumps(a), dumps(b)
    assert da.keys() == db.keys() and len(da) >= 3
    for f in da:
        assert set(da[f]) == set(db[f]) and all(da[f][k].tobytes() == db[f][k].tobytes() and da[f][k].dtype == db[f][k].dtype for k in da[f]), f
    ta, tb = ticks(a), ticks(b)
    assert ta.keys() == tb.keys() and all(ta[k].tobytes() == tb[k].tobytes() for k in ta)
    ja, jb = header(a), header(b)
    assert "histswap" not in ja and "histswap" not in jb
    assert {k: v for k, v in ja.items() if k != "ticks_file"} == {k: v for k, v in jb.items() if k != "ticks_file"}


def assert_same_arrays(a, b):
    da, db = dumps(a), dumps(b)
    assert da.keys() == db.keys() and len(da) >= 3
    for f in da:
        assert set(da[f]) == set(db[f]) and all(da[f][k].tobytes() == db[f][k].tobytes() and da[f][k].dtype == db[f][k].dtype for k in da[f]), f
    ta, tb = ticks(a), ticks(b)
    assert ta.keys() == tb.keys() and all(ta[k].tobytes() == tb[k].tobytes() for k in ta)
    ja, jb = header(a), header(b)
    assert {k: v for k, v in ja.items() if k != "ticks_file"} == {k: v for k, v in jb.items() if k != "ticks_file"}


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    J = D.load_join(Path(world["root"]) / "join_eval.npz")
    model = tiny_model(J)
    base = ("--smooth", "sg72", "--limit", "causal", "--dump-heads")
    out = {"tmp": tmp, "base": base, "y": run_cli(world, tmp, "y", *base, model=model)}      # the flag absent
    out["off"] = run_cli(world, tmp, "off", *base, "--histswap", "off")
    out["logged"] = run_cli(world, tmp, "logged", *base, "--histswap", "logged")
    out["hcv"] = run_cli(world, tmp, "hcv", *base, "--histswap", "cv_exec")
    out["hcv_stats"] = dict(T1.CAUSAL_STATS)
    out["hlv"] = run_cli(world, tmp, "hlv", *base, "--histswap", "cv_logged")
    out["hlv_stats"] = dict(T1.CAUSAL_STATS)
    return out


# ------------------------------------------------------------------------------------------------ flag, refusals, C-OFF
def test_flag_parser_default_refusals_and_mode_parser(world, tmp_path):
    base = ["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]
    assert T1.build_parser().parse_args(base).histswap == "off"
    for m in ("logged", *CV):
        assert T1.build_parser().parse_args(base + ["--histswap", m]).histswap == m
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(base + ["--histswap", "cv"])
    for m in ("logged", *CV):
        with pytest.raises(SystemExit) as e:
            T1.main(base + ["--histswap", m, "--limit", "causal", "--feasible", "kbox"])
        assert "REFUSED" in str(e.value)
    for m in CV:
        for lim in ("logged", "off", "causal_all"):
            with pytest.raises(SystemExit) as e:
                T1.main(base + ["--histswap", m, "--limit", lim])
            assert "REFUSED" in str(e.value) and "--limit causal" in str(e.value)
    assert T1.HISTSWAP_MODES == ("off", "logged", "cv_exec", "cv_logged") and T1.CV_MODES == CV
    assert [T1.hist_mode_of(x) for x in (False, None, "off", True, "logged", "cv_exec", "cv_logged")] == ["off", "off", "off", "logged", "logged", "cv_exec", "cv_logged"]
    with pytest.raises(ValueError):
        T1.hist_mode_of("executed")


def test_c_off_flag_absent_equals_off_and_both_equal_the_histswap_tool(world, runs):
    assert_same_run(runs["y"], runs["off"])
    hs = load_hs_tool()
    assert_same_run(runs["y"], run_cli(world, runs["tmp"], "hs_off", *runs["base"], tool=hs))
    assert_same_arrays(runs["logged"], run_cli(world, runs["tmp"], "hs_logged", *runs["base"], "--histswap", "logged", tool=hs))


# ------------------------------------------------------------------------------------------------ CV (direct rollout, tool under test passed in)
def capture(tool, world, tmp, hist_swap, limit="causal", m=M_STEPS):
    root = Path(world["root"])
    ck = torch.load(tmp / "ck.pt", map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = D.load_native_tracks(str(root / "native_tracks.pt"))
    J = D.load_join(str(root / "join_eval.npz"))
    store = D.ClipStore(str(root), "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(str(root / "eval_grid_windows.json"), J)[:12]
    proxy = tool._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    states = tool.make_states(store, tracks, J, wins, "deadreckon", hist_swap=hist_swap)
    cap = []

    def hook(n, batch, plan, sts):
        for b, st in enumerate(sts):
            cap.append({"n": n, "j": st.j, "t0": st.t0, "clip": str(J["clip"][st.ci]), "plan": np.array(plan[b], np.float64),
                        "pose_exec": np.array(st.d["pose"][st.j], np.float64), "v_exec": float(st.d["p10f"][st.j][0]),
                        "om_exec": float(np.nan_to_num(st.d["ch1"][st.j])), "kin3_exec": np.array(st.d["kin3"][st.j], np.float64),
                        "v_log": float(store.get(str(J["clip"][st.ci]))["p10f"][st.j][0]), "p10f_exec": np.array(st.d["p10f"][max(1, st.j - 9):st.j + 1], np.float64),
                        "batch": {k: batch[k][b].detach().cpu().numpy().copy() for k in (*SWAP_KEYS, "limit", "limit_ok", "a_t", "a_s")}})

    tool.CAUSAL_STATS.clear()
    planner = tool.model_planner(model, torch.device("cpu"))
    tool.rollout(planner, states, ds, proxy, m=m, schedule="periodic", scene="logged", limit=limit, device=torch.device("cpu"), hook=hook)
    cl = [st.executed_ego() for st in states]
    return {"cap": cap, "store": store, "states": states, "cl": np.asarray(cl), "stats": dict(tool.CAUSAL_STATS), "J": J, "ds": ds}


def exec_ok(c, m=M_STEPS):
    """EXEC: the tick-1 executed knots (t0 frame, from ``cl``) are the tick-1 plan composed with the tick-1 EXECUTED pose (the ticks file's), recomputed here."""
    by = {}
    for r in c["cap"]:
        by.setdefault((r["clip"], r["t0"]), {})[r["n"]] = r
    worst = 0.0
    for i, st in enumerate(c["states"]):
        rr = by[(str(c["J"]["clip"][st.ci]), st.t0)]
        p0, p1 = rr[0]["pose_exec"], rr[1]["pose_exec"]
        pl = rr[1]["plan"][:m]
        cs, sn = np.cos(p1[2]), np.sin(p1[2])
        w = p1[:2][None] + np.stack([pl[:, 0] * cs - pl[:, 1] * sn, pl[:, 0] * sn + pl[:, 1] * cs], -1)
        d = w - p0[:2][None]
        c0, s0 = np.cos(-p0[2]), np.sin(-p0[2])
        ego = np.stack([d[:, 0] * c0 - d[:, 1] * s0, d[:, 0] * s0 + d[:, 1] * c0], -1)
        worst = max(worst, float(np.abs(ego - c["cl"][i][m:2 * m]).max()))
    return worst


def kappa_of(om: float, v: float) -> float:
    """kinematic_prior._pose_kappa0, spelled here independently: clamp(omega / max(v, 2), +-0.3)."""
    return float(np.clip(om / max(v, 2.0), -0.3, 0.3))


def wrap(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def check_cv(c, mode: str):
    """Every ego-history key the network read at every tick is the constant-twist history at the stated speed (independent spelling); raises AssertionError otherwise."""
    hmax = c["ds"].hmax
    for r in c["cap"]:
        v = r["v_exec"] if mode == "cv_exec" else r["v_log"]
        om = r["om_exec"]
        k = kappa_of(om, v)
        b, j = r["batch"], r["j"]
        real = np.arange(j - hmax + 1, j + 1) >= 1
        p = b["p10f"][real].astype(np.float64)
        exp = np.array([v, 0.0, k, v * v, 0.0, v * k, v * v * k, 0.0])
        assert np.allclose(p, exp[None], atol=2e-5, rtol=1e-6), (mode, r["n"], r["clip"], p[-1], exp)
        assert np.all(p[:, 1] == 0.0) and np.all(p[:, 4] == 0.0) and np.all(p[:, 7] == 0.0), "the history must carry ZERO acceleration"
        bexp = D.kdx_base(np.array([[v, 0.0, k]]), D.KIN_UNITS["yaw"])[0]
        assert np.allclose(b["base"], bexp, atol=1e-3, rtol=1e-5), "base is not kd_x of (v, 0, kappa)"
        assert np.array_equal(b["pose_now"], r["pose_exec"][:3].astype(np.float32)), "the anchor must be the EXECUTED pose"
        for key, age in (("pose_t", int(b["a_t"])), ("pose_s", int(b["a_s"]))):
            T = age * 0.2
            chord = v * T if abs(om) < 1e-9 else 2.0 * abs(v / om) * abs(np.sin(om * T / 2.0))
            dxy = float(np.hypot(*(b["pose_now"][:2].astype(np.float64) - b[key][:2].astype(np.float64))))
            assert abs(dxy - chord) <= 5e-3 + 1e-4 * chord, (mode, key, age, dxy, chord)
            assert abs(wrap(float(b["pose_now"][2]) - float(b[key][2]) - om * T)) <= 2e-4, (mode, key, "heading")
        assert bool(b["limit_ok"]) and int(np.argmax(b["limit"])) == int(T1.speed_max_bin(max(v, 0.0))[0]), "the bucket must be the bucket of the stated speed alone"
        if mode == "cv_exec":
            assert abs(k - float(r["kin3_exec"][2])) <= 1e-7, "HCV keeps the executed curvature at the tick (omega is the float32 channel value)"


def test_cv_motion_track_has_the_stated_twist():
    for om in (0.0, 0.2, -0.35):
        tr = T1.cv_motion_track([10.0, -3.0, 0.7, 5.0], 6.5, om, 12)
        assert tr.shape == (25, 4) and np.array_equal(tr[-1, :3], [10.0, -3.0, 0.7])
        ch = D.ego_channels_native(tr)
        assert np.allclose(ch[1:, 0], 6.5) and np.allclose(ch[1:, 1], 0.0, atol=1e-12) and np.allclose(ch[1:, 2], om, atol=1e-9), (om, ch[1:3], ch[-1])
        chord = np.hypot(*np.diff(tr[:, :2], axis=0).T) / D.DT_NATIVE
        want = 6.5 if om == 0.0 else 2 * 6.5 / abs(om) * np.sin(abs(om) * D.DT_NATIVE / 2) / D.DT_NATIVE
        assert np.allclose(chord, want, rtol=1e-9)


def test_spell_channels_from_track_equals_what_finish_writes(world, runs):
    root = Path(world["root"])
    tracks = D.load_native_tracks(str(root / "native_tracks.pt"))
    J = D.load_join(str(root / "join_eval.npz"))
    store = D.ClipStore(str(root), "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(str(root / "eval_grid_windows.json"), J)[:3]
    for st in T1.make_states(store, tracks, J, wins, "deadreckon"):
        j0 = st.j
        plan = np.stack([np.arange(1, 31) * 0.2 * (6.0 + 0.5 * st.ci), np.arange(1, 31) ** 2 * 0.004], 1)
        st.commit_plan(plan, 10)
        ch = T1.channels_from_track(st.track[2 * j0:2 * (j0 + 10) + 1])
        for k in ("kin3", "ch1", "p10f", "base", "pose", "a", "v"):
            assert np.array_equal(st.d[k][j0 + 1:j0 + 11], np.asarray(ch[k]).astype(st.d[k].dtype)), k


def test_cv_the_network_reads_the_constant_twist_history_at_the_stated_speed(world, runs):
    sx = {}
    for mode in CV:
        c = capture(T1, world, runs["tmp"], mode)
        check_cv(c, mode)
        sx[mode] = c
        assert max(r["n"] for r in c["cap"]) >= 1
        assert max(float(np.abs(r["p10f_exec"][:, 1]).max()) for r in c["cap"]) > 0.05, "non-vacuous: the executed history carries acceleration the CV history removes"
        assert not np.array_equal(c["cl"], capture(T1, world, runs["tmp"], False)["cl"]), "non-vacuous: the CV input changed the rollout"
    x = sx["cv_logged"]
    assert all(r["v_log"] == r["v_exec"] for r in x["cap"] if r["n"] == 0), "tick 0: the executed state IS the logged one"
    assert max(abs(r["v_log"] - r["v_exec"]) for r in x["cap"] if r["n"] >= 1) > 0.05, "non-vacuous: the logged speed differs from the executed one after tick 0"
    assert x["stats"].get("swap_ticks", 0) > 0 and sx["cv_exec"]["stats"].get("swap_ticks", 0) > 0


def test_exec_the_plan_is_committed_to_the_executed_state_under_cv(world, runs):
    for mode in CV:
        assert exec_ok(capture(T1, world, runs["tmp"], mode)) < 2e-3


def test_mutations_turn_cv_red(world, runs):
    good = capture(T1, world, runs["tmp"], "cv_exec")
    check_cv(good, "cv_exec")
    for old, new, why in (('src = d if self.hist_mode == "cv_exec" else self.base_log', 'src = self.base_log', "cv_exec reading the LOGGED speed"),
                          ("np.full(n, float(v))", "np.linspace(0.8 * float(v), float(v), n)", "a speed ramp in the constant-speed track"),
                          ("    proxy.d = st.d_in\n    it = ds.item(", "    proxy.d = st.d\n    it = ds.item(", "the item builder on the executed dict")):
        mc = capture(mutated(old, new), world, runs["tmp"], "cv_exec")
        with pytest.raises(AssertionError):
            check_cv(mc, "cv_exec")


# ------------------------------------------------------------------------------------------------ TICKS and header
def test_ticks_v0_is_executed_v0_in_is_the_network_speed_and_the_floor_is_unchanged(runs):
    ty = ticks(runs["y"])
    for mode, key in (("cv_exec", "hcv"), ("cv_logged", "hlv")):
        t = ticks(runs[key])
        assert set(t) - set(ty) == {"v0_in", "histswap"} and not set(ty) - set(t) and str(t["histswap"]) == mode
        k0, k1 = (t["tick"] == 0), (t["tick"] >= 1)
        assert np.array_equal(t["v0_in"][k0], t["v0"][k0]), "tick 0: the logged speed IS the executed one"
        assert not np.array_equal(t["plan_raw"][k0], ty["plan_raw"][ty["tick"] == 0]), "non-vacuous: the CV history changed the tick-0 plan"
        if mode == "cv_exec":
            assert np.array_equal(t["v0_in"], t["v0"]), "HCV: the network reads the EXECUTED speed at every tick"
        else:
            assert (np.abs(t["v0_in"][k1].astype(np.float64) - t["v0"][k1].astype(np.float64)) > 0.05).any(), "HLVL: the network reads the LOGGED speed after tick 0"
        assert dumps(runs[key], "fl").tobytes() == dumps(runs["y"], "fl").tobytes(), "the P10 floor must not change"


def test_header_says_what_the_history_is(runs):
    h, o = header(runs["hcv"])["histswap"], header(runs["hlv"])["histswap"]
    assert h["mode"] == "cv_exec" and h["oracle"] == "no" and "EXECUTED" in h["speed_source"] and h["bucket_stats_model_ticks"]["swap_ticks"] > 0
    assert o["mode"] == "cv_logged" and o["oracle"].startswith("yes") and "LOGGED" in o["speed_source"]
