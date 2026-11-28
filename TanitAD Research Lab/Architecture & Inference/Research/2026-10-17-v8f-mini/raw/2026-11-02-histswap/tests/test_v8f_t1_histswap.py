"""HISTSWAP (H-HISTSWAP-1): ``--histswap {off,logged}`` of ``stack/scripts/v8f_t1_eval_histswap.py`` (the FEAS2-T1 tool + the HISTSWAP patch).

Run (from a tree that holds stack/ and taniteval/, the tool in stack/scripts/, this file in stack/tests/):
  CUDA_VISIBLE_DEVICES=-1  V8F_T1_FEAS2_TOOL=<the FEAS2-T1 tool, sha256 4b66cc86...>  pytest -q stack/tests/test_v8f_t1_histswap.py
  * C-OFF   ``--histswap off`` and the flag absent give byte-identical dumps, ticks and result JSON, and both equal the FEAS2-T1 tool (sha256 pinned; skipped without ``V8F_T1_FEAS2_TOOL``).
  * SWAP    under ``logged`` every ego-history key the network reads (p10f, base, pose_now / pose_t / pose_s, the causal speed bucket) EQUALS the LOGGED clip array at the tick (spelled here by direct
            indexing of the store's arrays, not by the tool's item builder), while the executed state differs from the logged one after tick 0 (non-vacuous); un-swapped states read the executed history.
  * EXEC    the plan is committed to the EXECUTED state: the executed knots of tick 1 are the tick-1 plan composed with the tick-1 EXECUTED pose (recomputed here from the ticks file).
  * TICKS   tick 0 equals the un-swapped run bit for bit; tick 1's executed ``pose`` and ``v0`` equal the un-swapped run's (same tick-0 plan); ``v0`` is the executed speed and ``v0_in`` the logged one; the
            P10 floor is unchanged; the flag refuses to combine with ``--feasible``.
  * MUTATION  feeding the executed dict to the item builder turns SWAP RED; committing the plan from the logged pose turns EXEC RED; reading the causal bucket from the executed speeds turns the bucket check RED.
"""
from __future__ import annotations

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
import v8f_t1_eval_histswap as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
FEAS2_TOOL_SHA256 = "4b66cc86cc5afe185736a65000c7f8cb1aa0c89d22bcd32770c99ad7971596df"
M_STEPS = 15
SWAP_KEYS = ("p10f", "base", "pose_now", "pose_t", "pose_s")


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


def load_feas2_tool():
    p = os.environ.get("V8F_T1_FEAS2_TOOL", "")
    if not p or not Path(p).is_file():
        pytest.skip("V8F_T1_FEAS2_TOOL=<the FEAS2-T1 tool 4b66cc86> not given")
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == FEAS2_TOOL_SHA256, "REFUSED: that file is not the FEAS2-T1 tool (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_feas2")


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


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    J = D.load_join(Path(world["root"]) / "join_eval.npz")
    model = tiny_model(J)
    base = ("--smooth", "sg72", "--limit", "causal", "--dump-heads")
    out = {"tmp": tmp, "y": run_cli(world, tmp, "y", *base, model=model)}                    # the flag absent
    out["off"] = run_cli(world, tmp, "off", *base, "--histswap", "off")
    out["x"] = run_cli(world, tmp, "x", *base, "--histswap", "logged")
    out["x_stats"] = dict(T1.CAUSAL_STATS)
    return out


# ------------------------------------------------------------------------------------------------ flag, refusal, C-OFF
def test_flag_parser_default_and_refusal(world, tmp_path):
    base = ["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]
    assert T1.build_parser().parse_args(base).histswap == "off"
    assert T1.build_parser().parse_args(base + ["--histswap", "logged"]).histswap == "logged"
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(base + ["--histswap", "executed"])
    with pytest.raises(SystemExit) as e:
        T1.main(base + ["--histswap", "logged", "--feasible", "kbox"])
    assert "REFUSED" in str(e.value)
    assert T1.HISTSWAP_MODES == ("off", "logged")


def test_c_off_flag_absent_equals_off_and_equals_the_feas2_tool(world, runs):
    assert_same_run(runs["y"], runs["off"])
    f2 = load_feas2_tool()
    out = run_cli(world, runs["tmp"], "f2", "--smooth", "sg72", "--limit", "causal", "--dump-heads", tool=f2)
    assert_same_run(runs["y"], out)


# ------------------------------------------------------------------------------------------------ SWAP and EXEC (direct rollout, tool under test passed in)
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
                        "pose_exec": np.array(st.d["pose"][st.j], np.float64),
                        "batch": {k: batch[k][b].detach().cpu().numpy().copy() for k in (*SWAP_KEYS, "limit", "limit_ok", "a_t", "a_s")}})

    tool.CAUSAL_STATS.clear()
    planner = tool.model_planner(model, torch.device("cpu"))
    tool.rollout(planner, states, ds, proxy, m=m, schedule="periodic", scene="logged", limit=limit, device=torch.device("cpu"), hook=hook)
    cl = [st.executed_ego() for st in states]
    return {"cap": cap, "store": store, "states": states, "cl": np.asarray(cl), "stats": dict(tool.CAUSAL_STATS), "J": J, "ds": ds}


def check_swap(c, expect_logged: bool):
    """Every ego-history key the network read equals the LOGGED clip array at the tick (direct indexing); returns how many (tick > 0) rows the executed history would have differed on."""
    hmax = c["ds"].hmax
    n_diff = 0
    for r in c["cap"]:
        d, j = c["store"].get(r["clip"]), r["j"]
        T = len(d["pose"])
        idc = np.clip(np.arange(j - hmax + 1, j + 1), 0, T - 1)
        a_t, a_s = int(r["batch"]["a_t"]), int(r["batch"]["a_s"])
        exp = {"p10f": np.nan_to_num(d["p10f"][idc], nan=0.0).astype(np.float32), "base": np.asarray(d["base"][j], np.float32), "pose_now": d["pose"][j][:3].astype(np.float32),
               "pose_t": d["pose"][j - a_t][:3].astype(np.float32), "pose_s": d["pose"][j - a_s][:3].astype(np.float32)}
        same = {k: bool(np.array_equal(r["batch"][k], exp[k])) for k in SWAP_KEYS}
        if expect_logged:
            assert all(same.values()), (r["n"], r["clip"], same, "the network did not read the logged history")
        elif r["n"] == 0:
            assert all(same.values()), "tick 0: the executed history IS the logged one"
        else:
            n_diff += int(not same["p10f"])
    return n_diff


def test_swap_the_network_reads_the_logged_history_and_unswapped_reads_the_executed(world, runs):
    sw = capture(T1, world, runs["tmp"], True)
    check_swap(sw, True)
    assert max(r["n"] for r in sw["cap"]) >= 1
    dev = max(float(np.abs(r["pose_exec"][:2] - sw["store"].get(r["clip"])["pose"][r["j"]][:2]).max()) for r in sw["cap"] if r["n"] >= 1)
    assert dev > 0.05, "non-vacuous: the executed pose left the logged path after tick 0"
    un = capture(T1, world, runs["tmp"], False)
    assert check_swap(un, False) > 0, "non-vacuous: without the swap the network reads an executed history that differs from the logged one"
    assert sw["cl"].tobytes() != un["cl"].tobytes()


def test_swap_the_causal_speed_bucket_reads_the_logged_speeds(world, runs):
    root = Path(world["root"])
    tracks = D.load_native_tracks(str(root / "native_tracks.pt"))
    J = D.load_join(str(root / "join_eval.npz"))
    store = D.ClipStore(str(root), "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ci, t0 = D.eval_grid_windows(str(root / "eval_grid_windows.json"), J)[0][:2]
    clip = str(J["clip"][ci])
    base = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in store.get(clip).items()}
    base["v"] = np.full_like(np.asarray(base["v"], np.float64), 15.0)                       # the logged history: fast (bucket 3)

    def bucket(tool, swap):
        st = tool.WindowState(base, tracks[clip], J["nav"][int(J["off"][ci]):int(J["off"][ci]) + int(J["n"][ci])], ci, t0, "deadreckon", swap)
        st.d["v"][:] = 1.0                                                                 # the executed history: crawling (bucket 0)
        st.j = t0 + 10
        proxy = tool._OneClipStore()
        ds = D.V8FMiniWindows(proxy, J, [], hist=4, hmax=10, max_age_t=2, max_age_s=6)
        stats = {}
        it = tool.build_item(ds, proxy, st, 1, 0, 0, "logged", "causal", stats)
        return int(np.argmax(it["limit"])), stats

    b_log, s_log = bucket(T1, True)
    b_exe, s_exe = bucket(T1, False)
    assert b_log != b_exe, "non-vacuous: the logged and the executed speeds fall in different buckets"
    assert b_log == int(T1.speed_max_bin(15.0)[0]) and b_exe == int(T1.speed_max_bin(1.0)[0])
    assert s_log.get("swap_ticks") == 1 and s_log.get("swap_bucket_differs") == 1 and "swap_ticks" not in s_exe
    mut = mutated('causal_bucket(st.d_in["v"], st.j,', 'causal_bucket(st.d["v"], st.j,')
    with pytest.raises(AssertionError):                                                    # MUTATION: the executed speeds feed the bucket under the swap
        assert bucket(mut, True)[0] == b_log


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


def test_exec_the_plan_is_committed_to_the_executed_state(world, runs):
    sw = capture(T1, world, runs["tmp"], True)
    assert exec_ok(sw) < 2e-3
    mut = mutated('        pose = self.d["pose"][j]\n        plan_ego, hit = envelope(plan_ego)', '        pose = self.d_in["pose"][j]\n        plan_ego, hit = envelope(plan_ego)')
    mc = capture(mut, world, runs["tmp"], True)
    assert exec_ok(mc) > 0.05, "MUTATION: committing from the logged pose must break the executed-state check"


def test_mutation_the_item_builder_on_the_executed_dict_turns_swap_red(world, runs):
    mut = mutated("    proxy.d = st.d_in\n", "    proxy.d = st.d\n")
    mc = capture(mut, world, runs["tmp"], True)
    with pytest.raises(AssertionError):
        check_swap(mc, True)


# ------------------------------------------------------------------------------------------------ TICKS and header
def test_ticks_tick0_equals_unswapped_tick1_state_equals_and_v0_is_the_executed_speed(world, runs):
    ty, tx = ticks(runs["y"]), ticks(runs["x"])
    assert set(tx) - set(ty) == {"v0_in", "histswap"} and not set(ty) - set(tx), (set(tx) ^ set(ty))
    assert str(tx["histswap"]) == "logged"
    k0, k1, k2 = (tx["tick"] == n for n in (0, 1, 2))
    for k in ("tau", "P", "plan", "plan_raw", "v0", "pose", "gt", "s_ckpt", "t_goal_prob"):
        assert np.array_equal(tx[k][k0], ty[k][ty["tick"] == 0]), f"tick 0 differs from the un-swapped run in {k}"
    assert np.array_equal(tx["v0_in"][k0], tx["v0"][k0])
    for k in ("pose", "v0"):                                                              # tick 1's executed state came from the SAME tick-0 plan
        assert np.array_equal(tx[k][k1], ty[k][ty["tick"] == 1]), f"tick 1 executed {k} differs: the executor did not run from the executed state"
    assert not np.array_equal(tx["plan_raw"][k1], ty["plan_raw"][ty["tick"] == 1]), "non-vacuous: the swapped input changed the tick-1 plan"
    assert not np.array_equal(tx["v0_in"][k1], tx["v0"][k1]), "non-vacuous: v0_in (logged) differs from v0 (executed) at tick 1"
    store = D.ClipStore(str(world["root"]), "eval", lru=8, grid="pool4x10", kin_mode="yaw", tracks=D.load_native_tracks(str(Path(world["root"]) / "native_tracks.pt")))
    for i in np.where(k1)[0]:
        d = store.get(str(tx["clip"][i]))
        assert tx["v0_in"][i] == np.nan_to_num(d["p10f"][int(tx["j"][i])][0], nan=0.0), "v0_in is the LOGGED measured speed at the tick"
    assert dumps(runs["y"], "fl").tobytes() == dumps(runs["x"], "fl").tobytes(), "the P10 floor changed under the swap"
    assert dumps(runs["y"], "cl").tobytes() != dumps(runs["x"], "cl").tobytes()


def test_header_carries_the_oracle_note_and_bucket_stats_only_under_the_swap(runs):
    hx, hy = header(runs["x"]), header(runs["y"])
    assert "histswap" not in hy and hx["histswap"]["mode"] == "logged" and "NOT deployable" in hx["histswap"]["oracle"]
    st = hx["histswap"]["bucket_stats_model_ticks"]
    assert st["swap_ticks"] == hx["n_windows"] * 2 and 0 <= st["swap_bucket_differs"] <= st["swap_ticks"]
    assert "swap_ticks" not in json.dumps(hy)
