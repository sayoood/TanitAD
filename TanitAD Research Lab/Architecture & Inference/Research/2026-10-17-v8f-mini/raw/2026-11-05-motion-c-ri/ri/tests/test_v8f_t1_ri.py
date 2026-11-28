"""REINDEXED (``--scene reindexed`` of ``stack/scripts/v8f_t1_eval_ri.py`` = the FRAMEFREEZE tool 0721fc01... + ``code/apply_ri_patch.py``): the frames follow the model's OWN progress along the logged path.

Run (from a tree that holds stack/ and taniteval/, the RI tool and the FRAMEFREEZE tool in stack/scripts/, this file in stack/tests/):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 pytest -q stack/tests/test_v8f_t1_ri.py        (``V8F_T1_FZ_TOOL=<the FRAMEFREEZE tool>`` overrides the default stack/scripts/v8f_t1_eval_framefreeze.py; sha256-pinned; absent => FAIL, never skip)
  * KRI1     identity: when the executed track equals the logged track (the tool's ``executor="logged"`` path) ``reindexed`` and ``logged`` hand the network the same bytes at every tick and issue the same plans; the receipt
             reads 0 rows reindexed, 0 future frames, 0 changed past rows.
  * KRI2     half speed: logged straight path 6 m/s, own track 3 m/s after t0: tau(i) = t0 + round((i - t0)/2) within +-1 on every row of every tick, no frame from after the row's time, the frames really are the logged ones.
  * KRI3     ahead: own 9 m/s > logged 6 m/s: tau(i) = i on every row, counted ``rows_clamped``, the rows equal the logged rows bit for bit.
  * KRI4     ``logged``, ``frozen``, ``hold_now`` and the flag absent are byte-identical to the SOURCE (FRAMEFREEZE) tool: dumps, ticks, header (CLI runs on a synthetic world) and the batches the network reads (unit level).
  * KRI5     the past rows (i <= t0) stay logged bit for bit even when the executed past is made to differ from the logged one (a state the constructor cannot produce: the guard test), and the receipt says so.
  * MUT      m1 the apply line deleted -> KRI2 RED; m2 the ``min(i, .)`` clamp dropped -> KRI3 RED; m3 the past rows reindexed too -> KRI5 RED (guard test AND the receipt's past-row check).
  * REACH    a CLI run with ``--scene reindexed`` writes the receipt with rows_reindexed > 0 and ``past_rows_bit_identical: true``; the reindex reaches the network (plans move at ticks >= 1, not at tick 0).
  * PURE     the projection / tie rules on a hand-made polyline (a stop with exact repeats, a dwell with jitter, the running maximum); a non-finite polyline and a missing ``feats`` are refused.
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
import v8f_t1_eval_ri as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
FZ_TOOL_SHA256 = "0721fc014089bea6b494e941a7d2a3d385e5c2ae89329848419fd843fbd9a0a9"
M_STEPS = 5                                                          # 6 ticks: rows are mixed past / future at ticks 1 and 2 (hmax 10)
N_TICKS = 30 // M_STEPS
APPLY = '        it["feats"], it["frame_ok"] = reindex_item(it, st, is_model=stats is not None)\n'
CLAMP = "np.minimum(i_rows, tau_star)"
PASTMASK = "past = i_rows <= t0"
DT = D.DT


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_ri")
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
    return tool_from_source(src.replace(old, new), "v8f_t1_eval_ri_mut")


def load_fz_tool():
    p = os.environ.get("V8F_T1_FZ_TOOL", "") or str(Path(T1.__file__).resolve().parent / "v8f_t1_eval_framefreeze.py")
    assert Path(p).is_file(), f"the SOURCE (FRAMEFREEZE) tool is not at {p}: KRI4 cannot run (failing, not skipping)"
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == FZ_TOOL_SHA256, "REFUSED: that file is not the FRAMEFREEZE tool (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_framefreeze_src")


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
    """Byte-identical dumps and ticks (same keys, same dtypes) and the same header apart from the ticks path."""
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
    out["logged"] = run_cli(world, tmp, "logged", *base, "--scene", "logged")
    out["frozen"] = run_cli(world, tmp, "frozen", *base, "--scene", "frozen")
    out["hold"] = run_cli(world, tmp, "hold", *base, "--scene", "hold_now", "--freeze-card-ref", "test sha line")
    out["ri"] = run_cli(world, tmp, "ri", *base, "--scene", "reindexed")
    return out


# ------------------------------------------------------------------------------------------------ unit level: a straight logged path
def make_straight(world, tool, v_log=6.0, t0=20, perturb_past=None):
    """One window on a synthetic clip whose LOGGED path is straight at ``v_log`` m/s and whose cached frame k is the constant k (so a row's source index is ``feats[r, 0, 0]``). Returns the pieces ``build_item`` needs."""
    root = Path(world["root"])
    tracks = D.load_native_tracks(str(root / "native_tracks.pt"))
    J = D.load_join(str(root / "join_eval.npz"))
    store = D.ClipStore(str(root), "eval", lru=4, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ci = 0
    clip = str(J["clip"][ci])
    d = store.get(clip)
    T = len(d["pose"])
    k = np.arange(T, dtype=np.float64)
    d["pose"][:] = np.stack([v_log * DT * k, np.zeros(T), np.zeros(T), np.full(T, v_log)], 1)
    d["feats"][:] = k[:, None, None]                                      # float16: exact below 2048
    n = np.arange(2 * T - 1, dtype=np.float64)
    tr = np.zeros((2 * T - 1, 4))
    tr[:, 0], tr[:, 3] = v_log * 0.5 * DT * n, v_log
    o, nr = int(J["off"][ci]), int(J["n"][ci])
    st = tool.WindowState(d, tr, J["nav"][o:o + nr], ci, t0, "deadreckon")
    if perturb_past is not None:
        st.d["pose"][:t0 + 1, 0] = perturb_past * DT * np.arange(t0 + 1)       # the executed past made to DIFFER from the logged one (the constructor cannot do this)
    cfg = M.V8FMiniConfig(**TINY)
    proxy = tool._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    return {"st": st, "ds": ds, "proxy": proxy, "d": d, "t0": t0, "hm": cfg.hmax, "tool": tool}


def build(sw, tick, scene="reindexed"):
    st = sw["st"]
    it = sw["tool"].build_item(sw["ds"], sw["proxy"], st, tick, 0, 0, scene, "off", stats={})          # stats={} => counted as the MODEL's rollout
    i_rows = np.arange(st.j - sw["hm"] + 1, st.j + 1)
    return it, i_rows


def drive(sw, v_own, scene="reindexed", ticks_=(0, 1, 2, 3)):
    """Tick 0 at t0, then execute a straight plan at ``v_own`` m/s for 5 steps and build the next item (rows are mixed past / future at ticks 1 and 2); returns [(it, i_rows, logged_it)] per tick (logged_it = the same item under --scene logged)."""
    plan = np.stack([v_own * DT * np.arange(1, 31), np.zeros(30)], 1)
    out = []
    for n in ticks_:
        if n:
            sw["st"].commit_plan(plan, 5)
        it, i_rows = build(sw, n, scene)
        lg, _ = build(sw, n, "logged")
        out.append((it, i_rows, lg))
    return out


def src_index(it):
    return np.asarray(it["feats"])[:, 0, 0].astype(np.int64)


def check_kri2(tool, world):
    """KRI2: logged 6 m/s, own 3 m/s: tau(i) = t0 + round((i - t0)/2) within 1 on every row after t0, tau <= i, past rows logged; non-vacuous (some row is more than 2 steps behind)."""
    sw = make_straight(world, tool, 6.0, 20)
    tool.ri_reset()
    t0, seen_far = sw["t0"], 0
    for it, i_rows, lg in drive(sw, 3.0):
        tau = src_index(it)
        for r, i in enumerate(i_rows):
            if i <= t0:
                assert tau[r] == max(i, 0) and np.array_equal(it["feats"][r], lg["feats"][r]) and bool(it["frame_ok"][r]) == bool(lg["frame_ok"][r]), (int(i), int(tau[r]))
            else:
                exp = t0 + int(round((i - t0) / 2))
                assert abs(int(tau[r]) - exp) <= 1, f"row i={int(i)}: tau {int(tau[r])}, expected {exp} +-1"
                assert tau[r] <= i, f"row i={int(i)} took the future frame {int(tau[r])}"
                seen_far += int(i - tau[r] > 2)
        assert it["feats"].shape == lg["feats"].shape and it["feats"].dtype == lg["feats"].dtype and it["frame_ok"].shape == lg["frame_ok"].shape
    assert seen_far >= 5, "KRI2 would pass vacuously: no row is more than 2 steps behind"
    return tool.RI_STATS["model"]


def check_kri3(tool, world):
    """KRI3: logged 6 m/s, own 9 m/s (ahead): tau(i) = i on every row, the rows are the logged rows bit for bit, every row after t0 counted clamped, none reindexed, no future frame."""
    sw = make_straight(world, tool, 6.0, 20)
    tool.ri_reset()
    n_fut = 0
    for it, i_rows, lg in drive(sw, 9.0):
        tau = src_index(it)
        assert np.array_equal(tau, np.clip(i_rows, 0, None)), (tau.tolist(), i_rows.tolist())
        assert np.array_equal(it["feats"], lg["feats"]) and np.array_equal(it["frame_ok"], lg["frame_ok"])
        n_fut += int((i_rows > sw["t0"]).sum())
    r = tool.RI_STATS["model"]
    assert n_fut > 0 and r["rows_clamped"] == n_fut and r["rows_reindexed"] == 0 and r["future_frame_rows"] == 0, (n_fut, r)
    return r


def check_kri5(tool, world):
    """KRI5 (guard): the executed past is made to differ (own half speed before t0): the rows with i <= t0 are still the logged rows, bit for bit, and the receipt's own past check says unchanged."""
    sw = make_straight(world, tool, 6.0, 20, perturb_past=3.0)
    tool.ri_reset()
    n_past, n_mixed = 0, 0
    for it, i_rows, lg in drive(sw, 3.0):
        past = i_rows <= sw["t0"]
        n_past += int(past.sum())
        n_mixed += int(past.any() and (~past).any())
        assert np.array_equal(np.asarray(it["feats"])[past], np.asarray(lg["feats"])[past]), "a past row was reindexed"
        assert np.array_equal(np.asarray(it["frame_ok"])[past], np.asarray(lg["frame_ok"])[past])
    assert n_past > 0 and n_mixed >= 1, "KRI5 would pass vacuously: no item mixes past and future rows"
    ps = tool.RI_STATS["past"]
    assert ps["rows_checked"] > 0 and ps["rows_changed"] == 0, ps
    return ps


def test_kri2_half_speed_tau_is_half_the_progress(world):
    r = check_kri2(T1, world)
    assert r["rows_reindexed"] > 0 and r["rows_clamped"] == 0 and r["future_frame_rows"] == 0
    lag = T1.ri_summary(r)["lag_cache_steps_reindexed_rows"]
    assert lag["n"] > 0 and lag["max"] >= 5 and lag["mean"] > 1.0


def test_kri3_ahead_is_clamped_to_the_logged_frame(world):
    check_kri3(T1, world)


def test_kri5_past_rows_stay_logged_even_when_the_executed_past_differs(world):
    check_kri5(T1, world)


# ------------------------------------------------------------------------------------------------ KRI1 identity
def capture_rollout(tool, world, tmp, scene, executor, limit="causal", m=M_STEPS):
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
    states = tool.make_states(store, tracks, J, wins, "deadreckon")
    cap = []

    def hook(n, batch, plan, sts):
        for b, st in enumerate(sts):
            cap.append({"n": n, "j": st.j, "t0": st.t0, "clip": str(J["clip"][st.ci]), "plan": np.array(plan[b], np.float64),
                        "batch": {k: v[b].detach().cpu().numpy().copy() for k, v in batch.items() if torch.is_tensor(v)}})

    tool.CAUSAL_STATS.clear()
    if hasattr(tool, "ri_reset"):
        tool.ri_reset()
    planner = tool.model_planner(model, torch.device("cpu"))
    kw = {"executor": "logged", "logged_tracks": {ci: tracks[str(J["clip"][ci])] for ci in range(len(J["clip"]))}} if executor == "logged" else {}
    tool.rollout(planner, states, ds, proxy, m=m, schedule="periodic", scene=scene, limit=limit, device=torch.device("cpu"), hook=hook, **kw)
    return {"cap": cap, "cl": np.asarray([st.executed_ego() for st in states])}


def test_kri1_identity_when_the_executed_track_is_the_logged_track(world, runs):
    a = capture_rollout(T1, world, runs["tmp"], "logged", "logged")
    b = capture_rollout(T1, world, runs["tmp"], "reindexed", "logged")
    assert len(a["cap"]) == len(b["cap"]) == 12 * N_TICKS
    for x, y in zip(a["cap"], b["cap"]):
        assert (x["n"], x["clip"], x["t0"]) == (y["n"], y["clip"], y["t0"])
        for k in x["batch"]:
            assert x["batch"][k].tobytes() == y["batch"][k].tobytes() and x["batch"][k].dtype == y["batch"][k].dtype, (x["n"], k)
        assert x["plan"].tobytes() == y["plan"].tobytes()
    assert a["cl"].tobytes() == b["cl"].tobytes()
    r = T1.RI_STATS["model"]
    assert r["items"] == 12 * N_TICKS and r["rows_reindexed"] == 0 and r["future_frame_rows"] == 0, r
    assert T1.RI_STATS["past"]["rows_checked"] > 0 and T1.RI_STATS["past"]["rows_changed"] == 0
    s = T1.ri_summary(r)
    assert s["lag_cache_steps_all_rows_after_t0"]["max"] == 0.0
    # non-vacuous: the same windows executed by the MODEL are not the logged track, and the batches then differ (the control can move)
    c = capture_rollout(T1, world, runs["tmp"], "reindexed", "plan")
    assert T1.RI_STATS["model"]["rows_reindexed"] > 0
    assert any(x["batch"]["feats"].tobytes() != y["batch"]["feats"].tobytes() for x, y in zip(a["cap"], c["cap"]))


# ------------------------------------------------------------------------------------------------ KRI4 bit identity of the other modes
def test_kri4_other_modes_are_bit_identical_to_the_source_tool(world, runs):
    src = load_fz_tool()
    assert_same_run(runs["y"], runs["logged"])
    assert_same_run(runs["y"], run_cli(world, runs["tmp"], "src_y", *runs["base"], tool=src))
    assert_same_run(runs["logged"], run_cli(world, runs["tmp"], "src_logged", *runs["base"], "--scene", "logged", tool=src))
    assert_same_run(runs["frozen"], run_cli(world, runs["tmp"], "src_frozen", *runs["base"], "--scene", "frozen", tool=src))
    assert_same_run(runs["hold"], run_cli(world, runs["tmp"], "src_hold", *runs["base"], "--scene", "hold_now", "--freeze-card-ref", "test sha line", tool=src))
    for k in ("y", "logged", "frozen", "hold"):
        assert "scene_reindex" not in header(runs[k])
    assert "scene" not in ticks(runs["y"]) and str(ticks(runs["hold"])["scene"]) == "hold_now"


def test_kri4_unit_level_batches_of_the_other_modes_equal_the_source_tools(world, runs):
    src = load_fz_tool()
    for scene in ("logged", "frozen", "hold_now"):
        a = capture_rollout(src, world, runs["tmp"], scene, "plan")
        b = capture_rollout(T1, world, runs["tmp"], scene, "plan")
        assert len(a["cap"]) == len(b["cap"]) == 12 * N_TICKS
        for x, y in zip(a["cap"], b["cap"]):
            assert all(x["batch"][k].tobytes() == y["batch"][k].tobytes() for k in x["batch"]) and x["plan"].tobytes() == y["plan"].tobytes(), (scene, x["n"])
        assert a["cl"].tobytes() == b["cl"].tobytes()
    assert T1.SCENE_MODES == ("logged", "frozen", "hold_now", "reindexed") and src.SCENE_MODES == ("logged", "frozen", "hold_now")


# ------------------------------------------------------------------------------------------------ mutations
def test_mutations_turn_the_checks_red(world):
    check_kri2(T1, world), check_kri3(T1, world), check_kri5(T1, world)               # the unmutated tool passes all three
    survived = []
    for old, new, why, check in ((APPLY, "        pass\n", "m1: the apply line deleted", check_kri2),
                                 (CLAMP, "tau_star", "m2: the min(i, .) clamp dropped", check_kri3),
                                 (PASTMASK, "past = i_rows < -(10 ** 9)", "m3: the past rows reindexed too", check_kri5)):
        mt = mutated(old, new)
        try:
            check(mt, world)
            survived.append(why)
        except AssertionError:
            pass
    assert not survived, f"mutations that were not turned RED: {survived}"


def test_m3_is_also_caught_by_the_receipts_own_past_row_check(world):
    """The header receipt compares bytes with its own mask: under m3 on a state whose executed past differs it reads rows_changed > 0 (past_rows_bit_identical false)."""
    mt = mutated(PASTMASK, "past = i_rows < -(10 ** 9)")
    sw = make_straight(world, mt, 6.0, 20, perturb_past=3.0)
    mt.ri_reset()
    drive(sw, 3.0)
    ps = mt.RI_STATS["past"]
    assert ps["rows_checked"] > 0 and ps["rows_changed"] > 0, ps


# ------------------------------------------------------------------------------------------------ REACH, header
def test_reach_cli_run_writes_a_receipt_with_rows_reindexed(runs):
    h = header(runs["ri"])
    assert h["scene"] == "reindexed" and "scene_reindex" in h
    sr = h["scene_reindex"]
    assert sr["rows_reindexed"] > 0 and sr["items"] == 12 * N_TICKS and sr["future_frame_rows"] == 0
    assert sr["rows"] == sr["items"] * 10 and sr["rows_logged"] + sr["rows_reindexed"] + sr["rows_on_time"] + sr["rows_clamped"] == sr["rows"]
    assert sr["past_rows_bit_identical"] is True and sr["past_rows_checked"] > 0 and sr["past_rows_changed"] == 0
    assert sr["floor"]["items"] == 12 * N_TICKS and sr["floor"]["future_frame_rows"] == 0
    lag = sr["lag_cache_steps_all_rows_after_t0"]
    assert lag["n"] > 0 and lag["max"] >= 1 and set(lag) == {"n", "mean", "p50", "p90", "max"}
    assert any("REINDEXED" in c and "NOT T2" in c for c in h["open_channels"])
    assert h["strict"]["camera"] is False and "frame_freeze" not in h and "camera_motion" not in h
    assert str(ticks(runs["ri"])["scene"]) == "reindexed"


def test_reindexed_reaches_the_network_only_after_the_window_start(runs):
    za, zb = ticks(runs["logged"]), ticks(runs["ri"])
    assert za["tick"].tobytes() == zb["tick"].tobytes() and za["win"].tobytes() == zb["win"].tobytes()
    t0, t1 = za["tick"] == 0, za["tick"] >= 1
    assert za["plan_raw"][t0].tobytes() == zb["plan_raw"][t0].tobytes(), "tick 0 has only past rows: the plans must be identical"
    assert float(np.abs(za["plan_raw"][t1] - zb["plan_raw"][t1]).max()) > 1e-3, "the reindexed frames did not reach the network on the tiny model"
    assert dumps(runs["logged"], "fl").tobytes() == dumps(runs["ri"], "fl").tobytes(), "the P10 floor reads no frames"


def test_flag_parser_and_refusals():
    base = ["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]
    assert T1.build_parser().parse_args(base).scene == "logged"
    assert T1.build_parser().parse_args(base + ["--scene", "reindexed"]).scene == "reindexed"
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(base + ["--scene", "reindex"])
    with pytest.raises(RuntimeError) as e:
        T1.reindex_item({"frame_ok": np.ones(3, bool)}, types.SimpleNamespace())
    assert "REFUSED" in str(e.value) and "feats" in str(e.value)
    with pytest.raises(ValueError) as e:
        T1.logged_arc(np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 0.0]]))
    assert "REFUSED" in str(e.value)


# ------------------------------------------------------------------------------------------------ the pure layer
def test_pure_projection_ties_dwell_and_running_maximum():
    k = np.arange(60)
    xy = np.stack([1.2 * k, np.zeros(60)], 1)
    t, off = T1.project_taus(xy, np.stack([1.2 * np.array([0, 5, 17, 59]) + 0.01, [0.0, 2.0, -3.0, 0.5]], 1))
    assert t.tolist() == [0, 5, 17, 59] and np.allclose(off, [0.0, 2.0, 3.0, 0.5], atol=1e-3)
    assert T1.project_taus(xy, np.array([[1e4, 7.0]]))[0].tolist() == [59]          # beyond the end: the last index
    assert T1.project_taus(xy, np.array([[-50.0, 0.0]]))[0].tolist() == [0]
    # a stop with EXACT repeats (indices 10..30 at one point): a point on it goes to the LATEST index
    stop = xy.copy()
    stop[10:31] = xy[10]
    stop[31:] = xy[10] + 1.2 * np.arange(1, 30)[:, None] * np.array([1.0, 0.0])
    assert T1.project_taus(stop, stop[[5, 10, 20, 30, 45]])[0].tolist() == [5, 30, 30, 30, 45]
    # a dwell with jitter (a few mm per step): the same, through RI_DWELL_M
    jit = stop.copy()
    jit[10:31, 0] += np.cumsum(np.full(21, 0.002))
    assert T1.project_taus(jit, jit[[20]])[0].tolist() == [30]
    # rows: identity on the logged path (including the stop): tau(i) = i, tau* >= i
    i_rows = np.arange(5, 45)
    rr = T1.reindex_rows(i_rows, 4, stop[i_rows], stop)
    assert np.array_equal(rr["tau"], i_rows) and (rr["tau_star"] >= i_rows).all() and not rr["raised"].any()
    # a reversing own track (tau* falls) is lifted by the running maximum, and past rows are never touched
    own = np.array([[1.2 * v, 0.0] for v in (10, 11, 12, 6, 7, 20, 21, 22)])
    i2 = np.arange(23, 31)
    rr = T1.reindex_rows(i2, 22, own, xy)
    assert rr["tau"].tolist() == [10, 11, 12, 12, 12, 20, 21, 22] and rr["raised"].tolist() == [False, False, False, True, True, False, False, False]
    assert rr["tau"].tolist() == np.maximum.accumulate(rr["tau"]).tolist() and (rr["tau"] <= i2).all()
    rr = T1.reindex_rows(np.arange(-3, 4), 1, np.zeros((7, 2)) + 3.0, xy)         # padded rows (i < 0): past, the clipped own index
    assert rr["tau"][:5].tolist() == [0, 0, 0, 0, 1] and rr["past"][:5].all()
