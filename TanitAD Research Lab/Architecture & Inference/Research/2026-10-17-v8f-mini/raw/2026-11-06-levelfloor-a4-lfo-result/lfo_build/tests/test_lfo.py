"""Unit tests of the LFO tool (PREREG_LEVELFLOOR_A4 sha256 8bfbc386...: ``--floor-vhat-source {heads,logged}``, the logged-speed CEILING diagnostic) and its known answer KA-O3.

Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/motion_c/lfo_build/pt/run <this file>
  LFO_TREE  the read tree holding stack/ + taniteval/ with the LFO tool and ``level_floor.py`` in stack/scripts/ (default F:/TanitAD-scratch/motion_c/lfo_read/tree)
  LFO_TOOL  the tool under test (default ``<LFO_TREE>/stack/scripts/v8f_t1_eval_ri.py``); ``code/run_mutations_lfo.py`` points it at a mutated copy
Known answer KA-O3 (every test whose name starts ``test_ka_o3``; the mutation runner counts those):
  * on a synthetic tick the oracle v_hat equals ``lf_read.logged_speed`` of the same ``gt`` to 1e-9 m/s (and the pose grid's own path length), through the tool's own ``gt_in_frame``;
  * ``gt_ok`` False gives no floor, also when the route values are finite (the guard itself);
  * the level input of ``LevelFloor`` is the oracle, not the heads; the heads still supply presence and gap;
  * ``rollout`` hands the tick's window states to the oracle, and the tick dump's gt gives the same speed as the input (the instrument and the input are one quantity).
The rest: ``heads`` is the LF tool bit for bit; the flag refusals; header / meta labels; the TickRecorder field sets.
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TREE = Path(os.environ.get("LFO_TREE", "F:/TanitAD-scratch/motion_c/lfo_read/tree"))
LF_TREE = Path("F:/TanitAD-scratch/motion_c/lf_read/tree")
MC = Path("E:/Projects/TanitAD-artifacts/dagland/motion_c")
LF_READ = MC / "lf_read" / "code" / "lf_read.py"
LF_READ_SHA = "862e4e6c357bc084c971eceebd3ead1576d2218b1296eb1b10799f76710050e8"
LF_TOOL_SHA = "6166178a604edc5343faa4fc6f139ec83dc65d00b91155a77b3f33dda65f4d70"
for _p in (TREE / "stack", TREE / "taniteval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import torch  # noqa: E402
import tanitad  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(TREE.resolve()), ("tanitad is not from the test tree", tanitad.__file__)
torch.set_num_threads(1)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


TOOL = Path(os.environ.get("LFO_TOOL", str(TREE / "stack" / "scripts" / "v8f_t1_eval_ri.py")))
T1 = _load("v8f_t1_eval_lfo_under_test", TOOL)
OP = _load("level_floor_lfo_under_test", TREE / "stack" / "scripts" / "level_floor.py")
assert _sha(LF_TREE / "stack" / "scripts" / "v8f_t1_eval_ri.py") == LF_TOOL_SHA, "the LF tool (6166178a) the LFO tool is compared with has changed"
LFT = _load("v8f_t1_eval_lf_under_test", LF_TREE / "stack" / "scripts" / "v8f_t1_eval_ri.py")
assert _sha(LF_READ) == LF_READ_SHA, "lf_read.py (the reader whose logged_speed defines the oracle) differs from its pin"
LFR = _load("lf_read_for_lfo_tests", LF_READ)
DT = 0.2


# ------------------------------------------------------------------------------------------------ synthetic clips
def pose_grid(T=120, seed=3, lo=4.5):
    """A curved path at 0.2 s steps with a varying speed in [lo, lo + 4] m/s; columns (x, y, yaw, 0)."""
    rng = np.random.default_rng(seed)
    k = np.arange(T)
    v = lo + 2.0 + 2.0 * np.sin(k * 0.05) + rng.uniform(0, 0.3, T)
    yaw = np.cumsum(0.02 * np.sin(k * 0.07) + 0.01)
    x = np.cumsum(v * DT * np.cos(yaw))
    y = np.cumsum(v * DT * np.sin(yaw))
    return np.stack([x, y, yaw, np.zeros(T)], 1)


def executed_pose(pg):
    """The executed pose track: the logged one moved and turned (the executed pose differs from the logged one), so the oracle must go through the ego-frame transform."""
    ex = pg.copy()
    ex[:, :2] += np.array([3.0, -2.0])
    ex[:, 2] += 0.4
    return ex


class Store:
    def __init__(self, grids):
        self.g = grids

    def get(self, clip):
        return {"pose": self.g[clip]}


def true_speed(pg, j):
    """Independent: the path length of the logged poses j+1 .. j+6 (five segments, 1.0 s)."""
    return float(np.linalg.norm(np.diff(pg[j + 1:j + 7, :2], axis=0), axis=1).sum() / 1.0)


def make_world(T_a=120, T_b=75):
    grids = {"clipA": pose_grid(T_a, 3), "clipB": pose_grid(T_b, 4)}
    J = {"clip": np.array(["clipA", "clipB"])}
    return grids, J, Store(grids)


def state(ci, j, grids, J, t0=0):
    ex = executed_pose(grids[str(J["clip"][ci])])
    return SimpleNamespace(ci=ci, j=j, t0=t0, d={"pose": ex})


# ------------------------------------------------------------------------------------------------ KA-O3
def test_ka_o3_oracle_equals_logged_speed_of_the_same_gt():
    grids, J, store = make_world()
    oracle = T1.make_logged_oracle(store, J)
    for j in (0, 7, 20, 41, 60, 85):
        states = [state(0, j, grids, J), state(0, j + 3, grids, J), state(1, min(j, 40), grids, J)]
        v = oracle(states)
        gts, oks = zip(*[T1.gt_in_frame(store.get(str(J["clip"][s.ci]))["pose"], np.array(s.d["pose"][s.j], np.float64), s.j) for s in states])
        want, ok = LFR.logged_speed(np.stack(gts), np.asarray(oks))
        assert v.shape == (3,) and v.dtype == np.float64
        assert np.array_equal(np.isfinite(v), ok) and ok.all()
        assert np.abs(v - want).max() <= 1e-9, (j, v, want)                                   # the reader's own definition, to 1e-9 m/s (it is exactly equal)
        ind = np.array([true_speed(grids[str(J["clip"][s.ci])], s.j) for s in states])
        assert np.abs(v - ind).max() <= 1e-9, (j, v, ind)                                     # and the pose grid's own path length (the transform keeps the length)
    assert 3.0 < true_speed(grids["clipA"], 20) < 12.0                                         # control: the synthetic speeds sit above F's 3 m/s band
    # a tick whose executed pose is moved reads the same speed as one at the logged pose (the speed is offset-free)
    s_moved, s_same = state(0, 20, grids, J), state(0, 20, grids, J)
    s_same.d = {"pose": grids["clipA"].copy()}
    assert abs(float(oracle([s_moved])[0]) - float(oracle([s_same])[0])) <= 1e-9


def test_ka_o3_gt_ok_false_gives_no_floor():
    grids, J, store = make_world()
    oracle = T1.make_logged_oracle(store, J)
    # (i) the logged route ends before j + 30: the tool's gt_in_frame says gt_ok False, the oracle has no speed
    j_end = len(grids["clipB"]) - 20
    v = oracle([state(1, j_end, grids, J), state(0, 20, grids, J)])
    assert np.isnan(v[0]) and np.isfinite(v[1])
    # (ii) the guard itself: FINITE route values with gt_ok False are still no speed
    gt = np.stack([np.stack([np.arange(30) * 0.9, np.zeros(30)], 1)] * 3)
    out = T1.logged_route_speed(gt, np.array([True, False, True]))
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])
    assert abs(out[0] - 4.5) <= 1e-9                                                           # 5 segments of 0.9 m = 4.5 m in 1.0 s
    assert np.isnan(T1.logged_route_speed(np.where(np.arange(30)[None, :, None] == 3, np.nan, gt), np.array([True] * 3))).all()   # a non-finite knot among 1..6 is no speed either
    # (iii) through the floor: a row with a speed binds, a row without gets no floor, its plan is untouched bit for bit
    lf = T1.LevelFloor(None, 0.75, "x.pt", "0" * 64, vhat_source="logged", oracle=oracle)
    lf.heads = lambda batch: (np.full(2, 10.0, np.float32), np.full(2, -6.0, np.float32), np.full(2, 0.5, np.float32))
    lf.states = [state(0, 20, grids, J), state(1, j_end, grids, J)]
    plan = np.stack([OP.T_KNOTS * 0.2, np.zeros(30)], 1)[None].repeat(2, 0).astype(np.float32)        # a 0.2 m/s crawl
    out_plan, info = lf({}, plan, np.array([0.2, 0.2]))
    assert info["bind"].tolist() == [True, False] and np.isnan(info["vf"][1]) and np.isnan(info["vhat"][1]) and info["vf"][0] > 0
    assert np.array_equal(out_plan[1], plan[1]) and not np.array_equal(out_plan[0], plan[0])
    assert lf.stats["oracle_no_route"] == 1


def test_ka_o3_the_level_input_is_the_oracle_not_the_heads():
    grids, J, store = make_world()
    oracle = T1.make_logged_oracle(store, J)
    heads_v = np.array([12.0, 12.0, 12.0], np.float32)                                         # deliberately far from the logged speed
    logit, gap = np.array([-6.0, -6.0, -6.0], np.float32), np.array([0.5, 0.5, 0.5], np.float32)
    states = [state(0, 20, grids, J), state(0, 33, grids, J), state(1, 25, grids, J)]
    plan = np.stack([OP.T_KNOTS * 0.3, np.zeros(30)], 1)[None].repeat(3, 0).astype(np.float32)
    v0 = np.array([0.3, 0.3, 0.3])
    v_log = np.array([true_speed(grids[str(J["clip"][s.ci])], s.j) for s in states])
    assert (np.abs(v_log - 12.0) > 1.0).all()
    lf = T1.LevelFloor(None, 0.75, "x.pt", "0" * 64, vhat_source="logged", oracle=oracle)
    lf.heads = lambda batch: (heads_v, logit, gap)
    lf.states = states
    out, info = lf({}, plan, v0)
    assert np.abs(info["vhat"] - v_log).max() <= 1e-9 and np.abs(info["vhat_heads"] - 12.0).max() == 0.0
    want_out, want = OP.apply_floor_batch(plan, v0, info["vhat"], OP.sigmoid(logit), gap, 0.75)       # F fed the oracle's own v_log directly (bit for bit)
    assert np.array_equal(out, want_out) and np.array_equal(info["vf"], want["vf"], equal_nan=True)
    assert np.allclose(info["vf"], 0.75 * v_log, atol=1e-9, rtol=0)                           # no lead (pi < 0.5): v_f = kappa * v_log
    assert (info["pi"] == want["pi"]).all() and (info["gap"] == want["gap"]).all()            # presence and gap come from the heads
    # the heads source is untouched: the same call reads the heads' 12 m/s
    lh = T1.LevelFloor(None, 0.75, "x.pt", "0" * 64)
    lh.heads = lambda batch: (heads_v, logit, gap)
    oh, ih = lh({}, plan, v0)
    assert np.abs(ih["vhat"] - 12.0).max() == 0.0 and "vhat_heads" not in ih and lh.vhat_source == "heads"
    with pytest.raises(RuntimeError):                                                          # logged without states / oracle is an error, not a silent fallback to the heads
        T1.LevelFloor(None, 0.75, "x.pt", "0" * 64, vhat_source="logged", oracle=oracle).oracle_vhat(3)
    lf2 = T1.LevelFloor(None, 0.75, "x.pt", "0" * 64, vhat_source="logged", oracle=oracle)
    lf2.heads = lambda batch: (heads_v, logit, gap)
    with pytest.raises(RuntimeError):
        lf2({}, plan, v0)


class _FakeModel:
    def __init__(self, raw):
        self.raw = raw

    def forward_infer(self, batch):
        t = torch.from_numpy(self.raw)
        return {"traj30": t, "p10": t * 0.9 + 0.01, "_keys_read": {"feats"}}


def _run_rollout(monkeypatch, grids, J, store, states, floor, hook=True):
    B = len(states)
    raw = np.stack([np.stack([OP.T_KNOTS * 0.25, np.zeros(30)], 1)] * B).astype(np.float32)
    assert "p10f" in T1.MODEL_KEYS and "a_t" in T1.MODEL_KEYS and "a_s" in T1.MODEL_KEYS
    monkeypatch.setattr(T1, "build_item", lambda *a, **k: {})
    monkeypatch.setattr(T1.D, "collate", lambda items: {k: torch.zeros(len(items), 3, 8, dtype=torch.float32) if k == "p10f" else torch.zeros(len(items), dtype=torch.long) for k in T1.MODEL_KEYS})
    planner = T1.model_planner(_FakeModel(raw), "cpu", keep_aux=True, smooth="off", feasible="off", floor=floor)
    rec = T1.TickRecorder(planner, store, J, heads=False, smooth="off", feasible="off", level_floor=True, floor_oracle=floor.vhat_source == "logged")
    T1.rollout(planner, states, None, None, m=10, schedule="periodic", scene="logged", limit="logged", device="cpu", hook=rec)
    return rec, planner


class _St:
    def __init__(self, ci, j, t0, grids, J):
        self.ci, self.j, self.t0 = ci, j, t0
        self.d = {"pose": executed_pose(grids[str(J["clip"][ci])])}

    def commit_plan(self, pl, m):
        self.j += m

    def executed_ego(self):
        return None


def test_ka_o3_rollout_hands_the_tick_states_to_the_oracle_and_the_dump_gives_the_same_speed(monkeypatch):
    grids, J, store = make_world(T_a=120, T_b=75)
    states = [_St(0, 20, 20, grids, J), _St(1, 30, 30, grids, J)]                              # clipB: the third tick (j = 50, 50 + 30 > 74) has no complete route
    oracle = T1.make_logged_oracle(store, J)
    lf = T1.LevelFloor(None, 0.75, "x.pt", "0" * 64, vhat_source="logged", oracle=oracle)
    lf.heads = lambda batch: (np.full(len(batch["p10f"]), 9.0, np.float32), np.full(len(batch["p10f"]), -6.0, np.float32), np.full(len(batch["p10f"]), 0.5, np.float32))
    rec, planner = _run_rollout(monkeypatch, grids, J, store, states, lf)
    vhat = np.asarray(rec.rows["lf_vhat"], np.float64)                                          # tick-major: [tick0 w0, tick0 w1, tick1 w0, ...]
    want = np.array([x for n in range(3) for x in (true_speed(grids["clipA"], 20 + 10 * n), true_speed(grids["clipB"], 30 + 10 * n) if 30 + 10 * n + 30 <= 74 else np.nan)])
    assert vhat.shape == (6,) and np.array_equal(np.isnan(vhat), np.isnan(want)) and np.isnan(vhat).sum() == 1
    assert np.nanmax(np.abs(vhat - want)) <= 1e-5                                               # float32 in the dump
    # the instrument and the input are one quantity: the reader's logged_speed of the dumped gt equals the dumped input
    v_dump, _ = LFR.logged_speed(np.asarray(rec.rows["gt"]), np.asarray(rec.rows["gt_ok"]))
    assert np.array_equal(np.isnan(v_dump), np.isnan(vhat)) and np.nanmax(np.abs(v_dump - vhat)) <= 1e-5
    assert np.abs(np.asarray(rec.rows["lf_vhat_heads"], np.float64) - 9.0).max() == 0.0
    assert np.isnan(np.asarray(rec.rows["lf_vf"], np.float64)[np.isnan(vhat)]).all() and not np.asarray(rec.rows["lf_bind"])[np.isnan(vhat)].any()
    assert lf.states is states                                                                  # rollout handed the window states over
    assert planner.states is states


# ------------------------------------------------------------------------------------------------ heads = the LF tool
def _stub_heads(lf, B, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.uniform(1.0, 14.0, B).astype(np.float32)
    logit = rng.uniform(-3.0, 3.0, B).astype(np.float32)
    gap = rng.uniform(0.0, 1.5, B).astype(np.float32)
    lf.heads = lambda batch: (v, logit, gap)


def test_heads_mode_is_the_lf_tool_bit_for_bit():
    rng = np.random.default_rng(11)
    B = 16
    plan = np.stack([np.stack([OP.T_KNOTS * rng.uniform(0.05, 6.0), np.cumsum(rng.normal(0, 0.05, 30))], 1) for _ in range(B)]).astype(np.float32)
    v0 = rng.uniform(0.0, 8.0, B)
    a, b = LFT.LevelFloor(None, 0.75, "x.pt", "0" * 64), T1.LevelFloor(None, 0.75, "x.pt", "0" * 64)
    for lf in (a, b):
        _stub_heads(lf, B)
    oa, ia = a({}, plan, v0)
    ob, ib = b({}, plan, v0)
    assert np.array_equal(oa, ob) and set(ia) == set(ib) and all(np.array_equal(ia[k], ib[k], equal_nan=True) for k in ia)
    assert a.stats == b.stats and ia["bind"].any()
    assert b.vhat_source == "heads" and "oracle_no_route" not in b.stats
    # model_planner / TickRecorder
    raw = np.stack([np.stack([OP.T_KNOTS * 0.1, np.zeros(30)], 1)] * 3).astype(np.float32)
    batch = {"p10f": torch.tensor([[[0.0] * 8, [0.0] * 8, [v] + [0.0] * 7] for v in (0.0, 1.0, 9.0)])}
    for smooth, feasible in (("off", "off"), ("sg72", "off"), ("sg72", "kbox")):
        fa, fb = LFT.LevelFloor(None, 0.5, "x.pt", "0" * 64), T1.LevelFloor(None, 0.5, "x.pt", "0" * 64)
        for lf in (fa, fb):
            _stub_heads(lf, 3, seed=2)
        pa = LFT.model_planner(_FakeModel(raw), "cpu", keep_aux=True, smooth=smooth, feasible=feasible, floor=fa)
        pb = T1.model_planner(_FakeModel(raw), "cpu", keep_aux=True, smooth=smooth, feasible=feasible, floor=fb)
        assert np.array_equal(pa(batch), pb(batch)) and np.array_equal(pa.plan_raw, pb.plan_raw)
        assert all(np.array_equal(pa.floor_info[k], pb.floor_info[k], equal_nan=True) for k in pa.floor_info)
        ra, rb = LFT.TickRecorder(pa, None, None, smooth=smooth, feasible=feasible, level_floor=True), T1.TickRecorder(pb, None, None, smooth=smooth, feasible=feasible, level_floor=True)
        assert list(ra.rows) == list(rb.rows)
        rc = T1.TickRecorder(pb, None, None, smooth=smooth, feasible=feasible, level_floor=True, floor_oracle=True)
        assert set(rc.rows) == set(rb.rows) | set(T1.TickRecorder.LFO_FIELDS)
        rd = T1.TickRecorder(pb, None, None, smooth=smooth, feasible=feasible, floor_oracle=True)       # no floor: the oracle flag adds nothing
        assert list(rd.rows) == list(T1.TickRecorder(pb, None, None, smooth=smooth, feasible=feasible).rows)
    assert T1.TickRecorder.LF_FIELDS == LFT.TickRecorder.LF_FIELDS and T1.TickRecorder.LFO_FIELDS == ("lf_vhat_heads",)


# ------------------------------------------------------------------------------------------------ flags, labels
_REQ = ["run", "--ckpt", "c.pt", "--tag", "t", "--tracks", "t.pt", "--eval-grid", "g.json", "--eval-join", "j.npz", "--out-dir", "o"]


def _args(extra):
    return T1.build_parser().parse_args(_REQ + extra)


def test_flag_refusals(tmp_path):
    ck = tmp_path / "f.pt"
    ck.write_bytes(b"x")
    on = ["--level-floor", "on", "--floor-kappa", "0.75", "--floor-ckpt", str(ck)]
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(["--floor-vhat-source", "logged"]))                              # logged without the floor
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(["--floor-vhat-source", "heads"]))                               # an explicit flag with the floor off would silently do nothing
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(["--level-floor", "off", "--floor-vhat-source", "logged"]))
    with pytest.raises(SystemExit):
        _args(on + ["--floor-vhat-source", "oracle"])                                           # not a choice
    a = _args(on)
    assert a.floor_vhat_source is None
    T1.lf_check_args(a)
    assert a.floor_vhat_source == "heads"                                                       # the default under the floor
    b = _args(on + ["--floor-vhat-source", "logged"])
    T1.lf_check_args(b)
    assert b.floor_vhat_source == "logged"
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(on + ["--floor-vhat-source", "logged", "--histswap", "logged"]))     # the LF refusals still apply
    c = _args([])
    T1.lf_check_args(c)                                                                          # floor off, no flag: fine
    assert c.floor_vhat_source is None


def test_header_and_meta_labels():
    lh = T1.LevelFloor(None, 0.75, "x.pt", "0" * 64)
    lo = T1.LevelFloor(None, 0.75, "x.pt", "0" * 64, vhat_source="logged", oracle=lambda s: None)
    assert lh.meta()["floor_vhat_source"] == "heads" and "label" not in lh.meta()
    assert lo.meta()["floor_vhat_source"] == "logged" and lo.meta()["label"] == "CEILING"
    dh, do = lh.detail(), lo.detail()
    assert dh["floor_vhat_source"] == "heads" and dh["vision_only"].startswith("v_hat, presence")
    assert do["floor_vhat_source"] == "logged" and do["label"] == "CEILING" and do["vision_only"].startswith("NO.") and "inadmissible" in do["vision_only"]
    assert do["counts"]["oracle_no_route"] == 0
    with pytest.raises(ValueError):
        T1.LevelFloor(None, 0.75, "x.pt", "0" * 64, vhat_source="oracle")
    code = TOOL.read_text(encoding="utf-8")                                                      # cmd_run's header lines (the real run is checked by the smoke / KA runs)
    assert 'header["floor_vhat_source"] = floor.vhat_source' in code and 'header["label"] = "CEILING"' in code
    assert code.count("floor.vhat_source == \"logged\"") >= 2
    assert hashlib.sha256((TREE / "stack" / "scripts" / "level_floor.py").read_bytes()).hexdigest() == "2781e23af3d0052cb45be2e5895f14ebd56a9779dc7fcd65104a58d5f141a5e7"      # the operator is the LF operator, unchanged
