"""FEAS2-T1 (H-FEAS2-T1-1): ``--feasible {off,box,kbox,abox,kabox}`` of ``stack/scripts/v8f_t1_eval.py``.

Run (from a tree that holds stack/ and taniteval/, this file in stack/tests/):
  CUDA_VISIBLE_DEVICES=-1  V8F_T1_KAPPA_TOOL=<the KAPPA-T1 tool, sha256 260bd175...>  pytest -q stack/tests/test_v8f_t1_feas2.py
  * the operator: ``off`` returns the same object; inf bounds run clean (no NaN, no raise); an envelope-feasible plan round-trips (< 1e-5 m) in every mode; kbox bounds curvature
    and leaves speeds alone, abox bounds accel and the entry speed, kabox is abox(kbox(.)); abox / kabox refuse without v0; unknown modes refuse.
  * C-OFF  ``--feasible off`` and the flag absent give byte-identical dumps, ticks and result JSON, and both equal the KAPPA-T1 tool (sha256 pinned; skipped without
    ``V8F_T1_KAPPA_TOOL``); ``--feasible box`` equals the KAPPA-T1 tool's box bit for bit (dumps, ticks).
  * REACH  (replan every 15 knots, so kbox's curvature clamp, which acts from knot 3 on and on this rig only beyond knot 10, is inside the executed knots) for each new mode the executor commits the projected plan: plan == mode(plan_s, tick v0), plan_s == P + SG(plan_raw - P), executed knots == the plan's, floor unchanged,
    the projection changed something (non-vacuous).
  * MUTATION  deleting the projection call turns the reach check RED, for every new mode; passing a wrong v0 turns abox's / kabox's reach check RED.
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
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from tanitad.refs import feasible_decode as FD  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
INF = float("inf")
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
KAPPA_TOOL_SHA256 = "260bd17505726a297afe4691fafe55b06f20af7c41bface7dacf947f5279de93"
CALL = "            return feasible_plan(plan.plan_s, feasible, v0=v0)\n"
NEW = ("kbox", "abox", "kabox")


def sg(plan, prior):
    plan, prior = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    return prior + savgol_filter(plan - prior, 7, 2, axis=1, mode="interp")


def pj(plan, mode, v0=None):
    """The card's operators, spelled out here and not taken from the tool."""
    p = np.asarray(plan, np.float64)
    full = torch.from_numpy(np.concatenate([np.zeros_like(p[:, :1]), p], 1))
    v = None if v0 is None else torch.from_numpy(np.asarray(v0, np.float64))
    if mode == "box":
        o = FD.project_feasible(full, dt=0.2, a_max=4.0, kappa_max=0.2, mu=None, clamp_entry=False)
    if mode in ("kbox", "kabox"):
        o = FD.project_feasible(full, dt=0.2, a_max=INF, kappa_max=0.2, mu=None, clamp_entry=False)
    if mode in ("abox", "kabox"):
        o = FD.project_feasible(full if mode == "abox" else o, v0=v, dt=0.2, a_max=4.0, kappa_max=INF, mu=None, clamp_entry=True)
    return o[:, 1:].numpy()


def arc(v, kap, T=30, dt=0.2):
    """[1, T, 2] constant speed v, constant curvature kap, knots 1..T."""
    s = v * dt * np.arange(1, T + 1)
    if kap == 0:
        return np.stack([s, 0 * s], -1)[None]
    return np.stack([np.sin(kap * s) / kap, (1 - np.cos(kap * s)) / kap], -1)[None]


def controls(plan):
    full = torch.from_numpy(np.concatenate([np.zeros((plan.shape[0], 1, 2)), np.asarray(plan, np.float64)], 1))
    sp, _, acc, lat = FD.recover_controls(full, dt=0.2)
    return sp.numpy(), acc.numpy(), (lat.abs() / sp[..., :-1].clamp_min(0.5).pow(2)).numpy()


def wild():
    """Two planted plans: a high-curvature zig-zag at 5 m/s, and a straight launch + hard accel / brake profile (step speeds 1 -> 9 -> 2 m/s, never a stop)."""
    base = arc(5.0, 0.0)[0]
    a = base + np.stack([0 * base[:, 0], 0.25 * np.sign(np.sin(np.arange(30) * 1.3)) * (np.arange(30) > 2)], -1)
    sp = np.concatenate([np.linspace(1, 9, 5), np.full(13, 9.0), np.linspace(9, 2, 6), np.full(6, 2.0)])
    b = np.stack([np.cumsum(sp * 0.2), np.zeros(30)], -1)
    return np.stack([a, b]).astype(np.float32), np.array([1.0, 3.0])


# ------------------------------------------------------------------------------------------------ the operator
def test_off_returns_the_same_object():
    x = np.random.default_rng(0).normal(size=(2, 30, 2)).astype(np.float32)
    assert T1.feasible_plan(x, "off") is x
    assert T1.feasible_plan(x, "off", v0=np.zeros(2)) is x


def test_inf_bounds_run_clean_and_every_mode_round_trips_a_feasible_plan():
    for v, k in ((5.0, 0.0), (5.0, 0.05), (3.0, -0.15), (8.0, 0.02)):
        x = arc(v, k)
        for mode in ("box",) + NEW:
            y = T1.feasible_plan(x, mode, v0=np.array([v]))
            assert np.isfinite(y).all(), (mode, v, k)
            assert np.abs(y - x).max() < 1e-5, (mode, v, k, float(np.abs(y - x).max()))
    x, v0 = wild()
    for mode in NEW:
        y = T1.feasible_plan(x, mode, v0=v0)
        assert np.isfinite(y).all() and y.shape == x.shape and y.dtype == x.dtype, mode


def test_kbox_bounds_curvature_and_leaves_the_speed_profile_alone():
    x, _ = wild()
    sp0, acc0, _ = controls(x.astype(np.float64))
    assert float(np.abs(acc0).max()) > 4.0 * 1.5                               # the plant violates the accel bound, so a surviving violation below is meaningful
    y = T1.feasible_plan(x, "kbox")
    sp, acc, kap = controls(y)
    assert float(kap.max()) <= 0.2 * (1 + 1e-4)
    assert float(np.abs(acc[1]).max()) > 4.0 * 1.5                             # the accel violation of the launch / brake plan survives (a_max = inf)
    assert np.abs(sp[1] - sp0[1]).max() < 1e-6                                 # and its speed profile is untouched
    assert np.abs(y - pj(x, "kbox")).max() < 1e-5
    # a_max = inf is a REAL different operator from the box: on the accel plant they differ
    assert np.abs(y[1] - T1.feasible_plan(x, "box")[1]).max() > 1.0


def test_abox_bounds_accel_and_the_entry_speed_and_needs_v0():
    x, v0 = wild()
    sp0, acc0, _ = controls(x.astype(np.float64))
    assert abs(sp0[1, 0] - v0[1]) > 4.0 * 0.2 + 0.5                            # the plant's first-step speed is outside v0 +- a_max*dt
    y = T1.feasible_plan(x, "abox", v0=v0)
    sp, acc, kap = controls(y)
    assert float(np.abs(acc).max()) <= 4.0 * (1 + 1e-4)
    assert abs(sp[1, 0] - v0[1]) <= 4.0 * 0.2 * (1 + 1e-4)                     # the entry bound
    assert abs(sp[0, 0] - v0[0]) <= 4.0 * 0.2 * (1 + 1e-4)
    assert np.abs(y - pj(x, "abox", v0)).max() < 1e-5
    with pytest.raises(ValueError):
        T1.feasible_plan(x, "abox")
    with pytest.raises(ValueError):
        T1.feasible_plan(x, "abox", v0=np.zeros(5))
    # clamp_entry is the only difference from aonly: with v0 equal to the plan's own first speed the entry bound is inactive
    own = sp0[:, 0]
    y2 = T1.feasible_plan(x, "abox", v0=own)
    full = torch.from_numpy(np.concatenate([np.zeros((2, 1, 2)), x.astype(np.float64)], 1))
    ref = FD.project_feasible(full, dt=0.2, a_max=4.0, kappa_max=INF, mu=None, clamp_entry=False)[:, 1:].numpy()
    assert np.abs(y2 - ref).max() < 1e-5


def test_kabox_is_abox_of_kbox():
    x, v0 = wild()
    y = T1.feasible_plan(x, "kabox", v0=v0)
    assert np.abs(y - pj(x, "kabox", v0)).max() < 1e-5
    mid = T1.feasible_plan(T1.feasible_plan(x, "kbox"), "abox", v0=v0)       # float32 hop between the two: equal to < 1e-4, the composition is the card's up to the cast
    assert np.abs(y - mid).max() < 1e-3
    sp, acc, _ = controls(y)
    assert float(np.abs(acc).max()) <= 4.0 * (1 + 1e-4) and abs(sp[1, 0] - v0[1]) <= 4.0 * 0.2 * (1 + 1e-4)


def test_refusals():
    for bad in ("lon", "kappa", ""):
        with pytest.raises(ValueError):
            T1.feasible_plan(np.zeros((1, 30, 2)), bad if bad else "nope")
    with pytest.raises(ValueError):
        T1.feasible_plan(np.zeros((30, 2)), "kbox")
    base = ["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(base + ["--feasible", "lon"])
    for m in ("off", "box") + NEW:
        assert T1.build_parser().parse_args(base + ["--feasible", m]).feasible == m
    assert T1.build_parser().parse_args(base).feasible == "off"


# ------------------------------------------------------------------------------------------------ fixtures (the SMOOTH-T1 / KAPPA-T1 rig)
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_feas2")
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
        torch.nn.init.normal_(model.head_o[2].weight, std=0.3)               # a rough O head, so the projections have something to cut
    return model, cfg


def tool_from_source(src: str, name: str):
    mod = types.ModuleType(name)
    mod.__file__ = str(Path(T1.__file__).resolve())
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def load_kappa_tool():
    p = os.environ.get("V8F_T1_KAPPA_TOOL", "")
    if not p or not Path(p).is_file():
        pytest.skip("V8F_T1_KAPPA_TOOL=<the KAPPA-T1 tool 260bd175> not given")
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == KAPPA_TOOL_SHA256, "REFUSED: that file is not the KAPPA-T1 tool (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_kappa")


def run_cli(world, tmp, label, *extra, tool=None, model=None, tag="syn", m=15):
    root = Path(world["root"])
    if model is not None:
        torch.save({"cfg": dataclasses.asdict(model[1]), "model": model[0].state_dict(), "grid": "pool4x10"}, tmp / "ck.pt")
    rc = (tool or T1).main(["run", "--ckpt", str(tmp / "ck.pt"), "--tag", tag, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root),
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


def assert_same_run(a, b, header=True):
    da, db = dumps(a), dumps(b)
    assert da.keys() == db.keys() and len(da) >= 3
    for f in da:
        assert set(da[f]) == set(db[f]) and all(da[f][k].tobytes() == db[f][k].tobytes() and da[f][k].dtype == db[f][k].dtype for k in da[f]), f
    ta, tb = ticks(a), ticks(b)
    assert ta.keys() == tb.keys() and all(ta[k].tobytes() == tb[k].tobytes() for k in ta)
    if header:
        ja, jb = json.load(open(a / "t1v8f_syn.json")), json.load(open(b / "t1v8f_syn.json"))
        assert "feasible" not in ja["t1v8f"] and "feasible" not in jb["t1v8f"]


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    J = D.load_join(Path(world["root"]) / "join_eval.npz")
    model = tiny_model(J)
    out = {"tmp": tmp, "s": run_cli(world, tmp, "s", "--smooth", "sg72", model=model)}
    for mode in ("box",) + NEW:
        out[mode] = run_cli(world, tmp, mode, "--smooth", "sg72", "--feasible", mode)
    return out


def check_reach(s, x, mode, m=15, v0_shift=0.0):
    ts, tx = ticks(s), ticks(x)
    assert "plan_s" in tx and "plan_s" not in ts and str(tx["feasible"]) == mode
    assert np.array_equal(tx["plan_s"][tx["tick"] == 0], ts["plan"][ts["tick"] == 0]), "(1) tick-0 S plan differs from the S-only run's plan"
    assert (np.abs(sg(tx["plan_raw"], tx["P"]) - tx["plan_s"]) <= 1e-5 + 1.5e-7 * np.abs(tx["plan_s"])).all(), "(2) plan_s is not P + SG(plan_raw - P) (float32 half-ulp tolerance)"
    v0 = tx["v0"] + v0_shift
    assert (np.abs(pj(tx["plan_s"], mode, v0) - tx["plan"]) <= 1e-5 + 1.5e-7 * np.abs(tx["plan"])).all(), "(3) the issued plan is not the projection of plan_s at the tick's v0 (float32 half-ulp tolerance)"
    assert np.abs(tx["plan"] - tx["plan_s"]).max() > 1e-3, "(3b) the projection changed nothing: the check is vacuous"
    k0 = tx["tick"] == 0
    ex = dumps(x, "cl")[:, :m]
    assert np.abs(ex - tx["plan"][k0][:, :m]).max() < 1e-4, "(4) the executed knots are not the projected plan's"
    assert np.abs(ex - tx["plan_s"][k0][:, :m]).max() > 1e-3, "(4b) the executed knots equal the S plan's: the projection did not reach the executor"
    assert dumps(s, "fl").tobytes() == dumps(x, "fl").tobytes(), "(5) the P10 floor changed"


# ------------------------------------------------------------------------------------------------ the wiring
def test_off_is_bit_for_bit_the_flag_absent_and_the_kappa_tool(runs, world):
    tmp = runs["tmp"]
    explicit = run_cli(world, tmp, "s_off", "--smooth", "sg72", "--feasible", "off")
    assert_same_run(runs["s"], explicit)
    a0 = run_cli(world, tmp, "plain", tag="syn")
    a1 = run_cli(world, tmp, "plain_off", "--feasible", "off")
    assert_same_run(a0, a1)
    tool = load_kappa_tool()
    assert_same_run(run_cli(world, tmp, "ref_s", "--smooth", "sg72", tool=tool), explicit)
    assert_same_run(run_cli(world, tmp, "ref_plain", tool=tool), a1)
    # box is the KAPPA-T1 arm, unchanged: dumps and ticks byte-identical to the KAPPA-T1 tool's box run
    ref_box = run_cli(world, tmp, "ref_box", "--smooth", "sg72", "--feasible", "box", tool=tool)
    assert_same_run(runs["box"], ref_box, header=False)


@pytest.mark.parametrize("mode", NEW)
def test_new_mode_reaches_the_executor(runs, mode):
    check_reach(runs["s"], runs[mode], mode)
    j = json.load(open(runs[mode] / "t1v8f_syn.json"))["t1v8f"]
    assert j["feasible"]["mode"] == mode and j["feasible"]["operator"] == T1.FEASIBLE_OPERATORS[mode] and "feas2_t1" in j["feasible"]["prereg"]


def test_the_three_modes_issue_different_plans(runs):
    p = {m: ticks(runs[m])["plan"] for m in ("box",) + NEW}
    for a in p:
        for b in p:
            if a < b:
                assert np.abs(p[a] - p[b]).max() > 1e-3, (a, b)


@pytest.mark.parametrize("mode", NEW)
def test_deleting_the_projection_call_turns_the_reach_check_red(runs, world, mode):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(CALL) == 1
    mutant = tool_from_source(src.replace(CALL, "            return plan.plan_s\n"), f"v8f_t1_eval_mut_noproj_{mode}")
    mut = run_cli(world, runs["tmp"], f"mut_{mode}", "--smooth", "sg72", "--feasible", mode, tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["s"], mut, mode)


@pytest.mark.parametrize("mode", ("abox", "kabox"))
def test_a_wrong_v0_turns_the_entry_clamped_reach_check_red(runs, world, mode):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    mutant = tool_from_source(src.replace(CALL, "            return feasible_plan(plan.plan_s, feasible, v0=v0 + 3.0)\n"), f"v8f_t1_eval_mut_v0_{mode}")
    mut = run_cli(world, runs["tmp"], f"mutv0_{mode}", "--smooth", "sg72", "--feasible", mode, tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["s"], mut, mode)
