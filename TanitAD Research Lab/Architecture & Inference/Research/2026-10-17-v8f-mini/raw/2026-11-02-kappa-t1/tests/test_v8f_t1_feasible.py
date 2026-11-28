"""KAPPA-T1 (H-KAPPA-T1-1): ``--feasible {off,box}`` of ``stack/scripts/v8f_t1_eval.py``.

Run: CUDA_VISIBLE_DEVICES=-1  V8F_T1_SMOOTH_TOOL=<the SMOOTH-T1 tool, sha256 caa71055...>  pytest -q stack/tests/test_v8f_t1_feasible.py
  * the operator: ``off`` returns the same object; an envelope-feasible plan round-trips (< 1e-5 m); a planted high-curvature plan comes back inside the box in the
    projection's own convention (``recover_controls``); refusals.
  * C-OFF  ``--feasible off`` and the flag absent give byte-identical dumps, ticks and result JSON, and both equal the SMOOTH-T1 tool (sha256 pinned; skipped without
    ``V8F_T1_SMOOTH_TOOL``), with and without ``--smooth sg72``.
  * REACH  the executor commits the projected plan: plan == project(plan_s), plan_s == P + SG(plan_raw - P), executed knots == the projected plan's, floor unchanged.
  * MUTATION  deleting the projection call turns the reach check RED.
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
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
SMOOTH_TOOL_SHA256 = "caa71055f50f8d1bc300c329374af71abdd68ee7c41bef840cb60fdbe599b183"
CALL = "            return feasible_plan(plan.plan_s, feasible)\n"


def sg(plan, prior):
    plan, prior = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    return prior + savgol_filter(plan - prior, 7, 2, axis=1, mode="interp")


def project(plan):
    """The card's operator, spelled out here and not taken from the tool."""
    p = np.asarray(plan, np.float64)
    full = torch.from_numpy(np.concatenate([np.zeros_like(p[:, :1]), p], 1))
    return FD.project_feasible(full, dt=0.2, a_max=4.0, kappa_max=0.2, mu=None, clamp_entry=False)[:, 1:].numpy()


def arc(v, kap, T=30, dt=0.2):
    """[1, T, 2] constant speed v, constant curvature kap, knots 1..T."""
    s = v * dt * np.arange(1, T + 1)
    if kap == 0:
        return np.stack([s, 0 * s], -1)[None]
    return np.stack([np.sin(kap * s) / kap, (1 - np.cos(kap * s)) / kap], -1)[None]


# ------------------------------------------------------------------------------------------------ the operator
def test_off_returns_the_same_object():
    x = np.random.default_rng(0).normal(size=(2, 30, 2)).astype(np.float32)
    assert T1.feasible_plan(x, "off") is x


def test_feasible_plan_round_trips():
    for v, k in ((5.0, 0.0), (5.0, 0.05), (3.0, -0.15)):
        x = arc(v, k)
        y = T1.feasible_plan(x, "box")
        assert np.abs(y - x).max() < 1e-5, (v, k, float(np.abs(y - x).max()))


def test_planted_high_curvature_comes_back_inside_the_box():
    rng = np.random.default_rng(1)
    base = arc(5.0, 0.0)[0]
    x = np.stack([base + np.stack([0 * base[:, 0], 0.25 * np.sign(np.sin(np.arange(30) * 1.3)) * (np.arange(30) > 2)], -1),
                  base + np.stack([0 * base[:, 0], rng.normal(0, 0.15, 30)], -1)]).astype(np.float32)
    full0 = torch.from_numpy(np.concatenate([np.zeros((2, 1, 2)), x.astype(np.float64)], 1))
    sp0, _, _, lat0 = FD.recover_controls(full0, dt=0.2)
    assert float((lat0.abs() / sp0[..., :-1].clamp_min(0.5).pow(2)).max()) > 0.5          # the plant is far outside the box
    y = T1.feasible_plan(x, "box")
    assert y.dtype == x.dtype and y.shape == x.shape
    full = torch.from_numpy(np.concatenate([np.zeros((2, 1, 2)), y.astype(np.float64)], 1))
    sp, _, acc, lat = FD.recover_controls(full, dt=0.2)
    kap = lat.abs() / sp[..., :-1].clamp_min(0.5).pow(2)
    assert float(kap.max()) <= 0.2 * (1 + 1e-4) and float(acc.abs().max()) <= 4.0 * (1 + 1e-4)
    assert np.abs(y - project(x)).max() < 1e-5


def test_refusals():
    with pytest.raises(ValueError):
        T1.feasible_plan(np.zeros((1, 30, 2)), "kappa")
    with pytest.raises(ValueError):
        T1.feasible_plan(np.zeros((30, 2)), "box")
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x", "--feasible", "lon"])
    assert T1.build_parser().parse_args(["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]).feasible == "off"


# ------------------------------------------------------------------------------------------------ fixtures (the SMOOTH-T1 rig)
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_feasible")
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
        torch.nn.init.normal_(model.head_o[2].weight, std=0.3)               # a rough O head, so the projection has something to cut
    return model, cfg


def tool_from_source(src: str, name: str):
    mod = types.ModuleType(name)
    mod.__file__ = str(Path(T1.__file__).resolve())
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def load_smooth_tool():
    p = os.environ.get("V8F_T1_SMOOTH_TOOL", "")
    if not p or not Path(p).is_file():
        pytest.skip("V8F_T1_SMOOTH_TOOL=<the SMOOTH-T1 tool caa71055> not given")
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == SMOOTH_TOOL_SHA256, "REFUSED: that file is not the SMOOTH-T1 tool (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_smooth")


def run_cli(world, tmp, label, *extra, tool=None, model=None, tag="syn", m=5):
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


def assert_same_run(a, b):
    da, db = dumps(a), dumps(b)
    assert da.keys() == db.keys() and len(da) >= 3
    for f in da:
        assert set(da[f]) == set(db[f]) and all(da[f][k].tobytes() == db[f][k].tobytes() and da[f][k].dtype == db[f][k].dtype for k in da[f]), f
    ta, tb = ticks(a), ticks(b)
    assert ta.keys() == tb.keys() and all(ta[k].tobytes() == tb[k].tobytes() for k in ta)
    ja, jb = json.load(open(a / "t1v8f_syn.json")), json.load(open(b / "t1v8f_syn.json"))
    assert "feasible" not in ja["t1v8f"] and "feasible" not in jb["t1v8f"]


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    J = D.load_join(Path(world["root"]) / "join_eval.npz")
    model = tiny_model(J)
    s = run_cli(world, tmp, "s", "--smooth", "sg72", model=model)
    box = run_cli(world, tmp, "box", "--smooth", "sg72", "--feasible", "box")
    return {"tmp": tmp, "s": s, "box": box}


def check_reach(s, box, m=5):
    ts, tb = ticks(s), ticks(box)
    assert "plan_s" in tb and "plan_s" not in ts and str(tb["feasible"]) == "box"
    assert np.array_equal(tb["plan_s"][tb["tick"] == 0], ts["plan"][ts["tick"] == 0]), "(1) tick-0 S plan differs from the S-only run's plan"
    assert np.abs(sg(tb["plan_raw"], tb["P"]) - tb["plan_s"]).max() < 1e-5, "(2) plan_s is not P + SG(plan_raw - P)"
    assert np.abs(project(tb["plan_s"]) - tb["plan"]).max() < 1e-5, "(3) the issued plan is not the projection of plan_s"
    assert np.abs(tb["plan"] - tb["plan_s"]).max() > 1e-3, "(3b) the projection changed nothing: the check is vacuous"
    k0 = tb["tick"] == 0
    ex = dumps(box, "cl")[:, :m]
    assert np.abs(ex - tb["plan"][k0][:, :m]).max() < 1e-4, "(4) the executed knots are not the projected plan's"
    assert np.abs(ex - tb["plan_s"][k0][:, :m]).max() > 1e-3, "(4b) the executed knots equal the S plan's: the projection did not reach the executor"
    assert dumps(s, "fl").tobytes() == dumps(box, "fl").tobytes(), "(5) the P10 floor changed"


# ------------------------------------------------------------------------------------------------ the wiring
def test_off_is_bit_for_bit_the_flag_absent_and_the_smooth_tool(runs, world):
    tmp = runs["tmp"]
    explicit = run_cli(world, tmp, "s_off", "--smooth", "sg72", "--feasible", "off")
    assert_same_run(runs["s"], explicit)
    a0 = run_cli(world, tmp, "plain", tag="syn")
    a1 = run_cli(world, tmp, "plain_off", "--feasible", "off")
    assert_same_run(a0, a1)
    tool = load_smooth_tool()
    assert_same_run(run_cli(world, tmp, "ref_s", "--smooth", "sg72", tool=tool), explicit)
    assert_same_run(run_cli(world, tmp, "ref_plain", tool=tool), a1)


def test_box_reaches_the_executor(runs):
    check_reach(runs["s"], runs["box"])
    j = json.load(open(runs["box"] / "t1v8f_syn.json"))["t1v8f"]
    assert j["feasible"]["mode"] == "box" and "project_feasible" in j["feasible"]["operator"]


def test_deleting_the_projection_call_turns_the_reach_check_red(runs, world):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(CALL) == 1
    mutant = tool_from_source(src.replace(CALL, "            return plan.plan_s\n"), "v8f_t1_eval_mut_noproj")
    mut = run_cli(world, runs["tmp"], "mut", "--smooth", "sg72", "--feasible", "box", tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["s"], mut)
