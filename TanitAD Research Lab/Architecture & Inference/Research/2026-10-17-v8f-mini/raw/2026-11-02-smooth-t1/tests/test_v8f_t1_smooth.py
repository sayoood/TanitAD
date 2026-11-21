"""SMOOTH-T1: ``--smooth {off,sg72}`` of ``stack/scripts/v8f_t1_eval.py`` -- PREREG_V8F_SMOOTH's SG(7, 2) residual smoother applied to the model's plan at every replan, before the executor. CPU only, synthetic world.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts  CUDA_VISIBLE_DEVICES=-1  V8F_T1_TIP_TOOL=<3a516754:stack/scripts/v8f_t1_eval.py> pytest -q stack/tests/test_v8f_t1_smooth.py

What is pinned, and the control that must go red for each:
  * OPERATOR  ``smooth_plan`` is ``P + savgol_filter(plan - P, 7, 2, axis=1, mode="interp")`` (PREREG_V8F_SMOOTH, evaluated here INLINE, not through the tool); a zero residual and a quadratic residual are unchanged; planted
              white noise loses most of its high-frequency curvature energy; ``off`` returns the very object; a bad mode / shape is refused.
  * C-OFF     ``--smooth off`` and the flag absent give byte-identical dumps and result JSON, and both equal the tip blob ``3a516754:stack/scripts/v8f_t1_eval.py`` (sha256 pinned; skipped without ``V8F_T1_TIP_TOOL``).
  * REACH     with ``--smooth sg72`` the plan the executor commits IS the smoothed plan, shown end to end by four independent facts (the same ``check_reach`` the mutation must break):
              (1) tick-0 ``plan_raw`` equals the off run's tick-0 ``plan`` bit for bit (same state), (2) ``plan`` equals ``P + SG(plan_raw - P)`` recomputed offline from the ticks file, (3) the first ``m`` executed knots
              of every window equal the SMOOTHED plan's first ``m`` knots (and differ from the raw ones), (4) the P10 floor ``fl`` is untouched.
  * MUTATION  deleting the one call (``return smooth_plan(...)`` -> ``return raw``) makes ``check_reach`` fail, and so does feeding the executor the RAW plan: a test that cannot go red is not a test.
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
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
#: sha256 of ``git show 3a516754:stack/scripts/v8f_t1_eval.py`` (LF): the tool as it was BEFORE ``--smooth``
TIP_SHA256 = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"
CALL = '        return smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)\n'


def sg(plan, prior):
    """PREREG_V8F_SMOOTH's operator, spelled out here and not taken from the tool."""
    plan, prior = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    return prior + savgol_filter(plan - prior, 7, 2, axis=1, mode="interp")


def kappa_rough(path):
    """High-frequency curvature energy of a path [B, T, 2] sampled every 0.2 s: mean squared first difference of the heading-rate-per-metre series."""
    d = np.diff(path, axis=1)
    psi = np.unwrap(np.arctan2(d[..., 1], d[..., 0]), axis=1)
    ds = np.maximum(np.linalg.norm(d, axis=-1), 1e-6)
    kap = np.diff(psi, axis=1) / ds[:, 1:]
    return float((np.diff(kap, axis=1) ** 2).mean())


# ------------------------------------------------------------------------------------------------ the operator
def test_off_returns_the_same_object():
    x = np.random.default_rng(0).normal(size=(3, 30, 2)).astype(np.float32)
    assert T1.smooth_plan(x, x * 0, "off") is x


def test_operator_is_the_prereg_arithmetic():
    rng = np.random.default_rng(1)
    P = np.cumsum(rng.uniform(0.5, 1.5, (4, 30, 1)) * np.array([1.0, 0.1]), axis=1).astype(np.float32)
    plan = (P + rng.normal(scale=0.3, size=P.shape)).astype(np.float32)
    got = T1.smooth_plan(plan, P, "sg72")
    assert got.dtype == plan.dtype and got.shape == plan.shape
    assert np.abs(got - sg(plan, P)).max() < 2e-6
    assert np.abs(got - plan).max() > 0.05                                  # not a no-op


def test_zero_and_quadratic_residuals_are_unchanged():
    rng = np.random.default_rng(2)
    P = np.cumsum(rng.uniform(0.5, 1.5, (4, 30, 2)), axis=1).astype(np.float32)
    assert np.array_equal(T1.smooth_plan(P, P, "sg72"), P)                  # zero residual: exactly P
    t = np.arange(30)[None, :, None] * 0.2
    quad = (0.3 + 0.2 * t - 0.05 * t ** 2) * np.array([1.0, -0.5])
    plan = (P + quad).astype(np.float32)
    assert np.abs(T1.smooth_plan(plan, P, "sg72") - plan).max() < 1e-5       # SG(., 2) reproduces a quadratic


def test_planted_noise_loses_its_high_frequency_curvature_energy():
    rng = np.random.default_rng(3)
    n, T = 64, 30
    v = rng.uniform(3.0, 8.0, (n, 1))
    P = np.stack([np.cumsum(np.full((n, T), 1.0) * v * 0.2, axis=1), np.zeros((n, T))], -1).astype(np.float32)
    noisy = (P + rng.normal(scale=0.25, size=P.shape)).astype(np.float32)
    sm = T1.smooth_plan(noisy, P, "sg72")
    r0, r1 = kappa_rough(noisy), kappa_rough(sm)
    assert r1 < 0.25 * r0, (r0, r1)
    rms = lambda a: float(np.sqrt((np.asarray(a, np.float64) ** 2).mean()))    # noqa: E731
    assert rms(sm - P) < 0.8 * rms(noisy - P)                               # white-noise amplitude: about 0.58x in the interior of SG(7, 2), more at the interp edges


def test_refusals():
    x = np.zeros((2, 30, 2), np.float32)
    with pytest.raises(ValueError):
        T1.smooth_plan(x, x, "sg11")
    with pytest.raises(ValueError):
        T1.smooth_plan(x, x[:, :20], "sg72")
    with pytest.raises(ValueError):
        T1.smooth_plan(x[:, :5], x[:, :5], "sg72")
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x", "--smooth", "sg11"])
    assert T1.build_parser().parse_args(["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]).smooth == "off"


# ------------------------------------------------------------------------------------------------ fixtures (the T1-HEADS rig)
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_smooth")
    w = D.make_synthetic_world(root, n_train=2, n_eval=3, T=90, seed=5)
    return {"root": root, **w}


def setup(world):
    root = Path(world["root"])
    tracks = D.load_native_tracks(root / "native_tracks.pt")
    J = D.load_join(root / "join_eval.npz")
    return tracks, J


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
        torch.nn.init.normal_(model.head_o[2].weight, std=0.3)               # a rough O head: the residual over P has high-frequency content, so S has something to remove
    return model, cfg


def write_ckpt(path, model, cfg):
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, path)


def tool_from_source(src: str, name: str):
    mod = types.ModuleType(name)
    mod.__file__ = str(Path(T1.__file__).resolve())
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def load_tip_tool():
    p = os.environ.get("V8F_T1_TIP_TOOL", "")
    if not p or not Path(p).is_file():
        pytest.skip("V8F_T1_TIP_TOOL=<3a516754:stack/scripts/v8f_t1_eval.py> not given")
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == TIP_SHA256, "REFUSED: that file is not the tip blob of v8f_t1_eval.py (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_tip")


def run_cli(world, tmp, label, *extra, tool=None, model=None, tag="syn", m=5):
    root = Path(world["root"])
    if model is not None:
        write_ckpt(tmp / "ck.pt", *model)
    rc = (tool or T1).main(["run", "--ckpt", str(tmp / "ck.pt"), "--tag", tag, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root),
                            "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(tmp / label),
                            "--replan-steps", str(m), "--batch", "6", "--max-windows", "12", "--n-boot", "20", "--device", "cpu", *extra])
    assert rc == 0
    return tmp / label


def dump_arrays(out, tag="syn"):
    return {f.name: {k: np.load(f)[k] for k in np.load(f).files} for f in sorted((out / f"dump_{tag}").iterdir())}


def assert_same_dumps(a, b):
    da, db = dump_arrays(a), dump_arrays(b)
    assert da.keys() == db.keys() and len(da) >= 3
    for f in da:
        assert set(da[f]) == set(db[f]), f
        for k in da[f]:
            x, y = da[f][k], db[f][k]
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (f, k)


def ticks(out, tag="syn"):
    z = np.load(out / "ticks" / f"ticks_{tag}.npz")
    return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    tracks, J = setup(world)
    model = tiny_model(J)
    off = run_cli(world, tmp, "off", "--dump-ticks", str(tmp / "off" / "ticks"), model=model)
    on = run_cli(world, tmp, "on", "--smooth", "sg72", "--dump-ticks", str(tmp / "on" / "ticks"))
    return {"tmp": tmp, "model": model, "off": off, "on": on, "world": world}


def exec_first_m(out, tag="syn", m=5):
    """Per window (ep file order, window order within the file): the executed knots 1..m of the window, in the t0 ego frame (``cl`` of the dump)."""
    return np.concatenate([np.load(f)["cl"][:, :m] for f in sorted((out / f"dump_{tag}").iterdir())])


def check_reach(off, on, m=5):
    """The four independent facts that the executor's plan IS the smoothed plan. Raises AssertionError on any."""
    t_off, t_on = ticks(off), ticks(on)
    assert "plan_raw" in t_on and "plan_raw" not in t_off and str(t_on["smooth"]) == "sg72"
    k0 = t_off["tick"] == 0
    assert np.array_equal(t_on["plan_raw"][t_on["tick"] == 0], t_off["plan"][k0]), "(1) tick-0 raw plan differs from the off run's plan"
    rec = sg(t_on["plan_raw"], t_on["P"])
    assert np.abs(rec - t_on["plan"]).max() < 1e-5, "(2) the recorded plan is not P + SG(plan_raw - P)"
    assert np.abs(t_on["plan"] - t_on["plan_raw"]).max() > 0.02, "(2b) S changed nothing: the check is vacuous"
    sm0, raw0 = t_on["plan"][t_on["tick"] == 0][:, :m], t_on["plan_raw"][t_on["tick"] == 0][:, :m]
    ex = exec_first_m(on, m=m)
    assert np.abs(ex - sm0).max() < 1e-4, "(3) the executed knots are not the smoothed plan's"
    assert np.abs(ex - raw0).max() > 1e-3, "(3b) the executed knots equal the RAW plan's: S did not reach the executor"
    fl_off = np.concatenate([np.load(f)["fl"] for f in sorted((off / "dump_syn").iterdir())])
    fl_on = np.concatenate([np.load(f)["fl"] for f in sorted((on / "dump_syn").iterdir())])
    assert fl_off.tobytes() == fl_on.tobytes(), "(4) the P10 floor changed"


# ------------------------------------------------------------------------------------------------ the wiring
def test_off_is_bit_for_bit_the_flag_absent_and_the_tip_tool(runs, world):
    tmp, model = runs["tmp"], runs["model"]
    absent = run_cli(world, tmp, "absent", "--dump-ticks", str(tmp / "absent" / "ticks"))
    explicit = runs["off"]
    assert_same_dumps(absent, explicit)
    ta, te = ticks(absent), ticks(explicit)
    assert ta.keys() == te.keys() and "plan_raw" not in ta and "smooth" not in ta
    assert all(ta[k].tobytes() == te[k].tobytes() for k in ta)
    ja, je = json.load(open(absent / "t1v8f_syn.json")), json.load(open(explicit / "t1v8f_syn.json"))
    assert "smooth" not in ja["t1v8f"] and "smooth" not in je["t1v8f"]
    tip = load_tip_tool()
    ref = run_cli(world, tmp, "tip", "--dump-ticks", str(tmp / "tip" / "ticks"), tool=tip)
    assert_same_dumps(ref, explicit)


def test_sg72_reaches_the_executor(runs):
    check_reach(runs["off"], runs["on"])
    j = json.load(open(runs["on"] / "t1v8f_syn.json"))["t1v8f"]
    assert j["smooth"]["mode"] == "sg72" and "savgol_filter(plan - P, 7, 2" in j["smooth"]["operator"]


def test_sg72_changes_the_executed_path_and_the_one_shot_plan(runs):
    off_cl, on_cl = (np.concatenate([np.load(f)["cl"] for f in sorted((runs[k] / "dump_syn").iterdir())]) for k in ("off", "on"))
    assert np.abs(off_cl - on_cl).max() > 1e-3
    off_ol, on_ol = (np.concatenate([np.load(f)["ol"] for f in sorted((runs[k] / "dump_syn").iterdir())]) for k in ("off", "on"))
    assert np.abs(off_ol - on_ol).max() > 1e-3                              # ol = the tick-0 plan: S applies to it as well


# ------------------------------------------------------------------------------------------------ the mutation: the check must go RED
def test_deleting_the_smoother_call_turns_the_reach_check_red(runs, world):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(CALL) == 1
    mutant = tool_from_source(src.replace(CALL, "        return raw\n"), "v8f_t1_eval_mut_nocall")
    mut = run_cli(world, runs["tmp"], "mut", "--smooth", "sg72", "--dump-ticks", str(runs["tmp"] / "mut" / "ticks"), tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["off"], mut)


def test_feeding_the_executor_the_raw_plan_turns_the_reach_check_red(runs, world):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    old = "                st.commit_plan(pl.astype(np.float64), m)\n"
    assert src.count(old) == 1
    mutant = tool_from_source(src.replace(old, "                st.commit_plan(plan_raw_for_exec[st_i].astype(np.float64), m)\n")
                              .replace("        for st, pl in zip(states, plan):\n", "        plan_raw_for_exec = planner.plan_raw if getattr(planner, 'plan_raw', None) is not None else plan\n        for st_i, (st, pl) in enumerate(zip(states, plan)):\n"),
                              "v8f_t1_eval_mut_rawexec")
    mut = run_cli(world, runs["tmp"], "mut2", "--smooth", "sg72", "--dump-ticks", str(runs["tmp"] / "mut2" / "ticks"), tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["off"], mut)
