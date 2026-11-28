"""FEAS3 (H-FEAS3-T1-1): ``--feasible {kbox2, lonfloor, kbox2lf}`` of ``stack/scripts/v8f_t1_eval_feas3.py`` (the FEAS2-T1 tool + the FEAS3 patch).

Run (from a tree that holds stack/ and taniteval/, the tool in stack/scripts/, this file in stack/tests/):
  CUDA_VISIBLE_DEVICES=-1  V8F_T1_FEAS2_TOOL=<the FEAS2-T1 tool, sha256 4b66cc86...>  pytest -q stack/tests/test_v8f_t1_feas3.py
  * the operators (spelled independently here): ``off`` returns the same object; ``lonfloor`` is INACTIVE (returns the same values) when every step sits above the floor, raises only the steps below
    it to exactly the floor, keeps every step heading and re-integrates; a zero-length step takes the prior's direction; ``kbox2`` equals ``project_feasible(min_speed=2.0)`` and equals ``kbox`` on a plan
    whose steps are all above 2 m/s; ``kbox2lf`` = lonfloor(kbox2); lonfloor / kbox2lf refuse without a prior of the plan's shape; unknown modes refuse.
  * C-OFF  ``--feasible off`` and the flag absent give byte-identical dumps, ticks and result JSON, and both equal the FEAS2-T1 tool (sha256 pinned; skipped without ``V8F_T1_FEAS2_TOOL``); the FEAS2 modes
    ``kbox`` / ``abox`` equal the FEAS2 tool's bit for bit (dumps and ticks).
  * REACH  for each new mode the executor commits the operator applied to the tick's plan_s (and P), and the operator changed something (non-vacuous).
  * MUTATION  deleting the operator call turns the reach check RED for every new mode; passing a shifted prior turns lonfloor's reach check RED; setting the margin so the floor never binds turns the non-vacuity check RED.
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
import v8f_t1_eval_feas3 as T1  # noqa: E402

torch.set_num_threads(1)
INF = float("inf")
DT = 0.2
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
FEAS2_TOOL_SHA256 = "4b66cc86cc5afe185736a65000c7f8cb1aa0c89d22bcd32770c99ad7971596df"
CALL = "            return feasible_plan(plan.plan_s, feasible, v0=v0, prior=out[\"p10\"].float().cpu().numpy())"
NEW = ("kbox2", "lonfloor", "kbox2lf")


def sg(plan, prior):
    plan, prior = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    return prior + savgol_filter(plan - prior, 7, 2, axis=1, mode="interp")


def floor_ref(plan, prior, margin=2.0):
    """lonfloor spelled step by step in a Python loop (not the tool's vectorised form)."""
    x, p = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    out = np.zeros_like(x)
    for b in range(x.shape[0]):
        cur, px, pp, active_any, steps = np.zeros(2), np.zeros(2), np.zeros(2), False, []
        for k in range(x.shape[1]):
            dx, dp = x[b, k] - px, p[b, k] - pp
            px, pp = x[b, k], p[b, k]
            sx, sp = np.hypot(*dx) / DT, np.hypot(*dp) / DT
            fl = max(sp - margin, 0.0)
            if sx < fl:
                active_any = True
                u = dx / np.hypot(*dx) if np.hypot(*dx) > 1e-9 else dp / np.hypot(*dp)
                dx = u * fl * DT
            steps.append(dx)
        out[b] = np.cumsum(np.array(steps), 0) if active_any else x[b]
    return out


def pj(plan, mode, prior=None, margin=2.0):
    """The card's operators, spelled out here and not taken from the tool."""
    p = np.asarray(plan, np.float64)
    full = torch.from_numpy(np.concatenate([np.zeros_like(p[:, :1]), p], 1))
    if mode in ("kbox2", "kbox2lf"):
        o = FD.project_feasible(full, dt=DT, a_max=INF, kappa_max=0.2, mu=None, clamp_entry=False, min_speed=2.0)[:, 1:].numpy()
        return o if mode == "kbox2" else floor_ref(o, prior, margin)
    return floor_ref(p, prior, margin)


def arc(v, kap, T=30):
    s = v * DT * np.arange(1, T + 1)
    if kap == 0:
        return np.stack([s, 0 * s], -1)[None]
    return np.stack([np.sin(kap * s) / kap, (1 - np.cos(kap * s)) / kap], -1)[None]


def steps_of(plan):
    full = np.concatenate([np.zeros_like(plan[:, :1]), plan], 1)
    d = np.diff(full, axis=1)
    return np.hypot(d[..., 0], d[..., 1]) / DT, np.arctan2(d[..., 1], d[..., 0])


def planted():
    """[2, 30, 2] plan + prior: row 0 dips to 1 m/s for steps 8-14 under a prior at 8 m/s (floor 6 m/s there), row 1 sits ABOVE its floor everywhere (plan = prior - 1 m/s)."""
    prior = np.concatenate([arc(8.0, 0.01), arc(6.0, 0.0)], 0)
    sp = np.full(30, 8.0)
    sp[7:14] = 1.0
    th = np.cumsum(np.full(30, 0.01))
    r0 = np.cumsum(np.stack([sp * DT * np.cos(th), sp * DT * np.sin(th)], -1), 0)
    r1 = prior[1] * (5.0 / 6.0)
    return np.stack([r0, r1]).astype(np.float32), prior.astype(np.float32)


# ------------------------------------------------------------------------------------------------ the operators
def test_off_returns_the_same_object():
    x = np.random.default_rng(0).normal(size=(2, 30, 2)).astype(np.float32)
    assert T1.feasible_plan(x, "off") is x
    assert T1.feasible_plan(x, "off", v0=np.zeros(2), prior=x) is x


def test_lonfloor_is_inactive_when_the_plan_is_above_the_floor_and_exact_when_it_is_below():
    x, pr = planted()
    y = T1.feasible_plan(x, "lonfloor", prior=pr)
    assert y.dtype == x.dtype and y.shape == x.shape
    assert np.array_equal(y[1], x[1]), "row 1 sits above its floor everywhere: it must come back unchanged (bit for bit)"
    sx, hx = steps_of(x.astype(np.float64))
    sy, hy = steps_of(y.astype(np.float64))
    sp, _ = steps_of(pr.astype(np.float64))
    fl = np.maximum(sp - 2.0, 0.0)
    assert np.allclose(sy[0, 7:14], fl[0, 7:14], atol=2e-5), "the dipped steps are raised to exactly the floor"
    assert (sx[0, 7:14] < fl[0, 7:14] - 1.0).all()
    assert np.allclose(sy[0, :7], sx[0, :7], atol=2e-5) and np.allclose(sy[0, 14:], sx[0, 14:], atol=2e-5), "steps at or above the floor keep their speed"
    assert np.abs(np.angle(np.exp(1j * (hy - hx)))).max() < 2e-5, "every step heading is kept"
    assert np.abs(y - floor_ref(x, pr)).max() < 1e-4, "equals the independent step-by-step spelling"
    assert np.abs(y[0] - x[0]).max() > 1.0, "non-vacuous: the dipped row moved"


def test_lonfloor_zero_length_step_takes_the_prior_direction_and_a_stopped_prior_stays_stopped():
    x, pr = planted()
    x = x.astype(np.float64).copy()
    x[0, 9:11] = x[0, 8]                                           # steps 9 and 10 have zero length in the plan
    y = T1.lon_floor(x, pr.astype(np.float64))
    sy, hy = steps_of(y)
    _, hp = steps_of(pr.astype(np.float64))
    assert np.allclose(sy[0, 9:11], np.maximum(steps_of(pr.astype(np.float64))[0][0, 9:11] - 2.0, 0.0), atol=1e-9)
    assert np.abs(np.angle(np.exp(1j * (hy[0, 9:11] - hp[0, 9:11])))).max() < 1e-9
    stop = np.zeros((1, 30, 2))
    assert np.array_equal(T1.lon_floor(stop, stop), stop)           # prior stopped: floor 0, nothing to do


def test_kbox2_is_kbox_with_the_denominator_floor_lifted_to_2_mps():
    for v, k in ((5.0, 0.0), (5.0, 0.05), (3.0, -0.15), (8.0, 0.02)):
        x = arc(v, k)
        y = T1.feasible_plan(x, "kbox2")
        assert np.isfinite(y).all() and np.abs(y - x).max() < 1e-5, (v, k)
    rng = np.random.default_rng(1)
    zig = arc(5.0, 0.0)[0] + np.stack([0 * np.arange(30.0), 0.25 * np.sign(np.sin(np.arange(30) * 1.3)) * (np.arange(30) > 2)], -1)
    zig = zig[None].astype(np.float32)
    assert np.abs(T1.feasible_plan(zig, "kbox2") - T1.feasible_plan(zig, "kbox")).max() < 1e-5, "all steps >= 2 m/s: kbox2 == kbox"
    assert np.abs(T1.feasible_plan(zig, "kbox2") - zig).max() > 0.01, "and the clamp is active on it"
    slow = arc(1.0, 0.0)[0] + np.stack([0 * np.arange(30.0), 0.03 * np.sign(np.sin(np.arange(30) * 1.3)) * (np.arange(30) > 2)], -1)
    slow = slow[None].astype(np.float32)
    a, b = T1.feasible_plan(slow, "kbox"), T1.feasible_plan(slow, "kbox2")
    assert np.abs(a - b).max() > 1e-3, "on a sub-2 m/s zig-zag the two caps differ"
    assert np.abs(b - slow).max() < np.abs(a - slow).max(), "kbox2 moves a sub-2 m/s plan less than kbox (the cap acts less there)"
    assert np.abs(b - pj(slow, "kbox2")).max() < 1e-5
    x, pr = planted()
    assert np.abs(T1.feasible_plan(x, "kbox2") - pj(x, "kbox2")).max() < 1e-5
    del rng


def test_kbox2lf_is_lonfloor_of_kbox2():
    x, pr = planted()
    zig = x.copy()
    zig[0] += np.stack([0 * np.arange(30.0), 0.2 * np.sign(np.sin(np.arange(30) * 1.3))], -1).astype(np.float32)
    y = T1.feasible_plan(zig, "kbox2lf", prior=pr)
    assert np.abs(y - pj(zig, "kbox2lf", pr.astype(np.float64))).max() < 1e-4
    mid = T1.lon_floor(T1.feasible_plan(zig, "kbox2").astype(np.float64), pr.astype(np.float64))
    assert np.abs(y - mid).max() < 1e-3
    assert np.abs(y - T1.feasible_plan(zig, "kbox2")).max() > 0.1 and np.abs(y - T1.feasible_plan(zig, "lonfloor", prior=pr)).max() > 1e-3


def test_refusals_and_the_flag():
    x, pr = planted()
    for mode in ("lonfloor", "kbox2lf"):
        with pytest.raises(ValueError):
            T1.feasible_plan(x, mode)
        with pytest.raises(ValueError):
            T1.feasible_plan(x, mode, prior=pr[:1])
    with pytest.raises(ValueError):
        T1.feasible_plan(x, "lon")
    with pytest.raises(ValueError):
        T1.lon_floor(x, pr[:, :20])
    base = ["run", "--ckpt", "x", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x"]
    with pytest.raises(SystemExit):
        T1.build_parser().parse_args(base + ["--feasible", "lon"])
    for m in ("off", "box", "kbox", "abox", "kabox") + NEW:
        assert T1.build_parser().parse_args(base + ["--feasible", m]).feasible == m
    assert T1.build_parser().parse_args(base).feasible == "off"
    assert T1.KBOX2_MIN_SPEED_MPS == 2.0 and T1.LONFLOOR_MARGIN_MPS == 2.0 and set(T1.FEASIBLE_OPERATORS) >= set(NEW)


# ------------------------------------------------------------------------------------------------ fixtures (the SMOOTH-T1 / KAPPA-T1 / FEAS2 rig)
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_feas3")
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


def load_feas2_tool():
    p = os.environ.get("V8F_T1_FEAS2_TOOL", "")
    if not p or not Path(p).is_file():
        pytest.skip("V8F_T1_FEAS2_TOOL=<the FEAS2-T1 tool 4b66cc86> not given")
    src = Path(p).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == FEAS2_TOOL_SHA256, "REFUSED: that file is not the FEAS2-T1 tool (sha256 pin)"
    return tool_from_source(src.decode("utf-8"), "v8f_t1_eval_feas2")


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
    for mode in NEW:
        out[mode] = run_cli(world, tmp, mode, "--smooth", "sg72", "--feasible", mode)
    return out


def op(plan_s, prior, mode, margin=2.0):
    return pj(plan_s, mode, prior, margin)


def check_reach(s, x, mode, m=15, margin=2.0, strict=False):
    ts, tx = ticks(s), ticks(x)
    assert "plan_s" in tx and "plan_s" not in ts and str(tx["feasible"]) == mode
    assert np.array_equal(tx["plan_s"][tx["tick"] == 0], ts["plan"][ts["tick"] == 0]), "(1) tick-0 S plan differs from the S-only run's plan"
    assert (np.abs(sg(tx["plan_raw"], tx["P"]) - tx["plan_s"]) <= 1e-5 + 1.5e-7 * np.abs(tx["plan_s"])).all(), "(2) plan_s is not P + SG(plan_raw - P)"
    prior = tx["P"].astype(np.float32).astype(np.float64)
    assert (np.abs(op(tx["plan_s"], prior, mode, margin) - tx["plan"]) <= 1e-4 + 1.5e-7 * np.abs(tx["plan"])).all(), "(3) the issued plan is not the operator applied to plan_s at the tick's prior"
    assert np.abs(tx["plan"] - tx["plan_s"]).max() > 1e-3, "(3b) the operator changed nothing: the check is vacuous"
    k0 = tx["tick"] == 0
    ex = dumps(x, "cl")[:, :m]
    assert np.abs(ex - tx["plan"][k0][:, :m]).max() < 1e-4, "(4) the executed knots are not the operator's plan's"
    if strict or np.abs(tx["plan"][k0][:, :m] - tx["plan_s"][k0][:, :m]).max() > 1e-3:      # lonfloor binds on few steps: (4b) needs a tick-0 window whose first m knots it changed
        assert np.abs(ex - tx["plan_s"][k0][:, :m]).max() > 1e-3, "(4b) the executed knots equal the S plan's: the operator did not reach the executor"
    assert dumps(s, "fl").tobytes() == dumps(x, "fl").tobytes(), "(5) the P10 floor changed"


# ------------------------------------------------------------------------------------------------ the wiring
def test_off_is_bit_for_bit_the_flag_absent_and_the_feas2_tool(runs, world):
    tmp = runs["tmp"]
    explicit = run_cli(world, tmp, "s_off", "--smooth", "sg72", "--feasible", "off")
    assert_same_run(runs["s"], explicit)
    a0 = run_cli(world, tmp, "plain", tag="syn")
    a1 = run_cli(world, tmp, "plain_off", "--feasible", "off")
    assert_same_run(a0, a1)
    tool = load_feas2_tool()
    assert_same_run(run_cli(world, tmp, "ref_s", "--smooth", "sg72", tool=tool), explicit)
    assert_same_run(run_cli(world, tmp, "ref_plain", tool=tool), a1)
    for mode in ("kbox", "abox"):                                    # the FEAS2 modes are the FEAS2 tool's, byte for byte (dumps and ticks)
        ref = run_cli(world, tmp, f"ref_{mode}", "--smooth", "sg72", "--feasible", mode, tool=tool)
        new = run_cli(world, tmp, f"new_{mode}", "--smooth", "sg72", "--feasible", mode)
        assert_same_run(ref, new, header=False)


@pytest.mark.parametrize("mode", ("kbox2", "lonfloor", "kbox2lf"))
def test_new_mode_reaches_the_executor(runs, mode):
    tx = ticks(runs[mode])
    if mode != "kbox2":
        sx, _ = steps_of(tx["plan_s"].astype(np.float64))
        sp, _ = steps_of(tx["P"].astype(np.float64))
        assert (sx < np.maximum(sp - 2.0, 0.0)).any(), "the synthetic rig never puts the S plan below the floor: the lonfloor check would be vacuous (see the mutation tests)"
    check_reach(runs["s"], runs[mode], mode)
    j = json.load(open(runs[mode] / "t1v8f_syn.json"))["t1v8f"]
    assert j["feasible"]["mode"] == mode and j["feasible"]["operator"] == T1.FEASIBLE_OPERATORS[mode] and "PREREG_FEAS3_FROZEN" in j["feasible"]["prereg"]


@pytest.mark.parametrize("mode", NEW)
def test_deleting_the_operator_call_turns_the_reach_check_red(runs, world, mode):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(CALL) == 1
    mutant = tool_from_source(src.replace(CALL, "            return plan.plan_s"), f"v8f_t1_eval_mut_noop_{mode}")
    mut = run_cli(world, runs["tmp"], f"mut_{mode}", "--smooth", "sg72", "--feasible", mode, tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["s"], mut, mode)


def test_a_shifted_prior_turns_lonfloors_reach_check_red(runs, world):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    mutant = tool_from_source(src.replace(CALL, CALL.replace('prior=out["p10"].float().cpu().numpy())', 'prior=out["p10"].float().cpu().numpy() * 0.5)')), "v8f_t1_eval_mut_prior")
    mut = run_cli(world, runs["tmp"], "mut_prior", "--smooth", "sg72", "--feasible", "lonfloor", tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["s"], mut, "lonfloor")


def test_a_margin_that_never_binds_turns_the_non_vacuity_check_red(runs, world):
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count("LONFLOOR_MARGIN_MPS = 2.0 ") == 1
    mutant = tool_from_source(src.replace("LONFLOOR_MARGIN_MPS = 2.0 ", "LONFLOOR_MARGIN_MPS = 500.0 "), "v8f_t1_eval_mut_margin")
    mut = run_cli(world, runs["tmp"], "mut_margin", "--smooth", "sg72", "--feasible", "lonfloor", tool=mutant)
    with pytest.raises(AssertionError):
        check_reach(runs["s"], mut, "lonfloor")


def test_lonfloor_reaches_the_executor_when_the_floor_binds_at_the_executed_knots(runs, world):
    """The synthetic rig's S plans rarely sit below prior - 2 m/s inside the first 15 knots of a tick-0 plan, so the strict (4b) reach is shown with a margin of -3 m/s (floor = prior + 3), which binds everywhere."""
    src = Path(T1.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count("LONFLOOR_MARGIN_MPS = 2.0 ") == 1
    mutant = tool_from_source(src.replace("LONFLOOR_MARGIN_MPS = 2.0 ", "LONFLOOR_MARGIN_MPS = -3.0 "), "v8f_t1_eval_wide_margin")
    for mode in ("lonfloor", "kbox2lf"):
        wide = run_cli(world, runs["tmp"], f"wide_{mode}", "--smooth", "sg72", "--feasible", mode, tool=mutant)
        check_reach(runs["s"], wide, mode, margin=-3.0, strict=True)
