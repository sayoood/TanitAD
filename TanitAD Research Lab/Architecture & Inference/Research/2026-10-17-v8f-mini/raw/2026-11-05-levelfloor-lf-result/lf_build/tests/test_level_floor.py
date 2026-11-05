"""Unit tests of the LEVEL-FLOOR operator F (PREREG_LEVELFLOOR_A3 section 2) and of its place in the RI tool (``--level-floor``).

Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/motion_c/lf_read/pt/run <this file>
  LF_TREE      the read tree holding stack/ + taniteval/ with the patched tool and ``level_floor.py`` in stack/scripts/ (default F:/TanitAD-scratch/motion_c/lf_read/tree)
  LF_OPERATOR  the operator under test (default ``../code/level_floor.py``); ``code/run_mutations_lf.py`` points it at a mutated copy
  LF_TOOL      the patched tool under test (default ``<LF_TREE>/stack/scripts/v8f_t1_eval_ri.py``); the mutation runner points it at a mutated copy (with its own level_floor.py beside it)
Cases:
  (a) KA  kappa = 0 is bit-identical to off (analytic identity: s_f = 0 <= s_k; the plan object comes back)
  (b) KB  a plan already faster than the floor comes back as the SAME object (single and batch); a mixed batch leaves the fast rows untouched bit for bit
  (c) KC  v0 = 0, v_hat = 10, kappa 0.5, no lead: the progress equals the closed-form ramp (0.5 t^2 to 5 m/s at t = 5 s, then linear) to 1e-9; the same for v0 > 0; ramp_progress equals a fine numerical integral
  (d) KD  the cap: pi 0.9, g 8 m -> v_f = 4.0 m/s; pi 0.4 does not cap; pi == 0.5 caps; g <= 2 m gives v_f = 0 (no bind)
  (e) KE  v_hat 2.9 gives no floor; v_hat 3.0 does
  (f) KF  the resampled points lie on the polyline or its extension (perpendicular distance < 1e-9), unchanged knots keep their value, the progress never drops below the plan's
  (g) KG  the argparse refusals of the tool and the checkpoint cfg refusals
  (h) KH  vision only: the floor model reads ``feats`` / ``frame_ok`` and nothing else; every other batch key (p10f, nav, ...) leaves the heads bit-identical; F reads only v_hat / pi / gap from the heads and p10f[:, -1, 0]
  (i) KI  placement in ``model_planner`` (A3_RULING_1: F after S in EVERY branch): bare arm (``--smooth sg72``, no ``--feasible``) = F(S(raw)); composite (sg72 + kbox) = kbox(F(S(raw))); smooth off = F(raw); kappa 0 == the unpatched
          planner bit for bit; plan_raw is the raw plan; off == the original tool (model_planner, TickRecorder)
"""
import dataclasses
import hashlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
TREE = Path(os.environ.get("LF_TREE", "F:/TanitAD-scratch/motion_c/lf_read/tree"))
OPERATOR = Path(os.environ.get("LF_OPERATOR", str(CODE / "level_floor.py")))
for _p in (TREE / "stack", TREE / "taniteval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import torch  # noqa: E402
import tanitad  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(TREE.resolve()), ("tanitad is not from the test tree", tanitad.__file__)
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(1)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OP = _load("level_floor_under_test", OPERATOR)
TOOL = Path(os.environ.get("LF_TOOL", str(TREE / "stack" / "scripts" / "v8f_t1_eval_ri.py")))
T1 = _load("v8f_t1_eval_lf_under_test", TOOL)
T = OP.T_KNOTS


def straight(v, n=30, heading=0.0):
    """A straight plan at constant speed v (m/s) along ``heading``."""
    s = v * T[:n]
    return np.stack([s * math.cos(heading), s * math.sin(heading)], 1)


def ramp_ref(v_f, v0, t):
    v0 = max(v0, 0.0)
    if v0 >= v_f:
        return v_f * t
    ts = (v_f - v0) / 1.0
    return np.where(t <= ts, v0 * t + 0.5 * t * t, v0 * ts + 0.5 * ts * ts + v_f * (t - ts))


def chord_progress(plan):
    full = np.concatenate([np.zeros((1, 2)), np.asarray(plan, np.float64)], 0)
    return np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))


def dist_to_polyline_or_ext(pt, plan):
    """Independent geometry: distance from ``pt`` to the origin-prepended polyline, or to the ray beyond its end along the last non-zero segment."""
    full = np.concatenate([np.zeros((1, 2)), np.asarray(plan, np.float64)], 0)
    best = np.inf
    last_d = None
    for a, b in zip(full[:-1], full[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 > 0:
            last_d = ab / math.sqrt(L2)
            u = min(max(float((pt - a) @ ab) / L2, 0.0), 1.0)
        else:
            u = 0.0
        best = min(best, float(np.linalg.norm(pt - (a + u * ab))))
    if last_d is not None:
        w = pt - full[-1]
        u = max(float(w @ last_d), 0.0)
        best = min(best, float(np.linalg.norm(w - u * last_d)))
    return best


# ------------------------------------------------------------------------------------------------ (a)
def test_a_kappa_zero_is_off_identity():
    rng = np.random.default_rng(0)
    for _ in range(40):
        plan = straight(rng.uniform(0.0, 8.0), heading=rng.uniform(-1, 1)) + rng.normal(0, 0.01, (30, 2))
        for v_hat in (3.0, 7.5, 20.0):
            for pi, gap in ((0.1, 0.9), (0.9, 0.2), (0.9, 0.01), (0.6, 1.0)):
                out, info = OP.apply_floor(plan, rng.uniform(0, 15), v_hat, pi, gap, 0.0)
                assert out is plan and info["bind"] is False and info["lift"] == 0.0
    b = np.stack([straight(0.2), straight(3.0)]).astype(np.float32)
    out, info = OP.apply_floor_batch(b, [0.0, 5.0], [9.0, 9.0], [0.2, 0.9], [0.9, 0.3], 0.0)
    assert out is b and not info["bind"].any() and (info["vf"] == 0.0).all()
    out_on, info_on = OP.apply_floor_batch(b, [0.0, 5.0], [9.0, 9.0], [0.2, 0.9], [0.9, 0.3], 0.5)       # control: a positive dose does move this batch
    assert out_on is not b and info_on["bind"][0]


# ------------------------------------------------------------------------------------------------ (b)
def test_b_faster_plan_is_the_same_object():
    fast = straight(20.0)
    out, info = OP.apply_floor(fast, 10.0, 10.0, 0.1, 0.5, 0.5)           # v_f = 5 m/s << 20 m/s
    assert out is fast and info["bind"] is False and info["vf"] == 5.0
    out, info = OP.apply_floor(fast, 10.0, 10.0, 0.1, 0.5, 1.0)           # even the estimate itself (kappa 1)
    assert out is fast and info["bind"] is False
    b = np.stack([fast, fast]).astype(np.float32)
    out, info = OP.apply_floor_batch(b, [10.0, 0.0], [10.0, 10.0], [0.0, 0.0], [0.5, 0.5], 0.75)
    assert out is b and not info["bind"].any()
    mixed = np.stack([fast, straight(0.05), fast]).astype(np.float32)
    out, info = OP.apply_floor_batch(mixed, [10, 0, 10], [10, 10, 10], [0, 0, 0], [0.5] * 3, 0.5)
    assert out is not mixed and info["bind"].tolist() == [False, True, False]
    assert np.array_equal(out[0], mixed[0]) and np.array_equal(out[2], mixed[2]) and out.dtype == np.float32
    assert not np.array_equal(out[1], mixed[1])
    assert np.array_equal(mixed[1], np.stack([straight(0.05)]).astype(np.float32)[0])                    # the input is never mutated


# ------------------------------------------------------------------------------------------------ (c)
def test_c_closed_form_ramp():
    slow = straight(0.01)
    out, info = OP.apply_floor(slow, 0.0, 10.0, 0.1, 0.5, 0.5)
    assert info["bind"] and info["vf"] == 5.0 and not info["cap"]
    prog = chord_progress(out)
    ref = np.where(T <= 5.0, 0.5 * T * T, 12.5 + 5.0 * (T - 5.0))
    assert np.abs(prog - ref).max() < 1e-9, np.abs(prog - ref).max()
    assert abs(prog[24] - 12.5) < 1e-9 and abs(prog[29] - 17.5) < 1e-9
    assert np.abs(out[:, 1]).max() == 0.0                                  # the path geometry is the plan's own (a straight +x line)
    out32, _ = OP.apply_floor(slow.astype(np.float32), 0.0, 10.0, 0.1, 0.5, 0.5)
    assert out32.dtype == np.float32 and np.abs(chord_progress(out32) - ref).max() < 1e-5
    for v0, v_f_kappa in ((3.0, 0.5), (4.9, 0.5), (7.0, 0.5), (-2.0, 0.75)):                              # v0 > 0, v0 ~ v_f, v0 > v_f, v0 < 0 (reads as 0)
        v_hat = 10.0
        out, info = OP.apply_floor(slow, v0, v_hat, 0.0, 0.5, v_f_kappa)
        ref = ramp_ref(v_f_kappa * v_hat, v0, T)
        got = chord_progress(out)
        want = np.maximum(chord_progress(slow), ref)
        assert np.abs(got - want).max() < 1e-9, (v0, np.abs(got - want).max())
    rng = np.random.default_rng(1)                                         # ramp_progress against a fine trapezoid integral of min(v_f, max(v0, 0) + t)
    for _ in range(20):
        v_f, v0 = rng.uniform(0, 12), rng.uniform(-3, 14)
        fine = np.linspace(0.0, 6.0, 600001)
        sp = np.minimum(v_f, max(v0, 0.0) + fine)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(fine))])
        num = np.interp(T, fine, cum)
        assert np.abs(OP.ramp_progress([v_f], [v0])[0] - num).max() < 1e-6
    assert np.array_equal(OP.ramp_progress([0.0], [5.0])[0], np.zeros(30))


# ------------------------------------------------------------------------------------------------ (d)
def test_d_lead_cap():
    vf, band, cap = OP.floor_speed([10.0], [0.9], [0.2], 0.5)               # g = 40 * 0.2 = 8 m -> (8 - 2) / 1.5 = 4.0
    assert band[0] and cap[0] and abs(vf[0] - 4.0) < 1e-12
    vf, band, cap = OP.floor_speed([10.0], [0.4], [0.2], 0.5)               # pi < 0.5: no cap
    assert not cap[0] and vf[0] == 5.0
    vf, _, cap = OP.floor_speed([10.0], [0.5], [0.2], 0.5)                  # pi >= 0.5 inclusive
    assert cap[0] and abs(vf[0] - 4.0) < 1e-12
    vf, _, cap = OP.floor_speed([10.0], [0.9], [0.8], 0.5)                  # cap speed (32 - 2) / 1.5 = 20 > 5: the cap is in force but does not lower v_f
    assert not cap[0] and vf[0] == 5.0
    vf, _, cap = OP.floor_speed([10.0], [0.9], [0.04], 0.5)                 # g = 1.6 m <= 2 m: max(0, .) = 0
    assert cap[0] and vf[0] == 0.0
    out, info = OP.apply_floor(straight(0.01), 0.0, 10.0, 0.9, 0.04, 0.5)   # v_f = 0 binds nothing: the plan comes back
    assert info["bind"] is False and info["vf"] == 0.0
    slow = straight(0.01)                                                   # the capped plan's progress is the ramp to 4.0 m/s
    out, info = OP.apply_floor(slow, 0.0, 10.0, 0.9, 0.2, 0.5)
    assert info["cap"] and np.abs(chord_progress(out) - ramp_ref(4.0, 0.0, T)).max() < 1e-9
    out2, info2 = OP.apply_floor(slow, 0.0, 10.0, 0.4, 0.2, 0.5)
    assert not info2["cap"] and np.abs(chord_progress(out2) - ramp_ref(5.0, 0.0, T)).max() < 1e-9
    _, _, cap = OP.floor_speed([2.0], [0.9], [0.2], 0.5)                    # no floor => no cap either
    assert not cap[0]
    assert abs(OP.sigmoid(np.array([0.0]))[0] - 0.5) == 0.0 and OP.sigmoid(np.array([-800.0]))[0] == 0.0 and OP.sigmoid(np.array([800.0]))[0] == 1.0


# ------------------------------------------------------------------------------------------------ (e)
def test_e_band():
    slow = straight(0.01)
    out, info = OP.apply_floor(slow, 0.0, 2.9, 0.1, 0.5, 0.75)
    assert out is slow and info["bind"] is False and math.isnan(info["vf"]) and info["band"] is False
    out, info = OP.apply_floor(slow, 0.0, 2.999999, 0.1, 0.5, 1.0)
    assert out is slow and math.isnan(info["vf"])
    out, info = OP.apply_floor(slow, 0.0, 3.0, 0.1, 0.5, 1.0)
    assert info["bind"] and info["vf"] == 3.0 and info["band"] is True
    for bad in (float("nan"), float("inf"), -5.0):                         # non-finite / negative estimates never floor
        out, info = OP.apply_floor(slow, 0.0, bad, 0.1, 0.5, 1.0)
        if bad != float("inf"):
            assert out is slow and not info["bind"]
    nanplan = straight(0.01)
    nanplan[3] = np.nan
    out, info = OP.apply_floor(nanplan, 0.0, 10.0, 0.1, 0.5, 0.5)           # a non-finite plan never binds
    assert out is nanplan and not info["bind"]


# ------------------------------------------------------------------------------------------------ (f)
def _curved(rng, v):
    th = np.cumsum(rng.normal(0, 0.06, 30)) + rng.uniform(-0.5, 0.5)
    step = v * 0.2 * (1 + 0.1 * rng.normal(size=30))
    return np.cumsum(np.stack([step * np.cos(th), step * np.sin(th)], 1), axis=0)


def test_f_points_on_polyline_or_extension():
    rng = np.random.default_rng(2)
    n_bound = n_ext = 0
    for i in range(60):
        plan = _curved(rng, rng.uniform(0.05, 2.5))
        v0 = rng.uniform(-1, 9)
        out, info = OP.apply_floor(plan, v0, rng.uniform(3, 20), rng.uniform(0, 1), rng.uniform(0, 1), rng.choice([0.5, 0.75]))
        if out is plan:
            continue
        n_bound += 1
        full = np.concatenate([np.zeros((1, 2)), plan], 0)
        end_s = chord_progress(plan)[-1]
        assert chord_progress(out)[-1] <= OP.ramp_progress([info["vf"]], [v0])[0, -1] + 1e-9
        for k in range(30):
            assert dist_to_polyline_or_ext(out[k], plan) < 1e-9, (i, k, dist_to_polyline_or_ext(out[k], plan))
        n_ext += int(np.linalg.norm(out[-1]) > 0 and chord_progress(out)[-1] > end_s)
        sp = OP.plan_arc(plan[None])[1][0, 1:]
        s_f = OP.ramp_progress([info["vf"]], [v0])[0]
        unchanged = s_f <= sp
        assert np.array_equal(out[unchanged], plan[unchanged])              # a knot the floor does not lift keeps its value exactly
    assert n_bound >= 40 and n_ext >= 20                                      # the test reached the binding and the extension branches
    # a plan accelerating through the floor: progress never drops below the plan's (the max), and the late knots are untouched
    acc = np.stack([np.cumsum(np.linspace(0.0, 15.0, 30) * 0.2), np.zeros(30)], 1)
    out, info = OP.apply_floor(acc, 0.0, 10.0, 0.1, 0.5, 0.75)
    assert np.all(chord_progress(out) >= chord_progress(acc) - 1e-12)
    assert np.array_equal(out[-1], acc[-1])
    # the extension direction is the LAST segment's
    bent = np.array([[0.0005 * k, 0.0] for k in range(1, 20)] + [[0.0095 + 0.0005 * j, 0.0005 * j * 2] for j in range(1, 12)])
    out, info = OP.apply_floor(bent, 0.0, 10.0, 0.1, 0.5, 0.5)
    d = out[-1] - out[-2]
    last = bent[-1] - bent[-2]
    assert abs(d[0] / np.linalg.norm(d) - last[0] / np.linalg.norm(last)) < 1e-9 and abs(d[1] / np.linalg.norm(d) - last[1] / np.linalg.norm(last)) < 1e-9
    # degenerate directions: a last segment of exactly 0 uses the last non-zero one; a plan with no motion extrapolates along +x
    tail0 = straight(0.01, heading=0.7)
    tail0[-3:] = tail0[-4]
    out, info = OP.apply_floor(tail0, 0.0, 10.0, 0.1, 0.5, 0.5)
    assert info["bind"] and info["dir_fallback"] and abs(math.atan2(out[-1][1], out[-1][0]) - 0.7) < 1e-9
    out, info = OP.apply_floor(np.zeros((30, 2)), 0.0, 10.0, 0.1, 0.5, 0.5)
    assert info["bind"] and info["dir_fallback"] and np.abs(out[:, 1]).max() == 0.0 and np.abs(chord_progress(out) - ramp_ref(5.0, 0.0, T)).max() < 1e-9
    # lift = (chord(out)[14] - chord(in)[14]) / 3 s on a straight plan
    s = straight(0.5)
    out, info = OP.apply_floor(s, 0.0, 10.0, 0.1, 0.5, 0.5)
    assert abs(info["lift"] - (ramp_ref(5.0, 0.0, T)[14] - 0.5 * T[14]) / 3.0) < 1e-9
    late = np.concatenate([straight(10.0)[:10], straight(10.0)[9] + straight(1.0)[:20]])                  # fast for 2 s, then 1 m/s: the floor (5 m/s, v0 = 10) binds only after 4.5 s
    out, info = OP.apply_floor(late, 10.0, 10.0, 0.1, 0.5, 0.5)
    assert info["bind"] and not info["bind_k10"] and info["lift"] == 0.0 and np.array_equal(out[:22], late[:22]) and not np.array_equal(out[22:], late[22:])
    _, info = OP.apply_floor(straight(0.5), 0.0, 10.0, 0.1, 0.5, 0.5)
    assert info["bind"] and info["bind_k10"]
    with pytest.raises(ValueError):
        OP.apply_floor_batch(np.zeros((2, 29, 2)), [0, 0], [10, 10], [0, 0], [0.5, 0.5], 0.5)


# ------------------------------------------------------------------------------------------------ (g)
def _argv(*extra):
    return ["run", "--ckpt", "x.pt", "--tag", "t", "--tracks", "x", "--eval-grid", "x", "--eval-join", "x", "--out-dir", "x", *extra]


def _refused(argv, needle):
    with pytest.raises(SystemExit) as e:
        T1.main(argv)
    assert "REFUSED" in str(e.value) and needle in str(e.value), str(e.value)


def test_g_argparse_refusals(tmp_path):
    ckpt = tmp_path / "head.pt"
    ckpt.write_bytes(b"x")
    _refused(_argv("--level-floor", "on"), "--floor-kappa")
    _refused(_argv("--level-floor", "on", "--floor-kappa", "0.5"), "--floor-ckpt")
    _refused(_argv("--level-floor", "on", "--floor-kappa", "0.5", "--floor-ckpt", str(tmp_path / "missing.pt")), "does not exist")
    _refused(_argv("--floor-kappa", "0.5"), "belong to --level-floor on")
    _refused(_argv("--floor-ckpt", str(ckpt)), "belong to --level-floor on")
    _refused(_argv("--level-floor", "off", "--floor-kappa", "0.5"), "belong to --level-floor on")
    for bad in ("-0.01", "1.0001", "nan", "inf"):
        _refused(_argv("--level-floor", "on", "--floor-kappa", bad, "--floor-ckpt", str(ckpt)), "finite number in [0, 1]")
    _refused(_argv("--level-floor", "on", "--floor-kappa", "0.5", "--floor-ckpt", str(ckpt), "--histswap", "logged"), "--histswap")
    p = T1.build_parser()
    for ok in (["--level-floor", "on", "--floor-kappa", "0", "--floor-ckpt", str(ckpt)], ["--level-floor", "on", "--floor-kappa", "1", "--floor-ckpt", str(ckpt)], []):
        T1.lf_check_args(p.parse_args(_argv(*ok)))                           # accepted: no SystemExit
    a = p.parse_args(_argv())
    assert a.level_floor == "off" and a.floor_kappa is None and a.floor_ckpt == ""


def _cfg(**kw):
    base = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw", d_feat=16, n_tok=4, hist=3, hmax=9, level_head="on", lead_head="on", w_level=0.2, w_lead=1.0, aux_cond="token")
    base.update(kw)
    return M.V8FMiniConfig(**base)


def test_g_checkpoint_cfg_refusals(tmp_path):
    good = _cfg()
    T1.lf_check_cfg(good, "pool4x10", _cfg(level_head="off", lead_head="off", w_level=0.0, w_lead=0.0, aux_cond="add"), "pool4x10")      # the planner (no heads) and the floor (both) read the same interface
    with pytest.raises(SystemExit, match="level_head on AND lead_head on"):
        T1.lf_check_cfg(_cfg(level_head="off", w_level=0.0, lead_head="on", aux_cond="add"), "pool4x10", good, "pool4x10")
    with pytest.raises(SystemExit, match="level_head on AND lead_head on"):
        T1.lf_check_cfg(_cfg(lead_head="off", w_lead=0.0, aux_cond="add"), "pool4x10", good, "pool4x10")
    with pytest.raises(SystemExit, match="hist"):
        T1.lf_check_cfg(_cfg(hist=2), "pool4x10", good, "pool4x10")
    with pytest.raises(SystemExit, match="grid"):
        T1.lf_check_cfg(good, "pool2x5", good, "pool4x10")
    nohead = _cfg(level_head="off", lead_head="off", w_level=0.0, w_lead=0.0, aux_cond="add")                                              # a real file through the loader
    f = tmp_path / "nohead.pt"
    torch.save({"cfg": dataclasses.asdict(nohead), "model": {}, "grid": "pool4x10"}, f)
    ns = T1.build_parser().parse_args(_argv("--level-floor", "on", "--floor-kappa", "0.5", "--floor-ckpt", str(f)))
    with pytest.raises(SystemExit, match="level_head on AND lead_head on"):
        T1.load_level_floor(ns, good, {"grid": "pool4x10"}, torch.device("cpu"))


# ------------------------------------------------------------------------------------------------ (h)
def _head_model(v_bias=10.0, presence_bias=3.0, gap_bias=-1.0):
    torch.manual_seed(0)
    m = M.V8FMini(_cfg(), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30))).eval()
    with torch.no_grad():
        m.lvl_head[-1].bias.fill_(v_bias)
        m.lead_head[-1].bias.copy_(torch.tensor([presence_bias, gap_bias]))
    return m


def _batch(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"feats": torch.randn(B, 9, 4, 16, generator=g), "frame_ok": torch.ones(B, 9, dtype=torch.bool), "p10f": torch.randn(B, 3, 8, generator=g).abs(),
            "base": torch.randn(B, 5, generator=g), "nav": torch.randn(B, 4, generator=g), "limit": torch.randn(B, generator=g), "a_t": torch.zeros(B, dtype=torch.long), "pose_s": torch.randn(B, 3, generator=g)}


def test_h_vision_only():
    fm = _head_model()
    b = _batch()
    v1, l1, g1, read = T1.floor_heads(fm, b)
    assert read == {"feats", "frame_ok"}, read
    b2 = {k: (torch.randn_like(v.float()).to(v.dtype) * 7 if k not in ("feats", "frame_ok") and v.dtype.is_floating_point else v) for k, v in b.items()}
    b2["p10f"] = b["p10f"] + 100.0
    v2, l2, g2, _ = T1.floor_heads(fm, b2)
    assert np.array_equal(v1, v2) and np.array_equal(l1, l2) and np.array_equal(g1, g2)       # every non-vision key perturbed: the heads are bit-identical
    b3 = dict(b, feats=_batch(seed=1)["feats"])
    v3, l3, g3, _ = T1.floor_heads(fm, b3)
    assert not np.array_equal(v1, v3) and not np.array_equal(l1, l3) and not np.array_equal(g1, g3)   # control: the frames DO move them
    assert abs(float(v1.mean()) - 10.0) < 3.0 and (np.abs(l1 - 3.0) < 3.0).all()
    # F reads only (v_hat, pi, gap) from the heads and p10f[:, -1, 0]
    lf = T1.LevelFloor(fm, 0.5, "x.pt", "0" * 64)
    lf.parity = {"checked_replans": 1, "max_abs_diff": 0.0}                                      # the forward_infer parity needs a full batch: it runs on the real data (KA / smoke)
    plan = np.stack([straight(0.05)] * 4).astype(np.float32)
    v0 = b["p10f"][:, -1, 0].double().numpy()
    o1, i1 = lf(b, plan, v0)
    b4 = dict(b2)
    b4["p10f"] = b2["p10f"].clone()
    b4["p10f"][:, -1, 0] = b["p10f"][:, -1, 0]                                                   # everything else of p10f differs, v0 is the same
    o2, i2 = lf(dict(b4, feats=b["feats"], frame_ok=b["frame_ok"]), plan, b4["p10f"][:, -1, 0].double().numpy())
    assert np.array_equal(o1, o2) and np.array_equal(i1["lift"], i2["lift"])
    o3, _ = lf(b, plan, v0 + 3.0)                                                                # control: a different v0 moves the plan
    assert not np.array_equal(o1, o3)
    assert lf.keys_read == {"feats", "frame_ok"} and lf.stats["replans"] == 12
    import inspect
    code = inspect.getsource(T1.floor_heads).split('"""')[2] + inspect.getsource(T1.LevelFloor.__call__) + inspect.getsource(T1.LevelFloor.heads)       # the code, not the docstrings
    for forbidden in ("p10f", "nav", "limit", "situation", "label", "batch["):
        assert forbidden not in code, forbidden


# ------------------------------------------------------------------------------------------------ (i)
class _FakeModel:
    def __init__(self, raw):
        self.raw = raw

    def forward_infer(self, batch):
        t = torch.from_numpy(self.raw)
        return {"traj30": t, "p10": t * 0.9 + 0.01, "_keys_read": {"feats"}}


class _FixedFloor:
    """The operator with fixed head outputs (the placement test needs no floor model)."""

    def __init__(self, kappa, v_hat=12.0, pi=0.1, gap=0.5):
        self.kappa, self.v_hat, self.pi, self.gap = kappa, v_hat, pi, gap
        self.seen = None

    def __call__(self, batch, plan, v0):
        self.seen = np.array(plan, copy=True)
        B = len(plan)
        return OP.apply_floor_batch(plan, v0, np.full(B, self.v_hat), np.full(B, self.pi), np.full(B, self.gap), self.kappa)


def _raw(B=3):
    rng = np.random.default_rng(5)
    return np.stack([_curved(rng, v) for v in (0.1, 0.4, 9.0)[:B]]).astype(np.float32)


def test_i_placement():
    raw = _raw()
    batch = {"p10f": torch.tensor([[[0.0] * 8, [0.0] * 8, [v] + [0.0] * 7] for v in (0.0, 1.0, 9.0)])}
    fm = _FakeModel(raw)
    for smooth, feasible in (("off", "off"), ("sg72", "kbox"), ("sg72", "off"), ("off", "kbox")):
        base = T1.model_planner(fm, "cpu", keep_aux=True, smooth=smooth, feasible=feasible)(batch)             # the unpatched path (floor=None)
        ff = _FixedFloor(0.5)
        pl = T1.model_planner(fm, "cpu", keep_aux=True, smooth=smooth, feasible=feasible, floor=ff)
        out = pl(batch)
        p10 = (torch.from_numpy(raw) * 0.9 + 0.01).numpy()
        s_plan = T1.smooth_plan(raw, p10, smooth)
        assert np.array_equal(ff.seen, s_plan)                                                               # F sees the plan AFTER S (raw when S is off) ...
        f_plan, info = OP.apply_floor_batch(s_plan, np.array([0.0, 1.0, 9.0]), np.full(3, 12.0), np.full(3, 0.1), np.full(3, 0.5), 0.5)
        want = T1.feasible_plan(f_plan, feasible, v0=np.array([0.0, 1.0, 9.0]))
        assert np.array_equal(out, want)                                                                     # ... and the projection runs AFTER F
        assert np.array_equal(pl.plan_raw, raw) and pl.floor_info["bind"].tolist() == info["bind"].tolist() and info["bind"][:2].all() and not info["bind"][2]
        assert not np.array_equal(out, base)
        if feasible != "off":
            assert np.array_equal(pl.plan_s, s_plan)
        z = T1.model_planner(fm, "cpu", keep_aux=True, smooth=smooth, feasible=feasible, floor=_FixedFloor(0.0))   # kappa 0 == the unpatched planner, bit for bit
        assert np.array_equal(z(batch), base) and not z.floor_info["bind"].any()
        none = T1.model_planner(fm, "cpu", keep_aux=True, smooth=smooth, feasible=feasible)
        assert none.floor_info is None


def test_i_off_never_imports_operator():
    code = TOOL.read_text(encoding="utf-8")
    assert code.count("floor is not None") >= 2 and "import level_floor" not in code                       # loaded by path, only through _lf()
    assert hashlib.sha256((TREE / "stack" / "scripts" / "level_floor.py").read_bytes()).hexdigest() == hashlib.sha256((CODE / "level_floor.py").read_bytes()).hexdigest() or "LF_OPERATOR" in os.environ or "LF_TOOL" in os.environ


def test_i_off_equals_the_original_tool():
    """Off = the unpatched tool: ``model_planner`` and the ``TickRecorder`` field sets of the RI tool 0417c838 (tool/orig) and of the patched tool agree bit for bit when no floor is asked for."""
    orig_path = CODE.parent / "tool" / "orig" / "v8f_t1_eval_ri_0417c838.py"
    assert hashlib.sha256(orig_path.read_bytes()).hexdigest().startswith("0417c838355a51f6")
    OR = _load("v8f_t1_eval_orig_under_test", orig_path)
    raw = _raw()
    batch = {"p10f": torch.tensor([[[0.0] * 8, [0.0] * 8, [v] + [0.0] * 7] for v in (0.0, 1.0, 9.0)])}
    fm = _FakeModel(raw)
    for smooth, feasible in (("off", "off"), ("sg72", "kbox"), ("sg72", "off"), ("off", "kbox"), ("sg72", "box")):
        a = OR.model_planner(fm, "cpu", keep_aux=True, smooth=smooth, feasible=feasible)
        b = T1.model_planner(fm, "cpu", keep_aux=True, smooth=smooth, feasible=feasible)
        assert np.array_equal(a(batch), b(batch))
        assert (a.plan_raw is None) == (b.plan_raw is None) and (a.plan_raw is None or np.array_equal(a.plan_raw, b.plan_raw))
        assert (a.plan_s is None) == (b.plan_s is None) and (a.plan_s is None or np.array_equal(a.plan_s, b.plan_s))
        ra, rb = OR.TickRecorder(a, None, None, smooth=smooth, feasible=feasible), T1.TickRecorder(b, None, None, smooth=smooth, feasible=feasible)
        assert list(ra.rows) == list(rb.rows)
        rc = T1.TickRecorder(b, None, None, smooth=smooth, feasible=feasible, level_floor=True)
        assert set(rc.rows) == set(rb.rows) | {"plan_raw"} | set(T1.TickRecorder.LF_FIELDS)
    assert T1.TickRecorder.LF_FIELDS[:7] == ("lf_bind", "lf_cap", "lf_vf", "lf_vhat", "lf_pi", "lf_gap", "lf_lift")
