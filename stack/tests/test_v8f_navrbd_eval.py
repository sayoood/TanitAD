"""PREREG_V8F_NAVRBD evaluator wrappers (BUILD_PLAN items 7 and 8): ``v8f_t1_eval_rbd.py`` (T1) and ``v8f_mini_eval_rbd.py`` (T0). CPU only, synthetic worlds; the two pinned tools stay byte-untouched.

Asserted: both tool pins equal the tools' bytes (and so does the wrapper's own pin); the refusals (a flag that differs from the checkpoint cfg's feat, a 15-column checkpoint whose cfg dropped the feat = G-GEN, a width the weight
does not have, a feat named with the channel off); in dead-reckoning mode c13 falls by arc / 100 per tick from its t0 value (clamped to [0, 3]) while c0..c12 and c14 stay at their t0 values, and a logged run keeps the logged rows;
with the flag off the wrapper patches nothing; the G-EVAL identity (a 15-column model with ``e_nav_rb.weight`` exactly 0.0 == the channel-off weights, bit for bit) at the tools' own batch sizes 32 and 64 and at 256; zeroing the
channel on windows whose rows are all 0 changes nothing (a non-zero weight, bias-free) while real rows DO move the output. Each mutation (delete the c13 arc subtraction, hold c13 fixed, drop the feat check in the T1 and the T0 wrapper)
turns a named check RED.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import tanitad  # noqa: E402
import v8f_t1_eval as T  # noqa: E402
import v8f_t1_eval_rb as RB1  # noqa: E402
import v8f_t1_eval_rbd as RBD  # noqa: E402
import v8f_mini_eval as E  # noqa: E402
import v8f_mini_eval_rb as RB0  # noqa: E402
import v8f_mini_eval_rbd as RBD0  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from test_v8f_navrbd import build, make_batch, world  # noqa: E402,F401
from test_v8f_t1 import setup as t1_setup  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(HERE.parents[1].resolve()), tanitad.__file__          # the study tree's tanitad
T1_PIN = "180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93"
T0_PIN = "008f48b79fd11ca553d118d8e29ece6aea29d1dd013653ac23b1dc7fe2dad558"
T1_RBD_P, T0_RBD_P = SCRIPTS / "v8f_t1_eval_rbd.py", SCRIPTS / "v8f_mini_eval_rbd.py"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mut_module(path, old, new, name):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (old[:70], text.count(old))
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(text.replace(old, new), str(path), "exec"), mod.__dict__)
    return mod


def test_the_pins_equal_the_tools_bytes_and_the_wrappers_pin_the_same_tools():
    assert _sha(T.__file__) == T1_PIN == RB1.TOOL_SHA256 == RBD.R.TOOL_SHA256 and RBD.R.tool_sha256() == T1_PIN
    assert _sha(E.__file__) == T0_PIN == RB0.TOOL_SHA256 == RBD0.R0.TOOL_SHA256 and RBD0.R0.tool_sha256() == T0_PIN
    assert RBD.MINI_EVAL_SHA256 == T0_PIN


# ------------------------------------------------------------------------------------------------ refusals (G-GEN)
def _cfg(nav_rb="on", feat="exitdist"):
    return argparse.Namespace(nav_rb=nav_rb, nav_rb_feat=feat)


def _ck(width):
    return {"model": {"e_nav_rb.weight": torch.zeros(32, width)}}


def _t1(**k):
    return argparse.Namespace(**{"nav_rb": "on", "nav_rb_feat": "exitdist", "nav_rb_npz": "t", "nav_rb_sha256": "s", "nav_rb_variant": "follow", **k})


def check_t1_refusals(mod):
    assert mod.refusals(_t1(), _cfg(), _ck(15)) == []
    for v in ("follow", "real", "zero", "shuffle"):
        assert mod.refusals(_t1(nav_rb_variant=v), _cfg(), _ck(15)) == []
    assert mod.refusals(_t1(nav_rb_feat="state"), _cfg("on", "exitdist"), _ck(15))                       # the flag names the narrow channel beside a wide cfg
    assert mod.refusals(_t1(nav_rb_feat="exitdist"), _cfg("on", "state"), _ck(1))                        # the flag names the wide channel beside a narrow cfg
    # G-GEN: a 15-column checkpoint whose cfg DROPPED the feat reads as state: refused under either flag
    dropped = M.V8FMiniConfig(**{k: v for k, v in M.V8FMiniConfig(nav_rb="on", nav_rb_feat="exitdist").__dict__.items() if k != "nav_rb_feat"})
    assert dropped.nav_rb_feat == "state"
    assert mod.refusals(_t1(nav_rb_feat="exitdist"), dropped, _ck(15))
    assert mod.refusals(_t1(nav_rb_feat="state"), dropped, _ck(15))                                      # flag == cfg, but the weight is 15 wide: the width clause
    assert mod.refusals(_t1(nav_rb="off", nav_rb_npz="", nav_rb_sha256=""), _cfg("off", "state"))        # a feat named with the channel off


def check_t1_feat_clause_alone(mod):
    """The cfg clause is reached ALONE (no checkpoint weight to fall back on): the flag differs from the cfg's feat."""
    assert mod.refusals(_t1(nav_rb_feat="state"), _cfg("on", "exitdist"), None) and mod.refusals(_t1(), _cfg("on", "state"), None)


def test_t1_refusals():
    check_t1_refusals(RBD)
    check_t1_feat_clause_alone(RBD)


def _t0r(**k):
    return argparse.Namespace(**{"nav_rb": "on", "nav_rb_feat": "exitdist", "nav_rb_npz": "t", "nav_rb_sha256": "s", "nav_rb_variant": "follow", **k})


def check_t0_refusals(mod):
    a = argparse.Namespace(nav_shuffle=False, nav_withheld=False)
    assert mod.refusals(_t0r(), _cfg(), a, _ck(15)) == []
    assert mod.refusals(_t0r(nav_rb_feat="state"), _cfg(), a, _ck(15)) and mod.refusals(_t0r(), _cfg("on", "state"), a, None)
    dropped = M.V8FMiniConfig(**{k: v for k, v in M.V8FMiniConfig(nav_rb="on", nav_rb_feat="exitdist").__dict__.items() if k != "nav_rb_feat"})
    assert mod.refusals(_t0r(), dropped, a, _ck(15)) and mod.refusals(_t0r(nav_rb_feat="state"), dropped, a, _ck(15))
    assert mod.refusals(_t0r(nav_rb="off", nav_rb_npz="", nav_rb_sha256=""), _cfg("off", "state"), a)
    assert mod.refusals(_t0r(), _cfg(), argparse.Namespace(nav_shuffle=True, nav_withheld=False), _ck(15))     # the NAVRB2 donor-nav rule still applies
    assert mod.refusals(_t0r(nav_rb_variant="shuffle"), _cfg(), argparse.Namespace(nav_shuffle=True, nav_withheld=False), _ck(15)) == []


def test_t0_refusals():
    check_t0_refusals(RBD0)


def test_the_t1_parser_carries_the_feat_and_the_shuffle_variant_and_the_t0_parser_too():
    sub = next(x for x in RBD.build_parser()._actions if isinstance(x, argparse._SubParsersAction)).choices["run"]
    acts = {x.dest: x for x in sub._actions}
    assert acts["nav_rb_feat"].default == "state" and tuple(acts["nav_rb_feat"].choices) == ("state", "exitdist")
    assert tuple(acts["nav_rb_variant"].choices) == ("follow", "real", "zero", "shuffle") and acts["nav_rb_variant"].default == "follow"
    r, rest = RBD0.rb_parser().parse_known_args(["--nav-rb", "on", "--nav-rb-feat", "exitdist", "--nav-rb-variant", "shuffle", "--ckpt", "x"])
    assert (r.nav_rb, r.nav_rb_feat, r.nav_rb_variant, rest) == ("on", "exitdist", "shuffle", ["--ckpt", "x"]) and RBD0.rb_parser().parse_args([]).nav_rb_feat == "state"


def test_flag_off_patches_nothing(monkeypatch):
    seen = {}
    monkeypatch.setattr(T, "cmd_run", lambda a: seen.setdefault("run", (T.MODEL_KEYS, T.make_states, T.build_item)) and 0)
    before = (T.MODEL_KEYS, T.make_states, T.build_item, T.TickRecorder, T.D.load_join)
    ck = {"cfg": {"nav_rb": "off", "d": 32, "ff": 64, "heads": 4, "n_q": 2, "depth_s": 1, "depth_t": 1, "depth_o": 1, "nb": 6}}
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)
    a = argparse.Namespace(ckpt="x", nav_rb="off", nav_rb_feat="state", nav_rb_npz="", nav_rb_sha256="", nav_rb_variant="follow")
    assert RBD.make_cmd_run(T.cmd_run)(a) == 0
    assert seen["run"][0] is before[0] and seen["run"][1] is before[1] and seen["run"][2] is before[2]
    assert (T.MODEL_KEYS, T.make_states, T.build_item, T.TickRecorder, T.D.load_join) == before


# ------------------------------------------------------------------------------------------------ c13 dead reckoning (T1)
@pytest.fixture(scope="module")
def t1world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_t1_rbd")
    return {"root": root, **D.make_synthetic_world(root, n_train=2, n_eval=3, T=80, seed=3)}


def _rows(n, c13, c14=1.0, seed=0):
    """A 15-column channel for one clip: c0..c12 random in every row (so a hold from t0 is distinguishable from the logged rows), c13 / c14 as given at EVERY row (the logged rows after t0 are not the held value)."""
    rng = np.random.default_rng(seed)
    rows = np.zeros((n, 15), np.float32)
    rows[:, :13] = rng.uniform(0.1, 1.0, (n, 13))
    rows[:, 13], rows[:, 14] = c13, c14
    return rows


def _state(mod, w, c13_0, nav_mode="deadreckon", seed=0):
    tracks, J, store, wins = t1_setup(w)
    ci, t0, _, _ = wins[1]
    clip = str(J["clip"][ci])
    o, n = int(J["off"][ci]), int(J["n"][ci])
    rows = _rows(n, 2.5, seed=seed)                                    # logged rows: c13 = 2.5 everywhere (after t0 too)
    rows[t0, 13] = c13_0
    return mod.RBDWindowState(store.get(clip), tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode, rows), t0, rows


def check_c13(mod, w):
    for c13_0 in (0.5, 0.01, 3.0, 0.0):
        st, t0, rows = _state(mod, w, c13_0)
        plan = np.column_stack([10.0 * T.DT * np.arange(1, 31), np.zeros(30)])                      # straight, 2.0 m per step
        prev = c13_0
        for k in range(1, 5):
            st.commit_plan(plan, 1)
            want = np.float32(np.clip(c13_0 - st.arc[t0 + k] / 100.0, 0.0, 3.0))
            assert st.arc[t0 + k] == pytest.approx(2.0 * k, abs=1e-6)
            assert st.nav_rb[t0 + k, 13] == pytest.approx(want, abs=1e-6), (c13_0, k)
            assert st.nav_rb[t0 + k, 13] <= prev + 1e-9 and (c13_0 == 0.0 or c13_0 >= 3.0 or st.nav_rb[t0 + k, 13] < prev or want == 0.0)   # it FALLS (or has reached the clamp)
            prev = float(st.nav_rb[t0 + k, 13])
            assert np.array_equal(st.nav_rb[t0 + k, :13], rows[t0, :13]) and st.nav_rb[t0 + k, 14] == rows[t0, 14]       # c0..c12 and c14 HELD from t0
        assert np.array_equal(st.nav_rb[t0, :], rows[t0, :])                                         # the t0 row itself is as logged
    st, t0, _ = _state(mod, w, 0.5)
    assert st.nav_rb[t0 + 1, 13] == np.float32(0.5) and (st.nav_rb[t0 + 5:, :13] == st.nav_rb0[:13]).all()  # before any commit: held, c13 not yet dead-reckoned


def test_c13_falls_by_arc_over_100_and_the_rest_is_held(t1world):
    check_c13(RBD, t1world)


def test_a_logged_run_keeps_the_logged_rows_and_a_state_channel_is_not_dead_reckoned(t1world):
    st, t0, rows = _state(RBD, t1world, 0.5, "logged")
    plan = np.column_stack([10.0 * T.DT * np.arange(1, 31), np.zeros(30)])
    st.commit_plan(plan, 2)
    assert np.array_equal(st.nav_rb, rows)                                                           # the rows after t0 are the logged ones
    tracks, J, store, wins = t1_setup(t1world)
    ci, t0, _, _ = wins[1]
    clip, o, n = str(J["clip"][ci]), int(J["off"][ci]), int(J["n"][ci])
    r1 = np.zeros((n, 1), np.float32)
    r1[t0] = 1.0
    s1 = RBD.RBDWindowState(store.get(clip), tracks[clip], J["nav"][o:o + n], ci, t0, "deadreckon", r1)
    s1.commit_plan(plan, 2)
    assert (s1.nav_rb[t0 + 1:] == 1.0).all()                                                         # the 1-column channel: the NAVRB2 hold, nothing else


def test_mutation_deleting_the_arc_subtraction_turns_the_c13_check_red(t1world):
    mod = mut_module(T1_RBD_P, "            self.nav_rb[j, C13] = np.float32(np.clip(self.nav_rb0[C13] - self.arc[j] / 100.0, 0.0, 3.0))", "            pass", "rbd_mut_noarc")
    assert RBD.RBDWindowState is not mod.RBDWindowState
    with pytest.raises(AssertionError):
        check_c13(mod, t1world)


def test_mutation_holding_c13_fixed_turns_the_c13_check_red(t1world):
    mod = mut_module(T1_RBD_P, "self.nav_rb0[C13] - self.arc[j] / 100.0", "self.nav_rb0[C13] - 0.0", "rbd_mut_hold")
    with pytest.raises(AssertionError):
        check_c13(mod, t1world)


def test_mutation_dropping_the_feat_check_turns_the_refusal_checks_red():
    m1 = mut_module(T1_RBD_P, "    if a.nav_rb_feat != cfg_feat:", "    if False:", "rbd_mut_feat")
    with pytest.raises(AssertionError):
        check_t1_feat_clause_alone(m1)
    with pytest.raises(AssertionError):
        check_t1_refusals(m1)
    m0 = mut_module(T0_RBD_P, "    if r.nav_rb_feat != cfg_feat:", "    if False:", "rbd0_mut_feat")
    with pytest.raises(AssertionError):
        check_t0_refusals(m0)


def test_mutation_dropping_the_width_clause_turns_the_g_gen_check_red():
    m1 = mut_module(T1_RBD_P, "    if w is not None and w != FEAT_WIDTH[a.nav_rb_feat]:", "    if False:", "rbd_mut_width")
    with pytest.raises(AssertionError):
        check_t1_refusals(m1)


# ------------------------------------------------------------------------------------------------ G-EVAL identity at the tools' batch sizes
def _tiled(world, n):
    b, J = make_batch(world, "exitdist", n=16)
    rep = -(-n // b["nav"].shape[0])
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = torch.cat([v] * rep)[:n]
        elif isinstance(v, dict):
            out[k] = {kk: (torch.cat([vv] * rep)[:n] if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out, J


def _cut(x, b0, bs):
    if isinstance(x, dict):
        return {kk: vv[b0:b0 + bs] if torch.is_tensor(vv) else vv for kk, vv in x.items()}
    return x[b0:b0 + bs] if torch.is_tensor(x) else x


def _pair(J, weight=None):
    mon = build(J, nav_rb="on", nav_rb_feat="exitdist")
    moff = build(J, nav_rb="off")
    moff.load_state_dict({k: v for k, v in mon.state_dict().items() if not k.startswith("e_nav_rb.")}, strict=True)
    if weight is not None:
        with torch.no_grad():
            mon.e_nav_rb.weight.copy_(weight)
    return mon.eval(), moff.eval()


@pytest.mark.parametrize("bs", [32, 64, 256])
def test_g_eval_identity_a_zero_15_column_weight_equals_the_channel_off_model(world, bs):
    b, J = _tiled(world, 256)
    mon, moff = _pair(J)
    assert tuple(mon.e_nav_rb.weight.shape) == (32, 15) and float(mon.e_nav_rb.weight.abs().sum()) == 0.0 and float(b["nav_rb"].abs().sum()) > 0.0 and b["nav_rb"].shape[-1] == 15
    n = b["nav"].shape[0]
    with torch.no_grad():
        for b0 in range(0, n, bs):
            bb = {k: _cut(v, b0, bs) for k, v in b.items()}
            o1 = mon.forward_infer(bb)
            o0 = moff.forward_infer({k: v for k, v in bb.items() if k != "nav_rb"})
            assert all(torch.equal(o1[k], o0[k]) for k in o1 if torch.is_tensor(o1[k]) and k in o0), bs


def test_g_eval_identity_is_sensitive_a_nonzero_weight_on_real_rows_moves_the_output(world):
    b, J = _tiled(world, 32)
    w = torch.randn(32, 15, generator=torch.Generator().manual_seed(4)) * 0.5
    mon, moff = _pair(J, w)
    with torch.no_grad():
        o1 = mon.forward_infer(b)
        o0 = moff.forward_infer({k: v for k, v in b.items() if k != "nav_rb"})
    assert any(not torch.equal(o1[k], o0[k]) for k in o1 if torch.is_tensor(o1[k]) and k in o0)


def test_zeroing_the_channel_on_windows_whose_rows_are_all_zero_changes_nothing(world):
    b, J = _tiled(world, 64)
    w = torch.randn(32, 15, generator=torch.Generator().manual_seed(5)) * 0.5
    mon, moff = _pair(J, w)
    quiet = (b["nav_rb"].abs().sum(dim=(1, 2)) == 0)                                  # windows whose rows are ALL 0
    b0 = dict(b)
    b0["nav_rb"] = torch.zeros_like(b["nav_rb"])                                       # every row zeroed
    with torch.no_grad():
        o_real, o_zero = mon.forward_infer(b), mon.forward_infer(b0)
        o_off = moff.forward_infer({k: v for k, v in b0.items() if k != "nav_rb"})
    assert all(torch.equal(o_zero[k], o_off[k]) for k in o_zero if torch.is_tensor(o_zero[k]) and k in o_off)        # a zero channel through a bias-free Linear == the channel off
    if quiet.any():                                                                    # on the windows that are all 0 already, real == zeroed
        for k in o_real:
            if torch.is_tensor(o_real[k]) and k in o_zero and o_real[k].shape[:1] == quiet.shape:
                assert torch.equal(o_real[k][quiet], o_zero[k][quiet]), k
    assert any(not torch.equal(o_real[k], o_zero[k]) for k in o_real if torch.is_tensor(o_real[k]) and k in o_zero)   # and the real rows DO move it (the control)
