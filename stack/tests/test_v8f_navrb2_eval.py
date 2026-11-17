"""PREREG_V8F_NAVRB2 evaluator wrappers (HANDOFF step 2): ``v8f_t1_eval_rb.py`` (T1), ``v8f_mini_eval_rb.py`` (T0), ``navrb_geval_identity.py`` (G-EVAL). CPU only, synthetic world; the pinned tools are untouched.

Asserted: both pins equal the tools' bytes; refusals (flag vs cfg, npz without sha, off with a named flag, shuffle beside an own channel); the dead-reckoned channel HOLDS the t0 value after t0 and never reads a logged row after t0 (a
mutant that reads the logged rows goes RED), the logged mode keeps the rows; real / zero / donor rows follow the tool's rule; with the flag off the wrapper patches nothing; the G-EVAL identity (zero channel weight == the channel off) is
bit-equal at the TOOLS' OWN batch sizes (T1 32, T0 64) and at 256, and the pair writer / comparer refuse and name a difference.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import v8f_t1_eval as T  # noqa: E402
import v8f_t1_eval_rb as RB1  # noqa: E402
import v8f_mini_eval as E  # noqa: E402
import v8f_mini_eval_rb as RB0  # noqa: E402
import navrb_geval_identity as GI  # noqa: E402
from test_v8f_navrb2 import M, D, build, make_batch, world  # noqa: E402,F401


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_pins_equal_the_tools_bytes():
    assert RB1.tool_sha256() == RB1.TOOL_SHA256 and RB0.tool_sha256() == RB0.TOOL_SHA256
    assert RB1.tool_sha256(T.__file__) == "180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93"


def _cfg(nav_rb):
    return argparse.Namespace(nav_rb=nav_rb)


def test_t1_refusals():
    ns = lambda **k: argparse.Namespace(**{"nav_rb": "off", "nav_rb_npz": "", "nav_rb_sha256": "", "nav_rb_variant": "follow", **k})   # noqa: E731
    assert RB1.refusals(ns(), _cfg("off")) == [] and RB1.refusals(ns(nav_rb="on", nav_rb_npz="t", nav_rb_sha256="s"), _cfg("on")) == []
    assert RB1.refusals(ns(), _cfg("on")) and RB1.refusals(ns(nav_rb="on", nav_rb_npz="t", nav_rb_sha256="s"), _cfg("off"))
    assert RB1.refusals(ns(nav_rb="on"), _cfg("on"))                                           # no npz / sha
    assert RB1.refusals(ns(nav_rb_variant="zero"), _cfg("off")) and RB1.refusals(ns(nav_rb_npz="t"), _cfg("off"))


def test_t0_refusals():
    r = lambda **k: argparse.Namespace(**{"nav_rb": "off", "nav_rb_npz": "", "nav_rb_sha256": "", "nav_rb_variant": "follow", **k})   # noqa: E731
    a = lambda **k: argparse.Namespace(**{"nav_shuffle": False, "nav_withheld": False, **k})   # noqa: E731
    on = dict(nav_rb="on", nav_rb_npz="t", nav_rb_sha256="s")
    assert RB0.refusals(r(), _cfg("off"), a()) == [] and RB0.refusals(r(**on), _cfg("on"), a()) == []
    assert RB0.refusals(r(), _cfg("on"), a()) and RB0.refusals(r(**on), _cfg("off"), a()) and RB0.refusals(r(nav_rb="on"), _cfg("on"), a())
    assert RB0.refusals(r(**on), _cfg("on"), a(nav_shuffle=True))                              # a donor nav beside an own channel
    assert RB0.refusals(r(**on, nav_rb_variant="zero"), _cfg("on"), a(nav_shuffle=True)) == []


def _fake_clip(Tn=40, t0=10):
    base = {k: np.zeros((Tn, 3), np.float64) for k in T.WindowState.EGO}
    base["gt_ok"] = np.ones(Tn, bool)
    return base, np.zeros((2 * Tn + 1, 4), np.float64)


def _fake_join(Tn=40):
    rb = np.zeros((2 * Tn, 1), np.float32)
    rb[5:12] = 1.0                                                  # clip 0: on over rows 5..11
    rb[Tn + 20:Tn + 30] = 1.0                                       # clip 1: on over its rows 20..29
    nav = np.zeros((2 * Tn, 5), np.float32)
    return {"off": np.array([0, Tn]), "n": np.array([Tn, Tn]), "nav": nav, "nav_rb": rb, "clip": np.array(["a", "b"])}


@pytest.mark.parametrize("nav_mode", ["deadreckon", "logged"])
def test_state_holds_the_t0_value_after_t0_and_logged_keeps_the_rows(nav_mode):
    J = _fake_join()
    base, track = _fake_clip()
    t0 = 8
    rows = RB1.rb_rows(J, 0, t0, "follow", None, "real")
    st = RB1.RBWindowState(base, track, J["nav"][:40], 0, t0, nav_mode, rows)
    assert st.nav_rb[t0, 0] == 1.0
    if nav_mode == "deadreckon":
        assert (st.nav_rb[t0 + 1:] == 1.0).all()                    # held from t0, though the logged rows 12.. are 0
    else:
        assert np.array_equal(st.nav_rb, J["nav_rb"][:40])
    st2 = RB1.RBWindowState(base, track, J["nav"][:40], 0, 20, nav_mode, RB1.rb_rows(J, 0, 20, "follow", None, "real"))
    assert st2.nav_rb[20, 0] == 0.0 and (st2.nav_rb[21:] == 0.0).all() if nav_mode == "deadreckon" else True


def test_variants_follow_the_tools_rule():
    J = _fake_join()
    real = RB1.rb_rows(J, 0, 8, "follow", None, "real")
    assert np.array_equal(real, J["nav_rb"][:40])
    assert not RB1.rb_rows(J, 0, 8, "follow", None, "zero").any() and not RB1.rb_rows(J, 0, 8, "zero", None, "real").any()      # follow under a zero nav / Z-EVAL
    assert np.array_equal(RB1.rb_rows(J, 0, 8, "real", None, "zero"), real)                                                       # real beside a moved nav
    don = RB1.rb_rows(J, 0, 8, "follow", (1, 25), "donor")                                                                         # donor window: clip 1, NOW at row 25 (on)
    assert np.array_equal(don[:9], J["nav_rb"][40 + 17:40 + 26]) and don[8, 0] == 1.0 and don[0, 0] == 0.0 and np.array_equal(don[9:], J["nav_rb"][9:40])
    with pytest.raises(ValueError):
        RB1.rb_rows(J, 0, 8, "follow", None, "donor")


def test_build_item_overwrites_the_logged_read_and_a_mutant_that_keeps_it_is_red():
    J = _fake_join()
    base, track = _fake_clip()
    st = RB1.RBWindowState(base, track, J["nav"][:40], 0, 8, "deadreckon", RB1.rb_rows(J, 0, 8, "follow", None, "real"))
    ds = argparse.Namespace(hmax=10)
    st.j = 14                                                       # after commits: the frames 5..14

    def tool_item(ds_, proxy, st_, *a, **k):                         # what the tool's item builder returns: the LOGGED rows of the frames (0 after row 11)
        idc = np.clip(np.arange(st_.j - ds_.hmax + 1, st_.j + 1), 0, st_.T - 1)
        return {"nav_rb": J["nav_rb"][idc].astype(np.float32), "nav": np.zeros((10, 5), np.float32)}
    it = RB1.build_item_factory(tool_item)(ds, None, st, 0, 0, 0, "logged", "logged")
    assert (it["nav_rb"] == 1.0).all()                              # frames 5..14 all carry the held t0 value
    mutant = tool_item(ds, None, st)
    assert not (mutant["nav_rb"] == 1.0).all()                      # the unwrapped read would see the logged zeros after row 11: the wrapper's overwrite is what the assertion above proves


def test_flag_off_patches_nothing(monkeypatch):
    seen = {}
    monkeypatch.setattr(T, "cmd_run", lambda a: seen.setdefault("run", (T.MODEL_KEYS, T.make_states, T.build_item)) and 0)
    before = (T.MODEL_KEYS, T.make_states, T.build_item, T.TickRecorder, T.D.load_join)
    import torch as _t
    ck = {"cfg": {"nav_rb": "off", "d": 32, "ff": 64, "heads": 4, "n_q": 2, "depth_s": 1, "depth_t": 1, "depth_o": 1, "nb": 6}}
    monkeypatch.setattr(_t, "load", lambda *a, **k: ck)
    a = argparse.Namespace(ckpt="x", nav_rb="off", nav_rb_npz="", nav_rb_sha256="", nav_rb_variant="follow")
    assert RB1.make_cmd_run(T.cmd_run)(a) == 0
    assert seen["run"][0] is before[0] and seen["run"][1] is before[1] and seen["run"][2] is before[2]
    assert (T.MODEL_KEYS, T.make_states, T.build_item, T.TickRecorder, T.D.load_join) == before


def _tiled(world, n):
    b, J = make_batch(world, n=min(n, 16))
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


@pytest.mark.parametrize("bs", [32, 64, 256])
def test_g_eval_identity_zero_weight_equals_off_at_the_tools_batch_sizes(world, bs):
    """A channel model whose e_nav_rb.weight is exactly 0.0 dumps bit-equal to the same weights with the channel off, fed in chunks of the evaluator's own batch size."""
    b, J = _tiled(world, 256)
    mon = build(J, nav_rb="on")
    moff = build(J, nav_rb="off")
    sd = {k: v for k, v in mon.state_dict().items() if not k.startswith("e_nav_rb.")}
    moff.load_state_dict(sd, strict=True)
    assert float(mon.e_nav_rb.weight.abs().sum()) == 0.0
    mon.eval(); moff.eval()
    n = b["nav"].shape[0]
    with torch.no_grad():
        for b0 in range(0, n, bs):
            cut = lambda x: ({kk: vv[b0:b0 + bs] if torch.is_tensor(vv) else vv for kk, vv in x.items()} if isinstance(x, dict) else (x[b0:b0 + bs] if torch.is_tensor(x) else x))   # noqa: E731
            bb = {k: cut(v) for k, v in b.items()}
            o1 = mon.forward_infer(bb)
            o0 = moff.forward_infer({k: v for k, v in bb.items() if k != "nav_rb"})
            assert all(torch.equal(o1[k], o0[k]) for k in o1 if torch.is_tensor(o1[k]) and k in o0), bs


def test_pair_writer_and_comparer(world, tmp_path):
    b, J = _tiled(world, 16)
    m = build(J, nav_rb="on")
    with torch.no_grad():
        m.e_nav_rb.weight.fill_(0.3)
    ck = {"cfg": {**{k: getattr(m.cfg, k) for k in m.cfg.__dataclass_fields__}}, "model": m.state_dict(), "step": 3}
    p = str(tmp_path / "c.pt")
    torch.save(ck, p)
    r = GI.pair(p, str(tmp_path / "o"))
    zon, zoff = torch.load(r["zon"], weights_only=False), torch.load(r["zoff"], weights_only=False)
    assert float(zon["model"]["e_nav_rb.weight"].abs().sum()) == 0.0 and zon["cfg"]["nav_rb"] == "on"
    assert zoff["cfg"]["nav_rb"] == "off" and not any(k.startswith("e_nav_rb.") for k in zoff["model"]) and r["dropped_keys"] == ["e_nav_rb.weight"]
    M.V8FMini(M.V8FMiniConfig(**zoff["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(J)).load_state_dict(zoff["model"], strict=True)
    with pytest.raises(SystemExit):
        torch.save({**ck, "cfg": {**ck["cfg"], "nav_rb": "off"}}, p)
        GI.pair(p, str(tmp_path / "o2"))
    a, c = str(tmp_path / "a.npz"), str(tmp_path / "c.npz")
    np.savez(a, pred=np.arange(6, dtype=np.float32), eid=np.arange(3))
    np.savez(c, pred=np.arange(6, dtype=np.float32), eid=np.arange(3))
    assert GI.compare(a, c) == []
    np.savez(c, pred=np.arange(6, dtype=np.float32) + np.float32(1e-7), eid=np.arange(3))
    assert GI.compare(a, c) and "pred" in GI.compare(a, c)[0]
