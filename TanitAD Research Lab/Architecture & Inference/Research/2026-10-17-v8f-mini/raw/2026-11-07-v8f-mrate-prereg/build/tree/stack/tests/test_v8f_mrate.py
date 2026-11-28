"""PREREG_V8F_MRATE, the model half: the multi-rate world model (``V8FMiniConfig.wm_k`` = K, ``WMAux``, ``hjepa`` dilation / causal_pad).

The card's controls this file pins (CPU, tiny readers, the real WM size: wm_dim 128, strides (2, 2), IDM on):

* K = 1 IDENTITY (control 2): ``wm_k`` 1 builds the earlier WM -- the same state-dict keys and shapes as M2b's 145 ``wm.*`` tensors (when the checkpoint is on this disk) and the same forward on a fixed batch,
  bit for bit; K = 3 under the same seed shares every parameter that is not a level encoder bit for bit.
* CAUSALITY: perturbing the frames from index j on leaves every state before j unchanged at EVERY level (L1, L2, L3), and the macro-actions of the transitions before j.
* READS: a K = 3 state reads exactly the documented lower states (oldest first, spaced at the level's own step, the first state repeated before the window start).
* HISTORY EXTENSION: prepending 4 frames (one L3 step) changes only the early states, which is what makes a start at frame 14 of a 28-frame window a clean read.
* SHAPES at hmax 28: 28 / 14 / 7 states, 27 / 13 / 6 transitions; the rollouts the WMACT reads (L1 k 10, L2 k 3, L3 k 3).
* MUTATIONS: each check is a plain function; deleting the dilation, the causal pad, or ``z1``'s front pad makes it raise.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
B, F, NT = 3, 28, 6                                   # windows, frames (hmax 28), tokens per frame (the tiny grid)
M2B_S0 = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt"


def cfg_of(k: int = 1, **kw) -> M.V8FMiniConfig:
    return M.V8FMiniConfig(**{**TINY, "wm": "b", "hmax": 28, "kin_mode": "yaw", "wm_k": k, **kw})


def wm_of(k: int, seed: int = 0, **kw) -> M.WMAux:
    torch.manual_seed(seed)
    return M.WMAux(cfg_of(k, **kw))


def data(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    adapt = torch.randn(B, F, NT, TINY["d"], generator=g)
    act = torch.stack([torch.randn(B, F, generator=g) * 1.5, torch.randn(B, F, generator=g) * 0.2], -1)
    return adapt, act


def z1_seq(seed: int = 3, n: int = F):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, n, 128, generator=g), torch.randn(B, n - 1, 2, generator=g)


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal up to GEMM batching (a [B*n, 3D] matmul and a [B, 3D] one do not reduce in the same order): 1e-6 absolute, 5 orders below any wrong read (~0.1)."""
    return a.shape == b.shape and bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6))


# ------------------------------------------------------------------------------------------------ the checks (plain functions: the mutation tests call them under a mutant)
def check_causality(wm: M.WMAux, js=(5, 12, 20), seed: int = 7) -> None:
    """Perturb the frames from j on: every state before j is bit-equal at every level; so is every macro-action whose whole span is before j."""
    adapt, act = data(seed)
    base = wm.stack.encode(wm.z1(adapt), wm.scale_actions(act)[:, :-1])
    for j in js:
        a2, c2 = adapt.clone(), act.clone()
        a2[:, j:] += torch.randn_like(a2[:, j:]) * 3.0
        c2[:, j:] += 2.0
        enc = wm.stack.encode(wm.z1(a2), wm.scale_actions(c2)[:, :-1])
        stride = {1: 1, 2: 2, 3: 4}                                          # L1 frames per state of level l (wm_strides (2, 2))
        for l in (1, 2, 3):
            n_before = -(-j // stride[l])                                    # states i with i * stride < j
            assert torch.equal(base.z[l - 1][:, :n_before], enc.z[l - 1][:, :n_before]), f"level {l}: a state before frame {j} moved"
            assert not torch.equal(base.z[l - 1][:, n_before:], enc.z[l - 1][:, n_before:]), f"level {l}: nothing after frame {j} moved (the perturbation did not reach)"
        # macro-actions of L2 (transition t pools actions 2t, 2t+1): all of those before j are unchanged
        t_before = j // 2
        assert torch.equal(base.act[1][:, :t_before], enc.act[1][:, :t_before]), f"L2 macro-action before frame {j} moved"


def check_reads(wm: M.WMAux, seed: int = 3) -> None:
    """A K = 3 state at level l >= 2 is the encoder over the lower states [i*s - 2d, i*s - d, i*s] (d = the level's own step, s = stride; indices < 0 repeat state 0), oldest first."""
    st = wm.stack
    z, a = z1_seq(seed)
    enc = st.encode(z, a)
    for l in (2, 3):
        lo, up = enc.z[l - 2], enc.z[l - 1]
        c = st.cfg[l - 1]
        assert (c.window, c.dilation, c.causal_pad) == (3, c.stride, True), (l, c)
        for i in range(up.shape[1]):
            idx = [max(i * c.stride - m * c.dilation, 0) for m in (2, 1, 0)]
            want = st.encoders[l - 2](torch.cat([lo[:, ix] for ix in idx], -1))
            assert close(up[:, i], want), f"level {l} state {i}: not the encoder over lower states {idx}"


def check_z1(wm: M.WMAux, seed: int = 7) -> None:
    """Level 1, K = 3: frame j's latent is enc1 over the projections of frames [j-2, j-1, j]; the first two repeat frame 0's."""
    adapt, _ = data(seed)
    p = wm.proj(adapt.mean(2))
    z1 = wm.z1(adapt)
    assert z1.shape == (B, F, 128), z1.shape
    for j in range(F):
        idx = [max(j - m, 0) for m in (2, 1, 0)]
        assert close(z1[:, j], wm.enc1(torch.cat([p[:, ix] for ix in idx], -1))), f"z1 frame {j}"


def check_history_extension(wm: M.WMAux, seed: int = 3) -> None:
    """Prepend 4 frames (= one L3 step = two L2 steps): every state whose full history lies inside the shorter sequence is bit-equal at the shifted index. L2 from index 2, L3 from index 3 (L1 from 0)."""
    z, a = z1_seq(seed, n=30)
    short = wm.stack.encode(z[:, 4:], a[:, 4:])
    long = wm.stack.encode(z, a)
    assert close(short.z[0], long.z[0][:, 4:])
    assert close(short.z[1][:, 2:], long.z[1][:, 2 + 2:]), "L2 state with a full history moved when 4 older frames were prepended"
    assert close(short.z[2][:, 3:], long.z[2][:, 1 + 3:]), "L3 state with a full history moved when 4 older frames were prepended"
    assert not close(short.z[2][:, 2:3], long.z[2][:, 1 + 2:1 + 3]), "L3 state 2 reads the padding in the short sequence: it must differ from the real-history state"


# ------------------------------------------------------------------------------------------------ K = 1 identity
def test_k1_builds_the_earlier_wm_state_dict_and_forward():
    a, b = wm_of(1, 0), M.WMAux(M.V8FMiniConfig(**{**TINY, "wm": "b", "hmax": 28, "kin_mode": "yaw"}))     # explicit wm_k=1 vs the field never named
    torch.manual_seed(0)
    b = M.WMAux(M.V8FMiniConfig(**{**TINY, "wm": "b", "hmax": 28, "kin_mode": "yaw"}))
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]) for k in sa)
    assert a.enc1 is None and not any(k.startswith("enc1") for k in sa)
    assert a.built_k() == {"k": [1, 1, 1], "dilation": [1, 1, 1], "causal_pad": [False, False, False]}
    adapt, act = data()
    a.eval(), b.eval()
    torch.manual_seed(5)
    oa = a(adapt, act)
    torch.manual_seed(5)
    ob = b(adapt, act)
    assert list(oa) == list(ob) and all(torch.equal(oa[k], ob[k]) for k in oa)
    assert torch.equal(a.z1(adapt), a.proj(adapt.mean(2)))                    # K = 1: the earlier z1 line, verbatim


@pytest.mark.skipif(not os.path.exists(M2B_S0), reason="M2b s0 checkpoint not on this disk")
def test_k1_state_dict_keys_and_shapes_equal_m2bs_145_wm_tensors():
    ck = torch.load(M2B_S0, map_location="cpu", weights_only=False)
    want = {k[len("wm."):]: tuple(v.shape) for k, v in ck["model"].items() if k.startswith("wm.")}
    cfg = M.V8FMiniConfig(**ck["cfg"])                                         # M2b's own config (d 256, hmax 16, wm_k absent = 1)
    assert cfg.wm_k == 1
    got = {k: tuple(v.shape) for k, v in M.WMAux(cfg).state_dict().items()}
    assert len(want) == 145 and got == want


def test_k3_shares_every_non_encoder_parameter_bit_for_bit_with_k1_under_one_seed():
    w0, w1 = wm_of(1, 4), wm_of(3, 4)
    s0, s1 = w0.state_dict(), w1.state_dict()
    enc = lambda k: ".encoders." in k or k.startswith("enc1.")                 # noqa: E731
    shared = [k for k in s0 if not enc(k)]
    assert shared and all(torch.equal(s0[k], s1[k]) for k in shared), [k for k in shared if not torch.equal(s0[k], s1[k])]
    assert set(s1) - set(s0) == {"enc1.net.0.weight", "enc1.net.0.bias", "enc1.net.2.weight", "enc1.net.2.bias"}
    assert tuple(s1["stack.encoders.0.net.0.weight"].shape) == (128, 3 * 128) and tuple(s0["stack.encoders.0.net.0.weight"].shape) == (128, 128)
    assert tuple(s1["enc1.net.0.weight"].shape) == (128, 3 * 128)
    # the global RNG stream after the build is the K = 1 build's: a module built next draws the same values
    torch.manual_seed(4)
    M.WMAux(cfg_of(1))
    n0 = torch.randn(3)
    torch.manual_seed(4)
    M.WMAux(cfg_of(3))
    assert torch.equal(n0, torch.randn(3))


def test_v8fmini_k3_leaves_every_non_wm_parameter_and_the_later_rng_stream_unchanged():
    mu, sd = torch.zeros(M.N_CONS), torch.ones(M.N_CONS)
    ms = []
    for k in (1, 3):
        torch.manual_seed(2)
        ms.append(M.V8FMini(cfg_of(k), M.P10Prior.synthetic(0), mu, sd))
    s0, s1 = ms[0].state_dict(), ms[1].state_dict()
    rest = [k for k in s0 if not k.startswith("wm.")]
    assert rest and all(torch.equal(s0[k], s1[k]) for k in rest)
    assert "wm_k" not in ms[0].built_levers()                                  # a W0 model keeps its lever key set
    lv = ms[1].built_levers()
    assert (lv["wm_k"], lv["wm_k_levels"], lv["wm_dilation_levels"], lv["wm_causal_pad_levels"]) == (3, [3, 3, 3], [1, 2, 2], [True, True, True])


# ------------------------------------------------------------------------------------------------ shapes at hmax 28
@pytest.mark.parametrize("k", [1, 3])
def test_hmax28_states_transitions_and_rollouts(k):
    wm = wm_of(k).eval()
    adapt, act = data()
    z1 = wm.z1(adapt)
    assert z1.shape == (B, 28, 128)
    enc = wm.stack.encode(z1, wm.scale_actions(act)[:, :-1])
    assert [tuple(z.shape) for z in enc.z] == [(B, 28, 128), (B, 14, 128), (B, 7, 128)]
    assert [tuple(a.shape) for a in enc.act] == [(B, 27, 32), (B, 13, 32), (B, 6, 32)]
    for l, ks in {1: (1, 5, 10), 2: (1, 3), 3: (1, 3)}.items():              # the rollouts the WMACT / LATSEM reads take
        acts = wm.scale_actions(act)[:, :max(ks)] if l == 1 else enc.act[l - 1][:, :max(ks)]
        roll = wm.stack.rollout(l, enc.z[l - 1][:, 0], acts)
        assert roll.shape == (B, max(ks), 128) and torch.isfinite(roll).all()
    out = wm(adapt, act)
    assert torch.isfinite(out["total"]) and {f"{t}/L{l}" for t in ("pred", "sigreg", "idm") for l in (1, 2, 3)} <= set(out)


# ------------------------------------------------------------------------------------------------ causality and reads (the live checks)
def test_k3_causality_at_every_level():
    check_causality(wm_of(3).eval())


def test_k1_causality_too_the_earlier_stack_is_causal_up_to_its_window_anchor():
    wm = wm_of(1).eval()                                                       # window 1: state i = z1[i * s], so the same bound holds
    check_causality(wm)


def test_k3_states_read_exactly_the_documented_lower_states():
    wm = wm_of(3).eval()
    check_reads(wm)
    check_z1(wm)


def test_k3_history_extension_leaves_full_history_states_unchanged():
    check_history_extension(wm_of(3).eval())


def test_k3_dilations_are_each_levels_own_step():
    wm = wm_of(3)
    c = wm.stack.cfg
    assert [(c[1].window, c[1].dilation, c[1].stride), (c[2].window, c[2].dilation, c[2].stride)] == [(3, 2, 2), (3, 2, 2)]
    wm = wm_of(2, wm_strides=(2, 3), hmax=28)
    c = wm.stack.cfg
    assert (c[1].dilation, c[2].dilation) == (2, 3) and wm.built_k()["dilation"] == [1, 2, 3]


# ------------------------------------------------------------------------------------------------ mutations: each must turn a check red
def test_mutation_deleting_the_dilation_turns_the_reads_check_red(monkeypatch):
    check_reads(wm_of(3).eval())
    orig = H._windows
    monkeypatch.setattr(H, "_windows", lambda z, w, s, d=1: orig(z, w, s, 1))     # the dilation is ignored: a wrong read (AssertionError) or a state count the macro-actions cannot follow (RuntimeError)
    with pytest.raises((AssertionError, RuntimeError)):
        check_reads(wm_of(3).eval())


def test_mutation_deleting_the_causal_pad_turns_causality_and_shapes_red(monkeypatch):
    check_causality(wm_of(3).eval())

    def no_pad(self, level, z, *, strict=False):                               # H-JEPA's _apply_encoder with the causal front-pad line deleted
        c = self.cfg[level - 1]
        lead, (t_, d_) = z.shape[:-2], z.shape[-2:]
        win = H._windows(z.reshape(-1, t_, d_), c.window, c.stride, c.dilation)
        out = self.encoders[level - 2](win)
        return out.reshape(*lead, out.shape[-2], out.shape[-1])
    monkeypatch.setattr(H.HJEPAStack, "_apply_encoder", no_pad)
    wm = wm_of(3).eval()
    adapt, act = data()
    enc = wm.stack.encode(wm.z1(adapt), wm.scale_actions(act)[:, :-1])
    assert enc.z[1].shape[1] != 14                                             # the state count moves (the shape test is red)
    with pytest.raises(AssertionError):
        check_causality(wm)                                                    # the state at i reads the FUTURE (red)


def test_mutation_deleting_z1s_front_pad_turns_the_z1_check_red(monkeypatch):
    check_z1(wm_of(3).eval())

    def no_pad(self, adapt):
        p = self.proj(adapt.mean(2))
        return self.enc1(H._windows(p, self.enc1.window, 1, 1))              # n = F - K + 1 frames: the pad line is gone
    monkeypatch.setattr(M.WMAux, "z1", no_pad)
    with pytest.raises(AssertionError):
        check_z1(wm_of(3).eval())


def test_mutation_a_future_reading_z1_turns_causality_red(monkeypatch):
    def future(self, adapt):                                                   # frame j reads frames j, j+1, j+2 (the window anchored forward)
        p = self.proj(adapt.mean(2))
        K = self.enc1.window
        return self.enc1(H._windows(torch.cat([p, p[:, -1:].expand(-1, K - 1, -1)], 1), K, 1, 1))
    monkeypatch.setattr(M.WMAux, "z1", future)
    with pytest.raises(AssertionError):
        check_causality(wm_of(3).eval())


# ------------------------------------------------------------------------------------------------ config validation
def test_config_refuses_a_lever_that_builds_nothing_and_a_bad_k():
    with pytest.raises(ValueError, match="wm_k"):
        M.V8FMiniConfig(**{**TINY, "wm": "off", "wm_k": 3})
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="wm_k"):
            M.V8FMiniConfig(**{**TINY, "wm": "b", "hmax": 28, "wm_k": bad})
    assert M.V8FMiniConfig(**{**TINY, "wm": "off"}).wm_k == 1
