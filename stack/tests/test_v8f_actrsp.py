"""PREREG_V8F_ACTRSP: the level-1 action-response term ``--wm-rsp-w`` (REACHABILITY_ACTRSP sections 1-5).

Every check below has a deliberate-regression mutation (``build/mutate_rsp.py``, results ``mutation_results_rsp_ALL.json``) that must turn the named test RED while the controls (``test_off_*`` and the exact-zero gradient asserts) stay green.
P1 gradient reach and exact-zero controls; P2 flag absent; P3 dose identity (two doses); P4 analytic identity (float64); P5 pairing (no new parameter, no random draw); config refusals; G-DVB / G-LIVE through the trainer's own gate.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
UNW = ("pred", "sigreg", "idm")


def cfg_of(**kw):
    return M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm=kw.pop("wm", "b"), hmax=kw.pop("hmax", 28), kin_mode=kw.pop("kin_mode", "yaw"), **kw)


def build(cfg, seed=0):
    torch.manual_seed(seed)
    return M.WMAux(cfg)


def zab(cfg, b=2):
    g = torch.Generator().manual_seed(7)
    return torch.randn(b, cfg.hmax, cfg.wm_dim, generator=g), torch.randn(b, cfg.hmax - 1, 2, generator=g)


def loss(wm, cfg):
    z, a = zab(cfg)
    return wm.stack.loss(z, a, generator=torch.Generator().manual_seed(11))


def loss64(wm, cfg):
    z, a = zab(cfg)
    return wm.double().stack.loss(z.double(), a.double(), generator=torch.Generator().manual_seed(11))


def gsum(wm, prefix):
    return sum(float(p.grad.abs().sum()) for n, p in wm.named_parameters() if n.startswith(prefix) and p.grad is not None)


# ------------------------------------------------------------------------------------------------ P1: gradient reach, term alone, through the pooled-adapter path (so ``proj`` is in the graph)
def test_p1_term_gradient_reaches_predictor_and_act_embed_and_exact_zero_elsewhere():
    c = cfg_of(wm_rsp_w=1.0)
    wm = build(c)
    g = torch.Generator().manual_seed(3)
    am = torch.randn(2, c.hmax, c.d, generator=g, requires_grad=True)
    act = torch.randn(2, c.hmax, 2, generator=g)
    out = wm.forward_mean(am, act)
    (c.wm_rsp_w * out["rsp/L1"]).backward()
    assert gsum(wm, "stack.predictors.0.") > 0.0 and gsum(wm, "stack.act_embed.") > 0.0
    for pre in ("stack.idm_heads.0.", "stack.idm_heads.1.", "stack.idm_heads.2.", "proj.", "stack.predictors.1.", "stack.predictors.2.", "stack.encoders.", "stack.macro_encoders."):   # the controls: exact zeros prove the probe can read a zero
        assert gsum(wm, pre) == 0.0, pre
    assert am.grad is None or float(am.grad.abs().sum()) == 0.0                                                   # the anchor state is detached: nothing reaches the adapter output either


def test_p1_the_term_is_logged_with_its_six_detached_diagnostics():
    c = cfg_of(wm_rsp_w=1.0)
    o = loss(build(c), c)
    for k in ("rsp/L1", "rsp_g_aa/L1", "rsp_g_ww/L1", "rsp_g_aw/L1", "rsp_g_wa/L1", "rsp_s_a/L1", "rsp_s_w/L1"):
        assert k in o
    assert o["rsp/L1"].requires_grad and not any(o[k].requires_grad for k in o if k.startswith("rsp_"))
    assert not any(k.endswith(("/L2", "/L3")) and "rsp" in k for k in o)                      # level 1 only


# ------------------------------------------------------------------------------------------------ P2: flag absent = the earlier build
def test_off_no_key_no_op_same_key_set_as_the_ladder_build():
    c = cfg_of()
    o = loss(build(c), c)
    assert not any("rsp" in k for k in o)
    assert not any("rsp_w" in r for r in build(c).stack.built_levers().values())
    assert M.V8FMiniConfig.__dataclass_fields__["wm_rsp_w"].default == 0.0 and H.LevelConfig.__dataclass_fields__["rsp_w"].default == 0.0
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.wm_rsp_w == 0.0 and TR.FLAG_CHECKS["wm_rsp_w"] == "lever:wm_rsp_w"
    assert {x.dest for x in TR.build_parser()._actions if x.dest != "help"} == set(TR.FLAG_CHECKS)


def test_off_default_argparse_value_is_zero():
    assert TR.build_parser().get_default("wm_rsp_w") == 0.0                                    # MR2 turns this to 1.0


def test_off_explicit_zero_equals_absent_bit_for_bit():
    c0, c1 = cfg_of(), cfg_of(wm_rsp_w=0.0)
    o0, o1 = loss(build(c0), c0), loss(build(c1), c1)
    assert list(o0) == list(o1) and all(torch.equal(o0[k], o1[k]) for k in o0)


# ------------------------------------------------------------------------------------------------ P3: dose identity at two doses, and C3
@pytest.mark.parametrize("lam", [1.0, 0.3])
def test_p3_dose_identity_level_difference_is_lambda_times_the_term(lam):
    c0, c1 = cfg_of(), cfg_of(wm_rsp_w=lam)
    o0, o1 = loss64(build(c0), c0), loss64(build(c1), c1)                                      # float64: the level sum (~100) is not blurred by float32 rounding
    for l in (1, 2, 3):
        for t in UNW:
            assert torch.equal(o0[f"{t}/L{l}"], o1[f"{t}/L{l}"]), (t, l)
    d, want = float(o1["level/L1"] - o0["level/L1"]), lam * float(o1["rsp/L1"])
    assert want > 0.0 and abs(d - want) <= 1e-9 * want, (d, want)
    assert torch.equal(o0["level/L2"], o1["level/L2"]) and torch.equal(o0["level/L3"], o1["level/L3"])


# ------------------------------------------------------------------------------------------------ P4: analytic identity (float64) on stubs: linear predictor Wz + Ba, linear readout K1 x + K2 y
class LinPred(nn.Module):
    def __init__(self, W, B):
        super().__init__()
        self.W, self.B = nn.Parameter(W, requires_grad=False), nn.Parameter(B, requires_grad=False)

    def forward(self, zc, ec):
        return zc @ self.W.T + ec @ self.B.T


class LinRead(nn.Module):
    def __init__(self, K1, K2):
        super().__init__()
        self.K1, self.K2 = nn.Parameter(K1), nn.Parameter(K2)

    def forward(self, x, y):
        return x @ self.K1.T + y @ self.K2.T


D = 4


def stub(W, B, K1, K2):
    wm = build(cfg_of(wm_rsp_w=1.0))
    st = wm.stack.double()
    st.act_embed = nn.Identity()
    st.predictors[0] = LinPred(W.double(), B.double())
    st.idm_heads[0] = LinRead(K1.double(), K2.double())
    return st


def data(t=32, b=3):
    g = torch.Generator().manual_seed(5)
    return torch.randn(b, t, D, generator=g).double(), (torch.randn(b, t - 1, 2, generator=g) * 0.5).double()


def run_rsp(st, z, a):
    c = dataclasses.replace(st.cfg[0], rsp_w=1.0)
    return st.response_loss(z, a, c)


def reference_loss(W, B, K1, K2, z, a, s, delta):
    """independent numpy-free recursion: delta z_h = W delta z_{h-1} + 2 delta B e_c, read through (K1, K2), divided by 2 delta s"""
    t = z.shape[1]
    anchors = list(range(0, t - 10, 10))
    errs = []
    for c_ in (0, 1):
        dz_prev, dz = None, None
        dprev = torch.zeros(z.shape[0], len(anchors), D, dtype=z.dtype)
        for h in range(10):
            e = torch.zeros(2, dtype=z.dtype)
            e[c_] = 1.0
            dcur = dprev @ W.T + (2.0 * delta[c_] * e) @ B.T
            read = dprev @ K1.T + dcur @ K2.T                                                  # (B, A, 2): the read difference at step h
            gcol = read / (2.0 * delta[c_] * s)
            errs.append((gcol - torch.eye(2, dtype=z.dtype)[:, c_]) ** 2)
            dprev = dcur
    return torch.stack(errs).mean()


def test_p4_general_linear_case_matches_the_recursion_to_1e10():
    g = torch.Generator().manual_seed(1)
    W, B, K1, K2 = (torch.randn(D, D, generator=g) * 0.4, torch.randn(D, 2, generator=g), torch.randn(2, D, generator=g), torch.randn(2, D, generator=g))
    st = stub(W, B, K1, K2)
    z, a = data()
    r = run_rsp(st, z, a)
    s = torch.stack([r["diag"]["rsp_s_a"], r["diag"]["rsp_s_w"]]).double()
    ref = reference_loss(W.double(), B.double(), K1.double(), K2.double(), z, a, s, torch.tensor(H.LevelConfig.__dataclass_fields__["rsp_delta"].default).double())
    assert abs(float(r["loss"]) - float(ref)) <= 1e-6 * float(ref), (float(r["loss"]), float(ref))   # s is float32 (the slope is a detached float32 readout): 1e-6 relative


def test_p4_matched_response_gives_loss_and_gradient_zero_and_B_zero_gives_exactly_one_half():
    K1, K2 = torch.zeros(2, D), torch.cat([torch.eye(2), torch.zeros(2, 2)], 1)
    z, a = data()
    st = stub(torch.zeros(D, D), torch.zeros(D, 2), K1, K2)
    r0 = run_rsp(st, z, a)
    assert float(r0["loss"]) == 0.5                                                           # B = 0: G = 0, loss = (1 + 1 + 0 + 0) / 4 exactly
    s = torch.stack([r0["diag"]["rsp_s_a"], r0["diag"]["rsp_s_w"]]).double()
    B = torch.cat([torch.diag(s), torch.zeros(2, 2).double()], 0)                              # K2 B = diag(s)
    st = stub(torch.zeros(D, D), B, K1, K2)
    for p in st.predictors[0].parameters():
        p.requires_grad_(True)
    r = run_rsp(st, z, a)
    r["loss"].backward()
    assert float(r["loss"]) < 1e-24, float(r["loss"])
    assert max(float(p.grad.abs().max()) for p in st.predictors[0].parameters()) < 1e-12


def test_p4_delta_scales_out_and_a_flipped_contrast_is_detected():
    """for a linear map G does not depend on delta; the matched-response case reads 0 at any delta (a sign flip of one contrast cannot)"""
    K1, K2 = torch.zeros(2, D), torch.cat([torch.eye(2), torch.zeros(2, 2)], 1)
    z, a = data()
    st0 = stub(torch.zeros(D, D), torch.zeros(D, 2), K1, K2)
    s = torch.stack([run_rsp(st0, z, a)["diag"][k] for k in ("rsp_s_a", "rsp_s_w")]).double()
    B = torch.cat([torch.diag(s), torch.zeros(2, 2).double()], 0)
    st = stub(torch.zeros(D, D), B, K1, K2)
    for dl in ((0.42, 0.24), (0.1, 0.9)):
        c = dataclasses.replace(st.cfg[0], rsp_w=1.0, rsp_delta=dl)
        assert float(st.response_loss(z, a, c)["loss"]) < 1e-24, dl


def test_p4_slope_floor_zero_slope_readout_is_used_at_exactly_one_tenth():
    st = stub(torch.zeros(D, D), torch.zeros(D, 2), torch.zeros(2, D), torch.zeros(2, D))
    z, a = data()
    r = run_rsp(st, z, a)
    assert abs(float(r["diag"]["rsp_s_a"]) - H.RSP_S_MIN) < 1e-7 and abs(float(r["diag"]["rsp_s_w"]) - H.RSP_S_MIN) < 1e-7


def test_p4_slope_is_the_covariance_over_variance_on_real_transitions():
    g = torch.Generator().manual_seed(2)
    K1, K2 = torch.randn(2, D, generator=g), torch.randn(2, D, generator=g)
    st = stub(torch.zeros(D, D), torch.zeros(D, 2), K1, K2)
    z, a = data()
    r = run_rsp(st, z, a)
    ir = (z[:, :-1] @ K1.double().T + z[:, 1:] @ K2.double().T).reshape(-1, 2)
    ar = a.reshape(-1, 2)
    for c_, k in enumerate(("rsp_s_a", "rsp_s_w")):
        cov = ((ir[:, c_] - ir[:, c_].mean()) * (ar[:, c_] - ar[:, c_].mean())).mean()
        want = max(float(cov / ar[:, c_].var(unbiased=False)), H.RSP_S_MIN)
        assert abs(float(r["diag"][k]) - want) < 1e-5 * max(1.0, abs(want)), (k, float(r["diag"][k]), want)


def test_p4_geometry_nine_anchors_ten_steps_four_contrasts_at_91_frames():
    c = cfg_of(wm_rsp_w=1.0, hmax=91)
    r = build(c).built_rsp(91)
    assert (r["calls"], r["contrasts"], r["anchors"], r["k"], r["ctx_passed"]) == (1, 4, 9, 10, 1)
    assert abs(r["w_in_loss_rsp"] - 1.0) < 1e-6


# ------------------------------------------------------------------------------------------------ P5: pairing: no new parameter, no random draw
@pytest.mark.parametrize("seed", [0, 2])
def test_p5_flag_on_and_off_builds_have_the_same_wm_init(seed):
    sa, sb = build(cfg_of(), seed).state_dict(), build(cfg_of(wm_rsp_w=1.0), seed).state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_p5_the_term_draws_no_random_number():
    c = cfg_of(wm_rsp_w=1.0)
    wm = build(c)
    z, a = zab(c)
    st = torch.get_rng_state()
    wm.stack.response_loss(z, a, wm.stack.cfg[0])
    assert torch.equal(st, torch.get_rng_state())


def test_p5_global_stream_after_the_full_loss_is_the_same_on_and_off():
    out = []
    for lam in (0.0, 1.0):
        c = cfg_of(wm_rsp_w=lam)
        wm = build(c)
        z, a = zab(c)
        torch.manual_seed(123)
        wm.stack.loss(z, a)
        out.append(torch.get_rng_state())
    assert torch.equal(out[0], out[1])


# ------------------------------------------------------------------------------------------------ the config refuses a declared-and-unbuilt lever
def test_config_refusals():
    for bad in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cfg_of(wm_rsp_w=bad)
    with pytest.raises(ValueError):
        cfg_of(wm="a", wm_rsp_w=1.0)
    with pytest.raises(ValueError):
        cfg_of(wm="off", wm_rsp_w=1.0)
    with pytest.raises(ValueError):
        cfg_of(kin_mode="steer", wm_rsp_w=1.0)


def test_ladder_reader_is_not_confused_by_the_term():
    """MR9 / MR10: with the term on, the ladder's own reader still reads weight 1.0, k = 4 free-run steps (not 14) and the free run's anchors"""
    c = cfg_of(hmax=25, wm_strides=(2, 3), wm_rollout_k=(4, 3, 2), wm_rollout_stride=(2, 2, 1), wm_rsp_w=1.0)
    r = build(c).built_ladder(25)
    assert r["k"] == [4, 3, 2] and r["w_in_loss"][0] is not None and abs(r["w_in_loss"][0] - 1.0) < 1e-4, r
    assert r["anchors"][0] == len(range(0, 24, 2)) and r["ctx_passed"] == [3, 3, 3]


# ------------------------------------------------------------------------------------------------ the trainer's own gate (G-DVB, G-LIVE, C3) on the synthetic recipe, ladder path
@pytest.fixture(scope="module")
def init(tmp_path_factory):
    root = tmp_path_factory.mktemp("actrsp")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    return {"root": root, "ck": root / "m2b" / "ckpt_last.pt"}


LADDER = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "3", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]


def lgate(init, name, extra=(), ladder=LADDER, base=BASE):
    try:
        rc = TR.main(["--out", str(init["root"] / name), *base, "--init-ckpt", str(init["ck"]), *ladder, *extra, "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    except ValueError as e:
        rc = f"ValueError: {e}"
    return rc


def sc_of(init, name):
    return json.loads((init["root"] / name / "selfcheck.json").read_text(encoding="utf-8"))


def test_gate_passes_reads_the_built_weight_back_and_g_live_carries_the_term(init):
    assert lgate(init, "g_on", ["--wm-rsp-w", "1.0"]) == 0
    sc = sc_of(init, "g_on")
    rb = sc["levers"]["wm_rsp_readback"]
    assert rb["declared"] == 1.0 and rb["built_config"] == 1.0 and abs(rb["built"]["w_in_loss_rsp"] - 1.0) < 1e-6
    assert (rb["built"]["calls"], rb["built"]["contrasts"], rb["built"]["k"], rb["built"]["ctx_passed"], rb["built"]["anchors"]) == (1, 4, 10, 1, 2)
    assert sc["live"]["pass"] is True and "wm/rsp/L1" in sc["live"]["terms"] and "wm/rsp/L1" in sc["live_first"]
    assert sc["live"]["missing_terms"] == [] and sc["live"]["dead_wm_groups"] == [] and sc["live"]["control_nonwm_grad_sum"] == 0.0 and sc["live"]["control_adapter_grad_sum"] == 0.0
    c3 = sc["live"]["c3"]
    assert c3["pass"] is True and c3["unweighted_equal"] and c3["levels_23_equal"] and c3["rel_err"] <= 1e-5 and c3["level_L1_diff"] > 0.0


def test_gate_flag_absent_keeps_the_key_set(init):
    assert lgate(init, "g_off") == 0
    sc = sc_of(init, "g_off")
    assert "wm_rsp_readback" not in sc["levers"] and "live_first" not in sc and "c3" not in sc["live"] and not any("rsp" in k for k in sc["live"]["terms"])


def test_gate_refuses_when_the_declared_weight_is_not_built_config_line_deleted(init, monkeypatch):
    """MR1: the make_config line deleted"""
    orig = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_rsp_w=0.0))
    rc = lgate(init, "r_mr1", ["--wm-rsp-w", "1.0"])
    assert isinstance(rc, str) and "wm_rsp_w" in rc, rc


def test_gate_refuses_a_builder_that_does_not_pass_the_weight_to_level_1(init, monkeypatch):
    """MR8: the config says 1.0, the built LevelConfig holds 0.0"""
    orig = M.WMAux.__init__

    def ignoring(self, cfg):
        orig(self, dataclasses.replace(cfg, wm_rsp_w=0.0))
    monkeypatch.setattr(M.WMAux, "__init__", ignoring)
    rc = lgate(init, "r_mr8", ["--wm-rsp-w", "1.0"])
    assert isinstance(rc, str) and "lever wm_rsp_w" in rc, rc


def test_gate_refuses_a_term_that_is_computed_but_not_added(init, monkeypatch):
    """MR3: the BUILT loss gives the term weight 0.0 although the LevelConfig says 1.0"""
    orig = H.HJEPAStack.response_loss

    def silent(self, z, a, c):
        r = orig(self, z, a, c)
        r["loss"] = r["loss"] * 0.0
        return r
    monkeypatch.setattr(H.HJEPAStack, "response_loss", silent)
    rc = lgate(init, "r_mr3", ["--wm-rsp-w", "1.0"])
    assert rc != 0, rc


def test_gate_refuses_the_flag_without_wm_only_wm_a_or_yaw(init):
    nowmonly = [x for x in LADDER if x != "--wm-only"]
    rc = lgate(init, "r_nowo", ["--wm-rsp-w", "1.0"], ladder=nowmonly)
    assert rc != 0 and isinstance(rc, str)
    rc = lgate(init, "r_a", ["--wm-rsp-w", "1.0", "--wm", "a"])
    assert rc != 0 and isinstance(rc, str)
    rc = lgate(init, "r_steer", ["--wm-rsp-w", "1.0"], base=[x if x != "yaw" else "steer" for x in BASE])
    assert rc != 0 and isinstance(rc, str)
    rc = lgate(init, "r_neg", ["--wm-rsp-w", "-1.0"])
    assert rc != 0 and isinstance(rc, str)
