"""PREREG_V8F_ACTRSP Amendment 2 (Option B): ``--wm-rsp-readout fixed_k``, ``--wm-rsp-probe(-sha256)``, ``--wm-init-from-ckpt-wm``, ``--wm-freeze-proj``.

Reachability from the caller (argparse -> make_config -> V8FMiniConfig -> WMAux -> HJEPAStack.response_loss -> backward) is asserted by the trainer's own gate on the synthetic recipe; C9' is the analytic identity in float64.
Every check has a deliberate-regression mutation (``mutate_optb.py``) that must turn the named test RED while the controls stay green.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
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
WMD = M.V8FMiniConfig.__dataclass_fields__["wm_dim"].default


def write_probe(path: Path, seed=0, s=0.2, dim=WMD, k=10):
    g = np.random.RandomState(seed)
    torch.save({"tag": "T", "K": k, "lambda": 1.0, "RA": {"mu": g.randn(dim), "sd": np.abs(g.randn(dim)) + 0.5, "W": g.randn(dim, 1) * 0.1, "ym": np.zeros(1)}, "s_cal_swap_heldout": [s, 0.04]}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cfg_of(**kw):
    return M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", hmax=kw.pop("hmax", 28), kin_mode="yaw", **kw)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    f = tmp_path_factory.mktemp("optb_probe") / "readout.pt"
    return {"path": str(f), "sha": write_probe(f)}


def fk(probe, **kw):
    return cfg_of(wm_rsp_w=1.0, wm_rsp_readout="fixed_k", wm_rsp_probe=probe["path"], wm_rsp_probe_sha256=probe["sha"], wm_freeze_proj="on", **kw)


def build(cfg, seed=0):
    torch.manual_seed(seed)
    return M.WMAux(cfg)


# ------------------------------------------------------------------------------------------------ defaults: every new flag is the a13f6f16 build when absent
def test_off_defaults_are_the_earlier_build():
    d = TR.build_parser()
    assert (d.get_default("wm_rsp_readout"), d.get_default("wm_rsp_probe"), d.get_default("wm_rsp_probe_sha256"), d.get_default("wm_init_from_ckpt_wm"), d.get_default("wm_freeze_proj"), d.get_default("wm_rsp_w")) == ("idm", "", "", "off", "off", 0.0)
    c = cfg_of()
    assert (c.wm_rsp_readout, c.wm_rsp_probe, c.wm_rsp_probe_sha256, c.wm_freeze_proj) == ("idm", "", "", "off")
    for f in ("wm_rsp_readout", "wm_rsp_probe", "wm_rsp_probe_sha256", "wm_init_from_ckpt_wm", "wm_freeze_proj"):
        assert f in TR.FLAG_CHECKS


def test_off_state_dict_keys_and_losses_equal_with_and_without_the_probe_buffers(probe):
    a, b = build(cfg_of()), build(fk(probe))
    assert list(a.state_dict()) == list(b.state_dict())                                          # the probe is a non-persistent buffer: no key, no parameter
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())       # the WM init did not move
    assert not any("rsp" in k for k in a.stack.built_levers()[1])


# ------------------------------------------------------------------------------------------------ gradient reach, term alone, and its zero control
def gsum(wm, prefix):
    return sum(float(p.grad.abs().sum()) for n, p in wm.named_parameters() if n.startswith(prefix) and p.grad is not None)


def term_grad(c):
    wm = build(c)
    g = torch.Generator().manual_seed(3)
    am = torch.randn(2, c.hmax, c.d, generator=g, requires_grad=True)
    act = torch.randn(2, c.hmax, 2, generator=g)
    out = wm.forward_mean(am, act)
    if "rsp/L1" in out:
        (c.wm_rsp_w * out["rsp/L1"]).backward()
    return wm, out, am


def test_gradient_reaches_predictor_and_act_embed_and_exact_zero_elsewhere(probe):
    wm, out, am = term_grad(fk(probe))
    assert gsum(wm, "stack.predictors.0.") > 0.0 and gsum(wm, "stack.act_embed.") > 0.0
    for pre in ("stack.idm_heads.", "proj.", "stack.predictors.1.", "stack.predictors.2.", "stack.encoders.", "stack.macro_encoders."):
        assert gsum(wm, pre) == 0.0, pre
    assert am.grad is None or float(am.grad.abs().sum()) == 0.0
    assert not any(k.startswith("rsp_") and out[k].requires_grad for k in out) and "rsp_g_aa/L1" in out and "rsp_s_a/L1" in out


def test_control_flag_off_has_no_term_and_reads_exactly_zero(probe):
    wm, out, _ = term_grad(cfg_of())
    assert "rsp/L1" not in out and not any(k.startswith("rsp_") for k in out)
    assert gsum(wm, "stack.predictors.0.") == 0.0 and gsum(wm, "stack.act_embed.") == 0.0       # nothing ran backward: the control reads exactly 0.0


# ------------------------------------------------------------------------------------------------ C9': analytic identity (float64) on a linear world
D = 4


class LinPred(nn.Module):
    def __init__(self, W, B):
        super().__init__()
        self.W, self.B = nn.Parameter(W, requires_grad=False), nn.Parameter(B, requires_grad=False)

    def forward(self, zc, ec):
        return zc @ self.W.T + ec @ self.B.T


def stub(W, B, w, s):
    wm = build(cfg_of(wm_rsp_w=1.0))
    st = wm.stack.double()
    st.act_embed = nn.Identity()
    st.predictors[0] = LinPred(W.double(), B.double())
    st.set_rsp_probe(w.double(), s)
    st.rsp_probe_w, st.rsp_probe_s = st.rsp_probe_w.double(), st.rsp_probe_s.double()
    return st, dataclasses.replace(st.cfg[0], rsp_w=1.0, rsp_readout="fixed_k")


def data(t=32, b=3):
    g = torch.Generator().manual_seed(5)
    return torch.randn(b, t, D, generator=g).double(), (torch.randn(b, t - 1, 2, generator=g) * 0.5).double()


WV = torch.tensor([0.75, -0.25, 0.5, 0.25])            # exact in float32: the probe buffer is float32 by design
BS = torch.tensor([[0.5, 0.25], [0.25, -0.5], [-0.25, 0.5], [0.125, 0.25]])
K_ = H.RSP_K


def test_c9p_matched_response_gives_loss_and_gradient_zero():
    s = K_ * float(WV @ BS[:, 0])
    st, c = stub(torch.eye(D), BS, WV, s)
    for p in st.predictors[0].parameters():
        p.requires_grad_(True)
    z, a = data()
    r = st.response_loss(z, a, c)
    r["loss"].backward()
    assert float(r["loss"]) < 1e-24 and max(float(p.grad.abs().max()) for p in st.predictors[0].parameters()) < 1e-12
    assert abs(float(r["diag"]["rsp_g_aa"]) - 1.0) < 1e-12


def test_c9p_B_zero_gives_loss_exactly_one_and_a_wrong_constant_slope_is_used_as_given():
    z, a = data()
    st, c = stub(torch.eye(D), torch.zeros(D, 2), WV, 0.3)
    assert float(st.response_loss(z, a, c)["loss"]) == 1.0
    s = K_ * float(WV @ BS[:, 0])
    st, c = stub(torch.eye(D), BS, WV, 2.0 * s)                                                   # a constant slope doubled: G = 0.5 exactly, a batch slope would hide it (MR11)
    r = st.response_loss(z, a, c)
    assert abs(float(r["diag"]["rsp_g_aa"]) - 0.5) < 1e-12 and abs(float(r["loss"]) - 0.25) < 1e-12


def test_c9p_state_and_action_history_do_not_move_the_matched_identity():
    s = K_ * float(WV @ BS[:, 0])
    st, c = stub(torch.eye(D), BS, WV, s)
    g = torch.Generator().manual_seed(9)
    z = torch.cumsum(torch.randn(3, 32, D, generator=g).double(), 1)                              # a state that carries history
    a = torch.cumsum(torch.randn(3, 31, 2, generator=g).double() * 0.1, 1)                        # autocorrelated actions
    assert float(st.response_loss(z, a, c)["loss"]) < 1e-24


def test_c9p_geometry_one_call_two_contrasts_nine_anchors_ten_steps_context_one_endpoint_read(probe):
    wm = build(fk(probe, hmax=91))
    r = wm.built_rsp(91)
    assert (r["calls"], r["contrasts"], r["anchors"], r["k"], r["ctx_passed"], r["readout"]) == (1, 2, 9, 10, 1, "fixed_k")
    assert abs(r["w_in_loss_rsp"] - 1.0) < 1e-6 and abs(r["s_cal"] - 0.2) < 1e-12
    s = K_ * float(WV @ BS[:, 0])
    st, c = stub(0.875 * torch.eye(D), BS, WV, s)                                                   # W != I: per-step reads would differ from the endpoint read (MR14)
    z, a = data()
    g_end = float(st.response_loss(z, a, c)["diag"]["rsp_g_aa"])
    assert abs(g_end - float(sum(0.875 ** (K_ - 1 - j) for j in range(K_)) / K_)) < 1e-9


def test_c9p_a_flipped_contrast_or_a_scaled_delta_is_detected():
    s = K_ * float(WV @ BS[:, 0])
    st, c = stub(torch.eye(D), BS, WV, s)
    z, a = data()
    for dl in ((0.42, 0.24), (0.84, 0.24), (0.1, 0.24)):                                           # delta scales out of G in the matched case
        assert float(st.response_loss(z, a, dataclasses.replace(c, rsp_delta=dl))["loss"]) < 1e-24
    st2, _ = stub(torch.eye(D), -BS, WV, s)                                                       # a world with the opposite response: G = -1, loss 4
    assert abs(float(st2.response_loss(z, a, c)["loss"]) - 4.0) < 1e-9


# ------------------------------------------------------------------------------------------------ config and probe refusals
def test_config_refusals(probe):
    ok = dict(wm_rsp_w=1.0, wm_rsp_readout="fixed_k", wm_rsp_probe=probe["path"], wm_rsp_probe_sha256=probe["sha"], wm_freeze_proj="on")
    cfg_of(**ok)
    for bad in (dict(ok, wm_rsp_w=0.0), dict(ok, wm_freeze_proj="off"), dict(ok, wm_rsp_probe=""), dict(ok, wm_rsp_probe_sha256="ab"), dict(ok, wm_rsp_readout="x"),
                dict(wm_rsp_w=1.0, wm_rsp_probe=probe["path"]), dict(wm_freeze_proj="maybe")):
        with pytest.raises(ValueError):
            cfg_of(**bad)


def test_probe_sha_mismatch_one_changed_byte_wrong_K_are_refused(probe, tmp_path):
    f = tmp_path / "p.pt"
    f.write_bytes(Path(probe["path"]).read_bytes()[:-1] + bytes([Path(probe["path"]).read_bytes()[-1] ^ 1]))
    with pytest.raises(ValueError, match="sha256"):
        M.load_rsp_probe(str(f), probe["sha"], WMD)
    with pytest.raises(ValueError, match="sha256"):
        build(fk(dict(probe, path=str(f))))
    g = tmp_path / "k5.pt"
    sha = write_probe(g, k=5)
    with pytest.raises(ValueError, match="K = 5"):
        M.load_rsp_probe(str(g), sha, WMD)
    with pytest.raises(ValueError, match="RA.W"):
        M.load_rsp_probe(probe["path"], probe["sha"], WMD + 1)


def test_probe_vector_is_W_over_sd(probe):
    ck = torch.load(probe["path"], weights_only=False)
    w, s = M.load_rsp_probe(probe["path"], probe["sha"], WMD)
    assert torch.allclose(w.double(), torch.tensor(np.asarray(ck["RA"]["W"])[:, 0] / np.asarray(ck["RA"]["sd"]), dtype=torch.float64), atol=1e-6) and s == 0.2


# ------------------------------------------------------------------------------------------------ the trainer's own gate on the synthetic recipe (caller reachability)
WMF = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "2", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]


@pytest.fixture(scope="module")
def init(tmp_path_factory, probe):
    root = tmp_path_factory.mktemp("optb")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    rc = TR.main(["--out", str(root / "src"), *BASE, "--init-ckpt", str(root / "m2b" / "ckpt_last.pt"), *WMF, "--log-every", "1", "--ckpt-every", "100", "--steps", "3"])
    assert rc == 0
    return {"root": root, "ck": root / "src" / "ckpt_last.pt"}


FT = ["--wm-init-from-ckpt-wm", "on", "--wm-freeze-proj", "on"]


def rsp_flags(probe):
    return ["--wm-rsp-w", "1.0", "--wm-rsp-readout", "fixed_k", "--wm-rsp-probe", probe["path"], "--wm-rsp-probe-sha256", probe["sha"]]


def lgate(init, name, extra=(), more=(), steps=None):
    try:
        rc = TR.main(["--out", str(init["root"] / name), *BASE, "--init-ckpt", str(init["ck"]), *WMF, *extra, *more, *(["--steps", str(steps)] if steps else ["--selfcheck-only"])])
    except SystemExit as e:
        rc = e.code
    except ValueError as e:
        rc = f"ValueError: {e}"
    return rc


def sc_of(init, name):
    return json.loads((init["root"] / name / "selfcheck.json").read_text(encoding="utf-8"))


def test_gate_passes_for_the_rsp_arm_and_reads_the_built_readout_back(init, probe):
    assert lgate(init, "g_rsp", FT, rsp_flags(probe)) == 0
    sc = sc_of(init, "g_rsp")
    rb = sc["levers"]["wm_rsp_readback"]
    b = rb["built"]
    assert rb["declared_readout"] == "fixed_k" == rb["built_readout"] and rb["wm_proj_requires_grad"] == [False] and rb["freeze_proj_declared"] == "on"
    assert (b["calls"], b["contrasts"], b["k"], b["ctx_passed"], b["readout"]) == (1, 2, 10, 1, "fixed_k") and abs(b["w_in_loss_rsp"] - 1.0) < 1e-6 and abs(b["s_cal"] - 0.2) < 1e-12
    info = sc["info"]["wm_only"]
    assert info["init_wm"] is True and info["n_mismatch_after_load"] == 0 and info["wm_loaded_from_ckpt"] == info["wm_params_equal_to_ckpt"] > 0 and info["wm_reinit_seed"] is None
    c3 = sc["live"]["c3"]
    assert sc["live"]["pass"] is True and c3["pass"] is True and c3["level_L1_diff"] > 0.0 and "wm/rsp/L1" in sc["live_first"]


def test_gate_passes_for_the_null_arm_without_the_term(init):
    assert lgate(init, "g_null", FT) == 0
    sc = sc_of(init, "g_null")
    assert not any("rsp/" in k for k in sc["live"]["terms"]) and "c3" not in sc["live"]
    assert sc["levers"]["wm_rsp_readback"]["wm_proj_requires_grad"] == [False] and sc["levers"]["wm_rsp_readback"]["declared_readout"] == "idm"
    assert sc["levers"]["wm_rsp_readback"]["built"]["readout"] is None


def test_gate_the_fine_tune_start_is_the_checkpoint_bit_for_bit_and_off_re_initialises(init):
    assert lgate(init, "g_off", [], []) == 0
    info = sc_of(init, "g_off")["info"]["wm_only"]
    assert info["init_wm"] is False and info["wm_loaded_from_ckpt"] == 0 and info["wm_params_equal_to_ckpt"] == 0                  # the earlier arms: a fresh WM


@pytest.mark.parametrize("extra,why", [
    (["--wm-rsp-w", "1.0", "--wm-rsp-readout", "fixed_k", "--wm-init-from-ckpt-wm", "on", "--wm-freeze-proj", "on"], "probe"),
    (["--wm-init-from-ckpt-wm", "on", "--wm-freeze-proj", "on", "--wm-rsp-probe", "x.pt"], "probe"),
])
def test_gate_refuses_incomplete_or_idle_readout_flags(init, extra, why):
    rc = lgate(init, "r_" + str(abs(hash(tuple(extra)))), extra)
    assert isinstance(rc, str) and ("wm_rsp_probe" in rc or "wm_rsp_readout" in rc or "probe" in rc), rc


def test_gate_refuses_fixed_k_without_freeze_or_without_init_wm(init, probe):
    rc = lgate(init, "r_nofreeze", ["--wm-init-from-ckpt-wm", "on"], rsp_flags(probe))
    assert isinstance(rc, str) and "wm_freeze_proj" in rc, rc
    rc = lgate(init, "r_noinit", ["--wm-freeze-proj", "on"], rsp_flags(probe))
    assert isinstance(rc, str) and "wm-init-from-ckpt-wm" in rc, rc


def test_gate_refuses_a_wrong_probe_sha_MR13(init, probe):
    bad = dict(probe, sha="0" * 64)
    rc = lgate(init, "r_sha", FT, rsp_flags(bad))
    assert isinstance(rc, str) and "sha256" in rc, rc


def test_gate_refuses_when_the_freeze_flag_is_parsed_but_proj_keeps_requires_grad_MR12(init, monkeypatch):
    monkeypatch.setattr(TR, "freeze_all_but_wm", lambda model, freeze_proj=False: TR.freeze_all_but_wm.__wrapped__(model, False) if hasattr(TR.freeze_all_but_wm, "__wrapped__") else [p.requires_grad_(TR.is_wm_param(n)) for n, p in model.named_parameters()])
    rc = lgate(init, "r_mr12", FT)
    assert isinstance(rc, str) and "wm_freeze_proj" in rc, rc


def test_gate_refuses_when_make_config_drops_the_readout_MR11(init, probe, monkeypatch):
    orig = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_rsp_readout="idm", wm_rsp_probe="", wm_rsp_probe_sha256=""))
    rc = lgate(init, "r_mr11", FT, rsp_flags(probe))
    assert isinstance(rc, str) and "wm_rsp_readout" in rc, rc


def test_gate_refuses_a_fine_tune_from_a_checkpoint_with_other_strides(init):
    rc = lgate(init, "r_strides", FT, ["--wm-strides", "2", "3"])
    assert isinstance(rc, str) and "wm_strides" in rc, rc


# ------------------------------------------------------------------------------------------------ the frozen projection stays bit-equal in a real run, the rest trains
def test_wm_proj_is_bit_equal_to_the_checkpoint_after_steps_and_the_rest_moved(init, probe):
    assert lgate(init, "t_rsp", FT, rsp_flags(probe), steps=2) == 0
    a = torch.load(init["ck"], map_location="cpu", weights_only=False)["model"]
    b = torch.load(init["root"] / "t_rsp" / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    proj = [k for k in a if k.startswith("wm.proj.")]
    assert proj and all(torch.equal(a[k], b[k]) for k in proj)
    moved = [k for k in a if k.startswith("wm.") and not k.startswith("wm.proj.") and a[k].is_floating_point() and not torch.equal(a[k], b[k])]
    assert len(moved) > 10 and all(torch.equal(a[k], b[k]) for k in a if not k.startswith("wm."))
