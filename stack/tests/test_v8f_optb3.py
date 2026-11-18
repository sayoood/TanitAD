"""PREREG_V8F_ACTRSP Amendment 3 (Option B target arm): ``--wm-rsp-target`` (default 1.0): the fixed_k term becomes mean (G - g*)^2.

C10 flag parity (absent == 1.0 == the 86da79cd trainer, bit for bit), C11 the analytic identity in float64 on the C9' world, the trainer's own gate (declared == built, read back in the selfcheck), and the refusals.
Every check has a deliberate-regression mutation (``mutate_optb3.py``) that must turn the named test RED.
Run the WHOLE file (module-scoped fixtures): ``CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_optb3.py``. C10's third leg needs the 86da79cd export (`git archive 86da79cd stack taniteval`) unpacked at ``OPTB3_BASE`` (default ``C:/Users/Admin/optb3/base``); it is skipped, and says so, when absent.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_optb import BASE, BS, FT, K_, WMF, WV, cfg_of, data, fk, lgate, rsp_flags, sc_of, stub, write_probe  # noqa: E402

torch.set_num_threads(1)
BASE_TREE = Path(os.environ.get("OPTB3_BASE", "C:/Users/Admin/optb3/base"))
HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    f = tmp_path_factory.mktemp("optb3_probe") / "readout.pt"
    return {"path": str(f), "sha": write_probe(f)}


@pytest.fixture(scope="module")
def init(tmp_path_factory, probe):
    root = tmp_path_factory.mktemp("optb3")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    assert TR.main(["--out", str(root / "src"), *BASE, "--init-ckpt", str(root / "m2b" / "ckpt_last.pt"), *WMF, "--log-every", "1", "--ckpt-every", "100", "--steps", "3"]) == 0
    return {"root": root, "ck": root / "src" / "ckpt_last.pt"}


def tgt(probe, t):
    return [*rsp_flags(probe), "--wm-rsp-target", str(t)]


# ------------------------------------------------------------------------------------------------ defaults and the flag table
def test_default_is_one_and_the_flag_has_its_gate_entry():
    p = TR.build_parser()
    assert p.get_default("wm_rsp_target") == 1.0 and isinstance(p.get_default("wm_rsp_target"), float)
    assert "wm_rsp_target" in TR.FLAG_CHECKS
    assert cfg_of().wm_rsp_target == 1.0 and H.LevelConfig.__dataclass_fields__["rsp_target"].default == 1.0
    a0, a1 = p.parse_args([*BASE, "--out", "x"]), p.parse_args([*BASE, "--out", "x", "--wm-rsp-target", "1.0"])
    assert vars(a0) == vars(a1) and TR.make_config(a0) == TR.make_config(a1)


def test_config_refusals_for_the_target(probe):
    for t in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            fk(probe, wm_rsp_target=t)
    with pytest.raises(ValueError):
        cfg_of(wm_rsp_target=1.5)                                                  # idm readout: a lever that is declared and does nothing
    with pytest.raises(ValueError):
        cfg_of(wm_rsp_w=1.0, wm_rsp_target=1.5)
    assert fk(probe, wm_rsp_target=1.5).wm_rsp_target == 1.5 and fk(probe, wm_rsp_target=1.0).wm_rsp_target == 1.0


# ------------------------------------------------------------------------------------------------ C10: flag parity
def _digest(tree, mode, work):
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONHASHSEED="0")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-I", str(HERE / "optb3_digest.py"), str(tree), mode, str(work)], capture_output=True, encoding="utf-8", errors="replace", env=env, stdin=subprocess.DEVNULL, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_c10_absent_equals_one_and_equals_the_86da79cd_trainer_bit_for_bit(tmp_path):
    new_absent, new_one = _digest(REPO, "absent", tmp_path / "a"), _digest(REPO, "explicit", tmp_path / "b")
    assert new_absent == new_one
    assert "rsp_resid/L1" not in new_absent["terms"] and "rsp/L1" in new_absent["terms"]
    if not BASE_TREE.is_dir():                                                     # the third leg needs `git archive 86da79cd stack taniteval` unpacked at OPTB3_BASE; the first two legs above stand without it
        pytest.skip(f"the 86da79cd export is missing at {BASE_TREE} (set OPTB3_BASE)")
    base = _digest(BASE_TREE, "absent", tmp_path / "c")
    assert new_absent == base, (new_absent, base)                                  # state dict, loss terms (key set and bits), outputs, gradients, RNG state


# ------------------------------------------------------------------------------------------------ C11: analytic identity (float64) on the C9' world z' = z + B* a
def matched_world(t, grad=False):
    s = K_ * float(WV @ BS[:, 0])
    st, c = stub(torch.eye(4), BS, WV, s)
    c = dataclasses.replace(c, rsp_target=t)
    B = st.predictors[0].B
    B.requires_grad_(grad)
    return st, c, B


def test_c11_target_one_gives_loss_zero_and_gradient_zero():
    st, c, B = matched_world(1.0, grad=True)
    z, a = data()
    r = st.response_loss(z, a, c)
    r["loss"].backward()
    assert float(r["loss"]) < 1e-24 and float(B.grad.abs().max()) < 1e-12 and abs(float(r["diag"]["rsp_g_aa"]) - 1.0) < 1e-12


@pytest.mark.parametrize("t,want", [(1.5, 0.25), (1.25, 0.0625), (0.75, 0.0625)])
def test_c11_target_other_than_one_gives_the_exact_loss_and_a_gradient_that_moves_G_towards_it(t, want):
    st, c, B = matched_world(t, grad=True)
    z, a = data()
    r = st.response_loss(z, a, c)
    assert abs(float(r["loss"]) - want) < 1e-12, float(r["loss"])
    assert abs(float(r["diag"]["rsp_g_aa"]) - 1.0) < 1e-12 and abs(float(r["diag"]["rsp_resid"]) - (1.0 - t)) < 1e-12
    r["loss"].backward()
    assert float(B.grad.abs().max()) > 1e-6
    with torch.no_grad():
        B -= 1e-3 * B.grad                                                         # one gradient-descent step on B*
    g1 = float(st.response_loss(z, a, c)["diag"]["rsp_g_aa"])
    assert (g1 > 1.0) == (t > 1.0) and abs(g1 - 1.0) > 1e-9                        # G rose for a target above 1, fell for one below


def test_c11_the_gradient_reaches_the_predictor_only_in_the_real_stack(probe):
    wm = M.WMAux(fk(probe, wm_rsp_target=1.5))
    g = torch.Generator().manual_seed(3)
    am = torch.randn(2, 28, wm.cfg.d if hasattr(wm, "cfg") else 32, generator=g, requires_grad=True)
    act = torch.randn(2, 28, 2, generator=g)
    out = wm.forward_mean(am, act)
    assert "rsp_resid/L1" in out and abs(float(out["rsp_g_aa/L1"] - out["rsp_resid/L1"]) - 1.5) < 1e-5
    out["rsp/L1"].backward()
    gs = lambda pre: sum(float(p.grad.abs().sum()) for n, p in wm.named_parameters() if n.startswith(pre) and p.grad is not None)  # noqa: E731
    assert gs("stack.predictors.0.") > 0.0 and gs("stack.act_embed.") > 0.0 and gs("stack.idm_heads.") == 0.0 and gs("proj.") == 0.0


# ------------------------------------------------------------------------------------------------ the trainer's own gate: declared == built, read back
def test_gate_passes_for_the_target_arm_and_reads_g_star_back(init, probe):
    assert lgate(init, "g_t15", FT, tgt(probe, 1.5)) == 0
    sc = sc_of(init, "g_t15")
    rb = sc["levers"]["wm_rsp_readback"]
    assert rb["declared_target"] == 1.5 == rb["built_target"] and abs(rb["built"]["target"] - 1.5) < 1e-9 and rb["built"]["readout"] == "fixed_k"
    assert sc["live"]["pass"] is True and sc["live"]["c3"]["pass"] is True
    assert lgate(init, "g_t125", FT, tgt(probe, 1.25)) == 0
    assert abs(sc_of(init, "g_t125")["levers"]["wm_rsp_readback"]["built"]["target"] - 1.25) < 1e-9
    assert lgate(init, "g_t1", FT, tgt(probe, 1.0)) == 0 and sc_of(init, "g_t1")["levers"]["wm_rsp_readback"]["built"]["target"] == 1.0


def test_gate_refuses_a_target_without_the_fixed_k_term_null_arm_and_idm(init, probe):
    rc = lgate(init, "r_null_t", [*FT, "--wm-rsp-target", "1.5"])                  # a NULL arm names no RSP flag
    assert isinstance(rc, str) and "wm_rsp_target" in rc, rc
    rc = lgate(init, "r_idm_t", [*FT, "--wm-rsp-w", "1.0", "--wm-rsp-target", "1.25"])
    assert isinstance(rc, str) and "wm_rsp_target" in rc, rc
    rc = lgate(init, "r_neg_t", FT, tgt(probe, -1.0))
    assert isinstance(rc, str) and "wm_rsp_target" in rc, rc


def test_gate_refuses_when_make_config_drops_the_target_MR16(init, probe, monkeypatch):
    orig = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_rsp_target=1.0))
    rc = lgate(init, "r_mr16", FT, tgt(probe, 1.5))
    assert isinstance(rc, str) and "wm_rsp_target" in rc, rc


def test_gate_refuses_when_the_built_loss_pulls_to_another_gain_MR17(init, probe, monkeypatch):
    orig = H.HJEPAStack._response_loss_fixed_k

    def hard(self, z, a, c):
        return orig(self, z, a, dataclasses.replace(c, rsp_target=1.0))             # the built loss ignores the config's g*

    monkeypatch.setattr(H.HJEPAStack, "_response_loss_fixed_k", hard)
    rc = lgate(init, "r_mr17", FT, tgt(probe, 1.5))
    assert isinstance(rc, str) and "wm_rsp_target" in rc, rc


def test_gate_refuses_when_the_built_levelconfig_holds_another_gain_MR18(init, probe, monkeypatch):
    orig = M.WMAux.__init__

    def init2(self, cfg, *a, **k):
        orig(self, cfg, *a, **k)
        self.stack.cfg[0] = dataclasses.replace(self.stack.cfg[0], rsp_target=1.0)

    monkeypatch.setattr(M.WMAux, "__init__", init2)
    rc = lgate(init, "r_mr18", FT, tgt(probe, 1.5))
    assert isinstance(rc, str) and "wm_rsp_target" in rc, rc


def test_a_real_two_step_run_of_the_target_arm_keeps_proj_bit_equal_and_trains_the_rest(init, probe):
    assert lgate(init, "t_t15", FT, tgt(probe, 1.5), steps=2) == 0
    a = torch.load(init["ck"], map_location="cpu", weights_only=False)["model"]
    b = torch.load(init["root"] / "t_t15" / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a if k.startswith("wm.proj."))
    assert len([k for k in a if k.startswith("wm.") and not k.startswith("wm.proj.") and a[k].is_floating_point() and not torch.equal(a[k], b[k])]) > 10
