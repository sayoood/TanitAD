"""PREREG_V8F_N1DC (frozen sha256 2eae7964): the skip from T's code into O's plan head, ``--o-code-skip on|off`` (model, trainer, loaders, gate). CPU only, fp32, ONE thread (card section 5, R2: every bit-identity check runs
where same-code runs are bit-identical).

Z0     flag off == the 134f5da1 model (snapshot): state dict keys + values, every loss term, the total, the gradients, forward_infer outputs, built_levers, the RNG stream; no module, no RNG draw.
Z1     flag on vs off, same init, same batch: every non-o_code parameter, every output, every loss term and the total bit-identical at step 0; the global RNG stream after the build is the off model's.
REACH  o_code.weight gets a NON-ZERO gradient (equal to the independent recomputation dL/dh^T . code, from the code ``run_o`` was handed); controls that must read EXACTLY 0.0.
LIVE   after one optimiser step the plan differs from the flag-off plan.
ADMIT  ``assert_no_situation_classifier_input`` stays green with the flag on.
G-DVB  every refusal clause (``bad_o_code``), the FLAG_CHECKS entry, the parser default, the WM-only default list, the init-ckpt fresh key, the resume check.
LOAD   every loader that rebuilds the model from a checkpoint cfg builds o_code when the cfg says so (strict load, equal outputs); the strat1_check G-EVAL regression arm refuses a cfg that drops it.
Each mutation (K-MUT a-f) is a source edit that must turn a NAMED test RED (``run_mutants.py`` drives them and writes ``mutation_table_n1dc.json``)."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "stack" / "scripts", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import tanitad  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_n1dw import (CH, MODEL_P, N1, RBD, TINY, _argv, _read, dvb, equal_runs, gate_parts, git_blob_sha, load_J, make_batch, mod_from_text, mutant, run_loss, sha,  # noqa: E402
                           write_nav_npz, write_rbd_npz)

assert Path(tanitad.__file__).resolve().is_relative_to(REPO.resolve()), tanitad.__file__          # the study tree's tanitad, never an editable install elsewhere

SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_n1dc.txt")
SNAP_GIT_BLOB = "a24a22b3b62ade7d31f11ee1ffd7af349d33df05"          # git blob of stack/tanitad/models/v8f_mini.py at 134f5da1 (the N1DW code of record, the parent of this build)
SCRIPTS = REPO / "stack" / "scripts"
TRAIN_P = SCRIPTS / "v8f_mini_train.py"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("n1dcworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "rbd": write_rbd_npz(root / "rbd.npz", J), "nav": write_nav_npz(root / "nav.npz", J), **w}


def base_module():
    return sys.modules.get("v8f_mini_snapshot_n1dc") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_snapshot_n1dc", SNAP)


def build(J, mod=M, seed=0, skip=False, **kw):
    """A tiny model (the nav_rb exitdist channel built, as N1D's is), global RNG seeded, then head_o's last layer and e_nav_rb given seeded non-zero values from private generators (so the global stream is untouched and
    the skip's step-0 identity is tested against a LIVE head, not a zero one). ``mod`` = the snapshot module has no o_code_skip field."""
    extra = {"o_code_skip": True} if skip else {}
    cfg = mod.V8FMiniConfig(**{**TINY, **CH, **extra, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
    with torch.no_grad():
        w = m.head_o[2].weight
        w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
        m.e_nav_rb.weight.copy_(torch.randn(m.e_nav_rb.weight.shape, generator=torch.Generator().manual_seed(12)) * 0.05)
    return m


def set_o_code(m, scale=0.05, seed=13):
    with torch.no_grad():
        m.o_code.weight.copy_(torch.randn(m.o_code.weight.shape, generator=torch.Generator().manual_seed(seed)) * scale)


# ------------------------------------------------------------------------------------------------ pins, flags, defaults
def test_the_snapshot_is_the_134f5da1_model_and_tanitad_is_the_study_tree():
    assert git_blob_sha(SNAP) == SNAP_GIT_BLOB
    assert Path(M.__file__).resolve() == MODEL_P.resolve()
    assert "o_code" not in SNAP.read_text(encoding="utf-8")


def test_flag_default_off_has_a_flag_check_a_choice_list_and_a_wm_only_default():
    assert TR.FLAG_CHECKS["o_code_skip"] == "lever:o_code_skip"
    assert TR.build_parser().get_default("o_code_skip") == "off"
    act = [a for a in TR.build_parser()._actions if a.dest == "o_code_skip"][0]
    assert act.option_strings == ["--o-code-skip"] and tuple(act.choices) == ("on", "off")
    assert "o_code_skip" in TR.WM_ONLY_DEFAULT_DESTS
    assert M.V8FMiniConfig(**TINY).o_code_skip is False
    assert TR.make_config(TR.build_parser().parse_args(["--out", "x"])).o_code_skip is False
    assert TR.make_config(TR.build_parser().parse_args(["--out", "x", "--o-code-skip", "on"])).o_code_skip is True
    assert M.O_CODE_KEY_PREFIXES == ("o_code.",) and "o_code." in M.NEW_LEVER_KEY_PREFIXES, "the init-ckpt gate must admit exactly this fresh key"


@pytest.mark.parametrize("bad", ["on", "off", 1, 0, None, 0.0])
def test_config_validator_refuses_a_non_bool_flag(bad):
    with pytest.raises(ValueError, match="o_code_skip"):
        M.V8FMiniConfig(**{**TINY, "o_code_skip": bad})
    for ok in (True, False):
        assert M.V8FMiniConfig(**{**TINY, "o_code_skip": ok}).o_code_skip is ok
    with pytest.raises(TypeError):
        M.V8FMiniConfig(o_code_skipz=True)


def test_the_argparse_refuses_a_value_outside_on_off():
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--o-code-skip", "maybe"])


# ------------------------------------------------------------------------------------------------ Z0
def test_z0_flag_off_build_is_the_134f5da1_model_bit_for_bit(world):
    b, J, _ = make_batch(world, with_rb=False)
    mt, mn = build(J, mod=base_module()), build(J, mod=M)
    assert mn.o_code is None and "o_code_skip" not in mn.built_levers()
    assert list(mt.state_dict()) == list(mn.state_dict())
    assert all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items()), "state dict values"
    assert mt.built_levers() == mn.built_levers() and "OCODE" not in mn.trainable_groups()
    ot, on = run_loss(mt, b), run_loss(mn, b)
    torch.manual_seed(3); _ = mn.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_n = torch.get_rng_state().clone()
    torch.manual_seed(3); _ = mt.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_t = torch.get_rng_state().clone()
    assert torch.equal(rng_t, rng_n), "the global RNG stream after one loss call"
    probs, extra = equal_runs(ot, on)
    assert probs == [] and extra == [], (probs, extra)
    assert torch.equal(ot["terms"]["total"], on["terms"]["total"]) and set(ot["terms"]) == set(on["terms"])
    ot["terms"]["total"].backward(); on["terms"]["total"].backward()
    pn = dict(mn.named_parameters())
    assert all(torch.equal(p.grad, pn[n].grad) for n, p in mt.named_parameters() if p.grad is not None), "gradients"
    mt.eval(); mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    assert all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k])), "forward_infer"


def test_z0_the_off_build_draws_nothing_from_the_global_rng_beyond_the_snapshots(world):
    J = load_J(world)
    s0 = torch.get_rng_state().clone()
    torch.manual_seed(0); build(J, mod=base_module()); a = torch.get_rng_state().clone()
    torch.manual_seed(0); build(J, mod=M); c = torch.get_rng_state().clone()
    assert torch.equal(a, c) and not torch.equal(a, s0)


# ------------------------------------------------------------------------------------------------ Z1
def test_z1_flag_on_at_step_0_equals_flag_off_bit_for_bit_and_the_rng_stream_is_the_off_models(world):
    b, J, _ = make_batch(world, with_rb=False)
    torch.manual_seed(0); m_off = build(J, skip=False); rng_off = torch.get_rng_state().clone()
    torch.manual_seed(0); m_on = build(J, skip=True); rng_on = torch.get_rng_state().clone()
    assert m_on.o_code is not None and float(m_on.o_code.weight.abs().sum()) == 0.0 and tuple(m_on.o_code.weight.shape) == (m_on.cfg.n_q * m_on.cfg.d, M.CODE_DIM)
    assert torch.equal(rng_off, rng_on), "building o_code must not move the global RNG stream (built LAST under a forked RNG)"
    so, sn = m_off.state_dict(), m_on.state_dict()
    assert list(sn) == list(so) + ["o_code.weight"], "o_code is the LAST key"
    assert all(torch.equal(v, sn[k]) for k, v in so.items()), "every non-o_code parameter and buffer, bit for bit"
    assert m_on.built_levers()["o_code_skip"] == "on" and "OCODE" in m_on.trainable_groups() and "o_code_skip" not in m_off.built_levers()
    o_off, o_on = run_loss(m_off, b), run_loss(m_on, b)
    probs, extra = equal_runs(o_off, o_on)
    assert probs == [] and extra == [], (probs, extra)
    assert torch.equal(o_off["terms"]["total"], o_on["terms"]["total"]) and set(o_off["terms"]) == set(o_on["terms"])
    m_off.eval(); m_on.eval()
    with torch.no_grad():
        i0, i1 = m_off.forward_infer(b), m_on.forward_infer(b)
    assert all(torch.equal(i0[k], i1[k]) for k in i0 if torch.is_tensor(i0[k])), "forward_infer"


def test_z1_every_non_o_code_gradient_at_step_0_is_bit_equal_too(world):
    b, J, _ = make_batch(world, with_rb=False)
    m_off, m_on = build(J, skip=False), build(J, skip=True)
    run_loss(m_off, b)["terms"]["total"].backward(); run_loss(m_on, b)["terms"]["total"].backward()
    pn = dict(m_on.named_parameters())
    assert all((p.grad is None) == (pn[n].grad is None) and (p.grad is None or torch.equal(p.grad, pn[n].grad)) for n, p in m_off.named_parameters())


# ------------------------------------------------------------------------------------------------ REACH
def test_reach_o_code_gets_a_nonzero_gradient_equal_to_the_independent_recompute_and_the_controls_read_exactly_zero(world):
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=True)
    calls, handed = [], []                                                                                              # every call of o_code (its input and its output, grad retained) and every code run_o was handed
    m.o_code.register_forward_hook(lambda mod, inp, outp: (outp.retain_grad(), calls.append((inp[0], outp)))[0])
    run_o0 = m.run_o
    m.run_o = lambda *a, **k: (handed.append(a[5] if len(a) > 5 else k["code"]), run_o0(*a, **k))[1]
    out = run_loss(m, b)
    T = out["terms"]
    T["total"].backward(retain_graph=True)
    # the independent recompute: dL/dW = sum over the calls of (dL/d o_code(code))^T . code, and ``code`` is exactly the tensor ``run_o`` was handed
    assert len(calls) == len(handed) >= 1 and all(torch.equal(i, c) for (i, _), c in zip(calls, handed)), "o_code reads the tensor run_o was handed"
    want = sum(o.grad.T @ i.detach() for i, o in calls)
    assert tuple(want.shape) == tuple(m.o_code.weight.shape)
    torch.testing.assert_close(m.o_code.weight.grad, want, rtol=1e-5, atol=1e-9)
    assert float(want.abs().sum()) > 0.0
    gw = torch.autograd.grad(T["total"], [m.o_code.weight], retain_graph=True)[0]
    assert float(gw.abs().sum()) > 0.0, "REACH: a non-zero gradient on o_code.weight"
    gade = torch.autograd.grad(T["ade"], [m.o_code.weight], retain_graph=True)[0]
    assert float(gade.abs().sum()) > 0.0, "and from O's own ade term"
    for k in ("ckpt", "goal", "lat", "lon", "cons"):                                                                      # CONTROLS: S's and T's terms do not depend on O: no gradient at all
        g = torch.autograd.grad(T[k], [m.o_code.weight], retain_graph=True, allow_unused=True)[0]
        assert g is None or float(g.abs().sum()) == 0.0, k
    # CONTROL: at the REAL init (head_o's last layer zero) every gradient upstream of it, o_code.weight's included, is EXACTLY 0.0
    mz = build(J, skip=True)
    with torch.no_grad():
        mz.head_o[2].weight.zero_(); mz.head_o[2].bias.zero_()
    gz = torch.autograd.grad(run_loss(mz, b)["terms"]["total"], [mz.o_code.weight], allow_unused=True)[0]
    assert gz is None or float(gz.abs().sum()) == 0.0
    assert all(p.grad is None for p in m.prior.parameters()), "the frozen P10 prior: no gradient at all"


def test_reach_the_skip_reads_the_code_run_o_was_handed(world):
    """Call ``run_o`` directly with two different codes: the plan moves with the code through o_code (with e_code's path held equal by zeroing e_code's weight), and o_code(code) is exactly what is added."""
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=True)
    set_o_code(m)
    m.eval()
    with torch.no_grad():
        adapt = m._encode(b)
        Fn = b["feats"].shape[1]
        B = b["feats"].shape[0]
        last = torch.full((B,), Fn - 1, dtype=torch.long)
        ck = m.ck_nom[None].expand(B, -1, -1)
        ok = torch.ones(B, 3)
        c1 = torch.rand(B, M.CODE_DIM, generator=torch.Generator().manual_seed(1))
        c2 = torch.rand(B, M.CODE_DIM, generator=torch.Generator().manual_seed(2))
        m.e_code.weight.zero_(); m.e_code.bias.zero_()                                    # e_code reads nothing: only the skip can carry the code
        t1, _ = m.run_o(b, adapt, last, ck, ok, c1)
        t2, _ = m.run_o(b, adapt, last, ck, ok, c2)
        assert float((t1 - t2).abs().max()) > 0.0
        m.o_code.weight.zero_()
        u1, _ = m.run_o(b, adapt, last, ck, ok, c1)
        u2, _ = m.run_o(b, adapt, last, ck, ok, c2)
        assert torch.equal(u1, u2), "with o_code zero and e_code zeroed the code reaches nothing"


def test_detach_upper_the_skip_puts_exactly_zero_gradient_on_s_and_t(world):
    """With --detach-upper (the N1D recipe) the skip reads the code AFTER the detach: O's ade term puts EXACTLY 0.0 on S's and T's parameters while o_code.weight is non-zero and head_o is live."""
    b, J, _ = make_batch(world, with_rb=False)
    for det, expect_zero in ((True, True), (False, False)):
        m = build(J, skip=True, detach_upper=det)
        set_o_code(m)
        T = run_loss(m, b)["terms"]
        up = [p for n, p in m.named_parameters() if n.startswith(("read_s.", "head_s.", "read_t.", "trunk_t.", "h_goal.", "h_lat.", "h_lon.", "h_cons."))]
        g = torch.autograd.grad(T["ade"], up, allow_unused=True)
        s = float(sum(float(x.abs().sum()) for x in g if x is not None))
        assert (s == 0.0) == expect_zero, (det, s)


# ------------------------------------------------------------------------------------------------ LIVE
def test_after_one_optimiser_step_the_plan_differs_from_the_flag_off_plan(world):
    b, J, _ = make_batch(world, with_rb=False)
    m_off, m_on = build(J, skip=False), build(J, skip=True)
    plans = []
    for m in (m_off, m_on):
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        run_loss(m, b)["terms"]["total"].backward()
        opt.step()
        plans.append(run_loss(m, b)["traj30"].detach())
    assert float(m_on.o_code.weight.abs().sum()) > 0.0, "o_code moved off its zero init"
    assert not torch.equal(plans[0], plans[1]) and float((plans[0] - plans[1]).abs().max()) > 0.0, "head_o's plan differs once the path has a weight"


# ------------------------------------------------------------------------------------------------ admissibility
def test_admissibility_assert_no_situation_classifier_input_stays_green_with_the_flag_on(world):
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=True)
    set_o_code(m)
    r = M.assert_no_situation_classifier_input(m, b)
    assert set(r["read"]) <= set(M.INFERENCE_INPUT_KEYS) | M.NAV_RB_KEYS
    assert M.scan_source_for_forbidden(Path(MODEL_P).read_text(encoding="utf-8")) == []
    # the skip reads nothing but T's code: forward_infer under a changed label block is unchanged
    m.eval()
    with torch.no_grad():
        x0 = m.forward_infer(b)["traj60"]
        x1 = m.forward_infer({k: v for k, v in b.items() if not k.startswith(("lab", "gt"))})["traj60"]
    assert torch.equal(x0, x1)


# ------------------------------------------------------------------------------------------------ G-DVB
def ocode_only(msgs):
    return [m for m in msgs if "o_code" in m]


def test_gdvb_passes_an_honoured_argv_and_names_the_flag_in_built_levers(world):
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_ok")
    assert ocode_only(dvb(parts)) == [], dvb(parts)
    assert parts[2].built_levers()["o_code_skip"] == "on" and parts[2].cfg.o_code_skip is True and parts[2].o_code is not None
    parts0 = gate_parts(world, [*RBD(world), "--o-code-skip", "off"], "oc_off")
    assert ocode_only(dvb(parts0)) == [] and "o_code_skip" not in parts0[2].built_levers() and parts0[2].o_code is None
    parts_d = gate_parts(world, [*RBD(world)], "oc_default")
    assert ocode_only(dvb(parts_d)) == []


def test_gdvb_refuses_declared_on_but_built_without_o_code(world):
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_dropped")
    parts[2].o_code = None                                                                  # the build dropped the module (the refcv6 class: declared, never built)
    assert any("lever o_code_skip: declared on but built off" in m for m in dvb(parts)), dvb(parts)
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_dropped_cfg")
    parts[2].o_code = None
    parts[2].cfg.o_code_skip = False
    assert any("lever o_code_skip: declared on but built off" in m for m in dvb(parts)), dvb(parts)


def test_gdvb_refuses_declared_off_but_built_with_o_code(world):
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "off"], "oc_extra")
    parts[2].cfg.o_code_skip = True
    parts[2].o_code = torch.nn.Linear(M.CODE_DIM, parts[2].cfg.n_q * parts[2].cfg.d, bias=False)
    msgs = dvb(parts)
    assert any("lever o_code_skip: declared off but built on" in m for m in msgs), msgs


def test_gdvb_refuses_a_non_zero_start_a_wrong_shape_a_bias_and_a_head_width_mismatch(world):
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_bad")
    model = parts[2]
    set_o_code(model)
    assert any("not zero at the start" in m for m in dvb(parts)), "a non-zero weight at the start"
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_shape")
    parts[2].o_code = torch.nn.Linear(M.CODE_DIM - 1, parts[2].cfg.n_q * parts[2].cfg.d, bias=False)
    torch.nn.init.zeros_(parts[2].o_code.weight)
    assert any("the built o_code is" in m for m in dvb(parts)), "a wrong input width"
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_bias")
    parts[2].o_code = torch.nn.Linear(M.CODE_DIM, parts[2].cfg.n_q * parts[2].cfg.d, bias=True)
    torch.nn.init.zeros_(parts[2].o_code.weight)
    assert any("without a bias" in m for m in dvb(parts)), "a bias"
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_width")
    parts[2].o_code = torch.nn.Linear(M.CODE_DIM, parts[2].cfg.n_q * parts[2].cfg.d + 1, bias=False)
    torch.nn.init.zeros_(parts[2].o_code.weight)
    msgs = dvb(parts)
    assert any("the built o_code is" in m for m in msgs) or any("head_o reads" in m for m in msgs), msgs


def test_gdvb_refuses_the_skip_with_o_interface_cut(world):
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on", "--o-interface", "cut"], "oc_cut")
    assert any("o_code_skip on with --o-interface cut" in m and "does nothing" in m for m in dvb(parts)), dvb(parts)


def test_gdvb_refuses_an_optimiser_that_lacks_or_doubles_o_code(world):
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_opt")
    a, cfg, model, opt, info = parts
    opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad and p is not model.o_code.weight], lr=a.lr)
    assert any("o_code.weight is in 0 optimiser groups" in m for m in ocode_only(dvb((a, cfg, model, opt2, info)))), "o_code missing from the optimiser"
    import types
    opt3 = types.SimpleNamespace(param_groups=[{"params": list(opt.param_groups[0]["params"]), "lr": a.lr, "weight_decay": a.wd}, {"params": [model.o_code.weight], "lr": a.lr, "weight_decay": a.wd}])   # torch itself refuses a doubled parameter: a stand-in holds it twice
    assert any("o_code.weight is in 2 optimiser groups" in m for m in ocode_only(dvb((a, cfg, model, opt3, info)))), "o_code twice"


def test_a_flag_without_a_gdvb_entry_is_refused_by_the_gate_itself(world, monkeypatch):
    parts = gate_parts(world, [*RBD(world), "--o-code-skip", "on"], "oc_noentry")
    assert [m for m in dvb(parts) if "declared-vs-built entry" in m] == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "o_code_skip"})
    bad = dvb(parts)
    assert any("flags without a declared-vs-built entry" in m and "o_code_skip" in m for m in bad), bad


def test_main_refuses_each_bad_argv_before_anything_trains(world):
    for tag, extra, needle in (("oc_cut", ["--o-code-skip", "on", "--o-interface", "cut"], "o_code_skip on with --o-interface cut"),):
        with pytest.raises(SystemExit) as e:
            TR.main(_argv(world, tag, ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world), *extra]))
        assert "ZZDVB-REFUSEDZZ" in str(e.value) or needle in str(e.value), (tag, str(e.value)[:300])
    with pytest.raises(SystemExit):                                                          # argparse: a value outside on|off
        TR.main(_argv(world, "oc_maybe", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world), "--o-code-skip", "maybe"]))


# ------------------------------------------------------------------------------------------------ the probe (G-LIVE)
def test_probe_off_runs_nothing_and_on_reads_gradients_identity_and_exact_zero_controls(world):
    b, J, _ = make_batch(world, with_rb=False)
    m0 = build(J, skip=False)
    r0 = TR.o_code_probe(m0, b, m0.cfg)
    assert r0["o_code"] == "off" and r0["pass"] is True
    for det in (False, True):
        m = build(J, skip=True, detach_upper=det)
        with torch.no_grad():
            m.head_o[2].weight.zero_(); m.head_o[2].bias.zero_()                              # the REAL init
        r = TR.o_code_probe(m, b, m.cfg)
        assert r["pass"] and r["grad_total"] > 0.0 and r["grad_ade"] > 0.0 and r["plan_move_when_weight_set"] > 0.0
        assert r["control_tside_grad"] == 0.0 and r["control_repeat_grad_diff"] == 0.0 and r["control_prior_grad"] == 0.0 and r["control_real_init_grad"] == 0.0
        assert r["identity_terms_differing"] == [] and r["identity_outputs_differing"] == [] and r["head_o_last_layer_zero_at_init"] is True
        assert r["control_ade_grad_on_upper"] == (0.0 if det else None)
    assert float(m.o_code.weight.abs().sum()) == 0.0, "the probe works on a COPY"


def test_probe_on_a_fine_tune_start_with_a_live_head_needs_no_reseeding(world):
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=True, detach_upper=True)                                              # head_o's last layer already non-zero (the M2b checkpoint's)
    r = TR.o_code_probe(m, b, m.cfg)
    assert r["pass"] and r["head_o_last_layer_zero_at_init"] is False and r["control_real_init_grad"] > 0.0


def test_probe_refuses_a_model_whose_skip_line_is_deleted(world):
    b, J, _ = make_batch(world, with_rb=False)
    mod = mutant(MODEL_P, "            h = h + self.o_code(code)\n", "            pass\n", "_mut_n1dc_probe_noline")
    m = build(J, mod=mod, skip=True)
    with pytest.raises(AssertionError, match="does not reach the loss"):
        TR.o_code_probe(m, b, m.cfg)


def test_probe_refuses_a_skip_that_reads_the_code_from_before_the_detach(world):
    b, J, _ = make_batch(world, with_rb=False)
    text = MODEL_P.read_text(encoding="utf-8")
    old1 = '        if cfg.detach_upper:\n            code = code.detach()\n        out["code"] = code'
    assert text.count(old1) == 1
    old2 = "            h = h + self.o_code(code)\n"
    assert text.count(old2) == 1
    mod = mod_from_text(text.replace(old1, '        b["_code_pre"] = code\n' + old1).replace(old2, '            h = h + self.o_code(b.get("_code_pre", code))\n'), "_mut_n1dc_probe_prebefore", MODEL_P)
    m = build(J, mod=mod, skip=True, detach_upper=True)
    with pytest.raises(AssertionError, match="reads the code from before the detach"):
        TR.o_code_probe(m, b, m.cfg)


# ------------------------------------------------------------------------------------------------ the trainer, end to end
N1DC_ARGS = ["--detach-upper"]


def test_end_to_end_n1dc_arm_probe_levers_group_and_checkpoint(world):
    out = Path(world["root"]) / "n1dc_on"
    assert TR.main(_argv(world, "n1dc_on", ["--steps", "4", *N1(world), *RBD(world), *N1DC_ARGS, "--o-code-skip", "on"])) == 0
    sc, sm = _read(out, "selfcheck.json"), _read(out, "summary.json")
    pr = sc["o_code_probe"]
    assert sc["declared_vs_built"] == "PASS" and pr["pass"] and pr["grad_total"] > 0.0 and pr["control_tside_grad"] == 0.0 and pr["control_ade_grad_on_upper"] == 0.0
    assert sc["levers"]["o_code_skip"] == "on" and sc["live"]["groups_grad"]["OCODE"] > 0.0 and sc["live"]["pass"] and "o_code.weight" not in sc["live"]["dead_params"]
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert ck["cfg"]["o_code_skip"] is True and "o_code.weight" in ck["model"] and float(ck["model"]["o_code.weight"].abs().sum()) > 0.0, "the weight trained"
    mm = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(D.load_join(Path(world["root"]) / "join_train.npz")))
    mm.load_state_dict(ck["model"])                                                             # the evaluators' build path, strict
    assert mm.o_code is not None and mm.built_levers()["o_code_skip"] == "on"
    assert sm["steps"] == 4 if "steps" in sm else True


def test_end_to_end_flag_off_and_explicit_off_keep_the_record_set(world):
    for tag, extra in (("ocoff", []), ("ocoff2", ["--o-code-skip", "off"])):
        out = Path(world["root"]) / tag
        assert TR.main(_argv(world, tag, ["--steps", "2", *N1(world), *RBD(world), *extra])) == 0
        sc = _read(out, "selfcheck.json")
        ck = torch.load(out / "ckpt_last.pt", weights_only=False)
        assert "o_code_probe" not in sc and "o_code_skip" not in sc["levers"] and "OCODE" not in sc["live"]["groups_grad"]
        assert not any(k.startswith("o_code.") for k in ck["model"]) and ck["cfg"]["o_code_skip"] is False


def test_fine_tune_from_a_checkpoint_without_o_code_passes_the_init_ckpt_gate_and_only_with_the_flag(world):
    """N1D's argv fine-tunes M2b_s0, which has no o_code: the init-ckpt G-DVB must admit exactly that fresh key, and only when --o-code-skip on (the N1R mint lesson of 2026-11-18)."""
    base = Path(world["root"]) / "oc_base_init"
    assert TR.main(_argv(world, "oc_base_init", ["--steps", "2", *N1(world)])) == 0
    ck0 = torch.load(base / "ckpt_last.pt", weights_only=False)
    assert not any(k.startswith("o_code") for k in ck0["model"])
    out = Path(world["root"]) / "oc_from_init"
    assert TR.main(_argv(world, "oc_from_init", ["--steps", "1", "--selfcheck-only", "--init-ckpt", str(base / "ckpt_last.pt"), *N1(world), *RBD(world), *N1DC_ARGS, "--o-code-skip", "on"])) == 0
    sc = _read(out, "selfcheck.json")
    ic = sc["info"]["init_ckpt"]
    assert sc["declared_vs_built"] == "PASS" and ic["applied"] and "o_code.weight" in ic["declared_new_keys"] and set(ic.get("fresh_keys", [])) <= set(ic["declared_new_keys"])
    assert sc["o_code_probe"]["pass"] and sc["o_code_probe"]["head_o_last_layer_zero_at_init"] is False, "after the load head_o is the checkpoint's trained head"
    out2 = Path(world["root"]) / "oc_from_init_off"
    assert TR.main(_argv(world, "oc_from_init_off", ["--steps", "1", "--selfcheck-only", "--init-ckpt", str(base / "ckpt_last.pt"), *N1(world), *RBD(world), *N1DC_ARGS])) == 0
    assert _read(out2, "selfcheck.json")["declared_vs_built"] == "PASS"


def test_resume_refuses_a_checkpoint_whose_cfg_skip_differs_from_the_argv(world):
    argv = _argv(world, "oc_resume", ["--steps", "3", *N1(world), *RBD(world), *N1DC_ARGS, "--o-code-skip", "on"])
    assert TR.main(argv) == 0
    p = Path(world["root"]) / "oc_resume" / "ckpt_last.pt"
    ck = torch.load(p, weights_only=False)
    assert TR.main(argv) == 0                                                    # the honest checkpoint resumes (already done: returns)
    ck["cfg"]["o_code_skip"] = False
    torch.save(ck, p)
    with pytest.raises(SystemExit, match="ZZRESUME-REFUSEDZZ the checkpoint's cfg.o_code_skip is False but the argv declares True"):
        TR.main(argv)


# ------------------------------------------------------------------------------------------------ the loaders: round trip
LOADERS = ("v8f_mini_eval.py", "v8f_mini_eval_oacc.py", "v8f_mini_eval_rb.py", "v8f_mini_eval_rbd.py", "v8f_mini_eval_speedavg.py", "v8f_mini_eval_speedcal.py", "v8f_t1_eval.py", "v8f_t1_eval_rb.py",
           "v8f_t1_eval_rbd.py", "v8f_dagger_bank.py", "v8f_kscore.py", "n1dh_klim.py")  # n1dh_klim.py (N1DH, f2e91e33): the K-LIM probe builds the v8f config from a checkpoint too
CFG_CALL = re.compile(r"V8FMiniConfig\(\*\*((?:[^()]|\([^()]*\))*)\)")


def loader_cfg_exprs(name):
    src = (SCRIPTS / name).read_text(encoding="utf-8")
    return CFG_CALL.findall(src)


def _ckpt_of(model):
    return {"model": {k: v.clone() for k, v in model.state_dict().items()}, "cfg": dict(model.cfg.__dict__)}


@pytest.mark.parametrize("name", LOADERS)
def test_roundtrip_each_loader_builds_o_code_from_the_checkpoint_cfg_strict_and_equal(name, world):
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=True, detach_upper=True)
    set_o_code(m)
    ck = _ckpt_of(m)
    exprs = loader_cfg_exprs(name)
    assert exprs, f"{name} has no V8FMiniConfig(**...) build line (find it and add it to this test)"
    for ex in exprs:
        cfgd = eval(ex, {"ck": ck, "cfgd": ck["cfg"], "cfg": ck["cfg"]})                      # the loader's OWN expression, evaluated over the checkpoint
        mm = M.V8FMini(M.V8FMiniConfig(**cfgd), M.P10Prior.synthetic(0), *M.cons_stats(J))
        mm.load_state_dict(ck["model"])                                                       # STRICT: an o_code-less rebuild raises (unexpected key o_code.weight)
        assert mm.o_code is not None and mm.built_levers()["o_code_skip"] == "on"
        assert list(mm.state_dict()) == list(m.state_dict()) and all(torch.equal(v, m.state_dict()[k]) for k, v in mm.state_dict().items())
        m.eval(); mm.eval()
        with torch.no_grad():
            o1, o2 = m.forward_infer(b), mm.forward_infer(b)
        assert all(torch.equal(o1[k], o2[k]) for k in o1 if torch.is_tensor(o1[k])), "outputs"


def test_roundtrip_a_flag_off_checkpoint_loads_through_every_loader_without_o_code(world):
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=False)
    ck = _ckpt_of(m)
    for name in LOADERS:
        for ex in loader_cfg_exprs(name):
            mm = M.V8FMini(M.V8FMiniConfig(**eval(ex, {"ck": ck, "cfgd": ck["cfg"], "cfg": ck["cfg"]})), M.P10Prior.synthetic(0), *M.cons_stats(J))
            mm.load_state_dict(ck["model"])
            assert mm.o_code is None


def test_roundtrip_an_old_checkpoint_cfg_without_the_key_loads_as_off(world):
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=False)
    ck = _ckpt_of(m)
    del ck["cfg"]["o_code_skip"]                                                               # a checkpoint written before the lever existed
    mm = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(J))
    mm.load_state_dict(ck["model"])
    assert mm.o_code is None and mm.cfg.o_code_skip is False


def test_roundtrip_strat1_check_geval_accepts_a_skip_checkpoint_and_refuses_a_cfg_that_drops_it(world, tmp_path):
    import strat1_check as SC
    b, J, _ = make_batch(world, with_rb=False)
    m = build(J, skip=True)
    set_o_code(m)
    ck = _ckpt_of(m)
    p = tmp_path / "ck.pt"
    torch.save(ck, p)
    bad, rec = SC.geval(str(p), selfcheck_levers=m.built_levers())
    assert bad == [] and rec["regression_arm_o_code_skip_refused"] is True and rec["built_levers"]["o_code_skip"] == "on", (bad, rec)
    # the evaluator's lever read: a selfcheck that says "off" for a skip checkpoint is a mismatch
    bad2, _ = SC.geval(str(p), selfcheck_levers={k: v for k, v in m.built_levers().items() if k != "o_code_skip"})
    assert any("o_code_skip" in x for x in bad2), bad2


def test_the_refcv_loaders_do_not_build_the_v8f_model():
    """The refcv6 / refcv7 loaders (``_build_model``, ``refcv7_loader.build_model``) build the RefC family; they never construct V8FMini, so the lever has no build path there (a test that fails when one starts to)."""
    for rel in ("tanitad/eval/refcv6_loader.py", "tanitad/eval/refcv7_loader.py"):
        txt = (REPO / "stack" / rel).read_text(encoding="utf-8")
        assert "V8FMini" not in txt and "v8f_mini" not in txt, rel


def test_every_stack_script_that_builds_the_v8f_config_from_a_checkpoint_is_in_the_loader_list():
    """A new evaluator that rebuilds V8FMini must be added to LOADERS (and so be round-tripped): the scan lists every stack script with a V8FMiniConfig(**ck[...]) call."""
    found = set()
    for f in sorted(SCRIPTS.glob("*.py")):
        if re.search(r'V8FMiniConfig\(\*\*ck\["cfg"\]\)', f.read_text(encoding="utf-8")):
            found.add(f.name)
    assert found == set(LOADERS), sorted(found ^ set(LOADERS))
