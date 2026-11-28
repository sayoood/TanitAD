"""PREREG_V8F_N1DH (frozen sha256 f1df05bc): the auxiliary route-decision head on O's plan feature whose detached posterior conditions the plan head, ``--aux-route off|cond`` + ``--w-aux-route`` (model, label stage,
trainer, loaders, gate). CPU only, fp32, ONE thread (card section 5, R2: every bit-identity check runs where same-code runs are bit-identical).

Z0     flag off == the bcf3ba3f model (snapshot): state dict keys + values, every loss term, the total, the gradients, forward_infer outputs, built_levers, the RNG stream; no module, no RNG draw.
Z1     cond at step 0 vs off, same init, same batch: every non-route parameter, every output, every loss term but ``route`` bit-identical; the total moves by exactly w x route; the RNG streams (global and the batch generator) are the off model's.
REACH  route_head.weight gets a NON-ZERO CE gradient, route_emb.weight a non-zero gradient (head_o's last layer given a seeded weight); controls that must read EXACTLY 0.0: the decoder term on route_head (the posterior is detached), the CE
       on a batch with no own-code row, the T/S-side terms, the frozen prior, route_emb at the real zero init.
OWN    the CE's rows are exactly the rows where O read T's own code (route_own is the use_t draw's complement; the code O read is T's own on those rows and the GT code elsewhere).
LABEL  K-LABEL: route_stage == the label source called directly (analytic arcs; the real D1-HO sample, 5,736 windows, majority share 0.338215), and a within-batch permutation turns it RED.
LIVE   after one optimiser step the plan differs from the flag-off plan.
ADMIT  ``assert_no_situation_classifier_input`` stays green with cond; forward_infer reads inference keys only.
G-DVB  every refusal clause (``bad_aux_route``), the FLAG_CHECKS entries, the parser defaults, ``ce`` refused naming the card, the WM-only defaults, the init-ckpt fresh keys, the resume check.
LOAD   every loader that rebuilds the model from a checkpoint cfg builds the head when the cfg says so (strict load, equal outputs).
Each mutation (K-MUT) is a source edit that must turn a NAMED test RED (``run_mutants.py`` drives them and writes ``mutation_table_n1dh.json``)."""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "stack" / "scripts", REPO / "taniteval", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import tanitad  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_n1dw import (CH, MODEL_P, N1, RBD, TINY, _argv, _read, dvb, equal_runs, gate_parts, git_blob_sha, load_J, make_batch, mod_from_text, mutant, run_loss, sha,  # noqa: E402
                           write_nav_npz, write_rbd_npz)

assert Path(tanitad.__file__).resolve().is_relative_to(REPO.resolve()), tanitad.__file__          # the study tree's tanitad, never an editable install elsewhere

SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_n1dh.txt")
SNAP_GIT_BLOB = "0d21ac9f9d3c92c2a1c4dbc8f58df7017b32d784"          # git blob of stack/tanitad/models/v8f_mini.py at bcf3ba3f (N1DC's tip, the parent of this build)
SCRIPTS = REPO / "stack" / "scripts"
W = 0.5                                                              # the card's CE weight
COND = {"aux_route": "cond", "w_aux_route": W}
D1HO_DUMP = Path(os.environ.get("N1DH_D1HO_DUMP", "F:/TanitAD-scratch/n1dc/read/in/n1dw/hb/out/hb0.2/n1d/t1/dump_NAVRBD-HO-N1D_s0_m10_causal_heads"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("n1dhworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "rbd": write_rbd_npz(root / "rbd.npz", J), "nav": write_nav_npz(root / "nav.npz", J), **w}


def base_module():
    return sys.modules.get("v8f_mini_snapshot_n1dh") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_snapshot_n1dh", SNAP)


def build(J, mod=M, seed=0, route=False, **kw):
    """A tiny model (the nav_rb exitdist channel built, as N1D's is), global RNG seeded, then head_o's last layer and e_nav_rb given seeded non-zero values from private generators (so the global stream is untouched and the
    step-0 identity is tested against a LIVE head, not a zero one). ``mod`` = the snapshot module has no aux_route field."""
    extra = dict(COND) if route else {}
    cfg = mod.V8FMiniConfig(**{**TINY, **CH, **extra, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
    with torch.no_grad():
        w = m.head_o[2].weight
        w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
        m.e_nav_rb.weight.copy_(torch.randn(m.e_nav_rb.weight.shape, generator=torch.Generator().manual_seed(12)) * 0.05)
    return m


def set_route_emb(m, scale=0.05, seed=13):
    with torch.no_grad():
        m.route_emb.weight.copy_(torch.randn(m.route_emb.weight.shape, generator=torch.Generator().manual_seed(seed)) * scale)


def routed(world, n=16, seed=0):
    """A collated batch that carries the loop's own route_y / route_ok (route_stage)."""
    b, J, wins = make_batch(world, n=n, seed=seed, with_rb=False)
    b["route_y"], b["route_ok"] = TR.route_stage(b)
    return b, J


def run_loss_p(m, b, p_gt, seed=3):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    out = m.loss(b, p_gt=p_gt, gen=gen)
    return out, gen.get_state()


# ------------------------------------------------------------------------------------------------ pins, flags, defaults
def test_the_snapshot_is_the_bcf3ba3f_model_and_tanitad_is_the_study_tree():
    assert git_blob_sha(SNAP) == SNAP_GIT_BLOB
    assert Path(M.__file__).resolve() == MODEL_P.resolve()
    assert "route_head" not in SNAP.read_text(encoding="utf-8")


def test_flags_default_off_have_flag_checks_a_choice_check_and_a_wm_only_default():
    assert TR.FLAG_CHECKS["aux_route"] == "lever:aux_route" and TR.FLAG_CHECKS["w_aux_route"] == "lever:w_aux_route"
    ps = TR.build_parser()
    assert ps.get_default("aux_route") == "off" and ps.get_default("w_aux_route") == 0.0
    acts = {a.dest: a for a in ps._actions}
    assert acts["aux_route"].option_strings == ["--aux-route"] and acts["w_aux_route"].option_strings == ["--w-aux-route"]
    assert {"aux_route", "w_aux_route"} <= set(TR.WM_ONLY_DEFAULT_DESTS)
    assert M.V8FMiniConfig(**TINY).aux_route == "off" and M.V8FMiniConfig(**TINY).w_aux_route == 0.0
    c = TR.make_config(ps.parse_args(["--out", "x", "--aux-route", "cond", "--w-aux-route", "0.5"]))
    assert c.aux_route == "cond" and c.w_aux_route == 0.5
    assert TR.make_config(ps.parse_args(["--out", "x"])).aux_route == "off"
    assert M.ROUTE_KEY_PREFIXES == ("route_head.", "route_emb.") and all(p in M.NEW_LEVER_KEY_PREFIXES for p in M.ROUTE_KEY_PREFIXES), "the init-ckpt gate must admit exactly these fresh keys"
    assert M.N_ROUTE == 3 and M.AUX_ROUTE_MODES == ("off", "cond")


def test_the_ce_mode_is_refused_in_argparse_naming_the_card(capsys):
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--aux-route", "ce", "--w-aux-route", "0.5"])
    err = capsys.readouterr().err
    assert "PREREG_V8F_N1DH" in err and "ce" in err and "NOT built" in err
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--aux-route", "maybe"])
    with pytest.raises(ValueError, match="aux_route"):
        M.V8FMiniConfig(**{**TINY, "aux_route": "ce"})


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf"), "0.5", None, True])
def test_config_refuses_a_bad_weight(bad):
    with pytest.raises(ValueError, match="w_aux_route"):
        M.V8FMiniConfig(**{**TINY, "aux_route": "cond", "w_aux_route": bad})
    assert M.V8FMiniConfig(**{**TINY, "aux_route": "cond", "w_aux_route": 0.5}).w_aux_route == 0.5
    with pytest.raises(TypeError):
        M.V8FMiniConfig(aux_routez="cond")


def test_config_refuses_the_two_modes_whose_own_code_mask_names_nothing():
    with pytest.raises(ValueError, match="code_mix whole"):
        M.V8FMiniConfig(**{**TINY, **COND, "code_mix": "kind"})
    with pytest.raises(ValueError, match="o_interface full"):
        M.V8FMiniConfig(**{**TINY, **COND, "o_interface": "cut"})


# ------------------------------------------------------------------------------------------------ Z0
def test_z0_flag_off_build_is_the_bcf3ba3f_model_bit_for_bit(world):
    b, J = routed(world)
    mt, mn = build(J, mod=base_module()), build(J, mod=M)
    assert mn.route_head is None and mn.route_emb is None and "aux_route" not in mn.built_levers() and "ROUTE" not in mn.trainable_groups()
    assert list(mt.state_dict()) == list(mn.state_dict())
    assert all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items()), "state dict values"
    assert mt.built_levers() == mn.built_levers()
    ot, on = run_loss(mt, b), run_loss(mn, b)
    torch.manual_seed(3); _ = mn.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_n = torch.get_rng_state().clone()
    torch.manual_seed(3); _ = mt.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_t = torch.get_rng_state().clone()
    assert torch.equal(rng_t, rng_n), "the global RNG stream after one loss call"
    probs, extra = equal_runs(ot, on)
    assert probs == [] and extra == [], (probs, extra)
    assert set(on) == set(ot) and set(on["terms"]) == set(ot["terms"]) and "route" not in on["terms"] and "route_logits" not in on, "no new key, no new term"
    assert torch.equal(ot["terms"]["total"], on["terms"]["total"])
    ot["terms"]["total"].backward(); on["terms"]["total"].backward()
    pn = dict(mn.named_parameters())
    assert all(torch.equal(p.grad, pn[n].grad) for n, p in mt.named_parameters() if p.grad is not None), "gradients"
    mt.eval(); mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    assert all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k])), "forward_infer"


def test_z0_a_batch_that_carries_route_keys_changes_nothing_on_an_off_model(world):
    b, J = routed(world)
    b2 = {k: v for k, v in b.items() if k not in ("route_y", "route_ok")}
    m = build(J)
    o1, o2 = run_loss(m, b), run_loss(m, b2)
    probs, extra = equal_runs(o1, o2)
    assert probs == [] and extra == [] and torch.equal(o1["terms"]["total"], o2["terms"]["total"])


def test_z0_the_off_build_draws_nothing_from_the_global_rng_beyond_the_snapshots(world):
    J = load_J(world)
    s0 = torch.get_rng_state().clone()
    torch.manual_seed(0); build(J, mod=base_module()); a = torch.get_rng_state().clone()
    torch.manual_seed(0); build(J, mod=M); c = torch.get_rng_state().clone()
    assert torch.equal(a, c) and not torch.equal(a, s0)


# ------------------------------------------------------------------------------------------------ Z1
def test_z1_cond_at_step_0_equals_off_bit_for_bit_except_the_route_term_and_the_rng_streams_are_the_off_models(world):
    b, J = routed(world)
    torch.manual_seed(0); m_off = build(J, route=False); rng_off = torch.get_rng_state().clone()
    torch.manual_seed(0); m_on = build(J, route=True); rng_on = torch.get_rng_state().clone()
    assert m_on.route_head is not None and float(m_on.route_emb.weight.abs().sum()) == 0.0
    assert tuple(m_on.route_head.weight.shape) == (3, m_on.cfg.n_q * m_on.cfg.d) and m_on.route_head.bias is not None and m_on.route_emb.bias is None
    assert tuple(m_on.route_emb.weight.shape) == (m_on.cfg.n_q * m_on.cfg.d, 3)
    assert torch.equal(rng_off, rng_on), "building the head must not move the global RNG stream (built LAST under a forked RNG)"
    so, sn = m_off.state_dict(), m_on.state_dict()
    assert list(sn) == list(so) + ["route_head.weight", "route_head.bias", "route_emb.weight"], "the route modules are the LAST keys"
    assert all(torch.equal(v, sn[k]) for k, v in so.items()), "every non-route parameter and buffer, bit for bit"
    assert m_on.built_levers()["aux_route"] == "cond" and m_on.built_levers()["w_aux_route"] == W and "ROUTE" in m_on.trainable_groups() and "aux_route" not in m_off.built_levers()
    (o_off, g_off), (o_on, g_on) = run_loss_p(m_off, b, 0.5), run_loss_p(m_on, b, 0.5)
    probs, extra = equal_runs(o_off, o_on)
    assert probs == [], probs
    assert extra == ["route"], extra
    assert torch.equal(g_off, g_on), "the batch generator's state: the own-code mask draws nothing new"
    assert torch.equal(o_off["traj30"], o_on["traj30"]) and torch.equal(o_off["traj60"], o_on["traj60"]) and torch.equal(o_off["code"], o_on["code"])
    expect_total = o_off["terms"]["total"] + W * o_on["terms"]["route"]
    torch.testing.assert_close(o_on["terms"]["total"], expect_total, rtol=1e-6, atol=0)      # x + w * route: equal to fp32 round-off
    assert float(o_on["terms"]["route"]) > 0.0
    m_off.eval(); m_on.eval()
    with torch.no_grad():
        i0, i1 = m_off.forward_infer(b), m_on.forward_infer(b)
    assert all(torch.equal(i0[k], i1[k]) for k in i0 if torch.is_tensor(i0[k])), "forward_infer (the plan head output) bit-identical at step 0"
    assert set(i1) - set(i0) == {"route_logits", "route_own"} and bool(i1["route_own"].all()), "at inference every row reads T's own code"


# ------------------------------------------------------------------------------------------------ REACH
def test_reach_nonzero_gradients_on_both_modules_and_the_controls_read_exactly_zero(world):
    b, J = routed(world)
    m = build(J, route=True, detach_upper=True)
    out, _ = run_loss_p(m, b, 0.0)
    T = out["terms"]
    assert float(out["route_own"].sum()) == b["route_y"].shape[0], "p_gt 0: every row reads T's own code"
    g = lambda t, w: (lambda x: 0.0 if x is None else float(x.abs().sum()))(torch.autograd.grad(t, [w], retain_graph=True, allow_unused=True)[0])
    assert g(T["route"], m.route_head.weight) > 0.0 and g(T["total"], m.route_head.weight) > 0.0, "REACH: the CE gradient on route_head.weight"
    assert g(T["route"], m.route_head.bias) > 0.0
    assert g(T["total"], m.route_emb.weight) > 0.0 and g(T["ade"], m.route_emb.weight) > 0.0, "REACH: a non-zero gradient on route_emb.weight (head_o is live)"
    mnz = build(J, route=True, detach_upper=True)
    set_route_emb(mnz)                                                                                                 # the control is vacuous at the zero init: read it with a live embedding, and show it is non-zero without the detach (next test)
    Tz = run_loss_p(mnz, b, 0.0)[0]["terms"]
    assert g(Tz["ade"], mnz.route_head.weight) == 0.0, "CONTROL: the decoder term puts EXACTLY 0.0 on route_head.weight (the posterior is detached)"
    assert g(Tz["route"], mnz.route_emb.weight) == 0.0
    assert g(T["route"], m.route_emb.weight) == 0.0, "CONTROL: the CE does not reach route_emb"
    for k in ("ckpt", "goal", "lat", "lon", "cons"):                                                                    # CONTROLS: S's and T's terms do not depend on O: no gradient at all
        for w in (m.route_head.weight, m.route_emb.weight):
            assert g(T[k], w) == 0.0, k
    # the independent recompute of route_head.weight's gradient: dL/dW = sum_b (softmax - onehot)_b^T . h_b / n over the CE rows, from the logits the loss used
    own_ok = (b["route_ok"].bool() & out["route_own"])
    lg = out["route_logits"].detach()
    dl = (torch.softmax(lg, -1) - torch.nn.functional.one_hot(b["route_y"], 3).float()) * own_ok[:, None].float() / own_ok.sum().clamp_min(1)
    hs = []
    hk = m.route_head.register_forward_hook(lambda mod, inp, outp: hs.append(inp[0].detach()))
    out2, _ = run_loss_p(m, b, 0.0)
    hk.remove()
    want = dl.T @ hs[-1]
    gw = torch.autograd.grad(out2["terms"]["route"], [m.route_head.weight])[0]
    torch.testing.assert_close(gw, want, rtol=1e-4, atol=1e-8)
    # CONTROL: at the REAL init (head_o's last layer zero) every gradient upstream of it, route_emb.weight's included, is EXACTLY 0.0
    mz = build(J, route=True)
    with torch.no_grad():
        mz.head_o[2].weight.zero_(); mz.head_o[2].bias.zero_()
    gz = torch.autograd.grad(run_loss_p(mz, b, 0.0)[0]["terms"]["total"], [mz.route_emb.weight], allow_unused=True)[0]
    assert gz is None or float(gz.abs().sum()) == 0.0
    assert all(p.grad is None for p in m.prior.parameters()), "the frozen P10 prior: no gradient at all"


def test_reach_a_batch_with_no_own_code_row_gives_exactly_zero_ce_value_and_gradient(world):
    b, J = routed(world)
    m = build(J, route=True, detach_upper=True)
    b2 = dict(b)
    b2["lab_t"] = {**b["lab_t"], "has": torch.ones_like(b["lab_t"]["has"])}                 # every row has a T label: p_gt = 1.0 substitutes the GT code on every row
    out, _ = run_loss_p(m, b2, 1.0)
    assert int(out["route_own"].sum()) == 0 and float(out["terms"]["route"]) == 0.0
    gw = torch.autograd.grad(out["terms"]["route"], [m.route_head.weight, m.route_head.bias], allow_unused=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in gw), "EXACTLY 0.0 CE gradient"
    out_n, _ = run_loss_p(m, b2, 0.0)
    assert float(out_n["terms"]["route"]) > 0.0, "the same batch at p_gt 0: the term is live (the 0.0 above is the mask, not a dead head)"


# ------------------------------------------------------------------------------------------------ the own-code mask
def test_own_the_ce_rows_are_exactly_the_rows_where_o_read_t_own_code(world):
    b, J = routed(world, n=32, seed=1)
    m = build(J, route=True, detach_upper=True)
    for p_gt in (0.0, 0.5, 1.0):
        out, _ = run_loss_p(m, b, p_gt)
        own = out["route_own"]
        own_code, gt_code = m._code_pred(out).detach(), m._code_gt(b["lab_t"]).detach()
        has = b["lab_t"]["has"]
        assert torch.equal(out["code"][own], own_code[own]), "rows named own read T's own code"
        sub = ~own
        assert torch.equal(out["code"][sub], gt_code[sub]) and bool(has[sub].all()), "the other rows read the GT code (and only rows that have a T label can)"
        assert bool(own[~has].all()), "a row with no T label reads T's own code"
        rm = (b["route_ok"].bool() & own).float()
        ce = torch.nn.functional.cross_entropy(out["route_logits"].detach().float(), b["route_y"], reduction="none")
        want = (ce * rm).sum() / rm.sum().clamp_min(1.0)
        torch.testing.assert_close(out["terms"]["route"].detach(), want, rtol=1e-6, atol=0)
        if p_gt == 0.0:
            assert bool(own.all())
        if p_gt == 1.0:
            assert bool((own == ~has).all())
    out, _ = run_loss_p(m, b, 0.5)
    assert 0 < int(out["route_own"].sum()) < b["route_y"].shape[0], "a mixed batch at p_gt 0.5"


# ------------------------------------------------------------------------------------------------ LIVE
def test_after_one_optimiser_step_the_plan_differs_from_the_flag_off_plan(world):
    b, J = routed(world)
    m_off, m_on = build(J, route=False), build(J, route=True)
    plans = []
    for m in (m_off, m_on):
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        run_loss(m, b)["terms"]["total"].backward()
        opt.step()
        plans.append(run_loss(m, b)["traj30"].detach())
    assert float(m_on.route_emb.weight.abs().sum()) > 0.0, "route_emb moved off its zero init"
    assert float((plans[0] - plans[1]).abs().max()) > 0.0, "the plan differs once the conditioning has a weight"


def test_the_conditioning_reads_the_detached_posterior_of_h_and_moves_the_plan(world):
    """run_o called directly: the plan moves with route_emb; with route_emb zero it is the off plan; the embedding adds route_emb(softmax(route_head(hf))) where hf is the flattened reader output."""
    b, J = routed(world)
    m = build(J, route=True)
    m.eval()
    with torch.no_grad():
        adapt = m._encode(b)
        B = b["feats"].shape[0]
        last = torch.full((B,), b["feats"].shape[1] - 1, dtype=torch.long)
        ck = m.ck_nom[None].expand(B, -1, -1)
        ok = torch.ones(B, 3)
        code = torch.rand(B, M.CODE_DIM, generator=torch.Generator().manual_seed(1))
        t0, _ = m.run_o(b, adapt, last, ck, ok, code)
        lg0 = m._route_logits.clone()
        set_route_emb(m)
        t1, _ = m.run_o(b, adapt, last, ck, ok, code)
        assert float((t1 - t0).abs().max()) > 0.0
        assert torch.equal(m._route_logits, lg0), "route_head reads h BEFORE the add: its logits do not depend on route_emb"


# ------------------------------------------------------------------------------------------------ admissibility
def test_admissibility_assert_no_situation_classifier_input_stays_green_with_cond(world):
    b, J = routed(world)
    m = build(J, route=True)
    set_route_emb(m)
    r = M.assert_no_situation_classifier_input(m, b)
    assert set(r["read"]) <= set(M.INFERENCE_INPUT_KEYS) | M.NAV_RB_KEYS
    assert M.scan_source_for_forbidden(Path(MODEL_P).read_text(encoding="utf-8")) == []
    m.eval()
    with torch.no_grad():                                                                  # no label, no route_y, no GT reaches the plan: removing every label block leaves the plan unchanged
        x0 = m.forward_infer(b)["traj60"]
        x1 = m.forward_infer({k: v for k, v in b.items() if not k.startswith(("lab", "gt", "route"))})["traj60"]
    assert torch.equal(x0, x1)


# ------------------------------------------------------------------------------------------------ K-LABEL
def _arc(kind, n=30, speed=10.0, dt=0.2, yaw_total=0.0):
    """An ego-frame track [n, 2]: constant speed, heading turning by yaw_total over the horizon (analytic)."""
    h = np.linspace(0.0, yaw_total, n)
    x = np.cumsum(speed * dt * np.cos(h)); y = np.cumsum(speed * dt * np.sin(h))
    return np.stack([x, y], 1)


def test_klabel_route_stage_equals_the_label_source_called_directly_on_analytic_arcs():
    from taniteval import four_families as FF
    from tanitad.refs import refc_tactical as RT
    tracks = [_arc("s", yaw_total=0.0), _arc("l", yaw_total=0.9), _arc("r", yaw_total=-0.9), _arc("l2", yaw_total=0.3), _arc("r2", yaw_total=-0.3), _arc("k", yaw_total=0.05), np.zeros((30, 2))]
    gt = torch.as_tensor(np.stack(tracks), dtype=torch.float32)
    y, ok = TR.route_stage({"gt": gt, "gt_ok": torch.ones(len(tracks), dtype=torch.bool)})
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(gt, 0.2)
    lat, _ = RT.factor_from_kinematics(dyaw, dv, v0, v1)
    assert torch.equal(y, lat) and y.dtype == torch.long and bool(ok.all())
    assert y.tolist() == [0, 1, 2, 1, 2, 0, 0], y.tolist()                                    # |dyaw| > 0.15 rad is a turn; the never-moved window is lane_keep
    # route_ok: gt_ok False, a non-finite knot, and aux_ok False each remove a row
    gt2 = gt.clone(); gt2[3, 5, 0] = float("nan")
    okm = torch.tensor([True, True, False, True, True, True, True])
    _, ok2 = TR.route_stage({"gt": gt2, "gt_ok": okm, "aux_ok": torch.tensor([True, True, True, True, False, True, True])})
    assert ok2.tolist() == [True, True, False, False, False, True, True]


def test_klabel_real_d1ho_sample_class_counts_equal_the_label_source_called_directly():
    """The real D1-HO GT tracks (the 5,736 held-out windows of the N1D dump): route_stage's class counts equal the counts of the two label-source calls (D1-HO's own ``labels``), and the majority share is the register's 0.338215."""
    import glob
    fs = sorted(glob.glob(str(D1HO_DUMP / "ep*.npz")))
    if not fs:
        pytest.skip(f"the D1-HO dump {D1HO_DUMP} is not on this box (set N1DH_D1HO_DUMP)")
    from taniteval import four_families as FF
    from tanitad.refs import refc_tactical as RT
    g = np.concatenate([np.load(f)["g"] for f in fs])
    assert g.shape == (5736, 30, 2), g.shape
    t = torch.as_tensor(np.asarray(g, np.float64))                                              # D1-HO's reader: the float64 tensor, DT 0.2
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, 0.2)
    lat_ref = RT.factor_from_kinematics(dyaw, dv, v0, v1)[0].numpy().astype(np.int64)
    y, ok = TR.route_stage({"gt": torch.as_tensor(g, dtype=torch.float32), "gt_ok": torch.ones(g.shape[0], dtype=torch.bool)})        # the trainer's call (float32 tracks, promoted to float64 inside)
    cnt_ref, cnt = np.bincount(lat_ref, minlength=3), np.bincount(y.numpy(), minlength=3)
    print("K-LABEL real D1-HO class counts (reader / route_stage):", cnt_ref.tolist(), cnt.tolist(), "majority share", cnt_ref.max() / cnt_ref.sum())
    assert bool(ok.all())
    assert cnt_ref.tolist() == cnt.tolist(), (cnt_ref.tolist(), cnt.tolist())
    assert abs(cnt_ref.max() / cnt_ref.sum() - 0.338215) < 5e-7, cnt_ref.tolist()
    assert (lat_ref == y.numpy()).all(), "bit for bit: the labels equal the reader's, window by window"


def test_klabel_the_label_is_computed_from_gt_only_and_a_within_batch_permutation_is_detected(world):
    """Mutation (c) of the card: ``route_y`` permuted within the batch. The label of row i must be the class of row i's own GT track: the sensitivity test K-LABEL uses (a permuted label disagrees on a batch with mixed classes)."""
    b, J, _ = make_batch(world, n=16, seed=0, with_rb=False)
    gt = b["gt"].clone()
    gt[:5] = torch.as_tensor(np.stack([_arc("l", yaw_total=0.9)] * 5), dtype=torch.float32)
    gt[5:10] = torch.as_tensor(np.stack([_arc("r", yaw_total=-0.9)] * 5), dtype=torch.float32)
    b2 = {**b, "gt": gt}
    y, ok = TR.route_stage(b2)
    assert y[:5].tolist() == [1] * 5 and y[5:10].tolist() == [2] * 5
    perm = torch.roll(torch.arange(16), 3)
    assert not torch.equal(y[perm], y), "a permuted label differs from the own-track label on a mixed batch"
    # row independence: one row's label does not depend on the rest of the batch
    y1, _ = TR.route_stage({"gt": gt[2:3], "gt_ok": b["gt_ok"][2:3]})
    assert int(y1[0]) == int(y[2])


# ------------------------------------------------------------------------------------------------ G-DVB
def aux_only(msgs):
    return [m for m in msgs if "aux_route" in m or "w_aux_route" in m]


ARGS = ["--aux-route", "cond", "--w-aux-route", "0.5", "--detach-upper"]


def test_gdvb_passes_an_honoured_argv_and_names_the_levers_in_built_levers(world):
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_ok")
    assert aux_only(dvb(parts)) == [], dvb(parts)
    lv = parts[2].built_levers()
    assert lv["aux_route"] == "cond" and lv["w_aux_route"] == 0.5 and parts[2].cfg.aux_route == "cond" and parts[2].route_head is not None
    parts0 = gate_parts(world, [*RBD(world), "--aux-route", "off"], "ar_off")
    assert aux_only(dvb(parts0)) == [] and "aux_route" not in parts0[2].built_levers() and parts0[2].route_head is None
    assert aux_only(dvb(gate_parts(world, [*RBD(world)], "ar_default"))) == []


def test_gdvb_refuses_declared_cond_but_built_without_the_modules(world):
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_dropped")
    parts[2].route_head, parts[2].route_emb = None, None                                    # the build dropped the modules (the refcv6 class: declared, never built)
    assert any("lever aux_route: declared cond" in m and "built off" in m for m in dvb(parts)), dvb(parts)
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_dropped_cfg")
    parts[2].route_head, parts[2].route_emb = None, None
    parts[2].cfg.aux_route = "off"
    assert any("lever aux_route: declared cond" in m for m in dvb(parts)), dvb(parts)


def test_gdvb_refuses_declared_off_but_built_with_the_modules(world):
    parts = gate_parts(world, [*RBD(world), "--aux-route", "off"], "ar_extra")
    n = parts[2].cfg.n_q * parts[2].cfg.d
    parts[2].cfg.aux_route = "cond"
    parts[2].route_head, parts[2].route_emb = torch.nn.Linear(n, 3), torch.nn.Linear(3, n, bias=False)
    msgs = dvb(parts)
    assert any("lever aux_route: declared off / weight 0.0 but built cond" in m for m in msgs), msgs


def test_gdvb_refuses_a_zero_weight_with_cond_and_a_weight_with_off(world):
    parts = gate_parts(world, [*RBD(world), "--aux-route", "cond", "--detach-upper"], "ar_w0")
    assert any("declared with --w-aux-route 0.0" in m and "does nothing" in m for m in dvb(parts)), dvb(parts)
    parts = gate_parts(world, [*RBD(world), "--aux-route", "off", "--w-aux-route", "0.5"], "ar_woff")
    assert any("declared but --aux-route is off" in m for m in dvb(parts)), dvb(parts)


def test_gdvb_refuses_a_nonzero_start_wrong_shapes_a_bias_and_a_head_width_mismatch(world):
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_bad")
    set_route_emb(parts[2])
    assert any("not zero at the start" in m for m in dvb(parts)), "a non-zero embedding at the start"
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_shape")
    n = parts[2].cfg.n_q * parts[2].cfg.d
    parts[2].route_head = torch.nn.Linear(n, 4)
    assert any("route_head is" in m for m in dvb(parts)), "a wrong class count"
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_bias")
    parts[2].route_emb = torch.nn.Linear(3, n, bias=True)
    torch.nn.init.zeros_(parts[2].route_emb.weight)
    assert any("route_emb is" in m and "without a bias" in m for m in dvb(parts)), "a bias on the embedding"
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_width")
    parts[2].route_emb = torch.nn.Linear(3, n + 1, bias=False)
    torch.nn.init.zeros_(parts[2].route_emb.weight)
    msgs = dvb(parts)
    assert any("route_emb is" in m for m in msgs) or any("head_o reads" in m for m in msgs), msgs


def test_gdvb_refuses_the_t2_bank_and_the_optimiser_that_lacks_or_doubles_a_module(world):
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_t2")
    parts[0].t2_bank_p = 0.1
    assert any("--t2-bank-p > 0" in m for m in aux_only(dvb(parts))), dvb(parts)
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_opt")
    a, cfg, model, opt, info = parts
    opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad and p is not model.route_emb.weight], lr=a.lr)
    assert any("route_emb.weight is in 0 optimiser groups" in m for m in aux_only(dvb((a, cfg, model, opt2, info)))), "route_emb missing from the optimiser"
    import types
    opt3 = types.SimpleNamespace(param_groups=[{"params": list(opt.param_groups[0]["params"]), "lr": a.lr, "weight_decay": a.wd}, {"params": [model.route_head.weight], "lr": a.lr, "weight_decay": a.wd}])
    assert any("route_head.weight is in 2 optimiser groups" in m for m in aux_only(dvb((a, cfg, model, opt3, info)))), "route_head twice"


def test_a_flag_without_a_gdvb_entry_is_refused_by_the_gate_itself(world, monkeypatch):
    parts = gate_parts(world, [*RBD(world), *ARGS], "ar_noentry")
    assert [m for m in dvb(parts) if "declared-vs-built entry" in m] == []
    for dest in ("aux_route", "w_aux_route"):
        monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != dest})
        bad = dvb(parts)
        assert any("flags without a declared-vs-built entry" in m and dest in m for m in bad), (dest, bad)
        monkeypatch.undo()


def test_main_refuses_each_bad_argv_before_anything_trains(world):
    for tag, extra, needle in (("ar_w0", ["--aux-route", "cond", "--detach-upper"], "w-aux-route"),
                               ("ar_cut", [*ARGS, "--o-interface", "cut"], "o_interface"),
                               ("ar_kind", [*ARGS, "--code-mix", "kind"], "code_mix")):
        with pytest.raises((SystemExit, ValueError)) as e:
            TR.main(_argv(world, tag, ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world), *extra]))
        assert "ZZDVB-REFUSEDZZ" in str(e.value) or needle in str(e.value), (tag, str(e.value)[:300])
    with pytest.raises(SystemExit):                                                          # argparse: ce
        TR.main(_argv(world, "ar_ce", ["--steps", "1", "--selfcheck-only", *N1(world), *RBD(world), "--aux-route", "ce", "--w-aux-route", "0.5"]))


# ------------------------------------------------------------------------------------------------ the probe (G-LIVE)
def test_probe_off_runs_nothing_and_on_reads_gradients_identity_and_exact_zero_controls(world):
    b, J = routed(world)
    m0 = build(J, route=False)
    r0 = TR.aux_route_probe(m0, b, m0.cfg)
    assert r0["aux_route"] == "off" and r0["pass"] is True
    for det in (False, True):
        m = build(J, route=True, detach_upper=det)
        with torch.no_grad():
            m.head_o[2].weight.zero_(); m.head_o[2].bias.zero_()                              # the REAL init
        r = TR.aux_route_probe(m, b, m.cfg)
        assert r["pass"] and r["head_grad_route"] > 0.0 and r["head_grad_total"] > 0.0 and r["emb_grad_total"] > 0.0 and r["emb_grad_ade"] > 0.0 and r["plan_move_when_emb_set"] > 0.0
        assert r["control_decoder_grad_on_head"] == 0.0 and r["control_tside_grad_on_head"] == 0.0 and r["control_tside_grad_on_emb"] == 0.0
        assert r["control_repeat_grad_diff"] == 0.0 and r["control_prior_grad"] == 0.0 and r["control_real_init_emb_grad"] == 0.0
        assert r["control_no_own_row_route_term"] == 0.0 and r["control_no_own_row_head_grad"] == 0.0 and r["own_rows_probe"] > 0
        assert r["identity_terms_differing"] == [] and r["identity_outputs_differing"] == [] and r["head_o_last_layer_zero_at_init"] is True
        assert sum(r["probe_class_counts"]) == r["probe_labelled_rows"] <= r["probe_rows"]
    assert float(m.route_emb.weight.abs().sum()) == 0.0, "the probe works on a COPY"


def test_probe_on_a_fine_tune_start_with_a_live_head_needs_no_reseeding(world):
    b, J = routed(world)
    m = build(J, route=True, detach_upper=True)                                              # head_o's last layer already non-zero (the M2b checkpoint's)
    r = TR.aux_route_probe(m, b, m.cfg)
    assert r["pass"] and r["head_o_last_layer_zero_at_init"] is False and r["control_real_init_emb_grad"] > 0.0


def _mut(world, old, new, name, **kw):
    b, J = routed(world)
    m = build(J, mod=mutant(MODEL_P, old, new, name), route=True, **kw)
    return b, m


def test_probe_refuses_a_model_whose_conditioning_line_is_deleted(world):
    b, m = _mut(world, '                hf = hf + self.route_emb(torch.softmax(self._route_logits, -1).detach())', "                pass", "_mut_n1dh_noline", detach_upper=True)
    with pytest.raises(AssertionError, match="conditioning does not reach the loss"):
        TR.aux_route_probe(m, b, m.cfg)


def test_probe_refuses_a_model_whose_ce_term_is_deleted_from_the_total(world):
    b, m = _mut(world, '        total = total + cfg.w_aux_route * T["route"] if cfg.aux_route != "off" else total', "        pass", "_mut_n1dh_noce", detach_upper=True)
    with pytest.raises(AssertionError, match="CE term does not reach the total"):
        TR.aux_route_probe(m, b, m.cfg)


def test_probe_refuses_a_model_without_the_own_code_mask(world):
    b, m = _mut(world, 'rm = (batch["route_ok"].bool() & out["route_own"]).float()', 'rm = batch["route_ok"].bool().float()', "_mut_n1dh_nomask", detach_upper=True)
    with pytest.raises(AssertionError, match="own-code mask is not applied"):
        TR.aux_route_probe(m, b, m.cfg)


def test_probe_refuses_a_model_whose_posterior_is_not_detached(world):
    b, m = _mut(world, "torch.softmax(self._route_logits, -1).detach())", "torch.softmax(self._route_logits, -1))", "_mut_n1dh_nodetach", detach_upper=True)
    with pytest.raises(AssertionError, match="posterior is detached"):
        TR.aux_route_probe(m, b, m.cfg)


# ------------------------------------------------------------------------------------------------ the trainer, end to end
def test_end_to_end_n1dh_arm_probe_levers_group_class_record_and_checkpoint(world):
    out = Path(world["root"]) / "n1dh_on"
    assert TR.main(_argv(world, "n1dh_on", ["--steps", "4", *N1(world), *RBD(world), *ARGS])) == 0
    sc, sm = _read(out, "selfcheck.json"), _read(out, "summary.json")
    pr = sc["aux_route_probe"]
    assert sc["declared_vs_built"] == "PASS" and pr["pass"] and pr["head_grad_route"] > 0.0 and pr["control_decoder_grad_on_head"] == 0.0 and pr["control_no_own_row_route_term"] == 0.0
    assert sc["levers"]["aux_route"] == "cond" and sc["levers"]["w_aux_route"] == 0.5
    assert sc["live"]["groups_grad"]["ROUTE"] > 0.0 and sc["live"]["pass"] and "route" in sc["live"]["terms"]
    assert not any(n.startswith(("route_head.", "route_emb.")) for n in sc["live"]["dead_params"])
    assert sm["aux_route"] == "cond" and sm["route_rows"] > 0 and sm["route_ok"] > 0 and sum(sm["route_class_counts"].values()) == sm["route_ok"]
    assert 0.0 < sm["route_masked_share"] <= 1.0 and abs(sum(sm["route_class_shares"].values()) - 1.0) < 1e-9
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert ck["cfg"]["aux_route"] == "cond" and ck["cfg"]["w_aux_route"] == 0.5 and "route_head.weight" in ck["model"] and float(ck["model"]["route_emb.weight"].abs().sum()) > 0.0, "the embedding trained"
    mm = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(D.load_join(Path(world["root"]) / "join_train.npz")))
    mm.load_state_dict(ck["model"])                                                             # the evaluators' build path, strict
    assert mm.route_head is not None and mm.built_levers()["aux_route"] == "cond"
    assert ck["route_totals"]["ok"] == sm["route_ok"]


def test_end_to_end_flag_off_and_explicit_off_keep_the_record_set(world):
    for tag, extra in (("aroff", []), ("aroff2", ["--aux-route", "off"])):
        out = Path(world["root"]) / tag
        assert TR.main(_argv(world, tag, ["--steps", "2", *N1(world), *RBD(world), *extra])) == 0
        sc, sm = _read(out, "selfcheck.json"), _read(out, "summary.json")
        ck = torch.load(out / "ckpt_last.pt", weights_only=False)
        assert "aux_route_probe" not in sc and "aux_route" not in sc["levers"] and "ROUTE" not in sc["live"]["groups_grad"] and "route_rows" not in sm and "route_totals" not in ck
        assert not any(k.startswith(("route_head.", "route_emb.")) for k in ck["model"]) and ck["cfg"]["aux_route"] == "off"


def test_fine_tune_from_a_checkpoint_without_the_route_keys_passes_the_init_ckpt_gate_and_only_with_the_flag(world):
    """N1D's argv fine-tunes M2b_s0, which has no route modules: the init-ckpt G-DVB must admit exactly those fresh keys, and only when --aux-route cond (the N1R mint lesson of 2026-11-18)."""
    base = Path(world["root"]) / "ar_base_init"
    assert TR.main(_argv(world, "ar_base_init", ["--steps", "2", *N1(world)])) == 0
    ck0 = torch.load(base / "ckpt_last.pt", weights_only=False)
    assert not any(k.startswith("route_") for k in ck0["model"])
    out = Path(world["root"]) / "ar_from_init"
    assert TR.main(_argv(world, "ar_from_init", ["--steps", "1", "--selfcheck-only", "--init-ckpt", str(base / "ckpt_last.pt"), *N1(world), *RBD(world), *ARGS])) == 0
    sc = _read(out, "selfcheck.json")
    ic = sc["info"]["init_ckpt"]
    assert sc["declared_vs_built"] == "PASS" and ic["applied"] and {"route_head.weight", "route_head.bias", "route_emb.weight"} <= set(ic["declared_new_keys"]) and set(ic.get("fresh_keys", [])) <= set(ic["declared_new_keys"])
    assert sc["aux_route_probe"]["pass"] and sc["aux_route_probe"]["head_o_last_layer_zero_at_init"] is False, "after the load head_o is the checkpoint's trained head"
    out2 = Path(world["root"]) / "ar_from_init_off"
    assert TR.main(_argv(world, "ar_from_init_off", ["--steps", "1", "--selfcheck-only", "--init-ckpt", str(base / "ckpt_last.pt"), *N1(world), *RBD(world), "--detach-upper"])) == 0
    assert _read(out2, "selfcheck.json")["declared_vs_built"] == "PASS"


def test_resume_refuses_a_checkpoint_whose_cfg_differs_from_the_argv(world):
    argv = _argv(world, "ar_resume", ["--steps", "3", *N1(world), *RBD(world), *ARGS])
    assert TR.main(argv) == 0
    p = Path(world["root"]) / "ar_resume" / "ckpt_last.pt"
    ck = torch.load(p, weights_only=False)
    assert TR.main(argv) == 0                                                    # the honest checkpoint resumes (already done: returns)
    ck["cfg"]["aux_route"], ck["cfg"]["w_aux_route"] = "off", 0.0
    torch.save(ck, p)
    with pytest.raises(SystemExit, match="ZZRESUME-REFUSEDZZ the checkpoint's aux_route is off / 0.0 but the argv declares cond / 0.5"):
        TR.main(argv)


# ------------------------------------------------------------------------------------------------ R2: determinism
def test_r2_cpu_fp32_one_thread_is_deterministic_across_two_runs_of_the_whole_training_step(world):
    b, J = routed(world)
    outs = []
    for _ in range(2):
        m = build(J, route=True, detach_upper=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        for s in range(2):
            opt.zero_grad(set_to_none=True)
            o, _ = run_loss_p(m, b, 0.5, seed=s)
            o["terms"]["total"].backward()
            opt.step()
        outs.append((o["terms"]["total"].detach().clone(), {k: v.clone() for k, v in m.state_dict().items()}))
    assert torch.equal(outs[0][0], outs[1][0]) and all(torch.equal(v, outs[1][1][k]) for k, v in outs[0][1].items())
    assert torch.get_num_threads() == 1


# ------------------------------------------------------------------------------------------------ the loaders: round trip
LOADERS = ("v8f_mini_eval.py", "v8f_mini_eval_oacc.py", "v8f_mini_eval_rb.py", "v8f_mini_eval_rbd.py", "v8f_mini_eval_speedavg.py", "v8f_mini_eval_speedcal.py", "v8f_t1_eval.py", "v8f_t1_eval_rb.py",
           "v8f_t1_eval_rbd.py", "v8f_dagger_bank.py", "v8f_kscore.py")
CFG_CALL = re.compile(r"V8FMiniConfig\(\*\*((?:[^()]|\([^()]*\))*)\)")


def loader_cfg_exprs(name):
    src = (SCRIPTS / name).read_text(encoding="utf-8")
    return CFG_CALL.findall(src)


def _ckpt_of(model):
    return {"model": {k: v.clone() for k, v in model.state_dict().items()}, "cfg": dict(model.cfg.__dict__)}


@pytest.mark.parametrize("name", LOADERS)
def test_roundtrip_each_loader_builds_the_route_modules_from_the_checkpoint_cfg_strict_and_equal(name, world):
    b, J = routed(world)
    m = build(J, route=True, detach_upper=True)
    set_route_emb(m)
    ck = _ckpt_of(m)
    exprs = loader_cfg_exprs(name)
    assert exprs, f"{name} has no V8FMiniConfig(**...) build line (find it and add it to this test)"
    for ex in exprs:
        cfgd = eval(ex, {"ck": ck, "cfgd": ck["cfg"], "cfg": ck["cfg"]})                      # the loader's OWN expression, evaluated over the checkpoint
        mm = M.V8FMini(M.V8FMiniConfig(**cfgd), M.P10Prior.synthetic(0), *M.cons_stats(J))
        mm.load_state_dict(ck["model"])                                                       # STRICT: a route-less rebuild raises (unexpected key route_head.weight)
        assert mm.route_head is not None and mm.built_levers()["aux_route"] == "cond"
        assert list(mm.state_dict()) == list(m.state_dict()) and all(torch.equal(v, m.state_dict()[k]) for k, v in mm.state_dict().items())
        m.eval(); mm.eval()
        with torch.no_grad():
            o1, o2 = m.forward_infer(b), mm.forward_infer(b)
        assert all(torch.equal(o1[k], o2[k]) for k in o1 if torch.is_tensor(o1[k])), "outputs"


def test_roundtrip_a_flag_off_checkpoint_loads_through_every_loader_without_the_modules(world):
    b, J = routed(world)
    m = build(J, route=False)
    ck = _ckpt_of(m)
    for name in LOADERS:
        for ex in loader_cfg_exprs(name):
            mm = M.V8FMini(M.V8FMiniConfig(**eval(ex, {"ck": ck, "cfgd": ck["cfg"], "cfg": ck["cfg"]})), M.P10Prior.synthetic(0), *M.cons_stats(J))
            mm.load_state_dict(ck["model"])
            assert mm.route_head is None


def test_roundtrip_an_old_checkpoint_cfg_without_the_keys_loads_as_off(world):
    b, J = routed(world)
    ck = _ckpt_of(build(J, route=False))
    del ck["cfg"]["aux_route"], ck["cfg"]["w_aux_route"]                                       # a checkpoint written before the lever existed
    mm = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(J))
    mm.load_state_dict(ck["model"])
    assert mm.route_head is None and mm.cfg.aux_route == "off" and mm.cfg.w_aux_route == 0.0


def test_roundtrip_strat1_check_geval_accepts_a_route_checkpoint(world, tmp_path):
    import strat1_check as SC
    b, J = routed(world)
    m = build(J, route=True)
    set_route_emb(m)
    p = tmp_path / "ck.pt"
    torch.save(_ckpt_of(m), p)
    bad, rec = SC.geval(str(p), selfcheck_levers=m.built_levers())
    assert bad == [] and rec["built_levers"]["aux_route"] == "cond", (bad, rec)
    bad2, _ = SC.geval(str(p), selfcheck_levers={k: v for k, v in m.built_levers().items() if k not in ("aux_route", "w_aux_route")})
    assert any("aux_route" in x for x in bad2), bad2


def test_the_refcv_loaders_do_not_build_the_v8f_model():
    for rel in ("tanitad/eval/refcv6_loader.py", "tanitad/eval/refcv7_loader.py"):
        txt = (REPO / "stack" / rel).read_text(encoding="utf-8")
        assert "V8FMini" not in txt and "v8f_mini" not in txt, rel


def test_every_stack_script_that_builds_the_v8f_config_from_a_checkpoint_is_in_the_loader_list():
    found = set()
    for f in sorted(SCRIPTS.glob("*.py")):
        if re.search(r'V8FMiniConfig\(\*\*ck\["cfg"\]\)', f.read_text(encoding="utf-8")):
            found.add(f.name)
    assert found == set(LOADERS), sorted(found ^ set(LOADERS))
