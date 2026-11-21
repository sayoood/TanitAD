"""PREREG_V8F_N1DM (draft): the mode-routed plan head, ``--aux-route modes`` + ``--w-aux-route`` (model, trainer, loaders, gate). CPU only, fp32, ONE thread.

modes = N1DH's ``ce`` half (route_head CE on the kinematic TAC-LAT label, own-code rows only) + K = 3 zero-init residual plan branches of ``head_o`` (``route_br.k``: Linear(n_q * d -> head_o's output width), no bias) and NO route_emb.
Plan coefficients = head_o(h) + sum_k p_k * branch_k(h), p = softmax(route_head(h).float(), -1).detach() (SOFT, one deterministic form at training and inference, no argmax).

Z0a    flag off == the 1f217bf0 model (snapshot): state dict, loss terms, total, gradients, forward_infer, the RNG stream.
Z0b    ``cond`` in the new code == ``cond`` in the 1f217bf0 snapshot (N1DH stays bit for bit what it was).
Z1     modes at step 0 vs off, same init, same batch: every non-route parameter, every output, every loss term but ``route`` bit-identical; the total moves by exactly w x route; the RNG streams are the off model's.
REACH  every one of the K branches gets a NON-ZERO gradient (at the real init, behind head_o's zero last layer, and with a live head); controls that must read EXACTLY 0.0: the decoder term on route_head (the detach), the CE on the
       branches, the T/S-side terms, the batch with no own-code row, the frozen prior.
FORMULA the plan equals head_o(h) + sum_k softmax(route_head(h))_k branch_k(h) recomputed from hooks (train and eval); p permuted across rows changes the plan to the formula with the permuted p; no argmax / one-hot in the source.
ADMIT  ``assert_no_situation_classifier_input`` stays green with modes; no label reaches the plan.
G-DVB  every refusal clause of ``bad_aux_route`` for modes, the init-ckpt fresh keys, the resume check.
LOAD   every loader that rebuilds the model from a checkpoint cfg builds the branches (strict load, equal outputs).
Each mutation is a source edit that must turn a NAMED test RED (``run_mutants_n1dm.py`` drives them and writes ``mutation_table_n1dm.json``)."""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "stack" / "scripts", REPO / "taniteval", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import tanitad  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_n1dh import (CFG_CALL, LOADERS, SCRIPTS, W, _ckpt_of, aux_only, build, loader_cfg_exprs, routed, run_loss_p, world)  # noqa: E402,F401  (world: the module fixture)
from test_v8f_n1dw import (MODEL_P, N1, RBD, _argv, _read, dvb, equal_runs, gate_parts, git_blob_sha, mod_from_text, mutant, run_loss)  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(REPO.resolve()), tanitad.__file__

SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_n1dm.txt")
SNAP_GIT_BLOB = "31df3f3dfde7bdd51515fb980d87173542e154d7"          # git blob of stack/tanitad/models/v8f_mini.py at 1f217bf0 (N1DH's code of record, the parent of this build)
ARGS_M = ["--aux-route", "modes", "--w-aux-route", "0.5", "--detach-upper"]
ARGS_C = ["--aux-route", "cond", "--w-aux-route", "0.5", "--detach-upper"]
K = 3


def snap_module():
    return sys.modules.get("v8f_mini_snapshot_n1dm") or mod_from_text(SNAP.read_text(encoding="utf-8"), "v8f_mini_snapshot_n1dm", SNAP)


def bm(J, mod=M, **kw):
    """A tiny ``modes`` model (head_o's last layer and the channel seeded non-zero by ``build``)."""
    return build(J, mod=mod, route=True, aux_route="modes", **kw)


def set_branches(m, scale=0.05, seed=13):
    with torch.no_grad():
        for k, br in enumerate(m.route_br):
            br.weight.copy_(torch.randn(br.weight.shape, generator=torch.Generator().manual_seed(seed + k)) * scale)


def o_inputs(m, b):
    adapt = m._encode(b)
    B = b["feats"].shape[0]
    last = torch.full((B,), b["feats"].shape[1] - 1, dtype=torch.long)
    ck = m.ck_nom[None].expand(B, -1, -1)
    return (b, adapt, last, ck, torch.ones(B, 3), torch.rand(B, M.CODE_DIM, generator=torch.Generator().manual_seed(1)))


def plan_and_parts(m, b, roll=False):
    """run_o called directly with hooks: the plan, the prior and the three intermediates the formula needs (head_o's input hf and output, route_head's logits)."""
    cap = {}
    h1 = m.head_o.register_forward_hook(lambda mod, i, o: cap.update(hf=i[0].detach().clone(), c0=o.detach().clone()))
    h2 = m.route_head.register_forward_hook(lambda mod, i, o: (cap.update(lg=(o.roll(1, 0) if roll else o).detach().clone()), o.roll(1, 0) if roll else None)[1])
    try:
        traj, p30 = m.run_o(*o_inputs(m, b))
    finally:
        h1.remove(); h2.remove()
    return traj.detach(), p30.detach(), cap


def formula_plan(m, p30, cap):
    p = torch.softmax(cap["lg"].float(), -1)
    coef = cap["c0"] + sum(p[:, k:k + 1] * m.route_br[k](cap["hf"]) for k in range(K))
    res = torch.einsum("tk,bck->btc", m.basis, coef.reshape(-1, 2, m.cfg.nb) * m.cfg.res_scale_m)
    return M.upsample_prior(p30) + res


# ------------------------------------------------------------------------------------------------ pins, flags
def test_the_snapshot_is_the_1f217bf0_model_and_the_modes_constants():
    assert git_blob_sha(SNAP) == SNAP_GIT_BLOB
    assert Path(M.__file__).resolve() == MODEL_P.resolve()
    assert "route_br" not in SNAP.read_text(encoding="utf-8") and "route_head" in SNAP.read_text(encoding="utf-8")
    assert M.AUX_ROUTE_MODES == ("off", "cond", "modes") and M.N_ROUTE == K and M.ROUTE_BR_KEY_PREFIXES == ("route_br.",) and M.ROUTE_KEY_PREFIXES == ("route_head.", "route_emb.")
    assert all(p in M.NEW_LEVER_KEY_PREFIXES for p in M.ROUTE_KEY_PREFIXES + M.ROUTE_BR_KEY_PREFIXES), "the init-ckpt gate must admit exactly these fresh keys"


def test_parser_takes_modes_still_refuses_ce_and_unknown_and_the_config_refuses_the_bad_ones(capsys):
    ps = TR.build_parser()
    c = TR.make_config(ps.parse_args(["--out", "x", "--aux-route", "modes", "--w-aux-route", "0.5"]))
    assert c.aux_route == "modes" and c.w_aux_route == 0.5
    for bad in ("ce", "mode", "maybe"):
        with pytest.raises(SystemExit):
            ps.parse_args(["--aux-route", bad, "--w-aux-route", "0.5"])
    assert "NOT built" in capsys.readouterr().err
    for kw in (dict(code_mix="kind"), dict(o_interface="cut"), dict(w_aux_route=float("nan")), dict(w_aux_route=-1.0), dict(w_aux_route=True)):
        with pytest.raises(ValueError):
            M.V8FMiniConfig(**{"aux_route": "modes", "w_aux_route": 0.5, **kw})
    assert TR.FLAG_CHECKS["aux_route"] == "lever:aux_route"


# ------------------------------------------------------------------------------------------------ Z0
def test_z0a_flag_off_build_is_the_1f217bf0_model_bit_for_bit(world):
    b, J = routed(world)
    mt, mn = build(J, mod=snap_module()), build(J, mod=M)
    assert mn.route_head is None and mn.route_emb is None and mn.route_br is None and "aux_route" not in mn.built_levers() and "ROUTE" not in mn.trainable_groups()
    assert list(mt.state_dict()) == list(mn.state_dict()) and all(torch.equal(v, mn.state_dict()[k]) for k, v in mt.state_dict().items())
    ot, on = run_loss(mt, b), run_loss(mn, b)
    torch.manual_seed(3); _ = mn.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_n = torch.get_rng_state().clone()
    torch.manual_seed(3); _ = mt.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3)); rng_t = torch.get_rng_state().clone()
    assert torch.equal(rng_t, rng_n)
    probs, extra = equal_runs(ot, on)
    assert probs == [] and extra == [] and set(on) == set(ot) and torch.equal(ot["terms"]["total"], on["terms"]["total"])
    ot["terms"]["total"].backward(); on["terms"]["total"].backward()
    pn = dict(mn.named_parameters())
    assert all(torch.equal(p.grad, pn[n].grad) for n, p in mt.named_parameters() if p.grad is not None)
    mt.eval(); mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    assert all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k]))


def test_z0b_cond_in_the_new_code_is_the_1f217bf0_cond_bit_for_bit(world):
    b, J = routed(world)
    ms, mn = build(J, mod=snap_module(), route=True), build(J, mod=M, route=True)
    assert mn.route_emb is not None and mn.route_br is None and mn.built_levers()["aux_route"] == "cond"
    assert list(ms.state_dict()) == list(mn.state_dict()) and all(torch.equal(v, mn.state_dict()[k]) for k, v in ms.state_dict().items())
    with torch.no_grad():
        for m in (ms, mn):
            m.route_emb.weight.copy_(torch.randn(m.route_emb.weight.shape, generator=torch.Generator().manual_seed(13)) * 0.05)      # a LIVE conditioning on both
    (os_, gs), (on, gn) = run_loss_p(ms, b, 0.5), run_loss_p(mn, b, 0.5)
    probs, extra = equal_runs(os_, on)
    assert probs == [] and extra == [] and torch.equal(os_["terms"]["total"], on["terms"]["total"]) and torch.equal(gs, gn)
    os_["terms"]["total"].backward(); on["terms"]["total"].backward()
    pn = dict(mn.named_parameters())
    assert all(torch.equal(p.grad, pn[n].grad) for n, p in ms.named_parameters() if p.grad is not None)
    ms.eval(); mn.eval()
    with torch.no_grad():
        it, inn = ms.forward_infer(b), mn.forward_infer(b)
    assert all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k]))


def test_z0c_the_modes_build_draws_nothing_from_the_global_rng(world):
    J = load_J_(world)
    torch.manual_seed(0); build(J, mod=M); a = torch.get_rng_state().clone()
    torch.manual_seed(0); bm(J); c = torch.get_rng_state().clone()
    assert torch.equal(a, c), "the head and the branches are built LAST under a forked RNG"


def load_J_(world):
    from test_v8f_n1dw import load_J
    return load_J(world)


# ------------------------------------------------------------------------------------------------ Z1
def test_z1_modes_at_step_0_equals_off_bit_for_bit_except_the_route_term(world):
    b, J = routed(world)
    torch.manual_seed(0); m_off = build(J, route=False); rng_off = torch.get_rng_state().clone()
    torch.manual_seed(0); m_on = bm(J); rng_on = torch.get_rng_state().clone()
    assert m_on.route_emb is None and len(m_on.route_br) == K and all(float(br.weight.abs().sum()) == 0.0 for br in m_on.route_br)
    do, fl = int(m_on.head_o[2].out_features), m_on.cfg.n_q * m_on.cfg.d
    assert do == 2 * m_on.cfg.nb and all(tuple(br.weight.shape) == (do, fl) and br.bias is None for br in m_on.route_br)
    assert tuple(m_on.route_head.weight.shape) == (K, fl) and m_on.route_head.bias is not None
    assert torch.equal(rng_off, rng_on)
    so, sn = m_off.state_dict(), m_on.state_dict()
    assert list(sn) == list(so) + ["route_head.weight", "route_head.bias"] + [f"route_br.{k}.weight" for k in range(K)], "the route modules are the LAST keys"
    assert all(torch.equal(v, sn[k]) for k, v in so.items())
    lv = m_on.built_levers()
    assert lv["aux_route"] == "modes" and lv["w_aux_route"] == W and "ROUTE" in m_on.trainable_groups() and "aux_route" not in m_off.built_levers()
    n_grp = sum(p.numel() for mod in m_on.trainable_groups()["ROUTE"] for p in mod.parameters())
    assert n_grp == K * fl + K + K * do * fl, "ROUTE group = route_head + the K branches"
    (o_off, g_off), (o_on, g_on) = run_loss_p(m_off, b, 0.5), run_loss_p(m_on, b, 0.5)
    probs, extra = equal_runs(o_off, o_on)
    assert probs == [] and extra == ["route"], (probs, extra)
    assert torch.equal(g_off, g_on)
    assert torch.equal(o_off["traj30"], o_on["traj30"]) and torch.equal(o_off["traj60"], o_on["traj60"]) and torch.equal(o_off["code"], o_on["code"])
    torch.testing.assert_close(o_on["terms"]["total"], o_off["terms"]["total"] + W * o_on["terms"]["route"], rtol=1e-6, atol=0)
    assert float(o_on["terms"]["route"]) > 0.0
    m_off.eval(); m_on.eval()
    with torch.no_grad():
        i0, i1 = m_off.forward_infer(b), m_on.forward_infer(b)
    assert all(torch.equal(i0[k], i1[k]) for k in i0 if torch.is_tensor(i0[k])), "forward_infer bit-identical at step 0"
    assert set(i1) - set(i0) == {"route_logits", "route_own"}


# ------------------------------------------------------------------------------------------------ REACH
def test_reach_every_branch_gets_a_nonzero_gradient_and_the_controls_read_exactly_zero(world):
    b, J = routed(world)
    g = lambda t, w: (lambda x: 0.0 if x is None else float(x.abs().sum()))(torch.autograd.grad(t, [w], retain_graph=True, allow_unused=True)[0])
    for live_head in (True, False):                                                       # True: head_o's last layer live; False: the REAL zero init (the branch add sits BEHIND it, so each branch still reads non-zero)
        m = bm(J, detach_upper=True)
        if not live_head:
            with torch.no_grad():
                m.head_o[2].weight.zero_(); m.head_o[2].bias.zero_()
        T = run_loss_p(m, b, 0.0)[0]["terms"]
        for k, br in enumerate(m.route_br):
            assert g(T["total"], br.weight) > 0.0 and g(T["ade"], br.weight) > 0.0, (live_head, k)
            assert g(T["route"], br.weight) == 0.0, "CONTROL: the CE does not reach a branch"
        assert g(T["route"], m.route_head.weight) > 0.0 and g(T["total"], m.route_head.weight) > 0.0
    m = bm(J, detach_upper=True)
    out, _ = run_loss_p(m, b, 0.0)
    T = out["terms"]
    for k in ("ckpt", "goal", "lat", "lon", "cons"):
        for w in [m.route_head.weight] + [br.weight for br in m.route_br]:
            assert g(T[k], w) == 0.0, k
    mz = bm(J, detach_upper=True)
    set_branches(mz)                                                                      # live branches: only then can a missing detach show
    Tz = run_loss_p(mz, b, 0.0)[0]["terms"]
    assert g(Tz["ade"], mz.route_head.weight) == 0.0, "CONTROL: the decoder term puts EXACTLY 0.0 on route_head.weight (p is detached)"
    assert g(Tz["total"], mz.route_head.weight) == pytest.approx(W * g(Tz["route"], mz.route_head.weight), rel=1e-5), "route_head is trained by the CE alone"
    assert all(p.grad is None for p in m.prior.parameters())


def test_reach_a_batch_with_no_own_code_row_gives_exactly_zero_ce_and_the_branches_still_train(world):
    b, J = routed(world)
    m = bm(J, detach_upper=True)
    b2 = dict(b)
    b2["lab_t"] = {**b["lab_t"], "has": torch.ones_like(b["lab_t"]["has"])}
    out, _ = run_loss_p(m, b2, 1.0)
    assert int(out["route_own"].sum()) == 0 and float(out["terms"]["route"]) == 0.0
    gw = torch.autograd.grad(out["terms"]["route"], [m.route_head.weight, m.route_head.bias], allow_unused=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in gw)
    gb = torch.autograd.grad(out["terms"]["total"], [br.weight for br in m.route_br], allow_unused=True)
    assert all(x is not None and float(x.abs().sum()) > 0.0 for x in gb), "the soft selection is on EVERY row (the mask restricts the CE only)"


# ------------------------------------------------------------------------------------------------ FORMULA
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_formula_the_plan_is_head_o_plus_the_p_weighted_branches_recomputed_from_hooks(world, mode):
    b, J = routed(world)
    m = bm(J)
    set_branches(m)
    getattr(m, mode)()
    with torch.no_grad():
        traj, p30, cap = plan_and_parts(m, b)
    p = torch.softmax(cap["lg"].float(), -1)
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[0])) and float(p.max()) < 1.0 and float(p.min()) > 0.0, "SOFT: no row is one-hot"
    torch.testing.assert_close(traj, formula_plan(m, p30, cap), rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        for br in m.route_br:
            br.weight.zero_()
        t0, _, _ = plan_and_parts(m, b)
    assert float((traj - t0).abs().max()) > 0.0, "the branches move the plan"


def test_formula_p_permuted_across_rows_changes_the_plan_to_the_formula_with_the_permuted_p(world):
    b, J = routed(world)
    m = bm(J)
    set_branches(m)
    m.eval()
    with torch.no_grad():
        t_a, _, cap_a = plan_and_parts(m, b)
        t_r, p30, cap_r = plan_and_parts(m, b, roll=True)
    assert float((t_a - t_r).abs().max()) > 0.0, "a row permutation of p changes the plan"
    torch.testing.assert_close(t_r, formula_plan(m, p30, cap_r), rtol=1e-5, atol=1e-6)


def test_formula_the_selection_is_one_deterministic_soft_form_no_argmax_no_sampling():
    src = inspect.getsource(M.V8FMini.run_o)
    blk = src[src.index("if self.route_br is not None"):src.index("coef = coef.reshape")]
    blk = "\n".join(ln.split("#")[0] for ln in blk.split("\n"))                           # code only: the comments name the forbidden words
    assert "softmax" in blk and ".detach()" in blk
    for bad in ("argmax", "one_hot", "topk", "gumbel", "multinomial", "rand", "sample", ".max("):
        assert bad not in blk, bad


def test_the_posterior_is_detached_decoder_gradient_on_route_head_is_exactly_zero_after_training_steps(world):
    b, J = routed(world)
    m = bm(J, detach_upper=True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for s in range(3):
        opt.zero_grad(set_to_none=True)
        o, _ = run_loss_p(m, b, 0.0, seed=s)
        o["terms"]["total"].backward()
        opt.step()
    assert all(float(br.weight.abs().sum()) > 0.0 for br in m.route_br), "every branch moved off its zero init"
    o, _ = run_loss_p(m, b, 0.0)
    gd = torch.autograd.grad(o["terms"]["ade"], [m.route_head.weight, m.route_head.bias], allow_unused=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in gd)


# ------------------------------------------------------------------------------------------------ admissibility
def test_admissibility_assert_no_situation_classifier_input_stays_green_with_modes(world):
    b, J = routed(world)
    m = bm(J)
    set_branches(m)
    r = M.assert_no_situation_classifier_input(m, b)
    assert set(r["read"]) <= set(M.INFERENCE_INPUT_KEYS) | M.NAV_RB_KEYS
    assert M.scan_source_for_forbidden(Path(MODEL_P).read_text(encoding="utf-8")) == []
    m.eval()
    with torch.no_grad():
        x0 = m.forward_infer(b)["traj60"]
        x1 = m.forward_infer({k: v for k, v in b.items() if not k.startswith(("lab", "gt", "route"))})["traj60"]
    assert torch.equal(x0, x1), "no label, no route_y, no GT reaches the plan"


# ------------------------------------------------------------------------------------------------ G-DVB
def test_gdvb_passes_an_honoured_modes_argv_and_names_the_levers(world):
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_ok")
    assert aux_only(dvb(parts)) == [], dvb(parts)
    m = parts[2]
    lv = m.built_levers()
    assert lv["aux_route"] == "modes" and lv["w_aux_route"] == 0.5 and m.cfg.aux_route == "modes" and m.route_head is not None and m.route_emb is None and len(m.route_br) == K
    assert aux_only(dvb(gate_parts(world, [*RBD(world), *ARGS_C], "am_cond_ok"))) == [], "the cond arm still passes"


def test_gdvb_refuses_declared_modes_built_without_or_with_the_wrong_modules(world):
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_nobr")
    parts[2].route_br = None
    assert any("declared modes" in m and "route_br False" in m for m in dvb(parts)), dvb(parts)
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_cond_built")
    n = parts[2].cfg.n_q * parts[2].cfg.d
    parts[2].route_br, parts[2].route_emb = None, torch.nn.Linear(3, n, bias=False)
    assert any("declared modes" in m for m in dvb(parts)), dvb(parts)
    parts = gate_parts(world, [*RBD(world), *ARGS_C], "am_modes_built")
    parts[2].route_emb, parts[2].route_br = None, torch.nn.ModuleList([torch.nn.Linear(n, 12, bias=False) for _ in range(K)])
    assert any("declared cond" in m for m in dvb(parts)), dvb(parts)
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_both")
    parts[2].route_emb = torch.nn.Linear(3, n, bias=False)
    assert any("route_emb True" in m for m in dvb(parts)), "modes with a route_emb is refused"


def test_gdvb_refuses_a_nonzero_start_wrong_shape_bias_a_wrong_count_and_a_missing_optimiser_entry(world):
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_nz")
    set_branches(parts[2])
    assert any("not zero at the start" in m for m in dvb(parts))
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_shape")
    n, do = parts[2].cfg.n_q * parts[2].cfg.d, int(parts[2].head_o[2].out_features)
    parts[2].route_br[1] = torch.nn.Linear(n, do + 1, bias=False)
    torch.nn.init.zeros_(parts[2].route_br[1].weight)
    assert any("route_br.1 is" in m for m in dvb(parts))
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_bias")
    parts[2].route_br[0] = torch.nn.Linear(n, do, bias=True)
    torch.nn.init.zeros_(parts[2].route_br[0].weight)
    assert any("route_br.0 is" in m and "without a bias" in m for m in dvb(parts))
    parts = gate_parts(world, [*RBD(world), *ARGS_M], "am_two")
    parts[2].route_br = torch.nn.ModuleList(list(parts[2].route_br)[:2])
    assert any("2 branches built" in m for m in dvb(parts))
    parts = list(gate_parts(world, [*RBD(world), *ARGS_M], "am_opt"))
    m = parts[2]
    parts[3] = torch.optim.AdamW([p for nm, p in m.named_parameters() if p.requires_grad and nm != "route_br.2.weight"], lr=1e-3)
    assert any("route_br.2.weight is in 0 optimiser groups" in x for x in dvb(tuple(parts))), dvb(tuple(parts))
    parts = gate_parts(world, [*RBD(world), "--aux-route", "modes", "--detach-upper"], "am_w0")
    assert any("declared with --w-aux-route 0.0" in x and "does nothing" in x for x in dvb(parts))


def test_gdvb_a_flag_without_a_gdvb_entry_is_still_refused_and_modes_does_not_weaken_it(world):
    assert TR.FLAG_CHECKS["aux_route"] == "lever:aux_route" and TR.FLAG_CHECKS["w_aux_route"] == "lever:w_aux_route"
    ps = TR.build_parser()
    for a in ps._actions:
        if a.dest in ("aux_route", "w_aux_route"):
            assert a.dest in TR.FLAG_CHECKS


# ------------------------------------------------------------------------------------------------ the probe (G-LIVE)
def test_probe_modes_reads_every_branch_the_permutation_the_identity_and_the_exact_zero_controls(world):
    b, J = routed(world)
    for zero_head in (True, False):
        m = bm(J, detach_upper=True)
        if zero_head:
            with torch.no_grad():
                m.head_o[2].weight.zero_(); m.head_o[2].bias.zero_()
        r = TR.aux_route_probe(m, b, m.cfg)
        assert r["pass"] and r["head_grad_route"] > 0.0 and r["head_grad_total"] > 0.0 and r["emb_grad_total"] > 0.0 and r["emb_grad_ade"] > 0.0 and r["plan_move_when_emb_set"] > 0.0
        assert len(r["real_init_emb_each"]) == K and all(x > 0.0 for x in r["real_init_emb_each"]) and all(x > 0.0 for x in r["emb_each"])
        assert r["plan_move_when_p_permuted"] > 0.0
        assert r["control_decoder_grad_on_head"] == 0.0 and r["control_tside_grad_on_head"] == 0.0 and r["control_tside_grad_on_emb"] == 0.0
        assert r["control_repeat_grad_diff"] == 0.0 and r["control_prior_grad"] == 0.0 and r["control_no_own_row_route_term"] == 0.0 and r["control_no_own_row_head_grad"] == 0.0 and r["own_rows_probe"] > 0
        assert r["identity_terms_differing"] == [] and r["identity_outputs_differing"] == []
    assert all(float(br.weight.abs().sum()) == 0.0 for br in m.route_br), "the probe works on a COPY"
    mc = build(J, route=True, detach_upper=True)
    rc = TR.aux_route_probe(mc, b, mc.cfg)
    assert rc["pass"] and rc["plan_move_when_p_permuted"] is None and len(rc["emb_each"]) == 1, "the cond probe is unchanged in kind"


def _mut(old, new, name):
    return mutant(MODEL_P, old, new, name)


def test_probe_refuses_a_model_whose_branch_add_is_deleted(world):
    b, J = routed(world)
    mod = _mut("                    coef = coef + p_mode[:, k:k + 1] * br(hf)\n", "                    pass\n", "m_noadd")
    m = bm(J, mod=mod, detach_upper=True)
    with pytest.raises(AssertionError, match="route_emb|branch"):
        TR.aux_route_probe(m, b, m.cfg)


def test_probe_refuses_a_model_whose_p_is_not_detached(world):
    b, J = routed(world)
    mod = _mut("p_mode = torch.softmax(self._route_logits.float(), -1).detach()", "p_mode = torch.softmax(self._route_logits.float(), -1)", "m_nodet")
    m = bm(J, mod=mod, detach_upper=True)
    with pytest.raises(AssertionError, match="detached|route_head"):
        TR.aux_route_probe(m, b, m.cfg)


def test_probe_refuses_a_model_whose_p_is_permuted_across_rows_through_the_stash(world):
    b, J = routed(world)
    mod = _mut("p_mode = torch.softmax(self._route_logits.float(), -1).detach()", "p_mode = torch.softmax(self._route_logits.float(), -1).detach().roll(1, 0)", "m_roll")
    m = bm(J, mod=mod, detach_upper=True)
    set_branches(m)
    m.eval()
    with torch.no_grad():
        traj, p30, cap = plan_and_parts(m, b)
    assert float((traj - formula_plan(m, p30, cap)).abs().max()) > 1e-6, "the formula test turns RED on a row-permuted selection"


def test_probe_refuses_a_model_with_a_hard_argmax_selection(world):
    b, J = routed(world)
    mod = _mut("p_mode = torch.softmax(self._route_logits.float(), -1).detach()", "p_mode = torch.nn.functional.one_hot(self._route_logits.argmax(-1), 3).float()", "m_hard")
    m = bm(J, mod=mod, detach_upper=True)
    set_branches(m)
    m.eval()
    with torch.no_grad():
        traj, p30, cap = plan_and_parts(m, b)
    assert float((traj - formula_plan(m, p30, cap)).abs().max()) > 1e-6, "the formula test turns RED on a hard selection"


# ------------------------------------------------------------------------------------------------ end to end
def test_end_to_end_n1dm_arm_probe_levers_group_class_record_checkpoint_and_the_checker_reads_the_real_selfcheck(world):
    out = Path(world["root"]) / "n1dm_on"
    argv = _argv(world, "n1dm_on", ["--steps", "4", *N1(world), *RBD(world), *ARGS_M])
    assert TR.main(argv) == 0
    sc, sm = _read(out, "selfcheck.json"), _read(out, "summary.json")
    pr = sc["aux_route_probe"]
    assert sc["declared_vs_built"] == "PASS" and pr["pass"] and pr["plan_move_when_p_permuted"] > 0.0 and pr["control_decoder_grad_on_head"] == 0.0
    assert sc["levers"]["aux_route"] == "modes" and sc["levers"]["w_aux_route"] == 0.5
    assert sc["live"]["groups_grad"]["ROUTE"] > 0.0 and sc["live"]["pass"] and "route" in sc["live"]["terms"]
    assert not any(n.startswith(("route_head.", "route_br.")) for n in sc["live"]["dead_params"])
    assert sm["aux_route"] == "modes" and sm["route_rows"] > 0 and sm["route_ok"] > 0 and sum(sm["route_class_counts"].values()) == sm["route_ok"]
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert ck["cfg"]["aux_route"] == "modes" and not any(k.startswith("route_emb.") for k in ck["model"])
    assert all(float(ck["model"][f"route_br.{k}.weight"].abs().sum()) > 0.0 for k in range(K)), "every branch trained"
    mm = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior.synthetic(0), *M.cons_stats(__import__("tanitad.data.v8f_mini_data", fromlist=["x"]).load_join(Path(world["root"]) / "join_train.npz")))
    mm.load_state_dict(ck["model"])
    assert mm.built_levers()["aux_route"] == "modes" and len(mm.route_br) == K
    import strat1_check as SC
    assert SC.m_gate_problems("N1DM", argv, sc) == [], SC.m_gate_problems("N1DM", argv, sc)
    init = {k: v for k, v in ck["model"].items() if not k.startswith("route_")}
    assert [x for x in SC.moved_identity(init, ck["model"], "N1DM") if "route" in x] == [], SC.moved_identity(init, ck["model"], "N1DM")


def test_init_ckpt_fresh_keys_and_resume_refusal(world):
    base = Path(world["root"]) / "am_base_init"
    assert TR.main(_argv(world, "am_base_init", ["--steps", "2", *N1(world)])) == 0
    out = Path(world["root"]) / "am_from_init"
    assert TR.main(_argv(world, "am_from_init", ["--steps", "1", "--selfcheck-only", "--init-ckpt", str(base / "ckpt_last.pt"), *N1(world), *RBD(world), *ARGS_M])) == 0
    ic = _read(out, "selfcheck.json")["info"]["init_ckpt"]
    assert ic["applied"] and {"route_head.weight", "route_head.bias", "route_br.0.weight", "route_br.1.weight", "route_br.2.weight"} <= set(ic["declared_new_keys"]) and set(ic.get("fresh_keys", [])) <= set(ic["declared_new_keys"])
    argv = _argv(world, "am_resume", ["--steps", "3", *N1(world), *RBD(world), *ARGS_M])
    assert TR.main(argv) == 0
    p = Path(world["root"]) / "am_resume" / "ckpt_last.pt"
    ck = torch.load(p, weights_only=False)
    ck["cfg"]["aux_route"] = "cond"
    torch.save(ck, p)
    with pytest.raises(SystemExit, match="ZZRESUME-REFUSEDZZ the checkpoint's aux_route is cond / 0.5 but the argv declares modes / 0.5"):
        TR.main(argv)


def test_r2_cpu_fp32_one_thread_is_deterministic_across_two_runs_of_the_whole_training_step(world):
    b, J = routed(world)
    outs = []
    for _ in range(2):
        m = bm(J, detach_upper=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        for s in range(2):
            opt.zero_grad(set_to_none=True)
            o, _ = run_loss_p(m, b, 0.5, seed=s)
            o["terms"]["total"].backward()
            opt.step()
        outs.append((o["terms"]["total"].detach().clone(), {k: v.clone() for k, v in m.state_dict().items()}))
    assert torch.equal(outs[0][0], outs[1][0]) and all(torch.equal(v, outs[1][1][k]) for k, v in outs[0][1].items())


# ------------------------------------------------------------------------------------------------ the loaders
@pytest.mark.parametrize("name", LOADERS)
def test_roundtrip_each_loader_builds_the_branches_from_the_checkpoint_cfg_strict_and_equal(name, world):
    b, J = routed(world)
    m = bm(J, detach_upper=True)
    set_branches(m)
    ck = _ckpt_of(m)
    exprs = loader_cfg_exprs(name)
    assert exprs, f"{name} has no V8FMiniConfig(**...) build line"
    for ex in exprs:
        cfgd = eval(ex, {"ck": ck, "cfgd": ck["cfg"], "cfg": ck["cfg"]})
        mm = M.V8FMini(M.V8FMiniConfig(**cfgd), M.P10Prior.synthetic(0), *M.cons_stats(J))
        mm.load_state_dict(ck["model"])                                                         # STRICT: a branch-less rebuild raises (unexpected key route_br.0.weight)
        assert mm.route_br is not None and mm.route_emb is None and mm.built_levers()["aux_route"] == "modes"
        assert list(mm.state_dict()) == list(m.state_dict()) and all(torch.equal(v, m.state_dict()[k]) for k, v in mm.state_dict().items())
        m.eval(); mm.eval()
        with torch.no_grad():
            o1, o2 = m.forward_infer(b), mm.forward_infer(b)
        assert all(torch.equal(o1[k], o2[k]) for k in o1 if torch.is_tensor(o1[k])), "outputs"


def test_roundtrip_strat1_check_geval_accepts_a_modes_checkpoint_and_refuses_a_cfg_that_drops_it(world, tmp_path):
    import strat1_check as SC
    b, J = routed(world)
    m = bm(J)
    set_branches(m)
    p = tmp_path / "ck.pt"
    torch.save(_ckpt_of(m), p)
    bad, rec = SC.geval(str(p), selfcheck_levers=m.built_levers())
    assert bad == [] and rec["built_levers"]["aux_route"] == "modes" and rec.get("regression_arm_aux_route_refused") is True, (bad, rec)
    bad2, _ = SC.geval(str(p), selfcheck_levers={k: v for k, v in m.built_levers().items() if k not in ("aux_route", "w_aux_route")})
    assert any("aux_route" in x for x in bad2), bad2
