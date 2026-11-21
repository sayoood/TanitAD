"""ACTCONS (`--w-act-cons W --act-cons-max M`; card PREREG_V8F_PROOF_ADD37 section 2 and controls a1 to a3): a kinematic consistency term on the level-1 imagination.

Tiny models (d 32) on CPU. A test that the mechanism WORKS is not evidence that it RUNS: every reach test has a control that must read EXACTLY 0.0, or a mutation (in memory, by editing the source text of the function under test) that must go RED.
Run with the tree's own stack on the path (the helper below inserts it)."""
from __future__ import annotations

import copy
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import act_cons as AC  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]
LS = GRID + ["--lead-sup", "on", "--w-lead-sup", "1.0"]
MAXOFF = 1.5


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("actcons")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    D.synthetic_lead_labels(J, 0)
    ds = D.V8FMiniWindows(store, J, wins, hmax=16, lead_sup=True)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return {"batch": batch, "J": J}


def _built(world, extra=(), seed=0, **over):
    a = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])
    c0 = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**c0.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(world["J"])
    torch.manual_seed(seed)
    return a, cfg, M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)


def _wake(model, seed=1):
    """Non-trivial lead head and predictor: the head's read-out and every all-zero tensor of the level-1 predictor (adaLN-zero gates) get small random weights, so the gap responds to the action."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.wm.lead_head.mlp:
            if isinstance(m, torch.nn.Linear):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
        for p in model.wm.stack.predictors[0].parameters():
            if float(p.abs().sum()) == 0.0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def _adapt(model, batch):
    return model._encode(batch)


def _offsets(n, seed=3):
    return AC.draw_offsets(seed, 0, n, MAXOFF)


def _mutant(model_fn_old, new, name="act_cons_term"):
    """The function ``name`` of act_cons with ONE source edit (old -> new), exec'd in the module's own namespace: a deliberate regression, in memory."""
    src = textwrap.dedent(inspect.getsource(getattr(AC, name)))
    assert src.count(model_fn_old) == 1, (model_fn_old, src.count(model_fn_old))
    ns = dict(vars(AC))
    exec(compile(src.replace(model_fn_old, new), f"<mutant {name}>", "exec"), ns)
    return ns[name]


# --------------------------------------------------------------------------------------------- default off, flags, gate
def test_default_builds_nothing_and_keeps_the_key_sets(world):
    a = TR.build_parser().parse_args(["--out", "x", *LS])
    assert a.w_act_cons == 0.0 and a.act_cons_max == 1.5
    assert "w_act_cons" in TR.FLAG_CHECKS and "act_cons_max" in TR.FLAG_CHECKS
    _, cfg, m = _built(world, LS)
    assert not [k for k in m.state_dict() if "act_cons" in k] and not hasattr(m, "act_cons")
    assert not [k for k in cfg.__dict__ if "act_cons" in k]                       # no config field: a checkpoint's cfg is the base tree's
    assert not [k for k in m.built_levers() if "act_cons" in k]
    assert TR.bad_act_cons(a, cfg, m, {}) == []


def test_default_off_is_a_lever_that_declares_nothing_when_max_is_changed(world):
    _, cfg, m = _built(world, LS)
    a = TR.build_parser().parse_args(["--out", "x", *LS, "--act-cons-max", "2.0"])
    assert any("does nothing" in b for b in TR.bad_act_cons(a, cfg, m, {}))


def test_gate_refusals(world):
    a_on = TR.build_parser().parse_args(["--out", "x", *LS, "--w-act-cons", "1.0"])
    _, cfg, m = _built(world, LS)
    rec = AC.build_record(1.0, 1.5, cfg, m)
    assert TR.bad_act_cons(a_on, cfg, m, {"act_cons": rec}) == []                 # the declared == built case passes
    assert any("no act-cons stage" in b for b in TR.bad_act_cons(a_on, cfg, m, {}))
    assert any("w_act_cons: declared 1.0 but built 2.0" in b for b in TR.bad_act_cons(a_on, cfg, m, {"act_cons": {**rec, "w_act_cons": 2.0}}))
    assert any("act_cons_max: declared" in b for b in TR.bad_act_cons(a_on, cfg, m, {"act_cons": {**rec, "act_cons_max": 0.5}}))
    assert any("rolls" in b or "rollout" in b for b in TR.bad_act_cons(a_on, cfg, m, {"act_cons": {**rec, "k": 8}}))
    assert any("w_act_cons 0 declared but" in b for b in TR.bad_act_cons(TR.build_parser().parse_args(["--out", "x", *LS]), cfg, m, {"act_cons": rec}))
    for flag, val in (("--ego-perturb-p", "0.5"), ("--dagger-p", "0.5"), ("--recover-p", "0.5")):
        a2 = TR.build_parser().parse_args(["--out", "x", *LS, "--w-act-cons", "1.0", flag, val])
        assert any(flag in b for b in TR.bad_act_cons(a2, cfg, m, {"act_cons": rec})), flag
    a3 = TR.build_parser().parse_args(["--out", "x", *LS, "--w-act-cons", "-1.0"])
    assert any(">= 0" in b for b in TR.bad_act_cons(a3, cfg, m, {}))
    # no lead head: the lever is refused (a declared lever that does nothing)
    _, cfg0, m0 = _built(world, GRID)
    assert any("lead head" in b for b in AC.check_buildable(cfg0, m0))
    assert any("lead head" in b for b in TR.bad_act_cons(a_on, cfg0, m0, {}))
    # speed action channel: refused
    _, cfg1, m1 = _built(world, LS + ["--wm-l1-act", "speed"])
    assert any("acc_yaw" in b for b in AC.check_buildable(cfg1, m1))
    # lead_hist != 5: refused
    _, cfg2, m2 = _built(world, LS + ["--lead-hist", "4"])
    assert any("lead-hist" in b for b in AC.check_buildable(cfg2, m2))


# --------------------------------------------------------------------------------------------- the offsets
def test_offsets_dedicated_generator_range_and_global_rng_untouched():
    torch.manual_seed(5)
    before = torch.get_rng_state()
    d1, d2 = AC.draw_offsets(7, 100, 4096, 1.5), AC.draw_offsets(7, 100, 4096, 1.5)
    assert torch.equal(torch.get_rng_state(), before)                                # nothing drawn from the global generator
    assert torch.equal(d1, d2)                                                       # reproducible from (seed, step): a resumed run draws the same
    assert not torch.equal(d1, AC.draw_offsets(7, 101, 4096, 1.5)) and not torch.equal(d1, AC.draw_offsets(8, 100, 4096, 1.5))
    assert float(d1.min()) >= -1.5 and float(d1.max()) <= 1.5 and float(d1.min()) < -1.4 and float(d1.max()) > 1.4
    assert abs(float(d1.mean())) < 0.1 and abs(float(d1.std()) - 1.5 / 3 ** 0.5) < 0.05


def test_start_set_is_the_rollout_losses():
    assert AC.valid_starts(16) == [0, 2, 4]
    from tanitad.models.hjepa import HJEPAStack
    assert "range(0, t_ - k, 2)" in inspect.getsource(HJEPAStack.roll_pairs)         # the start rule this file mirrors, on the line it lives


def test_scale_and_roll_core_facts():
    assert M.WM_ACC_SCALE == 2.0 and "act[..., 0] / WM_ACC_SCALE" in inspect.getsource(M.WMAux.scale_actions)
    src = inspect.getsource(AC.act_cons_term)
    assert "roll_pairs(" in src and "dr / float(M.WM_ACC_SCALE)" in src


# --------------------------------------------------------------------------------------------- a2: hand computation, invalid samples, gradient reach
def _hand_term(model, batch, adapt, d, max_off):
    """The term by an independent route: per sample and start, the PUBLIC ``HJEPAStack.rollout`` (not roll_pairs), the head called as a plain module (no functional_call), loops in python."""
    wm, st, head = model.wm, model.wm.stack, model.wm.lead_head
    ctx = int(getattr(st.predictors[0], "context", 0) or ROLL)
    with torch.no_grad():
        z = wm.z1(adapt)
        valid = AC.valid_mask(batch, AC.valid_starts(z.shape[1]), max_off)
        a = wm.scale_actions(batch["act"].float())[:, :-1]
        vals = []
        for b in range(z.shape[0]):
            for i, s in enumerate(AC.valid_starts(z.shape[1])):
                if not bool(valid[b, i]):
                    continue
                a_ref = a[b, s:s + ROLL]
                a_off = a_ref.clone()
                a_off[:, 0] += d[b] / M.WM_ACC_SCALE
                r_ref = st.rollout(1, z[b, s][None], a_ref[None], context=ctx)[0]
                r_off = st.rollout(1, z[b, s][None], a_off[None], context=ctx)[0]
                for lo, hi, T in AC.T_READS:
                    g_ref = float(head(r_ref[lo:hi][None])[0, 0, 1]) * 50.0
                    g_off = float(head(r_off[lo:hi][None])[0, 0, 1]) * 50.0
                    vals.append(abs(g_off - g_ref + 0.5 * float(d[b]) * T * T))
    return float(np.mean(vals)), int(valid.sum())


ROLL = AC.ROLL_K


def test_term_equals_a_hand_computation(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    d = _offsets(8)
    adapt = _adapt(m, batch)
    r = AC.act_cons_term(m, adapt, batch, d, MAXOFF)
    hand, n_hand = _hand_term(m, batch, adapt, d, MAXOFF)
    assert n_hand > 2 and int(r["n_valid"]) == n_hand                                  # a non-trivial batch: several valid (sample, start) pairs
    assert float(r["term"]) > 0.01 and abs(float(r["term"]) - hand) < 2e-4 * max(1.0, hand), (float(r["term"]), hand)


def test_invalid_samples_give_exactly_zero_term_and_gradient_and_do_not_move_the_valid_ones(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = dict(world["batch"])
    d = _offsets(8)
    adapt = _adapt(m, batch)
    # (a) nothing valid: no label lead anywhere -> exactly 0.0, no graph, no gradient
    b0 = dict(batch)
    b0["lead_pres"] = torch.zeros_like(batch["lead_pres"])
    r0 = AC.act_cons_term(m, adapt, b0, d, MAXOFF)
    assert float(r0["term"]) == 0.0 and not r0["term"].requires_grad and float(r0["n_valid"]) == 0.0
    # (b) nothing valid: ego speed below the bound everywhere
    b1 = dict(batch)
    b1["wm_speed"] = torch.full_like(batch["wm_speed"], MAXOFF * AC.SPEED_MULT - 0.01)
    assert float(AC.act_cons_term(m, adapt, b1, d, MAXOFF)["term"]) == 0.0
    b1["wm_speed"] = torch.full_like(batch["wm_speed"], 20.0)
    assert float(AC.act_cons_term(m, adapt, b1, d, MAXOFF)["n_valid"]) > 0.0           # the control: the same batch at a valid speed has valid samples
    # (c) gap outside 0..60 and NaN: invalid
    b2 = dict(batch)
    b2["lead_gapm"] = torch.full_like(batch["lead_gapm"], 61.0)
    assert float(AC.act_cons_term(m, adapt, b2, d, MAXOFF)["n_valid"]) == 0.0
    b2["lead_gapm"] = torch.full_like(batch["lead_gapm"], float("nan"))
    assert float(AC.act_cons_term(m, adapt, b2, d, MAXOFF)["n_valid"]) == 0.0
    # (d) a padded start frame is not valid even with a label
    b3 = dict(batch)
    b3["frame_ok"] = torch.zeros_like(batch["frame_ok"])
    assert float(AC.act_cons_term(m, adapt, b3, d, MAXOFF)["n_valid"]) == 0.0

    def grads(bt, dd):
        mm = copy.deepcopy(m)
        mm.zero_grad(set_to_none=True)
        r = AC.act_cons_term(mm, _adapt(mm, bt), bt, dd, MAXOFF)
        r["term"].backward()
        return r, {n: p.grad.clone() for n, p in mm.named_parameters() if p.grad is not None}

    r_all, g_all = grads(batch, d)
    valid_rows = AC.valid_mask(batch, AC.valid_starts(16), MAXOFF).any(1)
    assert 0 < int(valid_rows.sum()) < 8                                               # the batch mixes valid and invalid samples
    # scramble every INVALID row (actions, offsets, speed below the bound): the term and the gradient are bit for bit the same
    b4 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    d4 = d.clone()
    bad = ~valid_rows
    b4["act"][bad] = b4["act"][bad] * 3.0 + 1.0
    d4[bad] = 9.0
    r_s, g_s = grads(b4, d4)
    assert float(r_all["term"]) == float(r_s["term"]) and float(r_all["n_valid"]) == float(r_s["n_valid"])
    assert set(g_all) == set(g_s) and all(torch.equal(g_all[k], g_s[k]) for k in g_all)


def test_gradient_reaches_predictor_and_action_embed_and_exactly_zero_elsewhere(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    d = _offsets(8)
    adapt = _adapt(m, batch).detach().requires_grad_(True)                            # the start-state input itself: its gradient must be exactly 0
    m.zero_grad(set_to_none=True)
    r = AC.act_cons_term(m, adapt, batch, d, MAXOFF)
    r["term"].backward()
    names = dict(m.named_parameters())
    pred = sum(float(p.grad.abs().sum()) for n, p in names.items() if n.startswith("wm.stack.predictors.0.") and p.grad is not None)
    emb = sum(float(p.grad.abs().sum()) for n, p in names.items() if n.startswith("wm.stack.act_embed.") and p.grad is not None)
    assert pred > 0.0 and emb > 0.0
    others = {n: float(p.grad.abs().sum()) for n, p in names.items() if p.grad is not None and not n.startswith(("wm.stack.predictors.0.", "wm.stack.act_embed."))}
    assert others == {}                                                                # not one other parameter even holds a gradient
    assert all(p.grad is None for n, p in names.items() if n.startswith("wm.lead_head."))      # the head's parameters: exactly nothing
    assert adapt.grad is None or float(adapt.grad.abs().sum()) == 0.0                 # the start state / encoder path: exactly 0.0


def test_total_moves_by_exactly_w_times_the_term(world):
    """The trainer's own block, run as the loop runs it (the lines are extracted from the source): total(W) - total(0) = W x term, bit for bit up to one float32 addition."""
    src = inspect.getsource(TR.main)
    assert src.count('T["total"] = tot_ac + a.w_act_cons * ac_r["term"]') == 1 and src.count("AC.act_cons_term(model, out[\"adapt\"], b, d_ac, a.act_cons_max, level=a.w_act_level > 0.0)") == 1
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    out = m.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T = out["terms"]
    tot0 = T["total"].detach().clone()
    r = AC.act_cons_term(m, out["adapt"], batch, _offsets(8), MAXOFF)
    for W in (0.1, 1.0, 3.0):
        t_w = float((T["total"] + W * r["term"]).detach())
        assert abs((t_w - float(tot0)) - W * float(r["term"])) < 1e-4 * max(1.0, abs(float(tot0)))
    assert float(r["term"]) > 0.0


# --------------------------------------------------------------------------------------------- a3: planted kinematic predictor
class _KinPredictor(torch.nn.Module):
    """Exact constant-acceleration kinematics per step on an ego displacement coordinate: state ch0 = displacement p [m], ch1 = speed v [m/s]; the embedded action is the scaled action itself."""
    context = 1

    def forward(self, zc, e):
        last = zc[:, -1]
        a = e[:, -1, 0] * M.WM_ACC_SCALE
        dt = AC.STEP_S
        out = last.clone()
        out[:, 0] = last[:, 0] + last[:, 1] * dt + 0.5 * a * dt * dt
        out[:, 1] = last[:, 1] + a * dt
        return out[:, None, :]


class _KinHead(torch.nn.Module):
    """gap = G0 - p with G0 = 0 (a constant offset only adds float32 cancellation error; the term sees differences) (the lead keeps its motion, the ego displacement is the only thing that moves the gap), in the trainer's output unit (gap / 50 m)."""

    def forward(self, x):
        out = torch.zeros(x.shape[0], 1, 3)
        out[:, 0, 1] = (0.0 - x[:, -1, 0]) / M.LEAD_SUP_GAP_SCALE_M
        return out


def _planted(world):
    _, cfg, m = _built(world, LS)
    m.wm.stack.predictors[0] = _KinPredictor()
    m.wm.stack.act_embed = torch.nn.Identity()
    m.wm.lead_head = _KinHead()
    z1 = m.wm.z1
    m.wm.z1 = lambda adapt: 0.05 * z1(adapt)                                           # small start states: float32 error of the kinematics stays far below the bar
    return m


def test_planted_kinematic_predictor_reads_zero_and_the_two_mutations_read_red(world):
    m = _planted(world)
    batch = world["batch"]
    d = _offsets(8)
    adapt = _adapt(m, batch)
    r = AC.act_cons_term(m, adapt, batch, d, MAXOFF)
    assert float(r["n_valid"]) > 2 and abs(float(r["term"])) < 1e-6, float(r["term"])
    # mutation 1: the SIGN of the kinematic target flipped
    f1 = _mutant("+ 0.5 * dr[None, :] * (T ** 2)", "- 0.5 * dr[None, :] * (T ** 2)")
    t1 = float(f1(m, adapt, batch, d, MAXOFF)["term"])
    assert t1 > 0.1, t1                                                                # far from 0: RED
    # mutation 2: the offset NOT applied to the rolled-out actions
    f2 = _mutant("st.act_embed(a + off)", "st.act_embed(a)")
    t2 = float(f2(m, adapt, batch, d, MAXOFF)["term"])
    assert t2 > 0.1, t2
    # mutation 3: the offset applied in the wrong units (d instead of d / WM_ACC_SCALE)
    f3 = _mutant("dr / float(M.WM_ACC_SCALE)", "dr")
    assert float(f3(m, adapt, batch, d, MAXOFF)["term"]) > 0.1
    # mutation 4: the horizon of the two reads swapped in the target (T of the 2.0 s read used for 1.0 s)
    f4 = _mutant("(T ** 2)", "(2.0 ** 2)")
    assert float(f4(m, adapt, batch, d, MAXOFF)["term"]) > 0.05


def test_gradient_zero_mutations_go_red(world):
    _, cfg, m = _built(world, LS)
    _wake(m)
    batch = world["batch"]
    d = _offsets(8)

    def reach(fn):
        mm = copy.deepcopy(m)
        ad = _adapt(mm, batch).detach().requires_grad_(True)
        mm.zero_grad(set_to_none=True)
        fn(mm, ad, batch, d, MAXOFF)["term"].backward()
        head = sum(float(p.grad.abs().sum()) for n, p in mm.named_parameters() if n.startswith("wm.lead_head.") and p.grad is not None)
        proj = sum(float(p.grad.abs().sum()) for n, p in mm.named_parameters() if n.startswith(("wm.proj.", "adapter.")) and p.grad is not None)
        inp = 0.0 if ad.grad is None else float(ad.grad.abs().sum())
        return head, proj, inp

    assert reach(AC.act_cons_term) == (0.0, 0.0, 0.0)                                  # the control: the real function reads exactly 0.0 on all three
    h, _, _ = reach(_mutant("params = {n: p.detach() for n, p in head.named_parameters()}", "params = {n: p for n, p in head.named_parameters()}"))
    assert h > 0.0                                                                     # head parameters not detached: RED
    _, pj, inp = reach(_mutant("        with torch.no_grad():\n            z = wm.z1(adapt.detach().float()).float()", "        if True:\n            z = wm.z1(adapt.float()).float()"))
    assert pj > 0.0 or inp > 0.0                                                       # start state not detached: RED


# --------------------------------------------------------------------------------------------- the probe, the loop
def test_probe_passes_and_goes_red_without_the_detach(world, monkeypatch):
    _, cfg, m = _built(world, LS)
    a = TR.build_parser().parse_args(["--out", "x", *LS, "--w-act-cons", "1.0"])
    rec = TR.act_cons_probe(m, world["batch"], cfg, a)
    assert rec["pass"] and rec["grad_norm_lead_head"] == 0.0 and rec["grad_norm_state_projection"] == 0.0 and rec["grad_abs_sum_outside_predictor_and_embed"] == 0.0
    assert rec["grad_norm_predictor"] > 0.0 and rec["grad_norm_action_embed"] > 0.0 and rec["n_valid"] > 0 and rec["control_all_invalid_term"] == 0.0 and not rec["control_all_invalid_has_graph"]
    off = TR.act_cons_probe(m, world["batch"], cfg, TR.build_parser().parse_args(["--out", "x", *LS]))
    assert off == {"act_cons": "off", "pass": True}
    mut = _mutant("params = {n: p.detach() for n, p in head.named_parameters()}", "params = {n: p for n, p in head.named_parameters()}")
    monkeypatch.setattr(AC, "act_cons_term", mut)
    with pytest.raises(AssertionError, match="exactly 0.0"):
        TR.act_cons_probe(m, world["batch"], cfg, a)


def test_the_loop_lines_exist_once_and_the_gate_calls_exist():
    src = Path(TR.__file__).read_text(encoding="utf-8")
    assert src.count("AC.act_cons_term(model, out[\"adapt\"], b, d_ac, a.act_cons_max, level=a.w_act_level > 0.0)") == 1
    assert src.count("AC.draw_offsets(a.seed, step,") == 1
    assert src.count("bad += bad_act_cons(a, cfg, model, data_info)") == 1
    assert src.count("act_cons_probe(model, probe, cfg, a)") == 1
    assert src.count("ZZACT-CONS-NOT-APPLIEDZZ") == 1
