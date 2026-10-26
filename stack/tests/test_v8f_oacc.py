"""PREREG_V8F_OACC: does training O to match the human's ACCELERATION, not only its positions, make the deployed v8F output dominate its kinematic prior?  X = B (M2b) + `--w-acc <rule>` (ONE lever).

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): a lever whose library function is correct and whose CALLER never reaches it. So the lever is proven hop by hop FROM ARGV,
and every proof has a deliberate-regression arm that turns it RED (here as a mutated copy of the model module; in `raw/2026-10-25-oacc/mutate_oacc.py` as mutated repo files, output beside it):

  argv --w-acc -> make_config (cfg.w_acc) -> declared_vs_built (G-DVB: FLAG_CHECKS entry + the BUILT config + the BUILT lever) -> acc_probe (the BUILT loss's total moves by exactly w_acc x the term, O's head
  receives the term's gradient) -> V8FMini.loss -> `total = total + cfg.w_acc * T["acc"] if cfg.w_acc > 0.0 else total` -> .backward()

  (A) the term       `accel_term`: the second difference on the 0.2 s grid, origin knot included, of O's plan minus the GT's, squared, mean over the GT-valid cells -- against literals authored here.
  (B) w_acc = 0 IS B the default reproduces the saved pre-change module (`raw/2026-10-25-oacc/pre_oacc_v8f_mini.py.ref`) bit for bit: the built model, the loss terms, every gradient, a 200-step CPU trace.
  (C) reachability  the loss moves by w_acc x the term; O's head gets the term's gradient; the controls read EXACTLY 0.0; DELETE the line and the probe (and so the gate) goes RED; a real `TR.main` run.
  (D) G-DVB         the flag has its entry; declared 2.0 / built 0.0 is refused; a negative or non-finite weight is refused.
  (E) the rule      `stack/scripts/v8f_mini_eval_oacc.py wacc`: w_acc = 0.5 L_traj / L_acc on the trainer's own functions, against literals; the model route on a synthetic checkpoint; the banked cross-check.
  (F) the HX fit    `... hx`: HBLEND's own fit on X's checkpoint.
  (G) the reader    `code/v8f_oacc_read.py`: the committed bars, the outcome table verbatim, the conditional seed 1, every refusal.
  (H) the queue + chain  the stages print the command they run, one lever; every existing label is byte-identical; the chain's predicate, ordering, dry run.
Everything that needs the research package (the saved reference module, the reader, the queue, the chain) is SKIPPED, not failing, in a checkout that has only `stack/`."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
import v8f_mini_eval_oacc as OE  # noqa: E402

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
RAWO = PKG / "raw" / "2026-10-25-oacc"
REF_P = RAWO / "pre_oacc_v8f_mini.py.ref"
CODE = PKG / "code"
READ_P = CODE / "v8f_oacc_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_oacc.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_OACC.md"
MODEL_P = REPO / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"
HBLEND_NPZ = PKG / "raw" / "2026-10-25-hblend" / "fit_windows_M2b_s0.npz"
HERE_T = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _at(text, needle):
    """``text.index(needle)`` where a MISSING anchor FAILS BY ASSERTION: a mutated file loses its anchor, and that must be a red assertion naming the anchor, never a ValueError."""
    assert needle in text, f"the anchor is missing: {needle!r}"
    return text.index(needle)


def _no_crash(fn, *a, **k):
    """Call ``fn``; an exception out of the instrument under test is a FAILED assertion (it crashed on an input it must judge), never an error. SystemExit (a refusal) passes through."""
    try:
        return fn(*a, **k)
    except Exception as e:                                                             # noqa: BLE001
        err = f"{getattr(fn, '__name__', fn)} raised {type(e).__name__}: {str(e)[:300]}"
    pytest.fail(err)                                     # outside the except: no chained context, so the first E line of the report is this Failed, not the instrument's own traceback


def _refusal_code(fn, *a, **k):
    """The exit code of the refusal ``fn`` must raise; an acceptance or a crash is a FAILED assertion."""
    try:
        fn(*a, **k)
    except SystemExit as e:
        return e.code
    except Exception as e:                                                             # noqa: BLE001
        pytest.fail(f"{getattr(fn, '__name__', fn)} raised {type(e).__name__} instead of refusing: {str(e)[:300]}")
    pytest.fail(f"{getattr(fn, '__name__', fn)} accepted an input it must refuse")


def _load_ref(name, path):
    """A module loaded under a private name from a file whose suffix is not .py (the saved reference, a mutant). Registered in sys.modules BEFORE it executes (a dataclass with `from __future__ import
    annotations` looks its module up)."""
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        loader.exec_module(m)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return m


REF_M = _load_ref("v8f_mini_pre_oacc", REF_P) if REF_P.exists() else None
needs_ref = pytest.mark.skipif(REF_M is None, reason="raw/2026-10-25-oacc (the saved pre-change module) is not in this checkout")
_P = _load("test_v8f_pgt_for_oacc", HERE_T / "test_v8f_pgt.py")                  # the batch, the spy and the synthetic world the other v8F-mini tests use
world, tiny_model, make_batch = _P.world, _P.tiny_model, _P.make_batch
_T = _P._T                                                                       # test_v8f_mini
COMMON = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5",
          "--arm", "M2b"]
LINE = 'total = total + cfg.w_acc * T["acc"] if cfg.w_acc > 0.0 else total'         # the ONE line the mutation deletes


def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])


def _J(world):
    return D.load_join(Path(world["root"]) / "join_train.npz")


def _mk(mod, J, seed=3, **over):
    """A tiny model built from module `mod` (the real one, the saved reference, or a mutant), weights seeded; `over` are config fields."""
    cfg = mod.V8FMiniConfig(**{**_T.TINY, "wm": "off", **over})
    mu, sd = mod.cons_stats(J)
    torch.manual_seed(seed)
    return mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)


def _wake(model, seed=11):
    """The residual head is zero-init: at build O's output is P10 whatever it reads. Give it weights (seeded) so a probe can see a change."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w = model.head_o[2].weight
        w.copy_(torch.randn(w.shape, generator=g) * 0.05)
    return model


def _batch(world, n=16, seed=1, hmax=None):
    J = _J(world)
    torch.manual_seed(0)
    model, cfg = tiny_model(J, wm="off")
    b, _, _ = make_batch(world, n=n, seed=seed, hmax=cfg.hmax if hmax is None else hmax)
    B = b["feats"].shape[0]
    i = torch.arange(B)
    b["lab_t"]["has"] = (i % 3 != 0)
    b["lab_s"]["has"] = (i % 4 != 1)
    g = torch.Generator().manual_seed(123)
    b["lab_s"]["rc_ok"] = torch.rand(B, 3, generator=g) > 0.35
    return b


@pytest.fixture(autouse=True)
def _release_tmp_path(tmp_path):
    """Remove a test's OWN tmp_path when it finishes (a disk-limited box): the module-scoped worlds live under tmp_path_factory, which this never touches."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def _mutant_model(tmp_path, edits, name="v8f_mini_mut_oacc"):
    """A copy of the model module with each (anchor, replacement) applied exactly once. The anchor must occur exactly once in the ORIGINAL source."""
    s = MODEL_P.read_text(encoding="utf-8")
    for a, b in edits:
        assert s.count(a) == 1, (a, s.count(a))
        s = s.replace(a, b)
    p = Path(tmp_path) / f"{name}.py.mut"
    p.write_text(s, encoding="utf-8")
    return _load_ref(name, p)


# =============================================================================================================================================
# (A) THE TERM
# =============================================================================================================================================
def _ref_accel(traj, gt, ok):
    """The PREREG's formula, written out with plain loops from the text (never from the code under test): for each valid window, the second difference at the 29 interior knots of [0, plan_1 ... plan_30]
    (0.2 s grid, the origin knot included), plan minus GT, squared L2 norm over (x, y); the mean over the valid windows x 29 centres."""
    tot, n = 0.0, 0
    for b in range(traj.shape[0]):
        if not ok[b]:
            continue
        P = np.vstack([np.zeros((1, 2)), traj[b].astype(np.float64)])
        Y = np.vstack([np.zeros((1, 2)), gt[b].astype(np.float64)])
        for j in range(1, 30):
            d = (P[j - 1] - 2 * P[j] + P[j + 1]) - (Y[j - 1] - 2 * Y[j] + Y[j + 1])
            tot += float(d @ d)
            n += 1
    return tot / max(n, 1)


def test_accel_term_is_the_preregs_formula_against_a_loop_written_from_the_text():
    rng = np.random.default_rng(0)
    traj, gt = rng.normal(size=(5, 30, 2)).astype(np.float32), rng.normal(size=(5, 30, 2)).astype(np.float32)
    ok = np.array([1, 1, 0, 1, 0], np.float32)
    got = float(M.accel_term(torch.as_tensor(traj), torch.as_tensor(gt), torch.as_tensor(ok)))
    assert got == pytest.approx(_ref_accel(traj, gt, ok), rel=1e-5), "the mean is over the GT-valid windows x 29 centres"
    assert M.second_diff_02(torch.as_tensor(traj)).shape == (5, 29, 2), "31 knots (origin included) -> 29 interior centres"
    assert float(M.accel_term(torch.as_tensor(traj), torch.as_tensor(gt), torch.zeros(5))) == 0.0, "no valid window: 0.0, not nan"


def test_accel_term_literal_identities():
    """Each identity is arithmetic done by hand, so a wrong derivative order, a missing origin knot or a dropped GT term cannot pass."""
    gt = torch.zeros(1, 30, 2) + torch.randn(1, 30, 2, generator=torch.Generator().manual_seed(1))
    one = torch.ones(1)
    assert float(M.accel_term(gt, gt, one)) == 0.0, "a plan equal to the GT has the GT's acceleration: exactly 0.0"
    t = torch.arange(1, 31, dtype=torch.float32)[None, :, None]
    lin = gt + t * torch.tensor([0.3, -0.7]) + torch.tensor([2.0, 1.0])
    # a linear offset adds nothing to a second difference EXCEPT at the origin knot: plan(0) = 0, so centre 1 sees  -(offset(1)) + offset(2)... compute by hand below
    off = lambda k: k * torch.tensor([0.3, -0.7]) + torch.tensor([2.0, 1.0])      # noqa: E731  offset at step k = 1..30
    centre1 = (0.0 - 2 * off(1) + off(2))                                           # origin knot has NO offset: the plan's 0 vs 0 of the GT
    want = float((centre1 ** 2).sum()) / 29
    assert float(M.accel_term(lin, gt, one)) == pytest.approx(want, rel=1e-5), "the origin knot is part of the plan: dropping it would read 0.0 here"
    quad = gt + 0.1 * t ** 2 * torch.tensor([1.0, 0.0])                             # offset c k^2: second difference 2c at every centre (centre 1 too: 0 - 2c + 4c = 2c)
    assert float(M.accel_term(quad, gt, one)) == pytest.approx((0.2 ** 2), rel=1e-4), "a quadratic offset c k^2 has second difference 2c at all 29 centres: (2 * 0.1)^2 = 0.04"
    step = gt.clone()
    step[:, :, 0] += 1.0                                                            # a constant offset of all 30 knots, origin fixed at 0: ONLY centre 1 sees it (0 - 2a + a = -a)
    assert float(M.accel_term(step, gt, one)) == pytest.approx(1.0 / 29, rel=1e-5)
    two = torch.cat([step, gt]), torch.cat([gt, gt])
    assert float(M.accel_term(two[0], torch.cat([gt, gt]), torch.tensor([1.0, 1.0]))) == pytest.approx(0.5 / 29, rel=1e-5), "the mean is over windows too"
    assert float(M.accel_term(two[0], torch.cat([gt, gt]), torch.tensor([1.0, 0.0]))) == pytest.approx(1.0 / 29, rel=1e-5), "an invalid window is out of the mean AND out of the denominator"


def test_traj_terms_is_the_losses_own_spelling_of_ade_and_fde(world):
    J = _J(world)
    m = _wake(_mk(M, J))
    b = _batch(world)
    out = m.loss(b, p_gt=0.4, gen=torch.Generator().manual_seed(2))
    ade, fde = M.traj_terms(out["traj30"], b["gt"], b["gt_ok"].float())
    assert torch.equal(ade, out["terms"]["ade"]) and torch.equal(fde, out["terms"]["fde"])
    d = np.sqrt(((out["traj30"].detach().numpy().astype(np.float64) - b["gt"].numpy().astype(np.float64)) ** 2).sum(-1) + 1e-9)
    ok = b["gt_ok"].numpy().astype(bool)
    assert float(ade) == pytest.approx(d.mean(1)[ok].mean(), rel=1e-5) and float(fde) == pytest.approx(d[:, -1][ok].mean(), rel=1e-5)


# =============================================================================================================================================
# (B) w_acc = 0 IS B, BIT FOR BIT
# =============================================================================================================================================
def _old_pair(world, n=12, seed=2):
    J = _J(world)
    cfg = M.V8FMiniConfig(**{**_T.TINY, "wm": "off"})
    kw = {k: v for k, v in cfg.__dict__.items() if k != "w_acc"}
    mu, sd = M.cons_stats(J)
    torch.manual_seed(3)
    new = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    torch.manual_seed(3)
    old = REF_M.V8FMini(REF_M.V8FMiniConfig(**kw), REF_M.P10Prior.synthetic(0), mu, sd)
    b, _, _ = make_batch(world, n=n, seed=seed, hmax=cfg.hmax)
    B = b["feats"].shape[0]
    b["lab_t"]["has"] = (torch.arange(B) % 3 != 0)
    return new, old, b


@needs_ref
def test_w_acc_zero_builds_the_saved_pre_change_model_and_loss_and_gradients_bit_for_bit(world):
    new, old, b = _old_pair(world)
    sn, so = new.state_dict(), old.state_dict()
    assert sn.keys() == so.keys() and all(torch.equal(sn[k], so[k]) for k in sn), "identical built model"
    ln, lo = new.built_levers(), old.built_levers()
    assert set(ln) - set(lo) == {"w_acc"} and set(lo) <= set(ln) and {k: v for k, v in ln.items() if k in lo} == lo and ln["w_acc"] == 0.0     # additions only
    assert {f.name for f in dataclasses.fields(M.V8FMiniConfig)} - {f.name for f in dataclasses.fields(REF_M.V8FMiniConfig)} == {"w_acc"}
    _wake(new), _wake(old)
    for p in (0.0, 0.3, 1.0):
        torch.manual_seed(5)
        rn = new.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        gn = torch.autograd.grad(rn["terms"]["total"], [q for q in new.parameters() if q.requires_grad], allow_unused=True)
        torch.manual_seed(5)
        ro = old.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        go = torch.autograd.grad(ro["terms"]["total"], [q for q in old.parameters() if q.requires_grad], allow_unused=True)
        tn, to = rn["terms"], ro["terms"]
        assert set(tn) - set(to) == {"acc"} and all(torch.equal(tn[k], to[k]) for k in to), "every pre-existing loss term is bit-identical; the only new key is acc"
        assert len(gn) == len(go) and all((x is None) == (y is None) and (x is None or torch.equal(x, y)) for x, y in zip(gn, go)), "every gradient is bit-identical"
        assert float(tn["acc"]) > 0.0, "the term is LOGGED at w_acc 0 (the reader and the probe read it) and is not in the total"


def _trace(model, batches, steps=200, p0=1.0, p1=0.2, seed=17, lr=2e-3):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    model.train()
    out = []
    for k in range(steps):
        torch.manual_seed(seed + k)
        res = model.loss(batches[k % len(batches)], p_gt=p0 + (p1 - p0) * k / steps, gen=torch.Generator().manual_seed(1000 + k))
        opt.zero_grad(set_to_none=True)
        res["terms"]["total"].backward()
        opt.step()
        out.append(float(res["terms"]["total"].detach()))
    return out


def _trace_batches(world, hmax):
    bs = []
    for s in (2, 3, 4):
        b, _, _ = make_batch(world, n=8, seed=s, hmax=hmax)
        b["lab_t"]["has"] = (torch.arange(8) % 4 != 1)
        bs.append(b)
    return bs


@needs_ref
def test_a_200_step_cpu_loss_trace_at_w_acc_zero_is_the_saved_modules_exactly_and_at_a_positive_weight_it_is_not(world):
    new, old, _ = _old_pair(world)
    bs = _trace_batches(world, new.cfg.hmax)
    tn, to = _trace(_wake(new), bs), _trace(_wake(old), bs)
    assert tn == to, "the 200 per-step totals are bit-identical"
    new2, _, _ = _old_pair(world)
    new2.cfg.w_acc = 5.0
    t5 = _trace(_wake(new2), bs)
    assert t5 != to and t5[0] > to[0], "the control: a positive weight changes the trace (the probe can see a change)"


# =============================================================================================================================================
# (C) REACHABILITY: THE LOSS MOVES BY w_acc x THE TERM AND O's HEAD GETS ITS GRADIENT; DELETE THE LINE AND THE GATE GOES RED
# =============================================================================================================================================
def test_the_loss_total_moves_by_exactly_w_acc_times_the_logged_term(world):
    J, b = _J(world), _batch(world)
    m0, m3 = _wake(_mk(M, J)), _wake(_mk(M, J, w_acc=3.0))
    torch.manual_seed(5)
    r0 = m0.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(4))
    torch.manual_seed(5)
    r3 = m3.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(4))
    assert torch.equal(r0["terms"]["acc"], r3["terms"]["acc"])
    assert float(r3["terms"]["total"] - r0["terms"]["total"]) == pytest.approx(3.0 * float(r3["terms"]["acc"]), rel=1e-4)
    out = r3
    assert float(r3["terms"]["acc"]) == pytest.approx(float(M.accel_term(out["traj30"], b["gt"], b["gt_ok"].float())), rel=1e-6)


def test_the_probe_passes_at_a_positive_weight_and_at_zero_and_its_controls_read_exactly_zero(world):
    J, b = _J(world), _batch(world)
    for w in (0.7, 0.0):
        m = _mk(M, J, w_acc=w)
        r = TR.acc_probe(m, b, m.cfg)
        assert r["pass"] is True and r["acc"] > 0.0 and r["head_o_grad_diff"] > 0.0
        assert r["control_repeat_total_diff"] == 0.0 and r["control_repeat_grad_diff"] == 0.0 and r["control_prior_grad_diff"] == 0.0 and r["control_wm_grad_diff"] is None
        assert r["contribution"] == pytest.approx(r["w_x_acc"], rel=1e-3)
        assert r["w_probe"] == (w if w > 0 else 1.0)
        if w == 0.0:
            assert r["declared_zero_total_reproduces_exactly"] is True
    m = _mk(M, J, w_acc=0.7, wm="b", hmax=16)
    assert TR.acc_probe(m, _batch(world, hmax=16), m.cfg)["control_wm_grad_diff"] == 0.0, "the world-model auxiliary does not feed the plan: its gradient does not move with the weight"


def test_deleting_the_total_line_turns_the_probe_red(world, tmp_path):
    J, b = _J(world), _batch(world)
    mut = _mutant_model(tmp_path, [(LINE, "pass")])
    m = _mk(mut, J, w_acc=0.7)
    with pytest.raises(AssertionError, match="does not reach the loss with its weight"):
        TR.acc_probe(m, b, m.cfg)
    mut2 = _mutant_model(tmp_path, [(LINE, 'total = total + T["acc"] if cfg.w_acc > 0.0 else total')], name="v8f_mini_mut_oacc2")        # the weight is ignored
    m2 = _mk(mut2, J, w_acc=0.7)
    with pytest.raises(AssertionError, match="does not reach the loss with its weight"):
        TR.acc_probe(m2, b, m2.cfg)
    mut3 = _mutant_model(tmp_path, [(LINE, 'total = total + cfg.w_acc * T["acc"].detach() if cfg.w_acc > 0.0 else total')], name="v8f_mini_mut_oacc3")      # the term is in the total but cuts no gradient
    m3 = _mk(mut3, J, w_acc=0.7)
    with pytest.raises(AssertionError, match="receives no gradient"):
        TR.acc_probe(m3, b, m3.cfg)


def test_a_model_that_does_not_log_the_term_fails_g_live(world, tmp_path):
    J, b = _J(world), _batch(world)
    m = _mk(M, J, w_acc=0.7)
    assert "acc" in TR.live_check(m, b, m.cfg)["terms"]
    mut = _mutant_model(tmp_path, [('        T["acc"] = accel_term(out["traj30"], gt, gok)', '        T["acc_other"] = accel_term(out["traj30"], gt, gok)'), (LINE, "pass")], name="v8f_mini_mut_oacc4")
    mm = _mk(mut, J, w_acc=0.7)
    with pytest.raises(AssertionError, match="G-LIVE failed"):
        TR.live_check(mm, b, mm.cfg)


def _expect_assert(fn, match):
    """``fn`` must refuse with an AssertionError whose message matches ``match`` (the clause's OWN message: a neighbouring clause that happens to refuse the same state is not this clause). A crash or an
    acceptance FAILS the test; it is never an error."""
    try:
        fn()
    except AssertionError as e:
        assert re.search(match, str(e)), f"refused, but with another clause's message: {str(e)[:300]}"
        return
    except Exception as e:                                                             # noqa: BLE001
        pytest.fail(f"raised {type(e).__name__} instead of refusing: {str(e)[:300]}")
    pytest.fail(f"not refused: expected a refusal matching {match!r}")


def _wrap_loss(monkeypatch, post):
    """Wrap the model's loss: ``post(self, out)`` edits the returned terms in place. This is the STATE a probe must refuse, fed through the BUILT loss (acc_probe deep-copies the model; the class is patched)."""
    orig = M.V8FMini.loss

    def loss(self, *a, **k):
        out = orig(self, *a, **k)
        post(self, out)
        return out
    monkeypatch.setattr(M.V8FMini, "loss", loss)


def _neutral(x):
    """``x`` in the graph with VALUE exactly 0: it moves a gradient and no loss value."""
    return x - x.detach()


def _post_zero_term(self, out):
    out["terms"]["acc"] = out["terms"]["acc"] * 0.0


def _post_negative_term(self, out):
    out["terms"]["acc"] = out["terms"]["acc"] * -1.0


def _post_nan_term(self, out):
    out["terms"]["acc"] = out["terms"]["acc"] * float("nan")


def _post_weight_dependent_term(self, out):
    out["terms"]["acc"] = out["terms"]["acc"] * (1.0 + 0.01 * self.cfg.w_acc)


def _post_total_noise():
    calls = [0]

    def post(self, out):                                           # the two w = 0 calls (the repeat control) differ by 1e-4 in the total
        if self.cfg.w_acc == 0.0:
            calls[0] += 1
            out["terms"]["total"] = out["terms"]["total"] + 1e-4 * calls[0]
    return post


def _post_grad_noise():
    calls = [0]

    def post(self, out):                                           # the two w = 0 calls differ in a GRADIENT only (the value is untouched)
        if self.cfg.w_acc == 0.0:
            calls[0] += 1
            x = (1e-3 * calls[0]) * sum(p.sum() for p in self.head_o.parameters())
            out["terms"]["total"] = out["terms"]["total"] + _neutral(x)
    return post


def _post_prior_gradient(self, out):                               # the frozen prior's gradient moves with the weight (value-neutral, so no earlier clause sees it)
    ps = list(self.prior.parameters())
    for p in ps:
        p.requires_grad_(True)
    out["terms"]["total"] = out["terms"]["total"] + _neutral(self.cfg.w_acc * sum(p.sum() for p in ps))


def _post_wm_gradient(self, out):                                  # the world-model auxiliary's gradient moves with the weight
    out["terms"]["total"] = out["terms"]["total"] + _neutral(self.cfg.w_acc * sum(p.sum() for p in self.wm.parameters()))


_PROBE_CLAUSES = {
    "an_exactly_zero_term": (_post_zero_term, "finite and > 0 required", {}),
    "a_negative_term": (_post_negative_term, "finite and > 0 required", {}),
    "a_non_finite_term": (_post_nan_term, "loss term acc is not finite", {}),
    "a_logged_term_that_depends_on_the_weight": (_post_weight_dependent_term, "depends on the weight", {}),
    "a_non_deterministic_total": (_post_total_noise, "the probe is not deterministic", {}),
    "a_non_deterministic_gradient": (_post_grad_noise, "the probe is not deterministic", {}),
    "a_frozen_prior_whose_gradient_moves": (_post_prior_gradient, "frozen P10 prior's gradient moved with the weight", {}),
    "a_world_model_whose_gradient_moves": (_post_wm_gradient, "world-model auxiliary's gradient moved with the weight", {"wm": "b", "hmax": 16}),
}


@pytest.mark.parametrize("clause", list(_PROBE_CLAUSES))
def test_acc_probe_refuses_each_violating_state_by_its_own_clause(world, monkeypatch, clause):
    """One state per clause of ``acc_probe``, fed through the BUILT loss. The control first: the unpatched model PASSES (the probe is not simply refusing everything); then the state; the refusal's message
    is that clause's own."""
    post, match, mk = _PROBE_CLAUSES[clause]
    post = post() if post in (_post_total_noise, _post_grad_noise) else post
    J = _J(world)
    m = _mk(M, J, w_acc=0.7, **mk)
    b = _batch(world, hmax=m.cfg.hmax)
    assert _no_crash(TR.acc_probe, m, b, m.cfg)["pass"] is True, "the control: this model, unpatched, passes"
    _wrap_loss(monkeypatch, post)
    _expect_assert(lambda: TR.acc_probe(m, b, m.cfg), match)


def test_acc_probe_refuses_a_weight_the_float32_total_cannot_see(world):
    J = _J(world)
    m = _mk(M, J, w_acc=1e-9)
    _expect_assert(lambda: TR.acc_probe(m, _batch(world), m.cfg), "below float32 visibility")
    m2 = _mk(M, J, w_acc=0.7)
    assert _no_crash(TR.acc_probe, m2, _batch(world), m2.cfg)["pass"] is True, "a visible weight on the same batch passes"


def test_acc_probe_reads_o_heads_gradient_and_the_total_through_the_built_loss(world, monkeypatch):
    """Two of the probe's own clauses, through the loss: the total must move by w x the term, and O's head must receive the term's gradient."""
    J = _J(world)
    m = _mk(M, J, w_acc=0.7)
    b = _batch(world)
    _wrap_loss(monkeypatch, lambda self, out: out["terms"].__setitem__("total", out["terms"]["total"] - self.cfg.w_acc * out["terms"]["acc"]))      # the line that adds the term, deleted
    _expect_assert(lambda: TR.acc_probe(m, b, m.cfg), "does not reach the loss with its weight")
    monkeypatch.undo()
    m2 = _mk(M, J, w_acc=0.7)
    _wrap_loss(monkeypatch, lambda self, out: out["terms"].__setitem__("total", out["terms"]["total"] - self.cfg.w_acc * out["terms"]["acc"] + self.cfg.w_acc * out["terms"]["acc"].detach()))
    _expect_assert(lambda: TR.acc_probe(m2, b, m2.cfg), "receives no gradient")


def _gdvb_parts(world, argv, **over):
    """The launch gate's inputs for a tiny model: the args, the BUILT config / model / optimiser, and the data facts the gate compares (as test_v8f_flat builds them)."""
    a = _args(argv)
    cfg = TR.make_config(a)
    keep = {k: v for k, v in _T.TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(_J(world))
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    ws = cfg.window_start_effective
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None, "window_start": ws, "window_t_min": ws, "ds_window_start": ws}
    return a, cfg, model, opt, info


@pytest.mark.parametrize("argv", [[], ["--w-acc", "42.9311"], ["--w-acc", "0.3"]])
def test_g_dvb_passes_what_was_built_for_the_acc_weight(world, argv):
    a, cfg, model, opt, info = _gdvb_parts(world, argv)
    assert _no_crash(TR.declared_vs_built, a, cfg, model, opt, info) == []
    assert model.built_levers()["w_acc"] == a.w_acc == cfg.w_acc


def test_g_dvb_refuses_a_declared_weight_the_model_did_not_build_by_both_of_its_clauses(world):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--w-acc", "2.0"], w_acc=0.0)
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any("lever w_acc: declared 2.0 but built 0.0" in m for m in bad), bad
    assert any("cfg.w_acc: declared 2.0 but built 0.0" in m for m in bad), bad
    a, cfg, model, opt, info = _gdvb_parts(world, [], w_acc=2.0)
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any("lever w_acc: declared 0.0 but built 2.0" in m for m in bad) and any("cfg.w_acc: declared 0.0 but built 2.0" in m for m in bad), bad


@pytest.mark.parametrize("bad_w", [-0.5, float("nan"), float("inf"), float("-inf")])
def test_g_dvb_refuses_a_negative_or_non_finite_weight_even_when_declared_and_built_agree(world, bad_w):
    """The config refuses such a value at construction, so the gate's own clause is reached only by a model whose config was edited after the build: declared == built, and the weight is still refused."""
    a, cfg, model, opt, info = _gdvb_parts(world, ["--w-acc", "2.0"], w_acc=2.0)
    a.w_acc = bad_w
    model.cfg.w_acc = bad_w
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any(f"w_acc {bad_w!r} is not a finite number >= 0" in m for m in bad), bad
    ok = _gdvb_parts(world, ["--w-acc", "2.0"], w_acc=2.0)
    assert _no_crash(TR.declared_vs_built, *ok) == [], "the control: 2.0 declared and built is clean"


def test_g_dvb_refuses_a_flag_without_an_entry_and_an_acc_flag_whose_entry_is_missing(world, monkeypatch):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--w-acc", "0.3"])
    assert TR.FLAG_CHECKS["w_acc"] == "cfg:w_acc" and _no_crash(TR.declared_vs_built, a, cfg, model, opt, info) == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "w_acc"})
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any("flags without a declared-vs-built entry" in m and "w_acc" in m for m in bad), bad


def test_g_dvb_refuses_a_model_whose_built_lever_echoes_zero(world, tmp_path):
    """The built lever must be what the model DOES: a module whose built_levers() reports 0.0 whatever the config says is refused when 2.0 is declared."""
    mut = _mutant_model(tmp_path, [('"w_acc": cfg.w_acc,', '"w_acc": 0.0,')], name="v8f_mini_mut_oacc5")
    a, cfg, model, opt, info = _gdvb_parts(world, ["--w-acc", "2.0"])
    mm = mut.V8FMini(mut.V8FMiniConfig(**cfg.__dict__), mut.P10Prior.synthetic(0), *M.cons_stats(_J(world)))
    assert mm.cfg.w_acc == 2.0 and mm.built_levers()["w_acc"] == 0.0, "the echo: the config says 2.0, the lever says 0.0"
    assert _no_crash(TR.declared_vs_built, a, cfg, model, opt, info) == [], "the control: the real module passes"
    bad = _no_crash(TR.declared_vs_built, a, mm.cfg, mm, opt, info)
    assert any("lever w_acc: declared 2.0 but built 0.0" in m for m in bad), bad


def test_a_refusing_acc_probe_stops_the_trainer_before_it_writes_anything_a_stub(tmp_path, monkeypatch):
    """The probe's refusal must be a SystemExit carrying the ZZDVB-REFUSEDZZ marker the supervisor and the launch gate look for, and nothing may be written after it. The control: the same argv with the real
    probe runs to a PASS."""
    argv = ["--out", str(tmp_path / "ok"), *COMMON, "--steps", "2", "--selfcheck-only", "--w-acc", "0.3"]
    assert TR.main(argv) == 0
    sc_ok = json.loads((tmp_path / "ok" / "selfcheck.json").read_text(encoding="utf-8"))
    assert "acc_probe" in sc_ok, "selfcheck.json records the acc probe it ran"
    assert sc_ok["acc_probe"]["pass"] is True
    shutil.rmtree(tmp_path / "ok", ignore_errors=True)
    calls = []

    def refuse(model, batch, cfg, *a, **k):
        calls.append(cfg.w_acc)
        raise AssertionError("acc probe: the stub refuses")
    monkeypatch.setattr(TR, "acc_probe", refuse)
    bad = ["--out", str(tmp_path / "bad"), *COMMON, "--steps", "2", "--w-acc", "0.3"]
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ declared-vs-built: acc probe: the stub refuses"):
        TR.main(bad)
    assert calls == [0.3], "the trainer called the probe once, with the built config"
    for f in ("selfcheck.json", "argv.json", "ckpt_last.pt", "summary.json", "train_log.jsonl"):
        assert not (tmp_path / "bad" / f).exists(), f"a refused launch writes no {f}"


_RUN_VIOLATIONS = {
    "the_terms_line_deleted": lambda self, out: out["terms"].__setitem__("total", out["terms"]["total"] - self.cfg.w_acc * out["terms"]["acc"]),
    "a_gradient_noisy_probe": None,
}


@pytest.mark.parametrize("name", list(_RUN_VIOLATIONS))
def test_a_real_violation_in_the_built_loss_stops_the_trainer_at_the_gate_and_the_run_does_not_proceed(tmp_path, monkeypatch, name):
    """The probe's refusals, through the REAL `TR.main` on the synthetic world (no stub): the gate raises SystemExit('ZZDVB-REFUSEDZZ ...') and neither a selfcheck, an argv record, a checkpoint, a log nor a summary exists."""
    post = _post_grad_noise() if name == "a_gradient_noisy_probe" else _RUN_VIOLATIONS[name]
    argv = ["--out", str(tmp_path / "ok"), *COMMON, "--steps", "2", "--selfcheck-only", "--w-acc", "0.3"]
    assert TR.main(argv) == 0, "the control: unpatched, the gate passes"
    shutil.rmtree(tmp_path / "ok", ignore_errors=True)
    _wrap_loss(monkeypatch, post)
    out = tmp_path / "bad"
    try:
        TR.main(["--out", str(out), *COMMON, "--steps", "2", "--w-acc", "0.3"])
    except SystemExit as e:
        assert str(e.code).startswith("ZZDVB-REFUSEDZZ declared-vs-built: ") and "acc probe" in str(e.code), e.code
    except Exception as e:                                                             # noqa: BLE001
        pytest.fail(f"the gate crashed ({type(e).__name__}: {str(e)[:200]}) instead of refusing")
    else:
        pytest.fail("the trainer ran to the end on a loss the probe must refuse")
    for f in ("selfcheck.json", "argv.json", "ckpt_last.pt", "summary.json", "train_log.jsonl"):
        assert not (out / f).exists(), f"a refused launch writes no {f}"


class _FakeLoraEnc(torch.nn.Module):
    """The smallest encoder `live_check_lora`'s contract needs: a LoRA pair (B zero at init, so dL/dA is EXACTLY 0.0 at step 0), a frozen backbone weight (the control: no gradient) and the three lora_* methods."""

    def __init__(self, d_feat, r=2):
        super().__init__()
        self.backbone_w = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        g = torch.Generator().manual_seed(4)
        self.lora_A = torch.nn.Parameter(torch.randn(1, r, generator=g))
        self.lora_B = torch.nn.Parameter(torch.zeros(r, d_feat))

    def delta(self, prefix):
        return (torch.ones(*prefix, 1) @ self.lora_A) @ self.lora_B

    def lora_parameters(self):
        return [self.lora_A, self.lora_B]

    def lora_state_dict(self):
        return {"A": self.lora_A.detach().clone(), "B": self.lora_B.detach().clone()}

    def load_lora_state_dict(self, sd):
        with torch.no_grad():
            self.lora_A.copy_(sd["A"])
            self.lora_B.copy_(sd["B"])


def _lora_gate(world, monkeypatch, post=None, **mk):
    """Run `live_check_lora` (the LoRA run's G-LIVE) on a tiny model and the fake encoder. The real DINOv3 path needs the cache builder, which is not in every checkout; the gate's own logic does not."""
    import tanitad.models.v8f_lora_encoder as L
    J = _J(world)
    m = _mk(M, J, w_acc=0.7, **mk)
    b = _batch(world, hmax=m.cfg.hmax)
    enc = _FakeLoraEnc(b["feats"].shape[-1])
    base = b["feats"].clone()

    def cached_grad_step(enc_, frames, fmap, run_loss, chunk=None, device=None):
        out = run_loss(base + enc_.delta(base.shape[:-1]))
        out["terms"]["total"].backward()
        return out
    monkeypatch.setattr(L, "cached_grad_step", cached_grad_step)
    if post is not None:
        _wrap_loss(monkeypatch, post)
    batch = dict(b, frames_u8=torch.zeros(1), frame_map=torch.zeros(1))
    return lambda: TR.live_check_lora(m, enc, batch, m.cfg, _args(), torch.device("cpu"), 1)


def test_live_check_lora_passes_with_the_acc_term_logged_and_refuses_a_model_that_does_not_log_it(world, monkeypatch):
    res = _no_crash(_lora_gate(world, monkeypatch))
    assert res["pass"] is True and "acc" in res["terms"] and res["missing_terms"] == [] and res["control_prior_grad_sum"] == 0.0 and res["lora_A_grad_step0"] == 0.0 and res["lora_B_grad_step0"] > 0.0, res
    monkeypatch.undo()
    gate = _lora_gate(world, monkeypatch, post=lambda self, out: out["terms"].pop("acc"))
    _expect_assert(gate, r'G-LIVE \(lora\) failed.*"missing_terms": \["acc"\]')


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two real `TR.main` runs on the synthetic world (X with --w-acc 0.3, B without). The fixture NEVER raises: a refused or crashed launch is recorded (rc / exc) and the TESTS assert on it, so a gate that
    refuses the lever is a FAILED assertion naming the refusal, not an error in a fixture."""
    base = tmp_path_factory.mktemp("oacc_runs")
    r = {}
    for tag, extra in (("x", ["--w-acc", "0.3"]), ("b", [])):
        out = base / tag
        rec = {"out": out, "root": out / "_synthetic", "rc": None, "exc": None}
        try:
            rec["rc"] = TR.main(["--out", str(out), *COMMON, "--steps", "4", *extra])
        except SystemExit as e:
            rec["exc"] = f"SystemExit: {e.code}"
        except Exception as e:                                                         # noqa: BLE001
            rec["exc"] = f"{type(e).__name__}: {str(e)[:300]}"
        jl = lambda name: (json.loads((out / name).read_text(encoding="utf-8")) if (out / name).exists() else None)      # noqa: E731
        rec.update(ck=(torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False) if (out / "ckpt_last.pt").exists() else None), sc=jl("selfcheck.json"), sm=jl("summary.json"),
                   argv=(jl("argv.json") or {}).get("argv"),
                   log=([json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] if (out / "train_log.jsonl").exists() else []))
        r[tag] = rec
        if tag == "b":
            shutil.rmtree(Path(rec["root"]) / "train" / "clips", ignore_errors=True)
    return r


def test_trainer_argv_reaches_the_config_the_checkpoint_the_records_the_gate_and_the_log(runs):
    x, b = runs["x"], runs["b"]
    assert x["rc"] == 0 and x["exc"] is None, f"the X launch (--w-acc 0.3) did not run: rc {x['rc']}, {x['exc']}"
    assert b["rc"] == 0 and b["exc"] is None, f"the B launch did not run: rc {b['rc']}, {b['exc']}"
    assert x["argv"].count("--w-acc") == 1 and x["argv"][x["argv"].index("--w-acc") + 1] == "0.3", "the flag is NAMED in the argv (a default would otherwise test nothing)"
    assert x["ck"]["cfg"]["w_acc"] == 0.3 and x["sm"]["levers"]["w_acc"] == 0.3 and x["sc"]["levers"]["w_acc"] == 0.3
    assert "acc_probe" in x["sc"], "selfcheck.json carries the acc probe's record"
    pr = x["sc"]["acc_probe"]
    assert x["sc"]["declared_vs_built"] == "PASS" and pr["pass"] is True and pr["w_acc_declared"] == 0.3 and x["sm"]["done"] is True
    assert pr["contribution"] == pytest.approx(0.3 * pr["acc"], rel=1e-3) and pr["head_o_grad_diff"] > 0.0 and pr["control_prior_grad_diff"] == 0.0
    assert "acc" in x["sc"]["live"]["terms"] and x["sc"]["live"]["pass"] is True
    assert all("acc" in rec and math.isfinite(rec["acc"]) and rec["acc"] > 0.0 for rec in x["log"]), "the term is in the training log under its own key at every step"
    assert "--w-acc" not in b["argv"] and b["ck"]["cfg"]["w_acc"] == 0.0 and b["sc"]["levers"]["w_acc"] == 0.0 and b["sc"]["acc_probe"]["pass"] is True
    assert x["sm"]["levers"]["params"] == b["sm"]["levers"]["params"] and x["sm"]["levers"]["n_params_trainable"] == b["sm"]["levers"]["n_params_trainable"], "the lever adds no parameter"
    assert [r["step"] for r in x["log"]] == [r["step"] for r in b["log"]]


def test_the_loss_the_trainer_logs_at_a_positive_weight_includes_the_term_and_at_zero_does_not(runs):
    x, b = runs["x"], runs["b"]
    assert x["rc"] == 0 and b["rc"] == 0 and x["log"] and b["log"], (x["exc"], b["exc"])
    # the same seed, data and schedule: step 1 of both runs is the same forward (the weights are identical at step 0), so total_x - total_b = w x acc exactly at step 1
    r1x, r1b = x["log"][0], b["log"][0]
    assert r1x["step"] == r1b["step"] == 1
    assert r1x["total"] - r1b["total"] == pytest.approx(0.3 * r1x["acc"], rel=1e-4)
    assert r1x["acc"] == pytest.approx(r1b["acc"], rel=1e-6)
    assert x["log"][-1]["total"] != b["log"][-1]["total"], "and the trajectories separate: the term reaches the optimiser"


# =============================================================================================================================================
# (D) G-DVB
# =============================================================================================================================================
def test_the_flag_defaults_to_zero_has_its_g_dvb_entry_and_bad_values_are_refused():
    assert M.V8FMiniConfig().w_acc == 0.0
    a = _args()
    assert a.w_acc == 0.0 and TR.make_config(a).w_acc == 0.0
    c = _args(["--w-acc", "42.9311"])
    assert c.w_acc == 42.9311 and TR.make_config(c).w_acc == 42.9311
    assert TR.FLAG_CHECKS["w_acc"] == "cfg:w_acc"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} <= set(TR.FLAG_CHECKS), "every flag has an entry"
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="w_acc must be a finite number"):
            M.V8FMiniConfig(w_acc=bad)
    d = {k: v for k, v in TR.make_config(a).__dict__.items() if k != "w_acc"}
    assert d == {k: v for k, v in TR.make_config(c).__dict__.items() if k != "w_acc"}, "nothing else moves: the two configs differ in w_acc and in nothing else"


# =============================================================================================================================================
# (E) THE w_acc RULE
# =============================================================================================================================================
def test_rule_terms_against_hand_arithmetic():
    """gt = 0 everywhere, plan = a constant (3, 4) offset on all 30 knots: ade = fde = 5 (sqrt(25 + 1e-9)), L_traj = 5/3 + 0.5 * 5/6 = 2.083333; the origin knot is 0 so ONLY centre 1 sees the offset
    (0 - 2a + a = -a): L_acc = |a|^2 / 29 = 25 / 29."""
    traj = np.zeros((4, 30, 2), np.float32) + np.array([3.0, 4.0], np.float32)
    gt = np.zeros((4, 30, 2), np.float32)
    R = OE.rule_terms(traj, gt, np.ones(4, np.float32))
    assert R["L_ade"] == pytest.approx(5.0, rel=1e-6) and R["L_fde"] == pytest.approx(5.0, rel=1e-6)
    assert R["L_traj"] == pytest.approx(5 / 3 + 0.5 * 5 / 6, rel=1e-6) and R["L_acc"] == pytest.approx(25 / 29, rel=1e-5)
    v = OE.w_acc_from_terms(R["L_traj"], R["L_acc"])
    assert v["w_acc_raw"] == pytest.approx(0.5 * (5 / 3 + 5 / 12) / (25 / 29), rel=1e-5)
    assert v["w_acc"] == float(f"{v['w_acc_raw']:.6g}") and abs(v["w_acc"] * R["L_acc"] / R["L_traj"] - 0.5) < 1e-5
    R2 = OE.rule_terms(traj, gt, np.ones(4, np.float32), w_traj=2.0)
    assert R2["L_traj"] == pytest.approx(2 * R["L_traj"], rel=1e-6), "L_traj carries the trainer's w_traj"


def test_rule_refuses_a_term_it_cannot_divide_by_and_rounds_once():
    for lt, la in ((1.0, 0.0), (0.0, 1.0), (float("nan"), 1.0), (1.0, float("inf")), (-1.0, 1.0), (1.0, -1.0), (float("inf"), 1.0), (1.0, float("nan")), (0.0, 0.0)):
        assert _refusal_code(OE.w_acc_from_terms, lt, la) == 2, (lt, la)
    ok = OE.w_acc_from_terms(2.0, 4.0)
    assert ok["w_acc_raw"] == 0.25 and ok["w_acc"] == 0.25, "the control: 0.5 x 2 / 4"
    assert OE.w_acc_from_terms(2.0, 4.0, frac=1.0)["w_acc"] == 0.5 and OE.FRAC == 0.5
    assert OE.round_sig(42.93114562909151) == 42.9311 and OE.round_sig(0.000123456789) == 0.000123457
    assert float(repr(OE.round_sig(1234.56789))) == OE.round_sig(1234.56789)


@pytest.mark.skipif(not HBLEND_NPZ.exists(), reason="HBLEND's banked fit windows are not in this checkout")
def test_the_offline_route_on_hblends_banked_b_s0_arrays_pins_the_dry_check(tmp_path):
    out, txt = tmp_path / "w.json", tmp_path / "w.txt"
    assert OE.main(["wacc", "--from-npz", str(HBLEND_NPZ), "--out", str(out), "--out-txt", str(txt)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["n_windows"] == 9582 and r["L_ade"] == pytest.approx(2.1679, abs=5e-4), "HBLEND's own log: train ADE6 tau 2.1679"
    assert r["w_acc"] == 42.9311 and txt.read_text() == "42.9311\n" and float(txt.read_text()) == r["w_acc"]
    assert r["alt_readings_not_used"]["w_acc_if_L_traj_were_ade_only"] > r["w_acc"], "the declared L_traj (R1) is the trainer's, not ADE alone"
    assert abs(r["check_w_acc_x_L_acc_over_L_traj"] - 0.5) < 1e-5


@pytest.fixture(scope="module")
def fitworld(tmp_path_factory):
    """A trained-4-steps B checkpoint on the synthetic world, its cache / join / tracks, and a fit_clips.json over the synthetic train clips: the model route of the rule and the HX fit run on THIS."""
    base = tmp_path_factory.mktemp("oacc_fit")
    out = base / "b"
    assert TR.main(["--out", str(out), *COMMON, "--steps", "4"]) == 0
    root = out / "_synthetic"
    clips = json.loads(json.dumps(sorted(p.stem.split(".")[0] for p in (root / "train" / "clips").glob("*.pt"))))
    fc = base / "fit_clips.json"
    fc.write_text(json.dumps({"clips": clips, "k": 1, "sha256_newline_joined": "synthetic"}), encoding="utf-8")
    return {"ckpt": out / "ckpt_last.pt", "root": root, "fc": fc, "base": base, "clips": clips}


def _model_route_args(fw, out, extra=()):
    return ["--ckpt", str(fw["ckpt"]), "--fit-clips", str(fw["fc"]), "--cache-dir", str(fw["root"]), "--join-train", str(fw["root"] / "join_train.npz"), "--tracks", str(fw["root"] / "native_tracks.pt"),
            "--batch", "8", "--threads", "1", "--out", str(out), *extra]


def test_the_model_route_runs_the_checkpoint_on_the_fit_windows_and_agrees_with_the_offline_route_on_the_same_arrays(fitworld, tmp_path):
    fw = fitworld
    assert fw["clips"], "the synthetic train clips"
    out, txt, npz = tmp_path / "w.json", tmp_path / "w.txt", tmp_path / "fit.npz"
    model, cfg, ck = OE.load_ckpt_model(fw["ckpt"])
    clips = json.loads(fw["fc"].read_text())["clips"]
    O = OE.forward_fit_set(model, cfg, ck, clips, str(fw["root"]), str(fw["root"] / "join_train.npz"), str(fw["root"] / "native_tracks.pt"), batch=8, log=lambda *_: None)
    np.savez_compressed(npz, **O)
    assert O["traj30"].shape == (len(O["gt"]), 30, 2) and O["gt_ok"].all()
    assert OE.main(["wacc", *_model_route_args(fw, out, ["--out-txt", str(txt), "--banked-fit-windows", str(npz)])]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["banked_crosscheck"]["pass"] is True and max(r["banked_crosscheck"]["relative_difference"].values()) <= 1e-6 and r["banked_crosscheck"]["same_windows"] is True
    off = tmp_path / "off.json"
    assert OE.main(["wacc", "--from-npz", str(npz), "--out", str(off)]) == 0
    ro = json.loads(off.read_text(encoding="utf-8"))
    assert ro["w_acc"] == r["w_acc"] and ro["L_traj"] == pytest.approx(r["L_traj"], rel=1e-6)
    assert float(txt.read_text()) == r["w_acc"] > 0.0
    # the terms ARE the trainer's: L_traj equals the loss's own trajectory contribution on the same windows, L_acc its logged term
    b = {"feats": None}
    tt = torch.as_tensor
    ade, fde = M.traj_terms(tt(O["traj30"]), tt(O["gt"]), tt(O["gt_ok"].astype(np.float32)))
    assert r["L_traj"] == pytest.approx(cfg.w_traj * (float(ade) / 3.0 + 0.5 * float(fde) / 6.0), rel=1e-6)


def test_the_model_route_refuses_a_disagreeing_bank_a_wrong_sha_and_an_x_checkpoint(fitworld, tmp_path, capsys):
    fw = fitworld
    model, cfg, ck = OE.load_ckpt_model(fw["ckpt"])
    clips = json.loads(fw["fc"].read_text())["clips"]
    O = OE.forward_fit_set(model, cfg, ck, clips, str(fw["root"]), str(fw["root"] / "join_train.npz"), str(fw["root"] / "native_tracks.pt"), batch=8, log=lambda *_: None)
    def refused(args, want):
        capsys.readouterr()
        assert _refusal_code(OE.main, ["wacc", *args]) == 2
        err = capsys.readouterr().err
        assert err.startswith("OACC EVAL REFUSED:") and want in err, (want, err[:300])
    bad = dict(O)
    bad["traj30"] = O["traj30"] * 1.05
    npz = tmp_path / "bad.npz"
    np.savez_compressed(npz, **bad)
    refused(_model_route_args(fw, tmp_path / "w.json", ["--banked-fit-windows", str(npz)]), "the model route disagrees with HBLEND's banked arrays")      # a bank the route does not reproduce to 1e-3: a different fit set or checkpoint
    # the SAME numbers on other windows: the L terms agree to the last digit, the window identity does not -- refused all the same
    for key, shift in (("t", 1), ("ci", 1)):
        same_numbers = dict(O)
        same_numbers[key] = O[key] + shift
        npz2 = tmp_path / f"same_numbers_{key}.npz"
        np.savez_compressed(npz2, **same_numbers)
        refused(_model_route_args(fw, tmp_path / f"w_{key}.json", ["--banked-fit-windows", str(npz2)]), "the model route disagrees with HBLEND's banked arrays")
        rec = json.loads((tmp_path / f"w_{key}.json").read_text(encoding="utf-8"))
        assert max(rec["banked_crosscheck"]["relative_difference"].values()) == 0.0 and rec["banked_crosscheck"]["same_windows"] is False and rec["banked_crosscheck"]["pass"] is False, key
    refused(_model_route_args(fw, tmp_path / "w2.json", ["--expect-sha256", "0" * 64]), "has sha256")
    ck_x = torch.load(fw["ckpt"], map_location="cpu", weights_only=False)
    ck_x["cfg"] = {**ck_x["cfg"], "w_acc": 3.0}
    px = tmp_path / "x.pt"
    torch.save(ck_x, px)
    args = _model_route_args(fw, tmp_path / "w3.json")
    args[args.index("--ckpt") + 1] = str(px)
    refused(args, "the checkpoint was built with w_acc 3.0: the rule is evaluated on B (w_acc 0)")
    # a fit window without a 6 s future is refused (the fit set is not HBLEND's)
    orig = D.collate

    def no_future(items):
        b = orig(items)
        b["gt_ok"] = b["gt_ok"].clone()
        b["gt_ok"][0] = False
        return b
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(D, "collate", no_future)
        refused(_model_route_args(fw, tmp_path / "w4.json"), "a train window of the fit set has no 6 s future")
    finally:
        mp.undo()


def test_the_hx_fit_is_hblends_own_closed_form_on_the_checkpoints_held_outputs(fitworld, tmp_path):
    fw = fitworld
    out, npz = tmp_path / "hx.json", tmp_path / "hx.npz"
    assert _no_crash(OE.main, ["hx", *_model_route_args(fw, out, ["--tag", "M2b-oacc_s0", "--fit-windows-out", str(npz)])]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    w = np.stack([np.asarray(r["w_along_by_step"]), np.asarray(r["w_cross_by_step"])], 1)
    assert w.shape == (30, 2) and ((w >= 0) & (w <= 1)).all() and (np.diff(w, axis=0) >= -1e-12).all(), "clip [0, 1] and isotonic non-decreasing"
    H = OE.load_hblend()
    z = np.load(npz)
    R = H.fit_weights(z["traj30"], z["p30"], z["gt"])
    assert np.array_equal(R["w"], w), "the weights ARE HBLEND's fit_weights on the saved arrays"
    assert r["tag"] == "M2b-oacc_s0" and r["n_windows"] == len(z["gt"]) and r["w_acc_built"] == 0.0
    d = lambda p: np.linalg.norm(p.astype(np.float64) - z["gt"].astype(np.float64), axis=-1).mean()      # noqa: E731
    assert r["in_sample_train_ade6"]["H"] == pytest.approx(d(H.apply_w(z["p30"], z["traj30"], w)), rel=1e-9)
    assert r["in_sample_train_ade6"]["P"] == pytest.approx(d(z["p30"]), rel=1e-9) and r["in_sample_train_ade6"]["tau"] == pytest.approx(d(z["traj30"]), rel=1e-9)
    assert not np.array_equal(z["p30"], z["traj30"]), "the prior P is the checkpoint's FROZEN P10 output, not the model's plan (a fit of the plan on itself has no closed form)"
    assert r["ckpt_sha256"] == OE.sha256_file(fw["ckpt"]) and len(r["ckpt_sha256"]) == 64, "the record names the checkpoint it was fitted on, by sha256"
    # an X checkpoint (built with w_acc 3.0): the record carries the weight the checkpoint was BUILT with and ITS sha256, which the reader checks
    ck_x = torch.load(fw["ckpt"], map_location="cpu", weights_only=False)
    ck_x["cfg"] = {**ck_x["cfg"], "w_acc": 3.0}
    px = tmp_path / "x.pt"
    torch.save(ck_x, px)
    args = _model_route_args(fw, tmp_path / "hx3.json", ["--tag", "M2b-oacc_s1"])
    args[args.index("--ckpt") + 1] = str(px)
    assert _no_crash(OE.main, ["hx", *args]) == 0
    r3 = json.loads((tmp_path / "hx3.json").read_text(encoding="utf-8"))
    assert r3["w_acc_built"] == 3.0 and r3["ckpt_sha256"] == OE.sha256_file(px) != r["ckpt_sha256"] and r3["tag"] == "M2b-oacc_s1"
    assert [r3["w_along_by_step"], r3["w_cross_by_step"]] == [r["w_along_by_step"], r["w_cross_by_step"]], "the same weights: the checkpoint's outputs are the same (only the recorded cfg differs)"


# =============================================================================================================================================
# (G) THE READER: code/v8f_oacc_read.py
# =============================================================================================================================================
RAWF = PKG / "raw" / "full" / "eval"
HBIN = PKG / "raw" / "2026-10-25-hblend" / "inputs"
needs_reader = pytest.mark.skipif(not READ_P.exists(), reason="the v8F-mini package is not in this checkout")
needs_real = pytest.mark.skipif(not (RAWF / "eval_M2b_s0.npz").exists() or not (HBIN / "b2_p10_prior.pt").exists(), reason="the banked B evals / P10 files are not in this checkout")
RD = _load("v8f_oacc_read_t", READ_P) if READ_P.exists() else None
BOOL3 = [dict(ACC=a, KEEP=k, DOM=d) for a in (True, False) for k in (True, False) for d in (True, False)]


def _expected_outcome(states):
    """Authored from the PREREG's table and the declared interpretation I2 (never from the code under test)."""
    if any(not v["ACC"] for v in states.values()):
        return "NO EFFECT", False
    if any(not v["KEEP"] for v in states.values()):
        return "COSTS POSITION", False
    if len(states) == 1:
        return "PENDING SEED 1", True
    return ("DOMINATES ITS PRIOR" if all(v["DOM"] for v in states.values()) else "SMOOTHER, NOT DOMINANT"), False


@needs_reader
def test_the_outcome_partition_is_complete_over_every_cell_for_one_and_two_seeds():
    n = 0
    for s0 in BOOL3:
        assert RD.outcome({0: s0}) == _expected_outcome({0: s0})
        n += 1
        for s1 in BOOL3:
            assert RD.outcome({0: s0, 1: s1}) == _expected_outcome({0: s0, 1: s1}), (s0, s1)
            n += 1
    assert n == 8 + 64
    assert RD.outcome({0: dict(ACC=True, KEEP=True, DOM=False)}) == ("PENDING SEED 1", True), "seed 1 runs iff seed 0 passes ACC and KEEP, whatever DOMINANCE says"
    assert RD.outcome({0: dict(ACC=True, KEEP=False, DOM=True)}) == ("COSTS POSITION", False)
    for bad in ({1: BOOL3[0]}, {0: BOOL3[0], 1: BOOL3[0], 2: BOOL3[0]}, {0: dict(ACC=1, KEEP=True, DOM=True)}, {0: dict(ACC=True, KEEP=True)}):
        with pytest.raises(SystemExit) as e:
            RD.outcome(bad)
        assert e.value.code == 2


@needs_reader
def test_the_next_texts_and_the_four_outcome_names_are_the_preregs_verbatim():
    rows = {}
    for ln in PREREG.read_text(encoding="utf-8").splitlines():
        if ln.startswith("| **") and ln.count("|") >= 4:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows[cells[0].strip("*").strip()] = cells
    assert set(RD.OUTCOMES) <= set(rows), set(RD.OUTCOMES) - set(rows)
    for name in RD.OUTCOMES:
        assert rows[name][2] == RD.NEXT_TEXT[name], name


@needs_reader
def test_the_dominance_row_is_hspecs_verbatim():
    import ast
    src = (PKG / "raw" / "2026-10-25-hspec" / "v8f_hspec.py")
    if not src.exists():
        pytest.skip("HSPEC's package is not in this checkout")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    got = {t.targets[0].id: ast.literal_eval(t.value) for t in tree.body if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name) and t.targets[0].id in ("DOM_BETTER", "DOM_NOT_WORSE")}
    assert got == {"DOM_BETTER": RD.DOM_BETTER, "DOM_NOT_WORSE": RD.DOM_NOT_WORSE}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dk_flags")
    mine = next(n for n in ast.parse(READ_P.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "dk_flags")
    strip = lambda f: ast.dump(f.body[-2]) + ast.dump(f.body[-1])      # noqa: E731  the loop and the return (the docstring differs)
    assert strip(fn) == strip(mine), "dk_flags: the loop body and the return are HSPEC's"


def _row(mean=-0.1, lo=-0.2, hi=-0.05, sep=True, better=None, worse=None):
    return {"mean": mean, "lo": lo, "hi": hi, "separated": sep, "better": sep and mean < 0 if better is None else better, "worse": sep and mean > 0 if worse is None else worse}


def _paired_all(better=True):
    keys = list(RD.DOM_BETTER.values()) + list(RD.DOM_NOT_WORSE.values())
    return {k: (_row() if k in RD.DOM_BETTER.values() and better else _row(0.0, -0.1, 0.1, False)) for k in keys}


def _dkf(**over):
    base = {"headway min (m)": dict(delta=0.1, lo=-0.1, hi=0.3, separated=False, worse=False, better=False),
            "time-gap min (s)": dict(delta=0.0, lo=-0.1, hi=0.1, separated=False, worse=False, better=False),
            "LEAD-VIOL at-risk share": dict(delta=0.0, lo=-0.1, hi=0.1, separated=False, worse=False, better=False)}
    base.update(over)
    return base


@needs_reader
def test_dom_row_passes_on_separated_better_and_not_worse_and_fails_on_each_single_defect():
    ok = RD.dom_row(_paired_all(), _row(0.0, -0.05, 0.05, False), _dkf())
    assert ok["pass"] is True and ok["all_better_ok"] and ok["all_not_worse_ok"] and len(ok["rows"]) == 4 + 7 + 1 + 3
    for nm, key in RD.DOM_BETTER.items():                                      # each of the four 'separated-better' metrics, alone, not better: FAIL
        p = _paired_all()
        p[key] = _row(0.0, -0.1, 0.1, False)
        assert RD.dom_row(p, _row(0.0, -0.05, 0.05, False), _dkf())["pass"] is False, nm
    for nm, key in RD.DOM_NOT_WORSE.items():                                   # each of the seven 'not separated-worse' metrics, alone, separated-worse: FAIL
        p = _paired_all()
        p[key] = _row(0.2, 0.1, 0.3, True)
        assert RD.dom_row(p, _row(0.0, -0.05, 0.05, False), _dkf())["pass"] is False, nm
    assert RD.dom_row(_paired_all(), _row(0.1, 0.05, 0.2, True), _dkf())["pass"] is False, "CONSISTENCY(a) 'any' separated-worse"
    assert RD.dom_row(_paired_all(), _row(-0.1, -0.2, -0.05, True), _dkf())["pass"] is True, "separated-BETTER consistency is not a defect"
    for nm in ("headway min (m)", "time-gap min (s)", "LEAD-VIOL at-risk share"):
        assert RD.dom_row(_paired_all(), _row(0.0, -0.05, 0.05, False), _dkf(**{nm: dict(delta=1.0, lo=0.5, hi=1.5, separated=True, worse=True, better=False)}))["pass"] is False, nm


@needs_reader
def test_dk_flags_directions():
    pair = {"dk": {"headway_min_m": dict(delta=-1.0, lo=-2.0, hi=-0.5, separated=True), "time_gap_min_s": dict(delta=0.5, lo=0.2, hi=0.8, separated=True)},
            "lead_viol": {"at_risk": dict(delta=0.05, lo=0.01, hi=0.09, separated=True)}}
    f = RD.dk_flags(pair)
    assert f["headway min (m)"]["worse"] and not f["headway min (m)"]["better"], "less headway, separated, is worse"
    assert f["time-gap min (s)"]["better"] and not f["time-gap min (s)"]["worse"]
    assert f["LEAD-VIOL at-risk share"]["worse"], "a higher at-risk violation share, separated, is worse"


@needs_reader
def test_keep_row_bar_is_half_the_exact_f_and_the_printed_variant_is_beside():
    F = 0.12074
    bar = F / 2
    up, down = float(np.nextafter(bar, 1.0)), float(np.nextafter(bar, 0.0))
    assert RD.keep_row(_row(0.0, -0.1, bar), F)["pass"] is True, "exactly at F/2: the bar is '<='"
    assert RD.keep_row(_row(0.0, -0.1, up), F)["pass"] is False and RD.keep_row(_row(0.0, -0.1, down), F)["pass"] is True, "one ulp either side of F/2"
    assert RD.keep_row(_row(0.0, -0.1, F), F)["pass"] is False and RD.keep_row(_row(0.0, -0.1, 0.0), F)["pass"] is True, "the bar is F/2, not F, and a hi at 0 passes"
    k = RD.keep_row(_row(0.0, -0.1, 0.0603), F)
    assert k["pass"] is True and k["hi_le_printed_0.060"] is False and k["bar_hi_le"] == bar, "0.0603 passes the exact bar and misses the printed 0.060: both shown, the exact one decides (I1)"
    assert RD.keep_row(_row(0.0, -0.1, 0.060), F)["hi_le_printed_0.060"] is True and RD.keep_row(_row(0.0, -0.1, 0.0600001), F)["hi_le_printed_0.060"] is False, "the printed variant is '<= 0.060'"
    assert RD.keep_row(_row(0.0, -0.1, 0.0600001), F)["pass"] is True and RD.keep_row(_row(0.0, -0.1, 0.0604), F)["pass"] is False
    assert RD.keep_row(_row(-5.0, -6.0, 0.07), F)["pass"] is False, "the UPPER bound decides, not the point (a very good point with a wide interval fails)"
    assert RD.keep_row(_row(5.0, -6.0, 0.0), F)["pass"] is True, "the UPPER bound decides, not the point"
    r = RD.keep_row(_row(0.02, -0.01, 0.05, sep=False), F)
    assert (r["mean"], r["lo"], r["hi"], r["separated"]) == (0.02, -0.01, 0.05, False) and r["bar_hi_le"] == bar, "the row carries the paired row's own numbers"


@needs_reader
def test_seed_pass_acc_needs_negative_and_separated():
    bar = 0.12074 / 2
    keep, dom = {"mean": 0.0, "hi": 0.01, "bar_hi_le": bar}, {"pass": True}
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), keep, dom) == dict(ACC=True, KEEP=True, DOM=True)
    assert RD.seed_pass(_row(-0.1, -0.2, 0.05, False), keep, dom)["ACC"] is False, "negative but not separated"
    assert RD.seed_pass(_row(0.0, 0.0, 0.0, False), keep, dom)["ACC"] is False
    assert RD.seed_pass(_row(0.1, 0.05, 0.2, True), keep, dom)["ACC"] is False
    # the boundary at exactly 0: '< 0' (a separated record with a mean of exactly 0 is NOT a pass) and a hair either side
    assert RD.seed_pass(_row(0.0, -0.2, 0.2, True), keep, dom)["ACC"] is False, "exactly 0.0 with the separated flag set: not < 0"
    assert RD.seed_pass(_row(-1e-12, -2e-12, -1e-13, True), keep, dom)["ACC"] is True and RD.seed_pass(_row(1e-12, 1e-13, 2e-12, True), keep, dom)["ACC"] is False
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), {"mean": 0.0, "hi": 0.061, "bar_hi_le": 0.06}, dom)["KEEP"] is False
    # KEEP: the upper bound against the bar, '<=' (one ulp either side), never the point
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), {"mean": 0.0, "hi": bar, "bar_hi_le": bar}, dom)["KEEP"] is True, "hi exactly at the bar passes"
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), {"mean": 0.0, "hi": float(np.nextafter(bar, 1.0)), "bar_hi_le": bar}, dom)["KEEP"] is False
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), {"mean": 0.0, "hi": float(np.nextafter(bar, 0.0)), "bar_hi_le": bar}, dom)["KEEP"] is True
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), {"mean": -9.0, "hi": 1.0, "bar_hi_le": bar}, dom)["KEEP"] is False, "a very good point with a wide interval fails"
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), {"mean": 9.0, "hi": 0.0, "bar_hi_le": bar}, dom)["KEEP"] is True, "a bad point under a tight interval is the interval's"
    # DOMINANCE is the dominance row's own verdict
    assert RD.seed_pass(_row(-0.1, -0.2, -0.05, True), keep, {"pass": False}) == dict(ACC=True, KEEP=True, DOM=False)
    assert RD.seed_pass(_row(0.1, 0.05, 0.2, True), keep, {"pass": True}) == dict(ACC=False, KEEP=True, DOM=True)
    assert all(type(v) is bool for v in RD.seed_pass(_row(-0.1, -0.2, -0.05, True), keep, {"pass": 1}).values()), "plain booleans (the outcome table refuses anything else)"


def _gate_records(w=42.9311):
    sc = {"declared_vs_built": "PASS", "argv_sha": "a1", "live": {"pass": True, "control_prior_grad_sum": 0.0, "dead_params": [], "declared_dead": [], "terms": {"acc": 0.1}},
          "acc_probe": {"pass": True, "w_acc_declared": w, "head_o_grad_diff": 3.0, "control_prior_grad_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_repeat_total_diff": 0.0,
                        "contribution": 0.62, "w_x_acc": 0.62},
          "levers": {"w_acc": w, "params": {"O": 5, "S": 3}, "n_params_trainable": 8, "hmax": 16, "wm": "b"}}
    sm = {"done": True, "arm": "M2b", "seed": 0, "n_clips": 4512, "excluded_sha256": "H", "train_sha256": "T", "steps": 19164, "n_windows": 306566, "argv_sha": "a1", "levers": dict(sc["levers"])}
    rsm = {"train_sha256": "T", "steps": 19164, "n_windows": 306566, "levers": {"params": {"O": 5, "S": 3}, "n_params_trainable": 8, "hmax": 16, "wm": "b"}}
    rargv = ["--arm", "M2b", "--seed", "0", "--out", "D:/b", "--batch", "64", "--device", "cuda"]
    argv = ["--arm", "M2b", "--seed", "0", "--w-acc", str(w), "--out", "D:/x", "--batch", "64", "--device", "cuda"]
    return sm, argv, sc, rsm, rargv


@needs_reader
def test_check_x_run_accepts_b_plus_one_lever_and_names_every_defect():
    sm, argv, sc, rsm, rargv = _gate_records()
    assert _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm, rargv, 42.9311, holdout="H") == []
    cases = {}
    a2 = list(argv)
    a2[a2.index("--w-acc") + 1] = "42.0"
    cases["argv carries another value than the rule's"] = (sm, a2, sc, rsm, rargv)
    cases["--w-acc named twice"] = (sm, argv + ["--w-acc", "42.9311"], sc, rsm, rargv)
    cases["a second lever in the argv"] = (sm, argv + ["--code-mix", "kind"], sc, rsm, rargv)
    cases["--w-acc is not in the argv"] = (sm, [t for t in argv if t not in ("--w-acc", "42.9311")], sc, rsm, rargv)
    sc2 = json.loads(json.dumps(sc))
    sc2["acc_probe"]["head_o_grad_diff"] = 0.0
    cases["O's head got no gradient"] = (sm, argv, sc2, rsm, rargv)
    sc3 = json.loads(json.dumps(sc))
    del sc3["acc_probe"]
    cases["no acc_probe"] = (sm, argv, sc3, rsm, rargv)
    sc4 = json.loads(json.dumps(sc))
    sc4["acc_probe"]["control_prior_grad_diff"] = 1e-9
    cases["a control is not exactly 0.0"] = (sm, argv, sc4, rsm, rargv)
    sc5 = json.loads(json.dumps(sc))
    sc5["live"]["dead_params"] = ["e_code.weight"]
    cases["a dead weight"] = (sm, argv, sc5, rsm, rargv)
    sc6 = json.loads(json.dumps(sc))
    sc6["levers"]["w_acc"] = 0.0
    cases["built w_acc is 0"] = (sm, argv, sc6, rsm, rargv)
    sm7 = json.loads(json.dumps(sm))
    sm7["levers"]["n_params_trainable"] = 9
    cases["a parameter more than B"] = (sm7, argv, sc, rsm, rargv)
    cases["a different step count"] = (dict(sm, steps=1000), argv, sc, rsm, rargv)
    cases["B's argv names the lever"] = (sm, argv, sc, rsm, rargv + ["--w-acc", "1.0"])
    sc9 = json.loads(json.dumps(sc))
    sc9["acc_probe"]["w_acc_declared"] = 1.0
    cases["the probe ran for another weight"] = (sm, argv, sc9, rsm, rargv)
    for name, (a_, b_, c_, d_, e_) in cases.items():
        assert _no_crash(RD.check_x_run, 0, a_, b_, c_, d_, e_, 42.9311, holdout="H"), f"not refused: {name}"
    assert _no_crash(RD.check_x_run, 0, dict(sm, excluded_sha256="Z"), argv, sc, rsm, rargv, 42.9311, holdout="H"), "the holdout"


@needs_reader
def test_check_x_run_amendment_2_record_formats():
    """PREREG_V8F_OACC Amendment 2: B's older record (residual_zero_init after training, `w` without w_tac / w_cons) against X's current one is
    ONE lever; any real loss-weight difference is still refused, by its name."""
    sm, argv, sc, rsm, rargv = _gate_records()
    new_w = {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5}
    for rec in (sm, sc):
        rec["levers"] = dict(rec["levers"], w=dict(new_w), residual_zero_init=True)
    rsm["levers"] = dict(rsm["levers"], w={"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1}, residual_zero_init=False)
    assert _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm, rargv, 42.9311, holdout="H") == []
    for key, val in (("w_tac", 2.0), ("w_cons", 0.25), ("w_wm", 0.2), ("w_pen", None), ("w_new", 1.0)):
        w2 = dict(new_w)
        if val is None:
            w2.pop(key)
        else:
            w2[key] = val
        sm2 = dict(sm, levers=dict(sm["levers"], w=w2))
        got = _no_crash(RD.check_x_run, 0, sm2, argv, sc, rsm, rargv, 42.9311, holdout="H")
        assert got and any("loss weights" in g and key in g for g in got), (key, got)
    rsm2 = dict(rsm, levers=dict(rsm["levers"], w={"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 2.0}))   # B's record carries w_tac: compared as recorded
    assert any("w_tac" in g for g in _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm2, rargv, 42.9311, holdout="H"))


@needs_reader
def test_check_x_run_names_each_clause_of_the_token_by_its_own_message():
    """One defect per case, and the message of THAT clause must be in the reasons: a clause that a neighbour also catches is not tested by `not refused` (the first review's mutation run found the PASS-token clause unpinned)."""
    sm, argv, sc, rsm, rargv = _gate_records()

    def jd(o):
        return json.loads(json.dumps(o))

    def sc_with(f):
        s_ = jd(sc)
        f(s_)
        return (sm, argv, s_, rsm, rargv)

    def sm_with(f):
        s_ = jd(sm)
        f(s_)
        return (s_, argv, sc, rsm, rargv)

    cases = [
        ("declared_vs_built FAIL", sc_with(lambda s: s.__setitem__("declared_vs_built", "FAIL")), "declared_vs_built is not PASS"),
        ("declared_vs_built absent", sc_with(lambda s: s.pop("declared_vs_built")), "declared_vs_built is not PASS"),
        ("G-LIVE not pass", sc_with(lambda s: s["live"].__setitem__("pass", False)), "G-LIVE is not a pass"),
        ("G-LIVE pass absent", sc_with(lambda s: s["live"].pop("pass")), "G-LIVE is not a pass"),
        ("frozen-prior control not exactly 0.0", sc_with(lambda s: s["live"].__setitem__("control_prior_grad_sum", 1e-12)), "frozen-prior control of exactly 0.0"),
        ("a dead weight", sc_with(lambda s: s["live"].__setitem__("dead_params", ["e_code.weight"])), "dead-weight set"),
        ("a declared-dead weight", sc_with(lambda s: s["live"].__setitem__("declared_dead", ["e_code.weight"])), "dead-weight set"),
        ("acc not in the loss dict", sc_with(lambda s: s["live"].__setitem__("terms", {"ade": 1.0})), "carry no 'acc'"),
        ("no terms at all", sc_with(lambda s: s["live"].pop("terms")), "carry no 'acc'"),
        ("acc_probe not pass", sc_with(lambda s: s["acc_probe"].__setitem__("pass", False)), "no passing acc_probe"),
        ("acc_probe for another weight", sc_with(lambda s: s["acc_probe"].__setitem__("w_acc_declared", 1.0)), "acc_probe was run for w_acc"),
        ("O's head got no gradient", sc_with(lambda s: s["acc_probe"].__setitem__("head_o_grad_diff", 0.0)), "received no gradient"),
        ("O's head gradient absent", sc_with(lambda s: s["acc_probe"].pop("head_o_grad_diff")), "received no gradient"),
        ("prior control not 0", sc_with(lambda s: s["acc_probe"].__setitem__("control_prior_grad_diff", 1e-9)), "does not read exactly 0.0"),
        ("repeat-grad control not 0", sc_with(lambda s: s["acc_probe"].__setitem__("control_repeat_grad_diff", 1e-9)), "does not read exactly 0.0"),
        ("repeat-total control not 0", sc_with(lambda s: s["acc_probe"].__setitem__("control_repeat_total_diff", 1e-9)), "does not read exactly 0.0"),
        ("the total did not move by w x term", sc_with(lambda s: s["acc_probe"].__setitem__("contribution", 0.5)), "the total moved by"),
        ("selfcheck is another run's", sc_with(lambda s: s.__setitem__("argv_sha", "zz")), "is not this run's"),
        ("not done", sm_with(lambda s: s.__setitem__("done", False)), "is not done"),
        ("another arm", sm_with(lambda s: s.__setitem__("arm", "M2a")), "expected M2b seed 0"),
        ("another seed", sm_with(lambda s: s.__setitem__("seed", 1)), "expected M2b seed 0"),
        ("another clip count", sm_with(lambda s: s.__setitem__("n_clips", 4000)), "n_clips"),
        ("no holdout withheld", sm_with(lambda s: s.__setitem__("excluded_sha256", "Z")), "did not withhold the RA-1 holdout"),
        ("another clip list", sm_with(lambda s: s.__setitem__("train_sha256", "U")), "clip list"),
        ("another step count", sm_with(lambda s: s.__setitem__("steps", 1000)), "step count"),
        ("another window set", sm_with(lambda s: s.__setitem__("n_windows", 5)), "window set"),
        ("another arm on the argv", (sm, [("M2a" if a_ == "M2b" else a_) for a_ in argv], sc, rsm, rargv), "argv --arm"),
        ("a second lever", (sm, argv + ["--code-mix", "kind"], sc, rsm, rargv), "besides --w-acc and --out"),
        ("B's argv names the lever", (sm, argv, sc, rsm, rargv + ["--w-acc", "1.0"]), "which predates the flag"),
        ("the lever twice", (sm, argv + ["--w-acc", "42.9311"], sc, rsm, rargv), "exactly once"),
        ("the lever's value", (sm, [("42.0" if a_ == "42.9311" else a_) for a_ in argv], sc, rsm, rargv), "exactly once"),
        ("acc_probe absent", sc_with(lambda s: s.pop("acc_probe")), "carries no passing acc_probe"),
        ("acc_probe is not a dict", sc_with(lambda s: s.__setitem__("acc_probe", "PASS")), "carries no passing acc_probe"),
        ("summary: no built w_acc", sm_with(lambda s: s["levers"].pop("w_acc")), "summary.json levers record no built w_acc: the lever is unproven"),
        ("selfcheck: no built w_acc", sc_with(lambda s: s["levers"].pop("w_acc")), "selfcheck.json levers record no built w_acc: the lever is unproven"),
        ("summary: another built w_acc", sm_with(lambda s: s["levers"].__setitem__("w_acc", 41.0)), "summary.json built w_acc 41.0, the rule's value is 42.9311"),
        ("selfcheck: another built w_acc", sc_with(lambda s: s["levers"].__setitem__("w_acc", 0.0)), "selfcheck.json built w_acc 0.0, the rule's value is 42.9311"),
        ("summary: another built lever", sm_with(lambda s: s["levers"].__setitem__("hmax", 8)), "summary.json built levers ['hmax'] differ from B_s0's"),
        ("selfcheck: another built lever", sc_with(lambda s: s["levers"].__setitem__("wm", "off")), "selfcheck.json built levers ['wm'] differ from B_s0's"),
        ("summary: another parameter count", sm_with(lambda s: s["levers"]["params"].__setitem__("O", 6)), "summary.json parameter counts"),
        ("selfcheck: another trainable count", sc_with(lambda s: s["levers"].__setitem__("n_params_trainable", 9)), "selfcheck.json parameter counts"),
        ("B recorded no parameter counts", (sm, argv, sc, dict(rsm, levers={"hmax": 16, "wm": "b"}), rargv), "parameter counts"),
    ]
    for name, (a_, b_, c_, d_, e_), want in cases:
        got = _no_crash(RD.check_x_run, 0, a_, b_, c_, d_, e_, 42.9311, holdout="H")
        assert any(want in g for g in got), f"{name}: expected a reason containing {want!r}, got {got}"
    assert _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm, rargv, 42.9311, holdout="H") == [], "the good record is clean"
    assert _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm, rargv, 42.9311, holdout=None) == [], "no holdout given: that clause is skipped, nothing else is"
    assert _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm, rargv, 42.9311, n_train=4000, holdout="H") and any("n_clips" in g for g in RD.check_x_run(0, sm, argv, sc, rsm, rargv, 42.9311, n_train=4000, holdout="H")), "n_train is a parameter"


@pytest.fixture(scope="module")
def fab(tmp_path_factory):
    """The REAL banked B evals (raw/full/eval) as B, the REAL P10-yaw-4512 files, and a fabricated X whose predictions a test chooses: its eval npz / json written by the eval's OWN `evaluate`, its run dir
    (summary / argv / selfcheck / checkpoint with the real frozen P10), its HX weights, the rule's json, the checkpoint sha json. Everything the reader refuses on is a file here, so a test edits one file."""
    I = RD.Instruments()
    ns = type("A", (), {"eval_grid": str(RD.RAW / "eval_grid_windows.json"), "eval_kin": str(RD.RAW / "eval141_kin.npz"), "eval_join": str(RD.RAW / "join_eval141.npz")})
    return {"I": I, "world": RD.eval_world(I, ns), "base": tmp_path_factory.mktemp("oacc_fab")}


def _fab_arm(fab, name, tau_fn, w_acc=42.9311, nb=40, w_hx=None, w_cross=None, p10_viol_any=None):
    """Write a complete OACC X for BOTH seeds under fab['base']/name with predictions tau_fn(seed, B_pred, P10) -> [N, 30, 2]; return the directory. ``w_hx`` / ``w_cross``: the HX weights per step
    (along / cross; cross defaults to along). ``p10_viol_any``: a [N] bool array that REPLACES the P10 floors file's consistency flags ('any'), so a test knows P10's consistency rate by construction."""
    I, (grid, J, X) = fab["I"], fab["world"]
    H, ev, D = I.H, I.ev, I.D
    from tanitad.models import v8f_mini as M
    base = fab["base"] / name
    (base / "eval").mkdir(parents=True), (base / "runs").mkdir(), (base / "w").mkdir()
    P10 = np.load(HBIN / "b2_p10_eval_pred6.npy").astype(np.float64)
    if p10_viol_any is None:
        shutil.copy(RAWF / "floors_perwindow.npz", base / "eval" / "floors_perwindow.npz")
    else:
        fz = dict(np.load(RAWF / "floors_perwindow.npz"))
        assert np.asarray(p10_viol_any).shape == fz["P10_refit__viol_any"].shape
        fz["P10_refit__viol_any"] = np.asarray(p10_viol_any, dtype=fz["P10_refit__viol_any"].dtype)
        np.savez_compressed(base / "eval" / "floors_perwindow.npz", **fz)
    shas = {"M2b_s0": RD.BANKED_CKPT_SHA["M2b_s0"], "M2b_s1": RD.BANKED_CKPT_SHA["M2b_s1"]}
    for s in (0, 1):
        shutil.copy(RAWF / f"eval_M2b_s{s}.npz", base / "eval"), shutil.copy(RAWF / f"eval_M2b_s{s}.json", base / "eval")
        zB = np.load(RAWF / f"eval_M2b_s{s}.npz")
        bjB = json.loads((RAWF / f"eval_M2b_s{s}.json").read_text(encoding="utf-8"))
        tau = np.asarray(tau_fn(s, zB["pred"].astype(np.float64), P10), np.float64)
        r, pw = ev.evaluate(tau, X, H.extras_of(zB, D), nb, with_families=True)
        arrs = {k: zB[k] for k in zB.files}
        arrs.update({"pred": tau, "traj30": tau}, **{k: v for k, v in pw.items() if k in arrs})
        np.savez_compressed(base / "eval" / f"eval_{RD.x_tag(s)}.npz", **arrs)
        raw = H.jround(r)
        raw["strategic"] = bjB["raw"]["strategic"]
        (base / "eval" / f"eval_{RD.x_tag(s)}.json").write_text(json.dumps({"tag": RD.x_tag(s), "ckpt": f"D:/tanitad/runs/v8f_mini_full/{RD.x_dir(s)}/ckpt_last.pt", "o_interface": "full", "raw": raw}), encoding="utf-8")
        cfg = M.V8FMiniConfig(**{**_T.TINY, "wm": "off", "w_acc": w_acc, "kin_mode": "yaw"})        # the X checkpoint: a tiny model carrying the REAL frozen P10 (the reader evaluates its prior on the eval kin features)
        model = M.V8FMini(cfg, M.load_p10(HBIN / "b2_p10_prior.pt"))
        import torch
        for tag_dir, wa in ((RD.x_dir(s), w_acc), (f"M2b_s{s}", None)):
            rd = base / "runs" / tag_dir
            rd.mkdir()
            smx, argvx, scx, rsm, rargv = _gate_records(w_acc)
            hold = I.FR.PGT.HOLDOUT_SHA
            if wa is None:                                                     # B_s: the banked run's records (summary / argv / selfcheck), no lever
                smb = dict(rsm, done=True, arm="M2b", seed=s, n_clips=4512, excluded_sha256=hold, argv_sha="b1", levers=dict(rsm["levers"], p_gt_start=1.0, p_gt_end=0.2))
                (rd / "summary.json").write_text(json.dumps(smb)), (rd / "argv.json").write_text(json.dumps({"argv": rargv})), (rd / "selfcheck.json").write_text(json.dumps({"declared_vs_built": "PASS"}))
                continue
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "argv_sha": "a1", "step": 19164, "arm": "M2b", "seed": s}, rd / "ckpt_last.pt")
            sha = RD.sha256_file(rd / "ckpt_last.pt")
            shas[f"{RD.x_tag(s)}_trained"] = shas[f"{RD.x_tag(s)}_after_evals"] = sha
            smx.update(seed=s, excluded_sha256=hold)
            smx["levers"] = dict(smx["levers"], p_gt_start=1.0, p_gt_end=0.2)
            (rd / "summary.json").write_text(json.dumps(smx)), (rd / "argv.json").write_text(json.dumps({"argv": argvx})), (rd / "selfcheck.json").write_text(json.dumps(scx))
            w = np.full(30, 0.5) if w_hx is None else np.asarray(w_hx)
            wc = w if w_cross is None else np.asarray(w_cross)
            (base / "w" / f"weights_{RD.x_tag(s)}.json").write_text(json.dumps({"ckpt_sha256": sha, "w_acc_built": w_acc, "w_along_by_step": w.tolist(), "w_cross_by_step": wc.tolist(), "n_windows": 9582,
                                                                                  "fit_clips_sha256": "f"}))
    (base / "ckpt_sha256.json").write_text(json.dumps(shas))
    (base / "w_acc.json").write_text(json.dumps({"schema": "v8f-oacc-wacc-v1", "w_acc": w_acc}))
    return base


def _reader_args(base, seeds, out, nb=40, extra=()):
    return ["--eval-dir", str(base / "eval"), "--runs-dir", str(base / "runs"), "--weights-dir", str(base / "w"), "--wacc-json", str(base / "w_acc.json"), "--ckpt-sha", str(base / "ckpt_sha256.json"),
            "--seeds", seeds, "--out", str(out), "--n-boot", str(nb), "--dk-n-boot", str(nb), "--p10-pred", str(HBIN / "b2_p10_eval_pred6.npy"), "--lead-block-k30", str(HBIN / "b1_eval_lead_block_k30.npz"), *extra]


def _smooth(tau, deg=4):
    """Project each window's plan onto the polynomials of degree <= deg in time that vanish at the origin (the ego at NOW): the same positions to a few tens of cm, and a smooth acceleration."""
    t = np.arange(1, 31, dtype=np.float64)[:, None] / 30.0
    V = np.concatenate([t ** k for k in range(1, deg + 1)], 1)
    return np.einsum("tk,kn,bna->bta", V, np.linalg.pinv(V), tau)


def _read(base, seeds, tmp_path, capsys, nb=40, extra=(), name="r.json"):
    """Run the reader's CLI on a fabricated arm -> (rc, stderr, the output json or None, stdout)."""
    out = tmp_path / name
    capsys.readouterr()
    rc = _no_crash(RD.main, _reader_args(base, seeds, out, nb, extra))   # an exception out of the reader is a failed assertion; its refusals (SystemExit) pass through
    cap = capsys.readouterr()
    return rc, cap.err, (json.loads(out.read_text(encoding="utf-8")) if out.exists() and rc == 0 else None), cap.out


# =============================================================================================================================================
# THE ORACLE. Authored HERE, from the family definitions (taniteval/four_families._seq_geometry, written out below with plain numpy) and from the PREREG's text; it never calls the reader's code paths.
# Per window and arm: ADE_h = mean over the first 5h plan steps of |plan - GT|; speed_k = |p_k - p_(k-1)| / 0.2 (p_0 = the ego at NOW = the origin); accel_k = (speed_(k+1) - speed_k) / 0.2;
# curvature_k = d heading / mean(ds_k, ds_(k+1)); each MAE averages |plan - GT| (along / cross: the 30 knots; accel: 29 centres; heading: the steps BOTH paths move through; yaw rate / curvature: the PAIRS both move through).
# HX = P + w (tau - P) per step and axis (along = x, cross = y). The estimator is the programme's paired episode-cluster bootstrap (the library, not under test here): the same rows, from the oracle's arrays.
# =============================================================================================================================================
_DT = 0.2
_MIN_DS = 0.5 * _DT                 # a step carries a tangent only above 0.5 m/s
_LOWER_IS_BETTER_NOT = ("tac_lat", "tac_lon")      # the plan-derived decision agreements: higher is better


def _o_geom(p):
    q = np.concatenate([np.zeros((p.shape[0], 1, 2)), p], 1)
    d = np.diff(q, axis=1)
    ds = np.linalg.norm(d, axis=-1)
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (np.diff(h, axis=1) + math.pi) % (2 * math.pi) - math.pi
    v = ds / _DT
    return dict(x=q[:, 1:, 0], y=q[:, 1:, 1], h=h, v=v, a=np.diff(v, axis=1) / _DT, yaw=dh / _DT, curv=dh / (0.5 * (ds[:, 1:] + ds[:, :-1]) + 1e-8),
                ok=ds > _MIN_DS, pair=(ds[:, 1:] > _MIN_DS) & (ds[:, :-1] > _MIN_DS))


def _o_mean_over(x, mask):
    n = mask.sum(1)
    return np.where(n > 0, (x * mask).sum(1) / np.maximum(n, 1), np.nan)


def _o_metrics(pred, gt):
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    P, G = _o_geom(pred), _o_geom(gt)
    e = np.linalg.norm(pred - gt, axis=-1)
    both, pair = P["ok"] & G["ok"], P["pair"] & G["pair"]
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi      # noqa: E731
    return {"ade2": e[:, :10].mean(1), "ade4": e[:, :20].mean(1), "ade6": e.mean(1), "fde6": e[:, 29],
            "speed": np.abs(P["v"] - G["v"]).mean(1), "accel": np.abs(P["a"] - G["a"]).mean(1),
            "along": np.abs(P["x"] - G["x"]).mean(1), "cross": np.abs(P["y"] - G["y"]).mean(1),
            "heading": _o_mean_over(np.abs(wrap(P["h"] - G["h"])), both) * 180 / math.pi,
            "yaw": _o_mean_over(np.abs(P["yaw"] - G["yaw"]), pair) * 180 / math.pi,
            "curv": _o_mean_over(np.abs(P["curv"] - G["curv"]), pair)}


def _o_hx(P, tau, w_along, w_cross):
    w = np.stack([np.asarray(w_along, np.float64), np.asarray(w_cross, np.float64)], 1)
    return np.asarray(P, np.float64) + w[None] * (np.asarray(tau, np.float64) - np.asarray(P, np.float64))


# the PREREG's DOMINANCE row (HSPEC's PRIMARY-DOM, verbatim): (the row's name in the reader's output, the oracle's metric key, the PREREG's requirement)
_DOM_SPEC = (("ADE2", "ade2", "better"), ("ADE6", "ade6", "better"), ("accel MAE", "accel", "better"), ("curvature MAE", "curv", "better"),
             ("speed MAE", "speed", "not worse"), ("along MAE", "along", "not worse"), ("cross-track MAE", "cross", "not worse"), ("heading MAE", "heading", "not worse"),
             ("yaw-rate MAE", "yaw", "not worse"), ("plan-derived lateral decision agreement", "tac_lat", "not worse"), ("plan-derived longitudinal decision agreement", "tac_lon", "not worse"))
_METRICS = ("ade2", "ade4", "ade6", "fde6", "speed", "accel", "along", "cross", "heading", "yaw", "curv", "tac_lat", "tac_lon")


def _o_pw(pred, gt, BE):
    m = _o_metrics(pred, gt)
    fw = BE.family_windows(pred, gt, _DT)               # the two decision agreements are the programme's rule engine (no closed form to re-author): used on the RIGHT arms only
    m["tac_lat"], m["tac_lon"] = fw["tac_lat_acc"], fw["tac_lon_acc"]
    return m


def _oracle_seed(fab, base, s, nb):
    """Every number the reader must print for seed ``s`` of the world in ``base``, from the world's INPUT files (B's banked eval, X's eval, the real P10, the HX weights, the floors file)."""
    I, (grid, J, Xc) = fab["I"], fab["world"]
    BE, H, ev, D, DK = I.BE, I.H, I.ev, I.D, I.DK
    gt, eid = Xc["gt"].astype(np.float64), Xc["eid"]
    ed = base / "eval"
    zB, zX, zB0, zB1 = (np.load(ed / f"eval_{t}.npz") for t in (f"M2b_s{s}", f"M2b-oacc_s{s}", "M2b_s0", "M2b_s1"))
    fl = np.load(ed / "floors_perwindow.npz")
    P10 = np.load(HBIN / "b2_p10_eval_pred6.npy").astype(np.float64)
    W = json.loads((base / "w" / f"weights_M2b-oacc_s{s}.json").read_text(encoding="utf-8"))
    arms = {"B": zB["pred"].astype(np.float64), "X": zX["pred"].astype(np.float64), "P10": P10}
    arms["HX"] = _o_hx(P10, arms["X"], W["w_along_by_step"], W["w_cross_by_step"])
    pw = {k: _o_pw(v, gt, BE) for k, v in arms.items()}

    def row(a, b, key):
        return BE.paired_row(pw[a][key], pw[b][key], eid, nb, lower_is_better=key not in _LOWER_IS_BETTER_NOT)
    F = abs(float(zB0["ade6"].mean()) - float(zB1["ade6"].mean()))
    exp = {"F": F, "bar": F / 2, "acc": row("X", "B", "accel"), "keep": row("X", "B", "ade6")}
    exp["keep"]["pass"] = bool(exp["keep"]["hi"] <= F / 2)
    exp["acc"]["pass"] = bool(exp["acc"]["mean"] < 0 and exp["acc"]["separated"])
    exp["dom_rows"] = {nm: row("HX", "P10", key) for nm, key, _ in _DOM_SPEC}
    # CONSISTENCY(a) 'any': the eval's own flags of the HX plan against the P10 flags of the floors file (a lower violation rate is better)
    exX = H.extras_of(zX, D)
    _, pwHX = ev.evaluate(arms["HX"], Xc, exX, nb, with_families=True)
    _, pwXe = ev.evaluate(arms["X"], Xc, exX, nb, with_families=True)
    v = {"B": zB["viol_any"].astype(float), "X": pwXe["viol_any"].astype(float), "HX": pwHX["viol_any"].astype(float), "P10": fl["P10_refit__viol_any"].astype(float)}
    exp["viol"] = v
    exp["cons"] = BE.paired_row(v["HX"], v["P10"], eid, nb)
    exp["cons_decoy_x"] = BE.paired_row(v["X"], v["P10"], eid, nb)
    # distance keeping: the programme's reader, on the arms the PREREG names (K = 30), the same inputs the reader gives it
    order = list(grid["order"])
    preds = {f"B_s{s}": arms["B"], f"X_s{s}": arms["X"], f"HX_s{s}": arms["HX"], "P10": P10}
    dk = DK.read_dk(preds=preds, gt=gt, eid=eid, clip=np.array([order[e] for e in eid]), frame=2 * np.asarray(Xc["t"], np.int64), v0=np.asarray(Xc["kin"][:, 0], np.float64),
                    block=DK.load_block(str(HBIN / "b1_eval_lead_block_k30.npz")), k=30, dt=0.2, n_boot=nb, seed=0, saved_ade6={f"B_s{s}": zB["ade6"], f"X_s{s}": zX["ade6"]},
                    p10_floor_ade6=fl["P10_refit__ade6"], pairs=[(f"HX_s{s}", "P10"), (f"X_s{s}", "P10")])
    pairs = DK._clean(dk["read"])["pairs"]

    def dk_rows(pr):
        out = {}
        for nm, r, hi_worse in (("headway min (m)", pr["dk"]["headway_min_m"], False), ("time-gap min (s)", pr["dk"]["time_gap_min_s"], False), ("LEAD-VIOL at-risk share", pr["lead_viol"]["at_risk"], True)):
            d, sep = r.get("delta"), bool(r.get("separated"))
            out[nm] = {"mean": d, "lo": r.get("lo"), "hi": r.get("hi"), "separated": sep, "worse": bool(sep and (d > 0 if hi_worse else d < 0)), "better": bool(sep and (d < 0 if hi_worse else d > 0))}
        return out
    exp["dk"], exp["dk_decoy_x"] = dk_rows(pairs[f"HX_s{s}__minus__P10"]), dk_rows(pairs[f"X_s{s}__minus__P10"])
    # the verdicts, from the PREREG's rule
    ok_better = all(exp["dom_rows"][nm]["better"] for nm, _, need in _DOM_SPEC if need == "better")
    ok_nw = (all(not exp["dom_rows"][nm]["worse"] for nm, _, need in _DOM_SPEC if need == "not worse") and not exp["cons"]["worse"] and all(not r["worse"] for r in exp["dk"].values()))
    exp["dom_pass"] = bool(ok_better and ok_nw)
    exp["state"] = {"ACC": exp["acc"]["pass"], "KEEP": exp["keep"]["pass"], "DOM": exp["dom_pass"]}
    exp["pw"], exp["row"] = pw, row
    return exp


def _same_row(got, want, what, tol=1e-4):
    for k in ("mean", "lo", "hi"):
        assert got[k] == pytest.approx(want[k], abs=tol), f"{what}: {k} reads {got[k]}, the oracle computes {want[k]}"
    for k in ("separated", "better", "worse"):
        if k in want and k in got:
            assert got[k] == want[k], f"{what}: {k} reads {got[k]}, the oracle computes {want[k]}"


def _distinct(right, decoys, what, margin=4e-4):
    """The WORLD's own precondition: the right read differs from every wrong read by more than the tolerance of the comparison, or the world could not tell a mutant that reads the wrong thing."""
    for nm, d in decoys.items():
        assert abs(right - d) > margin, f"{what}: the world does not separate the right read ({right:+.5f}) from the decoy '{nm}' ({d:+.5f})"


def _assert_world(fab, base, r, s, nb, decoys=True):
    """The reader's seed-``s`` record against the oracle, row by row, and (``decoys``) the world's own proof that no wrong read would give the same number."""
    exp, got = _oracle_seed(fab, base, s, nb), r["per_seed"][f"s{s}"]
    row = exp["row"]
    assert r["F"] == pytest.approx(exp["F"], abs=1e-9) and r["keep_bar"] == pytest.approx(exp["F"] / 2, abs=1e-9) and abs(exp["F"] - 0.1207) < 5e-4
    # PRIMARY-ACC = accel MAE X - B (PREREG); PRIMARY-KEEP = ADE6 X - B, upper bound against F/2; DOMINANCE = HX - P10 on HSPEC's PRIMARY-DOM row
    _same_row(got["PRIMARY_ACC"], exp["acc"], f"s{s} PRIMARY-ACC")
    assert got["PRIMARY_ACC"]["pass"] is exp["acc"]["pass"]
    _same_row(got["PRIMARY_KEEP"], exp["keep"], f"s{s} PRIMARY-KEEP")
    assert got["PRIMARY_KEEP"]["bar_hi_le"] == pytest.approx(exp["bar"], abs=1e-9) and got["PRIMARY_KEEP"]["pass"] is exp["keep"]["pass"]
    assert got["PRIMARY_KEEP"]["hi_le_printed_0.060"] is bool(exp["keep"]["hi"] <= 0.060)
    dom = got["DOMINANCE"]
    for nm, key, need in _DOM_SPEC:
        _same_row(dom["rows"][nm], exp["dom_rows"][nm], f"s{s} DOMINANCE {nm}")
        assert dom["rows"][nm]["need"] == ("separated-better" if need == "better" else "not separated-worse")
    _same_row(dom["rows"]["CONSISTENCY(a) 'any'"], exp["cons"], f"s{s} DOMINANCE CONSISTENCY(a) 'any'")
    for nm, o in exp["dk"].items():
        g = dom["rows"][f"DK {nm}"]
        assert g["mean"] == pytest.approx(o["mean"], abs=1e-4) and g["separated"] == o["separated"] and g["worse"] == o["worse"] and g["better"] == o["better"], (nm, g, o)
    assert dom["pass"] is exp["dom_pass"] and got["state"] == exp["state"]
    assert len(dom["rows"]) == 4 + 7 + 1 + 3
    # beside: ADE / FDE rows for X - B, HX - B, HX - P10; the per-arm consistency rates; the families' X - B / HX - B / HX - X
    for nm, key in (("ADE2", "ade2"), ("ADE4", "ade4"), ("ADE6", "ade6"), ("FDE6", "fde6")):
        for lab, (a, b) in (("X_minus_B", ("X", "B")), ("HX_minus_B", ("HX", "B")), ("HX_minus_P10", ("HX", "P10"))):
            _same_row(got["beside"]["ADE"][nm][lab], row(a, b, key), f"s{s} beside {nm} {lab}")
    rate = got["beside"]["consistency_a"]["rate"]
    for arm in ("B", "X", "HX", "P10"):
        assert rate[arm]["viol_any"] == pytest.approx(float(exp["viol"][arm].mean()), abs=1e-9), arm
    _same_row(got["beside"]["consistency_a"]["HX_minus_P10_any"], exp["cons"], f"s{s} beside CONSISTENCY")
    fp = got["beside"]["families"]["longitudinal_lateral_paired_per_window"]
    for lab, (a, b) in (("X_minus_B", ("X", "B")), ("HX_minus_B", ("HX", "B")), ("HX_minus_X", ("HX", "X"))):
        for fk, key in (("fw_lon_speed_mae", "speed"), ("fw_lon_accel_mae", "accel"), ("fw_lat_curvature_mae_1pm", "curv"), ("fw_tac_lat_acc", "tac_lat"), ("fw_tac_lon_acc", "tac_lon")):
            _same_row(fp[lab][fk], row(a, b, key), f"s{s} families {lab} {fk}")
    if decoys:
        acc, keep = exp["acc"]["mean"], exp["keep"]["mean"]
        _distinct(acc, {"speed MAE X-B": row("X", "B", "speed")["mean"], "accel B-X": row("B", "X", "accel")["mean"], "accel HX-B": row("HX", "B", "accel")["mean"], "accel X-P10": row("X", "P10", "accel")["mean"],
                        "accel HX-P10": row("HX", "P10", "accel")["mean"], "accel HX-X": row("HX", "X", "accel")["mean"]}, f"s{s} ACC", margin=1e-3)
        _distinct(keep, {"ADE2 X-B": row("X", "B", "ade2")["mean"], "ADE4 X-B": row("X", "B", "ade4")["mean"], "FDE6 X-B": row("X", "B", "fde6")["mean"], "ADE6 B-X": row("B", "X", "ade6")["mean"],
                         "ADE6 HX-B": row("HX", "B", "ade6")["mean"], "ADE6 X-P10": row("X", "P10", "ade6")["mean"], "ADE6 HX-P10": row("HX", "P10", "ade6")["mean"]}, f"s{s} KEEP", margin=1e-3)
        for nm, key, _ in _DOM_SPEC:
            _distinct(exp["dom_rows"][nm]["mean"], {"HX-B": row("HX", "B", key)["mean"], "X-P10": row("X", "P10", key)["mean"], "X-B": row("X", "B", key)["mean"], "B-P10": row("B", "P10", key)["mean"]}, f"s{s} DOM {nm}")
        _distinct(exp["cons"]["mean"], {"consistency X-P10": exp["cons_decoy_x"]["mean"]}, f"s{s} CONSISTENCY", margin=2e-3)
        assert any(abs(exp["dk"][nm]["mean"] - exp["dk_decoy_x"][nm]["mean"]) > 1e-2 for nm in exp["dk"]), f"s{s}: the world does not separate HX-P10 from X-P10 on distance keeping"
    return exp


def _nan_none(o):
    """The reader writes strict JSON (NaN -> null); a banked eval json carries NaN literals: compare them as the same absence."""
    if isinstance(o, dict):
        return {k: _nan_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_none(v) for v in o]
    return None if isinstance(o, float) and not math.isfinite(o) else o


def _prereg_next(name):
    """The PREREG's committed next step for an outcome, from the PREREG file's own table."""
    for ln in PREREG.read_text(encoding="utf-8").splitlines():
        if ln.startswith("| **") and ln.count("|") >= 4:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if cells[0].strip("*").strip() == name:
                return cells[2]
    pytest.fail(f"the PREREG has no outcome row {name!r}")


_W_ALONG, _W_CROSS = np.linspace(0.2, 0.9, 30), np.linspace(0.1, 0.6, 30)      # distinct per axis and per step: a reader that swaps the axes or the arm reads another number
_P10_VIOL = (np.arange(2399) % 4 == 0)                                         # P10's consistency rate is 0.25 by construction


def _world(fab, name, tau_fn, **kw):
    kw.setdefault("w_hx", _W_ALONG)
    kw.setdefault("w_cross", _W_CROSS)
    kw.setdefault("p10_viol_any", _P10_VIOL)
    return _fab_arm(fab, name, tau_fn, **kw)


def _tau_near(fab):
    gt = fab["world"][2]["gt"].astype(np.float64)
    k = np.arange(1, 31)[None, :, None]
    eps = np.concatenate([0.05 * np.sin(0.3 * k) * np.ones((1, 1, 1)), 0.05 * np.cos(0.2 * k)], 2)
    return lambda s, tau, p: gt + eps


def _tau_costly(s, tau, p):
    return _smooth(tau, 4) + np.array([0.0, 1.5]) * (np.arange(1, 31)[None, :, None] / 30.0)      # smooth, plus a lateral drift growing to 1.5 m at 6 s


@needs_reader
@needs_real
def test_oacc_reader_oracle_is_the_programmes_family_windows_on_the_banked_arms(fab):
    """The control of the oracle itself: its independent ADE / speed / accel / along / cross / heading / yaw-rate / curvature agree with `family_windows` to float32 noise on B and on P10, so a
    comparison with it means something. (A wrong oracle would pass a wrong reader.)"""
    I, (grid, J, Xc) = fab["I"], fab["world"]
    gt = Xc["gt"].astype(np.float64)
    P10 = np.load(HBIN / "b2_p10_eval_pred6.npy").astype(np.float64)
    for pred in (np.load(RAWF / "eval_M2b_s0.npz")["pred"].astype(np.float64), P10):
        om, lib = _o_metrics(pred, gt), I.H.arm_pw(pred, gt, I.BE)
        for ok, lk in (("ade2", "ade2"), ("ade4", "ade4"), ("ade6", "ade6"), ("fde6", "fde6"), ("speed", "fw_lon_speed_mae"), ("accel", "fw_lon_accel_mae"), ("along", "fw_lon_along_mae"),
                       ("cross", "fw_lat_cross_mae"), ("heading", "fw_lat_heading_mae_deg"), ("yaw", "fw_lat_yaw_rate_mae_degps"), ("curv", "fw_lat_curvature_mae_1pm")):
            a, b = om[ok], lib[lk]
            assert np.array_equal(np.isfinite(a), np.isfinite(b)), ok
            m = np.isfinite(a)
            assert float(a[m].mean()) == pytest.approx(float(b[m].mean()), rel=1e-4, abs=1e-6), ok
            assert float(np.abs(a[m] - b[m]).max()) < 5e-3 * max(1.0, float(np.abs(b[m]).max())), ok
    assert _o_hx(P10, P10 + 1.0, np.zeros(30), np.zeros(30)).tolist() == P10.tolist() and np.allclose(_o_hx(P10, P10 + 1.0, np.ones(30), np.ones(30)), P10 + 1.0), "HX = P at w 0, tau at w 1"
    z = _o_hx(np.zeros((1, 30, 2)), np.ones((1, 30, 2)), np.arange(30) / 100.0, np.arange(30) / 10.0)
    assert z[0, 7, 0] == pytest.approx(0.07) and z[0, 7, 1] == pytest.approx(0.7), "x takes the along weight, y the cross weight, per step"


@needs_reader
@needs_real
def test_oacc_reader_world_x_equal_to_b_reads_no_effect_and_every_value_is_the_oracles(fab, tmp_path, capsys):
    base = _world(fab, "same", lambda s, tau, p: tau)
    rc, err, r, out = _read(base, "0", tmp_path, capsys)
    assert rc == 0, err
    assert "s1_required" in r, "the record says whether seed 1 is required"
    assert r["OUTCOME"] == "NO EFFECT" and r["s1_required"] is False and r["seeds_read"] == [0]
    s0 = r["per_seed"]["s0"]
    assert s0["PRIMARY_ACC"]["mean"] == 0.0 and s0["PRIMARY_ACC"]["separated"] is False and s0["PRIMARY_ACC"]["pass"] is False, "X == B: the paired accel difference is exactly 0 and not separated"
    assert s0["PRIMARY_KEEP"]["mean"] == 0.0 and s0["PRIMARY_KEEP"]["pass"] is True and s0["state"] == {"ACC": False, "KEEP": True, "DOM": s0["DOMINANCE"]["pass"]}
    exp = _assert_world(fab, base, r, 0, 40, decoys=False)
    assert exp["acc"]["mean"] == 0.0
    assert abs(r["F"] - 0.1207) < 5e-4 and r["keep_bar"] == r["F"] / 2 and r["n_windows"] == 2399 and r["n_episodes"] == 141 and r["w_acc"] == 42.9311 and r["estimator"].startswith("paired episode-cluster bootstrap, B=40")
    assert s0["identity_controls"]["all_pass"] is True and s0["identity_controls"]["w1_vs_X_pred_maxabs_m"] <= 1e-4 and s0["identity_controls"]["P_vs_P10_file_maxabs_m"] <= 1e-4
    fams = s0["beside"]["families"]
    bjB = json.loads((RAWF / "eval_M2b_s0.json").read_text(encoding="utf-8"))["raw"]
    assert fams["strategic"]["status"].startswith("READ (switched on") and "PAUSED" not in json.dumps(r)
    assert fams["strategic"]["B"] == _nan_none(bjB["strategic"]) and fams["strategic"]["X"] == _nan_none(bjB["strategic"]) and fams["strategic"]["B"] is not None, "the strategic family is READ from both evals"
    assert fams["tactical"]["B"] == _nan_none(bjB["tactical"]) and fams["tactical"]["X"] is not None
    assert fams["strategic"]["identical_to_B"] is True and fams["strategic"]["identical_by_construction"] is False and fams["tactical"]["HX_equals_X_by_construction"] is True
    assert set(fams["distance_keeping_k30"]["pairs"]) >= {"HX_s0__minus__P10", "X_s0__minus__B_s0"}
    assert r["checkpoint_sha256"].get("M2b_s0") == RD.BANKED_CKPT_SHA["M2b_s0"] and r["checkpoint_sha256"].get("M2b-oacc_s0") == s0["ckpt_sha256"], "the record names the measured checkpoints"
    want_next = f"NO EFFECT: {_prereg_next('NO EFFECT')}."
    assert r["next"] == want_next and r["reading"].startswith("PREREG_V8F_OACC: NO EFFECT; w_acc 42.9311;"), r["reading"]
    shown = json.loads(out)
    assert shown["OUTCOME"] == "NO EFFECT" and shown["s1_required"] is False and shown["next"] == want_next and shown["reading"] == r["reading"], "what the CLI prints is the record's own headline"
    shutil.rmtree(base, ignore_errors=True)


@needs_reader
@needs_real
def test_oacc_reader_world_costly_x_passes_acc_fails_keep_and_reads_costs_position(fab, tmp_path, capsys):
    base = _world(fab, "costly", _tau_costly)
    rc, err, r, _ = _read(base, "0", tmp_path, capsys)
    assert rc == 0, err
    exp = _assert_world(fab, base, r, 0, 40)
    s0 = r["per_seed"]["s0"]
    assert exp["state"]["ACC"] is True and exp["state"]["KEEP"] is False and s0["PRIMARY_KEEP"]["hi"] > exp["bar"] + 0.1, "a lateral drift growing to 1.5 m: ADE6 worse by far more than F/2"
    assert r["OUTCOME"] == "COSTS POSITION" and r["s1_required"] is False and r["next"] == f"COSTS POSITION: {_prereg_next('COSTS POSITION')}." and r["reading"].startswith("PREREG_V8F_OACC: COSTS POSITION; w_acc 42.9311;")
    shutil.rmtree(base, ignore_errors=True)


@needs_reader
@needs_real
def test_oacc_reader_world_smoothed_x_every_read_is_the_hand_computed_one_pending_seed_1_then_smoother_not_dominant(fab, tmp_path, capsys):
    base = _world(fab, "smooth", lambda s, tau, p: _smooth(tau, 4))
    rc, err, r0, _ = _read(base, "0", tmp_path, capsys, name="r0.json")
    assert rc == 0, err
    exp0 = _assert_world(fab, base, r0, 0, 40)
    assert exp0["state"] == {"ACC": True, "KEEP": True, "DOM": False} and exp0["acc"]["mean"] < 0 and exp0["acc"]["separated"] and exp0["keep"]["hi"] <= exp0["bar"], "the world: ACC and KEEP pass, DOMINANCE fails"
    assert exp0["dom_rows"]["ADE2"]["better"] is False and exp0["dom_rows"]["curvature MAE"]["worse"] is True and exp0["dom_rows"]["ADE6"]["better"] is True and exp0["dom_rows"]["accel MAE"]["better"] is True, "ADE2 is not separated-better and curvature is separated-WORSE"
    assert r0["OUTCOME"] == "PENDING SEED 1" and r0["s1_required"] is True and r0["next"] is None and r0["seeds_read"] == [0] and list(r0["per_seed"]) == ["s0"]
    assert r0["reading"].startswith("PREREG_V8F_OACC: PENDING SEED 1; w_acc 42.9311;") and "DOMINANCE HX - P10 FAIL" in r0["reading"]
    rc, err, r, out = _read(base, "0,1", tmp_path, capsys, name="r01.json")
    assert rc == 0, err
    exp1 = _assert_world(fab, base, r, 1, 40)
    _assert_world(fab, base, r, 0, 40, decoys=False)
    assert exp1["state"]["ACC"] is True and exp1["state"]["KEEP"] is True and exp1["state"]["DOM"] is False
    assert r["seeds_read"] == [0, 1] and r["seed_states"] == {"s0": exp0["state"], "s1": exp1["state"]} and r["s1_required"] is False
    assert r["OUTCOME"] == "SMOOTHER, NOT DOMINANT" and r["next"] == f"SMOOTHER, NOT DOMINANT: {_prereg_next('SMOOTHER, NOT DOMINANT')}."
    assert r["reading"].startswith("PREREG_V8F_OACC: SMOOTHER, NOT DOMINANT; w_acc 42.9311;") and "DOMINANCE HX - P10 FAIL / FAIL" in r["reading"]
    assert json.loads(out)["OUTCOME"] == "SMOOTHER, NOT DOMINANT"
    shutil.rmtree(base, ignore_errors=True)


@needs_reader
@needs_real
def test_oacc_reader_world_near_gt_x_dominates_its_prior_at_both_seeds(fab, tmp_path, capsys):
    base = _world(fab, "near", _tau_near(fab))
    rc, err, r, _ = _read(base, "0,1", tmp_path, capsys)
    assert rc == 0, err
    exp0 = _assert_world(fab, base, r, 0, 40)
    exp1 = _assert_world(fab, base, r, 1, 40)
    assert exp0["state"] == {"ACC": True, "KEEP": True, "DOM": True} and exp1["state"] == {"ACC": True, "KEEP": True, "DOM": True}
    assert all(exp0["dom_rows"][nm]["better"] for nm, _, need in _DOM_SPEC if need == "better") and not any(r_["worse"] for r_ in exp0["dom_rows"].values()), "every DOMINANCE row is separated-better in this world"
    assert r["OUTCOME"] == "DOMINATES ITS PRIOR" and r["s1_required"] is False and r["seed_states"] == {"s0": exp0["state"], "s1": exp1["state"]}
    assert r["next"] == f"DOMINATES ITS PRIOR: {_prereg_next('DOMINATES ITS PRIOR')}." and r["reading"].startswith("PREREG_V8F_OACC: DOMINATES ITS PRIOR; w_acc 42.9311;") and "DOMINANCE HX - P10 PASS / PASS" in r["reading"]
    shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# THE READER'S INPUT CHECKS: each is a file (or an argument) edited on a fabricated arm that otherwise READS; the reader must refuse with rc 2 AND with THAT check's own message, and write nothing.
# (An rc of 2 alone proves nothing: `main` turns ANY KeyError / ValueError into rc 2, so a check that was removed can still end in rc 2 two lines later. The message names the check.)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _jedit(path, fn):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    fn(d)
    Path(path).write_text(json.dumps(d), encoding="utf-8")


def _zedit(path, fn):
    z = dict(np.load(path))
    fn(z)
    np.savez_compressed(path, **z)


def _rewrite_ckpt(b, s=0, **ch):
    """Rewrite X's checkpoint with changed fields and keep EVERY sha256 record consistent (box 2's json, the HX weights), so that the ONLY thing wrong is the field."""
    import torch
    p = b / "runs" / RD.x_dir(s) / "ckpt_last.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for k, v in ch.items():
        if k == "w_acc":
            ck["cfg"] = {**ck["cfg"], "w_acc": v}
        else:
            ck[k] = v
    torch.save(ck, p)
    sha = RD.sha256_file(p)
    _jedit(b / "ckpt_sha256.json", lambda d: d.update({f"{RD.x_tag(s)}_trained": sha, f"{RD.x_tag(s)}_after_evals": sha}))
    _jedit(b / "w" / f"weights_{RD.x_tag(s)}.json", lambda d: d.update({"ckpt_sha256": sha}))


def _p10_scaled_to(tmp_path, ade6_target, I, world):
    """A P10 prediction file whose ADE6 on the eval grid is exactly ``ade6_target`` (the errors scale: plan = GT + c (P10 - GT))."""
    gt = world[2]["gt"].astype(np.float64)
    P10 = np.load(HBIN / "b2_p10_eval_pred6.npy").astype(np.float64)
    now = float(np.linalg.norm(P10 - gt, axis=-1).mean())
    q = tmp_path / "p10_scaled.npy"
    np.save(q, gt + (P10 - gt) * (ade6_target / now))
    return q


def _eval_leaf(raw, path):
    for k in path[:-1]:
        raw = raw[k]
    raw[path[-1]] += 0.05


def _hx_w(key, mut):
    def f(b, tp):
        p = b / "w" / "weights_M2b-oacc_s0.json"
        _jedit(p, lambda d: d[key].__setitem__(*mut) if isinstance(mut, tuple) else d.update({key: mut}))
    return f


def _argv_add(b, tp):
    _jedit(b / "runs" / "M2b_s0_oacc" / "argv.json", lambda d: d["argv"].extend(["--code-mix", "kind"]))


_REFUSALS = {
    # --- the checkpoint sha256 chain: B's two banked checkpoints, one value from the end of training to the evals, and the file pulled here
    "banked_s0_checkpoint_is_not_the_banked_one": (lambda b, tp: _jedit(b / "ckpt_sha256.json", lambda d: d.update({"M2b_s0": "0" * 64})), "the checkpoint M2b_s0 has sha256 0000000000000000", "0", ()),
    "banked_s1_checkpoint_is_not_the_banked_one": (lambda b, tp: _jedit(b / "ckpt_sha256.json", lambda d: d.update({"M2b_s1": "0" * 64})), "the checkpoint M2b_s1 has sha256 0000000000000000", "0", ()),
    "banked_s1_checkpoint_is_not_the_banked_one_at_seeds_0_1": (lambda b, tp: _jedit(b / "ckpt_sha256.json", lambda d: d.update({"M2b_s1": "0" * 64})), "the checkpoint M2b_s1 has sha256 0000000000000000", "0,1", ()),
    "checkpoint_moved_between_training_and_the_evals": (lambda b, tp: _jedit(b / "ckpt_sha256.json", lambda d: d.update({"M2b-oacc_s0_after_evals": "1" * 64})), "at the end of training", "0", ()),
    "checkpoint_sha_at_training_end_is_missing": (lambda b, tp: _jedit(b / "ckpt_sha256.json", lambda d: d.pop("M2b-oacc_s0_trained")), "at the end of training", "0", ()),
    "pulled_checkpoint_is_not_the_one_box_2_measured": (lambda b, tp: _jedit(b / "ckpt_sha256.json", lambda d: d.update({"M2b-oacc_s0_trained": "3" * 64, "M2b-oacc_s0_after_evals": "3" * 64})), "the checkpoint pulled into", "0", ()),
    "ckpt_sha_json_is_missing": (lambda b, tp: None, "--ckpt-sha is required and must exist", "0", lambda b, tp: ("--ckpt-sha", str(tp / "no_such.json"))),
    # --- the run records
    "second_lever_on_x_argv": (_argv_add, "besides --w-acc and --out", "0", ()),
    "x_selfcheck_missing": (lambda b, tp: (b / "runs" / "M2b_s0_oacc" / "selfcheck.json").unlink(), "selfcheck.json is missing (the run did not finish", "0", ()),
    "x_checkpoint_missing": (lambda b, tp: (b / "runs" / "M2b_s0_oacc" / "ckpt_last.pt").unlink(), "ckpt_last.pt", "0", ()),
    "wacc_record_is_zero": (lambda b, tp: (b / "w_acc.json").write_text(json.dumps({"schema": "v8f-oacc-wacc-v1", "w_acc": 0})), "not a v8f-oacc-wacc-v1 record with w_acc > 0", "0", ()),
    "wacc_record_is_negative": (lambda b, tp: (b / "w_acc.json").write_text(json.dumps({"schema": "v8f-oacc-wacc-v1", "w_acc": -42.9311})), "not a v8f-oacc-wacc-v1 record with w_acc > 0", "0", ()),
    "wacc_record_is_a_string": (lambda b, tp: (b / "w_acc.json").write_text(json.dumps({"schema": "v8f-oacc-wacc-v1", "w_acc": "42.9311"})), "not a v8f-oacc-wacc-v1 record with w_acc > 0", "0", ()),
    "wacc_record_has_another_schema": (lambda b, tp: (b / "w_acc.json").write_text(json.dumps({"schema": "v8f-oacc-wacc-v0", "w_acc": 42.9311})), "not a v8f-oacc-wacc-v1 record with w_acc > 0", "0", ()),
    "wacc_record_disagrees_with_the_run": (lambda b, tp: (b / "w_acc.json").write_text(json.dumps({"schema": "v8f-oacc-wacc-v1", "w_acc": 41.0})), "expected 41.0 exactly once", "0", ()),
    # --- the evals: which checkpoint, which interface, which grid
    "eval_scored_the_banked_b_checkpoint": (lambda b, tp: _jedit(b / "eval" / "eval_M2b-oacc_s0.json", lambda d: d.update({"ckpt": "D:/tanitad/runs/v8f_mini_full/M2b_s0/ckpt_last.pt"})), "not <runs>/M2b_s0_oacc/ckpt_last.pt", "0", ()),
    "eval_built_another_interface": (lambda b, tp: _jedit(b / "eval" / "eval_M2b-oacc_s0.json", lambda d: d.update({"o_interface": "cut"})), "the eval built o_interface 'cut'", "0", ()),
    "eval_is_not_on_the_committed_grid": (lambda b, tp: _zedit(b / "eval" / "eval_M2b-oacc_s0.npz", lambda z: z.update({"eid": z["eid"][:-1]})), "the evals are not on the committed eval grid", "0", ()),
    "committed_window_count_other": (lambda b, tp: None, "the evals hold 2399 windows over 141 episodes, the committed eval-141 grid has 2400 over 141", "0", ("--n-windows", "2400")),
    "committed_episode_count_other": (lambda b, tp: None, "the evals hold 2399 windows over 141 episodes, the committed eval-141 grid has 2399 over 140", "0", ("--n-episodes", "140")),
    "f_is_not_the_banked_one": (lambda b, tp: _zedit(b / "eval" / "eval_M2b_s1.npz", lambda z: z.update({"ade6": z["ade6"] + 0.3})), "F = |ADE6(B_s0) - ADE6(B_s1)| reads", "0", ()),
    "f_just_outside_its_tolerance": (lambda b, tp: _zedit(b / "eval" / "eval_M2b_s1.npz", lambda z: z.update({"ade6": z["ade6"] + 0.0008})), "F = |ADE6(B_s0) - ADE6(B_s1)| reads", "0", ()),
    # --- the identity controls and the HX route
    "prior_of_the_checkpoint_is_not_the_p10_file": (lambda b, tp: None, "identity controls failed", "0", lambda b, tp: ("--p10-pred", str(_p10_scaled_to(tp, 2.4870 + 0.02, None, _P10W[0])))),
    "hx_at_w_1_does_not_reproduce_the_x_block": (lambda b, tp: _jedit(b / "eval" / "eval_M2b-oacc_s0.json", lambda d: _eval_leaf(d["raw"], ["families", "longitudinal", "accel_mae_mps2"])), "HX at w = 1 does not reproduce X's own eval block", "0", ()),
    "hx_weights_file_missing": (lambda b, tp: (b / "w" / "weights_M2b-oacc_s0.json").unlink(), "cannot read", "0", ()),
    "hx_weights_fitted_on_another_checkpoint": (lambda b, tp: _hx_w("ckpt_sha256", "2" * 64)(b, tp), "the HX weights were fitted on checkpoint 2222222222222222", "0", ()),
    "hx_weights_checkpoint_built_with_another_w_acc": (lambda b, tp: _hx_w("w_acc_built", 41.0)(b, tp), "was built with w_acc 41.0, the rule's value is 42.9311", "0", ()),
    "hx_weights_not_monotone": (lambda b, tp: _hx_w("w_along_by_step", (5, 0.1))(b, tp), "the HX weights are not in [0, 1] and monotone non-decreasing", "0", ()),
    "hx_weights_above_one": (lambda b, tp: _hx_w("w_cross_by_step", (29, 1.2))(b, tp), "the HX weights are not in [0, 1] and monotone non-decreasing", "0", ()),
    "hx_weights_negative": (lambda b, tp: _hx_w("w_along_by_step", (0, -0.1))(b, tp), "the HX weights are not in [0, 1] and monotone non-decreasing", "0", ()),
    "hx_weights_have_29_steps": (lambda b, tp: (_hx_w("w_along_by_step", [0.5] * 29)(b, tp), _hx_w("w_cross_by_step", [0.5] * 29)(b, tp)), "the HX weights are not in [0, 1] and monotone non-decreasing", "0", ()),
    "hx_weights_have_one_axis_of_29_steps": (lambda b, tp: _hx_w("w_along_by_step", [0.5] * 29)(b, tp), "all input arrays must have the same shape", "0", ()),
    # --- the checkpoint's own cfg / argv_sha / step / arm / seed (consistent with every sha record: only the field is wrong)
    "checkpoint_cfg_w_acc_is_not_the_rules": (lambda b, tp: _rewrite_ckpt(b, 0, w_acc=0.0), "the checkpoint says w_acc 0.0", "0", ()),
    "checkpoint_argv_sha_is_not_the_runs": (lambda b, tp: _rewrite_ckpt(b, 0, argv_sha="zz"), "argv_sha zz", "0", ()),
    "checkpoint_step_is_not_the_runs": (lambda b, tp: _rewrite_ckpt(b, 0, step=100), "step 100", "0", ()),
    "checkpoint_arm_is_not_m2b": (lambda b, tp: _rewrite_ckpt(b, 0, arm="M2a"), "arm M2a", "0", ()),
    "checkpoint_seed_is_not_the_runs": (lambda b, tp: _rewrite_ckpt(b, 0, seed=1), "seed 1", "0", ()),
    # --- a missing family is refused, never dropped
    "x_eval_has_no_strategic_family": (lambda b, tp: _jedit(b / "eval" / "eval_M2b-oacc_s0.json", lambda d: d["raw"].pop("strategic")), "X's eval json has no strategic family", "0", ()),
    "x_eval_has_no_tactical_family_the_w_1_identity_refuses_it_first": (lambda b, tp: _jedit(b / "eval" / "eval_M2b-oacc_s0.json", lambda d: d["raw"].pop("tactical")), "HX at w = 1 does not reproduce X's own eval block", "0", ()),
    "b_eval_has_no_strategic_family": (lambda b, tp: _jedit(b / "eval" / "eval_M2b_s0.json", lambda d: d["raw"].pop("strategic")), "B's eval json has no strategic family", "0", ()),
    "b_eval_has_no_tactical_family": (lambda b, tp: _jedit(b / "eval" / "eval_M2b_s0.json", lambda d: d["raw"].pop("tactical")), "B's eval json has no tactical family", "0", ()),
    "seeds_other_than_0_or_0_1": (lambda b, tp: None, "--seeds '1': expected 0 or 0,1", "1", ()),
    "seeds_three": (lambda b, tp: None, "--seeds '0,1,2': expected 0 or 0,1", "0,1,2", ()),
}
_P10W = [None]


@pytest.fixture(scope="module")
def refbase(fab):
    """ONE fabricated arm that reads (X smooth, seeds 0 and 1); each refusal case copies it and edits ONE file."""
    _P10W[0] = fab["world"]
    return _world(fab, "ref", lambda s, tau, p: _smooth(tau, 4))


@needs_reader
@needs_real
@pytest.mark.parametrize("case", list(_REFUSALS))
def test_oacc_reader_refuses_with_the_checks_own_message_and_writes_nothing(fab, refbase, tmp_path, capsys, case):
    edit, want, seeds, extra = _REFUSALS[case]
    b = tmp_path / "arm"
    shutil.copytree(refbase, b)
    edit(b, tmp_path)
    ex = extra(b, tmp_path) if callable(extra) else extra
    rc, err, r, out = _read(b, seeds, tmp_path, capsys, extra=ex)
    assert rc == 2 and r is None and out == "", f"{case}: rc {rc}, stdout {out[:200]!r}, stderr {err[:300]!r}"
    assert not (tmp_path / "r.json").exists(), "a refused read writes no record"
    assert err.startswith(("OACC READ REFUSED:", "OACC READ INPUT ERROR:")) and want in err, f"{case}: expected a refusal containing {want!r}, got {err[:400]!r}"


@needs_reader
@needs_real
def test_oacc_reader_refuses_a_hx_evaluation_without_a_family(fab, refbase, tmp_path, capsys, monkeypatch):
    """The HX plan's own evaluate block must carry the longitudinal AND the lateral family (a missing family is refused, never dropped): the eval is wrapped to hand back a block without one on the HX call."""
    import types
    base_cls = RD.Instruments
    for fam in ("longitudinal", "lateral"):
        class _I2(base_cls):
            def __init__(self, fam=fam):
                super().__init__()
                orig, calls = self.ev.evaluate, []

                def evaluate(*a, **k):
                    res, pw = orig(*a, **k)
                    calls.append(1)
                    if len(calls) == 2:                                          # tauX, then HX, then HX at w = 1
                        res = dict(res, families={kk: v for kk, v in res["families"].items() if kk != fam})
                    return res, pw
                self.ev = types.SimpleNamespace(evaluate=evaluate)
        monkeypatch.setattr(RD, "Instruments", _I2)
        rc, err, r, out = _read(refbase, "0", tmp_path, capsys)
        assert rc == 2 and r is None and f"HX's evaluate block has no {fam} family" in err, (fam, rc, err[:300])
    monkeypatch.undo()
    rc, err, r, out = _read(refbase, "0", tmp_path, capsys)
    assert rc == 0 and r is not None, err


@needs_reader
@needs_real
def test_oacc_reader_check_inputs_refuses_with_the_checks_own_message(fab, tmp_path, capsys):
    base = _world(fab, "chk2", lambda s, tau, p: tau)
    I, world = fab["I"], fab["world"]
    args = ["--check-inputs", "--eval-dir", str(base / "eval"), "--runs-dir", str(base / "runs"), "--out", str(tmp_path / "c.json"), "--p10-pred", str(HBIN / "b2_p10_eval_pred6.npy")]

    def run(*extra, a=args):
        capsys.readouterr()
        rc = RD.main([*a, *extra])
        return rc, capsys.readouterr().err

    rc, err = run()
    assert rc == 0, err
    ok = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert ok["F"] == pytest.approx(abs(float(np.load(RAWF / "eval_M2b_s0.npz")["ade6"].mean()) - float(np.load(RAWF / "eval_M2b_s1.npz")["ade6"].mean())), abs=1e-9) and ok["keep_bar"] == ok["F"] / 2
    gt = world[2]["gt"].astype(np.float64)
    P10 = np.load(HBIN / "b2_p10_eval_pred6.npy").astype(np.float64)
    assert ok["P10_ade6"] == pytest.approx(float(np.linalg.norm(P10 - gt, axis=-1).mean()), abs=1e-9) and ok["banked_ckpt_sha256"] == RD.BANKED_CKPT_SHA
    for k, s in ((0, 0), (1, 1)):
        zb = np.load(RAWF / f"eval_M2b_s{k}.npz")
        assert ok["B_ade6"][f"s{s}"] == pytest.approx(float(zb["ade6"].mean()), abs=1e-9)
        assert ok["B_accel_mae"][f"s{s}"] == pytest.approx(float(np.nanmean(_o_metrics(zb["pred"], gt)["accel"])), abs=1e-4), "B's accel MAE, from the oracle"
    rc, err = run("--n-windows", "2400")
    assert rc == 2 and "the eval grid holds 2399 windows over 141 episodes, the committed one 2400 over 141" in err, err
    rc, err = run("--n-episodes", "140")
    assert rc == 2 and "the eval grid holds 2399 windows over 141 episodes, the committed one 2399 over 140" in err, err
    # P10's ADE6 is the PREREG's 2.4870 within 1e-3: 0.0005 inside, 0.0015 outside (the plan scales its error exactly)
    inside = _p10_scaled_to(tmp_path, 2.4870 + 0.0005, I, world)
    rc, err = run("--p10-pred", str(inside))
    assert rc == 0, err
    inside2 = _p10_scaled_to(tmp_path, 2.4870 - 0.0005, I, world)
    rc, err = run("--p10-pred", str(inside2))
    assert rc == 0, err
    outside = _p10_scaled_to(tmp_path, 2.4870 + 0.0015, I, world)
    rc, err = run("--p10-pred", str(outside))
    assert rc == 2 and "P10-yaw-4512 reads ADE6 2.4885 on the eval grid, the PREREG's known value is 2.487" in err, err
    outside2 = _p10_scaled_to(tmp_path, 2.4870 - 0.0015, I, world)
    rc, err = run("--p10-pred", str(outside2))
    assert rc == 2 and "P10-yaw-4512 reads ADE6 2.4855 on the eval grid" in err, err
    # B's two evals must sit on the committed grid, and F must be B's
    _zedit(base / "eval" / "eval_M2b_s1.npz", lambda z: z.update({"eid": z["eid"][::-1].copy()}))
    rc, err = run()
    assert rc == 2 and "M2b_s1's eval is not on the committed grid" in err, err
    _zedit(base / "eval" / "eval_M2b_s0.npz", lambda z: z.update({"eid": z["eid"][::-1].copy()}))
    rc, err = run()
    assert rc == 2 and "M2b_s0's eval is not on the committed grid" in err, err
    shutil.copy(RAWF / "eval_M2b_s0.npz", base / "eval"), shutil.copy(RAWF / "eval_M2b_s1.npz", base / "eval")
    _zedit(base / "eval" / "eval_M2b_s1.npz", lambda z: z.update({"ade6": z["ade6"] + 0.3}))
    rc, err = run()
    assert rc == 2 and "F = |ADE6(B_s0) - ADE6(B_s1)| reads" in err, err
    shutil.rmtree(base, ignore_errors=True)


@needs_reader
def test_oacc_reader_check_f_is_the_known_value_to_5e_minus_4_and_not_a_hair_more():
    def code(f):
        try:
            RD.check_F(f)
        except SystemExit as e:
            return e.code
        return None
    assert RD.F_PRINTED == 0.1207 and RD.F_TOL == 5e-4
    for f in (0.1207, 0.1207 + 4.9e-4, 0.1207 - 4.9e-4, 0.12074):
        assert code(f) is None, f
    for f in (0.1207 + 5.1e-4, 0.1207 - 5.1e-4, 0.1207 + 1e-3, 0.1207 - 3e-3, 0.0, 0.5):
        assert code(f) == 2, f
    zb = {"ade6": np.array([2.0, 3.0])}
    assert RD.exact_F(zb, {"ade6": np.array([1.0, 1.5])}) == pytest.approx(1.25) and RD.exact_F({"ade6": np.array([1.0, 1.5])}, zb) == pytest.approx(1.25), "F = |mean ADE6 (s0) - mean ADE6 (s1)|"


@needs_reader
def test_oacc_reader_the_committed_numbers_and_the_cli_defaults_are_the_preregs(monkeypatch, tmp_path):
    """B = 2,000 (both bootstraps), the eval-141 grid 2,399 windows / 141 episodes, F = 0.1207, P10's ADE6 2.4870 to 1e-3, the printed KEEP bar 0.060, the seeds 0,1."""
    import types
    assert (RD.B_BOOT, RD.EVAL_WINDOWS, RD.EVAL_EPISODES, RD.F_PRINTED, RD.F_TOL, RD.P10_ADE6, RD.P10_TOL, RD.PRINTED_KEEP_BAR) == (2000, 2399, 141, 0.1207, 5e-4, 2.4870, 1e-3, 0.060)
    seen = {}
    monkeypatch.setattr(RD, "Instruments", lambda: types.SimpleNamespace(DK=types.SimpleNamespace(_clean=lambda o: o)))
    monkeypatch.setattr(RD, "read_all", lambda a, I: seen.update(a=a) or {"reading": "x", "OUTCOME": "y", "s1_required": False, "next": None})
    assert RD.main(["--eval-dir", "e", "--runs-dir", "r", "--weights-dir", "w", "--wacc-json", "j", "--ckpt-sha", "c", "--out", str(tmp_path / "o.json")]) == 0
    a = seen["a"]
    assert (a.n_boot, a.dk_n_boot, a.n_windows, a.n_episodes, a.seeds) == (2000, 2000, 2399, 141, "0,1")
    assert a.eval_grid.endswith("eval_grid_windows.json") and a.eval_kin.endswith("eval141_kin.npz") and a.eval_join.endswith("join_eval141.npz")
    monkeypatch.setattr(RD, "check_inputs", lambda a, I: seen.update(c=a) or {"F": 0.0, "keep_bar": 0.0, "P10_ade6": 0.0, "n_windows": 0, "n_episodes": 0})
    assert RD.main(["--check-inputs", "--eval-dir", "e", "--runs-dir", "r", "--out", str(tmp_path / "o2.json")]) == 0
    assert (seen["c"].n_windows, seen["c"].n_episodes) == (2399, 141)


@needs_reader
@needs_real
def test_oacc_reader_paired_rows_read_the_tactical_agreements_as_higher_is_better_and_the_rest_as_lower(fab):
    I = fab["I"]
    eid = np.repeat(np.arange(12), 6)
    rng = np.random.default_rng(3)
    b = rng.uniform(0.2, 0.6, eid.size)
    a = b + 0.3 + 0.01 * rng.standard_normal(eid.size)           # a is HIGHER than b in every window, in every episode: separated
    rows = RD.paired_rows(I, {"fw_tac_lat_acc": a, "fw_tac_lon_acc": a, "fw_lon_accel_mae": a, "fw_lat_curvature_mae_1pm": a, "ade6": a}, {"fw_tac_lat_acc": b, "fw_tac_lon_acc": b, "fw_lon_accel_mae": b,
                                                                                                                                         "fw_lat_curvature_mae_1pm": b, "ade6": b}, eid, 200)
    for k in ("fw_tac_lat_acc", "fw_tac_lon_acc"):
        assert rows[k]["separated"] and rows[k]["better"] is True and rows[k]["worse"] is False and rows[k]["lower_is_better"] is False, k
    for k in ("fw_lon_accel_mae", "fw_lat_curvature_mae_1pm", "ade6"):
        assert rows[k]["separated"] and rows[k]["worse"] is True and rows[k]["better"] is False and rows[k]["lower_is_better"] is True, k
    assert rows["ade6"]["mean"] == pytest.approx(float(np.mean(a - b)), abs=1e-3), "the row is mean(first) - mean(second)"


@needs_reader
@needs_real
def test_check_inputs_reads_the_banked_b_side_alone(fab, tmp_path):
    base = _fab_arm(fab, "chk", lambda s, tau, p: tau)
    out = tmp_path / "c.json"
    assert RD.main(["--check-inputs", "--eval-dir", str(base / "eval"), "--runs-dir", str(base / "runs"), "--out", str(out), "--p10-pred", str(HBIN / "b2_p10_eval_pred6.npy")]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["n_windows"] == 2399 and r["n_episodes"] == 141 and abs(r["F"] - 0.1207) < 5e-4 and abs(r["P10_ade6"] - 2.4870) < 1e-3
    assert r["B_accel_mae"]["s0"] > r["B_accel_mae"]["s1"] > 0.5, "the PREREG's table: accel MAE 1.206 (s0), 0.931 (s1) against P10's 0.638"


# =============================================================================================================================================
# (H) THE QUEUE (code/v8f_mini_box2_queue.cmd) AND THE CHAIN (code/box2_ops/chain_v8f_oacc.sh)
# =============================================================================================================================================
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")
needs_pkg = pytest.mark.skipif(not (READ_P.exists() and (PKG / "raw" / "full" / "runs").exists()), reason="the v8F-mini package is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
OSTAGES = ("waccoacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc")
OACC_LABELS = set(OSTAGES) | {"oaccwacc"}
ORC_LABELS = {"orcarm", "gateorc", "trainorc", "evalorc"}           # PREREG_V8F_ORECOVER appended these after OACC's (test_v8f_orc_chain.py pins them); widened here, nothing else
# PREREG_V8F_FLAT's seven labels as they stand at the tip 4f561a2c (sha256[:16] of "\n" + the text up to the next label): this edit appended AFTER them and edited none
FLAT_PINS = {"gateflat": "cec296bd862eedcd", "trainflat": "9e28a04f9c79af1e", "evalflat": "024f6a7e06886322", "evalflat_1": "2f90baa2f624c148", "evalflat_2": "f3c16ba44e719907",
             "evalflat_3": "ea291c4013c2b747", "evalflat_4": "901e965d7b940ca4"}
MIXO_PINS = {"gatemixo": "b2ea5430b698741c", "trainmixo": "419a7db33d684844", "evalmixo": "7acc0975f2aa9ef3", "evalmixo_1": "1cc5381d64b27142", "evalmixo_2": "a70ba66191aed3cf",
             "evalmixo_3": "5d326de83e5d37b1", "evalmixo_4": "0cde88e556e0b0e4", "evalmixo_5": "479e39bc5b10065b", "evalmixo_6": "77271eba1510a820", "evalmixo_7": "901e965d7b940ca4"}
_K = _load("test_v8f_kind_for_oacc", HERE_T / "test_v8f_kind.py") if (QUEUE.exists() and (HERE_T / "test_v8f_kind.py").exists()) else None
_MX = _load("test_v8f_mixo_for_oacc", HERE_T / "test_v8f_mixo.py") if (QUEUE.exists() and (HERE_T / "test_v8f_mixo.py").exists()) else None
_H2 = _load("test_v8f_hist_for_oacc", HERE_T / "test_v8f_hist.py") if (HERE_T / "test_v8f_hist.py").exists() else None
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
FLAT_CHAIN = CODE / "box2_ops" / "chain_v8f_flat.sh"
HOLD_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


def _queue_text():
    return QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _blocks():
    parts = re.split(r"(?m)^(:[A-Za-z0-9_]+)$", _queue_text())
    return {parts[i][1:]: parts[i + 1] for i in range(1, len(parts), 2)}


def _block(label):
    return _queue_text().split(f"\n:{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")


def _code(lines):
    return [x for x in lines if x.strip() and not x.lower().startswith("rem ")]


def _argv_of(line):
    return [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', line.strip())]


def _dry(args, env_extra=None, clean_paths=False):
    env = dict(os.environ, V8F_DRYRUN="1")
    env.pop("WACC", None)                                           # the only env var a stage of THIS edit reads that a test must not inherit from the session
    if clean_paths:
        for k in ("REPO", "RUNSM", "CACHEFULL", "V9", "PY"):
            env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x]


def _dry1(args, env_extra=None):
    """The ONE command line a dry-run stage prints; a stage that prints nothing (it refused) FAILS by assertion."""
    rc, ln = _dry(args, env_extra)
    assert rc == 0 and len(ln) == 1, f"`{args}` should print exactly one command line, got rc {rc}, lines {ln}"
    return ln[0]


def _without(line, drop):
    t, out, i = _argv_of(line), [], 0
    while i < len(t):
        if t[i] in drop:
            i += 2
            continue
        out.append(t[i])
        i += 1
    return out


@needs_queue
@pytest.mark.skipif(_K is None or _MX is None, reason="test_v8f_kind.py / test_v8f_mixo.py (the pre-existing label pins) are not in this checkout")
def test_queue_oacc_is_crlf_every_existing_label_is_byte_identical_and_only_the_six_new_ones_are_added():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 900 and raw.count(b"\r") == raw.count(b"\r\n") and raw.endswith(b"\r\n"), "every line CRLF, no bare CR"
    q, blocks = _queue_text(), _blocks()
    for label, h in {**_K.PRE_EXISTING, **_MX.KIND_PINS, **MIXO_PINS, **FLAT_PINS}.items():
        assert label in blocks, f"the stage {label} was removed"
        assert hashlib.sha256(("\n" + blocks[label]).encode()).hexdigest()[:16] == h, f"the stage {label} was edited"
        assert "w-acc" not in blocks[label] and "oacc" not in blocks[label] and "WACC" not in blocks[label], f"{label} mentions the new stages"
    assert set(blocks) - ORC_LABELS == set(_K.PRE_EXISTING) | set(_MX.KIND_PINS) | set(MIXO_PINS) | set(FLAT_PINS) | OACC_LABELS and len(OACC_LABELS) == 6, "the only new labels are the six of this edit"
    for st in OSTAGES:
        assert q.count(f'if /i "%~1"=="{st}" goto :{st}\n') == 1, "a stage nobody dispatches never runs"
        assert q.count(f"\n:{st}\n") == 1
    assert q.count("\n:oaccwacc\n") == 1 and 'goto :oaccwacc' not in q and 'call :oaccwacc || exit /b 2' in q, "oaccwacc is a helper: called by the stages, never dispatched"
    target = {"train": "d_train", "eval": "d_eval"}
    for st in _K.OLD_DISPATCH + ("kind", "gatemixo", "trainmixo", "evalmixo", "gateflat", "trainflat", "evalflat"):
        assert f'if /i "%~1"=="{st}" goto :{target.get(st, st)}\n' in q, st
    disp = re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto :(\w+)$', q)
    assert [(a, b) for a, b in disp if b not in set(blocks)] == [], "a dispatch line whose target label does not exist"
    usage = next(x for x in q.split("\n") if x.startswith("echo usage:"))
    assert usage.endswith("^|gatedscale^|traindscale SEED^|evaldscale SEED^|identitylora^|smokelora") and "oacc" not in usage, "the usage line is untouched (the earlier pins on it stand)"
    for need in ("waccoacc", "gateoacc", "trainoacc SEED", "evaloacc SEED", "fitoacc SEED"):
        assert f"rem   v8f_mini_box2_queue.cmd {need}" in q, "the header documents the stage"
    assert q.split("\n:orcarm\n")[0].rstrip().endswith("echo [ok] HX weights M2b-oacc_s%~2\nexit /b 0"), "the new stages are the tail of the file (PREREG_V8F_ORECOVER's follow: widened)"
    assert q.split("\n:waccoacc\n")[0].endswith("\n:evalflat_4\nexit /b 0"), "the new stages start right after the last old label, which keeps its text (FLAT's own convention)"


@needs_queue
def test_queue_oacc_runs_the_command_it_prints_static():
    """A dry run can only print: the real line is checked statically -- the trainer / eval / rule script appear ONCE per command, inside the one string, and the real run executes that string."""
    for label, var, pre in (("gateoacc", "GX", '--arm M2b --w-acc %WACC% --out "%RUNSM%\\gate_M2b-oacc" '),
                            ("trainoacc", "TX", '--arm M2b --seed %~2 --grid pool4x10 --w-acc %WACC% --out "%OUT%" ')):
        code = _code(_block(label))
        assert [x for x in code if "%TRAIN%" in x] == [x for x in code if x.startswith(f'set "{var}=')], label
        assert any(x.startswith(f'set "{var}="%PY%" "%TRAIN%" ' + pre) for x in code), label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code, label
        assert sum(x.startswith(f"%{var}% ") for x in code) == 1, label
        assert sum(x.count("--w-acc") for x in code if x.startswith(f'set "{var}=')) == 1, f"{label}: --w-acc appears once, in the one string"
        assert not any(("pool8x20" in x or "--hist" in x or "--p-gt" in x or "--w-tac" in x or "--w-cons" in x or "--w-wm" in x or "--w-traj" in x or "--hmax" in x or "--window-start" in x
                        or "--code-mix" in x or "--o-interface" in x) for x in code), f"{label}: ONE lever"
        assert "call :oaccwacc || exit /b 2" in code, f"{label}: the weight comes from the rule's file (or the env) and a missing one is refused"
    code = _code(_block("evaloacc"))
    assert [x for x in code if "%EVAL%" in x] == [x for x in code if x.startswith('set "E0=')] and len([x for x in code if re.match(r'set "E\d=', x)]) == 1, "ONE eval string"
    assert "if defined V8F_DRYRUN (echo %E0% & exit /b 0)" in code and sum(x.strip().startswith("%E0% >> ") for x in code) == 1
    e0 = next(x for x in code if x.startswith('set "E0='))
    assert '--ckpt "%OUT%\\ckpt_last.pt"' in e0 and "--tag %TG%" in e0 and "%EC% %ER%" in e0 and e0.endswith("--project --families\"")
    assert "--tac" not in e0 and "--grid" not in e0 and "--w-acc" not in e0, "the eval reads the checkpoint's own cfg (cfg.w_acc); there is no flag for it and no oracle eval is needed"
    assert 'set "OUT=%RUNSM%\\M2b_s%~2_oacc"' in code and 'set "TG=M2b-oacc_s%~2"' in code
    code = _code(_block("waccoacc"))
    wx = next(x for x in code if x.startswith('set "WX='))
    assert [x for x in code if "%EVALOACC%" in x and not x.startswith("set \"EVALOACC=")] == [wx]
    assert wx.startswith('set "WX="%PY%" "%EVALOACC%" wacc --ckpt "%RUNSM%\\M2b_s0\\ckpt_last.pt" '), "the rule is evaluated with B's s0 checkpoint -- the banked one, not an X run"
    for need in ('--fit-clips "%OD%\\fit_clips.json"', '--banked-fit-windows "%OD%\\fit_windows_M2b_s0.npz"', '--out "%OD%\\w_acc.json"', '--out-txt "%OD%\\w_acc.txt"', "--join-train", "--tracks", "--cache-dir"):
        assert need in wx, need
    assert "%HB%" not in wx and "Architecture" not in wx and "&" not in wx, "no argument carries the ampersand of the repo path (the files are copied to %OD% first)"
    assert "if defined V8F_DRYRUN (echo %WX% & exit /b 0)" in code and sum(x.strip().startswith("%WX% >> ") for x in code) == 1
    assert sum(x.startswith("copy /y ") and x.rstrip().endswith("exit /b 2)") for x in code) == 2, "both HBLEND files are copied, and a failed copy is a refusal"
    assert 'if exist "%OD%\\w_acc.json" if exist "%OD%\\w_acc.txt" (echo [skip] w_acc & exit /b 0)' in code, "skipped only when BOTH files exist"
    code = _code(_block("fitoacc"))
    fx = next(x for x in code if x.startswith('set "FX='))
    assert [x for x in code if "%EVALOACC%" in x and not x.startswith("set \"EVALOACC=")] == [fx]
    assert fx.startswith('set "FX="%PY%" "%EVALOACC%" hx --ckpt "%OUT%\\ckpt_last.pt" ') and "--tag M2b-oacc_s%~2" in fx and "&" not in fx and "%HB%" not in fx
    assert "if defined V8F_DRYRUN (echo %FX% & exit /b 0)" in code and sum(x.strip().startswith("%FX% >> ") for x in code) == 1
    # every failure is noticed on its own line (flat `if errorlevel 1 (echo [FAIL] ... & exit /b 1)`, no parenthesised multi-line block: MEASURED to return rc 0 from `cmd /c`)
    for label in OSTAGES:
        body = _block(label)
        assert not any(x.rstrip().endswith("(") for x in body), f"{label}: a multi-line parenthesised block"
        assert sum(1 for x in body if x.startswith("if errorlevel 1 (echo [FAIL]") and x.rstrip().endswith("exit /b 1)")) >= 1, label
    assert _queue_text().split("\n:orcarm\n")[0].rstrip().endswith("echo [ok] HX weights M2b-oacc_s%~2\nexit /b 0")      # PREREG_V8F_ORECOVER's stages follow (widened)


@needs_queue
@win
def test_queue_oacc_stages_dry_one_lever(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"), WACC="0.123456")
    rc, ln = _dry("gateoacc", env)
    assert rc == 0 and len(ln) == 1 and "--selfcheck-only" in ln[0] and "gate_M2b-oacc" in ln[0], ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm"):t.index("--arm") + 4] == ["--arm", "M2b", "--w-acc", "0.123456"] and t.count("--w-acc") == 1, ln
    rc, ln = _dry("trainoacc 1", env)
    assert rc == 0 and len(ln) == 1, ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm") + 1] == "M2b" and t[t.index("--seed") + 1] == "1" and t[t.index("--w-acc") + 1] == "0.123456" and t.count("--w-acc") == 1, ln
    assert t[t.index("--grid") + 1] == "pool4x10" and t.count("--grid") == 1 and t[t.index("--out") + 1].endswith("M2b_s1_oacc"), ln
    assert "--exclude 540:600" in ln[0] and "--kin-mode yaw" in ln[0] and "--steps-mult 4" in ln[0] and "--batch 64" in ln[0], ln
    for no in ("--w-cons", "--w-tac", "--w-wm", "--w-traj", "--hmax", "--steps ", "--p-gt", "--hist", "--window-start", "--code-mix", "--o-interface"):
        assert no not in ln[0], f"ONE lever: {no} is not on the line"
    rc, ev = _dry("evaloacc 0", env)
    assert rc == 0 and len(ev) == 1, ev
    assert "--tag M2b-oacc_s0 " in ev[0] + " " and "--project --families" in ev[0] and f"M2b_s0_oacc{os.sep}ckpt_last.pt" in ev[0] and f"{os.sep}M2b_s0{os.sep}" not in ev[0], "THIS run's checkpoint, never B's"
    rc, wl = _dry("waccoacc", env)
    assert rc == 0 and len(wl) == 1 and "wacc --ckpt" in wl[0], wl
    ck = Path(_argv_of(wl[0])[_argv_of(wl[0]).index("--ckpt") + 1])
    assert ck.name == "ckpt_last.pt" and ck.parent.name == "M2b_s0", "the rule reads B's s0 checkpoint, never an X run"
    rc, fl = _dry("fitoacc 1", env)
    assert rc == 0 and len(fl) == 1 and " hx --ckpt" in fl[0] and f"M2b_s1_oacc{os.sep}ckpt_last.pt" in fl[0] and "--tag M2b-oacc_s1" in fl[0] and "weights_M2b-oacc_s1.json" in fl[0], fl
    # ONE lever, from the queue's own lines: OACC's gate / train line is B's line with --w-acc <v> (and the output dir) changed and NOTHING else
    drop = {"--w-acc", "--out"}
    b_train = _dry1("train M2b 1", env)
    assert "--w-acc" not in b_train
    assert _without(b_train, {"--out"}) == _without(_dry1("trainoacc 1", env), drop), "OACC's train line = B's + --w-acc <the rule's value>"
    b_gate = _dry1("gatearm M2b", env)
    assert _without(b_gate, {"--out"}) == _without(_dry1("gateoacc", env), drop), "OACC's gate line = B's gate line + --w-acc <the rule's value>"
    # the rule's value: from the file the waccoacc stage wrote when the env does not say; the env wins; a box that has not computed it prints a placeholder; the real run refuses
    runsm = tmp_path / "runsm2"
    (runsm / "oacc").mkdir(parents=True)
    env2 = dict(REPO=str(REPO), RUNSM=str(runsm), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    assert "WACC_FROM_RULE" in _dry1("gateoacc", env2) and "WACC_FROM_RULE" in _dry1("trainoacc 0", env2), "no value yet: the placeholder is printed, nothing is guessed"
    for text in (b"42.9311\n", b"42.9311\r\n", b"42.9311"):
        (runsm / "oacc" / "w_acc.txt").write_bytes(text)
        ln = _dry1("trainoacc 0", env2)
        t = _argv_of(ln)
        assert t[t.index("--w-acc") + 1] == "42.9311" and "WACC_FROM_RULE" not in ln, (text, ln)
    t = _argv_of(_dry1("gateoacc", dict(env2, WACC="7.5")))
    assert t[t.index("--w-acc") + 1] == "7.5", "an env WACC wins over the file (the chain's gate assertions compare the BUILT w_acc to the rule's value, so a rogue env is caught there)"


@needs_queue
@win
def test_queue_oacc_refuses_missing_arguments(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"), WACC="1.0")
    for args in ("trainoacc", "evaloacc", "fitoacc"):
        rc, ln = _dry(args, env)
        assert rc == 2 and ln == [], (args, rc, ln)


@needs_queue
@win
def test_queue_oacc_leaves_every_other_stage_untouched(tmp_path):
    """The edit only ADDS stages: the neighbouring arms' lines read as they did, and none of them mentions the lever."""
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("train M2b 0", "trainmixo 0", "evalmixo 0", "gatemixo", "trainflat 0", "evalflat 0", "gateflat", "trainpgt 0", "trainhist 0", "traindscale 0", "gatearm M2b", "gatepgt",
                 "gategrid M2b", "kind", "evalhist 0"):
        rc, ln = _dry(args, env)
        ln = [x.replace(str(tmp_path), "").replace(str(REPO), "") for x in ln]                  # the tmp dir is named after this test, the repo path after the work dir: neither is a stage's text
        assert rc == 0 and len(ln) >= 1 and all("--w-acc" not in x and "oacc" not in x and "WACC" not in x for x in ln), (args, ln)


def _oacc_tree(tmp):
    """test_v8f_hist's fake repo (a trainer / eval that record the argv they are handed and make the files the next stage waits for) plus a FAKE rule / fit script and HBLEND's two files."""
    repo, rec = _H2._fake_tree(tmp)
    (repo / "stack" / "scripts" / "v8f_mini_eval_oacc.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'oacc', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv\n"
        "if os.environ.get('FAKE_OACC') == 'fail':\n"
        "    sys.exit(3)\n"
        "out = a[a.index('--out') + 1]\n"
        "os.makedirs(os.path.dirname(out), exist_ok=True)\n"
        "open(out, 'w').write('{}')\n"
        "if a[1] == 'wacc' and os.environ.get('FAKE_OACC') != 'notxt':\n"
        "    open(a[a.index('--out-txt') + 1], 'w').write(os.environ.get('FAKE_WACC', '0.5') + '\\n')\n", encoding="utf-8")
    hb = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw" / "2026-10-25-hblend"
    hb.mkdir(parents=True)
    (hb / "fit_clips.json").write_text('{"clips": []}', encoding="utf-8")
    (hb / "fit_windows_M2b_s0.npz").write_bytes(b"banked")
    return repo, rec


def _oacc_env(tmp, repo):
    env = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(tmp / "runsm"), CACHEFULL=str(tmp / "cf"), V9=str(tmp / "v9"))
    for k in ("V8F_DRYRUN", "WACC", "FAKE_TRAIN", "FAKE_OACC", "FAKE_FAIL_TAG", "FAKE_WACC"):
        env.pop(k, None)
    return env


@needs_queue
@win
@pytest.mark.skipif(_H2 is None, reason="test_v8f_hist.py (the fake repo) is not in this checkout")
def test_queue_oacc_real_lines_execute_exactly_the_printed_commands(tmp_path):
    repo, rec = _oacc_tree(tmp_path)
    env = _oacc_env(tmp_path, repo)
    runsm = tmp_path / "runsm"

    def stage(args, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, **extra), capture_output=True, text=True, timeout=180, encoding="utf-8")
        return r.returncode, r.stdout

    def recs():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []

    def printed(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, V8F_DRYRUN="1"), capture_output=True, text=True, timeout=60, encoding="utf-8")
        return [_argv_of(x)[2:] for x in r.stdout.splitlines() if ".py" in x]       # drop python.exe and the script: sys.argv[1:]
    # no w_acc anywhere: a stage that carries it REFUSES -- nothing is run, nothing is guessed
    for args in ("gateoacc", "trainoacc 1"):
        rc, out = stage(args)
        assert rc == 2 and "REFUSED: no w_acc" in out and recs() == [], (args, rc, out)
    # THE RULE: the printed command is the executed one; HBLEND's two files are copied next to the outputs; both outputs exist; a finished rule is skipped
    want = printed("waccoacc")
    assert len(want) == 1 and want[0][0] == "wacc"
    rc, out = stage("waccoacc", FAKE_WACC="0.5")
    assert rc == 0 and "[ok] w_acc" in out, out
    assert recs() == [{"who": "oacc", "argv": want[0]}]
    od = runsm / "oacc"
    assert (od / "w_acc.json").exists() and (od / "w_acc.txt").read_text(encoding="utf-8").strip() == "0.5"
    assert (od / "fit_clips.json").read_text(encoding="utf-8") == '{"clips": []}' and (od / "fit_windows_M2b_s0.npz").read_bytes() == b"banked"
    rc, out = stage("waccoacc")
    assert rc == 0 and "[skip] w_acc" in out and len(recs()) == 1
    # THE GATE and THE RUN: the value in the argv is the one in the rule's file, once, and nowhere else
    want_gate, want_train = printed("gateoacc"), printed("trainoacc 1")
    assert len(want_gate) == 1 and len(want_train) == 1
    rc, out = stage("gateoacc")
    assert rc == 0 and "[ok] gate M2b-oacc" in out
    got = recs()
    assert len(got) == 2 and got[-1]["who"] == "train" and got[-1]["argv"] == want_gate[0] and "--selfcheck-only" in got[-1]["argv"]
    rc, out = stage("trainoacc 1")
    got = recs()
    assert rc == 0 and len(got) == 3 and got[-1]["argv"] == want_train[0], "the REAL trainoacc line hands the trainer the printed command"
    a = got[-1]["argv"]
    assert a.count("--w-acc") == 1 and a[a.index("--w-acc") + 1] == "0.5" and a[a.index("--arm") + 1] == "M2b" and a[a.index("--seed") + 1] == "1"
    assert (runsm / "M2b_s1_oacc" / "summary.json").exists() and not (runsm / "M2b_s1").exists(), "OACC has its own run dir; B's is never touched"
    rc, out = stage("trainoacc 1")
    assert rc == 0 and "[skip] M2b_s1_oacc" in out and len(recs()) == 3, "a finished run is not re-trained"
    # THE EVAL: one, of THIS run's checkpoint, with the printed command
    want_ev = printed("evaloacc 1")
    assert len(want_ev) == 1
    rc, out = stage("evaloacc 1")
    got = recs()
    assert rc == 0 and len(got) == 4 and got[-1]["who"] == "eval" and got[-1]["argv"] == want_ev[0]
    assert got[-1]["argv"][got[-1]["argv"].index("--tag") + 1] == "M2b-oacc_s1" and (runsm / "eval" / "eval_M2b-oacc_s1.npz").exists()
    rc, out = stage("evaloacc 1")
    assert rc == 0 and "[skip] eval M2b-oacc_s1" in out and len(recs()) == 4
    # THE HX FIT
    want_fit = printed("fitoacc 1")
    assert len(want_fit) == 1 and want_fit[0][0] == "hx"
    rc, out = stage("fitoacc 1")
    got = recs()
    assert rc == 0 and len(got) == 5 and got[-1]["who"] == "oacc" and got[-1]["argv"] == want_fit[0] and (od / "weights_M2b-oacc_s1.json").exists()
    rc, out = stage("fitoacc 1")
    assert rc == 0 and "[skip] HX weights M2b-oacc_s1" in out and len(recs()) == 5


@needs_queue
@win
@pytest.mark.skipif(_H2 is None, reason="test_v8f_hist.py (the fake repo) is not in this checkout")
def test_queue_oacc_real_stages_notice_failures_and_skip_exactly_what_is_finished(tmp_path):
    repo, rec = _oacc_tree(tmp_path)
    env = _oacc_env(tmp_path, repo)
    runsm = tmp_path / "runsm"

    def stage(args, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, **extra), capture_output=True, text=True, timeout=180, encoding="utf-8")
        return r.returncode, r.stdout
    rc, out = stage("waccoacc", FAKE_OACC="fail")
    assert rc == 1 and "[FAIL] w_acc rule" in out
    assert stage("waccoacc", FAKE_OACC="notxt")[0] == 1, "a rule that exits 0 without its number is a failure"
    rc, out = stage("waccoacc", FAKE_OACC="notxt")
    assert (runsm / "oacc" / "w_acc.json").exists() and not (runsm / "oacc" / "w_acc.txt").exists()
    n = len(rec.read_text(encoding="utf-8").splitlines())
    rc, out = stage("waccoacc", FAKE_WACC="0.25")
    assert rc == 0 and "[ok] w_acc" in out and len(rec.read_text(encoding="utf-8").splitlines()) == n + 1, "w_acc.json alone does not make the rule finished: it re-runs until BOTH exist"
    assert stage("waccoacc")[1].count("[skip] w_acc") == 1
    rc, out = stage("gateoacc", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] gate M2b-oacc" in out
    rc, out = stage("trainoacc 1", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] train M2b_s1_oacc - see" in out
    rc, out = stage("trainoacc 1", FAKE_TRAIN="nosummary")
    assert rc == 1 and "exited 0 without summary.json" in out
    for args in ("evaloacc 1", "fitoacc 1"):                                    # both refuse a run that has not finished training
        rc, out = stage(args)
        assert rc == 2 and "has not finished training" in out, (args, rc, out)
    assert stage("trainoacc 1")[0] == 0
    rc, out = stage("evaloacc 1", FAKE_FAIL_TAG="M2b-oacc_s1")
    assert rc == 1 and "[FAIL] eval M2b-oacc_s1 - see" in out
    rc, out = stage("fitoacc 1", FAKE_OACC="fail")
    assert rc == 1 and "[FAIL] HX fit M2b-oacc_s1 - see" in out
    (runsm / "oacc" / "fit_clips.json").unlink()
    rc, out = stage("fitoacc 1")
    assert rc == 2 and "fit_clips.json is missing - run waccoacc first" in out


@needs_queue
def test_the_guard_pattern_covers_the_new_eval_script_by_name():
    """v8f_guard.sh stops a run by matching our python processes on this pattern: the rule / fit script must match it, else the guard would be blind to a 3-hour CPU job."""
    g = GUARD.read_text(encoding="utf-8")
    m = re.search(r"CommandLine -match '([^']+)'", g)
    assert m, "the guard's process pattern"
    cmdline = '"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_mini_eval_oacc.py" hx --ckpt x'
    assert re.search(m.group(1), cmdline), "v8f_mini_eval_oacc.py is covered by the guard"
    assert not re.search(m.group(1), cmdline.replace("v8f_mini_eval_oacc", "v8f_something_else")), "and the pattern is not a catch-all"


# =============================================================================================================================================
# THE CHAIN: needs box 2 and ssh, so it cannot RUN here -- its predecessor predicate (RUN on synthetic logs), its order, its gate / rule assertions (RUN against synthetic records),
# its token record, its --dry-run (== the queue's own dry-run lines) and every load-bearing line are pinned
# =============================================================================================================================================
def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


def _chain_code():
    c = CHAIN.read_text(encoding="utf-8")
    return c, "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))


@needs_chain
def test_chain_oacc_static_ordering_the_wait_the_atomic_marker_the_rule_the_conditional_seed_1_and_no_base_decision():
    c, code = _chain_code()
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    assert "decide-base" not in code and "--base" not in code and "BASE" not in code, "B = M2b is fixed in the PREREG: nothing is decided at run time"
    # the order: predecessor -> marker -> refresh -> ckpt sha -> check-inputs on the REAL B side -> telemetry -> guard -> THE RULE -> gate -> token -> seed 0 -> the s0 read -> (seed 1 -> the final read)
    order = ["waiting for the predecessor chain", "( set -C;", "BOOT0=$(boot)", "fetch -q origin", 'ckptsha_b "before the rule"', 'v8f_oacc_read.py" --check-inputs', "launch telemetry_long",
             "v8f_guard.sh > ", "queue waccoacc $FIT_POLLS waccoacc", 'stop "w_acc rule assertions"', "WACC=$(tr -d", "GLINE=$(dry_stage", "queue gateoacc 120 gateoacc", 'stop "gate OACC assertions"',
             'stop "write the gate token"', "runone 0\n", "oaccread 0 oacc_read_s0.json", "runone 1\n", "oaccread 0,1 oacc_read.json"]
    idx = [_at(c, x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    assert 'queue "t_$g" $TRAIN_POLLS trainoacc $s' in c and 'queue "e_$g" $EVAL_POLLS evaloacc $s' in c and 'queue "f_$g" $FIT_POLLS fitoacc $s' in c
    # SEED 1 RUNS ONLY IF seed 0 PASSES PRIMARY-ACC AND PRIMARY-KEEP: the reader says so (s1_required), and the branch that does not run it is the end of the chain
    cond = 'if [ "$(jget oacc_read_s0.json s1_required)" != "True" ]; then\n  finish '
    assert cond in c
    after_s0 = c[_at(c, "oaccread 0 oacc_read_s0.json"):_at(c, "runone 1\n")]
    after_s0_code = "\n".join(x for x in after_s0.splitlines() if not x.lstrip().startswith("#"))
    assert after_s0_code.count("finish ") == 1 and after_s0_code.rstrip().endswith("fi") and "seed 1 NOT run" in after_s0_code
    assert _at(c, 'stop "w_acc rule assertions"') < _at(c, cond) and c.rstrip().endswith('finish "$(jget oacc_read.json reading) | $(jget oacc_read.json next)"')
    assert "runone 1\n" not in c[:_at(c, cond)], "no path runs seed 1 before the seed-0 read has been taken"
    assert _at(c, 'v8f_oacc_read.py" --check-inputs') < _at(c, "launch telemetry_long") < _at(c, "queue waccoacc"), "B's side is refused while box 2 is still idle and untouched"
    # the predecessor is FLAT's chain by default (override-able); markers are the shared ones, in THIS chain's own log; the busy marker is taken atomically (noclobber)
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_flat.out}" in c and "AFTER_GUARD=${AFTER_GUARD:-$L/v8fflat_guard.out}" in c
    assert "AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}" in c and "AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}" in c
    assert "chain_v8f_oacc.out" in c and 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh' in c
    assert "busy marker taken while arming" in c and "BOX2_BUSY.txt" in c
    # the training wait covers the longest run the guard lets live
    polls = int(re.search(r"^TRAIN_POLLS=(\d+)", c, re.M).group(1))
    maxh = float(re.search(r"V8F_MAX_RUN_H:-(\d+)", GUARD.read_text(encoding="utf-8")).group(1))
    assert polls == 1320 and polls * 30 / 3600 > maxh, f"a run the guard lets live ({maxh} h) outlasts the wait ({polls * 30 / 3600} h)"
    # the B side: the banked M2b files; the P10 guard file and the check-inputs come BEFORE any training
    for x in ("$L/v8f_mini/runs/M2b_s$s/$f", "$L/v8f_mini/eval/eval_M2b_s$s.npz", "floors_perwindow.npz"):
        assert x in c, x
    assert _at(c, "floors_perwindow.npz") < _at(c, 'v8f_oacc_read.py" --check-inputs') < _at(c, "queue waccoacc")
    # what an OACC run pulls: its selfcheck / summary / argv / log / ckpt (the reader reads the checkpoint's cfg), its ONE eval and its HX weights
    assert "for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull" in c
    assert 'for x in json npz log; do pull "eval/eval_$tg.$x" "$E/eval"; done' in c and 'pull "oacc/weights_$tg.json" "$E/w"' in c
    assert 'pull "gate_M2b-oacc/selfcheck.json"' in c
    arch = next(x for x in c.splitlines() if "archive --format=tar" in x)
    assert all(x in arch for x in ("stack", "taniteval", '"$HJ"', '"$PK"', '"$B1"')), "the reader, HBLEND's two files and the lead block ride in the archive"
    for f in ("v8f_oacc_read.py", "v8f_flat_read.py", "v8f_kind_read.py", "v8f_grid_read.py", "v8f_pgt_read.py", "v8f_dk_read.py", "v8f_wc_read.py", "v8f_tac_groups_read.py", "v8f_mini_full_report.py"):
        assert f in c, f"the chain insists the tip has {f}"
    for f in ("stack/scripts/v8f_mini_eval_oacc.py", "v8f_hblend.py", "fit_clips.json", "fit_windows_M2b_s0.npz", "p10_prior.pt", "p10_eval_pred6.npy"):
        assert f in c, f"the chain insists the tip has / sha-checks {f}"
    g = GUARD.read_text(encoding="utf-8")
    assert "ZZV8FMINI" in g and "(DONE|STOP)" in g
    # PROGRESS NOTE: PAUSED must never appear for the strategic family in anything this chain prints
    assert "PAUSED" not in c


@needs_chain
def test_chain_oacc_checkpoint_sha256_is_asserted_before_the_rule_and_after_the_evals():
    c = CHAIN.read_text(encoding="utf-8")
    assert 'ckptsha_b "before the rule"' in c and 'queue "e_$g" $EVAL_POLLS evaloacc $s\n  ckptsha_x $s after_evals' in c and 'queue "t_$g" $TRAIN_POLLS trainoacc $s\n  echo "" > $CUR\n  ckptsha_x $s trained' in c
    assert 'stop "$1: box 2\'s M2b_s$s checkpoint sha256 $g != the banked run\'s ${CKPT_SHA[$s]}"' in c
    assert 'stop "M2b_s${1}_oacc\'s checkpoint changed during the evals' in c
    assert 'ckset "M2b_s$s" "$g"' in c and 'ckset "M2b-oacc_s${1}_$2" "$g"' in c, "the measured values go into the json the reader takes as --ckpt-sha"
    m = re.search(r"CKPT_SHA=\((\w+) (\w+)\)", c)
    assert m and len(m.group(1)) == 64 and len(m.group(2)) == 64
    if READ_P.exists():
        R = _load("v8f_oacc_read_for_chain", READ_P)
        assert (m.group(1), m.group(2)) == (R.BANKED_CKPT_SHA["M2b_s0"], R.BANKED_CKPT_SHA["M2b_s1"]), "the chain and the reader assert the same two banked checkpoints"
    man = PKG / "raw" / "full" / "MANIFEST.md"
    if man.exists():
        t = man.read_text(encoding="utf-8")
        assert m.group(1) in t and m.group(2) in t, "raw/full/MANIFEST.md carries the banked sha256 values"


@needs_chain
def test_chain_oacc_syntax_and_dry_run_prints_the_queue_lines(tmp_path):
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's --dry-run")
    r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, V8F_QUEUE=str(QUEUE), TMPDIR=tmp_path.as_posix())
    for k in ("V8F_DRYRUN", "WACC"):
        env.pop(k, None)
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    stages = ["waccoacc", "gateoacc", "trainoacc 0", "evaloacc 0", "fitoacc 0", "trainoacc 1", "evaloacc 1", "fitoacc 1"]      # the order of the chain's own loop: seed 0 whole, then seed 1
    assert [x for x in out if x.startswith("== ") and x[3:] in stages] == [f"== {s}" for s in stages]
    want = []                                               # the chain's printed lines ARE the queue's own dry-run lines at box 2's default paths
    for st in stages:
        want += _dry(st, clean_paths=True)[1]
    assert [x for x in out if ".py" in x and "v8f_oacc_read" not in x] == want and len(want) == len(stages)
    t0 = next(x for x in out if "--seed 0" in x and "v8f_mini_train.py" in x)
    assert "--arm M2b --seed 0 --grid pool4x10 --w-acc WACC_FROM_RULE --out" in t0 and "D:\\tanitad\\repo" in t0 and "M2b_s0_oacc" in t0
    g0 = next(x for x in out if "--selfcheck-only" in x)
    assert "--arm M2b --w-acc WACC_FROM_RULE --out" in g0 and "gate_M2b-oacc" in g0
    r2 = subprocess.run([bash, str(CHAIN), "--dry-run"], env=dict(env, WACC="42.9311"), capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r2.returncode == 0 and "--w-acc 42.9311" in r2.stdout and "WACC_FROM_RULE" not in r2.stdout, "with the rule's value exported the printed lines carry it (the chain's GLINE)"
    assert "wait for <logs>/chain_v8f_flat.out: ^ZZV8FMINI-DONE" in r.stdout
    if READ_P.exists():
        R = _load("v8f_oacc_read_for_chain_dry", READ_P)
        for s, sha in ((0, R.BANKED_CKPT_SHA["M2b_s0"]), (1, R.BANKED_CKPT_SHA["M2b_s1"])):
            assert f"M2b_s{s} D:/tanitad/runs/v8f_mini_full/M2b_s{s}/ckpt_last.pt {sha}" in r.stdout
    reads = [x for x in out if x.startswith("v8f_oacc_read.py")]
    assert len(reads) == 3 and "--check-inputs" in reads[0] and "--seeds 0 " in reads[1] and "--seeds 0,1 " in reads[2] and all("--ckpt-sha" in x and "--wacc-json" in x for x in reads[1:])
    # every flag the chain hands the reader is one the reader's parser has
    flags = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', READ_P.read_text(encoding="utf-8")))
    for x in reads:
        assert set(re.findall(r"(--[a-z0-9-]+)", x)) <= flags, x
    # every stage the chain queues is a stage the queue dispatches
    c = CHAIN.read_text(encoding="utf-8")
    disp = set(re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto', _queue_text()))
    called = set(re.findall(r"\bqueue (?:\"[^\"]+\"|\w+) (?:\$\w+|\d+) (\w+)", c))
    assert called == set(OSTAGES) and called <= disp, called


# ---- the predecessor predicate (PREREG "Compute and order"), RUN on synthetic logs ----------------------------------------------------------------------
def _classify_many(tmp_path, cases):
    """cases: list of (out_lines | None, guard_lines | None) -> list of (rc, stdout). One bash per case (the chain's own predicate, not a copy)."""
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's predicate")
    res = []
    for n, (out_lines, guard_lines) in enumerate(cases):
        o = tmp_path / f"c{n}.out"
        o.write_bytes(("\n".join(out_lines) + "\n").encode("utf-8") if out_lines is not None else b"")
        args = [bash, str(CHAIN), "--classify-predecessor", o.as_posix()]
        if guard_lines is not None:
            g = tmp_path / f"c{n}.guard"
            g.write_bytes(("\n".join(guard_lines) + "\n").encode("utf-8"))
            args.append(g.as_posix())
        r = subprocess.run(args, capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
        res.append((r.returncode, r.stdout.strip()))
    return res


DONE_LINE = "ZZV8FMINI-DONE 10-26 04:10 tip 4f561a2c: PREREG_V8F_FLAT HIERARCHY BEATS FLAT"
PROCEED, HOLDING, WAIT = 0, 1, 2
J = "v8fflat10251200_"
WAITMSG = "(2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
# every `stop "<msg>"` text of the predecessor's chain (chain_v8f_flat.sh), rendered with plausible values, and the decision the PREREG commits for it. A run-level stop of FLAT's own jobs (a box-2 job
# returned rc != 0), FLAT's verdict-level stop (its reader refused), its launch gate's refusal and a PRE-FLIGHT stop (on the dev box, before box 2 ran anything) PROCEED; everything else is box-level or unknown
# and HOLDS.
FSTOPS = {
    "$1 rc=$rc": [(J + "gateflat rc=1", PROCEED), (J + "t_M2b_s0_flat rc=1", PROCEED), (J + "t_M2b_s1_flat rc=137", PROCEED), (J + "e_M2b_s0_flat rc=2", PROCEED), (J + "e_M2b_s1_flat rc=3", PROCEED),
                  (J + "refresh rc=1", HOLDING), (J + "gateflat rc=0", HOLDING), (J + "t_M2b_s2_flat rc=1", HOLDING), ("v8fmixo10251200_gatemixo rc=1", HOLDING), ("v8fflat1025120_gateflat rc=1", HOLDING),
                  ("t_x rc=1", HOLDING), ("rc=1", HOLDING), ("t_x rc=0", HOLDING), ("t_x rc=", HOLDING)],
    "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)": [(J + f"{j} wait={n} {WAITMSG}", HOLDING) for j, n in (("t_M2b_s0_flat", 2), ("e_M2b_s1_flat", 3), ("gateflat", 4), ("refresh", 3))],
    "flat read": [("flat read", PROCEED), ("flat read final", HOLDING), ("mixo read", HOLDING), ("oacc read 0", HOLDING)],
    "gate FLAT assertions": [("gate FLAT assertions", PROCEED), ("gate X assertions", HOLDING), ("gate OACC assertions", HOLDING)],
    "archive": [("archive", PROCEED)], "extract the reader": [("extract the reader", PROCEED)],
    "remote tip $T is not in the mirror": [("remote tip 4f561a2c is not in the mirror", PROCEED), ("remote tip abc is not in the mirror", HOLDING)],
    "the reader cannot use the M2b side (check-inputs)": [("the reader cannot use the M2b side (check-inputs)", PROCEED)],
    "stage the banked M2b_s$s/$f": [("stage the banked M2b_s0/summary.json", PROCEED), ("stage the banked M2b_s1/selfcheck.json", PROCEED)],
    "stage eval_M2b_s$s": [("stage eval_M2b_s0", PROCEED), ("stage eval_M2b_s1", PROCEED)],
    "stage $f": [(f"stage eval_M2b_s{s}-{x}", PROCEED) for s in (0, 1) for x in ("tacabs.npz", "tacabs.json", "tacorc.npz", "tacorc-all-goal.npz", "tacorc-all-latcons.npz")],
    "stage floors_perwindow.npz (the P10 reproduction guard)": [("stage floors_perwindow.npz (the P10 reproduction guard)", PROCEED)],
    "the tip's queue has no $st stage": [(f"the tip's queue has no {s} stage", PROCEED) for s in ("gateflat", "trainflat", "evalflat")] + [("the tip's queue has no gateoacc stage", HOLDING)],
    "the tip's queue prints no gateflat line": [("the tip's queue prints no gateflat line", PROCEED)],
    "the tip has no $f": [(f"the tip has no {f}.py", PROCEED) for f in ("v8f_flat_read", "v8f_kind_read", "v8f_grid_read", "v8f_pgt_read", "v8f_dk_read", "v8f_wc_read", "v8f_tac_groups_read", "v8f_mini_full_report")]
                          + [("the tip has no v8f_other_read.py", HOLDING)],
    "the tip has no per-window strata file": [("the tip has no per-window strata file", PROCEED)],
    "the tip has no B1 lead block": [("the tip has no B1 lead block", PROCEED)],
    "the tip's built_levers does not record o_interface": [("the tip's built_levers does not record o_interface", PROCEED)],
    "the tip's built_levers does not record p_gt_start": [("the tip's built_levers does not record p_gt_start", PROCEED)],
    "the tip's built_levers does not record p_gt_end": [("the tip's built_levers does not record p_gt_end", PROCEED)],
    "the tip's trainer does not run the o_interface probe in its in-process gate": [("the tip's trainer does not run the o_interface probe in its in-process gate", PROCEED)],
    "the tip's trainer has no --o-interface flag": [("the tip's trainer has no --o-interface flag", PROCEED)],
    "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --o-interface": [("the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --o-interface", PROCEED)],
    "the tip's v8f_mini.py has no O_INTERFACE_MODES": [("the tip's v8f_mini.py has no O_INTERFACE_MODES", PROCEED)],
    "the tip's v8f_mini.py has no cut line": [("the tip's v8f_mini.py has no cut line", PROCEED)],
    # box-level / unknown: HOLD
    "box 2 unreachable": [("box 2 unreachable", HOLDING)], "guard stop": [("guard stop", HOLDING)],
    "no fresh telemetry ($t1 / $t2): the temperature guard would be blind": [("no fresh telemetry (a / b): the temperature guard would be blind", HOLDING)],
    "pull $R/$1": [("pull D:/tanitad/runs/v8f_mini_full/M2b_s0_flat/ckpt_last.pt", HOLDING)],
    "ship $j": [("ship " + J + "gateflat", HOLDING)], "launch $j": [("launch " + J + "gateflat", HOLDING)],
    "ship the code archive": [("ship the code archive", HOLDING)], "ship refresh job": [("ship refresh job", HOLDING)], "launch refresh": [("launch refresh", HOLDING)],
    "write the gate token": [("write the gate token", HOLDING)],
    "box 2 $f != tip $T": [("box 2 stack/scripts/v8f_mini_train.py != tip 4f561a2c", HOLDING)],
    "box 2 v9 ${kv%%=*} md5 != ${kv#*=}": [("box 2 v9 v9_labels_train.npz md5 != 83e30bd3791f0767707eade5f4671759", HOLDING)],
    "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}": [("before the gate: box 2's M2b_s0 checkpoint sha256 aa != the banked run's bb", HOLDING)],
    "$1: cannot hash box 2's M2b_s$s checkpoint": [("before the gate: cannot hash box 2's M2b_s0 checkpoint", HOLDING)],
    "cannot hash box 2's M2b_s${1}_flat checkpoint ($2)": [("cannot hash box 2's M2b_s0_flat checkpoint (trained)", HOLDING)],
    "M2b_s${1}_flat's checkpoint changed during the evals: $g != $g0": [("M2b_s0_flat's checkpoint changed during the evals: a != b", HOLDING)],
    "v8f_guard.sh does not know this chain's markers (anchored): it would never exit": [("v8f_guard.sh does not know this chain's markers (anchored): it would never exit", HOLDING)],
    "<msg>": [("a stop text this predicate has never seen", HOLDING), ("", HOLDING)],
}
EXTRA_NON_STOP = [("busy marker present: v8F-mini PREREG_V8F_KIND since 10-25", HOLDING), ("busy marker taken while arming: x", HOLDING),
                  ("the predecessor's end does not clear the box: HOLD a stop that is not recognised: x", HOLDING)]


@needs_chain
def test_predecessor_done_proceeds_and_no_end_marker_waits(tmp_path):
    a, b, c, d, e = _classify_many(tmp_path, [(["10-25 12:00 boot0=2026-10-24 07:00:00", "10-25 12:30 t_M2b_s0_flat rc=0", DONE_LINE], None),
                                              (["10-25 12:00 boot0=2026-10-24 07:00:00", "10-25 12:30 t_M2b_s0_flat rc=0"], None), (None, None),
                                              (["ZZV8FMINI-STOP t_x rc=1", DONE_LINE], None), (["ZZV8FMINI-STOP guard stop", DONE_LINE], ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: x"])])
    assert a[0] == PROCEED and a[1].startswith("PROCEED the predecessor ended: ZZV8FMINI-DONE"), a
    assert b[0] == WAIT and b[1].startswith("WAIT no ZZV8FMINI-DONE"), b
    assert c[0] == WAIT, "an empty log: the predecessor has not started"
    assert d[0] == PROCEED and e[0] == PROCEED, "a DONE anywhere in the log wins (a stop followed by a done is a finished chain)"
    o = tmp_path / "crlf.out"
    o.write_bytes((DONE_LINE + "\r\n").encode())
    r = subprocess.run([_git_bash(), str(CHAIN), "--classify-predecessor", o.as_posix()], capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == PROCEED, "a CRLF log (the chain's output file) is read the same"


@needs_chain
def test_predecessor_stop_is_classified_run_gate_verdict_and_preflight_level_proceed_and_everything_else_holds(tmp_path):
    cases, want = [], []
    for tmpl, rows in FSTOPS.items():
        for msg, w in rows:
            cases.append((["10-25 12:00 boot0=2026-10-24 07:00:00", f"ZZV8FMINI-STOP {msg}".rstrip()], None))
            want.append((tmpl, msg, w))
    for msg, w in EXTRA_NON_STOP:
        cases.append((["10-25 12:00 boot0=2026-10-24 07:00:00", f"ZZV8FMINI-STOP {msg}"], None))
        want.append(("(echoed)", msg, w))
    got = _classify_many(tmp_path, cases)
    bad = [(t, m, w, g) for (t, m, w), g in zip(want, got) if g[0] != w or g[1].split()[0] != ("PROCEED" if w == PROCEED else "HOLD")]
    assert bad == [], bad
    assert len(want) >= 90
    refresh = [(m, g) for (t_, m, w_), g in zip(want, got) if "_refresh rc=" in m]
    assert refresh and all(g[1].startswith("HOLD box-level: FLAT's code-refresh job") for _, g in refresh), "FLAT's refresh job failing is named as box-level, not as an unrecognised text"


@needs_chain
@pytest.mark.skipif(not FLAT_CHAIN.exists(), reason="the predecessor chain is not in this checkout")
def test_every_stop_text_of_the_predecessors_chain_is_in_the_table_so_a_new_one_cannot_slip_through_unclassified():
    """The table above renders the predecessor chain's own `stop "<msg>"` calls. A stop text added to chain_v8f_flat.sh later is not in the table: this test names it, and the decision for
    it is then made on purpose (the predicate holds on anything it does not recognise, so the default is the safe one)."""
    src = FLAT_CHAIN.read_text(encoding="utf-8")
    stops = set(re.findall(r'stop "([^"]*)"', src))
    assert len(stops) >= 35
    assert sorted(stops - set(FSTOPS)) == [], "a stop text of the predecessor chain this test and FSTOPS have never classified"
    assert sorted(set(FSTOPS) - stops) == [], "FSTOPS names a stop text the predecessor chain no longer has"
    proceed = sorted(t for t, rows in FSTOPS.items() if any(w == PROCEED for _, w in rows))
    c = CHAIN.read_text(encoding="utf-8")
    assert "RUN_LEVEL_RE='^v8fflat[0-9]{8}_(gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat) rc=[1-9][0-9]*$'" in c and "VERDICT_LEVEL_RE='^flat read$'" in c and "GATE_LEVEL_RE='^gate FLAT assertions$'" in c
    assert len(proceed) == 24, proceed


@needs_chain
@pytest.mark.parametrize("ev", ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: 12 of the last 20 samples >= 61 C, max 63 C",
                                "ZZGUARD-STOP TIME 10-25 13:00 M2b_s0_flat projects 11.2 h (> 10 h) at 2.1 s/step",
                                "ZZCRASH 10-25 13:00 box 2 rebooted (boot time changed)", "ZZUNREACHABLE 10-25 13:00 ssh", "ZZTIMEOUT 10-25 13:00 t_x 1320 polls"])
def test_predecessor_run_level_stop_with_box_level_evidence_in_its_logs_holds(tmp_path, ev):
    """The evidence overrides the message: a run-level `rc=1` that came with a thermal / time stop, a reboot, an unreachable box or a timeout is the box's failure, not the run's."""
    run_stop = f"ZZV8FMINI-STOP {J}t_M2b_s0_flat rc=1"
    r = _classify_many(tmp_path, [(["10-25 12:00 boot0=x", ev, run_stop], None), (["10-25 12:00 boot0=x", run_stop], [ev]), (["10-25 12:00 boot0=x", "ZZV8FMINI-STOP flat read"], [ev]),
                                  (["10-25 12:00 boot0=x", run_stop], ["10-25 13:00 chain ended; guard off (max USB 52 C)"]), (["10-25 12:00 boot0=x", ev, DONE_LINE], None)])
    assert r[0][0] == HOLDING and r[0][1].startswith("HOLD box-level evidence"), r[0]
    assert r[1][0] == HOLDING and r[1][1].startswith("HOLD box-level evidence"), "the evidence may be in the predecessor's guard log"
    assert r[2][0] == HOLDING, "a verdict-level stop with box-level evidence holds too"
    assert r[3][0] == PROCEED, "a guard log that merely ended is no evidence"
    assert r[4][0] == PROCEED, "ZZV8FMINI-DONE proceeds whatever the log says before it: the predecessor FINISHED"


@needs_chain
def test_the_wait_loop_calls_the_same_predicate_and_the_chain_stops_on_a_hold():
    c = CHAIN.read_text(encoding="utf-8")
    assert 'd=$(after_decision "$AFTER_OUT" "$AFTER_GUARD"); r=$?' in c and 'if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor\'s end does not clear the box: $d"; exit 1; fi' in c
    assert 'if [ "${1:-}" = "--classify-predecessor" ]; then after_decision' in c
    assert _at(c, "after_decision() {") < _at(c, 'if [ "${1:-}" = "--classify-predecessor" ]') < _at(c, "waiting for the predecessor chain")
    assert "sleep 300" in c and "PREFLIGHT_LEVEL_RE=" in c


@needs_chain
def test_the_wait_loop_runs_end_to_end_proceeds_on_done_and_stops_on_a_box_level_stop(tmp_path):
    """Run the chain itself against a synthetic predecessor log with a stubbed box: DONE passes the wait (and the chain then dies at its first box-2 step: the stubbed `ssh` fails, the busy marker
    is taken and released); a thermal stop never gets past it, and takes no marker."""
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain")
    box = tmp_path / "box2_ship"
    (box / "logs").mkdir(parents=True)
    src = CHAIN.read_text(encoding="utf-8")
    # THE COPY CANNOT REACH ANYTHING REAL: the box-2 directory is this tmp dir, ssh / scp are `false` (so `boot` finds no box and the chain stops there, before it touches git or ships a file),
    # and the mirror is a path that does not exist. Each substitution is asserted: a chain whose text moved must not fall through to the real box or the real mirror.
    for old, new in (("B=/c/Users/Admin/box2_ship", f"B={box.as_posix()}"), ('SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"', 'SSH="false"'),
                     ('SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"', 'SCP="false"'), ("MIRROR=/c/Users/Admin/tanitad-push;", f"MIRROR={(tmp_path / 'no_mirror').as_posix()};")):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    code = "\n".join(x for x in src.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(?<![\w$])(ssh|scp)\s", code), "no literal ssh / scp command is left: every call goes through $SSH / $SCP, which are `false` here"
    script = tmp_path / "chain_under_test.sh"
    script.write_bytes(src.encode("utf-8"))
    env = dict(os.environ, TMPDIR=tmp_path.as_posix())
    for k in ("AFTER_OUT", "AFTER_GUARD", "AFTER_DONE", "AFTER_STOP", "WACC"):
        env.pop(k, None)

    def run(lines, guard=None):
        pred = box / "logs" / "chain_v8f_flat.out"
        pred.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        gp = box / "logs" / "v8fflat_guard.out"
        gp.write_bytes(("\n".join(guard) + "\n").encode("utf-8") if guard else b"")
        (box / "BOX2_BUSY.txt").unlink(missing_ok=True)
        r = subprocess.run([bash, str(script)], env=env, capture_output=True, text=True, timeout=120, encoding="utf-8", stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout
    rc, out = run([DONE_LINE])
    assert rc == 1 and "predecessor: PROCEED the predecessor ended" in out and "ZZV8FMINI-STOP box 2 unreachable" in out, out
    assert not (box / "BOX2_BUSY.txt").exists(), "a chain that stops releases the marker it took"
    rc, out = run([f"ZZV8FMINI-STOP {J}t_M2b_s0_flat rc=1"])
    assert rc == 1 and f"predecessor: PROCEED run-level stop: {J}t_M2b_s0_flat rc=1" in out and "ZZV8FMINI-STOP box 2 unreachable" in out, out
    rc, out = run(["ZZV8FMINI-STOP guard stop"], ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: x"])
    assert rc == 1 and "ZZV8FMINI-STOP the predecessor's end does not clear the box: HOLD" in out and "box 2 unreachable" not in out, out
    assert not (box / "BOX2_BUSY.txt").exists(), "a held chain never took the marker"
    rc, out = run([f"ZZV8FMINI-STOP {J}t_M2b_s0_flat wait=3 {WAITMSG}"])
    assert rc == 1 and "does not clear the box: HOLD a stop that is not recognised" in out and "box 2 unreachable" not in out, out


def _heredoc(name):
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"<<'EOF' \|\| stop \"" + re.escape(name) + r"\"\n(.*?)\nEOF\n", c, re.S)
    assert m, f"the chain's {name} are a python heredoc whose failure stops the chain"
    return m.group(1)


GLINE_OK = ('"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_mini_train.py" --arm M2b --w-acc 0.3 '
            '--out "D:\\tanitad\\runs\\v8f_mini_full\\gate_M2b-oacc" --kin-mode yaw --selfcheck-only')


def _gate_run(tmp_path, runs, mod=None, line=GLINE_OK, steps_arg="19164", windows_arg="306566", w=0.3):
    body = _heredoc("gate OACC assertions")
    real, _ = _H2._b_real(0)
    bl = real["summary.json"]["levers"]
    good = {"declared_vs_built": "PASS", "live": copy.deepcopy(runs["x"]["sc"]["live"]), "acc_probe": copy.deepcopy(runs["x"]["sc"]["acc_probe"]), "steps_total": 19164,
            "info": {"n_clips": 4512, "n_excluded": 60, "n_excluded_declared": 60, "excluded_in_train": 0, "excluded_sha256": HOLD_SHA,
                     "train_sha256": real["summary.json"]["train_sha256"], "kin_mode": "yaw", "lru": 174, "n_windows_all_clips": 306566,
                     "window_start": 3, "window_t_min": 3, "ds_window_start": 3},
            "levers": {"wm": "b", "navonly": False, "consistency": True, "cons_absent": "mask", "o_interface": "full", "code_mix": "whole", "w_acc": 0.3, "p_gt_start": 1.0, "p_gt_end": 0.2,
                       "hist": 4, "hmax": 16, "window_start": 3, "w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5},
                       "params": dict(bl["params"]), "n_params_trainable": bl["n_params_trainable"]}}
    good["live"]["dead_params"], good["live"]["declared_dead"] = [], []
    sc = copy.deepcopy(good)
    if mod:
        mod(sc)
    (tmp_path / "bsum.json").write_text(json.dumps(real["summary.json"]), encoding="utf-8")
    f = tmp_path / "sc.json"
    f.write_text(json.dumps(sc), encoding="utf-8")
    r = subprocess.run([sys.executable, "-", str(f), HOLD_SHA, steps_arg, str(tmp_path / "bsum.json"), line, windows_arg, repr(w)], input=body, capture_output=True, text=True, encoding="utf-8", timeout=60)
    return r.returncode, r.stdout + r.stderr


@needs_chain
@needs_pkg
@pytest.mark.skipif(_H2 is None, reason="test_v8f_hist.py (B's real record) is not in this checkout")
def test_chain_oacc_gate_assertions_pass_the_oacc_gate_and_refuse_every_way_it_can_be_wrong(tmp_path, runs):
    """The chain's gate block is RUN against a selfcheck.json built from B's REAL banked record and the trainer's OWN gate record (G-LIVE, the acc probe): a good OACC gate passes; each single-field
    corruption exits 1 and names the broken field (the PASS token, the dead set, the loss term, the probe for THIS weight, its controls, the lever, the schedule, the clip list, the holdout, the weights,
    the window set, the step count, the history and the PARAMETER COUNT cannot drift under a green gate)."""
    rc, out = _gate_run(tmp_path, runs)
    pb = _H2._b_real(0)[0]["summary.json"]["levers"]
    assert rc == 0 and "PASS" in out and "built w_acc 0.3" in out and f"params {pb['n_params_trainable']}" in out and "windows 306566" in out, out
    L = lambda k, v: (lambda s: s["levers"].__setitem__(k, v))                   # noqa: E731
    W = lambda k, v: (lambda s: s["levers"]["w"].__setitem__(k, v))              # noqa: E731
    I = lambda k, v: (lambda s: s["info"].__setitem__(k, v))                     # noqa: E731
    P = lambda k, v: (lambda s: s["levers"]["params"].__setitem__(k, v))         # noqa: E731
    Q = lambda k, v: (lambda s: s["acc_probe"].__setitem__(k, v))                # noqa: E731
    cases = (("declared_vs_built", lambda s: s.__setitem__("declared_vs_built", "FAIL"), "declared_vs_built"),
             ("live", lambda s: s["live"].__setitem__("pass", False), "live"),
             ("control grad", lambda s: s["live"].__setitem__("control_prior_grad_sum", 1.0), "live"),
             ("a weight dead", lambda s: s["live"].update(dead_params=["head_o.2.weight"]), "G-LIVE dead set"),
             ("a weight declared dead", lambda s: s["live"].update(declared_dead=["e_code.weight"]), "G-LIVE dead set"),
             ("the term is not in the loss dict", lambda s: s["live"]["terms"].pop("acc"), "carry no 'acc'"),
             ("no acc probe", lambda s: s.pop("acc_probe"), "acc_probe None"),
             ("acc probe not pass", Q("pass", False), "acc_probe"),
             ("probe for another weight", Q("w_acc_declared", 0.4), "the acc probe ran for w_acc"),
             ("probe weight other", Q("w_probe", 0.4), "the acc probe ran for w_acc"),
             ("O's head gets no gradient", Q("head_o_grad_diff", 0.0), "O's head receives no gradient"),
             ("the total did not move by w x term", Q("contribution", 0.0), "the total moved by"),
             ("prior control moved", Q("control_prior_grad_diff", 1e-9), "control_prior_grad_diff"),
             ("repeat control moved", Q("control_repeat_grad_diff", 1e-9), "control_repeat_grad_diff"),
             ("repeat total control moved", Q("control_repeat_total_diff", 1e-9), "control_repeat_total_diff"),
             ("the WM auxiliary moved with the weight", Q("control_wm_grad_diff", 1e-9), "world-model auxiliary"),
             ("n_clips", I("n_clips", 4000), "clips / holdout counts"), ("n_excluded", I("n_excluded", 59), "clips / holdout counts"),
             ("n_excluded_declared", I("n_excluded_declared", 59), "clips / holdout counts"), ("excluded_in_train", I("excluded_in_train", 1), "clips / holdout counts"),
             ("holdout", I("excluded_sha256", "0" * 64), "not the RA-1 holdout"), ("train list", I("train_sha256", "d" * 64), "TRAIN list"),
             ("kin_mode", I("kin_mode", "steer"), "kin_mode"), ("lru unrecorded", lambda s: s["info"].pop("lru"), "info.lru"),
             ("wm", L("wm", "a"), "levers"), ("navonly", L("navonly", True), "levers"), ("consistency", L("consistency", False), "levers"), ("cons_absent", L("cons_absent", "zero"), "levers"),
             ("built w_acc other", L("w_acc", 0.31), "built w_acc"), ("w_acc unrecorded", lambda s: s["levers"].pop("w_acc"), "built w_acc None"),
             ("built o_interface cut", L("o_interface", "cut"), "built o_interface 'cut' != 'full'"), ("built code_mix kind", L("code_mix", "kind"), "built code_mix 'kind' != 'whole'"),
             ("p_gt start", L("p_gt_start", 0.5), "built p_gt"), ("p_gt end", L("p_gt_end", 0.0), "built p_gt"),
             ("p_gt missing", lambda s: (s["levers"].pop("p_gt_start"), s["levers"].pop("p_gt_end")) and None, "built p_gt None -> None"),
             ("w_pen", W("w_pen", 0.6), "w_pen"), ("w_pen_self", W("w_pen_self", 0.3), "w_pen_self"), ("w_wm", W("w_wm", 2.0), "w_wm"),
             ("w_tac", W("w_tac", 2.0), "w_tac / w_cons"), ("w_cons", W("w_cons", 1.0), "w_tac / w_cons"),
             ("steps", lambda s: s.__setitem__("steps_total", 9000), "steps 9000"),
             ("window set is not B's", I("n_windows_all_clips", 279494), "the window set is not B's"),
             ("window set is not the PREREG's number", I("n_windows_all_clips", 306565), "the PREREG's 306566"),
             ("built hist 10", L("hist", 10), "built hist 10 != 4"), ("built hist unrecorded", lambda s: s["levers"].pop("hist"), "built hist None != 4"),
             ("built hmax 10", L("hmax", 10), "built hmax 10 != 16"), ("built window_start 9", L("window_start", 9), "built window_start 9 != 3"),
             ("the window list started at t = 9", I("window_t_min", 9), "window start (asked / first t"), ("the dataset clamps from 9", I("ds_window_start", 9), "window start (asked / first t"),
             ("one parameter too many", L("n_params_trainable", pb["n_params_trainable"] + 1), "n_params_trainable"),
             ("a group grew", P("T", pb["params"]["T"] + 5), "parameter group T"),
             ("params unrecorded", lambda s: s["levers"].pop("params"), ""))
    for name, mod, msg in cases:
        rc, out = _gate_run(tmp_path, runs, mod)
        assert rc != 0 and msg in out, (name, rc, out)
    # the lever, from the queue's own printed line
    for name, line, msg in (("no --w-acc on the line", GLINE_OK.replace(" --w-acc 0.3", ""), "--w-acc <the rule's value> exactly once"),
                            ("a different value on the line", GLINE_OK.replace("--w-acc 0.3", "--w-acc 0.31"), "--w-acc <the rule's value> exactly once"),
                            ("--w-acc twice", GLINE_OK.replace("--w-acc 0.3", "--w-acc 0.3 --w-acc 0.3"), "--w-acc <the rule's value> exactly once"),
                            ("another arm", GLINE_OK.replace("--arm M2b", "--arm M2"), "is not --arm M2b"),
                            ("--hist set", GLINE_OK + " --hist 10", "sets --hist"), ("--hmax set", GLINE_OK + " --hmax 16", "sets --hmax"),
                            ("--p-gt-start set", GLINE_OK + " --p-gt-start 0.5", "sets --p-gt-start"), ("--window-start set", GLINE_OK + " --window-start 3", "sets --window-start"),
                            ("--w-cons set", GLINE_OK + " --w-cons 1.0", "sets --w-cons"), ("--w-traj set", GLINE_OK + " --w-traj 2.0", "sets --w-traj"),
                            ("--grid set", GLINE_OK + " --grid pool8x20", "sets --grid"), ("--code-mix set", GLINE_OK + " --code-mix kind", "sets --code-mix"),
                            ("--o-interface set", GLINE_OK + " --o-interface cut", "sets --o-interface")):
        rc, out = _gate_run(tmp_path, runs, line=line)
        assert rc == 1 and msg in out, (name, rc, out)
    # a budget different from B's, however it came about, stops the chain at the gate
    rc, out = _gate_run(tmp_path, runs, steps_arg="17472")
    assert rc == 1 and "steps 19164 != B's 17472" in out, out
    # the rule's value is the one the gate is compared with: a gate that passed for 0.3 does not pass for 0.4
    rc, out = _gate_run(tmp_path, runs, w=0.4)
    assert rc == 1 and "built w_acc" in out, out


@needs_chain
def test_chain_oacc_rule_assertions_pass_a_real_model_route_record_and_refuse_every_way_it_can_be_wrong(fitworld, tmp_path):
    body = _heredoc("w_acc rule assertions")
    out, txt, npz = tmp_path / "w.json", tmp_path / "w.txt", tmp_path / "fit.npz"
    fw = fitworld
    model, cfg, ck = OE.load_ckpt_model(fw["ckpt"])
    clips = json.loads(fw["fc"].read_text())["clips"]
    O = OE.forward_fit_set(model, cfg, ck, clips, str(fw["root"]), str(fw["root"] / "join_train.npz"), str(fw["root"] / "native_tracks.pt"), batch=8, log=lambda *_: None)
    np.savez_compressed(npz, **O)
    assert OE.main(["wacc", *_model_route_args(fw, out, ["--out-txt", str(txt), "--banked-fit-windows", str(npz)])]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    sha, n = rec["source"]["ckpt_sha256"], rec["n_windows"]

    def run(mod=None, txt_text=None, b0=None, nfit=None):
        r = copy.deepcopy(rec)
        if mod:
            mod(r)
        jp, tp = tmp_path / "r.json", tmp_path / "r.txt"
        jp.write_text(json.dumps(r), encoding="utf-8")
        tp.write_text(txt_text if txt_text is not None else txt.read_text(encoding="ascii"), encoding="utf-8")
        p = subprocess.run([sys.executable, "-", str(jp), str(tp), b0 or sha, str(nfit or n)], input=body, capture_output=True, text=True, encoding="utf-8", timeout=60)
        return p.returncode, p.stdout + p.stderr
    rc, o = run()
    assert rc == 0 and "w_acc rule: PASS" in o, o
    for name, kw, msg in (("schema", dict(mod=lambda r: r.__setitem__("schema", "x")), "schema"),
                          ("zero weight", dict(mod=lambda r: r.__setitem__("w_acc", 0.0), txt_text="0.0\n"), "w_acc 0.0"),
                          ("txt differs from the json", dict(txt_text="1.5\n"), "w_acc.txt"),
                          ("txt not the repr", dict(txt_text=f"{rec['w_acc']:.12f}\n"), "w_acc.txt"),
                          ("cross-check failed", dict(mod=lambda r: r["banked_crosscheck"].__setitem__("pass", False)), "the cross-check against HBLEND's banked arrays"),
                          ("no cross-check", dict(mod=lambda r: r.pop("banked_crosscheck")), "the cross-check against HBLEND's banked arrays"),
                          ("another checkpoint", dict(b0="0" * 64), "banked M2b_s0 checkpoint"),
                          ("another fit set", dict(nfit=n + 1), "n_windows"),
                          ("the ratio is not one half", dict(mod=lambda r: r.__setitem__("check_w_acc_x_L_acc_over_L_traj", 1.0)), "w_acc x L_acc / L_traj != 0.5"),
                          ("frac", dict(mod=lambda r: r.__setitem__("frac", 1.0)), "frac")):
        rc, o = run(**kw)
        assert rc == 1 and msg in o, (name, rc, o)


@needs_chain
def test_chain_oacc_writes_a_gate_token_bound_to_the_commit_the_queue_the_selfcheck_the_weight_and_the_data_manifests(tmp_path):
    c = CHAIN.read_text(encoding="utf-8")
    body = _heredoc("write the gate token")
    sc = {"info": {"train_sha256": "t" * 64, "excluded_sha256": HOLD_SHA}, "argv_sha": "abc123"}
    scp, out = tmp_path / "sc.json", tmp_path / "gate_token.json"
    scp.write_text(json.dumps(sc), encoding="utf-8")
    keys = json.dumps({"stack/scripts/v8f_mini_train.py": "k1", "stack/tanitad/models/v8f_mini.py": "k2"})
    r = subprocess.run([sys.executable, "-", str(scp), str(out), "c0ffee", "q" * 64, keys, "42.9311"], input=body, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    tok = json.loads(out.read_text(encoding="utf-8"))
    assert tok["prereg"] == "PREREG_V8F_OACC" and tok["gate"] == "PASS" and tok["commit"] == "c0ffee" and tok["queue_sha256"] == "q" * 64 and tok["argv_sha"] == "abc123" and tok["w_acc"] == 42.9311
    assert tok["selfcheck_sha256"] == hashlib.sha256(scp.read_bytes()).hexdigest() and tok["key_files_sha256"] == json.loads(keys)
    assert tok["train_sha256"] == "t" * 64 and tok["excluded_sha256"] == HOLD_SHA and set(tok["v9_md5"]) == {"v9_labels_train.npz", "v9_labels_eval139.npz"}
    # ordering: the token is written only after the gate assertions passed, and before any training
    assert _at(c, '"gate OACC assertions"') < _at(c, '"write the gate token"') < _at(c, "runone 0\n")


@needs_chain
def test_chain_oacc_jget_prints_what_the_conditional_seed_1_branch_compares_with(tmp_path):
    """The branch `[ "$(jget oacc_read_s0.json s1_required)" != "True" ]` compares a python-printed bool with the string True: run the chain's own jget on both values and on a missing key."""
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's jget")
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"(?m)^jget\(\) \{.*\}$", c)
    assert m, "the chain's jget"
    (tmp_path / "t.json").write_text(json.dumps({"s1_required": True, "f": False}), encoding="utf-8")
    script = f'PY="{Path(sys.executable).as_posix()}"; E="{tmp_path.as_posix()}"\n{m.group(0)}\n[ "$(jget t.json s1_required)" != "True" ] && echo RUN_NO_S1 || echo RUN_S1\n[ "$(jget t.json f)" != "True" ] && echo RUN_NO_S1 || echo RUN_S1\n'
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.stdout.split() == ["RUN_S1", "RUN_NO_S1"], (r.stdout, r.stderr)
    script2 = script.replace("jget t.json f", "jget t.json absent")
    r = subprocess.run([bash, "-c", script2], capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.stdout.split()[-1] == "RUN_NO_S1", "a reader that wrote no s1_required does not start seed 1"


@needs_chain
def test_chain_oacc_finish_releases_the_box_before_it_announces_done_and_stop_releases_it_too():
    """The next chain waits for DONE and then refuses on a leftover BOX2_BUSY.txt: finish() must remove the marker and stop the telemetry BEFORE it prints ZZV8FMINI-DONE."""
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"\nfinish\(\) \{.*?\n\}\n", c, re.S)
    assert m, "no finish() function in the chain"
    body = m.group(0)
    for need in ("telemetry.stop", "rm -f $B/BOX2_BUSY.txt", 'echo "" > $CUR', "ZZV8FMINI-DONE", "exit 0"):
        assert need in body, f"finish() lost {need!r}"
    assert _at(body, "rm -f $B/BOX2_BUSY.txt") < _at(body, "ZZV8FMINI-DONE") < _at(body, "exit 0")
    assert re.search(r'^stop\(\) \{ echo "ZZV8FMINI-STOP \$\*"; echo "" > \$CUR; rm -f \$B/BOX2_BUSY\.txt; exit 1; \}$', c, re.M), "a stop after the marker was taken releases it"


@needs_chain
def test_chain_oacc_load_bearing_lines_are_present():
    """Every line whose loss would let a wrong run through is pinned. Substrings, not whole lines: line ends are not load-bearing."""
    c = CHAIN.read_text(encoding="utf-8")
    lines = (
        r"""for st in waccoacc gateoacc trainoacc evaloacc fitoacc; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done""",
        r"""grep -q '"w_acc": cfg.w_acc' || stop "the tip's built_levers does not record w_acc""",
        r"""grep -q '^def accel_term(' || stop "the tip's v8f_mini.py has no accel_term""",
        r"""grep -q 'total = total + cfg.w_acc \* T\["acc"\] if cfg.w_acc > 0.0 else total' || stop "the tip's v8f_mini.py has no acceleration line in the loss total""",
        r"""grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start""",
        r"""grep -q -- '"--w-acc"' || stop "the tip's trainer has no --w-acc flag""",
        r"""grep -q '"w_acc": "cfg:w_acc"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --w-acc""",
        r"""grep -q 'acc_probe(model, probe, cfg)' || stop "the tip's trainer does not run the acc probe in its in-process gate""",
        r"""git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block""",
        r"""|| stop "the reader cannot use the M2b side (check-inputs)""",
        r"""|| stop "oacc read $1""",
        r"""$PY "$RD/v8f_oacc_read.py" $(reader_args""",
        r"""GLINE=$(dry_stage "$QC" gateoacc | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gateoacc line""",
        r"""$HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS $WACC <<'EOF' || stop "gate OACC assertions""",
        r"""pull "gate_M2b-oacc/selfcheck.json" "$E/gate_M2b-oacc""",
        r"""$SCP "tanitad-box2:$R/oacc/w_acc.json" "$E/w_acc.json" && $SCP "tanitad-box2:$R/oacc/w_acc.txt" "$E/w_acc.txt" || stop "pull the w_acc rule files""",
        r"""WACC=$(tr -d '\r\n' < $E/w_acc.txt); export WACC""",
        r"""echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)""",
        r"""[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present""",
        r"""if [ $r = 0 ]; then echo "$(date '+%m-%d %H:%M') predecessor: $d"; break; fi""",
        r"""cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo""",
        r"""cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)""",
        r"""g="M2b_s${1}_oacc" tg="M2b-oacc_s${1}""",
        r"""grep ZZGUARD-STOP $L/v8foacc_guard.out 2>/dev/null && stop "guard stop""",
        r"""V8F_CHAIN_OUT=$L/chain_v8f_oacc.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh""",
        r"""[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry""",
        r"""M2B_STEPS=19164""",
        r"""M2B_WINDOWS=306566""",
        r"""FIT_WINDOWS=9582""",
        r"""TRAIN_POLLS=1320; EVAL_POLLS=600; FIT_POLLS=120""",
        r"""rm -f $CK""",
        r"""|| stop "write the gate token""",
        r"""|| stop "w_acc rule assertions""",
        r"""stop "box 2 unreachable""",
    )
    for x in lines:
        assert x in c, f"the chain lost: {x}"
