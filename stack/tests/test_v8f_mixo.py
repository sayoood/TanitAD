"""PREREG_V8F_MIXO: O is teacher-forced on codes mixed PER UNIT (`--code-mix kind`, ONE lever) instead of one GT-or-own switch per sample (`whole`, every earlier arm).

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): a lever whose library function is correct and whose CALLER never reaches it. So every
identity below runs from argv, hop by hop, and has a mutation that turns it RED (`raw/2026-10-25-mixo/mutate_mixo.py`, output beside it):

  argv --code-mix -> make_config (cfg.code_mix) -> declared_vs_built (G-DVB: FLAG_CHECKS entry + the BUILT lever + the probe of the code O actually reads) -> V8FMini._forward
  -> `_code_mix_mask` -> `torch.where(mask, GT code, T's own code)` -> run_o(..., code)   (the tensor O reads)

  (A) the units         `CODE_MIX_UNITS` = the decisions (goal + latdec + londec) + the six KIND cells: a PARTITION of the 60 code indices (pinned at import, and here).
  (B) `whole` IS TODAY  the default reproduces the saved pre-change module (`raw/2026-10-25-mixo/pre_mixo_v8f_mini.py.ref`) bit for bit: the built model, `_forward`, the RNG stream, the
                        loss, and a 200-step CPU loss trace (PREREG proof 2).
  (C) the mask          PREREG proof 3: at p_gt = 1 `kind` equals `whole`; at p_gt = 0 both equal T's own code; at p_gt = 0.5 the per-sample GT fraction is PARTIAL across 4,096 draws
                        (a mask that only ever gives all or nothing is a switch, not a mix); the exact Bernoulli draw replayed from the generator; independence of the units.
  (D) G-DVB             PREREG proof 1: `--code-mix` has its entry; a flag without one is refused; a declared `kind` the model did not build (and the reverse) is refused; `kind` on a constant
                        p_gt (a lever that does nothing) is refused; the code_mix probe reads the code O receives and refuses a model whose mix is not the declared one.
  (E) reachability      PREREG proof 4: a real `TR.main` run on the synthetic world: the argv flag reaches the config, the checkpoint, the records and the code `run_o` receives on EVERY
                        training step; the default run never calls the mixing line.
  (F) the reader        `code/v8f_mixo_read.py`: the committed decision table (PASS / FAIL-O-LEARNED / FAIL-O-DID-NOT-LEARN), D-M (TAC_A2's r(0.5)), HARM, the identity control, and every refusal (rc 2).
  (G) the queue + chain the stages print the command they run, one lever; every existing label is byte-identical; the chain's gate assertions, ordering and dry run.
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

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
RAWM = PKG / "raw" / "2026-10-25-mixo"
REF_P = RAWM / "pre_mixo_v8f_mini.py.ref"
CODE = PKG / "code"
READ_P = CODE / "v8f_mixo_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_mixo.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_MIXO.md"
HERE_T = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_ref(name, path):
    """A saved pre-change module, loaded under a private name. Registered in sys.modules BEFORE it executes (a dataclass with `from __future__ import annotations` looks its module up)."""
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


REF_M = _load_ref("v8f_mini_pre_mixo", REF_P) if REF_P.exists() else None
needs_ref = pytest.mark.skipif(REF_M is None, reason="raw/2026-10-25-mixo (the saved pre-change module) is not in this checkout")
_P = _load("test_v8f_pgt_for_mixo", HERE_T / "test_v8f_pgt.py")                 # the batch, the spy and the synthetic world the other v8F-mini tests use
world, tiny_model, make_batch = _P.world, _P.tiny_model, _P.make_batch
_mixed_batch, _spy, _own_outputs = _P._mixed_batch, _P._spy, _P._own_outputs
_T = _P._T                                                                       # test_v8f_mini
UNITS = list(M.CODE_MIX_UNITS)
U_IDX = [torch.as_tensor(M.CODE_MIX_UNITS[u], dtype=torch.long) for u in UNITS]
COMMON = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5",
          "--arm", "M2b"]


def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])


def _kind_of(model):
    """A copy of `model` whose config says code_mix kind (same weights: the lever adds no parameter)."""
    k = copy.deepcopy(model)
    k.cfg = dataclasses.replace(k.cfg, code_mix="kind")
    return k


# =============================================================================================================================================
# (A) THE UNITS
# =============================================================================================================================================
def test_units_are_a_partition_of_the_code_and_are_the_decisions_plus_the_six_kinds():
    assert tuple(M.CODE_MIX_UNITS) == ("decisions", *M.CODE_KINDS), "seven units: the decisions, then the six KIND cells in KIND's order"
    assert M.N_CODE_UNITS == 7 and tuple(M.CODE_KINDS) == ("speed", "angle", "tte", "timegap", "geom", "leadgap")
    flat = sorted(i for idx in M.CODE_MIX_UNITS.values() for i in idx)
    assert flat == list(range(M.CODE_DIM)) == list(range(60)), "every code index in exactly one unit"
    assert M.CODE_MIX_UNITS["decisions"] == tuple(i for n in ("goal", "latdec", "londec") for i in M.CODE_GROUPS[n]) == tuple(range(34))
    assert sorted(i for k in M.CODE_KINDS.values() for i in k) == list(range(34, 60)), "the six kinds are the 26 constraint cells"
    assert len(M.CODE_UNIT_OF) == 60 and all(M.CODE_MIX_UNITS[UNITS[u]].count(i) == 1 for i, u in enumerate(M.CODE_UNIT_OF))


def test_a_unit_set_that_is_not_a_partition_is_refused_at_import(tmp_path):
    """The pin lives in the module (ImportError), not only here: a hole or an overlap in `CODE_MIX_UNITS` cannot be imported."""
    src = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py").read_text(encoding="utf-8")
    anchor = '**CODE_KINDS}'
    assert src.count(anchor) == 1
    for tag, bad in (("hole", '**{k: v for k, v in CODE_KINDS.items() if k != "speed"}}'), ("overlap", '**CODE_KINDS, "extra": (0,)}')):
        path = tmp_path / f"v8f_mini_{tag}.py"
        path.write_text(src.replace(anchor, bad), encoding="utf-8")
        try:
            with pytest.raises(ImportError, match="not a partition"):
                _load_ref(f"v8f_mini_{tag}", path)
        finally:
            sys.modules.pop(f"v8f_mini_{tag}", None)


def test_the_flag_defaults_to_whole_everywhere_and_a_bad_value_is_refused():
    assert M.V8FMiniConfig().code_mix == "whole" and M.CODE_MIX_MODES == ("whole", "kind")
    a = _args()
    assert a.code_mix == "whole" and TR.make_config(a).code_mix == "whole"
    k = _args(["--code-mix", "kind"])
    assert k.code_mix == "kind" and TR.make_config(k).code_mix == "kind"
    with pytest.raises(SystemExit):
        _args(["--code-mix", "per_kind"])
    with pytest.raises(ValueError, match="code_mix must be one of"):
        M.V8FMiniConfig(code_mix="per_kind")
    # nothing else moves: the two configs differ in code_mix and in nothing else
    d = {key: v for key, v in TR.make_config(_args()).__dict__.items() if key != "code_mix"}
    assert d == {key: v for key, v in TR.make_config(k).__dict__.items() if key != "code_mix"}


def test_built_levers_record_the_built_code_mix_and_the_lever_adds_no_parameter(world):
    from tanitad.data import v8f_mini_data as DD
    J = DD.load_join(Path(world["root"]) / "join_train.npz")
    torch.manual_seed(0)
    mw, cfg = tiny_model(J, wm="off")
    torch.manual_seed(0)
    mk, _ = tiny_model(J, wm="off", code_mix="kind")
    assert mw.built_levers()["code_mix"] == "whole" and mk.built_levers()["code_mix"] == "kind"
    sw, sk = mw.state_dict(), mk.state_dict()
    assert sw.keys() == sk.keys() and all(torch.equal(sw[x], sk[x]) for x in sw), "same seed, same weights: the lever builds no parameter"
    lw, lk = mw.built_levers(), mk.built_levers()
    assert lw["params"] == lk["params"] and lw["n_params_trainable"] == lk["n_params_trainable"]
    assert {key: v for key, v in lw.items() if key != "code_mix"} == {key: v for key, v in lk.items() if key != "code_mix"}


# =============================================================================================================================================
# (B) `whole` IS TODAY, BIT FOR BIT
# =============================================================================================================================================
def _old_pair(world, n=12, seed=2):
    """The same weights and the same batch in the CURRENT module (code_mix whole) and in the saved pre-change module."""
    from tanitad.data import v8f_mini_data as DD
    J = DD.load_join(Path(world["root"]) / "join_train.npz")
    mu, sd = M.cons_stats(J)
    cfg = M.V8FMiniConfig(**{**_T.TINY, "wm": "off"})
    kw = {k: v for k, v in cfg.__dict__.items() if k != "code_mix"}
    torch.manual_seed(3)
    new = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    torch.manual_seed(3)
    old = REF_M.V8FMini(REF_M.V8FMiniConfig(**kw), REF_M.P10Prior.synthetic(0), mu, sd)
    b, _, _ = make_batch(world, n=n, seed=seed, hmax=cfg.hmax)
    B = b["feats"].shape[0]
    i = torch.arange(B)
    b["lab_t"]["has"] = (i % 3 != 0)
    return new, old, b, J


@needs_ref
def test_whole_builds_the_saved_pre_change_model_and_forward_bit_for_bit_with_the_same_rng_stream(world):
    new, old, b, _ = _old_pair(world)
    sn, so = new.state_dict(), old.state_dict()
    assert sn.keys() == so.keys() and all(torch.equal(sn[k], so[k]) for k in sn), "identical built model"
    ln, lo = new.built_levers(), old.built_levers()
    assert set(ln) - set(lo) == {"code_mix"} and set(lo) <= set(ln) and {k: v for k, v in ln.items() if k in lo} == lo
    assert {f.name for f in dataclasses.fields(M.V8FMiniConfig)} - {f.name for f in dataclasses.fields(REF_M.V8FMiniConfig)} == {"code_mix"}
    for p in (0.0, 0.2, 0.5, 0.9, 1.0):
        gn, go = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
        new.train(), old.train()
        torch.manual_seed(5)
        on = new._forward(b, p, True, gn)
        torch.manual_seed(5)
        oo = old._forward(b, p, True, go)
        assert on.keys() == oo.keys()
        for k in on:
            if isinstance(on[k], torch.Tensor):
                assert torch.equal(on[k], oo[k]), (p, k)
        assert torch.equal(gn.get_state(), go.get_state()), f"p_gt {p}: whole must consume the generator exactly as before (the RNG stream is B's)"
    # the loss
    for p in (0.3, 1.0):
        torch.manual_seed(5)
        ln_ = new.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        torch.manual_seed(5)
        lo_ = old.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        assert ln_["terms"].keys() == lo_["terms"].keys() and all(torch.equal(ln_["terms"][k], lo_["terms"][k]) for k in ln_["terms"])


def _trace(model, batches, steps=200, p0=1.0, p1=0.2, seed=17, lr=2e-3):
    """A short CPU training trace: AdamW, B's linear p_gt schedule, a seeded generator and global seed per step; returns the per-step total loss."""
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


def _trace_batches(world, cfg_hmax):
    bs = []
    for s in (2, 3, 4):
        b, _, _ = make_batch(world, n=8, seed=s, hmax=cfg_hmax)
        b["lab_t"]["has"] = (torch.arange(8) % 4 != 1)
        bs.append(b)
    return bs


@needs_ref
def test_whole_reproduces_the_saved_pre_change_200_step_loss_trace_exactly_and_kind_does_not(world):
    """PREREG proof 2: the same seed, the same batches, the same optimiser, 200 steps: every loss float of `whole` equals the saved pre-change module's. The control: `kind` leaves B's trace at
    the first step it can (p_gt 1.0 is the same mask; the next draw differs) -- the trace is sensitive to the lever it is meant to hold fixed."""
    new, old, _, J = _old_pair(world)
    bs = _trace_batches(world, new.cfg.hmax)
    t_new, t_old = _trace(new, bs), _trace(old, bs)
    assert len(t_new) == len(t_old) == 200 and all(math.isfinite(x) for x in t_new)
    assert t_new == t_old, f"first divergence at step {next(i for i, (a, b_) in enumerate(zip(t_new, t_old)) if a != b_)}"
    assert t_new[0] != t_new[-1], "the trace moves (the optimiser stepped)"
    new2, _, _, _ = _old_pair(world)
    new2.cfg = dataclasses.replace(new2.cfg, code_mix="kind")
    t_kind = _trace(new2, bs, steps=30)
    assert t_kind[0] == t_new[0], "step 0: p_gt = 1 makes kind and whole the same mask"
    assert t_kind != t_new[:30], "the lever changes the trace once p_gt < 1"


# =============================================================================================================================================
# (C) THE MASK: PREREG PROOF 3
# =============================================================================================================================================
def _kind_world(world, n=16):
    """(whole model, kind model with the same weights, a batch with unlabelled windows)."""
    model, b = _mixed_batch(world, n=n)
    return model, _kind_of(model), b


def test_p_gt_1_kind_equals_whole_and_the_gt_code_is_read_wherever_the_label_exists(world):
    mw, mk, b = _kind_world(world)
    ow, ok = [], []
    for m, store in ((mw, ow), (mk, ok)):
        cap = _spy(m)
        torch.manual_seed(1)
        m._forward(b, 1.0, True, torch.Generator().manual_seed(9))
        store.append(cap)
    gt, has = mw._code_gt(b["lab_t"]), b["lab_t"]["has"]
    assert has.any() and (~has).any()
    for cap in (ow[0], ok[0]):
        assert torch.equal(cap["code_o"][has], gt[has]), "p_gt = 1: GT everywhere the label exists"
        assert torch.equal(cap["code_o"][~has], cap["code_t"][~has]), "p_gt = 1: T's own where there is no label"
    assert torch.equal(ow[0]["code_o"], ok[0]["code_o"]), "p_gt = 1: kind == whole"


def test_p_gt_0_both_modes_hand_o_exactly_t_s_own_code(world):
    mw, mk, b = _kind_world(world)
    _, own = _own_outputs(mw, b)
    for m in (mw, mk):
        cap = _spy(m)
        m._forward(b, 0.0, True, torch.Generator().manual_seed(9))
        assert torch.equal(cap["code_o"], cap["code_t"]), "p_gt = 0: O reads T's own code"
        assert torch.equal(cap["code_o"], own)


def _classify(model, b, p, draws, seed0=100):
    """Run `_forward` `draws` times with fresh generators; per labelled sample and unit: True if O read the GT value, False if T's own. Also returns the unlabelled windows' read-GT count."""
    gt, has = model._code_gt(b["lab_t"]), b["lab_t"]["has"]
    rows, unlabelled_gt = [], 0
    for d in range(draws):
        cap = _spy(model)
        with torch.no_grad():
            model._forward(b, p, True, torch.Generator().manual_seed(seed0 + d))
        code, own = cap["code_o"], cap["code_t"]
        is_gt = torch.stack([(code[:, u] == gt[:, u]).all(1) & ~(own[:, u] == gt[:, u]).all(1) for u in U_IDX], 1)
        is_own = torch.stack([(code[:, u] == own[:, u]).all(1) & ~(own[:, u] == gt[:, u]).all(1) for u in U_IDX], 1)
        same = torch.stack([(own[:, u] == gt[:, u]).all(1) for u in U_IDX], 1)
        assert not (same & has[:, None]).any(), "a unit of T's own code equals the GT on a labelled window: its source could not be told"
        assert (has[:, None] & ~(is_gt | is_own)).sum() == 0, "a cell of the code O read is neither the GT nor T's own"
        unlabelled_gt += int((is_gt & ~has[:, None]).sum())
        rows.append(is_gt[has])
    return torch.cat(rows).numpy(), unlabelled_gt


def test_p_gt_half_the_per_sample_gt_fraction_is_partial_across_4096_draws_and_whole_is_all_or_nothing(world):
    mw, mk, b = _kind_world(world, n=16)
    nlab = int(b["lab_t"]["has"].sum())
    draws = math.ceil(4096 / 16)
    assert draws * 16 == 4096
    k_rows, k_unl = _classify(mk, b, 0.5, draws)
    w_rows, w_unl = _classify(mw, b, 0.5, draws)
    assert k_rows.shape == w_rows.shape == (draws * nlab, 7) and k_unl == w_unl == 0, "an unlabelled window never reads the GT code"
    fk, fw = k_rows.mean(1), w_rows.mean(1)
    partial_k, partial_w = float(((fk > 0) & (fk < 1)).mean()), float(((fw > 0) & (fw < 1)).mean())
    assert partial_w == 0.0 and set(np.unique(fw)) <= {0.0, 1.0}, "whole: one switch per sample -- the per-sample fraction is 0 or 1"
    assert partial_k > 0.9 and ((fk > 0) & (fk < 1)).sum() > 100, f"kind: a PARTIAL code is the rule, not the exception ({partial_k:.3f})"
    # P(a sample is all-GT or all-own) = 2 / 2^7 = 1.6 % at p = 0.5
    assert partial_k == pytest.approx(1 - 2 / 128, abs=0.02)
    # each unit's expected GT fraction is p (B's per-cell fraction), the units are independent
    rate = k_rows.mean(0)
    assert np.all(np.abs(rate - 0.5) < 0.06), dict(zip(UNITS, rate.round(3)))
    assert np.abs(w_rows.mean(0) - 0.5).max() < 0.12, "whole's marginals are p too (a coarser sample: one switch per sample)"
    c = np.corrcoef(k_rows.T.astype(float))
    off = c[~np.eye(7, dtype=bool)]
    assert np.abs(off).max() < 0.12, "kind: independent switches per unit"
    cw = np.corrcoef(w_rows.T.astype(float))
    assert np.allclose(cw, 1.0), "whole: every unit moves with the one switch"


@pytest.mark.parametrize("p", [0.2, 0.8])
def test_each_units_expected_gt_fraction_equals_p_gt_so_only_the_joint_changes(world, p):
    mw, mk, b = _kind_world(world, n=16)
    k_rows, _ = _classify(mk, b, p, 120)
    assert np.all(np.abs(k_rows.mean(0) - p) < 0.07), (p, dict(zip(UNITS, k_rows.mean(0).round(3))))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_kind_is_the_exact_bernoulli_draw_per_sample_and_unit_in_the_documented_order(world, p):
    """Replay the draws from the same generator: S's switch first (a [B] draw), then T->O's [B, 7] draw broadcast through CODE_UNIT_OF; `has` gates. The code O read is exactly
    where(mask, GT, T's own)."""
    mw, mk, b = _kind_world(world, n=32)
    B = b["feats"].shape[0]
    has, gt = b["lab_t"]["has"], mk._code_gt(b["lab_t"])
    unit = torch.as_tensor(M.CODE_UNIT_OF, dtype=torch.long)
    nontrivial = 0
    for seed in (3, 4, 5):
        g = torch.Generator().manual_seed(seed)
        torch.rand(B, generator=g)                                            # S's switch
        draw = torch.rand(B, 7, generator=g) < p
        want = (draw & has[:, None])[:, unit]
        cap = _spy(mk)
        mk._forward(b, p, True, torch.Generator().manual_seed(seed))
        assert torch.equal(cap["code_o"], torch.where(want, gt, cap["code_t"])), (p, seed)
        nontrivial += int(bool(want.any() and (~want & has[:, None]).any()))
    assert nontrivial >= 2, "the replayed masks are not all-or-nothing (the identity is not vacuous)"


def test_the_mask_function_is_gated_by_has_and_broadcasts_each_unit_to_exactly_its_indices():
    m = M.V8FMini.__new__(M.V8FMini)
    has = torch.tensor([True, False, True, True])
    for seed in range(5):
        mask = m._code_mix_mask(has, 0.5, torch.Generator().manual_seed(seed))
        assert mask.shape == (4, 60) and mask.dtype == torch.bool and not mask[1].any()
        for u, idx in zip(UNITS, U_IDX):
            blk = mask[:, idx]
            assert torch.equal(blk, blk[:, :1].expand_as(blk)), f"unit {u}: every index of a unit moves together"
    assert m._code_mix_mask(has, 1.0, torch.Generator().manual_seed(0))[has].all(), "p = 1: every unit GT where labelled"
    assert not m._code_mix_mask(has, 0.0, torch.Generator().manual_seed(0)).any(), "p = 0: nothing GT"


def test_kind_and_whole_differ_in_the_loss_at_a_mid_p_and_agree_at_the_ends(world):
    """The lever is not vacuous in the objective (O's input differs, so the loss does) and is exactly neutral where the PREREG says it is."""
    mw, _, b = _kind_world(world, n=16)
    g = torch.Generator().manual_seed(8)                       # at INIT O's code read is zero-gated (the loss would not see the code at all): move every weight, identically in both models
    with torch.no_grad():
        for prm in mw.parameters():
            if prm.requires_grad:
                prm.add_(0.05 * torch.randn(prm.shape, generator=g))
    mk = _kind_of(mw)
    for p, equal in ((1.0, True), (0.0, True), (0.5, False)):
        lw = mw.loss(b, p_gt=p, gen=torch.Generator().manual_seed(2))["terms"]["total"].detach()
        lk = mk.loss(b, p_gt=p, gen=torch.Generator().manual_seed(2))["terms"]["total"].detach()
        assert bool(torch.equal(lw, lk)) is equal, (p, float(lw), float(lk))


# =============================================================================================================================================
# (D) G-DVB: PREREG PROOF 1
# =============================================================================================================================================
def _gate_parts(world, argv, **over):
    from tanitad.data import v8f_mini_data as DD
    J = DD.load_join(Path(world["root"]) / "join_train.npz")
    a = _args(argv)
    cfg, model, opt, info = _built(a, J, **over)
    return a, cfg, model, opt, info


def _built(a, J, **over):
    cfg = TR.make_config(a)
    keep = {k: v for k, v in _T.TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    ws = cfg.window_start_effective
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None,
            "window_start": ws, "window_t_min": ws, "ds_window_start": ws}
    return cfg, model, opt, info


def test_the_flag_has_its_entry_and_a_new_flag_without_one_is_refused(world, monkeypatch):
    assert TR.FLAG_CHECKS.get("code_mix") == "lever:code_mix"
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert "code_mix" in dests and dests == set(TR.FLAG_CHECKS), "a flag without an entry, or an entry without a flag, is a launch blocker"
    a, cfg, model, opt, info = _gate_parts(world, ["--code-mix", "kind"])
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "code_mix"})
    assert any("code_mix" in m and "without a declared-vs-built entry" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


@pytest.mark.parametrize("argv, over", [([], {}), (["--code-mix", "kind"], {}), (["--code-mix", "whole"], {})])
def test_gdvb_passes_what_was_built_and_nothing_else_moves(world, argv, over):
    a, cfg, model, opt, info = _gate_parts(world, argv, **over)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    assert model.built_levers()["code_mix"] == a.code_mix


def test_gdvb_refuses_a_declared_kind_the_model_did_not_build_and_the_reverse(world):
    a, cfg, model, opt, info = _gate_parts(world, ["--code-mix", "kind"], code_mix="whole")
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("lever code_mix: declared kind but built whole" in m for m in bad), bad
    a, cfg, model, opt, info = _gate_parts(world, [], code_mix="kind")
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("lever code_mix: declared whole but built kind" in m for m in bad), bad


@pytest.mark.parametrize("p0, p1, why", [(1.0, 1.0, "every unit is GT"), (0.0, 0.0, "never teacher-forced")])
def test_gdvb_refuses_kind_on_a_schedule_where_it_does_nothing(world, p0, p1, why):
    a, cfg, model, opt, info = _gate_parts(world, ["--code-mix", "kind", "--p-gt-start", str(p0), "--p-gt-end", str(p1)])
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("a lever that is declared and does nothing" in m and why in m for m in bad), bad
    a, cfg, model, opt, info = _gate_parts(world, ["--p-gt-start", str(p0), "--p-gt-end", str(p1)])
    assert TR.declared_vs_built(a, cfg, model, opt, info) == [], "whole on a constant schedule is the pre-existing behaviour, not refused"


def test_the_probe_reads_the_code_o_receives_kind_is_partial_whole_is_not_and_a_wrong_mix_is_refused(world):
    model, b = _mixed_batch(world, n=16)
    cfg_w = model.cfg
    pw = TR.code_mix_probe(model, b, cfg_w)
    assert pw["pass"] is True and pw["code_mix"] == "whole" and pw["n_partial"] == 0, "the control reads EXACTLY 0 partial codes"
    assert pw["n_all_gt"] > 0 and pw["n_all_own"] > 0
    mk = _kind_of(model)
    pk = TR.code_mix_probe(mk, b, mk.cfg)
    assert pk["pass"] is True and pk["code_mix"] == "kind" and pk["n_partial"] > 0
    assert set(pk["unit_gt_rate"]) == set(UNITS)
    # declared kind, built whole (the model never mixes): REFUSED
    with pytest.raises(AssertionError, match="NO labelled sample read a partly-GT code"):
        TR.code_mix_probe(model, b, dataclasses.replace(cfg_w, code_mix="kind"))
    # declared whole, built kind (the model mixes): REFUSED
    with pytest.raises(AssertionError, match="whole is one switch per sample"):
        TR.code_mix_probe(mk, b, dataclasses.replace(mk.cfg, code_mix="whole"))
    # a mask that only ever gives all or nothing is a switch, not a mix: REFUSED even when the config says kind
    broken = _kind_of(model)
    broken._code_mix_mask = lambda has, p, gen=None: (torch.rand(has.shape[0], 1, generator=gen) < p).expand(-1, 60) & has[:, None]
    with pytest.raises(AssertionError, match="NO labelled sample read a partly-GT code"):
        TR.code_mix_probe(broken, b, broken.cfg)


# =============================================================================================================================================
# (E) REACHABILITY FROM THE ACTUAL CALLER: PREREG PROOF 4
# =============================================================================================================================================
def _train(out, extra, steps=4):
    """A real `TR.main` run on the synthetic world with spies on the mixing line and on what O is handed."""
    cap = {"mask": [], "own": None, "gt": None, "o": []}
    mix0, pred0, gt0, run_o0 = M.V8FMini._code_mix_mask, M.V8FMini._code_pred, M.V8FMini._code_gt, M.V8FMini.run_o

    def mix(self, has, p, gen=None):
        m = mix0(self, has, p, gen)
        cap["mask"].append({"p": float(p), "grad": torch.is_grad_enabled(), "m": m.detach().clone(), "line": sys._getframe(1).f_lineno})
        return m

    def pred(self, tout):
        c = pred0(self, tout)
        cap["own"] = c.detach().clone()
        return c

    def gt(self, lab):
        g = gt0(self, lab)
        cap["gt"] = g.detach().clone()
        return g

    def run_o(self, bb, ad, last, ck_o, ck_ok, code):
        cap["o"].append({"code": code.detach().clone(), "own": cap["own"], "gt": cap["gt"], "grad": torch.is_grad_enabled(), "n_mask": len(cap["mask"])})
        return run_o0(self, bb, ad, last, ck_o, ck_ok, code)
    M.V8FMini._code_mix_mask, M.V8FMini._code_pred, M.V8FMini._code_gt, M.V8FMini.run_o = mix, pred, gt, run_o
    try:
        assert TR.main(["--out", str(out), *COMMON, "--steps", str(steps), *extra]) == 0
    finally:
        M.V8FMini._code_mix_mask, M.V8FMini._code_pred, M.V8FMini._code_gt, M.V8FMini.run_o = mix0, pred0, gt0, run_o0
    out = Path(out)
    return {"cap": cap, "ck": torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False),
            "sc": json.loads((out / "selfcheck.json").read_text(encoding="utf-8")), "sm": json.loads((out / "summary.json").read_text(encoding="utf-8")),
            "argv": json.loads((out / "argv.json").read_text(encoding="utf-8"))["argv"],
            "log": [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("mixo_runs")
    return {"whole": _train(base / "whole", []), "kind": _train(base / "kind", ["--code-mix", "kind"])}


def test_trainer_argv_reaches_the_config_the_checkpoint_the_records_and_the_gate(runs):
    r = runs["kind"]
    assert r["argv"].count("--code-mix") == 1 and r["argv"][r["argv"].index("--code-mix") + 1] == "kind", "the flag is NAMED in the argv (a default of whole would otherwise test nothing)"
    assert r["ck"]["cfg"]["code_mix"] == "kind", "the checkpoint records the lever"
    assert r["sm"]["levers"]["code_mix"] == "kind" and r["sc"]["levers"]["code_mix"] == "kind"
    pr = r["sc"]["code_mix_probe"]
    assert r["sc"]["declared_vs_built"] == "PASS" and pr["pass"] is True and pr["code_mix"] == "kind" and pr["n_partial"] > 0
    assert r["sm"]["done"] is True
    w = runs["whole"]
    assert "--code-mix" not in w["argv"] and w["ck"]["cfg"]["code_mix"] == "whole" and w["sc"]["levers"]["code_mix"] == "whole"
    assert w["sc"]["code_mix_probe"]["n_partial"] == 0 and w["sc"]["code_mix_probe"]["pass"] is True, "the default run's probe reads exactly 0 partial codes"


def test_the_mixing_line_runs_on_every_training_step_with_the_scheduled_p_gt_and_o_receives_its_output(runs):
    r = runs["kind"]
    steps = r["sm"]["steps"]
    grad = [m for m in r["cap"]["mask"] if m["grad"]]
    train, gate = grad[-steps:], grad[:-steps]                       # the launch gate's G-LIVE forwards (p_gt 0.5, probe batch) come first; then one mix per training step
    assert len(train) == steps >= 4 and all(m["p"] == 0.5 for m in gate), "one mix per training step, after the gate's own forwards"
    sched = [x["p_gt"] for x in r["log"]]
    assert sched and sched == sorted(sched, reverse=True), "the loop's own p_gt (train_log.jsonl) falls with the schedule"
    assert [round(m["p"], 9) for m in train][: len(sched)] == [round(p, 9) for p in sched], "the mask drew at the loop's p_gt"
    o_train = [o for o in r["cap"]["o"] if o["grad"]][-steps:]
    assert len(o_train) == steps
    for m, o in zip(train, o_train):
        assert m["m"].shape == o["code"].shape == o["gt"].shape
        assert torch.equal(o["code"], torch.where(m["m"], o["gt"], o["own"])), "the code run_o was handed IS where(mask, GT, own)"
    assert any(bool(m["m"].any()) for m in train[1:]) and any(bool(m["m"].any(1).any() and (~m["m"]).any()) for m in train), "the code O reads mixes sources"


def test_the_default_run_never_calls_the_mixing_line_and_o_reads_whole_codes(runs):
    w = runs["whole"]
    assert [m for m in w["cap"]["mask"] if m["grad"]] == [], "whole: the mixing line is not on the path"
    assert w["sm"]["steps"] == runs["kind"]["sm"]["steps"] and w["sc"]["info"]["n_windows"] == runs["kind"]["sc"]["info"]["n_windows"], "same budget, same windows"
    assert w["sm"]["levers"]["params"] == runs["kind"]["sm"]["levers"]["params"], "the lever adds no parameter"


def test_the_mixing_call_site_is_in_forward_and_its_result_is_what_run_o_receives(runs):
    """The runtime call site of the mixing line is `_forward` (file:line from the frame, not from a grep)."""
    lines = {m["line"] for m in runs["kind"]["cap"]["mask"]}
    src = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py").read_text(encoding="utf-8").split("\n")
    assert len(lines) == 1
    ln = lines.pop()
    assert "self._code_mix_mask(b[\"lab_t\"][\"has\"], p_gt, gen)" in src[ln - 1] and "torch.where" in src[ln - 1]
    fwd = next(i for i, x in enumerate(src, 1) if x.startswith("    def _forward("))
    nxt = next(i for i, x in enumerate(src, 1) if i > fwd and x.startswith("    def "))
    assert fwd < ln < nxt, "inside V8FMini._forward"
    assert any("run_o(b, adapt, last_now, ck_for_o, ck_ok, code)" in x for x in src[ln:nxt]), "...and `code` goes on to O"


def test_the_trainer_source_has_every_hop_of_the_trace():
    """Static anchors, one per hop (the line numbers are printed by raw/2026-10-25-mixo/trace_reachability.py)."""
    tr = (REPO / "stack" / "scripts" / "v8f_mini_train.py").read_text(encoding="utf-8")
    mod = (REPO / "stack" / "tanitad" / "models" / "v8f_mini.py").read_text(encoding="utf-8")
    assert '"--code-mix", dest="code_mix", choices=M.CODE_MIX_MODES, default="whole"' in tr
    assert "code_mix=a.code_mix" in tr and 'FLAG_CHECKS' in tr and '"code_mix": "lever:code_mix"' in tr
    assert "code_mix_probe(model, probe, cfg)" in tr and 'sc = {"declared_vs_built": "PASS", "live": live, "code_mix_probe": mix' in tr
    assert "code_mix: str = \"whole\"" in mod and 'if cfg.code_mix == "kind":' in mod and "code = torch.where(self._code_mix_mask(b[\"lab_t\"][\"has\"], p_gt, gen), self._code_gt(b[\"lab_t\"]), code)" in mod
    assert '"code_mix": cfg.code_mix' in mod


def test_the_eval_never_teacher_forces_so_code_mix_cannot_reach_it(world):
    """Eval is unchanged: `forward_infer` calls `_forward(batch, 0.0, False)` -- use_labels False, so the mixing branch is unreachable and a kind model scores like a whole one."""
    mw, mk, b = _kind_world(world)
    for m in (mw, mk):
        m.eval()
    with torch.no_grad():
        ow, ok = mw.forward_infer(b), mk.forward_infer(b)
    assert all(torch.equal(ow[k], ok[k]) for k in ow if isinstance(ow[k], torch.Tensor))
    src = (REPO / "stack" / "scripts" / "v8f_mini_eval.py").read_text(encoding="utf-8")
    assert "code_mix" not in src, "the evaluator has no code_mix path"


# =============================================================================================================================================
# (F) THE READER (code/v8f_mixo_read.py) -- needs the v8F-mini research package: SKIPPED, not failing, in a checkout that has only `stack/`.
# A synthetic straight-road world whose answers are analytic (test_v8f_dk.World), whose B side is the REAL banked M2b run records (raw/full/runs), real files on disk, the real `main`.
# The X arm is derived from test_v8f_hist's world (an arm whose evals differ from B's by a controlled ADE6 delta), renamed to X's tags, with the dose files of PREREG_V8F_TAC Amendments 1/2 added
# for both arms (r(0.5) set exactly), X's run records = B's real records + `--code-mix kind` and the build facts, and a checkpoint sha256 json as the chain writes it.
# =============================================================================================================================================
BANKED = PKG / "raw" / "full" / "runs"
needs_pkg = pytest.mark.skipif(not (READ_P.exists() and BANKED.exists() and (HERE_T / "test_v8f_hist.py").exists()), reason="the v8F-mini package is not in this checkout")
R = _load("v8f_mixo_read_under_test", READ_P) if READ_P.exists() else None
_H = _load("test_v8f_hist_for_mixo", HERE_T / "test_v8f_hist.py") if (READ_P.exists() and (HERE_T / "test_v8f_hist.py").exists()) else None
XT = "M2b-mixo"


@pytest.fixture
def no_known(monkeypatch):
    """The banked premises (F 0.1207, LAMBDA +0.2678 / +0.1667, S5 excess) are the REAL banked M2b values; a synthetic world cannot hold them. Everything else is tested with the check replaced;
    `test_known_values_are_checked...` and the real-data proofs (raw/2026-10-25-mixo/real_proofs.py) run the real one. B's r(0.5) = 0.36 / 0.21 is NOT replaced: the worlds hold it exactly."""
    monkeypatch.setattr(R.PGT, "known_values", lambda *a, **k: {"checked": False, "why": "synthetic test input"})


def _ej(path, fn):
    """Edit a json file IN PLACE (`fn` mutates it; whatever it returns is ignored: a `.pop(...)` must not replace the file by the popped value)."""
    p = Path(path)
    j = json.loads(p.read_text(encoding="utf-8"))
    fn(j)
    p.write_text(json.dumps(j), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mworld(d, seeds=(0, 1), *, d_ade=-0.7, r_x=(0.60, 0.45), d1_x=(0.9, 0.9), r_b=(0.36, 0.21), d1_b=(1.2045, 1.0508), same_as_b=False, **hw):
    """Synthetic evals + runs for B (M2b, both seeds, the REAL banked records) and X = B + --code-mix kind for `seeds`. Knobs: `d_ade` = ADE6(X) - ADE6(B); `r_x` / `r_b` = r(0.5) of the
    arms (the constraints oracle at alpha 0.5 buys r of the full move d1); `d1_*` = d_cons(1), the full constraints move in metres; `same_as_b` = X's evals ARE B's (X against itself)."""
    f = _H._hworld(d, seeds, d_ade=d_ade, same_as_b=same_as_b, **hw)
    ev, ru, N, eid = f["eval"], f["runs"], f["w"].n, f["w"].eid
    if same_as_b:
        r_x, d1_x = r_b, d1_b
    for s in seeds:
        for suf in ("", "-tacabs", "-tacorc", "-tacorc-all-latcons"):                                  # X's four base evals are the H world's, renamed and pointed at X's checkpoint
            for ext in ("npz", "json"):
                src, dst = ev / f"eval_M2b-h10_s{s}{suf}.{ext}", ev / f"eval_{XT}_s{s}{suf}.{ext}"
                if ext == "json":
                    j = json.loads(src.read_text(encoding="utf-8"))
                    if "ckpt" in j:
                        j["ckpt"] = f"D:\\runs\\M2b_s{s}_mixo\\ckpt_last.pt"
                    dst.write_text(json.dumps(j), encoding="utf-8")
                else:
                    shutil.copyfile(src, dst)
                src.unlink()
    fb = np.load(ev / "eval_M2b_s0-tacabs.npz")["tac_oracle_fallback"]
    for tag, arm_seeds, rs, d1s, ck in (("M2b", (0, 1), r_b, d1_b, False), (XT, seeds, r_x, d1_x, True)):
        for s in arm_seeds:
            std = np.load(ev / f"eval_{tag}_s{s}.npz")
            prof = 1.0 + 0.3 * np.random.default_rng(900 + s).normal(size=N)
            prof /= prof.mean()                                                                   # mean 1: d_cons(a) = a-th share of d1 exactly
            base = np.asarray(std["ade6"], np.float64)
            files = {"-tacpred": (base, None), "-tacorc-constraints": (base - d1s[s] * prof, 1.0 if ck else None), "-tacorc-constraints-a50": (base - rs[s] * d1s[s] * prof, 0.5)}
            for suf, (a, alpha) in files.items():
                np.savez(ev / f"eval_{tag}_s{s}{suf}.npz", pred=std["pred"], ade6=a, eid=eid, tac_oracle_fallback=fb)
                j = {"tac": "pred"} if suf == "-tacpred" else {"tac_manipulation": {"mode": "oracle", "groups": "constraints", "n_code_dims_replaced": 26, **({"alpha": alpha} if alpha is not None else {})}}
                if ck:
                    j["ckpt"] = f"D:\\runs\\M2b_s{s}_mixo\\ckpt_last.pt"
                (ev / f"eval_{tag}_s{s}{suf}.json").write_text(json.dumps(j), encoding="utf-8")
    sha = {"M2b_s0": R.BANKED_CKPT_SHA["M2b_s0"], "M2b_s1": R.BANKED_CKPT_SHA["M2b_s1"]}
    for s in seeds:
        shutil.rmtree(ru / f"M2b_s{s}_h10")
        real, log = _H._b_real(s)
        bl = real["summary.json"]["levers"]
        argv = list(real["argv.json"]["argv"])
        at = argv.index("--out")
        argv[at:at] = ["--code-mix", "kind"]
        argv[argv.index("--out") + 1] = f"D:\\runs\\M2b_s{s}_mixo"
        lev = {**bl, "cons_absent": "mask", "p_gt_start": 1.0, "p_gt_end": 0.2, "residual_zero_init": False, "hist": 4, "hmax": 16, "window_start": 3, "code_mix": "kind",
               "w": {**bl["w"], "w_tac": 1.0, "w_cons": 0.5}}
        a_sha = f"x{s}M2b"
        sm = {**real["summary.json"], "argv_sha": a_sha, "levers": lev}
        live = {**(real["selfcheck.json"].get("live") or {}), "pass": True, "control_prior_grad_sum": 0.0}
        sc = {**real["selfcheck.json"], "declared_vs_built": "PASS", "argv_sha": a_sha, "levers": dict(lev, residual_zero_init=True), "live": live,
              "code_mix_probe": {"code_mix": "kind", "p_gt": 0.5, "n_labelled": 40, "n_draws_samples": 160, "n_partial": 150, "n_all_gt": 5, "n_all_own": 5, "pass": True},
              "info": {**(real["selfcheck.json"].get("info") or {}), "window_start": 3, "window_t_min": 3, "ds_window_start": 3}}
        rd = ru / f"M2b_s{s}_mixo"
        rd.mkdir()
        (rd / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
        (rd / "argv.json").write_text(json.dumps({"argv": argv, "sha": a_sha}), encoding="utf-8")
        (rd / "selfcheck.json").write_text(json.dumps(sc), encoding="utf-8")
        (rd / "train_log.jsonl").write_text(log, encoding="utf-8")
        torch.save({"model": {"time": torch.zeros(4, 256), "pos": torch.zeros(40, 4)},
                    "cfg": {"code_mix": "kind", "hist": 4, "hmax": 16, "window_start": -1, "d": 256, "n_tok": 40, "p_gt_start": 1.0, "p_gt_end": 0.2},
                    "grid": "pool4x10", "argv_sha": a_sha, "step": sm["steps"], "arm": "M2b", "seed": s}, rd / "ckpt_last.pt")
        sha[f"{XT}_s{s}_trained"] = sha[f"{XT}_s{s}_after_evals"] = _sha(rd / "ckpt_last.pt")
    ckj = d / "ckpt_sha256.json"
    ckj.write_text(json.dumps(sha), encoding="utf-8")
    args = list(f["args"])
    k = args.index("--out")
    args[k:k + 2] = ["--out", str(d / "mixo.json")]
    n_ep = int(np.unique(eid).size)
    f["args"] = args + ["--ckpt-sha", str(ckj), "--n-windows", str(N), "--n-episodes", str(n_ep)]
    f["ck"], f["sha"], f["out"] = ckj, sha, d / "mixo.json"
    return f


def _go(f, stage="final", extra=()):
    rc = R.main(["--stage", stage, *f["args"], *extra])
    return rc, (json.loads(f["out"].read_text(encoding="utf-8")) if rc == 0 else None)


def _xr(f, s=0):
    return f["runs"] / f"M2b_s{s}_mixo"


def _br(f, s=0):
    return f["runs"] / f"M2b_s{s}"


def _xe(f, s, suf=""):
    return f["eval"] / f"eval_{XT}_s{s}{suf}"


def _argv_edit(path, fn):
    def go(j):
        j["argv"] = fn(list(j["argv"]))
    _ej(path, go)


def _set_flag(argv, flag, val):
    argv[argv.index(flag) + 1] = val
    return argv


def _ck_edit(f, fn, s=0):
    p = _xr(f, s) / "ckpt_last.pt"
    o = torch.load(p, weights_only=False)
    fn(o)
    torch.save(o, p)


def _lev(path, fn):
    def go(j):
        fn(j["levers"])
    _ej(path, go)


def _npz_edit(path, fn):
    z = dict(np.load(path))
    fn(z)
    np.savez(path, **z)


def _shift_eid(z):
    z["eid"] = np.asarray(z["eid"]).copy()
    z["eid"][0] += 1


def _log_edit(f, fn, s=0):
    p = _xr(f, s) / "train_log.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    fn(rows)
    p.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def _sched(step, total, p0, p1):                                           # independent of the reader's and the model's
    return p0 + (p1 - p0) * min(max(step / total, 0.0), 1.0)


def _fewer_steps(f, n):
    """A run that stopped at `n` steps, with its own consistent log (so the step-count check, not the log check, is what must refuse)."""
    _ej(_xr(f) / "summary.json", lambda j: j.update(steps=n))
    _log_edit(f, lambda r: r.__setitem__(slice(None), [dict(x, p_gt=_sched(x["step"] - 1, n, 1.0, 0.2)) for x in r if x["step"] <= n]))


def _seed_worse(f, s):
    """X's seed-s ADE6 worse than B's: shift its plan laterally beyond 2 s (ade6 rewritten to match, as the eval would). The three dose files move by the same amount per window, so the
    dose chain (tacpred == the standard eval, d_cons) is untouched."""
    w = f["w"]
    p = f["eval"] / f"eval_{XT}_s{s}.npz"
    old = np.asarray(np.load(p)["ade6"], np.float64)
    for suf in ("", "-tacabs"):
        p = f["eval"] / f"eval_{XT}_s{s}{suf}.npz"
        z = dict(np.load(p))
        z["pred"] = z["pred"].copy()
        z["pred"][:, 10:, 1] += 3.0
        z["ade6"] = np.linalg.norm(z["pred"].astype(np.float64) - w.gt, axis=-1).mean(1)
        np.savez(p, **z)
        new = z["ade6"]
    for suf in ("-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50"):
        _npz_edit(f["eval"] / f"eval_{XT}_s{s}{suf}.npz", lambda z: z.update(ade6=np.asarray(z["ade6"], np.float64) + (new - old)))


# ---- the committed decision table --------------------------------------------------------------------------------------------------------------
@needs_pkg
def test_reader_decision_table_pass_and_the_two_fails(tmp_path, no_known):
    rc, r = _go(_mworld(tmp_path / "pass"))
    assert rc == 0 and r["VERDICT"] == "PASS" and r["DECISION"] is None and r["table_cell"] == "n/a" and r["base"] == "M2b", r["reading"]
    assert R.NEXT_TEXT["PASS"] in r["next"]
    F = r["F"]
    ev = tmp_path / "pass" / "eval"
    assert F == pytest.approx(abs(np.load(ev / "eval_M2b_s0.npz")["ade6"].mean() - np.load(ev / "eval_M2b_s1.npz")["ade6"].mean()), abs=1e-12) and r["two_F_bar"] == 0.2415
    for s in ("s0", "s1"):
        row = r["rows"][s]
        assert row["primary"]["point"] < -0.2415 and row["primary"]["separated"] and row["judge"]["passed"] and row["judge"]["bar_2F_m"] == 0.2415
    # PRIMARY missed at seed 0, r(0.5) rose by >= 0.10 over B's 0.36: O LEARNED
    rc, r = _go(_mworld(tmp_path / "learned", d_ade=-0.10, r_x=(0.60, 0.45)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-O-LEARNED" and R.NEXT_TEXT["FAIL-O-LEARNED"] in r["next"], r["reading"]
    assert r["rows"]["s0"]["primary"]["separated"] and 0 < abs(r["rows"]["s0"]["primary"]["point"]) < 0.2415, "beyond 0, under 2F"
    assert r["rows"]["s0"]["dm"]["learned"] is True and r["rows"]["s0"]["dm"]["rise_over_committed"] == pytest.approx(0.24)
    # ... rose by < 0.10: O DID NOT LEARN
    rc, r = _go(_mworld(tmp_path / "dnl", d_ade=-0.10, r_x=(0.40, 0.45)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-O-DID-NOT-LEARN" and R.NEXT_TEXT["FAIL-O-DID-NOT-LEARN"] in r["next"]
    assert r["rows"]["s0"]["dm"]["learned"] is False and r["rows"]["s0"]["dm"]["rise_over_committed"] == pytest.approx(0.04)
    # worse driving is never a pass; r that FELL is no rise
    rc, r = _go(_mworld(tmp_path / "worse", d_ade=0.8, r_x=(0.20, 0.45)), "s0")
    assert rc == 0 and r["VERDICT"] == "FAIL-O-DID-NOT-LEARN" and not r["rows"]["s0"]["judge"]["passed"] and r["rows"]["s0"]["dm"]["rise_over_committed"] < 0


@needs_pkg
def test_reader_the_primary_bar_is_the_committed_2F_not_the_measured_F(tmp_path, no_known):
    rc, r = _go(_mworld(tmp_path / "a", d_ade=-0.2300), "s0")
    assert rc == 0 and r["rows"]["s0"]["primary"]["point"] > -0.2415 and r["rows"]["s0"]["judge"]["passed"] is False
    rc, r = _go(_mworld(tmp_path / "b", d_ade=-0.2600), "s0")
    assert rc == 0 and r["rows"]["s0"]["primary"]["point"] < -0.2415 and r["rows"]["s0"]["primary"]["separated"] and r["rows"]["s0"]["judge"]["passed"] is True
    assert r["DECISION"] == "CONTINUE" and "train seed 1" in r["next"] and r["VERDICT"] is None, "only a seed-0 PASS trains seed 1"
    assert r["two_F_measured"] != 0.2415, "the measured 2F of this world is not the bar: the committed constant is"


@needs_pkg
def test_reader_the_rise_bar_is_0p10_over_b_s_committed_r_and_the_boundary_is_inclusive(tmp_path, no_known):
    for r_x, learned in ((0.46, True), (0.4599, False), (0.5, True), (0.36, False)):
        rc, r = _go(_mworld(tmp_path / f"r{r_x}", d_ade=-0.10, r_x=(r_x, 0.45)), "s0")
        assert rc == 0 and r["rows"]["s0"]["dm"]["learned"] is learned, (r_x, r["rows"]["s0"]["dm"]["rise_over_committed"])
        assert r["VERDICT"] == ("FAIL-O-LEARNED" if learned else "FAIL-O-DID-NOT-LEARN")


@needs_pkg
def test_reader_a_seed_1_miss_after_a_seed_0_pass_is_classified_by_the_same_bar_and_labelled_an_extension(tmp_path, no_known):
    f = _mworld(tmp_path / "ext", r_x=(0.60, 0.50))
    _seed_worse(f, 1)
    rc, r = _go(f)
    assert rc == 0 and r["rows"]["s0"]["judge"]["passed"] and not r["rows"]["s1"]["judge"]["passed"]
    assert r["VERDICT"] == "FAIL-O-LEARNED" and r["table_cell"] == "extension" and "table_cell extension" in r["next"] and R.NEXT_TEXT["FAIL-O-LEARNED"] in r["next"]
    assert r["rows"]["s1"]["dm"]["r_B_committed"] == 0.21 and r["rows"]["s1"]["dm"]["rise_over_committed"] == pytest.approx(0.29)
    g = _mworld(tmp_path / "ext2", r_x=(0.60, 0.25))
    _seed_worse(g, 1)
    rc, r = _go(g)
    assert rc == 0 and r["VERDICT"] == "FAIL-O-DID-NOT-LEARN" and r["table_cell"] == "extension"


@needs_pkg
def test_reader_refuses_a_final_read_after_a_seed_0_miss_and_without_seed_1(tmp_path, no_known, capsys):
    f = _mworld(tmp_path / "miss", d_ade=-0.10)
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "seed 0 did NOT pass" in capsys.readouterr().err and not f["out"].exists()
    g = _mworld(tmp_path / "one", seeds=(0,))
    assert R.main(["--stage", "final", *g["args"]]) == 2
    assert "is missing" in capsys.readouterr().err
    assert R.main(["--stage", "s0", *g["args"]]) == 0, "the same files are a complete stage-1 input"


@needs_pkg
@pytest.mark.skipif(not PREREG.exists(), reason="PREREG_V8F_MIXO.md is not in this checkout")
def test_the_committed_next_steps_and_constants_are_verbatim_the_preregs():
    text = PREREG.read_text(encoding="utf-8")
    for k, v in R.NEXT_TEXT.items():
        assert v in text, f"{k}: {v!r} is not in PREREG_V8F_MIXO.md"
        assert v in R.NEXT[k]
    assert "| **PASS** | PRIMARY met at both seeds |" in text and "| **FAIL, O LEARNED** |" in text and "| **FAIL, O DID NOT LEARN** |" in text
    assert "r(0.5) rises by ≥ 0.10 over B's" in text and R.RISE_BAR == 0.10 and "r(0.5) rises by < 0.10" in text
    assert "B = 2,000" in text and R.B == 2000 and "0.1207 m, 2F = 0.2415 m" in text and R.BAR_2F == 0.2415
    assert "0.36 / 0.21" in text and (R.R_B_COMMITTED[0], R.R_B_COMMITTED[1]) == (0.36, 0.21)
    assert "Seed 1 runs only after a seed-0 PASS" in text and "0.102 m" in text
    assert "(speed, angle, tte, timegap, geom, leadgap)" in text


# ---- the identity control and the dose definition ----------------------------------------------------------------------------------------------
@needs_pkg
def test_reader_x_against_itself_reads_exactly_zero_on_every_contrast_and_nothing_passes(tmp_path, no_known):
    rc, r = _go(_mworld(tmp_path / "same", same_as_b=True, d_ade=0.0), "s0")
    assert rc == 0 and r["VERDICT"] == "FAIL-O-DID-NOT-LEARN"
    zero = {"delta": 0.0, "lo": 0.0, "hi": 0.0, "point": 0.0}
    for s in ("s0",):
        row = r["rows"][s]
        assert row["primary"]["delta"] == 0.0 and row["primary"]["lo"] == 0.0 and row["primary"]["hi"] == 0.0 and not row["judge"]["passed"]
        assert row["dm"]["rise_over_committed"] == pytest.approx(0.0, abs=1e-9) and row["dm"]["learned"] is False
        assert row["beside"]["harm"]["oracle_ade6_delta"]["delta"] == 0.0
        idc = r["identity_control"][s]
        assert idc["dose_half"] == zero and idc["dose_full"] == zero and idc["primary"] == zero and idc["oracle_ade6"] == zero


@needs_pkg
def test_dose_is_tac_a2s_definition_d_cons_half_over_d_cons_full_both_against_t_s_own_code(tmp_path, no_known):
    f = _mworld(tmp_path / "d", d_ade=-0.7, r_x=(0.55, 0.40), d1_x=(0.8, 0.7))
    rc, r = _go(f)
    assert rc == 0
    ev = f["eval"]
    for tag, key in (("M2b", "B"), (XT, "X")):
        for s in (0, 1):
            a = {suf: np.load(ev / f"eval_{tag}_s{s}{suf}.npz")["ade6"] for suf in ("-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50")}
            want = (a["-tacorc-constraints-a50"].mean() - a["-tacpred"].mean()) / (a["-tacorc-constraints"].mean() - a["-tacpred"].mean())
            dm = r["rows"][f"s{s}"]["dm"]
            got = dm["r_X"] if key == "X" else dm["r_B_measured"]
            assert got == pytest.approx(want, abs=1e-12)
            assert dm[key]["r_tac_a2"] == pytest.approx(want, abs=2e-4), "TAC_A2's own (4-dp-rounded) form agrees to the rounding"
    assert [r["rows"][s]["dm"]["r_B_measured"] for s in ("s0", "s1")] == [pytest.approx(0.36), pytest.approx(0.21)], "B's banked dose reproduces the PREREG's numbers"
    assert r["rows"]["s0"]["dm"]["X"]["full"]["point"] == pytest.approx(-0.8) and r["rows"]["s1"]["dm"]["X"]["full"]["point"] == pytest.approx(-0.7)


@needs_pkg
def test_dose_is_undefined_not_zero_when_the_full_constraints_oracle_does_not_help_x(tmp_path, no_known):
    rc, r = _go(_mworld(tmp_path / "u", d_ade=-0.10, d1_x=(-0.3, 0.9)), "s0")
    dm = r["rows"]["s0"]["dm"]
    assert rc == 0 and dm["r_X"] is None and dm["r_defined"] is False and dm["learned"] is False and r["VERDICT"] == "FAIL-O-DID-NOT-LEARN"


@needs_pkg
def test_harm_is_the_paired_gt_code_oracle_ade6_x_minus_b_and_is_never_a_bar(tmp_path, no_known):
    f = _mworld(tmp_path / "h", d_ade=-0.7)
    for s in (0, 1):                                                          # X's oracle plans worse than B's by a constant: HARM reads it, the verdict does not move
        _npz_edit(_xe(f, s, "-tacorc.npz"), lambda z: z.update(ade6=np.asarray(z["ade6"]) + 0.5))
    rc, r = _go(f)
    assert rc == 0 and r["VERDICT"] == "PASS"
    for s in ("s0", "s1"):
        h = r["rows"][s]["beside"]["harm"]
        assert h["oracle_ade6_delta"]["point"] == pytest.approx(0.5) and h["oracle_ade6"]["X"] - h["oracle_ade6"]["B"] == pytest.approx(0.5) and "never a bar" in h["note"]


@needs_pkg
def test_reader_reports_the_four_families_parameters_pipeline_and_present_slots_beside_the_verdict(tmp_path, no_known):
    rc, r = _go(_mworld(tmp_path / "b"))
    assert rc == 0
    row = r["rows"]["s0"]
    b = row["beside"]
    assert set(b["eval_json"]) == {"B", "X"} and all(b["eval_json"][k].get("tactical") and b["eval_json"][k].get("strategic") for k in ("B", "X")), "tactical + strategic (NOT paused) per arm"
    assert "NOT PAUSED" in r["strategic_family"]
    assert b["distance_keeping_k10"]["pair_vs_b"] and set(b["distance_keeping_k10"]["arms"]) == {"M2b-mixo_s0", "M2b_s0", "P10"}
    assert row["mleak"]["delta"] and row["base"]["lambda_m2b_m"] is not None, "LAMBDA beside"
    assert b["parameters"]["n_params_trainable"]["X"] == b["parameters"]["n_params_trainable"]["B"] > 0
    assert b["parameters"]["groups"]["X"] == b["parameters"]["groups"]["B"]
    assert b["pipeline"]["X"]["steps"] == b["pipeline"]["B"]["steps"] == 19164 and b["pipeline"]["X"]["n_windows"] == b["pipeline"]["B"]["n_windows"] == 306566
    assert b["pipeline"]["X"]["steps_per_s"] and b["pipeline"]["B"]["steps_per_s"]
    assert b["present_slot"]["rmse"]["X"] > 0 and b["present_slot"]["rmse_group"]["X"] and b["present_slot"]["mse_delta"]["delta"] is not None
    assert b["tacpred_repro_max_dev_m"] == {"B": 0.0, "X": 0.0}
    assert set(r["checkpoint_sha256"]) >= {"M2b_s0", "M2b_s1", "M2b-mixo_s0", "M2b-mixo_s1"}


# ---- the refusals: every precondition of an X-vs-B read is a refusal, never a quiet pass ----------------------------------------------------------
def _f_missing_family(side, fam):
    def go(f):
        p = (_xe(f, 0, ".json") if side == "X" else f["eval"] / "eval_M2b_s0.json")
        _ej(p, lambda j: j["raw"].pop(fam))
    return go


def _b_a50_shift(f):
    _npz_edit(f["eval"] / "eval_M2b_s0-tacorc-constraints-a50.npz", lambda z: z.update(ade6=np.asarray(z["ade6"]) + 0.05))


def _sha_edit(key, val):
    return lambda f: _ej(f["ck"], lambda j: j.update({key: val}))


def _other_gt(z):
    n = z["tac_code"].shape[0]
    z["tac_code"] = np.where(np.arange(n)[:, None] % 7 == 3, z["tac_code"] + 1.0, z["tac_code"])


MUT = {                                # name -> (mutator(f), a message fragment)
    "x summary missing": (lambda f: (_xr(f) / "summary.json").unlink(), "is missing"),
    "x checkpoint missing": (lambda f: (_xr(f) / "ckpt_last.pt").unlink(), "is missing"),
    "x selfcheck missing": (lambda f: (_xr(f) / "selfcheck.json").unlink(), "is missing"),
    "x not done": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(done=False)), "is not done"),
    "x n_clips": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(n_clips=4000)), "n_clips 4000"),
    "x withheld nothing": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(excluded_sha256="0" * 64)), "did not withhold"),
    "x trained on another clip list": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(train_sha256="u" * 64)), "did not train on"),
    "x trained fewer steps": (lambda f: _fewer_steps(f, 9000), "trained for 9000 steps"),
    "x trained the budget a hist-10 window set would size": (lambda f: _fewer_steps(f, 17472), "trained for 17472 steps, M2b_s0 for 19164"),
    "x trained on another window set": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(n_windows=279494)), "trained on 279494 windows"),
    "x launch gate not PASS": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j.update(declared_vs_built="FAIL")), "declared_vs_built is not PASS"),
    "x G-LIVE not pass": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j["live"].update(**{"pass": False})), "G-LIVE is not a pass"),
    "x frozen-prior control not exactly zero": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j["live"].update(control_prior_grad_sum=1e-9)), "G-LIVE is not a pass"),
    "x no code_mix probe": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j.pop("code_mix_probe")), "no passing code_mix probe"),
    "x probe did not pass": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j["code_mix_probe"].update(**{"pass": False})), "no passing code_mix probe"),
    "x probe saw no partly-GT code": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j["code_mix_probe"].update(n_partial=0)), "saw no PARTLY-GT code"),
    "x probe was for whole": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j["code_mix_probe"].update(code_mix="whole")), "was run for 'whole'"),
    "x selfcheck from another run": (lambda f: _ej(_xr(f) / "selfcheck.json", lambda j: j.update(argv_sha="zzz")), "is not this run's"),
    "x argv has no --code-mix": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: [x for x in a if x not in ("--code-mix", "kind")]), "argv --code-mix None"),
    "x argv says --code-mix whole": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: _set_flag(a, "--code-mix", "whole")), "argv --code-mix whole"),
    "x argv names --code-mix twice": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--code-mix", "kind"]), "named 2 times"),
    "b argv names --code-mix": (lambda f: _argv_edit(_br(f) / "argv.json", lambda a: a + ["--code-mix", "kind"]), "predates the flag"),
    "x argv has a second lever": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--w-wm", "0.2"]), "not one lever"),
    "x argv moved hist": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--hist", "10"]), "not one lever"),
    "x argv moved the p_gt schedule": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--p-gt-start", "0.5"]), "not one lever"),
    "x argv moved the grid": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: _set_flag(a, "--grid", "pool8x20")), "not one lever"),
    "x argv lost a flag": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: [x for x in a if x not in ("--kin-mode", "yaw")]), "not one lever"),
    "x argv is another arm": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: _set_flag(a, "--arm", "M2")), "argv --arm"),
    "x built code_mix whole (summary)": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(code_mix="whole")), "built code_mix 'whole'"),
    "x built code_mix whole (selfcheck)": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.update(code_mix="whole")), "built code_mix 'whole'"),
    "x records no built code_mix": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.pop("code_mix")), "record no built code_mix"),
    "x built another p_gt": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(p_gt_start=0.7)), "was built with p_gt_start"),
    "x records no built p_gt": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.pop("p_gt_end")), "record no built p_gt_end"),
    "x built hmax 10": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(hmax=10)), "built hmax 10"),
    "x built another weight": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l["w"].update(w_pen=0.9)), "w.w_pen"),
    "x built another w_cons": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l["w"].update(w_cons=1.0)), "w.w_cons"),
    "x built another mask": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.update(wm="a")), "built levers"),
    "x has one parameter too many": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(n_params_trainable=l["n_params_trainable"] + 1)), "parameter counts are not B's"),
    "x grew a group": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l["params"].update(T=l["params"]["T"] + 5)), "group T"),
    "x checkpoint built whole": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(code_mix="whole")), "BUILT with code_mix 'whole'"),
    "x checkpoint has no code_mix": (lambda f: _ck_edit(f, lambda o: o["cfg"].pop("code_mix")), "BUILT with code_mix None"),
    "x checkpoint built hist 10": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(hist=10)), "BUILT at hist 10"),
    "x checkpoint built another hmax": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(hmax=10)), "BUILT with hmax 10, B's is 16"),
    "x checkpoint built window_start 9": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(window_start=9)), "BUILT with window_start 9"),
    "x checkpoint built another p_gt": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(p_gt_end=0.0)), "BUILT with p_gt_end 0.0"),
    "x checkpoint is an 8x20 checkpoint": (lambda f: _ck_edit(f, lambda o: o.update(grid="pool8x20")), "are not B's"),
    "x checkpoint built 160 tokens": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(n_tok=160)), "are not B's"),
    "x checkpoint from another argv": (lambda f: _ck_edit(f, lambda o: o.update(argv_sha="other")), "another argv"),
    "x checkpoint at another step": (lambda f: _ck_edit(f, lambda o: o.update(step=100)), "checkpoint is at step 100"),
    "x checkpoint of another seed": (lambda f: _ck_edit(f, lambda o: o.update(seed=1)), "checkpoint says arm"),
    "x train log carries the wrong p_gt": (lambda f: _log_edit(f, lambda r: r[3].update(p_gt=0.99)), "p_gt"),
    "x train log missing": (lambda f: (_xr(f) / "train_log.jsonl").unlink(), "is missing"),
    "ckpt-sha json missing": (lambda f: f["ck"].unlink(), "does not exist"),
    "B's checkpoint is not the banked file": (_sha_edit("M2b_s0", "0" * 64), "is not the banked checkpoint"),
    "X's checkpoint changed during the evals": (_sha_edit(f"{XT}_s0_after_evals", "1" * 64), "not one value"),
    "X's checkpoint is not the one pulled": (lambda f: _ej(f["ck"], lambda j: j.update({f"{XT}_s0_trained": "2" * 64, f"{XT}_s0_after_evals": "2" * 64})), "not the scored checkpoint"),
    "X's trained sha missing": (lambda f: _ej(f["ck"], lambda j: j.pop(f"{XT}_s0_trained")), "not one value"),
    "x eval scored B's checkpoint": (lambda f: _ej(_xe(f, 0, ".json"), lambda j: j.update(ckpt="D:\\runs\\M2b_s0\\ckpt_last.pt")), "scored"),
    "x dose eval scored B's checkpoint": (lambda f: _ej(_xe(f, 0, "-tacorc-constraints-a50.json"), lambda j: j.update(ckpt="D:\\runs\\M2b_s0\\ckpt_last.pt")), "scored"),
    "x oracle eval scored no checkpoint": (lambda f: _ej(_xe(f, 0, "-tacorc.json"), lambda j: j.pop("ckpt") and None), "scored None"),
    "x eval is on another eval grid": (lambda f: _npz_edit(_xe(f, 0, ".npz"), _shift_eid), "eid"),
    "x dose eval is on another eval grid": (lambda f: _npz_edit(_xe(f, 0, "-tacpred.npz"), _shift_eid), "not on the same eval grid"),
    "x evals (all) are on another eval grid than B's": (lambda f: [_npz_edit(_xe(f, s, suf), _shift_eid) for s in (0, 1) for suf in (".npz", "-tacabs.npz", "-tacorc.npz", "-tacorc-all-latcons.npz")],
                                                    "not on the same eval grid as M2b's"),
    "x oracle eval has other GT codes": (lambda f: _npz_edit(_xe(f, 0, "-tacorc.npz"), _other_gt), "the GT constraint block differs"),
    "x tacpred missing": (lambda f: _xe(f, 0, "-tacpred.npz").unlink(), "is missing"),
    "x full constraints oracle missing": (lambda f: _xe(f, 0, "-tacorc-constraints.npz").unlink(), "is missing"),
    "x alpha-0.5 oracle missing": (lambda f: _xe(f, 1, "-tacorc-constraints-a50.npz").unlink(), "is missing"),
    "x alpha-0.5 record says 0.75": (lambda f: _ej(_xe(f, 0, "-tacorc-constraints-a50.json"), lambda j: j["tac_manipulation"].update(alpha=0.75)), "alpha 0.75"),
    "x full record says alpha 0.5": (lambda f: _ej(_xe(f, 0, "-tacorc-constraints.json"), lambda j: j["tac_manipulation"].update(alpha=0.5)), "alpha 0.5"),
    "x full oracle is not the constraints group": (lambda f: _ej(_xe(f, 0, "-tacorc-constraints.json"), lambda j: j["tac_manipulation"].update(groups="all-goal")), "manipulation"),
    "x full oracle replaced 25 dims": (lambda f: _ej(_xe(f, 0, "-tacorc-constraints.json"), lambda j: j["tac_manipulation"].update(n_code_dims_replaced=25)), "manipulation"),
    "x tacpred is an oracle run": (lambda f: _ej(_xe(f, 0, "-tacpred.json"), lambda j: j.update(tac="oracle")), "not T's own code"),
    "x tacpred does not reproduce the standard eval": (lambda f: _npz_edit(_xe(f, 0, "-tacpred.npz"), lambda z: z.update(ade6=np.asarray(z["ade6"]) + 0.01)), "differs from the standard eval"),
    "b a50 file does not give the PREREG's r(0.5)": (_b_a50_shift, "not the inputs the PREREG was written against"),
    "tactical family missing (X)": (_f_missing_family("X", "tactical"), "a family is missing"),
    "strategic family missing (X)": (_f_missing_family("X", "strategic"), "a family is missing"),
    "tactical family missing (B)": (_f_missing_family("B", "tactical"), "a family is missing"),
    "a grid with other window counts": (lambda f: f["args"].extend(["--n-windows", "2399"]), "committed eval-141 grid"),
    "a grid with other episode counts": (lambda f: f["args"].extend(["--n-episodes", "141"]), "committed eval-141 grid"),
    "x eval missing": (lambda f: _xe(f, 0, "-tacorc-all-latcons.npz").unlink(), "is missing"),
}


@needs_pkg
@pytest.mark.parametrize("name", list(MUT))
def test_reader_refuses_rc2_with_the_reason(tmp_path, no_known, capsys, name):
    mut, frag = MUT[name]
    f = _mworld(tmp_path / "w")
    mut(f)
    capsys.readouterr()
    rc = R.main(["--stage", "final", *f["args"]])
    err = capsys.readouterr().err
    assert rc == 2 and frag in err and "MIXO READ REFUSED" in err, f"{name}: rc {rc}; stderr: {err[-500:]}"
    assert not f["out"].exists(), "a refused read must leave no result file for the next step to find"


@needs_pkg
def test_reader_missing_inputs_are_rc2_not_a_traceback(tmp_path, no_known, capsys):
    assert R.main(["--stage", "final", "--eval-dir", str(tmp_path), "--runs-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 2
    assert "is missing" in capsys.readouterr().err
    assert R.main(["--stage", "final"]) == 2 and R.main([]) == 2, "argparse: required arguments missing"
    capsys.readouterr()


@needs_pkg
def test_reader_requires_the_ckpt_sha_json_for_a_read(tmp_path, no_known, capsys):
    f = _mworld(tmp_path / "w")
    k = f["args"].index("--ckpt-sha")
    args = f["args"][:k] + f["args"][k + 2:]
    capsys.readouterr()
    assert R.main(["--stage", "final", *args]) == 2 and "--ckpt-sha is required" in capsys.readouterr().err


@needs_pkg
def test_reader_turns_an_inherited_refusal_into_rc2_not_a_traceback(tmp_path, no_known, capsys):
    f = _mworld(tmp_path / "w")

    def odd(z):
        z["frame"] = np.asarray(z["frame"]).copy()
        z["frame"][0] += 1
    _npz_edit(f["d"] / "gt.npz", odd)
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    err = capsys.readouterr().err
    assert "MIXO READ REFUSED" in err and "no row in the lead block" in err


@needs_pkg
def test_reader_breaks_loudly_when_the_paired_instrument_is_broken(tmp_path, no_known, capsys, monkeypatch):
    f = _mworld(tmp_path / "w")
    import taniteval.ci as ci
    orig = ci.paired_episode_cluster_bootstrap

    def biased(a, b, *r, **k):
        o = orig(a, b, *r, **k)
        if np.array_equal(np.asarray(a), np.asarray(b)):
            o = dict(o, delta=o["delta"] + 1e-9)
        return o
    monkeypatch.setattr(ci, "paired_episode_cluster_bootstrap", biased)
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "IDENTITY CONTROL FAILED" in capsys.readouterr().err


@needs_pkg
def test_known_values_are_checked_and_refuse_a_world_that_does_not_hold_them(tmp_path, capsys):
    f = _mworld(tmp_path / "w")
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    assert "does not reproduce the PREREG's known values" in capsys.readouterr().err


# ---- the pre-launch check of B's side ----------------------------------------------------------------------------------------------------------
@needs_pkg
def test_check_inputs_reads_b_alone_reproduces_r_half_and_the_identity_control_is_exactly_zero(tmp_path, no_known, capsys):
    f = _mworld(tmp_path / "c", seeds=())                                                # no X run exists yet: this is the check that runs BEFORE training
    for p in list(f["eval"].glob(f"eval_{XT}*")):
        p.unlink()
    assert R.main(["--check-inputs", *f["args"]]) == 0
    out = capsys.readouterr().out
    assert "MIXO INPUTS OK" in out and "identity control exactly +0 [0, 0]" in out and "expects 7 evals per seed" in out
    j = json.loads(f["out"].read_text(encoding="utf-8"))
    assert j["base"] == "M2b" and j["F"] > 0 and j["two_F_bar"] == 0.2415 and j["expected_x_suffixes"] == list(R.X_SUFFIXES) and len(j["expected_x_suffixes"]) == 7
    assert j["dose_b"]["s0"]["r_half"] == pytest.approx(0.36) and j["dose_b"]["s1"]["r_half"] == pytest.approx(0.21)
    assert j["checkpoint_sha256"] == {"M2b_s0": R.BANKED_CKPT_SHA["M2b_s0"], "M2b_s1": R.BANKED_CKPT_SHA["M2b_s1"]}
    assert j["tacpred_repro_max_dev_m"] == {"M2b_s0": 0.0, "M2b_s1": 0.0}


@needs_pkg
@pytest.mark.parametrize("what, frag", [("a50", "is missing"), ("alpha", "not the inputs the PREREG was written against"), ("sha", "is not the banked checkpoint")])
def test_check_inputs_refuses_a_b_side_that_is_not_the_preregs(tmp_path, no_known, capsys, what, frag):
    f = _mworld(tmp_path / "c", seeds=())
    {"a50": lambda: (f["eval"] / "eval_M2b_s1-tacorc-constraints-a50.npz").unlink(), "alpha": lambda: _b_a50_shift(f), "sha": lambda: _sha_edit("M2b_s1", "0" * 64)(f)}[what]()
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert frag in capsys.readouterr().err


@needs_pkg
def test_check_inputs_breaks_loudly_when_the_paired_instrument_is_broken(tmp_path, no_known, capsys, monkeypatch):
    f = _mworld(tmp_path / "c", seeds=())
    import taniteval.ci as ci
    orig = ci.paired_episode_cluster_bootstrap

    def biased(a, b, *r, **k):
        o = orig(a, b, *r, **k)
        return dict(o, delta=o["delta"] + 1e-9) if np.array_equal(np.asarray(a), np.asarray(b)) else o
    monkeypatch.setattr(ci, "paired_episode_cluster_bootstrap", biased)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "IDENTITY CONTROL FAILED" in capsys.readouterr().err


@needs_pkg
def test_reader_constants_match_the_model_the_trainer_the_queue_and_the_chain():
    assert R.ARM_X == XT and R.X_SUFFIXES == ("", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50")
    assert R.LEVER_FLAG == "--code-mix" and R.LEVER_VAL == "kind" and R.x_dir(1) == "M2b_s1_mixo" and R.SUF == "_mixo"
    assert R.B_P_GT == (M.V8FMiniConfig().p_gt_start, M.V8FMiniConfig().p_gt_end) == (1.0, 0.2)
    assert R.LEVER_VAL in M.CODE_MIX_MODES and R.EVAL_WINDOWS == 2399 and R.EVAL_EPISODES == 141
    assert set(R.NEXT) == {None, "PASS", "FAIL-O-LEARNED", "FAIL-O-DID-NOT-LEARN"} and set(R.NEXT_TEXT) == {"PASS", "FAIL-O-LEARNED", "FAIL-O-DID-NOT-LEARN"}
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"CKPT_SHA=\((\w+) (\w+)\)", c)
    assert (m.group(1), m.group(2)) == (R.BANKED_CKPT_SHA["M2b_s0"], R.BANKED_CKPT_SHA["M2b_s1"]), "the chain and the reader assert the same two banked checkpoints"
    man = PKG / "raw" / "full" / "MANIFEST.md"
    if man.exists():
        t = man.read_text(encoding="utf-8")
        assert R.BANKED_CKPT_SHA["M2b_s0"] in t and R.BANKED_CKPT_SHA["M2b_s1"] in t, "raw/full/MANIFEST.md carries the banked sha256 values"
    # the X suffixes are the queue's own seven tags
    q = QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")
    blk = q.split("\n:evalmixo\n", 1)[1].split("\n:evalmixo_1\n", 1)[0]
    tags = re.findall(r"--tag %TG%(\S*)", blk)
    assert tags == list(R.X_SUFFIXES)


# =============================================================================================================================================
# (G) THE QUEUE STAGES (code/v8f_mini_box2_queue.cmd: gatemixo / trainmixo / evalmixo): what is PRINTED is what RUNS, one lever; every existing label is byte-identical
# =============================================================================================================================================
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
MSTAGES = ("gatemixo", "trainmixo", "evalmixo")
MIXO_LABELS = {"gatemixo", "trainmixo", "evalmixo"} | {f"evalmixo_{i}" for i in range(1, 8)}
# PREREG_V8F_KIND's four labels as they stand at c0769eaa (sha256[:16] of "\n" + the text up to the next label): PREREG_V8F_MIXO appended AFTER them and edited none
KIND_PINS = {"kind": "d1445aa3e2135915", "kindseed": "9c13c94705b73355", "kindown": "df45fbd31218aea9", "kindorc": "e7051ae3f70c3a4f"}
_K = _load("test_v8f_kind_for_mixo", HERE_T / "test_v8f_kind.py") if (QUEUE.exists() and (HERE_T / "test_v8f_kind.py").exists()) else None


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
    if clean_paths:
        for k in ("REPO", "RUNSM", "CACHEFULL", "V9", "PY"):
            env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x]


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
@pytest.mark.skipif(_K is None, reason="test_v8f_kind.py (the pre-existing label pins) is not in this checkout")
def test_queue_mixo_is_crlf_every_existing_label_is_byte_identical_and_only_the_ten_new_ones_are_added():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 700 and raw.count(b"\r") == raw.count(b"\r\n") and raw.endswith(b"\r\n"), "every line CRLF, no bare CR"
    q, blocks = _queue_text(), _blocks()
    for label, h in {**_K.PRE_EXISTING, **KIND_PINS}.items():
        assert label in blocks, f"the stage {label} was removed"
        assert hashlib.sha256(("\n" + blocks[label]).encode()).hexdigest()[:16] == h, f"the stage {label} was edited"
        assert "mixo" not in blocks[label].lower() and "code-mix" not in blocks[label], f"{label} mentions the new stages"
    assert set(blocks) == set(_K.PRE_EXISTING) | set(KIND_PINS) | MIXO_LABELS and len(MIXO_LABELS) == 10, "the only new labels are the ten of this edit"
    for st in MSTAGES:
        assert q.count(f'if /i "%~1"=="{st}" goto :{st}\n') == 1, "a stage nobody dispatches never runs"
        assert q.count(f"\n:{st}\n") == 1
    target = {"train": "d_train", "eval": "d_eval"}
    for st in _K.OLD_DISPATCH + ("kind",):
        assert f'if /i "%~1"=="{st}" goto :{target.get(st, st)}\n' in q, st
    disp = re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto :(\w+)$', q)
    assert [(a, b) for a, b in disp if b not in set(blocks)] == [], "a dispatch line whose target label does not exist"
    usage = next(x for x in q.split("\n") if x.startswith("echo usage:"))
    assert usage.endswith("^|gatedscale^|traindscale SEED^|evaldscale SEED^|identitylora^|smokelora") and "mixo" not in usage, "the usage line is untouched (the earlier pins on it stand)"
    for need in ("gatemixo", "trainmixo SEED", "evalmixo SEED"):
        assert f"rem   v8f_mini_box2_queue.cmd {need}" in q, "the header documents the stage"


@needs_queue
def test_queue_mixo_runs_the_command_it_prints_static():
    """A dry run can only print: the real line is checked statically -- the trainer / eval appear ONCE per command, inside the one string, and the real run executes that string."""
    for label, var, pre in (("gatemixo", "GX", '--arm M2b --code-mix kind --out "%RUNSM%\\gate_M2b-mixo" '),
                            ("trainmixo", "TX", '--arm M2b --seed %~2 --grid pool4x10 --code-mix kind --out "%OUT%" ')):
        code = _code(_block(label))
        assert [x for x in code if "%TRAIN%" in x] == [x for x in code if x.startswith(f'set "{var}=')], label
        assert any(x.startswith(f'set "{var}="%PY%" "%TRAIN%" ' + pre) for x in code), label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code, label
        assert sum(x.startswith(f"%{var}% ") for x in code) == 1, label
        assert sum(x.count("--code-mix") for x in code if x.startswith(f'set "{var}=')) == 1, f"{label}: --code-mix appears once, in the one string"
        assert not any(("pool8x20" in x or "--hist" in x or "--p-gt" in x or "--w-" in x or "--hmax" in x or "--window-start" in x) for x in code), f"{label}: ONE lever"
    code = _code(_block("evalmixo"))
    assert [x for x in code if "%EVAL%" in x] == [x for x in code if re.match(r'set "E[0-6]=', x)], "the eval script appears only inside the seven strings"
    assert len([x for x in code if re.match(r'set "E[0-6]=', x)]) == 7
    assert "if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & echo %E4% & echo %E5% & echo %E6% & exit /b 0)" in code
    for k in range(7):
        assert sum(x.strip().startswith(f"%E{k}% >> ") for x in code) == 1, f"E{k} is executed exactly once"
    e = {k: next(x for x in code if x.startswith(f'set "E{k}=')) for k in range(7)}
    assert "--project --families" in e[0] and "--tac" not in e[0]
    assert "--tac absmask" in e[1] and "--tac oracle" in e[2] and "--tac-groups" not in e[2]
    assert "--tac oracle --tac-groups all-latcons --tac-alpha 1" in e[3]
    assert "--tac pred" in e[4] and "--tac oracle --tac-groups constraints --tac-alpha 1 " in e[5] + " " and "--tac oracle --tac-groups constraints --tac-alpha 0.50 " in e[6] + " "
    for k in range(7):
        assert '--ckpt "%OUT%\\ckpt_last.pt"' in e[k] and "%EC% %ER%" in e[k] and "--grid" not in e[k] and "--code-mix" not in e[k], \
            "the eval reads the checkpoint's own cfg; a code_mix flag on an eval would be a path the model never takes"
    # every failure is noticed on its own line (flat `if errorlevel 1 (... & exit /b 1)`, no parenthesised multi-line block: MEASURED to return rc 0 from `cmd /c`)
    for label in ("gatemixo", "trainmixo", "evalmixo"):
        body = _block(label)
        assert not any(x.rstrip().endswith("(") for x in body), f"{label}: a multi-line parenthesised block"
        assert sum(1 for x in body if x.startswith("if errorlevel 1 (echo [FAIL]") and x.rstrip().endswith("exit /b 1)")) >= (1 if label != "evalmixo" else 7)
    # the seven evals chain by goto through the flat labels evalmixo_1 .. evalmixo_7; the last one ends the stage
    q = _queue_text()
    for k in range(1, 8):
        assert q.count(f"\n:evalmixo_{k}\n") == 1 and f"goto :evalmixo_{k})" in q
    assert q.rstrip().endswith(":evalmixo_7\nexit /b 0")


@needs_queue
@win
def test_queue_mixo_stages_dry_one_lever(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    rc, ln = _dry("gatemixo", env)
    assert rc == 0 and len(ln) == 1 and "--selfcheck-only" in ln[0] and "gate_M2b-mixo" in ln[0], ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm"):t.index("--arm") + 4] == ["--arm", "M2b", "--code-mix", "kind"] and t.count("--code-mix") == 1, ln
    rc, ln = _dry("trainmixo 1", env)
    assert rc == 0 and len(ln) == 1, ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm") + 1] == "M2b" and t[t.index("--seed") + 1] == "1" and t[t.index("--code-mix") + 1] == "kind" and t.count("--code-mix") == 1, ln
    assert t[t.index("--grid") + 1] == "pool4x10" and t.count("--grid") == 1 and t[t.index("--out") + 1].endswith("M2b_s1_mixo"), ln
    assert "--exclude 540:600" in ln[0] and "--kin-mode yaw" in ln[0] and "--steps-mult 4" in ln[0] and "--batch 64" in ln[0], ln
    for no in ("--w-cons", "--w-tac", "--w-wm", "--hmax", "--steps ", "--p-gt", "--hist", "--window-start"):
        assert no not in ln[0], f"ONE lever: {no} is not on the line"
    rc, ev = _dry("evalmixo 0", env)
    assert rc == 0 and len(ev) == 7, ev
    for k, (suffix, frag) in enumerate((("", "--project --families"), ("-tacabs", "--tac absmask"), ("-tacorc", "--tac oracle"),
                                        ("-tacorc-all-latcons", "--tac oracle --tac-groups all-latcons --tac-alpha 1"), ("-tacpred", "--tac pred"),
                                        ("-tacorc-constraints", "--tac oracle --tac-groups constraints --tac-alpha 1"),
                                        ("-tacorc-constraints-a50", "--tac oracle --tac-groups constraints --tac-alpha 0.50"))):
        assert f"--tag M2b-mixo_s0{suffix} " in ev[k] + " " and frag in ev[k], (k, ev[k])
        assert f"M2b_s0_mixo{os.sep}ckpt_last.pt" in ev[k], "every eval reads THIS run's last checkpoint"
        assert f"{os.sep}M2b_s0{os.sep}" not in ev[k], "never B's checkpoint"
    # ONE lever, from the queue's own lines: X's gate / train line is B's line with --code-mix kind (and the output dir) changed and NOTHING else
    drop = {"--code-mix", "--out"}
    b_train = _dry("train M2b 1", env)[1][0]
    assert "--code-mix" not in b_train
    assert _without(b_train, {"--out"}) == _without(_dry("trainmixo 1", env)[1][0], drop), "X's train line = B's + --code-mix kind"
    b_gate = _dry("gatearm M2b", env)[1][0]
    assert _without(b_gate, {"--out"}) == _without(_dry("gatemixo", env)[1][0], drop), "X's gate line = B's gate line + --code-mix kind"


@needs_queue
@win
def test_queue_mixo_refuses_missing_arguments(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("trainmixo", "evalmixo"):
        rc, ln = _dry(args, env)
        assert rc == 2 and ln == [], (args, rc, ln)


@needs_queue
@win
def test_queue_mixo_leaves_every_other_stage_untouched(tmp_path):
    """The edit only ADDS stages: the neighbouring arms' lines read as they did, and none of them mentions the lever."""
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("train M2b 0", "trainpgt 0", "trainhist 0", "traindscale 0", "gatearm M2b", "gatepgt", "gategrid M2b", "kind", "evalhist 0"):
        rc, ln = _dry(args, env)
        assert rc == 0 and len(ln) >= 1 and all("--code-mix" not in x and "M2b-mixo" not in x and "_s0_mixo" not in x for x in ln), (args, ln)


@needs_queue
@win
def test_queue_mixo_real_lines_execute_exactly_the_printed_commands(tmp_path):
    repo, rec = _H._fake_tree(tmp_path)
    env = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    env.pop("V8F_DRYRUN", None)

    def real(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=180, encoding="utf-8")
        assert r.returncode == 0, (args, r.stdout, r.stderr)
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []

    def printed(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, V8F_DRYRUN="1"), capture_output=True, text=True, timeout=60, encoding="utf-8")
        return [_argv_of(x)[2:] for x in r.stdout.splitlines() if ".py" in x]       # drop python.exe and the script: sys.argv[1:]
    want_gate, want_train, want_eval = printed("gatemixo"), printed("trainmixo 1"), printed("evalmixo 1")        # BEFORE the runs: a finished stage prints [skip]
    assert len(want_gate) == 1 and len(want_train) == 1 and len(want_eval) == 7
    got = real("gatemixo")
    assert [g["argv"] for g in got] == want_gate and got[-1]["who"] == "train" and "--selfcheck-only" in got[-1]["argv"]
    n = len(got)
    got = real("trainmixo 1")[n:]
    assert [g["argv"] for g in got] == want_train and len(got) == 1, "the REAL trainmixo line hands the trainer the printed command"
    a = got[0]["argv"]
    assert a[a.index("--code-mix") + 1] == "kind" and a.count("--code-mix") == 1 and a[a.index("--arm") + 1] == "M2b" and a[a.index("--seed") + 1] == "1"
    assert (tmp_path / "runsm" / "M2b_s1_mixo" / "summary.json").exists() and not (tmp_path / "runsm" / "M2b_s1").exists(), "X has its own run dir; B's is never touched"
    n += 1
    got = real("evalmixo 1")[n:]
    assert [g["argv"] for g in got] == want_eval and len(got) == 7 and all(g["who"] == "eval" for g in got), "the REAL evalmixo lines hand the eval the printed commands"
    tags = [g["argv"][g["argv"].index("--tag") + 1] for g in got]
    assert tags == [f"M2b-mixo_s1{s}" for s in ("", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50")]
    n += 7                                              # a finished stage is skipped, not re-run
    assert len(real("evalmixo 1")) == n and len(real("trainmixo 1")) == n


@needs_queue
@win
def test_queue_mixo_real_stages_notice_failures_and_skip_exactly_what_is_finished(tmp_path):
    repo, rec = _H._fake_tree(tmp_path)
    runsm = tmp_path / "runsm"
    ev = runsm / "eval"
    base = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(runsm), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    base.pop("V8F_DRYRUN", None)

    def stage(args, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(base, **extra), capture_output=True, text=True, timeout=180, encoding="utf-8")
        return r.returncode, r.stdout

    def evals():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines() if '"eval"' in x] if rec.exists() else []
    rc, out = stage("gatemixo", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] gate M2b-mixo" in out
    rc, out = stage("trainmixo 1", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] train M2b_s1_mixo - see" in out
    rc, out = stage("trainmixo 1", FAKE_TRAIN="nosummary")                     # a trainer that exits 0 WITHOUT a summary (a crash that swallowed its code)
    assert rc == 1 and "exited 0 without summary.json" in out
    rc, out = stage("evalmixo 1")                                              # evaluation refuses a run that has not finished training
    assert rc == 2 and "has not finished training" in out and not evals()
    assert stage("trainmixo 1")[0] == 0
    rc, out = stage("trainmixo 1")
    assert rc == 0 and "[skip] M2b_s1_mixo" in out, "a finished run is not re-trained"
    sfx = ["", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50"]
    tags = [f"M2b-mixo_s1{s}" for s in sfx]
    for tag in tags:
        rc, out = stage("evalmixo 1", FAKE_FAIL_TAG=tag)
        assert rc == 1 and f"[FAIL] eval {tag} - see" in out, (tag, rc, out)
        for f in list(ev.glob("eval_M2b-mixo_s1*.npz")):
            f.unlink()
    ev.mkdir(exist_ok=True)
    for n, done in enumerate(([0, 2], [1], [0], [6], [3, 4], [0, 1, 2, 3, 4, 5], [1, 3, 5], [5, 6])):
        for f in list(ev.glob("eval_M2b-mixo_s1*")):
            f.rename(f.with_name(f"old{n}_" + f.name))
        rec.write_text("", encoding="utf-8")
        for k in done:
            (ev / f"eval_{tags[k]}.npz").write_bytes(b"x")
        assert stage("evalmixo 1")[0] == 0, done
        ran = [g["argv"][g["argv"].index("--tag") + 1] for g in evals()]
        assert ran == [tg for k, tg in enumerate(tags) if k not in done], (done, ran)


# =============================================================================================================================================
# THE CHAIN (code/box2_ops/chain_v8f_mixo.sh): needs box 2 and ssh, so it cannot RUN here -- its order, its gate assertions (RUN against synthetic selfchecks), its token record,
# its --dry-run (== the queue's own dry-run lines) and every load-bearing line are pinned
# =============================================================================================================================================
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


@needs_chain
def test_chain_mixo_static_ordering_the_wait_the_atomic_marker_and_no_base_decision():
    c = CHAIN.read_text(encoding="utf-8")
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    code = "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))
    assert "decide-base" not in code and "--base" not in code and "BASE" not in code, "B = M2b is fixed in the PREREG: nothing is decided at run time"
    # the order: predecessor -> marker -> refresh -> ckpt sha -> check-inputs on the REAL B side -> guard -> gate -> seed 0 -> read s0 -> (CONTINUE) seed 1 -> final read
    order = ["waiting for the predecessor chain", "( set -C;", "BOOT0=$(boot)", "fetch -q origin", 'ckptsha_b "before the gate"', 'v8f_mixo_read.py" --check-inputs', "v8f_guard.sh > ",
             "queue gatemixo 120 gatemixo", "runone 0\n", "mixoread s0 mixo_read_s0.json", "runone 1\n", "mixoread final mixo_read.json"]
    idx = [c.index(x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    assert 'queue "t_$g" $TRAIN_POLLS trainmixo $s' in c and 'queue "e_$g" $EVAL_POLLS evalmixo $s' in c
    assert '[ "$(jget mixo_read_s0.json DECISION)" = CONTINUE ] || finish' in c, "a STOP at seed 0 is a committed outcome (DONE), not a crash"
    assert c.index('v8f_mixo_read.py" --check-inputs') < c.index("launch telemetry_long") < c.index("queue gatemixo 120"), "B's side is refused while box 2 is still idle and untouched"
    # the predecessor is PREREG_V8F_KIND's chain by default (override-able); markers are the shared ones, in THIS chain's own log; the busy marker is taken atomically (noclobber)
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_kind.out}" in c and "AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}" in c and "AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}" in c
    assert "chain_v8f_mixo.out" in c and 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh' in c
    assert "busy marker taken while arming" in c and "BOX2_BUSY.txt" in c
    # the training wait covers the longest run the guard lets live
    polls = int(re.search(r"^TRAIN_POLLS=(\d+)", c, re.M).group(1))
    maxh = float(re.search(r"V8F_MAX_RUN_H:-(\d+)", GUARD.read_text(encoding="utf-8")).group(1))
    assert polls == 1320 and polls * 30 / 3600 > maxh, f"a run the guard lets live ({maxh} h) outlasts the wait ({polls * 30 / 3600} h)"
    # the B side: the banked M2b files incl. the dose chain's; the P10 guard file and the check-inputs come BEFORE any training
    for x in ("$L/v8f_mini/runs/M2b_s$s/$f", "$L/v8f_cz/eval/$f", "$L/v8f_tacgrp2/eval/$f", "floors_perwindow.npz", "eval_M2b_s$s-tacpred.npz", "eval_M2b_s$s-tacorc-constraints.npz",
              "eval_M2b_s$s-tacorc-constraints-a50.npz", "eval_M2b_s$s-tacorc-constraints-a50.json"):
        assert x in c, x
    assert c.index("floors_perwindow.npz") < c.index('v8f_mixo_read.py" --check-inputs') < c.index("queue gatemixo 120")
    # what an X run pulls: its selfcheck/summary/argv/log/ckpt (the reader reads the checkpoint's cfg) and the seven evals
    assert "for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull" in c
    assert 'for f in "" $X_SUFFIXES; do for x in json npz log; do pull "eval/eval_$tg$f.$x"' in c
    assert re.search(r'^X_SUFFIXES="-tacabs -tacorc -tacorc-all-latcons -tacpred -tacorc-constraints -tacorc-constraints-a50"$', c, re.M)
    arch = next(x for x in c.splitlines() if "archive --format=tar" in x)
    assert all(x in arch for x in ("stack", "taniteval", '"$HJ"', '"$PK"', '"$B1"')), "the reader and the lead block ride in the archive"
    for f in ("v8f_mixo_read.py", "v8f_grid_read.py", "v8f_pgt_read.py", "v8f_dk_read.py", "v8f_wc_read.py", "v8f_tac_groups_read.py", "v8f_mini_full_report.py"):
        assert f in c, f"the chain insists the tip has {f}"
    g = GUARD.read_text(encoding="utf-8")
    assert "ZZV8FMINI" in g and "(DONE|STOP)" in g


@needs_chain
def test_chain_mixo_checkpoint_sha256_is_asserted_before_the_gate_and_after_the_evals():
    c = CHAIN.read_text(encoding="utf-8")
    assert 'ckptsha_b "before the gate"' in c and "queue \"e_$g\" $EVAL_POLLS evalmixo $s\n  ckptsha_x $s after_evals" in c and "queue \"t_$g\" $TRAIN_POLLS trainmixo $s\n  echo \"\" > $CUR\n  ckptsha_x $s trained" in c
    assert 'stop "$1: box 2\'s M2b_s$s checkpoint sha256 $g != the banked run\'s ${CKPT_SHA[$s]}"' in c
    assert 'stop "M2b_s${1}_mixo\'s checkpoint changed during the evals' in c
    assert "ckset \"M2b_s$s\" \"$g\"" in c and 'ckset "M2b-mixo_s${1}_$2" "$g"' in c, "the measured values go into the json the reader takes as --ckpt-sha"


@needs_chain
def test_chain_mixo_syntax_and_dry_run_prints_the_queue_lines(tmp_path):
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's --dry-run")
    r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, V8F_QUEUE=str(QUEUE), TMPDIR=tmp_path.as_posix())
    env.pop("V8F_DRYRUN", None)
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    stages = ["gatemixo", "trainmixo 0", "trainmixo 1", "evalmixo 0", "evalmixo 1"]
    assert [x for x in out if x.startswith("== ") and x[3:] in stages] == [f"== {s}" for s in stages]
    want = []                                               # the chain's printed lines ARE the queue's own dry-run lines at box 2's default paths
    for st in stages:
        want += _dry(st, clean_paths=True)[1]
    assert [x for x in out if ".py" in x and "v8f_mixo_read" not in x] == want and len(want) == 1 + 2 + 14
    t0 = next(x for x in out if "--seed 0" in x and "v8f_mini_train.py" in x)
    assert "--arm M2b --seed 0 --grid pool4x10 --code-mix kind --out" in t0 and "D:\\tanitad\\repo" in t0 and "M2b_s0_mixo" in t0
    g0 = next(x for x in out if "--selfcheck-only" in x)
    assert "--arm M2b --code-mix kind --out" in g0 and "gate_M2b-mixo" in g0
    assert "wait for <logs>/chain_v8f_kind.out: ^ZZV8FMINI-DONE" in r.stdout
    for s, sha in ((0, R.BANKED_CKPT_SHA["M2b_s0"]), (1, R.BANKED_CKPT_SHA["M2b_s1"])) if R is not None else ():
        assert f"M2b_s{s} D:/tanitad/runs/v8f_mini_full/M2b_s{s}/ckpt_last.pt {sha}" in r.stdout
    reads = [x for x in out if x.startswith("v8f_mixo_read.py")]
    assert len(reads) == 3 and "--check-inputs" in reads[0] and "--stage s0" in reads[1] and "--stage final" in reads[2] and all("--ckpt-sha" in x for x in reads)


def _gate_assertions():
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"<<'EOF' \|\| stop \"gate X assertions\"\n(.*?)\nEOF\n", c, re.S)
    assert m, "the chain's gate assertions are a python heredoc whose failure stops the chain"
    return m.group(1)


GLINE_OK = ('"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_mini_train.py" --arm M2b --code-mix kind '
            '--out "D:\\tanitad\\runs\\v8f_mini_full\\gate_M2b-mixo" --kin-mode yaw --selfcheck-only')


def _gate_run(tmp_path, mod=None, line=GLINE_OK, steps_arg="19164", good_mod=None, windows_arg="306566"):
    body = _gate_assertions()
    real, _ = _H._b_real(0)
    bl = real["summary.json"]["levers"]
    good = {"declared_vs_built": "PASS", "live": {"pass": True, "control_prior_grad_sum": 0.0}, "steps_total": 19164,
            "code_mix_probe": {"code_mix": "kind", "p_gt": 0.5, "n_labelled": 40, "n_draws_samples": 160, "n_partial": 150, "n_all_gt": 5, "n_all_own": 5, "pass": True},
            "info": {"n_clips": 4512, "n_excluded": 60, "n_excluded_declared": 60, "excluded_in_train": 0, "excluded_sha256": HOLD,
                     "train_sha256": real["summary.json"]["train_sha256"], "kin_mode": "yaw", "lru": 174, "n_windows_all_clips": 306566,
                     "window_start": 3, "window_t_min": 3, "ds_window_start": 3},
            "levers": {"wm": "b", "navonly": False, "consistency": True, "cons_absent": "mask", "code_mix": "kind", "p_gt_start": 1.0, "p_gt_end": 0.2, "hist": 4, "hmax": 16, "window_start": 3,
                       "w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5},
                       "params": dict(bl["params"]), "n_params_trainable": bl["n_params_trainable"]}}
    if good_mod:
        good_mod(good)
    sc = copy.deepcopy(good)
    if mod:
        mod(sc)
    (tmp_path / "bsum.json").write_text(json.dumps(real["summary.json"]), encoding="utf-8")
    f = tmp_path / "sc.json"
    f.write_text(json.dumps(sc), encoding="utf-8")
    r = subprocess.run([sys.executable, "-", str(f), HOLD, steps_arg, str(tmp_path / "bsum.json"), line, windows_arg], input=body, capture_output=True, text=True, encoding="utf-8", timeout=60)
    return r.returncode, r.stdout + r.stderr


@needs_chain
@needs_pkg
def test_chain_mixo_gate_assertions_pass_the_x_gate_and_refuse_every_way_it_can_be_wrong(tmp_path):
    """The chain's gate block is RUN against a synthetic selfcheck.json built from B's REAL banked record: a good X gate passes; each single-field corruption exits 1 and names the broken
    field (the PASS token, the lever, the schedule, the clip list, the holdout, the weights, the window set, the step count, the history and the PARAMETER COUNT cannot drift under a green gate)."""
    rc, out = _gate_run(tmp_path)
    pb = _H._b_real(0)[0]["summary.json"]["levers"]
    assert rc == 0 and "PASS" in out and "built code_mix kind" in out and "built p_gt 1.0 -> 0.2" in out and f"params {pb['n_params_trainable']}" in out and "windows 306566" in out, out
    L = lambda k, v: (lambda s: s["levers"].__setitem__(k, v))                   # noqa: E731
    W = lambda k, v: (lambda s: s["levers"]["w"].__setitem__(k, v))              # noqa: E731
    I = lambda k, v: (lambda s: s["info"].__setitem__(k, v))                     # noqa: E731
    P = lambda k, v: (lambda s: s["levers"]["params"].__setitem__(k, v))         # noqa: E731
    PR = lambda k, v: (lambda s: s["code_mix_probe"].__setitem__(k, v))          # noqa: E731
    cases = (("declared_vs_built", lambda s: s.__setitem__("declared_vs_built", "FAIL"), "declared_vs_built"),
             ("live", lambda s: s["live"].__setitem__("pass", False), "live"),
             ("control grad", lambda s: s["live"].__setitem__("control_prior_grad_sum", 1.0), "live"),
             ("no probe", lambda s: s.pop("code_mix_probe"), "code_mix_probe None"),
             ("probe not pass", PR("pass", False), "code_mix_probe"),
             ("probe for whole", PR("code_mix", "whole"), "was run for 'whole'"),
             ("probe saw no partial", PR("n_partial", 0), "saw no PARTLY-GT code"),
             ("n_clips", I("n_clips", 4000), "clips / holdout counts"), ("n_excluded", I("n_excluded", 59), "clips / holdout counts"),
             ("n_excluded_declared", I("n_excluded_declared", 59), "clips / holdout counts"), ("excluded_in_train", I("excluded_in_train", 1), "clips / holdout counts"),
             ("holdout", I("excluded_sha256", "0" * 64), "not the RA-1 holdout"), ("train list", I("train_sha256", "d" * 64), "TRAIN list"),
             ("kin_mode", I("kin_mode", "steer"), "kin_mode"), ("lru unrecorded", lambda s: s["info"].pop("lru"), "info.lru"),
             ("wm", L("wm", "a"), "levers"), ("navonly", L("navonly", True), "levers"), ("consistency", L("consistency", False), "levers"), ("cons_absent", L("cons_absent", "zero"), "levers"),
             ("built code_mix whole", L("code_mix", "whole"), "built code_mix 'whole' != 'kind'"), ("code_mix unrecorded", lambda s: s["levers"].pop("code_mix"), "built code_mix None != 'kind'"),
             ("p_gt start", L("p_gt_start", 0.5), "built p_gt"), ("p_gt end", L("p_gt_end", 0.0), "built p_gt"),
             ("p_gt missing", lambda s: (s["levers"].pop("p_gt_start"), s["levers"].pop("p_gt_end")) and None, "built p_gt None -> None"),
             ("w_pen", W("w_pen", 0.6), "w_pen"), ("w_pen_self", W("w_pen_self", 0.3), "w_pen_self"), ("w_wm", W("w_wm", 2.0), "w_wm"),
             ("w_tac", W("w_tac", 2.0), "w_tac / w_cons"), ("w_cons", W("w_cons", 1.0), "w_tac / w_cons"),
             ("steps", lambda s: s.__setitem__("steps_total", 9000), "steps 9000"),
             ("window set is not B's", I("n_windows_all_clips", 279494), "the window set is not B's"),
             ("window set is not the PREREG's number", I("n_windows_all_clips", 306565), "the PREREG's 306566"),
             ("built hist 10", L("hist", 10), "built hist 10 != 4"), ("built hist unrecorded", lambda s: s["levers"].pop("hist"), "built hist None != 4"),
             ("built hmax 10", L("hmax", 10), "built hmax 10 != 16"), ("built hmax unrecorded", lambda s: s["levers"].pop("hmax"), "built hmax None != 16"),
             ("built window_start 9", L("window_start", 9), "built window_start 9 != 3"),
             ("built window_start unrecorded", lambda s: s["levers"].pop("window_start"), "built window_start None != 3"),
             ("the window list started at t = 9", I("window_t_min", 9), "window start (asked / first t"), ("the dataset clamps from 9", I("ds_window_start", 9), "window start (asked / first t"),
             ("one parameter too many", L("n_params_trainable", pb["n_params_trainable"] + 1), "n_params_trainable"),
             ("a parameter fewer", L("n_params_trainable", pb["n_params_trainable"] - 1), "n_params_trainable"),
             ("a group grew", P("T", pb["params"]["T"] + 5), "parameter group T"),
             ("a group moved to another", lambda s: (P("adapter", pb["params"]["adapter"] + 256)(s), P("O", pb["params"]["O"] - 256)(s)) and None, "parameter group"),
             ("params unrecorded", lambda s: s["levers"].pop("params"), ""))
    for name, mod, msg in cases:
        rc, out = _gate_run(tmp_path, mod)
        assert rc != 0 and msg in out, (name, rc, out)
    # the lever, from the queue's own printed line
    for name, line, msg in (("no --code-mix on the line", GLINE_OK.replace(" --code-mix kind", ""), "--code-mix kind exactly once"),
                            ("--code-mix whole", GLINE_OK.replace("--code-mix kind", "--code-mix whole"), "--code-mix kind exactly once"),
                            ("--code-mix twice", GLINE_OK.replace("--code-mix kind", "--code-mix kind --code-mix kind"), "--code-mix kind exactly once"),
                            ("another arm", GLINE_OK.replace("--arm M2b", "--arm M2"), "is not --arm M2b"),
                            ("--hist set", GLINE_OK + " --hist 10", "sets --hist"), ("--hmax set", GLINE_OK + " --hmax 16", "sets --hmax"),
                            ("--p-gt-start set", GLINE_OK + " --p-gt-start 0.5", "sets --p-gt-start"), ("--window-start set", GLINE_OK + " --window-start 3", "sets --window-start"),
                            ("--w-cons set", GLINE_OK + " --w-cons 1.0", "sets --w-cons"), ("--grid set", GLINE_OK + " --grid pool8x20", "sets --grid")):
        rc, out = _gate_run(tmp_path, line=line)
        assert rc == 1 and msg in out, (name, rc, out)
    # a budget different from B's, however it came about, stops the chain at the gate
    rc, out = _gate_run(tmp_path, steps_arg="17472")
    assert rc == 1 and "steps 19164 != B's 17472" in out, out


@needs_chain
@needs_pkg
def test_chain_mixo_writes_a_gate_token_bound_to_the_commit_the_queue_the_selfcheck_and_the_data_manifests(tmp_path):
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"<<'EOF' \|\| stop \"write the gate token\"\n(.*?)\nEOF\n", c, re.S)
    assert m, "the token record is written by the chain, never by hand"
    sc = {"info": {"train_sha256": "t" * 64, "excluded_sha256": HOLD}, "argv_sha": "abc123"}
    scp, out = tmp_path / "sc.json", tmp_path / "gate_token.json"
    scp.write_text(json.dumps(sc), encoding="utf-8")
    keys = json.dumps({"stack/scripts/v8f_mini_train.py": "k1", "stack/tanitad/models/v8f_mini.py": "k2"})
    r = subprocess.run([sys.executable, "-", str(scp), str(out), "c0ffee", "q" * 64, keys], input=m.group(1), capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    tok = json.loads(out.read_text(encoding="utf-8"))
    assert tok["prereg"] == "PREREG_V8F_MIXO" and tok["gate"] == "PASS" and tok["commit"] == "c0ffee" and tok["queue_sha256"] == "q" * 64 and tok["argv_sha"] == "abc123"
    assert tok["selfcheck_sha256"] == hashlib.sha256(scp.read_bytes()).hexdigest() and tok["key_files_sha256"] == json.loads(keys)
    assert tok["train_sha256"] == "t" * 64 and tok["excluded_sha256"] == HOLD and set(tok["v9_md5"]) == {"v9_labels_train.npz", "v9_labels_eval139.npz"}
    # ordering: the token is written only after the gate assertions passed, and before any training
    assert c.index('"gate X assertions"') < c.index('"write the gate token"') < c.index("runone 0\n")


@needs_chain
def test_chain_mixo_finish_releases_the_box_before_it_announces_done_and_stop_releases_it_too():
    """The next chain waits for DONE and then refuses on a leftover BOX2_BUSY.txt: finish() must remove the marker and stop the telemetry BEFORE it prints ZZV8FMINI-DONE."""
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"\nfinish\(\) \{.*?\n\}\n", c, re.S)
    assert m, "no finish() function in the chain"
    body = m.group(0)
    for need in ("telemetry.stop", "rm -f $B/BOX2_BUSY.txt", 'echo "" > $CUR', "ZZV8FMINI-DONE", "exit 0"):
        assert need in body, f"finish() lost {need!r}"
    assert body.index("rm -f $B/BOX2_BUSY.txt") < body.index("ZZV8FMINI-DONE") < body.index("exit 0")
    assert re.search(r'^stop\(\) \{ echo "ZZV8FMINI-STOP \$\*"; echo "" > \$CUR; rm -f \$B/BOX2_BUSY\.txt; exit 1; \}$', c, re.M), "a stop after the marker was taken releases it"


@needs_chain
def test_chain_mixo_load_bearing_lines_are_present():
    """Every line whose loss would let a wrong run through is pinned. Substrings, not whole lines: line ends are not load-bearing."""
    c = CHAIN.read_text(encoding="utf-8")
    lines = (
        r"""for st in gatemixo trainmixo evalmixo; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done""",
        r"""grep -q '"code_mix": cfg.code_mix' || stop "the tip's built_levers does not record code_mix""",
        r"""grep -q '^CODE_MIX_UNITS = {' || stop "the tip's v8f_mini.py has no CODE_MIX_UNITS partition""",
        r"""grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start""",
        r"""grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end""",
        r"""grep -q -- '"--code-mix"' || stop "the tip's trainer has no --code-mix flag""",
        r"""grep -q '"code_mix": "lever:code_mix"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --code-mix""",
        r"""grep -q 'code_mix_probe(model, probe, cfg)' || stop "the tip's trainer does not run the code_mix probe in its in-process gate""",
        r"""git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block""",
        r"""|| stop "the reader cannot use the M2b side (check-inputs)""",
        r"""|| stop "mixo read $1""",
        r"""$PY "$RD/v8f_mixo_read.py" --stage $1 $(reader_args""",
        r"""GLINE=$(dry_stage "$QC" gatemixo | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gatemixo line""",
        r"""$HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS <<'EOF' || stop "gate X assertions""",
        r"""pull "gate_M2b-mixo/selfcheck.json" "$E/gate_M2b-mixo""",
        r"""echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)""",
        r"""[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present""",
        r"""until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do""",
        r"""grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped""",
        r"""cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo""",
        r"""cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f (the banked dose chain of B)""",
        r"""cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz""",
        r"""[ "$(jget mixo_read_s0.json DECISION)" = CONTINUE ] || finish "$M1 -- seed 1 not trained (committed)""",
        r"""finish "$M1 | $(jget mixo_read.json reading) | $(jget mixo_read.json next)""",
        r"""g="M2b_s${1}_mixo" tg="M2b-mixo_s${1}""",
        r"""grep ZZGUARD-STOP $L/v8fmixo_guard.out 2>/dev/null && stop "guard stop""",
        r"""V8F_CHAIN_OUT=$L/chain_v8f_mixo.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh""",
        r"""[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry""",
        r"""M2B_STEPS=19164""",
        r"""M2B_WINDOWS=306566""",
        r"""TRAIN_POLLS=1320; EVAL_POLLS=600""",
        r"""rm -f $CK""",
        r"""|| stop "write the gate token""",
    )
    for x in lines:
        assert x in c, f"the chain lost: {x}"


# ---- bars that only a hard case can tell from a softer one -------------------------------------------------------------------------------------
def _split_effect(f, s, odd_delta):
    """Make X's seed-s effect HETEROGENEOUS across episodes: the odd episodes' plans are shifted laterally (beyond 2 s) until their mean ADE6 is B's + `odd_delta`; ade6 follows the plans
    (as the eval would write it) and the three dose files move by the same amount per window."""
    w, ev = f["w"], f["eval"]
    odd = np.asarray(w.eid) % 2 == 1

    def ade(pr):
        return np.linalg.norm(pr.astype(np.float32).astype(np.float64) - w.gt, axis=-1).mean(1)
    b = np.asarray(np.load(ev / f"eval_M2b_s{s}.npz")["ade6"], np.float64)
    p = ev / f"eval_{XT}_s{s}.npz"
    z = dict(np.load(p))
    old = np.asarray(z["ade6"], np.float64)
    target = b[odd].mean() + odd_delta

    def shifted(e):
        pr = z["pred"].astype(np.float64).copy()
        pr[odd, 10:, 1] += e
        return pr
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ade(shifted(mid))[odd].mean() < target else (lo, mid)
    e = 0.5 * (lo + hi)
    for suf in ("", "-tacabs"):
        pp = ev / f"eval_{XT}_s{s}{suf}.npz"
        zz = dict(np.load(pp))
        pr = zz["pred"].astype(np.float64).copy()
        pr[odd, 10:, 1] += e
        zz["pred"] = pr.astype(np.float32)
        zz["ade6"] = ade(zz["pred"])
        np.savez(pp, **zz)
    new = np.asarray(np.load(p)["ade6"], np.float64)
    for suf in ("-tacpred", "-tacorc-constraints", "-tacorc-constraints-a50"):
        _npz_edit(ev / f"eval_{XT}_s{s}{suf}.npz", lambda zz: zz.update(ade6=np.asarray(zz["ade6"], np.float64) + (new - old)))


@needs_pkg
def test_reader_the_primary_needs_separation_a_point_below_2F_that_the_episodes_do_not_support_is_a_miss(tmp_path, no_known):
    f = _mworld(tmp_path / "ns", d_ade=-2.0)
    _split_effect(f, 0, 0.8)
    rc, r = _go(f, "s0")
    p = r["rows"]["s0"]["primary"]
    assert rc == 0 and p["point"] < -0.2415 and p["separated"] is False and r["rows"]["s0"]["judge"]["passed"] is False, (p, r["reading"])
    assert r["DECISION"] == "STOP" and r["VERDICT"] in ("FAIL-O-LEARNED", "FAIL-O-DID-NOT-LEARN")


@needs_pkg
def test_reader_reads_the_rise_against_b_s_committed_r_not_the_one_it_measured(tmp_path, no_known):
    """B's banked evals reproduce 0.36 only to within the rounding of the PREREG's table (a measured 0.363 is admissible): the bar is +0.10 over the COMMITTED 0.36."""
    rc, r = _go(_mworld(tmp_path / "c", d_ade=-0.10, r_b=(0.363, 0.21), r_x=(0.46, 0.45)), "s0")
    dm = r["rows"]["s0"]["dm"]
    assert rc == 0 and dm["r_B_measured"] == pytest.approx(0.363) and dm["r_B_committed"] == 0.36
    assert dm["rise_over_committed"] == pytest.approx(0.10) and dm["rise_over_measured"] == pytest.approx(0.097) and dm["learned"] is True
    assert r["VERDICT"] == "FAIL-O-LEARNED"


@needs_pkg
def test_the_dose_identity_cells_are_enforced_on_their_own(tmp_path, no_known, capsys, monkeypatch):
    """PGT's identity control trips first on a globally biased estimator; this one is biased ONLY on B's alpha-0.5 oracle against itself, so only this reader's own dose cells can see it."""
    f = _mworld(tmp_path / "w")
    import taniteval.ci as ci
    orig = ci.paired_episode_cluster_bootstrap
    target = np.load(f["eval"] / "eval_M2b_s0-tacorc-constraints-a50.npz")["ade6"].astype(np.float64)

    def biased(a, b, *r, **k):
        o = orig(a, b, *r, **k)
        if np.array_equal(np.asarray(a, np.float64), target) and np.array_equal(np.asarray(b, np.float64), target):
            o = dict(o, delta=o["delta"] + 1e-9)
        return o
    monkeypatch.setattr(ci, "paired_episode_cluster_bootstrap", biased)
    capsys.readouterr()
    assert R.main(["--stage", "final", *f["args"]]) == 2
    err = capsys.readouterr().err
    assert "IDENTITY CONTROL FAILED" in err and "dose_half" in err, err[-300:]
