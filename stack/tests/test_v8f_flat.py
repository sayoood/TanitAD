"""PREREG_V8F_FLAT: does v8F-mini's hierarchy beat its own capacity-matched FLAT control?  FLAT = M2b + `--o-interface cut` (ONE lever: O reads a CONSTANT in place of both layers above it).

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): a lever whose library function is correct and whose CALLER never reaches it. So every identity
below runs from argv, hop by hop, and has a deliberate-regression arm that turns it RED (here, as a mutated copy of the model module; and in `raw/2026-10-25-flat/mutate_flat.py`, as
mutated repo files, output beside it):

  argv --o-interface -> make_config (cfg.o_interface) -> declared_vs_built (G-DVB: FLAG_CHECKS entry + the BUILT lever) -> live_check (G-LIVE: e_code.weight reads EXACTLY 0.0 under cut)
  -> o_interface_probe (what `run_o` is HANDED under a perturbation of everything above it) -> V8FMini._forward -> `if cfg.o_interface == "cut":` -> run_o(..., ck_for_o, ck_ok, code)

  (A) the lever         `O_INTERFACE_MODES`, the default `full` everywhere, a bad value refused, built_levers record it, the lever adds no parameter.
  (B) `full` IS TODAY   the default reproduces the saved pre-change module (`raw/2026-10-25-flat/pre_flat_v8f_mini.py.ref`) bit for bit: the built model, `_forward`, the RNG stream, the loss
                        and a 200-step CPU loss trace (PREREG proof 2). Arm: a mutant that applies the cut under `full` breaks it.
  (C) structural zero   PREREG proof 3: under `cut`, traj60 is bit-for-bit invariant to S's checkpoints and T's code (donor, GT, random code through forward_diag; shifted checkpoints; the
                        teacher forcing p_gt 0 vs 1), max |d traj60| = 0.0; under `full` the same perturbation moves it (the control that proves the probe can see a change).
  (D) gradient          PREREG proof 4: under `cut`, `e_code.weight.grad` is exactly 0.0 (the bias still trains), under `full` it is not; under BOTH, S's and T's heads get a non-zero gradient;
                        and the O-only loss reaches S's and T's parameters under `full` and exactly 0.0 under `cut`.
  (E) the mutants       PREREG proof 5: delete the cut line, or only its code half, or only its checkpoint half, or apply it BEFORE the overrides: proofs C and D go RED on each.
  (F) G-DVB             PREREG proof 1: `--o-interface` has its entry; a flag without one is refused; declared cut / built full (and the reverse) is refused.
  (G) reachability      a real `TR.main` run on the synthetic world: the argv flag reaches the config, the checkpoint, the records, G-LIVE's dead set, the probe and what `run_o` is handed on
                        EVERY training step; the default run hands O the real code. The trained cut checkpoint has an all-zero Adam state for e_code.weight and for nothing else.
  (H) G-EVAL            `v8f_mini_eval.py` builds the config FROM the checkpoint: a cut checkpoint is evaluated as cut, and its GT-code oracle eval reads EXACTLY its own predictions
                        (the PREREG's identity read) while the oracle moves a full checkpoint.
  (I) the reader        `code/v8f_flat_read.py`: the committed outcome table, applied verbatim, and every refusal (rc 2).
  (J) the queue + chain the stages print the command they run, one lever; every existing label is byte-identical; the chain's wait predicate (DONE proceeds; a verdict- or run-level STOP
                        proceeds; a guard, thermal, reboot or unreachable STOP does not), gate assertions, ordering and dry run.
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
import types
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
import v8f_mini_eval as EV  # noqa: E402

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
RAWF = PKG / "raw" / "2026-10-25-flat"
REF_P = RAWF / "pre_flat_v8f_mini.py.ref"
CODE = PKG / "code"
READ_P = CODE / "v8f_flat_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_flat.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_FLAT.md"
MODEL_P = REPO / "stack" / "tanitad" / "models" / "v8f_mini.py"
HERE_T = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_ref(name, path):
    """A module loaded under a private name from a file whose suffix is not .py (the saved reference, a mutant). Registered in sys.modules BEFORE it executes (a dataclass with
    `from __future__ import annotations` looks its module up)."""
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


REF_M = _load_ref("v8f_mini_pre_flat", REF_P) if REF_P.exists() else None
needs_ref = pytest.mark.skipif(REF_M is None, reason="raw/2026-10-25-flat (the saved pre-change module) is not in this checkout")
_P = _load("test_v8f_pgt_for_flat", HERE_T / "test_v8f_pgt.py")                  # the batch, the spy and the synthetic world the other v8F-mini tests use
world, tiny_model, make_batch = _P.world, _P.tiny_model, _P.make_batch
_mixed_batch = _P._mixed_batch
_T = _P._T                                                                       # test_v8f_mini
COMMON = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5",
          "--arm", "M2b"]


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
    """The residual head is zero-init: at build O's output is P10 whatever it reads, so NO wire could be seen. Give it weights (seeded) so a probe can see a change."""
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
    """pytest keeps every test's tmp_path until the session ends; on a disk-limited box that piles up (the file's peak basetemp was 131 MB against the coordinator's 100 MB cap).
    Remove a test's OWN tmp_path when it finishes; the module-scoped `runs` world lives under tmp_path_factory, which this never touches."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


# =============================================================================================================================================
# (A) THE LEVER
# =============================================================================================================================================
def test_the_flag_defaults_to_full_everywhere_and_a_bad_value_is_refused():
    assert M.V8FMiniConfig().o_interface == "full" and M.O_INTERFACE_MODES == ("full", "cut")
    a = _args()
    assert a.o_interface == "full" and TR.make_config(a).o_interface == "full"
    c = _args(["--o-interface", "cut"])
    assert c.o_interface == "cut" and TR.make_config(c).o_interface == "cut"
    with pytest.raises(SystemExit):
        _args(["--o-interface", "flat"])
    with pytest.raises(ValueError, match="o_interface must be one of"):
        M.V8FMiniConfig(o_interface="flat")
    d = {key: v for key, v in TR.make_config(_args()).__dict__.items() if key != "o_interface"}      # nothing else moves: the two configs differ in o_interface and in nothing else
    assert d == {key: v for key, v in TR.make_config(c).__dict__.items() if key != "o_interface"}


def test_built_levers_record_the_built_interface_and_the_lever_adds_no_parameter(world):
    J = _J(world)
    mf, mc = _mk(M, J), _mk(M, J, o_interface="cut")
    assert mf.built_levers()["o_interface"] == "full" and mc.built_levers()["o_interface"] == "cut"
    sf, sc = mf.state_dict(), mc.state_dict()
    assert sf.keys() == sc.keys() and all(torch.equal(sf[x], sc[x]) for x in sf), "same seed, same weights: the lever builds no parameter, S, T and O keep every one"
    lf, lc = mf.built_levers(), mc.built_levers()
    assert lf["params"] == lc["params"] and lf["n_params_trainable"] == lc["n_params_trainable"] and lf["params"]["S"] > 0 and lf["params"]["T"] > 0
    assert {key: v for key, v in lf.items() if key != "o_interface"} == {key: v for key, v in lc.items() if key != "o_interface"}


# =============================================================================================================================================
# (B) `full` IS TODAY, BIT FOR BIT
# =============================================================================================================================================
def _old_pair(world, n=12, seed=2, mod=None):
    """The same weights and the same batch in `mod` (default: the CURRENT module, o_interface full) and in the saved pre-change module."""
    mod = mod or M
    J = _J(world)
    cfg = mod.V8FMiniConfig(**{**_T.TINY, "wm": "off"})
    kw = {k: v for k, v in cfg.__dict__.items() if k != "o_interface"}
    mu, sd = M.cons_stats(J)
    torch.manual_seed(3)
    new = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
    torch.manual_seed(3)
    old = REF_M.V8FMini(REF_M.V8FMiniConfig(**kw), REF_M.P10Prior.synthetic(0), mu, sd)
    b, _, _ = make_batch(world, n=n, seed=seed, hmax=cfg.hmax)
    B = b["feats"].shape[0]
    b["lab_t"]["has"] = (torch.arange(B) % 3 != 0)
    return new, old, b, J


def _same_forward(new, old, b, ps=(0.0, 0.2, 0.5, 0.9, 1.0)):
    for p in ps:
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
        assert torch.equal(gn.get_state(), go.get_state()), f"p_gt {p}: the generator must be consumed exactly as before (the RNG stream is B's)"


@needs_ref
def test_full_builds_the_saved_pre_change_model_and_forward_bit_for_bit_with_the_same_rng_stream(world):
    new, old, b, _ = _old_pair(world)
    sn, so = new.state_dict(), old.state_dict()
    assert sn.keys() == so.keys() and all(torch.equal(sn[k], so[k]) for k in sn), "identical built model"
    ln, lo = new.built_levers(), old.built_levers()
    assert set(ln) - set(lo) == {"o_interface"} and set(lo) <= set(ln) and {k: v for k, v in ln.items() if k in lo} == lo
    assert {f.name for f in dataclasses.fields(M.V8FMiniConfig)} - {f.name for f in dataclasses.fields(REF_M.V8FMiniConfig)} == {"o_interface"}
    _same_forward(_wake(new), _wake(old), b)                                                    # woken, identically: at zero-init O is blind to what it reads and the identity would be vacuous
    new, old, b, _ = _old_pair(world)
    for p in (0.3, 1.0):                                                                        # the loss
        torch.manual_seed(5)
        ln_ = new.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        torch.manual_seed(5)
        lo_ = old.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        assert ln_["terms"].keys() == lo_["terms"].keys() and all(torch.equal(ln_["terms"][k], lo_["terms"][k]) for k in ln_["terms"])
    # the diagnostic and inference routes too (forward_diag with a code override is the eval's route)
    new.eval(), old.eval()
    with torch.no_grad():
        code = torch.rand(b["feats"].shape[0], M.CODE_DIM, generator=torch.Generator().manual_seed(8))
        for f in ("forward_infer",):
            on, oo = getattr(new, f)(b), getattr(old, f)(b)
            assert all(torch.equal(on[k], oo[k]) for k in on if isinstance(on[k], torch.Tensor))
        on, oo = new.forward_diag(b, code), old.forward_diag(b, code)
        assert all(torch.equal(on[k], oo[k]) for k in on if isinstance(on[k], torch.Tensor))


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
def test_full_reproduces_the_saved_pre_change_200_step_loss_trace_exactly_and_cut_does_not(world):
    """PREREG proof 2: the same seed, the same batches, the same optimiser, 200 steps: every loss float of `full` equals the saved pre-change module's. The control: `cut` leaves B's trace
    within three steps (once O's zero-init head has weights, O reads constants instead of S's checkpoints and T's code) -- the trace is sensitive to the lever it is meant to hold fixed."""
    new, old, _, J = _old_pair(world)
    bs = _trace_batches(world, new.cfg.hmax)
    t_new, t_old = _trace(new, bs), _trace(old, bs)
    assert len(t_new) == len(t_old) == 200 and all(math.isfinite(x) for x in t_new)
    assert t_new == t_old, f"first divergence at step {next(i for i, (a, b_) in enumerate(zip(t_new, t_old)) if a != b_)}"
    assert t_new[0] != t_new[-1], "the trace moves (the optimiser stepped)"
    cut, _, _, _ = _old_pair(world)
    cut.cfg = dataclasses.replace(cut.cfg, o_interface="cut")
    t_cut = _trace(cut, bs, steps=30)
    assert t_cut != t_new[:30], "the lever changes the trace"
    first = next(i for i, (x, y) in enumerate(zip(t_cut, t_new)) if x != y)
    assert t_cut[0] == t_new[0] and 1 <= first <= 3, f"step 0 is B's (O's residual head is zero-init, so O is blind to what it reads); the arms part as soon as the head has weights, at step {first}"


def _mutant(tmp_path, name, edits):
    """The model module with each (anchor, replacement) applied exactly once, loaded under a private name: a deliberate regression."""
    src = MODEL_P.read_text(encoding="utf-8")
    for a, b in edits:
        assert src.count(a) == 1, f"mutant {name}: anchor found {src.count(a)}x: {a!r}"
        src = src.replace(a, b)
    path = tmp_path / f"v8f_mini_{name}.py"
    path.write_text(src, encoding="utf-8")
    return _load_ref(f"v8f_mini_mut_{name}", path)


def _anchors():
    """The cut block's lines, read off the model source (so a re-wrapped comment cannot detach the mutants from the code), with the structure pinned."""
    nl = chr(10)
    lines = MODEL_P.read_text(encoding="utf-8").split(nl)
    i = [k for k, x in enumerate(lines) if x.startswith('        if cfg.o_interface == "cut":')]
    assert len(i) == 1, "exactly one cut line in the model"
    i = i[0]
    blk = lines[i:i + 4]
    assert "ck_nom" in blk[1] and "zeros_like(ck_ok)" in blk[2] and "zeros_like(code)" in blk[3] and lines[i + 4].lstrip().startswith("traj60, p30 = self.run_o("),         "the cut block is the cut line, the checkpoint line, the ok line, the code line, then O is called"
    ov = [x for x in lines if x.startswith("        if code_override is not None:")]
    assert len(ov) == 1
    return {"cut": '        if cfg.o_interface == "cut":', "block": nl.join(blk) + nl, "ck": blk[1] + nl, "ok": blk[2] + nl, "code": blk[3] + nl, "override": ov[0] + nl}



_A = _anchors()
CUT_LINE, CUT_BLOCK, CK_LINE, OK_LINE, CODE_LINE, OVERRIDE = _A["cut"], _A["block"], _A["ck"], _A["ok"], _A["code"], _A["override"]
MUTANTS = {
    "cut_line_deleted": [(CUT_BLOCK, "")],                                                                               # the cut line is deleted: cut == full
    "cut_never_runs": [(CUT_LINE, '        if False:')],
    "code_half_deleted": [(CODE_LINE, "            pass\n")],                                                             # O still reads T's code
    "checkpoint_half_deleted": [(CK_LINE, "            pass\n")],                                                         # O still reads S's checkpoints
    "ok_half_deleted": [(OK_LINE, "            pass\n")],                                                                 # O reads the nominal arcs but is told they are present
    "cut_before_the_overrides": [(CODE_LINE, "            pass\n"), (OVERRIDE, "        code = torch.zeros_like(code) if cfg.o_interface == \"cut\" else code\n" + OVERRIDE)],   # zeroed BEFORE the override
    "cut_applied_under_full": [(CUT_LINE, '        if True:')],
}


# =============================================================================================================================================
# (C) STRUCTURAL ZERO: PREREG PROOF 3
# =============================================================================================================================================
def proof_structural_zero(mod, J, b, mode):
    """Raises AssertionError unless the structural zero holds (mode cut) / the probe can see (mode full). Returns the max |d traj60| per perturbation."""
    model = _wake(_mk(mod, J, o_interface=mode))
    model.train()
    B = b["feats"].shape[0]
    res = {}
    with torch.no_grad():
        base = model.forward_infer(b)
        own = base["code"]
        g = torch.Generator().manual_seed(17)
        codes = {"donor_code": own.roll(1, 0), "gt_code": model._code_gt(b["lab_t"]), "random_code": torch.rand(B, mod.CODE_DIM, generator=g)}
        for name, c in codes.items():
            out = model.forward_diag(b, c)
            assert torch.equal(out["code"], c), f"{name}: the override must reach the interface (the probe would be blind)"
            res[name] = float((out["traj60"] - base["traj60"]).abs().max())
        s0 = type(model).run_s.__get__(model)
        model.run_s = lambda *a, **k: s0(*a, **k) + 2.5                                    # every S checkpoint shifted
        try:
            out = model.forward_infer(b)
        finally:
            model.__dict__.pop("run_s")
        assert float((out["ckpt"] - base["ckpt"]).abs().max()) > 2.0, "S's checkpoints did move"
        assert float((out["goal_logits"] - base["goal_logits"]).abs().max()) > 0.0, "T reads S (in BOTH modes)"
        res["shifted_checkpoints"] = float((out["traj60"] - base["traj60"]).abs().max())
        # the teacher forcing: p_gt 0 (T's own code, S's own checkpoints) vs p_gt 1 (the GT code, S's GT checkpoints where they exist)
        o0 = model._forward(b, 0.0, True, torch.Generator().manual_seed(2))
        o1 = model._forward(b, 1.0, True, torch.Generator().manual_seed(2))
        tf_moved = float((o1["code"] - o0["code"]).abs().max())
        res["teacher_forcing_p0_vs_p1"] = float((o1["traj60"] - o0["traj60"]).abs().max())
    if mode == "cut":
        assert all(v == 0.0 for v in res.values()), f"cut: traj60 must be bit-for-bit invariant to S's checkpoints and T's code, got max |d| {res}"
    else:
        assert all(v > 0.0 for v in res.values()), f"full: every perturbation must MOVE traj60 (the control that proves the probe can see), got {res}"
    assert tf_moved > 0.0, "the teacher forcing changed the code offered to the interface (the perturbation was applied)"
    return res


def test_under_cut_traj60_is_bit_for_bit_invariant_to_s_and_t_and_under_full_the_same_perturbation_moves_it(world):
    J, b = _J(world), _batch(world)
    rc, rf = proof_structural_zero(M, J, b, "cut"), proof_structural_zero(M, J, b, "full")
    assert set(rc) == set(rf) == {"donor_code", "gt_code", "random_code", "shifted_checkpoints", "teacher_forcing_p0_vs_p1"}
    assert max(rc.values()) == 0.0 and min(rf.values()) > 0.0


def test_under_cut_o_is_handed_the_constants_under_every_route(world):
    """What `run_o` receives, read off its arguments: zeros, the nominal arcs, ok = 0 -- in the training path, the inference path and the diagnostic path."""
    J, b = _J(world), _batch(world)
    model = _wake(_mk(M, J, o_interface="cut"))
    cap = {}
    o0 = model.run_o

    def run_o(bb, ad, last, ck_o, ck_ok, code):
        cap["ck_o"], cap["ck_ok"], cap["code"] = ck_o.detach().clone(), ck_ok.detach().clone(), code.detach().clone()
        return o0(bb, ad, last, ck_o, ck_ok, code)
    model.run_o = run_o
    B = b["feats"].shape[0]
    nom = model.ck_nom[None].expand(B, -1, -1)
    c = torch.rand(B, M.CODE_DIM, generator=torch.Generator().manual_seed(4))
    model.train()
    for route in (lambda: model._forward(b, 0.7, True, torch.Generator().manual_seed(1)), lambda: model.forward_infer(b), lambda: model.forward_diag(b, c), lambda: model.loss(b, p_gt=0.5)):
        cap.clear()
        route()
        assert torch.equal(cap["code"], torch.zeros_like(cap["code"])) and cap["code"].shape == (B, M.CODE_DIM)
        assert torch.equal(cap["ck_o"], nom) and torch.equal(cap["ck_ok"], torch.zeros(B, 3))
    full = _wake(_mk(M, J, o_interface="full"))
    o1 = full.run_o

    def run_o_f(bb, ad, last, ck_o, ck_ok, code):
        cap["code"], cap["ck_ok"] = code.detach().clone(), ck_ok.detach().clone()
        return o1(bb, ad, last, ck_o, ck_ok, code)
    full.run_o = run_o_f
    cap.clear()
    full.forward_diag(b, c)
    assert torch.equal(cap["code"], c) and torch.equal(cap["ck_ok"], torch.ones(B, 3)), "full: O is handed the override and the checkpoints flagged present"


def test_cut_reads_the_nominal_arcs_of_the_model_not_a_literal(world):
    """The constant IS the model's own `ck_nom` buffer (so a checkpoint's nominal arcs travel with it): changing the buffer changes what a cut O reads, and therefore its output."""
    J, b = _J(world), _batch(world)
    m = _wake(_mk(M, J, o_interface="cut"))
    m.train()
    with torch.no_grad():
        t0 = m.forward_infer(b)["traj60"]
        m.ck_nom.add_(1.0)
        t1 = m.forward_infer(b)["traj60"]
    assert not torch.equal(t0, t1)


# =============================================================================================================================================
# (D) GRADIENT REACHABILITY: PREREG PROOF 4
# =============================================================================================================================================
S_PARAMS = ("read_s.", "head_s.")
T_PARAMS = ("read_t.", "trunk_t.", "h_goal.", "h_lat.", "h_lon.", "h_cons.")


def _gsum(model, prefixes):
    return sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith(prefixes) and p.grad is not None)


def proof_gradient(mod, J, b, mode):
    """Raises AssertionError unless the gradient facts of `mode` hold. Returns the numbers."""
    model = _wake(_mk(mod, J, o_interface=mode))
    model.train()
    out = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(7))
    model.zero_grad(set_to_none=False)
    out["terms"]["total"].backward()
    res = {"e_code_weight": float(model.e_code.weight.grad.abs().sum()), "e_code_bias": float(model.e_code.bias.grad.abs().sum()),
           "S_total_loss": _gsum(model, S_PARAMS), "T_total_loss": _gsum(model, T_PARAMS)}
    for grp, pre in (("S", S_PARAMS), ("T", T_PARAMS)):                                  # every module of the group is trained by the total loss
        for mod_prefix in pre:
            assert _gsum(model, (mod_prefix,)) > 0.0, f"{mode}: {mod_prefix} receives no gradient from the total loss (S and T are still trained)"
    assert res["e_code_bias"] > 0.0, "the code token's bias is trained in both modes"
    # the O-only loss: ADE alone (p_gt 0.5: teacher forcing replaces some checkpoints and codes by constants, the rest flow)
    model.zero_grad(set_to_none=False)
    out = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(7))
    out["terms"]["ade"].backward()
    res.update({"S_ade_only": _gsum(model, S_PARAMS), "T_ade_only": _gsum(model, T_PARAMS), "e_code_weight_ade_only": float(model.e_code.weight.grad.abs().sum())})
    if mode == "cut":
        assert res["e_code_weight"] == 0.0, f"cut: e_code.weight.grad must be EXACTLY 0.0, got {res['e_code_weight']}"
        assert res["S_ade_only"] == 0.0 and res["T_ade_only"] == 0.0 and res["e_code_weight_ade_only"] == 0.0, f"cut: the O-only loss must not reach S or T: {res}"
    else:
        assert res["e_code_weight"] > 0.0, f"full: e_code.weight.grad must be non-zero, got {res['e_code_weight']}"
        assert res["S_ade_only"] > 0.0 and res["T_ade_only"] > 0.0, f"full: the O-only loss reaches S and T: {res}"
    return res


def test_gradient_under_cut_e_code_weight_is_exactly_zero_under_full_it_is_not_and_s_and_t_train_under_both(world):
    J, b = _J(world), _batch(world)
    rc, rf = proof_gradient(M, J, b, "cut"), proof_gradient(M, J, b, "full")
    assert rc["e_code_weight"] == 0.0 and rf["e_code_weight"] > 0.0
    assert rc["S_total_loss"] > 0.0 and rc["T_total_loss"] > 0.0 and rf["S_total_loss"] > 0.0 and rf["T_total_loss"] > 0.0
    assert rc["e_code_bias"] > 0.0 and rf["e_code_bias"] > 0.0


def test_gradient_to_the_checkpoint_token_survives_under_cut_because_t_shares_it(world):
    """e_ck is the SAME module for T and O, so under cut it still trains (T's side): the exact-zero control is e_code.weight alone, not e_ck."""
    J, b = _J(world), _batch(world)
    m = _wake(_mk(M, J, o_interface="cut"))
    m.train()
    m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(7))["terms"]["total"].backward()
    assert float(m.e_ck.weight.grad.abs().sum()) > 0.0


# =============================================================================================================================================
# (E) THE MUTANTS: PREREG PROOF 5 -- proofs C and D go RED
# =============================================================================================================================================
@pytest.mark.parametrize("name", ["cut_line_deleted", "cut_never_runs", "code_half_deleted", "checkpoint_half_deleted", "ok_half_deleted", "cut_before_the_overrides"])
def test_a_mutant_that_deletes_or_misplaces_the_cut_turns_the_structural_zero_red(world, tmp_path, name):
    mod = _mutant(tmp_path, name, MUTANTS[name])
    J, b = _J(world), _batch(world)
    try:
        if name == "ok_half_deleted":                                  # the ok flag is a third wire: the zero is bit-for-bit against S and T, but O is told the nominal arcs are present, which is not the
            with pytest.raises(AssertionError):                        # constant the PREREG names (ok = 0): the handed-constants test catches it
                _handed(mod, J, b)
        else:
            with pytest.raises(AssertionError, match="cut: traj60 must be bit-for-bit invariant"):
                proof_structural_zero(mod, J, b, "cut")
    finally:
        sys.modules.pop(f"v8f_mini_mut_{name}", None)


def _handed(mod, J, b):
    m = _wake(_mk(mod, J, o_interface="cut"))
    cap = {}
    o0 = m.run_o

    def run_o(bb, ad, last, ck_o, ck_ok, code):
        cap["ck_ok"] = ck_ok.detach().clone()
        return o0(bb, ad, last, ck_o, ck_ok, code)
    m.run_o = run_o
    m.train()
    m.forward_infer(b)
    assert torch.equal(cap["ck_ok"], torch.zeros_like(cap["ck_ok"])), "cut: O must be handed ok = 0"


@pytest.mark.parametrize("name", ["cut_line_deleted", "cut_never_runs", "code_half_deleted"])
def test_a_mutant_that_leaves_the_code_wire_turns_the_gradient_check_red(world, tmp_path, name):
    mod = _mutant(tmp_path, name, MUTANTS[name])
    J, b = _J(world), _batch(world)
    try:
        with pytest.raises(AssertionError, match="cut: e_code.weight.grad must be EXACTLY 0.0"):
            proof_gradient(mod, J, b, "cut")
    finally:
        sys.modules.pop(f"v8f_mini_mut_{name}", None)


def test_a_mutant_that_leaves_the_checkpoint_wire_turns_the_o_only_gradient_check_red(world, tmp_path):
    mod = _mutant(tmp_path, "checkpoint_half_deleted", MUTANTS["checkpoint_half_deleted"])
    J, b = _J(world), _batch(world)
    try:
        with pytest.raises(AssertionError, match="cut: the O-only loss must not reach S or T"):
            proof_gradient(mod, J, b, "cut")
    finally:
        sys.modules.pop("v8f_mini_mut_checkpoint_half_deleted", None)


@needs_ref
def test_a_mutant_that_applies_the_cut_under_full_breaks_the_identity_with_the_saved_module(world, tmp_path):
    mod = _mutant(tmp_path, "cut_applied_under_full", MUTANTS["cut_applied_under_full"])
    try:
        new, old, b, _ = _old_pair(world, mod=mod)
        with pytest.raises(AssertionError):
            _same_forward(_wake(new), _wake(old), b)
        new, old, _, _ = _old_pair(world, mod=mod)
        bs = _trace_batches(world, new.cfg.hmax)
        assert _trace(new, bs, steps=30) != _trace(old, bs, steps=30), "...and the loss trace parts from the saved module's within a few steps"
    finally:
        sys.modules.pop("v8f_mini_mut_cut_applied_under_full", None)


def test_the_cut_block_is_exactly_the_text_the_mutants_are_written_against():
    src = MODEL_P.read_text(encoding="utf-8")
    assert src.count(CUT_BLOCK) == 1 and src.count(CUT_LINE) == 1, "the mutants' anchors are the real lines"
    fwd = src.index("    def _forward(")
    nxt = src.index("    def forward_infer(")
    at = src.index(CUT_BLOCK)
    assert fwd < at < nxt and src.index(OVERRIDE) < at, "the cut line sits inside _forward, AFTER the code override"
    assert src.index("run_o(b, adapt, last_now, ck_for_o, ck_ok, code)") > at + len(CUT_BLOCK) - 1, "...and BEFORE O is called, with the rebound names"
    assert src.index("tout = self.run_t(b, adapt, last_t, ck_for_t, ck_ok)") < at, "T has read S before the cut: only the wires INTO O are cut"


# =============================================================================================================================================
# (F) G-DVB: PREREG PROOF 1
# =============================================================================================================================================
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


def _gate_parts(world, argv, **over):
    a = _args(argv)
    cfg, model, opt, info = _built(a, _J(world), **over)
    return a, cfg, model, opt, info


def test_the_flag_has_its_entry_and_a_new_flag_without_one_is_refused(world, monkeypatch):
    assert TR.FLAG_CHECKS.get("o_interface") == "lever:o_interface"
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert "o_interface" in dests and dests == set(TR.FLAG_CHECKS), "a flag without an entry, or an entry without a flag, is a launch blocker"
    a, cfg, model, opt, info = _gate_parts(world, ["--o-interface", "cut"])
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "o_interface"})
    assert any("o_interface" in m and "without a declared-vs-built entry" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


@pytest.mark.parametrize("argv", [[], ["--o-interface", "cut"], ["--o-interface", "full"]])
def test_gdvb_passes_what_was_built_and_nothing_else_moves(world, argv):
    a, cfg, model, opt, info = _gate_parts(world, argv)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    assert model.built_levers()["o_interface"] == a.o_interface


def test_gdvb_refuses_a_declared_cut_the_model_did_not_build_and_the_reverse(world):
    a, cfg, model, opt, info = _gate_parts(world, ["--o-interface", "cut"], o_interface="full")
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("lever o_interface: declared cut but built full" in m for m in bad), bad
    a, cfg, model, opt, info = _gate_parts(world, [], o_interface="cut")
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("lever o_interface: declared full but built cut" in m for m in bad), bad


def test_gdvb_refuses_a_model_built_by_a_mutant_that_ignores_the_config(world, tmp_path):
    """The built lever must be what the model DOES, not an echo of a flag: a module whose built_levers() reports the config while _forward never cuts still fails the probe (F2)."""
    mod = _mutant(tmp_path, "cut_never_runs_gdvb", MUTANTS["cut_never_runs"])
    try:
        J = _J(world)
        b = _batch(world)
        cfg = dataclasses.replace(TR.make_config(_args(["--o-interface", "cut"])), **{**_T.TINY, "wm": "off"})
        m = mod.V8FMini(mod.V8FMiniConfig(**cfg.__dict__), mod.P10Prior.synthetic(0), *M.cons_stats(J))
        assert m.built_levers()["o_interface"] == "cut", "the echo says cut ..."
        with pytest.raises(AssertionError, match="cut is declared but traj60 moved"):
            TR.o_interface_probe(m, b, m.cfg)                                              # ... and the probe, which reads what O is handed, refuses
    finally:
        sys.modules.pop("v8f_mini_mut_cut_never_runs_gdvb", None)


def test_live_check_requires_the_zero_under_cut_and_the_nonzero_under_full(world):
    """G-LIVE's dead-weight set: under cut e_code.weight is DECLARED dead and must read exactly 0.0 (declared_dead_but_alive otherwise); under full it is an ordinary trained weight."""
    J, b = _J(world), _batch(world)
    for mode, dead in (("cut", ["e_code.weight"]), ("full", [])):
        m = _mk(M, J, o_interface=mode)
        r = TR.live_check(m, b, m.cfg)
        assert r["pass"] is True and r["dead_params"] == dead and r["declared_dead"] == dead and r["unexpected_dead"] == [], (mode, r["dead_params"])
    # the deliberate regressions: a cut model that was built full (e_code.weight alive though declared dead) and a full model whose code is cut (dead though undeclared) both FAIL G-LIVE
    cfg_cut = dataclasses.replace(_mk(M, J).cfg, o_interface="cut")
    mf = _mk(M, J)
    with pytest.raises(AssertionError, match="G-LIVE failed"):
        TR.live_check(mf, b, cfg_cut)
    mc = _mk(M, J, o_interface="cut")
    with pytest.raises(AssertionError, match="G-LIVE failed"):
        TR.live_check(mc, b, dataclasses.replace(mc.cfg, o_interface="full"))


def test_the_probe_passes_both_modes_and_refuses_a_declaration_the_model_does_not_build(world):
    J, b = _J(world), _batch(world)
    mf, mc = _mk(M, J), _mk(M, J, o_interface="cut")
    pf, pc = TR.o_interface_probe(mf, b, mf.cfg), TR.o_interface_probe(mc, b, mc.cfg)
    assert pf["pass"] is True and pc["pass"] is True and pf["o_interface"] == "full" and pc["o_interface"] == "cut"
    assert all(v["max_abs_dtraj60"] == 0.0 and v["o_read_code_abs_max"] == 0.0 and v["o_read_ck_ok_abs_max"] == 0.0 for v in pc["perturbations"].values()), "the control reads EXACTLY 0.0 under cut"
    assert all(v["max_abs_dtraj60"] > 0.0 for v in pf["perturbations"].values()), "and moves under full"
    assert pc["perturbations"]["shifted_checkpoints"]["t_reads_s_max_abs_dgoal_logits"] > 0.0
    with pytest.raises(AssertionError, match="cut is declared but traj60 moved"):
        TR.o_interface_probe(mf, b, mc.cfg)                                                 # declared cut, built full
    with pytest.raises(AssertionError, match="full is declared but traj60 did not move"):
        TR.o_interface_probe(mc, b, mf.cfg)                                                 # declared full, built cut
    with pytest.raises(AssertionError, match="single window"):
        TR.o_interface_probe(mc, {"feats": b["feats"][:1]}, mc.cfg)                         # a cut gate on one window proves nothing: refused
    skipped = TR.o_interface_probe(mf, {"feats": b["feats"][:1]}, mf.cfg)                    # the default interface on a one-window smoke batch: no refusal, and it says it did not look
    assert skipped["pass"] is True and skipped["perturbations"] == {} and "single window" in skipped["skipped"]


# =============================================================================================================================================
# (G) REACHABILITY FROM THE ACTUAL CALLER
# =============================================================================================================================================
def _train(out, extra, steps=4):
    """A real `TR.main` run on the synthetic world with a spy on what `run_o` is handed."""
    cap = {"o": []}
    run_o0 = M.V8FMini.run_o

    def run_o(self, bb, ad, last, ck_o, ck_ok, code):
        cap["o"].append({"code": code.detach().clone(), "ck_o": ck_o.detach().clone(), "ck_ok": ck_ok.detach().clone(), "grad": torch.is_grad_enabled(), "nom": self.ck_nom.detach().clone(),
                         "mode": self.cfg.o_interface})
        return run_o0(self, bb, ad, last, ck_o, ck_ok, code)
    M.V8FMini.run_o = run_o
    try:
        assert TR.main(["--out", str(out), *COMMON, "--steps", str(steps), *extra]) == 0
    finally:
        M.V8FMini.run_o = run_o0
    out = Path(out)
    return {"cap": cap, "out": out, "root": out / "_synthetic", "ck": torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False),
            "sc": json.loads((out / "selfcheck.json").read_text(encoding="utf-8")), "sm": json.loads((out / "summary.json").read_text(encoding="utf-8")),
            "argv": json.loads((out / "argv.json").read_text(encoding="utf-8"))["argv"],
            "log": [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("flat_runs")
    r = {"full": _train(base / "full", []), "cut": _train(base / "cut", ["--o-interface", "cut"])}
    for x in r.values():                                   # the six TRAINING clips (2.5 MB each, 2 runs) are not read after the 4 steps; the eval clips, the joins and the checkpoint are
        shutil.rmtree(Path(x["root"]) / "train" / "clips", ignore_errors=True)
    return r


def test_trainer_argv_reaches_the_config_the_checkpoint_the_records_and_the_gate(runs):
    r = runs["cut"]
    assert r["argv"].count("--o-interface") == 1 and r["argv"][r["argv"].index("--o-interface") + 1] == "cut", "the flag is NAMED in the argv (a default of full would otherwise test nothing)"
    assert r["ck"]["cfg"]["o_interface"] == "cut", "the checkpoint records the lever (the eval builds from it)"
    assert r["sm"]["levers"]["o_interface"] == "cut" and r["sc"]["levers"]["o_interface"] == "cut"
    pr = r["sc"]["o_interface_probe"]
    assert r["sc"]["declared_vs_built"] == "PASS" and pr["pass"] is True and pr["o_interface"] == "cut" and r["sm"]["done"] is True
    assert all(v["max_abs_dtraj60"] == 0.0 for v in pr["perturbations"].values())
    assert r["sc"]["live"]["pass"] is True and r["sc"]["live"]["control_prior_grad_sum"] == 0.0 and r["sc"]["live"]["declared_dead"] == ["e_code.weight"] == r["sc"]["live"]["dead_params"]
    w = runs["full"]
    assert "--o-interface" not in w["argv"] and w["ck"]["cfg"]["o_interface"] == "full" and w["sc"]["levers"]["o_interface"] == "full"
    assert w["sc"]["o_interface_probe"]["pass"] is True and all(v["max_abs_dtraj60"] > 0.0 for v in w["sc"]["o_interface_probe"]["perturbations"].values())
    assert w["sc"]["live"]["dead_params"] == [] and w["sc"]["live"]["declared_dead"] == []


def test_o_is_handed_the_constants_on_every_training_step_of_the_cut_run_and_the_real_code_in_the_default_run(runs):
    steps = runs["cut"]["sm"]["steps"]
    for tag in ("cut", "full"):
        grad = [o for o in runs[tag]["cap"]["o"] if o["grad"]]
        train = grad[-steps:]
        assert len(train) == steps >= 4
        for o in train:
            if tag == "cut":
                B = o["code"].shape[0]
                assert o["mode"] == "cut" and torch.equal(o["code"], torch.zeros_like(o["code"])) and torch.equal(o["ck_o"], o["nom"][None].expand(B, -1, -1)) and torch.equal(o["ck_ok"], torch.zeros(B, 3))
            else:
                assert o["mode"] == "full" and float(o["code"].abs().sum()) > 0.0 and float(o["ck_ok"].min()) == 1.0
    assert runs["cut"]["sm"]["levers"]["params"] == runs["full"]["sm"]["levers"]["params"], "the lever adds no parameter"
    assert runs["cut"]["sm"]["steps"] == runs["full"]["sm"]["steps"] and runs["cut"]["sc"]["info"]["n_windows"] == runs["full"]["sc"]["info"]["n_windows"]


def _dead_adam(r):
    """Names of the trainable parameters whose Adam second moment is all zero in the saved optimiser state (Adam allocates state lazily and a zero gradient leaves it at 0)."""
    ck = r["ck"]
    names = [n for n, p in _names_of(r)]
    dead = []
    for i, n in enumerate(names):
        st = ck["opt"]["state"].get(i)
        if st is None or float(st["exp_avg_sq"].abs().sum()) == 0.0:
            dead.append(n)
    return dead


def _names_of(r):
    """The trainer's parameter order: the weight-decay group (ndim >= 2) first, then the rest, each in named_parameters order."""
    cfg = M.V8FMiniConfig(**r["ck"]["cfg"])
    mu, sd_ = M.cons_stats(D.load_join(Path(r["root"]) / "join_train.npz"))
    ps = [(n, p) for n, p in M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd_).named_parameters() if p.requires_grad]
    return [(n, p) for n, p in ps if p.ndim >= 2] + [(n, p) for n, p in ps if p.ndim < 2]


def test_the_trained_cut_checkpoint_has_exactly_one_untrained_weight_and_it_is_e_code_weight(runs):
    """The post-training gate of CLAUDE.md (Adam allocates state lazily): under cut the ONLY parameter whose second moment is all zero is e_code.weight; under full none is."""
    assert _dead_adam(runs["cut"]) == ["e_code.weight"]
    assert _dead_adam(runs["full"]) == []


def test_the_default_run_is_the_pre_change_trainer_for_every_other_field(runs):
    c, f = runs["cut"], runs["full"]
    assert {k: v for k, v in c["ck"]["cfg"].items() if k != "o_interface"} == {k: v for k, v in f["ck"]["cfg"].items() if k != "o_interface"}
    assert [x["p_gt"] for x in c["log"]] == [x["p_gt"] for x in f["log"]], "same p_gt schedule"
    # step 0: O's residual head is zero-init, so O is blind to everything it reads and EVERY loss term is B's; the lever acts through O alone
    assert all(c["log"][0][k] == f["log"][0][k] for k in c["log"][0] if k not in ("sec", "grad_norm")), (c["log"][0], f["log"][0])
    assert any(c["log"][i]["ade"] != f["log"][i]["ade"] for i in range(1, len(c["log"]))), "once the head is non-zero the two arms part"


def test_the_trainer_source_has_every_hop_of_the_trace():
    tr = (REPO / "stack" / "scripts" / "v8f_mini_train.py").read_text(encoding="utf-8")
    mod = MODEL_P.read_text(encoding="utf-8")
    assert '"--o-interface", dest="o_interface", choices=M.O_INTERFACE_MODES, default="full"' in tr
    assert "o_interface=a.o_interface" in tr and '"o_interface": "lever:o_interface"' in tr
    assert 'built.get("o_interface") != a.o_interface' in tr
    assert 'if cfg.o_interface == "cut":' in tr and 'n == "e_code.weight"' in tr
    assert "o_interface_probe(model, probe, cfg)" in tr and '"o_interface_probe": ocut' in tr
    assert 'o_interface: str = "full"' in mod and 'if cfg.o_interface == "cut":' in mod and '"o_interface": cfg.o_interface' in mod


# =============================================================================================================================================
# (H) G-EVAL: the eval builds the interface from the checkpoint; the identity read
# =============================================================================================================================================
def _eval_args(root, ck, tag, out, *extra):
    root = Path(root)
    return ["run", "--ckpt", str(ck), "--tag", tag, "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
            "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(out), "--n-boot", "20",
            "--batch", "16", "--lru", "4", "--tracks", str(root / "native_tracks.pt"), *extra]


def _eval(r, tag, out, *extra):
    cap = []
    init0 = M.V8FMini.__init__

    def init(self, cfg, *a, **k):
        cap.append(cfg.o_interface)
        return init0(self, cfg, *a, **k)
    M.V8FMini.__init__ = init
    try:
        assert EV.main(_eval_args(r["root"], r["out"] / "ckpt_last.pt", tag, out, *extra)) == 0
    finally:
        M.V8FMini.__init__ = init0
    z = np.load(Path(out) / f"eval_{tag}.npz")
    return json.loads((Path(out) / f"eval_{tag}.json").read_text(encoding="utf-8")), z, cap


def test_the_eval_builds_the_interface_from_the_checkpoint_and_the_gt_code_oracle_reads_the_identity(runs, tmp_path):
    """The PREREG's identity read, through the real eval: FLAT's GT-code oracle predicts EXACTLY what its own code predicts (max |d pred| = 0.0, ADE6 equal window by window), while the manipulation
    DID happen upstream (the code offered to the interface moved on every window that has a label); a full checkpoint's oracle moves."""
    out = {}
    for tag in ("cut", "full"):
        r = runs[tag]
        j0, z0, cap0 = _eval(r, f"{tag}_own", tmp_path)
        j1, z1, cap1 = _eval(r, f"{tag}_orc", tmp_path, "--tac", "oracle")
        assert cap0 == cap1 == [tag], f"the eval built its model with o_interface {cap0}/{cap1}; the checkpoint says {tag}"
        assert j0["o_interface"] == tag and j1["o_interface"] == tag
        assert j1["tac_manipulation"]["mode"] == "oracle" and j1["tac_manipulation"]["frac_code_changed"] > 0.0, "the oracle changed the code offered to the interface"
        out[tag] = (float(np.abs(z1["pred"] - z0["pred"]).max()), float(np.abs(z1["ade6"] - z0["ade6"]).max()), z0, z1)
    assert out["cut"][0] == 0.0 and out["cut"][1] == 0.0, "FLAT: the GT-code oracle minus own is EXACTLY 0"
    assert out["full"][0] > 0.0 and out["full"][1] > 0.0, "the control: the same eval moves a full checkpoint, so the zero is the cut's"
    assert float(np.abs(out["cut"][3]["tac_code"]).sum()) > 0.0, "tac_code is the offered (GT) code, not O's zeros"


def test_the_eval_of_an_older_checkpoint_without_the_key_builds_full(runs, tmp_path):
    """A checkpoint written before this lever has no `o_interface` in its cfg: the eval builds the dataclass default, full (every banked arm)."""
    r = runs["full"]
    ck = torch.load(r["out"] / "ckpt_last.pt", map_location="cpu", weights_only=False)
    ck["cfg"].pop("o_interface")
    d = tmp_path / "old"
    d.mkdir()
    torch.save(ck, d / "ckpt_last.pt")
    cap = []
    init0 = M.V8FMini.__init__

    def init(self, cfg, *a, **k):
        cap.append(cfg.o_interface)
        return init0(self, cfg, *a, **k)
    M.V8FMini.__init__ = init
    try:
        assert EV.main(_eval_args(r["root"], d / "ckpt_last.pt", "old", tmp_path)) == 0
    finally:
        M.V8FMini.__init__ = init0
    assert cap == ["full"]
    j0, z0, _ = _eval(r, "full_again", tmp_path)
    assert np.array_equal(np.load(tmp_path / "eval_old.npz")["pred"], z0["pred"])


def test_the_eval_source_builds_the_config_from_the_checkpoint_and_records_it():
    src = (REPO / "stack" / "scripts" / "v8f_mini_eval.py").read_text(encoding="utf-8")
    assert 'cfg = M.V8FMiniConfig(**ck["cfg"])' in src and '"o_interface": cfg.o_interface' in src
    assert "--o-interface" not in src and "o_interface=" not in src, "the eval has no flag of its own: a flag on an eval would be a path the checkpoint can contradict"


# =============================================================================================================================================
# (I) THE READER: the PREREG's outcome table applied verbatim, and every refusal
# =============================================================================================================================================
BANKED = PKG / "raw" / "full" / "runs"
_NEED_RD = READ_P.exists() and BANKED.exists() and (HERE_T / "test_v8f_hist.py").exists()
needs_pkg = pytest.mark.skipif(not _NEED_RD, reason="the v8F-mini package (the reader, the banked M2b runs) is not in this checkout")
R = _load("v8f_flat_read_under_test", READ_P) if READ_P.exists() else None
_H = _load("test_v8f_hist_for_flat", HERE_T / "test_v8f_hist.py") if _NEED_RD else None
XT = "M2b-flat"


@pytest.fixture
def no_known(monkeypatch):
    """The banked premises (F 0.1207, LAMBDA +0.2678 / +0.1667, S5 excess) are the REAL banked M2b values; a synthetic world cannot hold them. Everything else is tested with the check
    replaced; `test_known_values_are_checked...` runs the real one on the synthetic world and requires the refusal."""
    monkeypatch.setattr(R.PGT, "known_values", lambda *a, **k: {"checked": False, "why": "synthetic test input"})


def _ej(path, fn):
    """Edit a json file IN PLACE (`fn` mutates it; whatever it returns is ignored: a `.pop(...)` must not replace the file by the popped value)."""
    p = Path(path)
    j = json.loads(p.read_text(encoding="utf-8"))
    fn(j)
    p.write_text(json.dumps(j), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_strata(path, w, fb, *, md5=None, cu=None, tit=None, **over):
    """A strata file in the exact format of mk_window_strata.py, with the stratum written HERE (independently of the reader's own `stratum_codes`): JUNCTION (0) = turn_is_turn on a window
    with a v9 row; CURVE (1) = curve on one; NO-TURN (2) = every other window. Layout: JUNCTION over 11 episodes, CURVE over 4 (under the 10-episode floor), NO-TURN over all 12."""
    wi = np.tile(np.arange(w.n_win), w.n_ep)
    cu = ((w.eid >= 8) & (wi == 4)).astype(np.int8) if cu is None else cu
    tit = ((w.eid <= 10) & (wi == 3)).astype(np.int8) if tit is None else tit
    has = ~np.asarray(fb, bool)
    s = np.full(w.n, 2, np.int8)
    s[(cu == 1) & has] = 1
    s[(tit == 1) & has] = 0
    arrs = dict(stratum=s, curve=cu, turn_is_turn=tit, has=has, eid=w.eid, t=w.t, a_t=np.zeros(w.n, np.int64), rowT=np.arange(w.n),
                provenance=np.array(json.dumps({"release_md5": md5 or R.KR.RELEASE_MD5, "definition": "synthetic test strata"})))
    arrs.update(over)
    np.savez(path, **arrs)
    return path


@pytest.fixture(scope="module")
def gate_rec(runs):
    """The launch-gate records of the REAL tiny runs (`runs`, part G): what the trainer itself writes into selfcheck.json under cut (G-LIVE with its dead set, the o_interface probe) and under
    full. The reader's gate-token check is therefore tested against the trainer's own output, not against a record this file invented."""
    return {"cut": {k: runs["cut"]["sc"][k] for k in ("live", "o_interface_probe")}, "full": {k: runs["full"]["sc"][k] for k in ("live", "o_interface_probe")}}


def _fworld(d, gate, seeds=(0, 1), *, d_ade=0.7, same_as_b=False, **hw):
    """Synthetic evals + runs for B (M2b, both seeds, ALWAYS the real banked run records) and X = FLAT = B + --o-interface cut (`seeds`). Knobs: `d_ade` = ADE6(FLAT) - ADE6(B) (one value, or one
    per seed; the PREREG's Delta = ADE6(B) - ADE6(FLAT) is its NEGATIVE: d_ade +0.7 = the hierarchy is better by 0.7 m); `same_as_b` = FLAT's own-code evals ARE B's arrays.
    FLAT's two oracle evals carry FLAT's OWN predictions (the structural identity of the cut: O reads constants whatever the evaluator writes) and say they changed the code."""
    d = Path(d)
    da = tuple(d_ade) if isinstance(d_ade, (tuple, list)) else (d_ade, d_ade)
    f = _H._hworld(d, seeds, d_ade=da[0], same_as_b=same_as_b, **hw)
    if 1 in seeds and da[1] != da[0] and not same_as_b:
        g = _H._hworld(d / "alt", (1,), d_ade=da[1], **hw)
        for p in g["eval"].glob("eval_M2b-h10_s1*"):
            shutil.copyfile(p, f["eval"] / p.name)
        shutil.rmtree(d / "alt")
    ev, ru, N = f["eval"], f["runs"], f["w"].n
    with np.load(ev / "eval_M2b_s0-tacabs.npz") as zb:
        fb = np.asarray(zb["tac_oracle_fallback"]).copy()
    sha = {"M2b_s0": R.BANKED_CKPT_SHA["M2b_s0"], "M2b_s1": R.BANKED_CKPT_SHA["M2b_s1"]}
    for s in seeds:
        with np.load(ev / f"eval_M2b-h10_s{s}.npz") as std:                          # closed before the file is unlinked below (Windows refuses to delete an open file)
            own_pred, own_ade = np.asarray(std["pred"]).copy(), np.asarray(std["ade6"]).copy()
        for suf in R.X_SUFFIXES:
            for ext in ("npz", "json"):
                src, dst = ev / f"eval_M2b-h10_s{s}{suf}.{ext}", ev / f"eval_{XT}_s{s}{suf}.{ext}"
                if ext == "json":
                    j = json.loads(src.read_text(encoding="utf-8"))
                    j["ckpt"] = f"D:\\runs\\M2b_s{s}_flat\\ckpt_last.pt"
                    j["o_interface"] = "cut"
                    if suf == "-tacorc":
                        j["tac_manipulation"].update(code_shift_mean=0.8, frac_code_changed=0.62, n_oracle_fallback_no_v9_row=int(fb.sum()), n_windows=N)
                    dst.write_text(json.dumps(j), encoding="utf-8")
                else:
                    with np.load(src) as zz:
                        z = {k: zz[k] for k in zz.files}
                    if suf in ("-tacorc", "-tacorc-all-latcons"):
                        z["pred"], z["ade6"] = own_pred.copy(), own_ade.copy()
                    np.savez(dst, **z)
                src.unlink()
    for s in seeds:
        shutil.rmtree(ru / f"M2b_s{s}_h10")
        real, log = _H._b_real(s)
        bl = real["summary.json"]["levers"]
        argv = list(real["argv.json"]["argv"])
        at = argv.index("--out")
        argv[at:at] = ["--o-interface", "cut"]
        argv[argv.index("--out") + 1] = f"D:\\runs\\M2b_s{s}_flat"
        lev = {**bl, "cons_absent": "mask", "p_gt_start": 1.0, "p_gt_end": 0.2, "residual_zero_init": False, "hist": 4, "hmax": 16, "window_start": 3, "code_mix": "whole",
               "o_interface": "cut", "w": {**bl["w"], "w_tac": 1.0, "w_cons": 0.5}}
        a_sha = f"f{s}M2b"
        sm = {**real["summary.json"], "argv_sha": a_sha, "levers": lev}
        sc = {**real["selfcheck.json"], "declared_vs_built": "PASS", "argv_sha": a_sha, "levers": dict(lev, residual_zero_init=True),
              "live": copy.deepcopy(gate["cut"]["live"]), "o_interface_probe": copy.deepcopy(gate["cut"]["o_interface_probe"]),
              "info": {**(real["selfcheck.json"].get("info") or {}), "window_start": 3, "window_t_min": 3, "ds_window_start": 3}}
        rd = ru / f"M2b_s{s}_flat"
        rd.mkdir()
        (rd / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
        (rd / "argv.json").write_text(json.dumps({"argv": argv, "sha": a_sha}), encoding="utf-8")
        (rd / "selfcheck.json").write_text(json.dumps(sc), encoding="utf-8")
        (rd / "train_log.jsonl").write_text(log, encoding="utf-8")
        torch.save({"model": {"time": torch.zeros(4, 256), "pos": torch.zeros(40, 4)},
                    "cfg": {"o_interface": "cut", "code_mix": "whole", "hist": 4, "hmax": 16, "window_start": -1, "d": 256, "n_tok": 40, "p_gt_start": 1.0, "p_gt_end": 0.2},
                    "grid": "pool4x10", "argv_sha": a_sha, "step": sm["steps"], "arm": "M2b", "seed": s}, rd / "ckpt_last.pt")
        sha[f"{XT}_s{s}_trained"] = sha[f"{XT}_s{s}_after_evals"] = _sha(rd / "ckpt_last.pt")
    ckj = d / "ckpt_sha256.json"
    ckj.write_text(json.dumps(sha), encoding="utf-8")
    st = _write_strata(d / "strata.npz", f["w"], fb)
    args = list(f["args"])
    k = args.index("--out")
    args[k:k + 2] = ["--out", str(d / "flat.json")]
    n_ep = int(np.unique(f["w"].eid).size)
    f["args"] = args + ["--ckpt-sha", str(ckj), "--n-windows", str(N), "--n-episodes", str(n_ep), "--strata", str(st)]
    f["ck"], f["sha"], f["out"], f["strata"], f["fb"] = ckj, sha, d / "flat.json", st, fb
    return f


@pytest.fixture
def fw(gate_rec):
    """A world factory. Each world is removed when its test ends (about 8 MB each, ~120 of them: the C: drive these runs share has little room)."""
    made = []

    def mk(d, **kw):
        made.append(Path(d))
        return _fworld(d, gate_rec, **kw)
    yield mk
    for d in made:
        shutil.rmtree(d, ignore_errors=True)


def _go(f, extra=()):
    rc = R.main([*f["args"], *extra])
    return rc, (json.loads(f["out"].read_text(encoding="utf-8")) if rc == 0 else None)


def _xr(f, s=0):
    return f["runs"] / f"M2b_s{s}_flat"


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


def _npz(path):
    """An npz as a dict of arrays, the file closed (Windows refuses to overwrite or delete a file an NpzFile still holds open)."""
    with np.load(path) as zz:
        return {k: zz[k] for k in zz.files}


def _npz_edit(path, fn):
    with np.load(path) as zz:
        z = {k: zz[k] for k in zz.files}
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


def _fewer_steps(f, n):
    """A run that stopped at `n` steps, with its own consistent log (so the step-count check, not the log check, is what must refuse)."""
    _ej(_xr(f) / "summary.json", lambda j: j.update(steps=n))
    _log_edit(f, lambda r: r.__setitem__(slice(None), [dict(x, p_gt=_H._sched(x["step"] - 1, n, 1.0, 0.2)) for x in r if x["step"] <= n]))


def _shift_plan(f, s, dy, rows=None):
    """FLAT's seed-s plan moved laterally by `dy` m beyond 2 s on `rows` (default: all windows), in ALL FOUR X evals at once (the oracle evals keep reading FLAT's own plan: the identity holds)."""
    w = f["w"]
    m = np.ones(w.n, bool) if rows is None else np.asarray(rows, bool)
    for suf in R.X_SUFFIXES:
        p = f["eval"] / f"eval_{XT}_s{s}{suf}.npz"
        with np.load(p) as zz:
            z = {k: zz[k] for k in zz.files}
        pred = z["pred"].astype(np.float64).copy()
        pred[m, 10:, 1] += dy
        z["pred"] = pred.astype(np.float32)
        z["ade6"] = np.linalg.norm(z["pred"].astype(np.float64) - w.gt, axis=-1).mean(1)
        np.savez(p, **z)


# ---- the committed outcome table ----------------------------------------------------------------------------------------------------------------
def _table(s0, s1):
    """The PREREG's outcome table WITH AMENDMENT 1, restated HERE from the file's own wording (independent of the reader's `verdict`). LO = H = Delta <= -2F and separated,
    HI = F = Delta >= +2F and separated, MID = M = anything else."""
    if s0 == "LO" and s1 == "LO":
        return "HIERARCHY BEATS FLAT"
    if s0 == "HI" and s1 == "HI":
        return "FLAT BETTER"
    if {s0, s1} == {"LO", "HI"}:
        return "SEEDS DISAGREE"
    if "HI" not in (s0, s1):
        return "NOT ESTABLISHED"          # either seed misses -2F or is not separated, and NEITHER seed reads >= +2F
    return "FLAT LEANING"                  # Amendment 1: one seed F, the other M (never FLAT BETTER, which needs both seeds)


STATES = ("LO", "MID", "HI")


@needs_pkg
@pytest.mark.parametrize("s0", STATES)
@pytest.mark.parametrize("s1", STATES)
def test_the_verdict_function_is_the_preregs_table_cell_by_cell(s0, s1):
    vd, cell = R.verdict((s0, s1))
    assert vd == _table(s0, s1) and cell == "defined"


# ---- Amendment 1 (PREREG_V8F_FLAT.md at ef432e37, written before any FLAT data): the nine H / F / M pairs, each with exactly one outcome -----------------------------------------
H_F_M_TO_STATE = {"H": "LO", "F": "HI", "M": "MID"}                       # H = Delta <= -2F separated, F = Delta >= +2F separated, M = anything else
NINE = {("H", "H"): "HIERARCHY BEATS FLAT", ("F", "F"): "FLAT BETTER", ("H", "F"): "SEEDS DISAGREE", ("F", "H"): "SEEDS DISAGREE",
        ("F", "M"): "FLAT LEANING", ("M", "F"): "FLAT LEANING", ("H", "M"): "NOT ESTABLISHED", ("M", "H"): "NOT ESTABLISHED", ("M", "M"): "NOT ESTABLISHED"}


def _check_nine(verdict_fn):
    """Enumerate every (H/F/M, H/F/M) pairing and assert the exact outcome name for each (the expected names are the literal `NINE` above, not read from the reader)."""
    assert len(NINE) == 9 and {a for a, _ in NINE} == {b for _, b in NINE} == set("HFM")
    for s0 in "HFM":
        for s1 in "HFM":
            vd, cell = verdict_fn((H_F_M_TO_STATE[s0], H_F_M_TO_STATE[s1]))
            assert vd == NINE[(s0, s1)], f"({s0}, {s1}) must read {NINE[(s0, s1)]}, the reader says {vd}"
            assert cell == "defined"
    return True


@needs_pkg
def test_amendment_1_all_nine_pairs_map_to_their_exact_outcome():
    assert _check_nine(R.verdict)
    assert {R.verdict((H_F_M_TO_STATE[a], H_F_M_TO_STATE[b]))[0] for a, b in NINE} == set(R.OUTCOMES), "every outcome is reachable, none is invented"
    assert tuple(R.OUTCOMES) == ("HIERARCHY BEATS FLAT", "NOT ESTABLISHED", "FLAT BETTER", "SEEDS DISAGREE", "FLAT LEANING")
    assert [k for k, v in NINE.items() if v == "FLAT BETTER"] == [("F", "F")], "FLAT BETTER needs BOTH seeds at F: (F, M) / (M, F) never read FLAT BETTER"
    assert R.OUTCOME_OF == NINE and dict(R.HFM) == {v: k for k, v in H_F_M_TO_STATE.items()}
    assert all(R.verdict((a, b))[0] == _table(a, b) for a in R.HFM for b in R.HFM), "the nine-cell table is the pre-amendment table plus its one new row, nothing else"
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "NO ROW" not in src and '"gap"' not in src, "the NO ROW / gap path is gone"


@needs_pkg
@pytest.mark.parametrize("swap", [(("F", "M"), ("H", "M")), (("F", "F"), ("F", "M")), (("H", "H"), ("M", "M")), (("H", "F"), ("M", "F"))],
                         ids=["LEANING<->NOT_ESTABLISHED", "FLAT_BETTER<->LEANING", "BEATS<->NOT_ESTABLISHED", "DISAGREE<->LEANING"])
def test_amendment_1_mutation_swapping_two_outcomes_goes_red(monkeypatch, swap):
    """The deliberate-regression arm of the nine-cell test: swap the outcomes of two cells in the reader's table and the enumeration must fail."""
    a, b = swap
    assert R.OUTCOME_OF[a] != R.OUTCOME_OF[b], "a swap of two equal outcomes is not a mutation"
    mutated = dict(R.OUTCOME_OF)
    mutated[a], mutated[b] = R.OUTCOME_OF[b], R.OUTCOME_OF[a]
    monkeypatch.setattr(R, "OUTCOME_OF", mutated)
    with pytest.raises(AssertionError):
        _check_nine(R.verdict)


@needs_pkg
@pytest.mark.parametrize("bad", [("LO", "BOGUS"), ("", "MID"), ("lo", "hi"), ("LO",), ("LO", "LO", "LO"), (), None, "LOLO", ("H", "F")], ids=repr)
def test_amendment_1_an_unmapped_pair_is_a_hard_error_rc_2_never_a_verdict(bad, capsys):
    with pytest.raises(SystemExit) as e:
        R.verdict(bad)
    assert e.value.code == 2 and "cannot place" in capsys.readouterr().err


@needs_pkg
def test_amendment_1_a_table_with_a_missing_cell_is_a_hard_error_rc_2(monkeypatch, capsys):
    monkeypatch.setattr(R, "OUTCOME_OF", {k: v for k, v in R.OUTCOME_OF.items() if k != ("F", "M")})
    with pytest.raises(SystemExit) as e:
        R.verdict(("HI", "MID"))
    assert e.value.code == 2 and "no row in the nine-cell outcome table" in capsys.readouterr().err


@needs_pkg
def test_a_seed_state_needs_the_committed_2F_AND_a_separated_interval_on_the_right_side():
    bar = R.BAR_2F
    assert bar == 0.2415

    def p(point, lo, hi, sep):
        return {"point": point, "lo": lo, "hi": hi, "separated": sep}
    assert R.seed_state(p(-bar, -0.9, -0.1, True)) == "LO", "the bar is inclusive"
    assert R.seed_state(p(-bar + 1e-9, -0.9, -0.1, True)) == "MID", "a point just short of -2F misses"
    assert R.seed_state(p(-0.9, -1.5, 0.1, False)) == "MID", "the point clears -2F but the interval straddles 0: not separated"
    assert R.seed_state(p(-0.9, -1.5, 0.1, True)) == "MID", "`separated` alone is not enough: the interval must sit below 0"
    assert R.seed_state(p(bar, 0.1, 0.9, True)) == "HI" and R.seed_state(p(bar - 1e-9, 0.1, 0.9, True)) == "MID"
    assert R.seed_state(p(0.9, -0.1, 1.5, False)) == "MID"
    assert R.seed_state(p(0.9, -0.1, 1.5, True)) == "MID"
    assert R.seed_state(p(0.0, -0.1, 0.1, False)) == "MID"
    assert R.seed_state(p(-0.9, 0.1, 1.5, True)) == "MID", "a point on one side with an interval on the other (a broken record) is not LO"


CELLS = [((0.7, 0.7), "HIERARCHY BEATS FLAT", ("LO", "LO")), ((0.0, 0.0), "NOT ESTABLISHED", ("MID", "MID")), ((-0.7, -0.7), "FLAT BETTER", ("HI", "HI")),
         ((0.7, -0.7), "SEEDS DISAGREE", ("LO", "HI")), ((-0.7, 0.7), "SEEDS DISAGREE", ("HI", "LO")), ((0.7, 0.0), "NOT ESTABLISHED", ("LO", "MID")),
         ((0.0, 0.7), "NOT ESTABLISHED", ("MID", "LO")), ((-0.7, 0.0), "FLAT LEANING", ("HI", "MID")), ((0.0, -0.7), "FLAT LEANING", ("MID", "HI"))]


@needs_pkg
@pytest.mark.parametrize("d_ade, want, states", CELLS)
def test_reader_decision_table_every_cell_end_to_end(tmp_path, no_known, fw, capsys, d_ade, want, states):
    f = fw(tmp_path / "w", d_ade=d_ade)
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    assert r["VERDICT"] == want and tuple(r["seed_states"][f"s{s}"] for s in (0, 1)) == states
    assert r["VERDICT"] == _table(*states)
    for s, da in enumerate(d_ade):                                                  # Delta = ADE6(B) - ADE6(FLAT) = -d_ade, to the solver's tolerance
        assert r["rows"][f"s{s}"]["primary"]["point"] == pytest.approx(-da, abs=2e-3)
    assert r["next"] == R.NEXT[want] and r["table_cell"] == "defined"
    assert r["two_F_bar"] == 0.2415 and r["tier"].startswith("T1-class open loop") and "paired episode-cluster bootstrap, B=300" in r["estimator"]
    assert r["n_windows"] == 72 and r["n_episodes"] == 12 and r["schema"] == "v8f-flat-read-v1"
    assert r["VERDICT"] in R.OUTCOMES
    if want == "FLAT LEANING":                                                      # Amendment 1: one seed F, the other M -- never FLAT BETTER (which needs both seeds)
        assert r["VERDICT"] != "FLAT BETTER" and "reported to the PI as v8F not validated against flat" in r["next"] and "FLAT BETTER's committed arm" in r["next"]
    assert want in capsys.readouterr().out, "the reading is printed"


@needs_pkg
def test_reader_the_bar_is_the_committed_2F_not_the_measured_F(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "big", d_ade=0.5, gap_b=2.0)                                      # measured 2F ~ 1.2 m: a 0.5 m effect clears the COMMITTED 0.2415, not the measured bar
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    assert r["VERDICT"] == "HIERARCHY BEATS FLAT" and r["two_F_bar"] == 0.2415 and r["two_F_measured"] == pytest.approx(2 * r["F"]) and r["two_F_measured"] > 1.0
    assert abs(r["rows"]["s0"]["primary"]["point"]) < r["two_F_measured"], "the point is below the measured 2F: only the committed bar lets it through"
    g = fw(tmp_path / "small", d_ade=0.20, gap_b=0.02)                                  # measured 2F ~ 0.04 m: a 0.20 m effect clears it, and does NOT clear the committed 0.2415
    rc, r = _go(g)
    assert rc == 0, capsys.readouterr().err
    assert r["VERDICT"] == "NOT ESTABLISHED" and r["two_F_measured"] < 0.1 and r["seed_states"] == {"s0": "MID", "s1": "MID"}
    assert r["two_F_measured"] < abs(r["rows"]["s0"]["primary"]["point"]) < R.BAR_2F


@needs_pkg
def test_reader_a_point_beyond_2F_that_the_episodes_do_not_support_is_a_miss(tmp_path, no_known, fw, capsys):
    """FLAT's seed-0 plan is worse by 3 m on ONE episode only (6 of 72 windows): the pooled point clears -2F, the episode-cluster interval touches 0: MID, not LO."""
    f = fw(tmp_path / "w", d_ade=0.0)
    rows = np.asarray(f["w"].eid == 3)
    _shift_plan(f, 0, 6.0, rows)
    _shift_plan(f, 1, 6.0, rows)
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    p = r["rows"]["s0"]["primary"]
    assert p["point"] <= -R.BAR_2F and not (p["separated"] and p["hi"] < 0), p
    assert r["seed_states"] == {"s0": "MID", "s1": "MID"} and r["VERDICT"] == "NOT ESTABLISHED"


@needs_pkg
def test_reader_flat_against_itself_reads_exactly_zero_on_every_contrast_and_establishes_nothing(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "same", same_as_b=True, d_ade=0.0)
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    assert r["VERDICT"] == "NOT ESTABLISHED" and r["seed_states"] == {"s0": "MID", "s1": "MID"}
    for s in (0, 1):
        row = r["rows"][f"s{s}"]
        assert row["primary"]["point"] == 0.0 and row["primary"]["lo"] == 0.0 and row["primary"]["hi"] == 0.0 and row["primary"]["separated"] is False
    idc = r["identity_control"]
    assert idc, "the identity control (B against itself) is reported"


# ---- the identity read and what is reported beside the verdict ---------------------------------------------------------------------------------
@needs_pkg
def test_flat_identity_is_exactly_zero_per_window_on_an_oracle_that_changed_the_code(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "w")
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    for s in (0, 1):
        fid = r["flat_identity"][f"s{s}"]
        assert fid["oracle_minus_own"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0, "point": 0.0}
        assert fid["max_abs_per_window_dade6"] == 0.0 and fid["max_abs_dpred_oracle"] == 0.0 and fid["max_abs_dpred_all_latcons"] == 0.0 and fid["frac_code_changed"] == 0.62
        z, o = _npz(_xe(f, s, ".npz")), _npz(_xe(f, s, "-tacorc.npz"))
        assert np.array_equal(z["ade6"], o["ade6"]) and np.array_equal(z["pred"], o["pred"]), "the world holds the identity the reader requires"


@needs_pkg
def test_reader_reports_the_families_parameters_pipeline_and_present_slots_beside_the_verdict(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "w")
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    assert r["strategic_family"].startswith("NOT PAUSED") and "ESTIMATED" not in r["strategic_family"]
    for s in (0, 1):
        b = r["rows"][f"s{s}"]["beside"]
        assert {"ade6", "oracle_ade6", "present_slot", "eval_json", "distance_keeping_k10", "parameters", "built", "pipeline", "gt_source", "families"} <= set(b)
        pb, px = b["parameters"]["n_params_trainable"]["B"], b["parameters"]["n_params_trainable"]["X"]
        assert pb == px == 7214495, "the lever adds no parameter: FLAT is capacity-matched to the banked M2b"
        assert b["parameters"]["groups"]["B"] == b["parameters"]["groups"]["X"]
        assert b["built"]["o_interface"] == "cut" and b["built"]["b_hmax"] == 16
        assert b["pipeline"]["X"]["steps"] == b["pipeline"]["B"]["steps"] and b["pipeline"]["X"]["n_windows"] == b["pipeline"]["B"]["n_windows"]
        assert b["eval_json"]["B"]["tactical"] and b["eval_json"]["X"]["tactical"] and b["eval_json"]["B"]["strategic"] and b["eval_json"]["X"]["strategic"], "the decision families are read on both arms"
        assert b["distance_keeping_k10"]["pair_b_minus_x"], "the K = 10 distance-keeping family beside the verdict, B - FLAT"
        assert b["present_slot"]["note"].startswith("T is trained in both arms")
    assert set(r["train_log"]) == {"0", "1"}, "one train-log check per seed"
    assert r["checkpoint_sha256"][f"{XT}_s0"] == f["sha"][f"{XT}_s0_trained"]


# ---- the strata (descriptive; no bar) -----------------------------------------------------------------------------------------------------------
@needs_pkg
def test_strata_numbers_are_the_independent_recomputation_and_tile_the_windows(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "w", d_ade=(0.7, 0.5))
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    st, z = r["strata"], _npz(f["strata"])
    assert st["definition"].startswith("Delta = ADE6(B) - ADE6(FLAT) within the stratum") and "no bar" in st["definition"] and st["release_md5"] == R.KR.RELEASE_MD5
    assert set(st["strata"]) == set(R.KR.STRATA) == {"JUNCTION", "CURVE", "NO-TURN"}
    assert sum(st["strata"][nm]["n_windows"] for nm in R.KR.STRATA) == 72, "the three strata tile the windows"
    eid = f["w"].eid
    for i, nm in enumerate(R.KR.STRATA):
        m = z["stratum"] == i
        blk = st["strata"][nm]
        assert blk["n_windows"] == int(m.sum()) and blk["n_episodes"] == int(np.unique(eid[m]).size)
        for s in (0, 1):
            b, x = _npz(f["eval"] / f"eval_M2b_s{s}.npz")["ade6"].astype(np.float64)[m], _npz(_xe(f, s, ".npz"))["ade6"].astype(np.float64)[m]
            cell = blk["seeds"][f"s{s}"]
            if blk["n_episodes"] < 10:
                assert set(cell) == {"point", "interval"} and cell["interval"].startswith("NONE: fewer than 10 episodes") and "delta" not in cell
                assert cell["point"] == pytest.approx(float(b.mean() - x.mean()), abs=1e-9)
            else:
                assert cell["point"] == pytest.approx(float(b.mean() - x.mean()), abs=1e-9) and cell["lo"] <= cell["point"] <= cell["hi"] and set(cell) == {"delta", "lo", "hi", "point", "separated"}
    # JUNCTION is turn_is_turn on a window WITH a v9 row: episodes 0..10 minus episode 0, whose window 3 is one of the five oracle-fallback windows (no row) -> 10 episodes: exactly the floor
    assert [st["strata"][nm]["n_episodes"] for nm in R.KR.STRATA] == [10, 4, 12]
    assert st["strata"]["JUNCTION"]["seeds"]["s0"].keys() == {"delta", "lo", "hi", "point", "separated"}, "10 episodes is the floor: an interval is reported"


@needs_pkg
def test_strata_never_change_the_outcome_permuting_the_labels_leaves_the_decision_identical(tmp_path, no_known, fw, capsys):
    """The stratum read is BESIDE: permute which windows are JUNCTION / CURVE (among the windows that have a v9 row) and the whole deciding read is identical key for key; only `strata` moves."""
    f = fw(tmp_path / "w", d_ade=(0.7, 0.3))
    rc, base = _go(f)
    assert rc == 0, capsys.readouterr().err
    wi = np.tile(np.arange(f["w"].n_win), f["w"].n_ep)
    cu0, tit0 = ((f["w"].eid >= 8) & (wi == 4)).astype(np.int8), ((f["w"].eid <= 10) & (wi == 3)).astype(np.int8)
    moved = 0
    for seed in (1, 2, 3):
        idx = np.where(~f["fb"])[0]
        perm = np.random.default_rng(seed).permutation(idx)
        cu, tit = cu0.copy(), tit0.copy()
        cu[idx], tit[idx] = cu0[perm], tit0[perm]
        _write_strata(f["strata"], f["w"], f["fb"], cu=cu, tit=tit)
        rc, r = _go(f)
        assert rc == 0, capsys.readouterr().err
        for key in base:
            if key != "strata":
                assert json.dumps(r[key], sort_keys=True) == json.dumps(base[key], sort_keys=True), f"the stratum labels moved `{key}`"
        moved += int(json.dumps(r["strata"], sort_keys=True) != json.dumps(base["strata"], sort_keys=True))
    assert moved >= 2, "the permutation did move the stratum read (otherwise the test pins nothing)"


@needs_pkg
def test_a_strata_file_that_is_not_the_eval_grids_is_refused_with_rc2(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "w")
    w, fb = f["w"], f["fb"]
    good = _npz(f["strata"])
    eid_bad = w.eid.copy()
    eid_bad[[0, -1]] = eid_bad[[-1, 0]]
    s_wrong = good["stratum"].copy()
    s_wrong[np.where(good["stratum"] == 2)[0][0]] = 1                                      # a NO-TURN window marked CURVE with no curve label behind it
    cases = (("eid", dict(eid=eid_bad), "episode index differs", fb), ("tick", dict(t=w.t + 1), "tick differs", fb),
             ("labels", dict(stratum=s_wrong), "not JUNCTION (turn_is_turn = 1) / CURVE (curve = 1) / NO-TURN", fb), ("code", dict(stratum=np.full(w.n, 7, np.int8)), "outside {0 JUNCTION", fb),
             ("fallback", {}, "has != ~tac_oracle_fallback", np.zeros(w.n, bool)),                     # a file that is consistent with ITSELF but whose v9-row presence is not the evaluator's
             ("short", dict(stratum=good["stratum"][:-1]), "the strata file holds 71 windows", fb))
    for name, over, frag, fb_file in cases:
        _write_strata(f["strata"], w, fb_file, **over)
        capsys.readouterr()
        rc = R.main(f["args"])
        err = capsys.readouterr().err
        assert rc == 2 and frag in err and "FLAT READ REFUSED" in err, f"{name}: rc {rc}; {err[-400:]}"
        assert not f["out"].exists()
    _write_strata(f["strata"], w, fb, md5="0" * 32)
    assert R.main(f["args"]) == 2 and "was built from release" in capsys.readouterr().err
    Path(f["strata"]).unlink()
    assert R.main(f["args"]) == 2 and "does not exist" in capsys.readouterr().err


# ---- the refusals -------------------------------------------------------------------------------------------------------------------------------
def _f_missing_family(side, fam):
    def go(f):
        p = (_xe(f, 0, ".json") if side == "X" else f["eval"] / "eval_M2b_s0.json")
        _ej(p, lambda j: j["raw"].pop(fam))
    return go


def _sha_edit(key, val):
    return lambda f: _ej(f["ck"], lambda j: j.update({key: val}))


def _other_gt(z):
    n = z["tac_code"].shape[0]
    z["tac_code"] = np.where(np.arange(n)[:, None] % 7 == 3, z["tac_code"] + 1.0, z["tac_code"])


def _sc_edit(fn):
    return lambda f: _ej(_xr(f) / "selfcheck.json", fn)


def _probe_edit(fn):
    return _sc_edit(lambda j: fn(j["o_interface_probe"]))


def _pert(name, **kv):
    return _probe_edit(lambda p: p["perturbations"][name].update(kv))


def _oracle_edit(fn, suf="-tacorc", s=0):
    return lambda f: _npz_edit(_xe(f, s, suf + ".npz"), fn)


def _ade_off(z):
    z["ade6"] = np.asarray(z["ade6"]).copy()
    z["ade6"][3] += 1e-9


def _pred_off(z):
    z["pred"] = np.asarray(z["pred"]).copy()
    z["pred"][5, 20, 0] += np.float32(0.01)


MUT = {                                # name -> (mutator(f), a message fragment)
    "x summary missing": (lambda f: (_xr(f) / "summary.json").unlink(), "is missing"),
    "x checkpoint missing": (lambda f: (_xr(f) / "ckpt_last.pt").unlink(), "is missing"),
    "x selfcheck missing": (lambda f: (_xr(f) / "selfcheck.json").unlink(), "is missing"),
    "x argv missing": (lambda f: (_xr(f) / "argv.json").unlink(), "is missing"),
    "x not done": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(done=False)), "is not done"),
    "x summary says another arm": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(arm="M2")), "summary.json says M2 seed 0"),
    "x n_clips": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(n_clips=4000)), "n_clips 4000"),
    "x withheld nothing": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(excluded_sha256="0" * 64)), "did not withhold"),
    "x trained on another clip list": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(train_sha256="u" * 64)), "did not train on"),
    "x trained fewer steps": (lambda f: _fewer_steps(f, 9000), "trained for 9000 steps"),
    "x trained on another window set": (lambda f: _ej(_xr(f) / "summary.json", lambda j: j.update(n_windows=279494)), "trained on 279494 windows"),
    # ---- the launch-gate PASS token
    "x launch gate not PASS": (_sc_edit(lambda j: j.update(declared_vs_built="FAIL")), "declared_vs_built is not PASS"),
    "x launch gate silent": (_sc_edit(lambda j: j.pop("declared_vs_built")), "declared_vs_built is not PASS"),
    "x G-LIVE not pass": (_sc_edit(lambda j: j["live"].update(**{"pass": False})), "G-LIVE is not a pass"),
    "x frozen-prior control not exactly zero": (_sc_edit(lambda j: j["live"].update(control_prior_grad_sum=1e-9)), "G-LIVE is not a pass"),
    "x e_code.weight not dead (the cut left a wire)": (_sc_edit(lambda j: j["live"].update(dead_params=[], declared_dead=[])), "dead-weight set"),
    "x another weight is dead too": (_sc_edit(lambda j: j["live"].update(dead_params=["e_code.weight", "head_o.2.weight"])), "dead-weight set"),
    "x no o_interface probe": (_sc_edit(lambda j: j.pop("o_interface_probe")), "no passing o_interface probe"),
    "x probe did not pass": (_probe_edit(lambda p: p.update(**{"pass": False})), "no passing o_interface probe"),
    "x probe was for full": (_probe_edit(lambda p: p.update(o_interface="full")), "was run for 'full'"),
    "x probe lost a perturbation": (_probe_edit(lambda p: p["perturbations"].pop("donor_code")), "perturbed"),
    "x probe saw traj60 move under a donor code": (_pert("donor_code", max_abs_dtraj60=1e-9), "traj60 moved"),
    "x probe saw traj60 move under shifted checkpoints": (_pert("shifted_checkpoints", max_abs_dtraj60=1e-9), "traj60 moved"),
    "x probe saw O read the code": (_pert("random_code", o_read_code_abs_max=0.1), "was not handed the constants"),
    "x probe saw O read a checkpoint": (_pert("gt_code", o_read_ck_minus_nominal_abs_max=0.5), "was not handed the constants"),
    "x probe saw O read the ok flag": (_pert("shifted_checkpoints", o_read_ck_ok_abs_max=1.0), "was not handed the constants"),
    "x probe did not see T read S": (_pert("shifted_checkpoints", t_reads_s_max_abs_dgoal_logits=0.0), "T must still read S"),
    "x selfcheck from another run": (_sc_edit(lambda j: j.update(argv_sha="zzz")), "is not this run's"),
    # ---- the argv: B's, plus --o-interface cut, and nothing else
    "x argv has no --o-interface": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: [x for x in a if x not in ("--o-interface", "cut")]), "named 0 times"),
    "x argv says --o-interface full": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: _set_flag(a, "--o-interface", "full")), "argv --o-interface full"),
    "x argv names --o-interface twice": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--o-interface", "cut"]), "named 2 times"),
    "b argv names --o-interface": (lambda f: _argv_edit(_br(f) / "argv.json", lambda a: a + ["--o-interface", "cut"]), "predates the flag"),
    "x argv has a second lever": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--w-wm", "0.2"]), "not one lever"),
    "x argv moved hist": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--hist", "10"]), "not one lever"),
    "x argv moved the code mixing": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--code-mix", "kind"]), "not one lever"),
    "x argv moved the p_gt schedule": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: a + ["--p-gt-start", "0.5"]), "not one lever"),
    "x argv moved the grid": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: _set_flag(a, "--grid", "pool8x20")), "not one lever"),
    "x argv lost a flag": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: [x for x in a if x not in ("--kin-mode", "yaw")]), "not one lever"),
    "x argv is another arm": (lambda f: _argv_edit(_xr(f) / "argv.json", lambda a: _set_flag(a, "--arm", "M2")), "argv --arm"),
    # ---- the BUILT levers
    "x built o_interface full (summary)": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(o_interface="full")), "built o_interface 'full'"),
    "x built o_interface full (selfcheck)": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.update(o_interface="full")), "built o_interface 'full'"),
    "x records no built o_interface (summary)": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.pop("o_interface")), "record no built o_interface"),
    "x records no built o_interface (selfcheck)": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.pop("o_interface")), "record no built o_interface"),
    "x built code_mix kind (summary)": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(code_mix="kind")), "built code_mix 'kind'"),
    "x built code_mix kind (selfcheck)": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.update(code_mix="kind")), "built code_mix 'kind'"),
    "x built another p_gt": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(p_gt_start=0.7)), "was built with p_gt_start"),
    "x records no built p_gt": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.pop("p_gt_end")), "record no built p_gt_end"),
    "x built hmax 10": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(hmax=10)), "built hmax 10"),
    "x built another weight": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l["w"].update(w_pen=0.9)), "w.w_pen"),
    "x built another w_cons": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l["w"].update(w_cons=1.0)), "w.w_cons"),
    "x built another mask": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l.update(wm="a")), "built levers"),
    "x has one parameter too many": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l.update(n_params_trainable=l["n_params_trainable"] + 1)), "parameter counts are not B's"),
    "x grew a group": (lambda f: _lev(_xr(f) / "selfcheck.json", lambda l: l["params"].update(T=l["params"]["T"] + 5)), "group T"),
    "x lost a group": (lambda f: _lev(_xr(f) / "summary.json", lambda l: l["params"].pop("WM")), "group WM"),
    "B built the cut": (lambda f: _lev(_br(f) / "summary.json", lambda l: l.update(o_interface="cut")), "is not the full-interface run"),
    # ---- the checkpoint, as built
    "x checkpoint built full": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(o_interface="full")), "BUILT with o_interface 'full'"),
    "x checkpoint has no o_interface": (lambda f: _ck_edit(f, lambda o: o["cfg"].pop("o_interface")), "BUILT with o_interface None"),
    "x checkpoint built code_mix kind": (lambda f: _ck_edit(f, lambda o: o["cfg"].update(code_mix="kind")), "BUILT with code_mix 'kind'"),
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
    # ---- the checkpoint sha256 values
    "ckpt-sha json missing": (lambda f: f["ck"].unlink(), "does not exist"),
    "B's checkpoint is not the banked file": (_sha_edit("M2b_s0", "0" * 64), "is not the banked checkpoint"),
    "B's seed-1 checkpoint is not the banked file": (_sha_edit("M2b_s1", "0" * 64), "is not the banked checkpoint"),
    "X's checkpoint changed during the evals": (_sha_edit(f"{XT}_s0_after_evals", "1" * 64), "not one value"),
    "X's checkpoint is not the one pulled": (lambda f: _ej(f["ck"], lambda j: j.update({f"{XT}_s0_trained": "2" * 64, f"{XT}_s0_after_evals": "2" * 64})), "not the scored checkpoint"),
    "X's trained sha missing": (lambda f: _ej(f["ck"], lambda j: j.pop(f"{XT}_s0_trained")), "not one value"),
    # ---- the evals
    "x eval scored B's checkpoint": (lambda f: _ej(_xe(f, 0, ".json"), lambda j: j.update(ckpt="D:\\runs\\M2b_s0\\ckpt_last.pt")), "scored"),
    "x oracle eval scored B's checkpoint": (lambda f: _ej(_xe(f, 1, "-tacorc.json"), lambda j: j.update(ckpt="D:\\runs\\M2b_s1\\ckpt_last.pt")), "scored"),
    "x oracle eval scored no checkpoint": (lambda f: _ej(_xe(f, 0, "-tacorc.json"), lambda j: j.pop("ckpt") and None), "scored None"),
    "x eval built the full interface": (lambda f: _ej(_xe(f, 0, ".json"), lambda j: j.update(o_interface="full")), "built o_interface 'full'"),
    "x oracle eval built the full interface": (lambda f: _ej(_xe(f, 1, "-tacorc.json"), lambda j: j.update(o_interface="full")), "built o_interface 'full'"),
    "x eval does not say what it built": (lambda f: _ej(_xe(f, 0, "-tacabs.json"), lambda j: j.pop("o_interface")), "built o_interface None"),
    "B's eval built the cut": (lambda f: _ej(f["eval"] / "eval_M2b_s0.json", lambda j: j.update(o_interface="cut")), "B is the full-interface run"),
    "x oracle eval is not an oracle run": (lambda f: _ej(_xe(f, 0, "-tacorc.json"), lambda j: j["tac_manipulation"].update(mode="absmask")), "is not an oracle run"),
    "x eval is on another eval grid": (lambda f: _npz_edit(_xe(f, 0, ".npz"), _shift_eid), "eid"),
    "x oracle eval is on another eval grid": (lambda f: _npz_edit(_xe(f, 0, "-tacorc.npz"), _shift_eid), "not on the same eval grid"),
    "x evals (all) are on another eval grid than B's": (lambda f: [_npz_edit(_xe(f, s, suf), _shift_eid) for s in (0, 1) for suf in (".npz", "-tacabs.npz", "-tacorc.npz", "-tacorc-all-latcons.npz")],
                                                    "not on the same eval grid as M2b's"),
    "x oracle eval has other GT codes": (lambda f: _npz_edit(_xe(f, 0, "-tacorc.npz"), _other_gt), "the GT constraint block differs"),
    "x eval missing": (lambda f: _xe(f, 0, "-tacorc-all-latcons.npz").unlink(), "is missing"),
    "x absmask eval missing": (lambda f: _xe(f, 1, "-tacabs.npz").unlink(), "is missing"),
    # ---- the identity read: FLAT's GT-code oracle minus its own ADE6 is EXACTLY 0
    "identity: the oracle ADE6 differs in one window by 1e-9": (_oracle_edit(_ade_off), "IDENTITY VOID"),
    "identity: the oracle ADE6 differs at seed 1": (_oracle_edit(_ade_off, s=1), "IDENTITY VOID"),
    "identity: the oracle plan differs in one point": (_oracle_edit(_pred_off), "IDENTITY VOID"),
    "identity: the all-latcons oracle plan differs": (_oracle_edit(_pred_off, "-tacorc-all-latcons"), "IDENTITY VOID"),
    "identity: the oracle changed no code": (lambda f: _ej(_xe(f, 0, "-tacorc.json"), lambda j: j["tac_manipulation"].update(frac_code_changed=0.0)), "would say nothing"),
    "identity: the oracle json does not say whether it changed the code": (lambda f: _ej(_xe(f, 1, "-tacorc.json"), lambda j: j["tac_manipulation"].pop("frac_code_changed")), "would say nothing"),
    # ---- the families
    "tactical family missing (X)": (_f_missing_family("X", "tactical"), "a family is missing"),
    "strategic family missing (X)": (_f_missing_family("X", "strategic"), "a family is missing"),
    "tactical family missing (B)": (_f_missing_family("B", "tactical"), "a family is missing"),
    # ---- the grid
    "a grid with other window counts": (lambda f: f["args"].extend(["--n-windows", "2399"]), "committed eval-141 grid"),
    "a grid with other episode counts": (lambda f: f["args"].extend(["--n-episodes", "141"]), "committed eval-141 grid"),
}


@needs_pkg
@pytest.mark.parametrize("name", list(MUT))
def test_reader_refuses_rc2_with_the_reason(tmp_path, no_known, fw, capsys, name):
    mut, frag = MUT[name]
    f = fw(tmp_path / "w")
    mut(f)
    capsys.readouterr()
    rc = R.main(f["args"])
    err = capsys.readouterr().err
    assert rc == 2 and frag in err and "FLAT READ REFUSED" in err, f"{name}: rc {rc}; stderr: {err[-600:]}"
    assert not f["out"].exists(), "a refused read must leave no result file for the next step to find"


@needs_pkg
def test_reader_needs_both_seeds_and_refuses_a_one_seed_read_before_computing_anything(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "one", seeds=(0,))
    capsys.readouterr()
    assert R.main(f["args"]) == 2
    err = capsys.readouterr().err
    assert "M2b_s1_flat is missing" in err and "BOTH seeds" in err and "H-ESTIM-SEED-1" in err and not f["out"].exists()


@needs_pkg
def test_reader_missing_inputs_are_rc2_not_a_traceback(tmp_path, no_known, capsys):
    assert R.main(["--eval-dir", str(tmp_path), "--runs-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 2
    assert "is missing" in capsys.readouterr().err
    assert R.main([]) == 2 and "--eval-dir is required" in capsys.readouterr().err


@needs_pkg
def test_reader_requires_the_ckpt_sha_json_for_a_read(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "w")
    k = f["args"].index("--ckpt-sha")
    args = f["args"][:k] + f["args"][k + 2:]
    capsys.readouterr()
    assert R.main(args) == 2 and "--ckpt-sha is required" in capsys.readouterr().err


@needs_pkg
def test_reader_turns_an_inherited_refusal_into_rc2_not_a_traceback(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "w")

    def odd(z):
        z["frame"] = np.asarray(z["frame"]).copy()
        z["frame"][0] += 1
    _npz_edit(f["d"] / "gt.npz", odd)
    capsys.readouterr()
    assert R.main(f["args"]) == 2
    err = capsys.readouterr().err
    assert "FLAT READ REFUSED" in err and "no row in the lead block" in err


@needs_pkg
def test_reader_breaks_loudly_when_the_paired_instrument_is_broken(tmp_path, no_known, fw, capsys, monkeypatch):
    f = fw(tmp_path / "w")
    import taniteval.ci as ci
    orig = ci.paired_episode_cluster_bootstrap

    def biased(a, b, *r, **k):
        o = orig(a, b, *r, **k)
        if np.array_equal(np.asarray(a), np.asarray(b)):
            o = dict(o, delta=o["delta"] + 1e-9)
        return o
    monkeypatch.setattr(ci, "paired_episode_cluster_bootstrap", biased)
    capsys.readouterr()
    assert R.main(f["args"]) == 2
    assert "IDENTITY CONTROL FAILED" in capsys.readouterr().err


@needs_pkg
def test_known_values_are_checked_and_refuse_a_world_that_does_not_hold_them(tmp_path, fw, capsys):
    f = fw(tmp_path / "w")
    capsys.readouterr()
    assert R.main(f["args"]) == 2
    assert "does not reproduce the PREREG's known values" in capsys.readouterr().err


# ---- the pre-launch check of B's side ---------------------------------------------------------------------------------------------------------------
@needs_pkg
def test_check_inputs_reads_b_alone_validates_the_strata_and_the_identity_control_is_exactly_zero(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "c", seeds=())                                                # no X run exists yet: this is the check that runs BEFORE training
    assert not list(f["eval"].glob(f"eval_{XT}*")) and not list(f["runs"].glob("*_flat"))
    assert R.main(["--check-inputs", *f["args"]]) == 0
    out = capsys.readouterr().out
    assert "FLAT INPUTS OK" in out and "identity control exactly +0 [0, 0]" in out and "expects 4 evals per seed" in out
    j = json.loads(f["out"].read_text(encoding="utf-8"))
    assert j["base"] == "M2b" and j["F"] > 0 and j["two_F_bar"] == 0.2415 and j["expected_x_suffixes"] == list(R.X_SUFFIXES) and len(j["expected_x_suffixes"]) == 4
    assert j["n_windows"] == 72 and j["n_episodes"] == 12 and sum(v["n_windows"] for v in j["strata"].values()) == 72
    assert j["checkpoint_sha256"] == {"M2b_s0": R.BANKED_CKPT_SHA["M2b_s0"], "M2b_s1": R.BANKED_CKPT_SHA["M2b_s1"]}


@needs_pkg
def test_check_inputs_refuses_a_b_side_that_is_not_the_preregs(tmp_path, no_known, fw, capsys):
    f = fw(tmp_path / "c", seeds=())
    _sha_edit("M2b_s1", "0" * 64)(f)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "is not the banked checkpoint" in capsys.readouterr().err
    g = fw(tmp_path / "d", seeds=())
    k = g["args"].index("--strata")                                                  # the default strata file is the repo's REAL eval-141 file: not this synthetic grid
    args = g["args"][:k] + g["args"][k + 2:]
    capsys.readouterr()
    assert R.main(["--check-inputs", *args]) == 2
    assert "the strata file holds" in capsys.readouterr().err
    h = fw(tmp_path / "e", seeds=())
    _write_strata(h["strata"], h["w"], h["fb"], md5="0" * 32)
    assert R.main(["--check-inputs", *h["args"]]) == 2 and "was built from release" in capsys.readouterr().err


@needs_pkg
def test_check_inputs_breaks_loudly_when_the_paired_instrument_is_broken(tmp_path, no_known, fw, capsys, monkeypatch):
    f = fw(tmp_path / "c", seeds=())
    import taniteval.ci as ci
    orig = ci.paired_episode_cluster_bootstrap

    def biased(a, b, *r, **k):
        o = orig(a, b, *r, **k)
        return dict(o, delta=o["delta"] + 1e-9) if np.array_equal(np.asarray(a), np.asarray(b)) else o
    monkeypatch.setattr(ci, "paired_episode_cluster_bootstrap", biased)
    capsys.readouterr()
    assert R.main(["--check-inputs", *f["args"]]) == 2
    assert "IDENTITY CONTROL FAILED" in capsys.readouterr().err


# ---- the reader's constants against the model, the trainer, the PREREG, the queue and the chain -----------------------------------------------------
@needs_pkg
def test_the_readers_gate_token_accepts_the_trainers_own_cut_record_and_refuses_its_full_record(gate_rec):
    ok = {"declared_vs_built": "PASS", "argv_sha": "a", **gate_rec["cut"]}
    bad = []
    R.check_gate_token("t", {"argv_sha": "a"}, ok, bad)
    assert bad == [], bad
    bad = []
    R.check_gate_token("t", {"argv_sha": "a"}, {"declared_vs_built": "PASS", "argv_sha": "a", **gate_rec["full"]}, bad)
    assert len(bad) >= 3, "a full run's record fails the dead-set, the probe-mode and the probe-values checks"
    assert any("dead-weight set" in b for b in bad) and any("was run for 'full'" in b for b in bad) and any("traj60 moved" in b for b in bad)


@needs_pkg
@pytest.mark.skipif(not PREREG.exists(), reason="PREREG_V8F_FLAT.md is not in this checkout")
def test_the_committed_outcome_table_and_constants_are_verbatim_the_preregs():
    text = PREREG.read_text(encoding="utf-8")
    plain = text.replace("**", "")
    for k, v in R.NEXT_TEXT.items():
        assert v in plain, f"{k}: {v!r} is not in PREREG_V8F_FLAT.md (bold markers removed)"          # FLAT LEANING's cell carries **bold** markup in the file
        assert v in R.NEXT[k] and R.MEANING[k] in plain, k
    assert tuple(R.OUTCOMES) == ("HIERARCHY BEATS FLAT", "NOT ESTABLISHED", "FLAT BETTER", "SEEDS DISAGREE", "FLAT LEANING")
    assert "## Amendment 1 (2026-10-25, written BEFORE any FLAT data" in text and "| **FLAT LEANING** | one seed F, the other M |" in text
    assert "**H** (≤ −2F, separated), **F** (≥ +2F, separated) or **M** (anything else)" in text and "⛔ It is never reported as FLAT BETTER, which needs both seeds" in text
    for vd in R.OUTCOMES:
        assert sum(1 for ln in text.splitlines() if ln.startswith(f"| **{vd}** |")) == 1, vd
    assert "| **HIERARCHY BEATS FLAT** | Δ_0 ≤ −2F and Δ_1 ≤ −2F, both separated |" in text
    assert "| **NOT ESTABLISHED** | either seed misses −2F or is not separated, and neither seed reads ≥ +2F |" in text
    assert "| **FLAT BETTER** | Δ_0 ≥ +2F and Δ_1 ≥ +2F, both separated |" in text
    assert "| **SEEDS DISAGREE** | one seed ≤ −2F separated, the other ≥ +2F separated |" in text
    assert "B = 2,000" in text and R.B == 2000 and "0.1207 m, 2F = 0.2415 m" in text and R.BAR_2F == 0.2415
    assert "Δ_s = M2b_s − FLAT_s (negative = the hierarchy is better)" in text and "**≤ −2F and separated, at BOTH seeds**" in text
    assert "must read **exactly 0** at both seeds" in text
    assert "NO ROW" not in text, "the PREREG has no outcome called NO ROW: Amendment 1 closed the gap with FLAT LEANING"
    assert R.EVAL_WINDOWS == 2399 and R.EVAL_EPISODES == 141 and "2,399 windows" in text


@needs_pkg
def test_reader_constants_match_the_model_the_trainer_the_queue_and_the_chain():
    assert R.ARM_X == XT and R.X_SUFFIXES == ("", "-tacabs", "-tacorc", "-tacorc-all-latcons") and R.ARM_B == "M2b"
    assert R.LEVER_FLAG == "--o-interface" and R.LEVER_VAL == "cut" and R.x_dir(1) == "M2b_s1_flat" and R.SUF == "_flat"
    assert R.LEVER_VAL in M.O_INTERFACE_MODES and M.V8FMiniConfig().o_interface == "full"
    assert R.B_P_GT == (M.V8FMiniConfig().p_gt_start, M.V8FMiniConfig().p_gt_end) == (1.0, 0.2)
    assert R.B_GRID == "pool4x10" and R.N_TOK_B == 40
    assert set(R.NEXT) == set(R.OUTCOMES) and R.verdict(("HI", "MID"))[0] == "FLAT LEANING"
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"CKPT_SHA=\((\w+) (\w+)\)", c)
    assert (m.group(1), m.group(2)) == (R.BANKED_CKPT_SHA["M2b_s0"], R.BANKED_CKPT_SHA["M2b_s1"]), "the chain and the reader assert the same two banked checkpoints"
    man = PKG / "raw" / "full" / "MANIFEST.md"
    if man.exists():
        t = man.read_text(encoding="utf-8")
        assert R.BANKED_CKPT_SHA["M2b_s0"] in t and R.BANKED_CKPT_SHA["M2b_s1"] in t, "raw/full/MANIFEST.md carries the banked sha256 values"
    q = QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")
    blk = q.split("\n:evalflat\n", 1)[1].split("\n:evalflat_1\n", 1)[0]
    assert re.findall(r"--tag %TG%(\S*)", blk) == list(R.X_SUFFIXES), "the queue's four tags are the reader's four suffixes"
    assert 'set "OUT=%RUNSM%\\M2b_s%~2_flat"' in blk and 'set "TG=M2b-flat_s%~2"' in blk


# =============================================================================================================================================
# (J) THE QUEUE (code/v8f_mini_box2_queue.cmd) AND THE CHAIN (code/box2_ops/chain_v8f_flat.sh)
# =============================================================================================================================================
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
FSTAGES = ("gateflat", "trainflat", "evalflat")
FLAT_LABELS = {"gateflat", "trainflat", "evalflat"} | {f"evalflat_{i}" for i in range(1, 5)}
# PREREG_V8F_MIXO's ten labels as they stand at the tip 18c62306 (sha256[:16] of "\n" + the text up to the next label): this edit appended AFTER them and edited none
MIXO_PINS = {"gatemixo": "b2ea5430b698741c", "trainmixo": "419a7db33d684844", "evalmixo": "7acc0975f2aa9ef3", "evalmixo_1": "1cc5381d64b27142", "evalmixo_2": "a70ba66191aed3cf",
             "evalmixo_3": "5d326de83e5d37b1", "evalmixo_4": "0cde88e556e0b0e4", "evalmixo_5": "479e39bc5b10065b", "evalmixo_6": "77271eba1510a820", "evalmixo_7": "901e965d7b940ca4"}
_K = _load("test_v8f_kind_for_flat", HERE_T / "test_v8f_kind.py") if (QUEUE.exists() and (HERE_T / "test_v8f_kind.py").exists()) else None
_H2 = _H if _H is not None else (_load("test_v8f_hist_for_flat_q", HERE_T / "test_v8f_hist.py") if (HERE_T / "test_v8f_hist.py").exists() else None)
_MX = _load("test_v8f_mixo_for_flat", HERE_T / "test_v8f_mixo.py") if (QUEUE.exists() and (HERE_T / "test_v8f_mixo.py").exists()) else None
SUFFIX_TAGS = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")


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
@pytest.mark.skipif(_K is None or _MX is None, reason="test_v8f_kind.py / test_v8f_mixo.py (the pre-existing label pins) are not in this checkout")
def test_queue_flat_is_crlf_every_existing_label_is_byte_identical_and_only_the_seven_new_ones_are_added():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 800 and raw.count(b"\r") == raw.count(b"\r\n") and raw.endswith(b"\r\n"), "every line CRLF, no bare CR"
    q, blocks = _queue_text(), _blocks()
    for label, h in {**_K.PRE_EXISTING, **_MX.KIND_PINS, **MIXO_PINS}.items():
        assert label in blocks, f"the stage {label} was removed"
        assert hashlib.sha256(("\n" + blocks[label]).encode()).hexdigest()[:16] == h, f"the stage {label} was edited"
        assert "o-interface" not in blocks[label] and "_flat" not in blocks[label] and "M2b-flat" not in blocks[label], f"{label} mentions the new stages"
    assert set(blocks) == set(_K.PRE_EXISTING) | set(_MX.KIND_PINS) | set(MIXO_PINS) | FLAT_LABELS and len(FLAT_LABELS) == 7, "the only new labels are the seven of this edit"
    for st in FSTAGES:
        assert q.count(f'if /i "%~1"=="{st}" goto :{st}\n') == 1, "a stage nobody dispatches never runs"
        assert q.count(f"\n:{st}\n") == 1
    target = {"train": "d_train", "eval": "d_eval"}
    for st in _K.OLD_DISPATCH + ("kind", "gatemixo", "trainmixo", "evalmixo"):
        assert f'if /i "%~1"=="{st}" goto :{target.get(st, st)}\n' in q, st
    disp = re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto :(\w+)$', q)
    assert [(a, b) for a, b in disp if b not in set(blocks)] == [], "a dispatch line whose target label does not exist"
    usage = next(x for x in q.split("\n") if x.startswith("echo usage:"))
    assert usage.endswith("^|gatedscale^|traindscale SEED^|evaldscale SEED^|identitylora^|smokelora") and "flat" not in usage, "the usage line is untouched (the earlier pins on it stand)"
    for need in ("gateflat", "trainflat SEED", "evalflat SEED"):
        assert f"rem   v8f_mini_box2_queue.cmd {need}" in q, "the header documents the stage"
    assert q.rstrip().endswith(":evalflat_4\nexit /b 0") and q.count("\n:gateflat\n") == 1
    assert q.split("\n:gateflat\n")[0].endswith("\n:evalmixo_7\nexit /b 0"), "the new stages start right after the last old label, which keeps its text (MIXO's own convention)"


@needs_queue
def test_queue_flat_runs_the_command_it_prints_static():
    """A dry run can only print: the real line is checked statically -- the trainer / eval appear ONCE per command, inside the one string, and the real run executes that string."""
    for label, var, pre in (("gateflat", "GX", '--arm M2b --o-interface cut --out "%RUNSM%\\gate_M2b-flat" '),
                            ("trainflat", "TX", '--arm M2b --seed %~2 --grid pool4x10 --o-interface cut --out "%OUT%" ')):
        code = _code(_block(label))
        assert [x for x in code if "%TRAIN%" in x] == [x for x in code if x.startswith(f'set "{var}=')], label
        assert any(x.startswith(f'set "{var}="%PY%" "%TRAIN%" ' + pre) for x in code), label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code, label
        assert sum(x.startswith(f"%{var}% ") for x in code) == 1, label
        assert sum(x.count("--o-interface") for x in code if x.startswith(f'set "{var}=')) == 1, f"{label}: --o-interface appears once, in the one string"
        assert not any(("pool8x20" in x or "--hist" in x or "--p-gt" in x or "--w-" in x or "--hmax" in x or "--window-start" in x or "--code-mix" in x) for x in code), f"{label}: ONE lever"
    code = _code(_block("evalflat"))
    assert [x for x in code if "%EVAL%" in x] == [x for x in code if re.match(r'set "E[0-3]=', x)], "the eval script appears only inside the four strings"
    assert len([x for x in code if re.match(r'set "E[0-3]=', x)]) == 4
    assert "if defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)" in code
    for k in range(4):
        assert sum(x.strip().startswith(f"%E{k}% >> ") for x in code) == 1, f"E{k} is executed exactly once"
    e = {k: next(x for x in code if x.startswith(f'set "E{k}=')) for k in range(4)}
    assert "--project --families" in e[0] and "--tac" not in e[0]
    assert "--tac absmask" in e[1] and "--tac oracle" in e[2] and "--tac-groups" not in e[2]
    assert "--tac oracle --tac-groups all-latcons --tac-alpha 1" in e[3]
    for k in range(4):
        assert '--ckpt "%OUT%\\ckpt_last.pt"' in e[k] and "%EC% %ER%" in e[k] and "--grid" not in e[k] and "--o-interface" not in e[k], \
            "the eval reads the checkpoint's own cfg; an o_interface flag on an eval would be a path the model never takes"
    # every failure is noticed on its own line (flat `if errorlevel 1 (echo [FAIL] ... & exit /b 1)`, no parenthesised multi-line block: MEASURED to return rc 0 from `cmd /c`)
    for label in FSTAGES:
        body = _block(label)
        assert not any(x.rstrip().endswith("(") for x in body), f"{label}: a multi-line parenthesised block"
        assert sum(1 for x in body if x.startswith("if errorlevel 1 (echo [FAIL]") and x.rstrip().endswith("exit /b 1)")) >= (1 if label != "evalflat" else 4)
    # the four evals chain by goto through the flat labels evalflat_1 .. evalflat_4; the last one ends the stage
    q = _queue_text()
    for k in range(1, 5):
        assert q.count(f"\n:evalflat_{k}\n") == 1 and f"goto :evalflat_{k})" in q
    assert q.rstrip().endswith(":evalflat_4\nexit /b 0")


@needs_queue
@win
def test_queue_flat_stages_dry_one_lever(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    rc, ln = _dry("gateflat", env)
    assert rc == 0 and len(ln) == 1 and "--selfcheck-only" in ln[0] and "gate_M2b-flat" in ln[0], ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm"):t.index("--arm") + 4] == ["--arm", "M2b", "--o-interface", "cut"] and t.count("--o-interface") == 1, ln
    rc, ln = _dry("trainflat 1", env)
    assert rc == 0 and len(ln) == 1, ln
    t = _argv_of(ln[0])
    assert t[t.index("--arm") + 1] == "M2b" and t[t.index("--seed") + 1] == "1" and t[t.index("--o-interface") + 1] == "cut" and t.count("--o-interface") == 1, ln
    assert t[t.index("--grid") + 1] == "pool4x10" and t.count("--grid") == 1 and t[t.index("--out") + 1].endswith("M2b_s1_flat"), ln
    assert "--exclude 540:600" in ln[0] and "--kin-mode yaw" in ln[0] and "--steps-mult 4" in ln[0] and "--batch 64" in ln[0], ln
    for no in ("--w-cons", "--w-tac", "--w-wm", "--hmax", "--steps ", "--p-gt", "--hist", "--window-start", "--code-mix"):
        assert no not in ln[0], f"ONE lever: {no} is not on the line"
    rc, ev = _dry("evalflat 0", env)
    assert rc == 0 and len(ev) == 4, ev
    for k, (suffix, frag) in enumerate((("", "--project --families"), ("-tacabs", "--tac absmask"), ("-tacorc", "--tac oracle"),
                                        ("-tacorc-all-latcons", "--tac oracle --tac-groups all-latcons --tac-alpha 1"))):
        assert f"--tag M2b-flat_s0{suffix} " in ev[k] + " " and frag in ev[k], (k, ev[k])
        assert f"M2b_s0_flat{os.sep}ckpt_last.pt" in ev[k], "every eval reads THIS run's last checkpoint"
        assert f"{os.sep}M2b_s0{os.sep}" not in ev[k], "never B's checkpoint"
    # ONE lever, from the queue's own lines: FLAT's gate / train line is B's line with --o-interface cut (and the output dir) changed and NOTHING else
    drop = {"--o-interface", "--out"}
    b_train = _dry("train M2b 1", env)[1][0]
    assert "--o-interface" not in b_train
    assert _without(b_train, {"--out"}) == _without(_dry("trainflat 1", env)[1][0], drop), "FLAT's train line = B's + --o-interface cut"
    b_gate = _dry("gatearm M2b", env)[1][0]
    assert _without(b_gate, {"--out"}) == _without(_dry("gateflat", env)[1][0], drop), "FLAT's gate line = B's gate line + --o-interface cut"


@needs_queue
@win
def test_queue_flat_refuses_missing_arguments(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("trainflat", "evalflat"):
        rc, ln = _dry(args, env)
        assert rc == 2 and ln == [], (args, rc, ln)


@needs_queue
@win
def test_queue_flat_leaves_every_other_stage_untouched(tmp_path):
    """The edit only ADDS stages: the neighbouring arms' lines read as they did, and none of them mentions the lever."""
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    for args in ("train M2b 0", "trainmixo 0", "evalmixo 0", "gatemixo", "trainpgt 0", "trainhist 0", "traindscale 0", "gatearm M2b", "gatepgt", "gategrid M2b", "kind", "evalhist 0"):
        rc, ln = _dry(args, env)
        assert rc == 0 and len(ln) >= 1 and all("--o-interface" not in x and "M2b-flat" not in x and "_s0_flat" not in x for x in ln), (args, ln)


@needs_queue
@win
@pytest.mark.skipif(_H2 is None, reason="test_v8f_hist.py (the fake repo) is not in this checkout")
def test_queue_flat_real_lines_execute_exactly_the_printed_commands(tmp_path):
    repo, rec = _H2._fake_tree(tmp_path)
    env = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    env.pop("V8F_DRYRUN", None)

    def real(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=180, encoding="utf-8")
        assert r.returncode == 0, (args, r.stdout, r.stderr)
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []

    def printed(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, V8F_DRYRUN="1"), capture_output=True, text=True, timeout=60, encoding="utf-8")
        return [_argv_of(x)[2:] for x in r.stdout.splitlines() if ".py" in x]       # drop python.exe and the script: sys.argv[1:]
    want_gate, want_train, want_eval = printed("gateflat"), printed("trainflat 1"), printed("evalflat 1")        # BEFORE the runs: a finished stage prints [skip]
    assert len(want_gate) == 1 and len(want_train) == 1 and len(want_eval) == 4
    got = real("gateflat")
    assert [g["argv"] for g in got] == want_gate and got[-1]["who"] == "train" and "--selfcheck-only" in got[-1]["argv"]
    n = len(got)
    got = real("trainflat 1")[n:]
    assert [g["argv"] for g in got] == want_train and len(got) == 1, "the REAL trainflat line hands the trainer the printed command"
    a = got[0]["argv"]
    assert a[a.index("--o-interface") + 1] == "cut" and a.count("--o-interface") == 1 and a[a.index("--arm") + 1] == "M2b" and a[a.index("--seed") + 1] == "1"
    assert (tmp_path / "runsm" / "M2b_s1_flat" / "summary.json").exists() and not (tmp_path / "runsm" / "M2b_s1").exists(), "FLAT has its own run dir; B's is never touched"
    n += 1
    got = real("evalflat 1")[n:]
    assert [g["argv"] for g in got] == want_eval and len(got) == 4 and all(g["who"] == "eval" for g in got), "the REAL evalflat lines hand the eval the printed commands"
    tags = [g["argv"][g["argv"].index("--tag") + 1] for g in got]
    assert tags == [f"M2b-flat_s1{s}" for s in SUFFIX_TAGS]
    n += 4                                              # a finished stage is skipped, not re-run
    assert len(real("evalflat 1")) == n and len(real("trainflat 1")) == n


@needs_queue
@win
@pytest.mark.skipif(_H2 is None, reason="test_v8f_hist.py (the fake repo) is not in this checkout")
def test_queue_flat_real_stages_notice_failures_and_skip_exactly_what_is_finished(tmp_path):
    repo, rec = _H2._fake_tree(tmp_path)
    runsm = tmp_path / "runsm"
    ev = runsm / "eval"
    base = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(runsm), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"))
    base.pop("V8F_DRYRUN", None)

    def stage(args, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(base, **extra), capture_output=True, text=True, timeout=180, encoding="utf-8")
        return r.returncode, r.stdout

    def evals():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines() if '"eval"' in x] if rec.exists() else []
    rc, out = stage("gateflat", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] gate M2b-flat" in out
    rc, out = stage("trainflat 1", FAKE_TRAIN="fail")
    assert rc == 1 and "[FAIL] train M2b_s1_flat - see" in out
    rc, out = stage("trainflat 1", FAKE_TRAIN="nosummary")                     # a trainer that exits 0 WITHOUT a summary (a crash that swallowed its code)
    assert rc == 1 and "exited 0 without summary.json" in out
    rc, out = stage("evalflat 1")                                              # evaluation refuses a run that has not finished training
    assert rc == 2 and "has not finished training" in out and not evals()
    assert stage("trainflat 1")[0] == 0
    rc, out = stage("trainflat 1")
    assert rc == 0 and "[skip] M2b_s1_flat" in out, "a finished run is not re-trained"
    tags = [f"M2b-flat_s1{s}" for s in SUFFIX_TAGS]
    for tag in tags:
        rc, out = stage("evalflat 1", FAKE_FAIL_TAG=tag)
        assert rc == 1 and f"[FAIL] eval {tag} - see" in out, (tag, rc, out)
        for f in list(ev.glob("eval_M2b-flat_s1*.npz")):
            f.unlink()
    ev.mkdir(exist_ok=True)
    for n, done in enumerate(([0, 2], [1], [0], [3], [1, 2], [0, 1, 2], [1, 3], [2, 3], [0, 1, 2, 3])):
        for f in list(ev.glob("eval_M2b-flat_s1*")):
            f.rename(f.with_name(f"old{n}_" + f.name))
        rec.write_text("", encoding="utf-8")
        for k in done:
            (ev / f"eval_{tags[k]}.npz").write_bytes(b"x")
        assert stage("evalflat 1")[0] == 0, done
        ran = [g["argv"][g["argv"].index("--tag") + 1] for g in evals()]
        assert ran == [tg for k, tg in enumerate(tags) if k not in done], (done, ran)


# =============================================================================================================================================
# THE CHAIN: needs box 2 and ssh, so it cannot RUN here -- its predecessor predicate (RUN on synthetic logs, both arms), its order, its gate assertions (RUN against synthetic selfchecks),
# its token record, its --dry-run (== the queue's own dry-run lines) and every load-bearing line are pinned
# =============================================================================================================================================
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
MIXO_CHAIN = CODE / "box2_ops" / "chain_v8f_mixo.sh"
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


@needs_chain
def test_chain_flat_static_ordering_the_wait_the_atomic_marker_both_seeds_and_no_base_decision():
    c = CHAIN.read_text(encoding="utf-8")
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    code = "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))
    assert "decide-base" not in code and "--base" not in code and "BASE" not in code, "B = M2b is fixed in the PREREG: nothing is decided at run time"
    # the order: predecessor -> marker -> refresh -> ckpt sha -> check-inputs on the REAL B side -> guard -> gate -> seed 0 -> seed 1 -> the ONE read
    order = ["waiting for the predecessor chain", "( set -C;", "BOOT0=$(boot)", "fetch -q origin", 'ckptsha_b "before the gate"', 'v8f_flat_read.py" --check-inputs', "v8f_guard.sh > ",
             "queue gateflat 120 gateflat", "runone 0\n", "runone 1\n", "flatread flat_read.json"]
    idx = [c.index(x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    assert 'queue "t_$g" $TRAIN_POLLS trainflat $s' in c and 'queue "e_$g" $EVAL_POLLS evalflat $s' in c
    between = c[c.index("runone 0\n") + len("runone 0\n"):c.index("runone 1\n")]
    assert all(x.strip() == "" or x.lstrip().startswith("#") for x in between.splitlines()), "BOTH seeds ALWAYS: nothing is decided, read or skipped between them"
    assert "DECISION" not in c and "CONTINUE" not in c and "--stage" not in c, "the PREREG's table has no stage: one read, after both seeds"
    assert c.rstrip().endswith('finish "$(jget flat_read.json reading) | $(jget flat_read.json next)"')
    assert c.index('v8f_flat_read.py" --check-inputs') < c.index("launch telemetry_long") < c.index("queue gateflat 120"), "B's side is refused while box 2 is still idle and untouched"
    # the predecessor is MIXO's chain by default (override-able); markers are the shared ones, in THIS chain's own log; the busy marker is taken atomically (noclobber)
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_mixo.out}" in c and "AFTER_GUARD=${AFTER_GUARD:-$L/v8fmixo_guard.out}" in c
    assert "AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}" in c and "AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}" in c
    assert "chain_v8f_flat.out" in c and 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh' in c
    assert "busy marker taken while arming" in c and "BOX2_BUSY.txt" in c
    # the training wait covers the longest run the guard lets live
    polls = int(re.search(r"^TRAIN_POLLS=(\d+)", c, re.M).group(1))
    maxh = float(re.search(r"V8F_MAX_RUN_H:-(\d+)", GUARD.read_text(encoding="utf-8")).group(1))
    assert polls == 1320 and polls * 30 / 3600 > maxh, f"a run the guard lets live ({maxh} h) outlasts the wait ({polls * 30 / 3600} h)"
    # the B side: the banked M2b files incl. the oracles; the P10 guard file and the check-inputs come BEFORE any training
    for x in ("$L/v8f_mini/runs/M2b_s$s/$f", "$L/v8f_cz/eval/$f", "$L/v8f_tacgrp2/eval/$f", "floors_perwindow.npz"):
        assert x in c, x
    assert c.index("floors_perwindow.npz") < c.index('v8f_flat_read.py" --check-inputs') < c.index("queue gateflat 120")
    # what a FLAT run pulls: its selfcheck/summary/argv/log/ckpt (the reader reads the checkpoint's cfg) and the four evals
    assert "for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull" in c
    assert 'for f in "" $X_SUFFIXES; do for x in json npz log; do pull "eval/eval_$tg$f.$x"' in c
    assert re.search(r'^X_SUFFIXES="-tacabs -tacorc -tacorc-all-latcons"$', c, re.M)
    arch = next(x for x in c.splitlines() if "archive --format=tar" in x)
    assert all(x in arch for x in ("stack", "taniteval", '"$HJ"', '"$PK"', '"$B1"')), "the reader, the strata file and the lead block ride in the archive"
    for f in ("v8f_flat_read.py", "v8f_kind_read.py", "v8f_grid_read.py", "v8f_pgt_read.py", "v8f_dk_read.py", "v8f_wc_read.py", "v8f_tac_groups_read.py", "v8f_mini_full_report.py"):
        assert f in c, f"the chain insists the tip has {f}"
    assert "window_strata_eval141.npz" in c, "the chain insists the tip has the per-window strata file"
    g = GUARD.read_text(encoding="utf-8")
    assert "ZZV8FMINI" in g and "(DONE|STOP)" in g


@needs_chain
def test_chain_flat_checkpoint_sha256_is_asserted_before_the_gate_and_after_the_evals():
    c = CHAIN.read_text(encoding="utf-8")
    assert 'ckptsha_b "before the gate"' in c and "queue \"e_$g\" $EVAL_POLLS evalflat $s\n  ckptsha_x $s after_evals" in c and "queue \"t_$g\" $TRAIN_POLLS trainflat $s\n  echo \"\" > $CUR\n  ckptsha_x $s trained" in c
    assert 'stop "$1: box 2\'s M2b_s$s checkpoint sha256 $g != the banked run\'s ${CKPT_SHA[$s]}"' in c
    assert 'stop "M2b_s${1}_flat\'s checkpoint changed during the evals' in c
    assert "ckset \"M2b_s$s\" \"$g\"" in c and 'ckset "M2b-flat_s${1}_$2" "$g"' in c, "the measured values go into the json the reader takes as --ckpt-sha"


@needs_chain
def test_chain_flat_syntax_and_dry_run_prints_the_queue_lines(tmp_path):
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
    stages = ["gateflat", "trainflat 0", "trainflat 1", "evalflat 0", "evalflat 1"]
    assert [x for x in out if x.startswith("== ") and x[3:] in stages] == [f"== {s}" for s in stages]
    want = []                                               # the chain's printed lines ARE the queue's own dry-run lines at box 2's default paths
    for st in stages:
        want += _dry(st, clean_paths=True)[1]
    assert [x for x in out if ".py" in x and "v8f_flat_read" not in x] == want and len(want) == 1 + 2 + 8
    t0 = next(x for x in out if "--seed 0" in x and "v8f_mini_train.py" in x)
    assert "--arm M2b --seed 0 --grid pool4x10 --o-interface cut --out" in t0 and "D:\\tanitad\\repo" in t0 and "M2b_s0_flat" in t0
    g0 = next(x for x in out if "--selfcheck-only" in x)
    assert "--arm M2b --o-interface cut --out" in g0 and "gate_M2b-flat" in g0
    assert "wait for <logs>/chain_v8f_mixo.out: ^ZZV8FMINI-DONE" in r.stdout
    for s, sha in ((0, R.BANKED_CKPT_SHA["M2b_s0"]), (1, R.BANKED_CKPT_SHA["M2b_s1"])) if R is not None else ():
        assert f"M2b_s{s} D:/tanitad/runs/v8f_mini_full/M2b_s{s}/ckpt_last.pt {sha}" in r.stdout
    reads = [x for x in out if x.startswith("v8f_flat_read.py")]
    assert len(reads) == 2 and "--check-inputs" in reads[0] and "--stage" not in reads[1] and all("--ckpt-sha" in x for x in reads)


# ---- the predecessor predicate (PREREG "Compute and order"), RUN on synthetic logs --------------------------------------------------------------------
def _classify(tmp_path, out_lines, guard_lines=None, name="c"):
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's predicate")
    o = tmp_path / f"{name}.out"
    o.write_bytes(("\n".join(out_lines) + "\n").encode("utf-8") if out_lines is not None else b"")
    args = [bash, str(CHAIN), "--classify-predecessor", o.as_posix()]
    if guard_lines is not None:
        g = tmp_path / f"{name}.guard"
        g.write_bytes(("\n".join(guard_lines) + "\n").encode("utf-8"))
        args.append(g.as_posix())
    r = subprocess.run(args, capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout.strip(), r.stderr


DONE_LINE = "ZZV8FMINI-DONE 10-26 04:10 tip 18c62306: PREREG_V8F_MIXO PASS"
PROCEED, HOLDING, WAIT = 0, 1, 2


@needs_chain
def test_predecessor_done_proceeds_and_no_end_marker_waits(tmp_path):
    rc, out, err = _classify(tmp_path, ["10-25 12:00 boot0=2026-10-24 07:00:00", "10-25 12:30 t_M2b_s0_mixo rc=0", DONE_LINE])
    assert rc == PROCEED and out.startswith("PROCEED the predecessor ended: ZZV8FMINI-DONE"), (rc, out, err)
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=2026-10-24 07:00:00", "10-25 12:30 t_M2b_s0_mixo rc=0"], name="wait")
    assert rc == WAIT and out.startswith("WAIT no ZZV8FMINI-DONE"), (rc, out)
    rc, out, _ = _classify(tmp_path, None, name="empty")
    assert rc == WAIT, "an empty log: the predecessor has not started"
    o = tmp_path / "crlf.out"
    o.write_bytes((DONE_LINE + "\r\n").encode())
    r = subprocess.run([_git_bash(), str(CHAIN), "--classify-predecessor", o.as_posix()], capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == PROCEED, "a CRLF log (the chain's output file) is read the same"
    rc, out, _ = _classify(tmp_path, ["ZZV8FMINI-STOP t_x rc=1", DONE_LINE], name="both")
    assert rc == PROCEED, "a DONE anywhere in the log wins (a stop followed by a done is a finished chain)"


# every `stop "<msg>"` text of the predecessor's chain (chain_v8f_mixo.sh), rendered with plausible values, and the decision the PREREG commits for it. A run-level stop (a box-2 job returned
# rc != 0), a verdict-level stop (the reader refused) and the launch gate's refusal PROCEED; everything else is box-level or unknown and HOLDS.
STOP_TABLE = [
    ("t_M2b_s0_mixo rc=1", PROCEED), ("e_M2b_s1_mixo rc=2", PROCEED), ("v8fmixo10231904_refresh rc=3", PROCEED), ("gatemixo rc=1", PROCEED),
    ("mixo read s0", PROCEED), ("mixo read final", PROCEED), ("gate X assertions", PROCEED),
    ("t_M2b_s0_mixo wait=2 (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)", HOLDING), ("e_M2b_s1_mixo wait=3 (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)", HOLDING),
    ("gatemixo wait=4 (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)", HOLDING), ("box 2 unreachable", HOLDING), ("guard stop", HOLDING),
    ("no fresh telemetry (a / b): the temperature guard would be blind", HOLDING), ("the predecessor stopped: ZZV8FMINI-STOP t_x rc=1", HOLDING),
    ("busy marker present: v8F-mini PREREG_V8F_KIND", HOLDING), ("busy marker taken while arming: x", HOLDING),
    ("ship v8fmixo1_gatemixo", HOLDING), ("launch v8fmixo1_gatemixo", HOLDING), ("pull D:/tanitad/runs/v8f_mini_full/M2b_s0_mixo/ckpt_last.pt", HOLDING),
    ("archive", HOLDING), ("ship the code archive", HOLDING), ("ship refresh job", HOLDING), ("launch refresh", HOLDING), ("extract the reader", HOLDING), ("write the gate token", HOLDING),
    ("remote tip abc is not in the mirror", HOLDING), ("box 2 stack/scripts/v8f_mini_train.py != tip abc", HOLDING), ("box 2 v9 v9_labels_train.npz md5 != 83e3", HOLDING),
    ("before the gate: box 2's M2b_s0 checkpoint sha256 aa != the banked run's bb", HOLDING), ("cannot hash box 2's M2b_s0_mixo checkpoint (trained)", HOLDING),
    ("M2b_s0_mixo's checkpoint changed during the evals: a != b", HOLDING), ("the reader cannot use the M2b side (check-inputs)", HOLDING),
    ("stage floors_perwindow.npz (the P10 reproduction guard)", HOLDING), ("the tip's queue has no gatemixo stage", HOLDING), ("a stop text this predicate has never seen", HOLDING), ("", HOLDING),
    ("t_x rc=0", HOLDING), ("t_x rc=", HOLDING), ("rc=1", HOLDING),
    ("mixo read", PROCEED),             # a reader with no stage (this chain's own `flat read`) is a verdict-level stop too: the predicate reads the chains this one may itself be waited on by
]


@needs_chain
@pytest.mark.parametrize("msg, want", STOP_TABLE, ids=[f"{i:02d}" for i in range(len(STOP_TABLE))])
def test_predecessor_stop_is_classified_run_level_and_verdict_level_proceed_and_everything_else_holds(tmp_path, msg, want):
    rc, out, err = _classify(tmp_path, ["10-25 12:00 boot0=2026-10-24 07:00:00", f"ZZV8FMINI-STOP {msg}".rstrip()])
    assert rc == want, (msg, rc, out, err)
    assert out.split()[0] == ("PROCEED" if want == PROCEED else "HOLD")


@needs_chain
def test_every_stop_text_of_the_predecessors_chain_is_in_the_table_so_a_new_one_cannot_slip_through_unclassified():
    """The table above renders the predecessor chain's own `stop "<msg>"` calls. A stop text added to chain_v8f_mixo.sh later is not in the table: this test names it, and the
    decision for it is then made on purpose (the predicate holds on anything it does not recognise, so the default is the safe one)."""
    src = MIXO_CHAIN.read_text(encoding="utf-8")
    stops = sorted(set(re.findall(r'stop "([^"]*)"', src)))
    assert len(stops) >= 35
    covered = {"stage $f", "write the gate token", "v8f_guard.sh does not know this chain's markers (anchored): it would never exit", "the tip's v8f_mini.py has no CODE_MIX_UNITS partition",
               "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --code-mix", "the tip's trainer has no --code-mix flag", "the tip's trainer does not run the code_mix probe in its in-process gate",
               "the tip's queue prints no gatemixo line", "the tip's queue has no $st stage", "the tip's built_levers does not record p_gt_start", "the tip's built_levers does not record p_gt_end",
               "the tip's built_levers does not record code_mix", "the tip has no B1 lead block", "the tip has no $f", "the reader cannot use the M2b side (check-inputs)",
               "stage the banked M2b_s$s/$f", "stage floors_perwindow.npz (the P10 reproduction guard)", "stage eval_M2b_s$s", "stage $f (the banked dose chain of B)", "ship the code archive",
               "ship refresh job", "ship $j", "remote tip $T is not in the mirror", "pull $R/$1", "no fresh telemetry ($t1 / $t2): the temperature guard would be blind", "mixo read $1",
               "launch refresh", "launch $j", "guard stop", "gate X assertions", "extract the reader", "cannot hash box 2's M2b_s${1}_mixo checkpoint ($2)", "box 2 v9 ${kv%%=*} md5 != ${kv#*=}",
               "box 2 unreachable", "box 2 $f != tip $T", "archive", "M2b_s${1}_mixo's checkpoint changed during the evals: $g != $g0", "$1: cannot hash box 2's M2b_s$s checkpoint",
               "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}", "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)", "$1 rc=$rc"}
    assert [s for s in stops if s not in covered] == [], "a stop text of the predecessor chain this test and STOP_TABLE have never classified"
    run_level = [s for s in stops if s.endswith("rc=$rc")]
    verdict_level = [s for s in stops if re.search(r"read \$1$", s)]
    gate_level = [s for s in stops if re.fullmatch(r"gate \w+ assertions", s)]
    assert run_level == ["$1 rc=$rc"] and verdict_level == ["mixo read $1"] and gate_level == ["gate X assertions"], "the three proceeding classes are exactly these texts"


@needs_chain
@pytest.mark.parametrize("ev", ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: 12 of the last 20 samples >= 61 C, max 63 C",
                                "ZZGUARD-STOP TIME 10-25 13:00 M2b_s0_mixo projects 11.2 h (> 10 h) at 2.1 s/step",
                                "ZZCRASH 10-25 13:00 box 2 rebooted (boot time changed)", "ZZUNREACHABLE 10-25 13:00 ssh", "ZZTIMEOUT 10-25 13:00 t_x 1320 polls"])
def test_predecessor_run_level_stop_with_box_level_evidence_in_its_logs_holds(tmp_path, ev):
    """The evidence overrides the message: a run-level `rc=1` that came with a thermal / time stop, a reboot, an unreachable box or a timeout is the box's failure, not the run's."""
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=x", ev, "ZZV8FMINI-STOP t_M2b_s0_mixo rc=1"], name="inchain")
    assert rc == HOLDING and out.startswith("HOLD box-level evidence"), (rc, out)
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=x", "ZZV8FMINI-STOP t_M2b_s0_mixo rc=1"], [ev], name="inguard")
    assert rc == HOLDING and out.startswith("HOLD box-level evidence"), "the evidence may be in the predecessor's guard log"
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=x", "ZZV8FMINI-STOP mixo read final"], [ev], name="verdict")
    assert rc == HOLDING, "a verdict-level stop with box-level evidence holds too"
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=x", "ZZV8FMINI-STOP t_M2b_s0_mixo rc=1"], ["10-25 13:00 chain ended; guard off (max USB 52 C)"], name="clean")
    assert rc == PROCEED, "a guard log that merely ended is no evidence"
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=x", ev, DONE_LINE], name="done_wins")
    assert rc == PROCEED, "ZZV8FMINI-DONE proceeds whatever the log says before it: the predecessor FINISHED"


@needs_chain
def test_the_wait_loop_calls_the_same_predicate_and_the_chain_stops_on_a_hold():
    c = CHAIN.read_text(encoding="utf-8")
    assert 'd=$(after_decision "$AFTER_OUT" "$AFTER_GUARD"); r=$?' in c and 'if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor\'s end does not clear the box: $d"; exit 1; fi' in c
    assert 'if [ "${1:-}" = "--classify-predecessor" ]; then after_decision' in c
    assert c.index("after_decision() {") < c.index('if [ "${1:-}" = "--classify-predecessor" ]') < c.index("waiting for the predecessor chain")
    assert "sleep 300" in c and "RUN_LEVEL_RE='^[A-Za-z0-9_.-]+ rc=[1-9][0-9]*$'" in c


@needs_chain
def test_the_wait_loop_runs_end_to_end_proceeds_on_done_and_stops_on_a_box_level_stop(tmp_path):
    """Run the chain itself against a synthetic predecessor log with a stubbed box: DONE passes the wait (and the chain then dies at its first box-2 step: the stubbed `ssh` fails, the
    busy marker is taken and released); a thermal stop never gets past it, and takes no marker."""
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

    def run(lines, guard=None):
        pred = box / "logs" / "chain_v8f_mixo.out"
        pred.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        gp = box / "logs" / "v8fmixo_guard.out"
        gp.write_bytes(("\n".join(guard) + "\n").encode("utf-8") if guard else b"")
        (box / "BOX2_BUSY.txt").unlink(missing_ok=True)
        r = subprocess.run([bash, str(script)], env=env, capture_output=True, text=True, timeout=120, encoding="utf-8", stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout
    rc, out = run([DONE_LINE])
    assert rc == 1 and "predecessor: PROCEED the predecessor ended" in out and "ZZV8FMINI-STOP box 2 unreachable" in out, out
    assert not (box / "BOX2_BUSY.txt").exists(), "a chain that stops releases the marker it took"
    rc, out = run(["ZZV8FMINI-STOP t_M2b_s0_mixo rc=1"])
    assert rc == 1 and "predecessor: PROCEED run-level stop: t_M2b_s0_mixo rc=1" in out and "ZZV8FMINI-STOP box 2 unreachable" in out, out
    rc, out = run(["ZZV8FMINI-STOP guard stop"], ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: x"])
    assert rc == 1 and "ZZV8FMINI-STOP the predecessor's end does not clear the box: HOLD" in out and "box 2 unreachable" not in out, out
    assert not (box / "BOX2_BUSY.txt").exists(), "a held chain never took the marker"
    rc, out = run(["ZZV8FMINI-STOP t_M2b_s0_mixo wait=3 (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"])
    assert rc == 1 and "does not clear the box: HOLD a stop that is not recognised" in out and "box 2 unreachable" not in out, out


def _gate_assertions():
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"<<'EOF' \|\| stop \"gate FLAT assertions\"\n(.*?)\nEOF\n", c, re.S)
    assert m, "the chain's gate assertions are a python heredoc whose failure stops the chain"
    return m.group(1)


GLINE_OK = ('"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_mini_train.py" --arm M2b --o-interface cut '
            '--out "D:\\tanitad\\runs\\v8f_mini_full\\gate_M2b-flat" --kin-mode yaw --selfcheck-only')


def _gate_run(tmp_path, gate, mod=None, line=GLINE_OK, steps_arg="19164", windows_arg="306566"):
    body = _gate_assertions()
    real, _ = _H._b_real(0)
    bl = real["summary.json"]["levers"]
    good = {"declared_vs_built": "PASS", "live": copy.deepcopy(gate["cut"]["live"]), "o_interface_probe": copy.deepcopy(gate["cut"]["o_interface_probe"]), "steps_total": 19164,
            "info": {"n_clips": 4512, "n_excluded": 60, "n_excluded_declared": 60, "excluded_in_train": 0, "excluded_sha256": HOLD,
                     "train_sha256": real["summary.json"]["train_sha256"], "kin_mode": "yaw", "lru": 174, "n_windows_all_clips": 306566,
                     "window_start": 3, "window_t_min": 3, "ds_window_start": 3},
            "levers": {"wm": "b", "navonly": False, "consistency": True, "cons_absent": "mask", "o_interface": "cut", "code_mix": "whole", "p_gt_start": 1.0, "p_gt_end": 0.2, "hist": 4, "hmax": 16,
                       "window_start": 3, "w": {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5},
                       "params": dict(bl["params"]), "n_params_trainable": bl["n_params_trainable"]}}
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
def test_chain_flat_gate_assertions_pass_the_flat_gate_and_refuse_every_way_it_can_be_wrong(tmp_path, gate_rec):
    """The chain's gate block is RUN against a selfcheck.json built from B's REAL banked record and the trainer's OWN cut gate record (G-LIVE, the o_interface probe): a good FLAT gate
    passes; each single-field corruption exits 1 and names the broken field (the PASS token, the dead set, the probe, the lever, the schedule, the clip list, the holdout, the weights, the
    window set, the step count, the history and the PARAMETER COUNT cannot drift under a green gate)."""
    rc, out = _gate_run(tmp_path, gate_rec)
    pb = _H._b_real(0)[0]["summary.json"]["levers"]
    assert rc == 0 and "PASS" in out and "built o_interface cut" in out and "built p_gt 1.0 -> 0.2" in out and f"params {pb['n_params_trainable']}" in out and "windows 306566" in out, out
    assert "probe max |d traj60| 0.0" in out, out
    L = lambda k, v: (lambda s: s["levers"].__setitem__(k, v))                   # noqa: E731
    W = lambda k, v: (lambda s: s["levers"]["w"].__setitem__(k, v))              # noqa: E731
    I = lambda k, v: (lambda s: s["info"].__setitem__(k, v))                     # noqa: E731
    P = lambda k, v: (lambda s: s["levers"]["params"].__setitem__(k, v))         # noqa: E731
    PERT = lambda n, **kv: (lambda s: s["o_interface_probe"]["perturbations"][n].update(kv))  # noqa: E731
    cases = (("declared_vs_built", lambda s: s.__setitem__("declared_vs_built", "FAIL"), "declared_vs_built"),
             ("live", lambda s: s["live"].__setitem__("pass", False), "live"),
             ("control grad", lambda s: s["live"].__setitem__("control_prior_grad_sum", 1.0), "live"),
             ("the cut left e_code alive", lambda s: s["live"].update(dead_params=[], declared_dead=[]), "G-LIVE dead set"),
             ("another weight dead too", lambda s: s["live"].update(dead_params=["e_code.weight", "head_o.2.weight"]), "G-LIVE dead set"),
             ("no probe", lambda s: s.pop("o_interface_probe"), "o_interface_probe None"),
             ("probe not pass", lambda s: s["o_interface_probe"].__setitem__("pass", False), "o_interface_probe"),
             ("probe for full", lambda s: s["o_interface_probe"].__setitem__("o_interface", "full"), "was run for 'full'"),
             ("probe lost a perturbation", lambda s: s["o_interface_probe"]["perturbations"].pop("gt_code"), "perturbed"),
             ("traj60 moved", PERT("donor_code", max_abs_dtraj60=1e-9), "traj60 moved or O was handed a non-constant"),
             ("O read the code", PERT("random_code", o_read_code_abs_max=0.1), "traj60 moved or O was handed a non-constant"),
             ("O read a checkpoint", PERT("gt_code", o_read_ck_minus_nominal_abs_max=0.5), "traj60 moved or O was handed a non-constant"),
             ("O read ok", PERT("shifted_checkpoints", o_read_ck_ok_abs_max=1.0), "traj60 moved or O was handed a non-constant"),
             ("T does not read S", PERT("shifted_checkpoints", t_reads_s_max_abs_dgoal_logits=0.0), "T must still read S"),
             ("n_clips", I("n_clips", 4000), "clips / holdout counts"), ("n_excluded", I("n_excluded", 59), "clips / holdout counts"),
             ("n_excluded_declared", I("n_excluded_declared", 59), "clips / holdout counts"), ("excluded_in_train", I("excluded_in_train", 1), "clips / holdout counts"),
             ("holdout", I("excluded_sha256", "0" * 64), "not the RA-1 holdout"), ("train list", I("train_sha256", "d" * 64), "TRAIN list"),
             ("kin_mode", I("kin_mode", "steer"), "kin_mode"), ("lru unrecorded", lambda s: s["info"].pop("lru"), "info.lru"),
             ("wm", L("wm", "a"), "levers"), ("navonly", L("navonly", True), "levers"), ("consistency", L("consistency", False), "levers"), ("cons_absent", L("cons_absent", "zero"), "levers"),
             ("built o_interface full", L("o_interface", "full"), "built o_interface 'full' != 'cut'"), ("o_interface unrecorded", lambda s: s["levers"].pop("o_interface"), "built o_interface None != 'cut'"),
             ("built code_mix kind", L("code_mix", "kind"), "built code_mix 'kind' != 'whole'"),
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
        rc, out = _gate_run(tmp_path, gate_rec, mod)
        assert rc != 0 and msg in out, (name, rc, out)
    # the lever, from the queue's own printed line
    for name, line, msg in (("no --o-interface on the line", GLINE_OK.replace(" --o-interface cut", ""), "--o-interface cut exactly once"),
                            ("--o-interface full", GLINE_OK.replace("--o-interface cut", "--o-interface full"), "--o-interface cut exactly once"),
                            ("--o-interface twice", GLINE_OK.replace("--o-interface cut", "--o-interface cut --o-interface cut"), "--o-interface cut exactly once"),
                            ("another arm", GLINE_OK.replace("--arm M2b", "--arm M2"), "is not --arm M2b"),
                            ("--hist set", GLINE_OK + " --hist 10", "sets --hist"), ("--hmax set", GLINE_OK + " --hmax 16", "sets --hmax"),
                            ("--p-gt-start set", GLINE_OK + " --p-gt-start 0.5", "sets --p-gt-start"), ("--window-start set", GLINE_OK + " --window-start 3", "sets --window-start"),
                            ("--w-cons set", GLINE_OK + " --w-cons 1.0", "sets --w-cons"), ("--grid set", GLINE_OK + " --grid pool8x20", "sets --grid"),
                            ("--code-mix set", GLINE_OK + " --code-mix kind", "sets --code-mix")):
        rc, out = _gate_run(tmp_path, gate_rec, line=line)
        assert rc == 1 and msg in out, (name, rc, out)
    # a budget different from B's, however it came about, stops the chain at the gate
    rc, out = _gate_run(tmp_path, gate_rec, steps_arg="17472")
    assert rc == 1 and "steps 19164 != B's 17472" in out, out


@needs_chain
@needs_pkg
def test_chain_flat_writes_a_gate_token_bound_to_the_commit_the_queue_the_selfcheck_and_the_data_manifests(tmp_path):
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
    assert tok["prereg"] == "PREREG_V8F_FLAT" and tok["gate"] == "PASS" and tok["commit"] == "c0ffee" and tok["queue_sha256"] == "q" * 64 and tok["argv_sha"] == "abc123"
    assert tok["selfcheck_sha256"] == hashlib.sha256(scp.read_bytes()).hexdigest() and tok["key_files_sha256"] == json.loads(keys)
    assert tok["train_sha256"] == "t" * 64 and tok["excluded_sha256"] == HOLD and set(tok["v9_md5"]) == {"v9_labels_train.npz", "v9_labels_eval139.npz"}
    # ordering: the token is written only after the gate assertions passed, and before any training
    assert c.index('"gate FLAT assertions"') < c.index('"write the gate token"') < c.index("runone 0\n")


@needs_chain
def test_chain_flat_finish_releases_the_box_before_it_announces_done_and_stop_releases_it_too():
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
def test_chain_flat_load_bearing_lines_are_present():
    """Every line whose loss would let a wrong run through is pinned. Substrings, not whole lines: line ends are not load-bearing."""
    c = CHAIN.read_text(encoding="utf-8")
    lines = (
        r"""for st in gateflat trainflat evalflat; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done""",
        r"""grep -q '"o_interface": cfg.o_interface' || stop "the tip's built_levers does not record o_interface""",
        r"""grep -q '^O_INTERFACE_MODES = (' || stop "the tip's v8f_mini.py has no O_INTERFACE_MODES""",
        r"""grep -q 'if cfg.o_interface == "cut":' || stop "the tip's v8f_mini.py has no cut line""",
        r"""grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start""",
        r"""grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end""",
        r"""grep -q -- '"--o-interface"' || stop "the tip's trainer has no --o-interface flag""",
        r"""grep -q '"o_interface": "lever:o_interface"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --o-interface""",
        r"""grep -q 'o_interface_probe(model, probe, cfg)' || stop "the tip's trainer does not run the o_interface probe in its in-process gate""",
        r"""git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block""",
        r"""|| stop "the reader cannot use the M2b side (check-inputs)""",
        r"""|| stop "flat read""",
        r"""$PY "$RD/v8f_flat_read.py" $(reader_args""",
        r"""GLINE=$(dry_stage "$QC" gateflat | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gateflat line""",
        r"""$HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS <<'EOF' || stop "gate FLAT assertions""",
        r"""pull "gate_M2b-flat/selfcheck.json" "$E/gate_M2b-flat""",
        r"""echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)""",
        r"""[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present""",
        r"""if [ $r = 0 ]; then echo "$(date '+%m-%d %H:%M') predecessor: $d"; break; fi""",
        r"""cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo""",
        r"""cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done""",
        r"""cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)""",
        r"""finish "$(jget flat_read.json reading) | $(jget flat_read.json next)""",
        r"""g="M2b_s${1}_flat" tg="M2b-flat_s${1}""",
        r"""grep ZZGUARD-STOP $L/v8fflat_guard.out 2>/dev/null && stop "guard stop""",
        r"""V8F_CHAIN_OUT=$L/chain_v8f_flat.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh""",
        r"""[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry""",
        r"""M2B_STEPS=19164""",
        r"""M2B_WINDOWS=306566""",
        r"""TRAIN_POLLS=1320; EVAL_POLLS=600""",
        r"""rm -f $CK""",
        r"""|| stop "write the gate token""",
    )
    for x in lines:
        assert x in c, f"the chain lost: {x}"


# =============================================================================================================================================
# (H) THE EIGHT MUTANTS THAT SURVIVED THE FIRST RUN OF raw/2026-10-25-flat/mutate_flat.py (#20 #23 #27 #28 #29 #30 #37 #39)
# =============================================================================================================================================
# Each test below passes on the real code and FAILS (an assertion, not an error) on its mutant; each carries the control that proves the fault it injects is the only thing that differs.
def _bind(model, name, fn):
    """Install `fn(self, ...)` as an INSTANCE attribute `name` of `model`, bound to it. `copy.deepcopy` re-binds a bound method to the copy, so the probe (which deep-copies the model)
    runs the fault."""
    setattr(model, name, types.MethodType(fn, model))
    return model


def _deaf_o(model):
    """A model whose O reads the CONSTANTS whatever `_forward` hands it: traj60 is invariant to every perturbation (so the traj60 check cannot tell it from a cut model) while the tensors
    `_forward` hands `run_o` are still the real ones."""
    f = type(model).run_o

    def run_o(self, b, adapt, last_o, ck_o, ck_ok, code):
        return f(self, b, adapt, last_o, self.ck_nom[None].expand(ck_o.shape[0], -1, -1), torch.zeros_like(ck_ok), torch.zeros_like(code))
    return _bind(model, "run_o", run_o)


def test_the_probe_refuses_a_declared_cut_whose_o_is_handed_a_non_constant_even_though_traj60_does_not_move(world):
    """Mutant #27 (the 'handed a non-constant' check is deleted). The model is built FULL (its `_forward` hands O S's checkpoints, ok = 1 and T's code) but O is deaf to them, so traj60 is
    bit-for-bit invariant: only the check on what O is HANDED can see the wire."""
    J, b = _J(world), _batch(world)
    m = _deaf_o(_wake(_mk(M, J, o_interface="full")))
    cut_cfg = dataclasses.replace(m.cfg, o_interface="cut")
    with pytest.raises(AssertionError, match="full is declared but traj60 did not move"):               # the control: the deaf O really makes traj60 invariant (so the traj60 check is blind here)
        TR.o_interface_probe(m, b, m.cfg)
    with pytest.raises(AssertionError, match="cut is declared but O was handed a non-constant under donor_code") as ei:
        TR.o_interface_probe(m, b, cut_cfg)
    assert "traj60 moved" not in str(ei.value), "the refusal comes from the handed-tensor check, not the traj60 check"
    honest = _wake(_mk(M, J, o_interface="cut"))
    assert TR.o_interface_probe(honest, b, honest.cfg)["pass"] is True                                 # the same declaration on a model that really cuts passes


def test_the_probe_refuses_a_model_whose_t_no_longer_reads_s(world):
    """Mutant #28 (the 'T still reads S' check is deleted). The cut must leave S -> T intact: a model whose T ignores S's checkpoints passes every other check of the probe."""
    J, b = _J(world), _batch(world)
    honest = _wake(_mk(M, J, o_interface="cut"))
    assert TR.o_interface_probe(honest, b, honest.cfg)["perturbations"]["shifted_checkpoints"]["t_reads_s_max_abs_dgoal_logits"] > 0.0
    m = _wake(_mk(M, J, o_interface="cut"))
    f = type(m).run_t

    def run_t(self, bb, adapt, last_t, ck_t, ck_ok):
        return f(self, bb, adapt, last_t, self.ck_nom[None].expand(ck_t.shape[0], -1, -1), ck_ok)       # T reads the nominal arcs, never S's checkpoints
    _bind(m, "run_t", run_t)
    with pytest.raises(AssertionError, match="T's output did not move with S's checkpoints"):
        TR.o_interface_probe(m, b, m.cfg)


def test_the_probe_refuses_a_code_override_that_never_reached_the_interface(world):
    """Mutant #29 (the 'override reached the interface' check is deleted). A `_forward` that drops `code_override` leaves every other check of a cut model green (traj60 is invariant anyway)."""
    J, b = _J(world), _batch(world)
    m = _wake(_mk(M, J, o_interface="cut"))
    assert TR.o_interface_probe(m, b, m.cfg)["pass"] is True                                         # the control: the unfaulted model passes
    f = type(m)._forward

    def fwd(self, bb, p_gt, use_labels, gen=None, code_override=None):
        return f(self, bb, p_gt, use_labels, gen, None)                                                  # the override is dropped
    _bind(m, "_forward", fwd)
    with pytest.raises(AssertionError, match="the donor_code override did not reach the interface"):
        TR.o_interface_probe(m, b, m.cfg)


def test_the_probe_refuses_a_checkpoint_shift_that_did_not_move_s(world):
    """Mutant #30 (the 'shift moved S' check is deleted). Without it the probe still refuses -- but with the OTHER message (T did not move): the refusal must name the real cause, a shift that
    never reached S."""
    J, b = _J(world), _batch(world)
    m = _wake(_mk(M, J, o_interface="cut"))
    f = type(m)._forward

    def fwd(self, *a, **k):
        shifted = self.__dict__.pop("run_s", None)                                                       # the probe's shifted run_s is set aside: the class's run_s runs, S does not move
        try:
            return f(self, *a, **k)
        finally:
            if shifted is not None:
                self.__dict__["run_s"] = shifted
    _bind(m, "_forward", fwd)
    with pytest.raises(AssertionError, match="shifting S's output did not move S's checkpoints") as ei:
        TR.o_interface_probe(m, b, m.cfg)
    assert "T does not read S" not in str(ei.value)


def test_a_refusing_o_interface_probe_stops_the_trainer_before_it_writes_the_selfcheck(tmp_path, monkeypatch):
    """Mutant #23 (main turns the probe's AssertionError into a recorded `pass: False` and carries on). The refusal must be a SystemExit carrying the ZZDVB-REFUSEDZZ marker the supervisor and
    the launch gate look for, and nothing may be written after it. The control: the same argv with the real probe runs to a PASS."""
    argv = ["--out", str(tmp_path / "ok"), *COMMON, "--steps", "2", "--selfcheck-only", "--o-interface", "cut"]
    assert TR.main(argv) == 0
    assert json.loads((tmp_path / "ok" / "selfcheck.json").read_text(encoding="utf-8"))["o_interface_probe"]["pass"] is True
    shutil.rmtree(tmp_path / "ok", ignore_errors=True)
    calls = []

    def refuse(model, batch, cfg, *a, **k):
        calls.append(cfg.o_interface)
        raise AssertionError("o_interface probe: the stub refuses")
    monkeypatch.setattr(TR, "o_interface_probe", refuse)
    bad = ["--out", str(tmp_path / "bad"), *COMMON, "--steps", "2", "--selfcheck-only", "--o-interface", "cut"]
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ declared-vs-built: o_interface probe: the stub refuses"):
        TR.main(bad)
    assert calls == ["cut"], "the trainer called the probe once, with the built config"
    assert not (tmp_path / "bad" / "selfcheck.json").exists() and not (tmp_path / "bad" / "ckpt_last.pt").exists(), "a refused launch writes no selfcheck and no checkpoint"


_LORA = []


def _lora_helpers():
    """test_v8f_lora.py's tiny-encoder helpers, loaded lazily (that module skips itself when transformers has no DINOv3: only this test may skip, never the whole file)."""
    if not _LORA:
        try:
            _LORA.append(_load("test_v8f_lora_for_flat", HERE_T / "test_v8f_lora.py"))
        except BaseException as e:                                                                        # noqa: BLE001  (pytest.Skipped is a BaseException)
            pytest.skip(f"test_v8f_lora.py is not importable here: {type(e).__name__}")
    return _LORA[0]


@pytest.mark.parametrize("mode, dead", [("cut", ["e_code.weight"]), ("full", [])])
def test_live_check_lora_declares_e_code_weight_dead_under_cut_and_only_under_cut(tmp_path, monkeypatch, mode, dead):
    """Mutant #20 (live_check_lora forgets the cut's declared-dead weight). Through the REAL encoder path (the tiny stand-in for DINOv3, as test_v8f_lora.py uses): under cut e_code.weight reads
    EXACTLY 0.0 and is declared dead, so the LoRA gate passes with it in `dead_params`; under full it is an ordinary trained weight."""
    L = _lora_helpers()
    if not L.HAVE_BUILDER:
        pytest.skip("the cache builder (the tiny geometry's grid constants) is not in this checkout")
    pep, _ = L.E.load_builder()                                                      # test-scale geometry exactly as test_v8f_lora.py's autouse `tiny_geometry`: the builder's constants are patched, never its code
    monkeypatch.setattr(pep, "GH", 4)
    monkeypatch.setattr(pep, "GW", 10)
    monkeypatch.setattr(D, "FRAME_HW", L.TINY_HW)
    monkeypatch.setattr(TR, "build_encoder_raw", L._tiny_raw)
    out = tmp_path / "o"
    assert TR.main(L.lora_argv(tmp_path, out, ["--selfcheck-only", "--o-interface", mode], arm="M2b")) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    assert sc["live"]["pass"] is True and sc["live"]["dead_params"] == dead and sc["live"]["unexpected_dead"] == []
    assert sc["levers"]["o_interface"] == mode and sc["o_interface_probe"]["pass"] is True


def test_a_seed_state_needs_the_librarys_separated_flag_on_top_of_the_side_of_the_interval():
    """Mutants #37 / #39 (the `separated` conjunct of LO / HI is dropped). The reader takes `separated` FROM the library and the PREREG's rule names it (LO iff Delta <= -2F AND the interval is
    separated below 0): a record whose flag says 'not separated' is never LO or HI, whatever its bounds say. (From the library's own output the two cannot disagree -- see the next test --
    so this pins the rule at the function's boundary, the only place a record that disagrees with itself can be handed in.)"""
    bar = R.BAR_2F

    def p(point, lo, hi, sep):
        return {"point": point, "lo": lo, "hi": hi, "separated": sep}
    assert R.seed_state(p(-0.9, -1.5, -0.1, True)) == "LO" and R.seed_state(p(0.9, 0.1, 1.5, True)) == "HI", "the control: the same records with the flag set"
    assert R.seed_state(p(-0.9, -1.5, -0.1, False)) == "MID", "LO needs the library's separated flag, not only an upper bound below 0"
    assert R.seed_state(p(0.9, 0.1, 1.5, False)) == "MID", "HI needs the library's separated flag, not only a lower bound above 0"
    assert R.seed_state(p(-bar, -1.5, -1e-12, False)) == "MID" and R.seed_state(p(bar, 1e-12, 1.5, False)) == "MID"


def test_the_librarys_separated_follows_from_the_interval_sitting_on_the_bars_side():
    """The equivalence behind the previous test, measured on the library the reader calls: whenever the paired bootstrap returns hi < 0 or lo > 0 (rounded as the dict reports them) its own
    `separated` is True, so on library output the reader's `separated` conjunct never changes a state. If the library ever changes that, this test goes red and the conjunct becomes load-bearing
    (re-read the PREREG's rule before editing either)."""
    from taniteval.ci import paired_episode_cluster_bootstrap as pf
    rng = np.random.default_rng(7)
    n_ep, per = 10, 3
    eid = np.repeat(np.arange(n_ep), per)
    seen = {"hi<0": 0, "lo>0": 0, "sep_false": 0, "sep_true": 0}
    for mu in np.linspace(-1.2, 1.2, 49):                                                                # intervals that start well inside zero and end well outside, with many near the edge
        for rep in range(2):
            b = rng.normal(0.0, 1.0, n_ep * per)
            a = b + mu + rng.normal(0.0, 0.5, n_ep * per)
            r = pf(a, b, eid, n_boot=200, seed=int(rep))
            seen["sep_true" if r["separated"] else "sep_false"] += 1
            if r["hi"] < 0:
                seen["hi<0"] += 1
                assert r["separated"] is True, r
            if r["lo"] > 0:
                seen["lo>0"] += 1
                assert r["separated"] is True, r
    assert min(seen.values()) > 0, f"the sweep must cover both sides and both flags, else it proves nothing: {seen}"
