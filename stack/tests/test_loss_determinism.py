"""⛔ ``v6_loss_step`` must be reproducible from its own generator.

WHY THIS FILE EXISTS (MEASURED 2026-08-16, ``LOSS_DETERMINISM.md``). Two
identical ``v6_loss_step`` calls with the SAME ``generator`` returned different
numbers — S-W total **3.9301 vs 3.9227**, and the whole discrepancy sat in
**``o6``** (0.046874 vs 0.039470, **18.7 %**) while every other term was
bit-identical. Root cause: ``sigreg.py`` drew its M slice directions with a bare
``torch.randn``, i.e. from the GLOBAL RNG, which the passed ``generator`` does
not cover.

⚠️ A full training run was never affected — ``train()`` seeds globally. What was
broken is every **IN-PROCESS A/B**, which is exactly what an ablation harness
is: switching a term off moved the total by less than this noise, so *no*
masking / regularisation measure was attributable.

THE FOUR THINGS PROVED HERE, in the order that matters:

1. **Bit-exactness** with ``sigreg_generator`` — total AND every per-term
   tensor, across all four stages.
2. **The negative controls** — different seeds MUST differ, and the DEFAULT
   path MUST still be non-reproducible. Without these the suite passes
   vacuously (⭐ a vacuous pass has shipped two defects in this programme).
3. ⛔ **The default did not move**, proved against a **CONTENT-anchored**
   reference resolved from each file's own history — never ``HEAD`` (C75: a
   sibling's whole-index commit swept an in-progress file into ``HEAD`` and the
   guard silently became a self-comparison that would pass forever). v6F S-W is
   training on Thor from this exact code.
4. **The ENUMERATION itself**, dynamically. C74's failure mode is a list whose
   every entry is verified but whose completeness never was, so this does not
   spot-check a grep: it instruments ``torch``'s RNG surface and asserts that a
   full loss call with all levers on consumes the global stream ZERO times.
   A new un-seeded draw added anywhere in the loss path fails here.
"""
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
_ROOT = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.models.sigreg import SigReg, position_relaxed  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
from train_v6_staged import (  # noqa: E402
    V6LossWeights, o6_sigreg_loss, synthetic_train_batch, v6_loss_step)

STAGES = ("S-W", "S-T", "S-S", "S-J")

#: the stages whose weights actually switch ``o6`` on — the only ones the defect
#: could ever have reached. Asserted (not assumed) in
#: ``test_o6_is_live_in_exactly_the_stages_this_file_claims``.
O6_STAGES = ("S-W", "S-J")


# =========================================================================== #
# fixtures
# =========================================================================== #

def _sub_cfgs() -> dict:
    return dict(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                                  action_dim=3))


_BASE_KW = dict(
    d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32,
    f_blocks=1, aux_hidden=16, sigreg_slices=8, plan_steps=6, dt=0.1,
    op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0,
    hz_str=0.5, d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
    n_candidates=8)


def _build(seed: int = 0, **over) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(V6Config(**(copy.deepcopy(_BASE_KW) | _sub_cfgs() | over)))


def _maximal(seed: int = 0) -> V6Stack:
    """⛔ EVERY lever on. The minimal build never enters the selector, the
    anchor head, the band mask or the plan branch — and an enumeration that
    only walks the paths a default batch happens to take is not an enumeration
    (C74). ``sigreg_free_dims`` is non-zero so ``position_relaxed`` takes its
    OTHER branch too."""
    s = _build(seed, selector="goal", anchor_goal="snap_lat",
               goal_cat_args=True, goal_factored=True, goal_multilabel=True,
               n_anchors=8, n_lat_bins=4, plan_wta_eps=0.1, sigreg_free_dims=4)
    g = torch.Generator().manual_seed(3)
    steps = list(range(1, s.cfg.plan_steps + 1))
    s.anchor_head.load_anchor_table(
        torch.randn(s.cfg.n_anchors, len(steps), 2, generator=g) * 3.0,
        horizons=steps, dt=s.cfg.dt)
    return s


def _batch(stack: V6Stack) -> dict:
    b = synthetic_train_batch(stack, batch=2, k=4, seed=7)
    b["gt_wp"] = torch.zeros(2, 2, 2)
    return b


def _kw(stage: str, maximal: bool = False) -> dict:
    if not maximal:
        return dict(stage=stage, o1_k=2, o5_k=2, weights=V6LossWeights())
    return dict(stage=stage, o1_k=2, o5_k=2, o3_band_rows=1,
                o3_mode="context", o5_mode="linear-decay",
                weights=V6LossWeights(lambda_plan=1.0, w_select=1.0,
                                      w_anchor=1.0))


def _call(stack, batch, kw, *, sig_seed: int | None = 11, gen_seed: int = 5):
    """One loss call. ``sig_seed=None`` is the INCUMBENT (global-RNG) path.

    ⚠️ A FRESH generator per call, seeded identically — "the same generator"
    means the same STATE, and re-passing one live object advances it (which is
    correct behaviour, and is itself asserted below).
    """
    return v6_loss_step(
        stack, batch, generator=torch.Generator().manual_seed(gen_seed),
        sigreg_generator=(None if sig_seed is None
                          else torch.Generator().manual_seed(sig_seed)), **kw)


def _diff(a: dict, b: dict) -> list[str]:
    """Names of the tensors that are NOT bit-identical ('loss' + per term)."""
    bad = []
    if not torch.equal(a["loss"], b["loss"]):
        bad.append("loss")
    assert a["log"]["terms"] == b["log"]["terms"], "the TERM SET moved"
    bad += [t for t in a["log"]["terms"] if not torch.equal(a[t], b[t])]
    return bad


# =========================================================================== #
# the dynamic RNG instrument (used by the enumeration test)
# =========================================================================== #

_FUNCS = ("randn", "rand", "randint", "randperm", "normal", "bernoulli",
          "multinomial", "poisson", "randn_like", "rand_like", "randint_like")
_METHODS = ("normal_", "uniform_", "random_", "bernoulli_", "exponential_",
            "cauchy_", "log_normal_", "geometric_")
_DROPOUTS = ("dropout", "dropout1d", "dropout2d", "dropout3d",
             "alpha_dropout", "feature_alpha_dropout")


#: ⛔ the instrument's OWN frames. Without this exclusion ``_site`` reports
#: ITSELF as the culprit — it is the deepest in-repo frame at the moment it
#: walks the stack — and every future failure would name this file instead of
#: the code that drew. MEASURED while writing this: the negative control below
#: fired on exactly that, and the enumeration test still "passed" with a
#: useless label. Same family as CLAUDE.md's monitor-whose-filter-matches-its-
#: own-echo trap: keep the observer disjoint from the observed.
_INSTRUMENT_FRAMES = {"_site", "inner", "watch_global_rng"}


#: ⛔ Anchor on the COMPUTED repo root, never on the name "TanitAD": in the
#: off-Drive clone (wt-tanitad-local) no path contains that token, so the old
#: substring filter returned zero frames and the negative control failed in
#: any clone — measured 2026-08-18 while G: was down and the clone was primary.
_REPO_ROOT = str(Path(__file__).resolve().parents[2]).replace("\\", "/")


def _site() -> str:
    ours = [f for f in traceback.extract_stack()
            if f.filename.replace("\\", "/").startswith(_REPO_ROOT)
            and not (Path(f.filename) == Path(__file__)
                     and f.name in _INSTRUMENT_FRAMES)]
    if not ours:
        return "<outside the repo>"
    f = ours[-1]
    rel = f.filename.replace(chr(92), "/")[len(_REPO_ROOT):].lstrip("/")
    return f"{rel}:{f.lineno} ({f.name}) | {f.line}"


@contextmanager
def watch_global_rng():
    """Record every RNG call made WITHOUT a ``generator=`` while active.

    Patches the module-level names, which is what the call sites resolve at call
    time, so it sees draws anywhere in the call graph — including inside
    ``torch.nn`` — rather than only the ones a grep would have listed.
    """
    import torch.nn.functional as F
    hits: list[str] = []
    saved: list[tuple] = []

    def _wrap_fn(mod, name, orig):
        def inner(*a, **kw):
            if kw.get("generator") is None:
                hits.append(f"{mod.__name__}.{name} @ {_site()}")
            return orig(*a, **kw)
        return inner

    def _wrap_method(name, orig):
        def inner(self, *a, **kw):
            if kw.get("generator") is None:
                hits.append(f"Tensor.{name} @ {_site()}")
            return orig(self, *a, **kw)
        return inner

    def _wrap_dropout(name, orig):
        def inner(*a, **kw):
            p = kw.get("p", a[1] if len(a) > 1 else 0.5)
            training = kw.get("training", a[2] if len(a) > 2 else True)
            if training and float(p) > 0.0:
                hits.append(f"F.{name}(p={p}) @ {_site()}")
            return orig(*a, **kw)
        return inner

    for name in _FUNCS:
        f = getattr(torch, name, None)
        if f is not None:
            saved.append((torch, name, f))
            setattr(torch, name, _wrap_fn(torch, name, f))
    for name in _METHODS:
        f = getattr(torch.Tensor, name, None)
        if f is not None:
            saved.append((torch.Tensor, name, f))
            setattr(torch.Tensor, name, _wrap_method(name, f))
    for name in _DROPOUTS:
        f = getattr(F, name, None)
        if f is not None:
            saved.append((F, name, f))
            setattr(F, name, _wrap_dropout(name, f))
    try:
        yield hits
    finally:
        for obj, name, orig in saved:
            setattr(obj, name, orig)


# =========================================================================== #
# 0. the claim this file's scope rests on
# =========================================================================== #

def test_o6_is_live_in_exactly_the_stages_this_file_claims():
    """S-T and S-S carry no ``o6``, so the defect could never reach them —
    stated as an assertion rather than left as an assumption, because "that
    stage is unaffected" is the kind of sentence that goes stale silently."""
    for stage in STAGES:
        live = V6LossWeights().for_stage(stage).o6_sigreg > 0
        assert live == (stage in O6_STAGES), stage


# =========================================================================== #
# 1. ⭐ BIT-EXACTNESS
# =========================================================================== #

@pytest.mark.parametrize("stage", STAGES)
def test_same_sigreg_generator_gives_a_BIT_EXACT_total_and_every_term(stage):
    """The headline. ``torch.equal``, not ``approx`` — an ablation whose signal
    is smaller than the tolerance is unmeasurable, and 18.7 % on ``o6`` is far
    larger than any term-switch we intend to detect."""
    stack = _build()
    stack.train()                       # the regime the defect was measured in
    batch, kw = _batch(stack), _kw(stage)
    bad = _diff(_call(stack, batch, kw), _call(stack, batch, kw))
    assert bad == [], f"{stage}: NOT reproducible — {bad}"


@pytest.mark.parametrize("stage", STAGES)
def test_bit_exact_with_EVERY_LEVER_ON(stage):
    """Same claim on the maximal build (selector + anchor head + factored goal
    + band mask + non-zero ``sigreg_free_dims``), so the guard covers the
    branches the default batch never enters."""
    stack = _maximal()
    stack.train()
    batch, kw = _batch(stack), _kw(stage, maximal=True)
    bad = _diff(_call(stack, batch, kw), _call(stack, batch, kw))
    assert bad == [], f"{stage}: NOT reproducible with all levers — {bad}"


@pytest.mark.parametrize("stage", O6_STAGES)
def test_the_GRADIENT_is_reproducible_too_not_just_the_scalar(stage):
    """⭐ THE ONE AN ABLATION ACTUALLY CONSUMES. A harness compares UPDATES, not
    printed losses — a reproducible scalar over a non-reproducible gradient
    would still make every A/B noise. Bit-exact per parameter tensor.
    """
    stack = _build()
    stack.train()
    batch, kw = _batch(stack), _kw(stage)

    def grads():
        stack.zero_grad(set_to_none=True)
        _call(stack, batch, kw)["loss"].backward()
        return {n: p.grad.detach().clone()
                for n, p in stack.named_parameters() if p.grad is not None}

    a, b = grads(), grads()
    assert set(a) == set(b) and a, "no gradients reached the parameters"
    bad = [n for n in a if not torch.equal(a[n], b[n])]
    assert bad == [], f"{stage}: {len(bad)}/{len(a)} grads not reproducible: " \
                      f"{bad[:5]}"


def test_NEGATIVE_CONTROL_the_gradient_check_can_fire():
    """...and it is sensitive to the O6 stream, so it is really testing the
    seeded path and not just re-reading a cached graph."""
    stack = _build()
    stack.train()
    batch, kw = _batch(stack), _kw("S-W")

    def grads(seed):
        stack.zero_grad(set_to_none=True)
        _call(stack, batch, kw, sig_seed=seed)["loss"].backward()
        return {n: p.grad.detach().clone()
                for n, p in stack.named_parameters() if p.grad is not None}

    a, b = grads(11), grads(12)
    assert any(not torch.equal(a[n], b[n]) for n in a)


def test_reproducible_across_a_FRESH_STACK_with_the_same_build_seed():
    """A harness that rebuilds the model per arm (the honest way to ablate an
    architectural lever) must also land on the same numbers."""
    kw = _kw("S-J")
    a, b = _build(), _build()
    a.train(), b.train()
    bad = _diff(_call(a, _batch(a), kw), _call(b, _batch(b), kw))
    assert bad == [], f"a rebuilt stack diverged: {bad}"


# =========================================================================== #
# 2. ⛔ THE NEGATIVE CONTROLS — without these the file passes vacuously
# =========================================================================== #

@pytest.mark.parametrize("stage", O6_STAGES)
def test_NEGATIVE_CONTROL_different_sigreg_seeds_MUST_differ(stage):
    """If the fix had frozen the directions (a fixed buffer, or a seed baked
    into ``SigReg``), every test above would pass and SIGReg would be broken:
    the directions are resampled per call SPECIFICALLY to prevent adversarial
    anisotropic collapse. Different seeds must therefore still move ``o6``."""
    stack = _build()
    stack.train()
    batch, kw = _batch(stack), _kw(stage)
    a = _call(stack, batch, kw, sig_seed=11)
    b = _call(stack, batch, kw, sig_seed=12)
    assert "o6" in _diff(a, b), \
        f"{stage}: o6 did NOT move across sigreg seeds — the directions are "
    assert not torch.equal(a["loss"], b["loss"])


@pytest.mark.parametrize("stage", O6_STAGES)
def test_NEGATIVE_CONTROL_the_DEFAULT_path_is_still_non_reproducible(stage):
    """⛔ THE CONTROL THAT PROVES THE TEST MEASURES THE FIX. With
    ``sigreg_generator=None`` the incumbent global draw is unchanged, so two
    calls must STILL disagree. If this ever passes, the default moved — which
    is the thing that must not happen while v6F trains."""
    stack = _build()
    stack.train()
    batch, kw = _batch(stack), _kw(stage)
    assert "o6" in _diff(_call(stack, batch, kw, sig_seed=None),
                         _call(stack, batch, kw, sig_seed=None)), \
        f"{stage}: the DEFAULT path became reproducible — the incumbent moved"


def test_NEGATIVE_CONTROL_reusing_one_LIVE_generator_object_advances_it():
    """"The same generator" means the same STATE. Re-passing one live object
    is a different call, and it must behave like one — otherwise a harness that
    hoisted the generator out of its loop would silently get frozen directions.
    """
    stack = _build()
    stack.train()
    batch, kw = _batch(stack), _kw("S-W")
    g = torch.Generator().manual_seed(11)
    a = v6_loss_step(stack, batch, generator=torch.Generator().manual_seed(5),
                     sigreg_generator=g, **kw)
    b = v6_loss_step(stack, batch, generator=torch.Generator().manual_seed(5),
                     sigreg_generator=g, **kw)
    assert not torch.equal(a["o6"], b["o6"])


# =========================================================================== #
# 3. ⛔ THE GLOBAL STREAM IS NO LONGER TOUCHED — and the enumeration is proved
# =========================================================================== #

@pytest.mark.parametrize("stage", STAGES)
def test_the_loss_consumes_ZERO_global_RNG_when_seeded(stage):
    """Enumeration-free and airtight: fork the global RNG state around the call
    and compare bytes. Whatever the call graph contains, if the state did not
    move then nothing in it read the global stream."""
    stack = _maximal()
    stack.train()
    batch, kw = _batch(stack), _kw(stage, maximal=True)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    _call(stack, batch, kw)
    assert torch.equal(before, torch.random.get_rng_state()), \
        f"{stage}: something in the loss path still reads the GLOBAL RNG"


@pytest.mark.parametrize("stage", O6_STAGES)
def test_NEGATIVE_CONTROL_the_state_watcher_can_fire(stage):
    """The watcher above proves nothing unless it is capable of firing. On the
    DEFAULT path the global state must still move (that is the incumbent)."""
    stack = _maximal()
    stack.train()
    batch, kw = _batch(stack), _kw(stage, maximal=True)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    _call(stack, batch, kw, sig_seed=None)
    assert not torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("stage", STAGES)
def test_the_ENUMERATION_of_unseeded_draws_is_EMPTY(stage):
    """⛔ C74. Not a grep list — ``torch``'s whole RNG surface is instrumented
    and the loss is run with every lever on, so ANY un-generatored draw
    anywhere in the call graph (ours, or inside ``torch.nn``) is reported with
    its file:line. This is the test that keeps the enumeration true as the loss
    grows a sixth, seventh and eighth term."""
    stack = _maximal()
    stack.train()
    batch, kw = _batch(stack), _kw(stage, maximal=True)
    with watch_global_rng() as hits:
        _call(stack, batch, kw)
    assert hits == [], f"{stage}: un-seeded RNG in the loss path:\n" + \
        "\n".join(f"  - {h}" for h in dict.fromkeys(hits))


def test_NEGATIVE_CONTROL_the_enumerator_can_fire_AND_NAMES_THE_RIGHT_FILE():
    """...and it SEES the incumbent draw, at ``sigreg.py``. An instrument that
    reports 'clean' because it is blind reports clean forever.

    ⭐ The second assertion is not padding: this control caught the enumerator
    blaming ITSELF (``_site`` was the deepest in-repo frame), which left the
    positive test passing while its failure message pointed at the wrong file.
    A diagnostic that names the observer is worse than none.
    """
    stack = _maximal()
    stack.train()
    batch, kw = _batch(stack), _kw("S-W", maximal=True)
    with watch_global_rng() as hits:
        _call(stack, batch, kw, sig_seed=None)
    assert any("models/sigreg.py" in h for h in hits), hits
    assert not any("test_loss_determinism.py" in h for h in hits), \
        f"the enumerator is blaming ITSELF: {hits}"


def test_the_enumerator_restores_torch_on_the_way_out():
    """It monkeypatches a shared module; a leaked patch would poison every test
    that runs after this file."""
    before = (torch.randn, torch.rand, torch.Tensor.normal_)
    with watch_global_rng():
        pass
    assert (torch.randn, torch.rand, torch.Tensor.normal_) == before


# =========================================================================== #
# 4. ⛔ THE DEFAULT DID NOT MOVE — against a CONTENT-anchored reference
# =========================================================================== #

def _side_by_side(rel: str, marker: str, modname: str):
    """Import the newest revision of ``rel`` that does NOT contain ``marker``.

    ⛔ RESOLVED BY CONTENT, NOT BY ``HEAD`` (C75). ``HEAD`` moves under us — a
    sibling's whole-index commit swept an in-progress file into it — and a
    HEAD-relative identity test then compares a module with itself and passes
    forever. Walking the FILE's own history for the last revision lacking the
    change marker is stable however many commits land, and it is the
    semantically right reference: it IS the code the live run came from.

    Returns ``None`` when git cannot answer; the caller then SKIPS. A skipped
    test is honest, a self-comparison dressed as a real one is not.
    """
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", rel],
                             cwd=_ROOT, capture_output=True, timeout=180)
        if log.returncode != 0:
            return None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=_ROOT,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if marker.encode() in r.stdout:
                continue                       # already carries the change
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    tmp = Path(tempfile.mkdtemp()) / f"{modname}.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location(modname, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    mod._ref = ref
    return mod


def test_SIGREG_default_is_bit_identical_to_the_PRE_CHANGE_module():
    """⛔ THE ONE THAT PROTECTS THE LIVE RUN, at the module that changed.

    ``SigReg`` is parameter-free, so old and new are comparable directly: seed
    the global RNG identically and the statistic must land on the same bits —
    which also proves the un-seeded branch consumes the global stream in the
    same order and amount.
    """
    old = _side_by_side("stack/tanitad/models/sigreg.py", "sample_directions",
                        "sigreg_pre_determinism")
    if old is None:
        pytest.skip("git could not produce a pre-generator sigreg revision")
    z = torch.randn(16, 24, generator=torch.Generator().manual_seed(2))
    for slices, beta, free in ((8, 1.0, 0), (13, 0.7, 0), (8, 1.0, 5)):
        o, n = old.SigReg(slices, beta), SigReg(slices, beta)
        torch.manual_seed(9)
        lo = (old.position_relaxed(o, z, free) if free
              else o(z))
        torch.manual_seed(9)
        ln = (position_relaxed(n, z, free) if free else n(z))
        assert torch.equal(lo, ln), \
            f"SigReg MOVED against {old._ref} (slices={slices}, free={free})"
        # and the global stream was consumed identically, not merely to the
        # same value — a different draw COUNT desynchronises everything after.
        torch.manual_seed(9)
        (old.position_relaxed(o, z, free) if free else o(z))
        s_old = torch.random.get_rng_state().clone()
        torch.manual_seed(9)
        (position_relaxed(n, z, free) if free else n(z))
        assert torch.equal(s_old, torch.random.get_rng_state()), \
            f"the global RNG is consumed DIFFERENTLY than at {old._ref}"


@pytest.mark.parametrize("stage", STAGES)
def test_v6_loss_step_default_is_bit_identical_to_the_PRE_CHANGE_trainer(stage):
    """The same claim one level up, end to end: the PRE-``sigreg_generator``
    trainer against the current one with the parameter OMITTED. Both run
    against the CURRENT ``tanitad.models.v6`` — deliberate, so the MODEL is held
    fixed and only the LOSS code varies, which makes any difference mine.

    ⚠️ ``eval()`` + a re-seed per call: the incumbent path is non-reproducible
    BY CONSTRUCTION (that is the defect), so a guard run in ``train()`` here
    would fire on the very noise this file exists to remove.
    """
    old = _side_by_side("stack/scripts/train_v6_staged.py", "sigreg_generator",
                        "train_v6_staged_pre_determinism")
    if old is None:
        pytest.skip("git could not produce a pre-sigreg_generator trainer")
    stack = _build()
    stack.eval()
    batch = _batch(stack)
    kw = dict(stage=stage, o1_k=2, o5_k=2)
    torch.manual_seed(3)
    lo = old.v6_loss_step(stack, batch, weights=old.V6LossWeights(),
                          generator=torch.Generator().manual_seed(11), **kw)
    torch.manual_seed(3)
    ln = v6_loss_step(stack, batch, weights=V6LossWeights(),
                      generator=torch.Generator().manual_seed(11), **kw)
    assert _diff(lo, ln) == [], \
        f"{stage}: the DEFAULT loss MOVED against {old._ref}"
    _ADD_OK = {'o5_form', 'o6_rows', 'o6_row_renorm',
               # H-RANK-22: additive only -- records which O1 gradient-path
               # variant ran, so two arms differing only in --o1-detach-encoder
               # are distinguishable from their logs. Loss is unchanged.
               'o1_detach_encoder',
               # E-DEC-15: additive only -- records whether O1's separation term
               # stop-gradded the FACTUAL branch (LIT-3 / PhyLatent CASC:
               # "the factual prediction is treated as a STOP-GRADIENT
               # REFERENCE"). Two arms differing only in
               # --o1-stopgrad-factual must be distinguishable from their
               # logs. Loss, terms and RNG draw-count all verified
               # BIT-IDENTICAL on the default path for S-W and S-J.
               'o1_stopgrad_factual'}
    assert not (set(lo['log']) - set(ln['log'])), 'log keys REMOVED'
    assert (set(ln['log']) - set(lo['log'])) <= _ADD_OK, 'unexpected new log keys'
    assert torch.equal(torch.tensor(lo["log"]["loss"]),
                       torch.tensor(ln["log"]["loss"]))


def test_NEGATIVE_CONTROL_the_no_change_guard_CAN_fail():
    """The guard above compares two modules; prove the COMPARISON bites rather
    than the reference resolution quietly returning the same object. Passing a
    ``sigreg_generator`` changes ``o6`` — so the same ``_diff`` must report it.
    """
    stack = _build()
    stack.eval()
    batch, kw = _batch(stack), _kw("S-W")
    torch.manual_seed(3)
    a = v6_loss_step(stack, batch,
                     generator=torch.Generator().manual_seed(11), **kw)
    b = _call(stack, batch, kw, sig_seed=11, gen_seed=11)
    assert "o6" in _diff(a, b) and "loss" in _diff(a, b)


# =========================================================================== #
# 5. the API surface an ablation harness will lean on
# =========================================================================== #

def test_the_new_parameters_all_DEFAULT_to_the_incumbent():
    """A reproducibility switch that defaults ON is a silent behaviour change
    on a training run. Checked on the signatures an operator actually calls."""
    import inspect
    assert inspect.signature(v6_loss_step).parameters[
        "sigreg_generator"].default is None
    assert inspect.signature(o6_sigreg_loss).parameters[
        "generator"].default is None
    assert inspect.signature(SigReg.forward).parameters[
        "generator"].default is None
    assert inspect.signature(position_relaxed).parameters[
        "generator"].default is None


def test_generator_is_KEYWORD_ONLY_everywhere_it_was_added():
    """Positional would silently capture ``free_dims``-style arguments at the
    existing call sites in ``flagship_losses`` / ``dynamics_encoder``."""
    import inspect
    for fn in (SigReg.forward, position_relaxed):
        assert inspect.signature(fn).parameters["generator"].kind is \
            inspect.Parameter.KEYWORD_ONLY, fn


def test_o6_wrapper_and_the_raw_module_agree_under_the_same_generator():
    """``o6_sigreg_loss`` is a wrapper that exists so nobody re-implements the
    call; it must not have grown a second RNG path of its own."""
    stack = _build(sigreg_free_dims=4)
    z = torch.randn(32, stack.cfg.d_op,
                    generator=torch.Generator().manual_seed(4))
    a = o6_sigreg_loss(stack.sigreg, z, 4,
                       generator=torch.Generator().manual_seed(1))
    b = position_relaxed(stack.sigreg, z, 4,
                         generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_a_seeded_call_works_for_both_free_dims_branches():
    """``position_relaxed`` forwards the generator on BOTH sides of its
    ``free_dims`` branch — the ``<= 0`` path is the one every non-flagship
    caller takes, and forgetting it there is a one-word omission."""
    sr = SigReg(8, 1.0)
    z = torch.randn(16, 24, generator=torch.Generator().manual_seed(6))
    for free in (0, 5):
        a = position_relaxed(sr, z, free,
                             generator=torch.Generator().manual_seed(7))
        b = position_relaxed(sr, z, free,
                             generator=torch.Generator().manual_seed(7))
        assert torch.equal(a, b), f"free_dims={free} not reproducible"
