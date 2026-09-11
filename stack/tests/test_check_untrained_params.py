"""⛔ THE POST-TRAINING GATE FOR UNTRAINED PARAMETERS — guarded by MUTATION, not inspection.

⭐⭐ WHY THIS FILE EXISTS. The defect has surfaced FIVE times in this programme and was found
too late every single time: ``tac_goal_tok_head`` (11,286 params, ``grad_abs_sum`` exactly
0.00000 for all 40,284 steps of refcv5-v2), ``core.decoder.offset_head`` (6,160 at random init
in a PUBLISHED arm), ``scorer.goal_point`` (1,026 — zero-init AND zero gradient, so it emits
the constant origin), the >=GT truncation whose caller had no ``--gt-bar`` flag, and a
post-train spec guard that existed only inside its own test. **Every one passed its own unit
tests.** ⭐ *"Rollable and trained are different claims."*

``stack/scripts/check_untrained_params.py`` reads that off a checkpoint ANALYTICALLY: Adam
allocates per-parameter state lazily, so a registered parameter with no entry in
``opt["state"]`` never received a gradient. This file guards the gate.

⛔ AN AST CENSUS IS NOT A TEST. One in this programme read **0 suspects on BOTH the fixed and
the broken trainer**. Every claim here is made against a REAL checkpoint that a real
``torch.optim.Adam`` really stepped, and the detector is proved reachable by MUTATION ARMS
that reintroduce the defect and must go RED.

⛔ EVERY EXPECTATION IS A LITERAL, hand-derived from the fixture's architecture — never an
expression over the code under test. ``nn.Linear(4, 8)`` is 4*8 + 8 = **40** parameters
because that is what ``nn.Linear`` means, not because the gate says so. Re-running the
producer's own derivation and finding agreement measures determinism, not correctness.

⚠️ THE THREE FALSE-POSITIVE GENERATORS THIS FILE PINS, each of which would make the gate worse
than no gate:

1. **A probe that read nothing.** A count of 0 untrained from a gate that failed to read the
   checkpoint is indistinguishable from a genuine 0. Every zero asserted here is paired with a
   SAME-BREATH control that must read NON-ZERO.
2. **An optimizer that allocates no state at all.** Plain SGD has an EMPTY ``state`` dict, so
   the lazy-allocation argument would name *every* parameter untrained. The gate must return
   INCONCLUSIVE, and ``test_plain_sgd_is_INCONCLUSIVE_not_everything_untrained`` proves it.
3. **A guessed name.** The id->name mapping is the one inferential step. When the alignment is
   not forced the gate must REFUSE, and ``test_ambiguous_alignment_REFUSES_rather_than_guessing``
   drives it into that state deliberately.

⚠️ CPU ONLY, DELIBERATELY. The dev box's RTX 4060 is shared and Thor is running a sweep;
nothing here allocates on CUDA.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys

import pytest
import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

_GATE_PY = os.path.join(_STACK, "scripts", "check_untrained_params.py")


# ==========================================================================
# rig
# ==========================================================================
def _gate(name: str = "check_untrained_params_under_test", source: str | None = None,
          tmp_path=None):
    """Import the gate BY PATH (it is a script). ``source`` loads a mutated copy instead."""
    path = _GATE_PY
    if source is not None:
        assert tmp_path is not None, "a mutant needs a tmp_path to live in"
        path = os.path.join(str(tmp_path), name + ".py")
        io.open(path, "w", encoding="utf-8", newline="\n").write(source)
    if not os.path.exists(path):
        pytest.skip(f"gate not present at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source() -> str:
    return io.open(_GATE_PY, encoding="utf-8").read()


def _mutate(anchor: str, replacement: str) -> str:
    """Replace ``anchor`` exactly once.

    ⛔ The count is asserted, so a refactor that removes the anchor makes the mutation arm go
    RED rather than silently vacuous. A mutation arm that can no longer find the line it
    mutates is a test that has quietly stopped testing — the same family as a guard that
    shares the defect it checks for.
    """
    src = _source()
    assert src.count(anchor) == 1, (
        f"mutation anchor is not unique ({src.count(anchor)} occurrences); the gate was "
        f"refactored and this mutation arm no longer reintroduces the defect:\n{anchor!r}")
    return src.replace(anchor, replacement)


def _run(mod, ckpt, out, *extra) -> tuple[int, dict]:
    """Run the gate and return (exit code, the JSON verdict it wrote)."""
    rc = mod.main([str(ckpt), "--out", str(out), "--quiet", *[str(a) for a in extra]])
    with io.open(str(out), encoding="utf-8") as fh:
        return rc, json.load(fh)


# --------------------------------------------------------------------------
# fixtures: real checkpoints, really stepped by a real optimizer
# --------------------------------------------------------------------------
class _OneOrphan(nn.Module):
    """trunk -> head is in the loss. ``orphan_head`` is BUILT, forward-runnable, registered on
    the module, counted in any parameter breakdown -- and ABSENT from the graph. It is the
    stand-in for ``tac_goal_tok_head`` under ``--w-tac-goal 0.0``.

    Parameter counts, hand-derived from the architecture:
        trunk        nn.Linear(4, 8)   4*8 + 8  =  40
        head         nn.Linear(8, 3)   8*3 + 3  =  27
        orphan_head  nn.Linear(8, 5)   8*5 + 5  =  45      <- never trained
                                                   ---
                                                   112
    """

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.head = nn.Linear(8, 3)
        self.orphan_head = nn.Linear(8, 5)

    def forward(self, x):
        return self.head(self.trunk(x))


class _AllReached(_OneOrphan):
    """The ANALYTIC CONTROL: identical parameters, but every one is in the loss."""

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h).sum() + self.orphan_head(h).sum()


class _TwoOrphans(nn.Module):
    """Two untrained heads that differ ONLY in their init, so ``exactly_zero`` has a
    discriminator inside the same checkpoint.

        trunk        nn.Linear(4, 8)   4*8 + 8  =  40
        head         nn.Linear(8, 3)   8*3 + 3  =  27
        orphan_rand  nn.Linear(8, 5)   8*5 + 5  =  45   default init -> NOT exactly zero
        orphan_zero  nn.Linear(8, 2)   8*2 + 2  =  18   zero-init    -> EXACTLY zero
                                                   ---
                                                   130
    """

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.head = nn.Linear(8, 3)
        self.orphan_rand = nn.Linear(8, 5)
        self.orphan_zero = nn.Linear(8, 2)
        nn.init.zeros_(self.orphan_zero.weight)
        nn.init.zeros_(self.orphan_zero.bias)

    def forward(self, x):
        return self.head(self.trunk(x))


class _WithBuffers(nn.Module):
    """Carries BatchNorm buffers whose shapes COLLIDE with real parameters, which is what
    makes the id->name alignment ambiguous once the buffer filter is removed.

        trunk   nn.Linear(4, 8)     40
        bn      nn.BatchNorm1d(8)   16   (+ running_mean(8), running_var(8), num_batches)
        head    nn.Linear(8, 3)     27
        orphan  nn.Linear(8, 5)     45
                                   ---
                                   128
    """

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.head = nn.Linear(8, 3)
        self.orphan = nn.Linear(8, 5)

    def forward(self, x):
        return self.head(self.bn(self.trunk(x)))


def _bake(model, tmp_path, name="ckpt.pt", steps=3, opt_cls=torch.optim.Adam, **opt_kw):
    """Really train ``model`` for ``steps`` and save a real checkpoint.

    ⛔ ``zero_grad(set_to_none=True)`` is deliberate and load-bearing: ``set_to_none=False``
    would materialise ZERO gradients on the orphan heads, Adam would then allocate state for
    them, and the fixture would no longer contain the defect it exists to carry.
    """
    torch.manual_seed(0)
    opt = opt_cls(model.parameters(), **opt_kw)
    x = torch.randn(4, 4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = out if out.ndim == 0 else out.sum()
        loss.backward()
        opt.step()
    path = os.path.join(str(tmp_path), name)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": steps}, path)
    return path


# ==========================================================================
# 1 — the gate NAMES the untrained parameter, by literal
# ==========================================================================
def test_gate_NAMES_the_known_untrained_parameter(tmp_path):
    """⛔ The core claim. A head that never entered the loss must be named, with its count.

    ⚠️ The same-breath control is inside this test: ``n_with_state`` must read **4** (non-zero).
    A gate that read nothing would report 0 untrained AND 0 with state, and the pair
    distinguishes the two.
    """
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    rc, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 112)

    assert r["status"] == "FAIL"                                   # literal
    assert rc == 1                                                 # convenience only

    names = sorted(p["name"] for p in r["untrained"]["params"])
    assert names == ["orphan_head.bias", "orphan_head.weight"]     # literal
    assert r["untrained"]["count_params"] == 2                     # literal
    assert r["untrained"]["count_numel"] == 45                     # literal: 8*5 + 5
    assert r["unexpected_untrained"]["count_numel"] == 45          # literal

    # same-breath controls: these MUST read non-zero, or the zero above is a claim about the probe
    assert r["optimizer"]["n_registered"] == 6                     # literal
    assert r["optimizer"]["n_with_state"] == 4                     # literal
    assert r["cross_checks"]["controls"]["auto"]["n_nonzero_optimizer_moment"] == 4   # literal


def test_the_artifacts_own_total_matches_the_rows_it_actually_contains(tmp_path):
    """⚠️ A tally line that disagrees with its own rows is how a truncated artifact reads as
    finished. MEASURED in this programme: a pytest run reported 24 rows under a tally of 41."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    _, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 112)
    rows = r["untrained"]["params"]
    assert len(rows) == r["untrained"]["count_params"]
    assert sum(p["numel"] for p in rows) == r["untrained"]["count_numel"]
    assert r["untrained"]["count_numel"] == 45                     # literal


# ==========================================================================
# 2 — THE ANALYTIC CONTROL: a fully trained checkpoint reads EXACTLY 0
# ==========================================================================
def test_a_fully_trained_checkpoint_reports_EXACTLY_zero_untrained(tmp_path):
    """⭐ The no-information value, read exactly. Every parameter is in the loss, so every one
    has optimizer state and the gate must find nothing.

    ⚠️ Paired with a control that must read NON-ZERO (``n_with_state == 6``), because a 0 from
    a gate that could not read the checkpoint is indistinguishable from a genuine 0.
    """
    g = _gate()
    ck = _bake(_AllReached(), tmp_path)
    rc, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 112)

    assert r["untrained"]["count_params"] == 0                     # literal: exactly zero
    assert r["untrained"]["count_numel"] == 0                      # literal: exactly zero
    assert r["status"] == "PASS"
    assert rc == 0

    assert r["optimizer"]["n_registered"] == 6                     # literal, NON-zero control
    assert r["optimizer"]["n_with_state"] == 6                     # literal, NON-zero control
    assert r["cross_checks"]["param_total"]["mapped"] == 112       # literal: 40 + 27 + 45


# ==========================================================================
# 3 — DELIBERATE REGRESSION ARMS: remove the detector, require RED
# ==========================================================================
_ANCHOR_DIFFERENCE = """        if id_shapes[k] is not None:
            continue"""

_ANCHOR_ALLOWLIST = "entry = next((e for e in allow if _matches(name, e)), None)"

_ANCHOR_ZERO = '"exactly_zero": (None if absmax is None else absmax == 0.0),'


def test_REGRESSION_ARM_removing_the_state_set_difference_goes_RED(tmp_path):
    """⛔ THE MUTATION THE BRIEF ASKS FOR. ``id_shapes[k] is not None`` IS the set difference
    between registered ids and ids with optimizer state — the whole analytic method. Remove it
    and the gate can no longer tell trained from untrained.

    The mutant is run against the SAME fixture that the real gate just failed on. It must now
    report a clean PASS with zero untrained — i.e. exactly the silent pass that shipped
    ``tac_goal_tok_head``. This test is GREEN only because the mutant is BLIND.
    """
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)

    # the real gate sees it
    _, real = _run(g, ck, tmp_path / "real.json", "--param-total-ref", 112)
    assert real["untrained"]["count_numel"] == 45                  # literal
    assert real["status"] == "FAIL"

    # the mutant does not
    mut = _gate("gate_mutant_no_difference",
                _mutate(_ANCHOR_DIFFERENCE, "        if True:\n            continue"),
                tmp_path)
    rc, bad = _run(mut, ck, tmp_path / "mutant.json", "--param-total-ref", 112)

    assert bad["untrained"]["count_params"] == 0, (
        "the mutation arm is INERT: the gate still found untrained params with its own "
        "detector removed, so this suite is not actually testing the detector")
    assert bad["status"] == "PASS"
    assert rc == 0
    # ⭐ the discriminating statement: same checkpoint, opposite verdicts
    assert real["status"] != bad["status"]


def test_REGRESSION_ARM_an_allow_everything_allowlist_goes_RED(tmp_path):
    """⛔ The allow-list is what keeps the gate switched ON, so its discrimination is mutated
    too. A gate that allows everything cannot tell DELIBERATELY FROZEN from ACCIDENTALLY
    UNREACHED, which is the failure mode that gets a gate deleted within a week."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    _, real = _run(g, ck, tmp_path / "real.json", "--param-total-ref", 112)
    assert real["unexpected_untrained"]["count_params"] == 2       # literal

    mut = _gate("gate_mutant_allow_all",
                _mutate(_ANCHOR_ALLOWLIST, 'entry = "<allow-everything>"'), tmp_path)
    rc, bad = _run(mut, ck, tmp_path / "mutant.json", "--param-total-ref", 112)

    assert bad["unexpected_untrained"]["count_params"] == 0, "the allow-list mutation is INERT"
    assert bad["status"] == "PASS"
    assert rc == 0
    assert real["status"] != bad["status"]


def test_REGRESSION_ARM_blinding_the_exactly_zero_flag_goes_RED(tmp_path):
    """⛔ ``scorer.goal_point`` is zero-init PLUS zero-gradient, so it emits the constant
    origin — an identity, not a noisy measurement. Blinding that flag must be detectable."""
    g = _gate()
    ck = _bake(_TwoOrphans(), tmp_path)
    _, real = _run(g, ck, tmp_path / "real.json", "--param-total-ref", 130)
    assert sum(1 for p in real["untrained"]["params"] if p["exactly_zero"]) == 2   # literal

    mut = _gate("gate_mutant_zero_blind",
                _mutate(_ANCHOR_ZERO, '"exactly_zero": False,'), tmp_path)
    _, bad = _run(mut, ck, tmp_path / "mutant.json", "--param-total-ref", 130)
    assert sum(1 for p in bad["untrained"]["params"] if p["exactly_zero"]) == 0, (
        "the exactly-zero mutation is INERT")


# ==========================================================================
# 4 — exactly-zero, with its discriminator in the same checkpoint
# ==========================================================================
def test_zero_init_AND_zero_gradient_is_flagged_exactly_zero(tmp_path):
    """⭐ ``orphan_zero`` is the ``scorer.goal_point`` shape: zero-init and never reached, so
    its values are STILL exactly zero — an analytic target, not an estimate.

    ⚠️ The same-breath discriminator is ``orphan_rand`` IN THE SAME CHECKPOINT: equally
    untrained, default-initialised, and therefore NOT exactly zero. Without it, an
    ``exactly_zero`` that was stuck at True would look like a finding.
    """
    g = _gate()
    ck = _bake(_TwoOrphans(), tmp_path)
    _, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 130)

    by = {p["name"]: p for p in r["untrained"]["params"]}
    assert sorted(by) == ["orphan_rand.bias", "orphan_rand.weight",
                          "orphan_zero.bias", "orphan_zero.weight"]        # literal
    assert r["untrained"]["count_numel"] == 63                             # literal: 45 + 18

    assert by["orphan_zero.weight"]["exactly_zero"] is True                # literal
    assert by["orphan_zero.weight"]["n_nonzero"] == 0                      # literal
    assert by["orphan_zero.bias"]["exactly_zero"] is True                  # literal

    # ⭐ the discriminator: an equally-untrained head that is NOT zero
    assert by["orphan_rand.weight"]["exactly_zero"] is False               # literal
    assert by["orphan_rand.weight"]["n_nonzero"] == 40                     # literal: 8*5
    assert by["orphan_rand.weight"]["absmax"] > 0.0


# ==========================================================================
# 5 — the allow-list: deliberately frozen vs accidentally unreached
# ==========================================================================
def test_an_allow_listed_frozen_head_PASSES_and_is_still_reported(tmp_path):
    """⭐ Deliberately frozen is not a silent pass: the parameter is still named in the JSON,
    with the allow-list entry that covered it. A gate that hides what it allowed is a gate
    nobody can audit."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    rc, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 112,
                 "--expect-frozen", "orphan_head")

    assert r["status"] == "PASS"
    assert rc == 0
    assert r["unexpected_untrained"]["count_params"] == 0                  # literal
    # still reported, not hidden -- and the NON-zero control for the line above
    assert r["untrained"]["count_params"] == 2                             # literal
    assert r["untrained"]["count_numel"] == 45                             # literal
    assert all(p["allowed"] for p in r["untrained"]["params"])
    assert {p["allowlist_entry"] for p in r["untrained"]["params"]} == {"orphan_head"}


def test_an_allow_list_that_misses_one_head_still_FAILS_on_that_head(tmp_path):
    """⛔ Partial coverage must not buy a pass. This is the refcv5-v2 shape: two heads known
    and allow-listed, a third nobody had looked at."""
    g = _gate()
    ck = _bake(_TwoOrphans(), tmp_path)
    rc, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 130,
                 "--expect-frozen", "orphan_rand")

    assert r["status"] == "FAIL"
    assert rc == 1
    assert sorted(r["unexpected_untrained"]["names"]) == [
        "orphan_zero.bias", "orphan_zero.weight"]                          # literal
    assert r["unexpected_untrained"]["count_numel"] == 18                  # literal: 8*2 + 2


def test_a_STALE_allow_list_entry_is_reported_and_fails_under_strict(tmp_path):
    """⚠️ An allow-list nobody prunes is how a gate rots into a rubber stamp. An entry that
    matches nothing untrained is reported always, and fails under ``--strict-allowlist``."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)

    _, lax = _run(g, ck, tmp_path / "lax.json", "--param-total-ref", 112,
                  "--expect-frozen", "orphan_head", "--expect-frozen", "head")
    assert lax["status"] == "PASS"
    assert lax["allowlist"]["unmatched"] == ["head"]                       # literal

    rc, strict = _run(g, ck, tmp_path / "strict.json", "--param-total-ref", 112,
                      "--expect-frozen", "orphan_head", "--expect-frozen", "head",
                      "--strict-allowlist")
    assert strict["status"] == "FAIL"
    assert rc == 1
    assert strict["allowlist"]["unmatched"] == ["head"]                    # literal


# ==========================================================================
# 6 — THE FALSE-POSITIVE GENERATORS: the gate must refuse, never invent
# ==========================================================================
def test_plain_sgd_is_INCONCLUSIVE_not_everything_untrained(tmp_path):
    """⛔⛔ THE WORST FAILURE AVAILABLE TO THIS GATE. Plain SGD (no momentum) allocates NO
    per-parameter state, so the lazy-allocation argument would name **every** parameter
    untrained — a confident, catastrophic false positive on a perfectly healthy run.

    The gate must detect that its own precondition does not hold and return INCONCLUSIVE.
    ⚠️ Same family as a probe that tunes on the data it scores: the method is valid, and its
    SCOPE is narrower than the question being asked of it.
    """
    g = _gate()
    ck = _bake(_AllReached(), tmp_path, opt_cls=torch.optim.SGD, lr=0.01)
    rc, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 112)

    assert r["status"] == "INCONCLUSIVE"
    assert rc == 2
    assert r["optimizer"]["n_with_state"] == 0                             # literal
    assert "untrained" not in r, "a gate with no valid precondition must name NOTHING"
    assert any("no per-parameter state" in x or "EMPTY" in x for x in r["reasons"])


def test_ambiguous_alignment_REFUSES_rather_than_guessing(tmp_path):
    """⛔ The id->name mapping is the ONE inferential step in the gate. With the buffer filter
    removed, BatchNorm's ``running_mean``/``running_var`` collide in shape with real
    parameters and the order-preserving embedding is no longer forced.

    The leftmost and rightmost embeddings then disagree, and the gate must say so. ⭐ An
    element's position is forced exactly when leftmost == rightmost, because every feasible
    embedding places it at >= leftmost and <= rightmost — so a disagreement is a PROOF of
    ambiguity, not a heuristic.

    ⚠️ Paired with the filtered run on the SAME checkpoint, which must be forced and name the
    orphan. Without that control, an always-INCONCLUSIVE gate would pass this test.
    """
    g = _gate()
    ck = _bake(_WithBuffers(), tmp_path)

    # control: with the declared buffer filter the alignment IS forced
    _, ok = _run(g, ck, tmp_path / "ok.json", "--param-total-ref", 128)
    assert ok["alignment"]["forced"] is True
    assert ok["alignment"]["n_ambiguous"] == 0                             # literal
    assert sorted(p["name"] for p in ok["untrained"]["params"]) == [
        "orphan.bias", "orphan.weight"]                                    # literal
    assert ok["untrained"]["count_numel"] == 45                            # literal

    # remove the filter -> buffers become candidates -> the embedding is no longer unique
    rc, amb = _run(g, ck, tmp_path / "amb.json", "--param-total-ref", 128, "--no-buffer-filter")
    assert amb["status"] == "INCONCLUSIVE"
    assert rc == 2
    assert amb["alignment"]["forced"] is False
    assert amb["alignment"]["n_ambiguous"] > 0
    assert "untrained" not in amb, "an ambiguous alignment must name NOTHING"
    assert any("AMBIGUOUS" in x for x in amb["reasons"])


def test_a_wrong_parameter_total_is_INCONCLUSIVE_never_PASS(tmp_path):
    """⛔ Cross-check 2 is an INDEPENDENTLY AUTHORED reference. If the mapped total disagrees
    with it the id->name mapping is wrong, so every name in the report is suspect — and the
    gate must not launder that into a PASS just because the allow-list covered the names."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    rc, r = _run(g, ck, tmp_path / "v.json", "--param-total-ref", 999,
                 "--expect-frozen", "orphan_head")

    assert r["status"] == "INCONCLUSIVE"
    assert rc == 2
    assert r["cross_checks"]["param_total"]["mapped"] == 112               # literal
    assert r["cross_checks"]["param_total"]["reference"] == 999            # literal
    assert r["cross_checks"]["param_total"]["ok"] is False


def test_a_named_control_that_reads_untrained_is_INCONCLUSIVE(tmp_path):
    """⚠️ A same-breath control exists to exclude a global-zero artifact. If the control itself
    reads untrained, the gate has no standing to report anything and must say so."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)

    _, ok = _run(g, ck, tmp_path / "ok.json", "--param-total-ref", 112, "--control", "head")
    assert ok["cross_checks"]["controls"]["named"][0]["ok"] is True        # NON-zero control

    rc, bad = _run(g, ck, tmp_path / "bad.json", "--param-total-ref", 112,
                   "--control", "orphan_head")
    assert bad["status"] == "INCONCLUSIVE"
    assert rc == 2
    assert bad["cross_checks"]["controls"]["named"][0]["ok"] is False


# ==========================================================================
# 7 — THE VERDICT IS THE JSON, NEVER THE EXIT CODE
# ==========================================================================
@pytest.mark.parametrize("case", ["missing", "garbage"])
def test_an_unreadable_checkpoint_emits_INCONCLUSIVE_and_still_writes_the_JSON(tmp_path, case):
    """⛔⛔ MEASURED in this programme: a 25-minute ``timeout`` killed a pre-launch gate with
    zero output and NO JSON written, and its wrapper printed ``GATE_EXIT=0``.
    ⭐ *"The admissible evidence that this gate did not run is the MISSING JSON."*

    So the gate must never emit PASS for something it could not read, and it must still leave
    an artifact saying so. Silence is reserved for "it never ran".
    """
    if case == "missing":
        ck = tmp_path / "does_not_exist.pt"
    else:
        ck = tmp_path / "garbage.pt"
        io.open(str(ck), "wb").write(b"this is not a torch checkpoint")

    g = _gate()
    out = tmp_path / "v.json"
    rc = g.main([str(ck), "--out", str(out), "--quiet"])

    assert os.path.exists(str(out)), "the gate must leave a verdict even when it cannot read"
    with io.open(str(out), encoding="utf-8") as fh:
        r = json.load(fh)
    assert r["status"] == "INCONCLUSIVE"
    assert rc == 2
    assert r["status"] != "PASS"
    assert r["reasons"], "an INCONCLUSIVE verdict must say WHY"


def test_the_verdict_json_is_written_atomically_leaving_no_tmp_behind(tmp_path):
    """⚠️ A crash mid-write must not leave a truncated artifact that reads as finished.
    MEASURED in this programme: ``NOISE_FLOOR.md`` died on a cp1252 ``UnicodeEncodeError``
    mid-write and carried no numbers at all while reading like a complete document."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    out = tmp_path / "v.json"
    _run(g, ck, out, "--param-total-ref", 112)
    assert os.path.exists(str(out))
    assert not os.path.exists(str(out) + ".tmp")


def test_the_verdict_is_pure_ASCII_so_a_cp1252_console_cannot_kill_it(tmp_path):
    """⚠️ The dev box is cp1252. An artifact that can only be written under
    ``PYTHONIOENCODING=utf-8`` is an artifact that will one day be truncated."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    out = tmp_path / "v.json"
    _run(g, ck, out, "--param-total-ref", 112)
    raw = io.open(str(out), "rb").read()
    assert raw.decode("ascii")          # raises if any byte is non-ASCII
    assert raw.endswith(b"\n")


def test_out_is_REQUIRED_because_the_json_is_the_verdict(tmp_path):
    """⛔ There is no default artifact path. A gate whose verdict is optional is a gate whose
    verdict will be skipped."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)
    with pytest.raises(SystemExit):
        g.main([str(ck), "--quiet"])


def test_a_missing_parameter_total_reference_must_be_WAIVED_explicitly(tmp_path):
    """⛔ Cross-check 2 cannot silently not-happen. With no ``config.json`` beside the
    checkpoint and no reference passed, the gate is INCONCLUSIVE until an operator says, in
    the argv and therefore in the record, that the check is waived."""
    g = _gate()
    ck = _bake(_OneOrphan(), tmp_path)

    rc, silent = _run(g, ck, tmp_path / "a.json")
    assert silent["status"] == "INCONCLUSIVE"
    assert rc == 2

    rc, waived = _run(g, ck, tmp_path / "b.json", "--no-param-total-ref")
    assert waived["status"] == "FAIL"          # it can now report the real finding
    assert rc == 1
    assert waived["cross_checks"]["param_total"]["waived"] is True
    assert waived["untrained"]["count_numel"] == 45                        # literal


# ==========================================================================
# 8 — the refcv5-v2 reproduction, when the real checkpoint is to hand
# ==========================================================================
_REFCV5V2 = [
    r"C:/Users/Admin/refcv5v2_final/ckpt.pt",
    r"D:/Projects/TanitAD-artifacts/refcv5v2_final/ckpt.pt",
]


def _refcv5v2_path():
    return next((p for p in _REFCV5V2 if os.path.exists(p)), None)


@pytest.mark.slow
def test_reproduces_refcv5v2_exactly(tmp_path):
    """⛔ THE PROOF ON THE REAL ARTIFACT. refcv5-v2 (md5 9405ec73b2d797c4cebd44f82dbce54b)
    carries THREE untrained heads totalling 18,472 parameters. Every number here is a literal
    transcribed from the independent finding at
    ``TanitAD Research Lab/Architecture & Inference/Implementation/incoming/
    2026-09-10-refcv5v2-zerograd-heads/FINDING.md`` — none is derived from this gate.

    ⚠️ 1.3 GB, loaded on CPU. Marked slow; skipped when the artifact is not on this box.
    """
    ck = _refcv5v2_path()
    if ck is None:
        pytest.skip("refcv5-v2 checkpoint not on this box")

    g = _gate()
    rc, r = _run(g, ck, tmp_path / "v.json")           # config.json is found beside the ckpt

    assert r["status"] == "FAIL"
    assert rc == 1
    assert r["optimizer"]["n_registered"] == 357                           # literal
    assert r["optimizer"]["n_with_state"] == 351                           # literal
    assert r["checkpoint"]["step"] == 40284                                # literal

    assert r["untrained"]["count_params"] == 6                             # literal
    assert r["untrained"]["count_numel"] == 18472                          # literal

    by = {p["name"]: p for p in r["untrained"]["params"]}
    assert sorted(by) == [                                                 # literal
        "core.decoder.offset_head.bias",
        "core.decoder.offset_head.weight",
        "scorer.goal_point.bias",
        "scorer.goal_point.weight",
        "tac_goal_tok_head.net.bias",
        "tac_goal_tok_head.net.weight",
    ]
    head = {}
    for n, p in by.items():
        head[n.rsplit(".", 1)[0]] = head.get(n.rsplit(".", 1)[0], 0) + p["numel"]
    assert head["core.decoder.offset_head"] == 6160                        # literal
    assert head["tac_goal_tok_head.net"] == 11286                          # literal
    assert head["scorer.goal_point"] == 1026                               # literal

    # scorer.goal_point is zero-init AND zero-gradient -> an identity, not an estimate
    assert by["scorer.goal_point.weight"]["exactly_zero"] is True          # literal
    assert by["scorer.goal_point.bias"]["exactly_zero"] is True            # literal
    # ⭐ the discriminator: the other two heads are equally untrained and NOT zero
    assert by["tac_goal_tok_head.net.weight"]["exactly_zero"] is False     # literal
    assert by["core.decoder.offset_head.weight"]["exactly_zero"] is False  # literal

    # the three cross-checks, as published in FINDING.md
    assert r["cross_checks"]["shape_match"] == {
        "matched": 351, "total": 351, "ok": True, "mismatches": []}        # literal
    assert r["cross_checks"]["param_total"]["mapped"] == 108257502         # literal
    assert r["cross_checks"]["param_total"]["reference"] == 108257502      # literal
    assert r["alignment"]["forced"] is True
    assert r["alignment"]["n_ambiguous"] == 0                              # literal
    assert [c["name"] for c in r["alignment"]["unconsumed_candidates"]] == [
        "core.lat_log_prior", "core.lon_log_prior",
        "core.decoder.anchor_controls"]                                    # literal
