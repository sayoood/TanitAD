"""Pin the COMPLETE per-token reach of the frozen v7 vocabulary — to SOURCE.

⭐⭐ WHY THIS FILE EXISTS. ``test_tactical_label_reach.py`` pinned ``D-TLIGHT-1``
— a label minted on 779 of 4,572 clips and reached by no loss — but it is scoped
to **four traffic-light tokens and four supervised heads**. The condition it
describes can hold for ANY token, and nothing noticed the first time. This file
pins the reach of the WHOLE vocabulary, by asserting the CONSUMING CODE.

⛔⛔ THE FINDING THIS FILE CARRIES (MEASURED 2026-09-07): **the tactical-goal
head is BUILT AND UNWIRED IN THE ARM THAT IS RUNNING.** ``refc_v3.py:1320``
computes ``cache["tac_goal_logits"]`` and nothing consumes it, so under the LIVE
refcv5-v2 argv the head's **11,286 parameters take no gradient at all**
(``p.grad is None`` on both tensors — the exact discriminator
``tac_goal_head.py``'s own docstring names for *"this head is not wired"*).

⚠️ **THIS CONFIRMS A REGISTERED ROW; IT DOES NOT DISCOVER ONE.**
``GOALS_AND_CLAIMS.md`` already carries **``D-TACGOAL-TRAINER-SEAM-OPEN``** —
*"supervising it still needs the two additive edits … and the MANEUVER_WEIGHT
budget decision — an owner/PI call"*. What is new is the **gradient** evidence
(not a code-reading), and the fact that the live run passes
``--tac-goal-tok-head`` and is therefore carrying the dead parameters now.

⚠️ Two sibling documents read the other way and will mislead a hurried reader:
the ``D-TACGOAL-1`` register row headlines *"the 22-token tactical goal set NOW
REACHES A SUPERVISED HEAD"*, and ``test_tactical_label_reach.py``'s docstring
says *"⭐⭐ GAP CLOSED 2026-09-06 — D-TACGOAL-1"*. Both are true of the **head**
and false of the **trainer**.

⇒ all 22 tactical GOAL tokens, the four traffic-light tokens included, are
``AUDIT_OR_METRIC_ONLY``: they reach ``V7Label.audit['goal_flags']`` and the
label census that reads it, and they reach no loss.

⚠️ WHY BOTH EXISTING GUARDS PASS ON THIS, and why a third was needed:
  * ``assert_seams_are_built`` asks *"is the head BUILT?"* — it is (``built:
    true`` in the live ``config.json``), so it passes;
  * ``effective_weights_stamp_v3`` enumerates **declared loss weights** and asks
    which build a graph. The tactical-goal head has **no weight flag at all**,
    so it is not among the 5 terms the live record stamps — the instrument built
    to catch *"a weight whose gate is shut"* is structurally blind to *"a head
    with no weight"*.
  A guard that shares the defect it checks for is green forever; the missing
  question was *"does a gradient actually land on it?"*

⛔ THE RULE THIS FILE OBEYS: **A CROSS-CHECK MUST BE DERIVED INDEPENDENTLY OF
THE VALUE IT CHECKS.** Every expectation below is a LITERAL, never an expression
over the code under test, and every detector carries a mutation control that
must make it fire.

MEASURED STATE, 2026-09-07 (blob ``s2_labels_v7.2_{train,eval}.jsonl.gz``,
md5 train ``0ff902130ce76886b8a925eceed9e3a5`` / eval
``aa12c948f062181c3297265b51526ec5``; 4,572 + 147 = 4,719 records)::

    TRAINING_SIGNAL       23
    INFERENCE_INPUT        3   (the nav commands — an INPUT by PI ruling)
    AUDIT_OR_METRIC_ONLY  18   (every tactical GOAL token that is not also an
                                action name — the D-TLIGHT-1 condition)
    UNREACHED              0
    NOT_EMITTED            8   (== vocab_v7.NOT_YET_EXTRACTABLE, exactly)

Full census + mutation logs:
``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-07-v7-vocab-reach-census/``.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
_REPO = os.path.dirname(_STACK)                             # <repo>
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

_CENSUS_PY = os.path.join(_STACK, "scripts", "v7_vocab_reach_census.py")


def _census():
    """The census module BY PATH — it is a script, like ``refc_v3_train.py``."""
    if not os.path.exists(_CENSUS_PY):
        pytest.skip(f"census not present at {_CENSUS_PY}")
    spec = importlib.util.spec_from_file_location("v7_vocab_reach_census_test",
                                                  _CENSUS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# THE MEASURED STATE, AS LITERALS. ⛔ Never an expression over the code.        #
# --------------------------------------------------------------------------- #
#: surface id -> the status ``assert_consumers`` must report on 2026-09-07.
EXPECTED_CONSUMER_STATUS: dict[str, str] = {
    "tac_lat_ce": "CONSUMED",
    "tac_lon_ce": "CONSUMED",
    "str_goal_ce_v6": "CONSUMED",
    "str_action_ce_v6": "CONSUMED",
    "nav_input": "CONSUMED",
    "goal_audit": "CONSUMED",
    # ⭐⭐ WAS A GAP, CLOSED 2026-09-09 (D-TACGOAL-TRAINER-SEAM-OPEN).
    # `refc_v3_train.py` now calls `tac_goal_head.tac_goal_loss` on
    # `out["tac_goal_logits"]` behind `--w-tac-goal`, and a real backward puts
    # grad_abs_sum 3.602122873067856 on the head's two tensors (MEASURED at
    # smoke width; Research/2026-09-09-tacgoal-wiring/RESULT.md).
    # ⛔ This literal is NOT edited to make a test pass: `assert_consumers`
    # measures "CONSUMED" from its own detector, which is a derivation
    # independent of this map.
    # ⚠️ CONSUMED means A CONSUMER EXISTS, never that an ARM BOUGHT IT.
    # `--w-tac-goal` defaults to 0.0 and no live recipe passes it, so the 22
    # tokens are REACHABLE, not yet TRAINED. Reading this row as "the tokens
    # are a training signal in the live run" is the D-TACGOAL claim one step
    # too far -- the arm-level fact lives in config.json's effective-weights
    # block, which now carries a `--w-tac-goal` row for exactly this reason.
    "tac_goal_bce": "CONSUMED",
    # ⛔ THE TWO REMAINING GAPS. Each is a finding, and each has its own test.
    "tac_action_ce_v6": "NO_CONSUMER",   # v6 lands the ids, applies no CE
    "str_ce_refc": "NO_CONSUMER",        # refc has no strategic TOKEN head
}

#: The documents that carry these claims. A failure message that does not name
#: them turns a mechanical fix into a re-derivation.
DOCS_CARRYING_THE_CLAIM = (
    "Project Steering/GOALS_AND_CLAIMS.md  (D-TLIGHT-1, D-TACGOAL-1)",
    "Project Steering/GOALS_AND_CLAIMS.md  (D-TACGOAL-TRAINER-SEAM-OPEN — "
    "the row this file measures)",
    "stack/tests/test_tactical_label_reach.py  (its docstring says "
    "'GAP CLOSED 2026-09-06' — true of the HEAD, false of the TRAINER)",
    "TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-09-07-v7-vocab-reach-census/RESULT.md",
)


def _msg(what: str) -> str:
    docs = "\n".join(f"      - {d}" for d in DOCS_CARRYING_THE_CLAIM)
    return (f"\n  {what}\n"
            f"  If this is a DELIBERATE change, update BOTH this test and the\n"
            f"  documents below in the SAME commit:\n{docs}\n")


def _read(rel: str) -> str:
    p = os.path.join(_REPO, rel)
    if not os.path.exists(p):
        pytest.skip(f"{rel} not present")
    return open(p, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# 1 — the surface registry and the class ladder are LITERALS                    #
# --------------------------------------------------------------------------- #
def test_the_class_ladder_is_exactly_the_five_named_classes() -> None:
    C = _census()
    assert C.ALL_CLASSES == ("TRAINING_SIGNAL", "INFERENCE_INPUT",
                             "AUDIT_OR_METRIC_ONLY", "UNREACHED",
                             "NOT_EMITTED"), _msg(
        "the census class ladder changed; every downstream literal in this "
        "file and in RESULT.md is written against the five above")


def test_every_surface_declares_a_trainer_a_control_and_a_kind() -> None:
    C = _census()
    for sid, spec in C.SURFACES.items():
        for key in ("kind", "trainer", "markers", "control", "enabled_by"):
            assert key in spec, _msg(f"surface {sid!r} lost its {key!r}")
        assert spec["kind"] in ("loss_target", "model_input", "audit"), _msg(
            f"surface {sid!r} has an unknown kind {spec['kind']!r}")
        assert spec["markers"], _msg(
            f"surface {sid!r} has NO markers — it would report CONSUMED on any "
            f"file, which is a check that cannot fail")


# --------------------------------------------------------------------------- #
# 2 — the consumer assertions, pinned against the repo                          #
# --------------------------------------------------------------------------- #
def test_consumer_assertions_match_the_measured_state() -> None:
    """⛔ Each mismatch is a finding, in EITHER direction.

    ``CONSUMED`` -> ``NO_CONSUMER`` is a regression: a loss stopped reading a
    vocabulary. ``NO_CONSUMER`` -> ``CONSUMED`` is the GOOD news this file is
    waiting for, and it still fails, because the claim register and the docs
    above describe the gap and must be corrected in the same commit.
    """
    from pathlib import Path
    C = _census()
    got = C.assert_consumers(Path(_REPO))
    incon = {s: r for s, r in got.items() if r["status"] == "INCONCLUSIVE"}
    if incon:
        pytest.skip(f"consumer files unreadable (mount?): {sorted(incon)}")
    actual = {s: r["status"] for s, r in got.items()}
    assert actual == EXPECTED_CONSUMER_STATUS, _msg(
        f"the consumer map moved:\n"
        f"    expected {EXPECTED_CONSUMER_STATUS}\n"
        f"    actual   {actual}")


def test_D_TACGOAL_TRAINER_SEAM_IS_CLOSED_the_trainer_calls_the_goal_loss(
) -> None:
    """⭐⭐ THE HEADLINE, INVERTED 2026-09-09. This test used to assert the seam
    was OPEN -- ``assert "tac_goal_loss" not in trainer`` -- and it was RIGHT
    for three days: the head, the loss and the emitter all existed and the
    trainer called none of them, so refcv5-v2 trained 11,286 parameters for
    40,284 steps at ``grad_abs_sum`` exactly 0.

    It now asserts the OPPOSITE, and it is still two-sided: if the call is ever
    removed again, this goes RED and names what to re-derive.

    ⭐ Every half is a POSITIVE assertion with a same-breath control, because an
    absence found through a channel that could not read is not an absence
    (``rg`` under-reports on the G: mount and still exits 0). ⛔ The control
    matters MORE in this direction: a file that failed to read yields ``sym not
    in trainer`` = True for every symbol, so the OLD form of this test would
    have passed on an unreadable file. The new form fails on one.
    """
    trainer = _read("stack/scripts/refc_v3_train.py")
    # -- the same-breath control: this file WAS read
    assert "def compute_losses_v3(" in trainer, _msg(
        "the control marker is missing from refc_v3_train.py — the read is "
        "not trustworthy, so the assertions below prove nothing")
    # -- the CALL, the FLAG and the TARGET, each named explicitly
    for sym, what in (
            ("_tac_goal_head.tac_goal_loss(", "the loss CALL"),
            ('ap.add_argument("--w-tac-goal"', "the weight FLAG"),
            ('"flag": "--w-tac-goal"', "the effective-weight GATE row"),
            ('item["tac_goal_y"]', "the dataset TARGET"),
            ("tac_goal_logits", "the head's logits"),
    ):
        assert sym in trainer, _msg(
            f"⛔ refc_v3_train.py no longer contains {what} ({sym!r}) — "
            f"D-TACGOAL-TRAINER-SEAM-OPEN HAS RE-OPENED. The 22 tactical "
            f"goal tokens would fall back from TRAINING_SIGNAL to "
            f"AUDIT_OR_METRIC_ONLY. Re-run the census "
            f"(stack/scripts/v7_vocab_reach_census.py) and update the claim "
            f"register and RESULT.md.")
    # -- the term must be ABSENT at weight zero, not multiplied by zero: that
    #    is what keeps the OFF path bit-identical to the pre-wiring trainer.
    assert "if _w_tg > 0.0:" in trainer, _msg(
        "the tac_goal term is no longer gated on a positive weight — if it "
        "is now multiplied by zero instead of skipped, the default arm's "
        "autograd graph differs from the pre-wiring trainer's and the "
        "bit-identity proof in RESULT.md no longer holds.")
    # -- and the pieces it depends on, so 'present' cannot be read as 'whole'
    head = _read("stack/tanitad/refs/tac_goal_head.py")
    assert "def tac_goal_loss(" in head and "class TacGoalTokenHead(" in head, \
        _msg("tac_goal_head.py lost the loss or the head — the trainer's call "
             "now points at nothing")
    model = _read("stack/tanitad/refs/refc_v3.py")
    assert 'cache["tac_goal_logits"] = self.tac_goal_tok_head(z_tac)' in model, \
        _msg("refc_v3.py no longer computes tac_goal_logits — the head is not "
             "even forward-run; re-derive the census before quoting it")


def test_refc_has_no_strategic_TOKEN_head_only_a_geometric_one() -> None:
    """The 15 strategic tokens reach NO loss in the refc line.

    ``--goal-str`` supervises ``g_str`` against a LAN bearing/distance target;
    ``str_goal_head`` is ``nn.Linear(d_ctx, 3)``, not a softmax over the eight
    ``STRATEGIC_GOAL_TOKENS_V7``. Their only training path in the repo is
    ``train_v6_staged.py``'s ``s2_goal_loss``, behind ``w_s2_goal`` (default 0).
    """
    model = _read("stack/tanitad/refs/refc_v3.py")
    assert "self.str_goal_head = nn.Linear(d_ctx, 3)" in model, _msg(
        "refc's strategic goal head is no longer the 3-wide geometric readout "
        "— if it became a token softmax the strategic tokens' class changes")
    trainer = _read("stack/scripts/refc_v3_train.py")
    assert "def compute_losses_v3(" in trainer            # same-breath control
    assert 'HEADS["str_goal"]' not in trainer, _msg(
        "⭐ refc_v3_train.py now sizes something on HEADS['str_goal'] — the "
        "strategic tokens may have gained a refc training path; re-run the "
        "census.")


def test_v6_lands_the_factored_tactical_ids_but_applies_no_loss() -> None:
    v6 = _read("stack/scripts/train_v6_staged.py")
    assert "V72_TACTICAL_BATCH_KEYS" in v6                # same-breath control
    assert 'cross_entropy(out["a_lat"]' not in v6, _msg(
        "⭐ train_v6_staged.py now applies a tactical CE — the pre-registered "
        "follow-up landed; re-run the census.")
    # ...and the strategic CE it DOES apply
    assert 's2_goal_loss(out["g_str"], out["a_str"], batch,' in v6, _msg(
        "train_v6_staged.py no longer calls s2_goal_loss — the strategic "
        "tokens have lost their ONLY training path in the repo")


def test_the_nav_command_is_an_INPUT_surface_never_a_loss_target() -> None:
    """⛔ BINDING (PI): the nav command simulates the vehicle's nav system. It
    is an INPUT. Classing it as a training signal would be wrong, not a
    finding."""
    C = _census()
    assert C.SURFACES["nav_input"]["kind"] == "model_input", _msg(
        "the nav surface changed kind — a nav TARGET would be the flagship-v1 "
        "route echo (a bijection of its own input that scored 1.0000)")
    for sid, spec in C.SURFACES.items():
        if spec["kind"] == "loss_target":
            assert "nav" not in sid, _msg(
                f"surface {sid!r} makes the nav command a loss target")


# --------------------------------------------------------------------------- #
# 3 — ⭐ THE DETECTORS MUST FIRE. A census that cannot go RED proves nothing.   #
# --------------------------------------------------------------------------- #
def _mutant_repo(tmp_path, rel: str, old: str, new: str):
    """Copy the three consumer files into ``tmp_path`` and mutate one of them."""
    import shutil
    for r in ("stack/scripts/refc_v3_train.py",
              "stack/scripts/train_v6_staged.py",
              "stack/scripts/tactical_label_census.py"):
        src = os.path.join(_REPO, r)
        if not os.path.exists(src):
            pytest.skip(f"{r} not present")
        dst = tmp_path / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    target = tmp_path / rel
    s = target.read_text(encoding="utf-8")
    assert old in s, ("the string to mutate is not present — the mutation "
                      "would be VACUOUS and would prove nothing")
    target.write_text(s.replace(old, new), encoding="utf-8")
    return tmp_path


def test_the_consumer_detector_actually_fires(tmp_path) -> None:
    """Reintroduce the historical defect — a head detached from its loss — and
    prove the census flips that surface to ``NO_CONSUMER``."""
    C = _census()
    root = _mutant_repo(tmp_path, "stack/scripts/refc_v3_train.py",
                        'F.cross_entropy(out["lat_logits_tac"], lat_t)',
                        "torch.zeros(())  # MUTATED")
    got = C.assert_consumers(root)
    assert got["tac_lat_ce"]["status"] == "NO_CONSUMER", (
        "the consumer detector did NOT fire on a trainer whose tactical-lat CE "
        "has been deleted — the detector is dead and every CONSUMED verdict "
        "from it is uninformative")
    # the same-breath control: the OTHER surfaces are untouched
    assert got["tac_lon_ce"]["status"] == "CONSUMED", (
        "the mutation flipped an unrelated surface — the detector is not "
        "specific, so its verdicts cannot be attributed")


def test_a_failed_read_is_INCONCLUSIVE_and_never_an_absence(tmp_path) -> None:
    """⛔ '0 hits' is a claim about the SEARCH unless the read is asserted.

    Break the same-breath CONTROL and leave the markers alone: the surface must
    report INCONCLUSIVE, not NO_CONSUMER. On this mount a search tool returns
    'no matches' for a file it could not open, and that is indistinguishable
    from a genuine absence without a control.
    """
    C = _census()
    root = _mutant_repo(tmp_path, "stack/scripts/refc_v3_train.py",
                        "def compute_losses_v3(", "def RENAMED_losses_v3(")
    got = C.assert_consumers(root)
    assert got["tac_lat_ce"]["status"] == "INCONCLUSIVE", (
        "a surface whose control marker is gone reported a definite verdict — "
        "an unreadable file would then be indistinguishable from a real "
        "absence, which is the exact failure this control exists to prevent")


def test_the_census_goes_GREEN_when_the_gap_is_actually_closed(tmp_path) -> None:
    """⭐ THE OTHER DIRECTION, and it matters as much.

    A detector stuck at NO_CONSUMER would keep reporting D-TACGOAL-1 open after
    somebody fixed it. Add the two calls a real fix must make and prove the
    surface flips to CONSUMED.
    """
    C = _census()
    root = _mutant_repo(
        tmp_path, "stack/scripts/refc_v3_train.py",
        "def compute_losses_v3(",
        "# tac_goal_loss / TacGoalEmitter  (MUTATED: fix-forward)\n"
        "def compute_losses_v3(")
    got = C.assert_consumers(root)
    assert got["tac_goal_bce"]["status"] == "CONSUMED", (
        "the census cannot see a trainer that DOES wire the tactical-goal "
        "loss — it would report the gap open forever, including after a fix")


def test_the_class_resolver_is_ordered_and_reachable() -> None:
    """Every class must be reachable from ``_class_of`` — a ladder rung that no
    input can produce is a class that will never be reported."""
    C = _census()
    assert C._class_of(set(), 0) == "NOT_EMITTED"
    assert C._class_of({"loss_target"}, 5) == "TRAINING_SIGNAL"
    assert C._class_of({"model_input"}, 5) == "INFERENCE_INPUT"
    assert C._class_of({"audit"}, 5) == "AUDIT_OR_METRIC_ONLY"
    assert C._class_of(set(), 5) == "UNREACHED"
    # strength order: a loss beats an input beats an audit
    assert C._class_of({"loss_target", "model_input", "audit"}, 5) \
        == "TRAINING_SIGNAL"
    assert C._class_of({"model_input", "audit"}, 5) == "INFERENCE_INPUT"
