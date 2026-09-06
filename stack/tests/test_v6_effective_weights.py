"""⛔⛔ AN OPERATOR'S EXPLICIT LAUNCH FLAG IS SILENTLY DISCARDED.

THE DEFECT (MEASURED 2026-09-06)
--------------------------------
``V6LossWeights.for_stage(stage)`` rewrites the argparse defaults per stage.
``--stage S-T --w-o5 1.0`` therefore **trains nothing on O5**, stamps
``w_o5: 1.0`` into ``config.json``, and says nothing.  ⭐ That is the mechanism
behind the already-measured v7f defect: **42 of 138 optimizer tensors received
no gradient — 5,305,667 params = 52.2 % of a declared trainable budget** —
because a zero-weighted term is *guarded out* of the loss and its modules never
enter the autograd graph.  ⭐ ``p.grad is None`` is the sound discriminator; the
weight's value never was.

⚠️ THE TWO HALVES THIS FILE HAS TO KEEP APART, or the guard is worse than the
silence it replaces:

* a **default** of 0.0 is FINE — the zero defaults exist so that adding a seam
  to the code cannot change a run that does not ask for it;
* a **stage** zeroing a term the operator never mentioned is the staged ladder
  **working as designed**;
* ⛔ only a value the operator **ASKED FOR** being discarded is a refusal.

So the RED tests here are meaningless without the GREEN controls beside them,
and three of the controls exist purely to prove the guard is not refusing
everything.  ⭐ *A guard that refuses everything gets deleted; one that refuses
nothing measured nothing.*

⭐⭐ AND CORRECTNESS IS NOT WIRING.  A guard was recently found in this
programme **written, working when called, and called from one launch path of
two**.  ``test_preflight_is_on_every_v6_launch_path`` and
``test_refc_guard_is_on_the_real_path`` exist for that alone — the second one
matters because ``refc_v3_train.main`` runs ``preflight`` **only** under
``--preflight`` and otherwise calls ``train`` directly.
"""
from __future__ import annotations

import shlex
import sys
from dataclasses import fields
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad import effective_weights as ew            # noqa: E402
import train_v6_staged as V6                           # noqa: E402
import refc_v3_train as V3                             # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _v6(argv: list[str]):
    ap = V6.build_parser()
    a = ap.parse_args(argv)
    a._ew_parser, a._ew_argv = ap, list(argv)
    return a


def _v6_problems(argv: list[str]) -> list[str]:
    """ONLY the effective-weight refusals, so unrelated preflight noise cannot
    make a one-sided guard look two-sided."""
    return V6._preflight_effective_weights(_v6(argv))


def _st(*extra: str) -> list[str]:
    return ["--stage", "S-T", "--out", "/tmp/run",
            "--init-from", "/tmp/prev.pt", *extra]


def _v3(argv: list[str]):
    ap = V3.build_parser()
    a = ap.parse_args(argv)          # ⚠️ outside any try: an argparse
    a._ew_parser, a._ew_argv = ap, list(argv)   # SystemExit is not a refusal
    return a


def _v3_refusal(argv: list[str]) -> str | None:
    a = _v3(argv)
    try:
        V3.check_effective_weights(a)
        return None
    except SystemExit as e:
        return str(e)


# ---------------------------------------------------------------------------
# the zeroing table is DERIVED, so it cannot go stale
# ---------------------------------------------------------------------------

def test_stage_zeroing_is_derived_from_for_stage_itself():
    """A hand-copied table is a second source of truth, and this programme has
    measured repeatedly that the copy rots (the '2 of 36 features' count went
    stale FOUR times, inside the rule warning about stale counts)."""
    for stage in V6.STAGES:
        derived = V6.stage_zeroed_terms(stage)
        probe = V6.V6LossWeights(**{n: 1.0 for n in V6._W_FLOAT_TERMS})
        after = probe.for_stage(stage)
        truth = {n for n in V6._W_FLOAT_TERMS
                 if float(getattr(after, n)) == 0.0}
        assert derived == truth, f"{stage}: derived table disagrees with for_stage"


def test_the_measured_zeroing_counts():
    """MEASURED 2026-09-06. ⚠️ S-T zeroes TEN terms, not the nine an earlier
    enumeration reported: `w_s1_multi` was missed. S-S was not enumerated at
    all and zeroes FOURTEEN."""
    assert len(V6.stage_zeroed_terms("S-W")) == 10
    assert len(V6.stage_zeroed_terms("S-T")) == 10
    assert "w_s1_multi" in V6.stage_zeroed_terms("S-T")
    assert len(V6.stage_zeroed_terms("S-S")) == 14
    assert V6.stage_zeroed_terms("S-J") == frozenset()


# ---------------------------------------------------------------------------
# EXHAUSTIVENESS -- a new term cannot ship unaudited
# ---------------------------------------------------------------------------

def test_every_float_loss_weight_is_in_the_audit():
    """⛔ Adding a float term to `V6LossWeights` without listing it in
    `W_TERM_FLAGS` fails HERE, not in an audit months later."""
    missing = set(V6._W_FLOAT_TERMS) - set(V6.W_TERM_FLAGS)
    assert not missing, f"unaudited loss weights: {sorted(missing)}"
    extra = set(V6.W_TERM_FLAGS) - set(V6._W_FLOAT_TERMS)
    assert not extra, f"W_TERM_FLAGS names non-terms: {sorted(extra)}"


def test_argparse_defaults_agree_with_the_dataclass_defaults():
    """The table calls the dataclass default 'the declared default'. If the
    flag's own default disagreed, the table would be quoting a number the
    operator never sees."""
    ap = V6.build_parser()
    declared = V6.V6LossWeights()
    for term, (_flag, dest) in V6.W_TERM_FLAGS.items():
        if dest is None or term == "lambda_plan":
            continue                       # no flag / a three-layer resolver
        assert float(ap.get_default(dest)) == float(getattr(declared, term)), (
            f"{term}: argparse default != dataclass default")


def test_seam_op_has_no_flag_and_the_table_says_so():
    """⚠️ `_weights_from_args` never sets `seam_op`, so it is always the
    dataclass 1.0 and then zeroed in S-W/S-S. It is LISTED rather than omitted:
    a term absent from the table reads as a term that does not exist."""
    assert V6.W_TERM_FLAGS["seam_op"][1] is None
    rows, _ = V6.effective_weight_rows(_v6(_st()))
    seam = [r for r in rows if r.term == "seam_op"][0]
    assert seam.explicit is False and "no flag" in seam.flag


# ---------------------------------------------------------------------------
# ⛔ THE REFUSAL -- two-sided
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag,value", [("--w-o5", "1.0"),
                                        ("--w-o1-ctrl", "1.0"),
                                        ("--w-o3", "1.0"),
                                        ("--w-s1", "1.0")])
def test_RED_refuses_a_weight_the_stage_zeroes(flag, value):
    """The defect, reintroduced."""
    got = _v6_problems(_st(flag, value))
    assert got, f"{flag} on S-T was silently discarded -- the defect is back"
    assert flag in got[0] and "for_stage" in got[0]


def test_GREEN_same_flag_passes_on_a_stage_that_keeps_it():
    """⭐ The control that makes the RED half mean something. S-J keeps every
    term, so the identical flag must PASS."""
    assert _v6_problems(["--stage", "S-J", "--out", "/tmp/run",
                         "--init-from", "/tmp/p.pt", "--w-o5", "1.0"]) == []


def test_GREEN_a_silent_operator_is_the_ladder_working():
    """S-T zeroes `o5_rollout` from its 1.0 default every single time. That is
    the staged protocol, not a defect, and refusing it would fire on every
    honest launch -- which is how a flag gets deleted."""
    assert _v6_problems(_st()) == []


def test_GREEN_an_explicit_zero_is_declared_not_discarded():
    """`--w-o5 0` on S-T: the operator asked for zero and got zero."""
    assert _v6_problems(_st("--w-o5", "0")) == []


def test_explicitness_comes_from_argv_and_not_from_the_value():
    """⛔ THE LOAD-BEARING DISTINCTION. `--w-o5 1.0` is the DEFAULT value, so a
    value comparison cannot tell it from silence -- and the two must produce
    opposite verdicts."""
    assert float(V6.build_parser().get_default("w_o5")) == 1.0
    passed = _v6(_st("--w-o5", "1.0"))
    silent = _v6(_st())
    assert getattr(passed, "w_o5") == getattr(silent, "w_o5") == 1.0
    assert _v6_problems(_st("--w-o5", "1.0")) != []
    assert _v6_problems(_st()) == []


def test_the_escape_hatch_passes_and_is_recorded():
    """⭐ `--refuse-unreached` shipped WITH `--allow-unreached` because a bare
    refusal that fires on honest launches gets deleted. The override does not
    hide the finding: it is stamped."""
    assert _v6_problems(_st("--w-o5", "1.0", "--allow-discarded-weights")) == []
    stamp = V6.effective_weights_stamp(
        _v6(_st("--w-o5", "1.0", "--allow-discarded-weights")))
    assert stamp["acknowledged_discarded"] is True


# ---------------------------------------------------------------------------
# ⚠️ A MASKED TERM IS NOT A DEAD TERM
# ---------------------------------------------------------------------------

def test_a_masked_term_is_distinguished_from_a_zeroed_one():
    """MEASURED this week: the route head's apparent zero gradient was a
    VALIDITY MASK, not a dead head -- forcing `route_valid=True` moved the loss
    0.0 -> 0.687. A table that could not tell a masked term from a zeroed one
    would manufacture false alarms, which is worse than the silence."""
    rows, _ = V6.effective_weight_rows(
        _v6(["--stage", "S-S", "--out", "/tmp/run", "--init-from", "/tmp/p.pt",
             "--w-s2-goal", "1.0", "--s2-labels", "/tmp/s2.jsonl"]))
    s2 = [r for r in rows if r.term == "w_s2_goal"][0]
    # S-S KEEPS w_s2_goal, the label precondition is met, and the term rides
    # `s2_valid` -- so it trains, CONDITIONALLY, and the row says which.
    assert s2.status == ew.TRAINS_IF_MASK
    assert s2.builds_graph is True
    assert "s2_valid" in (s2.mask or "")
    # ...and the SAME weight on S-T is a stage kill, not a mask question.
    rows_t, _ = V6.effective_weight_rows(
        _v6(_st("--w-s2-goal", "1.0", "--s2-labels", "/tmp/s2.jsonl")))
    s2t = [r for r in rows_t if r.term == "w_s2_goal"][0]
    assert s2t.status == ew.DISCARDED and s2t.builds_graph is False


def test_a_stamped_but_untrainable_head_is_refused():
    """⛔ The `--goal-point-inject --goal-point-w 0` family, generalised: a
    weight > 0 whose module was never built. Here `--w-select` with no scorer."""
    got = _v6_problems(["--stage", "S-J", "--out", "/tmp/run",
                        "--init-from", "/tmp/p.pt", "--w-select", "1.0"])
    assert any("selector" in g for g in got), got
    # control: build the scorer and the same weight is fine
    ok = _v6_problems(["--stage", "S-J", "--out", "/tmp/run",
                       "--init-from", "/tmp/p.pt", "--w-select", "1.0",
                       "--selector", "goal"])
    assert not any("no scorer" in g for g in ok), ok


def test_a_LABEL_precondition_is_exempt_under_dry_run_but_a_MODULE_one_is_not():
    """⛔ THE FALSE REFUSAL THIS GUARD ACTUALLY PRODUCED, PINNED.

    MEASURED 2026-09-06: the first version refused `--w-s2-goal 1` with no
    `--s2-labels` unconditionally and broke
    `test_v6_s2_loss.py::test_preflight_refuses_weight_without_labels_on_a_REAL_run`,
    whose second half asserts the OPPOSITE for a dry run -- *"a dry-run may
    smoke the loss on synthetic keys without labels"*.

    ⚠ The distinction is real, not a test quirk: a **module** precondition
    (`--selector none`) is a structural absence in every mode, while a **label**
    precondition only binds on a run that trains. ⭐ A guard that fires on an
    honest launch gets deleted -- caught here by an existing test, which is what
    the suite is for.
    """
    base = ["--stage", "S-S", "--out", "/tmp/run", "--init-from", "/tmp/p.pt"]
    # LABEL precondition: refuses on a real run...
    assert any("--s2-labels" in p
               for p in _v6_problems(base + ["--w-s2-goal", "1"]))
    # ...and is silent on a dry run.
    assert not any("--s2-labels" in p
                   for p in _v6_problems(base + ["--w-s2-goal", "1",
                                                 "--dry-run"]))
    # MODULE precondition: binding in BOTH modes -- a scorer that was never
    # built is absent from the dry run too.
    for extra in ([], ["--dry-run"]):
        got = _v6_problems(["--stage", "S-J", "--out", "/tmp/run",
                            "--init-from", "/tmp/p.pt", "--w-select", "1.0",
                            *extra])
        assert any("selector" in p for p in got), (extra, got)


def test_the_stamp_carries_the_evidence_a_later_audit_needs():
    stamp = V6.effective_weights_stamp(_v6(_st("--w-o5", "1.0")))
    assert stamp["explicit_source"] == ew.SRC_ARGV
    assert stamp["where"] == "--stage S-T"
    assert stamp["n_terms"] == len(stamp["terms"])
    o5 = [t for t in stamp["terms"] if t["term"] == "o5_rollout"][0]
    assert o5["declared"] == 1.0 and o5["effective"] == 0.0
    assert o5["explicit"] is True and o5["status"] == ew.DISCARDED
    assert o5["builds_graph"] is False


def test_unknown_explicitness_is_reported_never_guessed():
    """⚠️ Without the command line the audit CANNOT answer 'did the operator
    ask?'. It must say so -- inferring from the value is the error this whole
    module exists to avoid, and silently reading UNKNOWN as 'nothing was
    explicit' is how a guard becomes cover."""
    ap = V6.build_parser()
    bare = ap.parse_args(_st("--w-o5", "1.0"))     # no _ew_parser / _ew_argv
    rows, src = V6.effective_weight_rows(bare)
    assert src == ew.SRC_UNAVAILABLE
    assert all(r.explicit is None for r in rows)
    assert ew.unknown_explicitness_warning(rows) is not None
    assert V6._preflight_effective_weights(bare) == []   # cannot refuse blind


# ---------------------------------------------------------------------------
# ⭐⭐ WIRING -- correctness and wiring are DIFFERENT CLAIMS
# ---------------------------------------------------------------------------

def test_preflight_is_on_every_v6_launch_path(monkeypatch, tmp_path):
    """⛔ A guard written, working when called, and called from ONE launch path
    of two has been found in this programme. `main` must refuse BEFORE reaching
    either training entry point -- the real one AND the dry run."""
    reached: list[str] = []
    monkeypatch.setattr(V6, "train", lambda a: reached.append("train"))
    monkeypatch.setattr(V6, "dry_run", lambda a: reached.append("dry_run"))

    argv = _st("--w-o5", "1.0")                     # a DISCARDED launch
    assert V6.main(argv) == 2
    assert reached == [], f"training was entered anyway: {reached}"

    assert V6.main(argv + ["--dry-run"]) == 2
    assert reached == [], f"the dry-run path bypassed the guard: {reached}"


def test_main_calls_preflight_before_any_training_entry_point(monkeypatch):
    """The positive half: with the guard satisfied, `main` DOES proceed -- so
    the test above is measuring a refusal, not a broken `main`."""
    order: list[str] = []
    monkeypatch.setattr(V6, "preflight", lambda a: order.append("preflight") or [])
    monkeypatch.setattr(V6, "dry_run", lambda a: order.append("dry_run"))
    monkeypatch.setattr(V6, "train", lambda a: order.append("train"))
    assert V6.main(_st("--dry-run")) == 0
    assert order == ["preflight", "dry_run"]
    order.clear()
    assert V6.main(_st()) == 0
    assert order == ["preflight", "train"]


def test_the_table_and_stamp_ride_the_one_builder_both_paths_use(monkeypatch):
    """`_run_config` is called by the dry-run path AND the real path, so the
    stamp cannot be present on one and missing on the other."""
    src = Path(V6.__file__).read_text(encoding="utf-8")
    assert src.count('"effective_weights": effective_weights_stamp(') == 1
    assert src.count("cfg_json = _run_config(") == 2


def test_run_config_stays_JSON_SERIALISABLE_with_the_argv_carriers(monkeypatch):
    """⛔⛔ THE BUG THIS FILE CAUGHT ON ITSELF, AND IT WOULD HAVE KILLED A RUN.

    `main` attaches `_ew_parser` (an **ArgumentParser**) and `_ew_argv` to the
    namespace so the audit can read explicitness from argv. `_run_config`
    serialises `vars(a)` wholesale into `config.json`, so the carrier made
    `json.dumps` raise `TypeError: Object of type ArgumentParser is not JSON
    serializable` -- **at the config.json write**, i.e. AFTER the model is built
    and the corpus is mounted. A guard against silent non-training that itself
    crashes the run late is not an improvement.

    ⚠ The lesson is the programme's own: an instrument is not correct until
    its OUTPUT is exercised. The audit's table printed perfectly in isolation;
    only writing the record found this.
    """
    from types import SimpleNamespace
    import json

    cfg = SimpleNamespace(
        to_dict=lambda: {"ok": True}, plan_steps=60, dt=0.1, horizon_s=6.0,
        op_band_s=(0.0, 2.0), tac_band_s=(2.0, 6.0))
    stack = SimpleNamespace(cfg=cfg, param_report=lambda: {"total": 1})
    monkeypatch.setattr(V6, "run_provenance", lambda d: {"device": str(d)})

    a = _v6(_st("--w-o5", "1.0"))
    assert hasattr(a, "_ew_parser") and hasattr(a, "_ew_argv")
    cfg_json = V6._run_config(a, stack, freeze={}, decl_audit=None)

    json.dumps(cfg_json)                       # the assertion that matters
    assert not [k for k in cfg_json["args"] if k.startswith("_ew_")]
    # ...and the audit itself still made it into the record.
    assert cfg_json["effective_weights"]["explicit_source"] == ew.SRC_ARGV
    o5 = [t for t in cfg_json["effective_weights"]["terms"]
          if t["term"] == "o5_rollout"][0]
    assert o5["status"] == ew.DISCARDED and o5["builds_graph"] is False


def test_a_refusal_that_cannot_be_PRINTED_has_refused_nothing(monkeypatch):
    """⛔⛔ MEASURED 2026-09-06 on the cp1252 dev box: EVERY preflight
    refusal in this trainer died inside its own `print` --

        UnicodeEncodeError: 'charmap' codec can't encode character '\\u26d4'

    -- so the process exited **1 with a traceback** instead of **2 with the
    reason**. The guard was correct, reached, and MUTE. ⚠ Confirmed
    PRE-EXISTING (the same argv fails identically on the pre-change trainer),
    but it made the new effective-weight refusal invisible exactly where an
    operator meets it, which is the whole deliverable.

    ⚠ And fixing the leading glyph alone was NOT enough: the next run died
    on `'\\u21d2'` at position 322, inside another guard's message body. The
    fix has to be at the WRITE, not in the content.
    """
    import io

    class _Cp1252Out(io.TextIOWrapper):
        pass

    buf = io.BytesIO()
    stream = _Cp1252Out(buf, encoding="cp1252", errors="strict",
                        write_through=True)
    monkeypatch.setattr(sys, "stdout", stream)

    # a message carrying BOTH offenders: the marker and a body arrow.
    V6._print_refusal("--w-o5 1 in --stage S-T \u21d2 the term is removed")
    stream.flush()
    out = buf.getvalue().decode("cp1252")

    assert "--w-o5 1 in --stage S-T" in out, out      # the reason survived
    assert "the term is removed" in out, out          # ...including past the arrow
    assert "REFUSED:" in out                          # ASCII marker on a legacy codepage


def test_the_refusal_marker_keeps_the_house_glyph_on_utf8(monkeypatch):
    """The degrade is conditional, not a blanket downgrade: a UTF-8 console
    still gets the marker every other refusal in this repo uses."""
    import io
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="utf-8", write_through=True)
    monkeypatch.setattr(sys, "stdout", stream)
    V6._print_refusal("plain ascii problem")
    stream.flush()
    assert "\u26d4" in buf.getvalue().decode("utf-8")


# ---------------------------------------------------------------------------
# refc_v3_train
# ---------------------------------------------------------------------------

#: MODEL_REGISTRY.md -> refcv5-ddim-b1-v72-40k, `config.json['argv']`, verbatim.
#: ⛔ The A40 runs this to ~2026-09-08 07:33 UTC and its resume must not break.
LIVE_REFCV5 = shlex.split(
    "--arm hier --size base --v2-cache /root/data/train "
    "--v7-labels /root/s2_train.jsonl.gz --eval-cache /root/data/eval "
    "--eval-labels /root/s2_eval.jsonl.gz --eval-every 500 --eval-batches 8 "
    "--image-hw 256 640 --steps 40284 --batch 20 --workers 6 "
    "--prefetch-factor 1 --v2-lru 24 --lr 1e-4 --warmup 2000 --seed 0 "
    "--log-every 50 --save-every 500 --nav-from-v7 --u8-batches "
    "--anchors /workspace/experiments/refcv5/anchors.pt --n-anchors 117 "
    "--anchor-v0-conditioned --anchor-control-units alat --sel-accel-max 2.0 "
    "--goal-str --ego-state-inject --ego-dropout 0.5 --sampler ddim "
    "--w-u0 0.5 --agents off --out /workspace/experiments/refcv5")


def test_GREEN_the_live_refcv5_command_still_passes():
    """⭐ The flag-design rule: make your default one that survives contact with
    real launch lines. `--w-u0 0.5` is explicit, its gate (`--sampler ddim`
    builds `control_head`) is open, so it TRAINS."""
    assert _v3_refusal(LIVE_REFCV5) is None
    rows, src = V3.effective_weight_rows_v3(_v3(LIVE_REFCV5))
    assert src == ew.SRC_ARGV
    u0 = [r for r in rows if r.flag == "--w-u0"][0]
    assert u0.explicit is True and u0.status == ew.TRAINS and u0.builds_graph


def test_REFC_WEIGHT_GATES_covers_every_weight_flag_the_parser_accepts():
    """⛔ THE EXHAUSTIVENESS CONTRACT -- the durable half of the refc change.
    A new `--w-*` cannot ship without a gate, in the shape of
    `test_preflight_paths.py`'s PATH_ARGS union NOT_A_PATH."""
    ap = V3.build_parser()
    found = set()
    for ac in ap._actions:
        if not ac.option_strings or ac.type is not float:
            continue
        o = ac.option_strings[0]
        if "-w-" in o or o.endswith("-w"):
            found.add(ac.dest)
    assert found == set(V3.REFC_WEIGHT_GATES), (
        f"ungated weight flags: {sorted(found - set(V3.REFC_WEIGHT_GATES))}; "
        f"gates for non-flags: {sorted(set(V3.REFC_WEIGHT_GATES) - found)}")


@pytest.mark.parametrize("extra,needle", [
    (["--w-u0", "0.5"], "control_head"),
    (["--w-agent", "1.0"], "agent_slots"),
    (["--agent-w-project", "1.0"], "AgentSeamConfig"),
    (["--goal-point-w", "1.0"], "goal-point-inject"),
])
def test_RED_refc_refuses_a_weight_whose_gate_is_shut(extra, needle):
    got = _v3_refusal(["--arm", "hier", "--out", "/tmp/o", *extra])
    assert got is not None and needle in got, got


@pytest.mark.parametrize("extra", [
    ["--w-u0", "0.5", "--sampler", "ddim"],
    ["--w-agent", "1.0", "--agents", "head", "--agent-join", "/tmp/j.jsonl"],
    ["--agent-w-project", "1.0", "--agents", "oracle"],
    ["--goal-point-w", "1.0", "--goal-point-inject"],
])
def test_GREEN_refc_passes_once_the_gate_is_open(extra):
    """⭐ Four controls, one per refusal: a guard that refuses everything gets
    deleted."""
    assert _v3_refusal(["--arm", "hier", "--out", "/tmp/o", *extra]) is None


def test_refc_guard_is_on_the_real_path(monkeypatch):
    """⛔⛔ THE WIRING CLAIM, AND IT IS NOT THEORETICAL HERE.
    `refc_v3_train.main` calls `preflight` ONLY under `--preflight`, and
    otherwise goes straight to `train`. A guard wired into `preflight` alone
    would cover ONE launch path of two -- which is why every existing guard in
    this file (`_check_nav_from_v7_args`, `_check_goal_point_args`,
    `_read_anchor_artifact`) is double-called, and why this one is too."""
    seen: list[str] = []
    monkeypatch.setattr(V3, "check_effective_weights",
                        lambda a: seen.append("called"))

    # (a) the REAL path: main -> train, no --preflight anywhere.
    args = _v3(["--arm", "hier", "--out", "/tmp/o"])
    try:
        V3.train(args)            # dies later on the absent corpus; we only
    except BaseException:         # care that the guard ran BEFORE that.
        pass
    assert seen == ["called"], "train() never called the effective-weight guard"

    # (b) the --preflight path.
    seen.clear()
    try:
        V3.preflight(args)
    except BaseException:
        pass
    assert seen == ["called"], "preflight() never called the guard"


def test_refc_stamp_reaches_config_json():
    src = Path(V3.__file__).read_text(encoding="utf-8")
    assert src.count('"effective_weights": effective_weights_stamp_v3(') == 1


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

def test_explicit_dests_restores_the_parser_defaults():
    """It mutates the live parser's defaults; leaving them swapped would
    silently change every later `parse_args`."""
    ap = V6.build_parser()
    before = {ac.dest: ac.default for ac in ap._actions}
    ew.explicit_dests(ap, _st("--w-o5", "1.0"))
    after = {ac.dest: ac.default for ac in ap._actions}
    assert before == after


def test_explicit_dests_reports_unknown_rather_than_lying():
    """An ambiguous abbreviation makes argparse exit; UNKNOWN is the honest
    answer, and it must not be confused with the empty set."""
    ap = V6.build_parser()
    assert ew.explicit_dests(ap, ["--stage", "S-T", "--out", "/x",
                                  "--w-o", "1.0"]) is None
    assert ew.explicit_dests(None, []) is None
    assert ew.explicit_dests(ap, None) is None
    got = ew.explicit_dests(ap, _st("--w-o5=1.0"))
    assert got is not None and "w_o5" in got      # the `=` form counts
