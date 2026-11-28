"""Pins the DECLARED ALLOWANCE that lets the RL pilot load its July cold start.

PI DECISION QUEUE item 8, option (c). ``tanitad.refs.cold_start`` permits ONE
state-dict key — ``decoder.anchor_controls`` — to be absent from a checkpoint,
and only when the decoder about to receive the weights is not v0-conditioned.

⛔ WHY THIS FILE IS WRITTEN THE WAY IT IS. ``CLAUDE.md``: *"a check that shares
the defect it checks for is green forever"*, MEASURED four times in one night —
and one of the four was ``test_rl_channel_guard.py``, which asserted *"if
missing: expect refusal; else: expect pass"*. That is an expected value written
as **whatever the code does**, and it was green while the preflight it guarded
was actively wrong. So, here:

* **every expectation is a LITERAL.** ``488``, ``487``, ``136``, ``135``,
  ``"decoder.anchor_controls"``, ``"defaulted-zeros-v0-unconditioned"``. Not one
  of them is an expression over the module under test. If ``refc.py`` changes
  the built key count, these tests go RED and a human decides — which is the
  point, because the whole defect was a key count changing under a loader.
* **the assertions are on the ARTIFACT.** ``load_cold_start`` returning without
  raising is a claim the call makes about itself. What is asserted is the state
  dict that exists on the model afterwards, and the bytes of ``config.json``
  read back off disk.
* **there is a deliberate-regression arm.**
  :func:`test_REGRESSION_missing_controls_with_v0_conditioned_is_refused` builds
  the exact configuration the guard exists to stop and requires a refusal.
  MEASURED 2026-09-10 by ``mutate_cold_start_guard.py``: with the run-condition
  gate DELETED from ``cold_start.py`` (mutant ``M1_remove_run_gate``) that arm
  goes RED — 4 arms in total, exactly the 4 the harness names as literals —
  and with the whole allowance replaced by ``strict=False`` (mutant
  ``M2_strict_false``, i.e. the forbidden option (a)) 15 arms go RED. A green
  suite is not evidence about a guard until it has been watched failing.
* **the ``strict=False`` prohibition is asserted on the SYNTAX TREE**, because a
  behavioural test cannot distinguish "loaded strictly with one declared key"
  from "loaded with ``strict=False``" on a checkpoint where they happen to
  agree. ⚠️ Its own control
  (:func:`test_the_strict_kwarg_checker_can_actually_see_a_violation`) plants
  the real defect and requires the checker to find it — the first draft of that
  check was a substring search and went RED on its own docstring, which is the
  harmless half of a search that cannot tell code from prose.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import os
import sys

import pytest
import torch

from tanitad.refs import cold_start as CS
from tanitad.refs import refc

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_STACK, "scripts")

# --------------------------------------------------------------------------- #
# LITERALS. Measured 2026-09-10 on this box against `refc.py`; every one of them
# is written out rather than computed, so a drift is a RED test and not a
# silently-updated expectation.
# --------------------------------------------------------------------------- #

#: `refc.RefCModel(refc.refc_config())` — the pilot's production build.
PROD_KEYS = 488
PROD_PARAMS = 104_191_577

#: `refc.RefCModel(refc.refc_smoke_config())` — the same class, test-sized.
SMOKE_KEYS = 136
SMOKE_KEYS_WITHOUT_CONTROLS = 135

#: The one key the allowance covers, and the stamp values.
KEY = "decoder.anchor_controls"
SOURCE_DEFAULTED = "defaulted-zeros-v0-unconditioned"
SOURCE_CHECKPOINT = "checkpoint"
STAMP_FIELD = "anchor_controls_source"


def _smoke_model(v0_conditioned: bool = False):
    cfg = refc.refc_smoke_config()
    cfg.anchors.v0_conditioned = bool(v0_conditioned)
    return refc.RefCModel(cfg)


def _july_state_dict(model) -> dict:
    """A faithful stand-in for `refc-diffusion-base-v21-30k`: the model's own
    weights with `decoder.anchor_controls` REMOVED.

    ⭐ This is the exact shape of the real defect — 487 keys against 488 built —
    without needing the 2026-07-20 artifact, which lives on no box this test can
    reach. The key removed is the key the July checkpoint genuinely lacks,
    because the buffer was added on 2026-09-04 (refc.py:1496).
    """
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del sd[KEY]
    return sd


# --------------------------------------------------------------------------- #
# 1. The numbers the whole item rests on.                                       #
# --------------------------------------------------------------------------- #

def test_production_model_builds_the_literal_488_keys():
    """⭐ The claim in PI queue item 8: 488 built, one of them anchor_controls.

    Written as literals. If this goes RED, the allowance's premise moved and
    `cold_start.py`'s docstring is stale — which is exactly when a human should
    be looking.
    """
    m = refc.RefCModel(refc.refc_config())
    sd = m.state_dict()
    assert len(sd) == 488
    assert sum(p.numel() for p in m.parameters()) == 104_191_577
    assert KEY in sd
    # the buffer is ZERO-initialised, not random — this is why a silent
    # `strict=False` produces a "do nothing" vocabulary rather than noise.
    assert int(torch.count_nonzero(sd[KEY])) == 0


def test_the_buffer_is_registered_unconditionally():
    """⛔ The premise of the whole allowance: the key set does NOT depend on
    `v0_conditioned`. If registration ever became conditional (option (b)), the
    allowance would be reasoning about the wrong thing.
    """
    off = _smoke_model(v0_conditioned=False).state_dict()
    on = _smoke_model(v0_conditioned=True).state_dict()
    assert len(off) == SMOKE_KEYS
    assert len(on) == SMOKE_KEYS
    assert KEY in off and KEY in on
    assert sorted(off) == sorted(on)


# --------------------------------------------------------------------------- #
# 2. THE PERMITTED CASE — and it is asserted on the state dict, not a return.   #
# --------------------------------------------------------------------------- #

def test_july_checkpoint_loads_and_the_model_holds_all_136_keys():
    m = _smoke_model(v0_conditioned=False)
    sd = _july_state_dict(m)
    assert len(sd) == SMOKE_KEYS_WITHOUT_CONTROLS      # 135, the defect's shape

    # perturb a real weight so a no-op load could not pass by accident
    victim = "decoder.feat_proj.weight"
    sd[victim] = torch.full_like(sd[victim], 0.125)

    fresh = _smoke_model(v0_conditioned=False)
    stamp = CS.load_cold_start(fresh, sd)

    after = fresh.state_dict()
    # ⭐ THE ARTIFACT: 136 keys on the model, the defaulted one present.
    assert len(after) == SMOKE_KEYS
    assert KEY in after
    assert int(torch.count_nonzero(after[KEY])) == 0
    # ⭐ and the checkpoint's real weights actually landed
    assert torch.allclose(after[victim], torch.full_like(after[victim], 0.125))

    assert stamp[STAMP_FIELD] == SOURCE_DEFAULTED
    assert stamp["defaulted_keys"] == [KEY]
    assert stamp["strict"] is True
    assert stamp["keys_after_load"] == SMOKE_KEYS


def test_a_complete_checkpoint_defaults_nothing_and_says_so():
    """The control that must read the NO-DEFAULTING value exactly.

    Without it, "the stamp says defaulted" would be unfalsifiable — a stamp that
    said `defaulted` for every load would pass the test above.
    """
    m = _smoke_model(v0_conditioned=False)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    assert len(sd) == SMOKE_KEYS

    fresh = _smoke_model(v0_conditioned=False)
    stamp = CS.load_cold_start(fresh, sd)
    assert stamp[STAMP_FIELD] == SOURCE_CHECKPOINT
    assert stamp["defaulted_keys"] == []
    assert len(fresh.state_dict()) == SMOKE_KEYS


def test_a_v0_conditioned_model_with_a_COMPLETE_checkpoint_still_loads():
    """⛔ The allowance must not have become a blanket refusal of v0-conditioned
    builds. A conditioned model whose checkpoint CARRIES the controls is
    entirely legal and must load untouched.
    """
    m = _smoke_model(v0_conditioned=True)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    sd[KEY] = torch.full_like(sd[KEY], 0.5)          # a real vocabulary

    fresh = _smoke_model(v0_conditioned=True)
    stamp = CS.load_cold_start(fresh, sd)
    assert stamp[STAMP_FIELD] == SOURCE_CHECKPOINT
    assert stamp["defaulted_keys"] == []
    assert torch.allclose(fresh.state_dict()[KEY],
                          torch.full_like(sd[KEY], 0.5))


# --------------------------------------------------------------------------- #
# 3. ⛔ THE DELIBERATE-REGRESSION ARM.                                          #
# --------------------------------------------------------------------------- #

def test_REGRESSION_missing_controls_with_v0_conditioned_is_refused():
    """⛔⛔ THE ARM THIS GUARD EXISTS FOR. A checkpoint missing
    `decoder.anchor_controls` while the decoder IS v0-conditioned must be
    REFUSED, and the test FAILS if it loads.

    ⭐ It also asserts the model was left UNTOUCHED, because "refused" that
    still mutated the model is the failure mode `assert_config_contract`'s
    comment warns about: a refusal after the weights land reads to the operator
    as a loading failure and leaves a mismatched model in memory.
    """
    m = _smoke_model(v0_conditioned=True)
    sd = _july_state_dict(m)
    assert len(sd) == SMOKE_KEYS_WITHOUT_CONTROLS

    fresh = _smoke_model(v0_conditioned=True)
    victim = "decoder.feat_proj.weight"
    sd[victim] = torch.full_like(sd[victim], 7.0)
    before = fresh.state_dict()[victim].detach().clone()

    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(fresh, sd)

    msg = str(exc.value)
    assert "v0-CONDITIONED" in msg
    assert "plausible-looking" in msg
    # ⛔ nothing landed
    assert torch.equal(fresh.state_dict()[victim], before)


def test_REGRESSION_checkpoint_config_claiming_v0_true_is_refused():
    """A checkpoint whose OWN config says `v0_conditioned = True` while lacking
    the buffer is an incoherent artifact. Even though the RUN is unconditioned
    (so the zeros would be unread), it is refused: the two facts cannot both be
    true, and defaulting the tensor would bury the question under a run that
    looks fine.
    """
    m = _smoke_model(v0_conditioned=False)
    sd = _july_state_dict(m)
    fresh = _smoke_model(v0_conditioned=False)
    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(fresh, sd,
                           ckpt_cfg_leaves={"anchors.v0_conditioned": True})
    assert "CHECKPOINT'S OWN CONFIG" in str(exc.value)


def test_checkpoint_config_saying_false_is_recorded_as_CONFIRMED():
    m = _smoke_model(v0_conditioned=False)
    sd = _july_state_dict(m)
    fresh = _smoke_model(v0_conditioned=False)
    stamp = CS.load_cold_start(
        fresh, sd, ckpt_cfg_leaves={"anchors.v0_conditioned": False})
    assert stamp["ckpt_confirmation"] == "CONFIRMED"
    assert stamp["ckpt_v0_conditioned"] is False


def test_absent_ckpt_config_is_reported_UNVERIFIED_never_as_agreement():
    """⚠️ The honesty requirement. A 2026-07-20 checkpoint CANNOT carry
    `anchors.v0_conditioned` — the field did not exist — so the stamp must say
    UNVERIFIED rather than quietly reading as confirmation.
    """
    m = _smoke_model(v0_conditioned=False)
    fresh = _smoke_model(v0_conditioned=False)
    stamp = CS.load_cold_start(fresh, _july_state_dict(m))
    assert stamp["ckpt_v0_conditioned"] is None
    assert stamp["ckpt_v0_source"] == "ABSENT"
    assert stamp["ckpt_confirmation"] == "UNVERIFIED_BY_CKPT_CONFIG"


def test_require_ckpt_confirmation_refuses_when_the_leaf_is_absent():
    m = _smoke_model(v0_conditioned=False)
    fresh = _smoke_model(v0_conditioned=False)
    with pytest.raises(CS.ColdStartRefused):
        CS.load_cold_start(fresh, _july_state_dict(m),
                           require_ckpt_confirmation=True)


def test_two_conflicting_config_copies_are_refused_not_tie_broken():
    m = _smoke_model(v0_conditioned=False)
    fresh = _smoke_model(v0_conditioned=False)
    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(fresh, _july_state_dict(m),
                           ckpt_cfg_leaves={"anchors.v0_conditioned": True,
                                            "core.anchors.v0_conditioned": False})
    assert "DISAGREE" in str(exc.value)


# --------------------------------------------------------------------------- #
# 4. ⛔ THE ALLOWANCE IS ONE LITERAL KEY, NOT "WHATEVER IS MISSING".             #
# --------------------------------------------------------------------------- #

def test_the_allowed_set_is_exactly_one_literal_key():
    assert CS.ALLOWED_ABSENT_KEYS == ("decoder.anchor_controls",)


def test_a_SECOND_missing_key_is_refused_even_though_the_first_is_allowed():
    """⛔ THE DISCRIMINATOR BETWEEN THIS AND `strict=False`. Under
    `strict=False` this load succeeds silently; here it must refuse.
    """
    m = _smoke_model(v0_conditioned=False)
    sd = _july_state_dict(m)
    victim = "decoder.feat_proj.weight"
    del sd[victim]
    assert len(sd) == SMOKE_KEYS_WITHOUT_CONTROLS - 1

    fresh = _smoke_model(v0_conditioned=False)
    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(fresh, sd)
    assert victim in str(exc.value)
    assert "OUTSIDE the" in str(exc.value)


def test_an_unexpected_key_is_refused():
    m = _smoke_model(v0_conditioned=False)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    sd["decoder.a_key_this_model_never_had"] = torch.zeros(3)
    fresh = _smoke_model(v0_conditioned=False)
    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(fresh, sd)
    assert "ARCHITECTURE mismatch" in str(exc.value)


def _strict_kwargs(path) -> list[object]:
    """Every literal value passed as a `strict=` keyword in EXECUTABLE code.

    ⛔ AST, not a text grep. The first version of this check was a substring
    search and it went RED on its own docstring — which is the harmless half of
    the same defect: a text search cannot tell code from prose, so it would
    equally have gone GREEN on a `strict=False` sitting inside a string. This
    walks the syntax tree, so comments and docstrings are structurally invisible
    to it.
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "strict" and isinstance(kw.value, ast.Constant):
                    out.append(kw.value.value)
    return out


def test_the_strict_kwarg_checker_can_actually_see_a_violation(tmp_path):
    """⭐ THE SAME-BREATH CONTROL. A checker that reads "no violations" from a
    file it could not parse is indistinguishable from one that read a clean
    file. This plants the real defect and requires the checker to find it —
    without this, the test below is unfalsifiable.
    """
    good = tmp_path / "good.py"
    good.write_text("m.load_state_dict(sd, strict=True)\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text('"""strict=False in a docstring is not code."""\n'
                   "# strict=False in a comment is not code either\n"
                   "m.load_state_dict(sd, strict=False)\n", encoding="utf-8")
    assert _strict_kwargs(good) == [True]
    assert _strict_kwargs(bad) == [False]


def test_strict_false_appears_nowhere_in_the_module():
    """⛔ Every `strict=` in `cold_start.py`'s executable code is `strict=True`.

    A behavioural test cannot tell a strict load with one declared key from a
    `strict=False` load on a checkpoint where the two agree — so the
    prohibition is pinned structurally, alongside (never instead of) the
    behavioural arms above.
    """
    got = _strict_kwargs(CS.__file__)
    assert got == [True], f"expected exactly one strict=True, got {got}"


# --------------------------------------------------------------------------- #
# 5. ⛔ CAN AN OPERATOR FAKE THE PRECONDITION? — asserted, not claimed.          #
# --------------------------------------------------------------------------- #

def _pilot():
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    spec = importlib.util.spec_from_file_location(
        "rl_pilot_refc21_under_test",
        os.path.join(_SCRIPTS, "rl_pilot_refc21.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_pilot_exposes_no_flag_that_can_set_v0_conditioned(monkeypatch):
    """⛔ THE ANTI-FAKING ASSERTION. The allowance keys on a flag read off the
    CONSTRUCTED decoder. That is only meaningful if an operator cannot set it —
    otherwise the guard lets them assert the precondition they are supposed to
    be proving.

    This captures the pilot's REAL parser (by intercepting `parse_args`) and
    asserts the option strings literally.
    """
    mod = _pilot()
    captured = {}

    def _spy(self, *a, **kw):
        captured["opts"] = [o for act in self._actions for o in act.option_strings]
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _spy)
    monkeypatch.setattr(sys, "argv", ["rl_pilot_refc21.py"])
    with pytest.raises(SystemExit):
        mod.main()

    opts = captured["opts"]
    assert opts, "the parser exposed no options at all — the spy did not fire"
    # LITERAL: the complete option surface of the pilot.
    #
    # ⭐ UPDATED 2026-09-11, DELIBERATELY, AND THE TRIPWIRE DID ITS JOB. The
    # 2026-09-10 list ended at `--lru`; adding V2's two on-switches made this
    # assertion FAIL, which is exactly what a literal surface pin is for — a
    # human has to look at the new flags and decide. Both were looked at:
    #   `--gt-bar`      turns on DDv2's >=GT positive mask. Touches the ADVANTAGE
    #                   (`posttrain.rl_objective` -> `composite_advantage`), never
    #                   the anchor vocabulary or the cold-start allowance.
    #   `--noise-mode`  selects the exploration policy over the EMITTED offset
    #                   (`refcv3_adapter.sample_offsets`). Also downstream of the
    #                   bank; `anchors.v0_conditioned` is read at `refc.py:1715`
    #                   while building it, which neither flag can reach.
    # ⛔ Neither can set `anchors.v0_conditioned` or waive the allowance, and the
    # substring guard below is what enforces that rather than this comment.
    assert sorted(opts) == sorted([
        "-h", "--help", "--ckpt", "--train-epdir", "--train-agents",
        "--val-epdir", "--val-agents", "--out", "--steps", "--batch",
        "--reward", "--proximity-safe-m", "--seed", "--w-anchor", "--lru",
        "--gt-bar", "--noise-mode",
    ])
    for bad in ("v0", "anchor-control", "anchor_control", "strict", "allow",
                "conditioned", "force"):
        assert not [o for o in opts if bad in o.lower()], \
            f"the pilot grew an option matching {bad!r} — if it can set " \
            f"anchors.v0_conditioned or waive the allowance, the guard is " \
            f"an operator assertion and this test must not be relaxed"


def test_NOTHING_an_operator_writes_can_GRANT_the_allowance():
    """⛔⛔ THE ASYMMETRY THAT MAKES THIS A GUARD AND NOT A SWITCH.

    The checkpoint's config is the one input an operator can trivially forge —
    it is a JSON file they can drop next to the weights. So the property that
    matters is directional: **the config can only ever REFUSE.** On a
    v0-conditioned build every one of these forged inputs must still be
    refused, because the safety condition is the CONSTRUCTED decoder's flag,
    which no file can reach.

    (`test_REGRESSION_checkpoint_config_claiming_v0_true_is_refused` covers the
    other direction: a forged `True` refuses an otherwise-legal load. Together
    they say the config moves the answer toward REFUSE and never toward ALLOW.)
    """
    forged = [
        None,
        {},
        {"anchors.v0_conditioned": False},
        {"core.anchors.v0_conditioned": False},
        {"anchors.v0_conditioned": False, "sel_reach_clamp": False},
        {"anchors.v0_conditioned": 0},          # falsy, not False
    ]
    for leaves in forged:
        model = _smoke_model(v0_conditioned=True)
        donor = _smoke_model(v0_conditioned=True)
        with pytest.raises(CS.ColdStartRefused):
            CS.load_cold_start(model, _july_state_dict(donor),
                               ckpt_cfg_leaves=leaves)


def test_refc_config_default_is_literally_false():
    """The value the allowance depends on, asserted as a LITERAL rather than
    read from the same expression the pilot uses."""
    assert refc.refc_config().anchors.v0_conditioned is False
    assert refc.RefCModel(refc.refc_smoke_config()).decoder.anchor_v0_cond is False


def test_the_flag_is_read_from_the_module_not_from_a_config_argument():
    """⭐ The owning module is resolved from the KEY'S OWN PATH. A model whose
    decoder is nested elsewhere must still have ITS flag read, not the root's —
    reading the wrong object's flag is precisely the failure this resolution
    exists to prevent.
    """
    class _Nested(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.core = inner

    inner = _smoke_model(v0_conditioned=True)
    nested = _Nested(inner)
    # the root has no `anchor_v0_cond`; the true owner (core.decoder) says True
    assert not hasattr(nested, "anchor_v0_cond")
    sd = {k: v.detach().clone() for k, v in nested.state_dict().items()}
    del sd["core.decoder.anchor_controls"]
    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(nested, sd)
    assert "v0-CONDITIONED" in str(exc.value)


def test_ONE_conditioned_owner_refuses_the_WHOLE_load():
    """⛔ Each missing key is judged against ITS OWN owner. With two anchor
    decoders — one conditioned, one not — the un-conditioned one must NOT be
    able to authorise a default for its conditioned sibling, which is the
    module that would actually read the zeros.
    """
    class _Two(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a, self.b = a, b

    m = _Two(_smoke_model(v0_conditioned=False),
             _smoke_model(v0_conditioned=True))
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del sd["a.decoder.anchor_controls"]
    del sd["b.decoder.anchor_controls"]
    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(m, sd)
    msg = str(exc.value)
    assert "v0-CONDITIONED" in msg
    # ⭐ and it must name the CONDITIONED one, not merely the first missing key
    assert "b.decoder.anchor_controls" in msg


def test_two_unconditioned_owners_both_default_cleanly():
    """The control for the test above: with NEITHER conditioned, both keys are
    defaulted and the load succeeds. Without this, the refusal above could be
    'any two missing keys refuse', which is a different rule.
    """
    class _Two(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a, self.b = a, b

    m = _Two(_smoke_model(v0_conditioned=False),
             _smoke_model(v0_conditioned=False))
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del sd["a.decoder.anchor_controls"]
    del sd["b.decoder.anchor_controls"]
    stamp = CS.load_cold_start(m, sd)
    assert stamp["defaulted_keys"] == ["a.decoder.anchor_controls",
                                       "b.decoder.anchor_controls"]
    after = m.state_dict()
    assert "a.decoder.anchor_controls" in after
    assert "b.decoder.anchor_controls" in after


def test_owner_without_the_flag_is_refused_not_assumed():
    class _Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.register_buffer("anchor_controls", torch.zeros(4, 2))

    m = _Bare()
    sd = {}
    with pytest.raises(CS.ColdStartRefused) as exc:
        CS.load_cold_start(m, sd)
    assert "no `anchor_v0_cond`" in str(exc.value)


# --------------------------------------------------------------------------- #
# 6. THE STAMP MUST LAND IN `config.json` — read back off disk.                  #
# --------------------------------------------------------------------------- #

def test_the_stamp_lands_in_the_runs_config_json(tmp_path):
    """⭐ Asserted by READING THE FILE BACK, not by checking that a dict was
    passed. `run_posttrain` writes `config.json`; the allowance must be in it.
    """
    from tanitad.rl.config import PostTrainConfig
    from tanitad.rl.posttrain import run_posttrain

    m = _smoke_model(v0_conditioned=False)
    fresh = _smoke_model(v0_conditioned=False)
    stamp = CS.load_cold_start(fresh, _july_state_dict(m))

    lin = torch.nn.Linear(2, 2)

    def sample_fn(batch, cfg):
        b, g, s = 2, cfg.group_size, 4
        traj = lin(torch.zeros(b, g, s, 2)).reshape(b, g, s, 2)
        logp = traj.sum(dim=(2, 3))
        return traj, logp, {"dt": cfg.dt}

    cfg = PostTrainConfig(method="grpo", group_size=2, steps=1, batch=2,
                          out_dir=str(tmp_path), trainable_prefixes=("",),
                          w_imitation=0.0, veto_enabled=False)
    run_posttrain(lin, sample_fn, cfg, batches=[0],
                  extra_record={"cold_start_load": stamp})

    path = tmp_path / "config.json"
    assert path.exists()
    rec = json.loads(io.open(path, encoding="utf-8").read())
    assert "cold_start_load" in rec
    got = rec["cold_start_load"]
    assert got[STAMP_FIELD] == SOURCE_DEFAULTED
    assert got["defaulted_keys"] == [KEY]
    assert got["ckpt_confirmation"] == "UNVERIFIED_BY_CKPT_CONFIG"
    assert got["strict"] is True
    # ⛔ and `extra_record` must not be able to rewrite what the run did
    assert rec["method"] == "grpo"


def test_extra_record_cannot_overwrite_the_runs_own_fields(tmp_path):
    from tanitad.rl.config import PostTrainConfig
    from tanitad.rl.posttrain import run_posttrain

    lin = torch.nn.Linear(2, 2)

    def sample_fn(batch, cfg):
        traj = lin(torch.zeros(2, cfg.group_size, 4, 2))
        return traj, traj.sum(dim=(2, 3)), {"dt": cfg.dt}

    cfg = PostTrainConfig(method="grpo", group_size=2, steps=1, batch=2,
                          out_dir=str(tmp_path), trainable_prefixes=("",),
                          w_imitation=0.0, veto_enabled=False)
    run_posttrain(lin, sample_fn, cfg, batches=[0],
                  extra_record={"method": "LIED", "seed": 999})
    rec = json.loads(io.open(tmp_path / "config.json", encoding="utf-8").read())
    assert rec["method"] == "grpo"
    assert rec["seed"] == 0


# --------------------------------------------------------------------------- #
# 7. The pilot actually uses it.                                                #
# --------------------------------------------------------------------------- #

def test_the_pilot_load_model_routes_through_the_allowance():
    """⛔ The `test_rl_channel_guard` lesson one level up: a guarded path that
    is not the executed path is not a guard. `load_model`'s source must call
    `load_cold_start` and must NOT call `load_state_dict` itself.
    """
    path = os.path.join(_SCRIPTS, "rl_pilot_refc21.py")
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "load_model")

    called = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            called.add(f.attr if isinstance(f, ast.Attribute) else
                       getattr(f, "id", ""))
    assert "load_cold_start" in called
    # ⛔ the pilot must not keep its own load beside the allowance — a guarded
    # path that is not the executed path is not a guard (`test_rl_channel_guard`).
    assert "load_state_dict" not in called
    # and nothing anywhere in the pilot may weaken a load
    assert False not in _strict_kwargs(path)
