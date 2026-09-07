"""⛔ EVERY BUILT MODULE MUST RECEIVE A GRADIENT — or be on a NAMED list of
known-unwired seams that MUST SHRINK.

⭐⭐ WHY THIS FILE EXISTS. The v7 vocabulary reach census (2026-09-07) measured
``tac_goal_tok_head`` at ``p.grad is None`` on **both** its tensors under the
LIVE refcv5-v2 argv — 11,286 parameters that are built, stamped, forward-run and
cannot learn. **Both existing guards were green on it, and neither is wrong:**

* ``assert_seams_are_built`` asks *"is the head BUILT?"* — it is;
* ``effective_weights_stamp_v3`` enumerates **declared loss weights** and asks
  which build a graph. Its own preamble cites the right measurement (*42/138
  optimizer tensors took no gradient — 52.2 % of a declared trainable budget*)
  and its ``_discriminator`` is exactly ``p.grad is None``. But
  ``tac_goal_tok_head`` **has no weight flag at all**, so it produces no row.
  An instrument that enumerates *weights* cannot see a head with *no weight*.

⇒ the missing question is neither *"is it built?"* nor *"is its weight zero?"*
but **"does a gradient actually land on it?"**, asked of EVERY built module.
That is what this file asks, once, structurally.

⭐ THE PRECEDENT IS ALREADY IN THIS TRAINER, ONE FLAG OVER.
``_check_goal_point_args`` refuses ``--goal-point-inject`` with
``--goal-point-w <= 0`` in exactly these words: *"a head that is built, stamped,
and **supervised by nothing** … a failure that is not about the lever is worse
than no run."* ``--tac-goal-tok-head`` is that same configuration and has no such
refusal, because it has no weight to key on.

⛔ THE DISCRIMINATOR IS ``p.grad is None``, NEVER THE GRADIENT'S VALUE. A module
whose gradient is present but ZERO is *wired and switched off* (a zero-init gate,
an all-ignored batch, a mask) — legitimate, and common at step 0. A module whose
gradient is ``None`` on **every** tensor was never in the graph. Only the second
is a defect, and conflating them is how the *"guarded term makes p.grad None"*
trap gets re-derived.

⛔ THE ALLOW-LIST IS A LITERAL AND IT MUST SHRINK. It is written as an explicit
``{module: reason}`` map, never as an expression over the model — the same shape
as ``vocab_v7.NOT_YET_EXTRACTABLE``, and for the same reason: an honest gap that
is NAMED cannot be silent, and a NEW unwired head fails immediately.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)
_SCRIPTS = os.path.join(_STACK, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_TRAINER_PY = os.path.join(_SCRIPTS, "refc_v3_train.py")

#: the census this guard came out of; every failure message names it.
CENSUS_DOC = ("TanitAD Research Lab/Architecture & Inference/Research/"
              "2026-09-07-v7-vocab-reach-census/RESULT.md")


def _trainer():
    """``refc_v3_train.py`` by PATH — the same convention as
    ``test_tac_goal_trainer_flag.py`` and ``refcv3_arm``."""
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_for_gradreach", _TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: ⛔ THE RECORDED argv OF THE LIVE refcv5-v2 ARM, copied from its own
#: ``config.json`` (banked at ``Research/2026-09-07-refcv5-v2-compose/raw/
#: refcv5_v2_launch_config_20260906.json``). Data paths are pod paths and are
#: never opened here — the smoke config supplies synthetic episodes.
LIVE_ARGV = [
    "--arm", "hier", "--size", "base",
    "--v2-cache", "/root/data/train",
    "--v7-labels", "/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz",
    "--image-hw", "256", "640", "--steps", "40284", "--batch", "20",
    "--lr", "1e-4", "--warmup", "2000", "--seed", "0", "--u8-batches",
    "--out", "/workspace/experiments/refcv5-v2-noagents-b1-v72-40k",
    "--nav-from-v7", "--ego-state-inject", "--ego-dropout", "0.5",
    "--n-anchors", "117", "--anchor-v0-conditioned",
    "--anchor-control-units", "alat", "--sel-accel-max", "2.0",
    "--sampler", "ddim", "--w-u0", "0.5", "--sel-refined",
    "--sel-score-emitted", "--goal-str", "--tac-goal-tok-head",
    "--agents", "off",
]

#: ⛔⛔ KNOWN-UNWIRED SEAMS. Every entry is a FINDING, not an exemption, and this
#: map MUST SHRINK. ⛔ Literals only — never derived from the model.
KNOWN_UNWIRED: dict[str, str] = {
    "tac_goal_tok_head":
        "D-TACGOAL-TRAINER-SEAM-OPEN (measured 2026-09-07): the head is built, "
        "stamped and forward-run (refc_v3.py:1320 -> cache['tac_goal_logits']) "
        "and NO trainer calls `tac_goal_loss` or `TacGoalEmitter`, so its "
        "11,286 parameters take no gradient. Closing it needs the two additive "
        "edits inside refc_v3_train.py AND the MANEUVER_WEIGHT budget decision "
        "(an owner/PI call). When it lands, DELETE this entry -- the test will "
        "tell you by failing the `must shrink` assertion below.",
}


def _module_grad_census(model) -> dict[str, dict]:
    """Per named child: how many parameter tensors took ``None`` gradient.

    ⛔ MODULE level, not tensor level. A module with SOME ``None`` gradients is
    partially masked, which is ordinary; a module where EVERY tensor is ``None``
    was never in the graph.
    """
    out: dict[str, dict] = {}
    for name, mod in model.named_children():
        ps = [p for p in mod.parameters() if p.requires_grad]
        if not ps:
            continue
        n_none = sum(1 for p in ps if p.grad is None)
        gsum = sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
        out[name] = {
            "n_tensors": len(ps),
            "n_params": sum(p.numel() for p in ps),
            "n_grad_none": n_none,
            "grad_abs_sum": gsum,
            "verdict": ("NOT_WIRED" if n_none == len(ps)
                        else "ZERO_GRAD" if gsum == 0.0
                        else "GRADIENT_REACHES"),
        }
    return out


def _build_and_backward(extra_argv: list[str] | None = None):
    """Build the arm the LIVE argv describes (smoke width), run the trainer's
    OWN loss, one backward. Returns ``(model, census, losses)``.

    ⛔ Backwards ``losses["loss"]`` — the trainer's own total. Summing every
    tensor that happens to carry grad would ADD gradient paths the run does not
    have and turn an unwired head into a reached one.
    """
    T = _trainer()
    args = T.build_parser().parse_args(LIVE_ARGV + list(extra_argv or []))
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model.train()

    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    n_lon = len(v7l.HEADS["tac_lon"])
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID], dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])

    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    total = losses["loss"]
    assert total.requires_grad, (
        "the trainer's own total loss does not require grad -- the probe is "
        "UNPOWERED, not clean")
    total.backward()
    return model, _module_grad_census(model), losses


# --------------------------------------------------------------------------- #
# THE GUARD                                                                     #
# --------------------------------------------------------------------------- #
def test_no_built_module_is_unwired_except_the_named_list() -> None:
    """⛔ THE CHECK THAT WAS MISSING. Any built module whose EVERY parameter
    takes a ``None`` gradient is unreachable from the loss."""
    _model, census, _losses = _build_and_backward()
    unwired = {k: v for k, v in census.items() if v["verdict"] == "NOT_WIRED"}
    unexpected = {k: v for k, v in unwired.items() if k not in KNOWN_UNWIRED}
    assert not unexpected, (
        f"\n  ⛔ BUILT BUT UNWIRED: {sorted(unexpected)}\n"
        f"     {unexpected}\n"
        f"  Every parameter of these modules took `p.grad is None` after the\n"
        f"  trainer's own loss and one backward -- they are built, stamped and\n"
        f"  cannot learn. This is the D-TACGOAL-1 shape.\n"
        f"  ⇒ EITHER wire the loss, OR add the module to KNOWN_UNWIRED with a\n"
        f"     reason and register the gap. Do NOT widen the check.\n"
        f"  Census: {CENSUS_DOC}\n")


def test_the_known_unwired_list_must_shrink_and_never_go_stale() -> None:
    """⛔ Two-sided. An entry that is no longer unwired is GOOD NEWS and still
    fails, because the register and the docs describe the gap."""
    _model, census, _losses = _build_and_backward()
    for name, reason in KNOWN_UNWIRED.items():
        assert name in census, (
            f"\n  KNOWN_UNWIRED names {name!r} but the model has no such built\n"
            f"  module. If it was removed, delete the entry. Reason on file:\n"
            f"  {reason}\n")
        assert census[name]["verdict"] == "NOT_WIRED", (
            f"\n  ⭐ GOOD NEWS: {name!r} now reads "
            f"{census[name]['verdict']} -- it is WIRED.\n"
            f"  ⇒ DELETE it from KNOWN_UNWIRED, and update in the SAME commit:\n"
            f"      - Project Steering/GOALS_AND_CLAIMS.md "
            f"(D-TACGOAL-TRAINER-SEAM-OPEN)\n"
            f"      - {CENSUS_DOC}\n"
            f"      - stack/tests/test_tactical_label_reach.py\n"
            f"      - stack/tests/test_v7_vocab_reach_census.py "
            f"(EXPECTED_CONSUMER_STATUS)\n")


def test_the_probe_is_powered_not_merely_quiet() -> None:
    """⛔ C109: a probe that cannot fire proves nothing. Modules that SHOULD be
    reached must actually read GRADIENT_REACHES in the same run."""
    _model, census, _losses = _build_and_backward()
    for name in ("core", "lat_head_tac", "lon_head_tac", "phi_tac"):
        assert name in census, f"{name!r} is not a built child of the model"
        assert census[name]["verdict"] == "GRADIENT_REACHES", (
            f"{name!r} reads {census[name]['verdict']} -- the probe is "
            f"UNPOWERED, so its NOT_WIRED verdicts elsewhere are "
            f"uninformative. Census row: {census[name]}")


def test_the_detector_actually_fires_on_a_deliberately_unwired_head() -> None:
    """⭐ MUTATION. Attach a brand-new head that no loss touches and prove the
    detector names it."""
    import torch.nn as nn
    model, _census, _losses = _build_and_backward()
    model.add_module("MUTANT_unwired_head", nn.Linear(4, 3))
    census = _module_grad_census(model)
    assert census["MUTANT_unwired_head"]["verdict"] == "NOT_WIRED", (
        "the detector did NOT fire on a head that plainly received no "
        "gradient -- it is dead, and every clean verdict from it is "
        "uninformative")
    unexpected = {k for k, v in census.items()
                  if v["verdict"] == "NOT_WIRED" and k not in KNOWN_UNWIRED}
    assert unexpected == {"MUTANT_unwired_head"}, (
        f"the mutation changed the verdict of unrelated modules "
        f"({unexpected}) -- the detector is not specific")


def test_a_zero_gradient_is_NOT_read_as_unwired() -> None:
    """⛔ THE DISCRIMINATOR. ``p.grad`` present-but-zero means wired and
    switched off (a zero-init gate, a mask, an all-ignored batch). Reading it
    as 'unwired' would re-derive the guarded-term trap in the opposite
    direction and make this guard useless at step 0."""
    import torch.nn as nn
    model, _census, _losses = _build_and_backward()
    lin = nn.Linear(4, 3)
    (lin(torch.zeros(1, 4)).sum() * 0.0).backward()      # real grad, all zeros
    assert lin.weight.grad is not None
    assert float(lin.weight.grad.abs().sum()) == 0.0
    model.add_module("MUTANT_zero_grad_head", lin)
    census = _module_grad_census(model)
    assert census["MUTANT_zero_grad_head"]["verdict"] == "ZERO_GRAD", (
        "a module with a real, all-zero gradient was classified as NOT_WIRED "
        "-- the guard would fire on every zero-init gate at step 0 and would "
        "be turned off within a day")

