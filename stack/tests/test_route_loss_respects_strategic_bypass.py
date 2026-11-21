"""⛔⛔ UNDER `--no-strategic` THE ROUTE HEAD MUST NOT SHAPE THE SHARED TRUNK.

⭐⭐ WHY THIS FILE EXISTS — AND THE TRAINER ALREADY SAID SO, ONE TERM OVER.
`refc_v3_train.py`'s own comment above the strategic-goal term states the rule:

    "with `--no-strategic` the E4 FiLM is skipped, so `g_str` has NO in-graph
     consumer at all. Supervising it anyway would push gradient through
     `str_goal_head` -> `StrategicCtx` -> the SHARED ENCODER, i.e. the strategic
     layer would still shape the trunk that produces the plan. That is a SECOND
     variable inside a one-variable arm."

That argument applies verbatim to the ROUTE readout, which the same flag bypasses —
and the guard had been applied to ONE term and not to its neighbour. `ROUTE_WEIGHT *
loss_route` sat in the loss sum unconditionally.

MEASURED 2026-09-22 with `no_strategic=True` (review package
`2026-09-22-refcv6-review/raw/q3b_route_grad.json`): the route cross-entropy put
non-zero gradient on **28 of 60 trunk tensors**, `sum|grad|` **91.66**, against a
no-backward control of **0/60** and a trajectory-loss reference that reaches the same
**28/60**. So refcv6's "strategic layer OFF" arm was training the shared trunk through
a strategic head — exactly the second variable the comment forbids.

⭐ THE CLASS: this is the ADVISORY's class G in a different file — *"the file's own
rule, six lines above the defect, already said ... the principle had been applied to one
term and not to its neighbour."* A rule stated in prose beside the code it does not
govern reads as if it governs it.

⛔ THE DISCRIMINATOR IS `p.grad is None`, NEVER THE GRADIENT'S VALUE — the convention
`test_built_heads_receive_gradient.py` establishes and for its reason: a module whose
gradient is PRESENT but zero is wired-and-switched-off (a zero-init gate, an all-ignored
batch), which is a different state from unreachable.

⛔ CPU only.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_PY = os.path.join(_HERE, "..", "scripts", "refc_v3_train.py")

# ⭐ REUSED, NOT RE-IMPLEMENTED. `test_built_heads_receive_gradient` already builds the
# live arm at smoke width, runs the TRAINER'S OWN total loss and does one backward. A
# second spelling of that harness is a second thing to drift.
_GRADREACH = os.path.join(_HERE, "test_built_heads_receive_gradient.py")


def _harness():
    if not os.path.exists(_GRADREACH) or not os.path.exists(_TRAINER_PY):
        pytest.skip("the gradient-reach harness or the trainer is not present")
    spec = importlib.util.spec_from_file_location("gradreach_for_route", _GRADREACH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_and_backward_with_a_JUDGEABLE_route(extra_argv=None):
    """The sibling harness, plus a route target the loss can actually score.

    ⛔⛔ THIS FUNCTION EXISTS BECAUSE THE CONTROL BELOW CAUGHT ITS ABSENCE.
    My first version called `_build_and_backward` directly and the POWERED control
    FAILED: the route head took **0 of 2** gradients with the strategic layer ACTIVE.
    MEASURED cause — the synthetic smoke batch carries `route_valid [False, False]`
    and `route_target [3, 3]`, and `3` is ROUTE_UNKNOWN under the v2.1 contract
    (`route < 3 <=> valid`). So `compute_losses_v3` takes its *"no judgeable window in
    this batch"* branch and `loss_route` is a **grapheless `zeros(())`** — the route
    path is never exercised, on EITHER arm.

    ⇒ without this injection the bypass test would have passed for entirely the wrong
    reason: not "the term is gated" but "the term was never live". That is the
    programme's own `a guard that cannot fail` class, and only the powered control
    separates the two. The injection is the same device the sibling harness already
    uses for `lat_v7` / `lon_v7` / `nav_cmd`, which the smoke dataset also lacks.
    """
    H = _harness()
    T = H._trainer()
    args = T.build_parser().parse_args(H.LIVE_ARGV + list(extra_argv or []))
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model.train()
    # ⚠️ 2026-09-23: `LIVE_ARGV` declares `--anchor-v0-conditioned` over the smoke
    # config's registered ZERO `anchor_controls`, i.e. an EXACTLY degenerate bank -- every
    # candidate the same straight line, spread 0.000000000 m. The decoder now refuses that
    # state on the TENSOR (`refc.py`, the 2026-09-22 review's F9 finding), so this fixture
    # carries a real control ladder -- the same one `test_refc_sampler.py` adopted.
    _ac = model.core.decoder.anchor_controls
    _n = int(_ac.shape[0])
    with torch.no_grad():
        _ac.copy_(torch.stack([torch.linspace(-2.0, 2.0, _n),
                               torch.linspace(-1.5, 1.5, _n)], dim=-1).to(_ac.dtype))

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
    # ⭐ THE INJECTION. Targets are LITERAL and < 3, so the v2.1 contract's
    # `route < 3 <=> valid = True` holds and the CE has a real graph.
    batch["route_target"] = torch.tensor([0, 1], dtype=torch.long)
    batch["route_valid"] = torch.tensor([True, True])

    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    total = losses["loss"]
    assert total.requires_grad, "the total loss does not require grad -- UNPOWERED"
    total.backward()
    return model, losses


def _route_head_grad_state(model) -> tuple[int, int, float]:
    """-> (n_params, n_with_a_grad_object, sum|grad|) for the route head only."""
    head = getattr(getattr(model, "core", model), "route_head", None)
    if head is None:
        pytest.skip("this build has no route_head")
    n, got, tot = 0, 0, 0.0
    for p in head.parameters():
        n += 1
        if p.grad is not None:
            got += 1
            tot += float(p.grad.abs().sum())
    return n, got, tot


# --------------------------------------------------------------------------- #
# THE GUARD
# --------------------------------------------------------------------------- #

def test_route_head_takes_NO_gradient_under_the_strategic_bypass() -> None:
    """⛔ THE DEFECT ITSELF. With `--no-strategic`, backward on the trainer's own
    total loss must leave the route head unreached."""
    model, _losses = _build_and_backward_with_a_JUDGEABLE_route(["--no-strategic"])
    n, got, tot = _route_head_grad_state(model)
    assert n >= 2, "the route head has fewer parameters than expected"
    assert got == 0, (
        f"under --no-strategic the route head still received a gradient on "
        f"{got}/{n} tensors (sum|grad| {tot:.4f}). The route term is back in the "
        f"loss sum, and the strategic layer is shaping the shared trunk inside a "
        f"one-variable arm.")


def test_the_probe_is_POWERED_not_merely_quiet() -> None:
    """⭐ THE CONTROL THAT MAKES THE TEST ABOVE MEAN SOMETHING.

    Without the bypass the route head MUST receive a gradient. A probe that reads
    "no gradient" on both arms is measuring its own harness, not the gate — and the
    programme has shipped exactly that: an AST census that read 0 suspects on BOTH the
    fixed and the broken trainer."""
    model, _losses = _build_and_backward_with_a_JUDGEABLE_route()
    n, got, tot = _route_head_grad_state(model)
    assert got == n and tot > 0.0, (
        f"with the strategic layer ACTIVE the route head took a gradient on only "
        f"{got}/{n} tensors (sum|grad| {tot:.6f}) -- the probe cannot see the "
        f"gradient it is supposed to detect, so the bypass test proves nothing")


def test_the_trunk_is_reached_on_BOTH_arms_so_the_difference_is_the_route_path() -> None:
    """⭐ THE SECOND CONTROL, and it closes the alternative explanation.

    If the bypass arm's trunk took no gradient AT ALL, the first test would pass for
    the wrong reason — a dead backward rather than a gated term. The trunk must be
    reached in both arms; only the ROUTE path may differ."""
    reached = {}
    for tag, argv in (("bypass", ["--no-strategic"]), ("active", [])):
        model, _l = _build_and_backward_with_a_JUDGEABLE_route(argv)
        enc = getattr(getattr(model, "core", model), "encoder", None)
        if enc is None:
            pytest.skip("this build exposes no core.encoder")
        reached[tag] = sum(1 for p in enc.parameters() if p.grad is not None)
    assert reached["bypass"] > 0 and reached["active"] > 0, (
        f"the trunk must take gradient on both arms; got {reached}. A zero here "
        f"means the backward is dead, not that the route term is gated.")


def test_the_run_record_states_whether_the_route_term_was_applied() -> None:
    """⛔ A GATED TERM THAT THE RECORD DOES NOT MENTION IS UNRECONSTRUCTABLE.

    Two arms that differ in whether a loss term existed must be distinguishable from
    their `config.json` alone -- the same three-fact discipline `goal_str_loss_applied`
    already carries for the neighbouring term."""
    src = open(_TRAINER_PY, encoding="utf-8").read()
    assert '"route_loss_applied"' in src, (
        "the seam stamp carries no `route_loss_applied` field, so a --no-strategic "
        "arm's record cannot be told from one that trained the route head into the "
        "shared trunk")
    # the control: the neighbouring field it is modelled on is present too, so a
    # match here is not an artifact of searching a file that contains everything
    assert '"goal_str_loss_applied"' in src
