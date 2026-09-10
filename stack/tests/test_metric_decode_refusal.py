"""The metric-decode refusal: `step_readout_op` at init may not emit metres.

⛔ THE DEFECT THIS PINS (MEASURED 2026-09-06). `step_readout_op` is the metric
trajectory readout (latent transition -> per-step dpose). It is reached ONLY
from `stage_a_losses`, inside `if w.o1_ctrl or w.o1_fact or w.o1_scene:`, and
the v7 recipe sets all three to 0.0 -- so it sits at random init for the whole
run. Across the 9 banked v7-tiny checkpoints it takes THREE distinct
fingerprints while the trained control `predictor_op.heads.1.weight` takes
NINE, and its LayerNorm is bit-exactly ones/zeros in all 9. Meanwhile SEVEN
banked T1 artifacts record `decoder = grounding.step['op']` on a v6/v7
checkpoint: ADE, FDE, heading and speed in METRES really did come out of a
random projection.

⭐ TWO CLAIMS, TESTED SEPARATELY -- the lesson its predecessor earned:
*"correctness and wiring are different claims, and only the first was ever
tested."*

  * §1-2 CORRECTNESS: the detector fires on a genuinely at-init readout and
    does NOT fire on a genuinely trained one. A guard that refuses everything
    gets deleted; a guard that refuses nothing measured nothing.
  * §3 WIRING: every path that can produce a metric trajectory ACTUALLY CALLS
    it -- proven by spying on the guard and executing the path, not by reading
    the source.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tanitad.models.metric_dynamics import StepDisplacementReadout
from tanitad.models.v6 import (UntrainedMetricReadout,
                               assert_metric_readout_trained,
                               metric_readout_status)

SD = 64          # state_dim; net.1 is Linear(128, 512) -> n = 65,536 >= MIN_N


@pytest.fixture
def tiny_stack():
    """FUNCTION-scoped on purpose: the wiring tests swap `step_readout_op` for
    a trained one to exercise the non-firing control, and a shared stack would
    carry that mutation (and the guard's memoised status) into its neighbours."""
    from tanitad.models.v6 import V6Stack
    return V6Stack()


def fresh(seed: int = 0) -> StepDisplacementReadout:
    torch.manual_seed(seed)
    return StepDisplacementReadout(SD)


def trained_by_n_adamw_steps(n: int = 1, seed: int = 0
                             ) -> StepDisplacementReadout:
    """A readout that REALLY trained, at the v7-tiny arm's own hyper-params.

    ⭐ n=1 is the strict form. MEASURED on a REAL artifact -- a flagship v4
    `grounding['step.op.net.1.weight']` banked at `step = 1` -- `max|w|/bound`
    already reads 1.0078, i.e. ONE optimizer step moves a weight past the init
    bound. So the positive control does not need a long run to be honest.
    """
    m = fresh(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
    g = torch.Generator().manual_seed(7)
    for _ in range(n):
        opt.zero_grad()
        z = torch.randn(8, SD, generator=g)
        m(z, torch.randn(8, SD, generator=g)).pow(2).mean().backward()
        opt.step()
    return m


# --------------------------------------------------------------------------- #
# 1. CORRECTNESS -- the NEGATIVE side (must fire)                              #
# --------------------------------------------------------------------------- #
def test_a_fresh_readout_reads_AT_INIT_on_every_evidence_channel():
    st = metric_readout_status(fresh())
    assert st["verdict"] == "AT_INIT" and not st["safe_to_decode"]
    assert st["evidence"]["layernorm_at_init"] is True
    assert st["evidence"]["linear_at_init"] is True
    assert st["n_tested"] >= 4, st


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_the_verdict_does_not_depend_on_the_init_SEED(seed):
    """Whichever seed drew it, an unfitted readout is unfitted -- and the
    emitted trajectory is a function of that seed, which is the whole point."""
    assert metric_readout_status(fresh(seed))["verdict"] == "AT_INIT"


def test_it_refuses_and_the_message_names_the_call_site_and_the_remedy():
    with pytest.raises(UntrainedMetricReadout) as ei:
        assert_metric_readout_trained(fresh(), where="UNIT_TEST_SITE")
    msg = str(ei.value)
    assert "UNIT_TEST_SITE" in msg
    assert "RANDOM PROJECTION" in msg
    assert "FITTED AT EVAL TIME" in msg        # the remedy, not just the verdict


# --------------------------------------------------------------------------- #
# 2. CORRECTNESS -- the POSITIVE side (must NOT fire)                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_steps", [1, 2, 5])
def test_a_readout_that_REALLY_TRAINED_passes(n_steps):
    """⛔ THE HALF THAT STOPS THIS GUARD BEING DELETED. One AdamW step at the
    arm's own `lr 1e-4, wd 0.05` is enough."""
    m = trained_by_n_adamw_steps(n_steps)
    st = metric_readout_status(m)
    assert st["verdict"] == "TRAINED" and st["safe_to_decode"], st
    assert_metric_readout_trained(m, where="positive control")   # must not raise


def test_the_detection_floor_AT_THE_REAL_v7_GEOMETRY_is_about_1e_minus_5():
    """⚠ THE FLOOR CARRIES ITS `n` OR IT IS NOT QUOTABLE. `max|w|` sits
    ~1/n below the init bound, so the smallest detectable move SCALES WITH THE
    TENSOR. MEASURED: flip at rel 3e-5 for n = 65,536 and at rel 1e-5 for both
    n = 524,288 and n = 2,097,152 -- the last being `step_readout_op.net.1` at
    the banked v7-tiny geometry (state_dim 2048). This test runs at THAT
    geometry, not at a convenient small one: a floor measured at the wrong size
    is the `df`/`step_s` scope error in a threshold's costume.

    ⚠ And the boundary is DRAW-DEPENDENT near 1e-6 -- whether a perturbation
    crosses the bound depends on whether it lands near the element already
    closest to it -- so the test BRACKETS the floor (1e-7 below, 1e-5 above)
    instead of pinning a knife-edge that would flake. What matters is the
    margin against the thing being guarded: ONE AdamW step at the arm's own
    `lr 1e-4, wd 0.05` reads `max_ratio 1.0064`, some three orders of magnitude
    past this floor.
    """
    m = StepDisplacementReadout(2048)             # the banked geometry exactly
    assert m.net[1].weight.numel() == 512 * 4096
    base = {k: v.detach().clone() for k, v in m.state_dict().items()}
    verdicts = {}
    for rel in (0.0, 1e-7, 1e-5, 1e-4, 1e-2):
        m.load_state_dict(base)
        g = torch.Generator().manual_seed(0)      # the SAME draw at every rel
        with torch.no_grad():
            w = m.net[1].weight
            w.add_(torch.randn(w.shape, generator=g) * float(w.std()) * rel)
        verdicts[rel] = metric_readout_status(m, refresh=True)["verdict"]
    assert verdicts[0.0] == verdicts[1e-7] == "AT_INIT", verdicts
    assert verdicts[1e-5] == verdicts[1e-4] == verdicts[1e-2] == "TRAINED",         verdicts


def test_max_abs_over_the_init_BOUND_is_the_load_bearing_statistic():
    """`nn.Linear` draws U(-b, b) with b = 1/sqrt(fan_in), so a fresh tensor
    sits just BELOW the bound and a trained one crosses it. Stated as a number
    so a future refactor that changes the init has to face it."""
    m = fresh()
    st = metric_readout_status(m)
    ev = st["linear"]["net.1.weight"]
    assert ev["bound"] == pytest.approx(1.0 / math.sqrt(2 * SD))
    assert ev["max_ratio"] <= 1.0
    t = metric_readout_status(trained_by_n_adamw_steps(1))
    assert t["linear"]["net.1.weight"]["max_ratio"] > 1.0


def test_tiny_tensors_are_SKIPPED_because_their_order_statistics_are_noise():
    """MEASURED: `net.3.bias` has n = 3 and reads `std_ratio 1.583` at a
    genuine fresh init. Testing it would make the guard refuse a correct
    readout at random, which is how guards get deleted."""
    st = metric_readout_status(fresh())
    assert "net.3.bias" not in st["linear"]
    assert "net.1.weight" in st["linear"] and "net.3.weight" in st["linear"]


# --------------------------------------------------------------------------- #
# 2b. The RECORD channel, and the contradiction it can expose                  #
# --------------------------------------------------------------------------- #
def test_the_run_record_alone_can_convict_when_the_weights_are_unreadable():
    """`w_o1_* == 0` means `stage_a_losses` never ran, which is the ONLY
    gradient path to the readout. Corroborating, never sufficient alone -- so
    it is exercised on a module with no testable Linear tensor."""
    args = {"w_o1_ctrl": 0.0, "w_o1_fact": 0.0, "w_o1_scene": 0.0}
    empty = nn.Sequential(nn.LayerNorm(8))                 # LN only, no Linear
    st = metric_readout_status(empty, run_args=args)
    assert st["verdict"] == "AT_INIT"
    assert metric_readout_status(nn.Sequential(nn.LayerNorm(8)),
                                 run_args=None)["verdict"] == "UNKNOWN"


def test_moved_weights_plus_a_zero_O1_record_is_a_CONTRADICTION_and_refuses():
    """The state that should never exist: the record says no gradient could
    reach the readout, and the weights moved anyway. One of the two facts is
    wrong, so a number decoded through it is the worst case, not the best."""
    m = trained_by_n_adamw_steps(2)
    args = {"w_o1_ctrl": 0.0, "w_o1_fact": 0.0, "w_o1_scene": 0.0}
    assert metric_readout_status(m, run_args=args,
                                 refresh=True)["verdict"] == "CONTRADICTION"
    with pytest.raises(UntrainedMetricReadout):
        assert_metric_readout_trained(m, where="contradiction", run_args=args)


def test_the_MEMO_is_keyed_on_the_record_and_cannot_answer_another_question():
    """⚠️ A BUG THIS TEST EXISTS BECAUSE OF. The status is memoised (the
    fingerprint is a full pass over ~2.1 M weights and `roll_consistency` runs
    inside an MPC inner loop). The first version keyed the memo on the MODULE
    alone, so a call carrying an `w_o1_* == 0` record cached CONTRADICTION and
    the next call WITHOUT the record read that back instead of TRAINED -- a
    cache silently answering a DIFFERENT question from the one asked, which is
    this programme's scope error wearing a memo."""
    m = trained_by_n_adamw_steps(2)
    zero = {"w_o1_ctrl": 0.0, "w_o1_fact": 0.0, "w_o1_scene": 0.0}
    assert metric_readout_status(m, run_args=zero)["verdict"] == "CONTRADICTION"
    assert metric_readout_status(m, run_args=None)["verdict"] == "TRAINED"
    assert metric_readout_status(m, run_args=zero)["verdict"] == "CONTRADICTION"
    live = {"w_o1_ctrl": 1.0, "w_o1_fact": 0.0, "w_o1_scene": 0.0}
    assert metric_readout_status(m, run_args=live)["verdict"] == "TRAINED"


def test_UNKNOWN_refuses_because_could_not_check_is_not_a_pass():
    with pytest.raises(UntrainedMetricReadout):
        assert_metric_readout_trained(nn.Identity(), where="unknown module")


def test_allow_is_an_ESCAPE_HATCH_that_still_returns_the_verdict():
    """`D-V7F-READOUT-DEAD` diagnosed the readout ITSELF, legitimately. The
    opt-in must therefore exist -- and must hand back the verdict so the
    caller can stamp it into the artifact rather than merely know it."""
    st = assert_metric_readout_trained(fresh(), where="diagnosis", allow=True)
    assert st["verdict"] == "AT_INIT" and st["allowed"] is True


# --------------------------------------------------------------------------- #
# 3. ⭐⭐ WIRING -- every metric-trajectory path ACTUALLY CALLS the guard        #
# --------------------------------------------------------------------------- #
class _Spy:
    """Records calls to the guard and then delegates to the real one."""

    def __init__(self, monkeypatch, *modules):
        self.calls = []
        import tanitad.models.v6 as v6
        real = v6.assert_metric_readout_trained

        def spy(readout, *, where, run_args=None, allow=False):
            self.calls.append(where)
            return real(readout, where=where, run_args=run_args, allow=allow)

        for m in modules:
            monkeypatch.setattr(m, "assert_metric_readout_trained", spy,
                                raising=True)

    @property
    def sites(self):
        return list(self.calls)


def test_WIRING_V6Grounding_step_calls_the_guard_and_refuses(monkeypatch,
                                                             tiny_stack):
    """`t1_eval.py --grounding-readout` reaches the readout ONLY as
    `grounding.step["op"]`, so this expression refusing IS t1_eval refusing."""
    import tanitad.eval.v6_probe_trunk as vpt
    spy = _Spy(monkeypatch, vpt)
    g = vpt.V6Grounding(tiny_stack)
    with pytest.raises(UntrainedMetricReadout):
        g.step["op"]
    assert any("V6Grounding.step" in s for s in spy.sites), spy.sites
    # the NON-FIRING control: the same expression on a trained readout works
    g2 = vpt.V6Grounding(tiny_stack)
    g2.stack.step_readout_op = trained_by_n_adamw_steps(1)
    assert g2.step["op"] is g2.stack.step_readout_op


def test_WIRING_V6Grounding_records_the_verdict_for_the_artifact(tiny_stack):
    import tanitad.eval.v6_probe_trunk as vpt
    g = vpt.V6Grounding(tiny_stack, allow_untrained_readout=True)
    assert g.readout_status["verdict"] == "AT_INIT"


def test_WIRING_roll_consistency_calls_the_guard(monkeypatch, tiny_stack):
    """`roll_consistency` returns METRES (`accumulate_se2` over the readout's
    dposes), so it is a metric-trajectory path even though its two internal
    uses are a variance signal and a zero-weighted regulariser."""
    import tanitad.models.v6 as v6
    spy = _Spy(monkeypatch, v6)
    b, n, k = 1, 2, 2
    w = int(tiny_stack.cfg.predictor.window)
    states = torch.zeros(b, w, tiny_stack.cfg.d_op)
    actions = torch.zeros(b, w, 3)
    a_ctl = torch.zeros(b, n, k)
    kappa = torch.zeros(b, n, k)
    v0 = torch.full((b,), 5.0)
    with pytest.raises(UntrainedMetricReadout):
        tiny_stack.roll_consistency(states, actions, a_ctl, kappa, v0, k=k)
    assert any("roll_consistency" in s for s in spy.sites), spy.sites


def test_WIRING_probe_saliency_p9_builtin_targets_calls_the_guard(monkeypatch,
                                                                  tiny_stack):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import probe_saliency_p9 as p9
    spy = _Spy(monkeypatch, p9)
    with pytest.raises(UntrainedMetricReadout):
        p9.builtin_targets(tiny_stack)
    assert any("probe_saliency_p9" in s for s in spy.sites), spy.sites
    tiny_stack.step_readout_op = trained_by_n_adamw_steps(1)
    assert len(p9.builtin_targets(tiny_stack)) == 2      # non-firing control


def test_WIRING_load_trunk_auto_threads_the_opt_in(monkeypatch, tiny_stack):
    """The flag has to REACH `V6Grounding`, or `t1_eval`'s own flag is a
    no-op that reads like a decision."""
    import tanitad.eval.v6_probe_trunk as m
    from tanitad.eval.v6_probe_trunk import V6ProbeTrunk
    monkeypatch.setattr(m, "load_v6_from_ck",
                        lambda ck, dev, **kw: (V6ProbeTrunk(tiny_stack), 1))
    _, g_off, _ = m.load_trunk_auto({"stack": {}, "config": {}}, "cpu")
    with pytest.raises(UntrainedMetricReadout):
        g_off.step["op"]
    _, g_on, _ = m.load_trunk_auto({"stack": {}, "config": {}}, "cpu",
                                   allow_untrained_readout=True)
    assert g_on.step["op"] is tiny_stack.step_readout_op


def test_WIRING_t1_eval_passes_the_flag_and_stamps_the_verdict():
    """A SOURCE assertion, and it is the right instrument here: running a T1
    rollout needs a corpus. It pins the two lines that make `t1_eval`'s own
    flag real -- the pass-through and the artifact stamp -- each with a
    same-breath control token that must also be present."""
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2] / "taniteval" / "tools"
         / "t1_eval.py")
    src = p.read_text(encoding="utf-8", errors="replace")
    assert "--allow-untrained-readout" in src
    assert "allow_untrained_readout=bool(" in src         # the pass-through
    assert '"readout_training_status"' in src             # the artifact stamp
    # same-breath NON-ZERO control: tokens that must be there either way
    assert "load_trunk_auto(" in src and "--grounding-readout" in src


def test_WIRING_every_consumer_that_decodes_METRES_can_be_named_and_survives():
    """⭐ THE SIXTH CONSUMER, found by tracing rather than by reading the
    register. `stage_a_probes.py` decodes `decode_transitions(
    grounding.step["op"], ...)` -- metres -- and was NOT in the hazard row's
    list of three. It is covered for free by the `V6Grounding.step` chokepoint,
    which is the argument for putting the refusal there; what it still needed
    was an OPT-IN, because a guard that leaves a live tool unrunnable with no
    way through is the shape that gets deleted.

    MEASURED: zero banked `stage_a_probes` artifacts exist on a v6/v7 stack
    (2,616 JSON files scanned, 0 unreadable), so refusing it retracts nothing.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for rel, needle in (
            ("stack/scripts/stage_a_probes.py", "allow_untrained_readout"),
            ("stack/scripts/probe_saliency_p9.py", "allow_untrained_readout"),
            ("taniteval/tools/t1_eval.py", "allow_untrained_readout")):
        src = (root / rel).read_text(encoding="utf-8", errors="replace")
        assert "--allow-untrained-readout" in src, rel   # the opt-in exists
        assert needle in src, rel                        # and it is threaded
        assert "grounding.step" in src or "step_readout_op" in src, rel


def test_WIRING_the_guard_has_callers_OUTSIDE_its_own_test_file():
    """⛔ The failure this whole family exists for: `assert_frozen_external`
    was fully built, pinned in both directions, and called by NOTHING outside
    its own test file -- so the trap it guarded happened anyway."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    tok = "assert_metric_readout_trained"
    hits = {p.relative_to(root).as_posix()
            for p in list(root.glob("stack/**/*.py")) + list(
                root.glob("taniteval/**/*.py"))
            if p.name != Path(__file__).name
            and tok in p.read_text(encoding="utf-8", errors="replace")}
    prod = {h for h in hits if "/tests/" not in h}
    assert len(prod) >= 3, sorted(hits)
    assert any("v6_probe_trunk" in h for h in prod), sorted(prod)
    assert any("models/v6.py" in h for h in prod), sorted(prod)
    assert any("probe_saliency_p9" in h for h in prod), sorted(prod)
