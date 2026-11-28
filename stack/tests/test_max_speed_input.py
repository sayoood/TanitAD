"""``--max-speed-input`` -- the pinned ladder, the units refusal, and the OFF proof.

⛔ THE POINT OF THIS FILE. Two obligations, and the second is the one that is
usually skipped:

  1. WITH THE FLAG OFF THE BUILD DID NOT MOVE -- bit-identical on a fixed seed.
  2. ⭐⭐ THAT ASSERTION CAN FAIL. Every "OFF is bit-identical" test in this
     programme carries a MUTATION CONTROL that reintroduces the defect and shows
     the same assertion going red. An equality that cannot fail proves nothing;
     it proves the two things being compared were never wired together at all.
     Pattern copied from ``test_goal_point_wiring.py`` (E15) and ``5e7001a``.

⚠️ SCOPE OF THE MODEL-LEVEL PROOF, STATED SO IT IS NOT OVER-READ. ``refc_v3.py``
has live siblings, so this package does NOT patch it; the exact two-site diff is
filed as an escalation
(``.../Research/2026-09-06-max-speed-input/WIRING_DIFF.md``). The model-level
tests here therefore inject at the TACTICAL half of the seam -- ``phi_tac``'s
output, which is literally what ``nav_to_tac`` and ``ego_to_tac`` are added to --
by wrapping the module. The strategic half is proved on the conditioner's own
arithmetic. What is NOT proved here is that someone applied the diff correctly;
that is what the diff's own parity test is for, and it is named in the report.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from tanitad.refs import max_speed_input as MSI
from tanitad.refs import refc_v3 as v3

KMH = 3.6


# ------------------------------------------------------------------ the flag

def test_the_flag_defaults_OFF_so_no_banked_recipe_moves():
    assert MSI.MaxSpeedConfig().enabled is False
    # same-breath POSITIVE control: the flag is real and can be turned on.
    assert MSI.MaxSpeedConfig(enabled=True).enabled is True


def test_quantized_is_the_DEFAULT_mode_and_raw_is_reachable():
    assert MSI.DEFAULT_MODE == "quantized"
    assert MSI.MaxSpeedConfig().mode == "quantized"
    assert set(MSI.MODES) == {"quantized", "raw"}
    # an unknown mode is refused at every entry point, not silently defaulted
    with pytest.raises(ValueError):
        MSI.MaxSpeedConditioner(4, 4, MSI.MaxSpeedConfig(mode="nonsense"))
    with pytest.raises(ValueError):
        MSI.encode_block([10.0], mode="nonsense")
    with pytest.raises(ValueError):
        MSI.artifact_meta("nonsense")
    # same-breath control: both REAL modes go through.
    for mode in MSI.MODES:
        assert MSI.encode_block([10.0], mode=mode).shape == (1, MSI.MAX_SPEED_DIMS)


# ------------------------------------------------------------- the step set

def test_the_pinned_ladder_is_the_measured_one():
    """⛔ If this changes, the ladder-selection evidence in the package is stale.
    The numbers it pins are in ``quantization_panel_train.json``."""
    assert MSI.POSTED_LIMIT_STEPS_KMH == (20, 30, 50, 70, 80, 100, 120, 130)
    assert MSI.POSTED_LIMIT_STEPS_MS == tuple(sorted(MSI.POSTED_LIMIT_STEPS_MS))
    assert MSI.V_SCALE_MS == pytest.approx(130 / KMH)


@pytest.mark.parametrize("v_kmh, want_kmh, over", [
    (0.0, 20, False),        # a STOPPED ego snaps to the LOWEST posted limit
    (19.9, 20, False),
    (20.0, 20, False),       # exactly on a step stays on it (<=, not <)
    (20.1, 30, False),
    (49.0, 50, False),
    (100.0, 100, False),
    (129.9, 130, False),
    (136.1, 130, True),      # ⭐ ABOVE the top real sign -> OVER THE CEILING
])
def test_snap_UP_and_the_over_ceiling_flag(v_kmh, want_kmh, over):
    q, o = MSI.quantize_up(v_kmh / KMH)
    assert q == pytest.approx(want_kmh / KMH)
    assert o is over


def test_the_ceiling_is_VIOLABLE_which_the_raw_value_is_not():
    """⭐ THE BEHAVIOURAL DIFFERENCE THAT JUSTIFIES QUANTIZING AT ALL. The raw
    value is max(ego's own future speed), so ``v_hi >= v`` holds for every frame
    in the band BY CONSTRUCTION -- it can never be exceeded. A quantized ceiling
    with a real top step CAN be."""
    fast = 137.0 / KMH
    q, over = MSI.quantize_up(fast)
    assert over is True and fast > q
    # same-breath control that must read the OTHER way, so a passing assertion is
    # not "quantize_up always says over".
    q2, over2 = MSI.quantize_up(60.0 / KMH)
    assert over2 is False and 60.0 / KMH <= q2


def test_quantize_up_array_matches_the_scalar_path():
    import numpy as np
    v = np.linspace(0.0, 40.0, 401)
    q, over = MSI.quantize_up_array(v)
    for i in range(0, len(v), 37):
        qs, os_ = MSI.quantize_up(float(v[i]))
        assert q[i] == pytest.approx(qs) and bool(over[i]) is os_


# ---------------------------------------------------------------- the units

def test_an_undeclared_unit_is_REFUSED():
    with pytest.raises(MSI.MaxSpeedUnitsMissing):
        MSI.read_max_speed_field({"v_max_ms": 11.19})
    # same-breath POSITIVE control: a DECLARED payload reads, so the refusal is
    # about the missing declaration and not about the reader being broken.
    got = MSI.read_max_speed_field({"v_max_ms": 11.19, "units": "m/s"})
    assert got["valid"] is True and got["control_units"] == "m_s"
    assert got["units_source"] == "artifact"


def test_a_declared_nonSI_unit_is_CONVERTED_not_guessed():
    a = MSI.read_max_speed_field({"v_max_ms": 50.0, "units": "km/h"})
    b = MSI.read_max_speed_field({"v_max_ms": 50.0 / KMH, "units": "m/s"})
    assert a["raw_ms"] == pytest.approx(b["raw_ms"])
    assert a["value_ms"] == pytest.approx(b["value_ms"])
    # ...and a unit nobody has heard of is refused rather than defaulted.
    with pytest.raises(MSI.MaxSpeedUnitsUnknown):
        MSI.read_max_speed_field({"v_max_ms": 50.0, "units": "furlongs/fortnight"})


def test_the_legacy_override_is_RECORDED_not_silent():
    got = MSI.read_max_speed_field({"v_max_ms": 50.0}, units_override="km/h")
    assert got["units_source"] == "caller-override"
    assert got["raw_ms"] == pytest.approx(50.0 / KMH)


def test_every_emitted_artifact_declares_its_units():
    for mode in MSI.MODES:
        meta = MSI.artifact_meta(mode)
        assert meta["control_units"] == "m_s" and meta["control_units"]
        assert meta["mode"] == mode
    assert MSI.artifact_meta("quantized")["steps_kmh"] == list(
        MSI.POSTED_LIMIT_STEPS_KMH)
    assert MSI.artifact_meta("raw")["steps_kmh"] is None


def test_the_v8_label_payload_declares_units_and_reads():
    """⛔ THE REAL PAYLOAD, not a fixture of one. ``build_v8_speedmax.py`` writes
    ``units: "m/s"`` into every ``speed_max_input`` block; if that ever stops, the
    consumer must refuse rather than assume, and this test says so."""
    payload = {"schema": "speed_max_input/1", "v_max_ms": 11.19,
               "v_min_ms": 8.64, "units": "m/s", "band_s": [2.0, 6.0],
               "provenance": "ego-future", "oracle": True}
    got = MSI.read_max_speed_field(payload)
    assert got["valid"] and got["quantized_ms"] == pytest.approx(50.0 / KMH)
    stripped = {k: v for k, v in payload.items() if k != "units"}
    with pytest.raises(MSI.MaxSpeedUnitsMissing):
        MSI.read_max_speed_field(stripped)


# ------------------------------------------------------------- the block

def test_the_validity_bit_GATES_the_value():
    """X15's rule: "withheld" and "genuinely zero" must differ in the FLAG, and
    that only works if the values really are zero when the flag is."""
    blk = MSI.encode_block([30.0, 30.0], [1.0, 0.0])
    assert torch.equal(blk[1], torch.zeros(MSI.MAX_SPEED_DIMS))
    assert blk[0, MSI.VALID_SLOT] == 1.0 and blk[0, 0] > 0.0


def test_a_zero_ceiling_and_a_withheld_ceiling_are_DIFFERENT_inputs():
    zero = MSI.encode_block([0.0], [1.0])       # "the limit here is very low"
    held = MSI.encode_block([0.0], [0.0])       # "no limit known"
    assert not torch.equal(zero, held)
    assert zero[0, MSI.VALID_SLOT] == 1.0 and held[0, MSI.VALID_SLOT] == 0.0


def test_raw_mode_does_not_quantize():
    q = MSI.encode_block([11.19], mode="quantized")
    r = MSI.encode_block([11.19], mode="raw")
    assert not torch.equal(q, r)
    assert r[0, 0] == pytest.approx(11.19 / MSI.V_SCALE_MS)


# --------------------------------------------- the conditioner: OFF and ON

def _cond(d_tac=32, d_ctx=8, seed=0, mode="quantized"):
    torch.manual_seed(seed)
    return MSI.MaxSpeedConditioner(d_tac, d_ctx,
                                   MSI.MaxSpeedConfig(enabled=True, mode=mode))


def test_nothing_fed_means_no_delta_at_all():
    dt, ds = _cond()(None)
    assert dt is None and ds is None


def test_the_edge_is_BIT_INERT_at_init():
    """Zero-init on both output projections, so an ON build equals an OFF build
    at step 0 -- E13's and E11''s contract verbatim."""
    dt, ds = _cond()(torch.tensor([11.19, 30.0]))
    assert torch.equal(dt, torch.zeros_like(dt))
    assert torch.equal(ds, torch.zeros_like(ds))


def test_the_edge_is_REACHABLE_once_the_projections_are_trained():
    """⛔ THE HALF A ZERO-INIT CHECK CANNOT GIVE YOU. An edge that is bit-inert at
    init AND unreachable after training is a DEAD WIRE that reads as a clean
    ablation."""
    c = _cond()
    v = torch.tensor([11.19, 30.0])
    with torch.no_grad():
        torch.manual_seed(1)
        c.to_tac.weight.normal_(std=0.5)
        c.to_str.weight.normal_(std=0.5)
        dt, ds = c(v)
    assert not torch.equal(dt, torch.zeros_like(dt))
    assert not torch.equal(ds, torch.zeros_like(ds))


def test_the_VALUE_reaches_the_delta_not_only_its_presence():
    """The E13 failure mode tested for directly: refcv4b's nav edge moved g_str
    18.6x more when REMOVED than when its VALUE changed -- a presence-gated bias."""
    c = _cond()
    with torch.no_grad():
        torch.manual_seed(2)
        c.to_tac.weight.normal_(std=0.5)
        a, _ = c(torch.tensor([30.0 / KMH]))       # 30 km/h bin
        b, _ = c(torch.tensor([120.0 / KMH]))      # 120 km/h bin
    assert not torch.equal(a, b)


# ----------------------------------------- ⭐⭐ THE MODEL-LEVEL OFF PROOF

class _AddDelta(nn.Module):
    """Wrap ``phi_tac`` and add the conditioner's tactical delta to its output --
    which is EXACTLY what ``z_tac_raw = z_tac_raw + nav_to_tac(e)`` does at the
    real seam."""

    def __init__(self, inner: nn.Module, cond: MSI.MaxSpeedConditioner,
                 v_max: torch.Tensor | None):
        super().__init__()
        self.inner, self.cond, self.v_max = inner, cond, v_max

    def forward(self, x):
        z = self.inner(x)
        if self.v_max is None:
            return z
        dt, _ = self.cond(self.v_max)
        return z + dt.to(z.dtype)


def _build(seed: int = 0):
    cfg = v3.refc_v3_smoke_config(hier=True)
    torch.manual_seed(seed)
    m = v3.RefCV3Model(cfg).eval()
    g = torch.Generator().manual_seed(7)
    h, w = cfg.core.encoder.image_hw()
    f = torch.rand(2, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                   generator=g)
    return cfg, m, f


def _run(m, f):
    with torch.no_grad():
        return m(f, v0=torch.tensor([3.0, 7.0]), steps=2)


def _same(a, b) -> bool:
    """⛔ FAILS LOUD ON A VACUOUS COMPARISON. A predicate that compared zero
    tensors would return True forever and every parity claim resting on it would
    be worthless -- the exact failure this file's mutation control exists to
    catch, one level down."""
    keys = ("traj", "z_tac", "g_tac", "g_str", "ctx", "anchor_logits")
    seen = 0
    for k in keys:
        if k in a and torch.is_tensor(a[k]):
            seen += 1
            if not torch.equal(a[k], b[k]):
                return False
    assert seen >= 3, f"parity predicate compared only {seen} tensors -- vacuous"
    return True


def test_OFF_is_BIT_IDENTICAL_on_a_fixed_seed():
    """⭐ THE ASSERTION. Attaching the conditioner and feeding it NOTHING leaves
    the model bit-identical to the untouched build on a fixed seed."""
    cfg, m, f = _build()
    before = _run(m, f)
    m.phi_tac = _AddDelta(m.phi_tac,
                          _cond(cfg.d_tac, cfg.core.strategic.d_ctx),
                          v_max=None)                     # flag OFF: nothing fed
    after = _run(m, f)
    assert _same(before, after)
    # the model also gained NO max-speed parameter it could train by accident
    assert not [k for k in v3.RefCV3Model(cfg).state_dict()
                if "max_speed" in k or "vmax" in k]


def test_ON_at_INIT_is_also_BIT_IDENTICAL():
    """The zero-init half: the flag ON, a real ceiling fed, output unchanged at
    step 0. This is what makes an ON-vs-OFF training comparison attributable."""
    cfg, m, f = _build()
    before = _run(m, f)
    m.phi_tac = _AddDelta(m.phi_tac,
                          _cond(cfg.d_tac, cfg.core.strategic.d_ctx),
                          v_max=torch.tensor([11.19, 30.0]))
    assert _same(before, _run(m, f))


def test_THE_MUTATION_CONTROL_the_bit_identity_assertion_CAN_FAIL():
    """⭐⭐ THE CONTROL THAT MAKES THE TWO TESTS ABOVE MEAN SOMETHING.

    Reintroduce the defect -- a NON-zero-init output projection, i.e. an edge
    that is live at step 0 -- and the very same ``_same(...)`` predicate that
    passed above must now read False. If it still read True, the two builds were
    never wired to each other and the passing assertions were vacuous.
    """
    cfg, m, f = _build()
    before = _run(m, f)
    c = _cond(cfg.d_tac, cfg.core.strategic.d_ctx)
    with torch.no_grad():
        torch.manual_seed(3)
        c.to_tac.weight.normal_(std=0.5)                  # THE MUTATION
        c.to_tac.bias.normal_(std=0.5)
    m.phi_tac = _AddDelta(m.phi_tac, c, v_max=torch.tensor([11.19, 30.0]))
    assert not _same(before, _run(m, f)), (
        "the bit-identity predicate could not detect a LIVE edge -- it is "
        "vacuous, and every OFF-parity claim resting on it is worthless")


def test_a_WITHHELD_ceiling_carries_NO_VALUE_even_with_a_live_edge():
    """⛔ THE THIRD CORNER, and it is NOT "the delta is zero".

    With the projections mutated the edge is live, so a withheld row still emits
    the BIAS path -- and that is CORRECT and is what ``ego_inj`` does with
    ``keep = 0``: "no ceiling known" must stay a distinct, LEARNABLE input rather
    than collapsing onto "the limit here is 0 m/s". What must be true is that the
    withheld delta does not depend on the withheld VALUE. Two very different
    ceilings, both withheld, must be bit-identical.

    ⚠️ Asserting `delta == 0` here would have been the wrong test and would have
    forced a design in which "withheld" and "0 m/s ceiling" are the same input.
    """
    c = _cond()
    with torch.no_grad():
        torch.manual_seed(3)
        c.to_tac.weight.normal_(std=0.5)
        c.to_tac.bias.normal_(std=0.5)
        c.proj.bias.normal_(std=0.5)
        off = torch.zeros(2)
        a_dt, a_ds = c(torch.tensor([5.0, 5.0]), off)
        b_dt, b_ds = c(torch.tensor([36.0, 36.0]), off)
        # same-breath control: with the SAME live edge and valid = 1, the two
        # values must NOT agree -- otherwise "identical when withheld" would be
        # explained by the edge being dead, not by the gate working.
        on = torch.ones(2)
        c_dt, _ = c(torch.tensor([5.0, 5.0]), on)
        d_dt, _ = c(torch.tensor([36.0, 36.0]), on)
    assert torch.equal(a_dt, b_dt) and torch.equal(a_ds, b_ds), (
        "a withheld ceiling still carried its VALUE: the validity bit does not "
        "gate the value, which is the `_ensure_ego` default trap in a speed "
        "costume")
    assert not torch.equal(c_dt, d_dt), (
        "the edge is dead, so 'identical when withheld' proves nothing")
