"""D-REFCV4B-ASTAR-GEOMETRY -- the two-sided mutation gate for `a_star`.

WHAT BROKE. `taniteval/tools/refcv3_arm.py` hoisted a loop-invariant
`anchors_bank = model.core.decoder.anchors` and argmin'd the GT-nearest anchor
against it. The trainer (`compute_losses_v3` in `stack/scripts/refc_v3_train.py`)
measures `a_star` against `out["anchor_bank"]` -- the bank the forward ACTUALLY
DECODED -- and says so in a comment that names this exact failure. On a
v0-CONDITIONED vocabulary the two objects are different geometries, so the eval
argmin'd an index in the reference-speed family and then read that index back
out of the window's own fan: a different trajectory. The `oracle_sel` "ceiling"
read +0.9179 m separated WORSE than the arm it bounds -- a ceiling below its
own floor.

WHY A MUTATION GATE AND NOT AN INSPECTION. A source census reads the same on a
fixed and a broken trainer (measured elsewhere in this programme), so this file
does not grep for the bad attribute. It re-introduces the defect by feeding the
SHIPPED argmin the WRONG bank and requires the check to FAIL, then feeds it the
right bank and requires it to PASS.

THE THIRD CLAUSE IS THE LOAD-BEARING ONE. On a FIXED vocabulary the mutation
must be INVISIBLE -- `roll_bank` returns `self.anchors[None].expand(...)`, the
same storage -- because that is precisely why the defect survived review on
refcv3 and why correcting it cannot move refcv3's numbers. A gate that cannot
fail measures nothing; a gate that fails everywhere measures nothing either.

ASCII-ONLY BY CHOICE: assertion text reaches a cp1252 console on this dev box,
where a non-ASCII byte in a failure message is fatal and would hide the failure
it is reporting.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "refcv3_arm.py"

HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
N_STEPS = len(HORIZONS)
REF_SPEED = 10.0
#: 13 x 9 = 117, the refcv4b fan width. Column 0 is accel (m/s^2), column 1 is
#: LATERAL ACCELERATION (m/s^2) -- `--anchor-control-units alat`. Reading
#: column 1 as curvature is its own documented trap and is not what this fan is.
N_ACC, N_LAT = 13, 9


def _load_tool():
    """Import the tool by path (the pattern `test_render_refcv3_video` uses)."""
    if not TOOL.exists():
        pytest.skip(f"tool not present: {TOOL}")
    try:
        spec = importlib.util.spec_from_file_location("_refcv3_arm_ut", TOOL)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_refcv3_arm_ut"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as ex:                       # pragma: no cover
        pytest.skip(f"refcv3_arm is not importable here: {ex!r}")


def _decoder(torch, refc, *, v0_conditioned: bool):
    """A REAL `AnchoredDiffusionDecoder`, built the way refcv4b / refcv3 are.

    For the v0-conditioned build `anchors` is set to the SAME family rolled at
    `ref_speed_ms`, which is what the decoder's own docstring says the
    checkpoint holds -- so the mutation below substitutes exactly the object a
    real refcv4b checkpoint would hand it, not a strawman.
    """
    acc = torch.linspace(-4.0, 2.0, N_ACC)
    lat = torch.linspace(-3.0, 3.0, N_LAT)
    ctrl = torch.stack([acc[:, None].expand(N_ACC, N_LAT).reshape(-1),
                        lat[None, :].expand(N_ACC, N_LAT).reshape(-1)], -1)
    n = ctrl.shape[0]
    assert n == 117, n
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=64, n_steps=N_STEPS, d_meas=8, d_ctx=8, tac_latent_dim=8,
        anchors=torch.zeros(n, N_STEPS, 2), cfg=refc.DecoderConfig(),
        hierarchy=False, graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=HORIZONS,
        v0_conditioned=v0_conditioned, ref_speed_ms=REF_SPEED,
        control_units="alat", alat_v_floor_ms=4.0, kappa_cap=0.12)
    with torch.no_grad():
        dec.anchor_controls.copy_(ctrl)
        # the checkpoint-visible `anchors`: this family at the REFERENCE speed.
        # `roll_bank(None, ...)` is the reference-speed roll by definition.
        was = dec.anchor_v0_cond
        dec.anchor_v0_cond = True
        ref = dec.roll_bank(None, None, 1, torch.float32)[0]     # [N, S, 2]
        dec.anchor_v0_cond = was
        dec.anchors.copy_(ref)
    return dec


def _gt_windows(torch, kin, n: int = 48):
    """GT paths + their v0, rolled through the programme's OWN integrator.

    Speeds span 4-30 m/s so most windows sit far from the 10 m/s reference --
    the regime the defect lives in. Controls are drawn OFF the anchor grid but
    INSIDE the family's reachable span, through the same `alat -> kappa`
    conversion `roll_bank` applies, so the fan floor is a real nearest-anchor
    distance. Feeding raw curvature here instead would put every GT path
    outside the vocabulary at speed and inflate the floor ~15x -- a fixture
    artifact that would make the gate's margin look larger than it is.
    """
    g = torch.Generator().manual_seed(20260906)
    v0 = 4.0 + 26.0 * torch.rand(n, generator=g)
    a = -4.0 + 6.0 * torch.rand(n, generator=g)
    alat = -3.0 + 6.0 * torch.rand(n, generator=g)
    kap = (alat / v0.clamp_min(4.0) ** 2).clamp(-0.12, 0.12)
    ctrl = torch.stack([a, kap], -1)[:, None, :].expand(n, 60, 2).contiguous()
    st = torch.zeros(n, 4)
    st[:, 3] = v0
    path = kin.rollout_unicycle(st, ctrl, dt=0.1)[..., :2]
    slots = [h - 1 for h in HORIZONS]
    return path[:, slots].contiguous(), v0                       # [n, S, 2]


def _pair(torch, tool, dec, gt, v0):
    """(correct a_star, mutated a_star) per window, from the SHIPPED argmin."""
    sv = torch.ones(N_STEPS)
    right, wrong = [], []
    for i in range(gt.shape[0]):
        tgt = gt[i:i + 1]                                        # [1, S, 2]
        good = dec.roll_bank(v0[i:i + 1], None, 1, torch.float32)
        bad = dec.anchors[None].to(torch.float32)                # THE DEFECT
        right.append(tool._oracle_anchor_index(good, tgt, sv))
        wrong.append(tool._oracle_anchor_index(bad, tgt, sv))
    return right, wrong


def _ade(torch, bank, idx, gt):
    """Mean L2 over slots of anchor `idx` against `gt`, per window."""
    return torch.stack([
        (bank[i, idx[i]] - gt[i]).pow(2).sum(-1).sqrt().mean()
        for i in range(gt.shape[0])])


# --------------------------------------------------------------------------
# A3 clause 3 -- THE CONTROL: on a FIXED vocabulary the mutation is INVISIBLE.
# This is refcv3. If it ever fails, the fix moved a working path.
# --------------------------------------------------------------------------
def test_fixed_vocabulary_mutation_is_invisible():
    torch = pytest.importorskip("torch")
    from tanitad.refs import refc
    from tanitad.models import kinematic as kin
    tool = _load_tool()

    dec = _decoder(torch, refc, v0_conditioned=False)
    gt, v0 = _gt_windows(torch, kin)

    # the structural claim, asserted rather than trusted: same storage.
    good = dec.roll_bank(v0[:1], None, 1, torch.float32)
    bad = dec.anchors[None].to(torch.float32)
    assert torch.equal(good, bad), (
        "roll_bank on a FIXED vocabulary must return decoder.anchors expanded, "
        "bit-identical. If this fails the refcv3 control is void and the "
        "correction can no longer be claimed to be a no-op there.")

    right, wrong = _pair(torch, tool, dec, gt, v0)
    assert right == wrong, (
        "on a fixed vocabulary the two bindings MUST agree on every window; "
        f"they differ on {sum(r != w for r, w in zip(right, wrong))}/"
        f"{len(right)}. refcv3's numbers would move, which is forbidden.")


# --------------------------------------------------------------------------
# A3 clauses 1-2 -- THE MUTATION: on a v0-CONDITIONED vocabulary the wrong
# binding must be DETECTABLY wrong, or this gate measures nothing.
# --------------------------------------------------------------------------
def test_v0_conditioned_mutation_is_detected():
    torch = pytest.importorskip("torch")
    from tanitad.refs import refc
    from tanitad.models import kinematic as kin
    tool = _load_tool()

    dec = _decoder(torch, refc, v0_conditioned=True)
    gt, v0 = _gt_windows(torch, kin)

    good = dec.roll_bank(v0[:1], None, 1, torch.float32)
    bad = dec.anchors[None].to(torch.float32)
    assert not torch.equal(good, bad), (
        "a v0-conditioned roll at a non-reference speed must differ from the "
        "reference-speed family, or the fixture is not exercising the defect.")

    right, wrong = _pair(torch, tool, dec, gt, v0)
    n_diff = sum(r != w for r, w in zip(right, wrong))
    assert n_diff > len(right) // 2, (
        f"the mutated binding changed only {n_diff}/{len(right)} indices. The "
        "gate must be able to FAIL loudly on this arm; a near-agreement here "
        "would mean the fixture, not the tool, is being tested.")


# --------------------------------------------------------------------------
# A1 at unit level -- THE CEILING MUST BE A CEILING. With identity refinement
# the oracle's error is the fan minimum BY CONSTRUCTION under the correct
# binding, and strictly worse under the mutated one. This is the mechanism of
# the +0.9179 m sign inversion, reproduced in isolation.
# --------------------------------------------------------------------------
def test_wrong_binding_inverts_the_ceiling():
    """The mutated binding must make the oracle STRICTLY worse, one-signed.

    ⚠️ ASSERTED ON THE TRAINER'S OWN OBJECTIVE, NOT ON ADE. `a_star` is the
    argmin of SUMMED SQUARED ERROR over valid slots; ADE is a mean of L2
    norms. They are different norms and their argmins need NOT coincide --
    MEASURED 47/48 agreement on an out-of-span fixture, 48/48 on this one. So
    "the oracle attains the ADE minimum" is a fixture coincidence, not an
    invariant, and asserting it would be a gate that passes for the wrong
    reason. The invariant is the SSE one, and it is asserted exactly.

    ⇒ This is also an honest caveat on the headline metric: `oracle_sel` is an
    EMPIRICAL ceiling (oracle selection + LEARNED refinement, scored in a norm
    it was not selected in), never a mathematical bound.
    """
    torch = pytest.importorskip("torch")
    from tanitad.refs import refc
    from tanitad.models import kinematic as kin
    tool = _load_tool()

    dec = _decoder(torch, refc, v0_conditioned=True)
    gt, v0 = _gt_windows(torch, kin)
    right, wrong = _pair(torch, tool, dec, gt, v0)

    # the DECODED fan -- what a refinement is read out of under EITHER binding
    fan = torch.cat([dec.roll_bank(v0[i:i + 1], None, 1, torch.float32)
                     for i in range(gt.shape[0])], 0)            # [n, N, S, 2]
    d2 = (fan - gt[:, None]).pow(2).sum(-1)                      # [n, N, S]
    sse, ade = d2.sum(-1), d2.sqrt().mean(-1)                    # [n, N]
    ar = torch.arange(gt.shape[0])
    ri, wi = torch.tensor(right), torch.tensor(wrong)

    # (1) the correct binding attains the trainer's objective, exactly.
    assert torch.allclose(sse[ar, ri], sse.min(1).values, atol=1e-4), (
        "under the CORRECT binding a_star must minimise the summed squared "
        "error over the DECODED fan -- the trainer's own objective. It did "
        "not, so the argmin no longer mirrors compute_losses_v3.")

    # (2) the mutation is strictly worse in that objective, on every window.
    assert (sse[ar, wi] >= sse[ar, ri] - 1e-4).all(), (
        "the mutated binding scored BELOW the fan minimum, which is "
        "impossible; the fixture is inconsistent.")

    # (3) and the damage is one-signed and large in the reported metric.
    worse = int((ade[ar, wi] > ade[ar, ri] + 1e-6).sum())
    better = int((ade[ar, wi] < ade[ar, ri] - 1e-6).sum())
    gap = float((ade[ar, wi] - ade[ar, ri]).mean())
    assert better == 0, (
        f"the mutated binding beat the oracle on {better} window(s). The "
        "defect must never look like an improvement, or a reader could take "
        "the broken ceiling for a real one.")
    assert worse >= gt.shape[0] // 2 and gap > 1.0, (
        f"mutation cost only {gap:.4f} m mean anchor ADE on {worse}/"
        f"{gt.shape[0]} windows. Thresholds sit an order of magnitude below "
        f"the MEASURED effect (44/48 windows, +11.5861 m) so this gate fails "
        f"on the defect and not on fixture noise.")


# --------------------------------------------------------------------------
# A4 -- the shipped accessor must return the DECODED bank and must REFUSE
# rather than fall back. A silent fallback to decoder.anchors is the defect.
# --------------------------------------------------------------------------
def test_decoded_bank_reads_out_and_refuses_absence():
    torch = pytest.importorskip("torch")
    tool = _load_tool()

    bank = torch.arange(2 * 5 * N_STEPS * 2, dtype=torch.float32).reshape(
        2, 5, N_STEPS, 2)
    got = tool._decoded_bank({"anchor_bank": bank}, row=0)
    assert got.shape == (1, 5, N_STEPS, 2)
    assert torch.equal(got, bank[0:1].float()), (
        "_decoded_bank must return row 0 of the forward's own anchor_bank -- "
        "the row every consumer reads anchor_traj back from.")
    assert torch.equal(tool._decoded_bank({"anchor_bank": bank}, row=1),
                       bank[1:2].float())

    with pytest.raises(SystemExit):
        tool._decoded_bank({"traj": torch.zeros(1)})


# --------------------------------------------------------------------------
# The eval argmin and the trainer argmin are ONE convention. This pins the
# arithmetic against a hand-computed expectation including the validity mask,
# so a future edit cannot quietly drop `sv` (a slot past the episode end must
# not vote) without turning this red.
# --------------------------------------------------------------------------
def test_argmin_mirrors_the_trainer_including_the_validity_mask():
    torch = pytest.importorskip("torch")
    tool = _load_tool()

    # 3 anchors, 2 slots. Anchor 0 wins on slot 0 only; anchor 1 wins overall
    # when both slots vote; anchor 2 is never best.
    bank = torch.tensor([[[0.0, 0.0], [9.0, 0.0]],
                         [[1.0, 0.0], [0.0, 0.0]],
                         [[5.0, 0.0], [5.0, 0.0]]])[None]        # [1, 3, 2, 2]
    tgt = torch.zeros(1, 2, 2)

    both = torch.ones(2)
    assert tool._oracle_anchor_index(bank, tgt, both) == 1, (
        "with both slots valid the summed squared error picks anchor 1")

    first_only = torch.tensor([1.0, 0.0])
    assert tool._oracle_anchor_index(bank, tgt, first_only) == 0, (
        "masking slot 1 must change the winner to anchor 0. If this reads 1 "
        "the validity mask is being ignored and windows whose horizon runs "
        "past the episode end are voting on the oracle.")
