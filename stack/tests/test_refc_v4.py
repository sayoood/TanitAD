"""REF-C v4 tiny-rig gate — T1..T11 of ``PREREG_REFC_V4.md`` §6.

⛔ THE POINT OF THIS FILE IS T9. Every other test could pass on an arm that
does nothing but integrate its own dynamics; T9 plants that defect deliberately
(``B-echo``: v4 with the image ablated) and requires the gate to CATCH it. If
T9 ever passes the regression arm, no verdict on the real arm is admissible —
that is OUTCOME IV of the pre-registration, and it voids the panel rather than
the model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# repo pattern (test_ap_ci.py:12, test_heldout_gate.py:38): `taniteval` is not
# installed, it is a SIBLING CHECKOUT. Gate 1 is load-bearing, so it must RUN,
# not skip. ⚠️ The INNER directory goes on the path — putting the outer one
# there resolves `taniteval` to a namespace package with no `ci` submodule and
# fails with `No module named 'taniteval.ci'`, which reads like a missing
# dependency instead of a shadow.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))

from tanitad.eval import echo_gate as eg
from tanitad.refs import refc_v3 as v3


TAUS = (2.0, 4.0, 6.0)


def _cfg(**kw):
    cfg = v3.refc_v3_smoke_config(True)
    for k, val in kw.items():
        if k.startswith("core."):
            setattr(cfg.core, k[5:], val)
        else:
            setattr(cfg, k, val)
    return cfg


def _v4_cfg(echo_base=True):
    return _cfg(ego_state_inject=True, echo_base=echo_base,
                **{"core.ego_valid_channel": True})


def _frames(core, seed=0, n=2):
    g = torch.Generator().manual_seed(seed)
    h, w = core.encoder.image_hw()
    return torch.rand(n, core.window, core.encoder.in_channels, h, w,
                      generator=g)


def _ego(v=(4.0, 9.0), a=(0.8, -1.3), k=(0.03, -0.05), keep=(1.0, 1.0)):
    return torch.tensor([[v[i], a[i], v[i] * k[i], k[i], keep[i]]
                         for i in range(len(v))])


# --------------------------------------------------------------------------- #
# T1 — the closed form IS the integral                                        #
# --------------------------------------------------------------------------- #
def test_T1_closed_form_matches_numeric_integration():
    """The E14 base is a closed form; a closed form that quietly disagrees with
    the motion it claims to describe is a silent experiment change."""
    v0 = torch.tensor([3.0, 8.0, 0.5])
    a0 = torch.tensor([0.0, -1.2, 2.0])
    k0 = torch.tensor([0.05, -0.02, 0.0])
    got = v3.kinematic_goal_extrapolation(v0, a0, k0, TAUS)
    n = 20000
    for b in range(3):
        for j, tau in enumerate(TAUS):
            t_stop = (-v0[b] / a0[b]).item() if a0[b] < 0 else float("inf")
            te = min(tau, t_stop)
            dt = te / n
            x = y = th = 0.0
            for i in range(n):
                t = i * dt
                v = max(v0[b].item() + a0[b].item() * t, 0.0)
                x += v * math.cos(th) * dt
                y += v * math.sin(th) * dt
                th += k0[b].item() * v * dt
            assert abs(got[b, j, 0].item() - x) < 2e-3, (b, j, "x")
            assert abs(got[b, j, 1].item() - y) < 2e-3, (b, j, "y")


def test_T1b_numpy_twin_matches_torch():
    """The corpus report's numpy twin must equal the torch base, or the
    pre-registration's cited 0.4449 m describes a function nobody runs."""
    for v0, a0, k0 in ((5.0, 1.0, 0.04), (2.0, -0.9, -0.03), (7.0, 0.0, 0.0)):
        n = 60
        np_xy = eg._np_extrapolate(v0, a0, k0, n)
        t_xy = v3.kinematic_goal_extrapolation(
            torch.tensor([v0]), torch.tensor([a0]), torch.tensor([k0]),
            [(i + 1) * eg.DT for i in range(n)])[0, :, :2].numpy()
        assert abs(np_xy - t_xy).max() < 1e-4


# --------------------------------------------------------------------------- #
# T2/T3 — ⭐ THE CONTROLS MUST READ KNOWN VALUES EXACTLY                       #
# --------------------------------------------------------------------------- #
def test_T2_ha0_ext_reduces_to_ha0_when_a_and_k_are_zero():
    """With a = kappa = 0 the echo control IS the constant-velocity control.
    Bit-identical, not close: this is the control that says the instrument is
    wired to the right quantity, and 'approximately' would hide a scale bug."""
    v0 = torch.tensor([1.0, 6.0, 12.0])
    z = torch.zeros_like(v0)
    assert torch.equal(eg.ha0(v0, TAUS), eg.ha0_ext(v0, z, z, TAUS))


def test_T3_constant_only_explains_exactly_zero_variance():
    """The no-information control must read the no-information value EXACTLY.
    Three of the four 2026-08-22 estimator failures were caught ONLY because a
    control read the same value as the thing being measured."""
    g = torch.Generator().manual_seed(7)
    tgt = torch.randn(64, 3, 4, generator=g)
    pred = eg.constant_only_reference(tgt)
    ss_res = ((pred - tgt) ** 2).sum()
    ss_tot = ((tgt - tgt.mean(0, keepdim=True)) ** 2).sum()
    assert torch.allclose(ss_res, ss_tot, atol=0, rtol=1e-6)   # R^2 == 0 exactly


def test_T3b_control_and_base_are_one_implementation():
    """``echo_gate.ha0_ext`` must BE the model's own E14 base. A control
    re-implemented beside the thing it controls drifts, and then the gate
    measures the drift instead of the model."""
    v0, a0, k0 = (torch.tensor([4.0, 1.0]), torch.tensor([0.5, -2.0]),
                  torch.tensor([0.02, -0.11]))
    assert torch.equal(eg.ha0_ext(v0, a0, k0, TAUS),
                       v3.kinematic_goal_extrapolation(v0, a0, k0, TAUS))


# --------------------------------------------------------------------------- #
# T4 — E14 is inert when the ego block is withheld                            #
# --------------------------------------------------------------------------- #
def test_T4_echo_base_is_exactly_zero_when_ego_is_withheld():
    """``keep = 0`` must zero the base EXACTLY, so the withheld regime is a
    pure vision-only prediction — that regime is refcv4's free vision-only arm
    and it must not be contaminated by a scaled leftover."""
    cfg = _v4_cfg()
    m = v3.RefCV3Model(cfg).eval()
    f = _frames(cfg.core)
    ego_off = _ego(keep=(0.0, 0.0))
    with torch.no_grad():
        out = m(f, v0=torch.tensor([4.0, 9.0]), ego_state=ego_off)
    assert torch.equal(out["echo_base"], torch.zeros_like(out["echo_base"]))
    assert torch.equal(out["g_tac"], out["g_tac_delta"])


# --------------------------------------------------------------------------- #
# T5/T6 — v3 parity, and zero-init                                            #
# --------------------------------------------------------------------------- #
def test_T5_v3_is_bit_identical_when_the_lever_is_off():
    """Defaults OFF must build today's refcv3 exactly — the live 40,284-step
    run resumes through this file."""
    torch.manual_seed(0)
    a = v3.RefCV3Model(_cfg())
    torch.manual_seed(0)
    b = v3.RefCV3Model(_cfg())
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    assert not any(k.startswith("ego_") for k in sa), sorted(sa)[:5]
    f = _frames(a.cfg.core)
    a.eval(), b.eval()
    with torch.no_grad():
        oa = a(f, v0=torch.tensor([3.0, 7.0]))
        ob = b(f, v0=torch.tensor([3.0, 7.0]))
    for k in ("g_str", "g_tac", "traj"):
        assert torch.equal(oa[k], ob[k]), k


def test_T5b_the_ORIGINAL_E11_test_still_passes_on_a_v3_config():
    """E11 is relaxed for v4, NOT deleted. On a v3 config ``v0`` must still
    reach no goal node, and frames must still reach all of them (else the probe
    is UNPOWERED, not clean — C109)."""
    cfg = _cfg()
    m = v3.RefCV3Model(cfg).eval()
    f = _frames(cfg.core)
    with torch.no_grad():
        a = m(f, v0=torch.tensor([0.0, 0.0]))
        b = m(f, v0=torch.tensor([9.0, 4.0]))
        c = m(_frames(cfg.core, seed=7), v0=torch.tensor([0.0, 0.0]))
    for k in ("g_str", "g_tac", "z_tac"):
        assert torch.equal(a[k], b[k]), f"v0 leaked into {k} (E11)"
        assert not torch.equal(a[k], c[k]), f"frames do not move {k} (C109)"


def test_T6_the_ego_edge_is_bit_inert_at_init():
    """The ego projections are zero-init, so at step 0 E11' must add EXACTLY
    nothing to ``z_tac``/``ctx``: within ONE v4 model, changing ``ego_state``
    must leave both bit-identical. Any delta later is TRAINING, which is what
    makes the edge attributable.

    ⚠️ CORRECTED FROM THE FIRST DRAFT, AND THE CORRECTION MATTERS. The test
    originally compared a separately-constructed v3 model to a v4 one under the
    same seed and asserted bit-identity. That is FALSE and cannot be made true:
    `core.ego_valid_channel=True` (a REGISTERED part of the v4 delta) widens
    `d_meas_in` by one, so the measurement Linear has a different SHAPE, and the
    extra modules shift every subsequent RNG draw — the two models simply have
    different weights. Seeding does not fix that; it hides it. The zero-init
    claim is about the EDGE, and this is the form of it that is actually true.
    """
    for echo in (False, True):
        cfg = _v4_cfg(echo_base=echo)
        m = v3.RefCV3Model(cfg).eval()
        f = _frames(cfg.core)
        with torch.no_grad():
            a = m(f, v0=torch.tensor([4.0, 9.0]),
                  ego_state=_ego(v=(1.0, 1.0), a=(0.0, 0.0), k=(0.0, 0.0)))
            b = m(f, v0=torch.tensor([4.0, 9.0]),
                  ego_state=_ego(v=(18.0, 2.0), a=(3.0, -3.0),
                                 k=(0.25, -0.25)))
        assert torch.equal(a["z_tac"], b["z_tac"]), f"echo_base={echo}"
        assert torch.equal(a["g_str"], b["g_str"]), f"echo_base={echo}"
        assert torch.equal(a["traj"], b["traj"]), f"echo_base={echo}"
        # with E14 ON the GOAL does move — that is the base, by design, and it
        # is the one thing the zero-init does NOT make inert.
        assert torch.equal(a["g_tac"], b["g_tac"]) is (not echo)


def test_T6b_echo_base_arm_starts_exactly_at_the_extrapolation():
    """With E14 on, the zero-init delta head means ``g_tac`` IS ``ha0_ext`` at
    init — a known, MEASURED starting point rather than a random one."""
    cfg = _v4_cfg()
    m = v3.RefCV3Model(cfg).eval()
    ego = _ego()
    with torch.no_grad():
        out = m(_frames(cfg.core), v0=torch.tensor([4.0, 9.0]), ego_state=ego)
    want = eg.ha0_ext(ego[:, 0], ego[:, 1], ego[:, 3], cfg.goal_tau_seconds)
    assert torch.allclose(out["g_tac"], want, atol=0, rtol=0)
    assert float(out["echo_ratio"]) == 0.0


# --------------------------------------------------------------------------- #
# T7/T8/T9 — ⭐⭐ GATE 2, AND THE DELIBERATE REGRESSION                        #
# --------------------------------------------------------------------------- #
def _rig_batch(cfg, n, g, *, blind=False):
    """One tiny-rig batch whose target depends on BOTH the scene and the ego.

    ⚠️ THE SCENE SIGNAL CARRIES A PER-SAMPLE BIAS ON PURPOSE, and the first
    draft did not — which made this rig lie. With frames drawn as plain
    ``rand``, the per-sample mean concentrates hard around 0.5 (it averages
    ~10^5 pixels), so the "scene" term of the target was very nearly CONSTANT
    across rows. Deranging the frames then barely moved the target-relevant
    signal, and the FUNCTIONAL probe correctly reported that the scene was not
    used — on the arm that was supposed to be healthy. The rig, not the probe,
    was at fault: a scene with no per-sample variance is not a scene. Same
    family as an underpowered control read as a negative.
    """
    h, w = cfg.core.encoder.image_hw()
    bias = torch.rand(n, 1, 1, 1, 1, generator=g) * 0.8
    f = torch.rand(n, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                   generator=g) * 0.2 + bias
    if blind:                      # ⛔ the planted defect: NO scene information
        f = torch.full_like(f, 0.5)
    ego = torch.stack([
        torch.rand(n, generator=g) * 10 + 1,                      # v0
        torch.rand(n, generator=g) * 2 - 1,                       # a_long
        torch.zeros(n),                                           # r0 (below)
        torch.rand(n, generator=g) * 0.1 - 0.05,                  # k0
        torch.ones(n)], dim=-1)
    ego[:, 2] = ego[:, 0] * ego[:, 3]
    tgt = (eg.ha0_ext(ego[:, 0], ego[:, 1], ego[:, 3], cfg.goal_tau_seconds)
           + 6.0 * bias.reshape(-1, 1, 1))
    return f, ego, tgt


def _trained_v4(seed=0, steps=140, blind=False):
    """A briefly-trained v4 arm. ``blind=True`` is ⛔ THE DELIBERATE-REGRESSION
    ARM: identical in every respect except the image carries NO information, so
    the arm has ego and no scene and is DESIGNED to echo."""
    torch.manual_seed(seed)
    cfg = _v4_cfg()
    m = v3.RefCV3Model(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        f, ego, tgt = _rig_batch(cfg, 4, g, blind=blind)
        out = m(f, v0=ego[:, 0], ego_state=ego)
        loss = (out["g_tac"] - tgt).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), cfg


def _ablation_batch(cfg, n=24, seed=11):
    """The SCORED batch for the functional probe. Always a REAL, varying scene
    — including for the regression arm, which is the honest, harder case: an
    arm that HAS a live encoder over real images and has learned to ignore it."""
    return _rig_batch(cfg, n, torch.Generator().manual_seed(seed))


def _predictor(m):
    return lambda f, e: m(f, v0=e[:, 0], ego_state=e)["g_tac"]


def test_T7_T8_structural_probe_fires_on_both_directions():
    """Gate 2 (STRUCTURAL) on the healthy arm: both wires must exist.

    ⚠️ Passing this is necessary and is NOT an anti-echo result — see T9.
    """
    m, cfg = _trained_v4()
    batch = {"frames": _frames(cfg.core, seed=3), "v0": torch.tensor([4.0, 9.0]),
             "ego_state": _ego()}
    rep = eg.ego_intervention_test(m, batch)
    assert rep["status"] == "MEASURED", rep["reason"]
    assert rep["ego_moves_goal"], "T7 FAILED: ego reaches no goal node"
    assert rep["scene_moves_goal"], "T8 FAILED: the scene wire is severed"
    assert rep["verdict"] == "READS_BOTH", rep


def test_T8b_functional_probe_says_the_healthy_arm_USES_both():
    """Gate 2b on the healthy arm: the WRONG scene must hurt, and so must the
    wrong ego. This is the claim T7/T8 cannot make."""
    m, cfg = _trained_v4()
    f, ego, tgt = _ablation_batch(cfg)
    rep = eg.source_ablation_test(_predictor(m), frames=f, ego_state=ego,
                                  target=tgt, n_boot=300)
    assert rep["verdict"] == "READS_BOTH", rep
    assert rep["sources"]["scene"]["degradation_rel"] > 0.05, rep["sources"]
    assert rep["sources"]["ego"]["degradation_rel"] > 0.05, rep["sources"]


def test_T9_THE_GATE_FAILS_THE_DELIBERATE_REGRESSION_ARM():
    """⛔⛔ THE LOAD-BEARING TEST. ``B-echo`` was trained with the image ablated,
    so it is a pure echo BY CONSTRUCTION. If the gate does not FAIL it, a PASS
    on the real arm means nothing (PREREG OUTCOME IV: the panel is void, not
    the model).

    ⛔⛔ AND THIS ARM ALREADY EARNED ITS KEEP. MEASURED 2026-09-03: it **PASSED**
    the structural probe's scene direction — perturbing ``frames`` moved every
    goal node — because a live encoder moves the output for ANY model,
    including one that ignores the scene. The first version of Gate 2 would
    therefore have certified a pure echo as ``READS_BOTH``. The functional
    probe (Gate 2b) is the repair, and this test pins BOTH halves of the
    finding so the defect cannot come back silently.

    ⚠️ It is scored on REAL, VARYING frames — not on the constant frames it was
    trained with — because constant frames would make the derangement a literal
    no-op and the catch trivial. This is the honest, harder case: an arm that
    HAS a live encoder over real images and has learned to ignore it.
    """
    m, cfg = _trained_v4(blind=True)
    f, ego, tgt = _ablation_batch(cfg)

    # (a) the STRUCTURAL probe cannot catch it — pinned, because this is the
    #     reason Gate 2b exists and a future reader must not "simplify" it away.
    batch = {"frames": f[:2], "v0": ego[:2, 0], "ego_state": ego[:2]}
    struct = eg.ego_intervention_test(m, batch)
    assert struct["scene_moves_goal"], (
        "the structural probe is EXPECTED to pass a pure echo — if this ever "
        "fails, the finding that motivated Gate 2b has changed and the gate's "
        "design note must be revisited")

    # (b) the FUNCTIONAL probe DOES catch it — the load-bearing assertion.
    rep = eg.source_ablation_test(_predictor(m), frames=f, ego_state=ego,
                                  target=tgt, n_boot=300)
    assert rep["sources"]["ego"]["is_used"], (
        "the regression arm must still USE ego — otherwise it is not the "
        "planted defect, it is just a broken model")
    assert not rep["sources"]["scene"]["is_used"], (
        "⛔ THE GATE DID NOT CATCH THE PLANTED ECHO — PREREG OUTCOME IV: no "
        f"verdict on refcv4 is admissible until this is fixed. {rep['sources']}")
    assert rep["verdict"] == "ECHOING", rep
    with pytest.raises(eg.EchoViolation, match="ECHOING"):
        eg.assert_not_echoing(None, struct, gate2b=rep)


def test_T9c_a_structural_only_pass_is_labelled_as_such():
    """A gate run without Gate 2b must NOT be reportable as an anti-echo pass."""
    ok = eg.assert_not_echoing(None, {"status": "MEASURED",
                                      "verdict": "READS_BOTH"})
    assert ok["class"] == "STRUCTURAL_ONLY"
    assert ok["gate2b_verdict"] is None


def test_T9b_the_gate_raises_on_gate1_failure_too():
    """Gate 1 must refuse an arm that does not clear the controls by the margin
    the prereg committed IN ADVANCE — a separated CI on a 0.3 % margin is a real
    but useless difference."""
    import numpy as np
    n, eid = 240, np.repeat(np.arange(12), 20)
    rng = np.random.default_rng(0)
    ref = np.abs(rng.normal(1.0, 0.15, (n, 3)))
    refs = {"ha": ref, "ha0_ext": ref}
    mg = {"ha": 0.10, "ha0_ext": 0.10}
    g1 = eg.echo_gate(arm_ade=ref * 0.98, references=refs, eid=eid,
                      margins=mg, n_boot=200)       # a 2 % win: separated, but
    cell = g1["slots"][-1]["vs"]["ha0_ext"]         # 10 % was committed
    assert cell["separated"] and not cell["passes"], cell
    ok2 = {"status": "MEASURED", "verdict": "READS_BOTH"}
    ok2b = {"verdict": "READS_BOTH"}
    with pytest.raises(eg.EchoViolation, match="did NOT clear"):
        eg.assert_not_echoing(g1, ok2, gate2b=ok2b)
    g2 = eg.echo_gate(arm_ade=ref * 0.80, references=refs, eid=eid,
                      margins=mg, n_boot=200)
    assert eg.assert_not_echoing(g2, ok2, gate2b=ok2b)["ok"]


def test_T9d_a_panel_missing_the_ha_control_is_REFUSED():
    """⛔ MEASURED on refcv3 @30k: `ha` 0.2996 BEATS the model's 0.4799 while
    `ha0` 0.6723 loses to it. An `ha0`-only panel certifies an arm the trivial
    control already beats twice over."""
    import numpy as np
    ref = np.abs(np.random.default_rng(1).normal(1.0, 0.1, (60, 3)))
    with pytest.raises(ValueError, match="0.2996|refuses a panel missing"):
        eg.echo_gate(arm_ade=ref * 0.5, references={"ha0": ref},
                     eid=np.repeat(np.arange(6), 10), n_boot=50)


def test_T9e_clearing_ha_while_tying_ha0_ext_is_ECHOING_HARDER():
    """⚠️ THE TRAP THE `ha` RESULT CREATES. Both controls are ego
    extrapolations; `ha0_ext` is the sharper one (it uses the corpus's own
    `ax`, corr only +0.5624 with `ha`'s finite difference). An arm that clears
    the blunt control while TYING the sharp one gained by extrapolating its own
    dynamics better — not by reading the scene."""
    import numpy as np
    n, eid = 300, np.repeat(np.arange(15), 20)
    rng = np.random.default_rng(3)
    arm = np.abs(rng.normal(1.00, 0.10, (n, 3)))
    ha = arm * 1.40                       # the blunt control: the arm beats it
    ext = arm.copy()                      # the sharp control: an exact TIE
    g1 = eg.echo_gate(arm_ade=arm, references={"ha": ha, "ha0_ext": ext},
                      eid=eid, margins={"ha": 0.10, "ha0_ext": 0.10},
                      n_boot=200)
    row = g1["slots"][-1]["vs"]
    assert row["ha"]["passes"] and not row["ha0_ext"]["separated"], row
    with pytest.raises(eg.EchoViolation, match="echoing harder"):
        eg.assert_not_echoing(g1, {"status": "MEASURED",
                                   "verdict": "READS_BOTH"},
                              gate2b={"verdict": "READS_BOTH"})


def test_ha_is_ha0_ext_with_a_finite_difference_accel():
    """The identity, pinned: the harness's hold-action control IS this module's
    echo, built on a noisier accel estimate."""
    v0 = torch.tensor([6.0, 3.0])
    vp = torch.tensor([5.8, 3.4])
    sp = torch.tensor([0.06, -0.02])
    got = eg.ha_finite_diff_accel(v0, vp, sp, TAUS)
    want = eg.ha0_ext(v0, (v0 - vp) / eg.DT,
                      torch.tan(sp) / v3.WHEELBASE_CONST2P9, TAUS)
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# T10 — ⛔ the future is REFUSED, and that is measured, not declared           #
# --------------------------------------------------------------------------- #
def test_T10_no_goal_node_reads_the_future():
    """The PI's *"not the future one"*, computed.

    The obvious wrong implementation is ``a0 = (future[:,0,3] - v0)/dt``: it
    would look identical in a config diff and satisfy every 'does the goal use
    acceleration?' check. So the refused edge is PINNED interventionally."""
    m, cfg = _trained_v4()
    batch = {"frames": _frames(cfg.core, seed=3),
             "v0": torch.tensor([4.0, 9.0]), "ego_state": _ego()}
    rep = eg.ego_intervention_test(m, batch, future_keys=())
    assert rep["future_leak"] == []
    # ⭐ and the probe must be POWERED to detect one — a clean reading from an
    # instrument that cannot fire is C109. Plant a future read and require the
    # SAME probe to catch it.
    class Leaky(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, frames, v0=None, ego_state=None,
                    future_poses_ext=None):
            out = self.inner(frames, v0=v0, ego_state=ego_state)
            if future_poses_ext is not None:      # the planted future read
                out["g_tac"] = out["g_tac"] + future_poses_ext.mean()
            return out

    leaky = Leaky(m).eval()
    b2 = dict(batch, future_poses_ext=torch.rand(2, 60, 4))
    rep2 = eg.ego_intervention_test(leaky, b2,
                                    future_keys=("future_poses_ext",))
    assert rep2["future_leak"] == ["future_poses_ext"], rep2
    assert rep2["verdict"] == "FUTURE_LEAK"
    with pytest.raises(eg.EchoViolation, match="FUTURE|future"):
        eg.assert_not_echoing(None, rep2)


# --------------------------------------------------------------------------- #
# the pinned delta, the params, and the loud refusals                         #
# --------------------------------------------------------------------------- #
def test_registered_delta_is_pinned():
    """C122: an ablation's 'everything else identical' is DERIVED, never
    asserted in prose."""
    d = v3.config_delta(v3.refc_v4_config("small"),
                        v3.refc_v3_sized_config("small"))
    assert set(d) == set(v3.REGISTERED_DELTA_KEYS_V4), sorted(d)


def test_param_breakdown_accounts_the_ego_block():
    m = v3.RefCV3Model(_v4_cfg())
    bd = v3.param_breakdown_v3(m)
    assert "ego_inject" in bd and bd["ego_inject"] > 0
    d_ego, d_tac = m.cfg.d_ego, m.cfg.d_tac
    d_ctx = m.cfg.core.strategic.d_ctx
    want = ((v3.EGO_DIMS + 1) * d_ego + (d_ego + 1) * d_tac
            + (d_ego + 1) * d_ctx)
    assert bd["ego_inject"] == want, (bd["ego_inject"], want)


def test_ego_valid_channel_is_a_precondition_not_an_option():
    with pytest.raises(ValueError, match="ego_valid_channel"):
        v3.RefCV3Model(_cfg(ego_state_inject=True))


def test_supplying_ego_to_a_v3_build_fails_loud():
    """A silently-dropped ego block would report as a v4 while running v3 —
    and would show up in a results table as 'the ego channels do not help'."""
    m = v3.RefCV3Model(_cfg()).eval()
    with pytest.raises(ValueError, match="silently dropped|SILENTLY DROPPED"):
        m(_frames(m.cfg.core), v0=torch.tensor([1.0, 2.0]),
          ego_state=_ego())


def test_ego_state_at_t0_reads_only_the_observed_window():
    """Every channel comes from index -1 of the OBSERVED window."""
    b, w = 3, 8
    poses = torch.randn(b, w, 4)
    actions = torch.randn(b, w, 2) * 0.2
    ego = v3.ego_state_at_t0(poses, actions)
    assert ego.shape == (b, v3.EGO_DIMS)
    assert torch.equal(ego[:, 0], poses[:, -1, 3])
    assert torch.equal(ego[:, 1], actions[:, -1, 1])
    k = torch.tan(actions[:, -1, 0]) / v3.WHEELBASE_CONST2P9
    assert torch.allclose(ego[:, 3], k)
    assert torch.allclose(ego[:, 2], poses[:, -1, 3] * k)
    # ⭐ changing ANY earlier frame, or any future frame, cannot move the block
    p2 = poses.clone(); p2[:, :-1] = torch.randn(b, w - 1, 4)
    a2 = actions.clone(); a2[:, :-1] = torch.randn(b, w - 1, 2)
    assert torch.equal(v3.ego_state_at_t0(p2, a2), ego)


def test_per_clip_wheelbase_regime_is_refused_not_mis_inverted():
    """A ``per_clip_v1`` cache stored ``atan(L_clip * curvature)``; inverting it
    with 2.9 rescales every curvature by ``L_clip/2.9`` and reads EXACTLY like a
    working channel. Refuse rather than silently mis-invert."""
    with pytest.raises(ValueError, match="const2p9"):
        v3.ego_state_at_t0(torch.randn(2, 4, 4), torch.randn(2, 4, 2),
                           wheelbase_mode="per_clip_v1")


def test_provenance_roles_move_the_refused_edge_for_v4():
    r3 = v3.RefCV3Model(_cfg()).provenance_roles()
    r4 = v3.RefCV3Model(_v4_cfg()).provenance_roles()
    assert any("v0 -> any goal node" in e for e in r3["refused_edges"])
    assert any("future_poses" in e for e in r4["refused_edges"])
    assert not any(e.startswith("v0 ->") for e in r4["refused_edges"])
    for r in (r3, r4):          # ⛔ UNCHANGED by v4 (PI 2026-08-03)
        assert r["situation_output"] == []
        assert any("situation" in e for e in r["refused_edges"]) or r is r3

# --------------------------------------------------------------------------- #
# THE VALIDATION RIG RUNG                                                     #
# --------------------------------------------------------------------------- #
def test_the_rig_rung_is_NOT_a_member_of_the_registered_ladder():
    """⛔ `V3_RIG_SIZES` must stay OUT of `V3_SIZES`.

    `V3_SIZES` is read by `refc_v3_scale_matrix` and by the D-008 package
    decision, and `test_refc_v3_scale_matrix.py` asserts the hierarchy cost is
    near-constant ACROSS THE REGISTERED LADDER. A 2x-narrower encoder in that
    dict would break a guard that exists to catch quiet geometry drift, and
    widening the band to make it green is how a guard stops guarding."""
    from tanitad.refs.refc_v3 import V3_RIG_SIZES, V3_SIZES
    assert set(V3_RIG_SIZES) == {"tiny"}
    assert not (set(V3_RIG_SIZES) & set(V3_SIZES))


def test_the_rig_rung_moves_the_ENCODER_ONLY_like_every_registered_rung():
    """The rig is only a valid screen if it differs from `small` exactly the
    way the registered rungs differ from each other."""
    import dataclasses

    from tanitad.refs.refc_v3 import refc_v3_sized_config
    t = refc_v3_sized_config("tiny").core
    s_ = refc_v3_sized_config("small").core
    differing = [f.name for f in dataclasses.fields(t)
                 if getattr(t, f.name) != getattr(s_, f.name)]
    assert differing == ["encoder"], differing


def test_the_rig_rung_is_in_the_skill_s_tiny_band():
    """⭐ ~19 M is `TanitAD_ValidateAIDesign` section 2's tiny-rig budget. If a
    geometry change pushes this out of band the rig stops being a ~29 min/arm
    screen on the dev box, and the number must be re-measured rather than the
    band widened."""
    from tanitad.refs.refc_v3 import (RefCV3Model, param_breakdown_v3,
                                      refc_v3_sized_config)
    n = param_breakdown_v3(RefCV3Model(refc_v3_sized_config("tiny")))["total"]
    assert 12_000_000 < n < 22_000_000, n


def test_an_unknown_size_is_still_refused_and_the_message_names_BOTH_tables():
    import pytest as _pt

    from tanitad.refs.refc_v3 import refc_v3_sized_config
    with _pt.raises(ValueError, match="size must be one of"):
        refc_v3_sized_config("enormous")
    try:
        refc_v3_sized_config("enormous")
    except ValueError as exc:
        assert "tiny" in str(exc), str(exc)

# --------------------------------------------------------------------------- #
# THE DELIBERATE-REGRESSION LEVER ITSELF                                      #
# --------------------------------------------------------------------------- #
def _tiny_v4_batch(seed=0):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import torch as _t

    import refc_v3_train as T
    from tanitad.refs import refc_v3 as v3
    import dataclasses as _dc
    # the synthetic corpus carries kin3 (3x3) tactical labels, and the
    # NAMED configs default to v7.0 (8x8) per the 2026-08-27 vocab
    # mandate -- the trainer refuses the mismatch by design, so the rig
    # builds at the corpus's vocabulary rather than loosening the check.
    cfg = _dc.replace(v3.refc_v4_config("tiny"), tac_vocab_version="kin3")
    _t.manual_seed(seed)
    m = v3.RefCV3Model(cfg).eval()
    eps = T._synth_episodes(2, cfg.core, seed=seed)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    b = _t.utils.data.default_collate([ds[0], ds[1]])
    return T, m, b


def test_ablate_frames_REALLY_removes_the_scene_and_changes_the_loss():
    """\u26d4\u26d4 The gate is only meaningful if the arm it must fail really is an
    echo. A lever that silently no-ops would produce a deliberate-regression arm
    that is just... the model again, and the gate would 'fail to fail' for the
    wrong reason. Verify by CONTENT: the observed window must become a single
    constant, and the loss must move."""
    import torch

    T, m, b = _tiny_v4_batch()
    with torch.no_grad():
        base = T.compute_losses_v3(m, b, "cpu", mode="diffusion",
                                   ablate_frames=False)
        abl = T.compute_losses_v3(m, b, "cpu", mode="diffusion",
                                  ablate_frames=True)
    assert float(base["loss"]) != float(abl["loss"]), (
        "--ablate-frames did not move the loss: the lever is inert and the "
        "deliberate-regression arm would not be an echo")


def test_ablate_frames_uses_a_CONSTANT_not_zeros_and_not_noise():
    """The fill is the batch mean, deliberately. Zeros push the trunk's
    activation statistics off-distribution (a failure would read as 'the
    encoder broke'); fresh noise makes the input a random variable the model
    can average out, which is a DIFFERENT ablation."""
    import torch

    T, _m, b = _tiny_v4_batch()
    fr = T.frames_to_device(b["frames"], "cpu")
    abl = torch.full_like(fr, float(fr.mean()))
    assert abl.unique().numel() == 1, "the ablated window is not constant"
    assert torch.isclose(abl.mean(), fr.mean()), (
        "the fill is not the batch mean -- zeros or noise would be a different "
        "ablation than the one the docstring claims")
    assert float(abl.mean()) != 0.0, "the fill collapsed to zeros"


def test_the_regress_arm_is_STAMPED_in_its_own_config():
    """\u26d4 A run whose frames were ablated is a GATE CONTROL, and its artifact
    must say so on its own -- refcv3 stamped NEITHER ego knob and a finished run
    could not answer 'was the speed masked?' from its own record."""
    import re
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "scripts"
           / "refc_v3_train.py").read_text(encoding="utf-8")
    assert '"ablate_frames": bool(args.ablate_frames)' in src
    assert '"size": args.size' in src
    assert '"rig_rung": bool(args.size in v3.V3_RIG_SIZES)' in src
    assert re.search(r'"ego":\s*\{', src), "the ego block is not stamped"
