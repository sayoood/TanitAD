"""E13 (nav to all layers) + the Caveat-A/B work — PI directives 2026-09-02.

⛔ THE METHODOLOGICAL TRAP THIS FILE ENCODES, because it caught me THREE times
in one session: REF-C v3 has THREE zero-init gates in series on the hierarchy
path — v3's `gstr_film`, and the core's `tgt_proj`/`tgt_film`. At step 0 every
one of them multiplies the downstream gradient by zero. So *any* probe of the
hierarchy run at initialisation reads "no gradient / no effect" REGARDLESS of
whether the wiring is right, and a 14-step glance under a 2000-step LR warmup
(lr 5e-8 at step 1, MEASURED) reads the same. ⇒ every test below that asks
"does this path carry signal" MUST first open the gates. Testing at init and
concluding "broken" is not a measurement — it is the zero-init stall reported
as a defect.
"""
import sys
from pathlib import Path

import pytest
import torch

from tanitad.refs import refc_v3 as v3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_refc_v3 import _frames                     # noqa: E402  (shared rig)


def _model(**kw):
    c = v3.refc_v3_smoke_config(hier=True)
    for k, val in kw.items():
        setattr(c, k, val)
    torch.manual_seed(0)
    return v3.RefCV3Model(c), c


def _open_gates(m):
    """Simulate ANY training having happened on the zero-init gates."""
    with torch.no_grad():
        m.gstr_film.weight.normal_(0, 0.1)
        m.gstr_film.bias.normal_(0, 0.1)
        for n, p in m.core.named_parameters():
            if "tgt_film" in n or "tgt_proj" in n:
                p.normal_(0, 0.1)


def _io(c, nav):
    f = _frames(c.core)
    b = f.shape[0]
    n = None if nav is None else torch.full((b,), nav, dtype=torch.long)
    return f, n, torch.zeros(b)


# ------------------------------------------------- E13: nav to all layers --
def test_nav_reaches_ALL_THREE_layers():
    """⭐ PI BINDING: "No training without feeding nav command to all layers."
    Before E13, nav_cmd went to `self.core` and nowhere else. Gates opened
    first, then nav is varied and every level must move."""
    m, c = _model()
    _open_gates(m)
    with torch.no_grad():
        m.nav_to_tac.weight.normal_(0, 0.3)
        m.nav_to_str.weight.normal_(0, 0.3)
    m.eval()
    f, na, v0 = _io(c, 0)
    _, nb, _ = _io(c, 2)
    with torch.no_grad():
        a, b_ = m(f, na, v0, steps=2), m(f, nb, v0, steps=2)
    assert not torch.equal(a["z_tac"], b_["z_tac"]), "TACTICAL is nav-blind"
    assert not torch.equal(a["g_str"], b_["g_str"]), "STRATEGIC is nav-blind"
    assert not torch.equal(a["traj"], b_["traj"]), "OPERATIVE is nav-blind"


def test_nav_injection_is_ZERO_INIT_so_the_arm_starts_comparable():
    """A conditioning edge that shifts behaviour at step 0 makes its own
    result unattributable — the delta must come from training."""
    m, c = _model()
    m.eval()
    f, na, v0 = _io(c, 0)
    _, nb, _ = _io(c, 2)
    with torch.no_grad():
        a, b_ = m(f, na, v0, steps=2), m(f, nb, v0, steps=2)
    assert torch.equal(a["z_tac"], b_["z_tac"])
    assert torch.equal(a["g_str"], b_["g_str"])


def test_nav_injection_is_reported_so_it_cannot_no_op_silently():
    """⚠️ `nav_cmd=None` is the eval default and ~75-79 % of windows are
    `follow`. An edge that quietly receives nothing is the advertised-but-inert
    defect, so the model REPORTS whether nav actually entered."""
    m, c = _model()
    m.eval()
    f, na, v0 = _io(c, 0)
    f2, none_nav, v02 = _io(c, None)
    with torch.no_grad():
        assert m(f, na, v0, steps=2)["nav_injected"] is True
        assert m(f2, none_nav, v02, steps=2)["nav_injected"] is False


def test_nav_does_NOT_relax_the_E11_refusal_of_v0():
    """E11 refuses ego state into every goal node. nav is a ROUTE command, not
    ego state — admitting it must not smuggle v0 in."""
    m, _ = _model()
    roles = m.provenance_roles()
    assert any("v0" in r for r in roles["refused_edges"])
    assert m.cfg.nav_inject is True


def test_nav_modules_exist_only_when_enabled():
    m_on, _ = _model(nav_inject=True)
    m_off, _ = _model(nav_inject=False)
    assert m_on.nav_inj is not None and m_off.nav_inj is None
    assert "nav_inj.weight" not in m_off.state_dict()


# ----------------------------------------- CAVEAT A: conditioned vs trained --
@pytest.mark.parametrize("uplink,expect_grad", [(True, True), (False, False)])
def test_CAVEAT_A_the_uplink_lever_actually_moves_gradient(uplink, expect_grad):
    """⭐ The hierarchy was CONDITIONED but never OPTIMISED: E4/E7 detached, so
    the strategic goal head never learned whether its goal HELPED. This is the
    registered D-4 lever. ⛔ Gates MUST be opened first — at init both settings
    read 0.0 and the lever would look broken (see the module docstring)."""
    m, c = _model(uplink_grad=uplink)
    _open_gates(m)
    m.train()
    f, na, v0 = _io(c, 0)
    m(f, na, v0, steps=2)["anchor_traj"].sum().backward()
    got = (m.str_goal_head.weight.grad is not None
           and float(m.str_goal_head.weight.grad.abs().sum()) > 0)
    assert got is expect_grad
    phi = next(m.phi_tac.parameters())
    got_phi = phi.grad is not None and float(phi.grad.abs().sum()) > 0
    assert got_phi is expect_grad


def test_CAVEAT_A_both_settings_read_zero_AT_INIT_and_that_is_not_a_defect():
    """The pin for the trap itself: a probe at step 0 cannot distinguish the
    two settings, so no future reader may conclude anything from one."""
    for up in (True, False):
        m, c = _model(uplink_grad=up)
        m.train()
        f, na, v0 = _io(c, 0)
        m(f, na, v0, steps=2)["anchor_traj"].sum().backward()
        g = m.str_goal_head.weight.grad
        assert g is None or float(g.abs().sum()) == 0.0


def test_CAVEAT_A_does_NOT_open_the_E9_winners_curse_firewall():
    """⛔ Two detaches, two reasons. E9's goal detach stops SELECTION training
    the goal head toward the fan — the winner's curse SEL-1 was refused for.
    The lever must not touch it."""
    import inspect
    src = inspect.getsource(v3.RefCV3Model.forward)
    assert ".detach()" in src, "the E9 goal detach vanished"
    head, _, tail = src.partition("g_tac")
    assert "uplink_grad" not in tail, \
        "the E9 goal detach must not be made conditional on the lever"


# --------------------------------------------- CAVEAT B: the goal gate ------
def test_CAVEAT_B_the_gate_is_observable_not_merely_asserted():
    """The gate is zero-init and must LEARN to open; if it never does, E9
    contributed nothing. It is NOT structurally stuck — d(graft)/d(gate) =
    score != 0 — so the honest instrument is to emit the gate AND the score
    scale it multiplies, and read the answer at 30k."""
    m, c = _model()
    m.eval()
    f, na, v0 = _io(c, 0)
    with torch.no_grad():
        out = m(f, na, v0, steps=2)
    assert float(out["goal_gate_value"]) == 0.0            # zero-init
    assert float(out["goal_score_absmean"]) > 0.0          # real signal to open on


def test_CAVEAT_B_the_gate_can_receive_gradient():
    """The claim "it can open" is asserted by measurement, not by argument."""
    m, c = _model()
    m.train()
    f, na, v0 = _io(c, 0)
    out = m(f, na, v0, steps=2)
    out["sel_score_v3"].sum().backward()
    assert m.goal_gate.grad is not None
    assert float(m.goal_gate.grad.abs()) > 0.0
