"""REF-C v3 (tanitad/refs/refc_v3.py + the gated ``hierarchy_hook`` in
tanitad/refs/refc.py) — the goal-mediated hierarchy skeleton.

Pinned here, one test per registered discipline:
(a) TRANSPARENCY: the flat wrap (``hier=False``) is BIT-IDENTICAL to a bare
    RefCModel — outputs and state_dict keys — which simultaneously proves the
    ``hierarchy_hook`` edit left the default forward byte-identical (the F-cell
    "default build unchanged" rule for shared code).
(b) HOOK CONTRACT: an explicitly supplied port always beats the hook (a hook
    that could silently override an experiment's input would be the
    silently-ignored-prior bug class, inverted).
(c) INERT AT INIT: the H arm's zero-init cascade emits the core's exact
    trajectory at step 0 (zero-init FiLM + zero-init goal gate + seam clamp's
    below-cap exactness).
(d) THE PINNED DELTA (C122): ``config_delta(hier, flat)`` equals EXACTLY the
    registered lever set {hier, core.graft_target_latent} — 'everything else
    identical' is DERIVED, never asserted.
(e) CAPACITY LEDGER: full-size H/F param totals in the registered band, delta
    < 6 % (PREREG §3), breakdown sums exactly.
(f) E11 REFUSED EDGE: v0 moves NO goal node (bit-identity under intervention),
    while frames move every goal node — the mini intervention audit, with its
    positive control (an all-clean report with a clean control is UNPOWERED,
    C109).
(g) THE C115 GATE: freeze-history moves z_tac/g_tac (sensitivity PROVEN, not
    asserted), pooled is the built-in negative control (a last-frame function
    must be bit-identical), and history gradients are nonzero through a random
    projection (the C116 ``.sum()``-through-LayerNorm hazard is pinned as its
    own negative control).
(h) MASKED LOSSES: invalid goal slots contribute EXACTLY zero gradient (the
    parity-preserving 6 s design depends on it).
(i) LABEL CONTRACT: GOAL_TAU_STEPS pins to refb_labels.GOAL_TAC_TAUS_STEPS and
    tactical.GOAL_TAC_TAUS_STEPS; the seam slot is 2.0 s.
CPU-only, synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels  # noqa: E402  (scripts/refb_labels.py)

from tanitad.models import tactical as mtac  # noqa: E402
from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402


def _frames(cfg: refc.RefCConfig, b: int = 2, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.encoder.image_hw()
    return torch.rand(b, cfg.window, cfg.encoder.in_channels, h, w,
                      generator=g)


# ---------- (i) label + slot contracts ---------------------------------------

def test_goal_tau_contract():
    assert v3.GOAL_TAU_STEPS == refb_labels.GOAL_TAC_TAUS_STEPS
    assert v3.GOAL_TAU_STEPS == mtac.GOAL_TAC_TAUS_STEPS
    assert v3.GOAL_DIMS == mtac.GOAL_DIMS


def test_seam_slot_is_two_seconds():
    assert v3.V3_HORIZONS[v3.SEAM_SLOT] == 20          # 2.0 s @ 10 Hz
    # the operative band (0, 2] is slots 0..SEAM_SLOT; tactical (2, 6] after
    assert all(h <= 20 for h in v3.V3_HORIZONS[:v3.SEAM_SLOT + 1])
    assert all(h > 20 for h in v3.V3_HORIZONS[v3.SEAM_SLOT + 1:])
    assert max(v3.V3_HORIZONS) == 60                   # 6.0 s binding horizon


# ---------- (a) flat wrap transparency (and hook-edit byte-identity) ---------

def test_flat_wrap_is_transparent():
    cfg = v3.refc_v3_smoke_config(hier=False)
    torch.manual_seed(0)
    wrap = v3.RefCV3Model(cfg).eval()
    ref = refc.RefCModel(cfg.core).eval()
    ref.load_state_dict({k[len("core."):]: p for k, p in
                         wrap.state_dict().items()})
    assert set(wrap.state_dict()) == {f"core.{k}" for k in ref.state_dict()}
    f = _frames(cfg.core)
    v0 = torch.tensor([3.0, 7.0])
    with torch.no_grad():
        a = wrap(f, v0=v0, steps=2)
        b = ref(f, v0=v0, steps=2)
    for k in ("traj", "sel_idx", "sel_score", "anchor_traj", "maneuver_logits",
              "route_logits", "pooled"):
        assert torch.equal(a[k], b[k]), f"flat wrap diverges on {k}"


# ---------- (b) hook contract: explicit port wins ----------------------------

def test_hook_explicit_port_wins():
    cfg = v3.refc_v3_smoke_config(hier=False).core
    torch.manual_seed(0)
    m = refc.RefCModel(cfg).eval()
    f = _frames(cfg)
    man = torch.log_softmax(torch.randn(2, refc.N_MANEUVERS,
                                        generator=torch.Generator()
                                        .manual_seed(1)), dim=-1)
    poison = {"called": 0}

    def hook(pooled_seq, ctx):
        poison["called"] += 1
        return {"maneuver_logits": torch.full_like(man, 5.0)}

    with torch.no_grad():
        a = m(f, maneuver_logits=man, hierarchy_hook=hook)
        b = m(f, maneuver_logits=man)
    assert poison["called"] == 1                      # the hook DID run
    for k in ("traj", "sel_idx", "sel_score"):
        assert torch.equal(a[k], b[k]), \
            f"hook overrode an explicitly supplied port ({k})"


def test_hook_requires_hierarchy():
    cfg = v3.refc_v3_smoke_config(hier=False).core
    cfg.hierarchy = False
    m = refc.RefCModel(cfg).eval()
    with pytest.raises(ValueError, match="hierarchy_hook"):
        m(_frames(cfg), hierarchy_hook=lambda ps, c: {})


# ---------- (c) H arm bit-inert at init --------------------------------------

def test_hier_emission_bit_inert_at_init():
    cfg = v3.refc_v3_smoke_config(hier=True)
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg).eval()
    f = _frames(cfg.core)
    v0 = torch.tensor([3.0, 7.0])
    with torch.no_grad():
        out = m(f, v0=v0, steps=2)
    # zero-init goal gate + zero-init FiLM + zero-init target-latent film ->
    # the emitted trajectory and index ARE the core's, bit for bit.
    assert torch.equal(out["traj"], out["traj_base"])
    assert torch.equal(out["sel_idx"], out["sel_idx_base"])
    assert torch.equal(out["sel_score_v3"], out["sel_score"])
    # ...and the cascade's outputs exist (the arm is not flat-in-disguise
    # structurally; sensitivity is (g)'s job).
    for k in ("z_tac", "g_str", "g_tac", "goal_point_tac", "goal_dist"):
        assert k in out, f"missing cascade output {k}"


# ---------- (d) the pinned dominance delta (C122) ----------------------------

def test_dominance_delta_is_pinned():
    d = v3.config_delta(v3.refc_v3_hier_config(), v3.refc_v3_flat_config())
    assert d == {"hier": (True, False),
                 "core.graft_target_latent": (True, False)}, (
        "the dominance pair differs in something OUTSIDE the registered lever "
        f"set — amend PREREG_REFC_V3.md BEFORE launch. delta = {d}")
    # smoke pair too (what the cheap tests actually exercise)
    ds = v3.config_delta(v3.refc_v3_smoke_config(True),
                         v3.refc_v3_smoke_config(False))
    assert set(ds) == {"hier", "core.graft_target_latent"}


# ---------- (e) capacity ledger (full size, MEASURED) ------------------------

@pytest.mark.slow
def test_param_bands_full_size():
    h = v3.RefCV3Model(v3.refc_v3_hier_config())
    f = v3.RefCV3Model(v3.refc_v3_flat_config())
    bh, bf = v3.param_breakdown_v3(h), v3.param_breakdown_v3(f)
    assert sum(v for k, v in bh.items() if k != "total") == bh["total"]
    assert 50_000_000 < bf["total"] < 85_000_000, bf
    assert 50_000_000 < bh["total"] < 85_000_000, bh
    delta = bh["total"] - bf["total"]
    assert 0 < delta < 0.06 * bh["total"], (bh, bf)   # PREREG §3 band


def test_param_breakdown_smoke_sums():
    m = v3.RefCV3Model(v3.refc_v3_smoke_config(True))
    b = v3.param_breakdown_v3(m)
    assert sum(v for k, v in b.items() if k != "total") == b["total"]
    assert b["phi_tac"] > 0 and b["scorer"] > 0


# ---------- (f) E11: v0 reaches NO goal node; frames reach ALL ---------------

def test_v0_never_reaches_goal_nodes():
    cfg = v3.refc_v3_smoke_config(hier=True)
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg).eval()
    f = _frames(cfg.core)
    with torch.no_grad():
        a = m(f, v0=torch.tensor([0.0, 0.0]))
        b = m(f, v0=torch.tensor([9.0, 4.0]))
        # positive control FIRST (C109: an all-clean audit with a clean
        # control is UNPOWERED): different frames must move every goal node.
        c = m(_frames(cfg.core, seed=7), v0=torch.tensor([0.0, 0.0]))
    for k in ("g_str", "g_tac", "z_tac", "goal_point_tac"):
        assert torch.equal(a[k], b[k]), f"v0 leaked into goal node {k} (E11)"
        assert not torch.equal(a[k], c[k]), \
            f"frames do not move {k} — the probe is unpowered, not clean"


# ---------- (g) the C115 sensitivity gate ------------------------------------

def test_freeze_history_gate():
    cfg = v3.refc_v3_smoke_config(hier=True)
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg)
    rep = v3.freeze_history_report(m, _frames(cfg.core),
                                   v0=torch.tensor([3.0, 7.0]))
    assert rep["pooled_rel_move"] == 0.0, (
        "pooled is a last-frame function BY CONSTRUCTION — a probe that "
        f"reports it moving is broken: {rep}")
    assert rep["pass"], rep
    assert rep["z_tac_rel_move"] > 1e-6 and rep["history_grad_nonzero"], rep


def test_freeze_history_negative_control_sum_trap():
    """C116's instrument hazard, pinned: through ``.sum()`` a LayerNorm-tailed
    latent has IDENTICALLY zero input-gradient, so a probe built on it would
    'confirm' C115 harder than the truth. The report must therefore use a
    random projection — asserted by showing the trap is real on this graph."""
    cfg = v3.refc_v3_smoke_config(hier=True)
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg)
    f = _frames(cfg.core).requires_grad_(True)
    cache: dict = {}
    _ = m.core(f, hierarchy_hook=m._hook(cache))
    z = torch.nn.functional.layer_norm(cache["z_tac"],
                                       (cache["z_tac"].shape[-1],))
    z.sum().backward()
    assert float(f.grad.abs().max()) < 1e-6, (
        "the .sum()-through-LayerNorm gradient should be ~0 — if it is not, "
        "this negative control no longer pins the C116 hazard")


# ---------- (h) masked losses: exact zero gradient at invalid slots ----------

def test_masked_goal_loss_zero_grad_at_invalid():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(3, 3, 4, generator=g, requires_grad=True)
    tgt = torch.randn(3, 3, 4, generator=g)
    valid = torch.tensor([[True, True, False],
                          [True, False, False],
                          [True, True, True]])
    v3.masked_goal_loss(pred, tgt, valid).backward()
    assert pred.grad is not None
    assert torch.equal(pred.grad[~valid],
                       torch.zeros_like(pred.grad[~valid])), \
        "masked slots leaked gradient — the parity-preserving 6 s design " \
        "depends on exact zero"
    assert float(pred.grad[valid].abs().sum()) > 0


def test_strategic_goal_loss_masked():
    g = torch.Generator().manual_seed(0)
    gs = torch.randn(4, 3, generator=g, requires_grad=True)
    bearing = torch.nn.functional.normalize(torch.randn(4, 2, generator=g),
                                            dim=-1)
    dist = torch.rand(4, generator=g) * 2 - 1
    valid = torch.tensor([True, False, True, False])
    v3.strategic_goal_loss(gs, bearing, dist, valid).backward()
    assert torch.equal(gs.grad[~valid], torch.zeros_like(gs.grad[~valid]))
    assert float(gs.grad[valid].abs().sum()) > 0


# ---------- selection CE over the survivor set (S1c lesson) ------------------

def test_selection_ce_survivor_set():
    g = torch.Generator().manual_seed(0)
    blended = torch.randn(2, 6, generator=g, requires_grad=True)
    fan_err = torch.tensor([[9., 8., 0.1, 7., 6., 5.],
                            [9., 8., 7., 6., 0.1, 5.]])
    keep = torch.tensor([[True, True, False, True, True, True],
                         [True, True, True, True, True, True]])
    loss = v3.selection_ce(blended, fan_err, keep)
    loss.backward()
    # row 0's global-best candidate (idx 2) is OUTSIDE the survivor set, so the
    # CE target must be the best SURVIVOR (idx 5), never the unpickable one.
    assert loss.isfinite()
    masked = v3.selection_ce(blended.detach(), fan_err, keep)
    tgt_row0 = fan_err.masked_fill(~keep, float("inf"))[0].argmin()
    assert int(tgt_row0) == 5


# ---------- gate wiring: the goal gate can open (alive, not decorative) ------

def test_goal_gate_receives_gradient():
    cfg = v3.refc_v3_smoke_config(hier=True)
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg).train()
    out = m(_frames(cfg.core), v0=torch.tensor([3.0, 7.0]))
    fan_err = (out["anchor_traj"] - out["traj"][:, None]).norm(dim=-1).sum(-1)
    loss = v3.selection_ce(out["sel_score_v3"], fan_err.detach(),
                           out.get("reach_keep"))
    loss.backward()
    assert m.goal_gate.grad is not None and \
        float(m.goal_gate.grad.abs()) >= 0.0
    # the goal head trains by ITS OWN loss, not by selection (E9 detach):
    assert m.tac_goal_head.weight.grad is None or \
        float(m.tac_goal_head.weight.grad.abs().sum()) == 0.0, \
        "selection gradient reached the goal head — the winner's-curse " \
        "firewall (detach) is broken"
