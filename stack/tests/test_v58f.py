"""v5.8f composition (``tanitad.models.v58f``) CPU smoke — tiny synthetic
modules, no checkpoint, no corpus, no GPU. The full path (real v5f trunk + W4
emission + W4b rescorer + v2 corpora) is pod-side (``eval_v58f.py``) and NOT
exercised here.

Mocking approach follows test_w4b.py: shape-faithful synthetic stand-ins and
hand-derived expected values. The mock head pushes its per-candidate query
through ``decoder.offset_head`` so the REAL ``OffsetFeatureTap`` seam is
exercised; the mock emission emits FIXED (a, kappa) rows integrated by the
REAL ``unicycle_rollout``, so fan geometry and controls agree — the property
the from-controls accel telemetry relies on.

Pinned (the task's four groups):
  * composition forwards: all three select_rules produce a valid ``sel_idx``
    and ``traj == fan[ar, sel_idx]``, with the right shapes end to end;
  * the kincost rule's ordering matches a hand-computed case (including that
    the shortlist PRUNES a globally-cheaper candidate the rescorer ranked
    low), and ``kinematic_cost`` itself matches hand arithmetic;
  * telemetry accel MAE matches a hand computation FROM THE CONTROLS;
  * loader arg validation: w4b_ckpt REQUIRED for rescorer rules (and refused
    for frozen-argmax), checked BEFORE any file I/O; plus the gate-decided
    default (``select_rule_from_gate``) on both pre-registered branches.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tanitad.models.v58f import (DT, SELECT_RULES, TOP_PRUNE_K, V58F,  # noqa: E402
                                 accel_mae_from_controls, kinematic_cost,
                                 load_v58f, select_candidate,
                                 select_rule_from_gate)
from train_v58f_unicycle_head import unicycle_rollout  # noqa: E402

B, N, K, F = 2, 10, 6, 12


# ---------------------------------------------------------------------------
# tiny synthetic modules (shape-faithful; test_w4b.py's approach)
# ---------------------------------------------------------------------------
class _World(nn.Module):
    def encode_window(self, frames):
        return frames                       # states pass-through; head ignores


class _Dec(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.offset_head = nn.Linear(f, 2)  # the REAL tap hooks onto this


class _Head(nn.Module):
    """Stand-in for FlagshipV4Head: emits anchor_traj/sel_idx/refined_logits
    and — crucially — pushes the per-candidate query q through
    decoder.offset_head so the OffsetFeatureTap captures it (the W4 seam)."""

    def __init__(self, n=N, k=K, f=F, frozen_pick=3):
        super().__init__()
        self.decoder = _Dec(f)
        self.n, self.k, self.f, self.frozen_pick = n, k, f, frozen_pick
        self.register_buffer("q_base",
                             torch.linspace(-1.0, 1.0, n * f).reshape(n, f))
        self.register_buffer("ref_logits", torch.linspace(0.0, 1.0, n))
        self.cfg = SimpleNamespace(horizons=tuple(range(1, k + 1)),
                                   cond_imagination=False, cond_vtarget=False,
                                   cond_route=False)

    def forward(self, st, v0, lambda_plan=1.0, **kw):
        b = v0.shape[0]
        q = self.q_base[None].expand(b, self.n, self.f)
        _ = self.decoder.offset_head(q)     # <- tap capture happens here
        fan = torch.zeros(b, self.n, self.k, 2)
        return {"anchor_traj": fan,
                "sel_idx": torch.full((b,), self.frozen_pick,
                                      dtype=torch.long),
                "refined_logits":
                    self.ref_logits[None].expand(b, self.n).clone(),
                "traj": fan[:, self.frozen_pick]}


class _Emission(nn.Module):
    """Fixed per-candidate (a, kappa) rows; waypoints via the REAL
    unicycle_rollout so controls and fan geometry agree."""

    def __init__(self, a_rows, kap_rows):               # [N, K] each
        super().__init__()
        self.register_buffer("a_rows", a_rows.float())
        self.register_buffer("kap_rows", kap_rows.float())

    def forward(self, feat, v0):
        b = feat.shape[0]
        a = self.a_rows[None].expand(b, -1, -1).contiguous()
        kap = self.kap_rows[None].expand(b, -1, -1).contiguous()
        wp, _ = unicycle_rollout(a, kap, v0)
        return a, kap, wp


class _Rescorer(nn.Module):
    def __init__(self, row):                            # [N] fixed logits
        super().__init__()
        self.register_buffer("row", row.float())
        self.variant = "mock"

    def forward(self, q, a_ctl=None, kappa=None):
        return self.row[None].expand(q.shape[0], -1).clone()


def _mk(select_rule, *, rescorer=None, a_rows=None, kap_rows=None,
        frozen_pick=3):
    a_rows = torch.zeros(N, K) if a_rows is None else a_rows
    kap_rows = torch.zeros(N, K) if kap_rows is None else kap_rows
    return V58F(_World(), nn.Identity(), _Head(frozen_pick=frozen_pick),
                _Emission(a_rows, kap_rows), rescorer,
                select_rule=select_rule, amp_on=False)


def _frames():
    return torch.zeros(B, 3, F)


def _v0():
    return torch.full((B,), 5.0)


# ---------------------------------------------------------------------------
# composition forwards: all three rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", SELECT_RULES)
def test_all_rules_produce_valid_selection(rule):
    resc = (_Rescorer(torch.linspace(0.0, 1.0, N))
            if rule.startswith("rescorer") else None)
    m = _mk(rule, rescorer=resc, a_rows=torch.rand(N, K) * 0.3)
    r = m.plan(_frames(), _v0())
    assert r["fan"].shape == (B, N, K, 2)
    assert r["controls"]["a"].shape == (B, N, K)
    assert r["controls"]["kappa"].shape == (B, N, K)
    sel = r["sel_idx"]
    assert sel.shape == (B,)
    assert int(sel.min()) >= 0 and int(sel.max()) < N
    ar = torch.arange(B)
    assert torch.equal(r["traj"], r["fan"][ar, sel])
    assert r["scores"].shape == (B, N)
    assert r["telemetry"]["select_rule"] == rule
    assert r["telemetry"]["n_candidates"] == N
    # no target supplied -> the from-controls accel MAE is declared absent,
    # never silently zero
    assert r["telemetry"]["accel_mae_selected_from_controls_ms2"] is None
    m.close()


def test_rescorer_argmax_picks_top_logit():
    row = torch.zeros(N)
    row[7] = 5.0
    m = _mk("rescorer-argmax", rescorer=_Rescorer(row))
    r = m.plan(_frames(), _v0())
    assert (r["sel_idx"] == 7).all()
    assert r["telemetry"]["scores_source"].startswith("w4b_rescorer")
    m.close()


def test_frozen_argmax_deploys_heads_own_pick_not_a_recomputed_argmax():
    """The frozen rule trusts out['sel_idx'] (the head's full select(), incl.
    vt-keep logic) — the mock decouples it from argmax(refined_logits) on
    purpose: refined argmax is N-1 while sel_idx is 3."""
    m = _mk("frozen-argmax", frozen_pick=3)
    r = m.plan(_frames(), _v0())
    assert (r["sel_idx"] == 3).all()
    assert int(r["scores"].argmax(dim=1)[0]) == N - 1        # decoupled
    assert r["telemetry"]["scores_source"] == "frozen_refined_logits"
    assert r["telemetry"]["sel_matches_frozen_frac"] == pytest.approx(1.0)
    m.close()


# ---------------------------------------------------------------------------
# kinematic cost: hand arithmetic + the top-8 pruning semantics
# ---------------------------------------------------------------------------
def test_kinematic_cost_hand_case():
    # mean|a| = (1+2+4)/3 = 7/3; jerk = (1, 2)/0.1 = (10, 20) -> mean 15
    a = torch.tensor([[1.0, 2.0, 4.0]])
    assert torch.allclose(kinematic_cost(a, dt=0.1),
                          torch.tensor([7.0 / 3.0 + 0.5 * 15.0]))
    # K == 1: no jerk term
    assert torch.allclose(kinematic_cost(torch.tensor([[2.0]])),
                          torch.tensor([2.0]))
    with pytest.raises(ValueError):
        kinematic_cost(torch.zeros(3, 0))


def test_kincost_jerk_separates_smooth_from_jittery():
    """Same mean|a|, different jerk -> the cost orders them. (The W1
    refutation was about WAYPOINT-space cost on the old infeasible fan; on
    controls the ordering is exactly the quantity the cost claims.)"""
    smooth = torch.ones(1, K)
    jitter = torch.ones(1, K)
    jitter[0, 1::2] = -1.0
    assert float(kinematic_cost(smooth)) < float(kinematic_cost(jitter))


def test_top8_kincost_rule_hand_case():
    """Candidate 8 is globally cheapest but ranked outside the rescorer's
    top-8 -> PRUNED; the pick is the cheapest INSIDE the shortlist (5)."""
    scores = torch.arange(N, 0, -1).float()      # top-8 = indices 0..7
    a_rows = torch.ones(N, K)                    # const rows: cost = mean|a|
    a_rows[5] = 0.1                              # cheapest inside shortlist
    a_rows[8] = 0.0                              # global cheapest, pruned out
    m = _mk("rescorer-top8-kincost", rescorer=_Rescorer(scores),
            a_rows=a_rows)
    r = m.plan(_frames(), _v0())
    assert (r["sel_idx"] == 5).all()
    assert r["shortlist"].shape == (B, TOP_PRUNE_K)
    assert 8 not in r["shortlist"][0].tolist()
    # cost ordering matches hand arithmetic (const rows -> jerk 0)
    cost = r["kincost"][0]
    assert cost[8] < cost[5] < cost[0]
    assert float(cost[5]) == pytest.approx(0.1, abs=1e-6)
    assert r["telemetry"]["kincost_selected_mean"] == pytest.approx(0.1,
                                                                    abs=1e-6)
    assert r["telemetry"]["shortlist_k"] == TOP_PRUNE_K
    m.close()


def test_select_candidate_validation():
    with pytest.raises(ValueError):
        select_candidate("bogus", frozen_sel_idx=torch.zeros(B,
                                                             dtype=torch.long))
    with pytest.raises(ValueError):
        select_candidate("frozen-argmax", frozen_sel_idx=None)
    with pytest.raises(ValueError):
        select_candidate("rescorer-argmax", scores=None)
    with pytest.raises(ValueError):
        select_candidate("rescorer-top8-kincost", scores=torch.zeros(B, N),
                         a_ctl=None)


def test_top8_clips_to_n_when_fan_is_small():
    sel, aux = select_candidate(
        "rescorer-top8-kincost", scores=torch.zeros(B, 4),
        a_ctl=torch.ones(B, 4, K) * torch.tensor([3.0, 1.0, 2.0, 4.0])[None, :,
                                                                       None])
    assert aux["shortlist"].shape == (B, 4)      # k_prune clipped to N=4
    assert (sel == 1).all()                      # global min cost reachable


# ---------------------------------------------------------------------------
# telemetry accel MAE — hand-computed FROM THE CONTROLS
# ---------------------------------------------------------------------------
def test_telemetry_accel_mae_matches_hand_computed_from_controls():
    a_rows = torch.zeros(N, K)
    a_rows[0] = 1.0                              # the selected candidate
    a_rows[1] = 0.5                              # the target's generator
    logits = torch.zeros(N)
    logits[0] = 5.0
    m = _mk("rescorer-argmax", rescorer=_Rescorer(logits), a_rows=a_rows)
    v0 = _v0()
    # target = candidate 1's own unicycle rollout (v0=5 keeps v>0: accel
    # profile is exactly 0.5 on every finite-difference slot)
    tgt, _ = unicycle_rollout(a_rows[None, 1:2].expand(B, 1, K).contiguous(),
                              torch.zeros(B, 1, K), v0)
    tgt = tgt[:, 0]
    r = m.plan(_frames(), v0, tgt=tgt)
    assert (r["sel_idx"] == 0).all()
    got = r["telemetry"]["accel_mae_selected_from_controls_ms2"]
    # hand value: selected commanded accel 1.0 const vs GT-derived 0.5 const
    assert got == pytest.approx(0.5, abs=1e-5)
    # and against a fully independent hand derivation of the GT accel profile
    z = torch.zeros(B, 1, 2)
    sp = torch.diff(torch.cat([z, tgt], dim=1), dim=1).norm(dim=-1) / DT
    acc_gt = torch.diff(sp, dim=-1) / DT
    hand = float((torch.ones(B, K - 1) - acc_gt).abs().mean())
    assert got == pytest.approx(hand, abs=1e-6)
    # the pure helper agrees too (same arithmetic, callable standalone)
    assert accel_mae_from_controls(torch.ones(B, K), tgt) == \
        pytest.approx(hand, abs=1e-6)
    m.close()


def test_accel_mae_from_controls_validation():
    tgt = torch.zeros(B, K, 2)
    with pytest.raises(ValueError):
        accel_mae_from_controls(torch.zeros(B, K + 1), tgt)   # K mismatch
    with pytest.raises(ValueError):
        accel_mae_from_controls(torch.zeros(B, K), tgt[..., :1])  # not xy
    with pytest.raises(ValueError):
        accel_mae_from_controls(torch.zeros(B, 1), torch.zeros(B, 1, 2))


def test_plan_without_target_keeps_control_magnitudes():
    m = _mk("frozen-argmax")                     # zero fan, zero controls
    r = m.plan(_frames(), _v0())
    t = r["telemetry"]
    assert t["accel_mae_selected_from_controls_ms2"] is None
    assert t["accel_mean_abs_selected_ms2"] == pytest.approx(0.0)
    assert t["jerk_mean_abs_selected_ms3"] == pytest.approx(0.0)
    assert t["kincost_selected_mean"] == pytest.approx(0.0)
    m.close()


# ---------------------------------------------------------------------------
# plan_batch: the IMPORTED frozen_forward path agrees with the mirror
# ---------------------------------------------------------------------------
def test_plan_batch_imports_frozen_forward_and_matches_plan():
    torch.manual_seed(0)
    m = _mk("rescorer-argmax", rescorer=_Rescorer(torch.linspace(0., 1., N)),
            a_rows=torch.rand(N, K) * 0.3)
    v0 = _v0()
    fr = _frames()
    pose_last = torch.stack([torch.zeros(B), torch.zeros(B),
                             torch.zeros(B), v0], dim=1)          # [B, 4]
    batch = {"frames": fr, "pose_last": pose_last,
             "future_poses": torch.zeros(B, K, 4)}
    rb = m.plan_batch(batch, "cpu")
    assert "tgt" in rb and rb["tgt"].shape == (B, K, 2)
    rp = m.plan(fr, v0, tgt=rb["tgt"])
    assert torch.equal(rb["sel_idx"], rp["sel_idx"])
    assert torch.allclose(rb["traj"], rp["traj"])
    assert rb["telemetry"]["accel_mae_selected_from_controls_ms2"] == \
        pytest.approx(
            rp["telemetry"]["accel_mae_selected_from_controls_ms2"],
            abs=1e-6)
    m.close()


# ---------------------------------------------------------------------------
# constructor + loader validation (w4b_ckpt required for rescorer rules)
# ---------------------------------------------------------------------------
def test_constructor_validation():
    with pytest.raises(ValueError, match="select_rule"):
        _mk("bogus-rule")
    with pytest.raises(ValueError, match="rescorer"):
        _mk("rescorer-argmax")                   # no rescorer supplied
    with pytest.raises(ValueError, match="rescorer"):
        _mk("rescorer-top8-kincost")
    _mk("frozen-argmax").close()                 # fine without a rescorer
    with pytest.raises(ValueError, match="w4_cond"):
        V58F(_World(), nn.Identity(), _Head(),
             _Emission(torch.zeros(N, K), torch.zeros(N, K)), None,
             select_rule="frozen-argmax", w4_cond="bogus")


def test_noncontiguous_horizons_refused():
    h = _Head()
    h.cfg = SimpleNamespace(horizons=(5, 10, 15, 20), cond_imagination=False)
    with pytest.raises(ValueError, match="contiguous"):
        V58F(_World(), nn.Identity(), h,
             _Emission(torch.zeros(N, K), torch.zeros(N, K)), None,
             select_rule="frozen-argmax")


def test_load_v58f_arg_validation_fails_before_any_file_io(tmp_path):
    """The paths do NOT exist: file I/O would raise FileNotFoundError, so a
    ValueError proves the argument gate fires FIRST."""
    missing = str(tmp_path / "nope.pt")
    with pytest.raises(ValueError, match="w4b_ckpt"):
        load_v58f(missing, missing, w4b_ckpt=None,
                  select_rule="rescorer-argmax", device="cpu")
    with pytest.raises(ValueError, match="w4b_ckpt"):
        load_v58f(missing, missing, w4b_ckpt=None,
                  select_rule="rescorer-top8-kincost", device="cpu")
    with pytest.raises(ValueError, match="ignored"):
        load_v58f(missing, missing, w4b_ckpt=missing,
                  select_rule="frozen-argmax", device="cpu")
    with pytest.raises(ValueError, match="select_rule"):
        load_v58f(missing, missing, select_rule="from-gate", device="cpu")


# ---------------------------------------------------------------------------
# the gate-decided default (both pre-registered branches)
# ---------------------------------------------------------------------------
def _gate(g1_pass, top8=0.12, pruner=True):
    return {"variant": "feat",
            "gate_G1_recalibration_suffices":
                {"pass": g1_pass, "threshold_m": 0.45,
                 "selected_ade": 0.41 if g1_pass else 0.60},
            "gate_G2_recalibration_insufficient":
                {"engaged": not g1_pass, "top8_oracle": top8,
                 "pruner_viable": pruner}}


def test_select_rule_from_gate_both_branches(tmp_path):
    rule, rec = select_rule_from_gate(_gate(True))
    assert rule == "rescorer-argmax"
    assert rec["g1_pass"] is True and rec["selected_ade"] == 0.41
    rule, rec = select_rule_from_gate(_gate(False))
    assert rule == "rescorer-top8-kincost"
    assert rec["g1_pass"] is False and rec["top8_oracle"] == 0.12
    assert "warning" not in rec                  # pruner viable
    # path form + the pruner-not-viable caveat travels with the decision
    p = tmp_path / "w4b_gate.json"
    p.write_text(json.dumps(_gate(False, top8=0.20, pruner=False)))
    rule, rec = select_rule_from_gate(p)
    assert rule == "rescorer-top8-kincost"
    assert rec["source"] == str(p)
    assert "warning" in rec and "W7" in rec["warning"]
    with pytest.raises(ValueError):
        select_rule_from_gate({"not": "a gate"})
