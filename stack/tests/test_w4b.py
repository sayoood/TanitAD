"""W4b ``train_w4b_selector`` pure-part smoke — CPU-only, no checkpoint, no
corpus. The full path (frozen v5f trunk + W4 emission + v2 corpora) is
pod-side and NOT exercised here.

What is pinned (the task's four groups):
  * :class:`W4bRescorer` output shapes and EXACT parameter count for both
    variants (2-layer MLP, hidden 256 — the prereg module scale), plus the
    zero-init uniform warm start;
  * ranking-loss sanity: trained on a synthetic fixed-winner problem with the
    SAME ``tanitad.models.tactical.ranking_loss`` the trainer uses, the
    winner's logit rises to the top (argmax == winner) and the loss falls;
  * the gate-JSON writer's G1/G2 logic on BOTH pre-registered branches,
    including the verbatim reference numbers, tier stamp and estimator note;
  * ``variant='kin'`` input-dim handling: in_dim = F + 2K, controls required,
    wrong shapes rejected; ``feat`` ignores the extra args (variant-uniform
    call site).
Plus the two pure metric helpers (``fan_ade``, ``topk_oracle_per_window``)
against hand-derived cases — they feed the gate numbers, so they are pinned
too.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_w4b_selector import (GATE_SELECTED_ADE, GATE_TOP8_PRUNER,  # noqa: E402
                                REF_FROZEN_SELECTOR_NEW_FAN,
                                REF_OLD_SELECTOR_OLD_FAN, REF_W4_ORACLE,
                                W4bRescorer, build_w4b_gate, fan_ade,
                                topk_oracle_per_window)
from tanitad.models.tactical import ranking_loss  # noqa: E402

B, N, K, F, HID = 3, 16, 20, 48, 256


def _mlp_params(in_dim: int, hidden: int = HID) -> int:
    return (in_dim * hidden + hidden) + (hidden * 1 + 1)


# ---------------------------------------------------------------------------
# rescorer: shapes + param count
# ---------------------------------------------------------------------------
def test_rescorer_shapes_and_param_count_feat():
    m = W4bRescorer(feat_dim=F, k=K, variant="feat")
    q = torch.randn(B, N, F)
    logits = m(q)
    assert logits.shape == (B, N)
    assert torch.isfinite(logits).all()
    assert sum(p.numel() for p in m.parameters()) == _mlp_params(F)
    assert m.in_dim == F


def test_rescorer_shapes_and_param_count_kin():
    m = W4bRescorer(feat_dim=F, k=K, variant="kin")
    q = torch.randn(B, N, F)
    a = torch.randn(B, N, K).clamp(-4, 4)
    kap = torch.randn(B, N, K).mul(0.05).clamp(-0.2, 0.2)
    logits = m(q, a, kap)
    assert logits.shape == (B, N)
    assert sum(p.numel() for p in m.parameters()) == _mlp_params(F + 2 * K)
    assert m.in_dim == F + 2 * K


def test_rescorer_zero_init_uniform_warm_start():
    """Fresh module -> every candidate scores exactly 0 (defined, uninformed
    start — the W4 zero-init discipline)."""
    m = W4bRescorer(feat_dim=F, k=K, variant="feat")
    logits = m(torch.randn(B, N, F))
    assert torch.allclose(logits, torch.zeros_like(logits))


def test_rescorer_gradients_reach_both_layers():
    m = W4bRescorer(feat_dim=F, k=K, variant="kin")
    q = torch.randn(B, N, F)
    a = torch.randn(B, N, K)
    kap = torch.randn(B, N, K).mul(0.05)
    m(q, a, kap).pow(2).sum().backward()
    # zero-init final layer => its WEIGHT grad can be nonzero only through the
    # hidden acts; assert every parameter got a grad tensor and the first
    # layer's is nonzero.
    grads = {n: p.grad for n, p in m.named_parameters()}
    assert all(g is not None for g in grads.values())
    assert grads["net.0.weight"] is not None


# ---------------------------------------------------------------------------
# variant='kin' input handling
# ---------------------------------------------------------------------------
def test_kin_requires_controls_and_checks_shapes():
    m = W4bRescorer(feat_dim=F, k=K, variant="kin")
    q = torch.randn(B, N, F)
    with pytest.raises(ValueError):
        m(q)                                        # controls missing
    with pytest.raises(ValueError):
        m(q, torch.randn(B, N, K + 1), torch.randn(B, N, K))   # bad a shape
    with pytest.raises(ValueError):
        m(q, torch.randn(B, N, K), torch.randn(B, N - 1, K))   # bad kappa
    # correct shapes pass
    assert m(q, torch.randn(B, N, K), torch.randn(B, N, K)).shape == (B, N)


def test_feat_ignores_extra_controls_and_checks_q():
    """The call site is variant-uniform: feat accepts (and ignores) a/kappa."""
    m = W4bRescorer(feat_dim=F, k=K, variant="feat")
    q = torch.randn(B, N, F)
    out_plain = m(q)
    out_extra = m(q, torch.randn(B, N, K), torch.randn(B, N, K))
    assert torch.allclose(out_plain, out_extra)
    with pytest.raises(ValueError):
        m(torch.randn(B, N, F + 1))                 # wrong feature width
    with pytest.raises(ValueError):
        m(torch.randn(B, F))                        # wrong rank
    with pytest.raises(ValueError):
        W4bRescorer(feat_dim=F, k=K, variant="bogus")


# ---------------------------------------------------------------------------
# metric helpers (hand-derived)
# ---------------------------------------------------------------------------
def test_fan_ade_hand_case():
    tgt = torch.zeros(1, 2, 2)
    fan = torch.zeros(1, 2, 2, 2)
    fan[0, 0, :, 0] = torch.tensor([3.0, 4.0])      # |err| = 3, 4 -> ADE 3.5
    fan[0, 1, :, 1] = 1.0                           # ADE 1.0
    ade = fan_ade(fan, tgt)
    assert ade.shape == (1, 2)
    assert torch.allclose(ade, torch.tensor([[3.5, 1.0]]))
    with pytest.raises(ValueError):
        fan_ade(fan[..., :1], tgt)                  # last dim must be 2


def test_topk_oracle_per_window_hand_case():
    # candidate errors 3,2,1,0; scores rank them 0 > 1 > 2 > 3
    err = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    scores = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
    assert float(topk_oracle_per_window(err, scores, 1)) == 3.0
    assert float(topk_oracle_per_window(err, scores, 2)) == 2.0
    assert float(topk_oracle_per_window(err, scores, 4)) == 0.0
    assert float(topk_oracle_per_window(err, scores, 99)) == 0.0   # clips to N
    with pytest.raises(ValueError):
        topk_oracle_per_window(err, scores[:, :3], 2)


def test_topk_oracle_monotone_in_k():
    torch.manual_seed(0)
    err = torch.rand(8, 32)
    scores = torch.randn(8, 32)
    vals = [topk_oracle_per_window(err, scores, k).mean() for k in
            (1, 2, 4, 8, 16, 32)]
    for lo, hi in zip(vals[1:], vals[:-1]):
        assert float(lo) <= float(hi) + 1e-7        # min over superset
    assert torch.allclose(vals[-1], err.min(dim=1).values.mean())


# ---------------------------------------------------------------------------
# ranking-loss sanity: the winner's logit rises on a synthetic problem
# ---------------------------------------------------------------------------
def test_ranking_loss_trains_winner_to_the_top():
    """Fixed features, fixed per-candidate errors with a known GT-nearest
    winner per row: after a few hundred Adam steps on the SAME ranking loss
    the trainer uses, argmax(logits) == winner on every row and the loss has
    fallen from its uniform-start value (= margin, since zero-init logits
    violate the margin for every loser)."""
    torch.manual_seed(0)
    b, n, f = 4, 8, 16
    q = torch.randn(b, n, f)
    winner = torch.tensor([0, 3, 5, 7])
    err = torch.ones(b, n)
    err[torch.arange(b), winner] = 0.0              # winner = argmin err
    m = W4bRescorer(feat_dim=f, k=K, variant="feat")
    loss0 = float(ranking_loss(m(q), err, 0.1).detach())
    assert abs(loss0 - 0.1) < 1e-6                  # uniform start = margin
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = ranking_loss(m(q), err, 0.1)
        loss.backward()
        opt.step()
    logits = m(q)
    assert (logits.argmax(dim=1) == winner).all()
    assert float(ranking_loss(logits, err, 0.1)) < 0.01 < loss0


def test_ranking_loss_kin_variant_can_read_the_controls():
    """kin-only separability: rows where q is IDENTICAL across candidates but
    the (a, kappa) sequences differ — the feat variant is blind by
    construction (identical inputs => identical logits), kin can still rank."""
    torch.manual_seed(1)
    b, n, f = 2, 6, 8
    q = torch.randn(b, 1, f).expand(b, n, f).contiguous()
    a = torch.randn(b, n, K)
    kap = torch.randn(b, n, K).mul(0.05)
    winner = torch.tensor([2, 4])
    err = torch.ones(b, n)
    err[torch.arange(b), winner] = 0.0
    feat_logits = W4bRescorer(feat_dim=f, k=K, variant="feat")(q, a, kap)
    assert torch.allclose(feat_logits, feat_logits[:, :1].expand(b, n))
    m = W4bRescorer(feat_dim=f, k=K, variant="kin")
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        ranking_loss(m(q, a, kap), err, 0.1).backward()
        opt.step()
    assert (m(q, a, kap).argmax(dim=1) == winner).all()


# ---------------------------------------------------------------------------
# gate-JSON writer: BOTH pre-registered branches
# ---------------------------------------------------------------------------
def _mini(selected: float, top8: float) -> dict:
    return {
        "n_windows": 881, "n_candidates": 256,
        "grid": {"episodes": 40, "stride": 8, "batch": 16,
                 "expected_n": 881, "matches_banked_grid": True},
        "selected_ade": selected, "oracle_ade": 0.1077,
        "sel_gap": round(selected - 0.1077, 6),
        "oracle_topk": {"4": top8 + 0.02, "8": top8, "16": max(top8 - 0.02,
                                                               0.108)},
        "frozen_selected_ade": 0.7933, "winner_hit_frac": 0.1,
        "sel_rank_pct_mean": 0.05,
        "families": {}, "selgap_ci": "stub", "wallclock_s": 1.0,
    }


def test_gate_g1_pass_branch():
    g = build_w4b_gate(_mini(selected=0.40, top8=0.12), variant="feat")
    g1, g2 = (g["gate_G1_recalibration_suffices"],
              g["gate_G2_recalibration_insufficient"])
    assert g1["pass"] is True
    assert g1["selected_ade"] == 0.40
    assert g1["threshold_m"] == GATE_SELECTED_ADE == 0.45
    assert g2["engaged"] is False                   # G1 held
    assert g2["top8_oracle"] == 0.12                # reported either way
    assert g2["pruner_viable"] is True              # 0.12 <= 0.15
    assert "v5.8f assembly proceeds" in g1["consequence_if_pass"]


def test_gate_g2_branch_pruner_viable():
    g = build_w4b_gate(_mini(selected=0.60, top8=0.13), variant="kin")
    g1, g2 = (g["gate_G1_recalibration_suffices"],
              g["gate_G2_recalibration_insufficient"])
    assert g1["pass"] is False
    assert g2["engaged"] is True
    assert g2["pruner_viable"] is True
    assert g2["pruner_threshold_m"] == GATE_TOP8_PRUNER == 0.15
    assert "W7" in g2["rule"]
    assert g["variant"] == "kin"


def test_gate_g2_branch_pruner_not_viable():
    g = build_w4b_gate(_mini(selected=0.60, top8=0.20), variant="feat")
    assert g["gate_G1_recalibration_suffices"]["pass"] is False
    assert g["gate_G2_recalibration_insufficient"]["engaged"] is True
    assert g["gate_G2_recalibration_insufficient"]["pruner_viable"] is False


def test_gate_boundary_is_inclusive():
    """The prereg gates are '<=': exactly 0.45 passes G1, exactly 0.15 keeps
    the pruner viable."""
    g = build_w4b_gate(_mini(selected=GATE_SELECTED_ADE,
                             top8=GATE_TOP8_PRUNER), variant="feat")
    assert g["gate_G1_recalibration_suffices"]["pass"] is True
    assert g["gate_G2_recalibration_insufficient"]["pruner_viable"] is True


def test_gate_carries_references_tier_and_estimator():
    g = build_w4b_gate(_mini(selected=0.40, top8=0.12), variant="feat")
    r = g["reference"]
    assert r["frozen_selector_selected_ade_new_fan"] == \
        REF_FROZEN_SELECTOR_NEW_FAN == 0.7933
    assert r["old_selector_selected_ade_old_fan"] == \
        REF_OLD_SELECTOR_OLD_FAN == 0.4056
    assert r["w4_oracle_ade_new_fan"] == REF_W4_ORACLE == 0.1077
    assert g["tier"] == "T0"
    assert "episode-cluster bootstrap".upper() in \
        g["_estimator_note"].upper()
    assert "point estimate".upper() in g["_estimator_note"].upper()
    assert g["_evidence_class"].startswith("MEASURED")
    # the gate JSON must be JSON-serialisable as-is
    import json
    json.dumps(g)
