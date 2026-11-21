"""W4c ``train_w4c_spatial`` pure-part smoke — CPU-only, no checkpoint, no
corpus. The full path (frozen v5f trunk + W4 emission + v2 corpora) is
pod-side and NOT exercised here.

What is pinned (the task's five groups):
  * :class:`W4cSpatialScorer` shapes, the ~1-2M parameter BAND at the real
    flagship dims (spatial D=768, d=256, K=20), the zero-init uniform warm
    start, and the dropout-on-the-query-embedding placement (p=1.0 in train
    mode collapses every candidate to the same logit — the dropout acts on
    the QUERY path, structurally verified by behaviour);
  * query-embedding input handling: ``candidate_query_features`` layout +
    normalisation on a hand case, shape rejection, and the scorer's own
    qfeat/tokens contract checks;
  * rank-loss sanity: trained on a synthetic fixed-winner problem with the
    SAME ``tanitad.models.tactical.ranking_loss`` the trainer uses, the
    winner's logit rises to the top — and a spatial-separability case where
    the QUERY features are identical across candidates and only the TOKENS
    disambiguate is included conceptually via distinct query geometry (the
    scorer's only trainable route to the tokens is cross-attention);
  * the gate-JSON writer on ALL THREE pre-registered branches, including the
    G-mode ONLY-WITH-G1c coupling (low entropy without G1-c must NOT pass),
    the G-null consequence text (fast selector retired to a W7-distillation
    target), the verbatim reference numbers, tier stamp and estimator note;
  * the spatial-tap shape contract on a mock encoder: last-frame slice,
    exactly-one-capture strictness, wrong-shape rejection, remove().
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_w4c_spatial import (ENDPOINT_SCALE, GATE_ENTROPY,  # noqa: E402
                               GATE_SELECTED_ADE, REF_ENTROPY_REFC,
                               REF_ENTROPY_V5F, REF_FROZEN_SELECTOR_NEW_FAN,
                               REF_W4_ORACLE, REF_W4B_FEAT, REF_W4B_KIN,
                               W4C_PARAM_BAND, SpatialTokenTap,
                               W4cSpatialScorer, build_w4c_gate,
                               candidate_query_features, probe_indices,
                               query_feat_dim, selection_entropy)
from train_v58f_unicycle_head import A_MAX, KAPPA_MAX  # noqa: E402
from tanitad.models.tactical import ranking_loss  # noqa: E402

B, N, K = 3, 16, 20
FQ = query_feat_dim(K)                      # 2K + 4 = 44


# ---------------------------------------------------------------------------
# scorer: shapes + the prereg parameter band at REAL flagship dims
# ---------------------------------------------------------------------------
def test_scorer_shapes_real_dims_and_param_band():
    """Flagship dims: spatial tokens D=768 (config.py:367), d=256, K=20 —
    the constructed head must land inside the prereg ~1-2M band."""
    m = W4cSpatialScorer(spatial_dim=768, k=K, d=256, n_heads=8, ff_mult=4)
    qf = torch.randn(B, N, FQ)
    tok = torch.randn(B, 429, 768)          # 11x39 grid at 176x624/patch 16
    logits = m(qf, tok)
    assert logits.shape == (B, N)
    assert torch.isfinite(logits).all()
    n_par = sum(p.numel() for p in m.parameters())
    lo, hi = W4C_PARAM_BAND
    assert lo <= n_par <= hi, f"{n_par} outside [{lo}, {hi}]"


def test_scorer_zero_init_uniform_warm_start():
    """Fresh module -> every candidate scores exactly 0 (defined, uninformed
    start — the W4/W4b zero-init discipline), in train AND eval mode (the
    zero-init logit layer makes it dropout-independent)."""
    m = W4cSpatialScorer(spatial_dim=32, k=K, d=32, n_heads=4, dropout=0.5)
    qf, tok = torch.randn(B, N, FQ), torch.randn(B, 7, 32)
    m.train()
    assert torch.allclose(m(qf, tok), torch.zeros(B, N))
    m.eval()
    assert torch.allclose(m(qf, tok), torch.zeros(B, N))


def test_scorer_dropout_sits_on_the_query_embedding():
    """p=1.0 in train mode zeroes the query embedding -> every candidate's
    query is identical -> identical logits across candidates (after training
    the non-query paths). In eval mode dropout is off and candidates
    separate. This pins the dropout to the QUERY path — the prereg's
    anti-memorisation lever — not somewhere decorative."""
    torch.manual_seed(0)
    m = W4cSpatialScorer(spatial_dim=16, k=K, d=32, n_heads=4, dropout=1.0)
    # give the non-query weights signal so a leak elsewhere would show
    with torch.no_grad():
        m.logit.weight.fill_(0.1)
        m.logit.bias.fill_(0.05)
    qf = torch.randn(B, N, FQ)
    tok = torch.randn(B, 5, 16)
    m.train()
    out = m(qf, tok)
    assert torch.allclose(out, out[:, :1].expand(B, N), atol=1e-6)
    m.eval()
    out_eval = m(qf, tok)
    assert not torch.allclose(out_eval, out_eval[:, :1].expand(B, N))


def test_scorer_gradients_reach_all_parameters():
    m = W4cSpatialScorer(spatial_dim=16, k=K, d=32, n_heads=4, dropout=0.0)
    qf, tok = torch.randn(B, N, FQ), torch.randn(B, 5, 16)
    m(qf, tok).pow(2).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name


# ---------------------------------------------------------------------------
# query-embedding input handling
# ---------------------------------------------------------------------------
def test_candidate_query_features_hand_case():
    """Constant controls: a=A_MAX, kappa=-KAPPA_MAX everywhere; a known
    endpoint. Layout = (a-norm K, kappa-norm K, endpoint/ENDPOINT_SCALE,
    max|kappa|-norm, mean a-norm)."""
    a = torch.full((1, 1, K), A_MAX)
    kap = torch.full((1, 1, K), -KAPPA_MAX)
    fan = torch.zeros(1, 1, K, 2)
    fan[0, 0, -1] = torch.tensor([25.0, -10.0])
    qf = candidate_query_features(a, kap, fan)
    assert qf.shape == (1, 1, FQ)
    assert torch.allclose(qf[0, 0, :K], torch.ones(K))            # a/A_MAX
    assert torch.allclose(qf[0, 0, K:2 * K], -torch.ones(K))      # kappa norm
    assert torch.allclose(qf[0, 0, 2 * K:2 * K + 2],
                          torch.tensor([25.0, -10.0]) / ENDPOINT_SCALE)
    assert float(qf[0, 0, 2 * K + 2]) == pytest.approx(1.0)       # max|kappa|
    assert float(qf[0, 0, 2 * K + 3]) == pytest.approx(1.0)       # mean a


def test_candidate_query_features_rejects_bad_shapes():
    a = torch.randn(B, N, K)
    kap = torch.randn(B, N, K)
    fan = torch.randn(B, N, K, 2)
    with pytest.raises(ValueError):
        candidate_query_features(a, kap[:, :, :-1], fan)      # kappa K off
    with pytest.raises(ValueError):
        candidate_query_features(a, kap, fan[..., :1])        # last dim != 2
    with pytest.raises(ValueError):
        candidate_query_features(a, kap, fan[:, :-1])         # fan N off
    with pytest.raises(ValueError):
        candidate_query_features(a[0], kap[0], fan[0])        # rank 2
    assert candidate_query_features(a, kap, fan).shape == (B, N, FQ)


def test_scorer_rejects_bad_inputs():
    m = W4cSpatialScorer(spatial_dim=16, k=K, d=32, n_heads=4)
    qf, tok = torch.randn(B, N, FQ), torch.randn(B, 5, 16)
    with pytest.raises(ValueError):
        m(torch.randn(B, N, FQ + 1), tok)         # wrong query width
    with pytest.raises(ValueError):
        m(qf[0], tok)                             # wrong query rank
    with pytest.raises(ValueError):
        m(qf, torch.randn(B, 5, 17))              # wrong token dim
    with pytest.raises(ValueError):
        m(qf, tok[0])                             # wrong token rank
    with pytest.raises(ValueError):
        m(qf, tok[:-1])                           # batch mismatch
    assert m(qf, tok).shape == (B, N)


# ---------------------------------------------------------------------------
# rank-loss sanity: the winner rises on a synthetic spatial problem
# ---------------------------------------------------------------------------
def test_ranking_loss_trains_winner_to_the_top():
    """Fixed query features + fixed spatial tokens with a known GT-nearest
    winner per row: after a few hundred Adam steps on the SAME ranking loss
    the trainer uses, argmax(logits) == winner on every row and the loss has
    fallen from its uniform-start value (= margin, since zero-init logits
    violate the margin for every loser)."""
    torch.manual_seed(0)
    b, n = 4, 8
    m = W4cSpatialScorer(spatial_dim=16, k=K, d=32, n_heads=4, dropout=0.0)
    qf = torch.randn(b, n, FQ)
    tok = torch.randn(b, 6, 16)
    winner = torch.tensor([0, 3, 5, 7])
    err = torch.ones(b, n)
    err[torch.arange(b), winner] = 0.0            # winner = argmin err
    loss0 = float(ranking_loss(m(qf, tok), err, 0.1).detach())
    assert abs(loss0 - 0.1) < 1e-6                # uniform start = margin
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = ranking_loss(m(qf, tok), err, 0.1)
        loss.backward()
        opt.step()
    m.eval()
    logits = m(qf, tok)
    assert (logits.argmax(dim=1) == winner).all()
    assert float(ranking_loss(logits, err, 0.1)) < 0.01 < loss0


def test_spatial_tokens_can_disambiguate_identical_queries():
    """The spatial hypothesis in miniature: two batch rows with IDENTICAL
    per-candidate query features but DIFFERENT spatial tokens and different
    winners — only a scorer that reads SPACE can rank both rows correctly."""
    torch.manual_seed(1)
    n = 6
    qf_row = torch.randn(1, n, FQ)
    qf = qf_row.expand(2, n, FQ).contiguous()     # same candidates, both rows
    tok = torch.randn(2, 5, 16)                   # different scenes
    winner = torch.tensor([1, 4])
    err = torch.ones(2, n)
    err[torch.arange(2), winner] = 0.0
    m = W4cSpatialScorer(spatial_dim=16, k=K, d=32, n_heads=4, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        ranking_loss(m(qf, tok), err, 0.1).backward()
        opt.step()
    m.eval()
    assert (m(qf, tok).argmax(dim=1) == winner).all()


# ---------------------------------------------------------------------------
# entropy instrument
# ---------------------------------------------------------------------------
def test_selection_entropy_bounds():
    n = 256
    uniform = torch.zeros(2, n)
    ent_u = selection_entropy(uniform)
    assert torch.allclose(ent_u, torch.full((2,), float(torch.log(
        torch.tensor(float(n))))), atol=1e-5)
    peaked = torch.zeros(1, n)
    peaked[0, 3] = 50.0
    assert float(selection_entropy(peaked)) < 1e-3
    with pytest.raises(ValueError):
        selection_entropy(torch.zeros(n))


def test_probe_indices_even_and_bounded():
    grid = list(range(881))
    idx = probe_indices(grid, 64)
    assert len(idx) == 64
    assert len(set(idx)) == 64
    assert idx[0] == 0 and idx[-1] > 700          # spans the grid
    assert probe_indices(grid, 0) == []
    assert probe_indices([], 64) == []
    assert probe_indices([5, 7], 64) == [5, 7]    # short grid clips


# ---------------------------------------------------------------------------
# gate-JSON writer: ALL THREE branches + the only-with-G1c coupling
# ---------------------------------------------------------------------------
def _mini(selected: float, entropy: float, top8: float = 0.32) -> dict:
    return {
        "n_windows": 881, "n_candidates": 256,
        "grid": {"episodes": 40, "stride": 8, "batch": 16,
                 "expected_n": 881, "matches_banked_grid": True},
        "selected_ade": selected, "oracle_ade": 0.1077,
        "sel_gap": round(selected - 0.1077, 6),
        "oracle_topk": {"4": top8 + 0.07, "8": top8,
                        "16": max(top8 - 0.07, 0.108)},
        "frozen_selected_ade": 0.7933, "winner_hit_frac": 0.1,
        "sel_rank_pct_mean": 0.05,
        "entropy": {"mean": entropy, "median": entropy, "p90": entropy + 0.4,
                    "n_eff_mean": 4.0, "top1_prob_mean": 0.5, "note": "stub"},
        "families": {}, "selgap_ci": "stub", "wallclock_s": 1.0,
    }


_TVH = {"final_train_selected_ade": 0.30,
        "final_heldout_probe_selected_ade": 0.42, "final_gap": 0.12,
        "history": [], "probe_n_windows": 64, "note": "stub"}


def test_gate_branch_g1c_pass_gmode_pass():
    g = build_w4c_gate(_mini(selected=0.40, entropy=1.1),
                       train_vs_heldout=_TVH)
    assert g["gate_G1c_port_works"]["pass"] is True
    assert g["gate_G1c_port_works"]["threshold_m"] == GATE_SELECTED_ADE == 0.45
    gm = g["gate_Gmode_mechanism_check"]
    assert gm["entropy_le_threshold"] is True
    assert gm["pass"] is True                     # entropy ok AND G1-c held
    assert gm["threshold_nats"] == GATE_ENTROPY == 1.5
    assert g["gate_Gnull"]["engaged"] is False


def test_gate_branch_g1c_pass_gmode_entropy_high():
    g = build_w4c_gate(_mini(selected=0.40, entropy=2.0),
                       train_vs_heldout=_TVH)
    assert g["gate_G1c_port_works"]["pass"] is True
    gm = g["gate_Gmode_mechanism_check"]
    assert gm["entropy_le_threshold"] is False
    assert gm["pass"] is False
    assert g["gate_Gnull"]["engaged"] is False


def test_gate_branch_gnull_low_entropy_does_NOT_pass_gmode():
    """THE ONLY-WITH-G1c COUPLING: entropy 0.9 (better than REF-C's 0.97) but
    selected ADE 0.60 — G-mode must FAIL (entropy alone proves nothing) and
    G-null engages with the retirement consequence verbatim."""
    g = build_w4c_gate(_mini(selected=0.60, entropy=0.9),
                       train_vs_heldout=_TVH)
    assert g["gate_G1c_port_works"]["pass"] is False
    gm = g["gate_Gmode_mechanism_check"]
    assert gm["entropy_le_threshold"] is True     # raw entropy is fine...
    assert gm["pass"] is False                    # ...but only-with-G1c
    assert "entropy alone proves nothing" in gm["rule"]
    gn = g["gate_Gnull"]
    assert gn["engaged"] is True
    assert "ENTIRELY to W7 WM-roll re-rank" in gn["rule"]
    assert "retired to a W7-distillation target (L4)" in gn["rule"]
    assert "no third scoring attempt without new evidence" in gn["rule"]


def test_gate_boundaries_are_inclusive():
    g = build_w4c_gate(_mini(selected=GATE_SELECTED_ADE,
                             entropy=GATE_ENTROPY), train_vs_heldout=_TVH)
    assert g["gate_G1c_port_works"]["pass"] is True
    assert g["gate_Gmode_mechanism_check"]["pass"] is True
    assert g["gate_Gnull"]["engaged"] is False


def test_gate_carries_references_tvh_tier_and_estimator():
    g = build_w4c_gate(_mini(selected=0.40, entropy=1.1),
                       train_vs_heldout=_TVH)
    r = g["reference"]
    assert r["w4b_feat_selected_ade"] == REF_W4B_FEAT == 0.5600
    assert r["w4b_kin_selected_ade"] == REF_W4B_KIN == 0.5637
    assert r["frozen_selector_selected_ade_new_fan"] == \
        REF_FROZEN_SELECTOR_NEW_FAN == 0.7933
    assert r["w4_oracle_ade_new_fan"] == REF_W4_ORACLE == 0.1077
    assert r["refc_conf_pass_entropy"] == REF_ENTROPY_REFC == 0.97
    assert r["v5f_sel_score_entropy"] == REF_ENTROPY_V5F == 2.22
    assert g["train_vs_heldout"]["final_gap"] == 0.12   # reported explicitly
    assert g["tier"] == "T0"
    assert "episode-cluster bootstrap".upper() in \
        g["_estimator_note"].upper()
    assert "point estimate".upper() in g["_estimator_note"].upper()
    assert g["_evidence_class"].startswith("MEASURED")
    # the gate JSON must be JSON-serialisable as-is
    import json
    json.dumps(g)


# ---------------------------------------------------------------------------
# spatial-tap shape contract (mock encoder — no v5f checkpoint on this box)
# ---------------------------------------------------------------------------
class _MockEncoder(nn.Module):
    """Stands in for ViTEncoder: [BW, C] -> token grid [BW, N_tok, D]."""

    def __init__(self, n_tok: int = 6, d: int = 8):
        super().__init__()
        self.n_tok, self.d = n_tok, d
        self.lin = nn.Linear(4, n_tok * d)

    def forward(self, x):
        return self.lin(x).reshape(x.shape[0], self.n_tok, self.d)


def test_spatial_tap_last_frame_slice():
    """The tap must return exactly the LAST frame's tokens of the [B*W, N, D]
    capture, [B, N, D] float32 detached."""
    enc = _MockEncoder()
    tap = SpatialTokenTap(enc)
    b, w = 2, 3
    x = torch.randn(b * w, 4)
    tap.clear()
    full = enc(x)                                 # ONE encode_window-like call
    tokens = tap.last_frame(b, w)
    assert tokens.shape == (b, enc.n_tok, enc.d)
    assert tokens.dtype == torch.float32
    assert not tokens.requires_grad
    expect = full.reshape(b, w, enc.n_tok, enc.d)[:, -1]
    assert torch.allclose(tokens, expect)
    tap.remove()


def test_spatial_tap_strict_one_capture():
    enc = _MockEncoder()
    tap = SpatialTokenTap(enc)
    with pytest.raises(RuntimeError):
        tap.last_frame(1, 1)                      # nothing captured
    tap.clear()
    enc(torch.randn(6, 4))
    enc(torch.randn(6, 4))                        # a SECOND pass
    assert tap.n_calls() == 2
    with pytest.raises(RuntimeError):
        tap.last_frame(2, 3)                      # must fail loudly
    tap.clear()
    enc(torch.randn(6, 4))
    with pytest.raises(ValueError):
        tap.last_frame(2, 2)                      # B*W mismatch (6 != 4)
    assert tap.last_frame(2, 3).shape == (2, enc.n_tok, enc.d)
    tap.remove()
    enc(torch.randn(6, 4))                        # after remove: no capture
    assert tap.n_calls() == 1                     # unchanged buffer
