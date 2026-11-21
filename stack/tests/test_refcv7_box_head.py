"""refcv7 A9 -- the REFINED slot heads (R1 focal presence, R2 deep supervision, R4 300 queries), the P0 detection
metrics, the G-DVB entries and the trainer wiring. SPEC_REFCV7 §14 / §14.1.

Every property is asserted with a LITERAL expectation and paired with a deliberate-regression arm that goes RED.
The five pins the brief names: the focal-vs-BCE presence optimum (analytic), the per-layer loss count == decoder
depth, 300 queries with 120 targets drop nothing, the DontCare scoring -- and (in test_refcv7_vis1.py) a hidden car
is not a positive.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import agent_slots as AS           # noqa: E402
from tanitad.models import box3d_head as B3            # noqa: E402
from tanitad.models import slot_presence as SP         # noqa: E402
from tanitad.eval import detection_metrics as DM       # noqa: E402


def _dec(n_q=12, depth=3, prior=None, deep=False, d_mem=16, n_mem=8, box3d=False):
    torch.manual_seed(0)
    cls = B3.Box3DSlotDecoder if box3d else AS.AgentSlotDecoder
    d = cls(d_mem, n_mem, n_queries=n_q, d_model=32, depth=depth, n_heads=4, enforce_band=False,
            presence_prior=prior)
    d.deep_supervision = deep
    return d


def _tgt(rows, pad=None, box3d=False):
    a = len(rows)
    p = a if pad is None else pad
    box = torch.zeros(1, p, 4)
    valid = torch.zeros(1, p, dtype=torch.bool)
    cls = torch.full((1, p), -1, dtype=torch.long)
    if a:
        box[0, :a] = torch.tensor([list(r[:4]) for r in rows], dtype=torch.float32)
        valid[0, :a] = True
        cls[0, :a] = torch.tensor([int(r[4]) if len(r) > 4 else 0 for r in rows])
    t = {"box": box, "yaw": torch.zeros(1, p), "cls": cls, "valid": valid, "occ": torch.full((1, p), -1.0),
         "rates": torch.zeros(1, p, 3), "rates_mask": torch.zeros(1, p, dtype=torch.bool)}
    if box3d:
        t.update(cz=torch.full((1, p), 0.8), h=torch.full((1, p), 1.6), zh_mask=valid.clone())
    return t


# ========================================================================================================= #
# R1 -- the presence objective: the ANALYTIC optimum                                                        #
# ========================================================================================================= #
def test_R1_the_05_gate_means_pi_075_under_focal_and_1_over_11_under_bce():
    assert SP.gate_match_belief(0.5, "focal") == pytest.approx(0.75, abs=1e-6)
    assert SP.gate_match_belief(0.5, "bce_noobj") == pytest.approx(1.0 / 11.0, abs=1e-6)
    # closed forms, independently: BCE-0.1 optimum p* = pi / (pi + 0.1 (1 - pi))
    for pi in (0.05, 0.3, 0.6, 0.9):
        assert SP.presence_optimum(pi, "bce_noobj") == pytest.approx(pi / (pi + 0.1 * (1 - pi)), abs=1e-6)
    # focal is UNDER-confident relative to BCE-0.1 at every pi (literature pass D1)
    for pi in (0.2, 0.5, 0.8):
        assert SP.presence_optimum(pi, "focal") < SP.presence_optimum(pi, "bce_noobj")


def test_R1_RED_ARM_a_swapped_alpha_moves_the_gate_to_pi_025(monkeypatch):
    """Mutation: alpha on the NEGATIVES (the classic focal-loss sign slip). The same analytic check must read 0.25,
    i.e. the check discriminates the correct loss from the broken one."""
    monkeypatch.setattr(SP, "FOCAL_ALPHA", 0.75)
    assert SP.gate_match_belief(0.5, "focal") == pytest.approx(0.25, abs=1e-6)


def test_R1_focal_elementwise_is_torchvisions_sigmoid_focal_loss():
    tv = pytest.importorskip("torchvision.ops")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(200, generator=g) * 3
    y = (torch.rand(200, generator=g) > 0.7).float()
    ref = tv.sigmoid_focal_loss(x, y, alpha=0.25, gamma=2.0, reduction="none")
    assert torch.allclose(SP.sigmoid_focal_elementwise(x, y), ref, atol=1e-6)


def test_R1_focal_cost_orders_confident_slots_first_and_is_mmdets_formula():
    logit = torch.tensor([-4.0, 0.0, 4.0])
    c = SP.focal_presence_cost(logit)
    assert c[2] < c[1] < c[0]
    p = torch.sigmoid(logit)
    ref = (-(p + 1e-12).log() * 0.25 * (1 - p) ** 2) - (-(1 - p + 1e-12).log() * 0.75 * p ** 2)
    assert torch.allclose(c, ref)


def test_R1_the_prior_initialises_the_bias_and_the_default_is_unchanged():
    d = _dec(prior=0.01)
    b = float(d.head.bias[AS.SLOT_SLICES["presence"]])
    assert b == pytest.approx(math.log(0.01 / 0.99), abs=1e-6)
    d0 = _dec()
    assert float(d0.head.bias[AS.SLOT_SLICES["presence"]]) == pytest.approx(math.log(0.05 / 0.95), abs=1e-6)
    assert d0.presence_prior == 0.05 and AS.AgentSlotDecoder.PRESENCE_PRIOR == 0.05
    d3 = _dec(prior=0.01, box3d=True)                  # the 3-D head COPIES the 2-D init
    assert float(d3.head.bias[B3.SLOT3D_SLICES["presence"]]) == pytest.approx(math.log(0.01 / 0.99), abs=1e-6)
    with pytest.raises(ValueError):
        _dec(prior=1.0)


def _tip_match_cost(pred, tgt, b, keep):
    """The PRE-A9 cost, re-stated LITERALLY from agent_slots.py:767-785 at tip 68e8ec6 (never imported)."""
    box_p, box_t = pred["box"][b], tgt["box"][b][keep]
    centre = (box_p[:, None, :2] - box_t[None, :, :2]).abs().sum(-1)
    size = (box_p[:, None, 2:] - box_t[None, :, 2:]).abs().sum(-1)
    cls_t = tgt["cls"][b][keep]
    prob = pred["cls_logits"][b].softmax(-1)
    cls_c = torch.where((cls_t >= 0)[None, :], -prob[:, cls_t.clamp_min(0)], torch.zeros(1))
    pres = -pred["presence_logit"][b].sigmoid()[:, None].expand_as(centre)
    return (1.0 * centre + 0.5 * size + 1.0 * cls_c + 1.0 * pres).detach().double().numpy()


def test_R1_the_sigmoid_matcher_is_bit_identical_to_the_tip_and_focal_changes_the_assignment():
    d = _dec(n_q=6)
    torch.manual_seed(3)
    with torch.no_grad():
        pred = d(torch.randn(2, 8, 16))
    tgt = _tgt([(10, 1, 4, 2, 0), (20, -2, 4, 2, 5)])
    tgt = {k: torch.cat([v, v], 0) for k, v in tgt.items()}
    keep = tgt["valid"][0].nonzero().flatten()
    assert np.array_equal(AS._match_cost(pred, tgt, 0, keep), _tip_match_cost(pred, tgt, 0, keep))
    assert np.array_equal(AS._match_cost(pred, tgt, 0, keep, presence_cost="sigmoid"),
                          _tip_match_cost(pred, tgt, 0, keep))
    # the literature's D2 case: slot A 1.0 m off at p 0.9, slot B dead-on at p 0.05. The pre-A9 cost is
    # geometry-dominated (B: -0.05 < A: 0.1); focal x 2.0 buys A (-1.80 < B: +1.35).
    p2 = {"box": torch.tensor([[[11.0, 1.0, 4, 2], [10.0, 1.0, 4, 2]]]),
          "presence_logit": torch.tensor([[math.log(0.9 / 0.1), math.log(0.05 / 0.95)]]),
          "cls_logits": torch.zeros(1, 2, 10)}
    t2 = _tgt([(10.0, 1.0, 4, 2, -1)])
    assert AS.match_slots(p2, t2)["rows"][0].tolist() == [1]
    assert AS.match_slots(p2, t2, presence_cost="focal")["rows"][0].tolist() == [0]
    with pytest.raises(ValueError):
        AS.match_slots(p2, t2, presence_cost="softmax")


def test_R1_focal_presence_is_normalised_by_matched_gt_not_by_slots():
    logit = torch.zeros(2, 10)
    m = {"rows": [torch.tensor([0, 1]), torch.tensor([3])], "cols": [], "n_target": [], "n_dropped": []}
    r = SP.presence_term(logit, m, mode="focal")
    t = torch.zeros(2, 10)
    t[0, 0] = t[0, 1] = t[1, 3] = 1.0
    ref = SP.sigmoid_focal_elementwise(logit, t).sum() / 3.0
    assert torch.allclose(r["loss"], ref) and r["n_matched"] == 3


def test_R1_bce_mode_reproduces_the_historical_presence_term_exactly():
    d = _dec(n_q=8)
    pred = d(torch.randn(1, 8, 16))
    tgt = _tgt([(10, 1, 4, 2, 0), (25, -3, 4, 2, 5)])
    m = AS.match_slots(pred, tgt)
    ref = AS.slot_set_loss(pred, tgt, match=m)["loss_presence"]
    got = SP.presence_term(pred["presence_logit"], m, mode="bce")["loss"]
    assert torch.equal(got, ref)


# ========================================================================================================= #
# R2 -- per-layer supervision                                                                               #
# ========================================================================================================= #
def test_R2_zero_new_parameters_and_the_last_layer_is_bit_identical():
    d = _dec(n_q=7, deep=False)
    n0 = d.n_params
    mem = torch.randn(2, 8, 16)
    off = d(mem)
    d.deep_supervision = True
    on = d(mem)
    assert d.n_params == n0
    for k in ("presence_logit", "cls_logits", "box", "yaw_vec", "rates", "occ_logit", "raw"):
        assert torch.equal(off[k], on[k]), k
    assert len(on["aux"]) == 2 and "aux" not in off


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_R2_the_per_layer_loss_count_equals_the_decoder_depth(depth):
    d = _dec(n_q=9, depth=depth, deep=True)
    pred = d(torch.randn(1, 8, 16))
    tgt = _tgt([(10, 1, 4, 2, 0), (30, -2, 4, 2, 5)])
    r = SP.refined_slot_losses(pred, tgt, head="agent", presence_loss="focal", vis1=False)
    assert r["n"]["layers"] == depth
    assert sorted(k for k in r if k.startswith("loss_layer")) == [f"loss_layer{i}" for i in range(depth)]
    tot = sum(r[f"loss_layer{i}"] for i in range(depth))
    assert torch.allclose(r["total"], tot)


def test_R2_RED_ARM_without_the_flag_only_one_layer_is_supervised():
    d = _dec(n_q=9, depth=3, deep=False)
    pred = d(torch.randn(1, 8, 16))
    r = SP.refined_slot_losses(pred, _tgt([(10, 1, 4, 2, 0)]), head="agent", presence_loss="focal", vis1=False)
    assert r["n"]["layers"] == 1 != 3


def test_R2_every_layer_is_rematched_and_every_layer_gets_gradient():
    d = _dec(n_q=9, depth=3, deep=True)
    pred = d(torch.randn(1, 8, 16))
    r = SP.refined_slot_losses(pred, _tgt([(10, 1, 4, 2, 0), (30, -2, 4, 2, 5)]), head="agent",
                               presence_loss="focal", vis1=False)
    r["loss_layer0"].backward(retain_graph=True)
    g_first = d.blocks.layers[0].linear1.weight.grad.abs().sum()
    assert float(g_first) > 0.0, "layer 0's own loss must reach layer 0"
    assert d.blocks.layers[2].linear1.weight.grad is None or \
        float(d.blocks.layers[2].linear1.weight.grad.abs().sum()) == 0.0, "layer 0's loss must not reach layer 2"


def test_R2_select_slots_narrows_the_aux_layers_too():
    d = _dec(n_q=5, depth=3, deep=True)
    pred = d(torch.randn(4, 8, 16))
    sel = torch.tensor([1, 3])
    s = SP.select_slots(pred, sel, 4)
    assert s["box"].shape[0] == 2 and all(a["box"].shape[0] == 2 for a in s["aux"])
    assert torch.equal(s["aux"][0]["presence_logit"], pred["aux"][0]["presence_logit"][sel])
    # the RED arm: the trainer's pre-A9 comprehension passes the list through unselected
    old = {k: (v.index_select(0, sel) if torch.is_tensor(v) and v.shape[:1] == (4,) else v)
           for k, v in pred.items()}
    assert old["aux"][0]["box"].shape[0] == 4 != 2


# ========================================================================================================= #
# R3 -- the IGNORE weight inside the loss                                                                   #
# ========================================================================================================= #
def test_R3_an_unmatched_slot_near_an_ignore_row_gets_zero_presence_weight_and_zero_gradient():
    d = _dec(n_q=4, depth=1, deep=False)
    with torch.no_grad():
        pred = d(torch.randn(1, 8, 16))
    box = torch.tensor([[[10.0, 0.0, 4, 2], [30.0, 5.0, 4, 2], [30.5, 5.5, 4, 2], [50.0, -8.0, 4, 2]]])
    logit = torch.zeros(1, 4, requires_grad=True)
    pred = {**pred, "box": box, "presence_logit": logit}
    tgt = _tgt([(10.0, 0.0, 4, 2, 0), (30.0, 5.2, 4, 2, 0)])        # row 1 will be IGNORE
    vis = {"n_full": torch.tensor([[1000, 1000]], dtype=torch.int32),
           "n_vis": torch.tensor([[900, 20]], dtype=torch.int32), "known": torch.ones(1, 2, dtype=torch.bool)}
    r = SP.refined_slot_losses(pred, tgt, head="agent", presence_loss="focal", vis1=True, vis=vis)
    assert r["n"]["vis1_n_positive"] == 1 and r["n"]["vis1_n_ignore"] == 1
    assert r["n"]["presence_exempt"] == 2                         # slots 1 and 2 sit within 2 m of the ignore row
    r["loss_presence"].backward()
    g = logit.grad[0]
    assert float(g[1]) == 0.0 and float(g[2]) == 0.0
    assert float(g[3]) != 0.0 and float(g[0]) != 0.0
    for k in ("loss_presence", "loss_cls", "loss_centre", "total"):
        assert torch.isfinite(r[k]).all(), k


def test_R3_RED_ARM_without_vis1_the_hidden_row_is_a_target_and_nothing_is_exempt():
    d = _dec(n_q=4, depth=1)
    with torch.no_grad():
        pred = d(torch.randn(1, 8, 16))
    tgt = _tgt([(10.0, 0.0, 4, 2, 0), (30.0, 5.2, 4, 2, 0)])
    r = SP.refined_slot_losses(pred, tgt, head="agent", presence_loss="focal", vis1=False)
    assert r["n"]["presence_exempt"] == 0 and r["n"]["target"] == 2


def test_R3_vis1_without_a_visibility_block_REFUSES():
    d = _dec(n_q=4, depth=1)
    with torch.no_grad():
        pred = d(torch.randn(1, 8, 16))
    with pytest.raises(ValueError, match="never"):
        SP.refined_slot_losses(pred, _tgt([(10, 0, 4, 2, 0)]), head="agent", presence_loss="focal", vis1=True)


# ========================================================================================================= #
# every new loss term: finite and with gradient (the G-LIVE clause, at unit scale)                          #
# ========================================================================================================= #
@pytest.mark.parametrize("head", ["agent", "box3d"])
def test_every_new_term_is_finite_and_reaches_the_head(head):
    d = _dec(n_q=12, depth=3, prior=0.01, deep=True, box3d=(head == "box3d"))
    pred = d(torch.randn(2, 8, 16))
    t = _tgt([(10, 1, 4, 2, 0), (22, -3, 4, 2, 5), (35, 4, 4, 2, 0), (80, 0, 4, 2, 0)], pad=6,
             box3d=(head == "box3d"))
    t = {k: torch.cat([v, v], 0) for k, v in t.items()}
    vis = {"n_full": torch.full((2, 6), 1000, dtype=torch.int32),
           "n_vis": torch.tensor([[900, 30, 800, 900, 0, 0]] * 2, dtype=torch.int32),
           "known": torch.ones(2, 6, dtype=torch.bool)}
    if head == "box3d":
        r = SP.refined_box3d_losses(pred, t, presence_loss="focal", vis1=True, vis=vis)
        assert r["n"]["z"] > 0
    else:
        r = SP.refined_slot_losses(pred, t, head="agent", presence_loss="focal", vis1=True, vis=vis)
    terms = [k for k in r if k.startswith("loss_")]
    assert {"loss_presence", "loss_layer0", "loss_layer1", "loss_layer2", "loss_presence_layer0"} <= set(terms)
    for k in terms:
        assert torch.isfinite(r[k]).all(), k
    r["total"].backward()
    for name, p in (("queries", d.queries), ("head", d.head.weight), ("layer0", d.blocks.layers[0].linear1.weight)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name


# ========================================================================================================= #
# the legacy path is untouched                                                                              #
# ========================================================================================================= #
def test_the_legacy_configuration_runs_the_ORIGINAL_losses():
    from tanitad.refs import refc_agents as RA
    assert SP.refined_is_legacy("bce", False, {"box": 0}) is True
    assert SP.refined_is_legacy("bce", False, {"aux": [1]}) is False
    assert SP.refined_is_legacy("focal", False) is False and SP.refined_is_legacy("bce", True) is False
    d = _dec(n_q=8)
    torch.manual_seed(5)
    pred = d(torch.randn(1, 8, 16))
    tgt = _tgt([(10, 1, 4, 2, 0), (25, -3, 4, 2, 5)])
    cfg = RA.AgentSeamConfig(enable=True, queries=8)
    a = RA.agent_losses(pred, tgt, cfg)
    assert "_refine" not in a and "loss_layer0" not in a
    m = AS.match_slots(pred, RA.visible_target_filter(tgt))
    ref = AS.slot_set_loss(pred, RA.visible_target_filter(tgt), match=m)["total"]
    assert torch.equal(a["total"], ref)


def test_the_refined_bce_machinery_matches_the_original_numerically():
    """Not a bypass: the refined machinery in bce mode reproduces the original total (to float rounding)."""
    from tanitad.refs import refc_agents as RA
    d = _dec(n_q=8)
    torch.manual_seed(6)
    pred = d(torch.randn(1, 8, 16))
    tgt = _tgt([(10, 1, 4, 2, 0), (25, -3, 4, 2, 5), (70, 0, 4, 2, 0)])
    ref = RA.agent_losses(pred, tgt, RA.AgentSeamConfig(enable=True, queries=8))["total"]
    r = SP.refined_slot_losses(pred, tgt, head="agent", presence_loss="bce", vis1=False)
    assert torch.allclose(r["total"], ref, atol=1e-6)


# ========================================================================================================= #
# R4 -- 300 queries                                                                                         #
# ========================================================================================================= #
def test_R4_the_one_spelling_is_300_and_every_refc_default_reads_it():
    from tanitad.models import refcv6_perception_branch as PB
    from tanitad.refs import refc_agents as RA
    assert AS.N_QUERIES_DEFAULT == 300
    assert RA.AgentSeamConfig().queries == 300
    assert PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0).n_queries == 300


def test_R4_300_queries_with_120_targets_drop_nothing_and_100_drops_20():
    rows = [(float(5 + (i % 12) * 4.5), float(-14 + (i // 12) * 3.0), 4.0, 2.0, 0) for i in range(120)]
    tgt = _tgt(rows)
    for n_q, want in ((AS.N_QUERIES_DEFAULT, 0), (100, 20)):
        d = _dec(n_q=n_q, depth=1)
        with torch.no_grad():
            pred = d(torch.randn(1, 8, 16))
        m = AS.match_slots(pred, tgt, presence_cost="focal")
        assert m["n_target"][0] == 120 and m["n_dropped"][0] == want, (n_q, m["n_dropped"])


def test_R4_both_production_heads_are_inside_the_param_band_at_300():
    """The literal parameter arithmetic at the refcv6/refcv7 production geometry (literature pass §1)."""
    lo, hi = AS.PARAM_BAND
    agent = AS.AgentSlotDecoder(2048, 416, n_queries=300, d_model=256, depth=3, n_heads=8)       # stride-32, 13x32
    box = B3.Box3DSlotDecoder(256, 1664 + 480, n_queries=300, d_model=256, depth=3, n_heads=8)  # s16 26x64 + BEV
    assert agent.n_params == 3_822_869 + 51_200 == 3_874_069
    assert box.n_params == 3_806_999 + 51_200 == 3_858_199
    assert lo <= agent.n_params <= hi and lo <= box.n_params <= hi
    # the headroom left under the band for the box head's memory (NEW-2 must stay inside it)
    assert (hi - box.n_params) // 256 == 553


# ========================================================================================================= #
# P0 -- the detection metrics                                                                               #
# ========================================================================================================= #
def _pack(det, gt, ign_idx=(), cls_pred=None):
    """det: [(x, y, p)], gt: [(x, y, cls)]; ign_idx = gt indices that are IGNORE (the rest are POSITIVE)."""
    n = len(det)
    cls = np.array(cls_pred if cls_pred is not None else [0] * n, np.int16)
    pk = {"ep": 0, "logit": np.array([math.log(p / (1 - p)) for (_, _, p) in det], np.float32),
          "xy": np.array([[x, y] for (x, y, _) in det], np.float32), "cls": cls, "cls_corr": cls.copy(),
          "matched": np.zeros(n, bool), "exempt": np.zeros(n, bool), "pair_err": np.zeros(0),
          "gt_xy": np.array([[x, y] for (x, y, _) in gt], np.float32).reshape(-1, 2),
          "gt_cls": np.array([c for (_, _, c) in gt], np.int16),
          "pos": np.array([i not in ign_idx for i in range(len(gt))], bool),
          "ign": np.array([i in ign_idx for i in range(len(gt))], bool),
          "hidden": np.zeros(len(gt), bool)}
    return pk


def test_P0_the_DontCare_scoring_literal():
    """2 positives + 1 IGNORE row; detections: TP (0.9), DontCare on the ignore row (0.8), a duplicate of the TP
    (0.7, FP), a hallucination (0.6, FP), the second positive found late (0.55, TP)."""
    pk = _pack(det=[(10.0, 0.0, 0.9), (30.0, 5.0, 0.8), (10.5, 0.2, 0.7), (45.0, -9.0, 0.6), (20.0, 2.0, 0.55)],
               gt=[(10.0, 0.0, 0), (20.0, 2.0, 0), (30.2, 5.1, 0)], ign_idx=(2,))
    rows = DM.greedy_rows(pk, 2.0)
    assert [r[1] for r in rows] == [1, -1, 0, 0, 1]
    m = DM.summarise([pk], "box3d")
    assert m["eval_box3d_prec@gate"] == pytest.approx(2 / 4)     # 4 counted (DontCare removed)
    assert m["eval_box3d_rec@gate"] == pytest.approx(1.0)
    assert m["eval_box3d_n_conf"] == 4.0 and m["eval_box3d_tp@gate"] == 2.0
    assert m["eval_box3d_conf_ratio"] == pytest.approx(4 / 2)
    assert m["eval_box3d_conf_ratio_alarm"] == 1.0              # 2.0 is outside A10's [0.5, 1.5]
    assert m["eval_box3d_n_pos"] == 2.0 and m["eval_box3d_n_ignore"] == 1.0
    # AP (all-point): P/R after each counted row: (1/1, .5), (1/2, .5), (1/3, .5), (2/4, 1.0)
    assert m["eval_box3d_det_ap2_all_all"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)


def test_P0_RED_ARM_without_DontCare_the_ignore_hit_is_a_false_positive():
    pk = _pack(det=[(10.0, 0.0, 0.9), (30.0, 5.0, 0.8)], gt=[(10.0, 0.0, 0), (30.2, 5.1, 0)], ign_idx=(1,))
    with_dc = DM.summarise([pk], "box3d")["eval_box3d_prec@gate"]
    pk2 = dict(pk)
    pk2["ign"] = np.array([False, False])                   # the pre-A9 scoring: the ignore row does not exist
    pk2["pos"] = np.array([True, False])
    without_dc = DM.summarise([pk2], "box3d")["eval_box3d_prec@gate"]
    assert with_dc == 1.0 and without_dc == 0.5


def test_P0_controls_gt_as_slots_reads_1_and_constant_presence_reads_auroc_half():
    gt = [(8.0, 1.0, 0), (21.0, -3.0, 5), (47.0, 6.0, 0)]
    det = [(x, y, 0.99) for (x, y, _c) in gt] + [(1000.0, 1000.0, 1e-6)] * 5
    pk = _pack(det, gt, cls_pred=[0, 5, 0, 0, 0, 0, 0, 0])
    pk["matched"] = np.array([True, True, True] + [False] * 5)
    m = DM.summarise([pk], "agent")
    for t in DM.DIST_THRESHOLDS_M:
        assert m[f"eval_agent_det_ap{DM.THR_KEY[t]}_all_all"] == pytest.approx(1.0)
    assert m["eval_agent_prec@gate"] == 1.0 and m["eval_agent_rec@gate"] == 1.0
    assert m["eval_agent_conf_ratio"] == 1.0 and m["eval_agent_auroc_matched"] == 1.0
    assert m["eval_agent_conf_ratio_alarm"] == 0.0 and m["eval_agent_ap2m"] == pytest.approx(1.0)
    assert m["eval_agent_cls_acc_tp"] == 1.0 and m["eval_agent_rec@gate_person"] == 1.0
    assert m["eval_agent_det_ap2_person_all"] == pytest.approx(1.0)
    assert m["eval_agent_det_npos_all_0_20"] == 1.0 and m["eval_agent_det_npos_all_40_60"] == 1.0
    assert DM.auroc(np.zeros(10), np.array([1, 0] * 5, bool)) == 0.5


def test_P0_the_key_contract_is_stable():
    keys = DM.metric_keys("box3d")
    assert len(keys) == len(set(keys)) == 18 + 2 * 10 + 4 * 11 * 4 + 4 * 4 + 11 * 4 == 274
    pk = _pack(det=[(10.0, 0.0, 0.9)], gt=[(10.0, 0.0, 0)])
    assert set(DM.summarise([pk], "box3d")) == set(keys)
    assert set(DM.train_row_keys([pk], "box3d")) == set(DM.train_row_key_names("box3d"))
    assert set(DM.calib_keys([pk], "box3d")) == set(DM.calib_key_names("box3d"))
    # the LOGGING_SPEC_BOX §2 names are present verbatim
    for k in ("prec@gate", "rec@gate", "conf_ratio", "ap2m", "auroc_matched", "auroc_objectness", "cls_acc_tp",
              "centre_err_p50"):
        assert f"eval_box3d_{k}" in keys


def test_P0_pooled_never_a_mean_of_batch_ratios():
    """LOGGING_SPEC_BOX test 3: two eval batches with (tp, n_conf) = (1, 1) and (0, 9) -> precision 0.10."""
    a = _pack(det=[(10.0, 0.0, 0.9)], gt=[(10.0, 0.0, 0)])
    b = _pack(det=[(10.0 + 5 * i, 12.0, 0.9) for i in range(9)], gt=[(55.0, -12.0, 0)])
    m = DM.summarise([a, b], "agent")
    assert m["eval_agent_prec@gate"] == pytest.approx(0.10)
    mean_of_ratios = 0.5 * (DM.summarise([a], "agent")["eval_agent_prec@gate"]
                            + DM.summarise([b], "agent")["eval_agent_prec@gate"])
    assert mean_of_ratios == pytest.approx(0.5) != m["eval_agent_prec@gate"]


def test_P0_informative_calibration_gate_and_prior_corrected_argmax():
    pk = _pack(det=[(10.0, 0.0, 0.9), (30.0, 0.0, 0.8), (20.0, 0.0, 0.7)], gt=[(10.0, 0.0, 0), (20.0, 0.0, 9)],
               cls_pred=[0, 0, 0])
    pk["cls_corr"] = np.array([0, 0, 9], np.int16)             # the prior correction fixes the tail class
    m = DM.summarise([pk], "box3d")
    assert m["eval_box3d_cls_acc_tp"] == 0.5 and m["eval_box3d_cls_acc_tp_priorcorr"] == 1.0
    c = DM.calib_keys([pk], "box3d")
    # rows 0.9 TP, 0.8 FP, 0.7 TP: P/R = (1, .5), (.5, .5) -> P = R first at rank 2, gate 0.8
    assert c["eval_box3d_calib_pr_gate"] == pytest.approx(0.8) and c["eval_box3d_calib_prec"] == 0.5


def test_P0_a_class_without_positives_reads_nan_never_zero():
    pk = _pack(det=[(10.0, 0.0, 0.9)], gt=[(10.0, 0.0, 0)])
    m = DM.summarise([pk], "box3d")
    assert math.isnan(m["eval_box3d_det_ap2_bus_all"]) and m["eval_box3d_det_npos_bus_all"] == 0.0
    assert math.isnan(m["eval_box3d_rec@gate_bus"]) and m["eval_box3d_npos_bus"] == 0.0


# ========================================================================================================= #
# G-LIVE presence sanity                                                                                    #
# ========================================================================================================= #
def test_G_LIVE_presence_sanity_literal_and_its_red_arm():
    assert SP.G_LIVE_PRESENCE_MAX_CONFIDENT_FRAC == 0.5
    ok = SP.presence_sanity(torch.full((4, 300), math.log(0.01 / 0.99)))
    assert ok["pass"] and ok["frac_confident"] == 0.0
    sat = torch.full((4, 300), 3.0)                         # refcv6's S-conf: ~all slots over the gate
    bad = SP.presence_sanity(sat)
    assert not bad["pass"] and bad["frac_confident"] == 1.0
    edge = SP.presence_sanity(torch.zeros(2, 10))            # sigma == 0.5 exactly counts as confident (A10: >=)
    assert edge["frac_confident"] == 1.0 and not edge["pass"]
