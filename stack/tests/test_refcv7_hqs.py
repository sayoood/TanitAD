"""refcv7 A14 (SPEC_REFCV7 §19): HEATMAP QUERY SELECTION (HQS) for the box head.

Every guard is proven by MUTATION: the property is asserted AND the defect it exists for is reintroduced and shown to go
RED. Expectations are LITERALS, never an expression over the code under test.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.bev_raster import GRID_DEFAULT, BEVGrid, cell_centers_xy  # noqa: E402
from tanitad.data.semantic_map_gt import CART_SHAPE  # noqa: E402
from tanitad.models import box3d_head as B3  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models import slot_query_select as SQS  # noqa: E402

_T = None


def _trainer():
    global _T
    if _T is None:
        spec = importlib.util.spec_from_file_location("refc_v3_train_for_hqs", str(ROOT / "scripts" / "refc_v3_train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T = mod
    return _T


def _tiny_branch(query_select="learned", n_queries=6):
    cfg = PB.PerceptionBranchConfig(
        w_map=1.0, w_box3d=1.0, d_bev=8,
        bev_cfg=PB.BEVEncoderConfig(d_in=8, d_model=8, d_out=8, dilations=(1,), norm_groups=2),
        n_queries=n_queries, d_model=16, bev_tokens_hw=(2, 2), enforce_param_band=False,
        query_select=query_select)
    return cfg, PB.PerceptionBranch(cfg, d_image=8, image_hw=(4, 8))


def _geom(b, z):
    return torch.zeros(b, z, CART_SHAPE[0], CART_SHAPE[1], 2), torch.ones(b, z, CART_SHAPE[0], CART_SHAPE[1],
                                                                         dtype=torch.bool)


def _targets(B=2, A=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    box = torch.zeros(B, A, 4)
    box[..., 0] = torch.rand(B, A, generator=g) * 40 + 8
    box[..., 1] = torch.rand(B, A, generator=g) * 16 - 8
    box[..., 2], box[..., 3] = 4.5, 1.9
    v = torch.ones(B, A, dtype=torch.bool)
    tgt = {"box": box, "yaw": torch.zeros(B, A), "cls": torch.zeros(B, A, dtype=torch.long), "valid": v,
           "occ": torch.full((B, A), -1.0), "rates": torch.zeros(B, A, 3),
           "rates_mask": torch.zeros(B, A, dtype=torch.bool), "cz": torch.full((B, A), 0.8),
           "h": torch.full((B, A), 1.6), "zh_mask": v.clone()}
    nv = torch.full((B, A), 900, dtype=torch.int32)
    nv[:, 0] = 20                                            # row 0: hidden -> IGNORE
    vis = {"n_full": torch.full((B, A), 1000, dtype=torch.int32), "n_vis": nv, "known": v.clone()}
    return tgt, vis


# --------------------------------------------------------------------------------------------------------- #
# the literals                                                                                              #
# --------------------------------------------------------------------------------------------------------- #
def test_the_A14_literals():
    assert SQS.QUERY_SELECT == ("learned", "heatmap", "learned_ref")
    assert (SQS.HEAT_PRIOR, SQS.ANCHOR_OFFSET_M, SQS.NMS_KERNEL, SQS.HEAT_LOSS_W) == (0.01, 4.0, 3, 1.0)
    assert (SQS.CN_ALPHA, SQS.CN_BETA) == (2.0, 4.0)
    assert PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0).query_select == "learned"
    with pytest.raises(ValueError):
        PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, query_select="anchors")


def test_grid_centres_are_the_rasteriser_s_centres():
    c = SQS.grid_centres(GRID_DEFAULT)
    assert tuple(c.shape) == (120, 64, 2)
    assert c[0, 0].tolist() == [0.25, -15.75] and c[119, 63].tolist() == [59.75, 15.75]
    X, Y = cell_centers_xy(GRID_DEFAULT)
    xy = torch.stack([torch.as_tensor(X), torch.as_tensor(Y)], dim=-1).to(torch.float32)
    assert torch.allclose(c, xy)


# --------------------------------------------------------------------------------------------------------- #
# the target + the loss                                                                                     #
# --------------------------------------------------------------------------------------------------------- #
def test_draw_targets_peak_is_exactly_one_outside_is_counted_and_ignore_is_masked():
    box = torch.tensor([[[10.1, 0.2, 4.5, 1.9], [70.0, 0.0, 4.5, 1.9], [30.2, -5.1, 4.5, 1.9]]])
    pos = torch.tensor([[True, True, False]])
    ign = torch.tensor([[False, False, True]])
    t = SQS.draw_targets(box, pos, ign, GRID_DEFAULT)
    assert t["n_pos"] == 1 and t["n_pos_outside"] == 1                 # x = 70 m is off the 60 m grid
    assert float(t["heat"][0, 20, 32]) == 1.0                           # floor(10.1/0.5), floor(16.2/0.5)
    assert 0.0 < float(t["heat"][0, 21, 32]) < 1.0
    assert bool(t["ignore"][0, 60, 21]) and float(t["heat"][0, 60, 21]) == 0.0
    assert not bool(t["ignore"][0, 20, 32])
    # a positive's Gaussian WINS over an overlapping IGNORE region
    t2 = SQS.draw_targets(torch.tensor([[[10.1, 0.2, 4.5, 1.9], [10.6, 0.2, 4.5, 1.9]]]),
                          torch.tensor([[True, False]]), torch.tensor([[False, True]]), GRID_DEFAULT)
    assert not bool((t2["ignore"] & (t2["heat"] > 0)).any())


def test_centernet_focal_reads_its_analytic_value_and_ignore_contributes_nothing():
    heat = torch.zeros(1, 2, 2)
    heat[0, 0, 0] = 1.0
    logits = torch.zeros(1, 2, 2)                                       # p = 0.5 everywhere
    one = -math.log(0.5) * 0.25                                          # both terms at p = 0.5, heat 0 / 1
    assert abs(float(SQS.centernet_focal(logits, heat)) - 4 * one) < 1e-6
    ign = torch.zeros(1, 2, 2, dtype=torch.bool)
    ign[0, 1, 1] = True
    assert abs(float(SQS.centernet_focal(logits, heat, ign)) - 3 * one) < 1e-6
    # RED arm: without the (1 - heat)^beta penalty reduction a near-peak cell would cost a full negative
    heat[0, 0, 1] = 0.9
    reduced = float(SQS.centernet_focal(logits, heat))
    assert reduced < 4 * one - 0.9 * one


# --------------------------------------------------------------------------------------------------------- #
# selection + the anchored decoder                                                                          #
# --------------------------------------------------------------------------------------------------------- #
def test_select_anchors_is_nms_then_topk_sorted_and_detached():
    grid = BEVGrid(x_fwd_m=2.0, y_half_m=1.0, cell_m=0.5)                # 4 x 4 cells
    lg = torch.full((1, 4, 4), -6.0, requires_grad=True)
    with torch.no_grad():
        lg[0, 1, 1], lg[0, 1, 2], lg[0, 3, 3] = 3.0, 2.0, 1.0            # (1,2) is suppressed by its neighbour
    a, s, idx = SQS.select_anchors(lg, 2, grid)
    assert idx.tolist() == [[1 * 4 + 1, 3 * 4 + 3]]
    assert a.tolist() == [[[0.75, -0.25], [1.75, 0.75]]]
    assert not a.requires_grad and not s.requires_grad
    # RED arm: without the NMS the suppressed neighbour would be the second anchor
    p = torch.sigmoid(lg.detach())
    assert torch.topk(p.reshape(1, -1), 2).indices.tolist() == [[5, 6]]


def _dec(Q=6, depth=3):
    torch.manual_seed(0)
    d = B3.Box3DSlotDecoder(16, 8, n_queries=Q, d_model=16, depth=depth, n_heads=4, enforce_band=False,
                            presence_prior=0.01)
    d.deep_supervision = True
    return d


def test_anchored_decode_keeps_the_centre_within_4m_of_the_anchor_and_every_other_field():
    d = _dec()
    raw = torch.randn(2, 6, d.head.out_features) * 10
    anc = torch.rand(2, 6, 2) * 40
    out = SQS.anchored_decode(d, raw, anc)
    ref = d.decode(raw)
    off = (out["box"][..., :2] - anc).abs()
    assert float(off.max()) <= 4.0
    assert torch.equal(out["box"][..., 2:], ref["box"][..., 2:])
    for k in ("presence_logit", "cls_logits", "yaw_vec"):
        assert torch.equal(out[k], ref[k])
    # RED arm: the unanchored decode is NOT anchor-relative (its centre ignores the anchor entirely)
    assert float((ref["box"][..., :2] - anc).abs().max()) > 4.0


def test_anchored_forward_positions_reach_every_layer_and_anchors_get_no_gradient():
    d = _dec()
    qpos_net = SQS.AnchorPosEmbed(16, GRID_DEFAULT)
    mem = torch.randn(2, 8, 16)
    a1 = torch.rand(2, 6, 2) * 40
    a2 = a1.clone()
    a2[:, :, 0] += 10.0
    o1 = SQS.anchored_forward(d, mem, a1, qpos_net(a1))
    o2 = SQS.anchored_forward(d, mem, a2, qpos_net(a2))
    assert len(o1["aux"]) == 2
    # the query POSITION changes what every slot predicts (not just the centre offset)
    assert not torch.allclose(o1["presence_logit"], o2["presence_logit"])
    # RED arm: zero position -> the presence no longer depends on where the anchor is
    z = torch.zeros(2, 6, 16)
    assert torch.allclose(SQS.anchored_forward(d, mem, a1, z)["presence_logit"],
                          SQS.anchored_forward(d, mem, a2, z)["presence_logit"])
    a = a1.clone().requires_grad_(False)
    q = qpos_net(a)
    SQS.anchored_forward(d, mem, a, q)["presence_logit"].sum().backward()
    assert d.queries.grad is not None and qpos_net.mlp[0].weight.grad is not None
    assert a.grad is None


def test_anchored_forward_equals_the_decoder_s_own_loop_when_the_position_is_zero_and_anchors_are_ignored():
    """The manual pre-norm layer is the SAME computation as nn.TransformerDecoderLayer (qpos = 0): the raw head
    outputs match the decoder's own forward -- so HQS changes ONLY the position and the centre."""
    d = _dec()
    mem = torch.randn(2, 8, 16)
    ref = d(mem)
    out = SQS.anchored_forward(d, mem, torch.zeros(2, 6, 2), torch.zeros(2, 6, 16))
    assert torch.allclose(out["raw"], ref["raw"], atol=1e-6)


# --------------------------------------------------------------------------------------------------------- #
# the branch                                                                                                #
# --------------------------------------------------------------------------------------------------------- #
def test_the_default_branch_builds_nothing_and_never_touches_the_HQS_module(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a 'learned' build reached slot_query_select")
    for name in ("select_anchors", "choose_anchors", "anchored_forward", "anchored_decode", "heat_term",
                 "LearnedRefPoints"):
        monkeypatch.setattr(SQS, name, boom)
    cfg, br = _tiny_branch("learned")
    assert br.box_heat is None and br.box_qpos is None
    assert "box_heat" not in br.param_breakdown()
    g, v = _geom(2, len(cfg.heights_m))
    out = br(torch.randn(2, 8, 4, 8), g, v)
    assert "heat_logits" not in out["box_slots"]
    tgt, vis = _targets()
    row = PB.box3d_loss_row(out["box_slots"], tgt, presence_loss="focal", vis1=True, vis=vis)
    assert "box3d_heat" not in row
    # RED arm: the SAME patch on a heatmap build must fail
    cfg2, br2 = _tiny_branch("heatmap")
    with pytest.raises(AssertionError, match="reached slot_query_select"):
        br2(torch.randn(2, 8, 4, 8), g, v)


def test_the_heatmap_branch_builds_selects_and_puts_its_loss_INSIDE_the_box_term():
    torch.manual_seed(1)
    cfg, br = _tiny_branch("heatmap")
    assert br.box_heat is not None and br.box_qpos is not None
    bd = br.param_breakdown()
    assert bd["box_heat"] > 0 and bd["box_qpos"] > 0 and list(bd)[-1] == "total"
    g, v = _geom(2, len(cfg.heights_m))
    out = br(torch.randn(2, 8, 4, 8), g, v)
    s = out["box_slots"]
    assert tuple(s["heat_logits"].shape) == (2, 120, 64) and tuple(s["anchors"].shape) == (2, 6, 2)
    assert float((s["box"][..., :2] - s["anchors"]).abs().max()) <= 4.0
    tgt, vis = _targets()
    row = PB.box3d_loss_row(s, tgt, presence_loss="focal", vis1=True, vis=vis)
    assert "box3d_heat" in row and float(row["box3d_heat"]) > 0.0
    assert row["box3d_n_heat_pos"] == 6.0                               # 2 frames x (4 rows - 1 IGNORE)
    s0 = {k: v for k, v in s.items() if k != "heat_logits"}
    row0 = PB.box3d_loss_row(s0, tgt, presence_loss="focal", vis1=True, vis=vis)
    assert abs(float(row["loss"]) - float(row0["loss"]) - float(row["box3d_heat"])) < 1e-4
    row["loss"].backward()
    assert br.box_heat.conv2.weight.grad is not None and br.box_qpos.mlp[0].weight.grad is not None


def test_the_heatmap_needs_BEV_features():
    with pytest.raises(ValueError, match="needs BEV features"):
        PB.PerceptionBranch(PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, n_queries=4, d_model=16,
                                                      enforce_param_band=False, query_select="heatmap"),
                            d_image=8, image_hw=(4, 8))


# --------------------------------------------------------------------------------------------------------- #
# the default is bit-identical (a digest recorded on the MAIN variant, like NEW-1's)                         #
# --------------------------------------------------------------------------------------------------------- #
#: MEASURED 2026-09-27 on the MAIN variant (tip cef9709 + LANDING_READY, no HQS code) by ``_learned_digest`` below,
#: dev box, one intra-op thread. The HQS variant with the flag at its default must reproduce it.
MAIN_LEARNED_DIGEST = "7d7c61e618b7bbf34adf72902f637c9bb4b03b12238a382a5b5f3e268ede8e3c"
DIGEST_PLATFORM = ("2.11.0+cu128", "win32")


def _learned_digest(**cfg_kw) -> str:
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        cfg = PB.PerceptionBranchConfig(
            w_map=1.0, w_box3d=1.0, d_bev=8,
            bev_cfg=PB.BEVEncoderConfig(d_in=8, d_model=8, d_out=8, dilations=(1,), norm_groups=2),
            n_queries=6, d_model=16, bev_tokens_hw=(2, 2), enforce_param_band=False,
            presence_loss="focal", presence_prior=0.01, deep_supervision=True, vis1=True, **cfg_kw)
        br = PB.PerceptionBranch(cfg, d_image=8, image_hw=(4, 8))
        g = torch.Generator().manual_seed(1)
        f = torch.randn(2, 8, 4, 8, generator=g)
        gg, vv = _geom(2, len(cfg.heights_m))
        gg = torch.randn(gg.shape, generator=g) * 0.5
        out = br(f, gg, vv)
        tgt, vis = _targets()
        row = PB.box3d_loss_row(out["box_slots"], tgt, presence_loss="focal", vis1=True, vis=vis)
        row["loss"].backward()
        h = hashlib.sha256()
        for k in sorted(out["box_slots"]):
            v = out["box_slots"][k]
            if torch.is_tensor(v):
                h.update(k.encode() + v.detach().contiguous().numpy().tobytes())
        h.update(row["loss"].detach().numpy().tobytes())
        for n, p in sorted(br.named_parameters()):
            if p.grad is not None:
                h.update(n.encode() + p.grad.contiguous().numpy().tobytes())
        return h.hexdigest()
    finally:
        torch.set_num_threads(nt)


def test_the_learned_default_is_BIT_IDENTICAL_to_the_MAIN_variant():
    if (torch.__version__, sys.platform) != DIGEST_PLATFORM:
        pytest.skip(f"the literal digest was recorded on torch {DIGEST_PLATFORM[0]} / {DIGEST_PLATFORM[1]}; this is "
                    f"{torch.__version__} / {sys.platform} -- float kernels differ across builds. The raise-on-touch "
                    f"test above still pins that the default never reaches HQS.")
    assert _learned_digest() == MAIN_LEARNED_DIGEST
    assert _learned_digest(query_select="learned") == MAIN_LEARNED_DIGEST
    # the DISCRIMINATING control: a heatmap build moves the digest
    assert _learned_digest(query_select="heatmap") != MAIN_LEARNED_DIGEST


# --------------------------------------------------------------------------------------------------------- #
# the trainer + G-DVB                                                                                       #
# --------------------------------------------------------------------------------------------------------- #
BASE = ["--arm", "hier", "--size", "tiny", "--out", "X"]


def test_the_flag_defaults_learned_and_the_pin_refuses_the_dead_combinations():
    T = _trainer()
    a = T.build_parser().parse_args(BASE)
    assert a.slot_query_select == "learned"
    with pytest.raises(SystemExit):
        T.build_parser().parse_args(BASE + ["--slot-query-select", "anchors"])
    cfg = T.v3.refc_v3_smoke_config(True)
    for extra, why in ((["--slot-query-select", "heatmap"], "w-box3d 0"),
                       (["--slot-query-select", "heatmap", "--w-box3d", "1.0"], "BEV features")):
        with pytest.raises(SystemExit, match=why.split()[0]):
            T._pin_slot_refine(cfg, T.build_parser().parse_args(BASE + extra))


def test_GDVB_reads_the_built_heatmap_both_ways():
    from tanitad.train import declared_vs_built as dvb
    _, learned = _tiny_branch("learned")
    _, heat = _tiny_branch("heatmap")
    e = dvb.REGISTRY["slot_query_select"]
    ok_l = e.check(types.SimpleNamespace(_perception=learned), types.SimpleNamespace(slot_query_select="learned"))
    ok_h = e.check(types.SimpleNamespace(_perception=heat), types.SimpleNamespace(slot_query_select="heatmap"))
    assert ok_l == [] and ok_h == []
    # RED arms: declared heatmap, built learned -- and the reverse
    bad1 = e.check(types.SimpleNamespace(_perception=learned), types.SimpleNamespace(slot_query_select="heatmap"))
    bad2 = e.check(types.SimpleNamespace(_perception=heat), types.SimpleNamespace(slot_query_select="learned"))
    assert bad1 and bad2 and bad1[0].lever == "--slot-query-select"


# --------------------------------------------------------------------------------------------------------- #
# the harness arms under HQS (A14): memory_zeros blinds the heatmap too; anchors_removed = learned ref points #
# --------------------------------------------------------------------------------------------------------- #
def _harness():
    spec = importlib.util.spec_from_file_location("g_box_overfit_hqs_t", str(ROOT / "scripts" / "g_box_overfit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _M(torch.nn.Module):
    def __init__(self, br):
        super().__init__()
        self._perception = br


def _fake_adapter(br):
    return types.SimpleNamespace(_build_model=lambda: _M(br), _hooks=[], _patched=[], device="cpu")


def test_memory_zeros_also_blinds_the_heatmap_under_HQS():
    G = _harness()
    _, br = _tiny_branch("heatmap")
    fake = _fake_adapter(br)
    G.TrainerAdapter.setup(fake, "memory_zeros", 0)
    try:
        x = torch.randn(2, 8, 120, 64)
        assert torch.equal(br.box_heat(x), br.box_heat(torch.zeros_like(x)))
        assert float(br.box_mem(torch.randn(2, 8, 4, 8), torch.randn(2, 8, 120, 64)).abs().max()) == 0.0
    finally:
        G.TrainerAdapter.teardown(fake)
    # RED arm (teardown restored): the heatmap sees its input again
    x = torch.randn(2, 8, 120, 64)
    assert not torch.equal(br.box_heat(x), br.box_heat(torch.zeros_like(x)))


def test_anchors_removed_swaps_the_heatmap_anchors_for_learned_reference_points_and_restores():
    G = _harness()
    cfg, br = _tiny_branch("heatmap")
    fake = _fake_adapter(br)
    orig = SQS.choose_anchors
    G.TrainerAdapter.setup(fake, "anchors_removed", 0)
    try:
        assert isinstance(br.box_hqs_refpts, torch.nn.Parameter) and tuple(br.box_hqs_refpts.shape) == (6, 2)
        g, v = _geom(2, len(cfg.heights_m))
        out = br(torch.randn(2, 8, 4, 8), g, v)
        assert torch.equal(out["box_slots"]["anchors"], br.box_hqs_refpts[None].expand(2, -1, -1))
        out["box_slots"]["box"].sum().backward()
        assert br.box_hqs_refpts.grad is not None                   # LEARNED reference points
    finally:
        G.TrainerAdapter.teardown(fake)
    assert SQS.choose_anchors is orig
    # and on a learned build the red arm refuses (it is HQS's red arm only)
    _, br2 = _tiny_branch("learned")
    with pytest.raises(SystemExit):
        G.TrainerAdapter.setup(_fake_adapter(br2), "anchors_removed", 0)


# --------------------------------------------------------------------------------------------------------- #
# learned_ref: the static learned anchors (DAB-DETR) -- the bench's best arm                                 #
# --------------------------------------------------------------------------------------------------------- #
def test_learned_ref_builds_trainable_anchors_no_heatmap_and_no_BEV_requirement():
    torch.manual_seed(0)
    cfg = PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, n_queries=6, d_model=16, enforce_param_band=False,
                                    query_select="learned_ref")
    br = PB.PerceptionBranch(cfg, d_image=8, image_hw=(4, 8))
    assert br.box_heat is None and br.box_refpts is not None and br.box_qpos is not None
    bd = br.param_breakdown()
    assert bd["box_refpts"] == 12 and "box_heat" not in bd and list(bd)[-1] == "total"
    xy = br.box_refpts.xy.detach()
    assert float(xy[:, 0].min()) >= 0.0 and float(xy[:, 0].max()) <= 60.0 and float(xy[:, 1].abs().max()) <= 16.0
    out = br(torch.randn(2, 8, 4, 8))
    s = out["box_slots"]
    assert torch.equal(s["anchors"], br.box_refpts.xy[None].expand(2, -1, -1))
    assert "heat_logits" not in s and float((s["box"][..., :2] - s["anchors"]).abs().max()) <= 4.0
    tgt, vis = _targets()
    row = PB.box3d_loss_row(s, tgt, presence_loss="focal", vis1=True, vis=vis)
    assert "box3d_heat" not in row
    row["loss"].backward()
    assert br.box_refpts.xy.grad is not None and float(br.box_refpts.xy.grad.abs().sum()) > 0.0
    # the init does not consume the global RNG stream (a learned build and a learned_ref build draw the same
    # decoder init): its generator is its own
    torch.manual_seed(5)
    a = torch.rand(3)
    torch.manual_seed(5)
    SQS.LearnedRefPoints(4, GRID_DEFAULT)
    assert torch.equal(torch.rand(3), a)


def test_GDVB_and_the_pin_know_learned_ref():
    from tanitad.train import declared_vs_built as dvb
    _, learned = _tiny_branch("learned")
    _, lref = _tiny_branch("learned_ref")
    _, heat = _tiny_branch("heatmap")
    e = dvb.REGISTRY["slot_query_select"]
    ns = types.SimpleNamespace
    assert e.check(ns(_perception=lref), ns(slot_query_select="learned_ref")) == []
    # RED arms: a learned_ref declaration on a learned or a heatmap build, and the reverse
    assert e.check(ns(_perception=learned), ns(slot_query_select="learned_ref"))
    assert e.check(ns(_perception=heat), ns(slot_query_select="learned_ref"))
    assert e.check(ns(_perception=lref), ns(slot_query_select="heatmap"))
    T = _trainer()
    with pytest.raises(SystemExit, match="w-box3d 0"):
        T._pin_slot_refine(T.v3.refc_v3_smoke_config(True),
                           T.build_parser().parse_args(BASE + ["--slot-query-select", "learned_ref"]))
