"""refcv9 B3 (REFCV9_DESIGN 2.4 / R4 / R11): quality-aware presence, query denoising, sqrt class weights, in-loss BEV-NMS.
CPU, tiny decoders. Each claim has a GREEN check and a RED arm (the same check under the mutation must FAIL), per
TanitAD_ValidateAIDesign. The reachability class (CLAUDE.md 'built, tested and unreachable') is traced argparse -> model -> loss ->
backward at the end, with a control that must read EXACTLY 0.0."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.bev_raster import GRID_DEFAULT  # noqa: E402
from tanitad.models import box_b3 as B  # noqa: E402
from tanitad.models import slot_presence as SP  # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES, match_slots  # noqa: E402
from tanitad.models.box3d_head import Box3DSlotDecoder, zh_targets  # noqa: E402
from tanitad.models.slot_query_select import AnchorPosEmbed, LearnedRefPoints, anchored_forward  # noqa: E402
from tanitad.train import box_b3_flag as BF  # noqa: E402

D = 32


def _dec(depth=3, nq=12, deep=True):
    torch.manual_seed(0)
    d = Box3DSlotDecoder(d_memory=D, n_memory=20, n_queries=nq, d_model=D, depth=depth, n_heads=4, enforce_band=False,
                         presence_prior=0.01)
    d.deep_supervision = deep
    return d


def _gt(b=2, a=5):
    box = torch.zeros(b, a, 4)
    box[..., 0] = torch.tensor([5.0, 10.0, 20.0, 30.0, 40.0])
    box[..., 1] = torch.tensor([0.0, 2.0, -3.0, 1.0, 4.0])
    box[..., 2], box[..., 3] = 4.0, 2.0
    valid = torch.ones(b, a, dtype=torch.bool)
    valid[1, 3:] = False
    return box, torch.zeros(b, a), torch.zeros(b, a, dtype=torch.long), valid


def _pack(qpos_module=None, seed=1):
    box, yaw, cls, valid = _gt()
    torch.manual_seed(seed)
    return B.build_dn_pack(box, yaw, cls, valid, torch.ones(2, dtype=torch.bool), x_fwd_m=60.0, y_half_m=16.0, qpos_module=qpos_module)


def _tgt():
    box, yaw, cls, valid = _gt()
    t = {"box": box, "yaw": yaw, "cls": cls, "valid": valid, "occ": torch.full_like(yaw, -1.0), "rates": torch.zeros(2, 5, 3),
         "rates_mask": torch.zeros(2, 5, dtype=torch.bool)}
    return zh_targets(t)


# ------------------------------------------------------------------ quality-aware presence (known answers + RED) ---------------------
def test_quality_target_known_answers():
    xy, yv = torch.zeros(1, 2), torch.tensor([[0.0, 1.0]])                            # yaw 0 = (sin 0, cos 0)
    assert float(B.quality_target(xy, yv, xy, torch.zeros(1))) == pytest.approx(1.0)
    assert float(B.quality_target(xy, yv, torch.tensor([[1.0, 0.0]]), torch.zeros(1))) == pytest.approx(math.exp(-1.0))
    assert float(B.quality_target(xy, yv, xy, torch.full((1,), math.pi))) == pytest.approx(0.0, abs=1e-6)     # flipped box
    assert float(B.quality_target(xy, yv, xy, torch.full((1,), math.pi / 2))) == pytest.approx(0.5, abs=1e-6)


def test_soft_target_optimum_is_the_target_and_hard_targets_are_the_old_focal():
    z = torch.linspace(-6, 6, 4001)
    for t in (0.2, 0.6):
        el = B.qfl_elementwise(z, torch.full_like(z, t))
        assert abs(float(torch.sigmoid(z[int(el.argmin())])) - t) < 5e-3
    zz = torch.randn(64)
    for t in (0.0, 1.0):
        assert torch.allclose(B.qfl_elementwise(zz, torch.full_like(zz, t)), SP.sigmoid_focal_elementwise(zz, torch.full_like(zz, t)), atol=1e-6)


def test_presence_term_b3_without_quality_is_the_tip_focal_term_bit_for_bit():
    torch.manual_seed(3)
    logit = torch.randn(2, 12)
    m = {"rows": [torch.tensor([1, 4]), torch.tensor([7])], "cols": [torch.tensor([0, 1]), torch.tensor([0])]}
    a, b = B.presence_term_b3(logit, m, None), SP.presence_term(logit, m, mode="focal")
    assert torch.allclose(a["loss"], b["loss"], rtol=1e-6) and a["n_matched"] == b["n_matched"]


def _quality_loss_check() -> bool:
    """A matched slot must carry the target exp(-d) * yawness, not 1: the loss equals the hand-computed quality focal term."""
    tgt = _tgt()
    dec = _dec()
    torch.manual_seed(5)
    o = dec(torch.randn(2, 20, D))
    r = B.refined_box3d_losses_b3(o, tgt, B.BoxB3Config(quality=True), presence_loss="focal", vis1=False, weights={})
    r0 = B.refined_box3d_losses_b3(o, tgt, B.BoxB3Config(), presence_loss="focal", vis1=False, weights={})
    p = o["presence_logit"]
    m = r["match"]
    t = torch.zeros_like(p)
    for b in range(2):
        rr, cc = m["rows"][b], m["cols"][b]
        d = (o["box"][b][rr][:, :2] - tgt["box"][b][cc][:, :2]).norm(dim=-1)
        t[b, rr] = (torch.exp(-d) * B.yawness(o["yaw_vec"][b][rr], tgt["yaw"][b][cc])).detach()
    want = B.qfl_elementwise(p, t).sum() / max(int(sum(x.numel() for x in m["rows"])), 1)
    return bool(torch.allclose(r["loss_presence"], want, rtol=1e-5)) and not torch.equal(r["loss_presence"], r0["loss_presence"])


def test_quality_green():
    assert _quality_loss_check()


def test_quality_red_target_replaced_by_one_FAILS(monkeypatch):
    monkeypatch.setattr(B, "quality_target", lambda pxy, pyv, txy, tyaw: torch.ones(pxy.shape[0], dtype=pxy.dtype))
    assert not _quality_loss_check()


# ------------------------------------------------------------------ DN: shapes, mask, leak, eval identity ----------------------------
def test_dn_pack_layout_3_groups_2_negatives_per_gt():
    pk = _pack()
    assert (pk["n_groups"], pk["group_size"], pk["gmax"]) == (3, 3 * 5, 5)
    assert int(pk["is_pos"][0].sum()) == 3 * 5 and int(pk["is_neg"][0].sum()) == 3 * 5 * 2   # per GT: 1 positive + 2 negatives, x 3 groups
    assert int(pk["is_pos"][1].sum()) == 3 * 3                                        # window 1 has 3 real GT
    d = (pk["anchors"] - pk["t_box"][..., :2]).norm(dim=-1)
    assert float(d[pk["is_neg"]].min()) >= B.DN_NEG_MIN_M - 1e-4                      # negatives sit outside the 2 m AP tolerance


def test_dn_mask_structure():
    n, g, s = 4, 3, 6
    m = B.dn_attn_mask(n, g, s)
    assert not m[:n, :n].any() and m[:n, n:].all() and m[n:, :n].all()               # matching <-> DN blocked both ways
    for i in range(g):
        a = n + i * s
        assert not m[a:a + s, a:a + s].any()
        for j in range(g):
            if j != i:
                assert m[a:a + s, n + j * s:n + j * s + s].all()                      # groups do not see each other


def _leak_check(anchored=False) -> dict:
    """Perturb the GT-derived DN content -> the MATCHING outputs must be bit-identical (forward leak); perturb the matching queries ->
    the DN outputs must be bit-identical (back leak)."""
    dec = _dec()
    dec.enable_b3(dn=True)
    dec.train()
    torch.manual_seed(7)
    mem = torch.randn(2, 20, D)
    ref = LearnedRefPoints(12, GRID_DEFAULT)
    qp = AnchorPosEmbed(D, GRID_DEFAULT)

    def run(pk):
        dec.set_dn(pk)
        return anchored_forward(dec, mem, ref(2), qp(ref(2))) if anchored else dec(mem)

    mk = qp if anchored else None
    base = run(_pack(mk, seed=1))
    other = run(_pack(mk, seed=2))                                                    # a different noised GT
    fwd_leak = not all(torch.equal(base[k], other[k]) for k in ("raw", "presence_logit", "box"))
    saved = dec.queries.data.clone()
    dec.queries.data += torch.randn_like(saved)
    moved = run(_pack(mk, seed=1))
    dec.queries.data.copy_(saved)
    back_leak = not torch.equal(base["_dn_out"]["layers"][-1]["raw"], moved["_dn_out"]["layers"][-1]["raw"])
    return {"fwd_leak": fwd_leak, "back_leak": back_leak,
            "control_dn_differs": not torch.equal(base["_dn_out"]["layers"][-1]["raw"], other["_dn_out"]["layers"][-1]["raw"])}


@pytest.mark.parametrize("anchored", [False, True])
def test_dn_no_leak_either_way_GREEN(anchored):
    assert _leak_check(anchored) == {"fwd_leak": False, "back_leak": False, "control_dn_differs": True}


@pytest.mark.parametrize("anchored", [False, True])
def test_dn_red_mask_removed_leak_test_FAILS(monkeypatch, anchored):
    monkeypatch.setattr(B, "dn_attn_mask", lambda n, g, s, device=None: torch.zeros(n + g * s, n + g * s, dtype=torch.bool, device=device))
    r = _leak_check(anchored)
    assert r["fwd_leak"] and r["back_leak"]                                           # the test that guards the mask DETECTS its absence


def test_eval_forward_is_bit_identical_with_dn_on_and_off():
    off = _dec()
    on = _dec()
    on.enable_b3(dn=True, nms_tau=0.1)
    on.load_state_dict(off.state_dict(), strict=False)
    off.eval(), on.eval()
    mem = torch.randn(2, 20, D)
    on.set_dn(_pack())                                                                # even a pending pack must be discarded in eval
    a, b = off(mem), on(mem)
    assert "_dn_out" not in b and on._dn_pending is None
    assert all(torch.equal(a[k], b[k]) for k in ("raw", "presence_logit", "box", "cls_logits"))
    assert all(torch.equal(x["raw"], y["raw"]) for x, y in zip(a["aux"], b["aux"]))
    ref, qp = LearnedRefPoints(12, GRID_DEFAULT), AnchorPosEmbed(D, GRID_DEFAULT)
    on.set_dn(_pack(qp))
    assert torch.equal(anchored_forward(off, mem, ref(2), qp(ref(2)))["raw"], anchored_forward(on, mem, ref(2), qp(ref(2)))["raw"])


def test_dn_only_adds_its_own_parameters_and_off_adds_none():
    off, on = _dec(), _dec()
    on.enable_b3(dn=True)
    assert set(on.state_dict()) - set(off.state_dict()) == {"dn_embed.weight", "dn_embed.bias", "dn_cls_embed.weight"}
    off2 = _dec()
    off2.enable_b3(dn=False, nms_tau=None)
    assert list(off2.state_dict()) == list(off.state_dict())


# ------------------------------------------------------------------ gradients: every new parameter live + controls exactly 0.0 ------
def _dn_grads(mask: bool, monkeypatch=None):
    if not mask:
        monkeypatch.setattr(B, "dn_attn_mask", lambda n, g, s, device=None: torch.zeros(n + g * s, n + g * s, dtype=torch.bool, device=device))
    dec = _dec()
    dec.enable_b3(dn=True)
    dec.train()
    ref, qp = LearnedRefPoints(12, GRID_DEFAULT), AnchorPosEmbed(D, GRID_DEFAULT)
    dec.set_dn(_pack(qp))
    o = anchored_forward(dec, torch.randn(2, 20, D), ref(2), qp(ref(2)))
    dl = B.dn_loss(o["_dn_out"], B.BoxB3Config())
    dl["loss"].backward()
    g = lambda p: 0.0 if p.grad is None else float(p.grad.abs().sum())
    return {"dn_embed": g(dec.dn_embed.weight), "dn_cls_embed": g(dec.dn_cls_embed.weight), "qpos": g(qp.mlp[0].weight),
            "head": g(dec.head.weight), "queries_control": g(dec.queries), "ref_control": g(ref.xy), "loss": float(dl["loss"])}


def test_dn_gradients_reach_every_new_parameter_and_the_matching_queries_read_exactly_zero():
    g = _dn_grads(mask=True)
    assert math.isfinite(g["loss"]) and g["loss"] > 0
    assert g["dn_embed"] > 0 and g["dn_cls_embed"] > 0 and g["qpos"] > 0 and g["head"] > 0
    assert g["queries_control"] == 0.0 and g["ref_control"] == 0.0                    # the DN loss cannot reach the matching queries


def test_dn_red_mask_removed_the_zero_control_reads_nonzero(monkeypatch):
    assert _dn_grads(mask=False, monkeypatch=monkeypatch)["queries_control"] > 0.0    # the control reads 0.0 only BECAUSE of the mask


# ------------------------------------------------------------------ sqrt class weights ------------------------------------------------
def _sqrt_check(fn) -> bool:
    s = fn(torch.tensor([1299.0, 100.0, 9.0, 1.0]))
    return bool(abs(float(s.mean()) - 1.0) < 1e-6 and abs(float(s[0] / s[3]) - math.sqrt(1299.0)) < 1e-3)


def test_sqrt_weights_green_and_red_old_weights_FAIL():
    assert _sqrt_check(B.sqrt_temper)
    assert not _sqrt_check(lambda w: w / w.mean())                                    # the pure 1,299 : 1 vector


def test_sqrt_is_what_the_loss_receives():
    m = type("M", (), {})()
    m._cls_class_weight = torch.tensor([1299.0, 100.0, 9.0, 1.0])
    m._r9_box_b3 = B.BoxB3Config(class_sqrt=True)
    m._r9_box_cls_weight = B.sqrt_temper(m._cls_class_weight)
    seen = {}

    class _P:
        @staticmethod
        def box3d_loss_row(slots, tgt, **kw):
            seen.update(kw)
            return {"loss": 0}
    BF.box3d_row(m, _P, {}, {}, cls_class_weight=m._cls_class_weight, visible_filter=True, presence_loss="focal", vis1=False, vis=None)
    assert torch.equal(seen["cls_class_weight"], m._r9_box_cls_weight) and not torch.equal(seen["cls_class_weight"], m._cls_class_weight)
    m._r9_box_b3 = None                                                               # OFF: the trainer's own vector, untouched
    BF.box3d_row(m, _P, {}, {}, cls_class_weight=m._cls_class_weight, visible_filter=True, presence_loss="focal", vis1=False, vis=None)
    assert torch.equal(seen["cls_class_weight"], m._cls_class_weight)


# ------------------------------------------------------------------ BEV-NMS (B3') ----------------------------------------------------
def _pred(centres, probs):
    n = len(centres)
    box = torch.zeros(1, n, 4)
    for i, (x, y) in enumerate(centres):
        box[0, i] = torch.tensor([x, y, 4.0, 2.0])
    return {"box": box, "yaw": torch.zeros(1, n), "presence_logit": torch.logit(torch.tensor([probs]))}


def test_bev_nms_known_answers():
    k = B.bev_nms_keep(_pred([(10, 0), (10.5, 0), (30, 0), (30, 8)], [0.9, 0.8, 0.7, 0.6]), tau=0.1)
    assert k.tolist() == [[True, False, True, True]]                                  # the lower-scored duplicate dies, the disjoint pair lives
    assert B.bev_nms_keep(_pred([(10, 0), (10.5, 0)], [0.9, 0.01]), tau=0.1).tolist() == [[True, True]]   # under the floor: never suppressed


def test_inloss_nms_match_is_over_survivors_and_runs_through_the_decoder():
    dec = _dec(depth=3, deep=True)
    dec.enable_b3(dn=False, nms_tau=0.1)
    dec.train()
    o = dec(torch.randn(2, 20, D))                                                    # routed through run_decoder (nms only)
    assert "_dn_out" not in o and len(o["aux"]) == 2
    tgt = _tgt()
    r = B.refined_box3d_losses_b3(o, tgt, B.BoxB3Config(inloss_nms=True), presence_loss="focal", vis1=False, weights={})
    assert math.isfinite(float(r["total"])) and "nms_suppressed" in r["n"]
    p = {**o, "presence_logit": torch.full_like(o["presence_logit"], -3.0), "box": o["box"].clone()}
    p["presence_logit"][:, 0], p["presence_logit"][:, 1] = 2.0, 1.0
    p["box"][:, 1, :2] = p["box"][:, 0, :2]                                           # slot 1 duplicates slot 0 at a lower score
    keep = B.bev_nms_keep(p)
    assert not bool(keep[:, 1].any()) and bool(keep[:, 0].all())
    rows = match_slots(B._match_view(p, keep), tgt, presence_cost="focal")["rows"]
    assert all(1 not in r_.tolist() for r_ in rows)                                   # a suppressed slot is never matched while a survivor is free


# ------------------------------------------------------------------ OFF path: nothing changes -----------------------------------------
def test_off_dispatch_is_the_tip_call_bit_for_bit():
    from tanitad.models import refcv6_perception_branch as perc
    dec = _dec()
    dec.train()
    torch.manual_seed(9)
    o = dec(torch.randn(2, 20, D))
    tgt = _tgt()
    m = type("M", (), {})()
    a = perc.box3d_loss_row(o, tgt, cls_class_weight=None, visible_filter=True, presence_loss="focal", vis1=False, vis=None)
    b = BF.box3d_row(m, perc, o, tgt, cls_class_weight=None, visible_filter=True, presence_loss="focal", vis1=False, vis=None)
    assert a.keys() == b.keys() and all(torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])) for k in a)
    assert BF.prepare(m, {}, "cpu") is None and not BF.is_any(type("A", (), {})())


def test_enable_b3_off_leaves_the_decoder_identical_and_the_train_forward_unchanged():
    a, b = _dec(), _dec()
    b.enable_b3(dn=False, nms_tau=None)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    a.train(), b.train()
    mem = torch.randn(2, 20, D)
    assert torch.equal(a(mem)["raw"], b(mem)["raw"])


# ------------------------------------------------------------------ flag module: refusals, G-DVB ---------------------------------------
class _A:
    def __init__(self, **k):
        self.__dict__.update(dict(w_box3d=1.0, slot_presence_loss="focal", slot_query_select="learned_ref", agent_cls_weight="b1",
                                  box3d_visible_filter=True, r9_box_quality=False, r9_box_dn=False, r9_box_class_sqrt=False,
                                  r9_box_inloss_nms=False))
        self.__dict__.update(k)


def test_pin_args_refusals():
    BF.pin_args(_A())                                                                 # all OFF: nothing checked
    BF.pin_args(_A(r9_box_dn=True))
    for bad in (dict(w_box3d=0.0), dict(slot_presence_loss="bce"), dict(slot_query_select="heatmap"), dict(box3d_visible_filter=False),
                dict(r9_box_class_sqrt=True, agent_cls_weight="off")):
        with pytest.raises(SystemExit):
            BF.pin_args(_A(r9_box_quality=True, **bad))


def test_gdvb_each_flag_has_a_built_lever_and_the_check_catches_a_lie():
    from tanitad.train import declared_vs_built as dvb
    BF.register_dvb_levers(dvb.register)
    for d in BF.DESTS.values():
        assert dvb.REGISTRY[d].kind == "built"
    model = type("M", (), {})()
    model._perception = type("Br", (), {})()
    model._perception.box_dec = _dec()
    model._cls_class_weight = torch.tensor([9.0, 1.0, 4.0])
    args = _A(r9_box_quality=True, r9_box_dn=True, r9_box_class_sqrt=True, r9_box_inloss_nms=True)
    BF.attach(model, args)
    for d in BF.DESTS.values():
        assert dvb.REGISTRY[d].check(model, args) == [], d
    bare = type("M2", (), {"_perception": type("Br", (), {})(), "_cls_class_weight": None})()
    for d in BF.DESTS.values():                                                       # RED: declare what was not built
        assert dvb.REGISTRY[d].check(bare, args) != [], d
    assert BF.stamp(model, args)["flags"] == {v: True for v in BF.FLAGS.values()}
    assert BF.stamp(type("M3", (), {})(), _A()) is None


# ------------------------------------------------------------------ reachability: argparse -> model -> loss -> backward -----------------
def test_perception_branch_trace_dn_reaches_loss_and_every_new_parameter():
    from tanitad.models import refcv6_perception_branch as perc
    cfg = perc.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, d_model=D, n_queries=12, enforce_param_band=False, presence_loss="focal",
                                      presence_prior=0.01, deep_supervision=True, vis1=False, query_select="learned_ref",
                                      bev_source="s16_lift")
    torch.manual_seed(0)
    br = perc.PerceptionBranch(cfg, d_image=16, image_hw=(4, 5))
    model = type("M", (), {})()
    model._perception = br
    model._cls_class_weight = torch.ones(len(AGENT_CLASSES))
    model._vis1 = False
    model.training = True
    BF.attach(model, _A(r9_box_quality=True, r9_box_dn=True, r9_box_class_sqrt=True, r9_box_inloss_nms=True))
    br.train()
    fm = torch.randn(2, 16, 4, 5, requires_grad=True)
    box, yaw, cls, valid = _gt()
    BF.prepare(model, {"agent_box": box, "agent_yaw": yaw, "agent_cls": cls, "agent_valid": valid}, "cpu")   # hop 1: the DN pack
    out = br(fm)                                                                      # hop 2: the forward consumes it (anchored path)
    assert "_dn_out" in out["box_slots"]
    row = BF.box3d_row(model, perc, out["box_slots"], _tgt(), cls_class_weight=model._cls_class_weight, visible_filter=True,
                       presence_loss="focal", vis1=False, vis=None, sel=torch.arange(2))      # hop 3: the loss
    assert row["box3d_n_dn_pos"] > 0 and torch.is_tensor(row["box3d_dn"]) and row["box3d_n_nms_suppressed"] >= 0
    row["loss"].backward()                                                            # hop 4: backward
    g = lambda p: 0.0 if p.grad is None else float(p.grad.abs().sum())
    for name, p in (("dn_embed", br.box_dec.dn_embed.weight), ("dn_cls_embed", br.box_dec.dn_cls_embed.weight),
                    ("box_qpos", br.box_qpos.mlp[0].weight), ("box_refpts", br.box_refpts.xy), ("queries", br.box_dec.queries)):
        assert g(p) > 0.0, name
    assert float(fm.grad.abs().sum()) > 0.0


TRAINER = ROOT / "scripts" / "refc_v3_train.py"


def _trainer_reaches(src: str) -> dict:
    return {"import": src.count("import box_b3_flag as _bbf"), "register": src.count("_bbf.register_dvb_levers(_dvb.register)"),
            "pin": src.count("_bbf.pin_args(args)"), "attach": src.count("_bbf.attach(model, args)"),
            "prepare": src.count("_bbf.prepare(model, batch, device)"), "dispatch": src.count("_bbf.box3d_row("),
            "stamp": src.count("_bbf.stamp(model, args)"), "flags": sum(src.count('"%s"' % f) for f in BF.FLAGS.values()),
            "prepare_before_forward": int(bool(re.search(r"_bbf\.prepare\(model, batch, device\)[^\n]*\n(?:\s*_bc5\.stage\(model, batch, device\)[^\n]*\n)?\s*out = model\(frames, perception_grid", src))),   # C5-PFQ: its camera stage may sit between
            "no_old_call": src.count("_perc.box3d_loss_row(")}


def _hops_ok(h: dict) -> bool:
    return (h["import"] == 1 and h["register"] == 1 and h["pin"] == 1 and h["attach"] == 1 and h["prepare"] == 2 and h["dispatch"] == 2
            and h["stamp"] == 1 and h["flags"] == 4 and h["prepare_before_forward"] == 1 and h["no_old_call"] == 0)


def test_trainer_reachability_green_and_each_mutation_FAILS():
    src = TRAINER.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert _hops_ok(_trainer_reaches(src))
    for needle in ("_bbf.attach(model, args)", "_bbf.pin_args(args)", "_bbf.box3d_row(", "_bbf.stamp(model, args)",
                   "_bbf.prepare(model, batch, device)"):
        assert not _hops_ok(_trainer_reaches(src.replace(needle, "pass  # " + needle[5:], 1))), needle


def test_trainer_parser_declares_the_four_flags_default_off():
    src = TRAINER.read_bytes().decode("utf-8")
    for f in BF.FLAGS.values():
        assert re.search(r'add_argument\("%s", action="store_true"' % re.escape(f), src), f
