"""refcv9 C5-REST (arm B4's box-head levers): the D2 size cap (lever 1) and ``--r9-box-layers N`` (lever 2).

CPU, tiny decoders except where a production-size count is the point.  Per TanitAD_ValidateAIDesign every claim has a GREEN check and a RED
arm (the same check under the mutation must FAIL); per CLAUDE.md ("built, tested and unreachable from its caller") the reachability chain is
traced argparse -> namespace -> model attribute -> forward -> loss -> backward, with a control that must read EXACTLY 0.0.

Levers 3 (``--r9-box-s8-horizon``) and 4 (``--r9-box-perframe-queries``) are NOT in this file yet: see HANDOFF.md of the hold.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import box3d_head as BH  # noqa: E402
from tanitad.models.agent_slots import PARAM_BAND, AGENT_CLASSES, AgentSlotDecoder  # noqa: E402
from tanitad.models.box3d_head import BOX3D_PARAM_CAP, Box3DMemory, Box3DSlotDecoder, zh_targets  # noqa: E402
from tanitad.train import box_b3_flag as BF  # noqa: E402
from tanitad.train import box_c5_flag as C5  # noqa: E402

D = 32
TRAINER = ROOT / "scripts" / "refc_v3_train.py"
LOADER = ROOT / "tanitad" / "eval" / "refcv7_loader.py"


def _A(**kw):
    base = dict(r9_box_layers=0, w_box3d=1.0, r9_box_quality=False, r9_box_dn=False, r9_box_class_sqrt=False, r9_box_inloss_nms=False,
                slot_presence_loss="focal", slot_query_select="learned_ref", box3d_visible_filter=True, agent_cls_weight="b1",
                r8_train_only_map=False, r9_train_only="")
    base.update(kw)
    return SimpleNamespace(**base)


def _dec(depth=3, nq=12, deep=True, enforce=False):
    torch.manual_seed(0)
    d = Box3DSlotDecoder(d_memory=D, n_memory=20, n_queries=nq, d_model=D, depth=depth, n_heads=4, enforce_band=enforce, presence_prior=0.01)
    d.deep_supervision = deep
    return d


def _sd_equal(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def _prod(depth, **kw):
    mem = Box3DMemory(d_image=1024, d_bev=96, d_model=256, image_hw=(26, 64))
    torch.manual_seed(0)
    return Box3DSlotDecoder(d_memory=256, n_memory=mem.n_tokens, n_queries=300, d_model=256, depth=depth, presence_prior=0.01, **kw)


# =============================================================== lever 1: the D2 size cap ============================================
def test_d2_cap_constant_is_the_pi_choice_and_the_agent_slots_keep_their_band():
    assert BOX3D_PARAM_CAP == 12_000_000
    assert PARAM_BAND == (2_000_000, 4_000_000)                       # the agent slots KEEP the band
    with pytest.raises(ValueError, match="band"):                     # ... and still refuse outside it (toy size is far below 2 M)
        AgentSlotDecoder(d_memory=D, n_memory=20, n_queries=4, d_model=D, depth=1, n_heads=4, enforce_band=True)


def test_existing_arm_param_counts_are_unchanged():
    """MEASURED on the raw tip 784687f (before any edit): the production box decoder (d256, 300 queries, resnet101 416x1024 memory = 1,664 s16
    + 480 BEV tokens) is 3,858,199 parameters -- the design's number, and in the OLD band.  The cap must admit it and change nothing."""
    d = _prod(3, enforce_band=True)
    assert d.n_params == 3_858_199 and d.n_trainable == 3_858_199
    assert PARAM_BAND[0] <= d.n_params <= PARAM_BAND[1]


def test_cap_admits_what_the_old_band_refused_and_refuses_above_12M():
    d6 = _prod(6, enforce_band=True)                                  # 7,018,519: ABOVE the old 4 M ceiling, now admitted
    assert d6.n_params == 7_018_519 and d6.n_params > PARAM_BAND[1] and d6.n_params <= BOX3D_PARAM_CAP
    tiny = _dec(1, enforce=True)                                      # far BELOW the old 2 M floor: no floor any more
    assert tiny.n_params < PARAM_BAND[0]
    with pytest.raises(ValueError, match="D2 cap"):                   # 14 layers ~ 14.0 M > 12 M
        _prod(14, enforce_band=True)
    _prod(14, enforce_band=False)                                     # shape tests still bypass it by name


def test_cap_red_arm_a_lower_cap_refuses_b4(monkeypatch):
    """The cap is what decides: tighten it to the old 4 M ceiling and the 6-layer decoder is refused (RED for 'B4 builds')."""
    monkeypatch.setattr(BH, "BOX3D_PARAM_CAP", 4_000_000)
    with pytest.raises(ValueError, match="D2 cap"):
        _prod(6, enforce_band=True)


def test_cap_counts_trainable_parameters_only():
    d = _dec(1, enforce=False)
    n_all = d.n_params
    for p in d.mem_pos, d.queries:
        p.requires_grad_(False)
    assert d.n_trainable == n_all - d.mem_pos.numel() - d.queries.numel() and d.n_params == n_all


# =============================================================== lever 2: --r9-box-layers N ==========================================
def test_extend_depth_equals_a_from_scratch_build_and_draws_no_rng():
    torch.manual_seed(1)
    a = Box3DSlotDecoder(d_memory=D, n_memory=20, n_queries=12, d_model=D, depth=3, n_heads=4, enforce_band=False, presence_prior=0.01)
    a.extend_depth(6)
    after_a = torch.rand(3)
    torch.manual_seed(1)
    b = Box3DSlotDecoder(d_memory=D, n_memory=20, n_queries=12, d_model=D, depth=6, n_heads=4, enforce_band=False, presence_prior=0.01)
    after_b = torch.rand(3)
    assert len(a.blocks.layers) == 6 == a.blocks.num_layers and _sd_equal(a, b)
    assert torch.equal(after_a, after_b)                              # no RNG consumed by the extension
    mem = torch.randn(2, 20, D)
    oa, ob = a(mem), b(mem)
    assert all(torch.equal(oa[k], ob[k]) for k in ("raw", "presence_logit", "box3d"))


def test_extend_depth_refusals(monkeypatch):
    d = _dec(4)
    with pytest.raises(ValueError, match="only ever deepened"):
        d.extend_depth(3)
    d.extend_depth(4)                                                 # a no-op extension is allowed
    assert len(d.blocks.layers) == 4
    e = _dec(3, enforce=True)
    monkeypatch.setattr(BH, "BOX3D_PARAM_CAP", e.n_trainable + 1)     # one more layer would cross the cap
    with pytest.raises(ValueError, match="D2 cap"):
        e.extend_depth(4)


def test_default_off_is_bit_identical_state_forward_and_grad():
    """Unnamed flag (the default 0): attach/stamp touch nothing, pin_args is a no-op, and the decoder is the tip's, bit for bit."""
    ref, mine = _dec(3), _dec(3)
    model = SimpleNamespace(_perception=SimpleNamespace(box_dec=mine))
    C5.pin_args(_A())
    C5.attach(model, _A())
    assert C5.stamp(model, _A()) is None and not C5.is_any(_A()) and C5.declared_depth(_A()) == 3
    assert _sd_equal(ref, mine) and len(mine.blocks.layers) == 3
    mem = torch.randn(2, 20, D)
    mem2 = mem.clone()
    o1, o2 = ref(mem), mine(mem2)
    assert all(torch.equal(o1[k], o2[k]) for k in ("raw", "presence_logit", "cls_logits", "box3d", "yaw"))
    assert len(o1["aux"]) == len(o2["aux"]) == 2
    (o1["raw"].sum()).backward()
    (o2["raw"].sum()).backward()
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(ref.parameters(), mine.parameters()))


def test_depth_follows_n_in_forward_and_deep_supervision():
    d = _dec(3)
    d.extend_depth(6)
    out = d(torch.randn(2, 20, D))
    assert len(out["aux"]) == 5                                       # N - 1 earlier layers through the SHARED head
    d.deep_supervision = False
    assert "aux" not in d(torch.randn(2, 20, D))


def test_pin_args_refusals():
    C5.pin_args(_A(r9_box_layers=6))
    for bad in (dict(r9_box_layers=-1), dict(r9_box_layers=6, w_box3d=0.0), dict(r9_box_layers=True), dict(r9_box_layers=2.5)):
        with pytest.raises(SystemExit):
            C5.pin_args(_A(**bad))
    from tanitad.train import map_only_ft as mof
    if hasattr(mof, "is_on"):
        orig = mof.is_on
        try:
            mof.is_on = lambda a: True
            with pytest.raises(SystemExit, match="freeze"):
                C5.pin_args(_A(r9_box_layers=6))
        finally:
            mof.is_on = orig


# ------------------------------------------------------------ B3's per-layer machinery follows N (tested at N = 6) ------------------------
def _gt(b=2, a=5):
    box = torch.zeros(b, a, 4)
    box[..., 0] = torch.tensor([5.0, 10.0, 20.0, 30.0, 40.0])
    box[..., 1] = torch.tensor([0.0, 2.0, -3.0, 1.0, 4.0])
    box[..., 2], box[..., 3] = 4.0, 2.0
    valid = torch.ones(b, a, dtype=torch.bool)
    valid[1, 3:] = False
    return box, torch.zeros(b, a), torch.zeros(b, a, dtype=torch.long), valid


def _tgt():
    box, yaw, cls, valid = _gt()
    t = {"box": box, "yaw": yaw, "cls": cls, "valid": valid, "occ": torch.full_like(yaw, -1.0), "rates": torch.zeros(2, 5, 3),
         "rates_mask": torch.zeros(2, 5, dtype=torch.bool)}
    return zh_targets(t)


def _build_branch(n_layers, attach=True, b3=True):
    """argparse-namespace -> attach -> the REAL PerceptionBranch (tiny) -> forward -> the B3 loss row.  Returns (model, branch, row)."""
    from tanitad.models import refcv6_perception_branch as perc
    cfg = perc.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, d_model=D, n_queries=12, enforce_param_band=False, presence_loss="focal",
                                      presence_prior=0.01, deep_supervision=True, vis1=False, query_select="learned_ref", bev_source="s16_lift")
    torch.manual_seed(0)
    br = perc.PerceptionBranch(cfg, d_image=16, image_hw=(4, 5))
    model = SimpleNamespace(_perception=br, _cls_class_weight=torch.ones(len(AGENT_CLASSES)), _vis1=False, training=True)
    args = _A(r9_box_layers=n_layers, r9_box_quality=b3, r9_box_dn=b3, r9_box_class_sqrt=b3, r9_box_inloss_nms=b3)
    if attach:
        C5.attach(model, args)
    if b3:
        BF.attach(model, args)
    br.train()
    return model, br, args, perc


def _run_row(model, br, args, perc):
    fm = torch.randn(2, 16, 4, 5, requires_grad=True)
    box, yaw, cls, valid = _gt()
    BF.prepare(model, {"agent_box": box, "agent_yaw": yaw, "agent_cls": cls, "agent_valid": valid}, "cpu")
    out = br(fm)
    row = BF.box3d_row(model, perc, out["box_slots"], _tgt(), cls_class_weight=model._cls_class_weight, visible_filter=True,
                       presence_loss="focal", vis1=False, vis=None, sel=torch.arange(2))
    return out, row


def _g(p) -> float:
    return 0.0 if p.grad is None else float(p.grad.abs().sum())


def _trace(attach=True):
    model, br, args, perc = _build_branch(6, attach=attach)
    out, row = _run_row(model, br, args, perc)
    row["loss"].backward()
    return model, br, out, row


def test_b3_machinery_follows_n_at_six_and_every_new_layer_gets_gradient():
    model, br, out, row = _trace(attach=True)
    dec = br.box_dec
    assert len(dec.blocks.layers) == 6
    assert row["box3d_n_layers"] == 6.0 and "box3d_layer5" in row and "box3d_presence_layer5" in row     # the per-layer loss loop saw 6
    assert len(out["box_slots"]["aux"]) == 5 and len(out["box_slots"]["_dn_out"]["layers"]) == 6          # aux + DN decode follow N
    for i in (3, 4, 5):                                                                                    # the NEW layers are live
        assert _g(dec.blocks.layers[i].linear1.weight) > 0.0 and _g(dec.blocks.layers[i].self_attn.in_proj_weight) > 0.0, i
    assert _g(br.box_dec.dn_embed.weight) > 0.0


def test_red_attach_removed_the_new_layers_do_not_exist():
    """MUTATION arm: delete the single hop that wires the lever (attach not called) and the reachability check goes RED."""
    model, br, out, row = _trace(attach=False)
    assert len(br.box_dec.blocks.layers) == 3 and row["box3d_n_layers"] == 3.0
    assert "box3d_layer5" not in row
    with pytest.raises(IndexError):
        br.box_dec.blocks.layers[5]


def test_zero_control_the_new_layers_read_exactly_zero_when_the_loss_cannot_reach_them():
    """A loss on layer 0's decode ALONE cannot depend on layers 3-5: their gradient must read EXACTLY 0.0 (the probe works), while the full
    loss (above) reads > 0 on the very same parameters."""
    model, br, args, perc = _build_branch(6, b3=False)
    out = br(torch.randn(2, 16, 4, 5))
    # box_slots["aux"][0] is layer 0's decode (deep supervision); it is computed BEFORE layers 1-5 touch the stream
    out["box_slots"]["aux"][0]["raw"].sum().backward()
    dec = br.box_dec
    for i in (1, 2, 3, 4, 5):
        assert all(_g(p) == 0.0 for p in dec.blocks.layers[i].parameters()), i
    assert _g(dec.blocks.layers[0].linear1.weight) > 0.0              # ... and layer 0 itself is live (the control is not trivially zero)


# ------------------------------------------------------------ G-DVB ------------------------------------------------------------------
def test_gdvb_layers_is_a_built_lever_and_the_check_catches_each_lie():
    from tanitad.train import declared_vs_built as dvb
    C5.register_dvb_levers(dvb.register)
    assert dvb.REGISTRY["r9_box_layers"].kind == "built"
    chk = dvb.REGISTRY["r9_box_layers"].check
    d3, d6 = _dec(3), _dec(3)
    d6.extend_depth(6)
    m3 = SimpleNamespace(_perception=SimpleNamespace(box_dec=d3))
    m6 = SimpleNamespace(_perception=SimpleNamespace(box_dec=d6))
    assert chk(m3, _A()) == [] and chk(m6, _A(r9_box_layers=6)) == [] and chk(m3, _A(r9_box_layers=3)) == []
    assert chk(m3, _A(r9_box_layers=6)) != []                         # argv names 6, the model built 3  (declared, not built)
    assert chk(m6, _A()) != []                                        # the model was deepened, the argv did not name it
    assert chk(m6, _A(r9_box_layers=5)) != []
    assert chk(SimpleNamespace(_perception=None), _A(r9_box_layers=6)) != [] and chk(SimpleNamespace(_perception=None), _A()) == []
    d6.blocks.num_layers = 5                                          # a stack whose own counter lies
    assert chk(m6, _A(r9_box_layers=6)) != []


def test_gdvb_registry_is_pinned_two_way_against_the_real_parser():
    import refc_v3_train as T
    from tanitad.train import declared_vs_built as dvb
    parser = T.build_parser()
    assert dvb.coverage(parser) == [] and "r9_box_layers" in dvb.REGISTRY
    act = [a for a in parser._actions if a.dest == "r9_box_layers"][0]
    assert act.default == 0 and act.type is int and act.option_strings == ["--r9-box-layers"]
    ns = parser.parse_args(["--r9-box-layers", "6"]) if not any(a.required for a in parser._actions) else None
    if ns is not None:
        assert ns.r9_box_layers == 6                                  # argparse -> namespace hop


# ------------------------------------------------------------ trainer reachability: source scan + mutations ---------------------------
def _hops(src: str) -> dict:
    return {"import": src.count("import box_c5_flag as _bc5"), "register": src.count("_bc5.register_dvb_levers(_dvb.register)"),
            "pin": src.count("_bc5.pin_args(args)"), "attach": src.count("_bc5.attach(model, args)"),
            "stamp": src.count("_bc5.stamp(model, args)"), "flag": src.count('"--r9-box-layers", type=int, default=0'),
            "pin_in_pin_refcv8": int(bool(re.search(r"def _pin_refcv8\(cfg, args\).*?_bc5\.pin_args\(args\)", src, re.S))),
            "attach_after_perception_build": int(bool(re.search(
                r"model\._perception = _perc\.build_perception_branch\(model, _pcfg\)\.to\(device\)\n\s*_bbf\.attach\(model, args\)[^\n]*\n"
                r"(?:\s*_[a-z0-9]+\.attach\(model, args\)[^\n]*\n)*?\s*_bc5\.attach\(model, args\)", src)))}


def _ok(h: dict) -> bool:
    return all(h[k] == 1 for k in h)


def test_trainer_reaches_the_lever_and_each_mutation_FAILS():
    src = TRAINER.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert _ok(_hops(src)), _hops(src)
    for needle in ("_bc5.attach(model, args)", "_bc5.pin_args(args)", "_bc5.stamp(model, args)", "_bc5.register_dvb_levers(_dvb.register)",
                   "import box_c5_flag as _bc5", '"--r9-box-layers", type=int, default=0'):
        assert not _ok(_hops(src.replace(needle, "pass  # " + needle[:8], 1))), needle
    # the eval loader (G-EVAL: strict load of a depth-N state_dict into the same build) calls the same attach
    lsrc = LOADER.read_bytes().decode("utf-8")
    assert lsrc.count("tr._bc5.attach(model, args)") == 1


def test_stamp_reads_the_built_decoder_and_is_absent_when_off():
    d = _dec(3)
    d.extend_depth(6)
    model = SimpleNamespace(_perception=SimpleNamespace(box_dec=d))
    s = C5.stamp(model, _A(r9_box_layers=6))
    assert s["flags"] == {"--r9-box-layers": 6} and s["layers"]["depth"] == 6 and s["layers"]["d2_cap"] == BOX3D_PARAM_CAP
    assert s["layers"]["box_decoder_params"] == d.n_params
    assert C5.stamp(model, _A()) is None


def test_production_param_counts_depth3_and_depth6_are_recorded():
    """depth 3 = 3,858,199 (the tip's, design 2.4 'today'); depth 6 = 7,018,519 (design ESTIMATE '~ 7.0 M': it holds).  One layer = 1,053,440."""
    d3, d6 = _prod(3, enforce_band=True), _prod(6, enforce_band=True)
    assert d3.n_params == 3_858_199 and d6.n_params == 7_018_519
    assert (d6.n_params - d3.n_params) == 3 * 1_053_440
    d3e = _prod(3, enforce_band=True)
    d3e.extend_depth(6)
    assert d3e.n_params == d6.n_params                                # in-place extension == the from-scratch count


# ------------------------------------------------------------ the warm start (--init-from refcv8, 3 layers) into a deeper decoder -------
def _ws_model(depth):
    from torch import nn
    m = nn.Module()
    m._perception = nn.Module()
    m._perception.box_dec = _dec(3)
    m._perception.box_dec.extend_depth(depth)
    m._perception.other = nn.Linear(3, 3)
    return m


def _ws_source(tmp_path):
    src = _ws_model(3)
    p = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "step": 7}, str(p))
    return src, str(p)


def test_warm_start_names_exactly_the_new_layers_and_loads_the_rest(tmp_path):
    from tanitad.refs import refcv9_perc_rig as rig
    src, p = _ws_source(tmp_path)
    assert rig.box_c5_exemptions(_ws_model(3)) == ()                                  # OFF / tip depth: the tip's warm start, no exemption
    m = _ws_model(6)
    assert rig.box_c5_exemptions(m) == tuple(f"_perception.box_dec.blocks.layers.{i}." for i in (3, 4, 5))
    assert rig.refcv9_warm_exemptions(m)[0] == rig.box_c5_exemptions(m)               # reaches the ONE shared list (rig + full-model warm starts)
    stamp = rig.warm_start_perc_rig(m, p)
    assert stamp["n_loaded"] == len(src.state_dict())
    for i in (0, 1, 2):                                                               # the source's 3 layers are loaded, not left at init
        assert torch.equal(m._perception.box_dec.blocks.layers[i].linear1.weight, src._perception.box_dec.blocks.layers[i].linear1.weight)
    assert torch.equal(m._perception.box_dec.queries, src._perception.box_dec.queries)


def test_red_warm_start_without_the_exemption_refuses_and_an_unnamed_key_still_refuses(tmp_path, monkeypatch):
    from tanitad.refs import refcv9_perc_rig as rig
    _, p = _ws_source(tmp_path)
    monkeypatch.setattr(rig, "box_c5_exemptions", lambda model: ())                   # delete the one hop: the tip's refusal
    with pytest.raises(SystemExit, match=r"rig key\(s\) the source lacks"):
        rig.warm_start_perc_rig(_ws_model(6), p)
    monkeypatch.undo()
    m = _ws_model(6)
    m._perception.stray = torch.nn.Linear(2, 2)                                       # a key beyond the NAMED layers must still refuse
    with pytest.raises(SystemExit):
        rig.warm_start_perc_rig(m, p)
