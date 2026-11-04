"""refcv9 C5 lever 4 -- ``--r9-box-perframe-queries`` (B4's 200 per-frame seeds + 100 normalised global anchors, LR x10 third optimizer group).

Tier A: the REAL ``PerceptionBranch`` + ``PerFrameQueries`` + ``box_c5_flag`` on a tiny geometry (s8 map 20 x 12 = 240 cells >= 200 seeds).  Tier B: source-level hop scans of the real
trainer / launch gate / harness (reachability: the flag reaches the loss AND the optimizer), and the REAL ``build_optimizer`` / ``assert_optimizer_matches_recipe``.
Every mechanism has a RED arm; the zero control reads EXACTLY 0.0."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STACK / "scripts"))

from tanitad.data.calib import CanonicalFrame  # noqa: E402
from tanitad.data.rig_projection import RigCamera, frame_to_cam_ray  # noqa: E402
from tanitad.models import persp3d_aux as P  # noqa: E402
from tanitad.models import refcv6_perception_branch as perc  # noqa: E402
from tanitad.models import slot_query_select as sqs  # noqa: E402
from tanitad.models.box3d_head import BOX3D_PARAM_CAP, Box3DSlotDecoder  # noqa: E402
from tanitad.refs.refc_agents import RigCameraBank  # noqa: E402
from tanitad.refs.refcv9_perc_rig import box_c5_exemptions  # noqa: E402
from tanitad.train import box_c5_flag as C5  # noqa: E402

D, D8, D16, H8, W8, IMG = 32, 8, 16, 20, 12, (4, 5)
FLAG = "--r9-box-perframe-queries"


def _src(rel: str) -> str:
    return (STACK / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


def _A(**kw):
    base = dict(r9_box_layers=0, r9_box_s8_horizon=False, r9_box_perframe_queries=True, w_box3d=1.0, map_hires="on", agent_rig_camera="extrinsics", opt="dd",
                slot_query_select="learned", r9_box_persp_aux=True, w_box_persp=0.1, r8_train_only_map=False, r9_train_only="", lr=1e-4, weight_decay=1e-4, encoder_lr_mult=0.5)
    base.update(kw)
    return SimpleNamespace(**base)


def _cam(theta=0.0):
    nom = RigCamera.nominal(CanonicalFrame(height=8 * H8, width=8 * W8, f_ref=40.0, projection="cylindrical"))
    import math
    c, s = math.cos(theta), math.sin(theta)
    Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=nom.R_cam_to_rig.dtype)
    return RigCamera(R_cam_to_rig=nom.R_cam_to_rig @ Rx, t_cam_in_rig=nom.t_cam_in_rig, frame=nom.frame)


class _M(nn.Module):
    """A model-shaped container: ``core.encoder`` (the dd optimizer's encoder prefix) + ``_perception`` (the real branch)."""

    def __init__(self, br):
        super().__init__()
        self.core = nn.Module()
        self.core.encoder = nn.Linear(2, 2)
        self.core.encoder.s8_tap, self.core.encoder.s8_dim, self.core.encoder.s16_dim, self.core.encoder.s8_shape = True, D8, D16, (H8, W8)
        self._perception = br
        self._rig_camera = RigCameraBank(by_episode={1: _cam(0.0), 2: _cam(0.03)})


def _model(*, pfq=True, persp=True, attach=True, seed=0, **akw):
    cfg = perc.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, d_model=D, n_queries=300, enforce_param_band=False, presence_loss="focal", presence_prior=0.01,
                                      deep_supervision=True, vis1=False, query_select="learned", bev_source="s16_lift")
    torch.manual_seed(seed)
    br = perc.PerceptionBranch(cfg, d_image=D16, image_hw=IMG)
    if persp:
        with torch.random.fork_rng(devices=[]):
            br.box_persp_aux = P.Persp3DAuxHead(D8, D16)
    m = _M(br)
    a = _A(r9_box_perframe_queries=pfq, **akw)
    if attach:
        C5.attach(m, a)
    br.train()
    return m, br, a


def _inp():
    g = torch.Generator().manual_seed(1)
    return torch.randn(2, D16, *IMG, generator=g), torch.randn(2, D8, H8, W8, generator=g)


def _fwd(m, br, *, stage=True):
    s16, s8 = _inp()
    if stage:
        C5.stage(m, {"agent_ep": torch.tensor([1, 2])}, "cpu")
    return br(s16, fmap_s8=s8)


def _loss(out):
    return sum(v.sum() for v in out["box_slots"].values() if torch.is_tensor(v) and v.requires_grad)


def _gsum(p) -> float:
    return 0.0 if p is None or p.grad is None else float(p.grad.abs().sum())


# ---------------------------------------------------------------------------------------------------- build + geometry =====
def test_build_300_slots_and_the_seed_lift_is_the_analytic_ground_point():
    m, br, a = _model()
    assert C5.dvb_check_pfq(m, a) == [] and br.wants_s8 and br.box_pfq.anchors.shape == (100, 2) and br.box_dec.n_queries == 300
    cam = _cam(0.0)
    heat = torch.full((1, 1, H8, W8), -9.0)
    heat[0, 0, 15, 6] = 9.0                                               # ONE strong cell
    depth = torch.full((1, 80, H8, W8), -30.0)
    depth[0, 40, 15, 6] = 30.0                                            # all mass on LID bin 40
    stub = lambda s8, s16: {8: {"heat": heat, "depth": depth}}            # noqa: E731
    R, t = cam.R_cam_to_rig[None].float(), cam.t_cam_in_rig[None].float()
    seeds = br.box_pfq.seeds(stub, torch.zeros(1, D8, H8, W8), torch.zeros(1, D16, *IMG), R, t, cam.frame)
    assert seeds.shape == (1, 200, 2)
    from tanitad.data import box_depth_targets as BD
    rng = float(BD.lid_centres()[40])
    ray = frame_to_cam_ray(torch.tensor(6.5 * 8 - 0.5), torch.tensor(15.5 * 8 - 0.5), cam.frame)
    p_rig = cam.to_rig((ray / ray.norm()) * rng)
    want = torch.stack([p_rig[0].clamp(0, br.box_pfq.x_fwd_m), p_rig[1].clamp(-br.box_pfq.y_half_m, br.box_pfq.y_half_m)])
    assert torch.allclose(seeds[0, 0], want.float(), atol=1e-3)           # the top seed IS the analytic lift (independent route)
    assert not seeds.requires_grad


# ---------------------------------------------------------------------- gradient, zero control, detached seeds, RED arms =====
def test_global_anchors_get_a_nonzero_gradient_and_the_detached_seed_head_reads_exactly_zero():
    m, br, a = _model()
    out = _fwd(m, br)
    assert out["box_slots"]["anchors"].shape == (2, 300, 2)
    _loss(out).backward()
    assert _gsum(br.box_pfq.anchors) > 0.0 and _gsum(br.box_qpos.mlp[0].weight) > 0.0
    assert all(_gsum(p) == 0.0 for p in br.box_persp_aux.parameters())    # ZERO CONTROL: the seeds are detached -- exactly 0.0 into the perspective head
    # ZERO CONTROL on the anchors themselves: a loss on the 200 SEED rows of the anchor tensor reads EXACTLY 0.0 on the global anchors; on the 100 global rows it is > 0
    m2, br2, _ = _model()
    _fwd(m2, br2)["box_slots"]["anchors"][:, :200].sum().backward()
    assert _gsum(br2.box_pfq.anchors) == 0.0
    m3, br3, _ = _model()
    _fwd(m3, br3)["box_slots"]["anchors"][:, 200:].sum().backward()
    assert _gsum(br3.box_pfq.anchors) > 0.0


def test_mutations_are_red_unstaged_forward_unattached_and_the_head_missing():
    m, br, a = _model()
    with pytest.raises(RuntimeError, match="no camera staged"):                      # RED: the staging hop deleted
        _fwd(m, br, stage=False)
    _fwd(m, br)
    with pytest.raises(RuntimeError, match="no camera staged"):                      # one-shot: a SECOND forward without a re-stage refuses
        _fwd(m, br, stage=False)
    m3, br3, a3 = _model(attach=False)                                               # RED: the attach hop deleted
    assert br3.box_pfq is None and [x.read_from for x in C5.dvb_check_pfq(m3, a3)] == ["model._perception.box_pfq"]
    m4, br4, a4 = _model(persp=False)                                                # RED: no seed head
    assert any("box_persp_aux" in x.read_from for x in C5.dvb_check_pfq(m4, a4))
    with pytest.raises(SystemExit, match="agent_ep"):
        C5.stage(m, {}, "cpu")


def test_default_off_is_bit_identical_and_stage_is_a_noop():
    m0, b0, a0 = _model(pfq=False, attach=False)
    m1, b1, a1 = _model(pfq=False, attach=True)
    assert b1.box_pfq is None and C5.stamp(m1, a1) is None and C5.dvb_check_pfq(m1, a1) == [] and not b1.wants_s8
    assert list(b0.state_dict()) == list(b1.state_dict()) and all(torch.equal(b0.state_dict()[k], b1.state_dict()[k]) for k in b0.state_dict())
    s16, _ = _inp()
    C5.stage(m1, {}, "cpu")                                                           # no box_pfq: nothing, even with no agent_ep
    assert torch.equal(b0(s16)["box_slots"]["box"], b1(s16)["box_slots"]["box"])
    # with the flag ON, every shared parameter is initialised exactly as OFF (the new ones draw inside fork_rng)
    m2, b2, _ = _model(pfq=True)
    assert all(torch.equal(b0.state_dict()[k], b2.state_dict()[k]) for k in b0.state_dict() if not k.startswith("box_persp_aux"))


def test_g_dvb_lies_and_pin_refusals():
    m, br, a = _model()
    assert C5.dvb_check_pfq(m, _A(r9_box_perframe_queries=False))                     # built-not-named
    br.box_pfq.anchors = nn.Parameter(torch.zeros(99, 2))
    assert any("anchors" in x.read_from for x in C5.dvb_check_pfq(m, a))                  # lie: shape
    for kw, needle in ((dict(r9_box_persp_aux=False), "r9-box-persp-aux"), (dict(w_box_persp=0.0), "r9-box-persp-aux"), (dict(opt="adam"), "--opt"),
                       (dict(map_hires="off"), "map-hires"), (dict(agent_rig_camera="off"), "extrinsics"), (dict(w_box3d=0.0), "w-box3d"),
                       (dict(slot_query_select="heatmap"), "slot-query-select")):
        with pytest.raises(SystemExit, match=needle):
            C5.pin_args(_A(**kw))
    C5.pin_args(_A())                                                                 # the B4 argv shape passes
    C5.pin_args(_A(r9_box_perframe_queries=False, r9_box_persp_aux=False))            # OFF: nothing checked


# ------------------------------------------------------------------------- the LR group, on the BUILT optimizer ==========
def _trainer():
    import refc_v3_train as tr
    return tr


def test_lr_x10_is_a_third_group_proven_on_the_built_optimizer_after_apply_lr_schedule():
    tr = _trainer()
    m, br, a = _model()
    opt = tr.build_optimizer(m, a)
    by = {g["name"]: g for g in opt.param_groups}
    assert set(by) == {"encoder", "head", "box_anchor"} and by["box_anchor"]["params"] == [br.box_pfq.anchors]
    assert not any(p is br.box_pfq.anchors for p in by["head"]["params"])             # split, not duplicated
    tr.apply_lr_schedule(opt, a, lambda s: 0.5, 0)
    assert by["box_anchor"]["lr"] == pytest.approx(10.0 * by["head"]["lr"], rel=1e-12) and by["encoder"]["lr"] == pytest.approx(0.5 * by["head"]["lr"], rel=1e-12)
    info = tr.assert_optimizer_matches_recipe(tr.build_optimizer(m, a), a, m)
    assert info["box_anchor_head_ratio_after_step"] == pytest.approx(10.0)
    # RED: the schedule overwriting the anchors' multiplier / the group absent while declared
    o2 = tr.build_optimizer(m, a)
    o2.param_groups[2]["lr"] = o2.param_groups[1]["lr"]
    o2.param_groups[2]["initial_lr"] = o2.param_groups[1]["lr"]
    with pytest.raises(SystemExit, match="box_anchor"):
        tr.assert_optimizer_matches_recipe(o2, a, m)
    with pytest.raises(SystemExit, match="recipe needs"):
        tr.assert_optimizer_matches_recipe(tr.build_optimizer(m, _A(r9_box_perframe_queries=False)), a, m)
    # OFF: still exactly two groups
    assert {g["name"] for g in tr.build_optimizer(m, _A(r9_box_perframe_queries=False)).param_groups} == {"encoder", "head"}


# ------------------------------------------------- classification, replay stand-in, warm start, B4 combined (D2 cap) =======
def test_classified_own_replay_standin_and_exemptions():
    import g_box_overfit as G
    import launch_gate as LG
    assert FLAG in LG.E1_BOX_OWN_FLAGS and FLAG in G.B3_FLAGS and tuple(LG.E1_BOX_OWN_FLAGS) == tuple(G.B3_FLAGS) and FLAG not in LG.E1_BOX_AUX_FLAGS
    assert LG._R9F_ATTACH[FLAG] == "_bc5.attach(model, args)"
    argv = ["--r9-box-layers", "6", FLAG, "--r9-box-persp-aux", "--w-box-persp", "0.1", "--opt", "dd"]
    build, aux = G.strip_aux(argv)
    assert FLAG in build and "--r9-box-persp-aux" not in build and aux == ["--r9-box-persp-aux"]
    rec = G.b3_record_fields(argv)
    assert FLAG in rec["b3"]["flags"] and rec[G.PFQ_STAMP] == [FLAG] and "stand-in" in rec[G.PFQ_STAMP_WHY]
    # the REPLAY build (persp head stripped): a frozen random stand-in, stamped as such; trainable parameters are the anchors + qpos only
    m, br, a = _model(persp=False, attach=False)
    m._pfq_seed_standin = True                                                          # what the harness sets right before the UNCHANGED `tr._bc5.attach(m, a)` line
    C5.attach(m, a)
    assert br.box_pfq.standin is not None and not any(p.requires_grad for p in br.box_pfq.standin.parameters())
    assert C5.dvb_check_pfq(m, a) == [] and C5.stamp(m, a)["perframe_queries"]["seed_source"].startswith("replay_standin")
    out = _fwd(m, br)
    _loss(out).backward()
    assert _gsum(br.box_pfq.anchors) > 0.0
    ex = box_c5_exemptions(m)
    assert "_perception.box_pfq." in ex and "_perception.box_qpos." in ex and box_c5_exemptions(_model(pfq=False)[0]) == ()


def test_b4_combined_depth6_param_count_under_the_d2_cap_and_pfq_adds_nothing_to_the_decoder():
    dec = Box3DSlotDecoder(d_memory=256, n_memory=2144, n_queries=300, d_model=256, enforce_band=False)
    dec.extend_depth(6)
    n6 = int(dec.n_params)
    assert n6 == 7_018_519 and n6 <= BOX3D_PARAM_CAP                                  # the C5-REST number (production geometry, 300 q)
    from tanitad.data.bev_raster import GRID_DEFAULT
    pfq, qpos = sqs.PerFrameQueries(GRID_DEFAULT), sqs.AnchorPosEmbed(256, GRID_DEFAULT)
    n_new = sum(p.numel() for p in pfq.parameters()) + sum(p.numel() for p in qpos.parameters())
    assert pfq.anchors.numel() == 200 and n_new == 200 + sum(p.numel() for p in qpos.parameters())
    assert int(dec.n_params) + n_new <= BOX3D_PARAM_CAP                                # even counted INTO the decoder it stays under 12 M
    # B4 shape through the real branch: layers 6 + pfq
    m, br, a = _model(r9_box_layers=6)
    assert len(br.box_dec.blocks.layers) == 6 and C5.dvb_check_pfq(m, a) == [] and C5.dvb_check_layers(m, a) == []
    out = _fwd(m, br)
    assert len(out["box_slots"]["aux"]) == 5 and out["box_slots"]["anchors"].shape == (2, 300, 2)


# ------------------------------------------------------------------------------ tier B: hops in the REAL sources ==========
def test_trainer_hops_parser_stage_optimizer_and_template():
    t = _src("scripts/refc_v3_train.py")
    assert t.count("_bc5.stage(model, batch, device)") == 2 and re.search(r'add_argument\("--r9-box-perframe-queries", action="store_true"', t)
    assert "_bc5.split_anchor_group(model, groups, args, float(args.lr))" in t and "_bc5.expected_groups(args)" in t and "box_anchor_head_ratio_after_step" in t
    assert t.index("_p3f.attach(model, args)") < t.index("_bc5.attach(model, args)")           # the seed head exists before pfq is attached
    for site in re.finditer(r"_bc5\.stage\(model, batch, device\)", t):                        # BEFORE the model forward at both sites
        assert t.index("out = model(frames", site.start()) > site.start() or t.index("out = _fwd(frames", site.start()) > site.start()
    rest = _src("tanitad/models/refcv6_perception_branch.py")
    assert "self.box_pfq(" in rest and "mem_wants_s8" in rest and 'if self.box_pfq is not None' in rest
    tpl = (STACK / "ops" / "runs.d" / "r9e1_B4.argv.TEMPLATE.md").read_text(encoding="utf-8")
    assert FLAG in tpl and "NOT BUILT" not in tpl
