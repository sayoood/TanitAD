"""refcv9 C5-S8B -- lever 3, ``--r9-box-s8-horizon`` (B4's s8 horizon tokens): reachability, the zero control, the mutation arms, default-off identity, G-DVB, the stamp.

CLAUDE.md "built, tested and unreachable from its caller": every test of the MECHANISM below is paired with one of the CALLER -- the real ``train()`` on the tiny E1 rig
(``run_trainer_until``), the harness replay build, the eval loader's call -- and each wiring hop has a mutation arm that must go RED.
  tier A (fast, no trainer): the real ``PerceptionBranch`` + ``Box3DMemory`` + ``Box3DSlotDecoder`` through ``box_c5_flag.attach``; geometry; G-DVB lies; pins; hops.
  tier B (the real caller): the tiny E1 rig argv + ``--r9-box-s8-horizon`` built by the real ``train()`` and by the harness replay.
"""
from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
STACK = Path(__file__).resolve().parents[1]
for _p in (str(STACK), str(STACK / "scripts"), str(STACK / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.calib import CanonicalFrame  # noqa: E402
from tanitad.data.rig_projection import RigCamera, project_cam_to_frame  # noqa: E402
from tanitad.models import refcv6_perception_branch as perc  # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES, AgentSlotDecoder  # noqa: E402
from tanitad.models.box3d_head import BOX3D_PARAM_CAP, Box3DMemory, Box3DSlotDecoder  # noqa: E402
from tanitad.refs.refc_agents import RigCameraBank  # noqa: E402
from tanitad.refs.refcv9_perc_rig import box_c5_exemptions  # noqa: E402
from tanitad.train import box_c5_flag as C5  # noqa: E402

TRAINER = STACK / "scripts" / "refc_v3_train.py"
LOADER = STACK / "tanitad" / "eval" / "refcv7_loader.py"
REFC_V3 = STACK / "tanitad" / "refs" / "refc_v3.py"
BRANCH = STACK / "tanitad" / "models" / "refcv6_perception_branch.py"
GBO = STACK / "scripts" / "g_box_overfit.py"
FLAG = "--r9-box-s8-horizon"
D, D8, H8, W8, IMG = 32, 8, 20, 6, (4, 5)            # tier A: d_model, s8 channels, s8 rows/cols (20 rows -> row0 in [0, 4]), the s16 map


def _src(p: Path) -> str:
    return p.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _A(**kw):
    base = dict(r9_box_layers=0, r9_box_s8_horizon=False, w_box3d=1.0, map_hires="on", agent_rig_camera="extrinsics", r8_train_only_map=False, r9_train_only="")
    base.update(kw)
    return SimpleNamespace(**base)


def _frame(h=8 * H8):
    return CanonicalFrame(height=h, width=48, f_ref=40.0, projection="cylindrical")


def _pitched(theta: float, frame=None) -> RigCamera:
    """A nominal camera pitched by ``theta`` rad about the camera x axis (positive = looking down)."""
    import math
    c, s = math.cos(theta), math.sin(theta)
    nom = RigCamera.nominal(frame or _frame())
    Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=nom.R_cam_to_rig.dtype)
    return RigCamera(R_cam_to_rig=nom.R_cam_to_rig @ Rx, t_cam_in_rig=nom.t_cam_in_rig, frame=nom.frame)


def _model(*, s8: bool, layers: int = 0, attach: bool = True, cam="nominal", stamped=None, seed: int = 0):
    cfg = perc.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, d_model=D, n_queries=12, enforce_param_band=False, presence_loss="focal",
                                      presence_prior=0.01, deep_supervision=True, vis1=False, query_select="learned_ref", bev_source="s16_lift")
    torch.manual_seed(seed)
    br = perc.PerceptionBranch(cfg, d_image=16, image_hw=IMG)
    enc = SimpleNamespace(s8_tap=True, s8_dim=D8, s8_shape=(H8, W8))
    m = SimpleNamespace(_perception=br, core=SimpleNamespace(encoder=enc), _rig_camera=_pitched(0.0) if cam == "nominal" else cam, training=True)
    args = _A(r9_box_s8_horizon=s8, r9_box_layers=layers)
    if attach:
        C5.attach(m, args, stamped=stamped)
    br.train()
    return m, br, args


def _inputs():
    g = torch.Generator().manual_seed(1)
    return torch.randn(2, 16, *IMG, generator=g), torch.randn(2, D8, H8, W8, generator=g)


def _g(p) -> float:
    return 0.0 if p is None or p.grad is None else float(p.grad.abs().sum())


def _mem_inputs(br):
    s16, s8 = _inputs()
    bev = torch.randn(2, 96, 8, 8) if br.box_mem.use_bev else None
    return s16, bev, s8


def _loss(out):
    return sum(v.sum() for v in out["box_slots"].values() if torch.is_tensor(v) and v.requires_grad)


# ================================================================================================================ geometry =====
def test_horizon_row_nominal_camera_is_the_frame_centre_and_pitch_moves_it_the_right_way():
    nom = C5.horizon_row0(_pitched(0.0), (H8, W8))
    assert nom["median_row_px"] == pytest.approx((8 * H8 - 1) / 2.0) and nom["row0"] == 2 and nom["n_rows"] == 16 and nom["n_tokens"] == 16 * W8
    # an INDEPENDENT route: the horizon is the rig ray (1, 0, 0) at infinity; project it through the camera's own frame
    for th in (-0.05, 0.03, 0.08):
        cam = _pitched(th)
        ray_cam = cam.R_cam_to_rig.transpose(0, 1) @ torch.tensor([1.0, 0.0, 0.0], dtype=cam.R_cam_to_rig.dtype)
        _col, row, _ok = project_cam_to_frame(ray_cam * 1e3, cam.frame)
        got = C5.horizon_row0(cam, (H8, W8))["median_row_px"]
        assert got == pytest.approx(float(row), abs=1e-6), th
    assert C5.horizon_row0(_pitched(0.08), (H8, W8))["median_row_px"] != C5.horizon_row0(_pitched(-0.05), (H8, W8))["median_row_px"]


def test_horizon_row_bank_is_the_median_of_its_cameras_and_the_band_clips_to_the_map():
    cams = {1: _pitched(-0.02), 2: _pitched(0.0), 3: _pitched(0.04)}
    info = C5.horizon_row0(RigCameraBank(by_episode=cams), (H8, W8))
    rows = sorted(C5.horizon_row0(c, (H8, W8))["median_row_px"] for c in cams.values())
    assert info["n_cameras"] == 3 and info["median_row_px"] == pytest.approx(rows[1]) and info["min_row_px"] == pytest.approx(rows[0]) and info["max_row_px"] == pytest.approx(rows[2])
    far = C5.horizon_row0(_pitched(0.6), (H8, W8))                       # horizon far BELOW the frame -> the band clips to the last 16 rows
    assert far["clipped"] and far["row0"] == H8 - 16
    assert C5.horizon_row0(_pitched(-0.6), (H8, W8))["row0"] == 0


def test_horizon_row_refusals():
    with pytest.raises(SystemExit, match="needs the rig camera"):
        C5.horizon_row0(None, (H8, W8))
    with pytest.raises(SystemExit, match="wrong grid"):
        C5.horizon_row0(_pitched(0.0, _frame(8 * H8 + 8)), (H8, W8))
    with pytest.raises(SystemExit, match="holds no camera"):
        C5.horizon_row0(RigCameraBank(by_episode={}), (H8, W8))


# ============================================================================== the mechanism: grads, zero control, RED ===========
def test_every_new_parameter_gets_a_nonzero_gradient_through_the_real_branch():
    m, br, _a = _model(s8=True)
    assert br.wants_s8 and br.box_mem.s8_rows == (2, 16) and br.box_dec.n_memory_in == br.box_mem.n_tokens + 16 * W8
    s16, s8 = _inputs()
    out = br(s16, fmap_s8=s8)
    _loss(out).backward()
    for name, p in (("s8_proj.weight", br.box_mem.s8_proj.weight), ("s8_proj.bias", br.box_mem.s8_proj.bias), ("src_embed_s8", br.box_mem.src_embed_s8),
                    ("mem_pos_s8", br.box_dec.mem_pos_s8)):
        assert _g(p) > 0.0, name
    assert _g(br.box_mem.src_embed) > 0.0                                  # the existing parameters still train


def test_zero_control_the_s8_parameters_read_exactly_zero_when_the_loss_cannot_reach_the_band():
    """A loss on the memory's NON-band tokens alone cannot depend on the s8 parameters: they must read EXACTLY 0.0 (the probe works) -- while the loss on
    the band tokens (below) reads > 0 on the very same parameters."""
    m, br, _a = _model(s8=True)
    s16, bev, s8 = _mem_inputs(br)
    n = br.box_mem.n_tokens
    mem = br.box_mem(s16, bev, s8=s8)
    assert mem.shape[1] == n + br.box_mem.s8_n_tokens
    mem[:, :n].sum().backward()
    assert _g(br.box_mem.s8_proj.weight) == 0.0 and _g(br.box_mem.s8_proj.bias) == 0.0 and _g(br.box_mem.src_embed_s8) == 0.0 and _g(br.box_dec.mem_pos_s8) == 0.0
    assert _g(br.box_mem.img_proj.weight) > 0.0                            # ... and the non-band path is live (the control is not trivially zero)
    br.zero_grad()
    br.box_mem(s16, bev, s8=s8)[:, n:].sum().backward()
    assert _g(br.box_mem.s8_proj.weight) > 0.0 and _g(br.box_mem.src_embed_s8) > 0.0
    br.zero_grad()
    br.box_mem(s16, bev, s8=s8)[:, n:].sum().backward()
    assert _g(br.box_mem.img_proj.weight) == 0.0                           # the mirror control: the band-only loss cannot reach the s16 projection


def test_s8_tokens_are_appended_LAST_and_only_the_horizon_rows_are_read():
    m, br, _a = _model(s8=True)
    s16, bev, s8 = _mem_inputs(br)
    mem = br.box_mem(s16, bev, s8=s8)
    n = br.box_mem.n_tokens
    r0, nr = br.box_mem.s8_rows
    want = br.box_mem.s8_proj(s8[:, :, r0:r0 + nr, :].flatten(2).transpose(1, 2)) + br.box_mem.src_embed_s8
    assert torch.equal(mem[:, n:], want)
    s8b = s8.clone()
    s8b[:, :, :r0] += 5.0                                                  # rows OUTSIDE the band do not enter the memory
    s8b[:, :, r0 + nr:] -= 5.0
    assert torch.equal(br.box_mem(s16, bev, s8=s8b), mem)


def test_RED_attach_removed_there_is_no_band_and_G_DVB_names_it():
    m, br, args = _model(s8=True, attach=False)
    assert not br.wants_s8 and not hasattr(br.box_mem, "s8_proj") and not any("s8" in k for k in br.state_dict())
    got = C5.dvb_check_s8(m, args)
    assert got and got[0].lever == "r9_box_s8_horizon" and got[0].declared is True and got[0].built is False


def test_RED_no_fmap_s8_reaches_a_branch_that_has_the_band_it_refuses_loudly():
    """The bev-hook hop deleted (the branch is called WITHOUT ``fmap_s8``): the memory refuses rather than silently dropping the band."""
    m, br, _a = _model(s8=True)
    with pytest.raises(ValueError, match="no fmap_s8 reached it"):
        br(_inputs()[0])


def test_default_off_is_bit_identical_state_keys_values_forward_and_hook():
    base, br0, _ = _model(s8=False, attach=False, seed=3)
    off, br1, a1 = _model(s8=False, attach=True, seed=3)
    s0, s1 = br0.state_dict(), br1.state_dict()
    assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0) and not any("s8" in k for k in s0)
    dec = br1.box_dec
    assert dec.memory_pos(torch.zeros(1, dec.n_memory, D)) is dec.mem_pos and dec.n_memory_in == dec.n_memory     # the hook returns TODAY's table, not a copy
    sentinel = object()
    assert AgentSlotDecoder.memory_pos(SimpleNamespace(mem_pos=sentinel), None) is sentinel                          # the base class too
    s16, s8 = _inputs()
    br0.eval(); br1.eval()
    with torch.no_grad():
        o0, o1 = br0(s16), br1(s16, fmap_s8=s8)                                                                        # an unasked-for fmap_s8 is IGNORED
    assert torch.equal(o0["box_slots"]["raw"], o1["box_slots"]["raw"]) if "raw" in o0["box_slots"] else True
    assert C5.stamp(off, a1) is None and C5.dvb_check_s8(off, a1) == [] and box_c5_exemptions(off) == ()
    assert C5.is_any(_A()) is False and C5.is_any(_A(r9_box_s8_horizon=True)) is True and C5.s8_named(SimpleNamespace()) is False       # an older namespace has no such dest


def test_attach_moves_no_other_parameters_init_and_leaves_the_global_rng_untouched():
    torch.manual_seed(11)
    m, br, args = _model(s8=False, attach=False, seed=11)
    before = torch.get_rng_state()
    C5.attach(m, _A(r9_box_s8_horizon=True))
    assert torch.equal(torch.get_rng_state(), before)                     # fork_rng: B4 vs B4-minus-s8 share every init that is built after the attach
    ref, _, _ = _model(s8=False, seed=11)
    sa, sr = br.state_dict(), ref.__dict__["_perception"].state_dict()
    assert all(torch.equal(sa[k], sr[k]) for k in sr)                     # every pre-existing parameter is byte-identical


# ====================================================================================== stamp, loader rebuild, warm start, B4 ===========
def test_stamp_reads_the_built_memory_and_the_loader_rebuilds_the_same_model_without_a_camera_bank():
    m, br, args = _model(s8=True, cam=_pitched(0.03))
    st = C5.stamp(m, args)
    h = st["s8_horizon"]
    assert st["flags"] == {FLAG: True} and "layers" not in st and h["row0"] == br.box_mem.s8_rows[0] and h["n_rows"] == 16 and h["s8_hw"] == [H8, W8]
    assert h["n_tokens"] == 16 * W8 and h["memory_tokens_total"] == br.box_dec.n_memory_in and h["source"] == "rig_camera"
    assert h["new_params"]["box_dec.mem_pos_s8"] == 16 * W8 * D and h["new_params"]["box_mem.src_embed_s8"] == D
    import json
    st = json.loads(json.dumps(st))                                       # through config.json
    m2, br2, args2 = _model(s8=True, cam=None, stamped=st)                # NO camera bank (the eval loader's situation), only the stamp
    s1, s2 = br.state_dict(), br2.state_dict()
    assert list(s1) == list(s2) and all(s1[k].shape == s2[k].shape for k in s1) and br2.box_mem.s8_rows == br.box_mem.s8_rows
    assert all(torch.equal(s1[k], s2[k]) for k in s1)                     # same seed, same build order -> the same init
    assert C5.stamp(m2, args2)["s8_horizon"]["source"] == "stamp" and C5.dvb_check_s8(m2, args2) == []
    with pytest.raises(SystemExit, match="needs the rig camera"):         # neither a bank nor a stamp -> loud
        _model(s8=True, cam=None)
    bad = json.loads(json.dumps(st)); bad["s8_horizon"]["s8_hw"] = [H8 + 1, W8]
    with pytest.raises(SystemExit, match="stamped s8 band"):
        _model(s8=True, cam=None, stamped=bad)
    assert re.search(r"tr\._bc5\.attach\(model, args, stamped=config\.get\(tr\._bc5\.STAMP_KEY\)\)", _src(LOADER))
    assert len(re.findall(r"_bc5\.attach\(", _src(LOADER))) == 1


def test_warm_start_names_exactly_the_new_s8_keys_and_only_when_built():
    m, br, _a = _model(s8=True, layers=6)
    ex = box_c5_exemptions(m)
    assert set(ex) == {f"_perception.box_dec.blocks.layers.{i}." for i in (3, 4, 5)} | {"_perception.box_dec.mem_pos_s8", "_perception.box_mem.s8_proj.", "_perception.box_mem.src_embed_s8"}
    keys = set(m.__dict__["_perception"].state_dict())
    covered = {k for k in keys if any(k.startswith(p.replace("_perception.", "", 1)) for p in ex)}
    assert {"box_dec.mem_pos_s8", "box_mem.s8_proj.weight", "box_mem.s8_proj.bias", "box_mem.src_embed_s8"} <= covered
    assert box_c5_exemptions(_model(s8=False)[0]) == ()


def test_B4_combined_with_six_layers_all_new_parameters_live_and_both_G_DVB_checks_clean():
    m, br, args = _model(s8=True, layers=6)
    assert len(br.box_dec.blocks.layers) == 6 and br.wants_s8
    assert C5.dvb_check_layers(m, args) == [] and C5.dvb_check_s8(m, args) == []
    s16, s8 = _inputs()
    _loss(br(s16, fmap_s8=s8)).backward()
    assert _g(br.box_mem.s8_proj.weight) > 0.0 and _g(br.box_dec.mem_pos_s8) > 0.0 and _g(br.box_dec.blocks.layers[5].linear1.weight) > 0.0
    st = C5.stamp(m, args)
    assert st["flags"] == {"--r9-box-layers": 6, FLAG: True} and st["layers"]["depth"] == 6 and st["s8_horizon"]["row0"] == 2
    assert [m_.lever for m_ in C5.dvb_check_s8(_model(s8=True, layers=6, attach=False)[0], args)] == ["r9_box_s8_horizon"]       # RED without attach


def test_D2_cap_production_numbers_6_layers_plus_the_s8_position_table():
    mem = Box3DMemory(d_image=1024, d_bev=96, d_model=256, image_hw=(26, 64))
    torch.manual_seed(0)
    dec = Box3DSlotDecoder(d_memory=256, n_memory=mem.n_tokens, n_queries=300, d_model=256, depth=3, presence_prior=0.01)
    dec.extend_depth(6)
    n6 = int(dec.n_params)
    assert n6 == 7_018_519
    dec.enable_s8_memory(16 * 128)                                        # 2,048 tokens at 416x1024
    assert int(dec.n_params) == n6 + 2048 * 256 and int(dec.n_params) <= BOX3D_PARAM_CAP and dec.n_memory_in == mem.n_tokens + 2048
    dec2 = Box3DSlotDecoder(d_memory=256, n_memory=mem.n_tokens, n_queries=300, d_model=256, depth=3, presence_prior=0.01)
    dec2.extend_depth(6)
    mem.enable_s8(128, (52, 128), 20)
    assert mem.s8_n_tokens == 2048 and sum(p.numel() for p in mem.s8_proj.parameters()) == 128 * 256 + 256


# ============================================================================================================== G-DVB lies ========
def test_gdvb_lie_tests_each_inconsistency_is_caught():
    m, br, args = _model(s8=True)
    assert C5.dvb_check_s8(m, args) == []
    assert [x.lever for x in C5.dvb_check_s8(m, _A())] == ["r9_box_s8_horizon"]            # built but the argv says off
    del br.box_dec.mem_pos_s8                                                                                      # the decoder lost its table
    assert any("positional table" in x.why for x in C5.dvb_check_s8(m, args))
    m, br, args = _model(s8=True)
    br.box_mem.s8_proj = torch.nn.Linear(D8 + 1, D)                                                                 # reads the wrong trunk width
    assert any("stride-8 width" in x.why for x in C5.dvb_check_s8(m, args))
    m, br, args = _model(s8=True)
    br.box_mem.s8_rows = (H8 - 3, 16)                                                                               # band off the map
    assert any("outside the s8 map" in x.why for x in C5.dvb_check_s8(m, args))
    m, br, args = _model(s8=True)
    br.box_mem.s8_hw = (H8 + 8, W8)
    assert any("another s8 map" in x.why for x in C5.dvb_check_s8(m, args))
    m, br, args = _model(s8=True)
    br.box_dec.mem_pos_s8 = torch.nn.Parameter(torch.zeros(1, 7, D))
    got = C5.dvb_check_s8(m, args)
    assert any("memory length" in x.why for x in got) and any("one row per band token" in x.why for x in got)
    assert [x.declared for x in C5.dvb_check_s8(SimpleNamespace(_perception=None), args)] == [True]               # no box memory at all
    from tanitad.train import declared_vs_built as dvb
    C5.register_dvb_levers(dvb.register)
    assert dvb.REGISTRY["r9_box_s8_horizon"].kind == "built"


def test_pin_args_refusals():
    C5.pin_args(_A())
    C5.pin_args(_A(r9_box_s8_horizon=True))
    C5.pin_args(_A(r9_box_s8_horizon=True, r9_box_layers=6))
    for kw, frag in ((dict(w_box3d=0.0), "--w-box3d is 0"), (dict(map_hires="off"), "--map-hires is not 'on'"),
                     (dict(agent_rig_camera="nominal"), "not 'extrinsics'"), (dict(agent_rig_camera="off"), "not 'extrinsics'"),
                     (dict(r9_train_only="planner"), "freeze is declared")):
        with pytest.raises(SystemExit, match="s8-horizon refused") as e:
            C5.pin_args(_A(r9_box_s8_horizon=True, **kw))
        assert frag in str(e.value), (kw, str(e.value))
    C5.pin_args(_A(w_box3d=0.0, map_hires="off"))                         # the flag OFF checks nothing


# ======================================================================================================= hop chain + mutations ==
def _hops(trainer: str, refc_v3: str, branch: str, loader: str) -> dict:
    tree = ast.parse(trainer)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    cfg_line = min(t.lineno for n in ast.walk(train) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name) and t.id == "cfg")
    cam = re.search(r"\n(\s*)if _bc5\.s8_named\(args\):[^\n]*\n\s*model\._rig_camera, _ = _build_rig_camera\(cfg, args\)[^\n]*\n\s*_bc5\.attach\(model, args\)", trainer)
    cam_ln = trainer[:cam.start()].count("\n") + 2 if cam else 10 ** 9
    return {"flag": trainer.count('g8.add_argument("--r9-box-s8-horizon", action="store_true"'),
            "cfg_in_scope_before_the_camera_line": int(cfg_line < cam_ln <= train.end_lineno),
            "camera_then_attach_in_train": int(bool(cam) and train.lineno < cam_ln < train.end_lineno),
            "pin": trainer.count("_bc5.pin_args(args)"), "stamp": trainer.count("_bc5.stamp(model, args)"), "register": trainer.count("_bc5.register_dvb_levers(_dvb.register)"),
            "bev_hook_passes_fmap_s8": refc_v3.count('**({"fmap_s8": fmap_s8} if getattr(br, "wants_s8", False) else {}))'),
            "branch_passes_s8_to_memory": branch.count('self.box_mem(fmap_s16, bev_feats, **({"s8": fmap_s8} if self.wants_s8 else {}))'),
            "branch_takes_fmap_s8": branch.count("map_banded: dict | None = None, fmap_s8: Tensor | None = None) -> dict:"),
            "loader_attach_with_stamp": loader.count("tr._bc5.attach(model, args, stamped=config.get(tr._bc5.STAMP_KEY))")}


def test_trainer_reaches_the_lever_and_each_mutation_FAILS():
    srcs = (_src(TRAINER), _src(REFC_V3), _src(BRANCH), _src(LOADER))
    h = _hops(*srcs)
    assert all(v == 1 for v in h.values()), h                             # includes: `cfg` is assigned in train() BEFORE the camera line (the HANDOFF's UNVERIFIED item)
    muts = [(0, 'g8.add_argument("--r9-box-s8-horizon", action="store_true"', 'g8.add_argument("--r9-box-s8-horizon-x", action="store_true"'),
            (0, "model._rig_camera, _ = _build_rig_camera(cfg, args)", "pass"), (0, "        _bc5.attach(model, args) ", "        pass  # "),
            (0, "_bc5.pin_args(args)", "pass"), (0, "_bc5.stamp(model, args)", "None"),
            (1, '**({"fmap_s8": fmap_s8} if getattr(br, "wants_s8", False) else {}))', ")"),
            (2, '**({"s8": fmap_s8} if self.wants_s8 else {})', ""), (3, "stamped=config.get(tr._bc5.STAMP_KEY)", "")]
    for i, old, new in muts:
        s = list(srcs)
        assert s[i].count(old) >= 1, old
        s[i] = s[i].replace(old, new, 1)
        assert any(v != 1 for v in _hops(*s).values()), old


def test_the_three_decoder_call_sites_use_the_memory_pos_hook_and_none_reads_mem_pos_directly():
    for rel in ("tanitad/models/agent_slots.py", "tanitad/models/box_b3.py", "tanitad/models/slot_query_select.py"):
        s = _src(STACK / rel)
        assert re.search(r"(dec|self)\.memory_pos\(memory\)\.to\(", s), rel
        assert not re.search(r"(dec|self)\.mem_pos\.to\(", s), rel
        assert "n_memory_in" in s, rel


# ====================================================================================== tier B: the REAL caller (tiny E1 rig) ====
@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    import test_g_box_overfit_near_lift as NL
    import test_g_box_overfit_rig as RG
    import launch_gate as LG
    d = tmp_path_factory.mktemp("c5s8b_rig")
    NL._synth_inputs(d)
    base = RG._rig_argv(d)
    i = base.index("--image-hw")
    base[i + 1:i + 3] = ["256", "640"]                                    # s8 = 32 x 80: the 16-row band has room to move (row0 in [0, 16])
    argvs = {"s8": base + [FLAG], "s8l6": base + [FLAG, "--r9-box-layers", "6"], "off": base}
    T = NL._load("refc_v3_train_c5s8b_ref", TRAINER)
    refs = {k: LG.run_trainer_until(T, v, "model")["model"] for k, v in argvs.items()}      # the REFERENCES: the real train()
    G = NL._load("g_box_overfit_c5s8b_t", GBO)
    return types.SimpleNamespace(d=d, T=T, G=G, NL=NL, LG=LG, argvs=argvs, ref=refs)


def _mutated_gbo(rig, tmp_path, name, edits):
    s = _src(GBO)
    for old, new in edits:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    p = tmp_path / f"{name}.py"
    p.write_text(s, encoding="utf-8", newline="\n")
    return rig.NL._load(f"{name}_t", p)


def test_the_real_train_builds_the_band_from_the_rig_camera_and_the_trained_forward_reaches_fmap_s8(rig):
    ref = rig.ref["s8"]
    br, enc = ref._perception, ref.core.encoder
    assert br.wants_s8 and enc.s8_tap and tuple(enc.s8_shape) == (32, 80) and br.box_mem.s8_n_tokens == 16 * 80 and br.box_dec.n_memory_in == br.box_mem.n_tokens + 1280
    want = C5.horizon_row0(ref._rig_camera, enc.s8_shape)                 # recomputed here, independently of the build
    assert br.box_mem.s8_rows == (want["row0"], 16) and want["n_cameras"] == len(ref._rig_camera.by_episode) > 1 and 0 < want["row0"] < 16
    args = rig.T.build_parser().parse_args(rig.argvs["s8"])
    assert rig.T._bc5.s8_named(args) and C5.dvb_check_s8(ref, args) == [] and C5.stamp(ref, args)["s8_horizon"]["row0"] == want["row0"]
    off = rig.ref["off"]._perception
    assert not off.wants_s8 and not any("s8" in k for k in off.state_dict())
    # the E1 rig trains through ``core.forward_perc_rig`` -> ``bev_hook(fmap_s16, fmap_s8=fmap_s8)``: the REAL trunk emits the map and the REAL hook hands it to the branch
    seen = {}
    orig = br.forward

    def rec(*a, **k):
        seen["fmap_s8"] = k.get("fmap_s8")
        raise StopIteration
    br.forward = rec
    try:
        grid, valid = ref._lift_bank_hires.for_episodes(torch.tensor(list(ref._rig_camera.by_episode)[:1]), device="cpu")
        frames = torch.rand(1, 3, 9, 256, 640)
        with pytest.raises(StopIteration):
            ref._forward_perc_rig(frames, grid, valid)
    finally:
        br.forward = orig
    assert seen["fmap_s8"] is not None and tuple(seen["fmap_s8"].shape[1:]) == (enc.s8_dim, 32, 80)


@pytest.mark.parametrize("kind", ["s8", "s8l6"])
def test_the_harness_replay_IS_the_s8_model_train_builds(rig, kind):
    m, ad = rig.NL._harness_model(rig.G, rig.T, rig.argvs[kind])
    ref = rig.ref[kind]
    assert ad.dvb_mismatches == [] and m._perception.box_mem.s8_rows == ref._perception.box_mem.s8_rows
    assert sorted(set(ref.state_dict()) ^ set(m.state_dict())) == []
    dt, dh = rig.LG.state_digests(ref.state_dict()), rig.LG.state_digests(m.state_dict())
    assert [k for k in dt if dt[k] != dh[k]] == []                        # same seed, same build order: the harness' init IS the launch init
    args = rig.T.build_parser().parse_args(rig.argvs[kind])
    assert ad.c5_stamp == rig.T._bc5.stamp(ref, args) and ad.c5_stamp["s8_horizon"]["n_tokens"] == 1280
    assert rig.G.b3_build_fields(ad.b3_stamp, ad.c5_stamp)["built_c5"]["s8_horizon"]["row0"] == m._perception.box_mem.s8_rows[0]
    if kind == "s8l6":
        assert len(m._perception.box_dec.blocks.layers) == 6


def test_RED_the_replay_without_the_attach_line_is_refused_by_G_DVB_and_without_the_camera_line_refuses_loudly(rig, tmp_path):
    G0 = _mutated_gbo(rig, tmp_path, "gbo_no_attach", [("        tr._bc5.attach(m, a)\n", "")])
    with pytest.raises(SystemExit) as e:
        rig.NL._harness_model(G0, rig.T, rig.argvs["s8"])
    assert "r9_box_s8_horizon" in str(e.value) and "NOT the declared launch model" in str(e.value)
    G1 = _mutated_gbo(rig, tmp_path, "gbo_no_camera", [("            m._rig_camera, _ = tr._build_rig_camera(cfg, a)\n", "            pass\n")])
    with pytest.raises(SystemExit, match="needs the rig camera"):
        rig.NL._harness_model(G1, rig.T, rig.argvs["s8"])


def test_OFF_replay_is_bit_identical_with_and_without_the_new_camera_line(rig, tmp_path):
    G1 = _mutated_gbo(rig, tmp_path, "gbo_off_no_camera", [("        if tr._bc5.s8_named(a):\n            m._rig_camera, _ = tr._build_rig_camera(cfg, a)\n", "")])
    m_new, ad_new = rig.NL._harness_model(rig.G, rig.T, rig.argvs["off"])
    m_old, _ad = rig.NL._harness_model(G1, rig.T, rig.argvs["off"])
    dn, do_, dt = (rig.LG.state_digests(x.state_dict()) for x in (m_new, m_old, rig.ref["off"]))
    assert sorted(dn) == sorted(do_) == sorted(dt) and [k for k in dn if dn[k] != do_[k] or dn[k] != dt[k]] == []
    assert ad_new.dvb_mismatches == [] and ad_new.c5_stamp is None


def test_launch_gate_classifies_the_flag_OWN_and_maps_it_to_the_c5_attach():
    import launch_gate as LG
    assert FLAG in LG.E1_BOX_OWN_FLAGS and LG._R9F_ATTACH[FLAG] == "_bc5.attach(model, args)"
    assert "_bc5.attach(model, args)" in _src(TRAINER)
    import g_box_overfit as G
    assert FLAG in G.B3_FLAGS and tuple(G.B3_FLAGS) == tuple(LG.E1_BOX_OWN_FLAGS)
