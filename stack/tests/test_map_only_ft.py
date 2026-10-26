"""``--r8-train-only-map`` -- the map-only fine-tune (PI decision D-R9-MAPHEAD-PRECISION, 2026-10-26): train ONLY the
10 cm map head's map-only modules (refine + near + near_refine; 105,032 params on refcv8), the planner BIT-IDENTICAL.

CPU, a tiny REAL RefCV3Model (resnet18 at 64x64, pretrained=False) with the 10 cm branch, the near lift and one near
refine block, driven through the trainer's OWN functions (``apply_freeze`` via the flag, ``build_optimizer``,
``assert_optimizer_matches_recipe``, ``compute_losses_v3``, ``WeightEMA``). Every expectation is a LITERAL authored
here (the tiny model's map-only sizes are derived by hand from its dimensions in ``TINY_GROUPS``); every guard carries
a DELIBERATE-REGRESSION arm that must go RED:

  1. the freeze trains exactly the three modules; frozen norms stay in eval through ``model.train()``;
  2. TWO optimiser steps under the flag: every non-map parameter AND buffer (trunk BN statistics,
     ``num_batches_tracked``, the class-prior buffers, the anchors) and the EMA export are bit-identical to the source;
     every map-only module moved;  RED: one planner tensor un-frozen; RED: the class-prior write un-gated;
  3. the post-run assert tool on synthetic checkpoints (model-only and the rolling ``ckpt.pt`` with an EMA shadow);
  4. the argv refusals, G-DVB, the EMA exclusion (a -0.0 survives), the source-file refusals;
  5. the launch gate's route: the profile's literals, the init md5 binding, the built-set check and its two arms,
     the G-LIVE map-only judgement and its red records; refcv8's own profile refuses the flag.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses as dc
import hashlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assert_map_only_finetune as AT  # noqa: E402
import launch_gate as LG  # noqa: E402
import refc_v3_train as T  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_V2  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_only_ft as M  # noqa: E402
from tanitad.train import weight_ema as W  # noqa: E402

CLIP = "synthetic-map-hires-clip"
#: the tiny branch below: d_up = d_model = 16 (GroupNorm affine = 2 x 16), 8 classes, the near lift reading the
#: resnet18 stride-8 map (128 ch) x 4 = 512 channels. By hand: refine = 16*16 + 32 + 16*16*9 + 32 + 16*16*9 + 32 +
#: (8*16 + 8); near = 16 (unobserved) + 16*512 + 16; near_refine.0 = 16*16*9 + 32 + 16*16*9.
TINY_GROUPS = {"_map_hires.refine.inp.0": 256, "_map_hires.refine.inp.1": 32, "_map_hires.refine.conv1": 2304,
               "_map_hires.refine.norm1": 32, "_map_hires.refine.conv2": 2304, "_map_hires.refine.norm2": 32,
               "_map_hires.refine.cls": 136, "_map_hires.near.lift": 16, "_map_hires.near.lift.proj": 8208,
               "_map_hires.near_refine.0.c1": 2304, "_map_hires.near_refine.0.n1": 32,
               "_map_hires.near_refine.0.c2": 2304}
TINY_NUMEL = 17_960
#: a planner tensor the trajectory loss REACHES on this rig (MEASURED: core.decoder.traj_proj receives a non-zero
#: gradient when the decoder is trainable) -- the mutation arm's target
PLANNER_TENSOR = "core.decoder.traj_proj.weight"
#: the refcv8 literals (PI decision; MEASURED off ckpt_ema_48852.pt -- see map_only_ft's docstring)
REFCV8_NUMEL = 105_032
REFCV8_MODULES = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
SHIPPED_EMA = Path("E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt")
SHIPPED_EMA_MD5 = "b4364029b68f29a33eeb4aebbbbd002b"


# =========================================================================== #
# the rig                                                                     #
# =========================================================================== #
def _build(pool: bool = True):
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
                                  trunk_pretrained=False, in_channels=9, image_size=64, image_width=None,
                                  trunk_frozen_bn=True)
    model = v3.RefCV3Model(cfg)
    model.core.encoder.enable_s8_tap()
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(EXTENT_V2.x_max_m), y_half_m=float(EXTENT_V2.y_half_m),
                            d_lift=16, d_model=16, d_up=16, dilations=(1, 2), grad_ckpt=True,
                            near_lift_x_m=20.0, near_refine_blocks=1)
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = 1.0
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank(
        {CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d = 0.0, 0.0
    if pool:          # A6: the planner's pooled BEV reads the SHARED 0.25 m encoder (the coupling that stays frozen)
        model._perception = PB.build_perception_branch(
            model, PB.PerceptionBranchConfig(w_map=0.0, w_box3d=0.0, bev_source="map_hires_pool"))
    # a TRAINED map head: the near lift / near refine are zero-initialised at build; give them values
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.startswith(("_map_hires.near.", "_map_hires.near_refine.")):
                p.add_(0.02 * torch.randn(p.shape, generator=g))
        # a NEGATIVE ZERO in a frozen float buffer: an EMA that shadowed it would export +0.0
        model.core.encoder.net.bn1.running_mean[0] = -0.0
    return model, cfg


def _batch(cfg):
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[CLIP, CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1, v7l.IGNORE_ID], dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    batch["tac_goal_y"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 8, (2, 600, 320), generator=g, dtype=torch.uint8)
    codes[:, :40] = 255
    batch.update({"map_fine": codes, "map_fine_label": torch.tensor([True, True]),
                  "map_ep": torch.tensor([int(stable_episode_id(CLIP))] * 2)})
    return batch


def _args(**over):
    a = argparse.Namespace(lr=1e-3, weight_decay=1e-4, opt="dd", encoder_lr_mult=0.5, r8_train_only_map=True)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _save_source(model, path):
    """The trainer's model-only EMA layout (`_save_ema_model`): the SHIPPED model the flag warm-starts from."""
    torch.save({"model": {k: v.detach().clone() for k, v in model.state_dict().items()}, "step": 48852,
                "ema_decay": 0.9, "num_updates": 3, "kind": "ema_model_only"}, path)


def _clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _setup(tmp_path, pool=True):
    model, cfg = _build(pool)
    src_path = tmp_path / "ckpt_ema_src.pt"
    _save_source(model, src_path)
    args = _args(init_from=str(src_path))
    rec = M.apply_freeze(model)
    opt = T.build_optimizer(model, args)
    T.assert_optimizer_matches_recipe(opt, args, model=model)
    src, srec = M.load_source(str(src_path))
    ver = M.verify_against_source(model, src, fresh=True)
    ema = W.WeightEMA(model, 0.9, exclude=M.ema_exclude)
    assert M.ema_problems(ema, model, src, fresh=True) == []
    guard = M.FrozenStateGuard(src)
    return model, cfg, args, rec, opt, src, srec, ver, ema, guard


def _steps(model, cfg, opt, ema, n=2):
    batch = _batch(cfg)
    model.train()
    out = []
    for _ in range(n):
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        out.append(sorted(k for k, v in losses.items() if torch.is_tensor(v) and v.requires_grad))
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        ema.update(model)
    return out


# =========================================================================== #
# 1. the freeze                                                               #
# =========================================================================== #
def test_the_freeze_trains_exactly_the_three_modules_and_pins_the_frozen_norms():
    model, _cfg = _build()
    rec = M.apply_freeze(model)
    assert rec["n_trainable_params"] == TINY_NUMEL
    assert rec["trainable_leaf_groups"] == TINY_GROUPS
    assert rec["modules"] == list(REFCV8_MODULES)
    live = {n for n, p in model.named_parameters() if p.requires_grad}
    assert live and all(n.startswith(("_map_hires.refine.", "_map_hires.near.", "_map_hires.near_refine."))
                        for n in live)
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if n.startswith(("core.", "_map_hires.lift.", "_map_hires.encoder.", "_perception.")))
    model.train()            # the trainer calls this every step and after every eval: the pin must survive it
    bn_train = [n for n, m in model.named_modules()
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training]
    assert bn_train == []
    assert rec["frozen_norms_pinned"] == 20              # resnet18: 20 BatchNorms, every one outside the map head


def test_the_freeze_refuses_a_build_without_the_near_modules():
    model, _ = _build()
    model._map_hires.near_refine = None
    with pytest.raises(SystemExit, match="near_refine"):
        M.apply_freeze(model)


# =========================================================================== #
# 2. two optimiser steps: the identity, and its two red arms                  #
# =========================================================================== #
def test_TWO_STEPS_under_the_flag_frozen_state_and_EMA_are_bit_identical_and_the_map_head_moves(tmp_path):
    model, cfg, args, rec, opt, src, srec, ver, ema, guard = _setup(tmp_path)
    assert srec["kind"] == "ema_model_only" and ver["result"] == "bit-identical" and ver["fresh"] is True
    assert [g.get("name") for g in opt.param_groups] == ["map_only"]
    assert sum(int(p.numel()) for p in opt.param_groups[0]["params"]) == TINY_NUMEL
    assert set(ema.shadow) == {n for n in model.state_dict() if n.startswith(REFCV8_MODULES)
                               and model.state_dict()[n].is_floating_point()}
    before = _clone_state(model)
    grad_keys = _steps(model, cfg, opt, ema, n=2)
    # the planner does not depend on the map-only modules: only the map term (and the total) carry a gradient
    assert grad_keys == [["loss", "map_hires"], ["loss", "map_hires"]]
    assert all(p.grad is None for n, p in model.named_parameters() if not M.is_map_only(n))
    after = model.state_dict()
    frozen_moved = [k for k in before if not M.is_map_only(k) and not M.bit_equal(before[k], after[k])]
    assert frozen_moved == []
    assert guard.changed(model) == []
    # the buffers that WOULD move under a plain training step, named
    assert M.bit_equal(before["core.lat_log_prior"], after["core.lat_log_prior"])
    assert M.bit_equal(before["core.encoder.net.bn1.num_batches_tracked"],
                       after["core.encoder.net.bn1.num_batches_tracked"])
    assert str(after["core.encoder.net.bn1.running_mean"][0].item()) == "-0.0"
    # the EMA export: every non-map tensor IS the source, bit for bit (the -0.0 included)
    ema_sd = ema.model_state_dict(model)
    assert [k for k in src if not M.is_map_only(k) and not M.bit_equal(ema_sd[k], src[k])] == []
    # every map-only module moved, in the raw weights and in the EMA
    for mod in REFCV8_MODULES:
        raw = [k for k in before if k.startswith(mod + ".") and not M.bit_equal(before[k], after[k])]
        emv = [k for k in src if k.startswith(mod + ".") and not M.bit_equal(ema_sd[k], src[k])]
        assert raw and emv, (mod, raw, emv)
    assert int(ema.num_updates) == 2


def test_DELIBERATE_REGRESSION_one_planner_tensor_unfrozen_goes_RED(tmp_path):
    model, cfg, args, rec, opt, src, srec, ver, ema, guard = _setup(tmp_path)
    p = dict(model.named_parameters())[PLANNER_TENSOR]
    p.requires_grad_(True)
    opt.param_groups[0]["params"].append(p)
    assert any("OUTSIDE the map-only modules are trainable" in b for b in M.optimizer_problems(opt, model, args))
    assert [str(m) for m in M.dvb_check(model, args)] and "trainable outside" in str(M.dvb_check(model, args)[0])
    before = _clone_state(model)
    _steps(model, cfg, opt, ema, n=2)
    after = model.state_dict()
    moved = [k for k in before if not M.is_map_only(k) and not M.bit_equal(before[k], after[k])]
    assert moved == [PLANNER_TENSOR]
    assert guard.changed(model) == [PLANNER_TENSOR]
    with pytest.raises(SystemExit, match="FROZEN tensor"):
        guard.assert_ok(model, "step 2")


def test_DELIBERATE_REGRESSION_the_class_prior_write_ungated_goes_RED(tmp_path):
    """compute_losses_v3 skips core.update_tactical_prior under the flag; without the freeze marker it writes the
    planner's lat/lon prior buffers on every training step and the guard must see it."""
    model, cfg, args, rec, opt, src, srec, ver, ema, guard = _setup(tmp_path)
    delattr(model, M.MARKER_ATTR)
    _steps(model, cfg, opt, ema, n=1)
    assert set(guard.changed(model)) == {"core.lat_log_prior", "core.lon_log_prior"}


# =========================================================================== #
# 3. the post-run assert tool                                                 #
# =========================================================================== #
def _tiny_state():
    g = torch.Generator().manual_seed(5)
    return {"core.decoder.w": torch.randn(3, 4, generator=g), "core.encoder.bn.running_mean": torch.zeros(4),
            "core.encoder.bn.num_batches_tracked": torch.tensor(7),
            "_map_hires.lift.proj.weight": torch.randn(2, 2, generator=g),
            "_map_hires.refine.cls.weight": torch.randn(8, 2, generator=g),
            "_map_hires.near.lift.unobserved": torch.zeros(2),
            "_map_hires.near_refine.0.c1.weight": torch.randn(2, 2, generator=g)}


def _run_tool(src_obj, dst_obj, tmp_path, *extra):
    s, d = tmp_path / "src.pt", tmp_path / "dst.pt"
    torch.save(src_obj, s)
    torch.save(dst_obj, d)
    return AT.main(["--src", str(s), "--dst", str(d), "--json", str(tmp_path / "r.json"), *extra])


def test_ASSERT_TOOL_identical_passes_and_a_perturbed_non_map_tensor_fails(tmp_path):
    sd = _tiny_state()
    src = {"model": sd, "step": 48852, "kind": "ema_model_only"}
    assert _run_tool(src, copy.deepcopy(src), tmp_path) == 0
    # a map-only change only: PASS, and the change is named
    dst = copy.deepcopy(src)
    dst["model"]["_map_hires.refine.cls.weight"] += 0.5
    assert _run_tool(src, dst, tmp_path) == 0
    res = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert [r["key"] for r in res["views"]["ema"]["map_changed"]] == ["_map_hires.refine.cls.weight"]
    # RED: one non-map tensor, one ulp; a shared-encoder tensor; a BN counter; a +0.0 -> -0.0 flip
    for key, edit in (("core.decoder.w", lambda t: torch.nextafter(t, t + 1)),
                      ("_map_hires.lift.proj.weight", lambda t: t * 1.0001),
                      ("core.encoder.bn.num_batches_tracked", lambda t: t + 1),
                      ("core.encoder.bn.running_mean", lambda t: -t)):
        dst = copy.deepcopy(src)
        dst["model"][key] = edit(dst["model"][key])
        assert _run_tool(src, dst, tmp_path) == 1, key
        res = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        assert [r["key"] for r in res["views"]["ema"]["non_map_changed"]] == [key]
    # RED: a key lost
    dst = copy.deepcopy(src)
    del dst["model"]["core.decoder.w"]
    assert _run_tool(src, dst, tmp_path) == 1


def test_ASSERT_TOOL_reads_the_rolling_ckpt_with_its_EMA_shadow(tmp_path):
    sd = _tiny_state()
    src = {"model": sd, "step": 48852, "kind": "ema_model_only"}
    raw = copy.deepcopy(sd)
    raw["_map_hires.refine.cls.weight"] = raw["_map_hires.refine.cls.weight"] + 1.0
    shadow = {"_map_hires.refine.cls.weight": sd["_map_hires.refine.cls.weight"] + 0.1,
              "_map_hires.near_refine.0.c1.weight": sd["_map_hires.near_refine.0.c1.weight"].clone()}
    ck = {"model": raw, "opt": {}, "step": 120, "data_pos": {}, "param_names": [],
          "ema": {"schema": "weight_ema/1", "decay": 0.999, "num_updates": 120, "shadow": shadow}}
    assert _run_tool(src, ck, tmp_path) == 0
    res = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert sorted(res["views"]) == ["ema", "raw"] and res["src_view"] == "ema"
    assert res["dst"]["n_shadow"] == 2 and res["dst"]["n_shadow_outside_map_only"] == 0
    # RED: the EMA shadow carries a frozen tensor whose value differs (the -0.0 / lerp class): the EMA view fails
    bad = copy.deepcopy(ck)
    bad["ema"]["shadow"]["core.encoder.bn.running_mean"] = -sd["core.encoder.bn.running_mean"]
    assert _run_tool(src, bad, tmp_path) == 1
    res = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert res["views"]["raw"]["ok"] is True and res["views"]["ema"]["ok"] is False
    # a file that is not a trainer checkpoint: exit 2
    assert _run_tool(src, {"weights": sd}, tmp_path) == 2


def test_ASSERT_TOOL_literal_is_the_trainers_and_the_gates():
    assert AT.MAP_ONLY_MODULES == REFCV8_MODULES == M.MAP_ONLY_MODULES == LG._MAPFT_MODULES


# =========================================================================== #
# 4. argv, G-DVB, the EMA exclusion, the source                                #
# =========================================================================== #
_ON = ["--arm", "hier", "--out", "X", "--refcv8", "--map-hires", "on", "--map-hires-near-lift-m", "20",
       "--map-hires-near-refine-blocks", "1", "--w-map-hires", "1.0", "--opt", "dd", "--init-from", "e.pt",
       "--conflict-detector", "off", "--r8-train-only-map"]


def _pin(argv):
    M.pin_args(T.build_parser().parse_args(argv))


def _drop(argv, flag, n=1):
    i = argv.index(flag)
    return argv[:i] + argv[i + 1 + n:]


def test_GREEN_the_conformant_argv_pins_and_the_flag_off_is_a_noop():
    _pin(_ON)
    _pin(["--arm", "hier", "--out", "X", "--b1-guard-ref-ap2m", "0.2"])     # off: nothing is checked


@pytest.mark.parametrize("argv,needle", [
    (_drop(_ON, "--refcv8", 0), "--refcv8"),
    (_drop(_ON, "--map-hires"), "--map-hires is not `on`"),
    (_drop(_ON, "--map-hires-near-lift-m"), "`near` module"),
    (_drop(_ON, "--map-hires-near-refine-blocks"), "`near_refine` module"),
    (_drop(_ON, "--w-map-hires"), "--w-map-hires is 0"),
    (_drop(_ON, "--opt"), "--opt is not dd"),
    (_drop(_ON, "--init-from"), "--init-from is not set"),
    (_drop(_ON, "--conflict-detector"), "--conflict-detector is not `off`"),
    (_ON + ["--conflict-every", "50"], "--conflict-every"),
    (_ON + ["--b1-guard-ref-ap2m", "0.21211"], "B1 guard"),
    (_ON + ["--grad-share-every", "50"], "--grad-share-every"),
    (_ON + ["--trunk-bn-recalib", "64"], "--trunk-bn-recalib"),
    (_ON + ["--r8-critic-drivable"], "--r8-critic-drivable")])
def test_every_refusal_of_the_flag(argv, needle):
    with pytest.raises(SystemExit, match=needle.replace("(", r"\(").replace("[", r"\[")):
        _pin(argv)


def test_the_trainers_own_pin_calls_the_flags_refusals():
    """`_pin_refcv8` is the trainer's argv pin; the flag's refusals must be reached through it."""
    import types
    from tanitad.refs import refcv8_conditioning as r8c
    args = T.build_parser().parse_args(_drop(_ON, "--init-from"))
    with pytest.raises(SystemExit, match="--init-from is not set"):
        T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)


def test_GDVB_the_lever_reads_the_built_trainable_set():
    on, off = _args(), _args(r8_train_only_map=False)
    model, _ = _build()
    assert [str(m) for m in M.dvb_check(model, off)] == []                      # off, not frozen: clean
    assert "no map-only freeze" in str(M.dvb_check(model, on)[0])                # declared, not built: RED
    M.apply_freeze(model)
    assert M.dvb_check(model, on) == []
    assert "map-only freeze" in str(M.dvb_check(model, off)[0])                  # built, not declared: RED
    assert D.REGISTRY["r8_train_only_map"].kind == "built"
    # the grad-unreachable declaration: everything outside the three modules is declared frozen by the marker
    assert [m for m in D.check_grad_unreachable(model, off) if m.lever == "frozen parameters"] == []
    # RED: a map-only module frozen too is NOT covered by the declaration, and the lever sees it
    for n, p in model.named_parameters():
        if n.startswith("_map_hires.near_refine."):
            p.requires_grad_(False)
    assert [m for m in D.check_grad_unreachable(model, off) if m.lever == "frozen parameters"]
    assert any("frozen" in str(m) for m in M.dvb_check(model, on))


def test_the_EMA_exclusion_keeps_a_frozen_negative_zero_and_the_plain_EMA_does_not():
    """Why `exclude`: `lerp(x, x, w)` is exact for every value EXCEPT the sign of zero (-0.0 + w*0 = +0.0)."""
    m = torch.nn.Module()
    m.frozen = torch.nn.Parameter(torch.tensor([-0.0, 1.0]), requires_grad=False)
    m.register_buffer("buf", torch.tensor([-0.0, 2.0]))
    m.train_me = torch.nn.Parameter(torch.tensor([3.0]))
    plain = W.WeightEMA(m, 0.9998)
    excl = W.WeightEMA(m, 0.9998, exclude=lambda n: n != "train_me")
    plain.update(m)
    excl.update(m)
    assert str(plain.model_state_dict(m)["buf"][0].item()) == "0.0"               # MEASURED: the sign flips
    assert str(excl.model_state_dict(m)["buf"][0].item()) == "-0.0"
    assert set(excl.shadow) == {"train_me"} and set(plain.shadow) == {"buf", "train_me"}
    excl.load_state_dict(excl.state_dict(), m)                                     # a resume of the same layout
    with pytest.raises(W.EMAStateError):
        plain.load_state_dict(excl.state_dict(), m)                                # a resume across layouts: refused


def test_the_source_must_be_the_shipped_EMA_model_and_equal_bit_for_bit(tmp_path):
    model, _ = _build()
    raw = tmp_path / "raw.pt"
    torch.save({"model": model.state_dict(), "step": 5}, raw)                      # a RAW milestone file
    with pytest.raises(SystemExit, match="kind None"):
        M.load_source(str(raw))
    ema = tmp_path / "ema.pt"
    _save_source(model, ema)
    src, rec = M.load_source(str(ema))
    assert rec["kind"] == "ema_model_only" and rec["step"] == 48852 and len(rec["md5"]) == 32
    M.apply_freeze(model)
    with torch.no_grad():
        dict(model.named_parameters())[PLANNER_TENSOR].view(-1)[0] += 1e-6
    with pytest.raises(SystemExit, match="differ from --init-from"):
        M.verify_against_source(model, src, fresh=False)
    with torch.no_grad():
        model._map_hires.refine.cls.weight.view(-1)[0] += 1.0
    src2 = dict(src)
    src2[PLANNER_TENSOR] = model.state_dict()[PLANNER_TENSOR].clone()
    assert M.verify_against_source(model, src2, fresh=False)["result"] == "bit-identical"   # resume: map moved
    with pytest.raises(SystemExit):
        M.verify_against_source(model, src2, fresh=True)                                    # fresh: it may not


@pytest.mark.skipif(not SHIPPED_EMA.is_file(), reason="the shipped refcv8 EMA model is not on this host")
def test_MEASURED_the_shipped_refcv8_EMA_model_has_the_literal_map_only_sizes():
    h = hashlib.md5()
    with open(SHIPPED_EMA, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    assert h.hexdigest() == SHIPPED_EMA_MD5
    st = torch.load(str(SHIPPED_EMA), map_location="cpu", weights_only=False, mmap=True)
    assert st["kind"] == "ema_model_only" and st["step"] == 48852
    groups: dict = {}
    for k, v in st["model"].items():
        if k.startswith(tuple(m + "." for m in REFCV8_MODULES)):
            g = k.rsplit(".", 1)[0]
            groups[g] = groups.get(g, 0) + int(v.numel())
    assert sum(groups.values()) == REFCV8_NUMEL
    assert groups == LG._MAPFT_GROUPS == M.REFCV8_MAP_ONLY_GROUPS


# =========================================================================== #
# 5. the launch gate's route                                                  #
# =========================================================================== #
MAPFT = LG.PROFILES["refcv8-mapft"]
LAUNCH = ROOT / "ops" / "runs.d" / "refcv8-r101-s0.argv.json"
THOR_INIT = "/home/nvidia/refcv8_mapft/ckpt_ema_48852.pt"


def test_the_profile_literals():
    mo = MAPFT["map_only"]
    assert mo["numel"] == REFCV8_NUMEL == M.REFCV8_MAP_ONLY_NUMEL
    assert mo["init_from_md5"] == SHIPPED_EMA_MD5 == M.REFCV8_EMA_48852_MD5
    assert set(mo["groups"]) == set(LG._NT_MAP_ONLY_GROUPS) and sum(mo["groups"].values()) == REFCV8_NUMEL
    assert mo["groups"] == M.REFCV8_MAP_ONLY_GROUPS
    assert "--r8-train-only-map" in MAPFT["required_flags"]
    assert MAPFT["required_values"]["--conflict-detector"] == "off"
    for f in ("--steps", "--warmup", "--ema-decay", "--eval-every", "--b1-guard-ref-ap2m", "--b1-guard-step"):
        assert f not in MAPFT["required_values"], f
    assert MAPFT["open_items"] == ()          # MAPFT-BUDGET CLOSED (mapft_close_budget.py, mode open-only)
    # the box head is frozen: G-BOX-OVERFIT has no subject here (and ONLY here); the map overfit stays required
    assert MAPFT["required_checks"] == ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE",
                                        "G-SUITE-PINNED", "G-MAP-OVERFIT")
    assert "G-BOX-OVERFIT" in LG.PROFILES["refcv8"]["required_checks"]
    assert "G-BOX-OVERFIT" in LG.PROFILES["refcv8-nt"]["required_checks"]
    assert MAPFT["map_hires"]["ga_keys"] == ("ga_mh_refine", "ga_mh_near", "ga_mh_near_refine")
    assert MAPFT["map_hires"]["ga_keys_zero"] == ("ga_mh_lift", "ga_mh_encoder", "ga_mh_trunk_s8_stage")


def _flag_set(argv, flag, vals):
    out, i, done = [], 0, False
    while i < len(argv):
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        if argv[i] == flag:
            if vals is not None and not done:
                out += [flag, *vals]
            done = True
        else:
            out += argv[i:j]
        i = j
    if not done and vals is not None:
        out += [flag, *vals]
    return out


@pytest.fixture(scope="module")
def mapft_launch(tmp_path_factory):
    """refcv8-r101-s0's committed launch argv -> the map-only launch: the flag, the shipped EMA model as the warm
    start, the frozen-gradient instruments off, a budget (any; MAPFT-BUDGET is open)."""
    d = json.loads(LAUNCH.read_text(encoding="utf-8"))
    a = list(d["argv"])
    for f, v in {"--r8-train-only-map": [], "--init-from": [THOR_INIT], "--conflict-detector": ["off"],
                 "--b1-guard-ref-ap2m": None, "--b1-guard-step": None, "--grad-share-every": None,
                 "--grad-share-terms": None, "--grad-share-at": None, "--conflict-every": None,
                 "--steps": ["3000"], "--warmup": ["100"], "--ema-decay": ["0.999"]}.items():
        a = _flag_set(a, f, v)
    tmp = tmp_path_factory.mktemp("mapft")
    init = tmp / "ckpt_ema_48852.pt"
    init.write_bytes(b"stand-in for the shipped EMA model")
    rec = tmp / "nav_tau.json"
    rec.write_text(json.dumps({"status": "OK", "tau": float(LG.flag_values(a, "--nav-compliance-tau-rad")[0]),
                               "split_v2_cache": LG.flag_values(a, "--v2-cache")[0],
                               "labels": LG.flag_values(a, "--v7-labels")[0]}), encoding="utf-8")
    prof = dict(MAPFT, map_only=dict(MAPFT["map_only"],
                                     init_from_md5=hashlib.md5(init.read_bytes()).hexdigest()))
    return a, prof, {"nav_tau_record": str(rec)}, [(THOR_INIT.rsplit("/", 1)[0], str(tmp))], init


def test_GATE_the_map_only_launch_argv_has_no_refusal(mapft_launch):
    a, prof, opts, pmap, _ = mapft_launch
    refusals, pending = LG.profile_argv_rules(prof, a, opts, pmap)
    assert refusals == [] and pending == [], refusals


@pytest.mark.parametrize("flag,vals,needle", [
    ("--r8-train-only-map", None, "requires --r8-train-only-map"),
    ("--conflict-detector", ["auto"], "--conflict-detector off is REQUIRED"),
    ("--b1-guard-ref-ap2m", ["0.21211"], "B1 guard"),
    ("--grad-share-every", ["250"], "gradient-share"),
    ("--conflict-every", ["50"], "dead cadence"),
    ("--trunk-bn-recalib", ["64"], "REWRITES"),
    ("--r8-critic-drivable", [], "READ the map head")])
def test_GATE_each_map_only_rule_refuses(mapft_launch, flag, vals, needle):
    a, prof, opts, pmap, _ = mapft_launch
    refusals, _p = LG.profile_argv_rules(prof, _flag_set(a, flag, vals), opts, pmap)
    assert any(needle in r for r in refusals), refusals


def test_GATE_the_warm_start_is_bound_by_content(mapft_launch, tmp_path):
    a, prof, opts, pmap, init = mapft_launch
    other = tmp_path / "ckpt_ema_other.pt"
    other.write_bytes(b"a different checkpoint")
    r, _ = LG.map_only_init_reasons(prof, _flag_set(a, "--init-from", [str(other)]), pmap)
    assert len(r) == 1 and "the profile pins" in r[0]
    r, _ = LG.map_only_init_reasons(prof, _flag_set(a, "--init-from", [str(tmp_path / "absent.pt")]), pmap)
    assert len(r) == 1 and "not readable" in r[0]
    r, _ = LG.map_only_init_reasons(MAPFT, a, pmap)              # the REAL literal against a stand-in: refused
    assert len(r) == 1 and SHIPPED_EMA_MD5 in r[0]
    assert LG.map_only_init_reasons(LG.PROFILES["refcv8"], a, pmap) == ([], {"on": False})   # inert elsewhere


def test_GATE_refcv8_and_its_derived_profiles_refuse_the_flag():
    for name in ("refcv8", "refcv8-nt", "v8f"):
        r, _ = LG.forbidden_lever_reasons(LG.PROFILES[name], ["--r8-train-only-map"])
        assert r and "refcv8-mapft" in r[0], name
    assert LG.forbidden_lever_reasons(MAPFT, ["--r8-train-only-map"])[0] == []


def _tiny_prof():
    return dict(MAPFT, map_only=dict(MAPFT["map_only"], groups=dict(TINY_GROUPS), numel=TINY_NUMEL))


def test_GATE_G_DVB_reads_the_built_set_and_both_arms_go_RED():
    prof, argv = _tiny_prof(), ["--r8-train-only-map"]
    model, _ = _build()
    M.apply_freeze(model)
    r, det = LG.map_only_built_reasons(prof, argv, model)
    assert r == [] and det["trainable_numel"] == TINY_NUMEL
    assert LG.map_only_built_reasons(LG.PROFILES["refcv8"], argv, model) == ([], {"on": False})
    r, _ = LG.map_only_built_reasons(MAPFT, argv, model)          # the REFCV8 literal against the tiny build: RED
    assert any("105032" in x for x in r)
    ctx = type("C", (), {"arm": "mapft_planner_trainable"})()
    note = LG._mapft_inject(ctx, model, None, REFCV8_MODULES)
    assert note.startswith("mapft_planner_trainable")
    r, _ = LG.map_only_built_reasons(prof, argv, model)
    assert any("OUTSIDE" in x for x in r)
    model2, _ = _build()
    M.apply_freeze(model2)
    LG._mapft_inject(type("C", (), {"arm": "mapft_map_frozen"})(), model2, None, REFCV8_MODULES)
    r, _ = LG.map_only_built_reasons(prof, argv, model2)
    assert any("map-only parameter(s) are frozen" in x for x in r) and any("17960" in x for x in r)


def _grads_ok():
    return {"groups": [{"group": g, "numel": n, "params": 1, "nz_params": 1} for g, n in TINY_GROUPS.items()],
            "nonfinite_params": []}


def _live_rec(**over):
    r = {"opt_steps": 100, "frozen_grad_nz": {}, "n_frozen_params_tracked": 300, "n_frozen_tensors_checked": 400,
         "frozen_changed": [], "arm": None}
    r.update(over)
    return r


def test_GATE_G_LIVE_map_only_judgement_and_its_red_records():
    prof = _tiny_prof()
    assert LG.judge_map_only_live(prof, _live_rec(), _grads_ok())[0] == []
    assert LG.judge_map_only_live(LG.PROFILES["refcv8"], None, {})[0] == []          # inert elsewhere
    # (a) the planner trainable: an extra group in the optimizer, its gradient, its tensors moved
    g = _grads_ok()
    g["groups"].append({"group": "core.decoder.traj_proj", "numel": 8, "params": 2, "nz_params": 2})
    r, _ = LG.judge_map_only_live(prof, _live_rec(frozen_grad_nz={"core.decoder.traj_proj.weight": 99},
                                                  frozen_changed=["core.decoder.traj_proj.weight"]), g)
    assert any("OUTSIDE" in x for x in r) and any("non-zero gradient" in x for x in r) and any("CHANGED" in x
                                                                                                for x in r)
    # (b) the map modules frozen too: nothing trains
    g = {"groups": [dict(e, nz_params=0) for e in _grads_ok()["groups"]], "nonfinite_params": []}
    r, _ = LG.judge_map_only_live(prof, _live_rec(), g)
    assert any("received NO non-zero gradient" in x for x in r)
    r, _ = LG.judge_map_only_live(prof, _live_rec(opt_steps=0), {"groups": [], "nonfinite_params": []})
    assert any("not trained" in x for x in r) and any("no optimizer step" in x for x in r)
    # a frozen BUFFER that moved, alone
    r, _ = LG.judge_map_only_live(prof, _live_rec(frozen_changed=["core.lat_log_prior"]), _grads_ok())
    assert len(r) == 1 and "core.lat_log_prior" in r[0]
    # a check that read nothing is not a pass
    r, _ = LG.judge_map_only_live(prof, _live_rec(n_frozen_tensors_checked=0, n_frozen_params_tracked=0),
                                  _grads_ok())
    assert len(r) == 2
    r, _ = LG.judge_map_only_live(prof, _live_rec(), dict(_grads_ok(), nonfinite_params=[
        "_map_hires.refine.cls.weight"]))
    assert any("non-finite" in x for x in r)
