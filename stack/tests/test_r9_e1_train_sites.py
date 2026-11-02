"""refcv9 C7 (A3) -- the sites of ``train()`` AFTER the optimiser build that read a model the E1 rig builds WITHOUT its planner.

MEASURED on Thor (smoke3, 2026-11-02 21:05 UTC, rc 1 after 336 s): the record check refused the run
(``stamp names a planner seam (sampler / w_u0 / cross_agent) the rig cannot hold``) because ``--agents head`` sets
``cfg.core.decoder.cross_agent = True`` (``_pin_refcv5_seams``) and the stamp copied that intent for a decoder that is not built.
The dev-box probe (``post_opt_probe``: every model-reading helper of ``train()`` run on a tiny REAL rig, one try/except each)
then found the next refusal BEFORE a second 8-minute relaunch: ``row['traj']`` on the rig's first log row (a KeyError, no
trajectory term exists), plus the ``--out`` of all three argv files being a git-bash-mangled ``C:/Program Files/Git/...`` path.

CPU, a tiny REAL ``RefCV3Model`` built the way the smoke builds it (``--arm flat``, the agent head, the box-3D perception branch,
the 10 cm map branch, ``perc_rig=True``). Every expectation is a LITERAL; each guard carries a RED arm.
"""
from __future__ import annotations

import dataclasses as dc
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r9_e1_make_argv as M  # noqa: E402
import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.refs import refc_agents as A  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import declared_vs_built as dvb  # noqa: E402
from tanitad.train import weight_ema as W  # noqa: E402

RUNS_D = ROOT / "ops" / "runs.d"


def _build_smoke_rig(hier: bool = False):
    """The smoke's rig in miniature: ``--arm flat`` (hier False), ``--agents head`` (cfg decoder.cross_agent = True, as
    ``_pin_refcv5_seams`` sets it), the box-3D perception branch, the 10 cm map branch, ``perc_rig=True``."""
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=hier)
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k", trunk_pretrained=False,
                                  in_channels=9, image_size=64, image_width=None, trunk_frozen_bn=True)
    cfg.core.agents = A.AgentSeamConfig(enable=True, queries=4)
    cfg.core.decoder.cross_agent = True
    model = v3.RefCV3Model(cfg, perc_rig=True)
    model.core.encoder.enable_s8_tap()
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(R.EXTENT_V2.x_max_m), y_half_m=float(R.EXTENT_V2.y_half_m),
                            d_lift=16, d_model=16, d_up=16, dilations=(1, 2), grad_ckpt=True, near_lift_x_m=20.0,
                            near_refine_blocks=1)
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = 1.0
    h, wd = model.cfg.core.encoder.image_hw()
    frame = R.frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank({R.CLIP: R.RigCamera.nominal(frame, height_m=1.5, x_m=1.5)},
                                                     frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d, model._w_agent = 0.0, 0.0, 0.0       # the batch carries no agent / box labels
    model._perception = PB.build_perception_branch(
        model, PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, bev_source="map_hires_pool"))
    return model, cfg


@pytest.fixture(scope="module")
def smoke_rig():
    return _build_smoke_rig()


def _args(rig_on: bool):
    a = T.build_parser().parse_args(["--arm", "flat", "--out", "x"])
    a.r9_perc_rig = rig_on
    return a


# =========================================================================== #
# SITE 1: the run record (refc_v3_train.py _seam_stamp "cross_agent")          #
# =========================================================================== #
def test_the_rig_record_states_what_the_built_rig_holds_and_the_checker_accepts_it(smoke_rig, monkeypatch):
    model, cfg = smoke_rig
    monkeypatch.setattr(T, "_build_rig_camera", lambda c, a: (None, {"stub": True}))      # 64x64 has no canonical frame
    assert cfg.core.decoder.cross_agent is True                # the config's INTENT (what `--agents head` sets)
    assert not rig._has_path(model, "core.decoder")            # ...for a decoder the rig does not build
    stamp = T._seam_stamp(cfg, _args(True))
    assert stamp["cross_agent"] is False and stamp["sampler"] == "none" and stamp["w_u0"] == 0.0
    rig.assert_seams_built_perc_rig(model, stamp)              # the rig's own check ACCEPTS it
    T.assert_seams_are_built(model, stamp)                     # ...and so does the trainer's entry point, as train() calls it


def test_the_OFF_path_stamp_is_the_tips(smoke_rig, monkeypatch):
    _, cfg = smoke_rig
    monkeypatch.setattr(T, "_build_rig_camera", lambda c, a: (None, {"stub": True}))
    assert T._seam_stamp(cfg, _args(False))["cross_agent"] is True          # flag absent: the config intent, unchanged
    a = T.build_parser().parse_args(["--arm", "flat", "--out", "x"])
    assert not hasattr(a, "r9_perc_rig") or a.r9_perc_rig is False
    assert T._seam_stamp(cfg, a)["cross_agent"] is True


def test_RED_the_checker_keeps_its_ON_path_meaning(smoke_rig, monkeypatch):
    model, cfg = smoke_rig
    monkeypatch.setattr(T, "_build_rig_camera", lambda c, a: (None, {"stub": True}))
    good = T._seam_stamp(cfg, _args(True))
    for k, v in (("sampler", "ddim"), ("w_u0", 0.5), ("cross_agent", True)):
        bad = dict(good, **{k: v})                  # a record that claims a planner seam the rig cannot hold
        with pytest.raises(SystemExit, match="planner seam"):
            rig.assert_seams_built_perc_rig(model, bad)


def test_RED_the_writer_does_not_launder_a_planner_flag_that_really_is_on(smoke_rig, monkeypatch):
    model, cfg = smoke_rig
    monkeypatch.setattr(T, "_build_rig_camera", lambda c, a: (None, {"stub": True}))
    cfg2 = dc.replace(cfg)                          # a shallow copy whose decoder config is mutated below
    cfg2.core = dc.replace(cfg.core, decoder=dc.replace(cfg.core.decoder, sampler="ddim"))
    stamp = T._seam_stamp(cfg2, _args(True))
    assert stamp["sampler"] == "ddim"               # stamped as it is -- only cross_agent (set BY a perception flag) is rewritten
    with pytest.raises(SystemExit, match="planner seam"):
        rig.assert_seams_built_perc_rig(model, stamp)


# =========================================================================== #
# SITE 2: the first log row's stdout line (refc_v3_train.py _step_log_line)    #
# =========================================================================== #
def test_the_log_line_has_no_traj_on_the_rig_and_the_default_line_is_unchanged():
    assert T._step_log_line("hier", 10, {"loss": 1.23456, "traj": 0.5}, False) == "[v3:hier] step 10 loss 1.2346 traj 0.5000"
    assert T._step_log_line("flat", 10, {"loss": 1.23456}, True) == \
        "[v3:flat] step 10 loss 1.2346 (perception-only rig: no traj term)"
    with pytest.raises(KeyError):                   # RED: the pre-fix line on a rig row (no `traj` key) is the crash
        T._step_log_line("flat", 10, {"loss": 1.23456}, False)


# =========================================================================== #
# SITES 3..: every model-reading helper between the optimiser and the loop     #
# =========================================================================== #
def test_config_json_builders_read_the_rig_without_a_planner(smoke_rig):
    model, cfg = smoke_rig
    assert rig._has_path(model, "core.agent_head") and not rig._has_path(model, "phi_tac")
    assert isinstance(v3.param_breakdown_v3(model), dict)                      # `_run_config["param_breakdown"]`
    assert isinstance(model.provenance_roles(), dict)                          # `_run_config["provenance_roles"]`
    T._slot_refine_block(_args(True), model)                                   # `_run_config["slot_refine"]`
    T._refcv6_tactical_block(_args(True), model, None)                         # `_run_config["refcv6_tactical"]`
    assert isinstance(T._trunk_levers_built(model), dict)                      # `_run_config["trunk_memory_levers"]`
    assert dvb.declared_grad_unreachable(model) is not None                    # `_run_config["declared_vs_built"]`
    assert T._conflict_aux_weights(model) == {"map_hires": 1.0}                # reads (no AttributeError); the smoke argv turns the detector OFF
    d = T._grad_reach_declaration(model, _args(True))                          # `_run_config["grad_reach_logging"]`
    assert d["declared"] is True and "ga_trunk" in d["keys"] and "ga_trunk_n" in d["keys"]
    assert any(k.startswith("ga_mh_") for k in d["keys"])                      # the 10 cm branch's parts are declared too
    assert not any(k.startswith(("ga_planner", "ga_tac_decoder")) for k in d["keys"])   # no planner part exists to report


def test_first_step_row_ema_checkpoint_and_in_run_eval_run_on_the_rig(smoke_rig, tmp_path):
    model, cfg = smoke_rig
    args = _args(True)
    args.seed, args.batch = 0, 2
    opt = T.build_optimizer(model, T.build_parser().parse_args(["--arm", "flat", "--out", "x", "--opt", "dd"]))
    ema = W.WeightEMA(model, 0.999, exclude=None)
    model.train()
    batch = R._batch(cfg)
    losses = T.compute_losses_v3(model, batch, "cpu", log_metrics=True)         # -> compute_losses_perc_rig
    assert "loss" in losses and "traj" not in losses
    opt.zero_grad(set_to_none=True)
    losses["loss"].backward()
    # the D3 check the loop runs on its FIRST log row: declared ga_* keys against the row
    row = T._train_row_scalars(losses, model)
    row["step"] = 10
    for k, v in T._perc.grad_reach_report(model).items():
        row[f"ga_{k}"], row[f"ga_{k}_n"] = v["grad_abs_sum"], v["n_params_with_grad"]
    for k, v in T._mhr.grad_reach_report_hires(model).items():
        row[f"ga_mh_{k}"], row[f"ga_mh_{k}_n"] = v["grad_abs_sum"], v["n_params_with_grad"]
    cfgd = {"grad_reach_logging": T._grad_reach_declaration(model, args)}
    assert dvb.check_logged_rows(cfgd, [row]) == []
    row.pop("ga_trunk")                                                         # RED: a declared key the row lacks
    assert any("ga_trunk" in str(m) for m in dvb.check_logged_rows(cfgd, [row]))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    ema.update(model)
    # the checkpoint save at --save-every (the loop's own dict is `_ckpt_payload`'s, key for key)
    payload = T._ckpt_payload(model, opt, 25, {"epoch": 0, "batch": 5}, 10, args, ema, None)
    assert set(payload) == {"model", "opt", "step", "data_pos", "param_names", "ema"} or \
        set(payload) >= {"model", "opt", "step", "data_pos", "param_names"}
    assert len(payload["param_names"]) == sum(len(g["params"]) for g in opt.param_groups)
    assert T._p10_ckpt_extra(model) == {}                                       # no decoder -> no P10 identity
    T._save_ema_model(ema, model, 25, tmp_path / "ckpt_ema_25.pt")
    assert torch.load(tmp_path / "ckpt_ema_25.pt", map_location="cpu", weights_only=False)["step"] == 25
    # the in-run eval: eval-mode loss, the row, the EMA row
    model.eval()
    with torch.no_grad():
        el = T.compute_losses_v3(model, R._batch(cfg), "cpu", mode=args.mode, ablate_frames=args.ablate_frames)
    model.train()
    acc = {k: float(v.detach()) for k, v in el.items() if torch.is_tensor(v) and v.ndim == 0}
    erow = T._eval_row_from_acc(acc, 1, model)
    assert "eval_loss" in erow and "eval_traj" not in erow
    erow.update(T._ema_eval_row(model, ema, [R._batch(cfg)], args, "cpu", None, None, None))
    assert "ema_eval_loss" in erow


# =========================================================================== #
# SITE 4: the argv files' --out (a git-bash path conversion, not a trainer site) #
# =========================================================================== #
@pytest.mark.parametrize("tag", ["C0", "C1"])
def test_the_landed_argv_files_name_a_posix_run_dir_on_thor(tag):
    argv = json.loads((RUNS_D / f"r9e1_{tag}.argv.json").read_text(encoding="utf-8"))["argv"]
    out = argv[argv.index("--out") + 1]
    assert out == f"/home/nvidia/r8ns/r9e1/runs/r9e1-{tag}"                    # LITERAL: not `C:/Program Files/Git/home/...`
    assert "Program Files" not in " ".join(argv)


def test_RED_the_argv_builder_refuses_a_git_bash_mangled_path():
    base = ["--size", "small", "--w-agent", "1.0"]
    ok = M.build(base, seed=0, out="/home/nvidia/r8ns/r9e1/runs/x", steps=50)
    assert ok[ok.index("--out") + 1] == "/home/nvidia/r8ns/r9e1/runs/x"
    for bad in ("C:/Program Files/Git/home/nvidia/r8ns/r9e1/runs/x", "C:\\runs\\x", "runs/x"):
        with pytest.raises(SystemExit, match="absolute POSIX path"):
            M.build(base, seed=0, out=bad, steps=50)
    with pytest.raises(SystemExit, match="absolute POSIX path"):
        M.build(base, seed=0, out="/o", init_from="C:/Program Files/Git/home/nvidia/x.pt")
