"""``--r9-train-only <set>[,<set>]`` -- the refcv9 WARM START's freeze (PI decisions 2026-10-27: warm start from refcv8
EMA-48,852, perception FROZEN, the planner and the map head trained) and the ``refcv9-warm`` launch-gate profile.

The sets are LITERAL module lists: ``planner`` = ``core.decoder`` (EXACTLY what the launch gate's
``mapft_planner_trainable`` arm un-freezes), ``map`` = the map-only fine-tune's three modules. ``--r9-train-only map``
trains what ``--r8-train-only-map`` trains, bit for bit. Every expectation is a LITERAL authored here; every guard
carries a DELIBERATE-REGRESSION arm that must go RED:

  1. the sets, their parsing and refusals;
  2. the trainable / frozen COUNTS for map, planner and planner,map -- on the CI rig (test_map_only_ft's tiny REAL
     RefCV3Model, MEASURED literals) AND on the REAL refcv8-r101-s0 build (the eval kit's config.json +
     ckpt_ema_48852.pt through refcv7_loader.build_model on CPU, in a subprocess; skipped when the kit is absent);
  3. TWO optimiser steps under planner,map: every frozen parameter AND buffer bit-identical, every planner and map
     tensor moved, the EMA export frozen-identical;  RED: one perception tensor trainable (guard, optimizer check and
     G-DVB lever all fire, each naming the freeze);
  4. ``--r9-train-only map`` == ``--r8-train-only-map`` after two steps, bit for bit (state, EMA export, record);
  5. the argv refusals, the trainer's pin reaching them, the G-DVB lever both ways;
  6. the ``refcv9-warm`` profile: its literals, the launch argv, each rule's refusal, the init md5 binding, the two RED
     arms at G-DVB (rig + real build) and at G-LIVE, the admitted-dead rule of the train-only G-LIVE judgement and its
     mutation, the open items that block a PASS.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402  -- the CI map rig (_build / _batch / _args / _steps)
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_only_ft as M  # noqa: E402
from tanitad.train import weight_ema as W  # noqa: E402

MAP_MODULES = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
PM_MODULES = ("core.decoder",) + MAP_MODULES
#: the CI rig (test_map_only_ft._build: resnet18 at 64x64, d_model 32, 2 decoder layers, the 10 cm branch), MEASURED
#: 2026-10-27 with apply_train_only: 220 parameter tensors, 12,341,401 values; 20 BatchNorms (all frozen in each set)
RIG_TOTAL = (220, 12_341_401)
RIG_COUNTS = {            # sets -> (trainable tensors, trainable values, frozen tensors, frozen values)
    ("map",): (18, 17_960, 202, 12_323_441),
    ("planner",): (47, 43_049, 173, 12_298_352),
    ("planner", "map"): (65, 61_009, 155, 12_280_392)}
#: the rig's planner leaf groups (MEASURED; feat_proj = 512 x 32 + 32 by hand: the resnet18 stride-32 map -> d_model)
RIG_PLANNER_GROUPS = {
    "core.decoder.feat_proj": 16416, "core.decoder.traj_proj": 544, "core.decoder.cond_proj": 544,
    "core.decoder.time_embed": 96, "core.decoder.layers.0.norm_q": 64, "core.decoder.layers.0.cross": 3168,
    "core.decoder.layers.0.cross.out_proj": 1056, "core.decoder.layers.0.norm_f": 64,
    "core.decoder.layers.0.film.to_scale_shift": 2112, "core.decoder.layers.0.mlp.0": 2112,
    "core.decoder.layers.0.mlp.2": 2080, "core.decoder.layers.1.norm_q": 64, "core.decoder.layers.1.cross": 3168,
    "core.decoder.layers.1.cross.out_proj": 1056, "core.decoder.layers.1.norm_f": 64,
    "core.decoder.layers.1.film.to_scale_shift": 2112, "core.decoder.layers.1.mlp.0": 2112,
    "core.decoder.layers.1.mlp.2": 2080, "core.decoder.conf_head": 33, "core.decoder.offset_head": 528,
    "core.decoder.ctx_to_cond": 288, "core.decoder.lat_to_anchor": 60, "core.decoder.lon_to_anchor": 60,
    "core.decoder.tgt_proj": 1056, "core.decoder.tgt_film.to_scale_shift": 2112}
RIG_PM_GROUPS = {**RIG_PLANNER_GROUPS, **R.TINY_GROUPS}
#: the REAL refcv8-r101-s0 build (CPU, refcv7_loader.build_model on the shipped EMA file), MEASURED 2026-10-27:
#: 848 parameter tensors, 100,598,182 values
REAL_TOTAL = (848, 100_598_182)
REAL_COUNTS = {
    ("map",): (18, 105_032, 830, 100_493_150),
    ("planner",): (186, 15_365_801, 662, 85_232_381),
    ("planner", "map"): (204, 15_470_833, 644, 85_127_349)}
KIT = Path("E:/refcv8_eval_kit")
KIT_CFG, KIT_EMA = KIT / "ckpt" / "config.json", KIT / "ckpt" / "ckpt_ema_48852.pt"
SHIPPED_EMA_MD5 = "b4364029b68f29a33eeb4aebbbbd002b"
#: the losses that carry a gradient on the rig under planner,map (MEASURED): the planner's and the map head's
RIG_PM_GRAD_KEYS = ["cls", "law", "loss", "map_hires", "sel_v3", "traj"]
PERCEPTION_TENSOR = "core.encoder.net.conv1.weight"


# =========================================================================== #
# 1. the sets                                                                 #
# =========================================================================== #
def test_the_sets_are_literal_module_lists():
    assert M.TRAIN_ONLY_SETS == {"planner": ("core.decoder",), "map": MAP_MODULES}
    assert M.R9_FLAG == "--r9-train-only" and M.R9_DEST == "r9_train_only" and M.R9_GROUP_NAME == "train_only"
    assert M.parse_sets("planner,map") == ("planner", "map") == M.parse_sets("map,planner")
    assert M.parse_sets("map") == ("map",) and M.parse_sets(" planner ") == ("planner",)
    assert M.modules_of_sets(("planner", "map")) == PM_MODULES
    for bad, needle in (("", "empty set name"), ("planner,", "empty set name"), ("tactical", "unknown set"),
                        ("map,map", "named twice"), ("perception", "unknown set")):
        with pytest.raises(SystemExit, match=needle):
            M.parse_sets(bad)
    # the gate's literal is the trainer's, written twice (never read across)
    assert LG._R9W_MODULES == PM_MODULES and LG._R9W_SETS == "planner,map"
    assert (M.REFCV8_PLANNER_TENSORS, M.REFCV8_PLANNER_NUMEL, M.REFCV8_MAP_ONLY_TENSORS) == (186, 15_365_801, 18)


# =========================================================================== #
# 2. the counts: the CI rig and the real refcv8 build                         #
# =========================================================================== #
@pytest.mark.parametrize("sets", list(RIG_COUNTS))
def test_CI_RIG_the_trainable_and_frozen_counts_of_each_set(sets):
    model, _ = R._build()
    assert (sum(1 for _ in model.parameters()), sum(int(p.numel()) for p in model.parameters())) == RIG_TOTAL
    rec = M.apply_train_only(model, sets)
    got = (rec["n_trainable_tensors"], rec["n_trainable_params"], rec["n_frozen_tensors"], rec["n_frozen_params"])
    assert got == RIG_COUNTS[sets]
    assert sum(1 for p in model.parameters() if p.requires_grad) == RIG_COUNTS[sets][0]
    assert rec["sets"] == list(sets) and rec["flag"] == "--r9-train-only" and rec["schema"] == "train_only_ft/1"
    assert rec["frozen_norms_pinned"] == 20
    want = {("map",): R.TINY_GROUPS, ("planner",): RIG_PLANNER_GROUPS, ("planner", "map"): RIG_PM_GROUPS}[sets]
    assert rec["trainable_leaf_groups"] == want
    model.train()
    assert [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training] == []


_REAL_BUILD = r'''
import json, os, sys
tree, cfgp, ckpt = sys.argv[1:4]
os.environ["REFCV7_REPO"] = tree
for p in (os.path.join(tree, "stack"), os.path.join(tree, "stack", "scripts")):
    sys.path.insert(0, p)
from tanitad.eval import refcv7_loader as L
L.bootstrap()
import torch
torch.set_num_threads(4)
import tanitad
model, cfg, args, rec = L.build_model(json.load(open(cfgp, encoding="utf-8")), ckpt, device="cpu")
from tanitad.train import map_only_ft as M
import launch_gate as LG
out = {"tanitad": tanitad.__file__, "dvb_build": [str(m) for m in (rec.get("declared_vs_built") or {}).get("mismatches", [])],
       "total": [sum(1 for _ in model.parameters()), sum(int(p.numel()) for p in model.parameters())], "counts": {}}
for sets in (("map",), ("planner",), ("planner", "map")):
    r = M.apply_train_only(model, sets)
    out["counts"][",".join(sets)] = [r["n_trainable_tensors"], r["n_trainable_params"], r["n_frozen_tensors"],
                                     r["n_frozen_params"], r["per_set"]]
prof = LG.PROFILES["refcv9-warm"]
argv = ["--r9-train-only", "planner,map"]
args.r9_train_only, args.r8_train_only_map = "planner,map", False
out["groups"] = r["trainable_leaf_groups"]
out["gate_clean"] = LG.map_only_built_reasons(prof, argv, model)[0]
out["dvb_clean"] = [str(m) for m in M.dvb_check_r9(model, args)]
for arm in ("refcv9_perception_trainable", "refcv9_planner_frozen"):
    M.apply_train_only(model, ("planner", "map"))
    note = LG._refcv9_inject(type("C", (), {"arm": arm})(), model, None, LG._R9W_MODULES)
    out[arm] = {"note": note, "gate": LG.map_only_built_reasons(prof, argv, model)[0],
                "dvb": [str(m) for m in M.dvb_check_r9(model, args)]}
print("R9JSON:" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def real_build(tmp_path_factory):
    if not (KIT_CFG.is_file() and KIT_EMA.is_file() and Path("E:/refcv6_eval_kit").is_dir()):
        pytest.skip("the refcv8 eval kit (E:/refcv8_eval_kit, E:/refcv6_eval_kit) is not on this host")
    tmp = tmp_path_factory.mktemp("r9real")
    ov = tmp / "overrides.json"
    ov.write_text(json.dumps({
        "--agent-rig-extrinsics": str(KIT / "data" / "refcv8_train4572_eval139_extrinsics.json"),
        "--clip-clock-sidecar": str(KIT / "data" / "refcv8_clip_clock_sidecar_ts33.jsonl"),
        "--vis1-sidecar": str(KIT / "data" / "refcv8" / "vis1_sidecar_refcv8b1_train4572_eval139.npz"),
        "--nav-compliance-tau-file": str(KIT / "data" / "refcv8" / "nav_compliance_tau_train4572.json"),
        "--join-defect-masks": str(ROOT / "tanitad" / "configs" / "refcv8_join_label_defects_train4572.json")}),
        encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REFCV6_", "REFCV7_"))}
    env.update({"REFCV7_REMAP_OVERRIDES": str(ov), "REFCV7_KIT": "E:/refcv6_eval_kit", "CUDA_VISIBLE_DEVICES": "-1",
                "PYTHONPATH": str(ROOT)})
    code = tmp / "real_build.py"
    code.write_text(_REAL_BUILD, encoding="utf-8")
    p = subprocess.run([sys.executable, "-X", "utf8", str(code), str(ROOT.parent), str(KIT_CFG), str(KIT_EMA)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
    line = [x for x in p.stdout.splitlines() if x.startswith("R9JSON:")]
    assert line, f"the real build produced no result (exit {p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    out = json.loads(line[0][len("R9JSON:"):])
    assert Path(out["tanitad"]).resolve().is_relative_to((ROOT / "tanitad").resolve()), out["tanitad"]
    return out


def test_REAL_BUILD_the_trainable_and_frozen_counts_of_each_set(real_build):
    assert real_build["dvb_build"] == []
    assert tuple(real_build["total"]) == REAL_TOTAL
    for sets, want in REAL_COUNTS.items():
        got = real_build["counts"][",".join(sets)]
        assert tuple(got[:4]) == want, (sets, got)
    pm = real_build["counts"]["planner,map"][4]
    assert (pm["planner"]["n_tensors"], pm["planner"]["n_params"]) == (186, 15_365_801)
    assert (pm["map"]["n_tensors"], pm["map"]["n_params"]) == (18, 105_032)
    # the gate's 111-group literal IS the real build's trainable set
    assert real_build["groups"] == LG._R9W_GROUPS and len(LG._R9W_GROUPS) == 111
    assert sum(LG._R9W_GROUPS.values()) == 15_470_833 == LG._R9W_NUMEL


def test_REAL_BUILD_G_DVB_is_clean_and_both_RED_arms_are_caught(real_build):
    assert real_build["gate_clean"] == [] and real_build["dvb_clean"] == []
    a = real_build["refcv9_perception_trainable"]
    assert a["note"].startswith("refcv9_perception_trainable: 316 perception tensor(s) un-frozen (first core.encoder.")
    assert any("parameter(s) OUTSIDE" in r for r in a["gate"]) and any("15470833" in r for r in a["gate"])
    assert any("the perception freeze is broken" in r for r in a["dvb"])
    b = real_build["refcv9_planner_frozen"]
    assert b["note"] == "refcv9_planner_frozen: 186 planner (core.decoder) tensors frozen (the --r9-train-only planner set)"
    assert any("186 map-only parameter(s) are frozen" in r for r in b["gate"]) and any("105032" in r for r in b["gate"])
    assert any("186 frozen" in r for r in b["dvb"])


# =========================================================================== #
# 3. two steps under planner,map, and the perception-trainable red arm        #
# =========================================================================== #
def _setup_pm(tmp_path):
    model, cfg = R._build()
    src_path = tmp_path / "ckpt_ema_src.pt"
    R._save_source(model, src_path)
    args = R._args(r8_train_only_map=False, r9_train_only="planner,map", init_from=str(src_path))
    rec = M.apply_train_only(model, M.r9_sets(args))
    mods = tuple(rec["modules"])
    opt = T.build_optimizer(model, args)
    T.assert_optimizer_matches_recipe(opt, args, model=model)
    src, srec = M.load_source(str(src_path))
    ver = M.verify_against_source(model, src, fresh=True, modules=mods)
    ema = W.WeightEMA(model, 0.9, exclude=M.ema_exclude_for(mods))
    assert M.ema_problems(ema, model, src, fresh=True, modules=mods) == []
    guard = M.FrozenStateGuard(src, modules=mods)
    return model, cfg, args, rec, mods, opt, src, ver, ema, guard


def test_TWO_STEPS_under_planner_map_the_perception_is_bit_identical_and_both_sets_move(tmp_path):
    model, cfg, args, rec, mods, opt, src, ver, ema, guard = _setup_pm(tmp_path)
    assert mods == PM_MODULES and ver["result"] == "bit-identical" and ver["n_trainable_keys"] == 67
    assert [g.get("name") for g in opt.param_groups] == ["train_only"]
    assert sum(int(p.numel()) for p in opt.param_groups[0]["params"]) == 61_009
    before = R._clone_state(model)
    assert R._steps(model, cfg, opt, ema, n=2) == [RIG_PM_GRAD_KEYS, RIG_PM_GRAD_KEYS]
    after = model.state_dict()
    moved = [k for k in before if not M.bit_equal(before[k], after[k])]
    assert [k for k in moved if not M.is_trainable_under(k, PM_MODULES)] == []
    assert len([k for k in moved if k.startswith("core.decoder.")]) == 47          # every planner tensor moved
    assert len([k for k in moved if M.is_map_only(k)]) == 18                       # every map tensor moved
    assert all(p.grad is None for n, p in model.named_parameters() if not M.is_trainable_under(n, PM_MODULES))
    assert guard.changed(model) == []
    assert M.bit_equal(before["core.lat_log_prior"], after["core.lat_log_prior"])  # the class-prior write skipped
    assert str(after["core.encoder.net.bn1.running_mean"][0].item()) == "-0.0"    # frozen BN stats untouched
    ema_sd = ema.model_state_dict(model)
    assert [k for k in src if not M.is_trainable_under(k, PM_MODULES) and not M.bit_equal(ema_sd[k], src[k])] == []
    assert int(ema.num_updates) == 2


def test_DELIBERATE_REGRESSION_one_perception_tensor_trainable_goes_RED(tmp_path):
    model, cfg, args, rec, mods, opt, src, ver, ema, guard = _setup_pm(tmp_path)
    p = dict(model.named_parameters())[PERCEPTION_TENSOR]
    p.requires_grad_(True)
    opt.param_groups[0]["params"].append(p)
    probs = M.optimizer_problems(opt, model, args)
    assert any("OUTSIDE the declared sets ['planner', 'map'] are trainable" in b for b in probs)
    dvb = [str(m) for m in M.dvb_check_r9(model, args)]
    assert any("the perception freeze is broken" in m and PERCEPTION_TENSOR in m for m in dvb)
    R._steps(model, cfg, opt, ema, n=2)
    assert guard.changed(model) == [PERCEPTION_TENSOR]
    with pytest.raises(SystemExit, match="the frozen perception is no longer the shipped perception"):
        guard.assert_ok(model, "step 2")


def test_DELIBERATE_REGRESSION_the_planner_frozen_too_is_refused_by_the_optimizer_and_G_DVB(tmp_path):
    model, cfg, args, rec, mods, opt, src, ver, ema, guard = _setup_pm(tmp_path)
    for n, q in model.named_parameters():
        if n.startswith("core.decoder."):
            q.requires_grad_(False)
    assert "a tensor in the optimizer is frozen" in M.optimizer_problems(opt, model, args)
    assert any("47 frozen" in str(m) for m in M.dvb_check_r9(model, args))


# =========================================================================== #
# 4. `--r9-train-only map` == `--r8-train-only-map`, bit for bit              #
# =========================================================================== #
def test_the_map_set_IS_the_map_only_fine_tune_bit_for_bit_after_two_steps(tmp_path):
    runs = {}
    for tag in ("r8", "r9"):
        model, cfg = R._build()
        src_path = tmp_path / f"src_{tag}.pt"
        R._save_source(model, src_path)
        if tag == "r8":
            args = R._args(init_from=str(src_path))
            rec = M.apply_freeze(model)
            ema = W.WeightEMA(model, 0.9, exclude=M.ema_exclude)
        else:
            args = R._args(r8_train_only_map=False, r9_train_only="map", init_from=str(src_path))
            rec = M.apply_train_only(model, M.r9_sets(args))
            assert M.ema_exclude_for(rec["modules"]) is M.ema_exclude
            ema = W.WeightEMA(model, 0.9, exclude=M.ema_exclude_for(rec["modules"]))
        opt = T.build_optimizer(model, args)
        assert T.assert_optimizer_matches_recipe(opt, args, model=model)
        keys = R._steps(model, cfg, opt, ema, n=2)
        runs[tag] = (model, rec, ema, keys, [n for n, p in model.named_parameters() if p.requires_grad])
    (m8, r8, e8, k8, t8), (m9, r9, e9, k9, t9) = runs["r8"], runs["r9"]
    assert k8 == k9 == [["loss", "map_hires"], ["loss", "map_hires"]] and t8 == t9
    assert r9["trainable_leaf_groups"] == r8["trainable_leaf_groups"] == R.TINY_GROUPS
    assert r9["n_trainable_params"] == r8["n_trainable_params"] == R.TINY_NUMEL
    assert r9["frozen_norms_pinned"] == r8["frozen_norms_pinned"] == 20
    s8, s9 = m8.state_dict(), m9.state_dict()
    assert list(s8) == list(s9) and [k for k in s8 if not M.bit_equal(s8[k], s9[k])] == []
    x8, x9 = e8.model_state_dict(m8), e9.model_state_dict(m9)
    assert [k for k in x8 if not M.bit_equal(x8[k], x9[k])] == []
    # the record differs ONLY in the fields that name the flag (schema, flag, sets, per_set, group)
    assert r8["flag"] == "--r8-train-only-map" and r9["flag"] == "--r9-train-only"


# =========================================================================== #
# 5. argv, the trainer's pin, G-DVB                                           #
# =========================================================================== #
_ON = ["--arm", "hier", "--out", "X", "--refcv8", "--map-hires", "on", "--map-hires-near-lift-m", "20",
       "--map-hires-near-refine-blocks", "1", "--w-map-hires", "1.0", "--opt", "dd", "--init-from", "e.pt",
       "--conflict-detector", "off", "--r9-train-only", "planner,map"]


def _pin(argv):
    M.pin_args(T.build_parser().parse_args(argv))


def test_GREEN_the_conformant_argv_pins_and_the_planner_set_alone_needs_no_map_head():
    _pin(_ON)
    _pin(R._drop(R._drop(R._drop(R._drop(R._drop(_ON, "--map-hires"), "--map-hires-near-lift-m"),
                                     "--map-hires-near-refine-blocks"), "--w-map-hires"), "--r9-train-only")
         + ["--r9-train-only", "planner"])
    _pin(["--arm", "hier", "--out", "X", "--b1-guard-ref-ap2m", "0.2"])          # off: nothing is checked


@pytest.mark.parametrize("argv,needle", [
    (R._drop(_ON, "--refcv8", 0), "--refcv8 is not set"),
    (R._drop(_ON, "--map-hires"), "the `map` set has no 10 cm map head"),
    (R._drop(_ON, "--map-hires-near-lift-m"), "`near` module of the `map` set"),
    (R._drop(_ON, "--map-hires-near-refine-blocks"), "`near_refine` module of the `map` set"),
    (R._drop(_ON, "--w-map-hires"), "--w-map-hires is 0"),
    (R._drop(_ON, "--opt"), "--opt is not dd"),
    (R._drop(_ON, "--init-from"), "--init-from is not set"),
    (R._drop(_ON, "--conflict-detector"), "--conflict-detector is not `off`"),
    (_ON + ["--conflict-every", "50"], "--conflict-every"),
    (_ON + ["--b1-guard-ref-ap2m", "0.21211"], "B1 guard"),
    (_ON + ["--grad-share-every", "50"], "--grad-share-every"),
    (_ON + ["--trunk-bn-recalib", "64"], "--trunk-bn-recalib"),
    (_ON + ["--r8-critic-drivable"], "--r8-critic-drivable"),
    (_ON + ["--r9-calib-stage"], "R9-PF calibration stage"),
    (_ON + ["--r8-train-only-map"], "ONE freeze per run"),
    (R._drop(_ON, "--r9-train-only") + ["--r9-train-only", "planner,tactical"], "unknown set")])
def test_every_refusal_of_the_flag(argv, needle):
    with pytest.raises(SystemExit, match=needle.replace("(", r"\(").replace("[", r"\[")):
        _pin(argv)


def test_the_trainers_own_pin_reaches_the_refusals():
    from tanitad.refs import refcv8_conditioning as r8c
    args = T.build_parser().parse_args(R._drop(_ON, "--init-from"))
    with pytest.raises(SystemExit, match="--init-from is not set"):
        T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)
    args = T.build_parser().parse_args(R._drop(_ON, "--r9-train-only") + ["--r9-train-only", "planner",
                                                                           "--w-map-fp", "0.5"])
    with pytest.raises(SystemExit, match="not the map head"):                  # --w-map-fp's pin, reached too
        T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)


def test_GDVB_the_lever_reads_the_built_freeze_both_ways():
    assert D.REGISTRY["r9_train_only"].kind == "built" and D.REGISTRY["r8_train_only_map"].kind == "built"
    on = R._args(r8_train_only_map=False, r9_train_only="planner,map")
    off = R._args(r8_train_only_map=False, r9_train_only=None)
    model, _ = R._build()
    assert M.dvb_check_r9(model, off) == [] and M.dvb_check(model, off) == []          # off, not frozen: clean
    assert "no train-only freeze" in str(M.dvb_check_r9(model, on)[0])                # declared, not built: RED
    M.apply_train_only(model, ("planner", "map"))
    assert M.dvb_check_r9(model, on) == []
    assert M.dvb_check(model, on) == []          # the r8 lever defers to the r9 one on an r9 freeze
    assert "a train-only freeze" in str(M.dvb_check_r9(model, off)[0])                # built, not declared: RED
    wrong = R._args(r8_train_only_map=False, r9_train_only="map")
    assert "the built freeze is not the declared one" in str(M.dvb_check_r9(model, wrong)[0])
    # the grad-unreachable declaration reads the record's modules: everything outside them is declared frozen
    assert [m for m in D.check_grad_unreachable(model, on) if m.lever == "frozen parameters"] == []


# =========================================================================== #
# 6. the refcv9-warm launch-gate profile                                      #
# =========================================================================== #
R9W = LG.PROFILES["refcv9-warm"]
LAUNCH = ROOT / "ops" / "runs.d" / "refcv8-r101-s0.argv.json"
THOR_INIT = "/home/nvidia/refcv9_warm/ckpt_ema_48852.pt"


def test_the_profile_literals():
    mo = R9W["map_only"]
    assert mo["flag"] == "--r9-train-only" and mo["modules"] == PM_MODULES
    assert mo["numel"] == 15_470_833 and len(mo["groups"]) == 111 and sum(mo["groups"].values()) == 15_470_833
    assert mo["init_from_md5"] == SHIPPED_EMA_MD5
    assert {g: n for g, n in mo["groups"].items() if g.startswith("_map_hires.")} == LG._MAPFT_GROUPS
    assert sum(n for g, n in mo["groups"].items() if g.startswith("core.decoder")) == 15_365_801
    assert mo["groups"]["core.decoder"] == 2                     # navc_gate + r8_rc_gate (direct parameters)
    assert R9W["required_checks"] == ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE",
                                      "G-SUITE-PINNED", "G-MAP-OVERFIT")
    for f in ("--r9-train-only", "--init-from", "--conflict-detector", "--w-map-fp", "--r9-alloc-force-schedule",
              "--r9-alloc-force-seed"):
        assert f in R9W["required_flags"], f
    rv = R9W["required_values"]
    assert (rv["--r9-train-only"], rv["--conflict-detector"], rv["--r9-alloc-force-seed"]) == (
        "planner,map", "off", "20261027")
    for f in ("--steps", "--warmup", "--ema-decay", "--eval-every", "--b1-guard-ref-ap2m", "--b1-guard-step",
              "--w-map-fp", "--r9-alloc-force-schedule"):
        assert f not in rv, f
    assert "--w-map-fp" in R9W["required_positive"]
    forb = {f for e in R9W["forbidden_levers"] for f in e[0]}
    for f in ("--r8-train-only-map", "--b1-guard-ref-ap2m", "--grad-share-every", "--conflict-every",
              "--trunk-bn-recalib", "--r8-critic-drivable", "--r9-progress-head", "--r9-calib-stage", "--w-r9-prog",
              "--w-r9-regime", "--r9-decision", "--r9-hidden", "--r9-decel-tau", "--r9-seed"):
        assert f in forb, f
    assert [(g, f) for g, f, _w in R9W["live_dead_admitted"]] == [
        ("core.decoder.ctx_to_cond", "--no-strategic"), ("core.decoder.lat_to_anchor", "--graft-tac8-prior"),
        ("core.decoder.lon_to_anchor", "--graft-tac8-prior"), ("core.decoder.r8_rc_to_cond", "--r8-no-rc"),
        ("core.decoder.r8_rc_gate", "--r8-no-rc")]
    assert [o["id"] for o in R9W["open_items"]] == ["R9W-TACTICAL", "R9W-BUDGET", "R9W-SCHEDULE", "R9W-MAP-FP-WEIGHT",
                                                    "R9W-TF-RESTART", "R9W-EMIT-START"]
    assert R9W["map_hires"]["ga_keys"] == ("ga_mh_refine", "ga_mh_near", "ga_mh_near_refine")
    assert R9W["map_hires"]["ga_keys_zero"] == ("ga_mh_lift", "ga_mh_encoder", "ga_mh_trunk_s8_stage")
    assert LG.REFCV9_ARMS == ("refcv9_perception_trainable", "refcv9_planner_frozen")
    # the open items BLOCK a pass (finalize reads them); refcv8-mapft is untouched
    assert len(LG.open_item_reasons(R9W)) == 6 and LG.open_item_reasons(LG.PROFILES["refcv8-mapft"]) == []
    assert LG.PROFILES["refcv8-mapft"]["map_only"]["flag"] == "--r8-train-only-map"


@pytest.fixture(scope="module")
def r9_launch(tmp_path_factory):
    """refcv8-r101-s0's committed launch argv -> the refcv9 warm-start launch (the budget, schedule and weight values
    are placeholders: R9W-BUDGET / -SCHEDULE / -MAP-FP-WEIGHT are open)."""
    d = json.loads(LAUNCH.read_text(encoding="utf-8"))
    a = list(d["argv"])
    for f, v in {"--r9-train-only": ["planner,map"], "--init-from": [THOR_INIT], "--conflict-detector": ["off"],
                 "--w-map-fp": ["0.5"], "--r9-alloc-force-schedule": ["linear:1.0:0.0:0:1500"],
                 "--r9-alloc-force-seed": ["20261027"],
                 "--b1-guard-ref-ap2m": None, "--b1-guard-step": None, "--grad-share-every": None,
                 "--grad-share-terms": None, "--grad-share-at": None, "--conflict-every": None,
                 "--steps": ["3000"], "--warmup": ["200"], "--ema-decay": ["0.999"]}.items():
        a = R._flag_set(a, f, v)
    tmp = tmp_path_factory.mktemp("r9w")
    init = tmp / "ckpt_ema_48852.pt"
    init.write_bytes(b"stand-in for the shipped EMA model")
    rec = tmp / "nav_tau.json"
    rec.write_text(json.dumps({"status": "OK", "tau": float(LG.flag_values(a, "--nav-compliance-tau-rad")[0]),
                               "split_v2_cache": LG.flag_values(a, "--v2-cache")[0],
                               "labels": LG.flag_values(a, "--v7-labels")[0]}), encoding="utf-8")
    prof = dict(R9W, map_only=dict(R9W["map_only"], init_from_md5=hashlib.md5(init.read_bytes()).hexdigest()))
    return a, prof, {"nav_tau_record": str(rec)}, [(THOR_INIT.rsplit("/", 1)[0], str(tmp))], init


def test_GATE_the_warm_start_launch_argv_has_no_refusal(r9_launch):
    a, prof, opts, pmap, _ = r9_launch
    refusals, pending = LG.profile_argv_rules(prof, a, opts, pmap)
    assert refusals == [] and pending == [], refusals
    # the trainer's own argv pins accept it too
    args = T.build_parser().parse_args(a)
    M.pin_args(args)
    from tanitad.train import map_precision as MP
    MP.pin_args(args)


@pytest.mark.parametrize("flag,vals,needle", [
    ("--r9-train-only", None, "--r9-train-only"),
    ("--r9-train-only", ["map"], "--r9-train-only"),
    ("--w-map-fp", None, "--w-map-fp"),
    ("--w-map-fp", ["0"], "--w-map-fp"),
    ("--r9-alloc-force-schedule", None, "--r9-alloc-force-schedule"),
    ("--conflict-detector", ["auto"], "--conflict-detector off is REQUIRED"),
    ("--b1-guard-ref-ap2m", ["0.21211"], "B1 guard"),
    ("--grad-share-every", ["250"], "gradient-share"),
    ("--conflict-every", ["50"], "dead cadence"),
    ("--trunk-bn-recalib", ["64"], "REWRITES"),
    ("--r8-train-only-map", [], "refcv8-mapft"),
    ("--r9-progress-head", [], "R9-PF is the SEPARATE stage-2"),
    ("--r9-calib-stage", [], "R9-PF is the SEPARATE stage-2")])
def test_GATE_each_rule_refuses(r9_launch, flag, vals, needle):
    a, prof, opts, pmap, _ = r9_launch
    refusals, _p = LG.profile_argv_rules(prof, R._flag_set(a, flag, vals), opts, pmap)
    assert any(needle in r for r in refusals), refusals


def test_GATE_the_warm_start_is_bound_by_content(r9_launch, tmp_path):
    a, prof, opts, pmap, init = r9_launch
    other = tmp_path / "ckpt_ema_other.pt"
    other.write_bytes(b"a different checkpoint")
    r, _ = LG.map_only_init_reasons(prof, R._flag_set(a, "--init-from", [str(other)]), pmap)
    assert len(r) == 1 and "the profile pins" in r[0] and r[0].startswith("--r9-train-only:")
    r, _ = LG.map_only_init_reasons(R9W, a, pmap)                # the REAL literal against a stand-in: refused
    assert len(r) == 1 and SHIPPED_EMA_MD5 in r[0]


def _tiny_r9w():
    return dict(R9W, map_only=dict(R9W["map_only"], groups=dict(RIG_PM_GROUPS), numel=61_009))


def test_GATE_G_DVB_reads_the_built_set_and_both_RED_arms_go_RED_on_the_rig():
    prof, argv = _tiny_r9w(), ["--r9-train-only", "planner,map"]
    model, _ = R._build()
    M.apply_train_only(model, ("planner", "map"))
    r, det = LG.map_only_built_reasons(prof, argv, model)
    assert r == [] and det["trainable_numel"] == 61_009
    r, _ = LG.map_only_built_reasons(R9W, argv, model)            # the REAL literal against the rig: RED
    assert any("15470833" in x for x in r)
    on = R._args(r8_train_only_map=False, r9_train_only="planner,map")
    # arm 1: perception trainable
    note = LG._refcv9_inject(type("C", (), {"arm": "refcv9_perception_trainable"})(), model, None, PM_MODULES)
    assert note.startswith("refcv9_perception_trainable: ") and "(first core.encoder." in note
    r, _ = LG.map_only_built_reasons(prof, argv, model)
    assert any("parameter(s) OUTSIDE" in x for x in r)
    assert any("the perception freeze is broken" in str(m) for m in M.dvb_check_r9(model, on))
    # arm 2: the planner frozen
    model2, _ = R._build()
    M.apply_train_only(model2, ("planner", "map"))
    note = LG._refcv9_inject(type("C", (), {"arm": "refcv9_planner_frozen"})(), model2, None, PM_MODULES)
    assert note == "refcv9_planner_frozen: 47 planner (core.decoder) tensors frozen (the --r9-train-only planner set)"
    r, _ = LG.map_only_built_reasons(prof, argv, model2)
    assert any("47 map-only parameter(s) are frozen" in x for x in r) and any("17960" in x for x in r)
    assert any("47 frozen" in str(m) for m in M.dvb_check_r9(model2, on))
    # any other arm: inert
    assert LG._refcv9_inject(type("C", (), {"arm": "mapft_map_frozen"})(), model2, None, PM_MODULES) is None


def test_GATE_the_build_arm_installs_at_the_trainers_freeze():
    ctx = types.SimpleNamespace(arm="refcv9_perception_trainable", prof=R9W)
    fakeT = types.SimpleNamespace(_mof=M)
    pat = LG._Patcher()
    try:
        assert LG.apply_refcv9_build_arm(ctx, pat, fakeT) == ("refcv9_perception_trainable: injected after the "
                                                              "trainer's --r9-train-only freeze")
        model, _ = R._build()
        M.apply_train_only(model, ("planner", "map"))           # the trainer's call, through the patched module
        assert any(p.requires_grad for n, p in model.named_parameters() if n.startswith("core.encoder."))
    finally:
        pat.restore()
    model, _ = R._build()
    M.apply_train_only(model, ("planner", "map"))               # restored: the freeze holds
    assert not any(p.requires_grad for n, p in model.named_parameters() if n.startswith("core.encoder."))
    assert LG.apply_refcv9_build_arm(types.SimpleNamespace(arm=None, prof=R9W), LG._Patcher(), fakeT) is None


def _grads(dead=()):
    return {"groups": [{"group": g, "numel": n, "params": 1, "nz_params": 0 if g in dead else 1}
                       for g, n in RIG_PM_GROUPS.items()], "nonfinite_params": []}


def test_GATE_G_LIVE_the_admitted_dead_rule_and_its_mutation():
    prof = _tiny_r9w()
    rec = R._live_rec()
    assert LG.judge_map_only_live(prof, rec, _grads())[0] == []
    # a trainable decoder group dead BY CONSTRUCTION (--no-strategic): RED unless admitted
    r, _ = LG.judge_map_only_live(prof, rec, _grads(dead=("core.decoder.ctx_to_cond",)))
    assert len(r) == 1 and "received NO non-zero gradient" in r[0] and "core.decoder.ctx_to_cond" in r[0]
    adm = {"core.decoder.ctx_to_cond": "the strategic context (no --strategic) [admitted while --no-strategic ...]"}
    r, det = LG.judge_map_only_live(prof, rec, _grads(dead=("core.decoder.ctx_to_cond",)), admitted_dead=adm)
    assert r == [] and det["dead_admitted"] == adm
    # MUTATION: the admission names ANOTHER group -> the dead one is RED again; a dead MAP group is never admitted by it
    r, _ = LG.judge_map_only_live(prof, rec, _grads(dead=("core.decoder.ctx_to_cond",)),
                                  admitted_dead={"core.decoder.lat_to_anchor": "x"})
    assert len(r) == 1 and "core.decoder.ctx_to_cond" in r[0]
    r, _ = LG.judge_map_only_live(prof, rec, _grads(dead=("_map_hires.refine.cls",)), admitted_dead=adm)
    assert len(r) == 1 and "_map_hires.refine.cls" in r[0]
    # refcv8-mapft: refcv8's admitted rows are all OUTSIDE its map-only modules -> its judgement is unchanged
    mprof = R._tiny_prof()
    adm8 = {g: w for g, _f, w in LG.PROFILES["refcv8-mapft"]["live_dead_admitted"]}
    g8 = {"groups": [dict(e, nz_params=0) for e in R._grads_ok()["groups"]], "nonfinite_params": []}
    assert LG.judge_map_only_live(mprof, rec, g8, admitted_dead=adm8)[0] == LG.judge_map_only_live(mprof, rec, g8)[0]
    # the G-LIVE call site passes the profile's admitted rows (the same dict judge_live reads)
    src = inspect.getsource(LG)
    assert 'judge_map_only_live(ctx.prof, run1.get("map_only"), gt, admitted_dead=admitted)' in src


def test_GATE_G_LIVE_the_two_RED_arms_records():
    prof = _tiny_r9w()
    # refcv9_perception_trainable at G-LIVE: a frozen (perception) tensor receives a gradient and moves
    r, _ = LG.judge_map_only_live(prof, R._live_rec(frozen_grad_nz={PERCEPTION_TENSOR: 2},
                                                    frozen_changed=[PERCEPTION_TENSOR]), _grads())
    assert any("FROZEN parameter(s) received a non-zero gradient" in x for x in r)
    assert any("CHANGED over the live steps" in x for x in r)
    # refcv9_planner_frozen at G-LIVE: the 25 planner groups stay in the optimizer and receive nothing
    r, _ = LG.judge_map_only_live(prof, R._live_rec(), _grads(dead=tuple(RIG_PLANNER_GROUPS)))
    assert len(r) == 1 and "25 map-only group(s) received NO non-zero gradient" in r[0]
