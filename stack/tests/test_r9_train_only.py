"""``--r9-train-only <set>[,<set>]`` -- the refcv9 WARM START's freeze (PI decisions 2026-10-27: warm start from refcv8
EMA-48,852, perception FROZEN, the planner + tactical heads and the map head trained) and the ``refcv9-warm``
launch-gate profile.

The sets are LITERAL module lists: ``planner`` = ``core.decoder`` (EXACTLY what the launch gate's
``mapft_planner_trainable`` arm un-freezes), ``tactical`` = the eleven tactical-branch modules (MM 2026-10-27,
R9W-TACTICAL), ``map`` = the map-only fine-tune's three modules. A subtree the BUILD declared grad-unreachable
(``tanitad/models/_gradreach.py``) inside a set stays FROZEN -- the round-2 freeze re-enabled five of them on refcv8
(186 / 15,365,801 -> 178 / 15,254,091 trained). ``--r9-train-only map`` trains what ``--r8-train-only-map`` trains, bit
for bit. Every expectation is a LITERAL authored here; every guard carries a DELIBERATE-REGRESSION arm that must go RED:

  1. the sets, their parsing and refusals;
  2. the trainable / frozen COUNTS -- on the map CI rig (test_map_only_ft's tiny REAL RefCV3Model: map / planner /
     planner,map; it has no tactical decoder, so `tactical` is REFUSED there), on the refcv8 CI rig
     (tests/_refcv8_rig.py: tactical / planner / planner,tactical, with its declared subtrees) AND on the REAL
     refcv8-r101-s0 build (the eval kit's config.json + ckpt_ema_48852.pt through refcv7_loader.build_model on CPU, in a
     subprocess; skipped when the kit is absent);
  3. the declared grad-unreachable subtrees stay frozen (RED: the round-2 defect re-introduced -- offset_head trainable
     -- fires the freeze's G-DVB lever, the optimizer check and the build's own grad-unreachable check);
  4. TWO optimiser steps under planner,map (map rig) and planner,tactical (refcv8 rig): every frozen parameter AND
     buffer bit-identical, the named tensors moved; RED: one perception tensor trainable;
  5. ``--r9-train-only map`` == ``--r8-train-only-map`` after two steps, bit for bit;
  6. the argv refusals, the trainer's pin reaching them, R9W-EMIT-START (``--r8-alloc-emit-start 0`` admitted under
     --r9-train-only only), R9W-TF-RESTART (the ratio held at refcv8's end value), the G-DVB lever both ways;
  7. the ``refcv9-warm`` profile: its literals, the launch argv, each rule's refusal (R9W-SCHEDULE's end step read off
     --steps), the four refcv8-family profiles refusing the flag, the env-failure pin (refcv8-mapft's copy, B5), the
     init md5 binding, the declared-subtree literal,
     the two RED arms at G-DVB (rigs + real build) and at G-LIVE, the admitted-dead rule, the open items;
  8. ``assert_map_only_finetune.py --sets``: the perception bit-equal, the changes named per set, ``--also-frozen``,
     ``--require-each-module-moved``; the legacy form unchanged.
"""
from __future__ import annotations

import copy
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

import assert_map_only_finetune as AT  # noqa: E402
import launch_gate as LG  # noqa: E402
import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402  -- the CI map rig (_build / _batch / _args / _steps)
import _refcv8_rig as R8  # noqa: E402  -- the CI refcv8 rig (tactical decoder v6, behaviour gate, refcv8 seams)
from tanitad.refs import refcv8_conditioning as r8c  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_only_ft as M  # noqa: E402
from tanitad.train import refcv8_train as r8train  # noqa: E402
from tanitad.train import weight_ema as W  # noqa: E402

MAP_MODULES = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
TAC_MODULES = ("phi_tac", "nav_to_tac", "lat_head_tac", "lon_head_tac", "tac_goal_head", "tac_latent_proj",
               "tac_decoder_v6", "tac_behaviour_gate_v6", "core.tactical_trunk", "core.lat_head", "core.lon_head")
PM_MODULES = ("core.decoder",) + MAP_MODULES
PT_MODULES = ("core.decoder",) + TAC_MODULES
PTM_MODULES = ("core.decoder",) + TAC_MODULES + MAP_MODULES
#: the declared grad-unreachable subtrees INSIDE `planner` on the REAL refcv8 build (MEASURED 2026-10-27; 8 tensors,
#: 111,710 values): offset_head 2 / 6,160, ctx_to_cond 2 / 98,688, lat_to_anchor 1 / 351, lon_to_anchor 1 / 351,
#: control_head 2 / 6,160
REAL_DECLARED = ("core.decoder.control_head", "core.decoder.ctx_to_cond", "core.decoder.lat_to_anchor",
                 "core.decoder.lon_to_anchor", "core.decoder.offset_head")
#: the map CI rig (test_map_only_ft._build: resnet18 at 64x64, d_model 32, 2 decoder layers, the 10 cm branch), MEASURED
#: 2026-10-27 with apply_train_only: 220 parameter tensors, 12,341,401 values; 20 BatchNorms (all frozen in each set);
#: its one declared subtree (scorer.goal_point) is OUTSIDE every set, so round 3 leaves these counts unchanged
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
#: the refcv8 CI rig (R8.build(T, True, seed=0, n_alloc=8, w_alloc_l1=1.0, w_sat=0.1)), MEASURED 2026-10-27: 343
#: parameter tensors, 14,767,695 values; it declares core.decoder.offset_head + core.decoder.control_head (4 tensors,
#: 1,056 values, INSIDE `planner`) and scorer.goal_point (outside every set); 20 BatchNorms pinned in each set
R8RIG_TOTAL = (343, 14_767_695)
R8RIG_DECLARED = ("core.decoder.control_head", "core.decoder.offset_head")
R8RIG_COUNTS = {          # sets -> (trainable tensors, values, frozen tensors, values, leaf groups, declared t / v)
    ("tactical",): (85, 52_978, 258, 14_714_717, 44, 0, 0),
    ("planner",): (88, 68_322, 255, 14_699_373, 46, 4, 1_056),
    ("planner", "tactical"): (173, 121_300, 170, 14_646_395, 90, 4, 1_056)}
R8RIG_ARGV = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
              "--n-anchors", "20", "--ego-history", "--residual-prior", "ha0_ext_pose", "--out", "X"] + R8.F1_F6
#: two steps under planner,tactical on the refcv8 rig (MEASURED 2026-10-27): module -> (moved, trained) tensors. The
#: five that do not move: core.decoder.r8_rc_gate / r8_rc_to_cond.{weight,bias} (--r8-no-rc: G-LIVE admits them),
#: tac_decoder_v6.r8_film.0.proj.weight and tac_behaviour_gate_v6.proj.weight (zero-init grafts; on the rig's batch
#: their gradient is zero -- the real build's G-LIVE smoke judges them)
R8RIG_PT_MOVED = {"core.decoder": (85, 88), "phi_tac": (24, 24), "nav_to_tac": (2, 2), "lat_head_tac": (2, 2),
                  "lon_head_tac": (2, 2), "tac_goal_head": (2, 2), "tac_latent_proj": (2, 2),
                  "tac_decoder_v6": (43, 44), "tac_behaviour_gate_v6": (0, 1), "core.tactical_trunk": (2, 2),
                  "core.lat_head": (2, 2), "core.lon_head": (2, 2)}
R8RIG_PT_UNMOVED = ["core.decoder.r8_rc_gate", "core.decoder.r8_rc_to_cond.bias", "core.decoder.r8_rc_to_cond.weight",
                    "tac_behaviour_gate_v6.proj.weight", "tac_decoder_v6.r8_film.0.proj.weight"]
R8RIG_PT_GRAD_KEYS = ["cascade", "cls", "goal_tac", "lat", "lat_tac", "law", "lon", "lon_tac", "loss", "r8", "sel_v3",
                      "tac_v6", "traj"]
#: the REAL refcv8-r101-s0 build (CPU, refcv7_loader.build_model on the shipped EMA file), MEASURED 2026-10-27:
#: 848 parameter tensors, 100,598,182 values
REAL_TOTAL = (848, 100_598_182)
REAL_COUNTS = {           # spec -> (trainable tensors, values, frozen tensors, values, declared tensors, values)
    "map": (18, 105_032, 830, 100_493_150, 0, 0),
    "planner": (178, 15_254_091, 670, 85_344_091, 8, 111_710),
    "tactical": (113, 5_476_306, 735, 95_121_876, 0, 0),
    "planner,tactical,map": (309, 20_835_429, 539, 79_762_753, 8, 111_710),
    "planner,map": (196, 15_359_123, 652, 85_239_059, 8, 111_710)}
#: the tactical set on the REAL build, per module (tensors, values) -- MEASURED 2026-10-27
REAL_TACTICAL = {"phi_tac": (24, 2_101_504), "nav_to_tac": (2, 33_280), "lat_head_tac": (2, 4_104),
                 "lon_head_tac": (2, 4_104), "tac_goal_head": (2, 6_156), "tac_latent_proj": (2, 262_656),
                 "tac_decoder_v6": (72, 2_272_802), "tac_behaviour_gate_v6": (1, 2_574),
                 "core.tactical_trunk": (2, 786_816), "core.lat_head": (2, 1_155), "core.lon_head": (2, 1_155)}
KIT = Path("E:/refcv8_eval_kit")
KIT_CFG, KIT_EMA = KIT / "ckpt" / "config.json", KIT / "ckpt" / "ckpt_ema_48852.pt"
SHIPPED_EMA_MD5 = "b4364029b68f29a33eeb4aebbbbd002b"
#: the losses that carry a gradient on the map rig under planner,map (MEASURED): the planner's and the map head's
RIG_PM_GRAD_KEYS = ["cls", "law", "loss", "map_hires", "sel_v3", "traj"]
PERCEPTION_TENSOR = "core.encoder.net.conv1.weight"


# =========================================================================== #
# 1. the sets                                                                 #
# =========================================================================== #
def test_the_sets_are_literal_module_lists():
    assert M.TRAIN_ONLY_SETS == {"planner": ("core.decoder",), "tactical": TAC_MODULES, "map": MAP_MODULES}
    assert list(M.TRAIN_ONLY_SETS) == ["planner", "tactical", "map"]
    assert M.R9_FLAG == "--r9-train-only" and M.R9_DEST == "r9_train_only" and M.R9_GROUP_NAME == "train_only"
    assert M.parse_sets("planner,map") == ("planner", "map") == M.parse_sets("map,planner")
    assert M.parse_sets("map,tactical,planner") == ("planner", "tactical", "map")
    assert M.parse_sets("map") == ("map",) and M.parse_sets(" tactical ") == ("tactical",)
    assert M.modules_of_sets(("planner", "map")) == PM_MODULES
    assert M.modules_of_sets(("planner", "tactical", "map")) == PTM_MODULES
    for bad, needle in (("", "empty set name"), ("planner,", "empty set name"), ("tactics", "unknown set"),
                        ("map,map", "named twice"), ("perception", "unknown set"), ("strategic", "unknown set")):
        with pytest.raises(SystemExit, match=needle):
            M.parse_sets(bad)
    # the gate's literal is the trainer's, written twice (never read across)
    assert LG._R9W_MODULES == PTM_MODULES and LG._R9W_SETS == "planner,tactical,map"
    assert LG._R9W_TACTICAL == TAC_MODULES
    assert (M.REFCV8_PLANNER_TENSORS, M.REFCV8_PLANNER_NUMEL, M.REFCV8_MAP_ONLY_TENSORS) == (178, 15_254_091, 18)
    assert (M.REFCV8_TACTICAL_TENSORS, M.REFCV8_TACTICAL_NUMEL) == (113, 5_476_306)
    assert tuple(sorted(M.REFCV8_PLANNER_DECLARED_FROZEN)) == REAL_DECLARED == LG._R9W_DECLARED_FROZEN
    # the gate reads the build's declaration attribute BY NAME: the name is _gradreach's
    from tanitad.models._gradreach import GRAD_UNREACHABLE_FLAG
    assert LG._GRAD_UNREACHABLE_ATTR == "_tanitad_grad_unreachable" == GRAD_UNREACHABLE_FLAG


# =========================================================================== #
# 2. the counts: the map rig, the refcv8 rig and the real refcv8 build        #
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
    assert rec["declared_frozen"] == {} and (rec["n_declared_frozen_tensors"], rec["n_declared_frozen_params"]) == (0, 0)
    want = {("map",): R.TINY_GROUPS, ("planner",): RIG_PLANNER_GROUPS, ("planner", "map"): RIG_PM_GROUPS}[sets]
    assert rec["trainable_leaf_groups"] == want
    model.train()
    assert [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training] == []


def test_CI_RIG_the_tactical_set_is_refused_on_a_build_without_the_tactical_decoder():
    model, _ = R._build()
    with pytest.raises(SystemExit, match=r"the built model has no \['tac_decoder_v6', 'tac_behaviour_gate_v6'\]"):
        M.apply_train_only(model, ("planner", "tactical", "map"))


def _r8rig():
    cfg, m = R8.build(T, True, seed=0, n_alloc=8, w_alloc_l1=1.0, w_sat=0.1)
    return cfg, m


@pytest.mark.parametrize("sets", list(R8RIG_COUNTS))
def test_R8_RIG_the_trainable_frozen_and_declared_counts_of_each_set(sets):
    _cfg, model = _r8rig()
    assert (sum(1 for _ in model.parameters()), sum(int(p.numel()) for p in model.parameters())) == R8RIG_TOTAL
    rec = M.apply_train_only(model, sets)
    got = (rec["n_trainable_tensors"], rec["n_trainable_params"], rec["n_frozen_tensors"], rec["n_frozen_params"],
           len(rec["trainable_leaf_groups"]), rec["n_declared_frozen_tensors"], rec["n_declared_frozen_params"])
    assert got == R8RIG_COUNTS[sets]
    assert sorted(rec["declared_frozen"]) == ([] if sets == ("tactical",) else list(R8RIG_DECLARED))
    assert rec["frozen_norms_pinned"] == 20
    # the declared subtrees are frozen, everything else inside the sets trainable
    for n, p in model.named_parameters():
        inside = any(n == m or n.startswith(m + ".") for s in sets for m in M.TRAIN_ONLY_SETS[s])
        decl = any(n.startswith(d + ".") for d in R8RIG_DECLARED)
        assert p.requires_grad == (inside and not decl), n


# =========================================================================== #
# 3. the declared grad-unreachable subtrees stay frozen                       #
# =========================================================================== #
def _r8rig_pt(tmp_path):
    cfg, m = _r8rig()
    src_path = tmp_path / "src.pt"
    torch.save({"model": {k: v.detach().clone() for k, v in m.state_dict().items()}, "step": 48852,
                "kind": "ema_model_only"}, src_path)
    args = T.build_parser().parse_args(R8RIG_ARGV + ["--r9-train-only", "planner,tactical", "--init-from",
                                                      str(src_path), "--lr", "1e-3", "--opt", "dd"])
    rec = M.apply_train_only(m, M.r9_sets(args))
    return cfg, m, args, rec, src_path


def test_DECLARED_the_built_declarations_inside_the_sets_stay_frozen_and_every_check_is_clean(tmp_path):
    cfg, m, args, rec, _src = _r8rig_pt(tmp_path)
    assert sorted(rec["declared_frozen"]) == list(R8RIG_DECLARED)
    assert [str(x) for x in D.check_grad_unreachable(m, args)] == []
    assert [str(x) for x in M.dvb_check_r9(m, args)] == []
    opt = T.build_optimizer(m, args)
    assert M.optimizer_problems(opt, m, args) == []
    assert sum(int(p.numel()) for p in opt.param_groups[0]["params"]) == 121_300


def test_DELIBERATE_REGRESSION_the_round2_defect_a_declared_subtree_trainable_again_goes_RED(tmp_path):
    """aeb5446's apply_train_only trained EVERY parameter under core.decoder, the build's declared subtrees included."""
    cfg, m, args, rec, _src = _r8rig_pt(tmp_path)
    opt = T.build_optimizer(m, args)
    for n, p in m.named_parameters():
        if n.startswith("core.decoder.offset_head."):
            p.requires_grad_(True)
    assert any("a declared subtree is trainable again" in str(x) and "core.decoder.offset_head." in str(x)
               for x in M.dvb_check_r9(m, args))
    assert any("2 DECLARED grad-unreachable parameter(s) are trainable again" in b
               for b in M.optimizer_problems(opt, m, args))
    assert any("declared frozen but trainable again" in str(x) for x in D.check_grad_unreachable(m, args))


# =========================================================================== #
# (real build)                                                                #
# =========================================================================== #
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
from tanitad.train import declared_vs_built as D
import launch_gate as LG
out = {"tanitad": tanitad.__file__, "dvb_build": [str(m) for m in (rec.get("declared_vs_built") or {}).get("mismatches", [])],
       "total": [sum(1 for _ in model.parameters()), sum(int(p.numel()) for p in model.parameters())], "counts": {}}
for spec in ("map", "planner", "tactical", "planner,map", "planner,tactical,map"):
    r = M.apply_train_only(model, tuple(spec.split(",")))
    out["counts"][spec] = [r["n_trainable_tensors"], r["n_trainable_params"], r["n_frozen_tensors"],
                           r["n_frozen_params"], r["n_declared_frozen_tensors"], r["n_declared_frozen_params"],
                           sorted(r["declared_frozen"]), r["per_set"]]
numel = {n: int(p.numel()) for n, p in model.named_parameters()}
out["per_module"] = {m: [sum(1 for n in r["trainable"] if n == m or n.startswith(m + ".")),
                         sum(numel[n] for n in r["trainable"] if n == m or n.startswith(m + "."))]
                     for m in M.TRAIN_ONLY_SETS["tactical"]}
prof = LG.PROFILES["refcv9-warm"]
argv = ["--r9-train-only", "planner,tactical,map"]
args.r9_train_only, args.r8_train_only_map = "planner,tactical,map", False
out["groups"] = r["trainable_leaf_groups"]
g, det = LG.map_only_built_reasons(prof, argv, model)
out["gate_clean"], out["gate_decl"] = g, det["declared_frozen_built"]
out["dvb_clean"] = [str(m) for m in M.dvb_check_r9(model, args)]
out["gu_clean"] = [str(m) for m in D.check_grad_unreachable(model, args)]
# MUTATION: the round-2 (aeb5446) freeze -- a declared subtree trainable again
for n, p in model.named_parameters():
    if n.startswith("core.decoder.offset_head."):
        p.requires_grad_(True)
out["mut_declared"] = {"gate": LG.map_only_built_reasons(prof, argv, model)[0],
                       "dvb": [str(m) for m in M.dvb_check_r9(model, args)],
                       "gu": [str(m) for m in D.check_grad_unreachable(model, args)]}
for arm in ("refcv9_perception_trainable", "refcv9_planner_frozen"):
    M.apply_train_only(model, ("planner", "tactical", "map"))
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


def test_REAL_BUILD_the_trainable_frozen_and_declared_counts_of_each_set(real_build):
    assert real_build["dvb_build"] == []
    assert tuple(real_build["total"]) == REAL_TOTAL
    for spec, want in REAL_COUNTS.items():
        got = real_build["counts"][spec]
        assert tuple(got[:6]) == want, (spec, got[:6])
        assert tuple(got[6]) == (REAL_DECLARED if "planner" in spec else ()), (spec, got[6])
    ptm = real_build["counts"]["planner,tactical,map"][7]
    assert (ptm["planner"]["n_tensors"], ptm["planner"]["n_params"]) == (178, 15_254_091)
    assert (ptm["tactical"]["n_tensors"], ptm["tactical"]["n_params"]) == (113, 5_476_306)
    assert (ptm["map"]["n_tensors"], ptm["map"]["n_params"]) == (18, 105_032)
    assert {m: tuple(v) for m, v in real_build["per_module"].items()} == REAL_TACTICAL
    # the gate's 164-group literal IS the real build's trainable set
    assert real_build["groups"] == LG._R9W_GROUPS and len(LG._R9W_GROUPS) == 164
    assert sum(LG._R9W_GROUPS.values()) == 20_835_429 == LG._R9W_NUMEL


def test_REAL_BUILD_G_DVB_is_clean_and_the_declared_and_both_RED_arms_are_caught(real_build):
    assert real_build["gate_clean"] == [] and real_build["dvb_clean"] == [] and real_build["gu_clean"] == []
    assert tuple(real_build["gate_decl"]) == REAL_DECLARED
    mut = real_build["mut_declared"]
    assert any("2 DECLARED grad-unreachable parameter(s) are trainable again" in r for r in mut["gate"])
    assert any("a declared subtree is trainable again" in r for r in mut["dvb"])
    assert any("declared frozen but trainable again" in r for r in mut["gu"])
    a = real_build["refcv9_perception_trainable"]
    assert a["note"].startswith("refcv9_perception_trainable: 316 perception tensor(s) un-frozen (first core.encoder.")
    assert any("parameter(s) OUTSIDE" in r for r in a["gate"]) and any("20835429" in r for r in a["gate"])
    assert any("the perception freeze is broken" in r for r in a["dvb"])
    b = real_build["refcv9_planner_frozen"]
    assert b["note"] == "refcv9_planner_frozen: 186 planner (core.decoder) tensors frozen (the --r9-train-only planner set)"
    assert any("178 map-only parameter(s) are frozen" in r for r in b["gate"]) and any("5581338" in r for r in b["gate"])
    assert any("178 frozen" in r for r in b["dvb"])


# =========================================================================== #
# 4. two steps: planner,map (map rig), planner,tactical (refcv8 rig)          #
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


def _r8rig_batch(cfg):
    eps = T._synth_episodes(2, cfg.core, seed=0, min_frames=120)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=60, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    n = len(ds)
    batch = torch.utils.data.default_collate([ds[i] for i in range(0, n, max(1, n // 3))][:3])
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0])
    batch["lon_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0])
    batch["tac_goal_y"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS))
    return batch


def test_TWO_STEPS_under_planner_tactical_on_the_refcv8_rig(tmp_path):
    cfg, m, args, rec, src_path = _r8rig_pt(tmp_path)
    mods, dec = tuple(rec["modules"]), tuple(rec["declared_frozen"])
    assert mods == PT_MODULES and dec == R8RIG_DECLARED
    opt = T.build_optimizer(m, args)
    assert T.assert_optimizer_matches_recipe(opt, args, model=m)
    src, _srec = M.load_source(str(src_path))
    ver = M.verify_against_source(m, src, fresh=True, modules=mods, declared=dec)
    assert ver["result"] == "bit-identical" and ver["n_trainable_keys"] == 176
    ema = W.WeightEMA(m, 0.9, exclude=M.ema_exclude_for(mods, dec))
    assert M.ema_problems(ema, m, src, fresh=True, modules=mods, declared=dec) == []
    guard = M.FrozenStateGuard(src, modules=mods, declared=dec)
    batch = _r8rig_batch(cfg)
    m._w_tac_v6 = 1.0
    m.train()
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    for i in range(2):
        torch.manual_seed(5 + i)
        L = T.compute_losses_v3(m, batch, "cpu", mode="diffusion")
        if i == 0:
            assert sorted(k for k, v in L.items() if torch.is_tensor(v) and v.requires_grad) == R8RIG_PT_GRAD_KEYS
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        ema.update(m)
    after = m.state_dict()
    moved = [k for k in before if not M.bit_equal(before[k], after[k])]
    # nothing outside the trained set moved (perception, strategic, the declared subtrees, every buffer)
    assert [k for k in moved if not M.is_trained(k, mods, dec)] == []
    assert guard.changed(m) == []
    got = {}
    for mod in mods:
        tr = [n for n, p in m.named_parameters() if p.requires_grad and M.is_trainable_under(n, (mod,))]
        got[mod] = (sum(1 for k in moved if M.is_trainable_under(k, (mod,))), len(tr))
    assert got == R8RIG_PT_MOVED
    assert sorted(n for n, p in m.named_parameters() if p.requires_grad and n not in moved) == R8RIG_PT_UNMOVED
    ema_sd = ema.model_state_dict(m)
    assert [k for k in src if not M.is_trained(k, mods, dec) and not M.bit_equal(ema_sd[k], src[k])] == []


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


def test_DELIBERATE_REGRESSION_a_declared_subtree_moving_trips_the_frozen_state_guard(tmp_path):
    cfg, m, args, rec, src_path = _r8rig_pt(tmp_path)
    mods, dec = tuple(rec["modules"]), tuple(rec["declared_frozen"])
    src, _srec = M.load_source(str(src_path))
    guard = M.FrozenStateGuard(src, modules=mods, declared=dec)
    assert guard.changed(m) == []
    with torch.no_grad():
        dict(m.named_parameters())["core.decoder.offset_head.weight"].add_(1e-3)
    assert guard.changed(m) == ["core.decoder.offset_head.weight"]
    # MUTATION of the guard itself: without the declaration it would not watch the subtree
    assert M.FrozenStateGuard(src, modules=mods).changed(m) == []


def test_DELIBERATE_REGRESSION_the_planner_frozen_too_is_refused_by_the_optimizer_and_G_DVB(tmp_path):
    model, cfg, args, rec, mods, opt, src, ver, ema, guard = _setup_pm(tmp_path)
    for n, q in model.named_parameters():
        if n.startswith("core.decoder."):
            q.requires_grad_(False)
    assert "a tensor in the optimizer is frozen" in M.optimizer_problems(opt, model, args)
    assert any("47 frozen" in str(m) for m in M.dvb_check_r9(model, args))


# =========================================================================== #
# 5. `--r9-train-only map` == `--r8-train-only-map`, bit for bit              #
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
            assert M.ema_exclude_for(rec["modules"], tuple(rec["declared_frozen"])) is M.ema_exclude
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
# 6. argv, the trainer's pin, emit start, TF hold, G-DVB                      #
# =========================================================================== #
_ON = ["--arm", "hier", "--out", "X", "--refcv8", "--map-hires", "on", "--map-hires-near-lift-m", "20",
       "--map-hires-near-refine-blocks", "1", "--w-map-hires", "1.0", "--opt", "dd", "--init-from", "e.pt",
       "--conflict-detector", "off", "--r9-train-only", "planner,map"]


def _pin(argv):
    M.pin_args(T.build_parser().parse_args(argv))


def _no_map(argv):
    for f in ("--map-hires", "--map-hires-near-lift-m", "--map-hires-near-refine-blocks", "--w-map-hires",
              "--r9-train-only"):
        argv = R._drop(argv, f)
    return argv


def test_GREEN_the_conformant_argv_pins_and_sets_without_map_need_no_map_head():
    _pin(_ON)
    _pin(R._drop(_ON, "--r9-train-only") + ["--r9-train-only", "planner,tactical,map"])
    for spec in ("planner", "tactical", "planner,tactical"):
        _pin(_no_map(_ON) + ["--r9-train-only", spec])
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
    (R._drop(_ON, "--r9-train-only") + ["--r9-train-only", "planner,tactics"], "unknown set"),
    (_no_map(_ON) + ["--r9-train-only", "tactical,map"], "the `map` set has no 10 cm map head")])
def test_every_refusal_of_the_flag(argv, needle):
    with pytest.raises(SystemExit, match=needle.replace("(", r"\(").replace("[", r"\[")):
        _pin(argv)


def test_the_trainers_own_pin_reaches_the_refusals():
    args = T.build_parser().parse_args(R._drop(_ON, "--init-from"))
    with pytest.raises(SystemExit, match="--init-from is not set"):
        T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)
    args = T.build_parser().parse_args(R._drop(_ON, "--r9-train-only") + ["--r9-train-only", "planner",
                                                                           "--w-map-fp", "0.5"])
    with pytest.raises(SystemExit, match="not the map head"):                  # --w-map-fp's pin, reached too
        T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)


#: test_refcv8_emit_schedule's refcv8 argv (an emitting allocation run) + the refcv9 warm start without the map set
_ON8 = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X", "--refcv8", "--w-r8-cons", "0.05",
        "--r8-n-alloc", "8", "--w-r8-alloc-l1", "1.0"]
_R9PT = ["--init-from", "c.pt", "--r9-train-only", "planner,tactical", "--opt", "dd", "--conflict-detector", "off"]


def _pin8(argv):
    args = T.build_parser().parse_args(argv)
    cfg = types.SimpleNamespace(refcv8=r8c.R8Config())
    T._pin_refcv8(cfg, args)
    return cfg.refcv8


def test_R9W_EMIT_START_zero_is_admitted_under_the_r9_freeze_only():
    assert _pin8(_ON8 + ["--r8-alloc-emit", "--r8-alloc-emit-start", "0"] + _R9PT).emit_start == 0
    assert _pin8(_ON8 + ["--r8-alloc-emit", "--r8-alloc-emit-start", "1000"] + _R9PT).emit_start == 1000
    # MUTATIONS: MM ruling Q1's two refusals keep their meaning for every other run
    for argv, needle in ((_ON8 + ["--r8-alloc-emit", "--r8-alloc-emit-start", "0", "--init-from", "c.pt"], ">= 1"),
                         (_ON8 + ["--r8-alloc-emit", "--r8-alloc-emit-start", "0"], ">= 1"),
                         (_ON8 + ["--r8-alloc-emit"] + _R9PT, "WARM-STARTED"),
                         (_ON8 + ["--r8-alloc-emit", "--init-from", "c.pt"], "WARM-STARTED")):
        with pytest.raises(SystemExit) as e:
            _pin8(argv)
        assert needle in str(e.value), str(e.value)
    src = inspect.getsource(T._pin_refcv8)
    assert "_es0_r9 = _es is not None and int(_es) == 0 and _mof.is_r9(args)" in src


def test_R9W_TF_RESTART_the_ratio_is_held_at_refcv8s_end_value():
    # refcv8-r101-s0: --r8-tf-start 1.0 --r8-tf-end 0.25 --steps 48852; its LAST step is 48,851
    assert r8train.tf_ratio(48851, 48852, 1.0, 0.25) == 0.25
    r = _pin8(_ON8 + _R9PT + ["--r8-tf-start", "0.25", "--r8-tf-end", "0.25"])
    assert (r.tf_start, r.tf_end) == (0.25, 0.25)
    for step, steps in ((0, 3000), (1, 3000), (1499, 3000), (2999, 3000), (0, 1)):
        assert r8train.tf_ratio(step, steps, r.tf_start, r.tf_end) == 0.25
    # MUTATION: refcv8's defaults restart the schedule at 1.0 (full teacher forcing) on the warm start's step 0
    r = _pin8(_ON8 + _R9PT)
    assert (r.tf_start, r.tf_end) == (1.0, 0.25)
    assert r8train.tf_ratio(0, 3000, r.tf_start, r.tf_end) == 1.0


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
# 7. the refcv9-warm launch-gate profile                                      #
# =========================================================================== #
R9W = LG.PROFILES["refcv9-warm"]
LAUNCH = ROOT / "ops" / "runs.d" / "refcv8-r101-s0.argv.json"
THOR_INIT = "/home/nvidia/refcv9_warm/ckpt_ema_48852.pt"


def test_the_profile_literals():
    mo = R9W["map_only"]
    assert mo["flag"] == "--r9-train-only" and mo["modules"] == PTM_MODULES
    assert mo["declared_frozen"] == REAL_DECLARED
    assert mo["numel"] == 20_835_429 and len(mo["groups"]) == 164 and sum(mo["groups"].values()) == 20_835_429
    assert mo["init_from_md5"] == SHIPPED_EMA_MD5
    assert {g: n for g, n in mo["groups"].items() if g.startswith("_map_hires.")} == LG._MAPFT_GROUPS
    assert sum(n for g, n in mo["groups"].items() if g.startswith("core.decoder")) == 15_254_091
    assert sum(n for g, n in mo["groups"].items() if any(g == t or g.startswith(t + ".") for t in TAC_MODULES)) \
        == 5_476_306
    assert mo["groups"]["core.decoder"] == 2                     # navc_gate + r8_rc_gate (direct parameters)
    assert [g for g in mo["groups"] if any(g == d or g.startswith(d + ".") for d in REAL_DECLARED)] == []
    assert R9W["required_checks"] == ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE",
                                      "G-SUITE-PINNED", "G-MAP-OVERFIT")
    for f in ("--r9-train-only", "--init-from", "--conflict-detector", "--w-map-fp", "--r9-alloc-force-schedule",
              "--r9-alloc-force-seed", "--r8-alloc-emit-start", "--r8-tf-start", "--r8-tf-end", "--steps"):
        assert f in R9W["required_flags"], f
    assert len(R9W["required_flags"]) == len(set(R9W["required_flags"]))
    rv = R9W["required_values"]
    assert (rv["--r9-train-only"], rv["--conflict-detector"], rv["--r9-alloc-force-seed"]) == (
        "planner,tactical,map", "off", "20261027")
    assert (rv["--r8-alloc-emit-start"], rv["--r8-tf-start"], rv["--r8-tf-end"]) == ("0", "0.25", "0.25")
    for f in ("--steps", "--warmup", "--ema-decay", "--eval-every", "--b1-guard-ref-ap2m", "--b1-guard-step",
              "--w-map-fp", "--r9-alloc-force-schedule"):
        assert f not in rv, f
    assert "--w-map-fp" in R9W["required_positive"]
    assert R9W["alloc_force_rule"] == {"flag": "--r9-alloc-force-schedule", "kind": "linear", "p0": 1.0, "p1": 0.0,
                                       "start": 0, "end_of": "--steps", "end_div": 2}
    forb = {f for e in R9W["forbidden_levers"] for f in e[0]}
    for f in ("--r8-train-only-map", "--b1-guard-ref-ap2m", "--grad-share-every", "--conflict-every",
              "--trunk-bn-recalib", "--r8-critic-drivable", "--r9-progress-head", "--r9-calib-stage", "--w-r9-prog",
              "--w-r9-regime", "--r9-decision", "--r9-hidden", "--r9-decel-tau", "--r9-seed"):
        assert f in forb, f
    assert "--r9-train-only" not in forb
    assert [(g, f) for g, f, _w in R9W["live_dead_admitted"]] == [
        ("core.decoder.r8_rc_to_cond", "--r8-no-rc"), ("core.decoder.r8_rc_gate", "--r8-no-rc")]
    # R9W-BUDGET + R9W-MAP-FP-WEIGHT closed open-only (MM 2026-10-27, the MAPFT-BUDGET precedent); the record stays
    assert R9W["open_items"] == ()
    assert [o["id"] for o in LG._R9W_OPEN_ITEMS] == ["R9W-BUDGET", "R9W-MAP-FP-WEIGHT"]
    assert ("w = 0.1 x map_hires / map_fp at the smoke's FIRST LOGGED step, rounded to ONE significant figure"
            in LG._R9W_OPEN_ITEMS[1]["what"])
    assert R9W["map_hires"]["ga_keys"] == ("ga_mh_refine", "ga_mh_near", "ga_mh_near_refine")
    assert R9W["map_hires"]["ga_keys_zero"] == ("ga_mh_lift", "ga_mh_encoder", "ga_mh_trunk_s8_stage")
    assert LG.REFCV9_ARMS == ("refcv9_perception_trainable", "refcv9_planner_frozen")
    # no open item blocks a pass now; MUTATION control: re-opening the recorded items blocks it again
    assert LG.open_item_reasons(R9W) == [] and LG.open_item_reasons(LG.PROFILES["refcv8-mapft"]) == []
    assert len(LG.open_item_reasons(dict(R9W, open_items=LG._R9W_OPEN_ITEMS))) == 2
    assert LG.PROFILES["refcv8-mapft"]["map_only"]["flag"] == "--r8-train-only-map"
    assert "alloc_force_rule" not in LG.PROFILES["refcv8-mapft"] and "alloc_force_rule" not in LG.PROFILES["refcv8"]


def test_B5_the_profile_resolves_to_refcv8_mapfts_env_failure_pin():
    want = ("stack/ops/launch_gate_thor_env_failures_mapft.json",
            "1af7c5dd7e45b31f0f419e992764d31a20f72e743d55c5b971f4c79162ef02f9")
    assert R9W["suite_env_failures"] == want == LG.PROFILES["refcv8-mapft"]["suite_env_failures"]
    assert R9W["suite_pinned_skip_ok"] == LG.PROFILES["refcv8-mapft"]["suite_pinned_skip_ok"]
    assert R9W["suite_pinned_skip_ok"][-1][0] == (r"tests\.test_v8f_kindx_s1::"
                                                  r"test_the_running_kscore_chain_is_the_one_in_the_tree")
    rec, prob = LG.env_failure_pin(R9W, ROOT.parent)              # the file in THIS tree, its sha256 checked
    assert prob is None and (len(rec["ids"]), len(rec["files"])) == (328, 40)
    # MUTATION: the SHARED pin refcv9-warm inherited at aeb5446 -- 168 ids in 35 files, without the 160 v8f-mini ids
    shared = ("stack/ops/launch_gate_thor_env_failures.json",
              "22d8fe2c900c8654c0fce4af754ee01db52da97a93272b61acb38a1932acaf21")
    assert LG.PROFILES["refcv8"]["suite_env_failures"] == shared
    rec2, prob2 = LG.env_failure_pin(dict(R9W, suite_env_failures=shared), ROOT.parent)
    assert prob2 is None and (len(rec2["ids"]), len(rec2["files"])) == (168, 35)
    assert len(set(rec["ids"]) - set(rec2["ids"])) == 160 and set(rec2["ids"]) <= set(rec["ids"])
    # a tampered literal is refused, not read
    _r, prob3 = LG.env_failure_pin(dict(R9W, suite_env_failures=(want[0], "0" * 64)), ROOT.parent)
    assert _r is None and "a changed list is a reviewed edit of the profile" in prob3


@pytest.mark.parametrize("name", ["refcv8", "refcv8-nt", "refcv8-mapft", "v8f", "v8f_steer"])
def test_the_refcv8_family_profiles_refuse_the_r9_freeze(name):
    prof = LG.PROFILES[name]
    r = LG.forbidden_lever_reasons(prof, ["--arm", "hier", "--r9-train-only", "planner,tactical,map"])[0]
    assert len(r) == 1 and "--r9-train-only" in r[0] and "refcv9-warm" in r[0], r
    # MUTATION control: the same profile without the flag has no such refusal
    assert LG.forbidden_lever_reasons(prof, ["--arm", "hier"])[0] == []
    assert LG.forbidden_lever_reasons(R9W, ["--arm", "hier", "--r9-train-only", "planner,tactical,map"])[0] == []


def test_the_gate_reads_declared_subtrees_as_frozen_in_G_LIVE():
    src = inspect.getsource(LG)
    assert "    def mo_is_frozen(n_):\n        return not _under_any(n_, mo_mods) or _under_any(n_, mo_decl)\n" in src
    assert src.count("mo_is_frozen(") == 4                         # the definition + the frozen set, snapshot, final
    assert 'mo_decl = tuple((ctx.prof.get("map_only") or {}).get("declared_frozen") or ())' in src


@pytest.fixture(scope="module")
def r9_launch(tmp_path_factory):
    """refcv8-r101-s0's committed launch argv -> the refcv9 warm-start launch (the budget and weight values are
    placeholders: R9W-BUDGET / -MAP-FP-WEIGHT are open; the schedule's end is floor(3000 / 2))."""
    d = json.loads(LAUNCH.read_text(encoding="utf-8"))
    a = list(d["argv"])
    for f, v in {"--r9-train-only": ["planner,tactical,map"], "--init-from": [THOR_INIT],
                 "--conflict-detector": ["off"], "--w-map-fp": ["0.5"],
                 "--r9-alloc-force-schedule": ["linear:1.0:0.0:0:1500"], "--r9-alloc-force-seed": ["20261027"],
                 "--r8-alloc-emit-start": ["0"], "--r8-tf-start": ["0.25"], "--r8-tf-end": ["0.25"],
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
    assert "--r8-alloc-emit" in a                                  # refcv8's emission flag, inherited
    # the trainer's own argv pins accept it too, and pin the emission start and the held ratio
    args = T.build_parser().parse_args(a)
    M.pin_args(args)
    from tanitad.train import map_precision as MP
    MP.pin_args(args)
    cfg = types.SimpleNamespace(refcv8=r8c.R8Config())
    T._pin_refcv8(cfg, args)
    assert (cfg.refcv8.emit_start, cfg.refcv8.tf_start, cfg.refcv8.tf_end) == (0, 0.25, 0.25)


@pytest.mark.parametrize("flag,vals,needle", [
    ("--r9-train-only", None, "--r9-train-only"),
    ("--r9-train-only", ["map"], "--r9-train-only"),
    ("--r9-train-only", ["planner,map"], "--r9-train-only"),
    ("--w-map-fp", None, "--w-map-fp"),
    ("--w-map-fp", ["0"], "--w-map-fp"),
    ("--r9-alloc-force-schedule", None, "--r9-alloc-force-schedule"),
    ("--r9-alloc-force-schedule", ["linear:1.0:0.0:0:1501"], "R9W-SCHEDULE"),
    ("--r9-alloc-force-schedule", ["linear:1.0:0.0:0:3000"], "R9W-SCHEDULE"),
    ("--r9-alloc-force-schedule", ["linear:1.0:0.0:1:1500"], "R9W-SCHEDULE"),
    ("--r9-alloc-force-schedule", ["linear:0.9:0.0:0:1500"], "R9W-SCHEDULE"),
    ("--r9-alloc-force-schedule", ["linear:1.0:0.1:0:1500"], "R9W-SCHEDULE"),
    ("--r9-alloc-force-schedule", ["const:1.0"], "R9W-SCHEDULE"),
    ("--steps", None, "--steps"),
    ("--steps", ["3002"], "R9W-SCHEDULE"),
    ("--r8-alloc-emit-start", None, "--r8-alloc-emit-start"),
    ("--r8-alloc-emit-start", ["1000"], "--r8-alloc-emit-start"),
    ("--r8-tf-start", None, "--r8-tf-start"),
    ("--r8-tf-start", ["1.0"], "--r8-tf-start"),
    ("--r8-tf-end", ["0.0"], "--r8-tf-end"),
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


def test_GATE_R9W_SCHEDULE_the_end_step_is_floor_steps_over_two(r9_launch):
    a, prof, _opts, _pmap, _ = r9_launch
    assert LG.r9_schedule_reasons(prof, a) == ([], {"on": True, "rule": dict(R9W["alloc_force_rule"]),
                                                    "value": "linear:1.0:0.0:0:1500", "want": "linear:1.0:0.0:0:1500",
                                                    "end": 1500, "ok": True})
    # an odd --steps: floor(3001 / 2) = 1500; numbers compare as numbers
    odd = R._flag_set(a, "--steps", ["3001"])
    assert LG.r9_schedule_reasons(prof, odd)[0] == []
    assert LG.r9_schedule_reasons(prof, R._flag_set(odd, "--r9-alloc-force-schedule", ["linear:1:0:0:1500"]))[0] == []
    r = LG.r9_schedule_reasons(prof, R._flag_set(odd, "--r9-alloc-force-schedule", ["linear:1.0:0.0:0:1501"]))[0]
    assert r == ["R9W-SCHEDULE: --r9-alloc-force-schedule 'linear:1.0:0.0:0:1501' is not the rule's "
                 "'linear:1.0:0.0:0:1500' (p 1.0 -> 0.0 from step 0 to floor(--steps 3001 / 2) = 1500)"]
    # inert without the rule (every other profile)
    assert LG.r9_schedule_reasons(LG.PROFILES["refcv8-mapft"], a) == ([], {"on": False})


def test_GATE_the_warm_start_is_bound_by_content(r9_launch, tmp_path):
    a, prof, opts, pmap, init = r9_launch
    other = tmp_path / "ckpt_ema_other.pt"
    other.write_bytes(b"a different checkpoint")
    r, _ = LG.map_only_init_reasons(prof, R._flag_set(a, "--init-from", [str(other)]), pmap)
    assert len(r) == 1 and "the profile pins" in r[0] and r[0].startswith("--r9-train-only:")
    r, _ = LG.map_only_init_reasons(R9W, a, pmap)                # the REAL literal against a stand-in: refused
    assert len(r) == 1 and SHIPPED_EMA_MD5 in r[0]


def _tiny_r9w():
    """The map rig's planner,map literals (the map rig has no tactical decoder and no declared subtree in the sets)."""
    return dict(R9W, map_only=dict(R9W["map_only"], modules=PM_MODULES, groups=dict(RIG_PM_GROUPS), numel=61_009,
                                   declared_frozen=()))


def test_GATE_G_DVB_reads_the_built_set_and_both_RED_arms_go_RED_on_the_rig():
    prof, argv = _tiny_r9w(), ["--r9-train-only", "planner,map"]
    model, _ = R._build()
    M.apply_train_only(model, ("planner", "map"))
    r, det = LG.map_only_built_reasons(prof, argv, model)
    assert r == [] and det["trainable_numel"] == 61_009 and det["declared_frozen_built"] == []
    r, _ = LG.map_only_built_reasons(R9W, argv, model)            # the REAL literal against the rig: RED
    assert any("20835429" in x for x in r)
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


def _r8rig_prof(declared):
    """The refcv8 rig's planner,tactical: the numel literal (121,300) and a declared-subtree literal under test; the
    leaf-group literal is the REAL build's concern (test_REAL_BUILD_...), so its reason is filtered here."""
    return dict(R9W, map_only=dict(R9W["map_only"], modules=PT_MODULES, groups={}, numel=121_300,
                                   declared_frozen=tuple(declared)))


def _non_group(reasons):
    return [r for r in reasons if "the trainable leaf groups are" not in r]


def test_GATE_G_DVB_holds_the_built_declarations_to_the_profiles_literal_on_the_refcv8_rig():
    argv = ["--r9-train-only", "planner,tactical"]
    _cfg, m = _r8rig()
    M.apply_train_only(m, ("planner", "tactical"))
    r, det = LG.map_only_built_reasons(_r8rig_prof(R8RIG_DECLARED), argv, m)
    assert _non_group(r) == [] and det["trainable_numel"] == 121_300
    assert det["declared_frozen_built"] == list(R8RIG_DECLARED)
    # MUTATION 1: the literal omits the declarations -> named, and the 4 declared tensors read as frozen trained ones
    r, _ = LG.map_only_built_reasons(_r8rig_prof(()), argv, m)
    assert _non_group(r) == ["MAP-ONLY: the build's declared grad-unreachable subtrees inside the trained modules are "
                             "['core.decoder.control_head', 'core.decoder.offset_head']; the profile's literal is [] -- "
                             "a declaration would hide a trained group, or a dead one would be trained",
                             "MAP-ONLY: 4 map-only parameter(s) are frozen, e.g. ['core.decoder.offset_head.weight', "
                             "'core.decoder.offset_head.bias', 'core.decoder.control_head.weight']"]
    # MUTATION 2: the round-2 freeze (a declared subtree trainable again)
    for n, p in m.named_parameters():
        if n.startswith("core.decoder.offset_head."):
            p.requires_grad_(True)
    r, _ = LG.map_only_built_reasons(_r8rig_prof(R8RIG_DECLARED), argv, m)
    assert any(x.startswith("MAP-ONLY: 2 DECLARED grad-unreachable parameter(s) are trainable again") for x in r)
    assert any("121828 trainable values" in x for x in r)          # 121,300 + offset_head 512 + 16


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


# =========================================================================== #
# 8. assert_map_only_finetune.py --sets (the post-run K-FROZEN)               #
# =========================================================================== #
def _r9_state():
    g = torch.Generator().manual_seed(11)
    return {"core.encoder.conv.weight": torch.randn(4, 3, generator=g),
            "core.encoder.bn.running_mean": torch.zeros(4), "nav_inj.weight": torch.randn(3, 2, generator=g),
            "core.strategic.gru.weight": torch.randn(2, 2, generator=g),
            "core.decoder.w": torch.randn(3, 4, generator=g),
            "core.decoder.offset_head.weight": torch.randn(2, 4, generator=g),
            "phi_tac.0.weight": torch.randn(4, 4, generator=g),
            "tac_decoder_v6.l.weight": torch.randn(2, 2, generator=g),
            "core.lat_head.weight": torch.randn(3, 4, generator=g),
            "_map_hires.lift.proj.weight": torch.randn(2, 2, generator=g),
            "_map_hires.refine.cls.weight": torch.randn(8, 2, generator=g),
            "_map_hires.near.lift.unobserved": torch.zeros(2),
            "_map_hires.near_refine.0.c1.weight": torch.randn(2, 2, generator=g)}


def _at(src_sd, dst_sd, tmp_path, *extra):
    s, d = tmp_path / "src.pt", tmp_path / "dst.pt"
    torch.save({"model": src_sd, "step": 48852, "kind": "ema_model_only"}, s)
    torch.save({"model": dst_sd, "step": 3000, "kind": "ema_model_only"}, d)
    rc = AT.main(["--src", str(s), "--dst", str(d), "--json", str(tmp_path / "r.json"), *extra])
    return rc, json.loads((tmp_path / "r.json").read_text(encoding="utf-8")) if rc in (0, 1) else None


def test_ASSERT_TOOL_the_sets_literal_is_the_trainers():
    assert AT.TRAIN_ONLY_SETS == M.TRAIN_ONLY_SETS == {"planner": ("core.decoder",), "tactical": TAC_MODULES,
                                                      "map": MAP_MODULES}
    assert AT.parse_sets("map,tactical,planner") == ("planner", "tactical", "map")
    for bad in ("", "planner,planner", "perception", "planner,"):
        with pytest.raises(ValueError):
            AT.parse_sets(bad)


def test_ASSERT_TOOL_r9_the_changes_are_named_per_set_and_the_perception_is_bit_equal(tmp_path):
    src = _r9_state()
    dst = copy.deepcopy(src)
    for k in ("core.decoder.w", "phi_tac.0.weight", "core.lat_head.weight", "_map_hires.refine.cls.weight"):
        dst[k] = dst[k] + 0.5
    rc, res = _at(src, dst, tmp_path, "--sets", "planner,tactical,map")
    assert rc == 0 and res["ok"] is True
    assert res["sets"] == ["planner", "tactical", "map"] and res["trained_modules"] == list(PTM_MODULES)
    v = res["views"]["ema"]
    assert (v["n_frozen_checked"], v["n_frozen_changed"], v["frozen_changed"]) == (5, 0, [])
    assert v["per_set"]["planner"]["changed"] == ["core.decoder.w"]
    assert v["per_set"]["tactical"]["changed"] == ["core.lat_head.weight", "phi_tac.0.weight"]
    assert v["per_set"]["map"]["changed"] == ["_map_hires.refine.cls.weight"]
    assert ({m: (e["n_tensors"], e["n_changed"]) for m, e in v["per_set"]["tactical"]["per_module"].items()} ==
            {"phi_tac": (1, 1), "nav_to_tac": (0, 0), "lat_head_tac": (0, 0), "lon_head_tac": (0, 0),
             "tac_goal_head": (0, 0), "tac_latent_proj": (0, 0), "tac_decoder_v6": (1, 0),
             "tac_behaviour_gate_v6": (0, 0), "core.tactical_trunk": (0, 0), "core.lat_head": (1, 1),
             "core.lon_head": (0, 0)})
    assert v["per_set"]["planner"]["n_tensors"] == 2 and v["per_set"]["map"]["n_tensors"] == 3


@pytest.mark.parametrize("key", ["core.encoder.conv.weight", "core.encoder.bn.running_mean", "nav_inj.weight",
                                 "core.strategic.gru.weight", "_map_hires.lift.proj.weight"])
def test_ASSERT_TOOL_r9_RED_a_perception_tensor_changed(tmp_path, key):
    src = _r9_state()
    dst = copy.deepcopy(src)
    dst[key] = torch.nextafter(dst[key], dst[key] + 1)              # one ulp
    rc, res = _at(src, dst, tmp_path, "--sets", "planner,tactical,map")
    assert rc == 1 and [r["key"] for r in res["views"]["ema"]["frozen_changed"]] == [key]


def test_ASSERT_TOOL_r9_also_frozen_holds_a_declared_subtree_bit_equal(tmp_path):
    src = _r9_state()
    dst = copy.deepcopy(src)
    dst["core.decoder.offset_head.weight"] = dst["core.decoder.offset_head.weight"] + 0.25
    rc, res = _at(src, dst, tmp_path, "--sets", "planner,tactical,map", "--also-frozen", "core.decoder.offset_head")
    assert rc == 1 and [r["key"] for r in res["views"]["ema"]["frozen_changed"]] == ["core.decoder.offset_head.weight"]
    assert res["views"]["ema"]["also_frozen"] == {"core.decoder.offset_head": 1}
    # MUTATION control: without --also-frozen the subtree is planner, and its change PASSES
    rc, res = _at(src, dst, tmp_path, "--sets", "planner,tactical,map")
    assert rc == 0 and res["views"]["ema"]["per_set"]["planner"]["changed"] == ["core.decoder.offset_head.weight"]


def test_ASSERT_TOOL_r9_require_each_module_moved(tmp_path):
    src = _r9_state()
    dst = copy.deepcopy(src)
    dst["core.decoder.w"] = dst["core.decoder.w"] + 0.5
    rc, res = _at(src, dst, tmp_path, "--sets", "planner,map")
    assert rc == 0 and res["views"]["ema"]["modules_unmoved"] == list(MAP_MODULES)
    rc, res = _at(src, dst, tmp_path, "--sets", "planner,map", "--require-each-module-moved")
    assert rc == 1 and res["views"]["ema"]["modules_unmoved"] == list(MAP_MODULES)
    for k in ("_map_hires.refine.cls.weight", "_map_hires.near.lift.unobserved", "_map_hires.near_refine.0.c1.weight"):
        dst[k] = dst[k] + 0.5
    rc, res = _at(src, dst, tmp_path, "--sets", "planner,map", "--require-each-module-moved")
    assert rc == 0 and res["views"]["ema"]["modules_unmoved"] == []


def test_ASSERT_TOOL_the_legacy_form_is_unchanged_and_the_r9_flags_need_sets(tmp_path):
    src = _r9_state()
    dst = copy.deepcopy(src)
    dst["_map_hires.refine.cls.weight"] = dst["_map_hires.refine.cls.weight"] + 0.5
    rc, res = _at(src, dst, tmp_path)
    assert rc == 0
    assert sorted(res) == ["dst", "map_only_modules", "ok", "src", "src_view", "tool", "views"]
    assert sorted(res["views"]["ema"]) == ["extra", "map_changed", "missing", "n_extra", "n_map_changed", "n_map_only",
                                           "n_missing", "n_non_map_changed", "n_non_map_checked", "non_map_changed",
                                           "ok"]
    # legacy: the planner (core.decoder) is FROZEN -- its change fails
    dst["core.decoder.w"] = dst["core.decoder.w"] + 0.5
    rc, res = _at(src, dst, tmp_path)
    assert rc == 1 and [r["key"] for r in res["views"]["ema"]["non_map_changed"]] == ["core.decoder.w"]
    for extra in (["--also-frozen", "core.decoder.offset_head"], ["--require-each-module-moved"]):
        assert _at(src, dst, tmp_path, *extra)[0] == 2
    assert _at(src, dst, tmp_path, "--sets", "planner,perception")[0] == 2
