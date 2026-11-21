"""refcv7 EVAL LOADER (`stack/tanitad/eval/refcv7_loader.py`) -- it rebuilds the model the TRAINER
builds, bit for bit, on a tiny rig that carries every refcv7 build block.

The REFERENCE is the trainer's own `train()`, run exactly as the launch gate's G-EVAL runs it
(`launch_gate.run_trainer_until(T, argv, "model")`: the real `main()` -> `train()`, captured at its
first data-source call, before any training data is read). The loader under test never contributes
to its own expectation: every expected value below is either the trainer-built model or a LITERAL.

The rig (CPU, ~10 s): resnet18 at the smallest DECLARED camera geometry (128 x 576), the smoke
decoder width, and the launch argv's refcv7 blocks -- the 10 cm map branch (NEW-2) with the R2
near lift and one R3 refine block at the /2 extent, the A6 pooled planner BEV, the A9 box head
(focal presence, deep supervision, VIS-1, 300 queries) with the A14 learned reference points, the
residual prior (NEW-1), the tac8 prior, the nav-compliance gate (+ its banked tau FILE), the
speed-ceiling filter, the ego history, the v6 tactical decoder, the max-speed input and the trunk
memory levers. Every data path is a NON-EXISTENT sentinel: a build that read one would fail.

RED arms (each must go red, with a literal expectation):
  * the VENDORED refcv6 loader -- a loader with NO map_hires block -- fails the STRICT load on the
    first key the Master Mind's G-EVAL dry run measured, `_map_hires.lift.unobserved`;
  * this loader with its map_hires block REMOVED refuses before any load (the A6 pool has no 10 cm
    branch to read);
  * this loader with the R2/R3 fields dropped from `MapHiresConfig` (the replay drift of
    `g_box_overfit.py`'s TrainerAdapter) fails the STRICT load on exactly the near-lift keys;
  * a record whose 10 cm stamp contradicts the rebuild, and an argv naming a file the kit does not
    carry, are REFUSED.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

STACK = Path(__file__).resolve().parents[1]
REPO = STACK.parent
for _p in (str(STACK), str(STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402

LOADER_PATH = STACK / "tanitad" / "eval" / "refcv7_loader.py"
REFCV6_LOADER_PATH = STACK / "tanitad" / "eval" / "refcv6_loader.py"
CLIPS = ("r7ldr-rig-clip-a", "r7ldr-rig-clip-b")
#: MEASURED on this rig at tip 37086c3 + the box A17 overlay: the 10 cm branch's state_dict keys
#: (lift 3, encoder 33, refine 11, near 3, near_refine 4 = 54) and the R2 + R3 part of them
N_HIRES_KEYS = 54
NEAR_KEYS = ["_map_hires.near.lift.proj.bias", "_map_hires.near.lift.proj.weight",
             "_map_hires.near.lift.unobserved", "_map_hires.near_refine.0.c1.weight",
             "_map_hires.near_refine.0.c2.weight", "_map_hires.near_refine.0.n1.bias",
             "_map_hires.near_refine.0.n1.weight"]


# --------------------------------------------------------------------------------------------- #
# the rig                                                                                       #
# --------------------------------------------------------------------------------------------- #
def _synth_inputs(d: Path) -> None:
    """The four MODEL inputs a refcv7 build reads, synthesised: a v0-conditioned 117-anchor
    artifact (F9 asserts 117) whose control grid contains {0, 0}, a PER-CLIP extrinsics table, the
    10 cm class weights AT the declared extent, and the banked tau file."""
    from tanitad.data.semantic_map_gt_fine import FINE_CLASSES
    from tanitad.models import map_head_hires as H
    from tanitad.refs import anchor_meta
    from tanitad.refs.refc_v3 import V3_HORIZONS
    acc = torch.linspace(-4.0, 4.0, 9)
    lat = torch.linspace(-3.0, 3.0, 13)
    ctrl = torch.stack(torch.meshgrid(acc, lat, indexing="ij"), -1).reshape(-1, 2)
    t = torch.tensor([h * 0.1 for h in V3_HORIZONS])
    x = 10.0 * t[None, :] + 0.5 * ctrl[:, :1] * t[None, :] ** 2
    y = 0.5 * ctrl[:, 1:2] * t[None, :] ** 2
    torch.save(anchor_meta.build_anchor_artifact(
        torch.stack([x, y], -1), ctrl, control_units="alat", horizons=V3_HORIZONS, dt=0.1,
        ref_speed_ms=10.0, kappa_cap=0.3, alat_v_floor=2.0, builder=None), d / "anchors.pt")
    # a level forward camera (camera->vehicle: +x right, +y down, +z boresight), two mount poses
    extr = {c: {"qx": -0.5, "qy": 0.5, "qz": -0.5 + 0.004 * i, "qw": 0.5, "x": 2.1 + 0.05 * i,
                "y": 0.0, "z": 1.45 + 0.1 * i} for i, c in enumerate(CLIPS)}
    (d / "extr.json").write_text(json.dumps(extr), encoding="utf-8")
    (d / "cw.json").write_text(json.dumps(
        {"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES),
         "weights": [1.0, 1.5, 3.0, 2.0, 4.0, 1.2, 2.5, 1.1], "dry_run": False,
         "extent": {"x_max_m": 60.0, "y_half_m": 16.0}}), encoding="utf-8")
    (d / "tau.json").write_text(json.dumps({"tau": 0.18}), encoding="utf-8")


def _argv(d: Path) -> list[str]:
    """The intended launch argv's refcv7 blocks at rig size (every data path a sentinel)."""
    s = str(d / "__unread__")
    return [
        "--arm", "hier", "--size", "small", "--smoke", "--device", "cpu",
        "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained",
        "--trunk-mode", "shared", "--trunk-fuse", "concat1x1", "--trunk-in-channels", "9",
        "--ego-history", "--no-strategic", "--image-hw", "128", "576",
        "--sampler", "ddim", "--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln",
        "--f5-focal", "--f5-emitting-conf", "--f6-w-u0-zero", "--f9-assert-vocab",
        "--anchors", str(d / "anchors.pt"), "--anchor-v0-conditioned", "--n-anchors", "117",
        "--v2-cache", s + "/train", "--v7-labels", s + "/labels.jsonl.gz", "--nav-from-v7",
        "--eval-cache", s + "/eval", "--eval-labels", s + "/labels_eval.jsonl.gz",
        "--eval-every", "500", "--eval-batches", "1",
        "--speed-max-sidecar-v6-eval", s + "/smax_eval.jsonl",
        "--tac-decoder-v6", "--w-tac-v6", "1.0", "--tac-decoder-d-bev", "96",
        "--graft-behaviour-sel", "--max-speed-input-v6", "--speed-max-sidecar-v6", s + "/smax.jsonl",
        "--agents", "head", "--agent-join", s + "/join.jsonl.xz",
        "--agent-rig-camera", "extrinsics", "--agent-rig-extrinsics", str(d / "extr.json"),
        "--agent-cls-weight", "b1", "--w-agent", "1.0",
        "--w-map", "0", "--map-gt-root", s + "/sam3", "--w-box3d", "1.0",
        "--join3d", s + "/join3d.jsonl.xz",
        "--slot-presence-loss", "focal", "--slot-presence-prior", "0.01",
        "--slot-deep-supervision", "--slot-vis1", "--vis1-sidecar", s + "/vis1.npz",
        "--slot-query-select", "learned_ref", "--bev-coupling", "--equalize-bottom-rows", "8",
        "--residual-prior", "ha0_ext_pose", "--graft-tac8-prior", "--graft-nav-compliance",
        "--nav-compliance-tau-rad", "0.18", "--nav-compliance-tau-file", str(d / "tau.json"),
        "--speed-ceiling-filter",
        "--map-hires", "on", "--w-map-hires", "1.0", "--map-hires-class-weights", str(d / "cw.json"),
        "--map-hires-decision-rule", "prior_corrected",
        "--map-hires-x-max-m", "60", "--map-hires-y-half-m", "16", "--map-hires-grad-ckpt", "on",
        "--map-hires-near-lift-m", "20", "--map-hires-near-refine-blocks", "1",
        "--bev-source", "map_hires_pool", "--bev-planner-crop-m", "60", "16",
        "--opt", "dd", "--lr", "1e-4", "--warmup", "2000", "--seed", "0", "--u8-batches",
        "--trunk-chunk-ckpt", "2", "--trunk-frozen-bn", "--trunk-fold-bn", "--trunk-dedup-frames",
        "--trunk-compile",
        "--clip-clock-sidecar", s + "/clock.jsonl", "--batch", "2", "--steps", "10",
        "--out", str(d / "out")]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _record(T, argv: list[str], **extra) -> dict:
    """The run record the loader reads: the post-FIX-3 trunk stamp, the TRAIN split's tactical-goal
    constants (a census every token passes; the trainable list derived from it), the bank mode."""
    from tanitad.models import vocab_v7
    census = {tok: {"pos": 300, "neg": 3000} for tok in vocab_v7.TACTICAL_GOAL_TOKENS_V7}
    return {"argv": list(argv), "seams": {"trunk_equalize_bottom_rows": 8},
            "tac_goal_stats": {"pos_weight": [10.0] * len(vocab_v7.TACTICAL_GOAL_TOKENS_V7),
                               "census": census,
                               "trainable": list(T._tac_goal_head.mask_report(census)["trainable"])},
            "withheld_bank": {"mode": "fixed"}, **extra}


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    d = tmp_path_factory.mktemp("refcv7_eval_loader")
    _synth_inputs(d)
    argv = _argv(d)
    with pytest.MonkeyPatch.context() as mp:
        for k in ("REFCV7_REPO", "REFCV7_KIT", "REFCV7_REMAP_OVERRIDES", "REFCV6_REMAP_OVERRIDES"):
            mp.delenv(k, raising=False)
        mp.setenv("REFCV6_REPO", str(REPO))
        # ⚠️ NO CUDA_VISIBLE_DEVICES here: `torch.cuda.is_available()` caches its first answer for
        # the whole process, so hiding the GPU inside one fixture would blind every later test of a
        # Thor suite session. The rig is CPU by its argv (`--device cpu`) and by `device="cpu"`.
        # the trainer under a PRIVATE module name: the gate's own cache (`refc_v3_train_gate`) is
        # never populated from a test
        T = _load_module("refc_v3_train_eval_loader_ref", STACK / "scripts" / "refc_v3_train.py")
        # the REFERENCE: the real train(), exactly as G-EVAL captures it (--trunk-compile dropped,
        # as the gate's CPU host drops it)
        cap = LG.run_trainer_until(T, LG.set_flag(argv, "--trunk-compile", None), "model")
        model_t, args_t = cap["model"], cap["args"]
        ck = d / "ckpt.pt"
        torch.save({"model": model_t.state_dict(), "step": 0}, ck)
        L = _load_module("refcv7_loader_under_test", LOADER_PATH)
        yield types.SimpleNamespace(d=d, argv=argv, T=T, model_t=model_t, args_t=args_t, ck=ck,
                                    L=L)


def _build(rig, config=None, **kw):
    return rig.L.build_model(config or _record(rig.T, rig.argv), str(rig.ck), device="cpu",
                             strict=True, remap={}, **kw)


def _batch(rig, cfg, idx=5):
    """One real window of the trainer's own V3Dataset over synthetic episodes stamped with the rig's
    clip ids (so the lift bank resolves them), plus the two batch keys a max-speed-v6 build reads."""
    T = rig.T
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=list(CLIPS))
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    ds.u8_frames = True
    ds.ego_history = True
    b = torch.utils.data.default_collate([ds[idx]])
    e_i, _t = ds.index[idx]
    b["map_ep"] = torch.tensor([int(eps[e_i].episode_id)], dtype=torch.long)
    b["v_max_ms"] = torch.tensor([13.8889])
    b["v_max_valid"] = torch.tensor([True])
    return b


# --------------------------------------------------------------------------------------------- #
# GREEN: the rebuild IS the trainer's model                                                     #
# --------------------------------------------------------------------------------------------- #
def test_the_loader_rebuilds_the_trainers_refcv7_model_bit_for_bit(rig):
    T, mt = rig.T, rig.model_t
    ml, cfg_l, args_l, rec = _build(rig)
    sd = rec["state_dict"]
    assert sd["missing"] == [] and sd["unexpected"] == []
    assert sd["n_map_hires_keys"] == N_HIRES_KEYS
    dt, dl = LG.state_digests(mt.state_dict()), LG.state_digests(ml.state_dict())
    assert set(dt) == set(dl)
    assert [k for k in dt if dt[k] != dl[k]] == []
    assert sum(k.startswith("_map_hires.") for k in dl) == N_HIRES_KEYS
    assert sum(k.startswith("_map_hires.") for k in dt) == N_HIRES_KEYS
    # every attribute G-EVAL compares (the `_w_*` carriers and the four named ones)
    names = sorted(a for a in set(vars(mt)) | set(vars(ml))
                   if a.startswith("_w_") or a in ("_cls_class_weight", "_map_lift_valid_mask",
                                                   "_box3d_visible_filter", "_bev_shuffle"))
    assert "_w_map_hires" in names
    diff = []
    for a in names:
        va, vb = getattr(mt, a, "<absent>"), getattr(ml, a, "<absent>")
        same = (LG.tensor_digest(va) == LG.tensor_digest(vb)) if torch.is_tensor(va) and \
            torch.is_tensor(vb) else (va == vb)
        if not same:
            diff.append(a)
    assert diff == []
    # the attributes the 10 cm forward reads that are NOT in G-EVAL's list
    assert ml._vis1 is True and mt._vis1 is True
    assert LG.tensor_digest(ml._map_hires_class_weight) == LG.tensor_digest(mt._map_hires_class_weight)
    assert ml._map_hires.cfg == mt._map_hires.cfg
    assert ml._perception.cfg == mt._perception.cfg
    assert ml.core.encoder.s8_tap is True and ml.core.encoder.s8_module == mt.core.encoder.s8_module
    assert ml._lift_bank_hires.clip_of == mt._lift_bank_hires.clip_of
    assert ml.cfg.nav_compliance_tau_sha256 == mt.cfg.nav_compliance_tau_sha256
    assert {k: int(v) for k, v in T.v3.param_breakdown_v3(mt).items()} == \
        {k: int(v) for k, v in T.v3.param_breakdown_v3(ml).items()}
    # the loader's OWN G-DVB check ran on its build and found nothing
    assert rec["declared_vs_built"]["mismatches"] == []
    # the forward on a fixed batch, as G-EVAL runs it -- with its determinism control
    batch = _batch(rig, cfg_l)
    mt.eval()
    ot = LG.output_digests(LG._forward_out(T, mt, batch, rig.args_t, 1234))
    assert [k for k, v in LG.output_digests(LG._forward_out(T, mt, batch, rig.args_t, 1234)).items()
            if ot.get(k) != v] == []
    ol = LG.output_digests(LG._forward_out(T, ml, batch, args_l, 1234))
    assert sorted(k for k in set(ot) | set(ol) if ot.get(k) != ol.get(k)) == []
    assert any(k.startswith("out.perception.map_hires_logits") for k in ot)
    # ... and a perturbed copy MUST differ (the comparison can see a change). ⚠️ The refine
    # block's LAST conv is zero-initialised, so perturbing its first conv would move NOTHING on a
    # fresh model: the control perturbs the last one.
    p0 = dict(ml.named_parameters())["_map_hires.near_refine.0.c2.weight"]
    with torch.no_grad():
        saved = p0.detach().clone()
        p0.add_(1e-2)
    try:
        op = LG.output_digests(LG._forward_out(T, ml, batch, args_l, 1234))
    finally:
        with torch.no_grad():
            p0.copy_(saved)
    assert op["out.perception.map_hires_logits"] != ot["out.perception.map_hires_logits"]


def test_the_record_names_its_departures_and_drops_only_trunk_compile(rig):
    _m, _c, _a, rec = _build(rig)
    assert rec["argv_remap"]["dropped"] == {"--trunk-compile": rig.L.DROP_FLAGS["--trunk-compile"]}
    assert "--trunk-compile" not in rec["argv_local"]
    assert any(x.startswith("argv --trunk-compile REMOVED") for x in rec["departures"])
    assert rec["ground_prior_probe"] == {"checked": False, "reason": "w_ground == 0 or no camera"}
    assert rec["map_hires"]["near_lift_x_m"] == 20.0 and rec["map_hires"]["near_refine_blocks"] == 1
    assert rec["perception"]["bev_source"] == "map_hires_pool"
    assert rec["perception"]["query_select"] == "learned_ref"
    assert rec["nav_compliance_tau_file"]["tau"] == 0.18


def test_the_loader_is_rng_neutral(rig):
    torch.manual_seed(7)
    want = torch.rand(4)
    torch.manual_seed(7)
    _build(rig)
    assert torch.equal(torch.rand(4), want)


# --------------------------------------------------------------------------------------------- #
# RED arms                                                                                      #
# --------------------------------------------------------------------------------------------- #
def test_RED_the_refcv6_loader_has_no_map_hires_block_and_FAILS_the_strict_load(rig):
    """The measured G-EVAL failure, on the rig: the vendored refcv6 loader (no 10 cm block) builds,
    and its STRICT load names the same first key the Master Mind's dry run did."""
    L6 = _load_module("refcv6_loader_red_arm", REFCV6_LOADER_PATH)
    with pytest.raises(RuntimeError) as e:
        L6.build_model(_record(rig.T, rig.argv), str(rig.ck), device="cpu", remap={}, strict=True)
    msg = str(e.value)
    assert msg.startswith("Error(s) in loading state_dict for RefCV3Model:")
    assert 'Unexpected key(s) in state_dict: "_map_hires.lift.unobserved"' in msg


def test_RED_this_loader_with_the_map_hires_block_REMOVED_refuses(rig, monkeypatch):
    def _no_block(tr, model, args, device):
        model._w_map_hires, model._map_hires, model._lift_bank_hires = 0.0, None, None
        model._map_hires_class_weight = model._map_hires_class_weight_stamp = None
        return None
    monkeypatch.setattr(rig.L, "build_map_hires_block", _no_block)
    with pytest.raises(SystemExit) as e:
        _build(rig)
    assert "bev_source 'map_hires_pool' but the 10 cm branch is not built" in str(e.value)


def test_RED_the_near_lift_and_refine_DROPPED_fails_the_strict_load(rig, monkeypatch):
    """`g_box_overfit.TrainerAdapter` builds MapHiresConfig without R2/R3 -- that drift, here."""
    orig = rig.L.map_hires_config

    def _r1_only(tr, args, sha):
        import dataclasses
        return dataclasses.replace(orig(tr, args, sha), near_lift_x_m=0.0, near_refine_blocks=0)
    monkeypatch.setattr(rig.L, "map_hires_config", _r1_only)
    with pytest.raises(RuntimeError) as e:
        _build(rig)
    msg = str(e.value)
    assert "Missing key(s)" not in msg
    got = sorted(k.strip(' "\n\t.') for k in msg.split("Unexpected key(s) in state_dict:")[1]
                 .split(",") if k.strip(' "\n\t.'))
    assert got == NEAR_KEYS


def test_RED_a_record_whose_10cm_stamp_contradicts_the_rebuild_is_REFUSED(rig):
    """The stamp is derived from the TRAINER's model (never from the loader); GREEN when it agrees,
    refused when it records R1 (no near lift) against an R2 argv."""
    mt = rig.model_t
    stamp = {**mt._map_hires.cfg.as_dict(), "branch_params": mt._map_hires.param_breakdown(),
             "class_weights": {"sha256": mt._map_hires_class_weight_stamp["sha256"]}}
    _m, _c, _a, rec = _build(rig, _record(rig.T, rig.argv, map_hires=stamp))
    sc = rec["stamp_checks"]
    assert sc["map_hires.config"]["equal"] is True
    assert sc["map_hires.branch_params"]["equal"] is True
    assert sc["map_hires.class_weights.sha256"]["equal"] is True
    bad = dict(stamp, near_lift_x_m=0.0, near_lift_rows=0)
    with pytest.raises(SystemExit) as e:
        _build(rig, _record(rig.T, rig.argv, map_hires=bad))
    assert "config.json CONTRADICTS the rebuilt model" in str(e.value)
    assert "map_hires.config" in str(e.value)


def test_RED_an_argv_file_the_kit_does_not_carry_is_REFUSED_and_an_override_is_exempt(
        rig, tmp_path, monkeypatch):
    L = rig.L
    monkeypatch.setattr(L, "KIT", tmp_path)            # a kit that EXISTS, on any host
    out, rec = L.remap_argv(["--anchors", "/home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt"])
    assert out == ["--anchors", L.PATH_REMAP_SPEC["--anchors"][1]]
    assert rec["remapped"]["--anchors"]["source"] == "kit default"
    with pytest.raises(SystemExit) as e:
        L.remap_argv(["--map-hires-class-weights", "/home/nvidia/data/refcv7/weights_60x16.json"])
    assert "'weights_60x16.json'" in str(e.value)
    assert "'map_hires_class_weights_train_100x30.json'" in str(e.value)
    out, rec = L.remap_argv(["--map-hires-class-weights", "/x/weights_60x16.json"],
                            remap={"--map-hires-class-weights": "C:/local/w.json"})
    assert out == ["--map-hires-class-weights", "C:/local/w.json"]
    assert rec["remapped"]["--map-hires-class-weights"]["source"] == "caller remap"
    # ... and a kit that does not exist on this host is named, not discovered at the first read
    monkeypatch.setattr(L, "KIT", tmp_path / "no_such_kit")
    with pytest.raises(SystemExit) as e:
        L.remap_argv(["--anchors", "/home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt"])
    assert "does not exist on this host" in str(e.value) and "--eval-kit" in str(e.value)


def test_RED_a_REFCV7_env_that_contradicts_the_gates_REFCV6_env_is_REFUSED(rig, tmp_path,
                                                                          monkeypatch):
    monkeypatch.setenv("REFCV6_KIT", str(tmp_path / "a"))
    monkeypatch.setenv("REFCV7_KIT", str(tmp_path / "b"))
    try:
        with pytest.raises(SystemExit) as e:
            _load_module("refcv7_loader_env_conflict", LOADER_PATH)
    finally:
        sys.modules.pop("refcv7_loader_env_conflict", None)
    assert "REFCV7_KIT" in str(e.value) and "disagree" in str(e.value)
