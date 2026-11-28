"""G-BOX-OVERFIT's model replay (`stack/scripts/g_box_overfit.py::TrainerAdapter._build_model`) builds the
LAUNCH model under the A18 argv -- the 10 cm branch WITH the R2 near lift and the R3 near refine block.

MEASURED 2026-09-27 (refcv7 eval-loader package, `raw/gbo_dvb_check.py`): the replay at b711411 (blob
3c051db7) built `MapHiresConfig` without `near_lift_x_m` / `near_refine_blocks`, so under
`--map-hires-near-lift-m 20 --map-hires-near-refine-blocks 1` it was NOT the launch model and its own G-DVB
check refused it (5 mismatches). The fix passes both, exactly as `refc_v3_train.train()` declares them.

The REFERENCE is the trainer's own `train()` on a tiny rig, captured as the launch gate captures it
(`launch_gate.run_trainer_until`, before any data is read); every expectation is that model or a LITERAL.
RED arm: the historical defect re-introduced into a COPY of the harness source (the two lines removed) --
its G-DVB refuses by name, and with G-DVB silenced its model lacks exactly the 7 literal near keys.
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

GBO_PATH = STACK / "scripts" / "g_box_overfit.py"
CLIPS = ("gbo-near-rig-clip-a", "gbo-near-rig-clip-b")
#: MEASURED on this rig (tip b711411 code): the R2 near lift (3) + the R3 refine block (4) keys
NEAR_KEYS = ["_map_hires.near.lift.proj.bias", "_map_hires.near.lift.proj.weight",
             "_map_hires.near.lift.unobserved", "_map_hires.near_refine.0.c1.weight",
             "_map_hires.near_refine.0.c2.weight", "_map_hires.near_refine.0.n1.bias",
             "_map_hires.near_refine.0.n1.weight"]
N_HIRES_KEYS = 54
#: the two lines of the fix -- removing them from a COPY re-installs the historical defect verbatim
FIX_LINES = ("                near_lift_x_m=float(_mhr.declared_near_lift_m(a)),\n",
             "                near_refine_blocks=int(_mhr.declared_near_refine_blocks(a)),\n")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synth_inputs(d: Path) -> None:
    """The model inputs a refcv7 build reads, synthesised (anchors with a {0,0} control node, a PER-CLIP
    extrinsics table, the 10 cm class weights AT the declared extent, the banked tau file)."""
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
    extr = {c: {"qx": -0.5, "qy": 0.5, "qz": -0.5 + 0.004 * i, "qw": 0.5, "x": 2.1 + 0.05 * i,
                "y": 0.0, "z": 1.45 + 0.1 * i} for i, c in enumerate(CLIPS)}
    (d / "extr.json").write_text(json.dumps(extr), encoding="utf-8")
    (d / "cw.json").write_text(json.dumps(
        {"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES),
         "weights": [1.0, 1.5, 3.0, 2.0, 4.0, 1.2, 2.5, 1.1], "dry_run": False,
         "extent": {"x_max_m": 60.0, "y_half_m": 16.0}}), encoding="utf-8")
    (d / "tau.json").write_text(json.dumps({"tau": 0.18}), encoding="utf-8")


def _argv(d: Path) -> list[str]:
    """The A18 launch argv's refcv7 blocks at rig size, WITH the near lift (20 m) and 1 refine block; every
    data path a non-existent sentinel. (The harness builds from the launch argv minus --trunk-compile.)"""
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
        "--clip-clock-sidecar", s + "/clock.jsonl", "--batch", "2", "--steps", "10",
        "--out", str(d / "out")]


def _adapter(G, T, argv):
    """`TrainerAdapter` without its data build (`_build_items` needs the TRAIN cache): exactly the three
    attributes `_build_model` reads."""
    ad = object.__new__(G.TrainerAdapter)
    ad.tr, ad.device, ad.args = T, "cpu", T.build_parser().parse_args(list(argv))
    return ad


def _harness_model(G, T, argv):
    torch.manual_seed(0)                       # TrainerAdapter.setup(arm, seed): the seed, then the build
    ad = _adapter(G, T, argv)
    return ad._build_model(), ad


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    d = tmp_path_factory.mktemp("gbo_near_lift")
    _synth_inputs(d)
    argv = _argv(d)
    T = _load("refc_v3_train_gbo_near_ref", STACK / "scripts" / "refc_v3_train.py")
    cap = LG.run_trainer_until(T, argv, "model")            # the REFERENCE: the real train()
    G = _load("g_box_overfit_near_lift_t", GBO_PATH)
    return types.SimpleNamespace(d=d, argv=argv, T=T, model_t=cap["model"], G=G)


def test_the_harness_replay_IS_the_launch_model_with_the_near_lift(rig):
    m, ad = _harness_model(rig.G, rig.T, rig.argv)
    assert ad.dvb_mismatches == []
    kt, kh = set(rig.model_t.state_dict()), set(m.state_dict())
    assert sorted(kt ^ kh) == []
    assert sorted(k for k in kh if k.startswith("_map_hires.near")) == NEAR_KEYS
    assert sum(k.startswith("_map_hires.") for k in kh) == N_HIRES_KEYS
    assert (m._map_hires.cfg.near_lift_x_m, m._map_hires.cfg.near_refine_blocks) == (20.0, 1)
    assert m._map_hires.cfg == rig.model_t._map_hires.cfg
    # the same seed, the same build order as train(): the harness's fresh init IS the launch init
    dt, dh = LG.state_digests(rig.model_t.state_dict()), LG.state_digests(m.state_dict())
    assert [k for k in dt if dt[k] != dh[k]] == []


def test_RED_the_historical_replay_is_refused_by_its_own_G_DVB_and_lacks_the_near_keys(rig, tmp_path,
                                                                                        monkeypatch):
    src = GBO_PATH.read_text(encoding="utf-8")
    for line in FIX_LINES:
        assert src.count(line) == 1, line
        src = src.replace(line, "")
    mut = tmp_path / "g_box_overfit_unfixed.py"
    mut.write_text(src, encoding="utf-8", newline="\n")
    G0 = _load("g_box_overfit_unfixed_t", mut)
    with pytest.raises(SystemExit) as e:
        _harness_model(G0, rig.T, rig.argv)
    msg = str(e.value)
    assert msg.startswith("[gbo] the replayed build is NOT the declared launch model:")
    assert "--map-hires-near-lift-m: declared 20.0 but BUILT 0.0" in msg
    assert "--map-hires-near-refine-blocks: declared 1 but BUILT 0" in msg
    # G-DVB silenced: the replayed model is short by EXACTLY the near keys
    from tanitad.train import declared_vs_built as dvb
    monkeypatch.setattr(dvb, "check", lambda *a, **k: [])
    m0, _ad = _harness_model(G0, rig.T, rig.argv)
    kt, k0 = set(rig.model_t.state_dict()), set(m0.state_dict())
    assert sorted(kt - k0) == NEAR_KEYS
    assert sorted(k0 - kt) == []
