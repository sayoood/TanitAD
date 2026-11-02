"""G-BOX-OVERFIT under the refcv9 E1 perception rig (``--r9-perc-rig``): the harness builds the model the argv DECLARES.

WHY (C12-E1 blocker 2, MEASURED STATIC by agent G2): ``g_box_overfit.py`` had 0 hits for ``perc_rig`` / ``r9_``. Its
``TrainerAdapter`` replayed ``train()``'s build lines with ``v3.RefCV3Model(cfg)`` = the FULL model (planner, tactical
brain, scorer), then ``declared_vs_built.check`` against the argv -- which, on the rig argv, declares the rig. And its
data build went through the vendored audit loader, which REFUSES an argv with no ``--eval-labels`` (the rig argv has no
``--v7-labels`` by design). So the box overfit record could not be produced under the rig argv at all.

The REFERENCE is the trainer's own ``train()`` on a tiny rig, captured as the launch gate captures it
(``launch_gate.run_trainer_until(..., "model")``, before any data is read); every expectation is that model or a LITERAL.

* the rig replay IS train()'s rig model: same state_dict keys, same fresh-init digests, G-DVB clean, no planner module;
* RED arm 1: the historical full-model replay (the rig branch removed from a COPY of the harness) is REFUSED by its own
  G-DVB by name -- "a rig argv run on the full model must FAIL" -- and with G-DVB silenced its model carries exactly the
  planner / tactical keys the rig does not build;
* the OFF path is BIT-IDENTICAL: on a non-rig argv the harness and a COPY with the three gated lines reverted build the
  same keys and the same digests, and both equal train()'s model;
* the rig dataset builder takes none of the label block (a stub loader whose label path raises is never entered).
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
for _p in (str(STACK), str(STACK / "scripts"), str(STACK / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402
import test_g_box_overfit_near_lift as NL                          # noqa: E402  (its tiny rig, inputs, adapter helpers)

GBO_PATH = STACK / "scripts" / "g_box_overfit.py"
#: the gated build line and its original; removing the first for the second re-installs the historical full-model replay
RIG_BUILD = ("        m = (v3.RefCV3Model(cfg, perc_rig=True) if rig_on else v3.RefCV3Model(cfg)).to(dev)\n")
FULL_BUILD = "        m = v3.RefCV3Model(cfg).to(dev)\n"
RIG_PIN = "            tr._r9rig_mod.pin_args(a)\n"


def _rig_argv(d: Path) -> list[str]:
    """The E1 rig argv's blocks at tiny size (the planner / tactical / residual / anchor levers are ABSENT, as they are in
    ``stack/ops/runs.d/r9e1_C0.argv.json``: --arm flat, no --v7-labels, no --nav-from-v7, --r9-perc-rig); every data path a
    non-existent sentinel (the build is read before any data is)."""
    s = str(d / "__unread__")
    return [
        "--arm", "flat", "--size", "small", "--smoke", "--device", "cpu",
        "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained",
        "--trunk-mode", "shared", "--trunk-fuse", "concat1x1", "--trunk-in-channels", "9",
        "--image-hw", "128", "576",
        "--v2-cache", s + "/train", "--eval-cache", s + "/eval", "--eval-every", "500", "--eval-batches", "1",
        "--agents", "head", "--agent-join", s + "/join.jsonl.xz",
        "--agent-rig-camera", "extrinsics", "--agent-rig-extrinsics", str(d / "extr.json"),
        "--agent-cls-weight", "b1", "--w-agent", "1.0",
        "--w-map", "0", "--map-gt-root", s + "/sam3", "--w-box3d", "1.0",
        "--join3d", s + "/join3d.jsonl.xz",
        "--slot-presence-loss", "focal", "--slot-presence-prior", "0.01",
        "--slot-deep-supervision", "--slot-vis1", "--vis1-sidecar", s + "/vis1.npz",
        "--slot-query-select", "learned_ref", "--equalize-bottom-rows", "8",
        "--map-hires", "on", "--w-map-hires", "1.0", "--map-hires-class-weights", str(d / "cw.json"),
        "--map-hires-decision-rule", "prior_corrected",
        "--map-hires-x-max-m", "60", "--map-hires-y-half-m", "16", "--map-hires-grad-ckpt", "on",
        "--map-hires-near-lift-m", "20", "--map-hires-near-refine-blocks", "1",
        "--bev-source", "map_hires_pool", "--bev-planner-crop-m", "60", "16",
        "--opt", "dd", "--lr", "1e-4", "--warmup", "200", "--seed", "0", "--u8-batches",
        "--trunk-chunk-ckpt", "2", "--trunk-frozen-bn", "--trunk-fold-bn",
        "--clip-clock-sidecar", s + "/clock.jsonl", "--r9-perc-rig", "--conflict-detector", "off",
        "--batch", "2", "--steps", "10", "--out", str(d / "out_rig")]


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    d = tmp_path_factory.mktemp("gbo_rig")
    NL._synth_inputs(d)
    argv_rig, argv_off = _rig_argv(d), NL._argv(d)
    T = NL._load("refc_v3_train_gbo_rig_ref", STACK / "scripts" / "refc_v3_train.py")
    cap_rig = LG.run_trainer_until(T, argv_rig, "model")           # the REFERENCES: the real train(), rig on / off
    cap_off = LG.run_trainer_until(T, argv_off, "model")
    G = NL._load("g_box_overfit_rig_t", GBO_PATH)
    return types.SimpleNamespace(d=d, T=T, G=G, argv_rig=argv_rig, argv_off=argv_off,
                                 model_rig=cap_rig["model"], model_off=cap_off["model"])


def _mutated(tmp_path: Path, name: str, edits: list[tuple[str, str]]):
    src = GBO_PATH.read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return NL._load(f"{name}_t", p)


def test_the_harness_replay_IS_the_rig_model_train_builds(rigs):
    m, ad = NL._harness_model(rigs.G, rigs.T, rigs.argv_rig)
    assert ad.dvb_mismatches == []
    assert bool(getattr(m, "_perc_rig", False) or getattr(m.core, "_perc_rig", False))
    kt, kh = set(rigs.model_rig.state_dict()), set(m.state_dict())
    assert sorted(kt ^ kh) == []
    assert not any(k.startswith(("core.decoder", "core.strategic", "scorer", "tac_decoder_v6")) for k in kh)
    # the same seed, the same build order as train(): the harness's fresh init IS the launch init
    dt, dh = LG.state_digests(rigs.model_rig.state_dict()), LG.state_digests(m.state_dict())
    assert [k for k in dt if dt[k] != dh[k]] == []
    # the record the harness writes names what was built
    assert ad.rig_record["n_keys"] == len(kh) and ad.rig_record["built_roots"] == [
        "core.encoder", "core.agent_head", "_perception", "_map_hires"]


def test_RED_the_FULL_model_replay_of_a_rig_argv_is_refused_by_name(rigs, tmp_path, monkeypatch):
    G0 = _mutated(tmp_path, "gbo_full_build", [(RIG_BUILD, FULL_BUILD)])
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, rigs.argv_rig)
    msg = str(e.value)
    assert msg.startswith("[gbo] the replayed build is NOT the declared launch model:")
    assert "r9_perc_rig" in msg
    # G-DVB silenced: the replayed model carries the planner / tactical modules the rig does not build
    from tanitad.train import declared_vs_built as dvb
    monkeypatch.setattr(dvb, "check", lambda *a, **k: [])
    monkeypatch.setattr(rigs.T._r9rig_mod, "assert_rig_built", lambda m: {})      # the harness's own record check
    m0, _ad = NL._harness_model(G0, rigs.T, rigs.argv_rig)
    extra = set(m0.state_dict()) - set(rigs.model_rig.state_dict())
    assert extra and any(k.startswith("core.decoder") for k in extra)
    assert set(rigs.model_rig.state_dict()) - set(m0.state_dict()) == set()


def test_RED_pin_args_is_the_trainers_own_refusal(rigs, tmp_path):
    """A rig argv that also names a planner lever is refused at the build (a silent drop reads as 'the lever did
    nothing'), by the trainer's own pin_args -- the harness calls it where train() does."""
    bad = list(rigs.argv_rig) + ["--ego-history"]
    with pytest.raises(SystemExit) as e:
        NL._harness_model(rigs.G, rigs.T, bad)
    assert str(e.value).startswith("[r9-perc-rig] refused: planner / tactical flag(s) ['ego_history']")
    # RED: with the pin call removed from a COPY the refusal is not THIS one (the failure moves to the dvb stage, if at all)
    G0 = _mutated(tmp_path, "gbo_no_pin", [(RIG_PIN, "            pass\n")])
    try:
        NL._harness_model(G0, rigs.T, bad)
    except SystemExit as e0:
        assert not str(e0).startswith("[r9-perc-rig] refused: planner / tactical flag(s)")


def test_OFF_path_is_bit_identical_to_the_pre_rig_harness_and_to_train(rigs, tmp_path):
    G0 = _mutated(tmp_path, "gbo_pre_rig", [(RIG_BUILD, FULL_BUILD),
                                            ("        rig_on = bool(getattr(a, \"r9_perc_rig\", False))\n"
                                             "        if rig_on:\n" + RIG_PIN, "        rig_on = False\n")])
    m_new, ad_new = NL._harness_model(rigs.G, rigs.T, rigs.argv_off)
    m_old, ad_old = NL._harness_model(G0, rigs.T, rigs.argv_off)
    assert ad_new.dvb_mismatches == ad_old.dvb_mismatches == []
    assert ad_new.rig_record is None
    dn, do_, dt = (LG.state_digests(x.state_dict()) for x in (m_new, m_old, rigs.model_off))
    assert sorted(dn) == sorted(do_) == sorted(dt)
    assert [k for k in dn if dn[k] != do_[k]] == []
    assert [k for k in dn if dn[k] != dt[k]] == []
    assert getattr(m_new, "_perc_rig", False) is False and getattr(m_new.core, "_perc_rig", False) is False


def test_the_OFF_data_path_still_calls_the_vendored_loader_once_and_the_rig_path_is_gated():
    src = GBO_PATH.read_text(encoding="utf-8")
    call = ("e_ds, e_eps, self.dataset_record = L.build_eval_dataset(None, cfg, a_loader, self.config,\n"
            "                                                                        with_perception_targets=True, **ds_kw)\n")
    assert src.count(call) == 1
    gate = 'if bool(getattr(a, "r9_perc_rig", False)):'
    assert src.count(gate) == 1 and src.index(gate) < src.index(call)
    assert src.count("rig_eval_dataset(") == 2                       # the def + the one gated call


def test_rig_eval_dataset_reads_no_label_and_sets_the_rig_row_flag(monkeypatch):
    G = NL._load("g_box_overfit_rig_ds_t", GBO_PATH)
    seen = {}

    class DS:
        def __init__(self, eps, **kw):
            seen["kw"], seen["eps"] = kw, eps
            self.agent_join = None
            self.map_clip_of_ep = None

        def __len__(self):
            return 3

    def boom(*a, **k):
        raise AssertionError("the rig dataset build must not enter the vendored label path")

    tr = types.SimpleNamespace(V3Dataset=DS)
    L = types.SimpleNamespace(trainer=lambda: tr, build_eval_dataset=boom)
    import tanitad.data.v2_dataset as v2d
    monkeypatch.setattr(v2d, "build_v2_providers", lambda caches, lru_size=0: [types.SimpleNamespace(episode_id=1)])
    cfg = types.SimpleNamespace(core=types.SimpleNamespace(window=3, encoder=types.SimpleNamespace(in_channels=9)))
    args = types.SimpleNamespace(eval_cache="x", v2_lru=4, u8_batches=True, agent_join=None, map_gt_root=None, join3d=None,
                                 graft_lan=False, goal_str=None)
    ds, eps, rec = G.rig_eval_dataset(L, cfg, args, {})
    assert seen["kw"] == {"window": 3, "max_horizon": 20, "channels": 9} and len(eps) == 1
    assert ds.u8_frames is True and ds.newest_row_only is True      # train(): bool(_r9rig_on and u8)
    assert rec["rig"] is True and rec["labels"] is None and "NOT ATTACHED" in rec["label_clock"]
    args.u8_batches = False
    ds2, _e, _r = G.rig_eval_dataset(L, cfg, args, {})
    assert ds2.newest_row_only is False
    with pytest.raises(SystemExit, match="dataset_cls"):
        G.rig_eval_dataset(L, cfg, args, {}, dataset_cls=object)
    args.graft_lan = True
    with pytest.raises(SystemExit, match="graft-lan"):
        G.rig_eval_dataset(L, cfg, args, {})
