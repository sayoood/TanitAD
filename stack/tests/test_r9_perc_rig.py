"""refcv9 C7 -- the E1 perception-only rig (`--r9-perc-rig`). Every expectation is a LITERAL; each guard has a RED arm.
CPU, the map CI rig of test_map_only_ft (tiny REAL RefCV3Model, resnet18 @ 64x64, the 10 cm branch)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402


def _build_rig(monkeypatch, perc_rig=True):
    orig = v3.RefCV3Model
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: orig(cfg, perc_rig=perc_rig))
    return R._build()


def test_skipped_list_is_the_literal_27_roots():
    assert len(rig.RIG_SKIPPED_MODULES) == 27
    assert rig.RIG_BUILT_ROOTS == ("core.encoder", "core.agent_head", "_perception", "_map_hires")
    assert rig.is_skipped_key("core.decoder.layers.0.cross.in_proj_weight") and not rig.is_skipped_key("core.encoder.x")
    assert rig.is_skipped_key("goal_gate") and rig.is_skipped_key("scorer.goal_point.0.weight")


def test_rig_builds_only_the_perception_modules(monkeypatch):
    model, _ = _build_rig(monkeypatch)
    rec = rig.assert_rig_built(model)
    assert rec["n_keys"] == 163 and rec["n_params"] == 12_199_912
    for path in rig.PLANNER_ATTR_PATHS:
        assert not rig._has_path(model, path), path


def test_RED_a_rig_that_silently_builds_the_planner_turns_red(monkeypatch):
    model, _ = _build_rig(monkeypatch, perc_rig=False)          # the FULL build stands in for a rig that built the planner
    with pytest.raises(SystemExit, match="were BUILT"):
        rig.assert_rig_built(model)


def test_loss_is_perception_only_and_matches_the_full_model_map_term(monkeypatch):
    model, cfg = _build_rig(monkeypatch)
    batch = R._batch(cfg)
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)     # the hook routes to compute_losses_perc_rig
    assert set(out) >= {"loss", "map_hires"} and "traj" not in out and "cls" not in out
    full, fcfg = R._build()                                               # the tip's full build, same seed
    sd = {k: v for k, v in full.state_dict().items() if not rig.is_skipped_key(k)}
    model.load_state_dict(sd, strict=True)
    fout = T.compute_losses_v3(full, R._batch(fcfg), "cpu", log_metrics=False)
    assert float(out["map_hires"]) == pytest.approx(float(fout["map_hires"]), rel=1e-5)


def test_pin_refuses_every_planner_flag_and_no_perception_term():
    import argparse
    a = argparse.Namespace(r9_perc_rig=True, w_box3d=1.0, tac_decoder_v6=True)
    with pytest.raises(SystemExit, match="tac_decoder_v6"):
        rig.pin_args(a)
    with pytest.raises(SystemExit, match="no perception term"):
        rig.pin_args(argparse.Namespace(r9_perc_rig=True))
    rig.pin_args(argparse.Namespace(r9_perc_rig=True, w_box3d=1.0))      # admitted
    rig.pin_args(argparse.Namespace(r9_perc_rig=False, tac_decoder_v6=True))   # OFF: nothing


def test_warm_start_names_the_skipped_modules_and_refuses_strays(monkeypatch, tmp_path):
    model, _ = _build_rig(monkeypatch)
    sd = dict(model.state_dict())
    sd["core.decoder.traj_proj.weight"] = torch.zeros(3, 3)            # a SKIPPED module's key: named, ignored
    p = tmp_path / "ck.pt"
    torch.save({"model": sd, "step": 48852}, p)
    rec = rig.warm_start_perc_rig(model, str(p))
    assert rec["n_skipped_source_keys"] == 1 and rec["skipped_modules_key_counts"]["core.decoder"] == 1
    sd["core.not_a_module.w"] = torch.zeros(1)                          # RED: a stray key is refused, never ignored
    torch.save({"model": sd, "step": 1}, p)
    with pytest.raises(SystemExit, match="neither built nor in RIG_SKIPPED_MODULES"):
        rig.warm_start_perc_rig(model, str(p))


def test_default_build_has_no_rig_state(monkeypatch):
    model, _ = _build_rig(monkeypatch, perc_rig=False)
    assert model._perc_rig is False and model.core._perc_rig is False and "_perc_rig" not in model.state_dict()
