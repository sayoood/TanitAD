"""``tanitad.refs.anchor_meta`` — the anchor artifact carries its OWN units.

MEASURED 2026-09-04 (D-REFCV4B-ANCHOR-UNITS): the live refcv4b ``anchors.pt``
is a dict with exactly ``anchors`` [117, 8, 2] and ``controls`` [117, 2]; column
1 of ``controls`` is lateral acceleration (m/s²) and nothing in the file said
so. Read as curvature the same bytes gave **396 g** at 36 m/s and 104/117
anchors over μ = 0.7; read correctly **0.31 g** and 0/117. Both tables looked
like answers. The run record was complete (``config.json['argv']`` carries
``--anchor-control-units alat``); only the standalone ``.pt`` was not.

Pinned here, one test per discipline:
(a) BUILD — :func:`build_anchor_artifact` writes ``control_units`` /
    ``horizon_s`` / ``dt`` / ``ref_speed_ms`` / ``kappa_cap`` / ``alat_v_floor``
    + a provenance stamp INTO the ``.pt``, derives ``horizon_s`` from the
    slots it actually has, and round-trips through
    ``torch.load(weights_only=True)``; it refuses a controls file that omits a
    re-roll constant, and a fixed-path file that claims control units.
(b) READ — a ``controls``-carrying file with no units is REFUSED, and the
    refusal names the incident (396 g / 0.31 g) and the override.
(c) OVERRIDE — the override loads a legacy file and is RECORDED as
    ``control_units_source='cli-override-legacy-file'`` (the live file's case);
    file + cli agreeing reads ``file+cli``; disagreeing is refused; a
    fixed-path file needs no units at all.
(d) MISMATCH — only DECLARED constants can mismatch the consumer's.
(e) ``scripts/build_refc_anchors.py`` writes the fields (``paths`` units).
(f) ``scripts/refc_v3_train.py`` — ``_seam_stamp`` carries the seam booleans
    that were absent from every refcv4b ``config.json``; a 1-step smoke
    ``train()`` refuses the legacy file without the override, accepts it with
    and stamps ``control_units_source`` + ``seams``; accepts a self-describing
    file with NO flag; refuses a conflicting flag; refuses a declared horizon
    the decoder does not have. ``--anchor-control-units`` defaults to None.
CPU-only, synthetic data. ⛔ The live ``anchors.pt`` is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_refc_anchors as B                               # noqa: E402
import refc_v3_train as T                                    # noqa: E402
from tanitad.refs import anchor_meta as am                   # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402

HZ = v3.V3_HORIZONS                     # (5, 10, 15, 20, 30, 40, 50, 60)
CONST = dict(ref_speed_ms=10.0, kappa_cap=0.12, alat_v_floor=4.0)


def _grid():
    """A 3×3 (a_lon, a_lat) grid that CONTAINS (0, 0) exactly, plus a
    placeholder bank of the decoder's [N, 8, 2] shape."""
    ctrl = torch.tensor([[a, c] for a in (-1.0, 0.0, 1.0)
                         for c in (-0.5, 0.0, 0.5)])
    t = torch.tensor([h * 0.1 for h in HZ])
    anchors = torch.zeros(9, len(HZ), 2)
    anchors[:, :, 0] = 10.0 * t
    return anchors, ctrl


def _self_describing(**over):
    a, c = _grid()
    kw = dict(control_units="alat", horizons=HZ, dt=0.1, builder=__file__,
              **CONST)
    kw.update(over)
    return am.build_anchor_artifact(a, c, **kw)


# ------------------------------------------------------------------ (a) build
def test_build_writes_required_fields_and_roundtrips(tmp_path):
    art = _self_describing()
    for k in am.REQUIRED_META:
        assert k in art, k
    assert art["schema"] == am.SCHEMA
    assert art["control_units"] == "alat"
    assert art["horizon_s"] == pytest.approx(6.0)       # max(HZ) * dt, DERIVED
    assert art["dt"] == 0.1 and art["horizons_steps"] == list(HZ)
    assert (art["ref_speed_ms"], art["kappa_cap"], art["alat_v_floor"]) \
        == (10.0, 0.12, 4.0)
    assert art["controls_columns"] == ["a_lon_ms2", "a_lat_ms2"]
    assert art["straight_ahead_control_present"] is True
    prov = art["provenance"]
    assert prov["builder"] == Path(__file__).name
    assert prov["builder_sha256"] and len(prov["builder_sha256"]) == 64
    assert prov["created_utc"] and prov["torch_version"] == torch.__version__
    p = tmp_path / "a.pt"
    torch.save(art, p)
    d = torch.load(p, weights_only=True)               # primitives only: safe
    assert torch.equal(d["anchors"], art["anchors"])
    assert torch.equal(d["controls"], art["controls"])
    assert d["control_units"] == "alat" and d["provenance"] == prov
    assert d["anchors_sha256"] == am.sha256_of_tensor(art["anchors"])


def test_build_refuses_what_a_consumer_could_not_re_roll():
    a, c = _grid()
    with pytest.raises(ValueError, match="must declare"):
        am.build_anchor_artifact(a, c, control_units="alat", horizons=HZ,
                                 builder=None, ref_speed_ms=10.0)   # no cap/floor
    with pytest.raises(ValueError, match="control_units"):
        am.build_anchor_artifact(a, c, control_units="paths", horizons=HZ,
                                 builder=None, **CONST)
    with pytest.raises(ValueError, match="fixed-path"):
        am.build_anchor_artifact(a, None, control_units="alat", horizons=HZ,
                                 builder=None)
    with pytest.raises(ValueError, match="horizons"):
        am.build_anchor_artifact(a, c, control_units="alat", horizons=HZ[:4],
                                 builder=None, **CONST)
    fixed = am.build_anchor_artifact(a, None, control_units=am.PATHS_ONLY,
                                     horizons=HZ, builder=None)
    assert fixed["control_units"] == "paths" and "controls" not in fixed
    assert fixed["ref_speed_ms"] is None and fixed["kappa_cap"] is None
    assert fixed["alat_v_floor"] is None and "path_units" in fixed


# ------------------------------------------------------------------- (b) read
def test_read_refuses_legacy_controls_file_and_names_the_incident(tmp_path):
    a, c = _grid()
    p = tmp_path / "legacy.pt"
    torch.save({"anchors": a, "controls": c}, p)      # EXACTLY the live shape
    with pytest.raises(am.AnchorUnitsMissing) as ei:
        am.read_anchor_artifact(p)
    msg = str(ei.value)
    assert "396 g" in msg and "0.31 g" in msg
    assert "--anchor-control-units" in msg
    assert "cli-override-legacy-file" in msg


# --------------------------------------------------------------- (c) override
def test_override_loads_legacy_and_is_recorded_as_such(tmp_path):
    a, c = _grid()
    p = tmp_path / "legacy.pt"
    torch.save({"anchors": a, "controls": c}, p)
    art = am.read_anchor_artifact(p, cli_control_units="alat")
    assert art.control_units == "alat"
    assert art.control_units_source == "cli-override-legacy-file"
    assert art.meta == {}
    assert all(v is None for v in art.declared.values())
    assert torch.equal(art.controls, c) and art.path == str(p)


def test_file_and_cli_matrix(tmp_path):
    p = tmp_path / "sd.pt"
    torch.save(_self_describing(), p)
    assert am.read_anchor_artifact(p).control_units_source == "file"
    both = am.read_anchor_artifact(p, cli_control_units="alat")
    assert both.control_units_source == "file+cli"
    assert both.declared["horizon_s"] == pytest.approx(6.0)
    with pytest.raises(am.AnchorUnitsConflict, match="declares control_units="):
        am.read_anchor_artifact(p, cli_control_units="kappa")
    with pytest.raises(ValueError, match="not in"):
        am.read_anchor_artifact(p, cli_control_units="radians")


def test_fixed_path_artifact_needs_no_units(tmp_path):
    a, _ = _grid()
    bare = am.read_anchor_artifact(a)                    # a bare tensor
    assert bare.controls is None and bare.control_units == "paths"
    assert bare.control_units_source == "n/a-fixed-paths"
    p = tmp_path / "fixed.pt"
    torch.save({"anchors": a, "method": "fps"}, p)      # pre-2026-09-05 dict
    art = am.read_anchor_artifact(p, cli_control_units="alat")   # ignored
    assert art.control_units == "paths" and art.meta == {"method": "fps"}


# --------------------------------------------------------------- (d) mismatch
def test_mismatches_report_only_declared_fields(tmp_path):
    a, c = _grid()
    legacy = am.read_anchor_artifact({"anchors": a, "controls": c},
                                     cli_control_units="alat")
    assert am.mismatches(legacy, horizon_s=2.0, ref_speed_ms=99.0) == []
    sd = am.read_anchor_artifact(_self_describing())
    assert am.mismatches(sd, horizon_s=6.0, dt=0.1, **CONST) == []
    bad = am.mismatches(sd, horizon_s=2.0, ref_speed_ms=12.0, kappa_cap=0.12)
    assert len(bad) == 2 and bad[0].startswith("horizon_s") \
        and bad[1].startswith("ref_speed_ms")


# ----------------------------------------------------------- (e) FPS builder
def test_build_refc_anchors_writes_a_self_describing_file(tmp_path):
    p = tmp_path / "fps.pt"
    B.main(["--out", str(p), "--smoke"])
    d = torch.load(p, weights_only=True)
    assert d["schema"] == am.SCHEMA and d["control_units"] == am.PATHS_ONLY
    assert d["horizon_s"] == pytest.approx(2.0)          # (5,10,15,20) * 0.1
    assert d["dt"] == 0.1 and d["horizons_steps"] == [5, 10, 15, 20]
    assert d["ref_speed_ms"] is None and d["kappa_cap"] is None
    assert d["provenance"]["builder"] == "build_refc_anchors.py"
    assert d["method"] == "fps" and d["n_anchors"] == 20   # legacy keys kept
    art = am.read_anchor_artifact(p)
    assert art.controls is None and tuple(art.anchors.shape) == (20, 4, 2)


# ---------------------------------------------------------------- (f) trainer
def test_seam_stamp_names_the_booleans_that_were_absent():
    cfg = v3.refc_v3_smoke_config(True)
    s = T._seam_stamp(cfg, argparse.Namespace(goal_str=True, graft_lan=False))
    for k in ("hierarchy", "graft_maneuver", "factored_maneuver",
              "graft_prior_center", "lan_enable", "goal_str"):
        assert k in s, k
    assert s["hier"] is True and s["hierarchy"] is True
    assert s["factored_maneuver"] is True            # forced by refc_v3.py
    assert s["graft_target_latent"] is True          # the registered delta
    assert s["lan_enable"] is True and s["goal_str"] is True
    assert s["graft_lan"] is False                   # E12: never an input
    flat = T._seam_stamp(v3.refc_v3_smoke_config(False),
                         argparse.Namespace(goal_str=False, graft_lan=False))
    assert flat["hier"] is False and flat["graft_target_latent"] is False
    assert flat["lan_enable"] is False and flat["goal_str"] is False


def test_control_units_flag_defaults_to_none_and_pin_tolerates_it(tmp_path):
    args = T.build_parser().parse_args(["--arm", "hier", "--out", str(tmp_path)])
    assert args.anchor_control_units is None
    cfg = T._pin_trainer_cfg(v3.refc_v3_smoke_config(True), argparse.Namespace(
        image_hw=None, anchor_v0_conditioned=True, n_anchors=9,
        _anchor_artifact_meta={"kappa_cap": 0.25, "alat_v_floor": 3.0}))
    assert cfg.core.anchors.control_units == "kappa"      # inert default
    assert cfg.core.anchors.kappa_cap == 0.25             # adopted from the file
    assert cfg.core.anchors.alat_v_floor_ms == 3.0


def _argv(out, anchors, *extra):
    return ["--arm", "hier", "--out", str(out), "--smoke",
            "--synth-episodes", "2", "--steps", "1", "--batch", "2",
            "--device", "cpu", "--log-every", "1", "--save-every", "1",
            "--anchors", str(anchors), "--anchor-v0-conditioned",
            "--n-anchors", "9", *extra]


def test_smoke_train_refuses_legacy_file_without_the_override(tmp_path):
    a, c = _grid()
    p = tmp_path / "legacy.pt"
    torch.save({"anchors": a, "controls": c}, p)
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="396 g"):
        T.train(T.build_parser().parse_args(_argv(out, p)))
    assert not (out / "config.json").exists()          # refused BEFORE data


def test_smoke_train_loads_legacy_file_with_override_and_stamps_it(tmp_path):
    a, c = _grid()
    p = tmp_path / "legacy.pt"
    torch.save({"anchors": a, "controls": c}, p)
    out = tmp_path / "run"
    T.train(T.build_parser().parse_args(
        _argv(out, p, "--anchor-control-units", "alat", "--goal-str")))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    an = cfg["anchors"]
    assert an["control_units"] == "alat" and an["v0_conditioned"] is True
    assert an["control_units_source"] == "cli-override-legacy-file"
    assert an["artifact_schema"] is None
    assert all(v is None for v in an["artifact_declared"].values())
    assert an["straight_ahead_control_present"] is True
    seams = cfg["seams"]
    for k in ("hierarchy", "graft_maneuver", "factored_maneuver",
              "graft_prior_center", "lan_enable", "goal_str"):
        assert isinstance(seams[k], bool), k
    assert seams["goal_str"] is True and seams["lan_enable"] is True
    assert seams["graft_lan"] is False and seams["hier"] is True
    assert json.loads((out / "summary.json").read_text())["done"] is True


def test_smoke_train_accepts_self_describing_file_without_any_flag(tmp_path):
    p = tmp_path / "sd.pt"
    torch.save(_self_describing(), p)
    out = tmp_path / "run"
    T.train(T.build_parser().parse_args(_argv(out, p)))
    an = json.loads((out / "config.json").read_text(encoding="utf-8"))["anchors"]
    assert an["control_units"] == "alat"
    assert an["control_units_source"] == "file"
    assert an["artifact_schema"] == am.SCHEMA
    assert an["artifact_declared"]["horizon_s"] == pytest.approx(6.0)
    assert an["artifact_declared"]["kappa_cap"] == 0.12
    assert an["artifact_provenance"]["builder"] == Path(__file__).name


def test_smoke_train_refuses_conflicting_flag_and_wrong_horizon(tmp_path):
    p = tmp_path / "sd.pt"
    torch.save(_self_describing(), p)
    with pytest.raises(SystemExit, match="declares control_units='alat'"):
        T.train(T.build_parser().parse_args(
            _argv(tmp_path / "r1", p, "--anchor-control-units", "kappa")))
    d = _self_describing()
    d["horizon_s"] = 2.0                                 # claims 2 s, has 6 s
    q = tmp_path / "h2.pt"
    torch.save(d, q)
    with pytest.raises(SystemExit, match="horizon_s"):
        T.train(T.build_parser().parse_args(_argv(tmp_path / "r2", q)))
    assert not (tmp_path / "r2" / "config.json").exists()


def test_grid_refusal_no_longer_prescribes_odd_counts(tmp_path):
    """RETRACTED 2026-09-05: 'Rebuild with odd counts' — np.linspace(-4, 3, 13)
    has 13 nodes and no zero. The refusal must ask for 0.0 to be a NODE."""
    a, c = _grid()
    c = c.clone()
    c[4] = torch.tensor([0.0, 0.25])                     # remove the (0, 0) row
    p = tmp_path / "nozero.pt"
    torch.save(_self_describing(), p)                    # shape-valid file …
    d = torch.load(p, weights_only=True)
    d["controls"] = c
    d["straight_ahead_control_present"] = False
    torch.save(d, p)
    with pytest.raises(SystemExit) as ei:
        T.train(T.build_parser().parse_args(_argv(tmp_path / "r", p)))
    msg = str(ei.value)
    assert "does not contain {a=0, kappa=0}" in msg
    assert "odd counts" not in msg
    assert "linspace(-4, 3, 13)" in msg and "0.0 is a node" in msg
