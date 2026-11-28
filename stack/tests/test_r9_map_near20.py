"""E1-NEAR20 (PREREG_M1_NEAR20): the banded map head's NEAR band end is configurable -- 10 m (default = M1 / M2, byte-identical to the tip) or 20 m
(arm M1N20: near 0-20 m @ 0.10 m = 200 x 600 = 120,000 cells, mid 20-40 m, far unchanged, C0's 20 m near lift).

The class this file exists for is "built, tested, and UNREACHABLE FROM ITS CALLER" (CLAUDE.md 2026-09-10/11): so it traces the flag to the loss
(argparse -> namespace -> MapHiresConfig -> built head -> forward -> the banded loss's near term -> backward), with a zero control, and it pins
every consumer of the band layout (planes, plan wires, G-DVB, the launch gate, make_argv, the verdict row). MUTATIONS live in
``tests/mutations_near20.py`` (run by hand, each must go RED) and in the ``test_MUT_*`` cases here that need no file edit.
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

import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_V2  # noqa: E402
from tanitad.eval import map_planes as MP  # noqa: E402
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import plan_wires_r9 as PW  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_banded_flag as MF  # noqa: E402

X, YH = 100.0, 30.0
BASE_ARGV = ["--arm", "flat", "--out", "x", "--map-hires", "on", "--w-map-hires", "1.0", "--r9-map-banded"]


def _args(*extra, near_lift="10"):
    return T.build_parser().parse_args(BASE_ARGV + ["--map-hires-near-lift-m", near_lift, *extra])


# =========================================================================== #
# geometry                                                                     #
# =========================================================================== #
def test_geometry_at_10_is_the_tips_and_at_20_is_the_preregistered_layout():
    g10, g20 = MB.band_geometry(X, YH), MB.band_geometry(X, YH, 20.0)
    assert g10 == MB.band_geometry(X, YH, 10.0)
    assert [(g.rows, g.cols, g.n_cells) for g in g10] == [(100, 600, 60_000), (120, 240, 28_800), (120, 120, 14_400)]
    assert [(g.rows, g.cols, g.n_cells) for g in g20] == [(200, 600, 120_000), (80, 240, 19_200), (120, 120, 14_400)]
    assert [(g.x0, g.x1) for g in g20] == [(0.0, 20.0), (20.0, 40.0), (40.0, 100.0)] and g20[0].cell == 0.10
    assert g20[2] == g10[2] and MB.band_starts(10.0) == MB.BAND_START_M and MB.band_starts(20.0) == (0.0, 20.0, 40.0)


@pytest.mark.parametrize("bad", [15.0, 0.0, 30.0, float("nan"), "x", None])
def test_geometry_refuses_a_near_end_outside_the_preregistered_set(bad):
    with pytest.raises(ValueError, match="near_m"):
        MB.band_geometry(X, YH, bad)


# =========================================================================== #
# refusals: config, argv                                                       #
# =========================================================================== #
def test_config_refuses_a_near_lift_that_is_not_the_near_band_and_a_near_end_without_the_banded_head():
    with pytest.raises(ValueError, match="near_lift_x_m 20"):
        H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=10.0, banded=True, banded_near_m=20.0)
    with pytest.raises(ValueError, match="near_lift_x_m 10"):
        H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=20.0, banded=True)                     # the tip's refusal, message unchanged
    with pytest.raises(ValueError, match="without banded"):
        H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=20.0, banded=False, banded_near_m=20.0)
    with pytest.raises(ValueError, match="near_m"):
        H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=15.0, banded=True, banded_near_m=15.0)
    c20 = H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=20.0, banded=True, banded_near_m=20.0)
    assert c20.as_dict()["banded_near_m"] == 20.0 and c20.near_rows == 200
    assert "banded_near_m" not in H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=10.0, banded=True).as_dict()      # the default's stamp is the tip's


def test_argv_refusals():
    MF.pin_args(_args())                                                                        # M1: unset, near lift 10
    MF.pin_args(_args("--r9-map-banded-near-m", "10"))
    MF.pin_args(_args("--r9-map-banded-near-m", "20", near_lift="20"))                          # M1N20
    for extra, nl, why in ((("--r9-map-banded-near-m", "20"), "10", "not 20"),                  # 20 with near lift 10
                           (("--r9-map-banded-near-m", "10"), "20", "not 10"),                  # the reverse
                           (("--r9-map-banded-near-m", "15"), "15", "pre-registered")):
        with pytest.raises(SystemExit, match=why):
            MF.pin_args(_args(*extra, near_lift=nl))
    a = T.build_parser().parse_args(["--arm", "flat", "--out", "x", "--map-hires", "on", "--w-map-hires", "1.0", "--map-hires-near-lift-m", "20",
                                     "--r9-map-banded-near-m", "20"])                           # a non-banded arm carrying the flag
    with pytest.raises(SystemExit, match="declared and never built"):
        MF.pin_args(a)
    assert MF.near_m(_args()) == 10.0 and MF.near_m(_args("--r9-map-banded-near-m", "20", near_lift="20")) == 20.0


def test_the_flag_is_attached_in_the_trainer_and_the_loader_and_the_harnesses():
    """Source-hop pins (the argv -> config hops that a unit test of the head cannot see): delete one and the lever is declared and not built."""
    src = {p: (ROOT / p).read_text(encoding="utf-8") for p in ("scripts/refc_v3_train.py", "tanitad/eval/refcv7_loader.py", "scripts/g_box_overfit.py",
                                                              "scripts/map_hires_overfit.py", "scripts/r9_e1_dump.py")}
    assert "banded_near_m=float(_mbf.near_m(args))" in src["scripts/refc_v3_train.py"]
    assert "near_m=float(br.cfg.banded_near_m)" in src["scripts/refc_v3_train.py"]                  # the loss's targets follow the BUILT head
    assert "banded_near_m=float(tr._mbf.near_m(args))" in src["tanitad/eval/refcv7_loader.py"]       # G-EVAL builds the identical model
    assert "banded_near_m=float(tr._mbf.near_m(a))" in src["scripts/g_box_overfit.py"]
    assert "banded_near_m=near_m" in src["scripts/map_hires_overfit.py"] and "near_m = MB.check_near_m(a.near_lift_m)" in src["scripts/map_hires_overfit.py"]
    assert 'near_m=float(getattr(_cfg, "banded_near_m", 10.0))' in src["scripts/r9_e1_dump.py"]
    assert T.build_parser().get_default("r9_map_banded_near_m") is None


# =========================================================================== #
# the built model + the trace to the loss                                      #
# =========================================================================== #
def _build(near_m: float, banded_near_m=None):
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k", trunk_pretrained=False,
                                  in_channels=9, image_size=64, image_width=None, trunk_frozen_bn=True)
    model = v3.RefCV3Model(cfg)
    model.core.encoder.enable_s8_tap()
    kw = {} if banded_near_m is None else {"banded_near_m": banded_near_m}
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(EXTENT_V2.x_max_m), y_half_m=float(EXTENT_V2.y_half_m), d_lift=16, d_model=16, d_up=16,
                            dilations=(1, 2), grad_ckpt=True, near_lift_x_m=near_m, near_refine_blocks=1, banded=True, **kw)
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = 1.0
    setattr(model, MF.MODEL_ATTR, True)
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank({R.CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d = 0.0, 0.0
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():            # a TRAINED near lift / refine (zero-initialised at build: their grads would read 0)
        for n, p in model.named_parameters():
            if n.startswith(("_map_hires.near.", "_map_hires.near_refine.")):
                p.add_(0.02 * torch.randn(p.shape, generator=g))
    return model, cfg


def _losses(model, cfg):
    torch.manual_seed(0)
    model.train()
    return T.compute_losses_v3(model, R._batch(cfg), "cpu", mode="diffusion")


def test_built_head_follows_the_config_at_10_and_20_and_the_state_dict_is_the_tips_at_10():
    m10, _ = _build(10.0)
    m10b, _ = _build(10.0, banded_near_m=10.0)
    assert list(m10.state_dict()) == list(m10b.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(m10.state_dict().values(), m10b.state_dict().values()))     # UNSET == explicit 10, bit for bit
    m20, _ = _build(20.0, banded_near_m=20.0)
    b10, b20 = m10._map_hires, m20._map_hires
    assert [tuple(g[:1]) + (g.rows,) for g in b10.banded.geom][0][1:] == (100,) and b20.banded.geom[0].rows == 200 and b20.banded.near_m == 20.0
    # the parameters of the banded head are the same shapes at 10 and 20 (all convs are fully convolutional): only the near lift's ROWS differ
    s10, s20 = ({k: tuple(v.shape) for k, v in m.state_dict().items() if "banded" in k} for m in (m10, m20))
    assert s10 == s20
    assert b10.cfg.near_rows == 100 and b20.cfg.near_rows == 200 and b20.near is not None


def test_TRACE_the_20_m_near_band_reaches_the_loss_with_a_nonzero_gradient_and_a_zero_control(monkeypatch):
    """argv -> namespace -> MapHiresConfig -> built head -> forward -> the banded loss's near term -> backward. The probe ISOLATES the new rows:
    near band rows 100-199 (10-20 m) are the only supervised cells; every parameter of the near path must then read NON-ZERO, and with those same rows
    unobserved (max-entropy off) the SAME parameters read EXACTLY 0.0. Under 10 those rows do not exist."""
    args = _args("--r9-map-banded-near-m", "20", near_lift="20")
    assert MF.near_m(args) == 20.0                                                              # argparse -> namespace
    model, cfg = _build(float(args.map_hires_near_lift_m), banded_near_m=MF.near_m(args))      # namespace -> config -> built head
    assert D.check(model, args) == [] or all("banded_near_m" not in str(m) for m in D.check(model, args))
    real = MB.unobs_masks
    mask = {"obs_rows": True}

    def only_rows_100_199(v, geom):
        out = [torch.ones_like(m) for m in real(v, geom)]                                      # everything unobserved ...
        if mask["obs_rows"]:
            out[0][:, 100:200] = False                                                          # ... except near rows 100-199
        return out
    monkeypatch.setattr(MB, "unobs_masks", only_rows_100_199)
    monkeypatch.setattr(MB, "MAXENT_W", 0.0)
    names = [n for n, _ in model.named_parameters() if n.startswith(("_map_hires.banded.head_near", "_map_hires.near.", "_map_hires.near_refine."))]
    assert names
    losses = _losses(model, cfg)                                                                # forward (the trainer's own path)
    assert float(losses["map_banded_n_near"]) > 0 and float(losses["map_banded_n_mid"]) == 0 and float(losses["map_banded_n_far"]) == 0
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()                                                                  # backward
    P = dict(model.named_parameters())
    dead = [n for n in names if P[n].grad is None or float(P[n].grad.abs().sum()) == 0.0]
    assert dead == [], dead
    mask["obs_rows"] = False                                                                    # CONTROL: rows 100-199 unobserved too
    model.zero_grad(set_to_none=True)
    losses = _losses(model, cfg)
    losses["loss"].backward()
    P = dict(model.named_parameters())
    assert float(losses["map_hires"]) == 0.0 and [n for n in names if P[n].grad is not None and float(P[n].grad.abs().sum()) != 0.0] == []


def test_MUT_a_model_that_ignores_the_near_end_is_caught(monkeypatch):
    """Mutation: the banded head's geometry ignores the near end (always 10 m) while the config says 20 -> G-DVB reads 100 near rows, declared 200."""
    real = MB.band_geometry
    monkeypatch.setattr(MB, "band_geometry", lambda x, y, near_m=10.0: real(x, y, 10.0))
    model, _ = _build(20.0, banded_near_m=20.0)
    assert model._map_hires.banded.geom[0].rows == 100
    bad = MF.dvb_near_m(model, _args("--r9-map-banded-near-m", "20", near_lift="20"))
    assert bad and any("near" in str(m) for m in bad)
    monkeypatch.undo()
    clean, _ = _build(20.0, banded_near_m=20.0)
    assert MF.dvb_near_m(clean, _args("--r9-map-banded-near-m", "20", near_lift="20")) == []


# =========================================================================== #
# G-DVB                                                                        #
# =========================================================================== #
def test_dvb_catches_a_built_vs_argv_mismatch_both_ways_and_is_registered():
    reg = {}
    MF.register_dvb_levers(lambda dest, kind, check=None, reason="": reg.update({dest: (kind, check)}))
    assert reg[MF.NEAR_M_DEST][0] == "built" and reg[MF.DEST][0] == "built"
    assert MF.NEAR_M_DEST in {a.dest for a in T.build_parser()._actions}
    m10, _ = _build(10.0)
    m20, _ = _build(20.0, banded_near_m=20.0)
    a10, a20 = _args(), _args("--r9-map-banded-near-m", "20", near_lift="20")
    assert MF.dvb_near_m(m10, a10) == [] and MF.dvb_near_m(m20, a20) == [] and MF.dvb_check(m20, a20) == [] and MF.dvb_check(m10, a10) == []
    assert MF.dvb_near_m(m10, a20) and MF.dvb_near_m(m20, a10)                                  # argv 20 / built 10, and argv unset / built 20
    assert MF.dvb_check(m10, a20) and MF.dvb_check(m20, a10)
    assert D.coverage(T.build_parser()) == [] or MF.NEAR_M_DEST not in D.coverage(T.build_parser())


# =========================================================================== #
# consumers: planes (the dump), plan wires                                     #
# =========================================================================== #
def _synthetic_banded(near_m: float, B: int = 1):
    """A banded output whose drv logit is +9 in a distinct band per x range, so the raster says WHICH band each metre was read from."""
    geom = MB.band_geometry(X, YH, near_m)
    zs, un = [], []
    for g, v in zip(geom, (9.0, -9.0, 0.0)):                    # near +9, mid -9, far 0
        z = torch.zeros(B, MB.N_CHANNELS, g.rows, g.cols)
        z[:, 0] = v
        zs.append(z)
        un.append(torch.zeros(B, g.rows, g.cols, dtype=torch.bool))
    return {"logits": zs, "unobs": un}


def test_dump_planes_read_10_to_20_m_from_the_near_band_under_20_and_from_the_mid_band_under_10():
    for nm, near_cells_x in ((10.0, 100), (20.0, 200)):
        mb = _synthetic_banded(nm)
        for kw in ({"near_m": nm}, {}):                                                          # the config's value, and the shape-derived default
            pl = MP.planes_from_banded(mb, x_max_m=X, y_half_m=YH, **kw)
            pd = pl.pdrv[0, :, 0]                                                                # sigmoid(drv) down the centre column, 1000 rows of 0.1 m
            assert bool((pd[:near_cells_x] > 0.99).all()) and bool((pd[near_cells_x:near_cells_x + 50] < 0.01).all()), (nm, kw)
    p20 = MP.planes_from_banded(_synthetic_banded(20.0), x_max_m=X, y_half_m=YH, near_m=20.0).pdrv[0, :, 0]
    assert bool((p20[100:200] > 0.99).all())                                                    # 10-20 m from the NEAR band under 20
    p10 = MP.planes_from_banded(_synthetic_banded(10.0), x_max_m=X, y_half_m=YH, near_m=10.0).pdrv[0, :, 0]
    assert bool((p10[100:200] < 0.01).all())                                                    # ... from the MID band under 10
    with pytest.raises(ValueError, match="band"):                                              # a layout that is not the logits': refused, never a silent mis-resample
        MP.planes_from_banded(_synthetic_banded(20.0), x_max_m=X, y_half_m=YH, near_m=10.0)


def test_plan_wires_bands_follow_the_logits_not_a_constant():
    for nm, want in ((10.0, [(0.0, 10.0), (10.0, 40.0), (40.0, 100.0)]), (20.0, [(0.0, 20.0), (20.0, 40.0), (40.0, 100.0)])):
        got = PW._geoms(_synthetic_banded(nm))
        assert [(g[0], g[1]) for g in got] == want
    ch10, ch20 = PW.channels_1m(_synthetic_banded(10.0)), PW.channels_1m(_synthetic_banded(20.0))
    assert tuple(ch10.shape) == tuple(ch20.shape) == (1, 5, 100, 60)                            # the 1 m field is the full extent either way
    assert bool((ch20[0, 0, 10:20] > 0.99).all()) and bool((ch10[0, 0, 10:20] < 0.01).all())


# =========================================================================== #
# the arm, the gate, the verdict                                               #
# =========================================================================== #
def test_make_argv_M1N20_is_M1_plus_exactly_the_near_end_and_the_near_lift():
    import r9_e1_make_argv as MA
    base = json.loads((ROOT / "ops/runs.d/r9e1_C0.argv.json").read_text(encoding="utf-8"))["argv"]
    m1, n20 = MA.map_arm_argv(base, "M1"), MA.map_arm_argv(base, "M1N20")
    assert m1.count("--r9-map-banded-near-m") == 0 and n20[n20.index("--r9-map-banded-near-m") + 1] == "20"
    assert n20[n20.index("--map-hires-near-lift-m") + 1] == "20" and m1[m1.index("--map-hires-near-lift-m") + 1] == "10"
    drop = lambda a: [t for i, t in enumerate(a) if not (t in ("--r9-map-banded-near-m",) or (i and a[i - 1] == "--r9-map-banded-near-m"))]    # noqa: E731
    same = lambda a: [("10" if a[i - 1] == "--map-hires-near-lift-m" else t) for i, t in enumerate(a)]                                         # noqa: E731
    out = lambda a: [t for i, t in enumerate(a) if not (i and a[i - 1] == "--out") and t != "--out"]                                           # noqa: E731
    assert out(same(drop(n20))) == out(m1)                                                       # nothing else differs
    MA.check_map_arm_pins(n20, "M1N20")
    MA.check_map_arm_pins(m1, "M1")
    with pytest.raises(SystemExit):
        MA.check_map_arm_pins(m1, "M1N20")                                                       # the flag dropped: refused
    f = ROOT / "ops/runs.d/r9e1_M1N20.argv.json"
    if f.exists():
        rec = json.loads(f.read_text(encoding="utf-8"))
        assert rec["arm"] == "r9e1_M1N20" and rec["argv"] == MA.map_arm_argv(base, "M1N20") and rec["lever"]["near_m"] == 20.0


def test_MUT_make_argv_that_drops_the_near_lift_is_refused_by_the_pins(monkeypatch):
    import r9_e1_make_argv as MA
    base = json.loads((ROOT / "ops/runs.d/r9e1_C0.argv.json").read_text(encoding="utf-8"))["argv"]
    monkeypatch.setitem(MA.NEAR20_ARMS, "M1N20", {"w_map_hires": 1.0, "near_m": "20", "near_lift_m": "10"})        # the near lift left at 10
    with pytest.raises(SystemExit, match="near-lift"):
        MA.check_map_arm_pins(MA.map_arm_argv(base, "M1N20"), "M1N20")


def test_gate_holds_M1N20_to_near_lift_20_and_M1_to_10_and_the_record_table_follows_the_argv():
    import launch_gate as LG
    prof = LG.PROFILES["refcv9-e1rig"]
    n20 = ["--r9-map-banded", "--r9-map-banded-near-m", "20", "--map-hires-near-lift-m", "20"]
    assert LG.trigger_on(n20, "--r9-map-banded-near-m=20") and not LG.trigger_on(["--r9-map-banded-near-m", "10"], "--r9-map-banded-near-m=20")
    assert not LG.trigger_on(["--r9-map-banded"], "--r9-map-banded-near-m=20")
    r_ok, d_ok = LG.required_on_reasons(prof, n20 + ["--w-map-hires", "1.0"])
    assert d_ok["required_values"]["--map-hires-near-lift-m"] == "20"
    r_bad, d_bad = LG.required_on_reasons(prof, ["--r9-map-banded", "--map-hires-near-lift-m", "20", "--w-map-hires", "1.0"])
    assert any("--map-hires-near-lift-m 10 is REQUIRED" in r for r in r_bad)                      # M1 with near lift 20: still refused
    assert "--r9-map-banded-near-m" in LG.E1_OWN_RECORDS["E1-MAP-OWN-RECORD"]["flags"]            # reachability: the own record can CLOSE the item
    bd = prof["map_overfit_banded"]
    assert LG.banded_table_for(bd, ["--r9-map-banded"]) is bd and LG.banded_table_for(bd, ["--r9-map-banded-near-m", "10"]) is bd
    t20 = LG.banded_table_for(bd, n20, prof["map_overfit"])
    assert t20["near_lift_m"] == 20.0 and t20["gate_rows"] == {"near": 200, "mid": 0, "far": 0} and t20["gated_bands"] == ("near",)
    assert bd["gate_rows"] == {"near": 100, "mid": 40, "far": 0}                                  # the default table is untouched
    assert LG.banded_arg(prof, n20 + ["--map-hires", "on"]) is None or len(LG.banded_arg(prof, n20 + ["--map-hires", "on"])) in (2, 3)
    ok, det = LG.banded_output_reasons({"perception": {"map_hires_logits": None, "map_banded": _synthetic_banded(20.0)}}, X, YH, 20.0)
    assert ok == [], ok
    bad, _ = LG.banded_output_reasons({"perception": {"map_hires_logits": None, "map_banded": _synthetic_banded(20.0)}}, X, YH, 10.0)
    assert bad                                                                                   # a 20 m head judged against the 10 m layout is a named FAIL


def test_verdict_has_the_M1N20_row_with_M1s_metrics_reference_C0_and_clause3_false_edge_rows():
    import r9_e1_verdict as V
    a, m1 = V.ARMS["M1N20"], V.ARMS["M1"]
    assert a["ref"] == "C0" and a["allowed"] == ("C0", "M1") and a["kind"] == "lever"
    strip = lambda ms, arm: [(m["role"], m["clause3"], tuple(m["keys"])) for m in ms]            # noqa: E731
    assert strip(a["metrics"], "M1N20") == strip(m1["metrics"], "M1")                          # M1's metrics (keys, roles, clause-3 flags) unchanged
    beside = [m for m in a["metrics"] if not m["clause3"]]
    assert len(beside) == 17 and all(m["role"] == "beside" for m in beside)                    # 15 edge rows + the 2 D3 rows (lane, boundary)
    assert {m["keys"][0] for m in beside} >= {V.D3_LANE_KEY, V.D3_BOUNDARY_KEY} and all(m["id"].startswith("M1N20.") for m in a["metrics"])
    assert "--r9-map-banded-near-m" in a["argv_flags"]


# =========================================================================== #
# the record's caller: the bind script renders the ARGV's near lift (MM 2026-11-07)  #
# =========================================================================== #
# The bind template carried a literal `--near-lift-m 10`, and `map_hires_overfit.py` builds its head BY HAND from that argument: M1N20's
# G-MAP-OVERFIT record would have been built at 10 m, bound to its 20 m argv, and refused by the gate after ~74 GPU-min. The lever was built,
# tested and unreachable from the record's caller. Now `{NEAR_LIFT}` = the gate's own `banded_near_m_of(argv)`, and the harness refuses a mismatch.
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import map_hires_overfit as O  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
_A18 = ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A18.json"
_BASH = shutil.which("bash")
_HASH = "4e36c721d0e1c8befe3371b73f92e4a455ff4732"


def _bind_dry(tmp_path, arm):
    env = dict(os.environ, PYTHON=sys.executable, MSYS_NO_PATHCONV="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [_BASH, (ROOT / "ops" / "r9e1_bind.sh").as_posix(), "--dry-run", "map-banded", arm, "--commit", _HASH, "--root",
           (tmp_path / "root").as_posix(), "--code", ROOT.parent.as_posix(), "--inputs", _A18.parent.as_posix()]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    return r.stdout.replace("\r", ""), tmp_path / "root" / f"ovf_{arm}_{_HASH[:12]}" / "map-banded.args"


@pytest.mark.skipif(not _BASH or not _A18.is_file(), reason="needs bash and the registered A18 spec (TanitAD Research Lab) in the tree")
@pytest.mark.parametrize("arm,near", [("M1", "10"), ("M2a", "10"), ("M1N20", "20")])
def test_bind_renders_the_argvs_near_band_end_into_the_banded_records_harness_args(tmp_path, arm, near):
    out, args = _bind_dry(tmp_path, arm)
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out, out
    assert f"banded near band 0-{near} m" in out
    toks = args.read_text(encoding="utf-8").replace("\r", "").split("\n")
    assert toks.count("--near-lift-m") == 1 and toks[toks.index("--near-lift-m") + 1] == near
    assert not any("{" in t and "}" in t for t in toks)


def test_the_template_has_no_literal_near_lift_and_the_bind_script_fills_it_from_the_gates_own_reader():
    tpl = (ROOT / "scripts" / "r9e1_bind_map-banded.args.tpl").read_text(encoding="utf-8").replace("\r", "")
    assert "--near-lift-m\n{NEAR_LIFT}\n" in tpl and "--near-lift-m\n10\n" not in tpl
    sh = (ROOT / "ops" / "r9e1_bind.sh").read_text(encoding="utf-8").replace("\r", "")
    assert '--sub "NEAR_LIFT=$NL"' in sh and "L.banded_near_m_of(L.load_argv_file(sys.argv[2]))" in sh


def test_the_harness_refuses_a_near_lift_that_is_not_the_launch_argvs_near_band_end():
    assert O.NEAR_M_FLAG == MF.NEAR_M_FLAG
    m1, m1n20 = O.launch_argv_facts(RUNS / "r9e1_M1.argv.json"), O.launch_argv_facts(RUNS / "r9e1_M1N20.argv.json")
    assert O.argv_banded_near_m(m1["argv"]) == 10.0 and O.argv_banded_near_m(m1n20["argv"]) == 20.0
    O.check_near_vs_argv(10.0, m1)
    O.check_near_vs_argv(20.0, m1n20)
    O.check_near_vs_argv(10.0, None)                                                            # no argv: nothing to check
    for nm, facts in ((10.0, m1n20), (20.0, m1)):
        with pytest.raises(SystemExit, match="does not train"):
            O.check_near_vs_argv(nm, facts)
    assert O.argv_banded_near_m(["--r9-map-banded", "--r9-map-banded-near-m=20"]) == 20.0
    src = Path(O.__file__).read_text(encoding="utf-8")
    i = src.index("own_enc, own_enc_source = resolve_own_enc(bool(a.own_enc), facts)")
    assert src.index("check_near_vs_argv(near_m, facts)", i) < src.index("from tanitad.models import timm_trunk as TT", i)   # the banded run CALLS it
