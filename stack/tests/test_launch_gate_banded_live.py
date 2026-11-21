"""BANDED-LIVE (MM 2026-11-05) -- the launch gate's NEW-2 / G-MAP LIVENESS checks for the BANDED map head (`--r9-map-banded`).

MEASURED 2026-11-05 01:12Z on Thor: E1 arm M1 (profile refcv9-e1rig) FAILED its final gate on checks that only knew the inherited 8-code `map_hires`
head -- G-DVB / G-EVAL "NEW-2: no forward output named with 'hires' and 'logit'", G-LIVE "the per-class probe never ran", "160 of the 160 per-class
argmax COUNT keys are absent", "no labelled 0_20 m cells for [8 classes]", "2 of 2 training rows lack `map_hires_lc_{cls}_{band}`". The banded head
returns `map_hires_logits: None` BY DESIGN and logs `map_banded_*` telemetry, so NO banded argv (M1, M2a; later E2 K2 / E3) could pass.

The fixture `tests/data/launch_gate_banded_m1_smoke_rows.jsonl` is M1's REAL Thor smoke `metrics.jsonl` (first run: train steps 50 and 100 and the in-run
eval at step 100; sha256 of the source file eb8910b677e15329865ee9530b0495594f7146f3d3d7ba303f51db4cc1a4892e), reduced to the keys the checks read: `step`,
`loss`, `eval_loss`, `[eval_]map_hires`, `[eval_]map_hires_n_labelled`, every key containing `map_banded` (102 per row, x3 prefixes on the eval row) and the
`eval_map_hires_iou_*` / `eval_map_hires_lshare_*` keys (null under the banded head -- the OLD checks read them, so the mutation arm reproduces Thor's reasons).

  P   the banded checks PASS on the real rows (eval + EMA eval, training rows, the loss-sum identity, spelling vs `telemetry_keys`)
  R   RED arms, each FAILING with its own reason: a banded loss key dropped, a count key dropped, a NaN, only 2 bands of logits, a wrong band shape,
      `map_hires_logits` not None, an unlabelled band, a broken loss sum, an EMA key dropped, a dead channel, a non-finite gradient, no targets, no probe
  M   MUTATION: delete the banded branch (`banded_on` -> False) and the M1 rows FAIL AGAIN with Thor's reasons -- G-MAP items and the output probe alike
  N   a NON-banded argv is judged by the 8-code functions it always used, unchanged
  W   the call sites are in place (source needles on `run_smoke` / `job_smoke` / `job_model`), each hop pinned
The output probe and the gradient probe also run on a REAL CPU-built banded model (the tiny resnet18 @ 64x64 rig of test_map_banded_wiring).
G-LIVE itself needs Thor CUDA: the MM re-runs M1's gate on Thor after landing."""
from __future__ import annotations

import copy
import inspect
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.train import map_banded_flag as MF  # noqa: E402

RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"
FIX = ROOT / "tests" / "data" / "launch_gate_banded_m1_smoke_rows.jsonl"
BANDS, CHANNELS, PROB = MB.BAND_NAMES, MB.CHANNELS, MB.PROB_CHANNELS


def argv(name: str) -> list[str]:
    return list(json.loads((RUNS / f"r9e1_{name}.argv.json").read_text(encoding="utf-8"))["argv"])


M1, M2A, C0 = argv("M1"), argv("M2a"), argv("C0")
CFG = {"map_hires": {"decision_rule": "prior_corrected"}}
EXT = LG.banded_extent(RIG, M1)


def fixture_rows() -> list[dict]:
    return [json.loads(l) for l in FIX.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture()
def rows():
    return copy.deepcopy(fixture_rows())


def trains(rs):
    return LG.train_rows(rs)


def evalrow(rs):
    return [r for r in rs if any(str(k).startswith("eval_") for k in r)]


def good_stats(*, bands=BANDS, grad=1.0, steps=2, raster=(1000, 600)) -> dict:
    """What `observe_banded_forward` records on a healthy smoke (synthetic: the real-model arms are below)."""
    st = {"_fwd": 2, "targets": {"steps": steps, "labelled_windows": 28, "raster_shape": list(raster)}, "bands": {}}
    for b in bands:
        st["bands"][b] = {"hooked": 2, "grad_obs": {c: grad for c in CHANNELS}, "grad_unobs": {c: 0.1 for c in CHANNELS},
                          "nonfinite": {c: 0 for c in CHANNELS}}
    return st


def live(argv_, rs, stats, cfg=CFG, prof=RIG):
    return LG.map_live_items(prof, argv_, {"map_banded": stats, "map_class": {}}, rs, cfg)


# ======================================================================================= pins and the fixture ==
def test_the_flag_literal_is_the_trainers_and_the_share_alarms():
    assert LG.BANDED_FLAG == MB.FLAG == MF.FLAG == LG._R9F_SHARE["flag"] == "--r9-map-banded"


def test_the_fixture_is_the_reduced_real_smoke_rows(rows):
    assert len(rows) == 3 and [r.get("step") for r in rows] == [50, 100, 100]
    assert FIX.stat().st_size < 40_000
    exact = {"step", "loss", "eval_loss", "map_hires", "map_hires_n_labelled", "eval_map_hires", "eval_map_hires_n_labelled"}
    for r in rows:
        assert all(k in exact or "map_banded" in k or k.startswith(("eval_map_hires_iou_", "eval_map_hires_lshare_")) for k in r)
    assert sum("map_banded" in k for k in rows[0]) == 102 and sum("map_banded" in k for k in rows[2]) == 204
    assert not any("map_hires_lc_" in k or "map_hires_inter_" in k or "map_hires_n_nocls" in k for r in rows for k in r)   # the 8-code keys Thor lacked


def test_banded_on_reads_the_argv_flag_and_the_extent_from_the_run():
    assert LG.banded_on(RIG, M1) and LG.banded_on(RIG, M2A) and EXT == (100.0, 30.0) == LG.banded_arg(RIG, M1)
    for arm in ("C0", "C1", "B1", "B2", "T1"):
        a = argv(arm)
        assert not LG.banded_on(RIG, a) and LG.banded_arg(RIG, a) is None
    assert LG.banded_extent(RIG, LG.set_flag(M1, "--map-hires-x-max-m", ["60"]))[0] == 60.0
    assert LG.banded_extent(RIG, LG.set_flag(M1, "--map-hires-x-max-m", ["abc"])) == (100.0, 30.0)     # a non-numeric flag falls back, never raises


# ============================================================================================ P: the PASS arms ==
def test_P_the_real_m1_rows_pass_every_banded_item(rows):
    r, d = live(M1, rows, good_stats())
    assert r == [], r
    assert set(d) == {"classes", "eval_keys", "train_rows", "decision_rule", "spelling"}
    assert LG.banded_eval_key_reasons(RIG, evalrow(rows), ema=True)[0] == []
    r, d = LG.banded_train_row_reasons(RIG, trains(rows))
    assert r == [] and d["n_rows"] == 2 and d["keys_per_row"] == 39 and d["rows_sum_mismatch"] == []
    r, d = LG.banded_eval_key_reasons(RIG, evalrow(rows), ema=True)
    assert d["expected_per_prefix"] == 102 and d["prefixes"] == ["eval_", "ema_eval_"] and d["bands_without_cells"] == []


def test_P_the_names_come_from_the_modules_own_constants_not_a_second_list():
    assert set(LG.banded_train_keys(MB)) == set(MB.telemetry_keys(with_metrics=False))
    assert set(LG.banded_train_keys(MB)) | set(LG.banded_metric_keys(MB)) == set(MB.telemetry_keys(with_metrics=True))
    assert len(MB.telemetry_keys(True)) == 102 and LG.banded_spelling_reasons(RIG)[0] == []
    src = inspect.getsource(LG.banded_train_keys) + inspect.getsource(LG.banded_metric_keys) + inspect.getsource(LG.judge_banded_classes)
    for band in ("near", "mid", "far"):
        assert f'"{band}"' not in src and f"'{band}'" not in src            # no hand-written band name in the banded key / probe code


def test_P_the_loss_sum_identity_holds_on_the_real_rows(rows):
    for r in trains(rows) + [rows[2]]:
        pre = "eval_" if "eval_loss" in r else ""
        tot = sum(w * (sum(r[f"{pre}map_banded_loss_{c}_{b}"] for c in CHANNELS) + MB.MAXENT_W * r[f"{pre}map_banded_maxent_{b}"])
                  for b, w in zip(BANDS, MB.BAND_WEIGHTS))
        assert abs(tot - r[f"{pre}map_hires"]) < 1e-5 * abs(r[f"{pre}map_hires"])


def test_P_the_not_applicable_items_say_so_never_a_silent_pass(rows, tmp_path):
    r, d = live(M1, rows, good_stats())
    assert d["decision_rule"]["applicability"].startswith(LG.BANDED_NA) and d["decision_rule"]["decision_rule"] == "prior_corrected"
    assert d["spelling"]["map_hires_8code_spelling"].startswith(LG.BANDED_NA)
    # the Watch: still a launch artifact (a missing one is a NAMED FAIL), its 8-way series / alarm tile are recorded as not applicable
    r, wd = LG.banded_watch_reasons(RIG, None)
    assert len(r) == 1 and "no refcv7 Training Watch artifact" in r[0]
    r, wd = LG.banded_watch_reasons(RIG, str(tmp_path / "nope.html"))
    assert len(r) == 1 and "does not exist" in r[0]
    w = tmp_path / "watch.html"
    w.write_text("<html>watch</html>", encoding="utf-8")
    r, wd = LG.banded_watch_reasons(RIG, str(w))
    assert r == [] and wd["per_class_series_and_alarm"].startswith(LG.BANDED_NA) and wd["sha256"] == LG.sha256_file(w)
    assert "renders the `map_banded_*` series" in wd["banded_series"] and "not gated" in wd["banded_series"]


# ======================================================================================== R: the RED arms (by name) ==
def test_R_a_dropped_training_loss_key_fails_with_its_own_reason(rows):
    del rows[0]["map_banded_loss_lane_ox_far"]
    r, d = LG.banded_train_row_reasons(RIG, trains(rows))
    assert len(r) == 1 and "1 of 2 training rows lack the banded per-channel x band telemetry" in r[0] and "step 50, 1 of 39 keys missing" in r[0], r
    r2, _ = live(M1, rows, good_stats())
    assert any("lack the banded per-channel x band telemetry" in x for x in r2)


def test_R_a_dropped_max_entropy_key_is_also_caught_it_is_part_of_the_row(rows):
    del rows[1]["map_banded_maxent_far"]
    r, _ = LG.banded_train_row_reasons(RIG, trains(rows))
    assert len(r) == 1 and "step 100, 1 of 39 keys missing" in r[0]


@pytest.mark.parametrize("key", ["eval_map_banded_pos_hatch_far", "eval_map_banded_inter_drv_near", "eval_map_banded_union_xwalk_mid",
                                 "eval_map_banded_n_mid", "eval_map_banded_loss_drv_sdf_near"])
def test_R_a_dropped_eval_count_key_fails_with_its_own_reason(rows, key):
    del rows[2][key]
    r, d = LG.banded_eval_key_reasons(RIG, evalrow(rows))
    assert len(r) == 1 and "lacks 1 of the 102 `eval_map_banded_*` telemetry keys" in r[0] and key in r[0], r
    assert d["eval_"]["missing"] == [key]


def test_R_a_dropped_ema_key_is_caught_when_the_run_keeps_an_ema_and_only_then(rows):
    del rows[2]["ema_eval_map_banded_inter_lane_mid"]
    r, _ = LG.banded_eval_key_reasons(RIG, evalrow(rows), ema=True)
    assert len(r) == 1 and "`ema_eval_map_banded_*`" in r[0] and "ema_eval_map_banded_inter_lane_mid" in r[0]
    assert LG.banded_eval_key_reasons(RIG, evalrow(rows), ema=False)[0] == []
    assert LG.ema_on(M1) and not LG.ema_on(LG.set_flag(M1, "--ema-decay", ["0"])) and not LG.ema_on(LG.set_flag(M1, "--ema-decay", None))
    rows[2]["ema_eval_error"] = "RuntimeError: boom"
    r, _ = LG.banded_eval_key_reasons(RIG, evalrow(rows), ema=True)
    assert any("EMA in-run eval failed" in x for x in r)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "x"])
def test_R_a_nonfinite_or_non_numeric_eval_value_fails(rows, bad):
    rows[2]["eval_map_banded_union_drv_mid"] = bad
    r, _ = LG.banded_eval_key_reasons(RIG, evalrow(rows))
    assert len(r) == 1 and "1 of the 102 `eval_map_banded_*` keys are non-finite or non-numeric" in r[0] and "eval_map_banded_union_drv_mid" in r[0], r


def test_R_a_nan_banded_loss_in_a_training_row_fails(rows):
    rows[1]["map_banded_loss_edge_h_mid"] = float("nan")
    r, _ = LG.banded_train_row_reasons(RIG, trains(rows))
    assert len(r) == 1 and "carry a non-finite banded key or loss" in r[0] and "map_banded_loss_edge_h_mid" in r[0]


def test_R_an_unlabelled_band_fails(rows):
    rows[2]["eval_map_banded_n_far"] = 0.0
    r, d = LG.banded_eval_key_reasons(RIG, evalrow(rows))
    assert len(r) == 1 and "no labelled cells in band(s) ['far']" in r[0] and d["bands_without_cells"] == ["far"]


def test_R_a_broken_loss_sum_fails(rows):
    rows[0]["map_hires"] = rows[0]["map_hires"] * 1.01
    r, _ = LG.banded_train_row_reasons(RIG, trains(rows))
    assert len(r) == 1 and "do not sum to `map_hires`" in r[0] and "step 50" in r[0]
    rows[0]["map_hires"] = rows[0]["map_hires"] / 1.01
    rows[0]["map_banded_loss_drv_near"] += 0.5                  # one term moved: the sum no longer reproduces the loss
    r, _ = LG.banded_train_row_reasons(RIG, trains(rows))
    assert len(r) == 1 and "do not sum" in r[0]


def test_R_the_all_ignore_batch_is_the_trainers_counted_zero_and_only_that_is_exempt(rows):
    z = {k: v for k, v in rows[0].items() if "map_banded" not in k}
    z.update(map_hires=0.0, map_hires_n_labelled=0.0, step=150)
    assert LG.banded_train_row_reasons(RIG, trains(rows) + [z])[0] == []
    assert LG.banded_train_row_reasons(RIG, trains(rows) + [dict(z, map_hires=0.3)])[0][0].startswith("G-MAP: 1 all-ignore training row(s)")
    nl = dict(z, map_hires_n_labelled=5.0)                      # labelled windows but no banded keys: NOT exempt
    assert "lack the banded per-channel" in LG.banded_train_row_reasons(RIG, trains(rows) + [nl])[0][0]
    assert "never reached the log" in LG.banded_train_row_reasons(RIG, [{"step": 50, "loss": 1.0}])[0][0]
    assert "wrote no in-run eval row" in LG.banded_eval_key_reasons(RIG, [])[0][0]


def test_R_the_gradient_probe_reasons_each_their_own(rows):
    tr = trains(rows)
    assert LG.judge_banded_classes(RIG, good_stats(), tr, extent=EXT)[0] == []
    assert "probe never ran" in LG.judge_banded_classes(RIG, {}, tr, extent=EXT)[0][0]                 # nothing hooked
    assert "probe never ran" in LG.judge_banded_classes(RIG, None, tr, extent=EXT)[0][0]
    r, _ = LG.judge_banded_classes(RIG, dict(good_stats(), probe_error="ImportError: x"), tr, extent=EXT)
    assert len(r) == 1 and "probe raised" in r[0]
    r, _ = LG.judge_banded_classes(RIG, good_stats(bands=("near", "mid")), tr, extent=EXT)               # only two bands carried a gradient
    assert len(r) == 1 and "band(s) ['far'] never carried a gradient" in r[0]
    st = good_stats()
    st["bands"]["mid"]["nonfinite"]["lane"] = 3
    r, _ = LG.judge_banded_classes(RIG, st, tr, extent=EXT)
    assert len(r) == 1 and "non-finite gradient" in r[0] and "lane_mid" in r[0]
    st = good_stats()
    st["bands"]["near"]["grad_obs"]["lane_oy"] = 0.0                    # a channel with a supervised loss in the smoke and exactly zero gradient
    r, _ = LG.judge_banded_classes(RIG, st, tr, extent=EXT)
    assert len(r) == 1 and "ZERO gradient at their observed cells" in r[0] and "lane_oy_near" in r[0], r
    st = good_stats()
    st["bands"]["far"]["grad_obs"] = {c: 0.0 for c in CHANNELS}         # a whole band dead
    r, _ = LG.judge_banded_classes(RIG, st, tr, extent=EXT)
    assert any("band(s) ['far']: ZERO gradient" in x for x in r)
    r, _ = LG.judge_banded_classes(RIG, good_stats(steps=0), tr, extent=EXT)                              # no raster / labelled window in any step
    assert len(r) == 1 and "carried the 10 cm raster `map_fine`" in r[0]
    r, _ = LG.judge_banded_classes(RIG, good_stats(raster=(600, 320)), tr, extent=EXT)                    # a raster of the wrong extent
    assert len(r) == 1 and "need [1000, 600]" in r[0]
    # a channel with NO supervised loss in the smoke rows is not required to have a gradient (a zero loss is exactly zero, the data's absence)
    for t in tr:
        t["map_banded_loss_lane_oy_near"] = 0.0
    assert LG.judge_banded_classes(RIG, st_zero_lane_oy(), tr, extent=EXT)[0] == []


def st_zero_lane_oy():
    st = good_stats()
    st["bands"]["near"]["grad_obs"]["lane_oy"] = 0.0
    return st


def test_R_the_spelling_pin_goes_red_when_a_format_drifts(monkeypatch):
    monkeypatch.setitem(LG.BANDED_KEY_FMT, "loss", "map_banded_lossx_{ch}_{band}")
    r, _ = LG.banded_spelling_reasons(RIG)
    assert r and all("differ from map_banded.telemetry_keys" in x for x in r)


# ---- the OUTPUT probe on a REAL CPU-built banded model (the tiny rig of test_map_banded_wiring) -----------------
@pytest.fixture(scope="module")
def tiny():
    import refc_v3_train as T
    import test_map_banded_wiring as W
    import test_map_only_ft as R
    model, cfg = W._build(banded=True)
    torch.manual_seed(0)
    batch = R._batch(cfg)
    model.train()
    cap, stats = {}, {}

    def hook(_m, _i, out):
        cap["out"] = out
        LG.observe_banded_forward(stats, out, batch)
    h = model.register_forward_hook(hook)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    losses["loss"].backward()
    h.remove()
    row = {"step": 50, "loss": float(losses["loss"].detach()), "map_hires": float(losses["map_hires"].detach()),
           "map_hires_n_labelled": float(losses["map_hires_n_labelled"])}
    row.update({k: float(torch.as_tensor(v).detach()) for k, v in losses.items() if k.startswith("map_banded_")})
    p2 = dict(RIG, map_hires=dict(RIG["map_hires"], logits_hw=(600, 320), lift_hw=(240, 128)))        # the tiny rig's 60 m x +-16 m extent
    return {"T": T, "model": model, "cfg": cfg, "batch": batch, "out": cap["out"], "stats": stats, "row": row, "prof": p2, "W": W, "R": R}


TINY_EXT = (60.0, 16.0)


def test_P_a_real_banded_forward_passes_the_output_probe(tiny):
    r, d = LG.banded_output_reasons(tiny["out"], *TINY_EXT)
    assert r == [] and d["map_hires_logits"] == "None" and d["logits"] == [[2, 10, 100, 320], [2, 10, 120, 128], [2, 10, 40, 64]]
    assert LG.hires_output_reasons(tiny["prof"], tiny["out"], (64, 64), banded=TINY_EXT)[0] == []       # fmap_s8 + lift still checked


def test_M_the_same_real_forward_without_the_banded_branch_fails_with_thors_reason(tiny):
    r = LG.hires_output_reasons(tiny["prof"], tiny["out"], (64, 64))[0]
    assert r == ["no forward output named with 'hires' and 'logit' -- the 10 cm logits were not produced"]


def test_R_only_two_bands_of_logits_fails(tiny):
    out = copy.copy(tiny["out"])
    out["perception"] = dict(out["perception"], map_banded=dict(out["perception"]["map_banded"],
                                                                  logits=out["perception"]["map_banded"]["logits"][:2]))
    r, _ = LG.banded_output_reasons(out, *TINY_EXT)
    assert len(r) == 1 and "`map_banded.logits` holds 2 tensor(s), the banded head has 3 bands" in r[0], r


def test_R_a_wrong_band_shape_fails(tiny):
    r, _ = LG.banded_output_reasons(tiny["out"], 100.0, 30.0)                  # the declared extent is not the built one
    assert len(r) == 6 and sum("are not [B, 10," in x for x in r) == 3 and sum("`unobs` is not a bool" in x for x in r) == 3
    mb = tiny["out"]["perception"]["map_banded"]
    cut = dict(mb, logits=[mb["logits"][0][:, :, :, :-1], mb["logits"][1], mb["logits"][2]])
    out = dict(tiny["out"], perception=dict(tiny["out"]["perception"], map_banded=cut))
    r, _ = LG.banded_output_reasons(out, *TINY_EXT)
    assert len(r) == 1 and r[0].startswith("band near: logits (2, 10, 100, 319) are not [B, 10, 100, 320]")
    five = dict(mb, logits=[z[:, :5] for z in mb["logits"]])                     # 5 channels instead of 10
    r, _ = LG.banded_output_reasons(dict(tiny["out"], perception=dict(tiny["out"]["perception"], map_banded=five)), *TINY_EXT)
    assert len(r) == 3 and all("are not [B, 10," in x for x in r)


def test_R_map_hires_logits_not_none_fails(tiny):
    out = dict(tiny["out"], perception=dict(tiny["out"]["perception"], map_hires_logits=torch.zeros(2, 8, 600, 320)))
    r, _ = LG.banded_output_reasons(out, *TINY_EXT)
    assert len(r) == 1 and "`perception.map_hires_logits` is not None under `--r9-map-banded`" in r[0]


def test_R_nonfinite_logits_missing_unobs_missing_head_and_missing_perception_each_fail(tiny):
    mb = tiny["out"]["perception"]["map_banded"]
    nan = mb["logits"][1].detach().clone()
    nan[0, 0, 0, 0] = float("nan")
    out = dict(tiny["out"], perception=dict(tiny["out"]["perception"], map_banded=dict(mb, logits=[mb["logits"][0], nan, mb["logits"][2]])))
    r, _ = LG.banded_output_reasons(out, *TINY_EXT)
    assert r == ["band mid: the logits carry NaN / Inf"]
    nu = dict(tiny["out"], perception=dict(tiny["out"]["perception"], map_banded={"logits": mb["logits"]}))
    r, _ = LG.banded_output_reasons(nu, *TINY_EXT)
    assert len(r) == 1 and "`perception.map_banded.unobs` is absent" in r[0]
    bad_u = dict(mb, unobs=[mb["unobs"][0].float(), mb["unobs"][1], mb["unobs"][2]])
    r, _ = LG.banded_output_reasons(dict(tiny["out"], perception=dict(tiny["out"]["perception"], map_banded=bad_u)), *TINY_EXT)
    assert len(r) == 1 and "band near: `unobs` is not a bool" in r[0]
    nh = dict(tiny["out"], perception={k: v for k, v in tiny["out"]["perception"].items() if k != "map_banded"})
    r, _ = LG.banded_output_reasons(nh, *TINY_EXT)
    assert len(r) == 1 and "`perception.map_banded.logits` is ABSENT" in r[0]
    r, _ = LG.banded_output_reasons({"fmap_s8": torch.zeros(1)}, *TINY_EXT)
    assert len(r) == 1 and "`perception` is ABSENT" in r[0]


# ---- the GRADIENT probe on the real model, with its control ------------------------------------------------------
def test_P_the_real_models_gradient_probe_reads_nonzero_everywhere_and_judges_clean(tiny):
    st = tiny["stats"]
    assert set(st["bands"]) == set(BANDS) and st["targets"] == {"steps": 1, "labelled_windows": 2, "raster_shape": [600, 320]}
    for b in BANDS:
        e = st["bands"][b]
        assert e["hooked"] == 1 and all(e["grad_obs"][c] > 0.0 for c in CHANNELS) and all(e["nonfinite"][c] == 0 for c in CHANNELS), b
    r, d = LG.judge_banded_classes(tiny["prof"], st, LG.train_rows([tiny["row"]]), extent=TINY_EXT)
    assert r == [] and d["n_channel_band_terms_with_supervision"] == 30
    assert LG.banded_train_row_reasons(tiny["prof"], LG.train_rows([tiny["row"]]))[0] == []              # the real trainer's own row obeys the sum identity


def test_R_CONTROL_with_every_cell_unobserved_the_supervised_gradient_reads_exactly_zero_and_the_probe_goes_red(tiny, monkeypatch):
    """The control that proves the probe works at all: with every cell unobserved the supervised terms are masked out, so the OBSERVED-cell
    gradient is EXACTLY 0.0 even though the max-entropy term still sends gradient (to the unobserved cells) -- a dead supervised term cannot hide
    behind it."""
    T, model, batch = tiny["T"], tiny["model"], tiny["batch"]
    real = MB.unobs_masks
    monkeypatch.setattr(MB, "unobs_masks", lambda v, g: [torch.ones_like(m) for m in real(v, g)])
    stats = {}
    h = model.register_forward_hook(lambda _m, _i, out: LG.observe_banded_forward(stats, out, batch))
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    T.compute_losses_v3(model, batch, "cpu", mode="diffusion")["loss"].backward()
    h.remove()
    assert all(stats["bands"][b]["grad_obs"][c] == 0.0 for b in BANDS for c in CHANNELS)
    assert any(stats["bands"][b]["grad_unobs"][c] > 0.0 for b in BANDS for c in PROB)                       # maxent still reaches the unobserved cells
    r, _ = LG.judge_banded_classes(tiny["prof"], stats, LG.train_rows([tiny["row"]]), extent=TINY_EXT)     # the healthy run's rows: every term has a supervised loss
    assert any("ZERO gradient at the observed cells" in x for x in r) and any("ZERO gradient at their observed cells" in x for x in r), r


def test_R_a_banded_term_that_never_reaches_the_loss_is_caught_by_the_probe(tiny, monkeypatch):
    """The unreachable-from-its-caller class (CLAUDE.md 2026-09-10/11): the trainer's banded term replaced by a zero -> every channel reads 0.0."""
    T, model, batch = tiny["T"], tiny["model"], tiny["batch"]
    monkeypatch.setattr(T, "_map_banded_loss", lambda model_, out, batch_, device, extra, with_metrics=True:
                        sum((z.sum() * 0.0 for z in out["perception"]["map_banded"]["logits"]), torch.zeros(())))
    stats = {}
    h = model.register_forward_hook(lambda _m, _i, out: LG.observe_banded_forward(stats, out, batch))
    model.zero_grad(set_to_none=True)
    T.compute_losses_v3(model, batch, "cpu", mode="diffusion")["loss"].backward()
    h.remove()
    assert all(stats["bands"][b]["grad_obs"][c] == 0.0 for b in BANDS for c in CHANNELS)
    r, _ = LG.judge_banded_classes(tiny["prof"], stats, LG.train_rows([tiny["row"]]), extent=TINY_EXT)
    assert any("ZERO gradient at their observed cells" in x for x in r)


def test_R_no_labelled_window_in_the_batch_means_no_targets(tiny):
    st = {}
    batch = dict(tiny["batch"], map_fine_label=torch.tensor([False, False]))
    LG.observe_banded_forward(st, tiny["out"], batch)
    assert st["targets"]["steps"] == 0
    r, _ = LG.judge_banded_classes(tiny["prof"], dict(good_stats(raster=(600, 320)), targets=st["targets"]), [], extent=TINY_EXT)
    assert len(r) == 1 and "carried the 10 cm raster `map_fine`" in r[0]


def test_P_the_static_eval_row_builder_check_on_the_real_trainer_and_model(tiny):
    T, model = tiny["T"], tiny["model"]
    r, d = LG.banded_eval_list_reasons(T, model=model)
    assert r == [] and d["n_expected"] == 102 and d["n_missing"] == 0
    assert "no eval-row builder" in LG.banded_eval_list_reasons(object(), model=model)[0][0]
    real = T._eval_row_from_acc
    T_drop = type("TT", (), {"_eval_row_from_acc": staticmethod(lambda acc, nb, m: {k: v for k, v in real(acc, nb, m).items()
                                                                                       if k != "eval_map_banded_pos_arrow_far"})})
    r, _ = LG.banded_eval_list_reasons(T_drop, model=model)
    assert len(r) == 1 and "omits 1 of the 102" in r[0] and "eval_map_banded_pos_arrow_far" in r[0]


# ============================================================ M: the MUTATION -- delete the banded branch ==
THOR = ("G-MAP: the per-class probe never ran -- no 10 cm logits tensor with a gradient and a 10 cm target were found in a training step",
        "G-MAP: 160 of the 160 per-class argmax COUNT keys are absent or non-finite in the eval row (first: ['eval_map_hires_inter_nocls_0_20', ",
        "G-MAP: the smoke's eval windows carry no labelled 0_20 m cells for ['nocls', 'drivable', 'lane', 'crosswalk', 'arrow', 'edge', 'hatched', "
        "'sidewalk'] -- extend the eval set; the check is not waived",
        "G-MAP: 2 of 2 training rows lack the per-class loss contributions `map_hires_lc_{cls}_{band}` (first: step 50, 40 keys missing)")


def test_M_deleting_the_banded_branch_makes_the_m1_rows_fail_again_with_thors_reasons(rows, monkeypatch):
    assert live(M1, rows, good_stats())[0] == []                                  # with the branch: clean
    monkeypatch.setattr(LG, "banded_on", lambda prof, argv_: False)                  # the mutation: the banded dispatch is gone
    r, _ = live(M1, rows, good_stats())
    assert len(r) == len(THOR), r
    for got, want in zip(r, THOR):
        assert got.startswith(want), (got, want)
    assert LG.banded_arg(RIG, M1) is None                                           # ... and the output probe falls back to the 8-way check too


def test_M_the_same_mutation_on_a_non_banded_flag_removal(rows):
    """The converse: the SAME M1 rows judged under an argv without the flag take the 8-code path -- Thor's four reasons."""
    no_flag = LG.set_flag(M1, LG.BANDED_FLAG, None)
    r, _ = live(no_flag, rows, good_stats())
    assert len(r) == 4 and all(g.startswith(w) for g, w in zip(r, THOR))


# ============================================================ N: a NON-banded argv is judged exactly as before ==
@pytest.mark.parametrize("arm", ["C0", "C1", "B1", "B2", "T1"])
def test_N_a_non_banded_argv_takes_the_8code_functions_unchanged(rows, arm):
    a = argv(arm)
    run1 = {"map_class": {}, "map_banded": good_stats()}
    got_r, got_d = LG.map_live_items(RIG, a, run1, rows, CFG, weighted=True)
    ev = evalrow(rows)
    want = [LG.judge_map_classes(RIG, {}), LG.map_eval_key_reasons(RIG, ev, weighted=True), LG.map_train_row_reasons(RIG, trains(rows)),
            LG.map_decision_rule_reasons(RIG, CFG), LG.map_spelling_reasons(RIG)]
    assert got_r == [x for rs, _ in want for x in rs]
    assert [got_d[k] for k in ("classes", "eval_keys", "train_rows", "decision_rule", "spelling")] == [d for _, d in want]
    assert LG.hires_output_reasons(RIG, {"fmap_s8": torch.zeros(1, 1, 52, 128)}, (416, 1024))[0] == \
        LG.hires_output_reasons(RIG, {"fmap_s8": torch.zeros(1, 1, 52, 128)}, (416, 1024), banded=None)[0]


def test_N_the_8code_judges_still_refuse_what_they_always_refused(rows):
    # a real 8-code regression on the unchanged path still fires: a row with an 8-code loss key but no per-class contributions
    r, _ = LG.map_train_row_reasons(RIG, [{"step": 50, "map_hires": 2.0, "loss": 3.0}])
    assert len(r) == 1 and "lack the per-class loss contributions" in r[0]
    assert "never ran" in LG.judge_map_classes(RIG, {"_hooked": 0})[0][0]


# ===================================================================== W: the call sites are in place (reachability) ==
SRC = Path(LG.__file__).read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_W_every_hop_from_the_gate_to_the_banded_checks_is_in_place_and_each_removal_is_seen():
    smoke, model_, live_ = inspect.getsource(LG.run_smoke), inspect.getsource(LG.job_model), inspect.getsource(LG.job_smoke)
    hops = [
        (smoke, "hires_output_reasons(ctx.prof, out, hw, banded=banded_arg(ctx.prof, ctx.argv))", 1),
        (smoke, "observe_banded_forward(rec[\"map_banded\"], out, cur_batch.get(\"b\"))", 1),
        (smoke, "\"map_banded\": {}", 1),
        (model_, "banded=banded_arg(ctx.prof, ctx.argv))", 2),                          # G-EVAL and G-DVB
        (model_, "lr, ld = banded_eval_list_reasons(T, model=model)", 1),
        (live_, "map_live_items(ctx.prof, ctx.argv, run1, rows1, cfg1, weighted=weighted)", 1),
        (live_, "banded_watch_reasons if banded_on(ctx.prof, ctx.argv) else map_watch_reasons", 1),
        (live_, "reasons += items_r + wr + csr", 1),
    ]
    for src, needle, n in hops:
        assert src.count(needle) == n, needle
        assert src.replace(needle, "", 1).count(needle) == n - 1                         # delete ONE hop: the scan sees it
    # the banded probe is fed BEFORE the 8-way one would be (the else branch), and run1 carries what map_live_items reads
    assert smoke.index("observe_banded_forward(rec") < smoke.index("_hook_map_classes(out)", smoke.index("observe_banded_forward(rec"))
    assert "\"map_banded\"" in live_ or "run1.get(\"map_banded\")" in inspect.getsource(LG.map_live_items)


def test_W_the_8code_functions_keep_their_signatures_and_the_new_kwarg_defaults_to_the_old_path():
    assert list(inspect.signature(LG.hires_output_reasons).parameters) == ["prof", "out", "image_hw", "banded"]
    assert inspect.signature(LG.hires_output_reasons).parameters["banded"].default is None
    for fn, params in ((LG.judge_map_classes, ["prof", "stats"]), (LG.map_eval_key_reasons, ["prof", "eval_rows", "weighted"]),
                       (LG.map_train_row_reasons, ["prof", "train_rows"]), (LG.map_watch_reasons, ["prof", "artifact"]),
                       (LG.map_decision_rule_reasons, ["prof", "config"]), (LG.map_eval_list_reasons, ["prof", "T", "model"]),
                       (LG.map_spelling_reasons, ["prof"])):
        assert list(inspect.signature(fn).parameters) == params, fn.__name__
