"""Tests for hicap_backbone_latency_bench.py (H-HC11 harness).  No torch, no GPU, no network needed.

Part 1 tests the pure logic and the mock backend (in-process and via subprocess).
Part 2 drives the REAL `HFBackend` code path against FAKE torch / transformers / PIL modules injected into
sys.modules.  That catches Python-level bugs (wrong kwargs, missing attributes, stage arithmetic, hook placement,
token guards, dtype guard, taps, thread mechanics) in code that otherwise only runs on a Jetson.  It does NOT
validate any real torch / transformers behaviour: the fakes encode our READING of those APIs (transformers 5.17
source), so a green Part 2 is not evidence that the hf backend works on hardware.  The fast-path proxy network
(`_build_proxy`, plain nn code) is replaced by a stub in Part 2 and is therefore UNTESTED here.

Run:  python -m pytest -q test_hicap_backbone_latency_bench.py
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "hicap_backbone_latency_bench.py"
sys.path.insert(0, str(HERE))
import hicap_backbone_latency_bench as hb  # noqa: E402


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def run_cli(*args, timeout=180):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout)


def run_mock(tmp_path, *extra, name="out.json"):
    out = tmp_path / name
    r = run_cli("--backend", "mock", "--model", "qwen3-vl-4b", "--out", str(out), *extra)
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text()), r


def measured(**over):
    """A healthy MEASURED-on-Thor input for evaluate_verdict; override single fields."""
    m = dict(n_ticks=1000, refresh_p95_ms=40.0, refresh_mean_ms=30.0, refresh_hz=2.0, token_age_p95_ms=480.0,
             fast_concurrent_n=1500, fast_concurrent_p95_ms=12.0, fast_overlapped_p95_ms=15.0, fast_overlapped_n=300,
             refresh_concurrent_p95_ms=45.0, refresh_concurrent_n=1000, preprocess_allowance_ms=8.0, tick_budget_ms=100.0,
             on_thor=True)
    m.update(over)
    return m


# ---------------------------------------------------------------------------------------------
# percentile / summary math
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arr", [[3.0], [1.0, 2.0], [5, 1, 4, 2, 3], list(range(1, 101)), [0.5, 0.5, 0.5, 9.0]])
@pytest.mark.parametrize("q", [0, 1, 25, 50, 90, 95, 99, 99.9, 100])
def test_percentile_matches_numpy_linear(arr, q):
    assert hb.percentile(arr, q) == pytest.approx(float(np.percentile(np.asarray(arr, dtype=float), q)), rel=1e-12, abs=1e-12)


def test_percentile_random_arrays_match_numpy():
    rng = np.random.default_rng(7)
    for n in (2, 7, 50, 1000):
        x = rng.lognormal(3.0, 0.5, n)
        for q in (50, 95, 99):
            assert hb.percentile(x.tolist(), q) == pytest.approx(float(np.percentile(x, q)), rel=1e-12)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        hb.percentile([], 50)


def test_summarize_known_values_and_empty():
    s = hb.summarize(list(range(1, 101)))
    assert s["n"] == 100 and s["min"] == 1 and s["max"] == 100
    assert s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05) and s["p99"] == pytest.approx(99.01)
    assert s["mean"] == pytest.approx(50.5) and s["std"] == pytest.approx(float(np.std(np.arange(1, 101))))
    e = hb.summarize([])
    assert e["n"] == 0 and all(e[k] is None for k in ("mean", "std", "min", "p50", "p95", "p99", "max"))
    json.dumps(e)


# ---------------------------------------------------------------------------------------------
# token-count arithmetic
# ---------------------------------------------------------------------------------------------
def test_tokens_per_frame_256x640_is_160():
    assert hb.tokens_per_image(256, 640) == 160          # (256/32) * (640/32) = 8 * 20


@pytest.mark.parametrize("cams,frames,visual,llm", [
    (1, 1, 160, 224), (3, 1, 480, 544), (7, 1, 1120, 1184),      # sec_eff.tex Table tab:thor n = 224 / 544 / 1184
    (1, 3, 480, 544), (3, 3, 1440, 1504), (7, 3, 3360, 3424),    # RA_vlm_backbones.md 5.2: 3 cam x 3 frames = 1,504
])
def test_token_budget_1_3_7_cameras(cams, frames, visual, llm):
    tb = hb.token_budget(cams, frames, 256, 640, prompt_tokens=64)
    assert tb["images_per_tick"] == cams * frames
    assert tb["visual_tokens"] == visual and tb["llm_input_tokens"] == llm


def test_cosmos_reason2_min_pixels_pitfall_is_286_not_160():
    # cosmos-reason2 scripts/inference_sample.py sets shortest_edge = 256 tokens * 32^2 px: it would UPSCALE a 256x640 frame.
    assert hb.tokens_per_image(256, 640, min_tokens=256) == 286


def test_pooled_counts_and_taps():
    assert hb.pooled_token_counts(1504, 128, 3) == {"pooled_tokens_per_tap": 128, "n_taps": 3, "pooled_tokens_total": 384}
    assert hb.pooled_token_counts(100, 128, 3)["pooled_tokens_per_tap"] == 100      # "<= pool-tokens"
    assert hb.resolve_taps("auto", 36) == [12, 24, 36]
    assert hb.resolve_taps("auto", 28) == [9, 19, 28]
    assert hb.resolve_taps("3,10,20", 36) == [3, 10, 20]
    with pytest.raises(ValueError):
        hb.resolve_taps("3,99", 36)


# ---------------------------------------------------------------------------------------------
# scheduler, token age, overlap
# ---------------------------------------------------------------------------------------------
def _const_refresh(ms):
    return lambda slot: {"total_ms": ms, "vision_ms": 0.0, "llm_prefill_ms": ms, "pool_ms": 0.0, "residual_ms": 0.0}


def test_schedule_steady_state_has_no_late_or_dropped():
    recs, s = hb.run_refresh_schedule(hb.VirtualClock(), _const_refresh(100.0), 50, 500.0)
    assert s["n_executed"] == 50 and s["n_dropped"] == 0 and s["n_late"] == 0 and s["n_overrun"] == 0
    assert [r["slot"] for r in recs] == list(range(50))
    assert all(recs[i + 1]["scheduled_ms"] - recs[i]["scheduled_ms"] == pytest.approx(500.0) for i in range(49))


def test_schedule_overrun_makes_ticks_late_then_drops_slots():
    # 730 ms is not a multiple of the 500 ms period, so no refresh ends exactly on a slot boundary
    recs, s = hb.run_refresh_schedule(hb.VirtualClock(), _const_refresh(730.0), 20, 500.0)
    assert s["n_overrun"] == 20
    assert s["n_late"] == 19 and recs[0]["late"] is False           # first slot starts on time, every later one is behind
    assert s["n_dropped"] > 0 and s["span_slots"] == 20 + s["n_dropped"]
    assert s["max_lateness_ms"] < 500.0                               # latest-wins: lateness is bounded by one period
    recs2, s2 = hb.run_refresh_schedule(hb.VirtualClock(), _const_refresh(1200.0), 10, 500.0)
    assert s2["n_dropped"] >= 10                                     # > 2 periods per refresh: slots are skipped every time


def test_schedule_unpaced_reports_none_for_late_and_dropped():
    _, s = hb.run_refresh_schedule(hb.VirtualClock(), _const_refresh(100.0), 10, 500.0, pace=False)
    assert s["paced"] is False and s["n_dropped"] is None and s["n_late"] is None


def test_token_age_hand_built_schedule():
    recs = [{"scheduled_ms": 0.0, "start_ms": 0.0, "end_ms": 200.0}, {"scheduled_ms": 500.0, "start_ms": 500.0, "end_ms": 700.0}]
    assert hb.token_ages(recs, 100.0) == pytest.approx([200, 300, 400, 500, 600, 200])


def test_token_age_grows_when_refresh_is_late():
    on_time = [{"scheduled_ms": 500.0 * i, "end_ms": 500.0 * i + 100} for i in range(20)]
    late = [{"scheduled_ms": 500.0 * i, "end_ms": 500.0 * i + 100 + 300} for i in range(20)]
    assert max(hb.token_ages(late, 100.0)) > max(hb.token_ages(on_time, 100.0))
    assert hb.percentile(hb.token_ages(on_time, 100.0), 95) <= 500 + 100      # healthy: age <= period + refresh time
    assert hb.token_ages([], 100.0) == []


def test_classify_overlap():
    busy = [(100.0, 200.0), (500.0, 600.0)]
    ticks = [{"start_ms": s, "end_ms": s + 5} for s in (50.0, 98.0, 150.0, 199.0, 205.0, 300.0, 590.0, 610.0)]
    over, non = hb.classify_overlap(ticks, busy)
    assert [t["start_ms"] for t in over] == [98.0, 150.0, 199.0, 590.0]
    assert [t["start_ms"] for t in non] == [50.0, 205.0, 300.0, 610.0]


# ---------------------------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------------------------
def test_verdict_all_pass():
    v = hb.evaluate_verdict(measured(), "MEASURED")
    assert all(r["status"] == "PASS" for r in v["rows"].values()), {k: r["status"] for k, r in v["rows"].items()}
    assert v["readings"]["synchronous"]["overall"] == "PASS" and v["readings"]["asynchronous"]["overall"] == "PASS"
    assert all(rd["decides_h_hc11"] for rd in v["readings"].values()) and "overall" not in v
    assert v["rows"]["duty_cycle_le_0.5"]["value"] == pytest.approx(0.06)
    assert v["drift"]["status"] == "UNMEASURED"


def test_verdict_boundaries_are_inclusive():
    v = hb.evaluate_verdict(measured(refresh_p95_ms=50.0, fast_concurrent_p95_ms=25.0, token_age_p95_ms=550.0,
                                     refresh_mean_ms=250.0), "MEASURED")
    assert v["rows"]["refresh_sync_p95_le_50ms"]["status"] == "PASS"
    assert v["rows"]["fastpath_concurrent_p95_le_25ms"]["status"] == "PASS"
    assert v["rows"]["token_age_p95_le_500ms_plus_refresh_p95"]["status"] == "PASS"     # 550 <= 500 + 50
    assert v["rows"]["duty_cycle_le_0.5"]["status"] == "PASS"                            # 0.25 s * 2 Hz = 0.5


def test_verdict_synchronous_fail_asynchronous_pass():
    v = hb.evaluate_verdict(measured(refresh_p95_ms=60.0, token_age_p95_ms=500.0), "MEASURED")
    assert v["rows"]["refresh_sync_p95_le_50ms"]["status"] == "FAIL"
    assert v["rows"]["refresh_async_p95_le_250ms"]["status"] == "PASS"
    assert v["readings"]["synchronous"]["overall"] == "FAIL"
    assert v["readings"]["asynchronous"]["overall"] == "PASS"
    assert v["readings"]["synchronous"]["decides_h_hc11"] is True      # a FAIL on a fully measured reading decides too


def test_verdict_asynchronous_reading_rows():
    # token age bound is 500 ms + THIS run's refresh p95
    ok = hb.evaluate_verdict(measured(refresh_p95_ms=100.0, token_age_p95_ms=600.0), "MEASURED")
    bad = hb.evaluate_verdict(measured(refresh_p95_ms=100.0, token_age_p95_ms=600.5), "MEASURED")
    assert ok["rows"]["token_age_p95_le_500ms_plus_refresh_p95"]["status"] == "PASS"
    assert ok["rows"]["token_age_p95_le_500ms_plus_refresh_p95"]["threshold"] == pytest.approx(600.0)
    assert bad["rows"]["token_age_p95_le_500ms_plus_refresh_p95"]["status"] == "FAIL"
    assert bad["readings"]["asynchronous"]["overall"] == "FAIL"
    slow = hb.evaluate_verdict(measured(refresh_p95_ms=240.0, refresh_mean_ms=300.0), "MEASURED")   # duty 0.6
    assert slow["rows"]["duty_cycle_le_0.5"]["status"] == "FAIL"
    assert slow["rows"]["refresh_async_p95_le_250ms"]["status"] == "PASS"
    assert slow["readings"]["asynchronous"]["overall"] == "FAIL"
    over = hb.evaluate_verdict(measured(refresh_p95_ms=300.0), "MEASURED")
    assert over["rows"]["refresh_async_p95_le_250ms"]["status"] == "FAIL"


def test_verdict_unmeasured_when_values_missing():
    v = hb.evaluate_verdict(measured(fast_concurrent_n=None, fast_concurrent_p95_ms=None, fast_overlapped_p95_ms=None,
                                     token_age_p95_ms=None), "MEASURED")
    assert v["rows"]["fastpath_concurrent_p95_le_25ms"]["status"] == "UNMEASURED"
    assert v["rows"]["e2e_command_latency_proxy_le_tick_budget"]["status"] == "UNMEASURED"
    assert v["rows"]["token_age_p95_le_500ms_plus_refresh_p95"]["status"] == "UNMEASURED"
    assert v["readings"]["synchronous"]["overall"] == "UNMEASURED" and v["readings"]["asynchronous"]["overall"] == "UNMEASURED"
    assert not any(rd["decides_h_hc11"] for rd in v["readings"].values())   # UNMEASURED gating rows -> cannot decide


def test_verdict_fail_dominates_unmeasured():
    v = hb.evaluate_verdict(measured(refresh_p95_ms=80.0, fast_concurrent_p95_ms=None, fast_concurrent_n=None), "MEASURED")
    assert v["readings"]["synchronous"]["overall"] == "FAIL"


def test_verdict_too_few_ticks_downgrades_pass_but_not_fail():
    v = hb.evaluate_verdict(measured(n_ticks=200), "MEASURED")
    assert v["rows"]["refresh_sync_p95_le_50ms"]["status"] == "UNMEASURED"
    assert "1000" in v["rows"]["refresh_sync_p95_le_50ms"]["reason"]
    assert not any(rd["decides_h_hc11"] for rd in v["readings"].values())
    f = hb.evaluate_verdict(measured(n_ticks=200, refresh_p95_ms=90.0), "MEASURED")
    assert f["rows"]["refresh_sync_p95_le_50ms"]["status"] == "FAIL"
    few_fast = hb.evaluate_verdict(measured(fast_concurrent_n=100), "MEASURED")
    assert few_fast["rows"]["fastpath_concurrent_p95_le_25ms"]["status"] == "UNMEASURED"


def test_verdict_mock_never_prints_pass_or_fail():
    for m in (measured(), measured(refresh_p95_ms=500.0, fast_concurrent_p95_ms=90.0)):
        v = hb.evaluate_verdict(m, "MOCK")
        assert {r["status"] for r in v["rows"].values()} == {"MOCK_ONLY"}
        assert v["readings"]["synchronous"]["overall"] == "MOCK_ONLY" and v["readings"]["asynchronous"]["overall"] == "MOCK_ONLY"
        assert not any(rd["decides_h_hc11"] for rd in v["readings"].values()) and "mock" in v["decides_h_hc11_gate"]
    v = hb.evaluate_verdict(measured(fast_concurrent_p95_ms=None, fast_concurrent_n=None), "MOCK")
    assert v["rows"]["fastpath_concurrent_p95_le_25ms"]["status"] == "UNMEASURED"       # no value -> still not a number


def test_verdict_not_on_thor_does_not_decide():
    v = hb.evaluate_verdict(measured(on_thor=False), "MEASURED")
    assert v["readings"]["synchronous"]["overall"] == "PASS" and "Thor" in v["decides_h_hc11_gate"]
    assert not any(rd["decides_h_hc11"] for rd in v["readings"].values())
    assert not any(rd["decides_h_hc11"] for rd in hb.evaluate_verdict(measured(on_thor=None), "MEASURED")["readings"].values())


def test_verdict_supplementary_rows_are_not_gating():
    v = hb.evaluate_verdict(measured(fast_overlapped_p95_ms=40.0, refresh_concurrent_p95_ms=400.0), "MEASURED")
    assert v["rows"]["supplementary_fastpath_p95_overlapped_only_le_25ms"]["status"] == "FAIL"
    assert v["rows"]["supplementary_refresh_p95_under_concurrency_le_250ms"]["status"] == "FAIL"
    assert v["readings"]["synchronous"]["overall"] == "PASS" and v["readings"]["asynchronous"]["overall"] == "PASS"


def test_verdict_supplementary_overlap_needs_100_overlapped_ticks():
    v = hb.evaluate_verdict(measured(fast_overlapped_n=40), "MEASURED")
    r = v["rows"]["supplementary_fastpath_p95_overlapped_only_le_25ms"]
    assert r["status"] == "UNMEASURED" and "100" in r["reason"]
    v = hb.evaluate_verdict(measured(fast_overlapped_n=40, fast_overlapped_p95_ms=50.0), "MEASURED")
    assert v["rows"]["supplementary_fastpath_p95_overlapped_only_le_25ms"]["status"] == "FAIL"    # a FAIL is never hidden


def test_verdict_e2e_is_fast_p95_plus_allowance():
    v = hb.evaluate_verdict(measured(fast_concurrent_p95_ms=20.0, preprocess_allowance_ms=8.0, tick_budget_ms=100.0), "MEASURED")
    r = v["rows"]["e2e_command_latency_proxy_le_tick_budget"]
    assert r["value"] == pytest.approx(28.0) and r["threshold"] == 100.0 and r["status"] == "PASS"
    v2 = hb.evaluate_verdict(measured(fast_concurrent_p95_ms=20.0, preprocess_allowance_ms=90.0), "MEASURED")
    assert v2["rows"]["e2e_command_latency_proxy_le_tick_budget"]["status"] == "FAIL"


# ---------------------------------------------------------------------------------------------
# feature comparison / drift
# ---------------------------------------------------------------------------------------------
def test_compare_pooled_known_geometry():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 16, 8)).astype(np.float32)
    same = hb.compare_pooled({"tap_0": a}, {"tap_0": a.copy()})["tap_0"]
    assert same["cos_token_mean"] == pytest.approx(1.0) and same["rel_l2_mean"] == pytest.approx(0.0, abs=1e-9)
    neg = hb.compare_pooled({"tap_0": a}, {"tap_0": -a})["tap_0"]
    assert neg["cos_token_mean"] == pytest.approx(-1.0) and neg["rel_l2_mean"] == pytest.approx(2.0)
    dbl = hb.compare_pooled({"tap_0": a}, {"tap_0": 2 * a})["tap_0"]
    assert dbl["cos_token_mean"] == pytest.approx(1.0) and dbl["rel_l2_mean"] == pytest.approx(1.0)
    b = a + 0.1 * rng.standard_normal(a.shape).astype(np.float32)
    r = hb.compare_pooled({"tap_0": a}, {"tap_0": b})["tap_0"]
    man_rel = np.mean([np.linalg.norm((a[i] - b[i]).ravel()) / np.linalg.norm(a[i].ravel()) for i in range(3)])
    assert r["rel_l2_mean"] == pytest.approx(float(man_rel), rel=1e-6)
    assert r["n_windows"] == 3 and r["n_tokens"] == 16 and r["width"] == 8
    with pytest.raises(ValueError):
        hb.compare_pooled({"tap_0": a}, {"tap_0": a[:, :8]})
    with pytest.raises(ValueError):
        hb.compare_pooled({"tap_0": a}, {"tap_1": a})


def _meta(**over):
    m = {"schema": hb.FEATURE_SCHEMA, "evidence_class": "MEASURED", "frames_source": "real", "frames_fingerprint": "abc", "taps": [1, 2, 3]}
    m.update(over)
    return m


def test_drift_verdict_labels():
    tap = {"tap_0": {"cos_token_mean": 0.995, "rel_l2_mean": 0.05}}
    assert hb.drift_verdict(tap, _meta(), _meta())["status"] == "NO_MARGIN_PREREGISTERED"
    assert hb.drift_verdict(tap, _meta(), _meta(), margin_cos=0.99)["status"] == "PASS"
    assert hb.drift_verdict(tap, _meta(), _meta(), margin_cos=0.999)["status"] == "FAIL"
    assert hb.drift_verdict(tap, _meta(), _meta(), margin_rel_l2=0.01)["status"] == "FAIL"
    assert hb.drift_verdict(tap, _meta(frames_source="synthetic"), _meta(), margin_cos=0.9)["status"] == "INADMISSIBLE_FOR_DRIFT"
    assert hb.drift_verdict(tap, _meta(), _meta(frames_fingerprint="zzz"), margin_cos=0.9)["status"] == "INADMISSIBLE_FRAMES_DIFFER"
    assert hb.drift_verdict(tap, _meta(evidence_class="MOCK"), _meta(), margin_cos=0.9)["status"] == "MOCK_ONLY"


def test_save_load_features_npz_roundtrip(tmp_path):
    feats = {"tap_0": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
    hb.save_features(tmp_path / "f.npz", _meta(), feats)
    meta, got = hb.load_features(tmp_path / "f.npz")
    assert meta["schema"] == hb.FEATURE_SCHEMA and np.array_equal(got["tap_0"], feats["tap_0"])
    with pytest.raises(hb.UsageError):
        hb.save_features(tmp_path / "f.bin", _meta(), feats)


def test_compare_features_cli_on_npz(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 8, 4)).astype(np.float32)
    b = a + 0.05 * rng.standard_normal(a.shape).astype(np.float32)
    hb.save_features(tmp_path / "bf16.npz", _meta(), {"tap_0": a, "tap_1": a})
    hb.save_features(tmp_path / "q.npz", _meta(), {"tap_0": b, "tap_1": b})
    out = tmp_path / "drift.json"
    r = run_cli("--compare-features", str(tmp_path / "bf16.npz"), str(tmp_path / "q.npz"), "--out", str(out))
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert set(d["per_tap"]) == {"tap_0", "tap_1"} and d["verdict"]["status"] == "NO_MARGIN_PREREGISTERED"
    assert 0.9 < d["per_tap"]["tap_0"]["cos_token_mean"] < 1.0
    out2 = tmp_path / "drift2.json"
    r = run_cli("--compare-features", str(tmp_path / "bf16.npz"), str(tmp_path / "q.npz"), "--drift-margin-cos", "0.9", "--out", str(out2))
    assert json.loads(out2.read_text())["verdict"]["status"] == "PASS"


def test_pt_features_without_torch_fail_loud(tmp_path):
    code = ("import sys, runpy; sys.modules['torch'] = None; sys.argv = ['x', '--compare-features', 'a.pt', 'b.pt'];"
            f"runpy.run_path(r'{SCRIPT}', run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode != 0 and "needs torch" in r.stderr


# ---------------------------------------------------------------------------------------------
# CLI (mock backend, subprocess) - JSON schema, MOCK labelling, fail-loud paths
# ---------------------------------------------------------------------------------------------
TOP_KEYS = {"schema", "schema_version", "hypothesis", "evidence_class", "evidence_note", "created_utc", "argv", "script", "config",
            "stamp", "model", "tokens", "processor", "frames", "refresh", "token_age_ms", "duty_cycle", "memory",
            "refresh_concurrent", "fast_path", "e2e_proxy", "notes", "verdict"}


def test_mock_cli_end_to_end_schema_and_mock_labelling(tmp_path):
    d, r = run_mock(tmp_path, "--cameras", "3", "--concurrent-fastpath")
    assert TOP_KEYS <= set(d), TOP_KEYS - set(d)
    assert d["schema"] == hb.SCHEMA and d["schema_version"] == hb.SCHEMA_VERSION and d["hypothesis"] == "H-HC11"
    assert d["evidence_class"] == "MOCK" and "MOCK" in d["evidence_note"]
    assert d["stamp"]["mock"] is True and d["stamp"]["torch"] == "NOT_IMPORTED"
    stages = d["refresh"]["stages"]
    assert set(stages) == {"vision_ms", "llm_prefill_ms", "pool_ms", "residual_ms", "total_ms"}
    for st in stages.values():
        assert set(st) == {"n", "mean", "std", "min", "p50", "p95", "p99", "max"} and st["n"] == 1000
    assert d["refresh"]["n_ticks"] == 1000 and d["refresh"]["warmup_ticks"] == 50
    sched = d["refresh"]["schedule"]
    assert {"n_executed", "n_dropped", "n_late", "period_ms", "policy", "paced"} <= set(sched)
    assert d["tokens"]["visual_tokens"] == 1440 and d["tokens"]["llm_input_tokens"] == 1504
    assert d["tokens"]["pooled_tokens_per_tap"] == 128 and d["tokens"]["pooled_tokens_total"] == 384
    assert d["memory"]["headline_probe"].startswith("torch.cuda.max_memory_allocated")
    assert d["frames"]["drift_label"] == "INADMISSIBLE_FOR_DRIFT"
    fp = d["fast_path"]
    assert fp["proxy"]["d_model"] == 512 and fp["proxy"]["layers"] == 6 and fp["proxy"]["tokens"] == 192
    assert fp["alone"]["n"] == 500 and fp["concurrent"]["n"] >= 500
    assert fp["p95_inflation_ratio_all_ticks"] > 1.0
    assert d["e2e_proxy"]["camera_frame_to_command_ms"] == pytest.approx(fp["concurrent"]["gpu_event_ms"]["p95"] + 8.0)
    v = d["verdict"]
    assert {row["status"] for row in v["rows"].values()} == {"MOCK_ONLY"}
    assert {rd["overall"] for rd in v["readings"].values()} == {"MOCK_ONLY"}
    assert not any(rd["decides_h_hc11"] for rd in v["readings"].values())
    assert '"PASS"' not in json.dumps(v) and '"FAIL"' not in json.dumps(v)         # a mock run prints neither
    assert set(v["readings"]) == {"synchronous", "asynchronous"}
    assert "MOCK" in r.stderr


def test_mock_without_concurrent_fastpath_leaves_fast_rows_unmeasured(tmp_path):
    d, _ = run_mock(tmp_path, "--ticks", "40", "--warmup", "2")
    assert d["fast_path"] is None and d["e2e_proxy"] is None
    rows = d["verdict"]["rows"]
    assert rows["fastpath_concurrent_p95_le_25ms"]["status"] == "UNMEASURED"
    assert rows["refresh_sync_p95_le_50ms"]["status"] == "MOCK_ONLY"


def test_mock_warmup_is_excluded_from_stats(tmp_path):
    common = ("--mock-jitter", "0", "--mock-spike-prob", "0", "--ticks", "30")
    cold, _ = run_mock(tmp_path, *common, "--warmup", "0", name="cold.json")
    warm, _ = run_mock(tmp_path, *common, "--warmup", "20", name="warm.json")
    ct, wt = cold["refresh"]["stages"]["total_ms"], warm["refresh"]["stages"]["total_ms"]
    assert ct["max"] == pytest.approx(3.0 * ct["min"], rel=1e-6)         # first 5 mock calls are 3x slower
    assert wt["max"] == pytest.approx(wt["min"], rel=1e-9) and wt["n"] == 30


def test_mock_overrun_reports_late_and_dropped_ticks(tmp_path):
    d, _ = run_mock(tmp_path, "--cameras", "3", "--mock-vit-ms-per-image", "100", "--ticks", "40", "--warmup", "0",
                    "--mock-jitter", "0", "--mock-spike-prob", "0")
    s = d["refresh"]["schedule"]
    assert s["n_late"] > 0 and s["n_dropped"] > 0 and s["n_overrun"] == 40
    assert d["duty_cycle"]["value"] > 1.0
    age = d["token_age_ms"]["alone_phase"]
    assert age["p95"] > 500 + d["refresh"]["stages"]["total_ms"]["p95"]    # queueing pushes age past period + compute


def test_mock_no_pace_makes_schedule_and_age_unmeasured(tmp_path):
    d, _ = run_mock(tmp_path, "--no-pace", "--ticks", "20", "--warmup", "0")
    assert d["refresh"]["schedule"]["n_dropped"] is None and d["refresh"]["schedule"]["n_late"] is None
    assert d["token_age_ms"]["alone_phase"] is None
    assert d["verdict"]["rows"]["token_age_p95_le_500ms_plus_refresh_p95"]["status"] == "UNMEASURED"


def test_mock_contention_model_inflates_overlapped_ticks(tmp_path):
    d, _ = run_mock(tmp_path, "--concurrent-fastpath", "--ticks", "40", "--warmup", "0", "--concurrent-ticks", "60",
                    "--mock-jitter", "0", "--mock-spike-prob", "0", "--mock-contention-alpha", "2.0")
    c = d["fast_path"]["concurrent"]
    assert c["overlapped_with_refresh"]["n"] > 0 and c["not_overlapped"]["n"] > 0
    assert c["overlapped_with_refresh"]["gpu_event_ms"]["p50"] > c["not_overlapped"]["gpu_event_ms"]["p50"]
    assert d["fast_path"]["alone"]["gpu_event_ms"]["p95"] == pytest.approx(3.0)      # jitter 0 -> base latency exactly
    assert d["fast_path"]["p95_inflation_ratio_overlapped_only"] > d["fast_path"]["p95_inflation_ratio_all_ticks"] or \
        d["fast_path"]["p95_inflation_ratio_overlapped_only"] > 1.0


def test_mock_is_deterministic(tmp_path):
    a, _ = run_mock(tmp_path, "--ticks", "60", "--warmup", "3", name="a.json")
    b, _ = run_mock(tmp_path, "--ticks", "60", "--warmup", "3", name="b.json")
    assert a["refresh"]["stages"] == b["refresh"]["stages"]


@pytest.mark.parametrize("backend", ["mock", "hf"])
@pytest.mark.parametrize("dtype", ["fp8", "nvfp4"])
def test_fp8_nvfp4_fail_loud_with_trt_instruction(backend, dtype, tmp_path):
    out = tmp_path / "x.json"
    r = run_cli("--backend", backend, "--model", "qwen3-vl-4b", "--dtype", dtype, "--out", str(out))
    assert r.returncode == 2
    assert "TensorRT-Edge-LLM" in r.stderr and "--compare-features" in r.stderr and dtype in r.stderr
    assert not out.exists()


def test_hf_backend_without_torch_fails_loud_instead_of_mocking(tmp_path):
    out = tmp_path / "x.json"
    code = ("import sys, runpy; sys.modules['torch'] = None;"          # makes `import torch` raise ImportError, torch present or not
            f"sys.argv = ['x', '--backend', 'hf', '--model', 'qwen3-vl-4b', '--out', r'{out}'];"
            f"runpy.run_path(r'{SCRIPT}', run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 3
    assert "needs torch" in r.stderr and "mock" in r.stderr.lower() and "Refusing" in r.stderr
    assert not out.exists()                                            # no JSON of any kind was produced


def test_hf_backend_requires_model_and_valid_resolution(tmp_path):
    r = run_cli("--backend", "hf", "--out", str(tmp_path / "x.json"))
    assert r.returncode == 2 and "--model is required" in r.stderr
    r = run_cli("--backend", "mock", "--resolution", "256-640", "--out", str(tmp_path / "x.json"))
    assert r.returncode == 2 and "HxW" in r.stderr


def test_fast_jitter_validation_and_overlap_share(tmp_path):
    r = run_cli("--backend", "mock", "--fast-jitter-frac", "0.6", "--out", str(tmp_path / "x.json"))
    assert r.returncode == 2 and "--fast-jitter-frac" in r.stderr
    d, _ = run_mock(tmp_path, "--concurrent-fastpath", "--ticks", "30", "--warmup", "0", "--concurrent-ticks", "100")
    fp = d["fast_path"]
    assert fp["fast_tick_jitter_frac"] == 0.25 and 0.0 < fp["overlap_share_of_concurrent_ticks"] < 1.0


def test_cameras_choices_enforced(tmp_path):
    r = run_cli("--backend", "mock", "--cameras", "5", "--out", str(tmp_path / "x.json"))
    assert r.returncode == 2


def test_help_epilog_has_one_thor_command_per_candidate():
    r = run_cli("--help")
    assert r.returncode == 0
    for key, c in hb.CANDIDATES.items():
        assert f"--model {key} " in r.stdout, key
    assert r.stdout.count("\n  OMP_NUM_THREADS=6 python3 hicap_backbone_latency_bench.py --backend hf") == len(hb.CANDIDATES)
    assert "for C in 1 3 7" in r.stdout and "--cameras 1 " in r.stdout
    assert "PYTHONPATH" in r.stdout and "--compare-features" in r.stdout and "MOCK_ONLY" in r.stdout


def test_list_candidates():
    r = run_cli("--list-candidates")
    assert r.returncode == 0
    for c in hb.CANDIDATES.values():
        assert c["hf_id"] in r.stdout
    assert "UNVERIFIED" in r.stdout


def test_mock_save_features_and_compare_are_mock_labelled(tmp_path):
    for seed in (0, 1):
        r = run_cli("--backend", "mock", "--model", "qwen3-vl-4b", "--ticks", "5", "--warmup", "0", "--seed", str(seed),
                    "--save-features", str(tmp_path / f"m{seed}.npz"), "--out", str(tmp_path / f"m{seed}.json"))
        assert r.returncode == 0, r.stderr
    d = json.loads((tmp_path / "m0.json").read_text())
    assert d["saved_features"]["meta"]["evidence_class"] == "MOCK" and d["saved_features"]["shapes"]["tap_0"] == [2, 128, 64]
    r = run_cli("--compare-features", str(tmp_path / "m0.npz"), str(tmp_path / "m1.npz"), "--out", str(tmp_path / "drift.json"))
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "drift.json").read_text())["verdict"]["status"] == "MOCK_ONLY"


# =============================================================================================
# Part 2 - the real HFBackend code path against FAKE torch / transformers / PIL
# =============================================================================================
class FDType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


BF16, FP16, FP32 = FDType("bfloat16"), FDType("float16"), FDType("float32")


class FT:
    """numpy-backed stand-in for a tensor; implements only what HFBackend touches."""

    def __init__(self, a, dtype=None):
        self.a, self.dtype = np.asarray(a), dtype

    shape = property(lambda self: self.a.shape)

    def to(self, *args, **kw):
        dt = next((x for x in args if isinstance(x, FDType)), kw.get("dtype"))
        return FT(self.a, dt or self.dtype)

    def pin_memory(self):
        return self

    def long(self):
        return FT(self.a.astype(np.int64))

    def __eq__(self, other):
        return FT(self.a == other)

    def sum(self):
        return FT(self.a.sum())

    def item(self):
        return self.a.item()

    def transpose(self, i, j):
        return FT(np.swapaxes(self.a, i, j), self.dtype)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.a, dtype=np.float32)

    def __getitem__(self, idx):
        return FT(self.a[idx], self.dtype)


class FEvent:
    def __init__(self, enable_timing=False):
        self.t = None

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        assert self.t is not None, "event synchronised before record"

    def elapsed_time(self, other):
        return (other.t - self.t) * 1e3


class FParam:
    def __init__(self, n, dtype):
        self.n, self.dtype, self.rg = n, dtype, True

    def numel(self):
        return self.n

    def element_size(self):
        return 4 if self.dtype is FP32 else 2

    def requires_grad_(self, flag):
        self.rg = flag


class FModule:
    sleep_ms = 0.0

    def __init__(self):
        self._pre, self._post, self._params = [], [], []

    def register_forward_pre_hook(self, h):
        self._pre.append(h)

    def register_forward_hook(self, h):
        self._post.append(h)

    def __call__(self, *a, **kw):
        for h in self._pre:
            h(self, a)
        time.sleep(self.sleep_ms / 1e3)
        out = self.forward(*a, **kw)
        for h in self._post:
            h(self, a, out)
        return out

    def forward(self, *a, **kw):
        return None

    def parameters(self):
        return list(self._params)

    def buffers(self):
        return []

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self


class FVisual(FModule):
    sleep_ms = 4.0

    def __init__(self, dtype):
        super().__init__()
        self._params = [FParam(1000, dtype)]


class FLLM(FModule):
    sleep_ms = 6.0

    def __init__(self, dtype):
        super().__init__()
        self._params = [FParam(4000, dtype)]
        self.config = SimpleNamespace(use_cache=True)


class FCore(FModule):
    """Mimics Qwen3VLModel.forward: NO use_cache / return_dict parameters (they would flow into the vision blocks)."""

    def __init__(self, dtype, n_layers):
        super().__init__()
        self.visual, self.language_model, self.n_layers = FVisual(dtype), FLLM(dtype), n_layers
        self.calls = []

    def parameters(self):
        return self.visual.parameters() + self.language_model.parameters()

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None,
                output_hidden_states=False):
        self.calls.append({"mm": mm_token_type_ids is not None, "ohs": output_hidden_states, "dtype": pixel_values.dtype})
        self.visual(pixel_values)
        self.language_model(input_ids)
        n = input_ids.shape[1]
        rng = np.random.default_rng(3)
        hs = tuple(FT(rng.standard_normal((1, n, 8)).astype(np.float32), BF16) for _ in range(self.n_layers + 1)) if output_hidden_states else None
        return SimpleNamespace(hidden_states=hs)


class FModel(FModule):
    def __init__(self, dtype, n_layers=6):
        super().__init__()
        self.model = FCore(dtype, n_layers)
        self.lm_head = FLLM(dtype)
        self.config = SimpleNamespace(use_cache=True, text_config=SimpleNamespace(num_hidden_layers=n_layers, hidden_size=8, use_cache=True))

    def parameters(self):
        return self.model.parameters() + (self.lm_head.parameters() if self.lm_head is not None else [])


class FIP:
    patch_size, merge_size, temporal_patch_size, size = 16, 2, 2, None


class FProc:
    image_token_id = 7

    def __init__(self, per_image=160):
        self.image_processor, self.per_image = FIP(), per_image

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True):
        content = conv[0]["content"]
        return f"IMG{sum(c['type'] == 'image' for c in content)}|" + content[-1]["text"]

    def __call__(self, text, images, return_tensors):
        head, words = text[0].split("|", 1)
        n_img, n_words = int(head[3:]), len(words.split())
        ids = [1] * 3 + [7] * (n_img * self.per_image) + [2] * (n_words + 2 * n_img)
        return {"input_ids": FT(np.array([ids])), "attention_mask": FT(np.ones((1, len(ids)), dtype=np.int64)),
                "pixel_values": FT(np.zeros((n_img * 4, 6), dtype=np.float32), FP32),
                "image_grid_thw": FT(np.tile([[1, 16, 40]], (n_img, 1)))}


class FImage:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.uint8)
        self.size = (self.arr.shape[1], self.arr.shape[0])

    def __array__(self, dtype=None, copy=None):
        return self.arr if dtype is None else self.arr.astype(dtype)

    def convert(self, mode):
        return self

    def resize(self, size, resample=None):
        return FImage(np.zeros((size[1], size[0], 3), dtype=np.uint8))


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    state = SimpleNamespace(load_kwargs=None, model=None, proc=FProc(), param_dtype=None)

    torch = types.ModuleType("torch")
    torch.__version__ = "fake-torch"
    torch.bfloat16, torch.float16, torch.float32 = BF16, FP16, FP32
    torch.version = SimpleNamespace(cuda="fake-cuda")
    torch.backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 0))
    torch.device = lambda s: s
    torch.set_num_threads = lambda n: None
    torch.manual_seed = lambda s: None
    torch.inference_mode = lambda: contextlib.nullcontext()

    class FStream:
        def __init__(self, device=None, priority=0):
            self.priority = priority

        def synchronize(self):
            pass

    torch.cuda = SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0, get_device_name=lambda d=0: "Fake Thor",
        get_device_capability=lambda d=0: (11, 0), get_device_properties=lambda d=0: SimpleNamespace(multi_processor_count=20),
        synchronize=lambda: None, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 4_000_000_000,
        mem_get_info=lambda: (123, 456), Event=FEvent, Stream=FStream, stream=lambda s: contextlib.nullcontext())

    def pool(x, k):
        parts = np.array_split(x.a, k, axis=-1)
        return FT(np.stack([p.mean(-1) for p in parts], axis=-1), x.dtype)
    torch.nn = SimpleNamespace(functional=SimpleNamespace(adaptive_avg_pool1d=pool))

    tf = types.ModuleType("transformers")
    tf.__version__ = "fake-transformers"

    class AutoModel:
        @staticmethod
        def from_pretrained(hf_id, **kw):
            state.load_kwargs = dict(kw, hf_id=hf_id)
            state.model = FModel(state.param_dtype or kw.get("dtype") or kw.get("torch_dtype"))
            return state.model

    class AutoProc:
        @staticmethod
        def from_pretrained(hf_id, **kw):
            return state.proc
    tf.AutoModelForImageTextToText, tf.AutoProcessor = AutoModel, AutoProc

    pil = types.ModuleType("PIL")
    pil.Image = SimpleNamespace(fromarray=lambda a: FImage(a), open=lambda p: FImage(np.zeros((256, 640, 3), np.uint8)), BICUBIC=3)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "PIL", pil)

    def stub_proxy(self):        # the real proxy is plain nn code; it cannot run on the fake torch (UNTESTED here)
        self._proxy = lambda x: None
        self._proxy_x = None
        self._proxy_info = {**hb.PROXY_SPEC, "kind": "STUB in tests", "params": 0}
    monkeypatch.setattr(hb.HFBackend, "_build_proxy", stub_proxy)
    return state


HF_ARGS = ["--backend", "hf", "--model", "qwen3-vl-4b", "--cameras", "1", "--frames-per-camera", "3", "--ticks", "6", "--warmup", "2",
           "--refresh-hz", "50", "--fast-hz", "200"]


def test_hf_backend_real_code_path_on_fake_torch(fakes, tmp_path, capsys):
    out, feats = tmp_path / "hf.json", tmp_path / "hf.npz"
    rc = hb.main([*HF_ARGS, "--concurrent-fastpath", "--fast-baseline-ticks", "20", "--concurrent-ticks", "5",
                  "--save-features", str(feats), "--log-inadmissible-memory", "--out", str(out)])
    assert rc == 0, capsys.readouterr().err
    d = json.loads(out.read_text())
    assert d["evidence_class"] == "MEASURED" and d["config"]["backend"] == "hf" and d["stamp"]["device_name"] == "Fake Thor"
    assert d["stamp"]["on_thor"] is True
    # loading: requested dtype passed, lm_head dropped, use_cache disabled through the configs
    assert fakes.load_kwargs["dtype"] is BF16 and fakes.load_kwargs["hf_id"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert fakes.model.lm_head is None and d["model"]["lm_head_dropped"] is True
    assert fakes.model.config.use_cache is False and fakes.model.model.language_model.config.use_cache is False
    # tokens: 1 camera x 3 frames -> 480 image tokens + 64 non-image tokens; taps auto on 6 layers; pooled to <=128
    t = d["tokens"]
    assert t["visual_tokens_actual"] == 480 == t["visual_tokens"] and t["llm_input_tokens_actual"] == 544
    assert t["prompt_tokens_actual_incl_markers_and_template"] == 64
    assert d["model"]["taps"] == [2, 4, 6] and d["model"]["hidden_states_len"] == 7
    assert t["pooled_tokens_per_tap"] == 128 and t["pooled_tokens_total"] == 384
    # the model was called with output_hidden_states and an auto-built mm_token_type_ids, pixel_values in the model dtype
    calls = fakes.model.model.calls
    assert calls and all(c["ohs"] is True and c["mm"] is True and c["dtype"] is BF16 for c in calls)
    # stage timing: hooks put the (slept) tower times in the right stage, and the stages add up to the total
    st = d["refresh"]["stages"]
    assert d["refresh"]["n_ticks"] == 6 and st["total_ms"]["n"] == 6
    assert 4.0 <= st["vision_ms"]["p50"] < 6.5 and 6.0 <= st["llm_prefill_ms"]["p50"] < 8.5
    assert st["total_ms"]["p50"] >= st["vision_ms"]["p50"] + st["llm_prefill_ms"]["p50"] - 1e-6
    assert st["residual_ms"]["min"] >= -1e-6
    assert d["memory"]["peak_allocated_bytes"] == 4_000_000_000 and d["memory"]["peak_allocated_gb"] == pytest.approx(4.0)
    assert d["memory"]["inadmissible_on_thor"]["label"] == "INADMISSIBLE_ON_THOR"
    assert d["model"]["weight_bytes"] == 2 * 1000 + 2 * 4000                # core only: lm_head excluded
    # frames: synthetic -> drift-inadmissible label
    assert d["frames"]["source"] == "synthetic" and d["frames"]["drift_label"] == "INADMISSIBLE_FOR_DRIFT"
    # paced schedule on the real clock
    assert d["refresh"]["schedule"]["paced"] is True and d["refresh"]["schedule"]["n_executed"] == 6
    # fast path phases ran on the real clock and are reported per phase
    fp = d["fast_path"]
    assert fp["alone"]["n"] == 20 and fp["concurrent"]["n"] > 0 and fp["concurrent"]["overlapped_with_refresh"]["n"] > 0
    # fewer than 1000 ticks -> nothing may claim PASS, and the run cannot decide H-HC11
    assert not any(r["status"] == "PASS" for r in d["verdict"]["rows"].values())
    assert not any(rd["decides_h_hc11"] for rd in d["verdict"]["readings"].values())
    # features saved, comparable to themselves
    meta, arrs = hb.load_features(feats)
    assert meta["frames_source"] == "synthetic" and meta["taps"] == [2, 4, 6] and arrs["tap_0"].shape == (1, 128, 8)
    assert hb.drift_verdict(hb.compare_pooled(arrs, arrs), meta, meta, margin_cos=0.99)["status"] == "INADMISSIBLE_FOR_DRIFT"


def test_hf_backend_token_guard_fails_loud(fakes, tmp_path, capsys):
    fakes.proc = FProc(per_image=286)                # the min-pixels upscale pitfall
    rc = hb.main([*HF_ARGS, "--out", str(tmp_path / "x.json")])
    assert rc == 1 and "image-token count" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_hf_backend_dtype_guard_fails_loud(fakes, tmp_path, capsys):
    fakes.param_dtype = FP32                          # loader silently ignored dtype=
    rc = hb.main([*HF_ARGS, "--out", str(tmp_path / "x.json")])
    assert rc == 1 and "fewer than 90%" in capsys.readouterr().err


def test_hf_backend_no_cuda_refuses(fakes, tmp_path, capsys):
    sys.modules["torch"].cuda.is_available = lambda: False
    rc = hb.main([*HF_ARGS, "--out", str(tmp_path / "x.json")])
    assert rc == 3 and "CUDA" in capsys.readouterr().err


def test_hf_backend_bad_taps_fail_loud(fakes, tmp_path, capsys):
    rc = hb.main([*HF_ARGS, "--taps", "1,2,99", "--out", str(tmp_path / "x.json")])
    assert rc == 1 and "outside" in capsys.readouterr().err
