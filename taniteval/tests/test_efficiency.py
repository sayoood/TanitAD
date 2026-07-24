"""Tests for ``taniteval/efficiency.py`` — the inference-efficiency panel.

These are the tests that keep an efficiency comparison HONEST. The failure mode
this panel invites is not a crash, it is a *plausible wrong number*: two arms
timed under different precision, a stage share that does not add up, or a
contaminated GPU published as a clean result. Each of those has a test here.

CPU-only by design (no CUDA needed) — the timing primitives themselves are
exercised on the pod by the real runs; what is pinned here is the arithmetic and
the fairness contract.

pytest is NOT installed on the eval pod, so these run standalone too:
  python taniteval/tests/test_efficiency.py
"""
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, "/root/taniteval")              # pod layout

from taniteval import efficiency as E  # noqa: E402


# --------------------------------------------------------------------------- #
# fairness contract                                                             #
# --------------------------------------------------------------------------- #
def test_precision_moves_both_tf32_switches_together():
    """Matmul-only TF32 silently favours the conv-heavy arm (REF-C) over the
    matmul-heavy arm (flagship ViT). The two switches must move together."""
    import torch
    for mode in E.PRECISIONS:
        _, rec = E._set_precision(mode)
        assert rec["matmul_allow_tf32"] == rec["cudnn_allow_tf32"], \
            f"{mode}: TF32 switches diverged -> unfair cross-arm comparison"
        assert rec["matmul_allow_tf32"] == torch.backends.cuda.matmul.allow_tf32
        assert rec["precision"] == mode
    _, rec = E._set_precision("fp32")
    assert rec["matmul_allow_tf32"] is False, "fp32 must disable TF32"
    _, rec = E._set_precision("amp16")
    assert rec["autocast"] == "float16"


def test_precision_is_recorded_in_every_measurement():
    """A latency number without its precision is not publishable."""
    _, rec = E._set_precision("tf32")
    for k in ("precision", "autocast", "matmul_allow_tf32", "cudnn_allow_tf32",
              "cudnn_benchmark", "weights_dtype"):
        assert k in rec, f"precision record missing {k}"


def test_unknown_precision_fails_loud():
    try:
        E._set_precision("bf8_turbo")
    except AssertionError:
        return
    raise AssertionError("unknown precision must fail loud, not fall back")


def test_measure_restores_backend_flags():
    """`measure` runs INSIDE the accuracy harness. If it leaves TF32 or
    cudnn.benchmark flipped, the NEXT arm's ADE is computed under different
    numerics — an efficiency probe must never be able to move an accuracy
    number. Uses a deliberately failing measurement so no GPU is needed."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    before = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        E.measure({"arch": "nope"}, {"model": None}, None, precision="tf32")
    except Exception:
        pass
    after = (torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    assert before == after, f"backend flags leaked: {before} -> {after}"


# --------------------------------------------------------------------------- #
# statistics                                                                    #
# --------------------------------------------------------------------------- #
def test_percentiles_are_ordered_and_bracketed():
    xs = [float(i) for i in range(1, 101)]
    p50, p95, p99 = E._pct(xs, .50), E._pct(xs, .95), E._pct(xs, .99)
    assert min(xs) <= p50 <= p95 <= p99 <= max(xs)
    assert abs(p50 - 50.5) <= 1.0          # 50th of 1..100
    assert p99 >= 99.0


def test_percentile_single_sample():
    assert E._pct([7.0], .99) == 7.0


# --------------------------------------------------------------------------- #
# stage decomposition                                                           #
# --------------------------------------------------------------------------- #
def _st(**kw):
    return {k: {"mean_ms": v} for k, v in kw.items()}


def test_shares_rollout_arm_decomposes_the_budget():
    """World-model arm: encoder + 20-step serial rollout must account for the
    plan step, and the marginal per-step cost must be recovered from k20 vs k1."""
    st = _st(**{"encode_window_8frames": 28.0, "encode_1frame": 4.0,
                "rollout_k20": 82.0, "rollout_k1": 6.0})
    sh = E._shares(st, 110.0, {"sequential_steps": 20})
    assert abs(sh["encoder_pct"] - 25.5) < 0.2
    assert abs(sh["rollout_pct"] - 74.5) < 0.2
    assert abs(sh["encoder_pct"] + sh["rollout_pct"] - 100.0) < 1.0
    # (82 - 6) / 19 = 4.0 ms per marginal rollout step
    assert abs(sh["per_rollout_step_ms"] - 4.0) < 1e-6
    # caching 7 of 8 encoder frames saves exactly enc - enc1
    assert abs(sh["cached_window_saving_ms"] - 24.0) < 1e-6
    assert abs(sh["plan_step_ms_if_encoder_cached"] - 86.0) < 1e-6


def test_shares_diffusion_arm_splits_classifier_from_denoise():
    """Anchored-diffusion arm: the denoise cost is decoder_full - classifier,
    and the per-pass cost divides by the declared step count."""
    st = _st(**{"encode_window_8frames": 38.0, "encode_1frame": 5.0,
                "decoder_classifier_steps0": 2.0, "decoder_full_steps2": 6.0,
                "imagination_h15": 1.0})
    sh = E._shares(st, 44.0, {"denoise_steps": 2})
    assert abs(sh["denoise_ms"] - 4.0) < 1e-6
    assert abs(sh["per_denoise_pass_ms"] - 2.0) < 1e-6
    assert abs(sh["decoder_total_ms"] - 6.0) < 1e-6
    assert sh["encoder_pct"] > sh["decoder_total_pct"], \
        "this fixture is encoder-dominated; the share ordering must reflect it"
    assert "imagination_pct" in sh


def test_shares_tolerates_missing_and_errored_stages():
    sh = E._shares({"encode_window_8frames": {"error": "boom"}}, 10.0, {})
    assert sh == {} or "encoder_pct" not in sh


# --------------------------------------------------------------------------- #
# report panel + artifact contract                                              #
# --------------------------------------------------------------------------- #
def _fake_eff(key, ms, gflops, ade):
    return {
        "key": key, "model": {"name": key},
        "ckpt_step": 29999,
        "fp32": {
            "precision": {"precision": "fp32"},
            "meta": {"decode": "test decode"},
            "params": {"total_params_m": 263.4},
            "plan_step": {"p50_ms": ms, "p95_ms": ms * 1.05, "p99_ms": ms * 1.1,
                          "mean_ms": ms},
            "memory": {"peak_alloc_mb": 1217.0},
            "flops": {"gflops": gflops},
            "stage_shares": {"encoder_pct": 25.3, "rollout_pct": 74.5},
            "realtime": {"budget_used_pct_p99": ms * 1.1,
                         "meets_10hz_p99": ms * 1.1 < 100},
        },
        "cost_per_accuracy": {"ade_0_2s_heldout": ade},
    }


def test_panel_rows_renders_and_flags_the_10hz_verdict():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "eff_a.json").write_text(json.dumps(_fake_eff("a", 110.0, 259.0,
                                                           0.4522)))
        (p / "eff_b.json").write_text(json.dumps(_fake_eff("b", 43.9, 701.0,
                                                           0.458)))
        html = E.panel_rows(p)
        assert html.count("<tr>") == 2
        assert "110.0" in html and "43.9" in html
        assert "0.4522" in html and "0.4580" in html
        assert "crit" in html, "the arm over the 100 ms budget must be flagged"
        assert "fp32" in html, "precision must appear in the row"


def test_contaminated_run_is_quarantined_not_published(monkeypatch=None):
    """A shared-GPU measurement must never overwrite a clean artifact. This is
    the exact accident of 2026-07-20: a 20-iteration smoke landed on top of a
    clean 200-iteration result with a 2x-slower (contaminated) number."""
    import types
    from taniteval import efficiency as M
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        clean = _fake_eff("x", 44.3, 700.0, 0.458)
        (p / "eff_x.json").write_text(json.dumps(clean))
        fake = {"precision": {"precision": "fp32"}, "meta": {}, "params": {},
                "plan_step": {"p50_ms": 91.8, "p95_ms": 94.3, "p99_ms": 94.5},
                "memory": {"peak_alloc_mb": 1.0},
                "realtime": {"budget_used_pct_p99": 94.5,
                             "meets_10hz_p99": True},
                "contamination_check": {"valid": False}}
        orig_load, orig_measure = M._load, M.measure
        M._load = lambda k, device="cuda": ({"key": k}, {"step": 1}, None)
        M.measure = lambda *a, **kw: dict(fake)
        try:
            M.run_and_save("x", res_dir=p)
        finally:
            M._load, M.measure = orig_load, orig_measure
        assert json.loads((p / "eff_x.json").read_text()) == clean, \
            "a contaminated run overwrote the clean artifact"
        q = list(p.glob("eff_x.CONTAMINATED-*.json"))
        assert len(q) == 1, "contaminated run must be quarantined to its own file"
        assert "QUARANTINED" in json.loads(q[0].read_text())


def test_panel_renders_canonical_artifacts_only():
    """Quarantined runs and hand-kept replicates sit beside the canonical file;
    they must not show up as extra leaderboard rows."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "eff_a.json").write_text(json.dumps(_fake_eff("a", 44.3, 700., .458)))
        q = _fake_eff("a", 91.8, 700., .458)
        q["QUARANTINED"] = "GPU was not exclusive"
        (p / "eff_a.CONTAMINATED-20260720-220309.json").write_text(json.dumps(q))
        (p / "eff_a.replicate-fp32-2153.json").write_text(
            json.dumps(_fake_eff("a", 97.1, 700., .458)))
        html = E.panel_rows(p)
        assert html.count("<tr>") == 1, "only the canonical eff_<key>.json renders"
        assert "44.3" in html and "91.8" not in html and "97.1" not in html


def test_panel_rows_empty_dir_is_not_an_error():
    with tempfile.TemporaryDirectory() as d:
        assert E.panel_rows(Path(d)) == ""


def test_panel_rows_skips_corrupt_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "eff_bad.json").write_text("{not json")
        (p / "eff_ok.json").write_text(json.dumps(_fake_eff("ok", 50.0, 100.0,
                                                            0.5)))
        assert E.panel_rows(p).count("<tr>") == 1


def test_realtime_budget_constants_match_the_eval_protocol():
    """The arms are trained/scored at 10 Hz over a 20-step 2 s horizon; the
    headroom verdict is meaningless if these drift from the harness."""
    from taniteval import rollout as ro
    assert E.K_MAX == ro.K_MAX == 20
    assert E.WINDOW == 8
    assert E.DT_HZ == 10.0 and E.RT_BUDGET_MS == 100.0
    assert abs(1.0 / ro.DT - E.DT_HZ) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                    # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"==== {len(fns) - bad}/{len(fns)} passed ====")
    sys.exit(1 if bad else 0)
