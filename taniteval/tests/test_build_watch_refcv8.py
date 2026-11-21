"""The refcv8 Training Watch builds from a run's own artifacts, compares it to refcv7's complete record --
and every guard on that comparison can FIRE.

`taniteval/tools/training_watch/build_watch_refcv8.py` reuses the refcv7 builder (a FRESH module instance,
configured for refcv8) and post-processes its page. It runs here with no network (`build()`, never `pull()`)
on tiny synthetic runs in a temp dir. The expected values are LITERALS written out below, never expressions
over the builder.

Deliberate-regression arms (each must go RED on the specific defect it names):
  * the refcv7 reference band is computed from refcv7's LAST 6 eval rows -- an early outlier row must not move
    it, and pointing the reference at refcv8's own (unfinished) record must REFUSE rather than compute a band;
  * a key the log lacks renders UNAVAILABLE and a JSON null renders "undefined" -- neither is ever a 0, and a
    real 0 still renders as 0;
  * the SAME-WINDOWS verdict goes INDICATIVE ONLY when the eval cache, the eval labels, the window count, the
    trainer evidence or ANY non-tactical label-derived counter differs; a tactical-only counter difference keeps
    the verdict but marks the tactical metrics;
  * `eval_loss_ex_r8` is derived from the logged terms (eval_loss - eval_r8_total), a row lacking the term is
    undefined, never eval_loss itself;
  * a refcv8 build leaves a separately loaded refcv7 builder exactly as it was.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "taniteval" / "tools" / "training_watch"
BUILDER8 = TOOLS / "build_watch_refcv8.py"
ARM7, ARM8 = "refcv7-r101-s0", "refcv8-r101-s0"
RUN8 = "/home/nvidia/refcv8_run/runs/refcv8-r101-s0"
DIAG_LINE = "W1015 10:27:37.197000 1077980 torch/_inductor/utils.py:1953] [0/2] Not enough SMs to use max_autotune_gemm mode"


def _refcv7_builder() -> Path:
    """build_watch_refcv7.py: this file's sibling when landed, else $REFCV7_BUILDER (the read-only launch-commit
    archive), else the dev box's archive."""
    for c in (TOOLS / "build_watch_refcv7.py", os.environ.get("REFCV7_BUILDER"),
              "C:/lgs/Lfb47a58376/taniteval/tools/training_watch/build_watch_refcv7.py"):
        if c and Path(c).is_file():
            return Path(c)
    pytest.skip("build_watch_refcv7.py not found (sibling, $REFCV7_BUILDER, or the launch-commit archive)")


def _load8(monkeypatch):
    monkeypatch.setenv("REFCV7_BUILDER", str(_refcv7_builder()))
    for k in ("REFCV7_RUN", "REFCV8_RUN", "REFCV8_WATCH_DIR", "REFCV7_REF_DIR", "REFCV7_ARM"):
        monkeypatch.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location("build_watch_refcv8_under_test", BUILDER8)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load7_separately(monkeypatch):
    monkeypatch.setenv("REFCV7_EVAL_PKG", str(Path(__file__).resolve().parent / "no-such-eval-package"))
    spec = importlib.util.spec_from_file_location("build_watch_refcv7_separate", _refcv7_builder())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the synthetic runs ----
N_CONST = 24          # constant label-derived counters in every eval row of both runs


def _counters(tac_diff=False, perception_diff=False):
    c = {"eval_windows": 128, "eval_batches": 8, "eval_box3d_n_pos": 339 + (1 if perception_diff else 0)}
    c.update({f"eval_agent_n_c{i}": float(10 + i) for i in range(N_CONST)})
    c["eval_tacv6_n_supervised_lat"] = 13.5 if tac_diff else 3.625
    return c


def _argv(steps, eval_cache="/data/eval139", extra=()):
    return ["--trunk-name", "resnet101.a1_in1k", "--eval-cache", eval_cache, "--eval-labels", "/data/s2_eval.jsonl.gz",
            "--speed-max-sidecar-v6-eval", "/data/speed_eval.jsonl", "--eval-every", "500", "--eval-batches", "8",
            "--save-every", "500", "--log-every", "50", "--batch", "16", "--steps", str(steps), *extra]


def _cfg(steps, *, train_windows, clips, eval_cache="/data/eval139", n_eval=23772, ema=None, extra=()):
    c = {"argv": _argv(steps, eval_cache, extra), "seed": 0, "v2_parity": {"clips_present": clips},
         "agent_join_stats": {"train": {"n_windows": train_windows}, "eval": {"n_windows": n_eval}}}
    if ema:
        c["ema"] = ema
    return c


def _write(d: Path, rows, cfg):
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _ref_dir(tmp_path, *, last_step=50400, outlier=True, drop_key=None):
    """refcv7's 'complete' record: eval rows every 500, then 50,400. eval_loss over the last six evals
    (48000, 48500, 49000, 49500, 50000, 50400) = 10,12,10,12,10,12 -> mean 11, sd(n-1) = sqrt(1.2)."""
    rows = []
    steps = list(range(500, 50001, 500)) + [50400]
    last6 = {48000: 10.0, 48500: 12.0, 49000: 10.0, 49500: 12.0, 50000: 10.0, 50400: 12.0}
    for st in steps:
        if st > last_step:
            break
        rows.append({"step": st, "loss": 50.0, "elapsed_s": st * 6.0, "lr": 1e-4 if st < 50000 else 5.3e-14})
        e = {"step": st, "eval_loss": last6.get(st, 100.0 - st / 1000.0), "eval_traj": 0.5,
             "eval_goal2s_err_m": 3.5, "eval_anchor_acc": 0.5, "eval_box3d_ap2m": 0.2, **_counters()}
        if outlier and st == 47500:
            e["eval_loss"] = 1.0e6            # a decoy OUTSIDE the last six: it must not move the band
        if st in last6:
            e["eval_box3d_ap2m"] = 0.2 + (0.001 if st % 1000 == 0 else 0.0)
        if drop_key:
            e.pop(drop_key, None)
        e["eval_tacv6_lat_ce"] = 0.4
        rows.append(e)
    rows.append({"step": rows[-1]["step"], "loss": 50.0, "elapsed_s": 1.0, "lr": 5.3e-14})
    d = tmp_path / "ref7"
    _write(d, rows, _cfg(50400, train_windows=746946, clips=4369))
    return d


def _run_dir(tmp_path, *, last_step=1500, drop_key=None, null_key=None, zero_key=None, tac_diff=False,
             perception_diff=False, eval_cache="/data/eval139", b1=True, gs=True, gs_drop=None, stderr=None,
             name="w8", steps=48852, guard_step=1000, evidence="ok", r8_term=True, aphigh=False):
    d = tmp_path / name
    rows = []
    for st in range(50, last_step + 1, 50):
        r = {"step": st, "loss": 60.0, "traj": 0.6, "elapsed_s": st * 9.0, "lr": 4.9e-05, "cuda_max_mem_gb": 31.0,
             "r8_emit_frac": 0.7852 if st <= 1000 else 1.0, "r8_total": 0.9, "r8_alloc_l1": 0.7, "r8_sat": 0.3,
             "r8_subscore": 0.4, "r8_cons": 0.03, "r8_v9_cons": 0.1, "r8_v9_mae_lat_c": 0.2,
             "r8_v9_mae_lon_c": 0.1, "r8_v9_mae_speed": 0.1}
        rows.append(r)
        if st % 500 == 0:
            e = {"step": st, "eval_loss": 99.0 - st / 100.0, "ema_eval_loss": 90.0, "eval_traj": 0.6,
                 "eval_goal2s_err_m": 4.0, "eval_anchor_acc": 0.6, "eval_box3d_ap2m": 0.1 if not aphigh else 0.3,
                 "ema_eval_box3d_ap2m": 0.19, "eval_tacv6_lat_ce": 0.1, **_counters(tac_diff, perception_diff)}
            if r8_term:
                e["eval_r8_total"] = 1.0
                e["ema_eval_r8_total"] = 1.0
            if drop_key:
                e.pop(drop_key, None)
            if null_key:
                e[null_key] = None
            if zero_key:
                e[zero_key] = 0.0
            rows.append(e)
        if st == 1000 and b1:
            rows.append({"step": 1000, "b1_guard": "keep", "b1_cut": False, "b1_ema_box3d_ap2m": 0.23,
                         "b1_ref_box3d_ap2m": 0.212, "b1_guard_step": 1000, "b1_head_peak_lr_before": None,
                         "b1_head_peak_lr_after": None})
        if st == 1000 and gs:
            g = {"step": 1000, "gs_n_terms": 3.0, "gs_n_passes": 4.0, "gs_measure_s": 555.0, "gs_fp32_replay": 1.0}
            for loc in ("trunk", "bev025", "bev_pool"):
                g.update({f"gs_{loc}_proj_map_hires": 0.001, f"gs_{loc}_norm_map_hires": 1.5,
                          f"gs_{loc}_proj_tac_v6": 0.002, f"gs_{loc}_norm_tac_v6": 0.4,
                          f"gs_{loc}_proj_rest": 0.997, f"gs_{loc}_norm_rest": 170.0,
                          f"gs_{loc}_norm_total": 170.1, f"gs_{loc}_lin_rel_err": 1e-6})
            if gs_drop:
                g.pop(gs_drop, None)
            rows.append(g)
    cfg = _cfg(steps, train_windows=781635, clips=4572, eval_cache=eval_cache,
               ema={"decay": 0.9998, "b1_guard": {"ref_ap2m": 0.212, "guard_step": guard_step,
                                                  "metric": "ema_eval_box3d_ap2m"}},
               extra=("--r8-alloc-emit-start", "1000", "--grad-share-at", "1000,16250,32500,48750",
                      "--w-r8-cons", "0.05", "--w-r8-listwise", "0.0"))
    _write(d, rows, cfg)
    if evidence == "ok":
        (d / "eval_set_evidence.json").write_text(json.dumps({
            "ev7": "[v3] held-out eval: 139 episodes -> 23772 windows, 8 fixed batches every 500 steps",
            "ev8": "[v3] held-out eval: 139 episodes -> 23772 windows, 8 fixed batches every 500 steps",
            "seed7": "g_ev = torch.Generator().manual_seed(12345)", "seed8": "g_ev = torch.Generator().manual_seed(12345)",
            "perm7": "perm = torch.randperm(len(e_ds), generator=g_ev)[:n_need].tolist()",
            "perm8": "perm = torch.randperm(len(e_ds), generator=g_ev)[:n_need].tolist()",
            "cache_mtime": "1790147635", "labels_mtime": "1790144389"}), encoding="utf-8")
    if stderr is not None:
        (d / "stderr_tail.log").write_text(stderr, encoding="utf-8")
        (d / "remote_state.json").write_text(json.dumps({
            "sup_pid": "101", "train_pid": "202", "sup_alive": "1", "train_alive": "1",
            "stderr_bytes": str(len(stderr.encode("utf-8"))), "ckpt": "1560403737 1792091977", "done": "0",
            "stopped": "0", "now": "1792092251", "run": RUN8, "arm": ARM8}), encoding="utf-8")
    return d


def _build(mod, d, ref):
    return mod.build(watch_dir=str(d), arm=ARM8, ref_dir=str(ref))


def _row(summary, key):
    rows = [r for r in summary["vs_refcv7"]["headline"] if r["key"] == key]
    assert len(rows) == 1, key
    return rows[0]


def _tr_for(page, key):
    """the table row of the headline table that carries `key` (the first one on the page)."""
    m = re.search(r'<tr><td>[^<]*(?:<span[^>]*>[^<]*</span>)?<br><code>%s</code>.*?</tr>' % re.escape(key), page, flags=re.S)
    assert m, key
    return m.group(0)


# ------------------------------------------------------------------ the page ----
def test_the_page_is_refcv8_self_contained_and_phone_and_theme_ready(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    assert "<title>refcv8 Training Watch</title>" in page and "<h1>refcv8 Training Watch</h1>" in page
    assert "refcv7 Training Watch" not in page
    assert re.findall(r'(?:src|href)="https?://', page) == []          # ONE self-contained file: no external fetch
    assert "fonts.googleapis" not in page
    assert 'name="viewport" content="width=device-width,initial-scale=1"' in page
    assert "prefers-color-scheme: dark" in page and "max-width:640px" in page
    assert 'id="vs-refcv7"' in page and 'id="refcv8-additions"' in page
    assert "No refcv8 NavSim or four-family battery milestone is read by this page" in page
    assert "BAR-R7-N1" not in page                     # the refcv7 NavSim copy is gone, not shown under refcv8
    assert s["step"] == 1500 and s["total"] == 48852 and s["arm"] == ARM8


def test_every_headline_metric_has_a_full_and_a_zoomed_figure_and_a_toggle(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, _ = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    for i in ("loss", "loss_ex", "traj", "goal2s", "anchor", "box_ap", "tac_lat"):
        assert f'data-chart="vs_{i}_full"' in page and f'data-chart="vs_{i}_zoom"' in page, i
    # a key neither fixture run carries (the map IoU) draws NOTHING and says so -- no empty axes, no zero line
    assert 'data-chart="vs_iou_driv_full"' not in page
    # 19 headline metrics, 7 present in the fixture (loss, loss_ex, traj, goal2s, anchor, box_ap, tac_lat) -> 12 x (full+zoom)
    assert page.count("UNAVAILABLE — neither run carries this key (nothing drawn).") == 24
    assert 'id="vs-toggle" hidden' in page and 'data-mode="zoom"' in page and "bar.hidden=false" in page


# ------------------------------------------------------------------ the reference band ----
def test_the_reference_band_is_computed_from_refcv7s_LAST_SIX_rows_and_not_refcv8s(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    r = _row(s, "eval_loss")
    assert r["refcv7_mean"] == pytest.approx(11.0, abs=1e-9)            # mean(10,12,10,12,10,12)
    assert r["refcv7_sd"] == pytest.approx(1.0954451150103321, abs=1e-9)   # sqrt(1.2), the n-1 estimator
    assert r["refcv7_n"] == 6
    assert r["refcv8_raw"] == pytest.approx(84.0, abs=1e-9)             # refcv8's own value (99 - 1500/100 = 84)
    assert r["refcv7_mean"] != r["refcv8_raw"]
    assert s["vs_refcv7"]["reference"]["last_step"] == 50400


def test_RED_pointing_the_reference_at_an_unfinished_record_REFUSES_instead_of_computing_a_band(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    d = _run_dir(tmp_path)
    page, s = _build(mod, d, d)                       # refcv8's OWN record as the 'reference': it ends at step 1,500
    assert s["vs_refcv7"]["state"] == "UNAVAILABLE"
    assert "is not final" in s["vs_refcv7"]["problem"] and "No comparison is drawn" in page
    assert 'data-chart="vs_loss_full"' not in page
    page2, s2 = _build(mod, _run_dir(tmp_path, name="w8b"), _ref_dir(tmp_path, last_step=50000))
    assert s2["vs_refcv7"]["state"] == "UNAVAILABLE" and "not 50,400" in s2["vs_refcv7"]["problem"]
    page3, s3 = _build(mod, _run_dir(tmp_path, name="w8c"), tmp_path / "no-such-ref")
    assert s3["vs_refcv7"]["state"] == "UNAVAILABLE" and "is not at" in s3["vs_refcv7"]["problem"]


# ------------------------------------------------------------------ UNAVAILABLE / undefined / a real zero ----
def test_a_key_missing_from_the_refcv8_log_renders_UNAVAILABLE_never_zero(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path, drop_key="eval_goal2s_err_m"), _ref_dir(tmp_path))
    r = _row(s, "eval_goal2s_err_m")
    assert r["refcv8_raw"] is None and r["refcv8_raw_state"] == "missing" and r["direction_raw"] == "UNAVAILABLE"
    assert r["delta_raw"] is None
    row = _tr_for(page, "eval_goal2s_err_m")
    assert "UNAVAILABLE" in row and "<b>0</b>" not in row and "<b>0.0</b>" not in row


def test_a_json_null_renders_undefined_and_a_real_zero_still_renders_as_zero(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path, null_key="eval_goal2s_err_m"), _ref_dir(tmp_path))
    r = _row(s, "eval_goal2s_err_m")
    assert r["refcv8_raw_state"] == "undef" and r["direction_raw"] == "undefined" and r["refcv8_raw"] is None
    assert "undefined" in _tr_for(page, "eval_goal2s_err_m")
    page0, s0 = _build(mod, _run_dir(tmp_path, zero_key="eval_goal2s_err_m", name="w8z"), _ref_dir(tmp_path))
    r0 = _row(s0, "eval_goal2s_err_m")
    assert r0["refcv8_raw_state"] == "value" and r0["refcv8_raw"] == 0.0 and r0["direction_raw"] == "better"
    assert "<b>0</b>" in _tr_for(page0, "eval_goal2s_err_m")          # the control: a measured 0 IS drawn


def test_a_key_missing_from_refcv7_reads_UNAVAILABLE_on_the_reference_side(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path, drop_key="eval_goal2s_err_m"))
    r = _row(s, "eval_goal2s_err_m")
    assert r["refcv7_mean"] is None and r["direction_raw"] == "no-reference"
    assert "UNAVAILABLE" in _tr_for(page, "eval_goal2s_err_m")


# ------------------------------------------------------------------ direction follows the sense ----
def test_direction_follows_each_metrics_sense(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    # refcv7 final levels: eval_loss 11 (lower is better), anchor 0.5, box3d AP@2m 0.2005; refcv8: 84, 0.6, 0.1
    assert _row(s, "eval_loss")["direction_raw"] == "worse"            # 84 > 11, lower is better
    assert _row(s, "eval_anchor_acc")["direction_raw"] == "better"     # 0.6 > 0.5, higher is better
    assert _row(s, "eval_box3d_ap2m")["direction_raw"] == "worse"      # 0.1 < 0.2005, higher is better
    assert _row(s, "eval_box3d_ap2m")["direction_ema"] == "worse"      # EMA 0.19 < 0.2005
    page2, s2 = _build(mod, _run_dir(tmp_path, name="w8d", aphigh=True), _ref_dir(tmp_path))
    assert _row(s2, "eval_box3d_ap2m")["direction_raw"] == "better"    # the flipped arm: 0.3 > 0.2005
    assert _row(s2, "eval_box3d_ap2m")["direction_ema"] == "worse"


def test_eval_loss_excluding_r8_is_derived_from_the_logged_terms(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    r = _row(s, "eval_loss_ex_r8")
    assert r["refcv8_raw"] == pytest.approx(83.0, abs=1e-9)            # eval_loss 84.0 - eval_r8_total 1.0
    assert r["refcv8_ema"] == pytest.approx(89.0, abs=1e-9)            # ema_eval_loss 90.0 - ema_eval_r8_total 1.0
    assert r["refcv7_mean"] == pytest.approx(11.0, abs=1e-9)           # refcv7 carries no r8 term: its eval_loss
    assert "NOT like-for-like" in page
    # a refcv8 log WITHOUT the r8 term: undefined, never eval_loss itself
    page2, s2 = _build(mod, _run_dir(tmp_path, name="w8e", r8_term=False), _ref_dir(tmp_path))
    r2 = _row(s2, "eval_loss_ex_r8")
    assert r2["refcv8_raw"] is None and r2["refcv8_raw_state"] == "missing" and r2["direction_raw"] == "UNAVAILABLE"


# ------------------------------------------------------------------ never hang on a flat series ----
def _ulps(v, n):
    import math
    for _ in range(n):
        v = math.nextafter(v, math.inf)
    return v


def _run_in_thread(fn, seconds=30):
    import threading
    box = {}

    def go():
        try:
            box["v"] = fn()
        except BaseException as e:                     # noqa: BLE001 -- reported by the assert below
            box["e"] = e
    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(seconds)
    assert not th.is_alive(), "the build did not finish: a flat series hung a chart"
    if "e" in box:
        raise box["e"]
    return box["v"]


def test_a_series_whose_spread_is_float_noise_builds_instead_of_hanging(tmp_path, monkeypatch):
    """The refcv7 builder's nice_ticks never advances when its step is below one ulp of the tick: a series that
    is constant up to 1e-17 hung it. A refcv8 page rebuilt on a schedule must not."""
    mod = _load8(monkeypatch)
    d = _run_dir(tmp_path)
    rows = [json.loads(ln) for ln in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    k = 0
    for r in rows:
        if "eval_loss" in r:
            r["eval_anchor_acc"] = _ulps(0.57812, k)      # 0.57812 then 1, 2, 3 ... ulps above it: a noise-sized spread
            r["ema_eval_anchor_acc"] = _ulps(0.57812, k)
            k += 1
        if "loss" in r and "eval_loss" not in r:
            r["r8_sat"] = _ulps(0.3, r["step"] // 50 % 4)      # a train series that is flat up to noise
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ref = _ref_dir(tmp_path)
    ev = [json.loads(ln) for ln in (ref / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in ev:
        if "eval_loss" in r:
            r["eval_anchor_acc"] = _ulps(0.57812, r["step"] // 500 % 3)
    (ref / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ev), encoding="utf-8")
    page, s = _run_in_thread(lambda: _build(mod, d, ref))
    assert 'data-chart="vs_anchor_full"' in page and 'data-chart="vs_anchor_zoom"' in page and 'data-chart="r8_alloc"' in page


# ------------------------------------------------------------------ the SAME-WINDOWS verdict ----
def test_the_same_windows_verdict_holds_when_every_input_is_the_same(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    w = s["vs_refcv7"]["windows"]
    assert w["same"] is True and w["failed"] == [] and w["verdict"] == "SAME EVAL WINDOWS — derived"
    assert w["n_counters_constant"] == 28 and w["n_counters_identical"] == 28      # 3 + 24 + 1 tactical
    assert "NOT a per-window-id dump" in w["proof_class"] and "SAME EVAL WINDOWS" in page
    assert w["label_changed_tactical"] == []


@pytest.mark.parametrize("arm_name, kw, ref_kw, failed_fragment", [
    ("cache", {"eval_cache": "/data/OTHER_eval"}, {}, "eval cache dir"),
    ("evidence", {"evidence": "absent"}, {}, "trainer log"),
    ("perception_counter", {"perception_diff": True}, {}, "label-derived counters"),
])
def test_RED_the_verdict_goes_INDICATIVE_ONLY_when_an_input_differs(tmp_path, monkeypatch, arm_name, kw, ref_kw,
                                                                    failed_fragment):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path, name="w8_" + arm_name, **kw), _ref_dir(tmp_path, **ref_kw))
    w = s["vs_refcv7"]["windows"]
    assert w["same"] is False, arm_name
    assert w["verdict"].startswith("INDICATIVE ONLY — ") and any(failed_fragment in f for f in w["failed"]), w["failed"]
    assert "INDICATIVE ONLY" in page and "SAME EVAL WINDOWS" not in page


def test_RED_a_different_window_count_goes_INDICATIVE_ONLY(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    d = _run_dir(tmp_path)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg["agent_join_stats"]["eval"]["n_windows"] = 23771
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    page, s = _build(mod, d, _ref_dir(tmp_path))
    assert s["vs_refcv7"]["windows"]["same"] is False
    assert any("eval-set windows" in f for f in s["vs_refcv7"]["windows"]["failed"])


def test_a_tactical_only_counter_difference_keeps_the_verdict_but_marks_the_tactical_metrics(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path, tac_diff=True), _ref_dir(tmp_path))
    w = s["vs_refcv7"]["windows"]
    assert w["same"] is True and w["label_changed_tactical"] == ["eval_tacv6_n_supervised_lat"]
    row = _tr_for(page, "eval_tacv6_lat_ce")
    assert "tactical label counters differ — INDICATIVE ONLY" in row
    assert "tactical label counters differ" not in _tr_for(page, "eval_traj")
    # the control: with no tactical difference the mark is absent from the table row too
    page2, s2 = _build(mod, _run_dir(tmp_path, name="w8f"), _ref_dir(tmp_path))
    assert "tactical label counters differ" not in _tr_for(page2, "eval_tacv6_lat_ce")


# ------------------------------------------------------------------ refcv8's own additions ----
def test_the_B1_guard_decision_row_is_read_and_a_missing_one_past_its_step_is_a_crit_tile(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    assert s["refcv8"]["b1_guard"] == {"decision": "keep", "step": 1000, "cut": False, "guard_step": 1000,
                                       "ref_ap2m": 0.212}
    assert "keep @ step 1,000" in page
    page2, s2 = _build(mod, _run_dir(tmp_path, name="w8g", b1=False), _ref_dir(tmp_path))
    assert "decision row MISSING" in page2 and s2["refcv8"]["b1_guard"]["decision"] is None
    page3, _ = _build(mod, _run_dir(tmp_path, name="w8h", b1=False, last_step=500), _ref_dir(tmp_path))
    assert "pending (decision at step 1,000)" in page3 and "decision row MISSING" not in page3


def test_the_gradient_share_readings_are_read_and_a_missing_key_is_UNAVAILABLE(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    assert s["refcv8"]["grad_share"] == {"planned": [1000, 16250, 32500, 48750], "done": [1000]}
    assert "<b>1,000</b> done" in page and "<b>16,250</b> pending" in page
    assert "<code>bev_pool</code>" in page
    page2, _ = _build(mod, _run_dir(tmp_path, name="w8i", gs_drop="gs_bev025_proj_tac_v6"), _ref_dir(tmp_path))
    assert page2.count('class="unav">UNAVAILABLE</span></td>') >= 1
    page3, _ = _build(mod, _run_dir(tmp_path, name="w8j", gs=False), _ref_dir(tmp_path))
    assert "No <code>gs_*</code> reading in the log yet" in page3


def test_the_allocation_emission_switch_is_read_from_the_log(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    assert s["refcv8"]["alloc_emit"]["start"] == "1000" and s["refcv8"]["alloc_emit"]["first_full_step"] == 1050
    assert "emit_frac 1.000 at step 1,500" in page


def test_the_one_diagnosed_stderr_line_is_diagnosed_and_a_new_instance_is_not(tmp_path, monkeypatch):
    mod = _load8(monkeypatch)
    page, s = _build(mod, _run_dir(tmp_path, stderr=DIAG_LINE + "\n"), _ref_dir(tmp_path))
    assert s["stderr_lines"] == 1 and s["stderr_undiagnosed"] == 0
    other = DIAG_LINE.replace("1077980", "1099999")                    # a relaunch carries a new pid: undiagnosed again
    page2, s2 = _build(mod, _run_dir(tmp_path, name="w8k", stderr=other + "\n"), _ref_dir(tmp_path))
    assert s2["stderr_undiagnosed"] == 1


# ------------------------------------------------------------------ the wrapper never disturbs refcv7 ----
def test_a_refcv8_build_leaves_a_separately_loaded_refcv7_builder_untouched(tmp_path, monkeypatch):
    b7 = _load7_separately(monkeypatch)
    before = (b7.ARM, b7.TOTAL_SPEC, b7.EVAL_PKG, b7.FACTS["stderr_diagnosed"], b7.navsim_html)
    mod = _load8(monkeypatch)
    _build(mod, _run_dir(tmp_path), _ref_dir(tmp_path))
    assert (b7.ARM, b7.TOTAL_SPEC, b7.FACTS["stderr_diagnosed"]) == ("refcv7-r101-s0", 50400, {})
    assert (b7.ARM, b7.TOTAL_SPEC, b7.EVAL_PKG, b7.FACTS["stderr_diagnosed"], b7.navsim_html) == before
    assert mod.load_refcv7() is not mod.load_refcv7()                       # always a FRESH instance
    page7, _s7 = b7.build(watch_dir=str(tmp_path / "ref7"), arm="refcv7-r101-s0")         # refcv7's own page still builds, as refcv7
    assert "<title>refcv7 Training Watch</title>" in page7 and "refcv8" not in page7.split("<h1>")[1].split("</h1>")[0]


# ------------------------------------------------------------------ fail loud, never skip ----
def test_a_page_without_a_hook_fails_loud(monkeypatch):
    mod = _load8(monkeypatch)
    with pytest.raises(SystemExit, match="ZZWATCH-HOOK-MISSING"):
        mod.inject("<html></html>", [("<title>refcv7 Training Watch</title>", "x", 1)])
    with pytest.raises(SystemExit, match="ZZWATCH-HOOK-MISSING"):
        mod.post_process("<title>refcv7 Training Watch</title>", "", "")


def test_another_arms_run_is_refused_before_any_pull(monkeypatch, tmp_path):
    mod = _load8(monkeypatch)
    pulls = []

    class Stub:                                                    # stands in for the refcv7 builder instance
        @staticmethod
        def resolve_run(arm, cli):
            return "/home/nvidia/refcv7_run/runs/refcv7-r101-s0", "$REFCV7_RUN"

        @staticmethod
        def pull(*a, **k):
            pulls.append(a)

    monkeypatch.setattr(mod, "load_refcv7", lambda: Stub)
    with pytest.raises(SystemExit, match="does not belong"):
        mod.main(["--dir", str(tmp_path / "x")])
    assert pulls == []


def test_the_pulled_run_dir_must_be_the_gated_argvs_or_the_constant(monkeypatch):
    mod = _load8(monkeypatch)
    assert mod.RUN8_DEFAULT == RUN8 and mod.TOTAL_SPEC == 48852 and mod.ARM == ARM8
    assert mod.REF_FINAL_STEP == 50400 and mod.REF_BAND_N == 6
