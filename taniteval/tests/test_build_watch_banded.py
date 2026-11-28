"""The refcv7 Training Watch renders the BANDED map head (refcv9 C1, `--r9-map-banded`) -- and its key SPELLING cannot drift.

MEASURED 2026-11-05 (commit 1d4ec82 taught the launch gate to read the banded head; the Watch was not updated): refcv9 E1 arms M1 and
M2a log `map_banded_*` keys, where the Watch read only `eval_map_hires_iou_<cls>_<band>` and `map_hires_{lshare,gno,n}_*` -- which are JSON
null (eval) or absent (train) under the banded head. An M1 / M2a run showed an empty map section and, worse, a green "40/40 keys ·
0-20 m above 0.05" chip over cells that were all undefined.

The fixture is M1's REAL Thor smoke rows (`stack/tests/data/launch_gate_banded_m1_smoke_rows.jsonl`: train steps 50 and 100, the in-run
eval at step 100 with `eval_` and `ema_eval_` banded keys), plus `elapsed_s` / `lr` on the train rows (the fixture was reduced to the keys
the GATE reads; the Watch's segment table needs a clock).

  S   SPELLING   the Watch's literals equal `tanitad.models.map_banded` (BAND_NAMES, CHANNELS, PROB_CHANNELS, L1_CHANNELS, BAND_WEIGHTS,
                 MAXENT_W, BAND_START_M, BAND_CELL_M, `telemetry_keys`) and the launch gate's (`BANDED_KEY_FMT`, `banded_train_keys`,
                 `banded_metric_keys`, the `eval_` / `ema_eval_` prefixes, the loss-sum tolerance). A rename in the head goes RED here.
                 The Watch stays stdlib-only: an AST check that it imports neither torch nor tanitad.
  R   RENDER     the real rows render the per-channel x band IoU / recall / precision, the eval loss terms, the per-band supervision, the
                 train-side per-band loss terms, maxent / n / unobs and the loss-sum identity -- every expected value DERIVED here from the
                 raw rows, never read back from the builder -- all labelled BANDED, with no 8-class thin-class tile.
  F   RED ARMS  every contract failure FIRES: a key absent, a null, a broken count identity, an empty band, a train row missing keys, a
                 loss sum that does not add up, an all-ignore row that is not exactly 0.0. An absent key is never drawn as 0.
  M   MUTATION   a mutated key prefix / channel / band spelling is DETECTED by the spelling check, and stops the Watch recognising the run.
  N   NON-BANDED a hires run carries no banded marker anywhere: no BANDED label, no `map_banded` summary key, its thin-class tile intact.
  L   LABELS     (E1-NEAR20) the band labels follow the RUN's near-band end, read from its own config.json (`map_hires.banded.near_m`, the
                 stamp `map_banded_flag.stamp` writes, or the `--r9-map-banded-near-m` argv flag; absent = 10 m): a 20 m run renders
                 "0-20 m" / "20-40 m", a default run "0-10 m" / "10-40 m", the tool's starts equal `map_banded.band_starts` and its spans the
                 BUILT `band_geometry`; a value outside (10, 20) is REFUSED; the old hard-coded label fails the 20 m check by mutation."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")                     # map_banded is torch code: the spelling pin needs it, loudly
from tanitad.models import map_banded as MB              # noqa: E402  (taniteval/conftest.py puts <repo>/stack on the path)
import launch_gate as LG                                 # noqa: E402  (... and <repo>/stack/scripts)
from tanitad.train import map_banded_flag as MBF         # noqa: E402  (the trainer's own writer of the near-band stamp)

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "taniteval" / "tools" / "training_watch" / "build_watch_refcv7.py"
FIX = ROOT / "stack" / "tests" / "data" / "launch_gate_banded_m1_smoke_rows.jsonl"
RIG = LG.PROFILES["refcv9-e1rig"]
ARM = "refcv9-e1-m1"
NO_PKG = str(Path(__file__).resolve().parent / "no-refcv7-eval-package-here")   # never created

# the contract, written out here from the gate's own docstring (30 loss + 9 per-band + 63 counts = 102 per prefix)
N_TRAIN, N_STAT, N_TOTAL = 39, 63, 102


# ------------------------------------------------------------------ loading ----
def _load(monkeypatch, src: str | None = None, name: str = "build_watch_banded_under_test"):
    for k in ("REFCV7_ARM", "REFCV7_RUN", "REFCV7_WATCH_DIR", "REFCV7_EVAL_LIVE", "REFCV7_NAVSIM_ARM"):
        monkeypatch.delenv(k, raising=False)
    assert not Path(NO_PKG).exists()
    monkeypatch.setenv("REFCV7_EVAL_PKG", NO_PKG)         # hermetic: no NavSim package, no battery
    if src is None:
        spec = importlib.util.spec_from_file_location(name, BUILDER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    mod = types.ModuleType(name)                           # a MUTATED copy of the source, executed as a module
    mod.__file__ = str(BUILDER)
    exec(compile(src, str(BUILDER), "exec"), mod.__dict__)
    return mod


def _fixture_rows() -> list[dict]:
    return [json.loads(l) for l in FIX.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_run(tmp_path, rows, name="run", config=None):
    """A metrics.jsonl the way the gate hands it to the Watch (`--metrics`): train rows get a clock. `config` = a config.json beside it."""
    d = tmp_path / name
    d.mkdir()
    out = []
    for r in rows:
        if "loss" in r and "eval_loss" not in r:
            r = {**r, "elapsed_s": r.get("elapsed_s", 6.5 * r["step"]), "lr": 1e-4}
        out.append(r)
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    if config is not None:
        (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return d


def _build(mod, d):
    return mod.build(metrics_path=str(d / "metrics.jsonl"), arm=ARM)


def _eval_row(rows):
    return [r for r in rows if "eval_loss" in r][-1]


def _drop(rows, *keys):
    return [{k: v for k, v in r.items() if k not in keys} for r in rows]


def _set(rows, **kv):
    """Set keys on the EVAL row only."""
    out = [dict(r) for r in rows]
    [r for r in out if "eval_loss" in r][-1].update(kv)
    return out


def _train_set(rows, step, **kv):
    out = [dict(r) for r in rows]
    for r in out:
        if "loss" in r and "eval_loss" not in r and r["step"] == step:
            r.update(kv)
    return out


def _section(page):
    a = page.index("<h2>Map BANDED")
    return page[a:page.index("<h2>Boxes", a)]


def _hm_cell(page, ch, band):
    m = re.search(r'<td class="hm[^"]*" data-cell="%s %s"[^>]*>(.*?)</td>' % (ch, band), page)
    assert m, (ch, band)
    return m.group(0)


def _payload(page, cid):
    m = re.search(r'<script type="application/json" data-series="%s">(.*?)</script>' % re.escape(cid), page, re.S)
    assert m, cid
    return json.loads(m.group(1))


# ------------------------------------------------------------------ S: spelling ----
def spelling_diffs(W) -> list[str]:
    """Every way the Watch module `W` disagrees with `map_banded.py` and the launch gate. [] = pinned."""
    out = []
    mine = W.banded_train_keys() + W.banded_stat_keys()
    theirs = list(MB.telemetry_keys(with_metrics=True))
    if set(mine) != set(theirs):
        out.append(f"keys: only in the Watch {sorted(set(mine) - set(theirs))[:3]}, only in map_banded {sorted(set(theirs) - set(mine))[:3]}")
    if set(W.banded_train_keys()) != set(MB.telemetry_keys(with_metrics=False)):
        out.append("the train keys differ from telemetry_keys(with_metrics=False)")
    gate = LG.banded_train_keys(MB) + LG.banded_metric_keys(MB)
    if mine != gate:
        out.append("key list or ORDER differs from the gate's banded_train_keys + banded_metric_keys")
    for attr, want in (("BANDED_BANDS", tuple(MB.BAND_NAMES)), ("BANDED_CHANNELS", tuple(MB.CHANNELS)),
                       ("BANDED_PROB_CHANNELS", tuple(MB.PROB_CHANNELS)), ("BANDED_L1_CHANNELS", tuple(MB.L1_CHANNELS)),
                       ("BANDED_BAND_WEIGHTS", tuple(MB.BAND_WEIGHTS)), ("BANDED_BAND_START_M", tuple(MB.BAND_START_M)),
                       ("BANDED_BAND_CELL_CM", tuple(round(c * 100) for c in MB.BAND_CELL_M)),
                       ("BANDED_STATS", tuple(LG.BANDED_STATS)), ("BANDED_MAXENT_W", MB.MAXENT_W)):
        if getattr(W, attr) != want:
            out.append(f"{attr}: Watch {getattr(W, attr)!r} != {want!r}")
    if dict(W.BANDED_KEY_FMT) != dict(LG.BANDED_KEY_FMT):
        out.append("BANDED_KEY_FMT differs from the gate's")
    if W.BANDED_SUM_REL_TOL != RIG["map_hires"]["lc_sum_rel_tol"]:
        out.append("BANDED_SUM_REL_TOL differs from the profile's lc_sum_rel_tol")
    if W.BANDED_SUM_SLACK != 0.5e-5 * (len(MB.CHANNELS) * len(MB.BAND_NAMES) + 1):
        out.append("BANDED_SUM_SLACK differs from the gate's rounding slack")
    return out


def test_the_watchs_banded_spelling_equals_map_banded_and_the_gate(monkeypatch):
    W = _load(monkeypatch)
    assert spelling_diffs(W) == []
    assert (len(W.banded_train_keys()), len(W.banded_stat_keys())) == (N_TRAIN, N_STAT)
    assert len(W.banded_train_keys() + W.banded_stat_keys()) == N_TOTAL == len(MB.telemetry_keys(with_metrics=True))
    assert len(set(W.banded_train_keys() + W.banded_stat_keys())) == N_TOTAL            # no duplicate
    # the sigmoid channels carry counts, the three regression channels do not
    assert set(W.BANDED_CHANNELS) - set(W.BANDED_PROB_CHANNELS) == set(W.BANDED_L1_CHANNELS)


def test_the_eval_prefixes_and_the_slack_are_the_gates_own_literals(monkeypatch):
    """The prefixes and the slack live inside the gate's function bodies, not in a constant: pin them by source needle."""
    W = _load(monkeypatch)
    src = inspect.getsource(LG.banded_eval_key_reasons)
    assert 'prefixes = ("eval_",) + (("ema_eval_",) if ema else ())' in src
    assert W.BANDED_EVAL_PREFIXES == ("eval_", "ema_eval_")
    assert "slack = 0.5e-5 * (n_loss + 1)" in inspect.getsource(LG.banded_train_row_reasons)


def test_the_watch_stays_stdlib_only():
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots, "no import found: the check read nothing"
    std = set(sys.stdlib_module_names) | {"__future__"}
    assert roots - std == set(), f"non-stdlib imports in the Watch: {sorted(roots - std)}"
    assert not roots & {"torch", "tanitad", "launch_gate", "numpy"}


# ------------------------------------------------------------------ R: the real rows render ----
def test_the_real_M1_rows_render_the_banded_series_labelled_BANDED(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    rows = _fixture_rows()
    page, s = _build(W, _write_run(tmp_path, rows))
    sec = _section(page)
    ev = _eval_row(rows)
    # labelled BANDED, in the heading, the chip, the tile, the chart titles and the evals table
    assert "<h2>Map BANDED (near / mid / far) — per channel and band</h2>" in page
    assert '<span class="chip info"><i></i>BANDED HEAD</span>' in sec
    assert 'id="banded-map-contract"' in page and "Banded map head (BANDED)" in page
    assert "map BANDED: 102/102 keys finite" in page and "map BANDED train: 39 keys per row" in page
    assert "<th class=\"num\">map BANDED loss</th>" in page
    # the 8-class tile, heatmap and per-class tables are NOT drawn, and no green 8-class chip claims anything
    assert 'id="thin-class-alarm"' not in page and "Thin-class alarm" not in page
    assert "map 10 cm: 40/40 keys" not in page and "THIN-CLASS ALARM" not in page
    assert "Per-class loss share" not in sec and "map signal: every class with cells is pulled" not in page
    # every eval series key spelling is map_banded's OWN (taken from telemetry_keys, not from the Watch)
    for k in MB.telemetry_keys(with_metrics=True):
        if k.startswith("map_banded_inter_") or k.startswith("map_banded_union_") or k.startswith("map_banded_pos_"):
            assert "eval_" + k in page, k
    # the IoU / recall / precision of every cell, derived HERE from the raw counts
    for c in MB.PROB_CHANNELS:
        for b in MB.BAND_NAMES:
            pos, inter, union = (ev[f"eval_map_banded_{st}_{c}_{b}"] for st in ("pos", "inter", "union"))
            cell = _hm_cell(page, c, b)
            iou = inter / union
            assert f">{iou:.3f}<small>" in cell, (c, b, iou, cell[:200])
            pred = union - pos + inter
            assert f"R {inter / pos:.2f} · P {inter / pred:.2f}" in cell, (c, b)
            assert f'data-key="eval_map_banded_pos_{c}_{b}; eval_map_banded_inter_{c}_{b}; eval_map_banded_union_{c}_{b}"' in cell
    # the EMA table is drawn too (the run keeps an EMA), from the ema_eval_ keys
    assert "EMA weights — eval at step 100 (102/102 keys finite)" in sec and "live weights — eval at step 100 (102/102 keys finite)" in sec
    assert 'data-key="ema_eval_map_banded_pos_drv_near; ema_eval_map_banded_inter_drv_near; ema_eval_map_banded_union_drv_near"' in sec
    # the per-band eval supervision, from the raw row
    for b in MB.BAND_NAMES:
        assert f'<td class="num">{ev[f"eval_map_banded_n_{b}"]:,.0f}</td><td class="num">{ev[f"eval_map_banded_unobs_frac_{b}"]:.4f}</td>' in sec, b
    # the 30 eval loss terms, one per channel x band
    for c in MB.CHANNELS:
        for b in MB.BAND_NAMES:
            assert f'<td class="num">{ev[f"eval_map_banded_loss_{c}_{b}"]:.4f}</td>' in sec, (c, b)
    # the train side: the per-band max-entropy / n / unobs of the LAST train row, and the loss-sum identity
    last = [r for r in rows if "loss" in r and "eval_loss" not in r][-1]
    for b in MB.BAND_NAMES:
        assert f'<td class="num">{last[f"map_banded_n_{b}"]:,.0f}</td><td class="num">{last[f"map_banded_unobs_frac_{b}"]:.4f}</td>' in sec, b
    assert "sum to the logged <code>map_hires</code> on 2 of 2 labelled train rows" in sec
    # the summary: the banded block is the reading, the 8-class block says NOT applicable (never "clear")
    assert s["map"]["head"] == "banded" and s["map"]["state"] == "NOT_APPLICABLE_BANDED" and s["map"]["alarm"] is False
    mbs = s["map_banded"]
    assert mbs["state"] == "clear" and mbs["alarm"] is False and mbs["ema_on"] is True and mbs["expected_per_prefix"] == 102
    assert mbs["present_latest"] == {"eval_": 102, "ema_eval_": 102} and mbs["rows_bad"] == []
    assert mbs["iou_latest"]["eval_"]["drv/near"] == pytest.approx(
        ev["eval_map_banded_inter_drv_near"] / ev["eval_map_banded_union_drv_near"])
    assert mbs["iou_latest"]["ema_eval_"]["lane/far"] == pytest.approx(
        ev["ema_eval_map_banded_inter_lane_far"] / ev["ema_eval_map_banded_union_lane_far"])
    t = mbs["train"]
    assert (t["state"], t["alarm"], t["n_rows"], t["n_data"], t["n_all_ignore"]) == ("clear", False, 2, 2, 0)
    assert t["sum_checked"] == 2 and t["sum_bad"] == [] and 0 < t["sum_max_abs"] < 1e-5
    shares = [t["bands"][b]["share"] for b in MB.BAND_NAMES]
    assert sum(shares) == pytest.approx(1.0) and all(x > 0 for x in shares)
    assert t["bands"]["far"]["weight"] == 0.5 and t["bands"]["near"]["n_last"] == last["map_banded_n_near"]


def test_the_charts_carry_the_eval_series_derived_from_the_counts(tmp_path, monkeypatch):
    """Seven per-channel charts (3 bands, live dots + EMA dashed) whose points are inter / union of each eval row."""
    W = _load(monkeypatch)
    base = _fixture_rows()
    trains = [r for r in base if "loss" in r and "eval_loss" not in r]
    ev0 = _eval_row(base)
    rows = []
    for i, step in enumerate((100, 200, 300)):
        for j, r in enumerate(trains):
            rows.append({**r, "step": step - 50 + 50 * j})
        e = dict(ev0, step=step)
        for k in list(e):                                      # a different union per eval, so the series is not flat
            if k.endswith("_near") and "_union_" in k:
                e[k] = e[k] * (1.0 + 0.1 * i)
        rows.append(e)
    page, s = _build(W, _write_run(tmp_path, rows))
    for c in MB.PROB_CHANNELS:
        assert f'data-chart="mb_{c}"' in page, c
        ser = {x["name"]: x for x in _payload(page, f"mb_{c}")["series"]}
        assert set(ser) == {"near", "near (EMA)", "mid", "mid (EMA)", "far", "far (EMA)"}, (c, set(ser))
        for b in MB.BAND_NAMES:
            want = [r[f"eval_map_banded_inter_{c}_{b}"] / r[f"eval_map_banded_union_{c}_{b}"] for r in rows if "eval_loss" in r]
            got = ser[b]["ys"]
            assert len(got) == 3 and got == pytest.approx([round(x, 4) for x in want], abs=1e-4), (c, b)
            assert ser[b]["key"] == f"eval_map_banded_inter_{c}_{b} / eval_map_banded_union_{c}_{b}"
            assert ser[f"{b} (EMA)"]["key"].startswith("ema_eval_map_banded_inter_")
    assert 'data-chart="mb_band_loss"' in page and 'data-chart="cmh"' in page
    assert "Map BANDED — total loss" in page and "Map (10 cm) — loss" not in page
    names = {x["name"] for x in _payload(page, "mb_band_loss")["series"]}
    assert names == {f"{b} {w}" for b in MB.BAND_NAMES for w in ("train (EMA)", "eval")}
    assert s["map_banded"]["n_evals"] == 3 and s["map_banded"]["state"] == "clear"


def test_the_gates_metrics_only_form_exits_0_and_writes_the_summary(tmp_path, monkeypatch, capsys):
    """`build_watch_refcv7.py --metrics M --out O` is how the launch gate builds the Watch from a smoke (G-MAP item 4)."""
    W = _load(monkeypatch)
    d = _write_run(tmp_path, _fixture_rows())
    out = tmp_path / "watch_refcv7.html"
    assert W.main(["--metrics", str(d / "metrics.jsonl"), "--out", str(out)]) == 0
    assert out.is_file() and "Banded map head (BANDED)" in out.read_text(encoding="utf-8")
    s = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert s["map_banded"]["state"] == "clear" and s["map"]["head"] == "banded"
    assert "ZZWATCH " in capsys.readouterr().out


# ------------------------------------------------------------------ F: RED arms ----
def test_RED_one_banded_eval_key_missing_is_UNAVAILABLE_and_never_drawn_as_0(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    page, s = _build(W, _write_run(tmp_path, _drop(_fixture_rows(), "eval_map_banded_inter_lane_near")))
    mb = s["map_banded"]
    assert mb["state"] == "UNAVAILABLE" and mb["alarm"] is True
    assert mb["missing_latest"]["eval_"] == ["eval_map_banded_inter_lane_near"] and mb["present_latest"]["eval_"] == 101
    assert mb["present_latest"]["ema_eval_"] == 102 and mb["rows_bad"] == [[100, "eval_", 1]]
    cell = _hm_cell(page, "lane", "near")
    assert ">UNAVAILABLE</td>" in cell and "0.000" not in cell and "data-key" not in cell
    assert "map BANDED: 1 keys UNAVAILABLE" in page and 'class="alarm crit" id="banded-map-contract"' in page
    assert "101/102 keys · UNAVAILABLE" in page


def test_RED_an_EMA_key_missing_is_read_off_the_EMA_prefix(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    page, s = _build(W, _write_run(tmp_path, _drop(_fixture_rows(), "ema_eval_map_banded_union_drv_far")))
    mb = s["map_banded"]
    assert mb["state"] == "UNAVAILABLE" and mb["missing_latest"]["ema_eval_"] == ["ema_eval_map_banded_union_drv_far"]
    assert mb["missing_latest"]["eval_"] == [] and mb["present_latest"] == {"eval_": 102, "ema_eval_": 101}
    assert ">UNAVAILABLE</td>" in re.search(r'<td class="hm[^"]*" data-cell="drv far"[^>]*>(.*?)</td>',
                                           page[page.index("EMA weights"):]).group(0)


def test_RED_a_null_or_non_finite_banded_key_is_BAD_not_a_value(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    page, s = _build(W, _write_run(tmp_path, _set(_fixture_rows(), eval_map_banded_loss_drv_near=None,
                                                  eval_map_banded_inter_sdw_mid=float("nan"))))
    mb = s["map_banded"]
    assert mb["state"] == "UNAVAILABLE" and mb["bad_latest"]["eval_"] == [
        "eval_map_banded_loss_drv_near", "eval_map_banded_inter_sdw_mid"]
    assert ">UNAVAILABLE</td>" in _hm_cell(page, "sdw", "mid") and "non-finite in" in _hm_cell(page, "sdw", "mid")
    assert '<td class="num muted">UNAVAILABLE</td>' in _section(page)            # the loss table cell too


def test_RED_a_broken_count_identity_is_an_ALARM(tmp_path, monkeypatch):
    """inter > union cannot happen: an IoU read from it would exceed 1."""
    W = _load(monkeypatch)
    page, s = _build(W, _write_run(tmp_path, _set(_fixture_rows(), eval_map_banded_inter_drv_near=200000.0)))
    mb = s["map_banded"]
    assert mb["state"] == "ALARM" and mb["alarm"] is True and mb["count_bad"]["eval_"] == [["drv", "near"]]
    assert "break the count identity inter ≤ pos ≤ union" in page and "holds on 20 of 21 cells (eval_)" in page


def test_an_empty_band_is_AMBER_not_a_failure(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    page, s = _build(W, _write_run(tmp_path, _set(_fixture_rows(), eval_map_banded_n_near=0.0)))
    mb = s["map_banded"]
    assert mb["state"] == "AMBER" and mb["alarm"] is False and mb["bands_without_cells"] == ["near"]
    assert 'class="alarm warn" id="banded-map-contract"' in page and "map BANDED: a band has no supervised cells" in page


def test_a_failed_EMA_eval_is_AMBER_and_named(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    page, s = _build(W, _write_run(tmp_path, _set(_fixture_rows(), ema_eval_error="RuntimeError: ema weights not built")))
    mb = s["map_banded"]
    assert mb["state"] == "AMBER" and mb["alarm"] is False and mb["bands_without_cells"] == []
    assert "the EMA in-run eval failed at step 100: RuntimeError: ema weights not built" in page
    assert "map BANDED: the EMA eval failed" in page


def test_an_empty_denominator_is_undefined_never_0(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    rows = _set(_fixture_rows(), eval_map_banded_pos_xwalk_near=0.0, eval_map_banded_inter_xwalk_near=0.0,
                eval_map_banded_union_xwalk_near=0.0)
    page, s = _build(W, _write_run(tmp_path, rows))
    cell = _hm_cell(page, "xwalk", "near")
    assert "undefined" in cell and "0.000" not in cell and "UNAVAILABLE" not in cell
    assert s["map_banded"]["state"] == "clear" and "xwalk/near" not in s["map_banded"]["iou_latest"]["eval_"]


def test_RED_a_train_row_missing_banded_keys_is_a_train_ALARM(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    rows = [{k: v for k, v in _r.items() if not (_r.get("step") == 100 and "eval_loss" not in _r and k == "map_banded_loss_lane_mid")}
            for _r in _fixture_rows()]
    page, s = _build(W, _write_run(tmp_path, rows))
    t = s["map_banded"]["train"]
    assert t["state"] == "ALARM" and t["alarm"] is True and t["rows_missing"] == [[100, 1]] and t["n_data"] == 1
    assert "MAP BANDED TRAIN: key or loss-sum contract broken" in page and "1 of 2 train rows lack banded keys" in page
    assert s["map_banded"]["state"] == "clear"                     # the eval contract is a separate reading


def test_RED_banded_loss_terms_that_do_not_sum_to_map_hires_are_an_ALARM(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    rows = _train_set(_fixture_rows(), 50, map_hires=24.5)         # the terms sum to 23.9116
    page, s = _build(W, _write_run(tmp_path, rows))
    t = s["map_banded"]["train"]
    assert t["state"] == "ALARM" and [x[0] for x in t["sum_bad"]] == [50] and t["sum_checked"] == 2
    assert t["sum_bad"][0][2] == 24.5 and t["sum_bad"][0][1] == pytest.approx(23.911628, abs=1e-5)
    assert "do not sum to <code>map_hires</code> on 1 train row(s)" in page


def test_an_all_ignore_train_row_logs_no_banded_keys_and_must_read_map_hires_exactly_0(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    rows = _fixture_rows()
    ig = {"step": 150, "loss": 3.0, "map_hires_n_labelled": 0.0, "map_hires": 0.0, "elapsed_s": 975.0, "lr": 1e-4}
    ok_rows = rows[:2] + [ig] + rows[2:]
    page, s = _build(W, _write_run(tmp_path, ok_rows, name="ok"))
    t = s["map_banded"]["train"]
    assert (t["state"], t["n_all_ignore"], t["rows_missing"], t["zero_bad"]) == ("clear", 1, [], [])   # exempt, not missing
    bad = rows[:2] + [{**ig, "map_hires": 0.7}] + rows[2:]
    page2, s2 = _build(W, _write_run(tmp_path, bad, name="bad"))
    t2 = s2["map_banded"]["train"]
    assert t2["state"] == "ALARM" and t2["zero_bad"] == [[150, 0.7]] and t2["rows_missing"] == []
    assert "all-ignore train row(s) do not read <code>map_hires</code> exactly 0.0" in page2


def test_before_the_first_eval_the_banded_section_renders_pending_from_the_train_rows(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    rows = [r for r in _fixture_rows() if "eval_loss" not in r]
    page, s = _build(W, _write_run(tmp_path, rows))
    assert s["map_banded"]["state"] == "pending" and s["map_banded"]["alarm"] is False
    assert "No in-run eval row yet: the banded eval series cannot be read." in page
    assert "map BANDED: no eval yet" in page and "BANDED training signal per band" in page
    assert s["map_banded"]["train"]["state"] == "clear" and s["map_banded"]["train"]["n_data"] == 2


# ------------------------------------------------------------------ M: mutation ----
def _mutated(monkeypatch, old, new):
    src = BUILDER.read_text(encoding="utf-8")
    assert src.count(old) == 1, (src.count(old), old)
    return _load(monkeypatch, src=src.replace(old, new), name="build_watch_banded_mutant")


@pytest.mark.parametrize("old,new", [
    ('BANDED_FAMILY = "map_banded_"', 'BANDED_FAMILY = "map_bandd_"'),                       # the prefix
    ('BANDED_BANDS = ("near", "mid", "far")', 'BANDED_BANDS = ("near", "middle", "far")'),    # a band
    ('"xwalk", "arrow", "hatch")      # sigmoid', '"crosswalk", "arrow", "hatch")      # sigmoid'),   # a channel
    ('BANDED_BAND_WEIGHTS = (1.0, 1.0, 0.5)', 'BANDED_BAND_WEIGHTS = (1.0, 1.0, 1.0)'),      # a weight
    ('"unobs": BANDED_FAMILY + "unobs_frac_{band}"', '"unobs": BANDED_FAMILY + "unobserved_{band}"'),   # a per-band key
])
def test_MUTATION_a_drifted_spelling_is_caught_by_the_pin(monkeypatch, old, new):
    """The deliberate-regression arm of the pin: each mutation must make `spelling_diffs` non-empty."""
    assert spelling_diffs(_load(monkeypatch)) == []
    assert spelling_diffs(_mutated(monkeypatch, old, new)) != []


def test_MUTATION_a_wrong_prefix_stops_the_Watch_recognising_a_banded_run(monkeypatch):
    rows = _fixture_rows()
    assert _load(monkeypatch).banded_run(rows) is True
    assert _mutated(monkeypatch, 'BANDED_FAMILY = "map_banded_"', 'BANDED_FAMILY = "map_bandd_"').banded_run(rows) is False


# ------------------------------------------------------------------ L: the labels follow the run's near-band end ----
TRAINER = ROOT / "stack" / "scripts" / "refc_v3_train.py"
ARGV_BASE = ["--arm", ARM, "--map-hires", "on", "--w-map-hires", "1.0", "--r9-map-banded"]


def _stamped(near_m):
    """The REAL `config.json["map_hires"]["banded"]` block: `map_banded_flag.stamp` (the trainer's own writer) on a built head."""
    head = MB.BandedMapHead(100.0, 30.0, d_up=8, groups=2, near_m=near_m)
    return MBF.stamp(SimpleNamespace(banded=head, refine=SimpleNamespace(cls=None)))


def _cfg(near_m):
    """A banded run's config.json whose near band ends at `near_m`: the stamp, and the argv flags only when the run passed them."""
    argv = list(ARGV_BASE)
    if near_m != MB.NEAR_M_DEFAULT:
        argv += ["--map-hires-near-lift-m", f"{near_m:g}", "--r9-map-banded-near-m", f"{near_m:g}"]
    return {"arm": ARM, "argv": argv, "map_hires": {"banded": _stamped(near_m)}}


def _want_labels(near_m) -> dict:
    """The label of each band, DERIVED from the BUILT geometry (`band_geometry`), never from the Watch."""
    out = {}
    for g in MB.band_geometry(100.0, 30.0, near_m):
        span = f"{g.x0:g}–{g.x1:g} m" if g.name != "far" else f"≥ {g.x0:g} m"
        out[g.name] = f"{g.name} · {span} · {round(g.cell * 100)} cm"
    return out


def labels_diffs(W, tmp_path, near_m, name="lab") -> list[str]:
    """Every way the Watch module `W` fails to label a banded run whose near band ends at `near_m`. [] = labelled by the run's own end."""
    page, s = _build(W, _write_run(tmp_path, _fixture_rows(), name=name, config=_cfg(near_m)))
    sec = _section(page)
    want = _want_labels(near_m)
    out = []
    for b, lab in want.items():
        for pat, n in ((f'<th class="num">{lab}</th>', 4), (f'<td>{lab}</td>', 2)):     # th: heatmap x2 (live, EMA) + loss + train-channel; td: 2 band tables
            if sec.count(pat) != n:
                out.append(f"{pat!r}: {sec.count(pat)} site(s), want {n}")
    if " · ".join(want.values()) not in sec:
        out.append("the section intro does not name the three bands as " + " · ".join(want.values()))
    other = _want_labels([a for a in MB.NEAR_M_ADMITTED if a != near_m][0])
    for b in ("near", "mid"):
        if other[b] in page:
            out.append(f"the OTHER layout's label {other[b]!r} is on the page")
    span = want["near"].split(" · ")[1]
    for who in ("drivable", "lane"):
        if f"<span>map BANDED {who} IoU, near {span}</span>" not in page:
            out.append(f"the {who} tile does not say near {span}")
    if (s["map_banded"]["near_m"], s["map_banded"]["near_m_source"].split(" (")[0]) != (
            float(near_m), "config.json map_hires.banded"):
        out.append(f"the summary reads {s['map_banded']['near_m']!r} from {s['map_banded']['near_m_source']!r}")
    return out


@pytest.mark.parametrize("near_m", MB.NEAR_M_ADMITTED)
def test_the_band_labels_follow_the_runs_near_band_end(tmp_path, monkeypatch, near_m):
    """(a) near 20 m renders 0–20 m / 20–40 m everywhere a band is named; the default 10 m renders 0–10 m / 10–40 m."""
    assert labels_diffs(_load(monkeypatch), tmp_path, near_m) == []


def test_the_labels_read_exactly_the_strings_the_PI_asked_for(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    page20, s20 = _build(W, _write_run(tmp_path, _fixture_rows(), name="n20", config=_cfg(20.0)))
    page10, s10 = _build(W, _write_run(tmp_path, _fixture_rows(), name="n10", config=_cfg(10.0)))
    for page, near, mid in ((page20, "0–20 m", "20–40 m"), (page10, "0–10 m", "10–40 m")):
        assert f"near · {near} · 10 cm" in page and f"mid · {mid} · 25 cm" in page and "far · ≥ 40 m · 50 cm" in page
    assert "near · 0–10 m" not in page20 and "mid · 10–40 m" not in page20
    assert "near · 0–20 m" not in page10 and "mid · 20–40 m" not in page10
    assert (s20["map_banded"]["near_m"], s10["map_banded"]["near_m"]) == (20.0, 10.0)


def test_an_older_run_with_no_near_band_end_in_its_config_keeps_the_10_m_labels(tmp_path, monkeypatch):
    """No config.json, or one with no banded stamp and no flag: the default 10 m -- the page the tool always drew, and said so."""
    W = _load(monkeypatch)
    for name, cfg in (("nocfg", None), ("oldcfg", {"arm": ARM, "argv": ARGV_BASE})):
        page, s = _build(W, _write_run(tmp_path, _fixture_rows(), name=name, config=cfg))
        assert "near · 0–10 m · 10 cm" in page and "mid · 10–40 m · 25 cm" in page and "near · 0–20 m" not in page
        assert s["map_banded"]["near_m"] == 10.0 and s["map_banded"]["near_m_source"].startswith("default 10 m")


def test_the_near_end_is_read_from_the_stamp_the_trainer_writes_and_from_the_flag(monkeypatch):
    W = _load(monkeypatch)
    assert "near_m" not in _stamped(10.0)                              # map_banded_flag.stamp omits it at the default ...
    assert _stamped(20.0)["near_m"] == 20.0                            # ... and writes it otherwise: the key the Watch reads
    assert W.banded_near_read(_cfg(20.0))[0] == 20.0 and W.banded_near_read(_cfg(10.0))[0] == 10.0
    # a config with only the flag (no stamp yet) is read from argv
    assert W.banded_near_read({"argv": ARGV_BASE + ["--r9-map-banded-near-m", "20"]}) == (20.0, "config.json argv --r9-map-banded-near-m")
    assert W.banded_near_read(None)[0] == 10.0 and W.banded_near_read({})[0] == 10.0


def test_the_flag_name_and_the_stamp_key_are_the_trainers_own(monkeypatch):
    W = _load(monkeypatch)
    assert W.BANDED_NEAR_M_FLAG == MBF.NEAR_M_FLAG == "--r9-map-banded-near-m"
    src = TRAINER.read_text(encoding="utf-8")
    assert f'add_argument("{MBF.NEAR_M_FLAG}", type=float, default=None, dest="{MBF.NEAR_M_DEST}"' in src      # the parser declares it
    assert 'map_hires_stamp["banded"] = _mbf.stamp(model._map_hires)' in src                                    # ... and the stamp is written
    assert '"argv": sys.argv[1:]' in src                                                                        # ... beside the raw argv


# (b) the tool's layout equals map_banded's, for both admitted values
def test_the_watchs_near_band_constants_and_starts_equal_map_banded(monkeypatch):
    W = _load(monkeypatch)
    assert W.BANDED_NEAR_M_DEFAULT == MB.NEAR_M_DEFAULT and tuple(W.BANDED_NEAR_M_ADMITTED) == tuple(MB.NEAR_M_ADMITTED)
    assert W.banded_band_starts() == tuple(MB.band_starts()) == tuple(W.BANDED_BAND_START_M)
    for near_m in MB.NEAR_M_ADMITTED:
        assert W.banded_band_starts(near_m) == tuple(MB.band_starts(near_m)), near_m
        assert W.banded_band_starts(str(near_m)) == tuple(MB.band_starts(str(near_m)))          # an argv token is a string
        for g in MB.band_geometry(100.0, 30.0, near_m):                                         # ... and the spans are the BUILT bands
            want = f"{g.x0:g}–{g.x1:g} m" if g.name != "far" else f"≥ {g.x0:g} m"
            assert W.banded_band_span(g.name, near_m) == want, (near_m, g.name)
            assert W.banded_band_label(g.name, near_m) == _want_labels(near_m)[g.name]


@pytest.mark.parametrize("bad", [0, 5, 15, 30, 10.5, "x", None, float("nan"), float("inf"), True])
def test_RED_a_near_end_outside_10_20_is_refused_exactly_where_map_banded_refuses(monkeypatch, bad):
    W = _load(monkeypatch)
    with pytest.raises(ValueError):
        MB.check_near_m(bad)
    with pytest.raises(ValueError):
        W.banded_check_near_m(bad)
    with pytest.raises(ValueError):
        W.banded_band_starts(bad)
    with pytest.raises(ValueError):
        W.banded_band_label("near", bad)


def test_RED_a_config_with_a_bad_or_self_contradicting_near_end_stops_the_Watch(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    stamp_bad = _cfg(20.0)
    stamp_bad["map_hires"]["banded"]["near_m"] = 15.0
    flag_bad = {"argv": ARGV_BASE + ["--r9-map-banded-near-m", "30"]}
    null_stamp = _cfg(20.0)
    null_stamp["map_hires"]["banded"]["near_m"] = None
    declared20_built10 = {"argv": ARGV_BASE + ["--r9-map-banded-near-m", "20"], "map_hires": {"banded": _stamped(10.0)}}
    declared10_built20 = {"argv": ARGV_BASE, "map_hires": {"banded": _stamped(20.0)}}
    for i, cfg in enumerate((stamp_bad, flag_bad, null_stamp, declared20_built10, declared10_built20)):
        with pytest.raises(ValueError, match=r"\[watch\]"):
            W.banded_near_read(cfg)
        with pytest.raises(ValueError, match=r"\[watch\]"):                      # ... and the page build refuses, never relabels
            _build(W, _write_run(tmp_path, _fixture_rows(), name=f"bad{i}", config=cfg))


def test_a_non_banded_run_never_reads_the_near_end(tmp_path, monkeypatch):
    """A hires run whose config.json holds nonsense in the banded slot is not refused: the banded read is behind `banded_run`."""
    W = _load(monkeypatch)
    d, arm = _hires_run(tmp_path, monkeypatch)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").is_file() else {}
    cfg.setdefault("map_hires", {})["banded"] = {"near_m": 15.0}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    page, s = W.build(watch_dir=str(d), arm=arm)
    assert "map_banded" not in s and "BANDED" not in page


# (c) the deliberate-regression arm: the old hard-coded label must FAIL the near 20 m check
def test_MUTATION_the_old_hard_coded_label_fails_the_near_20_m_check(tmp_path, monkeypatch):
    assert labels_diffs(_load(monkeypatch), tmp_path, 20.0, name="real") == []
    old_label = _mutated(monkeypatch, "    st = banded_band_starts(near_m)\n", "    st = BANDED_BAND_START_M\n")
    d = labels_diffs(old_label, tmp_path, 20.0, name="mut_label")
    assert d, "the hard-coded 0-10 m / 10-40 m label was NOT detected on a near 20 m run"
    assert any("near · 0–20 m · 10 cm" in x for x in d) and any("near · 0–10 m" in x for x in d), d
    assert labels_diffs(old_label, tmp_path, 10.0, name="mut_label10") == []     # on a default run the old label is the same page: the arm is real


def test_MUTATION_a_Watch_that_never_reads_the_config_fails_the_near_20_m_check(tmp_path, monkeypatch):
    never_reads = _mutated(monkeypatch, "near_m, near_src = banded_near_read(cfg) if banded else (None, None)",
                           'near_m, near_src = (BANDED_NEAR_M_DEFAULT, "mutant") if banded else (None, None)')
    assert labels_diffs(never_reads, tmp_path, 20.0, name="mut_cfg") != []


def test_MUTATION_a_drifted_admitted_set_or_default_is_caught_by_the_pin(monkeypatch):
    for old, new in (("BANDED_NEAR_M_ADMITTED = (10.0, 20.0)", "BANDED_NEAR_M_ADMITTED = (10.0, 20.0, 30.0)"),
                     ("BANDED_NEAR_M_DEFAULT = 10.0", "BANDED_NEAR_M_DEFAULT = 20.0")):
        m = _mutated(monkeypatch, old, new)
        assert (tuple(m.BANDED_NEAR_M_ADMITTED) != tuple(MB.NEAR_M_ADMITTED) or m.BANDED_NEAR_M_DEFAULT != MB.NEAR_M_DEFAULT)


# ------------------------------------------------------------------ N: a non-banded run is untouched ----
def _hires_run(tmp_path, monkeypatch):
    """The refcv7 suite's own synthetic hires run (a sibling test module's helper, imported, not copied)."""
    spec = importlib.util.spec_from_file_location("t7_helpers_for_banded", ROOT / "taniteval" / "tests" / "test_build_watch_refcv7.py")
    t7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t7)
    return t7._run_dir(tmp_path, last_step=1500), t7.ARM


def test_a_hires_run_carries_no_banded_marker_and_keeps_its_thin_class_tile(tmp_path, monkeypatch):
    W = _load(monkeypatch)
    d, arm = _hires_run(tmp_path, monkeypatch)
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert W.banded_run(rows) is False
    page, s = W.build(watch_dir=str(d), arm=arm)
    assert "BANDED" not in page and "map_banded" not in page and 'id="banded-map-contract"' not in page
    assert "map_banded" not in s and "head" not in s["map"]
    assert 'id="thin-class-alarm"' in page and "Thin-class alarm" in page and "<h2>Map (10 cm) — per class and band</h2>" in page
    assert s["map"]["state"] == "clear" and s["map"]["present_latest"] == 40
