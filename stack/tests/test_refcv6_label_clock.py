"""⛔ The v7/v8 LABEL CLOCK: a window's NOW must be read on the label's RAW timeline.

MEASURED 2026-09-26 on the live refcv6 run (A16 audit,
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv6-frozen-trunk-audit/`):
`V3Dataset` read labels at `(t + w - 1) * 0.1` on a PROVIDER row. The provider drops the first
`n_stack - 1` raw frames, a cache row is 0.1006666 s (median of 4,357 clips, recovered from the
100 Hz egomotion log), and the camera grid starts +0.113 s after the log's origin -- so the "8.0 s"
row was truly at 8.369 s and 10.6 % of the tactical-supervised windows sat outside the ±2 s band.
The fix: `t_now = grid_start_s + (t + w - 1 + n_stack - 1) * dt_s` (`tanitad/data/clip_clock.py`).

KNOWN-VALUE CLIPS. Each synthetic clip has an exact clock: a constant-speed track sampled at
`DT` per row, a 9-channel (n_stack 3) frame stack, and a label anchored at `t0_s` with the
shipped tactical band (2, 6) s -> admission |t_now - t0| <= 2 s. Every admitted-row set below is a
LITERAL worked by hand from those numbers (the arithmetic is in each comment), never from the code
under test. The DELIBERATE-REGRESSION arm flips the dataset's own `legacy_label_clock` switch --
the real function with a flag, not a restatement -- and must go RED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T  # noqa: E402
from tanitad.data import clip_clock as cc  # noqa: E402
from tanitad.data import v7_labels as v7l  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

DT = 0.100667          # s per row (the corpus median, MEASURED)
G0 = 0.113             # s, camera grid start after the log origin (the corpus median, MEASURED)
ROWS = 130
W = 8                  # refcv6's window (config.json `ego_history.steps`)


def _episode(clip: str, *, dt: float = DT, v: float = 10.0, n_ch: int = 9):
    r = torch.arange(ROWS, dtype=torch.float64)
    x = v * dt * r
    poses = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x),
                         torch.full_like(x, v)], dim=1).to(torch.float32)
    return SimpleNamespace(frames=torch.zeros(ROWS, n_ch, 8, 8, dtype=torch.uint8),
                           actions=torch.zeros(ROWS, 2), poses=poses,
                           episode_id=stable_episode_id(clip))


def _label(clip: str, t0: float = 8.0):
    return v7l.V7Label(clip_id=clip, tac_lat="LANE_KEEP", tac_lon="CRUISE",
                       str_action="HOLD_MAIN_ROAD", str_goal="FOLLOW_ROUTE", tac_anchor=None,
                       bands={"operative_s": [0.0, 2.0], "tactical_s": [2.0, 6.0],
                              "strategic_s": [8.0, 30.0]},
                       t0_s=t0, horizon={})


def _dataset(ep, lab):
    ds = T.V3Dataset([ep], window=W, max_horizon=20, channels=int(ep.frames.shape[1]))
    ds.v7_by_sid = {int(ep.episode_id): lab}
    ds.v7_dt = 0.1
    return ds


def _admitted(ds) -> list[int]:
    """NOW rows whose `lat_v7` is a real class -- read through the dataset's own __getitem__."""
    out = []
    for i in range(len(ds)):
        _e, t = ds.index[i]
        if int(ds[i]["lat_v7"]) != v7l.IGNORE_ID:
            out.append(t + W - 1)
    return out


def _sidecar(tmp: Path, ep, g0=G0, dt=DT) -> Path:
    p = tmp / "clock.jsonl"
    p.write_text(json.dumps({"sid": int(ep.episode_id), "grid_start_s": g0, "dt_s": dt}) + "\n",
                 encoding="utf-8")
    return p


#: sidecar clock: |0.113 + (r + 2) * 0.100667 - 8| <= 2  <=>  r + 2 in [58.48, 98.21]
#: <=> r in [57, 96]   (r = 56 -> 5.9517 s out; r = 97 -> 10.0790 s out)
TRUE_BAND_SIDECAR = list(range(57, 97))
#: no sidecar: dt from the poses (== DT exactly on this track), grid_start 0.0:
#: (r + 2) * 0.100667 in [6, 10]  <=>  r + 2 in [59.60, 99.34]  <=>  r in [58, 97]
TRUE_BAND_POSE_DT = list(range(58, 98))


def _assert_true_band(ds, expected) -> None:
    got = _admitted(ds)
    assert got == expected, (f"the 8.0 s label admitted NOW rows {got[:3]}..{got[-3:]} "
                             f"(n={len(got)}), expected {expected[0]}..{expected[-1]} "
                             f"(n={len(expected)})")


def test_the_8s_label_lands_on_the_TRUE_band_with_a_clock_sidecar(tmp_path) -> None:
    ep = _episode("clip-known-clock")
    ds = _dataset(ep, _label("clip-known-clock"))
    rep = ds.enable_clip_clock(str(_sidecar(tmp_path, ep)))
    assert rep["n_from_sidecar"] == 1 and rep["raw_offsets"] == [2]
    _assert_true_band(ds, TRUE_BAND_SIDECAR)


def test_without_a_sidecar_dt_comes_from_the_POSES_and_the_fallback_is_COUNTED() -> None:
    ep = _episode("clip-pose-clock")
    ds = _dataset(ep, _label("clip-pose-clock"))
    rep = ds.enable_clip_clock(None)
    assert rep["n_pose_dt_grid_start_0"] == 1 and rep["n_from_sidecar"] == 0
    assert abs(rep["dt_s_median"] - DT) < 1e-6
    _assert_true_band(ds, TRUE_BAND_POSE_DT)


def test_a_STATIONARY_clip_falls_back_to_the_nominal_step_and_says_so() -> None:
    """v = 0: no displacement, no identity -> 0.1 s, counted. t0 8.05 keeps every boundary off an
    exact float: (r + 2) * 0.1 in [6.05, 10.05]  <=>  r in [59, 98]."""
    ep = _episode("clip-stationary", v=0.0)
    ds = _dataset(ep, _label("clip-stationary", t0=8.05))
    rep = ds.enable_clip_clock(None)
    assert rep["n_nominal_dt_grid_start_0"] == 1
    _assert_true_band(ds, list(range(59, 99)))


def test_DELIBERATE_REGRESSION_the_old_formula_goes_RED(tmp_path) -> None:
    """`legacy_label_clock` restores `(t + w - 1) * 0.1` in the REAL `_now_s`. Row 57 (truly
    6.052 s, IN band) reads 5.7 s and is dropped; row 99 (truly 10.280 s, OUT) reads 9.9 s and is
    admitted -- so the true-band assertion must fail."""
    ep = _episode("clip-regression")
    ds = _dataset(ep, _label("clip-regression"))
    ds.enable_clip_clock(str(_sidecar(tmp_path, ep)))
    ds.legacy_label_clock = True
    got = _admitted(ds)
    assert 57 not in got and 99 in got
    with pytest.raises(AssertionError):
        _assert_true_band(ds, TRUE_BAND_SIDECAR)


def test_pose_dt_is_an_IDENTITY_and_refuses_a_doubled_cadence() -> None:
    """Analytic target: a constant-speed track sampled at a KNOWN step returns it; the same track
    at stride 2 (the advisory's class-D row 2: a 0.2 s displacement divided as 0.1 s) leaves the
    physical band and is refused (None), never returned as a clock."""
    for dt in (0.1, 0.100667, 0.1005):
        got = cc.pose_dt(_episode("a", dt=dt).poses)
        assert got is not None and abs(got - dt) < 1e-6, (dt, got)
    assert cc.pose_dt(_episode("b").poses[::2]) is None


def test_the_sidecar_reader_REFUSES_what_it_cannot_trust(tmp_path) -> None:
    def w(name, rows):
        p = tmp_path / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return str(p)
    for bad in (w("zero.jsonl", []),
                w("dt.jsonl", [{"sid": 1, "grid_start_s": 0.1, "dt_s": 0.2}]),
                w("us.jsonl", [{"sid": 1, "grid_start_s": 113000.0, "dt_s": 0.1}]),
                w("dup.jsonl", [{"sid": 1, "grid_start_s": 0.1, "dt_s": 0.1},
                                {"sid": 1, "grid_start_s": 0.2, "dt_s": 0.1}]),
                w("nosid.jsonl", [{"grid_start_s": 0.1, "dt_s": 0.1}])):
        with pytest.raises(cc.ClipClockError):
            cc.read_clip_clock_sidecar(bad)
    ep = _episode("clip-covered-by-nothing")
    ds = _dataset(ep, _label("clip-covered-by-nothing"))
    other = w("other.jsonl", [{"sid": 12345, "grid_start_s": 0.1, "dt_s": 0.1}])
    with pytest.raises(SystemExit):
        ds.enable_clip_clock(other)
