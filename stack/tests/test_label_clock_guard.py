"""⛔⛔ G3 / G-CLOCK — every label read lands within 0.05 s of the clip's MEASURED clock.

SPEC_REFCV7 FIX-2: the true per-clip clock landed in 82c2331 (D-REFCV6-LABEL-CLOCK: the trainer
read labels 0.369 s early at the median), but NOTHING checked it -- the A16 audit's Q6 G3 found
no guard (`…/2026-09-26-refcv6-frozen-trunk-audit/code/q6_guards.py`). The guard is now
`V3Dataset.assert_label_clock_true`, called by `train()` right after `enable_clip_clock` on the
train AND the eval split, before any batch.

KNOWN-VALUE CLIPS (the fixture of `test_refcv6_label_clock.py`): a constant-speed track sampled at
`DT` per row, a 9-channel (n_stack 3) stack, a sidecar clock `(G0, DT)`. The expected times are
LITERALS worked from those numbers. The DELIBERATE-REGRESSION arm flips the dataset's own
`legacy_label_clock` switch -- the historical `(t + w - 1) * 0.1` -- and must go RED.
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
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

DT = 0.100667          # s per row (the corpus median, MEASURED)
G0 = 0.113             # s, grid start after the log origin (the corpus median, MEASURED)
ROWS = 130
W = 8


def _episode(clip: str, v: float = 10.0):
    r = torch.arange(ROWS, dtype=torch.float64)
    x = v * DT * r
    poses = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x),
                         torch.full_like(x, v)], dim=1).to(torch.float32)
    return SimpleNamespace(frames=torch.zeros(ROWS, 9, 8, 8, dtype=torch.uint8),
                           actions=torch.zeros(ROWS, 2), poses=poses,
                           episode_id=stable_episode_id(clip))


def _dataset(eps):
    ds = T.V3Dataset(eps, window=W, max_horizon=20, channels=9)
    ds.v7_by_sid, ds.v7_dt = {}, 0.1
    return ds


def _sidecar(tmp: Path, eps) -> str:
    p = tmp / "clock.jsonl"
    p.write_text("".join(json.dumps({"sid": int(e.episode_id), "grid_start_s": G0, "dt_s": DT})
                         + "\n" for e in eps), encoding="utf-8")
    return str(p)


def test_the_TRUE_clock_PASSES_with_its_numbers(tmp_path):
    eps = [_episode("clip-a"), _episode("clip-b")]
    ds = _dataset(eps)
    side = _sidecar(tmp_path, eps)
    ds.enable_clip_clock(side)
    rep = ds.assert_label_clock_true(side, split="train")
    assert rep["g3"] == "PASS" and rep["tol_s"] == 0.05
    assert rep["n_clips"] == 2 and rep["n_unverified_clips"] == 0
    assert rep["n_reads_checked"] == 6                 # 3 rows x 2 clips
    assert rep["worst_abs_err_s"] < 1e-9
    assert rep["max_unverified_frac"] == 0.01 and rep["max_unverified_frac_source"] == "default"


def test_DELIBERATE_REGRESSION_the_historical_mapping_goes_RED(tmp_path):
    """`(t + w - 1) * 0.1`: at the first window (t = 0, row 7) the true time is
    0.113 + (7 + 2) * 0.100667 = 1.019 s against 0.700 s -- 0.319 s off, over the 0.05 s bar."""
    eps = [_episode("clip-a")]
    ds = _dataset(eps)
    side = _sidecar(tmp_path, eps)
    ds.enable_clip_clock(side)
    ds.legacy_label_clock = True
    with pytest.raises(SystemExit, match=r"G3 \(train\): 3 of 3 label reads"):
        ds.assert_label_clock_true(side, split="train")


def test_a_REAL_corpus_with_NO_sidecar_is_REFUSED():
    ds = _dataset([_episode("clip-a")])
    ds.enable_clip_clock(None)
    with pytest.raises(SystemExit, match="no --clip-clock-sidecar"):
        ds.assert_label_clock_true(None, split="eval")


def test_a_SYNTHETIC_corpus_is_skipped_and_SAYS_so():
    ds = _dataset([_episode("clip-a")])
    rep = ds.assert_label_clock_true(None, split="train", synthetic=True)
    assert rep == {"g3": "SKIPPED: synthetic corpus (no true clock exists)", "split": "train"}


def test_UNVERIFIED_clips_are_counted_and_over_the_cap_REFUSED(tmp_path):
    """3 clips, the sidecar covers 2: 1/3 = 33 % unverified -- over the 1 % cap."""
    eps = [_episode("clip-a"), _episode("clip-b"), _episode("clip-c")]
    ds = _dataset(eps)
    side = _sidecar(tmp_path, eps[:2])
    ds.enable_clip_clock(side)
    with pytest.raises(SystemExit, match=r"1 of 3 clips \(33.33%\) have no measured clock"):
        ds.assert_label_clock_true(side, split="train")
    rep = ds.assert_label_clock_true(side, split="train", max_unverified_frac=0.5)
    assert (rep["g3"], rep["n_unverified_clips"], rep["n_clips"]) == ("PASS", 1, 3)
    assert rep["max_unverified_frac_source"] == "operator (--label-clock-max-unverified)"


def test_the_trainer_wires_the_guard_after_BOTH_clock_resolutions():
    """Reachability from the caller: `train()` calls the guard on the train and eval splits."""
    import inspect
    src = inspect.getsource(T.train)
    assert src.count(".assert_label_clock_true(") == 2
    assert 'split="train"' in src and 'split="eval"' in src


# ============================================================================================ #
# E2(b): the EVAL split scores TACTICAL on verified clips only, never a raised cap              #
# ============================================================================================ #
def _labelled(eps):
    from tanitad.data import v7_labels as v7l
    ds = _dataset(eps)
    ds.v7_by_sid = {int(e.episode_id): v7l.V7Label(
        clip_id=f"c{i}", tac_lat="LANE_KEEP", tac_lon="CRUISE", str_action="HOLD_MAIN_ROAD",
        str_goal="FOLLOW_ROUTE", tac_anchor=None,
        bands={"operative_s": [0.0, 2.0], "tactical_s": [2.0, 6.0], "strategic_s": [8.0, 30.0]},
        t0_s=8.0, horizon={}) for i, e in enumerate(eps)}
    return ds, v7l.IGNORE_ID


def _real_tactical_rows(ds, sid, ignore):
    """Window indices of clip `sid` whose lat_v7 is a REAL class (read via __getitem__)."""
    return [i for i, (e_i, _t) in enumerate(ds.index)
            if int(ds.episodes[e_i].episode_id) == sid and int(ds[i]["lat_v7"]) != ignore]


def test_E2b_EVAL_mode_excludes_the_unverified_clips_TACTICAL_targets_only(tmp_path):
    eps = [_episode("clip-a"), _episode("clip-b"), _episode("clip-c")]
    ds, ignore = _labelled(eps)
    side = _sidecar(tmp_path, eps[:2])                       # clip-c has no measured clock
    ds.enable_clip_clock(side)
    rep = ds.assert_label_clock_true(side, split="eval", exclude_unverified_tactical=True)
    assert (rep["g3"], rep["tactical_excluded_clips"], rep["tactical_scored_on_clips"]) == \
        ("PASS", 1, 2)
    assert rep["tactical_excluded_sids"] == [int(eps[2].episode_id)]
    assert rep["families_on_all_clips"] == ["longitudinal", "lateral", "strategic"]
    assert _real_tactical_rows(ds, int(eps[2].episode_id), ignore) == []      # excluded
    assert len(_real_tactical_rows(ds, int(eps[0].episode_id), ignore)) == 40  # rows 57..96
    # every OTHER target of the excluded clip is still produced (the other families keep it)
    i_c = next(i for i, (e_i, _t) in enumerate(ds.index) if e_i == 2)
    assert "goal_tac" in ds[i_c] and "future_poses_ext" in ds[i_c]


def test_E2b_RED_the_train_policy_still_REFUSES_the_same_coverage(tmp_path):
    eps = [_episode("clip-a"), _episode("clip-b"), _episode("clip-c")]
    ds, _ = _labelled(eps)
    side = _sidecar(tmp_path, eps[:2])
    ds.enable_clip_clock(side)
    with pytest.raises(SystemExit, match="have no measured clock"):
        ds.assert_label_clock_true(side, split="train")


def test_E2b_RED_without_the_exclusion_the_unverified_clip_IS_scored(tmp_path):
    """The mechanism, proven by removing it: clear the excluded set and the unverified clip's
    tactical rows come back -- so the exclusion, not the fixture, is what removed them."""
    eps = [_episode("clip-a"), _episode("clip-b"), _episode("clip-c")]
    ds, ignore = _labelled(eps)
    side = _sidecar(tmp_path, eps[:2])
    ds.enable_clip_clock(side)
    ds.assert_label_clock_true(side, split="eval", exclude_unverified_tactical=True)
    ds.tactical_excluded_sids = frozenset()
    assert len(_real_tactical_rows(ds, int(eps[2].episode_id), ignore)) > 0
