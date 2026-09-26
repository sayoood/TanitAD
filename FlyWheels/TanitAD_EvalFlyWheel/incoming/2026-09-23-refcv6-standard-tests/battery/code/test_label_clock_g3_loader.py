"""G3 / G-CLOCK in the battery loader (SPEC_REFCV7 FIX-2; declared-vs-built E2(b), Master Mind 2026-09-26).

The loader's eval dataset must run the TREE UNDER TEST's `assert_label_clock_true(..., split="eval",
exclude_unverified_tactical=True)` right after `enable_clip_clock`, so TACTICAL is scored only on clips
whose label clock is MEASURED. Literal expectations on eval-139 with the run's post-switch config:
3 clips excluded (sha12 081b986f8888, 2aa810802777, 3db625a5f941; stationary, refused by the sidecar),
TACTICAL on 136 of 139. RED arm: without the call nothing is excluded (139).
On a pre-FIX tree (no G3): the record says "pre-FIX tree" -- asserted, not skipped.

run:  REFCV6_REPO=<tree> PYTHONPATH="<tree>/stack;<tree>/taniteval" python -m pytest -q test_label_clock_g3_loader.py
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402
import test_trunk_equalize_as_trained as T  # noqa: E402

POST_SWITCH_CONFIG = HERE.parent / "raw" / "thor_reads" / "config_resume34500_20260926.json"
KIT_CONFIG = Path("D:/refcv6_eval_kit/ckpt/config.json")
EXCLUDED = {"081b986f8888", "2aa810802777", "3db625a5f941"}


def _dataset(config):
    tr = L.trainer()
    args = L.parse_args(config)[0]
    cfg = T._pinned_cfg(tr, config)
    ds, eps, rec = L.build_eval_dataset(None, cfg, args, config, with_perception_targets=False)
    return tr, ds, rec


def _has_g3() -> bool:
    return hasattr(L.trainer().V3Dataset, "assert_label_clock_true")


def test_pre_fix_tree_records_no_g3():
    if _has_g3():
        pytest.skip("post-FIX tree: covered by the exclusion test below")
    if not (KIT_CONFIG.exists() and Path(str(L.KIT / "data/refcv6-b1-416x1024-eval139")).exists()):
        pytest.skip("the kit is not on this box")
    # a pre-FIX parser does not know --clip-clock-sidecar: the run's PRE-switch config is the one it can read
    tr, ds, rec = _dataset(json.load(open(KIT_CONFIG, encoding="utf-8")))
    assert rec["label_clock_g3"]["status"].startswith("pre-FIX tree")
    assert len(getattr(ds, "tactical_excluded_sids", frozenset())) == 0


def test_g3_excludes_the_three_unverified_clips_from_tactical_only():
    if not _has_g3():
        pytest.skip("pre-FIX tree: covered by test_pre_fix_tree_records_no_g3")
    if not (POST_SWITCH_CONFIG.exists() and Path(str(L.KIT / "data/refcv6-b1-416x1024-eval139")).exists()):
        pytest.skip("the post-switch config / kit eval cache is not on this box")
    config = json.load(open(POST_SWITCH_CONFIG, encoding="utf-8"))
    tr, ds, rec = _dataset(config)
    g3 = rec["label_clock_g3"]
    assert g3["g3"] == "PASS"
    assert (g3["n_clips"], g3["tactical_excluded_clips"], g3["tactical_scored_on_clips"]) == (139, 3, 136)
    assert set(g3["tactical_excluded_sha12"]) == EXCLUDED
    assert g3["tactical_excluded_reason"]
    assert len(ds.tactical_excluded_sids) == 3


def test_red_arm_without_the_call_nothing_is_excluded(monkeypatch):
    if not _has_g3():
        pytest.skip("pre-FIX tree: there is no call to remove")
    if not (POST_SWITCH_CONFIG.exists() and Path(str(L.KIT / "data/refcv6-b1-416x1024-eval139")).exists()):
        pytest.skip("the post-switch config / kit eval cache is not on this box")
    config = json.load(open(POST_SWITCH_CONFIG, encoding="utf-8"))
    monkeypatch.setattr(L, "label_clock_g3", lambda e_ds, args: {"status": "SKIPPED (red arm)"})
    tr, ds, rec = _dataset(config)
    assert rec["label_clock_g3"] == {"status": "SKIPPED (red arm)"}
    assert len(getattr(ds, "tactical_excluded_sids", frozenset())) == 0      # all 139 clips carry TACTICAL
