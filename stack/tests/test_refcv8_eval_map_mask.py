"""refcv8 WP-C fix 6 -- the eval clips WITHOUT map GT are MASKED by the existing code path (this test pins that; no code change).

D3 / D1: two EVAL clips (sha12 ``081b986f8888``, ``2aa810802777``; 342 windows = 171 + 171, 1.44 % of the 23,772 eval windows) have no
SAM3 map ground truth ("NO_V2"); all train clips have it. "Confirm the eval masks them." It does, at THREE layers, each pinned here with a
literal and a mutation that must go RED:

 1. DATASET   ``V3Dataset._map_fine_item``: a clip whose store raises ``FileNotFoundError`` yields an ALL-``NOT_SEEN`` (255) target and
              ``map_fine_label`` False -- a window with zero supervised cells, never an all-zero ("nocls") map.
 2. LOSS ROW  ``map_head_hires.map_hires_loss_row``: a window of all-255 codes contributes NOTHING to the loss or any of the 66 row keys (both cell
              counts, every per-class x band intersection / union / count). Appending such a window changes no number.
 3. TRAINER   ``_map_hires_loss``: the batch's ``map_fine_label`` selects the labelled windows BEFORE the loss row; a batch with no labelled window
              returns a counted zero (``n_map_hires_cells`` 0), never an absent key.
 +  COVERAGE  ``perception_targets.require_map_coverage`` counts them as ``no_file`` (frac_ok 0.9856 >= the 0.90 floor), so the run is not refused.

The pooled IoU of an eval row is a ratio of SUMS (``derived_per_class``), so an unlabelled window cannot dilute it; the eval windows are a
``randperm`` subset (no batch is ever wholly unlabelled), so the batch-mean of the plain loss is not diluted either (reasoned in the report).

REAL-DATA arm (skipped without the local eval kit): the eval manifest's 139 clips minus the 137 map files on disk is EXACTLY the two sha12s above, and
each of the 139 clips has ``T - 28`` windows, giving 342.
"""
from __future__ import annotations

import errno
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_v3_train as t                                       # noqa: E402
from tanitad.data import perception_targets as ptg             # noqa: E402
from tanitad.data import semantic_map_gt_fine as sf            # noqa: E402
from tanitad.models import map_head_hires as H                 # noqa: E402

NO_MAP_SHA12 = ("081b986f8888", "2aa810802777")
KIT = Path(os.environ.get("REFCV6_EVAL_KIT", "D:/refcv6_eval_kit/data"))
NOT_SEEN = 255


def _hw():
    ext = sf.EXTENT_V2
    return tuple(ext.fine_shape)


class _Store:
    """A fine-map store with files for every clip except the ones in ``missing`` (the real failure: ENOENT)."""
    extent = sf.EXTENT_V2

    def __init__(self, missing):
        self.missing = set(missing)

    def frames_for_windows(self, cid, idx, n_stack=3):
        if cid in self.missing:
            raise FileNotFoundError(errno.ENOENT, "no map GT for this clip")
        h, w = _hw()
        return SimpleNamespace(codes=np.full((1, h, w), 1, np.uint8), frame_idx=np.array([int(idx[0]) + n_stack - 1]))


def _ns(missing):
    return SimpleNamespace(map_clip_of_ep={11: "clip-with-map", 22: "clip-without-map"},
                           map_fine_store=_Store(missing), map_n_stack=3)


def _item(missing, eid):
    return t.V3Dataset._map_fine_item(_ns(missing), SimpleNamespace(episode_id=eid), 5)


# =========================================================================== #
# 1. dataset layer                                                             #
# =========================================================================== #
def test_a_clip_without_a_map_file_is_an_all_NOT_SEEN_target_with_label_False():
    it = _item({"clip-without-map"}, 22)
    assert bool(it["map_fine_label"]) is False
    m = it["map_fine"]
    assert m.dtype == torch.uint8 and tuple(m.shape) == _hw() and bool((m == NOT_SEEN).all())
    assert sf.NOT_SEEN_CODE == NOT_SEEN


def test_a_clip_with_a_map_file_is_labelled_and_untouched():
    it = _item({"clip-without-map"}, 11)
    assert bool(it["map_fine_label"]) is True and int(it["map_fine"].min()) == 1 == int(it["map_fine"].max())


def test_MUTATION_a_missing_file_that_fell_back_to_an_all_zero_map_would_be_a_supervised_class():
    """The regression this layer guards: 0 is the 'nocls' class code, so an all-zero fallback would be 'supervised background'."""
    zero = torch.zeros(_hw(), dtype=torch.uint8)
    assert bool((zero != NOT_SEEN).all()) and not bool((_item({"clip-without-map"}, 22)["map_fine"] != NOT_SEEN).any())


# =========================================================================== #
# 2. loss-row layer                                                            #
# =========================================================================== #
def _pair(fill_unlabelled):
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 200, 40)
    codes = torch.randint(0, 8, (2, 200, 40), dtype=torch.uint8)
    codes[1] = fill_unlabelled
    return logits, codes


def _row(logits, codes):
    return H.map_hires_loss_row(logits, codes, class_weight=torch.ones(8), decision_rule="prior_corrected")


def test_an_all_NOT_SEEN_window_changes_not_one_of_the_66_row_keys_nor_the_loss():
    logits, codes = _pair(NOT_SEEN)
    both, one = _row(logits, codes), _row(logits[:1], codes[:1])
    ks = [k for k in both if k != "loss"]
    assert len(ks) == 66 and set(ks) == {k for k in one if k != "loss"}              # 66 keys + the loss
    assert float(both["loss"]) == float(one["loss"])
    assert all(float(both[k]) == float(one[k]) for k in ks), [k for k in ks if float(both[k]) != float(one[k])]
    assert one["n_map_hires_cells"] == 200 * 40 == both["n_map_hires_cells"]         # literal: 8,000 supervised cells, all window 0's


def test_MUTATION_an_all_zero_window_is_NOT_masked_and_doubles_the_supervised_cells():
    logits, codes = _pair(0)
    both, one = _row(logits, codes), _row(logits[:1], codes[:1])
    assert both["n_map_hires_cells"] == 16000 != one["n_map_hires_cells"]
    assert float(both["loss"]) != float(one["loss"])


# =========================================================================== #
# 3. trainer layer                                                             #
# =========================================================================== #
def _stub_model():
    return SimpleNamespace(_map_hires=SimpleNamespace(cfg=SimpleNamespace(decision_rule="prior_corrected")),
                           _w_map_hires=1.0, _map_hires_class_weight=torch.ones(8), _map_lift_valid_mask=False)


def _trainer_loss(logits, codes, label):
    out = {"fmap_s8": torch.zeros(1), "perception": {"map_hires_logits": logits, "map_hires_lift_valid": torch.ones(logits.shape[0], 1, 1)}}
    batch = {"map_fine": codes, "map_fine_label": torch.tensor(label)}
    extra: dict = {}
    term = t._map_hires_loss(_stub_model(), out, batch, "cpu", extra, with_metrics=True)
    return term, extra


def test_the_trainer_selects_labelled_windows_before_the_loss_row():
    logits, codes = _pair(NOT_SEEN)
    term2, e2 = _trainer_loss(logits, codes, [True, False])
    term1, e1 = _trainer_loss(logits[:1], codes[:1], [True])
    assert e2["map_hires_n_windows"] == 2.0 and e2["map_hires_n_labelled"] == 1.0 and e1["map_hires_n_labelled"] == 1.0
    pk = [k for k in e1 if k.startswith("map_hires_") and k not in ("map_hires_n_windows", "map_hires_n_labelled")]
    assert len(pk) > 60 and all(float(e1[k]) == float(e2[k]) for k in pk)
    assert float(term1) == float(term2)


def test_a_batch_with_no_labelled_window_is_a_counted_zero_not_an_absent_key():
    logits, codes = _pair(NOT_SEEN)
    term, e = _trainer_loss(logits, codes, [False, False])
    assert float(term) == 0.0 and e["n_map_hires_cells"] == 0.0 and e["map_hires_n_labelled"] == 0.0 and "map_hires" in e


def test_MUTATION_ignoring_the_label_flag_still_cannot_leak_cells_but_the_label_count_would_lie():
    """If the trainer stopped reading ``map_fine_label`` the 255 codes would still contribute zero cells (layer 2), but the labelled-window
    count - the number a Watch row stamps - would read 2 instead of 1. The literal pins the count."""
    logits, codes = _pair(NOT_SEEN)
    _, e = _trainer_loss(logits, codes, [True, False])
    assert e["map_hires_n_labelled"] == 1.0 != 2.0


# =========================================================================== #
# +  coverage                                                                 #
# =========================================================================== #
def test_coverage_counts_the_missing_clips_as_no_file_and_the_run_is_not_refused():
    class Store:
        layout_of = {}

        def open(self, cid):
            if cid in NO_MAP:
                raise FileNotFoundError(errno.ENOENT, "none")
            return SimpleNamespace(n_frames=199)
    NO_MAP = {"c-no-1", "c-no-2"}
    windows = [(f"c{i}", w) for i in range(137) for w in range(171)] + [(c, w) for c in sorted(NO_MAP) for w in range(171)]
    rep = ptg.require_map_coverage(windows, Store(), n_stack=3)                 # default floor 0.90: must NOT raise
    assert rep["n_windows"] == 139 * 171 == 23769
    assert rep["n_no_file"] == 342 and rep["verdict"] == "PASS" and ptg.MIN_MAP_COVERAGE == 0.90
    n = 139 * 171
    assert rep["frac_ok"] == pytest.approx((n - 342) / n, abs=1e-12) and rep["frac_ok"] > 0.98


# =========================================================================== #
# REAL DATA                                                                    #
# =========================================================================== #
@pytest.mark.skipif(not (KIT / "refcv6-b1-416x1024-eval139" / "_v2manifest.pt").exists() or not (KIT / "sam3_gt_v3_eval").is_dir(),
                    reason="the local eval kit (eval139 manifest + SAM3 eval map files) is not on this machine")
def test_REAL_exactly_the_two_documented_eval_clips_have_no_map_file_and_they_hold_342_windows():
    man = torch.load(KIT / "refcv6-b1-416x1024-eval139" / "_v2manifest.pt", map_location="cpu", weights_only=False)
    sha = [hashlib.sha256(c.encode()).hexdigest()[:12] for c in man["clip_id"]]
    files = {f.split(".")[0] for f in os.listdir(KIT / "sam3_gt_v3_eval") if f.endswith(".sam3mapgt.npz")}
    assert len(sha) == 139 == len(set(sha)) and len(files) == 137
    assert sorted(set(sha) - files) == sorted(NO_MAP_SHA12) and not (files - set(sha))
    n_frames = [len(p) for p in man["poses"]]
    assert sum(n_frames) - 139 * 28 == 23772                                     # windows = frames - 28 per clip (D1: 23,772 eval windows)
    w = {s: n - 28 for s, n in zip(sha, n_frames)}
    assert w[NO_MAP_SHA12[0]] + w[NO_MAP_SHA12[1]] == 342
