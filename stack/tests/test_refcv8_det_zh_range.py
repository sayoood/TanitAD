"""refcv8 WP-C fix 5: z / h of the box head BY RANGE, beyond 30 m flagged LOW-TRUST (``tanitad.eval.detection_zh`` +
``detection_metrics.window_packs(.., with_zh_range=True)``), OPT-IN.

Source: D3 (RESULT.md "Boxes"; ``raw/c3b_agents_detail.json`` ``base_face_by_range``): the GT's base height is off the ground
plane by > 1.5 m for 0.906 % of boxes < 30 m, 11.70 % at 30-100 m, 25.94 % > 100 m -> "treat z as near-field only".

LITERALS ONLY. The synthetic scene uses 3-4-5-style triples so every range is exact by hand:
GT centres (6, 8) -> 10 m, (24, 7) -> 25 m, (36, 15) -> 39 m, (48, 14) -> 50 m, (58, 16) -> sqrt(3620) = 60.1664 m.
Deliberate regressions that must go RED: a range that ignores y; a "trust everything" report; a trust table whose threshold
moved. DEFAULT: ``window_packs(..)`` without the flag is byte-identical to the unmodified tip function (golden digest computed
from the TIP's ``detection_metrics.py`` before the change).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.eval import detection_metrics as det
from tanitad.eval import detection_zh as Z
from tanitad.models import agent_slots as AS

TRUST = Path(Z.__file__).resolve().parent.parent / "configs" / "refcv8_box_zh_range_trust.json"

GT_XY = [(6.0, 8.0), (24.0, 7.0), (36.0, 15.0), (48.0, 14.0), (58.0, 16.0)]
GT_CZ = [0.8, 0.9, 1.0, 1.1, 1.2]
GT_H = [1.6, 1.8, 2.0, 2.2, 2.4]
D_Z = [0.1, 0.2, 0.5, 1.0, 2.0]                   # |dz| per GT row
D_H = [0.05, 0.10, 0.4, 0.8, 1.6]                 # |dh| per GT row
ZH_MASK = [True, True, True, False, True]         # the 4th (50 m) row carries no 3-D label
N = len(GT_XY)


def _scene():
    C = len(AS.AGENT_CLASSES)
    box_gt = torch.tensor([[[x, y, 4.5, 1.9] for x, y in GT_XY]])
    pred = {"presence_logit": torch.full((1, N), -1.0), "box": box_gt.clone(), "cls_logits": torch.zeros(1, N, C),
            "cz": torch.tensor([[c + d for c, d in zip(GT_CZ, D_Z)]]), "h": torch.tensor([[h + d for h, d in zip(GT_H, D_H)]])}
    tgt = {"box": box_gt, "cls": torch.zeros(1, N, dtype=torch.long), "valid": torch.ones(1, N, dtype=torch.bool),
           "cz": torch.tensor([GT_CZ]), "h": torch.tensor([GT_H]), "zh_mask": torch.tensor([ZH_MASK])}
    vis = {"n_full": torch.full((1, N), 1000), "n_vis": torch.full((1, N), 900), "known": torch.ones(1, N, dtype=torch.bool)}
    match = {"rows": [torch.arange(N)], "cols": [torch.arange(N)]}
    return pred, tgt, vis, match


def _packs(**kw):
    pred, tgt, vis, match = _scene()
    return det.window_packs(pred, tgt, vis, match=match, **kw)


def _digest(pk: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(pk):
        v = pk[k]
        a = np.asarray(v)
        h.update(repr((k, a.dtype.str, a.shape)).encode() + (b"" if a.dtype == object else np.ascontiguousarray(a).tobytes()))
    return h.hexdigest()


#: computed by the TIP's UNMODIFIED detection_metrics.window_packs (blob 175b5e17) on exactly this scene, BEFORE the change
GOLDEN_DEFAULT_KEYS = ["cls", "cls_corr", "ep", "exempt", "gt_cls", "gt_xy", "hidden", "ign", "logit", "matched", "pair_err",
                       "pair_size_err", "pair_z_err", "pos", "xy"]
GOLDEN_DEFAULT_DIGEST = "339035681da98eac542fee218cf2222cdecb469b07c678da2329a309cbb57222"


# =========================================================================== #
# 1. DEFAULT UNCHANGED                                                         #
# =========================================================================== #
def test_default_window_packs_is_byte_identical_to_the_tip_and_carries_no_new_key():
    pk = _packs()[0]
    assert sorted(pk) == GOLDEN_DEFAULT_KEYS
    assert _digest(pk) == GOLDEN_DEFAULT_DIGEST
    assert pk["pair_z_err"].tolist() == pytest.approx([0.1, 0.2, 0.5, 2.0], abs=1e-6)      # the pre-existing key, same values


def test_zh_range_keys_with_no_trust_table_emits_nothing():
    assert Z.zh_range_keys(_packs(with_zh_range=True), "box3d", None) == {}


# =========================================================================== #
# 2. the pack extension                                                        #
# =========================================================================== #
def test_the_flagged_pack_carries_range_and_h_error_aligned_with_pair_z_err():
    pk = _packs(with_zh_range=True)[0]
    # only the z/h-LABELLED matched pairs (the 50 m row has zh_mask False): 10, 25, 39 and sqrt(3620) m
    assert pk["pair_zh_range"].tolist() == pytest.approx([10.0, 25.0, 39.0, math.sqrt(3620.0)], abs=1e-5)
    assert pk["pair_zh_range"][3] == pytest.approx(60.16643, abs=1e-5)
    assert pk["pair_z_err"].tolist() == pytest.approx([0.1, 0.2, 0.5, 2.0], abs=1e-6)
    assert pk["pair_h_err"].tolist() == pytest.approx([0.05, 0.10, 0.4, 1.6], abs=1e-6)
    assert pk["pair_zh_range"].shape == pk["pair_z_err"].shape == pk["pair_h_err"].shape
    # every other key is the default pack's, value for value
    base = _packs()[0]
    for k in base:
        assert np.array_equal(np.asarray(base[k]), np.asarray(pk[k])), k


def test_a_head_without_h_reports_nan_h_never_zero():
    pred, tgt, vis, match = _scene()
    del pred["h"]
    pk = det.window_packs(pred, tgt, vis, match=match, with_zh_range=True)[0]
    assert np.isnan(pk["pair_h_err"]).all() and pk["pair_h_err"].shape == pk["pair_z_err"].shape


# =========================================================================== #
# 3. the report                                                                #
# =========================================================================== #
def _trust():
    return Z.load_zh_trust(TRUST)[0]


def test_the_shipped_trust_table_is_d3s_and_the_near_field_is_derived_to_30_m():
    cfg, stamp = Z.load_zh_trust(TRUST)
    b = cfg["bins"]
    assert b["near<30"]["frac_abs_gt_1.5"] == 0.009060236934182363           # D3 raw/c3b_agents_detail.json, copied
    assert b["mid30-100"]["frac_abs_gt_1.5"] == 0.116997586004722
    assert b["far>100"]["frac_abs_gt_1.5"] == 0.2594109930981807
    assert cfg["low_trust_frac"] == 0.05 and cfg["near_field_m"] == 30.0
    assert stamp["provenance"]["source_md5"] == "353e934d8450e5d9144a345de58e2694"
    assert stamp["sha256"] == hashlib.sha256(TRUST.read_bytes()).hexdigest()


def test_the_report_by_range_has_the_hand_computed_numbers():
    k = Z.zh_range_keys(_packs(with_zh_range=True), "box3d", _trust())
    assert sorted(k) == sorted(Z.zh_key_names("box3d"))
    # 0-30 m: pairs at 10 m and 25 m
    assert k["eval_box3d_zh_0_30_n"] == 2.0
    assert k["eval_box3d_zh_0_30_z_mae"] == pytest.approx(0.15, abs=1e-6) and k["eval_box3d_zh_0_30_z_med"] == pytest.approx(0.15, abs=1e-6)
    assert k["eval_box3d_zh_0_30_h_mae"] == pytest.approx(0.075, abs=1e-6) and k["eval_box3d_zh_0_30_low_trust"] == 0.0
    # 30-60 m: the 39 m pair (the 50 m pair carries no label)
    assert k["eval_box3d_zh_30_60_n"] == 1.0 and k["eval_box3d_zh_30_60_z_mae"] == pytest.approx(0.5, abs=1e-6)
    assert k["eval_box3d_zh_30_60_h_mae"] == pytest.approx(0.4, abs=1e-6) and k["eval_box3d_zh_30_60_low_trust"] == 1.0
    # >= 60 m: the corner pair
    assert k["eval_box3d_zh_60_inf_n"] == 1.0 and k["eval_box3d_zh_60_inf_z_mae"] == pytest.approx(2.0, abs=1e-6)
    assert k["eval_box3d_zh_60_inf_low_trust"] == 1.0
    # the quotable headline pools ONLY the trusted bin
    assert k["eval_box3d_zh_nearfield_n"] == 2.0 and k["eval_box3d_zh_nearfield_z_mae"] == pytest.approx(0.15, abs=1e-6)
    assert k["eval_box3d_zh_nearfield_h_med"] == pytest.approx(0.075, abs=1e-6) and k["eval_box3d_zh_nearfield_range_m"] == 30.0
    assert k["eval_box3d_zh_lowtrust_pair_frac"] == pytest.approx(0.5, abs=1e-12)
    # and the all-range mean the trainer would have quoted is dominated by the untrusted far field
    allr = np.mean([0.1, 0.2, 0.5, 2.0])
    assert allr == pytest.approx(0.7, abs=1e-12) and allr > 4 * k["eval_box3d_zh_nearfield_z_mae"]


def test_a_pack_without_the_range_is_refused_not_read_as_no_error():
    with pytest.raises(ValueError, match="with_zh_range"):
        Z.zh_range_keys(_packs(), "box3d", _trust())
    with pytest.raises(ValueError, match="no packs"):
        Z.zh_range_keys([], "box3d", _trust())


def test_empty_bins_are_nan_never_zero():
    pk = _packs(with_zh_range=True)
    for p in pk:                                              # drop the >= 60 m pair
        keep = p["pair_zh_range"] < 60.0
        for k in ("pair_z_err", "pair_zh_range", "pair_h_err"):
            p[k] = p[k][keep]
    k = Z.zh_range_keys(pk, "box3d", _trust())
    assert k["eval_box3d_zh_60_inf_n"] == 0.0 and math.isnan(k["eval_box3d_zh_60_inf_z_mae"])
    assert math.isnan(k["eval_box3d_zh_60_inf_z_med"]) and math.isnan(k["eval_box3d_zh_60_inf_h_med"])


def test_MUTATION_a_range_that_ignores_y_moves_the_corner_pair_into_the_wrong_bin_and_goes_red():
    pk = _packs(with_zh_range=True)
    for p in pk:
        p["pair_zh_range"] = np.array([6.0, 24.0, 36.0, 58.0])          # |x| only (the bug: y dropped)
    k = Z.zh_range_keys(pk, "box3d", _trust())
    assert k["eval_box3d_zh_60_inf_n"] == 0.0 != 1.0                    # the real report has the corner pair here


def test_MUTATION_pooling_every_range_into_the_headline_goes_red():
    """The pre-fix behaviour: a trust table with no low-trust bin (near field = inf) reports ALL pairs as the headline."""
    d = json.loads(TRUST.read_text(encoding="utf-8"))
    for b in d["label_base_height_error"].values():
        b["frac_abs_gt_1.5"] = 0.0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        p.write_text(json.dumps(d))
        cfg = Z.load_zh_trust(p)[0]
    assert cfg["near_field_m"] == math.inf
    k = Z.zh_range_keys(_packs(with_zh_range=True), "box3d", cfg)
    assert k["eval_box3d_zh_nearfield_n"] == 4.0 != 2.0                  # the real headline excludes the 2 untrusted pairs
    assert k["eval_box3d_zh_lowtrust_pair_frac"] == 0.0


def test_MUTATION_a_looser_threshold_moves_the_near_field_and_goes_red(tmp_path):
    d = json.loads(TRUST.read_text(encoding="utf-8"))
    d["low_trust_frac"] = 0.2                                            # mid (0.117) is no longer 'low'; far (0.259) still is
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d))
    assert Z.load_zh_trust(p)[0]["near_field_m"] == 100.0 != 30.0


def test_the_loader_refuses_a_bad_table(tmp_path):
    base = json.loads(TRUST.read_text(encoding="utf-8"))

    def write(mut):
        d = json.loads(json.dumps(base))
        mut(d)
        p = tmp_path / "x.json"
        p.write_text(json.dumps(d))
        return p
    with pytest.raises(ValueError, match="does not exist"):
        Z.load_zh_trust(tmp_path / "none.json")
    with pytest.raises(ValueError, match="schema"):
        Z.load_zh_trust(write(lambda d: d.update(schema="x/1")))
    with pytest.raises(ValueError, match="low_trust_frac"):
        Z.load_zh_trust(write(lambda d: d.update(low_trust_frac=1.5)))
    with pytest.raises(ValueError, match="label_base_height_error"):
        Z.load_zh_trust(write(lambda d: d.pop("label_base_height_error")))
    with pytest.raises(ValueError, match="contiguous"):
        Z.load_zh_trust(write(lambda d: d["label_base_height_error"]["mid30-100"].update(lo_m=35.0)))
