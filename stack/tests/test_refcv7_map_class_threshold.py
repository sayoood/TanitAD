"""refcv7 diagnostics F1 + F2 (2026-10-04): the `class_threshold` 10 cm decision rule and the thresholded /
0.2 m boundary-tolerant in-run monitor keys (``tanitad.models.map_head_hires``).

Source of the fixes: ``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-map-box-diagnostics``
(RESULT.md sec. 1.2 / 3, fixes F1 + F2; the TRAIN-fitted thresholds in ``raw/fit_map.json``).

What is pinned, each with a LITERAL expected value and a deliberate-regression arm that must go RED:

A. ⛔ DEFAULT UNCHANGED. On an exact-integer-grid input the default path (``prior_corrected`` / ``raw``, no thresholds)
   matches goldens computed from the TIP's ORIGINAL module before this change: the decision codes (sha256), the
   131 row keys (sha256 of the sorted names), every integer count, the derived-key set and three IoU values. Passing the
   thresholds ADDS 80 + 32 keys and changes no existing key's value (bit-equal).
B. The rule by hand: a 1-cell case whose posterior is known (``p_hat = softmax(z - log w)``), a multi-label answer,
   the partition ``decide`` returns, a zero-weight class never decided, unsupervised cells never predicted, and every
   refusal (no thresholds, wrong shape, NaN, no weights).
C. The shipped thresholds file: its values are the diagnostics' ``fit_map.json`` (literals incl. RESULT.md's 3-dp
   probabilities), its provenance fields, and the loader's refusals (schema, class order, prob != sigmoid(logit),
   thresholds fitted under OTHER class weights).
D. ACCEPTANCE ON REAL LOGITS: 40,000 real TRAIN cells of refcv7-r101-s0 (step 50,400; the diagnostics' own banked
   subsample, ``tests/fixtures/refcv7_infer_fixes``). Decision masks (sha256 of the packed bits), per-class x band
   prediction / intersection / GT counts and the 2-cell tolerant counts equal the LITERALS computed by the
   DIAGNOSTICS' OWN code (``code/diag_metrics.py``: ``decision_masks`` + ``tolerant_counts``), for the declared rule,
   the raw rule and the thresholded rule.
E. F2 by hand: a 1-cell line shifted by 0 / 2 / 3 cells, a half-length line and a partial one across a band edge, with
   hand-derived IoU_0 and IoU_2 (and ``None`` where both counts are 0).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.models import map_head_hires as H

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "refcv7_infer_fixes"
CFG = Path(H.__file__).resolve().parent.parent / "configs" / "refcv7_map_hires_class_thresholds_train.json"
CLASSES = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")

#: refcv7-r101-s0's frozen class weights (float32 values; the diagnostics' acc.pt['class_weight'])
W_MF = torch.tensor([0.14101280272006989, 0.20596274733543396, 0.93941730260849, 1.0740513801574707,
                     2.957174062728882, 1.571183204650879, 2.152963638305664, 0.16493748128414154])
#: raw/fit_map.json tau_phat_logit, copied
TAU_LOGIT = [-0.41000000000000014, -0.5199999999999996, -1.8999999999999986, -2.129999999999999,
             -2.8599999999999994, -3.1400000000000006, -3.09, -0.8099999999999987]
#: RESULT.md sec. 1.2 prose: "tau* on p_hat = 0.130 lane, 0.106 crosswalk, 0.054 arrow, 0.041 edge, 0.044 hatched
#: (0.373 drivable, 0.308 sidewalk, 0.399 nocls)" -- an INDEPENDENT source for the same eight numbers
TAU_PROB_3DP = {"nocls": 0.399, "drivable": 0.373, "lane": 0.130, "crosswalk": 0.106, "arrow": 0.054,
                "edge": 0.041, "hatched": 0.044, "sidewalk": 0.308}


def _thr():
    return H.load_class_thresholds(CFG)[0]


# =========================================================================== #
# A. the DEFAULT path is unchanged                                             #
# =========================================================================== #
def _grid_inputs(B=2, H_=400, W_=24):
    """An EXACT integer-grid input (values k/8): reproducible bit for bit on any platform."""
    b = torch.arange(B).view(B, 1, 1, 1)
    c = torch.arange(8).view(1, 8, 1, 1)
    h = torch.arange(H_).view(1, 1, H_, 1)
    w = torch.arange(W_).view(1, 1, 1, W_)
    v = (b * 131 + c * 17 + h * 31 + w * 7 + ((h * w) % 11) * 13) % 97
    z = (v - 48).to(torch.float32) / 8.0
    bb = torch.arange(B).view(B, 1, 1)
    hh = torch.arange(H_).view(1, H_, 1)
    ww = torch.arange(W_).view(1, 1, W_)
    codes = ((hh // 7 + ww + bb) % 9).to(torch.uint8)
    return z, torch.where(codes == 8, torch.full_like(codes, 255), codes)


#: computed from the TIP's ORIGINAL map_head_hires.py (ae58018) BEFORE this change -- never from the code under test
GOLD = {
    "decide_prior_corrected_sha256": "b172c3961861c5713905734a063751311bde71d0e3c478aeccdaf53a422383bf",
    "decide_raw_sha256": "978ae263097fdf4228e4dec38df57e52e6fba2854efc414b1b9143b0a17a4d3a",
    "n_row_keys": 131,
    "row_keys_sha256": "6517e05f59c21ed808c8e840dac7699ab28d1b798674099f6bc6144cfd71e359",
    "sum_n": 17087.0, "sum_inter": 2135.0, "sum_union": 32039.0, "sum_interraw": 2106.0, "sum_unionraw": 32068.0,
    "loss": 5.49459171295166, "n_map_hires_cells": 17087.0,
    "n_derived_keys": 48,
    "derived_keys_sha256": "39d4979429f63e59e489f80297ab3a6823b9463803194dd79023bbbc7399d6c0",
    "iou_lane_0_20": 0.0, "iou_drivable_20_40": 0.03337969401947149, "iouraw_edge_0_20": 0.06841046277665996,
    "count_vector_sha256": "49e12da2de370c98ec8d5c6f6bfa5c655521e42d37882070a80793e76e185c08",
}


def _sha(a: bytes) -> str:
    return hashlib.sha256(a).hexdigest()


def _row_golden_check(row: dict) -> None:
    # the F1/F2 keys are NEW; the golden is about every key the tip's row already carried
    row = {k: v for k, v in row.items() if not any(("_%s_" % s) in k for s in H.PER_CLASS_STATS_THR)}
    keys = sorted(row)
    assert len(keys) == GOLD["n_row_keys"]
    assert _sha("\n".join(keys).encode()) == GOLD["row_keys_sha256"]
    for st in ("n", "inter", "union", "interraw", "unionraw"):
        assert sum(row[k] for k in row if k.startswith("map_hires_%s_" % st)) == GOLD["sum_" + st], st
    assert float(row["loss"]) == pytest.approx(GOLD["loss"], rel=1e-6)
    assert row["n_map_hires_cells"] == GOLD["n_map_hires_cells"]
    vec = [row[k] for k in sorted(row) if any(k.startswith("map_hires_%s_" % s)
                                              for s in ("n", "inter", "union", "interraw", "unionraw"))]
    assert _sha(json.dumps(vec).encode()) == GOLD["count_vector_sha256"]
    d = H.derived_per_class(row, band_keys=H.band_keys_for_rows(400))
    assert len(d) == GOLD["n_derived_keys"]
    assert _sha("\n".join(sorted(d)).encode()) == GOLD["derived_keys_sha256"]
    assert d["map_hires_iou_lane_0_20"] == GOLD["iou_lane_0_20"]
    assert d["map_hires_iou_drivable_20_40"] == pytest.approx(GOLD["iou_drivable_20_40"], rel=1e-12)
    assert d["map_hires_iouraw_edge_0_20"] == pytest.approx(GOLD["iouraw_edge_0_20"], rel=1e-12)


def test_A_the_default_rules_and_the_default_row_are_the_tips_bit_for_bit():
    z, codes = _grid_inputs()
    for rule in ("prior_corrected", "raw"):
        d = H.decide(z, rule, W_MF)
        assert _sha(d.numpy().astype("int64").tobytes()) == GOLD["decide_%s_sha256" % rule], rule
    _row_golden_check(H.map_hires_loss_row(z, codes, class_weight=W_MF, decision_rule="prior_corrected"))
    assert H.DECISION_RULES == ("prior_corrected", "raw", "class_threshold")
    assert H.DECISION_RULES[0] == "prior_corrected"                      # the default every caller reads
    assert H.MapHiresConfig(w_map_hires=1.0).decision_rule == "prior_corrected"
    assert H.PER_CLASS_STATS == ("n", "lc", "gn", "gno", "inter", "union", "interraw", "unionraw")
    assert not [k for k in H.MapHiresConfig(w_map_hires=1.0).as_dict() if "threshold" in k]


def test_A_the_thresholds_ADD_80_row_keys_and_32_derived_keys_and_change_no_existing_value():
    z, codes = _grid_inputs()
    base = H.map_hires_loss_row(z, codes, class_weight=W_MF, decision_rule="prior_corrected")
    with_thr = H.map_hires_loss_row(z, codes, class_weight=W_MF, decision_rule="prior_corrected",
                                    class_thresholds=_thr())
    assert set(base) < set(with_thr) and len(with_thr) - len(base) == 80          # 5 stats x 8 classes x 2 bands
    new = sorted(set(with_thr) - set(base))
    assert all(k.split("_")[2] in ("interthr", "unionthr", "predthr", "tppthr2", "tpgthr2") for k in new)
    for k, v in base.items():                                                       # BIT-equal, incl. the loss
        assert (torch.equal(v, with_thr[k]) if torch.is_tensor(v) else v == with_thr[k]), k
    dn = H.derived_per_class(with_thr, band_keys=H.band_keys_for_rows(400))
    d0 = H.derived_per_class(base, band_keys=H.band_keys_for_rows(400))
    assert len(dn) - len(d0) == 32 and all(dn[k] == d0[k] for k in d0)             # iouthr + iou2thr x 8 x 2
    _row_golden_check(with_thr)                                                     # and the golden still holds


def test_A_DELIBERATE_REGRESSION_a_changed_default_goes_RED(monkeypatch):
    """Reintroduce a changed default (the raw argmax where the declared rule belongs): the golden must fail."""
    z, codes = _grid_inputs()
    monkeypatch.setattr(H, "decide", lambda lg, rule, w=None, class_thresholds=None: lg.argmax(dim=1))
    with pytest.raises(AssertionError):
        assert _sha(H.decide(z, "prior_corrected", W_MF).numpy().astype("int64").tobytes()) \
            == GOLD["decide_prior_corrected_sha256"]
    with pytest.raises(AssertionError):
        _row_golden_check(H.map_hires_loss_row(z, codes, class_weight=W_MF, decision_rule="prior_corrected"))


# =========================================================================== #
# B. the rule, by hand                                                         #
# =========================================================================== #
P_CASE = [0.40, 0.30, 0.12, 0.08, 0.04, 0.03, 0.021, 0.009]            # sums to 1.000
TAU_PROB = [0.5, 0.5, 0.10, 0.05, 0.05, 0.02, 0.02, 0.5]
#: cell is class c iff P_c >= tau_c: lane .12>=.10, crosswalk .08>=.05, edge .03>=.02, hatched .021>=.02 (arrow .04<.05)
MASK_CASE = [False, False, True, True, False, True, True, False]
#: partition = argmax_c (logit P_c - logit tau_c): crosswalk +0.502 beats lane +0.205, edge +0.416, hatched +0.051
PARTITION_CASE = 3


def _logit(p):
    return math.log(p / (1.0 - p))


def _case(weights):
    """z = ln(P) + ln(w): then p_hat = softmax(z - ln w) = P exactly, whatever the weights."""
    w = torch.tensor(weights, dtype=torch.float32)
    z = torch.tensor([math.log(p) + math.log(float(x)) for p, x in zip(P_CASE, weights)],
                     dtype=torch.float32).view(1, 8, 1, 1)
    return z, w, torch.tensor([_logit(p) for p in TAU_PROB], dtype=torch.float32)


def _assert_hand_case(masks_fn, decide_fn) -> None:
    for weights in ([1.0] * 8, [0.5, 0.5, 4.0, 4.0, 8.0, 4.0, 8.0, 0.5]):
        z, w, tau = _case(weights)
        assert masks_fn(z, w, tau)[0, :, 0, 0].tolist() == MASK_CASE, weights
        assert int(decide_fn(z, w, tau)[0, 0, 0]) == PARTITION_CASE, weights


def test_B_the_rule_by_hand_multilabel_mask_and_partition_with_unit_and_nonunit_weights():
    _assert_hand_case(lambda z, w, t: H.decide_masks(z, w, t),
                      lambda z, w, t: H.decide(z, "class_threshold", w, t))
    z, w, tau = _case([0.5, 0.5, 4.0, 4.0, 8.0, 4.0, 8.0, 0.5])
    assert int(z.argmax(dim=1)[0, 0, 0]) == 2                       # the RAW argmax would say lane
    assert int(H.decide(z, "prior_corrected", w)[0, 0, 0]) == 0     # the declared rule says nocls (P .40)
    assert int(H.decide(z, "raw", w)[0, 0, 0]) == 2


def test_B_DELIBERATE_REGRESSION_ignoring_the_weights_or_the_partition_goes_RED(monkeypatch):
    def no_weights_masks(z, w, tau, sup=None):                     # p_hat = softmax(z): the weights dropped
        return H.class_posterior_logits(z.float()) >= tau.view(1, -1, 1, 1)
    with pytest.raises(AssertionError):
        _assert_hand_case(no_weights_masks, lambda z, w, t: H.decide(z, "class_threshold", w, t))
    with pytest.raises(AssertionError):                            # the partition silently = the declared rule
        _assert_hand_case(lambda z, w, t: H.decide_masks(z, w, t),
                          lambda z, w, t: H.decide(z, "prior_corrected", w))


def test_B_a_zero_weight_class_is_never_decided_and_unsupervised_cells_are_never_predicted():
    z = torch.zeros(1, 8, 2, 2)
    z[:, 2] = 50.0                                                 # lane's logit towers ...
    w = torch.ones(8)
    w[2] = 0.0                                                     # ... but lane was never supervised
    tau = torch.full((8,), -50.0)                                  # everything qualifies
    m = H.decide_masks(z, w, tau)
    assert not bool(m[:, 2].any()) and bool(m[:, [0, 1, 3, 4, 5, 6, 7]].all())
    assert int((H.decide(z, "class_threshold", w, tau) == 2).sum()) == 0
    sup = torch.tensor([[[True, False], [False, True]]])
    assert H.decide_masks(z, torch.ones(8), tau, sup).sum(dim=1).flatten().tolist() == [8, 0, 0, 8]


def test_B_every_refusal_is_loud():
    z, w, tau = _case([1.0] * 8)
    with pytest.raises(ValueError, match="needs the per-class thresholds"):
        H.decide(z, "class_threshold", w, None)
    with pytest.raises(ValueError, match="needs the per-class thresholds"):
        H.decide_masks(z, w, None)
    with pytest.raises(ValueError, match="needs class_thresholds"):
        H.per_class_signal(z, torch.zeros(1, 1, 1, dtype=torch.uint8), class_weight=w,
                           decision_rule="class_threshold")
    with pytest.raises(ValueError, match="needs class_thresholds"):
        H.map_hires_loss_row(z, torch.zeros(1, 1, 1, dtype=torch.uint8), class_weight=w,
                             decision_rule="class_threshold")
    with pytest.raises(ValueError, match="8 finite logit values"):
        H.decide_masks(z, w, tau[:7])
    with pytest.raises(ValueError, match="8 finite logit values"):
        H.decide_masks(z, w, torch.cat([tau[:7], torch.tensor([float("nan")])]))
    with pytest.raises(ValueError, match="class weights"):
        H.decide(z, "class_threshold", None, tau)
    with pytest.raises(ValueError, match="all class weights are 0"):
        H.decide_masks(z, torch.zeros(8), tau)
    with pytest.raises(ValueError, match="decision rule"):
        H.decide(z, "posterior", w, tau)


# =========================================================================== #
# C. the shipped thresholds file                                               #
# =========================================================================== #
def test_C_the_shipped_file_is_the_diagnostics_fit_with_its_provenance():
    th, st = H.load_class_thresholds(CFG)
    assert th.dtype == torch.float32 and th.tolist() == pytest.approx(TAU_LOGIT, abs=1e-6)
    d = json.loads(CFG.read_text(encoding="utf-8"))
    assert d["schema"] == "tanitad.map_hires_class_thresholds/1" and d["classes"] == list(CLASSES)
    assert d["tau_phat_logit"] == TAU_LOGIT                           # copied verbatim, not re-typed rounded
    for c, p in zip(CLASSES, d["tau_phat_prob"]):
        assert round(p, 3) == TAU_PROB_3DP[c], c                      # RESULT.md's prose numbers
    pv = d["provenance"]
    assert pv["fit_split"] == "TRAIN" and pv["n_windows"] == 1112 and pv["n_episodes"] == 139
    assert pv["checkpoint_step"] == 50400 and pv["run"] == "refcv7-r101-s0"
    assert pv["source_file"].endswith("2026-10-04-refcv7-map-box-diagnostics/raw/fit_map.json")
    assert pv["source_md5"] == "39a6341d05458f41fd3a898977324cbc"
    assert st["sha256"] == hashlib.sha256(CFG.read_bytes()).hexdigest() and st["provenance"] == pv
    th2, _ = H.load_class_thresholds(CFG, class_weight=W_MF)           # bound to the run's weights: accepted
    assert torch.equal(th, th2)


def _mutated(tmp_path, **edits):
    d = json.loads(CFG.read_text(encoding="utf-8"))
    d.update(edits)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_C_the_loader_refuses_a_bad_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        H.load_class_thresholds(tmp_path / "nope.json")
    with pytest.raises(ValueError, match="schema"):
        H.load_class_thresholds(_mutated(tmp_path, schema="other/1"))
    with pytest.raises(ValueError, match="class order"):
        H.load_class_thresholds(_mutated(tmp_path, classes=list(reversed(CLASSES))))
    with pytest.raises(ValueError, match="8 finite tau_phat_logit"):
        H.load_class_thresholds(_mutated(tmp_path, tau_phat_logit=TAU_LOGIT[:7]))
    with pytest.raises(ValueError, match="not the sigmoid"):
        H.load_class_thresholds(_mutated(tmp_path, tau_phat_prob=[0.5] * 8))
    other = W_MF.clone()
    other[5] = other[5] * 1.01                                         # thresholds fitted under OTHER weights
    with pytest.raises(ValueError, match="do not transfer"):
        H.load_class_thresholds(CFG, class_weight=other)


# =========================================================================== #
# D. ACCEPTANCE on 40,000 REAL TRAIN cells vs the DIAGNOSTICS' own literals    #
# =========================================================================== #
# (computed by code/diag_metrics.py decision_masks + tolerant_counts on exactly these cells; [class][band])
LIT_N = [5396, 5117, 7002, 7000, 1263, 7001, 1812, 5409]
LIT = {"pc": {"pred": [[4464, 4422], [10092, 10134], [274, 324], [276, 259], [23, 29], [2, 1], [0, 0], [4869, 4831]], "gt": [[2692, 2704], [2568, 2549], [3452, 3550], [3567, 3433], [660, 603], [3397, 3604], [922, 890], [2742, 2667]], "inter": [[2205, 2225], [1896, 1861], [240, 279], [207, 194], [23, 25], [2, 1], [0, 0], [1656, 1594]], "tpp2": [[4383, 4333], [9776, 9785], [274, 324], [274, 258], [23, 27], [2, 1], [0, 0], [4741, 4711]], "tpg2": [[2687, 2702], [2568, 2549], [1128, 1283], [1178, 1008], [41, 44], [11, 4], [0, 0], [2740, 2665]]},
       "raw": {"pred": [[3982, 3932], [9103, 9263], [1169, 1194], [802, 760], [95, 88], [230, 237], [1, 1], [4618, 4525]], "gt": [[2692, 2704], [2568, 2549], [3452, 3550], [3567, 3433], [660, 603], [3397, 3604], [922, 890], [2742, 2667]], "inter": [[2109, 2127], [1921, 1909], [903, 899], [605, 583], [68, 62], [184, 189], [0, 1], [1704, 1658]], "tpp2": [[3913, 3855], [8831, 8959], [1163, 1187], [799, 756], [80, 73], [229, 237], [1, 1], [4505, 4417]], "tpg2": [[2687, 2700], [2568, 2549], [2797, 2892], [2440, 2272], [128, 107], [967, 1027], [1, 3], [2738, 2665]]},
       "thr": {"pred": [[4693, 4627], [10502, 10598], [1396, 1452], [986, 921], [129, 124], [604, 642], [302, 266], [6982, 7033]], "gt": [[2692, 2704], [2568, 2549], [3452, 3550], [3567, 3433], [660, 603], [3397, 3604], [922, 890], [2742, 2667]], "inter": [[2278, 2292], [1938, 1912], [1019, 1047], [709, 665], [73, 68], [398, 437], [84, 76], [2103, 2058]], "tpp2": [[4613, 4528], [10170, 10237], [1388, 1444], [981, 916], [101, 89], [602, 641], [216, 213], [6805, 6848]], "tpg2": [[2690, 2703], [2568, 2549], [2962, 3094], [2706, 2511], [147, 125], [1925, 2096], [337, 320], [2742, 2667]]}}
MASK_SHA = {"pc": "481fa1100eb5234de88ee5567fe51d282df9f159563c68b0cd55f807bd4f3148", "raw": "ec883a2907824a06b0dae381ac352262e17ba3803273589a22f0edc3b5d1756c", "thr": "b350a5ee141a374625b43ff80ca60361350deab398103a5f1775984236b52330"}
#: the diagnostics' tolerant_summary IoU_0 / IoU_2 per class x band under thr_phat
LIT_IOU0 = [[0.44605443508909337, 0.45485215320500105], [0.17409270571325908, 0.1701824655095683], [0.266126926090363, 0.2647281921618205], [0.1844432882414152, 0.1802656546489563], [0.10195530726256982, 0.10318664643399089], [0.11046350263669166, 0.11472827513783145], [0.0736842105263158, 0.07037037037037036], [0.2759480383151818, 0.2693012300444909]]
LIT_IOU2 = [[0.9822354970904102, 0.978249677951364], [0.9683869739097315, 0.9659369692394792], [0.8538306384618961, 0.8673612178874895], [0.7556987289470921, 0.7285215966258529], [0.20977449983045096, 0.19167154818924229], [0.5656116327060526, 0.5810488439881328], [0.31907563098880476, 0.33002469374909216], [0.974649097679748, 0.9736954358026447]]


def _cells():
    d = np.load(FIX / "refcv7_map_cells_train_subsample.npz")
    z = torch.from_numpy(d["z"]).float().T.reshape(1, 8, 400, 100).contiguous()
    y = torch.from_numpy(d["y"]).reshape(1, 400, 100)
    return z, y, torch.from_numpy(d["class_weight"])


def _t(rows):
    return torch.tensor(rows, dtype=torch.int64)


def _assert_real_cells_match_the_diagnostics() -> None:
    z, y, cw = _cells()
    th, _ = H.load_class_thresholds(CFG, class_weight=cw)
    sup = torch.ones(1, 400, 100, dtype=torch.bool)
    m = H.decide_masks(z, cw, th, sup)
    assert _sha(np.packbits(m.numpy().astype(np.uint8)).tobytes()) == MASK_SHA["thr"]
    sig = H.per_class_signal(z, y, class_weight=cw, decision_rule="prior_corrected", class_thresholds=th)
    assert torch.equal(sig["n"].to(torch.int64), _t(LIT["thr"]["gt"]))
    assert torch.equal(sig["n"].to(torch.int64).sum(1), torch.tensor(LIT_N))
    for st, key in (("interthr", "inter"), ("predthr", "pred"), ("tppthr2", "tpp2"), ("tpgthr2", "tpg2")):
        assert torch.equal(sig[st].to(torch.int64), _t(LIT["thr"][key])), st
    assert torch.equal(sig["unionthr"].to(torch.int64),
                       _t(LIT["thr"]["pred"]) + _t(LIT["thr"]["gt"]) - _t(LIT["thr"]["inter"]))
    # the DECLARED (default) rules, through the same entry point: unchanged on real logits
    for rule, lit, ik, uk in (("prior_corrected", LIT["pc"], "inter", "union"), ("raw", LIT["raw"], "interraw", "unionraw")):
        assert torch.equal(sig[ik].to(torch.int64), _t(lit["inter"])), rule
        assert torch.equal(sig[uk].to(torch.int64), _t(lit["pred"]) + _t(lit["gt"]) - _t(lit["inter"])), rule
    # and with class_threshold DECLARED, inter / union ARE the thresholded rule's (multi-label)
    sig2 = H.per_class_signal(z, y, class_weight=cw, decision_rule="class_threshold", class_thresholds=th)
    assert torch.equal(sig2["inter"], sig2["interthr"]) and torch.equal(sig2["union"], sig2["unionthr"])


def test_D_real_logits_the_decision_masks_and_every_count_equal_the_diagnostics_literals():
    _assert_real_cells_match_the_diagnostics()
    z, y, cw = _cells()
    sup = torch.ones(1, 400, 100, dtype=torch.bool)
    for rule, key in (("prior_corrected", "pc"), ("raw", "raw")):
        codes = H.decide(z, rule, cw)
        oh = torch.nn.functional.one_hot(codes, 8).permute(0, 3, 1, 2).bool() & sup[:, None]
        assert _sha(np.packbits(oh.numpy().astype(np.uint8)).tobytes()) == MASK_SHA[key], rule


def test_D_real_logits_the_monitor_ious_equal_the_diagnostics_tolerant_summary():
    z, y, cw = _cells()
    th, _ = H.load_class_thresholds(CFG, class_weight=cw)
    row = H.map_hires_loss_row(z, y, class_weight=cw, decision_rule="prior_corrected", class_thresholds=th)
    d = H.derived_per_class(row, band_keys=H.band_keys_for_rows(400))
    for ci, c in enumerate(CLASSES):
        for bi, bk in enumerate(("0_20", "20_40")):
            assert d["map_hires_iouthr_%s_%s" % (c, bk)] == pytest.approx(LIT_IOU0[ci][bi], abs=1e-12), (c, bk)
            assert d["map_hires_iou2thr_%s_%s" % (c, bk)] == pytest.approx(LIT_IOU2[ci][bi], abs=1e-12), (c, bk)


def test_D_DELIBERATE_REGRESSION_the_old_probability_gate_the_dropped_dilation_and_dropped_weights_go_RED(monkeypatch):
    """Each of the three ways to get the diagnostics' numbers WRONG must fail the acceptance: (1) thresholding at
    p_hat >= 0.5 -- the declared rule's behaviour on a thin class; (2) no boundary tolerance (k = 0); (3) p_hat without
    the class weights."""
    orig_masks, orig_dilate = H.decide_masks, H.dilate_chebyshev

    def gate_half(lg, cw, tau, sup=None):
        m = H.class_posterior_logits(lg.float() - torch.log(cw).view(1, -1, 1, 1)) >= 0.0     # p_hat >= 0.5
        return m if sup is None else (m & sup[:, None])
    monkeypatch.setattr(H, "decide_masks", gate_half)
    with pytest.raises(AssertionError):
        _assert_real_cells_match_the_diagnostics()
    monkeypatch.setattr(H, "decide_masks", orig_masks)

    monkeypatch.setattr(H, "dilate_chebyshev", lambda mask, k: mask)                         # tolerance off
    with pytest.raises(AssertionError):
        _assert_real_cells_match_the_diagnostics()
    monkeypatch.setattr(H, "dilate_chebyshev", orig_dilate)

    def no_weights(lg, cw, tau, sup=None):
        m = H.class_posterior_logits(lg.float()) >= tau.view(1, -1, 1, 1)
        return m if sup is None else (m & sup[:, None])
    monkeypatch.setattr(H, "decide_masks", no_weights)
    with pytest.raises(AssertionError):
        _assert_real_cells_match_the_diagnostics()
    monkeypatch.setattr(H, "decide_masks", orig_masks)
    _assert_real_cells_match_the_diagnostics()                                                # and GREEN again


# =========================================================================== #
# E. F2 by hand: the 2-cell (0.2 m) boundary-tolerant IoU                      #
# =========================================================================== #
def _line_logits(line_cols, rows, H_=400, W_=20):
    """z for a thresholded prediction: background class 0 everywhere, class 5 (edge) at (rows, line_cols)."""
    z = torch.zeros(1, 8, H_, W_)
    z[:, 0] = 10.0
    for r in rows:
        for c in line_cols:
            z[0, 0, r, c] = 0.0
            z[0, 5, r, c] = 20.0
    return z


def _gt_codes(col, rows, H_=400, W_=20):
    g = torch.zeros(1, H_, W_, dtype=torch.uint8)
    g[0, list(rows), col] = 5
    return g


def _edge_row(z, codes):
    tau = torch.zeros(8)                                           # p_hat >= 0.5 <=> logit >= 0
    row = H.map_hires_loss_row(z, codes, class_weight=torch.ones(8), decision_rule="prior_corrected",
                               class_thresholds=tau)
    return row, H.derived_per_class(row, band_keys=H.band_keys_for_rows(400))


GT_ROWS = range(190, 210)            # 10 cells in band 0_20 (190..199) and 10 in band 20_40 (200..209), column 5


@pytest.mark.parametrize("pred_col,iou0,iou2", [(5, 1.0, 1.0),    # exact
                                                (7, 0.0, 1.0),    # 2 cells (0.2 m) off: IoU_0 0, IoU_2 1
                                                (8, 0.0, 0.0)])   # 3 cells off: outside the tolerance
def test_E_a_shifted_line_IoU0_and_IoU2_by_hand(pred_col, iou0, iou2):
    row, d = _edge_row(_line_logits([pred_col], GT_ROWS), _gt_codes(5, GT_ROWS))
    for bk in ("0_20", "20_40"):
        assert d["map_hires_iouthr_edge_%s" % bk] == iou0, (pred_col, bk)
        assert d["map_hires_iou2thr_edge_%s" % bk] == iou2, (pred_col, bk)
    assert row["map_hires_predthr_edge_0_20"] == 10.0 and row["map_hires_n_edge_20_40"] == 10.0


def test_E_a_partial_line_across_a_band_edge_by_hand():
    """Prediction = column 5, rows 190..204 (15 cells) against the 20-cell GT line: band 0_20 sees 10/10; band 20_40 sees
    pred 5 (rows 200..204), gt 10, inter 5, tpp2 5, tpg2 7 (GT rows 200..206 lie within 2 rows of a predicted cell),
    so IoU_0 = 5 / 10 = 0.5 and IoU_2 = F / (2 - F) with P = 1, R = 0.7: F = 1.4 / 1.7, IoU_2 = 0.7."""
    row, d = _edge_row(_line_logits([5], range(190, 205)), _gt_codes(5, GT_ROWS))
    assert (row["map_hires_predthr_edge_20_40"], row["map_hires_interthr_edge_20_40"],
            row["map_hires_tppthr2_edge_20_40"], row["map_hires_tpgthr2_edge_20_40"]) == (5.0, 5.0, 5.0, 7.0)
    assert d["map_hires_iouthr_edge_20_40"] == 0.5
    assert d["map_hires_iou2thr_edge_20_40"] == pytest.approx(0.7, abs=1e-12)
    assert d["map_hires_iouthr_edge_0_20"] == 1.0 and d["map_hires_iou2thr_edge_0_20"] == 1.0


def test_E_an_empty_prediction_is_zero_recall_and_both_empty_is_undefined():
    row, d = _edge_row(_line_logits([5], range(190, 200)), _gt_codes(5, GT_ROWS))      # nothing predicted in 20_40
    assert d["map_hires_iou2thr_edge_20_40"] == 0.0 and d["map_hires_iouthr_edge_20_40"] == 0.0
    assert d["map_hires_iou2thr_hatched_0_20"] is None and d["map_hires_iouthr_hatched_0_20"] is None
    assert H.tolerant_iou_from_counts(0, 0, 0, 0) is None
    assert H.tolerant_iou_from_counts(0, 7, 0, 0) == 0.0
    assert H.tolerant_iou_from_counts(7, 0, 0, 0) == 0.0


def test_E_DELIBERATE_REGRESSION_no_tolerance_goes_RED(monkeypatch):
    monkeypatch.setattr(H, "dilate_chebyshev", lambda mask, k: mask)
    row, d = _edge_row(_line_logits([7], GT_ROWS), _gt_codes(5, GT_ROWS))
    with pytest.raises(AssertionError):
        assert d["map_hires_iou2thr_edge_0_20"] == 1.0
