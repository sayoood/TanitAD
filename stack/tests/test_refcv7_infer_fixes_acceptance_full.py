"""refcv7 diagnostics F1 / F4 (2026-10-04) -- the FULL-SET ACCEPTANCE against the diagnostics' own banked binary evidence.

The committed unit tests (``test_refcv7_map_class_threshold.py``, ``test_refcv7_det_presence_gates.py``) run on small
fixtures cut from this evidence. THIS file runs on all of it, and is SKIPPED where the evidence is not on the machine
(``REFCV7_DIAG_BIN`` or ``D:/refcv7_diag_bin/2026-10-04``; two md5-verified copies exist, neither in the repo; the
diagnostics' ``code/`` is found in the repo's research package or at ``REFCV7_DIAG_PKG``):

* BOX -- ``eval_final.packs.pkl``: the 1,061 labelled EVAL-DIAG windows' per-window packs of refcv7-r101-s0 at step
  50,400. The census at the declared 0.5 gate (``train_row_keys``) and at the TRAIN P = R gates
  (``gated_census_keys`` with the shipped gates file) equals the diagnostics' ``raw/B_box.json`` EVAL numbers EXACTLY:
  box3d n_conf / tp 28 / 27 -> 3299 / 1014, conf_ratio 0.00826 -> 0.97316, F1 0.0158 -> 0.30318; agent 0 / 0 -> 3373 / 699,
  conf_ratio 0.0 -> 0.99499, F1 undefined -> 0.20671 (the literals below ARE those JSON values).
* MAP -- ``train_fitpass.acc.pt['sub']``: the 4,599,016-cell 1 % TRAIN subsample of REAL logits. The ``class_threshold``
  masks equal the diagnostics' ``diag_metrics.decision_masks`` (``thr_phat``) bit for bit on EVERY cell (the diagnostics'
  own code is imported from the research package, not re-implemented), the declared rule equals its ``pc`` argmax, and the
  pooled fraction of cells above each class's threshold agrees with the FULL pass's score histogram (460 M cells) to
  within 5 binomial standard errors -- two different data paths (a cell sample through the rule, a histogram sweep).

No GPU, no checkpoint: both inputs are per-window packs / cell logits the diagnostics banked.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import sys
from pathlib import Path

import pytest
import torch

from tanitad.eval import detection_metrics as det
from tanitad.models import map_head_hires as H

BIN = Path(os.environ.get("REFCV7_DIAG_BIN", "D:/refcv7_diag_bin/2026-10-04"))
REPO = Path(__file__).resolve().parents[2]
PKG = Path(os.environ.get("REFCV7_DIAG_PKG", str(
    REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-04-refcv7-map-box-diagnostics")))
STACK = Path(__file__).resolve().parents[1]
CFG = STACK / "tanitad" / "configs" / "refcv7_map_hires_class_thresholds_train.json"
GATES = STACK / "tanitad" / "configs" / "refcv7_det_presence_gates_train.json"

# raw/B_box.json: <head>.eval.transforms.T0 (declared gate 0.5) and .T3 (the TRAIN P = R gate); n_pos 3390 both heads
B_BOX = {
    "box3d": {"T0": dict(n_conf=28, tp=27, conf_ratio=0.008259587020648967),
              "T3": dict(n_conf=3299, tp=1014, conf_ratio=0.9731563421828908, prec=0.3073658684449833,
                         rec=0.2991150442477876, f1=0.3031843324861713)},
    "agent": {"T0": dict(n_conf=0, tp=0, conf_ratio=0.0),
              "T3": dict(n_conf=3373, tp=699, conf_ratio=0.9949852507374631, prec=0.20723391639490069,
                         rec=0.20619469026548673, f1=0.20671299719059588)},
}
N_POS = 3390
#: raw/fit_map.json train_iou_at_tau_phat (the diagnostics' TRAIN IoU at the thresholds, from the FULL pass)
TRAIN_IOU = [0.6407972033154268, 0.5913196812600565, 0.16163508728265136, 0.12318963013807123,
             0.07079523081860728, 0.042740441262647096, 0.05436837267112912, 0.49187805820695857]

need_box = pytest.mark.skipif(not (BIN / "eval_final.packs.pkl").exists(),
                              reason="the diagnostics' banked packs are not on this machine (REFCV7_DIAG_BIN)")
need_map = pytest.mark.skipif(not ((BIN / "train_fitpass.acc.pt").exists() and (PKG / "code" / "diag_metrics.py").exists()),
                              reason="the diagnostics' banked cell subsample / code are not on this machine")


@pytest.fixture(scope="module")
def packs():
    with open(BIN / "eval_final.packs.pkl", "rb") as fh:
        return pickle.load(fh)


@need_box
@pytest.mark.parametrize("head", ["box3d", "agent"])
def test_FULL_box_the_declared_and_the_gated_census_equal_the_diagnostics_on_all_1061_windows(packs, head):
    pk = packs[head]
    assert len(pk) == 1061
    t0 = det.train_row_keys(pk, head)                                     # the declared 0.5 gate (the tip's census)
    lit = B_BOX[head]["T0"]
    assert (t0["%s_n_conf" % head], t0["%s_tp@gate" % head]) == (lit["n_conf"], lit["tp"])
    assert t0["%s_n_pos" % head] == N_POS
    assert t0["%s_conf_ratio" % head] == pytest.approx(lit["conf_ratio"], rel=1e-12)
    g = det.gated_census_keys(pk, head, det.load_head_gates(GATES)[0])
    lit = B_BOX[head]["T3"]
    assert g["eval_%s_gated_n_conf" % head] == lit["n_conf"] and g["eval_%s_gated_tp" % head] == lit["tp"]
    assert g["eval_%s_gated_n_pos" % head] == N_POS
    for k in ("conf_ratio", "prec", "rec", "f1"):
        assert g["eval_%s_gated_%s" % (head, k)] == pytest.approx(lit[k], rel=1e-12), k
    assert g["eval_%s_gated_conf_ratio_alarm" % head] == 0.0              # in [0.5, 1.5] at the gate
    assert det.CONF_RATIO_BAND == (0.5, 1.5)


def _import_diag_metrics():
    spec = importlib.util.spec_from_file_location("refcv7_diag_metrics", PKG / "code" / "diag_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refcv7_diag_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


@need_map
def test_FULL_map_the_class_threshold_masks_equal_the_diagnostics_on_all_4p6M_real_cells_and_the_full_histogram():
    dm = _import_diag_metrics()
    a = torch.load(BIN / "train_fitpass.acc.pt", map_location="cpu", weights_only=False)
    sub, cw = a["sub"], a["class_weight"].float()
    th, _ = H.load_class_thresholds(CFG, class_weight=cw)
    lw = dm.log_w(cw, "cpu")
    N = int(sub["y"].shape[0])
    assert N == 4599016
    dec = {"pc": ("argmax", (-lw).tolist()), "thr_phat": ("thr_phat", th.tolist())}
    pred = torch.zeros(8, dtype=torch.int64)
    gt = torch.zeros(8, dtype=torch.int64)
    inter = torch.zeros(8, dtype=torch.int64)
    for a0 in range(0, N, 500_000):
        z = sub["z"][a0:a0 + 500_000].float().T.reshape(1, 8, -1, 1).contiguous()     # fp16 logits are exact in fp32
        y = sub["y"][a0:a0 + 500_000].long().reshape(1, -1, 1)
        sup = torch.ones(y.shape, dtype=torch.bool)
        ref = dm.decision_masks(z, sup, lw, dec)                                        # the DIAGNOSTICS' code
        mine = H.decide_masks(z, cw, th, sup)
        assert torch.equal(mine, ref["thr_phat"]), a0                                   # EVERY cell
        codes = H.decide(z, "prior_corrected", cw)
        oh = torch.nn.functional.one_hot(codes, 8).permute(0, 3, 1, 2).bool()
        assert torch.equal(oh, ref["pc"]), a0                                           # the default rule too
        g = dm.onehot_masks(y, sup)
        pred += mine.sum((0, 2, 3)).to(torch.int64)
        gt += g.sum((0, 2, 3)).to(torch.int64)
        inter += (mine & g).sum((0, 2, 3)).to(torch.int64)
    # the FULL pass's histogram: fraction of ALL supervised TRAIN cells with logit(p_hat_c) >= tau_c
    Hh = a["hist"]["phat"]                                                              # [8, nb, 2, 4000]
    n_all = float(Hh[0].sum())
    assert n_all == 459796436.0
    width = (dm.HIST_HI - dm.HIST_LO) / dm.HIST_NB
    for c in range(8):
        i = int(round((th[c].item() - dm.HIST_LO) / width))
        p_full = float(Hh[c].sum(dim=0)[:, i:].sum()) / n_all
        p_sub = int(pred[c]) / N
        se = math.sqrt(max(p_full * (1.0 - p_full), 1e-12) / N)
        assert abs(p_sub - p_full) <= 5.0 * se + 1e-9, (c, p_sub, p_full, se)           # 5 binomial SEs
    # the TRAIN IoU at the thresholds (the diagnostics' fit objective) from the SAMPLE: the thin classes are noisy
    # (1,812 hatched cells), so the tolerance is loose but not blind
    iou = [int(inter[c]) / max(int(pred[c]) + int(gt[c]) - int(inter[c]), 1) for c in range(8)]
    for c, (got, want) in enumerate(zip(iou, TRAIN_IOU)):
        assert got == pytest.approx(want, abs=0.02 if c in (3, 4, 5, 6) else 0.005), (c, got, want)
    # the MEASURED-vs-nothing control: the declared rule's lane / edge / hatched predictions are ~0 on the same cells
    pc_pred = torch.zeros(8, dtype=torch.int64)
    for a0 in range(0, N, 500_000):
        z = sub["z"][a0:a0 + 500_000].float().T.reshape(1, 8, -1, 1).contiguous()
        pc_pred += torch.bincount(H.decide(z, "prior_corrected", cw).flatten(), minlength=8)
    assert int(pc_pred[6]) == 0 and int(pc_pred[5]) < int(pred[5]) // 50                # hatched never, edge ~never


#: RESULT.md sec. 1.2 (M-c) EVAL-DIAG IoU at the TRAIN-fitted thresholds (all bands, 1,112 windows / 139 episodes), 3 dp
RESULT_EVAL_IOU = {"nocls": 0.608, "drivable": 0.576, "lane": 0.164, "crosswalk": 0.105, "arrow": 0.059,
                   "edge": 0.042, "hatched": 0.062, "sidewalk": 0.500}
RESULT_EVAL_PRED_OVER_GT = {"nocls": 1.190, "drivable": 1.015, "lane": 1.061, "crosswalk": 0.654, "arrow": 0.623,
                            "edge": 1.667, "hatched": 1.319, "sidewalk": 1.325}


@pytest.mark.skipif(not ((BIN / "eval_final.acc.pt").exists() and (PKG / "code" / "diag_metrics.py").exists()),
                    reason="the diagnostics' banked EVAL histograms / code are not on this machine")
def test_FULL_map_the_SHIPPED_thresholds_reproduce_the_RESULT_eval_ious_from_the_banked_eval_histograms():
    """The thresholds in the shipped file (read by ``load_class_thresholds``) applied as ``score >= tau`` to the EVAL-DIAG
    score histogram (the diagnostics' exact sweep, 4,000 bins over [-20, 20]) give RESULT.md's EVAL IoUs and
    predicted/GT ratios -- lane 0.164, edge 0.042, hatched 0.062 ... -- i.e. the copied file IS the fit, and ``>=`` is its
    inequality. (The EVAL LOGITS are not banked, only these histograms and the TRAIN cell subsample, so this is the closest
    thing to a re-run of the EVAL score that exists without a GPU.)"""
    dm = _import_diag_metrics()
    th, _ = H.load_class_thresholds(CFG)
    e = torch.load(BIN / "eval_final.acc.pt", map_location="cpu", weights_only=False)
    hist = e["hist"]["phat"]                                                    # [8, bands, 2 (gt-neg, gt-pos), 4000]
    width = (dm.HIST_HI - dm.HIST_LO) / dm.HIST_NB
    for c, name in enumerate(H.CLASS_KEYS):
        h = hist[c].sum(dim=0).numpy()                                          # all 20 m bands pooled
        i = int(round((th[c].item() - dm.HIST_LO) / width))                     # the bin edge the threshold sits on
        assert abs(dm.HIST_LO + i * width - th[c].item()) < 1e-6, name          # tau really is on the 0.01 grid
        iou, pred, _inter, gt = dm.iou_at_index(h[0], h[1], i)
        assert round(iou, 3) == RESULT_EVAL_IOU[name], (name, iou)
        assert round(pred / gt, 3) == RESULT_EVAL_PRED_OVER_GT[name], (name, pred / gt)
