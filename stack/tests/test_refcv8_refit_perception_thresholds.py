"""refcv8 WP-C item 2: the TRAIN re-fit tool for F1 (map thresholds), F4 (presence gates) and F4b (NMS radius + gate).

``stack/scripts/refit_perception_thresholds.py`` turns the TRAIN-DIAG outputs of a NEW checkpoint into the three JSON files the stack loads.
Pinned here, with LITERALS and a mutation per claim:

A. ``best_threshold`` on analytic histograms (a perfect separator reads IoU 1.0 at the separating bin; ties keep the FIRST edge).
B. MAP: the pooled TRAIN histogram of refcv7 (fixture: ``train_fitpass.acc.pt`` summed over bands) re-fits to the SHIPPED
   ``refcv7_map_hires_class_thresholds_train.json`` bit for bit - the 8 thresholds, their TRAIN IoUs and the class weights.
C. BOX: on the TRAIN fixture the gates equal ``detection_metrics.pr_equal_gate``; the NMS fit is pinned in test_refcv8_det_nms.py and, through
   the CLI, in D (literals from the route package's independent code).
D. END TO END: ``main(..)`` writes the three files, each is re-READ by the stack's own loader, and the provenance names the checkpoint, the fit
   set, the source md5 and this tool's sha256.
E. REFUSALS: a class with no positive cell, a missing head, a histogram of the wrong shape, an invocation that asks for nothing.
The full-set acceptance (the shipped files re-derived from the banked ``*.acc.pt`` / ``*.packs.pkl``) is at the bottom, skipped without the evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refit_perception_thresholds as R                       # noqa: E402
from tanitad.eval import detection_metrics as det             # noqa: E402
from tanitad.eval import detection_nms as N                   # noqa: E402
from tanitad.models import map_head_hires as H                # noqa: E402

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "refcv8_perception_fixes"
CFG = Path(N.__file__).resolve().parent.parent / "configs"
SHIPPED_MAP = CFG / "refcv7_map_hires_class_thresholds_train.json"
SHIPPED_GATES = CFG / "refcv7_det_presence_gates_train.json"
SHIPPED_NMS = CFG / "refcv7_det_nms_train.json"
TOOL = Path(R.__file__)


def _packs(head):
    d = np.load(FIX / "refcv8_box_packs_train_every9th.npz")
    g = lambda k: d[head + "__" + k]                                                        # noqa: E731
    out = []
    for i in range(g("logit").shape[0]):
        ng, npair = int(g("n_gt")[i]), int(g("n_pair")[i])
        out.append({"ep": None, "logit": g("logit")[i], "xy": g("xy")[i], "cls": g("cls")[i], "cls_corr": g("cls_corr")[i],
                    "matched": g("matched")[i], "exempt": g("exempt")[i], "pair_err": g("pair_err")[i][:npair],
                    "pair_size_err": g("pair_size_err")[i][:npair], "pair_z_err": g("pair_z_err")[i][:npair],
                    "gt_xy": g("gt_xy")[i][:ng], "gt_cls": g("gt_cls")[i][:ng], "pos": g("pos")[i][:ng], "ign": g("ign")[i][:ng],
                    "hidden": g("hidden")[i][:ng]})
    return out


def _hist():
    d = np.load(FIX / "refcv8_map_hist_train_pooled.npz")
    return d["hist_phat"], d["class_weight"]


# =========================================================================== #
# A. the sweep on analytic histograms                                          #
# =========================================================================== #
def test_A_a_perfect_separator_reads_IoU_1_at_the_separating_bin():
    neg = np.zeros(R.HIST_NB)
    pos = np.zeros(R.HIST_NB)
    neg[:2500] = 7.0                      # GT-negative cells all score below bin 2500
    pos[2500:2600] = 3.0                  # GT-positive cells all score in bins 2500..2599
    i, iou = R.best_threshold(neg, pos)
    assert i == 2500 and iou == 1.0
    assert R.hist_edges()[2500] == pytest.approx(5.0, abs=1e-9)      # -20 + 40/4000 * 2500 = 5.0 (a logit)


def test_A_ties_keep_the_FIRST_edge_and_the_overlap_case_has_a_hand_value():
    neg = np.zeros(R.HIST_NB)
    pos = np.zeros(R.HIST_NB)
    pos[100] = 10.0                       # 10 positives in bin 100, 10 negatives ALSO in bin 100
    neg[100] = 10.0
    i, iou = R.best_threshold(neg, pos)   # mask {bin >= i}: for i <= 100 it is the same set: inter 10, pred 20, union 20 -> 0.5
    assert i == 0 and iou == 0.5          # ties -> the first index
    j, iou2 = R.best_threshold(np.zeros(R.HIST_NB), pos)
    assert j == 0 and iou2 == 1.0


def test_A_MUTATION_swapping_negatives_and_positives_gives_a_different_threshold():
    h, _ = _hist()
    lane = 2
    i, _ = R.best_threshold(h[lane, 0], h[lane, 1])
    j, _ = R.best_threshold(h[lane, 1], h[lane, 0])
    assert i != j


# =========================================================================== #
# B. MAP -- the shipped file re-fits bit for bit                               #
# =========================================================================== #
def test_B_the_pooled_TRAIN_histogram_refits_to_the_shipped_thresholds_bit_for_bit():
    h, cw = _hist()
    fit = R.refit_map_thresholds(h, cw)
    ship = json.loads(SHIPPED_MAP.read_text(encoding="utf-8"))
    assert fit["tau_phat_logit"] == ship["tau_phat_logit"]
    assert fit["tau_phat_prob"] == ship["tau_phat_prob"]
    assert fit["train_iou_at_tau_phat"] == ship["provenance"]["train_iou_at_tau_phat"]
    assert fit["class_weight_values"] == ship["class_weight_values"]
    # literals (RESULT.md sec. 1.2): tau on p_hat 0.130 lane, 0.106 crosswalk, 0.054 arrow, 0.041 edge, 0.044 hatched
    assert [round(p, 3) for p in fit["tau_phat_prob"][2:7]] == [0.130, 0.106, 0.054, 0.041, 0.044]
    assert fit["tau_phat_logit"][5] == pytest.approx(-3.14, abs=1e-9)                         # edge
    # and the 4-D (per-band) and 3-D (pooled) spellings agree - with the counts split UNEVENLY over three bands (band 0 holds only the
    # even score bins), so a fit that read one band instead of summing them cannot pass (it would put the edge-class threshold elsewhere)
    b0 = h.copy()
    b0[..., 1::2] = 0
    four = np.stack([b0, h - b0, np.zeros_like(h)], axis=1)
    assert four.shape == (8, 3, 2, R.HIST_NB) and (four.sum(axis=1) == h).all()
    assert R.refit_map_thresholds(four, cw)["tau_phat_logit"] == fit["tau_phat_logit"]
    assert R.refit_map_thresholds(four[:, :1], cw)["tau_phat_logit"] != fit["tau_phat_logit"]      # one band alone is another fit


def test_B_MUTATION_a_histogram_from_another_model_refits_to_other_thresholds():
    h, cw = _hist()
    shifted = np.roll(h, 37, axis=2)                                   # every score moved +37 bins (+0.37 logit)
    assert R.refit_map_thresholds(shifted, cw)["tau_phat_logit"] != R.refit_map_thresholds(h, cw)["tau_phat_logit"]


# =========================================================================== #
# C. BOX -- gates and NMS                                                      #
# =========================================================================== #
#: pr_equal_gate of the TRAIN fixture (120 windows), the route package's independent run ('none' row)
GATE_FIXTURE = {"box3d": 0.2505699098110199, "agent": 0.2227129340171814}
#: ... and the NMS fit on it (see test_refcv8_det_nms.py::FIT)
NMS_FIXTURE = {"box3d": (2.5, 0.2082558274269104), "agent": (3.0, 0.18241560459136963)}


def test_C_refit_gates_equals_the_TRAIN_PR_gate_and_is_not_the_declared_half():
    packs = {h: _packs(h) for h in det.HEADS}
    fit = R.refit_gates(packs)
    for h in det.HEADS:
        assert fit["gates"][h] == pytest.approx(GATE_FIXTURE[h], rel=1e-12)
        assert fit["gates"][h] != 0.5 and fit["train_n_pos"][h] == 357
        p, r = fit["train_precision_recall_at_gate"][h]
        assert p <= r                                                  # the P = R point is the first rank whose precision <= recall


# (the NMS radius / gate fit is pinned by test_refcv8_det_nms.py::test_C and, through the CLI, by D below)


# =========================================================================== #
# D. end to end                                                                #
# =========================================================================== #
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    d = tmp_path_factory.mktemp("refit")
    pk = d / "train.packs.pkl"
    with open(pk, "wb") as fh:
        pickle.dump({h: _packs(h) for h in det.HEADS}, fh)
    h, cw = _hist()
    acc = d / "train.acc.pt"
    torch.save({"hist": {"phat": torch.from_numpy(h)[:, None]}, "class_weight": torch.from_numpy(cw), "episodes": list(range(139))}, acc)
    args = ["--run", "unit-test-run", "--checkpoint-step", "123", "--checkpoint-md5", "0" * 32, "--fit-set", "fixture 120 windows",
            "--box-train-packs", str(pk), "--out-gates", str(d / "gates.json"), "--nms", "--out-nms", str(d / "nms.json"),
            "--map-train-acc", str(acc), "--out-map", str(d / "map.json")]
    assert R.main(args) == 0
    return d, pk, acc


def test_D_the_three_files_are_written_and_READ_BACK_by_the_stacks_own_loaders(e2e):
    d, pk, acc = e2e
    gates, gs = det.load_head_gates(d / "gates.json")
    nms, ns = N.load_head_nms(d / "nms.json")
    th, ms = H.load_class_thresholds(d / "map.json", class_weight=torch.from_numpy(_hist()[1]))
    assert gates == pytest.approx(GATE_FIXTURE, rel=1e-12)
    assert {h: (v["radius_m"], v["gate"]) for h, v in nms.items()} == pytest.approx(NMS_FIXTURE, rel=1e-12)
    assert th.tolist() == pytest.approx(json.loads(SHIPPED_MAP.read_text(encoding="utf-8"))["tau_phat_logit"], abs=1e-6)
    assert len(gs["sha256"]) == len(ns["sha256"]) == len(ms["sha256"]) == 64


def test_D_provenance_names_the_checkpoint_the_fit_set_the_source_md5_and_the_tool(e2e):
    d, pk, acc = e2e
    for name, src in (("gates", pk), ("nms", pk), ("map", acc)):
        pv = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))["provenance"]
        assert pv["run"] == "unit-test-run" and pv["checkpoint_step"] == 123 and pv["checkpoint_md5"] == "0" * 32
        assert pv["fit_split"] == "TRAIN" and pv["fit_set"] == "fixture 120 windows"
        assert pv["source_file"] == src.name and pv["source_md5"] == hashlib.md5(src.read_bytes()).hexdigest()
        assert pv["tool_sha256"] == hashlib.sha256(TOOL.read_bytes()).hexdigest()
    nm = json.loads((d / "nms.json").read_text(encoding="utf-8"))
    assert set(nm["provenance"]["train_grid"]["box3d"]) == {"none", "0.5", "1", "1.5", "2", "2.5", "3", "4"}      # the choice is auditable
    assert json.loads((d / "map.json").read_text(encoding="utf-8"))["provenance"]["n_episodes"] == 139


def test_D_the_written_map_file_has_the_shipped_schema_and_key_set(e2e):
    d, *_ = e2e
    mine = json.loads((d / "map.json").read_text(encoding="utf-8"))
    ship = json.loads(SHIPPED_MAP.read_text(encoding="utf-8"))
    assert set(mine) == set(ship) and mine["schema"] == ship["schema"] and mine["classes"] == ship["classes"]
    assert mine["tau_phat_logit"] == ship["tau_phat_logit"] and mine["class_weight_values"] == ship["class_weight_values"]


def test_D_the_written_gates_and_nms_files_have_the_shipped_key_sets(e2e):
    d, *_ = e2e
    for name, ship in (("gates", SHIPPED_GATES), ("nms", SHIPPED_NMS)):
        mine = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
        s = json.loads(ship.read_text(encoding="utf-8"))
        assert set(mine) == set(s) and mine["schema"] == s["schema"], name
        assert set(mine["provenance"]) >= {"run", "checkpoint_step", "checkpoint_md5", "fit_split", "fit_set", "source_md5", "tool_sha256"}


# =========================================================================== #
# E. refusals                                                                  #
# =========================================================================== #
def test_E_a_class_with_no_positive_cell_is_refused_not_given_an_arbitrary_threshold():
    h, cw = _hist()
    bad = h.copy()
    bad[6, 1] = 0                                                      # no GT-hatched cell in the 'TRAIN' histogram
    with pytest.raises(ValueError, match="hatched"):
        R.refit_map_thresholds(bad, cw)


def test_E_bad_shapes_and_weights_are_refused():
    h, cw = _hist()
    with pytest.raises(ValueError, match="histogram shape"):
        R.refit_map_thresholds(h[:, :, :100], cw)
    with pytest.raises(ValueError, match="class_weight"):
        R.refit_map_thresholds(h, cw[:7])
    with pytest.raises(ValueError, match="class_weight"):
        R.refit_map_thresholds(h, np.zeros(8, np.float32))


def test_E_a_missing_head_or_no_positive_ranked_is_refused():
    with pytest.raises(ValueError, match="no TRAIN packs"):
        R.refit_gates({"box3d": _packs("box3d")})
    with pytest.raises(ValueError, match="no TRAIN packs"):
        R.refit_nms({"box3d": _packs("box3d"), "agent": []})


def test_E_the_cli_refuses_an_invocation_that_asks_for_nothing_or_half_of_it(tmp_path):
    base = ["--run", "r", "--checkpoint-step", "1", "--checkpoint-md5", "0" * 32, "--fit-set", "x"]
    with pytest.raises(SystemExit):
        R.main(base)                                                                    # nothing to fit
    with pytest.raises(SystemExit):
        R.main(base + ["--box-train-packs", str(tmp_path / "p.pkl")])                   # packs but no output
    with pytest.raises(SystemExit):
        R.main(base + ["--box-train-packs", str(tmp_path / "p.pkl"), "--out-gates", str(tmp_path / "g.json"), "--nms"])   # --nms w/o --out-nms
    with pytest.raises(SystemExit):
        R.main(base + ["--map-train-acc", str(tmp_path / "a.pt")])                      # acc but no output


# =========================================================================== #
# FULL-SET acceptance: the shipped files re-derived from the banked TRAIN outputs                          #
# =========================================================================== #
DIAG = Path(os.environ.get("REFCV7_DIAG_BIN", "D:/refcv7_diag_bin/2026-10-04"))
ROUTE = Path(os.environ.get("REFCV7_ROUTE_BIN", "D:/refcv7_route_bin/2026-10-04"))


@pytest.mark.skipif(not (DIAG / "train_fitpass.acc.pt").exists(), reason="the diagnostics' banked TRAIN accumulators are not on this machine")
def test_FULL_map_refit_from_train_fitpass_acc_reproduces_the_shipped_thresholds_bit_for_bit():
    acc = torch.load(DIAG / "train_fitpass.acc.pt", map_location="cpu", weights_only=False)
    fit = R.refit_map_thresholds(acc["hist"]["phat"].numpy(), acc["class_weight"].numpy())
    ship = json.loads(SHIPPED_MAP.read_text(encoding="utf-8"))
    assert fit["tau_phat_logit"] == ship["tau_phat_logit"] and fit["tau_phat_prob"] == ship["tau_phat_prob"]
    assert fit["train_iou_at_tau_phat"] == ship["provenance"]["train_iou_at_tau_phat"]
    assert fit["class_weight_values"] == ship["class_weight_values"]


@pytest.mark.skipif(not (DIAG / "train_fitpass.packs.pkl").exists(), reason="the diagnostics' banked TRAIN packs are not on this machine")
def test_FULL_gates_refit_from_train_fitpass_packs_reproduces_the_shipped_F4_gates_exactly():
    with open(DIAG / "train_fitpass.packs.pkl", "rb") as fh:
        packs = pickle.load(fh)
    fit = R.refit_gates(packs)
    ship = json.loads(SHIPPED_GATES.read_text(encoding="utf-8"))
    assert fit["gates"] == ship["gates"]
    assert fit["train_precision_recall_at_gate"] == ship["provenance"]["train_precision_recall_at_gate"]
    assert fit["train_n_pos"]["box3d"] == ship["provenance"]["train_n_pos"]["box3d"] == 3164


@pytest.mark.skipif(not (ROUTE / "train_s0.packs.pkl").exists(), reason="the route package's banked TRAIN packs are not on this machine")
def test_FULL_gates_refit_on_the_route_pass_agree_with_the_diagnostics_pass_to_1e_6():
    """Two forwards of the SAME checkpoint (route pass vs diagnostics pass) differ only by float noise: the gates agree to 1e-6."""
    with open(ROUTE / "train_s0.packs.pkl", "rb") as fh:
        packs = pickle.load(fh)
    fit = R.refit_gates(packs)
    ship = json.loads(SHIPPED_GATES.read_text(encoding="utf-8"))["gates"]
    for h in det.HEADS:
        assert fit["gates"][h] == pytest.approx(ship[h], abs=1e-6)
