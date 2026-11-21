"""refcv8 WP-C F4b: the centre-distance NMS on the box heads' detections (``tanitad.eval.detection_nms``), OPT-IN.

Source of the fix: ``2026-10-04-refcv7-route-following`` (RESULT.md sec. 4; ``code/analyze_box_nms.py`` + ``code/route_metrics.py``;
``raw/box_nms.json``). What is pinned, each with a LITERAL expected value and a deliberate-regression arm that must go RED:

A. THE SUPPRESSION RULE on analytic cases: descending-score greedy, inclusive radius, slots below ``P_FLOOR`` untouched, a
   tie keeps the lower slot index, radius 0 only removes coincident centres.
B. PARITY WITH THE ROUTE PACKAGE'S OWN CODE on 118 real EVAL windows (the committed refcv7 fixture): the literals were
   computed by ``analyze_box_nms.apply_nms`` / ``rows_all`` / ``ap_weighted`` / ``census`` / ``dup_stats`` -- an
   INDEPENDENTLY AUTHORED implementation -- before this module existed; this module reproduces AP@2 m, the census at the
   re-fitted gate, the boxes per object, and the "gate must move with the NMS" claim (conf_ratio at the OLD gate is out of band).
C. THE FIT (TRAIN only): radius = argmax TRAIN AP@2 m with ties -> the smaller radius, gate re-fitted after the NMS; literals from
   the route package's code on the TRAIN fixture (box3d 2.5 m / agent 3.0 m, as on the full set).
D. DEFAULT UNCHANGED + READ-ONLY: ``nms_census_keys(.., None)`` is ``{}``; the input packs are never mutated; ``detection_metrics``
   is value-for-value unchanged by running the NMS API.
E. THE PLANNER IS UNTOUCHED: no module under ``tanitad/models``, ``tanitad/refs``, ``tanitad/train`` imports this module
   (a scanner with a planted violation that must go RED), and ``AgentTokenEmbed`` is bit-identical after the API runs.
F. THE SHIPPED FILE and the loader's refusals.
The FULL-SET acceptance (every route-package number reproduced from the banked packs) is ``test_refcv8_det_nms_acceptance_full.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.eval import detection_metrics as det
from tanitad.eval import detection_nms as N
from tanitad.refs import refc_agents as ra

HERE = Path(__file__).resolve().parent
EVAL_FIX = HERE / "fixtures" / "refcv7_infer_fixes" / "refcv7_box_packs_eval_every9th.npz"
TRAIN_FIX = HERE / "fixtures" / "refcv8_perception_fixes" / "refcv8_box_packs_train_every9th.npz"
SHIPPED = Path(N.__file__).resolve().parent.parent / "configs" / "refcv7_det_nms_train.json"
STACK = Path(N.__file__).resolve().parents[2]


def _packs(path, head):
    d = np.load(path)
    g = lambda k: d[head + "__" + k]                                                         # noqa: E731
    out = []
    for i in range(g("logit").shape[0]):
        ng, npair = int(g("n_gt")[i]), int(g("n_pair")[i])
        out.append({"ep": None, "logit": g("logit")[i], "xy": g("xy")[i], "cls": g("cls")[i],
                    "cls_corr": g("cls_corr")[i], "matched": g("matched")[i], "exempt": g("exempt")[i],
                    "pair_err": g("pair_err")[i][:npair], "pair_size_err": g("pair_size_err")[i][:npair],
                    "pair_z_err": g("pair_z_err")[i][:npair], "gt_xy": g("gt_xy")[i][:ng], "gt_cls": g("gt_cls")[i][:ng],
                    "pos": g("pos")[i][:ng], "ign": g("ign")[i][:ng], "hidden": g("hidden")[i][:ng]})
    return out


def _logit(p):
    return math.log(p / (1.0 - p))


# =========================================================================== #
# A. the rule on analytic cases                                                #
# =========================================================================== #
def test_A_descending_greedy_keeps_the_best_of_a_cluster_and_the_far_slot():
    p = np.array([0.9, 0.8, 0.7])
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    assert N.centre_nms_keep(p, xy, 1.5).tolist() == [True, False, True]
    assert N.centre_nms_keep(p, xy, 0.5).tolist() == [True, True, True]
    assert N.centre_nms_keep(p, xy, 6.0).tolist() == [True, False, False]


def test_A_the_order_is_by_score_not_by_slot_index():
    p = np.array([0.5, 0.9, 0.7])                         # the middle slot is the best, and it eats both neighbours
    xy = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    assert N.centre_nms_keep(p, xy, 1.0).tolist() == [False, True, False]


def test_A_the_radius_is_INCLUSIVE_at_the_boundary():
    p = np.array([0.9, 0.8])
    xy = np.array([[0.0, 0.0], [3.0, 4.0]])               # distance exactly 5.0
    assert N.centre_nms_keep(p, xy, 5.0).tolist() == [True, False]
    assert N.centre_nms_keep(p, xy, 4.999999).tolist() == [True, True]


def test_A_slots_below_the_floor_are_never_suppressed_and_never_suppress():
    p = np.array([0.9, 0.049, 0.05, 0.04])
    xy = np.zeros((4, 2))                                 # all four coincide
    keep = N.centre_nms_keep(p, xy, 1.0)
    # 0.9 is kept; 0.05 is AT the floor (>=) -> a candidate, suppressed by 0.9; 0.049 and 0.04 are tail -> untouched
    assert keep.tolist() == [True, True, False, True]
    # a tail slot does not suppress a candidate either: only the candidate-set is greedy
    p2 = np.array([0.049, 0.06])
    assert N.centre_nms_keep(p2, np.zeros((2, 2)), 1.0).tolist() == [True, True]


def test_A_a_score_tie_keeps_the_LOWER_slot_index():
    p = np.array([0.5, 0.5, 0.5])
    xy = np.zeros((3, 2))
    assert N.centre_nms_keep(p, xy, 1.0).tolist() == [True, False, False]


def test_A_radius_zero_removes_only_coincident_centres():
    p = np.array([0.9, 0.8, 0.7])
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [1e-6, 0.0]])
    assert N.centre_nms_keep(p, xy, 0.0).tolist() == [True, False, True]


def test_A_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        N.centre_nms_keep(np.zeros(3), np.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError):
        N.centre_nms_keep(np.zeros(3), np.zeros((3, 2)), -1.0)
    with pytest.raises(ValueError):
        N.centre_nms_keep(np.zeros(3), np.zeros((3, 2)), float("nan"))


def test_A_DELIBERATE_REGRESSION_an_exclusive_boundary_or_an_ascending_order_goes_RED():
    def exclusive(p, xy, r):
        keep, kept = [], []
        for i in np.argsort(-p, kind="mergesort"):
            if all(math.hypot(*(xy[i] - xy[j])) >= r for j in kept):               # '<' instead of '<='
                kept.append(i)
        keep = np.zeros(len(p), bool)
        keep[kept] = True
        return keep
    p, xy = np.array([0.9, 0.8]), np.array([[0.0, 0.0], [3.0, 4.0]])
    assert exclusive(p, xy, 5.0).tolist() == [True, True] != N.centre_nms_keep(p, xy, 5.0).tolist()

    def ascending(p, xy, r):                                                       # the WORST slot wins the cluster
        keep, kept = np.zeros(len(p), bool), []
        for i in np.argsort(p, kind="mergesort"):
            if all(math.hypot(*(xy[i] - xy[j])) > r for j in kept):
                kept.append(i)
        keep[kept] = True
        return keep
    p, xy = np.array([0.9, 0.5]), np.array([[0.0, 0.0], [1.0, 0.0]])
    assert ascending(p, xy, 2.0).tolist() == [False, True] != N.centre_nms_keep(p, xy, 2.0).tolist()


# =========================================================================== #
# B. parity with the route package's own code on real windows                  #
# =========================================================================== #
#: computed by the route package's analyze_box_nms.py (apply_nms mode 'centre', rows_all, ap_weighted, census, dup_stats) on the
#: SAME 118 EVAL-DIAG windows, BEFORE detection_nms existed. (radius, refit gate at 4 dp as the route package carried it)
REF = {
    "box3d": dict(r=2.5, gate=0.2145, slots_before=35400, slots_after=25568,
                  ap0=0.25817887721581306, ap1=0.37086530395989403,
                  c0=dict(n_conf=787, tp=161, n_pos=367), c1=dict(n_conf=337, tp=142, n_pos=367),
                  c1_old_gate=dict(n_conf=156, tp=98, conf_ratio=0.4250681198910082),
                  conf_ratio_new_gate=0.9182561307901907, f1_new_gate=0.4034090909090909,
                  boxes_before=1.8796, boxes_after=1.0396, ge2_before=0.6019, ge2_after=0.0396, objects_before=108, objects_after=101),
    "agent": dict(r=3.0, gate=0.1809, slots_before=35400, slots_after=21066,
                  ap0=0.1535351533582067, ap1=0.3208314020974369,
                  c0=dict(n_conf=1720, tp=165, n_pos=367), c1=dict(n_conf=375, tp=134, n_pos=367),
                  c1_old_gate=dict(n_conf=39, tp=31, conf_ratio=0.10626702997275204),
                  conf_ratio_new_gate=1.021798365122616, f1_new_gate=0.3611859838274933,
                  boxes_before=2.5, boxes_after=1.0312, ge2_before=0.725, ge2_after=0.0312, objects_before=40, objects_after=32),
}
OLD_GATE = 0.2589                 # the in-run P = R gate the route package measured the duplicates at


@pytest.mark.parametrize("head", ["box3d", "agent"])
def test_B_AP_census_and_duplicates_equal_the_route_packages_independent_numbers(head):
    ref = REF[head]
    P = _packs(EVAL_FIX, head)
    Q = N.nms_packs(P, ref["r"])
    assert sum(len(p["logit"]) for p in P) == ref["slots_before"]
    assert sum(len(q["logit"]) for q in Q) == ref["slots_after"]
    assert N.ap2m(P) == pytest.approx(ref["ap0"], rel=1e-12)
    assert N.ap2m(Q) == pytest.approx(ref["ap1"], rel=1e-12)
    c0, c1 = N._census(P, ref["gate"]), N._census(Q, ref["gate"])
    for k in ("n_conf", "tp", "n_pos"):
        assert c0[k] == ref["c0"][k] and c1[k] == ref["c1"][k], (head, k)
    assert c1["conf_ratio"] == pytest.approx(ref["conf_ratio_new_gate"], rel=1e-12)
    assert c1["f1"] == pytest.approx(ref["f1_new_gate"], rel=1e-12)
    d0, d1 = N.duplicate_stats(P, OLD_GATE), N.duplicate_stats(Q, OLD_GATE)
    assert round(d0["boxes_per_object"], 4) == ref["boxes_before"] and round(d1["boxes_per_object"], 4) == ref["boxes_after"]
    assert round(d0["frac_objects_ge2"], 4) == ref["ge2_before"] and round(d1["frac_objects_ge2"], 4) == ref["ge2_after"]
    assert d0["n_detected_objects"] == ref["objects_before"] and d1["n_detected_objects"] == ref["objects_after"]


@pytest.mark.parametrize("head", ["box3d", "agent"])
def test_B_the_gate_must_move_with_the_NMS(head):
    """RESULT.md sec. 4: at the OLD gate the NMS halves conf_ratio (out of the [0.5, 1.5] band); at the re-fitted gate it is in band."""
    ref = REF[head]
    Q = N.nms_packs(_packs(EVAL_FIX, head), ref["r"])
    old = N._census(Q, OLD_GATE)
    new = N._census(Q, ref["gate"])
    lo, hi = det.CONF_RATIO_BAND
    assert old["n_conf"] == ref["c1_old_gate"]["n_conf"] and old["tp"] == ref["c1_old_gate"]["tp"]
    assert old["conf_ratio"] == pytest.approx(ref["c1_old_gate"]["conf_ratio"], rel=1e-12)
    assert old["conf_ratio_alarm"] == 1.0 and not (lo <= old["conf_ratio"] <= hi)
    assert new["conf_ratio_alarm"] == 0.0 and lo <= new["conf_ratio"] <= hi


def test_B_DELIBERATE_REGRESSION_a_wrong_radius_or_a_class_blind_off_by_one_goes_RED():
    ref = REF["box3d"]
    P = _packs(EVAL_FIX, "box3d")
    for wrong in (0.5, 4.0):                                                   # the grid's AP-poor ends
        assert N.ap2m(N.nms_packs(P, wrong)) != pytest.approx(ref["ap1"], rel=1e-6)
    assert N.ap2m(N.nms_packs(P, 0.0)) < ref["ap1"] - 0.05                     # no suppression of near-duplicates: far lower


# =========================================================================== #
# C. the fit                                                                   #
# =========================================================================== #
#: route package code on the TRAIN fixture (120 windows): chosen radius, the TRAIN P = R gate re-fitted after it, and the grid APs
FIT = {"box3d": dict(r=2.5, gate=0.2082558274269104, ap=0.35579081262565604, ap_none=0.2565721295003559,
                     gate_none=0.2505699098110199, ap_r1=0.27985231918368947),
       "agent": dict(r=3.0, gate=0.18241560459136963, ap=0.332245110054814, ap_none=0.13500220248734945,
                     gate_none=0.2227129340171814, ap_r1=0.1939104809424558)}


@pytest.mark.parametrize("head", ["box3d", "agent"])
def test_C_the_TRAIN_fit_picks_the_radius_and_refits_the_gate_like_the_route_package(head):
    f = N.fit_nms(_packs(TRAIN_FIX, head))
    lit = FIT[head]
    assert f["radius_m"] == lit["r"] and f["n_windows"] == 120 and f["p_floor"] == N.P_FLOOR == 0.05
    assert f["gate"] == pytest.approx(lit["gate"], rel=1e-12)
    assert f["train_ap2m"] == pytest.approx(lit["ap"], rel=1e-12)
    assert f["grid"]["none"]["train_ap2m"] == pytest.approx(lit["ap_none"], rel=1e-12)
    assert f["grid"]["none"]["train_pr_gate"] == pytest.approx(lit["gate_none"], rel=1e-12)
    assert f["grid"]["1"]["train_ap2m"] == pytest.approx(lit["ap_r1"], rel=1e-12)
    assert set(f["grid"]) == {"none", "0.5", "1", "1.5", "2", "2.5", "3", "4"}
    assert f["gate"] < f["grid"]["none"]["train_pr_gate"]          # the gate moves DOWN with the NMS (fewer, better boxes)


def test_C_a_tie_in_AP_keeps_the_SMALLER_radius_and_the_choice_never_reads_a_pack_it_was_not_given():
    # one isolated object, one slot: every radius suppresses nothing -> identical AP for all radii -> the smallest wins
    pk = {"logit": np.array([2.0, -8.0], np.float32), "xy": np.array([[10.0, 0.0], [30.0, 5.0]], np.float32),
          "cls": np.zeros(2, np.int16), "cls_corr": np.zeros(2, np.int16), "matched": np.zeros(2, bool),
          "exempt": np.zeros(2, bool), "gt_xy": np.array([[10.0, 0.0]], np.float32), "gt_cls": np.zeros(1, np.int16),
          "pos": np.ones(1, bool), "ign": np.zeros(1, bool), "hidden": np.zeros(1, bool),
          "pair_err": np.zeros(0), "pair_size_err": np.zeros(0), "pair_z_err": np.zeros(0)}
    f = N.fit_nms([pk], radii=(1.0, 2.0, 3.0))
    assert f["radius_m"] == 1.0 and len({round(v["train_ap2m"], 12) for k, v in f["grid"].items() if k != "none"}) == 1


# =========================================================================== #
# D. default unchanged + read-only                                             #
# =========================================================================== #
def test_D_a_default_run_emits_no_key_and_computes_nothing():
    assert N.nms_census_keys(_packs(EVAL_FIX, "agent"), "agent", None) == {}


def test_D_the_input_packs_are_never_mutated_and_gt_arrays_pass_through_by_identity():
    P = _packs(EVAL_FIX, "box3d")[:20]
    before = copy.deepcopy(P)
    Q = N.nms_packs(P, 2.5)
    for a, b in zip(P, before):
        for k in a:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    assert all(q["gt_xy"] is p["gt_xy"] and q["pos"] is p["pos"] for p, q in zip(P, Q))
    assert any(len(q["logit"]) < len(p["logit"]) for p, q in zip(P, Q))


def test_D_the_declared_detection_metrics_are_value_for_value_unchanged_by_running_the_nms_api():
    P = _packs(EVAL_FIX, "agent")
    h0 = hashlib.sha256(json.dumps([[k, repr(v)] for k, v in sorted(det.train_row_keys(P, "agent").items())]).encode()).hexdigest()
    N.nms_census_keys(P, "agent", {"agent": {"radius_m": 3.0, "gate": 0.18}})
    h1 = hashlib.sha256(json.dumps([[k, repr(v)] for k, v in sorted(det.train_row_keys(P, "agent").items())]).encode()).hexdigest()
    assert h0 == h1
    assert det.train_row_keys(P, "agent")["agent_n_conf"] == 0.0                # the declared 0.5 gate still reads 0


def test_D_the_emitted_keys_are_exactly_the_contract_and_carry_the_refit_operating_point():
    cfg = {"box3d": {"radius_m": 2.5, "gate": 0.2145}}
    k = N.nms_census_keys(_packs(EVAL_FIX, "box3d"), "box3d", cfg)
    assert sorted(k) == sorted(N.nms_key_names("box3d")) and len(k) == 13
    assert k["eval_box3d_nms_radius_m"] == 2.5 and k["eval_box3d_nms_gate"] == 0.2145
    assert k["eval_box3d_nms_n_conf"] == 337.0 and k["eval_box3d_nms_tp"] == 142.0 and k["eval_box3d_nms_n_pos"] == 367.0
    assert k["eval_box3d_nms_ap2m"] == pytest.approx(REF["box3d"]["ap1"], rel=1e-12)
    assert k["eval_box3d_nms_conf_ratio_alarm"] == 0.0
    with pytest.raises(ValueError):
        N.nms_census_keys(_packs(EVAL_FIX, "agent"), "agent", cfg)               # no silent fallback for a missing head
    with pytest.raises(ValueError):
        N.nms_census_keys(_packs(EVAL_FIX, "agent"), "bev", cfg)


# =========================================================================== #
# E. the PLANNER is untouched                                                  #
# =========================================================================== #
def _importers_of(module_name: str, roots) -> list:
    """Python files under ``roots`` that IMPORT ``module_name`` (AST: ``import a.b.m``, ``from a.b import m``, ``from a.b.m import x``, or a
    string constant naming the module for ``importlib``). Prose that merely mentions the name - a docstring, a G-DVB reason - is not an import."""
    import ast
    hits = []
    for root in roots:
        for f in Path(root).rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names = [node.value]
                if any(n == module_name or n.endswith("." + module_name) for n in names):
                    hits.append(str(f))
                    break
    return hits


def test_E_no_planner_side_module_imports_the_nms():
    roots = [STACK / "tanitad" / "models", STACK / "tanitad" / "refs", STACK / "tanitad" / "train"]
    assert all(r.is_dir() for r in roots)
    assert _importers_of("detection_nms", roots) == []
    # prose that merely names the module (a registry reason, a docstring) is not an import and must not trip the scan
    assert _importers_of("detection_nms", [HERE / "fixtures"]) == []


def test_E_DELIBERATE_REGRESSION_a_planner_module_that_imports_the_nms_is_caught(tmp_path):
    (tmp_path / "planner_head.py").write_text("from tanitad.eval import detection_nms\n", encoding="utf-8")
    (tmp_path / "planner_head2.py").write_text("from tanitad.eval.detection_nms import centre_nms_keep\n", encoding="utf-8")
    (tmp_path / "planner_head3.py").write_text("import importlib\nimportlib.import_module('tanitad.eval.detection_nms')\n", encoding="utf-8")
    (tmp_path / "prose_only.py").write_text('REASON = "read only by detection_nms.nms_census_keys, never by the planner"\n', encoding="utf-8")
    got = sorted(Path(h).name for h in _importers_of("detection_nms", [tmp_path]))
    assert got == ["planner_head.py", "planner_head2.py", "planner_head3.py"]       # prose_only.py (a mention, not an import) is NOT a hit


def test_E_the_planner_embed_is_bit_identical_after_the_nms_api_runs():
    n = 3
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    slots = {"presence_logit": torch.tensor([[-2.0, -0.5, 0.5]]), "cls_logits": torch.randn(1, n, ra.N_AGENT_CLASSES, generator=g),
             "box": torch.tensor([[[10.0 + 5 * i, 0.5 * i, 4.5, 1.9] for i in range(n)]]),
             "yaw_vec": torch.tensor([[[0.0, 1.0]] * n]), "rates": torch.zeros(1, n, 3)}
    embed = ra.AgentTokenEmbed(d_out=24, cfg=ra.AgentSeamConfig(d_model=32, presence_hard=False, presence_gate=0.5))
    with torch.no_grad():
        t0, p0 = embed(slots)
    N.nms_census_keys(_packs(EVAL_FIX, "agent")[:10], "agent", {"agent": {"radius_m": 3.0, "gate": 0.18}})
    with torch.no_grad():
        t1, p1 = embed(slots)
    assert t0.numpy().tobytes() == t1.numpy().tobytes() and p0.tolist() == p1.tolist()
    assert "detection_nms" not in Path(ra.__file__).read_text(encoding="utf-8")


# =========================================================================== #
# F. the shipped file and the loader                                           #
# =========================================================================== #
def test_F_the_shipped_file_is_the_route_packages_TRAIN_fit_with_provenance():
    cfg, stamp = N.load_head_nms(SHIPPED)
    assert cfg["box3d"]["radius_m"] == 2.5 and cfg["agent"]["radius_m"] == 3.0
    # raw/box_nms.json train_fit_grid `train_pr_gate` (4 dp): the gate re-fitted after the NMS
    assert round(cfg["box3d"]["gate"], 4) == 0.2145 and round(cfg["agent"]["gate"], 4) == 0.1809
    assert all(v["p_floor"] == 0.05 for v in cfg.values())
    d = json.loads(SHIPPED.read_text(encoding="utf-8"))
    pv = d["provenance"]
    assert d["schema"] == N.NMS_SCHEMA == "tanitad.det_nms/1"
    assert pv["fit_split"] == "TRAIN" and pv["checkpoint_step"] == 50400 and pv["run"] == "refcv7-r101-s0"
    assert pv["checkpoint_md5"] == "d5f104ee54ba6b2861e38030b4f6fcf1"
    assert stamp["sha256"] == hashlib.sha256(SHIPPED.read_bytes()).hexdigest()
    # the grid is in the file: the choice is auditable
    assert pv["train_grid"]["box3d"]["2.5"]["train_ap2m"] > pv["train_grid"]["box3d"]["2"]["train_ap2m"]


def _doc(tmp_path, mut):
    d = json.loads(SHIPPED.read_text(encoding="utf-8"))
    mut(d)
    p = tmp_path / "n.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_F_the_loader_refuses_what_cannot_be_applied_safely(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        N.load_head_nms(tmp_path / "nope.json")
    with pytest.raises(ValueError, match="schema"):
        N.load_head_nms(_doc(tmp_path, lambda d: d.update(schema="x/1")))
    with pytest.raises(ValueError, match="no `heads`"):
        N.load_head_nms(_doc(tmp_path, lambda d: d.pop("heads")))
    with pytest.raises(ValueError, match="heads are"):
        N.load_head_nms(_doc(tmp_path, lambda d: d["heads"].pop("agent")))
    with pytest.raises(ValueError, match="heads are"):
        N.load_head_nms(_doc(tmp_path, lambda d: d["heads"].update(bev={"radius_m": 1, "gate": 0.2})))
    for bad_r in (-0.1, 20.5, float("nan")):
        with pytest.raises(ValueError, match="radius_m"):
            N.load_head_nms(_doc(tmp_path, lambda d: d["heads"]["box3d"].update(radius_m=bad_r)))
    for bad_g in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="gate"):
            N.load_head_nms(_doc(tmp_path, lambda d: d["heads"]["agent"].update(gate=bad_g)))
    with pytest.raises(ValueError, match="p_floor"):
        N.load_head_nms(_doc(tmp_path, lambda d: d["heads"]["agent"].update(p_floor=0.0)))
    with pytest.raises(ValueError, match="numeric radius_m and gate"):
        N.load_head_nms(_doc(tmp_path, lambda d: d["heads"]["box3d"].pop("radius_m")))          # a missing field is a refusal, not a KeyError
    with pytest.raises(ValueError, match="numeric radius_m and gate"):
        N.load_head_nms(_doc(tmp_path, lambda d: d["heads"]["agent"].update(gate="high")))
