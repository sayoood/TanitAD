"""RULETRAIN (--rule-relabel): the label-fed distance rule applied to the training target of the executed plan (card PREREG_V8F_PROOF_ADD32).

Tiny models (d 32) on CPU. A test that the mechanism works is not evidence that it runs: the reach tests go from the stage to the loss and the gradient, with a control that must read exactly 0.0, and the loop line is
pinned by a source check plus the trainer run in checks/t3_mutation.py."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import dist_rule as R  # noqa: E402
from tanitad.data import rule_relabel as RRL  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

DIST_RULE_SHA256 = "40b5dd9173aeaf8c7d7a7d4bfbb52484a8f30b9a8480d3356f8fa0aee65746dc"
GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ruletrain")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    D.synthetic_lead_labels(J, 0)
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    RR = RRL.RuleRelabel("on", J, store)
    return {"J": J, "ds": ds, "wins": wins, "RR": RR, "store": store}


def _batch(world, idx):
    return D.collate([world["ds"].item(world["wins"][i][0], world["wins"][i][1], 1, 3) for i in idx])


def _eligible(world, idx):
    out = []
    for i in idx:
        b = _batch(world, [i])
        _, info = world["RR"].stage(b)
        out.append(info["n_relabel"] == 1)
    return np.array(out)


def test_rule_file_is_the_pinned_one():
    assert hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest() == DIST_RULE_SHA256


def test_flag_default_off_and_declared_in_the_gate_table():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.rule_relabel == "off" and TR.FLAG_CHECKS["rule_relabel"] == "data:rule_relabel"
    assert not RRL.RuleRelabel("off").on


def test_relabel_gt_equals_distance_rule_by_hand_and_keeps_rows_without_a_lead():
    g = np.zeros((3, 30, 2), np.float32)
    g[:, :, 0] = np.arange(1, 31)[None, :] * 2.0                         # 10 m/s straight
    gt = torch.as_tensor(g)
    ok = torch.ones(3, dtype=torch.bool)
    v0, vlog = torch.tensor([10.0, 10.0, 10.0]), torch.tensor([10.0, 10.0, 10.0])
    pres, gap, clos = np.array([1.0, 1.0, 0.0]), np.array([8.0, 80.0 / 4, np.nan]), np.array([0.0, -1.0, np.nan])
    new, info = RRL.relabel_gt(gt, ok, ok, v0, vlog, pres, gap, clos)
    assert info["n_relabel"] == 2 and not info["mask"][2]
    for r in (0, 1):
        by_hand = R.distance_rule(g[r:r + 1], np.array([10.0]), np.array([True]), gap[r:r + 1], gap[r:r + 1], np.array([max(0.0, 10.0 + clos[r])]), m=10, floor_lifts=False)
        assert torch.equal(new[r], torch.as_tensor(by_hand.plan[0]))
    assert torch.equal(new[2], gt[2]) and not torch.equal(new[0], gt[0])           # the lead at 8 m slows the plan; the row without a lead is untouched
    none, i0 = RRL.relabel_gt(gt, ok, ok, v0, vlog, pres, gap, clos, mask_override=np.zeros(3, bool))
    assert i0["n_relabel"] == 0 and torch.equal(none, gt)


def test_rows_without_gt_or_frame_or_closing_label_keep_their_target():
    g = torch.zeros(4, 30, 2)
    g[:, :, 0] = torch.arange(1, 31)[None, :] * 2.0
    ok = torch.tensor([True, False, True, True])
    fok = torch.tensor([True, True, False, True])
    new, info = RRL.relabel_gt(g, ok, fok, torch.full((4,), 10.0), torch.full((4,), 10.0), np.ones(4), np.full(4, 8.0), np.array([0.0, 0.0, 0.0, np.nan]))
    assert info["n_relabel"] == 1 and info["mask"].tolist() == [True, False, False, False]
    assert torch.equal(new[1:], g[1:])


def test_stage_on_a_world_batch_is_exact_against_the_rule_by_hand(world):
    RR, ds = world["RR"], world["ds"]
    assert not ds.lead_sup                                               # the stage reads the join itself: the dataset carries no new key
    b = _batch(world, list(range(0, min(64, len(world["wins"])), 2)))
    new, info = RR.stage(b)
    assert info["n_relabel"] > 0 and info["n_changed"] > 0
    for r in np.nonzero(info["mask"])[0]:
        ci, t = int(b["ci"][r]), int(b["t"][r])
        row = int(world["J"]["off"][ci]) + t
        d = world["store"].get(str(world["J"]["clip"][ci]))
        g, c = float(world["J"]["lead_gap"][row]), float(D.lead_sup_label_arrays(world["J"])["clos"][row])
        out = R.distance_rule(d["gt"][t][None].astype(np.float32), np.array([float(d["p10f"][t][0])]), np.array([True]), np.array([g]), np.array([g]), np.array([max(0.0, float(d["v"][t]) + c)]), m=10, floor_lifts=False)
        assert torch.equal(new[r], torch.as_tensor(out.plan[0]))
    assert torch.equal(new[~torch.as_tensor(info["mask"])], b["gt"][~torch.as_tensor(info["mask"])])


def _model(world):
    a = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *GRID])
    cfg = TR.make_config(a)
    mu, sd = M.cons_stats(world["J"])
    torch.manual_seed(0)
    return M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)


def _run(model, b):
    for p in model.parameters():
        p.grad = None
    torch.manual_seed(20261115)
    T = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
    T["total"].backward()
    return float(T["ade"]), float(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None))


def test_reach_loss_and_gradient_differ_with_a_lead_and_are_exactly_equal_without(world):
    model = _model(world)
    model.train()
    idx = list(range(len(world["wins"])))
    el = _eligible(world, idx)
    with_i, without_i = [i for i in idx if el[i]][:16], [i for i in idx if not el[i]][:16]
    assert with_i and without_i
    bA = _batch(world, with_i)
    gA, iA = world["RR"].stage(bA)
    assert iA["n_changed"] > 0
    a0, s0 = _run(model, dict(bA))
    a1, s1 = _run(model, {**bA, "gt": gA})
    assert a1 != a0 and s1 != s0
    bB = _batch(world, without_i)
    gB, iB = world["RR"].stage(bB)
    assert iB["n_relabel"] == 0 and torch.equal(gB, bB["gt"])
    assert _run(model, dict(bB)) == _run(model, {**bB, "gt": gB})            # the zero control: exactly equal loss term and gradient sum


def test_selfcheck_on_passes_with_a_count_and_off_has_no_key(tmp_path):
    for extra, on in (([], False), (["--rule-relabel", "on"], True)):
        out = tmp_path / ("on" if on else "off")
        rc = TR.main(["--synthetic", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", *GRID, *extra, "--exclude", "1:3", "--threads", "1", "--selfcheck-only", "--out", str(out)])
        assert rc == 0
        import json
        sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
        assert sc["declared_vs_built"] == "PASS"
        if on:
            assert sc["levers"]["rule_relabel"] == "on" and sc["info"]["rule_relabel_n"] > 0 and sc["rule_relabel_probe"]["pass"]
        else:
            assert not [k for k in list(sc["levers"]) + list(sc["info"]) + list(sc) if "rule_relabel" in k]


@pytest.mark.parametrize("extra", [["--ego-perturb-p", "0.5"], ["--recover-p", "0.5"]])
def test_refused_combinations(tmp_path, extra):
    with pytest.raises(SystemExit) as e:
        TR.main(["--synthetic", "--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", *GRID, "--rule-relabel", "on", *extra, "--exclude", "1:3", "--threads", "1", "--selfcheck-only", "--out", str(tmp_path / "x")])
    assert e.value.code not in (0, None)


def test_the_loop_attaches_the_relabelled_target_and_guards_it():
    src = Path(TR.__file__).read_text(encoding="utf-8")
    line = chr(10) + " " * 16 + 'b["gt"] = rr_gt' + chr(10)
    assert src.count(line) == 1 and src.count("ZZRULE-NOT-APPLIEDZZ") == 1
    assert src.index("                rr_gt, rr_i = RR.stage(b)") < src.index("                b = comp.apply(b, step)" + chr(10)) < src.index(line) < src.index("            b = to_dev(b, dev)" + chr(10))


def test_a_row_with_a_lead_and_a_nan_logged_speed_keeps_its_target_and_is_not_counted(world):
    b = _batch(world, list(range(0, min(64, len(world["wins"])), 2)))
    new0, i0 = world["RR"].stage(b)
    r = int(np.nonzero(i0["mask"])[0][0])
    ci, t = int(b["ci"][r]), int(b["t"][r])
    d = world["store"].get(str(world["J"]["clip"][ci]))
    for key in ("v", "p10f"):
        saved = d[key][t].copy() if key == "p10f" else float(d[key][t])
        d[key][t] = np.nan
        try:
            new, info = world["RR"].stage(b)
        finally:
            d[key][t] = saved
        assert torch.equal(new[r], b["gt"][r]) and not info["mask"][r] and info["n_relabel"] == i0["n_relabel"] - 1
    again, i1 = world["RR"].stage(b)
    assert torch.equal(again, new0) and i1["n_relabel"] == i0["n_relabel"]
