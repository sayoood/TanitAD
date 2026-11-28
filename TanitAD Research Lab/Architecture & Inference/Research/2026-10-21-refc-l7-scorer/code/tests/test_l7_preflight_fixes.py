"""Tests for the 2026-10-22 real-capture preflight fixes (train_l7.py / l7_targets_pai.py).  Every test states the REAL defect it pins and
carries a deliberate-regression arm (a reimplementation of the old behaviour that must go RED).  No optimizer is ever constructed here.

    python -m pytest tests/test_l7_preflight_fixes.py -q      (PYTHONPATH must put a stack that has tanitad.refs.refcv7_oracle first)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import refc_scorer as RS  # noqa: E402
import train_l7 as TL  # noqa: E402
import l7_targets_pai as TP  # noqa: E402

T = np.asarray(TP.SLOT_T)


# ------------------------------------------------------------------------------------------------ masks
def test_effective_keep_is_the_e9_masks_and_not_reach_alone():
    """Real defect: the first trainer masked with reach_keep alone (149.0/149 on the real capture) while the shipped pick also applies the
    set-speed ceiling (removes 15-17 candidates from 66 % of rows) and the emit mask, each with its empty-row fallback."""
    reach = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], bool)
    ceil = np.array([[1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]], bool)
    emit = np.array([[1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0]], bool)
    got = TL.effective_keep(reach, ceil, emit)
    want = np.array([[1, 1, 0, 0, 1, 1],       # reach, minus ceiling idx 2, minus emit idx 3
                     [1, 1, 1, 1, 1, 1],       # dead ceiling row -> reach-only fallback
                     [0, 0, 0, 1, 1, 1],       # reach {2..5} minus ceiling idx 2
                     [0, 0, 0, 1, 1, 0]], bool)  # reach AND ceiling AND emit is empty -> the emit mask alone (refc_v3.py e9_rank fallback)
    assert np.array_equal(got, want), got.astype(int).tolist()
    assert not np.array_equal(got, reach), "regression arm: reach alone is NOT the shipped mask"
    # the feature mask stays reach (integration.diff passes out["reach_keep"] at inference): to_batch must not swap it
    fanz, tg = TL.synthetic_bank(8, 0)
    fanz["ceil_keep"] = np.zeros_like(fanz["reach_keep"])
    fanz["ceil_keep"][:, :10] = 1
    fanz["r8_emit_keep"] = np.ones_like(fanz["reach_keep"])
    fanz["keep"] = TL.effective_keep(fanz["reach_keep"], fanz["ceil_keep"], fanz["r8_emit_keep"])
    inp_a, _, keep = TL.to_batch(fanz, tg, np.arange(4), "F")
    assert int(keep.sum(1).max()) == 10
    fanz2 = dict(fanz)
    fanz2["keep"] = np.ones_like(fanz["keep"])
    inp_b, _, _ = TL.to_batch(fanz2, tg, np.arange(4), "F")
    assert torch.equal(inp_a["model_feats"], inp_b["model_feats"]), "the loss mask must not leak into the mode-F features"


# ------------------------------------------------------------------------------------------------ AUC
def test_auc_literals():
    y = np.array([0, 0, 1, 1])
    assert TL.auc_binary(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert TL.auc_binary(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert TL.auc_binary(np.zeros(4), y) == 0.5                                   # all ties
    assert abs(TL.auc_binary(np.array([0.1, 0.4, 0.35, 0.8]), y) - 0.75) < 1e-12   # the textbook 0.75 example
    assert np.isnan(TL.auc_binary(np.arange(3.0), np.zeros(3)))


# ------------------------------------------------------------------------------------------------ several banks
def _write_bank(d: str, seed: int, n: int = 48):
    fanz, tg = TL.synthetic_bank(n, 0)        # the SAME windows for every seed (two inference seeds of one window set)
    rng = np.random.default_rng(seed)
    fan = fanz["anchor_traj"] + rng.normal(size=fanz["anchor_traj"].shape).astype(np.float32) * 0.01
    ceil = np.ones((n, 149), np.float32)
    ceil[:, 100:] = 0
    os.makedirs(os.path.join(d, "dump"), exist_ok=True)
    np.savez(os.path.join(d, "dump", "fan_extras.npz"), anchor_traj=fan.astype(np.float32), sel_score=fanz["sel_score"],
             refined_logits=fanz["refined_logits"], reach_keep=fanz["reach_keep"].astype(bool), ceil_keep=ceil.astype(bool),
             r8_emit_keep=np.ones((n, 149), bool), sel_idx=fanz["sel_idx"], clip_index=fanz["clip_index"], ws=fanz["ws"],
             v0=fanz["v0"], inference_seed=np.array([seed]))
    np.savez(os.path.join(d, "targets.npz"), v0=fanz["v0"], **tg)
    return os.path.join(d, "dump", "fan_extras.npz"), os.path.join(d, "targets.npz")


def test_two_banks_concatenate_and_validation_keeps_both_seeds_together():
    """Real need: the PILOT is TWO inference seeds of the same windows; the first loader took one bank."""
    with tempfile.TemporaryDirectory() as tmp:
        f0, t0 = _write_bank(os.path.join(tmp, "s0"), 0)
        f1, t1 = _write_bank(os.path.join(tmp, "s1"), 1)
        fanz, tg = TL.load_bank([f0, f1], [t0, t1])
        assert len(fanz["ws"]) == 96 and fanz["inference_seeds"] == [0, 1]
        assert fanz["keep"].sum(1).max() == 100 and fanz["keep"].sum(1).min() == 100        # ceiling applied, 100 of 149
        eid = np.asarray(tg["eid"]).astype(str)
        tr, va = TL.episode_split(eid, 0.2, 20261021)
        assert not (set(eid[tr]) & set(eid[va]))
        for e in set(eid[va]):                                                                # both seeds of a val episode are val
            assert set(fanz["bank_of_row"][eid == e]) == {0, 1} and set(np.where(eid == e)[0]) <= set(va)
        # evaluate() runs on the two-bank bank without any optimizer, with finite loss and a head_auc dict
        torch.manual_seed(0)
        m = RS.L7Scorer(RS.L7Config(mode="F"))
        before = [p.detach().clone() for p in m.parameters()]
        ev = TL.evaluate(m, fanz, tg, va, "F")
        assert np.isfinite(ev["loss"]) and np.isfinite(ev["pick_cost_m"]) and "comf" in ev["head_auc"]
        assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
        # an unpaired call is refused
        try:
            TL.load_bank([f0, f1], [t0])
        except SystemExit:
            pass
        else:
            raise AssertionError("fan/targets count mismatch was accepted")


# ------------------------------------------------------------------------------------------------ provenance guard
def _fake_dump(root: str, name: str, manifest_corpus=None, roll=None):
    d = os.path.join(root, name, "dump")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "fan_extras.npz"), "wb").close()
    if manifest_corpus is not None:
        json.dump({"corpus": manifest_corpus}, open(os.path.join(d, "manifest.json"), "w"))
    if roll is not None:
        json.dump(roll, open(os.path.join(root, name, "fan_roll_s0.json"), "w"))
    return os.path.join(d, "fan_extras.npz")


def _first_version_guard(fan_npz: str) -> bool:
    """The FIRST trainer's refusal, verbatim in logic: True = would refuse.  Reads only a ``windows_json`` key of a roll json beside the dump."""
    for cand in (os.path.join(os.path.dirname(fan_npz), "..", "fan_roll_s1.json"), os.path.join(os.path.dirname(fan_npz), "..", "fan_roll_s0.json")):
        if os.path.exists(cand):
            w = json.load(open(cand, encoding="utf-8")).get("windows_json", "")
            if "eval139" in w or "windows_s" in os.path.basename(w):
                return True
    return False


def test_provenance_guard_fires_on_the_real_layout_and_fails_closed():
    """Real defect: the capture wrapper's roll json has NO ``windows_json`` key (mode + child_argv instead) and the archive copy carries no roll
    json at all, so the first guard could never fire; a bank with no provenance passed silently."""
    real_eval_roll = {"mode": "eval139", "child_argv": ["/x/roll_seed_r8_pf.py", "--seed", "0", "--windows-json", "/x/windows_s1.json", "--os-only"]}
    real_train_roll = {"mode": "train139", "child_argv": ["/x/roll_seed_r8_train_pf.py", "--seed", "0", "--dump-dir", "/x/dump"]}
    eval_corpus = {"episodes": "/data/refcv6-b1-416x1024-eval139", "labels_path": "/data/labels/s2_labels_v8_eval.jsonl.gz"}
    train_corpus = {"episodes": "/x/l7cap/a6cache", "labels_path": "/data/labels/s2_labels_v8_train.jsonl.gz"}
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "eval_roll_only": (_fake_dump(tmp, "a", None, real_eval_roll), "eval"),
            "eval_manifest_only": (_fake_dump(tmp, "b", eval_corpus, None), "eval"),
            "train_both": (_fake_dump(tmp, "c", train_corpus, real_train_roll), "train"),
            "train_manifest_only": (_fake_dump(tmp, "d", train_corpus, None), "train"),
            "nothing": (_fake_dump(tmp, "e", None, None), "unknown"),
            "conflict_eval_wins": (_fake_dump(tmp, "f", eval_corpus, real_train_roll), "eval"),
        }
        for k, (p, want) in cases.items():
            got = TL._bank_provenance(p)["kind"]
            assert got == want, f"{k}: {got} != {want}"
        # deliberate-regression arm: the first version's guard does NOT refuse the real eval roll json
        assert _first_version_guard(cases["eval_roll_only"][0]) is False, "the old guard would have caught it -> this test has no teeth"
        assert _first_version_guard(cases["eval_manifest_only"][0]) is False


# ------------------------------------------------------------------------------------------------ target builder
def _synthetic_dump(root: str):
    """4 windows x 3 candidates, 2 s grid (no S6).  Candidate 0 = the human path (straight 10 m/s), 1 = 10 m to the left, 2 = slower."""
    gt8 = np.stack([10.0 * T, np.zeros(8)], -1)                                         # the human path, 10 m/s straight, all 8 slots
    fan8 = np.stack([gt8, gt8 + np.array([0.0, 10.0]), gt8 * 0.5]).astype(np.float32)   # [3,8,2]
    gt = gt8[:4]
    W = 4
    d = os.path.join(root, "dump")
    os.makedirs(d, exist_ok=True)
    fan = np.repeat(fan8[None], W, 0)
    sel = np.zeros(W, np.int64)
    ci, ws = np.zeros(W, np.int64), np.arange(W, dtype=np.int64)
    np.savez(os.path.join(d, "fan_extras.npz"), clip_index=ci, ws=ws, anchor_traj=fan, sel_idx=sel)
    np.savez(os.path.join(d, "ep000.npz"), g=np.repeat(gt[None], W, 0).astype(np.float32), v0=np.full(W, 10.0, np.float32),
             os=fan[np.arange(W), sel, :4], ws=ws, eid=np.array([0]), clip_index=np.array([0]))
    A = 2
    fut = np.zeros((W, 8, A, 5), np.float32)
    valid = np.zeros((W, 8, A), bool)
    lab = np.ones((W, 8), bool)
    for w in (0, 1):                                                                   # windows 0, 1: a 4 x 2 m box ON the human path at every slot
        for k in range(8):
            fut[w, k, 0] = [10.0 * T[k], 0.0, 4.0, 2.0, 0.0]
            valid[w, k, 0] = True
    lab[1, :4] = False                                                                 # window 1: the same agent, but its 2 s labels are MISSING
    # window 2: labelled, NO agents.  window 3: labelled, no agents, map_label False
    drv = np.full((W, 120, 64), 255, np.uint8)
    drv[:, :, 48:] = 0                                                                 # y >= 8 m is off-road (column = (y + 16) / 0.5)
    seen = np.ones((W, 120, 64), bool)
    mlab = np.array([True, True, True, False])
    np.savez(os.path.join(root, "labels.npz"), clip_index=ci, ws=ws, r7_agent_fut=fut, r7_agent_fut_valid=valid, r7_agent_fut_label=lab,
             map_drivable_u8=drv, map_seen=seen, map_label=mlab)
    return d, os.path.join(root, "labels.npz")


def test_builder_abstains_on_unlabelled_windows_and_builds_dac():
    """Real defects, MEASURED on the real eval139 s1 subset (24 windows, 12 with incomplete labels): the first builder gave nc_mask True and
    NC mean 0.94 on 12/12 unlabelled windows (the oracle's own mask is True whenever agents are PASSED), and ignored the map arrays so dac
    abstained everywhere."""
    from tanitad.refs import refcv7_oracle as O
    with tempfile.TemporaryDirectory() as tmp:
        d, lab = _synthetic_dump(tmp)
        res, ctl, _ = TP.build(d, None, lab, None, None, chunk=64)
        assert all(v.get("pass") is not False for v in ctl.values()), ctl
        nc, m = res["nc"], res["nc_mask"]
        # window 0: labelled, agent on the human path -> candidate 0 collides (NC 0), candidate 1 (10 m left) does not
        assert m[0].all() and nc[0, 0] == 0.0 and nc[0, 1] == 1.0
        # window 1: SAME agents but unlabelled -> abstain on every candidate (the old builder: mask True and candidate 0 reads whatever)
        assert not m[1].any() and not res["ttc_mask"][1].any()
        # window 2: labelled and CLEAR -> mask True, NC 1 everywhere (a legitimate "no collision", not an abstention)
        assert m[2].all() and (nc[2] == 1.0).all()
        # dac: candidate 1 sits at y = 10 m (off-road), candidate 0 / 2 on the road; window 3 has map_label False -> abstain
        assert res["dac_mask"][:3].all() and not res["dac_mask"][3].any()
        assert res["dac"][0, 0] == 1.0 and res["dac"][0, 1] == 0.0 and res["dac"][0, 2] == 1.0
        # deliberate-regression arm: the oracle ALONE (what the first builder returned) marks the unlabelled window IN
        with np.load(lab) as A:
            fut, val = A["r7_agent_fut"][1:2, :4].copy(), A["r7_agent_fut_valid"][1:2, :4].copy()
        o = O.oracle_subscores(torch.zeros(1, 3, 4, 2), torch.tensor(T[:4], dtype=torch.float32), torch.tensor([10.0]), torch.zeros(1, 4, 2),
                               agents=torch.from_numpy(fut).permute(0, 2, 1, 3),
                               agents_valid=torch.from_numpy(val).permute(0, 2, 1))
        assert bool(o["nc_mask"].all()), "the oracle itself no longer marks an unlabelled window IN -> the builder's override is not needed"


def test_builder_chunking_and_agent_pruning_are_exact():
    """The first builder passed all windows (x 397 padded agent columns) to the oracle in ONE call: ~34 MB/window of NC/TTC distances,
    ~165 GB for a 4,855-window bank.  Chunking and the invalid-column pruning must change nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        d, lab = _synthetic_dump(tmp)
        a, _, _ = TP.build(d, None, lab, None, None, chunk=64)
        b, _, _ = TP.build(d, None, lab, None, None, chunk=1)
        c, _, _ = TP.build(d, None, lab, None, None, chunk=3)
        for k in a:
            if k == "eid":
                continue
            assert np.array_equal(a[k], b[k]) and np.array_equal(a[k], c[k]), k


def test_builder_without_labels_still_abstains_on_nc_ttc_dac():
    with tempfile.TemporaryDirectory() as tmp:
        d, _ = _synthetic_dump(tmp)
        res, _, _ = TP.build(d, None, None, None, None)
        assert not res["nc_mask"].any() and not res["ttc_mask"].any() and not res["dac_mask"].any()
        assert res["comf_mask"].all() and res["ep_mask"].all()


def test_labels_file_with_agents_but_no_label_array_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        d, lab = _synthetic_dump(tmp)
        with np.load(lab) as Z:
            A = {k: Z[k] for k in Z.files}
        A.pop("r7_agent_fut_label")
        bad = os.path.join(tmp, "bad.npz")
        np.savez(bad, **A)
        try:
            TP.build(d, None, bad, None, None)
        except SystemExit:
            return
        raise AssertionError("an agents file without its _label array was accepted (every window would score NC = 1)")


def test_best_pt_format_round_trips_through_load_scorer():
    """train_l7 writes {"cfg": vars(L7Config), "state_dict", "epoch"}; RS.load_scorer must read it back bit-exactly (random init, no optimizer)."""
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0)
        cfg = RS.L7Config(mode="F")
        m = RS.L7Scorer(cfg)
        pth = os.path.join(tmp, "best.pt")
        torch.save({"cfg": vars(cfg), "state_dict": m.state_dict(), "epoch": 0}, pth)
        m2 = RS.load_scorer(pth)
        fanz, tg = TL.synthetic_bank(6, 0)
        inp, _, _ = TL.to_batch(fanz, tg, np.arange(6), "F")
        with torch.no_grad():
            a, b = m.eval()(**inp), m2(**inp)
        assert all(torch.equal(a[k], b[k]) for k in a)
        assert not any(p.requires_grad for p in m2.parameters())


if __name__ == "__main__":
    fails = []
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fails.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    if fails:
        print("ZZL7FIXTEST_FAIL " + " | ".join(fails))
        raise SystemExit(1)
    print(f"ZZL7FIXTEST_PASS {len(tests)}")
