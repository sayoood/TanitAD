"""Tests of qhead_fit.py (card ADD42, controls h4 and h6). Tiny synthetic data; every temp file goes under pytest's basetemp."""
import json
import os

import numpy as np
import pytest

import qhead_testlib as TL  # noqa: E402  (sets the source path first)
import qhead_fit as F  # noqa: E402
import qhead_model as QM  # noqa: E402
import torch  # noqa: E402

FAST = dict(steps=3000, batch=64, lr=1e-3, wd=1e-2, warmup=50, clip_grad=1.0)


def naive_history(ci, t, hist):
    key = {(int(c), int(g)): i for i, (c, g) in enumerate(zip(ci, t))}
    idx = np.full((len(t), hist), -1, np.int64)
    for i, (c, g) in enumerate(zip(ci, t)):
        row = [key.get((int(c), int(g) - (hist - 1 - j)), -1) for j in range(hist)]
        if all(r >= 0 for r in row):
            idx[i] = row
    return idx


def split_rows(P, R, n_held):
    held = R["clip"] >= (len(P["states"]["clips"]) - n_held)
    return held


def test_history_index_equals_naive_with_gaps():
    rng = np.random.RandomState(1)
    ci = np.repeat(np.arange(5), 30)
    t = np.tile(np.arange(30), 5)
    keep = rng.rand(len(t)) > 0.15                                   # missing frames break histories
    ci, t = ci[keep], t[keep]
    idx, ok = F.history_index(ci, t, 5)
    assert np.array_equal(idx, naive_history(ci, t, 5))
    assert np.array_equal(ok, (idx >= 0).all(1)) and ok.sum() > 0 and (~ok).sum() > 0


def test_build_rows_selection_and_counts():
    P = TL.planted(n_clips=6, T=30)
    R = F.build_rows(P["states"], P["pack"], 5)
    g, pr = P["pack"]["gap"], P["pack"]["pres"]
    t = P["states"]["t"]
    want = np.flatnonzero((t >= 4) & (pr == 1) & np.isfinite(g))
    assert np.array_equal(R["row"], want)
    assert R["counts"]["rows_used"] == len(want) and R["counts"]["with_history"] == int((t >= 4).sum())
    assert np.allclose(R["y"], g[want])
    assert np.array_equal(R["hist_idx"][:, -1], want)                # the end frame is the row itself
    pack_bad = dict(P["pack"], clips=P["pack"]["clips"][::-1].copy())
    with pytest.raises(SystemExit, match="manifest order"):
        F.build_rows(P["states"], pack_bad, 5)


def test_pinball_known_values():
    pred = torch.tensor([1.0, 1.0])
    y = torch.tensor([3.0, 0.0])
    # e = [2, -1]: q 0.2 -> max(0.4, -1.6)=0.4 ; max(-0.2, 0.8)=0.8 -> mean 0.6
    assert abs(float(F.pinball(pred, y, 0.2)) - 0.6) < 1e-7
    assert abs(float(F.pinball(pred, y, 0.5)) - 0.5 * (2 + 1) / 2) < 1e-7


def test_lr_schedule_matches_trainer_formula():
    assert abs(F.lr_at(0, 3e-4, 200, 19164) - 3e-4 / 200) < 1e-15
    assert abs(F.lr_at(199, 3e-4, 200, 19164) - 3e-4) < 1e-15
    assert abs(F.lr_at(19164, 3e-4, 200, 19164) - 3e-5) < 1e-12


def test_grad_check_h6_nonzero_new_exactly_zero_outside():
    P = TL.planted(n_clips=6, T=30)
    R = F.build_rows(P["states"], P["pack"], 5)
    torch.manual_seed(0)
    head = QM.QHead(16)
    banked = {k: np.random.RandomState(3).randn(*s).astype(np.float32) for k, s in zip(QM.TENSOR_KEYS, [(64, 16), (64,), (128, 320), (128,), (3, 128), (3,)])}
    r = F.make_grad_check(head, P["states"]["Z"], R["hist_idx"][:64], R["y"][:64], 0.2, banked=banked)
    assert r["pass"] and r["new_all_nonzero"] and r["outside_all_exactly_zero"]
    assert all(v > 0 for v in r["new_head"].values()) and len(r["new_head"]) == 6
    assert r["outside"]["control_outside_graph"] == 0.0 and r["n_banked_tensors_checked"] == 6
    assert all(r["outside"]["banked_" + k] == 0.0 for k in QM.TENSOR_KEYS)


def test_grad_check_deliberate_regression_head_outside_graph(monkeypatch):
    """A fit whose loss never reaches the head (pred ignored) must read all-zero gradients on the new head: the check goes red."""
    P = TL.planted(n_clips=6, T=30)
    R = F.build_rows(P["states"], P["pack"], 5)
    monkeypatch.setattr(F, "pinball", lambda pred, y, q: (0.0 * pred.detach() + y).mean())
    torch.manual_seed(0)
    r = F.make_grad_check(QM.QHead(16), P["states"]["Z"], R["hist_idx"][:64], R["y"][:64], 0.2)
    assert not r["new_all_nonzero"] and not r["pass"]


@pytest.fixture(scope="module")
def planted_fit():
    P = TL.planted(n_clips=200, T=60, seed=0)
    R = F.build_rows(P["states"], P["pack"], 5)
    held = split_rows(P, R, 50)
    head = F.fit_head(P["states"]["Z"], R["hist_idx"][~held], R["y"][~held], 0, FAST["steps"], FAST["batch"], FAST["lr"], FAST["wd"], FAST["warmup"], FAST["clip_grad"], read=P["read"])
    q = F.predict(head, P["states"]["Z"], R["hist_idx"][held])
    return P, R, held, head, q


def test_planted_coverage_is_020_within_003(planted_fit):
    """h6: on planted states with a known gap and noise that grows with the gap, the held-out coverage is 0.20 within 0.03."""
    P, R, held, head, q = planted_fit
    cov = F.coverage(q, R["y"][held])
    assert abs(cov["share_q_above_label"] - 0.20) <= 0.03, cov


def test_planted_conditional_coverage_follows_the_state(planted_fit):
    """h6 (state input): the noise scale is carried by a state direction that the gap read does not hold; coverage is 0.20 within 0.05 in BOTH noise groups.
    A head that sees only the read cannot get both groups right."""
    P, R, held, head, q = planted_fit
    u = P["u"][R["row"][held]]
    y = R["y"][held]
    for grp in (0.3, 1.0):
        m = u == grp
        assert m.sum() > 100
        assert abs(float((q[m] > y[m]).mean()) - 0.20) <= 0.05, (grp, float((q[m] > y[m]).mean()))


def test_planted_quantile_value_close_to_truth(planted_fit):
    P, R, held, head, q = planted_fit
    gt = P["gap_true"][R["row"][held]]
    u = P["u"][R["row"][held]]
    truth = gt + 0.1 * gt * u * (-0.8416212335729143)
    assert float(np.sqrt(np.mean((q - truth) ** 2))) < 2.0


def test_fit_reads_only_fit_clips(planted_fit):
    P, R, held, head, q = planted_fit
    assert held.sum() > 0 and (~held).sum() > held.sum()
    assert set(R["clip"][held]) == set(range(150, 200))


def run_main(tmp, arm="X_s0", extra=(), P=None, split="train", states_name=None):
    P = P or TL.planted(n_clips=40, T=60, seed=0)
    d = os.path.join(str(tmp), "in")
    sp, pp = TL.write_files(d, P, split=split, arm=arm)
    if states_name:
        os.replace(sp, os.path.join(d, states_name))
        os.replace(sp + ".json", os.path.join(d, states_name) + ".json")
        sp = os.path.join(d, states_name)
    out = os.path.join(str(tmp), "out")
    argv = ["--arm", arm, "--seed", "0", "--states", sp, "--pack", pp, "--out-dir", out, "--held-out", "10", "--steps", "300", "--batch", "64", "--lr", "3e-3", "--warmup", "20"] + list(extra)
    return F.main(argv), out, sp, pp


def test_main_end_to_end_writes_head_and_record(tmp_path):
    rc, out, sp, pp = run_main(tmp_path)
    assert rc == 0
    rec = json.load(open(os.path.join(out, "X_s0.qhead.json")))
    assert rec["card"] == "PREREG_V8F_PROOF_ADD42" and rec["quantile"] == 0.2 and rec["q4_only_train_files"] is True
    assert rec["n_fit_clips"] == 30 and rec["n_held_out_clips"] == 10 and rec["n_fit_rows"] + rec["n_held_out_rows"] == rec["counts"]["rows_used"]
    assert rec["n_held_out_rows"] > 0 and rec["fit_clip_max_index"] < rec["held_clip_min_index"] == 30 and rec["fit_clip_max_index"] == 29
    assert sp.replace("\\", "/") in rec["files_read_sha256"] and pp.replace("\\", "/") in rec["files_read_sha256"]
    assert rec["files_read_sha256"][sp.replace("\\", "/")] == QM.sha256_file(sp)
    assert rec["gradient_check"]["pass"] and isinstance(rec["h5_coverage_in_0.15_0.25"], bool)
    assert rec["coverage_fit"]["n"] == rec["n_fit_rows"] and rec["coverage_held_out"]["n"] == rec["n_held_out_rows"]
    assert rec["n_params"] == 16 * 64 + 64 + 320 * 128 + 128 + 128 + 1
    h = QM.load_head(os.path.join(out, "X_s0.qhead.npz"))
    assert h.hist == 5
    assert rec["head_file_sha256"][os.path.join(out, "X_s0.qhead.npz").replace("\\", "/")] == QM.sha256_file(os.path.join(out, "X_s0.qhead.npz"))
    with pytest.raises(SystemExit, match="exists"):                  # fitted once
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", pp, "--out-dir", out, "--steps", "10"])


def test_h4_refuses_evaluation_split_paths(tmp_path):
    P = TL.planted(n_clips=12, T=30)
    with pytest.raises(SystemExit, match="evaluation split"):
        run_main(tmp_path, P=P, states_name="X_s0_eval.npz")
    d = os.path.join(str(tmp_path), "in2")
    sp, pp = TL.write_files(d, P)
    pe = os.path.join(d, "eval_pack.npz")
    os.replace(pp, pe)
    with pytest.raises(SystemExit, match="evaluation split"):
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", pe, "--out-dir", os.path.join(str(tmp_path), "o2"), "--steps", "5"])
    with pytest.raises(SystemExit, match="evaluation split"):                       # also the optional files
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", os.path.join(d, "eval_pack.npz"), "--lead", os.path.join(d, "x_eval_lead.npz"), "--out-dir", os.path.join(str(tmp_path), "o3")])


def test_h4_refuses_record_with_split_eval(tmp_path):
    with pytest.raises(SystemExit, match="split"):
        run_main(tmp_path, P=TL.planted(n_clips=12, T=30), split="eval")


def test_h4_refuses_missing_record(tmp_path):
    P = TL.planted(n_clips=12, T=30)
    sp, pp = TL.write_files(os.path.join(str(tmp_path), "in"), P)
    os.replace(sp + ".json", sp + ".bak")
    with pytest.raises(SystemExit, match="no record"):
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", pp, "--out-dir", os.path.join(str(tmp_path), "o"), "--steps", "5"])


def test_is_eval_path():
    assert QM.is_eval_path("/a/b/LS_s0_eval.npz") and QM.is_eval_path("/a/eval141/x.npz") and QM.is_eval_path("C:\\x\\eval_pack.npz")
    assert not QM.is_eval_path("/workspace/v8proof/states/LS_s0_train.npz") and not QM.is_eval_path("/a/b/train_pack.npz")


def test_lead_file_gives_the_beside_numbers(tmp_path):
    P = TL.planted(n_clips=40, T=60, seed=0)
    d = os.path.join(str(tmp_path), "in")
    sp, pp = TL.write_files(d, P)
    lp = os.path.join(d, "X_s0_train_lead.npz")
    gh = (P["gap_true"] + 3.0).astype(np.float32)
    np.savez(lp, gap_m=gh)
    out = os.path.join(str(tmp_path), "out")
    assert F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", pp, "--out-dir", out, "--held-out", "10", "--steps", "300", "--lr", "3e-3", "--warmup", "20", "--lead", lp]) == 0
    rec = json.load(open(os.path.join(out, "X_s0.qhead.json")))
    b = rec["beside_lead_read"]
    assert 0.0 <= b["held_out_clips"]["share_used_above_label"] <= 1.0 and b["fit_clips"]["mean_shift_m"] >= 0.0
