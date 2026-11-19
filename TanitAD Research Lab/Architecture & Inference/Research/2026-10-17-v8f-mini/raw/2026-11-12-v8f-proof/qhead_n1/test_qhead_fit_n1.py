"""Tests of qhead_fit_n1.py (note PREREG_V8F_PROOF_ADD42_N1, sections 2 and 4). Small synthetic arrays; every temp file goes under pytest's basetemp."""
import json
import os

import numpy as np
import pytest

import qhead_testlib as TL  # noqa: E402  (sets the source path first)
import qhead_fit_n1 as F  # noqa: E402
import qhead_model as QM  # noqa: E402
import torch  # noqa: E402


def split(P, a, b, c):
    R = F.build_rows(P["states"], P["pack"], 5)
    CS = F.clip_sets(len(P["states"]["clips"]), a, b - a, c - b)
    ms = {k: F.set_mask(R["clip"], CS[k]) for k in CS}
    return R, CS, ms


def fit_planted(P, R, ms, steps, every, lr=3e-3, seed=0):
    return F.fit_head(P["states"]["Z"], R["hist_idx"][ms["fit"]], R["y"][ms["fit"]], seed, steps, 64, lr, 1e-2, 20, 1.0, R["hist_idx"][ms["select"]], R["y"][ms["select"]], every)


def test_clip_sets_disjoint_in_manifest_order_and_refusal():
    CS = F.clip_sets(1200, 800, 200, 200)
    assert CS == {"fit": [0, 800], "select": [800, 1000], "check": [1000, 1200]}
    CS = F.clip_sets(1500, 800, 200, 200)                              # a longer file: the first clips, the rest unused
    assert CS["check"] == [1000, 1200]
    with pytest.raises(SystemExit, match="fewer than"):
        F.clip_sets(1199, 800, 200, 200)
    P = TL.planted(n_clips=30, T=30)
    R, CS, ms = split(P, 15, 22, 30)
    assert not (ms["fit"] & ms["select"]).any() and not (ms["fit"] & ms["check"]).any() and not (ms["select"] & ms["check"]).any()
    assert R["clip"][ms["fit"]].max() < R["clip"][ms["select"]].min() and R["clip"][ms["select"]].max() < R["clip"][ms["check"]].min()
    assert ms["fit"].sum() + ms["select"].sum() + ms["check"].sum() == len(R["y"])


def test_selection_keeps_the_minimum_of_the_select_curve():
    P = TL.planted(n_clips=40, T=40, seed=1)
    R, CS, ms = split(P, 20, 30, 40)
    head = fit_planted(P, R, ms, 600, 100)
    steps = [s for s, _ in head.select_curve]
    vals = [v for _, v in head.select_curve]
    assert steps == [100, 200, 300, 400, 500, 600]                      # every 100 steps and the last
    assert head.selected_step == steps[int(np.argmin(vals))]
    # the kept parameters are those of the selected step: the select value of the returned head equals the curve's minimum
    assert abs(F.eval_select(head, P["states"]["Z"], R["hist_idx"][ms["select"]], R["y"][ms["select"]]) - min(vals)) < 1e-9


def test_last_step_is_evaluated_when_not_a_multiple():
    P = TL.planted(n_clips=40, T=40, seed=1)
    R, CS, ms = split(P, 20, 30, 40)
    head = fit_planted(P, R, ms, 250, 100)
    assert [s for s, _ in head.select_curve] == [100, 200, 250]


def test_tie_keeps_the_earlier_step(monkeypatch):
    P = TL.planted(n_clips=40, T=40, seed=1)
    R, CS, ms = split(P, 20, 30, 40)
    monkeypatch.setattr(F, "eval_select", lambda *a, **k: 1.25)
    head = fit_planted(P, R, ms, 300, 100)
    assert head.selected_step == 100 and [v for _, v in head.select_curve] == [1.25, 1.25, 1.25]


def test_memorising_fit_rows_selects_an_earlier_step_than_the_last():
    """Few fit rows (3 clips) and a wide head: the select value turns upward after the head memorises the fit rows; the kept step is earlier than the last."""
    P = TL.planted(n_clips=44, T=60, seed=2)
    R, CS, ms = split(P, 3, 43, 44)
    head = fit_planted(P, R, ms, 2400, 100, lr=3e-3)
    vals = [v for _, v in head.select_curve]
    assert head.selected_step < 2400, head.select_curve[-5:]
    assert vals[-1] > min(vals) + 1e-6


def test_shift_gives_020_on_select_rows_within_one_over_n():
    rng = np.random.RandomState(0)
    for n in (37, 200, 1001):
        q, y = rng.randn(n) * 3 + 20, rng.randn(n) * 3 + 20
        c = F.shift_value(q, y)
        share = float(((q - c) > y).mean())
        assert abs(share - 0.20) <= 1.0 / n + 1e-12, (n, share)


def run_main(tmp, P=None, arm="X_s0", extra=(), split_="train", states_name=None, tag="a", sets=(20, 10, 10), steps=300, every=100):
    P = P or TL.planted(n_clips=40, T=60, seed=0)
    d = os.path.join(str(tmp), "in" + tag)
    sp, pp = TL.write_files(d, P, split=split_, arm=arm)
    if states_name:
        os.replace(sp, os.path.join(d, states_name))
        os.replace(sp + ".json", os.path.join(d, states_name) + ".json")
        sp = os.path.join(d, states_name)
    out = os.path.join(str(tmp), "out" + tag)
    argv = ["--arm", arm, "--seed", "0", "--states", sp, "--pack", pp, "--out-dir", out, "--fit-clips", str(sets[0]), "--select-clips", str(sets[1]), "--check-clips", str(sets[2]),
            "--select-every", str(every), "--steps", str(steps), "--batch", "64", "--lr", "3e-3", "--warmup", "20"] + list(extra)
    return F.main(argv), out, sp, pp


def read_rec(out, arm="X_s0"):
    return json.load(open(os.path.join(out, arm + ".qhead.json")))


def test_main_end_to_end_record_fields_and_head_file(tmp_path):
    rc, out, sp, pp = run_main(tmp_path)
    assert rc == 0
    rec = read_rec(out)
    assert rec["card"] == "PREREG_V8F_PROOF_ADD42_N1" and rec["quantile"] == 0.2 and rec["q4_only_train_files"] is True
    assert rec["clip_sets"] == {"fit": [0, 20], "select": [20, 30], "check": [30, 40]}
    assert rec["n_fit_rows"] + rec["n_select_rows"] + rec["n_check_rows"] == rec["counts"]["rows_used"]
    for k in ("coverage_fit", "coverage_select", "coverage_check"):
        assert rec[k]["n"] > 0 and 0.0 <= rec[k]["share_q_above_label"] <= 1.0
    assert rec["coverage_fit"]["n"] == rec["n_fit_rows"] and rec["coverage_select"]["n"] == rec["n_select_rows"] and rec["coverage_check"]["n"] == rec["n_check_rows"]
    assert all(np.isfinite(rec[k]) for k in ("pinball_fit_m", "pinball_select_m", "pinball_check_m", "shift_c_m"))
    assert isinstance(rec["h5_check_coverage_in_0.15_0.25"], bool) and rec["gradient_check"]["pass"]
    assert rec["selected_step"] in [s for s, _ in rec["select_curve"]] and [s for s, _ in rec["select_curve"]] == [100, 200, 300] and rec["last_step"] == 300
    assert rec["selected_step"] == [s for s, v in rec["select_curve"] if v == min(v2 for _, v2 in rec["select_curve"])][0]
    assert rec["fold_check_max_abs_m"] <= 1e-5
    assert sp.replace("\\", "/") in rec["files_read_sha256"] and rec["files_read_sha256"][sp.replace("\\", "/")] == QM.sha256_file(sp)
    assert rec["n_params"] == 16 * 64 + 64 + 320 * 128 + 128 + 128 + 1
    hp = os.path.join(out, "X_s0.qhead.npz")
    assert rec["head_file_sha256"][hp.replace("\\", "/")] == QM.sha256_file(hp)
    assert rec["states_record"]["ckpt_sha256"] == "ab" * 32            # the keys the unchanged rollout tool reads
    z = np.load(hp, allow_pickle=False)
    assert all(k in z.files for k in QM.TENSOR_KEYS + ("hist",)) and z["mlp2_w"].shape[0] == 1
    assert QM.load_head(hp).hist == 5
    with pytest.raises(SystemExit, match="exists"):                  # fitted once
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", pp, "--out-dir", out, "--fit-clips", "20", "--select-clips", "10", "--check-clips", "10", "--steps", "10"])


def test_saved_head_reproduces_q_head_minus_c(tmp_path, monkeypatch):
    snap = {}
    real = F.fit_head

    def wrapped(*a, **k):
        h = real(*a, **k)
        snap["t"] = h.tensors()                                      # the head of the selected step, before the shift
        return h
    monkeypatch.setattr(F, "fit_head", wrapped)
    rc, out, sp, pp = run_main(tmp_path)
    rec = read_rec(out)
    c = rec["shift_c_m"]
    saved = QM.load_head(os.path.join(out, "X_s0.qhead.npz")).tensors()
    rng = np.random.RandomState(5)
    z = rng.randn(500, 5, 16).astype(np.float32) * 0.5
    q0 = QM.functional_forward(snap["t"], z)[:, 0] * 50.0
    q1 = QM.functional_forward(saved, z)[:, 0] * 50.0
    assert abs(c) > 1e-3
    assert float(np.max(np.abs(q1 - (q0 - c)))) <= 1e-5
    for k in QM.TENSOR_KEYS[:-1]:
        assert np.array_equal(saved[k], snap["t"][k])                # only the output bias carries the shift


def test_shift_is_020_on_select_rows_in_the_saved_head(tmp_path):
    rc, out, sp, pp = run_main(tmp_path, P=TL.planted(n_clips=60, T=60, seed=3), sets=(30, 15, 15))
    rec = read_rec(out)
    n = rec["n_select_rows"]
    assert abs(rec["coverage_select"]["share_q_above_label"] - 0.20) <= 1.0 / n + 1e-9, rec["coverage_select"]
    # and recomputed from the saved head file on the select rows
    P = TL.planted(n_clips=60, T=60, seed=3)
    R = F.build_rows(P["states"], P["pack"], 5)
    ms = F.set_mask(R["clip"], rec["clip_sets"]["select"])
    q = F.predict(QM.load_head(os.path.join(out, "X_s0.qhead.npz")), P["states"]["Z"], R["hist_idx"][ms])
    assert abs(float((q > R["y"][ms]).mean()) - 0.20) <= 1.0 / n + 1e-4


def test_check_rows_enter_no_gradient_no_selection_not_the_shift(tmp_path):
    """Two runs that differ only in the LABELS of the check clips (+100 m) give the same curve, step, shift and head bytes; the check coverage differs."""
    P = TL.planted(n_clips=40, T=60, seed=0)
    rc, out1, _, _ = run_main(tmp_path, P=P, tag="1")
    P2 = TL.planted(n_clips=40, T=60, seed=0)
    chk = P2["pack"]["clip_idx"] >= 30
    P2["pack"]["gap"] = np.where(chk, P2["pack"]["gap"] + 100.0, P2["pack"]["gap"])
    rc, out2, _, _ = run_main(tmp_path, P=P2, tag="2")
    r1, r2 = read_rec(out1), read_rec(out2)
    assert r1["select_curve"] == r2["select_curve"] and r1["selected_step"] == r2["selected_step"] and r1["shift_c_m"] == r2["shift_c_m"]
    assert r1["coverage_fit"] == r2["coverage_fit"] and r1["coverage_select"] == r2["coverage_select"]
    a, b = np.load(os.path.join(out1, "X_s0.qhead.npz")), np.load(os.path.join(out2, "X_s0.qhead.npz"))
    assert all(np.array_equal(a[k], b[k]) for k in QM.TENSOR_KEYS)
    assert r1["coverage_check"]["share_q_above_label"] != r2["coverage_check"]["share_q_above_label"]
    # the record says which clips each use saw
    u = r1["clip_index_range_of_rows_used"]
    assert u["gradient_steps"][1] < 20 <= u["selection"][0] and u["selection"][1] < 30 <= u["check_rows_only_coverage"][0] and u["shift"] == u["selection"]


def test_h4_refuses_evaluation_split_paths_and_bad_records(tmp_path):
    P = TL.planted(n_clips=12, T=30)
    with pytest.raises(SystemExit, match="evaluation split"):
        run_main(tmp_path, P=P, states_name="X_s0_eval.npz", sets=(6, 3, 3))
    d = os.path.join(str(tmp_path), "in2")
    sp, pp = TL.write_files(d, P)
    pe = os.path.join(d, "eval_pack.npz")
    os.replace(pp, pe)
    with pytest.raises(SystemExit, match="evaluation split"):
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", pe, "--out-dir", os.path.join(str(tmp_path), "o2"), "--steps", "5"])
    open(os.path.join(d, "ok_pack.npz"), "wb").write(open(pe, "rb").read())
    with pytest.raises(SystemExit, match="evaluation split"):
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", os.path.join(d, "ok_pack.npz"), "--lead", os.path.join(d, "x_eval_lead.npz"), "--out-dir", os.path.join(str(tmp_path), "o3")])
    with pytest.raises(SystemExit, match="split"):
        run_main(tmp_path, P=P, split_="eval", tag="3", sets=(6, 3, 3))
    sp4, pp4 = TL.write_files(os.path.join(str(tmp_path), "in4"), P)
    os.replace(sp4 + ".json", sp4 + ".bak")
    with pytest.raises(SystemExit, match="no record"):
        F.main(["--arm", "X_s0", "--seed", "0", "--states", sp4, "--pack", pp4, "--out-dir", os.path.join(str(tmp_path), "o4"), "--steps", "5"])


def test_refuses_fewer_clips_than_the_sum_and_empty_sets(tmp_path):
    with pytest.raises(SystemExit, match="fewer than"):
        run_main(tmp_path, P=TL.planted(n_clips=12, T=30), sets=(8, 3, 3))
    with pytest.raises(SystemExit, match="positive"):
        run_main(tmp_path, P=TL.planted(n_clips=12, T=30), sets=(8, 0, 3), tag="b")
    P = TL.planted(n_clips=12, T=30)
    P["pack"]["pres"] = np.where(P["pack"]["clip_idx"] >= 9, 0.0, P["pack"]["pres"])      # the check clips have no usable row
    with pytest.raises(SystemExit, match="no usable row"):
        run_main(tmp_path, P=P, sets=(6, 3, 3), tag="c")


def test_is_eval_path():
    assert QM.is_eval_path("/a/b/LS_s0_eval.npz") and QM.is_eval_path("/a/eval141/x.npz") and not QM.is_eval_path("/workspace/v8proof/states/LS_s0_train.npz")


def test_lead_file_gives_the_beside_numbers(tmp_path):
    P = TL.planted(n_clips=40, T=60, seed=0)
    d = os.path.join(str(tmp_path), "in")
    sp, pp = TL.write_files(d, P)
    lp = os.path.join(d, "X_s0_train_lead.npz")
    np.savez(lp, gap_m=(P["gap_true"] + 3.0).astype(np.float32))
    out = os.path.join(str(tmp_path), "out")
    assert F.main(["--arm", "X_s0", "--seed", "0", "--states", sp, "--pack", pp, "--out-dir", out, "--fit-clips", "20", "--select-clips", "10", "--check-clips", "10", "--steps", "200",
                   "--select-every", "100", "--lr", "3e-3", "--warmup", "20", "--lead", lp]) == 0
    b = read_rec(out)["beside_lead_read"]
    assert set(b) == {"fit_clips", "select_clips", "check_clips"} and b["fit_clips"]["mean_shift_m"] >= 0.0
