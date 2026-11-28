"""Tests of card ADD153's tools (CPU): labels, windows, the head's reach, the reads and their planted identities, and a dry run of every arm on fake files."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "leadspeed"))
sys.path.insert(0, HERE)
import leadacc as A  # noqa: E402
import leadspeed as L  # noqa: E402
import train_acc_box2 as TA  # noqa: E402
import train_full_box2 as F  # noqa: E402
from test_train_full import _fake  # noqa: E402

torch = pytest.importorskip("torch")


def _rand_rows(rng, T):
    k = np.arange(2 * T)
    has = (rng.uniform(size=2 * T) > 0.1).astype(int)
    lead = np.ones(2 * T, int)
    gap = 20 + np.cumsum(rng.normal(0, 0.3, 2 * T))
    gap[rng.integers(0, 2 * T, 5)] += 5.0                                                     # broken steps
    return k, has, lead, gap


def test_closing_labels_run5_is_leadspeed():
    rng = np.random.default_rng(0)
    for _ in range(20):
        k, has, lead, gap = _rand_rows(rng, 60)
        pres, g, clos = L.token_labels(k, has, lead, gap, 60)
        c5 = A.closing_labels(pres, g, run_max=5)
        assert np.array_equal(np.isnan(c5), np.isnan(clos)) and np.allclose(c5[~np.isnan(c5)], clos[~np.isnan(clos)], rtol=0, atol=0)
        c10 = A.closing_labels(pres, g, run_max=10)
        assert np.array_equal(np.isnan(c10), np.isnan(clos))                                    # same frames have a label (run >= 3 either way)


def test_d_labels():
    c = np.array([np.nan, 1, 2, 3, 4, 5, 6, np.nan, 8.0])
    d = A.d_labels(c)
    assert np.isnan(d[:5]).all() and d[6] == 5.0 and np.isnan(d[5]) and np.isnan(d[7]) and d[8] == 5.0


def test_window_pads_and_masks():
    tok = np.arange(12 * 2 * 3, dtype=np.uint8).reshape(12, 2, 3) + 1
    w, m = A.window(tok, 3, 6)
    assert w.shape == (6, 2, 3) and not w[:2].any() and np.array_equal(w[2:], tok[:4]) and m.tolist() == [False, False, True, True, True, True]
    w, m = A.window(tok, 11, 6)
    assert np.array_equal(w, tok[6:12]) and m.all()


@pytest.mark.parametrize("arm", sorted(A.ARMS))
def test_head_shapes_and_mask(arm):
    c = A.ARMS[arm]
    torch.manual_seed(0)
    h = A.make_acc_head(c["H"], c["dil"], c["motion"], d_in=16, width=8).eval()
    x = torch.randn(3, c["H"], 5, 16)
    v = torch.ones(3, c["H"], dtype=torch.bool)
    v[0, :4] = False
    x[0, :4] = 0.0
    o = h(x, v)
    assert o.shape == (3, 4) and torch.isfinite(o).all()
    x2 = x.clone()
    x2[0, :4] = 7.0                                                                             # content of masked frames does not reach the output
    assert torch.equal(h(x2, v), o)


def test_reach_long_history_and_row217_blind_frame():
    """Section 0: row 217's stack sees 9 of 10 frames; the dilated stacks see their whole window."""
    torch.manual_seed(1)
    for arm, sees_first in (("K10D", False), ("K25D", True), ("K40D", True), ("K25DM", True)):
        c = A.ARMS[arm]
        h = A.make_acc_head(c["H"], c["dil"], c["motion"], d_in=16, width=8).eval()
        x = torch.randn(2, c["H"], 5, 16)
        v = torch.ones(2, c["H"], dtype=torch.bool)
        x2 = x.clone()
        x2[:, 0] = torch.randn_like(x2[:, 0]) * 2.0                                            # (a constant shift would be removed by the LayerNorm)
        moved = not torch.allclose(h(x, v), h(x2, v))
        assert moved == sees_first, arm
        assert A.receptive_field(c["dil"]) >= c["H"] or not sees_first


def test_reads_planted_and_floor():
    rng = np.random.default_rng(4)
    pcs = []
    for _ in range(12):
        T = 40
        v = rng.uniform(3, 15, T)
        cl = rng.normal(0, 1, T)
        cl[:2] = np.nan
        pcs.append({"v": v, "clos_lab": cl, "clos_lab2": cl + 0.1 * rng.normal(size=T), "clos_head": cl.copy(), "d_head": A.d_labels(cl)})
    for d in pcs:
        d["clos_head"][:A.T_MIN] = np.nan
        d["d_head"][:A.T_MIN] = np.nan
    q = A.reads([A.clip_sums(A.clip_vectors(d)) for d in pcs])
    assert abs(q["S"] - 1) < 1e-9 and abs(q["A"] - 1) < 1e-9 and abs(q["A_derived"] - 1) < 1e-9 and abs(q["A_rel"] - 1) < 1e-9
    assert q["n_A"] == sum(len(A.clip_vectors(d)["A"][0]) for d in pcs) > 0
    # the floor head (closing 0, d 0) reads A equal to A_floor exactly
    z = [dict(d, clos_head=np.where(np.isnan(d["clos_head"]), np.nan, 0.0), d_head=np.where(np.isnan(d["d_head"]), np.nan, 0.0)) for d in pcs]
    cz = [A.clip_sums(A.clip_vectors(d)) for d in z]
    qz = A.reads(cz)
    assert abs(qz["A"] - qz["A_floor"]) < 1e-12
    p = A.paired(cz, "A", cz, "A_floor", n_boot=50)
    assert abs(p["diff"]) < 1e-12 and max(abs(x) for x in p["ci95"]) < 1e-12
    cs = [A.clip_sums(A.clip_vectors(d)) for d in pcs]
    p2 = A.paired(cs, "A", cz, "A", n_boot=200)
    assert p2["diff"] > 0 and p2["ci95"][0] > 0
    # the A frames equal row 217's: same count as train_full_box2.read_quantities
    rq = F.read_quantities([dict(d, pres_lab=np.ones(40, bool), pres_head=np.full(40, 0.5), gap_lab=np.full(40, 20.0), gap_head=np.full(40, 20.0)) for d in pcs])
    assert rq["n_A"] == q["n_A"] and rq["n_S"] == q["n_S"] and abs(rq["A"] - q["A_derived"]) < 1e-9


def test_names():
    assert A.name_of(0.30, 0.01, 0.0) == "ACCHIST10K-CARRIED"
    assert A.name_of(0.31, 0.0, 0.5) == "ACCHIST10K-SPEED-ONLY"
    assert A.name_of(0.29, 0.2, 0.29) == "ACCHIST10K-NOT-CARRIED"


@pytest.mark.parametrize("arm", sorted(A.ARMS))
def test_dry_run_every_arm(tmp_path, arm):
    rng = np.random.default_rng(0)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=14, n_eval=3)
    ref = os.path.join(str(tmp_path), "ref.pt")
    torch.manual_seed(3)
    torch.save(L.make_head(d_in=1024, width=64, temporal=True).state_dict(), ref)
    out = os.path.join(str(tmp_path), "out")
    r = TA.main(["--arm", arm, "--out", out, "--steps", "6", "--ckpt-every", "3", "--n-fit", "10", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard",
                 "--ref-ckpt", ref, "--ref-sha", F.sha_file(ref)[:16], "--no-marker"])
    assert "error" not in r, r.get("traceback")
    rj = json.load(open(os.path.join(out, "ACC.json")))
    for c in ("c2_disjoint", "c3_planted_read", "c4_checkpoint", "c5_token_bytes", "c6_window", "c7_d_trained"):
        assert rj["controls"][c]["PASS"], (c, rj["controls"][c])
    assert rj["controls"]["c7_d_trained"]["grad_abs_other_rows"] == 0.0
    assert rj["held_out"]["n_A"] > 0 and rj["eval141"]["n_S"] > 0
    assert set(rj["train"]["ckpt_sha256"]) == {"3", "6"}
    assert "dA" in rj["paired_vs_ref_held_out"] and len(rj["dA_floor_held_out"]["ci95"]) == 2
    assert rj["label_run"] == 5


def test_label_run_10_changes_training_label_only(tmp_path):
    rng = np.random.default_rng(1)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=10, n_eval=2)
    out = os.path.join(str(tmp_path), "o10")
    r = TA.main(["--arm", "K10D", "--out", out, "--steps", "3", "--ckpt-every", "3", "--n-fit", "8", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard",
                 "--label-run", "10", "--no-marker"])
    assert "error" not in r, r.get("traceback")
    assert r["label_run"] == 10 and r["controls"]["c3_planted_read"]["PASS"]


def test_smoke_projection_and_stop(tmp_path):
    import smoke_acc as S
    rng = np.random.default_rng(2)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=14, n_eval=3)
    ref = os.path.join(str(tmp_path), "ref.pt")
    torch.save(L.make_head(d_in=1024, width=64, temporal=True).state_dict(), ref)
    r = S.main(["--out", os.path.join(str(tmp_path), "sm"), "--ref-ckpt", ref, "--ref-sha", F.sha_file(ref)[:16], "--arms", "K10D",
                "--extra=" + " ".join(["--steps", "4", "--ckpt-every", "4", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard"])])
    assert "error" not in r, r.get("traceback")
    assert r["projection"]["hours_per_arm"]["K10D"] > 0
    slow = S.projection({"x": {"s_per_step": 1.0, "load_s_per_clip": 0.1, "read_ho_s_per_clip": 1.0, "read_ev_s_per_clip": 1.0}})
    assert slow["stop"] and slow["total_hours"] > 11
    fast = S.projection({"x": {"s_per_step": 0.05, "load_s_per_clip": 0.1, "read_ho_s_per_clip": 1.0, "read_ev_s_per_clip": 1.0}})
    assert not fast["stop"]


def test_select_and_jobs(tmp_path):
    import mk_jobs_acc as MK
    import select_stage2 as SEL
    ok = lambda a: {"controls_all_pass": True, "held_out": {"A": a, "S": 0.4}, "name": "x"}  # noqa: E731
    best, tab = SEL.pick({"K25D": ok(0.25), "K25DM": dict(ok(0.40), controls_all_pass=False), "K10D": ok(0.21), "K40D": None})
    assert best == "K25D" and not tab["K25DM"]["controls_all_pass"] and not tab["K40D"]["controls_all_pass"]
    assert SEL.pick({"K25D": None})[0] is None
    ps = MK.stage1(str(tmp_path)) + MK.stage2("K40D", str(tmp_path))
    assert len(ps) == 7
    for p in ps:
        b = open(p, "rb").read()
        assert b.count(b"\r\n") == b.count(b"\n") == 8 and rb"D:\tanitad\venv\Scripts\python.exe D:\tanitad\leadacc" + b"\\" in b
    s2 = open(os.path.join(str(tmp_path), "acc_s2smooth.cmd"), "rb").read()
    assert b"--arm K40D" in s2 and b"--label-run 10" in s2 and b"--train-seed 148" in s2
    assert b"--train-seed 149" in open(os.path.join(str(tmp_path), "acc_s2rep.cmd"), "rb").read()
