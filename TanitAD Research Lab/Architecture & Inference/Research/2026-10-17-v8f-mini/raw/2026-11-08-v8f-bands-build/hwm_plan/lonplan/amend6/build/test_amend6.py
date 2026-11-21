"""AMEND_HWMPLAN_6 build tests. CPU, serial, synthetic latents (no model, no box 2). Each green check has a mutation that must read RED."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch
import amend6_env  # noqa: F401
import fit_ra6 as F
import select_clips6 as S6
import amend6_controls as C6
import amend6_read as RD6

ROOT = os.path.dirname(HERE)
TAG = "R35_s0"


def synth(n_clips=14, T=44, seed=0, names=None, w=0.7):
    """Latents file in the layout of latents_train_R35_s{k}.pt: linear truth acc = w . (z10 - z0) + noise, 16-frame windows every 4 frames."""
    r = np.random.RandomState(seed)
    wv = r.randn(128) * 0.05
    Z, A, OK, TT, EID = [], [], [], [], []
    for c in range(n_clips):
        name = names[c] if names else f"clip{c:03d}"
        z = np.cumsum(r.randn(T, 128) * 0.3, 0)
        a = np.zeros((T, 2)); a[:, 0] = np.convolve(np.r_[z[1:] - z[:-1], np.zeros((1, 128))] @ wv, np.ones(1)) + r.randn(T) * 0.05
        for t in range(15, T, 4):
            Z.append(z[t - 15:t + 1]); A.append(a[t - 15:t + 1]); OK.append(np.ones(16, bool)); TT.append(t); EID.append(name)
    return dict(Z1={TAG: torch.tensor(np.stack(Z), dtype=torch.float32)}, A_raw=torch.tensor(np.stack(A), dtype=torch.float32), ok=torch.tensor(np.stack(OK)), t=np.array(TT), eid=EID)


ident = lambda x: x


# ---------------------------------------------------------------------------------------------- the fit and the K-control
def test_kcontrol_green_and_mutation_red():
    LT = synth()
    Dr, Yf, EP, clips = F.pairs(LT, ident)
    ra5 = F.fit_ra(Dr, Yf, EP, len(clips))                    # stands in for AMEND_5's frozen readout
    res = F.kcontrol_pure(LT, ident, ra5)
    assert res["green"]["ok"] and res["green"]["max_abs_dW"] < 1e-12
    assert not res["regression_drop_one_clip"]["ok"]            # the 140-clip fit must read RED
    assert res["passed"]


def test_compare_flags_a_small_perturbation():
    LT = synth(); Dr, Yf, EP, clips = F.pairs(LT, ident)
    a = F.fit_ra(Dr, Yf, EP, len(clips)); b = dict(a, W=a["W"] + 1e-6)
    assert F.compare(a, a)["ok"] and not F.compare(a, b)["ok"]
    assert not F.compare(a, dict(a, lam=a["lam"] * 10))["ok"]


def test_fit_recovers_the_linear_truth_and_uses_the_amend1_fold_map():
    LT = synth(n_clips=30, T=60)
    Dr, Yf, EP, clips = F.pairs(LT, ident)
    f = F.fit_ra(Dr, Yf, EP, len(clips))
    p = F.ridge_pred((f["mu"], f["sd"], f["W"], f["ym"]), Dr)[:, 0]
    assert np.corrcoef(p, Yf[:, 0])[0, 1] > 0.9
    fo = F.fold_map(len(clips), EP)
    assert set(np.unique(fo)) == {0, 1, 2, 3, 4} and (fo == (np.random.RandomState(0).permutation(len(clips)) % 5)[EP]).all()


def test_pairs_need_eleven_consecutive_finite_frames():
    LT = synth(n_clips=3)
    first = [i for i, e in enumerate(LT["eid"]) if e == "clip000"]
    LT["A_raw"][first, :, 0] = float("nan")
    Dr, Yf, EP, clips = F.pairs(LT, ident)
    assert 0 not in set(EP.tolist()) and {1, 2} <= set(EP.tolist()) and np.isfinite(Yf).all()


def test_merge_refuses_a_repeated_clip_and_keeps_order():
    a, b = synth(3, names=["a0", "a1", "a2"]), synth(2, seed=1, names=["b0", "b1"])
    M = F.merge_latents([a, b])
    assert len(M["eid"]) == len(a["eid"]) + len(b["eid"]) and M["Z1"][TAG].shape[0] == len(M["eid"])
    with pytest.raises(ValueError):
        F.merge_latents([a, synth(2, seed=2, names=["a2", "z9"])])


def test_curve_shapes_and_oof():
    LT = synth(n_clips=25, T=60)
    Dr, Yf, EP, clips = F.pairs(LT, ident)
    res = F.curve_pure(Dr, Yf, EP, len(clips), [10, 25], n_boot=20)
    assert set(res) == {"10", "25"} and res["25"]["r"] > 0.8 and res["10"]["n_pairs"] < res["25"]["n_pairs"]
    assert res["25"]["r2_ci"][0] <= res["25"]["r2"] <= res["25"]["r2_ci"][1] + 1e-9


# ---------------------------------------------------------------------------------------------- selection
def _man():
    mt = [json.loads(l) for l in open(S6.MAN_TRAIN, encoding="utf-8")]
    me = [json.loads(l) for l in open(S6.MAN_EVAL, encoding="utf-8")]
    return mt, me, json.load(open(S6.CLIPS141, encoding="utf-8"))


def test_selection_reproduces_the_banked_list_and_is_prefix_nested():
    mt, me, c141 = _man()
    banked = json.load(open(ROOT + "/clips6.json", encoding="utf-8"))
    again = S6.select(mt, me, c141, ())
    assert again["clips"] == banked["clips"] and again["clips_sha256"] == banked["clips_sha256"] == "27d2bb2f45d49611aef2ba526bc0937c4d28855f565c16aa8abeaa4010d91649"
    small = S6.select(mt, me, c141, (), n_new=141)
    assert small["clips"] == banked["clips"][:141]


def test_c_sel6_green_and_planted_eval_id_reads_red():
    mt, me, c141 = _man()
    banked = json.load(open(ROOT + "/clips6.json", encoding="utf-8"))["clips"]
    ex = [m["clip"] for m in sorted(mt, key=lambda m: m["sha12"])[540:600]]
    ev = {m["clip"] for m in me}
    assert C6.check_sel(banked, c141["clips"], ex, ev, set())["ok"]
    for src, name in ((sorted(ev)[0], "eval"), (c141["clips"][0], "c141"), (ex[0], "excluded")):
        bad = list(banked); bad[0] = src
        assert not C6.check_sel(bad, c141["clips"], ex, ev, set())["ok"], name
    assert not C6.check_sel(banked, c141["clips"], ex, ev, {banked[5]})["ok"]       # a window eid


def test_selection_refuses_a_manifest_that_is_not_the_trainers_order():
    mt, me, c141 = _man()
    bad = [dict(m) for m in mt]; bad[0]["sha12"] = "zzzzzzzzzzzz"
    with pytest.raises(S6.Refused):
        S6.select(bad, me, c141, ())
    with pytest.raises(S6.Refused):
        S6.select(mt, me, dict(c141, clips_sha256="0" * 64), ())


# ---------------------------------------------------------------------------------------------- chunks
def test_chunk_parts_checks_sha256_and_returns_the_requested_clips(tmp_path):
    order = json.load(open(ROOT + "/clips6.json", encoding="utf-8"))["clips"]
    A = synth(3, names=order[0:3]); B = synth(2, seed=3, names=order[3:5])
    J = dict(chunks=[])
    for i, (d, cl) in enumerate(((A, order[0:3]), (B, order[3:5]))):
        fn = f"chunk_{i:03d}.pt"
        torch.save(dict(d, clips=cl, starts={}, pull_s=[]), tmp_path / fn)
        J["chunks"].append(dict(file=fn, clips=cl, n_windows=len(d["eid"]), sha256=hashlib.sha256((tmp_path / fn).read_bytes()).hexdigest(), md5_verified=len(cl)))
    (tmp_path / "CHUNKS.json").write_text(json.dumps(J))
    parts = F.chunk_parts(TAG, 4, str(tmp_path / "CHUNKS.json"), str(tmp_path))            # 3 + 1 of the second chunk
    assert sorted({e for p in parts for e in p["eid"]}) == sorted(order[:4])
    assert C6.check_chunks(J, str(tmp_path), order)["ok"]
    raw = bytearray((tmp_path / "chunk_000.pt").read_bytes()); raw[len(raw) // 2] ^= 1
    (tmp_path / "chunk_000.pt").write_bytes(bytes(raw))                                  # the mutation
    assert not C6.check_chunks(J, str(tmp_path), order)["ok"]
    with pytest.raises(RuntimeError):
        F.chunk_parts(TAG, 4, str(tmp_path / "CHUNKS.json"), str(tmp_path))
    J2 = dict(chunks=[J["chunks"][0], dict(J["chunks"][0])])
    assert not C6.check_chunks(J2, str(tmp_path), order)["ok"]                            # duplicate clips


def test_largest_checkpoint_rule(tmp_path):
    def mk(n):
        p = tmp_path / f"c{n}.json"; p.write_text(json.dumps(dict(chunks=[dict(clips=list(range(n)))]))); return str(p)
    assert F.largest_checkpoint(mk(859)) == 859 and F.largest_checkpoint(mk(500)) == 423 and F.largest_checkpoint(mk(141)) == 141
    with pytest.raises(RuntimeError):
        F.largest_checkpoint(mk(100))


# ---------------------------------------------------------------------------------------------- the outcome table (section 6)
def test_outcome_table():
    ok, no = {"s0": True, "s1": True}, {"s0": False, "s1": False}
    mix = {"s0": True, "s1": False}
    assert RD6.outcome6("CEILING", ok, ok, no) == "R6-CEILING"
    assert RD6.outcome6("HARM", ok, ok, no) == "HARM"
    assert RD6.outcome6("NARROWS", ok, ok, no) == "R6-PRECISION"
    assert RD6.outcome6("NARROWS", ok, mix, no) == "R6-SAME"                  # one chassis only: not PRECISION
    assert RD6.outcome6("NARROWS", ok, no, no) == "R6-SAME" and RD6.outcome6("NULL", no, no, no) == "R6-SAME"
    assert RD6.outcome6("NARROWS", ok, no, mix) == "R6-WORSE"
    assert RD6.outcome6("NARROWS", ok, {}, no) == "R6-SAME"


def test_env_never_falls_back_to_the_amend5_readout():
    import amend5_stage as ST
    assert ST.READOUT_FALLBACKS == [] and ST.ROOT == ROOT and "amend5" not in ST.readout_path("s0")
    assert ST.SPEC.endswith("AMEND_HWMPLAN_6.md") and ST.SPEC_SHA == open(ROOT + "/AMEND_HWMPLAN_6.sha256").read().split()[0]
    assert ST.RA_SHA["s0"] != "edab7dce7b57f9ae21d90e5cc317c914aed432751fde7a7d4e6cf6af0748a6cf"
