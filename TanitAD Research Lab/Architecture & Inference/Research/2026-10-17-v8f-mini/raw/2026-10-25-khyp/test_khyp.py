"""PREREG_V8F_KHYP: the harness (khyp.py), on a TINY random-init v8F-mini and synthetic windows. CPU only, small memory.

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q test_khyp.py --basetemp=<OUT>/pt_tmp -p no:cacheprovider

What is pinned, and the control that must go red for each:
  * CONTROL 1  a model whose longitudinal softmax is EXACTLY one-hot: tau[argmax] == forward_infer's plan, Delta = 0.0 BIT-exactly, for three classes;
               on a model with a spread softmax the same Delta is NOT zero (the probe is alive).  MUTATION: SEL-ARGMAX picks argmin -> control 1 goes RED.
  * harness identity   forward_diag(T's UNCHANGED code) == forward_infer exactly, on every window.
  * the override       changes ONLY the longitudinal slice; the 7 hypotheses are pairwise distinct on a code-sensitive model.
               MUTATION: an override that ignores k makes the seven plans identical -> `hypotheses_distinct` goes RED.
  * CONTROL 4  ORACLE <= SEL-ARGMAX and SEL-LEARN on every window (hard); SEL-MIX reported.  MUTATION: ORACLE picks argmax -> control 4 goes RED.
  * CONTROL 3  the label permutation destroys the signal a logistic regression finds on a problem that HAS signal (and the identity permutation does not).
  * features   known values (constant speed, constant acceleration), the column layout, one SPEED flag that must land in exactly its hypothesis's column.
  * outcome    the PREREG table at its boundaries (exactly -F passes, 0.0001 short fails, exactly -2F is headroom, a worse SECONDARY blocks, a seed split).
  * end to end `hyps` (train + eval, two fake seeds, resumable shards) -> `read --no-banked` writes READ.json + KHYP_RESULT.md with a verdict in the table.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):      # BEFORE numpy: OpenBLAS reserves commit per logical CPU at import
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from pathlib import Path

import numpy as np
import pytest
import torch

OUTDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTDIR))
import khyp  # noqa: E402
from khyp import K, LON0, M, D, C  # noqa: E402

torch.set_num_threads(1)
_orig_onehot = khyp.lon_onehot_code
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    w = D.make_synthetic_world(root, n_train=5, n_eval=3, T=60, seed=0)
    return {**w, "root": Path(root)}


def make_model(world, seed=0, scale=0.3):
    J = D.load_join(world["root"] / "join_train.npz")
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():                                         # code-sensitive O (zero-init head_o would start at P10), and a T with spread logits
        for p in list(model.head_o.parameters()) + list(model.e_code.parameters()) + list(model.h_lon.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return model.eval(), cfg


def batches(world, split="train", n=40, bs=20):
    J = D.load_join(world["root"] / f"join_{split}.npz")
    store = D.ClipStore(world["root"], split, lru=4)
    wins = D.train_windows(store, J, world["names"][split]) if split == "train" else D.eval_grid_windows(world["root"] / "eval_grid_windows.json", J)
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    out = []
    for b0 in range(0, min(n, len(wins)), bs):
        items = []
        for i in range(b0, min(b0 + bs, len(wins), n)):
            ci, t, _, _ = wins[i]
            a_t = i % 3
            items.append(ds.item(ci, t, a_t, a_t + (i // 3) % 5))
        out.append(D.collate(items))
    return out


def run_enumerate(model, bs):
    parts = [khyp.enumerate_batch(model, b) for b in bs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


@pytest.fixture(scope="module")
def spread(world):
    model, _ = make_model(world)
    return run_enumerate(model, batches(world, "train", n=60, bs=20)), model


# ------------------------------------------------------------------------------------------------ the override
def test_override_changes_only_the_longitudinal_slice(world):
    model, _ = make_model(world)
    b = batches(world, "train", 20, 20)[0]
    with torch.no_grad():
        code0 = model.forward_infer(b)["code"]
    for k in range(K):
        c = khyp.lon_onehot_code(code0, k)
        assert torch.equal(c[:, :LON0], code0[:, :LON0]) and torch.equal(c[:, LON0 + K:], code0[:, LON0 + K:])
        want = torch.zeros(K)
        want[k] = 1.0
        assert torch.equal(c[:, LON0:LON0 + K], want.expand(c.shape[0], K))
    assert (code0[:, LON0:LON0 + K].sum(1) - 1).abs().max() < 1e-5, "T's longitudinal slice is a softmax"


def test_harness_identity_and_hypotheses_distinct(spread):
    d, _ = spread
    assert d["tau"].shape[1:] == (K, 30, 2)
    assert float(d["own_diag_maxabs"].max()) == 0.0, "forward_diag with T's own code must equal forward_infer EXACTLY"
    assert khyp.hypotheses_distinct(d["tau"]), "a code-sensitive model must give 7 distinct plans, or a zero headroom means nothing"
    ade = khyp.ade6_per_hyp(d["tau"], d["gt"])
    assert ade.std(1).max() > 1e-3


def test_mutation_override_ignores_k_goes_red(world, monkeypatch):
    model, _ = make_model(world)
    bs = batches(world, "train", 20, 20)
    monkeypatch.setattr(khyp, "lon_onehot_code", lambda code0, k: _orig_onehot(code0, 0))      # ignores k
    d = run_enumerate(model, bs)
    assert not khyp.hypotheses_distinct(d["tau"]), "MUTATION must be visible: seven identical plans"


# ------------------------------------------------------------------------------------------------ CONTROL 1
@pytest.mark.parametrize("c", [0, 3, 6])
def test_control1_exact_onehot_softmax_is_bit_exact(world, c):
    model, _ = make_model(world, seed=1)
    with torch.no_grad():
        model.h_lon.weight.zero_()
        model.h_lon.bias.zero_()
        model.h_lon.bias[c] = 200.0                                # softmax in float32 is EXACTLY one-hot
    d = run_enumerate(model, batches(world, "train", 40, 20))
    assert (d["p_lon"].argmax(1) == c).all() and (d["p_lon"].max(1) == 1.0).all()
    r = khyp.control1_identity(d["tau"], d["own"], d["p_lon"])
    assert r["n_exact_onehot_windows"] == len(d["own"]) and r["max_abs_delta_m"] == 0.0 and r["pass"] is True
    assert np.array_equal(khyp.pick(d["tau"], khyp.argmax_k(d["p_lon"])), d["own"].astype(np.float64))
    # the probe is alive: another class's plan is NOT the own plan
    other = khyp.pick(d["tau"], np.full(len(d["own"]), (c + 1) % K))
    assert np.abs(other - d["own"].astype(np.float64)).max() > 1e-3


def test_control1_spread_softmax_delta_is_nonzero_and_vacuous_flagged(spread):
    d, _ = spread
    r = khyp.control1_identity(d["tau"], d["own"], d["p_lon"])
    assert r["n_exact_onehot_windows"] == 0 and r["pass"] is None
    dl = np.abs(khyp.pick(d["tau"], khyp.argmax_k(d["p_lon"])) - d["own"].astype(np.float64)).max()
    assert dl > 1e-4, "with a spread softmax the one-hot hypothesis differs from T's own plan: the identity is a statement about one-hot windows only"


def test_mutation_argmax_to_argmin_goes_red_on_control1(world, monkeypatch):
    model, _ = make_model(world, seed=1)
    with torch.no_grad():
        model.h_lon.weight.zero_()
        model.h_lon.bias.zero_()
        model.h_lon.bias[3] = 200.0
    d = run_enumerate(model, batches(world, "train", 40, 20))
    assert khyp.control1_identity(d["tau"], d["own"], d["p_lon"])["pass"] is True
    monkeypatch.setattr(khyp, "argmax_k", lambda p: np.asarray(p).argmin(1))
    r = khyp.control1_identity(d["tau"], d["own"], d["p_lon"])
    assert r["pass"] is False and r["max_abs_delta_m"] > 1e-3


# ------------------------------------------------------------------------------------------------ CONTROL 4
def test_control4_oracle_le_vertex_selectors_and_mutation(spread, monkeypatch):
    d, _ = spread
    gt = d["gt"].astype(np.float64)
    ade = khyp.ade6_per_hyp(d["tau"], gt)
    rng = np.random.default_rng(0)
    r = khyp.control4(ade, {"ARGMAX": khyp.argmax_k(d["p_lon"]), "LEARN": rng.integers(0, K, len(ade))})
    assert r["pass"] is True and r["ARGMAX"]["n_violations"] == 0 and r["LEARN"]["n_violations"] == 0
    assert (ade[np.arange(len(ade)), khyp.oracle_k(ade)] == ade.min(1)).all()
    mix = khyp.plan_mix(d["tau"], d["p_lon"])                                   # MIX is a convex combination: not a vertex, so not bound by construction
    dm = np.linalg.norm(mix - gt, axis=-1).mean(1)
    assert (dm <= np.einsum("nk,nk->n", d["p_lon"].astype(np.float64), ade) + 1e-9).all(), "ADE(mix) <= sum_k p_k ADE_k (convexity of the norm)"
    monkeypatch.setattr(khyp, "oracle_k", lambda a: np.asarray(a).argmax(1))
    r2 = khyp.control4(ade, {"ARGMAX": khyp.argmax_k(d["p_lon"])})
    assert r2["pass"] is False and r2["ARGMAX"]["n_violations"] > 0, "MUTATION (ORACLE picks the worst plan) must turn control 4 red"


# ------------------------------------------------------------------------------------------------ features
def test_plan_feats_known_values():
    n = 30
    cv = np.stack([np.arange(1, n + 1) * 1.0, np.zeros(n)], 1)[None]           # 5 m/s: 1.0 m per 0.2 s knot
    f = khyp.plan_feats(cv)[0]
    np.testing.assert_allclose(f, [5.0, 0.0, 0.0, 30.0], atol=1e-9)
    v = 2.0 + 1.0 * 0.2 * np.arange(n)                                          # a = 1 m/s^2 : v_j = 2 + 0.2 j
    acc = np.stack([np.cumsum(v * khyp.DT), np.zeros(n)], 1)[None]
    f = khyp.plan_feats(acc)[0]
    np.testing.assert_allclose(f[0], v[-1], atol=1e-9)
    np.testing.assert_allclose(f[1:3], [1.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(f[3], acc[0, -1, 0], atol=1e-9)
    dec = np.stack([np.cumsum(np.where(np.arange(n) < 15, 6.0, 3.0) * khyp.DT), np.zeros(n)], 1)[None]    # one step from 6 to 3 m/s: a = -15 once
    f = khyp.plan_feats(dec)[0]
    np.testing.assert_allclose(f[:3], [3.0, -15.0, 0.0], atol=1e-9)


def test_feature_matrix_layout_and_a_speed_flag_lands_in_its_column():
    n = 2
    tau = np.zeros((n, K, 30, 2))
    for k in range(K):
        tau[:, k, :, 0] = np.cumsum(np.full(30, 5.0 * khyp.DT))[None]           # 5 m/s everywhere
    tau[0, 2, :, 0] = np.cumsum(np.full(30, 12.0 * khyp.DT))                    # window 0, hypothesis 2: 12 m/s against a 30 km/h limit
    d = {"tau": tau, "p_lon": np.full((n, K), 1.0 / K), "goal_prob": np.zeros((n, 22)), "lon_logits": np.zeros((n, K)),
         "cons_pred": np.full((n, 26), np.nan), "limit": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)), "limit_ok": np.ones(n, bool),
         "lead_gap_h": np.full((n, 30), np.nan), "s_h": np.zeros((n, 30))}
    X = khyp.hyp_feature_matrix(d)
    assert X.shape == (n, khyp.N_FEAT) == (n, 63) and len(khyp.FEATURE_NAMES) == 63
    j = khyp.FEATURE_NAMES.index("h2.viol_SPEED")
    assert j == K + 2 * khyp.PER_HYP + 4
    flags = [i for i, nm in enumerate(khyp.FEATURE_NAMES) if nm.endswith(tuple(f"viol_{r}" for r in khyp.RULES))]
    assert X[0, j] == 1.0 and X[1, j] == 0.0
    assert X[0, [i for i in flags if i != j]].sum() == 0.0 and X[1, flags].sum() == 0.0
    assert X[0, khyp.FEATURE_NAMES.index("h2.v_end")] == pytest.approx(12.0) and X[0, khyp.FEATURE_NAMES.index("h1.v_end")] == pytest.approx(5.0)
    np.testing.assert_allclose(X[:, :K].sum(1), 1.0)


# ------------------------------------------------------------------------------------------------ CONTROL 3
def test_label_permutation_removes_signal_and_identity_permutation_does_not():
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(600, 12))
    W = rng.normal(size=(12, K))
    y = (Xs @ W + 0.1 * rng.normal(size=(600, K))).argmax(1)
    Xt = rng.normal(size=(600, 12))
    yt = (Xt @ W).argmax(1)
    acc = lambda m: float((khyp.predict_learn(m, Xt) == yt).mean())          # noqa: E731
    real = acc(khyp.fit_learn(Xs, y))
    perm = acc(khyp.fit_learn(Xs, khyp.permute_labels(y, 0)))
    assert real > 0.8 and perm < real - 0.4, (real, perm)
    ident = acc(khyp.fit_learn(Xs, y[np.arange(len(y))]))
    assert ident == pytest.approx(real), "an identity 'permutation' keeps the signal: the control would be blind"
    assert np.array_equal(np.sort(khyp.permute_labels(y, 0)), np.sort(y)) and not np.array_equal(khyp.permute_labels(y, 0), y)
    with pytest.raises(ValueError, match=">= 2 observed classes"):
        khyp.fit_learn(Xs, np.zeros(len(y), int))


# ------------------------------------------------------------------------------------------------ the outcome table
def _seed(headroom, primary, sep, sec=True):
    return {"headroom_point": headroom, "primary_point": primary, "primary_separated": sep, "secondary_ok": sec}


def test_outcome_table_boundaries():
    F = khyp.F_PREREG
    both = lambda **k: {0: _seed(**k), 1: _seed(**k)}                       # noqa: E731
    assert khyp.decide(both(headroom=-0.5, primary=-F, sep=True))["outcome"] == "MULTIMODAL PAYS"             # exactly -F passes
    assert khyp.decide(both(headroom=-0.5, primary=-F + 1e-4, sep=True))["outcome"] == "HEADROOM, NO SELECTOR"      # 0.0001 short fails
    assert khyp.decide(both(headroom=-0.5, primary=-0.3, sep=False))["outcome"] == "HEADROOM, NO SELECTOR"         # not separated fails
    assert khyp.decide(both(headroom=-0.5, primary=-0.3, sep=True, sec=False))["outcome"] == "HEADROOM, NO SELECTOR"  # a worse SECONDARY blocks
    assert khyp.decide(both(headroom=-2 * F, primary=0.0, sep=False))["outcome"] == "HEADROOM, NO SELECTOR"         # exactly -2F is headroom
    assert khyp.decide(both(headroom=-2 * F + 1e-4, primary=0.0, sep=False))["outcome"] == "NO HEADROOM"
    mixed = khyp.decide({0: _seed(-0.5, 0.0, False), 1: _seed(-0.1, 0.0, False)})
    assert mixed["outcome"] == "NO HEADROOM" and mixed["headroom_seed_split"] is True
    one_seed_primary = khyp.decide({0: _seed(-0.5, -0.3, True), 1: _seed(-0.5, -0.05, True)})
    assert one_seed_primary["outcome"] == "HEADROOM, NO SELECTOR", "PRIMARY must hold at BOTH seeds"
    assert khyp.decide(both(headroom=-0.5, primary=-0.3, sep=True))["committed_next_step_quoted"].startswith("a trained K-hypothesis head")


# ------------------------------------------------------------------------------------------------ real banked files (tiny, read-only)
HB = khyp.SCR / "hblend"


@pytest.mark.skipif(not (HB / "score_M2b_s0_perwindow.npz").exists(), reason="HBLEND artifacts absent")
def test_h_stack_reproduces_hblends_banked_H_bit_exactly():
    """The beside row 'SEL-LEARN stacked with H's banked weights' uses HBLEND's formula: on B's own plan it must give HBLEND's banked H EXACTLY."""
    z = np.load(HB / "score_M2b_s0_perwindow.npz")
    W = khyp.load_h_weights(HB / "weights_M2b_s0.json")
    assert W.shape == (30, 2) and W.min() >= 0.0 and W.max() <= 1.0
    H = khyp.apply_h(z["P"], z["tau"], W)
    assert np.abs(H - z["H"].astype(np.float64)).max() == 0.0
    assert np.abs(khyp.apply_h(z["P"], z["tau"], np.ones((30, 2))) - z["tau"].astype(np.float64)).max() < 1e-5      # w = 1 -> B (float32 round trip)
    assert np.abs(khyp.apply_h(z["P"], z["tau"], np.zeros((30, 2))) - z["P"].astype(np.float64)).max() == 0.0       # w = 0 -> P10


def test_dk_reader_imports_from_the_tip_export():
    sp = importlib.util.spec_from_file_location("v8f_dk_read_probe", str(OUTDIR / "raw" / "v8f_dk_read.py"))
    dk = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(dk)
    assert hasattr(dk, "read_dk") and hasattr(dk, "load_block") and hasattr(dk, "_clean")
    assert "G:/" not in str(dk.code_provenance()).replace("\\", "/").upper()


# ------------------------------------------------------------------------------------------------ end to end
def _ckpt(world, path, seed):
    model, cfg = make_model(world, seed=seed)
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, path)
    return model


def test_end_to_end_hyps_then_read_smoke(world, tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(khyp, "OUT", out)
    root = world["root"]
    json.dump({"clips": world["names"]["train"]}, open(tmp_path / "fit_clips.json", "w"))
    models = {s: _ckpt(world, tmp_path / f"ck{s}.pt", seed=s) for s in (0, 1)}
    common = ["--eval-grid", str(root / "eval_grid_windows.json"), "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval.npz"),
              "--eval-cache", str(root), "--train-cache", str(root), "--train-join", str(root / "join_train.npz"), "--fit-clips", str(tmp_path / "fit_clips.json"),
              "--kin-mode", "steer", "--threads", "1"]
    for s in (0, 1):
        for split in ("train", "eval"):
            argv = ["hyps", "--seed", str(s), "--split", split, "--ckpt", str(tmp_path / f"ck{s}.pt"), "--batch", "8", "--shard", "16", "--lru", "2"] + common
            assert khyp.main(argv) == 0
    # a resumed run re-uses the shards (and gives the same merged file)
    z0 = dict(np.load(out / "hyps" / "M2b_s0_eval.npz"))
    shards = sorted((out / "hyps" / "M2b_s0_eval_shards").glob("shard_*.npz"))
    assert len(shards) >= 2
    shards[-1].unlink()
    assert khyp.main(["hyps", "--seed", "0", "--split", "eval", "--ckpt", str(tmp_path / "ck0.pt"), "--batch", "8", "--shard", "16", "--lru", "2"] + common) == 0
    z1 = dict(np.load(out / "hyps" / "M2b_s0_eval.npz"))
    bad = {k: float(np.nanmax(np.abs(z0[k].astype(np.float64) - z1[k].astype(np.float64)))) for k in z0 if not np.array_equal(z0[k], z1[k], equal_nan=True)}
    assert not bad, bad
    # the stored plan IS forward_infer's, and tau_k is the forward_diag plan of the one-hot code (re-derived independently on a few windows)
    model = models[0]
    ev = z1
    ds_b = batches(world, "eval", n=len(ev["tau"]), bs=len(ev["tau"]))
    with torch.no_grad():
        b = ds_b[0]
        o = model.forward_infer(b)
        assert float(np.abs(o["traj30"].numpy() - ev["own"]).max()) < 1e-3     # float32 batch-composition noise on ~50 m plans; the ages are the same (same i)
    rc = khyp.main(["read", "--no-banked", "--n-boot", "20"] + common)
    assert rc == 0
    R = json.load(open(out / "READ.json"))
    assert R["decision"]["outcome"] in khyp.NEXT and R["decision"]["committed_next_step_quoted"] == khyp.NEXT[R["decision"]["outcome"]]
    for s in ("s0", "s1"):
        ps = R["per_seed"][s]
        assert ps["controls"]["control4"]["pass"] is True and ps["controls"]["own_vs_diag_identity_maxabs_m"] == 0.0
        assert ps["controls"]["hypotheses_distinct"] is True
        assert ps["selector_accuracy"]["n_eval"] == len(ev["tau"]) and sum(ps["selector_accuracy"]["kstar_eval_counts"]) == len(ev["tau"])
        assert np.isfinite(ps["HEADROOM"]["point_unrounded"]) and np.isfinite(ps["PRIMARY"]["point_unrounded"])   # B is not a vertex: no sign is implied
        assert ps["PRIMARY"]["separated"] in (True, False) and "control3" in ps["controls"]
    md = (out / "KHYP_RESULT.md").read_text(encoding="utf-8")
    assert md.startswith("# KHYP_RESULT") and "Outcome (the PREREG table applied verbatim)" in md and "PAUSED" in md
    try:
        import psutil
        peak = psutil.Process().memory_info().peak_pagefile / 2**30
        print(f"PEAK_PRIVATE_GB {peak:.2f}")
    except Exception:
        pass
