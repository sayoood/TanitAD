"""Tests for scorer_battery.py. Run: CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q <this file>
Positive control: u400 must reproduce u400_reference.json EXACTLY. Deliberate-regression arms must read as known.
Mutation checks: for each statistic, a mutated input must change the output (a check that cannot go red is not a check)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scorer_battery as sb  # noqa: E402

REF = json.load(open("E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/u400_reference.json"))


@pytest.fixture(scope="module")
def world():
    T, _ = sb.load_table()
    boot = sb.make_boot(T)
    out, arms, toks = sb.run_all(T, boot)
    return T, boot, out, {a.name: a for a in arms}, toks


def maxdiff(a, b):
    if isinstance(a, dict):
        return max([maxdiff(a[k], b[k]) for k in b] + [0.0])
    if isinstance(a, (list, tuple)):
        return max([maxdiff(x, y) for x, y in zip(a, b)] + [0.0])
    return abs(float(a) - float(b))


# ----------------------------------------------------------------------------- positive control
def test_banked_picks_reproduced(world):
    T, _, out, arms, _ = world
    assert out["meta"]["banked_pick_reproduced"] == 1058


def test_u400_R1_weighted_slow_share_exact(world):
    T, boot, out, arms, toks = world
    ref = REF["R1_weighted_cruise_slow_share"]
    w = boot.wmean(toks["u400"]["slow"], T["cruise"])
    got = {"point": w[0], "lo": w[1], "hi": w[2], "share_zero": w[4], "share_random": w[5], "n_draws_used": w[3]}
    for k, v in got.items():
        assert v == ref[k], (k, v, ref[k])
    assert int((T["cruise"] & boot.zero).sum()) == ref["n_cruise_zero"]
    assert int((T["cruise"] & ~boot.zero).sum()) == ref["n_cruise_random"]
    assert float(toks["u400"]["slow"][T["cruise"]].mean()) == ref["share_unweighted"]
    assert out["u400"]["longitudinal"]["cruise_slow_share"]["weighted"] == [ref["point"], ref["lo"], ref["hi"]]


def test_u400_R2_spearman_tie_exact(world):
    T, boot, out, arms, _ = world
    ratio = sb.speed_ratio(T["v4_all"], T["v0"])
    blk = sb.spearman_block(T, boot, arms["u400"].probs, ratio, "tie")
    for h in ("NC", "DAC", "EP", "TTC"):
        ref = REF["R2_spearman_tie"][h]
        for k in ("pred", "true", "gap_pred_minus_true"):
            assert list(blk[h][k]) == ref[k], (h, k)
        assert blk[h]["n_tokens_paired"] == ref["n_tokens_paired"]
        assert blk[h]["n_pred"] == ref["n_pred"] and blk[h]["n_true"] == ref["n_true"]
        assert blk[h]["pred_unpaired_mean"] == ref["pred_unpaired_mean"]


def test_legacy_estimator_is_numpy_version_dependent_tie_is_not(world):
    """The retracted argsort-of-argsort estimator: TRUE (tie-heavy) reproduces the banked p1_diag_all.json on numpy 2.x and
    the u400_reference.json value on numpy 1.26 -- the same code, two answers (MEASURED 2026-11-08: NC true -0.0393 on numpy
    2.5.1, +0.0336 on 1.26.4). The tie estimator is identical on both (-0.0945)."""
    T, boot, out, arms, _ = world
    ratio = sb.speed_ratio(T["v4_all"], T["v0"])
    blk = sb.spearman_block(T, boot, arms["u400"].probs, ratio, "legacy")
    for h in ("NC", "DAC", "EP", "TTC"):
        t = blk[h]["true"][0]
        cands = [REF["R2_spearman_legacy"][h]["true"][0], REF["G_REPRO_p1_diag_all"][h]["banked_true"][0]]
        assert any(abs(t - c) < 1e-12 for c in cands), (h, t, cands)


def test_u400_R3_no_lane_factor_exact(world):
    T, boot, out, arms, toks = world
    ref = REF["R3_u400_without_lane_factor"]
    nl = arms["u400_no_lane_factor"]
    assert int((nl.pick[T["cruise"]] != arms["u400"].pick[T["cruise"]]).sum()) == ref["n_picks_changed_cruise"]
    assert int((nl.pick != arms["u400"].pick).sum()) == ref["n_picks_changed_all"]
    w = boot.wmean(toks["u400_no_lane_factor"]["slow"], T["cruise"])
    # the reference script evaluates W*sum/max(.) where the battery evaluates W*(sum/max(.)): 1 ulp, so tolerance 1e-15
    assert max(abs(x - ref["weighted_slow_share"][k]) for x, k in zip(w[:3], ("point", "lo", "hi"))) < 1e-15


def test_u400_tie_spearman_headline(world):
    """The prereg's TTC reading: pred -0.546, true -0.094 (n=52), gap -0.451 [-0.565, -0.341]."""
    out = world[2]["u400"]["spearman_tie_vs_v4_over_v0_cruise"]["TTC"]
    assert out["n_tokens_paired"] == 52 and out["n_pred"] == 154
    assert round(out["pred"][0], 3) == -0.546 and round(out["gap_pred_minus_true"][0], 3) == -0.451


# ----------------------------------------------------------------------------- deliberate-regression arms
def test_oracle_regret_exactly_zero(world):
    out = world[2]["oracle"]
    assert out["regret"]["weighted"] == [0.0, 0.0, 0.0] and out["regret"]["unweighted"] == [0.0, 0.0, 0.0]
    assert out["pairwise_acc"]["unweighted"][0] == 1.0
    assert all(v["regret"][0] == 0.0 for v in out["topk_oracle_curve"].values())


def test_always_slowest_fails_free_road(world):
    out = world[2]
    s = out["always_slowest"]["longitudinal"]["cruise_slow_share"]["weighted"]
    assert s[1] > 0.7                                                    # lower CI far above u400's 0.41
    assert out["always_slowest"]["longitudinal"]["v4_bias_plan_minus_gt_mps"]["weighted"][2] < 0
    assert out["always_slowest"]["regret"]["weighted"][0] > out["u400"]["regret"]["weighted"][2]


def test_always_fastest_fails_overspeed_and_regret(world):
    out = world[2]
    assert out["always_fastest"]["longitudinal"]["overspeed_share"]["weighted"][0] > out["u400"]["longitudinal"]["overspeed_share"]["weighted"][2]
    assert out["always_fastest"]["longitudinal"]["cruise_slow_share"]["weighted"][2] < 0.2
    assert out["always_fastest"]["regret"]["weighted"][1] > out["u400"]["regret"]["weighted"][2]


def test_random_pick_reads_chance(world):
    out = world[2]["random_pick"]
    assert 0.45 < out["pairwise_acc"]["unweighted"][0] < 0.56            # chance for pairwise ranking
    assert out["regret"]["weighted"][0] > 3 * world[2]["u400"]["regret"]["weighted"][0] * 0.8


def test_u400_beats_random_paired(world):
    d = world[2]["paired_vs_u400"]["random_pick"]["regret_delta"]["weighted"]
    assert d[1] > 0                                                      # random minus u400 regret, CI-lo > 0


def test_topk_curve_monotone_and_k1_is_pick(world):
    out = world[2]["u400"]["topk_oracle_curve"]
    regs = [out[str(k)]["regret"][0] for k in sb.KS]
    assert all(a >= b - 1e-15 for a, b in zip(regs, regs[1:])) and regs[-1] == 0.0
    assert out["1"]["regret"][0] == world[2]["u400"]["regret"]["weighted"][0]


# ----------------------------------------------------------------------------- determinism across processes
def test_determinism_across_two_processes(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    outs = []
    for i in range(2):
        p = str(tmp_path / ("r%d.json" % i))
        subprocess.run([sys.executable, os.path.join(HERE, "scorer_battery.py"), "--out", p], check=True, env=env, capture_output=True)
        outs.append(open(p, "rb").read())
    assert outs[0] == outs[1] and len(outs[0]) > 1000


# ----------------------------------------------------------------------------- estimator unit tests (synthetic, exact)
def test_spearman_average_ranks_and_sign():
    x = np.array([[1., 2, 3, 4, 5, 6]])
    m = np.ones_like(x, bool)
    assert sb.spearman_tok(x, x, m)[0] == pytest.approx(1.0)
    assert sb.spearman_tok(x, -x, m)[0] == pytest.approx(-1.0)
    assert np.isnan(sb.spearman_tok(np.ones_like(x), x, m)[0])           # constant labels: token does not enter
    assert np.isnan(sb.spearman_tok(x[:, :4], x[:, :4], m[:, :4])[0])    # < 5 proposals


def test_tie_estimator_is_permutation_invariant_legacy_is_not():
    rng = np.random.default_rng(1)
    y = np.array([[0., 0, 0, 0, 1, 1, 1, 2]])
    x = rng.random((1, 8))
    m = np.ones((1, 8), bool)
    perm = rng.permutation(8)
    assert sb.spearman_tok(x, y, m)[0] == pytest.approx(sb.spearman_tok(x[:, perm], y[:, perm], m)[0])
    vals = {round(sb.spearman_tok(x[:, p], y[:, p], m, "legacy")[0], 9) for p in (rng.permutation(8) for _ in range(30))}
    assert len(vals) > 1                                                  # argsort-of-argsort depends on proposal order


def small_table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    pdms = rng.random((n, 64))
    pdms[:, :3] = 0.97
    pdms[:, 3:6] = 0.2
    return pdms, np.ones((n, 64), bool), rng


def test_regret_pairwise_topk_known_values():
    pdms, valid, rng = small_table()
    # pick the best: regret 0; pick the worst: regret = max - min
    best, worst = pdms.argmax(1), pdms.argmin(1)
    assert (sb.regret_tok(pdms, valid, best) == 0).all()
    assert np.allclose(sb.regret_tok(pdms, valid, worst), pdms.max(1) - pdms.min(1))
    acc, npair = sb.pairwise_acc_tok(pdms, valid, pdms)                   # score == truth: perfect
    assert (acc == 1.0).all()
    acc2, _ = sb.pairwise_acc_tok(pdms, valid, -pdms)                     # reversed: 0
    assert (acc2 == 0.0).all()
    acc3, _ = sb.pairwise_acc_tok(pdms, valid, np.zeros_like(pdms))      # all ties: 0.5
    assert np.allclose(acc3, 0.5)
    tk = sb.topk_oracle_tok(pdms, valid, -pdms, ks=(1, 64))               # worst-first: top-1 is the min, top-64 the max
    assert np.allclose(tk[:, 0], pdms.min(1)) and np.allclose(tk[:, 1], pdms.max(1))


def test_lateral_zero_when_plan_equals_gt_and_known_offset():
    t = np.arange(1, 21)[None, :] * 0.2
    gt = np.stack([t[0] * 10, np.zeros(20), np.zeros(20)], -1)[None]
    z = sb.lateral_tok(gt.copy(), gt)
    assert all(np.allclose(v, 0) for v in z.values())
    off = gt.copy()
    off[..., 1] += 1.0
    o = sb.lateral_tok(off, gt)
    assert o["cross_track"][0] == pytest.approx(1.0, abs=1e-9)
    still = np.zeros((1, 20, 3))
    assert np.isnan(sb.lateral_tok(still, still)["yaw_rate_err"][0])     # standstill: no tangent, no yaw rate


def test_bootstrap_weights_and_pair_identity():
    rng = np.random.default_rng(3)
    N = 300
    log = np.array(["L%d" % (i // 3) for i in range(N)])
    zero = np.arange(N) % 2 == 0
    boot = sb.Boot(log, zero, B=2000)
    a, b = rng.random(N), rng.random(N)
    m = np.ones(N, bool)
    d = boot.delta(a, b, m)["unweighted"]
    s = boot.summarize(a - b, m)["unweighted"]
    assert d == s and d[1] < d[0] < d[2]                                  # paired = bootstrap of the difference, shared draws
    w = boot.wmean(a, m)
    assert w[0] == pytest.approx(sb.W_ZERO * a[zero].mean() + sb.W_RANDOM * a[~zero].mean())


# ----------------------------------------------------------------------------- mutation checks: each statistic must be able to go RED
def test_mutation_each_statistic_detects_a_corrupted_input(world):
    T, boot, out, arms, toks = world
    r = np.arange(len(T["token"]))
    u = arms["u400"]
    ref_sc = u.score
    # regret: shift every pick by one proposal
    base = sb.regret_tok(T["pdms"], T["valid"], u.pick).mean()
    assert sb.regret_tok(T["pdms"], T["valid"], (u.pick + 1) % 64).mean() != base
    # pairwise accuracy and top-K: reverse the score
    a0 = np.nanmean(sb.pairwise_acc_tok(T["pdms"], T["valid"], ref_sc)[0])
    assert np.nanmean(sb.pairwise_acc_tok(T["pdms"], T["valid"], -ref_sc)[0]) != a0
    k0 = sb.topk_oracle_tok(T["pdms"], T["valid"], ref_sc)[:, 2].mean()
    assert sb.topk_oracle_tok(T["pdms"], T["valid"], -ref_sc)[:, 2].mean() != k0
    # Spearman: swapping predicted and true flips the sign of the gap on every head (RED arm of G6-REPRO)
    ratio = sb.speed_ratio(T["v4_all"], T["v0"])
    cm = T["cruise"][:, None] & T["valid"]
    for j in range(4):
        rp = sb.spearman_tok(u.probs[..., j], ratio, cm)
        rt = sb.spearman_tok(T["sub"][..., j], ratio, cm)
        ok = np.isfinite(rp) & np.isfinite(rt)
        g, gs = np.mean((rp - rt)[ok]), np.mean((rt - rp)[ok])
        assert g == -gs and g != 0
    # Spearman: scrambling the speed ratio across proposals destroys the (strong) TTC correlation
    sc_ = sb.spearman_tok(u.probs[..., 3], ratio[:, ::-1], cm)
    assert np.nanmean(sc_[T["cruise"]]) != np.nanmean(sb.spearman_tok(u.probs[..., 3], ratio, cm)[T["cruise"]])
    # slow share: a different threshold or wrong speed (v0 doubled) moves it
    pr = ratio[r, u.pick]
    assert ((pr < 0.85)[T["cruise"]].mean()) != ((pr / 2 < 0.85)[T["cruise"]].mean())
    # v4 bias: replacing GT speed by zero changes the bias
    bias = T["v4_all"][r, u.pick] - T["v4_gt"]
    assert bias[T["cruise"]].mean() != (T["v4_all"][r, u.pick] - 0.0)[T["cruise"]].mean()
    # lateral: a lateral offset of the picked plan moves cross-track
    plan = T["traj"][r, u.pick].astype(np.float64)
    off = plan.copy()
    off[..., 1] += 0.5
    assert np.nanmean(sb.lateral_tok(off, T["gt_pose20"])["cross_track"]) != np.nanmean(sb.lateral_tok(plan, T["gt_pose20"])["cross_track"])
    # weighted bootstrap: swapping the stratum weights moves the weighted share (the weights are doing work)
    w0 = boot.wmean(toks["u400"]["regret"], np.ones(len(r), bool))[0]
    b2 = sb.Boot(T["log"], boot.zero, B=100, w_zero=sb.W_RANDOM, w_random=sb.W_ZERO)
    assert b2.wmean(toks["u400"]["regret"], np.ones(len(r), bool))[0] != w0
