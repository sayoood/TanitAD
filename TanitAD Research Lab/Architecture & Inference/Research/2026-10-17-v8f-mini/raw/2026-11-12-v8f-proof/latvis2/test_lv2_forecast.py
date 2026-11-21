"""Tests of the read at equal calibration (note PREREG_V8F_PROOF_ADD38_N3): synthetic worlds only.
A world where the imagined state is only the present state moved toward the average must read FLAT; a world where it carries the future must be named;
every control has a deliberate regression that fails it."""
import numpy as np
import pytest

import lv2_forecast_read as F
import lv2_pod_tokens as T
import lv2_read as R

NC = T.NCELL


def world(kind, n=600, n_ep=30, dim=48, w=2, seed=0):
    """n windows of 16 own frames each; tokens of a frame = a fixed linear map of its state + noise; the future state = 0.5 x present + an innovation u the present does not hold (the future regresses toward the average, so a shrunk copy beats the copy).
    kind: 'shrink' (imagined = present moved toward the average), 'transport' (imagined = present + most of u), 'flat_future' (the future state holds nothing new)."""
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), n // n_ep)
    fold = ep % 5
    P = rng.normal(size=(n, dim))
    u = rng.normal(size=(n, dim)) * (0.0 if kind == "flat_future" else 0.87)
    Tt = 0.5 * P + u
    Xf = rng.normal(size=(n * 16, dim))
    fidx = np.arange(n * 16).reshape(n, 16)
    Xf[fidx[:, T.S0]] = P
    Xf[fidx[:, 14]] = Tt
    M = rng.normal(size=(dim, NC * w))
    Y = (Xf @ M + 2.0 * rng.normal(size=(n * 16, NC * w))).astype(np.float32)   # noisy targets and 48 to 116 inputs on 480 windows: the training skill lies above the held-out skill, as on real data
    A = rng.normal(size=(n, 10, 2))
    if kind == "shrink":
        Zim = 0.5 * P
    else:
        Zim = 0.5 * P + 0.8 * u + 0.1 * rng.normal(size=(n, dim))
    return Xf, fidx, fold, Zim, A, Y, ep


def run_world(kind, mutate=None, seed=0):
    Xf, fidx, fold, Zim, A, Y, ep = world(kind, seed=seed)
    res = {"fc_" + nm: np.full((len(fold), NC), np.nan, np.float32) for nm in T.FC_INPUTS}
    recs = []
    for k in range(5):
        out, rec = T.forecast_block(k, Xf, fidx, fold, Zim, A, Y, mutate=mutate)
        for nm in T.FC_INPUTS:
            res["fc_" + nm][out["win"]] = out["fc_" + nm]
        recs.append(rec)
    d = dict(res)
    d["eid"] = np.array([f"e{int(e):03d}" for e in ep])
    d["tok_change"] = T.cell_sq(Y[fidx[:, 14]], Y[fidx[:, T.S0]]).astype(np.float32)
    for nm in T.FC_INPUTS:
        d["fc_skill_heldout_" + nm] = np.array([r["skill_heldout_" + nm] for r in recs])
        d["fc_skill_train_" + nm] = np.array([r["skill_train_" + nm] for r in recs])
        d["fc_lam_" + nm] = np.array([r["lam_" + nm] for r in recs])
    d["fc_skill_heldout_permuted_I"] = np.array([r["skill_heldout_permuted_I"] for r in recs])
    return d


def stats_of(dumps, B=400):
    plan = R.boot_plan(len(dumps), len(np.unique(dumps[0]["eid"])), B, 0)
    out = {}
    for c, (b, a) in F.CONTRASTS.items():
        out[c], _ = R.summarize([R.seed_stats(F.view(d, b, a)) for d in dumps], plan, B)
    return out, F.fc_controls(dumps, plan, B)


def test_transport_world_is_named_and_controls_pass():
    dumps = [run_world("transport", seed=0), run_world("transport", seed=1)]
    st, ctrl = stats_of(dumps)
    assert all(c["pass"] for c in ctrl), [(c["name"], c["pass"]) for c in ctrl]
    assert F.above0(st["C1"]["D_mov"]) and F.above0(st["C2"]["D_mov"])
    assert st["C1"]["share_mov"]["pooled"] > 0.4


def test_shrink_world_reads_flat():
    dumps = [run_world("shrink", seed=0), run_world("shrink", seed=1)]
    st, ctrl = stats_of(dumps)
    assert not F.above0(st["C2"]["D_mov"])
    assert abs(st["C1"]["D_mov"]["pooled"]) < 0.02 * st["C1"]["H_mov"]["pooled"]      # the imagined state is a linear function of the present one: no gain at equal calibration
    # and the SAME world gives a clear gain against the copy when both go through one same-frame map (what D of the first read measures), so the two reads differ where they should
    Xf, fidx, fold, Zim, A, Y, ep = world("shrink", seed=0)
    tr = np.flatnonzero(np.repeat(fold, 16) != 0)
    m = T.ridge_fit(Xf[tr], Y[tr], 1e-2)
    w0 = np.flatnonzero(fold == 0)
    Yt = Y[fidx[w0, 14]]
    d_copy = T.cell_sq(T.ridge_apply(m, Xf[fidx[w0, T.S0]]), Yt).mean() - T.cell_sq(T.ridge_apply(m, Zim[w0]), Yt).mean()
    assert d_copy > 0


def test_known_zero_at_the_pod_level():
    d = run_world("transport", mutate="imagined_is_present")
    assert np.array_equal(d["fc_I"], d["fc_P"])


def test_f2_fails_when_the_future_holds_nothing_new():
    dumps = [run_world("flat_future", seed=0), run_world("flat_future", seed=1)]
    _, ctrl = stats_of(dumps)
    assert [c for c in ctrl if c["name"] == "f2"][0]["pass"] is False


def test_f3_and_f4_fail_on_their_regressions():
    dumps = [run_world("transport", seed=0), run_world("transport", seed=1)]
    bad = [dict(d) for d in dumps]
    bad[1]["fc_skill_heldout_permuted_I"] = np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    _, ctrl = stats_of(bad)
    assert [c for c in ctrl if c["name"] == "f3"][0]["pass"] is False
    bad = [dict(d) for d in dumps]
    bad[0]["fc_skill_train_PAI"] = bad[0]["fc_skill_heldout_PAI"] - 0.01
    _, ctrl = stats_of(bad)
    assert [c for c in ctrl if c["name"] == "f4"][0]["pass"] is False
    bad = [dict(d) for d in dumps]
    bad[0]["fc_skill_heldout_P"] = np.zeros(5)
    _, ctrl = stats_of(bad)
    assert [c for c in ctrl if c["name"] == "f4"][0]["pass"] is False


def test_f1_is_exact_and_reader_refuses_a_dump_without_forecasts():
    dumps = [run_world("transport", seed=0)]
    _, ctrl = stats_of(dumps)
    f1 = [c for c in ctrl if c["name"] == "f1"][0]
    assert f1["pass"] is True and f1["max_abs_episode_sum_of_D"] == [0.0]
    with pytest.raises(SystemExit):
        F.read_fc([{"eid": dumps[0]["eid"]}], "G", [0])
