"""v8F-mini paired TACTICAL / STRATEGIC families (`stack/scripts/v8f_paired_families.py`). CPU only; a synthetic world (no repo data, no checkpoint) whose answers
are known, plus one smoke test on the banked MIXO pair when its dumps are on this machine.

What is pinned, and the control that must go red for each (the mutants live in `raw/2026-10-25-pairfam2/mutate_pairfam.py`):
  * IDENTITY. The whole pipeline on A against A reads EXACTLY 0.0 [0.0, 0.0], not separated, on every cell (a shuffled B, a quadrature interval, a flipped
    `separated` all break it).
  * PAIRED, NOT QUADRATURE. On two strongly correlated arms the paired interval is much narrower than the quadrature of the two per-arm intervals.
  * PER-RESAMPLE F1. The paired macro-F1 interval equals an independent brute-force bootstrap (own confusion matrix, own loop, the same episode draws) that
    re-computes the F1 on every resample; a cell that computes F1 once collapses to a zero-width interval and fails it.
  * DIRECTION. A - B is A minus B: swapping the arms negates every delta, and a better arm reads negative on an error cell.
  * REPRODUCTION. Each arm's own points and intervals equal the programme's own functions (`four_families.tactical_from_trajectory`, `v8f_mini_eval.
    tactical_metrics` / `checkpoint_metrics`), and the JSON-reproduction control has teeth (a perturbed JSON value fails it).
  * N AND REASONS. Every masked cell's n is the mask's count, and its reason states the mask's own counts. A goal token with no positive is UNAVAILABLE with its
    n, never dropped. The heading cells are UNAVAILABLE (with the deviation) when the held-age rebuild does not reproduce the stored ck_pos_err.
  * REFUSALS (rc 2): different episode ids, different length, a dump on another grid, a within-episode reordering that the episode ids cannot see.
  * STRATEGIC is reported in full: the word PAUSED appears nowhere in the output.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for _p in (REPO / "stack", REPO / "taniteval"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SRC = Path(os.environ.get("V8F_PAIRFAM_SRC") or (HERE.parents[1] / "scripts" / "v8f_paired_families.py"))
pytestmark = pytest.mark.skipif(not SRC.exists(), reason="stack/scripts/v8f_paired_families.py is not in this checkout")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("v8f_paired_families_under_test", SRC) if SRC.exists() else None
NB = 120                       # bootstrap draws in the synthetic tests (the real read uses 2,000)
N_EP, PER_EP = 24, 12
N = N_EP * PER_EP


# ------------------------------------------------------------------------------------------------ the synthetic world
def _paths(rng, n):
    """[n, 30, 2] ego-frame paths on the 0.2 s grid with a mix of lane keeping / turns and braking / steady / accelerating."""
    v0 = rng.uniform(4.0, 14.0, n)
    acc = rng.choice([-2.5, 0.0, 1.8], n, p=[0.35, 0.35, 0.30])
    kap = rng.choice([0.0, 0.07, -0.07], n, p=[0.6, 0.2, 0.2])
    still = rng.random(n) < 0.05                                       # a parked vehicle: the human goal sits on the origin
    v0[still], acc[still] = 0.0, 0.0
    out = np.zeros((n, 30, 2))
    x = np.zeros(n)
    y = np.zeros(n)
    yaw = np.zeros(n)
    for k in range(30):
        v = np.maximum(v0 + acc * 0.2 * (k + 1), 0.0)
        yaw = yaw + v * kap * 0.2
        x = x + v * np.cos(yaw) * 0.2
        y = y + v * np.sin(yaw) * 0.2
        out[:, k, 0], out[:, k, 1] = x, y
    return out


def _bits(mask):
    return (mask.astype(np.int64) * (1 << np.arange(22, dtype=np.int64))).sum(1)


def make_ctx(seed=0):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(N_EP), PER_EP)
    gt = _paths(rng, N)
    has = rng.random(N) > 0.06
    lat_cls = np.where(has, rng.choice([0, 1, 2], N, p=[0.7, 0.15, 0.15]), -100)
    lat_cls[has & (rng.random(N) < 0.05)] = -100                       # a row with no lateral class (IGNORE)
    lon_cls = np.where(has, rng.integers(0, 7, N), -100)
    lon_cls[has & (rng.random(N) < 0.04)] = -100
    gy = rng.random((N, 22)) < 0.18
    gy[:, 7] = False                                                   # token 7: no positive anywhere
    gw = rng.random((N, 22)) < 0.7
    gw[:, 14] = False                                                  # token 14: never labelled
    gy &= gw
    rc = np.zeros((N, 3, 3), np.float32)
    for m, arc in enumerate((30.0, 50.0, 80.0)):
        rc[:, m, 0] = arc + rng.normal(0, 1.0, N)
        rc[:, m, 1] = rng.normal(0, 4.0, N)
        rc[:, m, 2] = rng.normal(0, 0.1, N)
    rc_ok = (rng.random((N, 3)) > np.array([0.03, 0.04, 0.06])) & has[:, None]
    pose_s = np.concatenate([rng.normal(0, 40, (N, 2)), rng.uniform(-3, 3, (N, 1))], 1).astype(np.float32)
    pose_now = (pose_s + np.concatenate([rng.normal(0, 3, (N, 2)), rng.normal(0, 0.1, (N, 1))], 1)).astype(np.float32)
    return {"eid": eid, "gt": gt, "has": has, "lat_cls": lat_cls.astype(np.int16), "lon_cls": lon_cls.astype(np.int16),
            "goal_y": _bits(gy).astype(np.uint32), "goal_w": _bits(gw).astype(np.uint32), "rc": rc, "rc_ok": rc_ok, "pose_s": pose_s, "pose_now": pose_now,
            "_gy": gy, "_gw": gw}


def make_dump(ctx, quality, seed):
    """One arm: `quality` in (0, 1] scales how close its outputs are to the truth (1 = near perfect)."""
    import torch
    from taniteval import v8f_consistency as C
    from tanitad.models.v8f_mini import transform_ckpt
    rng = np.random.default_rng(1000 + seed)
    noise = 2.0 * (1.0 - quality) + 0.05
    gt = ctx["gt"]
    scale = 1.0 + rng.normal(0, 0.25 * (1 - quality) + 0.01, (N, 1, 1))
    pred = (gt * scale + rng.normal(0, noise, gt.shape)).astype(np.float32)
    ade6 = np.linalg.norm(pred.astype(np.float64) - gt, axis=-1).mean(1)

    def logits(true, k):
        oh = np.zeros((N, k))
        t = np.where(true >= 0, true, 0)
        oh[np.arange(N), t] = 1.0
        return (3.0 * quality * oh + rng.normal(0, 1.0, (N, k))).astype(np.float32)

    gprob = np.clip(np.where(ctx["_gy"], 0.5 + 0.45 * quality, 0.5 - 0.45 * quality) + rng.normal(0, 0.25, (N, 22)), 0, 1).astype(np.float32)
    ck_now = (ctx["rc"] + rng.normal(0, 1, ctx["rc"].shape) * (6.0 * (1 - quality) + 0.2)).astype(np.float32)
    ckpt = transform_ckpt(torch.as_tensor(ck_now), torch.as_tensor(ctx["pose_now"]), torch.as_tensor(ctx["pose_s"])).numpy()
    d = {"pred": pred, "eid": ctx["eid"].astype(np.int64), "lat_logits": logits(ctx["lat_cls"], 5), "lon_logits": logits(ctx["lon_cls"], 7), "goal_prob": gprob,
         "ckpt": ckpt, "ade6": ade6}
    ck = transform_ckpt(torch.as_tensor(ckpt), torch.as_tensor(ctx["pose_s"]), torch.as_tensor(ctx["pose_now"])).numpy()
    d["ck_pos_err"] = C.checkpoint_errors(ck, ctx["rc"], ctx["rc_ok"])["pos_err_m"]
    d["_path"], d["_sha256"] = "<synthetic>", "0" * 64
    return d


@pytest.fixture(scope="module")
def world():
    ctx = make_ctx(0)
    return ctx, make_dump(ctx, 0.9, 1), make_dump(ctx, 0.5, 2)


def _cells(res):
    return {p: c for p, c in P.iter_cells({"tactical": res["tactical"], "strategic": res["strategic"]})}


@pytest.fixture(scope="module")
def pair_res(world):
    ctx, A, B = world
    return P.compute_pairs(A, B, ctx, NB, 0)


def test_the_world_is_diverse_enough_to_mean_something(world):
    ctx, A, B = world
    lat, lon, man, _ = P.plan_decisions(ctx["gt"])
    assert len(set(lat.tolist())) == 3 and len(set(lon.tolist())) == 3 and len(set(man.tolist())) >= 4, (np.bincount(lat), np.bincount(lon), np.bincount(man))
    assert 0 < (~ctx["rc_ok"][:, 0]).sum() < N and 0 < (~ctx["has"]).sum() < N


# ------------------------------------------------------------------------------------------------ identity, direction, pairing
def test_identity_every_cell_is_exactly_zero(world):
    ctx, A, _ = world
    r = P.identity_control(A, ctx, NB, 0)
    assert r["n_cells"] > 40, r["n_cells"]
    assert r["ok"] and r["n_not_zero"] == 0, r["not_zero"][:5]


def test_a_minus_b_is_a_minus_b_and_swapping_negates_every_delta(world, pair_res):
    ctx, A, B = world
    swapped = P.compute_pairs(B, A, ctx, NB, 0)
    ab, ba = _cells(pair_res), _cells(swapped)
    n = 0
    for p, c in ab.items():
        if c["status"] != "OK":
            continue
        d = ba[p]
        n += 1
        assert abs(c["delta"] + d["delta"]) < 1e-9, (p, c["delta"], d["delta"])
        assert abs(c["lo"] + d["hi"]) < 1e-9 and abs(c["hi"] + d["lo"]) < 1e-9, (p, c, d)
        assert abs((c["a_point"] - c["b_point"]) - c["delta"]) < 1e-4 + 5e-5, (p, c["a_point"], c["b_point"], c["delta"])      # delta really is a minus b
    assert n > 40


def test_the_better_arm_reads_negative_on_an_error_cell_and_positive_on_a_skill_cell(pair_res):
    cells = _cells(pair_res)
    for arc in ("30", "50", "80"):
        c = cells[f"strategic.position_error_m.{arc}"]
        assert c["status"] == "OK" and c["delta"] < 0 and c["separated"] and c["hi"] < 0, c
        h = cells[f"strategic.heading_error_deg.{arc}"]
        assert h["status"] == "OK" and h["a"] < h["b"] and h["delta"] < 0, h
    g = cells["tactical.goal_setting.goal_point_error_m"]
    assert g["delta"] < 0 and g["separated"]
    for p in ("tactical.heads.lateral.macro_f1", "tactical.heads.longitudinal.accuracy", "tactical.heads.goal.macro_f1"):
        assert cells[p]["delta"] > 0 and cells[p]["separated"], (p, cells[p])


def _perturbed_copy(d, rng, frac=0.05):
    """B = A with a small random fraction of its outputs changed: the two arms are strongly correlated window by window."""
    e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    flip = rng.random(N) < frac
    e["lat_logits"][flip] = rng.normal(0, 1, (int(flip.sum()), 5))
    e["lon_logits"][flip] = rng.normal(0, 1, (int(flip.sum()), 7))
    e["goal_prob"][flip] = rng.random((int(flip.sum()), 22))
    return e


def test_the_paired_interval_is_not_the_quadrature_of_two_intervals(world):
    ctx, A, _ = world
    B = _perturbed_copy(A, np.random.default_rng(7), 0.02)          # B differs from A on 2 % of the windows: the two arms are strongly correlated
    res = P.compute_pairs(A, B, ctx, NB, 0)
    ratio = {}
    for p, c in _cells(res).items():
        if c["status"] == "OK" and p.startswith("tactical.heads") and ".per_token." not in p:
            quad = float(np.hypot(c["a_ci"][1] - c["a_ci"][0], c["b_ci"][1] - c["b_ci"][0]))
            assert quad > 1e-3, p
            ratio[p] = (c["hi"] - c["lo"]) / quad
    assert len(ratio) == 8, sorted(ratio)
    ht = [v for k, v in ratio.items() if k.startswith(("tactical.heads.lateral", "tactical.heads.longitudinal"))]
    assert len(ht) == 6 and max(ht) < 0.5, ratio                  # paired, the interval is a fraction of the quadrature of the two per-arm intervals
    assert max(ratio.values()) < 0.85 and float(np.median(list(ratio.values()))) < 0.45, ratio


# ------------------------------------------------------------------------------------------------ per-resample F1 against an independent bootstrap
def _brute_macro_f1(true, pred, k):
    f = []
    for c in range(k):
        if (true == c).sum() == 0:
            continue
        tp = ((true == c) & (pred == c)).sum()
        fp = ((true != c) & (pred == c)).sum()
        fn = ((true == c) & (pred != c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def _brute_paired(true, pa, pb, eid, k, n_boot, seed):
    """The paired bootstrap, written out: the SAME drawn episodes for both arms, the F1 re-computed on every draw."""
    from taniteval.ci import episode_index
    uniq, idx = episode_index(eid)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        d.append(_brute_macro_f1(true[sel], pa[sel], k) - _brute_macro_f1(true[sel], pb[sel], k))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return _brute_macro_f1(true, pa, k) - _brute_macro_f1(true, pb, k), float(lo), float(hi)


@pytest.mark.parametrize("head,logits,cls,k", [("lateral", "lat_logits", "lat_cls", 5), ("longitudinal", "lon_logits", "lon_cls", 7)])
def test_the_head_macro_f1_interval_is_a_per_resample_bootstrap(world, pair_res, head, logits, cls, k):
    ctx, A, B = world
    m = ctx["has"] & (ctx["lat_cls" if head == "lateral" else "lon_cls"] >= 0)
    true = ctx[cls][m].astype(int)
    delta, lo, hi = _brute_paired(true, A[logits].argmax(-1)[m], B[logits].argmax(-1)[m], ctx["eid"][m], k, NB, 0)
    c = pair_res["tactical"]["heads"][head]["macro_f1"]
    assert abs(c["delta"] - delta) < 6e-5 and abs(c["lo"] - lo) < 6e-5 and abs(c["hi"] - hi) < 6e-5, (c["delta"], c["lo"], c["hi"], delta, lo, hi)
    assert c["hi"] - c["lo"] > 0.01, "a macro-F1 computed once would have no interval"
    assert c["n_windows"] == int(m.sum())


def test_the_goal_head_macro_f1_is_a_per_resample_bootstrap(world, pair_res):
    ctx, A, B = world
    gy, gw = ctx["_gy"], ctx["_gw"] & ctx["has"][:, None]
    sub = np.flatnonzero(ctx["has"])

    def mf1(p, rows):
        f = []
        for g in range(22):
            m = gw[rows, g]
            if m.sum() == 0 or gy[rows, g][m].sum() == 0:
                continue
            pp, yy = p[rows, g][m] > 0.5, gy[rows, g][m]
            tp, fp, fn = (pp & yy).sum(), (pp & ~yy).sum(), (~pp & yy).sum()
            f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.mean(f))

    from taniteval.ci import episode_index
    uniq, idx = episode_index(ctx["eid"][sub])
    rng = np.random.default_rng(0)
    d = []
    for _ in range(NB):
        sel = sub[np.concatenate([idx[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])]
        d.append(mf1(A["goal_prob"], sel) - mf1(B["goal_prob"], sel))
    lo, hi = np.percentile(d, [2.5, 97.5])
    c = pair_res["tactical"]["heads"]["goal"]["macro_f1"]
    assert abs(c["lo"] - lo) < 6e-5 and abs(c["hi"] - hi) < 6e-5 and c["hi"] - c["lo"] > 0.005, (c["lo"], c["hi"], lo, hi)
    assert abs(c["delta"] - (mf1(A["goal_prob"], sub) - mf1(B["goal_prob"], sub))) < 6e-5


# ------------------------------------------------------------------------------------------------ the programme's own functions
def test_every_per_arm_point_and_interval_is_the_programmes_own(world, pair_res):
    import torch
    from taniteval import four_families as ff
    from taniteval import v8f_consistency as C
    ctx, A, B = world
    cells = _cells(pair_res)
    for arm, d, key in (("A", A, "a"), ("B", B, "b")):
        lib = ff.tactical_from_trajectory(torch.as_tensor(d["pred"].astype(np.float64), dtype=torch.float32), torch.as_tensor(ctx["gt"], dtype=torch.float32),
                                          C.DT, eid=ctx["eid"], n_boot=NB, seed=0)
        for nm in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
            for met in ("accuracy", "kappa"):
                c, ref = cells[f"tactical.plan_derived.{nm}.{met}"], lib[nm]["ci"][met]
                assert abs(c[key] - ref["mean"]) < 1e-9 and abs(c[key + "_ci"][0] - ref["lo"]) < 1e-9 and abs(c[key + "_ci"][1] - ref["hi"]) < 1e-9, (arm, nm, met, c, ref)
        gs = lib["goal_setting"]
        for nm in ("goal_point_error_m", "goal_long_bias_m", "goal_lat_bias_m", "goal_bearing_mae_deg"):
            c, ref = cells[f"tactical.goal_setting.{nm}"], gs["ci"][nm]
            assert abs(c[key] - ref["mean"]) < 1e-9 and abs(c[key + "_ci"][0] - ref["lo"]) < 1e-9 and abs(c[key + "_ci"][1] - ref["hi"]) < 1e-9, (arm, nm, c, ref)
            assert c["n_windows"] == ref["n_windows"]
        assert round(cells["tactical.goal_setting.goal_bearing_bias_deg"][key + "_point"], 4) == gs["goal_bearing_bias_deg"]
        assert round(cells["tactical.goal_setting.goal_range_ratio"][key + "_point"], 4) == gs["goal_range_ratio"]


def test_the_head_and_strategic_cells_equal_the_evals_own_functions(world, pair_res):
    import torch
    from tanitad.models.v8f_mini import transform_ckpt
    from taniteval import v8f_consistency as C
    from taniteval.ci import episode_cluster_bootstrap
    ctx, A, B = world
    cells = _cells(pair_res)
    for d, key in ((A, "a"), (B, "b")):
        m = ctx["has"] & (ctx["lat_cls"] >= 0)
        f1 = C.macro_f1(np.where(m, ctx["lat_cls"], -100), d["lat_logits"].argmax(-1), 5)["macro_f1"]
        assert abs(cells["tactical.heads.lateral.macro_f1"][key + "_point"] - f1) < 1e-12
        m = ctx["has"] & (ctx["lon_cls"] >= 0)
        f1 = C.macro_f1(np.where(m, ctx["lon_cls"], -100), d["lon_logits"].argmax(-1), 7)["macro_f1"]
        assert abs(cells["tactical.heads.longitudinal.macro_f1"][key + "_point"] - f1) < 1e-12
        ck = transform_ckpt(torch.as_tensor(d["ckpt"]), torch.as_tensor(ctx["pose_s"]), torch.as_tensor(ctx["pose_now"])).numpy()
        e = C.checkpoint_errors(ck, ctx["rc"], ctx["rc_ok"])
        for mi, arc in enumerate((30, 50, 80)):
            ok = ctx["rc_ok"][:, mi]
            for fam, arr in (("position_error_m", e["pos_err_m"]), ("heading_error_deg", e["head_err_deg"])):
                ref = episode_cluster_bootstrap(arr[ok, mi], ctx["eid"][ok], n_boot=NB, seed=0)
                c = cells[f"strategic.{fam}.{arc}"]
                assert abs(c[key] - ref["mean"]) < 1e-9 and abs(c[key + "_ci"][0] - ref["lo"]) < 1e-9 and abs(c[key + "_ci"][1] - ref["hi"]) < 1e-9, (fam, arc, c, ref)
                assert c["n_windows"] == int(ok.sum()) and c["n_episodes"] == len(np.unique(ctx["eid"][ok]))


def test_the_goal_token_accuracy_and_f1_are_the_thresholded_head(world, pair_res):
    ctx, A, B = world
    gw = ctx["_gw"] & ctx["has"][:, None]
    for d, key in ((A, "a"), (B, "b")):
        p = d["goal_prob"] > 0.5
        acc = float(((p == ctx["_gy"]) & gw).sum() / gw.sum())
        assert abs(pair_res["tactical"]["heads"]["goal"]["token_accuracy"][key + "_point"] - acc) < 1e-12
        g = 3
        m = gw[:, g]
        tp, fp, fn = (p[m, g] & ctx["_gy"][m, g]).sum(), (p[m, g] & ~ctx["_gy"][m, g]).sum(), (~p[m, g] & ctx["_gy"][m, g]).sum()
        f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
        from tanitad.data import v8f_mini_data as D
        c = pair_res["tactical"]["heads"]["goal"]["per_token"][D.GOAL22[g]]
        assert c["status"] in ("OK", "POINT_ONLY") and abs(c[key + "_point"] - f1) < 1e-12


# ------------------------------------------------------------------------------------------------ n, reasons, unavailable cells
def test_every_masked_cell_states_its_n_and_the_masks_own_counts(world, pair_res):
    ctx, A, B = world
    N_ = len(ctx["eid"])
    n_no_row = int((~ctx["has"]).sum())
    cells = _cells(pair_res)
    for mi, arc in enumerate((30, 50, 80)):
        ok = ctx["rc_ok"][:, mi]
        for fam in ("position_error_m", "heading_error_deg"):
            c = cells[f"strategic.{fam}.{arc}"]
            assert c["n_windows"] == int(ok.sum()) < N_
            r = c["n_reason"]
            assert f"{int(ok.sum()):,} of {N_:,}" in r and f"{n_no_row:,} have no v9 label row" in r and f"{int((ctx['has'] & ~ok).sum()):,} have a row" in r, r
            assert f"{int((~ok).sum()):,} windows without a valid {arc} m" in r
    for head, cls in (("lateral", "lat_cls"), ("longitudinal", "lon_cls")):
        m = ctx["has"] & (ctx[cls] >= 0)
        r = cells[f"tactical.heads.{head}.macro_f1"]["n_reason"]
        assert f"{int(m.sum()):,} of {N_:,}" in r and f"{n_no_row:,} have no v9 label row" in r and f"{int((ctx['has'] & (ctx[cls] < 0)).sum()):,} have a row" in r, r
    mv = np.linalg.norm(ctx["gt"][:, -1].astype(np.float32).astype(np.float64), axis=-1) > 0.5
    assert 0 < (~mv).sum() < N_
    c = cells["tactical.goal_setting.goal_bearing_mae_deg"]
    assert c["n_windows"] == int(mv.sum()) and f"the {int((~mv).sum()):,} whose human goal" in c["n_reason"]
    assert cells["tactical.goal_setting.goal_point_error_m"]["n_windows"] == N_
    assert cells["tactical.plan_derived.lateral_decision.kappa"]["n_windows"] == N_


def test_a_token_with_no_positive_is_unavailable_with_its_n_never_dropped(pair_res):
    from tanitad.data import v8f_mini_data as D
    toks = pair_res["tactical"]["heads"]["goal"]["per_token"]
    assert list(toks) == list(D.GOAL22)
    for g, why in ((7, "no positive"), (14, "no positive")):
        c = toks[D.GOAL22[g]]
        assert c["status"] == "UNAVAILABLE" and c["n_positive"] == 0 and "n_windows" in c and why in c["reason"], c
    assert toks[D.GOAL22[14]]["n_windows"] == 0 and toks[D.GOAL22[7]]["n_windows"] > 0
    assert sum(c["status"] in ("OK", "POINT_ONLY") for c in toks.values()) == 20
    by_design = pair_res["tactical"]["unavailable_by_design"]
    assert by_design["anchor_selection"]["status"] == "UNAVAILABLE" and by_design["selected_vs_executed"]["status"] == "UNAVAILABLE"


def test_the_heading_cells_are_unavailable_with_the_deviation_when_the_rebuild_does_not_reproduce(world):
    ctx, A, B = world
    ok = P.compute_pairs(A, B, ctx, 40, 0)["strategic"]
    assert ok["heading_rebuild"]["status"] == "OK" and max(ok["heading_rebuild"]["rebuild_vs_stored_ck_pos_err_max_dev_m"].values()) < 1e-6
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in A.items()}
    bad["ckpt"] = bad["ckpt"] + np.random.default_rng(3).normal(0, 0.5, bad["ckpt"].shape).astype(np.float32)        # the raw tick-frame ckpt no longer matches ck_pos_err
    res = P.compute_pairs(bad, B, ctx, 40, 0)["strategic"]
    assert res["heading_rebuild"]["status"] == "UNAVAILABLE" and res["heading_rebuild"]["rebuild_vs_stored_ck_pos_err_max_dev_m"]["A"] > 1e-3
    for arc in ("30", "50", "80"):
        h = res["heading_error_deg"][arc]
        assert h["status"] == "UNAVAILABLE" and "does not reproduce" in h["reason"] and h["n_windows"] > 0
        assert res["position_error_m"][arc]["status"] == "OK"               # position comes straight from the stored ck_pos_err


def test_strategic_is_active_and_the_output_never_says_paused(world, pair_res):
    ctx, A, B = world
    assert P.STRATEGIC_STATUS.startswith("ACTIVE")
    txt = json.dumps(pair_res["strategic"], default=str)
    res = {"tier": P.TIER, "estimator": "x", "tactical": pair_res["tactical"], "strategic": pair_res["strategic"]}
    md = P.render_markdown(res, "X", "B")
    assert "PAUSED" not in txt and "PAUSED" not in md and "PAUSED" not in P.STRATEGIC_STATUS and "ACTIVE" in txt
    assert pair_res["strategic"]["arcs_m"] == [30, 50, 80]


def test_the_strategic_arcs_are_the_right_columns_and_the_mask_is_applied(world, pair_res):
    ctx, A, B = world
    for mi, arc in enumerate((30, 50, 80)):
        ok = ctx["rc_ok"][:, mi]
        assert np.isnan(A["ck_pos_err"][~ok, mi]).all() and not np.isnan(A["ck_pos_err"][ok, mi]).any()
        c = pair_res["strategic"]["position_error_m"][str(arc)]
        assert abs(c["a_point"] - float(A["ck_pos_err"][ok, mi].astype(np.float64).mean())) < 1e-9 and np.isfinite(c["delta"])
    h30, h80 = pair_res["strategic"]["heading_error_deg"]["30"], pair_res["strategic"]["heading_error_deg"]["80"]
    assert h30["a_point"] != h80["a_point"]


def test_the_identity_control_fails_on_a_deliberate_regression(world, monkeypatch):
    """A control that can only pass is not a control: feed it a pipeline whose B is not A and it must say so, cell by cell."""
    ctx, A, _ = world
    real = P.compute_pairs
    perturbed = _perturbed_copy(A, np.random.default_rng(11), 0.2)
    monkeypatch.setattr(P, "compute_pairs", lambda a, b, c, nb, sd: real(a, perturbed, c, nb, sd))
    r = P.identity_control(A, ctx, NB, 0)
    assert r["ok"] is False and r["n_not_zero"] >= 5 and len(r["not_zero"]) == r["n_not_zero"], r


# ------------------------------------------------------------------------------------------------ the controls have teeth
def _eval_raw(d, ctx):
    """The `raw` block an eval JSON would carry, from the programme's own functions."""
    import torch
    from tanitad.models.v8f_mini import transform_ckpt
    from taniteval import four_families as ff
    from taniteval import v8f_consistency as C
    scripts = str(SRC.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import v8f_mini_eval as E
    except Exception as e:                                                  # pragma: no cover
        pytest.skip(f"v8f_mini_eval is not importable here ({type(e).__name__})")
    X = dict(ctx)
    ck = transform_ckpt(torch.as_tensor(d["ckpt"]), torch.as_tensor(ctx["pose_s"]), torch.as_tensor(ctx["pose_now"])).numpy()
    cm = E.checkpoint_metrics(ck, X, NB)
    tf = ff.tactical_from_trajectory(torch.as_tensor(d["pred"].astype(np.float64), dtype=torch.float32), torch.as_tensor(ctx["gt"], dtype=torch.float32), C.DT,
                                     eid=ctx["eid"], n_boot=NB, seed=0)
    return {"tactical": E.tactical_metrics(d["goal_prob"], d["lat_logits"], d["lon_logits"], X), "strategic": {k: v for k, v in cm.items() if k != "_pw"},
            "families": {"tactical_from_trajectory": tf}}


def test_the_json_reproduction_control_passes_on_the_evals_own_blocks_and_fails_on_a_perturbed_one(world, pair_res):
    ctx, A, B = world
    cells = {"tactical": pair_res["tactical"], "strategic": pair_res["strategic"]}
    raw = json.loads(json.dumps(P._clean(_eval_raw(A, ctx))))                 # what the eval's jdump would have written
    r = P.json_reproduction("A", cells, raw)
    assert r["ok"] and r["n_checked"] > 100 and r["max_abs_diff"] <= 1e-6, r["failed"][:5]
    for tweak in (lambda x: x["tactical"]["lat"].__setitem__("macro_f1", x["tactical"]["lat"]["macro_f1"] + 0.01),
                  lambda x: x["strategic"]["pos_err_m@50"].__setitem__("hi", x["strategic"]["pos_err_m@50"]["hi"] + 0.05),
                  lambda x: x["families"]["tactical_from_trajectory"]["lateral_decision"]["ci"]["kappa"].__setitem__("mean", 0.0),
                  lambda x: x["tactical"]["goal_f1_per_token"].__setitem__("FOLLOW_LANE", 0.123)):
        bad = json.loads(json.dumps(raw))
        tweak(bad)
        rr = P.json_reproduction("A", cells, bad)
        assert not rr["ok"] and rr["n_failed"] >= 1, rr


# ------------------------------------------------------------------------------------------------ refusals
def _perm_within_episodes(d, ctx, seed):
    rng = np.random.default_rng(seed)
    perm = np.arange(N)
    for e in range(N_EP):
        rows = np.flatnonzero(ctx["eid"] == e)
        perm[rows] = rng.permutation(rows)
    return {k: (v[perm] if isinstance(v, np.ndarray) and v.shape[:1] == (N,) and k != "eid" else v) for k, v in d.items()}


def test_misaligned_windows_are_refused(world):
    ctx, A, B = world
    facts = P.check_alignment(A, B, ctx)
    assert facts["n_windows"] == N and facts["eid_identical"] and facts["ade6_recompute_max_dev"]["A"] == 0.0
    other = dict(B)
    other["eid"] = np.roll(B["eid"], 7)
    with pytest.raises(P.Refusal, match="different windows"):
        P.check_alignment(A, other, ctx)
    short = {k: (v[:-1] if isinstance(v, np.ndarray) and v.shape[:1] == (N,) else v) for k, v in B.items()}
    with pytest.raises(P.Refusal, match="rows|pred of shape"):
        P.check_alignment(A, short, ctx)
    shifted_ctx = dict(ctx)
    shifted_ctx["eid"] = (ctx["eid"] + 1) % N_EP
    with pytest.raises(P.Refusal, match="not on this grid"):
        P.check_alignment(A, B, shifted_ctx)
    with pytest.raises(P.Refusal, match="ade6"):                          # same episode ids, windows reordered inside the episodes: only the GT can see it
        P.check_alignment(A, _perm_within_episodes(B, ctx, 5), ctx)
    ctx2 = dict(ctx)
    ctx2["rc_ok"] = ctx["rc_ok"].copy()
    ctx2["rc_ok"][5, 0] = not ctx["rc_ok"][5, 0]
    with pytest.raises(P.Refusal, match="NaN mask"):
        P.check_alignment(A, B, ctx2)


def _write_world(tmp_path, ctx, A, B, name="x"):
    cpath = tmp_path / "ctx.npz"
    np.savez(cpath, **{k: ctx[k] for k in P.CTX_KEYS})
    pa, pb = tmp_path / f"{name}_a.npz", tmp_path / f"{name}_b.npz"
    np.savez(pa, **{k: A[k] for k in P.DUMP_KEYS})
    np.savez(pb, **{k: B[k] for k in P.DUMP_KEYS})
    return cpath, pa, pb


def test_the_cli_refuses_rc2_on_misaligned_dumps_and_writes_nothing(world, tmp_path, capsys):
    ctx, A, B = world
    for tag, bad in (("eid", dict(B, eid=np.roll(B["eid"], 11))), ("within", _perm_within_episodes(B, ctx, 9)),
                     ("short", {k: (v[:-5] if isinstance(v, np.ndarray) and v.shape[:1] == (N,) else v) for k, v in B.items()})):
        cpath, pa, pb = _write_world(tmp_path, ctx, A, bad, tag)
        out = tmp_path / f"out_{tag}.json"
        rc = P.main(["--a", str(pa), "--b", str(pb), "--out", str(out), "--ctx-npz", str(cpath), "--n-boot", "20", "--no-identity-control"])
        assert rc == 2 and not out.exists(), (tag, rc)
        assert "PAIRFAM REFUSED" in capsys.readouterr().err
    rc = P.main(["--a", str(tmp_path / "nope.npz"), "--b", str(tmp_path / "nope2.npz"), "--out", str(tmp_path / "o.json"), "--ctx-npz", str(cpath)])
    assert rc == 2


def test_the_cli_end_to_end_on_aligned_dumps(world, tmp_path, capsys):
    ctx, A, B = world
    cpath, pa, pb = _write_world(tmp_path, ctx, A, B)
    out, md = tmp_path / "o.json", tmp_path / "o.md"
    rc = P.main(["--a", str(pa), "--b", str(pb), "--out", str(out), "--md", str(md), "--ctx-npz", str(cpath), "--n-boot", "40", "--label-a", "X", "--label-b", "B"])
    assert rc == 0, capsys.readouterr().err
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["schema"] == P.SCHEMA and r["tier"] == "T1-class open loop (own held outputs; no recorded future action)"
    assert r["estimator"] == "paired episode-cluster bootstrap, B=40, 24 episodes" and r["n_windows"] == N and r["n_episodes"] == N_EP
    assert r["a"]["sha256"] == P.sha256_file(pa) and r["b"]["sha256"] == P.sha256_file(pb) and r["a"]["label"] == "X"
    idc = r["controls"][0]
    assert idc["ok"] is True and idc["n_not_zero"] == 0
    assert r["controls_ok"] is True
    assert "PAUSED" not in out.read_text(encoding="utf-8") and "PAUSED" not in md.read_text(encoding="utf-8")
    assert r["strategic"]["position_error_m"]["30"]["delta"] < 0 and r["a"]["npz"].endswith("x_a.npz")        # A is the better arm: it reads negative


def test_a_failed_control_is_rc3_and_still_writes_the_json(world, tmp_path, monkeypatch):
    ctx, A, B = world
    cpath, pa, pb = _write_world(tmp_path, ctx, A, B)
    out = tmp_path / "o.json"
    monkeypatch.setattr(P, "identity_control", lambda *a, **k: {"name": "identity", "ok": False, "n_cells": 1, "n_not_zero": 1, "not_zero": []})
    rc = P.main(["--a", str(pa), "--b", str(pb), "--out", str(out), "--ctx-npz", str(cpath), "--n-boot", "20"])
    assert rc == 3 and out.exists() and json.loads(out.read_text(encoding="utf-8"))["controls_ok"] is False


def test_the_cli_runs_the_json_reproduction_beside_the_dumps_and_rc3_on_a_perturbed_json(world, tmp_path):
    ctx, A, B = world
    cpath, pa, pb = _write_world(tmp_path, ctx, A, B)
    for d, path in ((A, pa), (B, pb)):
        raw = json.loads(json.dumps(P._clean(_eval_raw(d, ctx))))
        (tmp_path / (path.stem + ".json")).write_text(json.dumps({"tag": path.stem, "ages": "cycle", "kin_mode": "yaw", "raw": raw}), encoding="utf-8")
    out = tmp_path / "o.json"
    rc = P.main(["--a", str(pa), "--b", str(pb), "--out", str(out), "--ctx-npz", str(cpath), "--n-boot", str(NB), "--no-identity-control"])
    r = json.loads(out.read_text(encoding="utf-8"))
    rep = [c for c in r["controls"] if c["name"].startswith("reproduction")]
    assert rc == 0 and len(rep) == 2 and all(c["ok"] is True and c["n_checked"] > 100 and c["max_abs_diff"] <= 1e-6 for c in rep), [c.get("failed") for c in rep]
    assert r["ages"] == "cycle"
    jb = tmp_path / (pb.stem + ".json")
    j = json.loads(jb.read_text(encoding="utf-8"))
    j["raw"]["strategic"]["head_err_deg@50"]["mean"] += 0.01
    jb.write_text(json.dumps(j), encoding="utf-8")
    out2 = tmp_path / "o2.json"
    rc2 = P.main(["--a", str(pa), "--b", str(pb), "--out", str(out2), "--ctx-npz", str(cpath), "--n-boot", str(NB), "--no-identity-control"])
    r2 = json.loads(out2.read_text(encoding="utf-8"))
    assert rc2 == 3 and r2["controls_ok"] is False and out2.exists()
    bad = [c for c in r2["controls"] if c.get("ok") is False]
    assert len(bad) == 1 and bad[0]["failed"][0]["name"] == "strategic.heading_error_deg@50.mean"


# ------------------------------------------------------------------------------------------------ the banked pair (only where its dumps exist)
MIXO_EVAL = Path("C:/Users/Admin/box2_ship/logs/v8f_mixo/eval")
_RAW = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw"
_HAVE_MIXO = (MIXO_EVAL / "eval_M2b_s0.npz").exists() and (MIXO_EVAL / "eval_M2b-mixo_s0.npz").exists() and (_RAW / "join_eval141.npz").exists()


@pytest.mark.skipif(not _HAVE_MIXO, reason="the banked MIXO s0 dumps are not on this machine")
def test_the_banked_mixo_pair_aligns_and_reproduces_its_eval_jsons_points():
    ctx = P.build_context(str(_RAW / "eval_grid_windows.json"), str(_RAW / "eval141_kin.npz"), str(_RAW / "join_eval141.npz"))
    X, Bd = P.load_dump(str(MIXO_EVAL / "eval_M2b-mixo_s0.npz")), P.load_dump(str(MIXO_EVAL / "eval_M2b_s0.npz"))
    facts = P.check_alignment(X, Bd, ctx)
    assert facts["n_windows"] == 2399 and facts["n_episodes"] == 141
    res = P.compute_pairs(X, Bd, ctx, 30, 0)
    assert res["strategic"]["heading_rebuild"]["rebuild_vs_stored_ck_pos_err_max_dev_m"] == {"A": 0.0, "B": 0.0}
    cells = {"tactical": res["tactical"], "strategic": res["strategic"]}
    for arm, path in (("A", "eval_M2b-mixo_s0.json"), ("B", "eval_M2b_s0.json")):
        raw = json.loads((MIXO_EVAL / path).read_text(encoding="utf-8"))["raw"]
        r = P.json_reproduction(arm, cells, raw)
        points = [f for f in r["failed"] if not re.search(r"\.(lo|hi)$", f["name"])]         # the intervals need B = 2,000 draws; the points do not
        assert not points, points[:5]
    assert abs(Bd["ade6"].mean() - 2.1753) < 5e-5
