#!/usr/bin/env python3
"""Behavioural tests of scene_tap_probe.py.  Plain python (no pytest needed):  python test_scene_tap_probe.py  ->  ZZTEST_PASS <n>.

Every guard has a deliberate-regression arm: the test PASSES only if the GREEN path passes AND the MUTATED path raises `GuardFailed` (the guard is live).
(a) log-disjointness   (b) the arm reads the right tap   (c) P1 IS aux_s1_readout's function   (d) C's labels are permuted within the train fold only   (e) no held-out token in a training batch.
Plus: the fold rule against the pre-registered folds.json sha256, the committed outcome table (all four rows reachable, boundaries), the seed floor / separation rule, and an END-TO-END synthetic run of
the whole grid (35 units + 6 init floors) through `run_unit` and `assemble`, because the real assembly runs once at the end of a multi-hour job and must not be its own first run.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import scene_tap_probe as S  # noqa: E402

PREREG_FOLDS_SHA256 = "20dec3431663de2201190020f97c74f7975468a8e9e178f3db4d52fbacd5885f"      # PREREG_SCENE_TAP_PROBE.md section 2
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def expect_guard(fn, *a, **k):
    """The mutated arm: the guard MUST fire (RED = the regression was caught)."""
    try:
        fn(*a, **k)
    except S.GuardFailed:
        return True
    raise AssertionError(f"{getattr(fn, '__name__', fn)}: the deliberate regression was NOT caught (guard is dead)")


# ---- synthetic world ---------------------------------------------------------------------------------------------------------------------------------------------------
def synth_labels(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    ys = -32.0 + (np.arange(128) + 0.5) * 0.5
    out = np.zeros((n, 160, 128), np.int8)
    for i in range(n):
        w, off = rng.uniform(3, 8), rng.uniform(-2, 2)
        out[i] = np.clip(np.round((w - np.abs(ys - off)) * 10), -127, 127).astype(np.int8)[None, :]
    return torch.from_numpy(out)


def synth_universe(n_logs: int = 8, per_log: int = 5):
    tokens, tl = [], {}
    for l in range(n_logs):
        for j in range(per_log):
            t = f"tok{l:02d}_{j}"
            tokens.append(t)
            tl[t] = f"log{l:02d}"
    U = S.Universe(tokens, tl)
    U.fold = np.asarray([int(U.logs[i][3:]) % S.K_FOLDS for i in range(U.n)], np.int64)       # deterministic: log l -> fold l % 5 (every fold non-empty)
    return U


def synth_world(seed=0):
    U = synth_universe()
    g = torch.Generator().manual_seed(seed)
    ctxs = {"visual": torch.randn((U.n, 24, 256), generator=g), "scene": torch.randn((U.n, 8, 256), generator=g)}
    lab = synth_labels(U.n, seed)
    return U, ctxs, lab, S.LabelCache(U.keys, lab)


def oracle_iou_0_40(labels: torch.Tensor, logit: np.ndarray) -> float:
    """An INDEPENDENT numpy IoU (the test's own oracle): 2 m blocks of the int8 SDF -> mask fraction >= 0.5, cells whose centre x is in [0, 40) m."""
    s = labels.numpy().astype(np.float64) * 0.1
    frac = (s > 0).reshape(-1, 40, 4, 32, 4).mean((2, 4))
    tm, pm = frac >= 0.5, logit > 0
    xc = -8.0 + (np.arange(40) + 0.5) * 2.0
    band = ((xc >= 0.0) & (xc < 40.0))[None, :, None]
    return float((tm & pm & band).sum() / max(((tm | pm) & band).sum(), 1))


# ---- the folds ---------------------------------------------------------------------------------------------------------------------------------------------------------
@test
def test_fold_rule_is_the_committed_one():
    import hashlib
    for lg in ("2021.05.25.14.16.10_veh-35_00083_00485", "abc"):
        assert S.fold_of(lg) == int(hashlib.sha256(lg.encode("utf-8")).hexdigest(), 16) % 5


@test
def test_real_folds_match_the_preregistered_sha256_and_are_log_disjoint():
    U = S.load_universe()
    doc = json.load(open(S.TOKENS_JSON, encoding="utf-8"))
    b = S.dump_json_bytes(S.folds_doc(U.tokens, doc["token_log"]))
    assert S.sha256_bytes(b) == PREREG_FOLDS_SHA256, "folds.json drifted from the pre-registered sha256"
    assert open(os.path.join(HERE, "folds.json"), "rb").read() == b, "folds.json on disk != the rebuilt one"
    assert U.n == 1058 and len(set(U.logs.tolist())) == 123
    sizes = []
    for f in range(S.K_FOLDS):
        tr, ho = S.make_fold_split(U.fold, f)
        S.guard_split(tr, ho, U.logs, U.n)
        sizes.append(len(ho))
    assert sizes == [197, 242, 196, 218, 205] and sum(sizes) == U.n
    for lg in set(U.logs.tolist()):                                                  # every log sits in exactly one fold
        assert len(set(U.fold[U.logs == lg].tolist())) == 1


# ---- (a) log-disjointness ---------------------------------------------------------------------------------------------------------------------------------------------
@test
def test_a_guard_split_green_and_red():
    U = synth_universe()
    tr, ho = S.make_fold_split(U.fold, 2)
    S.guard_split(tr, ho, U.logs, U.n)                                                  # GREEN
    bad_tok = np.append(tr, ho[0])                                                      # RED 1: one token on both sides
    expect_guard(S.guard_split, bad_tok, ho, U.logs, U.n)
    same_log_other_tok = [i for i in tr if True][:0]                                    # RED 2: a token of a held-out LOG on the train side (different token, same log)
    tr2 = np.append(np.setdiff1d(tr, []), [])
    held_log = U.logs[ho[0]]
    cand = [i for i in range(U.n) if U.logs[i] == held_log and i not in set(ho.tolist())]
    assert not cand                                                                     # (a log is wholly in one fold by construction)
    ho_short = ho[1:]                                                                   # the other half of RED 2: move one held-out token to the train side
    tr_leak = np.append(tr, ho[0])
    expect_guard(S.guard_split, tr_leak, ho_short, U.logs, U.n)
    expect_guard(S.guard_split, tr[1:], ho, U.logs, U.n)                                # RED 3: a token on neither side (coverage)


@test
def test_a_run_unit_refuses_a_mutated_split():
    U, ctxs, lab, src = synth_world()
    real = S.make_fold_split

    def leaky(fold_arr, f):
        tr, ho = real(fold_arr, f)
        return np.append(tr, ho[0]), ho

    with patched(S, "DEPTH", 1), patched(S, "make_fold_split", leaky), tempfile.TemporaryDirectory() as d:
        expect_guard(S.run_unit, U, lab, ctxs, "V", 0, 0, n_updates=2, device="cpu", source=src, root=d)


# ---- (b) the arm reads the right tap -------------------------------------------------------------------------------------------------------------------------------
@test
def test_b_guard_tap_green_and_red():
    U, ctxs, lab, src = synth_world()
    S.guard_tap(S.select_ctx("visual", ctxs), "visual", ctxs)
    S.guard_tap(S.select_ctx("scene", ctxs), "scene", ctxs)
    expect_guard(S.guard_tap, ctxs["visual"], "scene", ctxs)                            # visual fed into S
    expect_guard(S.guard_tap, ctxs["scene"], "visual", ctxs)
    expect_guard(S.guard_tap, ctxs["visual"].clone(), "visual", ctxs)                   # a COPY is not the declared tensor (same values, other storage)


@test
def test_b_run_unit_refuses_visual_fed_to_S():
    U, ctxs, lab, src = synth_world()
    with patched(S, "DEPTH", 1), patched(S, "select_ctx", lambda tap, c: c["visual"]), tempfile.TemporaryDirectory() as d:
        expect_guard(S.run_unit, U, lab, ctxs, "S", 0, 0, n_updates=2, device="cpu", source=src, root=d)


@test
def test_b_the_head_input_shape_follows_the_tap_and_S_equals_V_except_the_tap():
    U, ctxs, lab, src = synth_world()
    seen = {}
    real_make = S.make_box

    def spy(tap, seed, device):
        box = real_make(tap, seed, device)
        box.head.register_forward_pre_hook(lambda m, args: seen.setdefault(tap, set()).add(tuple(args[0].shape[1:])))
        return box

    with patched(S, "DEPTH", 1), patched(S, "make_box", spy), tempfile.TemporaryDirectory() as d:
        S.run_unit(U, lab, ctxs, "V", 0, 0, n_updates=2, device="cpu", source=src, root=d)
        S.run_unit(U, lab, ctxs, "S", 0, 0, n_updates=2, device="cpu", source=src, root=d)
    assert seen["visual"] == {(24, 256)} and seen["scene"] == {(8, 256)}, seen
    bv, bs = real_make("visual", 3, "cpu"), real_make("scene", 3, "cpu")
    a, b = dict(bv.cfg.__dict__), dict(bs.cfg.__dict__)
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"tap"}, diff
    for (k1, v1), (k2, v2) in zip(bv.head.state_dict().items(), bs.head.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)                                         # same init: S = V except the tap


# ---- (c) P1 is aux_s1_readout's own function -----------------------------------------------------------------------------------------------------------------------
@test
def test_c_p1_function_identity_green_and_red():
    S.guard_p1_function()
    import aux_s1_readout as ASR
    assert S.P1_READ is ASR.perception_reads and S.STATS_FN is ASR.frame_stats
    copy = lambda *a, **k: ASR.perception_reads(*a, **k)                                # a behaviourally-identical COPY is still not the readout's own object
    with patched(S, "P1_READ", copy):
        expect_guard(S.guard_p1_function)
    with patched(S, "STATS_FN", lambda *a, **k: ASR.frame_stats(*a, **k)):
        expect_guard(S.guard_p1_function)


@test
def test_c_pooled_p1_equals_an_independent_oracle_and_a_reimplementation_goes_red():
    U, ctxs, lab, src = synth_world(1)
    s = lab.numpy().astype(np.float64) * 0.1
    frac = (s > 0).reshape(-1, 40, 4, 32, 4).mean((2, 4))
    rng = np.random.default_rng(5)
    logit = np.where(frac >= 0.5, 1.0, -1.0) * np.where(rng.random(frac.shape) < 0.8, 1.0, -1.0)            # a predictor with ~20 % flipped cells
    sdf = np.zeros_like(logit)
    res, _ = S.pooled_p1(U.keys, logit.astype(np.float32), sdf.astype(np.float32), src, n_boot=50)
    got = res["p1"]["all"]["0-40"]["est"]
    want = oracle_iou_0_40(lab, logit)
    assert abs(got - want) < 1e-12, (got, want)
    perfect, _ = S.pooled_p1(U.keys, np.where(frac >= 0.5, 1.0, -1.0).astype(np.float32), sdf.astype(np.float32), src, n_boot=50)
    assert perfect["p1"]["all"]["0-40"]["est"] == 1.0

    def reimplementation(pred, st, dz, n_boot=50, seed=0):                              # RED: a mean of per-frame IoUs is NOT the pooled ratio of sums
        iou = st["inter3"] / np.maximum(st["union3"], 1)
        out = {"p1": {"all": {"0-40": {"est": float(iou.mean()), "ci95": [0, 0]}}}, "p2": None, "k2": None}
        return out

    with patched(S, "P1_READ", reimplementation):
        res2, _ = S.pooled_p1(U.keys, logit.astype(np.float32), sdf.astype(np.float32), src, n_boot=50)
        assert abs(res2["p1"]["all"]["0-40"]["est"] - want) > 1e-6, "the mutated reader should differ from the oracle"
        expect_guard(S.guard_p1_function)


# ---- (d) C's labels are permuted within the train fold only --------------------------------------------------------------------------------------------------------
@test
def test_d_label_permutation_green_and_red():
    U, ctxs, lab, src = synth_world(2)
    tr, ho = S.make_fold_split(U.fold, 1)
    new, srcidx = S.permute_train_labels(lab, tr, 1)
    S.guard_label_perm(lab, new, srcidx, tr, ho)
    assert torch.equal(new[torch.as_tensor(ho)], lab[torch.as_tensor(ho)])             # held-out bit-identical
    assert (srcidx != tr).all() and sorted(srcidx.tolist()) == sorted(tr.tolist())
    assert not torch.equal(new[torch.as_tensor(tr)], lab[torch.as_tensor(tr)])
    new2, s2 = S.permute_train_labels(lab, tr, 1)
    assert torch.equal(new, new2) and (s2 == srcidx).all()                              # one FIXED permutation per fold
    new3, s3 = S.permute_train_labels(lab, tr, 2)
    assert not (s3 == srcidx).all()                                                     # another fold, another derangement
    allidx = np.arange(U.n)                                                             # RED 1: a permutation over ALL tokens touches the held-out fold
    srcall = np.asarray(S.AUX.derangement(list(allidx), 7), np.int64)
    bad = lab.clone()
    bad[torch.as_tensor(allidx)] = lab[torch.as_tensor(srcall)]
    expect_guard(S.guard_label_perm, lab, bad, srcall, tr, ho)
    bad2 = new.clone()                                                                  # RED 2: one held-out label changed
    bad2[int(ho[0])] = lab[int(tr[0])]
    expect_guard(S.guard_label_perm, lab, bad2, srcidx, tr, ho)
    fixed = srcidx.copy()                                                               # RED 3: a fixed point
    fixed[0] = tr[0]
    expect_guard(S.guard_label_perm, lab, new, fixed, tr, ho)


@test
def test_d_run_unit_C_trains_on_permuted_labels_and_evaluates_on_true_labels():
    U, ctxs, lab, src = synth_world(3)
    seen = {}
    real = S.train_unit

    def spy(ctx_all, labels_train, train_idx, held_idx, *a, **k):
        seen["labels"], seen["train"], seen["held"] = labels_train.clone(), np.asarray(train_idx), np.asarray(held_idx)
        return real(ctx_all, labels_train, train_idx, held_idx, *a, **k)

    with patched(S, "DEPTH", 1), patched(S, "train_unit", spy), tempfile.TemporaryDirectory() as d:
        S.run_unit(U, lab, ctxs, "C", 0, 3, n_updates=2, device="cpu", source=src, root=d)
        S.run_unit(U, lab, ctxs, "V", 0, 3, n_updates=2, device="cpu", source=src, root=d)
    # arm V gets the TRUE labels object; arm C's train rows were permuted, its held rows are the truth
    with patched(S, "DEPTH", 1), patched(S, "train_unit", spy), tempfile.TemporaryDirectory() as d:
        S.run_unit(U, lab, ctxs, "C", 0, 3, n_updates=2, device="cpu", source=src, root=d)
        lt, tr, ho = seen["labels"], seen["train"], seen["held"]
        assert not torch.equal(lt[torch.as_tensor(tr)], lab[torch.as_tensor(tr)])
        assert torch.equal(lt[torch.as_tensor(ho)], lab[torch.as_tensor(ho)])
        S.run_unit(U, lab, ctxs, "V", 0, 3, n_updates=2, device="cpu", source=src, root=d)
        assert torch.equal(seen["labels"], lab)
    real_perm = S.permute_train_labels

    def all_token_perm(labels, train_idx, fold, seed=S.PERM_SEED):
        idx = list(range(labels.shape[0]))
        s = np.asarray(S.AUX.derangement(idx, 11), np.int64)
        new = labels.clone()
        new[torch.as_tensor(idx)] = labels[torch.as_tensor(s)]
        return new, s

    with patched(S, "DEPTH", 1), patched(S, "permute_train_labels", all_token_perm), tempfile.TemporaryDirectory() as d:
        expect_guard(S.run_unit, U, lab, ctxs, "C", 0, 3, n_updates=2, device="cpu", source=src, root=d)


# ---- (e) no held-out token in a training batch ------------------------------------------------------------------------------------------------------------------
@test
def test_e_no_heldout_token_in_any_of_the_2000_batches():
    U = S.load_universe()
    for f in range(S.K_FOLDS):
        tr, ho = S.make_fold_split(U.fold, f)
        held = set(ho.tolist())
        trs = set(tr.tolist())
        seen = set()
        n = 0
        for b in S.draw_batches(tr, S.N_UPDATES, S.BATCH, 0 * 100 + f):
            assert len(b) == S.BATCH
            bs = set(b.tolist())
            assert bs <= trs and not (bs & held)
            seen |= bs
            n += 1
        assert n == S.N_UPDATES and len(seen) > 0.99 * len(tr)                          # 2,000 x 8 draws cover ~19 passes of the train fold


@test
def test_e_train_unit_guard_green_and_red():
    U, ctxs, lab, src = synth_world(4)
    tr, ho = S.make_fold_split(U.fold, 0)
    with patched(S, "DEPTH", 1):
        _h, info = S.train_unit(ctxs["scene"], lab, tr, ho, "scene", 0, 0, n_updates=3, device="cpu")
        assert info["updates"] == 3 and info["distinct_train_tokens_drawn"] > 0

        def drawer_ignoring_train_idx(train_idx, n_updates, batch, gen_seed):         # RED 1: draws from ALL tokens
            g = torch.Generator().manual_seed(gen_seed)
            for _ in range(n_updates):
                yield torch.randperm(U.n, generator=g)[:batch]

        expect_guard(S.train_unit, ctxs["scene"], lab, tr, ho, "scene", 0, 0, n_updates=40, device="cpu", draw=drawer_ignoring_train_idx)
        tr_leak = np.append(tr, ho)                                                    # RED 2: the held-out fold smuggled into the train indices
        expect_guard(S.train_unit, ctxs["scene"], lab, tr_leak, ho, "scene", 0, 0, n_updates=60, device="cpu")


# ---- schedule / loss are S1's ---------------------------------------------------------------------------------------------------------------------------------------
@test
def test_schedule_and_loss_are_the_s1_head_recipe():
    assert abs(S.lr_at(0, 2000) - 2e-4 / 50) < 1e-15 and abs(S.lr_at(49, 2000) - 2e-4) < 1e-15
    assert S.lr_at(50, 2000) == 2e-4 and S.lr_at(1999, 2000) < 1e-8
    assert all(S.lr_at(u, 2000) >= S.lr_at(u + 1, 2000) for u in range(50, 1999))
    box = S.make_box("visual", 0, "cpu")
    assert sum(p.numel() for p in box.head.parameters()) == 2440450 and box.w_aux == 0.1 and box.cfg.w_sdf == 1.0 and box.cfg.feed == "off" and box.cfg.depth == 2
    lab = synth_labels(2)
    lg, sd = torch.randn(2, 40, 32), torch.randn(2, 40, 32)
    l1, _b, _h = box.loss(lg, sd, lab, torch.ones(2, dtype=torch.bool))
    l2, _, _ = S.AUX.aux_loss(lg, sd, lab, torch.ones(2, dtype=torch.bool))
    assert float(l1) == float(l2)
    assert abs(float(box.combine(torch.zeros(()), l1)) - 0.1 * float(l1)) < 1e-7


# ---- the estimator and the committed outcome table -----------------------------------------------------------------------------------------------------
@test
def test_seed_floor_and_separation_rule():
    assert S.seed_floor([0.50, 0.52, 0.49]) == max(abs(0.50 - 0.52), abs(0.50 - 0.49), abs(0.52 - 0.49))
    assert S.seed_floor([0.5]) is None
    reps_a = np.full(1000, 0.60) + np.random.default_rng(0).normal(0, 0.005, 1000)
    reps_b = np.full(1000, 0.50) + np.random.default_rng(1).normal(0, 0.005, 1000)
    c = S.contrast(reps_a, reps_b, 0.60, 0.50, {"A": 0.02, "B": 0.03})
    assert c["ci_excludes_zero"] and c["abs_point_exceeds_all_floors"] and c["separated_beyond_floor"]
    c2 = S.contrast(reps_a, reps_b, 0.60, 0.50, {"A": 0.02, "B": 0.15})                  # CI excludes 0 but |point| is inside B's seed floor: NOT separated
    assert c2["ci_excludes_zero"] and not c2["separated_beyond_floor"]
    c3 = S.contrast(reps_b, reps_b + np.random.default_rng(2).normal(0, 0.05, 1000), 0.5, 0.5, {"A": 0.0, "B": None})
    assert not c3["separated_beyond_floor"] and c3["floors_missing"] == ["B"]


def _sv(sep, point):
    return {"separated_beyond_floor": sep, "point": point}


@test
def test_decide_reaches_all_four_rows_and_the_boundaries():
    assert S.decide(0.85, 0.30, _sv(False, -0.5))["row"] == "ROW1_V_CARRIES"
    assert S.decide(0.80, 0.30, _sv(False, 0.0))["row"] == "ROW1_V_CARRIES"             # >= 0.80 on the unrounded value
    assert S.decide(0.7999999, 0.30, _sv(False, 0.0))["row"] == "ROW4_SCENE_ELIMINATED"
    assert S.decide(0.50, 0.82, _sv(True, 0.32))["row"] == "ROW2_S_CARRIES"
    assert S.decide(0.50, 0.80, _sv(True, 0.30))["row"] == "ROW2_S_CARRIES"
    assert S.decide(0.50, 0.60, _sv(True, 0.10))["row"] == "ROW3_S_BETTER_NOT_ENOUGH"
    assert S.decide(0.50, 0.82, _sv(False, 0.32))["row"] == "ROW4_SCENE_ELIMINATED"      # S >= 0.80 but S-V not separated
    assert S.decide(0.50, 0.60, _sv(False, 0.10))["row"] == "ROW4_SCENE_ELIMINATED"
    assert S.decide(0.60, 0.50, _sv(True, -0.10))["row"] == "ROW4_SCENE_ELIMINATED"      # separated but ADVERSE
    for r in S.ROWS:
        assert S.ROWS[r][0] and S.ROWS[r][1]


@test
def test_insample_read_d1_d2():
    assert S.insample_read(0.50, 0.95)["read"] == "D1"
    assert S.insample_read(0.50, 0.60)["read"] == "D2"
    assert S.insample_read(0.50, 0.80)["read"] == "D1"
    assert S.insample_read(0.85, 0.99)["read"].startswith("NOT_APPLICABLE")


# ---- end to end -----------------------------------------------------------------------------------------------------------------------------------------------------------
def build_synthetic_grid(d, U, ctxs, lab, src, n_updates=2):
    with patched(S, "DEPTH", 1):
        for tap in ("visual", "scene"):
            for seed in (0, 1, 2):
                S.run_init(U, ctxs, tap, seed, device="cpu", root=d)
        for (arm, seed, fold) in S.UNIT_ORDER:
            S.run_unit(U, lab, ctxs, arm, seed, fold, n_updates=n_updates, device="cpu", source=src, root=d)


@test
def test_end_to_end_synthetic_grid_assemble_and_refusals():
    U, ctxs, lab, src = synth_world(7)
    assert len(S.UNIT_ORDER) == 35
    d = tempfile.mkdtemp(prefix="stp_e2e_")
    try:
        build_synthetic_grid(d, U, ctxs, lab, src)
        with patched(S, "DEPTH", 1):
            res = S.assemble(d, U, src, 2)
            assert res["decision"]["row"] in S.ROWS and res["n_tokens"] == U.n and res["n_logs"] == 8
            for arm, seeds in S.SEEDS.items():
                assert sorted(res["arms"][arm]["seeds"]) == [str(s) for s in seeds]
                for s in seeds:
                    assert 0.0 <= res["arms"][arm]["seeds"][str(s)]["heldout_iou_0_40"] <= 1.0
                    assert len(res["arms"][arm]["seeds"][str(s)]["insample_iou_0_40_per_fold"]) == 5
            assert res["arms"]["C"]["seed_floor"] is None and res["arms"]["V"]["seed_floor"] is not None
            assert set(res["contrasts"]) == {"S-V", "V-C", "S-C"} and res["contrasts"]["V-C"]["floors_missing"] == ["C"]
            assert set(res["init_floor"]) == {"visual", "scene"}
            S.write_result(d, res)
            json.loads(open(os.path.join(d, "PROBE_RESULT.json"), encoding="utf-8").read())
            # the pooled held-out read of one arm equals the read of its units, each token once: recompute V s0 independently
            lg = np.zeros((U.n, 40, 32), np.float32)
            sd = np.zeros_like(lg)
            for f in range(5):
                h, a, b, _m = S.load_unit(S.unit_path(d, "V", 0, f))
                lg[h], sd[h] = a, b
            r2, _ = S.pooled_p1(U.keys, lg, sd, src)
            assert r2["p1"]["all"]["0-40"]["est"] == res["arms"]["V"]["seeds"]["0"]["heldout_iou_0_40"]
            assert abs(r2["p1"]["all"]["0-40"]["est"] - oracle_iou_0_40(lab, lg)) < 1e-12
            # refusal 1: a unit with another config
            with patched(S, "WEIGHT_DECAY", 0.02):
                expect_guard(S.assemble, d, U, src, 2)
            # refusal 2: a token predicted twice / not at all
            p = S.unit_path(d, "S", 1, 2)
            h, a, b, m = S.load_unit(p)
            shutil.copy(p, p + ".bak")
            np.savez(p, held_idx=np.concatenate([h[:-1], h[:1]]), logit=a, sdf=b, meta_json=np.asarray(json.dumps(m)))
            expect_guard(S.assemble, d, U, src, 2)
            shutil.copy(p + ".bak", p)
            S.assemble(d, U, src, 2)
            # refusal 3 (c): the P1 function replaced
            with patched(S, "P1_READ", lambda *x, **k: S.ASR.perception_reads(*x, **k)):
                expect_guard(S.assemble, d, U, src, 2)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test
def test_real_selfcheck_universe_labels_and_permutation():
    U = S.load_universe()
    lab, src = S.load_true_labels(U)
    assert lab.shape == (1058, 160, 128) and lab.dtype == torch.int8
    tr, ho = S.make_fold_split(U.fold, 0)
    new, s = S.permute_train_labels(lab, tr, 0)
    S.guard_label_perm(lab, new, s, tr, ho)
    S.guard_p1_function()


def main() -> int:
    ok, bad = 0, []
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
            ok += 1
        except BaseException as e:                                                     # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            bad.append(fn.__name__)
    if bad:
        print(f"ZZTEST_FAIL {len(bad)} of {len(TESTS)}: {bad}")
        return 1
    print(f"ZZTEST_PASS {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
