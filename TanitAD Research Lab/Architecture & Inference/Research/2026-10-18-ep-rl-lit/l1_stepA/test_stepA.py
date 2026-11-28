"""Tests for stepA.py + tts_cem.py on SYNTHETIC tensors (numpy; torch only to execute the LANDED aggregate).  CPU, a few seconds.

Every check is a function `chk_*() -> bool` (True = healthy).  `test_green` requires every check to hold on the real code; `test_red_under_mutation` applies a DELIBERATE MUTATION to the code
under test (a monkeypatch of one function) and requires the SAME check to turn False.  A check that cannot go red is not a check (CLAUDE.md: built, tested, and unreachable).

    L1_REFE_DIR=<.../2026-09-20-refe-plan/refe> python -m pytest -q test_stepA.py       (planner.py is read from there for the aggregate-equality check; missing = FAIL, never skip)
"""
import json
import math
import os
import sys
import tempfile
import ast

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stepA as A  # noqa: E402
import tts_cem as C  # noqa: E402

REFE_DIR = os.environ.get("L1_REFE_DIR") or os.path.normpath(os.path.join(HERE, "..", "..", "2026-09-20-refe-plan", "refe"))


# ============================================================================================ synthetic world
def plan_xy(v=8.0, curve=0.0, T=20):
    t = 0.2 * np.arange(1, T + 1)
    x = v * t
    y = curve * x ** 2 / 40.0
    yaw = np.arctan(2 * curve * x / 40.0)
    return np.stack([x, y, yaw], 1)


def toy_fan(rng, corrupt_last_yaw=True):
    fan = np.stack([plan_xy(v=rng.uniform(2, 14), curve=rng.uniform(-1.5, 1.5)) for _ in range(64)]).astype(np.float32)
    if corrupt_last_yaw:
        fan[:, -1, 2] = rng.uniform(-4, 4, 64)               # the raw last heading is corrupted (A7)
    return fan


def truth(ex):
    """true labels of an EXECUTED plan [20,3] -> (sub [6] nc,dac,ep,ttc,c,ddc ; pdms)"""
    x, y = ex[-1, 0], ex[-1, 1]
    nc = 0.0 if abs(y) > 2.0 else 1.0
    dac = 0.0 if abs(y) > 2.5 else 1.0
    ep = float(np.clip(x / 35.0, 0, 1))
    ttc = 1.0 if x < 32.0 else 0.0
    sub = np.array([nc, dac, ep, ttc, 1.0, 1.0])
    return sub, nc * dac * (5 * ep + 5 * ttc + 2 * 1.0) / 12.0


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def perfect_scorer(S):
    """logits whose probabilities ARE the true labels of the executed plan (a scorer that is right everywhere)"""
    out = []
    for tr in np.asarray(S, np.float64):
        sub, _ = truth(A.repair_last_heading_np(tr))
        out.append(logit(sub))
    return np.array(out, np.float32)


def adversarial_scorer(S):
    return -perfect_scorer(S)


def set_scorer(w_set):
    """set-CONDITIONAL toy: each row's EP logit shifts with the set mean (the self-attention premise)"""
    def f(S):
        lg = perfect_scorer(S)
        ep = np.asarray(S)[:, -1, 0] / 35.0
        lg[:, 2] = lg[:, 2] + w_set * (ep - ep.mean())
        return lg
    return f


def gradient_scorer(S):
    """prefers faster plans (EP rises with arclength) and tolerates everything else: a search has something to climb"""
    S = np.asarray(S, np.float64)
    lg = np.full((S.shape[0], 6), 4.0)
    lg[:, 2] = 0.25 * S[:, -1, 0] - 4.0
    return lg.astype(np.float32)


def flat_scorer(S):
    return np.full((np.asarray(S).shape[0], 6), 1.0, np.float32)


def make_bundle(N=6, n_logs=3, seed=0, scorer=perfect_scorer):
    rng = np.random.default_rng(seed)
    fans = np.stack([toy_fan(rng) for _ in range(N)])
    logits = np.stack([scorer(f) for f in fans])
    sub = np.zeros((N, 64, 6))
    pdms = np.zeros((N, 64))
    for n in range(N):
        for k in range(64):
            sub[n, k], pdms[n, k] = truth(A.repair_last_heading_np(fans[n, k]))
    return {"token": np.array([f"{n:016x}" for n in range(N)]), "log": np.array([f"log{n % n_logs}" for n in range(N)]), "traj_raw": fans, "logits": logits.astype(np.float32),
            "pick": np.argmax(A.aggregate(logits), 1), "sub": sub, "pdms": pdms, "valid": np.ones((N, 64), bool)}


def run_pipeline(scorer, N=6, n_logs=3, B=400):
    b = make_bundle(N, n_logs, scorer=scorer)
    pe = A.make_perts(b)
    sc = A.run_scores(scorer, b, pe, gate_tokens=2)
    sub = np.zeros(pe["anchor"].shape + (6,))
    pd = np.zeros(pe["anchor"].shape)
    for n in range(N):
        for q in range(pe["anchor"].shape[1]):
            sub[n, q], pd[n, q] = truth(pe["exec"][n, q])
    names = sorted(set(map(str, b["log"])))
    P, li, _ = A.assemble_pairs(b, pe, sc, {"sub": sub, "pdms": pd}, names)
    return A.analyse(P, li, len(names), B=B), sc, pe


# ============================================================================================ checks
def chk_sampler():
    rng = np.random.default_rng(1)
    sizes = rng.integers(20, 200, 24)
    toks, logs = [], []
    for i, s in enumerate(sizes):
        toks += [f"t{i:02d}_{j:04d}" for j in range(s)]
        logs += [f"log{i:02d}"] * int(s)
    a = A.sample_stratified(toks, logs, 400, 7)
    perm = rng.permutation(len(toks))
    b = A.sample_stratified([toks[i] for i in perm], [logs[i] for i in perm], 400, 7)
    cnt = {f"log{i:02d}": sum(1 for t in a if t.startswith(f"t{i:02d}_")) for i in range(24)}
    spare, tot = 400 - 24, int((sizes - 1).sum())
    ideal = {f"log{i:02d}": 1 + spare * (sizes[i] - 1) / tot for i in range(24)}
    return bool(len(a) == 400 == len(set(a)) and a == b and all(c >= 1 for c in cnt.values()) and all(abs(cnt[k] - ideal[k]) <= 1.0 for k in cnt))


def chk_g0():
    ok = A.g0_check(["a", "b"], ["L1", "L2"], ["x", "y"], ["L3", "L4"])
    bad = A.g0_check(["a", "b"], ["L1", "L2"], ["x", "b"], ["L3", "L2"])
    return bool(ok["passed"] and not bad["passed"] and bad["logs_overlap"] == 1 and bad["tokens_overlap"] == 1)


def chk_speed_preserves_path():
    cur = plan_xy(v=9.0, curve=1.2)
    P = np.concatenate([np.zeros((1, 3)), cur], 0)
    seg = P[1:, :2] - P[:-1, :2]
    last = seg[-1] / np.linalg.norm(seg[-1])

    def dist_to_extended_polyline(pt):
        best = 1e9
        for i in range(len(P) - 1):
            a, b = P[i, :2], P[i + 1, :2]
            ab = b - a
            t = np.clip(np.dot(pt - a, ab) / np.dot(ab, ab), 0, 1)
            best = min(best, np.linalg.norm(pt - (a + t * ab)))
        a = P[-1, :2]                                                              # extension past the end along the last segment
        t = max(np.dot(pt - a, last), 0.0)
        return min(best, np.linalg.norm(pt - (a + t * last)))

    ok = True
    for s in (0.85, 0.92, 1.08, 1.15):
        q = A.scale_speed(cur, s)
        ok &= all(dist_to_extended_polyline(p[:2]) < 1e-7 for p in q)
    st = plan_xy(v=8.0, curve=0.0)
    ok &= bool(np.allclose(A.scale_speed(st, 1.15)[:, 0], 1.15 * st[:, 0]) and np.allclose(A.scale_speed(st, 0.85)[:, 0], 0.85 * st[:, 0]))
    return bool(ok)


def chk_lateral():
    st = plan_xy(v=8.0, curve=0.0)
    left, right = A.lateral_offset(st, 0.3), A.lateral_offset(st, -0.3)
    ramp = left[:, 1]
    return bool(abs(left[-1, 1] - 0.3) < 1e-9 and abs(right[-1, 1] + 0.3) < 1e-9 and np.allclose(left[:, 0], st[:, 0]) and ramp[0] < 0.01 and np.all(np.diff(ramp) > 0)
                and np.allclose(A.lateral_offset(st, 0.3, "const")[:, 1], 0.3))


def chk_identity_and_yaw():
    raw = toy_fan(np.random.default_rng(3))[0]
    ex, si = A.make_perturbation(raw, A.KIND_LON, 1.0)
    ex2, si2 = A.make_perturbation(raw, A.KIND_LAT, 0.0)
    ex3, si3 = A.make_perturbation(raw, A.KIND_LON, 1.15)
    return bool(np.array_equal(si, raw) and np.array_equal(ex, A.repair_last_heading_np(raw)) and np.array_equal(si2, raw) and ex3[-1, 2] == ex3[-2, 2] and si3[-1, 2] == raw[-1, 2]
                and not np.array_equal(si3[:, :2], raw[:, :2]))


def chk_aggregate_matches_landed():
    import torch
    src = open(os.path.join(REFE_DIR, "planner.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "REFePlanner")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "aggregate")
    w = next(n for n in cls.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "V1_W" for t in n.targets))
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "planner_aggregate", "exec"), ns)
    stub = type("S", (), {"rule": "navsim_v1", "V1_W": ast.literal_eval(w.value)})()
    lg = np.random.default_rng(0).normal(0, 2.5, (500, 6)).astype(np.float32)
    landed = ns["aggregate"](stub, torch.from_numpy(lg)).numpy()
    return bool(tuple(A.V1_W) == tuple(stub.V1_W) and np.allclose(landed, A.aggregate(lg), atol=1e-6))


def chk_concordance():
    num, den = A.concordance_counts([0, 0, 0, 0], [1, 1, -1, 0], [0, 0, 0, 0], [1, 1, 1, 0.01])        # 2 concordant, 1 discordant, 1 pair below the 0.05 true-difference threshold
    n2, d2 = A.concordance_counts([0, 0], [0, 0], [0, 0], [1, 1])                                        # scorer ties count 0.5
    return bool(num.sum() == 2 and den.sum() == 3 and n2.sum() == 1.0 and d2.sum() == 2)


def _pairs():
    # rows: 0 = scorer rises, true PDMS falls 0.5, NC 1->0 ; 1 = rises, falls 0.05 (below 0.1), nothing breached ; 2 = no rise, falls 0.9, DAC 1->0 ; 3 = rises 0.5, true rises
    n = 4
    lg_a = np.zeros((n, 6))
    lg_p = np.zeros((n, 6))
    lg_a[:, :] = [0, 0, 0, 0, 0, 0]
    lg_p[0] = lg_p[1] = lg_p[3] = [2, 2, 2, 2, 2, 2]                     # aggregate rises on pairs 0, 1, 3
    lg_p[2] = [-2, -2, -2, -2, -2, -2]
    asub = np.tile([1.0, 1, 0.5, 1, 1, 1], (n, 1))
    psub = asub.copy()
    psub[0, 0] = 0.0                                                      # NC 1 -> 0
    psub[2, 1] = 0.0                                                      # DAC 1 -> 0
    apd = np.full(n, 0.8)
    ppd = np.array([0.3, 0.75, 0.0, 0.9])
    return {"a_logit": lg_a, "p_logit": lg_p, "a_sub": asub, "p_sub": psub, "a_pdms": apd, "p_pdms": ppd}


def chk_a4_a5():
    T = A.admission_terms(_pairs())
    f = lambda k: (T[k][0].sum(), T[k][1].sum())
    return bool(f("A4") == (1.0, 3.0) and f("A5") == (1.0, 3.0) and f("A5b") == (1.0, 2.0))


def chk_bootstrap_is_log_clustered():
    rng = np.random.default_rng(0)
    L, n = 24, 200
    rate = np.where(np.arange(L) % 2 == 0, 0.55, 0.95)                    # strong between-log clustering
    num = np.array([rng.binomial(n, r) for r in rate], float)
    den = np.full(L, float(n))
    r = A.ratio_ci(num, den, A.boot_indices(L, 2000, 1))
    return bool(abs(r["point"] - num.sum() / den.sum()) < 1e-12 and (r["hi"] - r["lo"]) > 0.08)


def chk_classify():
    ci = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi}
    cases = [
        (ci(0.90, 0.85, 0.95), ci(0.05, 0.02, 0.08), ci(0.01, 0.0, 0.03), "O1", "O1"),
        (ci(0.60, 0.50, 0.70), ci(0.05, 0.02, 0.08), ci(0.01, 0.0, 0.03), "O2", "O2"),          # A1 interval wholly below 0.80
        (ci(0.78, 0.70, 0.86), ci(0.05, 0.02, 0.08), ci(0.01, 0.0, 0.03), "O3", "O2"),          # the prereg's overlap: point < 0.80 AND the interval straddles it
        (ci(0.90, 0.85, 0.95), ci(0.20, 0.15, 0.26), ci(0.01, 0.0, 0.03), "O2", "O2"),          # A4 interval wholly above 0.10
        (ci(0.90, 0.85, 0.95), ci(0.05, 0.02, 0.08), ci(0.09, 0.04, 0.14), "O2", "O2"),          # A5 point above 0.05 has no interval clause
        (ci(0.90, 0.85, 0.95), ci(0.09, 0.05, 0.14), ci(0.01, 0.0, 0.03), "O3", "O3"),          # A4 straddles, points pass
    ]
    return all(A.classify(a1, a4, a5, "interval_aware") == ia and A.classify(a1, a4, a5, "literal") == li for a1, a4, a5, ia, li in cases)


def chk_pipeline_perfect_and_adversarial():
    good, _, _ = run_pipeline(perfect_scorer)
    bad, _, _ = run_pipeline(adversarial_scorer)
    g, b = good["all"], bad["all"]
    return bool(good["outcome"] == "O1" and g["A1"]["point"] == 1.0 and g["A4"]["point"] == 0.0 and g["A5"]["point"] == 0.0 and g["A3"]["point"] == 1.0
                and bad["outcome"] == "O2" and b["A1"]["point"] == 0.0 and good["n_pairs"] > 100 and good["n_pairs_lon"] > 0 and good["n_pairs_lat"] > 0 and good["lon"]["A1"]["den"] > 0
                and good["strategic"].startswith("PAUSED"))


def chk_anchors():
    rng = np.random.default_rng(5)
    ok = True
    for i in range(20):
        lg = rng.normal(0, 2, (64, 6))
        a = A.choose_anchors(lg, f"tok{i}")
        agg = A.aggregate(lg)
        top = [x for x, r in a if r == "top"]
        rnd = [x for x, r in a if r == "rand"]
        ok &= bool(len(a) == 16 and set(top) == set(np.argsort(-agg)[:8]) and not (set(top) & set(rnd)) and len(set(rnd)) == 8 and a == A.choose_anchors(lg, f"tok{i}"))
    return ok


def chk_g2_identity_and_dependence():
    fan = toy_fan(np.random.default_rng(2))
    sf = set_scorer(0.3)
    ident = A.g2_identity(sf, fan, 7)
    dep = A.g2_set_dependence(sf, fan, 7)
    nodep = A.g2_set_dependence(perfect_scorer, fan, 7)                      # a set-independent scorer: the premise detector must say so
    return bool(ident["passed"] and ident["max_abs_diff"] == 0.0 and dep["premise_true"] and not nodep["premise_true"])


def chk_g3_reachability():
    fan = toy_fan(np.random.default_rng(2))
    r1 = A.g3_reachability(perfect_scorer, fan, 11, A.KIND_LON, 1.15)
    r2 = A.g3_reachability(perfect_scorer, fan, 11, A.KIND_LAT, -0.3)
    return bool(r1["passed"] and r2["passed"] and r1["slots_changed"] == [11])


def chk_cli_flags_reach_the_scorer():
    b = make_bundle(2, 2)
    with tempfile.TemporaryDirectory() as d:
        bp, op = os.path.join(d, "b.npz"), os.path.join(d, "p.npz")
        np.savez(bp, **b)
        A.main(["perturb", "--bundle", bp, "--out", op, "--speed-scales", "0.5", "--lat-offsets", "0.7"])
        pe = A.load_npz(op)
    lon, lat = pe["param"][pe["kind"] == A.KIND_LON], pe["param"][pe["kind"] == A.KIND_LAT]
    # and the perturbed plan really is the requested one, not the default grid
    ex, _ = A.make_perturbation(b["traj_raw"][0, int(pe["anchor"][0, 0])], A.KIND_LON, 0.5)
    return bool(set(lon.tolist()) == {0.5} and set(lat.tolist()) == {0.7} and pe["anchor"].shape == (2, 16 * 2) and np.array_equal(pe["exec"][0, 0], ex))


def chk_default_grid_is_the_prereg():
    pe = A.make_perts(make_bundle(1, 1))
    return bool(pe["anchor"].shape == (1, 96) and sorted({float(x) for x in pe["param"][pe["kind"] == 0]}) == [0.85, 0.92, 1.08, 1.15]
                and sorted({float(x) for x in pe["param"][pe["kind"] == 1]}) == [-0.3, 0.3] and A.N_BOOT == 10_000 and A.SEED == 20260927 and A.PREREG_SHA256.startswith("a76f62d8"))


def chk_g1():
    a = np.random.default_rng(0).uniform(0, 1, (64, 7))
    return bool(A.g1_check(a, a)["passed"] and not A.g1_check(a, a + 1e-3)["passed"] and not A.g1_check(a, np.where(np.arange(64)[:, None] == 3, np.nan, a))["passed"])


def chk_cem_identity():
    rng = np.random.default_rng(4)
    fan = toy_fan(rng)
    pick = int(np.argmax(A.aggregate(gradient_scorer(fan))))
    r0 = C.cem_search(gradient_scorer, fan, pick, n_iters=0)
    r5 = C.cem_search(gradient_scorer, fan, pick, n_iters=5, seed=1)
    return bool(not r0["moved"] and np.array_equal(r0["raw"], fan[pick]) and np.array_equal(r0["exec"], A.repair_last_heading_np(fan[pick])) and r5["moved"]
                and float(A.aggregate(A.score_replaced(gradient_scorer, fan, pick, r5["raw"])[pick])) > float(A.aggregate(gradient_scorer(fan)[pick])))


def chk_cem_scores_complete_sets():
    fan = toy_fan(np.random.default_rng(6))
    rec = A.Recorder(gradient_scorer)
    pick = 3
    C.cem_search(rec, fan, pick, n_iters=2, seed=2)
    shapes_ok = all(s.shape == (64, 20, 3) for s in rec.sets)
    final = rec.sets[2:]                                                      # after 2 population calls: the base fan, then elites replaced INSIDE the original fan
    diff_ok = all([i for i in range(64) if not np.array_equal(s[i], fan[i])] in ([], [pick]) for s in final)
    return bool(len(rec.sets) >= 4 and shapes_ok and diff_ok and np.array_equal(final[0], fan))


def chk_inert_detection():
    rng = np.random.default_rng(8)
    fans = [toy_fan(rng) for _ in range(12)]
    flat_out, flat_pick, grad_out, grad_pick = [], [], [], []
    for f in fans:
        pk = int(np.argmax(A.aggregate(flat_scorer(f))))
        r = C.cem_search(flat_scorer, f, pk, n_iters=5, seed=3)
        flat_out.append(r["exec"]); flat_pick.append(A.repair_last_heading_np(f[pk]))
        pk = int(np.argmax(A.aggregate(gradient_scorer(f))))
        r = C.cem_search(gradient_scorer, f, pk, n_iters=5, seed=3)
        grad_out.append(r["exec"]); grad_pick.append(A.repair_last_heading_np(f[pk]))
    vf, vg = C.inert_verdict(flat_out, flat_pick), C.inert_verdict(grad_out, grad_pick)
    return bool(vf["inert"] and vf["frac_moved"] == 0.0 and not vg["inert"] and vg["frac_moved"] > 0.5)


def chk_cli_sample_and_analyse_roundtrip():
    rng = np.random.default_rng(0)
    toks = [f"{i:016x}" for i in range(300)]
    logs = [f"log{i % 10}" for i in range(300)]
    b = make_bundle(6, 3)
    with tempfile.TemporaryDirectory() as d:
        tab = os.path.join(d, "t.npz")
        np.savez(tab, token=np.array(toks), log=np.array(logs))
        A.main(["sample", "--table", tab, "--n", "40", "--out", os.path.join(d, "s.json")])
        smp = json.load(open(os.path.join(d, "s.json")))
        pe = A.make_perts(b)
        sc = A.run_scores(perfect_scorer, b, pe, gate_tokens=1)
        sub = np.zeros(pe["anchor"].shape + (6,))
        pd = np.zeros(pe["anchor"].shape)
        for n in range(6):
            for q in range(pe["anchor"].shape[1]):
                sub[n, q], pd[n, q] = truth(pe["exec"][n, q])
        files = {"bundle": b, "perts": pe, "scores": {"p_logit": sc["p_logit"], "a_logit": sc["a_logit"]}, "labels": {"sub": sub, "pdms": pd}}
        args = ["analyse"]
        for k, v in files.items():
            np.savez(os.path.join(d, k + ".npz"), **v)
            args += [f"--{k}", os.path.join(d, k + ".npz")]
        A.main(args + ["--out", os.path.join(d, "R.json"), "--boot", "300"])
        res = json.load(open(os.path.join(d, "R.json")))
    return bool(smp["n"] == 40 == len(set(smp["tokens"])) and smp["prereg_sha256"] == A.PREREG_SHA256 and res["outcome"] == "O1" and res["all"]["A1"]["point"] == 1.0
                and set(res["inputs"]) == {"bundle", "perts", "scores", "labels"} and res["choices"]["lat_profile"] == "ramp" and res["choices"]["speed_scales"] == [0.85, 0.92, 1.08, 1.15])


def chk_gpu_refused_without_the_lock():
    old = os.environ.pop("L1_GPU_LOCKED", None)
    try:
        refused = False
        try:
            A.main(["score", "--bundle", "x", "--perts", "y", "--out", "z", "--ckpt", "c", "--ctx", "k", "--device", "cuda"])
        except SystemExit as e:
            refused = e.code == 4
        except Exception:
            refused = False
        os.environ["L1_GPU_LOCKED"] = "1"
        passes_guard = True
        try:
            A.require_lock_for_gpu("cuda")
        except SystemExit:
            passes_guard = False
        A.require_lock_for_gpu("cpu")
        return bool(refused and passes_guard)
    finally:
        os.environ.pop("L1_GPU_LOCKED", None)
        if old is not None:
            os.environ["L1_GPU_LOCKED"] = old


def chk_delta_and_spread():
    li = np.repeat(np.arange(6), 10)
    d = C.delta_ci(np.full(60, 0.4), li, 6, B=500)
    s = C.seed_spread([0.3, 0.5, 0.4])
    return bool(abs(d["point"] - 0.4) < 1e-12 and abs(d["lo"] - 0.4) < 1e-12 and abs(d["hi"] - 0.4) < 1e-12 and abs(s["range"] - 0.2) < 1e-12)


# ============================================================================================ mutations (one deliberate fault per check)
def mut_sampler(orig):
    def f(tokens, logs, n, seed=A.SEED):
        return sorted(np.random.default_rng(seed).choice(tokens, size=n, replace=False).tolist())        # drops the stratification
    return f


def mut_g0(orig):
    return lambda *a, **k: {**orig(*a, **k), "passed": True}


def mut_scale_about_origin(orig):
    def f(traj, s):
        out = np.array(traj, copy=True)
        out[:, :2] *= s                                                                                 # scales the geometry, not the speed profile
        return out
    return f


def mut_lateral_sign(orig):
    return lambda traj, d, profile=A.LAT_PROFILE: orig(traj, -d, profile)


def mut_scorer_in_ignores_raw_yaw(orig):
    def f(raw_anchor, kind, param, profile=A.LAT_PROFILE):
        ex, si = orig(raw_anchor, kind, param, profile)
        return ex, ex                                                                                   # the scorer would read the repaired heading
    return f


def mut_vw():
    return (5.0, 5.0, 4.0)


def mut_concordance_ignores_threshold(orig):
    return lambda s_a, s_p, t_a, t_p, thr=A.THR_TRUE_PAIR: orig(s_a, s_p, t_a, t_p, 0.0)


def mut_a4_direction(orig):
    def f(P):
        P2 = dict(P)
        P2["a_logit"], P2["p_logit"] = P["p_logit"], P["a_logit"]                                         # reads the move backwards
        return orig(P2)
    return f


def mut_pair_bootstrap(orig):
    def f(num, den, idx, alpha=0.05):
        n = int(np.sum(den))
        rng = np.random.default_rng(0)
        k = rng.binomial(n, float(np.sum(num) / n), size=idx.shape[0]) / n                             # resamples PAIRS, not logs
        return {"point": float(np.sum(num) / n), "lo": float(np.quantile(k, alpha / 2)), "hi": float(np.quantile(k, 1 - alpha / 2)), "n_valid_resamples": len(k), "den": float(n)}
    return f


def mut_classify(orig):
    def f(a1, a4, a5, mode="interval_aware"):
        r = orig(a1, a4, a5, mode)
        return {"O1": "O2", "O2": "O3", "O3": "O1"}[r]
    return f


def mut_identity_not_identity(orig):
    def f(traj_exec, s):
        out = orig(traj_exec, s)
        return out + (1e-6 if s == 1.0 else 0.0)                                                         # the identity perturbation leaks a bias
    return f


def mut_drop_perturbation_line(orig):
    def f(score_fn, fan_raw, slot, scorer_in):
        return np.asarray(score_fn(np.array(fan_raw, copy=True)), dtype=np.float64)                    # the replaced slot never reaches the scorer
    return f


def mut_grid_ignores_flags(orig):
    return lambda speed_scales=A.SPEED_SCALES, lat_offsets=A.LAT_OFFSETS: orig(A.SPEED_SCALES, A.LAT_OFFSETS)


def mut_drop_a_scale(orig):
    return lambda speed_scales=A.SPEED_SCALES, lat_offsets=A.LAT_OFFSETS: orig(speed_scales[:-1], lat_offsets)


def mut_anchors_random_from_all(orig):
    def f(logits, token, seed=A.SEED, n_top=A.N_TOP, n_rand=A.N_RAND):
        a = orig(logits, token, seed, n_top, n_rand)
        rng = np.random.default_rng(A._seed_for(token, seed))
        rnd = sorted(int(x) for x in rng.choice(64, size=n_rand, replace=False))                          # may re-pick the scorer's own top proposals
        return [x for x in a if x[1] == "top"] + [(i, "rand") for i in rnd]
    return f


def mut_g1_loose(orig):
    return lambda labelled, banked, tol=1e-6: orig(labelled, banked, 1e9)


def mut_cem_one_extra_iter(orig):
    return lambda *a, n_iters=C.K_ITERS, **k: orig(*a, n_iters=n_iters + 1, **k)


def mut_score_single(orig):
    return lambda score_fn, fan_raw, slot, scorer_in: np.asarray(score_fn(np.asarray(scorer_in)[None]), dtype=np.float64)    # scores the candidate ALONE, not as a 64-set


def mut_inert_never(orig):
    return lambda out, pick, thr_m=C.INERT_ADE_M, min_frac=C.INERT_FRAC: {**orig(out, pick, thr_m, min_frac), "inert": False}


def mut_composite_identity(orig):
    return lambda raw_pick, u, d, profile=A.LAT_PROFILE: (A.repair_last_heading_np(np.asarray(raw_pick, np.float32)), np.asarray(raw_pick, np.float32))     # the search samples nothing


def mut_analyse_ignores_labels(orig):
    def f(bundle, perts, scores, labels, log_names):
        lab = {"sub": np.repeat(bundle["sub"][:, :1], perts["anchor"].shape[1], 1), "pdms": np.repeat(bundle["pdms"][:, :1], perts["anchor"].shape[1], 1)}     # the perturbation labels never arrive
        return orig(bundle, perts, scores, lab, log_names)
    return f


def mut_guard_removed(orig):
    return lambda device: None


def mut_delta_not_clustered(orig):
    return lambda delta, li, n_logs, B=A.N_BOOT, seed=A.SEED: {"point": float(np.mean(delta)), "lo": float(np.mean(delta)) - 1.0, "hi": float(np.mean(delta)) + 1.0}


# (name, check, module, attribute, mutant factory or None for a constant)
TABLE = [
    ("sampler_stratified", chk_sampler, A, "sample_stratified", mut_sampler),
    ("g0_overlap_control", chk_g0, A, "g0_check", mut_g0),
    ("speed_keeps_the_path", chk_speed_preserves_path, A, "scale_speed", mut_scale_about_origin),
    ("lateral_sign_and_ramp", chk_lateral, A, "lateral_offset", mut_lateral_sign),
    ("identity_and_raw_last_yaw", chk_identity_and_yaw, A, "make_perturbation", mut_scorer_in_ignores_raw_yaw),
    ("identity_perturbation_is_exact", chk_g2_identity_and_dependence, A, "scale_speed", mut_identity_not_identity),
    ("aggregate_equals_landed", chk_aggregate_matches_landed, A, "V1_W", mut_vw),
    ("concordance_threshold", chk_concordance, A, "concordance_counts", mut_concordance_ignores_threshold),
    ("a4_a5_direction", chk_a4_a5, A, "admission_terms", mut_a4_direction),
    ("bootstrap_clusters_logs", chk_bootstrap_is_log_clustered, A, "ratio_ci", mut_pair_bootstrap),
    ("outcome_rule", chk_classify, A, "classify", mut_classify),
    ("pipeline_perfect_vs_adversarial", chk_pipeline_perfect_and_adversarial, A, "admission_terms", mut_a4_direction),
    ("anchors_top_plus_other_random", chk_anchors, A, "choose_anchors", mut_anchors_random_from_all),
    ("g3_perturbation_reaches_scorer", chk_g3_reachability, A, "score_replaced", mut_drop_perturbation_line),
    ("cli_flags_reach_perturbations", chk_cli_flags_reach_the_scorer, A, "perturbation_grid", mut_grid_ignores_flags),
    ("prereg_grid_is_the_default", chk_default_grid_is_the_prereg, A, "perturbation_grid", mut_drop_a_scale),
    ("g1_label_identity", chk_g1, A, "g1_check", mut_g1_loose),
    ("cem_identity_zero_iterations", chk_cem_identity, C, "cem_search", mut_cem_one_extra_iter),
    ("cem_scores_complete_sets", chk_cem_scores_complete_sets, A, "score_replaced", mut_score_single),
    ("inert_detection_flags_flat_scorer", chk_inert_detection, C, "inert_verdict", mut_inert_never),
    ("inert_detection_needs_a_live_search", chk_inert_detection, C, "composite", mut_composite_identity),
    ("delta_ci_clusters_logs", chk_delta_and_spread, C, "delta_ci", mut_delta_not_clustered),
    ("gpu_needs_the_launcher_lock", chk_gpu_refused_without_the_lock, A, "require_lock_for_gpu", mut_guard_removed),
    ("cli_analyse_reads_the_labels_file", chk_cli_sample_and_analyse_roundtrip, A, "assemble_pairs", mut_analyse_ignores_labels),
]
GREEN_ONLY = []                                                           # nothing is exempt from a mutation arm


@pytest.mark.parametrize("name,check", [(t[0], t[1]) for t in TABLE], ids=[t[0] for t in TABLE])
def test_green(name, check):
    assert check(), f"{name}: the check is not healthy on the real code"


@pytest.mark.parametrize("name,check,mod,attr,mutant", TABLE, ids=[t[0] for t in TABLE])
def test_red_under_mutation(name, check, mod, attr, mutant, monkeypatch):
    orig = getattr(mod, attr)
    monkeypatch.setattr(mod, attr, mutant(orig) if callable(mutant) and attr != "V1_W" else mutant())
    try:
        healthy = check()
    except Exception:                                                     # a mutant that makes the instrument crash is also caught (loudly), not green
        healthy = False
    assert not healthy, f"{name}: the check stayed GREEN under its deliberate mutation of {mod.__name__}.{attr}: it cannot detect the fault it exists for"


def test_prereg_file_hash_pinned():
    p = os.environ.get("L1_PREREG")
    if not p:
        pytest.skip("set L1_PREREG=<PREREG_L1_SEARCH_ADMISSION.md> to pin the hash")
    import hashlib
    b = open(p, "rb").read().replace(b"\r\n", b"\n")
    assert hashlib.sha256(b).hexdigest() == A.PREREG_SHA256
