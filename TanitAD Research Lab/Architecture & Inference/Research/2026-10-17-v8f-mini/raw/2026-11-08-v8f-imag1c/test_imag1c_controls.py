"""IMAG1C control 4 (IMPL_CHOICES_IMAG1C section 2): synthetic worlds through the SAME functions as the real read. Run: python test_imag1c_controls.py (pytest also collects the test_ functions).
Worlds: planted (a label channel in the bank latents, the windows' Oracle / Predicted / Copy are the latent of the horizon frame, a noisy copy of it, the start frame), copy, deranged labels, Gaussian."""
import os, sys, types
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imag1c_lib as ML  # noqa: E402
IL = ML.IL
sys.path.insert(0, os.path.join(HERE, "..", "capladder"))
import capladder_lib as CL  # noqa: E402

E, T, D = 40, 60, 24
BOOT = 500
R_T, N_T = 59, 8


def world(kind, seed=0, pred_noise=0.4):
    """kind: planted | copy | deranged | gauss. Bank = every frame of E episodes (E x T). Windows: every start g0 in 0..T-16, old = every 4th."""
    rng = np.random.default_rng(seed)
    beid = np.repeat([f"ep{e:03d}" for e in range(E)], T)
    bg = np.tile(np.arange(T), E)
    y = np.zeros((E, T))
    for e in range(E):
        y[e, 0] = rng.normal()
        for g in range(1, T):
            y[e, g] = 0.9 * y[e, g - 1] + np.sqrt(1 - 0.81) * rng.normal()
    present = np.zeros((E, T), bool)
    for e in range(E):                                     # lead present on a random half of the frames, in runs
        a = rng.integers(0, T - 30); present[e, a:a + rng.integers(20, 30)] = True
    u = rng.normal(size=D); u /= np.linalg.norm(u)
    c = types.SimpleNamespace()
    c.tags = list(ML.TAGS)
    c.BZ, c.Z0, c.PRED, c.ORA = {}, {}, {}, {}
    qe, g0 = [], []
    for e in range(E):
        for g in range(0, T - 15):
            qe.append(f"ep{e:03d}"); g0.append(g)
    qe, g0 = np.array(qe), np.array(g0)
    nw = len(qe)
    bidx = np.array([[int(e[2:]) * T + g + f for f in range(16)] for e, g in zip(qe, g0)])
    for tag in c.tags:
        trng = np.random.default_rng(100 + len(c.BZ))
        c.BZ[tag], c.Z0[tag], c.PRED[tag], c.ORA[tag] = {}, {}, {}, {}
        for l in ML.LEVELS:
            if kind == "gauss":
                X = trng.normal(size=(E * T, D))
            else:
                X = 1.5 * y.reshape(-1, 1) * u[None] + trng.normal(size=(E * T, D))
            c.BZ[tag][l] = X
            c.Z0[tag][l] = X[bidx[:, 0]]
            for k in ML.KS:
                fut = X[bidx[:, k * ML.STRIDE[l]]]
                c.ORA[tag][(l, k)] = fut
                c.PRED[tag][(l, k)] = c.Z0[tag][l].copy() if kind == "copy" else fut + pred_noise * trng.normal(size=fut.shape) * 0.5
    c.beid, c.bg, c.qe = beid, bg, qe
    flat = lambda a: a.reshape(-1)
    pf = flat(present)
    Y = {"lead_gap": np.where(pf, 10 + 4 * flat(y), np.nan), "lead_closing": np.where(pf, 0.5 * flat(y), np.nan), "ttc": np.where(pf, 3 + flat(y), np.nan), "speed": flat(y)}
    Yw = {q: Y[q][bidx] for q in Y}
    if kind == "deranged":                                  # the fit labels do not belong to the latents; the windows keep the TRUE horizon labels
        for q in Y:
            val = np.isfinite(Y[q])
            Y[q] = Y[q].copy()
            Y[q][val] = IL.perm_labels(Y[q][val], beid[val], np.random.default_rng(5))
    c.Y, c.Yw = Y, Yw
    c.old = (g0 % 4 == 0)
    ML.finish_ctx(c)
    c.W = IL.boot_weights(c.uniq, n_boot=BOOT)
    return c


def read(c, R=R_T, N=N_T, workers=None):
    real = ML.run_draw(c, 0, "real")
    refs = [ML.run_draw(c, s, "ref") for s in ML.REF_SEEDS[:R]]
    cals = [ML.run_draw(c, s, "cal") for s in ML.CAL_SEEDS[:N]]
    return ML.analyse(c, real, refs, cals)


_cache = {}


def get(kind):
    if kind not in _cache:
        c = world(kind)
        _cache[kind] = (c, read(c))
    return _cache[kind]


def test_a_planted_readable_and_imagines():
    c, A = get("planted")
    labs = {k: v["label"] for k, v in A["labels"].items() if k.startswith("all|") and k.endswith("|lead_gap")}
    far = [v for k, v in labs.items() if not k.startswith("all|1|1|")]
    print("planted labels", labs)
    assert all(v in ("IMAGINES", "COPIES") for v in labs.values()) and all(v != "UNREADABLE" for v in labs.values()), labs
    assert sum(v == "IMAGINES" for v in far) >= len(far) - 2, labs          # the noisy future beats the start frame at most horizons
    for tag in c.tags:                                                      # Oracle NMSE well below 1 where planted
        assert A["rows"][f"all|{tag}|2|3|lead_gap"]["nmse"]["oracle"]["point"] < 0.6


def test_b_copy_world_copies():
    c, A = get("copy")
    labs = {k: v["label"] for k, v in A["labels"].items() if k.startswith("all|") and k.endswith("|lead_gap")}
    print("copy labels", labs)
    assert all(v == "COPIES" for v in labs.values()) and not any(v in ("IMAGINES", "WORSE-THAN-COPY") for v in labs.values()), labs


def test_c_deranged_labels_unreadable():
    c, A = get("deranged")
    labs = {k: v["label"] for k, v in A["labels"].items() if k.startswith("all|") and not k.endswith("|speed")}
    print("deranged labels", sorted(set(labs.values())))
    assert all(v == "UNREADABLE" for v in labs.values()), labs


def test_d_gaussian_unreadable():
    c, A = get("gauss")
    labs = {k: v["label"] for k, v in A["labels"].items() if k.startswith("all|")}
    print("gauss labels", sorted(set(labs.values())))
    assert all(v == "UNREADABLE" for v in labs.values()), labs


def test_e_primal_equals_dual():
    rng = np.random.default_rng(1)
    n, d = 300, 20
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, d)
    y = X[:, 0] - 0.5 * X[:, 3] + rng.normal(size=n)
    ep = np.array([f"e{i // 10:02d}" for i in range(n)])
    inner = ML.inner_fold(ep)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    # dual ridge of CAPLADDER on the same standardised rows: one split, F = first 240 rows, S = last 60, at every c
    F, S = np.arange(240), np.arange(240, 300)
    K = Xs @ Xs.T
    sp = CL.Split(K, F, S, y[F][:, None])
    p = ML.fit_ridge(Xs[F], y[F], inner[F])
    # fit_ridge selects c by CV; compare at the selected c
    pred_primal = (Xs[S] - p["mu"]) @ p["w"] + p["ym"]
    pred_dual = sp.predict(CL.C_GRID[p["c"]])[:, 0]
    assert np.abs(pred_primal - pred_dual).max() < 1e-8, np.abs(pred_primal - pred_dual).max()
    # and the selection itself: same c as CAPLADDER's oof_fold on the same inner folds
    rows = np.arange(n)
    pr, cstar, _ = CL.oof_fold(K, rows, F, S, inner[F], y[F][:, None])
    assert int(cstar[0]) == p["c"], (cstar, p["c"])
    assert np.abs(pr[:, 0] - pred_primal).max() < 1e-8


def test_f_regression_arm_control2_fails():
    c, A = get("planted")
    assert A["control2"]["pass"] and A["control2"]["cells"] > 0, A["control2"]
    c2 = types.SimpleNamespace(**c.__dict__)
    c2.nf_idx = np.arange(len(c.qe))                         # not-future := the Oracle's own latent
    real = ML.run_draw(c2, 0, "real")
    refs = [ML.run_draw(c2, s, "ref") for s in ML.REF_SEEDS[:R_T]]
    cals = [ML.run_draw(c2, s, "cal") for s in ML.CAL_SEEDS[:2]]
    A2 = ML.analyse(c2, real, refs, cals)
    print("control 2 with not-future = Oracle:", A2["control2"]["pass"], len(A2["control2"]["failed"]), "failed cells of", A2["control2"]["cells"])
    assert not A2["control2"]["pass"]


def test_g_calibration_share():
    for kind in ("planted", "gauss"):
        c, A = get(kind)
        sh = A["control3"]["shares"]
        print(kind, {k: round(v["share"], 3) for k, v in sh.items() if k.startswith(("lead_gap", "READ3"))})
        assert A["control3"]["pass"], sh
        assert A["banner"] in ("VALID",) or kind == "gauss"


def test_h_reference_stream_is_deterministic():
    c, _ = get("planted")
    a = ML.run_draw(c, 7, "ref"); b = ML.run_draw(c, 7, "ref")
    assert a["tags"]["M2b_s0"]["ia2"] == b["tags"]["M2b_s0"]["ia2"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name, flush=True)
            except Exception as ex:
                import traceback; traceback.print_exc(); print("FAIL", name, repr(ex), flush=True); fails += 1
    print("passed" if fails == 0 else f"{fails} failed")
    sys.exit(1 if fails else 0)
