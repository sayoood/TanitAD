"""test_trigger_probe4.py -- pytest for trigger probe 4 (PREREG_FROZEN.md 7).  SYNTHETIC data only for every fit (no real label is fitted by a test); the real inputs are touched read-only for structure
(array shapes / dtypes / fingerprints).  Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q -p no:cacheprovider test_trigger_probe4.py"""
from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ.setdefault("P4_THREADS", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import p4_lib as P  # noqa: E402
import p4_run as R  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

L = P.L
PY = sys.executable


# ------------------------------------------------------------------------------------------------ synthetic helpers
def planted(N, delta, seed, pos=(6, 10), T=160):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, T, 16)).astype(np.float32)
    y = (rng.random(N) < 0.5).astype(int)
    X[:, pos[0] * 20 + pos[1], 0] += delta * y
    E = rng.standard_normal((N, 8)).astype(np.float32)
    return X, E, y


def fit_planted(arch, delta, uniform=False, seed=0):
    Xtr, Etr, ytr = planted(7600, delta, 0)
    Xva, Eva, yva = planted(1900, delta, 1)
    Xte, Ete, yte = planted(3000, delta, 2)
    m, info = P.train_one(arch, (8, 20), Xtr, Etr, ytr, Xva, Eva, yva, seed, P.CFG, uniform=uniform)
    o, a = P.predict(m, Xte, Ete, want_attn=True)
    return roc_auc_score(yte, o), a, info


def fake_bundle(pop="pstop", N=600, n_clips=60, seed=0):
    rng = np.random.default_rng(seed)
    clip = np.repeat(np.arange(n_clips), N // n_clips)
    clip_sha = np.array([f"{int(x):012x}" for x in rng.integers(0, 2 ** 47, n_clips)])
    sha = clip_sha[clip]
    outer, inner = L.outer_fold(sha), L.inner_fold(sha)
    y = (rng.random(N) < 0.35).astype(int)
    d = {"idx": np.arange(N), "y": y, "clip": clip, "sha": sha, "outer": outer, "inner": inner, "E": rng.standard_normal((N, 8)), "v0": rng.random(N), "t": np.arange(N)}
    for k, p in (("TL_ANY", 0.7), ("TL_RED", 0.3), ("TL_GREEN", 0.3), ("TL_YELLOW", 0.01), ("NO_TL", 0.3), ("lead_now", 0.5), ("no_lead", 0.5), ("lead_depart", 0.25)):
        d[k] = rng.random(N) < p
    S = {"G8": rng.standard_normal((6, N, 2560)).astype(np.float32), "G4": rng.standard_normal((6, N, 640)).astype(np.float32)}
    bk = {"y": y, "clip": clip}
    for k in ("E", "V+E", "V+E(mlp)", "E(mlp)", "P3:G4+E", "P3:G8+E"):
        bk[k] = rng.standard_normal(N)
    return {"pop": pop, "d": d, "S": S, "bk": bk}


FAST = dict(P.CFG, max_epochs=2, n_boot=40)


# ------------------------------------------------------------------------------------------------ 1. reused code and inputs
def test_tp_lib_is_byte_identical():
    assert P.sha256_file(f"{HERE}/tp_lib.py") == P.TP_LIB_SHA
    assert P.sha256_file(f"{P.P3}/tp_lib.py") == P.TP_LIB_SHA
    assert P.sha256_file(f"{P.OLD}/tp_lib.py") == P.TP_LIB_SHA


def test_g4_and_projection_copies_equal_probe3():
    """p4_lib.g4_from_g8 / project_tokens equal probe 3's g_lib versions bit-for-bit (probe 3's module is imported in a SUBPROCESS so its global state never touches this process)."""
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, r'{P.P3}')\n"
        "import g_lib as G\n"
        "rng = np.random.default_rng(5)\n"
        "x = rng.standard_normal((6, 160, 1024)).astype(np.float32)\n"
        "pc = [{'mu': rng.standard_normal(1024).astype(np.float32), 'comp': rng.standard_normal((16, 1024)).astype(np.float32)} for _ in range(2)]\n"
        "np.savez(sys.argv[1], g4=G.g4_from_g8(x), pr=G.project_tokens(x, pc), pr4=G.project_tokens(G.g4_from_g8(x), pc))\n"
    )
    out = f"{HERE}/raw/_g_lib_ref.npz"
    os.makedirs(f"{HERE}/raw", exist_ok=True)
    subprocess.run([PY, "-I", "-c", code, out], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    ref = np.load(out)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, 160, 1024)).astype(np.float32)
    pc = [{"mu": rng.standard_normal(1024).astype(np.float32), "comp": rng.standard_normal((16, 1024)).astype(np.float32)} for _ in range(2)]
    assert np.array_equal(P.g4_from_g8(x), ref["g4"])
    assert np.allclose(P.project_tokens(x, pc), ref["pr"], rtol=1e-4, atol=1e-3)           # BLAS summation order differs across thread counts: not bit-identical, equal to float32 rounding
    assert np.allclose(P.project_tokens(P.g4_from_g8(x), pc), ref["pr4"], rtol=1e-4, atol=1e-3)


def test_real_token_files_layout_structure_only():
    """Header-only reads of probe 3's S files: float32 [6, 12000, T*16], token-major (token t, component j at column 16 t + j)."""
    for pop in P.POPS:
        s8 = np.load(f"{P.P3SCR}/S8_{pop}.npy", mmap_mode="r")
        s4 = np.load(f"{P.P3SCR}/S4_{pop}.npy", mmap_mode="r")
        assert s8.shape == (6, 12000, 160 * 16) and s4.shape == (6, 12000, 40 * 16)
        assert s8.dtype == np.float32 and s4.dtype == np.float32
        info = json.load(open(f"{P.P3RAW}/pcaG_{pop}.json"))
        assert info["shapes"] == {"S8": [6, 12000, 2560], "S4": [6, 12000, 640]}
        # fold mapping from code: pca_g.py fits PCA f on the outer-train windows (outer != f) for f = 0..4 and fold 5 on all rows
        assert [f["windows"] for f in info["folds"]][5] == 12000
    src = open(f"{P.P3}/pca_g.py").read()
    assert "outer fold f = all outer-train windows" in src and "fold 5 = all rows" in src
    g = open(f"{P.P3}/g_lib.py").read()
    assert "blocks = [b for b in range(5) if b != f]" in g and "for f in range(6)" in g


def test_fold_view_and_cv_uses_fold_proper_projection():
    B = fake_bundle(N=300, n_clips=30)
    asked = []

    def sget(f):
        asked.append(f)
        return B["S"]["G4"][f]

    cfg = dict(FAST, max_epochs=1)
    P.cv_oof_nn(sget, B["d"]["E"], B["d"]["y"], B["d"]["outer"], B["d"]["sha"], "max", P.GRID_HW["G4"], 0, cfg)
    assert asked == [0, 1, 2, 3, 4]                       # never the all-rows fit 5
    assert P.fold_view(B["S"]["G4"], 3) is not None and np.array_equal(P.fold_view(B["S"]["G4"], 3), B["S"]["G4"][3])


def test_standardisers_use_outer_train_rows_only(monkeypatch):
    B = fake_bundle(N=300, n_clips=30)
    seen = []
    orig = P.tok_standardiser
    monkeypatch.setattr(P, "tok_standardiser", lambda X: (seen.append(len(X)), orig(X))[1])
    P.cv_oof_nn(lambda f: B["S"]["G4"][f], B["d"]["E"], B["d"]["y"], B["d"]["outer"], B["d"]["sha"], "max", P.GRID_HW["G4"], 0, dict(FAST, max_epochs=1))
    assert seen == [int((B["d"]["outer"] != f).sum()) for f in range(5)]


def test_real_data_bundle_row_order_and_fingerprint():
    """real_data asserts: banked probe-1 y / clip equal load_pop's, and the S files equal probe 3's recorded PCA fingerprints.  (structure + hashes; no fit)"""
    B = P.real_data("pstop")
    assert B["S"]["G8"].shape == (6, 12000, 2560) and B["S"]["G4"].shape == (6, 12000, 640)
    assert len(B["d"]["y"]) == 12000 and B["bk"]["E"].shape == (12000,) and B["bk"]["V+E"].shape == (12000,)


def test_reachability_g8_with_s4_file_raises():
    """'built, tested, unreachable' class: the G8 entry point accepts ONLY the [6, 12000, 2560] file; the S4 file must raise (RED)."""
    with pytest.raises(AssertionError):
        P.open_tokens("G8", f"{P.P3SCR}/S4_pstop.npy")
    with pytest.raises(AssertionError):
        P.open_tokens("G4", f"{P.P3SCR}/S8_pstop.npy")
    with pytest.raises(AssertionError):
        P.layer_shape_ok("G8", (6, 12000, 640))
    assert P.layer_shape_ok("G8", (6, 12000, 2560)) and P.layer_shape_ok("G4", (6, 12000, 640))
    assert {a: P.ARM_GRID[a] for a in P.ARMS} == {"G8att": "G8", "G4att": "G4", "G8max": "G8", "G4max": "G4"}
    assert R.CTRL_ARMS == ("G8att", "G4att")                # controls exist for both decision arms (the probe-1 `src == "V"` trap, cf. probe 3 C7)


# ------------------------------------------------------------------------------------------------ 2. the models
def test_model_shapes_and_attention_normalised():
    torch = P._torch()
    m = P.make_model("att", (8, 20))
    assert m.head.in_features == 4 * 32 + 8 and m.q.shape == (4, 32) and m.row.shape == (8, 16) and m.col.shape == (20, 16)
    x, e = torch.randn(5, 160, 16), torch.randn(5, 8)
    o, a = m(x, e, return_attn=True)
    assert o.shape == (5,) and a.shape == (5, 4, 160)
    assert torch.allclose(a.sum(-1), torch.ones(5, 4), atol=1e-5)
    m4 = P.make_model("att", (4, 10))
    assert m4(torch.randn(3, 40, 16), torch.randn(3, 8)).shape == (3,)
    mm = P.make_model("max", (8, 20))
    assert mm.tok.out_features == 32 and mm.head.in_features == 40 and mm(x, e).shape == (5,)
    # the max readout is permutation invariant over tokens (no position information): shuffling tokens leaves the logit unchanged
    perm = torch.randperm(160)
    assert torch.allclose(mm(x, e), mm(x[:, perm], e), atol=1e-6)
    # the attention readout is NOT (the position embedding binds token identity to location)
    assert not torch.allclose(m(x, e), m(x[:, perm], e), atol=1e-4)


def test_uniform_mutation_flag_gives_uniform_attention():
    torch = P._torch()
    m = P.make_model("att", (8, 20), uniform=True)
    _, a = m(torch.randn(2, 160, 16), torch.randn(2, 8), return_attn=True)
    assert torch.allclose(a, torch.full_like(a, 1 / 160))


def test_inner_val_split_is_clip_grouped_and_about_20_percent():
    rng = np.random.default_rng(0)
    clips = np.array([f"{int(x):012x}" for x in rng.integers(0, 2 ** 47, 4000)])
    sh = P.sha_ints(clips)
    for seed, fold in ((0, 0), (1, 3), (2, 4)):
        v = P.inner_val_mask(sh, seed, fold)
        assert abs(v.mean() - 0.2) < 0.02
    # grouped: the decision is a function of the clip sha alone (rows of one clip, repeated, always agree)
    rows = P.sha_ints(np.repeat(clips[:200], 7))
    v = P.inner_val_mask(rows, 1, 2).reshape(200, 7)
    assert (v == v[:, :1]).all()
    # independent of the outer fold digit (sha % 5): every outer fold has ~20 % val clips
    of = np.array([int(c, 16) % 5 for c in clips])
    for f in range(5):
        assert abs(P.inner_val_mask(sh, 0, 0)[of == f].mean() - 0.2) < 0.05


# ------------------------------------------------------------------------------------------------ 3. known answer, mutation, null (synthetic tokens through the REAL training code)
def test_known_answer_attention_reads_the_planted_token():
    auc, a, info = fit_planted("att", 3.0)
    mass = a[:, :, 6 * 20 + 10].astype(np.float64).mean(0)           # per-query window-mean attention mass on token (6,10)
    print("known answer: AUC", auc, "mass per query", mass, info)
    assert auc > 0.9
    assert mass.max() >= 0.5, mass
    assert a.shape[1:] == (4, 160) and np.allclose(a.astype(np.float64).sum(-1), 1, atol=5e-3)


def test_mutation_uniform_attention_destroys_the_known_answer():
    auc_live, a_live, _ = fit_planted("att", 3.0)
    auc_mut, a_mut, _ = fit_planted("att", 3.0, uniform=True)
    print("live", auc_live, "uniform", auc_mut)
    assert auc_live > 0.9
    assert auc_mut < 0.75 and auc_live - auc_mut > 0.2             # the attention path is the live carrier of the signal
    assert np.allclose(a_mut[:, :, 130].astype(np.float64), 1 / 160, atol=1e-3)


def test_max_readout_path_is_live_on_a_planted_amplitude():
    auc, _, _ = fit_planted("max", 3.0)
    print("R-MAX planted AUC", auc)
    assert auc > 0.7


def test_null_no_planted_signal_reads_chance():
    auc, a, _ = fit_planted("att", 0.0)
    assert abs(auc - 0.5) < 0.04                                      # SE of an AUC on 3,000 windows ~ 0.011
    assert a[:, :, 130].mean() < 0.2


# ------------------------------------------------------------------------------------------------ 4. controls and decision rule
def test_shuffle_within_folds_keeps_fold_prevalence_and_breaks_labels():
    B = fake_bundle()
    d = B["d"]
    ys = P.shuffle_within_folds(d["y"], d["outer"], 1)
    for f in range(5):
        assert ys[d["outer"] == f].sum() == d["y"][d["outer"] == f].sum()
    assert (ys != d["y"]).any()


def test_c1_c2_verdicts():
    assert P.c1_verdict({"auc": 0.505, "lo": 0.48, "hi": 0.53})
    assert not P.c1_verdict({"auc": 0.54, "lo": 0.49, "hi": 0.59})            # CI covers 0.5 but the point is outside the band
    assert not P.c1_verdict({"auc": 0.52, "lo": 0.51, "hi": 0.54})
    assert P.c2_verdict({"auc": 0.5, "lo": 0.48, "hi": 0.52})[0] is True
    assert P.c2_verdict({"auc": 0.54, "lo": 0.51, "hi": 0.57}, [0.54])[0] is None             # pending
    assert P.c2_verdict({"auc": 0.54, "lo": 0.51, "hi": 0.57}, [0.54, 0.49, 0.51, 0.50, 0.52])[0] is True
    assert P.c2_verdict({"auc": 0.56, "lo": 0.51, "hi": 0.60}, [0.56, 0.55, 0.54, 0.55, 0.56])[0] is False


def cell(delta, lo, hi=None):
    return {"delta": delta, "lo": lo, "hi": hi if hi is not None else delta + 0.02}


def pdec(a, b, ctrl_ok=True, sp8=0.005, sp4=0.005):
    return {"A": a, "B": b, "spread": {"G8att": sp8, "G4att": sp4}, "ctrl": {"G8att": ctrl_ok, "G4att": ctrl_ok, "pair": ctrl_ok}}


def test_decide_helps_requires_size_ci_and_twice_the_seed_spread():
    ok = {"pstop": pdec(cell(0.04, 0.01), cell(0.00, -0.01)), "pmove": pdec(cell(0.0, -0.02), cell(0.0, -0.02))}
    r = P.decide(ok)
    assert r["verdict"].startswith("SPATIAL READOUT HELPS") and "pstop:A" in r["spatial_helps_cells"] and not r["resolution_helps_cells"]
    # delta below 0.03
    assert "HELPS" not in P.decide({"pstop": pdec(cell(0.029, 0.01), cell(0.0, -0.01)), "pmove": pdec(cell(0.0, -0.02), cell(0.0, -0.02))})["verdict"]
    # CI lower bound <= 0
    assert "HELPS" not in P.decide({"pstop": pdec(cell(0.05, 0.0), cell(0.0, -0.01)), "pmove": pdec(cell(0.0, -0.02), cell(0.0, -0.02))})["verdict"]
    # delta not above twice the seed spread
    assert "HELPS" not in P.decide({"pstop": pdec(cell(0.04, 0.01), cell(0.0, -0.01), sp8=0.02), "pmove": pdec(cell(0.0, -0.02), cell(0.0, -0.02))})["verdict"]
    # resolution helps (cell B) in P-MOVE
    r2 = P.decide({"pstop": pdec(cell(0.0, -0.02), cell(0.0, -0.02)), "pmove": pdec(cell(0.0, -0.02), cell(0.035, 0.01))})
    assert r2["verdict"].startswith("RESOLUTION HELPS") and r2["resolution_helps_cells"] == ["pmove:B"]
    # both
    r3 = P.decide({"pstop": pdec(cell(0.04, 0.01), cell(0.04, 0.01)), "pmove": pdec(cell(0.0, -0.02), cell(0.0, -0.02))})
    assert "SPATIAL READOUT HELPS" in r3["verdict"] and "RESOLUTION HELPS" in r3["verdict"]


def test_decide_does_not_help_and_inconclusive_and_incomplete_and_ctrlfail():
    z = lambda: pdec(cell(0.002, -0.01), cell(-0.03, -0.05))
    assert P.decide({"pstop": z(), "pmove": z()})["verdict"] == "DOES NOT HELP"
    mid = {"pstop": pdec(cell(0.015, -0.01), cell(0.0, -0.02)), "pmove": z()}
    assert P.decide(mid)["verdict"] == "INCONCLUSIVE"
    # a dAUC of 0.04 whose CI includes 0 is not HELPS and not NOT-HELP
    assert P.decide({"pstop": pdec(cell(0.04, -0.01), cell(0.0, -0.02)), "pmove": z()})["verdict"] == "INCONCLUSIVE"
    # a missing cell -> INCOMPLETE, never NOT-HELP
    assert P.decide({"pstop": z()})["verdict"] == "INCOMPLETE"
    assert P.decide({"pstop": z(), "pmove": dict(z(), B=None)})["verdict"] == "INCOMPLETE"
    # controls pending -> INCOMPLETE; a failed control -> the cell is CONTROL-FAILED, cannot support HELPS and cannot support NOT-HELP
    pend = {"pstop": z(), "pmove": dict(z(), ctrl={"G8att": None, "G4att": True, "pair": True})}
    assert P.decide(pend)["verdict"] == "INCOMPLETE"
    ctrlfail_helps = {"pstop": pdec(cell(0.05, 0.02), cell(0.0, -0.02), ctrl_ok=False), "pmove": z()}
    r = P.decide(ctrlfail_helps)
    assert "HELPS" not in r["verdict"] and r["cells"]["pstop:A"]["status"] == "CONTROL-FAILED" and r["cells"]["pstop:A"]["helps_if_controls_ok"]
    ctrlfail_nothelp = {"pstop": pdec(cell(0.0, -0.02), cell(0.0, -0.02), ctrl_ok=False), "pmove": z()}
    assert P.decide(ctrlfail_nothelp)["verdict"] == "INCONCLUSIVE"
    # cell A needs only the G8att controls
    pa = {"pstop": dict(pdec(cell(0.04, 0.01), cell(0.0, -0.02)), ctrl={"G8att": True, "G4att": False, "pair": False}), "pmove": z()}
    assert P.decide(pa)["verdict"].startswith("SPATIAL READOUT HELPS")


# ------------------------------------------------------------------------------------------------ 5. the whole pipeline on a synthetic bundle (same functions as the real chain)
def test_pipeline_end_to_end_on_synthetic_bundle(tmp_path):
    raw, scr = tmp_path / "raw", tmp_path / "scr"
    raw.mkdir()
    scr.mkdir()
    raw, scr = str(raw).replace("\\", "/"), str(scr).replace("\\", "/")
    B = fake_bundle("pstop")
    logs = []
    R.stage_arms(B, FAST, raw, scr, ram_min=0, log=logs.append)
    oof = dict(np.load(f"{raw}/oof4_pstop.npz"))
    for arm in P.ARMS:
        assert f"{arm}+E" in oof and all(f"{arm}+E__s{s}" in oof for s in FAST["seeds"])
        assert np.allclose(oof[f"{arm}+E"], np.mean([oof[f"{arm}+E__s{s}"] for s in FAST["seeds"]], 0))
    for arm in ("G8att", "G4att"):
        for s in FAST["seeds"]:
            a = np.load(f"{scr}/attn4_pstop_{arm}_s{s}.npy")
            assert a.shape == (600, 4, 160 if arm == "G8att" else 40)
    # resumable: a second call changes nothing
    before = {k: v.copy() for k, v in oof.items()}
    R.stage_arms(B, FAST, raw, scr, ram_min=0, log=lambda *_: None)
    again = dict(np.load(f"{raw}/oof4_pstop.npz"))
    assert all(np.array_equal(before[k], again[k]) for k in before)
    R.stage_controls(B, FAST, raw, scr, ram_min=0, log=logs.append)
    C = dict(np.load(f"{raw}/ctrl4_pstop.npz"))
    for arm in ("G8att", "G4att"):
        assert f"C1__shuffle__{arm}+E" in C and f"C2__donor__{arm}+E" in C
    R.stage_post(B, FAST, raw, scr, log=logs.append)
    st = json.load(open(f"{raw}/results_strata4_pstop.json"))
    assert st["strata"]["TL_YELLOW"]["auc"].startswith("too small") and isinstance(st["strata"]["TL_ANY"]["auc"], dict)
    mp = np.load(f"{raw}/maps4_pstop.npz")
    assert mp["G8att__ALL__GO"].shape == (8, 20) and mp["G4att__ALL__STAY"].shape == (4, 10)
    assert os.path.exists(f"{raw}/attn_maps_pstop_G8att.png")
    # eval-141 absent -> explicit NOT RUN
    R.stage_eval141(B, FAST, raw, scr, p3raw=raw + "/nope", p3scr=scr + "/nope", oldraw=raw, ram_min=0, log=logs.append)
    assert json.load(open(f"{raw}/results_eval141_4_pstop.json"))["run"] is False
    # eval-141 present (synthetic probe-3 outputs)
    p3r, p3s, old = tmp_path / "p3r", tmp_path / "p3s", tmp_path / "old"
    for p in (p3r, p3s, old):
        p.mkdir()
    rng = np.random.default_rng(3)
    ne = 60
    json.dump({"gate_pass": True}, open(p3r / "eval_g_extract.json", "w"))
    np.save(p3s / "G8_eval141.npy", rng.standard_normal((ne, 163840)).astype(np.float16))
    np.savez(p3r / "pcaG_pstop.npz", mu8=rng.standard_normal((6, 1024)).astype(np.float32), comp8=rng.standard_normal((6, 16, 1024)).astype(np.float32),
             mu4=rng.standard_normal((6, 1024)).astype(np.float32), comp4=rng.standard_normal((6, 16, 1024)).astype(np.float32))
    np.savez(old / "eval141_features.npz", E=rng.standard_normal((ne, 8)), v0=rng.random(ne) * 0.4, eid=np.repeat(np.arange(12), 5), go=rng.random(ne) < 0.4)
    R.stage_eval141(B, FAST, raw, scr, p3raw=str(p3r).replace("\\", "/"), p3scr=str(p3s).replace("\\", "/"), oldraw=str(old).replace("\\", "/"), ram_min=0, log=logs.append)
    ev = json.load(open(f"{raw}/results_eval141_4_pstop.json"))
    assert ev["run"] and ev["n"] == ne and "G8att+E" in ev["arms"] and "G8att+E_minus_G4att+E" in ev["paired"]
    # tables + verdict (only one population present -> INCOMPLETE, never a verdict)
    dec = R.stage_tables([B], FAST, raw, log=lambda *_: None, p3raw=raw + "/nope")
    assert dec["verdict"] == "INCOMPLETE"
    V = json.load(open(f"{raw}/VERDICT.json"))
    assert V["verdict"] == "INCOMPLETE" and "pstop" in V["detail"] and V["cfg"]["max_epochs"] == 2
    assert os.path.exists(f"{raw}/tables4.txt")


def test_frozen_cfg_values():
    c = P.CFG
    assert (c["lr"], c["wd"], c["batch"], c["max_epochs"], c["patience"], c["n_query"], c["d_key"], c["d_tok"], c["d_hidden_max"]) == (1e-3, 1e-2, 256, 40, 5, 4, 32, 16, 32)
    assert tuple(c["seeds"]) == (0, 1, 2) and c["n_boot"] == 2000 and c["boot_seed"] == 0
    assert P.P3_MARKERS == ("CHAIN_c2band.done", "CHAIN_mlpctl.done", "CHAIN_evalmlp.done", "CHAIN_mutate.done", "CHAIN_pf.done")


def test_p3_consistency_check(tmp_path):
    B = fake_bundle(N=300, n_clips=30)
    d, bk = B["d"], B["bk"]
    good = {"arms": {a: {"fold": {"auc": P.fold_mean_auc(bk[f"P3:{a}"], d["y"], d["outer"])}} for a in ("G4+E", "G8+E")}}
    p = tmp_path / "resg_pstop_linear.json"
    p.write_text(json.dumps(good))
    assert R.p3_consistency("pstop", bk, d, str(tmp_path).replace("\\", "/"))["ok"] is True
    bad = json.loads(json.dumps(good))
    bad["arms"]["G8+E"]["fold"]["auc"] += 1e-6
    p.write_text(json.dumps(bad))
    assert R.p3_consistency("pstop", bk, d, str(tmp_path).replace("\\", "/"))["ok"] is False
    assert R.p3_consistency("pstop", bk, d, str(tmp_path / "nope"))["checked"] is False
