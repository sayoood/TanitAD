"""test_q5_read.py -- AMENDMENT 7 read side: C5, C3 (a)-(e), the beside arms, the fit plan, and an end-to-end dry-run on SYNTHETIC dirs (no real data exists yet).
Every control has a deliberate-regression arm that must go RED.
Run: OMP_NUM_THREADS=2 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_q5_read.py --basetemp <dir>
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q5_controls as C  # noqa: E402
import q5_fitplan as P  # noqa: E402
import q5_now_encode as Q  # noqa: E402
import q5_read as R  # noqa: E402

TREE = os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree")
N_TR, N_CLIP, N_EV, N_EP = 240, 24, 200, 20


def _ci():
    sys.path.insert(0, TREE)
    sys.path.insert(0, TREE + "/taniteval")
    from taniteval import ci
    return ci


# ------------------------------------------------------------------------------------------------------------ synthetic dirs
def _fp8(x):
    import torch
    return Q.to_fp8(torch.from_numpy(x.astype(np.float32)))


def _vision(rng, y, plant):
    """v40 [n,40,1024], v160 [n,160,1024] fp8 bytes, mean fp16.  `plant` adds a fixed direction to one cell on positives only."""
    n = len(y)
    d = np.random.default_rng(5).normal(size=1024)
    d /= np.linalg.norm(d)
    v40 = rng.normal(size=(n, 40, 1024)).astype(np.float32)
    v160 = rng.normal(size=(n, 160, 1024)).astype(np.float32)
    if plant:
        v40[y == 1, 7, :] += 8.0 * d * 32
        v160[y == 1, 30, :] += 8.0 * d * 32
    return _fp8(v40), _fp8(v160), v40.mean(1).astype(np.float16)


def _E(rng, y):
    E = rng.normal(size=(len(y), 18)).astype(np.float32)
    E[:, 0] += 1.5 * y
    E[:, 17] = (rng.random(len(y)) < 0.9)
    return E


def make_synth(base, plant416=True, seed=0):
    """c0/{S1,S2}/ + c0/s0p_eval.npz (the 256 route's vision, no signal), pod_out/{S}_{416,256p}/ + pod_out/eval_{...}/ -- shaped exactly as `q5_frames_pod.py gather` and C0 write them."""
    import q5_now_encode  # noqa: F401
    sys.path.insert(0, C.S0P)
    import s0p_lib as L
    rng = np.random.default_rng(seed)
    j = lambda *p: os.path.join(base, *p)                                              # noqa: E731
    clips_ev = np.asarray([f"e{i % N_EP:02d}" for i in range(N_EV)]).astype("U40")
    y_ev = (np.arange(N_EV) % 5 == 0).astype(np.uint8)
    E_ev = _E(rng, y_ev)
    meta_ev = {"c2_second_route_max_abs": {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}}
    os.makedirs(j("c0"), exist_ok=True)
    v = _vision(rng, y_ev, False)
    L.save_npz(j("c0", "s0p_eval.npz"), E_ev, *v, y_ev, clips_ev, np.arange(N_EV) % 90, np.arange(N_EV) % N_EP + 1000, meta_ev, ci=np.arange(N_EV))
    np.save(j("c0", "sidecar_limit_eval.npy"), (E_ev[:, 13:17] + (np.arange(N_EV)[:, None] % 3 == 0)).astype(np.float32))      # differs from E's causal limit on ~1/3 of windows
    for tag in ("416", "256p"):                                                      # the pod's eval rows: 416 carries the plant, 256p none
        d = j("pod_out", f"eval_{tag}")
        os.makedirs(d, exist_ok=True)
        vv = _vision(np.random.default_rng(seed + 1), y_ev, plant416 and tag == "416")
        for k, a in zip(("v40", "v160", "mean"), vv):
            np.save(f"{d}/{k}.npy", a)
        for k, a in (("y", y_ev), ("clip", clips_ev), ("ws", np.arange(N_EV) % 90), ("eid", np.arange(N_EV) % N_EP + 1000)):
            np.save(f"{d}/{k}.npy", a)
    for si, s in enumerate(("S1", "S2")):
        y = (rng.random(N_TR) < 0.25).astype(np.uint8)
        clip = np.asarray([f"t{si}{i % N_CLIP:02d}" for i in range(N_TR)]).astype("U40")
        ws = rng.integers(0, 90, N_TR).astype(np.int64)
        eid = (np.arange(N_TR) % N_CLIP + 5000 + si * 100).astype(np.int64)
        c0 = j("c0", s)
        os.makedirs(c0, exist_ok=True)
        v = _vision(rng, y, False)
        for k, a in (("E", _E(rng, y)), ("y", y), ("clip", clip), ("ws", ws), ("eid", eid), ("v40", v[0]), ("v160", v[1]), ("mean", v[2])):
            np.save(f"{c0}/{k}.npy", a)
        json.dump({"sample_sha256": f"synthetic{s}", "n_sample": N_TR, "c2_second_route_max_abs": {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}, "causal_limit_ok_false_fraction": 0.1},
                  open(f"{c0}/meta.json", "w", encoding="utf-8"))
        for tag in ("416", "256p"):
            d = j("pod_out", f"{s}_{tag}")
            os.makedirs(d, exist_ok=True)
            vv = _vision(np.random.default_rng(seed + 10 + si), y, plant416 and tag == "416")
            for k, a in zip(("v40", "v160", "mean"), vv):
                np.save(f"{d}/{k}.npy", a)
            for k, a in (("y", y), ("clip", clip), ("ws", ws), ("eid", eid)):
                np.save(f"{d}/{k}.npy", a)


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    """The end-to-end dry-run: synthetic dirs -> assemble -> plan -> run every fit line -> read.  Built once."""
    base = str(tmp_path_factory.mktemp("q5dry")).replace("\\", "/")
    make_synth(base)
    P.assemble(base, ["S1", "S2"], log=lambda *_: None)
    P.main(["plan", "--base", base, "--python", sys.executable])
    P.run_plan(P.commands(base, ["S1", "S2"], sys.executable), log=lambda *_: None)
    return base


def _load(dry_base, s="S1", tag="416"):
    F, L = C.s0p_modules()
    pp = P.paths(dry_base, ["S1", "S2"])
    return F, L, F.load_any(pp["samples"][s][f"train{tag}"]), F.load_any(pp["eval"][f"eval{tag}"]), pp["samples"][s][f"work{tag}"]


# ------------------------------------------------------------------------------------------------------------ the fit plan
def test_fitplan_names_every_call(tmp_path):
    cmds = P.commands(str(tmp_path), ["S1", "S2"], "py")
    assert len(cmds) == 2 * ((1 + 10 + 2) + (1 + 6))
    labels = [c[0] for c in cmds]
    for s in ("S1", "S2"):
        for arm in P.ARMS_416:
            for m in P.MODELS:
                assert f"{s} 416 fit {arm} {m}" in labels
        for arm in P.ARMS_256P:
            for m in P.MODELS:
                assert f"{s} 256p fit {arm} {m}" in labels
        assert f"{s} 416 fit V40+E-limit linear" in labels and f"{s} 256p prep" in labels
        assert f"{s} 256p fit E linear" not in labels                       # E is fitted once per sample
    for _, argv in cmds:
        assert argv[1].endswith(("s0p_fit.py", "q5_fit_extra.py"))


def test_dry_run_end_to_end(dry):
    plan = open(os.path.join(dry, "fitplan.sh"), encoding="utf-8").read()
    assert plan.count("s0p_fit.py") >= 36 and "q5_read.py" in plan and "OMP_NUM_THREADS=2" in plan
    cfg = json.load(open(os.path.join(dry, "read_cfg.json"), encoding="utf-8"))
    assert all(v is None for v in cfg["controls"].values())                  # fail closed until filled
    for s in ("S1", "S2"):
        for t in ("416", "256p"):
            w = P.paths(dry, ["S1", "S2"])["samples"][s][f"work{t}"]
            for arm in (P.ARMS_416 if t == "416" else P.ARMS_256P):
                for m in P.MODELS:
                    assert os.path.exists(f"{w}/fit_{arm}_{m}.npy")
        assert os.path.exists(P.paths(dry, ["S1", "S2"])["samples"][s]["work416"] + "/fit_V40+E-limit_linear.npy")
    # with the externals left null the read REFUSES (fail closed)
    cfg["tree"] = TREE
    p = os.path.join(dry, "cfg_refuse.json")
    cfg["out"] = os.path.join(dry, "refuse.json")
    json.dump(cfg, open(p, "w", encoding="utf-8"))
    res = R.read(p)
    assert res["outcome"].startswith("REFUSED") and "C0" in res["outcome"]


GOOD_EXT = {"c0_differing_bytes": 0, "c2_auroc": 0.995, "c2_outcome": "RES416-ADDS-40", "c4_E_bitwise_equal_C0": True, "c6a_max_abs_diff": 1, "c6b_mae_255": 0.6, "c6c_bitwise": True,
            "c7_ok": True, "c8_bitwise": True, "c9_counts": {"eval_n": 2365, "eval_pos": 131, "eval_episodes": 139, "train_n": 16000, "train_pos": 4000}}


def _cfg(dry, name, extra=None, controls=None):
    cfg = json.load(open(os.path.join(dry, "read_cfg.json"), encoding="utf-8"))
    cfg["tree"] = TREE
    cfg["out"] = os.path.join(dry, name + ".json")
    cfg["controls"] = {**GOOD_EXT, **(controls or {})}
    cfg.update(extra or {})
    p = os.path.join(dry, name + "_cfg.json")
    json.dump(cfg, open(p, "w", encoding="utf-8"))
    return p, cfg


def test_read_end_to_end_on_synthetic(dry):
    p, cfg = _cfg(dry, "read_ok")
    res = R.read(p)
    bad = [c for c in res["controls"] if not c["ok"]]
    assert not bad, bad
    assert res["outcome"] in ("RES416-ADDS-40", "RES416-ADDS-160", "SAMPLES-DISAGREE", "RES416-FLAT")
    # C1 computed from the dirs: identity gives exactly 0 and not-ADDS
    c1 = [c for c in res["controls"] if c["id"] == "C1"][0]
    assert c1["detail"][0] == 0.0 and c1["detail"][1] == [0.0, 0.0] and c1["detail"][2] == "RES416-FLAT"
    b = res["beside"]["S1"]
    assert "M" in b["arms"] and "E-limit" in b["arms"]                          # computed from fits that exist
    blocked = b["BLOCKED"]
    assert "P-B vs P-A" in blocked and "work416pb" in blocked["P-B vs P-A"] and "V416tok+E (13x32) vs V40@416+E" in blocked
    assert "route V*@256b" in blocked and "fit_V40+E_linear.npy" in blocked["route V*@256b"]
    assert "HEAD-UNDERUSES" in blocked                                            # no STOPDEC files in the synthetic world


def test_beside_arms_cannot_change_the_outcome(dry):
    """Same cells and same code with the beside inputs present and absent; and decide() has no beside argument."""
    import inspect
    assert list(inspect.signature(R.decide).parameters)[:1] == ["inp"] and "beside" not in inspect.signature(R.decide).parameters
    base = R.read(_cfg(dry, "bs_without")[0])
    pp = P.paths(dry, ["S1", "S2"])
    extra = {}
    for s in ("S1", "S2"):
        # a wildly different "banked 256b" score set and a P-B dir copied from the 256p fits: if beside rows leaked into the code, the outcome would move
        for tag, src in (("256b", "work256p"), ("416pb", "work256p")):
            dst = os.path.join(dry, f"beside_{s}_{tag}")
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(pp["samples"][s][src]):
                if f.startswith("fit_") and f.endswith(".npy"):
                    a = np.load(os.path.join(pp["samples"][s][src], f))
                    np.save(os.path.join(dst, f), a[::-1].copy())
            extra.setdefault(s, {})[f"work{tag}"] = dst
    p, cfg = _cfg(dry, "bs_with")
    for s in extra:
        cfg["samples"][s].update(extra[s])
    json.dump(cfg, open(p, "w", encoding="utf-8"))
    with_b = R.read(p)
    assert with_b["outcome"] == base["outcome"] and with_b["detail"] == base["detail"]
    assert with_b["inputs"] == base["inputs"]                                   # the SAME resample: outcome inputs identical
    assert "route" in with_b["beside"]["S1"]["arms"] and "P-B" in with_b["beside"]["S1"]["arms"]
    assert "P-B vs P-A" not in with_b["beside"]["S1"]["BLOCKED"]


def test_head_underuses_rule_is_printed_beside(dry):
    F, L, tr, ev, w = _load(dry)
    yev = np.asarray(ev["y"], np.float64)
    rng = np.random.default_rng(3)
    T = {t: yev * 0.05 + rng.normal(size=len(yev)) for t in R.T_TAGS}           # a weak T
    d = {"work416": w, "work256p": _load(dry, "S1", "256p")[4]}
    xs, xd, blocked, groups = R.gather_beside(d, ev, t_scores_fn=lambda _e: T)
    assert "HEAD-UNDERUSES" not in blocked and len(groups["HEAD-UNDERUSES"]) == 2 * len(R.FAMILIES)
    sc = {**{f"V40@416+E|{f}": np.load(f"{w}/fit_V40+E_{f}.npy") for f in R.FAMILIES}, **xs}
    for f in R.FAMILIES:
        sc[f"E|{f}"] = np.load(f"{w}/fit_E_{f}.npy")
        for g in R.GRIDS:
            sc[f"V{g}@416+E|{f}"] = np.load(f"{w}/fit_V{g}+E_{f}.npy")
            sc[f"V{g}@256p+E|{f}"] = sc[f"V{g}@416+E|{f}"]
    B, _ = R.bootstrap_sample(sc, ev["y"], ev["eid"], _ci(), n_boot=100, extra_diffs=xd)
    rep = R.beside_report(B, groups, blocked)
    assert set(rep["HEAD-UNDERUSES_rule"]) == set(R.FAMILIES)


# ------------------------------------------------------------------------------------------------------------ C5
def test_c5_real_fit_above_null_and_unshuffled_goes_red(dry):
    F, L, tr, ev, w = _load(dry)
    good = C.c5_shuffle_null(F, L, w, tr, ev, ci_mod=_ci(), n_boot=100)
    assert len(good["null_aurocs"]) == 20 and good["real_auroc"] > 0.7
    assert good["pass"] and good["real_auroc"] > good["null_max"]
    assert "ci_covers_0.5" in good["single_draw"]
    bad = C.c5_shuffle_null(F, L, w, tr, ev, shuffle=False)                         # regression: labels left as they are -> null == real fit
    assert not bad["pass"]
    assert not R.evaluate_controls({**_all_good(), "c5_real_auroc": bad["real_auroc"], "c5_null_max": bad["null_max"]})[1]


def _all_good():
    return {**GOOD_EXT, "c1_dauc": 0.0, "c1_ci": [0.0, 0.0], "c1_outcome": "RES416-FLAT", "c3a_leak_auroc": 1.0, "c3b_limit_max_abs_diff": 0.0, "c3c_E_windows_changed": 5, "c3d_shared_clips": 0,
            "c3e_E_y_clip_ws_eid_bitwise": True, "c5_real_auroc": 0.86, "c5_null_max": 0.6}


def test_all_good_passes_and_each_miss_refuses():
    ok, allok = R.evaluate_controls(_all_good())
    assert allok
    for k in ("c3a_leak_auroc", "c3b_limit_max_abs_diff", "c3c_E_windows_changed", "c3d_shared_clips", "c3e_E_y_clip_ws_eid_bitwise", "c5_real_auroc", "c5_null_max"):
        bad = dict(_all_good())
        bad[k] = None
        assert not R.evaluate_controls(bad)[1], k
    assert not R.evaluate_controls({**_all_good(), "c6a_max_abs_diff": 2})[1]       # AMENDMENT 7a: > 1 count refuses
    assert R.evaluate_controls({**_all_good(), "c6a_max_abs_diff": 1})[1]


# ------------------------------------------------------------------------------------------------------------ C3
def test_c3a_leak_reads_one_and_regression_without_label_goes_red(dry):
    F, L, tr, ev, w = _load(dry)
    assert C.c3a_leak_auroc(F, L, w, tr, ev) >= 0.999
    red = C.c3a_leak_auroc(F, L, w, tr, ev, append_label=False)                    # the label column zeroed: the control must not pass
    assert red < 0.999
    assert not R.evaluate_controls({**_all_good(), "c3a_leak_auroc": red})[1]


def test_c3b_second_route_zero_and_nonzero_goes_red():
    z = {"c2_second_route_max_abs": {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}}
    assert C.c3b_limit_max_abs_diff(z, z) == 0.0
    nz = {"c2_second_route_max_abs": {"limit": 1e-7, "limit_ok": 0.0}}
    v = C.c3b_limit_max_abs_diff(z, nz)
    assert v == 1e-7 and not R.evaluate_controls({**_all_good(), "c3b_limit_max_abs_diff": v})[1]
    assert C.c3b_limit_max_abs_diff(z, {}) is None                                  # an absent second route is a MISS, not a pass
    E = np.random.default_rng(0).normal(size=(5, 18))
    E2 = E.copy()
    assert C.second_route_diff(E, E2) == 0.0
    E2[3, 15] += 1e-3
    assert C.second_route_diff(E, E2) > 0


def test_c3c_mutation_changes_E_and_skipped_substitution_is_blind():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(30, 18)).astype(np.float32)
    side = E[:, 13:17].copy()
    side[[2, 9]] += 1.0
    assert C.c3c_windows_changed(E, side) == 2
    assert C.c3c_windows_changed(E, E[:, 13:17]) == 0                              # an identical sidecar changes nothing
    blind = C.c3c_windows_changed(E, side, apply=False)                            # regression: the substitution skipped
    assert blind == 0 and not R.evaluate_controls({**_all_good(), "c3c_E_windows_changed": blind})[1]
    assert not np.shares_memory(E, E.copy())                                        # the input E is never mutated in place
    E0 = E.copy()
    C.c3c_windows_changed(E, side)
    assert np.array_equal(E, E0)


def test_c3d_shared_clips():
    assert C.c3d_shared_clips(["a", "b"], ["c", "d"]) == 0
    n = C.c3d_shared_clips(["a", "b", "c"], ["c", "d"])
    assert n == 1 and not R.evaluate_controls({**_all_good(), "c3d_shared_clips": n})[1]


def test_c3e_bitwise_and_one_flipped_y_byte_goes_red(dry):
    F, L, tr416, ev416, _ = _load(dry, "S1", "416")
    _, _, tr256, ev256, _ = _load(dry, "S1", "256p")
    assert C.c3e_bitwise(tr416, tr256) and C.c3e_bitwise(ev416, ev256)
    bad = {k: (np.array(v).copy() if k in C.BIT_KEYS else v) for k, v in tr256.items()}
    bad["y"][3] ^= 1                                                                # one flipped y byte
    assert not C.c3e_bitwise(tr416, bad)
    assert not R.evaluate_controls({**_all_good(), "c3e_E_y_clip_ws_eid_bitwise": False})[1]
    e2 = {k: (np.array(v).copy() if k in C.BIT_KEYS else v) for k, v in tr256.items()}
    e2["E"][0, 0] = np.nextafter(e2["E"][0, 0], np.float32(1e9))                    # one ulp in E
    assert not C.c3e_bitwise(tr416, e2)
    t2 = {k: v for k, v in tr256.items()}
    t2["ws"] = np.asarray(tr256["ws"]).astype(np.int32)                             # same values, other dtype: not bitwise
    assert not C.c3e_bitwise(tr416, t2)


def test_compute_fit_side_end_to_end_and_blocked_sidecar(dry):
    cfg = json.load(open(os.path.join(dry, "read_cfg.json"), encoding="utf-8"))
    ctl, det = C.compute_fit_side(cfg, ci_mod=_ci(), log=lambda *_: None, c5_seeds=range(100, 105))
    assert ctl["c3a_leak_auroc"] >= 0.999 and ctl["c3b_limit_max_abs_diff"] == 0.0 and ctl["c3d_shared_clips"] == 0 and ctl["c3e_E_y_clip_ws_eid_bitwise"] is True
    assert ctl["c5_real_auroc"] > ctl["c5_null_max"]
    assert ctl["c3c_E_windows_changed"] >= 1                                       # the synthetic sidecar differs from E on ~1/3 of eval windows
    cfg2 = dict(cfg, sidecar_limit_eval=os.path.join(dry, "no_such_sidecar.npy"))
    ctl2, det2 = C.compute_fit_side(cfg2, ci_mod=None, log=lambda *_: None, c5_seeds=range(100, 102))
    assert ctl2["c3c_E_windows_changed"] is None and "BLOCKED" in det2["c3c"] and "no_such_sidecar.npy" in det2["c3c"]
    assert not R.evaluate_controls({**_all_good(), **{k: v for k, v in ctl2.items() if v is not None}, "c3c_E_windows_changed": None})[1]


def test_fit_extra_arm_does_not_touch_s0p_fit_file():
    p = os.path.join(C.S0P, "s0p_fit.py")
    import hashlib
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    import q5_fit_extra as X
    F, _ = C.s0p_modules()
    X.extend(F)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h


def test_new_files_have_encoding_on_every_open():
    import ast
    for fn in ("q5_controls.py", "q5_fitplan.py", "q5_fit_extra.py", "q5_read.py"):
        tree = ast.parse(open(os.path.join(HERE, fn), encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open":
                mode = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else ""
                assert "b" in str(mode) or any(k.arg == "encoding" for k in n.keywords), (fn, n.lineno)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run":
                assert not any(k.arg == "text" for k in n.keywords) or any(k.arg == "encoding" for k in n.keywords), (fn, n.lineno)
