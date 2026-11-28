"""test_closeread.py -- tests of closeread_fit.py and closeread_dump.py (card PREREG_V8F_PROOF_ADD77). Synthetic data only, CPU only, small.
The tool files can be swapped by environment (the mutation runner does): CLOSEREAD_FIT_FILE, CLOSEREAD_DUMP_FILE. The reader module tok_reader.py is read from CLOSEREAD_BUILD_DIR (default ../integ10k).
Run: PYTHONDONTWRITEBYTECODE=1 <venv python> -m pytest -x -q -p no:cacheprovider --basetemp <NEW folder> test_closeread.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("CLOSEREAD_BUILD_DIR") or os.path.join(HERE, "..", "integ10k")
sys.path.insert(0, BUILD)


def _load(name, env, default):
    path = os.environ.get(env) or os.path.join(HERE, default)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


CF = _load("closeread_fit_t", "CLOSEREAD_FIT_FILE", "closeread_fit.py")
CD = _load("closeread_dump_t", "CLOSEREAD_DUMP_FILE", "closeread_dump.py")
import tok_reader as TR  # noqa: E402


def ids(n, held, prefix="tc"):
    out, i = [], 0
    while len(out) < n:
        c = "%s-%06d" % (prefix, i)
        i += 1
        if bool(CF.heldout_rule(c)) == held and bool(TR.is_heldout(c)) == held:
            out.append(c)
    return out


def small_dump(clips=("a", "b", "c"), frames=30):
    n = len(clips) * frames
    fr = np.tile(np.arange(frames), len(clips))
    return {"clips": np.asarray(clips), "cid": np.repeat(np.arange(len(clips)), frames), "clip_index": np.repeat(np.arange(len(clips)), frames) + 5, "frame": fr,
            "p": np.full(n, 0.9), "gap": 20.0 + np.arange(n) * 0.1, "clos": np.arange(n) * 0.01, "valid": fr >= 4, "lab_present": np.ones(n, bool), "lab_gap": 20.0 + np.arange(n) * 0.1,
            "lab_closing": 0.5 + np.arange(n) * 0.01, "lab_run": np.full(n, 5), "dt_s": 0.2}


# ------------------------------------------------------------------------------------------------ held-out rule and lists
def test_heldout_rule_equals_the_readers_rule():
    cl = ["clip-%d" % i for i in range(500)]
    want = [TR.is_heldout(c) for c in cl]
    assert 20 < sum(want) < 120
    assert [CF.heldout_rule(c) for c in cl] == want
    assert [CD.heldout_rule(c) for c in cl] == want


def _lists_and_meta(n_man=80, ex=(10, 14)):
    man = ids(30, True, "m") + ids(n_man - 30, False, "m")
    exs = set(sorted(man, key=CF.sha12)[ex[0]:ex[1]])
    held = sorted(c for c in man if c not in exs and TR.is_heldout(c))
    meta = {"heldout": {"rule": "int(sha256(clip id)[:8], 16) % 10 == 0", "n_heldout": len(held), "n_trained": len(man) - len(exs) - len(held), "heldout_sha256": CF.list_sha256(held)},
            "trainer_argv": "--cache-dir x --exclude %d:%d --batch 64" % ex}
    return man, meta, held, exs


def test_reader_lists_recompute_and_refusals():
    man, meta, held, exs = _lists_and_meta()
    L = CD.reader_lists(man, meta)
    assert L["heldout_ids"] == held and set(L["excluded_ids"]) == exs and len(L["train_ids"]) == meta["heldout"]["n_trained"]
    assert not set(L["train_ids"]) & set(held) and not set(L["train_ids"]) & exs
    assert CD.parse_exclude({"trainer_argv": ["--x", "1", "--exclude", "5:9"]}) == (5, 9) and CD.parse_exclude({"trainer_argv": "--exclude 5:9"}) == (5, 9)
    bad = json.loads(json.dumps(meta))
    bad["heldout"]["heldout_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        CD.reader_lists(man, bad)
    bad = json.loads(json.dumps(meta))
    bad["heldout"]["n_heldout"] += 1
    with pytest.raises(SystemExit):
        CD.reader_lists(man, bad)
    noex = json.loads(json.dumps(meta))
    noex["trainer_argv"] = "--cache-dir x"
    with pytest.raises(SystemExit):
        CD.reader_lists(man, noex)          # the exclusion changes the counts: ignoring it is refused


def test_c1_clean_and_planted():
    fits, scores, _ = CF.make_synth(seed=2, n_fit=20, n_score=20)
    lists = CF.synth_lists(fits)
    assert CF.c1_no_leak(fits, scores, lists)["pass"]
    f2 = {k: dict(v) for k, v in fits.items()}
    f2[0]["clips"] = f2[0]["clips"].copy()
    f2[0]["clips"][0] = scores[0]["clips"][0]
    assert not CF.c1_no_leak(f2, scores, lists)["pass"]
    l2 = {k: dict(v) for k, v in lists.items()}
    l2[0]["train_ids"] = list(lists[0]["train_ids"]) + [fits[0]["clips"][0]]
    assert not CF.c1_no_leak(fits, scores, l2)["pass"]
    l3 = {k: dict(v) for k, v in lists.items()}
    l3[0]["meta_n_heldout"] = lists[0]["meta_n_heldout"] + 1
    assert not CF.c1_no_leak(fits, scores, l3)["pass"]


# ------------------------------------------------------------------------------------------------ rows and targets
def test_build_rows_window_and_target():
    D = small_dump()
    r = CF.build_rows(D, 5)
    assert len(r["y"]) == 3 * (30 - 8)
    lab = D["lab_closing"]
    at = {(int(c), int(f)): i for i, (c, f) in enumerate(zip(D["cid"], D["frame"]))}
    for k in range(len(r["y"])):
        i = at[(["a", "b", "c"].index(r["clip"][k]), int(r["frame"][k]))]
        assert r["y"][k] == lab[i]                                           # the label of the LAST frame, with its sign
        assert np.array_equal(r["X"][k, :, 0], D["gap"][i - 4:i + 1])        # five frames of the SAME clip, oldest first
        assert np.array_equal(r["X"][k, :, 2], D["clos"][i - 4:i + 1])
    assert int(r["frame"].min()) >= 8                                         # no window crosses into the clip before it


def test_build_rows_drops_rows_without_label_closing():
    D = small_dump()
    D["lab_present"][20] = False
    D["lab_run"][21] = 2
    D["lab_closing"][22] = np.nan
    r = CF.build_rows(D, 10)
    assert not {20, 21, 22} & set(int(f) for f, c in zip(r["frame"], r["clip"]) if c == "a")


def test_episode_map():
    D = small_dump()
    r = CF.build_rows(D, 5, {"a": "E1", "b": "E1", "c": "E2"})
    assert set(r["ep"].tolist()) == {"E1", "E2"} and all(e == ("E2" if c == "c" else "E1") for e, c in zip(r["ep"], r["clip"]))
    assert set(CF.build_rows(D, 5)["ep"].tolist()) == {"a", "b", "c"}


# ------------------------------------------------------------------------------------------------ fitting
def _rows(clips, n_per, rng, relation=True):
    X = rng.normal(0, 1, (len(clips) * n_per, 5, 3))
    y = 0.5 * X[:, -1, 0] - 0.3 * X[:, 0, 1] + 0.01 * rng.normal(0, 1, len(X)) if relation else rng.normal(0, 1, len(X))
    return {"X": X, "y": y, "clip": np.repeat(np.asarray(clips), n_per), "frame": np.tile(np.arange(n_per), len(clips)), "ep": np.repeat(np.asarray(clips), n_per), "H": 5, "dt_s": 0.2}


def test_ridge_weight_is_chosen_on_the_fit_set_by_clip():
    rng = np.random.default_rng(0)
    fit = _rows(ids(24, True), 40, rng)
    score = _rows(ids(10, False, "s"), 40, rng, relation=False)
    est, info = CF.run_cell("L", fit, score)
    va = CF.split_by_clip(fit["clip"], 0.25, 0)
    Xf = fit["X"].reshape(len(fit["y"]), -1)
    lam = CF.select_ridge(Xf[~va], fit["y"][~va], Xf[va], fit["y"][va])
    assert info["lambda"] == lam and lam <= 1e-3
    assert not set(fit["clip"][va]) & set(fit["clip"][~va])


def test_split_by_clip_has_no_clip_on_both_sides():
    clips = np.repeat(np.asarray(ids(20, True)), 30)
    va = CF.split_by_clip(clips, 0.25, 3)
    assert 0 < va.sum() < len(va) and not set(clips[va]) & set(clips[~va])


def test_score_rows_never_enter_the_fit():
    rng = np.random.default_rng(1)
    fit = _rows(ids(24, True), 40, rng)
    score = _rows(ids(10, False, "s"), 40, rng)
    e1, _ = CF.run_cell("L", fit, score)
    s2 = dict(score)
    s2["y"] = rng.permutation(score["y"]) * 7.0
    e2, _ = CF.run_cell("L", fit, s2)
    assert np.array_equal(e1, e2)
    assert np.corrcoef(e1, score["y"])[0, 1] > 0.9                            # the planted relation is read on the score set


def test_permuted_labels_change_the_fit():
    rng = np.random.default_rng(2)
    fit = _rows(ids(24, True), 40, rng)
    score = _rows(ids(10, False, "s"), 40, rng)
    e1, _ = CF.run_cell("L", fit, score)
    e2, _ = CF.run_cell("L", fit, score, perm_seed=5)
    assert not np.allclose(e1, e2) and np.corrcoef(e1, score["y"])[0, 1] - abs(np.corrcoef(e2, score["y"])[0, 1]) > 0.3


def test_leak_guard_refuses():
    rng = np.random.default_rng(3)
    fit = _rows(ids(12, True), 20, rng)
    score = _rows(ids(6, False, "s"), 20, rng)
    CF.run_cell("L", fit, score)
    leak = dict(score)
    leak["clip"] = score["clip"].astype(object)
    leak["clip"][0] = fit["clip"][0]
    with pytest.raises(SystemExit):
        CF.run_cell("L", fit, leak)
    notheld = dict(fit)
    notheld["clip"] = np.repeat(np.asarray(ids(12, False, "n")), 20)
    with pytest.raises(SystemExit):
        CF.run_cell("L", notheld, score)
    with pytest.raises(SystemExit):
        CF.run_cell("L", fit, score, train_ids=[fit["clip"][0]])


def test_arm_n_is_deterministic_and_reads_a_planted_relation():
    rng = np.random.default_rng(4)
    fit = _rows(ids(24, True), 40, rng)
    score = _rows(ids(8, False, "s"), 40, rng)
    e1, i1 = CF.run_cell("N", fit, score, n_epochs=15)
    e2, _ = CF.run_cell("N", fit, score, n_epochs=15)
    assert np.array_equal(e1, e2) and len(i1["nets"]) == 3
    assert np.corrcoef(e1, score["y"])[0, 1] > 0.8


# ------------------------------------------------------------------------------------------------ statistics and names
def test_cell_summary_matches_direct_numbers():
    rng = np.random.default_rng(5)
    rows, ests = [], []
    for s in range(2):
        y = rng.normal(0, 1, 300)
        est = 0.6 * y + rng.normal(0, 0.5, 300)
        rows.append({"y": y, "ep": np.asarray(["e%d" % (i % 15) for i in range(300)])})
        ests.append(est)
    S = CF.cell_summary(ests, rows, 200, 1)
    ally, alle = np.concatenate([r["y"] for r in rows]), np.concatenate(ests)
    assert abs(S["r_pooled"] - np.corrcoef(ally, alle)[0, 1]) < 1e-9
    assert abs(S["mae_est"] - np.abs(alle - ally).mean()) < 1e-9 and abs(S["mae_zero"] - np.abs(ally).mean()) < 1e-9
    assert abs(S["diff_abs_err"] - (np.abs(alle - ally).mean() - np.abs(ally).mean())) < 1e-9
    assert S["r_ci"][0] < S["r_pooled"] < S["r_ci"][1] and S["n_episodes"] == 15 and S["n_frames"] == 600
    assert all(abs(S["r_per_seed"][i] - np.corrcoef(rows[i]["y"], ests[i])[0, 1]) < 1e-9 for i in range(2))


def test_bootstrap_resamples_reader_seeds_too():
    st = [{"n": np.array([100.0]), "sae": np.array([1.0]), "say": np.array([2.0])}, {"n": np.array([10.0]), "sae": np.array([1.0]), "say": np.array([2.0])}]
    _, boot = CF.boot_totals(st, 400, 1)
    assert set(np.unique(boot["n"]).tolist()) == {20.0, 110.0, 200.0}


def test_name_rule():
    base = {"r_pooled": 0.5, "r_ci": [0.1, 0.8], "r_per_seed": [0.5, 0.45], "diff_ci": [-0.3, -0.1]}
    assert CF.name_of("L", 25, base) == ("CLOSEREAD-L-H25-READS-CLOSING", True)
    assert CF.name_of("N", 5, dict(base, r_per_seed=[0.5, 0.2]))[0] == "CLOSEREAD-N-H5-NO-CLOSING"
    assert CF.name_of("L", 5, dict(base, diff_ci=[-0.3, 0.1]))[0].endswith("NO-CLOSING")
    assert CF.name_of("L", 5, dict(base, r_ci=[-0.1, 0.8]))[0].endswith("NO-CLOSING")
    assert CF.name_of("L", 5, dict(base, r_pooled=0.29, r_per_seed=[0.5, 0.5]))[0].endswith("NO-CLOSING")
    assert CF.a_name([False, True]) == "CLOSEREAD-A-CLOSING-IS-READABLE" and CF.a_name([False]) == "CLOSEREAD-NOT-READABLE-FROM-READER-OUTPUTS"


def test_beside_rows():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (10, 10, 3))
    b = CF.beside({"X": X, "dt_s": 0.2}, 10)
    assert np.array_equal(b["reader_closing"], X[:, -1, 2])
    assert np.allclose(b["gap_two_point"], (X[:, -1, 0] - X[:, 0, 0]) / (9 * 0.2))


# ------------------------------------------------------------------------------------------------ controls
def test_c2_known_value_and_planted_clock():
    _, scores, _ = CF.make_synth(seed=3)
    assert CF.c2_definition(scores[0])["pass"]
    sh = dict(scores[0])
    sh["lab_closing"] = np.roll(scores[0]["lab_closing"], 1000)
    assert not CF.c2_definition(sh)["pass"]
    neg = dict(scores[0])
    neg["lab_closing"] = -scores[0]["lab_closing"]
    assert not CF.c2_definition(neg)["pass"]


def test_c3_equality_and_planted_difference():
    _, scores, _ = CF.make_synth(seed=3)
    rl = CF.synth_rulelogs(scores)
    assert CF.c3_dump_is_rule_reader(scores[0], rl[0]["log"], rl[0]["qmap"])["pass"]
    bad = CF.synth_rulelogs(scores, perturb=0.01)
    assert not CF.c3_dump_is_rule_reader(scores[0], bad[0]["log"], bad[0]["qmap"])["pass"]
    log = dict(rl[0]["log"])
    log["p_src"] = log["p_src"] + 0.01
    assert not CF.c3_dump_is_rule_reader(scores[0], log, rl[0]["qmap"])["pass"]
    nomap = (np.array([5.0, 20.0, 60.0]), np.array([5.0, 20.0, 60.0]))
    assert not CF.c3_dump_is_rule_reader(scores[0], rl[0]["log"], nomap)["pass"]          # a different map is a different read


def test_c6_and_c7():
    assert CF.c6_estimator()["pass"]
    big = {"y": np.zeros(2500), "ep": np.asarray(["e%d" % (i % 30) for i in range(2500)])}
    small = {"y": np.zeros(400), "ep": np.asarray(["e%d" % (i % 30) for i in range(400)])}
    few = {"y": np.zeros(900), "ep": np.asarray(["e%d" % (i % 10) for i in range(900)])}
    assert CF.c7_counts(big, big)["pass"] and not CF.c7_counts(small, big)["pass"] and not CF.c7_counts(big, small)["pass"] and not CF.c7_counts(big, few)["pass"]


def test_c5_planted_signal():
    assert CF.c5_planted_signal(10, 200)["pass"]


def test_selftest_of_the_fit_tool(capsys):
    assert CF.selftest() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ZZ CLOSEREAD rc 0 controls_all_pass True"


# ------------------------------------------------------------------------------------------------ the dump tool
def ref_label_read(J, ci, j, dt=0.2):
    """an independent copy of LabelSource.read (v8f_t1_eval_rule.py:92-112)."""
    o, n = int(J["off"][ci]), int(J["n"][ci])
    gap, val = np.asarray(J["lead_gap"], np.float64), J["lead_valid"]
    ok = lambda r: bool(val[r] == 1 and np.isfinite(gap[r]) and 0.0 <= gap[r] <= 60.0)
    if j < 0 or j >= n:
        return False, float("nan"), 0.0
    r = o + int(j)
    if not ok(r):
        return False, float("nan"), 0.0
    idx = [r]
    f = r - 1
    while f >= o and len(idx) < 5 and ok(f) and abs(gap[f + 1] - gap[f]) <= 3.0:
        idx.append(f)
        f -= 1
    g = float(gap[r])
    if len(idx) < 3:
        return True, g, 0.0
    idx = idx[::-1]
    G = gap[idx]
    t = np.arange(len(idx), dtype=np.float64) * dt
    tb = t.mean()
    return True, g, float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())


def test_label_clip_matches_the_label_source():
    rng = np.random.default_rng(7)
    n_clips, T = 4, 40
    gap = np.cumsum(rng.normal(0, 0.8, n_clips * T)) + 30.0
    gap[rng.integers(0, len(gap), 6)] += 20.0                                   # jumps above the step bound
    gap[rng.integers(0, len(gap), 5)] = np.nan
    gap[rng.integers(0, len(gap), 5)] = 80.0                                    # beyond the gap bound
    val = (rng.random(len(gap)) > 0.1).astype(int)
    J = {"off": np.arange(n_clips) * T, "n": np.full(n_clips, T - 3), "lead_gap": gap, "lead_valid": val}
    for ci in range(n_clips):
        pr, g, c, run = CD.label_clip(J, ci, T)
        for j in range(T):
            p0, g0, c0 = ref_label_read(J, ci, j)
            assert pr[j] == p0 and (np.isnan(g[j]) and np.isnan(g0) or g[j] == g0) and abs(c[j] - c0) < 1e-12
            assert (run[j] > 0) == p0 and (c0 == 0.0 or run[j] >= 3)
    assert any(CD.label_clip(J, ci, T)[3].max() >= 3 for ci in range(n_clips))


def make_reader_ckpt(path, seed=0, W=4):
    import torch
    g = np.random.default_rng(seed)
    r = lambda *s: torch.from_numpy(g.normal(0, 0.05, s).astype(np.float32))
    sd = {"tok.0.weight": r(128, 1024), "tok.0.bias": r(128), "tok.2.weight": r(W, 128), "tok.2.bias": r(W), "head.frame.weight": r(64, 40 * W), "head.frame.bias": r(64),
          "head.mlp.0.weight": r(128, 320), "head.mlp.0.bias": r(128), "head.mlp.2.weight": r(3, 128) * 4, "head.mlp.2.bias": r(3)}
    torch.save({"kind": "tokprobe", "input_norm": "ln", "grid": "pool4x10", "state_dict": sd, "hist": 5, "n_tok": 40, "arm": "TK16_s%d" % seed, "step": 1, "W": W, "seed": seed}, path)


def test_torch_reader_equals_the_numpy_reader_and_scales(tmp_path):
    import torch
    p = str(tmp_path / "ck.pt")
    make_reader_ckpt(p)
    rd = TR.TokReader.from_ckpt(p)
    x = np.random.default_rng(1).normal(0, 1, (14, 40, 1024)).astype(np.float16).astype(np.float32)
    trd = CD.TorchReader(rd, "cpu")
    pp, gg, cc, valid = trd.clip(x)
    assert valid.tolist() == [False] * 4 + [True] * 10 and np.isnan(pp[:4]).all()
    w = rd.w
    for f in range(4, 14):
        E = rd.frame_embed(x[f - 4:f + 1][None]).reshape(1, -1)
        h = TR._gelu(E @ w["head.mlp.0.weight"].T + w["head.mlp.0.bias"])
        o = (h @ w["head.mlp.2.weight"].T + w["head.mlp.2.bias"])[0].astype(np.float64)
        assert abs(pp[f] - 1 / (1 + np.exp(-o[0]))) < 1e-4 and abs(gg[f] - 50.0 * o[1]) < 1e-3 and abs(cc[f] - 10.0 * o[2]) < 1e-3     # literal scales: 50 m, 10 m/s
        rp, rg = rd.read(x[f - 4:f + 1][None])
        assert abs(pp[f] - rp[0]) < 1e-4 and abs(gg[f] - rg[0]) < 1e-3                                                                    # the path the rule wrapper reads
    assert float(np.nanmax(np.abs(cc))) > 0.0


def test_dump_end_to_end(tmp_path, capsys):
    import torch
    root = tmp_path
    man, meta, held, exs = _lists_and_meta(n_man=40, ex=(6, 9))
    if len(held) < 4:
        pytest.skip("too few held-out clips in the synthetic manifest")
    ev = ids(5, False, "ev")
    ck = str(root / "ck.pt")
    make_reader_ckpt(ck, seed=1)
    g = np.random.default_rng(2)
    T = 12

    def cache(split, clips):
        d = root / split / split / "clips"
        d.mkdir(parents=True)
        for c in clips:
            torch.save({"pool4x10": torch.from_numpy(g.normal(0, 1, (T, 40, 1024)).astype(np.float16))}, str(d / (c + ".pt")))
        with open(root / split / split / "manifest.jsonl", "w", encoding="utf-8") as fh:
            for c in clips:
                fh.write(json.dumps({"clip": c, "n_frames": T}) + "\n")
        return str(root / split)

    tc = cache("train", [c for c in man if c in set(held)])                       # only the held-out tokens are needed on disk: the manifest below lists all
    with open(root / "train" / "train" / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for c in man:
            fh.write(json.dumps({"clip": c, "n_frames": T}) + "\n")
    ec = cache("eval", ev)

    def join(clips):
        n = len(clips)
        gap = 25.0 + np.cumsum(g.normal(0, 0.3, n * T))
        return {"clip": np.asarray(clips), "off": np.arange(n) * T, "n": np.full(n, T), "lead_gap": gap, "lead_valid": np.ones(n * T, int)}
    Jt, Je = join(man), join(ev)
    np.savez(str(root / "jt.npz"), **Jt)
    np.savez(str(root / "je.npz"), **Je)
    json.dump(meta, open(root / "meta.json", "w"))
    out = str(root / "out")
    args = ["--reader", ck, "--reader-seed", "1", "--train-cache", tc, "--eval-cache", ec, "--join-train", str(root / "jt.npz"), "--join-eval", str(root / "je.npz"), "--train-meta", str(root / "meta.json"),
            "--build-dir", BUILD, "--out-dir", out, "--device", "cpu"]
    assert CD.main(args) == 0
    text = capsys.readouterr().out
    assert " r " not in text and "corr" not in text.lower()
    assert CD.main(args) == 2                                                      # an existing output folder is refused
    fit, score = CF.load_dump(os.path.join(out, "reader_s1_fit.npz")), CF.load_dump(os.path.join(out, "reader_s1_score.npz"))
    assert sorted(fit["clips"].tolist()) == held and score["clips"].tolist() == ev
    assert len(fit["frame"]) == len(held) * T and len(score["frame"]) == len(ev) * T
    assert fit["valid"].tolist() == ([False] * 4 + [True] * (T - 4)) * len(held)
    row = {c: i for i, c in enumerate(man)}
    for k in range(len(held)):
        sl = slice(k * T, (k + 1) * T)
        pr, lg, lc, lr = CD.label_clip(Jt, row[fit["clips"][k]], T)
        assert np.array_equal(fit["lab_present"][sl], pr) and np.array_equal(fit["lab_closing"][sl], lc) and np.array_equal(fit["lab_run"][sl], lr)
        assert (fit["clip_index"][sl] == row[fit["clips"][k]]).all()
    assert np.isfinite(fit["gap"][fit["valid"]]).all() and np.isnan(fit["gap"][~fit["valid"]]).all()
    L = json.load(open(os.path.join(out, "reader_s1_lists.json"), encoding="utf-8"))
    assert L["heldout_ids"] == held and L["meta_n_heldout"] == len(held)
    meta2 = json.load(open(os.path.join(out, "reader_s1_meta.json"), encoding="utf-8"))
    assert meta2["torch_vs_numpy_parity_max_abs_diff"] < 1e-3
    bad = json.loads(json.dumps(meta))
    bad["heldout"]["heldout_sha256"] = "1" * 64
    json.dump(bad, open(root / "meta_bad.json", "w"))
    a2 = list(args)
    a2[a2.index("--train-meta") + 1] = str(root / "meta_bad.json")
    a2[a2.index("--out-dir") + 1] = str(root / "out2")
    with pytest.raises(SystemExit):
        CD.main(a2)
    assert not os.path.exists(str(root / "out2" / "reader_s1_fit.npz"))
