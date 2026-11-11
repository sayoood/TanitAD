"""Unit + integration tests for the S0-P pipeline.  Every guard has a deliberate-regression arm that must go red.
Run (dev box): CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 python -m pytest -q test_s0p.py --basetemp=F:/TanitAD-scratch/stopperc_s0p/pt   (the parent of --basetemp must exist)
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s0p_fit as F  # noqa: E402
import s0p_lib as L  # noqa: E402

TREE = os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree")


# ------------------------------------------------------------------------------------------------ sampler
def _pool(n_clips=40, per=30, seed=0):
    rng = np.random.default_rng(seed)
    clip = np.repeat([f"clip-{i:03d}" for i in range(n_clips)], per)
    ws = np.tile(np.arange(3, 3 + per), n_clips)
    obs = rng.random(len(clip)) < 0.9
    pos = obs & (rng.random(len(clip)) < 0.15)
    return clip, ws, pos, obs & ~pos


def _naive_sample(clip, ws, pos, neg, seed=L.SEED, n_pos=50, n_neg=150):
    """The regression arm: the same draws WITHOUT the (clip id, window start) sort (pool taken in array order)."""
    rng = np.random.default_rng(seed)
    out = []
    for mask, k in ((pos, n_pos), (neg, n_neg)):
        ix = np.where(mask)[0]
        out.append(np.sort(ix[rng.choice(len(ix), size=k, replace=False)]))
    return out


def test_sampler_counts_disjoint_deterministic():
    clip, ws, pos, neg = _pool()
    ip, ineg = L.draw_sample(clip, ws, pos, neg, n_pos=50, n_neg=150)
    assert len(ip) == 50 and len(ineg) == 150 and len(set(ip) | set(ineg)) == 200
    assert pos[ip].all() and neg[ineg].all()
    ip2, ineg2 = L.draw_sample(clip, ws, pos, neg, n_pos=50, n_neg=150)
    assert (ip == ip2).all() and (ineg == ineg2).all()
    ip3, _ = L.draw_sample(clip, ws, pos, neg, seed=1, n_pos=50, n_neg=150)
    assert not (ip == ip3).all(), "the seed must matter"


def test_sampler_is_invariant_to_pool_order_and_the_naive_arm_is_not():
    """The card sorts the pool by (clip id, window start) before drawing, so the pick must not depend on the order the windows arrive in. Deliberate regression: the unsorted draw DOES depend on it."""
    clip, ws, pos, neg = _pool()
    perm = np.random.default_rng(5).permutation(len(clip))
    ip, ineg = L.draw_sample(clip, ws, pos, neg, n_pos=50, n_neg=150)
    ipp, inegp = L.draw_sample(clip[perm], ws[perm], pos[perm], neg[perm], n_pos=50, n_neg=150)
    sel = set(zip(clip[ip], ws[ip])), set(zip(clip[ineg], ws[ineg]))
    selp = set(zip(clip[perm][ipp], ws[perm][ipp])), set(zip(clip[perm][inegp], ws[perm][inegp]))
    assert sel == selp
    nip, nineg = _naive_sample(clip, ws, pos, neg)
    nipp, ninegp = _naive_sample(clip[perm], ws[perm], pos[perm], neg[perm])
    nsel = set(zip(clip[nip], ws[nip]))
    nselp = set(zip(clip[perm][nipp], ws[perm][nipp]))
    assert nsel != nselp, "regression arm must be order dependent (else this test cannot see the sort being removed)"


def test_sampler_refuses_a_short_pool_and_overlapping_masks():
    clip, ws, pos, neg = _pool(n_clips=2, per=10)
    with pytest.raises(ValueError):
        L.draw_sample(clip, ws, pos, neg, n_pos=500, n_neg=1)
    with pytest.raises(AssertionError):
        L.draw_sample(clip, ws, pos, pos, n_pos=1, n_neg=1)


# ------------------------------------------------------------------------------------------------ outcome logic
def _s(res_pt=0.0, res_lo=-0.1, hu=((0.0, 0.5, 0.8), (0.0, 0.5, 0.8)), mlp=None):
    # hu entries: (paired diff, paired-diff CI lower bound, T point[, probe CI lower bound]) -- AMENDMENT_2_S0P.md
    def hu_block(h):
        return {t: {"diff": x[0], "diff_lo": x[1], "t_point": x[2], "probe_lo": x[3] if len(x) > 3 else x[2] - 0.01} for t, x in zip(F.T_TAGS, h)}
    s = {"res": {"linear": {"point": res_pt, "lo": res_lo}, "mlp": {"point": 0.0, "lo": -1.0}}, "hu": {"linear": hu_block(hu)}}
    if mlp:
        s["hu"]["mlp"] = hu_block(mlp)
    return s


CASES = [
    # (summary, expected RES-ADDS, expected HEAD-UNDERUSES)
    (_s(), False, False),
    (_s(res_pt=0.02, res_lo=0.001), True, False),                    # exactly at the bar, CI above 0
    (_s(res_pt=0.0199, res_lo=0.001), False, False),                 # just under the bar
    (_s(res_pt=0.05, res_lo=0.0), False, False),                     # CI lower bound exactly 0 is not > 0
    (_s(hu=((0.037, 0.001, 0.80), (0.04, 0.002, 0.80))), False, True),  # at the bar at both T arms, paired CI above 0
    (_s(hu=((0.037, 0.001, 0.80), (0.036, 0.002, 0.80))), False, False),  # one T arm under 0.037
    (_s(hu=((0.05, 0.0, 0.80), (0.05, 0.01, 0.80))), False, False),   # paired-diff CI lower bound exactly 0 is not > 0
    (_s(hu=((0.05, -0.01, 0.80, 0.85), (0.05, -0.01, 0.80, 0.85))), False, False),  # probe CI above T's point but the PAIRED CI covers 0: not HU (amendment 2)
    (_s(mlp=((0.05, 0.01, 0.8), (0.05, 0.01, 0.8))), False, True),     # MLP alone is enough (linear OR mlp)
    (_s(res_pt=0.03, res_lo=0.01, hu=((0.05, 0.01, 0.8), (0.05, 0.01, 0.8))), True, True),
]


def _check_decide(fn):
    bad = []
    for i, (s, res, hu) in enumerate(CASES):
        d = fn(s)
        if (d["RES-ADDS"], d["HEAD-UNDERUSES"], d["FLAT"]) != (res, hu, not (res or hu)):
            bad.append(i)
    return bad


def test_decide_outcomes_table():
    assert _check_decide(F.decide_outcomes) == []


def test_decide_outcomes_regression_arms_go_red():
    """Three planted mutants of the rule must each fail at least one case: a lowered RES bar, a dropped CI condition, an HU rule that needs only one T arm."""
    def mut_bar(s):
        old = F.RES_MIN
        F.RES_MIN = 0.0
        try:
            return F.decide_outcomes(s)
        finally:
            F.RES_MIN = old

    def mut_ci(s):
        s2 = json.loads(json.dumps(s))
        for r in s2["res"].values():
            r["lo"] = 1.0
        return F.decide_outcomes(s2)

    def mut_one_arm(s):
        s2 = json.loads(json.dumps(s))
        for m in s2["hu"]:
            for t in list(s2["hu"][m])[1:]:
                del s2["hu"][m][t]
        return F.decide_outcomes(s2)

    # a single-arm summary is judged on the dict as given: with len(per_t) != 2 the real rule refuses, so a mutant must be run through a rule that accepts one arm
    def mut_hu_any(s):
        res_models = [m for m, r in s["res"].items() if r["point"] >= F.RES_MIN and r["lo"] > 0]
        hu_models = [m for m, per in s["hu"].items() if any(v["diff"] >= F.HU_MIN and v["diff_lo"] > 0 for v in per.values())]
        return {"RES-ADDS": bool(res_models), "HEAD-UNDERUSES": bool(hu_models), "FLAT": not (res_models or hu_models)}
    def mut_unpaired(s):                                   # the reading AMENDMENT_2_S0P.md set aside: the probe's own CI vs T's point
        res_models = [m for m, r in s["res"].items() if r["point"] >= F.RES_MIN and r["lo"] > 0]
        hu_models = [m for m, per in s["hu"].items() if len(per) == len(F.T_TAGS) and all(v["diff"] >= F.HU_MIN and v["probe_lo"] > v["t_point"] for v in per.values())]
        return {"RES-ADDS": bool(res_models), "HEAD-UNDERUSES": bool(hu_models), "FLAT": not (res_models or hu_models)}
    for name, mut in (("bar", mut_bar), ("ci", mut_ci), ("hu_any", mut_hu_any), ("unpaired", mut_unpaired)):
        assert _check_decide(mut) != [], f"mutant {name} was not caught"
    del mut_one_arm


def test_select_best_ties_go_to_the_smaller_model():
    lin = [({"dim": d, "C": c}, 0.8) for d in (16, 32, 64) for c in (0.001, 0.01, 0.1, 1.0)]
    assert F.select_best(lin) == {"dim": 16, "C": 0.001}
    mlp = [({"dim": 32, "hidden": h, "alpha": a}, 0.8) for h in (64, 256) for a in (1e-4, 1e-2)]
    assert F.select_best(mlp) == {"dim": 32, "hidden": 64, "alpha": 1e-2}
    assert F.select_best([({"dim": 64, "C": 1.0}, 0.9), ({"dim": 16, "C": 0.001}, 0.8)]) == {"dim": 64, "C": 1.0}


def test_config_grids_are_the_cards():
    assert len(F.config_grid("V40+E", "linear")) == 12 and len(F.config_grid("V40+E", "mlp")) == 12
    assert len(F.config_grid("E", "linear")) == 4 and len(F.config_grid("E-limit", "mlp")) == 4
    assert {c["C"] for c in F.config_grid("M+E", "linear")} == {0.001, 0.01, 0.1, 1.0}
    assert {(c["hidden"], c["alpha"]) for c in F.config_grid("E", "mlp")} == {(64, 1e-4), (64, 1e-2), (256, 1e-4), (256, 1e-2)}


def test_episode_permuted_labels_no_clip_keeps_its_labels():
    clip = np.repeat(np.array(["a", "b", "c", "d"]), [5, 5, 6, 4])
    y = np.zeros(len(clip), int)
    y[clip == "a"] = 1                                               # only clip a carries positives
    for seed in range(5):
        yp = F.episode_permuted_labels(y, clip, seed)
        assert (yp[clip == "a"] == 0).all(), "clip a must have lost its labels"
        assert yp.sum() > 0 and len(yp) == len(y)


def test_auroc_planted_values():
    y = np.array([0, 0, 1, 1, 0, 1], bool)
    assert L.auroc(y.astype(float), y) == 1.0 and L.auroc(np.zeros(6), y) == 0.5 and L.auroc(1.0 - y, y) == 0.0


# ------------------------------------------------------------------------------------------------ train-side extractor on a synthetic world
def _world(tmp_path):
    L.setup_tree(TREE)
    import torch
    from tanitad.data import v8f_mini_data as D
    w = D.make_synthetic_world(tmp_path / "world", n_train=6, n_eval=3, T=70)
    root = tmp_path / "world"
    rng = np.random.default_rng(0)
    for p in (root / "train" / "clips").glob("*.pt"):                # add pool8x20 + mean, as the real cache has
        o = torch.load(p, weights_only=True)
        T = o["pool4x10"].shape[0]
        o["pool8x20"] = torch.as_tensor(rng.normal(0, 1, (T, 160, o["pool4x10"].shape[2])).astype(np.float32)).to(torch.float8_e4m3fn)
        o["mean"] = o["pool4x10"].to(torch.float32).mean(1).to(torch.float16)
        torch.save(o, p)
    J = D.load_join(root / "join_train.npz")
    gy = J["goal_y"].astype(np.uint32)
    t_of_row = np.concatenate([np.arange(int(n)) for n in J["n"]])
    gy[(t_of_row % 7) == 0] |= np.uint32(1 << L.GOAL_STOP_BIT)       # STOP_POINT positives on t % 7 == 0
    J["goal_y"] = gy
    D.save_join(J, root / "join_train.npz")
    return root, D, J


def _extract(root, out, extra=()):
    import s0p_extract_train as X
    return X.main(["--tree", TREE, "--cache-dir", str(root), "--split", "train", "--join", str(root / "join_train.npz"), "--tracks", str(root / "native_tracks.pt"),
                   "--out-dir", str(out), "--exclude", "none", "--n-pos", "20", "--n-neg", "60", "--expect-pool", "none", "--eval-cache-dir", str(root), *extra])


def check_limit_is_causal(E_arr, d_by_clip, clips, ws, E):
    """True iff every row's limit / limit_ok equal the causal bucket of the clip's past speed at its window start."""
    for i, (c, t) in enumerate(zip(clips, ws)):
        lim, lok = E.causal_bucket(d_by_clip[str(c)], int(t), E.CAUSAL_STEPS)
        if not (np.array_equal(E_arr[i, 13:17], lim) and E_arr[i, 17] == float(lok)):
            return False
    return True


def test_extractor_end_to_end_and_regression_arms(tmp_path):
    import torch
    root, D, J = _world(tmp_path)
    L.setup_tree(TREE)
    E = L.load_eval_module(TREE)
    # eval cache dir = the same world's eval split: disjoint from train by construction
    meta = _extract(root, tmp_path / "out")
    out = tmp_path / "out"
    a = {k: np.load(out / f"{k}.npy") for k in L.FIELDS}
    assert len(a["y"]) == 80 and int(a["y"].sum()) == 20, "20 positives + 60 negatives"
    assert meta["n_pos_sample"] == 20 and meta["disjoint"]["overlap"] == 0
    # labels re-read from the join bits
    clips_all = [str(c) for c in J["clip"].tolist()]
    ci_of = {c: i for i, c in enumerate(clips_all)}
    rows = np.array([int(J["off"][ci_of[str(c)]]) + int(t) for c, t in zip(a["clip"], a["ws"])])
    assert ((J["goal_y"][rows].astype(np.int64) >> L.GOAL_STOP_BIT) & 1).astype(np.uint8).tolist() == a["y"].tolist()
    assert (J["has"][rows].astype(bool)).all() and (((J["goal_w"][rows].astype(np.int64) >> L.GOAL_STOP_BIT) & 1) == 1).all(), "goal-observed windows only"
    assert (a["ws"] >= 3).all() and (a["ws"] < 70 - 30 + 0).all()
    # grid rows equal the stored bytes
    cache = {}
    for i in range(len(a["y"])):
        c = str(a["clip"][i])
        if c not in cache:
            cache[c] = torch.load(root / "train" / "clips" / f"{c}.pt", weights_only=True)
        o, t = cache[c], int(a["ws"][i])
        assert np.array_equal(a["v40"][i], o["pool4x10"][t].view(torch.uint8).numpy())
        assert np.array_equal(a["v160"][i], o["pool8x20"][t].view(torch.uint8).numpy())
        assert np.array_equal(a["mean"][i], o["mean"][t].numpy())
        assert np.array_equal(a["E"][i, 8:13], J["nav"][int(rows[i])])
    # the limit is the CAUSAL bucket, never the logged sidecar
    d_by_clip = {c: cache[c]["v"].numpy() for c in cache}
    assert check_limit_is_causal(a["E"], d_by_clip, a["clip"], a["ws"], E)
    logged = np.array([0, 1, 0, 0], np.float32)
    assert not (a["E"][:, 13:17] == logged).all(), "synthetic world must make the causal and logged buckets differ (else the regression arm below proves nothing)"
    mut = a["E"].copy()
    mut[:, 13:17], mut[:, 17] = logged, 1.0                          # regression arm: the logged sidecar leaks in
    assert not check_limit_is_causal(mut, d_by_clip, a["clip"], a["ws"], E), "the causal-limit check must go red on a logged limit"
    # determinism + the sample is the sampler's
    meta2 = _extract(root, tmp_path / "out2")
    assert meta2["sample_sha256"] == meta["sample_sha256"]
    assert all(np.array_equal(np.load(out / f"{k}.npy"), np.load(tmp_path / "out2" / f"{k}.npy")) for k in ("E", "v40", "v160", "mean", "y", "ws"))


def test_extractor_pool_assert_goes_red_on_a_drifted_rule(tmp_path):
    root, D, J = _world(tmp_path)
    import s0p_extract_train as X
    with pytest.raises(AssertionError, match="pool counts"):
        X.main(["--tree", TREE, "--cache-dir", str(root), "--join", str(root / "join_train.npz"), "--tracks", str(root / "native_tracks.pt"), "--out-dir", str(tmp_path / "o"), "--exclude", "none",
                "--n-pos", "5", "--n-neg", "5", "--dry-run"])                 # default --expect-pool = the banked counts, which the synthetic pool cannot equal


def test_extractor_resumes_after_an_interruption(tmp_path):
    root, D, J = _world(tmp_path)
    ref = _extract(root, tmp_path / "ref")
    import s0p_extract_train as X
    X.CKPT_EVERY = 1
    orig = D.V8FMiniWindows.item
    state = {"n": 0}

    def boom(self, *a, **k):
        state["n"] += 1
        if state["n"] > 30:
            raise KeyboardInterrupt("simulated power cut")
        return orig(self, *a, **k)
    D.V8FMiniWindows.item = boom
    try:
        with pytest.raises(KeyboardInterrupt):
            _extract(root, tmp_path / "cut")
    finally:
        D.V8FMiniWindows.item = orig
    prog = json.load(open(tmp_path / "cut" / "progress.json", encoding="utf-8"))
    assert 0 < prog["n_clips_done"] < ref["n_clips_in_sample"]
    meta = _extract(root, tmp_path / "cut")
    assert meta["sample_sha256"] == ref["sample_sha256"]
    for k in ("E", "v40", "v160", "mean", "y", "clip", "ws", "eid"):
        assert np.array_equal(np.load(tmp_path / "cut" / f"{k}.npy"), np.load(tmp_path / "ref" / f"{k}.npy")), k


def test_extractor_refuses_overlapping_train_and_eval_clips(tmp_path):
    root, D, J = _world(tmp_path)
    import shutil
    ev = tmp_path / "evalroot" / "eval"
    (ev / "clips").mkdir(parents=True)
    first = sorted(p.name for p in (root / "train" / "clips").glob("*.pt"))[0]
    shutil.copy(root / "train" / "clips" / first, ev / "clips" / first)
    with open(ev / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"clip": first[:-3], "n_frames": 70}) + "\n")
    import s0p_extract_train as X
    with pytest.raises(AssertionError, match="overlap"):
        X.main(["--tree", TREE, "--cache-dir", str(root), "--join", str(root / "join_train.npz"), "--tracks", str(root / "native_tracks.pt"), "--out-dir", str(tmp_path / "o"), "--exclude", "none",
                "--n-pos", "5", "--n-neg", "5", "--expect-pool", "none", "--eval-cache-dir", str(tmp_path / "evalroot"), "--dry-run"])


# ------------------------------------------------------------------------------------------------ shards (--shard i/K) and load_any over shard dirs
def _extract_shards(root, out, K):
    return [_extract(root, out, extra=("--shard", f"{i}/{K}")) for i in range(K)]


def _cat_shards(out, K, k):
    return np.concatenate([np.load(out / f"shard_{i}" / f"{k}.npy") for i in range(K)])


@pytest.mark.parametrize("K", [4, 3])
def test_shards_concatenate_to_the_unsharded_run_byte_for_byte(tmp_path, K):
    root, D, J = _world(tmp_path)
    ref = _extract(root, tmp_path / "ref")
    metas = _extract_shards(root, tmp_path / "sh", K)
    out = tmp_path / "sh"
    assert [m["shard"]["lo"] for m in metas] == [(i * 80) // K for i in range(K)] and metas[-1]["shard"]["hi"] == 80
    assert sum(m["n_shard"] for m in metas) == 80 and all(m["sample_sha256"] == ref["sample_sha256"] for m in metas)
    for k in L.FIELDS:
        a, b = np.load(tmp_path / "ref" / f"{k}.npy"), _cat_shards(out, K, k)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), f"{k}: concatenated shards differ from the unsharded run"
    for i in range(K):                                              # every shard's own manifest verifies (complete shard)
        for line in open(out / f"shard_{i}" / "MANIFEST.sha256", encoding="utf-8"):
            h, n = line.split()
            assert L.sha256_file(str(out / f"shard_{i}" / n)) == h
    # load_any over the parent dir equals the unsharded load, lazily, including slices that straddle shard boundaries
    t, u = F.load_any(str(out)), F.load_any(str(tmp_path / "ref"))
    for k in ("E", "y", "clip", "ws", "eid"):
        assert np.array_equal(np.asarray(t[k]), np.asarray(u[k])), k
    for k in ("v40", "v160", "mean"):
        assert len(t[k]) == len(u[k]) and t[k].shape == u[k].shape and t[k].dtype == u[k].dtype
        for lo, hi in ((0, 80), (0, 1), (19, 41), (79, 80), (30, 31), (50, 200)):
            assert np.array_equal(t[k][lo:hi], u[k][lo:hi]), (k, lo, hi)
        assert np.array_equal(t[k][-1], u[k][-1]) and np.array_equal(t[k][41], u[k][41])
    assert F.fit_pca(t["v40"])["n_tokens"] == F.fit_pca(u["v40"])["n_tokens"]
    pa, pb = F.fit_pca(t["v40"]), F.fit_pca(u["v40"])
    assert np.allclose(pa["mean"], pb["mean"]) and np.allclose(F.project(t["v40"], pa), F.project(u["v40"], pb))
    assert t["meta"]["shards"] == K and t["meta"]["n_sample"] == 80


def test_shard_regression_arms_go_red(tmp_path):
    root, D, J = _world(tmp_path)
    ref = _extract(root, tmp_path / "ref")
    _extract_shards(root, tmp_path / "sh", 4)
    out = tmp_path / "sh"
    # a flipped byte in one shard breaks the byte comparison (the check is not vacuous)
    m = np.load(out / "shard_2" / "v160.npy")
    m[0, 0, 0] ^= 1
    assert _cat_shards(out, 4, "v160").tobytes() == np.load(tmp_path / "ref" / "v160.npy").tobytes()   # control: the unmutated shards are equal
    np.save(out / "shard_2" / "v160_mut.npy", m)
    mutated = np.concatenate([np.load(out / f"shard_{i}" / ("v160_mut.npy" if i == 2 else "v160.npy")) for i in range(4)])
    assert mutated.tobytes() != np.load(tmp_path / "ref" / "v160.npy").tobytes()
    # swapped shard order breaks it too
    swapped = np.concatenate([np.load(out / f"shard_{i}" / "E.npy") for i in (1, 0, 2, 3)])
    assert swapped.tobytes() != np.load(tmp_path / "ref" / "E.npy").tobytes()
    # load_any refuses a missing shard and a mixed sample
    import shutil
    shutil.copytree(out, tmp_path / "miss")
    shutil.rmtree(tmp_path / "miss" / "shard_3")
    with pytest.raises(AssertionError, match="do not match the K"):
        F.load_any(str(tmp_path / "miss"))
    shutil.copytree(out, tmp_path / "mix")
    mp = tmp_path / "mix" / "shard_1" / "meta.json"
    mj = json.load(open(mp, encoding="utf-8"))
    mj["sample_sha256"] = "0" * 64
    json.dump(mj, open(mp, "w", encoding="utf-8"))
    with pytest.raises(AssertionError, match="different samples"):
        F.load_any(str(tmp_path / "mix"))


def test_shard_dry_run_resume_and_arg_guards(tmp_path):
    root, D, J = _world(tmp_path)
    import s0p_extract_train as X
    assert X.parse_shard(None) is None and X.parse_shard("2/4") == (2, 4)
    for bad in ("4/4", "-1/4", "0/0"):
        with pytest.raises(AssertionError):
            X.parse_shard(bad)
    tiles = [X.shard_bounds(80, i, 4) for i in range(4)]
    assert tiles[0][0] == 0 and tiles[-1][1] == 80 and all(tiles[i][1] == tiles[i + 1][0] for i in range(3))
    d = _extract(root, tmp_path / "dry", extra=("--shard", "1/4", "--dry-run"))
    assert d["shard"]["lo"] == 20 and d["shard"]["hi"] == 40 and d["n_shard"] == 20
    assert (tmp_path / "dry" / "sample.json").exists() and not (tmp_path / "dry" / "shard_1" / "E.npy").exists()
    # a shard dir resumed under a different K is refused (its progress key names the shard)
    import shutil
    _extract(root, tmp_path / "x", extra=("--shard", "1/4"))
    shutil.copytree(tmp_path / "x" / "shard_1", tmp_path / "y" / "shard_1")
    with pytest.raises(AssertionError, match="different sample"):
        _extract(root, tmp_path / "y", extra=("--shard", "1/3"))
