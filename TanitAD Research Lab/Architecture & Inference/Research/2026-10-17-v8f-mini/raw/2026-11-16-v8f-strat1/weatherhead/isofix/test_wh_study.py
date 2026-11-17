"""Tests of the PREREG_V8F_WEATHERHEAD STUDY scripts (fit library, P-COMMIT streaming, reader outcome rules, token).  The model-side tests are tree/stack/tests/test_v8f_weatherhead.py.

Run: pytest build/test_wh_study.py --basetemp=<work>/tmp/bt  (env.sh: tree-first PYTHONPATH, CUDA_VISIBLE_DEVICES=-1).
Each mechanism has a MUTATION that must turn its check RED."""
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wh_lib as W  # noqa: E402
import wh_fit  # noqa: E402
import wh_check as C  # noqa: E402
import wh_read as RD  # noqa: E402

M = W.M


def fake_cache(root, n_clips=5, n_frames=14, seed=0):
    os.makedirs(f"{root}/clips", exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    names = []
    for c in range(n_clips):
        nm = f"clip-{c:02d}"
        names.append(nm)
        x = (torch.randn(n_frames, 40, 1024, generator=g) * 2).to(torch.float8_e4m3fn)
        torch.save({"pool4x10": x}, f"{root}/clips/{nm}.pt")
    return names


def fake_table(n_clips=5, n_frames=14, seed=0):
    rng = np.random.default_rng(seed)
    N = n_clips * n_frames
    t = {"train_clip": np.array([f"clip-{c:02d}" for c in range(n_clips)]), "train_off": np.arange(n_clips) * n_frames, "train_n": np.full(n_clips, n_frames), "train_flag": np.zeros(N, np.uint8)}
    t["train_flag"][3] = 2
    for v, n in zip(M.CTX_VARS, M.CTX_N):
        t[f"train_{v}_id"] = rng.integers(0, n, N).astype(np.int8)
        t[f"train_{v}_valid"] = np.ones(N, bool)
    t["train_illumination_id"][10:14] = M.CTX_TWILIGHT_ID
    return t


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("whcache"))
    return root, fake_cache(root)


# ------------------------------------------------------------------------------------------------ P-COMMIT (ruling 6)
def test_pcommit_chooses_stream_when_commit_is_short_and_ram_when_it_is_not():
    n = 36_000
    assert W.choose_source(n, "auto", headroom=1e9)["mode"] == "stream"
    assert W.choose_source(n, "auto", headroom=W.feature_bytes(n) + W.COMMIT_MARGIN + 1)["mode"] == "ram"
    assert W.choose_source(n, "auto", headroom=W.feature_bytes(n) + W.COMMIT_MARGIN - 1)["mode"] == "stream"
    assert W.feature_bytes(36_000) == 36_000 * 4 * 40 * 1024 and 5.8e9 < W.feature_bytes(36_500) < 6.1e9


def test_streamed_features_are_byte_equal_to_in_ram_features_on_two_batches(cache):
    root, names = cache
    tab = fake_table()
    rs = W.train_rowset(tab)
    ram, st = W.RamSource(root, names, rs), W.StreamSource(root, names, rs)
    rng = np.random.RandomState(0)
    for _ in range(2):
        ix = rng.choice(len(rs["ci"]), 11, replace=False)
        a, b = ram.gather(ix).view(torch.uint8), st.gather(ix).view(torch.uint8)
        assert torch.equal(a, b)


def test_regression_arm_an_offset_of_one_window_reads_red(cache):
    root, names = cache
    rs = W.train_rowset(fake_table())
    ram, st = W.RamSource(root, names, rs), W.StreamSource(root, names, rs, offset=1)
    ix = np.arange(0, len(rs["ci"]), 3)
    assert not torch.equal(ram.gather(ix).view(torch.uint8), st.gather(ix).view(torch.uint8))


def test_window_set_is_the_probes_and_rows_are_off_plus_t():
    tab = fake_table()
    rs = W.train_rowset(tab)
    assert np.array_equal(rs["row"], tab["train_off"][rs["ci"]] + rs["t"]) and rs["t"].min() >= 3 and rs["t"].max() <= 13
    per = [len(rs["t"][rs["ci"] == c]) for c in range(5)]
    assert all(1 <= p <= 8 for p in per)


# ------------------------------------------------------------------------------------------------ labels, G-CLOCK
def test_labels_mask_flag_invalid_and_twilight():
    tab = fake_table()
    y = W.fit_labels(tab, np.arange(14))
    assert (y[3] == -1).all()                                                           # flag 2
    tab["train_weather_valid"][5] = False
    assert W.fit_labels(tab, np.array([5]))[0, 1] == -1
    tab["train_illumination_id"][11] = M.CTX_TWILIGHT_ID
    assert W.fit_labels(tab, np.array([11]))[0, 3] == -1


def test_clock_matches_the_reader_and_a_shift_reads_red():
    tab = fake_table()
    rs = W.train_rowset(tab)
    ev = {"eval_" + k[6:]: v for k, v in tab.items() if k.startswith("train_")}
    assert np.array_equal(RD.R.label_rows(ev, rs["row"]), W.fit_labels(tab, rs["row"]))
    assert not np.array_equal(RD.R.label_rows(ev, rs["row"]), W.fit_labels(tab, rs["row"], shift=1))


def test_r1_permutes_between_clips_and_keeps_the_label_multiset_per_variable():
    tab = fake_table()
    rs = W.train_rowset(tab)
    y = W.fit_labels(tab, rs["row"])
    yp = W.permute_labels_between_clips(y, rs["ci"], 0)
    assert not np.array_equal(y, yp) and yp.shape == y.shape


# ------------------------------------------------------------------------------------------------ epoch choice
def test_pick_epochs_argmax_with_ties_to_the_smaller_E():
    sc = {4: {v: 0.1 for v in M.CTX_VARS}, 8: {v: 0.3 for v in M.CTX_VARS}, 16: {v: 0.3 for v in M.CTX_VARS}}
    sc[16]["weather"] = 0.31
    sc[4]["surface"] = sc[8]["surface"] = sc[16]["surface"] = None
    e = W.pick_epochs(sc)
    assert e["country"] == 8 and e["weather"] == 16 and e["surface"] == 4


def test_heldout_split_is_80_20_by_clip_and_deterministic():
    ci = np.repeat(np.arange(50), 8)
    fi, hi = W.heldout_split(ci)
    assert len(np.unique(ci[hi])) == 10 and not set(ci[fi]) & set(ci[hi]) and np.array_equal(hi, W.heldout_split(ci)[1])


# ------------------------------------------------------------------------------------------------ the fit step reaches the heads (G-LIVE on the study side)
def tiny_model():
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, ctx_pool="on")
    return M.V8FMini(cfg, W.p10_zero())


def tiny_batch(B=12):
    g = torch.Generator().manual_seed(2)
    f = (torch.randn(B, 4, 40, 1024, generator=g) * 2).to(torch.float8_e4m3fn)
    y = torch.stack([torch.randint(0, n, (B,), generator=g) for n in M.CTX_N], 1)
    return f, y


def test_live_check_reads_pass_and_each_mutation_reads_red():
    f, y = tiny_batch()
    assert W.live_check(tiny_model(), f, y)["pass"]
    ns = dict(W.__dict__)
    exec(inspect.getsource(W.fit_step).replace('loss = T["ctx_pool"]', 'loss = logits.sum() * 0.0'), ns)         # m1
    assert not W.live_check(tiny_model(), f, y, step=ns["fit_step"])["pass"]
    ns2 = dict(M.__dict__)
    import textwrap
    exec(textwrap.dedent(inspect.getsource(M.V8FMini.ctx_pool_fit_logits)).replace("torch.set_grad_enabled(torch.is_grad_enabled() and train_adapter)", "torch.enable_grad()"), ns2)   # m2
    mut = type("Mut", (M.V8FMini,), {"ctx_pool_fit_logits": ns2["ctx_pool_fit_logits"]})
    m = mut(M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, ctx_pool="on"), W.p10_zero())
    r = W.live_check(m, f, y)
    assert not r["pass"] and r["other_grad_abs_sum"] > 0


def test_fit_head_trains_only_the_head_and_the_census_reads_55():
    class Src:
        def gather(self, ix):
            f, _ = tiny_batch(len(ix))
            return f
    m = tiny_model()
    before = {k: v.clone() for k, v in m.state_dict().items() if not k.startswith("ctx_pool_head.")}
    y = tiny_batch(24)[1].numpy()
    rec = W.fit_head(m, Src(), y, np.arange(24), 2, 0, torch.device("cpu"), False, lambda *a: None, bs=8)
    assert rec["census"]["pass"] and rec["census"]["n_with_state"] == 55
    assert all(torch.equal(before[k], m.state_dict()[k]) for k in before)             # zero driving parameters moved
    assert rec["loss_per_epoch"][1] <= rec["loss_per_epoch"][0] + 1.0


def test_fit_seed_sets_the_head_init_and_the_batch_order():
    m = tiny_model()
    W.reinit_head(m, 3)
    a = W.head_state(m)
    W.reinit_head(m, 3)
    b = W.head_state(m)
    W.reinit_head(m, 4)
    c = W.head_state(m)
    assert all(torch.equal(a[k], b[k]) for k in a) and not all(torch.equal(a[k], c[k]) for k in a)


def test_iso_a_flags_a_changed_s1_tensor_and_a_stray_new_key():
    sd = {"a.weight": torch.arange(4.0), "b": torch.ones(2)}
    ok = dict(sd, **{"ctx_pool_head.x": torch.zeros(1)})
    assert W.iso_a(sd, ok) == []
    bad = dict(ok)
    bad["a.weight"] = torch.arange(4.0) + 1e-3
    assert W.iso_a(sd, bad) and W.iso_a(sd, dict(ok, stray=torch.zeros(1))) and W.iso_a(sd, {"a.weight": sd["a.weight"]})


# ------------------------------------------------------------------------------------------------ the reader
def planted(world, acc_by_var, seed=0):
    """logits whose argmax is the label with probability acc (per variable) -> [N, 41]"""
    rng = np.random.RandomState(seed)
    out = []
    for k, n in enumerate(M.CTX_N):
        y = world.y[:, k]
        pred = np.where(rng.rand(len(y)) < acc_by_var[k], np.maximum(y, 0), (np.maximum(y, 0) + 1) % n)
        z = np.full((len(y), n), -5.0)
        z[np.arange(len(y)), pred] = 5.0
        out.append(z)
    return np.concatenate(out, 1)


@pytest.fixture(scope="module")
def world():
    return RD.World(nb=200)


def arm_of(world, logits):
    return {"rows": world.rows(logits), "correct": world.correct(logits), "logits": logits}


def test_k0_reproduces_every_strat1_row_exactly_and_a_perturbation_reads_red():
    world = RD.World(nb=2000)
    assert not RD.k0(RD.World(nb=200))["pass"]                                          # a different B is not the anchor
    assert RD.k0(world)["pass"]
    lg = RD.shipped_logits(None, 0).copy()
    lg[0, 25:29] = lg[0, 25:29][::-1]
    assert not RD.k0(world, shipped=lambda s: lg if s == 0 else RD.shipped_logits(None, s))["pass"]


def test_controls_read_their_known_values(world):
    c = RD.controls(world, nb=100)
    assert all(c[k]["pass"] for k in ("K2", "K3", "K5", "K6")) and abs(c["K6"]["read"] - 0.0779) < 5e-4


def test_an_h_dump_without_ctx_pool_logits_reads_no_output_and_never_falls_back(tmp_path, world):
    p = tmp_path / "eval_WH-H_s0_real.npz"
    np.savez(p, ctx_logits=np.zeros((len(world.ci), 41)), win_ci=world.ci, win_t=world.t)
    assert RD.head_logits(str(p)) is None                                               # m5: the evaluator dropped --ctx-pool -> "no output"
    assert RD.head_logits(str(tmp_path / "missing.npz")) is None


def test_k_id_reads_pass_on_equal_logits_and_red_when_the_reader_substitutes_ctx_logits(tmp_path, world):
    lg = planted(world, [0.9] * 5)
    ev, kid = tmp_path / "eval.npz", tmp_path / "kid.npz"
    np.savez(ev, ctx_pool_logits=lg, ctx_logits=lg + 1.0, win_ci=world.ci, win_t=world.t)
    np.savez(kid, ctx_pool_logits=lg + 1e-7, win_ci=world.ci, win_t=world.t)
    assert RD.k_id(str(ev), str(kid))["pass"]
    orig = RD.head_logits
    RD.head_logits = lambda p: np.load(p, allow_pickle=True)["ctx_logits"] if os.path.exists(p) else None          # m6: the reader substitutes ctx_logits
    try:
        assert not RD.k_id(str(ev), str(kid))["pass"]
    finally:
        RD.head_logits = orig


PASS = {k: "pass" for k in RD.INSTRUMENT_REQUIRED}


def decide_with(world, arms, ins=None, pair=None, F=None):
    A = {k: arm_of(world, v) for k, v in arms.items()}
    P = {}
    for s in (0, 1):
        h = A.get(f"H_s{s}")
        if not h:
            continue
        for k, v in enumerate(M.CTX_VARS):
            for nm, other in (("H-RA", f"RA_s{s}"), ("H-shipped", f"SH_s{s}")):
                if other in A:
                    P[(nm, s, v)] = world.paired(h["correct"], A[other]["correct"], k)
    return RD.decide(A, {"pair": pair or P, "F_fit": F or {}, "instrument": ins or dict(PASS)})


def test_outcome_weather_detects_and_the_wording_follows_the_ra_rows(world):
    good, ra_low, ra_same = planted(world, [0.95] * 5, 1), planted(world, [0.6] * 5, 2), planted(world, [0.95] * 5, 3)
    arms = {"H_s0": good, "H_s1": planted(world, [0.95] * 5, 4), "K4p_s0": planted(world, [0.1] * 5, 5), "K4p_s1": planted(world, [0.1] * 5, 6), "RA_s0": ra_low, "RA_s1": ra_low}
    out = decide_with(world, arms, F={v: 0.02 for v in M.CTX_VARS})
    assert out["weather"]["outcome"] == "WH-DETECTS + WH-ADAPTER-ADDS"
    arms["RA_s0"] = arms["RA_s1"] = ra_same
    out = decide_with(world, arms, F={v: 0.02 for v in M.CTX_VARS})
    assert out["weather"]["outcome"] == "WH-DETECTS + WH-INPUT-FEATURES" and "input features" in out["weather"]["wording"]


def test_outcome_weather_not_detected_names_the_next_arm_and_the_feat_branch(world):
    low = planted(world, [0.3] * 5, 7)
    arms = {"H_s0": low, "H_s1": low, "K4p_s0": low, "K4p_s1": low, "FEAT_s0": planted(world, [0.97] * 5, 8), "FEAT_s1": planted(world, [0.97] * 5, 9)}
    out = decide_with(world, arms)
    assert out["weather"]["outcome"] == "WH-NOT-DETECTED" and "a-W" in out["weather"]["next"] and "form (b)" in out["weather"]["next"]
    arms["FEAT_s0"] = arms["FEAT_s1"] = low
    assert "representation lever" in decide_with(world, arms)["weather"]["next"]


def test_a_detecting_k4p_or_a_failed_instrument_quotes_no_number(world):
    arms = {"H_s0": planted(world, [0.95] * 5, 1), "H_s1": planted(world, [0.95] * 5, 4), "K4p_s0": planted(world, [0.95] * 5, 5), "K4p_s1": planted(world, [0.1] * 5, 6)}
    assert decide_with(world, arms)["weather"]["outcome"] == "WH-NOT-DETECTED"          # K4p detects at s0: the variable is not DETECTS
    out = decide_with(world, arms, ins={**PASS, "T-ISO-d": "fail"})
    assert out["__all__"] == "INSTRUMENT-FAIL" and "T-ISO-d" in out["failed"]
    out = decide_with(world, arms, ins={k: v for k, v in PASS.items() if k != "K-ID"})
    assert out["__all__"] == "INSTRUMENT-INCOMPLETE"


def test_outcome_other_variables_replace_gain_or_keep_shipped(world):
    h = planted(world, [0.95] * 5, 1)
    base = {"H_s0": h, "H_s1": planted(world, [0.95] * 5, 4), "K4p_s0": planted(world, [0.1] * 5, 5), "K4p_s1": planted(world, [0.1] * 5, 6)}
    win = decide_with(world, {**base, "SH_s0": planted(world, [0.5] * 5, 2), "SH_s1": planted(world, [0.5] * 5, 3)}, F={v: 0.01 for v in M.CTX_VARS})
    assert win["illumination"]["outcome"] == "POOL-GAIN(illumination)"
    tie = decide_with(world, {**base, "SH_s0": h, "SH_s1": base["H_s1"]}, F={v: 0.01 for v in M.CTX_VARS})
    assert tie["illumination"]["outcome"] == "POOL-REPLACES(illumination)"
    worse = decide_with(world, {**base, "SH_s0": planted(world, [0.999] * 5, 2), "SH_s1": planted(world, [0.999] * 5, 3)}, F={v: 0.01 for v in M.CTX_VARS})
    assert worse["period_of_day"]["outcome"] == "SHIPPED-KEPT(period_of_day)"


def test_the_output_map_is_derived_from_the_rules_and_never_chosen(world):
    out = {"weather": {"outcome": "WH-DETECTS + WH-INPUT-FEATURES"}, "country": {"outcome": "POOL-GAIN(country)"}, "surface": {"outcome": "SHIPPED-KEPT(surface)"},
           "illumination": {"outcome": "POOL-REPLACES(illumination)"}, "period_of_day": {"outcome": "NOT-YET"}}
    m = RD.output_map(out)
    assert m["weather"]["source"] == "ctx_pool_head.weather" and m["surface"]["source"].startswith("ctx_head") and m["period_of_day"]["source"] == "undecided" and m["country"]["source"] == "ctx_pool_head.country"


# ------------------------------------------------------------------------------------------------ the token
def test_every_arm_names_the_lever_and_k4p_has_no_fit_flags():
    for arm, s in C.ARMS.items():
        toks = C.render(arm, "R")
        if s["kind"] == "fit":
            assert toks[toks.index("--ctx-pool") + 1] == "on" and "--e-grid" in toks and "--fit-seed" in toks and "--init" in toks
        else:
            assert "--ckpt" in toks
    assert len(C.ARMS) == 13 and sorted({s["prio"] for s in C.ARMS.values()}) == list(range(1, 10))


def test_the_fit_refuses_without_a_token_and_a_ctx_pool_off_argv_does_not_parse(tmp_path):
    with pytest.raises(SystemExit):
        wh_fit.main(["fit", "--arm", "H_s0", "--init", "S1_s0", "--ctx-pool", "on", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        wh_fit.build_parser().parse_args(["fit", "--arm", "x", "--init", "S1_s0", "--ctx-pool", "off", "--out", "o"])
    with pytest.raises(SystemExit):
        wh_fit.build_parser().parse_args(["fit", "--arm", "x", "--init", "S1_s0", "--out", "o"])                    # a default that disables the lever must be named


def test_a_token_is_written_only_when_every_check_and_every_regression_arm_pass(tmp_path):
    ok = {c: {"pass": True, "regression_arm": {"pass": True}} for c in C.REQUIRED}
    C.write_token(str(tmp_path / "t.json"), {"arm": "H_s0"}, ok)
    bad = json.loads(json.dumps(ok))
    bad["G-LIVE"]["regression_arm"]["pass"] = False                                      # a blind check
    with pytest.raises(ValueError):
        C.write_token(str(tmp_path / "t2.json"), {"arm": "H_s0"}, bad)
    bad2 = json.loads(json.dumps(ok))
    bad2["G-EVAL"]["pass"] = False
    with pytest.raises(ValueError):
        C.write_token(str(tmp_path / "t3.json"), {"arm": "H_s0"}, bad2)


def test_a_hand_edited_token_or_a_changed_argv_is_refused(tmp_path):
    toks = C.render("H_s0", str(tmp_path))
    gs = tmp_path / "gate" / "gsuite_result.json"
    os.makedirs(gs.parent, exist_ok=True)
    gs.write_text("{}")
    b = C.binding("H_s0", toks, str(gs))
    ok = {c: {"pass": True, "regression_arm": {"pass": True}} for c in C.REQUIRED}
    tp = C.token_path(str(tmp_path), "H_s0")
    C.write_token(tp, b, ok)
    assert C.verify_token(tp, toks, "H_s0", str(gs))[1] is None
    assert "argv_sha256" in C.verify_token(tp, toks + ["--x", "1"], "H_s0", str(gs))[1]
    t = json.load(open(tp))
    t["checks"]["G-LIVE"]["pass"] = False
    json.dump(t, open(tp, "w"))
    assert "edited" in C.verify_token(tp, toks, "H_s0", str(gs))[1]
    assert C.verify_token("", toks, "H_s0")[1] == "no --token given"


def test_gate_subtracts_regressions_against_the_base_results():
    base = {"a::t1": "pass", "a::t2": "pass", "a::t3": "fail"}
    assert C.suite_compare(base, {"a::t1": "pass", "a::t2": "pass", "a::t3": "fail", "b::new": "pass"})["n_regressions"] == 0
    assert C.suite_compare(base, {"a::t1": "pass", "a::t2": "fail", "a::t3": "fail"})["n_regressions"] == 1


def test_the_lock_is_exclusive_and_only_its_owner_releases_it(tmp_path, monkeypatch):
    lock = str(tmp_path / "gpu.lock")
    monkeypatch.setattr(W, "GPU_LOCK", lock)
    monkeypatch.setattr(W, "gpu_used_mib", lambda: 0.0)
    assert W.gpu_lock_acquire("jobA", 0, log=lambda *a: None)
    assert not W.gpu_lock_acquire("jobB", 0, log=lambda *a: None)                       # never forced, never overwritten
    assert not W.gpu_lock_release("jobB") and os.path.exists(lock)
    assert W.gpu_lock_release("jobA") and not os.path.exists(lock)
    monkeypatch.setattr(W, "gpu_used_mib", lambda: 5000.0)
    assert not W.gpu_lock_acquire("jobC", 0, log=lambda *a: None) and not os.path.exists(lock)       # a busy GPU releases the lock it just took


def test_prefetch_yields_the_same_batches_in_the_same_order(cache):
    root, names = cache
    rs = W.train_rowset(fake_table())
    st = W.StreamSource(root, names, rs)
    chunks = [np.arange(i, min(i + 7, len(rs["ci"]))) for i in range(0, len(rs["ci"]), 7)]
    got = list(W.prefetched(st, chunks))
    assert [c.tolist() for c, _ in got] == [c.tolist() for c in chunks]
    assert all(torch.equal(f.view(torch.uint8), st.gather(c).view(torch.uint8)) for c, f in got)


def test_an_instrument_fail_withholds_every_number(tmp_path, world):
    t0 = tmp_path / "t0"
    os.makedirs(t0)
    lg = planted(world, [0.9] * 5)
    for s in (0, 1):
        np.savez(t0 / f"eval_WH-H_s{s}_real.npz", ctx_pool_logits=lg, win_ci=world.ci, win_t=world.t)
    rep = RD.read_all(world, str(t0), "", {"T-ISO-d": "fail"}, smoke=True)
    assert rep["outcomes"]["__all__"] == "INSTRUMENT-FAIL" and rep["arms"] == {"withheld": "INSTRUMENT-FAIL"} and rep["ctx_output_map"] == {"__all__": "INSTRUMENT-FAIL"}


def test_no_banned_words_in_the_study_docs_and_scripts_headers():
    banned = ("void", "superseded", "retracted", "formerly", "correction", "withdrawn", "no longer", "closed loop")
    for f in ("wh_lib.py", "wh_fit.py", "wh_read.py", "wh_run.py", "wh_iso.py", "wh_check.py", "wh_chain.py"):
        head = open(os.path.join(HERE, f), encoding="utf-8").read().split('"""')[1].lower()
        assert not [w for w in banned if w in head], f


# ------------------------------------------------------------------------------------------------ T-ISO (b)/(T-VIS) probe inside the recording wrapper (the 2026-11-17 RecursionError)
def _iso_rig(monkeypatch, iso_mod):
    """a pooled-head model, a fake T0 evaluator that calls ``forward_infer`` on 3 batches, and probe stand-ins that call ``forward_infer`` twice like the real T-ISO (b) -> (model, calls, evaluator-run fn)"""
    import importlib.util
    import types
    calls = {"fwd": 0, "iso_b": 0, "vis": 0}

    def fake_forward(self, batch):
        calls["fwd"] += 1
        return {"traj60": torch.full((2, 3), float(batch)), "ctx_pool_logits": torch.zeros(2, 1)}

    def fake_iso_b(model, batch):
        calls["iso_b"] += 1
        model.forward_infer(batch)
        model.forward_infer(batch)
        return []

    def fake_vis(model, batch):
        calls["vis"] += 1
        return {"n": 1}
    model = tiny_model()
    monkeypatch.setattr(M.V8FMini, "forward_infer", fake_forward)
    monkeypatch.setattr(M, "ctx_pool_isolation_problems", fake_iso_b)
    monkeypatch.setattr(M, "assert_ctx_pool_vision_only", fake_vis)
    ev = types.SimpleNamespace(main=lambda av: [model.forward_infer(b) for b in range(3)])
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a, **k: types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda m: None)))
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda spec: ev)
    monkeypatch.setattr(iso_mod.RUN, "t0_argv", lambda *a, **k: ["py", "ev", "x"])
    return model, calls


def test_record_run_probe_runs_on_the_unwrapped_method_without_recursion_or_extra_records(monkeypatch):
    import wh_iso
    model, calls = _iso_rig(monkeypatch, wh_iso)
    fake = M.V8FMini.forward_infer
    probe = {}
    rec = wh_iso._record_run("ck", "t", "on", "out", probe)
    assert calls["iso_b"] == 1 and calls["vis"] == 1 and set(probe) == {"T-ISO-b", "T-VIS"}
    assert len(rec) == 3 and [float(r["traj60"][0, 0]) for r in rec] == [0.0, 1.0, 2.0]       # the probe's 2 extra forward_infer calls were NOT recorded
    assert calls["fwd"] == 3 + 2                                                                 # 3 real batches + the probe's two calls
    assert M.V8FMini.forward_infer is fake                                                       # the wrapper is removed again


def test_mutation_the_probe_called_on_the_wrapped_method_reads_red(monkeypatch):
    import textwrap
    import wh_iso
    model, calls = _iso_rig(monkeypatch, wh_iso)
    src = textwrap.dedent(inspect.getsource(wh_iso._record_run))
    line = "M.V8FMini.forward_infer = orig                       # the probe calls forward_infer itself"
    assert line in src
    mut_src = src.replace(line, "pass                                                  # MUTANT: no unwrap; the probe calls forward_infer itself")
    ns = dict(wh_iso.__dict__)
    exec(mut_src, ns)
    with pytest.raises(RecursionError):
        ns["_record_run"]("ck", "t", "on", "out", {})
