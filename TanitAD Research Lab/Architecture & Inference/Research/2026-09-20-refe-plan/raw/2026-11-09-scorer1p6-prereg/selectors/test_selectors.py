"""Tests of fit_selectors / read_selectors / selcore. CPU only.

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_selectors.py --basetemp F:/tanitad_scratch/p6sel/pt

Synthetic packs only (packtools.make_synthetic_pack, the exact fit_pack.npz format); no navtest truth is read. Every control that must read a known value is paired with a
deliberate-regression arm that must make it fail.
"""
import json
import os
import pickle
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("SEL_THREADS", "2")
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selcore as C  # noqa: E402
import fit_selectors as FS  # noqa: E402
import packtools as PT  # noqa: E402
import read_selectors as RS  # noqa: E402
R = RS.R

FAST = {"mlp_epochs": 1, "pg_epochs": 1, "gbdt_pdms_iter": 8, "gbdt_heads_iter": 6}
NAVTEST = os.path.join(HERE, "navtest_tokens_e3.txt")


def args(pack, out, **kw):
    d = dict(pack=str(pack), out=str(out), threads=2, only=None, navtest_tokens=[NAVTEST], min_free_gb=6.0, no_ram_wait=True, prereg=None, force=False, cfg=json.dumps(FAST),
             smoke_standin=False)
    d.update(kw)
    return type("A", (), d)()


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("pack") / "fit_pack.npz"
    PT.make_synthetic_pack(p, n_logs=24, sets_per_log=16, seed=11)
    return p


# ------------------------------------------------------------------------------------------------ split
def test_split_determinism_and_partition(pack_path):
    P = C.Pack(pack_path)
    a, b = C.make_split(P.log), C.make_split(P.log)
    assert C.canon(a) == C.canon(b)
    iA, iB = C.split_rows(P.log, a)
    assert len(iA) + len(iB) == P.n and not set(iA) & set(iB)
    assert not set(P.log[iA]) & set(P.log[iB]), "a log is on both sides"
    assert 0.6 <= len(iA) / P.n <= 0.85
    assert C.canon(C.make_split(P.log, seed=1)) != C.canon(a), "the seed matters"


def test_split_written_with_sha_and_refuses_when_different(pack_path, tmp_path):
    P = C.Pack(pack_path)
    out = tmp_path / "o"
    out.mkdir()
    sp, sha, fresh = FS.write_split(str(out), P)
    assert fresh and open(out / "split.sha256").read().split()[0] == sha == C.sha256_file(out / "split.json")
    assert FS.write_split(str(out), P)[2] is False                      # identical: verified, not rewritten
    d = json.load(open(out / "split.json"))                             # deliberate regression: a tampered split must be refused
    d["logs_A"], d["logs_B"] = d["logs_B"], d["logs_A"]
    open(out / "split.json", "w").write(json.dumps(d))
    with pytest.raises(FS.Refused):
        FS.write_split(str(out), P)
    out2 = tmp_path / "o2"
    out2.mkdir()
    FS.write_split(str(out2), P)
    P2 = C.Pack(pack_path)
    P2.log = np.array([x.replace("log00", "logZZ") for x in P2.log])  # a different log structure than the stored split
    with pytest.raises(FS.Refused):
        FS.write_split(str(out2), P2)


# ------------------------------------------------------------------------------------------------ leakage
def _poisoned_copy(src, dst, logs_b):
    Z = dict(np.load(src, allow_pickle=False))
    rng = np.random.default_rng(5)
    m = np.isin(Z["log"], logs_b)
    Z["pdm"] = Z["pdm"].copy()
    Z["pdm"][m] = rng.random(Z["pdm"][m].shape).astype(np.float32)
    Z["logits"] = Z["logits"].copy()
    Z["logits"][m] = rng.normal(0, 3, Z["logits"][m].shape).astype(np.float32)
    Z["traj_xy"] = Z["traj_xy"].copy()
    Z["traj_xy"][m] += rng.normal(0, 2, Z["traj_xy"][m].shape).astype(np.float32)
    np.savez(dst, **Z)


def _params_hashes(out):
    return {v: json.load(open(os.path.join(out, "variants", v, "meta.json")))["params_sha256"] for v in C.VARIANTS}


def test_no_leakage_fit_reads_fit_a_only_and_a_mutation_reading_fit_b_turns_red(pack_path, tmp_path, monkeypatch):
    P = C.Pack(pack_path)
    sp = C.make_split(P.log)
    pois = tmp_path / "poisoned.npz"
    _poisoned_copy(pack_path, pois, sp["logs_B"])
    o1, o2 = tmp_path / "real", tmp_path / "pois"
    FS.run(args(pack_path, o1), log=lambda *_: None)
    FS.run(args(pois, o2), log=lambda *_: None)
    h1, h2 = _params_hashes(o1), _params_hashes(o2)
    assert h1 == h2, f"a fit depended on FIT-B rows: {[v for v in h1 if h1[v] != h2[v]]}"
    # deliberate regression: let the fit read EVERY row (a FIT-B read); the same comparison must now fail
    orig = C.split_rows
    allrows = lambda log, split: (np.arange(len(log)), orig(log, split)[1])   # noqa: E731
    monkeypatch.setattr(FS.C, "split_rows", allrows)
    o3, o4 = tmp_path / "leak_real", tmp_path / "leak_pois"
    FS.run(args(pack_path, o3), log=lambda *_: None)
    FS.run(args(pois, o4), log=lambda *_: None)
    h3, h4 = _params_hashes(o3), _params_hashes(o4)
    assert h3 != h4, "the leak check is blind: reading FIT-B did not change any fitted parameter"
    assert {v for v in h3 if h3[v] != h4[v]} >= {"R1", "C_platt", "C_isotonic", "S_gbdt_pdms", "P_pg"}


# ------------------------------------------------------------------------------------------------ run / resume / refusal
def test_end_to_end_resume_artifacts_and_nominee_files(pack_path, tmp_path):
    out = tmp_path / "e2e"
    nom = FS.run(args(pack_path, out), log=lambda *_: None)
    for v in C.VARIANTS:
        d = out / "variants" / v
        assert (d / "meta.json").exists() and (d / "params.pkl").exists() and (d / "pick_fit_b.npy").exists()
        m = json.load(open(d / "meta.json"))
        assert m["done"] and C.sha256_file(d / "params.pkl") == m["params_sha256"] and len(np.load(d / "pick_fit_b.npy")) == m["n_fit_b"]
    for f in ("split.json", "split.sha256", "nominees.json", "nominees.sha256", "FIT_REPORT.md", "R_candidates_fit_a.json"):
        assert (out / f).exists(), f
    assert len(json.load(open(out / "R_candidates_fit_a.json"))) == 401
    assert open(out / "nominees.sha256").read().split()[0] == C.sha256_file(out / "nominees.json")
    assert len(nom["nominees"]) >= 1 and all(n["ci_lo"] > 0 for n in nom["nominees"])
    mt = os.path.getmtime(out / "variants" / "P_pg" / "meta.json")
    FS.run(args(pack_path, out), log=lambda *_: None)                          # resume: everything skipped
    assert os.path.getmtime(out / "variants" / "P_pg" / "meta.json") == mt
    with pytest.raises(FS.Refused):                                           # a different config over finished fits is refused
        FS.run(args(pack_path, out, cfg=json.dumps({**FAST, "pg_epochs": 2}), only="P_pg"), log=lambda *_: None)


def test_navtest_tokens_refuse_and_missing_list_fails_closed(pack_path, tmp_path):
    Z = dict(np.load(pack_path, allow_pickle=False))
    nav = [ln.strip() for ln in open(NAVTEST) if ln.strip()]
    tok = Z["token"].copy().astype(object)
    tok[3] = nav[17]
    Z["token"] = np.array(tok.astype(str))
    bad = tmp_path / "bad.npz"
    np.savez(bad, **Z)
    with pytest.raises(FS.Refused, match="navtest"):
        FS.run(args(bad, tmp_path / "o"), log=lambda *_: None)
    assert not (tmp_path / "o" / "variants").exists(), "a fit started before the refusal"
    with pytest.raises(FS.Refused):
        FS.run(args(pack_path, tmp_path / "o2", navtest_tokens=[str(tmp_path / "nope.txt")]), log=lambda *_: None)
    assert FS.main(["--pack", str(bad), "--out", str(tmp_path / "o3"), "--navtest-tokens", NAVTEST, "--no-ram-wait"]) == 2


def test_prereg_sha_checked(tmp_path):
    card = os.path.join(HERE, "..", "PREREG_P6_SELECTORS.md")
    if os.path.exists(card):
        assert FS.check_prereg(card)["checked"]
        assert RS.PREREG_SHA == FS.PREREG_SHA
        bad = tmp_path / "PREREG_P6_SELECTORS.md"
        shutil.copy(card, bad)
        open(bad, "ab").write(b"x")
        with pytest.raises(FS.Refused):
            FS.check_prereg(str(bad))


# ------------------------------------------------------------------------------------------------ nomination, bars at exact boundaries
def _meta(out, split_sha, v, val, lo):
    d = os.path.join(out, "variants", v)
    os.makedirs(d)
    json.dump(dict(variant=v, done=True, split_sha256=split_sha, fit_b_mean_pdms_x100=val, p6_rule_fit_b_mean_pdms_x100=80.0, delta_vs_p6_x100=val - 80, ci_lo=lo, ci_hi=lo + 1,
                   n_pick_differs_from_p6=1, fit_secs=0, params_file="x", params_sha256="s", v0_source="measured"), open(os.path.join(d, "meta.json"), "w"))


def test_nomination_rule_exact_boundaries(tmp_path):
    out = str(tmp_path)
    _meta(out, "S", "R1", 81.0, 1e-12)            # CI-lo just above 0: nominated
    _meta(out, "S", "R2", 90.0, 0.0)              # CI-lo exactly 0: NOT nominated, however high its mean
    _meta(out, "S", "C_platt", 82.0, 0.5)
    _meta(out, "S", "C_isotonic", 83.0, 0.6)
    _meta(out, "S", "C_platt_binned", 84.0, 0.7)  # three eligible: only the best two by FIT-B mean
    _meta(out, "S", "S_gbdt_pdms", 85.0, -1e-12)  # just below 0: none for family S
    nom = FS.nominate(out, "S", log=lambda *_: None)
    f = nom["families"]
    assert f["R"]["nominated"] == ["R1"]
    assert f["C"]["nominated"] == ["C_platt_binned", "C_isotonic"]
    assert f["S"]["nominated"] == [] and "no variant" in f["S"]["note"]
    assert f["P"]["nominated"] == [] and f["P"]["n_variants_fitted"] == 0
    assert [n["family_rank"] for n in nom["nominees"] if n["family"] == "C"] == [1, 2]
    _meta(out, "OTHER", "P_pg", 99.0, 5.0)        # a meta fitted under another split is ignored
    assert FS.nominate(out, "S", log=lambda *_: None)["families"]["P"]["nominated"] == []


def test_bar_helpers_at_exact_boundaries():
    assert RS.p_clears(1e-12) and not RS.p_clears(0.0) and not RS.p_clears(-1e-12) and not RS.p_clears(None)
    assert RS.g1_all({"TTC": -0.5, "NC": -0.5, "DAC": -0.5}) == {"TTC": True, "NC": True, "DAC": True}
    assert RS.g1_all({"TTC": -0.5000001, "NC": 0.0, "DAC": 0.3})["TTC"] is False and RS.g1_all({"TTC": -0.4999, "NC": -0.51, "DAC": 0})["NC"] is False
    assert RS.g2_ok(0.0999999) and not RS.g2_ok(0.10) and not RS.g2_ok(0.1000001)
    assert R.goal_status(94.55, 94.6) == "GOAL MET" and R.goal_status(94.5, 94.55) == "GOAL NOT EXCLUDED" and R.goal_status(94.5, 94.54) == "BELOW GOAL"


def test_outcomes_s1_s2_s3_exact():
    ok = dict(clears_P=True, guards_ok=True, lost_e2=False, lost_e3=False)
    assert RS.decide_s([ok])["primary"] == "S1"
    assert RS.decide_s([dict(ok, guards_ok=False)])["primary"] == "S2"             # clears P but fails a guard: not S1
    assert RS.decide_s([dict(ok, clears_P=False)])["primary"] == "S2"
    assert RS.decide_s([])["primary"] == "S2"
    assert RS.decide_s([dict(ok, clears_P=False, lost_e3=True)])["primary"] == "S3"
    assert RS.decide_s([dict(ok, clears_P=False, lost_e2=True)])["primary"] == "S3"
    both = RS.decide_s([ok, dict(ok, clears_P=False, lost_e2=True)])
    assert both["matches"] == ["S1", "S3"] and both["primary"] == "S1"


def test_shuffled_logits_selector_is_not_nominated(pack_path, tmp_path):
    P = C.Pack(pack_path)
    sp = C.make_split(P.log)
    iA, iB = C.split_rows(P.log, sp)
    ctxB = C.Ctx(P.L, P.X8, P.v0).take(iB)
    PB, ul_li = P.P[iB], C.log_boot(P.log[iB])
    ul, li, draws = ul_li
    ref = C.p6_pick(ctxB.L)
    # the informative rule: positive control; shuffled logits across the 64 proposals of every set: must lose
    v_ok, d_ok, lo_ok, _ = FS.evaluate_picks(PB, np.argmax(C.rule_scores(ctxB.L, (1, 1, 2, 5, 0)), 1), ref, li, len(ul), draws)
    rng = np.random.default_rng(0)
    Ls = np.stack([rng.permutation(l, axis=0) for l in ctxB.L])
    pk = np.argmax(C.rule_scores(Ls, C.P6_RULE), 1)
    v, d, lo, hi = FS.evaluate_picks(PB, pk, ref, li, len(ul), draws)
    assert d < 0 and lo <= 0, (d, lo, hi)
    out = str(tmp_path)
    _meta(out, "S", "S_mlp_heads", v, lo)
    assert FS.nominate(out, "S", log=lambda *_: None)["nominees"] == []


# ------------------------------------------------------------------------------------------------ core
def test_core_matches_the_programme_code_and_ties_are_handled():
    pytest.importorskip("selc6_analyze")
    sc = R.sc
    rng = np.random.default_rng(0)
    L = rng.normal(0, 2, (50, 64, 6)).astype(np.float32)
    L[3, :, :] = 0.0                                                      # a fully tied set: first index
    assert C.p6_scores(L).tobytes() == sc.agg_torch(L, lane=False).tobytes()
    assert np.array_equal(C.p6_pick(L), R.pick_logits(L)) and C.p6_pick(L)[3] == 0
    assert np.array_equal(C.rule_scores(L, C.P6_RULE), C.p6_scores(L))
    log = np.array([f"l{i % 7}" for i in range(60)])
    ul1, li1, d1 = C.log_boot(log, 200)
    ul2, li2, d2 = sc.log_boot_idx(log, 200, 20260927)
    assert np.array_equal(ul1, ul2) and np.array_equal(li1, li2) and np.array_equal(d1, d2)
    # tie-correct ranks: an all-tied row is 0.5 everywhere, not 0..1 by arbitrary order
    a = np.zeros((2, 64))
    a[1, :32] = 1
    r = C.rank01(a)
    assert np.allclose(r[0], 0.5) and r[1, 0] == r[1, 31] > r[1, 32] == r[1, 63]


def test_bonferroni_interval_widens_and_percentiles():
    assert C.pct_pair(1) == (2.5, 97.5) and C.pct_pair(5) == (0.5, 99.5) and C.pct_pair(0) == (2.5, 97.5)
    r = np.random.default_rng(1).normal(0, 1, 100000)
    w = [np.subtract(*C.interval(r, K)[::-1]) for K in (1, 2, 5, 20)]
    assert all(w[i] < w[i + 1] for i in range(3))


def test_v0_source_mismatch_refused(pack_path):
    P = C.Pack(pack_path)
    A = C.FitData(C.Ctx(P.L, P.X8, P.v0), P.P, P.S, P.log)
    prm = C.fit_variants(A, ["C_platt_binned"])["C_platt_binned"]
    assert prm["v0_source"] == "measured"
    with pytest.raises(ValueError, match="v0 source"):
        C.apply_params(prm, C.Ctx(P.L, P.X8, None))
    assert C.Ctx(P.L, P.X8, None).v0_source == "proxy_median_first_chord"


def test_pack_refuses_malformed(tmp_path, pack_path):
    Z = dict(np.load(pack_path, allow_pickle=False))
    Z["traj_xy"] = Z["traj_xy"][:, :, :6]
    np.savez(tmp_path / "bad.npz", **Z)
    with pytest.raises(ValueError, match="8 poses"):
        C.Pack(tmp_path / "bad.npz")
    Z = dict(np.load(pack_path, allow_pickle=False))
    Z.pop("pdm")
    np.savez(tmp_path / "bad2.npz", **Z)
    with pytest.raises(ValueError, match="lacks"):
        C.Pack(tmp_path / "bad2.npz")


# ------------------------------------------------------------------------------------------------ the read
def _read_world(tmp_path, n_logs=30, per=18, seed=77):
    rp = tmp_path / "read_pack.npz"
    PT.make_synthetic_pack(rp, n_logs=n_logs, sets_per_log=per, seed=seed, token_prefix="rd")
    Z = np.load(rp, allow_pickle=False)
    tok = [str(t) for t in Z["token"]]
    log = [str(x) for x in Z["log"]]
    X8 = np.concatenate([Z["traj_xy"], Z["yaw"][..., None]], -1).astype(np.float32)
    sub = Z["pdm"][..., :6].astype(np.float64)
    pdms = sub[..., 0] * sub[..., 1] * (5 * sub[..., 2] + 5 * sub[..., 3] + 2 * sub[..., 4]) / 12
    L = Z["logits"].astype(np.float32)
    best = pdms.argmax(1)
    human8 = X8[np.arange(len(tok)), best].astype(np.float64)
    v0 = Z["v0"].astype(np.float64)
    world = R.World(tok, log, human8, v0, np.zeros(len(tok)), None, ["fp"] * len(tok))
    pk = R.pick_logits(L)
    r = np.arange(len(tok))
    rows = {t: (True, float(pdms[i, pk[i]]), sub[i, pk[i]]) for i, t in enumerate(tok)}
    table = dict(token=np.array(tok), pdms=pdms, sub=sub, valid=np.ones_like(pdms, bool), red=None)
    return world, L, X8, pk, rows, table


def _nominees(pack_path, tmp_path, only=None):
    out = tmp_path / "fitnom"
    FS.run(args(pack_path, out, only=only), log=lambda *_: None)
    return str(out / "nominees.json")


@pytest.fixture(scope="module")
def nominees_file(pack_path, tmp_path_factory):
    d = tmp_path_factory.mktemp("nom")
    return _nominees(pack_path, d)


def _inp(tmp_path, with_full=True, n_e2=120):
    world, L, X8, pk, rows, table = _read_world(tmp_path)
    e2 = list(world.token[:n_e2])
    e2_idx = np.arange(n_e2)
    e2t = dict(token=table["token"][:n_e2], pdms=table["pdms"][:n_e2], sub=table["sub"][:n_e2], valid=table["valid"][:n_e2], red=None)
    return RS.ReadInputs(world, L, X8, pk, rows, e2, e2t, table if with_full else None, {}, {"synthetic": True}, tactical=False)


def test_read_e2_e3_ledger_control_and_adjusted_interval(nominees_file, tmp_path):
    inp = _inp(tmp_path)
    doc, params = RS.load_nominees(nominees_file, False)
    led = C.Ledger(tmp_path / "ledger.json")
    assert led.K == 0
    r2 = RS.run_read(inp, doc, params, "e2", led, out_dir=str(tmp_path))
    assert r2["status"] == "READ", r2.get("why")
    n_nom = len(doc["nominees"])
    assert r2["ledger"]["K"] == n_nom == led.K and n_nom >= 1
    assert r2["control_p6_rule_as_candidate"]["all_endpoint_differences_exactly_zero"] and r2["control_p6_rule_as_candidate"]["n_pick_changed"] == 0
    assert "PAUSED" in r2["strategic"] and r2["tier"] and r2["evidence_class"].startswith("MEASURED")
    r3 = RS.run_read(inp, doc, params, "e3", led, e2_result=r2, out_dir=str(tmp_path))
    assert r3["status"] == "READ", r3.get("why")
    n_e3 = len(RS.select_e3_nominees(doc))
    assert led.K == n_nom + n_e3 and r3["ledger"]["K"] == led.K
    fams = [e["family"] for e in r3["nominees"]]
    assert len(fams) == len(set(fams)), "more than one nominee of a family on E3"
    for e in r3["nominees"]:
        P = e["P"]
        assert P["K"] == led.K and P["lo_adj"] <= P["lo95"] and P["hi_adj"] >= P["hi95"]
        assert P["lo_adj"] < P["lo95"], "the adjusted interval did not widen at K>1"
        assert P["clears"] == (P["lo_adj"] > 0)
        assert "family_table" in e and e["verdict"] in ("ADOPT", "PDMS GAIN WITH COST", "NO ADOPTION")
    assert r3["outcome"]["primary"] in ("S1", "S2", "S3")
    # the same read repeated is the same read: K does not move
    k = led.K
    RS.run_read(inp, doc, params, "e3", led, e2_result=r2, out_dir=str(tmp_path))
    assert C.Ledger(tmp_path / "ledger.json").K == k
    md = RS.to_markdown(r3)
    assert "STRATEGIC: PAUSED" in md and "T1-class open loop" in md and f"K = {k}" in md


def test_p6_rule_candidate_is_exactly_zero_and_a_regression_is_not(nominees_file, tmp_path):
    inp = _inp(tmp_path)
    p6 = C.p6_pick(inp.p6_logits)
    pool = RS.build_pool(inp, p6, True)
    ix = np.arange(len(inp.world.token))
    e = RS.endpoints(inp, pool, ix, RS.picks_of(dict(kind="rule", cand=list(C.P6_RULE)), inp), p6, 1)
    vals = [e["P"][k] for k in ("point", "lo95", "hi95", "lo_adj", "hi_adj")] + [e["G1"][h][k] for h in RS.G1_HEADS for k in ("point", "lo", "hi")] + \
           [e["G2_overspeed"][k] for k in ("point", "lo", "hi")]
    assert all(v == 0.0 for v in vals) and e["n_pick_changed_vs_p6"] == 0
    bad = RS.picks_of(dict(kind="rule", cand=[1.0, 1.0, 5.0, 5.0, 0.0]), inp)           # a different rule: must NOT be all zero
    e2 = RS.endpoints(inp, pool, ix, bad, p6, 1)
    assert e2["n_pick_changed_vs_p6"] > 0 and e2["P"]["point"] != 0.0


def test_shuffled_selector_does_not_clear_P_on_the_read(nominees_file, tmp_path):
    inp = _inp(tmp_path)
    p6 = C.p6_pick(inp.p6_logits)
    pool = RS.build_pool(inp, p6, True)
    rng = np.random.default_rng(3)
    Ls = np.stack([rng.permutation(l, axis=0) for l in inp.p6_logits])
    pk = C.p6_pick(Ls)
    e = RS.endpoints(inp, pool, np.arange(len(p6)), pk, p6, 3)
    assert not e["P"]["clears"] and e["P"]["point"] < 0


def test_incomplete_without_full_table_then_changed_csv_completes(nominees_file, tmp_path):
    inp = _inp(tmp_path, with_full=False)
    doc, params = RS.load_nominees(nominees_file, False)
    led = C.Ledger(tmp_path / "l.json")
    r = RS.run_read(inp, doc, params, "e3", led, out_dir=str(tmp_path / "seams"))
    assert r["status"] == "INCOMPLETE" and r["seams"] and led.K == 0, "an incomplete read must not be counted"
    W = inp.world
    # score the changed picks with the truth (stand-in for the harness) in the harness csv layout, then read again
    world, L, X8, pk, rows, table = _read_world(tmp_path)
    for nid, s in r["seams"].items():
        picks = RS.picks_of(params[nid], inp)
        need = np.where(~RS.build_pool(inp, C.p6_pick(inp.p6_logits), False).known[np.arange(len(picks)), picks])[0]
        import csv
        p = tmp_path / f"{nid}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["token", "valid", "score"] + list(R.CSV_SUB))
            for i in need:
                w.writerow([W.token[i], "True", repr(float(table["pdms"][i, picks[i]]))] + [repr(float(x)) for x in table["sub"][i, picks[i]]])
        inp.changed[nid] = R.read_harness_csv(p)
    r2 = RS.run_read(inp, doc, params, "e3", led, out_dir=str(tmp_path / "seams"))
    assert r2["status"] == "READ", r2.get("why")
    ref = RS.run_read(_inp(tmp_path), doc, params, "e3", C.Ledger(tmp_path / "l2.json"), out_dir=str(tmp_path))
    assert [e["P"]["point"] for e in r2["nominees"]] == [e["P"]["point"] for e in ref["nominees"]], "csv route and full-table route disagree"


def test_read_stops_when_a_gate_fails(nominees_file, tmp_path):
    inp = _inp(tmp_path)
    inp.p6_pick_banked = inp.p6_pick_banked.copy()
    inp.p6_pick_banked[0] = (inp.p6_pick_banked[0] + 1) % 64
    doc, params = RS.load_nominees(nominees_file, False)
    r = RS.run_read(inp, doc, params, "e3", C.Ledger(tmp_path / "l.json"))
    assert r["status"] == "READ STOPPED" and "G0-PICK" in r["why"] and "nominees" not in r


def test_smoke_and_real_nominees_never_mix(nominees_file):
    with pytest.raises(SystemExit, match="never mix"):
        RS.load_nominees(nominees_file, True)


def test_no_nominee_family_reads_nothing(tmp_path):
    inp = _inp(tmp_path)
    led = C.Ledger(tmp_path / "l.json")
    r = RS.run_read(inp, {"nominees": [], "families": {}}, {}, "e3", led)
    assert r["status"] == "READ" and r["verdict"]["primary"] == "S2" and led.K == 0


def test_params_tamper_refused(nominees_file, tmp_path):
    d = os.path.dirname(nominees_file)
    doc = json.load(open(nominees_file))
    pf = os.path.join(d, doc["nominees"][0]["params_file"])
    blob = open(pf, "rb").read()
    open(pf, "wb").write(blob + b"x")
    try:
        with pytest.raises(SystemExit, match="sha256"):
            RS.load_nominees(nominees_file, False)
    finally:
        open(pf, "wb").write(blob)
