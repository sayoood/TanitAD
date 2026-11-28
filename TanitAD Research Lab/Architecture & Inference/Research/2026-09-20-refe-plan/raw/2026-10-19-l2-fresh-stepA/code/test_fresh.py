"""Tests of the L2-FRESH-STEPA tooling.  CPU only (CUDA_VISIBLE_DEVICES=-1).  Run:  python -m pytest test_fresh.py -q
Each guard has a RED arm: a mutated copy of the thing must FAIL the same check.  Nothing here reads a fresh-carve label or score (none exists); the instrument-validity test runs on Step A's OWN banked data."""
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fresh_carve as FC  # noqa: E402
import fit_final_heads as FH  # noqa: E402
import score_fresh as SF  # noqa: E402
import analyse_fresh as AF  # noqa: E402
import l2lib  # noqa: E402
import stepA  # noqa: E402


def toy_plan(n_logs=6, per=12):
    tokens, tl = [], {}
    for i in range(n_logs):
        for j in range(per):
            t = f"{i:02d}{j:02d}" + "0" * 12
            tokens.append(t)
            tl[t] = f"log{i}"
    return {"tokens": tokens, "token_log": tl}


def toy_stepA(plan, logs=("log0", "log1", "log2"), k=5):
    out = []
    for l in logs:
        out += [t for t in plan["tokens"] if plan["token_log"][t] == l][:k]
    return sorted(out)


def assert_log_disjoint(rec):
    assert rec["stepA_overlap"]["logs"] == 0, f"carve shares {rec['stepA_overlap']['logs']} logs with Step A"


# ---------------------------------------------------------------------------------------------------------------- the carve
def test_literal_carve_excludes_every_stepA_log():
    p = toy_plan()
    sA = toy_stepA(p)
    rec = FC.carve("literal", p, sA, n=15, seed=FC.NEW_SEED)
    assert rec["status"] == "OK" and rec["n"] == 15 and rec["universe_logs"] == 3
    assert_log_disjoint(rec)
    assert rec["stepA_overlap"]["tokens"] == 0


def test_literal_carve_RED_mutation_token_exclusion_is_caught():
    """the mutation: exclude Step A's TOKENS instead of its LOGS (what a careless reading of 'disjoint' does).  It must trip the log-disjointness check."""
    p = toy_plan()
    sA = toy_stepA(p)
    rec = FC.carve("token_disjoint", p, sA, n=15, seed=FC.NEW_SEED)
    assert rec["stepA_overlap"]["tokens"] == 0
    with pytest.raises(AssertionError):
        assert_log_disjoint(rec)


def test_empty_universe_is_reported_EMPTY_never_as_a_zero_sample():
    p = toy_plan(n_logs=3)
    sA = toy_stepA(p, logs=("log0", "log1", "log2"))
    rec = FC.carve("literal", p, sA, n=10, seed=1)
    assert rec["status"] == "EMPTY" and rec["n"] == 0 and rec["tokens"] == []


def test_short_universe_takes_all_and_says_so():
    p = toy_plan()
    sA = toy_stepA(p)
    rec = FC.carve("literal", p, sA, n=500, seed=1)
    assert rec["status"] == "SHORT" and rec["n"] == 36


def test_same_seed_same_carve_and_new_seed_differs():
    p = toy_plan()
    sA = toy_stepA(p)
    a = FC.carve("token_disjoint", p, sA, n=20, seed=FC.NEW_SEED)["tokens"]
    b = FC.carve("token_disjoint", p, sA, n=20, seed=FC.NEW_SEED)["tokens"]
    c = FC.carve("token_disjoint", p, sA, n=20, seed=stepA.SEED)["tokens"]
    assert a == b and a != c
    assert FC.NEW_SEED != stepA.SEED


def test_more_logs_than_n_is_refused_not_altered():
    p = toy_plan(n_logs=8, per=3)
    with pytest.raises(ValueError):
        FC.carve("log_disjoint_c", p, [], n=4, seed=1, c_rows=[{"token": t, "log": p["token_log"][t], "served": False} for t in p["tokens"]])


def test_REAL_inputs_literal_universe_is_empty():
    """MEASURED fact the brief's recipe runs into: Step A's 24 logs are ALL of the plan's logs."""
    plan = json.load(open(FC.PLAN))
    sA = json.load(open(f"{FC.W}/sample.json"))
    toks, logs, _ = FC.universe("literal", plan, sA["tokens"])
    assert toks == [] and logs == []
    tb, lb, _ = FC.universe("token_disjoint", plan, sA["tokens"])
    assert len(tb) == 1305 and len(set(lb)) == 24


def test_g0_empty_is_not_a_vacuous_pass():
    g = FC.g0_for({"tokens": [], "logs": []}, [])
    assert g["evaluable"] is False


# ---------------------------------------------------------------------------------------------------------------- the seal of the head fit
def test_final_head_seal_checks_pass_on_the_sealed_files_and_RED_on_a_wrong_pin():
    rec = FH.seal_checks()
    assert rec["ok"]
    bad = {"prereg": FH.PREREG_SHA256_PIN, "recipe": "0" * 64, "code": FH.SEALED_CODE}
    with pytest.raises(SystemExit) as e:
        FH.seal_checks(pins=bad)
    assert "SEAL MISMATCH" in str(e.value)
    badc = dict(FH.SEALED_CODE, **{"l2lib.py": "1" * 64})
    with pytest.raises(SystemExit):
        FH.seal_checks(pins={"prereg": FH.PREREG_SHA256_PIN, "recipe": FH.RECIPE_SHA256_PIN, "code": badc})


def test_banked_final_heads_are_what_final_heads_json_says():
    base = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads"
    man = json.load(open(f"{base}/final_heads.json"))
    P2 = np.load("D:/Projects/TanitAD-artifacts/l2_offprop/p2_score_head.npz")
    assert man["g_live"]["pass"] and man["c_id"]["pass"]
    for s in ("s0", "s1"):
        p = f"{base}/final_head_{s}.npz"
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == man["heads"][s]["sha256"]
        W, b = SF.head_arrays(p)
        fr = [r for r in range(6) if r != l2lib.EP]
        assert np.array_equal(W[fr], P2["weight"][fr]) and np.array_equal(b[fr], P2["bias"][fr])
        assert not np.array_equal(W[l2lib.EP], P2["weight"][l2lib.EP])          # the EP row MOVED (a head that equals P2 would be the C-ID arm, not the fit)


# ---------------------------------------------------------------------------------------------------------------- C-FEAT
def test_cfeat_passes_on_consistent_tokens_and_RED_when_a_banked_score_is_wrong():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(6, 8)).astype(np.float32)
    b = rng.normal(size=6).astype(np.float32)
    ti, tp = rng.normal(size=(3, 64, 8)).astype(np.float32), rng.normal(size=(3, 12, 8)).astype(np.float32)
    al, pl = ti.astype(np.float64) @ W.T.astype(np.float64) + b, tp.astype(np.float64) @ W.T.astype(np.float64) + b
    assert SF.c_feat(ti, tp, W, b, al, pl)["pass"]
    pl2 = pl.copy()
    pl2[..., l2lib.EP] += 0.01 * (rng.random(pl2.shape[:-1]) < 0.5)                   # half the rows off by 0.01 > 1e-4
    assert not SF.c_feat(ti, tp, W, b, al, pl2)["pass"]


# ---------------------------------------------------------------------------------------------------------------- the seal of the read
def _sealed_file(tmp_path, body, name="am.md"):
    p = tmp_path / name
    p.write_bytes(body.encode())
    return str(p), AF.sha_lf(str(p))


def test_read_seal_guard_RED_arms(tmp_path):
    pins = {"carve token-list sha256": "a" * 64, "final head s0 sha256": "b" * 64}
    good = "Status: SEALED\ncarve " + "a" * 64 + "\nhead " + "b" * 64 + "\n"
    p, h = _sealed_file(tmp_path, good)
    assert AF.check_seal(p, h, pins)["prereg_sha256"] == h
    for body in ("Status: DRAFT\n" + good, good.replace("b" * 64, "")):             # draft marker present / a pinned head hash not named
        p2, h2 = _sealed_file(tmp_path, body, "bad.md")
        with pytest.raises(SystemExit) as e:
            AF.check_seal(p2, h2, pins)
        assert "SEAL REFUSED" in str(e.value)
    with pytest.raises(SystemExit):
        AF.check_seal(p, "0" * 64, pins)                                                # wrong sha
    with pytest.raises(SystemExit):
        AF.check_seal(str(tmp_path / "nope.md"), h, pins)


def test_combine_seeds_is_most_severe_and_O1_needs_both():
    assert AF.combine_seeds({"L2_s0": "O1", "L2_s1": "O1"})["combined"] == "O1"
    assert AF.combine_seeds({"L2_s0": "O1", "L2_s1": "O3"})["combined"] == "O3"
    assert AF.combine_seeds({"L2_s0": "O1", "L2_s1": "O2"})["combined"] == "O2"
    assert AF.combine_seeds({"L2_s0": "O3", "L2_s1": "O2"})["combined"] == "O2"
    assert AF.combine_seeds({"L2_s0": "O1", "L2_s1": "O3"})["seeds_agree"] is False


# ---------------------------------------------------------------------------------------------------------------- instrument validity on Step A's OWN banked data (not the fresh carve)
@pytest.mark.slow
def test_evaluate_all_reproduces_StepA_RESULT_for_P2():
    """the step-7 code path, run on Step A's data with P2's EP head, must return L1's banked numbers (RESULT.json) exactly: A1 0.6850728155339806, den 9888"""
    D = l2lib.load_real("D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz")
    ep_in, ep_p = l2lib.ep_apply(D.tok_in, D.W6, D.b6), l2lib.ep_apply(D.tok_p, D.W6, D.b6)
    res = AF.evaluate_all(D, {"P2": (ep_in, ep_p)})
    R = json.load(open("D:/Projects/TanitAD-artifacts/l1_stepA_run/RESULT.json"))
    a1 = res["arms"]["P2"]["A"]["all"]["A1"]
    assert a1["point"] == pytest.approx(R["all"]["A1"]["point"], abs=1e-12) and a1["den"] == R["all"]["A1"]["den"]
    for k in ("A3", "A4", "A5"):
        assert res["arms"]["P2"]["A"]["all"][k]["point"] == pytest.approx(R["all"][k]["point"], abs=1e-12)
    assert res["arms"]["P2"]["outcome_interval_aware"] == "O2"


@pytest.mark.slow
def test_render_md_on_StepA_own_data():
    D = l2lib.load_real("D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz")
    P2 = (l2lib.ep_apply(D.tok_in, D.W6, D.b6), l2lib.ep_apply(D.tok_p, D.W6, D.b6))
    z = np.load("D:/Projects/TanitAD-artifacts/l2_fresh_stepA/_selftest_stepA/fresh_ep_scores.npz")
    res = AF.evaluate_all(D, {"P2": P2, "L2_s0": (z["ep_in_L2_s0"], z["ep_p_L2_s0"]), "L2_s1": (z["ep_in_L2_s1"], z["ep_p_L2_s1"])})
    md = AF.render_md(res)
    assert "| P2 |" in md and "| L2_s0 |" in md and "most severe" in md
    # P2 numbers are L1's banked ones; the L2 heads here are scored IN-SAMPLE on their own training tokens (a code-path test, NOT a result)
    assert res["arms"]["P2"]["A"]["all"]["A1"]["point"] == pytest.approx(0.6850728155339806, abs=1e-12)
    assert "A3_minus_P2_paired" in res["arms"]["L2_s0"] and "A1_minus_P2_paired" in res["arms"]["L2_s0"]


def test_check_head_RED_arms():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(6, 8)).astype(np.float32)
    b = rng.normal(size=6).astype(np.float32)
    W2, b2 = W.copy(), b.copy()
    W2[l2lib.EP] += 0.1
    SF.check_head("ok", W2, b, W, b)                                   # EP row moved, frozen rows identical: accepted
    with pytest.raises(SystemExit):
        SF.check_head("same", W.copy(), b.copy(), W, b)                # identical to P2: the L2 arm would be P2
    bad = W2.copy()
    bad[3] += 0.1
    with pytest.raises(SystemExit):
        SF.check_head("frozen", bad, b, W, b)                          # a frozen row moved


# ================================================================================================================ the log-OOF companion (MM ruling: reported-only, never gated)
import logoof_companion as LC  # noqa: E402

PKG = os.path.dirname(HERE)
LOGOOF_PKG_DIR = f"{PKG}/raw/logoof"
STEPA_FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
BANKED_ARMS = "D:/Projects/TanitAD-artifacts/l2_offprop/run/arms"


def test_score_by_fold_toy_scores_each_token_by_its_own_fold_and_refuses_a_gap():
    rng = np.random.default_rng(3)
    heads = [{"W": rng.normal(size=(6, 4)).astype(np.float32), "b": rng.normal(size=6).astype(np.float32)} for _ in range(5)]
    ti, tp = rng.normal(size=(10, 64, 4)).astype(np.float32), rng.normal(size=(10, 96, 4)).astype(np.float32)
    fold = np.arange(10) % 5
    ei, ep = LC.score_by_fold(ti, tp, fold, heads)
    for n in range(10):
        assert np.array_equal(ei[n], l2lib.ep_apply(ti[n], heads[fold[n]]["W"], heads[fold[n]]["b"]))
    bad = fold.copy()
    bad[0] = 7                                                      # a fold id with no head: must be refused, not scored as zeros
    with pytest.raises(RuntimeError):
        LC.score_by_fold(ti, tp, bad, heads)


def test_compatible_logs_RED_on_a_different_log_set():
    assert LC.compatible_logs(["a", "b"], ["b", "a"])[0]
    ok, why = LC.compatible_logs(["a", "b", "c"], ["a", "b"])
    assert not ok and "differ" in why


def test_banked_logoof_heads_reproduce_the_banked_OOF_scores_BIT_FOR_BIT_and_RED_arms():
    """IDENTITY on the banked heads file (no refit): the fold heads in logoof_heads.npz, applied to Step A's tokens by fold, equal L2's banked arms/H_s{s}.npz exactly.  RED: rolled folds differ."""
    h = LC.load_heads(f"{LOGOOF_PKG_DIR}/{LC.HEADS_NPZ}", f"{LOGOOF_PKG_DIR}/{LC.HEADS_JSON}")
    assert h["manifest"]["reproduction"]["pass"]
    D = l2lib.load_real(STEPA_FEATS)
    ok, _ = LC.compatible_logs(D.log_names, h["manifest"]["log_names"])
    assert ok
    for s, heads in h["heads"].items():
        z = np.load(f"{BANKED_ARMS}/H_s{s}.npz", allow_pickle=False)
        ei, ep = LC.score_by_fold(D.tok_in, D.tok_p, D.fold, heads)
        assert np.array_equal(ei, z["ep_in"]) and np.array_equal(ep, z["ep_p"])
        rolled = [heads[(f + 1) % LC.N_FOLDS] for f in range(LC.N_FOLDS)]
        ei2, _ = LC.score_by_fold(D.tok_in, D.tok_p, D.fold, rolled)
        assert not np.array_equal(ei2, z["ep_in"])
        frozen = [r for r in range(6) if r != l2lib.EP]
        assert all(np.array_equal(hh["W"][frozen], np.asarray(D.W6, np.float32)[frozen]) for hh in heads)       # only the EP row moved


def test_load_heads_RED_on_a_tampered_file_or_a_failed_reproduction(tmp_path):
    import shutil
    d = tmp_path / "lo"
    d.mkdir()
    for fn in (LC.HEADS_NPZ, LC.HEADS_JSON):
        shutil.copy(f"{LOGOOF_PKG_DIR}/{fn}", d / fn)
    LC.load_heads(str(d / LC.HEADS_NPZ), str(d / LC.HEADS_JSON))
    with open(d / LC.HEADS_NPZ, "ab") as f:
        f.write(b"\0")                                              # one trailing byte: the sha256 must no longer match the manifest
    with pytest.raises(SystemExit):
        LC.load_heads(str(d / LC.HEADS_NPZ), str(d / LC.HEADS_JSON))
    shutil.copy(f"{LOGOOF_PKG_DIR}/{LC.HEADS_NPZ}", d / LC.HEADS_NPZ)
    man = json.load(open(d / LC.HEADS_JSON))
    man["reproduction"]["pass"] = False
    json.dump(man, open(d / LC.HEADS_JSON, "w"))
    with pytest.raises(SystemExit):
        LC.load_heads(str(d / LC.HEADS_NPZ), str(d / LC.HEADS_JSON))


def test_reproduction_record_RED_a_wrong_seed_fails_the_identity():
    """the machinery's own RED: feeding fold heads fitted at seed 1 against the banked seed-0 scores must fail the identity (so a pass is not vacuous).  Uses the banked heads, no refit."""
    h = LC.load_heads(f"{LOGOOF_PKG_DIR}/{LC.HEADS_NPZ}", f"{LOGOOF_PKG_DIR}/{LC.HEADS_JSON}")
    D = l2lib.load_real(STEPA_FEATS)
    z0 = np.load(f"{BANKED_ARMS}/H_s0.npz", allow_pickle=False)
    ei1, ep1 = LC.score_by_fold(D.tok_in, D.tok_p, D.fold, h["heads"][1])
    assert not LC._equal(ei1, z0["ep_in"])["bit_equal"] and not LC._equal(ep1, z0["ep_p"])["bit_equal"]


def test_companion_arms_reach_the_report_and_never_the_verdict():
    """on Step A's OWN tokens the companion IS L2's banked OOF read: its A1 must equal L2_RESULT.json's H_s0 / H_s1 (0.827 / 0.825); it must not enter L2_outcome; it carries no outcome; the A3 consequence reads the FINAL heads only"""
    h = LC.load_heads(f"{LOGOOF_PKG_DIR}/{LC.HEADS_NPZ}", f"{LOGOOF_PKG_DIR}/{LC.HEADS_JSON}")
    D = l2lib.load_real(STEPA_FEATS)
    P2 = (l2lib.ep_apply(D.tok_in, D.W6, D.b6), l2lib.ep_apply(D.tok_p, D.W6, D.b6))
    z = np.load("D:/Projects/TanitAD-artifacts/l2_fresh_stepA/_selftest_stepA/fresh_ep_scores.npz")
    scores = {"P2": P2, "L2_s0": (z["ep_in_L2_s0"], z["ep_p_L2_s0"]), "L2_s1": (z["ep_in_L2_s1"], z["ep_p_L2_s1"])}
    for s, heads in h["heads"].items():
        scores[f"LOGOOF_s{s}"] = LC.score_by_fold(D.tok_in, D.tok_p, D.fold, heads)
    res = AF.evaluate_all(D, scores)
    R = json.load(open("D:/Projects/TanitAD-artifacts/l2_offprop/run/L2_RESULT.json"))
    for s in (0, 1):
        got = res["arms"][f"LOGOOF_s{s}"]["A"]["all"]["A1"]
        banked = R["arms"][f"H_s{s}"]["all"]["A1"]
        for k in ("point", "lo", "hi"):
            assert got[k] == pytest.approx(banked[k], abs=1e-12)                       # the companion on Step A's own tokens IS L2's banked OOF read
        assert res["arms"][f"LOGOOF_s{s}"]["outcome_interval_aware"] is None and res["arms"][f"LOGOOF_s{s}"]["gated"] is False
    assert set(res["L2_outcome"]["per_seed"]) == {"L2_s0", "L2_s1"}                  # the companion never enters the verdict
    assert set(res["a3_consequence"]["dA3_upper_per_seed"]) == {"L2_s0", "L2_s1"}
    md = AF.render_md(res)
    assert "reported-only, no outcome" in md and "LOGOOF_s0" in md and "LOG-SHARED" in md


def test_a3_consequence_triggers_only_when_the_upper_CI_is_below_zero_on_BOTH_seeds():
    def arms(h0, h1):
        return {"L2_s0": {"A3_minus_P2_paired": {"point": -0.03, "lo": -0.06, "hi": h0}}, "L2_s1": {"A3_minus_P2_paired": {"point": -0.03, "lo": -0.06, "hi": h1}},
                "LOGOOF_s0": {"A3_minus_P2_paired": {"point": -0.5, "lo": -0.6, "hi": -0.4}}}
    assert AF.a3_consequence(arms(-0.01, -0.005))["triggered"] is True
    assert AF.a3_consequence(arms(-0.01, 0.002))["triggered"] is False            # one seed straddles 0: not triggered
    assert AF.a3_consequence(arms(0.01, 0.02))["triggered"] is False
    assert "EP-RELIAB" in AF.a3_consequence(arms(-0.01, -0.005))["consequence"]
    assert AF.a3_consequence({"P2": {}}) is None


# ================================================================================================================ the real-read path, end to end, on Step A's OWN data (a code-path test, NOT a result)
SELFTEST = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/_selftest_stepA"


def _pins_text(pins, code_pins, status="Status: SEALED"):
    L = [status] + [f"* {k}: {v}" for k, v in pins.items()] + [f"* code {n} {h}" for n, h in code_pins.items()]
    return "\n".join(L) + "\n"


@pytest.mark.slow
def test_real_read_main_end_to_end_on_StepA_own_data_with_a_generated_seal(tmp_path, capsys):
    """AF.main on the selftest run dir (Step A's own tokens, in-sample final heads) with a seal generated here: the guard accepts, the companion arms appear (on Step A's tokens they ARE L2's banked OOF read),
    the files are written, and the L2 verdict ignores the companion.  Reads no fresh carve."""
    sample = json.load(open(f"{SELFTEST}/selftest_sample.json"))
    pins = AF.disk_pins(sample, AF.HEADS_DIR, AF.LOGOOF_DIR if os.path.exists(f"{AF.LOGOOF_DIR}/{LC.HEADS_NPZ}") else LOGOOF_PKG_DIR)
    logdir = AF.LOGOOF_DIR if os.path.exists(f"{AF.LOGOOF_DIR}/{LC.HEADS_NPZ}") else LOGOOF_PKG_DIR
    p = tmp_path / "selftest_sealed.md"
    p.write_bytes(_pins_text(pins, AF.current_code_pins()).encode())
    out = tmp_path / "out"
    out.mkdir()
    rc = AF.main(["--run-dir", SELFTEST, "--prereg", str(p), "--prereg-sha256", AF.sha_lf(str(p)), "--sample", f"{SELFTEST}/selftest_sample.json", "--out", str(out), "--logoof-dir", logdir])
    assert rc == 0
    res = json.load(open(out / "FRESH_RESULT.json"))
    assert {"P2", "L2_s0", "L2_s1", "LOGOOF_s0", "LOGOOF_s1"} <= set(res["arms"])
    assert res["arms"]["LOGOOF_s0"]["outcome_interval_aware"] is None
    assert set(res["L2_outcome"]["per_seed"]) == {"L2_s0", "L2_s1"}
    assert res["companion"]["reproduction_pass"] is True and "omitted" not in res["companion"]
    assert abs(res["arms"]["LOGOOF_s0"]["A"]["all"]["A1"]["point"] - 0.8267597087378641) < 1e-12
    md = open(out / "FRESH_RESULT.md", encoding="utf-8").read()
    assert "Log-OOF companion" in md and "LOG-SHARED" in md and "A3 (reported, ungated)" in md
    # RED: the same read with a one-byte-different seal file is refused before anything is computed
    with open(p, "ab") as f:
        f.write(b" ")
    with pytest.raises(SystemExit):
        AF.main(["--run-dir", SELFTEST, "--prereg", str(p), "--prereg-sha256", AF.sha_lf(str(p))[:-1] + "0", "--sample", f"{SELFTEST}/selftest_sample.json", "--out", str(out), "--logoof-dir", logdir])


def test_check_seal_RED_arms_for_the_code_pins_and_the_positive_marker(tmp_path):
    pins = {"carve token-list sha256": "a" * 64}
    cp = {"analyse_fresh.py": "c" * 64, "l2lib.py": "d" * 64}
    good = _pins_text(pins, cp)
    p = tmp_path / "g.md"
    p.write_bytes(good.encode())
    assert AF.check_seal(str(p), AF.sha_lf(str(p)), pins, cp)["code_pins"] == cp
    cases = {"code hash edited after the seal": dict(cp, **{"l2lib.py": "e" * 64}),          # the file on disk is NOT what the sealed text names
             "code file never sealed": dict(cp, **{"stepA.py": "f" * 64})}
    for why, now in cases.items():
        with pytest.raises(SystemExit) as e:
            AF.check_seal(str(p), AF.sha_lf(str(p)), pins, now)
        assert "SEAL REFUSED" in str(e.value), why
    for body in (good.replace("Status: SEALED", "Status: IN REVIEW"), good.replace("Status: SEALED", "Status: DRAFT")):      # no positive marker / the draft marker
        q = tmp_path / "b.md"
        q.write_bytes(body.encode())
        with pytest.raises(SystemExit):
            AF.check_seal(str(q), AF.sha_lf(str(q)), pins, cp)
    swapped = good.replace("analyse_fresh.py " + "c" * 64, "analyse_fresh.py " + "0" * 64).replace("l2lib.py " + "d" * 64, "l2lib.py " + "c" * 64)   # a hash present in the text but on the WRONG file's line
    q = tmp_path / "s.md"
    q.write_bytes(swapped.encode())
    with pytest.raises(SystemExit):
        AF.check_seal(str(q), AF.sha_lf(str(q)), pins, cp)


# ================================================================================================================ the SEALED amendment (PREREG_L2_FRESH.md) through the real CLI, no fresh label or score read
SEALED_PATH = f"{PKG}/PREREG_L2_FRESH.md"


@pytest.mark.skipif(not os.path.exists(SEALED_PATH), reason="no sealed amendment in this checkout")
def test_sealed_amendment_accepted_and_every_RED_copy_refused_with_exit_5():
    import prove_guard
    o = prove_guard.cases()
    assert o["ACCEPT"]["exit"] == 0
    for k in prove_guard.RED_KEYS:
        assert o[k]["exit"] == 5 and "SEAL REFUSED (5)" in o[k]["last_line"], (k, o[k])
    assert o["pass"]


@pytest.mark.skipif(not os.path.exists(SEALED_PATH), reason="no sealed amendment in this checkout")
def test_sealed_text_carries_the_ruling_verbatim_the_stamp_and_the_current_pins():
    txt = open(SEALED_PATH, encoding="utf-8").read()
    ruling = open(f"{PKG}/MM_RULING_L2_FRESH.md", "rb").read().replace(b"\r\n", b"\n").decode("utf-8").rstrip("\n")
    assert ruling in txt and "Revision 1 (MM ruling, pre-seal)" in txt
    assert AF.DRAFT_MARKER not in txt and AF.SEALED_MARKER in txt
    for must in ("token-fresh, LOG-SHARED", "interval_aware", "O3 is the EXPECTED outcome", "READ 2 on universe D", "L2-BRANCH", "most severe", "EP-RELIAB-class risk", "reported-only", "NEVER gated"):
        assert must in txt, must
    assert open(f"{PKG}/PREREG_L2_FRESH.sha256").read().split()[0] == AF.sha_lf(SEALED_PATH)
    sample = json.load(open(f"{PKG}/raw/fresh_sample_token_disjoint.json"))
    pins = AF.disk_pins(sample, AF.HEADS_DIR, AF.LOGOOF_DIR)
    assert AF.check_seal(SEALED_PATH, AF.sha_lf(SEALED_PATH), pins, AF.current_code_pins())["prereg_sha256"] == AF.sha_lf(SEALED_PATH)
