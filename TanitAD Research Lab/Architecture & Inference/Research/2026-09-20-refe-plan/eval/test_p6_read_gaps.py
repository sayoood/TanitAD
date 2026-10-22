#!/usr/bin/env python3
"""P6-READ-GAPS (2026-10-22) tests: G1 (C2 comparator), G2 (proposal dump wiring, selection quality), G3 (banked, pinned harness).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/test_p6_read_gaps.py            # fast tests, every one with a deliberate-regression arm
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/test_p6_read_gaps.py --slow     # + the full stage-3 pipeline control on the real 12,146-token tables

Real-data tests read banked arms' navtest tables (u400, P2): KNOWN-VALUE controls, never a P6 checkpoint. Each RED arm MUTATES the thing under test and requires the check to fail.
"""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness_pins as HP  # noqa: E402
import stage3_full as S3  # noqa: E402

DATA = Path(os.environ.get("REFE_NAVTEST_DIR", "E:/Projects/TanitAD/data/refe_navtest"))
PUBLISHED_C2_MINUS_C1 = (0.5714677935040782, 0.017916605286551753, 1.1755185398706811)   # raw/p2_readout/p2_full/r2_result.json R2.primary (MEASURED 2026-10-14)
RESULTS: list = []


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
                RESULTS.append((name, True, ""))
            except Exception as e:                                           # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {str(e)[:300]}"))
        run.__name__ = fn.__name__
        run.check_name = name
        return run
    return deco


def must_fail(fn, *a, **k):
    """the deliberate-regression arm: `fn` must raise AssertionError/SystemExit (the check went RED)"""
    try:
        fn(*a, **k)
    except (AssertionError, SystemExit):
        return True
    raise AssertionError("the deliberate regression was NOT caught: the check cannot fail")


# ============================================================ G1
def _export_logs():
    exp = json.load(gzip.open(S3.EXPORT, "rt", encoding="utf-8"))["tokens"]
    return {t: v["log_name"] for t, v in exp.items()}


def g1_control(base_name="u400", arm_name="p2"):
    """the registered estimator on the C2 and C1 tables: C2 - C1 must be the published value, to the bit"""
    a, b = S3.pick_baseline(arm_name), S3.pick_baseline(base_name)
    assert a["pin_ok"] and b["pin_ok"], (a["pin_ok"], b["pin_ok"])
    arm, base = S3.load_table(a["csv"]), S3.load_table(b["csv"])
    tl = _export_logs()
    toks = sorted(arm)
    r = S3.read_one(arm, base, toks, tl, "PDMS")
    got = (r["mean"], r["ci95"][0], r["ci95"][1])
    assert got == PUBLISHED_C2_MINUS_C1, (got, PUBLISHED_C2_MINUS_C1)
    assert (round(got[0], 2), round(got[1], 2), round(got[2], 2)) == (0.57, 0.02, 1.18)       # the registered +0.57 [+0.02, +1.18]
    return r


@check("G1 C2 comparator: pins hold on the real P2 files, and C2 - C1 = +0.5715 [+0.0179, +1.1755] to the bit (published +0.57 [+0.02, +1.18])")
def t_g1_known_value():
    g1_control()


@check("G1 RED: a WRONG pin on the p2 entry (u400's csv md5) makes the pin gate fail and the control fail")
def t_g1_wrong_pin_red():
    orig = copy.deepcopy(S3.BASELINES["p2"])
    try:
        S3.BASELINES["p2"]["csv_md5"] = S3.BASELINES["u400"]["csv_md5"]
        assert S3.pick_baseline("p2")["pin_ok"] is False
        must_fail(g1_control)
        S3.BASELINES["p2"]["csv_md5"] = orig["csv_md5"]
        S3.BASELINES["p2"]["seam_md5"] = "0" * 32
        assert S3.pick_baseline("p2")["pin_ok"] is False
        must_fail(g1_control)
    finally:
        S3.BASELINES["p2"] = orig
    g1_control()                                                              # restored: green again (the RED arm did not break the instrument)


@check("G1 RED: the control reads the right table (arm and base swapped must NOT reproduce the published value)")
def t_g1_swapped_red():
    must_fail(g1_control, "p2", "u400")


@check("G1 a pin may be a tuple (LF and CRLF forms); a md5 outside the tuple is refused")
def t_g1_tuple_pin():
    p = DATA / "score/refe_stage2_p2_full/refe_stage2_p2_full.csv"
    m = S3.md5_file(p)
    path, why = S3.first_matching([p], (m, "1" * 32))
    assert path == p and not why
    path, why = S3.first_matching([p], ("1" * 32, "2" * 32))
    assert path is None and why
    path, why = S3.first_matching([p], m)                                        # the old plain-string pin still works
    assert path == p


# ============================================================ G3
def _copy_harness(td: Path) -> Path:
    dst = td / "harness"
    shutil.copytree(HP.HARNESS, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _caller_ok_path() -> Path:
    """a caller file that is a known variant: the one beside this test if it is, else the pre-NT-FIX1 working-tree copy"""
    pins = HP.load_pins()
    c = HERE / "score_navtest_refe.py"
    if c.exists() and HP.sha256_lf(c) in pins["caller_variants"].values():
        return c
    raise AssertionError(f"{c}: not a pinned caller variant (sha256_lf {HP.sha256_lf(c) if c.exists() else 'absent'})")


@check("G3 the banked harness equals its pins (every file, none missing, none unpinned, the caller a known variant)")
def t_g3_pins_hold():
    r = HP.verify(HP.HARNESS, caller=_caller_ok_path())
    assert r["ok"], r
    assert r["n_files"] == 21, r["n_files"]


def f6_first(cands):
    """what stage3_full.resolve_families6 would pick from `cands` (no REFE_FAMILIES6 override): it must be the banked, pinned copy"""
    got = next((Path(c) for c in cands if c and Path(c).exists()), None)
    banked = HP.HARNESS / "ev6e" / "families6.py"
    assert got is not None and got.resolve() == banked.resolve(), (str(got), str(banked))
    assert HP.sha256_lf(got) == HP.load_pins()["files"]["ev6e/families6.py"]["sha256_lf"], "the resolved families6 is not the pinned file"
    return got


@check("G3 stage3_full resolves families6 to the banked, pinned copy FIRST (the D: W3 original and a scratchpad are fallbacks only)")
def t_g3_stage3_families6_banked_first():
    cands = tuple(c for c in S3.FAMILIES6_CANDIDATES if c != os.environ.get("REFE_FAMILIES6", "") or not c)
    f6_first(cands)
    must_fail(f6_first, tuple(reversed(cands)))                                          # the old order (W3 original first, banked copy last) must fail the check
    assert not any("scratchpad" in str(c) or "/Temp/" in str(c) for c in S3.FAMILIES6_CANDIDATES), "a %TEMP% scratchpad path is still a candidate"


@check("G3 byte identity to the scratchpad files the SFT-4 / C2 results used (skipped, stated, if the scratchpad is gone)")
def t_g3_byte_identical_to_scratchpad():
    pins = HP.load_pins()
    sp = Path(pins["scratchpad"])
    if not sp.exists():
        RESULTS.append(("G3 scratchpad byte identity: SKIPPED (scratchpad absent; the sha256_lf pins carry the claim)", True, ""))
        return
    n_same = n_edit = 0
    for rel, rec in pins["files"].items():
        if rec["origin"] == "scratchpad_byte_identical":
            assert (sp / rel).read_bytes() == (HP.HARNESS / rel).read_bytes(), rel
            n_same += 1
        elif rec["origin"] == "scratchpad_plus_one_line":
            a = (sp / rel).read_text(encoding="utf-8").split("\n")
            b = (HP.HARNESS / rel).read_text(encoding="utf-8").split("\n")
            assert len(a) == len(b) and [i + 1 for i in range(len(a)) if a[i] != b[i]] == [49], rel
            n_edit += 1
    assert n_same == 19 and n_edit == 1, (n_same, n_edit)


def _expect_refused(td_mutator, caller=None):
    with tempfile.TemporaryDirectory() as td:
        h = _copy_harness(Path(td))
        td_mutator(h)
        r = HP.verify(h, caller=caller if caller is not None else _caller_ok_path())
        assert r["ok"] is False, r
        try:
            HP.require(h, caller=caller if caller is not None else _caller_ok_path())
        except SystemExit as e:
            return str(e), r
        raise AssertionError("require() did not refuse")


@check("G3 RED: one changed byte in run_v1.py is refused and the message names the file")
def t_g3_changed_byte():
    def mut(h):
        p = h / "w3e/code/run_v1.py"
        b = bytearray(p.read_bytes())
        b[200] ^= 1
        p.write_bytes(bytes(b))
    msg, r = _expect_refused(mut)
    assert "w3e/code/run_v1.py" in msg and r["mismatch"][0]["file"] == "w3e/code/run_v1.py"


@check("G3 RED: a missing file, an UNPINNED extra .py and a missing pins file are each refused")
def t_g3_missing_extra():
    msg, _ = _expect_refused(lambda h: (h / "ev6e/families6.py").unlink())
    assert "families6.py" in msg
    msg, _ = _expect_refused(lambda h: (h / "w3e/code/run_v1_extra.py").write_text("x = 1\n"))
    assert "UNPINNED" in msg
    with tempfile.TemporaryDirectory() as td:
        h = _copy_harness(Path(td))
        (h / "HARNESS_PINS.json").unlink()
        assert HP.verify(h, caller=None)["ok"] is False


@check("G3 RED: an unknown scoring caller (score_navtest_refe.py) is refused")
def t_g3_unknown_caller():
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "score_navtest_refe.py"
        c.write_text(_caller_ok_path().read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        msg, _ = _expect_refused(lambda h: None, caller=c)
        assert "CALLER" in msg


@check("G3 control: a CRLF checkout (core.autocrlf) of the same files still verifies; the guard is not so brittle that it refuses a checkout")
def t_g3_crlf_ok():
    with tempfile.TemporaryDirectory() as td:
        h = _copy_harness(Path(td))
        for p in h.rglob("*.py"):
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        r = HP.verify(h, caller=_caller_ok_path())
        assert r["ok"], r


# ---- stage2_sft.py: the seam command, the harness path, the guard order
def _load_stage2(src_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, src_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run_stage2_capture(S2, argv, harness=None, no_seam_file=True):
    """drive S2.main() to its seam and scoring calls with every side effect stubbed; returns the recorded commands"""
    cap = {"calls": [], "lock": 0}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "seams").mkdir()
        toks = {"tokens": ["t1", "t2"], "token_log": {"t1": "L", "t2": "L"}}
        (td / "tokens.json").write_text(json.dumps(toks))
        with gzip.open(td / "export.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"tokens": {"t1": {"log_name": "L"}, "t2": {"log_name": "L"}, "t3": {"log_name": "L"}}}, f)
        S2.PKG, S2.D, S2.EXPORT = td / "pkg", td, str(td / "export.json.gz")
        (S2.PKG / "eval").mkdir(parents=True)
        sp_was_harness = (S2.SP == str(S2.HARNESS))                  # a mutant that restores a scratchpad path must keep it
        S2.HARNESS = harness if harness is not None else HP.HARNESS
        if sp_was_harness:
            S2.SP = str(S2.HARNESS)
            S2.EV6E = f"{S2.SP}/ev6e"
        S2.take_lock = lambda me: cap.__setitem__("lock", cap["lock"] + 1)
        S2.release = lambda me: None
        S2.free_mb = lambda: 99999
        real_sleep = S2.time.sleep
        S2.time.sleep = lambda s: None

        def fake_call(cmd, **kw):
            cap["calls"].append({"cmd": [str(c) for c in cmd], "cwd": str(kw.get("cwd")), "env": {k: kw["env"].get(k) for k in ("RUN_V1_FIX1_ALLOW_DRIFT", "PYTHONIOENCODING")} if kw.get("env") else None})
            if "stdout" in kw and kw["stdout"] is not None:
                kw["stdout"].write('{"status": "FAIL"}\n')
                kw["stdout"].flush()
            return 0
        real_call = S2.subprocess.call
        S2.subprocess.call = fake_call
        sys.argv = ["stage2_sft.py", "--ckpt", "C:/x/model_final.pt", "--name", "gap", "--rule", "navsim_v1", "--tokens", str(td / "tokens.json"), *argv]
        try:
            rc = S2.main()
        finally:
            S2.time.sleep = real_sleep
            S2.subprocess.call = real_call
        cap["rc"] = rc
        cap["td"] = str(td)
        cap["result"] = json.load(open(S2.PKG / "raw" / "2026-10-05-sft-stage2" / "gap" / "result.json", encoding="utf-8"))
    return cap


def check_stage2(cap, S2, dump: bool):
    seam_calls = [c for c in cap["calls"] if "refe_navtest_seam.py" in c["cmd"]]
    assert len(seam_calls) == 1 and cap["lock"] == 1, (len(seam_calls), cap["lock"])
    cmd = seam_calls[0]["cmd"]
    td = cap["td"].replace("\\", "/")
    old = [S2.PY_D, "refe_navtest_seam.py", "--ckpt", "C:/x/model_final.pt", "--frames", str(Path(td) / "frames"), "--db-dir",
           "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test", "--export", str(Path(td) / "export.json.gz"), "--tokens", str(Path(td) / "tokens.json"),
           "--out", str(Path(td) / "seams" / "stage2_gap.npz"), "--arm", "REFe_gap", "--rule", "navsim_v1"]
    want = old + (["--dump-proposals", str(Path(td) / "seams" / "stage2_gap.proposals.npz")] if dump else [])
    assert cmd == want, (cmd, want)                                              # the OLD command, plus exactly the dump flag when asked for
    sc = [c for c in cap["calls"] if "score_e.py" in c["cmd"]]
    assert sc, "no scoring call recorded"
    for c in sc:
        assert c["cwd"] == str(S2.HARNESS), c["cwd"]                             # the banked harness, not a scratchpad
        assert c["env"]["RUN_V1_FIX1_ALLOW_DRIFT"] == "1"
    assert "Temp/claude" not in " ".join(str(x) for c in cap["calls"] for x in (c["cmd"], c["cwd"])).replace("\\", "/")
    assert cap["result"]["harness_pins"]["n_files"] == 21


@check("G2/G3 stage2_sft.py: seam command = the OLD command exactly without --dump-proposals; + exactly '--dump-proposals <seams/stage2_<name>.proposals.npz>' with it; scoring runs in the banked harness")
def t_stage2_wiring():
    S2 = _load_stage2(HERE / "stage2_sft.py", "stage2_sft_t1")
    # point the pin guard at THIS tree's caller (the pre- or post-NT-FIX1 variant, both known)
    orig_req = HP.require
    HP.require = lambda root, pins=None, caller=None: orig_req(root, pins, _caller_ok_path())
    try:
        check_stage2(run_stage2_capture(S2, []), S2, dump=False)
        S2b = _load_stage2(HERE / "stage2_sft.py", "stage2_sft_t2")
        check_stage2(run_stage2_capture(S2b, ["--dump-proposals"]), S2b, dump=True)
    finally:
        HP.require = orig_req


def _mutant(old: str, new: str, name: str):
    src = (HERE / "stage2_sft.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, (src.count(old), old[:60])
    td = Path(tempfile.mkdtemp())
    p = td / "stage2_sft.py"
    p.write_text(src.replace(old, new), encoding="utf-8")
    return _load_stage2(p, name)


@check("G2/G3 RED: mutants of stage2_sft.py (dump flag dropped / scratchpad path restored / pin guard removed / drift env dropped) each fail the wiring check")
def t_stage2_mutants_red():
    orig_req = HP.require
    HP.require = lambda root, pins=None, caller=None: orig_req(root, pins, _caller_ok_path())
    try:
        def go(S2, dump):
            check_stage2(run_stage2_capture(S2, ["--dump-proposals"] if dump else []), S2, dump=dump)
        must_fail(go, _mutant('                cmd += ["--dump-proposals", str(dump)]', "                pass", "s2_m1"), True)
        must_fail(go, _mutant('SP = str(HARNESS)', 'SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/x/scratchpad"', "s2_m2"), False)
        must_fail(go, _mutant('RUN_V1_FIX1_ALLOW_DRIFT="1"', 'RUN_V1_FIX1_ALLOW_DRIFT="0"', "s2_m4"), False)
        # guard removed: a TAMPERED harness must then be (wrongly) accepted -> the guard test below catches that
        S2g = _mutant("    hp = HP.require(", "    hp = (lambda *a, **k: {'n_files': 21, 'pins_file_sha256': None, 'caller': None})(", "s2_m3")
        with tempfile.TemporaryDirectory() as td:
            h = _copy_harness(Path(td))
            p = h / "w3e/code/run_v1.py"
            p.write_bytes(p.read_bytes() + b"# tampered\n")
            S2g_ok = run_stage2_capture(S2g, [], harness=h)               # without the guard the tampered harness runs: that is the defect
            assert any("score_e.py" in c["cmd"] for c in S2g_ok["calls"])
    finally:
        HP.require = orig_req


@check("G3 stage2_sft.py REFUSES a tampered harness before taking the GPU lock or starting the seam (guard first)")
def t_stage2_refuses_tampered():
    S2 = _load_stage2(HERE / "stage2_sft.py", "stage2_sft_t3")
    orig_req = HP.require
    HP.require = lambda root, pins=None, caller=None: orig_req(root, pins, _caller_ok_path())
    try:
        with tempfile.TemporaryDirectory() as td:
            h = _copy_harness(Path(td))
            p = h / "ev6e/families6.py"
            p.write_bytes(p.read_bytes() + b"# tampered\n")
            try:
                run_stage2_capture(S2, [], harness=h)
            except SystemExit as e:
                assert "families6.py" in str(e) and "NOTHING was started" in str(e)
                return
            raise AssertionError("the tampered harness was accepted")
    finally:
        HP.require = orig_req


# ============================================================ G2: selection quality (random / actual / best-of-64)
import selection_quality as SQ  # noqa: E402
import csv as csv_mod  # noqa: E402

P2_TABLE = Path(os.environ.get("P2_FULL_TABLE", "C:/Users/Admin/p2full_work/final/table_p2_full.npz"))
# raw/2026-10-16-p2-full-table/anatomy.json (MEASURED 2026-10-18, zero_tail.Boot, B 10000, seed 20260927): the banked KNOWN VALUES of P2's own 64-proposal table
P2_ANATOMY = {"actual": (88.8184088897293, 87.86107970695247, 89.77011829946386), "best_of_64": (98.23109369448527, 97.81901516296715, 98.61925747503028),
              "random": (73.00314921968497, 71.48988586296944, 74.5661639319074)}
P2_ANATOMY_COUNTS = {"n_pick_zero": 509, "n_best_zero": 64, "n_best_strictly_above_pick": 8292}


def selq_vs_anatomy(z=None, tol=1e-9):
    z = z if z is not None else dict(np.load(P2_TABLE, allow_pickle=False))
    r = SQ.read_selection_quality(z["pdms"], z["pick"], [str(t) for t in z["token"]], [str(l) for l in z["log"]])
    for k, (m, lo, hi) in P2_ANATOMY.items():
        got = r["levels_x100"][k]
        assert abs(got["mean"] - m) < tol and abs(got["ci95"][0] - lo) < tol and abs(got["ci95"][1] - hi) < tol, (k, got, (m, lo, hi))
    for k, v in P2_ANATOMY_COUNTS.items():
        assert r[k] == v, (k, r[k], v)
    return r


@check("G2 selection quality: P2's banked 64-proposal navtest table reproduces anatomy.json (pick 88.8184 / best-of-64 98.2311 / random 73.0031, CIs, counts) to 1e-9")
def t_g2_selq_known_value():
    if not P2_TABLE.exists():
        raise AssertionError(f"{P2_TABLE} absent: the known-value control cannot run")
    r = selq_vs_anatomy()
    assert r["random_control"]["actual_beats_random"] is True and "INVALID" not in r


@check("G2 RED: a shuffled pick / a duplicated proposal column / a token-shuffled table each FAIL the known-value control")
def t_g2_selq_known_value_red():
    z = dict(np.load(P2_TABLE, allow_pickle=False))
    rng = np.random.default_rng(3)
    a = dict(z)
    a["pick"] = rng.integers(0, 64, len(z["pick"]))                                 # the pick no longer the scorer's
    must_fail(selq_vs_anatomy, a)
    b = dict(z)
    b["pdms"] = z["pdms"].copy()
    b["pdms"][:, 63] = z["pdms"][:, 0]                                              # 63 distinct proposals: random and best move
    must_fail(selq_vs_anatomy, b)
    c = dict(z)
    c["pdms"] = z["pdms"][rng.permutation(len(z["pdms"]))]                          # token -> scores mapping broken
    must_fail(selq_vs_anatomy, c)


def level_vs_anatomy(csv_path, tl, tol=1e-9):
    r = SQ.level_from_csv(csv_path, tl)
    m, lo, hi = P2_ANATOMY["actual"]
    got = r["level_x100"]
    assert r["n_tokens"] == 12146 and r["n_logs"] == 136, (r["n_tokens"], r["n_logs"])
    assert abs(got["mean"] - m) < tol and abs(got["ci95"][0] - lo) < tol and abs(got["ci95"][1] - hi) < tol, (got, (m, lo, hi))
    return r


@check("G2 level CI (the PREREG goal-bar test): selection_quality.level_from_csv on P2's own harness csv reproduces its banked level 88.8184 [87.8611, 89.7701] to 1e-9, from the csv alone")
def t_g2_level_known_value():
    a = S3.pick_baseline("p2")
    assert a["pin_ok"], "the p2 baseline pins do not hold"
    level_vs_anatomy(a["csv"], _export_logs())


@check("G2 RED: the level control fails on another arm's csv (u400), on a csv with a token that has no log, and on an empty csv")
def t_g2_level_red():
    tl = _export_logs()
    must_fail(level_vs_anatomy, S3.pick_baseline("u400")["csv"], tl)                # a different arm: 88.25, not 88.8184
    drop = dict(tl)
    first = sorted(S3.load_table(S3.pick_baseline("p2")["csv"]))[0]
    del drop[first]
    try:
        SQ.level_from_csv(S3.pick_baseline("p2")["csv"], drop)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown token was silently dropped")
    with tempfile.TemporaryDirectory() as td:
        e = Path(td) / "empty.csv"
        e.write_text("token,score\naverage,0.5\n", encoding="utf-8")
        try:
            SQ.level_from_csv(e, tl)
        except ValueError:
            return
        raise AssertionError("an empty csv was accepted as a level")


def _synthetic(n_tok=40, n_log=5, seed=11):
    rng = np.random.default_rng(seed)
    pdms = rng.uniform(0, 1, (n_tok, 64))
    pick = np.argmax(pdms, 1)                                                       # a perfect scorer on half the tokens, random on the rest
    pick[::2] = rng.integers(0, 64, len(pick[::2]))
    toks = [f"tok{i:03d}" for i in range(n_tok)]
    logs = [f"log{i % n_log}" for i in range(n_tok)]
    return pdms, pick, toks, logs


@check("G2 selection quality on a synthetic world with analytic answers: actual = the pick's PDMS, random = the mean over 64, best = the max over 64; differences consistent")
def t_g2_selq_synthetic():
    pdms, pick, toks, logs = _synthetic()
    r = SQ.read_selection_quality(pdms, pick, toks, logs, n=200)
    L = r["levels_x100"]
    assert L["actual"]["mean"] == float((pdms[np.arange(len(pick)), pick] * 100).mean())
    assert abs(L["random"]["mean"] - float(pdms.mean() * 100)) < 1e-9
    assert abs(L["best_of_64"]["mean"] - float(pdms.max(1).mean() * 100)) < 1e-9
    D = r["paired_differences_x100"]
    assert abs(D["best_minus_actual"]["mean"] - (L["best_of_64"]["mean"] - L["actual"]["mean"])) < 1e-9
    assert abs(D["actual_minus_random"]["mean"] - (L["actual"]["mean"] - L["random"]["mean"])) < 1e-9
    assert L["random"]["mean"] < L["actual"]["mean"] < L["best_of_64"]["mean"]
    bad = pdms.copy()
    bad[3, 5] = np.nan
    must_fail(SQ.read_selection_quality, bad, pick, toks, logs, 50)                # NaN is not a read
    must_fail(SQ.read_selection_quality, pdms[:, :63], pick, toks, logs, 50)       # 63 proposals is not a read


@check("G2 selection quality: random >= actual is flagged INVALID (an instrument failure, C3's rule), never reported as a result")
def t_g2_selq_invalid_flag():
    pdms, _pick, toks, logs = _synthetic()
    worst = np.argmin(pdms, 1)                                                      # a scorer that picks the WORST proposal
    with tempfile.TemporaryDirectory() as td:
        tb = Path(td) / "t.npz"
        np.savez(tb, pdms=pdms, pick=worst, token=np.array(toks), log=np.array(logs))
        r = SQ.read_table(tb)
        assert "INVALID" in r and r["random_control"]["random_ge_actual_point"] is True
        np.savez(tb, pdms=pdms, pick=np.argmax(pdms, 1), token=np.array(toks), log=np.array(logs))
        assert "INVALID" not in SQ.read_table(tb)


def _fake_dump(td: Path, n=6):
    rng = np.random.default_rng(5)
    props = rng.uniform(10, 90, (n, 64, 8, 3)).astype(np.float32)
    toks = np.array([f"tok{i}" for i in range(n)])
    pick = rng.integers(0, 64, n)
    p = td / "dump.npz"
    np.savez(p, token=toks, fingerprint=np.array([f"fp{i}" for i in range(n)]), proposals=props, logits=np.zeros((n, 64, 6), np.float32), pick=pick, sampling=np.array([8, 0.5]))
    return p, toks, props, pick


def _fake_runner_factory(calls, fail_k=None):
    def runner(label, seam, tokens, out, log):
        calls.append(label)
        k = int(label.rsplit("_p", 1)[1])
        with np.load(seam) as z:
            zpos, ztok = z["poses"], z["token"]
        sc = np.clip(zpos[:, -1, 0] / 100.0, 0, 1)
        d = Path(out) / label
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{label}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["token", "valid", "score", *SQ.SUB])
            for t, s in zip(ztok, sc):
                w.writerow([str(t), "True", repr(float(s)), *["1.0"] * 6])
            w.writerow(["average", "True", "0.5", *["1.0"] * 6])
        st = {"label": label, "status": "FAIL" if k == fail_k else "PASS", "csv_valid_rows": len(sc)}
        txt = json.dumps(st) + "\n"
        Path(log).write_text(txt, encoding="utf-8")
        return txt
    return runner


@check("G2 the 64-column chain (seams -> harness columns -> table -> read) on a synthetic dump with a fake runner: gates PASS, resumable, and every RED arm fails")
def t_g2_selq_chain():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dump, toks, props, pick = _fake_dump(td)
        root = td / "data"
        n = SQ.write_column_seams(dump, "x", root)
        assert n == 6 and all(SQ.seam_k("x", k, root).exists() for k in range(64))
        with np.load(SQ.seam_k("x", 7, root)) as zz:
            assert np.array_equal(zz["poses"], props[:, 7])                              # column k = proposals[:, k]
        calls = []
        res = SQ.score_columns("x", n, workers=3, runner=_fake_runner_factory(calls), root=root)
        assert res["ok"] and len(calls) == 64
        calls2 = []
        res2 = SQ.score_columns("x", n, workers=3, runner=_fake_runner_factory(calls2), root=root)
        assert res2["ok"] and calls2 == [], "a PASSed column was re-scored: not resumable"
        # the arm's own csv = the harness score of the picked proposal
        arm = td / "arm.csv"
        with open(arm, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["token", "valid", "score", *SQ.SUB])
            for i, t in enumerate(toks):
                w.writerow([str(t), "True", repr(float(np.clip(props[i, pick[i], -1, 0] / 100.0, 0, 1))), *["1.0"] * 6])
            w.writerow(["average", "True", "0.5", *["1.0"] * 6])
        tl = {str(t): f"log{i % 2}" for i, t in enumerate(toks)}
        g = SQ.assemble_table(dump, "x", arm, td / "table.npz", tl, root)
        assert g["ok"] and g["G2_pdms_at_pick_equals_arm_csv"]["n_mismatch"] == 0, g
        r = SQ.read_table(td / "table.npz")
        with np.load(td / "table.npz") as z:
            assert abs(r["levels_x100"]["best_of_64"]["mean"] - float(z["pdms"].max(1).mean() * 100)) < 1e-9
        # RED 1: the arm's csv disagrees with the table at the pick on one token -> G2 fails (the dump is not the run that was scored)
        bad = td / "arm_bad.csv"
        bad.write_text(arm.read_text(encoding="utf-8").replace(repr(float(np.clip(props[2, pick[2], -1, 0] / 100.0, 0, 1))), "0.123456", 1), encoding="utf-8")
        assert SQ.assemble_table(dump, "x", bad, td / "t2.npz", tl, root)["ok"] is False
        # RED 2: one column FAILs the harness -> score_columns not ok, names the column
        root2 = td / "data2"
        SQ.write_column_seams(dump, "y", root2)
        r2 = SQ.score_columns("y", n, workers=2, runner=_fake_runner_factory([], fail_k=9), root=root2)
        assert r2["ok"] is False and r2["failed"] == [9]
        # RED 3: the column holding the pick scored from ANOTHER proposal's poses -> the table at the pick no longer equals the arm's csv -> G2 fails
        with np.load(dump) as d:
            dtok, dfp = d["token"], d["fingerprint"]
        np.savez(SQ.seam_k("x", int(pick[0]), root), token=dtok, fingerprint=dfp, poses=props[:, (int(pick[0]) + 1) % 64], sampling=np.array([8, 0.5]), arm=np.array("m"))
        for k in range(64):
            shutil.rmtree(SQ.csv_k("x", k, root).parent, ignore_errors=True)
        for p in (root / "score" / "_selq_x_logs").glob("*.log"):
            p.unlink()
        SQ.score_columns("x", n, workers=3, runner=_fake_runner_factory([]), root=root)
        assert SQ.assemble_table(dump, "x", arm, td / "t3.npz", tl, root)["ok"] is False


# ============================================================ G4: P6's commands.json as the command source of the S0 dump and the candidate-bank builder
PKG_ROOT = HERE.parent
DUMP_PY = PKG_ROOT / "raw" / "2026-10-18-sel-rank" / "code" / "srttc_s0_dump.py"
BCB_PY = PKG_ROOT / "raw" / "2026-10-19-p6-read-prep" / "code" / "build_candidate_bank.py"
P6_COMMANDS = Path(os.environ.get("P6_COMMANDS_JSON", "D:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json"))
S0_PLAN = Path("C:/Users/Admin/srttc_s0/run/plan/master_plan.json")
P6_EPOCH6 = Path("D:/Projects/TanitAD-artifacts/p6dry/hf/run/ckpt_last.pt")


def _load_mod(path: Path, name: str, src: str | None = None):
    if src is not None:
        td = Path(tempfile.mkdtemp())
        p = td / path.name
        p.write_bytes(src.encode("utf-8"))
        path = p
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(str(path.parent))
    return m


def _small_table(td: Path, rows=None):
    rows = rows or {"LOG_A/tokA": [1, 0, 0, 0], "LOG_A/tokB": [0, 1, 0, 0], "LOG_B/tokC": [0, 0, 1, 0]}
    p = td / "commands.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def g4_dump_helpers(DM):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = _small_table(td)
        tab = DM.load_command_table(str(p))
        assert DM.command_vec(tab, "LOG_A", "tokA") == [1, 0, 0, 0] and DM.command_vec(tab, "LOG_B", "tokC") == [0, 0, 1, 0]
        try:
            DM.command_vec(tab, "LOG_A", "tokMISSING")
        except RuntimeError as e:
            assert "no command" in str(e)
        else:
            raise AssertionError("a token with NO command was answered (a default command): the refusal is gone")
        for bad in ({"LOG_A/t": [1, 1, 0, 0]}, {"LOG_A/t": [0, 0, 0, 0]}, {"t": [1, 0, 0, 0]}, {}):
            q = td / "bad.json"
            q.write_text(json.dumps(bad), encoding="utf-8")
            try:
                DM.load_command_table(str(q))
            except SystemExit:
                continue
            raise AssertionError(f"a malformed table was accepted: {bad}")
        try:
            DM.load_command_table(str(p), sha256="0" * 64)
        except SystemExit as e:
            assert "sha256" in str(e)
        else:
            raise AssertionError("a wrong sha256 pin was accepted")


@check("G4 srttc_s0_dump: command-table helpers load, look up, and REFUSE a missing token / malformed row / wrong sha256 pin")
def t_g4_dump_helpers():
    g4_dump_helpers(_load_mod(DUMP_PY, "srttc_s0_dump_g4"))


@check("G4 RED: a mutant dump that answers a missing token with a default command fails the helper check")
def t_g4_dump_helpers_red():
    src = DUMP_PY.read_text(encoding="utf-8")
    old = '        raise RuntimeError(f"token {tok} (log {log}): no command in {table[\'path\']}: a command-mode checkpoint is never run on a default command")'
    assert src.count(old) == 1
    mut = _load_mod(DUMP_PY, "srttc_s0_dump_g4_mut", src.replace(old, "        return [0, 1, 0, 0]"))
    must_fail(g4_dump_helpers, mut)


def _fake_dump_env(nav_mode):
    """stub every heavy import of srttc_s0_dump.main so it can run to its command-source decision; returns the sentinel raised right after it"""
    class Stop(Exception):
        pass

    class Cfg:
        pass

    Cfg.nav_mode = nav_mode

    class Model:
        cfg = Cfg()

        class score_head:
            @staticmethod
            def register_forward_hook(f):
                raise Stop("passed the command-source decision")

    class FakePlanner:
        def __init__(self, **kw):
            self.model, self.trained, self.per_sample_calib, self.device, self.select, self.rule = Model(), True, True, "cpu", "best", "navsim_v1"
            self.sanitize_goal, self.goal_fix, self.repair_last_heading = False, "pdm_route", True
    mods = {}
    for n in ("torch", "navtrain_scenarios", "planner", "refe_navtest_seam", "nuplan", "nuplan.planning", "nuplan.planning.simulation", "nuplan.planning.simulation.planner",
              "nuplan.planning.simulation.planner.abstract_planner"):
        mods[n] = types.ModuleType(n)
    mods["planner"].REFePlanner = FakePlanner
    mods["planner"].FrameResolver = lambda **kw: types.SimpleNamespace(**kw)
    mods["refe_navtest_seam"].to_navsim = lambda x: x
    mods["refe_navtest_seam"].set_driving_command = lambda *a, **k: True
    mods["nuplan.planning.simulation.planner.abstract_planner"].PlannerInitialization = object
    return mods, Stop


def run_dump_main(DM, commands, nav_mode="command"):
    mods, Stop = _fake_dump_env(nav_mode)
    saved = {n: sys.modules.get(n) for n in mods}
    sys.modules.update(mods)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        plan = {"tokens": ["tokA"], "token_log": {"tokA": "LOG_A"}, "token_root": {"tokA": "R"}, "db": {"LOG_A": str(td / "LOG_A.db")}}
        (td / "plan.json").write_text(json.dumps(plan))
        argv = ["srttc_s0_dump.py", "--ckpt", "x.pt", "--plan", str(td / "plan.json"), "--out", str(td / "o.npz"), "--device", "cpu", "--lean"]
        if commands:
            argv += ["--commands", str(commands)]
        old_argv, old_env = sys.argv, {k: os.environ.pop(k, None) for k in ("SRTTC_COMMANDS_JSON", "SRTTC_COMMANDS_SHA256")}
        sys.argv = argv
        try:
            try:
                rc = DM.main()
                return {"rc": rc, "passed_decision": False}
            except Stop:
                return {"rc": None, "passed_decision": True}
        finally:
            sys.argv = old_argv
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m


@check("G4 srttc_s0_dump.main: a command-mode checkpoint with NO source is still REFUSED (rc 2); with a source it passes the decision; a goal-mode checkpoint is untouched")
def t_g4_dump_main_decision():
    DM = _load_mod(DUMP_PY, "srttc_s0_dump_g4_main")
    with tempfile.TemporaryDirectory() as td:
        src = _small_table(Path(td))
        r0 = run_dump_main(DM, None)
        assert r0 == {"rc": 2, "passed_decision": False}, r0                   # the refusal kept
        r1 = run_dump_main(DM, src)
        assert r1["passed_decision"] is True, r1                                 # a source: past the refusal
        r2 = run_dump_main(DM, None, nav_mode="goal")
        assert r2["passed_decision"] is True, r2                                 # goal mode: the landed behaviour, no source needed


@check("G4 RED: a mutant dump that drops the refusal (accepts command-mode with no source) fails the decision check")
def t_g4_dump_main_decision_red():
    src = DUMP_PY.read_text(encoding="utf-8")
    old = "        if not a.commands:"
    old2 = "        cmd_table = load_command_table(a.commands, a.commands_sha256)"
    assert src.count(old) == 1 and src.count(old2) == 1
    mut_src = src.replace(old, '        if a.commands == "NEVER":').replace(old2, '        cmd_table = {"path": "-", "sha256": "0" * 64, "rows": {}} if not a.commands else load_command_table(a.commands, a.commands_sha256)')
    mut = _load_mod(DUMP_PY, "srttc_s0_dump_g4_main_mut", mut_src)

    def go():
        r0 = run_dump_main(mut, None)
        assert r0 == {"rc": 2, "passed_decision": False}, r0
    must_fail(go)


@check("G4 srttc_s0_dump: the per-token hop sits BEFORE planner.infer in the loop and calls the seam's own set_driving_command with the table's one-hot (static; the hop itself is MEASURED in the dry run and in the e2e CPU run)")
def t_g4_dump_hop_static():
    src = DUMP_PY.read_text(encoding="utf-8")
    i_hop = src.index("set_driving_command(planner, tok, {\"ego_statuses\": [{\"driving_command\": command_vec(cmd_table, log, tok)}]})")
    i_inf = src.index("traj, _score, k = planner.infer(ego, img)")
    assert i_hop < i_inf


def _bcb():
    return _load_mod(BCB_PY, "build_candidate_bank_g4")


@check("G4 build_candidate_bank: S0_PINS accepts the landed AND the G4 srttc_s0_dump.py, refuses any third form")
def t_g4_bcb_pins():
    BB = _bcb()
    if not Path(BB.S0_CODE).exists():
        RESULTS.append(("G4 build_candidate_bank pins: SKIPPED (no dev-box S0 code dir)", True, ""))
        return
    got = BB.check_s0_pins()                                                      # the landed dev-box S0 code
    assert got["srttc_s0_dump.py"] == BB.S0_DUMP_LANDED_SHA256
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "s0code"
        shutil.copytree(BB.S0_CODE, d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(DUMP_PY, d / "srttc_s0_dump.py")
        assert BB.check_s0_pins(str(d))["srttc_s0_dump.py"] == BB.S0_DUMP_G4_SHA256
        (d / "srttc_s0_dump.py").write_bytes(DUMP_PY.read_bytes() + b"# a third form\r\n")
        try:
            BB.check_s0_pins(str(d))
        except BB.BuildError as e:
            assert "srttc_s0_dump.py" in str(e)
        else:
            raise AssertionError("a third form of the dump was accepted")


@check("G4 build_candidate_bank.check_command_source: P6's real commands.json covers ALL 8,922 S0 plan tokens; a table missing one / a non-one-hot row / a wrong sha256 is refused")
def t_g4_bcb_command_source():
    BB = _bcb()
    if not (P6_COMMANDS.exists() and S0_PLAN.exists()):
        raise AssertionError(f"{P6_COMMANDS} or {S0_PLAN} absent: the coverage control cannot run")
    plan_tokens = json.load(open(S0_PLAN, encoding="utf-8"))["tokens"]
    rec = BB.check_command_source(str(P6_COMMANDS), None, plan_tokens)
    assert rec["plan_tokens_covered"] == len(plan_tokens) == 8922 and rec["rows"] == 103037, rec
    tab = json.load(open(P6_COMMANDS, encoding="utf-8"))
    tk, lg = next(iter(plan_tokens.items()))
    with tempfile.TemporaryDirectory() as td:
        miss = dict(tab)
        del miss[f"{lg}/{tk}"]
        p = Path(td) / "miss.json"
        p.write_text(json.dumps(miss))
        try:
            BB.check_command_source(str(p), None, plan_tokens)
        except BB.BuildError as e:
            assert "lacks 1 of the 8922" in str(e)
        else:
            raise AssertionError("a table missing a plan token was accepted")
        bad = dict(tab)
        bad[f"{lg}/{tk}"] = [1, 1, 0, 0]
        p.write_text(json.dumps(bad))
        try:
            BB.check_command_source(str(p), None, plan_tokens)
        except BB.BuildError as e:
            assert "non-one-hot" in str(e)
        else:
            raise AssertionError("a non-one-hot row was accepted")
    try:
        BB.check_command_source(str(P6_COMMANDS), "0" * 64, plan_tokens)
    except BB.BuildError as e:
        assert "sha256" in str(e)
    else:
        raise AssertionError("a wrong sha256 was accepted")


def g4_apply_cases(BB):
    cand_src = {"commands": {"path": str(P6_COMMANDS), "sha256": BB.sha256_file(str(P6_COMMANDS))} if P6_COMMANDS.exists() else None,
                "s0_code_pins": {"srttc_s0_dump.py": BB.S0_DUMP_G4_SHA256}}
    # command-mode + source + the G4 dump: exported
    env = {}
    r = BB.apply_command_source(cand_src, "command", env)
    assert r["commands_exported"] and env["SRTTC_COMMANDS_JSON"] == cand_src["commands"]["path"]
    # command-mode, NO source: REFUSED (the kept refusal)
    for c in ({**cand_src, "commands": None},):
        try:
            BB.apply_command_source(c, "command", {})
        except BB.BuildError as e:
            assert "needs a command source" in str(e)
        else:
            raise AssertionError("a command-mode checkpoint with NO source was accepted")
    # command-mode + source + the LANDED dump: refused (it would refuse at the GPU, after the lock)
    try:
        BB.apply_command_source({**cand_src, "s0_code_pins": {"srttc_s0_dump.py": BB.S0_DUMP_LANDED_SHA256}}, "command", {})
    except BB.BuildError as e:
        assert "G4 srttc_s0_dump.py" in str(e)
    else:
        raise AssertionError("command-mode with the landed dump was accepted")
    # goal-mode: the source is ignored and nothing is exported, even if a stale export is present
    env = {"SRTTC_COMMANDS_JSON": "stale", "SRTTC_COMMANDS_SHA256": "stale"}
    r = BB.apply_command_source({**cand_src, "s0_code_pins": {"srttc_s0_dump.py": BB.S0_DUMP_LANDED_SHA256}}, "goal", env)
    assert r["commands_exported"] is False and not env


@check("G4 build_candidate_bank.apply_command_source: command-mode + source exports it; NO source is REFUSED; the landed dump is refused for command-mode; goal-mode exports nothing")
def t_g4_bcb_apply():
    if not P6_COMMANDS.exists():
        raise AssertionError(f"{P6_COMMANDS} absent")
    g4_apply_cases(_bcb())


@check("G4 RED: a mutant that exports a default / drops the no-source refusal fails the apply cases")
def t_g4_bcb_apply_red():
    src = BCB_PY.read_text(encoding="utf-8")
    old = '    if nav_mode == "command":\n        if not src:'
    assert src.count(old) == 1
    mut = _load_mod(BCB_PY, "build_candidate_bank_g4_mut", src.replace(old, '    if nav_mode == "command" and src:\n        if not src:'))
    must_fail(g4_apply_cases, mut)


@check("G4 build_candidate_bank.ckpt_nav_mode reads meta.nav_mode: the epoch-6 P6 file reads 'command'; a goal-mode file reads 'goal' (skipped, stated, if the files are absent)")
def t_g4_bcb_nav_mode():
    BB = _bcb()
    if not P6_EPOCH6.exists():
        RESULTS.append(("G4 ckpt_nav_mode: SKIPPED (epoch-6 checkpoint absent)", True, ""))
        return
    assert BB.ckpt_nav_mode(str(P6_EPOCH6)) == "command"
    if Path(BB.P2_CKPT).exists():
        assert BB.ckpt_nav_mode(BB.P2_CKPT) == "goal"


def collect():
    return [v for k, v in sorted(globals().items()) if k.startswith("t_") and hasattr(v, "check_name")]


def main(argv=None) -> int:
    slow = "--slow" in (argv or sys.argv[1:])
    for t in collect():
        t()
    if slow:
        for t in [v for k, v in sorted(globals().items()) if k.startswith("slow_") and hasattr(v, "check_name")]:
            t()
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, why in RESULTS:
        print(("PASS " if ok else "FAIL ") + name + (f"  <-- {why}" if why else ""))
    print(f"ZZP6GAPS {'PASS' if npass == len(RESULTS) else 'FAIL'} {npass}/{len(RESULTS)}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
