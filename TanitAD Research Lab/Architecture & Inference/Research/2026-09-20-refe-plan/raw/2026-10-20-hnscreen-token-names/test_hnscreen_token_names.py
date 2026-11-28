#!/usr/bin/env python3
"""HNSCREEN-TOKEN-NAMES (BACKLOG; launch blocker of the P6-HN screen): the names the screen chain reads G14 / G15 under must be checks a REAL launch-gate token carries.
Run:  PYTHONIOENCODING=utf-8 python raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py   ->   ZZTEST_PASS <n>   (prints the mutation table).

THE DEFECT THIS PINS.  `hnscreen_devbox.py` defaulted `gate.token_check_names` to {"G14": "G14", "G15": "G15"}; `token_checks()` returns False on a missing key; `assemble_gates` reads G14 / G15 from it.
A REAL token (launch_gate.PROFILES["refe_hnscreen"]) carries only G-HYG, G-DVB, G-PREFLIGHT, G-SUITE (+ G-CLOCK as na_ok): G14 / G15 are judged INSIDE G-PREFLIGHT.  So a real PASS token read
G14 = G15 = False and the screen would have ended SCREEN-INVALID after its whole GPU spend.  The chain tests passed because their STUB token wrote checks named G14 / G15 ("built, tested,
unreachable from its caller").  Nothing here copies the profile: the REAL `launch_gate` is imported and its PROFILES table is read, and the token is minted by the REAL `gate_token.mint`.

WHAT IS PROVEN (against the module under test; every mutation below edits hnscreen_devbox.py SOURCE, each snippet exactly once, and must turn the named checks RED):
  A  the INVARIANT: every VALUE of the chain's effective `gate.token_check_names` is in `launch_gate.PROFILES[cfg.gate.profile]["checks"]` (and the keys are exactly G14 / G15).  It holds for the
     default config; a config override still wins (and merges key by key); a BAD override is caught by the same function.
  B  the two RED arms of the invariant itself: (1) the OLD default {"G14": "G14", "G15": "G15"} fails it; (2) a profile whose checks lack G-PREFLIGHT fails it.
  C  the invariant is not only membership (G-HYG would also be a member): the named check is the one whose evidence function in launch_gate calls judge_g14 AND judge_g15 (read from its AST), and
     the argv of the screen (--hn-bank / --ctx-cache) obliges G14 and G15 (gate_checks.scorer_required_gates).
  D  the REAL-TOKEN test: a token minted by gate_token.mint with the REAL profile's check list (every check PASS) makes `assemble_gates` read G14 = G15 = True (all ten gates True, seeds 0-1);
     with G-PREFLIGHT FAIL / MISSING / REJECTED / absent it reads G14 = G15 = False; G15 still needs its own evidence file; the other gates are untouched by the token names.
  E  the chain test's stub token carries exactly the checks of a real token, and the OLD stub list ([G14, G15]) does not.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("HNTOK_PKG") or os.path.dirname(os.path.dirname(HERE))          # <...>/2026-09-20-refe-plan
REFE = os.path.join(PKG, "refe")
sys.path.insert(0, REFE)
import gate_checks as C  # noqa: E402  -- the REAL checks
import gate_token as GT  # noqa: E402  -- the REAL token minting
import launch_gate as L  # noqa: E402  -- the REAL profile table
SRC_PATH = os.path.join(REFE, "hnscreen_devbox.py")
SEEDS = (0, 1)
KEY = b"hnscreen-token-names-test-key-0123456789"
OLD_PAIR = {"G14": "G14", "G15": "G15"}


def load_module(src, tag):
    mod = types.ModuleType(f"hnscreen_devbox_tn_{tag}")
    mod.__file__ = SRC_PATH
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"hnscreen_devbox_tn_{tag}.py", "exec"), mod.__dict__)    # noqa: S102 -- the module under test, possibly mutated
    return mod


def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                                            # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("HNT_DEBUG"):
            import traceback
            traceback.print_exc()
            print("  [check raised]", name, type(e).__name__, str(e)[:300])
    if not ok:
        bad.append(name)


# ====================================================================================================================== the invariant (the function the issue asked for)
def invariant_problems(M, doc=None, profiles=None):
    """[] iff every VALUE of the chain's EFFECTIVE `gate.token_check_names` (defaults merged with `doc`, as Cfg does) is a check of `PROFILES[cfg.gate.profile]['checks']`.
    The keys must be exactly the two gates the chain reads.  `profiles` defaults to the REAL launch_gate.PROFILES (never a copy)."""
    profiles = L.PROFILES if profiles is None else profiles
    cfg = M.Cfg(doc or {}, "x")
    names, profile = cfg.get("gate.token_check_names"), cfg.get("gate.profile")
    if profile not in profiles:
        return [f"gate.profile {profile!r} is not a launch_gate profile"]
    checks = tuple(profiles[profile]["checks"])
    if not isinstance(names, dict) or set(names) != {"G14", "G15"}:
        return [f"token_check_names keys are {sorted(names) if isinstance(names, dict) else names!r}, not exactly G14 / G15"]
    return [f"{gate} is read from token check {v!r}, which profile {profile} does not mint (its checks: {list(checks)})" for gate, v in sorted(names.items()) if v not in checks]


def preflight_judge_names():
    """the check name(s) launch_gate's evidence function emits for the function that calls judge_g14 AND judge_g15 -- read from the AST of the REAL launch_gate.py"""
    tree = ast.parse(open(L.__file__, encoding="utf-8").read())
    out = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        called = {(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)) for n in ast.walk(fn) if isinstance(n, ast.Call)}
        if {"judge_g14", "judge_g15"} <= called:
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("evidence", "verdict") and n.args and isinstance(n.args[0], ast.Constant):
                    out.add(n.args[0].value)
    return out


# ====================================================================================================================== a real token, a real chain object
def real_check_names(profile="refe_hnscreen"):
    prof = L.PROFILES[profile]
    return tuple(prof["checks"]) + tuple(prof.get("na_ok", ()))


def mint_real_token(out_dir, *, profile="refe_hnscreen", status=None, drop=()):
    """the REAL gate_token.mint with `checks` built the way launch_gate.finalize builds them: one entry per profile check (+ na_ok), status PASS.  `status` = {check: status} overrides, `drop` removes keys"""
    prof = L.PROFILES[profile]
    checks = {}
    for c in real_check_names(profile):
        if c in drop:
            continue
        checks[c] = {"status": (status or {}).get(c, "PASS"), "evidence_sha256": "0" * 64, "finished_utc": GT.utc_now(), "reasons": []}
        if c in prof.get("na_ok", ()):
            checks[c]["na_reason"] = "NA by the profile"
    # a real gate never writes a PASS_ file for a token whose check is not PASS (mint names it FAIL_*, which the chain does not even look at).  For the arms that carry a non-PASS / missing check
    # `verdict_override="PASS"` keeps the PASS_ file name, so the chain's OWN status read (token_checks) is what the arm tests -- defence in depth, not the gate's path.
    odd = bool(status) or bool(drop)
    verdict, path, tok = GT.mint(profile=prof["name"], commit="a" * 40, tree_man={"refe/x.py": "0" * 64}, argv=["--seed", "0"], trainer=prof["trainer"], data={}, checks=checks, reasons=[],
                                 key=KEY, out_dir=out_dir, gate_inputs={}, gate_self_sha="", verdict_override="PASS" if odd else None)
    return str(path), tok


def build_chain(M, root, *, cfg_doc=None, token_status=None, token_drop=(), seeds=SEEDS, g15_evidence=True, token_for=None, per_seed=None):
    """a Chain over a state dir that holds exactly what `assemble_gates` reads: the gate evidence, the P-stage marks and one REAL minted token per pass"""
    ch = M.Chain(M.Cfg(cfg_doc or {}, "x"), root, door=object(), runner=object(), printer=lambda *_: None)
    for gid in ("I1", "I2", "I3", "I6") + (("G15",) if g15_evidence else ()):
        M.jdump({"ok": True}, ch.gate_path(gid))
    for s in seeds:
        for gid in M.PER_PASS_GATES:
            M.jdump({"ok": True}, ch.gate_path(gid, s))
        M.jdump({"schema": M.SCHEMA, "stage": f"P{s}", "status": "DONE", "fp": "x", "utc": "x", "outputs": {}, "extra": {"token_verified": True}}, ch.marks.path(f"P{s}"))
        if token_for is None or s in token_for:
            kw = {"status": token_status, "drop": token_drop, **((per_seed or {}).get(s) or {})}
            mint_real_token(f"{ch.pass_dir(s)}/lg", **kw)
    return ch


def gates_of(M, tmp, tag, **kw):
    ch = build_chain(M, os.path.join(tmp, tag), **kw)
    return ch.assemble_gates(kw.get("seeds", SEEDS))


# ====================================================================================================================== the checks
def A_default_names_are_checks_of_the_real_profile(M, tmp):
    cfg = M.Cfg({}, "x")
    return (cfg.get("gate.profile") == "refe_hnscreen" and invariant_problems(M) == [] and cfg.get("gate.token_check_names") == {"G14": "G-PREFLIGHT", "G15": "G-PREFLIGHT"}
            and all(v in L.PROFILES["refe_hnscreen"]["checks"] for v in cfg.get("gate.token_check_names").values()))


def A_a_config_override_wins_and_merges_key_by_key(M, tmp):
    cfg = M.Cfg({"gate": {"token_check_names": {"G14": "G-DVB"}}}, "x")
    n = cfg.get("gate.token_check_names")
    both = M.Cfg({"gate": {"token_check_names": {"G14": "G-DVB", "G15": "G-SUITE"}}}, "x").get("gate.token_check_names")
    return n == {"G14": "G-DVB", "G15": "G-PREFLIGHT"} and both == {"G14": "G-DVB", "G15": "G-SUITE"}


def A_a_bad_override_is_caught_by_the_same_invariant(M, tmp):
    bad = invariant_problems(M, {"gate": {"token_check_names": {"G14": "G-NOPE"}}})
    other = invariant_problems(M, {"gate": {"profile": "no_such_profile"}})
    keys = invariant_problems(M, {"gate": {"token_check_names": {"G16": "G-PREFLIGHT"}}})
    return len(bad) == 1 and "G14" in bad[0] and "G-NOPE" in bad[0] and len(other) == 1 and len(keys) == 1


def A_the_override_reaches_assemble_gates(M, tmp):
    """the override is not decorative: G14 mapped to a check the token FAILS reads False while G15 (still G-PREFLIGHT) reads True"""
    g = gates_of(M, tmp, "A5", cfg_doc={"gate": {"token_check_names": {"G14": "G-SUITE"}}}, token_status={"G-SUITE": "FAIL"})
    return g["G14"] is False and g["G15"] is True


def B_RED_arm_1_the_old_default_fails_the_invariant(M, tmp):
    p = invariant_problems(M, {"gate": {"token_check_names": dict(OLD_PAIR)}})
    return len(p) == 2 and all("does not mint" in x for x in p) and any(x.startswith("G14") for x in p) and any(x.startswith("G15") for x in p)


def B_RED_arm_2_a_profile_without_G_PREFLIGHT_fails_the_invariant(M, tmp):
    real = L.PROFILES["refe_hnscreen"]
    lacking = {**L.PROFILES, "refe_hnscreen": {**real, "checks": tuple(c for c in real["checks"] if c != "G-PREFLIGHT")}}
    p = invariant_problems(M, profiles=lacking)
    return len(p) == 2 and "G-PREFLIGHT" not in lacking["refe_hnscreen"]["checks"] and invariant_problems(M) == [] and all("G-PREFLIGHT" in x for x in p)


def B_control_the_real_profile_has_the_checks_the_issue_names(M, tmp):
    """the premise, read from the REAL profile: G14 / G15 are NOT checks of it, G-PREFLIGHT is, and so is nothing else that would judge them"""
    chk = L.PROFILES["refe_hnscreen"]["checks"]
    return "G14" not in chk and "G15" not in chk and "G-PREFLIGHT" in chk and tuple(chk) == tuple(L.PROFILES["refe_rls1"]["checks"]) and "G-CLOCK" in L.PROFILES["refe_hnscreen"]["na_ok"]


def C_the_named_check_is_the_one_that_judges_G14_and_G15(M, tmp):
    judged = preflight_judge_names()
    names = M.Cfg({}, "x").get("gate.token_check_names")
    ns = types.SimpleNamespace(lane_labels_train="", lane_labels_heldout="", pdm_labels_train="", pdm_labels_heldout="", lane_head=False, epo=False, epo_con="off", aux_drv=False,
                               hn_bank="bank.npz", ctx_cache="cache/")
    req = C.scorer_required_gates(ns)
    off = C.scorer_required_gates(types.SimpleNamespace(**{**vars(ns), "hn_bank": "", "ctx_cache": ""}))
    return (judged == {"G-PREFLIGHT"} and set(names.values()) <= judged and "G14" in req and "G15" in req and "G14" not in off and "G15" not in off
            and "refe_hnscreen" in L.PROFILES and "G-PREFLIGHT" in L.PROFILES["refe_hnscreen"]["checks"])


def D_a_real_token_reads_G14_and_G15_true(M, tmp):
    g = gates_of(M, tmp, "D1")
    return sorted(g) == sorted(M.GATE_IDS) and g["G14"] is True and g["G15"] is True and all(g.values())


def D_the_token_in_use_is_a_real_mint(M, tmp):
    """control for D: the token the chain reads is the REAL mint's (signed, kind launch-gate-token, PASS) and carries exactly the profile's checks -- and NO check named G14 / G15"""
    ch = build_chain(M, os.path.join(tmp, "D2"))
    tok = json.load(open(ch.token_file(0), encoding="utf-8"))
    return (tok["kind"] == "launch-gate-token" and tok["verdict"] == "PASS" and GT.signature_ok(tok, KEY) and set(tok["checks"]) == set(real_check_names())
            and "G14" not in tok["checks"] and "G15" not in tok["checks"] and os.path.basename(ch.token_file(0)).startswith("PASS_"))


def D_G_PREFLIGHT_not_PASS_reads_G14_and_G15_false(M, tmp):
    out = []
    for i, st in enumerate(("FAIL", "MISSING", "REJECTED", "INCOMPLETE")):
        g = gates_of(M, tmp, f"D3{i}", token_status={"G-PREFLIGHT": st})
        out.append(g["G14"] is False and g["G15"] is False and g["I1"] and g["I4"] and g["launch_gate"])         # only the preflight-borne gates fall
    return all(out)


def D_a_token_without_G_PREFLIGHT_reads_G14_and_G15_false(M, tmp):
    g = gates_of(M, tmp, "D4", token_drop=("G-PREFLIGHT",))
    return g["G14"] is False and g["G15"] is False and g["I1"] is True


def D_one_bad_pass_is_enough(M, tmp):
    """G14 / G15 are `all` over the passes: seed 0 good + seed 1 with a non-PASS G-PREFLIGHT -> False; seed 0 good + NO token for seed 1 -> False"""
    g1 = gates_of(M, tmp, "D5", per_seed={1: {"status": {"G-PREFLIGHT": "FAIL"}}})
    g2 = gates_of(M, tmp, "D6", token_for=(0,))
    return g1["G14"] is False and g1["G15"] is False and g2["G14"] is False and g2["G15"] is False and g1["I1"] is True


def D_G15_still_needs_its_own_evidence_file(M, tmp):
    g = gates_of(M, tmp, "D7", g15_evidence=False)
    return g["G15"] is False and g["G14"] is True and g["I6"] is True


def D_a_failed_token_stays_failed_for_the_launch_gate_row(M, tmp):
    """the launch_gate row is the P-stage's token_verified mark, not a token check: a token with every check PASS but a mark that is not True reads launch_gate False, G14 / G15 still True"""
    ch = build_chain(M, os.path.join(tmp, "D8"))
    M.jdump({"schema": M.SCHEMA, "stage": "P1", "status": "DONE", "fp": "x", "utc": "x", "outputs": {}, "extra": {"token_verified": False}}, ch.marks.path("P1"))
    g = ch.assemble_gates(SEEDS)
    return g["launch_gate"] is False and g["G14"] is True and g["G15"] is True


def E_the_chain_test_stub_token_carries_the_checks_of_a_real_token(M, tmp):
    sys.path.insert(0, REFE)
    import test_hnscreen_chain as TC
    stub = set(TC.REAL_TOKEN_CHECKS)
    return stub == set(real_check_names()) and "G14" not in stub and "G15" not in stub and stub != set(OLD_PAIR)


CHECKS = [
    ("A_default_names_are_checks_of_the_real_profile", A_default_names_are_checks_of_the_real_profile),
    ("A_a_config_override_wins_and_merges_key_by_key", A_a_config_override_wins_and_merges_key_by_key),
    ("A_a_bad_override_is_caught_by_the_same_invariant", A_a_bad_override_is_caught_by_the_same_invariant),
    ("A_the_override_reaches_assemble_gates", A_the_override_reaches_assemble_gates),
    ("B_RED_arm_1_the_old_default_fails_the_invariant", B_RED_arm_1_the_old_default_fails_the_invariant),
    ("B_RED_arm_2_a_profile_without_G_PREFLIGHT_fails_the_invariant", B_RED_arm_2_a_profile_without_G_PREFLIGHT_fails_the_invariant),
    ("B_control_the_real_profile_has_the_checks_the_issue_names", B_control_the_real_profile_has_the_checks_the_issue_names),
    ("C_the_named_check_is_the_one_that_judges_G14_and_G15", C_the_named_check_is_the_one_that_judges_G14_and_G15),
    ("D_a_real_token_reads_G14_and_G15_true", D_a_real_token_reads_G14_and_G15_true),
    ("D_the_token_in_use_is_a_real_mint", D_the_token_in_use_is_a_real_mint),
    ("D_G_PREFLIGHT_not_PASS_reads_G14_and_G15_false", D_G_PREFLIGHT_not_PASS_reads_G14_and_G15_false),
    ("D_a_token_without_G_PREFLIGHT_reads_G14_and_G15_false", D_a_token_without_G_PREFLIGHT_reads_G14_and_G15_false),
    ("D_one_bad_pass_is_enough", D_one_bad_pass_is_enough),
    ("D_G15_still_needs_its_own_evidence_file", D_G15_still_needs_its_own_evidence_file),
    ("D_a_failed_token_stays_failed_for_the_launch_gate_row", D_a_failed_token_stays_failed_for_the_launch_gate_row),
    ("E_the_chain_test_stub_token_carries_the_checks_of_a_real_token", E_the_chain_test_stub_token_carries_the_checks_of_a_real_token),
]


def suite(M, only=None):
    tmp = tempfile.mkdtemp(prefix="hntok_")
    bad = []
    try:
        for name, fn in CHECKS:
            if only is None or name in only:
                check(bad, name, lambda fn=fn: fn(M, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


D_ALL = ["D_a_real_token_reads_G14_and_G15_true"]
# each mutation: (name, [(old, new)], [checks that must go RED])
OLD = 'TOKEN_CHECK_NAMES = {"G14": "G-PREFLIGHT", "G15": "G-PREFLIGHT"}\n'
MUTATIONS = [
    ("DEFAULTS: gate.token_check_names is the pre-fix G14 / G15 pair (the defect itself)",
     [(OLD, 'TOKEN_CHECK_NAMES = {"G14": "G14", "G15": "G15"}\n')],
     ["A_default_names_are_checks_of_the_real_profile", "C_the_named_check_is_the_one_that_judges_G14_and_G15", "D_a_real_token_reads_G14_and_G15_true"]),
    ("DEFAULTS: only G15 keeps the old name (a half fix)",
     [(OLD, 'TOKEN_CHECK_NAMES = {"G14": "G-PREFLIGHT", "G15": "G15"}\n')],
     ["A_default_names_are_checks_of_the_real_profile", "D_a_real_token_reads_G14_and_G15_true"]),
    ("DEFAULTS: both gates are read from G-HYG (a real check, so MEMBERSHIP alone passes; it is the wrong judge)",
     [(OLD, 'TOKEN_CHECK_NAMES = {"G14": "G-HYG", "G15": "G-HYG"}\n')],
     ["C_the_named_check_is_the_one_that_judges_G14_and_G15", "D_G_PREFLIGHT_not_PASS_reads_G14_and_G15_false", "D_a_token_without_G_PREFLIGHT_reads_G14_and_G15_false"]),
    ("assemble_gates: the config override is ignored (the module constant is used)",
     [('        g, names = {}, self.cfg.get("gate.token_check_names")\n', '        g, names = {}, dict(TOKEN_CHECK_NAMES)\n')],
     ["A_the_override_reaches_assemble_gates"]),
    ("token_checks: any status but FAIL counts as PASS",
     [('["checks"][name]["status"] == "PASS"', '["checks"][name]["status"] != "FAIL"')],
     ["D_G_PREFLIGHT_not_PASS_reads_G14_and_G15_false"]),
    ("assemble_gates: G15 ignores its own evidence file",
     [('        g["G15"] = ev_ok(self.gate_path("G15")) and all(', '        g["G15"] = all(')],
     ["D_G15_still_needs_its_own_evidence_file"]),
    ("assemble_gates: G14 / G15 are `any` over the passes, not `all`",
     [('        g["G14"] = all(self.token_checks(', '        g["G14"] = any(self.token_checks(')],
     ["D_one_bad_pass_is_enough"]),
]


def apply_edits(src, edits, name):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:90]!r}")
        src = src.replace(old, new)
    return src


def main():
    src = open(SRC_PATH, encoding="utf-8").read()
    real = suite(load_module(src, "real"))
    print(f"unmutated suite: {len(CHECKS)} checks, {len(real)} failing {real}", flush=True)
    if real:
        print(f"ZZTEST_FAIL the unmutated hnscreen_devbox.py FAILS {len(real)} of its own checks")
        return 1
    problems, n = [], len(CHECKS)
    print(f"{'mutation':100s} caught-by (failing checks)")
    for k, (name, edits, expect) in enumerate(MUTATIONS, 1):
        try:
            m_src = apply_edits(src, edits, name)
            ast.parse(m_src)
            failing = suite(load_module(m_src, f"m{k}"), only=set(expect))
        except AssertionError as e:
            problems.append(str(e))
            continue
        except (Exception, SystemExit) as e:                           # noqa: BLE001 -- a mutated source that breaks the suite's own setup is RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        if not failing:
            problems.append(f"mutation {name!r} stayed GREEN: the suite cannot see it")
            continue
        for c in expect:
            if c not in failing and not any(f.startswith("suite setup raised") for f in failing):
                problems.append(f"mutation {name!r}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        print(f"{name[:100]:100s} {', '.join(failing[:2])[:90]}", flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p for p in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  ({len(CHECKS)} checks on the unmutated module, 2 of them the RED arms of the invariant, + {len(MUTATIONS)} RED mutations of hnscreen_devbox.py; the REAL launch_gate.PROFILES and gate_token.mint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
