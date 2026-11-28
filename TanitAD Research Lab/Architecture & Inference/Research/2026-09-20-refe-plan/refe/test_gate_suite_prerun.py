#!/usr/bin/env python3
"""Tests for the G-SUITE PRE-RUN evidence: `launch_gate.py suite-prerun` and `launch_gate.py run --suite-evidence`
(gate_token.make_suite_evidence / verify_suite_evidence, launch_gate.suite_check, run_checks, finalize, cmd_run).  Wiring: raw/2026-10-16-gate-suite-prerun/WIRING.md.

WHAT IS PROVED.  G-SUITE may ACCEPT a pre-computed result only if every binding re-measures equal at run time; every other case RUNS THE SUITE LIVE (a bad evidence file costs time, never strength).
Each REGRESSION ARM below hands the gate an evidence it must NOT accept (a different tree / profile / suite entry / commit / gate code / environment, a stale or future-dated file, a FAIL verdict,
a skip, a tampered byte, another key, a token, an unreadable file, an active regression arm ...) and requires the live path; its POSITIVE TWIN (the same call with the evidence untouched, and the
same evidence re-signed unchanged) must be ACCEPTED -- so a rejection is the one field's doing.  The live path is proved by a probe script that appends to a log on every run: accepted = the log
did not grow; live = it grew.  Then a SOURCE-MUTATION MATRIX: for each acceptance condition (and for the plumbing from the CLI to it) the line is neutralised in a COPY of refe/ and the arm that
condition guards must flip to ACCEPTED (= the test goes RED).  "Built, tested, and unreachable from its caller" is checked end to end: the arms marked e2e go through `launch_gate.py run`.

A dev-box test.  It is deliberately NOT in any profile's SUITE (G-SUITE must not grow by the thing meant to shorten it).

  python refe/test_gate_suite_prerun.py                     -> ZZTEST_PASS <n>
  python refe/test_gate_suite_prerun.py --no-mutations      the arms only (the matrix is the slow half)
  python refe/test_gate_suite_prerun.py --arm-table OUT.json [--only a,b,c]      INTERNAL: the matrix runs this in a MUTATED COPY of refe/ and judges the table
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.pop("TANITAD_GATE_ARM", None)                  # a regression arm in the caller's env would turn every acceptance into a (correct) rejection
import gate_checks as C  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

PY = sys.executable
N = 0
KEY = object()                                            # "use the world's key" (None is a legitimate argument: no key)


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


PROBE = '''import sys
from pathlib import Path
a = sys.argv[1:]
with open(Path(__file__).with_name("runs.log"), "a", encoding="utf-8") as fh:
    fh.write(" ".join(a) + "\\n")
for i, t in enumerate(a):
    if t == "--touch":
        with open(a[i + 1], "a", encoding="utf-8") as fh:
            fh.write("# touched while the suite ran\\n")
if "--skip" in a:
    print("ZZTEST_SKIP probe skipped on request")
if "--fail" in a:
    sys.exit(1)
print("ZZSYN_PASS")
'''


def mini_pkg(root: Path) -> Path:
    for rel, txt in (("refe/a.py", "print(1)\n"), ("refe/b.py", "print(2)\n"), ("code/c.sh", "echo 3\n"), ("splits/d.yaml", "x: 1\n")):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8", newline="\n")
    return root


def suite_of(probe, *tail) -> list:
    return [{"script": str(probe), "token": "ZZSYN_PASS", "args": ["--one"]}, {"script": str(probe), "token": "ZZSYN_PASS", "args": ["--two", *tail]}]


def synth_profile(suite, name="refe_synth", scorer=False) -> dict:
    base = L.PROFILES["refe_rls1" if scorer else "refe_p2"]
    return {**base, "name": name, "required_flags": {}, "provisional_flags": (), "require_commit": False, "checks": ("G-SUITE",), "na_ok": (), "data_flags": (), "forbidden_flags": (),
            "live_terms": (), "suite": suite, **({"arm_rules": None} if scorer else {})}


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


def utc(offset_s: float = 0.0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_s))


@contextlib.contextmanager
def edited(path: Path, new_text: str | None):
    """Temporarily rewrite (or, with None, delete) a file; always restore it."""
    old = path.read_bytes() if path.exists() else None
    try:
        if new_text is None:
            path.unlink()
        else:
            path.write_text(new_text, encoding="utf-8", newline="\n")
        yield
    finally:
        if old is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(old)


@contextlib.contextmanager
def envvar(name: str, value: str | None):
    old = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


class World:
    """A mini shipped package, two gate keys, a probe script, a synthetic profile (two probe entries) and ONE valid pre-run made through the real CLI."""

    def __init__(self, td: Path):
        self.td = td
        self.pkg = mini_pkg(td / "pkg")
        self.key_file, self.key2_file = td / "k" / "gate.key", td / "k2" / "gate.key"
        self.key = GT.load_key(self.key_file, create=True, forbid_under=[self.pkg])
        self.key2 = GT.load_key(self.key2_file, create=True, forbid_under=[self.pkg])
        self.probe = td / "probe.py"
        self.probe.write_text(PROBE, encoding="utf-8", newline="\n")
        self.prof = synth_profile(suite_of(self.probe))
        self.prof_scorer = synth_profile(suite_of(self.probe), name="refe_synth_scorer", scorer=True)
        self.last_why = None
        self.valid, rc = self.prerun("valid.json")
        assert rc == 0, "the world's own pre-run must pass"

    def runs(self) -> int:
        p = self.td / "runs.log"
        return len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0

    def prerun(self, name, prof=None, key_file=None, commit=None, pkg=None, env_arm=None):
        prof = prof or self.prof
        L.PROFILES[prof["name"]] = prof
        out = self.td / name
        argv = ["suite-prerun", "--profile", prof["name"], "--pkg", str(pkg or self.pkg), "--out", str(out), "--key-file", str(key_file or self.key_file)] + (["--commit", commit] if commit else [])
        with envvar("TANITAD_GATE_ARM", env_arm):
            rc, _ = quiet(L.main, argv)
        return out, rc

    def resign(self, mut, name, key=None, src=None) -> Path:
        d = GT.read_json(src or self.valid)
        mut(d)
        d.pop("hmac_sha256", None)
        d["hmac_sha256"] = GT.sign(d, key or self.key)
        return GT.write_json(self.td / f"{name}.json", d)

    def judge(self, evidence, *, prof=None, commit=None, arm=None, key=KEY, pkg=None):
        ev = quiet(L.suite_check, prof or self.prof, pkg or self.pkg, evidence=evidence, key=self.key if key is KEY else key, commit=commit, arm=arm)[0]
        self.last_why = ev["details"]["suite_source"].get("rejected")
        return ev["details"]["suite_source"]["path"] == "accepted", ev

    def latest_token(self, out_dir: Path) -> dict | None:
        fs = sorted(out_dir.glob("PASS_*.json")) + sorted(out_dir.glob("FAIL_*.json"))
        return GT.read_json(fs[0]) if fs else None

    def e2e(self, tag, *, evidence=None, scorer=False, pkg=None, commit=None):
        """`launch_gate.py run --only G-SUITE [--suite-evidence F]` through the real CLI, run_checks and finalize.  -> (token, live_suite_runs)"""
        prof = self.prof_scorer if scorer else self.prof
        L.PROFILES[prof["name"]] = prof
        out_dir = self.td / f"out_{tag}"
        argv = ["run", "--profile", prof["name"], "--args", "--zz 1", "--pkg", str(pkg or self.pkg), "--out-dir", str(out_dir), "--key-file", str(self.key_file), "--only", "G-SUITE"]
        if evidence:
            argv += ["--suite-evidence", str(evidence)]
        if commit:
            argv += ["--commit", commit]
        before = self.runs()
        quiet(L.main, argv)
        return self.latest_token(out_dir), self.runs() - before


# ------------------------------------------------------------------------------------------ the arms
ARMS: dict = {}                                           # name -> (fn(world) -> accepted?, expected_accepted, e2e?)


def arm(name, accept=False, e2e=False):
    def deco(fn):
        ARMS[name] = (fn, accept, e2e)
        return fn
    return deco


def _set(path, value):
    def m(d):
        cur = d
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value
    return m


ROW = ["suite", "details", "scripts", 0]
RESIGNED = {                                              # name -> (mutator of the evidence body, accepted?).  The body is RE-SIGNED with the right key: only the one field differs.
    "resigned_unchanged": (lambda d: None, True),         # the control for every re-signed arm below: the helper itself yields an acceptable evidence
    "age_11h59": (lambda d: d.__setitem__("created_utc", utc(-(11 * 3600 + 59 * 60))), True),
    "stale_13h": (lambda d: d.__setitem__("created_utc", utc(-13 * 3600)), False),
    "stale_12h_plus": (lambda d: d.__setitem__("created_utc", utc(-(12 * 3600 + 120))), False),
    "future_dated": (lambda d: d.__setitem__("created_utc", utc(3600)), False),
    "created_unparseable": (_set(["created_utc"], "yesterday"), False),
    "verdict_FAIL_only": (_set(["verdict"], "FAIL"), False),
    "status_FAIL_only": (_set(["suite", "status"], "FAIL"), False),
    "reasons_nonempty_only": (_set(["suite", "reasons"], ["x"]), False),
    "script_rc_1_only": (_set(ROW + ["rc"], 1), False),
    "token_not_seen_only": (_set(ROW + ["token_seen"], False), False),
    "skip_in_row_only": (_set(ROW + ["skips"], ["ZZTEST_SKIP x"]), False),
    "row_count_short": (lambda d: d["suite"]["details"]["scripts"].pop(), False),
    "row_script_renamed": (_set(ROW + ["script"], "other.py"), False),
    "check_name_wrong": (_set(["suite", "check"], "G-LIVE"), False),
    "kind_is_token": (_set(["kind"], "launch-gate-token"), False),
    "schema_wrong": (_set(["schema"], GT.GATE_SCHEMA), False),
    "gate_sha_changed": (_set(["binding", "gate_sha256"], "0" * 64), False),
    "gate_checks_sha_changed": (_set(["binding", "gate_checks_sha256"], "0" * 64), False),
    "gate_token_sha_changed": (_set(["binding", "gate_token_sha256"], "0" * 64), False),
    "env_sha_changed": (_set(["binding", "env_sha256"], "0" * 64), False),
    "tree_sha_changed_resigned": (_set(["binding", "tree_sha256"], "0" * 64), False),
    "suite_sha_changed_resigned": (_set(["binding", "suite_sha256"], "0" * 64), False),
    "arm_recorded": (_set(["arm"], "suite_failing_test"), False),
}
def _resigned_arm(name, mut):
    return lambda w: w.judge(w.resign(mut, "r_" + name))[0]


for _n, (_m, _acc) in RESIGNED.items():
    ARMS[_n] = (_resigned_arm(_n, _m), _acc, False)


@arm("valid", accept=True)
def _valid(w):
    return w.judge(w.valid)[0]


@arm("valid_same_commit", accept=True)
def _same_commit(w):
    ev, rc = w.prerun("v_commit.json", commit="c" * 40)
    return rc == 0 and w.judge(ev, commit="c" * 40)[0]


@arm("commit_changed")
def _commit_changed(w):
    ev, _ = w.prerun("v_commit2.json", commit="c" * 40)
    return w.judge(ev, commit="d" * 40)[0]


@arm("commit_none_vs_set")
def _commit_none_vs_set(w):
    return w.judge(w.valid, commit="c" * 40)[0]


@arm("tree_edited_file")
def _tree_edit(w):
    with edited(w.pkg / "refe" / "b.py", "print(99)\n"):
        return w.judge(w.valid)[0]


@arm("tree_added_file")
def _tree_add(w):
    with edited(w.pkg / "refe" / "new.py", "print(0)\n"):
        return w.judge(w.valid)[0]


@arm("tree_removed_file")
def _tree_rm(w):
    with edited(w.pkg / "code" / "c.sh", None):
        return w.judge(w.valid)[0]


@arm("profile_renamed_same_suite")
def _renamed(w):
    return w.judge(w.valid, prof={**w.prof, "name": "refe_synth_other"})[0]


@arm("profile_other_real_profile")
def _other_real(w):
    return w.judge(w.valid, prof=L.PROFILES["refe_p2rep"])[0]


@arm("suite_args_edited")
def _args_edited(w):
    return w.judge(w.valid, prof={**w.prof, "suite": suite_of(w.probe, "--extra")})[0]


@arm("suite_token_edited")
def _token_edited(w):
    s = suite_of(w.probe)
    s[1] = {**s[1], "token": "ZZSYN_PASS2"}
    return w.judge(w.valid, prof={**w.prof, "suite": s})[0]


@arm("suite_entry_dropped")
def _dropped(w):
    return w.judge(w.valid, prof={**w.prof, "suite": suite_of(w.probe)[:1]})[0]


@arm("suite_entry_added")
def _added(w):
    return w.judge(w.valid, prof={**w.prof, "suite": suite_of(w.probe) + [{"script": str(w.probe), "token": "ZZSYN_PASS", "args": ["--three"]}]})[0]


@arm("suite_entries_reordered")
def _reordered(w):
    return w.judge(w.valid, prof={**w.prof, "suite": suite_of(w.probe)[::-1]})[0]


@arm("tampered_one_byte")
def _tampered(w):
    raw = w.valid.read_bytes()
    i = raw.index(b'"seconds": ') + len(b'"seconds": ')
    bad = raw[:i] + bytes([raw[i] ^ 0x01]) + raw[i + 1:]                       # ONE byte of a number that no other check reads
    assert len(bad) == len(raw) and sum(x != y for x, y in zip(raw, bad)) == 1 and isinstance(json.loads(bad.decode("utf-8")), dict)
    p = w.td / "tampered.json"
    p.write_bytes(bad)
    return w.judge(p)[0]


@arm("key_other")
def _key_other(w):
    ev, rc = w.prerun("v_key2.json", key_file=w.key2_file)
    return rc == 0 and w.judge(ev)[0]


@arm("hmac_removed")
def _hmac_removed(w):
    d = GT.read_json(w.valid)
    d.pop("hmac_sha256")
    return w.judge(GT.write_json(w.td / "nohmac.json", d))[0]


@arm("hmac_garbage")
def _hmac_garbage(w):
    d = GT.read_json(w.valid)
    d["hmac_sha256"] = "z" * 64
    return w.judge(GT.write_json(w.td / "garbage.json", d))[0]


@arm("no_key")
def _no_key(w):
    return w.judge(w.valid, key=None)[0]


@arm("token_as_evidence")
def _token_as_evidence(w):
    tree = GT.tree_manifest(w.pkg)
    _, path, _ = GT.mint(profile=w.prof["name"], commit=None, tree_man=tree, argv=["--x"], trainer="refe/train.py", data={}, checks={"G-SUITE": {"status": "PASS"}}, reasons=[],
                         key=w.key, out_dir=w.td / "tok")
    return w.judge(path)[0]


@arm("regression_arm_active")
def _arm_active(w):
    return w.judge(w.valid, arm="suite_failing_test")[0]


@arm("env_var_set_after_prerun")
def _env_after(w):
    with envvar("REFE_ZZ_PROBE", "1"):
        return w.judge(w.valid)[0]


@arm("env_var_set_at_prerun_only")
def _env_before(w):
    with envvar("REFE_ZZ_PROBE", "1"):
        ev, rc = w.prerun("v_env.json")
    return rc == 0 and w.judge(ev)[0]


@arm("env_var_unbound_is_ignored", accept=True)
def _env_ignored(w):
    with envvar("ZZ_NOT_A_BOUND_PREFIX", "1"), envvar("REFE_ZZ_API_TOKEN", "secret-value"):
        return w.judge(w.valid)[0]


@arm("real_prerun_FAIL")
def _real_fail(w):
    prof = synth_profile(suite_of(w.probe, "--fail"), name="refe_synth_fail")
    ev, rc = w.prerun("v_fail.json", prof=prof)
    return rc == 1 and GT.read_json(ev)["verdict"] == "FAIL" and w.judge(ev, prof=prof)[0]


@arm("real_prerun_skip")
def _real_skip(w):
    prof = synth_profile(suite_of(w.probe, "--skip"), name="refe_synth_skip")
    ev, rc = w.prerun("v_skip.json", prof=prof)
    return rc == 1 and GT.read_json(ev)["verdict"] == "FAIL" and w.judge(ev, prof=prof)[0]


@arm("real_prerun_pass_under_arm_env")
def _real_under_arm(w):
    ev, rc = w.prerun("v_under_arm.json", env_arm="suite_failing_test")
    return rc == 1 and GT.read_json(ev)["verdict"] == "FAIL" and w.judge(ev)[0]


@arm("real_prerun_tree_moved_during")
def _real_moved(w):
    victim = w.pkg / "refe" / "a.py"
    prof = synth_profile(suite_of(w.probe, "--touch", str(victim)), name="refe_synth_touch")
    with edited(victim, "print(1)\n"):
        ev, rc = w.prerun("v_touch.json", prof=prof)
        d = GT.read_json(ev)
        got = rc == 1 and d["verdict"] == "FAIL" and any("changed WHILE the suite ran" in r for r in d["reasons"]) and "tree_sha256" in " ".join(d["reasons"])
        victim.write_text("print(1)\n", encoding="utf-8", newline="\n")
        return got and w.judge(ev, prof=prof)[0]


@arm("evidence_missing")
def _missing(w):
    return w.judge(w.td / "no_such_file.json")[0]


@arm("evidence_empty")
def _empty(w):
    p = w.td / "empty.json"
    p.write_bytes(b"")
    return w.judge(p)[0]


@arm("evidence_not_json")
def _not_json(w):
    p = w.td / "notjson.json"
    p.write_text("ZZGATE-PASS", encoding="utf-8")
    return w.judge(p)[0]


@arm("evidence_json_list")
def _json_list(w):
    return w.judge(GT.write_json(w.td / "list.json", [1, 2]))[0]


@arm("evidence_truncated")
def _truncated(w):
    raw = w.valid.read_bytes()
    p = w.td / "trunc.json"
    p.write_bytes(raw[: len(raw) // 2])
    return w.judge(p)[0]


@arm("e2e_main_accepted", accept=True, e2e=True)
def _e2e_main(w):
    tok, live = w.e2e("main_acc", evidence=w.valid)
    src = ((tok or {}).get("checks") or {}).get("G-SUITE", {}).get("suite_source") or {}
    return bool(tok) and tok["verdict"] == "PASS" and live == 0 and src.get("path") == "accepted" and src.get("evidence_sha256") == GT.sha256_file(w.valid)


@arm("e2e_scorer_accepted", accept=True, e2e=True)
def _e2e_scorer(w):
    ev, rc = w.prerun("v_scorer.json", prof=w.prof_scorer)
    tok, live = w.e2e("scorer_acc", evidence=ev, scorer=True)
    src = ((tok or {}).get("checks") or {}).get("G-SUITE", {}).get("suite_source") or {}
    return rc == 0 and bool(tok) and tok["verdict"] == "PASS" and live == 0 and src.get("path") == "accepted" and src.get("evidence_sha256") == GT.sha256_file(ev)


@arm("e2e_ship_manifest_commit_accepted", accept=True, e2e=True)
def _e2e_ship(w):
    pkg = mini_pkg(w.td / "pkg_shipped")
    GT.write_json(pkg / GT.SHIP_MANIFEST, {"commit": "e" * 40, "files": GT.tree_manifest(pkg)})
    ev, rc = w.prerun("v_ship.json", pkg=pkg)
    tok, live = w.e2e("ship_acc", evidence=ev, pkg=pkg)
    src = ((tok or {}).get("checks") or {}).get("G-SUITE", {}).get("suite_source") or {}
    return rc == 0 and GT.read_json(ev)["binding"]["commit"] == "e" * 40 and bool(tok) and tok["verdict"] == "PASS" and tok["binding"]["commit"] == "e" * 40 and live == 0 \
        and src.get("path") == "accepted"


@arm("e2e_main_rejected_falls_back_live", e2e=True)
def _e2e_main_rej(w):
    with edited(w.pkg / "refe" / "b.py", "print(98)\n"):
        tok, live = w.e2e("main_rej", evidence=w.valid)
    src = ((tok or {}).get("checks") or {}).get("G-SUITE", {}).get("suite_source") or {}
    return not (bool(tok) and tok["verdict"] == "PASS" and live == 2 and src.get("path") == "live" and any("code tree differs" in r for r in src.get("rejected", []))
                and src.get("evidence_sha256") == GT.sha256_file(w.valid))


@arm("e2e_scorer_rejected_falls_back_live", e2e=True)
def _e2e_scorer_rej(w):
    ev, _ = w.prerun("v_scorer2.json", prof=w.prof_scorer)
    with edited(w.pkg / "refe" / "b.py", "print(97)\n"):
        tok, live = w.e2e("scorer_rej", evidence=ev, scorer=True)
    src = ((tok or {}).get("checks") or {}).get("G-SUITE", {}).get("suite_source") or {}
    return not (bool(tok) and tok["verdict"] == "PASS" and live == 2 and src.get("path") == "live" and any("code tree differs" in r for r in src.get("rejected", [])))


@arm("e2e_no_flag_is_todays_live_run", e2e=True)
def _e2e_noflag(w):
    tok, live = w.e2e("noflag")
    chk = ((tok or {}).get("checks") or {}).get("G-SUITE") or {}
    return "suite_source" in chk or live != 2 or not tok or tok["verdict"] != "PASS"      # accepted-like = the record appeared, or the suite did not run live


def run_arms(w, only=None, why=None) -> dict:
    out = {}
    for name, (fn, _acc, _e2e) in ARMS.items():
        if only is not None and name not in only:
            continue
        w.last_why = None
        try:
            out[name] = bool(fn(w))
            if why is not None:
                why[name] = w.last_why
        except Exception as e:                            # noqa: BLE001 -- a crash is not an acceptance; the matrix reads it as "differs from expected" which is what a mutation wants
            out[name] = f"ERR {type(e).__name__}: {str(e)[:100]}"
    return out


# ------------------------------------------------------------------------------------------ the source-mutation matrix
LGP, GTP = "launch_gate.py", "gate_token.py"
SCORER_ANCHOR = "emit(suite_check(prof, pkg, evidence=suite_evidence, key=key, commit=commit, arm=arm, rehearsal=rehearsal))"
MAIN_ANCHOR = "emit(suite_check(prof2, pkg, evidence=suite_evidence, key=key, commit=commit, arm=arm, rehearsal=rehearsal))"
MUTATIONS = [                                             # (id, file, anchor [must occur ONCE], replacement, arms that must flip to the other value)
    ("accept_tree_skipped", GTP, 'if b.get("tree_sha256") != want["tree_sha256"]:', "if False:", {"tree_edited_file", "tree_added_file", "tree_removed_file", "tree_sha_changed_resigned"}),
    ("accept_signature_skipped", GTP, "if not signature_ok(ev, key):", "if False:", {"tampered_one_byte", "key_other", "hmac_removed", "hmac_garbage"}),
    ("accept_kind_schema_skipped", GTP, 'if ev.get("kind") != SUITE_EVIDENCE_KIND or ev.get("schema") != SUITE_EVIDENCE_SCHEMA:', "if False:", {"kind_is_token", "schema_wrong"}),
    ("accept_profile_skipped", GTP, 'if b.get("profile") != want["profile"]:', "if False:", {"profile_renamed_same_suite"}),
    ("accept_commit_skipped", GTP, 'if b.get("commit") != want["commit"]:', "if False:", {"commit_changed", "commit_none_vs_set"}),
    ("accept_suite_entries_skipped", GTP, 'if b.get("suite_sha256") != want["suite_sha256"]:', "if False:", {"suite_args_edited", "suite_sha_changed_resigned"}),
    ("accept_gate_code_skipped", GTP, 'if b.get(k) != want[k]:', "if False:", {"gate_sha_changed", "gate_checks_sha_changed", "gate_token_sha_changed"}),
    ("accept_environment_skipped", GTP, 'if b.get("env_sha256") != want["env_sha256"]:', "if False:", {"env_sha_changed", "env_var_set_after_prerun", "env_var_set_at_prerun_only"}),
    ("accept_age_skipped", GTP, "elif t_now - created > max_age_s:", "elif False:", {"stale_13h", "stale_12h_plus"}),
    ("accept_future_skipped", GTP, "elif created - t_now > SUITE_EVIDENCE_FUTURE_S:", "elif False:", {"future_dated"}),
    ("accept_arm_skipped", GTP, 'if arm or ev.get("arm"):', "if False:", {"regression_arm_active", "arm_recorded"}),
    ("accept_verdict_skipped", GTP, 'if ev.get("verdict") != "PASS":', "if False:", {"verdict_FAIL_only"}),
    ("accept_suite_status_skipped", GTP, 'if res.get("status") != "PASS" or res.get("reasons"):', "if False:", {"status_FAIL_only", "reasons_nonempty_only"}),
    ("accept_script_rc_token_skipped", GTP, 'if r.get("rc") != 0 or r.get("token_seen") is not True:', "if False:", {"script_rc_1_only", "token_not_seen_only"}),
    ("accept_skips_skipped", GTP, 'if r.get("skips"):', "if False:", {"skip_in_row_only"}),
    ("accept_row_count_skipped", GTP, "if len(rows) != len(suite):", "if False:", {"row_count_short"}),
    ("accept_row_names_skipped", GTP, 'if not isinstance(r, dict) or r.get("script") != e.get("script", "raw") or r.get("token") != e.get("token"):', "if False:", {"row_script_renamed"}),
    ("accept_check_name_skipped", GTP, 'if res.get("check") != "G-SUITE":', "if False:", {"check_name_wrong"}),
    ("accept_everything", LGP, 'if ok:\n        pre = info["evidence"]', 'if True:\n        pre = info.get("evidence") or GT.read_json(evidence)', {"tree_edited_file", "stale_13h", "verdict_FAIL_only", "tampered_one_byte", "key_other", "profile_renamed_same_suite"}),
    ("noflag_takes_the_acceptance_path", LGP, "    if not evidence:\n        return C.check_suite(prof, arm=arm, rehearsal=rehearsal)\n", "", {"e2e_no_flag_is_todays_live_run"}),
    ("plumb_cmd_run_drops_the_flag", LGP, "suite_evidence=a.suite_evidence, key=key", "suite_evidence=None, key=key", {"e2e_main_accepted", "e2e_scorer_accepted", "e2e_ship_manifest_commit_accepted"}),
    ("plumb_cmd_run_drops_the_key", LGP, "suite_evidence=a.suite_evidence, key=key", "suite_evidence=a.suite_evidence, key=None", {"e2e_main_accepted", "e2e_scorer_accepted"}),
    ("plumb_main_branch_runs_live_suite", LGP, MAIN_ANCHOR, "emit(C.check_suite(prof2, arm=arm, rehearsal=rehearsal))", {"e2e_main_accepted"}),
    ("plumb_scorer_branch_runs_live_suite", LGP, SCORER_ANCHOR, "emit(C.check_suite(prof, arm=arm, rehearsal=rehearsal))", {"e2e_scorer_accepted"}),
    ("plumb_token_forgets_the_path", LGP, 'checks[c]["suite_source"] = src', "pass", {"e2e_main_accepted", "e2e_scorer_accepted"}),
    ("plumb_prerun_commit_ignored", LGP, "commit = _commit_of(pkg, a.commit)\n    arm =", "commit = a.commit\n    arm =", {"e2e_ship_manifest_commit_accepted"}),
]


def one_mutation(ix: int, mut: tuple, td: Path) -> tuple:
    mid, fname, anchor, repl, flips = mut
    dst = td / f"copy_{ix:02d}" / "refe"
    shutil.copytree(HERE, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    src = (dst / fname).read_text(encoding="utf-8")
    cnt = src.count(anchor)
    if cnt != 1:
        return mid, flips, None, f"the anchor occurs {cnt} times in {fname} (must be exactly 1) -- a vacuous mutation proves nothing"
    (dst / fname).write_text(src.replace(anchor, repl), encoding="utf-8", newline="\n")
    compile((dst / fname).read_text(encoding="utf-8"), fname, "exec")                          # a mutation that does not even parse proves nothing either
    out = td / f"copy_{ix:02d}" / "table.json"
    env = {k: v for k, v in os.environ.items() if k != "TANITAD_GATE_ARM"}
    env["PYTHONUTF8"] = "1"
    r = subprocess.run([PY, str(dst / "test_gate_suite_prerun.py"), "--arm-table", str(out), "--only", ",".join(sorted(flips))], cwd=str(dst), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if not out.is_file():
        return mid, flips, None, f"no table written (rc {r.returncode}): {r.stderr[-400:]}"
    return mid, flips, json.loads(out.read_text(encoding="utf-8")), ""


def matrix(td: Path) -> list:
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("ZZ_MATRIX_WORKERS", "4"))) as pool:
        for res in pool.map(lambda ix_m: one_mutation(ix_m[0], ix_m[1], td), list(enumerate(MUTATIONS))):
            rows.append(res)
    return rows


def report_matrix(rows: list) -> int:
    """Judge the matrix: every mutation must turn EXACTLY its guarded arms from the expected value to the other (= the test goes RED on them)."""
    print(f"  mutation matrix: {len(rows)} source mutations")
    for mid, flips, tab, why in rows:
        ok(tab is not None, f"mutation {mid}: {why}")
        caught = sorted(a for a in flips if tab.get(a) is not ARMS[a][1])
        ok(set(caught) == set(flips), f"mutation {mid}: the test must go RED on {sorted(flips)}; it did on {caught} (table {tab})")
        print(f"    {mid:36s} RED on {len(caught)}/{len(flips)}  {caught[:4]}")
    if "--matrix-only" in sys.argv:
        print(f"ZZTEST_PASS {N}")
    return 0


# ------------------------------------------------------------------------------------------ main
def check_arms(w, table: dict) -> None:
    for name, (_fn, accept, _e2e) in ARMS.items():
        ok(table.get(name) is accept, f"arm {name}: expected {'ACCEPTED' if accept else 'REJECTED (live)'}, got {table.get(name)!r}")


def details(w, table: dict) -> None:
    ev = GT.read_json(w.valid)
    b = ev["binding"]
    ok(ev["kind"] == GT.SUITE_EVIDENCE_KIND and ev["schema"] == GT.SUITE_EVIDENCE_SCHEMA and ev["verdict"] == "PASS" and ev["arm"] is None and GT.signature_ok(ev, w.key))
    ok(GT.SUITE_EVIDENCE_MAX_AGE_S == 12 * 3600 and ev["key_id"] == GT.key_id(w.key))
    ok(b["tree_sha256"] == GT.tree_digest(GT.tree_manifest(w.pkg)) and b["suite_sha256"] == GT.suite_entries_sha256(w.prof["suite"]) and b["profile"] == "refe_synth" and b["commit"] is None)
    ok(b["gate_sha256"] == GT.sha256_file(L.HERE / "launch_gate.py"), "gate_sha256 is the notion `mint` binds (finalize: gate_self_sha=sha256_file(HERE / launch_gate.py))")
    ok(b["gate_checks_sha256"] == GT.sha256_file(C.HERE / "gate_checks.py") and b["gate_token_sha256"] == GT.sha256_file(HERE / "gate_token.py"))
    ok(GT.parse_utc(ev["created_utc"]) is not None and abs(time.time() - GT.parse_utc(ev["created_utc"])) < 3600)
    rows = ev["suite"]["details"]["scripts"]
    ok(len(rows) == 2 and all(r["rc"] == 0 and r["token_seen"] and r["skips"] == [] and "seconds" in r for r in rows) and ev["suite"]["check"] == "G-SUITE" and ev["suite"]["status"] == "PASS")
    ok(set(ev["env"]) >= {"python", "executable", "dists_sha256", "suite_dir", "vars"} and ev["env"]["suite_dir"] == os.path.normcase(os.path.abspath(str(C.HERE))))
    # the age boundary, exactly (injected clock), and ALL differences are collected, not the first
    kw = dict(profile_name="refe_synth", commit=None, pkg=w.pkg, suite=w.prof["suite"], suite_dir=C.HERE, key=w.key)
    t = GT.parse_utc(ev["created_utc"])
    ok(GT.verify_suite_evidence(w.valid, now=t + 12 * 3600, **kw)[0], "exactly 12 h old is accepted (limit is <= 12 h)")
    r = GT.verify_suite_evidence(w.valid, now=t + 12 * 3600 + 1, **kw)
    ok(not r[0] and any("h old" in x for x in r[1]))
    r = GT.verify_suite_evidence(w.valid, now=t - 121, **kw)
    ok(not r[0] and any("in the future" in x for x in r[1]))
    ok(GT.verify_suite_evidence(w.valid, now=t - 119, **kw)[0])
    ev_k2, _ = w.prerun("v_multi.json", key_file=w.key2_file)
    with edited(w.pkg / "refe" / "b.py", "print(55)\n"):
        r = GT.verify_suite_evidence(ev_k2, now=t + 13 * 3600, **kw)
    ok(not r[0] and len(r[1]) >= 3 and any("HMAC" in x for x in r[1]) and any("code tree" in x for x in r[1]) and any("h old" in x for x in r[1]), r[1])
    # it is not a token, a token is not it, and no secret reaches the file
    good, why, _ = GT.verify_token(w.valid, ["--x"], w.pkg, key=w.key)
    ok(not good and any("not a launch-gate token" in r for r in why))
    txt = w.valid.read_text(encoding="utf-8")
    ok(w.key.hex() not in txt and "hmac_sha256" in txt)
    with envvar("REFE_ZZ_PROBE", "sentinel-value-12345"), envvar("REFE_ZZ_API_TOKEN", "secret-value-67890"):
        ev2, rc2 = w.prerun("v_secret.json")
    t2 = ev2.read_text(encoding="utf-8")
    ok(rc2 == 0 and "REFE_ZZ_PROBE" in t2 and "sentinel-value-12345" not in t2 and "secret-value-67890" not in t2 and "REFE_ZZ_API_TOKEN" not in t2, "env vars are bound by a hash of the value; a secret-named one is not read")
    # the suite-entry binding is order- and content-sensitive, per real profile
    shas = {n: GT.suite_entries_sha256(p.get("suite", ())) for n, p in L.PROFILES.items() if not n.startswith("refe_synth")}
    ok(len(shas) >= 10 and all(re.fullmatch(r"[0-9a-f]{64}", s) for s in shas.values()))
    ok(GT.suite_entries_sha256(L.PROFILES["refe_p2"]["suite"]) != GT.suite_entries_sha256(L.PROFILES["refe_p2rep"]["suite"]))
    ok(GT.suite_env(C.HERE)["n_dists"] > 5 and GT.suite_env(C.HERE) == GT.suite_env(C.HERE))
    # WITHOUT the flag: suite_check IS check_suite (same rows, no suite_source), and a token minted that way carries no suite_source
    live_a = C.check_suite(w.prof)
    live_b = quiet(L.suite_check, w.prof, w.pkg)[0]
    strip = lambda e: {**e, "finished_utc": None, "details": {"scripts": [{k: v for k, v in r.items() if k != "seconds"} for r in e["details"]["scripts"]]}}      # noqa: E731
    ok(strip(live_a) == strip(live_b) and "suite_source" not in live_b["details"])
    tok_nf, live_nf = w.e2e("noflag2")
    tok_acc, live_acc = w.e2e("acc2", evidence=w.valid)
    chk_nf, chk_acc = tok_nf["checks"]["G-SUITE"], tok_acc["checks"]["G-SUITE"]
    ok(tok_nf["verdict"] == "PASS" and tok_acc["verdict"] == "PASS" and live_nf == 2 and live_acc == 0)
    ok("suite_source" not in chk_nf and set(chk_acc) - set(chk_nf) == {"suite_source"} and set(tok_acc) == set(tok_nf), "the only new token field is checks.G-SUITE.suite_source, and only with the flag")
    ok(chk_acc["suite_source"]["path"] == "accepted" and chk_acc["suite_source"]["evidence_sha256"] == GT.sha256_file(w.valid) and "age_s" in chk_acc["suite_source"])
    good, why, _ = GT.verify_token(next((w.td / "out_acc2").glob("PASS_*.json")), ["--zz", "1"], w.pkg, key=w.key, profile="refe_synth")
    ok(good and not why, why)
    ok(GT.sha256_file(w.td / "out_acc2" / "evidence_G-SUITE.json") == chk_acc["evidence_sha256"], "the token binds the sha256 of the G-SUITE evidence file, which names the pre-run's")
    # the CLI surface
    h = subprocess.run([PY, str(HERE / "launch_gate.py"), "run", "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    ok("--suite-evidence" in h)
    h2 = subprocess.run([PY, str(HERE / "launch_gate.py"), "suite-prerun", "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    ok("--profile" in h2 and "--pkg" in h2 and "--out" in h2)
    _, out_fail = quiet(L.main, ["suite-prerun", "--profile", "refe_synth_fail", "--pkg", str(w.pkg), "--out", str(w.td / "pf.json"), "--key-file", str(w.key_file)])
    ok("ZZSUITE-PRERUN-FAIL" in out_fail)
    _, out_pass = quiet(L.main, ["suite-prerun", "--profile", "refe_synth", "--pkg", str(w.pkg), "--out", str(w.td / "pp.json"), "--key-file", str(w.key_file)])
    ok("ZZSUITE-PRERUN-PASS" in out_pass)


def main() -> int:
    if "--arm-table" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--arm-table") + 1])
        only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
        with tempfile.TemporaryDirectory() as td:
            w = World(Path(td))
            out.write_text(json.dumps(run_arms(w, only), indent=1), encoding="utf-8")
        return 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if "--matrix-only" in sys.argv:
            return report_matrix(matrix(td / "m"))
        w = World(td / "w")
        t0 = time.time()
        why: dict = {}
        table = run_arms(w, why=why)
        if "--arm-report" in sys.argv:
            rows = {n: {"expected": "ACCEPTED" if ARMS[n][1] else "REJECTED", "observed": "ACCEPTED" if table[n] is True else ("REJECTED" if table[n] is False else str(table[n])), "e2e": ARMS[n][2],
                        "first_reason": (why.get(n) or [None])[0]} for n in ARMS}
            Path(sys.argv[sys.argv.index("--arm-report") + 1]).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"  {len(table)} arms in {time.time() - t0:.0f}s: {sum(1 for n in table if ARMS[n][1])} accepted (positive twins / controls), {sum(1 for n in table if not ARMS[n][1])} rejected -> live")
        check_arms(w, table)
        details(w, table)
        if "--no-mutations" not in sys.argv:
            report_matrix(matrix(td / "m"))
    print(f"ZZTEST_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
