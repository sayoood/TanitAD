"""G-SUITE-PINNED SPLIT MODE (PI ruling 2026-11-28, chat, verbatim choice "Split: Thor + dev box").

Thor runs every pinned test on the FULL tip archive; the dev box re-runs ONLY the test files Thor cannot pass,
on a REDUCED tree; every test must pass (or skip for a registered reason) on ONE host at the tip; the bar is
unchanged. Profile `refcv9-full-d1` ONLY, through a MODULE-level allow-list -- never a key inside a
`PROFILES[...]` dict, whose by-value fingerprints are pinned in tests and bound to launch-gate records.

Every guard below has a deliberate-regression arm (mutate, do not inspect): the control must read the
expected verdict, the mutated arm must read the OPPOSITE. Every temp dir is `tmp_path`/`tmp_path_factory`.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import launch_gate as LG  # noqa: E402

PROFILE = "refcv9-full-d1"
THOR_KEY = "G-SUITE-PINNED-THOR-HALF"
FILE_A = {"sub": "stack/tests", "file": "test_a.py"}
FILE_B = {"sub": "stack/tests", "file": "test_b.py"}
FILE_C = {"sub": "stack/tests", "file": "test_c.py"}
FILE_GONE = {"sub": "stack/tests", "file": "test_gone.py"}
KEY_A, KEY_B, KEY_C = ("stack/tests/test_a.py", "stack/tests/test_b.py", "stack/tests/test_c.py")
ARGV = ["--arm", "split", "--steps", "10"]
REDUCE = 1000          # bytes, for the tiny repo

_TESTS = {
    "test_a.py": "def test_one():\n    assert 1 == 1\n\n\ndef test_two():\n    assert 2 == 2\n",
    # host-bound: `test_hostbound` needs the dev-box environment, `test_thor_only` cannot pass there
    "test_b.py": ("import os\n\n\ndef test_hostbound():\n    assert os.environ.get('PINSPLIT_HOST') == 'devbox'\n\n\n"
                  "def test_thor_only():\n    assert os.environ.get('PINSPLIT_HOST') != 'devbox'\n\n\n"
                  "def test_plain():\n    assert True\n"),
    "test_c.py": "def test_always_bad():\n    assert False\n",
}


# ---------------------------------------------------------------------------------------------- #
# a tiny throwaway git repo (NEVER tanitad-push's index)                                          #
# ---------------------------------------------------------------------------------------------- #
def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(*a, cwd, check=True):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *a], cwd=str(cwd), env=_clean_env(),
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=check)


def _write(p: Path, data: bytes | str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def _mk_repo(base: Path) -> tuple[Path, str, str]:
    src = base / "src"
    src.mkdir(parents=True)
    _git("init", "-q", cwd=src)
    _write(src / "stack" / "tanitad" / "__init__.py", "X = 1\n")
    for n, body in _TESTS.items():
        _write(src / "stack" / "tests" / n, body)
    _write(src / "taniteval" / "__init__.py", "T = 1\n")
    _write(src / "tools" / "big_tool.bin", b"t" * 3000)                      # >= REDUCE but under a KEPT root
    _write(src / "Project Steering" / "big note.md", b"m" * 3000)            # >= REDUCE, KEPT (a space in the root)
    _write(src / "TanitAD Research Lab" / "big.bin", b"r" * 3000)            # >= REDUCE: dropped
    _write(src / "TanitAD Research Lab" / "deep dir" / "also big.bin", b"r" * 1000)   # == REDUCE: dropped
    _write(src / "TanitAD Research Lab" / "small.txt", b"s" * 999)           # < REDUCE: kept
    _write(src / "FlyWheels" / "fw" / "big.bin", b"f" * 5000)                # dropped
    _write(src / "FlyWheels" / "fw" / "small.txt", b"f" * 10)                # kept
    _write(src / "top.bin", b"z" * 5000)                                     # a top-level file: kept
    _git("add", "-A", cwd=src)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c", cwd=src)
    sha = _git("rev-parse", "HEAD", cwd=src).stdout.strip()
    return src, str(src / ".git"), sha


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    base = tmp_path_factory.mktemp("pinsplit_repo")
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    return _mk_repo(base)


# ---------------------------------------------------------------------------------------------- #
# running the two halves for real (pytest subprocesses on the tiny archive)                        #
# ---------------------------------------------------------------------------------------------- #
def _ctx(repo_t, work: Path, half: str, thor_path: str | None = None, profile: str = PROFILE,
         argv: list | None = None, out: Path | None = None) -> LG.Ctx:
    src, git_dir, sha = repo_t
    argv = list(argv or ARGV)
    out = out or (work / f"out_{half}")
    opts = {"git_dir": git_dir, "suite_work": str(work / "sw"), "omp": 1, "cpu_only": True,
            "pinned_half": half, "pinned_thor_half": thor_path, "pinned_reduce_bytes": REDUCE}
    return LG.Ctx(profile=profile, tree=str(src), commit=sha, argv=argv, out_dir=str(out), path_map=[],
                  tree_sha256=LG.tree_digest(LG.tree_manifest(src)), argv_sha256=LG.argv_sha256(argv),
                  options=opts)


def _run_job(monkeypatch, ctx: LG.Ctx, pin_files: list[dict], audit_ok: bool = True) -> dict:
    pin = {"files": pin_files, "ids": [], "flaky": []}
    monkeypatch.setattr(LG, "env_failure_pin", lambda prof, tree: (pin, None))
    monkeypatch.setattr(LG, "_pinned_run_audit", lambda dest, logs, env, o: (
        {"ok": True, "summary": "=== 3/3 defects CAUGHT", "rc": 0, "why": None} if audit_ok else
        {"ok": False, "summary": None, "rc": 1, "why": "guard mutation audit: no summary line (rc 1)"}))
    return LG.job_suite_pinned(ctx, ["G-SUITE-PINNED"])["G-SUITE-PINNED"]


def _two_halves(repo_t, work: Path, pin_files: list[dict], devbox_env: bool = True,
                audit_ok: bool = True) -> tuple[dict, Path, dict, LG.Ctx]:
    """Thor half (PINSPLIT_HOST unset), then the dev-box complement (PINSPLIT_HOST=devbox unless told not)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("PINSPLIT_HOST", raising=False)
        c1 = _ctx(repo_t, work, "thor")
        tev = _run_job(mp, c1, pin_files, audit_ok)
        tpath = LG.write_evidence(c1.out_dir, tev)
        if devbox_env:
            mp.setenv("PINSPLIT_HOST", "devbox")
        c2 = _ctx(repo_t, work, "devbox-complement", str(tpath))
        dev = _run_job(mp, c2, pin_files, audit_ok)
    return tev, tpath, dev, c2


@pytest.fixture(scope="module")
def pass_case(repo, tmp_path_factory):
    work = tmp_path_factory.mktemp("pinsplit_pass")
    return _two_halves(repo, work, [FILE_A, FILE_B]) + (work,)


@pytest.fixture(scope="module")
def fail_case(repo, tmp_path_factory):
    work = tmp_path_factory.mktemp("pinsplit_fail")
    return _two_halves(repo, work, [FILE_A, FILE_B, FILE_C]) + (work,)


def _tid(outcomes: dict, tail: str) -> str:
    m = [t for t in outcomes if t.endswith("::" + tail)]
    assert len(m) == 1, (tail, sorted(outcomes))
    return m[0]


# ---------------------------------------------------------------------------------------------- #
# the merge, end to end                                                                           #
# ---------------------------------------------------------------------------------------------- #
def test_THOR_HALF_is_recorded_under_its_own_key_and_is_not_a_PASS(pass_case):
    tev, tpath, _dev, _ctx2, _w = pass_case
    assert tev["check"] == THOR_KEY and tpath.name == THOR_KEY + ".json"
    assert tev["status"] == LG.PINNED_THOR_HALF_OK and tev["status"] not in ("PASS", "PI-DECISION")
    d = tev["details"]
    assert d["half"] == "thor" and d["mode"] == "thor-half" and not d["archive"].get("reduced")
    assert not d["archive"].get("dropped")                       # the FULL archive: nothing dropped
    oc = d["files"][KEY_B]["outcomes"]
    assert oc[_tid(oc, "test_hostbound")] == "fail" and oc[_tid(oc, "test_thor_only")] == "pass"
    assert d["complement_files"] == [KEY_B]                      # a.py passed everywhere: not re-run
    assert d["n_not_pass"] == 1 and d["n_tests"] == 5
    assert LG._collect_evidence(Path(tpath).parent.parent, []) == {}        # never read as a gate check


def test_MERGE_PASSES_when_every_Thor_failure_passes_on_the_devbox(pass_case):
    tev, tpath, dev, ctx2, work = pass_case
    assert dev["check"] == "G-SUITE-PINNED" and dev["status"] == "PASS", dev["reasons"]
    d = dev["details"]
    assert d["mode"] == "split" and d["half"] == "devbox-complement"
    assert d["thor_half"]["sha256"] == LG.sha256_file(tpath) and re.fullmatch(r"[0-9a-f]{64}", d["thor_half"]["sha256"])
    assert d["complement_files"] == [KEY_B] and list(d["files"]) == [KEY_B]     # ONLY the file with a Thor failure
    assert (d["n_ok_both"], d["n_ok_thor_only"], d["n_ok_devbox_only"], d["n_fail_both"]) == (1, 3, 1, 0)
    # a test that passed on Thor and FAILED on the dev box is reported (count + ids), and does not fail the check
    assert d["n_passed_thor_failed_devbox"] == 1 and d["passed_thor_failed_devbox"][0].endswith("::test_thor_only")
    assert dev["binding"] == ctx2.binding() and not dev["arm"]
    # the reduced tree: the dropped list carries (path, size) and is exactly the >= threshold Research Lab / FlyWheels members
    assert sorted(map(tuple, d["dropped"])) == [("FlyWheels/fw/big.bin", 5000), ("TanitAD Research Lab/big.bin", 3000),
                                               ("TanitAD Research Lab/deep dir/also big.bin", 1000)]
    assert d["archive"]["reduced"] is True and d["archive"]["reduce_bytes"] == REDUCE
    # the dev box's archive is a DIFFERENT directory from Thor's full one, and Thor's holds every member
    full = work / "sw" / f"f{ctx2.commit[:10]}"
    red = work / "sw" / f"p{ctx2.commit[:10]}"
    assert (full / "FlyWheels" / "fw" / "big.bin").is_file() and not (red / "FlyWheels" / "fw" / "big.bin").exists()
    assert (red / "FlyWheels" / "fw" / "small.txt").is_file() and (red / "Project Steering" / "big note.md").is_file()
    assert d["skip_worktree"]["rc"] == 0 and d["skip_worktree"]["n"] == 3
    # REGRESSION ARM (mutate the host environment): without the dev-box environment the host-bound test fails on BOTH
    # hosts, and the very same merge FAILS -- the PASS above came from the dev box passing it, not from the merge being lax
    # (checked in test_MERGE_FAILS_when_a_test_fails_on_BOTH_hosts and the mutation arm below)


def test_MERGE_FAILS_when_a_test_fails_on_BOTH_hosts_and_names_both_reasons(fail_case):
    _tev, _tp, dev, _c, _w = fail_case
    assert dev["status"] == "FAIL"
    assert dev["details"]["n_fail_both"] == 1
    bad = dev["details"]["fail_both"][0]
    assert bad[0].endswith("::test_always_bad") and bad[1] == "fail" and bad[2] == "fail"
    assert any("EITHER host" in r and "[Thor: fail | dev box: fail]" in r for r in dev["reasons"]), dev["reasons"]
    assert dev["details"]["complement_files"] == [KEY_B, KEY_C]


def test_REGRESSION_ARM_the_host_bound_test_fails_on_both_when_the_devbox_environment_is_absent(repo, tmp_path):
    """Same pins as the PASS case, the dev-box environment NOT applied: `test_hostbound` is red on both hosts."""
    _tev, _tp, dev, _c = _two_halves(repo, tmp_path, [FILE_A, FILE_B], devbox_env=False)
    assert dev["status"] == "FAIL" and dev["details"]["n_fail_both"] == 1
    assert dev["details"]["fail_both"][0][0].endswith("::test_hostbound")


def test_nothing_to_rerun_means_no_archive_and_a_PASS(repo, tmp_path):
    _tev, _tp, dev, _c = _two_halves(repo, tmp_path, [FILE_A])
    assert dev["status"] == "PASS" and dev["details"]["complement_files"] == []
    assert dev["details"]["archive"] is None and not (tmp_path / "sw" / f"p{repo[2][:10]}").exists()
    assert dev["details"]["n_ok_thor_only"] == 2


def test_a_Thor_pinned_file_with_no_tests_is_a_STRUCTURAL_failure_the_complement_refuses(repo, tmp_path):
    tev, _tp, dev, _c = _two_halves(repo, tmp_path, [FILE_A, FILE_GONE])
    assert tev["status"] == "FAIL" and any("test_gone.py collected ZERO tests" in r for r in tev["reasons"]), tev["reasons"]
    assert dev["status"] == "FAIL" and any("REFUSED" in r for r in dev["reasons"]), dev["reasons"]
    assert dev["details"].get("archive") is None                  # refused BEFORE anything was archived


# ---------------------------------------------------------------------------------------------- #
# the Thor half is validated: binding, full archive, junit for every file, outcomes hash           #
# ---------------------------------------------------------------------------------------------- #
def _tamper(pass_case, tmp_path, mutate) -> dict:
    tev, tpath, _dev, ctx2, _w = pass_case
    doc = json.loads(Path(tpath).read_text(encoding="utf-8"))
    mutate(doc)
    p = tmp_path / "thor_half_tampered.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    c = copy.deepcopy(ctx2)
    c.options = dict(c.options, pinned_thor_half=str(p))
    pin = {"files": [FILE_A, FILE_B], "ids": [], "flaky": []}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LG, "env_failure_pin", lambda prof, tree: (pin, None))
        return LG.job_suite_pinned(c, ["G-SUITE-PINNED"])["G-SUITE-PINNED"]


def test_CONTROL_the_untampered_Thor_half_is_accepted(pass_case):
    assert LG.check_pinned_thor_half(json.loads(Path(pass_case[1]).read_text(encoding="utf-8")), pass_case[3],
                                     {}, [FILE_A, FILE_B]) == []


@pytest.mark.parametrize("field,value", [("commit", "0" * 40), ("tree_sha256", "1" * 64),
                                         ("argv_sha256", "2" * 64), ("profile", "refcv9-full")])
def test_a_Thor_half_binding_mismatch_is_REFUSED(pass_case, tmp_path, field, value):
    ev = _tamper(pass_case, tmp_path, lambda d: d["binding"].__setitem__(field, value))
    assert ev["status"] == "FAIL" and ev["check"] == "G-SUITE-PINNED"
    assert any(f"binding {field} differs" in r for r in ev["reasons"]), ev["reasons"]
    assert ev["details"].get("archive") is None                   # refused BEFORE anything was archived


def test_a_missing_Thor_junit_is_REFUSED(pass_case, tmp_path):
    ev = _tamper(pass_case, tmp_path, lambda d: d["details"]["files"][KEY_B].pop("outcomes"))
    assert ev["status"] == "FAIL" and any("test_b.py has no junit in the Thor half" in r for r in ev["reasons"]), ev["reasons"]


def test_edited_Thor_outcomes_do_not_match_their_recorded_hash(pass_case, tmp_path):
    def flip(d):
        oc = d["details"]["files"][KEY_B]["outcomes"]
        oc[_tid(oc, "test_hostbound")] = "pass"                 # a hand-edited Thor failure
    ev = _tamper(pass_case, tmp_path, flip)
    assert ev["status"] == "FAIL" and any("outcomes_sha256" in r for r in ev["reasons"]), ev["reasons"]


def test_a_Thor_half_that_was_not_the_FULL_archive_is_REFUSED(pass_case, tmp_path):
    ev = _tamper(pass_case, tmp_path, lambda d: d["details"]["archive"].update(reduced=True, dropped=[["x", 1]]))
    assert ev["status"] == "FAIL" and any("FULL archive" in r for r in ev["reasons"]), ev["reasons"]


def test_a_Thor_half_with_a_failing_status_or_an_arm_is_REFUSED(pass_case, tmp_path):
    ev = _tamper(pass_case, tmp_path, lambda d: d.update(status="FAIL", arm="x"))
    assert ev["status"] == "FAIL" and any("not 'HALF-COMPLETE'" in r for r in ev["reasons"])
    assert any("regression arm" in r for r in ev["reasons"])


def test_a_G_SUITE_PINNED_file_offered_as_the_Thor_half_is_REFUSED(pass_case, tmp_path):
    ev = _tamper(pass_case, tmp_path, lambda d: d.update(check="G-SUITE-PINNED"))
    assert ev["status"] == "FAIL" and any("not a Thor-half evidence" in r for r in ev["reasons"])


# ---------------------------------------------------------------------------------------------- #
# who may use it: refcv9-full-d1 ONLY, by a MODULE-level allow-list                                #
# ---------------------------------------------------------------------------------------------- #
def test_the_allow_list_is_module_level_and_no_profile_dict_carries_it():
    assert LG.PINNED_SPLIT_PROFILES == frozenset({"refcv9-full-d1"})
    for name, prof in LG.PROFILES.items():
        assert not any("pinned_split" in str(k) or "pinned_half" in str(k) or "thor_half" in str(k) for k in prof), name


@pytest.mark.parametrize("profile", sorted(set(LG.PROFILES) - {PROFILE}))
def test_split_flags_are_REFUSED_on_every_other_profile(profile):
    with pytest.raises(LG.GateError, match="REFUSED"):
        LG.pinned_split_refusals(profile, "thor", None, LG.PINNED_REDUCE_BYTES_DEFAULT, "g")
    with pytest.raises(LG.GateError, match="REFUSED"):
        LG.pinned_split_refusals(profile, None, "x.json", LG.PINNED_REDUCE_BYTES_DEFAULT, None)
    LG.pinned_split_refusals(profile, None, None, LG.PINNED_REDUCE_BYTES_DEFAULT, None)     # no split option: a no-op


def test_the_option_set_is_consistent_on_the_allowed_profile():
    D = LG.PINNED_REDUCE_BYTES_DEFAULT
    LG.pinned_split_refusals(PROFILE, "thor", None, D, "g")
    LG.pinned_split_refusals(PROFILE, "devbox-complement", "t.json", 5, "g")
    for args in [("devbox-complement", None, D, "g"), ("thor", "t.json", D, "g"), ("thor", None, 5, "g"),
                 (None, "t.json", D, "g"), (None, None, 5, "g"), ("thor", None, D, None)]:
        with pytest.raises(LG.GateError):
            LG.pinned_split_refusals(PROFILE, *args)


def test_REGRESSION_ARM_the_refusal_is_the_allow_list_and_nothing_else(monkeypatch):
    with pytest.raises(LG.GateError):
        LG.pinned_split_refusals("refcv7", "thor", None, LG.PINNED_REDUCE_BYTES_DEFAULT, "g")
    monkeypatch.setattr(LG, "PINNED_SPLIT_PROFILES", frozenset({"refcv7", PROFILE}))
    LG.pinned_split_refusals("refcv7", "thor", None, LG.PINNED_REDUCE_BYTES_DEFAULT, "g")      # now silent


def test_the_job_itself_refuses_a_non_allow_listed_profile(repo, tmp_path):
    c = _ctx(repo, tmp_path, "thor", profile="refcv9-full")
    ev = LG.job_suite_pinned(c, ["G-SUITE-PINNED"])["G-SUITE-PINNED"]
    assert ev["status"] == "FAIL" and "refused for profile 'refcv9-full'" in ev["reasons"][0]
    assert not (tmp_path / "sw").exists()                              # nothing was archived


def test_the_CLI_exposes_the_split_flags_and_the_default_keeps_the_ordinary_path():
    ap = LG.build_parser()
    a = ap.parse_args(["run", "--profile", PROFILE, "--tree", "t", "--commit", "c", "--argv-file", "a",
                       "--out-dir", "o", "--pinned-half", "devbox-complement", "--pinned-thor-half", "x.json",
                       "--pinned-reduce-bytes", "7"])
    assert (a.pinned_half, a.pinned_thor_half, a.pinned_reduce_bytes) == ("devbox-complement", "x.json", 7)
    b = ap.parse_args(["run", "--profile", PROFILE, "--tree", "t", "--commit", "c", "--argv-file", "a", "--out-dir", "o"])
    assert (b.pinned_half, b.pinned_thor_half, b.pinned_reduce_bytes) == (None, None, 1_000_000)


# ---------------------------------------------------------------------------------------------- #
# the final gate: Thor-half evidence is never G-SUITE-PINNED; split only for the allow-list         #
# ---------------------------------------------------------------------------------------------- #
def _binding(profile=PROFILE) -> dict:
    return {"profile": profile, "commit": "a" * 40, "tree_sha256": "b" * 64, "argv_sha256": "c" * 64}


def _split_ev(binding: dict, **details) -> dict:
    return {"kind": "evidence", "check": "G-SUITE-PINNED", "status": "PASS", "binding": binding, "inputs_read": {},
            "arm": None, "details": {"mode": "split", "half": "devbox-complement",
                                     "thor_half": {"sha256": "d" * 64}, **details}}


def test_evidence_rejections_control_and_split_rules():
    plain = {"kind": "evidence", "check": "G-SUITE-PINNED", "status": "PASS", "binding": _binding(),
             "inputs_read": {}, "arm": None, "details": {}}
    assert LG.evidence_rejections(plain, _binding(), {}) == []                      # the ordinary path: untouched
    assert LG.evidence_rejections(_split_ev(_binding()), _binding(), {}) == []        # the real merge on the allow-list
    # not allow-listed
    other = _binding("refcv9-full")
    why = LG.evidence_rejections(_split_ev(other), other, {})
    assert any("not admitted for profile 'refcv9-full'" in w for w in why)
    # no sha256 of the Thor half / not the complement's merge
    ev = _split_ev(_binding())
    ev["details"]["thor_half"] = {}
    assert LG.evidence_rejections(ev, _binding(), {})
    ev = _split_ev(_binding(), half="thor")
    assert LG.evidence_rejections(ev, _binding(), {})


def test_a_Thor_half_offered_as_G_SUITE_PINNED_is_always_rejected_and_never_collected(tmp_path):
    for details in ({"half": "thor"}, {"mode": "thor-half"}, {"half": "thor", "mode": "split"}):
        ev = {"kind": "evidence", "check": "G-SUITE-PINNED", "status": "PASS", "binding": _binding(), "inputs_read": {},
              "arm": None, "details": details}
        assert any("never G-SUITE-PINNED" in w for w in LG.evidence_rejections(ev, _binding(), {})), details
    # the Thor half's own key is not a gate check: _collect_evidence never returns it, even from an imported dir
    d = tmp_path / "imp"
    d.mkdir()
    (d / (THOR_KEY + ".json")).write_text(json.dumps({"kind": "evidence", "check": THOR_KEY, "status": "PASS"}),
                                          encoding="utf-8")
    assert THOR_KEY not in LG.CHECKS and LG._collect_evidence(tmp_path / "none", [str(d)]) == {}
    # ... and a file NAMED G-SUITE-PINNED.json whose body says it is the Thor half is the rejected case above (check
    # field and half marker are both read, never the file name)


def test_the_ordinary_job_path_is_untouched_when_no_half_is_given(repo, tmp_path, monkeypatch):
    def boom(ctx):
        raise AssertionError("the split path ran without --pinned-half")
    monkeypatch.setattr(LG, "job_suite_pinned_split", boom)
    monkeypatch.setattr(LG, "env_failure_pin", lambda prof, tree: ({"files": [], "ids": []}, None))
    c = _ctx(repo, tmp_path, "thor")
    c.options = {"suite_work": str(tmp_path / "sw")}                               # no half, no git dir
    ev = LG.job_suite_pinned(c, ["G-SUITE-PINNED"])["G-SUITE-PINNED"]
    assert ev["check"] == "G-SUITE-PINNED" and ev["status"] == "FAIL" and "needs --git-dir" in ev["reasons"][0]


# ---------------------------------------------------------------------------------------------- #
# the reduced archive                                                                              #
# ---------------------------------------------------------------------------------------------- #
def _members(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != LG._ARCHIVE_DONE}


ALL = {"stack/tanitad/__init__.py", "stack/tests/test_a.py", "stack/tests/test_b.py", "stack/tests/test_c.py",
       "taniteval/__init__.py", "tools/big_tool.bin", "Project Steering/big note.md", "TanitAD Research Lab/big.bin",
       "TanitAD Research Lab/deep dir/also big.bin", "TanitAD Research Lab/small.txt", "FlyWheels/fw/big.bin",
       "FlyWheels/fw/small.txt", "top.bin"}
_ARCHIVE_NAME = ".gate_archive_complete.json"
DROPPED = {"TanitAD Research Lab/big.bin", "TanitAD Research Lab/deep dir/also big.bin", "FlyWheels/fw/big.bin"}


def _check_reduced(repo_t, dest: Path) -> dict:
    _src, git_dir, sha = repo_t
    rec = LG.stream_archive(git_dir, sha, dest, drop=lambda n, s: LG.pinned_reduce_drop(n, s, REDUCE))
    got = _members(dest)
    assert got == ALL - DROPPED, sorted(got ^ (ALL - DROPPED))            # kept every stack/taniteval/tools/Steering member
    assert {d[0] for d in rec["dropped"]} == DROPPED and rec["dropped_files"] == 3 and rec["dropped_bytes"] == 9000
    assert {d[0]: d[1] for d in rec["dropped"]}["FlyWheels/fw/big.bin"] == 5000
    return rec


def test_the_reduced_archive_keeps_every_kept_root_member_and_drops_only_big_RL_FW_members(repo, tmp_path):
    rec = _check_reduced(repo, tmp_path / "red")
    assert rec["reduced"] is True and (tmp_path / "red" / LG._ARCHIVE_DONE).is_file()
    # the stack/+taniteval/ digest is that of the full commit (the tree digest the gate binds does not move)
    full = tmp_path / "full"
    frec = LG.stream_archive(repo[1], repo[2], full)
    assert "dropped" not in frec and "reduced" not in frec                  # no drop => no new key: the ordinary record
    assert _members(full) == ALL
    assert LG.tree_digest(LG.tree_manifest(full)) == LG.tree_digest(LG.tree_manifest(tmp_path / "red"))
    assert LG.tree_digest(LG.tree_manifest(full)) == LG.tree_digest(LG.tree_manifest(repo[0]))


def test_REGRESSION_ARM_a_drop_rule_that_ignores_the_root_is_caught(repo, tmp_path, monkeypatch):
    _check_reduced(repo, tmp_path / "ok")                                  # control: the real rule reads green
    monkeypatch.setattr(LG, "pinned_reduce_drop", lambda name, size, thr: int(size) >= int(thr))   # drops tools/ + Steering
    with pytest.raises(AssertionError):
        _check_reduced(repo, tmp_path / "mutated")
    monkeypatch.setattr(LG, "pinned_reduce_drop", lambda name, size, thr: False)                      # drops nothing
    with pytest.raises(AssertionError):
        _check_reduced(repo, tmp_path / "mutated2")


def test_the_drop_rule_literals():
    f = LG.pinned_reduce_drop
    assert f("TanitAD Research Lab/x", 1_000_000, 1_000_000) and not f("TanitAD Research Lab/x", 999_999, 1_000_000)
    assert f("FlyWheels/a/b", 5_000_000, 1_000_000)
    for keep in ("stack/x", "taniteval/x", "tools/x", "Project Steering/x", "README.md", "TanitAD Research Lab", "Flywheels/x"):
        assert not f(keep, 10 ** 9, 1_000_000), keep


def test_dropped_members_are_skip_worktree_so_git_status_reads_clean(repo, tmp_path):
    _src, git_dir, sha = repo
    dest = tmp_path / "red"
    rec = _check_reduced(repo, dest)
    got = LG.throwaway_repo(dest, git_dir, sha)
    assert got["head"] == sha
    env = LG.safe_dir_env(_clean_env(), dest)
    before = subprocess.run(["git", "-C", str(dest), "status", "--porcelain"], env=env, capture_output=True, text=True,
                            encoding="utf-8").stdout
    assert sum(1 for ln in before.splitlines() if ln.startswith((" D", "D "))) == 3          # control: 3 phantom deletions
    r = LG._mark_skip_worktree(dest, [d[0] for d in rec["dropped"]])
    assert r["rc"] == 0 and r["n"] == 3, r
    after = subprocess.run(["git", "-C", str(dest), "status", "--porcelain"], env=env, capture_output=True, text=True,
                           encoding="utf-8").stdout
    assert [ln for ln in after.splitlines() if _ARCHIVE_NAME not in ln] == [], after   # only the gate's own marker file is untracked


def test_a_reused_reduced_archive_with_another_threshold_is_refused(repo, tmp_path):
    work = tmp_path
    c = _ctx(repo, work, "devbox-complement")
    det: dict = {}
    dest, why = LG._pinned_split_archive(c, {}, det, c.commit, reduced=True)
    assert dest is not None and why == [] and det["archive"]["reduce_bytes"] == REDUCE
    c.options["pinned_reduce_bytes"] = 5000
    dest2, why2 = LG._pinned_split_archive(c, {}, {}, c.commit, reduced=True)
    assert dest2 is None and "another archive" in why2[0]
    # and the FULL archive never reuses a reduced directory (different dir name), nor the reverse
    assert (work / "sw" / f"p{c.commit[:10]}").is_dir() and not (work / "sw" / f"f{c.commit[:10]}").exists()


# ---------------------------------------------------------------------------------------------- #
# the merge as a pure function, with the MUTATION arm                                               #
# ---------------------------------------------------------------------------------------------- #
FILES = [{"sub": "s", "file": "a.py"}, {"sub": "s", "file": "b.py"}]
SKIP_OK = (("x::reg", r"^registered$"),)


def _f(outcomes: dict, rc=1) -> dict:
    return {"rc": rc, "n_tests": len(outcomes), "outcomes": outcomes}


def _merge(thor: dict, dev: dict, comp=("s/b.py",), flaky=()):
    return LG.merge_pinned_split(FILES, thor, dev, comp, SKIP_OK, flaky)


def _assert_fails_on_both(thor_fail="fail", dev_fail="fail"):
    reasons, det = _merge({"s/a.py": _f({"x::a1": "pass"}), "s/b.py": _f({"x::b1": thor_fail, "x::b2": "pass"})},
                          {"s/b.py": _f({"x::b1": dev_fail, "x::b2": "pass"})})
    assert reasons and det["n_fail_both"] == 1 and det["fail_both"][0][0] == "x::b1", (reasons, det)
    assert "EITHER host" in reasons[0]


def test_merge_PASS_every_Thor_failure_passes_on_the_devbox():
    reasons, det = _merge({"s/a.py": _f({"x::a1": "pass"}), "s/b.py": _f({"x::b1": "fail", "x::b2": "error", "x::b3": "pass"})},
                          {"s/b.py": _f({"x::b1": "pass", "x::b2": "pass", "x::b3": "fail"})})
    assert reasons == []
    assert (det["n_ok_thor_only"], det["n_ok_devbox_only"], det["n_ok_both"]) == (2, 2, 0)
    assert det["passed_thor_failed_devbox"] == ["x::b3"] and det["n_passed_thor_failed_devbox"] == 1


def test_merge_FAIL_when_one_id_fails_on_both_hosts_and_names_both_reasons():
    _assert_fails_on_both()
    _assert_fails_on_both("error", "fail")
    reasons, det = _merge({"s/a.py": _f({"x::a1": "pass"}), "s/b.py": _f({"x::b1": "fail"})},
                          {"s/b.py": _f({"x::b1": "skip:not registered"})})
    assert det["fail_both"][0][1:] == ["fail", "unregistered skip (not registered)"]


def test_MUTATION_ARM_counting_Thor_failures_as_passes_turns_the_fails_on_both_test_RED(monkeypatch):
    _assert_fails_on_both()                                              # control: green against the real merge
    real = LG._pinned_ok
    monkeypatch.setattr(LG, "_pinned_ok", lambda tid, o, pats: True if o in ("fail", "error") else real(tid, o, pats))
    with pytest.raises(AssertionError):                                  # the mutated merge no longer fails => the test is RED
        _assert_fails_on_both()
    monkeypatch.setattr(LG, "_pinned_ok", lambda tid, o, pats: True)     # a second mutation: nothing ever fails
    with pytest.raises(AssertionError):
        _assert_fails_on_both()


def test_merge_a_registered_skip_is_ok_on_one_host_an_unregistered_one_is_not():
    thor = {"s/a.py": _f({"x::a1": "pass"}), "s/b.py": _f({"x::reg": "skip:registered", "x::other": "skip:registered"})}
    reasons, det = _merge(thor, {"s/b.py": _f({"x::reg": "pass", "x::other": "pass"})})
    assert reasons == []
    reasons, det = _merge(thor, {"s/b.py": _f({"x::reg": "skip:registered", "x::other": "skip:registered"})})
    assert [b[0] for b in det["fail_both"]] == ["x::other"]            # the registry admits x::reg ONLY (id regex fullmatch)


def test_merge_structural_failures_are_never_silent():
    ok_a = _f({"x::a1": "pass"})
    r, _ = _merge({"s/a.py": ok_a}, {})                                   # b.py: no Thor result at all
    assert any("s/b.py produced no junit on Thor" in x for x in r)
    r, _ = _merge({"s/a.py": ok_a, "s/b.py": {"rc": 2, "tail": "boom"}}, {})
    assert any("produced no junit on Thor (rc 2)" in x for x in r)
    r, _ = _merge({"s/a.py": ok_a, "s/b.py": {"rc": 5, "n_tests": 0, "outcomes": {}}}, {}, comp=())
    assert any("ZERO tests on Thor" in x for x in r)
    r, _ = _merge({"s/a.py": ok_a, "s/b.py": _f({"x::b1": "fail"})}, {"s/b.py": {"rc": "timeout", "tail": "t"}})
    assert any("s/b.py produced no junit on the dev box" in x for x in r)
    r, _ = _merge({"s/a.py": ok_a, "s/b.py": _f({"x::b1": "fail"})}, {})        # the complement file was never run
    assert any("no junit on the dev box" in x for x in r)


def test_merge_flaky_rule_applies_unchanged_and_only_to_failures():
    thor = {"s/a.py": _f({"x::a1": "pass"}), "s/b.py": _f({"x::fl": "fail"})}
    r, det = _merge(thor, {"s/b.py": _f({"x::fl": "fail"})}, flaky=("x::fl",))
    assert r == [] and det["flaky_failing"] == ["x::fl"]
    r, det = _merge(thor, {"s/b.py": _f({"x::fl": "fail"})})                    # not admitted => FAIL
    assert r and det["n_fail_both"] == 1
    r, det = _merge(thor, {"s/b.py": _f({"x::fl": "skip:whatever"})}, flaky=("x::fl",))   # a skip is not a flaky failure
    assert r and det["n_fail_both"] == 1


def test_complement_files_are_exactly_the_files_with_a_non_pass_Thor_test():
    thor = {"s/a.py": _f({"x::a1": "pass", "x::reg": "skip:registered"}), "s/b.py": _f({"x::b1": "skip:nope"})}
    assert LG.pinned_complement_files(FILES, thor, SKIP_OK) == [FILES[1]]
    thor["s/b.py"] = _f({"x::b1": "pass"})
    assert LG.pinned_complement_files(FILES, thor, SKIP_OK) == []
    assert LG.pinned_complement_files(FILES, {"s/a.py": {"rc": 1}}, SKIP_OK) == []     # no junit: structural, never rescued


def test_junit_outcomes_parser_literals(tmp_path):
    x = tmp_path / "j.xml"
    x.write_text('<testsuites><testsuite>'
                 '<testcase classname="c" name="p"/>'
                 '<testcase classname="c" name="f"><failure message="m"/></testcase>'
                 '<testcase classname="c" name="e"><error message="m"/></testcase>'
                 '<testcase classname="c" name="s"><skipped message="why not"/></testcase>'
                 '<testcase classname="c" name="d"/><testcase classname="c" name="d"><error message="td"/></testcase>'
                 '</testsuite></testsuites>', encoding="utf-8")
    assert LG._parse_junit_outcomes(x) == {"c::p": "pass", "c::f": "fail", "c::e": "error", "c::s": "skip:why not",
                                           "c::d": "error"}


# ---------------------------------------------------------------------------------------------- #
# nothing moved: fingerprints and the ordinary path                                                 #
# ---------------------------------------------------------------------------------------------- #
def test_profile_dicts_do_not_contain_any_split_key_so_no_fingerprint_can_have_moved():
    keys = {k for p in LG.PROFILES.values() for k in p}
    assert not {k for k in keys if "split" in k or "complement" in k or "half" in k}


# ---------------------------------------------------------------------------------------------- #
# the ops scripts                                                                                   #
# ---------------------------------------------------------------------------------------------- #
OPS = ROOT / "ops" / "r9e1_gate3"


@pytest.mark.parametrize("name,markers", [
    ("r9gate3_thor_pinned_half.sh", ("ZZR9PH-${ARM}-OKZZ", "ZZR9PH-${ARM}-FAILZZ", "--pinned-half thor",
                                     "R9_THOR_GIT_DIR", "launch_tree", "export R9_STUDY=e3", "runs.d/${R9_PREFIX}_${ARM}.argv.json")),
    ("r9gate3_devbox_complement.sh", ("ZZR9DC-${ARM}-OKZZ", "ZZR9DC-${ARM}-FAILZZ", "--pinned-half devbox-complement",
                                      "--pinned-thor-half", "devbox_evidence_${C12}_${ARM}", "tanitad-thor")),
])
def test_ops_scripts_parse_and_carry_their_markers(name, markers):
    p = OPS / name
    assert p.is_file(), p
    txt = p.read_text(encoding="utf-8")
    assert "\r" not in txt
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not on PATH")
    r = subprocess.run([bash, "-n", str(p)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    for m in markers:
        assert m in txt, m
    assert "pgrep -f" not in txt and "pkill -f" not in txt
