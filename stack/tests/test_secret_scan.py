"""Falsifiers for ``tools/secret_scan.py`` -- the C111 credential gate.

⛔ **PINNED BOTH WAYS, ON PURPOSE.** This programme shipped a rejects-everything
guard and a passes-everything guard inside one day (C95/C97), so half these tests
prove the scanner CATCHES a planted credential and the other half prove it does
NOT fire on this repo's real content. Either half alone is the failure mode.

⚠️ **EVERY PLANTED TOKEN IN THIS FILE IS ASSEMBLED AT RUNTIME FROM FRAGMENTS.**
Not one credential-shaped literal appears in the source. That is not fastidiousness:
this file is itself inside the tree the scanner scans, and a whole literal makes
the repo-wide gate fire on its own test suite -- the same self-match trap as a
polling monitor whose filter contains the pattern it greps for. MEASURED
2026-08-18: a single PEM literal in ``tools/tests/test_safe_commit.py`` was 1 of
the 7 blocking findings in the first whole-repo run.

⚠️ **The tests that matter most are the BINDING ones at the bottom.** C108's class
-- and C111's real root cause -- is not a missing mechanism but a correct
mechanism nothing calls: ``safe_commit.py`` has caught this exact token shape
since 2026-07-25 and was never wired to anything. ``test_hook_is_installed_and_current``
and ``test_installed_hook_actually_refuses_a_commit`` are what stop that
happening again.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tools"))

import secret_scan as ss  # noqa: E402


# --------------------------------------------------------------- fixtures/helpers
#
# Assembled, never literal. See the module docstring.
def _hf() -> str:
    return "hf" + "_" + ("Zq7" * 12)[:34]


def _sk() -> str:
    return "sk" + "-" + ("Ab9" * 14)[:40]


def _ghp() -> str:
    return "gh" + "p_" + ("Kk3" * 12)[:36]


def _gh_fine() -> str:
    return "github" + "_pat_" + ("Xy4" * 20)[:56]


def _akia() -> str:
    return "AK" + "IA" + "QRSTUVWX23456789"


def _asia() -> str:
    return "AS" + "IA" + "QRSTUVWX23456789"


def _xox() -> str:
    return "xo" + "xb-" + "1234567890-0987654321-abcdefghij"


def _aiza() -> str:
    return "AI" + "za" + "Sy" + ("B7k" * 12)[:33]


def _pem() -> str:
    return "-----BEGIN RSA PRIVATE" + " KEY-----"


def _high_entropy_value() -> str:
    return "v4Xq8Lm2Pd7Rt5Yn3Bw9Zc6Hj1Ks4Gf"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir(parents=True)
    _git(r, "init", "-b", "work")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(r, "add", "seed.txt")
    _git(r, "commit", "-m", "seed")
    return r


def _plant(repo: Path, rel: str, payload: str, *, line: int = 11,
           binary: bool = False) -> Path:
    """Reproduce C111's shape: a payload buried at a given line of a run log."""
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        p.write_bytes(b"\x00\x01\x02header\n" + payload.encode() + b"\n\x00tail")
    else:
        body = [f"[boot] line {i}" for i in range(1, line)]
        body.append(payload)
        body += [f"[post] line {i}" for i in range(line + 1, line + 9)]
        p.write_text("\n".join(body) + "\n", encoding="utf-8")
    return p


def _stage(repo: Path, rel: str, payload: str, *, binary: bool = False,
           force: bool = False, line: int = 11) -> None:
    _plant(repo, rel, payload, line=line, binary=binary)
    _git(repo, "add", *(["-f"] if force else []), "--", rel)


# =============================================================== IT MUST CATCH


def test_c111_exact_shape_is_caught_in_a_log(repo):
    """The incident, reproduced: an HF token on line 11 of a rescued run log.

    C111's own words -- our procedure staged it, committed it, and would have
    pushed it; GitHub's push protection was the only control that fired."""
    rel = "rescued/rq_out/logs/contention.log"
    _stage(repo, rel, f"hf download --token {_hf()} Sayood/x")

    staged = ss.scan_staged(repo)
    assert staged.blocking, "the C111 shape must block a commit"
    f = staged.blocking[0]
    assert f.pattern == "huggingface"
    assert f.path == rel
    assert f.line == 11, "the finding must name the LINE, as GH013 did"

    tree = ss.scan_tree(repo)
    assert tree.blocking, "C111's rule is 'scanned BEFORE it is staged'"


@pytest.mark.parametrize("ext", [".log", ".json", ".txt", ".md", ".yaml", ".sh"])
def test_a_planted_token_is_caught_whatever_the_extension(repo, ext):
    """C111's lesson was that the exhaust is unguarded, not the key store. The
    scan must judge by CONTENT, never by extension."""
    rel = f"rescued/x{ext}"
    _stage(repo, rel, f'  "hf_token": "{_hf()}",')
    assert ss.scan_staged(repo).blocking


@pytest.mark.parametrize("name,payload", [
    ("huggingface", _hf),
    ("openai-anthropic", _sk),
    ("github-pat", _ghp),
    ("github-fine", _gh_fine),
    ("aws-akid", _akia),
    ("aws-akid", _asia),
    ("slack", _xox),
    ("google-api", _aiza),
    ("private-key", _pem),
])
def test_every_provider_shape_is_caught(name, payload):
    findings = ss.scan_bytes(f"cfg = {payload()}\n".encode(), "x.log")
    blocking = [f for f in findings if f.blocking]
    assert blocking, f"{name} shape was not caught"
    assert blocking[0].pattern == name


def test_a_token_inside_a_BINARY_blob_is_caught(repo):
    """⛔ THE GAP THE OLD SCANNER HAD, and the reason this module exists at all.

    ``safe_commit`` scanned ``git diff --cached`` -- diff TEXT. For a blob git
    classifies as binary that diff is the single line "Binary files ... differ"
    and carries no content, so a token inside one was structurally invisible.
    MEASURED 2026-08-18: the old scan missed this case; reading the BLOB via
    ``git cat-file`` catches it."""
    _stage(repo, "rescued/state.bin", f"tok {_hf()}", binary=True)
    rep = ss.scan_staged(repo)
    assert rep.blocking, "a token in a binary blob must still block"
    assert rep.blocking[0].pattern == "huggingface"


def test_generic_high_entropy_assignment_is_caught(repo):
    _stage(repo, "rescued/settings.yaml", f'api_key = "{_high_entropy_value()}"')
    rep = ss.scan_staged(repo)
    assert [f for f in rep.blocking if f.pattern == "generic-assignment"]


@pytest.mark.parametrize("key", ["api_key", "apikey", "secret", "token",
                                 "password", "access_key", "client_secret"])
def test_each_generic_key_name_is_covered(key):
    findings = ss.scan_bytes(f'{key}: "{_high_entropy_value()}"\n'.encode(), "c.yaml")
    assert [f for f in findings if f.blocking], f"{key} assignment not caught"


def test_a_gitignored_file_staged_with_force_is_caught(repo):
    """The only way ``Keys.txt`` reaches an index is ``git add -f``, and that
    leaves this exact fingerprint. Probed with git's OWN ignore verdict -- the
    tool that owns the fact."""
    (repo / ".gitignore").write_text("Keys.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _stage(repo, "Keys.txt", "notes about keys", force=True)
    rep = ss.scan_staged(repo)
    assert [f for f in rep.blocking
            if f.pattern in ("git-ignored-but-staged", "credential-filename")]


@pytest.mark.parametrize("rel", ["host.pem", "id_ed25519", "app.key",
                                 ".env", ".netrc", "gotty_url.txt",
                                 "my_secret_stuff.txt"])
def test_credential_shaped_filenames_are_caught(rel):
    findings = ss.scan_path_shape(f"rescued/{rel}")
    assert [f for f in findings if f.blocking], f"{rel} not flagged by path"


# ========================================================== IT MUST NOT FIRE
#
# Every case below is REAL CONTENT from this repo, or a shape measured to have
# produced a false positive in the first whole-repo run.


def test_ordinary_hf_identifiers_are_advisory_not_blocking():
    """The brief's loose ``hf_[A-Za-z0-9]+`` matches this repo's own committed
    ``hf_export.py`` / ``hf_relay`` / ``hf_repo_state_2026-07-25.json``. Blocking
    on those makes the gate unusable, so Tier A carries the real length floor and
    the loose form stays ADVISORY. Nothing is dropped silently -- it is counted.

    The identifiers below are REAL matches taken from this repo's own committed
    code (``hf_download``, ``hf_flagship``), not invented ones."""
    src = ("def hf_relay():\n"
           "    return hf_download(hf_flagship, 'hf_repo_state')\n")
    findings = ss.scan_bytes(src.encode(), "tools/hf_export.py")
    assert not [f for f in findings if f.blocking]
    assert [f for f in findings if not f.blocking], "must still be REPORTED"


def test_the_scanner_itself_is_not_flagged_by_its_own_path_globs():
    """⛔ MEASURED SELF-MATCH, 2026-08-18: the glob ``*secret*`` fired on
    ``tools/secret_scan.py`` and on this very file, turning the repo-wide gate
    red on the day it was written -- the polling-monitor trap in the path tier."""
    for rel in ("tools/secret_scan.py", "stack/tests/test_secret_scan.py",
                "docs/credentials.md", "notes/secrets_policy.rst"):
        assert not [f for f in ss.scan_path_shape(rel) if f.blocking], rel
    # ...but a real credential CONTAINER with the same words still blocks.
    for rel in ("conf/secrets.yaml", "conf/client_credentials.json"):
        assert [f for f in ss.scan_path_shape(rel) if f.blocking], rel


def test_a_code_expression_assigned_to_tokens_is_not_a_finding():
    """⚠️ MEASURED FALSE POSITIVE, 2 of the first run's 7 blocking findings.

    ``tokens = head.build_tokens(st4`` is 21 chars at 4.01 bits/char and sailed
    through the entropy gate. Real credentials come from a narrow alphabet; code
    expressions carry brackets and dotted calls."""
    src = "        tokens = head.build_tokens(st4, None)\n"
    assert not [f for f in ss.scan_bytes(src.encode(), "v5_sel.py") if f.blocking]


def test_a_dotted_identifier_assigned_to_a_secret_name_is_not_a_finding():
    src = "secret = config.runtime.credentials_provider\n"
    assert not [f for f in ss.scan_bytes(src.encode(), "x.py") if f.blocking]


@pytest.mark.parametrize("digest", [
    "d41d8cd98f00b204e9800998ecf8427e",                                  # md5
    "da39a3ee5e6b4b0d3255bfef95601890afd80709",                          # sha1
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # sha256-ish
])
def test_hex_digests_are_not_credentials(digest):
    """This repo commits thousands of md5/sha256 digests -- every rescue manifest
    is a wall of them -- and they are maximally high-entropy by construction.
    Without this gate the generic rule is pure noise."""
    src = f'token = "{digest}"\n'
    assert not [f for f in ss.scan_bytes(src.encode(), "MANIFEST.md") if f.blocking]


@pytest.mark.parametrize("val", [
    "your_api_key_here_please", "REDACTED_BY_POLICY_XXXX", "${HF_TOKEN_ENV_VAR}",
    "os.environ['HF_TOKEN_X']", "changeme_changeme_1234", "<insert-token-here-1>",
    "xxxxxxxxxxxxxxxxxxxxxxxx", "example_token_value_123",
])
def test_placeholders_are_not_credentials(val):
    src = f'api_key = "{val}"\n'
    assert not [f for f in ss.scan_bytes(src.encode(), "README.md") if f.blocking]


def test_a_supervisor_run_manifest_env_is_advisory_not_blocking():
    """⚠️ MEASURED FALSE POSITIVE, 4 of the first run's 7 blocking findings, and
    4 of 4 were artifacts. In this programme ``*.env`` is the supervisor RUN
    MANIFEST convention (``stack/ops/runs.d/*.env``), not a dotenv secret store.
    A gate that refuses a normal deliverable is switched off inside a week."""
    findings = ss.scan_path_shape("stack/ops/runs.d/flagship-v5f-w120-30k.env")
    assert not [f for f in findings if f.blocking]
    assert findings and not findings[0].blocking, "downgraded, not deleted"


def test_dotenv_proper_is_still_blocking():
    """The narrowing must not swallow the shape it was narrowed around."""
    assert [f for f in ss.scan_path_shape("svc/.env") if f.blocking]
    assert [f for f in ss.scan_path_shape("svc/.env.production") if f.blocking]


def test_the_scanner_does_not_match_its_own_source():
    """A scanner that fires on itself is the polling-monitor self-match trap in
    another costume -- and it would make the repo-wide gate permanently red."""
    for rel in ("tools/secret_scan.py", "tools/safe_commit.py",
                "stack/tests/test_secret_scan.py", "tools/tests/test_safe_commit.py"):
        p = _REPO / rel
        if not p.exists():
            continue
        findings = ss.scan_bytes(p.read_bytes(), rel)
        assert not [f for f in findings if f.blocking], \
            f"{rel} matches the scanner's own patterns"


def test_repo_source_surface_is_clean():
    """The fast standing pin: every source tree that a human or agent edits.

    The FULL 6,192-file / 661 MB tracked scan takes ~27 s -- over ci_gate's 15 s
    per-test budget -- so it is opt-in below. This subset is where a regression
    would actually appear (a planted literal in a fixture, a new pattern that
    matches ordinary code) and it runs in a couple of seconds."""
    total_blocking = []
    for sub in ("tools", "stack/scripts", "stack/tanitad", "stack/tests",
                "taniteval", ".github"):
        d = _REPO / sub
        if not d.is_dir():
            continue
        rep = ss.scan_tree(d, scope=f"source:{sub}")
        total_blocking += [(sub, f) for f in rep.blocking]
    assert not total_blocking, (
        "blocking findings in the repo's own source surface: "
        + ", ".join(f"{s}:{f.path}:{f.line}[{f.pattern}]" for s, f in total_blocking))


@pytest.mark.skipif(os.environ.get("SECRET_SCAN_FULL") != "1",
                    reason="full 6,192-file tracked scan takes ~27 s, over "
                           "ci_gate's 15 s per-test budget. Run it with "
                           "SECRET_SCAN_FULL=1, or directly: "
                           "python tools/secret_scan.py --tracked")
def test_full_tracked_tree_is_clean():
    """MEASURED 2026-08-18: 6,192 files / 661.6 MB / **0 blocking** / 64 advisory."""
    rep = ss.scan_tracked(_REPO)
    assert not rep.blocking, [f"{f.path}:{f.line}[{f.pattern}]" for f in rep.blocking]


# ================================================================ IT MUST NOT LEAK


def test_a_matched_value_is_never_printed(repo, capsys):
    """C111 was handled by naming the file and line and never reproducing the
    token. The tool must make that the only possible behaviour."""
    secret = _hf()
    _stage(repo, "rescued/run.log", f"--token {secret}")
    rc = ss.main(["--repo", str(repo), "--staged"])
    out = capsys.readouterr()
    assert rc == 1
    blob = out.out + out.err
    assert secret not in blob, "the scanner printed the credential it caught"
    assert "redacted" in blob
    assert "rescued/run.log" in blob, "it must still say WHERE"


def test_the_json_report_never_carries_a_value(repo, tmp_path):
    secret = _hf()
    _stage(repo, "rescued/run.log", f"--token {secret}")
    out = tmp_path / "r.json"
    ss.main(["--repo", str(repo), "--staged", "--json", str(out)])
    text = out.read_text(encoding="utf-8")
    assert secret not in text
    assert json.loads(text)["blocking_count"] >= 1


def test_keys_txt_contents_are_never_emitted(repo, capsys):
    """``Keys.txt`` is git-ignored and must never be read into an argument,
    printed, or committed. The scanner has to handle it without ever emitting it
    -- including when its contents genuinely match."""
    secret = _hf()
    (repo / "Keys.txt").write_text(f"HF_WRITE={secret}\nnote: live\n",
                                   encoding="utf-8")
    rep = ss.scan_tree(repo)
    rendered = ss.render(rep)
    assert secret not in rendered
    assert "HF_WRITE" not in rendered or "redacted" in rendered
    assert "Keys.txt" in rendered, "it must still be REPORTED, by path"
    assert secret not in json.dumps(rep.to_json())


def test_redact_keeps_no_more_than_a_scheme_prefix():
    secret = _hf()
    red = ss.redact(secret)
    assert secret not in red
    assert secret[8:] not in red
    assert str(len(secret)) in red


# ===================================================== IT MUST ACTUALLY BE CALLED
#
# ⭐ C108's class, and C111's real root cause: a correct mechanism that nothing
# invokes. `safe_commit.py` has caught this token shape since 2026-07-25 and the
# 2026-07-26 program harvest recorded it as "imported_by: Nothing". These are the
# tests that make that impossible to repeat quietly.


def test_hook_is_installed_and_current():
    """⛔ THE ANTI-C108 TEST. If this fails, the credential gate is NOT armed on
    this clone/worktree and a raw ``git commit`` is unguarded."""
    state, path = ss.hook_status(_REPO)
    assert state == "current", (
        f"pre-commit credential gate is '{state}' at {path}. "
        f"Arm it with:  python tools/secret_scan.py --install-hook")


def test_installed_hook_actually_refuses_a_commit(repo):
    """End-to-end through real git: install the hook, plant a token, and prove
    ``git commit`` FAILS. Asserting the hook file exists proves nothing about
    whether git runs it."""
    shutil.copytree(_REPO / "tools", repo / "tools",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    ok, msg = ss.install_hook(repo)
    assert ok, msg
    _stage(repo, "rescued/run.log", f"--token {_hf()}")
    _git(repo, "add", "--", "tools")
    proc = _git(repo, "commit", "-m", "should be refused")
    assert proc.returncode != 0, (
        "git commit SUCCEEDED with a planted token staged -- the hook did not run"
    )
    combined = proc.stdout + proc.stderr
    assert "secret_scan" in combined
    assert _hf() not in combined
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert head, "repo should still have its seed commit"
    assert _git(repo, "log", "--oneline").stdout.count("\n") == 1


def test_the_hook_lets_a_clean_commit_through(repo):
    """The other half: a guard that refuses everything is the C95/C97 failure."""
    shutil.copytree(_REPO / "tools", repo / "tools",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    ok, _ = ss.install_hook(repo)
    assert ok
    _stage(repo, "notes/ordinary.md", "nothing secret here, just prose")
    _git(repo, "add", "--", "tools")
    proc = _git(repo, "commit", "-m", "ordinary work")
    assert proc.returncode == 0, (
        f"the hook refused a CLEAN commit: {proc.stdout}\n{proc.stderr}")


def test_the_hook_fails_CLOSED_when_python_is_missing():
    """⚠️ An unrunnable gate must not read as a pass. That is the `pgrep`/monitor
    lesson: absence of a finding is only reassuring if the check actually ran."""
    body = ss.HOOK_BODY
    tail = body[body.index("no python on PATH"):]
    assert "exit 1" in tail
    assert "exit 0" not in tail


def test_install_refuses_to_clobber_a_foreign_hook(repo):
    hooks = ss.hooks_dir(repo)
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\necho someone elses hook\n",
                                      encoding="utf-8")
    ok, msg = ss.install_hook(repo)
    assert not ok and "REFUSING" in msg
    ok2, _ = ss.install_hook(repo, force=True)
    assert ok2
    assert ss.hook_status(repo)[0] == "current"


def test_install_hook_is_idempotent(repo):
    ss.install_hook(repo)
    state_1 = ss.hook_status(repo)
    ok, msg = ss.install_hook(repo)
    assert ok and "already current" in msg
    assert ss.hook_status(repo) == state_1


def test_hook_survives_a_stale_version(repo):
    ss.install_hook(repo)
    path = ss.hooks_dir(repo) / "pre-commit"
    path.write_text(ss.HOOK_BODY + "\n# drifted\n", encoding="utf-8")
    assert ss.hook_status(repo)[0] == "stale"
    ss.install_hook(repo)
    assert ss.hook_status(repo)[0] == "current"


# ===================================================== the filter is stated as data


def test_skipped_files_are_counted_AND_named(repo):
    """C110: a count is a claim about the filter until the filter is stated. A
    file the scan could not have found the answer in must be visible, not merely
    tallied."""
    (repo / "big.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 1000)
    rep = ss.scan_tree(repo)
    assert rep.skipped.get("compressed-suffix", 0) >= 1
    assert "big.zip" in rep.skipped_paths.get("compressed-suffix", [])
    assert "_warning" in rep.filter_rule


def test_the_filter_rule_travels_with_the_json(repo):
    rep = ss.scan_tree(repo)
    doc = rep.to_json()
    assert doc["filter_rule"]["tier_a_patterns"]
    assert "generic-assignment" not in doc["filter_rule"]["tier_a_patterns"]
    assert doc["filter_rule"]["tier_a2_generic_assignment"]["gates"]


def test_paths_with_spaces_survive_the_staged_listing(repo):
    """⚠️ This repo's paths contain spaces (``TanitAD Research Hub/...``) and an
    unquoted read degenerates into a sweep that reports success on nothing. That
    trap has caught three separate streams, including during C117's own
    verification."""
    rel = "TanitAD Research Hub/Some Area/run log.log"
    _stage(repo, rel, f"--token {_hf()}")
    assert rel in ss.staged_paths(repo)
    rep = ss.scan_staged(repo)
    assert rep.blocking and rep.blocking[0].path == rel


def test_a_scan_that_read_NOTHING_is_unusable_not_clean(repo):
    """⛔ MEASURED FALSE ALL-CLEAR, 2026-08-18.

    The Google-Drive-backed volume this repo lives on dropped mid-session. Every
    ``git cat-file`` returned 128, and ``--history`` printed *"files scanned = 0
    ... BLOCKING (0) -- clean"* and exited **0**. A perfect pass, produced by
    reading nothing. ⇒ Candidates-without-reads and any git/read error now make
    the whole report UNUSABLE, which is a non-zero exit, because absence of
    evidence is an alarm and never an all-clear."""
    rep = ss.Report(scope="synthetic", candidates=400, files_scanned=0)
    assert rep.unusable
    assert not rep.blocking
    assert "ZERO were read" in (rep.unusable_reason or "")
    assert "UNUSABLE" in ss.render(rep)

    rep2 = ss.Report(scope="synthetic", candidates=10, files_scanned=10,
                     errors=["git cat-file --batch failed (rc=128)"])
    assert rep2.unusable, "a read error must invalidate the verdict"

    clean = ss.scan_staged(repo)
    assert not clean.unusable, "an empty index is genuinely clean, not unusable"


def test_unusable_reports_exit_non_zero(repo, capsys, monkeypatch):
    """The verdict has to reach the EXIT CODE, or every caller still sees a pass."""
    real = ss.scan_staged

    def broken(r, max_bytes=ss.DEFAULT_MAX_BYTES):
        rep = real(r, max_bytes)
        rep.candidates, rep.files_scanned = 400, 0
        return rep

    monkeypatch.setattr(ss, "scan_staged", broken)
    rc = ss.main(["--repo", str(repo), "--staged"])
    assert rc == 1
    assert "UNUSABLE" in capsys.readouterr().out


def test_the_report_json_carries_the_unusable_flag():
    doc = ss.Report(scope="s", candidates=5, files_scanned=0).to_json()
    assert doc["unusable"] is True
    assert doc["candidates"] == 5


def test_not_a_git_repo_exits_3(tmp_path, capsys):
    assert ss.main(["--repo", str(tmp_path), "--staged"]) == 3


def test_clean_tree_exits_0(repo, capsys):
    _stage(repo, "notes/x.md", "ordinary prose")
    assert ss.main(["--repo", str(repo), "--staged"]) == 0


# ================================== PARTIAL failure is UNUSABLE, never clean
#
# ⛔ THE MEASURED HOLE (2026-08-18, the Google-Drive mount flapping): the
# history scan's `git cat-file --batch` calls failed rc=128 over ~1,200
# objects, the failures were PRINTED in scrollback -- and the run still ended
# "BLOCKING (0) -- clean", exit 0. The 0-read guard above could not see it
# because SOME objects were read. A clean verdict is admissible ONLY when
# every enumerated object was actually scanned; anything less is
# `SCAN UNUSABLE (partial): read X of Y objects, Z batch failures`, non-zero.


def _commit_history_files(repo: Path, n: int = 4) -> None:
    """n distinct committed blobs, so --history has several chunks to read."""
    for i in range(n):
        (repo / f"hist{i}.log").write_text(f"[boot] ordinary line {i}\n",
                                           encoding="utf-8")
        _git(repo, "add", "--", f"hist{i}.log")
    _git(repo, "commit", "-m", "history bulk")


def _make_flaky_batch(monkeypatch, fail_on_call: int):
    """The mount flaps on exactly one batch: that call returns rc=128-shaped
    failure, every other call is the REAL batch runner."""
    real_batch = ss._cat_file_batch
    calls = {"n": 0}

    def flaky(r, revs, max_bytes):
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            return {}, {}, [
                f"git cat-file --batch failed (rc=128) over {len(revs)} "
                f"object(s): fatal: not a git repository"], True
        return real_batch(r, revs, max_bytes)

    monkeypatch.setattr(ss, "_cat_file_batch", flaky)


def test_history_batch_failure_mid_scan_is_unusable_partial_not_clean(
        repo, capsys, monkeypatch):
    """The incident, PARTIAL variant: chunk 2 of 3 dies rc=128, chunks 1 and 3
    read fine. files_scanned is healthy-looking and nonzero -- and the verdict
    must still be UNUSABLE with a non-zero exit, never 'clean'."""
    _commit_history_files(repo, 4)                # + seed.txt = 5 blobs
    real_scan = ss.scan_history
    _make_flaky_batch(monkeypatch, fail_on_call=2)
    monkeypatch.setattr(ss, "scan_history", lambda root: real_scan(root, batch=2))

    rc = ss.main(["--repo", str(repo), "--history"])
    out = capsys.readouterr().out
    assert rc != 0, "a scan that lost objects must not exit 0"
    assert "SCAN UNUSABLE (partial): read 3 of 5 objects, 1 batch failures" in out
    assert "BLOCKING (0) -- clean" not in out, \
        "the false all-clear summary line is exactly the measured defect"


def test_history_partial_accounting_is_exact_and_travels_in_json(
        repo, monkeypatch):
    """read + unread + policy-skips must equal the enumeration, and the
    accounting must reach the JSON, not just the prose."""
    _commit_history_files(repo, 4)
    _make_flaky_batch(monkeypatch, fail_on_call=2)

    rep = ss.scan_history(repo, batch=2)
    assert rep.partial and rep.unusable
    assert rep.batch_failures == 1
    assert rep.objects_unread == 2                # the failed chunk of 2
    assert rep.candidates == 5 and rep.files_scanned == 3
    assert rep.files_scanned + rep.objects_unread == rep.candidates
    doc = rep.to_json()
    assert doc["unusable"] is True and doc["partial"] is True
    assert doc["batch_failures"] == 1 and doc["objects_unread"] == 2


def test_an_enumerated_object_that_reads_missing_is_a_failure_not_a_skip(
        repo, monkeypatch):
    """git itself answers '<sha> missing' for an object the enumeration
    certified moments earlier: the store lost it mid-scan (flap / concurrent
    prune). In --history that is an object-read failure. ('missing-or-not-blob'
    stays a legitimate SKIP only in --staged, where a submodule gitlink really
    has no blob to read.) Exercised through the REAL git and the REAL parser:
    one requested sha is swapped for one that does not exist."""
    real_batch = ss._cat_file_batch

    def vanishing(r, revs, max_bytes):
        return real_batch(r, ["0" * 40] + list(revs)[1:], max_bytes)

    monkeypatch.setattr(ss, "_cat_file_batch", vanishing)
    rep = ss.scan_history(repo)
    assert rep.partial and rep.unusable
    assert rep.objects_unread == 1
    assert rep.batch_failures == 0, "no command failed; an OBJECT vanished"
    assert not rep.skipped.get("missing-or-not-blob"), \
        "in history this is reclassified as a failure, never a skip"
    assert any("missing" in e for e in rep.errors), "and it is named loudly"


def test_non_blob_bodies_are_consumed_not_desynced(repo):
    """`git cat-file --batch` STREAMS a body for tree/commit/tag objects too.
    The old parser skipped a non-blob header without consuming its body, so
    every later header in the batch was read from inside the previous body --
    a silent stream desync that dropped real blobs from the scan (and, in
    --history, would now surface as phantom unread objects)."""
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    tree = _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    blob = _git(repo, "rev-parse", "HEAD:seed.txt").stdout.strip()
    got, skipped, errors, failed = ss._cat_file_batch(
        repo, [head, tree, blob], ss.DEFAULT_MAX_BYTES)
    assert blob in got, "the blob AFTER a commit+tree must still be read"
    assert got[blob] == b"seed\n"
    assert skipped.get("missing-or-not-blob") == 2, "commit+tree, counted"
    assert not errors and not failed


def test_history_enumeration_failure_is_unusable_not_an_empty_clean(
        repo, monkeypatch):
    """rc=128 on the ENUMERATION itself used to yield candidates=0, which
    sailed past every guard -- a perfect clean over an object DB never seen."""
    real_git = ss.git

    def flaky_git(r, *args, **kw):
        if "--batch-all-objects" in args:
            return subprocess.CompletedProcess(
                ["git", *args], 128, b"", b"fatal: not a git repository")
        return real_git(r, *args, **kw)

    monkeypatch.setattr(ss, "git", flaky_git)
    rep = ss.scan_history(repo)
    assert rep.unusable and rep.partial
    assert rep.batch_failures >= 1
    assert "read 0 of 0 objects" in (rep.unusable_reason or "")


def test_history_rev_list_failure_invalidates_the_verdict(repo, monkeypatch):
    """A failed rev-list silently labelled EVERY blob UNREACHABLE, demoting a
    committed credential to non-blocking -- exit 0 with the finding buried as
    advisory. The labels feed the verdict, so their source failing is a scan
    failure."""
    real_git = ss.git

    def flaky_git(r, *args, **kw):
        if args[:2] == ("rev-list", "--objects"):
            return subprocess.CompletedProcess(
                ["git", *args], 128, b"", b"fatal: unable to read tree")
        return real_git(r, *args, **kw)

    monkeypatch.setattr(ss, "git", flaky_git)
    rep = ss.scan_history(repo)
    assert rep.unusable and rep.partial
    assert rep.batch_failures >= 1
    assert any("rev-list" in e for e in rep.errors)


def test_history_happy_path_scans_every_object_and_stays_clean(repo, capsys):
    """The other half of the pin (C95/C97): the guard must not turn a healthy
    repo red. Clean is earned by reading EVERYTHING the enumeration listed."""
    _commit_history_files(repo, 2)
    rc = ss.main(["--repo", str(repo), "--history"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BLOCKING (0) -- clean" in out
    assert "UNUSABLE" not in out
    rep = ss.scan_history(repo)
    assert not rep.unusable
    assert rep.batch_failures == 0 and rep.objects_unread == 0
    assert rep.candidates == rep.files_scanned > 0, \
        "clean is admissible ONLY when every enumerated object was scanned"


def test_a_committed_token_is_still_caught_by_history(repo, capsys):
    """Regression guard: the partial-failure guard must not blunt DETECTION.
    A token committed into history is REACHABLE, blocking, exit 1 -- and the
    value itself is never printed."""
    _plant(repo, "rescued/run.log", f"--token {_hf()}")
    _git(repo, "add", "--", "rescued/run.log")
    _git(repo, "commit", "-m", "rescued log (oops)")
    rc = ss.main(["--repo", str(repo), "--history"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "huggingface" in out and "REACHABLE" in out
    assert _hf() not in out
