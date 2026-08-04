"""Tests for tools/corpus_census.py.

The bug class these guard against is not "the code throws". It is
**a failed probe being read as an absence** — which is exactly how the
2026-08-03 "the parity corpus is on no live machine" claim was manufactured
(it was on Thor the whole time), and how a chunked relay once reported
``have=0`` while 29 files existed.

So the tests are weighted toward the UNKNOWN/ABSENT boundary and toward the
copy-counting rule, not toward formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import corpus_census as cc  # noqa: E402


# ---------------------------------------------------------------------------
# The remote payload / parser boundary
# ---------------------------------------------------------------------------
def test_parse_remote_output_reads_present_and_absent():
    text = ("OK|/a/b|dir|2376|278780000000\n"
            "MISS|/c/d|-|0|0\n"
            "__CENSUS_DONE__\n")
    res, done = cc.parse_remote_output(text)
    assert done is True
    assert res["/a/b"]["status"] == cc.PRESENT
    assert res["/a/b"]["count"] == 2376
    assert res["/a/b"]["bytes"] == 278780000000
    assert res["/c/d"]["status"] == cc.ABSENT


def test_truncated_payload_is_not_completed():
    """No sentinel => the stream was cut. This is the have=0 defect."""
    text = "OK|/a/b|dir|10|100\nMISS|/c/d|-|0|0\n"
    res, done = cc.parse_remote_output(text)
    assert done is False
    assert res["/a/b"]["status"] == cc.PRESENT   # positive evidence survives
    assert res["/c/d"]["status"] == cc.ABSENT    # downgraded by probe_host


def test_probe_host_downgrades_absent_to_unknown_on_truncation(monkeypatch):
    """A MISS from an incomplete stream MUST become UNKNOWN.

    This is the single most important assertion in the file: reading it as
    ABSENT is what turns a network hiccup into a fabricated single-copy risk.
    """
    class FakeProc:
        stdout = b"OK|/a|dir|5|50\nMISS|/b|-|0|0\n"   # no sentinel
        stderr = b"Connection reset by peer\n"

    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: FakeProc())
    res, err = cc.probe_host("h", [("/a", "*.pt"), ("/b", "*.pt")], 10)
    assert err is not None and "incomplete" in err
    assert res["/a"]["status"] == cc.PRESENT
    assert res["/b"]["status"] == cc.UNKNOWN, "truncated MISS must not read as ABSENT"


def test_probe_host_sends_payload_in_argv_not_stdin(monkeypatch):
    """Regression: `ssh -n` redirects stdin from /dev/null.

    Measured 2026-08-03 — the first live census run reported every host as
    "incomplete payload (no sentinel and no stderr)" because the script was
    piped to `ssh -n ... sh` and `-n` discarded it. That reads as a fleet-wide
    outage. The payload must travel in argv, and stdin must be DEVNULL.
    """
    seen: dict = {}

    class FakeProc:
        stdout = b"OK|/a|dir|5|50\n__CENSUS_DONE__\n"
        stderr = b""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return FakeProc()

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    res, err = cc.probe_host("h", [("/a", "*.pt")], 10)
    assert err is None and res["/a"]["status"] == cc.PRESENT
    assert "-n" not in seen["cmd"], "`ssh -n` would discard the payload"
    assert seen["kw"].get("stdin") is cc.subprocess.DEVNULL
    assert "input" not in seen["kw"], "payload must not go via stdin"
    assert "__CENSUS_DONE__" in seen["cmd"][-1], "payload must be the argv tail"


def test_probe_host_timeout_yields_no_absences(monkeypatch):
    def boom(*a, **k):
        raise cc.subprocess.TimeoutExpired(cmd="ssh", timeout=5)

    monkeypatch.setattr(cc.subprocess, "run", boom)
    res, err = cc.probe_host("h", [("/a", "*.pt")], 5)
    assert res == {}
    assert "timeout" in err


def test_build_remote_script_quotes_paths_with_spaces():
    script = cc.build_remote_script([("/a b/c", "*.pt")])
    assert "'/a b/c::*.pt'" in script
    assert "__CENSUS_DONE__" in script


# ---------------------------------------------------------------------------
# The member contract: a short directory is not a copy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count,expected,want", [
    (2376, 2376, cc.PRESENT),
    (2400, 2376, cc.PRESENT),     # more is fine (skip markers, DONE)
    (439, 2376, cc.PARTIAL),      # Thor's partial train pull
    (0, 2376, cc.ABSENT),
    (None, 2376, cc.UNKNOWN),
    (1, None, cc.PRESENT),        # single-file artifact
])
def test_classify_member_contract(count, expected, want):
    assert cc.classify(cc.PRESENT, count, expected) == want


def test_classify_never_upgrades_a_non_present_status():
    for st in (cc.UNKNOWN, cc.UNREACHABLE, cc.ABSENT):
        assert cc.classify(st, 2376, 2376) == st


# ---------------------------------------------------------------------------
# Copy counting
# ---------------------------------------------------------------------------
def _art(**kw):
    base = dict(key="k", kind="corpus", desc="d", members=10,
                candidates=[], pattern="ep_*.pt")
    base.update(kw)
    return cc.Artifact(**base)


def test_two_paths_on_one_host_are_one_copy():
    """epcache and epcache_prefix on Thor die together."""
    art = _art(candidates=[("thor", "/a"), ("thor", "/b")])
    host_results = {"thor": {"/a": {"status": cc.PRESENT, "count": 10, "bytes": 1},
                             "/b": {"status": cc.PRESENT, "count": 10, "bytes": 1}}}
    c = cc.build_census(host_results, {}, {}, {}, artifacts=[art])
    assert c["artifacts"]["k"]["copies"] == 1
    assert c["artifacts"]["k"]["verdict"] == "SINGLE_COPY"


def test_partial_does_not_count_as_a_copy():
    art = _art(candidates=[("thor", "/a"), ("pod5", "/b")])
    host_results = {
        "thor": {"/a": {"status": cc.PRESENT, "count": 4, "bytes": 1}},   # short
        "pod5": {"/b": {"status": cc.PRESENT, "count": 10, "bytes": 1}},
    }
    c = cc.build_census(host_results, {}, {}, {}, artifacts=[art])
    entry = c["artifacts"]["k"]
    assert entry["copies"] == 1
    assert entry["partial_locations"] == ["thor:/a"]
    assert entry["verdict"] == "SINGLE_COPY"


def test_unreachable_host_makes_verdict_unresolved_not_zero():
    """Nothing found + nothing probed is UNRESOLVED, never ZERO_COPIES."""
    art = _art(candidates=[("thor", "/a")])
    c = cc.build_census({}, {"thor": "ssh timeout"}, {}, {}, artifacts=[art])
    entry = c["artifacts"]["k"]
    assert entry["copies"] == 0
    assert entry["verdict"] == "UNRESOLVED"
    assert entry["locations"][0]["status"] == cc.UNREACHABLE
    assert any("UNKNOWN, not absent" in w for w in c["warnings"])


def test_genuine_absence_everywhere_is_zero_copies():
    art = _art(candidates=[("thor", "/a"), ("pod5", "/b")])
    host_results = {
        "thor": {"/a": {"status": cc.ABSENT, "count": 0, "bytes": 0}},
        "pod5": {"/b": {"status": cc.ABSENT, "count": 0, "bytes": 0}},
    }
    c = cc.build_census(host_results, {}, {}, {}, artifacts=[art])
    assert c["artifacts"]["k"]["verdict"] == "ZERO_COPIES"


def test_ok_with_unknown_location_is_flagged_as_lower_bound():
    art = _art(candidates=[("thor", "/a"), ("pod5", "/b"), ("pod4", "/c")])
    host_results = {
        "thor": {"/a": {"status": cc.PRESENT, "count": 10, "bytes": 1}},
        "pod5": {"/b": {"status": cc.PRESENT, "count": 10, "bytes": 1}},
    }
    c = cc.build_census(host_results, {"pod4": "refused"}, {}, {},
                        artifacts=[art])
    entry = c["artifacts"]["k"]
    assert entry["verdict"] == "OK"
    assert "LOWER BOUND" in entry["note"]


def test_a_rented_pod_is_not_a_durable_copy():
    """Two copies where one is a RunPod pod is one termination from SINGLE_COPY.

    pod2 was terminated on 2026-08-03 and pod1/pod3/eval went to `Connection
    refused` the same week, so this is not a hypothetical.
    """
    art = _art(candidates=[("hf:R", "pre"), ("pod5", "/b")])
    trees = {"R": {f"pre/ep_{i:05d}.pt": {"size": 1, "sha256": "x"}
                   for i in range(10)}}
    host_results = {"pod5": {"/b": {"status": cc.PRESENT, "count": 10,
                                    "bytes": 10}}}
    c = cc.build_census(host_results, {}, trees, {}, artifacts=[art])
    e = c["artifacts"]["k"]
    assert e["copies"] == 2 and e["verdict"] == "OK"
    assert e["durable_copies"] == 1
    assert e["volatile_machines"] == ["pod5"]
    assert "one termination from SINGLE_COPY" in e["volatility_warning"]


def test_two_durable_copies_carry_no_volatility_warning():
    art = _art(candidates=[("repo", "r"), ("thor", "/b")])
    host_results = {"thor": {"/b": {"status": cc.PRESENT, "count": 10,
                                    "bytes": 1}}}
    c = cc.build_census(host_results, {}, {}, {}, artifacts=[art],
                        repo_root=Path(__file__).parent)
    e = c["artifacts"]["k"]
    assert "volatility_warning" not in e


@pytest.mark.parametrize("host,durable", [
    ("hf:Sayood/x", True), ("github", True), ("repo", True), ("thor", True),
    ("pod4", False), ("pod5", False), ("eval", False),
])
def test_durability_tiers(host, durable):
    assert cc.is_durable_host(host) is durable


def test_path_missing_from_probe_output_is_unknown():
    """The host answered but said nothing about this path — not an absence."""
    art = _art(candidates=[("thor", "/a")])
    c = cc.build_census({"thor": {}}, {}, {}, {}, artifacts=[art])
    loc = c["artifacts"]["k"]["locations"][0]
    assert loc["status"] == cc.UNKNOWN


# ---------------------------------------------------------------------------
# HF probe
# ---------------------------------------------------------------------------
def test_count_hf_members_is_non_recursive_and_glob_filtered():
    tree = {
        "pre/ep_00000.pt": {"size": 10, "sha256": "a"},
        "pre/ep_00001.pt": {"size": 10, "sha256": "b"},
        "pre/DONE": {"size": 1, "sha256": None},
        "pre/sub/ep_00002.pt": {"size": 10, "sha256": "c"},  # nested: excluded
        "other/ep_00003.pt": {"size": 10, "sha256": "d"},
    }
    n, b = cc.count_hf_members(tree, "pre", "ep_*.pt")
    assert (n, b) == (2, 20)


def test_hf_error_is_unreachable_not_absent():
    art = _art(candidates=[("hf:R", "pre")])
    c = cc.build_census({}, {}, {}, {"R": "403 storage full"}, artifacts=[art])
    loc = c["artifacts"]["k"]["locations"][0]
    assert loc["status"] == cc.UNREACHABLE
    assert c["artifacts"]["k"]["verdict"] == "UNRESOLVED"


def test_hf_single_file_artifact_resolves_by_exact_path():
    art = _art(members=None, candidates=[("hf:R", "dir/model.pt")])
    trees = {"R": {"dir/model.pt": {"size": 42, "sha256": "x"}}}
    c = cc.build_census({}, {}, trees, {}, artifacts=[art])
    loc = c["artifacts"]["k"]["locations"][0]
    assert loc["status"] == cc.PRESENT and loc["count"] == 1
    assert loc["bytes"] == 42


# ---------------------------------------------------------------------------
# git-remote probe — a pushed blob lives on GitHub too, which is a real
# second machine. Missing this made the first census report four artifacts as
# SINGLE_COPY that were in fact mirrored on origin.
# ---------------------------------------------------------------------------
def _git_runner(refs, tree):
    """Fake `git` — refs list, then per-ref `ls-tree` output."""
    class P:
        def __init__(self, out):
            self.stdout = out.encode()

    def run(args):
        if args[1] == "branch":
            return P("\n".join(refs))
        ref = args[4]
        return P("\n".join(tree.get(ref, [])))
    return run


def test_git_remote_finds_single_file_blob():
    r = cc.probe_git_remote(
        "a/b.pt", "*.pt",
        runner=_git_runner(["origin/main"], {"origin/main": ["a/b.pt"]}))
    assert r["status"] == cc.PRESENT and r["count"] == 1
    assert r["ref"] == "origin/main"


def test_git_remote_counts_directory_members_non_recursively():
    tree = {"origin/main": ["res/windows_a.pt", "res/fan_b.pt",
                            "res/notes.json", "res/sub/windows_c.pt"]}
    r = cc.probe_git_remote("res", "windows_*.pt fan_*.pt",
                            runner=_git_runner(["origin/main"], tree))
    assert r["status"] == cc.PRESENT and r["count"] == 2


def test_git_remote_absent_when_no_ref_has_it():
    r = cc.probe_git_remote(
        "a/b.pt", "*.pt",
        runner=_git_runner(["origin/main"], {"origin/main": ["other.pt"]}))
    assert r["status"] == cc.ABSENT


def test_git_remote_with_no_tracking_refs_is_unknown_not_absent():
    """"We never fetched" is not "it was never pushed"."""
    r = cc.probe_git_remote("a/b.pt", "*.pt",
                            runner=_git_runner([], {}))
    assert r["status"] == cc.UNKNOWN
    assert "never fetched" in r["error"]


def test_git_remote_unavailable_is_unknown_not_absent():
    def boom(args):
        raise OSError("git not found")
    r = cc.probe_git_remote("a/b.pt", "*.pt", runner=boom)
    assert r["status"] == cc.UNKNOWN and "git unavailable" in r["error"]


def test_git_remote_skips_the_head_alias_line():
    """`origin/HEAD -> origin/main` is not a ref to search."""
    refs = ["origin/HEAD -> origin/main", "origin/main"]
    seen = []

    def run(args):
        class P:
            stdout = b""
        if args[1] == "branch":
            P.stdout = "\n".join(refs).encode()
            return P
        seen.append(args[4])
        P.stdout = b"a/b.pt"
        return P
    r = cc.probe_git_remote("a/b.pt", "*.pt", runner=run)
    assert r["status"] == cc.PRESENT
    assert seen == ["origin/main"], f"searched bogus refs: {seen}"


# ---------------------------------------------------------------------------
# Local repo probe
# ---------------------------------------------------------------------------
def test_probe_repo_counts_multiple_globs(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    (d / "windows_a.pt").write_bytes(b"x" * 5)
    (d / "fan_b.pt").write_bytes(b"y" * 7)
    (d / "ignore.json").write_text("{}")
    r = cc.probe_repo("results", "windows_*.pt fan_*.pt", tmp_path)
    assert r["status"] == cc.PRESENT and r["count"] == 2 and r["bytes"] == 12


def test_probe_repo_missing_is_absent(tmp_path):
    assert cc.probe_repo("nope", "*.pt", tmp_path)["status"] == cc.ABSENT


# ---------------------------------------------------------------------------
# Exit codes — this is what makes it a guard rather than a report
# ---------------------------------------------------------------------------
def test_exit_codes():
    def mk(verdicts):
        return {"artifacts": {str(i): {"verdict": v}
                              for i, v in enumerate(verdicts)}}
    assert cc.census_exit_code(mk(["OK", "OK"])) == 0
    assert cc.census_exit_code(mk(["OK", "SINGLE_COPY"])) == 1
    assert cc.census_exit_code(mk(["SINGLE_COPY", "ZERO_COPIES"])) == 2
    assert cc.census_exit_code(mk(["OK", "UNRESOLVED"])) == 2


# ---------------------------------------------------------------------------
# The declared contract itself
# ---------------------------------------------------------------------------
def test_parity_corpus_contract_is_the_committed_one():
    """Parity is sacred: nothing may quietly re-select episodes."""
    art = {a.key: a for a in cc.ARTIFACTS}["raw-train-epcache-256px"]
    assert art.members == 2376
    assert "e438721ae894" in art.parity and "f09e44db" in art.parity


def test_every_artifact_declares_at_least_two_candidate_machines():
    """Absence at ONE location is not absence — so a one-location artifact
    cannot even be probed correctly."""
    for art in cc.ARTIFACTS:
        machines = {h for h, _ in art.candidates}
        assert len(machines) >= 2, f"{art.key} probes only {machines}"


def test_format_table_renders_every_artifact():
    c = cc.build_census({}, {"thor": "down"}, {}, {},
                        artifacts=[_art(candidates=[("thor", "/a")])])
    out = cc.format_table(c)
    assert "UNRESOLVED" in out and "k" in out
