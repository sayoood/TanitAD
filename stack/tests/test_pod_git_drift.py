"""Tests for the box-vs-git drift detector.

The verdict logic is what protects us from the 2026-07-20 failure mode (code
living on exactly one pod disk), so it is tested directly rather than through
ssh. HOST_ONLY is the finding that matters: it means the file exists nowhere
else on earth.

⚠️ Four of these pin the 2026-08-18 repair, and each has a measured defect
behind it: the index used to walk ``.claude/worktrees/`` (**8,079 indexed files
against the repo's 2,132**); it matched by basename anywhere in the repo; it did
no CRLF normalisation (**28.1 % of ``stack/`` sources contain CRLF**, and ~94 %
of the rows it would print as drift were that artifact); and an unreachable host
returned ``[]``, which downstream read as *"clean"*.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pod_git_drift import (  # noqa: E402
    ARTIFACT, ARTIFACT_SUFFIXES, CRLF_ONLY, DRIFTED, HOST_ONLY, IN_GIT,
    MAX_BYTES, NAME_DRIFT, NAME_ONLY, POD_ONLY, SOURCE, SOURCE_SUFFIXES,
    _match_by_path, classify, inclusion_rule, is_sensitive, kind_of,
    repo_index, sha256_bytes, sha256_lf,
)


def _index(files: dict[str, bytes]) -> dict:
    """Build an index directly, mirroring repo_index's output shape."""
    by_hash, by_lf, by_name, by_path, name_paths = {}, {}, {}, {}, {}
    for path, data in files.items():
        raw, lf = sha256_bytes(data), sha256_lf(data)
        name = path.rsplit("/", 1)[-1]
        by_hash.setdefault(raw, []).append(path)
        by_lf.setdefault(lf, []).append(path)
        by_name.setdefault(name, set()).add(raw)
        name_paths.setdefault(name, set()).add(path)
        by_path[path] = {"raw": raw, "lf": lf}
    return {"by_hash": by_hash, "by_lf": by_lf, "by_name": by_name,
            "name_paths": name_paths, "by_path": by_path}


def test_identical_content_is_in_git():
    idx = _index({"stack/scripts/a.py": b"print(1)\n"})
    got = classify([(sha256_bytes(b"print(1)\n"), "/root/a.py")], idx)
    assert got[0]["verdict"] == IN_GIT


def test_same_path_different_content_is_drifted():
    """The box is running something we cannot rebuild from HEAD."""
    idx = _index({"stack/scripts/refb_train.py": b"v1\n"})
    got = classify(
        [(sha256_bytes(b"v2-modified\n"),
          "/root/TanitAD/stack/scripts/refb_train.py")], idx)
    assert got[0]["verdict"] == DRIFTED
    assert "stack/scripts/refb_train.py" in got[0]["note"]


def test_unknown_file_is_host_only():
    """This is the REF-B v2 / TanitEval case — one disk, no copy anywhere."""
    idx = _index({"stack/scripts/a.py": b"x\n"})
    got = classify([(sha256_bytes(b"secret sauce\n"), "/root/refb_v4.py")], idx)
    assert got[0]["verdict"] == HOST_ONLY == POD_ONLY
    assert "no file of this name" in got[0]["note"]


# --------------------------------------------------------------------------
# ⚠️ the 2026-08-18 repair — four defects, each of which read as an answer
# --------------------------------------------------------------------------


def test_crlf_difference_is_its_own_verdict_and_is_never_drift():
    """⛔ The largest single effect of the repair.

    The repo tree is CRLF, every box is LF. MEASURED: 28.1 % of ``stack/`` and
    19.3 % of ``taniteval/`` sources contain CRLF, and on a well-synced box
    ~94 % of the rows this printed as drift were that artifact (C105, 47 of 50).
    """
    crlf = b"import os\r\ndef f():\r\n    return 1\r\n"
    lf = crlf.replace(b"\r\n", b"\n")
    idx = _index({"stack/scripts/roll.py": crlf})
    got = classify([(sha256_bytes(lf),
                     "/home/nvidia/TanitAD/stack/scripts/roll.py")], idx)
    assert got[0]["verdict"] == CRLF_ONLY
    assert "NOT drift" in got[0]["note"]
    # and the byte delta really is one \r per line — the arithmetic C105 used
    assert len(crlf) - len(lf) == 3


def test_a_genuine_edit_survives_lf_normalisation_and_is_still_drift():
    """The normaliser must not become a way to hide real drift."""
    idx = _index({"stack/scripts/roll.py": b"A = 1\r\nB = 2\r\n"})
    got = classify([(sha256_bytes(b"A = 1\nB = 3\n"),
                     "/home/nvidia/TanitAD/stack/scripts/roll.py")], idx)
    assert got[0]["verdict"] == DRIFTED


def test_a_same_basename_in_an_unrelated_tree_is_not_called_drift():
    """⛔ The basename lottery.

    ``/root/experiments/scratch/utils.py`` has nothing to do with
    ``stack/tanitad/utils.py``. Calling that DRIFTED manufactures a finding —
    and ``__init__.py`` alone has dozens of repo copies.
    """
    idx = _index({"stack/tanitad/utils.py": b"real\n"})
    got = classify([(sha256_bytes(b"unrelated\n"),
                     "/root/experiments/scratch/utils.py")], idx)
    assert got[0]["verdict"] == NAME_ONLY
    assert "not drift" in got[0]["note"]


def test_longest_path_suffix_wins_over_a_shallow_coincidence():
    """A full-path agreement must outrank a two-segment collision."""
    by_path = {"stack/scripts/v6_chain.py": {}, "tools/v6_chain.py": {}}
    assert (_match_by_path("/home/nvidia/TanitAD/stack/scripts/v6_chain.py",
                           by_path) == "stack/scripts/v6_chain.py")
    # nothing plausible-but-wrong when there is no match at all
    assert _match_by_path("/root/other/zzz.py", by_path) is None


def test_an_empty_host_list_is_an_error_not_a_clean_result():
    """⛔ ``--hosts`` with no values scanned nothing and reported ``TOTAL: 0``.

    Same shape as the four dead default pods: a reassuring answer to a question
    that was never asked. Absence of a finding is not a finding of absence.
    """
    from pod_git_drift import main
    assert main(["--hosts"]) == 2


def test_repo_index_excludes_worktree_copies(tmp_path):
    """⛔ 14 stale full repo copies inflated the index 3.8× (8,079 vs 2,132).

    Every IN_GIT / DRIFTED judgement was being made against that pool.
    """
    (tmp_path / "stack").mkdir()
    (tmp_path / "stack/live.py").write_bytes(b"current\n")
    wt = tmp_path / ".claude" / "worktrees" / "agent-x" / "stack"
    wt.mkdir(parents=True)
    (wt / "live.py").write_bytes(b"ancient\n")
    (wt / "ghost.py").write_bytes(b"only in the worktree\n")

    idx = repo_index(tmp_path)

    assert set(idx["by_path"]) == {"stack/live.py"}
    assert "ghost.py" not in idx["by_name"], (
        "a worktree-only file must not mask a genuine HOST_ONLY finding")
    assert sha256_bytes(b"ancient\n") not in idx["by_hash"]


def test_relocated_file_still_counts_as_in_git():
    """Content match wins over path — a rescued file lands at a new repo path."""
    idx = _index({"stack/experiments/refb-v2/refb_v4.py": b"arch\n"})
    got = classify([(sha256_bytes(b"arch\n"), "/root/refb_v4.py")], idx)
    assert got[0]["verdict"] == IN_GIT


def test_mixed_batch_classifies_each_independently():
    idx = _index({"s/a.py": b"A\n", "s/b.sh": b"B\n"})
    got = classify(
        [
            (sha256_bytes(b"A\n"), "/root/s/a.py"),
            (sha256_bytes(b"B-changed\n"), "/root/s/b.sh"),
            (sha256_bytes(b"C\n"), "/root/s/c.py"),
        ],
        idx,
    )
    assert [g["verdict"] for g in got] == [IN_GIT, DRIFTED, HOST_ONLY]


def test_repo_index_picks_up_sources_and_skips_noise(tmp_path):
    """⚠️ UPDATED for C110: ``.md`` used to be asserted ABSENT here.

    That assertion pinned the very filter that made the tool miss 46 of 102
    stranded files. Documents, result JSONs and run logs are now in scope; the
    caches must still not be.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "real.py").write_bytes(b"code\n")
    (tmp_path / "pkg" / "tool.sh").write_bytes(b"#!/bin/sh\n")
    (tmp_path / "pkg" / "notes.md").write_bytes(b"# doc\n")
    (tmp_path / "pkg" / "result.json").write_bytes(b'{"ade": 0.45}\n')
    (tmp_path / "pkg" / "run.log").write_bytes(b"step 1\n")
    (tmp_path / "pkg" / "model.pt").write_bytes(b"\x00binary\n")
    cache = tmp_path / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "real.py").write_bytes(b"compiled\n")

    idx = repo_index(tmp_path)

    assert "real.py" in idx["by_name"]
    assert "tool.sh" in idx["by_name"]
    # ⛔ the C110 fix: these three were invisible BY CONSTRUCTION
    assert "notes.md" in idx["by_name"]
    assert "result.json" in idx["by_name"]
    assert "run.log" in idx["by_name"]
    # and the over-correction guard: binaries and caches stay out
    assert "model.pt" not in idx["by_name"]
    assert idx["by_name"]["real.py"] == {sha256_bytes(b"code\n")}


# --------------------------------------------------------------------------
# ⛔ C110 — the filter was the finding, and NAME_ONLY was the escape hatch
# --------------------------------------------------------------------------


def test_the_widened_filter_sees_what_the_old_one_missed_by_construction():
    """⛔ ``SUFFIXES`` was ``(".py", ".sh")``: 46 of 102 stranded files (45 %)
    were invisible. Each of these is one of the classes C110 names."""
    old = (".py", ".sh")
    for name in ("REPORT.md", "gate.json", "train.log", "cfg.yaml",
                 "model.py.bak", "ids.txt"):
        suffix = "." + name.rsplit(".", 1)[-1]
        assert suffix in SOURCE_SUFFIXES + ARTIFACT_SUFFIXES, name
        if suffix not in old:
            idx = _index({"stack/scripts/a.py": b"x\n"})
            got = classify([(sha256_bytes(b"stranded\n"), f"/root/{name}")],
                           idx)
            assert got[0]["verdict"] == HOST_ONLY, name


def test_source_and_artifact_are_counted_as_different_things():
    """⚠️ A run log is NOT noise — C110 kept all 17 as raw measurement
    transcripts — but it needs a different judgement from a .py, and merging
    them is how a widened filter becomes 293 rows of undifferentiated noise."""
    assert kind_of("/root/refb_v4.py") == SOURCE
    assert kind_of("/root/NOTES.md") == SOURCE
    assert kind_of("/root/gate.json") == ARTIFACT
    assert kind_of("/root/train.log") == ARTIFACT

    idx = _index({"stack/a.py": b"x\n"})
    got = classify([(sha256_bytes(b"1\n"), "/root/arch.py"),
                    (sha256_bytes(b"2\n"), "/root/gate.json")], idx)
    assert [g["kind"] for g in got] == [SOURCE, ARTIFACT]


def test_an_unambiguous_program_authored_basename_is_name_drift_not_a_shrug():
    """⛔ THE thor_profile.py CASE, which escaped on the word "weak".

    The banked script could not have produced its own banked JSON — a different
    model had been profiled. The tool saw the file and downgraded it.
    """
    idx = _index({"TanitAD Research Hub/Architecture & Inference/Implementation"
                  "/incoming/2026-08-02-thor-deployment-profile/thor_profile.py":
                  b"def main():\n    out = {}\n"})
    got = classify([(sha256_bytes(b"def main():\n    out['frame'] = f\n"),
                     "/home/nvidia/probe/thor_profile.py")], idx)
    assert got[0]["verdict"] == NAME_DRIFT
    # ⛔ the exact downgrade phrasing it escaped on must be gone
    assert "weak evidence, not drift" not in got[0]["note"]
    assert "not weak evidence" in got[0]["note"]
    assert "thor_profile.py" in got[0]["note"]


def test_an_artifact_basename_collision_is_never_promoted():
    """⛔ THE ANTI-OVER-CORRECTION CLAUSE, and it is MEASURED, not assumed.

    On Thor the widened filter promoted **478** basename hits, of which **304
    (63.6 %) were artifacts** — result JSONs and logs whose names merely
    coincide with a repo file. *A tool that prints 478 rows of which 304 are
    noise is not more useful than one that printed 45.* C110's real case was a
    ``.py``; an artifact's basename colliding says nothing about provenance.
    """
    idx = _index({"stack/experiments/run-a/strategic_gt.json": b'{"ade": 1}\n'})
    got = classify([(sha256_bytes(b'{"ade": 2}\n'),
                     "/home/nvidia/nurec-gsplat/results/strategic_gt.json")],
                   idx)
    assert got[0]["verdict"] == NAME_ONLY
    assert got[0]["kind"] == ARTIFACT

    # the SAME collision on a source file IS promoted — the clause must not
    # become a way to silence the finding it exists to surface
    idx = _index({"stack/scripts/strategic_gt.py": b"a\n"})
    got = classify([(sha256_bytes(b"b\n"),
                     "/home/nvidia/probe/strategic_gt.py")], idx)
    assert got[0]["verdict"] == NAME_DRIFT


def test_an_ambiguous_basename_is_still_only_weak_evidence():
    """The over-correction guard: promoting every basename hit would flood the
    report. ``__init__.py`` alone has dozens of repo copies."""
    many = _index({"stack/tanitad/helper.py": b"a\n",
                   "taniteval/helper.py": b"b\n"})
    got = classify([(sha256_bytes(b"c\n"), "/root/scratch/helper.py")], many)
    assert got[0]["verdict"] == NAME_ONLY
    assert "not drift" in got[0]["note"]

    # a name so common it carries no information even when unique here
    ubiq = _index({"stack/tanitad/__init__.py": b"a\n"})
    got = classify([(sha256_bytes(b"z\n"), "/root/other/__init__.py")], ubiq)
    assert got[0]["verdict"] == NAME_ONLY


def test_a_unique_basename_outside_our_trees_is_not_promoted():
    """⚠️ 5 ``glam``/``numpy`` build scripts were once reported as stranded
    deliverables. A unique name in a vendored tree is still not our drift."""
    idx = _index({"third_party/glam/build_helper.py": b"a\n"})
    got = classify([(sha256_bytes(b"b\n"), "/root/x/build_helper.py")], idx)
    assert got[0]["verdict"] == NAME_ONLY


def test_crlf_normalisation_still_applies_to_the_newly_included_types():
    """⚠️ The 94 %-false-positive problem SCALES with the suffix set — docs and
    JSON are line-ending-bearing too, so widening without this would have made
    the noise worse, not the census better."""
    for path, crlf in (("Project Steering/NOTES.md", b"# t\r\n- a\r\n"),
                       ("stack/gate.json", b'{\r\n "a": 1\r\n}\r\n')):
        idx = _index({path: crlf})
        got = classify([(sha256_bytes(crlf.replace(b"\r\n", b"\n")),
                         "/home/nvidia/" + path)], idx)
        assert got[0]["verdict"] == CRLF_ONLY, path
        assert "NOT drift" in got[0]["note"]


def test_the_inclusion_rule_is_reported_as_data():
    """⛔ C110's whole lesson: a count is a claim about the FILTER until the
    filter is stated next to it. So the filter is emitted, not just printed."""
    rule = inclusion_rule(["/home/nvidia"], 6)
    assert ".md" in rule["source_suffixes"]
    assert ".json" in rule["artifact_suffixes"]
    assert rule["max_bytes"] == MAX_BYTES
    assert "NOT A CENSUS" in rule["_warning"]
    assert "46 of 102" in rule["_warning"]


def test_the_size_cap_is_symmetric(tmp_path):
    """⚠️ If only the remote were capped, a large repo file would go unindexed
    while its box copy was scanned — a HOST_ONLY row invented out of the
    tool's own asymmetry."""
    (tmp_path / "big.json").write_bytes(b"x" * (MAX_BYTES + 1))
    (tmp_path / "small.json").write_bytes(b"x" * 16)
    idx = repo_index(tmp_path)
    assert "small.json" in idx["by_name"]
    assert "big.json" not in idx["by_name"]
    assert idx["skipped_large"] == 1


def test_a_secret_looking_path_is_flagged_before_anyone_pulls_it():
    """⚠️ C111: the 117-file Thor rescue banked a run log whose line 11 held a
    LIVE Hugging Face token, and GitHub's push protection — not ours — caught
    it. A widened suffix set sweeps in exactly that exhaust."""
    assert is_sensitive("/root/Keys.txt")
    assert is_sensitive("/home/nvidia/.ssh/id_ed25519")
    assert not is_sensitive("/root/train_v6_staged.py")

    idx = _index({"stack/a.py": b"x\n"})
    got = classify([(sha256_bytes(b"hf_live\n"), "/root/Keys.txt")], idx)
    assert got[0]["sensitive"] is True
    assert "READ" in got[0]["note"].upper()
