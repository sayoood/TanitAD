#!/usr/bin/env python3
"""Pins for `stack/scripts/mktree_commit.py`.

⛔ WHY THESE EXIST. On 2026-09-04 this committer corrupted HEAD for the whole
fleet: it fed `git mktree` its records through a Windows TEXT-mode pipe, which
translates LF into CRLF, so the stray CR landed INSIDE every entry NAME. The
commit's root tree read `.claude<CR>`, `CLAUDE.md<CR>`, `taniteval<CR>` … and
from that moment `git read-tree` failed for every agent with
`error: invalid path '.claude?/…'` — i.e. nobody could commit at all.

The tool DID verify its tree, but only AFTER `update-ref`. That is the class
(RETRACTION_LOG C16): **a correct check run after the irreversible step**. These
tests pin the two repairs — binary I/O, and a name check that runs BEFORE the
ref moves — against a real throwaway git repository, because the failure was in
git's byte-level contract and a mocked one would not have caught it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"


def _load_mktree(repo: Path):
    """Import the script with MKTREE_REPO pointed at a throwaway repo."""
    import importlib.util
    os.environ["MKTREE_REPO"] = str(repo)
    path = _SCRIPTS / "mktree_commit.py"
    if not path.exists():                                    # pragma: no cover
        pytest.skip(f"{path} not present in this checkout")
    spec = importlib.util.spec_from_file_location("mktree_commit_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mktree_commit_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *a: str, stdin: bytes | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *a], input=stdin,
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return r.stdout.decode("utf-8", "replace").strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    if not subprocess.run(["git", "--version"], capture_output=True).returncode == 0:
        pytest.skip("git not available")                     # pragma: no cover
    r = tmp_path / "r"
    (r / "pkg" / "sub").mkdir(parents=True)
    (r / "pkg" / "a.txt").write_text("a\n", encoding="utf-8")
    (r / "pkg" / "sub" / "b.txt").write_text("b\n", encoding="utf-8")
    (r / "top.txt").write_text("top\n", encoding="utf-8")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    return r


def test_mktree_stdin_is_binary_so_names_carry_no_cr(repo: Path):
    """THE REGRESSION. The record written to `mktree` must reach it with a bare
    LF; a CR in the name is the fleet-wide corruption."""
    m = _load_mktree(repo)
    (repo / "pkg" / "a.txt").write_text("a2\n", encoding="utf-8")
    blob = m.git("hash-object", "-w", "--", "pkg/a.txt")
    head = m.git("rev-parse", "HEAD")
    root = m.build_tree(head, {"pkg/a.txt": blob})
    names = list(m.read_tree_entries(root))
    assert names == sorted(names)
    assert not any("\r" in n or "\n" in n for n in names), names
    for n in m.read_tree_entries(f"{root}:pkg"):
        assert "\r" not in n, n
    # and the content really is the new one
    assert m.git("rev-parse", f"{root}:pkg/a.txt") == blob
    # untouched siblings survive
    assert (m.git("rev-parse", f"{root}:top.txt")
            == m.git("rev-parse", f"{head}:top.txt"))
    assert (m.git("rev-parse", f"{root}:pkg/sub/b.txt")
            == m.git("rev-parse", f"{head}:pkg/sub/b.txt"))


def test_assert_names_preserved_refuses_a_lost_entry(repo: Path):
    """The guard must fire BEFORE `commit-tree`, on a name that vanished."""
    m = _load_mktree(repo)
    head = m.git("rev-parse", "HEAD")
    old = m.read_tree_entries(f"{head}^{{tree}}")
    # build a tree that DROPS `top.txt`
    body = "".join(f"{meta}\t{name}\n" for name, meta in sorted(old.items())
                   if name != "top.txt")
    bad = m.git("mktree", stdin=body)
    with pytest.raises(SystemExit) as ex:
        m.assert_names_preserved("<root>", old, bad, set())
    assert "REFUSING to commit" in str(ex.value)
    assert "top.txt" in str(ex.value)


def test_assert_names_preserved_refuses_a_cr_mangled_name(repo: Path):
    """The exact 2026-09-04 corruption, reconstructed: every name gains a CR."""
    m = _load_mktree(repo)
    head = m.git("rev-parse", "HEAD")
    old = m.read_tree_entries(f"{head}^{{tree}}")
    body = "".join(f"{meta}\t{name}\r\n" for name, meta in sorted(old.items()))
    mangled = m.git("mktree", stdin=body)
    assert mangled != m.git("rev-parse", f"{head}^{{tree}}")
    with pytest.raises(SystemExit) as ex:
        m.assert_names_preserved("<root>", old, mangled, set())
    assert "REFUSING to commit" in str(ex.value)


def test_assert_names_preserved_accepts_an_intended_change(repo: Path):
    m = _load_mktree(repo)
    (repo / "pkg" / "a.txt").write_text("a3\n", encoding="utf-8")
    blob = m.git("hash-object", "-w", "--", "pkg/a.txt")
    head = m.git("rev-parse", "HEAD")
    old = m.read_tree_entries(f"{head}:pkg")
    new_entries = dict(old)
    new_entries["a.txt"] = f"100644 blob {blob}"
    body = "".join(f"{meta}\t{name}\n"
                   for name, meta in sorted(new_entries.items()))
    tree = m.git("mktree", stdin=body)
    m.assert_names_preserved("pkg", old, tree, {"a.txt"})     # must not raise


def test_missing_ok_is_narrow(repo: Path):
    """A NEW directory reads as empty; a transient/other failure must still
    raise. Confusing the two would turn a mount blink into a commit that
    silently empties a subtree."""
    m = _load_mktree(repo)
    head = m.git("rev-parse", "HEAD")
    assert m.read_tree_entries(f"{head}:does/not/exist", missing_ok=True) == {}
    with pytest.raises(SystemExit):
        m.read_tree_entries(f"{head}:does/not/exist")


def test_new_file_in_a_new_directory_is_committable(repo: Path):
    """The path that broke the first real use: a directory absent from HEAD."""
    m = _load_mktree(repo)
    (repo / "fresh").mkdir()
    (repo / "fresh" / "n.txt").write_text("n\n", encoding="utf-8")
    blob = m.git("hash-object", "-w", "--", "fresh/n.txt")
    head = m.git("rev-parse", "HEAD")
    root = m.build_tree(head, {"fresh/n.txt": blob})
    assert m.git("rev-parse", f"{root}:fresh/n.txt") == blob
    assert (m.git("rev-parse", f"{root}:top.txt")
            == m.git("rev-parse", f"{head}:top.txt"))
    assert not any("\r" in n for n in m.read_tree_entries(root))
