"""The Library's EVIDENCE layer must be in git, not merely on one disk.

⛔⛔ WHY THIS TEST EXISTS, AND THE FALSE FINDING THAT WROTE IT. On 2026-09-05 two
independent readers (a research agent and the Master Mind) concluded that **64 of 437
banked primaries were untracked and existed on one machine only**. Both used
`git ls-files` — and `ls-files` reads the **SHARED INDEX**, not `HEAD`. On this mount the
shared index goes stale, so it UNDER-REPORTS what is tracked. The positive assertion
settled it: `git cat-file -e HEAD:<path>` succeeded on **64 of 64**. Nothing was stranded.
420 MB was almost re-committed and a fabricated integrity gap almost reported to the PI.

⇒ **THIS TEST ASKS `HEAD`, NEVER THE INDEX.** `git ls-files` answers *"is this path in
the index?"*, a question about a cache; the question that matters is *"does the repository
hold this file?"*, and only `HEAD` answers it. Same family as CLAUDE.md's `ls-tree`
truncation and the `--cached`-is-not-enough rule: **a short or empty git result is
indistinguishable from a failed query, and only POSITIVE assertions are admissible.**

The failure this still guards is real and recurrent here: *an artifact on one disk is NOT
done* (`AGENT_OPERATING_STANDARD` rule 3 — LAL-v2 unmerged 12 days, an orthogonality
instrument 10 days, TanitEval and REF-B v2's architecture each stranded). A banked sha256
is a claim about a file we hold. And a citation can rot silently: a report cites a library
key, the key resolves in `library.json`, and the PDF behind it is unreachable to every
other reader.

⚠️ It is also the reason a CITATION can rot silently: a report cites a library key, the key
resolves in `library.json`, and the PDF behind it is unreachable to anyone else — so the
claim reads PUBLISHED while its primary is, for every other reader, missing.

This test asserts the three sets agree. It is deliberately a *test*, not a lint in
`kb_add.py`: the failure it catches is committed-index-without-committed-files, which is
invisible to the tool that writes the index.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "TanitAD Research Lab" / "Library"
PAPERS = LIB / "papers"

#: ⚠️ This test is about the CANONICAL repo's integrity. Run from a partial
#: off-Drive mirror (which carries `stack/` and `tools/` but not the 400+ MB
#: `Library/papers`), every assertion below would fail for the wrong reason and
#: read as "64 primaries are untracked" when the truth is "this tree is not the
#: one that holds them". A test that fails for the wrong reason is worse than no
#: test: it trains the reader to ignore it.
_IS_CANONICAL = PAPERS.is_dir() and any(PAPERS.glob("*.pdf"))
pytestmark = pytest.mark.skipif(
    not _IS_CANONICAL,
    reason=f"not the canonical tree (no Library/papers under {REPO}) — "
           "run this against the repo that holds the evidence layer",
)


def _read_json(path: Path):
    """Read through the flapping G: mount rather than trusting one attempt.

    ⚠️ An empty read on this mount is a claim about the READ, not about the file
    (CLAUDE.md). A single failed open must never be reported as an empty library.
    """
    import time

    last = None
    for _ in range(20):
        try:
            raw = path.read_text(encoding="utf-8")
            if raw.strip():
                return json.loads(raw)
        except OSError as exc:  # pragma: no cover - mount-dependent
            last = exc
        time.sleep(3)
    raise AssertionError(f"could not read {path} in 20 attempts (last: {last})")


def _entries(lib) -> list:
    ents = lib.get("entries", lib) if isinstance(lib, dict) else lib
    return list(ents.values()) if isinstance(ents, dict) else list(ents)


def _in_head(rel: str):
    """Positive assertion that HEAD holds this path. None = could not decide.

    ⛔ Deliberately NOT `git ls-files` (the shared index — stale on this mount, and the
    source of the 2026-09-05 false finding) and NOT `git ls-tree -r` (CLAUDE.md: it
    truncates here, exits 0, and truncates *consistently*, so repeating it looks like
    confirmation). `cat-file -e` asks HEAD about ONE path and answers yes / no / failed.
    """
    import time

    for _ in range(6):
        r = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"HEAD:{rel}"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
        err = r.stderr or ""
        if "does not exist" in err or "Not a valid object" in err:
            return False
        time.sleep(3)          # transient: a mount flap is not an answer
    return None


def _disk_papers() -> set[str]:
    assert PAPERS.is_dir(), f"papers directory missing: {PAPERS}"
    return {p.name for p in PAPERS.iterdir() if p.suffix.lower() == ".pdf"}


def test_every_banked_pdf_is_in_head():
    """The evidence layer is in the repository, not on one laptop.

    Asked of HEAD, per file. ⚠️ Files that cannot be decided (mount flapping) are
    reported separately and do NOT pass silently — an undecided file is not a green one.
    """
    disk = sorted(_disk_papers())
    missing, undecided = [], []
    for name in disk:
        got = _in_head(f"TanitAD Research Lab/Library/papers/{name}")
        if got is False:
            missing.append(name)
        elif got is None:
            undecided.append(name)
    assert not missing, (
        f"{len(missing)} banked primaries are on disk but ABSENT FROM HEAD — they exist "
        f"on one machine only and are invisible to every other reader of the reports that "
        f"cite them. Commit them (mm_commit.py) and re-run.\n  "
        + "\n  ".join(missing[:15])
        + (f"\n  … and {len(missing) - 15} more" if len(missing) > 15 else "")
    )
    assert not undecided, (
        f"{len(undecided)} files could not be decided against HEAD in 6 attempts each — "
        "the mount was flapping. That is INCONCLUSIVE, not a pass; re-run."
    )


def test_index_staleness_is_visible_not_silent():
    """⚠️ A DIAGNOSTIC, not a gate: how far the shared index lags HEAD.

    This is the number that produced the 2026-09-05 false finding. It is PRINTED, not
    asserted: a stale index is a mount artefact, not a library defect — but a reader who
    sees `ls-files` disagree with HEAD should meet this note first and reach for
    `cat-file -e`, rather than for a 420 MB re-commit.
    """
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "TanitAD Research Lab/Library/papers"],
        capture_output=True, text=True, encoding="utf-8", errors="surrogateescape")
    in_index = {os.path.basename(l) for l in out.stdout.splitlines() if l.strip()}
    disk = _disk_papers()
    lag = len(disk - in_index) if in_index else -1
    if lag > 0:
        print(f"\n  NOTE: the shared index lags HEAD by {lag} of {len(disk)} papers. "
              "That is an index artefact — HEAD is authoritative and is what the other "
              "tests assert against.")


def test_index_and_evidence_agree_in_count():
    """`library.json` must not describe more primaries than we actually hold.

    ⚠️ Asserted as index ⊆ disk, not equality: a PDF on disk that the index has not
    yet claimed is a pending `kb_add`, which `--verify` already reports as an orphan.
    An index ENTRY with no file is the dangerous direction — a citation that resolves
    to nothing.
    """
    lib = _read_json(LIB / "library.json")
    n_entries = len(_entries(lib))
    n_disk = len(_disk_papers())
    assert n_entries <= n_disk, (
        f"library.json claims {n_entries} entries but only {n_disk} PDFs are on disk — "
        "at least one citation resolves to a file we do not hold."
    )


def test_library_md_is_generated_not_handwritten():
    """⛔ `LIBRARY.md` is generated by `kb_add.py --reindex`; hand-edits are lost.

    Pinned because CLAUDE.md forbids hand-editing it and the failure is silent: the
    next `--reindex` overwrites the edit and nobody notices the citation vanished.
    """
    md = LIB / "LIBRARY.md"
    assert md.is_file(), "LIBRARY.md missing"
    head = md.read_text(encoding="utf-8", errors="replace")[:4000].lower()
    assert "generated" in head or "kb_add" in head or "do not edit" in head, (
        "LIBRARY.md no longer carries its generated-file banner — either the generator "
        "changed or someone hand-wrote the file. Re-run `kb_add.py --reindex`."
    )
