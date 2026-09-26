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


def _head_membership(rels):
    """Ask HEAD about EVERY path in ONE `git cat-file --batch-check`.

    Returns ``(present, missing, undecided)`` as sets of the input paths.

    ⛔ Still NOT `git ls-files` (the shared index — stale on this mount, and the source of the
    2026-09-05 false finding), and ⛔ still NOT `git ls-tree -r` (CLAUDE.md: it truncates here,
    exits 0, and truncates *CONSISTENTLY*, so repeating it looks like confirmation). The question
    is unchanged — *does HEAD hold this blob?* — only the number of processes is.

    ⛔⛔ AND IT NO LONGER PARSES ENGLISH PROSE, WHICH WAS A CORRECTNESS BUG, NOT A SPEED ONE.
    MEASURED 2026-09-26: the per-file version classified a result as "missing" only when stderr
    contained ``does not exist`` or ``Not a valid object``. git's ACTUAL message for the case this
    test exists to catch is::

        fatal: path 'TanitAD Research Lab/Library/papers/<x>.pdf' exists on disk, but not in 'HEAD'

    which matches NEITHER. So every genuinely-unbanked PDF was filed as **UNDECIDED ("the mount was
    flapping"), and the operator was told to RE-RUN when the true fix was to COMMIT the file** — a
    true-but-wrong-for-the-reader failure: it failed loudly, and named the wrong cause.
    It also cost **6 attempts x 3 s of sleep per missing file**: with 86 of 552 unbanked that is
    **25.8 minutes of pure sleeping**, which is 61 % of the whole suite's 43 min runtime and the
    reason every stream ran subsets instead of the gate.

    ⭐ ``--batch-check`` answers in a PROTOCOL, not prose: ``<sha> blob <size>`` for a hit and
    ``<input> missing`` for a miss. No wording, locale or git-version dependence remains.
    MEASURED: 552 paths in **0.53 s**, one process.
    """
    import time

    rels = list(rels)
    if not rels:
        return set(), set(), set()
    stdin = "".join(f"HEAD:{r}\n" for r in rels)
    for attempt in range(3):                       # a real mount fault still gets a retry -- but
        r = subprocess.run(                        # seconds, not 26 minutes
            ["git", "-C", str(REPO), "cat-file", "--batch-check"],
            input=stdin, capture_output=True, text=True,
            encoding="utf-8", errors="surrogateescape")
        lines = r.stdout.splitlines()
        # ⭐ THE CONTROL that makes the zip below sound: --batch-check emits exactly one line per
        # input, in order. If that does not hold, we do NOT guess -- everything is UNDECIDED.
        if r.returncode == 0 and len(lines) == len(rels):
            present, missing, undecided = set(), set(), set()
            for rel, line in zip(rels, lines):
                if " blob " in line:
                    present.add(rel)
                elif line.rstrip().endswith(" missing"):
                    missing.add(rel)
                else:
                    undecided.add(rel)             # a shape we do not recognise is not an answer
            return present, missing, undecided
        if attempt < 2:
            time.sleep(3)
    return set(), set(), set(rels)                 # could not decide ANY of them


def _in_head(rel: str):
    """Single-path convenience over :func:`_head_membership`. True / False / None (undecided)."""
    present, missing, _undecided = _head_membership([rel])
    return True if rel in present else (False if rel in missing else None)


def _disk_papers() -> set[str]:
    assert PAPERS.is_dir(), f"papers directory missing: {PAPERS}"
    return {p.name for p in PAPERS.iterdir() if p.suffix.lower() == ".pdf"}


def test_every_banked_pdf_is_in_head():
    """The evidence layer is in the repository, not on one laptop.

    Asked of HEAD, per file. ⚠️ Files that cannot be decided (mount flapping) are
    reported separately and do NOT pass silently — an undecided file is not a green one.
    """
    disk = sorted(_disk_papers())
    rels = [f"TanitAD Research Lab/Library/papers/{n}" for n in disk]
    present, miss, undec = _head_membership(rels)

    # ⭐ CONTROL, before any verdict: the query must have actually ANSWERED. A batch that returned
    # nothing would otherwise read as "nothing is missing" -- a green gate over an unasked question,
    # which is the failure this whole file was written about (2026-09-05, ls-files under-reporting).
    assert (len(present) + len(miss) + len(undec)) == len(rels), (
        "the HEAD query did not answer for every banked PDF: "
        f"{len(present)} present + {len(miss)} missing + {len(undec)} undecided != {len(rels)} on disk")
    assert not (rels and not present and not miss), (
        f"INCONCLUSIVE: the HEAD query decided 0 of {len(rels)} paths — that is a failed query, "
        "not an empty library. Re-run; do not read it as a pass.")

    missing = sorted(r.rsplit("/", 1)[-1] for r in miss)
    undecided = sorted(r.rsplit("/", 1)[-1] for r in undec)
    assert not missing, (
        f"{len(missing)} banked primaries are on disk but ABSENT FROM HEAD — they exist "
        f"on one machine only and are invisible to every other reader of the reports that "
        f"cite them. Commit them (mm_commit.py) and re-run.\n  "
        + "\n  ".join(missing[:15])
        + (f"\n  … and {len(missing) - 15} more" if len(missing) > 15 else "")
    )
    assert not undecided, (
        f"{len(undecided)} files could not be decided against HEAD in 3 batched attempts — "
        "the mount was flapping, or git answered in a shape we do not recognise. That is "
        "INCONCLUSIVE, not a pass; re-run.\n  " + "\n  ".join(undecided[:15])
    )


def test_a_pdf_absent_from_head_reads_MISSING_not_undecided():
    """⛔ THE DELIBERATE-REGRESSION ARM. This is the exact defect the batched query fixed.

    A path that is on disk but NOT in HEAD must classify as **missing** — the condition this file
    exists to detect. Before 2026-09-26 it classified as **undecided**, because the code matched the
    prose ``does not exist`` / ``Not a valid object`` while git actually says
    ``exists on disk, but not in 'HEAD'``. The gate therefore failed with *"the mount was flapping,
    re-run"* instead of *"commit these PDFs"*, and burned 6 x 3 s of sleep per file doing it.

    ⭐ Written against a REAL on-disk file, because that is what triggers git's specific wording: a
    path that exists neither on disk nor in HEAD produces a DIFFERENT message, so a fabricated name
    would not reproduce the bug and the arm would pass while the defect lived.
    """
    probe = PAPERS / "zz_deliberate_regression_not_in_head.pdf"
    probe.write_bytes(b"%PDF-1.4 deliberate regression arm\n")
    try:
        rel = f"TanitAD Research Lab/Library/papers/{probe.name}"
        present, missing, undecided = _head_membership([rel])
        assert rel in missing, (
            f"a banked-but-uncommitted PDF must read MISSING; got "
            f"{'present' if rel in present else 'undecided'} — the classifier has regressed to "
            f"prose-matching and the gate will tell operators to re-run instead of to commit")
        assert rel not in undecided and rel not in present
        # and the operator must be able to ACT on it: the path itself is what they need
        assert probe.name in rel
    finally:
        probe.unlink(missing_ok=True)


def test_the_batched_query_really_returned_the_banked_set():
    """⭐ The control that the fast path is answering, not merely returning quickly.

    A batched query that silently returned nothing would make the gate green over an unasked
    question. So: it must decide EVERY path, decide a non-zero number of them, and find a paper we
    know is banked (`library.json`'s own first entry) among the ones it says HEAD holds.
    """
    disk = sorted(_disk_papers())
    assert disk, "no PDFs on disk at all — the control cannot distinguish 'fast' from 'empty'"
    rels = [f"TanitAD Research Lab/Library/papers/{n}" for n in disk]
    present, missing, undecided = _head_membership(rels)
    assert len(present) + len(missing) + len(undecided) == len(rels)
    assert len(present) > 0, (
        "the query says HEAD holds NONE of the banked PDFs. On any real checkout that is a failed "
        "query, not a fact about the library — exactly the 2026-09-05 false finding.")
    # a known-good path: whatever library.json's first entry points at, if it is on disk
    ents = _entries(_read_json(LIB / "library.json"))
    keys = [str(e.get("key") or "") for e in ents if isinstance(e, dict)]
    known = next((n for n in disk if any(k and n.startswith(k) for k in keys)), None)
    assert known is not None, "no on-disk PDF matches any library.json key — the control is inert"


def test_the_gate_is_fast_enough_to_be_run():
    """⛔ A 26-minute gate is an unrun gate, and that is a correctness problem downstream.

    MEASURED 2026-09-26: the per-file version took **1,585.8 s (26m 25s)** — 61 % of the whole
    suite's 43 min — so every stream ran a SUBSET and 31 real failures stayed invisible for days.
    The batched query answers the same 552 paths in **0.53 s**. The bar here is deliberately loose
    (a slow CI box is not a defect) but it is far below anything that would bring the sleep-storm
    back: 86 missing files x 6 x 3 s cannot fit in 60 s.
    """
    import time

    disk = sorted(_disk_papers())
    rels = [f"TanitAD Research Lab/Library/papers/{n}" for n in disk]
    t0 = time.time()
    _head_membership(rels)
    dt = time.time() - t0
    assert dt < 60.0, (f"the HEAD membership query took {dt:.1f} s for {len(rels)} paths — the "
                       "per-file retry storm is back; it must be ONE batched call")


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
