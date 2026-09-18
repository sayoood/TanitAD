"""`tools/backlog_drift.py`: does it FIRE, and does it refuse when blind?

⛔ WHY THE MUTATION ARMS ARE THE IMPORTANT ONES. Per the validation standard, a
PASS on the fixed arm means nothing unless the gate FAILS the regression arm. A
drift checker that always reports zero is indistinguishable from a tidy backlog --
and "reports zero because it read nothing" is precisely the family this programme
keeps paying for (`pod_currency_audit.py` scanned an MSYS-mangled root, found
nothing, and reported NO DRIFT on a pre-launch gate).

Every mutation below is proven RED by `raw/run_mutation_proof.py`, which patches
the source, re-runs this suite, records the returncode, and restores by md5.

⚠️ THE FALSE-POSITIVE FIXTURES ARE VERBATIM LIVE ROWS, not invented strings. The
first version of this tool matched done-words as SUBSTRINGS and fired on five real
rows -- `closed-loop`, `done-marker`, `roll_closed`, `D-STEER-INTERFACE-RESOLVED`,
and a DEPENDENCY cell reading "fix landed". Five false alarms in thirteen hits is
the gate nobody reads, so those five strings are pinned here as negative controls.
A test suite whose negative controls are hypothetical cannot catch the error that
actually happened.
"""
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "backlog_drift.py"
REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(TOOL.parent))
import backlog_drift as m  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
#: ⚠️ MIN_BYTES is deliberately NOT a CLI flag -- a short-read floor that can be
#: tuned away is not a floor. Fixtures pad instead.
PAD = "\n<!-- padding so the fixture clears MIN_BYTES; the byte floor exists\n" \
      "     because a truncated read on a flaky mount is indistinguishable\n" \
      "     from a short file unless something asserts a size. -->\n" * 20


def write_backlog(tmp_path: Path, body: str, name="BACKLOG.md") -> Path:
    p = tmp_path / name
    p.write_text(body + PAD, encoding="utf-8")
    return p


def run(tmp_path: Path, *args, name="BACKLOG.md"):
    """End-to-end through the real subprocess boundary.

    ⚠️ Not `main(argv)` in-process: the things most likely to be wrong are the
    exit code and the printed encoding, and neither survives an in-process call
    faithfully (`sys.stdout` in pytest is not the cp1252 console).
    """
    return subprocess.run(
        [sys.executable, str(TOOL), "--repo", str(tmp_path), "--backlog", name,
         *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace")


# a minimum viable healthy list: >= 1 closed row, >= 1 open row, nothing stale
CLEAN = """# BACKLOG

Strike items through when done, with the commit.

## A. 0-GPU

| # | item | why |
|---|---|---|
| ~~A1~~ | ~~Build the emitters~~ **DONE 2026-08-01** (`abc1234`) | closed properly |
| A2 | Re-verify the factorised path vocabulary | a real open item |
| A3 | Bank the remaining pod-only siblings | another open item |
"""


# --------------------------------------------------------------------------
# 1. the live file -- the tool must work on its actual subject
# --------------------------------------------------------------------------
def test_reads_the_LIVE_backlog_and_clears_the_row_floor():
    """⭐ The anchor. A parser that works only on fixtures is a parser for
    fixtures. `Project Steering/BACKLOG.md` read 117 distinct ids over 133
    occurrences on 2026-09-18."""
    r = subprocess.run([sys.executable, str(TOOL), "--repo", str(REPO), "--json"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode in (0, 1), r.stdout + r.stderr
    import json
    d = json.loads(r.stdout)
    assert d["rows"] >= m.MIN_ROWS, "the live list must clear its own floor"
    assert d["counts"].get("OPEN", 0) > 0, "a pull-list with nothing open?"
    assert d["counts"].get("CLOSED", 0) > 0, "the live list has struck ids"


def test_the_live_backlog_is_CURRENTLY_DRIFTED_and_the_gate_says_so():
    """⭐ The finding this tool was written for, pinned. If someone sweeps the
    list, this test goes red -- and that is the correct time to update it, with
    the sweep's commit."""
    r = subprocess.run([sys.executable, str(TOOL), "--repo", str(REPO)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 1, (
        "expected drift on the live list; if it was genuinely swept, update this "
        "test IN THE SAME COMMIT as the sweep\n" + r.stdout)
    for rid in ("A11", "A13", "B4", "C3", "R1", "R2", "R4"):
        assert rid in r.stdout, f"{rid} declares DONE with an unstruck id"
    assert "STALE-BLOCKER  C2" in r.stdout


# --------------------------------------------------------------------------
# 2. the anchored matcher -- negative controls FIRST
# --------------------------------------------------------------------------
@pytest.mark.parametrize("live_text", [
    "**Owner = the AlpaSim closed-loop stream** -- a 9.6 MB bank",          # A14
    "plus the R7/R8 discipline in the header (done-marker = off-switch)",   # F2
    "refcv3 is one-shot with no rollout, so `roll_closed` cannot be ported",  # R30
    "decide when `--action-units steer` becomes the **default** "
    "(D-STEER-INTERFACE-RESOLVED)",                                        # R21
    "`taniteval/tools/REFAV1_ARM.md` | C-REFAV1-PLAN-NOGOAL fix landed |",  # R3
])
def test_a_done_WORD_is_not_a_done_CLAIM(live_text):
    """⛔ VERBATIM LIVE ROWS that a substring matcher called finished. Each is a
    compound word, an identifier, a decision id, or a DEPENDENCY that landed --
    which is the row's reason to stay open, not to close."""
    assert not m.declares_done(live_text), live_text


@pytest.mark.parametrize("live_text", [
    "~~**Rescue the 45 stranded files on Thor**~~ * **DONE 2026-08-18, 0 GPU**",
    "~~Matched-capacity camera head~~ **DONE 2026-08-03, 0 pod GPU-h**",
    "**v5 gate verdict** | ~~v5 reaching step 2000~~ * **CLOSED 2026-08-16**",
    "* **DONE 2026-09-02 (`3a6a48c`)** -- Review the UNSTAGED worktree diff",
])
def test_the_real_closure_shapes_ARE_detected(live_text):
    """The positive control. Without it the negative controls above could be
    satisfied by a matcher that simply never fires."""
    assert m.declares_done(live_text), live_text


@pytest.mark.parametrize("live_text", [
    "PARTLY STRUCK**: the seed reproduces its goal 64/64",                  # R38
    "REFRAMED (not struck)**: the chord cost alone flips 1 of 25 windows",  # R29
])
def test_a_NEGATED_closure_is_not_a_closure(live_text):
    """⛔ Both are live rows. A close-detector that reads these as finished
    reports two OPEN items as done -- this tool's own error, sign flipped.

    ⭐ These pass because of FIRST-WORD ANCHORING, not because of a negation
    list. A negation list WAS written for exactly these two rows and the mutation
    proof showed deleting it left the suite green and moved 0 of 117 live
    verdicts, so it was removed (see `_is_done_phrase`). Mutation `M2` is what
    holds this test up: under substring matching both go red."""
    assert not m.declares_done(live_text, bold_src="**" + live_text), live_text


def test_the_bullet_bold_PREFIX_is_restored_or_spans_mis_pair():
    """⛔ A BULLET's opening `**` is eaten by the id match. Unrestored, the row's
    CLOSING `**` pairs with the NEXT opener and the phantom span can begin with a
    closure verb -- a FALSE CLOSURE, the direction that silently removes work
    from the list. The payload is live row F2's own phrase."""
    s = "check X** DONE-marker discipline **applies**"
    assert m.declares_done(s), "un-prefixed: the phantom span reads as a closure"
    assert not m.declares_done(s, bold_src="**" + s), "prefixed: pairs correctly"


def test_the_bold_PREFIX_is_applied_AT_THE_CALL_SITE_in_parse(tmp_path):
    """⛔ EARNED BY A SURVIVING MUTATION. The unit test above pins the FUNCTION
    and passed happily while `parse()` was mutated to stop passing `bold_src` at
    all -- a test of the right behaviour at the wrong boundary. This one drives
    the same payload through the parser, where the defect actually lives."""
    write_backlog(tmp_path, CLEAN + """
## bullets

- **R99 - check X** DONE-marker discipline **applies** to this row.
""")
    import json
    v = {x["id"]: x["verdict"]
         for x in json.loads(run(tmp_path, "--min-rows", "1", "--json").stdout)["verdicts"]}
    assert v["R99"] == "OPEN", "mis-paired spans invented a closure for R99"


def test_DIAGNOSED_and_IN_PROGRESS_are_not_closures():
    """R5 reads `DIAGNOSED ... fix (save-before-eval) in progress`. Diagnosis is
    a waypoint, not a deliverable -- Rule Zero in one row."""
    assert not m.declares_done(
        "* **DIAGNOSED 2026-09-02 (cgroup OOM kill) - fix in progress** -- "
        "Diagnose refcv3's silent deaths at eval boundaries")


# --------------------------------------------------------------------------
# 3. aggregation by id -- the convention that stops the gate crying wolf
# --------------------------------------------------------------------------
def test_a_LATER_line_closes_an_earlier_bullet_and_the_gate_stays_quiet(tmp_path):
    """⛔ THE R8 SHAPE, and it is the majority of the file. The bullet sections
    close an item by ADDING `- **R8 -- STRUCK ...**` rather than striking the
    original. A per-occurrence scan calls every such original stale and fires on
    ~14 rows that are correctly closed -- which is how a gate gets muted."""
    write_backlog(tmp_path, CLEAN + """
## Added later

- **R8 - v7f LAUNCH BLOCKER: eval-clip exclusion in the v7 trainer's B1 path.

## Added still later

- **R8 - STRUCK 2026-09-03** (D-V7-EVAL-EXCLUSION): exclusion is ON by default.
""")
    r = run(tmp_path, "--min-rows", "1", "--json")
    import json
    v = {x["id"]: x["verdict"] for x in json.loads(r.stdout)["verdicts"]}
    assert v["R8"] == "CLOSED-ELSEWHERE"
    assert r.returncode == 0, "CLOSED-ELSEWHERE must never fail the gate"


def test_a_MULTI_ID_bullet_closes_EVERY_id_it_names(tmp_path):
    """`- **R12, R22 -- STRUCK**` is live. A parser taking only the first id
    silently leaves R22 reading as open work."""
    write_backlog(tmp_path, CLEAN + """
## first

- **R12 - read the banked GS-9 transition-probe JSON.
- **R22 - run actdiv in BOTH unit conventions.

## later

- **R12, R22 - STRUCK**: the banked GS-9 raw is read and the argument tested.
""")
    import json
    v = {x["id"]: x["verdict"]
         for x in json.loads(run(tmp_path, "--min-rows", "1", "--json").stdout)["verdicts"]}
    assert v["R12"] == "CLOSED-ELSEWHERE" and v["R22"] == "CLOSED-ELSEWHERE"


def test_a_row_that_declares_itself_done_with_an_UNSTRUCK_id_FIRES(tmp_path):
    """⭐ THE REGRESSION CASE -- the situation that actually occurred, four
    consecutive times in one night."""
    write_backlog(tmp_path, CLEAN + """
| A9 | ~~Rescue the stranded files~~ **DONE 2026-08-18** (`aabbcc1`) | stale |
""")
    r = run(tmp_path, "--min-rows", "1")
    assert r.returncode == 1
    assert "DONE-BUT-OPEN" in r.stdout and "A9" in r.stdout


# --------------------------------------------------------------------------
# 4. the stale-blocker class
# --------------------------------------------------------------------------
def test_a_STRUCK_BLOCKER_on_an_open_row_is_its_own_class(tmp_path):
    """The operating standard's named class: a blocker note not revisited when
    the thing it blocks on lands, so it keeps reading as a live gap."""
    write_backlog(tmp_path, CLEAN + """
| C9 | Score both arms on the leak-free episodes | ~~v2corpus reaching 30 k~~ |
""")
    r = run(tmp_path, "--min-rows", "1")
    assert r.returncode == 1 and "STALE-BLOCKER" in r.stdout and "C9" in r.stdout


def test_a_struck_ITEM_is_not_reported_as_a_cleared_BLOCKER(tmp_path):
    """⛔ `cells[3:]`, not `cells[2:]`. A11/A13 strike their ITEM text; labelling
    that a cleared blocker attaches a wrong reason to a right row, and a reader
    who catches the tool being wrong once stops believing it."""
    write_backlog(tmp_path, CLEAN + """
| C9 | ~~Score both arms on the leak-free episodes~~ still pending | no blocker |
""")
    import json
    v = {x["id"]: x["verdict"]
         for x in json.loads(run(tmp_path, "--min-rows", "1", "--json").stdout)["verdicts"]}
    assert v["C9"] == "OPEN", "a struck item is not a cleared blocker"


# --------------------------------------------------------------------------
# 5. THE GUARDS -- every one exits 2, never 0
# --------------------------------------------------------------------------
def test_a_clean_list_exits_0(tmp_path):
    """⚠️ The guard must be SATISFIABLE. One that can never go green gets muted,
    and a muted gate is worse than none."""
    write_backlog(tmp_path, CLEAN)
    r = run(tmp_path, "--min-rows", "1")
    assert r.returncode == 0, r.stdout
    assert "OK" in r.stdout


def test_a_MISSING_backlog_REFUSES(tmp_path):
    r = run(tmp_path, name="NOTHING.md")
    assert r.returncode == 2 and "REFUSING" in r.stdout


def test_a_SHORT_read_REFUSES(tmp_path):
    """A truncated read on a flaky mount looks exactly like a short file.

    ⛔ ASSERTS THE BYTE-FLOOR MESSAGE, NOT JUST rc==2. Earned by a surviving
    mutation: with the byte floor removed this file still refused -- on the ROW
    floor, for a different reason -- and a test reading only the exit code
    called that a pass. `--min-rows 1` removes the rescuer so the byte guard is
    the only thing that can produce the refusal."""
    (tmp_path / "BACKLOG.md").write_text("| A1 | x | y |\n", encoding="utf-8")
    r = run(tmp_path, "--min-rows", "1")
    assert r.returncode == 2, r.stdout
    assert "bytes" in r.stdout and "short read" in r.stdout, r.stdout


def test_ZERO_ROWS_REFUSES_rather_than_reporting_clean(tmp_path):
    """⛔⛔ THE CENTRAL GUARD. A scan that reads zero rows is UNTRUSTWORTHY, not
    clean. Without this the tool's happiest output is produced by a file it could
    not parse at all."""
    write_backlog(tmp_path, "# BACKLOG\n\nprose only, no rows at all.\n")
    r = run(tmp_path)
    assert r.returncode == 2, r.stdout
    assert "0 rows" in r.stdout and "UNTRUSTWORTHY" in r.stdout


def test_BELOW_THE_ROW_FLOOR_REFUSES(tmp_path):
    """A parser regression that drops a whole row family looks exactly like a
    backlog somebody tidied."""
    write_backlog(tmp_path, CLEAN)
    r = run(tmp_path, "--min-rows", "500")
    assert r.returncode == 2 and "--min-rows" in r.stdout


def test_ZERO_CLOSED_ROWS_REFUSES(tmp_path):
    """⛔ The silent direction. Break the strike-detector so nothing reads as
    closed and DONE-BUT-OPEN can still report zero; the file provably contains
    struck ids, so finding none means the detector is broken."""
    write_backlog(tmp_path, """# BACKLOG

## A

| # | item | why |
|---|---|---|
| A1 | an open item | because |
| A2 | another open item | because |
""")
    r = run(tmp_path, "--min-rows", "1")
    assert r.returncode == 2 and "close-detector" in r.stdout


def test_ALL_ROWS_CLOSED_REFUSES(tmp_path):
    """The other side: if nothing reads as open, either the open-detector is
    broken or the list is finished -- and "finished" is a human's call."""
    write_backlog(tmp_path, """# BACKLOG

## A

| # | item | why |
|---|---|---|
| ~~A1~~ | ~~done~~ **DONE 2026-08-01** | x |
| ~~A2~~ | ~~done~~ **DONE 2026-08-02** | x |
""")
    r = run(tmp_path, "--min-rows", "1")
    assert r.returncode == 2 and "all " in r.stdout.lower()


# --------------------------------------------------------------------------
# 6. output discipline
# --------------------------------------------------------------------------
def test_stdout_is_ASCII_on_the_LIVE_glyph_dense_file():
    """⚠️ BEHAVIOUR, not a source scan. The sibling suite asserts that source
    lines beginning `print(` are ASCII, which says nothing about the ROW TEXT
    those lines interpolate -- and the live backlog is dense with house glyphs.
    The dev box console is cp1252; a guard that dies printing its own subject
    fails closed for the wrong reason (MEASURED 2026-08-21)."""
    r = subprocess.run([sys.executable, str(TOOL), "--repo", str(REPO),
                        "--list-open"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode in (0, 1), r.stderr
    assert r.stdout.isascii(), "non-ASCII reached stdout"
    assert len(r.stdout) > 400, "a near-empty stdout would pass this vacuously"


def test_heading_items_are_ANCHORED_not_substring_matched():
    """⛔ An unanchored match also caught the SECTION CONTAINER `## B. ONE GPU
    -- executable now (pod3; (glyph) eval stays free ...)`, whose glyph is a note
    inside a heading rather than a status on it. An item-heading LEADS with its
    status."""
    rows, sections, blocks, missed = m.parse(
        "## B. ONE GPU - executable now (pod3; ⛔ eval stays free)\n"
        "## ⛔⛔ BLOCKING INTEGRATION ITEM - seams are NOT IN HEAD\n"
        "## ✅ D-ROLL-1h - CLOSED 2026-09-06\n")
    assert len(blocks) == 2, [b["title"] for b in blocks]
    assert [b["closed"] for b in blocks] == [False, True]


def test_git_context_never_turns_a_readable_backlog_into_a_refusal(tmp_path):
    """⚠️ tmp_path is not a git repo. Context is context: unknown provenance must
    not become a refusal, and must not be mistaken for 'no drift' either."""
    write_backlog(tmp_path, CLEAN)
    r = run(tmp_path, "--min-rows", "1")
    assert r.returncode == 0
    assert "last written" in r.stdout


def test_a_bullet_the_parser_CANNOT_SEE_is_reported_not_swallowed(tmp_path):
    """⛔ EARNED, not imagined. The first BULLET_ROW regex accepted only an
    em-dash separator and silently dropped two hyphen-typed fixtures -- a silent
    miss inside the tool whose entire subject is silent misses. The list is
    hand-written markdown; a row typed slightly differently must be VISIBLE."""
    rows, sections, blocks, missed = m.parse(
        "- **R99 :: a separator this parser does not accept\n")
    assert rows == [] and missed == [1]
