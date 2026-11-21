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


def test_the_live_backlog_reads_the_NAMED_ID_family_and_the_gate_runs():
    """⭐ REPLACES `..._is_CURRENTLY_DRIFTED_and_the_gate_says_so` (BACKLOG-HYGIENE-2,
    2026-10-22). That test pinned rc == 1 and the ids A11 ... R4 on the LIVE file,
    and said itself: *"If someone sweeps the list, this test goes red -- and that
    is the correct time to update it, with the sweep's commit."* The sweep landed,
    so the pin moves to where it cannot rot: the old drift shapes are now pinned on
    VERBATIM FIXTURE rows (section 7), and the live file is held to STRUCTURE.

    What the live file must still prove: (1) the gate RUNS (rc 0 or 1, never a
    refusal), (2) it READS the named-id table family -- ids like `EP-R1UB` that
    `TABLE_ROW` never matched, which is how ~80 rows went unread for five days --
    and (3) it reports no UNPARSED item-table row. A live drift COUNT is
    deliberately not pinned: a red suite every time a row goes stale would make
    `pytest` the backlog's gate, and the gate is this tool's exit code."""
    import json
    r = subprocess.run([sys.executable, str(TOOL), "--repo", str(REPO), "--json"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode in (0, 1), r.stdout + r.stderr
    d = json.loads(r.stdout)
    named = [v for v in d["verdicts"] if v["kind"] == "named"]
    assert len(named) >= 30, (
        "the named-id family (`| EP-R1UB | ...`) is unread: %d rows" % len(named))
    r2 = subprocess.run([sys.executable, str(TOOL), "--repo", str(REPO)],
                        capture_output=True, text=True, encoding="utf-8")
    assert "UNPARSED" not in r2.stdout, (
        "an item-table row no id regex reads: " + r2.stdout)


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


# --------------------------------------------------------------------------
# 7. BACKLOG-HYGIENE-2 (2026-10-22): the named-id family and CLOSED-LATER
# --------------------------------------------------------------------------
# ⛔ WHY THIS SECTION EXISTS. On 2026-10-22 the Master Mind measured 12 rows that
# LEAD `**NEXT ...**` and later record their own closure (`**DONE 2026-10-19 ...**`)
# while the tool read 91 OPEN rows. The diagnosis in the brief was "the tool reads
# only the leading verdict". ⭐ THE MEASURED CAUSE WAS ONE LAYER DOWN: none of those
# 12 rows was PARSED. `TABLE_ROW` took only `[A-Z]\d+` ids (`A11`, `R4`); the 80-odd
# rows added from 2026-10-17 on carry ids like `EP-R1UB`, so the tool did not read
# them as open and miss their closure -- it never saw them. The row family was
# invisible, and the one guard against invisible rows (`UNPARSED`) covered BULLETS
# only. Two fixes, two groups of tests: the FAMILY is read, and CLOSED-LATER fires.
import json  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "backlog_rows_2026-10-22.md"


@pytest.fixture(scope="module")
def live_rows():
    """{id: verdict-dict} over the VERBATIM live rows in the fixture."""
    rows, _s, _b, missed = m.parse(FIXTURE.read_text(encoding="utf-8"))
    assert missed == [], "the fixture's own rows must all parse: %s" % missed
    return {v["id"]: v for v in m.classify(rows)}


def _verdict(text: str) -> dict:
    """Parse a one-row inline backlog and return that row's verdict dict."""
    rows, _s, _b, _missed = m.parse(
        "## S\n\n| # | item | why |\n|---|---|---|\n" + text + "\n")
    out = m.classify(rows)
    assert len(out) == 1, out
    return out[0]


#: the 12 rows the Master Mind listed, with the shape each must fire by.
MM_LIST = [
    ("EP-R1UB", "dated"), ("HEADROW-PROBE", "dated"),
    ("PICKRULE-AUDIT-NC", "dated"), ("NC-FASTCAL", "dated"),
    ("EP-RELIAB-PICKS", "dated"), ("EP-ADMISSION-R1", "dated"),
    ("EP-ADMISSION-R2-CONFIRM", "dated"), ("PRIVGEOM-SAFE", "dated"),
    ("SEARCH-DOSE", "dated"), ("EPADM-SEARCHBANK-S0", "dated"),
    ("PRIVGEOM-DAC", "update"), ("L2-FRESH-STEPA", "qualified"),
]


@pytest.mark.parametrize("rid,shape", MM_LIST)
def test_every_row_the_MM_LISTED_fires_as_CLOSED_LATER(live_rows, rid, shape):
    """⭐ THE ACCEPTANCE LIST. Verbatim live rows; each leads open-type and closes
    itself later in a dated bold span. `shape` pins WHICH grammar branch fired, so
    a regression that fires for the wrong reason is red too."""
    v = live_rows[rid]
    assert v["verdict"] == "CLOSED-LATER", (rid, v["verdict"])
    assert v["closed_later_shape"] == shape, (rid, v["closed_later_shape"])
    assert v["kind"] == "named"


def test_the_NAMED_ID_family_is_PARSED_not_just_classified(tmp_path):
    """⛔ The root cause, end to end through the subprocess: an item table whose ids
    are `EP-R1UB`-shaped yields rows, a CLOSED-LATER verdict, and a failing gate."""
    write_backlog(tmp_path, CLEAN + """
## named

| # | item | unblocked by / GPU |
|---|---|---|
| EP-R1UB | **NEXT, CPU** fit the head **DONE 2026-10-19: NONE-FEASIBLE** (`D-X`). | 0 GPU |
""")
    r = run(tmp_path, "--min-rows", "1", "--json")
    d = json.loads(r.stdout)
    v = {x["id"]: x for x in d["verdicts"]}
    assert v["EP-R1UB"]["kind"] == "named", "the family is not being read"
    assert v["EP-R1UB"]["verdict"] == "CLOSED-LATER"
    assert r.returncode == 1, "CLOSED-LATER must FAIL the gate"
    txt = run(tmp_path, "--min-rows", "1").stdout
    assert "CLOSED-LATER" in txt and "EP-R1UB" in txt


@pytest.mark.parametrize("hdr", ["| step | time (Z) |", "| symbol | HEAD | worktree |"])
def test_a_NON_item_table_is_not_read_as_backlog_rows(hdr):
    """⛔ CONTROL for the table gate. BACKLOG.md carries an indented timing table
    whose first cells are `G-LIVE`, `G-EVAL` and `**G-SUITE**` -- named-id-shaped
    words that are NOT backlog items. Reading them would invent open rows. The gate
    is the table HEADER (`#` / `id`), not the id's shape."""
    rows, _s, _b, missed = m.parse(
        "## S\n\n  %s\n  |---|---|\n  | G-LIVE | 15:38:48 |\n"
        "  | **G-SUITE** | **16:03:26 (22 min)** |\n" % hdr)
    assert rows == [] and missed == [], (rows, missed)


def test_an_UNREADABLE_row_in_an_ITEM_table_is_REPORTED_not_swallowed():
    """⛔ The guard the table family LACKED. `UNPARSED` covered bullets only, so a
    table row whose id no regex reads was invisible -- which is precisely how the
    named family went unread. Same-breath control: the readable row beside it
    parses, so the report is about the bad row and not about the table."""
    rows, _s, _b, missed = m.parse(
        "## S\n\n| # | item | why |\n|---|---|---|\n"
        "| EP-R1UB | **NEXT** x | y |\n"
        "| ??? | an id nothing reads | y |\n")
    assert [r.rid for r in rows] == ["EP-R1UB"]
    assert missed == [6], missed


def test_the_gate_PRINTS_unparsed_item_table_rows(tmp_path):
    write_backlog(tmp_path, CLEAN + "| ??? | an id nothing reads | y |\n")
    out = run(tmp_path, "--min-rows", "1").stdout
    assert "UNPARSED" in out, out


# ---- controls: shapes that must NOT fire -------------------------------------
def test_CONTROL_a_NEXT_row_that_mentions_done_in_PROSE_does_not_fire():
    """The MM's first control. `done` in running text is not a closure."""
    v = _verdict("| X-1 | **NEXT, CPU** once the DONE-marker discipline is done, "
                 "re-run the probe; the loop is done when the gate is green. | y |")
    assert v["verdict"] == "OPEN", v
    assert v["closed_later_shape"] is None


def test_CONTROL_a_BOLD_compound_is_not_a_dated_closure():
    """`**DONE-marker discipline**` is live row F2's phrase. The tool's OLD
    first-word reading still calls such a span a closure (a latent false positive
    pinned by `test_the_bullet_bold_PREFIX_is_restored_or_spans_mis_pair`, so NOT
    touched here); the NEW branch must not add to it, because it needs a DATE and
    refuses the compound outright."""
    assert m.later_closure(
        "**NEXT, CPU** wire it **DONE-marker discipline 2026-10-19** applies") is None
    assert m.later_closure(
        "**NEXT, CPU** wire it **DONE-marker discipline** applies") is None


def test_CONTROL_UPDATE_with_no_closure_word_is_progress_not_closure():
    """The MM's second control: `**UPDATE <date>: SEALED**` records progress."""
    assert m.later_closure(
        "**NEXT** run it **UPDATE 2026-10-19: SEALED** (sha `5006ed84`)") is None
    v = _verdict("| X-2 | **NEXT** run it **UPDATE 2026-10-19: SEALED** | y |")
    assert v["verdict"] == "OPEN", v


def test_CONTROL_UPDATE_with_a_closure_word_DOES_fire():
    """The positive pair of the control above -- without it the control could be
    satisfied by a branch that never fires."""
    assert m.later_closure(
        "**NEXT** run it **UPDATE 2026-10-19: superseded by Y**") == "update"
    assert m.later_closure(
        "**NEXT** run it **UPDATE 2026-10-19 16:00: CLOSED, not needed**") == "update"


@pytest.mark.parametrize("rid", ["P7-WIRE-RESERVE", "P7-WIRE-HEAD"])
def test_CONTROL_the_P7_WIRE_rows_AS_THEY_STAND_do_not_fire(live_rows, rid):
    """The MM's third control, verbatim: both lead `**ACTIVE (0 GPU)**` and carry no
    dated closure. They are the live rows nearest the firing shape."""
    assert live_rows[rid]["verdict"] == "OPEN", live_rows[rid]


@pytest.mark.parametrize("rid", ["SEL-RANK", "P6-HN", "PAPER-720", "R0E-PICK",
                                 "SH5-DAC-PROX", "EP-ADMISSION", "P6-3C",
                                 "SEARCH-PESSIMISTIC", "FASTNEG-F25-RERUN"])
def test_CONTROL_live_multi_step_rows_with_PARTIAL_done_markers_stay_OPEN(
        live_rows, rid):
    """⛔ Verbatim live rows that carry `STEP 1 DONE`, `S1 DONE`, `Step 0 DONE`,
    `B3, B6, B7 DONE`, `C3, C4, C11 DONE`, `grids 3-4 DONE`: PARTIAL-PROGRESS
    markers on rows with work left. Counting any of them as a closure would strike
    nine live rows that are still open."""
    assert live_rows[rid]["verdict"] == "OPEN", live_rows[rid]


def test_a_QUALIFIED_done_closes_only_as_the_rows_LAST_emphasised_span():
    """THE L2-FRESH-STEPA JUDGEMENT CALL (brief: *"the read closed, but did the
    row's work?"*). DECIDED: a qualified DONE (`READ 1 DONE <date>`) is a partial
    marker everywhere in this file, so it is NOT a closure -- EXCEPT as the LAST
    bold span of a row that LEADS open-type. Rows are append-only, so a qualified
    DONE that nothing emphasised follows, under a lead still saying NEXT, is the
    row's final word. A later open marker keeps the row open."""
    last = "**NEXT** run both reads **READ 1 DONE 2026-10-19 20:12: O3** (A1 0.840)"
    assert m.later_closure(last) == "qualified"
    then_more = last + " then **READ 2 NEXT** on the second carve"
    assert m.later_closure(then_more) is None
    # a NEXT lead with an EARLIER qualified DONE and a later plain step: open
    mid = "**NEXT** two steps **STEP 1 DONE 2026-10-18** and **Step 2: build it**"
    assert m.later_closure(mid) is None


def test_the_QUALIFIED_decision_is_pinned_on_the_VERBATIM_row(live_rows):
    v = live_rows["L2-FRESH-STEPA"]
    assert v["verdict"] == "CLOSED-LATER" and v["closed_later_shape"] == "qualified"


def test_an_UNDATED_closure_word_is_not_CLOSED_LATER():
    """`CLOSED` / `DONE` with no date: the old DONE-BUT-OPEN path may still read it
    (live row SEARCH-STALE-T1), but the NEW verdict requires the date -- it is what
    separates a verdict from prose."""
    assert m.later_closure("**PROPOSED** x **CLOSED, NOT NEEDED** (D-Y)") is None
    v = _verdict("| X-3 | **PROPOSED** x **CLOSED, NOT NEEDED** (D-Y) | y |")
    assert v["verdict"] == "DONE-BUT-OPEN", v


def test_the_verdict_splits_on_the_LEAD_not_on_the_closure():
    """CLOSED-LATER = the lead LIES (open-type, closure later). DONE-BUT-OPEN = the
    row is honest but its id is not struck. A row that OPENS with the verdict, or
    with a struck/plain title, is the second kind -- as it always was."""
    lead_lies = _verdict("| X-4 | **NEXT** do it **DONE 2026-10-19** | y |")
    honest = _verdict("| X-5 | **DONE 2026-10-19** did it. **NEXT: X-4** | y |")
    plain = _verdict("| X-6 | do it **DONE 2026-10-19** | y |")
    struck_title = _verdict("| X-7 | ~~do it~~ **DONE 2026-10-19** | y |")
    assert lead_lies["verdict"] == "CLOSED-LATER"
    assert honest["verdict"] == "DONE-BUT-OPEN"
    assert plain["verdict"] == "DONE-BUT-OPEN"
    assert struck_title["verdict"] == "DONE-BUT-OPEN"


def test_a_STRUCK_id_with_a_later_closure_is_CLOSED_and_quiet():
    """The sweep's own output shape must satisfy the tool: strike the id, keep the
    evidence, and the row reads CLOSED."""
    v = _verdict("| ~~X-8~~ | **NEXT** do it **DONE 2026-10-19** | y |")
    assert v["verdict"] == "CLOSED"


def test_RESOLVED_closes_a_BLOCKER_row_and_nothing_else():
    """`D-SPACE` and `L2-STEPB-GATE` lead `**BLOCKER ...**` and later say
    `**RESOLVED <date>**`. Under any other lead `resolved` is a sub-blocker's end or
    a decision id (`D-STEER-INTERFACE-RESOLVED`), never the row's closure."""
    assert m.later_closure(
        "**BLOCKER (needs the PI)** D: is full **RESOLVED 2026-10-19 19:15:** deleted"
    ) == "resolved"
    assert m.later_closure(
        "**NEXT** wire it **RESOLVED 2026-10-19:** one sub-blocker; wiring continues"
    ) is None
    assert m.later_closure(
        "**BLOCKER** wait on the pod **RESOLVED, see D-STEER-INTERFACE-RESOLVED**") is None


@pytest.mark.parametrize("rid", ["D-SPACE", "L2-STEPB-GATE"])
def test_the_two_live_BLOCKER_rows_fire_by_the_resolved_shape(live_rows, rid):
    assert live_rows[rid]["closed_later_shape"] == "resolved", live_rows[rid]


@pytest.mark.parametrize("word", ["DONE", "CLOSED", "SUPERSEDED", "STRUCK",
                                  "DISCHARGED", "WITHDRAWN", "CANCELLED"])
def test_every_closure_word_the_MM_named_fires_when_dated(word):
    assert m.later_closure("**NEXT** x **%s 2026-10-20: why**" % word) == "dated"


def test_the_lead_span_itself_is_never_its_own_later_closure():
    """`spans[0]` IS the lead: `**DONE 2026-10-19**` leading a row is the row's
    verdict (DONE-BUT-OPEN territory), not a 'later' closure."""
    assert m.later_closure("**DONE 2026-10-19** did it **NEXT: other**") is None
    assert m.lead_status("**DONE 2026-10-19** did it") == "done"
    assert m.lead_status("~~old~~ **NEXT** x") == ""        # no lead span
    assert m.lead_status("plain **NEXT** x") == ""


# ---- the old drift shapes, pinned on verbatim rows ---------------------------
@pytest.mark.parametrize("rid", ["A11", "A13", "B4", "C3", "R1", "R2", "R4"])
def test_the_OLD_DONE_BUT_OPEN_shapes_still_fire_on_verbatim_rows(live_rows, rid):
    """MOVED here from the live-file pin (section 1): the seven rows the tool was
    written to catch, as they stood before the 2026-10-22 sweep struck them."""
    assert live_rows[rid]["verdict"] == "DONE-BUT-OPEN", live_rows[rid]


def test_the_OLD_STALE_BLOCKER_shape_still_fires_on_verbatim_C2(live_rows):
    assert live_rows["C2"]["verdict"] == "STALE-BLOCKER", live_rows["C2"]


def test_the_gate_prints_the_CLOSED_LATER_advice_and_fails_on_the_FIXTURE(tmp_path):
    """End to end on the verbatim fixture, through the subprocess boundary."""
    # ⚠️ the fixture holds NO struck id, and the tool REFUSES a list with zero closed
    # rows (the silent direction) -- so one properly-struck row is appended.
    tail = "\n## closed\n\n| # | item |\n|---|---|\n| ~~Z-1~~ | ~~x~~ **DONE 2026-10-01** |\n"
    (tmp_path / "BACKLOG.md").write_bytes(
        FIXTURE.read_bytes() + tail.encode("utf-8"))
    r = subprocess.run([sys.executable, str(TOOL), "--repo", str(tmp_path),
                        "--backlog", "BACKLOG.md", "--min-rows", "1", "--max", "200"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 1, r.stdout
    assert r.stdout.isascii()
    assert "CLOSED-LATER" in r.stdout and "EP-R1UB" in r.stdout
    assert "reads the lead" in r.stdout


@pytest.mark.parametrize("rid", ["UNSEEN-STRATUM-DATA", "S4R-1", "SEARCH-STALE-T1"])
def test_the_LABEL_stays_DONE_BUT_OPEN_when_the_lead_is_not_open_type(live_rows, rid):
    """Verbatim rows whose lead is a verdict (`S4R-1`), a title (`UNSEEN-STRATUM-DATA`
    leads `O3 next arm`, then closes with a dated CLOSED) or an UNDATED closure
    (`SEARCH-STALE-T1`). They fire, but as the OLD class -- the lead gate is what
    keeps the two labels apart."""
    assert live_rows[rid]["verdict"] == "DONE-BUT-OPEN", live_rows[rid]
    assert live_rows[rid]["closed_later_shape"] is None


def test_a_dated_closure_under_a_NON_open_lead_is_not_CLOSED_LATER():
    v = _verdict("| X-9 | **O3 next arm** of Y **S1 DONE 2026-10-19** and then "
                 "**CLOSED 2026-10-19: the route does not exist** | y |")
    assert v["verdict"] == "DONE-BUT-OPEN", v
