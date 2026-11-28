"""⛔ RETIRED 2026-09-05 — THIS TOOL COULD SHRED `GOALS_AND_CLAIMS.md`.

It chose its line ending with a CRLF membership test (`"\\r\\n" in text`). The register is
MIXED — MEASURED 2,777 LF against 24 CRLF — so the test answered "CRLF" and the tool then
split an 854 KB, 2,777-line file into **25** "lines". An insert computed on that splitting
rewrites the register with 2,752 line boundaries destroyed.

It never fired. The register was verified intact the same morning (854,085 B, 2,777 LF,
24 CRLF, 0 lone CR; every `D-*` row asserted present in HEAD by `cat-file -e`). This file
is left in place, refusing, rather than deleted — so that anyone who finds a reference to
it in a report reads why instead of resurrecting it.

⭐ THE SAFE PATTERN NEEDS NO TOOL, and every other stream already uses it:

    data = path.read_bytes() + row.encode("utf-8")     # bytes, never str.split
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    assert tmp.read_bytes() == data                    # verify BEFORE replacing
    os.replace(tmp, path)
    assert path.read_bytes() == data                   # and after

It appends bytes and never splits lines, so a mixed-ending file cannot be corrupted by it.
Re-read the register immediately before the edit (siblings append concurrently), INSERT
rather than rewrite, and re-assert the sibling rows afterwards.
"""

raise SystemExit(
    "insert_rows.py is RETIRED: it splits a mixed-ending register into 25 lines and would "
    "destroy 2,752 line boundaries. Append BYTES with read-back verification instead — see "
    "this file's docstring for the four-line pattern."
)
