# HANDOFF v40 (built and landed by the Master Mind)

Base: the landed v39 report (`R/report/2026-11-05-v39/`, md `8ccffcf0…`, html `3f2fb6d1…`). Tip at build: `ffa2637a` (the A4 card landing).

## What v40 adds
* 7.40, inserted after 7.39.8: addendum A3, the level floor LF (card, rulings 1 and 2, build and known answers, the read at κ 0.50 and κ 0.75, the instruments, what it establishes, the ceiling card A4 as the next lever, provenance). 7 tables, each copied verbatim from `lf_read/RESULTS.md`.
* Edits: the status line (version, tip, version list, the LF sentence), the abstract recovery and next bullets (the LF sentence), the 7.39.7 pointer to 7.40, 8.1 item 30, 8.3 Update (v40), 8.4 row 1az, 9 item 18.

## Build and checks
`bash build_v40.sh <tip>`: patch (anchors asserted once each, tables as source lines, 55 italic spans verbatim), numbers audit (0 unmatched), 25 checks (25 PASS), offline render (233 tables, 56 h3, 323 math, 0 ragged).

## Files
`patch_v40.py`, `audit_v40.py`, `checks_v40.py`, `build_v40.sh`, `render_check_v40.js`, `edits_v40.md`, `sec_740_levelfloor.md`, `out/` (report md, html, DIRECTIVES.txt), `v40_NUMBERS_AUDIT.md`, `v40_checks.txt`, `v40_render_check.txt`, `MANIFEST.txt`, `msg_v40.txt`.
