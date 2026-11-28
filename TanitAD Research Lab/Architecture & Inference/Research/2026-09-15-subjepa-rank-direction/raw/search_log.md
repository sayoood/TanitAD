# Search / probe log — 2026-09-15 Sub-JEPA rank direction

| # | probe | result |
|---|---|---|
| L1 | V-1: `library.json` for `2605.09241` | banked 2026-05 era, PDF present (debt D-9: banked-but-unread) |
| R1 | full text by local pypdf extraction (40,154 chars), `arXiv:2605.09241v1 [cs.LG] 10 May 2026` | read in full: Tables 1–4, Eq. 9, §4.2.2, §4.3.3 |
| G1 | `grep "sub32c\|sub64c" Project Steering/*.md` | 5 rows in `GOALS_AND_CLAIMS.md` (H-RANK-5, -9, -14, -15 and one more), **all rank/participation reads**; ⛔ **no quality read found** for these arms. Second probe needed before "none exists" is claimed: the arm JSONs (`~/v7tiny/val_rank_3way.json`, `stage_gate.json`) sit on a host not probed this pass → folded into AI15-1 |
| G2 | `grep H-RANK-16 / participation` | floor definition, H-RANK-12 / -15, D-V7F-NONCOLLAPSE located |
| W1 | `arXiv 2609 JEPA world model planning September 2026` (shared with the frontier scan A2) | `2609.03565` (IDM + SA), `2609.04264` (spectral-target head, scan only) |
| W2 | HTML full text `arxiv.org/html/2609.03565` | read (fetch summariser, numbers quoted from its extraction). ⚠️ No std reported for the 3 seeds |
