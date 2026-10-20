# HANDOFF LATENCY-BUDGET (2026-10-20). DONE, CPU-only; nothing waits.

Memo `LATENCY_BUDGET_PROPOSAL.md` (decision block on top, 3 options, recommendation (2) two-rate), evidence in `raw/EVIDENCE_LEDGER.md`, arithmetic in `raw/budget_arithmetic.py` (+ `.out.txt`). Base tip `abd4dd81`.

**Next step (Master Mind):** land `out/` via `paths.json`. `library.json` and `LIBRARY.md` carry `base=<tip blob>`; if the tip's copies moved, do NOT replace wholesale: re-run `kb_add.py` for the ids in `raw/library_entries_added.json`, then `--reindex` and `--verify`. Put the PI_DECISION block to the PI. Then schedule M1 (trunk + pick tick on Thor), the number that decides the option; `SEARCH-THOR-LATENCY` does not cover it.

**Blocked / not mine:** the PI's choice of option; a Thor slot; a written pre-registration for M1 before it runs. **UNVERIFIED:** every REFe-on-Thor number (all ESTIMATED), P2's run geometry, ARM host factor, C1's +0.56 (from the brief).
Work dir `latbudget/export/` (4.5 GB raw export) may be deleted; it is reproducible with `raw/export_tip.py`.
