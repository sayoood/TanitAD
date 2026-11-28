# ADDENDUM A8 (Master Mind, 2026-10-18 ~07:05 CEST): the confirm's family leg was not computed because of a staging defect; it is computed now. Written after the PDMS leg was seen and BEFORE any family number. No bar, gate or row changes.

**What happened.** The one pre-registered confirm ran (`confirm_run.log`, `confirm_out/`, 04:40Z): `UNRESOLVED (family leg not computed)`,
DeltaPDMS x100 +0.0235 [+0.0002, +0.0546], 11 / 12,146 picks changed, no adverse PDMS sub-score. CONFIRM.json `families.reason` reads
`fingerprint_available: false`. Cause: `confirm.py` takes the seam fingerprint and `sampling` from a `proposals.npz` beside the table
(`confirm.py:108-113`, `aligned_fingerprint` :162). The Master Mind's watcher (`selrule_confirm_watch.sh`) staged only `table_p2_full.npz`
and the marker. The instrument was right; the staging was incomplete.

**What is done now.** `proposals.npz` is built from banked bytes only: `token` and `fingerprint` from `table_p2_full.npz`
(sha256 33441082...4906e, the staged table), and `sampling` = (8.0, 0.5), identical in all 14 P2 dumps (`dump_main_c00..c13.npz`).
Then `confirm.py` is re-run unchanged, with the same FIT sha256 and the same table, into `confirm_out_a8/`. The first run's
`confirm_out/` is kept as it is.

**Control (must hold, or the re-run is void and the first run stands as UNRESOLVED):** the PDMS leg reproduces the first run exactly:
DeltaPDMS +0.0235 [+0.0002, +0.0546], 11 picks changed, the same zero / DAC=0 / NC=0 counts. The bootstrap seed is fixed and the
selection is deterministic, so any difference is a defect.

**Both outcomes committed now (section 4 + A6 + A7, unchanged):** the family leg shows no adverse component -> **ADOPT** as section 4
defines it. An adverse family component -> **NOT ADOPTED (adverse component: <names>)** per A7, escalated to the Master Mind.

**Disclosed with it, decided before the family leg:** the reach is +0.0235 PDMS x100 against an O-DAC oracle of +3.184. The rule captures
under 1 % of the zero-tail reach it targets. Whatever the row, SELRULE-DAC is not the zero-tail lever. That lever stays P6, the trunk DAC
gradient.
