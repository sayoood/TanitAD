# MM ruling on R_COLUMNS.md (DSCALE10K block 1, arm R_s0), 2026-11-17

**Ruling: the R release stays AS BUILT. All 14 `clip__` columns are kept as the receiver clip's own; nothing is rebuilt. The R release md5 `fbfc6e3f...` is the one R_s0 binds.**

## Why
* **What the trainer reads (MEASURED by the gate agent, source grep on tree `cdbff971`):** the trainer path reads 4 of the 14 clip columns: `sid`, `row0`, `n_rows` and `k0` (`stack/tanitad/data/v9_labels.py:104-105`). These four are layout and key, and they must stay under any rule.
* **No other reader:** nothing else under `stack/tanitad` or `taniteval` reads `clip__*`. The builder (`build_v9_labels.py:1245`) is the only other file that names them.
* **What a move would change:** moving `has_record`, `has_lead` and `has_lc` (the proposal's 9 / 10 / 11) cannot change one number R_s0 trains on.
  * It would change the R md5 and `dscale10k_merge.py derange` / `verify_derange`.
  * It would also change 844 clip-table rows, all from `has_lead`: `has_record` and `has_lc` read 0 on all 10,000 clips.
* **Changing code and data for an effect the experiment cannot see** adds a rebuild and a re-verify, and no information.
* **The design of R (card 2.4)** is that the supervision rows move and the clip's own inputs stay. That includes its own `nav_*`, its own clock, and the `lead_union.npz` sidecar that feeds `lead_gap_h`. The kept `n_turns` / `n_announced` agree with the own nav, as the proposal argues.

## What the R release's clip table therefore means (recorded, so no later reader misreads it)
* In `v9_labels_R_s0.npz`, the 14 `clip__` columns describe the RECEIVER clip, not the rows it carries.
* `has_lead` disagrees with the moved `lead_*` rows on 844 of 10,000 clips (8.44 %, MEASURED, `R_columns_probe.json`).
* Any future analysis that reads `clip__has_*` from the R release must use the union release's values for the DONOR (`v9_labels_R_s0.derangement.json` maps receiver to donor).

## Conditions on the R_s0 mint (unchanged by this ruling)
* The stage-A mint of R_s0 still requires every check to read PASS, including G-LIVE: every loss term finite on the real R data.
* The `lead_gap` (own) / `lead_valid` (donor) mismatch reaches the constraint penalty (`v8f_mini.py:631-632`). If any term is non-finite there, the mint refuses, and the refusal is a design question back to the MM, not a footnote.

Evidence class: MEASURED (the gate agent's source reads and `R_columns_probe.json`). Ruling by the MM. No PI decision is needed: it changes no arm, dose or cost.
