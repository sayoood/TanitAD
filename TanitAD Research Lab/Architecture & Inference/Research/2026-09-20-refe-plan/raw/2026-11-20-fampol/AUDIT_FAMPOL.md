# AUDIT_FAMPOL — the four-family polarity defect (TTC / headway / time-gap), 2026-11-20

Descriptive only. No banked artifact was edited. Tip `c98e6114` (mirror `C:/Users/Admin/tanitad-push`) plus `E:/Projects/TanitAD-artifacts/p5c_read/out/`.
Evidence class: MEASURED (scripts `audit_scan.py`, `audit_margin_cells.py`, `audit_ctx.py`, `dirtable.py`; outputs `audit_cells.txt`, `audit_ctx.txt`, `dirtable.txt` in this folder).
Scope of the scan: every JSON on the tip that contains the word `adverse` AND one of the three names (11 files), plus every `fam_ci*/famci_*` JSON (21 on the tip, 3 in `p5c_read/out`) read cell by cell. Verdict JSONs that name no margin in an adverse/guard list (P6 `result_m6*.json`, P7 `fam_ci_P6_vs_P7`) are unaffected: P6_vs_P7 has zero separated margin cells in either direction.

## The rule, before and after

Inherited `fam_direction`: `higher` only for `within_`/`accuracy`/`kappa`, else `lower` (bias -> magnitude, progress_ratio -> closer_to_1). So a margin that ROSE (safer) with lo > 0 was listed adverse, and a margin that FELL with hi < 0 was **never** listed (the second half of the defect, not in the brief: the rule is blind in both directions). Corrected: `min_ttc_s`, `headway_min_m`, `time_gap_min_s` (last dotted segment, any family prefix, also the battery's `mean_` form) are `higher`; adverse when the paired diff has hi < 0.

## A. Banked verdict / read JSONs with a margin in an adverse or guard list (11 files)

| file (under `raw/`, PK = refe-plan) | arm pair (diff = B-A) | margin, value [CI] | other adverse in the same list | verdict under corrected rule |
|---|---|---|---|---|
| `2026-11-20-p5c-read/read.json` (`g2.adverse`; `verdict.named.G2`) | C1 - C0, tracked | `min_ttc_s` +0.2970 [+0.1572, +0.4361] | 14 (speed_mae, speed_bias, speed_rmse, 3x within_, along_mae/bias/final_bias, accel_mae, progress_ratio_mean, progress_error_mean, cross_mae, cross_final_mae) | **GUARD-FAIL stands.** G2 stays FAIL on the 14; `min_ttc_s` leaves the named list (14 named, not 15). |
| `2026-11-20-p5c-read/families/families_summary.json` | C1-C0 tracked (same cell); C1-base tracked | `min_ttc_s` +0.2970 [+0.1572, +0.4361]; +0.2401 [+0.0972, +0.3849] | 14 each | Guard unchanged (FAIL); the margin entry drops out of both lists. |
| `2026-11-20-p5c-read/families/famci_C1_vs_C0.json` (cells, not a list) | C1 - C0 | transparent/reference frames: headway -0.1646 [-0.2487, -0.0848], time_gap -0.0495 [-0.0934, -0.0110], min_ttc -0.4998 [-0.7507, -0.2801] (reference: -0.163, -0.0469, -0.505) | n/a | **Newly adverse under the corrected rule, but only in the transparent and reference frames, which are never a guard (card 8.2)**. Tracked frame: only the RISE above. No verdict change; worth a sentence in the P5-C report: with the tracker the lead margin rises, without it it falls. |
| `2026-11-20-p5c-read/families/famci_C1_vs_base.json` | C1 - base | tracked: min_ttc +0.2401 (rise, not adverse); transparent/reference: min_ttc -0.5796 / -0.6011, headway -0.1234 / -0.1347, time_gap -0.0412 / -0.0404, all CI < 0 | n/a | no guard change (non-guard frames). |
| `2026-11-20-p5c-read/families/famci_C0_vs_base.json` | C0 - base | reference only: `min_ttc_s` -0.1127 [-0.2212, -0.0023] | n/a | none (diagnostic frame). |
| `2026-11-20-p5c-next/srule_fam/families/families_summary.json` | C1rule - C1pick | tracked: `min_ttc_s` +0.1141 [+0.0492, +0.1736]; transparent +0.1553 [+0.0971, +0.2060]; reference same | tracked: 1; transparent: 2; reference: 2 | Tracked guard still FAIL on the 1 non-margin component; the margin entry is a rise (safer) and drops out. |
| `2026-11-20-field-read/dry_runs/dry_C1_vs_C0.json`, `…/dry_selfread_C1_vs_C1.json`, `…/rawdac/dry_C1_vs_C0_rawdac.json`, `…/rawdac/dry_selfread_C1_vs_C1_rawdac.json` | stand-ins (C1 vs C0 as F vs C) | `min_ttc_s` +0.2970 [+0.1572, +0.4361] in `guard_longitudinal_lateral.adverse` and `verdict.clauses.family_adverse` | 14 | DRY RUN, stamped "NOT A FIELD VERDICT". Guard FAIL / "NOT PROVEN" unchanged; the margin leaves the list (v3 re-run: `dry_C1_vs_C0_fampol_v3.txt`). |
| `2026-11-13-p5c-read/dryrun/families_summary.json` | C0 - base (synthetic dry run) | tracked `time_gap_min_s` +0.0753 [+0.0030, +0.1665]; transparent/reference also `headway_min_m` +0.3562 [+0.0095, +0.7866], `time_gap_min_s` +0.1051 [+0.0229, +0.2061] | **0** other | Dry-run data. The ONLY list where the correction flips a guard: C0-base tracked FAIL -> PASS (the sole adverse entry was a RISE of a margin). Not a banked verdict. |
| `2026-09-27-m6-proxy/epoch2/result_m6_e2.json`, `…/result_m6.json`, `2026-10-17-safety-objective/navtestE/example/FIXTURE_P2REP_navtest_families.json` | | names appear, but in none of their adverse/guard lists | | no effect. |

## B. Cells the inherited rule MISSED (the margin FELL, separated, never listed adverse)

None of these sit in a banked verdict, but they are decision-relevant context (tracked frame = the guard frame):

| fam_ci file | pair | tracked margin cells that fell (corrected: adverse) |
|---|---|---|
| `2026-11-12-instruments/fam_ci/demo/fam_ci_RJ5R_vs_R0R.json` | RJ5_R - R0_R | headway -0.2625 [-0.4621, -0.1155], time_gap -0.0600 [-0.0850, -0.0193], min_ttc -0.8602 [-1.4156, -0.4390] |
| `2026-11-13-p7-read-route/rehearsal/fam_ci_U400_vs_P6RH.json` | P6RH - U400 (rehearsal) | headway -0.2858 [-0.4336, -0.1257], time_gap -0.0887 [-0.1271, -0.0518], min_ttc -0.4779 [-0.6625, -0.2752] |
| `2026-11-13-p5c-read/b10b12/dryrun/famci_C1_vs_C0.json`, `…/dryrun/famci_C1_vs_C0.json` | C1 - C0 (dry runs) | headway -0.69 / -0.58, time_gap -0.17 / -0.22 (CIs < 0) |
| `2026-11-11-teacher-fidelity/n3_dose/…/fam_ci_RJ3R_vs_RJ5R.json` | RJ3_R - RJ5_R | tracked margins ROSE (headway +0.0237, min_ttc +0.0528): inherited rule listed them adverse; corrected: not |
| `…/n4/…/fam_ci_N4D_vs_N2R.json`, `…/n6_tol/…/fam_ci_N6H_vs_N5A.json` | N4D-N2R, N6H-N5A | reference/transparent only, |diff| < 0.005 |

`fam_ci_RJ5R_vs_R0R` (a demo) is the clearest: a tracked-frame lead-margin collapse of -0.86 s TTC that the old guard could not see.

## C. Direction table of every metric the instruments emit (executed_tracked frame of `famci_C1_vs_C0.json`), corrected rule

Correct as given: headway/time_gap/min_ttc (now higher), lead_viol (lower), lead_at_risk (lower), speed_mae/rmse (lower), speed/along/cross/curvature biases (magnitude), within_* (higher), along_mae/accel_mae/cross_mae/heading/yaw_rate/curvature_mae/last_pose_heading_err (lower), progress_ratio_* (closer_to_1), progress_error_mean/under_progress_rate (lower), tactical accuracy/kappa (higher), goal_point_error_m / goal_bearing_mae_deg (lower), goal_*_bias (magnitude). The two `ego_progress.gt_*` rows are HUMAN-ONLY and skip-listed.

**Flagged, NOT changed (ambiguous or unreachable; listed for the PI/MM):**
1. `tactical.*.per_class.*.recall` / `.precision` -> rule gives `lower`; correct is higher-is-better. Unambiguous, but unreachable today: `fam_adverse`, G2 and `field_read` only walk longitudinal/lateral, and `scorerft_read` excludes per-class cells explicitly (`polarity_ok`). Fix when a caller starts judging them.
2. `tactical.goal_setting.goal_range_ratio` -> `lower`; should be `closer_to_1` (name does not contain `progress_ratio`). Unreachable today (tactical cells are never judged adverse).
3. `longitudinal.lead_progress_ratio` -> `closer_to_1` (via `progress_ratio`). Whether lead progress should be closer-to-1 or higher-is-better is a design choice (r7base uses a CLIPPED ratio as the decision reading); needs the PI/MM ruling.
