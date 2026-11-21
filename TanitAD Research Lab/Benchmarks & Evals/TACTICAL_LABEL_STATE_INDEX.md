# TACTICAL LABEL STATE — where the results live, and what they currently say

**Why this file exists.** The tactical-label results are spread across six packages, two code
modules and the registry, and the newest finding (2026-08-17) **invalidates a number in an older
one**. Anyone reviewing "the current state" from a single package will read something stale. This is
the index, newest-first, with the status of each.

Index compiled 2026-08-18. ⚠️ It points at sources; it does not restate their numbers as new
results.

---

## 1. START HERE — the two that change how you read everything older

| # | source | status |
|---|---|---|
| ⭐ 1 | `…/incoming/2026-08-17-maneuver-label-mismatch/MANEUVER_LABEL_MISMATCH.md` | **`maneuver_acc` was scored against v1 labels for every `--v2` arm.** Blast radius is **one** banked number set (`flagship-v2corpus-30k`, `seam_ctx_to_tactical.maneuver_acc` **0.0239**, n=418/19 eps); the deployed v1 and `v1arch-v2bal-30k` are **v1-labelled and clean**. ⛔ The corrected number **cannot be recovered from anything banked** — `man_pred` is on disk nowhere (four probes); only a GPU re-run of `hierarchy.run` with `labels_v2=True` can produce it. The instrument gap is closed going forward (the panel now banks per-window arrays and derives BOTH label families at scoring time). **Registry rows the PI must correct are listed in its §6 — not yet applied.** |
| ⭐ 2 | `…/incoming/2026-08-03-dtac1-tactical-head/DTAC1_RESULTS.md` (+ `DTAC1B_RESULTS.md`, `adversarial-verification/ADVERSARIAL_RECORD.md`) | REF-C's manoeuvre head, diagnosed and fixed. Shipped 5-way decode: **accuracy 0.7581, macro-recall 0.5313, `never_predicted: ["accelerate"]`**. Factored (lat × lon): **macro-recall 0.8290, accuracy 0.9348**. ⛔ **The label itself destroys 9.68 % (132/1364) of longitudinal decisions** — windows carrying a live longitudinal manoeuvre that the 5-way label calls something else. Controls fired (`control_shuffled` auc 0.4933, at chance). |

## 2. THE MECHANISM BEHIND BOTH

The 5-way manoeuvre softmax **mixes the lateral and longitudinal axes**. One mechanism explains: the
0/881 `accelerate` predictions, the speed-fan, the absence of longitudinal signal in selection, and
why no arm beats hold-`v0` at cruising speed. The factorised head (lat × lon) is the repair —
`STREAM A` (task #50), landed in `refc.py`, and carried into REF-A v1's tactical brain by
construction.

## 3. THE LABEL DEFINITIONS THEMSELVES (live code, not a report)

| source | what it gives you |
|---|---|
| `stack/scripts/v4_labels.py` → `mintability_report()` | ⭐ **the authoritative machine-readable statement of what IS and IS NOT mintable**: the v3 route tokens, the strategic scalars, and `not_mintable_needs_data` with a reason per class. Run it — do not read a summary of it. |
| `stack/tanitad/refs/refb_labels.py` | `MANEUVER_CLASSES`, `ROUTE_CLASSES`, `classify_maneuver` — the classifier the panels score against |
| `stack/tanitad/models/flagship_v4.py:60-70` | the factorised vocabulary widths (`N_LAT 8`, `N_LON 7`, `N_DIST 8`) and why the lead-referenced LON modes are absent from the built corpus |
| `taniteval/taniteval/hierarchy.py:592` | where `man_tgt` is computed — the exact line the 2026-08-17 mismatch is about |

⚠️ **Known-stale-then-corrected wording:** the "lead_state is a None stub" reason string was a rotted
absence-claim and was retired at all four live sites on 2026-08-18 (`4276613a`). The lead state now
exists program-wide; the LON lead modes are unmintable because **the minter consumes no lead input**,
not because the state is missing. A rebuild could mint them.

## 4. SCORING SURFACES (where a tactical number comes from)

| surface | file | note |
|---|---|---|
| open-loop tactical panel | `taniteval/taniteval/planning.py` | `maneuver_acc`, `turn_recall`, `tactical_wp_ade`, `goal_latent_cos` |
| closed-loop TACTICAL family | `taniteval/taniteval/four_families.py` | the binding family; never pooled with the other three |
| hierarchy seams | `taniteval/taniteval/hierarchy.py` | `seam_ctx_to_tactical` — the row the mismatch affects |
| gate-dependence of κ | `…/incoming/2026-08-18-dir-yaw-reread/` | `kappa_turn_subset` is **gate-dependent** and `_gate_sensitivity` does not sweep it; envelope at gate 0.10 computed over five banked panels (200-case self-test passing). **Partial — the pass was cut short by a session limit; PREREG and envelope are banked.** |

## 5. OLDER CONTEXT (still valid, superseded in places)

`…/2026-07-29-situation-labels/` · `…/2026-07-28-tactical-action-input/` ·
`…/2026-07-27-percandidate-labels/` · `…/2026-07-26-h2-label-v2/` (+ its `PRE_REGISTRATION_L2.md`) ·
`…/2026-07-26-4brain-dominance-program/STRATEGIC_TACTICAL_PROBLEM_SPEC.md` ·
registry **§1.13b** (tactical stage-0, first trained instance, T0) and **§11.1a / §11.2** (what the
A2 labels contain; the PH1-fused hierarchical label layer with its named 57.2 % perception hole).

## 6. ⛔ OPEN ITEMS A REVIEWER SHOULD KNOW ARE OPEN

1. **The registry rows named in `MANEUVER_LABEL_MISMATCH.md` §6 have not been applied.** Until they
   are, the registry states a `maneuver_acc` for a `--v2` arm that was scored on v1 labels.
2. **The corrected `maneuver_acc` needs a GPU re-run** — no estimator can substitute.
3. **The paper's κ-collapse (0.253 → 0.0072) is a SEPARATE issue** and is not cleared by the label
   fix. Do not merge the two into one retraction.
4. **The DIR_YAW 0.15 → 0.10 re-read is unfinished** — pre-registration and envelope banked, the
   driver has a path-depth bug, no verdict yet.
